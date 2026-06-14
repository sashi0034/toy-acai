from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal


class ActorCritic(nn.Module):
    """方策(actor)と価値関数(critic)を 1 つのネットワークで共有するモデル。"""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        fire_bias_init: float = 0.4,
        log_std_init: float = -0.8,
    ):
        super().__init__()
        # backbone は観測ベクトルを特徴量に変換する共通部分。
        # その上に「行動を出す頭」と「状態価値を出す頭」を載せる。
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, 2)
        self.fire_logits = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        # 連続行動(加速・旋回)は正規分布からサンプルする。
        # 平均は観測ごとに出し、標準偏差は学習可能なパラメータとして持つ。
        self.log_std = nn.Parameter(torch.full((2,), log_std_init))
        # fire は 0/1 の離散行動なので Bernoulli 分布の logit として扱う。
        # 初期値を少し正にして、学習初期からまったく撃たない状態を避ける。
        nn.init.constant_(self.fire_logits.bias, fire_bias_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
        # 標準偏差が大きすぎると行動が荒れ、小さすぎると探索しなくなる。
        # clamp で探索量の範囲を制限している。
        log_std = torch.clamp(self.log_std, -2.5, 0.0).expand_as(mean)
        return mean, log_std, self.fire_logits(hidden), self.value(hidden).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        fire_threshold: float = 0.5,
    ):
        mean, log_std, fire_logits, value = self.forward(obs)
        if deterministic:
            # 評価時は分布からランダムに引かず、平均行動を使って実力を測る。
            raw_cont = mean
            fire = (torch.sigmoid(fire_logits) >= fire_threshold).float()
        else:
            # 学習時は確率的に行動する。いろいろ試すことで、後から良い行動を強められる。
            raw_cont = Normal(mean, log_std.exp()).sample()
            fire = Bernoulli(logits=fire_logits).sample()
        action = torch.cat([raw_cont, fire], dim=-1)
        # PPO の確率計算には tanh 前の raw_cont を使い、
        # 環境には [-1, 1] に収めた env_action を渡す。
        env_action = torch.cat([torch.tanh(raw_cont), fire], dim=-1)
        log_prob, entropy = self.evaluate_actions(obs, action)
        return action, env_action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        # 保存しておいた行動が、現在の方策ではどれくらいの確率で出るかを再計算する。
        # PPO の更新では「古い方策の確率」と「現在の方策の確率」の比を使う。
        mean, log_std, fire_logits, _ = self.forward(obs)
        cont_dist = Normal(mean, log_std.exp())
        fire_dist = Bernoulli(logits=fire_logits)
        raw_cont = actions[:, :2]
        fire = actions[:, 2:3]
        log_prob = cont_dist.log_prob(raw_cont).sum(dim=-1) + fire_dist.log_prob(fire).sum(dim=-1)
        entropy = cont_dist.entropy().sum(dim=-1) + fire_dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[3]


@dataclass
class PPOConfig:
    # gamma: 将来報酬をどれだけ重視するか。1 に近いほど長期的な結果を重視する。
    gamma: float = 0.995
    # gae_lambda: advantage 推定の滑らかさ。大きいほど長い未来を見るが分散も増える。
    gae_lambda: float = 0.95
    # clip: 方策を一度に変えすぎないための PPO 特有の制限幅。
    clip: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256
    rollout_steps: int = 2048
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    fire_bias_init: float = 0.4
    eval_fire_threshold: float = 0.15
    hidden_dim: int = 256
    log_std_init: float = -0.8


class AgentRolloutBuffer:
    """1 機ぶんの経験を時系列でためて、PPO 更新用テンソルに変換する。"""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def tensors(
        self,
        device: torch.device,
        last_value: float,
        config: PPOConfig,
        normalize_advantages: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Python list のままだとミニバッチ学習しづらいので、NumPy/Torch の配列にまとめる。
        obs = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        # GAE(Generalized Advantage Estimation)を後ろから計算する。
        # advantage は「価値関数の予想より、実際の行動がどれだけ良かったか」の目安。
        for step in reversed(range(rewards.shape[0])):
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + config.gamma * next_value * next_nonterminal - values[step]
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
            next_value = float(values[step])

        returns = advantages + values
        flat = {
            "obs": torch.as_tensor(obs, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "old_log_probs": torch.as_tensor(old_log_probs, device=device),
            "advantages": torch.as_tensor(advantages, device=device),
            "returns": torch.as_tensor(returns, device=device),
        }
        if normalize_advantages:
            flat["advantages"] = normalize_advantages_tensor(flat["advantages"])
        return flat


def normalize_advantages_tensor(advantages: torch.Tensor) -> torch.Tensor:
    # advantage のスケールをそろえると、方策更新の大きさが安定しやすい。
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


class RolloutBuffer:
    """複数エージェントの経験を、エージェントごとのバッファに分けて保持する。"""

    def __init__(self, agent_count: int = 4):
        self.agent_count = agent_count
        self.agent_buffers = [AgentRolloutBuffer() for _ in range(agent_count)]

    def add(self, obs, action, log_prob, reward, done, value):
        # 環境は味方全機ぶんの結果をまとめて返すので、ここで 1 機ずつに分解する。
        observations = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(action, dtype=np.float32)
        log_probs = np.asarray(log_prob, dtype=np.float32)
        rewards = np.asarray(reward, dtype=np.float32)
        values = np.asarray(value, dtype=np.float32)
        if observations.shape[0] != self.agent_count:
            raise ValueError(
                f"expected {self.agent_count} agent observations, got {observations.shape[0]}"
            )
        for agent_id in range(self.agent_count):
            self.agent_buffers[agent_id].add(
                observations[agent_id],
                actions[agent_id],
                log_probs[agent_id],
                rewards[agent_id],
                done,
                values[agent_id],
            )

    def __len__(self):
        if not self.agent_buffers:
            return 0
        return min(len(buffer) for buffer in self.agent_buffers)

    def clear(self):
        for buffer in self.agent_buffers:
            buffer.clear()


class _SingleAgentPPOTrainer:
    """1 つの ActorCritic モデルを学習させるための実装。"""

    def __init__(self, obs_dim: int, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = ActorCritic(
            obs_dim,
            hidden_dim=config.hidden_dim,
            fire_bias_init=config.fire_bias_init,
            log_std_init=config.log_std_init,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def act(self, observation: np.ndarray, deterministic: bool = False):
        # モデルは batch 次元つきのテンソルを受け取るため、1 機ぶんの観測にも [None, :] を付ける。
        obs_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            actions, env_actions, log_probs, _, values = self.model.act(
                obs_tensor,
                deterministic=deterministic,
                fire_threshold=self.config.eval_fire_threshold,
            )
        return (
            actions[0].cpu().numpy(),
            env_actions[0].cpu().numpy(),
            float(log_probs[0].cpu()),
            float(values[0].cpu()),
        )

    def value(self, observation: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return float(self.model.values(obs_tensor)[0].cpu())

    def update(self, buffer: AgentRolloutBuffer, last_value: float) -> Dict[str, float]:
        data = buffer.tensors(
            self.device,
            last_value,
            self.config,
            normalize_advantages=False,
        )
        data["advantages"] = normalize_advantages_tensor(data["advantages"])
        return self._update_tensors(data)

    def update_many(
        self,
        buffers: List[AgentRolloutBuffer],
        last_values: np.ndarray,
    ) -> Dict[str, float]:
        # 共有方策の場合、全機の経験をまとめて「同じ 1 つの方策」を更新する。
        # これによりデータ量が増え、各機が同じ操作ルールを学ぶ。
        last_values = np.asarray(last_values, dtype=np.float32)
        if len(buffers) != len(last_values):
            raise ValueError(
                f"expected {len(buffers)} last values, got {len(last_values)}"
            )
        per_agent_data = [
            buffer.tensors(
                self.device,
                float(last_value),
                self.config,
                normalize_advantages=False,
            )
            for buffer, last_value in zip(buffers, last_values)
            if len(buffer) > 0
        ]
        if not per_agent_data:
            return {}
        data = {
            key: torch.cat([agent_data[key] for agent_data in per_agent_data], dim=0)
            for key in per_agent_data[0]
        }
        data["advantages"] = normalize_advantages_tensor(data["advantages"])
        return self._update_tensors(data)

    def _update_tensors(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        count = data["obs"].shape[0]
        indices = np.arange(count)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, count, self.config.batch_size):
                # ためた経験を小さなミニバッチに分け、同じ rollout を数 epoch 再利用する。
                batch_idx = torch.as_tensor(indices[start : start + self.config.batch_size], device=self.device)
                obs = data["obs"][batch_idx]
                actions = data["actions"][batch_idx]
                old_log_probs = data["old_log_probs"][batch_idx]
                advantages = data["advantages"][batch_idx]
                returns = data["returns"][batch_idx]

                log_probs, entropy = self.model.evaluate_actions(obs, actions)
                values = self.model.values(obs)
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                # ratio > 1 なら「その行動を以前より出しやすくした」、
                # ratio < 1 なら「以前より出しにくくした」という意味になる。
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * advantages
                # clip された目的関数を使うことで、良さそうな行動でも一気に確率を上げすぎない。
                policy_loss = -torch.min(unclipped, clipped).mean()
                # critic は、実際に得られた return に value を近づけるように学習する。
                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_mean = entropy.mean()
                # entropy は探索の多さ。loss から引くことで、早すぎる決め打ちを少し抑える。
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip).float().mean()
                )
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                # 勾配が大きすぎると学習が壊れやすいので、最大ノルムで丸める。
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"] += float(value_loss.detach().cpu())
                stats["entropy"] += float(entropy_mean.detach().cpu())
                stats["approx_kl"] += float(approx_kl.detach().cpu())
                stats["clip_fraction"] += float(clip_fraction.detach().cpu())
                updates += 1

        if updates:
            for key in stats:
                stats[key] /= updates
        stats["action_std"] = float(torch.exp(torch.clamp(self.model.log_std, -2.5, 0.0)).mean().detach().cpu())
        return stats

class PPOTrainer:
    """複数の味方機をまとめて扱うラッパー。共有方策と個別方策の両方に対応する。"""

    def __init__(
        self,
        obs_dim: int,
        config: PPOConfig,
        device: torch.device,
        agent_count: int = 4,
        shared_policy: bool = True,
    ):
        self.config = config
        self.device = device
        self.agent_count = agent_count
        self.shared_policy = shared_policy
        if shared_policy:
            # 共有方策: 全機が同じモデルを使う。サンプル効率が良く、まずはこちらが分かりやすい。
            self.shared_agent = _SingleAgentPPOTrainer(obs_dim, config, device)
            self.agents = [self.shared_agent for _ in range(agent_count)]
        else:
            # 個別方策: 各機が別々のモデルを持つ。役割分担を学べる可能性があるが難しくなる。
            self.shared_agent = None
            self.agents = [
                _SingleAgentPPOTrainer(obs_dim, config, device)
                for _ in range(agent_count)
            ]

    def act(self, observations: np.ndarray, deterministic: bool = False):
        # observations は [agent_count, obs_dim]。各行を各機の trainer に渡す。
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape[0] != self.agent_count:
            raise ValueError(
                f"expected {self.agent_count} agent observations, got {observations.shape[0]}"
            )
        results = [
            agent.act(observations[agent_id], deterministic=deterministic)
            for agent_id, agent in enumerate(self.agents)
        ]
        actions, env_actions, log_probs, values = zip(*results)
        return (
            np.asarray(actions, dtype=np.float32),
            np.asarray(env_actions, dtype=np.float32),
            np.asarray(log_probs, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
        )

    def values(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape[0] != self.agent_count:
            raise ValueError(
                f"expected {self.agent_count} agent observations, got {observations.shape[0]}"
            )
        return np.asarray(
            [
                agent.value(observations[agent_id])
                for agent_id, agent in enumerate(self.agents)
            ],
            dtype=np.float32,
        )

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray) -> Dict[str, float]:
        # last_values は rollout の最後の観測に対する価値推定。
        # エピソード途中で切れた経験の return 計算に使う。
        if buffer.agent_count != self.agent_count:
            raise ValueError(
                f"buffer agent_count={buffer.agent_count} does not match trainer agent_count={self.agent_count}"
            )
        last_values = np.asarray(last_values, dtype=np.float32)
        if self.shared_policy:
            stats = self.shared_agent.update_many(buffer.agent_buffers, last_values)
            stats["shared_policy"] = 1.0
            return stats
        per_agent_stats = [
            agent.update(buffer.agent_buffers[agent_id], float(last_values[agent_id]))
            for agent_id, agent in enumerate(self.agents)
        ]
        stats = self._mean_stats(per_agent_stats)
        for agent_id, agent_stats in enumerate(per_agent_stats):
            for key, value in agent_stats.items():
                stats[f"agent_{agent_id}_{key}"] = float(value)
        return stats

    def save(self, path: Path, extra: Dict[str, object]) -> None:
        # 再開に必要な重み・設定・補足情報をまとめて保存する。
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": [agent.model.state_dict() for agent in self.agents],
                "agent_count": self.agent_count,
                "shared_policy": self.shared_policy,
                "config": self.config.__dict__,
                **extra,
            },
            path,
        )

    def load(self, path: Path) -> Dict[str, object]:
        # checkpoint の agent 数や形式を確認してから読み込む。
        # 共有方策へ読み込む場合は、保存されている各機の重みを平均して 1 つにする。
        checkpoint = torch.load(path, map_location=self.device)
        if "models" not in checkpoint:
            raise ValueError(
                "checkpoint uses the old single-model format; start a new run or provide a multi-agent checkpoint"
            )
        checkpoint_models = checkpoint["models"]
        checkpoint_agent_count = int(checkpoint.get("agent_count", len(checkpoint_models)))
        if checkpoint_agent_count != self.agent_count:
            raise ValueError(
                f"checkpoint agent_count={checkpoint_agent_count} does not match trainer agent_count={self.agent_count}"
            )
        if len(checkpoint_models) != self.agent_count:
            raise ValueError(
                f"checkpoint has {len(checkpoint_models)} models, expected {self.agent_count}"
            )
        if self.shared_policy:
            self.shared_agent.model.load_state_dict(
                self._average_state_dicts(checkpoint_models)
            )
        else:
            for agent, state_dict in zip(self.agents, checkpoint_models):
                agent.model.load_state_dict(state_dict)
        return checkpoint

    @staticmethod
    def _mean_stats(per_agent_stats: List[Dict[str, float]]) -> Dict[str, float]:
        if not per_agent_stats:
            return {}
        keys = sorted({key for stats in per_agent_stats for key in stats})
        return {
            key: float(np.mean([stats[key] for stats in per_agent_stats if key in stats]))
            for key in keys
        }

    @staticmethod
    def _average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("checkpoint does not contain any model state dicts")
        averaged = {}
        for key in state_dicts[0]:
            values = [state_dict[key] for state_dict in state_dicts]
            if torch.is_floating_point(values[0]):
                averaged[key] = torch.stack(values, dim=0).mean(dim=0)
            else:
                averaged[key] = values[0]
        return averaged
