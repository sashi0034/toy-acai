from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        fire_bias_init: float = 0.4,
        log_std_init: float = -0.8,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, 2)
        self.fire_logits = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), log_std_init))
        nn.init.constant_(self.fire_logits.bias, fire_bias_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
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
            raw_cont = mean
            fire = (torch.sigmoid(fire_logits) >= fire_threshold).float()
        else:
            raw_cont = Normal(mean, log_std.exp()).sample()
            fire = Bernoulli(logits=fire_logits).sample()
        action = torch.cat([raw_cont, fire], dim=-1)
        env_action = torch.cat([torch.tanh(raw_cont), fire], dim=-1)
        log_prob, entropy = self.evaluate_actions(obs, action)
        return action, env_action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256
    rollout_steps: int = 2048
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    fire_bias_init: float = 0.4
    eval_fire_threshold: float = 0.15
    hidden_dim: int = 128
    log_std_init: float = -0.8


class AgentRolloutBuffer:
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

    def tensors(self, device: torch.device, last_value: float, config: PPOConfig) -> Dict[str, torch.Tensor]:
        obs = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
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
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std(unbiased=False) + 1e-8)
        return flat


class RolloutBuffer:
    def __init__(self, agent_count: int = 4):
        self.agent_count = agent_count
        self.agent_buffers = [AgentRolloutBuffer() for _ in range(agent_count)]

    def add(self, obs, action, log_prob, reward, done, value):
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
        data = buffer.tensors(self.device, last_value, self.config)
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
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_mean = entropy.mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip).float().mean()
                )
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
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

    def imitation_update(
        self,
        observations: np.ndarray,
        target_env_actions: np.ndarray,
        epochs: int = 4,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(target_env_actions, dtype=torch.float32, device=self.device)
        target_cont = torch.atanh(torch.clamp(targets[:, :2], -0.999, 0.999))
        target_fire = targets[:, 2:3]
        indices = np.arange(obs_tensor.shape[0])
        stats = {"bc_loss": 0.0, "bc_cont_loss": 0.0, "bc_fire_loss": 0.0}
        updates = 0

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = torch.as_tensor(indices[start : start + batch_size], device=self.device)
                mean, _, fire_logits, _ = self.model.forward(obs_tensor[batch_idx])
                cont_loss = F.mse_loss(mean, target_cont[batch_idx])
                fire_loss = F.binary_cross_entropy_with_logits(
                    fire_logits, target_fire[batch_idx]
                )
                loss = cont_loss + fire_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["bc_loss"] += float(loss.detach().cpu())
                stats["bc_cont_loss"] += float(cont_loss.detach().cpu())
                stats["bc_fire_loss"] += float(fire_loss.detach().cpu())
                updates += 1

        if updates:
            for key in stats:
                stats[key] /= updates
        return stats


class PPOTrainer:
    def __init__(
        self,
        obs_dim: int,
        config: PPOConfig,
        device: torch.device,
        agent_count: int = 4,
    ):
        self.config = config
        self.device = device
        self.agent_count = agent_count
        self.agents = [
            _SingleAgentPPOTrainer(obs_dim, config, device)
            for _ in range(agent_count)
        ]

    def act(self, observations: np.ndarray, deterministic: bool = False):
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
        if buffer.agent_count != self.agent_count:
            raise ValueError(
                f"buffer agent_count={buffer.agent_count} does not match trainer agent_count={self.agent_count}"
            )
        last_values = np.asarray(last_values, dtype=np.float32)
        per_agent_stats = [
            agent.update(buffer.agent_buffers[agent_id], float(last_values[agent_id]))
            for agent_id, agent in enumerate(self.agents)
        ]
        stats = self._mean_stats(per_agent_stats)
        for agent_id, agent_stats in enumerate(per_agent_stats):
            for key, value in agent_stats.items():
                stats[f"agent_{agent_id}_{key}"] = float(value)
        return stats

    def imitation_update(
        self,
        observations: np.ndarray,
        target_env_actions: np.ndarray,
        epochs: int = 4,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        observations = np.asarray(observations, dtype=np.float32)
        target_env_actions = np.asarray(target_env_actions, dtype=np.float32)
        if observations.ndim != 3 or observations.shape[1] != self.agent_count:
            raise ValueError(
                f"expected demonstrations shaped (steps, {self.agent_count}, obs_dim), got {observations.shape}"
            )
        per_agent_stats = [
            agent.imitation_update(
                observations[:, agent_id, :],
                target_env_actions[:, agent_id, :],
                epochs=epochs,
                batch_size=batch_size,
            )
            for agent_id, agent in enumerate(self.agents)
        ]
        stats = self._mean_stats(per_agent_stats)
        for agent_id, agent_stats in enumerate(per_agent_stats):
            for key, value in agent_stats.items():
                stats[f"agent_{agent_id}_{key}"] = float(value)
        return stats

    def save(self, path: Path, extra: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": [agent.model.state_dict() for agent in self.agents],
                "agent_count": self.agent_count,
                "config": self.config.__dict__,
                **extra,
            },
            path,
        )

    def load(self, path: Path) -> Dict[str, object]:
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
