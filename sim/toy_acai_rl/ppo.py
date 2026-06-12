from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
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
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.mean(hidden), self.log_std.expand_as(self.mean(hidden)), self.fire_logits(hidden), self.value(hidden).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std, fire_logits, value = self.forward(obs)
        if deterministic:
            raw_cont = mean
            fire = (torch.sigmoid(fire_logits) >= 0.5).float()
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
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class RolloutBuffer:
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
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32))
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.dones.append(np.full_like(np.asarray(reward, dtype=np.float32), float(done)))
        self.values.append(np.asarray(value, dtype=np.float32))

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def tensors(self, device: torch.device, last_values: np.ndarray, config: PPOConfig) -> Dict[str, torch.Tensor]:
        obs = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
        next_values = np.asarray(last_values, dtype=np.float32)
        for step in reversed(range(rewards.shape[0])):
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + config.gamma * next_values * next_nonterminal - values[step]
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
            next_values = values[step]

        returns = advantages + values
        flat = {
            "obs": torch.as_tensor(obs.reshape(-1, obs.shape[-1]), device=device),
            "actions": torch.as_tensor(actions.reshape(-1, actions.shape[-1]), device=device),
            "old_log_probs": torch.as_tensor(old_log_probs.reshape(-1), device=device),
            "advantages": torch.as_tensor(advantages.reshape(-1), device=device),
            "returns": torch.as_tensor(returns.reshape(-1), device=device),
        }
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std(unbiased=False) + 1e-8)
        return flat


class PPOTrainer:
    def __init__(self, obs_dim: int, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = ActorCritic(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def act(self, observations: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions, env_actions, log_probs, _, values = self.model.act(obs_tensor, deterministic=deterministic)
        return (
            actions.cpu().numpy(),
            env_actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def values(self, observations: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model.values(obs_tensor).cpu().numpy()

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray) -> Dict[str, float]:
        data = buffer.tensors(self.device, last_values, self.config)
        count = data["obs"].shape[0]
        indices = np.arange(count)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
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
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"] += float(value_loss.detach().cpu())
                stats["entropy"] += float(entropy_mean.detach().cpu())
                updates += 1

        if updates:
            for key in stats:
                stats[key] /= updates
        return stats

    def save(self, path: Path, extra: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": self.config.__dict__, **extra}, path)
