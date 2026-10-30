import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Critic: 状態価値 V(s) を出力する"""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.net(obs)).squeeze(-1)
