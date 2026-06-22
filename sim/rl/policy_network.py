import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli


class PolicyNetwork(nn.Module):
    """Actor: 行動分布を出力する"""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.action_mean = nn.Linear(hidden_dim, 2)

        self.log_std = nn.Parameter(torch.full((2,), -0.5))

        self.fire_logit = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)

        mean = self.action_mean(x)  # shape: (batch_size, 2)

        std = self.log_std.exp()  # shape: (2,)
        std = std.expand_as(mean)  # shape: (batch_size, 2)

        fire_logit = self.fire_logit(x)  # shape: (batch_size, 1)
        fire_logit = fire_logit.squeeze(-1)  # shape: (batch_size,)

        return mean, std, fire_logit

    def sample_action(self, obs: torch.Tensor):
        single_case = obs.dim() == 1
        if single_case:
            # 単一の観測に対してアクションをサンプリングする場合
            obs = obs.unsqueeze(0)

        mean, std, fire_logit = self.forward(obs)

        # 平均 mean, 標準偏差 std の正規分布からサンプリング
        normal = Normal(mean, std)
        raw_action = normal.sample()  # shape: (batch_size, 2)

        # [:, N] は [全部, N番目のアクション] という意味
        accel = torch.tanh(raw_action[:, 0])  # [-1.0, 1.0], shape: (batch_size,)
        turn = torch.tanh(raw_action[:, 1])  # [-1.0, 1.0], shape: (batch_size,)

        # Bernoulli(torch.sigmoid(fire_logit)) と同じ
        fire_dist = Bernoulli(logits=fire_logit)  # shape: (batch_size,)
        fire = fire_dist.sample()  # {0, 1}, shape: (batch_size,)

        action = torch.stack([accel, turn, fire], dim=-1)  # shape: (batch_size, 3)

        if single_case:
            action = action.squeeze(0)
            raw_action = raw_action.squeeze(0)

        return action, raw_action

    def log_prob_from_raw_action(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor,
        fires: torch.Tensor,
    ) -> torch.Tensor:
        """アクションに対するサンプリング時の log 確率を計算する。"""
        mean, std, fire_logit = self.forward(observations)
        normal = Normal(mean, std)
        fire_dist = Bernoulli(logits=fire_logit)

        # 各行動成分ごとに、その値が正規分布 N(mean, std) から出る log 確率密度を計算する
        # 2次元行動の log 確率を足し合わせて、各サンプルごとの log π(a|s) にする
        # つまり log π(a|s) = log N(a_1|mean_1, std_1) + log N(a_2|mean_2, std_2)
        log_prob = normal.log_prob(raw_actions).sum(dim=-1)  # shape: (batch_size,)

        log_prob = log_prob + fire_dist.log_prob(fires)  # shape: (batch_size,)

        return log_prob

    def supervised_loss(self, observations: torch.Tensor, actions: torch.Tensor):
        mean, _, fire_logit = self.forward(observations)

        # 平均二乗誤差 (Mean Squared Error)
        return F.mse_loss(
            torch.tanh(mean),  # (batch_size, 2)
            actions[:, :2],
        )  # + F.binary_cross_entropy_with_logits(fire_logit, actions[:, 2])
