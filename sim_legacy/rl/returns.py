import torch


# Monte Carlo return の計算
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0

    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.append(g)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def normalize_returns(returns: torch.Tensor) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)


# Generalized Advantage Estimation (GAE)
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    gae_lambda=0 --> 1-step TD Advantage
    gae_lambda=1 --> Monte Carlo Advantage
    """

    advantages = torch.zeros_like(rewards)

    gae = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

    next_value = torch.as_tensor(last_value, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(len(rewards))):
        # TD誤差: 次ステップの価値予測の実際の価値のずれ
        delta = (rewards[t] + gamma * next_value) - values[t]

        # Advantage は TD誤差をそのまま使うのではなく、将来 TD誤差を減衰させながら伝播させる
        # これにより、安定した長期評価が出来る
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

        next_value = values[t]

        # gae_lambda=1 のとき:
        # gae = ((rewards[t] + γ * values[t+1]) - values[t]) + γ * ((rewards[t+1] + γ * values[t+1] + γ * (...))
        #     = (rewards[t] + γ * rewards[t+1] + γ^2 * rewards[t+2] + ...) - values[t]
        # 中で value が打ち消し合って Monte Carlo となる

    value_targets = advantages + values

    return advantages, value_targets
