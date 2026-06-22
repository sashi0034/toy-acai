import torch


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
