import torch
import torch.nn as nn

from .transformer import (
    ObservationTokenizer,
    QueryAttentionBlock,
    SelfAttentionBlock,
    validate_transformer_dimensions,
)


class ValueNetwork(nn.Module):
    """Critic: 状態価値 V(s) を出力する"""

    def __init__(
        self,
        obs_dim: int,
        token_dim: int,
        attention_heads: int,
        ffn_dim: int,
    ):
        super().__init__()
        validate_transformer_dimensions(token_dim, attention_heads, ffn_dim)

        self.tokenizer = ObservationTokenizer(obs_dim, token_dim)
        self.self_attention = SelfAttentionBlock(token_dim, attention_heads, ffn_dim)
        self.aggregate_attention = QueryAttentionBlock(
            token_dim, attention_heads, ffn_dim
        )
        self.aggregate_token = nn.Parameter(
            torch.zeros(1, 1, token_dim)
        )  # 学習可能な集約トークン
        self.value_head = nn.Linear(token_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        single_case = obs.dim() == 1
        if single_case:
            obs = obs.unsqueeze(0)

        tokens, padding_mask = self.tokenizer(obs)

        # 第一層
        tokens = self.self_attention(tokens, padding_mask)

        # 第二層
        aggregate_query = self.aggregate_token.expand(obs.shape[0], -1, -1)
        x = self.aggregate_attention(aggregate_query, tokens, padding_mask).squeeze(1)

        value = self.value_head(x).squeeze(-1)

        return value.squeeze(0) if single_case else value
