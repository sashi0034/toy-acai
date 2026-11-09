import torch
import torch.nn as nn

from .observation import (
    AGENT_FEATURES,
    AGENT_MISSILE_COUNT,
    BOUNDARY_FEATURES,
    ENEMY_MISSILE_COUNT,
    ENTITY_FEATURES,
    OBS_DIM,
    OPPONENT_COUNT,
)


class ObservationTokenizer(nn.Module):
    """固定長の観測を種類別の Linear で Transformer トークンに変換する"""

    def __init__(self, obs_dim: int, token_dim: int):
        super().__init__()
        if obs_dim != OBS_DIM:
            raise ValueError(f"obs_dim must be {OBS_DIM}, got {obs_dim}")

        self.obs_dim = obs_dim
        self.agent = nn.Linear(AGENT_FEATURES, token_dim)
        self.boundary = nn.Linear(BOUNDARY_FEATURES, token_dim)
        self.opponent = nn.Linear(ENTITY_FEATURES - 1, token_dim)
        self.enemy_missile = nn.Linear(ENTITY_FEATURES - 1, token_dim)
        self.agent_missile = nn.Linear(ENTITY_FEATURES - 1, token_dim)

    @staticmethod
    def _take_entities(
        observations: torch.Tensor, offset: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        end = offset + count * ENTITY_FEATURES
        entities = observations[:, offset:end].reshape(
            observations.shape[0], count, ENTITY_FEATURES
        )

        # 各 entity の先頭特徴量は alive である前提とする
        # 存在しない entity の位置が True になるマスクを作り、Attention においてその位置の Key, Value を無視する
        padding_mask = entities[:, :, 0] <= 0.0
        return entities[:, :, 1:], padding_mask, end

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.dim() != 2 or observations.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observations must have shape (batch, {self.obs_dim}), "
                f"got {tuple(observations.shape)}"
            )

        offset = 0
        agent_features = observations[:, offset : offset + AGENT_FEATURES]
        offset += AGENT_FEATURES
        boundary_features = observations[:, offset : offset + BOUNDARY_FEATURES]
        offset += BOUNDARY_FEATURES

        opponent_features, opponent_mask, offset = self._take_entities(
            observations, offset, OPPONENT_COUNT
        )
        enemy_missile_features, enemy_missile_mask, offset = self._take_entities(
            observations, offset, ENEMY_MISSILE_COUNT
        )
        agent_missile_features, agent_missile_mask, offset = self._take_entities(
            observations, offset, AGENT_MISSILE_COUNT
        )
        assert offset == self.obs_dim

        tokens = torch.cat(
            (
                self.agent(agent_features).unsqueeze(1),
                self.boundary(boundary_features).unsqueeze(1),
                self.opponent(opponent_features),
                self.enemy_missile(enemy_missile_features),
                self.agent_missile(agent_missile_features),
            ),
            dim=1,
        )
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (observations.shape[0], 2),
                    dtype=torch.bool,
                    device=observations.device,
                ),
                opponent_mask,
                enemy_missile_mask,
                agent_missile_mask,
            ),
            dim=1,
        )
        return tokens, padding_mask


class FeedForward(nn.Module):
    def __init__(self, token_dim: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class SelfAttentionBlock(nn.Module):
    """全トークンを更新する Pre-LN Transformer ブロック"""

    def __init__(self, token_dim: int, attention_heads: int, ffn_dim: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + attention_output
        return tokens + self.ffn(self.ffn_norm(tokens))


class QueryAttentionBlock(nn.Module):
    """1 つの Query だけを更新する Pre-LN Transformer ブロック"""

    def __init__(self, token_dim: int, attention_heads: int, ffn_dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        attention_output, _ = self.attention(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        query = query + attention_output
        return query + self.ffn(self.ffn_norm(query))


def validate_transformer_dimensions(
    token_dim: int, attention_heads: int, ffn_dim: int
) -> None:
    if token_dim <= 0 or attention_heads <= 0 or ffn_dim <= 0:
        raise ValueError("Transformer dimensions must be positive")
    if token_dim % attention_heads != 0:
        raise ValueError(
            f"token_dim ({token_dim}) must be divisible by "
            f"attention_heads ({attention_heads})"
        )
