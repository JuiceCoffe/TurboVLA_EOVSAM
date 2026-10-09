from __future__ import annotations

import torch
from torch import nn
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_seq_len

from .components.utils import MLP
from .configuration import ActionHeadConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ObjectTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """PyTorch decoder layer plus a separate object-memory attention block."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_attn = nn.MultiheadAttention(
            self.self_attn.embed_dim, self.self_attn.num_heads,
            dropout=self.dropout.p, batch_first=self.self_attn.batch_first,
        )
        self.object_norm = nn.LayerNorm(self.self_attn.embed_dim, eps=self.norm1.eps)
        self.object_dropout = nn.Dropout(self.dropout.p)

    def _object_block(self, x, objects):
        return self.object_dropout(self.object_attn(x, objects, objects, need_weights=False)[0])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False, object_memory=None):
        if object_memory is None:
            return super().forward(tgt, memory, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask,
                                   tgt_is_causal, memory_is_causal)
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._object_block(self.object_norm(x), object_memory)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.object_norm(x + self._object_block(x, object_memory))
            x = self.norm3(x + self._ff_block(x))
        return x


class ObjectTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=False, object_memory=None):
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal,
                           object_memory=object_memory)
        return self.norm(output) if self.norm is not None else output


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int,
                 object_feature_dim: int | None = None, num_views: int = 2) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        layer_type = ObjectTransformerDecoderLayer if object_feature_dim is not None else nn.TransformerDecoderLayer
        decoder_layer = layer_type(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_type = ObjectTransformerDecoder if object_feature_dim is not None else nn.TransformerDecoder
        self.decoder = decoder_type(decoder_layer, num_layers=config.num_layers)
        self.object_projections = nn.ModuleList([
            nn.Linear(object_feature_dim, hidden_dim) for _ in range(num_views)
        ]) if object_feature_dim is not None else None
        if self.object_projections is not None:
            self.object_view_embedding = nn.Parameter(torch.randn(1, num_views, 1, hidden_dim) * 0.02)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(self, memory: torch.Tensor, object_features: torch.Tensor | None = None) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if self.object_projections is not None:
            if object_features is None or object_features.ndim != 3:
                raise ValueError("Expected EOVSAM features [B,V*Q,D]")
            num_views = self.object_view_embedding.shape[1]
            if object_features.shape[0] != memory.shape[0] or object_features.shape[1] % num_views:
                raise ValueError("EOVSAM batch/query dimensions do not match the policy")
            view_features = object_features.reshape(object_features.shape[0], num_views, -1, object_features.shape[-1])
            objects = torch.stack([
                projection(view_features[:, view_index])
                for view_index, projection in enumerate(self.object_projections)
            ], dim=1)
            objects = (objects + self.object_view_embedding.to(objects)).flatten(1, 2)
            hidden = self.decoder(tgt=queries, memory=memory, object_memory=objects)
        else:
            hidden = self.decoder(tgt=queries, memory=memory)
        return torch.tanh(self.action_projection(hidden))


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int,
                 object_feature_dim: int | None = None, num_views: int = 2) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward, object_feature_dim, num_views)

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor,
                object_features: torch.Tensor | None = None) -> torch.Tensor:
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        return self.decoder(torch.cat([vision_language_tokens, state_tokens], dim=1), object_features)
