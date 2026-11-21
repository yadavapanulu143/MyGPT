"""Rotary positional embeddings and grouped-query causal self-attention.

The implementation is deliberately plain PyTorch (no fused CUDA kernels) so the
same code path runs on CPU, and so the exported weights map 1:1 onto the
"llama" tensor layout used by llama.cpp.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig

# torch >= 2.5 can broadcast KV heads inside SDPA; older versions need an
# explicit repeat. Detected once at import time.
try:  # pragma: no cover - version dependent
    import inspect

    _SDPA_GQA = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
except Exception:  # pragma: no cover
    _SDPA_GQA = False


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for RoPE (non-parametric, not checkpointed)."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[-2]
        if offset + seq_len > self.cos_cached.shape[0]:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds max_position_embeddings "
                f"{self.cos_cached.shape[0]}"
            )
        cos = self.cos_cached[offset : offset + seq_len].to(q.dtype)[None, None]
        sin = self.sin_cached[offset : offset + seq_len].to(q.dtype)[None, None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    """Multi-head causal attention with grouped key/value heads (GQA)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, seq_len, _ = x.shape
        offset = 0 if past_kv is None else past_kv[0].shape[-2]

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rope(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=-2)
            v = torch.cat((past_kv[1], v), dim=-2)
        new_kv = (k, v) if use_cache else None

        key_len = k.shape[-2]
        # A causal mask is only required when a query can see future positions.
        # With a single query token over a full past there is nothing to mask.
        is_causal = seq_len > 1 and key_len == seq_len
        attn_mask = None
        if seq_len > 1 and key_len != seq_len:
            causal = torch.ones(seq_len, key_len, dtype=torch.bool, device=x.device).tril(
                diagonal=key_len - seq_len
            )
            attn_mask = causal

        if _SDPA_GQA:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True
            )
        else:  # pragma: no cover - legacy torch
            k_rep = k.repeat_interleave(self.num_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_groups, dim=1)
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out), new_kv
