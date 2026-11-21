"""Pre-norm transformer block: RMSNorm -> GQA attention -> RMSNorm -> SwiGLU MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import Attention, RotaryEmbedding
from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Cheaper than LayerNorm: one pass, no mean, and it matches the norm used by
    llama.cpp's ``llama`` architecture so the exported weights are native there.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLUMLP(nn.Module):
    """Gated feed-forward network: down(silu(gate(x)) * up(x)).

    Three matrices instead of two, but the gating noticeably improves quality per
    parameter; 2.75x expansion (2816 for d=1024) keeps the budget at ~0.5B total.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d, ffn = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(d, ffn, bias=config.mlp_bias)
        self.up_proj = nn.Linear(d, ffn, bias=config.mlp_bias)
        self.down_proj = nn.Linear(ffn, d, bias=config.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attn_out, new_kv = self.attn(self.attn_norm(x), rope, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv
