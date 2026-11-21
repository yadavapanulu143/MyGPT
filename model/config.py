"""Model configuration with exact parameter accounting.

Every architectural knob lives here so that the parameter count can be derived
analytically (see ``parameter_breakdown``) and cross-checked against the real
module (see ``scripts/count_parameters.py`` and ``tests/test_parameters.py``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    # --- vocabulary / embeddings -------------------------------------------
    vocab_size: int = 32768
    hidden_size: int = 1024
    num_hidden_layers: int = 41
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2816
    max_position_embeddings: int = 2048

    # --- numerics ----------------------------------------------------------
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-05
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False

    # --- structure ---------------------------------------------------------
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    initializer_seed: int = 1337

    # --- special tokens ----------------------------------------------------
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    name: str = "mygpt-0.5b"
    model_type: str = "mygpt"
    architectures: tuple[str, ...] = ("MyGPTForCausalLM",)

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads * head_dim "
                f"({self.num_attention_heads} * {self.head_dim})"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.vocab_size < 256:
            raise ValueError("vocab_size must leave room for all 256 byte tokens")
        # Q4_K/Q6_K super-blocks in llama.cpp are 256 elements wide; keeping the
        # feed-forward and embedding rows multiples of 256 avoids fallback paths.
        if self.intermediate_size % 256:
            raise ValueError("intermediate_size must be a multiple of 256 for k-quants")

    # ------------------------------------------------------------------ sizes
    @property
    def num_parameters(self) -> int:
        """Analytically derived parameter count (no autograd, no buffers)."""
        return self.parameter_breakdown()["total"]

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def parameter_breakdown(self) -> dict[str, int]:
        d = self.hidden_size
        kv = self.kv_dim
        ffn = self.intermediate_size
        L = self.num_hidden_layers

        attn_q = d * d + (d if self.attention_bias else 0)
        attn_k = d * kv + (kv if self.attention_bias else 0)
        attn_v = d * kv + (kv if self.attention_bias else 0)
        attn_out = d * d + (d if self.attention_bias else 0)
        mlp_gate = d * ffn + (ffn if self.mlp_bias else 0)
        mlp_up = d * ffn + (ffn if self.mlp_bias else 0)
        mlp_down = ffn * d + (d if self.mlp_bias else 0)
        norms_per_layer = 2 * d

        per_layer = attn_q + attn_k + attn_v + attn_out + mlp_gate + mlp_up + mlp_down + norms_per_layer
        embedding = self.vocab_size * d
        final_norm = d
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * d

        return {
            "per_layer": per_layer,
            "attention": L * (attn_q + attn_k + attn_v + attn_out),
            "mlp": L * (mlp_gate + mlp_up + mlp_down),
            "layer_norms": L * norms_per_layer,
            "embedding": embedding,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": L * per_layer + embedding + final_norm + lm_head,
        }

    # ------------------------------------------------------------- kv cache
    def kv_cache_bytes(self, context_length: int, dtype_bytes: int = 2) -> int:
        """KV cache size for one sequence (K and V, all layers, f16 by default)."""
        return 2 * self.num_hidden_layers * self.kv_dim * context_length * dtype_bytes

    # -------------------------------------------------------------- (de)ser
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["architectures"] = list(self.architectures)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if "architectures" in clean:
            clean["architectures"] = tuple(clean["architectures"])
        return cls(**clean)
