"""Decoder-only GPT model (Llama-style topology, independently implemented)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .attention import RotaryEmbedding
from .config import ModelConfig
from .embeddings import LMHead, TokenEmbedding
from .transformer_block import RMSNorm, TransformerBlock

__all__ = ["ModelConfig", "MyGPT", "CausalLMOutput"]


@dataclass
class CausalLMOutput:
    logits: Tensor
    past_key_values: list[tuple[Tensor, Tensor]] | None = None


class MyGPT(nn.Module):
    """Decoder-only transformer.

    Topology: token embedding -> N x (RMSNorm, GQA+RoPE attention, RMSNorm,
    SwiGLU MLP) -> final RMSNorm -> tied LM head. No biases anywhere, which
    matches the GGUF ``llama`` tensor set exactly (minus any pre-trained
    vocabulary, since the tokenizer and weights are ours).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = LMHead(config, self.embed_tokens)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Default init so a freshly constructed model is always valid.

        ``initialize_weights.initialize_model`` re-runs this under a fixed seed
        and verifies the resulting distribution.
        """
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, TokenEmbedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    # ------------------------------------------------------------- plumbing
    def get_input_embeddings(self) -> TokenEmbedding:
        return self.embed_tokens

    def num_parameters(self, trainable_only: bool = True) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            if trainable_only and not p.requires_grad:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        seen: set[int] = set()
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            key = name.replace("embed_tokens", "embedding").replace("lm_head", "lm_head")
            out[key] = p.numel()
        return out

    # -------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D (batch, seq), got shape {tuple(input_ids.shape)}")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must provide one (k, v) pair per layer")

        seq_len = input_ids.shape[1]
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_len + seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"total length {past_len + seq_len} exceeds context window "
                f"{self.config.max_position_embeddings}"
            )

        hidden = self.embed_tokens(input_ids)
        new_kv: list[tuple[Tensor, Tensor]] | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            hidden, kv = layer(hidden, self.rotary, past_kv=layer_past, use_cache=use_cache)
            if new_kv is not None and kv is not None:
                new_kv.append(kv)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        if not return_dict:
            return (logits, new_kv)
        return CausalLMOutput(logits=logits, past_key_values=new_kv)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_token_id: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        """Greedy / nucleus sampling with an incremental KV cache."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=input_ids.device).manual_seed(seed)

        out = self(input_ids, use_cache=True)
        past = out.past_key_values
        next_token = self._sample(out.logits[:, -1, :], temperature, top_k, top_p, generator)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
            out = self(next_token[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = self._sample(out.logits[:, -1, :], temperature, top_k, top_p, generator)
            generated.append(next_token)

        return torch.cat([input_ids] + [t[:, None] for t in generated], dim=1)

    @staticmethod
    def _sample(
        logits: Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature <= 0:  # greedy
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= top_p
            keep[..., 0] = True
            mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_idx, keep)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # --------------------------------------------------------- persistence
    def save_pretrained(self, directory: str | Path, filename: str = "model.safetensors") -> Path:
        """Save config + weights (safetensors if available, else torch .pt)."""
        from .io import save_checkpoint

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        return save_checkpoint(self, directory / filename)

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location: str | torch.device = "cpu", filename: str | None = None) -> "MyGPT":
        from .io import load_checkpoint

        directory = Path(directory)
        config = ModelConfig.load(directory / "config.json")
        model = cls(config)
        load_checkpoint(model, directory, map_location=map_location, filename=filename)
        model.eval()
        return model
