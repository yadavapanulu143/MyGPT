"""Checkpoint I/O.

Weights are stored once even when the LM head is tied to the embedding matrix.
``safetensors`` is used when installed (mmap-friendly, no pickle); otherwise a
plain ``torch.save`` state dict is written. Loading is strict so a mismatched
architecture fails loudly instead of silently dropping tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save

    HAS_SAFETENSORS = True
except Exception:  # pragma: no cover
    HAS_SAFETENSORS = False


def unique_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict with aliased (tied) parameters stored only once."""
    seen: set[int] = set()
    out: dict[str, torch.Tensor] = {}
    # state_dict() returns fresh detached views, so dedupe on the storage pointer
    # (tied lm_head/embedding share one storage).
    for name, tensor in model.state_dict(keep_vars=True).items():
        key = tensor.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        out[name] = tensor.detach().contiguous()
    return out


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    state = unique_state_dict(model)
    if HAS_SAFETENSORS and path.suffix == ".safetensors":
        _st_save(state, str(path), metadata={"format": "pt", "origin": "mygpt-random-init"})
    else:
        if path.suffix == ".safetensors":
            path = path.with_suffix(".pt")
        torch.save(state, path)
    return path


def resolve_checkpoint(directory: str | Path, filename: str | None = None) -> Path:
    directory = Path(directory)
    if filename is not None:
        return directory / filename
    for candidate in ("model.safetensors", "model.pt", "model.bin"):
        if (directory / candidate).exists():
            return directory / candidate
    raise FileNotFoundError(f"no checkpoint found in {directory}")


def load_checkpoint(
    model: nn.Module,
    directory: str | Path,
    map_location: str | torch.device = "cpu",
    filename: str | None = None,
) -> nn.Module:
    path = resolve_checkpoint(directory, filename)
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required to load .safetensors checkpoints")
        state = _st_load(str(path), device=str(map_location))
    else:
        state = torch.load(path, map_location=map_location, weights_only=True)

    target = model.state_dict()
    missing = [k for k in target if k not in state]
    unexpected = [k for k in state if k not in target]
    # A tied lm_head may legitimately be absent from disk.
    missing = [k for k in missing if not (k.endswith("lm_head.weight") and "embed_tokens.weight" in state)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    model.load_state_dict(state, strict=False)
    return model
