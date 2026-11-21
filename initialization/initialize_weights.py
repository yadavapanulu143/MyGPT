"""Random weight initialization and verification.

No pre-trained weights are used anywhere in this project. Every parameter is
drawn from a seeded distribution, so the checkpoint is reproducible: the same
seed always yields the same fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from model import ModelConfig, MyGPT
from model.io import unique_state_dict
from model.transformer_block import RMSNorm


def initialize_model(config: ModelConfig, seed: int | None = None, dtype: torch.dtype = torch.float32) -> MyGPT:
    """Build the model and fill every parameter with seeded random values."""
    seed = config.initializer_seed if seed is None else seed
    torch.manual_seed(seed)
    model = MyGPT(config)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if isinstance(model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model, RMSNorm):
                param.fill_(1.0)
            elif param.dim() >= 2:
                param.normal_(mean=0.0, std=config.initializer_range)
            else:
                param.zero_()  # only 1-D params are norm weights, handled above
    return model.to(dtype)


def weight_statistics(model: nn.Module) -> dict[str, float]:
    """Distribution stats over the sampled matrices (norm scales are constant 1.0)."""
    full = unique_state_dict(model)
    total_parameters = sum(t.numel() for t in full.values())
    state = {k: v for k, v in full.items() if v.dim() >= 2}
    total = sum(t.numel() for t in state.values())
    flat_sum = 0.0
    flat_sq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for tensor in state.values():
        t = tensor.float()
        flat_sum += float(t.sum())
        flat_sq += float((t * t).sum())
        minimum = min(minimum, float(t.min()))
        maximum = max(maximum, float(t.max()))
    mean = flat_sum / total
    var = flat_sq / total - mean * mean
    return {
        "num_parameters": total_parameters,
        "num_sampled_matrix_parameters": total,
        "mean": mean,
        "std": var**0.5,
        "min": minimum,
        "max": maximum,
    }


def fingerprint(model: nn.Module) -> str:
    """SHA-256 over tensor names, dtypes, shapes and raw bytes (order-independent)."""
    digest = hashlib.sha256()
    state = unique_state_dict(model)
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


def verify_random_initialization(model: MyGPT, seed: int, tolerance: float = 5e-4) -> dict[str, object]:
    """Check that the weights really are freshly sampled from the configured prior.

    Tests performed:
      1. empirical mean is ~0 and std is ~initializer_range for 2-D weights,
      2. tensors are not constant (would indicate a bad init),
      3. re-running the same seed reproduces the identical fingerprint,
      4. a different seed produces a different fingerprint.
    """
    config = model.config
    stats = weight_statistics(model)
    flat = torch.cat([t.flatten() for t in unique_state_dict(model).values() if t.dim() >= 2])
    constant = bool(torch.all(flat == flat[0]))

    same = fingerprint(initialize_model(config, seed=seed)) == fingerprint(model)
    other = fingerprint(initialize_model(config, seed=seed + 1)) != fingerprint(model)

    checks = {
        "mean_near_zero": abs(stats["mean"]) < 1e-3,
        "std_matches_initializer_range": abs(stats["std"] - config.initializer_range) < tolerance,
        "not_constant": not constant,
        "reproducible_with_same_seed": same,
        "different_with_other_seed": other,
    }
    return {
        "seed": seed,
        "fingerprint": fingerprint(model),
        "statistics": stats,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_manifest(path: str | Path, model: MyGPT, seed: int) -> Path:
    report = verify_random_initialization(model, seed)
    report["config"] = model.config.to_dict()
    report["initializer"] = "torch normal(0, initializer_range); RMSNorm weights = 1.0"
    report["pretrained_weights_used"] = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Initialize MyGPT weights and verify randomness")
    parser.add_argument("--config", default="config/model_config.json")
    parser.add_argument("--out", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = ModelConfig.load(args.config)
    seed = cfg.initializer_seed if args.seed is None else args.seed
    model = initialize_model(cfg, seed=seed)
    model.save_pretrained(args.out)
    manifest = write_manifest(Path(args.out) / "init_manifest.json", model, seed)
    report = json.loads(manifest.read_text())
    print(json.dumps({"passed": report["passed"], "statistics": report["statistics"], "checks": report["checks"]}, indent=2))
