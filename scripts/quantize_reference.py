"""Group-wise INT8/INT4 weight-only quantization of the reference model.

This is an independent, dependency-free quantization path (pure PyTorch) used to
(a) size the storage formats, (b) measure reconstruction error, and (c) show how
much the logits drift. The deployment format used for the laptop is GGUF via
llama.cpp; this script exists so the numbers can be reproduced without it.

Nothing here is meaningful as a *quality* measure for the model: the weights are
random, so the only thing being quantified is numeric error, not language skill.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MyGPT  # noqa: E402


def quantize_groupwise(weight: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax quantization per contiguous group along the input axis."""
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"in_features {in_features} not divisible by group {group_size}")
    flat = weight.reshape(-1, group_size)
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    max_abs = flat.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round(flat / scale), qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_groupwise(q: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int], group_size: int) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32)).reshape(shape)


def storage_bytes(numel: int, bits: int, group_size: int) -> int:
    scales = math.ceil(numel / group_size) * 2  # f16 scale per group
    return math.ceil(numel * bits / 8) + scales


def apply_quantization(model: MyGPT, bits: int, group_size: int, skip_norms: bool = True) -> tuple[MyGPT, dict]:
    """Return a copy of the model whose 2-D weights were quantized and dequantized."""
    quantized = MyGPT(model.config)
    quantized.load_state_dict(model.state_dict())

    total_bytes = 0
    numel = 0
    rel_error_num = 0.0
    rel_error_den = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if skip_norms and param.dim() < 2:
                total_bytes += param.numel() * 4
                continue
            q, scale = quantize_groupwise(param.detach(), bits, group_size)
            deq = dequantize_groupwise(q, scale, tuple(param.shape), group_size)
            quantized.get_parameter(name).copy_(deq)
            total_bytes += storage_bytes(param.numel(), bits, group_size)
            numel += param.numel()
            rel_error_num += float((deq - param).pow(2).sum())
            rel_error_den += float(param.pow(2).sum())
    stats = {
        "bits": bits,
        "group_size": group_size,
        "quantized_parameters": numel,
        "estimated_storage_mib": round(total_bytes / 2**20, 1),
        "relative_weight_error": round(math.sqrt(rel_error_num / rel_error_den), 6),
    }
    return quantized, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = MyGPT.from_pretrained(args.checkpoint)
    ids = torch.randint(4, 260, (1, args.seq_len))
    with torch.no_grad():
        reference = model(ids).logits

    report = {"fp32_storage_mib": round(sum(p.numel() * 4 for p in model.parameters()) / 2**20, 1), "runs": []}
    for bits, group in ((8, 64), (8, 128), (4, 32), (4, 64), (4, 128)):
        quantized, stats = apply_quantization(model, bits, group)
        with torch.no_grad():
            logits = quantized(ids).logits
        diff = logits - reference
        stats["logits_mse"] = round(float(diff.pow(2).mean()), 6)
        stats["logits_max_abs_diff"] = round(float(diff.abs().max()), 4)
        stats["logits_cosine_similarity"] = round(
            float(torch.nn.functional.cosine_similarity(logits.flatten(), reference.flatten(), dim=0)), 6
        )
        stats["top1_token_agreement"] = round(
            float((logits.argmax(-1) == reference.argmax(-1)).float().mean()), 4
        )
        report["runs"].append(stats)
        print(json.dumps(stats))

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
