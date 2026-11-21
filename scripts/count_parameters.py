"""Print the exact parameter count with a per-component breakdown.

Compares the analytic formula against the instantiated module and (optionally)
against a checkpoint on disk, so a mismatch cannot go unnoticed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ModelConfig, MyGPT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_config.json")
    parser.add_argument("--checkpoint", default=None, help="optional saved checkpoint to cross-check")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = ModelConfig.load(args.config)
    analytic = config.parameter_breakdown()
    model = MyGPT(config)
    actual = model.num_parameters()

    report = {
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "analytic_total": analytic["total"],
        "instantiated_total": actual,
        "matches": analytic["total"] == actual,
        "breakdown": analytic,
    }

    if args.checkpoint:
        loaded = MyGPT.from_pretrained(args.checkpoint)
        report["checkpoint_total"] = loaded.num_parameters()
        report["checkpoint_matches"] = loaded.num_parameters() == analytic["total"]

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"model              : {config.name}")
    print(f"layers / d_model   : {config.num_hidden_layers} / {config.hidden_size}")
    print(f"heads / kv heads   : {config.num_attention_heads} / {config.num_key_value_heads} (head_dim {config.head_dim})")
    print(f"ffn / vocab / ctx  : {config.intermediate_size} / {config.vocab_size} / {config.max_position_embeddings}")
    print()
    for key in ("per_layer", "attention", "mlp", "layer_norms", "embedding", "final_norm", "lm_head", "total"):
        print(f"{key:>12}: {analytic[key]:>15,}")
    print()
    print(f"analytic total     : {analytic['total']:,}")
    print(f"instantiated total : {actual:,}")
    print(f"match              : {analytic['total'] == actual}")
    for dtype, nbytes in (("fp32", 4), ("fp16/bf16", 2), ("int8", 1), ("int4 (packed)", 0.5)):
        print(f"raw weight size {dtype:>14}: {analytic['total'] * nbytes / 2**20:8.1f} MiB")


if __name__ == "__main__":
    main()
