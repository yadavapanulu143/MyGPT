"""Memory and latency profiling of the reference PyTorch implementation.

Measures *this* machine, not the target laptop: peak RSS after loading the
weights, RSS growth while decoding, prefill/decode throughput and the size of
the KV cache. Estimates for the laptop live in docs/memory_analysis.md and are
derived from these numbers, never invented.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ModelConfig, MyGPT  # noqa: E402


def rss_mib() -> float:
    with open("/proc/self/statm", encoding="ascii") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(model: MyGPT, ctx: int, new_tokens: int, warmup: int = 2) -> dict:
    ids = torch.randint(4, 260, (1, ctx))
    with torch.no_grad():
        for _ in range(warmup):
            model(ids[:, : min(8, ctx)])

        start = time.perf_counter()
        out = model(ids, use_cache=True)
        prefill = time.perf_counter() - start
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)

        start = time.perf_counter()
        for _ in range(new_tokens):
            out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        decode = time.perf_counter() - start

    return {
        "context": ctx,
        "prefill_seconds": round(prefill, 3),
        "prefill_tokens_per_second": round(ctx / prefill, 1),
        "decode_tokens": new_tokens,
        "decode_tokens_per_second": round(new_tokens / decode, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--contexts", default="64,256")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    base_rss = rss_mib()
    report: dict = {
        "machine": {
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "baseline_process_rss_mib": round(base_rss, 1),
        "runs": [],
    }

    config = ModelConfig.load(Path(args.checkpoint) / "config.json")
    report["kv_cache_mib"] = {
        str(ctx): round(config.kv_cache_bytes(ctx, dtype_bytes=4) / 2**20, 1)
        for ctx in (128, 512, 1024, 2048)
    }

    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        gc.collect()
        before = rss_mib()
        t0 = time.perf_counter()
        model = MyGPT.from_pretrained(args.checkpoint, map_location="cpu").to(dtype)
        load_seconds = time.perf_counter() - t0
        after = rss_mib()
        weights_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

        entry = {
            "dtype": dtype_name,
            "load_seconds": round(load_seconds, 2),
            "weights_mib": round(weights_mib, 1),
            "rss_after_load_mib": round(after, 1),
            "rss_delta_load_mib": round(after - before, 1),
            "benchmarks": [],
        }
        for ctx in [int(c) for c in args.contexts.split(",")]:
            result = measure(model, ctx, args.new_tokens)
            result["rss_mib"] = round(rss_mib(), 1)
            entry["benchmarks"].append(result)
        report["runs"].append(entry)

        del model
        gc.collect()

    report["peak_rss_mib"] = round(peak_rss_mib(), 1)
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
