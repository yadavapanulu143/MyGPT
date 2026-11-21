"""End-to-end inference smoke test for the reference PyTorch implementation.

Runs the full path: text -> tokenizer -> model -> sampling -> detokenizer, and
optionally the same prompt through llama.cpp. Output is *not* expected to be
coherent: the weights are random, so this proves the plumbing works, nothing
more.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MyGPT  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402


def run_pytorch(checkpoint: str, prompt: str, max_new_tokens: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    tok = ByteLevelBPETokenizer.load("tokenizer")
    t0 = time.perf_counter()
    model = MyGPT.from_pretrained(checkpoint)
    load_s = time.perf_counter() - t0

    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0)
    gen_s = time.perf_counter() - t0

    new_ids = out[0, ids.shape[1] :].tolist()
    return {
        "backend": "pytorch",
        "load_seconds": round(load_s, 2),
        "generation_seconds": round(gen_s, 2),
        "tokens_per_second": round(max_new_tokens / gen_s, 2),
        "prompt_ids": ids[0].tolist(),
        "generated_ids": new_ids,
        "generated_text_repr": repr(tok.decode(new_ids)),
    }


def run_llamacpp(harness: str, gguf: str, prompt: str, max_new_tokens: int) -> dict:
    result = subprocess.run(
        [harness, gguf, prompt, str(max_new_tokens)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    data = json.loads(result.stdout.strip().splitlines()[-1])
    data["backend"] = "llama.cpp"
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gguf", default=None, help="optional GGUF file to also test through llama.cpp")
    parser.add_argument("--harness", default="/tmp/llama_parity")
    args = parser.parse_args()

    report = {"pytorch": run_pytorch(args.checkpoint, args.prompt, args.max_new_tokens, args.threads)}
    if args.gguf:
        report["llama_cpp"] = run_llamacpp(args.harness, args.gguf, args.prompt, args.max_new_tokens)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
