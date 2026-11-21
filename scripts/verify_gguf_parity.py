"""Check that the GGUF export reproduces the PyTorch reference model.

Two checks:
  1. tokenization parity  - llama.cpp's BPE must produce the same ids as ours,
  2. decoding parity      - greedy next-token ids from llama.cpp must match the
                            PyTorch reference for the same prompt.

Requires a llama.cpp build plus the small harness in ``tools/llama_parity.cpp``.
If the harness is missing the script reports "skipped" instead of failing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MyGPT  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

PROMPT = "Hello world, this is a test of the model."


def pytorch_reference(checkpoint: str, prompt: str, n_predict: int) -> dict:
    tok = ByteLevelBPETokenizer.load("tokenizer")
    model = MyGPT.from_pretrained(checkpoint, map_location="cpu")
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        generated = []
        for _ in range(n_predict):
            next_id = int(out.logits[0, -1].argmax())
            generated.append(next_id)
            out = model(torch.tensor([[next_id]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
    return {"prompt_ids": ids[0].tolist(), "generated_ids": generated}


def llama_cpp_reference(harness: str, gguf: str, prompt: str, n_predict: int) -> dict:
    result = subprocess.run(
        [harness, gguf, prompt, str(n_predict)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--gguf", default="dist/mygpt-0.5b-f16.gguf")
    parser.add_argument("--harness", default="/tmp/llama_parity")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--n-predict", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(4)
    reference = pytorch_reference(args.checkpoint, args.prompt, args.n_predict)
    report = {"prompt": args.prompt, "pytorch": reference}

    if not shutil.which(args.harness) and not Path(args.harness).exists():
        report["llama_cpp"] = None
        report["status"] = "skipped (llama_parity harness not built)"
        print(json.dumps(report, indent=2))
        return 0

    llama = llama_cpp_reference(args.harness, args.gguf, args.prompt, args.n_predict)
    report["llama_cpp"] = llama
    report["tokenizer_match"] = llama["prompt_ids"] == reference["prompt_ids"]
    report["greedy_match"] = llama["generated_ids"] == reference["generated_ids"]
    report["status"] = "ok" if report["tokenizer_match"] and report["greedy_match"] else "mismatch"
    print(json.dumps(report, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
