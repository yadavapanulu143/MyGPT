"""Check that the GGUF export reproduces the PyTorch reference model.

Two checks:
  1. tokenization parity - llama.cpp's BPE must produce the same ids as our
     Python tokenizer for a suite of prompts (plain text, code, Unicode,
     numbers/URLs, contractions, abbreviations, long input),
  2. decoding parity      - greedy next-token ids from llama.cpp must match the
     PyTorch reference for the same prompt.

Requires a llama.cpp build plus the small harness in ``tools/llama_parity.cpp``.
If the harness is missing the script reports "skipped" instead of failing.

    python3 scripts/verify_gguf_parity.py --gguf dist/mygpt-0.5b-f16.gguf
    python3 scripts/verify_gguf_parity.py --gguf dist/mygpt-0.5b-q4_k_m.gguf --skip-greedy
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

PROMPTS = {
    "plain": PROMPT,
    "contractions": "I don't think we'll ever see it's done, but they're sure you'd know.",
    "numbers_url": "Ratio 3.14159 vs 1e-3, see https://example.com/docs?page=2&q=bpe#results or a@b.co.uk",
    "code": "def train_step(model, batch):\n    loss = model(**batch).loss\n    return loss.item()",
    "unicode": "Café naïve — “quoted” text… 日本語 🙂 3.14159",
    "abbreviations": "Dr. Smith (e.g. the CEO) met Mr. Jones at 5 p.m. on Jan. 3rd, i.e. yesterday.",
    "long": ("The quick brown fox jumps over the lazy dog. " * 60).strip(),
}


def pytorch_ids(prompt: str) -> list[int]:
    return ByteLevelBPETokenizer.load("tokenizer").encode(prompt)


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


def llama_cpp_reference(harness: str, gguf: str, prompt: str, n_predict: int, n_ctx: int = 2048) -> dict:
    result = subprocess.run(
        [harness, gguf, prompt, str(n_predict), str(n_ctx)],
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--gguf", default="dist/mygpt-0.5b-f16.gguf")
    parser.add_argument("--harness", default="/tmp/llama_parity")
    parser.add_argument("--n-predict", type=int, default=8)
    parser.add_argument("--skip-greedy", action="store_true", help="tokenizer parity only (no PyTorch model load)")
    parser.add_argument(
        "--allow-greedy-mismatch",
        action="store_true",
        help="quantized files: logits are perturbed, so a greedy chain may leave the F16 argmax",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    report: dict = {"gguf": args.gguf, "tokenizer": {}, "status": "ok"}

    if not shutil.which(args.harness) and not Path(args.harness).exists():
        report["status"] = "skipped (llama_parity harness not built)"
        print(json.dumps(report, indent=2))
        return 0

    for name, prompt in PROMPTS.items():
        llama = llama_cpp_reference(args.harness, args.gguf, prompt, 0)
        ours = pytorch_ids(prompt)
        entry = {
            "tokens": len(ours),
            "llama_tokens": len(llama["prompt_ids"]),
            "match": llama["prompt_ids"] == ours,
        }
        report["tokenizer"][name] = entry
        if not entry["match"]:
            report["status"] = "mismatch"
            entry["ours"] = ours[:40]
            entry["llama"] = llama["prompt_ids"][:40]

    if not args.skip_greedy:
        torch.set_num_threads(4)
        reference = pytorch_reference(args.checkpoint, PROMPT, args.n_predict)
        llama = llama_cpp_reference(args.harness, args.gguf, PROMPT, args.n_predict)
        greedy = {
            "prompt": PROMPT,
            "pytorch_ids": reference["generated_ids"],
            "llama_ids": llama["generated_ids"],
            "match": reference["generated_ids"] == llama["generated_ids"],
        }
        report["greedy"] = greedy
        if not greedy["match"]:
            greedy["note"] = "quantization noise moves near-tie argmax; tokenizer parity is unaffected"
            if not args.allow_greedy_mismatch:
                report["status"] = "mismatch"

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
