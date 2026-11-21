"""Compare full logit vectors: PyTorch reference vs a GGUF file through llama.cpp.

The token-id check in ``verify_gguf_parity.py`` proves the tokenizer agrees. This
proves the numbers do: same prompt, same final-position logits, compared by cosine
similarity, maximum absolute difference and top-k overlap.

F16 is expected to match to ~1e-6 cosine. Quantized files are lossy by design, so
they get a looser threshold and the top-1 token is reported rather than required.

Usage:
    LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin \\
    python3 scripts/verify_gguf_logits.py --gguf dist/mygpt-0.5b-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from model.architecture import MyGPT  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

PROMPT = "def train(model, data): return model.fit(data)"
THRESHOLDS = {"f16": 0.9999, "q8": 0.999, "q5": 0.99, "q4": 0.98}


def torch_logits(checkpoint: str, tokenizer_dir: str, prompt: str) -> torch.Tensor:
    tok = ByteLevelBPETokenizer.load(tokenizer_dir)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    model = MyGPT.from_pretrained(checkpoint, map_location="cpu")
    with torch.no_grad():
        return model(ids, use_cache=False).logits[0, -1]


def llama_logits(harness: str, gguf: str, prompt: str) -> list[float]:
    result = subprocess.run(
        [harness, gguf, prompt, "0", "512", "logits"],
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["logits"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--gguf", default="dist/mygpt-0.5b-f16.gguf")
    parser.add_argument("--harness", default="/tmp/llama_parity")
    parser.add_argument("--quant", choices=sorted(THRESHOLDS), default="f16")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not shutil.which(args.harness) and not Path(args.harness).exists():
        print(json.dumps({"gguf": args.gguf, "status": "skipped (harness not built)"}, indent=2))
        return 0

    reference = torch_logits(args.checkpoint, args.tokenizer, PROMPT)
    observed = torch.tensor(llama_logits(args.harness, args.gguf, PROMPT))

    cosine = torch.nn.functional.cosine_similarity(reference, observed, dim=0).item()
    report = {
        "gguf": args.gguf,
        "prompt": PROMPT,
        "cosine": round(cosine, 8),
        "max_abs_diff": round((reference - observed).abs().max().item(), 6),
        "torch_argmax": int(reference.argmax()),
        "llama_argmax": int(observed.argmax()),
        "top1_match": int(reference.argmax()) == int(observed.argmax()),
        "torch_top5": reference.topk(5).indices.tolist(),
        "llama_top5": observed.topk(5).indices.tolist(),
        "threshold": THRESHOLDS[args.quant],
    }
    report["top5_overlap"] = len(set(report["torch_top5"]) & set(report["llama_top5"]))
    report["status"] = "ok" if cosine >= THRESHOLDS[args.quant] else "below threshold"
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
