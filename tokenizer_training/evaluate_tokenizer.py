"""Evaluate a trained tokenizer and emit a JSON + markdown report.

Metrics: compression on held-out text, round-trip exactness, vocabulary
integrity, category probes (code, URLs, numbers, contractions, Unicode) and,
optionally, id-for-id parity against the Hugging Face tokenizer that produced
the merges.

Usage:
    python3 tokenizer_training/evaluate_tokenizer.py --tokenizer tokenizer \
        --eval-text /workspace/corpus/processed/eval/heldout.txt \
        --json-out tokenizer_training/manifests/tokenizer_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.tokenizer import ByteLevelBPETokenizer, bytes_to_unicode  # noqa: E402

PROBES = {
    "plain_english": "The quick brown fox jumps over the lazy dog, and then it rests.",
    "contractions": "I don't think we'll ever see it's done, but they're sure you'd know.",
    "numbers_math": "The result is 3.14159 * (2 + 7) = 28.27431, roughly 1e-3 off at t=0.95.",
    "code": (
        "def train_step(model, batch, optimizer):\n"
        "    loss = model(**batch).loss\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    return loss.item()  # <class 'float'>\n"
    ),
    "url_email": "Mail support@example.co.uk or see https://example.com/docs?page=2&q=bpe#results",
    "abbreviations": "Dr. Smith (e.g. the CEO) met Mr. Jones at 5 p.m. on Jan. 3rd, i.e. yesterday.",
    "unicode": "Café naïve résumé — “quoted” text… 日本語のテキスト, Ελληνικά, emoji 🙂🚀",
    "whitespace": "line one\nline two\n\n\tindented line\r\nwindows newline",
    "technical": "Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` then run main.py.",
}


def load_sample(path: Path, limit_bytes: int) -> str:
    chunks: list[str] = []
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            chunks.append(line)
            total += len(line.encode("utf-8"))
            if total >= limit_bytes:
                break
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--eval-text", default="/workspace/corpus/processed/eval/heldout.txt")
    parser.add_argument("--sample-mb", type=float, default=1.0)
    parser.add_argument("--hf-model", default=None, help="tokenizer.json produced by the HF trainer, for parity")
    parser.add_argument("--hf-parity-mb", type=float, default=0.2)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer.load(args.tokenizer)

    # ---- vocabulary integrity
    tokens = tok.gguf_tokens()
    vocab_report = {
        "vocab_size": tok.vocab_size,
        "target_vocab_size": tok.vocab_size_target,
        "num_merges": len(tok.merges),
        "unique_ids": len({i for i in range(tok.vocab_size)}),
        "unique_token_bytes": len(set(tok.id_to_bytes)),
        "placeholder_tokens": sum(1 for t in tokens if t.startswith("<unused_")),
        "special_tokens": tok.special_tokens,
        "duplicate_byte_strings": tok.vocab_size - len(set(tok.id_to_bytes)),
    }
    longest = sorted(
        ((len(tok.token_bytes(i)), i, tok.token_text(i)) for i in range(tok.vocab_size)),
        reverse=True,
    )[:15]
    vocab_report["longest_tokens"] = [{"id": i, "bytes": n, "text": t} for n, i, t in longest]

    # ---- held-out compression and round trip
    text = load_sample(Path(args.eval_text), int(args.sample_mb * 2**20))
    raw_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    ids = tok.encode(text)
    encode_seconds = time.perf_counter() - start
    decoded = tok.decode(ids)
    roundtrip_ok = decoded == text
    stats = {
        "sample_bytes": raw_bytes,
        "sample_tokens": len(ids),
        "bytes_per_token": round(raw_bytes / max(1, len(ids)), 3),
        "tokens_per_kib": round(len(ids) / max(1e-9, raw_bytes / 1024), 1),
        "encode_seconds": round(encode_seconds, 2),
        "roundtrip_exact": roundtrip_ok,
        "unknown_tokens": sum(1 for i in ids if i == tok.special_tokens["<unk>"]),
        "special_tokens_in_output": sum(1 for i in ids if i in tok.special_ids),
    }
    if not roundtrip_ok:
        for i, (a, b) in enumerate(zip(text, decoded)):
            if a != b:
                stats["first_mismatch"] = {"index": i, "expected": a, "got": b}
                break

    # ---- per-category probes
    probes = {}
    for name, probe in PROBES.items():
        pids = tok.encode(probe)
        probes[name] = {
            "chars": len(probe),
            "tokens": len(pids),
            "chars_per_token": round(len(probe) / max(1, len(pids)), 2),
            "roundtrip_exact": tok.decode(pids) == probe,
            "ids": pids[:24],
        }

    # ---- determinism
    determinism = tok.encode(text[:50_000]) == tok.encode(text[:50_000])

    # ---- optional parity against the HF trainer output
    parity = None
    if args.hf_model:
        from tokenizers import Tokenizer

        hf = Tokenizer.from_file(args.hf_model)
        sample = load_sample(Path(args.eval_text), int(args.hf_parity_mb * 2**20))
        encoding = hf.encode(sample, add_special_tokens=False)
        our_ids = tok.encode(sample)
        # The HF trainer numbers its 256 base tokens in codepoint order, this
        # project numbers them in byte order, so compare token *strings*.
        char_to_byte = {ch: b for b, ch in bytes_to_unicode().items()}
        hf_bytes = [bytes(char_to_byte[c] for c in token) for token in encoding.tokens]
        our_bytes = [tok.token_bytes(i) for i in our_ids]
        same_len = len(hf_bytes) == len(our_bytes)
        mismatches = sum(1 for a, b in zip(hf_bytes, our_bytes) if a != b) if same_len else None
        parity = {
            "sample_bytes": len(sample.encode("utf-8")),
            "hf_tokens": len(hf_bytes),
            "our_tokens": len(our_bytes),
            "length_match": same_len,
            "mismatched_positions": mismatches,
            "match_rate": round(1 - mismatches / max(1, len(hf_bytes)), 6) if same_len else None,
        }

    report = {
        "tokenizer_dir": str(args.tokenizer),
        "vocabulary": vocab_report,
        "heldout": stats,
        "deterministic": determinism,
        "probes": probes,
        "hf_parity": parity,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
