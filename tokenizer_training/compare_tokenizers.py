"""Compare tokenizer candidates on English, code, identifiers and terminology.

Every measurement runs through this repository's ``ByteLevelBPETokenizer``, the
same code path llama.cpp is verified against, so the numbers describe the
tokenizer that will actually be shipped rather than the trainer's own encoder.

Usage:
    python3 tokenizer_training/compare_tokenizers.py \
        --tokenizer baseline=tokenizer \
        --tokenizer specials=tokenizer_candidates/stage2.1-specials \
        --tokenizer code=tokenizer_candidates/stage2.1-code \
        --english /workspace/corpus/processed/eval/heldout.txt \
        --code /workspace/corpus/code/eval/code-0000.txt \
        --json-out tokenizer_training/manifests/tokenizer_comparison.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.tokenizer import BYTE_OFFSET, NUM_BYTES, ByteLevelBPETokenizer  # noqa: E402

JAVA_IDENTIFIERS = [
    "AbstractSingletonProxyFactoryBean",
    "ConcurrentHashMap",
    "IllegalArgumentException",
    "getParameterAnnotations",
    "serialVersionUID",
    "doFilterInternal",
    "setAccessible",
    "ResponseEntity",
    "InputStreamReader",
    "UnsupportedOperationException",
    "applicationContext",
    "transactionManager",
    "ObjectOutputStream",
    "checkNotNull",
    "toString",
    "equals",
    "hashCode",
    "getDeclaredFields",
]

PYTHON_IDENTIFIERS = [
    "self",
    "None",
    "__init__",
    "staticmethod",
    "collections",
    "defaultdict",
    "itertools",
    "deque",
    "getattr",
    "isinstance",
    "subprocess",
    "DataFrame",
    "np",
    "append",
    "kwargs",
    "pytest",
    "ValueError",
    "asyncio",
    "dataclass",
    "enumerate",
    "zip",
    "lambda",
]

TECHNICAL_TERMS = [
    "polymorphism", "inheritance", "encapsulation", "abstraction", "concurrency",
    "parallelism", "asynchronous", "serialization", "deserialization", "dependency injection",
    "garbage collection", "memory allocation", "stack overflow", "heap", "pointer",
    "compiler", "interpreter", "bytecode", "virtual machine", "runtime",
    "database", "transaction", "isolation level", "normalization", "index",
    "kubernetes", "containerization", "microservice", "load balancer", "reverse proxy",
    "encryption", "authentication", "authorization", "hashing", "checksum",
    "regression", "classification", "clustering", "gradient descent", "backpropagation",
    "neural network", "transformer", "attention mechanism", "tokenization", "embedding",
    "quantization", "inference", "checkpoint", "hyperparameter", "overfitting",
]


def load_slice(path: Path, target_bytes: int, seed: int = 1234) -> str:
    """Deterministic contiguous slice from the middle of a held-out file."""
    data = path.read_bytes()
    if len(data) <= target_bytes:
        return data.decode("utf-8", "ignore")
    rng = random.Random(seed)
    start = rng.randrange(0, len(data) - target_bytes)
    # align to a line boundary so we do not split a UTF-8 sequence
    start = data.find(b"\n", start) + 1
    end = data.find(b"\n", start + target_bytes)
    if end == -1:
        end = len(data)
    return data[start:end].decode("utf-8", "ignore")


def compression(tok: ByteLevelBPETokenizer, text: str) -> dict:
    start = time.perf_counter()
    ids = tok.encode(text)
    elapsed = time.perf_counter() - start
    raw = len(text.encode("utf-8"))
    return {
        "bytes": raw,
        "tokens": len(ids),
        "bytes_per_token": round(raw / len(ids), 3),
        "tokens_per_kib": round(len(ids) / (raw / 1024), 1),
        "encode_seconds": round(elapsed, 2),
    }


def identifier_stats(tok: ByteLevelBPETokenizer, names: list[str]) -> dict:
    rows = {name: len(tok.encode(name)) for name in names}
    single = sum(1 for n in rows.values() if n == 1)
    return {
        "identifiers": len(names),
        "total_tokens": sum(rows.values()),
        "mean_tokens": round(statistics.mean(rows.values()), 3),
        "single_token": single,
        "single_token_rate": round(single / len(names), 3),
        "per_identifier": rows,
    }


def term_stats(tok: ByteLevelBPETokenizer, terms: list[str]) -> dict:
    rows = {term: len(tok.encode(term)) for term in terms}
    return {
        "terms": len(terms),
        "total_tokens": sum(rows.values()),
        "mean_tokens": round(statistics.mean(rows.values()), 3),
        "single_token": sum(1 for n in rows.values() if n == 1),
        "per_term": rows,
    }


def structural_checks(tok: ByteLevelBPETokenizer, rng: random.Random) -> dict:
    tokens = tok.gguf_tokens()
    types = tok.gguf_token_types()
    merges = tok.gguf_merges()
    token_set = set(tokens)
    unresolved = [m for m in merges if any(part not in token_set for part in m.split(" "))]
    duplicates = len(tokens) - len(set(tokens))

    lossless = True
    for _ in range(200):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
        text = blob.decode("latin-1")
        ids = tok.encode(text)
        if tok.decode_bytes(ids, skip_special_tokens=False) != text.encode("utf-8"):
            lossless = False
            break

    special_ids = sorted(tok.special_tokens.values())
    return {
        "vocab_size": tok.vocab_size,
        "vocab_size_target": tok.vocab_size_target,
        "merges": len(tok.merges),
        "first_merge_id": BYTE_OFFSET + NUM_BYTES,
        "last_merge_id": tok.merge_limit - 1,
        "special_ids": special_ids,
        "special_tokens": {t: i for t, i in sorted(tok.special_tokens.items(), key=lambda kv: kv[1])},
        "control_tokens": sum(1 for t in types if t == 3),
        "unused_slots": sum(1 for t in types if t == 5),
        "duplicate_token_strings": duplicates,
        "unresolved_merges": len(unresolved),
        "byte_level_roundtrip_lossless": lossless,
        "gguf_token_count": len(tokens),
        "chat_template": tok.chat_template is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", action="append", required=True, help="label=directory")
    parser.add_argument("--english", default="/workspace/corpus/processed/eval/heldout.txt")
    parser.add_argument("--code", default="/workspace/corpus/code/eval/code-0000.txt")
    parser.add_argument("--english-bytes", type=int, default=1_000_000)
    parser.add_argument("--code-bytes", type=int, default=1_000_000)
    parser.add_argument("--json-out", default="tokenizer_training/manifests/tokenizer_comparison.json")
    args = parser.parse_args()

    english = load_slice(Path(args.english), args.english_bytes)
    code = load_slice(Path(args.code), args.code_bytes, seed=99)
    print(f"[data] english={len(english) / 1e6:.2f} MB code={len(code) / 1e6:.2f} MB")

    results: dict[str, dict] = {}
    rng = random.Random(7)
    for spec in args.tokenizer:
        label, _, directory = spec.partition("=")
        tok = ByteLevelBPETokenizer.load(directory)
        print(f"[eval] {label} <- {directory}", flush=True)
        entry = {
            "directory": directory,
            "structural": structural_checks(tok, rng),
            "english": compression(tok, english),
            "code": compression(tok, code),
            "java_identifiers": identifier_stats(tok, JAVA_IDENTIFIERS),
            "python_identifiers": identifier_stats(tok, PYTHON_IDENTIFIERS),
            "technical_terms": term_stats(tok, TECHNICAL_TERMS),
        }
        mixed = english + "\n" + code
        entry["mixed_93_7"] = compression(tok, mixed)
        entry["fingerprint"] = hashlib.sha256(
            json.dumps(sorted(tok.special_tokens.items())).encode()
            + b"".join(tok.token_bytes(i) for i in range(min(1024, tok.vocab_size)))
        ).hexdigest()[:16]
        results[label] = entry

    baseline_label = next(iter(results))
    baseline = results[baseline_label]
    for label, entry in results.items():
        entry["deltas_vs_baseline"] = {
            "english_bytes_per_token": round(
                entry["english"]["bytes_per_token"] - baseline["english"]["bytes_per_token"], 3
            ),
            "english_tokens_pct": round(
                100 * (entry["english"]["tokens"] / baseline["english"]["tokens"] - 1), 2
            ),
            "code_bytes_per_token": round(
                entry["code"]["bytes_per_token"] - baseline["code"]["bytes_per_token"], 3
            ),
            "code_tokens_pct": round(100 * (entry["code"]["tokens"] / baseline["code"]["tokens"] - 1), 2),
            "java_identifier_tokens_pct": round(
                100 * (entry["java_identifiers"]["total_tokens"] / baseline["java_identifiers"]["total_tokens"] - 1), 2
            ),
            "python_identifier_tokens_pct": round(
                100 * (entry["python_identifiers"]["total_tokens"] / baseline["python_identifiers"]["total_tokens"] - 1), 2
            ),
            "technical_term_tokens_pct": round(
                100 * (entry["technical_terms"]["total_tokens"] / baseline["technical_terms"]["total_tokens"] - 1), 2
            ),
        }

    payload = {"baseline": baseline_label, "english_source": args.english, "code_source": args.code, "results": results}
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    header = f"{'metric':<34}" + "".join(f"{label:>14}" for label in results)
    print("\n" + header)
    print("-" * len(header))
    rows = [
        ("vocab size", lambda r: r["structural"]["vocab_size"]),
        ("merges", lambda r: r["structural"]["merges"]),
        ("english bytes/token", lambda r: r["english"]["bytes_per_token"]),
        ("english tokens", lambda r: r["english"]["tokens"]),
        ("code bytes/token", lambda r: r["code"]["bytes_per_token"]),
        ("code tokens", lambda r: r["code"]["tokens"]),
        ("java ident mean tokens", lambda r: r["java_identifiers"]["mean_tokens"]),
        ("python ident mean tokens", lambda r: r["python_identifiers"]["mean_tokens"]),
        ("technical term mean", lambda r: r["technical_terms"]["mean_tokens"]),
        ("technical single-token", lambda r: r["technical_terms"]["single_token"]),
    ]
    for name, getter in rows:
        print(f"{name:<34}" + "".join(f"{getter(r):>14}" for r in results.values()))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
