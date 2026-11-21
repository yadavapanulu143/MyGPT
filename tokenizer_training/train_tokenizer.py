"""Train the production 32,768-entry byte-level BPE tokenizer.

Implementation choice
---------------------
Training runs on the Rust BPE trainer from Hugging Face ``tokenizers`` (fast,
well tested, deterministic for a pinned version). Inference, export and the GGUF
vocabulary are produced by this repository's own dependency-free
``ByteLevelBPETokenizer``, which is the code path llama.cpp is verified against.
The two are reconciled here: HF assigns ids 256+i to merge i, and this script
rebuilds the same id space with the four specials at 0-3 and the 256 byte tokens
at 4-259, then asserts that every id maps to the same byte string in both.

Why not SentencePiece: its ▁-marker/NFKC design would change the token-id space,
break byte-exact round trips for arbitrary UTF-8, and require a different GGUF
tokenizer model. The existing GPT-2-style byte-level BPE already matches
llama.cpp's ``gpt2`` tokenizer implementation exactly (verified in Stage 1).

Usage:
    python3 tokenizer_training/train_tokenizer.py \
        --corpus /workspace/corpus/processed/train \
        --out tokenizer --vocab-size 32768
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.tokenizer import (  # noqa: E402
    BYTE_OFFSET,
    NUM_BYTES,
    SPECIAL_TOKENS,
    ByteLevelBPETokenizer,
    bytes_to_unicode,
)


def iter_corpus(shard_paths: list[Path], block_chars: int = 4_000_000):
    """Yield large blocks of text; the trainer handles pre-tokenisation itself."""
    for path in shard_paths:
        buffer: list[str] = []
        size = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                buffer.append(line)
                size += len(line)
                if size >= block_chars:
                    yield "".join(buffer)
                    buffer, size = [], 0
        if buffer:
            yield "".join(buffer)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="/workspace/corpus/processed/train")
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-shards", type=int, default=None, help="use only the first N shards (pilot runs)")
    parser.add_argument("--corpus-manifest", default=None, help="corpus manifest to hash into the tokenizer manifest")
    parser.add_argument("--label", default="full", help="free-form label recorded in the manifest")
    parser.add_argument("--work-dir", default="/tmp/tokenizer_train")
    args = parser.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    shards = sorted(Path(args.corpus).glob("*.txt"))
    if not shards:
        raise SystemExit(f"no corpus shards found in {args.corpus}")
    if args.max_shards:
        shards = shards[: args.max_shards]
    corpus_bytes = sum(p.stat().st_size for p in shards)

    n_specials = len(SPECIAL_TOKENS)
    hf_vocab_size = args.vocab_size - n_specials
    n_merges_expected = hf_vocab_size - NUM_BYTES
    if n_merges_expected <= 0:
        raise SystemExit("vocab size too small")

    print(f"[train] shards={len(shards)} bytes={corpus_bytes / 2**20:.1f} MiB")
    print(f"[train] target vocab={args.vocab_size} = {n_specials} specials + {NUM_BYTES} bytes + {n_merges_expected} merges")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=hf_vocab_size,
        min_frequency=args.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
        special_tokens=[],
    )

    start = time.perf_counter()
    tokenizer.train_from_iterator(iter_corpus(shards), trainer=trainer)
    train_seconds = time.perf_counter() - start
    hf_vocab = tokenizer.get_vocab()
    print(f"[train] hf vocab={len(hf_vocab)} in {train_seconds:.1f}s")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    hf_json_path = work_dir / "tokenizer.json"
    tokenizer.save(str(hf_json_path))
    hf_model = json.loads(hf_json_path.read_text(encoding="utf-8"))["model"]
    hf_merges = hf_model["merges"]
    hf_vocab = hf_model["vocab"]

    # GPT-2 byte-unicode alphabet, inverted: printable char -> raw byte.
    char_to_byte = {ch: b for b, ch in bytes_to_unicode().items()}

    def to_bytes(token: str) -> bytes:
        try:
            return bytes(char_to_byte[ch] for ch in token)
        except KeyError as exc:  # pragma: no cover - would mean a different alphabet
            raise SystemExit(f"token {token!r} uses a character outside the GPT-2 byte alphabet") from exc

    merges: list[tuple[int, int]] = []
    merged_bytes_to_id: dict[bytes, int] = {bytes([b]): BYTE_OFFSET + b for b in range(NUM_BYTES)}
    for rank, merge in enumerate(hf_merges):
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        left_id = merged_bytes_to_id[to_bytes(left)]
        right_id = merged_bytes_to_id[to_bytes(right)]
        merged = to_bytes(left + right)
        new_id = BYTE_OFFSET + NUM_BYTES + rank
        expected = hf_vocab.get(left + right)
        if expected is not None and expected + n_specials != new_id:
            raise SystemExit(f"id mismatch at merge {rank}: hf={expected} ours={new_id - n_specials}")
        merged_bytes_to_id[merged] = new_id
        merges.append((left_id, right_id))

    tokenizer_out = ByteLevelBPETokenizer(vocab_size=args.vocab_size, merges=merges)
    if tokenizer_out.vocab_size != args.vocab_size:
        raise SystemExit(
            f"trained tokenizer has {tokenizer_out.vocab_size} entries, expected {args.vocab_size} "
            f"(merges={len(merges)}); lower --min-frequency or add corpus"
        )

    # Cross-check every id against the HF tokenizer's byte string.
    hf_by_id = {idx: tok for tok, idx in hf_vocab.items()}
    for token_id in range(BYTE_OFFSET + NUM_BYTES, args.vocab_size):
        hf_token = hf_by_id.get(token_id - n_specials)
        if hf_token is None:
            raise SystemExit(f"hf vocab missing id {token_id - n_specials}")
        if to_bytes(hf_token) != tokenizer_out.token_bytes(token_id):
            raise SystemExit(f"byte mismatch at id {token_id}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = tokenizer_out.to_dict()
    config["description"] = (
        "Independent byte-level BPE trained for English on a documented corpus "
        "(see tokenizer/tokenizer_manifest.json). GPT-2 compatible: llama.cpp "
        "loads it as tokenizer.ggml.model=gpt2."
    )
    config["trained"] = True
    config["training"] = {
        "algorithm": "byte-level BPE (GPT-2 style, add_prefix_space=False, use_regex=True)",
        "trainer": f"tokenizers {__import__('tokenizers').__version__} (BpeTrainer, training only)",
        "vocab_size": args.vocab_size,
        "num_merges": len(merges),
        "min_frequency": args.min_frequency,
        "corpus_shards": [p.name for p in shards],
        "corpus_bytes": corpus_bytes,
        "corpus_manifest_sha256": sha256_file(Path(args.corpus_manifest)) if args.corpus_manifest else None,
        "train_seconds": round(train_seconds, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Interop artifacts: GPT-2 / HF style vocab and merges files.
    vocab = {}
    gguf_tokens = tokenizer_out.gguf_tokens()
    for token_id in range(args.vocab_size):
        if token_id in tokenizer_out.special_ids:
            vocab[tokenizer_out.special_ids[token_id]] = token_id
        else:
            vocab[gguf_tokens[token_id]] = token_id
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")

    enc = bytes_to_unicode()
    merge_lines = ["#version: 0.2"]
    for left_id, right_id in merges:
        left = "".join(enc[b] for b in tokenizer_out.token_bytes(left_id))
        right = "".join(enc[b] for b in tokenizer_out.token_bytes(right_id))
        merge_lines.append(f"{left} {right}")
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "special_tokens": tokenizer_out.special_tokens,
                "byte_offset": BYTE_OFFSET,
                "num_byte_tokens": NUM_BYTES,
                "first_merge_id": BYTE_OFFSET + NUM_BYTES,
                "vocab_size": args.vocab_size,
                "notes": "ids 0-3 specials, 4-259 raw bytes, 260+ merges in rank order",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "label": args.label,
        "vocab_size": tokenizer_out.vocab_size,
        "num_merges": len(merges),
        "special_tokens": tokenizer_out.special_tokens,
        "corpus": {
            "shards": [p.name for p in shards],
            "bytes": corpus_bytes,
            "manifest_sha256": sha256_file(Path(args.corpus_manifest)) if args.corpus_manifest else None,
        },
        "training": config["training"],
        "artifacts": {
            name: sha256_file(out_dir / name)
            for name in ("config.json", "vocab.json", "merges.txt", "special_tokens_map.json")
        },
    }
    (out_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[done] vocab={tokenizer_out.vocab_size} merges={len(merges)} -> {out_dir}")
    print(json.dumps(manifest["artifacts"], indent=2))


if __name__ == "__main__":
    main()
