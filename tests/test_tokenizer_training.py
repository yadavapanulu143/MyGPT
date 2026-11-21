"""Stage 2: the training pipeline itself (tiny synthetic corpus, no downloads)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("tokenizers", reason="training dependency 'tokenizers' not installed")


def _write_corpus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng_words = [
        "tokenizer", "training", "corpus", "english", "language", "model", "weights",
        "attention", "transformer", "vocabulary", "merges", "bytes", "encoding",
    ]
    lines = []
    for i in range(4000):
        words = [rng_words[(i * 7 + j * 13) % len(rng_words)] for j in range(12)]
        lines.append(f"{i}: " + " ".join(words) + ".\n")
    path = directory / "train-0000.txt"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _train(corpus: Path, out: Path, vocab_size: int) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tokenizer_training" / "train_tokenizer.py"),
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--vocab-size",
            str(vocab_size),
            "--min-frequency",
            "1",
            "--work-dir",
            str(out / "_hf"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads((out / "tokenizer_manifest.json").read_text())


def test_training_pipeline_is_reproducible(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus")
    first = _train(corpus.parent, tmp_path / "tok_a", vocab_size=1024)
    second = _train(corpus.parent, tmp_path / "tok_b", vocab_size=1024)

    assert first["vocab_size"] == second["vocab_size"] == 1024
    assert first["num_merges"] == second["num_merges"] == 1024 - 260
    assert first["artifacts"]["config.json"] == second["artifacts"]["config.json"]
    assert first["artifacts"]["merges.txt"] == second["artifacts"]["merges.txt"]

    from tokenizer.tokenizer import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer.load(tmp_path / "tok_a")
    assert tok.vocab_size == 1024
    text = "tokenizer training corpus english language model"
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode(text) == ByteLevelBPETokenizer.load(tmp_path / "tok_b").encode(text)
