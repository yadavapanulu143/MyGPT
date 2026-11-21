"""Stage 2: production tokenizer (32,768 entries, trained) validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from model import ModelConfig
from tokenizer.tokenizer import (
    BOS,
    EOS,
    PAD,
    UNK,
    ByteLevelBPETokenizer,
    load_trained_tokenizer,
    pretokenize,
)

REPO = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = REPO / "tokenizer"
VOCAB_SIZE = 32768

SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def train_step(model, batch):\n    loss = model(**batch).loss\n    return loss.item()",
    "Line one\r\nLine two\ttabbed\n\nblank line above",
    "Unicode: café, naïve, 日本語, Ελληνικά, русский, العربية, עברית",
    "Emoji: 🚀🧠✨🙂 and math ∑∫√ ≈ 3.14159",
    "Mixed 12345 numbers, 0x1F, 1e-9, 3.4e38, 2026-11-21T10:19:00Z",
    "https://example.com/docs?page=2&q=bpe#results or support@example.co.uk",
    "Dr. Smith (e.g. the CEO) met Mr. Jones at 5 p.m. on Jan. 3rd, i.e. yesterday.",
    "   leading and trailing spaces   ",
    "",
]


@pytest.fixture(scope="module")
def tok() -> ByteLevelBPETokenizer:
    return load_trained_tokenizer(TOKENIZER_DIR)


# 1. vocabulary size
def test_vocabulary_size_is_exactly_32768(tok):
    assert tok.vocab_size == VOCAB_SIZE
    assert tok.vocab_size_target == VOCAB_SIZE
    assert len(tok.gguf_tokens()) == VOCAB_SIZE
    assert len(tok.merges) == VOCAB_SIZE - 4 - 256


# 2. ids valid and unique
def test_all_ids_unique_and_valid(tok):
    assert len(set(tok.id_to_bytes)) == VOCAB_SIZE
    for token_id in range(VOCAB_SIZE):
        assert isinstance(tok.token_bytes(token_id), bytes)
        assert len(tok.token_bytes(token_id)) >= 1


# 3. no placeholders
def test_no_placeholder_entries(tok):
    tokens = tok.gguf_tokens()
    assert not [t for t in tokens if t.startswith("<unused_")]
    assert all(tok.token_bytes(i) for i in range(VOCAB_SIZE))
    # every merge must resolve to an existing pair of tokens
    for left, right in tok.merges:
        assert tok.token_bytes(left) + tok.token_bytes(right) in tok.bytes_to_id


# 4. special tokens
def test_special_token_ids_documented(tok):
    assert tok.special_tokens == {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    mapping = json.loads((TOKENIZER_DIR / "special_tokens_map.json").read_text())
    assert mapping["special_tokens"] == {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    assert mapping["vocab_size"] == VOCAB_SIZE
    assert mapping["byte_offset"] == 4
    assert mapping["first_merge_id"] == 260

    cfg = ModelConfig.load(REPO / "config" / "model_config.json")
    assert (cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id, cfg.unk_token_id) == (0, 1, 2, 3)
    assert cfg.vocab_size == VOCAB_SIZE

    assert tok.token_text(0) == PAD
    assert tok.token_text(1) == BOS
    assert tok.token_text(2) == EOS
    assert tok.token_text(3) == UNK
    assert tok.encode("hi", add_bos=True, add_eos=True) == [1, *tok.encode("hi"), 2]


# 5. round trips
@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.decode_bytes(ids) == text.encode("utf-8")


# 6. unicode
def test_unicode_specifics(tok):
    for text in [
        "combining: e\u0301 a\u0308 n\u0303",
        "cjk: 漢字 かな カナ 한국어",
        "rtl: مرحبا שלום",
        "currency: $ € £ ¥ ₹ ₽",
        "math: ∀x∈ℝ: x² ≥ 0 ∑ ∞ √ ∂",
        "zero width: a\u200bb\u200dc",
    ]:
        assert tok.decode(tok.encode(text)) == text


# 7. empty input
def test_empty_input(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode_bytes([]) == b""


# 8. long text
def test_long_text(tok):
    text = ("Paragraph one has ordinary English sentences. " * 60 + "\n\n") * 60
    assert len(text.encode()) > 100_000
    ids = tok.encode(text)
    assert len(ids) < len(text.encode())
    assert tok.decode(ids) == text


# 9. numbers and punctuation
def test_numbers_and_punctuation(tok):
    text = "1,234.567 -89 +0.5% $1,000 (a) [b] {c} <d> 'e' \"f\" — – … ;:!?.,"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("1234567890")) < 10


# 10. code
def test_code_snippets(tok):
    code = (
        "import torch\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, d=1024):\n"
        "        super().__init__()\n"
        "        self.w = nn.Linear(d, d, bias=False)\n"
        "    def forward(self, x):  # shape (B, T, D)\n"
        "        return self.w(x) * (x > 0)\n"
    )
    ids = tok.encode(code)
    assert tok.decode(ids) == code
    # indentation and identifiers must not explode into single bytes
    assert len(ids) < len(code.encode()) / 2


# 11. determinism
def test_encoding_is_deterministic(tok):
    text = SAMPLES[3] + SAMPLES[4]
    first = tok.encode(text)
    for _ in range(5):
        assert tok.encode(text) == first
    assert tok.encode_batch([text, text]) == [first, first]


# 12. save/load
def test_save_load_is_identical(tmp_path, tok):
    tok.save(tmp_path)
    reloaded = ByteLevelBPETokenizer.load(tmp_path)
    assert reloaded.merges == tok.merges
    assert reloaded.vocab_size == tok.vocab_size
    for text in SAMPLES:
        assert reloaded.encode(text) == tok.encode(text)
        assert reloaded.decode(reloaded.encode(text)) == text


# 13. GGUF vocabulary reproduces the Python tokenizer
def test_gguf_vocabulary_rebuilds_identically(tok):
    tokens = tok.gguf_tokens()
    id_of = {token: i for i, token in enumerate(tokens)}
    pairs = [tuple(id_of[part] for part in merge.split(" ")) for merge in tok.gguf_merges()]
    rebuilt = ByteLevelBPETokenizer(vocab_size=VOCAB_SIZE, merges=pairs)
    assert rebuilt.vocab_size == VOCAB_SIZE
    for text in SAMPLES + ["Tokenizer parity matters more than size."]:
        assert rebuilt.encode(text) == tok.encode(text)


def test_gguf_token_types(tok):
    types = tok.gguf_token_types()
    assert types[:4] == [3, 3, 3, 2]  # control, control, control, unknown
    assert set(types[4:]) == {1}
    assert 5 not in types  # no unused slots left


# 14. llama.cpp integration (needs a built GGUF and harness; skipped otherwise)
@pytest.mark.skipif(
    not os.environ.get("MYGPT_GGUF"),
    reason="set MYGPT_GGUF=/path/to/model.gguf to run the llama.cpp parity check",
)
def test_llama_cpp_tokenizes_identically(tok):
    harness = os.environ.get("MYGPT_PARITY_HARNESS", "/tmp/llama_parity")
    gguf = os.environ["MYGPT_GGUF"]
    if not shutil.which(harness) and not Path(harness).exists():
        pytest.skip("llama_parity harness not built")
    for text in [SAMPLES[1], SAMPLES[2], SAMPLES[4], SAMPLES[7]]:
        result = subprocess.run(
            [harness, gguf, text, "0", "2048"],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        ids = json.loads(result.stdout.strip().splitlines()[-1])["prompt_ids"]
        assert ids == tok.encode(text)


# reproducibility of the shipped artifacts
def test_manifest_hashes_match_artifacts():
    manifest = json.loads((TOKENIZER_DIR / "tokenizer_manifest.json").read_text())
    assert manifest["vocab_size"] == VOCAB_SIZE
    assert manifest["num_merges"] == VOCAB_SIZE - 260
    for name, expected in manifest["artifacts"].items():
        digest = hashlib.sha256((TOKENIZER_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, f"{name} does not match the manifest"


def test_vocab_and_merges_files_are_consistent(tok):
    vocab = json.loads((TOKENIZER_DIR / "vocab.json").read_text())
    assert len(vocab) == VOCAB_SIZE
    assert vocab[PAD] == 0 and vocab[BOS] == 1 and vocab[EOS] == 2 and vocab[UNK] == 3

    lines = (TOKENIZER_DIR / "merges.txt").read_text().splitlines()
    assert lines[0] == "#version: 0.2"
    assert len(lines) - 1 == len(tok.merges)
    tokens = tok.gguf_tokens()
    for rank, line in enumerate(lines[1:]):
        left, right = line.split(" ")
        assert vocab[left] == tok.merges[rank][0]
        assert vocab[right] == tok.merges[rank][1]
        assert vocab[left + right] == 260 + rank
        assert tokens[260 + rank] == left + right


def test_compression_is_reasonable(tok):
    text = (
        "The history of computing is a history of abstractions: from relays to "
        "transistors to vector units, each layer hid the one below it. "
    ) * 20
    bytes_per_token = len(text.encode()) / len(tok.encode(text))
    assert bytes_per_token > 3.5, f"compression too weak: {bytes_per_token:.2f} bytes/token"


def test_no_unknown_token_emitted(tok):
    text = "§¶†‡•‰′″‴※→←↔⇐⇑⇒⇓"
    ids = tok.encode(text)
    assert tok.special_tokens[UNK] not in ids
    assert tok.decode(ids) == text


def test_pretokenizer_reassembles_for_trained_text(tok):
    for text in SAMPLES:
        assert "".join(pretokenize(text)) == text
