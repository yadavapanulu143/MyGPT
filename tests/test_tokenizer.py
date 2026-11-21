"""9. Tokenizer foundation tests."""

import json

import pytest

from tokenizer.tokenizer import (
    BOS,
    EOS,
    PAD,
    UNK,
    ByteLevelBPETokenizer,
    build_default_tokenizer,
    pretokenize,
)

SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def train(model, data):\n    return model.fit(data)\n",
    "Line one\r\nLine two\ttabbed",
    "Unicode: café, naïve, 日本語, Ελληνικά, русский",
    "Emoji: 🚀🧠✨ and math ∑∫√",
    "Mixed 12345 numbers and 3.14159 floats",
    "   leading and trailing spaces   ",
    "",
]


@pytest.fixture(scope="module")
def tok():
    return build_default_tokenizer()


def test_vocab_layout(tok):
    assert tok.vocab_size_target == 32768
    assert tok.vocab_size == 260  # 4 specials + 256 bytes, no merges yet
    assert tok.special_tokens == {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    assert tok.token_bytes(4) == b"\x00"
    assert tok.token_bytes(259) == b"\xff"
    assert tok.token_bytes(ord("A") + 4) == b"A"


def test_roundtrip_lossless(tok):
    for text in SAMPLES:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"roundtrip failed for {text!r}"
        assert tok.decode_bytes(ids) == text.encode("utf-8")


def test_byte_level_has_no_unknown_tokens(tok):
    ids = tok.encode("any text at all — even symbols like §¶†")
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert UNK not in [tok.token_text(i) for i in ids]


def test_special_tokens(tok):
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tok.decode(ids) == "hi"
    assert tok.decode(ids, skip_special_tokens=False).startswith("<bos>")


def test_pretokenizer_splits_like_gpt2(tok):
    assert pretokenize("Hello, world!") == ["Hello", ",", " world", "!"]
    assert pretokenize("It's 2026") == ["It", "'s", " 2026"]
    # pre-tokens must reassemble to the original string
    for text in SAMPLES:
        assert "".join(pretokenize(text)) == text


def test_empty_and_long_input(tok):
    assert tok.encode("") == []
    long_text = "word " * 500
    assert tok.decode(tok.encode(long_text)) == long_text


def test_save_load(tmp_path):
    tok = build_default_tokenizer()
    path = tok.save(tmp_path)
    assert json.loads(path.read_text())["vocab_size"] == 32768
    loaded = ByteLevelBPETokenizer.load(tmp_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("check 1 2 3") == tok.encode("check 1 2 3")
    assert loaded.special_tokens == tok.special_tokens


def test_project_config_file_is_loadable():
    tok = ByteLevelBPETokenizer.load("tokenizer/config.json".rsplit("/", 1)[0])
    assert tok.vocab_size_target == 32768


def test_training_algorithm_on_synthetic_strings():
    """Exercises the BPE trainer; no dataset is used (tiny synthetic strings)."""
    tok = ByteLevelBPETokenizer(vocab_size=280)
    tok.train(["hello hello hello world", "hello world again"], vocab_size=280, min_frequency=1)
    assert tok.vocab_size > 260, "trainer must add merges"
    ids = tok.encode("hello world")
    assert len(ids) < len("hello world")
    assert tok.decode(ids) == "hello world"


def test_training_is_deterministic():
    corpus = ["the cat sat on the mat", "the dog sat on the log"]
    a = ByteLevelBPETokenizer(vocab_size=275).train(corpus, vocab_size=275, min_frequency=1)
    b = ByteLevelBPETokenizer(vocab_size=275).train(corpus, vocab_size=275, min_frequency=1)
    assert a.merges == b.merges


def test_gguf_vocabulary_view(tok):
    tokens = tok.gguf_tokens()
    types = tok.gguf_token_types()
    assert len(tokens) == len(types) == 32768
    assert tokens[:4] == [PAD, BOS, EOS, UNK]
    assert types[:4] == [3, 3, 3, 2]
    assert tokens[4] == "\u0100"  # GPT-2 byte-unicode for 0x00
    assert tokens[4 + ord("A")] == "A"
    assert tokens[-1].startswith("<unused_")


def test_vocab_size_matches_model_embedding():
    from model import ModelConfig

    cfg = ModelConfig.load("config/model_config.json")
    assert cfg.vocab_size == build_default_tokenizer().vocab_size_target


def test_no_pretrained_artifacts_used():
    """Guards against silently loading a downloaded tokenizer/vocab."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    forbidden = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model", "spiece.model"]
    found = [str(p) for p in repo.rglob("*") if p.name in forbidden]
    assert not found, f"pre-trained tokenizer artifacts found: {found}"
