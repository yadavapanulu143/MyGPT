"""Stage 2.1 candidate tokenizers: reserved conversational tokens + code corpus.

These tests pin the properties the candidate relies on. The most important one is
that adding eight specials only rewrites the top eight ids: every byte id and
every merge id below them must stay byte-identical to Stage 2, so the Stage 2
checkpoint's embedding rows remain valid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tokenizer.tokenizer import BYTE_OFFSET, NUM_BYTES, ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "tokenizer"
SPECIALS_ONLY = ROOT / "tokenizer_candidates" / "stage2.1-specials"
CANDIDATE = ROOT / "tokenizer_candidates" / "stage2.1-code"

CONVERSATIONAL = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end_message|>",
    "<|end_turn|>",
    "<|tool_call|>",
    "<|reserved|>",
]

candidates = pytest.mark.parametrize(
    "directory", [SPECIALS_ONLY, CANDIDATE], ids=["specials-only", "code-enriched"]
)


def load(directory: Path) -> ByteLevelBPETokenizer:
    if not (directory / "config.json").exists():
        pytest.skip(f"{directory} not present")
    return ByteLevelBPETokenizer.load(directory)


@candidates
def test_vocabulary_size_is_unchanged(directory: Path) -> None:
    tok = load(directory)
    assert tok.vocab_size == 32768
    assert tok.vocab_size_target == 32768
    assert len(tok.merges) == 32500
    assert tok.merge_limit == 32760


@candidates
def test_special_token_ids(directory: Path) -> None:
    tok = load(directory)
    assert tok.special_tokens["<pad>"] == 0
    assert tok.special_tokens["<bos>"] == 1
    assert tok.special_tokens["<eos>"] == 2
    assert tok.special_tokens["<unk>"] == 3
    for offset, token in enumerate(CONVERSATIONAL):
        assert tok.special_tokens[token] == 32760 + offset, token
    assert sorted(tok.special_tokens.values()) == list(range(4)) + list(range(32760, 32768))


@candidates
def test_byte_block_matches_stage2(directory: Path) -> None:
    stage2 = load(STAGE2)
    tok = load(directory)
    for token_id in range(BYTE_OFFSET, BYTE_OFFSET + NUM_BYTES):
        assert tok.token_bytes(token_id) == stage2.token_bytes(token_id)


def test_specials_only_keeps_stage2_merge_prefix() -> None:
    stage2 = load(STAGE2)
    tok = load(SPECIALS_ONLY)
    assert tok.merges == stage2.merges[: len(tok.merges)]
    for token_id in range(BYTE_OFFSET + NUM_BYTES, tok.merge_limit):
        assert tok.token_bytes(token_id) == stage2.token_bytes(token_id), token_id


@candidates
def test_no_duplicate_or_unresolved_tokens(directory: Path) -> None:
    tok = load(directory)
    tokens = tok.gguf_tokens()
    assert len(tokens) == 32768
    assert len(set(tokens)) == 32768
    token_set = set(tokens)
    for merge in tok.gguf_merges():
        left, right = merge.split(" ")
        assert left in token_set and right in token_set
    types = tok.gguf_token_types()
    # 12 specials: <unk> is UNKNOWN (2), the other 11 are CONTROL (3)
    assert sum(1 for t in types if t == 3) == 11
    assert sum(1 for t in types if t == 2) == 1
    assert sum(1 for t in types if t == 5) == 0


@candidates
def test_round_trip_is_lossless(directory: Path) -> None:
    tok = load(directory)
    samples = [
        "",
        "Hello, world!",
        "Café naïve — “quoted” 🙂 日本語",
        "def f(x):\n    return x ** 2\n",
        "SELECT * FROM t WHERE id = 1;",
        "a\r\nb\r\nc",
        "<|user|> is ordinary text when parse_special is off",
    ]
    for text in samples:
        assert tok.decode(tok.encode(text)) == text


@candidates
def test_parse_special_is_opt_in(directory: Path) -> None:
    tok = load(directory)
    text = "<|user|>hi<|end_message|>"
    plain = tok.encode(text)
    parsed = tok.encode(text, parse_special=True)
    assert not any(i >= 32760 for i in plain)
    assert parsed[0] == tok.special_tokens["<|user|>"]
    assert parsed[-1] == tok.special_tokens["<|end_message|>"]


@candidates
def test_special_tokens_are_skipped_when_decoding(directory: Path) -> None:
    tok = load(directory)
    ids = tok.encode("hi", parse_special=True) + [tok.special_tokens["<|end_turn|>"]]
    assert tok.decode(ids) == "hi"
    assert "<|end_turn|>" in tok.decode(ids, skip_special_tokens=False)


@candidates
def test_manifest_hashes_match_artifacts(directory: Path) -> None:
    manifest = json.loads((directory / "tokenizer_manifest.json").read_text())
    assert manifest["vocab_size"] == 32768
    assert manifest["num_merges"] == 32500
    for name, expected in manifest["artifacts"].items():
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert digest == expected, name


@candidates
def test_training_metadata_records_corpus(directory: Path) -> None:
    config = json.loads((directory / "config.json").read_text())
    training = config["training"]
    assert training["vocab_size"] == 32768
    assert training["num_merges"] == 32500
    assert training["corpus_bytes"] > 0
    assert "highest ids conversational specials" in config["special_token_layout"]


def test_code_candidate_improves_code_compression() -> None:
    stage2 = load(STAGE2)
    candidate = load(CANDIDATE)
    samples = [
        "public class Main { public static void main(String[] args) { System.out.println(42); } }",
        "import java.util.concurrent.ConcurrentHashMap;\nprivate static final long serialVersionUID = 1L;",
        "def __init__(self, name: str) -> None:\n    self.name = name",
        "const { data } = await axios.get('/api/v1/items');",
    ]
    before = sum(len(stage2.encode(s)) for s in samples)
    after = sum(len(candidate.encode(s)) for s in samples)
    assert after < before, f"code tokens did not improve: {before} -> {after}"


def test_code_candidate_keeps_english_cost_bounded() -> None:
    stage2 = load(STAGE2)
    candidate = load(CANDIDATE)
    text = (
        "The committee reviewed the proposal and concluded that the evidence was "
        "insufficient to justify a change in policy, although several members "
        "expressed reservations about the timeline.\n"
    ) * 40
    before = len(stage2.encode(text))
    after = len(candidate.encode(text))
    assert abs(after - before) / before < 0.02, f"english cost {before} -> {after} exceeds 2%"


def test_code_candidate_ships_chat_template() -> None:
    candidate = load(CANDIDATE)
    assert candidate.chat_template is not None
    assert "<|assistant|>" in candidate.chat_template
    assert load(SPECIALS_ONLY).chat_template is None
