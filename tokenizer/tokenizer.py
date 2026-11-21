"""Independent byte-level BPE tokenizer (no pre-trained vocabulary, no downloads).

Design
------
* Byte-level alphabet: every possible byte has an id, so any UTF-8 (or even
  arbitrary binary) input round-trips losslessly. This is the same trick GPT-2
  and Llama-3 use; it removes the out-of-vocabulary problem entirely.
* Id layout (fixed, documented in ``tokenizer/config.json``)::

      0        <pad>
      1        <bos>
      2        <eos>
      3        <unk>
      4..259   raw byte 0x00 .. 0xFF
      260..    BPE merges produced by ``train()`` (empty at this stage)

* ``vocab_size = 32768`` matches the model's embedding matrix. The shipped
  vocabulary in ``tokenizer/config.json`` is trained (4 specials + 256 bytes +
  32508 merges); see ``tokenizer_training/`` for the pipeline that produced it.
  A freshly constructed instance with no merges still reports ``<unused_N>``
  placeholders so the GGUF export stays valid before training.
* The pre-tokenizer regex mirrors the GPT-2 pattern used by llama.cpp, so a
  tokenizer trained here exports to GGUF without behavioural drift.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
BYTE_OFFSET = 4
NUM_BYTES = 256
DEFAULT_VOCAB_SIZE = 32768

# GPT-2 pre-tokenization, expressed with Python's Unicode-aware classes:
#   \p{L} ~ [^\W\d_]      \p{N} ~ \d
_PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE,
)


def pretokenize(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-codepoint map (for GGUF export)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteLevelBPETokenizer:
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        merges: Iterable[tuple[int, int]] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        if vocab_size < BYTE_OFFSET + NUM_BYTES:
            raise ValueError(f"vocab_size must be >= {BYTE_OFFSET + NUM_BYTES}")
        self.vocab_size_target = vocab_size
        self.special_tokens = dict(special_tokens or SPECIAL_TOKENS)
        self.special_ids = {v: k for k, v in self.special_tokens.items()}

        self.id_to_bytes: list[bytes] = []
        self.bytes_to_id: dict[bytes, int] = {}
        self._init_base_vocab()

        self.merges: list[tuple[int, int]] = []
        self.ranks: dict[tuple[int, int], int] = {}
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self._add_merge(tuple(pair))

    # ------------------------------------------------------------- building
    def _init_base_vocab(self) -> None:
        for token_id in range(BYTE_OFFSET):
            token = self.special_ids.get(token_id, f"<unused_{token_id}>")
            self.id_to_bytes.append(token.encode("utf-8"))
        for b in range(NUM_BYTES):
            self.id_to_bytes.append(bytes([b]))
            self.bytes_to_id[bytes([b])] = BYTE_OFFSET + b

    def _add_merge(self, pair: tuple[int, int]) -> int:
        if pair in self.ranks:
            return self.ranks[pair]
        new_id = len(self.id_to_bytes)
        merged = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
        if merged in self.bytes_to_id:  # identical byte string already present
            return self.bytes_to_id[merged]
        self.id_to_bytes.append(merged)
        self.bytes_to_id[merged] = new_id
        self.ranks[pair] = len(self.merges)
        self.merges.append(pair)
        self._cache.clear()
        return new_id

    # ----------------------------------------------------------- properties
    def __len__(self) -> int:
        return len(self.id_to_bytes)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    def token_bytes(self, token_id: int) -> bytes:
        return self.id_to_bytes[token_id]

    def token_text(self, token_id: int) -> str:
        if token_id in self.special_ids:
            return self.special_ids[token_id]
        return self.id_to_bytes[token_id].decode("utf-8", errors="replace")

    def get_vocab(self) -> dict[bytes, int]:
        return dict(self.bytes_to_id)

    # --------------------------------------------------------------- encode
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        ids = [self.bytes_to_id[bytes([b])] for b in pretoken.encode("utf-8")]
        if self.ranks and len(ids) > 1:
            while True:
                best_rank = None
                best_idx = -1
                for i in range(len(ids) - 1):
                    rank = self.ranks.get((ids[i], ids[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank, best_idx = rank, i
                if best_rank is None:
                    break
                merged = self._add_merge_lookup((ids[best_idx], ids[best_idx + 1]))
                ids[best_idx : best_idx + 2] = [merged]
        self._cache[pretoken] = ids
        return ids

    def _add_merge_lookup(self, pair: tuple[int, int]) -> int:
        merged = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
        token_id = self.bytes_to_id.get(merged)
        if token_id is None:  # pragma: no cover - defensive
            token_id = self._add_merge(pair)
        return token_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.special_tokens[BOS])
        for pretoken in pretokenize(text):
            ids.extend(self._encode_pretoken(pretoken))
        if add_eos:
            ids.append(self.special_tokens[EOS])
        return ids

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    # --------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True, strict: bool = False) -> str:
        return self.decode_bytes(ids, skip_special_tokens=skip_special_tokens, strict=strict).decode(
            "utf-8", errors="replace"
        )

    def decode_bytes(self, ids: Iterable[int], skip_special_tokens: bool = True, strict: bool = False) -> bytes:
        chunks = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.id_to_bytes):
                # Ids above the trained vocabulary are placeholders until the
                # tokenizer is trained; an untrained model samples them freely.
                if strict:
                    raise IndexError(f"token id {token_id} out of range (vocab={len(self.id_to_bytes)})")
                continue
            if skip_special_tokens and token_id in self.special_ids:
                continue
            chunks.append(self.id_to_bytes[token_id])
        return b"".join(chunks)

    # ---------------------------------------------------------------- train
    def train(
        self,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "ByteLevelBPETokenizer":
        """Learn BPE merges from a corpus of raw strings.

        Not executed during this stage (no dataset is used yet). It is provided
        so the future training pipeline can build the real 32k vocabulary with
        the same code that produced the model config.
        """
        target = vocab_size or self.vocab_size_target
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(pretokenize(text))

        words: list[list[int]] = [
            [self.bytes_to_id[bytes([b])] for b in word.encode("utf-8")] for word in word_counts
        ]
        counts = list(word_counts.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}
        for idx, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += counts[idx]
                pair_to_words.setdefault(pair, set()).add(idx)

        while len(self.id_to_bytes) < target:
            if not pair_counts:
                break
            # deterministic tie-break: highest frequency, then smallest pair
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            pair, freq = best
            if freq < min_frequency:
                break

            self._add_merge(pair)
            for idx in list(pair_to_words.pop(pair, ())):
                ids = words[idx]
                i = 0
                while i < len(ids) - 1:
                    if (ids[i], ids[i + 1]) == pair:
                        for p in ((ids[i - 1], ids[i]) if i > 0 else None, (ids[i], ids[i + 1]), (ids[i + 1], ids[i + 2]) if i + 2 < len(ids) else None):
                            if p is not None:
                                pair_counts[p] -= counts[idx]
                                if pair_counts[p] <= 0:
                                    pair_counts.pop(p, None)
                                if p in pair_to_words:
                                    pair_to_words[p].discard(idx)
                        ids[i : i + 2] = [self.bytes_to_id[self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]]]
                        for p in ((ids[i - 1], ids[i]) if i > 0 else None, (ids[i], ids[i + 1]) if i + 1 < len(ids) else None):
                            if p is not None:
                                pair_counts[p] += counts[idx]
                                pair_to_words.setdefault(p, set()).add(idx)
                    else:
                        i += 1
            if verbose and len(self.merges) % 1000 == 0:
                print(f"merges={len(self.merges)} vocab={len(self.id_to_bytes)}")

        self._cache.clear()
        return self

    # ------------------------------------------------------------- gguf view
    def gguf_tokens(self, unicode_encode: bool = True) -> list[str]:
        """Vocabulary in id order, in GPT-2 byte-unicode form for llama.cpp."""
        enc = bytes_to_unicode() if unicode_encode else None
        tokens: list[str] = []
        for token_id in range(self.vocab_size_target):
            if token_id >= len(self.id_to_bytes):
                tokens.append(f"<unused_{token_id}>")
                continue
            raw = self.id_to_bytes[token_id]
            if token_id in self.special_ids:
                tokens.append(self.special_ids[token_id])
            elif enc is not None:
                tokens.append("".join(enc[b] for b in raw))
            else:  # pragma: no cover
                tokens.append(raw.decode("utf-8", errors="replace"))
        return tokens

    def gguf_merges(self) -> list[str]:
        """Merges as ``"left right"`` token strings, the form GGUF/llama.cpp expect."""
        tokens = self.gguf_tokens()
        return [f"{tokens[a]} {tokens[b]}" for a, b in self.merges]

    def gguf_token_types(self) -> list[int]:
        """llama.cpp token type ids: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 5=UNUSED."""
        types = []
        for token_id in range(self.vocab_size_target):
            if token_id >= len(self.id_to_bytes):
                types.append(5)
            elif token_id == self.special_tokens[UNK]:
                types.append(2)
            elif token_id in self.special_ids:
                types.append(3)
            else:
                types.append(1)
        return types

    # ---------------------------------------------------------------- (de)ser
    def to_dict(self) -> dict:
        return {
            "tokenizer_class": "ByteLevelBPETokenizer",
            "description": "Independent byte-level BPE for English, GPT-2 compatible.",
            "vocab_size": self.vocab_size_target,
            "current_vocab_size": len(self.id_to_bytes),
            "special_tokens": self.special_tokens,
            "byte_offset": BYTE_OFFSET,
            "num_byte_tokens": NUM_BYTES,
            "pre_tokenizer": "gpt2",
            "byte_level": True,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, directory: str | Path, filename: str = "config.json") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "ByteLevelBPETokenizer":
        return cls(
            vocab_size=data.get("vocab_size", DEFAULT_VOCAB_SIZE),
            merges=[tuple(p) for p in data.get("merges", [])],
            special_tokens=data.get("special_tokens"),
        )

    @classmethod
    def load(cls, directory: str | Path, filename: str = "config.json") -> "ByteLevelBPETokenizer":
        return cls.from_dict(json.loads((Path(directory) / filename).read_text(encoding="utf-8")))


TOKENIZER_DIR = Path(__file__).resolve().parent


def build_default_tokenizer(vocab_size: int = DEFAULT_VOCAB_SIZE) -> ByteLevelBPETokenizer:
    """Untrained foundation (bytes only): used for algorithm tests and before training."""
    return ByteLevelBPETokenizer(vocab_size=vocab_size)


def load_trained_tokenizer(directory: str | Path = TOKENIZER_DIR) -> ByteLevelBPETokenizer:
    """The production tokenizer shipped with the repository."""
    return ByteLevelBPETokenizer.load(directory)


if __name__ == "__main__":  # pragma: no cover
    tok = load_trained_tokenizer()
    sample = "Hello, world! The quick brown fox jumps over the lazy dog."
    ids = tok.encode(sample)
    print("vocab:", tok.vocab_size, "of", tok.vocab_size_target)
    print("tokens:", len(ids), ids[:20], "...")
    print("roundtrip ok:", tok.decode(ids) == sample)
