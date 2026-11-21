# Tokenizer evaluation report (Stage 2)

All numbers below were measured in the development environment (4 vCPU EPYC,
no GPU, Debian 13, Python 3.13). Held-out text is
`/workspace/corpus/processed/eval/heldout.txt`, 9.59 MiB / 2,289 documents that
were never seen during tokenizer training.

## 1. Vocabulary integrity

| Check | Result |
|---|---|
| Vocabulary size | 32,768 (target 32,768) |
| Merges | 32,508 |
| Distinct byte strings | 32,768 / 32,768 |
| Placeholder (`<unused_N>`) entries | 0 |
| Duplicate ids | 0 |
| Special tokens | `<pad>` 0, `<bos>` 1, `<eos>` 2, `<unk>` 3 |
| `<unk>` emitted on any test input | never |

## 2. Held-out compression

| Metric | Value |
|---|---|
| Sample | 524,405 bytes of held-out text |
| Tokens | 112,785 |
| **Bytes per token** | **4.650** |
| Tokens per KiB | 220.2 |
| Round-trip exact | yes |
| Encode speed (pure Python) | 0.24 s for 0.5 MiB |

For reference, the Stage 1 placeholder tokenizer produced 1 byte/token on the
same text, and GPT-2's 50k vocabulary is usually quoted at ~4.0–4.3 bytes/token
on English web text. 4.65 bytes/token with half the vocabulary means the merge
list is spending its budget on English words rather than on rare Unicode.

Corpus-size effect (same held-out file):

| Training text | Bytes/token | Relative |
|---|---|---|
| 24.3 MiB | 4.458 | baseline |
| 80.0 MiB | 4.535 | +1.7% |
| 849.2 MiB | 4.650 | +4.3% |

## 3. Category behaviour

| Category | Probe | Chars | Tokens | Chars/token | Round-trip |
|---|---|---:|---:|---:|---|
| Plain English | "The quick brown fox jumps over the lazy dog, and then it rests." | 63 | 15 | 4.20 | exact |
| Contractions | "I don't think we'll ever see it's done, but they're sure you'd know." | 68 | 20 | 3.40 | exact |
| Numbers/math | "The result is 3.14159 * (2 + 7) = 28.27431, roughly 1e-3 off at t=0.95." | 71 | 34 | 2.09 | exact |
| Code | 4-line Python function with a comment | 155 | 50 | 3.10 | exact |
| URL/email | "Mail support@example.co.uk or see https://example.com/docs?page=2&q=bpe#results" | 79 | 31 | 2.55 | exact |
| Abbreviations | "Dr. Smith (e.g. the CEO) met Mr. Jones at 5 p.m. on Jan. 3rd, i.e. yesterday." | 77 | 33 | 2.33 | exact |
| Unicode | "Café naïve — “quoted” text… 日本語のテキスト, Ελληνικά, emoji 🙂🚀" | 63 | 57 | 1.11 | exact |
| Whitespace | "line one\nline two\n\n\tindented line\r\nwindows newline" | 50 | 15 | 3.33 | exact |
| Technical | "Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` then run main.py." | 90 | 33 | 2.73 | exact |

### Example tokenizations

```
"The quick brown fox jumps over the lazy dog."
  The | Ġquick | Ġbrown | Ġfox | Ġjumps | Ġover | Ġthe | Ġlazy | Ġdog | .
  10 tokens / 44 bytes = 4.40 bytes/token

"I don't think we'll ever see it's done."
  I | Ġdon | 't | Ġthink | Ġwe | 'll | Ġever | Ġsee | Ġit | 's | Ġdone | .
  12 tokens / 39 bytes = 3.25 bytes/token

"def train_step(model, batch):\n    return model(batch).loss.item()"
  def | Ġtrain | _ | step | ( | mod | el | , | Ġbatch | ): | ĊĠĠĠ | Ġreturn | Ġmodel |
  ( | b | atch | ). | loss | . | item | ()
  21 tokens / 65 bytes = 3.10 bytes/token

"https://example.com/docs?page=2&q=bpe"
  https | :// | example | . | com | / | doc | s | ? | page | = | 2 | & | q | = | b | pe
  17 tokens / 37 bytes = 2.18 bytes/token

"Café naïve — “quoted” text… 日本語 🙂"
  C | af | Ã© | Ġna | Ã¯ | ve | ĠâĢĶ | ĠâĢľ | qu | oted | âĢĿ | Ġtext | âĢ | ¦ | Ġæ | Ĺ | ¥ | ...
  27 tokens / 52 bytes = 1.93 bytes/token
```

(`Ġ` marks a leading space, `Ċ` a newline — the GPT-2 byte-unicode convention.
The `Ã©` style tokens are the byte-level encoding of `é`, which is why Unicode
costs ~2 bytes per character here.)

## 4. Determinism and persistence

| Check | Result |
|---|---|
| Same input encoded 5× | identical ids |
| Save → load → encode | identical ids for all samples |
| Two training runs on the same corpus | byte-identical `config.json` and `merges.txt` |
| Artifacts vs manifest SHA-256 | match |

## 5. Parity against the trainer (Hugging Face `tokenizers`)

| Metric | Value |
|---|---|
| Held-out sample | 105,080 bytes |
| HF tokens / our tokens | 22,800 / 22,800 |
| Mismatched token positions | **0** |
| Match rate | 100.000% |

The HF trainer numbers its 256 base tokens in codepoint order while this project
numbers them in byte order, so the comparison is on token *strings*, not ids.
Ids 260+ agree exactly (both are assigned in merge order).

## 6. Parity against llama.cpp

`scripts/verify_gguf_parity.py` runs the harness in `tools/llama_parity.cpp`
against the exported GGUF:

| Prompt | Tokens | llama.cpp ids == Python ids |
|---|---:|---|
| plain | 11 | yes |
| contractions | 20 | yes |
| numbers + URL + email | 42 | yes |
| code (multi-line) | 26 | yes |
| Unicode + emoji | 32 | yes |
| abbreviations | 33 | yes |
| long (600 tokens) | 600 | yes |

Numerical parity of the exported weights (logit cosine vs the PyTorch reference,
three prompts):

| File | Logit cosine | Max abs diff | Top-1 token agrees |
|---|---|---|---|
| F16 | 0.99999881–0.99999935 | 0.0030–0.0045 | 3/3 |
| Q8_0 | 0.99927–0.99942 | 0.090–0.106 | 3/3 |
| Q4_K_M | 0.952–0.961 | 0.752–0.791 | 0/3 |

Greedy decoding from the F16 file reproduces the PyTorch sequence token for
token (8/8). Q4_K_M does not: with randomly initialized weights the top logits
are separated by less than the quantization error, so the argmax moves. This is
a property of untrained weights, not a tokenizer or export defect — the
tokenizer ids are identical for every prompt at every quantization level.

## 7. Test suite

```
python3 -m pytest tests -q
97 passed, 1 skipped in 28.83s
```

The skipped test is the llama.cpp integration check, which needs a built harness:

```
MYGPT_GGUF=dist/mygpt-0.5b-f16.gguf python3 -m pytest tests/test_tokenizer_production.py -q
30 passed
```

Stage 1's 67 tests still pass; the only Stage 1 test that had to change was
`test_no_pretrained_artifacts_used`, which used to forbid any `vocab.json` /
`merges.txt` in the repository. It now verifies those files match the hashes
recorded by our own trainer, which is the property that actually matters.

## 8. Remaining limitations

- Compression is measured on Wikipedia/Gutenberg-heavy held-out text; code and
  conversational text compress less well (3.1 and ~3.4 bytes/token).
- Non-Latin scripts are byte-level (1.1–2.0 chars/token), by design for an
  English-only vocabulary.
- The vocabulary is full at 32,768 ids, so future chat tokens require retraining
  the tokenizer (or reusing merge ids).
- Downstream effect on pretraining loss is unknown until pretraining runs.
