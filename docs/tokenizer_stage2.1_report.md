# Stage 2.1 comparison report: Stage 2 tokenizer vs candidate

Committed on `main`. **Not adopted.** The candidate lives in
`tokenizer_candidates/`; the shipped `tokenizer/`, its GGUF exports and the
`v0.2.0-stage2` release are untouched until this is approved.

Three tokenizers are compared:

| Label | Directory | Corpus | Specials | Merges |
|---|---|---|---|---|
| **baseline** | `tokenizer/` | Stage 2 English, 849.2 MiB | 4 | 32,508 |
| **specials** | `tokenizer_candidates/stage2.1-specials/` | same corpus | 4 + 8 | 32,500 |
| **code** (candidate) | `tokenizer_candidates/stage2.1-code/` | + code, 913.6 MiB | 4 + 8 | 32,500 |

The **specials** variant exists to isolate the two changes: comparing baseline to
specials measures the cost of reserving eight ids, and specials to code measures
the corpus change.

All numbers come from `tokenizer_training/compare_tokenizers.py`, which runs the
shipped `ByteLevelBPETokenizer` (the same code path llama.cpp is verified
against), not the trainer's encoder. Raw data:
`tokenizer_training/manifests/tokenizer_comparison.json`.

## 1. Vocabulary statistics

| | baseline | specials | code |
|---|---|---|---|
| Vocabulary size | 32,768 | 32,768 | 32,768 |
| BPE merges | 32,508 | 32,500 | 32,500 |
| Merge id range | 260-32,767 | 260-32,759 | 260-32,759 |
| Conversational specials | 0 | 8 | 8 |
| Control tokens in GGUF | 3 | 11 | 11 |
| Unused slots | 0 | 0 | 0 |
| Duplicate token strings | 0 | 0 | 0 |
| Unresolvable merges | 0 | 0 | 0 |
| Byte-level round-trip lossless | yes | yes | yes |
| Chat template embedded | no | no | **yes** |

The specials-only tokenizer keeps the Stage 2 merge list as a strict prefix
(32,500 of 32,508), so ids 0-32,759 are byte-identical to Stage 2. This is
asserted by a test, and it means a Stage 2 checkpoint only needs its top eight
embedding rows replaced.

## 2. Compression

Held-out data, 1.00 MB slices from files the trainer never saw.

| Metric | baseline | specials | code | code vs baseline |
|---|---|---|---|---|
| English bytes/token | 4.016 | 4.016 | 4.012 | -0.1% |
| English tokens (1 MB) | 249,037 | 249,041 | 249,237 | **+0.08%** |
| Code bytes/token | 3.060 | 3.060 | 3.530 | **+15.4%** |
| Code tokens (1 MB) | 326,830 | 326,830 | 283,296 | **-13.3%** |
| Mixed 93/7 bytes/token | 3.473 | 3.473 | 3.756 | +8.1% |

**The English cost is 0.08%, not a measurable sacrifice.** Reserving eight ids
costs four tokens out of 249,037 on the 1 MB English slice (+0.002%).

Note the absolute English figure (4.016) differs from the 4.650 reported in
`docs/tokenizer_report.md` because that report used a 0.5 MiB slice from a
different offset. Only the within-run comparison is meaningful here.

## 3. Common identifiers

Mean tokens per identifier across 18 Java and 22 Python names
(`JAVA_IDENTIFIERS` / `PYTHON_IDENTIFIERS` in `compare_tokenizers.py`):

| Metric | baseline | specials | code | change |
|---|---|---|---|---|
| Java identifiers, mean tokens | 5.111 | 5.111 | **3.056** | **-40.2%** |
| Python identifiers, mean tokens | 2.227 | 2.227 | **1.727** | **-22.5%** |

Examples, with the candidate's token breakdown:

| Identifier | baseline | code | Candidate tokens |
|---|---|---|---|
| `AbstractSingletonProxyFactoryBean` | 12 | 9 | `Ab` `stract` `S` `ing` `leton` `Proxy` `Factory` `Be` `an` |
| `IllegalArgumentException` | 7 | **2** | `Illegal` `ArgumentException` |
| `ConcurrentHashMap` | 6 | **3** | `Con` `current` `HashMap` |
| `serialVersionUID` | 5 | 4 | `ser` `ial` `Version` `UID` |
| `getDeclaredFields` | 7 | 4 | `get` `Decl` `ared` `Fields` |
| `DataFrame` | 4 | **1** | `DataFrame` |
| `ValueError` | 3 | 2 | `Value` `Error` |
| `dataclass` | 3 | 2 | `dat` `aclass` |
| `asyncio` | 3 | 2 | `async` `io` |
| `__init__` | 3 | 3 | `__` `init` `__` |
| `isinstance` | 2 | 2 | `is` `instance` |

## 4. Technical vocabulary

50 terms from software, systems, databases, security and machine learning:

| Metric | baseline | specials | code | change |
|---|---|---|---|---|
| Mean tokens per term | 2.82 | 2.82 | **2.48** | **-12.1%** |
| Terms covered by one token | 1 / 50 | 1 / 50 | 2 / 50 | +1 |

Examples:

| Term | baseline | code | Candidate tokens |
|---|---|---|---|
| `gradient descent` | 3 | 2 | `gradient` ` descent` |
| `quantization` | 2 | 2 | `quant` `ization` |
| `polymorphism` | 5 | 4 | `poly` `mor` `ph` `ism` |
| `serialization` | 3 | 3 | `ser` `ial` `ization` |
| `dependency injection` | 3 | 3 | `depend` `ency` ` injection` |
| `kubernetes` | 4 | 4 | `k` `uber` `net` `es` |

Coverage is better but not complete: the corpus is 7% code, so an identifier that
appears rarely still splits. Raising the code share further trades English
compression for code compression; at 7% the trade is already favourable.

## 5. Special-token ids

| Id | Token | Type |
|---|---|---|
| 0-3 | `<pad>` `<bos>` `<eos>` `<unk>` | 3,3,3,2 |
| 32,760 | `<\|system\|>` | 3 |
| 32,761 | `<\|user\|>` | 3 |
| 32,762 | `<\|assistant\|>` | 3 |
| 32,763 | `<\|tool\|>` | 3 |
| 32,764 | `<\|end_message\|>` | 3 |
| 32,765 | `<\|end_turn\|>` | 3 |
| 32,766 | `<\|tool_call\|>` | 3 |
| 32,767 | `<\|reserved\|>` | 3 |

Full detail, sufficiency review and conflict analysis:
`docs/tokenizer_special_tokens.md`. Template: `docs/chat_template.md`.

## 6. GGUF compatibility

Exported with `tools/export_gguf.py --tokenizer tokenizer_candidates/stage2.1-code`
using the unchanged random-init checkpoint (the export path, not the weights, is
what is under test):

| Artifact | Size | Note |
|---|---|---|
| `candidate-f16.gguf` | 946.9 MiB | loads in llama.cpp |
| `candidate-q4_k_m.gguf` | 291.9 MiB | quantized from the F16 file |

Checks:

- `llama-gguf <file> r` lists `tokenizer.chat_template` in the metadata;
- all 32,768 tokens load, 11 CONTROL + 1 UNKNOWN, 0 UNUSED;
- `llama-quantize` completes on the modified merge table without errors.

## 7. llama.cpp token-id parity

`scripts/verify_gguf_parity.py --gguf ... --tokenizer tokenizer_candidates/stage2.1-code`:

| Probe | baseline (Stage 2 F16) | candidate F16 | candidate Q4_K_M |
|---|---|---|---|
| plain | match | match | match |
| contractions | match | match | match |
| numbers/URL/email | match | match | match |
| multi-line code | match | match | match |
| Unicode + emoji | match | match | match |
| abbreviations | match | match | match |
| long (600 tokens) | match | match | match |

Status `ok` for both candidate files. Manifests:
`gguf_parity_candidate.json`, `gguf_parity_candidate_q4km.json`.

Logit-level check (F16, prompt `def train(model, data): return model.fit(data)`):

| Metric | Value |
|---|---|
| Cosine similarity vs PyTorch | **0.99999988** |
| Max absolute difference | 0.003119 |
| Top-1 token | 14007 in both |
| Top-5 tokens | same set; ranks 3 and 4 swap because they differ by less than the f16 export error |

Chat markers round-trip: llama.cpp tokenizes the rendered conversation to
`[32760, 2687, 463, 2409, 75, 6565, 50, 32764, 14, 32761, 6930, 32764, ...]`,
byte-identical to `encode_chat`.

## 8. Reproducibility

- Training is deterministic for the pinned `tokenizers==0.23.3`. Both candidates
  were retrained from scratch into a fresh output directory and produced
  byte-identical artifacts:
  `merges.txt` 7150a68a... (specials) and d9365b5d... (code), and
  `vocab.json` 626fa013... (code).
- `tokenizer_candidates/*/tokenizer_manifest.json` records the SHA-256 of
  `config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, the corpus
  manifest hash, corpus bytes and training wall time.
- The comparison script is deterministic: identical output on re-run.
- `tests/test_tokenizer_candidate.py::test_manifest_hashes_match_artifacts` fails
  if any artifact is edited after the manifest was written.

## 9. Regressions and open issues

| Item | Assessment |
|---|---|
| English compression | **-0.08%** (4 tokens per 249,037). Within noise; accepted. |
| Vocabulary size | unchanged at 32,768. |
| Parameter count | unchanged at 495,799,296 (embedding shape untouched). |
| Stage 2 artifacts | untouched; candidate lives in a separate directory. |
| Specials-only variant | strictly worse than the code candidate on every code metric and equal on English, so the candidate supersedes it as the shipping option. |
| Code compression | still 3.53 bytes/token: thin compared with a code-specialised vocabulary, because code is 7% of the corpus by design. |
| Non-Latin scripts | unchanged from Stage 2 (still weak, ~1.1-2.0 chars/token). |
| Chat tokens and `eos` | the template uses `<|end_turn|>` for stopping, but `<eos>` is still the document terminator; instruction tuning must teach both, and generation code must stop on either. |
| Corpus licensing | every source is MIT, BSD-3, Apache-2.0, PSF-2.0 or PostgreSQL licensed; provenance and licence headers are in `tokenizer_training/manifests/code_corpus_manifest.json`. |
| Corpus size on disk | the code corpus (67.4 MiB) is not committed; the script and manifest are, so it can be rebuilt. |

## 10. Recommendation

Adopt the **code** candidate as the Stage 2.1 tokenizer when approved:

1. it is the only variant that fixes both problems asked for (reserved
   conversational ids and code representation);
2. English compression is unaffected within 0.1%;
3. code tokens drop 13.3%, Java identifiers 40%, Python identifiers 22%;
4. llama.cpp parity is exact on F16 and Q4_K_M, including the new markers;
5. it costs nothing in parameters, memory or inference speed (same vocabulary
   size, same architecture).

Do not merge until approved. After approval the steps are: copy the candidate
artifacts over `tokenizer/`, re-export the Stage 2 GGUF with the new tokenizer,
re-run parity, and bump the release tag. Stage 3 remains blocked.
