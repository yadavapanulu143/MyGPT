# AGENTS.md - project memory

Read this first. It records what exists, why, and what is next, so a future
session does not have to rediscover it.

## Project in one paragraph

`MyGPT` is an independently built decoder-only LLM with **495,799,296**
parameters, written from scratch in PyTorch, randomly initialized, verified
structurally, exported to GGUF, quantized to Q4_K_M, and benchmarked. Deployment
target: Windows 11 laptop, AMD Ryzen 3 3250U (2C/4T, AVX2, Vega 3), 8 GB RAM
(~5.94 GB usable), CPU-only, offline. Stage 1 built the architecture and export
path; Stage 2 trained the 32,768-entry tokenizer. No language-model training has
happened.

## Hard constraints (do not violate)

1. No pre-trained weights, tokenizers or configs. Everything is generated here.
2. Do not start pretraining, instruction tuning or RL without explicit approval.
3. The model must remain loadable by stock llama.cpp (GGUF `llama` architecture).
4. Do not claim the model is intelligent, helpful or uncensored: it is untrained.
5. Never invent benchmark numbers. Dev-machine and laptop numbers in the docs
   are measured; anything estimated is labelled as such.
6. No keyword-based censorship anywhere. Behaviour is a configurable prompt
   policy (`config/behaviour_policy.json`), not a filter.
7. Keep the fp32 checkpoint as the source of truth for all quantized artifacts.

## Architecture decisions (locked unless there is a measured reason)

| Decision | Value | Why |
|---|---|---|
| Layers / d_model | 41 / 1024 | 500M budget with short sequential chain for CPU |
| Heads / KV heads | 16 / 4 | GQA cuts KV cache 4x (328 -> 82 MiB at 2048) |
| FFN | 2816 (multiple of 256) | SwiGLU at 2.75x; 256 keeps k-quants fast/possible |
| Vocab | 32768, tied, trained BPE | 6.8% of params; tying removes a second 33.5M matrix |
| Context | 2048 | 82 MiB KV at fp16; enough for a conversation turn |
| Positions | RoPE base 10000 | no extra params, GGUF-native |
| Norm / act | RMSNorm / SwiGLU | llama.cpp-native, no biases |
| Master dtype | fp32 | training-ready; quantize on export |

Exact count: 41 x 11,274,240 + 33,554,432 + 1,024 = 495,799,296.
`tests/test_parameters.py` asserts this number; changing the config breaks tests
on purpose.

## Layout and entry points

| Path | Role |
|---|---|
| `model/` | `MyGPT`, `Attention` (GQA+RoPE), `TransformerBlock`, `RMSNorm`, `SwiGLUMLP`, `io.py`, `policy.py` |
| `tokenizer/tokenizer.py` | `ByteLevelBPETokenizer`: ids 0-3 specials, 4-259 bytes, 260+ merges (32,508 trained) |
| `tokenizer/` artifacts | `config.json` (canonical), `vocab.json`, `merges.txt`, `special_tokens_map.json`, `tokenizer_manifest.json` (hashes) |
| `tokenizer_training/` | `prepare_corpus.py`, `train_tokenizer.py`, `evaluate_tokenizer.py`, `manifests/` |
| `initialization/initialize_weights.py` | seeded init, stats, fingerprint, `init_manifest.json` |
| `tools/gguf_writer.py` | GGUF v3 writer, hand-rolled, no `gguf` package |
| `tools/export_gguf.py` | checkpoint -> GGUF; 1-D tensors stay F32 |
| `tools/llama_parity.cpp` | llama.cpp harness printing prompt/greedy token ids |
| `scripts/` | count_parameters, benchmark_memory, test_inference, quantize_reference, verify_gguf_parity, chat |
| `docs/` | architecture, memory_analysis, training_plan, tokenizer, tokenizer_report, final_report |
| `deployment/` | laptop quick start and plan |

## Commands that matter

```bash
python3 -m pytest tests -q                       # 137 passed, 1 skipped, ~30 s
python3 scripts/count_parameters.py              # 495,799,296 everywhere
python3 initialization/initialize_weights.py --out checkpoints/mygpt-0.5b-init
python3 tools/export_gguf.py --checkpoint checkpoints/mygpt-0.5b-init --output dist/mygpt-0.5b-f16.gguf
llama-quantize dist/mygpt-0.5b-f16.gguf dist/mygpt-0.5b-q4_k_m.gguf Q4_K_M
python3 scripts/verify_gguf_parity.py --gguf dist/mygpt-0.5b-f16.gguf
python3 scripts/verify_gguf_logits.py --gguf dist/mygpt-0.5b-f16.gguf --quant f16
python3 scripts/benchmark_memory.py --contexts 64,256 --dtypes float32,bfloat16

# tokenizer stage (needs requirements-tokenizer.txt and a corpus)
python3 tokenizer_training/prepare_corpus.py --out-dir /workspace/corpus/processed
python3 tokenizer_training/train_tokenizer.py --corpus /workspace/corpus/processed/train --out tokenizer
python3 tokenizer_training/evaluate_tokenizer.py --tokenizer tokenizer \
    --eval-text /workspace/corpus/processed/eval/heldout.txt --json-out /tmp/eval.json
```

llama.cpp build used for verification: `cmake -B build -DCMAKE_BUILD_TYPE=Release
-DLLAMA_CURL=OFF` then targets `llama-cli llama-quantize llama-bench`.

## Gotchas discovered the hard way

- `safetensors` refuses tensors that share storage; `model/io.py` dedupes by
  `untyped_storage().data_ptr()` (tied lm_head/embedding), not by `id()`.
- llama.cpp aborts on mixed f16/f32 element-wise mul, so **norm weights must be
  exported as F32** while matrices are F16.
- `intermediate_size` must be a multiple of 256 or Q4_K quantization fails.
- llama.cpp expects GGUF token strings in GPT-2 "byte-unicode" form; the
  tokenizer exposes `gguf_tokens()`. Merges must be written as `"left right"`
  token strings (`gguf_merges()`), never as id pairs.
- **RoPE layout**: this model uses HF `rotate_half`; ggml uses interleaved
  `rope_norm`. `tools/export_gguf.py` must permute `attn_q`/`attn_k` weights
  (`permute_rope_weight`) or the GGUF still runs but its logits only correlate
  with PyTorch at cosine 0.982. With the permutation: 0.999999 and identical
  greedy output.
- Token ids are numbered in merge order from 260; HF `tokenizers` numbers its 256
  base tokens by codepoint, so compare HF parity on token *strings*, not ids.
- Random-init models generate degenerate repeated tokens; that is expected, not a
  bug. Use the parity harness (token ids), not text, to verify correctness.
- PyTorch is 4-5x slower than llama.cpp on CPU; do not benchmark the laptop path
  with PyTorch.
- Special tokens live at the *top* of the id space (`merge_limit`), never in the
  middle: inserting them earlier would renumber every merge and invalidate the
  checkpoint's embeddings.
- `encode(parse_special=False)` is the default and must stay the default; the
  chat path (`scripts/render_chat.py::encode_chat`) appends marker ids instead of
  re-parsing rendered text, which is what stops prompt injection via user text.
- The chat template exists twice (Jinja file and Python renderer); `tests/
  test_chat_template.py` diffs them, so edit both or the test fails.

## Status (2026-11-21)

Stage 1 done: architecture, init + manifest, GGUF export, 5 quantizations,
parity harness, dev-machine benchmarks, memory profiling, INT8/INT4 study,
deployment plan.

Stage 2 done: 849 MiB five-source corpus (with licences recorded), corpus-size
pilot, trained 32,768-entry BPE (4.650 bytes/token held-out), full artifact set
with hashes, evaluation report, HF and llama.cpp parity, 97 tests. Fixed two
export bugs found on the way: merges written as ids, and missing RoPE
permutation. Laptop generation measured by the owner: 25.26-26.93 tokens/s
(Q4_K_M, `llama-bench`).

Stage 2.1 adopted (2026-11-21, release `v0.3.1-stage2.1`): eight
conversational specials reserved at ids 32,760-32,767 (vocabulary stays 32,768,
merges drop to 32,500, ids 0-32,759 stay byte-identical to Stage 2), a
ChatML-like chat template in `config.json` and GGUF, and a 42.1 MiB
permissively-licensed code corpus added to training. Code tokens -13.3%, Java
identifiers -40%, English cost +0.08%. llama.cpp token-id parity exact on F16 and
Q4_K_M (F16 logit cosine 0.99999988, Q8_0 0.99945, Q4_K_M 0.9545 on random
weights). 137 tests pass, 1 skipped. Stage 2 artifacts frozen in
`tokenizer_candidates/stage2-baseline/`; release `v0.2.0-stage2` untouched.

Not done (needs approval): pretraining, instruction tuning, behaviour tuning,
long-context extension, multilingual tokenizer.

## Next steps, in order

1. Approve the Stage 3 plan (`docs/stage3_plan.md`): corpus, GPU budget, evaluation
   thresholds. Nothing trains before that approval.
2. Pretrain on rented GPUs; monitor validation loss against the plan's gates.
3. Instruction-tune, then behaviour-tune (DPO/ORPO against over-refusal).
4. Evaluate: perplexity, lm-eval-harness, XSTest over-refusal, safety floor.
5. Re-export GGUF, re-quantize, benchmark on the actual laptop with `llama-bench`,
   and re-run the logit gates now that weights are trained.

## Files that must be pushed for persistence

Everything except generated artifacts (all of these are tracked):

- `model/`, `tokenizer/` (including the 1.7 MiB vocab/merge artifacts),
  `tokenizer_training/`, `config/`, `initialization/`, `tests/`, `scripts/`,
  `tools/`, `docs/`, `deployment/`, `README.md`, `AGENTS.md`,
  `requirements.txt`, `requirements-tokenizer.txt`, `pyproject.toml`,
  `.gitignore`

Git-ignored and regenerable: `checkpoints/`, `dist/`, `/workspace/corpus/`,
`__pycache__/`, `.pytest_cache/`. The Q4_K_M artifact is published as a release
asset on `v0.2.0-stage2` (rebuild after the RoPE fix and update the sha256 in
`deployment/README.md`); never commit GGUF files to git.
