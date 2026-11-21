# AGENTS.md - project memory

Read this first. It records what exists, why, and what is next, so a future
session does not have to rediscover it.

## Project in one paragraph

`MyGPT` is an independently built decoder-only LLM with **495,799,296**
parameters, written from scratch in PyTorch, randomly initialized, verified
structurally, exported to GGUF, quantized to Q4_K_M, and benchmarked. Deployment
target: Windows 11 laptop, AMD Ryzen 3 3250U (2C/4T, AVX2, Vega 3), 8 GB RAM
(~5.94 GB usable), CPU-only, offline. Stage 1 (this work) deliberately stops
before any training.

## Hard constraints (do not violate)

1. No pre-trained weights, tokenizers or configs. Everything is generated here.
2. Do not start pretraining, instruction tuning or RL without explicit approval.
3. The model must remain loadable by stock llama.cpp (GGUF `llama` architecture).
4. Do not claim the model is intelligent, helpful or uncensored: it is untrained.
5. Never invent laptop benchmark numbers. Dev-machine numbers are measured;
   laptop numbers must be labelled estimates with the derivation shown.
6. No keyword-based censorship anywhere. Behaviour is a configurable prompt
   policy (`config/behaviour_policy.json`), not a filter.
7. Keep the fp32 checkpoint as the source of truth for all quantized artifacts.

## Architecture decisions (locked unless there is a measured reason)

| Decision | Value | Why |
|---|---|---|
| Layers / d_model | 41 / 1024 | 500M budget with short sequential chain for CPU |
| Heads / KV heads | 16 / 4 | GQA cuts KV cache 4x (328 -> 82 MiB at 2048) |
| FFN | 2816 (multiple of 256) | SwiGLU at 2.75x; 256 keeps k-quants fast/possible |
| Vocab | 32768, tied | 6.8% of params; tying removes a second 33.5M matrix |
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
| `tokenizer/tokenizer.py` | `ByteLevelBPETokenizer`: ids 0-3 specials, 4-259 bytes, 260+ merges (empty now) |
| `initialization/initialize_weights.py` | seeded init, stats, fingerprint, `init_manifest.json` |
| `tools/gguf_writer.py` | GGUF v3 writer, hand-rolled, no `gguf` package |
| `tools/export_gguf.py` | checkpoint -> GGUF; 1-D tensors stay F32 |
| `tools/llama_parity.cpp` | llama.cpp harness printing prompt/greedy token ids |
| `scripts/` | count_parameters, benchmark_memory, test_inference, quantize_reference, verify_gguf_parity, chat |
| `docs/` | architecture, memory_analysis, training_plan, final_report |
| `deployment/` | laptop quick start and plan |

## Commands that matter

```bash
python3 -m pytest tests -q                       # 67 tests, ~28 s
python3 scripts/count_parameters.py              # 495,799,296 everywhere
python3 initialization/initialize_weights.py --out checkpoints/mygpt-0.5b-init
python3 tools/export_gguf.py --checkpoint checkpoints/mygpt-0.5b-init --output dist/mygpt-0.5b-f16.gguf
llama-quantize dist/mygpt-0.5b-f16.gguf dist/mygpt-0.5b-q4_k_m.gguf Q4_K_M
python3 scripts/verify_gguf_parity.py --gguf dist/mygpt-0.5b-f16.gguf
python3 scripts/benchmark_memory.py --contexts 64,256 --dtypes float32,bfloat16
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
  tokenizer exposes `gguf_tokens()` for exactly that. Empty `merges` array must
  still be present or loading throws "cannot find tokenizer merges".
- Random-init models generate degenerate repeated tokens; that is expected, not a
  bug. Use the parity harness (token ids), not text, to verify correctness.
- PyTorch is 4-5x slower than llama.cpp on CPU; do not benchmark the laptop path
  with PyTorch.

## Status (2026-11-21)

Done: architecture, tokenizer foundation, init + manifest, 67 tests, GGUF export,
5 quantizations, parity check, dev-machine benchmarks, memory profiling,
reference INT8/INT4 study, all docs, deployment plan.

Not done (needs approval): tokenizer training, pretraining, instruction tuning,
behaviour tuning, laptop benchmarking, long-context extension.

## Next steps, in order

1. Approve training scope (corpus, GPU budget, evaluation thresholds).
2. Train the tokenizer on 10-30 GB English text; re-export GGUF; re-verify parity.
3. Pretrain 10-20B tokens on rented GPUs; monitor validation loss.
4. Instruction-tune, then behaviour-tune (DPO/ORPO against over-refusal).
5. Evaluate: perplexity, lm-eval-harness, XSTest over-refusal, safety floor.
6. Re-export GGUF, re-quantize, benchmark on the actual laptop with `llama-bench`.

## Files that must be pushed for persistence

Everything except generated artifacts (all of these are tracked):

- `model/`, `tokenizer/`, `config/`, `initialization/`, `tests/`, `scripts/`,
  `tools/`, `docs/`, `deployment/`, `README.md`, `AGENTS.md`,
  `requirements.txt`, `pyproject.toml`, `.gitignore`

Git-ignored and regenerable: `checkpoints/`, `dist/`, `__pycache__/`,
`.pytest_cache/`. If the checkpoint or GGUF files must be preserved, attach them
to a GitHub release (or use Git LFS); do not commit them.
