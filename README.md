# MyGPT - an independent 0.5B-parameter LLM built from scratch

A decoder-only transformer with **495,799,296 parameters**, implemented and
randomly initialized in this repository. No pre-trained weights, no downloaded
tokenizer, no fine-tuning of an existing model. Designed for CPU-only, offline
inference on a Ryzen 3 3250U laptop with 8 GB RAM.

> **Status: stage 2.1 adopted (tokenizer final); training not started.** The
> architecture, the trained 32,768-entry tokenizer, the verification suite and
> the quantized GGUF deployment path are complete and tested. The weights are
> random, so the model produces noise - it is not yet a conversational
> assistant. Pretraining is a separate, explicitly approved stage
> (`docs/training_plan.md`).

## What is proven here

| Item | Result |
|---|---|
| Parameter count | 495,799,296 analytic = instantiated = on disk |
| Tests | 137 passing, 1 skipped (`pytest tests -q`) |
| Tokenizer | trained 32,768-entry byte-level BPE: 4 specials + 256 bytes + 32,500 merges + 8 chat specials |
| Compression | 4.012 bytes/token English, 3.53 code, 13.3% fewer code tokens than Stage 2 at +0.08% English cost (`docs/tokenizer_stage2.1_report.md`) |
| Chat template | ChatML-like, in `tokenizer/config.json` and GGUF; Jinja and Python renderers verified identical (`docs/chat_template.md`) |
| Random init | verified (mean ~0, std 0.0200, seed-reproducible, no pre-trained data) |
| Causal masking | verified (future tokens cannot change past logits) |
| KV cache | verified equivalent to a full forward pass |
| GGUF export | loads in llama.cpp, token ids match for every prompt, F16 greedy decode is token-for-token identical (logit cosine 0.999999) |
| Quantization | F16 / Q8_0 / Q5_K_M / Q4_K_M / Q4_0 produced and benchmarked |
| Laptop footprint | Q4_K_M: 292 MiB file, ~400-550 MiB RAM, no GPU, no network |

## Quick start (development machine)

```bash
pip install -r requirements.txt

# 1. architecture + exact parameter count
python3 scripts/count_parameters.py

# 2. run the verification suite (137 tests, 1 skipped without llama.cpp)
python3 -m pytest tests -q

# 3. create the randomly initialized checkpoint (1.85 GiB, fp32)
python3 initialization/initialize_weights.py --out checkpoints/mygpt-0.5b-init

# 4. export to GGUF for llama.cpp and quantize
python3 tools/export_gguf.py --checkpoint checkpoints/mygpt-0.5b-init \
    --output dist/mygpt-0.5b-f16.gguf
llama-quantize dist/mygpt-0.5b-f16.gguf dist/mygpt-0.5b-q4_k_m.gguf Q4_K_M

# 5. benchmarks and profiling
python3 scripts/benchmark_memory.py --contexts 64,256 --dtypes float32,bfloat16
python3 scripts/quantize_reference.py
python3 scripts/verify_gguf_parity.py --gguf dist/mygpt-0.5b-f16.gguf
```

Chat prompts are rendered with the adopted template; `encode_chat` appends marker
ids instead of re-parsing text, so user content cannot forge roles:

```bash
python3 scripts/render_chat.py --system "You are MyGPT." --user "Explain RoPE." --json
python3 tokenizer_training/compare_tokenizers.py --tokenizer stage2=tokenizer_candidates/stage2-baseline \
    --tokenizer adopted=tokenizer
```

Retraining the tokenizer needs the extra training dependencies
(`pip install -r requirements-tokenizer.txt`) and a corpus; see
`docs/tokenizer.md` for the four commands and the corpus provenance.

## Model at a glance

| | |
|---|---|
| Parameters | 495,799,296 (495.8M) |
| Layers / d_model | 41 / 1024 |
| Heads / KV heads | 16 / 4 (GQA, head_dim 64) |
| FFN | 2816, SwiGLU |
| Vocab / context | 32768 (tied embeddings, trained BPE) / 2048 |
| Positions / norm | RoPE (base 10000) / RMSNorm |
| Dtypes | fp32 source, fp16 GGUF, Q8_0, Q5_K_M, Q4_K_M, Q4_0 |

## Measured performance (dev machine: 4 vCPU EPYC, no GPU)

| Model | Size | Generation | Prompt processing |
|---|---|---|---|
| F16 | 947 MiB | 19.5-24.6 t/s | 167-181 t/s |
| Q8_0 | 503 MiB | 37.5-49.7 t/s | 168-187 t/s |
| Q4_K_M | 292 MiB | 65.8-67.1 t/s | 266-282 t/s |

### Measured on the target laptop (AMD Ryzen 3 3250U, 2 threads, Q4_K_M)

`llama-bench` on the actual machine, not an estimate:

| Benchmark | Measured |
|---|---:|
| Prompt processing, 64 tokens | 40.37 tokens/s |
| Prompt processing, 256 tokens | 52.85 tokens/s |
| Generation, 32 tokens | 26.93 tokens/s |
| Generation, 128 tokens | 25.26 tokens/s |

So the laptop generates ~25 tokens/s, faster than the 8-15 tokens/s estimate in
`docs/memory_analysis.md` (the estimate assumed a smaller effective AVX2 penalty
and 2 cores at 2.6 GHz). Generation is the number that matters for chat; prompt
processing at 40-53 tokens/s means a 500-token prompt costs ~10 s before the
first token.

## Documentation

| Document | Contents |
|---|---|
| `docs/final_report.md` | full handover: specs, counts, sizes, benchmarks, tests, limitations |
| `docs/architecture.md` | design decisions and the parameter math |
| `docs/memory_analysis.md` | storage formats, runtime memory, measured vs estimated speed |
| `docs/tokenizer.md` | tokenizer design, corpus, licences, training, reproduction |
| `docs/tokenizer_report.md` | tokenizer evaluation: compression, probes, HF/llama.cpp parity |
| `docs/training_plan.md` | data, compute, cost and evaluation plan for the training stage |
| `docs/stage3_plan.md` | **current** Stage 3 proposal: corpora and licensing, mixture, GPU budget, optimiser, checkpointing, refusal evaluation. Not approved, nothing trained |
| `docs/tokenizer_stage2.1_report.md` | adopted tokenizer: specials, code corpus, before/after measurements, adoption verification |
| `docs/tokenizer_special_tokens.md` | every reserved id, why the set is sufficient, conflict review |
| `docs/chat_template.md` | chat wire format, injection safety, llama.cpp usage |
| `deployment/README.md` | Windows 11 quick start |
| `deployment/deployment_plan.md` | runtime choice, artifact pipeline, transfer, offline checks |
| `AGENTS.md` | project memory: decisions, status, next steps |

## Behaviour policy

`config/behaviour_policy.json` holds the system prompt and sampling defaults for a
broad, low-refusal assistant: politics, religion, history, mature fiction,
relationships, controversial and sensitive-but-legitimate topics are all in
scope. Refusals are limited to concrete help with serious real-world harm. There
is **no keyword filter** anywhere in the codebase, and `tests/test_policy.py`
enforces that. The policy is a configuration, not a trained behaviour: the model
must be trained and then evaluated for over-refusal before any claim about how it
actually behaves.

## Layout

```
model/          architecture, attention, blocks, embeddings, I/O, policy
tokenizer/      trained 32,768-entry byte-level BPE + vocab/merges artifacts
tokenizer_training/  corpus preparation, BPE training, evaluation, manifests
config/         model_config.json, behaviour_policy.json
initialization/ seeded random init + verification manifest
tests/          architecture, parameters, forward pass, save/load, tokenizer, policy
scripts/        count_parameters, benchmark_memory, test_inference,
                quantize_reference, verify_gguf_parity, chat
tools/          GGUF writer, GGUF exporter, llama.cpp parity harness
docs/           architecture, memory analysis, training plan, final report
deployment/     laptop quick start and deployment plan
```

## Provenance

All code and weights in this repository were produced for this project. No
third-party model weights or tokenizer artifacts are included or downloaded.
PyTorch is the tensor library; llama.cpp is used only as the inference runtime
for the GGUF artifacts.
