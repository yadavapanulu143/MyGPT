# MyGPT - an independent 0.5B-parameter LLM built from scratch

A decoder-only transformer with **495,799,296 parameters**, implemented and
randomly initialized in this repository. No pre-trained weights, no downloaded
tokenizer, no fine-tuning of an existing model. Designed for CPU-only, offline
inference on a Ryzen 3 3250U laptop with 8 GB RAM.

> **Status: stage 1 of 2.** The architecture, tokenizer foundation, verification
> suite and quantized GGUF deployment path are complete and tested. The weights
> are random, so the model produces noise - it is not yet a conversational
> assistant. Training is a separate, explicitly approved stage
> (`docs/training_plan.md`).

## What is proven here

| Item | Result |
|---|---|
| Parameter count | 495,799,296 analytic = instantiated = on disk |
| Tests | 67 passing (`pytest tests -q`) |
| Random init | verified (mean ~0, std 0.0200, seed-reproducible, no pre-trained data) |
| Causal masking | verified (future tokens cannot change past logits) |
| KV cache | verified equivalent to a full forward pass |
| GGUF export | loads in llama.cpp, tokenizer and greedy decode match PyTorch exactly |
| Quantization | F16 / Q8_0 / Q5_K_M / Q4_K_M / Q4_0 produced and benchmarked |
| Laptop footprint | Q4_K_M: 292 MiB file, ~400-550 MiB RAM, no GPU, no network |

## Quick start (development machine)

```bash
pip install -r requirements.txt

# 1. architecture + exact parameter count
python3 scripts/count_parameters.py

# 2. run the verification suite (67 tests)
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

## Model at a glance

| | |
|---|---|
| Parameters | 495,799,296 (495.8M) |
| Layers / d_model | 41 / 1024 |
| Heads / KV heads | 16 / 4 (GQA, head_dim 64) |
| FFN | 2816, SwiGLU |
| Vocab / context | 32768 (tied embeddings) / 2048 |
| Positions / norm | RoPE (base 10000) / RMSNorm |
| Dtypes | fp32 source, fp16 GGUF, Q8_0, Q5_K_M, Q4_K_M, Q4_0 |

## Measured performance (dev machine: 4 vCPU EPYC, no GPU)

| Model | Size | Generation | Prompt processing |
|---|---|---|---|
| F16 | 947 MiB | 19.5-24.6 t/s | 167-181 t/s |
| Q8_0 | 503 MiB | 37.5-49.7 t/s | 168-187 t/s |
| Q4_K_M | 292 MiB | 65.8-67.1 t/s | 266-282 t/s |

Laptop estimate for Q4_K_M: **8-15 tokens/s**, ~400-550 MiB RAM, ~1 s to first
token. These are estimates derived from the measurements above, not laptop
benchmarks - see `docs/memory_analysis.md`.

## Documentation

| Document | Contents |
|---|---|
| `docs/final_report.md` | full handover: specs, counts, sizes, benchmarks, tests, limitations |
| `docs/architecture.md` | design decisions and the parameter math |
| `docs/memory_analysis.md` | storage formats, runtime memory, measured vs estimated speed |
| `docs/training_plan.md` | data, compute, cost and evaluation plan for the training stage |
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
tokenizer/      independent byte-level BPE (encode/decode/train/GGUF view)
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
