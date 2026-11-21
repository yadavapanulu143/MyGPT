# Final report: independent 0.5B-parameter LLM

Stage 1 deliverable: architecture, tokenizer foundation, random initialization,
structural verification, memory/performance analysis, quantized GGUF deployment
path, and laptop instructions. No training has been performed.

Date: 2026-11-21
Model: MyGPT 0.5B (`mygpt-0.5b`)

---

## 1. Architecture specifications

| Property | Value |
|---|---|
| Type | Decoder-only autoregressive transformer |
| Layers | 41 |
| d_model | 1024 |
| Attention heads | 16 (head_dim 64) |
| KV heads | 4 (grouped-query attention, 4:1) |
| FFN | 2816, SwiGLU (gate/up/down) |
| Vocabulary | 32768, byte-level BPE, tied to output projection |
| Context | 2048 tokens |
| Positions | RoPE, base 10000, Q and K only |
| Norm | RMSNorm (pre-norm), eps 1e-5 |
| Biases | none |
| Init | normal(0, 0.02), seed 1337, RMSNorm weights = 1.0 |
| Reference dtype | fp32 |

Rationale for each choice is in `docs/architecture.md`.

## 2. Exact parameter count

```
41 x 11,274,240 (per layer)      = 462,243,840
token embedding 32768 x 1024     =  33,554,432
final RMSNorm                    =       1,024
tied LM head                     =           0
--------------------------------------------------
total                            = 495,799,296
```

- Analytic: **495,799,296**
- Instantiated module: **495,799,296**
- Read back from the checkpoint: **495,799,296**
- Deviation from 500M: **-0.84%**

Verified by `scripts/count_parameters.py` and `tests/test_parameters.py`.

Component split: attention 107,479,040 (21.7%), MLP 354,680,832 (71.5%),
norms 83,968, embedding 33,554,432 (6.8%).

## 3. Model file sizes (all produced in this run)

| Artifact | Bytes | MiB |
|---|---|---|
| `model.safetensors` (fp32) | 1,983,235,576 | 1,891.3 |
| `mygpt-0.5b-f16.gguf` | 992,630,144 | 946.6 |
| `mygpt-0.5b-q8_0.gguf` | 527,898,048 | 503.4 |
| `mygpt-0.5b-q5_k_m.gguf` | 355,235,264 | 338.8 |
| `mygpt-0.5b-q4_k_m.gguf` | 305,722,816 | 291.6 |
| `mygpt-0.5b-q4_0.gguf` | 288,691,648 | 275.3 |

## 4. Memory requirements

KV cache (fp16) = `2 x 41 layers x 256 kv_dim x C x 2 bytes`:
10 MiB @ 256 tokens, 21 MiB @ 512, 41 MiB @ 1024, 82 MiB @ 2048.

Measured process RSS with llama.cpp (weights in RAM, no mmap, 32 tokens decoded):

| Model | ctx 256 | ctx 2048 |
|---|---|---|
| F16 | 977 MiB | 1049 MiB |
| Q8_0 | 534 MiB | 606 MiB |
| Q5_K_M | 371 MiB | 443 MiB |
| Q4_K_M | 324 MiB | 396 MiB |
| Q4_0 | 306 MiB | 378 MiB |

PyTorch reference: fp32 RSS 2126 MiB after load, bf16 weights 946 MiB (peak RSS
during the whole two-dtype run: 4066 MiB).

## 5. CPU inference benchmarks (measured on the dev machine)

Dev machine: AMD EPYC 9B14, 4 vCPU, no GPU, llama.cpp `b1-1fdda89`, 4 threads.

`llama-bench` (tokens/s):

| Model | pp64 | pp256 | tg32 | tg128 |
|---|---|---|---|---|
| F16 | 181.1 | 167.5 | 24.6 | 19.5 |
| Q8_0 | 168.4 | 187.3 | 49.7 | 37.5 |
| Q5_K_M | 354.3 | 392.2 | 49.9 | 56.5 |
| Q4_K_M | 282.2 | 266.1 | 65.8 | 67.1 |
| Q4_0 | 274.1 | 234.9 | 68.4 | 62.7 |

Reference PyTorch implementation (greedy, 16 tokens):

| dtype | ctx64 prefill | ctx64 decode | ctx256 prefill | ctx256 decode |
|---|---|---|---|---|
| fp32 | 99.5 t/s | 6.6 t/s | 120.7 t/s | 5.8 t/s |
| bf16 | 253.7 t/s | 14.9 t/s | 211.9 t/s | 12.4 t/s |

Load time (llama.cpp, from page cache): 0.9-2.7 s including 32 generated tokens.
PyTorch load: 4.6-4.8 s.

**No laptop benchmark was run.** The laptop estimates in section 8 are derived
from these numbers with the scaling argument documented in
`docs/memory_analysis.md` section 4.

## 6. Quantization results

GGUF quantization, done with llama.cpp's `llama-quantize` from our own F16 file:

| Type | Size | Generation speed | Note |
|---|---|---|---|
| F16 | 947 MiB | 19.5-24.6 t/s | reference |
| Q8_0 | 503 MiB | 37.5-49.7 t/s | near-lossless |
| Q5_K_M | 339 MiB | 49.9-56.5 t/s | good balance |
| Q4_K_M | 292 MiB | 65.8-67.1 t/s | **recommended** |
| Q4_0 | 275 MiB | 62.7-68.4 t/s | legacy format |

Independent group-wise quantization of the PyTorch weights (pure PyTorch,
`scripts/quantize_reference.py`), measured on a 64-token batch:

| Scheme | Est. size | Relative weight error | Logit cosine | Top-1 agreement |
|---|---|---|---|---|
| INT8, group 64 | 488 MiB | 0.59% | 0.99959 | 98.4% |
| INT8, group 128 | 481 MiB | 0.65% | 0.99949 | 95.3% |
| INT4, group 32 | 266 MiB | 9.70% | 0.879 | 10.9% |
| INT4, group 64 | 252 MiB | 10.8% | 0.857 | 37.5% |
| INT4, group 128 | 244 MiB | 11.7% | 0.858 | 31.3% |

Top-1 agreement is a poor metric for this model: with random weights the logits
are near-ties, so tiny errors flip the argmax. Weight error and logit cosine are
the meaningful signals at this stage; real quality measurement needs a trained
model and perplexity/benchmark scores.

## 7. Test results

`python3 -m pytest tests -q` → **67 passed** (28.5 s).

| Test file | Covers |
|---|---|
| `test_architecture.py` (11) | imports, config invariants, module wiring, no biases, weight tying, RMSNorm scale invariance, RoPE offset/relative property, SwiGLU shapes |
| `test_parameters.py` (7) | exact count 495,799,296 analytic vs instantiated, breakdown sums, tied vs untied, 5% band |
| `test_forward_pass.py` (13) | output shapes, causality (future tokens cannot change past logits), prefix stability, batch independence, KV-cache equivalence, context limit, generation with/without seed, EOS stop |
| `test_save_load.py` (11) | random-init verification (mean/std/reproducibility/different-seed), no NaN/inf, manifest, bit-identical reload, config round-trip, tied weight saved once, mismatch detection, full-size 1.85 GiB round-trip |
| `test_tokenizer.py` (17) | vocab layout, lossless round-trip (ASCII/Unicode/emoji/CRLF), no unknown tokens, specials, GPT-2-style pre-tokenization, save/load, trainer determinism, GGUF vocab view, no downloaded tokenizer artifacts |
| `test_policy.py` (8) | policy loads from JSON, no keyword-based refusal, no blocklist attributes, sampling defaults, configurability |

Additional verification:

- **GGUF parity (F16)**: tokenizer ids and 8 greedy tokens from llama.cpp match the
  PyTorch reference exactly on `"Hello world, this is a test of the model."`
  (`scripts/verify_gguf_parity.py`, harness `tools/llama_parity.cpp`).
- **GGUF parity (Q4_K_M)**: prompt ids still match exactly; the first generated
  token can differ (22177 vs 25567 in the run recorded here) because with random
  weights the top logits are near-ties and quantization noise decides the argmax.
  This is expected for an untrained model and says nothing about quantized quality
  on a trained one.
- **Random initialization**: mean -5.8e-07, std 0.0200004 (target 0.02), min
  -0.1153, max 0.1149, fingerprint reproducible from seed 1337, different for
  seed 1338. No pre-trained weights anywhere (`pretrained_weights_used: false`
  in `checkpoints/mygpt-0.5b-init/init_manifest.json`).

## 8. Laptop compatibility assessment

Target: Ryzen 3 3250U (2 cores / 4 threads, AVX2, Vega 3 shared memory),
8 GB RAM / 5.94 GB usable, Windows 11, SSD.

| Question | Answer |
|---|---|
| Does it fit in RAM? | Yes, comfortably. Q4_K_M needs ~400-550 MiB including KV cache at 2048 context; Windows leaves ~2.5-3.4 GB. |
| Does it need a GPU? | No. CPU-only path verified; no CUDA, no ROCm. |
| Does it need the internet? | No. One 292 MiB file plus a llama.cpp binary. |
| Expected generation speed | **8-15 tokens/s** (Q4_K_M, 2 threads) - estimate |
| Expected prompt processing | 50-80 tokens/s - estimate |
| Expected first-token latency | ~1 s for a 50-token prompt - estimate |
| Load time | 1-3 s from SSD - estimate |
| Storage | 292 MiB model + ~50 MiB binary |
| Risk | CPU is the bottleneck, not memory. Thermal throttling and battery power plans can halve throughput. |

The estimate is derived by scaling the measured 67 t/s on 4 AVX-512 cores down by
the laptop's ~4-6x lower vector throughput (2 cores, AVX2, lower clocks), which
is consistent with the memory-bandwidth ceiling (~19-25 GB/s effective).

## 9. Future training requirements

Summary (details in `docs/training_plan.md`):

| Stage | Data | Compute | Cost estimate |
|---|---|---|---|
| Tokenizer training | 10-30 GB text | minutes on CPU | none |
| Pretraining | 10-40B tokens (~50-80 GB raw) | 1x A100: 3-6 days for 10-20B tokens; 8x H100: <1 day | ~$60-250 at spot prices |
| Instruction tuning | 200k-1M pairs | 1-2 GPU-days on 1x A100 | ~$30-100 |
| Behaviour tuning (DPO/ORPO) | 20-50k preference pairs | hours on 1x A100 | ~$10-30 |

Training RAM: 10-14 GB of GPU memory with gradient checkpointing (bf16 weights 1
GB, gradients 1 GB, fp32 master 2 GB, Adam 4 GB, activations 2-6 GB).
CPU-only training is not practical: ~50 years for 10B tokens on this dev box.

## 10. Remaining limitations

1. **The model cannot hold a conversation.** The weights are random; output is
   noise. Only the architecture, plumbing and deployment path are proven.
2. The tokenizer has 260 of 32768 tokens populated (4 specials + 256 bytes).
   Ids 260-32767 are placeholders until tokenizer training runs.
3. Laptop performance numbers are estimates, not measurements.
4. Q4_K_M quality on a trained model is unmeasured; the choice is based on
   standard practice and on our own error metrics.
5. No chat template is embedded in the GGUF yet; `llama-server` will not apply
   one automatically until the trained tokenizer config carries it.
6. No long-context (>2048) support; RoPE base can be adjusted later, but the
   model has never seen longer sequences.
7. No safety evaluation of actual behaviour is possible before training, so no
   claim is made that the model is "uncensored" or "helpful" - only that the
   policy layer is configurable and contains no keyword filter.
8. The dev box has 25 GB of disk: it can hold the fp32 checkpoint plus all
   quantized artifacts (about 4.3 GB), but not a training corpus.

## 11. Project structure

```
MyGPT/
├── AGENTS.md                        project memory: decisions, status, next steps
├── README.md                        overview and quick start
├── .gitignore                       excludes checkpoints, dist, *.gguf
├── requirements.txt
├── config/
│   ├── model_config.json            the 495.8M architecture
│   └── behaviour_policy.json        configurable low-refusal policy (prompt level)
├── model/
│   ├── config.py                    ModelConfig + analytic parameter accounting
│   ├── attention.py                 RoPE + GQA causal attention
│   ├── transformer_block.py         RMSNorm, SwiGLU, block
│   ├── embeddings.py                token embedding + tied LM head
│   ├── architecture.py              MyGPT (forward, KV-cache generate, save/load)
│   ├── io.py                        safetensors I/O, tied-weight dedupe
│   └── policy.py                    behaviour policy loader
├── tokenizer/
│   ├── tokenizer.py                 byte-level BPE (encode/decode/train/GGUF view)
│   └── config.json                  vocab layout, 32768 target, empty merges
├── initialization/
│   └── initialize_weights.py        seeded init + verification + manifest
├── tests/
│   ├── conftest.py                  tiny + full configs
│   ├── test_architecture.py
│   ├── test_parameters.py
│   ├── test_forward_pass.py
│   ├── test_save_load.py
│   ├── test_tokenizer.py
│   └── test_policy.py
├── scripts/
│   ├── count_parameters.py
│   ├── benchmark_memory.py
│   ├── test_inference.py
│   ├── quantize_reference.py        pure-PyTorch INT8/INT4 experiments
│   ├── verify_gguf_parity.py        PyTorch vs llama.cpp token/id comparison
│   └── chat.py                      interactive shell (untrained model)
├── tools/
│   ├── gguf_writer.py               GGUF v3 writer (no external deps)
│   ├── export_gguf.py               checkpoint -> GGUF for llama.cpp
│   └── llama_parity.cpp             tiny llama.cpp harness for parity tests
├── docs/
│   ├── architecture.md
│   ├── memory_analysis.md
│   ├── training_plan.md
│   └── final_report.md              this file
├── deployment/
│   ├── README.md                    laptop quick start
│   └── deployment_plan.md           runtime choice, pipeline, transfer, checks
├── checkpoints/mygpt-0.5b-init/     generated, git-ignored (1.85 GiB + manifest)
└── dist/                            generated, git-ignored (5 GGUF files)
```

Generated artifacts (not in git) are reproduced by the commands in
`deployment/deployment_plan.md` section 2.
