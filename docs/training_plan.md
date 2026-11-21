# Training strategy (future stage - not executed, needs approval)

Nothing in this document has been run. It is the plan for turning the current
randomly initialized 495.8M-parameter model into something that can hold a
conversation, with the storage/RAM/compute numbers worked out so the decision is
made on facts rather than optimism.

## 0. What exists now and what it can do

| Stage | Status | Capability |
|---|---|---|
| Architecture + random init | done | none: logits are essentially arbitrary |
| Tokenizer foundation | done | lossless byte-level encoding |
| Tokenizer training | **done (stage 2, revised in stage 2.1)** | 32,768 ids, 32,500 merges, 8 chat specials, 4.012 bytes/token English + 3.53 code held-out; see `docs/tokenizer_stage2.1_report.md` |
| Pretraining | not started | needs a GPU cluster or rented GPUs |
| Instruction tuning | not started | needs pretraining first |
| Behaviour tuning | not started | needs instruction tuning first |

A random model cannot answer questions. Everything below is about closing that
gap; it is not a description of current behaviour.

## 1. Tokenizer training (completed in stage 2)

Executed with 849 MiB of English text (WikiText-103, Project Gutenberg, OASST1,
CPython stdlib, 44 IETF RFCs), byte-level BPE, 32,768 ids, `min_frequency=2`.
Result: 4.650 bytes/token on 0.5 MiB of held-out text, 100% HF parity, identical
ids in llama.cpp. Details and reproduction commands are in `docs/tokenizer.md`;
measurements are in `docs/tokenizer_report.md`.

What is left for a later retrain, if the corpus changes:

1. Keep the specials and byte block fixed so ids 0-259 never move.
2. Reserve chat tokens **before** training: the vocabulary is full at 32,768, so
   either drop ~8 merges to free ids, or plan on `<bos>/<eos>` plus text role
   markers.
3. Add more code (the current 14.9 MiB leaves identifiers like `model` split) and
   cap whitespace runs (about 15 slots went to 43-51 space tokens).
4. Re-run `train_tokenizer.py`, `evaluate_tokenizer.py`, re-export the GGUF and
   re-run `scripts/verify_gguf_parity.py`.

No tokenizer artifact is downloaded at any stage; the trainer library
(`tokenizers`) is a build-time dependency only.

## 2. Pretraining data

Chinchilla-optimal for 495.8M parameters is ~20 tokens per parameter = ~10B
tokens. Small models benefit from over-training, so 20-40B tokens is the
sensible range; 10B is the minimum for a coherent model.

| Corpus | Tokens | Raw storage | Notes |
|---|---|---|---|
| FineWeb-Edu (sample) | 10B | ~30-40 GB (gzip) | main English web text |
| English Wikipedia | 3B | ~6 GB | factual grounding |
| Books / Gutenberg | 1B | ~3 GB | long-form prose |
| Code (permissive) | 2B | ~4 GB | programming ability |
| Deduplicated instruction-free math | 1B | ~2 GB | arithmetic/reasoning |

Practical budget: 15-20B tokens, ~50 GB raw text, ~10 GB after packing into
uint16/uint32 token shards. Disk on the dev box (25 GB) is not enough - training
storage belongs on the training host or a mounted volume.

## 3. Compute reality

Training FLOPs ≈ 6 x N x D:

| Tokens | FLOPs |
|---|---|
| 10B | 3.0 x 10^19 |
| 20B | 6.0 x 10^19 |
| 40B | 1.2 x 10^20 |

| Hardware | Effective throughput | 10B tokens | 20B tokens |
|---|---|---|---|
| 1x RTX 4090 (24 GB, bf16) | ~25 TFLOPS (30% MFU) | ~14 days | ~28 days |
| 1x A100 80 GB | ~125 TFLOPS (40% MFU) | ~2.8 days | ~5.5 days |
| 4x A100 80 GB | ~500 TFLOPS | ~17 hours | ~1.4 days |
| 8x H100 | ~2 PFLOPS | ~4 hours | ~8 hours |
| This dev box (4 vCPU) | ~0.02 TFLOPS | ~50 years | not feasible |

Conclusion: **CPU-only cloud training is not practical** at this scale. Training
needs rented GPUs (or a research allocation). Rough cost at current spot prices
for 20B tokens: ~$60-150 on a single A100, ~$100-250 on 8x H100 for a shorter
wall clock, plus storage/egress. This is an estimate, not a quote.

## 4. Memory during training

AdamW with mixed precision, 495.8M parameters:

| Item | Size |
|---|---|
| bf16 weights | 1.0 GB |
| bf16 gradients | 1.0 GB |
| fp32 master weights | 2.0 GB |
| Adam m and v (fp32) | 4.0 GB |
| activations (gradient checkpointing, seq 2048, batch 8) | 2-6 GB |
| total | **10-14 GB** |

So a single 24 GB GPU is enough with gradient checkpointing; 2 GPUs with data
parallelism or ZeRO-2 make it comfortable. Without checkpointing, activations at
2048 tokens and batch 8 need ~20+ GB on their own.

## 5. Pretraining recipe

- Optimizer: AdamW, beta 0.9/0.95, weight decay 0.1, grad clip 1.0.
- LR: 3e-4 peak with 2000-step warmup, cosine decay to 10% of peak.
- Batch: 0.5-1M tokens per step (gradient accumulation on 1-2 GPUs).
- Precision: bf16 autocast, fp32 master weights.
- Sequence length: 2048 (matches the architecture); pack documents, no padding.
- Checkpoint every 500 steps, keep the last 3 plus the best validation loss.
- Track: train loss, held-out validation loss, gradient norm, throughput.
- Expected loss: a 0.5B model on 20B tokens of FineWeb-Edu should land around
  2.6-2.9 nats/token on held-out web text. Anything above 3.2 means something is
  wrong (data pipeline, LR, or tokenizer mismatch).

## 6. Instruction tuning

After pretraining:

1. Build or license an instruction dataset: 200k-1M prompt/response pairs
   (open datasets such as Tulu-3 SFT mixture, OpenHermes, plus hand-written
   examples for the behaviour you actually want).
2. Format with the chat template in `config/behaviour_policy.json` so training and
   inference agree.
3. Fine-tune 2-3 epochs at LR 5e-5 cosine, batch 64-128 sequences of 2048.
   Cost: ~1-2 GPU-days on an A100 for 500k examples.
4. Loss on completion tokens only, not on the prompt.

## 7. Behaviour tuning for a low-refusal model

The requirement is a broadly capable model with minimal unnecessary censorship.
The mechanism is data plus a configurable system prompt - not a runtime filter:

- Include SFT examples that answer politics, religion, sexuality, relationships,
  mature fiction, controversial history and other sensitive-but-legitimate
  topics directly.
- Include a small set of refusals only for the categories listed in
  `config/behaviour_policy.json` (`out_of_scope`), phrased briefly and without
  moralizing.
- Optional preference optimisation (DPO/ORPO) on pairs where the preferred answer
  engages with the question and the rejected answer refuses it. This is the most
  effective known lever against over-refusal, and it is cheap at 0.5B: a few
  hours on one GPU for 20-50k pairs.
- Keep the system prompt configurable so the user can raise or lower strictness
  without retraining. `model/policy.py` already loads it from JSON.
- Do not add a keyword blocklist. If a filter is ever wanted, it should be a
  classifier evaluated on false-positive rate, and it should be off by default.

## 8. Evaluation

| What | How | Target |
|---|---|---|
| Language modelling | held-out perplexity on FineWeb-Edu + Wikipedia | < 20 ppl (web) |
| Knowledge/reasoning | lm-evaluation-harness: MMLU, ARC-easy, HellaSwag, PIQA, GSM8K | MMLU 25-32%, ARC-e 45-55%, HellaSwag 40-50% |
| Instruction following | MT-Bench-style rubric with a stronger model as judge, or human spot checks | 3.5-4.5 / 10 |
| Over-refusal | XSTest plus a custom 200-prompt suite of legitimate sensitive questions | > 90% answered, < 10% unnecessary refusals |
| Safety floor | 50 prompts covering the out-of-scope categories | > 90% declined |
| Regression | fixed 100-prompt set, compare answers before/after every tuning step | no capability loss > 2% |

The two refusal metrics must always be reported together. Answering everything
is not a success, and refusing nothing is not a success either.

## 9. Storage and RAM summary for training

| Item | Size |
|---|---|
| fp32 checkpoint | 2.0 GB |
| bf16 checkpoint | 1.0 GB |
| optimizer state (Adam, fp32) | 6.0 GB (m, v, master) |
| tokenized corpus (20B tokens, uint16) | ~40 GB |
| raw text corpus | ~50-80 GB |
| GPU RAM per device | 10-14 GB (with checkpointing) |

## 10. What must be approved before starting

1. Pretraining corpus and its licence/provenance.
2. GPU budget (single A100 vs multi-GPU).
3. Instruction/behaviour dataset composition.
4. Evaluation suite and the refusal-rate thresholds above.
