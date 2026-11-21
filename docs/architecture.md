# Architecture: an independent 0.5B decoder-only transformer

Everything here describes the model that exists in `model/`. It was designed
from the deployment constraints (2 cores, ~6 GB usable RAM, no CUDA, CPU-only
inference) and then sized to land at 500M parameters. No pre-trained weights,
tokenizer or config is involved at any point.

## 1. Final shape

| Item | Value |
|---|---|
| Type | Decoder-only autoregressive transformer (Llama/GPT topology) |
| Parameters | **495,799,296** (495.8M, 99.2% of 500M) |
| Layers | 41 |
| d_model | 1024 |
| Attention heads | 16 (head_dim 64) |
| KV heads | 4 (GQA, 4 query heads per KV head) |
| FFN width | 2816 (2.75x d_model) |
| Vocabulary | 32768 (trained byte-level BPE, see `docs/tokenizer.md`) |
| Context | 2048 tokens |
| Position encoding | RoPE, base 10000, applied to Q and K only |
| Normalization | RMSNorm, pre-norm, eps 1e-5 |
| Activation | SwiGLU (SiLU gating) |
| Biases | none anywhere |
| Output projection | tied to the input embedding |
| Master dtype | fp32; inference fp32 / bf16 / GGUF fp16 + k-quants |

## 2. Parameter count, derived by hand

Per decoder layer:

```
q_proj      d * d            = 1024 * 1024  = 1,048,576
k_proj      d * (kv*hd)      = 1024 *  256  =   262,144
v_proj      d * (kv*hd)      = 1024 *  256  =   262,144
o_proj      d * d            = 1024 * 1024  = 1,048,576
gate_proj   d * ffn          = 1024 * 2816  = 2,883,584
up_proj     d * ffn          = 1024 * 2816  = 2,883,584
down_proj   ffn * d          = 2816 * 1024  = 2,883,584
2 x RMSNorm                  = 2 * 1024    =     2,048
------------------------------------------------------
per layer                                  = 11,274,240
```

Whole model:

```
41 layers         41 * 11,274,240 = 462,243,840
token embedding       32768 * 1024 =  33,554,432
final RMSNorm                     =       1,024
tied LM head                      =           0
------------------------------------------------------
total                             = 495,799,296
```

Verification: `python3 scripts/count_parameters.py` prints the analytic total,
the instantiated total and, with `--checkpoint`, the total read back from disk.
All three agree. `tests/test_parameters.py` asserts the exact value 495,799,296,
so any accidental config change breaks the build.

## 3. Why these choices

**41 layers / d_model 1024.** At a fixed budget, depth buys reasoning and width
buys per-token capacity. 41 x 1024 sits in the range that works for 0.4-0.6B
models, while keeping the sequential chain short: every layer is a separate pass
over memory, so depth costs latency per token at equal parameter count.

**GQA with 4 KV heads (4:1).** The KV cache dominates long-context memory. With
4 KV heads instead of 16 the cache shrinks from 328 MiB to 82 MiB at 2048 tokens
(fp16). Larger models use 8:1, but at width 1024 that leaves only 2 KV heads
(128 values per token) and noticeably less capacity, so 4:1 is the compromise.

**FFN 2816 = 2.75 x d_model, multiple of 256.** SwiGLU uses three matrices, so a
4x expansion would cost 12 d^2 per layer and blow the budget. 2.75x puts 71% of
parameters in the MLP. The multiple-of-256 constraint matters for deployment:
llama.cpp k-quant super-blocks are 256 elements wide, and a row length that is
not a multiple of 256 forces slower fallback paths (Q4_K can even refuse).

**Vocab 32768 with tied embeddings.** The embedding matrix is 33.5M parameters
(6.8% of the model). Tying it to the output projection removes a second matrix
of that size, which matters when the deployable artifact must fit in ~300 MiB.
32768 covers English, code and punctuation comfortably at byte level: 256 byte
tokens + 4 specials + ~32k learned merges.

**Context 2048.** At Q4 the KV cache costs 82 MiB for 2048 tokens and grows
linearly. That is enough for a conversation turn plus a document snippet while
leaving the laptop comfortable. RoPE has no learned position table, so context
can later be extended by changing a number, without new parameters.

**RMSNorm, RoPE, SwiGLU, no biases.** These components make the model
expressible as GGUF `llama` tensors one to one, which is what lets the laptop run
it through llama.cpp with Q4_K_M. A learned-position, LayerNorm-with-bias, GeLU
model would need a custom runtime - exactly the paper-only architecture this
project avoids.

**fp32 master weights.** 1.85 GiB on disk. Training needs fp32 (or bf16 with an
fp32 master copy); inference converts to fp16 or quantized. Keeping fp32 as the
source of truth means every quantized artifact can be regenerated.

## 3a. Tokenizer interface

The architecture only requires that the vocabulary size matches the embedding
matrix. The Stage 2 tokenizer fills it with a trained 32,768-entry byte-level
BPE vocabulary: ids 0-3 are `<pad>/<bos>/<eos>/<unk>`, 4-259 are raw bytes,
260-32767 are merges. Because the 256 byte tokens exist, every UTF-8 string is
representable and `<unk>` is never emitted. That property is what makes the
model safe to pair with llama.cpp's `gpt2` tokenizer implementation: no
normalisation, no unknown-token path, identical ids on both sides.

Two export details matter for numerical parity and are easy to get wrong:

- **RoPE layout.** This implementation uses the HF `rotate_half` convention
  (pairs `x[i]` with `x[i + d/2]`), while ggml's `rope_norm` pairs `x[2i]` with
  `x[2i+1]`. `tools/export_gguf.py` permutes `attn_q`/`attn_k` weights on export
  (`permute_rope_weight`). Without it the GGUF still loads and generates, but its
  logits only correlate with PyTorch at cosine 0.982 instead of 0.999999.
- **Merges.** GGUF stores merges as `"left right"` token strings, not id pairs.

## 4. What is deliberately not in the model

- No pre-trained weights, no distillation, no fine-tuning of an existing model.
- No biases, no dropout, no inference-time hacks.
- No keyword filter or content classifier (see `model/policy.py`).

## 5. Files

| File | Contents |
|---|---|
| `model/config.py` | `ModelConfig`, analytic parameter accounting, KV-cache sizing |
| `model/attention.py` | `RotaryEmbedding`, `Attention` (GQA + causal SDPA) |
| `model/transformer_block.py` | `RMSNorm`, `SwiGLUMLP`, `TransformerBlock` |
| `model/embeddings.py` | `TokenEmbedding`, `LMHead` (tied) |
| `model/architecture.py` | `MyGPT`: forward, KV-cache generation, save/load |
| `model/io.py` | checkpoint save/load, tied-weight deduplication |
| `model/policy.py` | configurable behaviour policy (prompt level, no keyword filter) |
