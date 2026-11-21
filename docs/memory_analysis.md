# Memory, storage and performance

Two different machines appear in this document and they are never mixed:

- **Dev machine (measured)**: AMD EPYC 9B14, 4 vCPU (Zen 5, AVX-512), ~15 GiB RAM,
  Debian 13, no GPU. Every number marked *measured* comes from here.
- **Target laptop (estimated)**: AMD Ryzen 3 3250U, 2 cores / 4 threads (Zen+,
  AVX2, no AVX-512), 8 GB RAM with ~5.94 GB usable, Vega 3 integrated graphics,
  SSD, Windows 11. No benchmark here was run on that laptop, so every laptop
  number is an estimate with its derivation shown.

## 1. Model storage by format

| Format | Bytes/param | File size | Notes |
|---|---|---|---|
| fp32 (source of truth) | 4.00 | 1891 MiB / 1.98 GB | training checkpoint, `model.safetensors` |
| fp16 / bf16 | 2.00 | 946 MiB | bf16 needs no support on the laptop, fp16 is fine |
| INT8 (Q8_0) | ~1.06 | 503 MiB | measured, llama.cpp |
| Q5_K_M | ~0.68 | 339 MiB | measured, llama.cpp |
| Q4_K_M | ~0.59 | 292 MiB | measured, llama.cpp - recommended default |
| Q4_0 | ~0.55 | 275 MiB | measured, slightly worse quality than Q4_K_M |

All file sizes are from real artifacts produced in `dist/`:

```
947M  mygpt-0.5b-f16.gguf
504M  mygpt-0.5b-q8_0.gguf
339M  mygpt-0.5b-q5_k_m.gguf
292M  mygpt-0.5b-q4_k_m.gguf
275M  mygpt-0.5b-q4_0.gguf
1.9G  checkpoints/mygpt-0.5b-init/model.safetensors
```

## 2. Runtime memory is not just the weights

For one sequence at context length `C`, KV cache bytes =
`2 * layers * kv_dim * C * dtype_bytes` = `2 * 41 * 256 * C * 2` at fp16.

| Context | KV cache fp16 | KV cache fp32 | Q4_K_M total (weights + cache + ~150 MiB runtime) |
|---|---|---|---|
| 256 | 10 MiB | 21 MiB | ~480 MiB |
| 512 | 21 MiB | 41 MiB | ~490 MiB |
| 1024 | 41 MiB | 82 MiB | ~510 MiB |
| 2048 | 82 MiB | 164 MiB | ~550 MiB |

Measured process RSS with llama.cpp (weights loaded into RAM, no mmap, 32 tokens
generated), which is the honest number for "what will this cost me":

| Model | RSS @ ctx 256 | RSS @ ctx 2048 | Load + 32 tokens |
|---|---|---|---|
| F16 | 977 MiB | 1049 MiB | 2.7 s / 2.4 s |
| Q8_0 | 534 MiB | 606 MiB | 1.4 s / 1.4 s |
| Q5_K_M | 371 MiB | 443 MiB | 0.9 s / 1.0 s |
| Q4_K_M | 324 MiB | 396 MiB | 0.9 s / 1.0 s |
| Q4_0 | 306 MiB | 378 MiB | 1.2 s / 1.1 s |

The PyTorch reference path is heavier, because PyTorch keeps its own allocator
arenas and (for bf16) briefly holds the fp32 tensors during conversion:

| dtype | weights | RSS after load | load time |
|---|---|---|---|
| fp32 | 1891 MiB | 2126 MiB | 4.6 s |
| bf16 | 946 MiB | 2118 MiB (transient fp32 load) | 4.8 s |

## 3. CPU throughput, measured on the dev machine

`llama-bench`, 4 threads, no GPU (t/s, higher is better; pp = prompt processing,
tg = token generation):

| Model | pp64 | pp256 | tg32 | tg128 |
|---|---|---|---|---|
| F16 | 181.1 | 167.5 | 24.6 | 19.5 |
| Q8_0 | 168.4 | 187.3 | 49.7 | 37.5 |
| Q5_K_M | 354.3 | 392.2 | 49.9 | 56.5 |
| Q4_K_M | 282.2 | 266.1 | 65.8 | 67.1 |
| Q4_0 | 274.1 | 234.9 | 68.4 | 62.7 |

Reference PyTorch implementation (4 threads, greedy decoding, 16 tokens):

| dtype | ctx 64 prefill | ctx 64 decode | ctx 256 prefill | ctx 256 decode |
|---|---|---|---|---|
| fp32 | 99.5 t/s | 6.6 t/s | 120.7 t/s | 5.8 t/s |
| bf16 | 253.7 t/s | 14.9 t/s | 211.9 t/s | 12.4 t/s |

Two observations that matter for the laptop:

1. Quantization is what makes this usable on a CPU. Q4_K_M is 3.4x faster than
   F16 at generation and 4x smaller.
2. PyTorch is 4-5x slower than llama.cpp at the same precision (6.6 vs 24.6 t/s
   in fp32/fp16 terms). The laptop deployment therefore uses llama.cpp, not
   PyTorch.

## 4. Laptop estimates (not measurements)

Token generation with a batch of one is memory- and vector-throughput bound:
each token reads every weight once, and does ~2 x 495.8M = 1 GFLOP of work.

Scaling from the dev machine:

- **Vector throughput.** Dev: 4 cores with AVX-512 (2x the FLOP/cycle of AVX2),
  boost ~3.5 GHz. Laptop: 2 cores with AVX2, 2.6 GHz base / ~3.5 GHz boost.
  Ratio is roughly 4-6x less peak vector throughput, and quantized GEMV kernels
  reach a similar fraction of peak on both.
- **Memory bandwidth.** Dev at Q4_K_M ran at 67 t/s x 292 MiB = ~19.6 GB/s
  effective. Laptop DDR4-2400 dual channel peaks at 38.4 GB/s with ~50-65%
  achievable (~19-25 GB/s), so bandwidth alone would not be the binding limit -
  compute is.

Estimated range for the laptop (2 threads, `-c 2048`):

| Model | Generation | Prompt processing | Expected first-token latency (50-token prompt) |
|---|---|---|---|
| Q4_K_M | **8-15 t/s** (central ~11) | 50-80 t/s | ~1 s |
| Q5_K_M | 7-13 t/s | 50-80 t/s | ~1 s |
| Q8_0 | 6-10 t/s | 30-50 t/s | ~1.5 s |
| F16 | 4-6 t/s | 30-50 t/s | ~1.5 s |

Treat these as engineering estimates, not promises. The dev machine is 4-6x
faster per token than the laptop is expected to be; the honest way to know is to
run `llama-bench` on the laptop, which takes under a minute (see
`deployment/deployment_plan.md`).

## 5. Will it fit on the laptop?

Usable RAM is ~5.94 GB, Windows 11 plus background apps typically take 2.5-3.5 GB,
leaving ~2.5-3.4 GB for the model process.

| Choice | Process RSS (est.) | Verdict |
|---|---|---|
| Q4_K_M, ctx 2048 | ~400-550 MiB | comfortable, recommended |
| Q5_K_M, ctx 2048 | ~450-600 MiB | comfortable, slightly better quality |
| Q8_0, ctx 2048 | ~620-800 MiB | fits, ~30% slower |
| F16, ctx 2048 | ~1.1-1.3 GB | fits but slow; only for comparison |

The model is not the constraint on this laptop - the CPU is. Even fp32 PyTorch
(2.1 GB) would fit, which is why the deployment plan can offer a fallback that
needs no quantization at all.

## 6. KV cache growth and long prompts

The KV cache is preallocated to the context size, so a 2048-token context costs
82 MiB even for a 20-token prompt. If RAM is tight, `-c 1024` halves it to 41 MiB
with no measurable quality effect (only shorter conversations). Nothing else in
the runtime grows with prompt length.
