# Deployment plan: from this repo to the Ryzen 3 laptop

## 1. Runtime decision

| Candidate | Verdict |
|---|---|
| **llama.cpp + GGUF (chosen)** | CPU-first, no CUDA, Q4_K_M supported natively, single binary, mmap or plain RAM load, works offline on Windows. Verified in this project: our GGUF loads, tokenizes and decodes identically to the PyTorch reference. |
| PyTorch (reference implementation) | Kept for development and verification. 4-5x slower, ~2.1 GB RAM, needs Python. Not the deployment path. |
| ONNX Runtime | Viable but no benefit here: llama.cpp already gives quantized CPU kernels, and ONNX INT4 support on CPU is weaker. |
| Ollama | Not required. It wraps llama.cpp; the raw binary is smaller and has no background service. Use it only if you prefer its model management. |
| Cloud/API inference | Rejected by requirement: must work offline. |

## 2. Artifact pipeline (already scripted)

```
checkpoints/mygpt-0.5b-init/model.safetensors   (fp32, 1.85 GiB)
        │  python3 tools/export_gguf.py --dtype f16
        ▼
dist/mygpt-0.5b-f16.gguf                        (947 MiB)
        │  llama-quantize ... Q4_K_M
        ▼
dist/mygpt-0.5b-q4_k_m.gguf                     (292 MiB)  <- ship this
```

Reproduce everything on any Linux/macOS box with a C++ toolchain:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-cli llama-quantize llama-bench

cd /path/to/MyGPT
python3 initialization/initialize_weights.py --out checkpoints/mygpt-0.5b-init
python3 tools/export_gguf.py --checkpoint checkpoints/mygpt-0.5b-init --output dist/mygpt-0.5b-f16.gguf
llama.cpp/build/bin/llama-quantize dist/mygpt-0.5b-f16.gguf dist/mygpt-0.5b-q4_k_m.gguf Q4_K_M
```

## 3. Transfer to the laptop

Pick whichever is convenient:

- **USB drive / external SSD**: copy the `.gguf` file. Simplest and works with no
  network at all.
- **Git LFS or a GitHub release asset**: `git lfs track "*.gguf"`, then attach the
  file to a release. Keep the 947 MiB fp16 file out of git history.
- **Direct download** over your LAN with `python -m http.server` on the dev box.

Do not commit `.gguf` files to the repository; `.gitignore` already excludes them.

## 4. Laptop layout

```
C:\MyGPT\
├── llama\                     llama.cpp Windows release (avx2 build)
│   ├── llama-cli.exe
│   ├── llama-server.exe
│   └── *.dll
└── mygpt-0.5b-q4_k_m.gguf     292 MiB
```

## 5. Offline verification checklist

1. Disconnect the network adapter.
2. `llama-cli.exe -m mygpt-0.5b-q4_k_m.gguf -p "test" -n 8 -t 2` produces tokens.
3. `llama-server.exe ... --host 127.0.0.1 --port 8080`, open the local page, get a reply.
4. Check Task Manager: process working set should be ~400-550 MiB at `-c 2048`.
5. `llama-bench.exe` prints generation and prompt-processing numbers; record them.

## 6. Expected performance on this laptop (estimates)

| Metric | Estimate | Basis |
|---|---|---|
| Load time (Q4_K_M) | 1-3 s from SSD | 292 MiB read + tensor setup; measured 0.9-1.3 s on the dev box |
| RAM, ctx 2048 | 400-550 MiB | measured 396 MiB on the dev box, plus Windows overhead |
| Generation | 8-15 t/s | dev box measured 67 t/s with 4 AVX-512 cores; laptop has ~4-6x less vector throughput |
| Prompt processing | 50-80 t/s | dev box measured 266-282 t/s |
| First token, 50-token prompt | ~1 s | 50 / 60 t/s + overhead |

These are estimates. The dev machine is not the laptop, and no laptop benchmark
was run. `llama-bench` on the laptop replaces them with facts in one minute.

## 7. If the model is later trained

Nothing in the deployment changes. Re-export and re-quantize:

```bash
python3 tools/export_gguf.py --checkpoint checkpoints/mygpt-0.5b-sft --output dist/mygpt-0.5b-f16.gguf
llama-quantize dist/mygpt-0.5b-f16.gguf dist/mygpt-0.5b-q4_k_m.gguf Q4_K_M
```

Copy the new file over the old one. The chat template in
`config/behaviour_policy.json` should be embedded in the tokenizer config at that
point so `llama-server` can apply it automatically.

## 8. What would change the plan

- If the trained model needs > 2048 context, the KV cache grows linearly
  (82 MiB per 2048 tokens at fp16) and `-c` must be raised; Q4_K_M still fits.
- If a GPU is added later, llama.cpp can offload layers (`-ngl`), but the laptop's
  Vega 3 shares system RAM, so CPU inference stays the better choice.
- If quality at Q4_K_M is insufficient after training, ship Q5_K_M (339 MiB, +16%
  size, ~15% slower).
