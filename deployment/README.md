# Running MyGPT on the laptop

Short version: copy **one file** to the laptop (`mygpt-0.5b-q4_k_m.gguf`, 292 MiB),
copy the llama.cpp Windows binary next to it, and run one command. No Python, no
GPU driver, no internet connection, no account.

## 1. What you need

| Item | Where it comes from |
|---|---|
| `mygpt-0.5b-q4_k_m.gguf` (292 MiB) | release asset: https://github.com/yadavapanulu143/MyGPT/releases/download/v0.1.0-stage1/mygpt-0.5b-q4_k_m.gguf (sha256 `02f1da92760a8feee593ab25cad922c67609a0fe1c468765e992ba8106bbc265`) |
| llama.cpp Windows build | https://github.com/ggml-org/llama.cpp/releases/latest - download the asset named `llama-bNNNNN-bin-win-cpu-x64.zip` (~24 MiB). The x64 CPU build dispatches AVX2 at runtime, which is what this CPU needs. Do **not** pick cuda/rocm/sycl/vulkan builds. |
| ~1 GB free disk | model + binary |

Optional: `mygpt-0.5b-q5_k_m.gguf` (339 MiB) if you want a little more quality, or
`mygpt-0.5b-f16.gguf` (947 MiB) for the unquantized reference.

## 2. Install (Windows 11)

1. Create a folder, e.g. `C:\MyGPT`.
2. Unzip `llama-bNNNNN-bin-win-cpu-x64.zip` into `C:\MyGPT\llama\`. You should see
   `llama-cli.exe`, `llama-server.exe`, `llama-bench.exe` and a bunch of `.dll` files.
3. Download `mygpt-0.5b-q4_k_m.gguf` into `C:\MyGPT\` (link in the table above), then
   check it arrived intact:

   ```powershell
   Get-FileHash .\mygpt-0.5b-q4_k_m.gguf -Algorithm SHA256
   # must print 02F1DA92760A8FEEE593AB25CAD922C67609A0FE1C468765E992BA8106BBC265
   ```

## 3. Run

Open PowerShell in `C:\MyGPT` (this is the whole thing - no Python, no installers):

```powershell
cd C:\MyGPT
.\llama\llama-cli.exe -m mygpt-0.5b-q4_k_m.gguf -t 2 -c 2048 -n 256 --temp 0.8 --top-p 0.95 -p "Hello, how are you?"
```

Flags that matter on this laptop:

| Flag | Why |
|---|---|
| `-t 2` | one thread per physical core; SMT gives little for this workload and can hurt |
| `-c 2048` | context window; `-c 1024` halves KV-cache RAM (41 MiB instead of 82 MiB) |
| `-n 256` | maximum tokens to generate |
| `--temp 0.8 --top-p 0.95` | sane sampling for conversation |

Interactive chat mode: drop `-p` and the CLI reads from stdin, or run
`llama-cli -m ... -cnv` on builds that support it. A browser UI is available
through the server:

```powershell
.\llama\llama-server.exe -m mygpt-0.5b-q4_k_m.gguf -t 2 -c 2048 --host 127.0.0.1 --port 8080
```

then open http://127.0.0.1:8080. It works fully offline; nothing leaves the
machine. Note that this model has no chat template yet, so the web UI will send
the prompt as plain completion rather than as a chat turn.

## 4. Check the speed on your own machine

```powershell
.\llama\llama-bench.exe -m mygpt-0.5b-q4_k_m.gguf -p 64,256 -n 32,128 -t 2
```

Expect roughly 8-15 tokens/s generation and 50-80 tokens/s prompt processing
(estimates in `docs/memory_analysis.md`; measure to replace them with facts).
If it is much slower, check that Windows is not throttling the CPU on battery and
that `-t 2` is used rather than `-t 4`.

## 5. Current state of the model

The weights in this repository are **randomly initialized**. The model loads and
generates tokens at the speeds above, but it has no language ability yet - the
output is noise. It becomes useful only after the training stage described in
`docs/training_plan.md`. The same command works unchanged once a trained
checkpoint is exported with `tools/export_gguf.py`.

## 6. Alternative: PyTorch on the laptop

Possible but 4-5x slower and needs ~2.1 GB of RAM:

```powershell
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
py -m pip install safetensors
py scripts\test_inference.py --checkpoint checkpoints\mygpt-0.5b-init --max-new-tokens 16 --threads 2
```

This path is for verification, not daily use.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown model architecture` | the GGUF was built for a different llama.cpp version; use a release from 2025 or later |
| `failed to load model ... invalid magic` | the download was truncated; compare the SHA256 above and re-download |
| very slow (< 3 t/s) | check `-t 2`, close background apps, plug in the charger (battery mode limits clocks) |
| out of memory | use `-c 1024` or `-c 512`; Q4_K_M itself only needs ~330 MiB |
| garbled output | expected: the model is untrained |
| `llama-server` port busy | add `--port 8081` |
