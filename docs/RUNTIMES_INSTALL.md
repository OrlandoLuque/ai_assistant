# Local Inference Runtimes — Install Guide

ai_assistant works with four first-class local runtimes. All four
speak OpenAI-compatible HTTP, so the difference shows up in
performance characteristics, not API surface.

| Runtime   | Default port | Model format      | Best for                        |
|-----------|--------------|-------------------|---------------------------------|
| Ollama    | 11434        | GGUF (pulled)     | Single-user interactive chat    |
| LM Studio | 1234         | GGUF (GUI-picked) | Exploration with a GUI          |
| llama.cpp | 8080         | GGUF (manual)     | CPU-only, exotic quantizations  |
| vLLM      | 8000         | HuggingFace repo  | GPU multi-agent / batch         |

Pick one with:

```
ai_setup recommend --workload <kind>
```

Then install with:

```
ai_setup install <ollama|vllm|llamacpp>
```

The guide below mirrors what `ai_setup install` prints, with more
context on trade-offs.

---

## Ollama

The easiest local runtime. One command to install, one command to
pull a model. If you don't know what you want, pick this.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve   # (most distros run this automatically via systemd)
```

### macOS

Download the installer from <https://ollama.com/download> and run it.
Creates a menubar app that manages the server.

### Windows

Download the installer from <https://ollama.com/download>. Runs as a
user service — no WSL required.

### Point ai_assistant at Ollama

```toml
provider = "ollama"
selected_model = "llama3.1:8b"
[urls]
ollama = "http://localhost:11434"
```

---

## LM Studio

GUI-based model browser and server. Best if you want to browse
quantizations visually or A/B models side-by-side.

Install from <https://lmstudio.ai/>. After picking a model in the UI,
enable the **Local Server** tab (defaults to port 1234).

### Point ai_assistant at LM Studio

```toml
provider = "lm_studio"
selected_model = "TheBloke/Llama-3.1-8B-Instruct-GGUF"
[urls]
lm_studio = "http://localhost:1234"
```

---

## llama.cpp

The underlying engine behind Ollama and LM Studio, but usable
directly. Pick it when you need:

- Quantizations that Ollama doesn't expose (e.g. `Q1_0` on the
  PrismML fork for Bonsai models).
- CPU-only operation on a machine without a supported GPU.
- Tight control over `-c` (context), `-ngl` (GPU layers), `-t`
  (threads).

### Linux

Pre-built CUDA binaries from <https://github.com/ggml-org/llama.cpp/releases>,
or build from source:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### macOS

```bash
brew install llama.cpp
```

Apple Silicon is a first-class target — Metal offload is on by
default.

### Windows

```powershell
winget install ggml.llamacpp
```

Or download a release zip and add the folder containing
`llama-server.exe` to PATH.

### Run

```bash
llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080
```

### Point ai_assistant at llama.cpp

```toml
provider = "llama_cpp"
selected_model = "my-model"
[urls]
llama_cpp = "http://localhost:8080"
```

---

## vLLM

GPU-optimised serving engine with PagedAttention and continuous
batching. Pick it when you have a CUDA GPU and any of:

- Multi-agent orchestration (≥2 concurrent requests).
- Eval batches over many prompts.
- Long-running autonomous coding loops (Aider/Cline-style).
- Research pipelines with many sequential queries per document.

Typically **2-10x higher throughput** than Ollama/llama.cpp under
concurrent load on the same GPU.

### Linux (native)

**Prerequisites:** NVIDIA GPU with CUDA 12.1+, Python 3.9–3.12.

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

### Docker (recommended for Windows, optional elsewhere)

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

`--ipc=host` is required — vLLM uses NCCL shared memory.

### Windows

vLLM has **no native Windows build**. Two options:

1. **WSL2 + Ubuntu:**
   ```powershell
   wsl --install -d Ubuntu
   # inside WSL:
   pip install vllm
   ```
2. **Docker Desktop** with GPU passthrough (`--gpus all`).

### macOS

vLLM on macOS is experimental (CPU/MPS only). For Apple Silicon,
**llama.cpp is usually faster** — stick with llama.cpp until
upstream Metal support matures.

### Gated repos

Some HuggingFace repos (`meta-llama/*`, some `mistralai/*`) require
license acceptance + an auth token:

```bash
export HF_TOKEN=hf_...
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

`ai_setup recommend` and the curated catalog flag which models are
gated.

### Point ai_assistant at vLLM

```toml
provider = "vllm"
selected_model = "Qwen/Qwen2.5-7B-Instruct"
[urls]
vllm = "http://localhost:8000"
```

---

## Which one? — quick rules of thumb

- **Single-user, interactive:** Ollama. Model management is one
  command.
- **GUI lover / model browser:** LM Studio.
- **No GPU / exotic quants / Apple Silicon:** llama.cpp.
- **GPU + multi-agent / autonomous / eval batch / research pipeline:**
  vLLM. 2-10x throughput win under concurrent load.

When in doubt, run `ai_setup recommend --workload <kind>` and trust
the butler.

---

## See also

- [RUNTIMES_COMPARISON.md](RUNTIMES_COMPARISON.md) — workload-by-
  workload speedup table and when each one wins.
- [IMPROVEMENTS_V103.md](IMPROVEMENTS_V103.md) — design notes for the
  butler runtime recommender.
