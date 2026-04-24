# Runtime Comparison — Ollama / LM Studio / llama.cpp / vLLM

This document exists to answer one question: *given what I'm actually
doing, which of these four runtimes should I use?*

The short answer is encoded in `Butler::recommend_runtime`. This
document explains the reasoning so you can override the butler when
you know something it doesn't.

---

## The workloads we consider

| Workload              | Example                                 | Characteristic                      |
|-----------------------|------------------------------------------|-------------------------------------|
| Interactive chat      | single user, one prompt at a time        | Latency matters most                |
| Code assist (IDE)     | 1–3 concurrent autocomplete requests     | Short prompts, medium frequency     |
| Agentic coding        | Aider/Cline-style loops, self-editing    | Long prompts, persistent sessions   |
| Research pipeline     | RAG over documents, synthesis            | Long context, sequential queries    |
| Multi-agent           | N agents coordinating on one model       | Parallel requests, long contexts    |
| Eval batch            | benchmark run over hundreds of prompts   | Fully parallelisable                |
| Autonomous scheduler  | cron-driven batch work                   | Bursty parallel load                |

The first two are **latency-bound**. The last five are
**throughput-bound**.

---

## Speedup table (rough, single-GPU)

Numbers are aggregate-throughput estimates vs. Ollama on the same
hardware. They're order-of-magnitude, not benchmarks — run
`ai_assistant eval_benchmarks` against your own workload for
exact numbers.

| Workload              | Ollama | LM Studio | llama.cpp | vLLM      |
|-----------------------|--------|-----------|-----------|-----------|
| Interactive chat      | 1x     | 1x        | 1x        | 1x        |
| Code assist (IDE)     | 1x     | 1x        | 1x        | 1–1.5x    |
| Agentic coding        | 1x     | 1x        | 1x        | **2–3x**  |
| Research pipeline     | 1x     | 1x        | 1x        | **2–4x**  |
| Multi-agent (4+)      | 1x     | 1x        | 1.1x      | **4–10x** |
| Eval batch (≥100)     | 1x     | 1x        | 1.1x      | **5–10x** |
| Autonomous scheduler  | 1x     | 1x        | 1x        | **3–6x**  |

The vLLM wins come from continuous batching: new requests fuse into
the in-flight batch instead of queueing. Ollama, LM Studio, and
llama.cpp process requests sequentially by default.

---

## When each runtime wins

### Ollama

- **Ergonomics.** `ollama pull <model>` handles discovery, download,
  quantization choice, and loading in one command.
- **Baseline performance.** For single-user interactive chat, it's
  within noise of every other option.
- **Cold-start penalty.** Switching models takes seconds. Ollama
  hot-swaps better than the others for interactive browsing.

### LM Studio

- **GUI model browser.** The best UX for trying quantizations side
  by side without touching a terminal.
- **Model management.** Visual VRAM estimator before loading.

### llama.cpp

- **CPU performance.** Best CPU inference of the four.
- **Apple Silicon.** Metal offload is a first-class target.
- **Exotic quants.** `Q1_0` / ternary kernels on the PrismML fork
  (Bonsai models).
- **Tight control.** Direct `-ngl`, `-c`, `-t` flags.

### vLLM

- **PagedAttention.** KV-cache fragmentation gone. More concurrent
  sequences fit.
- **Continuous batching.** Requests fuse into the in-flight batch.
- **Tensor parallelism.** `--tensor-parallel-size N` shards a large
  model across N GPUs.
- **LoRA hot-swap.** `--enable-lora` + `/v1/load_lora_adapter` swaps
  adapters without restarting.

---

## When each runtime *loses*

### Ollama

- **Concurrent load.** Sequential request handling. A second request
  waits for the first to finish.
- **Large context.** KV-cache fragmentation limits how many long
  prompts you can keep resident.

### LM Studio

- **Automation.** The GUI is a feature for interactive use and a
  liability for unattended pipelines.
- **Not available everywhere.** No Linux server install — macOS /
  Windows GUI only.

### llama.cpp

- **Raw GPU throughput.** Under concurrent load, vLLM wins. The
  llama.cpp server processes requests sequentially.
- **vRAM efficiency.** Without PagedAttention, KV-cache fragmentation
  wastes GPU memory on long-context multi-agent workloads.

### vLLM

- **Windows.** No native build. WSL2 or Docker only.
- **macOS.** Experimental. Apple Silicon users: stick with llama.cpp.
- **Cold start.** Slower to load a model than Ollama. If you swap
  models often, the overhead dominates.
- **Non-GGUF.** Loads from HuggingFace repos, not local GGUF files.
  If your workflow is already GGUF-centric, switching costs effort.
- **Gated repos.** Needs `HF_TOKEN` for many production-grade models
  (`meta-llama/*`, some `mistralai/*`).

---

## Decision tree (what the butler uses)

```
workload is multi-agent / eval / autonomous / agentic / research?
├── GPU available?
│   ├── yes → vLLM  (install hint: pip install vllm)
│   └── no  → llama.cpp  (caveat: vLLM would be 5-10x faster with a GPU)
│
workload is interactive chat or code assist?
├── Ollama already running?     → Ollama
├── LM Studio already running?  → LM Studio
├── llama.cpp already running?  → llama.cpp
├── vLLM already running?       → vLLM
└── nothing running             → Ollama  (install hint: ollama.com)

workload is Auto (unknown)?
├── GPU + vLLM running?  → vLLM
├── Ollama running?      → Ollama
├── llama.cpp running?   → llama.cpp
├── LM Studio running?   → LM Studio
└── nothing running      → Ollama
```

Every branch returns a fallback in case the primary isn't installed,
plus caveats surfaced in `RuntimeRecommendation.caveats` (e.g.
"vLLM does not support Windows natively").

---

## Running more than one at once

Nothing stops you. Each runtime listens on a different default port
(11434 / 1234 / 8080 / 8000). You can have Ollama serving chat on
11434 while vLLM serves a multi-agent pipeline on 8000.

The provider is per-`AiConfig`, so different agents or sessions can
point at different runtimes. `Butler::recommend_runtime` is called
once per workload — pass the right `WorkloadHint` and the butler
picks the right port.

---

## See also

- [RUNTIMES_INSTALL.md](RUNTIMES_INSTALL.md) — per-OS install
  instructions.
- [IMPROVEMENTS_V103.md](IMPROVEMENTS_V103.md) — rationale and test
  coverage for the V103 additions.
