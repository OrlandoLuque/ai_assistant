# V103 — vLLM provider + runtime recommendation (parity with llama.cpp)

**Version:** 0.2.35 → 0.2.36
**Date:** 2026-04-24
**Scope:** Add `AiProvider::VLLM` as a first-class local runtime and wire
it into the Butler so the library can tell the user *which* local
runtime to use for their workload (Ollama / LM Studio / llama.cpp /
vLLM).

V102 finished llama.cpp parity with Ollama. V103 does the same for
vLLM, then steps back and asks the harder question: *given what the
user is actually doing, which of these four runtimes should they pick?*
The answer is encoded in `Butler::recommend_runtime`.

---

## Why vLLM?

vLLM is a GPU-optimised LLM serving engine. Two features make it
dominant for the workloads this library targets:

- **PagedAttention** — KV-cache fragmentation is eliminated, so the
  same GPU holds more concurrent sequences.
- **Continuous batching** — new requests are fused into an in-flight
  batch instead of waiting for the current batch to finish.

For single-user interactive chat, Ollama wins on ergonomics. Once you
have ≥2 concurrent requests (multi-agent, autonomous loops, eval
batches, research pipelines), vLLM typically gives **2-10x higher
aggregate throughput** on the same hardware. V103 puts this
recommendation in the butler.

---

## F1 — `AiProvider::VLLM` + `vllm_url` config field

New variant `AiProvider::VLLM` (all-caps, matches the `LMStudio`
convention for acronyms). OpenAI-compatible endpoints, so it slots
into the existing `fetch_openai_*` codepaths with minimal wiring.

- `display_name()` → `"vLLM"`, `icon()` → `"🚀"`
- Default URL: `http://localhost:8000`
- New `AiConfig::vllm_url: String` field (serde default
  `default_vllm_url()`).
- `UrlConfig` parser accepts `"vllm" | "v_llm" | "v-llm"`.

**Tests:** 5 config-level tests covering default URL, `is_cloud`,
`is_openai_compatible`, display name, and `get_provider_url`.

---

## F2 — `VLlmCapability` probe

New module `src/vllm_capability.rs` (always compiled). Hits
`/v1/models`, `/version`, `/health`, and `/v1/load_lora_adapter`
(OPTIONS) and returns:

```rust
pub struct VLlmCapability {
    pub engine_version: Option<String>,
    pub served_models: Vec<VLlmServedModel>,
    pub healthy: bool,
    pub supports_lora: bool,
}

pub struct VLlmServedModel {
    pub id: String,
    pub owned_by: Option<String>,
    pub max_model_len: Option<u32>,
    pub parent: Option<String>,
}

pub fn probe_vllm(base_url: &str) -> Result<VLlmCapability, String>;
```

LoRA detection uses OPTIONS on `/v1/load_lora_adapter`: 404 → LoRA
disabled, any other response → enabled.

**Tests:** 12. Offline parsing is split from the HTTP probe so tests
never hit the network.

---

## F3 — `HfModelInfo` HuggingFace metadata resolver

New module `src/huggingface.rs` (always compiled). vLLM loads models by
HuggingFace repo ID, so the butler needs to be able to tell the user
ahead of time:

- is the repo gated (needs `HF_TOKEN`)?
- is it private?
- approximate on-disk size (sum of `siblings[].size`)
- pipeline tag (reject non-chat models for vLLM serving)

```rust
pub struct HfModelInfo {
    pub id: String,
    pub pipeline_tag: Option<String>,
    pub gated: bool,
    pub private: bool,
    pub total_size_bytes: Option<u64>,
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

pub fn huggingface_model_info(repo_id: &str) -> Result<HfModelInfo, String>;
```

Handles `gated` as either a bool or a string (`"manual"`, `"auto"`).

**Tests:** 10. Covers public / gated / private / size-sum / embedding
model / tag-only text-generation paths.

---

## F4 — Launch helpers: `vllm_launch_command` + `vllm_docker_command`

New module `src/vllm_launch.rs`. Generates ready-to-paste command
strings; **does not execute anything**.

```rust
pub struct VLlmLaunchConfig {
    pub repo: String,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub tensor_parallel_size: Option<u8>,
    pub max_model_len: Option<u32>,
    pub quantization: Option<String>,   // awq, gptq, fp8, bitsandbytes
    pub dtype: Option<String>,
    pub gpu_memory_utilization: Option<f32>,
    pub enable_lora: bool,
    pub trust_remote_code: bool,
    pub served_api_key: Option<String>,
    pub hf_token_required: bool,
}

pub fn vllm_launch_command(cfg: &VLlmLaunchConfig) -> String;
pub fn vllm_docker_command(cfg: &VLlmLaunchConfig, image: Option<&str>) -> String;
```

Docker command uses `--gpus all`, mounts `~/.cache/huggingface` so
weights persist, and sets `--ipc=host` (vLLM requires it for NCCL
shared memory).

**Tests:** 11.

---

## F5 — 8 new curated vLLM entries

`src/curated_models.rs` extended with HuggingFace repo IDs (vLLM loads
by repo, not local GGUF):

| Model | Quant | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | fp16 | Default chat model |
| `meta-llama/Llama-3.1-8B-Instruct` | fp16 | Gated — needs `HF_TOKEN` |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | AWQ | Fits on 24 GB VRAM |
| `meta-llama/Llama-3.1-70B-Instruct` | fp16 | Gated, tensor-parallel |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | fp16 | Reasoning model |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | fp16 | Coding assistant |
| `neuralmagic/Meta-Llama-3-8B-Instruct-FP8` | FP8 | Needs Hopper/Ada |
| `BAAI/bge-m3` | fp16 | Embedding model (not chat) |

**Tests:** 5 new curated-model tests (catalog non-empty, ids are
HF repo format, coder entry present, embedding entry present, gated
entries flag `hf_token_required`).

---

## F12 — `ai_setup install vllm` + `install llamacpp`

`src/setup/prereq.rs` gets new `check_llamacpp()` + `check_vllm()`
detectors (probe `--version` with fallbacks), plus
`install_llamacpp()` + `install_vllm()` that return per-OS install
instructions:

- **vLLM on Linux:** `pip install vllm` (CUDA 12.1+, Python 3.9–3.12).
- **vLLM on Windows:** WSL2 or Docker (vLLM has no native Windows
  support).
- **vLLM on macOS:** experimental CPU/MPS; llama.cpp is usually faster
  on Apple Silicon.
- **llama.cpp on Windows:** `winget install ggml.llamacpp`.
- **llama.cpp on macOS:** `brew install llama.cpp`.
- **llama.cpp on Linux:** pre-built release download or
  `cmake --build ... -DGGML_CUDA=ON`.

`check_prerequisites()` now returns 7 items (+ llama.cpp, + vLLM).

**Tests:** 3 new prereq tests.

---

## F18 — Butler integration

`src/butler.rs` now ships `VLlmDetector` + `LlamaCppDetector` following
the `OllamaDetector` pattern. Both probe `/v1/models` with a 2-second
timeout. `Butler::with_root` registers them (14 built-in detectors,
up from 12). `Butler::scan` populates `EnvironmentReport.llm_providers`
with any detected vLLM / llama.cpp servers. `Butler::suggest_config`
picks up vLLM before LM Studio when both are available.

### `recommend_runtime(report, workload) -> RuntimeRecommendation`

The headline API. Rule-based, deterministic, never hits the network.
Takes a scanned `EnvironmentReport` and a `WorkloadHint`:

```rust
pub enum WorkloadHint {
    Auto,
    InteractiveChat,
    CodeAssist,
    MultiAgent { concurrent_agents: usize },
    AgenticCoding,
    ResearchPipeline,
    EvalBatch { prompt_count: usize },
    AutonomousScheduler,
}

pub enum RuntimeKind { Ollama, LmStudio, LlamaCpp, VLlm }

pub struct RuntimeRecommendation {
    pub preferred: RuntimeKind,
    pub fallback: Option<RuntimeKind>,
    pub reason: String,
    pub estimated_speedup: String,  // e.g. "2-10x vs. Ollama"
    pub caveats: Vec<String>,
    pub install_hint: Option<String>,
}
```

Decision logic:

- **vLLM** when: GPU present AND (MultiAgent ≥ 2 OR EvalBatch ≥ 20 OR
  AgenticCoding OR ResearchPipeline OR AutonomousScheduler).
- **llama.cpp** when: gpu-heavy workload but no GPU detected (fallback
  path, flagged as a caveat).
- **Ollama** when: InteractiveChat or CodeAssist. Picked by default
  because `ollama pull` is the lowest-friction model management.
- **Whatever is running** on Auto, with a mild preference for vLLM
  when a GPU is detected.

### Advisor rule SC5

`ButlerAdvisor::check_scalability` now fires a `RecommendationPriority::High`
suggestion when a GPU is present, `multi-agent` / `autonomous` /
`agents` features are active, but vLLM is not running.

**Tests:** 12 new butler tests (detector defaults, detector env-var
override, `recommend_runtime` across all workload kinds,
`suggest_config` picking up vLLM / llama.cpp, `RuntimeKind` display).
Existing `test_butler_has_12_detectors` renamed to
`test_butler_has_14_detectors`.

---

## F15 + F19 — CLI

`ai_setup recommend [--workload <kind>]` — new subcommand that scans
the environment and prints a full `RuntimeRecommendation` to the
terminal:

```
ai_setup recommend --workload multi-agent

Recommended runtime: vLLM
  Fallback: llama.cpp

  Reason:
    GPU-bound parallel workload detected. vLLM's continuous batching +
    PagedAttention typically gives 2-10x higher throughput than Ollama/llama.cpp
    once you have multiple concurrent requests. ...

  Speedup: 2-10x vs. Ollama for concurrent requests

  Caveats:
    - vLLM does not support Windows natively — use WSL2 or Docker.

  Install: ai_setup install vllm  # (or: docker run vllm/vllm-openai:latest)
```

Accepted workload kinds: `auto`, `chat`, `code`, `agentic`, `research`,
`multi-agent`, `eval`, `autonomous`.

`ai_setup install` accepts `vllm`, `llamacpp` (alias `llama.cpp`,
`llama-cpp`) in addition to the pre-existing targets.

---

## F6 + F7 — Re-exports

`lib.rs` now re-exports:

```rust
pub use butler::{
    Butler, ButlerAdvisor, /* ... */
    LlamaCppDetector, VLlmDetector,
    RuntimeKind, RuntimeRecommendation, WorkloadHint,
    EnvironmentReport,
};
pub use huggingface::{huggingface_model_info, parse_hf_response, HfModelInfo};
pub use vllm_capability::{
    parse_models_response, parse_version_response, probe_vllm,
    VLlmCapability, VLlmServedModel,
};
pub use vllm_launch::{
    vllm_docker_command, vllm_launch_command, VLlmLaunchConfig,
    DEFAULT_VLLM_DOCKER_IMAGE, DEFAULT_VLLM_PORT,
};
```

The curated-model picker widget (`curated_model_picker`) gets a new
test verifying that vLLM entries are HuggingFace repo IDs.

---

## What V103 does **not** add

- **No new feature flags.** vLLM wiring sits under the existing
  `butler` feature gate for the CLI command; the provider itself is
  always compiled. This follows `feedback_minimize_execution_modes`.
- **No new runtime dependencies.** Everything reuses `ureq`,
  `serde_json`, and the existing retry helper.
- **No execution of install commands.** `install_vllm()` returns
  instructions; `vllm_launch_command()` returns a string. The user
  copies and runs.
- **No vLLM Windows shim.** Windows recommendation is Docker or WSL2 —
  matches upstream's own guidance.

---

## Test count delta

| Module | New tests |
|---|---|
| `config.rs` | +5 |
| `config_file.rs` | +3 |
| `providers.rs` | +1 |
| `vllm_capability.rs` | +12 |
| `huggingface.rs` | +10 |
| `vllm_launch.rs` | +11 |
| `curated_models.rs` | +5 |
| `setup/prereq.rs` | +3 |
| `butler.rs` | +12 |
| `widgets.rs` | +1 |
| **Total** | **+63** |

All pre-existing tests continue to pass. Total library tests go from
~6096 to ~6159.

---

## See also

- [RUNTIMES_INSTALL.md](RUNTIMES_INSTALL.md) — step-by-step install
  guide for Ollama / LM Studio / llama.cpp / vLLM across Linux /
  Windows / macOS.
- [RUNTIMES_COMPARISON.md](RUNTIMES_COMPARISON.md) — workload-by-
  workload comparison with speedup estimates.
