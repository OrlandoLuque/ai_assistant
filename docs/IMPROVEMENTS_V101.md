# V101 — llama.cpp provider + curated model catalog (PrismML Bonsai)

**Version**: 0.2.32 → 0.2.33
**Feature flag**: none (core)

V101 elevates `llama.cpp` to a first-class provider and ships a curated
model catalog module that makes the new PrismML Bonsai family
(ultra-compressed 1-bit and ternary Qwen3 derivatives) discoverable by
name instead of only via generic `OpenAICompatible { base_url }`.

## Motivation

PrismML (https://prismml.com/) distributes quantized LLMs — `Bonsai 8B`
with 1.125 bits/weight `Q1_0` quantization (1.16 GB on disk, 8.19B
parameters) and `Ternary Bonsai` with `{-1, 0, 1}` weights. Weights are
shipped as GGUF on Hugging Face (`prism-ml/*`). Running them requires
`llama.cpp`'s `llama-server` — and specifically PrismML's fork
(`github.com/PrismML-Eng/llama.cpp`) because upstream llama.cpp does not
yet ship the `Q1_0` / ternary kernels.

Before V101, a user had to route llama.cpp via the generic
`OpenAICompatible { base_url }` variant, losing provider-level
identification, icon, default URL, and preset entries. V101 fixes that.

## New `AiProvider::LlamaCpp` variant

`src/config.rs`

```rust
pub enum AiProvider {
    // ...
    /// llama.cpp `llama-server` (OpenAI-compatible).
    ///
    /// Works with both upstream llama.cpp and forks such as PrismML's
    /// `PrismML-Eng/llama.cpp` (which adds the custom `Q1_0` quantization
    /// type used by the Bonsai 1-bit models).
    LlamaCpp,
    // ...
}
```

Display name: `llama.cpp`. Icon: 🦫. `is_openai_compatible() == true`.
`is_cloud() == false`. Default URL: `http://localhost:8080` (matches
upstream `llama-server` default).

### AiConfig

- New field `llamacpp_url: String` (default `http://localhost:8080`).
- `get_provider_url(&AiProvider::LlamaCpp)` → `self.llamacpp_url.clone()`.
- Wired through all four existing dispatch paths in `providers.rs`:
  - `generate_openai_response` (blocking)
  - `generate_openai_streaming`
  - `generate_openai_streaming_cancellable`
  - `fetch_model_context_size`
- Reuses the same OpenAI-compatible request path as `LMStudio` —
  llama-server's `/v1/chat/completions` is byte-compatible.

### config_file (TOML / JSON persistence)

- `UrlConfig` gains `llamacpp: String` field.
- String-tag parser accepts `"llamacpp"` / `"llama_cpp"` / `"llama.cpp"` /
  `"llama-cpp"` in the `[provider]` section.
- `to_toml` / `from_ai_config` round-trip the new URL.

## Why not a separate `LlamaCppPrismML` variant?

The PrismML fork's wire protocol is identical to upstream (both expose
the standard `llama-server` OpenAI-compatible API). The difference is
only which build of `llama-server` the user runs — a deployment
concern, not a protocol concern. Adding a second enum variant would
force every match arm to handle both identically. The fork is surfaced
instead via the `requirements` field on curated model entries.

## Curated model catalog — `src/curated_models.rs`

New module, always compiled (no feature flag), providing a short,
hand-picked list of recommended models per provider — the opposite of
a live `/models` fetch.

```rust
pub struct CuratedModel {
    pub provider: AiProvider,
    pub id: &'static str,               // e.g. "Bonsai-8B-Q1_0.gguf"
    pub display_name: &'static str,
    pub description: &'static str,
    pub parameters: &'static str,       // "8B", "4B", "1.7B"
    pub approx_size: &'static str,
    pub quantization: &'static str,     // "Q1_0", "Q4_K_M", ...
    pub source_url: Option<&'static str>,
    pub requirements: Option<&'static str>, // PrismML fork / API key note
}

pub fn suggested_models_for(provider: &AiProvider) -> Vec<CuratedModel>;
pub fn all_curated_models() -> &'static [CuratedModel];
```

### Entries shipped in V101

**llama.cpp**:

| ID | Size | Quant | PrismML-fork required |
|----|------|-------|---------------------|
| `Bonsai-8B-Q1_0.gguf` | 8B | Q1_0 (1.125 bpw) | ✓ |
| `Bonsai-4B-Q1_0.gguf` | 4B | Q1_0 (1.125 bpw) | ✓ |
| `Bonsai-1.7B-Q1_0.gguf` | 1.7B | Q1_0 (1.125 bpw) | ✓ |
| `TernaryBonsai-8B.gguf` | 8B | Ternary | ✓ |
| `TernaryBonsai-4B.gguf` | 4B | Ternary | ✓ |
| `TernaryBonsai-1.7B.gguf` | 1.7B | Ternary | ✓ |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 7B | Q4_K_M | — |
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 8B | Q4_K_M | — |

**Ollama**: `qwen2.5:7b-instruct`, `llama3.1:8b-instruct`,
`mistral:7b-instruct`, `deepseek-coder:6.7b`.

**Cloud anchors** (for UI pickers that want a sensible default):
Anthropic `claude-opus-4-7`, OpenAI `gpt-4o`, Gemini `gemini-2.0-flash`.

The catalog is a `const &[CuratedModel]` — zero runtime cost to read,
extensible by appending literals. It is not a registry: custom /
community models are still valid; the list is only what we recommend
out of the box.

## Public API additions

```rust
pub use config::{AiConfig, AiProvider};   // AiProvider gains LlamaCpp
pub use curated_models::{
    CuratedModel,
    all_curated_models,
    suggested_models_for,
};
```

## Tests

- `config::tests::test_ai_provider_display_names` — asserts
  `LlamaCpp.display_name() == "llama.cpp"`.
- `config::tests::test_ai_provider_openai_compatibility` — `LlamaCpp` is
  marked OpenAI-compatible.
- `config::tests::test_llamacpp_default_url` — default is
  `http://localhost:8080`.
- `config::tests::test_llamacpp_get_provider_url` — URL override works.
- `config::tests::test_llamacpp_not_cloud` — `is_cloud()` = false.
- `curated_models::tests` (6): entries exist for llama.cpp and Ollama,
  Bonsai/Ternary entries declare the PrismML-fork requirement, cloud
  entries mention API keys, no empty fields, catalog non-empty.

**Net new tests: 11.** All passing.

## Not in V101 (possible follow-ups)

- GUI widget / picker that consumes `suggested_models_for()` and
  renders the `requirements` banner inline.
- Auto-downloader for GGUF weights from the `source_url` (would need a
  Hugging Face auth path for gated models).
- A `LlamaCppCapability` probe that hits
  `llama-server`'s `/props` endpoint to detect whether the running
  build supports `Q1_0` (would let us warn users whose build is
  upstream-only when they try to load a Bonsai model).
- An Ollama-side `ollama pull` flow pre-configured for the Bonsai GGUFs
  once they are uploaded as an Ollama Library entry.
