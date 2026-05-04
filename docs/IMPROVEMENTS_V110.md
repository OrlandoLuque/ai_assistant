# V110 — Phase A.3 (iter 3): Candle CPU backend (real impl)

**Date**: 2026-05-03
**Version**: 0.2.57
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § A.3
**Tasks**: #319 (Candle CPU backend real impl)

## Why

V108 defined the `Backend` trait + `StubBackend`. V109 wrapped it in
a CLI bin + auditor pair. Both shipped with `BackendKind::Candle`
returning `BackendError::NotImplemented("candle")` so callers could
detect missing features at runtime. V110 closes that gap: a real
in-process Llama forward pass via `candle-transformers`, gated by
the `local-inference-candle` sub-feature so the default-features
build stays free of native deps.

This is the first time the library can actually generate text from
a HuggingFace Llama-format checkpoint without going through any
external HTTP server (Ollama / LM Studio / cloud). It runs on CPU
only — CUDA / Metal / Accelerate are intentionally deferred until
the CPU baseline is verified end-to-end against TinyLlama 1.1B.

## What

### New sub-feature: `local-inference-candle`

```toml
local-inference-candle = [
    "local-inference",
    "dep:candle-core",
    "dep:candle-nn",
    "dep:candle-transformers",
    "dep:tokenizers",
]
```

Pulls in:

- `candle-core 0.10` (no default features → CPU only, no CUDA/Metal)
- `candle-nn 0.10`
- `candle-transformers 0.10` (provides `models::llama::{Llama, Cache, LlamaConfig}`)
- `tokenizers 0.23` with `["esaxx_fast", "fancy-regex"]` — pure-Rust
  regex backend, no native `onig` dependency

### `src/local_inference_candle.rs` (new, ~180 lines)

Module gated by `#[cfg(feature = "local-inference-candle")]`. Exposes
exactly one entry point:

```rust
pub(crate) fn load_candle(
    cfg: &LocalInferenceConfig,
) -> Result<Box<dyn Backend>, BackendError>
```

Loader contract:

1. `cfg.model_path` must be a directory containing
   `config.json` + `tokenizer.json` + `model.safetensors`.
   Missing files surface as `BackendError::ModelNotFound(path)`.
2. Parses `LlamaConfig` from JSON (Llama / TinyLlama / Mistral-7B share
   this format).
3. Loads tokenizer via `Tokenizer::from_file`.
4. Memory-maps weights via
   `VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)`.
   Forced to F32: candle 0.10 CPU kernels are f32-only; bf16 weights
   get cast at load time.
5. Builds the KV `Cache` and `Llama` model.
6. Extracts EOS token id from `LlamaEosToks::Single` / `Multiple`.

`CandleBackend::generate()` implements the streaming loop:

- Tokenize prompt with `add_special_tokens=true`.
- Build `LogitsProcessor::new(seed=42, Some(temperature), top_p)`.
  When `temperature == 0` it falls through to greedy sampling.
- Loop up to `params.max_tokens`:
  - On step 0, feed the full prompt; on later steps feed only the
    last generated token. The KV cache holds the rest.
  - `model.forward(&input, index_pos, &mut cache)` → squeeze batch
    dim → sample next token → push.
  - Break early on EOS.
  - **Incremental decode**: decode the cumulative `generated_tokens`
    buffer each step, emit only the suffix vs. the previous frame.
    Per-token decoding leaves broken UTF-8 on Llama BPE for
    multi-byte glyphs; this avoids that.
  - Break on any of `params.stop` matching the current decoded tail.

Returns `GenStats { prompt_tokens, generated_tokens, time_ms,
tokens_per_sec, peak_vram_mib: None }`. VRAM is None on CPU.

### Wiring

`src/lib.rs`:

```rust
#[cfg(feature = "local-inference-candle")]
mod local_inference_candle;
```

`src/local_inference.rs` — `load()` factory now dispatches:

```rust
BackendKind::Candle => {
    #[cfg(feature = "local-inference-candle")]
    { crate::local_inference_candle::load_candle(config) }
    #[cfg(not(feature = "local-inference-candle"))]
    { Err(BackendError::NotImplemented("candle")) }
}
```

Default-features build behavior is unchanged — `BackendKind::Candle`
still surfaces `NotImplemented` until the caller opts in.

### Tests

Existing `tests/local_inference_smoke.rs::tiny_model_smoke` becomes
meaningful with no test-side change. Run:

```bash
AI_LOCAL_INFER_TINY_MODEL=../TinyLlama-1.1B-Chat-v1.0 \
AI_LOCAL_INFER_BACKEND=candle \
cargo test --release --features local-inference-candle \
    --test local_inference_smoke -- --nocapture
```

Asserts: `load_ms < 30000`, `first_chunk_ms < 5000` (CPU dev budget;
production target is 1s), `tokens_per_sec >= 1.0`, at least one
streamed chunk, non-zero generated tokens.

The unit test `load_candle_unimplemented` in `local_inference.rs`
already accepts `NotImplemented` OR `ModelNotFound` so it stays
green under both feature configurations.

## How to use

```bash
# Build with candle backend
cargo build --release --features local-inference-candle \
    --bin ai_local_infer

# Generate against a HuggingFace Llama-format directory
./target/release/ai_local_infer generate \
    --backend candle \
    --model ../TinyLlama-1.1B-Chat-v1.0 \
    --prompt "The capital of France is" \
    --max-tokens 32

# Audit the resulting SLO log
./target/release/ai_local_infer_audit audit
```

## What's next

- **#314** — `llama-cpp-2` backend (GGUF) + VRAM auto-clamp end-to-end.
  Mirror this PR's structure: `local_inference_llama_cpp.rs` module
  gated by `local-inference-llama-cpp` sub-feature, dispatched from
  the same `load()` factory.
- **GPU support for candle** — `candle-core` features `cuda` /
  `metal` / `accelerate` exist but require system libs. Add as
  separate sub-features once the CPU baseline is documented and
  audited.

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.56 → 0.2.57; add `local-inference-candle` sub-feature; add candle-core/nn/transformers + tokenizers deps |
| `src/lib.rs` | declare `#[cfg(local-inference-candle)] mod local_inference_candle` |
| `src/local_inference.rs` | dispatch `BackendKind::Candle` to `local_inference_candle::load_candle` |
| `src/local_inference_candle.rs` | new — ~180 lines, real Candle CPU Llama loader + streaming generate |
| `CHANGELOG.md` | V110 entry |
| `docs/IMPROVEMENTS_V110.md` | this file |
