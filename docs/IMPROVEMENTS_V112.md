# V112 — Phase A.3 (iter 5): llama-cpp-2 backend

**Date**: 2026-05-04
**Version**: 0.2.59
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § A.3
**Tasks**: #314 (llama-cpp-2 backend, was blocked on LLVM/libclang install)

## Why

V110 + V111 shipped real CPU inference via candle, including GGUF. That
covers the **single-stream, single-GPU** case. Two capabilities are still
missing:

1. **Continuous batching** — N concurrent sequences sharing one model
   load on a single GPU. This is what makes multi-agent throughput
   reasonable (Ollama's "models stay warm but only one request at a
   time" model is a hard ceiling with N agents).
2. **Tensor-split** across multiple GPUs.

Both live inside llama.cpp itself. The Rust bindings (`llama-cpp-2` over
`llama-cpp-sys-2` / `bindgen`) expose them with the same `BackendKind`
trait surface as candle, so callers pick the backend at config time and
the rest of the library doesn't change.

V112 wires the CPU baseline. The continuous-batching loop and multi-GPU
tensor split are scaffolded (n_seq_max=1, `with_n_gpu_layers` honoured)
and become live the moment the upstream crate is rebuilt with `cuda` /
`metal` features (separate sub-feature, deferred).

## What

### Sub-feature

```toml
local-inference-llama-cpp = [
    "local-inference",
    "dep:llama-cpp-2",
    "dep:encoding_rs",
]
```

Strictly opt-in. Build dep: libclang at compile time (`LIBCLANG_PATH`
or `LLVM\bin` on PATH). The dep itself stays default-features-off so we
don't pull `cuda` / `metal` into the CPU baseline.

### Module: `src/local_inference_llama_cpp.rs`

| Concern | Implementation |
|---|---|
| Backend init | Process-wide `OnceLock<LlamaBackend>` (init errors on second call). |
| GGUF metadata | `GgufContext::from_file(path)` peeked once; `llama.block_count` → total transformer layers (fallback 32 = Llama-3 8B shape). |
| VRAM clamp | V108 `vram::clamp_gpu_layers(model_mib, requested, total_layers, free)` applied end-to-end. Skipped when `requested == 0` (CPU only) or `allow_gpu_clamp = false`. |
| Model load | `LlamaModelParams::default().with_n_gpu_layers(used)` then `LlamaModel::load_from_file(backend, path, &params)`. |
| Context | Per-`generate()` (KV cache is per-context). `LlamaContextParams::default().with_n_ctx(NonZeroU32::new(cfg.ctx_size))`. |
| Prompt | `model.str_to_token(prompt, AddBos::Always)` → single batch, `logits=true` only on the last token. |
| Sampler | `temperature ≤ 0` → `LlamaSampler::greedy()`. Otherwise: `temp(t) → top_p(p, 1) → dist(seed=42)` chained via `chain_simple`. |
| Decode loop | sample → accept → `token_to_piece` (UTF-8 incremental decoder) → check `is_eog_token` / EOS / 64-char tail stop-string → `batch.clear; batch.add(next, pos, &[0], true); ctx.decode`. |
| EOS | `model.token_eos()` *and* `model.is_eog_token(next)` (covers ChatML / Llama-3 special tokens that aren't the canonical EOS). |
| Stops | Tail buffer trimmed to 64 chars at char-boundary. |

### Tests

`tests/local_inference_smoke.rs::tiny_model_smoke` is already backend-agnostic.

```bash
# Drive llama-cpp-2 against a TinyLlama Q4 GGUF
AI_LOCAL_INFER_TINY_MODEL=../TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
AI_LOCAL_INFER_BACKEND=llama-cpp \
cargo test --features local-inference-llama-cpp --test local_inference_smoke -- --nocapture
```

SLO assertions: load < 30s, first-chunk < 5s, ≥1 tok/s on CPU.

## How to use

```bash
# CLI — same shape as candle
ai_local_infer generate \
    --backend llama-cpp \
    --model ../TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --prompt "The capital of France is" \
    --max-tokens 32 \
    --n-gpu-layers 999          # auto-clamps to what fits in free VRAM
```

```rust
// Library
use ai_assistant::local_inference::{load, BackendKind, LocalInferenceConfig, GenParams};

let cfg = LocalInferenceConfig::builder(BackendKind::LlamaCpp, "model.gguf")
    .ctx_size(4096)
    .n_gpu_layers(999)          // clamp will trim if VRAM is tight
    .build();
let mut backend = load(&cfg)?;
backend.generate("Hello", &GenParams::default(), &mut |chunk| {
    print!("{chunk}");
})?;
```

## What's next

- **Continuous batching iteration** — widen `LlamaBatch::new(n_tokens,
  n_seq_max)` past 1, track per-sequence positions, expose a multi-prompt
  variant of `Backend::generate`. Unblocks N-agent in-process throughput.
- **GPU sub-features** — `local-inference-llama-cpp-cuda`,
  `local-inference-llama-cpp-metal`. Re-export upstream features so the
  same module gets GPU offload without code changes here. Tensor split
  across multiple GPUs is then automatic via `LlamaSplitMode::Layer` /
  `Row`.

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.58 → 0.2.59; add `local-inference-llama-cpp` feature; add `llama-cpp-2` + `encoding_rs` optional deps |
| `src/lib.rs` | declare `local_inference_llama_cpp` module behind cfg |
| `src/local_inference.rs` | `load()` dispatches `BackendKind::LlamaCpp` to new module |
| `src/local_inference_llama_cpp.rs` | new — `LlamaCppBackend` impl |
| `src/bin/ai_local_infer.rs` | `cmd_info` reports availability |
| `CHANGELOG.md` | V112 entry |
| `docs/IMPROVEMENTS_V112.md` | this file |
