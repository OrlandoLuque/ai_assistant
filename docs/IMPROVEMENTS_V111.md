# V111 — Phase A.3 (iter 4): Candle GGUF support

**Date**: 2026-05-05
**Version**: 0.2.58
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § A.3
**Tasks**: #320 (candle GGUF — extends the existing local-inference-candle sub-feature)

## Why

V110 shipped real CPU Llama inference via candle, but only for HuggingFace
safetensors directories. The dominant on-disk format for local LLMs in 2026
is **GGUF** (single-file quantized format used by llama.cpp, Ollama, LM Studio,
text-gen-webui). Without GGUF support, users had to re-download `.safetensors`
versions of models they already had as `.gguf`, paying 2-4x the disk cost and
losing quantization (Q4_K_M, Q5_K_M, …).

V111 closes that gap by extending the existing `local-inference-candle`
sub-feature to handle both formats from the same `BackendKind::Candle`. **No
new deps**: candle 0.10 already includes `candle_core::quantized::gguf_file`
and `candle_transformers::models::quantized_llama`. Selection is fully
automatic — the loader looks at the path and dispatches.

This is the **single-GPU / single-stream** GGUF backend. Multi-agent
in-process throughput on one GPU and tensor-split across multiple GPUs
remain reserved for V112 (`local-inference-llama-cpp`).

## What

### Auto-detect by path

`load_candle()` now dispatches based on `cfg.model_path`:

| Path shape | Loader |
|---|---|
| `*.gguf` (file) | `load_gguf` → `quantized_llama::ModelWeights` |
| directory | `load_safetensors_dir` → `Llama` (V110, unchanged) |
| neither | `BackendError::Backend` with explanatory message |

Public surface unchanged: callers still pass a `LocalInferenceConfig` with
`BackendKind::Candle` and a `model_path`. No new enum variants, no new
sub-feature.

### `load_gguf` contract

1. `cfg.model_path` must be a readable `.gguf` file.
2. A sibling `tokenizer.json` must exist in the **same directory**. Reason:
   candle 0.10's `quantized_llama` doesn't decode the GGUF metadata
   tokenizer (the BPE merges / vocab embedded inside). The standard pattern
   is to copy `tokenizer.json` from the original HF source repo alongside
   the `.gguf` file. Ollama / LM Studio do this implicitly inside their
   model stores; standalone GGUF downloads (e.g. from TheBloke) need it
   explicitly.
3. EOS id is read best-effort from the GGUF metadata key
   `tokenizer.ggml.eos_token_id`. Missing → no early stop on EOS, generation
   runs until `max_tokens` or a `stop` string.
4. `gguf_file::Content::read` parses the metadata header. Then
   `QuantizedLlama::from_gguf(content, &mut file, &device)` builds the
   model with weights kept in their original quantization (Q4_K_M, Q5_K_M,
   IQ2_XS, …).

### Internal model enum

`CandleBackend` now holds:

```rust
enum LoadedModel {
    Safetensors { model: Llama, cache: Cache },
    Gguf { model: QuantizedLlama },
}
```

The KV cache is **external** for safetensors (`candle_transformers::models::llama::Cache`)
and **internal** for GGUF (`QuantizedLlama` manages it). A small `forward`
adapter on `LoadedModel` papers over the difference so `generate()` is
identical for both formats:

```rust
fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
    match self {
        Self::Safetensors { model, cache } => model.forward(input, index_pos, cache),
        Self::Gguf { model } => model.forward(input, index_pos),
    }
}
```

`generate()` itself (tokenize → loop → LogitsProcessor → incremental
UTF-8-safe decode → EOS / stop-string check) is unchanged.

### Tests

`tests/local_inference_smoke.rs::tiny_model_smoke` is already path-agnostic:
point `AI_LOCAL_INFER_TINY_MODEL` at a `.gguf` file (or a safetensors dir)
and the test runs the same SLO assertions. No test-side change.

## How to use

```bash
# GGUF — point at the file. tokenizer.json must be in the same directory.
ai_local_infer generate \
    --backend candle \
    --model ../TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --prompt "The capital of France is" \
    --max-tokens 32

# Safetensors directory — unchanged from V110.
ai_local_infer generate \
    --backend candle \
    --model ../TinyLlama-1.1B-Chat-v1.0 \
    --prompt "..."
```

## What's next

- **V112 / #314** — `local-inference-llama-cpp` sub-feature. Adds
  `BackendKind::LlamaCpp` real impl with **continuous batching** (multi-agent
  in-process: N concurrent sequences sharing one model load) and
  **tensor split** (multi-GPU). Blocked on LLVM (libclang) install for the
  `bindgen` step in `llama-cpp-sys-2`.
- candle CUDA / Metal — separate sub-features once the CPU baseline for
  both safetensors and GGUF is documented and audited.

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.57 → 0.2.58 |
| `src/local_inference_candle.rs` | refactor: dispatch `load_candle` by path; add `load_gguf`; add `LoadedModel` enum; tweak doc-header |
| `CHANGELOG.md` | V111 entry |
| `docs/IMPROVEMENTS_V111.md` | this file |
