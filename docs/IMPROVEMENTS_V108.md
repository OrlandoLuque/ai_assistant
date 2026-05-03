# V108 — Phase A.3 (iter 1): in-process local inference scaffolding

**Date**: 2026-05-03
**Version**: 0.2.55
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § A.3

## Why

Phase A.2 (V107) shipped the **ACP server** so editors can drive
`ai_assistant` as a coding agent. A.3 closes the loop in the other
direction: let `ai_assistant` itself **load and run a model in-process**
instead of always shelling out to a separate `llama-server` / Ollama /
LM Studio process.

In-process inference matters for:

- Single-binary distribution (no llama-server install).
- Latency: skip the HTTP roundtrip for short prompts.
- Resource control: the same process owns the VRAM budget, so we can
  gracefully clamp `n_gpu_layers` instead of crashing on OOM.
- Embedded use cases (CLI tools, batch eval) that don't want the
  operational footprint of a long-running server.

This iteration ships **only the scaffolding** so the API is stable,
testable, and dep-free. Real Candle + llama-cpp-2 backends are queued
as separate tasks that build on this trait surface.

## What

### `src/local_inference.rs` (gated `feature = "local-inference"`)

| Item | Role |
|---|---|
| `BackendKind` enum | `Candle` / `LlamaCpp` / `Stub` — picks the engine |
| `LocalInferenceConfig` + builder | `ctx_size`, `n_gpu_layers`, `allow_gpu_clamp`, `model_size_mib` |
| `GenParams` | `max_tokens`, `temperature`, `top_p`, `stop` |
| `GenStats` | `prompt_tokens`, `generated_tokens`, `time_ms`, `tokens_per_sec`, `peak_vram_mib` |
| `BackendError` | `NotImplemented` / `ModelNotFound` / `Io` / `Backend(String)` |
| `Backend` trait | `generate(prompt, params, on_chunk)` + `kind()` + `unload()` |
| `load(config)` | factory; today returns `Stub` or `NotImplemented` |
| `StubBackend` | echoes prompts so trait + SLO machinery can be tested |
| `SloRecord` | per-generation stats persisted as JSONL by future bins |

### `local_inference::vram` (pure, unit-testable)

- `detect_nvidia_mib() -> Option<(total, free)>` — best-effort
  `nvidia-smi --query-gpu=memory.total,memory.free` shell-out.
- `detect_available_mib() -> Option<u64>` — convenience.
- `clamp_gpu_layers(model_size_mib, requested, total, available) -> u32`
  — pure function. Reduces requested layers proportionally so the
  model fits, with documented edge cases (zero total, zero requested,
  zero VRAM, request above total).

## Architectural decision: not an `AiProvider`

`AiProvider` (in `src/config.rs`) dispatches HTTP to external LLM
endpoints (Ollama, llama-server, OpenAI, etc.). The existing
`embedded_server::LlamaServerConfig` already lives **next to** providers
without being one — it manages a subprocess. `local_inference` follows
the same pattern: it's a direct in-process API, not an enum variant.

Benefits:

- `config.rs` / `providers.rs` stay untouched, so the provider enum
  remains stable and we don't need to gate exhaustive matches on a
  feature flag.
- Callers that need in-process inference invoke
  `local_inference::load(&cfg)` directly. Callers that don't, pay
  zero compile cost (sub-features are off by default).
- Future bins (`ai_local_infer`) and auditors (`ai_local_infer_audit`,
  `_gui`) consume this API without going through the provider system.

## Auto-clamp policy

`clamp_gpu_layers` assumes layer cost is approximately linear in the
total. That holds for transformer architectures where the dominant
per-layer cost is attention + MLP weight memory. The policy:

```
per_layer_mib = model_size_mib / total_layers
max_fittable  = floor(available_mib / per_layer_mib)
result        = min(requested, total_layers, max_fittable)
```

Edge cases:

| Case | Result | Why |
|---|---|---|
| `total_layers == 0` | `requested` | Caller hasn't loaded the model yet — nothing to clamp against |
| `requested == 0` | `0` | CPU only; nothing to do |
| `available_mib == 0` | `0` | No VRAM, force CPU |
| `requested > total_layers` | `min(total, fit)` | Cap to model size first |
| Model fits | `requested` | No clamp needed |

## Test coverage

```
$ cargo test --features local-inference --lib local_inference::
running 14 tests
test local_inference::tests::clamp_fits_returns_requested ... ok
test local_inference::tests::clamp_partial_fit_below_request ... ok
test local_inference::tests::clamp_request_above_total_treated_as_total ... ok
test local_inference::tests::clamp_undersized_reduces ... ok
test local_inference::tests::clamp_zero_requested_returns_zero ... ok
test local_inference::tests::clamp_zero_total_layers_passthrough ... ok
test local_inference::tests::clamp_zero_vram_returns_zero ... ok
test local_inference::tests::config_builder_chains ... ok
test local_inference::tests::config_builder_defaults ... ok
test local_inference::tests::load_candle_unimplemented ... ok
test local_inference::tests::load_missing_real_model_errors ... ok
test local_inference::tests::load_stub_succeeds ... ok
test local_inference::tests::slo_record_serializes ... ok
test local_inference::tests::stub_backend_generates_and_streams ... ok
test result: ok. 14 passed; 0 failed
```

Default-feature build (`cargo check --lib`) also clean — gating works.

## Deferred — follow-up tasks

| # | Task |
|---|---|
| 314 | llama-cpp-2 backend (GGUF) + pinned exact version + auto-clamp E2E |
| 316 | `ai_local_infer` bin + `ai_local_infer_audit` (CLI) + `_audit_gui` |
| 317 | Smoke test gated by tiny-model env var (TinyLlama 1.1B Q4) |
| 319 | Real Candle CPU backend behind `local-inference-candle` sub-feature |

CUDA opt-in (`local-inference-cuda`) is queued behind the Candle
backend — adding it before there's a CPU baseline doesn't shorten
delivery and complicates CI.

## Lessons

- **Keeping the trait surface dep-free pays off**: 14 tests + clean
  build + clean clippy in 2 minutes vs. waiting on candle's compile
  graph. Real backends drop in behind sub-features later without
  re-shaping the API.
- **`StubBackend` is not a hack**: it's the right way to keep downstream
  callers (bins, auditors, integration tests) testable without
  hardware. Same pattern V107 used to verify ACP framing without a
  real LLM.
- **Auto-clamp as a pure function** lets us assert behavior in unit
  tests without GPU. The detection layer (nvidia-smi shell-out) is
  separately mockable.

## Next

Iter 2: Candle CPU backend (task #319). Iter 3: llama-cpp-2 with
pinned version + ABI smoke test (task #314). Iter 4: bins + auditors
(task #316). Each closes one observable user-facing gap.
