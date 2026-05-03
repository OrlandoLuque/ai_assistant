# V109 — Phase A.3 (iter 2): local-inference CLI bin + auditor pair + smoke test

**Date**: 2026-05-03
**Version**: 0.2.56
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § A.3
**Tasks**: #316 (bin + auditor), #317 (smoke test)

## Why

V108 shipped the in-process inference **trait surface** (`Backend`,
`LocalInferenceConfig`, `vram::clamp_gpu_layers`, `StubBackend`,
`SloRecord`) so the API would stabilize before any native dep
(Candle / llama-cpp-2) landed. V109 closes the loop: a CLI binary
exercises the trait end-to-end, a dedicated auditor pair surfaces
SLO compliance per the `feedback_auditable_subsystems` rule, and
an integration smoke test gated by an env var becomes meaningful
the moment #319 / #314 wire real backends.

Net effect: the entire local-inference subsystem now has a
caller-visible surface and an auditable trail, even though only
the StubBackend can actually serve generations.

## What

### `src/bin/ai_local_infer.rs` (`--features local-inference`)

| Verb | Effect |
|---|---|
| `info` | Backend availability + best-effort `nvidia-smi` VRAM detection |
| `generate [opts]` | Single prompt, streams to stdout, persists `SloRecord` JSONL |
| `bench [opts] [--iters N]` | Repeat generate N times, per-iter + aggregate summary |

Honors every `LocalInferenceConfig` field via flags
(`--backend`, `--model`, `--ctx-size`, `--n-gpu-layers`, `--no-clamp`,
`--max-tokens`, `--temperature`, `--top-p`, `--prompt`, `--log-dir`).

Default log dir: `.ai_assistant/local_infer_logs/`. JSONL filename
encodes Unix timestamp + PID, matching the `ai_acp serve` convention
so a single GUI auditor can browse mixed runs.

### `src/bin/ai_local_infer_audit.rs` (`--features local-inference`)

CLI auditor mirroring `ai_acp_audit`:

| Verb | Effect |
|---|---|
| `list [--dir D]` | List discovered logs, per-file record + backend count |
| `show <FILE>` | Pretty-print every record (backend, load, first chunk, total, tokens, tps) |
| `audit [--dir D] [--strict]` | Aggregate SLO check; non-zero exit on breach if `--strict` |

SLO budgets:

- `load_ms < 30000` — model load + backend init
- `first_chunk_ms < 1000` — TTFT (production target)
- `tokens_per_sec ≥ 5` — CPU-only baseline; GPU backends should easily clear

### `src/bin/ai_local_infer_audit_gui.rs` (`--features gui-local-inference`)

egui visual auditor with file list + per-record table (red-coded
breaches in `load_ms`, `first_chunk_ms`, `tokens_per_sec`) + summary
panel. Read-only. Default dir matches the CLI default.

### `tests/local_inference_smoke.rs`

Integration test with four cases:

| Test | Effect |
|---|---|
| `stub_backend_full_roundtrip` | Drives StubBackend through the trait, validates `SloRecord` round-trips |
| `vram_detection_returns_consistent_shape` | Best-effort: asserts `free <= total` if any GPU reported |
| `vram_clamp_policy_under_realistic_inputs` | Llama-shaped numbers (4 GiB/32 layers, edge cases) |
| `tiny_model_smoke` | Gated by `AI_LOCAL_INFER_TINY_MODEL`; skips silently when unset |

The tiny-model case is the **forward-compatible** layer: when #319
(Candle CPU) or #314 (llama-cpp-2 GGUF) lands, exporting
`AI_LOCAL_INFER_TINY_MODEL=path/to/tinyllama.gguf` flips this test
into a real generation smoke without any test-side change. SLO
budgets in the gated case are intentionally lenient
(`first_chunk < 5 s`, `tps ≥ 1`) to accommodate CPU-only runs of
quantized tiny models.

## Cargo.toml additions

```toml
gui-local-inference = ["local-inference", "dep:eframe"]

[[bin]]
name = "ai_local_infer"
required-features = ["local-inference"]

[[bin]]
name = "ai_local_infer_audit"
required-features = ["local-inference"]

[[bin]]
name = "ai_local_infer_audit_gui"
required-features = ["gui-local-inference"]
```

All three are `bench = false` per the project-wide rule (462f3c4).

## Smoke

```text
$ ai_local_infer info
ai_local_infer — backend availability
  stub         available (always)
  candle       not compiled in (#319 — local-inference-candle)
  llama-cpp-2  not compiled in (#314 — local-inference-llama-cpp)

VRAM detection (best-effort, NVIDIA only)
  total: 16376 MiB
  free:  13682 MiB

$ ai_local_infer generate --backend stub --prompt "smoke test prompt"
[stub] smoke test prompt
--- SLO record ---
  backend: stub  load_ms: 0  first_chunk_ms: 0  total_ms: 0
  prompt_tokens: 3  generated_tokens: 4  tokens_per_sec: 61255.7

$ ai_local_infer bench --backend stub --iters 3
... 3 iters ...

$ ai_local_infer_audit audit --dir /tmp/li_smoke --strict
Local-inference audit (...)
  Files: 2  Records: 4  load breaches: 0  first_chunk breaches: 0  tps breaches: 0
OK: all records within SLO targets
```

Integration tests: 4 / 4 pass (the `tiny_model_smoke` case
silently skips when `AI_LOCAL_INFER_TINY_MODEL` is unset).

## Lessons

- **The auditor pair is cheaper to land than the real backend.** With
  the CLI + JSONL flow in place, when Candle / llama-cpp-2 actually
  arrives the only thing left is the model loop — observability is
  already taken care of.
- **Forward-compatible env-var test is a free win.** The
  `tiny_model_smoke` test costs nothing today but starts asserting
  real generation the moment #319 lands. No test-side maintenance.
- **Audit GUI shape is reusable.** The third near-identical egui
  auditor (after `ai_logs_gui` and `ai_acp_audit_gui`) suggests there's
  a generic `SloAuditApp<R>` that could collapse all three. Not done
  here — the file shapes diverge enough (e.g. `acp::SloRecord` has
  `session_id`, `local_inference::SloRecord` has `n_gpu_layers_*`)
  that templating today would just shift the per-field wiring inside
  a trait. Worth revisiting once a fourth auditor wants to ship.

## Next

Iter 3 (#319): real Candle CPU backend behind sub-feature
`local-inference-candle`. Iter 4 (#314): llama-cpp-2 GGUF backend
behind `local-inference-llama-cpp`, with pinned exact version + ABI
smoke test. Both will pull in significant native deps (~100 MB of
crates each), so they're paywalled behind sub-features so default
builds stay lean.
