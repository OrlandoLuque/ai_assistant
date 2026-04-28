# V90.26 — Multimodal projector (mmproj) support

**Version**: 0.2.50 → 0.2.51
**Feature flag**: `vision` (additive — no new flag)
**Date**: 2026-04-28

## Why

`llama.cpp` / `llama-server` / `koboldcpp` enable vision by loading a
**multimodal projector** (`mmproj-*.gguf`) at server startup, separately
from the base LLM. Until V90.26 the library:

* had no concept of a projector path,
* could not tell whether a running `llama-server` actually had one
  loaded, and
* surfaced cryptic upstream errors when callers sent vision requests to
  a server started without `--mmproj`.

V90.26 closes that gap with a small, additive surface area. The library
still does **not** spawn `llama-server` itself — that remains an
operator concern — but it now validates the user's configured path,
detects projector status from `/props`, and produces actionable error
messages when something is missing.

## Architectural notes

mmproj is a **server-startup** concern, not a per-request concern. The
HTTP API of every supported runtime (`llama-server`, LM Studio,
KoboldCpp) loads the projector at boot; there is no `mmproj_path` field
in the OpenAI-compat chat-completions schema. Consequently:

* `AiConfig.mmproj_path` is a **hint**: it persists the operator's
  intent and feeds the CLI/GUI; it is never injected into outgoing HTTP
  bodies.
* Detection is a **runtime probe** of `/props`, parsed best-effort
  because field names vary across forks.

Validation is intentionally non-fatal at config load: a stale path in
a config file should not stop the assistant from running text-only
requests.

## New surface

| Symbol | File | Purpose |
|--------|------|---------|
| `MultimodalProjector` | `src/mmproj.rs` | Validated handle (path, size, GGUF-magic flag) |
| `MmprojValidationError` | `src/mmproj.rs` | Typed validation error |
| `MIN_PROJECTOR_BYTES` (1 MiB), `GGUF_MAGIC` (`0x47475546`) | `src/mmproj.rs` | Constants |
| `AiConfig.mmproj_path: Option<PathBuf>` | `src/config.rs` | User-configured path |
| `AiConfig::validated_mmproj()` | `src/config.rs` | Lazy validation entry point |
| `LlamaCppCapability.multimodal: Option<bool>` | `src/llamacpp_capability.rs` | Projector-loaded status |
| `agent_bridge::vision_runtime_ready_for(config, capability)` | `src/vision.rs` | Combined transport + model + mmproj gate |
| `providers::looks_like_mmproj_error` | `src/providers.rs` | Error mapper |
| `cmd_vision_check` | `src/bin/ai_cli.rs` | CLI pre-flight |

## Validation pipeline

`MultimodalProjector::from_path` performs:

1. Reject `..` components **before** canonicalize (defense-in-depth
   against symlink-race substitution).
2. `canonicalize` to an absolute path.
3. Reject directories / device files.
4. Reject anything below `MIN_PROJECTOR_BYTES` (1 MiB) — real
   projectors are 100 MB – 2 GB.
5. Read the first **4 bytes only** and check the GGUF magic. Bounded
   read so a device file or pipe can't hang the validator.

The handle stores only `path`, `size_bytes`, and a `gguf_validated`
flag. SHA-256 / full content reads are intentionally not exposed in v1
(no new dependency, no surprise I/O).

## Multimodal detection in `/props`

`detect_multimodal(body)` accepts any of these as "projector loaded":

* `multimodal: bool == true`
* `has_clip: bool == true`
* `mmproj_loaded: bool == true`
* `mmproj: <non-empty string>`
* `clip_model: <non-empty string>`
* `clip_model_path: <non-empty string>`

Reachable server with **none** of those fields → `Some(false)`. No
probe yet → `None`. The boolean tri-state lets
`vision_runtime_ready_for` distinguish "explicitly absent" from "we
just don't know".

## Error mapping

`generate_openai_compat_response_with_images` now wraps upstream
failures whose body contains any of `mmproj`, `multimodal`, `no clip`,
`clip model`, `clip not loaded`, `vision model not loaded`,
`no vision`, `image input not supported` (case-insensitive). The new
message reads:

```
vision request rejected by server (`<base_url>`): looks like the
multimodal projector is not loaded. Start `llama-server` /
`llama.cpp` with `--mmproj <projector.gguf>` matching the base
model, or load a multimodal preset in LM Studio. Original: <upstream>
```

False positives only soften the surface message; they do not hide real
failures.

## CLI

```
ai_cli vision-check --provider llamacpp --model llava \
  --url http://localhost:8080 --mmproj /models/llava/mmproj.gguf
```

Output (text mode):

```
vision-check
  provider              : LlamaCpp
  model                 : llava
  transport supported   : yes
  model in known set    : yes
  mmproj                : OK (mmproj.gguf)
  llama-server projector: loaded
```

Exit codes: `0` all green, `2` any gate failed, `1` argument error.
`--json` emits a structured object with the same fields.

## Test additions

* `mmproj::tests` — 8 unit tests (valid, wrong magic, too small,
  missing, directory, traversal, filename hygiene, absolute-path).
* `llamacpp_capability::tests` — 4 new (`multimodal: Some(true)` via
  bool, `Some(true)` via path, `Some(false)` no fields, `Some(false)`
  empty path).
* `vision::agent_bridge::tests` — 4 new for `vision_runtime_ready_for`.
* `providers::mmproj_error_tests` — 5 new for the error-string
  classifier.
* `tests/mmproj_integration.rs` — 11 cross-module integration tests.

Total: **32 new vision-gated tests**.

## Out of scope

* Spawning `llama-server --mmproj <path>` ourselves. Embedded launcher
  is a separate, larger feature.
* KoboldCpp vision dispatch. `vision_supported_for` still excludes it
  pending a dedicated `generate_kobold_response_with_images` path.
* Auto-download of projector files from HuggingFace.
* GGUF tensor-table parsing for dimension-mismatch detection. The
  runtime does the real load-time compatibility check; we only validate
  magic + size.
* SHA-256 / hash storage on the projector handle. Add later if a
  caller needs content-addressing.
