# V79 — C FFI bindings (Proposal 5 foundation)

> **Version**: 0.2.10 → 0.2.11
> **Primary driver**: NPCs in video games (Proposal 5)
> **Status**: complete — 24 unit tests + 5 cross-crate tests green

## Context

V78 hardened `ai_proxy` into a production API gateway. V78.1 added the
gateway e2e integration tests. The next gap was **native consumption
from non-Rust languages**: the 410 K LOC library was pure Rust with zero
`extern "C"`, zero `cdylib`, zero `cbindgen`.

Three live proposals depended on closing this gap:

- **Proposal 5 (primary)**: NPCs in video games — Unity, Unreal, and
  Bevy need a native C ABI to drive dialogue and agent behavior.
  C# P/Invoke and C ABI are the baseline.
- **Proposal 6**: P2P team pod — needs the same FFI surface to
  expose a thin client API to other languages.
- **General**: the crate becomes consumable from C, C++, C#, Python
  (via `ctypes`), Lua, JavaScript (via N-API), and any language with
  a C FFI bridge.

V79 is intentionally **Tier 1**: create a handle, configure provider
& model, send one message (blocking), stream tokens via callback,
clear the conversation, release resources. Tier 2 (RAG indexing,
memory, cancellation, MCP tool-calling, async FFI) is deferred.

## Workstreams

### WS-1. Cargo.toml scaffolding

- Version bump `0.2.10 → 0.2.11` (patch).
- Added `[lib] crate-type = ["rlib", "cdylib", "staticlib"]`. `rlib`
  keeps the 20 existing binaries building; `cdylib` produces the
  `.so` / `.dylib` / `.dll`; `staticlib` produces the `.a` / `.lib`.
- Added zero-dep `ffi = []` feature.
- Added `cbindgen = "0.27"` to `[build-dependencies]`.

### WS-2. `build.rs`

Extended the existing Windows-icon-embedding build script with a
`generate_ffi_header()` function that:

1. Re-runs on `CARGO_FEATURE_FFI` env toggles.
2. Fast-path returns when `ffi` is off.
3. Detects `release` + `panic = "abort"` and emits a `cargo:warning`
   (because `catch_unwind` is a no-op in that combo).
4. Creates `include/` if missing.
5. Reads `cbindgen.toml` and writes `include/ai_assistant.h`.
6. Treats all failures as warnings — a broken cbindgen run never
   blocks a regular build.

### WS-3. `cbindgen.toml`

- `language = "C"`, `style = "both"`, `cpp_compat = true`.
- `include_guard = "AI_ASSISTANT_H"`, `pragma_once = true`.
- `[enum] rename_variants = "ScreamingSnakeCase"` +
  `prefix_with_name = true` → `AI_PROVIDER_KIND_OLLAMA`, ...
- `[parse] parse_deps = false`, **no `parse.expand`** — hand-written
  FFI, no macros, saves ~10-15 s per FFI build.
- `[export] item_types = ["functions", "globals", "enums", "structs"]`
  — excludes cross-crate `pub const` pollution from the header.

### WS-4. `src/ffi.rs` — the module

**~1,100 LOC** (including tests) covering:

1. **Type defs**: `AiAssistantHandle` opaque, `AiProviderKind` flat
   enum (17 variants), 9 return-code statics, `Inner` struct with
   `UnsafeCell<AiAssistant>` + `Cell<AiProviderKind>` +
   `RefCell<Option<String>>` for data-bearing provider variants +
   debug-only `AtomicU64 owner_thread`.
2. **Helpers**: `set_last_error`, `clear_last_error`,
   `handle_to_inner`, `cstr_to_str`, `panic_message`, `guard`
   (panic boundary), `current_thread_id` (thread-pin),
   `check_thread`, `build_provider` (exhaustive match over 17
   variants), `make_handle`.
3. **Lifecycle**: `ai_assistant_new`, `ai_assistant_new_with_prompt`,
   `ai_assistant_free`.
4. **Configuration** (9 setters): `set_system_prompt`,
   `set_provider`, `set_openai_compatible_url`, `set_bedrock_region`,
   `set_model`, `set_api_key`, `set_ollama_url`, `set_temperature`
   (strict reject on NaN/±Inf/out-of-range), `set_max_history`.
5. **Messaging**: `ai_assistant_send_message` (blocking, `#[cfg]`
   dispatch between `generate_sync(msg, "")` and
   `generate_sync_with_rag(msg)`), `ai_assistant_send_message_stream`
   (callback loop over `poll_response` with 10 ms sleep, 60 000
   iteration cap).
6. **Session**: `ai_assistant_clear_conversation`,
   `ai_assistant_new_session`.
7. **Errors/memory**: `ai_assistant_free_string`,
   `ai_assistant_last_error`, `ai_assistant_version`,
   `ai_assistant_abi_version`.
8. **Tests**: 24 unit + 3 ignored (see WS-5).

`src/lib.rs` wires the module with:

```rust
#[cfg(feature = "ffi")]
pub mod ffi;
```

### WS-5. Unit tests in `src/ffi.rs`

24 automated + 3 ignored tests, all in `#[cfg(test)] mod tests`:

1. `test_new_and_free_roundtrip`
2. `test_free_null_is_safe`
3. `test_new_with_prompt_ok`
4. `test_new_with_prompt_null_returns_null_and_sets_error`
5. `test_set_system_prompt_happy_path`
6. `test_set_system_prompt_null_prompt_returns_err`
7. `test_set_system_prompt_invalid_utf8` (crafted `0xFF` byte)
8. `test_set_provider_every_unit_variant` (15 unit variants)
9. `test_set_provider_openai_compatible_orderings` (3 orderings)
10. `test_set_provider_bedrock_orderings` (2 orderings)
11. `test_set_model_happy_path`
12. `test_set_api_key_happy_path`
13. `test_set_temperature_valid_range` (0.0, 0.7, 1.0, 2.0)
14. `test_set_temperature_rejects_nan`
15. `test_set_temperature_rejects_out_of_range` (-0.1, 2.5, ±Inf)
16. `test_set_max_history_happy_path`
17. `test_clear_conversation_happy_path`
18. `test_new_session_happy_path`
19. `test_free_string_null_is_safe`
20. `test_last_error_null_after_success`
21. `test_last_error_thread_local_isolation` (2-thread barrier)
22. `test_abi_version_is_one`
23. `test_version_returns_static_nonnull`
24. `test_wrong_thread_use_panics` (debug-only)
25. `#[ignore] test_send_message_live_ollama`
26. `#[ignore] test_send_message_stream_live_ollama`
27. `#[ignore] test_double_free_is_documented_not_safe` (doc-only)

### WS-6. Cross-crate integration test

`tests/ffi_integration.rs` — 5 tests that exercise the `extern "C"`
symbols in a **separate compilation unit**. Catches missing `pub use`
re-exports that in-module tests would miss.

### WS-7. FFI examples (C, Python, Node.js, Java)

Four language examples demonstrate the full lifecycle (create → configure
→ send → stream → free):

| Language | Directory | Bridge | Build step? |
|----------|-----------|--------|-------------|
| **C** | `examples/ffi_c/` | Direct linking | `gcc` / `cl` |
| **Python** | `examples/ffi_python/` | `ctypes` (stdlib) | None |
| **Node.js** | `examples/ffi_node/` | `koffi` (pure-JS) | `npm install` |
| **Java** | `examples/ffi_java/` | JNA | `javac` + JNA jar |

Each example includes a README with per-platform instructions and
troubleshooting.

### WS-8. Documentation

- `docs/FFI.md` (350+ LOC) — full API reference with threading,
  memory, error, security, and build sections.
- `docs/BINARIES.md` — new "Library artifacts" section with
  per-platform output table.
- `docs/USE_CASES.md` — new case #9 "NPCs in games via FFI".
- `CHANGELOG.md` — v35 entry.
- `ai_assistant-website/binaries.html` — matching library artifacts
  section.
- `ai_assistant-website/use_cases.html` — matching case #9.

## API surface — 20 entry points

| Entry point | Purpose |
|-------------|---------|
| `ai_assistant_new()` | Default config, returns opaque handle |
| `ai_assistant_new_with_prompt(prompt)` | With initial system prompt |
| `ai_assistant_free(handle)` | Release handle (null-safe) |
| `ai_assistant_set_system_prompt(h, prompt)` | Update system prompt |
| `ai_assistant_set_provider(h, kind)` | Set `AiProviderKind` |
| `ai_assistant_set_openai_compatible_url(h, url)` | Companion for `OpenAICompatible` |
| `ai_assistant_set_bedrock_region(h, region)` | Companion for `Bedrock` |
| `ai_assistant_set_model(h, model)` | Update `selected_model` |
| `ai_assistant_set_api_key(h, key)` | Update `api_key` |
| `ai_assistant_set_ollama_url(h, url)` | Update `ollama_url` |
| `ai_assistant_set_temperature(h, f32)` | Update `temperature` (0.0-2.0, strict reject) |
| `ai_assistant_set_max_history(h, usize)` | Update `max_history_messages` |
| `ai_assistant_send_message(h, prompt, out)` | Blocking; `#[cfg]` dispatch on `rag` feature |
| `ai_assistant_send_message_stream(h, prompt, cb, ud)` | Callback streaming |
| `ai_assistant_clear_conversation(h)` | Wipe history |
| `ai_assistant_new_session(h)` | Start fresh session |
| `ai_assistant_free_string(s)` | Free Rust-allocated C string |
| `ai_assistant_last_error()` | Thread-local error message (borrowed) |
| `ai_assistant_version()` | Static `const char*` — Cargo package version |
| `ai_assistant_abi_version()` | Returns `1` — bumps on breaking ABI |

## Design decisions

### Handle pattern: `UnsafeCell` + single-thread contract

Rejected `Mutex<AiAssistant>` because `AiAssistant` holds ~100 fields
including many feature-gated trait objects and third-party types
(`RagDb`, `MemoryManager`, `EventBus`, `BrowserSession`,
`Scheduler`, `MultiLayerGraph`, ...) whose `Send` status is not
universally guaranteed. The standard FFI pattern (SQLite non-
serialized, libcurl, libsodium) is single-thread contract + unsafe
impl, which sidesteps the `Send` requirement entirely.

Debug builds enforce the contract via an `AtomicU64` thread-pin with
`compare_exchange(0, id, AcqRel, Acquire)`. Release builds compile
the pin out for zero overhead.

### Data-bearing provider variants

`AiProvider::OpenAICompatible { base_url }` and
`AiProvider::Bedrock { region }` cannot be expressed in a flat C
enum. V79 exposes `AiProviderKind` with 17 unit values and adds
companion setters:

```c
ai_assistant_set_provider(h, AI_PROVIDER_KIND_OPEN_AI_COMPATIBLE);
ai_assistant_set_openai_compatible_url(h, "http://localhost:1234/v1");
```

The order doesn't matter because the `AiProvider` value is built
**lazily** inside `send_message` via an exhaustive match on
`AiProviderKind`. Adding a new Rust variant causes a compile error
in `src/ffi.rs`, preventing silent ABI drift.

### Panic-strategy caveat

The default `release` Cargo profile uses `panic = "abort"`, which
makes `catch_unwind` a no-op. FFI consumers almost certainly want
`--profile release-fast` (`panic = "unwind"`). `build.rs` emits a
`cargo:warning` when it detects the dangerous combo.

### `rag` feature dispatch

`generate_sync_with_rag` is `#[cfg(feature = "rag")]`-gated, so the
FFI cannot unconditionally call it. The dispatch is a two-branch
`#[cfg]` inside `ai_assistant_send_message`:

```rust
#[cfg(feature = "rag")]
let result = a.generate_sync_with_rag(msg);
#[cfg(not(feature = "rag"))]
let result = a.generate_sync(msg, "");
```

This keeps `ffi = []` zero-dep and enables RAG auto-context only
when the downstream consumer builds `--features "ffi,rag"`.

### Temperature: strict reject, no clamp

`set_temperature` rejects NaN, ±Inf, and any finite value outside
`[0.0, 2.0]` with `AI_ERR_INTERNAL`. Clamping was considered and
rejected: it hides caller bugs. Reject-only is more predictable and
consistent with NaN handling.

## Security summary — 21 mitigations

| #  | Risk                                  | Mitigation                                            |
|----|---------------------------------------|-------------------------------------------------------|
| 1  | Null handle deref                     | `handle_to_inner` checks on every entry               |
| 2  | Null C string arg                     | `cstr_to_str` checks on every string input            |
| 3  | Invalid UTF-8 input                   | `CStr::to_str()` → `AI_ERR_INVALID_UTF8`              |
| 4  | Use-after-free (handle)               | Documented single-owner contract                      |
| 5  | Double-free (handle)                  | `ai_assistant_free` null-safe                         |
| 6  | Double-free (string)                  | `ai_assistant_free_string` null-safe                  |
| 7  | Panic crossing FFI                    | `catch_unwind` on every entry                         |
| 8  | `panic = "abort"` trap                | `build.rs` warning; docs recommend `release-fast`     |
| 9  | Sync primitive poisoning              | `AI_ERR_POISONED` reserved                            |
| 10 | Data race on last-error               | `thread_local!` — per-thread, no sharing              |
| 11 | Unaligned handle ptr                  | `Box::into_raw` guarantees alignment                  |
| 12 | ABI drift on provider additions       | Exhaustive match in `build_provider`                  |
| 13 | Output string buffer issues           | `CString::into_raw` NUL-terminates                    |
| 14 | Callback re-entrancy deadlock         | Documented                                            |
| 15 | Callback panic UB                     | Not caught (user responsibility)                      |
| 16 | Unchecked `f32` NaN/Inf               | Strict reject on `set_temperature`                    |
| 17 | Wrong-thread handle use               | Debug-only `AtomicU64` pin                            |
| 18 | Aliasing mutable borrow               | `UnsafeCell` + single-thread contract                 |
| 19 | Header drift from source              | Committed `include/ai_assistant.h`, auto-regenerated  |
| 20 | `#[non_exhaustive]` match caveat      | Documented: `ffi.rs` is a module, not sibling crate   |
| 21 | `rag` feature dispatch                | `#[cfg]` branch selects `generate_sync` variant       |

## Files changed

| File                                         | Delta          | Type |
|----------------------------------------------|----------------|------|
| `Cargo.toml`                                 | +15 / -1       | EDIT |
| `build.rs`                                   | +70            | EDIT |
| `cbindgen.toml`                              | +42            | NEW  |
| `src/lib.rs`                                 | +7             | EDIT |
| `src/ffi.rs`                                 | +1,100         | NEW  |
| `tests/ffi_integration.rs`                   | +70            | NEW  |
| `include/ai_assistant.h`                     | +275 generated | NEW  |
| `examples/ffi_c/main.c`                      | +90            | NEW  |
| `examples/ffi_c/README.md`                   | +115           | NEW  |
| `examples/ffi_python/main.py`                | +185           | NEW  |
| `examples/ffi_python/README.md`              | +75            | NEW  |
| `examples/ffi_node/index.js`                 | +175           | NEW  |
| `examples/ffi_node/package.json`             | +12            | NEW  |
| `examples/ffi_node/README.md`                | +90            | NEW  |
| `examples/ffi_java/AiAssistantDemo.java`     | +165           | NEW  |
| `examples/ffi_java/README.md`                | +100           | NEW  |
| `docs/FFI.md`                                | +370           | NEW  |
| `docs/IMPROVEMENTS_V79.md`                   | +this file     | NEW  |
| `docs/BINARIES.md`                           | +50            | EDIT |
| `docs/USE_CASES.md`                          | +80            | EDIT |
| `CHANGELOG.md`                               | +90            | EDIT |
| `ai_assistant-website/binaries.html`         | +50            | EDIT |
| `ai_assistant-website/use_cases.html`        | +80            | EDIT |

## Verification

```bash
# 1. Baseline — default build (full feature set) still works
cargo check

# 2. FFI feature matrix
cargo check --features ffi
cargo check --features "full,ffi"

# 3. Unit tests (24 + 3 ignored)
cargo test --features ffi --lib ffi::

# 4. Cross-crate integration tests (5)
cargo test --features ffi --test ffi_integration

# 5. Full test suite (regression guard)
cargo test

# 6. Generated header idempotency
cargo build --features ffi
git diff --exit-code include/ai_assistant.h

# 7. Live smoke (requires Ollama)
cargo test --features ffi --lib -- --ignored ffi::tests::test_send_message_live_ollama
```

## Deferred to V79.1+

- RAG document indexing from C
- Memory system enable/disable from C
- Fallback provider chain from C
- Cancellation tokens
- MCP tool-calling with C callback registration
- PyO3 Python bindings
- Unity / Unreal / Bevy sample projects
- Async FFI (tokio-style callbacks)
- C# P/Invoke sample project
- CI header-drift check
- Valgrind / ASAN integration
- C++ RAII wrapper headers

## Stats

- **LOC delta**: ~2,400 (code + docs), of which ~1,100 is
  `src/ffi.rs` itself (including tests) and ~1,300 is
  documentation + examples.
- **Tests added**: 32 (24 unit + 5 cross-crate + 3 ignored).
- **New runtime deps**: 0.
- **New build-deps**: 1 (`cbindgen 0.27`).
- **Feature combos verified**: `ffi`, `full,ffi` (and `ffi,rag` via
  compile-check — runtime testing is on the CI ignored list).
- **Version**: 0.2.10 → 0.2.11 (patch).
