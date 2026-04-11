# IMPROVEMENTS V78 — Gateway Hardening (`ai_proxy`)

## Context

V77 shipped the docs/binaries/UX pass (`ai_jobs`, `ai_cli cost`, `BINARIES.md`,
`USE_CASES.md`). That pass surfaced a real product gap: `ai_proxy` was a
683-LOC round-robin router with optional Bearer auth, but it had **no**
PII filter, no budget enforcement, no audit log, no response cache, and
no rate limiting. Teams wanting to put a shared LLM endpoint behind a
guardrailed gateway had to either bolt middlewares on top or run a
reverse proxy in front.

V78 closes that gap by turning `ai_proxy` into a production gateway. The
**core library is untouched**; every new line lives inside
`src/bin/ai_proxy.rs`. All middleware wiring reuses already-existing
modules (`guardrail_pipeline`, `pii_detection`, `cost_integration`).

### TL;DR

- `ai_proxy`: 683 LOC → ~2,350 LOC (+1,670 LOC)
- 55 unit tests (up from 7), zero new crates (toml / parking_lot /
  libc-on-unix promoted from transitive to direct)
- TOML config file with `deny_unknown_fields`, CLI still works and
  overrides the file
- Full OpenAI-compatible `/v1/chat/completions` path goes through:
  rate limit → content-length guard → PII input guard → toxicity input
  guard → attack guard → budget pre-check → cache lookup → backend →
  PII output guard → toxicity output guard → budget post-update → cache
  store → audit log
- Plain `server-axum` build keeps V77 parity (router-only); the full
  gateway requires `server-axum + security`
- Streaming requests (`stream: true`) and `/v1/embeddings` are passed
  through unmodified in V78 — structured streaming guardrails are
  deferred to V80

---

## Workstreams

### WS-1 — Config file loader

- **File**: `src/bin/ai_proxy.rs` (+~350 LOC), `examples/ai_proxy.toml`
  (new, 92 LOC)
- New flag: `--config <PATH>`. CLI flags still work and **override**
  any field in the file
- `ProxyConfig` parsed with `#[serde(deny_unknown_fields)]` so typos
  fail loud
- Size cap: **1 MiB** enforced in `load_config` to stop a pathological
  config from exhausting memory
- Sections: `[server]`, `[[backends]]`, `[middleware]`, `[audit]`
- Precedence chain applied by `merge_cli_and_config`:
  `defaults → file → AI_PROXY_API_KEY env → CLI flags`
- `--dry-run` now validates the config file, prints the resolved
  middleware flag table, and exits without binding

### WS-3 — Response cache + per-key rate limiter

- **File**: `src/bin/ai_proxy.rs` (+~280 LOC)
- Hand-rolled LRU `ResponseCache` using `DashMap<CacheKey,
  CachedResponse>` plus a `parking_lot::Mutex<VecDeque<CacheKey>>` for
  ordering. **No new crate**.
- `CacheKey { model, temperature_milli, max_tokens, prompt_sha256 }`
  — `temperature` is quantized to `u32` milli-units so the key stays
  hashable (`(temp * 1000.0).round() as u32`)
- `put()` **rejects** any response that came from a request that hit
  the PII pre-filter (`pii_free=false`). Oversize bodies (>1 MiB) are
  also rejected
- `get()` enforces TTL lazily on the read path
- Hand-rolled per-key `KeyRateLimiter` sliding window
  (`DashMap<String, Mutex<VecDeque<Instant>>>`), NOT the library's
  single-global `RateLimitGuard`. Hard cap of 100,000 buckets; a
  cleanup pass drains stale ones
- Bucket key priority: `key:sha256(bearer) → sess:id → ip:addr`

### WS-4 — Audit log writer

- **File**: `src/bin/ai_proxy.rs` (+~280 LOC)
- Append-only JSONL; one entry per request
- `AuditEntry { ts, request_id, client, key_hash, session_id, model,
  status, latency_ms, prompt_sha256, prompt_tokens_est, outcome }`
- `AuditOutcome` is a tagged enum:
  `Ok | Blocked | BudgetBlock | OutputBlocked | CacheHit | Streamed | Error`
- **Symlink-safe open**: Unix uses `libc::O_NOFOLLOW | O_APPEND |
  O_CREAT`; Windows pre-checks via `std::fs::symlink_metadata`
- Rotation by size AND count (`max_files` + `max_bytes`). Shifts
  `audit.jsonl → audit.jsonl.1 → audit.jsonl.2 → …`, drops tail
- `BufWriter` with flush every 16 entries
- API key is never written directly — only SHA-256 hex hash

### WS-5 — Budget middleware wrapper

- **File**: `src/bin/ai_proxy.rs` (+~160 LOC)
- `BudgetGate` wraps `DefaultCostMiddleware` behind a
  `parking_lot::Mutex` (the inner `post_response` takes `&mut self`)
- `pre_request(model, est_tokens)` → `BudgetCheck { Allow | Warn |
  Block }`, mapped from `CostDecision`. `Block` returns 429 with
  `X-Reason: budget-exceeded`
- `post_response(model, tokens_in, tokens_out)` updates the underlying
  `CostDashboard` with backend-reported usage
- `CostAwareConfig` is `#[non_exhaustive]`; built via
  `Default::default()` + field mutation (see `config_from_middleware`)
- `cost_snapshot_path` is stored for a future V78.1 periodic snapshot
  flusher; current V78 keeps cost state in-process only

### WS-2 — Guardrail wiring (request path)

- **File**: `src/bin/ai_proxy.rs` (+~750 LOC)
- `GatewayContext` holds `proxy, pipeline, cache, rate_limiter,
  budget, audit, middleware_cfg`
- `build_gateway_context` constructs the `GuardrailPipeline` with the
  guards whose flags are on:
  - `ContentLengthGuard::new(65_536)` (always)
  - `PiiGuard::new()` (if `enable_pii_input || enable_pii_output`)
  - `ToxicityGuard::new()` (if `enable_toxicity_*`)
  - `AttackGuard::new()` (if `enable_attack_filter`)
- Pipeline sits behind `Arc<parking_lot::Mutex<GuardrailPipeline>>`
  because `check_input`/`check_output` are synchronous `&mut self`
- **Fail-closed on guard panic** is already built into
  `GuardrailPipeline::run_stage` (`src/guardrail_pipeline.rs:183-206`);
  `ai_proxy` does NOT re-wrap with `catch_unwind`
- `build_gateway_router` wires three routes:
  - `GET  /health` → `gateway_health_handler` (same semantics as
    V77 plain router)
  - `POST /v1/chat/completions` → `gateway_chat_handler` (full
    pipeline)
  - Fallback → `gateway_passthrough_handler` (auth + rate limit +
    forward + audit; no guardrails, no cache)
- `gateway_chat_handler` flow:
  1. Auth (`Bearer expected-key`)
  2. Rate limit (`key:/sess:/ip:` priority)
  3. Parse body with `MAX_REQUEST_BODY = 16 MiB` cap
  4. If `stream: true`, passthrough via `forward_core`, audit
     `Streamed`, done
  5. Extract scan text from `messages[]` where role ∈ {user, system}
  6. `pipeline.check_input(&scan_text)` → 403 `Blocked` on fail
  7. `budget.pre_request(model, est_tokens)` → 429 `BudgetBlock` on
     fail; `est_tokens = scan_text.chars().count() / 4`
  8. Cache lookup (respects `Cache-Control: no-cache`)
  9. `forward_core` → upstream (uses the V77 LB / session affinity)
  10. Extract `choices[].message.content` and `usage.*tokens`
  11. `pipeline.check_output(&output_text)` → 503 `OutputBlocked` on
      fail
  12. `budget.post_response(model, usage_in, usage_out)`
  13. Cache store (if `pii_free` and body under 1 MiB)
  14. Audit `Ok`, return with `X-Request-Id` + `X-Cache: MISS|HIT`
- **Main wiring**: `main()` picks between `build_proxy_router` (V77
  parity) and `build_gateway_router` based on
  `any_middleware_enabled(&middleware, &audit)`. The `security`
  feature gate controls whether the gateway branch is compiled at all.

### WS-6 — CLI flags

- **File**: `src/bin/ai_proxy.rs` (+~60 LOC)
- New flags (all optional, all override the config file):
  - `--audit-log <PATH>`
  - `--audit-max-files <N>`
  - `--enable-pii-redaction`
  - `--disable-cache`
  - `--cost-snapshot <PATH>`
- `--api-key` now emits a **deprecation warning** on stderr pointing
  to `AI_PROXY_API_KEY` (keeps working for backward compatibility)
- `print_usage()` updated; the env-variable section documents that
  `AI_PROXY_API_KEY` **wins** over both the config file AND the
  `--api-key` flag

### WS-7 — Tests

- 55 unit tests total (up from 7), all in
  `src/bin/ai_proxy.rs::tests`:
  - 6 V77 baseline tests (preserved as-is)
  - 13 WS-1 tests (parse/load/merge/precedence/size cap)
  - 10 WS-3 tests (cache key stability, TTL, PII rejection, LRU;
    rate limiter allow/reject/independent/cleanup)
  - 5 WS-4 tests (audit hash stability, JSONL round-trip, rotation;
    Unix-only symlink reject is `#[cfg(unix)]`)
  - 3 WS-5 tests (budget disabled/enabled/config build)
  - 3 WS-6 tests (flag parsing, number validation, CLI overrides
    file)
  - 15 WS-7 helper tests (`extract_scan_text` roles, legacy prompt,
    `extract_response_text_and_usage` chat/completion/invalid,
    `pick_rate_limit_key` Bearer/session/IP,
    `extract_client_ip` unknown, `any_middleware_enabled`,
    `build_gateway_context` on/off, `build_gateway_router` smoke)
- Full end-to-end integration tests with a mock upstream backend
  (the 13-case `tests/ai_proxy_gateway.rs` matrix) are **deferred to
  V78.1** — the helpers they would exercise are already covered by
  direct unit tests on the extraction logic.

### WS-8 — Security mitigations

| # | Vector | Mitigation | Where |
|---|---|---|---|
| S-1 | Audit symlink attack | `libc::O_NOFOLLOW` (Unix) / `symlink_metadata` (Windows) | `audit::reject_symlink`, `open_nofollow_append` |
| S-2 | Audit unbounded growth | `max_files` + `max_bytes_per_file` rotation | `audit::AuditWriter::rotate` |
| S-3 | API key leak in logs | Only SHA-256 hex hash is ever persisted | `audit::hash_api_key` |
| S-4 | API key visibility via `ps` | Prefer `AI_PROXY_API_KEY` env; CLI flag warns | `parse_args` `--api-key` branch |
| S-5 | Cache-key collision on float temp | Quantize to `u32` milli-units | `cache::CacheKey.temperature_milli` |
| S-6 | PII leak through cache | `pii_free` flag; `put()` rejects tainted | `cache::ResponseCache::put` |
| S-7 | Middleware panic → silent pass-through | Fail-closed built into `GuardrailPipeline::run_stage` | `guardrail_pipeline.rs:183-206` |
| S-8 | Config file DoS | 1 MiB cap in `load_config` | `MAX_CONFIG_SIZE` |
| S-9 | Oversized body prompt extraction | `MAX_REQUEST_BODY = 16 MiB` cap | `gateway_chat_handler` |
| S-10 | Toxicity bypass via stop tokens | Output pipeline runs after decode | WS-2 step 11 |
| S-11 | Budget race under concurrent load | `Mutex<DefaultCostMiddleware>` | `budget::BudgetGate` |
| S-12 | Log injection via prompt content | Entire entry is `serde_json::to_string` | `audit::AuditWriter::write_entry` |
| S-13 | Unknown TOML fields silently ignored | `#[serde(deny_unknown_fields)]` on all sections | `ProxyConfig` |

---

## Design decisions

- **Single binary, two feature profiles**: `server-axum` alone keeps
  V77 parity; `server-axum + security` activates the full gateway.
  The new middleware code is gated by `#[cfg(feature = "security")]`
  in-place (no new feature flag invented).
- **Streaming policy**: requests with `stream: true` are **passed
  through unmodified in V78**. They bypass the pipeline and the cache,
  and are audited with `outcome: "Streamed"`. Wiring
  `StreamingGuardrailPipeline` into the SSE/WebSocket forward path is
  deferred to V80.
- **Weighted LB deferred**: the TOML schema accepts a `weight` field
  per backend for forward-compatibility, but V78 still uses plain
  round-robin. Weighted RR is tracked for V80.
- **Fail-closed on guard panic**: reused the library's built-in
  `catch_unwind` in `run_stage` — `ai_proxy` does not re-wrap.
- **`on_middleware_error = "block"` default**: malformed bodies / lock
  poisoning return 503 `X-Reason: middleware-error` rather than
  forwarding unchecked.
- **Budget snapshot persistence**: the field is accepted and stored
  but the periodic flusher task is deferred. Current V78 keeps cost
  state in-process only.

---

## API reuse (nothing reimplemented)

| Reused from | How |
|---|---|
| `ai_assistant::guardrail_pipeline::GuardrailPipeline` | Built with `ContentLengthGuard`, `PiiGuard`, `ToxicityGuard`, `AttackGuard`; wrapped in `parking_lot::Mutex` |
| `ai_assistant::cost_integration::DefaultCostMiddleware` | Wrapped in `BudgetGate`; built via `CostAwareConfig::default()` + field mutation (non_exhaustive) |
| `toml = "0.9"` | TOML parser, already transitive, promoted to direct dep via `server-axum` feature |
| `parking_lot = "0.12"` | `Mutex` for pipeline + budget, already transitive |
| `dashmap` | Cache entries and rate-limit buckets, already direct dep |
| `sha2` | Key/prompt hashing; `security = ["dep:sha2"]` made it explicit |
| `uuid` v4 | Request IDs, already direct dep |
| `libc` (Unix only) | `O_NOFOLLOW` for audit log, added under `[target.'cfg(unix)'.dependencies]` |

---

## Files changed

| File | Delta | Type |
|---|---|---|
| `src/bin/ai_proxy.rs` | +~1,670 | EDIT (683 → ~2,350 LOC) |
| `examples/ai_proxy.toml` | +92 | NEW |
| `Cargo.toml` | +6 | EDIT (version 0.2.9 → 0.2.10, `security = ["dep:sha2"]`, `toml`/`parking_lot` optional deps, `libc` target-unix) |
| `src/server_axum.rs` | +4 | EDIT (pre-existing V67 bug: `audio_model_registry` guarded only by `rag`; now `all(rag, audio)`) |
| `docs/IMPROVEMENTS_V78.md` | +this | NEW |
| `docs/BINARIES.md` | +30 | EDIT |
| `docs/USE_CASES.md` | +20 | EDIT (case #4 gateway mention) |
| `CHANGELOG.md` | +60 | EDIT (v34 entry) |
| `ai_assistant-website/binaries.html` | +30 | EDIT |
| `ai_assistant-website/use_cases.html` | +20 | EDIT |

---

## Verification

```bash
# Min and full feature builds
cargo check --bin ai_proxy --features server-axum
cargo check --bin ai_proxy --features "server-axum,security"

# Tests
cargo test --bin ai_proxy --features "server-axum,security"
# → 55 passed

# Dry-run smoke test
cargo run --bin ai_proxy --features "server-axum,security" -- \
  --config examples/ai_proxy.toml --dry-run
# → prints resolved config and middleware flag table, exits 0
```

---

## Deferred to V78.1 / V79+

- Full end-to-end integration tests with a mock upstream backend
  (`tests/ai_proxy_gateway.rs`, 13-case matrix)
- Periodic cost snapshot flusher (atomic temp-file + rename)
- Weighted round-robin LB
- Distributed rate-limiter state sync (Redis / peer gossip)
- Prometheus `/metrics` exposition (currently JSON)
- OIDC / JWT validation
- Dynamic config reload on SIGHUP
- Structured streaming guardrails over SSE/WebSocket
- Native TLS termination via `rustls`

---

## Stats

- **Version**: 0.2.9 → 0.2.10 (patch bump)
- **LOC**: +~1,670 in `src/bin/ai_proxy.rs`, +92 in `examples/ai_proxy.toml`
- **Tests**: 7 → 55 (+48)
- **New crates**: 0 (toml / parking_lot / libc promoted from transitive)
- **Security mitigations**: 13 documented (S-1…S-13)
- **Deprecations**: `--api-key` CLI flag (still works, warns)
