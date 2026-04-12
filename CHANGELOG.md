# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - v35 (2026-04-11) — V79: C FFI bindings

### Added
- **C FFI bindings (V79)** — 20 `extern "C"` entry points wrapping
  `AiAssistant` behind a new zero-dep `ffi` Cargo feature. Enables
  native consumption from C, C++, C#, Unity, Unreal, Bevy, Python
  (via `ctypes`), and any language with a C FFI bridge. Primary
  driver: NPCs in video games (Proposal 5).
  - **Lifecycle**: `ai_assistant_new`, `ai_assistant_new_with_prompt`,
    `ai_assistant_free` (null-safe).
  - **Configuration** (9 setters): system prompt, provider, model,
    API key, Ollama URL, `OpenAICompatible` base URL, Bedrock region,
    temperature (strict-reject NaN/±Inf/out-of-range), max history.
  - **Messaging**: `ai_assistant_send_message` (blocking, wraps
    `generate_sync`; dispatches to `generate_sync_with_rag` when
    `ffi,rag` feature combo is active via `#[cfg]` branch) and
    `ai_assistant_send_message_stream` (callback-based streaming).
  - **Session**: `ai_assistant_clear_conversation`,
    `ai_assistant_new_session`.
  - **Diagnostics**: `ai_assistant_last_error` (thread-local borrowed
    pointer), `ai_assistant_version`, `ai_assistant_abi_version` (ABI=1).
  - **Memory**: `ai_assistant_free_string` (null-safe).
- **Opaque handle with single-thread contract** — SQLite-style
  `UnsafeCell<AiAssistant>` + `unsafe impl Send + Sync`. A debug-only
  `AtomicU64` thread-pin panics on cross-thread use; release builds
  compile the pin out for zero overhead.
- **Panic boundary** — every entry wraps its body in
  `std::panic::catch_unwind` + `AssertUnwindSafe`, stashes the message
  in a thread-local `LAST_ERROR`, and returns `AI_ERR_PANIC` (or NULL
  for pointer-returning functions).
- **Return code enum** — 9 int constants
  (`AI_OK`, `AI_ERR_NULL_PTR`, `AI_ERR_INVALID_UTF8`, `AI_ERR_PANIC`,
  `AI_ERR_POISONED`, `AI_ERR_INTERNAL`, `AI_ERR_UNKNOWN_PROVIDER`,
  `AI_ERR_SEND_FAILED`, `AI_ERR_NO_RESPONSE`).
- **Flat `AiProviderKind` C enum** — 17 unit variants mirroring the
  Rust `AiProvider` positionally. Data-bearing variants
  (`OpenAICompatible`, `Bedrock`) are configured via companion
  setters. The Rust→FFI converter uses an **exhaustive match** so
  adding a Rust variant forces a compile error in `src/ffi.rs`.
- **`build.rs`** — extended from Windows-icon-embedding-only to also
  invoke `cbindgen` and regenerate `include/ai_assistant.h` when
  building with `--features ffi`. Emits a `cargo:warning` on the
  dangerous `release` + `panic=abort` + `ffi` combo. All failures
  degrade to warnings, never panics.
- **`cbindgen.toml`** — new config file at repo root. Restricts
  emitted item types to functions/globals/enums/structs to keep
  cross-crate `pub const` definitions out of the FFI header.
- **FFI examples** in four languages:
  - **C**: `examples/ffi_c/main.c` (~90 LOC NPC-style driver) + README
    with per-platform build instructions and library-naming table.
  - **Python** (ctypes): `examples/ffi_python/main.py` — zero-dep,
    uses the standard library's `ctypes`. Includes blocking + streaming.
  - **Node.js** (koffi): `examples/ffi_node/index.js` — pure-JS FFI
    bridge, no native compilation step. Includes blocking + streaming.
  - **Java** (JNA): `examples/ffi_java/AiAssistantDemo.java` — zero-JNI,
    standard `com.sun.jna` mapping. Includes blocking + streaming.
- **Documentation**:
  - `docs/FFI.md` — 350+ line API reference with threading,
    memory, error, security, and build sections.
  - `docs/IMPROVEMENTS_V79.md` — workstream writeup + 21-row
    security mitigation table.
  - `docs/BINARIES.md` — new "Library artifacts" section listing
    cdylib + staticlib outputs.
  - `docs/USE_CASES.md` — new use case #9 "NPCs in games via FFI".
- **Tests** — 24 automated unit tests in `src/ffi.rs::tests` + 5
  cross-crate integration tests in `tests/ffi_integration.rs` + 3
  ignored live-smoke / documentation tests.

### Changed
- **`[lib] crate-type`** — now `["rlib", "cdylib", "staticlib"]`
  (was implicit `rlib` only). `rlib` keeps the 20 existing binaries
  building; `cdylib` produces the `.so` / `.dylib` / `.dll` shared
  library; `staticlib` produces the `.a` / `.lib` for static linking
  (Unreal prefers this).
- **Version** — `0.2.10` → `0.2.11` (patch bump per
  `feedback_versioning.md`).
- **Added build-dependency** — `cbindgen = "0.27"`. Non-optional so
  `build.rs` doesn't need conditional compilation voodoo; the actual
  invocation is gated inside `build.rs` on `CARGO_FEATURE_FFI`.

### Fixed
- nothing

### Deprecated
- nothing

### Security
- 21 explicit mitigations documented in `docs/FFI.md` and
  `docs/IMPROVEMENTS_V79.md`. Notable additions: debug-only
  thread-pin (S-17), UnsafeCell aliasing contract (S-18), committed
  header (S-19), `non_exhaustive` match caveat (S-20), `rag` feature
  dispatch safety (S-21).

### Stats
- ~1,650 LOC delta across 16 files (`src/ffi.rs` is the bulk at
  ~1,100 LOC including tests)
- +32 tests (24 unit + 5 integration + 3 ignored)
- +1 build-dep (`cbindgen`), 0 new runtime deps
- FFI feature matrix: `ffi` / `ffi,rag` / `full,ffi` — all compile
  and test green

## [Unreleased] - v34 (2026-04-11)

### Added
- **`ai_proxy` gateway hardening (V78)** — turned the 683-LOC round-robin
  router into a production gateway while keeping the core library untouched.
  All new code lives in `src/bin/ai_proxy.rs` and is gated by
  `#[cfg(feature = "security")]` so `--features server-axum` alone keeps V77
  parity (router + health + session affinity only).
  - **TOML config file** via new `--config <PATH>` flag, 1 MiB size cap,
    `#[serde(deny_unknown_fields)]` on every section so typos fail loud.
    Precedence: `defaults → file → AI_PROXY_API_KEY env → CLI flags`.
  - **New example**: `examples/ai_proxy.toml` documenting every section.
  - **Guardrail wiring**: `POST /v1/chat/completions` goes through the full
    pipeline — rate limit → content-length guard → PII input → toxicity input
    → attack guard → budget pre-check → cache lookup → backend → PII output
    → toxicity output → budget post-update → cache store → audit log.
    Streaming (`stream: true`) and `/v1/embeddings` are passed through
    unmodified and flagged in audit.
  - **Per-key sliding-window rate limiter** (`DashMap<String, Mutex<VecDeque<Instant>>>`),
    hand-rolled; key priority `key:sha256(bearer) → sess:id → ip:addr`;
    hard cap of 100,000 buckets with a stale-bucket cleanup pass.
  - **LRU response cache** — hand-rolled over `DashMap` +
    `parking_lot::Mutex<VecDeque>`, no new crate. `CacheKey` quantizes
    `temperature` to `u32` milli-units. `put()` rejects any response that
    came from a PII-tainted request and any body > 1 MiB.
  - **Append-only JSONL audit log** with rotation by size and count. Unix
    opens with `libc::O_NOFOLLOW`, Windows pre-checks `symlink_metadata`.
    API keys are only ever written as SHA-256 hex hash.
  - **Budget enforcement** via `DefaultCostMiddleware` wrapped in a
    `BudgetGate`; `pre_request` returns 429 `X-Reason: budget-exceeded` on
    block, `post_response` updates the cost dashboard with backend-reported
    `usage.prompt_tokens`/`usage.completion_tokens`.
  - **New CLI flags**: `--config`, `--audit-log`, `--audit-max-files`,
    `--enable-pii-redaction`, `--disable-cache`, `--cost-snapshot`.
    `--dry-run` now validates the config and prints the merged middleware
    flag table.
  - **Response headers**: every response now carries `X-Request-Id`; cached
    responses add `X-Cache: HIT|MISS`.
  - **Security**: 13 mitigations documented in `docs/IMPROVEMENTS_V78.md`
    (symlink, log rotation, key-hash-only logs, env-prefers-CLI, float-temp
    quantization, PII cache guard, built-in guard-panic catch, config DoS
    cap, 16 MiB request cap, post-decode toxicity, budget concurrency,
    JSON-escape-safe audit, TOML deny-unknown).
  - **Tests**: 55 unit tests in `ai_proxy` (up from 7), zero new crates
    added. Full end-to-end integration tests with a mock upstream backend
    are deferred to V78.1.
- `docs/IMPROVEMENTS_V78.md` — workstream breakdown, security summary,
  deferred items.

### Changed
- `security` feature now pulls `sha2` explicitly
  (`security = ["dep:sha2"]`) so the audit log and rate-limit key hashing
  are always available with the feature on.
- `server-axum` feature now pulls `toml` and `parking_lot` (both were
  already transitive, promoted to direct deps).
- `libc` added as a Unix-only target dep (`[target.'cfg(unix)'.dependencies]`)
  for `O_NOFOLLOW` on the audit log — no effect on Windows builds.

### Deprecated
- `--api-key` CLI flag — still works, now emits a deprecation warning
  pointing to `AI_PROXY_API_KEY`. The env variable wins over both the
  config file and the CLI flag.

### Fixed
- **Pre-existing V67 regression in `src/server_axum.rs`** surfaced by V78
  feature-gate validation: the `audio_model_registry` call site was only
  guarded by `rag`, but the module itself is `audio`-gated. Tightened to
  `#[cfg(all(feature = "rag", feature = "audio"))]`.

### Stats
- Version bump: 0.2.9 → 0.2.10
- `ai_proxy`: 683 → ~2,350 LOC (+~1,670 LOC)
- 48 new tests (`ai_proxy` 7 → 55)
- 0 new crates
- 13 documented security mitigations

## [Unreleased] - v33 (2026-04-11)

### Added
- **`ai_jobs` binary** (new, ~970 LOC) — cron-like job daemon with two runtime modes:
  - `delegated` *(default)*: shells out to `ai_cli` or any shell command. Always available.
  - `embedded`: runs an in-process `AiAssistant` with access to RAG, tools, memory, and session state. Gated behind `--features full`.
  - Manifest format is **JSON** (parallel schema defined inside the binary so no Serde derives leak into the core `scheduler::*` types).
  - Subcommands: `validate`, `list`, `dry-run`, `run`, `help`.
  - Security: `MAX_JOBS = 1000` cap, per-job `timeout_secs` (default 60s), `std::panic::catch_unwind` guards the daemon, API key env vars referenced by name only.
  - 14 unit tests + 6 integration tests (`tests/ai_jobs_integration.rs`).
- **`ai_cli cost` subcommand** — CLI access to V75 cost intelligence:
  - `cost report [--snapshot <path>]` — formatted dashboard report
  - `cost budget --snapshot <path>` — JSON budget status
  - `cost savings --snapshot <path>` — informational stub (AllocationResult persistence deferred to V78)
  - `cost projection --snapshot <path>` — daily / monthly / per-1k projections
  - `cost export --snapshot <path> --output <file.csv> [--force]` — CSV export (refuses to overwrite without `--force`)
  - 6 new unit tests for the subcommand helpers.
- `examples/jobs.json` — 4-job demo manifest used by the integration tests.
- `docs/BINARIES.md` — authoritative 20-binary catalogue, grouped by role, with feature-flag matrix and per-binary security notes for `ai_jobs`.
- `docs/USE_CASES.md` — 8 end-to-end scenarios wiring multiple binaries (local RAG, CI cost gate, scheduled briefs, TLS team server, distributed cluster, voice assistant, butler bootstrap, MCP backend).
- `docs/IMPROVEMENTS_V77.md` — context, workstream breakdown, deferred items.
- Website pages `ai_assistant-website/binaries.html` and `ai_assistant-website/use_cases.html` — HTML counterparts of the new docs, linked from `index.html`.

### Fixed
- **V76 regressions surfaced by V77 integration tests** — three binaries were missing `required-features` in `Cargo.toml`, so they failed to compile once V76 moved their dependencies behind feature gates:
  - `ai_test_harness`: added `required-features = ["full", "browser"]` (uses `CrawlPolicy`)
  - `ai_virtual_mic_host`: added `required-features = ["audio"]` (uses `group_queue_host`)
  - `ai_gpu_share`: tightened from `["full"]` to `["full", "gpu-sharing"]`

### Stats
- Version bump: 0.2.8 → 0.2.9
- New binary: `ai_jobs` (total: 20)
- ~26 new tests
- 3 latent V76 compile-error regressions fixed

## [Unreleased] - v32 (2026-04-10)

### Changed
- **Feature hygiene**: 15 modules moved behind their rightful Cargo features
  so minimal builds stop compiling hardware- or protocol-specific code:
  - `audio_filter`, `audio_model_registry`, `audio_priority_protocol`,
    `group_queue_host`, `group_queue_runtime` → `audio`
  - `browser_policy`, `crawl_policy` → `browser`
  - `distributed_rag` → `distributed`
  - `video_filter` → `video-io`
  - `wasm`, `wasm_hooks` → `wasm`
  - `gpu_sharing`, `collusion_detection`, `credit_system`, `dynamic_pricing` → `gpu-sharing`
- `mcp_voice_tools` gate tightened from `tools` to `all(tools, audio)` —
  the previous gate was a latent bug that would fail to compile if `tools`
  was enabled without `audio`.
- `voice-agent` feature now implies `audio` in Cargo.toml (was `dep:tokio` only).
- `pub use mcp_voice_tools::register_voice_tools` cfg aligned with the new
  module gate.

### Removed
- `core = []` marker feature — empty, had zero `#[cfg]` references, only
  inflated the feature list. Dropped from `full = [...]`.

### Docs
- `docs/IMPROVEMENTS_V76.md` — full rationale, workstream breakdown, and
  the list of 64 modules deferred to V80.
- `adapters = []` marker now explicitly documented as an intentional label
  for the `adapters_demo` example.

### Stats
- Version bump: 0.2.7 → 0.2.8
- 360+ source modules
- 7,492+ passing tests (no change from v31 — V76 is a compilation-only pass)
- 59 Cargo feature flags (was 60; `core` removed)

## [Unreleased] - v31 (2026-04-09)

### Added
- **Cost Intelligence**: CostDashboard auto-wired in `poll_response()` — automatic cost recording per LLM call
- `with_cost_config()` builder on `AiAssistant` — budget enforcement via `CostAwareConfig`
- Savings estimation in `AllocationResult`: `total_candidate_tokens`, `tokens_saved`, `compression_ratio`, `estimated_cost_saved()`
- Cost projections: `projected_daily_cost()`, `projected_monthly_cost()`, `projected_cost_for_requests()`
- `CostDashboardSnapshot` with `snapshot()` / `restore()` for session persistence (schema versioned)
- 3 MCP tools: `cost_report`, `cost_budget_status`, `cost_savings_summary` (read-only, annotated)
- **Security hardening**: `validate_cost()` (NaN/Infinity/negative → 0.0), `sanitize_csv_field()` (formula injection prevention), `MAX_ENTRIES` cap (100K, evicts oldest)
- Projections section in `format_report()` (daily, monthly, requests/hour)
- 23 new tests (context_budget: 4, cost_integration: 16, assistant: 3)

### Changed
- `CostDashboard::record()` validates cost with `validate_cost()` before storing
- `CostDashboard::export_csv()` sanitizes all fields against CSV formula injection
- `AllocationResult` includes savings metrics in both `build()` and `build_from_items()`

### Security
- S1: CSV injection prevention in `export_csv()` (CRITICAL → mitigated)
- S2: Unbounded entries Vec capped at `MAX_ENTRIES` (HIGH → mitigated)
- S4: Float NaN/Infinity budget bypass via `validate_cost()` (MEDIUM → mitigated)
- S6: Persistence tampering defended by schema version + cost validation on restore
- S7: MCP tools read-only with `read_only_hint: true`, aggregated data only
- S8: Negative pricing clamped in `estimated_cost_saved()`

### Stats
- 360+ source modules
- 7,492+ passing tests (from 7,469 in v74)
- 60 Cargo feature flags
- 0 clippy warnings

## [Unreleased] - v30 (2026-04-09)

### Added
- `ContextBudgetConfig` struct: centralizes all hardcoded allocator values (15 configurable fields)
- `ScoringMode` enum: 4 dynamic scoring modes (Static, Heuristic, LlmEnhanced, Hybrid)
- Intent-based context scoring: maps 16 intent types to per-source score boosts
- Knowledge graph as separate `ContextItem` (extracted from `build_rag_context()`, prevents double-counting)
- `StrategyBandit` wired into production: UCB1 arm selection with utilization reward
- `LlmEnhancerCompressor` bridge: adapts `LlmEnhancer` → `LlmCompressor` with fallback
- `context_scoring_mode` in `RagFeatures`: per-tier scoring mode override
- `arm_to_strategy()` for bandit arm → `OverflowStrategy` conversion
- CI: `FEATURES_STD` / `FEATURES_NETWORK` env vars for standardized feature sets
- CI: Feature-matrix expanded from 19 to 36 combinations
- CI: `cargo audit` security scan job
- CI: Integration tests (`cargo test --test '*'`)
- CI: Binary compilation verification (5 binaries)
- 82 new tests (context_budget: 16 new, total 34)

### Changed
- `build_allocated_context()` uses `ContextBudgetConfig` instead of hardcoded values
- Graph context extracted from `build_rag_context()` to standalone `build_graph_context_string()`
- RAG tier defaults: Enhanced=Heuristic, Thorough/Agentic/Graph=Hybrid(0.6)
- CI coverage aligned with `FEATURES_STD`
- Release pipeline updated: `needs: [check, test, clippy, fmt, binaries]`

### Stats
- 360+ source modules
- 7,469 passing tests (from 7,387 in v73)
- 60 Cargo feature flags
- 0 clippy warnings

## [Unreleased] - v29 (2026-03-06)

### Added
- OpenAI-compatible API: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`
- Full enrichment pipeline: 7 sub-configs, 52 configurable fields
- Selective guardrail pipeline: individual guard toggles, rate limiting, pattern blocking
- Budget manager: daily/monthly/per-request cost limits with HTTP 429
- Output guardrails: configurable PII redaction (per-type toggles) and toxicity filtering
- Butler Advisor: 30 optimization recommendations across 6 categories
- Advanced routing: Thompson Sampling, UCB1, NFA/DFA pipeline, 10 MCP routing tools
- Routing enhancements: composite rewards, per-query preferences, private arms, context-aware routing
- 5 new benchmark suites: LiveCodeBench, AiderPolyglot, TerminalBench, APPS, CodeContests
- RAG tier expansion: 20 → 28 features (discourse chunking, dedup, cascade reranking, etc.)
- 12 MCP tools: 6 config management + 6 evaluation tools
- Unified BPE tokenizer with model-aware routing (GPT, Claude, Gemini, Mistral, DeepSeek)
- Emoticon/emoji detection and sentiment analysis

### Changed
- Token estimation unified across 7 modules → central `crate::context::estimate_tokens`
- `concepts.html` rendering fix for unescaped HTML in code blocks
- `framework_comparison.html` new "Documentation, DX & Economics" category

### Stats
- 220+ source modules
- 6,565+ passing tests (from 6,401 in v28)
- 20+ Cargo feature flags
- 0 clippy warnings

## [0.1.0] - 2026-02-19

### Added

#### Core
- Multi-provider LLM support: Ollama, LM Studio, Kobold, LocalAI, OpenAI, Anthropic, Google Gemini, Mistral AI, HuggingFace Inference, AWS Bedrock
- OpenAI-compatible presets: Groq, Together AI, Fireworks, DeepSeek, vLLM
- Provider auto-discovery with failover and API key rotation
- Context window management with auto-truncation
- Session persistence with journal compaction and snapshots
- Adaptive thinking and response quality analysis

#### RAG & Knowledge
- 5-tier RAG: Self-RAG, CRAG, Graph RAG, RAPTOR, auto-selection
- Vector DB backends: InMemory, Qdrant, LanceDB, Pinecone, Chroma, Milvus, pgvector
- Document parsing: PDF, EPUB, DOCX, ODT, HTML, TXT, CSV, EML, PPTX, XLSX, image metadata
- Knowledge graph with entity/relation extraction
- Embedding-based semantic chunking
- Encrypted knowledge packages (.kpkg) with AES-256-GCM
- Query expansion, citations, and reranking

#### Multi-Agent & Autonomous
- 5-role multi-agent orchestration (Coordinator, Researcher, Analyst, Writer, Reviewer)
- Autonomous agent with 5 autonomy levels and policy-based sandbox
- Task board with undo, priorities, and listener callbacks
- Cron scheduler with event-driven triggers (FileChange, FeedUpdate)
- Butler environment auto-detection
- Chrome DevTools Protocol browser automation
- Distributed agent execution across nodes

#### Security
- RBAC with MFA, CIDR ranges, time windows, and usage limits
- Constitutional AI guardrails and bias detection (8 dimensions)
- Toxicity detection (9 categories) and injection detection (6 types)
- PII detection with 4 redaction strategies
- AES-256-GCM content encryption

#### Streaming & API
- SSE streaming with aggregation and chunking
- WebSocket (RFC 6455) with handshake from scratch
- Resumable streaming with checkpoint/replay
- Stream compression (Deflate, Gzip)
- MCP protocol (2025-03-26 spec) with tool annotations and pagination

#### Distributed Computing
- CRDTs (5 types), DHT (Kademlia), MapReduce with consistent hashing
- QUIC/TLS 1.3 transport with mutual TLS and node security
- Phi-accrual failure detection and Merkle sync
- P2P networking with STUN/UPnP/NAT-PMP and ICE

#### Analytics & Observability
- Prometheus-compatible metrics and flow analysis
- OpenTelemetry integration for traces, spans, and metrics
- Conversation analytics and engagement tracking
- LLM-as-judge evaluation

#### Infrastructure
- Cloud connectors (S3, Google Drive)
- Code sandbox for safe agent execution
- AWS SigV4 authentication for Bedrock
- Binary integrity verification
- WASM support (web-sys, js-sys, wasm-bindgen)
- egui chat widgets

### Stats
- 190+ source modules
- 2010+ passing tests
- 20+ Cargo feature flags
- Zero external service requirements for core functionality
