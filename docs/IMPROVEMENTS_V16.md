# v16 Improvements Changelog

> Date: 2026-02-26

v16 focuses on **zero-test module coverage, production `.unwrap()` cleanup, new feature examples, new benchmarks, and crate-level documentation**. Continues the code quality and completeness work from v12-v15.

---

## Summary Metrics

| Metric | v15 | v16 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,603 | 5,636 | +33 |
| Examples | 47 | 51 | +4 |
| Benchmarks | 34 | 42 | +8 |
| Source files | ~285 | ~285 | 0 |
| Production `.lock().unwrap()` in resumable_streaming | 11 | 0 | -11 |
| Compiler warnings | 0 | 0 | 0 |

---

## Phase 1 — mcp_protocol/v2_oauth.rs Test Coverage (0 → 19)

**Problem**: `v2_oauth.rs` had 580 LOC with ZERO tests — a security-critical OAuth 2.1/PKCE module with no test coverage.

**Fix**: Added 19 tests covering all previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| SHA-256 known vectors | 2 | `sha256()` against NIST test vectors |
| Base64url encoding | 2 | `base64url_encode()`, `base64url_encode_nopad()` |
| PKCE generation | 3 | `generate_code_verifier()` length, `generate_code_challenge()` S256, verify roundtrip |
| OAuth config | 2 | `OAuthConfig` serialization, default values |
| Token management | 4 | `OAuthToken` exchange, refresh, expiry check, serialize |
| Authorization server | 3 | Discovery URL construction, metadata parsing, error handling |
| Dynamic client registration | 2 | Registration request, response parsing |
| State parameter | 1 | State validation for CSRF prevention |

**Files changed**: `src/mcp_protocol/v2_oauth.rs` (+459 lines)

---

## Phase 2 — hallucination_detection.rs Test Coverage Boost (4 → 18)

**Problem**: `hallucination_detection.rs` had 614 LOC with only 4 tests (6.5/1000) — one of the lowest ratios in the codebase.

**Fix**: Added 14 new tests covering all previously untested areas:

| Area | Tests Added | APIs Covered |
|------|-----------|-------------|
| Config/Builder | 2 | `HallucinationConfig::default()`, builder pattern |
| HallucinationType | 1 | All 8 variants `display_name()` |
| Detection (basic) | 4 | Empty input, clean text, context-based, known facts |
| Detection (advanced) | 4 | Contradictions, unsupported claims, invented entities, multiple issues |
| Reliability scoring | 2 | Score range (0.0-1.0), scoring with/without context |
| Edge cases | 1 | Very long input handling |

**Files changed**: `src/hallucination_detection.rs` (+284 lines)

---

## Phase 3 — Production `.unwrap()` Cleanup in resumable_streaming.rs

**Problem**: `resumable_streaming.rs` had 11 production `.lock().unwrap()` calls that would panic on a poisoned mutex.

**Fix**: All 11 calls replaced with `.lock().expect("resumable stream state lock")` providing descriptive context for any panic.

**Files changed**: `src/resumable_streaming.rs` (11 lines changed)

---

## Phase 4 — 4 New Feature Examples (47 → 51)

| # | Example | Feature | Key APIs Demonstrated |
|---|---------|---------|----------------------|
| 1 | `server_tls_demo.rs` | `server-tls` | TlsConfig, load_tls_config, ServerConfig with TLS, AuthConfig, CorsConfig |
| 2 | `vector_lancedb_demo.rs` | `vector-lancedb` | LanceVectorDb, VectorDb trait, insert, search, batch insert, get, delete |
| 3 | `whisper_local_demo.rs` | `whisper-local` | WhisperLocalProvider, SpeechProvider trait, AudioFormat, transcription API |
| 4 | `distributed_network_demo.rs` | `distributed-network` | NetworkNode, NetworkConfig, KV store, consistent hashing, events, join tokens |

Each example: 110-170 lines, demonstrates the full API surface of the feature, no external service dependencies.

**Files changed**: 4 new example files, `Cargo.toml` (+16 lines for `[[example]]` entries)

---

## Phase 5 — 8 New Benchmarks (34 → 42)

| # | Benchmark | What it Measures | Feature Gate |
|---|----------|-----------------|-------------|
| 1 | `guardrail_pipeline_6_guards` | Full pipeline with 6 guards (length+pattern+rate+toxicity+PII+attack) | none |
| 2 | `attack_detect_clean_input` | AttackDetector on benign input | none |
| 3 | `attack_detect_adversarial_input` | AttackDetector on adversarial input | none |
| 4 | `vector_db_search_1k_euclidean_128d` | InMemoryVectorDb search (1000 vectors, 128d, Euclidean) | none |
| 5 | `bpe_token_count_200_words` | BPE tokenization on ~200-word passage | none |
| 6 | `hnsw_build_1k_128d` | HNSW index build from scratch (1000 inserts, 128d) | none |
| 7 | `mapreduce_word_count_20_chunks` | MapReduce word count pipeline (20 data chunks) | `distributed` |
| 8 | `dht_find_closest_1k_nodes` | XOR-distance routing table lookup (1000 nodes) | `distributed` |
| 9 | `crdt_gcounter_merge_100_entries` | GCounter merge with 100 node entries | `distributed` |

Note: 8 benchmark functions producing 10 benchmark test points (2 fns have 2 sub-benchmarks each).

**Files changed**: `benches/core_benchmarks.rs` (+244 lines)

---

## Phase 6 — Minor Doc Fixes

- `src/vector_db_pgvector.rs`: Converted `//` header comments to `//!` inner doc comments for `cargo doc` visibility
- `src/lib.rs`: Updated crate-level `//!` documentation to list all 46 current features in a structured table (was listing only ~14)

**Files changed**: 2 files

---

## Phase 7 — Documentation Updates

- `docs/TESTING.md`: Updated test count 5,603 → 5,636, benchmarks 34 → 42, added v16 history row
- `docs/feature_matrix.html`: v21 → v22, updated all counts
- `docs/framework_comparison.html`: v18 → v19, updated all counts and history
- `docs/IMPROVEMENTS_V16.md`: This file
- `docs/backups/`: Timestamped HTML backups created before editing

---

## Verification

All checks pass:

```
cargo test --lib --features "full,...,devtools,vector-pgvector,cloud-connectors"  → 5636 passed, 0 failed
cargo clippy --features "full,...,devtools" -- -W clippy::all                     → 0 warnings
cargo build --examples --features "full,...,devtools,vector-pgvector,cloud-connectors,server-tls,distributed-network" → 51 examples compile
cargo bench --features "full,constrained-decoding" --no-run                      → 42 benchmarks compile
cargo bench --features "full,constrained-decoding" --bench core_benchmarks -- --test → 42 benchmarks pass
```
