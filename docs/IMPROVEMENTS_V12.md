# v12 Improvements Changelog

> Date: 2026-02-25

v12 focuses on **code quality, feature gating correctness, dead code hygiene, test coverage depth, and examples**. No new user-facing features were added; instead, v12 strengthens the foundation laid by v9-v11.

---

## Summary Metrics

| Metric | v11 | v12 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,156 | 5,350 | +194 |
| Examples | 31 | 37 | +6 |
| `#[allow(dead_code)]` annotations | 54 | 37 | -17 |
| Source files | ~285 | ~285 | 0 |
| Feature flags (real, excl. default/full) | 45 | 45 | 0 |
| Benchmarks | 16 | 16 | 0 |
| Compiler warnings | 0 | 0 | 0 |

---

## Phase 1: GUIDE.md Section Renumbering

**Problem**: Sections 61-65 in GUIDE.md were duplicated. The original sections 61-65 (MCP Protocol through Access Control) existed, and then a second set of sections 61-65 were added later (covering different topics). This caused ToC inconsistency.

**Fix**: Renumbered the duplicate sections 61-65 to 131-135, updating both the Table of Contents and the section headers. The ToC now has 135 entries with no gaps or duplicates.

**Files changed**: `docs/GUIDE.md`

---

## Phase 2: Feature Gating Corrections

**Problem**: Two modules were compiled unconditionally despite having corresponding feature flags:
- `cloud_connectors` (feature: `cloud-connectors`) — Cloud provider configuration presets
- `vector_db_pgvector` (feature: `vector-pgvector`) — PostgreSQL-backed vector storage

Additionally, the `core` feature flag was identified as an intentional marker (no conditional code depends on it) and documented as such.

**Fix**:
- Added `#[cfg(feature = "cloud-connectors")]` to `cloud_connectors` module declaration in `lib.rs`
- Added `#[cfg(feature = "vector-pgvector")]` to `vector_db_pgvector` module declaration in `lib.rs`
- Verified feature dependency chains in `Cargo.toml` (`containers` -> `shared_folder` -> `cloud-connectors`)
- Documented `core` as an intentional marker feature

**Files changed**: `crates/ai_assistant/src/lib.rs`

---

## Phase 3: aws-bedrock Dependency Fix

**Problem**: The `aws-bedrock` feature was missing a `dep:sha2` dependency, causing compilation errors when the feature was enabled in isolation.

**Fix**: Added `dep:sha2` to the `aws-bedrock` feature's dependency list in `Cargo.toml`.

**Files changed**: `crates/ai_assistant/Cargo.toml`

---

## Phase 4: Test Coverage — rag_debug.rs

**Before**: 11 tests
**After**: 63 tests (+52)

New tests cover: RAG pipeline debug tracing, chunk scoring visibility, retrieval strategy analysis, embedding distance diagnostics, query expansion debugging, reranking pipeline inspection, context window utilization, source attribution verification, and edge cases.

**Files changed**: `crates/ai_assistant/src/rag_debug.rs`

---

## Phase 5: Test Coverage — structured.rs

**Before**: 16 tests
**After**: 93 tests (+77)

New tests cover: JSON Schema validation paths, type coercion, nested object validation, array constraints, enum enforcement, optional vs required fields, default value injection, format validation (email, URI, date-time), pattern matching, composition (allOf, anyOf, oneOf), recursive schema handling, error message quality, and edge cases.

**Files changed**: `crates/ai_assistant/src/structured.rs`

---

## Phase 6: Test Coverage — provider_plugins.rs

**Before**: 22 tests
**After**: 87 tests (+65)

New tests cover: PromptToolFallback prompt construction, multi-format tool call parsing (JSON, XML, code blocks), provider capability detection, tool schema serialization, error handling for malformed tool calls, concurrent provider access, provider plugin lifecycle, and edge cases.

**Files changed**: `crates/ai_assistant/src/provider_plugins.rs`

---

## Phase 7: Dead Code Audit

**Before**: 54 `#[allow(dead_code)]` annotations across the codebase
**After**: 37 `#[allow(dead_code)]` annotations (-17 removed)

**Process**: Every annotation was individually examined and categorized:

| Category | Action | Count |
|----------|--------|-------|
| False positive (code is actually used) | Removed annotation | 17 |
| Feature-gated field (used only with specific features) | Kept annotation | ~15 |
| Serialization field (needed for wire format) | Kept annotation | ~10 |
| Stored-but-unread configuration field | Kept annotation | ~12 |

**Files audited**: `server.rs`, `aws_auth.rs`, `providers.rs`, `autonomous_loop.rs`, `distributed_agents.rs`, `trigger_system.rs`, `browser_tools.rs`, `os_tools.rs`, `agent_sandbox.rs`, `user_interaction.rs`, `interactive_commands.rs`, `a2a_protocol.rs`, `mcp_protocol/`, `cloud_connectors.rs`, `advanced_memory/`, `voice_agent.rs`, `media_generation.rs`, and others.

---

## Phase 8: New Examples

6 new examples added (31 -> 37 total):

| Example | Feature Flags | Description |
|---------|--------------|-------------|
| `butler_demo` | `autonomous`, `butler` | Butler auto-configuration system demonstration |
| `browser_demo` | `autonomous`, `browser` | CDP browser automation demonstration |
| `distributed_agents_demo` | `autonomous`, `distributed-agents` | Multi-node agent distribution demonstration |
| `devtools_demo` | `autonomous`, `devtools` | Agent debugging and profiling tools demonstration |
| `bedrock_demo` | `aws-bedrock` | AWS Bedrock provider integration demonstration |
| `advanced_memory_demo` | `full` | Advanced memory with semantic facts demonstration |

All examples include `required-features` in their `[[example]]` entries in `Cargo.toml`.

**Files changed**: `examples/butler_demo.rs`, `examples/browser_demo.rs`, `examples/distributed_agents_demo.rs`, `examples/devtools_demo.rs`, `examples/bedrock_demo.rs`, `examples/advanced_memory_demo.rs`, `crates/ai_assistant/Cargo.toml`

---

## Documentation Updates

The following documentation files were updated as part of v12:

| File | Changes |
|------|---------|
| `docs/TESTING.md` | Test count 5156 -> 5350, added v12 row to history table |
| `docs/CONCEPTS.md` | Added sections 148 (Feature Gating Strategy) and 149 (Dead Code Hygiene), updated ToC with missing entries 142-147 |
| `docs/AGENT_SYSTEM_DESIGN.md` | Added section 52 (Arquitectura de Feature Gating Condicional), updated ToC with missing entries 45-51 |
| `docs/GUIDE.md` | Sections 131-135 renumbering verified consistent (done in Phase 1) |
| `docs/feature_matrix.html` | Updated to v18: test count 5350, examples 37, added backup history entry |
| `docs/framework_comparison.html` | Updated to v15: test count 5350, examples 37, added backup history entry, updated all active stat references |
| `docs/IMPROVEMENTS_V12.md` | This file (new) |
