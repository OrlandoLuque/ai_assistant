# Modus Operandi — ai_assistant development sessions

## How to continue after a session saturates

When starting a new session, say:
```
Continúa con el desarrollo del proyecto. Lee docs/modus-operandi.md y docs/IMPROVEMENTS.md para saber dónde estamos.
```

## Current state (updated 2026-02-21)

- **v1 roadmap**: 39/39 items COMPLETE (Phases 1-11)
- **v2 roadmap**: Following `docs/IMPROVEMENTS.md`
  - Phase 1 (Providers): 5/5 HECHO
  - Phase 2 (Infrastructure): 4/4 HECHO
  - Phase 3 (Advanced): 5/5 HECHO
  - Phase 4 (Ecosystem): 4/4 HECHO
  - Phase 5 (Testing): 4/4 HECHO
- **v2 roadmap**: COMPLETE (all 22 items HECHO)
- **Test count**: 2510 (0 failures, 0 clippy warnings)
- **`cargo publish --dry-run`**: PASSES
- **Last commit**: pending (all v2 work not yet committed)

## Standard workflow per phase

### 1. Audit first
Before implementing anything, audit existing code — many items turn out to be already implemented. Use Explore agents to search the codebase.

### 2. Plan mode
Enter plan mode for each phase. Key structure:
- Identify what's truly NEW vs already done
- Mark already-done items as HECHO in IMPROVEMENTS.md immediately
- Design workstreams (WS1, WS2...) for parallel implementation

### 3. Parallel agent pattern
Launch background agents (`run_in_background: true`) for independent workstreams:
- Each agent creates ONE new file (module) with full implementation + tests
- Agents should NOT modify lib.rs — the main thread does wiring after agents finish
- Agent prompt must include: exact types, method signatures, test count target, codebase patterns to follow

### 4. Wiring (main thread, after agents finish)
1. Add `pub mod <new_module>;` in lib.rs under the correct feature gate section
2. Add `pub use <new_module>::{Type1, Type2, ...};` re-exports
3. Run `cargo check --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network"`
4. Run `cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents" --lib`
5. Run `cargo clippy --features "full,autonomous,scheduler,butler,browser,distributed-agents" -- -W clippy::all`

### 5. Documentation updates (parallel agents)
Launch parallel agents for:
- **HTML docs** (with timestamped backups FIRST):
  - `docs/feature_matrix.html` — add rows for new modules
  - `docs/framework_comparison.html` — update roadmap section
- **Markdown docs**:
  - `docs/GUIDE.md` — add numbered sections at end
  - `docs/AGENT_SYSTEM_DESIGN.md` — add numbered sections at end
  - `docs/TESTING.md` — update test count
  - `docs/CONCEPTS.md` — add concept explanations
  - `docs/IMPROVEMENTS.md` — mark items HECHO/PARCIAL

### 6. Commit
All changes for a phase go in one commit. User must explicitly ask for commit.

## Key patterns & gotchas

### lib.rs module organization
```
// CORE MODULES (always available) — line ~105
// MULTI-AGENT FEATURE — #[cfg(feature = "multi-agent")]
// ASYNC RUNTIME FEATURE — #[cfg(feature = "async-runtime")]
// DISTRIBUTED FEATURE — #[cfg(feature = "distributed")]
// ANALYTICS FEATURE — #[cfg(feature = "analytics")]
// VISION FEATURE — #[cfg(feature = "vision")]
// EMBEDDINGS FEATURE — #[cfg(feature = "embeddings")]
// ADVANCED STREAMING — #[cfg(feature = "advanced-streaming")]
// ADAPTERS FEATURE — #[cfg(feature = "adapters")]
// TOOLS FEATURE — #[cfg(feature = "tools")]
// RAG FEATURE — #[cfg(feature = "rag")]
// AUTONOMOUS — #[cfg(feature = "autonomous")]
// etc.
```

### Concurrent agent modification hazard
NEVER let agents modify lib.rs — they will overwrite each other's changes. Only the main thread should edit lib.rs after all agents complete.

### HTML backup policy
MANDATORY before changing HTML docs:
```bash
cp docs/feature_matrix.html "docs/backups/feature_matrix_YYYY-MM-DD_HHMM_description.html"
cp docs/framework_comparison.html "docs/backups/framework_comparison_YYYY-MM-DD_HHMM_description.html"
```

### Feature flag rules
- `dep:X` prefix: if ANY feature uses `dep:X`, ALL must use `dep:X` (never mix)
- `full` feature includes lightweight features only
- Heavy features (`distributed-network`, `autonomous`, `p2p`, etc.) are opt-in

### Async pattern
Uses `Pin<Box<dyn Future<Output = T> + Send + '_>>` — NOT `async-trait` crate.

### Test commands
```bash
# Standard full test (most features)
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents" --lib

# With distributed network
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network" --lib

# P2P only
cargo test --features "full,p2p" --lib -- p2p::

# Quick check (lightweight features only)
cargo test --features full --lib
```

### Build check (all features)
```bash
cargo check --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network"
```

## What's next

**v2 roadmap COMPLETE** — all 22 items across 5 phases are HECHO.
- `cargo publish --dry-run` passes
- All v2 work needs to be committed (user has not requested commit yet)
- Ready for `cargo publish` when desired

## File map (key files)

| File | Purpose |
|------|---------|
| `src/lib.rs` | Module declarations + re-exports (~1300 lines) |
| `src/providers.rs` | Main provider routing (generate_response, streaming) |
| `src/config.rs` | AiProvider enum + AiConfig |
| `src/cloud_providers.rs` | Cloud API routing (OpenAI, Anthropic, Gemini, etc.) |
| `src/tools.rs` | ProviderPlugin trait + ToolCall/ToolDefinition |
| `src/async_providers.rs` | AsyncHttpClient trait, create_runtime() |
| `src/async_provider_plugin.rs` | AsyncProviderPlugin trait + bridge adapters |
| `src/llm_judge.rs` | LLM-as-Judge evaluation system |
| `src/embedding_providers.rs` | EmbeddingProvider trait + 4 implementations |
| `src/gemini_provider.rs` | GeminiProvider (Google Gemini API) |
| `src/provider_plugins.rs` | OllamaProvider, KoboldCpp, LMStudio + PromptToolFallback |
| `src/guardrail_pipeline.rs` | Guard trait + GuardrailPipeline + 6 built-in guards |
| `.github/workflows/ci.yml` | GitHub Actions CI (check, test, clippy, fmt) |
| `Cargo.toml` | Feature flags + dependencies |
| `docs/IMPROVEMENTS.md` | v2 roadmap tracking (source of truth) |
| `docs/GUIDE.md` | User guide (numbered sections) |
| `docs/AGENT_SYSTEM_DESIGN.md` | Architecture docs (numbered sections) |
