# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - v51 (2026-04-20) — V95: StallHeuristic robustness + LLM-light backend (0.2.27)

### Added
- **`StallSignal::Overheating`** — third signal for rate-based detection.
  Fires when the sliding window of tool-call timestamps exceeds
  `RateThresholds::max_calls` within `RateThresholds::window`.
- **`StallLanguage` enum** (`English`, `Spanish`, `French`, `German`) +
  **`StallKeywordLexicon`** with compact per-language frustration word
  lists and `contains_frustration(text, lang)` helper.
- **`RateThresholds { window, max_calls }`** struct + `Default` impl +
  constants `DEFAULT_RATE_WINDOW = 60s`, `DEFAULT_RATE_MAX_CALLS = 30`.
- **`KeywordStallDetector` builders:** `with_language(StallLanguage)` and
  `with_rate_thresholds(RateThresholds)`. Introspection: `language()`,
  `rate_thresholds()`, `recent_timestamp_count()`.
- **New feature flag `stall-detection-llm`** — implies `stall-detection`,
  zero new dependencies. Adds module `src/stall_detection_llm.rs` with:
  - `LlmVerdict` (`Stalled(StallSignal)` | `Continue` | `Abstain`).
  - `LlmVerdictInput { recent_tool_names, last_user_message }`.
  - `LlmVerdictFn = Arc<dyn Fn(&LlmVerdictInput) -> LlmVerdict + Send + Sync>`.
  - `LlmAssistedStallDetector<H>` wrapper — `new`, `with_min_interval`,
    `cached_verdict`, `inner`, `inner_mut`. Caller-provided LLM callback is
    called at most once per cooldown (default 30s via
    `DEFAULT_LLM_COOLDOWN`); tool-name trail capped at
    `TOOL_TRAIL_CAP = 16`.
  - 11 unit tests.
- **16 new tests** in `stall_detection::tests` covering overheating, rate
  thresholds, multi-language lexicons, and signal precedence.
- **Docs** — `docs/IMPROVEMENTS_V95.md` with design rationale for signal
  precedence (`RepeatedToolCall > Overheating > Frustrated`), the
  English-vs-lexicon split, and the cooldown model.

### Changed
- **`StallSignal` is now `#[non_exhaustive]`** — future signals can be added
  without a major bump. Callers matching exhaustively must add a `_` arm.
- `observe_user_message` in `KeywordStallDetector` dispatches by language —
  English still routes through `KeywordEmotionDetector`; other languages use
  the new lexicon (they do **not** populate `last_emotion()`).
- `check()` precedence: RepeatedToolCall > Overheating > Frustrated.
- `src/lib.rs` re-exports `RateThresholds`, `StallKeywordLexicon`,
  `StallLanguage`, `DEFAULT_RATE_WINDOW`, `DEFAULT_RATE_MAX_CALLS` under
  `feature = "stall-detection"`, and `LlmAssistedStallDetector`, `LlmVerdict`,
  `LlmVerdictFn`, `LlmVerdictInput`, `DEFAULT_LLM_COOLDOWN`, `TOOL_TRAIL_CAP`
  under `feature = "stall-detection-llm"`.
- Version `0.2.26 → 0.2.27` (patch-level, additive only).

### Notes
- No new telemetry counters or OTel spans. Existing `record_user_stall` and
  `start_user_stall_span` accept any signal `&str`, so `"Overheating"` flows
  through the V93 paths unchanged.
- LLM wrapper holds the user message only for the callback invocation — the
  struct has no persistent `String` field reachable after `check()` returns.

### AgenticLoop auto-integration
- `AgenticLoop` gained an optional `Box<dyn StallHeuristic>` field, gated on
  `feature = "stall-detection"`, plus builders/accessors:
  `with_stall_heuristic`, `stall_heuristic`, `stall_heuristic_mut`.
- `process()` forwards the user message to `observe_user_message` and, after
  each iteration, hashes new `ToolCall`s via `hash_tool_call` and feeds them
  to `observe_tool_call` + `check()`. A `Stalled` verdict sets
  `state.status = LoopStatus::UserStalled` and breaks the loop.
- 2 new tests in `agentic_loop::tests` cover the builder surface and the
  frustrated-user-message path.

## [Unreleased] - v50 (2026-04-20) — V94: Ephemeral sub-agent spawning (0.2.26)

### Added
- **`sub-agents` feature flag** — opt-in, composes
  `["multi-agent", "analytics"]` (both zero-dep). Zero new dependencies added.
- **`src/sub_agents.rs`** — new module with:
  - `SubAgentKind` enum (`Fork`, `Teammate`, `Explore`) — structural
    equivalent of Claude Code's `Task` tool sub-types.
  - `IsolationLevel` enum (`InProcess`, `ContextIsolated`, `ExternalProcess`).
  - `SubAgentSpec` with fluent builder (`with_role`, `with_context_summary`,
    `with_isolation`, `with_budget_hint`).
  - `SubAgentStatus` (`Completed`, `Failed`, `Cancelled`, `Deferred`) +
    `is_success()` helper.
  - `SubAgentResult` + `::deferred(id, reason)` helper.
  - `trait SubAgentRunner: Send + Sync` — `supports` + `run`.
  - Default `InProcessSubAgentRunner` — accepts `InProcess` and
    `ContextIsolated`; returns `Deferred` for `ExternalProcess` isolation so
    callers can chain runners. LLM-free by design — hermetic tests, no
    required network deps.
  - Constant `SPAN_NAME = "agent.sub_agent_spawned"`.
  - 15 unit tests.
- **Telemetry** in `src/telemetry.rs`:
  - `AggregatedMetrics::sub_agents_spawned_total: u64`.
  - `AggregatedMetrics::sub_agents_completed_total: u64` (only incremented
    when `record_sub_agent_complete(..., success = true)`).
  - `TelemetryCollector::record_sub_agent_spawn(kind: &str, isolation: &str)`.
  - `TelemetryCollector::record_sub_agent_complete(kind: &str, status: &str, success: bool)`.
- **OpenTelemetry** in `src/opentelemetry_integration.rs`:
  - `OtelTracer::start_sub_agent_span(kind: &str, isolation: &str) -> AiSpan`,
    operation `agent.sub_agent_spawned`, attributes `kind` + `isolation`.
- **Docs** — `docs/IMPROVEMENTS_V94.md` with framing (orthogonal to
  multi-agent orchestrator), design rationale (LLM-free default, Deferred vs
  Failed, &str signals for telemetry portability), and roadmap pointer.

### Changed
- `src/lib.rs` re-exports `sub_agents::*` under `feature = "sub-agents"`.
- Version `0.2.25 → 0.2.26` (patch-level, additive only).

### Notes
- Real filesystem/process isolation (git worktree, spawned subprocess) stays
  a caller concern (`memory/feedback_library_framing.md` rule). Callers that
  need host-level isolation implement `SubAgentRunner` themselves; the
  default `Deferred` path routes those specs explicitly instead of pretending
  to handle them.

## [Unreleased] - v49 (2026-04-20) — V93: In-crate StallHeuristic (0.2.25)

### Added
- **`stall-detection` feature flag** — opt-in, composes
  `["autonomous", "audio", "analytics"]` (all three zero-dep). Zero new
  dependencies added.
- **`src/stall_detection.rs`** — new module with:
  - `StallSignal` (`Frustrated`, `RepeatedToolCall`) and `StallDecision`
    (`Continue`, `Stalled(StallSignal)`).
  - `trait StallHeuristic` — `observe_tool_call`, `observe_user_message`,
    `check`, `reset`.
  - `KeywordStallDetector` — default implementation backed by a
    `VecDeque<u64>` ring buffer (capacity 8) of FNV-1a hashes plus
    `KeywordEmotionDetector` applied to the latest user message. Stores
    only derived signals — no raw text.
  - `hash_tool_call(name, args_bytes)` helper (FNV-1a).
  - Constants `RING_BUFFER_SIZE = 8`, `REPEAT_THRESHOLD = 3`,
    `SPAN_NAME = "agent.user_stall_detected"`.
  - 14 unit tests.
- **`LoopStatus::UserStalled`** variant in `src/agentic_loop.rs`. Present
  unconditionally so exhaustive matches stay stable regardless of feature
  selection; only ever produced when `stall-detection` is enabled.
- **`TelemetryCollector::record_user_stall(&self, signal: &str)`** in
  `src/telemetry.rs`, with new `AggregatedMetrics::user_stall_events_total:
  u64` counter. Accepts a `&str` signal so telemetry remains callable
  without the `stall-detection` feature compiled in.
- **`OtelTracer::start_user_stall_span(&self, signal: &str)`** in
  `src/opentelemetry_integration.rs`. Produces an `AiSpan` with operation
  `agent.user_stall_detected` and attribute
  `signal=Frustrated|RepeatedToolCall`.
- **Docs** — `docs/IMPROVEMENTS_V93.md` with design rationale, privacy
  guarantees, feature composition, and roadmap pointer to task #155.

### Changed
- `src/lib.rs` re-exports `stall_detection::*` under
  `feature = "stall-detection"`.
- Version `0.2.24 → 0.2.25` (patch-level, additive only).

### Privacy
- The stall heuristic persists only a `u64` hash per tool call and an
  `Option<EmotionCategory>` for the latest user message. Raw text is never
  stored, consistent with `pii_tokenizer` guarantees.

### Notes
- Signal precedence: when both fire, `RepeatedToolCall` dominates
  `Frustrated` (stronger invariant — budget is being burned this tick).
- Task #155 will add an LLM-assisted fallback, multi-language lexicons, and
  an overheating/burn-rate signal.

## [Unreleased] - v48 (2026-04-20) — V92: Claude Code permission-label adapter (0.2.24)

### Added
- **`PermissionRequirement`** (src/agent_policy.rs) — presentation-layer
  adapter bundling `ActionType` + `RiskLevel` + `DefaultDecision`. Build
  directly with `PermissionRequirement::new(...)` or derive from an action and
  a policy with `PermissionRequirement::from_policy(&policy, &action)`.
- **`DefaultDecision`** enum — `Allow` / `Prompt` / `Deny`. Captures what the
  policy decides before any user interaction, distinct from runtime approval
  handler decisions.
- **`to_claude_code_label`** — renders a `PermissionRequirement` using Claude
  Code's vocabulary (`ReadOnly` / `WorkspaceWrite` / `DangerFullAccess` /
  `Prompt` / `Allow`). Useful for docs, UIs, and examples that prefer the
  Claude Code naming without changing the internal permission taxonomy.
- **12 unit tests** in `agent_policy::tests` covering every branch of the
  mapping table plus the three policy presets.
- **Docs** — `docs/IMPROVEMENTS_V92.md` with the full mapping table and
  design rationale.

### Changed
- `src/lib.rs` now re-exports `DefaultDecision` and `PermissionRequirement`
  under `feature = "autonomous"` alongside `AgentPolicy`.
- Version `0.2.23 → 0.2.24` (patch-level, additive only; no runtime paths
  changed, no new dependencies, no API breakage).

### Notes
- The adapter is presentation-only: `to_claude_code_label` does not influence
  approval decisions. Runtime behaviour still flows through `AgentPolicy` +
  `ApprovalHandler`.
- Claude Code's label set has no explicit `Deny`; denials surface as
  `"Prompt"`. Callers that need the distinction should read
  `requirement.default_decision` directly.

## [Unreleased] - v47 (2026-04-20) — V91: Composable prompt fragments (0.2.23)

### Added
- **`prompt_fragments` module** — composable conditional prompt assembly.
  Structural equivalent of Claude Code's ~110 conditional instruction strings,
  but extensible by the caller rather than hardcoded.
- **Public API** — `PromptBuilder`, `PromptContext`, `PromptFragment`,
  `PromptPreset`, `FragmentCategory`, `Platform`, `AppliedFragment`.
- **Built-in catalog** — 11 fragments under `prompt_fragments::catalog::*`:
  shell notes (Windows/Unix), tool-use guidance, plan/execute mode, RAG
  citation reminder, GDPR-EU notice, TDD workflow, git commit conventions,
  Rust idioms, academic citation style.
- **Six curated presets** — `Minimal`, `ToolUseChatbot`, `RagAssistant`,
  `AgenticLoop`, `ResearchAgent`, `CodeDeveloper`.
- **Introspection** — `build_with_trace` returns the applied fragments in
  output order for debugging and OpenTelemetry spans.
- **Example** — `examples/prompt_fragments.rs` with 4 scenarios
  (agentic loop, code developer, RAG + EU GDPR, custom-signal fragment).
- **Docs** — `docs/PROMPT_FRAGMENTS.md` (complete guide) and
  `docs/IMPROVEMENTS_V91.md` (design rationale + status).
- **Website** — new `prompt_fragments.html` guide page, link cards on
  `index.html` / `product_overview.html` / `ai_assistant_overview.html`, new
  row in `feature_matrix.html`, cross-links from the anti-hallucination and
  research guide pages.
- **Butler integration (Phase 3)** —
  `Butler::recommend_prompt_fragments(intent, &report) -> PromptRecommendation`.
  Rule-based keyword dispatch picks a seed `PromptPreset` (research / code /
  RAG / autonomous / chat), with a project-type fallback, and overlays extras
  (`git_commit_conventions` when a VCS is detected, `rust_idioms` for Rust
  projects, platform shell notes that self-gate by host OS). Returns the
  preset, overlay keys, and a human-readable justification.
- **CLI** — `ai_cli butler recommend-prompt --intent "<description>"`
  surfaces the recommendation for a user-supplied intent against the scanned
  environment.
- **10 unit tests** for `Butler::recommend_prompt_fragments`
  (`butler::tests::prompt_fragments_tests`) in addition to the 23 tests in
  `prompt_fragments.rs`.

### Changed
- Everything gated behind new `feature = "prompt-fragments"` (opt-in, not in
  `full`). Butler integration additionally requires `feature = "butler"`.
  Zero new dependencies, zero API breakage for existing callers.
- Reuses `OperationMode` from `mode_manager` when `feature = "autonomous"` is
  active — no type duplication.

### Notes
- Fragment text is trusted input; it is concatenated verbatim into the system
  prompt. Never build fragments directly from end-user input (prompt-injection
  vector). The module docs and guide both spell this out.
- An LLM-assisted variant of `recommend_prompt_fragments` is deferred to a
  follow-up behind a separate feature flag; the rule-based path already covers
  the intended shape.

## [Unreleased] - v46 (2026-04-19) — V90: Dataset hallucination/faithfulness benchmarks (0.2.22)

### Added
- **`eval_benchmarks` module** — uniform `BenchmarkLoader` trait, on-disk cache,
  HTTP downloader with atomic writes + 200 MB cap, runner, post-hoc threshold
  calibrator, and text/JSON report renderers.
- **Five loaders** — `truthfulqa`, `halueval_qa`, `factscore`, `ragas_wikiqa`,
  `fever` (opt-in, CC-BY-SA 3.0). Datasets fetched on demand, never vendored.
- **CLI** — `ai_cli benchmark <list|info|download|run|calibrate>` with
  `--json`, `--limit`, `--objective`, `--accept-license`, `--cache-dir`.
- **HTTP server** — `GET /benchmarks` and `GET /benchmarks/<name>` (read-only;
  also under `/api/v1/benchmarks`).
- **MCP** — `list_benchmarks` and `get_benchmark` tools (read-only, idempotent)
  via `mcp_protocol::register_benchmark_tools(&mut server)`.
- **Example** — `examples/eval_benchmarks_demo.rs` exercises the full pipeline
  with an in-tree fixture and a mock generator (no network, no LLM).
- **Docs** — `docs/IMPROVEMENTS_V90.md` + new *Dataset Benchmarks (V90)*
  section in `docs/GUIDE_ANTI_HALLUCINATION.md` and the matching HTML guide.

### Changed
- Zero new dependencies: CSV parser hand-rolled, HTTP via existing `ureq`,
  RAGAS via HF datasets-server JSON API (no `parquet`), cache root resolved
  from `CARGO_TARGET_DIR` (no `dirs`).
- Everything gated behind `feature = "eval"` — default builds unchanged.

## [Unreleased] - v45 (2026-04-11) — V89: Wire all binary stubs (0.2.21)

### Added
- **`ai_cli` cost savings** — `cost savings` replaces the old stub with a real
  `CostDashboardSnapshot` loader, cost-by-model breakdown, top-5 most expensive
  requests, and hypothetical single-model projection.
- **`ai_cli tool` / `ai_cli workflow`** — new subcommands that delegate to a
  local LLM via `run_delegated_llm`, wiring the existing tool and workflow
  APIs end-to-end.
- **Stubs removed** — audit of the 20 binaries in `src/bin/` found 5 real
  stubs across 4 binaries; every one is now backed by a real implementation
  using already-available library APIs.

### Changed
- Zero new dependencies for V89.

## [Unreleased] - v44 (2026-04-11) — V88: Wiring Completo, Butler, Binarios

### Added
- **Anti-hallucination wiring (V88)** — full integration across all layers:
  - `assistant.rs`: opt-in `anti_hallucination_config` and `quality_gate_runner` fields.
  - `config_file.rs`: `AntiHallucinationFileConfig`, `QualityGateFileConfig`, `ResearchFileConfig`.
  - `server_axum.rs`: 6 new REST endpoints (`/api/v1/verify/*`, `/api/v1/research/*`).
  - MCP: 9 new tools (6 research + 3 verification: check_faithfulness, verify_claims, run_quality_gates).
- **Context budget (V88)** — `ContextSourceType::AcademicPaper` with peer-reviewed boost (0.75).
- **RAG tiers (V88)** — `estimate_extra_calls()` now includes 7 anti-hallucination features.
- **Telemetry (V88)** — 5 new convenience methods: `record_faithfulness_check`, `record_academic_search`,
  `record_quality_gate_run`, `record_cove_verification`, `record_abstention`.
- **OpenTelemetry (V88)** — 5 new spans: `anti_hallucination.pipeline`, `faithfulness.score`,
  `cove.verify`, `academic.search`, `quality.gate`.
- **Cost tracking (V88)** — `RequestType::Verification`, `RequestType::AcademicSearch` in cost_integration.
  `CostTracker`: `verification_cost`, `verification_calls`, `academic_search_cost`, `academic_search_calls`.
- **Autonomous loop (V88)** — `AgentResult.quality_score: Option<f64>`.
- **Butler (V88)** — 8 new recommendations (Q7-Q11 quality, C6 cost, 2 research).
  `DeploymentScenario::ResearchWorkstation`. New `AdvisorConfig` fields:
  `anti_hallucination_enabled`, `quality_gates_configured`, `research_mode_enabled`, `academic_api_keys_present`.
- **Agent wiring (V88)** — system prompts for `ResearchAssistant`, `PeerReviewer`, `WritingCoach` roles.
- **ai_cli (V88)** — 3 new subcommands: `verify`, `research` (gated), `quality`.
- **ai_test_harness (V88)** — 5 new categories: anti-hallucination, quality-gates, faithfulness,
  verification (eval), research (research feature).
- ~30 new integration tests across harness categories.

### Changed
- Version 0.2.19 → 0.2.20.

## [Unreleased] - v43 (2026-04-11) — V87: Quality Gates & RAG Tier Integration

### Added
- **Quality gates (V87)** — configurable quality gates that check LLM outputs
  against minimum thresholds. Five metrics: Faithfulness, Confidence, GroundingRatio,
  ConsistencyScore, CitationCoverage. Three actions: Fail, Warn, Log.
  - New module: `quality_gates.rs` (~400 lines, gated `eval` feature).
  - `QualityGateRunner` — presets: `production_defaults()`, `strict()`.
  - `QualityScores` — overall score, badge color (green/yellow/red).
  - `QualityGateResult` — per-gate results, summary, pass/fail.
- **Feature group helpers (V87)** — in `rag_tiers.rs`:
  - `enable_verification_mode()` — all anti-hallucination features (7 fields).
  - `enable_research_mode()` — attribution + reranking (4 fields).
  - `enable_academic_mode()` — combined research + verification.
- 25 new tests (21 quality_gates + 4 rag_tiers).

### Changed
- Version 0.2.18 → 0.2.19.

## [Unreleased] - v42 (2026-04-11) — V86: Literature Review Pipeline + MCP Tools

### Added
- **Literature review pipeline (V86)** — end-to-end pipeline: search → filter → categorize → synthesize → format. Four synthesis styles (Narrative, Systematic, Annotated, Comparative). Multiple bibliography formats (BibTeX, APA, MLA, Chicago, IEEE).
  - New module: `literature_review.rs` (~600 lines, gated `research` feature).
  - `LiteratureReviewPipeline` — configurable with `SearchDepth` and `SynthesisStyle`.
  - `LiteratureReview` — output with sections, bibliography, BibTeX, statistics.
  - Presets: `quick()` (10 papers, annotated), `systematic()` (50 papers, deep).
- **MCP research tools (V86)** — 6 MCP tool definitions for research operations.
  - New module: `mcp_research_tools.rs` (~300 lines, gated `research` feature).
  - `ResearchToolRegistry` — tool discovery and dispatch.
  - Tools: `search_papers`, `get_paper_metadata`, `import_bibtex`, `export_bibtex`, `literature_review`, `extract_paper_metadata`.
  - Immediate dispatch for `import_bibtex` and `extract_paper_metadata`.
- 31 new tests (20 literature_review + 11 mcp_research_tools).

### Changed
- Version 0.2.17 → 0.2.18.

## [Unreleased] - v41 (2026-04-11) — V85: Paper Metadata & Agent Roles

### Added
- **Paper metadata extraction (V85)** — heuristic-based extraction of title,
  authors, abstract, DOI, year, keywords, sections, and references from
  academic paper text. Section type classification (10 types).
  - New module: `paper_metadata.rs` (~400 lines, gated `research` feature).
  - `PaperMetadataExtractor` — configurable extraction with confidence scoring.
  - `PaperSection` — detected sections with heading, content, level, and type.
  - `SectionType` — Abstract, Introduction, RelatedWork, Methodology, Results,
    Discussion, Conclusion, References, Appendix, Other.
- **Research agent roles (V85)** — 3 new `AgentRole` variants in `multi_agent.rs`:
  `ResearchAssistant`, `PeerReviewer`, `WritingCoach`.
- **Knowledge graph entity types (V85)** — `EntityType::Paper` and
  `EntityType::Author` in `knowledge_graph.rs` with aliases.
- 20 new tests (paper_metadata).

### Changed
- `EntityType::all()` returns 9 variants (was 7).
- Version 0.2.16 → 0.2.17.

## [Unreleased] - v40 (2026-04-11) — V84: Academic APIs & BibTeX

### Added
- **Academic search APIs (V84)** — unified `AcademicSearchProvider` trait with
  three provider implementations: `ArxivProvider` (Atom/XML), `SemanticScholarProvider`
  (REST/JSON), `PubMedProvider` (E-utilities XML). Multi-provider aggregation via
  `AcademicSearchEngine` with DOI-based deduplication.
  - New module: `academic_search.rs` (~800 lines, gated `research` feature).
  - `AcademicPaper` — full metadata: authors, abstract, year, venue, DOI, citations,
    fields of study, external IDs.
  - Rate limiting per provider (arXiv 3s, S2 100/5min, PubMed 3/s).
  - API keys via env vars (`SEMANTIC_SCHOLAR_API_KEY`, `NCBI_API_KEY`).
- **BibTeX parser/generator (V84)** — parse `.bib` files and generate BibTeX
  from academic papers.
  - New module: `bibtex.rs` (~500 lines, gated `research` feature).
  - `BibParser` — handles brace nesting, quoted values, bare numbers, `@comment`/`@preamble`/`@string`.
  - `BibGenerator` — deterministic output, `from_paper()` for automatic cite key generation.
  - Security: LaTeX injection sanitization (strips `\input`, `\write18`, `\immediate`, etc.).
  - Limits: max 10MB file, 10K entries, 10K chars per field.
  - `latex_to_unicode()` — common accent commands to Unicode.
- **`AcademicSearchAdapter`** — in `web_search.rs`, wraps academic providers to
  implement `SearchProvider` for integration with fact verification pipeline.
- **Academic paper source fields** — `doi`, `venue`, `citation_count` added to
  `Source` in `citations.rs`.
- **`research` feature flag** — new Cargo feature, included in `full`.
- 54 new tests (26 academic_search + 23 bibtex + 3 web_search + 2 citations).

### Changed
- `Source` struct in `citations.rs` now has 3 optional fields for academic papers.
- Version 0.2.15 → 0.2.16.

## [Unreleased] - v39 (2026-04-11) — V83: Verification Pipeline

### Added
- **Chain-of-Verification (V83)** — CoVe pipeline that extracts claims from
  LLM responses, verifies each against RAG/web search sources, and corrects
  or annotates the response. Configurable `VerificationSource` (RagOnly,
  WebSearchOnly, RagThenWeb, Both) and `CorrectionMode` (Replace, Annotate,
  Footnote). Hard cap `max_claims_to_verify=10` to control cost.
  - New module: `chain_of_verification.rs` (~490 lines).
  - `CoVeConfig` — strict/permissive presets, budget-aware.
  - `CoVeResult` — per-claim verdicts, corrections, overall accuracy.
- **Search-integrated fact verification** — `FactVerifier::verify_with_search()`
  and `verify_with_rag()` in `fact_verification.rs` for verifying claims against
  web search results or RAG chunks with source provenance tracking.
- **Divergence metrics** — `ConsistencyResult::measure_divergence()` in
  `self_consistency.rs` computes Shannon entropy, max group ratio, effective
  distinct count, and derives a `ConsistencyRecommendation` (High/Medium/Low/Abstain).
- **`search_for_claim()`** — keyword-based claim search helper in `web_search.rs`
  with stopword filtering and relevance scoring.
- **RagFeatures verification fields** — 2 new: `chain_of_verification`,
  `fact_check_search`. Enabled at Agentic+ tier. Total RagFeatures: 45.
- 45 new tests across 5 modules.

## [Unreleased] - v38 (2026-04-11) — V82: Faithfulness & Grounded Generation

### Added
- **Faithfulness NLI scoring (V82)** — NLI-based claim-level faithfulness
  evaluation against retrieved context. `FaithfulnessScorer` decomposes
  responses into atomic claims and evaluates each via word overlap (zero-cost)
  or LLM-based NLI.
  - New module: `faithfulness.rs` (~380 lines).
  - `NliVerdict` — Entailed, Contradicted, Neutral per claim.
  - `FaithfulnessReport` — overall score, per-claim verdicts, processed text.
- **Grounded generation** — anchor every response sentence to a source chunk.
  `GroundedGenerator` in `anti_hallucination.rs` with `ChunkAnchorMethod`
  (PostHoc, Prompted) and configurable similarity threshold.
- **`decompose_atomic()`** — finer-grained atomic claim decomposition in
  `hallucination_detection.rs` for faithfulness NLI evaluation.
- **`anchor_to_sources()`** — sentence-to-source anchoring in `citations.rs`
  with word overlap similarity.
- **`SourceType::AcademicPaper`** — new citation source type.
- **`FaithfulnessEvaluator`** — evaluator implementing `Evaluator` trait in
  `evaluation.rs` with `MetricType::Faithfulness` and `MetricType::GroundingRatio`.
- **RagFeatures fields** — 2 new: `faithfulness_scoring`, `grounded_generation`.
  Enabled at Thorough+ tier. Total RagFeatures: 43.
- 50 new tests across 6 modules.

## [Unreleased] - v37 (2026-04-11) — V81: Anti-Hallucination Orchestrator + Foundation

### Added
- **Anti-Hallucination Pipeline (V81)** — central orchestrator
  (`AntiHallucinationPipeline`) with 7 configurable strategies (Omit, Mark,
  Warn, Footnote, VerifyThenMark, VerifyThenOmit, Ask), calibrated abstention,
  per-claim confidence scoring, and auto-temperature for factual queries.
  - New module: `anti_hallucination.rs` (~580 lines).
  - `is_factual_query()` — heuristic factual vs creative detection.
  - Preset configs: `production()`, `strict()`, `permissive()`.
- **Per-claim confidence scoring** — `ConfidenceScorer::score_per_claim()`
  and `score_texts()` methods in `confidence_scoring.rs`.
- **Auto-temperature** — `AdaptiveThinkingConfig.auto_temperature_factual`
  forces lower temperature for factual queries, reducing hallucination risk.
  `QueryClassifier::is_factual_query()` public API for integration.
- **AbstentionGuard** — guardrail that blocks low-confidence responses
  (PostReceive stage), with configurable threshold and custom message.
- **AttributionGuard** — guardrail that warns on ungrounded claim patterns
  ("studies show", "experts say", etc.), with configurable severity.
- **RagFeatures anti-hallucination fields** — 3 new fields:
  `calibrated_abstention`, `mandatory_attribution`, `auto_temperature`.
  Mapped to tiers: Enhanced+ gets attribution+auto-temp, Thorough+ gets all.
- 67 new tests across 5 modules.

## [Unreleased] - v36 (2026-04-11) — V80: Azure OpenAI as first-class provider

### Added
- **Azure OpenAI Service (V80)** — first-class provider with dedicated
  `AiProvider::AzureOpenAI { endpoint, deployment }` variant. Uses the
  correct `api-key` header (NOT `Authorization: Bearer`) and Azure-specific
  URL pattern (`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21`).
  - Blocking + streaming + cancellable dispatch paths.
  - Config file support: `provider = "azure"` or `"azure_openai"`.
  - Env var fallback: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
    `AZURE_OPENAI_DEPLOYMENT`.
  - FFI bindings: `AiProviderKind::AzureOpenAI` + companion setters
    `ai_assistant_set_azure_endpoint` / `ai_assistant_set_azure_deployment`.
  - Static model list: gpt-4o, gpt-4o-mini, gpt-4, gpt-4-turbo,
    gpt-35-turbo, o1, o1-mini, o3-mini.
  - 12 new tests (config, cloud_providers, FFI, cross-crate integration).

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
- **Flat `AiProviderKind` C enum** — 18 unit variants mirroring the
  Rust `AiProvider` positionally. Data-bearing variants
  (`OpenAICompatible`, `Bedrock`, `AzureOpenAI`) are configured via
  companion setters. The Rust→FFI converter uses an **exhaustive
  match** so adding a Rust variant forces a compile error in `src/ffi.rs`.
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
