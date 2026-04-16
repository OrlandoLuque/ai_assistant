# IMPROVEMENTS V88 — Wiring Completo, Butler, Binarios (0.2.20)

## Context

V88 integrates all V81-V87 modules (anti-hallucination, faithfulness, CoVe,
academic search, literature review, quality gates) into the full application
stack: assistant, config, REST API, MCP tools, telemetry, cost tracking,
butler advisor, CLI binaries, and test harness.

---

## Changes

### A) Wiring — Orchestration Central

**`src/assistant.rs`** — Opt-in anti-hallucination pipeline:
- `anti_hallucination_config: Option<AntiHallucinationConfig>` (gated `eval`)
- `quality_gate_runner: Option<QualityGateRunner>` (gated `eval`)

**`src/config_file.rs`** — Three new config sections:
- `AntiHallucinationFileConfig` — enabled, strategy, thresholds, features
- `QualityGateFileConfig` — name, metric, threshold, action
- `ResearchFileConfig` — API keys, default providers, bibliography format

**`src/server_axum.rs`** — REST endpoints + MCP tools:

| Endpoint | Method | Feature | Description |
|----------|--------|---------|-------------|
| `/api/v1/verify/quality-check` | POST | eval | Run quality gates on text |
| `/api/v1/verify/config` | GET | eval | Current verification config |
| `/api/v1/verify/faithfulness` | POST | eval | Evaluate faithfulness |
| `/api/v1/research/search` | POST | research | Search academic papers |
| `/api/v1/research/bibtex/import` | POST | research | Import BibTeX file |
| `/api/v1/research/bibtex/export` | POST | research | Export citations |

MCP tools registered: 6 research + 3 verification = 9 new tools.

### B) Wiring — Support Modules

**`src/context_budget.rs`**:
- New `ContextSourceType::AcademicPaper` (base score: 0.75 peer-reviewed boost)
- Added to both source_order arrays and Display impl

**`src/rag_tiers.rs`**:
- `estimate_extra_calls()` updated for 7 anti-hallucination features

**`src/agent_wiring.rs`**:
- System prompts for `ResearchAssistant`, `PeerReviewer`, `WritingCoach`

### C) Wiring — Telemetry & Costs

**`src/telemetry.rs`** — 5 new convenience methods:
- `record_faithfulness_check(score, claims_count, duration)`
- `record_academic_search(provider, query, results_count)`
- `record_quality_gate_run(passed, score)`
- `record_cove_verification(corrections, accuracy, duration)`
- `record_abstention(reason, confidence)`

**`src/opentelemetry_integration.rs`** — 5 new spans:
- `anti_hallucination.pipeline`, `faithfulness.score`, `cove.verify`
- `academic.search`, `quality.gate`

**`src/cost_integration.rs`**:
- `RequestType::Verification`, `RequestType::AcademicSearch`

**`src/cost.rs`**:
- `CostTracker` fields: `verification_cost`, `verification_calls`,
  `academic_search_cost`, `academic_search_calls`

**`src/autonomous_loop.rs`**:
- `AgentResult.quality_score: Option<f64>`

### D) Butler — New Recommendations

**`src/butler.rs`**:

| ID | Title | Category | Priority |
|----|-------|----------|----------|
| Q7 | Enable anti-hallucination pipeline | Quality | High |
| Q8 | Enable faithfulness scoring | Quality | Medium |
| Q9 | Configure quality gates | Quality | Medium |
| Q10 | Enable CoVe for critical queries | Quality | Low |
| Q11 | Configure academic search providers | Quality | Medium |
| Q12 | Enable research mode | Quality | Low |
| C6 | Set anti-hallucination LLM call budget | Cost | High |

New `AdvisorConfig` fields: `anti_hallucination_enabled`, `quality_gates_configured`,
`research_mode_enabled`, `academic_api_keys_present`.

New `DeploymentScenario::ResearchWorkstation` — `full` + `research` (~28 MB).

### E) Binaries

**`src/bin/ai_cli.rs`** — 3 new subcommands:
- `ai_cli verify [--strategy] [--min-confidence] [--faithfulness] [--cove] [--quality-gates] <prompt>`
- `ai_cli research [--providers] [--max-results] [--bibtex] <query>` (gated `research`)
- `ai_cli quality gates list|check`

**`src/bin/ai_test_harness.rs`** — 5 new categories:
- `anti-hallucination` (3 tests, gated `eval`)
- `quality-gates` (4 tests, gated `eval`)
- `faithfulness` (2 tests, gated `eval`)
- `verification` (2 tests, gated `eval`)
- `research` (4 tests, gated `research`)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/assistant.rs` | +2 fields (eval-gated) |
| `src/config_file.rs` | +3 config structs, +3 fields |
| `src/server_axum.rs` | +6 endpoints, +1 AppState field, +9 MCP tools |
| `src/context_budget.rs` | +1 ContextSourceType variant |
| `src/rag_tiers.rs` | Extended estimate_extra_calls() |
| `src/agent_wiring.rs` | +3 role system prompts |
| `src/telemetry.rs` | +5 convenience methods |
| `src/opentelemetry_integration.rs` | +5 span helpers |
| `src/cost_integration.rs` | +2 RequestType variants |
| `src/cost.rs` | +4 CostTracker fields |
| `src/autonomous_loop.rs` | +1 AgentResult field |
| `src/butler.rs` | +7 recommendations, +1 scenario, +4 AdvisorConfig fields |
| `src/bin/ai_cli.rs` | +3 subcommands (~200 lines) |
| `src/bin/ai_test_harness.rs` | +5 categories (~150 lines) |

---

## Test Summary

| Category | New Tests |
|----------|-----------|
| ai_test_harness: anti-hallucination | 3 |
| ai_test_harness: quality-gates | 4 |
| ai_test_harness: faithfulness | 2 |
| ai_test_harness: verification | 2 |
| ai_test_harness: research | 4 |
| butler (updated existing) | 1 |
| **Total** | **~16** |

Full test suite: **6095 passing** (with `full,butler` features).

---

## Version

- **0.2.19 → 0.2.20**
