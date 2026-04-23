# V100 — Self-Correction: Tool, Research, Agent-Handoff, Safety Tasks

**Version**: 0.2.31 → 0.2.32
**Feature flag**: `self-correction` (same as V98/V99)

V100 completes the task-type matrix for the self-correction harness.
After V98 (claims) and V99 (code), the remaining domains — tool calls,
research citations, agent handoffs, safety guardrails — now have first-
class `CorrectableTask` implementations.

## New task types

### 1. `ToolCallTask` — `src/self_correction/tool_call.rs`

Retry a tool invocation whose arguments (usually JSON) fail validation.

```rust
pub enum ToolCallIssue {
    InvalidJson { detail: String },
    SchemaViolation { detail: String },
    ConstraintViolation { detail: String },
    UnknownTool { tool_name: String },
}
```

All retryable by default (the LLM just re-emits). Feedback prompt includes
optional `with_schema_hint(json_schema_text)` so the regenerator sees the
full target schema.

Example feedback tail:

```
Regenerate the tool call, applying these rules:
  1. Produce valid JSON only — no prose, no code fences.
  2. Match the schema exactly: required fields, types, enums.
  3. Respect domain constraints mentioned above.

Tool schema:
{"name": string, "count": int, ...}
```

### 2. `ResearchCitationTask` — `src/self_correction/research.rs`

Retry until in-text citations are valid (resolvable + cover claims).

```rust
pub enum CitationIssue {
    DanglingReference { marker: String },          // [5] with no bib entry
    UnusedReference { marker: String },            // bib entry nobody cites
    UnsupportedClaim { claim_excerpt: String },    // factual claim, no cite
    UnresolvableTarget { marker, detail },         // DOI/URL/arXiv 404
    LowCoverage { ratio: f64, threshold: f64 },    // cov < threshold
}
```

Coverage threshold defaults to 0.7; configurable via
`with_coverage_threshold(t)`. All retryable — citation accuracy is a
rewrite problem, not an intent problem.

### 3. `AgentHandoffTask` — `src/self_correction/agent_handoff.rs`

Retry until a planner/executor-agent handoff payload is complete.

```rust
pub enum HandoffIssue {
    MissingField { field: String },
    InvalidField { field: String, detail: String },
    UnknownTarget { target: String },
    DependencyNotMet { detail: String },
}
```

Builder methods `with_required_fields(iter)` and `with_valid_targets(iter)`
are displayed verbatim in the feedback prompt so the LLM sees the exact
field/target vocabulary.

### 4. `SafetyGuardrailTask` — `src/self_correction/safety.rs`

Retry safety violations **with per-variant retryability**:

```rust
pub enum SafetyIssue {
    PiiLeak { kind, sample_redacted },              // RETRYABLE — redact
    PromptInjection { detail, retryable },          // caller decides
    DisallowedContent { category, retryable },      // caller decides
    JailbreakAttempt { pattern },                   // FATAL — is_retryable=false
    PolicyError { detail },                         // FATAL
}
```

`JailbreakAttempt` and `PolicyError` are hard-coded non-retryable: the
engine stops with `FatalIssue(msg)`, and the caller is expected to refuse
rather than emit a retried output. This extends the V98 fatal-issue pattern
to the safety domain.

The validator reports `SafetyIssueSpec { kind, detail, sub_kind,
retryable }` and the task constructs the right `SafetyIssue` variant.

## The `RefCell<ValidateFn>` pattern

All four new tasks follow the same interior-mutability pattern as V99's
`CodeCompileTaskCell`: the validator closure is `Box<dyn FnMut(…) + Send>`
but the trait gives `&self` to `validate()`, so the closure is stored in
`RefCell<_>` and `borrow_mut()` in `validate`. This is the canonical
answer to the `Fn` / `FnMut` tension and keeps the trait surface simple.

## Quality-score policies

| Task | Empty issues | Non-empty |
|------|--------------|-----------|
| ToolCall | 1.0 | `1.0 - 0.2 × N` |
| ResearchCitation | 1.0 | `1.0 - 0.1 × N` |
| AgentHandoff | 1.0 | `1.0 - 0.15 × N` |
| Safety | 1.0 | **0.0 if any fatal**, else `1.0 - 0.25 × N` |

All clamped to `[0.0, 1.0]`. Safety's fatal-is-0 policy ensures the
regression detector correctly treats jailbreak-then-retry as maximum
regression.

## Public API additions

```rust
#[cfg(feature = "self-correction")]
pub use self_correction::{
    // V98:
    ClaimVerificationTask, ClaimIssue, /* trait + engine */
    // V99:
    CodeCompileTask, CodeCompileTaskCell, CodeTestTask,
    // V100:
    ToolCallTask, ToolCallIssue, ToolValidateFn, ToolValidationResult, ToolRegenerateFn,
    ResearchCitationTask, CitationIssue, CitationValidateFn, CitationValidationResult, CitationRegenerateFn,
    AgentHandoffTask, HandoffIssue, HandoffValidateFn, HandoffValidationResult, HandoffRegenerateFn,
    SafetyGuardrailTask, SafetyIssue, SafetyIssueSpec, SafetyCheckResult, SafetyValidateFn, SafetyRegenerateFn,
};
```

## Tests

- `tool_call`: 5 tests (valid first try, schema violation → retry,
  persistent invalid JSON → fail, feedback includes schema hint, Display)
- `research`: 5 tests (clean, dangling → retry, low coverage emitted,
  feedback mentions rules, Display variants)
- `agent_handoff`: 5 tests (clean, missing field → retry, unknown target,
  feedback lists required/valid, Display)
- `safety`: 8 tests (clean, PII retryable, jailbreak fatal, disallowed
  non-retryable fatal, injection retryable, feedback mentions PII rule,
  `quality_score` = 0 for fatal, Display)

**Total V100 tests: 23**
**Total self-correction tests: 72** (V98=36 + V99=13 + V100=23).

## Summary of the three-version arc

| Version | Domain | Tasks added | Tests |
|---------|--------|-------------|-------|
| V98 | Claims / framework | Trait, engine, ledger, `ClaimVerificationTask` | 36 |
| V99 | Code | `CodeCompileTask(Cell)`, `CodeTestTask` + cargo helpers | 13 |
| V100 | Tool / Research / Agent / Safety | 4 task types | 23 |
| **Total** | | 8 concrete tasks | **72** |

## What's NOT in V100 (follow-up)

The harness now covers every domain, but surface-area wiring remains:

- Auditor binaries (`ai_corrections`, `ai_corrections_gui`) — CLI and egui
  tools for inspecting the JSONL ledger, mirroring `ai_breeder` /
  `ai_breeder_gui`.
- `ai_cli` flags: `verify --auto-correct`, `code --auto-fix`, etc.
- HTTP endpoint: `POST /api/v1/correct`.
- MCP tools: `self_correct_claim`, `self_correct_code`,
  `self_correct_tool_call`, `self_correct_research`,
  `self_correct_agent_handoff`, `self_correct_safety`.
- GUI integration (egui widget for live attempt visualization).
- `SelfCorrectionFileConfig` in `config_file.rs`.
- `record_correction_attempt` in `telemetry.rs`.

These are grouped as a "surface wiring" follow-up because they share code
across all V-versions and are easier to land as one coherent batch than
piecemeal.
