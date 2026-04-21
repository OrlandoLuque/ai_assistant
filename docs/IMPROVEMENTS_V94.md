# IMPROVEMENTS V94 — Ephemeral sub-agent spawning (0.2.26)

## Context

Task #154 in the Claude Code addendum roadmap: provide a structural equivalent
of Claude Code's `Task` tool (Fork / Teammate / Explore patterns) for the
library. A main agent should be able to delegate a discrete sub-task to a
short-lived sub-agent, receive a single consolidated result, and discard the
sub-agent.

This is **orthogonal** to the persistent multi-agent team that already lives in
`multi_agent::AgentOrchestrator`. Orchestrators manage long-lived roles,
message passing, and tasks across sessions; sub-agents are fire-and-forget
delegations — they run to completion and go away.

Per `memory/feedback_library_framing.md`, real filesystem/process isolation
(git worktree, spawned subprocess) is explicitly host-level and stays with the
caller. What lives in the crate is the declarative data model, a pluggable
runner trait, a default in-process runner, and telemetry.

## Scope

* **New module** `src/sub_agents.rs` behind `feature = "sub-agents"`.
* **New feature flag:** `sub-agents = ["multi-agent", "analytics"]` — both
  implied deps are zero-dep gates, so enabling `sub-agents` adds no new
  transitive dependencies.
* **Public API:**
  * `SubAgentKind` enum: `Fork`, `Teammate`, `Explore` + `as_str()` + `Display`.
  * `IsolationLevel` enum: `InProcess`, `ContextIsolated`, `ExternalProcess`
    + `as_str()`.
  * `SubAgentStatus` enum: `Completed`, `Failed`, `Cancelled`, `Deferred`
    + `as_str()` + `is_success()`.
  * `SubAgentSpec` struct with fluent builder (`with_role`,
    `with_context_summary`, `with_isolation`, `with_budget_hint`).
  * `SubAgentResult` struct + `SubAgentResult::deferred(id, reason)` helper.
  * `trait SubAgentRunner: Send + Sync` with `supports` + `run`.
  * Default `InProcessSubAgentRunner` — accepts `InProcess` and
    `ContextIsolated`; returns `Deferred` for `ExternalProcess`.
  * Constant `SPAN_NAME = "agent.sub_agent_spawned"`.
* **Telemetry** (`src/telemetry.rs`):
  * New counter `AggregatedMetrics::sub_agents_spawned_total: u64`.
  * New counter `AggregatedMetrics::sub_agents_completed_total: u64`
    (only incremented when `record_sub_agent_complete` is called with
    `success == true`).
  * `TelemetryCollector::record_sub_agent_spawn(kind: &str, isolation: &str)`.
  * `TelemetryCollector::record_sub_agent_complete(kind: &str, status: &str, success: bool)`.
  * Signals are `&str` (not typed enums) so telemetry stays callable without
    the `sub-agents` feature compiled in — matches the V93 pattern.
* **OpenTelemetry** (`src/opentelemetry_integration.rs`):
  * `OtelTracer::start_sub_agent_span(kind: &str, isolation: &str) -> AiSpan`
    with operation `agent.sub_agent_spawned` and attributes `kind` + `isolation`.
* **Tests:** 15 unit tests in `sub_agents::tests` + 2 in `telemetry::tests` +
  1 in `opentelemetry_integration::tests`.

## Design notes

### Why a separate module instead of extending `multi_agent`

`AgentOrchestrator` is a persistent container: agents register, exchange
messages, own tasks. Sub-agents are the opposite — ephemeral, one-shot,
single-result. Mixing both into the same module would force the persistent
types to grow opaque "spawn-and-forget" modes and callers would have to know
which flavour applies. A separate module keeps the mental model clean and lets
the caller pick exactly one (or compose both).

### Why the default runner is LLM-free

The default `InProcessSubAgentRunner` does **not** call an LLM. It is a thin
harness that returns a structured `Completed` result echoing the spec. Callers
wire an LLM-backed runner on top when they want real delegation. Keeping the
default LLM-free preserves the library's "no required network deps" property
and keeps `cargo test --features sub-agents` fully hermetic.

### Why `IsolationLevel::ExternalProcess` returns `Deferred`, not `Failed`

`Deferred` signals "this runner cannot handle the spec, please route it to
another runner." `Failed` would imply the sub-agent ran and did not succeed,
which is different. The distinction lets callers chain runners: try the
default first, fall back to a custom worktree runner on `Deferred`.

### Why `supports(spec)` is separate from `run`

Callers that maintain a list of runners want to cheaply pick the right one
before paying the cost of `run`. `supports` is the hook for that. A runner
that always returns `Deferred` for unsupported specs is also valid — the two
are equivalent up to cost.

### Why signal types are `&str` in telemetry, not typed enums

Same rationale as V93's `StallSignal`: `TelemetryCollector` compiles in every
configuration; `sub_agents.rs` does not. Using `&str` lets telemetry stay
independent of the feature gate while still getting a stable signal name
(`"Fork"` / `"Teammate"` / `"Explore"` / `"InProcess"` / …). `SubAgentKind::as_str()`
and `IsolationLevel::as_str()` / `SubAgentStatus::as_str()` return exactly
those strings, so callers with the feature on can pass `.as_str()` directly.

## Feature gating

```
sub-agents = ["multi-agent", "analytics"]
```

Both are zero-dep feature gates in this crate. Enabling `sub-agents` adds no
new transitive dependencies.

## Version bump

`0.2.25 → 0.2.26` (patch-level; additive, no API breakage).

## Verification

```bash
cargo build --features sub-agents
cargo test --features sub-agents --lib sub_agents
# → 15 passed

cargo test --features sub-agents --lib telemetry::tests::test_record_sub_agent
# → 2 passed

cargo test --features sub-agents --lib opentelemetry_integration::tests::test_start_sub_agent
# → 1 passed

cargo build   # default features, no sub-agents
# → clean
```

## Roadmap (task #154 → #155)

* LLM-backed default runner: optional feature that wires a sub-agent runner
  through the existing `AiAssistant` provider stack, using `PromptPreset::Minimal`
  and a budget-aware prompt. Would live behind `sub-agents-llm`.
* Worktree-backed runner: thin adapter for git worktrees (depends on the
  caller providing git access). Probably lives outside the crate proper as a
  companion example, not as a feature — consistent with the
  `feedback_library_framing.md` guideline that host-level resources stay with
  the caller.
* Budget enforcement: the default runner ignores `budget_hint`. A future
  `BudgetedRunner` wrapper that turns the hint into a token/time ceiling is a
  good next step.
