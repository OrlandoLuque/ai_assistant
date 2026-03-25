# Improvements V62 — Agent Methodology

Defines HOW an autonomous agent approaches tasks: workflow phases, reasoning strategies, review triggers, quality gates, and presets.

## Items

| # | Item | Estado |
|---|------|--------|
| 1 | `AgentMethodology` struct — approach, reasoning, planning, review, recovery, communication, risk tolerance | HECHO |
| 2 | `WorkflowProtocol` — 6 configurable phases (ANALYZE→PLAN→VALIDATE→EXECUTE→REVIEW→CONCLUDE) with PhaseConfig (enabled/mandatory/max_duration) | HECHO |
| 3 | `ReviewTriggers` — 8 trigger conditions (iterations, milestone, tool failure, cost, time, user interrupt, periodic self-check) | HECHO |
| 4 | `QualityGate` — GateCheck (OutputNotEmpty, ContainsKeywords, NoErrors, CostWithinBudget, TimeWithinLimit, LlmJudge) + GateAction | HECHO |
| 5 | 4 presets: Careful, Balanced, Fast, Research | HECHO |
| 6 | `should_run_phase()` and `should_review()` methods | HECHO |
| 7 | Wired into `AutonomousAgent` — `methodology` field, builder method, `should_review_now()` | HECHO |
| 8 | Re-exports in lib.rs (WorkflowPhase aliased as MethodologyPhase to avoid collision with agent_profiles::WorkflowPhase) | HECHO |
| 9 | Concepts 220-221 (Agent Methodology, Quality Gates & Review Triggers) | HECHO |
| 10 | 13 new tests (all passing) | HECHO |

## Test count

- Before: 7,069
- After: 7,082 (+13)

## Files modified

| File | Change |
|------|--------|
| `src/agent_methodology.rs` | NEW — 620 LOC, complete module |
| `src/lib.rs` | pub mod + re-exports |
| `src/autonomous_loop.rs` | methodology field, builder, should_review_now() |
| `docs/CONCEPTS.md` | Concepts 220-221 |
| `docs/TESTING.md` | Test count 7,082, V62 row |
| `docs/modus-operandi.md` | V62 entry |
