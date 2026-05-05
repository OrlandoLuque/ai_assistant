# V117 — Phase C.2: ErrorCode rollout to long-tail subsystems

**Date**: 2026-05-05
**Version**: 0.2.64
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #325 (long-tail subsystems error migration)

## Why

V114 migrated the umbrella `AiError` and the 8 most-used sub-types.
V115 added the RAG triad. V116 added the provider/network triad.
After those three iterations, 16 types emitted fine-grained codes —
but the long-tail subsystems (`Workflow`, `AdvancedMemory`, `A2A`,
`VoiceAgent`, `MediaGeneration`, `Distillation`, `ConstrainedDecoding`,
`Hitl`, `McpClient`, `AgentEval`, `RedTeam`, `Mcts`, `DevTools`,
`EvalSuite`, `AdvancedRouting`) still flattened to coarse strings
through the `AiError::ErrorCode` impl.

That meant a `MctsError::RefinementExhausted { iterations, last_improvement }`
and a `MctsError::NoValidActions { state_description }` both surfaced
as `"MCTS"` in the trait code path. Dashboards that segment by error
code couldn't distinguish "search budget exhausted" from "dead-end
state" without parsing free-text.

V117 closes that gap for every long-tail subsystem under `AiError`.

## What changed

### 1. 15 new `impl ErrorCode` blocks

Each long-tail wrapper now has its own trait impl in `src/error.rs`:

| Type | Codes |
|---|---|
| `WorkflowError` | 8 (`WORKFLOW_NODE_NOT_FOUND`, `WORKFLOW_CYCLE_DETECTED`, …) |
| `AdvancedMemoryError` | 6 (`MEMORY_STORE_FAILED`, `MEMORY_CAPACITY_EXCEEDED`, …) |
| `A2AError` | 7 (`A2A_TASK_NOT_FOUND`, `A2A_PROTOCOL_ERROR`, …) |
| `VoiceAgentError` | 6 (`VOICE_STREAM_FAILED`, `VOICE_VAD_ERROR`, …) |
| `MediaGenerationError` | 6 (`MEDIA_PROVIDER_UNAVAILABLE`, `MEDIA_JOB_TIMEOUT`, …) |
| `DistillationError` | 6 (`DISTILL_NO_VALID_TRAJECTORIES`, …) |
| `ConstrainedDecodingError` | 5 (`CDEC_GRAMMAR_COMPILE_FAILED`, `CDEC_VALIDATION_FAILED`, …) |
| `HitlError` | 6 (`HITL_APPROVAL_TIMEOUT`, `HITL_POLICY_VIOLATION`, …) |
| `McpClientError` | 7 (`MCP_CLIENT_CONNECTION_FAILED`, `MCP_CLIENT_PROTOCOL_MISMATCH`, …) |
| `AgentEvalError` | 6 (`EVAL_TRAJECTORY_EMPTY`, `EVAL_TOOL_CALL_MISMATCH`, …) |
| `RedTeamError` | 5 (`REDTEAM_GENERATION_FAILED`, `REDTEAM_INVALID_CATEGORY`, …) |
| `MctsError` | 6 (`MCTS_MAX_ITERATIONS`, `MCTS_REFINEMENT_EXHAUSTED`, …) |
| `DevToolsError` | 5 (`DEVTOOLS_RECORDING_FAILED`, `DEVTOOLS_BREAKPOINT_INVALID`, …) |
| `EvalSuiteError` | 10 (`EVAL_SUITE_DATASET_LOAD_FAILED`, `EVAL_SUITE_TIMEOUT`, …) |
| `AdvancedRoutingError` | 10 (`ROUTING_INVALID_CONFIG`, `ROUTING_NO_PATH`; `ROUTING_MERGE_CONFLICT` gated on `feature = "distributed"`) |

**Total: 99 new fine-grained codes.**

Each `fields()` extracts the variant payload as `(&'static str, String)`
pairs — strings are cloned, numerics use `.to_string()`, `f64`s format
as `"{:.4}"`. Empty-payload variants (`AdvancedRoutingError::CycleDetected`,
`AdvancedRoutingError::EmptyEnsemble`) return `Vec::new()`.

### 2. `AiError::ErrorCode::code()` flipped from coarse to delegating

Previously the long-tail arms in `<AiError as ErrorCode>::code()`
returned coarse fallbacks (`"WORKFLOW"`, `"MEMORY"`, …). They now
delegate: `AiError::Workflow(e) => e.code()`, etc.

The **inherent** `AiError::code()` (called as `err.code()` without trait
disambiguation) is unchanged — it still returns coarse category strings
for API compatibility.

This is the same dual-access pattern V114 introduced for the
already-migrated subsystems:

```rust
let e = AiError::Workflow(WorkflowError::NodeNotFound { node_id: "n1".into() });

// Coarse — for backward compat:
assert_eq!(e.code(), "WORKFLOW");

// Fine-grained — for telemetry / structured logs:
assert_eq!(<AiError as ErrorCode>::code(&e), "WORKFLOW_NODE_NOT_FOUND");
let f = <AiError as ErrorCode>::fields(&e);
assert_eq!(f[0], ("node_id", "n1".to_string()));
```

### 3. i18n catalog

`errors/en.json` and `errors/es.json` both grew **83 → 182 codes (+99)**.
Every new code has both locales with `{field}` placeholder substitution.
Examples:

```json
"WORKFLOW_NODE_NOT_FOUND": "Workflow node not found: {node_id}",
"MCTS_REFINEMENT_EXHAUSTED": "MCTS refinement exhausted after {iterations} iterations (last improvement: {last_improvement})",
"HITL_APPROVAL_TIMEOUT": "HITL approval for tool '{tool_name}' timed out after {timeout_secs}s"
```

Spanish mirror:

```json
"WORKFLOW_NODE_NOT_FOUND": "Nodo del workflow no encontrado: {node_id}",
"MCTS_REFINEMENT_EXHAUSTED": "Refinamiento MCTS agotado tras {iterations} iteraciones (última mejora: {last_improvement})",
"HITL_APPROVAL_TIMEOUT": "La aprobación HITL para la herramienta '{tool_name}' caducó tras {timeout_secs}s"
```

## Tests

- **16 new** `test_errorcode_*` tests in `error::tests` (one per
  long-tail type + one cross-cutting localization spot-check).
- **1 renamed**: `test_errorcode_aierror_long_tail_keeps_coarse` →
  `test_errorcode_aierror_long_tail_delegates` (still asserts the
  dual-access pattern; the assertion just flipped from coarse-only
  to fine-grained-via-trait + coarse-via-inherent).
- All **27** `test_errorcode_*` tests pass.

## Coverage state after V117

| Iteration | Types added | Cumulative codes (en/es) |
|---|---|---|
| V113 (core) | trait + StructuredError + 4 seed codes | 4 |
| V114 (umbrella) | AiError + 8 sub-types | 42 |
| V115 (RAG triad) | RagPipeline + Embedding + Kpkg | 65 |
| V116 (provider triad) | Anthropic + OpenAI + HF + Resilient | 83 |
| **V117 (long-tail)** | **15 types** | **182** |

`AiError`'s 24 wrapping variants are now fully covered: every
`AiError::*(inner)` pattern resolves through the trait to a
fine-grained code with structured fields. The umbrella migration
(originally Phase C.2) is complete.

## What's next

- **V118**: wire `StructuredError::to_json()` into
  `opentelemetry_integration.rs::AiSpan` — sets `error.code` and
  `error.fields.*` attributes from `StructuredError::from_err(&err)`
  on every span that records an error. This is the payoff for the
  V113–V117 chain: dashboards finally see the fine-grained codes.
- **Optional follow-up**: long-tail submodule errors that don't sit
  under `AiError` directly — `BulkheadError`, `RetryableError`,
  `BrowserError`, etc. They're lower-priority because they're
  internal to specific subsystems and rarely cross subsystem
  boundaries.
