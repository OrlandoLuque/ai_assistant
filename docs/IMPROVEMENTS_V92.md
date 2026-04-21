# IMPROVEMENTS V92 — Claude Code permission-label adapter (0.2.24)

## Context

Claude Code groups action permissions using a five-label vocabulary —
`ReadOnly`, `WorkspaceWrite`, `DangerFullAccess`, `Prompt`, `Allow` — that is
convenient for docs, UIs, and user-facing explanations. Our internal permission
model speaks a different language: `ActionType` + `RiskLevel` + policy
configuration (`require_approval_above`, per-tool overrides, deny lists).

V92 ships an additive **presentation-layer adapter** that maps our triple onto
Claude Code's labels, without touching the runtime or altering the internal
taxonomy. Callers that want to render permissions in Claude Code's vocabulary
now can; everyone else is unaffected.

## Scope

* New `DefaultDecision` enum in `src/agent_policy.rs`:
  `Allow` / `Prompt` / `Deny`.
* New `PermissionRequirement` struct in `src/agent_policy.rs`:
  `(ActionType, RiskLevel, DefaultDecision)` triple.
* `PermissionRequirement::new(action_type, risk, default_decision)`.
* `PermissionRequirement::from_policy(&AgentPolicy, &ActionDescriptor)` —
  reuses `policy.assess_risk()` and `policy.needs_approval()` so the label
  stays in sync with runtime behaviour.
* `PermissionRequirement::to_claude_code_label(&self) -> &'static str` —
  returns one of the five Claude Code labels.
* 12 unit tests covering every branch of the mapping table plus the four
  relevant policy presets (default / paranoid / autonomous).

All symbols are re-exported under `feature = "autonomous"` from `src/lib.rs`.

## Mapping

| Condition                                                          | Label               |
|--------------------------------------------------------------------|---------------------|
| `default_decision == Prompt` or `Deny`                             | `"Prompt"`          |
| Auto-`Allow` + `FileRead` (any risk)                               | `"ReadOnly"`        |
| Auto-`Allow` + `ToolCall` / `McpCall` / `HttpRequest` at Safe/Low  | `"ReadOnly"`        |
| Auto-`Allow` + `FileWrite` / `FileDelete` / `BrowserAction`        | `"WorkspaceWrite"`  |
| Auto-`Allow` + `ShellExec`                                         | `"DangerFullAccess"`|
| Auto-`Allow` + any action at `High` / `Critical` risk              | `"DangerFullAccess"`|
| Anything else still auto-`Allow`                                   | `"Allow"`           |

Claude Code's label set has no explicit `Deny`; denials surface as `"Prompt"`
(the user is asked and the policy rejects). Callers that need the distinction
read `requirement.default_decision` directly.

## Why an adapter, not a rename

* The internal triple carries strictly more information than the label
  (risk level is useful for telemetry, thresholds, and analytics). We do not
  want to collapse it.
* Claude Code's labels are product-specific. Other consumers (RBAC systems,
  compliance exports) speak different vocabularies. Keeping the label as a
  presentation helper lets us add more adapters later without churn.

## Feature gating

`PermissionRequirement` and `DefaultDecision` live behind the existing
`feature = "autonomous"` flag alongside `AgentPolicy`. No new deps, no API
breakage, no runtime paths touched.

## Version bump

`0.2.23 → 0.2.24` (patch-level; additive only).

## Verification

```bash
cargo test --features autonomous --lib agent_policy::tests
# → 29 passed, 0 failed
```
