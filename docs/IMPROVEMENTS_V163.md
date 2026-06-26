# IMPROVEMENTS_V163 — split `advanced_routing.rs` into a module

**Version:** 0.2.114 → 0.2.115
**Scope:** `src/advanced_routing.rs` → `src/advanced_routing/`
**Feature:** none new

## Why

The code-quality audit flagged `advanced_routing.rs` (~9.6K lines) as the
clearest "split-worthy" god file: one cohesive domain (request routing)
but bundling ≥6 independent router implementations plus their snapshots,
MCP tools and ~4K lines of tests in a single file. V161 deferred it;
V163 does the split.

## What changed

Pure reorganization — **no logic, no behavior, no public-path changes.**
`src/advanced_routing.rs` became `src/advanced_routing/mod.rs` plus nine
cohesive submodules:

| Submodule | ~lines | Contents |
|-----------|-------:|----------|
| `mod.rs` | 186 | crate/module docs, shared foundational types (`ArmId`, `QueryFeatures`, `RoutingOutcome`, `ArmFeedback`), `AdvancedRoutingError`, and `pub use <submodule>::*` re-exports |
| `bandit.rs` | 1760 | the MAB core — `BanditRouter`, `BanditConfig`, arms, reward policy, routing context/preferences |
| `automata.rs` | 2693 | NFA/DFA routers, `NfaDfaCompiler`, NFA-rule synthesis, NFA/DFA snapshots + mergers |
| `hierarchical.rs` | 561 | `RoutingDag` (hierarchical DAG router) |
| `ensemble.rs` | 478 | `EnsembleRouter`, `RoutingVoter` trait, voting strategies |
| `contextual.rs` | 1985 | `ContextualDiscovery`, `AdaptivePerQueryRouter`, query feature extraction |
| `bootstrap.rs` | 540 | `EvalFeedbackMapper`, `BanditBootstrapper` (all `eval-suite`-gated) |
| `distributed.rs` | 698 | distributed bandit state + snapshot merge/serialize |
| `pipeline.rs` | 1153 | `RoutingPipeline`, `PipelineConfig`, pipeline snapshot |
| `mcp_tools.rs` | 861 | `register_routing_tools` (MCP integration) |

### Public API preserved
`src/lib.rs` was **not touched** — its `pub mod advanced_routing;` works
unchanged for a directory module, and every `advanced_routing::<Item>`
re-export still resolves because `mod.rs` does `pub use <submodule>::*`
for all submodules. External callers see an identical API.

### Tests co-located
The single end-of-file `#[cfg(test)] mod tests` block was partitioned so
each test now lives in a `mod tests` inside the submodule whose
(often-private) items it exercises — the idiomatic layout. Test count is
identical: **256** (default) / **272** (`--features eval-suite`),
zero lost or disabled.

### Minimal visibility widening
Three production cross-module accesses required widening fields to
`pub(crate)` (a strict widening — cannot break callers):
- `BanditRouter`'s 6 fields — read/reconstructed by `distributed.rs`
  (snapshot export/import).
- `NfaRouter`'s 3 fields — reconstructed by `pipeline.rs` from a snapshot.
- `AdaptivePerQueryRouter.domain_bandits` — used by `ensemble.rs`'s
  `RoutingVoter` impl.

## Tests

Verified: rustfmt clean; clippy `-D warnings` clean (lib + bins,
`FEATURES_STD`, toolchain 1.93.0 = CI); `cargo test` count unchanged;
`ai_test_harness --all` 585/585.

## Still deferred

- Sweep the 59 `#[allow(dead_code)]` (V164).
