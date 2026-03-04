# v28 Improvements Changelog

> Date: 2026-03-04

v28 enhances the **Advanced Routing System** (v27) with 6 comprehensive improvements: composite reward policy, per-query routing preferences, private arms for distributed sharing, auto-benchmark bootstrapping, extended routing context, and feature importance tracking.

---

## Summary Metrics

| Metric | v27 | v28 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 6,316 | 6,401 | +85 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 0 | 0 |
| Lines added (advanced_routing.rs) | ~6,500 | ~7,900 | +1,400 |

---

## GAP A: Composite Reward — RewardPolicy

**Problem**: `ArmFeedback.latency_ms` and `.cost` were stored but never used in reward computation — reward was quality-only.

**Solution**: `RewardPolicy` computes composite reward from quality + latency + cost with configurable weights and normalization references.

### New types

- **`RewardPolicy`** — configurable composite reward policy
  - `quality_weight: f64` (default 0.7)
  - `latency_weight: f64` (default 0.2)
  - `cost_weight: f64` (default 0.1)
  - `latency_ref_ms: f64` (normalization reference, default 5000.0)
  - `cost_ref: f64` (normalization reference, default 0.01)
  - `compute_reward(&self, feedback: &ArmFeedback) -> f64` — weighted composite
  - `normalize_weights(&self) -> (f64, f64, f64)` — sum-to-1 normalization

### Changes to existing code

- `BanditConfig` — added `#[serde(default)] pub reward_policy: RewardPolicy`
- `BanditRouter::record_outcome()` — replaced inline reward with `self.config.reward_policy.compute_reward(feedback)`
- `RoutingPipeline::record_outcome_with_context()` — same replacement

### Backward compatibility

When `latency_ms=None` and `cost=None` (the common case), weight redistribution gives 100% to quality → identical behavior to v27.

### Tests (12)

`test_reward_policy_default_values`, `test_reward_policy_quality_only`, `test_reward_policy_all_components`, `test_reward_policy_all_weights_zero`, `test_reward_policy_latency_none_redistributes`, `test_reward_policy_cost_none_redistributes`, `test_reward_policy_both_none_quality_only`, `test_reward_policy_high_latency_penalized`, `test_reward_policy_high_cost_penalized`, `test_reward_policy_zero_ref_values`, `test_record_outcome_uses_reward_policy`, `test_record_outcome_backward_compat`

---

## GAP B: Per-Query Routing Preferences

**Problem**: No way to say "select model ignoring cost" or "prefer fast models for this query."

**Solution**: `RoutingPreferences` allows per-query weight overrides (applied at recording time) and arm exclusion/boosting (applied at selection time).

### New types

- **`RoutingPreferences`** — per-query overrides
  - `quality_weight: Option<f64>`, `latency_weight: Option<f64>`, `cost_weight: Option<f64>` — override base policy weights
  - `excluded_arms: Vec<ArmId>` — arms to exclude from selection
  - `preferred_arms: Vec<ArmId>` — arms to boost during selection
  - `prefer_boost: f64` (default 2.0) — multiplier for preferred arms
  - Convenience constructors: `ignore_cost()`, `minimize_latency()`, `quality_only()`
  - `apply_to_policy(&self, base: &RewardPolicy) -> RewardPolicy` — merge overrides

### New methods

- `BanditRouter::select_with_preferences()` — arm exclusion/boosting at selection time
- `BanditRouter::record_outcome_with_preferences()` — weight overrides at recording time
- `RoutingPipeline::route_with_preferences()` — pipeline-level routing with preferences
- `RoutingPipeline::record_outcome_with_context_and_preferences()` — pipeline-level recording

### Design decision

Weight overrides apply at **recording time**, not selection time. Reason: bandit posteriors already bake in historical composite rewards and can't be retroactively decomposed. The "ignore cost" use case works by not penalizing the arm for cost when recording THIS query's outcome → over time, the bandit learns that expensive-but-good arms are fine for these queries.

### Tests (10)

`test_routing_preferences_default`, `test_routing_preferences_ignore_cost`, `test_routing_preferences_minimize_latency`, `test_routing_preferences_apply_to_policy`, `test_select_with_preferences_excludes_arms`, `test_select_with_preferences_boosts_preferred`, `test_select_with_preferences_all_excluded_errors`, `test_record_outcome_with_preferences_custom_weights`, `test_pipeline_route_with_preferences`, `test_preferences_serialize_deserialize`

---

## GAP C: Private Arms for Distributed Sharing

**Problem**: Distributed sharing exports ALL arms/NFA rules — no way to keep local-only models private.

**Solution**: `HashSet<ArmId>` side-channel on `BanditRouter` tracks which arms should NOT be shared. No changes to `BanditArm` struct (zero impact on existing struct literals).

### New types

- **`ArmVisibility`** — `Public` (default) | `Private`

### New methods on BanditRouter

- `set_arm_private(&mut self, arm_id: &str)` — mark arm as local-only
- `set_arm_public(&mut self, arm_id: &str)` — restore shareable status
- `is_arm_private(&self, arm_id: &str) -> bool` — query visibility
- `private_arm_ids(&self) -> &HashSet<ArmId>` — get all private arms

### Changes to existing code

- `BanditRouter` — added `private_arms: HashSet<ArmId>` field
- `BanditSnapshot` — added `#[serde(default, skip_serializing_if = "HashSet::is_empty")] pub private_arms: HashSet<ArmId>`
- `BanditStateMerger::extract_state()` — filters out private arms from `global_arms` and `task_bandits`
- `NfaStateMerger::extract_state_filtered()` — new method that accepts `private_arms` parameter to filter accepting states

### Tests (11)

`test_arm_visibility_default_is_public`, `test_set_arm_private_and_query`, `test_set_arm_public_reverses_private`, `test_private_arms_accessor`, `test_extract_state_filters_private_global_arms`, `test_extract_state_filters_private_task_arms`, `test_extract_state_preserves_public_arms`, `test_private_arm_still_selectable_locally`, `test_snapshot_preserves_private_arms`, `test_nfa_extract_state_with_private_arms`, `test_nfa_extract_state_backward_compat`

---

## GAP D: Auto-Benchmark Bootstrapper

**Problem**: Cold start wastes initial queries; eval-suite results aren't auto-converted to bandit priors.

**Solution**: `BanditBootstrapper` converts `ComparisonMatrix` and `SubtaskAnalysis` results into warm-start priors, eliminating the cold-start exploration penalty.

### New types (behind `#[cfg(feature = "eval-suite")]`)

- **`BanditBootstrapper`** — converts eval results to bandit priors
  - `from_comparison_matrix(matrix, reward_policy) -> HashMap<String, HashMap<ArmId, BetaParams>>` — uses `mean_score` (metric index 1) and `cost_effectiveness`, weighted by RewardPolicy
  - `from_subtask_analysis(analysis, scale) -> HashMap<String, HashMap<ArmId, BetaParams>>` — per-subtask priors from SubtaskPerformance scores
  - `bootstrap_pipeline(priors, bandit_config, pipeline_config) -> RoutingPipeline` — full warm-started pipeline

### Algorithm

For `from_comparison_matrix`: mean_score × adjusted_quality_weight + cost_effectiveness_normalized × cost_weight → composite → alpha = composite × 10, beta = (1 - composite) × 10.

### Tests (10)

`test_bootstrapper_from_empty_matrix`, `test_bootstrapper_from_single_model`, `test_bootstrapper_from_multiple_models`, `test_bootstrapper_uses_reward_policy_weights`, `test_bootstrapper_from_subtask_analysis_empty`, `test_bootstrapper_from_subtask_analysis_basic`, `test_bootstrapper_from_subtask_analysis_multiple`, `test_bootstrapper_bootstrap_pipeline`, `test_bootstrapper_round_trip_select`, `test_bootstrapper_scale_effect`

---

## GAP E: Extended RoutingContext

**Problem**: `QueryFeatures` captures text-level info but not agent-level context (RAG active, budget, tier).

**Solution**: `RoutingContext` wraps `QueryFeatures` with agent-level metadata and auto-derives routing preferences (e.g., low budget → boost cost_weight).

### New types

- **`RoutingContext`** — extended context
  - `features: QueryFeatures`
  - `rag_active: bool`
  - `budget_remaining: Option<f64>`
  - `agent_tier: Option<String>`
  - `session_cost_so_far: Option<f64>`
  - `preferred_provider: Option<String>`
  - `new(features)`, `derive_preferences(&self, base_policy) -> RoutingPreferences`
  - `From<QueryFeatures>` implementation

### New methods on RoutingPipeline

- `route_with_context(&mut self, ctx: &RoutingContext) -> Result<RoutingOutcome, AdvancedRoutingError>` — route with auto-derived preferences

### Tests (8)

`test_routing_context_new`, `test_routing_context_from_features`, `test_derive_preferences_low_budget`, `test_derive_preferences_no_budget`, `test_derive_preferences_normal_budget`, `test_pipeline_route_with_context`, `test_routing_context_serialize_deserialize`, `test_routing_context_with_rag`

---

## GAP F: Feature Importance Tracking

**Problem**: `ContextualDiscovery` finds splits but doesn't rank which features matter most.

**Solution**: `feature_importance()` method aggregates discovered splits by dimension to rank feature discriminative power.

### New types

- **`FeatureImportance`** — per-dimension importance
  - `dimension: FeatureDimension`
  - `total_gain: f64`
  - `split_count: usize`
  - `domains_affected: usize`

### New method on ContextualDiscovery

- `feature_importance(&self) -> Vec<FeatureImportance>` — sorted by total_gain descending

### Tests (5)

`test_feature_importance_empty`, `test_feature_importance_single_dimension`, `test_feature_importance_multiple_sorted`, `test_feature_importance_domains_count`, `test_feature_importance_no_splits`

---

## Changes to Existing Files

### `src/advanced_routing.rs` (+~1,400 lines)

All 6 GAPs implemented in the existing file. Production code + tests.

### `src/lib.rs` (+6 re-exports)

- Added: `ArmVisibility`, `FeatureImportance`, `RewardPolicy`, `RoutingContext`, `RoutingPreferences`
- `#[cfg(feature = "eval-suite")]`: `BanditBootstrapper`

---

## Design Decisions

1. **Weight redistribution for missing components**: When `latency_ms` or `cost` is `None` in feedback, their weights are proportionally redistributed to active components. This means the common case (quality-only feedback) gives 100% weight to quality — backward compatible.

2. **Recording-time vs selection-time preferences**: Weight overrides at recording time, arm filtering at selection time. Bandit posteriors bake in historical rewards; changing selection scores on the fly would break the Thompson Sampling distribution.

3. **HashSet side-channel for private arms**: Instead of adding a `visibility` field to `BanditArm` (which would touch 9 struct literals across the codebase), a `HashSet<ArmId>` on `BanditRouter` tracks private arms. Zero changes to existing struct construction.

4. **BanditBootstrapper uses metric index 1**: ComparisonMatrix stores `["accuracy", "mean_score", "mean_latency_ms", "total_cost"]`. The bootstrapper uses `mean_score` (index 1) for quality and `cost_effectiveness` for cost. Latency is redistributed to quality weight since latency data isn't granular enough in ComparisonMatrix.

5. **RoutingContext auto-derives preferences**: Low budget (< 0.1) → boost cost_weight to 0.5. This is a reasonable default that can be overridden by passing explicit preferences.

---

## Test Summary (~85 new tests)

| Component | Tests | Key scenarios |
|-----------|-------|---------------|
| RewardPolicy | 12 | Composite reward, weight redistribution, backward compat |
| RoutingPreferences | 10 | Ignore cost, minimize latency, arm exclusion/boosting |
| Private Arms | 11 | Set/query private, distributed filter, snapshot persist |
| BanditBootstrapper | 10 | ComparisonMatrix, SubtaskAnalysis, bootstrap pipeline |
| RoutingContext | 8 | Auto-derive preferences, budget-based, serialize |
| Feature Importance | 5 | Sorted by gain, domains count, empty |
| Distributed (private arms) | 4 | Filter global/task arms, NFA filter, backward compat |
| Integration (eval feedback) | 6 | EvalFeedbackMapper end-to-end |
| Cross-feature integration | 19 | Pipeline preferences, context routing, bootstrapper |

---

## Verification

```bash
# Compile check
cargo check --features "full,eval-suite"

# Run new tests
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_reward_policy
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_routing_pref
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_arm_visibility
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_routing_context
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_feature_importance
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::test_bootstrapper

# Distributed tests
cargo test --features "full,eval-suite,distributed-agents,distributed-network" --lib -- advanced_routing::tests::distributed_tests

# Full regression
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite" --lib

# Clippy
cargo clippy --features "full,eval-suite,distributed-agents,distributed-network" -- -D warnings
```

Expected: 6,401 tests passed, 2 pre-existing failures (byte serialization), 0 clippy warnings.
