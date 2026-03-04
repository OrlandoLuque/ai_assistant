# v27 Improvements Changelog

> Date: 2026-03-04

v27 adds the **Advanced Routing System** — a complete model routing framework combining Multi-Armed Bandits (Thompson Sampling, UCB1, epsilon-greedy), feature-matching NFAs, DFA compilation (powerset construction + Hopcroft minimization), closed-loop pipeline, MCP runtime control, export/import, merge operations, distributed sharing, and zero-config defaults.

---

## Summary Metrics

| Metric | v26 | v27 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 6,094 | 6,316 | +222 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 1 | +1 (`advanced_routing.rs`) |
| Lines added | 0 | ~6,500 | +6,500 |

---

## New Module: `advanced_routing.rs` (~6,500 lines, ~183 tests)

**Purpose**: Unified model routing system combining online learning (bandit), declarative rules (NFA), and compiled fast routing (DFA) in a self-improving closed loop.

### Section 1: Multi-Armed Bandit Router

- **`BanditRouter`** — Per-task-type bandit with 3 strategies
  - `BanditStrategy::ThompsonSampling` — Bayesian posterior sampling (Beta distribution via Jöhnk algorithm)
  - `BanditStrategy::Ucb1` — Upper Confidence Bound, deterministic
  - `BanditStrategy::EpsilonGreedy { epsilon }` — random exploration with probability ε
  - `BanditConfig` — strategy, priors, decay factor, min pulls before prune
  - `BanditArm` — per-arm stats: pull_count, total_reward, BetaParams(alpha, beta)
  - Per-task bandits (`HashMap<String, Vec<BanditArm>>`) + global bandit
  - Custom LCG PRNG (deterministic, no external `rand` dependency)
  - Methods: `select()`, `record()`, `warm_start()`, `warm_start_for_task()`, `add_arm()`, `add_arm_for_task()`, `remove_arm()`, `task_types()`, `all_arms()`, `all_arms_vec()`, `total_pulls()`

### Section 2: Feature-Matching NFA

- **`NfaRouter`** — Non-deterministic finite automaton for rule-based routing
  - `NfaState` — id, label, accepting_arm, priority
  - `NfaSymbol` — Domain, ComplexityRange, TokenRange, BoolFeature, Epsilon, Any
  - `QueryFeatures` — domain, complexity, token_estimate, bool_features
  - Fixed-point iteration routing (follows all matching transitions until no new states)
  - Epsilon closure via BFS
  - Methods: `add_state()`, `add_transition()`, `route()`, `state_count()`, `merge()`

### Section 3: NFA→DFA Compiler

- **`NfaDfaCompiler`** — Powerset/subset construction
  - Converts NFA to equivalent DFA
  - Handles epsilon closures, symbol deduplication, accepting state resolution
  - Priority from highest-priority accepting NFA state in each DFA state set
- **`DfaRouter`** — Deterministic routing engine
  - O(transitions) routing per query
  - `minimize()` — Hopcroft's algorithm, O(n log n) partition refinement

### Section 4: Ensemble & DAG Routing

- **`EnsembleRouter`** — Combines multiple routers (Voting, ConfidenceWeighted, PriorityCascade)
- **`RoutingDag`** — Directed acyclic graph for complex routing pipelines
- **`RoutingVoter`** — Plurality voting across router suggestions

### Section 5: NFA Rule Builder

- **`NfaRuleBuilder`** — Fluent builder for declarative NFA construction
  - `.rule(label).when(symbol).and(symbol).route_to(arm).priority(p).done()`
  - `.fallback(arm, priority)` — catch-all via `Any` transition
  - `.build()` → `Result<NfaRouter, AdvancedRoutingError>`
  - Each rule becomes a chain: `start →[cond1]→ inter →[cond2]→ ... → accepting`

### Section 6: Bandit→NFA Synthesizer

- **`BanditNfaSynthesizer::synthesize(bandit, min_pulls, quality_threshold)`**
  - Iterates all task types, ranks arms by mean reward
  - Best arm per task → Domain rule with priority = quality × 100
  - Arms above threshold → alternative paths
  - Global best → fallback

### Section 7: NFA/DFA Export/Import

- **`NfaSnapshot`** / **`DfaSnapshot`** — versioned serializable snapshots
  - `to_json()` / `from_json()` — JSON serialization
  - `to_bytes()` / `from_bytes()` — bincode (with `binary-storage` feature) or JSON fallback
  - Version checking on deserialization

### Section 8: NFA Merge

- **`NfaRouter::merge(&self, &other)`** — NFA union construction
  - New start state with epsilon transitions to both originals
  - State renumbering to avoid ID conflicts
- **`merge_and_compile_nfas(a, b)`** — Merge + compile + minimize in one step

### Section 9: Distributed NFA Sharing (cfg `distributed`)

- **`DistributedNfaState`** — node_id, timestamp, NfaSnapshot
- **`NfaStateMerger`** — extract, merge multiple nodes' NFAs, merge_into_router
- Federated bandit merging: `sum(alpha) - (N-1)*prior` to avoid double-counting

### Section 10: Closed-Loop Pipeline

- **`RoutingPipeline`** — Orchestrates bandit + NFA + DFA lifecycle
  - `route(features)` — DFA if available, else bandit fallback
  - `record_outcome(feedback)` — updates bandit, triggers auto-resynthesis
  - `maybe_resynthesize()` / `force_resynthesize()` — bandit→NFA→DFA cycle
  - `seed_bandit_from_nfa()` — extracts arms from NFA, warm-starts priors
  - `with_initial_nfa()`, `with_initial_rules()` — start with pre-built NFA
  - `export_snapshot()` / `to_json()` / `from_json()` — full pipeline serialization
- **`PipelineConfig`** — synthesis_interval, min_pulls, quality_threshold, auto_minimize
- **`PipelineSnapshot`** — bandit + NFA + config serialized together

### Section 11: Zero-Config Constructors

- **`RoutingPipeline::for_models(&[&str], config)`** — Pure bandit, auto-synthesizes
- **`RoutingPipeline::with_tiered_models(&[(&str, ModelTier)], config)`** — Auto NFA rules by tier
- **`ModelTier`** — Premium (code+complex), Standard (medium), Economy (simple+fallback)

### Section 12: MCP Runtime Routing Tools

- **`register_routing_tools(server, Arc<Mutex<RoutingPipeline>>)`** — 10 MCP tools:
  - `routing.get_stats` — bandit statistics
  - `routing.add_arm` — add model with optional priors
  - `routing.remove_arm` — remove model
  - `routing.warm_start` — set alpha/beta priors
  - `routing.record_outcome` — feed back results
  - `routing.add_rule` — add NFA rule, recompile DFA
  - `routing.force_resynthesize` — force bandit→NFA→DFA
  - `routing.export` / `routing.import` — pipeline state transfer
  - `routing.get_config` — read configuration

---

## Changes to Existing Files

### `src/error.rs`

- `AdvancedRoutingError` enum with variants: `InvalidConfig`, `CompilationError`, `NoRoutingPath`, `SerializationFailed`, `IncompatibleVersion`, `MergeConflict`, `SynthesisError`

### `src/lib.rs` (+15 lines)

- Added re-exports: `BanditRouter`, `BanditConfig`, `BanditStrategy`, `BanditArm`, `BanditSnapshot`, `BetaParams`, `NfaRouter`, `NfaState`, `NfaSymbol`, `NfaRuleBuilder`, `NfaSnapshot`, `DfaRouter`, `DfaSnapshot`, `NfaDfaCompiler`, `EnsembleRouter`, `RoutingDag`, `RoutingVoter`, `QueryFeatures`, `QueryFeatureExtractor`, `RoutingOutcome`, `ArmFeedback`, `RoutingPipeline`, `PipelineConfig`, `PipelineSnapshot`, `ModelTier`, `BanditNfaSynthesizer`, `merge_and_compile_nfas`, `register_routing_tools`
- `#[cfg(feature = "distributed")]`: `DistributedBanditState`, `BanditStateMerger`, `DistributedNfaState`, `NfaStateMerger`

---

## Design Decisions

1. **Custom LCG PRNG instead of `rand` crate**: The bandit needs random sampling for Thompson Sampling and epsilon-greedy. Using a built-in LCG (linear congruential generator) avoids adding `rand` as a dependency and makes tests deterministic (seeded state).

2. **Fixed-point iteration for NFA routing**: A single transition step is insufficient for multi-condition rule chains (e.g., Domain→Complexity→Accept). The route function iterates until no new states are reachable, supporting arbitrary chain depth.

3. **Priority-based disambiguation**: When multiple NFA accepting states are reached, the highest priority wins. This is simpler and more predictable than weighted scoring.

4. **Bandit seeding from NFA**: When the pipeline starts with an initial NFA, it reverse-engineers the NFA structure to populate the bandit with arms and warm-start their priors. This ensures the bandit doesn't start from scratch when rules already exist.

5. **DFA minimization after merge**: `merge_and_compile_nfas()` automatically calls `minimize()` after compilation. Union construction + powerset can produce redundant states; Hopcroft eliminates them.

6. **Arc<Mutex<>> for MCP tools**: The 10 MCP tool handlers share a single pipeline via `Arc<Mutex<RoutingPipeline>>`. This is the same pattern used by the HTTP server endpoints. Lock contention is not a concern because routing operations are fast (microseconds).

---

## Test Summary (~222 new tests across v27)

| Component | Tests | Key scenarios |
|-----------|-------|---------------|
| BanditRouter | 12 | Thompson, UCB1, ε-greedy, warm-start, per-task, decay, remove_arm |
| NfaRouter | 14 | route, epsilon closure, fixed-point, chains, priority |
| DfaRouter | 8 | compile, minimize, route, state merging |
| NfaRuleBuilder | 8 | single rule, chained conditions, fallback, priority |
| BanditNfaSynthesizer | 6 | synthesis, min_pulls filter, quality threshold |
| NFA/DFA Export/Import | 12 | JSON round-trip, version check, bytes |
| NFA Merge | 6 | union, renumbering, merge+route, chain 3 |
| RoutingPipeline | 16 | route, resynthesize, seed, for_models, tiered |
| MCP Tools | 12 | all 10 tools + workflow integration |
| Ensemble/DAG | 12 | voting, confidence, priority cascade, DAG |
| Distributed (cfg) | 8 | bandit state merge, NFA state merge, extract |
| Integration | 5 | end-to-end cycles, task types |

---

## Documentation Updates

- **CONCEPTS.md**: Added sections 150-157 covering Multi-Armed Bandits, Thompson Sampling/UCB1/ε-greedy, Feature-Matching NFA, Powerset Construction, Hopcroft Minimization, NFA Builder & Synthesizer, Closed-Loop Pipeline, MCP Runtime Routing
- **GUIDE.md**: Added section 136 with code examples for all routing APIs
- **AGENT_SYSTEM_DESIGN.md**: Added section 53 with architecture diagrams, algorithm tables, component relationships, and competitive analysis
- **TESTING.md**: Updated test count to 6,316 and added routing test coverage

---

## Verification

```bash
# Compile check
cargo check --features full,eval-suite

# Run routing tests
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests

# Distributed routing tests
cargo test --features "full,eval-suite" --lib -- advanced_routing::tests::distributed

# Full regression
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite" --lib

# Clippy
cargo clippy --features "full,eval-suite" -- -D warnings
```

Expected: 6,316 tests, 0 failures, 0 clippy warnings.
