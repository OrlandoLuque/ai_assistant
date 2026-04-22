# V96 — Self-Learning (Skill Forge + Fragment Synthesis + Feedback Loop)

**Version:** 0.2.28
**Date:** 2026-04-22
**Scope:** Three opt-in subsystems that let the crate learn from its own
execution traces, combined with a uniform CLI + GUI auditor story.

## Why

V95 closed the "single-run quality" loop (StallHeuristic + LLM-light backend).
V96 closes the "across-run improvement" loop: promote skills that work,
select prompt fragments that yield higher reward, and feed the whole thing
back via an auditable dispatcher. Everything is opt-in and gated behind
independent feature flags so callers can adopt one phase at a time.

## F1 — Skill Forge (`skill-forge`)

Declarative DSL + WASM-Rust execution. Every skill carries a content Blake3
hash and an Ed25519 signature. The `SkillLedger` chains registry events
(Registered / Updated / PromotionChanged / Retired) with Blake3 self-hashes
so tampering is detectable by anyone who has the chain.

Capability model: path globs (e.g. `/data/**/*.txt` matches nested
directories via a recursive glob matcher), net allow-list, fuel + memory
caps enforced by wasmtime's `ResourceLimiter`. The declarative executor
rejects capabilities that are not explicitly granted.

Promotion is pipelined through 6 gates (declared in `promotion.rs`); a
skill only becomes `Active` after every gate returns `Pass`.

**Binaries**: `ai_skills` (list / inspect / verify / ledger-verify /
ledger-show / export), `ai_skills_gui` (directory browser with live chain
verification).

**Tests**: 58.

## F2 — Fragment Synthesis (`prompt-synthesis`)

Contextual bandit over prompt-fragment combinations. Embeddings produced by
the existing `embeddings` subsystem are bucketed into `IntentCluster`s by an
adaptive `IntentClusterManager` (bounds `[1, 64]`, grow threshold 0.80 on
cosine similarity, idle prune). Each cluster carries a pool of `PromptArm`s
partitioned by `ProviderFingerprint` so reward from a local model never
pollutes a cloud model's statistics.

Selection: Bayesian UCB with Beta(α=1, β=1) prior + the classic UCB1 bonus
`μ + c·√(ln(total)/samples)` where `c = √2`. A 5% ε-random floor prevents
local-minima lock-in — these random picks are tagged with `ArmOrigin::EpsilonRandom`
so downstream analytics can distinguish "real" exploration from safety
noise. A deterministic xorshift\* PRNG keeps the ε-random stream reproducible.

Reward: fixed weights in v1 (success 0.5, latency 0.1, faithfulness 0.25,
user 0.15). Weights are normalized at construction; negative weights become
zero; all-zero weights fall back to uniform.

`FragmentLedger` records ArmCreated / ArmSelected / RewardRecorded /
ArmRetired / ClusterResized / FreezeChanged, Blake3-chained.

**Binaries**: `ai_prompt_synth` (ledger-show / ledger-verify / arms-summary),
`ai_prompt_synth_gui` (cluster list → arm table → reward sparklines per arm).

**Tests**: 48.

## F3 — Feedback Loop (`feedback-loop`)

The dispatcher that closes the loop. A `TrajectoryRecord` carries principal,
fragment arm id, skill ids used, intent cluster, outcome, `RewardComponents`,
and a `PrivacyTier`. `FeedbackDispatcher::submit` runs the pipeline inline:

1. Ledger receipt (always, even if dropped — audit trail intact).
2. Freeze check (`LearningFreezeConfig::freeze_feedback_loop`).
3. Privacy drop for `PrivacyTier::Confidential`.
4. `minimum_sources` check (default 2) — defense against reward hacking via
   single signal.
5. Fan-out to registered `FeedbackSink`s. Idempotent on `TrajectoryId`
   (sinks silently no-op duplicates). Partial sink failure is ledgered but
   does not abort the dispatch.

`FeedbackQueue` sits in front of the dispatcher when producers can't block
on sink fan-out. Priority lane preempts normal on pop; normal lane drops
oldest on overflow; priority lane errors rather than drop.

`RetractionLedger` is a sibling chain dedicated to GDPR / compliance events.
`FeedbackDispatcher::retract` fans `sink.retract(&id)` across every sink
and records each propagation.

Included sinks: `CollectorSink` (in-memory, tests + dry-run), `FailingSink`
(tests), `DatasetWriter` (JSONL for GEPA/MIPRO — skips `Confidential`,
writes tombstones on retract).

**Binaries**: `ai_feedback` (ledger-show / ledger-verify / retractions /
stats), `ai_feedback_gui` (overview + sinks + drops + retractions + events).

**Tests**: 35.

## Wiring

- `src/lib.rs` declares `pub mod skill_forge`, `pub mod prompt_synthesis`,
  `pub mod feedback_loop` under their feature gates and re-exports the
  public API (with `Feedback*` / `Fragment*` aliases where names collide
  with existing types).
- `Cargo.toml` adds `[[bin]]` entries for all six new binaries with
  `required-features` matching their module.
- `src/learning_control.rs` gets three new `freeze_*` fields, three new
  `LearningSubsystem` variants, and updates the Display impl + freeze
  count (now 11).

## Threat model highlights

- **Skill poisoning via trajectory** — skill output flagged `untrusted`,
  not promoted directly to memory.
- **Reward hacking via user_signal** — dispatcher enforces
  `minimum_sources ≥ 2` before accepting a record.
- **Embedding model swap** — `ProviderFingerprint` + `IntentEmbedding` carry
  model identity so cluster assignments become invalid on change.
- **Queue overflow DoS** — bounded queue with drop-oldest; priority lane
  refuses silent drops and returns `QueueError::PriorityFull`.
- **Privacy retraction post-dispatch** — `RetractionLedger` + per-sink
  `retract()` contract + `DatasetWriter` tombstones.
- **Freeze bypass** — `LearningFreezeConfig` fields wired into all three
  subsystems. Every mutator consults the runtime `is_frozen()` before
  acting: `FeedbackDispatcher::set_frozen(true)` still records the
  trajectory in the ledger (as `Dropped{reason:"frozen"}`) but never
  forwards; `FragmentBandit::set_frozen` rejects `add_arm`/reward updates
  with `BanditError::Frozen`; `SkillRegistry::set_frozen` rejects
  `insert`/`set_status` with `SkillRegistryError::Frozen`. The per-skill
  `SkillStatus::Frozen` is an orthogonal lifecycle state of a single
  skill, not the global switch.

## Counts

| | F1 | F2 | F3 | total |
|---|---|---|---|---|
| Binaries added | 2 | 2 | 2 | 6 |
| Tests added | 58 | 48 | 35 | 141 |
| Feature flag | `skill-forge` | `prompt-synthesis` | `feedback-loop` | |

## Not in scope

Per V96 plan: adaptive reward weights (deferred to V97+), async worker
supervisor inside the dispatcher (callers run their own), full GEPA/MIPRO
integration (dataset writer lands the file, trainer is external).
