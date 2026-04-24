# V97 — PromptBreeder (self-referential prompt evolution)

**Version:** 0.2.29
**Date:** 2026-04-23
**Scope:** New opt-in subsystem `prompt_breeder` that evolves
`(task_prompt, mutation_prompt)` pairs using the algorithm from Fernando et
al. (2023, DeepMind) extended with 19 configurable axes, provider
fingerprint isolation, Blake3 hash-chained ledger, and a matching CLI + GUI
auditor — in line with `feedback_auditable_subsystems`.

## Why

V96 taught the crate to promote skills, select prompt fragments, and close
the trajectory loop. V97 closes the prompt-itself loop: the mutation
prompts used to rewrite task prompts are themselves candidates for
mutation. The breeder is feature-gated (`prompt-breeder`) so callers who
only need providers, RAG, and streaming keep a minimal dependency surface.

## Algorithm

A `Population` of `Unit`s each carries `(task_prompt, mutation_prompt)`
plus fitness, provenance, and a `ProviderFingerprint`. One generation
does:

1. **Selection** over surviving units (5 strategies: Tournament,
   RouletteWheel, RankBased, Truncation, Boltzmann).
2. **Mutation** via one of 9 operators chosen by the scheduler
   (`Uniform` / UCB1 bandit / `Adaptive` rolling window / `Curriculum`):
   - `ZeroOrder` — rewrite task prompt from scratch.
   - `FirstOrder` — rewrite guided by the unit's mutation prompt.
   - `Eda` — sample from a population-level distribution.
   - `EdaRankAndIndex` — EDA weighted by fitness rank.
   - `LineageBased` — mutate conditional on ancestor trajectory.
   - `HyperMutationZeroOrder` — rewrite the mutation prompt itself.
   - `HyperMutationFirstOrder` — rewrite mut-prompt guided by itself.
   - `Lamarckian` — learn the rewrite that lifts a specific example.
   - `PromptCrossover` — splice two parent prompts.
3. **Safety filter** — `PromptInjectionBlock` / `PiiBlock` /
   `Constitutional` / `Composite`. Violations are ledgered and rejected.
4. **Token cap** — `max_tokens` per unit (4-chars-per-token heuristic).
5. **Fitness** via `FitnessEvaluator` (ExactMatch / Contains / Regex /
   JsonSchema / LlmJudge) with optional `FitnessSmoothing`
   (`Single` / `MeanOfK` / `SelfConsistency{Majority/Plurality/BestOfN}` /
   `Bayesian` Beta(α,β)).
6. **Replacement** (`Generational` / `SteadyState` / `Elitism{k}` /
   `TournamentReplace`).
7. **Diversity** check (`EditDistance` Levenshtein / `NGramJaccard` /
   `EmbeddingCosine` fallback to Jaccard).

Every step emits an event on the `BreederLedger` (Blake3 chain), and
optional `CheckpointPolicy::Every{n_generations, path}` writes an atomic
snapshot so runs are resumable and auditable.

## Provider fingerprint isolation

`ProviderFingerprint(provider/model)` tags every `FitnessScore`. Fitness
from a cloud Opus never mixes with fitness from a local Mistral — the
breeder's selection/replacement logic always filters by fingerprint first.
Shape-compatible with `prompt_synthesis::arm::ProviderFingerprint` so a
V97.1 bridge can share statistics where it makes sense.

## Budget + cache

`BudgetMeter` tracks LLM calls, tokens, wall time, and USD cost. A breach
appends `BudgetExhausted{kind, value}` and ends the run cleanly. `EvalCache`
keyed by `(prompt, input, fingerprint, sample_idx)` avoids duplicate
evaluations under the same fingerprint and is bypassed on different
fingerprints automatically.

## Checkpoints

`Checkpoint{run_id, generation, config_hash_hex, ledger_tip_hash_hex,
population, lineage}` with MAGIC bytes `AIBR-CKPT\x01`. Written atomically
via `.tmp` + rename. `matches_config(&config)` detects shape changes so
resume is refused on an incompatible config edit.

## Binaries

- **`ai_breeder`** (read-only CLI): `list-runs <DIR>`, `show-run <CKPT>`,
  `ledger-verify <JSONL>`, `ledger-show <JSONL> [--last N]`,
  `export-population <CKPT> <OUT_JSON>`, `compare-runs <A> <B>`.
- **`ai_breeder_gui`** (egui/eframe): tabs Overview / Population /
  Lineage / Ledger / Events / Fitness. Lives-reload every 5s if enabled.

## Tests

77 passing, exceeding the plan's ≥57 target.

| | config | rng | eval | fitness | cache | budget | llm | safety | operators | population | checkpoint | ledger | breeder | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tests | 5 | 3 | 5 | 6 | 5 | 5 | 5 | 5 | 6 | 4 | 3 | 4 | 7 | **77** |

## Wiring

- `src/lib.rs` declares `pub mod prompt_breeder` under `feature =
  "prompt-breeder"` and re-exports the public API (with `Breeder*`
  aliases where names collide with existing types — `CostEstimator`,
  `ProviderFingerprint`, `TokenUsage`, `LlmClient`).
- `Cargo.toml` adds `[[bin]]` entries for `ai_breeder` and
  `ai_breeder_gui`, each with `required-features` matching their module.
- `Cargo.toml` declares the `prompt-breeder` feature, which enables
  `dep:blake3`.

## Threat model highlights

- **Prompt injection in bootstrapped seeds** — `SafetyFilter::Composite`
  runs before the unit enters the population; rejections are ledgered
  with `RejectReason::SafetyViolation{pattern_id}`.
- **Token-budget abuse from runaway mutations** — every operator clamps
  the rewritten prompt at `max_tokens` via `cap_tokens`; violation
  rejected with `RejectReason::TokenLimitExceeded{got, cap}`.
- **Fingerprint leakage across providers** — `EvalCache` key includes
  fingerprint; `FitnessScore` carries fingerprint; selection filters by
  fingerprint before scoring.
- **LLM failure denial-of-service** — `RetryingLlmClient` with configurable
  `RetryPolicy`; persistent failure ledgered as
  `MutationRejected{LlmCallFailed{retries_exhausted}}` and the generation
  continues.
- **Checkpoint tampering** — config hash embedded in the checkpoint;
  `matches_config` refuses resume on mismatch.
- **Ledger tampering** — Blake3 self-hash + prev-hash chain; optional
  Ed25519 `BreederSigner` trait for users who want signatures. `ai_breeder
  ledger-verify` replays the chain and fails on any break.
- **Freeze bypass** — `PromptBreeder::set_frozen(true)` records
  `FreezeChanged{frozen:true}` on the ledger and aborts the next
  `generation_tick` with `BreederError::Frozen` before any LLM call.

## Counts

| | V97 |
|---|---|
| Binaries added | 2 (`ai_breeder`, `ai_breeder_gui`) |
| Tests added | 77 |
| Feature flag | `prompt-breeder` (enables `dep:blake3`) |
| Opt-in axes | 19 |
| Mutation operators | 9 |
| Selection strategies | 5 |
| Replacement policies | 4 |
| Crossover strategies | 6 |
| Diversity metrics | 3 |
| Fitness smoothing modes | 4 |
| Safety filters | 4 |

## Not in scope

- Live cost tracking from provider APIs (callers use `CostEstimator` with
  override prices).
- Distributed breeder workers (the run loop is single-threaded;
  parallelism happens inside individual `FitnessEvaluator`
  implementations if wired).
- Web UI (GUI is native egui/eframe; website has static documentation
  pages only).
