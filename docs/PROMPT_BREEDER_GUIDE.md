# PromptBreeder Guide (V97)

Self-referential evolution of `(task_prompt, mutation_prompt)` pairs,
gated behind the `prompt-breeder` feature flag. Paper:
*Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*,
Fernando et al. (DeepMind, 2023).

## Feature flag

```toml
[dependencies]
ai_assistant = { path = "../ai_assistant", features = ["prompt-breeder"] }
```

`prompt-breeder` pulls in `blake3` for the ledger chain. Nothing else is
added to the dependency graph.

## Minimum viable run

```rust
use ai_assistant::{
    BreederLedger, BreederLlmClient, CompositeEvaluator, EvalDataset,
    EvalExample, FitnessObjective, MockLlmClient, PromptBreeder,
    PromptBreederConfig, SeedSource,
};
use std::sync::Arc;

// 1. Config — 19 axes; at minimum the fingerprint + task description.
let mut cfg = PromptBreederConfig::new("ollama", "mistral:7b");
cfg.task_description = "Sort a list of integers ascending".into();
cfg.population_size = 8;
cfg.num_generations = 5;
cfg.seed_source = SeedSource::Manual(vec![
    ("Sort this list ascending: {input}".into(), "Rewrite to be clearer:".into()),
    ("Return the sorted version of: {input}".into(), "Tighten the wording:".into()),
]);
cfg.fitness_objective = FitnessObjective::Maximize;

// 2. Tiny eval dataset.
let dataset = EvalDataset::from_examples(vec![
    EvalExample::new("[3,1,2]", "[1,2,3]"),
    EvalExample::new("[9,0,4]", "[0,4,9]"),
]);

// 3. LLM — real provider or mock.
let llm: Arc<dyn BreederLlmClient> =
    Arc::new(MockLlmClient::returning("[1,2,3]"));

// 4. Fitness evaluator — ExactMatch against expected_output.
let evaluator = Arc::new(
    CompositeEvaluator::new()
        .with_exact_match(1.0),
);

// 5. Run.
let ledger = BreederLedger::in_memory();
let mut breeder = PromptBreeder::new(cfg, llm, evaluator, dataset, ledger);
let outcome = breeder.run().unwrap();

println!("best: {} @ fitness {:.4}",
    outcome.best_unit.as_ref().map(|u| u.id.as_str()).unwrap_or("-"),
    outcome.best_unit.as_ref().map(|u| u.fitness_value()).unwrap_or(0.0));
```

## 19 configurable axes

1. **Seed source** — `Manual(Vec<(task,mut)>)` / `Random{pool_size}` /
   `LlmBootstrapped{n_units}`.
2. **Population size** — typically 8–50 for local models, 16–200 for
   cloud.
3. **Number of generations** — early-stop when diversity drops below the
   configured minimum.
4. **Selection strategy** — `Tournament{k}` / `RouletteWheel` /
   `RankBased{s}` / `Truncation{top_pct}` / `Boltzmann{temperature}`.
5. **Replacement policy** — `Generational` / `SteadyState{n_survivors}`
   / `Elitism{k}` / `TournamentReplace{k}`.
6. **Mutation operator set** — any subset of 9; the scheduler picks
   among them.
7. **Operator scheduler** — `Uniform` / `Ucb1{c}` / `Adaptive{window}` /
   `Curriculum{phases}`.
8. **Crossover strategy** — `None` / `SinglePoint` / `TwoPoint` /
   `Uniform{rate}` / `SemanticLlm` / `LineageInformed`.
9. **Fitness smoothing** — `Single` / `MeanOfK{k}` /
   `SelfConsistency{k, VoteRule::Majority|Plurality|BestOfN}` /
   `Bayesian{prior_alpha, prior_beta}`.
10. **Diversity metric** — `EditDistance` / `NGramJaccard{n}` /
    `EmbeddingCosine{model_id}`.
11. **Max tokens per unit** — per-mutation cap (4 chars / token).
12. **Safety filter** — `None` / `PromptInjectionBlock` / `PiiBlock` /
    `Constitutional{principles}` / `Composite(vec)`.
13. **Retry policy** — `max_retries` + `Backoff::Fixed{ms} |
    Exponential{base_ms, factor}`.
14. **Budget limit** — `MaxCalls(u64)` / `MaxTokens(u64)` /
    `MaxWallTime(Duration)` / `MaxCostUsd(f64)` (via `CostEstimator`).
15. **Cache mode** — `Disabled` / `Memory{capacity}` /
    `MemoryAndDisk{path, capacity}`.
16. **Eval augmenter** — optional `EvalAugmenter::Synonym|Paraphrase|
    Noise|LlmRephrase` to expand the dataset deterministically.
17. **Output parser** — `Raw` / `StripMarkdown` / `FirstJsonBlock` /
    `Regex{pattern, group}`.
18. **Checkpoint policy** — `Disabled` /
    `Every{n_generations, path}` / `OnBudgetExhaustion{path}`.
19. **Lineage narrator** — `Disabled` / `Summary` / `LlmGenerated`.

## Ledger events

Emitted in order by `PromptBreeder::run`:

- `RunStarted{run_id, config_hash_hex, fingerprint}`
- `SeedBootstrapped{n, source}` / `SeedInserted{unit_id, source}`
- `GenerationStarted{generation}`
- `MutationApplied{parent_id, child_id, operator}` /
  `MutationRejected{parent_id, operator, reason}`
- `FitnessEvaluated{unit_id, score, cached}`
- `SelectionPerformed{strategy, survivors}`
- `DiversityComputed{generation, score}`
- `EvalAugmented{n_added, augmenter_kind}` (if augmenter enabled)
- `LineageNarrated{unit_id, narrative_hash_hex}` (if narrator enabled)
- `SmoothingSampled{unit_id, samples}` (for `MeanOfK` / `SelfConsistency`)
- `BudgetExhausted{kind, value}` — ends the run cleanly
- `CheckpointWritten{path, tip_hash_hex}`
- `FreezeChanged{frozen}`
- `SafetyFilterApplied{filter_kind}`
- `RunCompleted{run_id, best_id, generations}` /
  `RunAborted{run_id, reason}`

Every entry carries `seq` + `prev_hash_hex` + `self_hash_hex` +
`signature_hex` (empty under `NoopBreederSigner`). Verify with
`ledger.verify()` or `ai_breeder ledger-verify`.

## Checkpoints

Serialisable `(run_id, generation, config_hash_hex, ledger_tip_hash_hex,
population, lineage)`. Written atomically (`.tmp` + rename). Magic bytes
`AIBR-CKPT\x01`. Load with `ai_assistant::prompt_breeder::checkpoint::read`
or inspect with `ai_breeder show-run`.

Resume behaviour: if `matches_config(&config)` returns false on load the
breeder refuses to resume — a config edit between runs means the
fitness landscape changed shape and resuming would mix scores from
different regimes.

## CLI quick-reference (`ai_breeder`)

```bash
ai_breeder list-runs <DIR>                        # scan *.ckpt
ai_breeder show-run <CKPT_FILE>                   # summary
ai_breeder ledger-verify <LEDGER_JSONL>           # chain integrity
ai_breeder ledger-show <LEDGER_JSONL> [--last N]  # events
ai_breeder export-population <CKPT> <OUT_JSON>    # dump units
ai_breeder compare-runs <CKPT_A> <CKPT_B>         # side-by-side
```

Build: `cargo build --release --bin ai_breeder \
  --features prompt-breeder`.

## GUI quick-reference (`ai_breeder_gui`)

Desktop companion (egui/eframe). Tabs: Overview / Population / Lineage /
Ledger / Events / Fitness. Enter a checkpoint path + optional ledger
JSONL path, press Reload. Auto-refresh every 5s available.

Build: `cargo build --release --bin ai_breeder_gui \
  --features "prompt-breeder gui-pro"`.

## When to use it

Use PromptBreeder when:

- You have a **fixed task + gold labels** (dataset with
  `expected_output`).
- You can afford **N × M LLM calls** where N = population_size,
  M = num_generations × examples_per_eval.
- You want **reproducible** prompt optimisation — the xorshift\* PRNG
  makes the same `(config, seed)` produce the same run.
- You care about **audit trail** — every mutation, rejection, and
  evaluation lands on the Blake3 chain.

Prefer `prompt_synthesis` (V96) when the dataset is too noisy for batch
optimisation and you want online contextual-bandit selection over a
small set of human-authored fragments instead.

Prefer a hand-written prompt when the task is small enough that one
engineer-day beats N GPU-hours.
