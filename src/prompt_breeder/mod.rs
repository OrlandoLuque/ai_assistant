//! V97 — PromptBreeder: self-referential evolution of prompt/mutation-prompt
//! pairs (Fernando et al. 2023). Gated behind the `prompt-breeder` feature.
//!
//! Module layout mirrors the V97 plan:
//!
//! - `config`         — 19 opt-in axes (fully serde-compatible)
//! - `rng`            — deterministic xorshift\* PRNG
//! - `ledger`         — Blake3 hash-chained event log (optional signer)
//! - `safety`         — prompt-injection + PII + constitutional filters
//! - `eval`           — dataset + augmenters + output parser
//! - `fitness`        — 5 built-in evaluators + NSGA-II helpers
//! - `cache`          — `(prompt, input, fp, sample_idx)` → score memo
//! - `budget`         — call/token/time/cost meter
//! - `llm`            — `LlmClient` trait + `MockLlmClient` + retry wrapper
//! - `operators`      — 9 mutation operators with deterministic fallbacks
//! - `population`     — `Unit`, `Population`, `LineageDag`
//! - `checkpoint`     — atomic serialisable snapshots
//! - `breeder`        — the `PromptBreeder` run loop

pub mod budget;
pub mod cache;
pub mod checkpoint;
pub mod config;
pub mod eval;
pub mod fitness;
pub mod ledger;
pub mod llm;
pub mod operators;
pub mod population;
pub mod rng;
pub mod safety;

pub mod breeder;

// =============================================================================
// Re-exports — everything `lib.rs` expects to surface under `ai_assistant::*`
// =============================================================================

pub use breeder::{BreederError, BreederOutcome, PromptBreeder};
pub use budget::{BudgetBreach, BudgetMeter};
pub use cache::{CacheHit, CacheKey, EvalCache};
pub use checkpoint::{Checkpoint, CheckpointError};
pub use config::{
    Backoff, BudgetLimit, CheckpointPolicy, ConfigError, CrossoverStrategy, DiversityMetric,
    EvalAugmenter, EvalCacheMode, FitnessObjective, FitnessSmoothing, LineageNarrator, Metric,
    MutationOperator, OperatorPhase, OperatorScheduler, OutputParser, Perturbation,
    PromptBreederConfig, ProviderFingerprint, ReplacementPolicy, RetryPolicy, SafetyFilter,
    SeedProvenance, SeedSource, SelectionStrategy, VoteRule,
};
pub use eval::{augment_deterministic, parse_output, EvalDataset, EvalExample};
pub use fitness::{
    crowding_distance, pareto_ranks, CompositeEvaluator, ContainsEvaluator, ExactMatchEvaluator,
    FitnessEvaluator, FitnessScore, JsonSchemaEvaluator, LlmJudgeEvaluator, RegexEvaluator,
};
pub use ledger::{
    AbortReason, BreederEvent, BreederLedger, BreederLedgerError, BreederSigner, BudgetKind,
    LedgerEntry, NoopBreederSigner, RejectReason,
};
pub use llm::{
    CostEstimator, FailMode, LlmClient, LlmError, LlmResponse, MockLlmClient, RetryingLlmClient,
    TokenUsage,
};
pub use operators::{apply_operator, MutationContext};
pub use population::{LineageDag, Population, Unit};
pub use rng::BreederRng;
pub use safety::{check as safety_check, SafetyOutcome};
