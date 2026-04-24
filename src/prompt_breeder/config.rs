//! `PromptBreederConfig` — the 19 opt-in axes that drive every aspect of a
//! PromptBreeder run. Every field is a `pub` enum or numeric with a sane,
//! cheap, deterministic default. LLM-augmented variants are always opt-in.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

// =============================================================================
// Provider fingerprint
// =============================================================================

/// Segments fitness across `(provider, model)` pairs. Fitness recorded under
/// one fingerprint is never compared to another — a slow/accurate cloud model
/// and a fast/small local model live on separate leaderboards.
///
/// Shape-compatible with `prompt_synthesis::arm::ProviderFingerprint` so V97.1
/// can bridge between both sides without conversion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderFingerprint(String);

impl ProviderFingerprint {
    pub fn new(provider: &str, model: &str) -> Self {
        Self(format!("{provider}/{model}"))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// =============================================================================
// Provenance of a seed unit
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SeedProvenance {
    Manual,
    Random { seed: u64 },
    LlmBootstrapped { prompt_hash: String },
    Imported { source: String, original_id: String },
}

impl fmt::Display for SeedProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manual => f.write_str("manual"),
            Self::Random { seed } => write!(f, "random({seed})"),
            Self::LlmBootstrapped { .. } => f.write_str("llm_bootstrapped"),
            Self::Imported { source, .. } => write!(f, "imported({source})"),
        }
    }
}

// =============================================================================
// 19 configurable axes
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SeedSource {
    /// Caller provides explicit `(task_prompt, mutation_prompt)` pairs.
    Manual(Vec<(String, String)>),
    /// Sample from a pool of built-in prompt skeletons.
    Random { pool_size: usize },
    /// LLM generates N initial units from the task description. Requires
    /// `LlmClient` to be wired.
    LlmBootstrapped {
        n: usize,
        system_prompt: Option<String>,
    },
}

impl Default for SeedSource {
    fn default() -> Self {
        Self::Manual(Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EvalAugmenter {
    /// Deterministic bootstrap-resampling of the existing dataset.
    Bootstrap { factor: f32 },
    /// LLM synthesizes `n` new eval examples in the given style.
    LlmSynthesized { n: usize, style: String },
    /// Adversarial perturbation of existing examples.
    Adversarial { perturbation: Perturbation },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Perturbation {
    TypoInjection { rate: f32 },
    CaseFlip,
    PunctuationStrip,
    TokenShuffle { window: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DiversityMetric {
    /// Mean pairwise Levenshtein distance (character-level).
    EditDistance,
    /// Mean pairwise 1 - Jaccard on n-grams.
    NGramJaccard { n: usize },
    /// Cosine of embeddings produced by the LLM.
    EmbeddingCosine,
    /// Cluster embeddings into `target_clusters` and measure spread.
    LlmCluster { target_clusters: usize },
}

impl Default for DiversityMetric {
    fn default() -> Self {
        Self::EditDistance
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LineageNarrator {
    /// Deterministic template rendering the ancestor chain.
    TemplateSummary,
    /// LLM-generated narrative of ≤ `max_chars` characters.
    LlmSummary { max_chars: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FitnessSmoothing {
    /// Single evaluation per unit per generation.
    Single,
    /// Average over `k` independent evaluations.
    MeanOfK { k: usize },
    /// LLM sampled `k` times with `vote` rule to extract a single answer.
    SelfConsistency { k: usize, vote: VoteRule },
    /// Bayesian smoothing with Beta(α, β) prior.
    Bayesian { prior_alpha: f32, prior_beta: f32 },
}

impl Default for FitnessSmoothing {
    fn default() -> Self {
        Self::Single
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VoteRule {
    /// Strict majority — at least > k/2 agree.
    Majority,
    /// Plurality — most-common answer wins.
    Plurality,
    /// Best-of — pick highest-fitness candidate from the k samples.
    BestOf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OperatorScheduler {
    /// Every operator is equally likely.
    Uniform,
    /// UCB1 over operator ids with exploration constant `c` and a minimum
    /// of `min_pulls` pulls per operator before exploiting.
    Bandit { c: f32, min_pulls: usize },
    /// Favour operators whose recent children had high fitness.
    Adaptive { window: usize },
    /// Fixed schedule: each phase pins one operator subset for N generations.
    Curriculum { schedule: Vec<OperatorPhase> },
}

impl Default for OperatorScheduler {
    fn default() -> Self {
        Self::Uniform
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorPhase {
    pub generations: u32,
    pub operators: Vec<MutationOperator>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MutationOperator {
    /// "Write a new prompt for {task}." — no context.
    ZeroOrder,
    /// LLM rewrites a chosen parent prompt.
    FirstOrder,
    /// Estimation-of-Distribution: sample context from current population.
    Eda,
    /// EDA + rank annotation so LLM sees quality signal.
    EdaRankAndIndex,
    /// Sample ancestors from the lineage DAG and pass as context.
    LineageBased,
    /// Self-referential: apply a random mutation prompt to itself.
    HyperMutationZeroOrder,
    /// LLM rewrites a chosen mutation-prompt (meta level).
    HyperMutationFirstOrder,
    /// Reverse-engineer a prompt from a set of working (input, output) pairs.
    Lamarckian,
    /// Recombine two parents' task prompts.
    PromptCrossover,
}

impl MutationOperator {
    pub const ALL: [MutationOperator; 9] = [
        MutationOperator::ZeroOrder,
        MutationOperator::FirstOrder,
        MutationOperator::Eda,
        MutationOperator::EdaRankAndIndex,
        MutationOperator::LineageBased,
        MutationOperator::HyperMutationZeroOrder,
        MutationOperator::HyperMutationFirstOrder,
        MutationOperator::Lamarckian,
        MutationOperator::PromptCrossover,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ZeroOrder => "ZeroOrder",
            Self::FirstOrder => "FirstOrder",
            Self::Eda => "Eda",
            Self::EdaRankAndIndex => "EdaRankAndIndex",
            Self::LineageBased => "LineageBased",
            Self::HyperMutationZeroOrder => "HyperMutationZeroOrder",
            Self::HyperMutationFirstOrder => "HyperMutationFirstOrder",
            Self::Lamarckian => "Lamarckian",
            Self::PromptCrossover => "PromptCrossover",
        }
    }

    pub fn index(&self) -> usize {
        Self::ALL.iter().position(|o| o == self).unwrap_or(0)
    }
}

impl fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SelectionStrategy {
    /// K-way tournament.
    Tournament { k: usize },
    /// Probability proportional to fitness.
    RouletteWheel,
    /// Probability proportional to rank.
    RankBased,
    /// Keep the top `top_frac` fraction deterministically.
    Truncation { top_frac: f32 },
    /// Softmax with inverse temperature.
    Boltzmann { temperature: f32 },
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self::Tournament { k: 3 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CrossoverStrategy {
    None,
    SinglePoint,
    TwoPoint,
    Uniform {
        p: f32,
    },
    /// LLM-driven semantic recombination.
    SemanticLlm {
        prompt_template: String,
    },
    /// Crossover guided by lineage ancestry.
    LineageInformed,
}

impl Default for CrossoverStrategy {
    fn default() -> Self {
        Self::SinglePoint
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReplacementPolicy {
    Generational,
    SteadyState { replace_n: usize },
    Elitism { k: usize },
    TournamentReplace { k: usize },
}

impl Default for ReplacementPolicy {
    fn default() -> Self {
        Self::Elitism { k: 2 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FitnessObjective {
    /// Use the first metric as the sole aggregate score.
    Single,
    /// Weighted linear combination of metrics.
    WeightedSum { weights: Vec<(Metric, f32)> },
    /// NSGA-II style Pareto front over multiple objectives (crowding dist).
    Pareto { objectives: Vec<Metric> },
}

impl Default for FitnessObjective {
    fn default() -> Self {
        Self::Single
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Metric {
    Accuracy,
    ExactMatch,
    Contains,
    RegexMatch,
    JsonSchemaValid,
    BleuScore,
    LlmJudgeScore,
    LatencyMs,
    Tokens,
    CostUsd,
    PromptLength,
    Custom(String),
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accuracy => f.write_str("accuracy"),
            Self::ExactMatch => f.write_str("exact_match"),
            Self::Contains => f.write_str("contains"),
            Self::RegexMatch => f.write_str("regex_match"),
            Self::JsonSchemaValid => f.write_str("json_schema_valid"),
            Self::BleuScore => f.write_str("bleu"),
            Self::LlmJudgeScore => f.write_str("llm_judge"),
            Self::LatencyMs => f.write_str("latency_ms"),
            Self::Tokens => f.write_str("tokens"),
            Self::CostUsd => f.write_str("cost_usd"),
            Self::PromptLength => f.write_str("prompt_length"),
            Self::Custom(s) => f.write_str(s),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BudgetLimit {
    None,
    MaxLlmCalls(u64),
    MaxTokens(u64),
    MaxWallTime(Duration),
    MaxCostUsd(f64),
    /// OR-ed — first to trip aborts the run.
    Composite(Vec<BudgetLimit>),
}

impl Default for BudgetLimit {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EvalCacheMode {
    Disabled,
    Enabled,
    Persistent { path: PathBuf },
}

impl Default for EvalCacheMode {
    fn default() -> Self {
        Self::Enabled
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CheckpointPolicy {
    Disabled,
    Every { n_generations: u32, path: PathBuf },
    OnBudgetExhaustion { path: PathBuf },
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SafetyFilter {
    None,
    /// Block known prompt-injection patterns (chat template tokens,
    /// "ignore previous instructions", delimiter escapes).
    PromptInjectionBlock,
    /// Block PII patterns (emails, SSN-like, credit-card-like).
    PiiBlock,
    /// Apply a caller-specified constitutional policy (free-form text that
    /// will be used in a safety-check prompt when an LLM is wired).
    Constitutional {
        policy: String,
    },
    /// All of the above combined.
    Composite(Vec<SafetyFilter>),
}

impl Default for SafetyFilter {
    fn default() -> Self {
        Self::PromptInjectionBlock
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff: Backoff,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff: Backoff::Exponential {
                base_ms: 100,
                factor: 2.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Backoff {
    Fixed { ms: u64 },
    Exponential { base_ms: u64, factor: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputParser {
    /// Pass-through: LLM response is used verbatim.
    Raw,
    /// Strip Markdown code-fences before matching.
    StripMarkdown,
    /// Match first JSON object / array and use it.
    JsonFirst,
    /// Take the text after the last occurrence of `marker`.
    AfterMarker { marker: String },
    /// Apply a regex capture group.
    RegexCapture { pattern: String, group: usize },
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::Raw
    }
}

// =============================================================================
// Aggregate config
// =============================================================================

/// Complete PromptBreeder configuration. Every field is opt-in with a cheap,
/// deterministic default. LLM-augmented variants never activate unless
/// explicitly selected AND an `LlmClient` is wired.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptBreederConfig {
    // Core
    pub population_size: usize,
    pub generations: u32,
    pub provider_fingerprint: ProviderFingerprint,
    pub rng_seed: Option<u64>,
    pub operator_set: Vec<MutationOperator>,

    // Seeding
    pub seed_source: SeedSource,
    pub task_description: String,

    // Evaluation
    pub fitness_objective: FitnessObjective,
    pub fitness_smoothing: FitnessSmoothing,
    pub eval_augmenter: Option<EvalAugmenter>,
    pub output_parser: OutputParser,

    // Operation
    pub selection_strategy: SelectionStrategy,
    pub crossover_strategy: CrossoverStrategy,
    pub replacement_policy: ReplacementPolicy,
    pub operator_scheduler: OperatorScheduler,
    pub diversity_metric: DiversityMetric,

    // Narrative / observability
    pub lineage_narrator: Option<LineageNarrator>,

    // Controls
    pub budget: BudgetLimit,
    pub eval_cache: EvalCacheMode,
    pub checkpoint: CheckpointPolicy,
    pub safety_filter: SafetyFilter,
    pub retry_policy: RetryPolicy,
    pub max_prompt_tokens: usize,

    // Freeze
    pub frozen: bool,
}

impl PromptBreederConfig {
    /// Minimal config with sane defaults. Caller must supply the fingerprint
    /// because mixing fitness across providers would corrupt the learning
    /// signal — we make that decision explicit.
    pub fn new(provider: &str, model: &str) -> Self {
        Self {
            population_size: 20,
            generations: 50,
            provider_fingerprint: ProviderFingerprint::new(provider, model),
            rng_seed: None,
            operator_set: MutationOperator::ALL.to_vec(),

            seed_source: SeedSource::default(),
            task_description: String::new(),

            fitness_objective: FitnessObjective::default(),
            fitness_smoothing: FitnessSmoothing::default(),
            eval_augmenter: None,
            output_parser: OutputParser::default(),

            selection_strategy: SelectionStrategy::default(),
            crossover_strategy: CrossoverStrategy::default(),
            replacement_policy: ReplacementPolicy::default(),
            operator_scheduler: OperatorScheduler::default(),
            diversity_metric: DiversityMetric::default(),

            lineage_narrator: None,
            budget: BudgetLimit::default(),
            eval_cache: EvalCacheMode::default(),
            checkpoint: CheckpointPolicy::default(),
            safety_filter: SafetyFilter::default(),
            retry_policy: RetryPolicy::default(),
            max_prompt_tokens: 2048,

            frozen: false,
        }
    }

    /// Reject configurations that cannot produce a sensible run. Called at
    /// `PromptBreeder::new(config)`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.population_size < 4 {
            return Err(ConfigError::PopulationTooSmall(self.population_size));
        }
        if self.generations == 0 {
            return Err(ConfigError::ZeroGenerations);
        }
        if self.operator_set.is_empty() {
            return Err(ConfigError::EmptyOperatorSet);
        }
        if self.provider_fingerprint.as_str().is_empty()
            || self.provider_fingerprint.as_str() == "/"
        {
            return Err(ConfigError::MissingFingerprint);
        }
        if let FitnessObjective::Pareto { objectives } = &self.fitness_objective {
            if objectives.len() < 2 {
                return Err(ConfigError::ParetoNeedsMultipleObjectives(objectives.len()));
            }
        }
        if let FitnessObjective::WeightedSum { weights } = &self.fitness_objective {
            if weights.is_empty() {
                return Err(ConfigError::WeightedSumEmpty);
            }
        }
        if let SelectionStrategy::Tournament { k } = &self.selection_strategy {
            if *k == 0 || *k > self.population_size {
                return Err(ConfigError::InvalidTournamentK(*k));
            }
        }
        if let SelectionStrategy::Truncation { top_frac } = &self.selection_strategy {
            if !(0.0 < *top_frac && *top_frac <= 1.0) {
                return Err(ConfigError::InvalidTruncationFrac(*top_frac));
            }
        }
        if let ReplacementPolicy::Elitism { k } = &self.replacement_policy {
            if *k >= self.population_size {
                return Err(ConfigError::ElitismTooLarge(*k));
            }
        }
        if self.max_prompt_tokens < 32 {
            return Err(ConfigError::MaxPromptTokensTooSmall(self.max_prompt_tokens));
        }
        Ok(())
    }

    /// Stable canonical hash for ledger bookkeeping. Does NOT include the
    /// freeze field (that is a runtime toggle, not a run-shape property).
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut v = Vec::with_capacity(256);
        v.extend_from_slice(self.provider_fingerprint.as_str().as_bytes());
        v.extend_from_slice(self.task_description.as_bytes());
        v.extend_from_slice(&self.population_size.to_le_bytes());
        v.extend_from_slice(&self.generations.to_le_bytes());
        v.extend_from_slice(&self.max_prompt_tokens.to_le_bytes());
        for op in &self.operator_set {
            v.push(op.index() as u8);
        }
        if let Some(seed) = self.rng_seed {
            v.extend_from_slice(&seed.to_le_bytes());
        }
        #[cfg(feature = "prompt-breeder")]
        {
            blake3::hash(&v).into()
        }
        #[cfg(not(feature = "prompt-breeder"))]
        {
            let mut h = [0u8; 32];
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            v.hash(&mut hasher);
            let r = hasher.finish().to_le_bytes();
            h[..8].copy_from_slice(&r);
            h
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    PopulationTooSmall(usize),
    ZeroGenerations,
    EmptyOperatorSet,
    MissingFingerprint,
    ParetoNeedsMultipleObjectives(usize),
    WeightedSumEmpty,
    InvalidTournamentK(usize),
    InvalidTruncationFrac(f32),
    ElitismTooLarge(usize),
    MaxPromptTokensTooSmall(usize),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PopulationTooSmall(n) => {
                write!(f, "population_size={n} too small; minimum is 4")
            }
            Self::ZeroGenerations => f.write_str("generations must be >= 1"),
            Self::EmptyOperatorSet => f.write_str("operator_set must not be empty"),
            Self::MissingFingerprint => f.write_str("provider_fingerprint is empty"),
            Self::ParetoNeedsMultipleObjectives(n) => {
                write!(f, "Pareto objective needs >=2 metrics, got {n}")
            }
            Self::WeightedSumEmpty => f.write_str("WeightedSum weights must not be empty"),
            Self::InvalidTournamentK(k) => {
                write!(f, "invalid tournament size k={k}")
            }
            Self::InvalidTruncationFrac(v) => {
                write!(f, "Truncation top_frac={v} must be in (0, 1]")
            }
            Self::ElitismTooLarge(k) => write!(f, "Elitism k={k} must be < population_size"),
            Self::MaxPromptTokensTooSmall(n) => {
                write!(f, "max_prompt_tokens={n} too small; minimum 32")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PromptBreederConfig {
        PromptBreederConfig::new("ollama", "mistral:7b")
    }

    #[test]
    fn default_validates() {
        let mut c = cfg();
        c.task_description = "sort numbers".into();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_small_population() {
        let mut c = cfg();
        c.population_size = 2;
        assert!(matches!(
            c.validate(),
            Err(ConfigError::PopulationTooSmall(2))
        ));
    }

    #[test]
    fn rejects_zero_generations() {
        let mut c = cfg();
        c.generations = 0;
        assert!(matches!(c.validate(), Err(ConfigError::ZeroGenerations)));
    }

    #[test]
    fn rejects_empty_operator_set() {
        let mut c = cfg();
        c.operator_set = vec![];
        assert!(matches!(c.validate(), Err(ConfigError::EmptyOperatorSet)));
    }

    #[test]
    fn rejects_pareto_with_one_objective() {
        let mut c = cfg();
        c.fitness_objective = FitnessObjective::Pareto {
            objectives: vec![Metric::Accuracy],
        };
        assert!(matches!(
            c.validate(),
            Err(ConfigError::ParetoNeedsMultipleObjectives(1))
        ));
    }

    #[test]
    fn rejects_tournament_k_zero() {
        let mut c = cfg();
        c.selection_strategy = SelectionStrategy::Tournament { k: 0 };
        assert!(matches!(
            c.validate(),
            Err(ConfigError::InvalidTournamentK(0))
        ));
    }

    #[test]
    fn canonical_hash_stable_across_runs() {
        let c1 = cfg();
        let c2 = cfg();
        assert_eq!(c1.canonical_hash(), c2.canonical_hash());
    }

    #[test]
    fn canonical_hash_changes_on_population_change() {
        let mut c1 = cfg();
        let mut c2 = cfg();
        c1.population_size = 20;
        c2.population_size = 30;
        assert_ne!(c1.canonical_hash(), c2.canonical_hash());
    }

    #[test]
    fn all_operators_listed() {
        assert_eq!(MutationOperator::ALL.len(), 9);
        let names: Vec<&str> = MutationOperator::ALL.iter().map(|o| o.as_str()).collect();
        assert!(names.contains(&"ZeroOrder"));
        assert!(names.contains(&"HyperMutationZeroOrder"));
        assert!(names.contains(&"PromptCrossover"));
    }
}
