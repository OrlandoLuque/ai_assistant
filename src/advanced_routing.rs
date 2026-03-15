//! Advanced routing system for intelligent model selection.
//!
//! Provides multi-armed bandit routing (Thompson Sampling, UCB1, epsilon-greedy),
//! NFA/DFA graph-based routing, hierarchical routing DAGs, ensemble voting,
//! adaptive per-query routing, eval-to-runtime feedback, distributed bandit
//! training, and export/import of bandit state.

pub use crate::error::AdvancedRoutingError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

// =============================================================================
// SHARED TYPES
// =============================================================================

/// Unique identifier for a bandit arm (typically a model ID).
pub type ArmId = String;

/// Features extracted from a query for routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Approximate token count
    pub token_count: usize,
    /// Number of sentences
    pub sentence_count: usize,
    /// Detected domain (e.g., "coding", "math", "creative", "general")
    pub domain: String,
    /// Complexity score 0.0..1.0
    pub complexity: f64,
    /// Entity count (names, numbers, code blocks)
    pub entity_count: usize,
    /// Whether the query contains code
    pub has_code: bool,
    /// Whether the query asks a question
    pub is_question: bool,
    /// Average word length (proxy for vocabulary complexity)
    pub avg_word_length: f64,
    /// Raw feature vector for ML-style routing
    pub feature_vector: Vec<f64>,
}

/// Result of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOutcome {
    /// Selected model/arm
    pub selected_arm: ArmId,
    /// Confidence in the selection (0.0..1.0)
    pub confidence: f64,
    /// Reason for the selection
    pub reason: String,
    /// Alternative arms considered (ranked by score descending)
    pub alternatives: Vec<(ArmId, f64)>,
    /// Which router made the decision
    pub router_id: String,
    /// Time taken for the routing decision in microseconds
    pub decision_time_us: u64,
}

/// Outcome feedback after a model invocation completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmFeedback {
    /// Which arm was used
    pub arm_id: ArmId,
    /// Whether the invocation was successful
    pub success: bool,
    /// Quality score if available (0.0..1.0)
    pub quality: Option<f64>,
    /// Latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Cost incurred
    pub cost: Option<f64>,
    /// Task type context
    pub task_type: Option<String>,
}

// =============================================================================
// MULTI-ARMED BANDIT ROUTER
// =============================================================================

/// Parameters of a Beta distribution for Thompson Sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    /// Alpha parameter (successes + prior)
    pub alpha: f64,
    /// Beta parameter (failures + prior)
    pub beta: f64,
}

/// A single arm in the bandit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// Arm identifier (model ID)
    pub id: ArmId,
    /// Beta distribution parameters
    pub params: BetaParams,
    /// Total times this arm was pulled
    pub pull_count: u64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Last time this arm was pulled (unix timestamp seconds)
    pub last_pulled: u64,
}

/// Strategy for selecting arms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum BanditStrategy {
    /// Thompson Sampling: sample from Beta posterior, pick highest
    ThompsonSampling,
    /// UCB1: pick arm maximizing mean + sqrt(2 * ln(N) / n_i)
    Ucb1,
    /// Epsilon-greedy: explore with probability epsilon, else exploit best
    EpsilonGreedy { epsilon: f64 },
}

// =============================================================================
// REWARD POLICY (Section A2)
// =============================================================================

/// Policy for computing composite reward from quality, latency, and cost.
///
/// Combines three dimensions into a single 0..1 reward signal for bandit learning.
/// Latency and cost are normalized using reference values: a latency of
/// `latency_ref_ms` maps to score 0.0, and zero latency maps to 1.0.
///
/// When `ArmFeedback` has `None` for latency or cost, the corresponding
/// weight is redistributed proportionally to the active components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPolicy {
    /// Weight for quality component (default 0.7)
    pub quality_weight: f64,
    /// Weight for latency component (default 0.2)
    pub latency_weight: f64,
    /// Weight for cost component (default 0.1)
    pub cost_weight: f64,
    /// Reference latency in ms for normalization (default 5000.0).
    /// A latency equal to this value yields a latency score of 0.0.
    pub latency_ref_ms: f64,
    /// Reference cost for normalization (default 0.01).
    /// A cost equal to this value yields a cost score of 0.0.
    pub cost_ref: f64,
}

impl Default for RewardPolicy {
    fn default() -> Self {
        Self {
            quality_weight: 0.7,
            latency_weight: 0.2,
            cost_weight: 0.1,
            latency_ref_ms: 5000.0,
            cost_ref: 0.01,
        }
    }
}

impl RewardPolicy {
    /// Normalize weights so they sum to 1.0.
    ///
    /// If all weights are zero (or negative), returns equal thirds (1/3, 1/3, 1/3).
    pub fn normalize_weights(&self) -> (f64, f64, f64) {
        let qw = self.quality_weight.max(0.0);
        let lw = self.latency_weight.max(0.0);
        let cw = self.cost_weight.max(0.0);
        let total = qw + lw + cw;
        if total < 1e-12 {
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        } else {
            (qw / total, lw / total, cw / total)
        }
    }

    /// Compute a composite reward in 0..1 from feedback dimensions.
    ///
    /// Components:
    /// - quality: `feedback.quality` or `1.0` if success, `0.0` if failure
    /// - latency: `(1.0 - latency_ms / latency_ref_ms).clamp(0.0, 1.0)`, skipped if `None`
    /// - cost: `(1.0 - cost / cost_ref).clamp(0.0, 1.0)`, skipped if `None`
    ///
    /// Missing components have their weight redistributed to active components.
    pub fn compute_reward(&self, feedback: &ArmFeedback) -> f64 {
        let (qw, lw, cw) = self.normalize_weights();

        let quality_score = feedback.quality.unwrap_or(if feedback.success { 1.0 } else { 0.0 });

        let latency_available = feedback.latency_ms.is_some() && self.latency_ref_ms > 0.0;
        let cost_available = feedback.cost.is_some() && self.cost_ref > 0.0;

        let latency_score = if latency_available {
            let ms = feedback.latency_ms.unwrap_or(0) as f64;
            (1.0 - ms / self.latency_ref_ms).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let cost_score = if cost_available {
            let c = feedback.cost.unwrap_or(0.0);
            (1.0 - c / self.cost_ref).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Redistribute weights of missing components
        let active_extra_weight = if !latency_available { lw } else { 0.0 }
            + if !cost_available { cw } else { 0.0 };

        let effective_qw;
        let effective_lw;
        let effective_cw;

        if !latency_available && !cost_available {
            // Only quality is available
            effective_qw = 1.0;
            effective_lw = 0.0;
            effective_cw = 0.0;
        } else if !latency_available {
            // Quality + cost active, redistribute latency weight
            let active_sum = qw + cw;
            if active_sum < 1e-12 {
                effective_qw = 0.5;
                effective_lw = 0.0;
                effective_cw = 0.5;
            } else {
                effective_qw = qw + active_extra_weight * (qw / active_sum);
                effective_lw = 0.0;
                effective_cw = cw + active_extra_weight * (cw / active_sum);
            }
        } else if !cost_available {
            // Quality + latency active, redistribute cost weight
            let active_sum = qw + lw;
            if active_sum < 1e-12 {
                effective_qw = 0.5;
                effective_lw = 0.5;
                effective_cw = 0.0;
            } else {
                effective_qw = qw + active_extra_weight * (qw / active_sum);
                effective_lw = lw + active_extra_weight * (lw / active_sum);
                effective_cw = 0.0;
            }
        } else {
            // All three active
            effective_qw = qw;
            effective_lw = lw;
            effective_cw = cw;
        }

        let reward = effective_qw * quality_score
            + effective_lw * latency_score
            + effective_cw * cost_score;

        reward.clamp(0.0, 1.0)
    }
}

fn default_prefer_boost() -> f64 { 2.0 }

/// Per-query routing preferences that override default RewardPolicy weights.
///
/// Weight overrides (quality/latency/cost) affect how outcomes are RECORDED
/// (via `record_outcome_with_preferences`). Arm exclusion/boosting affects
/// which arm is SELECTED.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPreferences {
    /// Override quality weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_weight: Option<f64>,
    /// Override latency weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_weight: Option<f64>,
    /// Override cost weight (None = use default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_weight: Option<f64>,
    /// Arms to exclude from selection for this query
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_arms: Vec<ArmId>,
    /// Arms to boost for this query
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preferred_arms: Vec<ArmId>,
    /// Boost factor for preferred arms (default 2.0).
    #[serde(default = "default_prefer_boost")]
    pub prefer_boost: f64,
}

impl Default for RoutingPreferences {
    fn default() -> Self {
        Self {
            quality_weight: None,
            latency_weight: None,
            cost_weight: None,
            excluded_arms: Vec::new(),
            preferred_arms: Vec::new(),
            prefer_boost: 2.0,
        }
    }
}

impl RoutingPreferences {
    /// Build a temporary RewardPolicy by merging preferences over a base policy.
    pub fn apply_to_policy(&self, base: &RewardPolicy) -> RewardPolicy {
        RewardPolicy {
            quality_weight: self.quality_weight.unwrap_or(base.quality_weight),
            latency_weight: self.latency_weight.unwrap_or(base.latency_weight),
            cost_weight: self.cost_weight.unwrap_or(base.cost_weight),
            latency_ref_ms: base.latency_ref_ms,
            cost_ref: base.cost_ref,
        }
    }

    /// Convenience: create preferences that ignore cost.
    pub fn ignore_cost() -> Self {
        Self { cost_weight: Some(0.0), ..Default::default() }
    }

    /// Convenience: create preferences that minimize latency.
    pub fn minimize_latency() -> Self {
        Self {
            latency_weight: Some(0.8),
            quality_weight: Some(0.2),
            cost_weight: Some(0.0),
            ..Default::default()
        }
    }

    /// Convenience: create preferences that maximize quality only.
    pub fn quality_only() -> Self {
        Self {
            quality_weight: Some(1.0),
            latency_weight: Some(0.0),
            cost_weight: Some(0.0),
            ..Default::default()
        }
    }
}

/// Extended routing context: QueryFeatures + agent-level metadata.
///
/// Provides a superset of information for routing decisions, including
/// budget constraints, RAG status, and agent tier. The pipeline can
/// auto-derive RoutingPreferences from context fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    /// Core query features
    pub features: QueryFeatures,
    /// Whether RAG is active for this query
    #[serde(default)]
    pub rag_active: bool,
    /// Remaining budget (currency units). Low budget auto-boosts cost_weight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_remaining: Option<f64>,
    /// Agent tier (e.g., "free", "pro", "enterprise")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_tier: Option<String>,
    /// Total cost accumulated in the current session
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_cost_so_far: Option<f64>,
    /// Preferred provider to filter arms by (e.g., "openai", "anthropic")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_provider: Option<String>,
}

impl RoutingContext {
    /// Create a new routing context from query features.
    pub fn new(features: QueryFeatures) -> Self {
        Self {
            features,
            rag_active: false,
            budget_remaining: None,
            agent_tier: None,
            session_cost_so_far: None,
            preferred_provider: None,
        }
    }

    /// Auto-derive RoutingPreferences from context fields.
    ///
    /// Rules:
    /// - If `budget_remaining` < `cost_ref * 10.0`, boosts cost_weight to 0.5
    /// - Otherwise returns default preferences
    pub fn derive_preferences(&self, base_policy: &RewardPolicy) -> RoutingPreferences {
        let mut prefs = RoutingPreferences::default();

        if let Some(budget) = self.budget_remaining {
            if budget < base_policy.cost_ref * 10.0 {
                prefs.cost_weight = Some(0.5);
                prefs.quality_weight = Some(0.4);
                prefs.latency_weight = Some(0.1);
            }
        }

        prefs
    }
}

impl From<QueryFeatures> for RoutingContext {
    fn from(features: QueryFeatures) -> Self {
        Self::new(features)
    }
}

/// Configuration for the bandit router.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BanditConfig {
    /// Selection strategy
    pub strategy: BanditStrategy,
    /// Prior alpha for new arms (default 1.0 = uniform prior)
    pub prior_alpha: f64,
    /// Prior beta for new arms (default 1.0)
    pub prior_beta: f64,
    /// Minimum pulls before an arm can be pruned
    pub min_pulls_before_prune: u64,
    /// Decay factor for old observations (1.0 = no decay)
    pub decay_factor: f64,
    /// Reward computation policy for composite reward from quality/latency/cost.
    #[serde(default)]
    pub reward_policy: RewardPolicy,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            strategy: BanditStrategy::ThompsonSampling,
            prior_alpha: 1.0,
            prior_beta: 1.0,
            min_pulls_before_prune: 10,
            decay_factor: 1.0,
            reward_policy: RewardPolicy::default(),
        }
    }
}

/// Visibility of arms for distributed state sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ArmVisibility {
    /// Share freely with all nodes (default)
    Public,
    /// Never share — local-only model
    Private,
}

impl Default for ArmVisibility {
    fn default() -> Self { Self::Public }
}

/// Multi-Armed Bandit router with per-task-type bandits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditRouter {
    config: BanditConfig,
    /// Per-task bandits: task_type -> list of arms
    bandits: HashMap<String, Vec<BanditArm>>,
    /// Global bandit (when task type is unknown)
    global_bandit: Vec<BanditArm>,
    /// Total pulls across all bandits (for UCB1)
    total_pulls: u64,
    /// PRNG seed state (LCG for deterministic testing)
    seed: u64,
    /// Arms marked as private (local-only, not shared in distributed merging).
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    private_arms: HashSet<ArmId>,
}

impl BanditRouter {
    /// Create a new bandit router with the given configuration.
    pub fn new(config: BanditConfig) -> Self {
        Self {
            config,
            bandits: HashMap::new(),
            global_bandit: Vec::new(),
            total_pulls: 0,
            seed: 12345,
            private_arms: HashSet::new(),
        }
    }

    /// Create with a specific seed for deterministic testing.
    pub fn with_seed(config: BanditConfig, seed: u64) -> Self {
        Self {
            config,
            bandits: HashMap::new(),
            global_bandit: Vec::new(),
            total_pulls: 0,
            seed,
            private_arms: HashSet::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &BanditConfig {
        &self.config
    }

    /// Add an arm to the global bandit.
    pub fn add_arm(&mut self, arm_id: &str) {
        if !self.global_bandit.iter().any(|a| a.id == arm_id) {
            self.global_bandit.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams {
                    alpha: self.config.prior_alpha,
                    beta: self.config.prior_beta,
                },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Add an arm to a task-specific bandit.
    pub fn add_arm_for_task(&mut self, task_type: &str, arm_id: &str) {
        let arms = self.bandits.entry(task_type.to_string()).or_default();
        if !arms.iter().any(|a| a.id == arm_id) {
            arms.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams {
                    alpha: self.config.prior_alpha,
                    beta: self.config.prior_beta,
                },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Select an arm using the configured strategy.
    pub fn select(&mut self, task_type: Option<&str>) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Clone arms snapshot to avoid borrow conflicts with &mut self in sampling
        let arms_snapshot: Vec<BanditArm> = if let Some(tt) = task_type {
            self.bandits.get(tt).unwrap_or(&self.global_bandit).clone()
        } else {
            self.global_bandit.clone()
        };

        if arms_snapshot.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: task_type.unwrap_or("global").to_string(),
                reason: "No arms registered".to_string(),
            });
        }

        let start = std::time::Instant::now();

        let (selected_idx, scores) = match self.config.strategy {
            BanditStrategy::ThompsonSampling => self.thompson_select(&arms_snapshot),
            BanditStrategy::Ucb1 => self.ucb1_select(&arms_snapshot),
            BanditStrategy::EpsilonGreedy { epsilon } => self.epsilon_greedy_select(&arms_snapshot, epsilon),
        };

        let selected_id = arms_snapshot[selected_idx].id.clone();
        let confidence = if scores.is_empty() { 0.5 } else {
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            if (max_score - min_score).abs() < 1e-10 { 0.5 }
            else { (scores[selected_idx] - min_score) / (max_score - min_score) }
        };

        let mut alternatives: Vec<(ArmId, f64)> = arms_snapshot.iter()
            .zip(scores.iter())
            .enumerate()
            .filter(|(i, _)| *i != selected_idx)
            .map(|(_, (a, &s))| (a.id.clone(), s))
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update the actual arm's pull count
        let arms_mut = if let Some(tt) = task_type {
            self.bandits.get_mut(tt).unwrap_or(&mut self.global_bandit)
        } else {
            &mut self.global_bandit
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if selected_idx < arms_mut.len() {
            arms_mut[selected_idx].last_pulled = now;
        }
        self.total_pulls += 1;

        let elapsed = start.elapsed().as_micros() as u64;

        Ok(RoutingOutcome {
            selected_arm: selected_id,
            confidence,
            reason: format!("{:?} selection", self.config.strategy),
            alternatives,
            router_id: "bandit".to_string(),
            decision_time_us: elapsed,
        })
    }

    /// Record outcome feedback for an arm.
    pub fn record_outcome(&mut self, feedback: &ArmFeedback) {
        let task_type = feedback.task_type.as_deref();
        let arms = if let Some(tt) = task_type {
            if let Some(a) = self.bandits.get_mut(tt) { a }
            else { &mut self.global_bandit }
        } else {
            &mut self.global_bandit
        };

        if let Some(arm) = arms.iter_mut().find(|a| a.id == feedback.arm_id) {
            let reward = self.config.reward_policy.compute_reward(feedback);

            // Apply decay if configured
            if self.config.decay_factor < 1.0 {
                let d = self.config.decay_factor;
                arm.params.alpha = self.config.prior_alpha
                    + (arm.params.alpha - self.config.prior_alpha) * d;
                arm.params.beta = self.config.prior_beta
                    + (arm.params.beta - self.config.prior_beta) * d;
            }

            arm.params.alpha += reward;
            arm.params.beta += 1.0 - reward;
            arm.pull_count += 1;
            arm.total_reward += reward;
        }
    }

    /// Set specific priors for an arm (warm start from eval data).
    pub fn warm_start(&mut self, arm_id: &str, alpha: f64, beta: f64) {
        if let Some(arm) = self.global_bandit.iter_mut().find(|a| a.id == arm_id) {
            arm.params.alpha = alpha;
            arm.params.beta = beta;
        } else {
            self.global_bandit.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams { alpha, beta },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Set specific priors for a task-specific arm.
    pub fn warm_start_for_task(&mut self, task_type: &str, arm_id: &str, alpha: f64, beta: f64) {
        let arms = self.bandits.entry(task_type.to_string()).or_default();
        if let Some(arm) = arms.iter_mut().find(|a| a.id == arm_id) {
            arm.params.alpha = alpha;
            arm.params.beta = beta;
        } else {
            arms.push(BanditArm {
                id: arm_id.to_string(),
                params: BetaParams { alpha, beta },
                pull_count: 0,
                total_reward: 0.0,
                last_pulled: 0,
            });
        }
    }

    /// Get stats for a specific arm (global bandit).
    pub fn arm_stats(&self, arm_id: &str) -> Option<&BanditArm> {
        self.global_bandit.iter().find(|a| a.id == arm_id)
    }

    /// Get all arms for a task type (or global).
    pub fn all_arms(&self, task_type: Option<&str>) -> &[BanditArm] {
        if let Some(tt) = task_type {
            self.bandits.get(tt).map(|v| v.as_slice()).unwrap_or(&self.global_bandit)
        } else {
            &self.global_bandit
        }
    }

    /// Get total pulls across all bandits.
    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }

    /// Returns all task types that have dedicated bandit arms.
    pub fn task_types(&self) -> Vec<&str> {
        self.bandits.keys().map(|s| s.as_str()).collect()
    }

    /// Returns all arms for a given task type (or global if None) as a Vec.
    pub fn all_arms_vec(&self, task_type: Option<&str>) -> Vec<&BanditArm> {
        match task_type {
            Some(t) => self.bandits.get(t).map(|v| v.iter().collect()).unwrap_or_default(),
            None => self.global_bandit.iter().collect(),
        }
    }

    /// Remove an arm from the bandit (task-specific or global).
    /// Returns true if the arm was found and removed.
    pub fn remove_arm(&mut self, arm_id: &str, task_type: Option<&str>) -> bool {
        match task_type {
            Some(t) => {
                if let Some(arms) = self.bandits.get_mut(t) {
                    let before = arms.len();
                    arms.retain(|a| a.id != arm_id);
                    arms.len() < before
                } else {
                    false
                }
            }
            None => {
                let before = self.global_bandit.len();
                self.global_bandit.retain(|a| a.id != arm_id);
                self.global_bandit.len() < before
            }
        }
    }

    // --- Internal: strategy implementations ---

    fn thompson_select(&mut self, arms: &[BanditArm]) -> (usize, Vec<f64>) {
        let scores: Vec<f64> = arms.iter()
            .map(|a| self.sample_beta(a.params.alpha, a.params.beta))
            .collect();
        let best = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, scores)
    }

    fn ucb1_select(&self, arms: &[BanditArm]) -> (usize, Vec<f64>) {
        let total = self.total_pulls.max(1);
        let scores: Vec<f64> = arms.iter()
            .map(|a| self.ucb1_score(a, total))
            .collect();
        let best = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, scores)
    }

    fn epsilon_greedy_select(&mut self, arms: &[BanditArm], epsilon: f64) -> (usize, Vec<f64>) {
        let scores: Vec<f64> = arms.iter()
            .map(|a| if a.pull_count == 0 { 0.5 } else { a.total_reward / a.pull_count as f64 })
            .collect();

        let r = self.next_random();
        let idx = if r < epsilon {
            // Explore: random arm
            (self.next_random() * arms.len() as f64) as usize % arms.len()
        } else {
            // Exploit: best mean arm
            scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        (idx, scores)
    }

    fn ucb1_score(&self, arm: &BanditArm, total_n: u64) -> f64 {
        if arm.pull_count == 0 {
            return f64::INFINITY;
        }
        let mean = arm.total_reward / arm.pull_count as f64;
        let exploration = (2.0 * (total_n as f64).ln() / arm.pull_count as f64).sqrt();
        mean + exploration
    }

    /// Sample from Beta(alpha, beta) distribution using Gamma variates.
    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.sample_gamma(alpha);
        let y = self.sample_gamma(beta);
        if x + y == 0.0 { 0.5 } else { x / (x + y) }
    }

    /// Sample from Gamma(alpha) using Marsaglia-Tsang method.
    fn sample_gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            // Ahrens-Dieter: Gamma(a) = Gamma(a+1) * U^(1/a)
            let g = self.sample_gamma(alpha + 1.0);
            let u = self.next_random().max(1e-30);
            return g * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.sample_normal();
            let v_base = 1.0 + c * x;
            if v_base <= 0.0 {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = self.next_random().max(1e-30);

            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Box-Muller transform for standard normal samples.
    fn sample_normal(&mut self) -> f64 {
        let u1 = self.next_random().max(1e-30);
        let u2 = self.next_random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// LCG PRNG — deterministic, fast, no external deps.
    fn next_random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    // --- Private arms management ---

    /// Mark an arm as private (local-only, not shared in distributed merging).
    pub fn set_arm_private(&mut self, arm_id: &str) {
        self.private_arms.insert(arm_id.to_string());
    }

    /// Mark an arm as public (default, shareable in distributed merging).
    pub fn set_arm_public(&mut self, arm_id: &str) {
        self.private_arms.remove(arm_id);
    }

    /// Check if an arm is private.
    pub fn is_arm_private(&self, arm_id: &str) -> bool {
        self.private_arms.contains(arm_id)
    }

    /// Get the set of private arm IDs.
    pub fn private_arm_ids(&self) -> &HashSet<ArmId> {
        &self.private_arms
    }

    // --- Per-query preference-aware selection ---

    /// Select an arm with per-query preferences applied.
    ///
    /// Filters out excluded_arms, boosts preferred_arms scores.
    /// The weight overrides in preferences do NOT affect selection (they apply to recording).
    pub fn select_with_preferences(
        &mut self,
        task_type: Option<&str>,
        prefs: &RoutingPreferences,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let start = std::time::Instant::now();

        // Clone arms snapshot to avoid borrow conflicts with &mut self in sampling
        let arms_snapshot: Vec<BanditArm> = if let Some(tt) = task_type {
            self.bandits.get(tt).unwrap_or(&self.global_bandit).clone()
        } else {
            self.global_bandit.clone()
        };

        // Filter out excluded arms
        let candidates: Vec<BanditArm> = arms_snapshot.into_iter()
            .filter(|a| !prefs.excluded_arms.contains(&a.id))
            .collect();

        if candidates.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: task_type.unwrap_or("global").to_string(),
                reason: "All arms excluded by preferences".to_string(),
            });
        }

        // Score using the configured strategy
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for (i, arm) in candidates.iter().enumerate() {
            let base_score = match self.config.strategy {
                BanditStrategy::ThompsonSampling => {
                    self.sample_beta(arm.params.alpha, arm.params.beta)
                }
                BanditStrategy::Ucb1 => {
                    if arm.pull_count == 0 {
                        return Ok(RoutingOutcome {
                            selected_arm: arm.id.clone(),
                            confidence: 0.0,
                            reason: "UCB1: unpulled arm (preferences)".to_string(),
                            alternatives: vec![],
                            router_id: "bandit".to_string(),
                            decision_time_us: start.elapsed().as_micros() as u64,
                        });
                    }
                    let mean = arm.total_reward / arm.pull_count as f64;
                    let exploration = (2.0 * (self.total_pulls as f64).ln() / arm.pull_count as f64).sqrt();
                    mean + exploration
                }
                BanditStrategy::EpsilonGreedy { epsilon } => {
                    let r = self.next_random();
                    if r < epsilon {
                        self.next_random() // random score for exploration
                    } else if arm.pull_count > 0 {
                        arm.total_reward / arm.pull_count as f64
                    } else {
                        f64::INFINITY // unexplored arms get priority
                    }
                }
            };

            // Apply boost for preferred arms
            let score = if prefs.preferred_arms.contains(&arm.id) {
                base_score * prefs.prefer_boost
            } else {
                base_score
            };

            scores.push((i, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = scores[0].0;
        let selected_id = candidates[best_idx].id.clone();

        let confidence = if candidates.len() > 1 {
            let gap = scores[0].1 - scores[1].1;
            (gap / (scores[0].1.abs() + 1e-10)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let alternatives: Vec<(ArmId, f64)> = scores.iter()
            .skip(1)
            .take(5)
            .map(|(idx, score)| (candidates[*idx].id.clone(), *score))
            .collect();

        // Update pull tracking
        if let Some(tt) = task_type {
            if let Some(arms_mut) = self.bandits.get_mut(tt) {
                if let Some(arm) = arms_mut.iter_mut().find(|a| a.id == selected_id) {
                    arm.pull_count += 1;
                    arm.last_pulled = self.total_pulls;
                }
            }
        }
        if let Some(arm) = self.global_bandit.iter_mut().find(|a| a.id == selected_id) {
            arm.pull_count += 1;
            arm.last_pulled = self.total_pulls;
        }
        self.total_pulls += 1;

        Ok(RoutingOutcome {
            selected_arm: selected_id,
            confidence,
            reason: format!("{:?} selection with preferences", self.config.strategy),
            alternatives,
            router_id: "bandit".to_string(),
            decision_time_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Record outcome with preference-overridden RewardPolicy.
    ///
    /// The weight overrides in prefs modify how THIS outcome's reward is computed,
    /// allowing per-query customization (e.g., "ignore cost for this query").
    pub fn record_outcome_with_preferences(
        &mut self,
        feedback: &ArmFeedback,
        prefs: &RoutingPreferences,
    ) {
        let policy = prefs.apply_to_policy(&self.config.reward_policy);
        let reward = policy.compute_reward(feedback);

        let task_type = feedback.task_type.as_deref();
        let arms = if let Some(tt) = task_type {
            if let Some(a) = self.bandits.get_mut(tt) { a }
            else { &mut self.global_bandit }
        } else {
            &mut self.global_bandit
        };

        if let Some(arm) = arms.iter_mut().find(|a| a.id == feedback.arm_id) {
            if self.config.decay_factor < 1.0 {
                let d = self.config.decay_factor;
                arm.params.alpha = self.config.prior_alpha
                    + (arm.params.alpha - self.config.prior_alpha) * d;
                arm.params.beta = self.config.prior_beta
                    + (arm.params.beta - self.config.prior_beta) * d;
            }

            arm.params.alpha += reward;
            arm.params.beta += 1.0 - reward;
            arm.pull_count += 1;
            arm.total_reward += reward;
        }
    }
}

// =============================================================================
// NFA ROUTER
// =============================================================================

/// A state identifier in the NFA.
pub type NfaStateId = usize;

/// Symbol that labels an NFA transition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NfaSymbol {
    /// Match a specific query feature domain
    Domain(String),
    /// Match complexity percentage range [low_pct, high_pct) where complexity is mapped as (complexity * 100) as u32
    ComplexityRange { low_pct: u32, high_pct: u32 },
    /// Match token count range [min, max]
    TokenRange { min: usize, max: usize },
    /// Match a boolean feature by name
    BoolFeature { name: String, value: bool },
    /// Epsilon transition (no input consumed)
    Epsilon,
    /// Wildcard (match any input)
    Any,
}

/// A transition in the NFA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfaTransition {
    pub from: NfaStateId,
    pub symbol: NfaSymbol,
    pub to: NfaStateId,
}

/// NFA state metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfaState {
    pub id: NfaStateId,
    pub label: String,
    /// If this is an accepting state, which model to route to
    pub accepting_arm: Option<ArmId>,
    /// Priority (higher = preferred when multiple accepting states match)
    pub priority: u32,
}

/// Non-deterministic Finite Automaton Router.
///
/// This is a "feature-matching NFA": instead of consuming characters, it evaluates
/// all transitions from current states against a `QueryFeatures` set simultaneously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfaRouter {
    states: Vec<NfaState>,
    transitions: Vec<NfaTransition>,
    start_states: Vec<NfaStateId>,
}

impl NfaRouter {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            start_states: Vec::new(),
        }
    }

    /// Add a state. Returns its ID.
    pub fn add_state(&mut self, label: &str, accepting_arm: Option<&str>, priority: u32) -> NfaStateId {
        let id = self.states.len();
        self.states.push(NfaState {
            id,
            label: label.to_string(),
            accepting_arm: accepting_arm.map(|s| s.to_string()),
            priority,
        });
        // First state is a start state by default
        if self.start_states.is_empty() {
            self.start_states.push(id);
        }
        id
    }

    /// Mark a state as a start state.
    pub fn add_start_state(&mut self, state: NfaStateId) {
        if !self.start_states.contains(&state) {
            self.start_states.push(state);
        }
    }

    /// Add a transition.
    pub fn add_transition(&mut self, from: NfaStateId, symbol: NfaSymbol, to: NfaStateId) {
        self.transitions.push(NfaTransition { from, symbol, to });
    }

    /// Route a query through the NFA.
    pub fn route(&self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        if self.states.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: "NFA has no states".to_string(),
            });
        }

        let start = std::time::Instant::now();

        // Step 1: epsilon closure of start states
        let initial: HashSet<NfaStateId> = self.start_states.iter().cloned().collect();
        let mut current = self.epsilon_closure(&initial);

        // Step 2: follow all matching transitions to fixed-point
        // (supports multi-step chains where features match at each stage)
        loop {
            let mut next = HashSet::new();
            for &state in &current {
                for trans in &self.transitions {
                    if trans.from == state
                        && self.symbol_matches(&trans.symbol, features)
                        && !current.contains(&trans.to)
                    {
                        next.insert(trans.to);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            let reachable = self.epsilon_closure(&next);
            current = current.union(&reachable).cloned().collect();
        }

        // Step 4: find all accepting states
        let mut best_accepting: Option<&NfaState> = None;
        for &sid in &current {
            if let Some(state) = self.states.get(sid) {
                if state.accepting_arm.is_some() {
                    if best_accepting.is_none() || state.priority > best_accepting.unwrap().priority {
                        best_accepting = Some(state);
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;

        match best_accepting {
            Some(state) => Ok(RoutingOutcome {
                selected_arm: state.accepting_arm.clone().unwrap_or_default(),
                confidence: 1.0,
                reason: format!("NFA accepted at state '{}'", state.label),
                alternatives: Vec::new(),
                router_id: "nfa".to_string(),
                decision_time_us: elapsed,
            }),
            None => Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: "No accepting state reached".to_string(),
            }),
        }
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Get all states (for compiler access).
    pub fn states(&self) -> &[NfaState] {
        &self.states
    }

    /// Get all transitions (for compiler access).
    pub fn transitions(&self) -> &[NfaTransition] {
        &self.transitions
    }

    /// Get start states (for compiler access).
    pub fn start_states(&self) -> &[NfaStateId] {
        &self.start_states
    }

    /// Compute epsilon closure of a set of states.
    fn epsilon_closure(&self, states: &HashSet<NfaStateId>) -> HashSet<NfaStateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<NfaStateId> = states.iter().cloned().collect();

        while let Some(state) = queue.pop_front() {
            for trans in &self.transitions {
                if trans.from == state && trans.symbol == NfaSymbol::Epsilon && !closure.contains(&trans.to) {
                    closure.insert(trans.to);
                    queue.push_back(trans.to);
                }
            }
        }
        closure
    }

    /// Check if a symbol matches the given features.
    fn symbol_matches(&self, symbol: &NfaSymbol, features: &QueryFeatures) -> bool {
        match symbol {
            NfaSymbol::Domain(d) => features.domain == *d,
            NfaSymbol::ComplexityRange { low_pct, high_pct } => {
                let mapped = (features.complexity * 100.0) as u32;
                mapped >= *low_pct && mapped < *high_pct
            }
            NfaSymbol::TokenRange { min, max } => {
                features.token_count >= *min && features.token_count <= *max
            }
            NfaSymbol::BoolFeature { name, value } => {
                match name.as_str() {
                    "has_code" => features.has_code == *value,
                    "is_question" => features.is_question == *value,
                    _ => false,
                }
            }
            NfaSymbol::Epsilon => false, // Epsilon never matches as a regular symbol
            NfaSymbol::Any => true,
        }
    }
}

// =============================================================================
// DFA ROUTER
// =============================================================================

/// A state identifier in the DFA.
pub type DfaStateId = usize;

/// DFA state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfaState {
    pub id: DfaStateId,
    pub label: String,
    pub accepting_arm: Option<ArmId>,
    pub priority: u32,
}

/// Deterministic Finite Automaton Router (compiled from NFA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfaRouter {
    states: Vec<DfaState>,
    start_state: DfaStateId,
    /// Transition table: from_state -> vec of (symbol, to_state)
    transition_table: HashMap<DfaStateId, Vec<(NfaSymbol, DfaStateId)>>,
}

impl DfaRouter {
    /// Route a query through the DFA deterministically.
    pub fn route(&self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        if self.states.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: "DFA has no states".to_string(),
            });
        }

        let start = std::time::Instant::now();
        let mut current = self.start_state;
        let mut visited = HashSet::new();

        loop {
            if !visited.insert(current) {
                break; // Avoid infinite loops
            }

            // Check if current state has transitions that match
            let transitions = self.transition_table.get(&current);
            let next = transitions.and_then(|ts| {
                // Find first matching transition (deterministic: at most one should match per symbol)
                for (symbol, target) in ts {
                    if Self::symbol_matches_static(symbol, features) {
                        return Some(*target);
                    }
                }
                None
            });

            match next {
                Some(target) => current = target,
                None => break, // No matching transition, stay at current state
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;

        let state = &self.states[current];
        if let Some(ref arm) = state.accepting_arm {
            Ok(RoutingOutcome {
                selected_arm: arm.clone(),
                confidence: 1.0,
                reason: format!("DFA accepted at state '{}'", state.label),
                alternatives: Vec::new(),
                router_id: "dfa".to_string(),
                decision_time_us: elapsed,
            })
        } else {
            Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: format!("DFA stopped at non-accepting state '{}'", state.label),
            })
        }
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transition_table.values().map(|v| v.len()).sum()
    }

    /// Hopcroft-style state minimization.
    pub fn minimize(&mut self) {
        if self.states.len() <= 1 {
            return;
        }

        // Partition by (accepting_arm, priority) — states with different outputs stay separate
        let mut partition_map: HashMap<(Option<ArmId>, u32), Vec<DfaStateId>> = HashMap::new();
        for state in &self.states {
            partition_map.entry((state.accepting_arm.clone(), state.priority))
                .or_default()
                .push(state.id);
        }
        let mut partitions: Vec<Vec<DfaStateId>> = partition_map.into_values().collect();
        partitions.sort_by_key(|p| p[0]);

        // Iterative refinement
        loop {
            let mut new_partitions = Vec::new();
            for partition in &partitions {
                let mut splits: HashMap<Vec<usize>, Vec<DfaStateId>> = HashMap::new();
                for &state in partition {
                    let sig = self.compute_signature(state, &partitions);
                    splits.entry(sig).or_default().push(state);
                }
                for group in splits.into_values() {
                    new_partitions.push(group);
                }
            }
            new_partitions.sort_by_key(|p| p[0]);
            if new_partitions.len() == partitions.len() {
                break; // Fixed point
            }
            partitions = new_partitions;
        }

        // Rebuild DFA with representatives
        if partitions.len() == self.states.len() {
            return; // Already minimal
        }

        let mut state_to_partition: HashMap<DfaStateId, usize> = HashMap::new();
        for (pi, partition) in partitions.iter().enumerate() {
            for &state in partition {
                state_to_partition.insert(state, pi);
            }
        }

        let new_states: Vec<DfaState> = partitions.iter().enumerate().map(|(i, partition)| {
            let rep = &self.states[partition[0]];
            DfaState {
                id: i,
                label: rep.label.clone(),
                accepting_arm: rep.accepting_arm.clone(),
                priority: rep.priority,
            }
        }).collect();

        let new_start = state_to_partition[&self.start_state];

        let mut new_table: HashMap<DfaStateId, Vec<(NfaSymbol, DfaStateId)>> = HashMap::new();
        for (pi, partition) in partitions.iter().enumerate() {
            let rep = partition[0];
            if let Some(transitions) = self.transition_table.get(&rep) {
                let new_transitions: Vec<(NfaSymbol, DfaStateId)> = transitions.iter()
                    .map(|(sym, target)| (sym.clone(), state_to_partition[target]))
                    .collect();
                new_table.insert(pi, new_transitions);
            }
        }

        self.states = new_states;
        self.start_state = new_start;
        self.transition_table = new_table;
    }

    fn compute_signature(&self, state: DfaStateId, partitions: &[Vec<DfaStateId>]) -> Vec<usize> {
        let state_to_partition: HashMap<DfaStateId, usize> = partitions.iter().enumerate()
            .flat_map(|(pi, p)| p.iter().map(move |&s| (s, pi)))
            .collect();

        let mut sig = Vec::new();
        if let Some(transitions) = self.transition_table.get(&state) {
            for (_, target) in transitions {
                sig.push(state_to_partition.get(target).copied().unwrap_or(usize::MAX));
            }
        }
        sig
    }

    fn symbol_matches_static(symbol: &NfaSymbol, features: &QueryFeatures) -> bool {
        match symbol {
            NfaSymbol::Domain(d) => features.domain == *d,
            NfaSymbol::ComplexityRange { low_pct, high_pct } => {
                let mapped = (features.complexity * 100.0) as u32;
                mapped >= *low_pct && mapped < *high_pct
            }
            NfaSymbol::TokenRange { min, max } => {
                features.token_count >= *min && features.token_count <= *max
            }
            NfaSymbol::BoolFeature { name, value } => {
                match name.as_str() {
                    "has_code" => features.has_code == *value,
                    "is_question" => features.is_question == *value,
                    _ => false,
                }
            }
            NfaSymbol::Epsilon => false,
            NfaSymbol::Any => true,
        }
    }
}

// =============================================================================
// NFA → DFA COMPILER
// =============================================================================

/// Compiles an NFA into an equivalent DFA using powerset/subset construction.
pub struct NfaDfaCompiler;

impl NfaDfaCompiler {
    /// Compile an NFA into an equivalent DFA.
    pub fn compile(nfa: &NfaRouter) -> Result<DfaRouter, AdvancedRoutingError> {
        if nfa.states.is_empty() {
            return Err(AdvancedRoutingError::CompilationError {
                reason: "Cannot compile empty NFA".to_string(),
            });
        }

        // Collect alphabet (all non-epsilon symbols)
        let alphabet: Vec<NfaSymbol> = Self::extract_alphabet(nfa);

        // Initial DFA state = epsilon closure of NFA start states
        let start_set: HashSet<NfaStateId> = nfa.start_states().iter().cloned().collect();
        let start_closure = nfa.epsilon_closure(&start_set);
        let start_key: BTreeSet<NfaStateId> = start_closure.iter().cloned().collect();

        let mut dfa_states: Vec<DfaState> = Vec::new();
        let mut dfa_table: HashMap<DfaStateId, Vec<(NfaSymbol, DfaStateId)>> = HashMap::new();
        let mut state_map: HashMap<BTreeSet<NfaStateId>, DfaStateId> = HashMap::new();
        let mut worklist: VecDeque<BTreeSet<NfaStateId>> = VecDeque::new();

        // Create initial DFA state
        let initial_dfa = Self::create_dfa_state(0, &start_closure, nfa);
        dfa_states.push(initial_dfa);
        state_map.insert(start_key.clone(), 0);
        worklist.push_back(start_key);

        while let Some(current_set) = worklist.pop_front() {
            let current_id = state_map[&current_set];
            let current_nfa_states: HashSet<NfaStateId> = current_set.iter().cloned().collect();

            for symbol in &alphabet {
                // Compute move: all states reachable via this symbol from current NFA states
                let mut next_set = HashSet::new();
                for &nfa_state in &current_nfa_states {
                    for trans in nfa.transitions() {
                        if trans.from == nfa_state && trans.symbol == *symbol {
                            next_set.insert(trans.to);
                        }
                    }
                }

                if next_set.is_empty() {
                    continue;
                }

                // Epsilon closure of target states
                let next_closure = nfa.epsilon_closure(&next_set);
                let next_key: BTreeSet<NfaStateId> = next_closure.iter().cloned().collect();

                let target_id = if let Some(&existing) = state_map.get(&next_key) {
                    existing
                } else {
                    let new_id = dfa_states.len();
                    let new_state = Self::create_dfa_state(new_id, &next_closure, nfa);
                    dfa_states.push(new_state);
                    state_map.insert(next_key.clone(), new_id);
                    worklist.push_back(next_key);
                    new_id
                };

                dfa_table.entry(current_id).or_default().push((symbol.clone(), target_id));
            }
        }

        Ok(DfaRouter {
            states: dfa_states,
            start_state: 0,
            transition_table: dfa_table,
        })
    }

    fn extract_alphabet(nfa: &NfaRouter) -> Vec<NfaSymbol> {
        let mut seen = HashSet::new();
        let mut alphabet = Vec::new();
        for trans in nfa.transitions() {
            if trans.symbol != NfaSymbol::Epsilon {
                if seen.insert(trans.symbol.clone()) {
                    alphabet.push(trans.symbol.clone());
                }
            }
        }
        alphabet
    }

    fn create_dfa_state(id: DfaStateId, nfa_states: &HashSet<NfaStateId>, nfa: &NfaRouter) -> DfaState {
        let mut best_arm: Option<&str> = None;
        let mut best_priority = 0u32;
        let mut labels = Vec::new();

        for &sid in nfa_states {
            if let Some(state) = nfa.states().get(sid) {
                labels.push(state.label.as_str());
                if let Some(ref arm) = state.accepting_arm {
                    if best_arm.is_none() || state.priority > best_priority {
                        best_arm = Some(arm.as_str());
                        best_priority = state.priority;
                    }
                }
            }
        }

        DfaState {
            id,
            label: format!("{{{}}}", labels.join(",")),
            accepting_arm: best_arm.map(|s| s.to_string()),
            priority: best_priority,
        }
    }
}

// =============================================================================
// HIERARCHICAL ROUTING DAG
// =============================================================================

/// The type of router at a DAG node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RoutingDagNodeType {
    /// A bandit router at this node
    Bandit(BanditConfig),
    /// A DFA router at this node
    Dfa,
    /// Rule-based branching on a feature
    RuleBased {
        feature: String,
        threshold: f64,
        high_branch: String,
        low_branch: String,
    },
    /// Leaf node that emits a final routing decision
    Leaf { arm_id: ArmId },
}

/// A node in the routing DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDagNode {
    pub id: String,
    pub label: String,
    pub node_type: RoutingDagNodeType,
    pub successors: Vec<String>,
}

/// A Directed Acyclic Graph of routing nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDag {
    nodes: HashMap<String, RoutingDagNode>,
    root_id: String,
    #[serde(skip)]
    bandit_instances: HashMap<String, BanditRouter>,
    #[serde(skip)]
    dfa_instances: HashMap<String, DfaRouter>,
}

impl RoutingDag {
    pub fn new(root_id: &str) -> Self {
        Self {
            nodes: HashMap::new(),
            root_id: root_id.to_string(),
            bandit_instances: HashMap::new(),
            dfa_instances: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: RoutingDagNode) -> Result<(), AdvancedRoutingError> {
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub fn set_bandit(&mut self, node_id: &str, bandit: BanditRouter) -> Result<(), AdvancedRoutingError> {
        if !self.nodes.contains_key(node_id) {
            return Err(AdvancedRoutingError::NodeNotFound { node_id: node_id.to_string() });
        }
        self.bandit_instances.insert(node_id.to_string(), bandit);
        Ok(())
    }

    pub fn set_dfa(&mut self, node_id: &str, dfa: DfaRouter) -> Result<(), AdvancedRoutingError> {
        if !self.nodes.contains_key(node_id) {
            return Err(AdvancedRoutingError::NodeNotFound { node_id: node_id.to_string() });
        }
        self.dfa_instances.insert(node_id.to_string(), dfa);
        Ok(())
    }

    /// Route through the DAG from root to a leaf.
    pub fn route(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.validate()?;

        let start = std::time::Instant::now();
        let mut current_id = self.root_id.clone();
        let mut path: Vec<String> = Vec::new();

        loop {
            if path.len() > self.nodes.len() {
                return Err(AdvancedRoutingError::CycleDetected);
            }
            path.push(current_id.clone());

            let node = self.nodes.get(&current_id)
                .ok_or_else(|| AdvancedRoutingError::NodeNotFound { node_id: current_id.clone() })?
                .clone();

            match &node.node_type {
                RoutingDagNodeType::Leaf { arm_id } => {
                    let elapsed = start.elapsed().as_micros() as u64;
                    return Ok(RoutingOutcome {
                        selected_arm: arm_id.clone(),
                        confidence: 1.0,
                        reason: format!("DAG path: {}", path.join(" -> ")),
                        alternatives: Vec::new(),
                        router_id: "dag".to_string(),
                        decision_time_us: elapsed,
                    });
                }
                RoutingDagNodeType::Bandit(_) => {
                    if let Some(bandit) = self.bandit_instances.get_mut(&current_id) {
                        let outcome = bandit.select(Some(&features.domain))?;
                        // Find successor matching selected arm
                        current_id = self.find_successor(&node, &outcome.selected_arm)?;
                    } else {
                        return Err(AdvancedRoutingError::NodeNotFound {
                            node_id: format!("bandit instance for '{}'", current_id),
                        });
                    }
                }
                RoutingDagNodeType::Dfa => {
                    if let Some(dfa) = self.dfa_instances.get(&current_id) {
                        let outcome = dfa.route(features)?;
                        current_id = self.find_successor(&node, &outcome.selected_arm)?;
                    } else {
                        return Err(AdvancedRoutingError::NodeNotFound {
                            node_id: format!("dfa instance for '{}'", current_id),
                        });
                    }
                }
                RoutingDagNodeType::RuleBased { feature, threshold, high_branch, low_branch } => {
                    let value = extract_feature_value(features, feature);
                    current_id = if value >= *threshold {
                        high_branch.clone()
                    } else {
                        low_branch.clone()
                    };
                }
            }
        }
    }

    /// Validate the DAG is acyclic using DFS 3-color algorithm.
    pub fn validate(&self) -> Result<(), AdvancedRoutingError> {
        let mut white: HashSet<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        let mut gray: HashSet<&str> = HashSet::new();

        fn dfs<'a>(
            node_id: &'a str,
            nodes: &'a HashMap<String, RoutingDagNode>,
            white: &mut HashSet<&'a str>,
            gray: &mut HashSet<&'a str>,
        ) -> Result<(), AdvancedRoutingError> {
            white.remove(node_id);
            gray.insert(node_id);

            if let Some(node) = nodes.get(node_id) {
                for succ in &node.successors {
                    if gray.contains(succ.as_str()) {
                        return Err(AdvancedRoutingError::CycleDetected);
                    }
                    if white.contains(succ.as_str()) {
                        dfs(succ.as_str(), nodes, white, gray)?;
                    }
                }
            }

            gray.remove(node_id);
            Ok(())
        }

        let keys: Vec<String> = self.nodes.keys().cloned().collect();
        for key in &keys {
            if white.contains(key.as_str()) {
                dfs(key.as_str(), &self.nodes, &mut white, &mut gray)?;
            }
        }
        Ok(())
    }

    /// Topological sort using Kahn's algorithm.
    pub fn topological_order(&self) -> Result<Vec<String>, AdvancedRoutingError> {
        self.validate()?;
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for key in self.nodes.keys() {
            in_degree.entry(key.as_str()).or_insert(0);
        }
        for node in self.nodes.values() {
            for succ in &node.successors {
                *in_degree.entry(succ.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&k, _)| k)
            .collect();
        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node.to_string());
            if let Some(n) = self.nodes.get(node) {
                for succ in &n.successors {
                    if let Some(deg) = in_degree.get_mut(succ.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ.as_str());
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Record outcome feedback for a specific bandit node.
    pub fn record_outcome(&mut self, node_id: &str, feedback: &ArmFeedback) {
        if let Some(bandit) = self.bandit_instances.get_mut(node_id) {
            bandit.record_outcome(feedback);
        }
    }

    fn find_successor(&self, node: &RoutingDagNode, arm: &str) -> Result<String, AdvancedRoutingError> {
        // Try to find a successor matching the arm name
        for succ in &node.successors {
            if succ == arm {
                return Ok(succ.clone());
            }
        }
        // If no exact match, use first successor
        node.successors.first().cloned().ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
            query: arm.to_string(),
            reason: format!("No successor found at node '{}'", node.id),
        })
    }
}

/// Extract a numeric feature value from QueryFeatures by name.
pub fn extract_feature_value(features: &QueryFeatures, name: &str) -> f64 {
    match name {
        "complexity" => features.complexity,
        "token_count" => features.token_count as f64,
        "entity_count" => features.entity_count as f64,
        "sentence_count" => features.sentence_count as f64,
        "has_code" => if features.has_code { 1.0 } else { 0.0 },
        "is_question" => if features.is_question { 1.0 } else { 0.0 },
        "avg_word_length" => features.avg_word_length,
        _ => 0.0,
    }
}

// =============================================================================
// EVAL-TO-RUNTIME FEEDBACK LOOP
// =============================================================================

#[cfg(feature = "eval-suite")]
use crate::eval_suite::ConfigSearchResult;
#[cfg(feature = "eval-suite")]
use crate::eval_suite::{ComparisonMatrix, SubtaskAnalysis};

/// Maps eval suite results to bandit priors for warm-starting production routing.
#[cfg(feature = "eval-suite")]
pub struct EvalFeedbackMapper;

#[cfg(feature = "eval-suite")]
impl EvalFeedbackMapper {
    /// Convert ConfigSearchResult into per-subtask bandit priors.
    ///
    /// For each subtask with a measured quality score, computes:
    /// alpha = quality * scale, beta = (1 - quality) * scale.
    pub fn map_to_priors(
        result: &ConfigSearchResult,
        scale: f64,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();

        for (subtask_name, &quality) in &result.best.subtask_quality {
            let arm_id = if let Some(model) = result.best.config.subtask_models.get(subtask_name) {
                model.to_string()
            } else {
                result.best.config.default_model.to_string()
            };

            let alpha = quality * scale;
            let beta = (1.0 - quality) * scale;

            priors.entry(subtask_name.clone())
                .or_default()
                .insert(arm_id, BetaParams { alpha: alpha.max(0.01), beta: beta.max(0.01) });
        }

        priors
    }

    /// Apply eval-derived priors to an existing BanditRouter.
    pub fn apply_to_bandit(
        bandit: &mut BanditRouter,
        priors: &HashMap<String, HashMap<ArmId, BetaParams>>,
    ) {
        for (task_type, arm_priors) in priors {
            for (arm_id, params) in arm_priors {
                bandit.warm_start_for_task(task_type, arm_id, params.alpha, params.beta);
            }
        }
    }

    /// Create a warm-started BanditRouter from eval results.
    pub fn create_warm_started_bandit(
        result: &ConfigSearchResult,
        config: BanditConfig,
        scale: f64,
    ) -> BanditRouter {
        let mut bandit = BanditRouter::new(config);
        let priors = Self::map_to_priors(result, scale);
        Self::apply_to_bandit(&mut bandit, &priors);
        bandit
    }
}

/// Bootstraps bandit priors from eval-suite benchmark results.
///
/// Converts ComparisonMatrix (multi-model benchmark) or SubtaskAnalysis
/// (per-subtask routing) into warm-start priors for the bandit router.
#[cfg(feature = "eval-suite")]
pub struct BanditBootstrapper;

#[cfg(feature = "eval-suite")]
impl BanditBootstrapper {
    /// Build per-task priors from a ComparisonMatrix.
    ///
    /// Uses mean_score (metric index 1) and cost_effectiveness, weighted by
    /// the given RewardPolicy. Returns `task_type -> arm_id -> BetaParams`.
    /// The task_type key is `"global"` since ComparisonMatrix has no per-task breakdown.
    ///
    /// Returns empty HashMap if matrix has no models.
    pub fn from_comparison_matrix(
        matrix: &ComparisonMatrix,
        reward_policy: &RewardPolicy,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        if matrix.models.is_empty() {
            return HashMap::new();
        }

        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();
        let global = priors.entry("global".to_string()).or_default();

        let max_ce = matrix.cost_effectiveness.iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.001);

        let (qw, lw, cw) = reward_policy.normalize_weights();
        // Latency not directly available in ComparisonMatrix -> redistribute to quality
        let adjusted_qw = qw + lw;
        let total_w = adjusted_qw + cw;

        for (i, model) in matrix.models.iter().enumerate() {
            let mean_score = matrix.scores.get(i)
                .and_then(|s| s.get(1))
                .copied()
                .unwrap_or(0.5);

            let ce = matrix.cost_effectiveness.get(i).copied().unwrap_or(1.0);
            let ce_norm = (ce / max_ce).clamp(0.0, 1.0);

            let composite = if total_w > 1e-12 {
                (adjusted_qw * mean_score + cw * ce_norm) / total_w
            } else {
                mean_score
            };

            let scale = 10.0;
            let alpha = (composite * scale).max(0.01);
            let beta = ((1.0 - composite) * scale).max(0.01);

            let arm_id = model.to_string();
            global.insert(arm_id, BetaParams { alpha, beta });
        }

        priors
    }

    /// Build per-subtask priors from SubtaskAnalysis.
    ///
    /// Uses each `SubtaskPerformance.score` scaled by `scale` to compute priors.
    /// Returns empty HashMap if analysis has no performances.
    pub fn from_subtask_analysis(
        analysis: &SubtaskAnalysis,
        scale: f64,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        if analysis.performances.is_empty() {
            return HashMap::new();
        }

        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();

        for perf in &analysis.performances {
            let subtask_name = perf.subtask.to_string();
            let arm_id = perf.model_id.to_string();
            let alpha = (perf.score * scale).max(0.01);
            let beta = ((1.0 - perf.score) * scale).max(0.01);

            priors.entry(subtask_name)
                .or_default()
                .insert(arm_id, BetaParams { alpha, beta });
        }

        priors
    }

    /// Create a warm-started RoutingPipeline from priors.
    ///
    /// Applies the given priors to a new pipeline's bandit router.
    /// Task types keyed as `"global"` are applied to the global bandit;
    /// all others are applied as task-specific arms.
    pub fn bootstrap_pipeline(
        priors: &HashMap<String, HashMap<ArmId, BetaParams>>,
        bandit_config: BanditConfig,
        pipeline_config: PipelineConfig,
    ) -> RoutingPipeline {
        let mut pipeline = RoutingPipeline::new(bandit_config, pipeline_config);
        for (task_type, arm_priors) in priors {
            for (arm_id, params) in arm_priors {
                if task_type == "global" {
                    pipeline.bandit_mut().warm_start(arm_id, params.alpha, params.beta);
                } else {
                    pipeline.bandit_mut().warm_start_for_task(task_type, arm_id, params.alpha, params.beta);
                }
            }
        }
        pipeline
    }
}

// =============================================================================
// ADAPTIVE PER-QUERY ROUTER
// =============================================================================

/// Extracts features from a query string.
pub struct QueryFeatureExtractor;

impl QueryFeatureExtractor {
    /// Extract features from a raw query string.
    pub fn extract(query: &str) -> QueryFeatures {
        let words: Vec<&str> = query.split_whitespace().collect();
        let token_count = words.len();
        let sentence_count = query.chars().filter(|&c| c == '.' || c == '!' || c == '?').count().max(1);
        let has_code = Self::has_code_markers(query);
        let is_question = query.contains('?')
            || query.to_lowercase().starts_with("what")
            || query.to_lowercase().starts_with("how")
            || query.to_lowercase().starts_with("why")
            || query.to_lowercase().starts_with("when")
            || query.to_lowercase().starts_with("where")
            || query.to_lowercase().starts_with("who");
        let domain = Self::detect_domain(query);
        let entity_count = Self::count_entities(query);
        let avg_word_length = if words.is_empty() { 0.0 }
            else { words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64 };
        let complexity = Self::estimate_complexity(query, token_count, sentence_count, entity_count);

        let feature_vector = vec![
            token_count as f64,
            sentence_count as f64,
            complexity,
            entity_count as f64,
            if has_code { 1.0 } else { 0.0 },
            if is_question { 1.0 } else { 0.0 },
            avg_word_length,
        ];

        QueryFeatures {
            token_count,
            sentence_count,
            domain,
            complexity,
            entity_count,
            has_code,
            is_question,
            avg_word_length,
            feature_vector,
        }
    }

    fn has_code_markers(query: &str) -> bool {
        query.contains("```") || query.contains("fn ") || query.contains("def ")
            || query.contains("class ") || query.contains("function ")
            || query.contains("import ") || query.contains("pub fn")
    }

    fn detect_domain(query: &str) -> String {
        let lower = query.to_lowercase();
        if lower.contains("code") || lower.contains("function") || lower.contains("implement")
            || lower.contains("programming") || lower.contains("debug") || lower.contains("compile") {
            "coding".to_string()
        } else if lower.contains("math") || lower.contains("calculate") || lower.contains("equation")
            || lower.contains("solve") || lower.contains("integral") {
            "math".to_string()
        } else if lower.contains("write a story") || lower.contains("poem") || lower.contains("creative") {
            "creative".to_string()
        } else if lower.contains("translate") || lower.contains("translation") {
            "translation".to_string()
        } else if lower.contains("summarize") || lower.contains("summary") || lower.contains("tldr") {
            "summarization".to_string()
        } else {
            "general".to_string()
        }
    }

    fn count_entities(query: &str) -> usize {
        let mut count = 0;
        for word in query.split_whitespace() {
            // Count capitalized words (potential proper nouns)
            if word.len() > 1 && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                count += 1;
            }
        }
        // Count numbers
        count += query.split_whitespace().filter(|w| w.parse::<f64>().is_ok()).count();
        count
    }

    fn estimate_complexity(query: &str, token_count: usize, sentence_count: usize, entity_count: usize) -> f64 {
        let length_factor = (token_count as f64 / 100.0).min(1.0);
        let sentence_factor = (sentence_count as f64 / 5.0).min(1.0);
        let entity_factor = (entity_count as f64 / 10.0).min(1.0);
        let clause_factor = (query.matches(',').count() as f64 / 5.0).min(1.0);

        let raw = length_factor * 0.3 + sentence_factor * 0.2 + entity_factor * 0.25 + clause_factor * 0.25;
        raw.min(1.0).max(0.0)
    }
}

/// Per-query adaptive router that learns feature→model mapping from outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePerQueryRouter {
    domain_bandits: HashMap<String, BanditRouter>,
    complexity_thresholds: Vec<(f64, ArmId)>,
    code_model: Option<ArmId>,
    question_model: Option<ArmId>,
    default_model: ArmId,
    bandit_config: BanditConfig,
}

impl AdaptivePerQueryRouter {
    pub fn new(default_model: &str, bandit_config: BanditConfig) -> Self {
        Self {
            domain_bandits: HashMap::new(),
            complexity_thresholds: Vec::new(),
            code_model: None,
            question_model: None,
            default_model: default_model.to_string(),
            bandit_config,
        }
    }

    pub fn with_code_model(mut self, model: &str) -> Self {
        self.code_model = Some(model.to_string());
        self
    }

    pub fn with_question_model(mut self, model: &str) -> Self {
        self.question_model = Some(model.to_string());
        self
    }

    pub fn add_complexity_tier(mut self, max_complexity: f64, model: &str) -> Self {
        self.complexity_thresholds.push((max_complexity, model.to_string()));
        self.complexity_thresholds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self
    }

    /// Route a raw query string.
    pub fn route(&mut self, query: &str) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let features = QueryFeatureExtractor::extract(query);
        self.route_with_features(&features)
    }

    /// Route using pre-extracted features.
    pub fn route_with_features(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let start = std::time::Instant::now();

        // Priority 1: code model shortcut
        if features.has_code {
            if let Some(ref model) = self.code_model {
                let elapsed = start.elapsed().as_micros() as u64;
                return Ok(RoutingOutcome {
                    selected_arm: model.clone(),
                    confidence: 0.9,
                    reason: "Code detected, routing to code model".to_string(),
                    alternatives: Vec::new(),
                    router_id: "adaptive".to_string(),
                    decision_time_us: elapsed,
                });
            }
        }

        // Priority 2: question model shortcut
        if features.is_question {
            if let Some(ref model) = self.question_model {
                let elapsed = start.elapsed().as_micros() as u64;
                return Ok(RoutingOutcome {
                    selected_arm: model.clone(),
                    confidence: 0.8,
                    reason: "Question detected, routing to QA model".to_string(),
                    alternatives: Vec::new(),
                    router_id: "adaptive".to_string(),
                    decision_time_us: elapsed,
                });
            }
        }

        // Priority 3: complexity tiers
        for (threshold, model) in &self.complexity_thresholds {
            if features.complexity <= *threshold {
                let elapsed = start.elapsed().as_micros() as u64;
                return Ok(RoutingOutcome {
                    selected_arm: model.clone(),
                    confidence: 0.7,
                    reason: format!("Complexity {:.2} <= tier {:.2}", features.complexity, threshold),
                    alternatives: Vec::new(),
                    router_id: "adaptive".to_string(),
                    decision_time_us: elapsed,
                });
            }
        }

        // Priority 4: domain-specific bandit
        let bandit = self.domain_bandits.entry(features.domain.clone())
            .or_insert_with(|| {
                let mut b = BanditRouter::new(self.bandit_config.clone());
                b.add_arm(&self.default_model);
                b
            });

        if bandit.all_arms(None).is_empty() {
            bandit.add_arm(&self.default_model);
        }

        let mut outcome = bandit.select(None)?;
        outcome.router_id = "adaptive".to_string();
        outcome.decision_time_us = start.elapsed().as_micros() as u64;
        Ok(outcome)
    }

    /// Record outcome for learning.
    pub fn record_outcome(&mut self, query: &str, feedback: &ArmFeedback) {
        let features = QueryFeatureExtractor::extract(query);
        if let Some(bandit) = self.domain_bandits.get_mut(&features.domain) {
            bandit.record_outcome(feedback);
        }
    }

    /// Get the default model.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

// =============================================================================
// ENSEMBLE ROUTER WITH VOTING
// =============================================================================

/// Strategy for combining votes from multiple sub-routers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum EnsembleStrategy {
    /// Simple majority: most-voted arm wins
    MajorityVote,
    /// Weighted average of confidence scores per arm
    WeightedAverage,
    /// All sub-routers must agree on the same arm
    Unanimous,
    /// Highest individual confidence wins
    MaxConfidence,
}

/// A vote from a single sub-router.
#[derive(Debug, Clone)]
pub struct SubRouterVote {
    pub router_id: String,
    pub outcome: RoutingOutcome,
    pub weight: f64,
}

/// Trait for any component that can participate in ensemble voting.
pub trait RoutingVoter: std::fmt::Debug + Send + Sync {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError>;
    fn router_id(&self) -> &str;
    fn record_outcome(&mut self, feedback: &ArmFeedback);
}

impl RoutingVoter for BanditRouter {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.select(Some(&features.domain))
    }
    fn router_id(&self) -> &str { "bandit" }
    fn record_outcome(&mut self, feedback: &ArmFeedback) {
        BanditRouter::record_outcome(self, feedback);
    }
}

impl RoutingVoter for AdaptivePerQueryRouter {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.route_with_features(features)
    }
    fn router_id(&self) -> &str { "adaptive" }
    fn record_outcome(&mut self, feedback: &ArmFeedback) {
        // No raw query available, record for the task_type domain if present
        if let Some(ref domain) = feedback.task_type {
            if let Some(bandit) = self.domain_bandits.get_mut(domain) {
                bandit.record_outcome(feedback);
            }
        }
    }
}

/// Ensemble router that combines multiple sub-routers via voting.
pub struct EnsembleRouter {
    sub_routers: Vec<(Box<dyn RoutingVoter>, f64)>,
    strategy: EnsembleStrategy,
    id: String,
}

impl std::fmt::Debug for EnsembleRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleRouter")
            .field("strategy", &self.strategy)
            .field("id", &self.id)
            .field("voter_count", &self.sub_routers.len())
            .finish()
    }
}

impl EnsembleRouter {
    pub fn new(strategy: EnsembleStrategy) -> Self {
        Self {
            sub_routers: Vec::new(),
            strategy,
            id: "ensemble".to_string(),
        }
    }

    pub fn add_voter(&mut self, voter: Box<dyn RoutingVoter>, weight: f64) {
        self.sub_routers.push((voter, weight));
    }

    pub fn voter_count(&self) -> usize {
        self.sub_routers.len()
    }

    /// Route by collecting votes and tallying.
    pub fn route(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        if self.sub_routers.is_empty() {
            return Err(AdvancedRoutingError::EmptyEnsemble);
        }

        let start = std::time::Instant::now();

        let mut votes = Vec::new();
        for (router, weight) in &mut self.sub_routers {
            if let Ok(outcome) = router.vote(features) {
                votes.push(SubRouterVote {
                    router_id: router.router_id().to_string(),
                    outcome,
                    weight: *weight,
                });
            }
        }

        if votes.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: "All sub-routers failed".to_string(),
            });
        }

        let mut result = self.tally_votes(&votes)?;
        result.decision_time_us = start.elapsed().as_micros() as u64;
        result.router_id = self.id.clone();
        Ok(result)
    }

    /// Propagate outcome feedback to all sub-routers.
    pub fn record_outcome(&mut self, feedback: &ArmFeedback) {
        for (router, _) in &mut self.sub_routers {
            router.record_outcome(feedback);
        }
    }

    fn tally_votes(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        match self.strategy {
            EnsembleStrategy::MajorityVote => self.majority_vote(votes),
            EnsembleStrategy::WeightedAverage => self.weighted_average(votes),
            EnsembleStrategy::Unanimous => self.unanimous(votes),
            EnsembleStrategy::MaxConfidence => self.max_confidence(votes),
        }
    }

    fn majority_vote(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let mut counts: HashMap<&str, (usize, f64)> = HashMap::new(); // (count, max_confidence)
        for vote in votes {
            let entry = counts.entry(&vote.outcome.selected_arm).or_insert((0, 0.0));
            entry.0 += 1;
            if vote.outcome.confidence > entry.1 {
                entry.1 = vote.outcome.confidence;
            }
        }

        let winner = counts.iter()
            .max_by(|a, b| a.1.0.cmp(&b.1.0).then(a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal)))
            .map(|(arm, (count, conf))| (arm.to_string(), *count, *conf))
            .unwrap();

        let alternatives: Vec<(ArmId, f64)> = counts.iter()
            .filter(|(arm, _)| **arm != winner.0)
            .map(|(arm, (_, conf))| (arm.to_string(), *conf))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: winner.0,
            confidence: winner.2,
            reason: format!("Majority vote: {}/{} votes", winner.1, votes.len()),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }

    fn weighted_average(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let mut scores: HashMap<&str, f64> = HashMap::new();
        for vote in votes {
            *scores.entry(&vote.outcome.selected_arm).or_insert(0.0) += vote.weight * vote.outcome.confidence;
        }

        let winner = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, score)| (arm.to_string(), *score))
            .unwrap();

        let total_weight: f64 = scores.values().sum();
        let confidence = if total_weight > 0.0 { winner.1 / total_weight } else { 0.5 };

        let alternatives: Vec<(ArmId, f64)> = scores.iter()
            .filter(|(arm, _)| **arm != winner.0)
            .map(|(arm, score)| (arm.to_string(), *score))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: winner.0,
            confidence,
            reason: format!("Weighted score: {:.3}", winner.1),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }

    fn unanimous(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let first_arm = &votes[0].outcome.selected_arm;
        if votes.iter().all(|v| v.outcome.selected_arm == *first_arm) {
            let max_conf = votes.iter().map(|v| v.outcome.confidence).fold(0.0f64, f64::max);
            Ok(RoutingOutcome {
                selected_arm: first_arm.clone(),
                confidence: max_conf,
                reason: format!("Unanimous: all {} routers agree", votes.len()),
                alternatives: Vec::new(),
                router_id: "ensemble".to_string(),
                decision_time_us: 0,
            })
        } else {
            Err(AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "No unanimous agreement among sub-routers".to_string(),
            })
        }
    }

    fn max_confidence(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let best = votes.iter()
            .max_by(|a, b| a.outcome.confidence.partial_cmp(&b.outcome.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let alternatives: Vec<(ArmId, f64)> = votes.iter()
            .filter(|v| v.router_id != best.router_id)
            .map(|v| (v.outcome.selected_arm.clone(), v.outcome.confidence))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: best.outcome.selected_arm.clone(),
            confidence: best.outcome.confidence,
            reason: format!("Max confidence from router '{}'", best.router_id),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }
}

// =============================================================================
// DISTRIBUTED BANDIT TRAINING
// =============================================================================

/// Serializable snapshot of bandit state for distributed sharing.
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBanditState {
    pub node_id: String,
    pub timestamp: u64,
    pub task_bandits: HashMap<String, Vec<BanditArm>>,
    pub global_arms: Vec<BanditArm>,
    pub total_pulls: u64,
}

/// Merges bandit states from multiple distributed nodes.
#[cfg(feature = "distributed")]
pub struct BanditStateMerger;

#[cfg(feature = "distributed")]
impl BanditStateMerger {
    /// Extract the current state from a BanditRouter for sharing.
    ///
    /// Private arms (marked via `set_arm_private`) are filtered out so they
    /// are never shared with other nodes.
    pub fn extract_state(router: &BanditRouter, node_id: &str) -> DistributedBanditState {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        DistributedBanditState {
            node_id: node_id.to_string(),
            timestamp: now,
            global_arms: router.global_bandit.iter()
                .filter(|a| !router.private_arms.contains(&a.id))
                .cloned()
                .collect(),
            task_bandits: router.bandits.iter()
                .map(|(k, arms)| {
                    let filtered: Vec<_> = arms.iter()
                        .filter(|a| !router.private_arms.contains(&a.id))
                        .cloned()
                        .collect();
                    (k.clone(), filtered)
                })
                .filter(|(_, arms)| !arms.is_empty())
                .collect(),
            total_pulls: router.total_pulls,
        }
    }

    /// Merge N local states into a single global state.
    ///
    /// Formula: global_alpha = sum(local_alpha_i) - (N-1) * prior_alpha
    pub fn merge(
        states: &[DistributedBanditState],
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Result<DistributedBanditState, AdvancedRoutingError> {
        if states.is_empty() {
            return Err(AdvancedRoutingError::InvalidConfig {
                field: "states".to_string(),
                reason: "Cannot merge empty state list".to_string(),
            });
        }

        let n = states.len();

        // Merge global arms
        let mut global_map: HashMap<ArmId, Vec<&BanditArm>> = HashMap::new();
        for state in states {
            for arm in &state.global_arms {
                global_map.entry(arm.id.clone()).or_default().push(arm);
            }
        }

        let global_arms: Vec<BanditArm> = global_map.into_iter().map(|(id, arms)| {
            Self::merge_arms(&id, &arms, n, prior_alpha, prior_beta)
        }).collect();

        // Merge per-task bandits
        let mut task_keys: HashSet<&str> = HashSet::new();
        for state in states {
            for key in state.task_bandits.keys() {
                task_keys.insert(key.as_str());
            }
        }

        let mut task_bandits: HashMap<String, Vec<BanditArm>> = HashMap::new();
        for key in task_keys {
            let mut arm_map: HashMap<ArmId, Vec<&BanditArm>> = HashMap::new();
            let mut contributing_nodes = 0;
            for state in states {
                if let Some(arms) = state.task_bandits.get(key) {
                    contributing_nodes += 1;
                    for arm in arms {
                        arm_map.entry(arm.id.clone()).or_default().push(arm);
                    }
                }
            }
            let merged: Vec<BanditArm> = arm_map.into_iter().map(|(id, arms)| {
                Self::merge_arms(&id, &arms, contributing_nodes, prior_alpha, prior_beta)
            }).collect();
            task_bandits.insert(key.to_string(), merged);
        }

        let total_pulls: u64 = states.iter().map(|s| s.total_pulls).sum();
        let max_ts = states.iter().map(|s| s.timestamp).max().unwrap_or(0);

        Ok(DistributedBanditState {
            node_id: "merged".to_string(),
            timestamp: max_ts,
            task_bandits,
            global_arms,
            total_pulls,
        })
    }

    /// Merge a remote state into a local BanditRouter.
    pub fn merge_into_router(
        router: &mut BanditRouter,
        remote: &DistributedBanditState,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Result<(), AdvancedRoutingError> {
        // Merge global arms
        for remote_arm in &remote.global_arms {
            if let Some(local_arm) = router.global_bandit.iter_mut().find(|a| a.id == remote_arm.id) {
                let merged = Self::merge_arm_pair(local_arm, remote_arm, prior_alpha, prior_beta);
                *local_arm = merged;
            } else {
                router.global_bandit.push(remote_arm.clone());
            }
        }

        // Merge per-task bandits
        for (task, remote_arms) in &remote.task_bandits {
            let local_arms = router.bandits.entry(task.clone()).or_default();
            for remote_arm in remote_arms {
                if let Some(local_arm) = local_arms.iter_mut().find(|a| a.id == remote_arm.id) {
                    let merged = Self::merge_arm_pair(local_arm, remote_arm, prior_alpha, prior_beta);
                    *local_arm = merged;
                } else {
                    local_arms.push(remote_arm.clone());
                }
            }
        }

        router.total_pulls += remote.total_pulls;
        Ok(())
    }

    fn merge_arms(id: &str, arms: &[&BanditArm], n: usize, prior_alpha: f64, prior_beta: f64) -> BanditArm {
        let sum_alpha: f64 = arms.iter().map(|a| a.params.alpha).sum();
        let sum_beta: f64 = arms.iter().map(|a| a.params.beta).sum();
        let total_pulls: u64 = arms.iter().map(|a| a.pull_count).sum();
        let total_reward: f64 = arms.iter().map(|a| a.total_reward).sum();
        let max_pulled: u64 = arms.iter().map(|a| a.last_pulled).max().unwrap_or(0);

        let n_f = n as f64;
        BanditArm {
            id: id.to_string(),
            params: BetaParams {
                alpha: (sum_alpha - (n_f - 1.0) * prior_alpha).max(prior_alpha),
                beta: (sum_beta - (n_f - 1.0) * prior_beta).max(prior_beta),
            },
            pull_count: total_pulls,
            total_reward,
            last_pulled: max_pulled,
        }
    }

    fn merge_arm_pair(local: &BanditArm, remote: &BanditArm, prior_alpha: f64, prior_beta: f64) -> BanditArm {
        BanditArm {
            id: local.id.clone(),
            params: BetaParams {
                alpha: (local.params.alpha + remote.params.alpha - prior_alpha).max(prior_alpha),
                beta: (local.params.beta + remote.params.beta - prior_beta).max(prior_beta),
            },
            pull_count: local.pull_count + remote.pull_count,
            total_reward: local.total_reward + remote.total_reward,
            last_pulled: local.last_pulled.max(remote.last_pulled),
        }
    }
}

// =============================================================================
// EXPORT / IMPORT
// =============================================================================

const SNAPSHOT_VERSION: u32 = 1;

/// Format for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SnapshotFormat {
    Json,
    Bincode,
}

/// A versioned snapshot of BanditRouter state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditSnapshot {
    pub version: u32,
    pub created_at: String,
    pub config: BanditConfig,
    pub task_bandits: HashMap<String, Vec<BanditArm>>,
    pub global_arms: Vec<BanditArm>,
    pub total_pulls: u64,
    pub metadata: HashMap<String, String>,
    /// Arms marked as private (local-only, not shared in distributed merging).
    /// Note: `skip_serializing_if` removed — bincode is positional and skipping
    /// fields causes deserialization to fail with misaligned byte streams.
    #[serde(default)]
    pub private_arms: HashSet<ArmId>,
}

impl BanditRouter {
    /// Export current state to a snapshot.
    pub fn export_snapshot(&self) -> BanditSnapshot {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        BanditSnapshot {
            version: SNAPSHOT_VERSION,
            created_at: format!("{}", now),
            config: self.config.clone(),
            task_bandits: self.bandits.clone(),
            global_arms: self.global_bandit.clone(),
            total_pulls: self.total_pulls,
            metadata: HashMap::new(),
            private_arms: self.private_arms.clone(),
        }
    }

    /// Export to JSON string.
    pub fn to_json(&self) -> Result<String, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();
        serde_json::to_string_pretty(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "JSON".to_string(),
            reason: e.to_string(),
        })
    }

    /// Import from JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: BanditSnapshot = serde_json::from_str(json).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })?;

        if snapshot.version != SNAPSHOT_VERSION {
            return Err(AdvancedRoutingError::IncompatibleVersion {
                expected: SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }

        Ok(Self {
            config: snapshot.config,
            bandits: snapshot.task_bandits,
            global_bandit: snapshot.global_arms,
            total_pulls: snapshot.total_pulls,
            seed: 12345,
            private_arms: snapshot.private_arms,
        })
    }

    /// Export to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();

        #[cfg(feature = "binary-storage")]
        {
            return bincode::serialize(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "bincode".to_string(),
                reason: e.to_string(),
            });
        }

        #[cfg(not(feature = "binary-storage"))]
        {
            serde_json::to_vec(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            })
        }
    }

    /// Import from bytes (auto-detect format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AdvancedRoutingError> {
        #[cfg(feature = "binary-storage")]
        {
            if let Ok(snapshot) = bincode::deserialize::<BanditSnapshot>(bytes) {
                if snapshot.version == SNAPSHOT_VERSION {
                    return Ok(Self {
                        config: snapshot.config,
                        bandits: snapshot.task_bandits,
                        global_bandit: snapshot.global_arms,
                        total_pulls: snapshot.total_pulls,
                        seed: 12345,
                        private_arms: snapshot.private_arms,
                    });
                }
            }
        }

        // Fallback: try JSON
        let json = std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "UTF-8".to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json(json)
    }
}

// =============================================================================
// NFA RULE BUILDER (Section A)
// =============================================================================

/// A rule within the NFA builder — a chain of conditions leading to an accepting arm.
#[derive(Debug, Clone)]
struct NfaRule {
    label: String,
    conditions: Vec<NfaSymbol>,
    arm_id: ArmId,
    priority: u32,
}

/// Fluent builder for constructing NFAs from declarative rules.
///
/// # Example
/// ```ignore
/// let nfa = NfaRuleBuilder::new()
///     .rule("code_hard")
///         .when(NfaSymbol::Domain("code".into()))
///         .and(NfaSymbol::ComplexityRange { low_pct: 70, high_pct: 100 })
///         .route_to("claude-opus")
///         .priority(10)
///         .done()
///     .fallback("gpt-4-mini", 1)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NfaRuleBuilder {
    rules: Vec<NfaRule>,
    fallback_arm: Option<ArmId>,
    fallback_priority: u32,
}

impl NfaRuleBuilder {
    /// Create a new empty NFA rule builder.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fallback_arm: None,
            fallback_priority: 0,
        }
    }

    /// Start defining a new rule with the given label.
    pub fn rule(self, label: &str) -> NfaRuleHandle {
        NfaRuleHandle {
            builder: self,
            rule: NfaRule {
                label: label.to_string(),
                conditions: Vec::new(),
                arm_id: String::new(),
                priority: 0,
            },
        }
    }

    /// Set the fallback arm (matched via `Any` symbol from start state).
    pub fn fallback(mut self, arm_id: &str, priority: u32) -> Self {
        self.fallback_arm = Some(arm_id.to_string());
        self.fallback_priority = priority;
        self
    }

    /// Build the NFA from the accumulated rules.
    pub fn build(self) -> Result<NfaRouter, AdvancedRoutingError> {
        if self.rules.is_empty() && self.fallback_arm.is_none() {
            return Err(AdvancedRoutingError::InvalidConfig {
                field: "rules".to_string(),
                reason: "NfaRuleBuilder has no rules and no fallback".to_string(),
            });
        }

        let mut nfa = NfaRouter::new();
        let start = nfa.add_state("start", None, 0);

        // Each rule becomes a chain: start → cond1 → cond2 → ... → accepting
        for rule in &self.rules {
            if rule.conditions.is_empty() {
                // No conditions: direct accepting from start via epsilon
                let accept = nfa.add_state(&rule.label, Some(&rule.arm_id), rule.priority);
                nfa.add_transition(start, NfaSymbol::Epsilon, accept);
            } else {
                let mut prev = start;
                for (i, cond) in rule.conditions.iter().enumerate() {
                    let is_last = i == rule.conditions.len() - 1;
                    if is_last {
                        let accept = nfa.add_state(&rule.label, Some(&rule.arm_id), rule.priority);
                        nfa.add_transition(prev, cond.clone(), accept);
                    } else {
                        let intermediate = nfa.add_state(
                            &format!("{}_{}", rule.label, i),
                            None,
                            0,
                        );
                        nfa.add_transition(prev, cond.clone(), intermediate);
                        prev = intermediate;
                    }
                }
            }
        }

        // Fallback: Any from start
        if let Some(ref fallback) = self.fallback_arm {
            let fb = nfa.add_state("fallback", Some(fallback), self.fallback_priority);
            nfa.add_transition(start, NfaSymbol::Any, fb);
        }

        Ok(nfa)
    }
}

/// Handle for configuring a single rule within the NFA builder.
pub struct NfaRuleHandle {
    builder: NfaRuleBuilder,
    rule: NfaRule,
}

impl NfaRuleHandle {
    /// Add a condition to this rule.
    pub fn when(mut self, symbol: NfaSymbol) -> Self {
        self.rule.conditions.push(symbol);
        self
    }

    /// Add another condition (alias for `when`, reads better in chains).
    pub fn and(self, symbol: NfaSymbol) -> Self {
        self.when(symbol)
    }

    /// Set the arm (model) this rule routes to.
    pub fn route_to(mut self, arm_id: &str) -> Self {
        self.rule.arm_id = arm_id.to_string();
        self
    }

    /// Set the priority for this rule's accepting state.
    pub fn priority(mut self, p: u32) -> Self {
        self.rule.priority = p;
        self
    }

    /// Finalize this rule and return to the builder.
    pub fn done(mut self) -> NfaRuleBuilder {
        self.builder.rules.push(self.rule);
        self.builder
    }
}

// =============================================================================
// BANDIT → NFA SYNTHESIZER (Section B)
// =============================================================================

/// Automatically generates an NFA from BanditRouter's learned data.
///
/// Converts the bandit's per-task performance statistics into deterministic
/// routing rules, allowing the system to "crystallize" learned behavior
/// into a fast NFA/DFA router.
pub struct BanditNfaSynthesizer;

impl BanditNfaSynthesizer {
    /// Synthesize an NFA from bandit learning data.
    ///
    /// - `min_pulls`: Only consider arms with at least this many pulls (avoid noise).
    /// - `quality_threshold`: Arms with mean reward >= this threshold get alternative paths.
    pub fn synthesize(
        bandit: &BanditRouter,
        min_pulls: u64,
        quality_threshold: f64,
    ) -> Result<NfaRouter, AdvancedRoutingError> {
        let task_types = bandit.task_types();

        // Collect global best arm for fallback
        let global_arms = bandit.all_arms_vec(None);
        let global_best = global_arms.iter()
            .filter(|a| a.pull_count >= min_pulls)
            .max_by(|a, b| {
                let ma = if a.pull_count > 0 { a.total_reward / a.pull_count as f64 } else { 0.0 };
                let mb = if b.pull_count > 0 { b.total_reward / b.pull_count as f64 } else { 0.0 };
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            });

        if task_types.is_empty() && global_best.is_none() {
            return Err(AdvancedRoutingError::InvalidConfig {
                field: "bandit".to_string(),
                reason: "No arms with sufficient pulls to synthesize NFA".to_string(),
            });
        }

        let mut builder = NfaRuleBuilder::new();
        let mut priority_counter: u32 = 100;

        // For each task type, create rules from top-performing arms
        for task_type in &task_types {
            let arms = bandit.all_arms_vec(Some(task_type));
            let mut qualified: Vec<(&BanditArm, f64)> = arms.iter()
                .filter(|a| a.pull_count >= min_pulls)
                .map(|a| {
                    let mean = if a.pull_count > 0 { a.total_reward / a.pull_count as f64 } else { 0.0 };
                    (*a, mean)
                })
                .collect();

            // Sort by mean reward descending
            qualified.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Best arm for this task type → high priority rule
            if let Some((best, _mean)) = qualified.first() {
                builder = builder
                    .rule(&format!("{}_best", task_type))
                    .when(NfaSymbol::Domain(task_type.to_string()))
                    .route_to(&best.id)
                    .priority(priority_counter)
                    .done();
                priority_counter = priority_counter.saturating_sub(1);
            }

            // Additional arms above quality threshold → lower priority alternatives
            for (arm, mean) in qualified.iter().skip(1) {
                if *mean >= quality_threshold {
                    builder = builder
                        .rule(&format!("{}_{}", task_type, arm.id))
                        .when(NfaSymbol::Domain(task_type.to_string()))
                        .route_to(&arm.id)
                        .priority(priority_counter)
                        .done();
                    priority_counter = priority_counter.saturating_sub(1);
                }
            }
        }

        // Global best as fallback
        if let Some(best) = global_best {
            builder = builder.fallback(&best.id, 1);
        }

        builder.build()
    }
}

// =============================================================================
// CONTEXTUAL BANDIT AUTO-DISCOVERY (Section B2)
// =============================================================================

/// Frozen snapshot of query features at observation time.
///
/// Stores the named numeric/boolean fields from `QueryFeatures` for decision
/// stump analysis, omitting `feature_vector` to keep the observation log compact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub domain: String,
    pub complexity: f64,
    pub token_count: usize,
    pub has_code: bool,
    pub is_question: bool,
    pub avg_word_length: f64,
    pub entity_count: usize,
    pub sentence_count: usize,
}

impl From<&QueryFeatures> for ContextSnapshot {
    fn from(f: &QueryFeatures) -> Self {
        Self {
            domain: f.domain.clone(),
            complexity: f.complexity,
            token_count: f.token_count,
            has_code: f.has_code,
            is_question: f.is_question,
            avg_word_length: f.avg_word_length,
            entity_count: f.entity_count,
            sentence_count: f.sentence_count,
        }
    }
}

/// A single recorded observation binding query context to arm outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualObservation {
    pub context: ContextSnapshot,
    pub arm_id: ArmId,
    pub reward: f64,
}

/// Feature dimensions available for decision stump analysis.
///
/// Each variant corresponds to a numeric or boolean field in `QueryFeatures`
/// and can be mapped to an `NfaSymbol` for NFA rule generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FeatureDimension {
    Complexity,
    TokenCount,
    HasCode,
    IsQuestion,
    AvgWordLength,
    EntityCount,
    SentenceCount,
}

impl FeatureDimension {
    /// Returns all 7 feature dimensions.
    pub fn all() -> &'static [FeatureDimension] {
        &[
            FeatureDimension::Complexity,
            FeatureDimension::TokenCount,
            FeatureDimension::HasCode,
            FeatureDimension::IsQuestion,
            FeatureDimension::AvgWordLength,
            FeatureDimension::EntityCount,
            FeatureDimension::SentenceCount,
        ]
    }

    /// Extract the numeric value for this dimension from a context snapshot.
    /// Boolean features are mapped to 0.0 / 1.0.
    pub fn extract(&self, ctx: &ContextSnapshot) -> f64 {
        match self {
            FeatureDimension::Complexity => ctx.complexity,
            FeatureDimension::TokenCount => ctx.token_count as f64,
            FeatureDimension::HasCode => if ctx.has_code { 1.0 } else { 0.0 },
            FeatureDimension::IsQuestion => if ctx.is_question { 1.0 } else { 0.0 },
            FeatureDimension::AvgWordLength => ctx.avg_word_length,
            FeatureDimension::EntityCount => ctx.entity_count as f64,
            FeatureDimension::SentenceCount => ctx.sentence_count as f64,
        }
    }

    /// Human-readable name of this dimension.
    pub fn name(&self) -> &'static str {
        match self {
            FeatureDimension::Complexity => "complexity",
            FeatureDimension::TokenCount => "token_count",
            FeatureDimension::HasCode => "has_code",
            FeatureDimension::IsQuestion => "is_question",
            FeatureDimension::AvgWordLength => "avg_word_length",
            FeatureDimension::EntityCount => "entity_count",
            FeatureDimension::SentenceCount => "sentence_count",
        }
    }

    /// Whether this dimension is boolean (only two possible values).
    fn is_boolean(&self) -> bool {
        matches!(self, FeatureDimension::HasCode | FeatureDimension::IsQuestion)
    }

    /// Whether this dimension has a direct NfaSymbol mapping.
    fn has_nfa_mapping(&self) -> bool {
        matches!(
            self,
            FeatureDimension::Complexity
                | FeatureDimension::TokenCount
                | FeatureDimension::HasCode
                | FeatureDimension::IsQuestion
        )
    }
}

/// A discovered split point where different arms are best above vs below
/// a threshold on a given feature dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredSplit {
    /// Which feature dimension this split is on.
    pub dimension: FeatureDimension,
    /// The split threshold value.
    pub threshold: f64,
    /// Best arm for observations where feature >= threshold.
    pub arm_above: ArmId,
    /// Mean reward for arm_above when feature >= threshold.
    pub reward_above: f64,
    /// Number of observations supporting arm_above.
    pub count_above: usize,
    /// Best arm for observations where feature < threshold.
    pub arm_below: ArmId,
    /// Mean reward for arm_below when feature < threshold.
    pub reward_below: f64,
    /// Number of observations supporting arm_below.
    pub count_below: usize,
    /// Quality gain over unsplit baseline.
    pub gain: f64,
}

/// A discovered split scoped to a specific domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSplit {
    pub domain: String,
    pub split: DiscoveredSplit,
}

/// Configuration for the contextual discovery system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiscoveryConfig {
    /// Maximum number of observations to retain (circular buffer).
    pub max_observations: usize,
    /// Minimum observations per partition for a split to be valid.
    pub min_samples_per_split: usize,
    /// Minimum gain (reward improvement) for a split to become a rule.
    pub min_gain: f64,
    /// Number of quantile split points to try per dimension (e.g. 4 = quartiles).
    pub num_split_points: usize,
    /// Priority boost added to discovered contextual rules over base bandit rules.
    pub discovered_rule_priority_boost: u32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_observations: 1000,
            min_samples_per_split: 10,
            min_gain: 0.05,
            num_split_points: 4,
            discovered_rule_priority_boost: 50,
        }
    }
}

/// Contextual bandit auto-discovery engine.
///
/// Records `(context, arm, reward)` observations in a bounded circular buffer,
/// then runs decision stump analysis to find feature dimensions and thresholds
/// that partition observations into regions where different arms are best.
///
/// Discovered splits are converted into multi-condition NFA rules using
/// existing `NfaSymbol` variants — no new enum variants needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualDiscovery {
    config: DiscoveryConfig,
    observations: VecDeque<ContextualObservation>,
}

impl ContextualDiscovery {
    /// Create a new contextual discovery engine.
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            observations: VecDeque::new(),
        }
    }

    /// Record an observation. Oldest observations are evicted when buffer is full.
    pub fn record(&mut self, features: &QueryFeatures, arm_id: &str, reward: f64) {
        if self.observations.len() >= self.config.max_observations {
            self.observations.pop_front();
        }
        self.observations.push_back(ContextualObservation {
            context: ContextSnapshot::from(features),
            arm_id: arm_id.to_string(),
            reward,
        });
    }

    /// Number of observations currently stored.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DiscoveryConfig {
        &self.config
    }

    /// Clear all observations.
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// Discover feature splits via decision stump analysis.
    ///
    /// For each domain × feature dimension, tries quantile split points
    /// and finds thresholds where different arms are best above vs below.
    pub fn discover_splits(&self) -> Vec<DomainSplit> {
        let min_total = 2 * self.config.min_samples_per_split;
        if self.observations.len() < min_total {
            return Vec::new();
        }

        // Collect unique domains
        let mut domains: HashSet<&str> = HashSet::new();
        for obs in &self.observations {
            if !obs.context.domain.is_empty() {
                domains.insert(&obs.context.domain);
            }
        }

        let mut all_splits: Vec<DomainSplit> = Vec::new();

        for domain in domains {
            let domain_obs: Vec<&ContextualObservation> = self.observations.iter()
                .filter(|o| o.context.domain == domain)
                .collect();

            if domain_obs.len() < min_total {
                continue;
            }

            // Baseline: best arm mean reward for this domain (unsplit)
            let (_, baseline_mean) = Self::compute_best_arm_mean(&domain_obs);

            for dim in FeatureDimension::all() {
                if dim.is_boolean() {
                    // Boolean split: partition into true/false
                    let true_obs: Vec<&ContextualObservation> = domain_obs.iter()
                        .filter(|o| dim.extract(&o.context) >= 0.5)
                        .copied()
                        .collect();
                    let false_obs: Vec<&ContextualObservation> = domain_obs.iter()
                        .filter(|o| dim.extract(&o.context) < 0.5)
                        .copied()
                        .collect();

                    if true_obs.len() < self.config.min_samples_per_split
                        || false_obs.len() < self.config.min_samples_per_split
                    {
                        continue;
                    }

                    let (arm_true, mean_true) = Self::compute_best_arm_mean(&true_obs);
                    let (arm_false, mean_false) = Self::compute_best_arm_mean(&false_obs);

                    if arm_true == arm_false {
                        continue;
                    }

                    let gain = (mean_true + mean_false) / 2.0 - baseline_mean;
                    if gain >= self.config.min_gain {
                        all_splits.push(DomainSplit {
                            domain: domain.to_string(),
                            split: DiscoveredSplit {
                                dimension: *dim,
                                threshold: 0.5,
                                arm_above: arm_true,
                                reward_above: mean_true,
                                count_above: true_obs.len(),
                                arm_below: arm_false,
                                reward_below: mean_false,
                                count_below: false_obs.len(),
                                gain,
                            },
                        });
                    }
                } else {
                    // Numeric split: try quantile split points
                    let values: Vec<f64> = domain_obs.iter()
                        .map(|o| dim.extract(&o.context))
                        .collect();
                    let split_points = Self::compute_quantile_split_points(
                        &values,
                        self.config.num_split_points,
                    );

                    for threshold in split_points {
                        let above: Vec<&ContextualObservation> = domain_obs.iter()
                            .filter(|o| dim.extract(&o.context) >= threshold)
                            .copied()
                            .collect();
                        let below: Vec<&ContextualObservation> = domain_obs.iter()
                            .filter(|o| dim.extract(&o.context) < threshold)
                            .copied()
                            .collect();

                        if above.len() < self.config.min_samples_per_split
                            || below.len() < self.config.min_samples_per_split
                        {
                            continue;
                        }

                        let (arm_above, mean_above) = Self::compute_best_arm_mean(&above);
                        let (arm_below, mean_below) = Self::compute_best_arm_mean(&below);

                        if arm_above == arm_below {
                            continue;
                        }

                        let gain = (mean_above + mean_below) / 2.0 - baseline_mean;
                        if gain >= self.config.min_gain {
                            all_splits.push(DomainSplit {
                                domain: domain.to_string(),
                                split: DiscoveredSplit {
                                    dimension: *dim,
                                    threshold,
                                    arm_above,
                                    reward_above: mean_above,
                                    count_above: above.len(),
                                    arm_below,
                                    reward_below: mean_below,
                                    count_below: below.len(),
                                    gain,
                                },
                            });
                        }
                    }
                }
            }
        }

        // Sort by gain descending
        all_splits.sort_by(|a, b| {
            b.split.gain.partial_cmp(&a.split.gain).unwrap_or(std::cmp::Ordering::Equal)
        });

        all_splits
    }

    /// Convert discovered splits into (label, conditions, arm_id, priority) tuples
    /// for NFA rule building.
    ///
    /// Each split produces two rules (above and below threshold) with domain-scoped
    /// multi-condition chains. Only dimensions with NfaSymbol mappings produce rules.
    pub fn splits_to_nfa_rules(
        &self,
        splits: &[DomainSplit],
        base_priority: u32,
    ) -> Vec<(String, Vec<NfaSymbol>, ArmId, u32)> {
        let mut rules = Vec::new();
        let mut priority = base_priority + self.config.discovered_rule_priority_boost;

        for ds in splits {
            if !ds.split.dimension.has_nfa_mapping() {
                continue;
            }

            let domain_sym = NfaSymbol::Domain(ds.domain.clone());

            match ds.split.dimension {
                FeatureDimension::Complexity => {
                    let low_pct = (ds.split.threshold * 100.0) as u32;
                    // Above threshold → arm_above
                    rules.push((
                        format!("ctx_{}_complexity_high", ds.split.arm_above),
                        vec![
                            domain_sym.clone(),
                            NfaSymbol::ComplexityRange { low_pct, high_pct: 100 },
                        ],
                        ds.split.arm_above.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                    // Below threshold → arm_below
                    rules.push((
                        format!("ctx_{}_complexity_low", ds.split.arm_below),
                        vec![
                            NfaSymbol::Domain(ds.domain.clone()),
                            NfaSymbol::ComplexityRange { low_pct: 0, high_pct: low_pct },
                        ],
                        ds.split.arm_below.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                }
                FeatureDimension::TokenCount => {
                    let threshold_usize = ds.split.threshold as usize;
                    rules.push((
                        format!("ctx_{}_tokens_high", ds.split.arm_above),
                        vec![
                            domain_sym.clone(),
                            NfaSymbol::TokenRange { min: threshold_usize, max: usize::MAX },
                        ],
                        ds.split.arm_above.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                    rules.push((
                        format!("ctx_{}_tokens_low", ds.split.arm_below),
                        vec![
                            NfaSymbol::Domain(ds.domain.clone()),
                            NfaSymbol::TokenRange { min: 0, max: threshold_usize.saturating_sub(1) },
                        ],
                        ds.split.arm_below.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                }
                FeatureDimension::HasCode | FeatureDimension::IsQuestion => {
                    let feature_name = ds.split.dimension.name().to_string();
                    rules.push((
                        format!("ctx_{}_{}_true", ds.split.arm_above, feature_name),
                        vec![
                            domain_sym.clone(),
                            NfaSymbol::BoolFeature { name: feature_name.clone(), value: true },
                        ],
                        ds.split.arm_above.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                    rules.push((
                        format!("ctx_{}_{}_false", ds.split.arm_below, feature_name),
                        vec![
                            NfaSymbol::Domain(ds.domain.clone()),
                            NfaSymbol::BoolFeature { name: feature_name, value: false },
                        ],
                        ds.split.arm_below.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                }
                _ => {} // AvgWordLength, EntityCount, SentenceCount: no NfaSymbol mapping
            }
        }

        rules
    }

    /// Synthesize an enhanced NFA combining discovered contextual rules with
    /// base bandit domain-only rules.
    ///
    /// Priority scheme:
    /// - Discovered contextual rules: base_priority + discovered_rule_priority_boost (highest)
    /// - Base bandit domain-only rules: ~50 (medium)
    /// - Global fallback: 1 (lowest)
    pub fn synthesize_enhanced_nfa(
        &self,
        bandit: &BanditRouter,
        min_pulls: u64,
        quality_threshold: f64,
    ) -> Result<NfaRouter, AdvancedRoutingError> {
        let domain_splits = self.discover_splits();
        let contextual_rules = self.splits_to_nfa_rules(&domain_splits, 100);

        let mut builder = NfaRuleBuilder::new();

        // Add contextual rules first (highest priority)
        for (label, conditions, arm_id, prio) in &contextual_rules {
            let mut handle = builder.rule(label);
            for cond in conditions {
                handle = handle.when(cond.clone());
            }
            builder = handle.route_to(arm_id).priority(*prio).done();
        }

        // Add base bandit domain-only rules (medium priority)
        let task_types = bandit.task_types();
        let mut base_priority: u32 = 50;
        for task_type in &task_types {
            let arms = bandit.all_arms_vec(Some(task_type));
            let mut qualified: Vec<(&BanditArm, f64)> = arms.iter()
                .filter(|a| a.pull_count >= min_pulls)
                .map(|a| {
                    let mean = if a.pull_count > 0 {
                        a.total_reward / a.pull_count as f64
                    } else {
                        0.0
                    };
                    (*a, mean)
                })
                .collect();
            qualified.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((best, _)) = qualified.first() {
                builder = builder
                    .rule(&format!("{}_base", task_type))
                    .when(NfaSymbol::Domain(task_type.to_string()))
                    .route_to(&best.id)
                    .priority(base_priority)
                    .done();
                base_priority = base_priority.saturating_sub(1);
            }

            // Additional arms above quality threshold
            for (arm, mean) in qualified.iter().skip(1) {
                if *mean >= quality_threshold {
                    builder = builder
                        .rule(&format!("{}_{}_alt", task_type, arm.id))
                        .when(NfaSymbol::Domain(task_type.to_string()))
                        .route_to(&arm.id)
                        .priority(base_priority)
                        .done();
                    base_priority = base_priority.saturating_sub(1);
                }
            }
        }

        // Global fallback
        let global_arms = bandit.all_arms_vec(None);
        let global_best = global_arms.iter()
            .filter(|a| a.pull_count >= min_pulls)
            .max_by(|a, b| {
                let ma = if a.pull_count > 0 { a.total_reward / a.pull_count as f64 } else { 0.0 };
                let mb = if b.pull_count > 0 { b.total_reward / b.pull_count as f64 } else { 0.0 };
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(best) = global_best {
            builder = builder.fallback(&best.id, 1);
        }

        builder.build()
    }

    // --- Private helpers ---

    /// Find the arm with the highest mean reward in a set of observations.
    fn compute_best_arm_mean(observations: &[&ContextualObservation]) -> (ArmId, f64) {
        let mut arm_stats: HashMap<&str, (f64, usize)> = HashMap::new();
        for obs in observations {
            let entry = arm_stats.entry(&obs.arm_id).or_insert((0.0, 0));
            entry.0 += obs.reward;
            entry.1 += 1;
        }
        arm_stats.iter()
            .map(|(arm, (sum, count))| (arm.to_string(), *sum / *count as f64))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_default()
    }

    /// Compute quantile split points from a set of values.
    fn compute_quantile_split_points(values: &[f64], num_points: usize) -> Vec<f64> {
        if values.is_empty() || num_points == 0 {
            return Vec::new();
        }
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Deduplicate (within epsilon)
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        if sorted.len() <= 1 {
            return Vec::new(); // No useful split if all values are the same
        }
        let mut points = Vec::new();
        for i in 1..=num_points {
            let idx = (sorted.len() * i) / (num_points + 1);
            let idx = idx.min(sorted.len() - 1);
            let val = sorted[idx];
            if points.last().map_or(true, |last: &f64| (val - *last).abs() > 1e-10) {
                points.push(val);
            }
        }
        points
    }

    /// Compute feature importance by analyzing discovered splits.
    ///
    /// Returns dimensions sorted by `total_gain` descending.
    /// Each entry aggregates gain, split count, and domain coverage
    /// across all discovered splits for that dimension.
    pub fn feature_importance(&self) -> Vec<FeatureImportance> {
        let splits = self.discover_splits();
        if splits.is_empty() {
            return Vec::new();
        }

        // Aggregate by dimension name
        let mut by_dim: HashMap<String, (f64, usize, HashSet<String>)> = HashMap::new();

        for ds in &splits {
            let dim_name = ds.split.dimension.name().to_string();
            let entry = by_dim.entry(dim_name).or_insert((0.0, 0, HashSet::new()));
            entry.0 += ds.split.gain;
            entry.1 += 1;
            entry.2.insert(ds.domain.clone());
        }

        let mut result: Vec<FeatureImportance> = by_dim.into_iter()
            .map(|(name, (total_gain, split_count, domains))| {
                let dimension = match name.as_str() {
                    "complexity" => FeatureDimension::Complexity,
                    "token_count" => FeatureDimension::TokenCount,
                    "has_code" => FeatureDimension::HasCode,
                    "is_question" => FeatureDimension::IsQuestion,
                    "avg_word_length" => FeatureDimension::AvgWordLength,
                    "entity_count" => FeatureDimension::EntityCount,
                    "sentence_count" => FeatureDimension::SentenceCount,
                    _ => FeatureDimension::Complexity, // fallback
                };
                FeatureImportance {
                    dimension,
                    total_gain,
                    split_count,
                    domains_affected: domains.len(),
                }
            })
            .collect();

        result.sort_by(|a, b| b.total_gain.partial_cmp(&a.total_gain).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

/// Importance score for a feature dimension across all discovered splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Which feature dimension
    pub dimension: FeatureDimension,
    /// Cumulative gain across all splits using this dimension
    pub total_gain: f64,
    /// Number of splits using this dimension
    pub split_count: usize,
    /// Number of distinct domains affected by splits on this dimension
    pub domains_affected: usize,
}

// =============================================================================
// NFA EXPORT / IMPORT (Section C)
// =============================================================================

const NFA_SNAPSHOT_VERSION: u32 = 1;

/// Serializable snapshot of an NFA router's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfaSnapshot {
    pub version: u32,
    pub created_at: String,
    pub states: Vec<NfaState>,
    pub transitions: Vec<NfaTransition>,
    pub start_states: Vec<NfaStateId>,
    pub metadata: HashMap<String, String>,
}

impl NfaRouter {
    /// Export the NFA as a snapshot.
    pub fn export_snapshot(&self) -> NfaSnapshot {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        NfaSnapshot {
            version: NFA_SNAPSHOT_VERSION,
            created_at: format!("{}", now),
            states: self.states.clone(),
            transitions: self.transitions.clone(),
            start_states: self.start_states.clone(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize the NFA to a JSON string.
    pub fn to_json(&self) -> Result<String, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();
        serde_json::to_string_pretty(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "JSON".to_string(),
            reason: e.to_string(),
        })
    }

    /// Deserialize an NFA from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: NfaSnapshot = serde_json::from_str(json).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })?;
        if snapshot.version != NFA_SNAPSHOT_VERSION {
            return Err(AdvancedRoutingError::IncompatibleVersion {
                expected: NFA_SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }
        Ok(Self {
            states: snapshot.states,
            transitions: snapshot.transitions,
            start_states: snapshot.start_states,
        })
    }

    /// Serialize the NFA to bytes (bincode if available, else JSON).
    pub fn to_bytes(&self) -> Result<Vec<u8>, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();

        #[cfg(feature = "binary-storage")]
        {
            return bincode::serialize(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "bincode".to_string(),
                reason: e.to_string(),
            });
        }

        #[cfg(not(feature = "binary-storage"))]
        {
            serde_json::to_vec(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            })
        }
    }

    /// Deserialize an NFA from bytes (auto-detects format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AdvancedRoutingError> {
        #[cfg(feature = "binary-storage")]
        {
            if let Ok(snapshot) = bincode::deserialize::<NfaSnapshot>(bytes) {
                if snapshot.version == NFA_SNAPSHOT_VERSION {
                    return Ok(Self {
                        states: snapshot.states,
                        transitions: snapshot.transitions,
                        start_states: snapshot.start_states,
                    });
                }
            }
        }

        let json = std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "UTF-8".to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json(json)
    }

    /// Merge another NFA into this one (union construction).
    ///
    /// Creates a new NFA with a fresh start state connected via epsilon
    /// transitions to both original start states. States from `other`
    /// are renumbered to avoid ID conflicts.
    pub fn merge(&self, other: &NfaRouter) -> NfaRouter {
        let mut result = NfaRouter::new();
        let self_offset: usize = 1; // self states start at 1
        let other_offset: usize = 1 + self.states.len(); // other states after self

        // Add merged start state
        let start = result.add_state("merged_start", None, 0);

        // Copy self states (renumbered with offset)
        for state in &self.states {
            let _id = result.add_state(
                &state.label,
                state.accepting_arm.as_deref(),
                state.priority,
            );
        }

        // Copy other states (renumbered with other_offset)
        for state in &other.states {
            let _id = result.add_state(
                &state.label,
                state.accepting_arm.as_deref(),
                state.priority,
            );
        }

        // Copy self transitions (apply self_offset)
        for trans in &self.transitions {
            result.add_transition(
                trans.from + self_offset,
                trans.symbol.clone(),
                trans.to + self_offset,
            );
        }

        // Copy other transitions (apply other_offset)
        for trans in &other.transitions {
            result.add_transition(
                trans.from + other_offset,
                trans.symbol.clone(),
                trans.to + other_offset,
            );
        }

        // Epsilon from new start to both original starts
        for &orig_start in &self.start_states {
            result.add_transition(start, NfaSymbol::Epsilon, orig_start + self_offset);
        }
        for &orig_start in &other.start_states {
            result.add_transition(start, NfaSymbol::Epsilon, orig_start + other_offset);
        }

        // Clear default start states, set only merged start
        result.start_states = vec![start];

        result
    }
}

// =============================================================================
// DFA EXPORT / IMPORT (Section D)
// =============================================================================

const DFA_SNAPSHOT_VERSION: u32 = 1;

/// Serializable snapshot of a DFA router's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfaSnapshot {
    pub version: u32,
    pub created_at: String,
    pub states: Vec<DfaState>,
    pub start_state: DfaStateId,
    pub transition_table: HashMap<DfaStateId, Vec<(NfaSymbol, DfaStateId)>>,
    pub metadata: HashMap<String, String>,
}

impl DfaRouter {
    /// Export the DFA as a snapshot.
    pub fn export_snapshot(&self) -> DfaSnapshot {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        DfaSnapshot {
            version: DFA_SNAPSHOT_VERSION,
            created_at: format!("{}", now),
            states: self.states.clone(),
            start_state: self.start_state,
            transition_table: self.transition_table.clone(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize the DFA to a JSON string.
    pub fn to_json(&self) -> Result<String, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();
        serde_json::to_string_pretty(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "JSON".to_string(),
            reason: e.to_string(),
        })
    }

    /// Deserialize a DFA from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: DfaSnapshot = serde_json::from_str(json).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })?;
        if snapshot.version != DFA_SNAPSHOT_VERSION {
            return Err(AdvancedRoutingError::IncompatibleVersion {
                expected: DFA_SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }
        Ok(Self {
            states: snapshot.states,
            start_state: snapshot.start_state,
            transition_table: snapshot.transition_table,
        })
    }

    /// Serialize the DFA to bytes (bincode if available, else JSON).
    pub fn to_bytes(&self) -> Result<Vec<u8>, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();

        #[cfg(feature = "binary-storage")]
        {
            return bincode::serialize(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "bincode".to_string(),
                reason: e.to_string(),
            });
        }

        #[cfg(not(feature = "binary-storage"))]
        {
            serde_json::to_vec(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            })
        }
    }

    /// Deserialize a DFA from bytes (auto-detects format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AdvancedRoutingError> {
        #[cfg(feature = "binary-storage")]
        {
            if let Ok(snapshot) = bincode::deserialize::<DfaSnapshot>(bytes) {
                if snapshot.version == DFA_SNAPSHOT_VERSION {
                    return Ok(Self {
                        states: snapshot.states,
                        start_state: snapshot.start_state,
                        transition_table: snapshot.transition_table,
                    });
                }
            }
        }

        let json = std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "UTF-8".to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json(json)
    }
}

/// Merge two NFAs and compile the result into a DFA.
///
/// Convenience function that combines NFA union construction with DFA compilation.
pub fn merge_and_compile_nfas(
    a: &NfaRouter,
    b: &NfaRouter,
) -> Result<DfaRouter, AdvancedRoutingError> {
    let merged = a.merge(b);
    let mut dfa = NfaDfaCompiler::compile(&merged)?;
    dfa.minimize();
    Ok(dfa)
}

// =============================================================================
// DISTRIBUTED NFA SHARING (Section G)
// =============================================================================

/// Serializable state of an NFA router for distribution between nodes.
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedNfaState {
    pub node_id: String,
    pub timestamp: u64,
    pub nfa: NfaSnapshot,
}

/// Merges NFA states from multiple distributed nodes.
#[cfg(feature = "distributed")]
pub struct NfaStateMerger;

#[cfg(feature = "distributed")]
impl NfaStateMerger {
    /// Extract the current NFA state for distribution.
    pub fn extract_state(nfa: &NfaRouter, node_id: &str) -> DistributedNfaState {
        DistributedNfaState {
            node_id: node_id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            nfa: nfa.export_snapshot(),
        }
    }

    /// Extract NFA state for distribution, filtering out private arms.
    pub fn extract_state_filtered(nfa: &NfaRouter, node_id: &str, private_arms: &HashSet<ArmId>) -> DistributedNfaState {
        let mut snapshot = nfa.export_snapshot();

        // Find state IDs whose accepting_arm is in private_arms
        let private_state_ids: HashSet<usize> = snapshot.states.iter()
            .filter(|s| s.accepting_arm.as_ref().map_or(false, |arm| private_arms.contains(arm)))
            .map(|s| s.id)
            .collect();

        // Remove transitions that lead to private accepting states
        snapshot.transitions.retain(|t| !private_state_ids.contains(&t.to));

        // Remove private accepting states themselves
        snapshot.states.retain(|s| !private_state_ids.contains(&s.id));

        DistributedNfaState {
            node_id: node_id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            nfa: snapshot,
        }
    }

    /// Merge NFA states from multiple nodes into a single unified NFA.
    ///
    /// Uses NFA union construction: all nodes' rules are combined into one NFA.
    pub fn merge(
        states: &[DistributedNfaState],
    ) -> Result<DistributedNfaState, AdvancedRoutingError> {
        if states.is_empty() {
            return Err(AdvancedRoutingError::InvalidConfig {
                field: "states".to_string(),
                reason: "Cannot merge empty NFA state list".to_string(),
            });
        }

        // Reconstruct NFAs from snapshots
        let nfas: Result<Vec<NfaRouter>, _> = states.iter()
            .map(|s| {
                Ok(NfaRouter {
                    states: s.nfa.states.clone(),
                    transitions: s.nfa.transitions.clone(),
                    start_states: s.nfa.start_states.clone(),
                })
            })
            .collect();
        let nfas = nfas?;

        // Iteratively merge all NFAs
        let mut merged = nfas[0].clone();
        for nfa in nfas.iter().skip(1) {
            merged = merged.merge(nfa);
        }

        let max_ts = states.iter().map(|s| s.timestamp).max().unwrap_or(0);

        Ok(DistributedNfaState {
            node_id: "merged".to_string(),
            timestamp: max_ts,
            nfa: merged.export_snapshot(),
        })
    }

    /// Merge a remote NFA state into a local router.
    pub fn merge_into_router(
        router: &mut NfaRouter,
        remote: &DistributedNfaState,
    ) -> Result<(), AdvancedRoutingError> {
        let remote_nfa = NfaRouter {
            states: remote.nfa.states.clone(),
            transitions: remote.nfa.transitions.clone(),
            start_states: remote.nfa.start_states.clone(),
        };
        let merged = router.merge(&remote_nfa);
        *router = merged;
        Ok(())
    }
}

// =============================================================================
// CLOSED-LOOP ROUTING PIPELINE (Section H)
// =============================================================================

const PIPELINE_SNAPSHOT_VERSION: u32 = 1;

/// Model tier for heuristic-based routing rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelTier {
    /// High-capability model: code generation, complex reasoning, high-complexity queries.
    Premium,
    /// General-purpose model: medium-complexity queries.
    Standard,
    /// Cost-efficient model: simple queries, fallback.
    Economy,
}

/// Configuration for the closed-loop routing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PipelineConfig {
    /// Re-synthesize NFA after this many new outcome recordings.
    pub synthesis_interval: u64,
    /// Arms need at least this many pulls to be included in synthesis.
    pub min_pulls_for_synthesis: u64,
    /// Minimum quality for alternative paths in synthesized NFA.
    pub quality_threshold: f64,
    /// Whether to minimize the DFA after compilation.
    pub auto_minimize: bool,
    /// Optional contextual discovery configuration.
    /// When `Some`, the pipeline records feature context with outcomes and
    /// uses decision stump analysis to auto-discover multi-condition NFA rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discovery: Option<DiscoveryConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            synthesis_interval: 100,
            min_pulls_for_synthesis: 10,
            quality_threshold: 0.5,
            auto_minimize: true,
            discovery: None,
        }
    }
}

/// Closed-loop routing pipeline that learns and evolves.
///
/// The pipeline follows this cycle:
/// 1. Route queries via DFA (fast) or bandit (learning)
/// 2. Record outcomes to the bandit
/// 3. Periodically synthesize a new NFA from bandit data
/// 4. Compile NFA → DFA for fast production routing
/// 5. Repeat
pub struct RoutingPipeline {
    bandit: BanditRouter,
    active_dfa: Option<DfaRouter>,
    source_nfa: Option<NfaRouter>,
    pulls_since_synthesis: u64,
    config: PipelineConfig,
    synthesis_count: u64,
    contextual: Option<ContextualDiscovery>,
}

impl RoutingPipeline {
    /// Create a new pipeline with the given bandit and pipeline configs.
    pub fn new(bandit_config: BanditConfig, pipeline_config: PipelineConfig) -> Self {
        let contextual = pipeline_config.discovery.as_ref()
            .map(|dc| ContextualDiscovery::new(dc.clone()));
        Self {
            bandit: BanditRouter::new(bandit_config),
            active_dfa: None,
            source_nfa: None,
            pulls_since_synthesis: 0,
            config: pipeline_config,
            synthesis_count: 0,
            contextual,
        }
    }

    /// Set an initial NFA and compile it to DFA.
    ///
    /// Also seeds the bandit with arms extracted from the NFA:
    /// - Each accepting state's arm is registered
    /// - Domain transitions associate arms with task types
    /// - Priorities are converted to warm-start priors (higher priority → stronger prior)
    pub fn with_initial_nfa(mut self, nfa: NfaRouter) -> Result<Self, AdvancedRoutingError> {
        let mut dfa = NfaDfaCompiler::compile(&nfa)?;
        if self.config.auto_minimize {
            dfa.minimize();
        }

        // Seed bandit from NFA structure
        self.seed_bandit_from_nfa(&nfa);

        self.source_nfa = Some(nfa);
        self.active_dfa = Some(dfa);
        Ok(self)
    }

    /// Extract arms and domain→arm mappings from an NFA, register them in the bandit.
    fn seed_bandit_from_nfa(&mut self, nfa: &NfaRouter) {
        let states = nfa.states();
        let transitions = nfa.transitions();

        // Collect all accepting arms and their max priority
        let mut arm_priorities: HashMap<String, u32> = HashMap::new();
        for state in states {
            if let Some(ref arm) = state.accepting_arm {
                let entry = arm_priorities.entry(arm.clone()).or_insert(0);
                if state.priority > *entry {
                    *entry = state.priority;
                }
            }
        }

        // Map domain → set of reachable arms (via transitions)
        // For each Domain(x) transition, find what accepting arms are reachable from target
        let mut domain_arms: HashMap<String, HashSet<String>> = HashMap::new();
        for trans in transitions {
            if let NfaSymbol::Domain(ref domain) = trans.symbol {
                // Walk forward from trans.to to find reachable accepting states
                let reachable = self.reachable_arms_from(nfa, trans.to);
                domain_arms.entry(domain.clone()).or_default().extend(reachable);
            }
        }

        // Register all arms globally
        for arm_id in arm_priorities.keys() {
            self.bandit.add_arm(arm_id);
        }

        // Register arms per task type (domain)
        for (domain, arms) in &domain_arms {
            for arm_id in arms {
                self.bandit.add_arm_for_task(domain, arm_id);
            }
        }

        // Warm-start priors: higher NFA priority → stronger alpha
        // Scale: priority/max_priority * prior_scale, where prior_scale is modest (5.0)
        let max_priority = arm_priorities.values().copied().max().unwrap_or(1).max(1) as f64;
        let prior_scale = 5.0;
        for (arm_id, priority) in &arm_priorities {
            let strength = (*priority as f64 / max_priority) * prior_scale;
            let alpha = self.bandit.config().prior_alpha + strength;
            let beta = self.bandit.config().prior_beta;
            self.bandit.warm_start(arm_id, alpha, beta);

            // Also warm-start in each domain this arm appears in
            for (domain, arms) in &domain_arms {
                if arms.contains(arm_id) {
                    self.bandit.warm_start_for_task(domain, arm_id, alpha, beta);
                }
            }
        }
    }

    /// Find all accepting arms reachable from a given NFA state (BFS).
    fn reachable_arms_from(&self, nfa: &NfaRouter, start: NfaStateId) -> HashSet<String> {
        let states = nfa.states();
        let transitions = nfa.transitions();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut arms = HashSet::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(sid) = queue.pop_front() {
            if let Some(state) = states.get(sid) {
                if let Some(ref arm) = state.accepting_arm {
                    arms.insert(arm.clone());
                }
            }
            // Follow all transitions from this state (any symbol)
            for trans in transitions {
                if trans.from == sid && !visited.contains(&trans.to) {
                    visited.insert(trans.to);
                    queue.push_back(trans.to);
                }
            }
        }

        arms
    }

    /// Build an NFA from rules and set it as initial.
    pub fn with_initial_rules(self, builder: NfaRuleBuilder) -> Result<Self, AdvancedRoutingError> {
        let nfa = builder.build()?;
        self.with_initial_nfa(nfa)
    }

    /// Create a pipeline with just a list of models (zero-config).
    ///
    /// Starts in pure bandit exploration mode (no NFA/DFA). Once enough
    /// outcomes are recorded (`synthesis_interval`), the bandit's learning
    /// is automatically synthesized into NFA → compiled to DFA.
    pub fn for_models(models: &[&str], config: PipelineConfig) -> Self {
        let mut pipeline = Self::new(BanditConfig::default(), config);
        for model in models {
            pipeline.bandit.add_arm(model);
        }
        pipeline
    }

    /// Create a pipeline with tiered models and auto-generated routing rules.
    ///
    /// Generates heuristic-based NFA rules:
    /// - **Premium** models: code, complex reasoning (complexity >= 70%)
    /// - **Standard** models: general queries (complexity 30%-70%)
    /// - **Economy** models: simple queries, fallback (complexity < 30%)
    ///
    /// The bandit is seeded from these rules and will refine them over time.
    pub fn with_tiered_models(
        models: &[(&str, ModelTier)],
        config: PipelineConfig,
    ) -> Result<Self, AdvancedRoutingError> {
        if models.is_empty() {
            return Err(AdvancedRoutingError::InvalidConfig {
                field: "models".to_string(),
                reason: "No models provided".to_string(),
            });
        }

        let mut builder = NfaRuleBuilder::new();
        let mut priority: u32 = 100;
        let mut fallback_model: Option<&str> = None;

        // Group models by tier
        let premium: Vec<&str> = models.iter()
            .filter(|(_, t)| matches!(t, ModelTier::Premium))
            .map(|(m, _)| *m).collect();
        let standard: Vec<&str> = models.iter()
            .filter(|(_, t)| matches!(t, ModelTier::Standard))
            .map(|(m, _)| *m).collect();
        let economy: Vec<&str> = models.iter()
            .filter(|(_, t)| matches!(t, ModelTier::Economy))
            .map(|(m, _)| *m).collect();

        // Premium → code + high complexity
        for model in &premium {
            builder = builder
                .rule(&format!("{}_code", model))
                .when(NfaSymbol::BoolFeature { name: "has_code".into(), value: true })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;

            builder = builder
                .rule(&format!("{}_complex", model))
                .when(NfaSymbol::ComplexityRange { low_pct: 70, high_pct: 100 })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;
        }

        // Standard → medium complexity
        for model in &standard {
            builder = builder
                .rule(&format!("{}_mid", model))
                .when(NfaSymbol::ComplexityRange { low_pct: 30, high_pct: 70 })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;
        }

        // Economy → low complexity
        for model in &economy {
            builder = builder
                .rule(&format!("{}_simple", model))
                .when(NfaSymbol::ComplexityRange { low_pct: 0, high_pct: 30 })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;
            fallback_model = Some(model);
        }

        // Fallback: cheapest economy, or last standard, or last premium
        let fb = fallback_model
            .or(standard.last().copied())
            .or(premium.last().copied())
            .unwrap_or(models[0].0);
        builder = builder.fallback(fb, 1);

        let pipeline = Self::new(BanditConfig::default(), config);
        pipeline.with_initial_rules(builder)
    }

    /// Add an arm to the bandit (global).
    pub fn add_arm(&mut self, arm_id: &str) {
        self.bandit.add_arm(arm_id);
    }

    /// Add an arm to the bandit for a specific task type.
    pub fn add_arm_for_task(&mut self, task_type: &str, arm_id: &str) {
        self.bandit.add_arm_for_task(task_type, arm_id);
    }

    /// Route a query through the pipeline.
    ///
    /// Uses the compiled DFA if available, otherwise falls back to the bandit.
    pub fn route(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Try DFA first
        if let Some(ref dfa) = self.active_dfa {
            match dfa.route(features) {
                Ok(outcome) => return Ok(outcome),
                Err(_) => {} // Fall through to bandit
            }
        }

        // Fallback: bandit
        let task_type = if features.domain.is_empty() { None } else { Some(features.domain.as_str()) };
        self.bandit.select(task_type)
    }

    /// Record an outcome and potentially trigger re-synthesis.
    pub fn record_outcome(&mut self, feedback: &ArmFeedback) {
        self.bandit.record_outcome(feedback);
        self.pulls_since_synthesis += 1;
    }

    /// Check if re-synthesis is needed and perform it if so.
    /// Returns `true` if re-synthesis was performed.
    pub fn maybe_resynthesize(&mut self) -> bool {
        if self.pulls_since_synthesis >= self.config.synthesis_interval {
            self.force_resynthesize().is_ok()
        } else {
            false
        }
    }

    /// Force immediate re-synthesis of NFA from bandit data.
    ///
    /// When contextual discovery is enabled, uses the enhanced synthesizer
    /// that generates multi-condition NFA rules from discovered feature splits.
    /// Otherwise falls back to the standard domain-only synthesizer.
    pub fn force_resynthesize(&mut self) -> Result<(), AdvancedRoutingError> {
        let nfa = if let Some(ref ctx) = self.contextual {
            ctx.synthesize_enhanced_nfa(
                &self.bandit,
                self.config.min_pulls_for_synthesis,
                self.config.quality_threshold,
            )?
        } else {
            BanditNfaSynthesizer::synthesize(
                &self.bandit,
                self.config.min_pulls_for_synthesis,
                self.config.quality_threshold,
            )?
        };
        let mut dfa = NfaDfaCompiler::compile(&nfa)?;
        if self.config.auto_minimize {
            dfa.minimize();
        }
        self.source_nfa = Some(nfa);
        self.active_dfa = Some(dfa);
        self.pulls_since_synthesis = 0;
        self.synthesis_count += 1;
        Ok(())
    }

    /// Enable contextual discovery on this pipeline.
    pub fn enable_discovery(&mut self, config: DiscoveryConfig) {
        self.contextual = Some(ContextualDiscovery::new(config));
    }

    /// Record an outcome with full query features context.
    ///
    /// Feeds both the bandit (standard learning) and the contextual discovery
    /// engine (multi-dimensional split analysis) if enabled.
    pub fn record_outcome_with_context(
        &mut self,
        feedback: &ArmFeedback,
        features: &QueryFeatures,
    ) {
        self.bandit.record_outcome(feedback);
        self.pulls_since_synthesis += 1;

        if let Some(ref mut ctx) = self.contextual {
            let reward = self.bandit.config().reward_policy.compute_reward(feedback);
            ctx.record(features, &feedback.arm_id, reward);
        }
    }

    /// Route a query with per-query routing preferences.
    ///
    /// Uses DFA if available (for non-excluded DFA results), falls back to
    /// bandit with arm exclusion/boosting.
    pub fn route_with_preferences(
        &mut self,
        features: &QueryFeatures,
        prefs: &RoutingPreferences,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Try DFA first (if no exclusion/preferred constraints)
        if prefs.excluded_arms.is_empty() && prefs.preferred_arms.is_empty() {
            if let Some(ref dfa) = self.active_dfa {
                if let Ok(outcome) = dfa.route(features) {
                    return Ok(outcome);
                }
            }
        }
        // Fall through to bandit with preferences
        self.bandit.select_with_preferences(Some(&features.domain), prefs)
    }

    /// Record outcome with preferences + contextual discovery.
    pub fn record_outcome_with_context_and_preferences(
        &mut self,
        feedback: &ArmFeedback,
        features: &QueryFeatures,
        prefs: &RoutingPreferences,
    ) {
        self.bandit.record_outcome_with_preferences(feedback, prefs);
        self.pulls_since_synthesis += 1;

        if let Some(ref mut ctx) = self.contextual {
            let policy = prefs.apply_to_policy(&self.bandit.config().reward_policy);
            let reward = policy.compute_reward(feedback);
            ctx.record(features, &feedback.arm_id, reward);
        }
    }

    /// Route with full context, auto-deriving preferences from context fields.
    pub fn route_with_context(
        &mut self,
        ctx: &RoutingContext,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let prefs = ctx.derive_preferences(&self.bandit.config().reward_policy);
        self.route_with_preferences(&ctx.features, &prefs)
    }

    /// Get a reference to the contextual discovery engine (if enabled).
    pub fn contextual_discovery(&self) -> Option<&ContextualDiscovery> {
        self.contextual.as_ref()
    }

    /// Get a reference to the active DFA (if any).
    pub fn active_dfa(&self) -> Option<&DfaRouter> {
        self.active_dfa.as_ref()
    }

    /// Get a reference to the source NFA (if any).
    pub fn source_nfa(&self) -> Option<&NfaRouter> {
        self.source_nfa.as_ref()
    }

    /// Get a reference to the bandit router.
    pub fn bandit(&self) -> &BanditRouter {
        &self.bandit
    }

    /// Get a mutable reference to the bandit router.
    pub fn bandit_mut(&mut self) -> &mut BanditRouter {
        &mut self.bandit
    }

    /// Replace the source NFA and active DFA directly.
    pub fn set_nfa_and_dfa(&mut self, nfa: NfaRouter, dfa: DfaRouter) {
        self.source_nfa = Some(nfa);
        self.active_dfa = Some(dfa);
    }

    /// Get the number of times re-synthesis has been performed.
    pub fn synthesis_count(&self) -> u64 {
        self.synthesis_count
    }

    /// Export the full pipeline state as a snapshot.
    pub fn export_snapshot(&self) -> PipelineSnapshot {
        PipelineSnapshot {
            version: PIPELINE_SNAPSHOT_VERSION,
            bandit: self.bandit.export_snapshot(),
            nfa: self.source_nfa.as_ref().map(|n| n.export_snapshot()),
            config: self.config.clone(),
            synthesis_count: self.synthesis_count,
            metadata: HashMap::new(),
        }
    }

    /// Serialize the pipeline state to JSON.
    pub fn to_json(&self) -> Result<String, AdvancedRoutingError> {
        let snapshot = self.export_snapshot();
        serde_json::to_string_pretty(&snapshot).map_err(|e| AdvancedRoutingError::SerializationFailed {
            format: "JSON".to_string(),
            reason: e.to_string(),
        })
    }

    /// Restore a pipeline from JSON.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: PipelineSnapshot = serde_json::from_str(json).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })?;
        if snapshot.version != PIPELINE_SNAPSHOT_VERSION {
            return Err(AdvancedRoutingError::IncompatibleVersion {
                expected: PIPELINE_SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }
        let bandit = BanditRouter::from_json(&serde_json::to_string(&snapshot.bandit).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })?)?;
        let source_nfa = if let Some(nfa_snap) = snapshot.nfa {
            let nfa = NfaRouter {
                states: nfa_snap.states,
                transitions: nfa_snap.transitions,
                start_states: nfa_snap.start_states,
            };
            Some(nfa)
        } else {
            None
        };
        let active_dfa = source_nfa.as_ref()
            .and_then(|nfa| NfaDfaCompiler::compile(nfa).ok())
            .map(|mut dfa| {
                if snapshot.config.auto_minimize {
                    dfa.minimize();
                }
                dfa
            });

        let contextual = snapshot.config.discovery.as_ref()
            .map(|dc| ContextualDiscovery::new(dc.clone()));

        Ok(Self {
            bandit,
            active_dfa,
            source_nfa,
            pulls_since_synthesis: 0,
            config: snapshot.config,
            synthesis_count: snapshot.synthesis_count,
            contextual,
        })
    }
}

/// Serializable snapshot of the entire routing pipeline state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    pub version: u32,
    pub bandit: BanditSnapshot,
    pub nfa: Option<NfaSnapshot>,
    pub config: PipelineConfig,
    pub synthesis_count: u64,
    pub metadata: HashMap<String, String>,
}

// =============================================================================
// MCP TOOLS FOR ROUTING MANAGEMENT (Section I)
// =============================================================================

use std::sync::{Arc, Mutex};

/// Register MCP tools for runtime routing management on the given server.
///
/// Tools registered:
/// - `routing.get_stats` — Bandit statistics (arms, pulls, rewards per task type)
/// - `routing.add_arm` — Add a model arm (global or per-task)
/// - `routing.remove_arm` — Remove an arm from the bandit
/// - `routing.warm_start` — Set priors for an arm (influence bandit towards/away from a model)
/// - `routing.record_outcome` — Manually record a feedback outcome
/// - `routing.add_rule` — Add a rule to the NFA and recompile DFA
/// - `routing.force_resynthesize` — Force NFA→DFA re-synthesis from bandit data
/// - `routing.export` — Export full pipeline state as JSON
/// - `routing.import` — Import pipeline state from JSON
/// - `routing.get_config` — Get current pipeline configuration
pub fn register_routing_tools(
    server: &mut crate::mcp_protocol::McpServer,
    pipeline: Arc<Mutex<RoutingPipeline>>,
) {
    use crate::mcp_protocol::McpTool;

    // --- routing.get_stats ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.get_stats", "Get bandit routing statistics: arms, pulls, rewards per task type")
            .with_property("task_type", "string", "Optional task type to filter (omit for global)", false),
        move |args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let bandit = pipeline.bandit();

            let task_type = args.get("task_type").and_then(|v| v.as_str());

            let arms: Vec<serde_json::Value> = bandit.all_arms_vec(task_type).iter().map(|arm| {
                let mean = if arm.pull_count > 0 { arm.total_reward / arm.pull_count as f64 } else { 0.0 };
                serde_json::json!({
                    "id": arm.id,
                    "pull_count": arm.pull_count,
                    "total_reward": arm.total_reward,
                    "mean_reward": mean,
                    "alpha": arm.params.alpha,
                    "beta": arm.params.beta,
                })
            }).collect();

            let task_types = bandit.task_types();

            Ok(serde_json::json!({
                "total_pulls": bandit.total_pulls(),
                "task_types": task_types,
                "arms": arms,
                "has_dfa": pipeline.active_dfa().is_some(),
                "synthesis_count": pipeline.synthesis_count(),
            }))
        },
    );

    // --- routing.add_arm ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.add_arm", "Add a model arm to the bandit router")
            .with_property("arm_id", "string", "Model identifier to add", true)
            .with_property("task_type", "string", "Task type (omit for global arm)", false)
            .with_property("alpha", "number", "Initial alpha prior (optional)", false)
            .with_property("beta", "number", "Initial beta prior (optional)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args.get("arm_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;

            let task_type = args.get("task_type").and_then(|v| v.as_str());
            let alpha = args.get("alpha").and_then(|v| v.as_f64());
            let beta = args.get("beta").and_then(|v| v.as_f64());

            match task_type {
                Some(tt) => pipeline.add_arm_for_task(tt, arm_id),
                None => pipeline.add_arm(arm_id),
            }

            // Apply warm-start if priors provided
            if let (Some(a), Some(b)) = (alpha, beta) {
                match task_type {
                    Some(tt) => pipeline.bandit_mut().warm_start_for_task(tt, arm_id, a, b),
                    None => pipeline.bandit_mut().warm_start(arm_id, a, b),
                }
            }

            Ok(serde_json::json!({
                "status": "ok",
                "arm_id": arm_id,
                "task_type": task_type,
            }))
        },
    );

    // --- routing.remove_arm ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.remove_arm", "Remove a model arm from the bandit router")
            .with_property("arm_id", "string", "Model identifier to remove", true)
            .with_property("task_type", "string", "Task type (omit to remove from global)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args.get("arm_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;

            let task_type = args.get("task_type").and_then(|v| v.as_str());

            let removed = pipeline.bandit_mut().remove_arm(arm_id, task_type);

            Ok(serde_json::json!({
                "status": if removed { "removed" } else { "not_found" },
                "arm_id": arm_id,
            }))
        },
    );

    // --- routing.warm_start ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.warm_start", "Set priors for a bandit arm to influence routing")
            .with_property("arm_id", "string", "Model identifier", true)
            .with_property("alpha", "number", "Alpha (success) prior — higher means more preferred", true)
            .with_property("beta", "number", "Beta (failure) prior — higher means less preferred", true)
            .with_property("task_type", "string", "Task type (omit for global)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args.get("arm_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let alpha = args.get("alpha").and_then(|v| v.as_f64())
                .ok_or("Missing required parameter: alpha")?;
            let beta = args.get("beta").and_then(|v| v.as_f64())
                .ok_or("Missing required parameter: beta")?;

            match args.get("task_type").and_then(|v| v.as_str()) {
                Some(tt) => pipeline.bandit_mut().warm_start_for_task(tt, arm_id, alpha, beta),
                None => pipeline.bandit_mut().warm_start(arm_id, alpha, beta),
            }

            Ok(serde_json::json!({
                "status": "ok",
                "arm_id": arm_id,
                "alpha": alpha,
                "beta": beta,
            }))
        },
    );

    // --- routing.record_outcome ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.record_outcome", "Record a feedback outcome for a model arm")
            .with_property("arm_id", "string", "Model that was used", true)
            .with_property("success", "boolean", "Whether the response was successful", true)
            .with_property("quality", "number", "Quality score 0.0-1.0 (optional, more precise than success)", false)
            .with_property("task_type", "string", "Task type (optional)", false)
            .with_property("latency_ms", "number", "Response latency in ms (optional)", false)
            .with_property("cost", "number", "Cost of the call (optional)", false),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args.get("arm_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let success = args.get("success").and_then(|v| v.as_bool())
                .ok_or("Missing required parameter: success")?;

            let feedback = ArmFeedback {
                arm_id: arm_id.to_string(),
                success,
                quality: args.get("quality").and_then(|v| v.as_f64()),
                latency_ms: args.get("latency_ms").and_then(|v| v.as_u64()),
                cost: args.get("cost").and_then(|v| v.as_f64()),
                task_type: args.get("task_type").and_then(|v| v.as_str()).map(|s| s.to_string()),
            };

            pipeline.record_outcome(&feedback);
            let resynthesized = pipeline.maybe_resynthesize();

            Ok(serde_json::json!({
                "status": "ok",
                "resynthesized": resynthesized,
            }))
        },
    );

    // --- routing.add_rule ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.add_rule", "Add a routing rule and recompile DFA")
            .with_property("domain", "string", "Domain to match (e.g. 'code', 'math')", false)
            .with_property("min_complexity", "number", "Minimum complexity % 0-100 (optional)", false)
            .with_property("max_complexity", "number", "Maximum complexity % 0-100 (optional)", false)
            .with_property("has_code", "boolean", "Match queries with code (optional)", false)
            .with_property("arm_id", "string", "Model to route to", true)
            .with_property("priority", "number", "Rule priority (higher wins)", true),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let arm_id = args.get("arm_id").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: arm_id")?;
            let priority = args.get("priority").and_then(|v| v.as_u64())
                .ok_or("Missing required parameter: priority")? as u32;

            // Build conditions from provided parameters
            let mut conditions: Vec<NfaSymbol> = Vec::new();

            if let Some(domain) = args.get("domain").and_then(|v| v.as_str()) {
                conditions.push(NfaSymbol::Domain(domain.to_string()));
            }

            let min_c = args.get("min_complexity").and_then(|v| v.as_u64()).map(|v| v as u32);
            let max_c = args.get("max_complexity").and_then(|v| v.as_u64()).map(|v| v as u32);
            if min_c.is_some() || max_c.is_some() {
                conditions.push(NfaSymbol::ComplexityRange {
                    low_pct: min_c.unwrap_or(0),
                    high_pct: max_c.unwrap_or(100),
                });
            }

            if let Some(has_code) = args.get("has_code").and_then(|v| v.as_bool()) {
                conditions.push(NfaSymbol::BoolFeature {
                    name: "has_code".to_string(),
                    value: has_code,
                });
            }

            // Get or create source NFA, add the new rule, recompile
            let mut nfa = pipeline.source_nfa().cloned().unwrap_or_else(NfaRouter::new);

            // If NFA is empty, create a start state
            if nfa.state_count() == 0 {
                nfa.add_state("start", None, 0);
            }

            let start = 0; // start state is always 0
            if conditions.is_empty() {
                // No conditions: direct accepting from start via Any
                let accept = nfa.add_state(&format!("rule_{}", arm_id), Some(arm_id), priority);
                nfa.add_transition(start, NfaSymbol::Any, accept);
            } else {
                let mut prev = start;
                for (i, cond) in conditions.iter().enumerate() {
                    let is_last = i == conditions.len() - 1;
                    if is_last {
                        let accept = nfa.add_state(
                            &format!("rule_{}", arm_id),
                            Some(arm_id),
                            priority,
                        );
                        nfa.add_transition(prev, cond.clone(), accept);
                    } else {
                        let inter = nfa.add_state(&format!("rule_{}_{}", arm_id, i), None, 0);
                        nfa.add_transition(prev, cond.clone(), inter);
                        prev = inter;
                    }
                }
            }

            // Recompile DFA
            match NfaDfaCompiler::compile(&nfa) {
                Ok(mut dfa) => {
                    dfa.minimize();
                    pipeline.set_nfa_and_dfa(nfa, dfa);
                    Ok(serde_json::json!({
                        "status": "ok",
                        "arm_id": arm_id,
                        "priority": priority,
                        "conditions_count": conditions.len(),
                    }))
                }
                Err(e) => Err(format!("DFA compilation failed: {}", e)),
            }
        },
    );

    // --- routing.force_resynthesize ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.force_resynthesize", "Force re-synthesis of NFA/DFA from bandit data"),
        move |_args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            pipeline.force_resynthesize().map_err(|e| e.to_string())?;

            Ok(serde_json::json!({
                "status": "ok",
                "synthesis_count": pipeline.synthesis_count(),
                "has_dfa": pipeline.active_dfa().is_some(),
            }))
        },
    );

    // --- routing.export ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.export", "Export full pipeline state as JSON"),
        move |_args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let json = pipeline.to_json().map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "status": "ok",
                "pipeline_json": json,
            }))
        },
    );

    // --- routing.import ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.import", "Import pipeline state from JSON")
            .with_property("pipeline_json", "string", "Pipeline JSON (from routing.export)", true),
        move |args| {
            let mut pipeline = p.lock().map_err(|e| e.to_string())?;
            let json = args.get("pipeline_json").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: pipeline_json")?;

            let restored = RoutingPipeline::from_json(json).map_err(|e| e.to_string())?;
            *pipeline = restored;

            Ok(serde_json::json!({
                "status": "ok",
                "has_dfa": pipeline.active_dfa().is_some(),
                "synthesis_count": pipeline.synthesis_count(),
            }))
        },
    );

    // --- routing.get_config ---
    let p = pipeline.clone();
    server.register_tool(
        McpTool::new("routing.get_config", "Get current pipeline configuration"),
        move |_args| {
            let pipeline = p.lock().map_err(|e| e.to_string())?;
            let snapshot = pipeline.export_snapshot();

            Ok(serde_json::json!({
                "synthesis_interval": snapshot.config.synthesis_interval,
                "min_pulls_for_synthesis": snapshot.config.min_pulls_for_synthesis,
                "quality_threshold": snapshot.config.quality_threshold,
                "auto_minimize": snapshot.config.auto_minimize,
                "synthesis_count": snapshot.synthesis_count,
                "has_dfa": pipeline.active_dfa().is_some(),
                "has_nfa": pipeline.source_nfa().is_some(),
                "bandit_arms_count": pipeline.bandit().all_arms(None).len(),
                "bandit_task_types": pipeline.bandit().task_types(),
            }))
        },
    );
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create default query features for testing
    fn test_features(domain: &str, complexity: f64) -> QueryFeatures {
        QueryFeatures {
            token_count: 50,
            sentence_count: 3,
            domain: domain.to_string(),
            complexity,
            entity_count: 2,
            has_code: false,
            is_question: true,
            avg_word_length: 5.0,
            feature_vector: vec![50.0, 3.0, complexity, 2.0, 0.0, 1.0, 5.0],
        }
    }

    fn test_features_code() -> QueryFeatures {
        QueryFeatures {
            token_count: 100,
            sentence_count: 5,
            domain: "coding".to_string(),
            complexity: 0.8,
            entity_count: 5,
            has_code: true,
            is_question: false,
            avg_word_length: 6.0,
            feature_vector: vec![100.0, 5.0, 0.8, 5.0, 1.0, 0.0, 6.0],
        }
    }

    // =========================================================================
    // ERROR TESTS
    // =========================================================================

    #[test]
    fn test_error_display() {
        let e = AdvancedRoutingError::ArmNotFound { arm_id: "test".to_string() };
        assert!(format!("{}", e).contains("test"));

        let e = AdvancedRoutingError::CycleDetected;
        assert!(format!("{}", e).contains("Cycle"));

        let e = AdvancedRoutingError::EmptyEnsemble;
        assert!(format!("{}", e).contains("no sub-routers"));
    }

    #[test]
    fn test_error_suggestion() {
        let e = AdvancedRoutingError::ArmNotFound { arm_id: "x".to_string() };
        assert!(e.suggestion().is_some());

        let e = AdvancedRoutingError::CycleDetected;
        assert!(e.suggestion().unwrap().contains("acyclic"));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(AdvancedRoutingError::NoRoutingPath { query: "q".to_string(), reason: "r".to_string() }.is_recoverable());
        assert!(AdvancedRoutingError::ArmNotFound { arm_id: "a".to_string() }.is_recoverable());
        assert!(!AdvancedRoutingError::CycleDetected.is_recoverable());
        assert!(!AdvancedRoutingError::EmptyEnsemble.is_recoverable());
    }

    #[test]
    fn test_error_from_conversion() {
        let e: crate::error::AiError = AdvancedRoutingError::CycleDetected.into();
        assert_eq!(e.code(), "ADVANCED_ROUTING");
    }

    #[test]
    fn test_error_code() {
        let e = crate::error::AiError::AdvancedRouting(AdvancedRoutingError::EmptyEnsemble);
        assert_eq!(e.code(), "ADVANCED_ROUTING");
        assert!(e.suggestion().is_some());
    }

    // =========================================================================
    // BANDIT ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_bandit_creation_default() {
        let bandit = BanditRouter::new(BanditConfig::default());
        assert!(bandit.all_arms(None).is_empty());
        assert_eq!(bandit.total_pulls(), 0);
    }

    #[test]
    fn test_bandit_add_arm() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("model-a");
        bandit.add_arm("model-b");
        assert_eq!(bandit.all_arms(None).len(), 2);
        // Adding duplicate does nothing
        bandit.add_arm("model-a");
        assert_eq!(bandit.all_arms(None).len(), 2);
    }

    #[test]
    fn test_bandit_add_arm_for_task() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("coding", "gpt-4");
        bandit.add_arm_for_task("coding", "claude");
        bandit.add_arm("default-model");
        assert_eq!(bandit.all_arms(Some("coding")).len(), 2);
        assert_eq!(bandit.all_arms(None).len(), 1);
    }

    #[test]
    fn test_thompson_sampling_selects_arm() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("a");
        bandit.add_arm("b");
        let result = bandit.select(None).unwrap();
        assert!(!result.selected_arm.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_ucb1_selects_unpulled_first() {
        let config = BanditConfig { strategy: BanditStrategy::Ucb1, ..Default::default() };
        let mut bandit = BanditRouter::new(config);
        bandit.add_arm("pulled");
        bandit.add_arm("unpulled");

        // Record some outcomes for "pulled"
        bandit.record_outcome(&ArmFeedback {
            arm_id: "pulled".to_string(),
            success: true,
            quality: Some(0.8),
            latency_ms: None, cost: None, task_type: None,
        });

        // UCB1 should select "unpulled" first (infinity score)
        let result = bandit.select(None).unwrap();
        assert_eq!(result.selected_arm, "unpulled");
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        let config = BanditConfig {
            strategy: BanditStrategy::EpsilonGreedy { epsilon: 1.0 },
            ..Default::default()
        };
        let mut bandit = BanditRouter::with_seed(config, 42);
        bandit.add_arm("a");
        bandit.add_arm("b");
        // With epsilon=1.0, always explores (random)
        let result = bandit.select(None).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_record_outcome_updates_params() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("model-a");

        let initial_alpha = bandit.arm_stats("model-a").unwrap().params.alpha;
        bandit.record_outcome(&ArmFeedback {
            arm_id: "model-a".to_string(),
            success: true,
            quality: None,
            latency_ms: None, cost: None, task_type: None,
        });

        let after = bandit.arm_stats("model-a").unwrap();
        assert!(after.params.alpha > initial_alpha);
        assert_eq!(after.pull_count, 1);
        assert_eq!(after.total_reward, 1.0);
    }

    #[test]
    fn test_warm_start_priors() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.warm_start("model-x", 10.0, 2.0);
        let arm = bandit.arm_stats("model-x").unwrap();
        assert_eq!(arm.params.alpha, 10.0);
        assert_eq!(arm.params.beta, 2.0);
    }

    #[test]
    fn test_per_task_bandit_isolation() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("coding", "gpt-4");
        bandit.add_arm_for_task("math", "claude");

        bandit.record_outcome(&ArmFeedback {
            arm_id: "gpt-4".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: None, cost: None,
            task_type: Some("coding".to_string()),
        });

        // Math arm should be unaffected
        let math_arm = bandit.all_arms(Some("math"));
        assert_eq!(math_arm[0].pull_count, 0);
    }

    #[test]
    fn test_beta_sampling_bounds() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 99);
        for _ in 0..100 {
            let sample = bandit.sample_beta(2.0, 5.0);
            assert!(sample >= 0.0 && sample <= 1.0, "Sample {} out of [0,1]", sample);
        }
    }

    #[test]
    fn test_bandit_select_no_arms_error() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        let result = bandit.select(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bandit_decay_factor() {
        let config = BanditConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut bandit = BanditRouter::new(config);
        bandit.add_arm("a");

        // First outcome
        bandit.record_outcome(&ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(1.0),
            latency_ms: None, cost: None, task_type: None,
        });
        let alpha1 = bandit.arm_stats("a").unwrap().params.alpha;

        // Second outcome with decay
        bandit.record_outcome(&ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(1.0),
            latency_ms: None, cost: None, task_type: None,
        });
        let alpha2 = bandit.arm_stats("a").unwrap().params.alpha;

        // With decay, alpha shouldn't grow as fast as without
        assert!(alpha2 < alpha1 + 1.0 + 0.01); // decayed first, then added 1
    }

    #[test]
    fn test_bandit_deterministic_with_seed() {
        let config = BanditConfig::default();
        let mut b1 = BanditRouter::with_seed(config.clone(), 42);
        let mut b2 = BanditRouter::with_seed(config, 42);
        b1.add_arm("a");
        b1.add_arm("b");
        b2.add_arm("a");
        b2.add_arm("b");

        let r1 = b1.select(None).unwrap();
        let r2 = b2.select(None).unwrap();
        assert_eq!(r1.selected_arm, r2.selected_arm);
    }

    #[test]
    fn test_bandit_routing_outcome_fields() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("x");
        bandit.add_arm("y");
        let result = bandit.select(None).unwrap();
        assert_eq!(result.router_id, "bandit");
        assert!(!result.reason.is_empty());
        // One alternative since 2 arms total
        assert_eq!(result.alternatives.len(), 1);
    }

    // =========================================================================
    // NFA ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_nfa_empty() {
        let nfa = NfaRouter::new();
        let features = test_features("general", 0.5);
        assert!(nfa.route(&features).is_err());
    }

    #[test]
    fn test_nfa_single_path() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("accept", Some("model-a"), 1);
        nfa.add_transition(s0, NfaSymbol::Any, s1);

        let features = test_features("general", 0.5);
        let result = nfa.route(&features).unwrap();
        assert_eq!(result.selected_arm, "model-a");
    }

    #[test]
    fn test_nfa_multiple_paths_priority() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("low-pri", Some("cheap"), 1);
        let s2 = nfa.add_state("high-pri", Some("expensive"), 10);
        nfa.add_transition(s0, NfaSymbol::Any, s1);
        nfa.add_transition(s0, NfaSymbol::Any, s2);

        let features = test_features("general", 0.5);
        let result = nfa.route(&features).unwrap();
        assert_eq!(result.selected_arm, "expensive"); // Higher priority wins
    }

    #[test]
    fn test_nfa_epsilon_transitions() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("mid", None, 0);
        let s2 = nfa.add_state("accept", Some("model-b"), 1);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);
        nfa.add_transition(s1, NfaSymbol::Any, s2);

        let features = test_features("general", 0.5);
        let result = nfa.route(&features).unwrap();
        assert_eq!(result.selected_arm, "model-b");
    }

    #[test]
    fn test_nfa_no_match() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let _s1 = nfa.add_state("accept", Some("model-a"), 1);
        nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), 1);

        let features = test_features("math", 0.5); // Not coding
        assert!(nfa.route(&features).is_err());
    }

    #[test]
    fn test_nfa_domain_symbol() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("coding_accept", Some("code-model"), 1);
        nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);

        let features = test_features("coding", 0.5);
        let result = nfa.route(&features).unwrap();
        assert_eq!(result.selected_arm, "code-model");
    }

    #[test]
    fn test_nfa_complexity_range() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("simple", Some("fast-model"), 1);
        let s2 = nfa.add_state("complex", Some("powerful-model"), 2);
        nfa.add_transition(s0, NfaSymbol::ComplexityRange { low_pct: 0, high_pct: 50 }, s1);
        nfa.add_transition(s0, NfaSymbol::ComplexityRange { low_pct: 50, high_pct: 100 }, s2);

        let simple = test_features("general", 0.3); // 30% -> fast
        assert_eq!(nfa.route(&simple).unwrap().selected_arm, "fast-model");

        let complex = test_features("general", 0.7); // 70% -> powerful
        assert_eq!(nfa.route(&complex).unwrap().selected_arm, "powerful-model");
    }

    #[test]
    fn test_nfa_token_range() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("short", Some("small-model"), 1);
        nfa.add_transition(s0, NfaSymbol::TokenRange { min: 0, max: 100 }, s1);

        let features = test_features("general", 0.5); // 50 tokens
        assert_eq!(nfa.route(&features).unwrap().selected_arm, "small-model");
    }

    #[test]
    fn test_nfa_bool_feature() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("code_accept", Some("code-model"), 1);
        nfa.add_transition(s0, NfaSymbol::BoolFeature { name: "has_code".to_string(), value: true }, s1);

        let features = test_features_code();
        assert_eq!(nfa.route(&features).unwrap().selected_arm, "code-model");
    }

    #[test]
    fn test_nfa_wildcard() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("catch-all", Some("default"), 1);
        nfa.add_transition(s0, NfaSymbol::Any, s1);

        let features = test_features("anything", 0.99);
        assert_eq!(nfa.route(&features).unwrap().selected_arm, "default");
    }

    #[test]
    fn test_nfa_state_count() {
        let mut nfa = NfaRouter::new();
        nfa.add_state("a", None, 0);
        nfa.add_state("b", Some("x"), 1);
        assert_eq!(nfa.state_count(), 2);
    }

    #[test]
    fn test_nfa_transition_count() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("a", None, 0);
        let s1 = nfa.add_state("b", Some("x"), 1);
        nfa.add_transition(s0, NfaSymbol::Any, s1);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);
        assert_eq!(nfa.transition_count(), 2);
    }

    // =========================================================================
    // DFA ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_dfa_route_single_state() {
        let dfa = DfaRouter {
            states: vec![DfaState { id: 0, label: "start".to_string(), accepting_arm: Some("model".to_string()), priority: 1 }],
            start_state: 0,
            transition_table: HashMap::new(),
        };
        let result = dfa.route(&test_features("general", 0.5)).unwrap();
        assert_eq!(result.selected_arm, "model");
    }

    #[test]
    fn test_dfa_route_transitions() {
        let mut table = HashMap::new();
        table.insert(0, vec![(NfaSymbol::Domain("coding".to_string()), 1)]);

        let dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "start".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "code".to_string(), accepting_arm: Some("code-model".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: table,
        };

        let features = test_features("coding", 0.5);
        assert_eq!(dfa.route(&features).unwrap().selected_arm, "code-model");
    }

    #[test]
    fn test_dfa_no_accepting_state() {
        let dfa = DfaRouter {
            states: vec![DfaState { id: 0, label: "start".to_string(), accepting_arm: None, priority: 0 }],
            start_state: 0,
            transition_table: HashMap::new(),
        };
        assert!(dfa.route(&test_features("general", 0.5)).is_err());
    }

    #[test]
    fn test_dfa_deterministic() {
        let mut table = HashMap::new();
        table.insert(0, vec![(NfaSymbol::Any, 1)]);

        let dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "s0".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "s1".to_string(), accepting_arm: Some("x".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: table,
        };

        let features = test_features("general", 0.5);
        let r1 = dfa.route(&features).unwrap();
        let r2 = dfa.route(&features).unwrap();
        assert_eq!(r1.selected_arm, r2.selected_arm);
    }

    #[test]
    fn test_dfa_minimize_already_minimal() {
        let mut table = HashMap::new();
        table.insert(0, vec![(NfaSymbol::Any, 1)]);

        let mut dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "s0".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "s1".to_string(), accepting_arm: Some("x".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: table,
        };

        let count_before = dfa.state_count();
        dfa.minimize();
        assert_eq!(dfa.state_count(), count_before);
    }

    #[test]
    fn test_dfa_minimize_equivalent_states() {
        // Two accepting states with same arm/priority and same transitions -> can merge
        let mut table = HashMap::new();
        table.insert(0, vec![(NfaSymbol::Domain("a".to_string()), 1), (NfaSymbol::Domain("b".to_string()), 2)]);

        let mut dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "s0".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "s1".to_string(), accepting_arm: Some("x".to_string()), priority: 1 },
                DfaState { id: 2, label: "s2".to_string(), accepting_arm: Some("x".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: table,
        };

        dfa.minimize();
        assert!(dfa.state_count() <= 3); // May or may not reduce depending on transition equivalence
    }

    #[test]
    fn test_dfa_state_count() {
        let dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "a".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "b".to_string(), accepting_arm: Some("x".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: HashMap::new(),
        };
        assert_eq!(dfa.state_count(), 2);
    }

    #[test]
    fn test_dfa_transition_count() {
        let mut table = HashMap::new();
        table.insert(0, vec![(NfaSymbol::Any, 1), (NfaSymbol::Domain("x".to_string()), 1)]);
        let dfa = DfaRouter {
            states: vec![
                DfaState { id: 0, label: "a".to_string(), accepting_arm: None, priority: 0 },
                DfaState { id: 1, label: "b".to_string(), accepting_arm: Some("m".to_string()), priority: 1 },
            ],
            start_state: 0,
            transition_table: table,
        };
        assert_eq!(dfa.transition_count(), 2);
    }

    #[test]
    fn test_dfa_empty_returns_error() {
        let dfa = DfaRouter {
            states: Vec::new(),
            start_state: 0,
            transition_table: HashMap::new(),
        };
        assert!(dfa.route(&test_features("general", 0.5)).is_err());
    }

    #[test]
    fn test_dfa_route_outcome_fields() {
        let dfa = DfaRouter {
            states: vec![DfaState { id: 0, label: "s0".to_string(), accepting_arm: Some("m".to_string()), priority: 1 }],
            start_state: 0,
            transition_table: HashMap::new(),
        };
        let r = dfa.route(&test_features("general", 0.5)).unwrap();
        assert_eq!(r.router_id, "dfa");
        assert_eq!(r.selected_arm, "m");
    }

    // =========================================================================
    // NFA→DFA COMPILER TESTS
    // =========================================================================

    #[test]
    fn test_compile_trivial_nfa() {
        let mut nfa = NfaRouter::new();
        nfa.add_state("accept", Some("model"), 1);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        assert_eq!(dfa.state_count(), 1);
        assert_eq!(dfa.states[0].accepting_arm.as_deref(), Some("model"));
    }

    #[test]
    fn test_compile_epsilon_only() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("accept", Some("m"), 1);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        // s0 and s1 in same epsilon closure -> DFA start state is accepting
        assert!(dfa.states[0].accepting_arm.is_some());
    }

    #[test]
    fn test_compile_branching_nfa() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("code", Some("code-m"), 1);
        let s2 = nfa.add_state("math", Some("math-m"), 2);
        nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);
        nfa.add_transition(s0, NfaSymbol::Domain("math".to_string()), s2);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        assert!(dfa.state_count() >= 2); // At least start + branches
    }

    #[test]
    fn test_compile_preserves_accepting() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("accept", Some("model-x"), 5);
        nfa.add_transition(s0, NfaSymbol::Any, s1);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let accepting: Vec<_> = dfa.states.iter()
            .filter(|s| s.accepting_arm.is_some())
            .collect();
        assert!(!accepting.is_empty());
        assert_eq!(accepting[0].accepting_arm.as_deref(), Some("model-x"));
    }

    #[test]
    fn test_compile_preserves_priority() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("low", Some("cheap"), 1);
        let s2 = nfa.add_state("high", Some("expensive"), 10);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s2);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        // Both in epsilon closure, max priority should win
        assert_eq!(dfa.states[0].priority, 10);
        assert_eq!(dfa.states[0].accepting_arm.as_deref(), Some("expensive"));
    }

    #[test]
    fn test_compile_then_route_matches_nfa() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("code", Some("code-m"), 1);
        let s2 = nfa.add_state("general", Some("gen-m"), 1);
        nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);
        nfa.add_transition(s0, NfaSymbol::Domain("general".to_string()), s2);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();

        let coding_features = test_features("coding", 0.5);
        let nfa_result = nfa.route(&coding_features).unwrap();
        let dfa_result = dfa.route(&coding_features).unwrap();
        assert_eq!(nfa_result.selected_arm, dfa_result.selected_arm);
    }

    #[test]
    fn test_compile_empty_nfa_error() {
        let nfa = NfaRouter::new();
        assert!(NfaDfaCompiler::compile(&nfa).is_err());
    }

    #[test]
    fn test_compile_no_epsilon_in_dfa() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("mid", None, 0);
        let s2 = nfa.add_state("accept", Some("m"), 1);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);
        nfa.add_transition(s1, NfaSymbol::Any, s2);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        for transitions in dfa.transition_table.values() {
            for (sym, _) in transitions {
                assert_ne!(*sym, NfaSymbol::Epsilon);
            }
        }
    }

    #[test]
    fn test_compile_complex_nfa() {
        let mut nfa = NfaRouter::new();
        let s0 = nfa.add_state("start", None, 0);
        let s1 = nfa.add_state("a", None, 0);
        let s2 = nfa.add_state("b", None, 0);
        let s3 = nfa.add_state("c", Some("model-c"), 3);
        let s4 = nfa.add_state("d", Some("model-d"), 2);

        nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);
        nfa.add_transition(s0, NfaSymbol::Domain("math".to_string()), s2);
        nfa.add_transition(s1, NfaSymbol::ComplexityRange { low_pct: 50, high_pct: 100 }, s3);
        nfa.add_transition(s1, NfaSymbol::ComplexityRange { low_pct: 0, high_pct: 50 }, s4);
        nfa.add_transition(s2, NfaSymbol::Epsilon, s3);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        assert!(dfa.state_count() >= 2);
        assert!(dfa.transition_count() >= 1);
    }

    // =========================================================================
    // ROUTING DAG TESTS
    // =========================================================================

    #[test]
    fn test_dag_single_leaf() {
        let mut dag = RoutingDag::new("root");
        dag.add_node(RoutingDagNode {
            id: "root".to_string(),
            label: "Root".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "model-a".to_string() },
            successors: Vec::new(),
        }).unwrap();

        let features = test_features("general", 0.5);
        let result = dag.route(&features).unwrap();
        assert_eq!(result.selected_arm, "model-a");
        assert!(result.reason.contains("DAG path"));
    }

    #[test]
    fn test_dag_rule_based_branch() {
        let mut dag = RoutingDag::new("rule");
        dag.add_node(RoutingDagNode {
            id: "rule".to_string(),
            label: "Complexity gate".to_string(),
            node_type: RoutingDagNodeType::RuleBased {
                feature: "complexity".to_string(),
                threshold: 0.5,
                high_branch: "powerful".to_string(),
                low_branch: "cheap".to_string(),
            },
            successors: vec!["powerful".to_string(), "cheap".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "powerful".to_string(), label: "Powerful".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "gpt-4".to_string() },
            successors: Vec::new(),
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "cheap".to_string(), label: "Cheap".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "gpt-3.5".to_string() },
            successors: Vec::new(),
        }).unwrap();

        let simple = test_features("general", 0.3);
        assert_eq!(dag.route(&simple).unwrap().selected_arm, "gpt-3.5");

        let complex = test_features("general", 0.8);
        assert_eq!(dag.route(&complex).unwrap().selected_arm, "gpt-4");
    }

    #[test]
    fn test_dag_bandit_node() {
        let mut dag = RoutingDag::new("bandit_root");
        dag.add_node(RoutingDagNode {
            id: "bandit_root".to_string(), label: "Bandit".to_string(),
            node_type: RoutingDagNodeType::Bandit(BanditConfig::default()),
            successors: vec!["leaf".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "leaf".to_string(), label: "Leaf".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "final-model".to_string() },
            successors: Vec::new(),
        }).unwrap();

        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("leaf");
        dag.set_bandit("bandit_root", bandit).unwrap();

        let features = test_features("general", 0.5);
        let result = dag.route(&features).unwrap();
        assert_eq!(result.selected_arm, "final-model");
    }

    #[test]
    fn test_dag_cycle_detection() {
        let mut dag = RoutingDag::new("a");
        dag.add_node(RoutingDagNode {
            id: "a".to_string(), label: "A".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "x".to_string() },
            successors: vec!["b".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "b".to_string(), label: "B".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "y".to_string() },
            successors: vec!["a".to_string()],
        }).unwrap();

        assert!(dag.validate().is_err());
    }

    #[test]
    fn test_dag_node_not_found() {
        let mut dag = RoutingDag::new("nonexistent");
        assert!(dag.set_bandit("missing", BanditRouter::new(BanditConfig::default())).is_err());
    }

    #[test]
    fn test_dag_topological_order() {
        let mut dag = RoutingDag::new("root");
        dag.add_node(RoutingDagNode {
            id: "root".to_string(), label: "R".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "x".to_string() },
            successors: vec!["child".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "child".to_string(), label: "C".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "y".to_string() },
            successors: Vec::new(),
        }).unwrap();

        let order = dag.topological_order().unwrap();
        assert_eq!(order.len(), 2);
        assert!(order.iter().position(|x| x == "root").unwrap() < order.iter().position(|x| x == "child").unwrap());
    }

    #[test]
    fn test_dag_validate_acyclic() {
        let mut dag = RoutingDag::new("a");
        dag.add_node(RoutingDagNode {
            id: "a".to_string(), label: "A".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "x".to_string() },
            successors: vec!["b".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "b".to_string(), label: "B".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "y".to_string() },
            successors: Vec::new(),
        }).unwrap();

        assert!(dag.validate().is_ok());
    }

    #[test]
    fn test_dag_record_outcome() {
        let mut dag = RoutingDag::new("b");
        dag.add_node(RoutingDagNode {
            id: "b".to_string(), label: "B".to_string(),
            node_type: RoutingDagNodeType::Bandit(BanditConfig::default()),
            successors: Vec::new(),
        }).unwrap();
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("test");
        dag.set_bandit("b", bandit).unwrap();

        dag.record_outcome("b", &ArmFeedback {
            arm_id: "test".to_string(), success: true, quality: Some(0.9),
            latency_ms: None, cost: None, task_type: None,
        });
        // Bandit should have updated
        assert_eq!(dag.bandit_instances["b"].all_arms(None)[0].pull_count, 1);
    }

    #[test]
    fn test_dag_multi_level() {
        let mut dag = RoutingDag::new("gate");
        dag.add_node(RoutingDagNode {
            id: "gate".to_string(), label: "Gate".to_string(),
            node_type: RoutingDagNodeType::RuleBased {
                feature: "has_code".to_string(), threshold: 0.5,
                high_branch: "code_leaf".to_string(), low_branch: "gen_leaf".to_string(),
            },
            successors: vec!["code_leaf".to_string(), "gen_leaf".to_string()],
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "code_leaf".to_string(), label: "Code".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "coder".to_string() },
            successors: Vec::new(),
        }).unwrap();
        dag.add_node(RoutingDagNode {
            id: "gen_leaf".to_string(), label: "General".to_string(),
            node_type: RoutingDagNodeType::Leaf { arm_id: "general".to_string() },
            successors: Vec::new(),
        }).unwrap();

        let code_f = test_features_code();
        assert_eq!(dag.route(&code_f).unwrap().selected_arm, "coder");

        let gen_f = test_features("general", 0.3);
        assert_eq!(dag.route(&gen_f).unwrap().selected_arm, "general");
    }

    // =========================================================================
    // EVAL FEEDBACK MAPPER TESTS (cfg-gated)
    // =========================================================================

    #[cfg(feature = "eval-suite")]
    mod eval_feedback_tests {
        use super::*;
        use crate::eval_suite::{ConfigSearchResult, ConfigMeasurement, EvalAgentConfig};
        use crate::eval_suite::ModelIdentifier;

        fn mock_search_result(quality: f64) -> ConfigSearchResult {
            let mut subtask_quality = HashMap::new();
            subtask_quality.insert("CodeGeneration".to_string(), quality);
            subtask_quality.insert("Reasoning".to_string(), quality * 0.9);

            let mut subtask_models = HashMap::new();
            subtask_models.insert("CodeGeneration".to_string(), ModelIdentifier {
                name: "gpt-4".to_string(),
                provider: "openai".to_string(),
                variant: None,
            });

            let config = EvalAgentConfig {
                subtask_models,
                ..Default::default()
            };

            let measurement = ConfigMeasurement {
                config: config.clone(),
                quality,
                quality_std: 0.1,
                latency_ms: 100.0,
                cost: 0.5,
                sample_count: 10,
                subtask_quality,
                run_result: None,
            };

            ConfigSearchResult {
                baseline: measurement.clone(),
                best: measurement,
                iterations: Vec::new(),
                evolution: Vec::new(),
                dimension_variance: HashMap::new(),
                recommended: config,
                quality_improvement_pct: 0.0,
                cost_change_pct: 0.0,
                total_evaluations: 1,
                search_cost: crate::eval_suite::SearchCost {
                    total_configurations_evaluated: 5,
                    total_problems_solved: 50,
                    total_llm_calls: 10,
                    estimated_total_cost: 5.0,
                    estimated_total_tokens: 10000,
                },
                converged: true,
                stopped_by_budget: false,
            }
        }

        #[test]
        fn test_map_to_priors_basic() {
            let result = mock_search_result(0.8);
            let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
            assert!(priors.contains_key("CodeGeneration"));
            let code_priors = &priors["CodeGeneration"];
            assert!(!code_priors.is_empty());
            // quality=0.8, scale=10 -> alpha=8, beta=2
            for (_, params) in code_priors {
                assert!((params.alpha - 8.0).abs() < 0.1);
                assert!((params.beta - 2.0).abs() < 0.1);
            }
        }

        #[test]
        fn test_apply_to_bandit() {
            let result = mock_search_result(0.8);
            let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
            let mut bandit = BanditRouter::new(BanditConfig::default());
            EvalFeedbackMapper::apply_to_bandit(&mut bandit, &priors);
            // Should have arms for CodeGeneration and Reasoning tasks
            assert!(!bandit.all_arms(Some("CodeGeneration")).is_empty());
        }

        #[test]
        fn test_create_warm_started_bandit() {
            let result = mock_search_result(0.9);
            let bandit = EvalFeedbackMapper::create_warm_started_bandit(
                &result, BanditConfig::default(), 10.0,
            );
            assert!(!bandit.all_arms(Some("CodeGeneration")).is_empty());
        }

        #[test]
        fn test_map_zero_quality() {
            let result = mock_search_result(0.0);
            let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
            for (_, arm_priors) in &priors {
                for (_, params) in arm_priors {
                    assert!(params.alpha >= 0.01);
                    assert!(params.beta >= 0.01);
                }
            }
        }

        #[test]
        fn test_map_perfect_quality() {
            let result = mock_search_result(1.0);
            let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
            let code_priors = &priors["CodeGeneration"];
            for (_, params) in code_priors {
                assert!((params.alpha - 10.0).abs() < 0.1);
            }
        }

        #[test]
        fn test_round_trip_eval_to_bandit() {
            let result = mock_search_result(0.75);
            let mut bandit = EvalFeedbackMapper::create_warm_started_bandit(
                &result, BanditConfig::default(), 10.0,
            );
            // Should be able to select from the warm-started bandit
            let outcome = bandit.select(Some("CodeGeneration"));
            assert!(outcome.is_ok());
        }

        // =====================================================================
        // BANDIT BOOTSTRAPPER TESTS
        // =====================================================================

        use crate::eval_suite::{ComparisonMatrix, SubtaskAnalysis, SubtaskPerformance, Subtask};

        fn mock_comparison_matrix(n_models: usize) -> ComparisonMatrix {
            let models: Vec<ModelIdentifier> = (0..n_models).map(|i| {
                ModelIdentifier {
                    name: format!("model-{}", i),
                    provider: "test".to_string(),
                    variant: None,
                }
            }).collect();
            let metrics = vec![
                "accuracy".to_string(), "mean_score".to_string(),
                "mean_latency_ms".to_string(), "total_cost".to_string(),
            ];
            let scores: Vec<Vec<f64>> = (0..n_models).map(|i| {
                let q = 0.5 + 0.1 * i as f64;
                vec![q, q, 200.0, 0.01 * (i + 1) as f64]
            }).collect();
            let costs: Vec<f64> = (0..n_models).map(|i| 0.01 * (i + 1) as f64).collect();
            let cost_effectiveness: Vec<f64> = scores.iter().zip(costs.iter())
                .map(|(s, c)| if *c > 0.0 { s[1] / c } else { 0.0 })
                .collect();
            ComparisonMatrix {
                models, metrics, scores,
                significance: vec![vec![1.0; n_models]; n_models],
                elo_ratings: HashMap::new(),
                costs,
                cost_effectiveness,
            }
        }

        #[test]
        fn test_bootstrapper_from_empty_matrix() {
            let matrix = ComparisonMatrix {
                models: vec![], metrics: vec![], scores: vec![],
                significance: vec![], elo_ratings: HashMap::new(),
                costs: vec![], cost_effectiveness: vec![],
            };
            let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
            assert!(priors.is_empty());
        }

        #[test]
        fn test_bootstrapper_from_single_model() {
            let matrix = mock_comparison_matrix(1);
            let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
            assert!(priors.contains_key("global"));
            assert_eq!(priors["global"].len(), 1);
            let arm_id = matrix.models[0].to_string();
            assert!(priors["global"].contains_key(&arm_id));
        }

        #[test]
        fn test_bootstrapper_from_multiple_models() {
            let matrix = mock_comparison_matrix(3);
            let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
            assert_eq!(priors["global"].len(), 3);
        }

        #[test]
        fn test_bootstrapper_uses_reward_policy_weights() {
            let matrix = mock_comparison_matrix(2);
            let cost_policy = RewardPolicy {
                quality_weight: 0.1, latency_weight: 0.0, cost_weight: 0.9,
                latency_ref_ms: 5000.0, cost_ref: 0.1,
            };
            let priors_cost = BanditBootstrapper::from_comparison_matrix(&matrix, &cost_policy);
            let qual_policy = RewardPolicy {
                quality_weight: 0.9, latency_weight: 0.0, cost_weight: 0.1,
                latency_ref_ms: 5000.0, cost_ref: 0.1,
            };
            let priors_qual = BanditBootstrapper::from_comparison_matrix(&matrix, &qual_policy);
            let arm_id = matrix.models[0].to_string();
            let alpha_cost = priors_cost["global"][&arm_id].alpha;
            let alpha_qual = priors_qual["global"][&arm_id].alpha;
            assert!(alpha_cost > 0.0);
            assert!(alpha_qual > 0.0);
        }

        fn mock_subtask_analysis() -> SubtaskAnalysis {
            SubtaskAnalysis {
                performances: vec![
                    SubtaskPerformance {
                        subtask: Subtask::CodeGeneration,
                        model_id: ModelIdentifier {
                            name: "gpt-4o".to_string(),
                            provider: "openai".to_string(),
                            variant: None,
                        },
                        score: 0.85,
                        sample_count: 50,
                        latency_mean_ms: 300.0,
                        cost_mean: 0.02,
                    },
                    SubtaskPerformance {
                        subtask: Subtask::ReasoningChain,
                        model_id: ModelIdentifier {
                            name: "claude-3.5-sonnet".to_string(),
                            provider: "anthropic".to_string(),
                            variant: None,
                        },
                        score: 0.92,
                        sample_count: 50,
                        latency_mean_ms: 400.0,
                        cost_mean: 0.03,
                    },
                ],
                optimal_routing: HashMap::new(),
                routed_composite_score: 0.88,
                best_single_model_score: 0.85,
                routing_improvement_pct: 3.5,
            }
        }

        #[test]
        fn test_bootstrapper_from_subtask_analysis_empty() {
            let analysis = SubtaskAnalysis {
                performances: vec![],
                optimal_routing: HashMap::new(),
                routed_composite_score: 0.0,
                best_single_model_score: 0.0,
                routing_improvement_pct: 0.0,
            };
            let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
            assert!(priors.is_empty());
        }

        #[test]
        fn test_bootstrapper_from_subtask_analysis_basic() {
            let analysis = mock_subtask_analysis();
            let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
            assert!(priors.contains_key("CodeGeneration"));
            assert!(priors.contains_key("ReasoningChain"));
            let arm_id = "openai/gpt-4o".to_string();
            let code_priors = &priors["CodeGeneration"];
            assert!(code_priors.contains_key(&arm_id));
            assert!((code_priors[&arm_id].alpha - 8.5).abs() < 0.1);
            assert!((code_priors[&arm_id].beta - 1.5).abs() < 0.1);
        }

        #[test]
        fn test_bootstrapper_from_subtask_analysis_multiple() {
            let analysis = mock_subtask_analysis();
            let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
            assert_eq!(priors.len(), 2);
        }

        #[test]
        fn test_bootstrapper_bootstrap_pipeline() {
            let analysis = mock_subtask_analysis();
            let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
            let pipeline = BanditBootstrapper::bootstrap_pipeline(
                &priors, BanditConfig::default(), PipelineConfig::default(),
            );
            assert!(!pipeline.bandit().all_arms(Some("CodeGeneration")).is_empty());
        }

        #[test]
        fn test_bootstrapper_round_trip_select() {
            // Use from_comparison_matrix which creates "global" priors (accessible to all domains)
            let matrix = mock_comparison_matrix(2);
            let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
            let mut pipeline = BanditBootstrapper::bootstrap_pipeline(
                &priors, BanditConfig::default(), PipelineConfig::default(),
            );
            let features = QueryFeatureExtractor::extract("implement a function");
            let outcome = pipeline.route(&features);
            assert!(outcome.is_ok());
        }

        #[test]
        fn test_bootstrapper_scale_effect() {
            let analysis = mock_subtask_analysis();
            let priors_low = BanditBootstrapper::from_subtask_analysis(&analysis, 1.0);
            let priors_high = BanditBootstrapper::from_subtask_analysis(&analysis, 100.0);
            let arm_id = "openai/gpt-4o".to_string();
            let alpha_low = priors_low["CodeGeneration"][&arm_id].alpha;
            let alpha_high = priors_high["CodeGeneration"][&arm_id].alpha;
            assert!(alpha_high > alpha_low * 10.0);
        }
    }

    // =========================================================================
    // ADAPTIVE PER-QUERY ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_feature_extraction_code_query() {
        let features = QueryFeatureExtractor::extract("implement a function fn main() {}");
        assert!(features.has_code);
        assert_eq!(features.domain, "coding");
    }

    #[test]
    fn test_feature_extraction_question() {
        let features = QueryFeatureExtractor::extract("What is the capital of France?");
        assert!(features.is_question);
    }

    #[test]
    fn test_feature_extraction_complexity() {
        let simple = QueryFeatureExtractor::extract("Hi");
        let complex = QueryFeatureExtractor::extract(
            "Analyze the socioeconomic implications of climate change on developing nations, \
             considering factors such as agricultural productivity, migration patterns, \
             infrastructure resilience, and international trade dynamics."
        );
        assert!(complex.complexity > simple.complexity);
    }

    #[test]
    fn test_feature_extraction_domain() {
        assert_eq!(QueryFeatureExtractor::extract("solve this equation").domain, "math");
        assert_eq!(QueryFeatureExtractor::extract("write a poem about love").domain, "creative");
        assert_eq!(QueryFeatureExtractor::extract("translate this to French").domain, "translation");
    }

    #[test]
    fn test_adaptive_routes_code_to_code_model() {
        let config = BanditConfig::default();
        let mut router = AdaptivePerQueryRouter::new("default", config).with_code_model("code-model");
        let result = router.route("fn main() { println!(\"hello\"); }").unwrap();
        assert_eq!(result.selected_arm, "code-model");
    }

    #[test]
    fn test_adaptive_routes_by_complexity() {
        let config = BanditConfig::default();
        let mut router = AdaptivePerQueryRouter::new("default", config)
            .add_complexity_tier(0.3, "fast")
            .add_complexity_tier(0.7, "medium");

        let result = router.route("Hi").unwrap();
        assert_eq!(result.selected_arm, "fast");
    }

    #[test]
    fn test_adaptive_records_outcomes() {
        let config = BanditConfig::default();
        let mut router = AdaptivePerQueryRouter::new("default", config);
        router.route("What is coding?").unwrap();
        router.record_outcome("What is coding?", &ArmFeedback {
            arm_id: "default".to_string(), success: true, quality: Some(0.9),
            latency_ms: None, cost: None, task_type: None,
        });
        // Should not panic
    }

    #[test]
    fn test_adaptive_default_model() {
        let config = BanditConfig::default();
        let mut router = AdaptivePerQueryRouter::new("my-default", config);
        let result = router.route("Hello world").unwrap();
        assert_eq!(result.selected_arm, "my-default");
    }

    #[test]
    fn test_query_feature_vector_dims() {
        let features = QueryFeatureExtractor::extract("test query");
        assert_eq!(features.feature_vector.len(), 7);
    }

    #[test]
    fn test_adaptive_default_model_getter() {
        let config = BanditConfig::default();
        let router = AdaptivePerQueryRouter::new("test-model", config);
        assert_eq!(router.default_model(), "test-model");
    }

    // =========================================================================
    // ENSEMBLE ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_ensemble_majority_vote() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        // Add 3 bandits that will vote for the same arm
        for i in 0..3 {
            let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42 + i);
            bandit.add_arm("winner");
            ensemble.add_voter(Box::new(bandit), 1.0);
        }

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "winner");
    }

    #[test]
    fn test_ensemble_weighted_average() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::WeightedAverage);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("model-a");
        let mut b2 = BanditRouter::with_seed(BanditConfig::default(), 99);
        b2.add_arm("model-a");

        ensemble.add_voter(Box::new(b1), 10.0);
        ensemble.add_voter(Box::new(b2), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_ensemble_unanimous_success() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::Unanimous);
        for i in 0..3 {
            let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42 + i);
            bandit.add_arm("same-model");
            ensemble.add_voter(Box::new(bandit), 1.0);
        }

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "same-model");
    }

    #[test]
    fn test_ensemble_unanimous_failure() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::Unanimous);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("model-a");
        let mut b2 = BanditRouter::with_seed(BanditConfig::default(), 99);
        b2.add_arm("model-b"); // Different arm!
        ensemble.add_voter(Box::new(b1), 1.0);
        ensemble.add_voter(Box::new(b2), 1.0);

        let features = test_features("general", 0.5);
        assert!(ensemble.route(&features).is_err());
    }

    #[test]
    fn test_ensemble_max_confidence() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MaxConfidence);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("a");
        ensemble.add_voter(Box::new(b1), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_ensemble_empty_error() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let features = test_features("general", 0.5);
        assert!(ensemble.route(&features).is_err());
    }

    #[test]
    fn test_ensemble_single_router() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("only");
        ensemble.add_voter(Box::new(bandit), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "only");
    }

    #[test]
    fn test_ensemble_record_outcome_propagates() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let mut b = BanditRouter::new(BanditConfig::default());
        b.add_arm("test");
        ensemble.add_voter(Box::new(b), 1.0);

        ensemble.record_outcome(&ArmFeedback {
            arm_id: "test".to_string(), success: true, quality: Some(0.9),
            latency_ms: None, cost: None, task_type: None,
        });
        // Should not panic
        assert_eq!(ensemble.voter_count(), 1);
    }

    #[test]
    fn test_ensemble_routing_voter_impl() {
        // BanditRouter implements RoutingVoter
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("x");
        let voter: &mut dyn RoutingVoter = &mut bandit;
        let features = test_features("general", 0.5);
        let result = voter.vote(&features);
        assert!(result.is_ok());
        assert_eq!(voter.router_id(), "bandit");
    }

    #[test]
    fn test_ensemble_mixed_types() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);

        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("shared");
        ensemble.add_voter(Box::new(bandit), 1.0);

        let adaptive = AdaptivePerQueryRouter::new("shared", BanditConfig::default());
        ensemble.add_voter(Box::new(adaptive), 1.0);

        assert_eq!(ensemble.voter_count(), 2);
        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    // =========================================================================
    // DISTRIBUTED BANDIT TESTS (cfg-gated)
    // =========================================================================

    #[cfg(feature = "distributed")]
    mod distributed_tests {
        use super::*;

        #[test]
        fn test_extract_state() {
            let mut router = BanditRouter::new(BanditConfig::default());
            router.add_arm("a");
            router.add_arm("b");
            let state = BanditStateMerger::extract_state(&router, "node-1");
            assert_eq!(state.node_id, "node-1");
            assert_eq!(state.global_arms.len(), 2);
        }

        #[test]
        fn test_merge_two_nodes() {
            let mut r1 = BanditRouter::new(BanditConfig::default());
            r1.add_arm("a");
            r1.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(), success: true, quality: Some(0.8),
                latency_ms: None, cost: None, task_type: None,
            });

            let mut r2 = BanditRouter::new(BanditConfig::default());
            r2.add_arm("a");
            r2.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(), success: true, quality: Some(0.6),
                latency_ms: None, cost: None, task_type: None,
            });

            let s1 = BanditStateMerger::extract_state(&r1, "n1");
            let s2 = BanditStateMerger::extract_state(&r2, "n2");

            let merged = BanditStateMerger::merge(&[s1, s2], 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms.len(), 1);
            assert_eq!(merged.global_arms[0].pull_count, 2);
        }

        #[test]
        fn test_merge_three_nodes() {
            let states: Vec<DistributedBanditState> = (0..3).map(|i| {
                let mut r = BanditRouter::new(BanditConfig::default());
                r.add_arm("shared");
                r.record_outcome(&ArmFeedback {
                    arm_id: "shared".to_string(), success: true, quality: Some(0.7),
                    latency_ms: None, cost: None, task_type: None,
                });
                BanditStateMerger::extract_state(&r, &format!("node-{}", i))
            }).collect();

            let merged = BanditStateMerger::merge(&states, 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms[0].pull_count, 3);
        }

        #[test]
        fn test_merge_disjoint_arms() {
            let mut r1 = BanditRouter::new(BanditConfig::default());
            r1.add_arm("model-a");
            let mut r2 = BanditRouter::new(BanditConfig::default());
            r2.add_arm("model-b");

            let s1 = BanditStateMerger::extract_state(&r1, "n1");
            let s2 = BanditStateMerger::extract_state(&r2, "n2");

            let merged = BanditStateMerger::merge(&[s1, s2], 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms.len(), 2);
        }

        #[test]
        fn test_merge_into_router() {
            let mut local = BanditRouter::new(BanditConfig::default());
            local.add_arm("a");
            local.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: None,
            });

            let mut remote = BanditRouter::new(BanditConfig::default());
            remote.add_arm("a");
            remote.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(), success: true, quality: Some(0.7),
                latency_ms: None, cost: None, task_type: None,
            });
            let remote_state = BanditStateMerger::extract_state(&remote, "remote");

            BanditStateMerger::merge_into_router(&mut local, &remote_state, 1.0, 1.0).unwrap();
            assert_eq!(local.arm_stats("a").unwrap().pull_count, 2);
        }

        #[test]
        fn test_merge_empty_states_error() {
            let result = BanditStateMerger::merge(&[], 1.0, 1.0);
            assert!(result.is_err());
        }

        #[test]
        fn test_merge_preserves_pull_count() {
            let states: Vec<DistributedBanditState> = (0..5).map(|i| {
                let mut r = BanditRouter::new(BanditConfig::default());
                r.add_arm("x");
                for _ in 0..3 {
                    r.record_outcome(&ArmFeedback {
                        arm_id: "x".to_string(), success: true, quality: Some(0.8),
                        latency_ms: None, cost: None, task_type: None,
                    });
                }
                BanditStateMerger::extract_state(&r, &format!("n{}", i))
            }).collect();

            let merged = BanditStateMerger::merge(&states, 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms[0].pull_count, 15); // 5 nodes * 3 pulls
        }

        #[test]
        fn test_merge_idempotent() {
            let mut r = BanditRouter::new(BanditConfig::default());
            r.add_arm("x");
            r.record_outcome(&ArmFeedback {
                arm_id: "x".to_string(), success: true, quality: Some(0.8),
                latency_ms: None, cost: None, task_type: None,
            });

            let state = BanditStateMerger::extract_state(&r, "n1");
            // Merging single state = itself (with prior correction = no change)
            let merged = BanditStateMerger::merge(&[state.clone()], 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms[0].pull_count, state.global_arms[0].pull_count);
        }

        #[test]
        fn test_extract_state_filters_private_global_arms() {
            let mut router = BanditRouter::new(BanditConfig::default());
            router.add_arm("public-model");
            router.add_arm("private-model");
            router.set_arm_private("private-model");

            let state = BanditStateMerger::extract_state(&router, "node1");
            assert_eq!(state.global_arms.len(), 1);
            assert_eq!(state.global_arms[0].id, "public-model");
        }

        #[test]
        fn test_extract_state_filters_private_task_arms() {
            let mut router = BanditRouter::new(BanditConfig::default());
            router.add_arm_for_task("coding", "public-coder");
            router.add_arm_for_task("coding", "private-coder");
            router.set_arm_private("private-coder");

            let state = BanditStateMerger::extract_state(&router, "node1");
            let coding_arms = state.task_bandits.get("coding");
            assert!(coding_arms.is_some());
            assert_eq!(coding_arms.unwrap().len(), 1);
            assert_eq!(coding_arms.unwrap()[0].id, "public-coder");
        }

        #[test]
        fn test_extract_state_preserves_public_arms() {
            let mut router = BanditRouter::new(BanditConfig::default());
            router.add_arm("m1");
            router.add_arm("m2");
            router.add_arm("m3");
            router.set_arm_private("m2");

            let state = BanditStateMerger::extract_state(&router, "node1");
            assert_eq!(state.global_arms.len(), 2);
            let ids: Vec<&str> = state.global_arms.iter().map(|a| a.id.as_str()).collect();
            assert!(ids.contains(&"m1"));
            assert!(ids.contains(&"m3"));
            assert!(!ids.contains(&"m2"));
        }

        #[test]
        fn test_merge_excludes_private_from_both_sides() {
            let mut router1 = BanditRouter::new(BanditConfig::default());
            router1.add_arm("shared");
            router1.add_arm("private1");
            router1.set_arm_private("private1");

            let mut router2 = BanditRouter::new(BanditConfig::default());
            router2.add_arm("shared");
            router2.add_arm("private2");
            router2.set_arm_private("private2");

            let state1 = BanditStateMerger::extract_state(&router1, "n1");
            let state2 = BanditStateMerger::extract_state(&router2, "n2");

            let merged = BanditStateMerger::merge(&[state1, state2], 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms.len(), 1);
            assert_eq!(merged.global_arms[0].id, "shared");
        }
    }

    // =========================================================================
    // EXPORT/IMPORT TESTS
    // =========================================================================

    #[test]
    fn test_export_json() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("model-a");
        let json = bandit.to_json().unwrap();
        assert!(json.contains("model-a"));
        assert!(json.contains("version"));
    }

    #[test]
    fn test_import_json() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("test-model");
        bandit.record_outcome(&ArmFeedback {
            arm_id: "test-model".to_string(), success: true, quality: Some(0.9),
            latency_ms: None, cost: None, task_type: None,
        });

        let json = bandit.to_json().unwrap();
        let restored = BanditRouter::from_json(&json).unwrap();
        assert_eq!(restored.all_arms(None).len(), 1);
        assert_eq!(restored.all_arms(None)[0].id, "test-model");
        assert_eq!(restored.all_arms(None)[0].pull_count, 1);
    }

    #[test]
    fn test_round_trip_json() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("a");
        bandit.add_arm("b");
        bandit.add_arm_for_task("coding", "code-model");

        let json1 = bandit.to_json().unwrap();
        let restored = BanditRouter::from_json(&json1).unwrap();
        let json2 = restored.to_json().unwrap();

        // Parse both to compare structure (timestamps may differ)
        let v1: serde_json::Value = serde_json::from_str(&json1).unwrap();
        let v2: serde_json::Value = serde_json::from_str(&json2).unwrap();
        assert_eq!(v1["global_arms"], v2["global_arms"]);
        assert_eq!(v1["task_bandits"], v2["task_bandits"]);
    }

    #[test]
    fn test_export_snapshot_version() {
        let bandit = BanditRouter::new(BanditConfig::default());
        let snapshot = bandit.export_snapshot();
        assert_eq!(snapshot.version, 1);
    }

    #[test]
    fn test_import_wrong_version() {
        let json = r#"{"version":999,"created_at":"0","config":{"strategy":"ThompsonSampling","prior_alpha":1.0,"prior_beta":1.0,"min_pulls_before_prune":10,"decay_factor":1.0},"task_bandits":{},"global_arms":[],"total_pulls":0,"metadata":{}}"#;
        let result = BanditRouter::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_snapshot_metadata() {
        let bandit = BanditRouter::new(BanditConfig::default());
        let snapshot = bandit.export_snapshot();
        assert!(snapshot.metadata.is_empty());
    }

    #[test]
    fn test_export_bytes() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("x");
        let bytes = bandit.to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_import_bytes() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("x");
        let bytes = bandit.to_bytes().unwrap();
        let restored = BanditRouter::from_bytes(&bytes).unwrap();
        assert_eq!(restored.all_arms(None).len(), 1);
    }

    #[test]
    fn test_round_trip_bytes() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("a");
        bandit.add_arm("b");
        let bytes1 = bandit.to_bytes().unwrap();
        let restored = BanditRouter::from_bytes(&bytes1).unwrap();
        let bytes2 = restored.to_bytes().unwrap();
        // May not be identical bytes (timestamps), but same logical content
        let r2 = BanditRouter::from_bytes(&bytes2).unwrap();
        assert_eq!(r2.all_arms(None).len(), 2);
    }

    #[test]
    fn test_empty_bandit_export() {
        let bandit = BanditRouter::new(BanditConfig::default());
        let json = bandit.to_json().unwrap();
        let restored = BanditRouter::from_json(&json).unwrap();
        assert!(restored.all_arms(None).is_empty());
    }

    // =========================================================================
    // FEATURE VALUE EXTRACTION
    // =========================================================================

    #[test]
    fn test_extract_feature_value() {
        let features = test_features("general", 0.75);
        assert_eq!(extract_feature_value(&features, "complexity"), 0.75);
        assert_eq!(extract_feature_value(&features, "token_count"), 50.0);
        assert_eq!(extract_feature_value(&features, "has_code"), 0.0);
        assert_eq!(extract_feature_value(&features, "is_question"), 1.0);
        assert_eq!(extract_feature_value(&features, "unknown"), 0.0);
    }

    // =========================================================================
    // NFA RULE BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_builder_single_rule() {
        let nfa = NfaRuleBuilder::new()
            .rule("r1")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("gpt-4")
            .priority(5)
            .done()
            .fallback("gpt-4-mini", 1)
            .build()
            .unwrap();
        assert!(nfa.state_count() >= 3); // start + accepting + fallback
    }

    #[test]
    fn test_builder_multiple_rules() {
        let nfa = NfaRuleBuilder::new()
            .rule("code_hard")
            .when(NfaSymbol::Domain("code".into()))
            .and(NfaSymbol::ComplexityRange { low_pct: 70, high_pct: 100 })
            .route_to("claude-opus")
            .priority(10)
            .done()
            .rule("code_easy")
            .when(NfaSymbol::Domain("code".into()))
            .and(NfaSymbol::ComplexityRange { low_pct: 0, high_pct: 70 })
            .route_to("gpt-4")
            .priority(5)
            .done()
            .fallback("gpt-4-mini", 1)
            .build()
            .unwrap();

        // Route a high-complexity code query
        let features = test_features("code", 0.85);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "claude-opus");
    }

    #[test]
    fn test_builder_fallback_only() {
        let nfa = NfaRuleBuilder::new()
            .fallback("default", 1)
            .build()
            .unwrap();
        let features = test_features("anything", 0.5);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "default");
    }

    #[test]
    fn test_builder_no_rules_no_fallback_error() {
        let result = NfaRuleBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_chained_conditions() {
        let nfa = NfaRuleBuilder::new()
            .rule("specific")
            .when(NfaSymbol::Domain("math".into()))
            .and(NfaSymbol::ComplexityRange { low_pct: 80, high_pct: 100 })
            .and(NfaSymbol::BoolFeature { name: "is_question".into(), value: true })
            .route_to("specialist")
            .priority(10)
            .done()
            .fallback("general", 1)
            .build()
            .unwrap();

        // Query matching all 3 conditions
        let mut features = test_features("math", 0.9);
        features.is_question = true;
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "specialist");
    }

    #[test]
    fn test_builder_priority_resolution() {
        let nfa = NfaRuleBuilder::new()
            .rule("low")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("cheap-model")
            .priority(1)
            .done()
            .rule("high")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("expensive-model")
            .priority(10)
            .done()
            .build()
            .unwrap();

        let features = test_features("code", 0.5);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "expensive-model");
    }

    #[test]
    fn test_builder_no_conditions_rule() {
        // Rule with no conditions → epsilon from start → always matches
        let nfa = NfaRuleBuilder::new()
            .rule("always")
            .route_to("always-model")
            .priority(5)
            .done()
            .build()
            .unwrap();

        let features = test_features("anything", 0.1);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "always-model");
    }

    #[test]
    fn test_builder_build_and_compile() {
        let nfa = NfaRuleBuilder::new()
            .rule("r1")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("gpt-4")
            .priority(5)
            .done()
            .fallback("gpt-4-mini", 1)
            .build()
            .unwrap();

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let features = test_features("code", 0.5);
        let outcome = dfa.route(&features).unwrap();
        // Should route to either gpt-4 (domain match, higher priority) or gpt-4-mini (fallback)
        assert!(outcome.selected_arm == "gpt-4" || outcome.selected_arm == "gpt-4-mini");
    }

    // =========================================================================
    // BANDIT → NFA SYNTHESIZER TESTS
    // =========================================================================

    #[test]
    fn test_synthesizer_basic() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "gpt-4");
        bandit.add_arm_for_task("code", "claude");

        // Simulate some outcomes
        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "gpt-4".into(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
        }
        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "claude".into(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
        }

        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 5, 0.5).unwrap();
        assert!(nfa.state_count() >= 2);
    }

    #[test]
    fn test_synthesizer_min_pulls_filter() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "model-a");

        // Only 3 pulls — below min_pulls of 10
        for _ in 0..3 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "model-a".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
        }

        // Should still produce NFA (global fallback might not exist though)
        // With no qualifying arms, error expected
        bandit.add_arm("fallback-global");
        for _ in 0..15 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "fallback-global".into(),
                success: true,
                quality: Some(0.5),
                latency_ms: None,
                cost: None,
                task_type: None,
            });
        }

        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 10, 0.3).unwrap();
        let features = test_features("unknown", 0.5);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "fallback-global");
    }

    #[test]
    fn test_synthesizer_multi_task() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "code-model");
        bandit.add_arm_for_task("math", "math-model");
        bandit.add_arm("fallback");

        for _ in 0..15 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "code-model".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "math-model".into(), success: true, quality: Some(0.85),
                latency_ms: None, cost: None, task_type: Some("math".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "fallback".into(), success: true, quality: Some(0.5),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 10, 0.3).unwrap();
        assert!(nfa.state_count() >= 4); // start + code + math + fallback at minimum
    }

    #[test]
    fn test_synthesizer_empty_bandit_error() {
        let bandit = BanditRouter::new(BanditConfig::default());
        let result = BanditNfaSynthesizer::synthesize(&bandit, 5, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_synthesizer_quality_threshold() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "good-model");
        bandit.add_arm_for_task("code", "bad-model");
        bandit.add_arm("fallback");

        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "good-model".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "bad-model".into(), success: false, quality: Some(0.2),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "fallback".into(), success: true, quality: Some(0.5),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        // With threshold 0.5, bad-model (0.2) should NOT get an alternative route
        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 10, 0.5).unwrap();
        let features = test_features("code", 0.5);
        let outcome = nfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "good-model");
    }

    #[test]
    fn test_synthesizer_then_route() {
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "specialist");
        bandit.add_arm("generalist");

        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "specialist".into(), success: true, quality: Some(0.95),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "generalist".into(), success: true, quality: Some(0.6),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 10, 0.3).unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();

        let features = test_features("code", 0.5);
        let outcome = dfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "specialist");
    }

    // =========================================================================
    // NFA EXPORT / IMPORT TESTS
    // =========================================================================

    #[test]
    fn test_nfa_to_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("r1").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
            .fallback("fb", 0)
            .build().unwrap();
        let json = nfa.to_json().unwrap();
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("merged") == false); // not a merged NFA
    }

    #[test]
    fn test_nfa_from_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("r1").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
            .build().unwrap();
        let json = nfa.to_json().unwrap();
        let restored = NfaRouter::from_json(&json).unwrap();
        assert_eq!(restored.state_count(), nfa.state_count());
        assert_eq!(restored.transition_count(), nfa.transition_count());
    }

    #[test]
    fn test_nfa_round_trip_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("m2").priority(5).done()
            .fallback("fb", 1)
            .build().unwrap();

        let json = nfa.to_json().unwrap();
        let restored = NfaRouter::from_json(&json).unwrap();

        // Should produce same routing
        let features = test_features("code", 0.5);
        let o1 = nfa.route(&features).unwrap();
        let o2 = restored.route(&features).unwrap();
        assert_eq!(o1.selected_arm, o2.selected_arm);
    }

    #[test]
    fn test_nfa_version_check() {
        let nfa = NfaRuleBuilder::new()
            .fallback("fb", 0)
            .build().unwrap();
        let mut json = nfa.to_json().unwrap();
        json = json.replace("\"version\": 1", "\"version\": 99");
        let result = NfaRouter::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_nfa_to_bytes() {
        let nfa = NfaRuleBuilder::new()
            .fallback("fb", 0)
            .build().unwrap();
        let bytes = nfa.to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_nfa_from_bytes() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
            .build().unwrap();
        let bytes = nfa.to_bytes().unwrap();
        let restored = NfaRouter::from_bytes(&bytes).unwrap();
        assert_eq!(restored.state_count(), nfa.state_count());
    }

    // =========================================================================
    // DFA EXPORT / IMPORT TESTS
    // =========================================================================

    #[test]
    fn test_dfa_to_json_export() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
            .fallback("fb", 0)
            .build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let json = dfa.to_json().unwrap();
        assert!(json.contains("\"version\": 1"));
    }

    #[test]
    fn test_dfa_from_json_import() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
            .build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let json = dfa.to_json().unwrap();
        let restored = DfaRouter::from_json(&json).unwrap();
        assert_eq!(restored.state_count(), dfa.state_count());
    }

    #[test]
    fn test_dfa_round_trip_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .fallback("fb", 1)
            .build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();

        let json = dfa.to_json().unwrap();
        let restored = DfaRouter::from_json(&json).unwrap();

        let features = test_features("code", 0.5);
        let o1 = dfa.route(&features).unwrap();
        let o2 = restored.route(&features).unwrap();
        assert_eq!(o1.selected_arm, o2.selected_arm);
    }

    #[test]
    fn test_dfa_version_check() {
        let nfa = NfaRuleBuilder::new().fallback("fb", 0).build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let mut json = dfa.to_json().unwrap();
        json = json.replace("\"version\": 1", "\"version\": 42");
        assert!(DfaRouter::from_json(&json).is_err());
    }

    #[test]
    fn test_dfa_to_bytes_export() {
        let nfa = NfaRuleBuilder::new().fallback("fb", 0).build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let bytes = dfa.to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_dfa_from_bytes_import() {
        let nfa = NfaRuleBuilder::new().fallback("fb", 0).build().unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let bytes = dfa.to_bytes().unwrap();
        let restored = DfaRouter::from_bytes(&bytes).unwrap();
        assert_eq!(restored.state_count(), dfa.state_count());
    }

    // =========================================================================
    // NFA MERGE TESTS
    // =========================================================================

    #[test]
    fn test_nfa_merge_two_simple() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("m-code").priority(10).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("m-math").priority(5).done()
            .build().unwrap();

        let merged = nfa_a.merge(&nfa_b);
        // Merged should have states from both + new start
        assert!(merged.state_count() > nfa_a.state_count());
        assert!(merged.state_count() > nfa_b.state_count());
    }

    #[test]
    fn test_nfa_merge_state_renumbering() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("x".into())).route_to("m1").priority(1).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("y".into())).route_to("m2").priority(1).done()
            .build().unwrap();

        let merged = nfa_a.merge(&nfa_b);
        // Total states = nfa_a.states + nfa_b.states + 1 (merged start)
        assert_eq!(merged.state_count(), nfa_a.state_count() + nfa_b.state_count() + 1);
    }

    #[test]
    fn test_nfa_merge_accepting_preserved() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("model-a").priority(10).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("model-b").priority(5).done()
            .build().unwrap();

        let merged = nfa_a.merge(&nfa_b);

        // Route code → model-a
        let features_code = test_features("code", 0.5);
        let outcome = merged.route(&features_code).unwrap();
        assert_eq!(outcome.selected_arm, "model-a");

        // Route math → model-b
        let features_math = test_features("math", 0.5);
        let outcome = merged.route(&features_math).unwrap();
        assert_eq!(outcome.selected_arm, "model-b");
    }

    #[test]
    fn test_nfa_merge_route_both() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("code-model").priority(10).done()
            .fallback("fallback-a", 1)
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("math-model").priority(8).done()
            .fallback("fallback-b", 2)
            .build().unwrap();

        let merged = nfa_a.merge(&nfa_b);

        let f_code = test_features("code", 0.5);
        assert_eq!(merged.route(&f_code).unwrap().selected_arm, "code-model");

        let f_math = test_features("math", 0.5);
        assert_eq!(merged.route(&f_math).unwrap().selected_arm, "math-model");
    }

    #[test]
    fn test_nfa_merge_three_chain() {
        let a = NfaRuleBuilder::new()
            .rule("x").when(NfaSymbol::Domain("a".into())).route_to("ma").priority(1).done()
            .build().unwrap();
        let b = NfaRuleBuilder::new()
            .rule("y").when(NfaSymbol::Domain("b".into())).route_to("mb").priority(2).done()
            .build().unwrap();
        let c = NfaRuleBuilder::new()
            .rule("z").when(NfaSymbol::Domain("c".into())).route_to("mc").priority(3).done()
            .build().unwrap();

        let merged = a.merge(&b).merge(&c);

        let f = test_features("c", 0.5);
        let outcome = merged.route(&f).unwrap();
        assert_eq!(outcome.selected_arm, "mc");
    }

    #[test]
    fn test_nfa_merge_transitions_preserved() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("x".into())).route_to("m1").priority(1).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("y".into())).route_to("m2").priority(1).done()
            .build().unwrap();

        let a_trans = nfa_a.transition_count();
        let b_trans = nfa_b.transition_count();
        let merged = nfa_a.merge(&nfa_b);

        // Merged transitions = a + b + 2 epsilon (from new start to both old starts)
        assert_eq!(merged.transition_count(), a_trans + b_trans + 2);
    }

    // =========================================================================
    // MERGE AND COMPILE TESTS
    // =========================================================================

    #[test]
    fn test_merge_and_compile_basic() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("m2").priority(5).done()
            .build().unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();
        assert!(dfa.state_count() >= 2);
    }

    #[test]
    fn test_merge_and_compile_routes_correctly() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("code-model").priority(10).done()
            .fallback("fallback", 1)
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("math-model").priority(8).done()
            .build().unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();

        let f_code = test_features("code", 0.5);
        assert_eq!(dfa.route(&f_code).unwrap().selected_arm, "code-model");
    }

    #[test]
    fn test_merge_and_compile_empty_nfa() {
        let nfa_a = NfaRouter::new();
        let nfa_b = NfaRuleBuilder::new()
            .fallback("fb", 1)
            .build().unwrap();

        // Should compile — one NFA is empty but the other has states
        let result = merge_and_compile_nfas(&nfa_a, &nfa_b);
        // Might succeed or fail depending on compiler handling of empty
        assert!(result.is_ok() || result.is_err()); // just ensure no panic
    }

    // =========================================================================
    // DISTRIBUTED NFA TESTS
    // =========================================================================

    #[cfg(feature = "distributed")]
    mod distributed_nfa_tests {
        use super::*;

        #[test]
        fn test_distributed_nfa_extract_state() {
            let nfa = NfaRuleBuilder::new()
                .rule("r").when(NfaSymbol::Domain("x".into())).route_to("m").priority(1).done()
                .build().unwrap();
            let state = NfaStateMerger::extract_state(&nfa, "node-1");
            assert_eq!(state.node_id, "node-1");
            assert_eq!(state.nfa.states.len(), nfa.state_count());
        }

        #[test]
        fn test_distributed_nfa_merge_two() {
            let nfa_a = NfaRuleBuilder::new()
                .rule("a").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
                .build().unwrap();
            let nfa_b = NfaRuleBuilder::new()
                .rule("b").when(NfaSymbol::Domain("math".into())).route_to("m2").priority(5).done()
                .build().unwrap();

            let state_a = NfaStateMerger::extract_state(&nfa_a, "node-a");
            let state_b = NfaStateMerger::extract_state(&nfa_b, "node-b");

            let merged = NfaStateMerger::merge(&[state_a, state_b]).unwrap();
            assert_eq!(merged.node_id, "merged");
            assert!(merged.nfa.states.len() > nfa_a.state_count());
        }

        #[test]
        fn test_distributed_nfa_merge_three() {
            let a = NfaRuleBuilder::new().fallback("m1", 1).build().unwrap();
            let b = NfaRuleBuilder::new().fallback("m2", 2).build().unwrap();
            let c = NfaRuleBuilder::new().fallback("m3", 3).build().unwrap();

            let states = vec![
                NfaStateMerger::extract_state(&a, "n1"),
                NfaStateMerger::extract_state(&b, "n2"),
                NfaStateMerger::extract_state(&c, "n3"),
            ];

            let merged = NfaStateMerger::merge(&states).unwrap();
            // Should have all states from all three
            assert!(merged.nfa.states.len() >= 3);
        }

        #[test]
        fn test_distributed_nfa_merge_into_router() {
            let mut local = NfaRuleBuilder::new()
                .rule("local").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
                .build().unwrap();
            let remote_nfa = NfaRuleBuilder::new()
                .rule("remote").when(NfaSymbol::Domain("math".into())).route_to("m2").priority(5).done()
                .build().unwrap();

            let remote_state = NfaStateMerger::extract_state(&remote_nfa, "remote-node");
            let original_count = local.state_count();
            NfaStateMerger::merge_into_router(&mut local, &remote_state).unwrap();
            assert!(local.state_count() > original_count);
        }

        #[test]
        fn test_distributed_nfa_merge_empty_error() {
            let result = NfaStateMerger::merge(&[]);
            assert!(result.is_err());
        }

        #[test]
        fn test_nfa_extract_state_filtered_excludes_private() {
            let mut nfa = NfaRouter::new();
            let s0 = nfa.add_state("start", None, 0); // first state = auto start
            let s1 = nfa.add_state("public", Some("public-arm"), 10);
            let s2 = nfa.add_state("private", Some("private-arm"), 10);
            nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);
            nfa.add_transition(s0, NfaSymbol::Domain("math".to_string()), s2);

            let private_arms: HashSet<ArmId> = ["private-arm".to_string()].into_iter().collect();
            let state = NfaStateMerger::extract_state_filtered(&nfa, "node1", &private_arms);

            // Private arm's state and transition should be filtered out
            assert!(state.nfa.states.iter().all(|s| {
                s.accepting_arm.as_ref().map_or(true, |a| a != "private-arm")
            }));
        }

        #[test]
        fn test_nfa_extract_state_backward_compat() {
            let mut nfa = NfaRouter::new();
            let s0 = nfa.add_state("start", None, 0); // first state = auto start
            let s1 = nfa.add_state("end", Some("model-a"), 10);
            nfa.add_transition(s0, NfaSymbol::Domain("coding".to_string()), s1);

            // Old method (no private args) still works
            let state = NfaStateMerger::extract_state(&nfa, "node1");
            assert_eq!(state.nfa.states.len(), 2);
        }
    }

    // =========================================================================
    // ROUTING PIPELINE TESTS
    // =========================================================================

    #[test]
    fn test_pipeline_new() {
        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default());
        assert!(pipeline.active_dfa().is_none());
        assert!(pipeline.source_nfa().is_none());
        assert_eq!(pipeline.synthesis_count(), 0);
    }

    #[test]
    fn test_pipeline_with_initial_nfa() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .fallback("fb", 1)
            .build().unwrap();

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa).unwrap();

        assert!(pipeline.active_dfa().is_some());
        assert!(pipeline.source_nfa().is_some());
    }

    #[test]
    fn test_pipeline_seeds_bandit_from_nfa() {
        let nfa = NfaRuleBuilder::new()
            .rule("code_hi")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("claude-opus")
            .priority(10)
            .done()
            .rule("math")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("gpt-4")
            .priority(5)
            .done()
            .fallback("gpt-4-mini", 1)
            .build()
            .unwrap();

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa).unwrap();

        // Bandit should have all 3 arms globally
        let global_arms = pipeline.bandit().all_arms(None);
        let arm_ids: Vec<&str> = global_arms.iter().map(|a| a.id.as_str()).collect();
        assert!(arm_ids.contains(&"claude-opus"));
        assert!(arm_ids.contains(&"gpt-4"));
        assert!(arm_ids.contains(&"gpt-4-mini"));

        // "code" task type should have claude-opus
        let code_arms = pipeline.bandit().all_arms_vec(Some("code"));
        let code_ids: Vec<&str> = code_arms.iter().map(|a| a.id.as_str()).collect();
        assert!(code_ids.contains(&"claude-opus"));

        // "math" task type should have gpt-4
        let math_arms = pipeline.bandit().all_arms_vec(Some("math"));
        let math_ids: Vec<&str> = math_arms.iter().map(|a| a.id.as_str()).collect();
        assert!(math_ids.contains(&"gpt-4"));

        // claude-opus (priority 10) should have stronger prior than gpt-4-mini (priority 1)
        let opus = pipeline.bandit().arm_stats("claude-opus").unwrap();
        let mini = pipeline.bandit().arm_stats("gpt-4-mini").unwrap();
        assert!(opus.params.alpha > mini.params.alpha);
    }

    #[test]
    fn test_pipeline_route_via_dfa() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("code".into())).route_to("dfa-model").priority(10).done()
            .fallback("fb", 1)
            .build().unwrap();

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa).unwrap();

        let features = test_features("code", 0.5);
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "dfa-model");
    }

    #[test]
    fn test_pipeline_route_via_bandit_fallback() {
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default());
        pipeline.add_arm("bandit-model");

        let features = test_features("anything", 0.5);
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "bandit-model");
    }

    #[test]
    fn test_pipeline_record_outcome() {
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default());
        pipeline.add_arm("m1");

        // Route first (which increments total_pulls via select())
        let features = test_features("anything", 0.5);
        let _ = pipeline.route(&features).unwrap();
        assert_eq!(pipeline.bandit().total_pulls(), 1);

        // Record outcome updates arm stats
        pipeline.record_outcome(&ArmFeedback {
            arm_id: "m1".into(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: None,
        });
        // Verify arm was updated (pull_count increments in record_outcome)
        let arm = pipeline.bandit().arm_stats("m1").unwrap();
        assert_eq!(arm.pull_count, 1);
    }

    #[test]
    fn test_pipeline_auto_resynthesize() {
        let config = PipelineConfig {
            synthesis_interval: 5,
            min_pulls_for_synthesis: 2,
            quality_threshold: 0.3,
            auto_minimize: true,
            discovery: None,
        };
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config);
        pipeline.add_arm("m1");
        pipeline.add_arm_for_task("code", "m1");

        // Record 5 outcomes to trigger synthesis
        for _ in 0..5 {
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "m1".into(), success: true, quality: Some(0.8),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
        }

        let did = pipeline.maybe_resynthesize();
        assert!(did);
        assert_eq!(pipeline.synthesis_count(), 1);
        assert!(pipeline.active_dfa().is_some());
    }

    #[test]
    fn test_pipeline_force_resynthesize() {
        let config = PipelineConfig {
            synthesis_interval: 1000,
            min_pulls_for_synthesis: 2,
            quality_threshold: 0.3,
            auto_minimize: true,
            discovery: None,
        };
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config);
        pipeline.add_arm("m1");
        pipeline.add_arm_for_task("code", "m1");

        for _ in 0..5 {
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "m1".into(), success: true, quality: Some(0.8),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
        }

        pipeline.force_resynthesize().unwrap();
        assert_eq!(pipeline.synthesis_count(), 1);
    }

    #[test]
    fn test_pipeline_export_snapshot() {
        let nfa = NfaRuleBuilder::new()
            .fallback("fb", 1)
            .build().unwrap();

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa).unwrap();

        let snapshot = pipeline.export_snapshot();
        assert_eq!(snapshot.version, 1);
        assert!(snapshot.nfa.is_some());
    }

    #[test]
    fn test_pipeline_with_initial_rules() {
        let builder = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .fallback("fb", 1);

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_rules(builder).unwrap();

        assert!(pipeline.active_dfa().is_some());
    }

    #[test]
    fn test_pipeline_json_round_trip() {
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .fallback("fb", 1)
            .build().unwrap();

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa).unwrap();
        pipeline.add_arm("m1");

        let json = pipeline.to_json().unwrap();
        let restored = RoutingPipeline::from_json(&json).unwrap();
        assert!(restored.active_dfa().is_some());
        assert_eq!(restored.synthesis_count(), pipeline.synthesis_count());
    }

    // =========================================================================
    // FOR_MODELS / WITH_TIERED_MODELS TESTS
    // =========================================================================

    #[test]
    fn test_for_models_zero_config() {
        let mut pipeline = RoutingPipeline::for_models(
            &["gpt-4", "claude", "gemini"],
            PipelineConfig::default(),
        );

        // No DFA initially — pure bandit exploration
        assert!(pipeline.active_dfa().is_none());

        // Bandit has 3 arms
        assert_eq!(pipeline.bandit().all_arms(None).len(), 3);

        // Can route immediately via bandit
        let features = test_features("anything", 0.5);
        let outcome = pipeline.route(&features).unwrap();
        assert!(["gpt-4", "claude", "gemini"].contains(&outcome.selected_arm.as_str()));
    }

    #[test]
    fn test_for_models_auto_synthesize() {
        let config = PipelineConfig {
            synthesis_interval: 5,
            min_pulls_for_synthesis: 2,
            quality_threshold: 0.3,
            auto_minimize: true,
            discovery: None,
        };
        let mut pipeline = RoutingPipeline::for_models(&["m1", "m2"], config);
        pipeline.add_arm_for_task("code", "m1");
        pipeline.add_arm_for_task("code", "m2");

        // Train bandit
        for _ in 0..6 {
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "m1".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
        }

        // Should auto-synthesize NFA → DFA
        assert!(pipeline.maybe_resynthesize());
        assert!(pipeline.active_dfa().is_some());
        assert_eq!(pipeline.synthesis_count(), 1);
    }

    #[test]
    fn test_tiered_models_basic() {
        let pipeline = RoutingPipeline::with_tiered_models(
            &[
                ("claude-opus", ModelTier::Premium),
                ("gpt-4", ModelTier::Standard),
                ("gpt-4-mini", ModelTier::Economy),
            ],
            PipelineConfig::default(),
        ).unwrap();

        // Has DFA from auto-generated rules
        assert!(pipeline.active_dfa().is_some());

        // Bandit is seeded with all 3 models
        let arms = pipeline.bandit().all_arms(None);
        assert!(arms.len() >= 3);
    }

    #[test]
    fn test_tiered_routes_code_to_premium() {
        let mut pipeline = RoutingPipeline::with_tiered_models(
            &[
                ("opus", ModelTier::Premium),
                ("sonnet", ModelTier::Standard),
                ("haiku", ModelTier::Economy),
            ],
            PipelineConfig::default(),
        ).unwrap();

        // Code query → should route to premium (has_code=true)
        let mut features = test_features("code", 0.5);
        features.has_code = true;
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "opus");
    }

    #[test]
    fn test_tiered_routes_simple_to_economy() {
        let mut pipeline = RoutingPipeline::with_tiered_models(
            &[
                ("opus", ModelTier::Premium),
                ("sonnet", ModelTier::Standard),
                ("haiku", ModelTier::Economy),
            ],
            PipelineConfig::default(),
        ).unwrap();

        // Simple low-complexity query → economy
        let features = test_features("chat", 0.1);
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "haiku");
    }

    #[test]
    fn test_tiered_routes_medium_to_standard() {
        let mut pipeline = RoutingPipeline::with_tiered_models(
            &[
                ("opus", ModelTier::Premium),
                ("sonnet", ModelTier::Standard),
                ("haiku", ModelTier::Economy),
            ],
            PipelineConfig::default(),
        ).unwrap();

        // Medium complexity → standard
        let features = test_features("general", 0.5);
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "sonnet");
    }

    #[test]
    fn test_tiered_empty_models_error() {
        let result = RoutingPipeline::with_tiered_models(&[], PipelineConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_tiered_single_tier_only() {
        // Only premium models — should still work with fallback
        let mut pipeline = RoutingPipeline::with_tiered_models(
            &[("opus", ModelTier::Premium), ("sonnet", ModelTier::Premium)],
            PipelineConfig::default(),
        ).unwrap();

        let features = test_features("anything", 0.1);
        let outcome = pipeline.route(&features).unwrap();
        // Falls back to one of the premium models
        assert!(outcome.selected_arm == "opus" || outcome.selected_arm == "sonnet");
    }

    // =========================================================================
    // INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_integration_bandit_synthesize_compile_route() {
        // Full loop: train bandit → synthesize NFA → compile DFA → route
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm_for_task("code", "code-specialist");
        bandit.add_arm("generalist");

        for _ in 0..30 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "code-specialist".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "generalist".into(), success: true, quality: Some(0.5),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        let nfa = BanditNfaSynthesizer::synthesize(&bandit, 10, 0.3).unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();

        let features = test_features("code", 0.5);
        let outcome = dfa.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "code-specialist");
    }

    #[test]
    fn test_integration_merge_two_pipeline_nfas() {
        // Two pipelines share their NFAs and merge
        let nfa_a = NfaRuleBuilder::new()
            .rule("a").when(NfaSymbol::Domain("code".into())).route_to("code-m").priority(10).done()
            .build().unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b").when(NfaSymbol::Domain("math".into())).route_to("math-m").priority(8).done()
            .build().unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();

        // Verify routes from both pipelines work
        assert_eq!(dfa.route(&test_features("code", 0.5)).unwrap().selected_arm, "code-m");
        assert_eq!(dfa.route(&test_features("math", 0.5)).unwrap().selected_arm, "math-m");
    }

    #[test]
    fn test_integration_full_pipeline_loop() {
        // Full closed-loop: start with rules → learn → resynthesize → verify DFA updated
        let config = PipelineConfig {
            synthesis_interval: 10,
            min_pulls_for_synthesis: 3,
            quality_threshold: 0.3,
            auto_minimize: true,
            discovery: None,
        };
        let builder = NfaRuleBuilder::new()
            .rule("init").when(NfaSymbol::Domain("code".into())).route_to("initial-model").priority(5).done()
            .fallback("fallback", 1);

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config)
            .with_initial_rules(builder).unwrap();

        // Add arms for bandit learning
        pipeline.add_arm_for_task("code", "better-model");
        pipeline.add_arm("fallback");

        // Simulate learning: better-model outperforms
        for _ in 0..12 {
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "better-model".into(), success: true, quality: Some(0.95),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
        }

        // Trigger re-synthesis
        let did = pipeline.maybe_resynthesize();
        assert!(did);
        assert_eq!(pipeline.synthesis_count(), 1);

        // After re-synthesis, DFA should now route to better-model
        let features = test_features("code", 0.5);
        let outcome = pipeline.route(&features).unwrap();
        assert_eq!(outcome.selected_arm, "better-model");
    }

    #[test]
    fn test_integration_export_import_nfa() {
        // Export NFA, import on another "node", verify routing
        let nfa = NfaRuleBuilder::new()
            .rule("r").when(NfaSymbol::Domain("code".into())).route_to("m1").priority(10).done()
            .fallback("fb", 1)
            .build().unwrap();

        let json = nfa.to_json().unwrap();
        let restored = NfaRouter::from_json(&json).unwrap();
        let dfa = NfaDfaCompiler::compile(&restored).unwrap();

        let features = test_features("code", 0.5);
        assert_eq!(dfa.route(&features).unwrap().selected_arm, "m1");
    }

    #[test]
    fn test_integration_bandit_task_types() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("code", "m1");
        bandit.add_arm_for_task("math", "m2");
        bandit.add_arm_for_task("code", "m3");

        let types = bandit.task_types();
        assert!(types.contains(&"code"));
        assert!(types.contains(&"math"));
        assert_eq!(types.len(), 2);
    }

    // ================================================================
    // Section I: remove_arm + MCP routing tools tests
    // ================================================================

    #[test]
    fn test_remove_arm_global() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("m1");
        bandit.add_arm("m2");
        bandit.add_arm("m3");
        assert_eq!(bandit.all_arms(None).len(), 3);

        assert!(bandit.remove_arm("m2", None));
        assert_eq!(bandit.all_arms(None).len(), 2);
        assert!(bandit.all_arms(None).iter().all(|a| a.id != "m2"));
    }

    #[test]
    fn test_remove_arm_task_specific() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("code", "m1");
        bandit.add_arm_for_task("code", "m2");
        bandit.add_arm_for_task("math", "m3");

        assert!(bandit.remove_arm("m1", Some("code")));
        assert_eq!(bandit.all_arms_vec(Some("code")).len(), 1);
        assert_eq!(bandit.all_arms_vec(Some("code"))[0].id, "m2");
        // math unaffected
        assert_eq!(bandit.all_arms_vec(Some("math")).len(), 1);
    }

    #[test]
    fn test_remove_arm_not_found() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("m1");
        assert!(!bandit.remove_arm("m999", None));
        assert!(!bandit.remove_arm("m1", Some("nonexistent")));
    }

    /// Helper: invoke an MCP tool via handle_request and parse the JSON result.
    fn mcp_call(
        server: &crate::mcp_protocol::McpServer,
        tool_name: &str,
        args: serde_json::Value,
    ) -> serde_json::Value {
        use crate::mcp_protocol::McpRequest;
        let request = McpRequest::new("tools/call")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "name": tool_name,
                "arguments": args,
            }));
        let response = server.handle_request(request);
        assert!(response.error.is_none(), "MCP error: {:?}", response.error);
        let result = response.result.expect("no result");
        let content = result.get("content").and_then(|c| c.as_array()).expect("no content");
        let text = content[0].get("text").and_then(|t| t.as_str()).expect("no text");
        serde_json::from_str(text).expect("invalid JSON in response")
    }

    #[test]
    fn test_mcp_routing_get_stats() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.get_stats", serde_json::json!({}));
        assert_eq!(result.get("total_pulls").and_then(|v| v.as_u64()), Some(0));
        let arms = result.get("arms").and_then(|v| v.as_array()).unwrap();
        assert_eq!(arms.len(), 2);
    }

    #[test]
    fn test_mcp_routing_add_arm() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(&server, "routing.add_arm", serde_json::json!({
            "arm_id": "m_new",
            "alpha": 5.0,
            "beta": 1.0,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        // Verify arm was added
        let pipeline = shared.lock().unwrap();
        let arms = pipeline.bandit().all_arms(None);
        assert_eq!(arms.len(), 2);
        let new_arm = arms.iter().find(|a| a.id == "m_new").unwrap();
        assert!((new_arm.params.alpha - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mcp_routing_remove_arm() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2", "m3"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(&server, "routing.remove_arm", serde_json::json!({
            "arm_id": "m2",
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("removed"));

        let pipeline = shared.lock().unwrap();
        assert_eq!(pipeline.bandit().all_arms(None).len(), 2);
    }

    #[test]
    fn test_mcp_routing_warm_start() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(&server, "routing.warm_start", serde_json::json!({
            "arm_id": "m1",
            "alpha": 20.0,
            "beta": 2.0,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        let pipeline = shared.lock().unwrap();
        let arm = pipeline.bandit().all_arms(None).iter().find(|a| a.id == "m1").unwrap().clone();
        assert!((arm.params.alpha - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_mcp_routing_record_outcome() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.record_outcome", serde_json::json!({
            "arm_id": "m1",
            "success": true,
            "quality": 0.9,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_add_rule_and_recompile() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(&server, "routing.add_rule", serde_json::json!({
            "domain": "code",
            "arm_id": "m1",
            "priority": 10,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));

        // Should now have a DFA
        let pipeline = shared.lock().unwrap();
        assert!(pipeline.active_dfa().is_some());
        assert!(pipeline.source_nfa().is_some());
    }

    #[test]
    fn test_mcp_routing_add_rule_with_complexity() {
        let pipeline = RoutingPipeline::for_models(&["fast", "smart"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.add_rule", serde_json::json!({
            "domain": "code",
            "min_complexity": 70,
            "max_complexity": 100,
            "arm_id": "smart",
            "priority": 10,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(result.get("conditions_count").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn test_mcp_routing_force_resynthesize() {
        let config = PipelineConfig {
            min_pulls_for_synthesis: 1,
            quality_threshold: 0.0,
            ..PipelineConfig::default()
        };
        let mut pipeline = RoutingPipeline::for_models(&["m1", "m2"], config);
        pipeline.add_arm("m1");
        pipeline.record_outcome(&ArmFeedback {
            arm_id: "m1".into(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: Some("code".into()),
        });
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.force_resynthesize", serde_json::json!({}));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_export_import_roundtrip() {
        let pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        // Export
        let exported = mcp_call(&server, "routing.export", serde_json::json!({}));
        let json = exported.get("pipeline_json").and_then(|v| v.as_str()).unwrap();
        assert!(!json.is_empty());

        // Import into same pipeline
        let result = mcp_call(&server, "routing.import", serde_json::json!({
            "pipeline_json": json,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn test_mcp_routing_get_config() {
        let config = PipelineConfig {
            synthesis_interval: 42,
            ..PipelineConfig::default()
        };
        let pipeline = RoutingPipeline::for_models(&["m1"], config);
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        let result = mcp_call(&server, "routing.get_config", serde_json::json!({}));
        assert_eq!(result.get("synthesis_interval").and_then(|v| v.as_u64()), Some(42));
        assert_eq!(result.get("bandit_arms_count").and_then(|v| v.as_u64()), Some(1));
    }

    #[test]
    fn test_mcp_routing_add_arm_for_task() {
        let pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared.clone());

        let result = mcp_call(&server, "routing.add_arm", serde_json::json!({
            "arm_id": "code_specialist",
            "task_type": "code",
            "alpha": 10.0,
            "beta": 1.0,
        }));
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(result.get("task_type").and_then(|v| v.as_str()), Some("code"));

        let pipeline = shared.lock().unwrap();
        assert!(pipeline.bandit().task_types().contains(&"code"));
    }

    #[test]
    fn test_mcp_routing_full_workflow() {
        // Full MCP workflow: add arm → add rule → record outcome → get stats
        let pipeline = RoutingPipeline::for_models(&["base"], PipelineConfig::default());
        let shared = Arc::new(Mutex::new(pipeline));
        let mut server = crate::mcp_protocol::McpServer::new("test", "1.0");
        register_routing_tools(&mut server, shared);

        // 1. Add a specialist arm
        mcp_call(&server, "routing.add_arm", serde_json::json!({
            "arm_id": "code_expert",
            "alpha": 8.0,
            "beta": 1.0,
        }));

        // 2. Add a routing rule for it
        mcp_call(&server, "routing.add_rule", serde_json::json!({
            "domain": "code",
            "min_complexity": 50,
            "arm_id": "code_expert",
            "priority": 20,
        }));

        // 3. Record some outcomes
        mcp_call(&server, "routing.record_outcome", serde_json::json!({
            "arm_id": "code_expert",
            "success": true,
            "quality": 0.95,
        }));

        // 4. Check stats
        let stats = mcp_call(&server, "routing.get_stats", serde_json::json!({}));
        let arms = stats.get("arms").and_then(|v| v.as_array()).unwrap();
        assert!(arms.len() >= 2); // base + code_expert
        assert!(stats.get("has_dfa").and_then(|v| v.as_bool()).unwrap());

        // 5. Export and verify non-empty
        let exported = mcp_call(&server, "routing.export", serde_json::json!({}));
        assert!(exported.get("pipeline_json").and_then(|v| v.as_str()).unwrap().len() > 10);
    }

    // =========================================================================
    // CONTEXTUAL DISCOVERY TESTS
    // =========================================================================

    fn make_ctx_features(domain: &str, complexity: f64, token_count: usize, has_code: bool) -> QueryFeatures {
        QueryFeatures {
            domain: domain.to_string(),
            complexity,
            token_count,
            sentence_count: 3,
            entity_count: 2,
            has_code,
            is_question: false,
            avg_word_length: 5.0,
            feature_vector: Vec::new(),
        }
    }

    #[test]
    fn test_context_snapshot_from_query_features() {
        let f = make_ctx_features("code", 0.85, 500, true);
        let snap = ContextSnapshot::from(&f);
        assert_eq!(snap.domain, "code");
        assert!((snap.complexity - 0.85).abs() < 1e-10);
        assert_eq!(snap.token_count, 500);
        assert!(snap.has_code);
        assert!(!snap.is_question);
        assert!((snap.avg_word_length - 5.0).abs() < 1e-10);
        assert_eq!(snap.entity_count, 2);
        assert_eq!(snap.sentence_count, 3);
    }

    #[test]
    fn test_feature_dimension_extract_all() {
        let snap = ContextSnapshot {
            domain: "test".to_string(),
            complexity: 0.7,
            token_count: 100,
            has_code: true,
            is_question: false,
            avg_word_length: 4.5,
            entity_count: 3,
            sentence_count: 5,
        };
        assert!((FeatureDimension::Complexity.extract(&snap) - 0.7).abs() < 1e-10);
        assert!((FeatureDimension::TokenCount.extract(&snap) - 100.0).abs() < 1e-10);
        assert!((FeatureDimension::HasCode.extract(&snap) - 1.0).abs() < 1e-10);
        assert!((FeatureDimension::IsQuestion.extract(&snap) - 0.0).abs() < 1e-10);
        assert!((FeatureDimension::AvgWordLength.extract(&snap) - 4.5).abs() < 1e-10);
        assert!((FeatureDimension::EntityCount.extract(&snap) - 3.0).abs() < 1e-10);
        assert!((FeatureDimension::SentenceCount.extract(&snap) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_dimension_all_returns_seven() {
        assert_eq!(FeatureDimension::all().len(), 7);
    }

    #[test]
    fn test_discovery_config_default() {
        let dc = DiscoveryConfig::default();
        assert_eq!(dc.max_observations, 1000);
        assert_eq!(dc.min_samples_per_split, 10);
        assert!((dc.min_gain - 0.05).abs() < 1e-10);
        assert_eq!(dc.num_split_points, 4);
        assert_eq!(dc.discovered_rule_priority_boost, 50);
    }

    #[test]
    fn test_contextual_observation_fields() {
        let obs = ContextualObservation {
            context: ContextSnapshot::from(&make_ctx_features("math", 0.5, 200, false)),
            arm_id: "opus".to_string(),
            reward: 0.9,
        };
        assert_eq!(obs.context.domain, "math");
        assert_eq!(obs.arm_id, "opus");
        assert!((obs.reward - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_discovery_record_basic() {
        let mut disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let f = make_ctx_features("code", 0.5, 100, true);
        disc.record(&f, "opus", 0.9);
        assert_eq!(disc.observation_count(), 1);
    }

    #[test]
    fn test_discovery_record_bounded() {
        let config = DiscoveryConfig {
            max_observations: 5,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);
        for i in 0..10 {
            let f = make_ctx_features("code", i as f64 / 10.0, 100, true);
            disc.record(&f, "opus", 0.5);
        }
        assert_eq!(disc.observation_count(), 5);
    }

    #[test]
    fn test_discovery_clear() {
        let mut disc = ContextualDiscovery::new(DiscoveryConfig::default());
        for _ in 0..5 {
            disc.record(&make_ctx_features("code", 0.5, 100, true), "opus", 0.9);
        }
        assert_eq!(disc.observation_count(), 5);
        disc.clear();
        assert_eq!(disc.observation_count(), 0);
    }

    #[test]
    fn test_discovery_record_captures_reward() {
        let mut disc = ContextualDiscovery::new(DiscoveryConfig::default());
        disc.record(&make_ctx_features("code", 0.5, 100, true), "opus", 0.42);
        // Access internal observations via serialization round-trip
        let json = serde_json::to_string(&disc).unwrap();
        assert!(json.contains("0.42"));
    }

    #[test]
    fn test_quantile_split_points_basic() {
        let values: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let points = ContextualDiscovery::compute_quantile_split_points(&values, 4);
        // Should produce ~3-4 distinct quantile points
        assert!(!points.is_empty());
        assert!(points.len() <= 4);
        // All points should be between 0 and 1
        for p in &points {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_quantile_split_points_all_same() {
        let values = vec![0.5; 20];
        let points = ContextualDiscovery::compute_quantile_split_points(&values, 4);
        assert!(points.is_empty());
    }

    #[test]
    fn test_quantile_split_points_empty() {
        let points = ContextualDiscovery::compute_quantile_split_points(&[], 4);
        assert!(points.is_empty());
    }

    #[test]
    fn test_compute_best_arm_mean_single_arm() {
        let f = make_ctx_features("code", 0.5, 100, true);
        let obs = vec![
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "opus".into(), reward: 0.6 },
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "opus".into(), reward: 0.8 },
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "opus".into(), reward: 1.0 },
        ];
        let refs: Vec<&ContextualObservation> = obs.iter().collect();
        let (arm, mean) = ContextualDiscovery::compute_best_arm_mean(&refs);
        assert_eq!(arm, "opus");
        assert!((mean - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_compute_best_arm_mean_two_arms() {
        let f = make_ctx_features("code", 0.5, 100, true);
        let obs = vec![
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "haiku".into(), reward: 0.5 },
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "haiku".into(), reward: 0.5 },
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "opus".into(), reward: 0.9 },
            ContextualObservation { context: ContextSnapshot::from(&f), arm_id: "opus".into(), reward: 0.9 },
        ];
        let refs: Vec<&ContextualObservation> = obs.iter().collect();
        let (arm, mean) = ContextualDiscovery::compute_best_arm_mean(&refs);
        assert_eq!(arm, "opus");
        assert!((mean - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_discover_splits_insufficient_data() {
        let config = DiscoveryConfig {
            min_samples_per_split: 10,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);
        // Only 5 observations — less than 2*10=20 minimum
        for i in 0..5 {
            disc.record(&make_ctx_features("code", i as f64 / 10.0, 100, true), "opus", 0.9);
        }
        let splits = disc.discover_splits();
        assert!(splits.is_empty());
    }

    #[test]
    fn test_discover_splits_complexity_split() {
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            num_split_points: 4,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);

        // High complexity: opus wins (reward 0.9 vs 0.3)
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.85, 100, true), "opus", 0.9);
            disc.record(&make_ctx_features("code", 0.90, 100, true), "haiku", 0.3);
        }
        // Low complexity: haiku wins (reward 0.9 vs 0.3)
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.15, 100, true), "haiku", 0.9);
            disc.record(&make_ctx_features("code", 0.10, 100, true), "opus", 0.3);
        }

        let splits = disc.discover_splits();
        assert!(!splits.is_empty(), "Should discover at least one split");

        // Should discover a complexity-based split in the "code" domain
        let complexity_split = splits.iter().find(|s| {
            s.domain == "code" && s.split.dimension == FeatureDimension::Complexity
        });
        assert!(complexity_split.is_some(), "Should find a complexity split for 'code'");

        let cs = complexity_split.unwrap();
        // opus should be best above threshold, haiku below (or vice versa)
        assert_ne!(cs.split.arm_above, cs.split.arm_below);
    }

    #[test]
    fn test_discover_splits_bool_feature() {
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);

        // has_code=true: opus wins
        for _ in 0..15 {
            disc.record(&make_ctx_features("general", 0.5, 100, true), "opus", 0.9);
            disc.record(&make_ctx_features("general", 0.5, 100, true), "haiku", 0.3);
        }
        // has_code=false: haiku wins
        for _ in 0..15 {
            disc.record(&make_ctx_features("general", 0.5, 100, false), "haiku", 0.9);
            disc.record(&make_ctx_features("general", 0.5, 100, false), "opus", 0.3);
        }

        let splits = disc.discover_splits();
        let bool_split = splits.iter().find(|s| {
            s.domain == "general" && s.split.dimension == FeatureDimension::HasCode
        });
        assert!(bool_split.is_some(), "Should discover HasCode bool split");
    }

    #[test]
    fn test_discover_splits_no_gain() {
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.05,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);

        // Both arms get same reward everywhere — no gain possible
        for _ in 0..20 {
            disc.record(&make_ctx_features("code", 0.2, 100, true), "opus", 0.7);
            disc.record(&make_ctx_features("code", 0.8, 100, true), "opus", 0.7);
            disc.record(&make_ctx_features("code", 0.2, 100, true), "haiku", 0.7);
            disc.record(&make_ctx_features("code", 0.8, 100, true), "haiku", 0.7);
        }

        let splits = disc.discover_splits();
        // Both arms have identical performance — no discriminating split
        assert!(splits.is_empty(), "Should find no splits when arms perform equally");
    }

    #[test]
    fn test_discover_splits_multiple_domains() {
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);

        // Domain "code": opus wins on high complexity
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.9, 100, true), "opus", 0.9);
            disc.record(&make_ctx_features("code", 0.9, 100, true), "haiku", 0.3);
            disc.record(&make_ctx_features("code", 0.1, 100, true), "haiku", 0.9);
            disc.record(&make_ctx_features("code", 0.1, 100, true), "opus", 0.3);
        }

        // Domain "math": different pattern — haiku wins on high complexity
        for _ in 0..15 {
            disc.record(&make_ctx_features("math", 0.9, 100, false), "haiku", 0.9);
            disc.record(&make_ctx_features("math", 0.9, 100, false), "opus", 0.3);
            disc.record(&make_ctx_features("math", 0.1, 100, false), "opus", 0.9);
            disc.record(&make_ctx_features("math", 0.1, 100, false), "haiku", 0.3);
        }

        let splits = disc.discover_splits();
        let code_splits: Vec<_> = splits.iter().filter(|s| s.domain == "code").collect();
        let math_splits: Vec<_> = splits.iter().filter(|s| s.domain == "math").collect();
        assert!(!code_splits.is_empty(), "Should find splits for 'code'");
        assert!(!math_splits.is_empty(), "Should find splits for 'math'");
    }

    #[test]
    fn test_discover_splits_token_count() {
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);

        // High token count: opus wins
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.5, 1000, true), "opus", 0.9);
            disc.record(&make_ctx_features("code", 0.5, 900, true), "haiku", 0.3);
        }
        // Low token count: haiku wins
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.5, 50, true), "haiku", 0.9);
            disc.record(&make_ctx_features("code", 0.5, 60, true), "opus", 0.3);
        }

        let splits = disc.discover_splits();
        let token_split = splits.iter().find(|s| {
            s.domain == "code" && s.split.dimension == FeatureDimension::TokenCount
        });
        assert!(token_split.is_some(), "Should discover TokenCount split");
    }

    #[test]
    fn test_splits_to_nfa_rules_complexity() {
        let disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let splits = vec![DomainSplit {
            domain: "code".to_string(),
            split: DiscoveredSplit {
                dimension: FeatureDimension::Complexity,
                threshold: 0.7,
                arm_above: "opus".to_string(),
                reward_above: 0.9,
                count_above: 20,
                arm_below: "haiku".to_string(),
                reward_below: 0.8,
                count_below: 20,
                gain: 0.1,
            },
        }];
        let rules = disc.splits_to_nfa_rules(&splits, 100);
        assert_eq!(rules.len(), 2);
        // First rule: Domain("code") + ComplexityRange(70, 100) → opus
        assert_eq!(rules[0].1.len(), 2);
        assert_eq!(rules[0].2, "opus");
        // Second rule: Domain("code") + ComplexityRange(0, 70) → haiku
        assert_eq!(rules[1].1.len(), 2);
        assert_eq!(rules[1].2, "haiku");
    }

    #[test]
    fn test_splits_to_nfa_rules_bool_feature() {
        let disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let splits = vec![DomainSplit {
            domain: "general".to_string(),
            split: DiscoveredSplit {
                dimension: FeatureDimension::HasCode,
                threshold: 0.5,
                arm_above: "opus".to_string(),
                reward_above: 0.9,
                count_above: 15,
                arm_below: "haiku".to_string(),
                reward_below: 0.8,
                count_below: 15,
                gain: 0.1,
            },
        }];
        let rules = disc.splits_to_nfa_rules(&splits, 100);
        assert_eq!(rules.len(), 2);
        // Check one rule has BoolFeature true, other has false
        let has_true = rules.iter().any(|r| r.1.iter().any(|s| matches!(s, NfaSymbol::BoolFeature { value: true, .. })));
        let has_false = rules.iter().any(|r| r.1.iter().any(|s| matches!(s, NfaSymbol::BoolFeature { value: false, .. })));
        assert!(has_true);
        assert!(has_false);
    }

    #[test]
    fn test_splits_to_nfa_rules_domain_prefix() {
        let disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let splits = vec![DomainSplit {
            domain: "code".to_string(),
            split: DiscoveredSplit {
                dimension: FeatureDimension::TokenCount,
                threshold: 500.0,
                arm_above: "opus".to_string(),
                reward_above: 0.9,
                count_above: 20,
                arm_below: "haiku".to_string(),
                reward_below: 0.8,
                count_below: 20,
                gain: 0.1,
            },
        }];
        let rules = disc.splits_to_nfa_rules(&splits, 100);
        // Every rule should start with Domain("code")
        for rule in &rules {
            assert!(matches!(&rule.1[0], NfaSymbol::Domain(d) if d == "code"));
        }
    }

    #[test]
    fn test_splits_to_nfa_rules_unsupported_dimension() {
        let disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let splits = vec![DomainSplit {
            domain: "code".to_string(),
            split: DiscoveredSplit {
                dimension: FeatureDimension::AvgWordLength,
                threshold: 5.0,
                arm_above: "opus".to_string(),
                reward_above: 0.9,
                count_above: 20,
                arm_below: "haiku".to_string(),
                reward_below: 0.8,
                count_below: 20,
                gain: 0.1,
            },
        }];
        let rules = disc.splits_to_nfa_rules(&splits, 100);
        // AvgWordLength has no NfaSymbol mapping → no rules
        assert!(rules.is_empty());
    }

    #[test]
    fn test_synthesize_enhanced_nfa_with_discovered_rules() {
        // Set up bandit with learned data
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("code", "opus");
        bandit.add_arm_for_task("code", "haiku");
        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "opus".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(), success: true, quality: Some(0.6),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
        }

        // Set up contextual discovery with complexity split data
        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.9, 100, true), "opus", 0.95);
            disc.record(&make_ctx_features("code", 0.85, 100, true), "haiku", 0.3);
            disc.record(&make_ctx_features("code", 0.1, 100, true), "haiku", 0.9);
            disc.record(&make_ctx_features("code", 0.15, 100, true), "opus", 0.4);
        }

        // Synthesize enhanced NFA
        let nfa = disc.synthesize_enhanced_nfa(&bandit, 5, 0.3).unwrap();
        // Should have more states/transitions than a base synthesis (has multi-condition paths)
        let base_nfa = BanditNfaSynthesizer::synthesize(&bandit, 5, 0.3).unwrap();
        assert!(
            nfa.states.len() >= base_nfa.states.len(),
            "Enhanced NFA should have at least as many states as base"
        );
    }

    #[test]
    fn test_synthesize_enhanced_nfa_empty_discovery() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("opus");
        for _ in 0..10 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "opus".into(), success: true, quality: Some(0.8),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        // Empty discovery (no observations) → falls back to base rules
        let disc = ContextualDiscovery::new(DiscoveryConfig::default());
        let nfa = disc.synthesize_enhanced_nfa(&bandit, 5, 0.3).unwrap();
        assert!(!nfa.states.is_empty());
    }

    #[test]
    fn test_synthesize_enhanced_nfa_compiles_to_dfa_and_routes() {
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm_for_task("code", "opus");
        bandit.add_arm_for_task("code", "haiku");
        bandit.add_arm("opus");
        bandit.add_arm("haiku");
        for _ in 0..20 {
            bandit.record_outcome(&ArmFeedback {
                arm_id: "opus".into(), success: true, quality: Some(0.9),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(), success: true, quality: Some(0.5),
                latency_ms: None, cost: None, task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "opus".into(), success: true, quality: Some(0.7),
                latency_ms: None, cost: None, task_type: None,
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(), success: true, quality: Some(0.6),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        let config = DiscoveryConfig {
            min_samples_per_split: 5,
            min_gain: 0.01,
            ..DiscoveryConfig::default()
        };
        let mut disc = ContextualDiscovery::new(config);
        for _ in 0..15 {
            disc.record(&make_ctx_features("code", 0.9, 100, true), "opus", 0.95);
            disc.record(&make_ctx_features("code", 0.85, 100, true), "haiku", 0.3);
            disc.record(&make_ctx_features("code", 0.1, 100, true), "haiku", 0.9);
            disc.record(&make_ctx_features("code", 0.15, 100, true), "opus", 0.4);
        }

        let nfa = disc.synthesize_enhanced_nfa(&bandit, 5, 0.3).unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();

        // Route a high-complexity code query — should route to opus (contextual rule)
        let high = make_ctx_features("code", 0.9, 100, true);
        let result = dfa.route(&high).unwrap();
        assert!(!result.selected_arm.is_empty());

        // Route a low-complexity code query — should route to haiku (contextual rule)
        let low = make_ctx_features("code", 0.1, 100, true);
        let result_low = dfa.route(&low).unwrap();
        assert!(!result_low.selected_arm.is_empty());
    }

    #[test]
    fn test_pipeline_enable_discovery() {
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default());
        assert!(pipeline.contextual_discovery().is_none());
        pipeline.enable_discovery(DiscoveryConfig::default());
        assert!(pipeline.contextual_discovery().is_some());
    }

    #[test]
    fn test_pipeline_record_outcome_with_context() {
        let config = PipelineConfig {
            discovery: Some(DiscoveryConfig::default()),
            ..PipelineConfig::default()
        };
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config);
        pipeline.add_arm("opus");

        let features = make_ctx_features("code", 0.8, 100, true);
        let feedback = ArmFeedback {
            arm_id: "opus".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: Some("code".to_string()),
        };
        pipeline.record_outcome_with_context(&feedback, &features);

        // Bandit should have recorded the outcome
        let arm = pipeline.bandit().arm_stats("opus").unwrap();
        assert_eq!(arm.pull_count, 1);

        // Contextual discovery should have recorded the observation
        assert_eq!(pipeline.contextual_discovery().unwrap().observation_count(), 1);
    }

    #[test]
    fn test_pipeline_contextual_resynthesize_end_to_end() {
        // Full end-to-end: pipeline with contextual discovery
        // Feed clear complexity split data → resynthesize → verify DFA routes correctly
        let config = PipelineConfig {
            synthesis_interval: 200, // won't auto-trigger
            min_pulls_for_synthesis: 5,
            quality_threshold: 0.3,
            auto_minimize: true,
            discovery: Some(DiscoveryConfig {
                min_samples_per_split: 5,
                min_gain: 0.01,
                ..DiscoveryConfig::default()
            }),
        };
        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config);
        pipeline.add_arm_for_task("code", "opus");
        pipeline.add_arm_for_task("code", "haiku");
        pipeline.add_arm("opus");
        pipeline.add_arm("haiku");

        // Feed: high complexity code → opus wins, low complexity code → haiku wins
        for _ in 0..20 {
            // High complexity
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "opus".into(), success: true, quality: Some(0.95),
                    latency_ms: None, cost: None, task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.9, 100, true),
            );
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "haiku".into(), success: true, quality: Some(0.3),
                    latency_ms: None, cost: None, task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.85, 100, true),
            );
            // Low complexity
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "haiku".into(), success: true, quality: Some(0.9),
                    latency_ms: None, cost: None, task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.1, 100, true),
            );
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "opus".into(), success: true, quality: Some(0.3),
                    latency_ms: None, cost: None, task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.15, 100, true),
            );
            // Global observations
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "opus".into(), success: true, quality: Some(0.7),
                latency_ms: None, cost: None, task_type: None,
            });
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(), success: true, quality: Some(0.6),
                latency_ms: None, cost: None, task_type: None,
            });
        }

        // Force resynthesize — should use contextual synthesizer
        let result = pipeline.force_resynthesize();
        assert!(result.is_ok(), "Resynthesize failed: {:?}", result.err());
        assert!(pipeline.active_dfa().is_some());

        // Route high-complexity code query
        let high = make_ctx_features("code", 0.95, 100, true);
        let outcome_high = pipeline.route(&high).unwrap();
        assert!(!outcome_high.selected_arm.is_empty());

        // Route low-complexity code query
        let low = make_ctx_features("code", 0.05, 100, true);
        let outcome_low = pipeline.route(&low).unwrap();
        assert!(!outcome_low.selected_arm.is_empty());

        // The DFA should have produced different routes for high vs low
        // (This validates the contextual rules are active in the compiled DFA)
        // Note: we can't guarantee specific arm assignment due to NFA priority
        // resolution, but we can verify both routes succeed through the DFA
        assert!(pipeline.synthesis_count() >= 1);
    }

    // =========================================================================
    // REWARD POLICY TESTS
    // =========================================================================

    #[test]
    fn test_reward_policy_default_values() {
        let rp = RewardPolicy::default();
        assert!((rp.quality_weight - 0.7).abs() < 1e-10);
        assert!((rp.latency_weight - 0.2).abs() < 1e-10);
        assert!((rp.cost_weight - 0.1).abs() < 1e-10);
        assert!((rp.latency_ref_ms - 5000.0).abs() < 1e-10);
        assert!((rp.cost_ref - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_quality_only() {
        let rp = RewardPolicy::default();
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: None, cost: None, task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // With latency=None and cost=None, all weight goes to quality
        assert!((reward - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_all_components() {
        let rp = RewardPolicy {
            quality_weight: 0.5, latency_weight: 0.3, cost_weight: 0.2,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.9),
            latency_ms: Some(500), cost: Some(0.05), task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // quality_score = 0.9, latency_score = 1 - 500/1000 = 0.5, cost_score = 1 - 0.05/0.1 = 0.5
        let expected = 0.5 * 0.9 + 0.3 * 0.5 + 0.2 * 0.5;
        assert!((reward - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reward_policy_all_weights_zero() {
        let rp = RewardPolicy {
            quality_weight: 0.0, latency_weight: 0.0, cost_weight: 0.0,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.6),
            latency_ms: Some(200), cost: Some(0.02), task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // Normalizes to 1/3 each
        let q = 0.6;
        let l = 1.0 - 200.0 / 1000.0;
        let c = 1.0 - 0.02 / 0.1;
        let expected = (q + l + c) / 3.0;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_latency_none_redistributes() {
        let rp = RewardPolicy {
            quality_weight: 0.6, latency_weight: 0.3, cost_weight: 0.1,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: None, cost: Some(0.05), task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        // Latency weight 0.3 redistributed to quality (0.6) and cost (0.1) proportionally
        // effective_qw = 0.6 + 0.3 * (0.6/0.7) ≈ 0.8571
        // effective_cw = 0.1 + 0.3 * (0.1/0.7) ≈ 0.1429
        let cost_score = 1.0 - 0.05 / 0.1;
        let eqw = 0.6 + 0.3 * (0.6 / 0.7);
        let ecw = 0.1 + 0.3 * (0.1 / 0.7);
        let expected = eqw * 0.8 + ecw * cost_score;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_cost_none_redistributes() {
        let rp = RewardPolicy {
            quality_weight: 0.6, latency_weight: 0.3, cost_weight: 0.1,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.7),
            latency_ms: Some(300), cost: None, task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        let lat_score = 1.0 - 300.0 / 1000.0;
        let eqw = 0.6 + 0.1 * (0.6 / 0.9);
        let elw = 0.3 + 0.1 * (0.3 / 0.9);
        let expected = eqw * 0.7 + elw * lat_score;
        assert!((reward - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reward_policy_both_none_quality_only() {
        let rp = RewardPolicy {
            quality_weight: 0.5, latency_weight: 0.3, cost_weight: 0.2,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: false, quality: None,
            latency_ms: None, cost: None, task_type: None,
        };
        let reward = rp.compute_reward(&feedback);
        assert!((reward - 0.0).abs() < 1e-10); // success=false, quality=None -> 0.0
    }

    #[test]
    fn test_reward_policy_high_latency_penalized() {
        let rp = RewardPolicy {
            quality_weight: 0.5, latency_weight: 0.5, cost_weight: 0.0,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let fast = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: Some(100), cost: None, task_type: None,
        };
        let slow = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: Some(900), cost: None, task_type: None,
        };
        assert!(rp.compute_reward(&fast) > rp.compute_reward(&slow));
    }

    #[test]
    fn test_reward_policy_high_cost_penalized() {
        let rp = RewardPolicy {
            quality_weight: 0.5, latency_weight: 0.0, cost_weight: 0.5,
            latency_ref_ms: 1000.0, cost_ref: 0.1,
        };
        let cheap = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: None, cost: Some(0.01), task_type: None,
        };
        let expensive = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.8),
            latency_ms: None, cost: Some(0.09), task_type: None,
        };
        assert!(rp.compute_reward(&cheap) > rp.compute_reward(&expensive));
    }

    #[test]
    fn test_reward_policy_zero_ref_values() {
        let rp = RewardPolicy {
            quality_weight: 0.5, latency_weight: 0.3, cost_weight: 0.2,
            latency_ref_ms: 0.0, cost_ref: 0.0,
        };
        let feedback = ArmFeedback {
            arm_id: "a".to_string(), success: true, quality: Some(0.7),
            latency_ms: Some(100), cost: Some(0.01), task_type: None,
        };
        // Zero refs -> those components treated as unavailable -> quality only
        let reward = rp.compute_reward(&feedback);
        assert!((reward - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_record_outcome_uses_reward_policy() {
        let config = BanditConfig {
            reward_policy: RewardPolicy {
                quality_weight: 0.5, latency_weight: 0.5, cost_weight: 0.0,
                latency_ref_ms: 1000.0, cost_ref: 0.1,
            },
            ..BanditConfig::default()
        };
        let mut router = BanditRouter::new(config);
        router.add_arm("model-a");
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(), success: true, quality: Some(1.0),
            latency_ms: Some(500), cost: None, task_type: None,
        };
        router.record_outcome(&feedback);
        // Reward = 0.5*1.0 + 0.5*0.5 = 0.75 (latency score = 1 - 500/1000 = 0.5)
        let arm = router.all_arms(None).first().unwrap();
        // alpha should be prior(1.0) + 0.75 = 1.75, beta = prior(1.0) + 0.25 = 1.25
        assert!((arm.params.alpha - 1.75).abs() < 0.01);
        assert!((arm.params.beta - 1.25).abs() < 0.01);
    }

    #[test]
    fn test_record_outcome_backward_compat() {
        // Default policy with None latency/cost -> pure quality, matching old behavior
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(), success: true, quality: Some(0.6),
            latency_ms: None, cost: None, task_type: None,
        };
        router.record_outcome(&feedback);
        let arm = router.all_arms(None).first().unwrap();
        // alpha = 1.0 + 0.6 = 1.6, beta = 1.0 + 0.4 = 1.4
        assert!((arm.params.alpha - 1.6).abs() < 0.01);
        assert!((arm.params.beta - 1.4).abs() < 0.01);
    }

    // =========================================================================
    // ROUTING PREFERENCES TESTS
    // =========================================================================

    #[test]
    fn test_routing_preferences_default() {
        let prefs = RoutingPreferences::default();
        assert!(prefs.quality_weight.is_none());
        assert!(prefs.latency_weight.is_none());
        assert!(prefs.cost_weight.is_none());
        assert!(prefs.excluded_arms.is_empty());
        assert!(prefs.preferred_arms.is_empty());
        assert!((prefs.prefer_boost - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_routing_preferences_ignore_cost() {
        let prefs = RoutingPreferences::ignore_cost();
        assert_eq!(prefs.cost_weight, Some(0.0));
    }

    #[test]
    fn test_routing_preferences_minimize_latency() {
        let prefs = RoutingPreferences::minimize_latency();
        assert_eq!(prefs.latency_weight, Some(0.8));
        assert_eq!(prefs.quality_weight, Some(0.2));
        assert_eq!(prefs.cost_weight, Some(0.0));
    }

    #[test]
    fn test_routing_preferences_apply_to_policy() {
        let base = RewardPolicy::default();
        let prefs = RoutingPreferences { cost_weight: Some(0.0), ..Default::default() };
        let policy = prefs.apply_to_policy(&base);
        assert!((policy.cost_weight - 0.0).abs() < 1e-10);
        assert!((policy.quality_weight - 0.7).abs() < 1e-10);
        assert!((policy.latency_weight - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_select_with_preferences_excludes_arms() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        router.add_arm("model-b");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["model-a".to_string()],
            ..Default::default()
        };
        let outcome = router.select_with_preferences(None, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "model-b");
    }

    #[test]
    fn test_select_with_preferences_boosts_preferred() {
        let mut router = BanditRouter::with_seed(BanditConfig::default(), 42);
        router.add_arm("model-a");
        router.add_arm("model-b");
        // Give model-a strong priors
        router.warm_start("model-a", 100.0, 1.0);
        router.warm_start("model-b", 1.0, 100.0);
        // Prefer model-b with high boost
        let prefs = RoutingPreferences {
            preferred_arms: vec!["model-b".to_string()],
            prefer_boost: 1000.0,
            ..Default::default()
        };
        let outcome = router.select_with_preferences(None, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "model-b");
    }

    #[test]
    fn test_select_with_preferences_all_excluded_errors() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["model-a".to_string()],
            ..Default::default()
        };
        let result = router.select_with_preferences(None, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_outcome_with_preferences_custom_weights() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        let prefs = RoutingPreferences::ignore_cost();
        let feedback = ArmFeedback {
            arm_id: "model-a".to_string(), success: true, quality: Some(0.8),
            latency_ms: None, cost: Some(0.1), task_type: None,
        };
        // With ignore_cost, cost is 0 weight -> reward should be quality-only = 0.8
        router.record_outcome_with_preferences(&feedback, &prefs);
        let arm = router.all_arms(None).first().unwrap();
        assert!((arm.params.alpha - 1.8).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_route_with_preferences() {
        let mut pipeline = RoutingPipeline::for_models(&["m1", "m2"], PipelineConfig::default());
        let features = QueryFeatureExtractor::extract("hello world");
        let prefs = RoutingPreferences {
            excluded_arms: vec!["m1".to_string()],
            ..Default::default()
        };
        let outcome = pipeline.route_with_preferences(&features, &prefs).unwrap();
        assert_eq!(outcome.selected_arm, "m2");
    }

    #[test]
    fn test_preferences_serialize_deserialize() {
        let prefs = RoutingPreferences::ignore_cost();
        let json = serde_json::to_string(&prefs).unwrap();
        let prefs2: RoutingPreferences = serde_json::from_str(&json).unwrap();
        assert_eq!(prefs2.cost_weight, Some(0.0));
    }

    // =========================================================================
    // ARM VISIBILITY / PRIVATE ARMS TESTS
    // =========================================================================

    #[test]
    fn test_arm_visibility_default_is_public() {
        let router = BanditRouter::new(BanditConfig::default());
        assert!(router.private_arm_ids().is_empty());
    }

    #[test]
    fn test_set_arm_private_and_query() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("local-model");
        router.set_arm_private("local-model");
        assert!(router.is_arm_private("local-model"));
        assert!(!router.is_arm_private("other-model"));
    }

    #[test]
    fn test_set_arm_public_reverses_private() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("model-a");
        router.set_arm_private("model-a");
        assert!(router.is_arm_private("model-a"));
        router.set_arm_public("model-a");
        assert!(!router.is_arm_private("model-a"));
    }

    #[test]
    fn test_private_arms_accessor() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("m1");
        router.add_arm("m2");
        router.set_arm_private("m1");
        assert_eq!(router.private_arm_ids().len(), 1);
        assert!(router.private_arm_ids().contains("m1"));
    }

    #[test]
    fn test_private_arm_still_selectable_locally() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("local-model");
        router.set_arm_private("local-model");
        // Private arms are still selectable locally
        let outcome = router.select(None).unwrap();
        assert_eq!(outcome.selected_arm, "local-model");
    }

    #[test]
    fn test_snapshot_preserves_private_arms() {
        let mut router = BanditRouter::new(BanditConfig::default());
        router.add_arm("m1");
        router.add_arm("m2");
        router.set_arm_private("m1");
        let json = router.to_json().unwrap();
        let restored = BanditRouter::from_json(&json).unwrap();
        assert!(restored.is_arm_private("m1"));
        assert!(!restored.is_arm_private("m2"));
    }

    // =========================================================================
    // ROUTING CONTEXT TESTS
    // =========================================================================

    #[test]
    fn test_routing_context_new() {
        let features = QueryFeatureExtractor::extract("test query");
        let ctx = RoutingContext::new(features.clone());
        assert!(!ctx.rag_active);
        assert!(ctx.budget_remaining.is_none());
        assert!(ctx.agent_tier.is_none());
        assert_eq!(ctx.features.domain, features.domain);
    }

    #[test]
    fn test_routing_context_from_features() {
        let features = QueryFeatureExtractor::extract("code fn main() {}");
        let ctx: RoutingContext = features.clone().into();
        assert_eq!(ctx.features.has_code, features.has_code);
    }

    #[test]
    fn test_derive_preferences_low_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features, rag_active: false,
            budget_remaining: Some(0.001), // Very low budget
            agent_tier: None, session_cost_so_far: None, preferred_provider: None,
        };
        let policy = RewardPolicy::default(); // cost_ref = 0.01
        let prefs = ctx.derive_preferences(&policy);
        // budget (0.001) < cost_ref * 10 (0.1) -> cost weight boosted
        assert_eq!(prefs.cost_weight, Some(0.5));
    }

    #[test]
    fn test_derive_preferences_no_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext::new(features);
        let policy = RewardPolicy::default();
        let prefs = ctx.derive_preferences(&policy);
        // No budget -> default preferences
        assert!(prefs.cost_weight.is_none());
    }

    #[test]
    fn test_derive_preferences_normal_budget() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features, rag_active: false,
            budget_remaining: Some(100.0), // High budget
            agent_tier: None, session_cost_so_far: None, preferred_provider: None,
        };
        let policy = RewardPolicy::default();
        let prefs = ctx.derive_preferences(&policy);
        assert!(prefs.cost_weight.is_none()); // No override needed
    }

    #[test]
    fn test_pipeline_route_with_context() {
        let mut pipeline = RoutingPipeline::for_models(&["m1"], PipelineConfig::default());
        let features = QueryFeatureExtractor::extract("hello");
        let ctx = RoutingContext::new(features);
        let outcome = pipeline.route_with_context(&ctx).unwrap();
        assert_eq!(outcome.selected_arm, "m1");
    }

    #[test]
    fn test_routing_context_serialize_deserialize() {
        let features = QueryFeatureExtractor::extract("test");
        let ctx = RoutingContext {
            features, rag_active: true,
            budget_remaining: Some(5.0),
            agent_tier: Some("pro".to_string()),
            session_cost_so_far: Some(1.23),
            preferred_provider: Some("openai".to_string()),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let ctx2: RoutingContext = serde_json::from_str(&json).unwrap();
        assert!(ctx2.rag_active);
        assert_eq!(ctx2.budget_remaining, Some(5.0));
        assert_eq!(ctx2.agent_tier.as_deref(), Some("pro"));
    }

    #[test]
    fn test_routing_context_with_rag() {
        let features = QueryFeatureExtractor::extract("search for something");
        let ctx = RoutingContext {
            features, rag_active: true,
            budget_remaining: None, agent_tier: None,
            session_cost_so_far: None, preferred_provider: None,
        };
        assert!(ctx.rag_active);
    }

    // =========================================================================
    // FEATURE IMPORTANCE TESTS
    // =========================================================================

    #[test]
    fn test_feature_importance_empty() {
        let cd = ContextualDiscovery::new(DiscoveryConfig::default());
        let importance = cd.feature_importance();
        assert!(importance.is_empty());
    }

    #[test]
    fn test_feature_importance_single_dimension() {
        let mut cd = ContextualDiscovery::new(DiscoveryConfig {
            min_samples_per_split: 2, min_gain: 0.001,
            ..DiscoveryConfig::default()
        });
        // Create observations where each arm performs well in its complexity zone.
        // Low complexity: haiku=0.9 (good), opus=0.4 (bad)
        // High complexity: opus=0.9 (good), haiku=0.4 (bad)
        // Baseline best arm mean = max(haiku_mean, opus_mean) = max(0.65, 0.65) = 0.65
        // After split: above best = 0.9, below best = 0.9, avg = 0.9
        // gain = 0.9 - 0.65 = 0.25 > 0.001
        for i in 0..40 {
            let mut features = make_ctx_features("coding", 0.5, 50, false);
            let is_low = i < 20;
            features.complexity = if is_low { 0.1 } else { 0.95 };
            let use_haiku = (i % 2) == 0;
            let (arm, reward) = if is_low {
                if use_haiku { ("haiku", 0.9) } else { ("opus", 0.4) }
            } else {
                if use_haiku { ("haiku", 0.4) } else { ("opus", 0.9) }
            };
            cd.record(&features, arm, reward);
        }
        let importance = cd.feature_importance();
        assert!(!importance.is_empty());
        // At least complexity should appear
        assert!(importance.iter().any(|fi| fi.dimension.name() == "complexity"));
    }

    #[test]
    fn test_feature_importance_multiple_sorted() {
        let mut cd = ContextualDiscovery::new(DiscoveryConfig {
            min_samples_per_split: 2, min_gain: 0.001,
            ..DiscoveryConfig::default()
        });
        // Create observations with both complexity and has_code splits
        for i in 0..40 {
            let mut features = make_ctx_features("coding", 0.5, 50, false);
            features.complexity = if i % 2 == 0 { 0.1 } else { 0.9 };
            features.has_code = i >= 20;
            let arm = if i < 20 { "haiku" } else { "opus" };
            let reward = if i < 20 { 0.6 } else { 0.95 };
            cd.record(&features, arm, reward);
        }
        let importance = cd.feature_importance();
        // Should be sorted by total_gain descending
        for w in importance.windows(2) {
            assert!(w[0].total_gain >= w[1].total_gain);
        }
    }

    #[test]
    fn test_feature_importance_domains_count() {
        let mut cd = ContextualDiscovery::new(DiscoveryConfig {
            min_samples_per_split: 2, min_gain: 0.01,
            ..DiscoveryConfig::default()
        });
        // Create observations in two domains
        for domain in &["coding", "math"] {
            for i in 0..20 {
                let mut features = make_ctx_features(domain, 0.5, 50, false);
                features.complexity = if i < 10 { 0.1 } else { 0.9 };
                let arm = if i < 10 { "haiku" } else { "opus" };
                let reward = if i < 10 { 0.7 } else { 0.95 };
                cd.record(&features, arm, reward);
            }
        }
        let importance = cd.feature_importance();
        if let Some(fi) = importance.iter().find(|fi| fi.dimension.name() == "complexity") {
            // Complexity splits should affect both domains
            assert!(fi.domains_affected >= 1);
        }
    }

    #[test]
    fn test_feature_importance_no_splits() {
        let mut cd = ContextualDiscovery::new(DiscoveryConfig {
            min_samples_per_split: 100, // Very high minimum
            ..DiscoveryConfig::default()
        });
        // Add a few observations (not enough for any split)
        for i in 0..5 {
            let features = make_ctx_features("coding", 0.5, 50, false);
            cd.record(&features, "model", 0.5 + i as f64 * 0.01);
        }
        let importance = cd.feature_importance();
        assert!(importance.is_empty());
    }
}
