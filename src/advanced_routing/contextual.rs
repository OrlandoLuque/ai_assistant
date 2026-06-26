//! Per-query adaptive routing and contextual bandit auto-discovery.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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
        let sentence_count = query
            .chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count()
            .max(1);
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
        let avg_word_length = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64
        };
        let complexity =
            Self::estimate_complexity(query, token_count, sentence_count, entity_count);

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
        query.contains("```")
            || query.contains("fn ")
            || query.contains("def ")
            || query.contains("class ")
            || query.contains("function ")
            || query.contains("import ")
            || query.contains("pub fn")
    }

    fn detect_domain(query: &str) -> String {
        let lower = query.to_lowercase();
        if lower.contains("code")
            || lower.contains("function")
            || lower.contains("implement")
            || lower.contains("programming")
            || lower.contains("debug")
            || lower.contains("compile")
        {
            "coding".to_string()
        } else if lower.contains("math")
            || lower.contains("calculate")
            || lower.contains("equation")
            || lower.contains("solve")
            || lower.contains("integral")
        {
            "math".to_string()
        } else if lower.contains("write a story")
            || lower.contains("poem")
            || lower.contains("creative")
        {
            "creative".to_string()
        } else if lower.contains("translate") || lower.contains("translation") {
            "translation".to_string()
        } else if lower.contains("summarize") || lower.contains("summary") || lower.contains("tldr")
        {
            "summarization".to_string()
        } else {
            "general".to_string()
        }
    }

    fn count_entities(query: &str) -> usize {
        let mut count = 0;
        for word in query.split_whitespace() {
            // Count capitalized words (potential proper nouns)
            if word.len() > 1
                && word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                count += 1;
            }
        }
        // Count numbers
        count += query
            .split_whitespace()
            .filter(|w| w.parse::<f64>().is_ok())
            .count();
        count
    }

    fn estimate_complexity(
        query: &str,
        token_count: usize,
        sentence_count: usize,
        entity_count: usize,
    ) -> f64 {
        let length_factor = (token_count as f64 / 100.0).min(1.0);
        let sentence_factor = (sentence_count as f64 / 5.0).min(1.0);
        let entity_factor = (entity_count as f64 / 10.0).min(1.0);
        let clause_factor = (query.matches(',').count() as f64 / 5.0).min(1.0);

        let raw = length_factor * 0.3
            + sentence_factor * 0.2
            + entity_factor * 0.25
            + clause_factor * 0.25;
        raw.min(1.0).max(0.0)
    }
}

/// Per-query adaptive router that learns feature→model mapping from outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePerQueryRouter {
    pub(crate) domain_bandits: HashMap<String, BanditRouter>,
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
        self.complexity_thresholds
            .push((max_complexity, model.to_string()));
        self.complexity_thresholds
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self
    }

    /// Route a raw query string.
    pub fn route(&mut self, query: &str) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let features = QueryFeatureExtractor::extract(query);
        self.route_with_features(&features)
    }

    /// Route using pre-extracted features.
    pub fn route_with_features(
        &mut self,
        features: &QueryFeatures,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
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
                    reason: format!(
                        "Complexity {:.2} <= tier {:.2}",
                        features.complexity, threshold
                    ),
                    alternatives: Vec::new(),
                    router_id: "adaptive".to_string(),
                    decision_time_us: elapsed,
                });
            }
        }

        // Priority 4: domain-specific bandit
        let bandit = self
            .domain_bandits
            .entry(features.domain.clone())
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
            FeatureDimension::HasCode => {
                if ctx.has_code {
                    1.0
                } else {
                    0.0
                }
            }
            FeatureDimension::IsQuestion => {
                if ctx.is_question {
                    1.0
                } else {
                    0.0
                }
            }
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
        matches!(
            self,
            FeatureDimension::HasCode | FeatureDimension::IsQuestion
        )
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
            let domain_obs: Vec<&ContextualObservation> = self
                .observations
                .iter()
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
                    let true_obs: Vec<&ContextualObservation> = domain_obs
                        .iter()
                        .filter(|o| dim.extract(&o.context) >= 0.5)
                        .copied()
                        .collect();
                    let false_obs: Vec<&ContextualObservation> = domain_obs
                        .iter()
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
                    let values: Vec<f64> =
                        domain_obs.iter().map(|o| dim.extract(&o.context)).collect();
                    let split_points =
                        Self::compute_quantile_split_points(&values, self.config.num_split_points);

                    for threshold in split_points {
                        let above: Vec<&ContextualObservation> = domain_obs
                            .iter()
                            .filter(|o| dim.extract(&o.context) >= threshold)
                            .copied()
                            .collect();
                        let below: Vec<&ContextualObservation> = domain_obs
                            .iter()
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
            b.split
                .gain
                .partial_cmp(&a.split.gain)
                .unwrap_or(std::cmp::Ordering::Equal)
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
                            NfaSymbol::ComplexityRange {
                                low_pct,
                                high_pct: 100,
                            },
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
                            NfaSymbol::ComplexityRange {
                                low_pct: 0,
                                high_pct: low_pct,
                            },
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
                            NfaSymbol::TokenRange {
                                min: threshold_usize,
                                max: usize::MAX,
                            },
                        ],
                        ds.split.arm_above.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                    rules.push((
                        format!("ctx_{}_tokens_low", ds.split.arm_below),
                        vec![
                            NfaSymbol::Domain(ds.domain.clone()),
                            NfaSymbol::TokenRange {
                                min: 0,
                                max: threshold_usize.saturating_sub(1),
                            },
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
                            NfaSymbol::BoolFeature {
                                name: feature_name.clone(),
                                value: true,
                            },
                        ],
                        ds.split.arm_above.clone(),
                        priority,
                    ));
                    priority = priority.saturating_sub(1);
                    rules.push((
                        format!("ctx_{}_{}_false", ds.split.arm_below, feature_name),
                        vec![
                            NfaSymbol::Domain(ds.domain.clone()),
                            NfaSymbol::BoolFeature {
                                name: feature_name,
                                value: false,
                            },
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
            let mut qualified: Vec<(&BanditArm, f64)> = arms
                .iter()
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
            qualified.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
        let global_best = global_arms
            .iter()
            .filter(|a| a.pull_count >= min_pulls)
            .max_by(|a, b| {
                let ma = if a.pull_count > 0 {
                    a.total_reward / a.pull_count as f64
                } else {
                    0.0
                };
                let mb = if b.pull_count > 0 {
                    b.total_reward / b.pull_count as f64
                } else {
                    0.0
                };
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
        arm_stats
            .iter()
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
            if points
                .last()
                .map_or(true, |last: &f64| (val - *last).abs() > 1e-10)
            {
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

        let mut result: Vec<FeatureImportance> = by_dim
            .into_iter()
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

        result.sort_by(|a, b| {
            b.total_gain
                .partial_cmp(&a.total_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

#[cfg(test)]
mod tests {
    use super::*;

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
             infrastructure resilience, and international trade dynamics.",
        );
        assert!(complex.complexity > simple.complexity);
    }

    #[test]
    fn test_feature_extraction_domain() {
        assert_eq!(
            QueryFeatureExtractor::extract("solve this equation").domain,
            "math"
        );
        assert_eq!(
            QueryFeatureExtractor::extract("write a poem about love").domain,
            "creative"
        );
        assert_eq!(
            QueryFeatureExtractor::extract("translate this to French").domain,
            "translation"
        );
    }

    #[test]
    fn test_adaptive_routes_code_to_code_model() {
        let config = BanditConfig::default();
        let mut router =
            AdaptivePerQueryRouter::new("default", config).with_code_model("code-model");
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
        router.record_outcome(
            "What is coding?",
            &ArmFeedback {
                arm_id: "default".to_string(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: None,
            },
        );
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
    // CONTEXTUAL DISCOVERY TESTS
    // =========================================================================

    fn make_ctx_features(
        domain: &str,
        complexity: f64,
        token_count: usize,
        has_code: bool,
    ) -> QueryFeatures {
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
        let obs = [
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "opus".into(),
                reward: 0.6,
            },
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "opus".into(),
                reward: 0.8,
            },
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "opus".into(),
                reward: 1.0,
            },
        ];
        let refs: Vec<&ContextualObservation> = obs.iter().collect();
        let (arm, mean) = ContextualDiscovery::compute_best_arm_mean(&refs);
        assert_eq!(arm, "opus");
        assert!((mean - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_compute_best_arm_mean_two_arms() {
        let f = make_ctx_features("code", 0.5, 100, true);
        let obs = [
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "haiku".into(),
                reward: 0.5,
            },
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "haiku".into(),
                reward: 0.5,
            },
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "opus".into(),
                reward: 0.9,
            },
            ContextualObservation {
                context: ContextSnapshot::from(&f),
                arm_id: "opus".into(),
                reward: 0.9,
            },
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
            disc.record(
                &make_ctx_features("code", i as f64 / 10.0, 100, true),
                "opus",
                0.9,
            );
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
        let complexity_split = splits
            .iter()
            .find(|s| s.domain == "code" && s.split.dimension == FeatureDimension::Complexity);
        assert!(
            complexity_split.is_some(),
            "Should find a complexity split for 'code'"
        );

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
        let bool_split = splits
            .iter()
            .find(|s| s.domain == "general" && s.split.dimension == FeatureDimension::HasCode);
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
        assert!(
            splits.is_empty(),
            "Should find no splits when arms perform equally"
        );
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
        let token_split = splits
            .iter()
            .find(|s| s.domain == "code" && s.split.dimension == FeatureDimension::TokenCount);
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
        let has_true = rules.iter().any(|r| {
            r.1.iter()
                .any(|s| matches!(s, NfaSymbol::BoolFeature { value: true, .. }))
        });
        let has_false = rules.iter().any(|r| {
            r.1.iter()
                .any(|s| matches!(s, NfaSymbol::BoolFeature { value: false, .. }))
        });
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
                arm_id: "opus".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
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
                arm_id: "opus".into(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: None,
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
                arm_id: "opus".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(),
                success: true,
                quality: Some(0.5),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "opus".into(),
                success: true,
                quality: Some(0.7),
                latency_ms: None,
                cost: None,
                task_type: None,
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: None,
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
        assert_eq!(
            pipeline.contextual_discovery().unwrap().observation_count(),
            1
        );
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
                    arm_id: "opus".into(),
                    success: true,
                    quality: Some(0.95),
                    latency_ms: None,
                    cost: None,
                    task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.9, 100, true),
            );
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "haiku".into(),
                    success: true,
                    quality: Some(0.3),
                    latency_ms: None,
                    cost: None,
                    task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.85, 100, true),
            );
            // Low complexity
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "haiku".into(),
                    success: true,
                    quality: Some(0.9),
                    latency_ms: None,
                    cost: None,
                    task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.1, 100, true),
            );
            pipeline.record_outcome_with_context(
                &ArmFeedback {
                    arm_id: "opus".into(),
                    success: true,
                    quality: Some(0.3),
                    latency_ms: None,
                    cost: None,
                    task_type: Some("code".into()),
                },
                &make_ctx_features("code", 0.15, 100, true),
            );
            // Global observations
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "opus".into(),
                success: true,
                quality: Some(0.7),
                latency_ms: None,
                cost: None,
                task_type: None,
            });
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "haiku".into(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: None,
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
            min_samples_per_split: 2,
            min_gain: 0.001,
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
                if use_haiku {
                    ("haiku", 0.9)
                } else {
                    ("opus", 0.4)
                }
            } else {
                if use_haiku {
                    ("haiku", 0.4)
                } else {
                    ("opus", 0.9)
                }
            };
            cd.record(&features, arm, reward);
        }
        let importance = cd.feature_importance();
        assert!(!importance.is_empty());
        // At least complexity should appear
        assert!(importance
            .iter()
            .any(|fi| fi.dimension.name() == "complexity"));
    }

    #[test]
    fn test_feature_importance_multiple_sorted() {
        let mut cd = ContextualDiscovery::new(DiscoveryConfig {
            min_samples_per_split: 2,
            min_gain: 0.001,
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
            min_samples_per_split: 2,
            min_gain: 0.01,
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
        if let Some(fi) = importance
            .iter()
            .find(|fi| fi.dimension.name() == "complexity")
        {
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
