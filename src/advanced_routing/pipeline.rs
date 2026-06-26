//! Closed-loop routing pipeline tying bandit, NFA/DFA, and discovery together.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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
        let contextual = pipeline_config
            .discovery
            .as_ref()
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
                domain_arms
                    .entry(domain.clone())
                    .or_default()
                    .extend(reachable);
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
        let premium: Vec<&str> = models
            .iter()
            .filter(|(_, t)| matches!(t, ModelTier::Premium))
            .map(|(m, _)| *m)
            .collect();
        let standard: Vec<&str> = models
            .iter()
            .filter(|(_, t)| matches!(t, ModelTier::Standard))
            .map(|(m, _)| *m)
            .collect();
        let economy: Vec<&str> = models
            .iter()
            .filter(|(_, t)| matches!(t, ModelTier::Economy))
            .map(|(m, _)| *m)
            .collect();

        // Premium → code + high complexity
        for model in &premium {
            builder = builder
                .rule(&format!("{}_code", model))
                .when(NfaSymbol::BoolFeature {
                    name: "has_code".into(),
                    value: true,
                })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;

            builder = builder
                .rule(&format!("{}_complex", model))
                .when(NfaSymbol::ComplexityRange {
                    low_pct: 70,
                    high_pct: 100,
                })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;
        }

        // Standard → medium complexity
        for model in &standard {
            builder = builder
                .rule(&format!("{}_mid", model))
                .when(NfaSymbol::ComplexityRange {
                    low_pct: 30,
                    high_pct: 70,
                })
                .route_to(model)
                .priority(priority)
                .done();
            priority -= 1;
        }

        // Economy → low complexity
        for model in &economy {
            builder = builder
                .rule(&format!("{}_simple", model))
                .when(NfaSymbol::ComplexityRange {
                    low_pct: 0,
                    high_pct: 30,
                })
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
    pub fn route(
        &mut self,
        features: &QueryFeatures,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Try DFA first
        if let Some(ref dfa) = self.active_dfa {
            match dfa.route(features) {
                Ok(outcome) => return Ok(outcome),
                Err(_) => {} // Fall through to bandit
            }
        }

        // Fallback: bandit
        let task_type = if features.domain.is_empty() {
            None
        } else {
            Some(features.domain.as_str())
        };
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
        self.bandit
            .select_with_preferences(Some(&features.domain), prefs)
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
        serde_json::to_string_pretty(&snapshot).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })
    }

    /// Restore a pipeline from JSON.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: PipelineSnapshot =
            serde_json::from_str(json).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            })?;
        if snapshot.version != PIPELINE_SNAPSHOT_VERSION {
            return Err(AdvancedRoutingError::IncompatibleVersion {
                expected: PIPELINE_SNAPSHOT_VERSION,
                found: snapshot.version,
            });
        }
        let bandit =
            BanditRouter::from_json(&serde_json::to_string(&snapshot.bandit).map_err(|e| {
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
        let active_dfa = source_nfa
            .as_ref()
            .and_then(|nfa| NfaDfaCompiler::compile(nfa).ok())
            .map(|mut dfa| {
                if snapshot.config.auto_minimize {
                    dfa.minimize();
                }
                dfa
            });

        let contextual = snapshot
            .config
            .discovery
            .as_ref()
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

#[cfg(test)]
mod tests {
    use super::*;

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
            .rule("r")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa)
            .unwrap();

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
            .with_initial_nfa(nfa)
            .unwrap();

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
            .rule("r")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("dfa-model")
            .priority(10)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa)
            .unwrap();

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
                arm_id: "m1".into(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
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
                arm_id: "m1".into(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
        }

        pipeline.force_resynthesize().unwrap();
        assert_eq!(pipeline.synthesis_count(), 1);
    }

    #[test]
    fn test_pipeline_export_snapshot() {
        let nfa = NfaRuleBuilder::new().fallback("fb", 1).build().unwrap();

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa)
            .unwrap();

        let snapshot = pipeline.export_snapshot();
        assert_eq!(snapshot.version, 1);
        assert!(snapshot.nfa.is_some());
    }

    #[test]
    fn test_pipeline_with_initial_rules() {
        let builder = NfaRuleBuilder::new()
            .rule("r")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .fallback("fb", 1);

        let pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_rules(builder)
            .unwrap();

        assert!(pipeline.active_dfa().is_some());
    }

    #[test]
    fn test_pipeline_json_round_trip() {
        let nfa = NfaRuleBuilder::new()
            .rule("r")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), PipelineConfig::default())
            .with_initial_nfa(nfa)
            .unwrap();
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
        let mut pipeline =
            RoutingPipeline::for_models(&["gpt-4", "claude", "gemini"], PipelineConfig::default());

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
                arm_id: "m1".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
                arm_id: "code-specialist".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "generalist".into(),
                success: true,
                quality: Some(0.5),
                latency_ms: None,
                cost: None,
                task_type: None,
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
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("code-m")
            .priority(10)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("math-m")
            .priority(8)
            .done()
            .build()
            .unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();

        // Verify routes from both pipelines work
        assert_eq!(
            dfa.route(&test_features("code", 0.5)).unwrap().selected_arm,
            "code-m"
        );
        assert_eq!(
            dfa.route(&test_features("math", 0.5)).unwrap().selected_arm,
            "math-m"
        );
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
            .rule("init")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("initial-model")
            .priority(5)
            .done()
            .fallback("fallback", 1);

        let mut pipeline = RoutingPipeline::new(BanditConfig::default(), config)
            .with_initial_rules(builder)
            .unwrap();

        // Add arms for bandit learning
        pipeline.add_arm_for_task("code", "better-model");
        pipeline.add_arm("fallback");

        // Simulate learning: better-model outperforms
        for _ in 0..12 {
            pipeline.record_outcome(&ArmFeedback {
                arm_id: "better-model".into(),
                success: true,
                quality: Some(0.95),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
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
            .rule("r")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();

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
}
