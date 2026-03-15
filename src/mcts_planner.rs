//! Monte Carlo Tree Search (MCTS) Planning with Process Reward Models and Refinement Loop.
//!
//! This module provides a complete MCTS implementation for agent-style planning,
//! including UCB1-based tree search, pluggable simulation policies, process reward
//! models (rule-based and LLM-backed), and an iterative refinement loop.
//!
//! # Feature gate
//! Requires `autonomous` feature.

use crate::error::MctsError;
use serde::{Deserialize, Serialize};

// =============================================================================
// 8.1 — MCTS Planning
// =============================================================================

/// Defines the problem space for MCTS search.
///
/// Implementors describe states, valid transitions, terminal conditions,
/// and reward signals that guide the tree search.
pub trait MctsState: Clone + std::fmt::Debug {
    /// The type of actions that can be taken from a state.
    type Action: Clone + std::fmt::Debug + PartialEq;

    /// Returns all valid actions from the current state.
    fn available_actions(&self) -> Vec<Self::Action>;

    /// Applies an action to produce a new state.
    fn apply_action(&self, action: &Self::Action) -> Result<Self, MctsError>;

    /// Whether this state is terminal (no further actions possible/needed).
    fn is_terminal(&self) -> bool;

    /// The reward value of this state (higher is better).
    fn reward(&self) -> f64;

    /// Human-readable description of the state.
    fn description(&self) -> String;
}

/// Configuration for the MCTS planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MctsConfig {
    /// Maximum number of MCTS iterations (selection-expansion-simulation-backprop cycles).
    pub max_iterations: usize,
    /// UCB1 exploration constant C. Higher values favor exploration over exploitation.
    pub exploration_constant: f64,
    /// Maximum depth of the search tree.
    pub max_depth: usize,
    /// Maximum depth of simulation (rollout) from a leaf node.
    pub simulation_depth: usize,
    /// Discount factor gamma for future rewards (0.0-1.0).
    pub discount_factor: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            exploration_constant: std::f64::consts::SQRT_2,
            max_depth: 50,
            simulation_depth: 20,
            discount_factor: 0.99,
        }
    }
}

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
pub struct MctsNode<S: MctsState> {
    /// The state at this node.
    pub state: S,
    /// The action that led to this node (None for root).
    pub action: Option<S::Action>,
    /// Number of times this node has been visited.
    pub visits: usize,
    /// Cumulative reward accumulated through this node.
    pub total_reward: f64,
    /// Child nodes.
    pub children: Vec<MctsNode<S>>,
    /// Visit count of the parent (used in UCB1 calculation).
    pub parent_visits: usize,
    /// Depth of this node in the tree.
    pub depth: usize,
    /// Whether all possible actions from this state have been expanded.
    pub fully_expanded: bool,
}

impl<S: MctsState> MctsNode<S> {
    /// Create a new MCTS node.
    pub fn new(state: S, action: Option<S::Action>, depth: usize) -> Self {
        Self {
            state,
            action,
            visits: 0,
            total_reward: 0.0,
            children: Vec::new(),
            parent_visits: 0,
            depth,
            fully_expanded: false,
        }
    }

    /// Compute the UCB1 score for this node.
    ///
    /// UCB1 = avg_reward + C * sqrt(ln(parent_visits) / visits)
    ///
    /// Returns `f64::MAX` if visits == 0 (unvisited nodes are always selected first).
    pub fn ucb1(&self, exploration_c: f64) -> f64 {
        if self.visits == 0 {
            return f64::MAX;
        }
        let avg = self.avg_reward();
        let parent_v = if self.parent_visits > 0 {
            self.parent_visits as f64
        } else {
            1.0
        };
        avg + exploration_c * ((parent_v).ln() / self.visits as f64).sqrt()
    }

    /// Average reward for this node.
    pub fn avg_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }

    /// Returns the index of the child with the highest UCB1 score.
    pub fn best_child(&self, exploration_c: f64) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, child) in self.children.iter().enumerate() {
            let score = child.ucb1(exploration_c);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some(best_idx)
    }

    /// Returns the action of the child with the highest visit count (exploitation).
    pub fn best_action(&self) -> Option<&S::Action> {
        if self.children.is_empty() {
            return None;
        }
        let mut best_visits = 0;
        let mut best_action = None;
        for child in &self.children {
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action.as_ref();
            }
        }
        best_action
    }

    /// Whether this node is a leaf (has no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Whether all available actions have been expanded as children.
    pub fn is_fully_expanded(&self) -> bool {
        self.fully_expanded
    }
}

/// Result of an MCTS search.
#[derive(Debug, Clone)]
pub struct MctsResult<A: Clone + std::fmt::Debug> {
    /// The best immediate action to take from the root.
    pub best_action: Option<A>,
    /// The best sequence of actions following the most-visited path.
    pub best_action_sequence: Vec<A>,
    /// The estimated value of the root state.
    pub root_value: f64,
    /// Number of MCTS iterations actually performed.
    pub iterations_used: usize,
    /// Maximum depth reached during the search.
    pub max_depth_reached: usize,
    /// Total number of simulations (rollouts) performed.
    pub total_simulations: usize,
}

/// Policy used during the simulation (rollout) phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SimulationPolicy {
    /// Pick a random action at each step.
    Random,
    /// Heuristic: prefer first available action, or random if prefer_unexplored is false.
    Heuristic { prefer_unexplored: bool },
    /// Pick the action that leads to the highest immediate reward.
    MaxReward,
}

/// The main MCTS planner.
pub struct MctsPlanner {
    config: MctsConfig,
    policy: SimulationPolicy,
}

impl MctsPlanner {
    /// Create a planner with the given config and Random simulation policy.
    pub fn new(config: MctsConfig) -> Self {
        Self {
            config,
            policy: SimulationPolicy::Random,
        }
    }

    /// Create a planner with default config and Random simulation policy.
    pub fn with_defaults() -> Self {
        Self {
            config: MctsConfig::default(),
            policy: SimulationPolicy::Random,
        }
    }

    /// Create a planner with the given config and policy.
    pub fn with_policy(config: MctsConfig, policy: SimulationPolicy) -> Self {
        Self { config, policy }
    }

    /// Access the configuration.
    pub fn config(&self) -> &MctsConfig {
        &self.config
    }

    /// Access the simulation policy.
    pub fn policy(&self) -> &SimulationPolicy {
        &self.policy
    }

    /// Run the full MCTS search from the given initial state.
    ///
    /// Returns the best action and action sequence, along with statistics.
    pub fn search<S: MctsState>(&self, initial_state: &S) -> Result<MctsResult<S::Action>, MctsError> {
        // Terminal initial state: no search needed
        if initial_state.is_terminal() {
            return Ok(MctsResult {
                best_action: None,
                best_action_sequence: Vec::new(),
                root_value: initial_state.reward(),
                iterations_used: 0,
                max_depth_reached: 0,
                total_simulations: 0,
            });
        }

        let actions = initial_state.available_actions();
        if actions.is_empty() {
            return Err(MctsError::NoValidActions {
                state_description: initial_state.description(),
            });
        }

        let mut root = MctsNode::new(initial_state.clone(), None, 0);
        let mut max_depth_reached: usize = 0;
        let mut total_simulations: usize = 0;
        let mut rng_state = self.init_rng();

        for _iteration in 0..self.config.max_iterations {
            // 1. Selection: walk down the tree picking best UCB1 children
            let path = self.select(&root);

            // 2. Expansion: expand one untried action at the selected leaf
            //    We need to do expansion on the actual tree (mutable).
            let leaf_depth = Self::path_depth(&root, &path);
            let expanded_path = self.expand(&mut root, &path, &mut rng_state);

            // 3. Simulation: rollout from the expanded node
            let sim_node_state = Self::get_node_state(&root, &expanded_path);
            let sim_reward = self.simulate(&sim_node_state, &mut rng_state);
            total_simulations += 1;

            let node_depth = expanded_path.len();
            if node_depth > max_depth_reached {
                max_depth_reached = node_depth;
            }
            if leaf_depth > max_depth_reached {
                max_depth_reached = leaf_depth;
            }

            // 4. Backpropagation: propagate the reward up the tree
            self.backpropagate(&mut root, &expanded_path, sim_reward);
        }

        // Extract best action (most visited child of root)
        let best_action = root.best_action().cloned();
        let best_action_sequence = self.extract_best_sequence(&root);
        let root_value = root.avg_reward();

        Ok(MctsResult {
            best_action,
            best_action_sequence,
            root_value,
            iterations_used: self.config.max_iterations,
            max_depth_reached,
            total_simulations,
        })
    }

    /// Initialize a simple pseudo-random state from the current time.
    fn init_rng(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Simple xorshift64 pseudo-random number generator.
    fn next_rng(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// Select a path from root to a leaf using UCB1.
    /// Returns a Vec of child indices.
    fn select<S: MctsState>(&self, root: &MctsNode<S>) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;

        loop {
            if current.is_leaf() || current.state.is_terminal() {
                break;
            }
            if !current.is_fully_expanded() {
                break;
            }
            match current.best_child(self.config.exploration_constant) {
                Some(idx) => {
                    path.push(idx);
                    current = &current.children[idx];
                }
                None => break,
            }
        }

        path
    }

    /// Expand one untried action at the node indicated by `path`.
    /// Returns the path to the newly expanded node.
    fn expand<S: MctsState>(
        &self,
        root: &mut MctsNode<S>,
        path: &[usize],
        rng_state: &mut u64,
    ) -> Vec<usize> {
        let node = Self::get_node_mut(root, path);

        if node.state.is_terminal() || node.depth >= self.config.max_depth {
            return path.to_vec();
        }

        let available = node.state.available_actions();
        if available.is_empty() {
            node.fully_expanded = true;
            return path.to_vec();
        }

        // Find actions not yet expanded as children
        let tried_actions: Vec<_> = node
            .children
            .iter()
            .filter_map(|c| c.action.clone())
            .collect();

        let untried: Vec<_> = available
            .into_iter()
            .filter(|a| !tried_actions.contains(a))
            .collect();

        if untried.is_empty() {
            node.fully_expanded = true;
            // Select best child instead
            if let Some(best_idx) = node.best_child(self.config.exploration_constant) {
                let mut new_path = path.to_vec();
                new_path.push(best_idx);
                return new_path;
            }
            return path.to_vec();
        }

        // Pick one untried action (pseudo-random)
        let action_idx = (Self::next_rng(rng_state) as usize) % untried.len();
        let action = untried[action_idx].clone();

        let new_depth = node.depth + 1;
        match node.state.apply_action(&action) {
            Ok(new_state) => {
                let child = MctsNode::new(new_state, Some(action), new_depth);
                node.children.push(child);
                let child_idx = node.children.len() - 1;

                // Check if fully expanded now
                let remaining_untried = untried.len() - 1;
                if remaining_untried == 0 {
                    node.fully_expanded = true;
                }

                let mut new_path = path.to_vec();
                new_path.push(child_idx);
                new_path
            }
            Err(_) => {
                // If action fails, mark as expanded and return current path
                if untried.len() == 1 {
                    node.fully_expanded = true;
                }
                path.to_vec()
            }
        }
    }

    /// Get the state at the node indicated by `path`.
    fn get_node_state<S: MctsState>(root: &MctsNode<S>, path: &[usize]) -> S {
        let mut current = root;
        for &idx in path {
            current = &current.children[idx];
        }
        current.state.clone()
    }

    /// Get the depth of the node indicated by `path`.
    fn path_depth<S: MctsState>(root: &MctsNode<S>, path: &[usize]) -> usize {
        let mut current = root;
        for &idx in path {
            if idx < current.children.len() {
                current = &current.children[idx];
            } else {
                break;
            }
        }
        current.depth
    }

    /// Get a mutable reference to the node at `path`.
    fn get_node_mut<'a, S: MctsState>(
        root: &'a mut MctsNode<S>,
        path: &[usize],
    ) -> &'a mut MctsNode<S> {
        let mut current = root;
        for &idx in path {
            current = &mut current.children[idx];
        }
        current
    }

    /// Simulate (rollout) from a state using the configured policy.
    fn simulate<S: MctsState>(&self, state: &S, rng_state: &mut u64) -> f64 {
        let mut current = state.clone();
        let mut depth = 0;
        let mut cumulative_discount = 1.0;
        let mut total_reward = 0.0;

        while !current.is_terminal() && depth < self.config.simulation_depth {
            let actions = current.available_actions();
            if actions.is_empty() {
                break;
            }

            let action = match &self.policy {
                SimulationPolicy::Random => {
                    let idx = (Self::next_rng(rng_state) as usize) % actions.len();
                    actions[idx].clone()
                }
                SimulationPolicy::Heuristic { prefer_unexplored } => {
                    if *prefer_unexplored {
                        // Prefer first available action (deterministic heuristic)
                        actions[0].clone()
                    } else {
                        let idx = (Self::next_rng(rng_state) as usize) % actions.len();
                        actions[idx].clone()
                    }
                }
                SimulationPolicy::MaxReward => {
                    // Pick action leading to highest immediate reward
                    let mut best_action = actions[0].clone();
                    let mut best_reward = f64::NEG_INFINITY;
                    for a in &actions {
                        if let Ok(next_state) = current.apply_action(a) {
                            let r = next_state.reward();
                            if r > best_reward {
                                best_reward = r;
                                best_action = a.clone();
                            }
                        }
                    }
                    best_action
                }
            };

            match current.apply_action(&action) {
                Ok(next) => {
                    current = next;
                }
                Err(_) => break,
            }

            total_reward += cumulative_discount * current.reward();
            cumulative_discount *= self.config.discount_factor;
            depth += 1;
        }

        // Terminal reward contribution
        if current.is_terminal() || depth == 0 {
            total_reward += cumulative_discount * current.reward();
        }

        total_reward
    }

    /// Backpropagate the simulation reward up the path from leaf to root.
    fn backpropagate<S: MctsState>(
        &self,
        root: &mut MctsNode<S>,
        path: &[usize],
        reward: f64,
    ) {
        // Update root
        root.visits += 1;
        root.total_reward += reward;

        // Update parent_visits for all root's children
        for child in &mut root.children {
            child.parent_visits = root.visits;
        }

        // Walk down the path updating each node
        let mut current = root;
        for &idx in path {
            if idx < current.children.len() {
                current = &mut current.children[idx];
                current.visits += 1;
                current.total_reward += reward;

                // Update parent_visits for this node's children
                let visits = current.visits;
                for child in &mut current.children {
                    child.parent_visits = visits;
                }
            }
        }
    }

    /// Extract the best action sequence by following the most-visited child at each level.
    fn extract_best_sequence<S: MctsState>(&self, root: &MctsNode<S>) -> Vec<S::Action> {
        let mut sequence = Vec::new();
        let mut current = root;

        loop {
            if current.children.is_empty() {
                break;
            }

            // Find child with most visits
            let mut best_visits = 0;
            let mut best_idx = 0;
            for (i, child) in current.children.iter().enumerate() {
                if child.visits > best_visits {
                    best_visits = child.visits;
                    best_idx = i;
                }
            }

            if best_visits == 0 {
                break;
            }

            if let Some(action) = current.children[best_idx].action.as_ref() {
                sequence.push(action.clone());
            }
            current = &current.children[best_idx];
        }

        sequence
    }
}

// =============================================================================
// 8.2 — Process Reward Models
// =============================================================================

/// A score assigned to a step by a Process Reward Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepScore {
    /// Quality score (0.0-1.0).
    pub score: f64,
    /// Confidence in the score (0.0-1.0).
    pub confidence: f64,
    /// Human-readable feedback.
    pub feedback: String,
}

/// Trait for models that score individual reasoning/planning steps.
pub trait ProcessRewardModel: Send + Sync {
    /// Score a single step given its description and the context of previous steps.
    fn score_step(&self, step_description: &str, context: &[String]) -> StepScore;

    /// Name of this reward model.
    fn name(&self) -> &str;
}

/// A single rule used by `RuleBasedPRM`.
#[derive(Debug, Clone)]
pub struct PrmRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// The check to apply.
    pub check: PrmRuleCheck,
    /// Weight of this rule in the aggregate score.
    pub weight: f64,
}

/// Types of checks a PRM rule can perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrmRuleCheck {
    /// Step description must contain the given keyword.
    ContainsKeyword(String),
    /// Step description must not exceed this length.
    MaxLength(usize),
    /// Step description must be at least this long.
    MinLength(usize),
    /// Step description must not be empty.
    NotEmpty,
    /// Step description must match a simple substring pattern.
    MatchesPattern(String),
}

impl PrmRuleCheck {
    /// Evaluate the rule against a step description.
    fn evaluate(&self, step_description: &str) -> bool {
        match self {
            PrmRuleCheck::ContainsKeyword(keyword) => {
                step_description
                    .to_lowercase()
                    .contains(&keyword.to_lowercase())
            }
            PrmRuleCheck::MaxLength(max) => step_description.len() <= *max,
            PrmRuleCheck::MinLength(min) => step_description.len() >= *min,
            PrmRuleCheck::NotEmpty => !step_description.trim().is_empty(),
            PrmRuleCheck::MatchesPattern(pattern) => {
                // Simple wildcard matching: '*' matches any sequence
                Self::simple_pattern_match(pattern, step_description)
            }
        }
    }

    /// Simple pattern matching with '*' as wildcard.
    fn simple_pattern_match(pattern: &str, text: &str) -> bool {
        let pattern_lower = pattern.to_lowercase();
        let text_lower = text.to_lowercase();

        let parts: Vec<&str> = pattern_lower.split('*').collect();
        if parts.len() == 1 {
            // No wildcard: exact match
            return text_lower == pattern_lower;
        }

        let mut pos = 0;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            match text_lower[pos..].find(part) {
                Some(found) => {
                    // First part must match at start
                    if i == 0 && found != 0 {
                        return false;
                    }
                    pos += found + part.len();
                }
                None => return false,
            }
        }

        // Last part must match at end (unless it's empty, meaning pattern ends with *)
        if let Some(last) = parts.last() {
            if !last.is_empty() && !text_lower.ends_with(last) {
                return false;
            }
        }

        true
    }
}

/// A rule-based Process Reward Model that evaluates steps using configurable rules.
pub struct RuleBasedPRM {
    rules: Vec<PrmRule>,
}

impl RuleBasedPRM {
    /// Create an empty rule-based PRM.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to this PRM.
    pub fn add_rule(&mut self, rule: PrmRule) {
        self.rules.push(rule);
    }

    /// Create a PRM with sensible default rules (NotEmpty + MaxLength).
    pub fn with_default_rules() -> Self {
        let mut prm = Self::new();
        prm.add_rule(PrmRule {
            name: "not_empty".to_string(),
            check: PrmRuleCheck::NotEmpty,
            weight: 1.0,
        });
        prm.add_rule(PrmRule {
            name: "max_length".to_string(),
            check: PrmRuleCheck::MaxLength(10000),
            weight: 1.0,
        });
        prm
    }
}

impl ProcessRewardModel for RuleBasedPRM {
    fn score_step(&self, step_description: &str, _context: &[String]) -> StepScore {
        if self.rules.is_empty() {
            return StepScore {
                score: 1.0,
                confidence: 0.0,
                feedback: "No rules configured".to_string(),
            };
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut failed_rules = Vec::new();

        for rule in &self.rules {
            let passed = rule.check.evaluate(step_description);
            let rule_score = if passed { 1.0 } else { 0.0 };
            weighted_sum += rule_score * rule.weight;
            total_weight += rule.weight;

            if !passed {
                failed_rules.push(rule.name.clone());
            }
        }

        let score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            1.0
        };

        let feedback = if failed_rules.is_empty() {
            "All rules passed".to_string()
        } else {
            format!("Failed rules: {}", failed_rules.join(", "))
        };

        StepScore {
            score,
            confidence: 1.0, // Rule-based is deterministic
            feedback,
        }
    }

    fn name(&self) -> &str {
        "RuleBasedPRM"
    }
}

/// An LLM-backed Process Reward Model that delegates scoring to a callback.
pub struct LlmPRM {
    scorer: Box<dyn Fn(&str, &[String]) -> f64 + Send + Sync>,
    name: String,
}

impl LlmPRM {
    /// Create a new LLM PRM with the given name and scoring function.
    pub fn new(
        name: impl Into<String>,
        scorer: impl Fn(&str, &[String]) -> f64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            scorer: Box::new(scorer),
            name: name.into(),
        }
    }
}

impl ProcessRewardModel for LlmPRM {
    fn score_step(&self, step_description: &str, context: &[String]) -> StepScore {
        let score = (self.scorer)(step_description, context);
        let clamped = score.clamp(0.0, 1.0);
        StepScore {
            score: clamped,
            confidence: 0.8, // LLM-based confidence is moderate
            feedback: format!("LLM scored {:.4}", clamped),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Strategy for aggregating scores from multiple PRMs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AggregationStrategy {
    /// Arithmetic mean of all scores.
    Average,
    /// Minimum score (most conservative).
    Minimum,
    /// Weighted average with specified weights per model.
    WeightedAverage(Vec<f64>),
    /// Product of all scores.
    Product,
}

/// Aggregates scores from multiple Process Reward Models.
pub struct PrmAggregator {
    models: Vec<Box<dyn ProcessRewardModel>>,
    strategy: AggregationStrategy,
}

impl PrmAggregator {
    /// Create a new aggregator with the given strategy.
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            models: Vec::new(),
            strategy,
        }
    }

    /// Add a Process Reward Model to this aggregator.
    pub fn add_model(&mut self, model: Box<dyn ProcessRewardModel>) {
        self.models.push(model);
    }

    /// Score a step by aggregating scores from all models.
    pub fn score_step(&self, step: &str, context: &[String]) -> StepScore {
        if self.models.is_empty() {
            return StepScore {
                score: 0.0,
                confidence: 0.0,
                feedback: "No models configured".to_string(),
            };
        }

        let scores: Vec<StepScore> = self
            .models
            .iter()
            .map(|m| m.score_step(step, context))
            .collect();

        let raw_scores: Vec<f64> = scores.iter().map(|s| s.score).collect();

        let aggregated = match &self.strategy {
            AggregationStrategy::Average => {
                let sum: f64 = raw_scores.iter().sum();
                sum / raw_scores.len() as f64
            }
            AggregationStrategy::Minimum => {
                raw_scores
                    .iter()
                    .copied()
                    .fold(f64::MAX, f64::min)
            }
            AggregationStrategy::WeightedAverage(weights) => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                for (i, &score) in raw_scores.iter().enumerate() {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    weighted_sum += score * w;
                    total_weight += w;
                }
                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                }
            }
            AggregationStrategy::Product => {
                raw_scores.iter().copied().fold(1.0, |acc, s| acc * s)
            }
        };

        let avg_confidence = {
            let sum: f64 = scores.iter().map(|s| s.confidence).sum();
            sum / scores.len() as f64
        };

        let feedbacks: Vec<String> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| {
                format!("[{}] {}", self.models[i].name(), s.feedback)
            })
            .collect();

        StepScore {
            score: aggregated.clamp(0.0, 1.0),
            confidence: avg_confidence,
            feedback: feedbacks.join("; "),
        }
    }

    /// Number of models in this aggregator.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

// =============================================================================
// 8.3 — Plan Refinement Loop
// =============================================================================

/// Configuration for the refinement loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RefinementConfig {
    /// Maximum number of refinement iterations.
    pub max_iterations: usize,
    /// Minimum improvement threshold to continue refining.
    pub improvement_threshold: f64,
    /// Whether to re-plan when a step fails.
    pub replan_on_failure: bool,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            improvement_threshold: 0.01,
            replan_on_failure: true,
        }
    }
}

/// Feedback from executing a single plan step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionFeedback {
    /// Index of the step in the plan.
    pub step_index: usize,
    /// Whether the step executed successfully.
    pub success: bool,
    /// Actual reward observed after execution.
    pub actual_reward: f64,
    /// Expected reward before execution.
    pub expected_reward: f64,
    /// Error message if the step failed.
    pub error_message: Option<String>,
}

/// Strategy for refining a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RefinementStrategy {
    /// Re-plan from the point of failure.
    ReplanFromFailure,
    /// Re-plan the entire sequence.
    ReplanEntire,
    /// Patch only the failed step.
    PatchStep,
}

/// Result of a refinement iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementResult {
    /// Number of refinement iterations actually performed.
    pub iterations_used: usize,
    /// Reward before refinement.
    pub initial_reward: f64,
    /// Reward after refinement.
    pub final_reward: f64,
    /// Relative improvement.
    pub improvement: f64,
    /// Strategy that was used.
    pub strategy_used: RefinementStrategy,
    /// Whether the refinement converged (no more improvement needed).
    pub converged: bool,
}

/// Iteratively refines plans based on execution feedback.
pub struct RefinementLoop {
    config: RefinementConfig,
    strategy: RefinementStrategy,
}

impl RefinementLoop {
    /// Create a new refinement loop with the given config and strategy.
    pub fn new(config: RefinementConfig, strategy: RefinementStrategy) -> Self {
        Self { config, strategy }
    }

    /// Create a refinement loop with default config and ReplanFromFailure strategy.
    pub fn with_defaults() -> Self {
        Self {
            config: RefinementConfig::default(),
            strategy: RefinementStrategy::ReplanFromFailure,
        }
    }

    /// Whether refinement is needed based on the given feedbacks.
    ///
    /// Returns true if any step failed or if there's room for improvement.
    pub fn should_refine(&self, feedbacks: &[ExecutionFeedback]) -> bool {
        if feedbacks.is_empty() {
            return false;
        }

        // Check if any step failed
        if feedbacks.iter().any(|f| !f.success) {
            return true;
        }

        // Check if actual vs expected diverges beyond threshold
        for f in feedbacks {
            let diff = (f.actual_reward - f.expected_reward).abs();
            if diff > self.config.improvement_threshold {
                return true;
            }
        }

        false
    }

    /// Identify the first failure point in execution feedbacks.
    pub fn identify_failure_point(&self, feedbacks: &[ExecutionFeedback]) -> Option<usize> {
        feedbacks
            .iter()
            .find(|f| !f.success)
            .map(|f| f.step_index)
    }

    /// Compute the relative improvement between initial and current rewards.
    pub fn compute_improvement(&self, initial: f64, current: f64) -> f64 {
        let denominator = initial.abs().max(1e-10);
        (current - initial) / denominator
    }

    /// Run one refinement iteration: check if refinement is needed, re-plan, and report.
    pub fn evaluate_plan<S: MctsState>(
        &self,
        planner: &MctsPlanner,
        state: &S,
        feedbacks: &[ExecutionFeedback],
    ) -> Result<RefinementResult, MctsError> {
        // Compute initial reward from feedbacks
        let initial_reward = if feedbacks.is_empty() {
            state.reward()
        } else {
            feedbacks.iter().map(|f| f.actual_reward).sum::<f64>()
                / feedbacks.len() as f64
        };

        // Check if refinement is needed
        if !self.should_refine(feedbacks) {
            return Ok(RefinementResult {
                iterations_used: 0,
                initial_reward,
                final_reward: initial_reward,
                improvement: 0.0,
                strategy_used: self.strategy.clone(),
                converged: true,
            });
        }

        // Run MCTS search from current state
        let result = planner.search(state)?;
        let new_reward = result.root_value;
        let improvement = self.compute_improvement(initial_reward, new_reward);

        let converged = improvement.abs() < self.config.improvement_threshold;

        Ok(RefinementResult {
            iterations_used: 1,
            initial_reward,
            final_reward: new_reward,
            improvement,
            strategy_used: self.strategy.clone(),
            converged,
        })
    }

    /// Access the configuration.
    pub fn config(&self) -> &RefinementConfig {
        &self.config
    }
}

// =============================================================================
// AgentMctsState — concrete MctsState for agent-style planning
// =============================================================================

/// A concrete `MctsState` for agent-style planning where actions are tool names.
#[derive(Debug, Clone)]
pub struct AgentMctsState {
    /// Current step index.
    pub current_step: usize,
    /// Actions (tool names) that have been completed.
    pub completed_actions: Vec<String>,
    /// Tools available for use.
    pub available_tools: Vec<String>,
    /// Remaining budget (each action costs 1.0).
    pub remaining_budget: f64,
    /// Reward accumulated so far.
    pub accumulated_reward: f64,
    /// Maximum number of steps allowed.
    pub max_steps: usize,
}

impl MctsState for AgentMctsState {
    type Action = String;

    fn available_actions(&self) -> Vec<Self::Action> {
        if self.remaining_budget <= 0.0 {
            return Vec::new();
        }
        self.available_tools
            .iter()
            .filter(|t| !self.completed_actions.contains(t))
            .cloned()
            .collect()
    }

    fn apply_action(&self, action: &Self::Action) -> Result<Self, MctsError> {
        if !self.available_tools.contains(action) {
            return Err(MctsError::StateError {
                action: action.clone(),
                reason: "Tool not available".to_string(),
            });
        }
        if self.completed_actions.contains(action) {
            return Err(MctsError::StateError {
                action: action.clone(),
                reason: "Tool already used".to_string(),
            });
        }
        if self.remaining_budget <= 0.0 {
            return Err(MctsError::StateError {
                action: action.clone(),
                reason: "No remaining budget".to_string(),
            });
        }

        let mut new_state = self.clone();
        new_state.completed_actions.push(action.clone());
        new_state.current_step += 1;
        new_state.remaining_budget -= 1.0;
        new_state.accumulated_reward += 0.1;
        Ok(new_state)
    }

    fn is_terminal(&self) -> bool {
        self.current_step >= self.max_steps
            || self.remaining_budget <= 0.0
            || self.available_actions().is_empty()
    }

    fn reward(&self) -> f64 {
        self.accumulated_reward
    }

    fn description(&self) -> String {
        format!(
            "Step {}/{}, budget: {:.1}, completed: [{}], reward: {:.2}",
            self.current_step,
            self.max_steps,
            self.remaining_budget,
            self.completed_actions.join(", "),
            self.accumulated_reward
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TestState — deterministic MctsState for testing
    // =========================================================================

    /// A simple deterministic MctsState for testing.
    #[derive(Debug, Clone)]
    struct TestState {
        value: i32,
        max_value: i32,
        actions_taken: Vec<i32>,
    }

    impl TestState {
        fn new(max_value: i32) -> Self {
            Self {
                value: 0,
                max_value,
                actions_taken: Vec::new(),
            }
        }
    }

    impl MctsState for TestState {
        type Action = i32;

        fn available_actions(&self) -> Vec<Self::Action> {
            if self.is_terminal() {
                return Vec::new();
            }
            // Actions: +1, +2, +3
            vec![1, 2, 3]
        }

        fn apply_action(&self, action: &Self::Action) -> Result<Self, MctsError> {
            let mut new_state = self.clone();
            new_state.value += action;
            new_state.actions_taken.push(*action);
            Ok(new_state)
        }

        fn is_terminal(&self) -> bool {
            self.value >= self.max_value
        }

        fn reward(&self) -> f64 {
            // Higher reward the closer to (but not exceeding) max_value
            if self.value > self.max_value {
                (self.max_value as f64) / (self.value as f64)
            } else {
                self.value as f64 / self.max_value.max(1) as f64
            }
        }

        fn description(&self) -> String {
            format!("TestState(value={}, max={})", self.value, self.max_value)
        }
    }

    // =========================================================================
    // MctsConfig tests
    // =========================================================================

    #[test]
    fn test_mcts_config_default_values() {
        let config = MctsConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert!((config.exploration_constant - std::f64::consts::SQRT_2).abs() < 1e-10);
        assert_eq!(config.max_depth, 50);
        assert_eq!(config.simulation_depth, 20);
        assert!((config.discount_factor - 0.99).abs() < 1e-10);
    }

    // =========================================================================
    // MctsNode tests
    // =========================================================================

    #[test]
    fn test_mcts_node_new() {
        let state = TestState::new(10);
        let node = MctsNode::new(state.clone(), None, 0);
        assert_eq!(node.visits, 0);
        assert!((node.total_reward - 0.0).abs() < 1e-10);
        assert!(node.children.is_empty());
        assert!(node.action.is_none());
        assert_eq!(node.depth, 0);
        assert!(!node.fully_expanded);
    }

    #[test]
    fn test_mcts_node_new_with_action() {
        let state = TestState::new(10);
        let node = MctsNode::new(state, Some(42), 3);
        assert_eq!(node.action, Some(42));
        assert_eq!(node.depth, 3);
    }

    #[test]
    fn test_mcts_node_ucb1_zero_visits_returns_max() {
        let state = TestState::new(10);
        let node = MctsNode::new(state, None, 0);
        assert_eq!(node.ucb1(1.414), f64::MAX);
    }

    #[test]
    fn test_mcts_node_ucb1_calculation() {
        let state = TestState::new(10);
        let mut node = MctsNode::new(state, None, 0);
        node.visits = 10;
        node.total_reward = 5.0;
        node.parent_visits = 100;

        let c = std::f64::consts::SQRT_2;
        let expected = 0.5 + c * ((100.0_f64.ln()) / 10.0).sqrt();
        let actual = node.ucb1(c);
        assert!((actual - expected).abs() < 1e-10, "UCB1: expected {}, got {}", expected, actual);
    }

    #[test]
    fn test_mcts_node_avg_reward() {
        let state = TestState::new(10);
        let mut node = MctsNode::new(state, None, 0);

        // Zero visits: avg is 0
        assert!((node.avg_reward() - 0.0).abs() < 1e-10);

        node.visits = 4;
        node.total_reward = 2.0;
        assert!((node.avg_reward() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mcts_node_best_child() {
        let state = TestState::new(10);
        let mut root = MctsNode::new(state.clone(), None, 0);
        root.visits = 30;

        // Add three children with different visit counts and rewards
        let mut child0 = MctsNode::new(state.clone(), Some(1), 1);
        child0.visits = 10;
        child0.total_reward = 3.0;
        child0.parent_visits = 30;

        let mut child1 = MctsNode::new(state.clone(), Some(2), 1);
        child1.visits = 15;
        child1.total_reward = 9.0;
        child1.parent_visits = 30;

        let mut child2 = MctsNode::new(state.clone(), Some(3), 1);
        child2.visits = 5;
        child2.total_reward = 4.0;
        child2.parent_visits = 30;

        root.children = vec![child0, child1, child2];

        // With exploration constant 0, purely exploit: child1 has highest avg reward (0.6)
        // child2 avg = 0.8, actually child2 is best purely by avg
        // child0: 0.3, child1: 0.6, child2: 0.8
        let idx = root.best_child(0.0).unwrap();
        assert_eq!(idx, 2, "Child 2 has highest avg reward (0.8)");
    }

    #[test]
    fn test_mcts_node_best_child_empty() {
        let state = TestState::new(10);
        let root = MctsNode::<TestState>::new(state, None, 0);
        assert!(root.best_child(1.0).is_none());
    }

    #[test]
    fn test_mcts_node_best_action() {
        let state = TestState::new(10);
        let mut root = MctsNode::new(state.clone(), None, 0);

        let mut child0 = MctsNode::new(state.clone(), Some(1), 1);
        child0.visits = 50;

        let mut child1 = MctsNode::new(state.clone(), Some(2), 1);
        child1.visits = 100;

        let mut child2 = MctsNode::new(state.clone(), Some(3), 1);
        child2.visits = 30;

        root.children = vec![child0, child1, child2];

        // Best action is the one with most visits
        assert_eq!(root.best_action(), Some(&2));
    }

    #[test]
    fn test_mcts_node_best_action_empty() {
        let state = TestState::new(10);
        let root = MctsNode::<TestState>::new(state, None, 0);
        assert!(root.best_action().is_none());
    }

    #[test]
    fn test_mcts_node_is_leaf() {
        let state = TestState::new(10);
        let mut node = MctsNode::new(state.clone(), None, 0);
        assert!(node.is_leaf());

        node.children.push(MctsNode::new(state, Some(1), 1));
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_mcts_node_is_fully_expanded() {
        let state = TestState::new(10);
        let mut node = MctsNode::new(state, None, 0);
        assert!(!node.is_fully_expanded());

        node.fully_expanded = true;
        assert!(node.is_fully_expanded());
    }

    // =========================================================================
    // MctsPlanner tests
    // =========================================================================

    #[test]
    fn test_mcts_planner_with_defaults() {
        let planner = MctsPlanner::with_defaults();
        assert_eq!(planner.config().max_iterations, 1000);
        assert!(matches!(planner.policy(), SimulationPolicy::Random));
    }

    #[test]
    fn test_mcts_planner_new() {
        let config = MctsConfig {
            max_iterations: 500,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::new(config);
        assert_eq!(planner.config().max_iterations, 500);
    }

    #[test]
    fn test_mcts_planner_with_policy() {
        let config = MctsConfig::default();
        let planner = MctsPlanner::with_policy(config, SimulationPolicy::MaxReward);
        assert!(matches!(planner.policy(), SimulationPolicy::MaxReward));
    }

    #[test]
    fn test_mcts_planner_search_simple() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into(), "analyze".into(), "report".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        let config = MctsConfig {
            max_iterations: 200,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        // Should find an action
        assert!(result.best_action.is_some());
        assert!(!result.best_action_sequence.is_empty());
        assert!(result.iterations_used > 0);
        assert!(result.total_simulations > 0);
    }

    #[test]
    fn test_mcts_planner_search_finds_best_action() {
        let state = TestState::new(5);
        let config = MctsConfig {
            max_iterations: 500,
            exploration_constant: 0.5,
            max_depth: 10,
            simulation_depth: 10,
            discount_factor: 0.99,
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        // The search should find an action
        assert!(result.best_action.is_some());
        let action = result.best_action.unwrap();
        // Action should be one of the valid actions (1, 2, or 3)
        assert!([1, 2, 3].contains(&action));
    }

    #[test]
    fn test_mcts_planner_search_terminal_initial_state() {
        let mut state = TestState::new(5);
        state.value = 10; // Already past terminal
        let planner = MctsPlanner::with_defaults();
        let result = planner.search(&state).unwrap();
        assert!(result.best_action.is_none());
        assert!(result.best_action_sequence.is_empty());
        assert_eq!(result.iterations_used, 0);
    }

    #[test]
    fn test_mcts_planner_search_no_actions() {
        // A state that is not terminal but has no actions
        #[derive(Debug, Clone)]
        struct NoActionState;
        impl MctsState for NoActionState {
            type Action = i32;
            fn available_actions(&self) -> Vec<i32> { Vec::new() }
            fn apply_action(&self, _action: &i32) -> Result<Self, MctsError> {
                Err(MctsError::NoValidActions { state_description: "no actions".into() })
            }
            fn is_terminal(&self) -> bool { false }
            fn reward(&self) -> f64 { 0.0 }
            fn description(&self) -> String { "NoActionState".into() }
        }

        let planner = MctsPlanner::with_defaults();
        let result = planner.search(&NoActionState);
        assert!(result.is_err());
        match result.unwrap_err() {
            MctsError::NoValidActions { .. } => {}
            other => panic!("Expected NoValidActions, got: {:?}", other),
        }
    }

    #[test]
    fn test_mcts_planner_search_multiple_iterations_valid() {
        let state = TestState::new(10);
        let config = MctsConfig {
            max_iterations: 300,
            exploration_constant: std::f64::consts::SQRT_2,
            max_depth: 20,
            simulation_depth: 15,
            discount_factor: 0.99,
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        assert!(result.best_action.is_some());
        assert!(result.total_simulations > 0);
        assert!(result.max_depth_reached > 0);
        assert_eq!(result.iterations_used, 300);
    }

    #[test]
    fn test_mcts_exploration_constant_zero_exploits_only() {
        let state = TestState::new(6);
        let config = MctsConfig {
            max_iterations: 200,
            exploration_constant: 0.0,
            max_depth: 10,
            simulation_depth: 10,
            discount_factor: 0.99,
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        // Should still find a valid action
        assert!(result.best_action.is_some());
        let action = result.best_action.unwrap();
        assert!([1, 2, 3].contains(&action));
    }

    #[test]
    fn test_mcts_high_exploration() {
        let state = TestState::new(6);
        let config = MctsConfig {
            max_iterations: 200,
            exploration_constant: 10.0, // Very high exploration
            max_depth: 10,
            simulation_depth: 10,
            discount_factor: 0.99,
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        // Should still produce a valid result
        assert!(result.best_action.is_some());
        assert!(result.total_simulations > 0);
    }

    #[test]
    fn test_simulation_policy_random_rollout() {
        let state = TestState::new(6);
        let config = MctsConfig {
            max_iterations: 100,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::with_policy(config, SimulationPolicy::Random);
        let result = planner.search(&state).unwrap();
        assert!(result.best_action.is_some());
    }

    #[test]
    fn test_simulation_policy_max_reward_rollout() {
        let state = TestState::new(6);
        let config = MctsConfig {
            max_iterations: 100,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::with_policy(config, SimulationPolicy::MaxReward);
        let result = planner.search(&state).unwrap();
        assert!(result.best_action.is_some());
    }

    #[test]
    fn test_simulation_policy_heuristic_prefer_unexplored() {
        let state = TestState::new(6);
        let config = MctsConfig {
            max_iterations: 100,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::with_policy(
            config,
            SimulationPolicy::Heuristic { prefer_unexplored: true },
        );
        let result = planner.search(&state).unwrap();
        assert!(result.best_action.is_some());
    }

    // =========================================================================
    // MctsResult tests
    // =========================================================================

    #[test]
    fn test_mcts_result_with_best_action_sequence() {
        let result: MctsResult<String> = MctsResult {
            best_action: Some("search".into()),
            best_action_sequence: vec!["search".into(), "analyze".into(), "report".into()],
            root_value: 0.75,
            iterations_used: 100,
            max_depth_reached: 3,
            total_simulations: 100,
        };
        assert_eq!(result.best_action_sequence.len(), 3);
        assert_eq!(result.best_action, Some("search".into()));
        assert!((result.root_value - 0.75).abs() < 1e-10);
    }

    // =========================================================================
    // AgentMctsState tests
    // =========================================================================

    #[test]
    fn test_agent_mcts_state_available_actions() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: vec!["search".into()],
            available_tools: vec!["search".into(), "analyze".into(), "report".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        let actions = state.available_actions();
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&"analyze".to_string()));
        assert!(actions.contains(&"report".to_string()));
        assert!(!actions.contains(&"search".to_string()));
    }

    #[test]
    fn test_agent_mcts_state_available_actions_no_budget() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into()],
            remaining_budget: 0.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        assert!(state.available_actions().is_empty());
    }

    #[test]
    fn test_agent_mcts_state_apply_action() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into(), "analyze".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        let new_state = state.apply_action(&"search".to_string()).unwrap();
        assert_eq!(new_state.current_step, 1);
        assert_eq!(new_state.completed_actions, vec!["search".to_string()]);
        assert!((new_state.remaining_budget - 4.0).abs() < 1e-10);
        assert!((new_state.accumulated_reward - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_agent_mcts_state_apply_invalid_action() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        let result = state.apply_action(&"nonexistent".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_mcts_state_is_terminal_budget_exhausted() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into()],
            remaining_budget: 0.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        assert!(state.is_terminal());
    }

    #[test]
    fn test_agent_mcts_state_is_terminal_max_steps() {
        let state = AgentMctsState {
            current_step: 5,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        assert!(state.is_terminal());
    }

    #[test]
    fn test_agent_mcts_state_is_terminal_tools_exhausted() {
        let state = AgentMctsState {
            current_step: 1,
            completed_actions: vec!["search".into()],
            available_tools: vec!["search".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };
        assert!(state.is_terminal());
    }

    #[test]
    fn test_agent_mcts_state_reward_accumulates() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["a".into(), "b".into(), "c".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };

        let s1 = state.apply_action(&"a".to_string()).unwrap();
        assert!((s1.reward() - 0.1).abs() < 1e-10);

        let s2 = s1.apply_action(&"b".to_string()).unwrap();
        assert!((s2.reward() - 0.2).abs() < 1e-10);

        let s3 = s2.apply_action(&"c".to_string()).unwrap();
        assert!((s3.reward() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_agent_mcts_state_description() {
        let state = AgentMctsState {
            current_step: 1,
            completed_actions: vec!["search".into()],
            available_tools: vec!["search".into(), "analyze".into()],
            remaining_budget: 4.0,
            accumulated_reward: 0.1,
            max_steps: 5,
        };
        let desc = state.description();
        assert!(desc.contains("Step 1/5"));
        assert!(desc.contains("budget: 4.0"));
        assert!(desc.contains("search"));
    }

    // =========================================================================
    // StepScore tests
    // =========================================================================

    #[test]
    fn test_step_score_construction() {
        let score = StepScore {
            score: 0.85,
            confidence: 0.92,
            feedback: "Good step".to_string(),
        };
        assert!((score.score - 0.85).abs() < 1e-10);
        assert!((score.confidence - 0.92).abs() < 1e-10);
        assert_eq!(score.feedback, "Good step");
    }

    // =========================================================================
    // PrmRuleCheck tests
    // =========================================================================

    #[test]
    fn test_prm_rule_check_not_empty_passes() {
        let check = PrmRuleCheck::NotEmpty;
        assert!(check.evaluate("hello"));
        assert!(check.evaluate("  content  "));
    }

    #[test]
    fn test_prm_rule_check_not_empty_fails() {
        let check = PrmRuleCheck::NotEmpty;
        assert!(!check.evaluate(""));
        assert!(!check.evaluate("   "));
        assert!(!check.evaluate("\t\n"));
    }

    #[test]
    fn test_prm_rule_check_max_length_passes() {
        let check = PrmRuleCheck::MaxLength(10);
        assert!(check.evaluate("short"));
        assert!(check.evaluate("1234567890"));
    }

    #[test]
    fn test_prm_rule_check_max_length_fails() {
        let check = PrmRuleCheck::MaxLength(5);
        assert!(!check.evaluate("toolong"));
    }

    #[test]
    fn test_prm_rule_check_min_length() {
        let check = PrmRuleCheck::MinLength(3);
        assert!(check.evaluate("abc"));
        assert!(check.evaluate("abcdef"));
        assert!(!check.evaluate("ab"));
    }

    #[test]
    fn test_prm_rule_check_contains_keyword() {
        let check = PrmRuleCheck::ContainsKeyword("analyze".into());
        assert!(check.evaluate("We should analyze the data"));
        assert!(check.evaluate("ANALYZE this"));
        assert!(!check.evaluate("We should search the data"));
    }

    #[test]
    fn test_prm_rule_check_matches_pattern() {
        let check = PrmRuleCheck::MatchesPattern("*analyze*".into());
        assert!(check.evaluate("first analyze then report"));
        assert!(!check.evaluate("just searching"));

        let exact = PrmRuleCheck::MatchesPattern("hello".into());
        assert!(exact.evaluate("hello"));
        assert!(!exact.evaluate("hello world"));

        let prefix = PrmRuleCheck::MatchesPattern("hello*".into());
        assert!(prefix.evaluate("hello world"));
        assert!(!prefix.evaluate("say hello"));
    }

    // =========================================================================
    // RuleBasedPRM tests
    // =========================================================================

    #[test]
    fn test_rule_based_prm_with_default_rules() {
        let prm = RuleBasedPRM::with_default_rules();
        assert_eq!(prm.rules.len(), 2);
    }

    #[test]
    fn test_rule_based_prm_score_step_passing() {
        let prm = RuleBasedPRM::with_default_rules();
        let score = prm.score_step("Analyze the dataset for patterns", &[]);
        assert!((score.score - 1.0).abs() < 1e-10);
        assert!((score.confidence - 1.0).abs() < 1e-10);
        assert!(score.feedback.contains("All rules passed"));
    }

    #[test]
    fn test_rule_based_prm_score_step_empty_fails() {
        let prm = RuleBasedPRM::with_default_rules();
        let score = prm.score_step("", &[]);
        // NotEmpty fails, MaxLength passes => 0.5
        assert!((score.score - 0.5).abs() < 1e-10);
        assert!(score.feedback.contains("Failed rules"));
        assert!(score.feedback.contains("not_empty"));
    }

    #[test]
    fn test_rule_based_prm_custom_rules() {
        let mut prm = RuleBasedPRM::new();
        prm.add_rule(PrmRule {
            name: "has_keyword".into(),
            check: PrmRuleCheck::ContainsKeyword("data".into()),
            weight: 2.0,
        });
        prm.add_rule(PrmRule {
            name: "not_empty".into(),
            check: PrmRuleCheck::NotEmpty,
            weight: 1.0,
        });

        let score = prm.score_step("Analyze the data", &[]);
        assert!((score.score - 1.0).abs() < 1e-10); // Both pass

        let score = prm.score_step("Analyze the results", &[]);
        // has_keyword fails (weight=2), not_empty passes (weight=1) => 1/3
        assert!((score.score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rule_based_prm_name() {
        let prm = RuleBasedPRM::new();
        assert_eq!(prm.name(), "RuleBasedPRM");
    }

    // =========================================================================
    // LlmPRM tests
    // =========================================================================

    #[test]
    fn test_llm_prm_delegates_to_scorer() {
        let prm = LlmPRM::new("test_llm", |step, _ctx| {
            if step.contains("good") {
                0.9
            } else {
                0.3
            }
        });

        let score = prm.score_step("This is a good step", &[]);
        assert!((score.score - 0.9).abs() < 1e-10);

        let score = prm.score_step("This is a bad step", &[]);
        assert!((score.score - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_llm_prm_clamps_score() {
        let prm = LlmPRM::new("clamper", |_step, _ctx| 1.5);
        let score = prm.score_step("anything", &[]);
        assert!((score.score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_llm_prm_name() {
        let prm = LlmPRM::new("my_model", |_, _| 0.5);
        assert_eq!(prm.name(), "my_model");
    }

    #[test]
    fn test_llm_prm_uses_context() {
        let prm = LlmPRM::new("ctx_model", |_step, ctx| {
            ctx.len() as f64 * 0.1
        });
        let context = vec!["step1".into(), "step2".into()];
        let score = prm.score_step("step3", &context);
        assert!((score.score - 0.2).abs() < 1e-10);
    }

    // =========================================================================
    // PrmAggregator tests
    // =========================================================================

    #[test]
    fn test_prm_aggregator_average() {
        let mut agg = PrmAggregator::new(AggregationStrategy::Average);
        agg.add_model(Box::new(LlmPRM::new("a", |_, _| 0.8)));
        agg.add_model(Box::new(LlmPRM::new("b", |_, _| 0.4)));

        let score = agg.score_step("test", &[]);
        assert!((score.score - 0.6).abs() < 1e-10);
        assert_eq!(agg.model_count(), 2);
    }

    #[test]
    fn test_prm_aggregator_minimum() {
        let mut agg = PrmAggregator::new(AggregationStrategy::Minimum);
        agg.add_model(Box::new(LlmPRM::new("a", |_, _| 0.8)));
        agg.add_model(Box::new(LlmPRM::new("b", |_, _| 0.3)));
        agg.add_model(Box::new(LlmPRM::new("c", |_, _| 0.9)));

        let score = agg.score_step("test", &[]);
        assert!((score.score - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_prm_aggregator_product() {
        let mut agg = PrmAggregator::new(AggregationStrategy::Product);
        agg.add_model(Box::new(LlmPRM::new("a", |_, _| 0.5)));
        agg.add_model(Box::new(LlmPRM::new("b", |_, _| 0.8)));

        let score = agg.score_step("test", &[]);
        assert!((score.score - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_prm_aggregator_weighted_average() {
        let mut agg = PrmAggregator::new(AggregationStrategy::WeightedAverage(vec![3.0, 1.0]));
        agg.add_model(Box::new(LlmPRM::new("a", |_, _| 0.8)));
        agg.add_model(Box::new(LlmPRM::new("b", |_, _| 0.4)));

        let score = agg.score_step("test", &[]);
        // (0.8*3 + 0.4*1) / 4 = 2.8 / 4 = 0.7
        assert!((score.score - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_prm_aggregator_empty() {
        let agg = PrmAggregator::new(AggregationStrategy::Average);
        let score = agg.score_step("test", &[]);
        assert!((score.score - 0.0).abs() < 1e-10);
        assert_eq!(agg.model_count(), 0);
    }

    // =========================================================================
    // RefinementConfig tests
    // =========================================================================

    #[test]
    fn test_refinement_config_defaults() {
        let config = RefinementConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert!((config.improvement_threshold - 0.01).abs() < 1e-10);
        assert!(config.replan_on_failure);
    }

    // =========================================================================
    // RefinementLoop tests
    // =========================================================================

    #[test]
    fn test_refinement_loop_should_refine_all_success() {
        let rl = RefinementLoop::with_defaults();
        let feedbacks = vec![
            ExecutionFeedback {
                step_index: 0,
                success: true,
                actual_reward: 0.5,
                expected_reward: 0.5,
                error_message: None,
            },
            ExecutionFeedback {
                step_index: 1,
                success: true,
                actual_reward: 0.8,
                expected_reward: 0.8,
                error_message: None,
            },
        ];
        assert!(!rl.should_refine(&feedbacks));
    }

    #[test]
    fn test_refinement_loop_should_refine_with_failure() {
        let rl = RefinementLoop::with_defaults();
        let feedbacks = vec![
            ExecutionFeedback {
                step_index: 0,
                success: true,
                actual_reward: 0.5,
                expected_reward: 0.5,
                error_message: None,
            },
            ExecutionFeedback {
                step_index: 1,
                success: false,
                actual_reward: 0.0,
                expected_reward: 0.8,
                error_message: Some("Timeout".into()),
            },
        ];
        assert!(rl.should_refine(&feedbacks));
    }

    #[test]
    fn test_refinement_loop_should_refine_empty() {
        let rl = RefinementLoop::with_defaults();
        assert!(!rl.should_refine(&[]));
    }

    #[test]
    fn test_refinement_loop_identify_failure_point() {
        let rl = RefinementLoop::with_defaults();
        let feedbacks = vec![
            ExecutionFeedback {
                step_index: 0,
                success: true,
                actual_reward: 0.5,
                expected_reward: 0.5,
                error_message: None,
            },
            ExecutionFeedback {
                step_index: 1,
                success: false,
                actual_reward: 0.0,
                expected_reward: 0.8,
                error_message: Some("Error".into()),
            },
            ExecutionFeedback {
                step_index: 2,
                success: false,
                actual_reward: 0.0,
                expected_reward: 0.7,
                error_message: Some("Error".into()),
            },
        ];
        // Should return the first failure
        assert_eq!(rl.identify_failure_point(&feedbacks), Some(1));
    }

    #[test]
    fn test_refinement_loop_identify_failure_point_none() {
        let rl = RefinementLoop::with_defaults();
        let feedbacks = vec![ExecutionFeedback {
            step_index: 0,
            success: true,
            actual_reward: 1.0,
            expected_reward: 1.0,
            error_message: None,
        }];
        assert_eq!(rl.identify_failure_point(&feedbacks), None);
    }

    #[test]
    fn test_refinement_loop_compute_improvement() {
        let rl = RefinementLoop::with_defaults();

        // 50% improvement
        let imp = rl.compute_improvement(0.5, 0.75);
        assert!((imp - 0.5).abs() < 1e-10);

        // 100% improvement
        let imp = rl.compute_improvement(0.5, 1.0);
        assert!((imp - 1.0).abs() < 1e-10);

        // No improvement
        let imp = rl.compute_improvement(0.5, 0.5);
        assert!((imp - 0.0).abs() < 1e-10);

        // Negative improvement
        let imp = rl.compute_improvement(0.5, 0.25);
        assert!((imp - (-0.5)).abs() < 1e-10);

        // Zero initial (uses 1e-10 denominator)
        let imp = rl.compute_improvement(0.0, 0.5);
        assert!(imp > 0.0);
    }

    // =========================================================================
    // ExecutionFeedback tests
    // =========================================================================

    #[test]
    fn test_execution_feedback_construction() {
        let fb = ExecutionFeedback {
            step_index: 3,
            success: false,
            actual_reward: 0.2,
            expected_reward: 0.8,
            error_message: Some("Timeout after 30s".into()),
        };
        assert_eq!(fb.step_index, 3);
        assert!(!fb.success);
        assert!((fb.actual_reward - 0.2).abs() < 1e-10);
        assert!((fb.expected_reward - 0.8).abs() < 1e-10);
        assert_eq!(fb.error_message.as_deref(), Some("Timeout after 30s"));
    }

    // =========================================================================
    // RefinementStrategy tests
    // =========================================================================

    #[test]
    fn test_refinement_strategy_variants() {
        // Verify all variants exist and can be constructed
        let _s1 = RefinementStrategy::ReplanFromFailure;
        let _s2 = RefinementStrategy::ReplanEntire;
        let _s3 = RefinementStrategy::PatchStep;

        // Verify serialization round-trip
        let json = serde_json::to_string(&RefinementStrategy::ReplanEntire).unwrap();
        let deser: RefinementStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, RefinementStrategy::ReplanEntire));
    }

    // =========================================================================
    // RefinementResult tests
    // =========================================================================

    #[test]
    fn test_refinement_result_converged() {
        let result = RefinementResult {
            iterations_used: 3,
            initial_reward: 0.5,
            final_reward: 0.95,
            improvement: 0.9,
            strategy_used: RefinementStrategy::ReplanFromFailure,
            converged: true,
        };
        assert!(result.converged);
        assert_eq!(result.iterations_used, 3);
        assert!((result.improvement - 0.9).abs() < 1e-10);
    }

    // =========================================================================
    // RefinementLoop::evaluate_plan tests
    // =========================================================================

    #[test]
    fn test_refinement_loop_evaluate_plan_no_refinement_needed() {
        let rl = RefinementLoop::with_defaults();
        let config = MctsConfig {
            max_iterations: 50,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::new(config);

        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["a".into(), "b".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };

        // All steps succeeded with matching rewards
        let feedbacks = vec![ExecutionFeedback {
            step_index: 0,
            success: true,
            actual_reward: 0.5,
            expected_reward: 0.5,
            error_message: None,
        }];

        let result = rl.evaluate_plan(&planner, &state, &feedbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations_used, 0);
    }

    #[test]
    fn test_refinement_loop_evaluate_plan_with_failure() {
        let rl = RefinementLoop::new(
            RefinementConfig {
                max_iterations: 3,
                improvement_threshold: 0.01,
                replan_on_failure: true,
            },
            RefinementStrategy::ReplanEntire,
        );

        let config = MctsConfig {
            max_iterations: 50,
            ..MctsConfig::default()
        };
        let planner = MctsPlanner::new(config);

        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec!["search".into(), "analyze".into(), "report".into()],
            remaining_budget: 5.0,
            accumulated_reward: 0.0,
            max_steps: 5,
        };

        let feedbacks = vec![
            ExecutionFeedback {
                step_index: 0,
                success: true,
                actual_reward: 0.1,
                expected_reward: 0.1,
                error_message: None,
            },
            ExecutionFeedback {
                step_index: 1,
                success: false,
                actual_reward: 0.0,
                expected_reward: 0.1,
                error_message: Some("Failed".into()),
            },
        ];

        let result = rl.evaluate_plan(&planner, &state, &feedbacks).unwrap();
        assert_eq!(result.iterations_used, 1);
        assert!(matches!(result.strategy_used, RefinementStrategy::ReplanEntire));
    }

    // =========================================================================
    // Full integration test
    // =========================================================================

    #[test]
    fn test_full_mcts_search_produces_valid_result() {
        let state = AgentMctsState {
            current_step: 0,
            completed_actions: Vec::new(),
            available_tools: vec![
                "search".into(),
                "analyze".into(),
                "summarize".into(),
                "report".into(),
            ],
            remaining_budget: 10.0,
            accumulated_reward: 0.0,
            max_steps: 10,
        };

        let config = MctsConfig {
            max_iterations: 300,
            exploration_constant: std::f64::consts::SQRT_2,
            max_depth: 10,
            simulation_depth: 10,
            discount_factor: 0.99,
        };
        let planner = MctsPlanner::new(config);
        let result = planner.search(&state).unwrap();

        // Verify the result is coherent
        assert!(result.best_action.is_some());
        let action = result.best_action.unwrap();
        assert!(state.available_tools.contains(&action));

        // The best action sequence should contain only valid tools
        for a in &result.best_action_sequence {
            assert!(state.available_tools.contains(a));
        }

        // Sequence should not contain duplicates (since our state doesn't allow reuse)
        let mut seen = std::collections::HashSet::new();
        for a in &result.best_action_sequence {
            assert!(seen.insert(a.clone()), "Duplicate action in sequence: {}", a);
        }

        assert!(result.root_value >= 0.0);
        assert!(result.iterations_used > 0);
        assert!(result.total_simulations > 0);
    }
}
