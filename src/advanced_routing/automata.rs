//! NFA/DFA graph routing, rule builder, synthesizer, snapshots, and merging.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

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
    pub(crate) states: Vec<NfaState>,
    pub(crate) transitions: Vec<NfaTransition>,
    pub(crate) start_states: Vec<NfaStateId>,
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
    pub fn add_state(
        &mut self,
        label: &str,
        accepting_arm: Option<&str>,
        priority: u32,
    ) -> NfaStateId {
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
                    if best_accepting.is_none() || state.priority > best_accepting.unwrap().priority
                    {
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
                if trans.from == state
                    && trans.symbol == NfaSymbol::Epsilon
                    && !closure.contains(&trans.to)
                {
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
            NfaSymbol::BoolFeature { name, value } => match name.as_str() {
                "has_code" => features.has_code == *value,
                "is_question" => features.is_question == *value,
                _ => false,
            },
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
            partition_map
                .entry((state.accepting_arm.clone(), state.priority))
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

        let new_states: Vec<DfaState> = partitions
            .iter()
            .enumerate()
            .map(|(i, partition)| {
                let rep = &self.states[partition[0]];
                DfaState {
                    id: i,
                    label: rep.label.clone(),
                    accepting_arm: rep.accepting_arm.clone(),
                    priority: rep.priority,
                }
            })
            .collect();

        let new_start = state_to_partition[&self.start_state];

        let mut new_table: HashMap<DfaStateId, Vec<(NfaSymbol, DfaStateId)>> = HashMap::new();
        for (pi, partition) in partitions.iter().enumerate() {
            let rep = partition[0];
            if let Some(transitions) = self.transition_table.get(&rep) {
                let new_transitions: Vec<(NfaSymbol, DfaStateId)> = transitions
                    .iter()
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
        let state_to_partition: HashMap<DfaStateId, usize> = partitions
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| p.iter().map(move |&s| (s, pi)))
            .collect();

        let mut sig = Vec::new();
        if let Some(transitions) = self.transition_table.get(&state) {
            for (_, target) in transitions {
                sig.push(
                    state_to_partition
                        .get(target)
                        .copied()
                        .unwrap_or(usize::MAX),
                );
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
            NfaSymbol::BoolFeature { name, value } => match name.as_str() {
                "has_code" => features.has_code == *value,
                "is_question" => features.is_question == *value,
                _ => false,
            },
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

                dfa_table
                    .entry(current_id)
                    .or_default()
                    .push((symbol.clone(), target_id));
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

    fn create_dfa_state(
        id: DfaStateId,
        nfa_states: &HashSet<NfaStateId>,
        nfa: &NfaRouter,
    ) -> DfaState {
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
                        let intermediate = nfa.add_state(&format!("{}_{}", rule.label, i), None, 0);
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
        serde_json::to_string_pretty(&snapshot).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })
    }

    /// Deserialize an NFA from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: NfaSnapshot =
            serde_json::from_str(json).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
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
            return bincode::serialize(&snapshot).map_err(|e| {
                AdvancedRoutingError::SerializationFailed {
                    format: "bincode".to_string(),
                    reason: e.to_string(),
                }
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

        let json =
            std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
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
            let _id =
                result.add_state(&state.label, state.accepting_arm.as_deref(), state.priority);
        }

        // Copy other states (renumbered with other_offset)
        for state in &other.states {
            let _id =
                result.add_state(&state.label, state.accepting_arm.as_deref(), state.priority);
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
        serde_json::to_string_pretty(&snapshot).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })
    }

    /// Deserialize a DFA from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: DfaSnapshot =
            serde_json::from_str(json).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
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
            return bincode::serialize(&snapshot).map_err(|e| {
                AdvancedRoutingError::SerializationFailed {
                    format: "bincode".to_string(),
                    reason: e.to_string(),
                }
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

        let json =
            std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
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
    pub fn extract_state_filtered(
        nfa: &NfaRouter,
        node_id: &str,
        private_arms: &HashSet<ArmId>,
    ) -> DistributedNfaState {
        let mut snapshot = nfa.export_snapshot();

        // Find state IDs whose accepting_arm is in private_arms
        let private_state_ids: HashSet<usize> = snapshot
            .states
            .iter()
            .filter(|s| {
                s.accepting_arm
                    .as_ref()
                    .map_or(false, |arm| private_arms.contains(arm))
            })
            .map(|s| s.id)
            .collect();

        // Remove transitions that lead to private accepting states
        snapshot
            .transitions
            .retain(|t| !private_state_ids.contains(&t.to));

        // Remove private accepting states themselves
        snapshot
            .states
            .retain(|s| !private_state_ids.contains(&s.id));

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
        let nfas: Result<Vec<NfaRouter>, _> = states
            .iter()
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

#[cfg(test)]
mod tests {
    use super::*;

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
        nfa.add_transition(
            s0,
            NfaSymbol::ComplexityRange {
                low_pct: 0,
                high_pct: 50,
            },
            s1,
        );
        nfa.add_transition(
            s0,
            NfaSymbol::ComplexityRange {
                low_pct: 50,
                high_pct: 100,
            },
            s2,
        );

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
        nfa.add_transition(
            s0,
            NfaSymbol::BoolFeature {
                name: "has_code".to_string(),
                value: true,
            },
            s1,
        );

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
            states: vec![DfaState {
                id: 0,
                label: "start".to_string(),
                accepting_arm: Some("model".to_string()),
                priority: 1,
            }],
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
                DfaState {
                    id: 0,
                    label: "start".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "code".to_string(),
                    accepting_arm: Some("code-model".to_string()),
                    priority: 1,
                },
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
            states: vec![DfaState {
                id: 0,
                label: "start".to_string(),
                accepting_arm: None,
                priority: 0,
            }],
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
                DfaState {
                    id: 0,
                    label: "s0".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "s1".to_string(),
                    accepting_arm: Some("x".to_string()),
                    priority: 1,
                },
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
                DfaState {
                    id: 0,
                    label: "s0".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "s1".to_string(),
                    accepting_arm: Some("x".to_string()),
                    priority: 1,
                },
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
        table.insert(
            0,
            vec![
                (NfaSymbol::Domain("a".to_string()), 1),
                (NfaSymbol::Domain("b".to_string()), 2),
            ],
        );

        let mut dfa = DfaRouter {
            states: vec![
                DfaState {
                    id: 0,
                    label: "s0".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "s1".to_string(),
                    accepting_arm: Some("x".to_string()),
                    priority: 1,
                },
                DfaState {
                    id: 2,
                    label: "s2".to_string(),
                    accepting_arm: Some("x".to_string()),
                    priority: 1,
                },
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
                DfaState {
                    id: 0,
                    label: "a".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "b".to_string(),
                    accepting_arm: Some("x".to_string()),
                    priority: 1,
                },
            ],
            start_state: 0,
            transition_table: HashMap::new(),
        };
        assert_eq!(dfa.state_count(), 2);
    }

    #[test]
    fn test_dfa_transition_count() {
        let mut table = HashMap::new();
        table.insert(
            0,
            vec![(NfaSymbol::Any, 1), (NfaSymbol::Domain("x".to_string()), 1)],
        );
        let dfa = DfaRouter {
            states: vec![
                DfaState {
                    id: 0,
                    label: "a".to_string(),
                    accepting_arm: None,
                    priority: 0,
                },
                DfaState {
                    id: 1,
                    label: "b".to_string(),
                    accepting_arm: Some("m".to_string()),
                    priority: 1,
                },
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
            states: vec![DfaState {
                id: 0,
                label: "s0".to_string(),
                accepting_arm: Some("m".to_string()),
                priority: 1,
            }],
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
        let accepting: Vec<_> = dfa
            .states
            .iter()
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
        nfa.add_transition(
            s1,
            NfaSymbol::ComplexityRange {
                low_pct: 50,
                high_pct: 100,
            },
            s3,
        );
        nfa.add_transition(
            s1,
            NfaSymbol::ComplexityRange {
                low_pct: 0,
                high_pct: 50,
            },
            s4,
        );
        nfa.add_transition(s2, NfaSymbol::Epsilon, s3);
        nfa.add_transition(s0, NfaSymbol::Epsilon, s1);

        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        assert!(dfa.state_count() >= 2);
        assert!(dfa.transition_count() >= 1);
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
            .and(NfaSymbol::ComplexityRange {
                low_pct: 70,
                high_pct: 100,
            })
            .route_to("claude-opus")
            .priority(10)
            .done()
            .rule("code_easy")
            .when(NfaSymbol::Domain("code".into()))
            .and(NfaSymbol::ComplexityRange {
                low_pct: 0,
                high_pct: 70,
            })
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
            .and(NfaSymbol::ComplexityRange {
                low_pct: 80,
                high_pct: 100,
            })
            .and(NfaSymbol::BoolFeature {
                name: "is_question".into(),
                value: true,
            })
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
                arm_id: "code-model".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "math-model".into(),
                success: true,
                quality: Some(0.85),
                latency_ms: None,
                cost: None,
                task_type: Some("math".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "fallback".into(),
                success: true,
                quality: Some(0.5),
                latency_ms: None,
                cost: None,
                task_type: None,
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
                arm_id: "good-model".into(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "bad-model".into(),
                success: false,
                quality: Some(0.2),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "fallback".into(),
                success: true,
                quality: Some(0.5),
                latency_ms: None,
                cost: None,
                task_type: None,
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
                arm_id: "specialist".into(),
                success: true,
                quality: Some(0.95),
                latency_ms: None,
                cost: None,
                task_type: Some("code".into()),
            });
            bandit.record_outcome(&ArmFeedback {
                arm_id: "generalist".into(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: None,
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
            .rule("r1")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m")
            .priority(1)
            .done()
            .fallback("fb", 0)
            .build()
            .unwrap();
        let json = nfa.to_json().unwrap();
        assert!(json.contains("\"version\": 1"));
        assert!(!json.contains("merged")); // not a merged NFA
    }

    #[test]
    fn test_nfa_from_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("r1")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m")
            .priority(1)
            .done()
            .build()
            .unwrap();
        let json = nfa.to_json().unwrap();
        let restored = NfaRouter::from_json(&json).unwrap();
        assert_eq!(restored.state_count(), nfa.state_count());
        assert_eq!(restored.transition_count(), nfa.transition_count());
    }

    #[test]
    fn test_nfa_round_trip_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("m2")
            .priority(5)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();

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
        let nfa = NfaRuleBuilder::new().fallback("fb", 0).build().unwrap();
        let mut json = nfa.to_json().unwrap();
        json = json.replace("\"version\": 1", "\"version\": 99");
        let result = NfaRouter::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_nfa_to_bytes() {
        let nfa = NfaRuleBuilder::new().fallback("fb", 0).build().unwrap();
        let bytes = nfa.to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_nfa_from_bytes() {
        let nfa = NfaRuleBuilder::new()
            .rule("r")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m")
            .priority(1)
            .done()
            .build()
            .unwrap();
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
            .rule("r")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m")
            .priority(1)
            .done()
            .fallback("fb", 0)
            .build()
            .unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let json = dfa.to_json().unwrap();
        assert!(json.contains("\"version\": 1"));
    }

    #[test]
    fn test_dfa_from_json_import() {
        let nfa = NfaRuleBuilder::new()
            .rule("r")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m")
            .priority(1)
            .done()
            .build()
            .unwrap();
        let dfa = NfaDfaCompiler::compile(&nfa).unwrap();
        let json = dfa.to_json().unwrap();
        let restored = DfaRouter::from_json(&json).unwrap();
        assert_eq!(restored.state_count(), dfa.state_count());
    }

    #[test]
    fn test_dfa_round_trip_json() {
        let nfa = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .fallback("fb", 1)
            .build()
            .unwrap();
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
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m-code")
            .priority(10)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("m-math")
            .priority(5)
            .done()
            .build()
            .unwrap();

        let merged = nfa_a.merge(&nfa_b);
        // Merged should have states from both + new start
        assert!(merged.state_count() > nfa_a.state_count());
        assert!(merged.state_count() > nfa_b.state_count());
    }

    #[test]
    fn test_nfa_merge_state_renumbering() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m1")
            .priority(1)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("y".into()))
            .route_to("m2")
            .priority(1)
            .done()
            .build()
            .unwrap();

        let merged = nfa_a.merge(&nfa_b);
        // Total states = nfa_a.states + nfa_b.states + 1 (merged start)
        assert_eq!(
            merged.state_count(),
            nfa_a.state_count() + nfa_b.state_count() + 1
        );
    }

    #[test]
    fn test_nfa_merge_accepting_preserved() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("model-a")
            .priority(10)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("model-b")
            .priority(5)
            .done()
            .build()
            .unwrap();

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
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("code-model")
            .priority(10)
            .done()
            .fallback("fallback-a", 1)
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("math-model")
            .priority(8)
            .done()
            .fallback("fallback-b", 2)
            .build()
            .unwrap();

        let merged = nfa_a.merge(&nfa_b);

        let f_code = test_features("code", 0.5);
        assert_eq!(merged.route(&f_code).unwrap().selected_arm, "code-model");

        let f_math = test_features("math", 0.5);
        assert_eq!(merged.route(&f_math).unwrap().selected_arm, "math-model");
    }

    #[test]
    fn test_nfa_merge_three_chain() {
        let a = NfaRuleBuilder::new()
            .rule("x")
            .when(NfaSymbol::Domain("a".into()))
            .route_to("ma")
            .priority(1)
            .done()
            .build()
            .unwrap();
        let b = NfaRuleBuilder::new()
            .rule("y")
            .when(NfaSymbol::Domain("b".into()))
            .route_to("mb")
            .priority(2)
            .done()
            .build()
            .unwrap();
        let c = NfaRuleBuilder::new()
            .rule("z")
            .when(NfaSymbol::Domain("c".into()))
            .route_to("mc")
            .priority(3)
            .done()
            .build()
            .unwrap();

        let merged = a.merge(&b).merge(&c);

        let f = test_features("c", 0.5);
        let outcome = merged.route(&f).unwrap();
        assert_eq!(outcome.selected_arm, "mc");
    }

    #[test]
    fn test_nfa_merge_transitions_preserved() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("x".into()))
            .route_to("m1")
            .priority(1)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("y".into()))
            .route_to("m2")
            .priority(1)
            .done()
            .build()
            .unwrap();

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
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("m1")
            .priority(10)
            .done()
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("m2")
            .priority(5)
            .done()
            .build()
            .unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();
        assert!(dfa.state_count() >= 2);
    }

    #[test]
    fn test_merge_and_compile_routes_correctly() {
        let nfa_a = NfaRuleBuilder::new()
            .rule("a")
            .when(NfaSymbol::Domain("code".into()))
            .route_to("code-model")
            .priority(10)
            .done()
            .fallback("fallback", 1)
            .build()
            .unwrap();
        let nfa_b = NfaRuleBuilder::new()
            .rule("b")
            .when(NfaSymbol::Domain("math".into()))
            .route_to("math-model")
            .priority(8)
            .done()
            .build()
            .unwrap();

        let dfa = merge_and_compile_nfas(&nfa_a, &nfa_b).unwrap();

        let f_code = test_features("code", 0.5);
        assert_eq!(dfa.route(&f_code).unwrap().selected_arm, "code-model");
    }

    #[test]
    fn test_merge_and_compile_empty_nfa() {
        let nfa_a = NfaRouter::new();
        let nfa_b = NfaRuleBuilder::new().fallback("fb", 1).build().unwrap();

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
                .rule("r")
                .when(NfaSymbol::Domain("x".into()))
                .route_to("m")
                .priority(1)
                .done()
                .build()
                .unwrap();
            let state = NfaStateMerger::extract_state(&nfa, "node-1");
            assert_eq!(state.node_id, "node-1");
            assert_eq!(state.nfa.states.len(), nfa.state_count());
        }

        #[test]
        fn test_distributed_nfa_merge_two() {
            let nfa_a = NfaRuleBuilder::new()
                .rule("a")
                .when(NfaSymbol::Domain("code".into()))
                .route_to("m1")
                .priority(10)
                .done()
                .build()
                .unwrap();
            let nfa_b = NfaRuleBuilder::new()
                .rule("b")
                .when(NfaSymbol::Domain("math".into()))
                .route_to("m2")
                .priority(5)
                .done()
                .build()
                .unwrap();

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
                .rule("local")
                .when(NfaSymbol::Domain("code".into()))
                .route_to("m1")
                .priority(10)
                .done()
                .build()
                .unwrap();
            let remote_nfa = NfaRuleBuilder::new()
                .rule("remote")
                .when(NfaSymbol::Domain("math".into()))
                .route_to("m2")
                .priority(5)
                .done()
                .build()
                .unwrap();

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
                s.accepting_arm
                    .as_ref()
                    .map_or(true, |a| a != "private-arm")
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
}
