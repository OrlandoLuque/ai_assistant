//! Decision trees for conditional flow evaluation.
//!
//! This module provides a complete decision tree implementation where nodes can be
//! questions, conditions, actions, or terminal results. Trees can be traversed with
//! a context map to produce a `DecisionPath` recording all visited nodes, actions taken,
//! and the final result.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::mpsc::Sender;

// ─────────────────────────────────────────────────────────────────────────────
// Condition Operator
// ─────────────────────────────────────────────────────────────────────────────

/// Operators used for evaluating conditions against context variables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    /// Regex match against the variable value.
    Matches,
    /// True if the variable is empty string, null, or empty array.
    IsEmpty,
    /// True if the variable is non-empty.
    IsNotEmpty,
    /// True if the value (as array) contains the variable value.
    InList,
}

// ─────────────────────────────────────────────────────────────────────────────
// Condition
// ─────────────────────────────────────────────────────────────────────────────

/// A single condition that checks a context variable against a value using an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub variable: String,
    pub operator: ConditionOperator,
    pub value: Value,
    pub description: Option<String>,
}

impl Condition {
    /// Create a new condition.
    pub fn new(variable: impl Into<String>, operator: ConditionOperator, value: Value) -> Self {
        Self {
            variable: variable.into(),
            operator,
            value,
            description: None,
        }
    }

    /// Add a description to this condition (builder pattern).
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Evaluate this condition against the provided context.
    ///
    /// Returns `true` if the condition is satisfied, `false` otherwise.
    /// If the variable is not found in the context, it is treated as `Value::Null`.
    pub fn evaluate(&self, context: &HashMap<String, Value>) -> bool {
        let var_value = context.get(&self.variable).cloned().unwrap_or(Value::Null);

        match &self.operator {
            ConditionOperator::Equals => var_value == self.value,
            ConditionOperator::NotEquals => var_value != self.value,

            ConditionOperator::Contains => {
                let var_str = value_to_string(&var_value);
                let val_str = value_to_string(&self.value);
                var_str.contains(&val_str)
            }
            ConditionOperator::NotContains => {
                let var_str = value_to_string(&var_value);
                let val_str = value_to_string(&self.value);
                !var_str.contains(&val_str)
            }
            ConditionOperator::StartsWith => {
                let var_str = value_to_string(&var_value);
                let val_str = value_to_string(&self.value);
                var_str.starts_with(&val_str)
            }
            ConditionOperator::EndsWith => {
                let var_str = value_to_string(&var_value);
                let val_str = value_to_string(&self.value);
                var_str.ends_with(&val_str)
            }

            ConditionOperator::GreaterThan => {
                compare_as_f64(&var_value, &self.value, |a, b| a > b)
            }
            ConditionOperator::LessThan => {
                compare_as_f64(&var_value, &self.value, |a, b| a < b)
            }
            ConditionOperator::GreaterOrEqual => {
                compare_as_f64(&var_value, &self.value, |a, b| a >= b)
            }
            ConditionOperator::LessOrEqual => {
                compare_as_f64(&var_value, &self.value, |a, b| a <= b)
            }

            ConditionOperator::Matches => {
                let var_str = value_to_string(&var_value);
                let pattern = value_to_string(&self.value);
                match regex::Regex::new(&pattern) {
                    Ok(re) => re.is_match(&var_str),
                    Err(_) => false,
                }
            }

            ConditionOperator::IsEmpty => is_empty_value(&var_value),
            ConditionOperator::IsNotEmpty => !is_empty_value(&var_value),

            ConditionOperator::InList => {
                if let Value::Array(arr) = &self.value {
                    arr.contains(&var_value)
                } else {
                    false
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a JSON value to a string representation for string operations.
fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Compare two values as f64 using the given comparator.
fn compare_as_f64(a: &Value, b: &Value, cmp: fn(f64, f64) -> bool) -> bool {
    let a_f64 = value_as_f64(a);
    let b_f64 = value_as_f64(b);
    match (a_f64, b_f64) {
        (Some(av), Some(bv)) => cmp(av, bv),
        _ => false,
    }
}

/// Try to extract an f64 from a JSON value.
fn value_as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Check if a value is considered "empty".
fn is_empty_value(val: &Value) -> bool {
    match val {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(arr) => arr.is_empty(),
        Value::Object(obj) => obj.is_empty(),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Branch
// ─────────────────────────────────────────────────────────────────────────────

/// A branch in a condition node that pairs a condition with a target node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBranch {
    pub condition: Condition,
    pub target_node_id: String,
    pub label: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Node Type
// ─────────────────────────────────────────────────────────────────────────────

/// The type and associated data for a decision tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionNodeType {
    /// Evaluate branches in order; first match wins, else use default.
    Condition {
        branches: Vec<DecisionBranch>,
        default_branch: Option<String>,
    },
    /// Perform an action, then continue to next node.
    Action {
        action_type: String,
        parameters: HashMap<String, Value>,
        next_node: Option<String>,
    },
    /// End node with a result value.
    Terminal {
        result: Value,
        label: Option<String>,
    },
    /// Ask for user input, store in a variable, then continue.
    Question {
        prompt: String,
        variable: String,
        options: Vec<String>,
        next_node: Option<String>,
    },
    /// Execute an LLM prompt, store result in context variable.
    /// Prompt text supports `{{variable}}` template substitution.
    Prompt {
        system_prompt: Option<String>,
        user_prompt: String,
        output_variable: String,
        next_node: Option<String>,
    },
    /// Execute a registered function by name, store result in context.
    /// Arguments support `{{variable}}` template substitution.
    Function {
        function_name: String,
        arguments: HashMap<String, Value>,
        output_variable: Option<String>,
        next_node: Option<String>,
    },
    /// Execute children in order; stop on first failure.
    /// All children must reach a Terminal with a non-error result.
    Sequence {
        children: Vec<String>,
    },
    /// Try children in order; return first successful result.
    Selector {
        children: Vec<String>,
    },
    /// Execute children concurrently, collect results.
    Parallel {
        children: Vec<String>,
        require_all: bool,
    },
    /// Reference another decision tree by ID.
    /// Maps parent context into child tree and stores child result.
    SubTree {
        tree_id: String,
        input_mapping: HashMap<String, String>,
        output_variable: Option<String>,
        next_node: Option<String>,
    },
    /// Use an LLM to decide which branch to take.
    /// The LLM response is matched against branch labels (case-insensitive).
    LlmCondition {
        prompt: String,
        branches: HashMap<String, String>,
        default_branch: Option<String>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Node
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in a decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub id: String,
    pub node_type: DecisionNodeType,
    pub description: Option<String>,
    pub metadata: HashMap<String, Value>,
}

impl DecisionNode {
    /// Create a new condition node.
    pub fn new_condition(
        id: impl Into<String>,
        branches: Vec<DecisionBranch>,
        default: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Condition {
                branches,
                default_branch: default,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new action node.
    pub fn new_action(
        id: impl Into<String>,
        action_type: impl Into<String>,
        parameters: HashMap<String, Value>,
        next: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Action {
                action_type: action_type.into(),
                parameters,
                next_node: next,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new terminal node.
    pub fn new_terminal(
        id: impl Into<String>,
        result: Value,
        label: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Terminal { result, label },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new question node.
    pub fn new_question(
        id: impl Into<String>,
        prompt: impl Into<String>,
        variable: impl Into<String>,
        options: Vec<String>,
        next: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Question {
                prompt: prompt.into(),
                variable: variable.into(),
                options,
                next_node: next,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new LLM prompt node.
    pub fn new_prompt(
        id: impl Into<String>,
        user_prompt: impl Into<String>,
        output_variable: impl Into<String>,
        next: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Prompt {
                system_prompt: None,
                user_prompt: user_prompt.into(),
                output_variable: output_variable.into(),
                next_node: next,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new function call node.
    pub fn new_function(
        id: impl Into<String>,
        function_name: impl Into<String>,
        arguments: HashMap<String, Value>,
        output_variable: Option<String>,
        next: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Function {
                function_name: function_name.into(),
                arguments,
                output_variable,
                next_node: next,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new sequence node.
    pub fn new_sequence(id: impl Into<String>, children: Vec<String>) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Sequence { children },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new selector node.
    pub fn new_selector(id: impl Into<String>, children: Vec<String>) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Selector { children },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new parallel node.
    pub fn new_parallel(id: impl Into<String>, children: Vec<String>, require_all: bool) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::Parallel { children, require_all },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new subtree reference node.
    pub fn new_subtree(
        id: impl Into<String>,
        tree_id: impl Into<String>,
        input_mapping: HashMap<String, String>,
        output_variable: Option<String>,
        next: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::SubTree {
                tree_id: tree_id.into(),
                input_mapping,
                output_variable,
                next_node: next,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new LLM condition node.
    pub fn new_llm_condition(
        id: impl Into<String>,
        prompt: impl Into<String>,
        branches: HashMap<String, String>,
        default_branch: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: DecisionNodeType::LlmCondition {
                prompt: prompt.into(),
                branches,
                default_branch,
            },
            description: None,
            metadata: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Path
// ─────────────────────────────────────────────────────────────────────────────

/// The result of traversing a decision tree, recording the path taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPath {
    /// IDs of nodes visited during traversal, in order.
    pub nodes_visited: Vec<String>,
    /// Actions taken: (action_type, parameters).
    pub actions_taken: Vec<(String, HashMap<String, Value>)>,
    /// The terminal result, if reached.
    pub result: Option<Value>,
    /// Whether traversal reached a terminal node.
    pub complete: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Tree
// ─────────────────────────────────────────────────────────────────────────────

/// A complete decision tree with named nodes and a root entry point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub nodes: HashMap<String, DecisionNode>,
    pub root_node_id: String,
    pub metadata: HashMap<String, Value>,
}

impl DecisionTree {
    /// Create a new decision tree.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        root_node_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            nodes: HashMap::new(),
            root_node_id: root_node_id.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add a description (builder pattern).
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a node to the tree.
    pub fn add_node(&mut self, node: DecisionNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node by ID, returning it if found.
    pub fn remove_node(&mut self, id: &str) -> Option<DecisionNode> {
        self.nodes.remove(id)
    }

    /// Get a reference to a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&DecisionNode> {
        self.nodes.get(id)
    }

    /// Traverse the tree from the root, evaluating conditions against the context.
    ///
    /// Returns a `DecisionPath` recording nodes visited, actions taken, and the result.
    /// Detects cycles by tracking visited nodes.
    pub fn evaluate(&self, context: &HashMap<String, Value>) -> DecisionPath {
        let mut path = DecisionPath {
            nodes_visited: Vec::new(),
            actions_taken: Vec::new(),
            result: None,
            complete: false,
        };

        let mut current_id = self.root_node_id.clone();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();

        loop {
            // Cycle detection
            if visited.contains(&current_id) {
                break;
            }
            visited.insert(current_id.clone());

            let node = match self.nodes.get(&current_id) {
                Some(n) => n,
                None => break,
            };

            path.nodes_visited.push(current_id.clone());

            match &node.node_type {
                DecisionNodeType::Condition {
                    branches,
                    default_branch,
                } => {
                    let mut next = None;
                    for branch in branches {
                        if branch.condition.evaluate(context) {
                            next = Some(branch.target_node_id.clone());
                            break;
                        }
                    }
                    if next.is_none() {
                        next = default_branch.clone();
                    }
                    match next {
                        Some(next_id) => current_id = next_id,
                        None => break,
                    }
                }

                DecisionNodeType::Action {
                    action_type,
                    parameters,
                    next_node,
                } => {
                    path.actions_taken
                        .push((action_type.clone(), parameters.clone()));
                    match next_node {
                        Some(next_id) => current_id = next_id.clone(),
                        None => break,
                    }
                }

                DecisionNodeType::Terminal { result, .. } => {
                    path.result = Some(result.clone());
                    path.complete = true;
                    break;
                }

                DecisionNodeType::Question { .. } => {
                    // Question nodes require external input; stop traversal.
                    path.complete = false;
                    break;
                }

                DecisionNodeType::Prompt { .. } => {
                    // Prompt nodes require LLM; stop traversal in sync mode.
                    path.complete = false;
                    break;
                }

                DecisionNodeType::LlmCondition { .. } => {
                    // LLM condition nodes require LLM; stop traversal in sync mode.
                    path.complete = false;
                    break;
                }

                DecisionNodeType::Function {
                    function_name,
                    arguments,
                    next_node,
                    ..
                } => {
                    // Record as action (actual execution requires a ToolRegistry)
                    let resolved_args: HashMap<String, Value> = arguments
                        .iter()
                        .map(|(k, v)| (k.clone(), substitute_value(v, context)))
                        .collect();
                    path.actions_taken
                        .push((function_name.clone(), resolved_args));
                    match next_node {
                        Some(next_id) => current_id = next_id.clone(),
                        None => break,
                    }
                }

                DecisionNodeType::Sequence { children } => {
                    // Execute children in order; all must complete successfully
                    let mut last_result = None;
                    let mut all_ok = true;
                    for child_id in children {
                        let child_path = self.evaluate_subtree(child_id, context);
                        path.nodes_visited.extend(child_path.nodes_visited);
                        path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            last_result = child_path.result;
                        } else {
                            all_ok = false;
                            break;
                        }
                    }
                    if all_ok {
                        path.result = last_result;
                        path.complete = true;
                    }
                    break;
                }

                DecisionNodeType::Selector { children } => {
                    // Try children in order; first success wins
                    for child_id in children {
                        let child_path = self.evaluate_subtree(child_id, context);
                        path.nodes_visited.extend(child_path.nodes_visited);
                        path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            path.result = child_path.result;
                            path.complete = true;
                            break;
                        }
                    }
                    break;
                }

                DecisionNodeType::Parallel { children, require_all } => {
                    // Execute all children, collect results
                    let mut results = Vec::new();
                    let mut all_complete = true;
                    for child_id in children {
                        let child_path = self.evaluate_subtree(child_id, context);
                        path.nodes_visited.extend(child_path.nodes_visited);
                        path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            if let Some(r) = child_path.result {
                                results.push(r);
                            }
                        } else {
                            all_complete = false;
                        }
                    }
                    if *require_all && all_complete || !*require_all && !results.is_empty() {
                        path.result = Some(Value::Array(results));
                        path.complete = true;
                    }
                    break;
                }

                DecisionNodeType::SubTree { .. } => {
                    // SubTree nodes require a tree registry; stop traversal in basic mode.
                    path.complete = false;
                    break;
                }
            }
        }

        path
    }

    /// Evaluate a subtree starting from a given node ID (used by Sequence/Selector/Parallel).
    fn evaluate_subtree(&self, start_id: &str, context: &HashMap<String, Value>) -> DecisionPath {
        let mut path = DecisionPath {
            nodes_visited: Vec::new(),
            actions_taken: Vec::new(),
            result: None,
            complete: false,
        };

        let mut current_id = start_id.to_string();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();

        loop {
            if visited.contains(&current_id) {
                break;
            }
            visited.insert(current_id.clone());

            let node = match self.nodes.get(&current_id) {
                Some(n) => n,
                None => break,
            };

            path.nodes_visited.push(current_id.clone());

            match &node.node_type {
                DecisionNodeType::Condition { branches, default_branch } => {
                    let mut next = None;
                    for branch in branches {
                        if branch.condition.evaluate(context) {
                            next = Some(branch.target_node_id.clone());
                            break;
                        }
                    }
                    if next.is_none() {
                        next = default_branch.clone();
                    }
                    match next {
                        Some(next_id) => current_id = next_id,
                        None => break,
                    }
                }
                DecisionNodeType::Action { action_type, parameters, next_node } => {
                    path.actions_taken.push((action_type.clone(), parameters.clone()));
                    match next_node {
                        Some(next_id) => current_id = next_id.clone(),
                        None => break,
                    }
                }
                DecisionNodeType::Terminal { result, .. } => {
                    path.result = Some(result.clone());
                    path.complete = true;
                    break;
                }
                DecisionNodeType::Function { function_name, arguments, next_node, .. } => {
                    let resolved_args: HashMap<String, Value> = arguments
                        .iter()
                        .map(|(k, v)| (k.clone(), substitute_value(v, context)))
                        .collect();
                    path.actions_taken.push((function_name.clone(), resolved_args));
                    match next_node {
                        Some(next_id) => current_id = next_id.clone(),
                        None => break,
                    }
                }
                _ => {
                    // Other node types stop subtree evaluation
                    break;
                }
            }
        }

        path
    }

    /// Evaluate a single step from the given node, returning the next node ID if any.
    pub fn evaluate_step(
        &self,
        node_id: &str,
        context: &HashMap<String, Value>,
    ) -> Option<String> {
        let node = self.nodes.get(node_id)?;

        match &node.node_type {
            DecisionNodeType::Condition {
                branches,
                default_branch,
            } => {
                for branch in branches {
                    if branch.condition.evaluate(context) {
                        return Some(branch.target_node_id.clone());
                    }
                }
                default_branch.clone()
            }
            DecisionNodeType::Action { next_node, .. } => next_node.clone(),
            DecisionNodeType::Terminal { .. } => None,
            DecisionNodeType::Question { next_node, .. } => next_node.clone(),
            DecisionNodeType::Prompt { next_node, .. } => next_node.clone(),
            DecisionNodeType::Function { next_node, .. } => next_node.clone(),
            DecisionNodeType::SubTree { next_node, .. } => next_node.clone(),
            DecisionNodeType::Sequence { children } => children.first().cloned(),
            DecisionNodeType::Selector { children } => children.first().cloned(),
            DecisionNodeType::Parallel { children, .. } => children.first().cloned(),
            DecisionNodeType::LlmCondition { default_branch, .. } => default_branch.clone(),
        }
    }

    /// Validate the tree structure, returning a list of issues found.
    ///
    /// Checks for:
    /// - Dangling references (node IDs referenced but not present)
    /// - Unreachable nodes (nodes not reachable from root)
    /// - Missing root node
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check root exists
        if !self.nodes.contains_key(&self.root_node_id) {
            issues.push(format!(
                "Root node '{}' not found in tree",
                self.root_node_id
            ));
        }

        // Collect all referenced node IDs
        let mut referenced_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        referenced_ids.insert(self.root_node_id.clone());

        for node in self.nodes.values() {
            match &node.node_type {
                DecisionNodeType::Condition {
                    branches,
                    default_branch,
                } => {
                    for branch in branches {
                        referenced_ids.insert(branch.target_node_id.clone());
                        if !self.nodes.contains_key(&branch.target_node_id) {
                            issues.push(format!(
                                "Node '{}' references non-existent target '{}'",
                                node.id, branch.target_node_id
                            ));
                        }
                    }
                    if let Some(default) = default_branch {
                        referenced_ids.insert(default.clone());
                        if !self.nodes.contains_key(default) {
                            issues.push(format!(
                                "Node '{}' has non-existent default branch '{}'",
                                node.id, default
                            ));
                        }
                    }
                }
                DecisionNodeType::Action { next_node, .. } => {
                    if let Some(next) = next_node {
                        referenced_ids.insert(next.clone());
                        if !self.nodes.contains_key(next) {
                            issues.push(format!(
                                "Node '{}' references non-existent next node '{}'",
                                node.id, next
                            ));
                        }
                    }
                }
                DecisionNodeType::Question { next_node, .. }
                | DecisionNodeType::Prompt { next_node, .. }
                | DecisionNodeType::Function { next_node, .. }
                | DecisionNodeType::SubTree { next_node, .. } => {
                    if let Some(next) = next_node {
                        referenced_ids.insert(next.clone());
                        if !self.nodes.contains_key(next) {
                            issues.push(format!(
                                "Node '{}' references non-existent next node '{}'",
                                node.id, next
                            ));
                        }
                    }
                }
                DecisionNodeType::Sequence { children }
                | DecisionNodeType::Selector { children }
                | DecisionNodeType::Parallel { children, .. } => {
                    for child_id in children {
                        referenced_ids.insert(child_id.clone());
                        if !self.nodes.contains_key(child_id) {
                            issues.push(format!(
                                "Node '{}' references non-existent child '{}'",
                                node.id, child_id
                            ));
                        }
                    }
                }
                DecisionNodeType::LlmCondition { branches, default_branch, .. } => {
                    for target in branches.values() {
                        referenced_ids.insert(target.clone());
                        if !self.nodes.contains_key(target) {
                            issues.push(format!(
                                "Node '{}' references non-existent branch target '{}'",
                                node.id, target
                            ));
                        }
                    }
                    if let Some(default) = default_branch {
                        referenced_ids.insert(default.clone());
                        if !self.nodes.contains_key(default) {
                            issues.push(format!(
                                "Node '{}' has non-existent default branch '{}'",
                                node.id, default
                            ));
                        }
                    }
                }
                DecisionNodeType::Terminal { .. } => {}
            }
        }

        // Find reachable nodes via BFS from root
        let mut reachable: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut queue: std::collections::VecDeque<String> =
            std::collections::VecDeque::new();

        if self.nodes.contains_key(&self.root_node_id) {
            queue.push_back(self.root_node_id.clone());
        }

        while let Some(node_id) = queue.pop_front() {
            if reachable.contains(&node_id) {
                continue;
            }
            reachable.insert(node_id.clone());

            if let Some(node) = self.nodes.get(&node_id) {
                match &node.node_type {
                    DecisionNodeType::Condition {
                        branches,
                        default_branch,
                    } => {
                        for branch in branches {
                            if !reachable.contains(&branch.target_node_id) {
                                queue.push_back(branch.target_node_id.clone());
                            }
                        }
                        if let Some(default) = default_branch {
                            if !reachable.contains(default) {
                                queue.push_back(default.clone());
                            }
                        }
                    }
                    DecisionNodeType::Action { next_node, .. } => {
                        if let Some(next) = next_node {
                            if !reachable.contains(next) {
                                queue.push_back(next.clone());
                            }
                        }
                    }
                    DecisionNodeType::Question { next_node, .. }
                    | DecisionNodeType::Prompt { next_node, .. }
                    | DecisionNodeType::Function { next_node, .. }
                    | DecisionNodeType::SubTree { next_node, .. } => {
                        if let Some(next) = next_node {
                            if !reachable.contains(next) {
                                queue.push_back(next.clone());
                            }
                        }
                    }
                    DecisionNodeType::Sequence { children }
                    | DecisionNodeType::Selector { children }
                    | DecisionNodeType::Parallel { children, .. } => {
                        for child_id in children {
                            if !reachable.contains(child_id) {
                                queue.push_back(child_id.clone());
                            }
                        }
                    }
                    DecisionNodeType::LlmCondition { branches, default_branch, .. } => {
                        for target in branches.values() {
                            if !reachable.contains(target) {
                                queue.push_back(target.clone());
                            }
                        }
                        if let Some(default) = default_branch {
                            if !reachable.contains(default) {
                                queue.push_back(default.clone());
                            }
                        }
                    }
                    DecisionNodeType::Terminal { .. } => {}
                }
            }
        }

        // Check for unreachable nodes
        for node_id in self.nodes.keys() {
            if !reachable.contains(node_id) {
                issues.push(format!("Node '{}' is unreachable from root", node_id));
            }
        }

        issues
    }

    /// Return the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of terminal nodes.
    pub fn terminal_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| matches!(n.node_type, DecisionNodeType::Terminal { .. }))
            .count()
    }

    /// Serialize the tree to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize a tree from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let tree: DecisionTree = serde_json::from_str(json)?;
        Ok(tree)
    }

    /// Export the tree as a Mermaid flowchart string.
    pub fn to_mermaid(&self) -> String {
        let mut lines = Vec::new();
        lines.push("flowchart TD".to_string());

        // Sort node IDs for deterministic output
        let mut node_ids: Vec<&String> = self.nodes.keys().collect();
        node_ids.sort();

        for node_id in &node_ids {
            let node = &self.nodes[*node_id];
            let label = node
                .description
                .as_deref()
                .unwrap_or_else(|| match &node.node_type {
                    DecisionNodeType::Condition { .. } => "Condition",
                    DecisionNodeType::Action { action_type, .. } => action_type.as_str(),
                    DecisionNodeType::Terminal { .. } => "Terminal",
                    DecisionNodeType::Question { prompt, .. } => prompt.as_str(),
                    DecisionNodeType::Prompt { user_prompt, .. } => user_prompt.as_str(),
                    DecisionNodeType::Function { function_name, .. } => function_name.as_str(),
                    DecisionNodeType::Sequence { .. } => "Sequence",
                    DecisionNodeType::Selector { .. } => "Selector",
                    DecisionNodeType::Parallel { .. } => "Parallel",
                    DecisionNodeType::SubTree { tree_id, .. } => tree_id.as_str(),
                    DecisionNodeType::LlmCondition { .. } => "LLM Condition",
                });

            // Node shape depends on type
            let node_def = match &node.node_type {
                DecisionNodeType::Condition { .. }
                | DecisionNodeType::LlmCondition { .. } => {
                    format!("    {}{{{{{}}}}}", node_id, escape_mermaid(label))
                }
                DecisionNodeType::Terminal { .. } => {
                    format!("    {}([{}])", node_id, escape_mermaid(label))
                }
                DecisionNodeType::Question { .. }
                | DecisionNodeType::Prompt { .. } => {
                    format!("    {}[/{}\\]", node_id, escape_mermaid(label))
                }
                DecisionNodeType::Action { .. }
                | DecisionNodeType::Function { .. } => {
                    format!("    {}[{}]", node_id, escape_mermaid(label))
                }
                DecisionNodeType::Sequence { .. }
                | DecisionNodeType::Selector { .. }
                | DecisionNodeType::Parallel { .. } => {
                    format!("    {}[[{}]]", node_id, escape_mermaid(label))
                }
                DecisionNodeType::SubTree { .. } => {
                    format!("    {}[[\"{}\n(subtree)\"]]", node_id, escape_mermaid(label))
                }
            };
            lines.push(node_def);
        }

        // Edges
        for node_id in &node_ids {
            let node = &self.nodes[*node_id];
            match &node.node_type {
                DecisionNodeType::Condition {
                    branches,
                    default_branch,
                } => {
                    for branch in branches {
                        let edge_label = branch
                            .label
                            .as_deref()
                            .or(branch.condition.description.as_deref())
                            .unwrap_or("match");
                        lines.push(format!(
                            "    {} -->|{}| {}",
                            node_id,
                            escape_mermaid(edge_label),
                            branch.target_node_id
                        ));
                    }
                    if let Some(default) = default_branch {
                        lines.push(format!(
                            "    {} -->|default| {}",
                            node_id, default
                        ));
                    }
                }
                DecisionNodeType::Action { next_node, .. }
                | DecisionNodeType::Question { next_node, .. }
                | DecisionNodeType::Prompt { next_node, .. }
                | DecisionNodeType::Function { next_node, .. }
                | DecisionNodeType::SubTree { next_node, .. } => {
                    if let Some(next) = next_node {
                        lines.push(format!("    {} --> {}", node_id, next));
                    }
                }
                DecisionNodeType::Sequence { children }
                | DecisionNodeType::Selector { children }
                | DecisionNodeType::Parallel { children, .. } => {
                    for (i, child_id) in children.iter().enumerate() {
                        lines.push(format!("    {} -->|{}| {}", node_id, i + 1, child_id));
                    }
                }
                DecisionNodeType::LlmCondition { branches, default_branch, .. } => {
                    for (label, target) in branches {
                        lines.push(format!(
                            "    {} -->|{}| {}",
                            node_id,
                            escape_mermaid(label),
                            target
                        ));
                    }
                    if let Some(default) = default_branch {
                        lines.push(format!("    {} -->|default| {}", node_id, default));
                    }
                }
                DecisionNodeType::Terminal { .. } => {}
            }
        }

        lines.join("\n")
    }
}

/// Substitute `{{variable}}` placeholders in a template string with values from context.
///
/// Unknown variables are left as-is in the output.
pub fn substitute_template(template: &str, context: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();
    for (key, value) in context {
        let placeholder = format!("{{{{{}}}}}", key);
        if result.contains(&placeholder) {
            let replacement = value_to_string(value);
            result = result.replace(&placeholder, &replacement);
        }
    }
    result
}

/// Substitute template variables in a JSON Value (recursively for strings).
fn substitute_value(value: &Value, context: &HashMap<String, Value>) -> Value {
    match value {
        Value::String(s) => Value::String(substitute_template(s, context)),
        Value::Array(arr) => Value::Array(arr.iter().map(|v| substitute_value(v, context)).collect()),
        Value::Object(obj) => {
            let mut new_obj = serde_json::Map::new();
            for (k, v) in obj {
                new_obj.insert(k.clone(), substitute_value(v, context));
            }
            Value::Object(new_obj)
        }
        other => other.clone(),
    }
}

/// Escape special Mermaid characters in labels.
fn escape_mermaid(s: &str) -> String {
    s.replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree Executor (Async Execution)
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted during async tree execution.
#[derive(Debug, Clone)]
pub enum TreeEvent {
    /// A node was entered during traversal.
    NodeEntered { node_id: String },
    /// An action node was executed.
    ActionExecuted { node_id: String, action_type: String, result: Value },
    /// A prompt node needs LLM response.
    PromptNeeded { node_id: String, system_prompt: Option<String>, user_prompt: String, output_variable: String },
    /// An LLM condition node needs LLM response to decide branching.
    LlmConditionNeeded { node_id: String, prompt: String, options: Vec<String> },
    /// A function was executed.
    FunctionExecuted { node_id: String, function_name: String, result: Result<Value, String> },
    /// A subtree execution has started.
    SubTreeStarted { node_id: String, tree_id: String },
    /// Tree execution completed with a result.
    TreeCompleted { result: Value, path: DecisionPath },
    /// Tree execution encountered an error.
    TreeError { node_id: String, error: String },
}

/// State of the tree executor.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorState {
    /// Ready to start or continue execution.
    Ready,
    /// Currently executing nodes.
    Running,
    /// Waiting for an LLM response (prompt or condition).
    WaitingForLlm { node_id: String },
    /// Execution completed.
    Completed,
    /// Execution encountered an error.
    Error(String),
}

/// Function handler type for the executor.
pub type FunctionHandler = Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

/// Async tree executor that emits events via a channel.
///
/// This executor traverses the decision tree and sends `TreeEvent`s through
/// a channel. When it encounters Prompt or LlmCondition nodes, it pauses
/// and waits for the caller to provide a response via `resume_with_response()`.
pub struct TreeExecutor {
    tree: DecisionTree,
    sub_trees: HashMap<String, DecisionTree>,
    functions: HashMap<String, FunctionHandler>,
    context: HashMap<String, Value>,
    current_node: Option<String>,
    event_sender: Sender<TreeEvent>,
    state: ExecutorState,
    path: DecisionPath,
    visited: std::collections::HashSet<String>,
}

impl TreeExecutor {
    /// Create a new executor for the given tree with an event sender.
    pub fn new(tree: DecisionTree, sender: Sender<TreeEvent>) -> Self {
        let root_id = tree.root_node_id.clone();
        Self {
            tree,
            sub_trees: HashMap::new(),
            functions: HashMap::new(),
            context: HashMap::new(),
            current_node: Some(root_id),
            event_sender: sender,
            state: ExecutorState::Ready,
            path: DecisionPath {
                nodes_visited: Vec::new(),
                actions_taken: Vec::new(),
                result: None,
                complete: false,
            },
            visited: std::collections::HashSet::new(),
        }
    }

    /// Set initial context variables.
    pub fn with_context(mut self, context: HashMap<String, Value>) -> Self {
        self.context = context;
        self
    }

    /// Register a subtree that can be referenced by SubTree nodes.
    pub fn register_subtree(&mut self, tree: DecisionTree) {
        self.sub_trees.insert(tree.id.clone(), tree);
    }

    /// Register a function handler.
    pub fn register_function<F>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        self.functions.insert(name.into(), Box::new(handler));
    }

    /// Get the current executor state.
    pub fn state(&self) -> &ExecutorState {
        &self.state
    }

    /// Get the current context.
    pub fn context(&self) -> &HashMap<String, Value> {
        &self.context
    }

    /// Run the executor until it completes or needs external input.
    ///
    /// Returns the state after execution pauses or completes.
    pub fn run(&mut self) -> &ExecutorState {
        self.state = ExecutorState::Running;

        loop {
            let current_id = match &self.current_node {
                Some(id) => id.clone(),
                None => {
                    self.state = ExecutorState::Error("No current node".to_string());
                    break;
                }
            };

            // Cycle detection
            if self.visited.contains(&current_id) {
                self.state = ExecutorState::Error(format!("Cycle detected at node '{}'", current_id));
                let _ = self.event_sender.send(TreeEvent::TreeError {
                    node_id: current_id,
                    error: "Cycle detected".to_string(),
                });
                break;
            }
            self.visited.insert(current_id.clone());

            let node = match self.tree.nodes.get(&current_id) {
                Some(n) => n.clone(),
                None => {
                    self.state = ExecutorState::Error(format!("Node '{}' not found", current_id));
                    break;
                }
            };

            self.path.nodes_visited.push(current_id.clone());
            let _ = self.event_sender.send(TreeEvent::NodeEntered { node_id: current_id.clone() });

            match &node.node_type {
                DecisionNodeType::Condition { branches, default_branch } => {
                    let mut next = None;
                    for branch in branches {
                        if branch.condition.evaluate(&self.context) {
                            next = Some(branch.target_node_id.clone());
                            break;
                        }
                    }
                    if next.is_none() {
                        next = default_branch.clone();
                    }
                    match next {
                        Some(next_id) => self.current_node = Some(next_id),
                        None => {
                            self.state = ExecutorState::Error("No matching branch".to_string());
                            break;
                        }
                    }
                }

                DecisionNodeType::Action { action_type, parameters, next_node } => {
                    let resolved_params: HashMap<String, Value> = parameters
                        .iter()
                        .map(|(k, v)| (k.clone(), substitute_value(v, &self.context)))
                        .collect();
                    self.path.actions_taken.push((action_type.clone(), resolved_params.clone()));
                    let _ = self.event_sender.send(TreeEvent::ActionExecuted {
                        node_id: current_id,
                        action_type: action_type.clone(),
                        result: Value::Object(resolved_params.into_iter().collect()),
                    });
                    self.current_node = next_node.clone();
                    if next_node.is_none() {
                        self.state = ExecutorState::Completed;
                        break;
                    }
                }

                DecisionNodeType::Terminal { result, .. } => {
                    self.path.result = Some(result.clone());
                    self.path.complete = true;
                    self.state = ExecutorState::Completed;
                    let _ = self.event_sender.send(TreeEvent::TreeCompleted {
                        result: result.clone(),
                        path: self.path.clone(),
                    });
                    break;
                }

                DecisionNodeType::Question { .. } => {
                    // Question nodes pause execution like prompts
                    self.state = ExecutorState::WaitingForLlm { node_id: current_id };
                    break;
                }

                DecisionNodeType::Prompt { system_prompt, user_prompt, output_variable, next_node: _ } => {
                    let resolved_prompt = substitute_template(user_prompt, &self.context);
                    let resolved_system = system_prompt.as_ref()
                        .map(|s| substitute_template(s, &self.context));
                    let _ = self.event_sender.send(TreeEvent::PromptNeeded {
                        node_id: current_id.clone(),
                        system_prompt: resolved_system,
                        user_prompt: resolved_prompt,
                        output_variable: output_variable.clone(),
                    });
                    self.state = ExecutorState::WaitingForLlm { node_id: current_id };
                    break;
                }

                DecisionNodeType::LlmCondition { prompt, branches, default_branch: _ } => {
                    let resolved_prompt = substitute_template(prompt, &self.context);
                    let options: Vec<String> = branches.keys().cloned().collect();
                    let _ = self.event_sender.send(TreeEvent::LlmConditionNeeded {
                        node_id: current_id.clone(),
                        prompt: resolved_prompt,
                        options,
                    });
                    self.state = ExecutorState::WaitingForLlm { node_id: current_id };
                    break;
                }

                DecisionNodeType::Function { function_name, arguments, output_variable, next_node } => {
                    let resolved_args = Value::Object(
                        arguments.iter()
                            .map(|(k, v)| (k.clone(), substitute_value(v, &self.context)))
                            .collect()
                    );

                    let result = if let Some(handler) = self.functions.get(function_name) {
                        handler(resolved_args.clone())
                    } else {
                        Err(format!("Function '{}' not registered", function_name))
                    };

                    let _ = self.event_sender.send(TreeEvent::FunctionExecuted {
                        node_id: current_id.clone(),
                        function_name: function_name.clone(),
                        result: result.clone(),
                    });

                    match result {
                        Ok(value) => {
                            if let Some(var) = output_variable {
                                self.context.insert(var.clone(), value);
                            }
                            self.current_node = next_node.clone();
                            if next_node.is_none() {
                                self.state = ExecutorState::Completed;
                                break;
                            }
                        }
                        Err(e) => {
                            self.state = ExecutorState::Error(e);
                            break;
                        }
                    }
                }

                DecisionNodeType::SubTree { tree_id, input_mapping, output_variable, next_node } => {
                    let _ = self.event_sender.send(TreeEvent::SubTreeStarted {
                        node_id: current_id.clone(),
                        tree_id: tree_id.clone(),
                    });

                    if let Some(sub_tree) = self.sub_trees.get(tree_id) {
                        // Build child context from input mapping
                        let mut child_context = HashMap::new();
                        for (child_var, parent_var) in input_mapping {
                            if let Some(val) = self.context.get(parent_var) {
                                child_context.insert(child_var.clone(), val.clone());
                            }
                        }

                        // Evaluate subtree synchronously
                        let sub_path = sub_tree.evaluate(&child_context);
                        self.path.nodes_visited.extend(sub_path.nodes_visited);
                        self.path.actions_taken.extend(sub_path.actions_taken);

                        if let Some(result) = sub_path.result {
                            if let Some(var) = output_variable {
                                self.context.insert(var.clone(), result);
                            }
                        }

                        self.current_node = next_node.clone();
                        if next_node.is_none() {
                            self.state = ExecutorState::Completed;
                            break;
                        }
                    } else {
                        self.state = ExecutorState::Error(format!("SubTree '{}' not found", tree_id));
                        break;
                    }
                }

                DecisionNodeType::Sequence { children } => {
                    let mut last_result = None;
                    let mut all_ok = true;
                    for child_id in children {
                        let child_path = self.tree.evaluate_subtree(child_id, &self.context);
                        self.path.nodes_visited.extend(child_path.nodes_visited);
                        self.path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            last_result = child_path.result;
                        } else {
                            all_ok = false;
                            break;
                        }
                    }
                    if all_ok {
                        self.path.result = last_result.clone();
                        self.path.complete = true;
                        self.state = ExecutorState::Completed;
                        let _ = self.event_sender.send(TreeEvent::TreeCompleted {
                            result: last_result.unwrap_or(Value::Null),
                            path: self.path.clone(),
                        });
                    } else {
                        self.state = ExecutorState::Error("Sequence child failed".to_string());
                    }
                    break;
                }

                DecisionNodeType::Selector { children } => {
                    let mut found = false;
                    for child_id in children {
                        let child_path = self.tree.evaluate_subtree(child_id, &self.context);
                        self.path.nodes_visited.extend(child_path.nodes_visited);
                        self.path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            self.path.result = child_path.result.clone();
                            self.path.complete = true;
                            self.state = ExecutorState::Completed;
                            let _ = self.event_sender.send(TreeEvent::TreeCompleted {
                                result: child_path.result.unwrap_or(Value::Null),
                                path: self.path.clone(),
                            });
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        self.state = ExecutorState::Error("No selector child succeeded".to_string());
                    }
                    break;
                }

                DecisionNodeType::Parallel { children, require_all } => {
                    let mut results = Vec::new();
                    let mut all_complete = true;
                    for child_id in children {
                        let child_path = self.tree.evaluate_subtree(child_id, &self.context);
                        self.path.nodes_visited.extend(child_path.nodes_visited);
                        self.path.actions_taken.extend(child_path.actions_taken);
                        if child_path.complete {
                            if let Some(r) = child_path.result {
                                results.push(r);
                            }
                        } else {
                            all_complete = false;
                        }
                    }
                    let ok = *require_all && all_complete || !*require_all && !results.is_empty();
                    if ok {
                        let arr = Value::Array(results);
                        self.path.result = Some(arr.clone());
                        self.path.complete = true;
                        self.state = ExecutorState::Completed;
                        let _ = self.event_sender.send(TreeEvent::TreeCompleted {
                            result: arr,
                            path: self.path.clone(),
                        });
                    } else {
                        self.state = ExecutorState::Error("Parallel execution failed".to_string());
                    }
                    break;
                }
            }
        }

        &self.state
    }

    /// Resume execution after providing an LLM response.
    ///
    /// For Prompt nodes: `response` is stored in the output_variable.
    /// For LlmCondition nodes: `response` is matched against branch labels.
    /// For Question nodes: `response` is stored in the variable.
    pub fn resume_with_response(&mut self, response: &str) -> &ExecutorState {
        let node_id = match &self.state {
            ExecutorState::WaitingForLlm { node_id } => node_id.clone(),
            _ => {
                self.state = ExecutorState::Error("Not waiting for LLM response".to_string());
                return &self.state;
            }
        };

        let node = match self.tree.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => {
                self.state = ExecutorState::Error(format!("Node '{}' not found", node_id));
                return &self.state;
            }
        };

        match &node.node_type {
            DecisionNodeType::Prompt { output_variable, next_node, .. } => {
                self.context.insert(output_variable.clone(), Value::String(response.to_string()));
                self.current_node = next_node.clone();
            }
            DecisionNodeType::LlmCondition { branches, default_branch, .. } => {
                let response_lower = response.to_lowercase().trim().to_string();
                let mut target = None;
                for (label, node_id) in branches {
                    if response_lower.contains(&label.to_lowercase()) {
                        target = Some(node_id.clone());
                        break;
                    }
                }
                if target.is_none() {
                    target = default_branch.clone();
                }
                self.current_node = target;
            }
            DecisionNodeType::Question { variable, next_node, .. } => {
                self.context.insert(variable.clone(), Value::String(response.to_string()));
                self.current_node = next_node.clone();
            }
            _ => {
                self.state = ExecutorState::Error("Node is not awaiting response".to_string());
                return &self.state;
            }
        }

        self.state = ExecutorState::Ready;
        self.run()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision Tree Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for constructing decision trees with a fluent API.
#[derive(Debug, Clone)]
pub struct DecisionTreeBuilder {
    id: String,
    name: String,
    description: Option<String>,
    root_node_id: Option<String>,
    nodes: Vec<DecisionNode>,
}

impl DecisionTreeBuilder {
    /// Create a new builder with the given tree ID and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            root_node_id: None,
            nodes: Vec::new(),
        }
    }

    /// Set the root node ID.
    pub fn root(mut self, node_id: &str) -> Self {
        self.root_node_id = Some(node_id.to_string());
        self
    }

    /// Set the tree description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a node to the tree.
    pub fn node(mut self, node: DecisionNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add a condition node.
    pub fn condition_node(
        self,
        id: impl Into<String>,
        branches: Vec<DecisionBranch>,
        default: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_condition(id, branches, default))
    }

    /// Add an action node.
    pub fn action_node(
        self,
        id: impl Into<String>,
        action_type: impl Into<String>,
        parameters: HashMap<String, Value>,
        next: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_action(id, action_type, parameters, next))
    }

    /// Add a terminal node.
    pub fn terminal_node(
        self,
        id: impl Into<String>,
        result: Value,
        label: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_terminal(id, result, label))
    }

    /// Add a prompt node.
    pub fn prompt_node(
        self,
        id: impl Into<String>,
        user_prompt: impl Into<String>,
        output_variable: impl Into<String>,
        next: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_prompt(id, user_prompt, output_variable, next))
    }

    /// Add a function call node.
    pub fn function_node(
        self,
        id: impl Into<String>,
        function_name: impl Into<String>,
        arguments: HashMap<String, Value>,
        output_variable: Option<String>,
        next: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_function(id, function_name, arguments, output_variable, next))
    }

    /// Add a sequence node.
    pub fn sequence_node(self, id: impl Into<String>, children: Vec<String>) -> Self {
        self.node(DecisionNode::new_sequence(id, children))
    }

    /// Add a selector node.
    pub fn selector_node(self, id: impl Into<String>, children: Vec<String>) -> Self {
        self.node(DecisionNode::new_selector(id, children))
    }

    /// Add a parallel node.
    pub fn parallel_node(self, id: impl Into<String>, children: Vec<String>, require_all: bool) -> Self {
        self.node(DecisionNode::new_parallel(id, children, require_all))
    }

    /// Add a subtree reference node.
    pub fn subtree_node(
        self,
        id: impl Into<String>,
        tree_id: impl Into<String>,
        input_mapping: HashMap<String, String>,
        output_variable: Option<String>,
        next: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_subtree(id, tree_id, input_mapping, output_variable, next))
    }

    /// Add an LLM condition node.
    pub fn llm_condition_node(
        self,
        id: impl Into<String>,
        prompt: impl Into<String>,
        branches: HashMap<String, String>,
        default_branch: Option<String>,
    ) -> Self {
        self.node(DecisionNode::new_llm_condition(id, prompt, branches, default_branch))
    }

    /// Build the decision tree.
    ///
    /// Uses the first node's ID as root if no root was explicitly set.
    pub fn build(self) -> DecisionTree {
        let root_id = self
            .root_node_id
            .unwrap_or_else(|| self.nodes.first().map(|n| n.id.clone()).unwrap_or_default());

        let mut tree = DecisionTree::new(self.id, self.name, root_id);
        tree.description = self.description;
        for node in self.nodes {
            tree.add_node(node);
        }
        tree
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper to build a simple age-based decision tree.
    fn build_age_tree() -> DecisionTree {
        let branches = vec![
            DecisionBranch {
                condition: Condition::new(
                    "age",
                    ConditionOperator::GreaterOrEqual,
                    json!(18),
                ),
                target_node_id: "adult".to_string(),
                label: Some(">=18".to_string()),
            },
        ];

        let root = DecisionNode::new_condition("check_age", branches, Some("minor".to_string()));
        let adult = DecisionNode::new_terminal("adult", json!("allowed"), Some("Adult".to_string()));
        let minor = DecisionNode::new_terminal("minor", json!("denied"), Some("Minor".to_string()));

        DecisionTreeBuilder::new("age_tree", "Age Check")
            .root("check_age")
            .description("Checks if user is an adult")
            .node(root)
            .node(adult)
            .node(minor)
            .build()
    }

    #[test]
    fn test_condition_equals() {
        let cond = Condition::new("status", ConditionOperator::Equals, json!("active"));
        let mut ctx = HashMap::new();
        ctx.insert("status".to_string(), json!("active"));
        assert!(cond.evaluate(&ctx));

        ctx.insert("status".to_string(), json!("inactive"));
        assert!(!cond.evaluate(&ctx));
    }

    #[test]
    fn test_condition_numeric_comparisons() {
        let gt = Condition::new("score", ConditionOperator::GreaterThan, json!(50));
        let lt = Condition::new("score", ConditionOperator::LessThan, json!(50));
        let gte = Condition::new("score", ConditionOperator::GreaterOrEqual, json!(50));
        let lte = Condition::new("score", ConditionOperator::LessOrEqual, json!(50));

        let mut ctx = HashMap::new();
        ctx.insert("score".to_string(), json!(75));
        assert!(gt.evaluate(&ctx));
        assert!(!lt.evaluate(&ctx));
        assert!(gte.evaluate(&ctx));
        assert!(!lte.evaluate(&ctx));

        ctx.insert("score".to_string(), json!(50));
        assert!(!gt.evaluate(&ctx));
        assert!(!lt.evaluate(&ctx));
        assert!(gte.evaluate(&ctx));
        assert!(lte.evaluate(&ctx));

        ctx.insert("score".to_string(), json!(25));
        assert!(!gt.evaluate(&ctx));
        assert!(lt.evaluate(&ctx));
        assert!(!gte.evaluate(&ctx));
        assert!(lte.evaluate(&ctx));
    }

    #[test]
    fn test_condition_string_operations() {
        let contains = Condition::new("name", ConditionOperator::Contains, json!("world"));
        let starts = Condition::new("name", ConditionOperator::StartsWith, json!("hello"));
        let ends = Condition::new("name", ConditionOperator::EndsWith, json!("world"));
        let not_contains = Condition::new("name", ConditionOperator::NotContains, json!("xyz"));

        let mut ctx = HashMap::new();
        ctx.insert("name".to_string(), json!("hello world"));

        assert!(contains.evaluate(&ctx));
        assert!(starts.evaluate(&ctx));
        assert!(ends.evaluate(&ctx));
        assert!(not_contains.evaluate(&ctx));
    }

    #[test]
    fn test_condition_regex_matches() {
        let cond = Condition::new("email", ConditionOperator::Matches, json!(r"^[\w.]+@[\w.]+$"));
        let mut ctx = HashMap::new();

        ctx.insert("email".to_string(), json!("user@example.com"));
        assert!(cond.evaluate(&ctx));

        ctx.insert("email".to_string(), json!("invalid-email"));
        assert!(!cond.evaluate(&ctx));
    }

    #[test]
    fn test_condition_is_empty_and_in_list() {
        let empty_cond = Condition::new("field", ConditionOperator::IsEmpty, json!(null));
        let not_empty_cond = Condition::new("field", ConditionOperator::IsNotEmpty, json!(null));
        let in_list = Condition::new(
            "role",
            ConditionOperator::InList,
            json!(["admin", "moderator"]),
        );

        let mut ctx = HashMap::new();
        ctx.insert("field".to_string(), json!(""));
        assert!(empty_cond.evaluate(&ctx));
        assert!(!not_empty_cond.evaluate(&ctx));

        ctx.insert("field".to_string(), json!("data"));
        assert!(!empty_cond.evaluate(&ctx));
        assert!(not_empty_cond.evaluate(&ctx));

        ctx.insert("role".to_string(), json!("admin"));
        assert!(in_list.evaluate(&ctx));

        ctx.insert("role".to_string(), json!("user"));
        assert!(!in_list.evaluate(&ctx));
    }

    #[test]
    fn test_tree_traversal_adult() {
        let tree = build_age_tree();
        let mut ctx = HashMap::new();
        ctx.insert("age".to_string(), json!(21));

        let path = tree.evaluate(&ctx);
        assert!(path.complete);
        assert_eq!(path.result, Some(json!("allowed")));
        assert_eq!(path.nodes_visited, vec!["check_age", "adult"]);
    }

    #[test]
    fn test_tree_traversal_minor() {
        let tree = build_age_tree();
        let mut ctx = HashMap::new();
        ctx.insert("age".to_string(), json!(15));

        let path = tree.evaluate(&ctx);
        assert!(path.complete);
        assert_eq!(path.result, Some(json!("denied")));
        assert_eq!(path.nodes_visited, vec!["check_age", "minor"]);
    }

    #[test]
    fn test_tree_with_action_node() {
        let action_node = DecisionNode::new_action(
            "log_action",
            "log",
            {
                let mut p = HashMap::new();
                p.insert("message".to_string(), json!("User checked in"));
                p
            },
            Some("result".to_string()),
        );
        let terminal = DecisionNode::new_terminal("result", json!("done"), None);

        let tree = DecisionTreeBuilder::new("action_tree", "Action Test")
            .root("log_action")
            .node(action_node)
            .node(terminal)
            .build();

        let ctx = HashMap::new();
        let path = tree.evaluate(&ctx);

        assert!(path.complete);
        assert_eq!(path.actions_taken.len(), 1);
        assert_eq!(path.actions_taken[0].0, "log");
        assert_eq!(path.result, Some(json!("done")));
    }

    #[test]
    fn test_tree_question_stops_traversal() {
        let question = DecisionNode::new_question(
            "ask_name",
            "What is your name?",
            "user_name",
            vec!["Alice".to_string(), "Bob".to_string()],
            Some("greet".to_string()),
        );
        let terminal = DecisionNode::new_terminal("greet", json!("Hello!"), None);

        let tree = DecisionTreeBuilder::new("q_tree", "Question Test")
            .root("ask_name")
            .node(question)
            .node(terminal)
            .build();

        let ctx = HashMap::new();
        let path = tree.evaluate(&ctx);

        assert!(!path.complete);
        assert_eq!(path.nodes_visited, vec!["ask_name"]);
        assert!(path.result.is_none());
    }

    #[test]
    fn test_tree_serialization_roundtrip() {
        let tree = build_age_tree();
        let json_str = tree.to_json();

        let restored = DecisionTree::from_json(&json_str).expect("Failed to deserialize");
        assert_eq!(restored.id, tree.id);
        assert_eq!(restored.name, tree.name);
        assert_eq!(restored.root_node_id, tree.root_node_id);
        assert_eq!(restored.node_count(), tree.node_count());
        assert_eq!(restored.terminal_count(), tree.terminal_count());

        // Verify the restored tree works the same
        let mut ctx = HashMap::new();
        ctx.insert("age".to_string(), json!(21));
        let path = restored.evaluate(&ctx);
        assert!(path.complete);
        assert_eq!(path.result, Some(json!("allowed")));
    }

    #[test]
    fn test_tree_validation() {
        let mut tree = build_age_tree();

        // Valid tree should have no issues
        let issues = tree.validate();
        assert!(issues.is_empty(), "Expected no issues, got: {:?}", issues);

        // Add a node with a dangling reference
        let bad_node = DecisionNode::new_action(
            "bad",
            "noop",
            HashMap::new(),
            Some("nonexistent".to_string()),
        );
        tree.add_node(bad_node);

        let issues = tree.validate();
        assert!(issues.iter().any(|i| i.contains("nonexistent")));
        assert!(issues.iter().any(|i| i.contains("unreachable")));
    }

    #[test]
    fn test_evaluate_step() {
        let tree = build_age_tree();
        let mut ctx = HashMap::new();
        ctx.insert("age".to_string(), json!(21));

        let next = tree.evaluate_step("check_age", &ctx);
        assert_eq!(next, Some("adult".to_string()));

        ctx.insert("age".to_string(), json!(10));
        let next = tree.evaluate_step("check_age", &ctx);
        assert_eq!(next, Some("minor".to_string()));

        // Terminal node has no next
        let next = tree.evaluate_step("adult", &ctx);
        assert_eq!(next, None);
    }

    #[test]
    fn test_mermaid_export() {
        let tree = build_age_tree();
        let mermaid = tree.to_mermaid();

        assert!(mermaid.starts_with("flowchart TD"));
        assert!(mermaid.contains("check_age"));
        assert!(mermaid.contains("adult"));
        assert!(mermaid.contains("minor"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_cycle_detection() {
        // Build a tree that has a cycle: A -> B -> A
        let branch_a = vec![DecisionBranch {
            condition: Condition::new("x", ConditionOperator::Equals, json!(true)),
            target_node_id: "node_b".to_string(),
            label: None,
        }];
        let branch_b = vec![DecisionBranch {
            condition: Condition::new("x", ConditionOperator::Equals, json!(true)),
            target_node_id: "node_a".to_string(),
            label: None,
        }];

        let node_a = DecisionNode::new_condition("node_a", branch_a, None);
        let node_b = DecisionNode::new_condition("node_b", branch_b, None);

        let tree = DecisionTreeBuilder::new("cycle_tree", "Cycle Test")
            .root("node_a")
            .node(node_a)
            .node(node_b)
            .build();

        let mut ctx = HashMap::new();
        ctx.insert("x".to_string(), json!(true));

        let path = tree.evaluate(&ctx);
        // Should not loop forever; cycle detection breaks out
        assert!(!path.complete);
        assert_eq!(path.nodes_visited.len(), 2);
    }

    #[test]
    fn test_condition_with_missing_variable() {
        let cond = Condition::new("missing", ConditionOperator::IsEmpty, json!(null));
        let ctx = HashMap::new();
        // Missing variable treated as Null, which is empty
        assert!(cond.evaluate(&ctx));

        let cond2 = Condition::new("missing", ConditionOperator::Equals, json!(null));
        assert!(cond2.evaluate(&ctx));
    }

    #[test]
    fn test_builder_default_root() {
        let node = DecisionNode::new_terminal("only_node", json!("result"), None);
        let tree = DecisionTreeBuilder::new("t", "T")
            .node(node)
            .build();

        // Without explicit root, uses first node
        assert_eq!(tree.root_node_id, "only_node");
    }

    #[test]
    fn test_node_count_and_terminal_count() {
        let tree = build_age_tree();
        assert_eq!(tree.node_count(), 3);
        assert_eq!(tree.terminal_count(), 2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New node type tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_template_substitution() {
        let mut ctx = HashMap::new();
        ctx.insert("name".to_string(), json!("Alice"));
        ctx.insert("age".to_string(), json!(30));

        let result = substitute_template("Hello {{name}}, you are {{age}} years old!", &ctx);
        assert_eq!(result, "Hello Alice, you are 30 years old!");

        // Unknown variables left as-is
        let result2 = substitute_template("Hello {{unknown}}", &ctx);
        assert_eq!(result2, "Hello {{unknown}}");
    }

    #[test]
    fn test_function_node_in_tree() {
        let func_node = DecisionNode::new_function(
            "call_fn",
            "greet",
            {
                let mut args = HashMap::new();
                args.insert("name".to_string(), json!("{{user_name}}"));
                args
            },
            Some("greeting".to_string()),
            Some("done".to_string()),
        );
        let terminal = DecisionNode::new_terminal("done", json!("completed"), None);

        let tree = DecisionTreeBuilder::new("fn_tree", "Function Test")
            .root("call_fn")
            .node(func_node)
            .node(terminal)
            .build();

        let mut ctx = HashMap::new();
        ctx.insert("user_name".to_string(), json!("Bob"));

        let path = tree.evaluate(&ctx);
        assert!(path.complete);
        assert_eq!(path.actions_taken.len(), 1);
        assert_eq!(path.actions_taken[0].0, "greet");
        // Check template substitution happened
        assert_eq!(path.actions_taken[0].1.get("name"), Some(&json!("Bob")));
    }

    #[test]
    fn test_sequence_node() {
        // Sequence of two terminal-reaching subtrees
        let action1 = DecisionNode::new_action(
            "step1",
            "log",
            HashMap::new(),
            Some("result1".to_string()),
        );
        let result1 = DecisionNode::new_terminal("result1", json!("first"), None);
        let action2 = DecisionNode::new_action(
            "step2",
            "log2",
            HashMap::new(),
            Some("result2".to_string()),
        );
        let result2 = DecisionNode::new_terminal("result2", json!("second"), None);

        let seq = DecisionNode::new_sequence(
            "seq",
            vec!["step1".to_string(), "step2".to_string()],
        );

        let tree = DecisionTreeBuilder::new("seq_tree", "Sequence Test")
            .root("seq")
            .node(seq)
            .node(action1)
            .node(result1)
            .node(action2)
            .node(result2)
            .build();

        let ctx = HashMap::new();
        let path = tree.evaluate(&ctx);

        assert!(path.complete);
        // Last child's result
        assert_eq!(path.result, Some(json!("second")));
        assert_eq!(path.actions_taken.len(), 2);
    }

    #[test]
    fn test_selector_node() {
        // First child fails (no terminal), second succeeds
        let fail_node = DecisionNode::new_action(
            "fail_step",
            "noop",
            HashMap::new(),
            None, // No next → incomplete
        );
        let success_action = DecisionNode::new_action(
            "success_step",
            "ok",
            HashMap::new(),
            Some("result".to_string()),
        );
        let result = DecisionNode::new_terminal("result", json!("found"), None);

        let sel = DecisionNode::new_selector(
            "sel",
            vec!["fail_step".to_string(), "success_step".to_string()],
        );

        let tree = DecisionTreeBuilder::new("sel_tree", "Selector Test")
            .root("sel")
            .node(sel)
            .node(fail_node)
            .node(success_action)
            .node(result)
            .build();

        let path = tree.evaluate(&HashMap::new());
        assert!(path.complete);
        assert_eq!(path.result, Some(json!("found")));
    }

    #[test]
    fn test_parallel_node() {
        let t1 = DecisionNode::new_terminal("t1", json!("a"), None);
        let t2 = DecisionNode::new_terminal("t2", json!("b"), None);

        let par = DecisionNode::new_parallel(
            "par",
            vec!["t1".to_string(), "t2".to_string()],
            true,
        );

        let tree = DecisionTreeBuilder::new("par_tree", "Parallel Test")
            .root("par")
            .node(par)
            .node(t1)
            .node(t2)
            .build();

        let path = tree.evaluate(&HashMap::new());
        assert!(path.complete);
        assert_eq!(path.result, Some(json!(["a", "b"])));
    }

    #[test]
    fn test_prompt_node_stops_sync() {
        let prompt = DecisionNode::new_prompt(
            "ask",
            "Tell me about {{topic}}",
            "answer",
            Some("done".to_string()),
        );
        let terminal = DecisionNode::new_terminal("done", json!("ok"), None);

        let tree = DecisionTreeBuilder::new("p_tree", "Prompt Test")
            .root("ask")
            .node(prompt)
            .node(terminal)
            .build();

        let path = tree.evaluate(&HashMap::new());
        // Prompt stops sync traversal
        assert!(!path.complete);
        assert_eq!(path.nodes_visited, vec!["ask"]);
    }

    #[test]
    fn test_llm_condition_stops_sync() {
        let mut branches = HashMap::new();
        branches.insert("yes".to_string(), "positive".to_string());
        branches.insert("no".to_string(), "negative".to_string());

        let llm_cond = DecisionNode::new_llm_condition(
            "decide",
            "Is {{item}} good?",
            branches,
            Some("neutral".to_string()),
        );
        let pos = DecisionNode::new_terminal("positive", json!("good"), None);
        let neg = DecisionNode::new_terminal("negative", json!("bad"), None);
        let neu = DecisionNode::new_terminal("neutral", json!("meh"), None);

        let tree = DecisionTreeBuilder::new("llm_tree", "LLM Condition Test")
            .root("decide")
            .node(llm_cond)
            .node(pos)
            .node(neg)
            .node(neu)
            .build();

        let path = tree.evaluate(&HashMap::new());
        assert!(!path.complete); // LLM condition stops sync
    }

    #[test]
    fn test_executor_basic_flow() {
        use std::sync::mpsc;

        let terminal = DecisionNode::new_terminal("end", json!("done"), None);
        let tree = DecisionTreeBuilder::new("exec_tree", "Executor Test")
            .root("end")
            .node(terminal)
            .build();

        let (tx, rx) = mpsc::channel();
        let mut executor = TreeExecutor::new(tree, tx);
        executor.run();

        assert_eq!(*executor.state(), ExecutorState::Completed);

        // Check events
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.len() >= 2); // NodeEntered + TreeCompleted
    }

    #[test]
    fn test_executor_with_function() {
        use std::sync::mpsc;

        let func_node = DecisionNode::new_function(
            "calc",
            "double",
            {
                let mut args = HashMap::new();
                args.insert("value".to_string(), json!(21));
                args
            },
            Some("result".to_string()),
            Some("end".to_string()),
        );
        let terminal = DecisionNode::new_terminal("end", json!("finished"), None);

        let tree = DecisionTreeBuilder::new("fn_exec", "Function Executor Test")
            .root("calc")
            .node(func_node)
            .node(terminal)
            .build();

        let (tx, rx) = mpsc::channel();
        let mut executor = TreeExecutor::new(tree, tx);
        executor.register_function("double", |args| {
            let val = args.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!(val * 2))
        });
        executor.run();

        assert_eq!(*executor.state(), ExecutorState::Completed);
        assert_eq!(executor.context().get("result"), Some(&json!(42)));

        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, TreeEvent::FunctionExecuted { .. })));
    }

    #[test]
    fn test_executor_prompt_resume() {
        use std::sync::mpsc;

        let prompt = DecisionNode::new_prompt(
            "ask",
            "What is your name?",
            "name",
            Some("greet".to_string()),
        );
        let greet = DecisionNode::new_terminal("greet", json!("hello"), None);

        let tree = DecisionTreeBuilder::new("prompt_exec", "Prompt Executor Test")
            .root("ask")
            .node(prompt)
            .node(greet)
            .build();

        let (tx, _rx) = mpsc::channel();
        let mut executor = TreeExecutor::new(tree, tx);
        executor.run();

        // Should be waiting for LLM
        assert!(matches!(executor.state(), ExecutorState::WaitingForLlm { .. }));

        // Resume with response
        executor.resume_with_response("Alice");
        assert_eq!(*executor.state(), ExecutorState::Completed);
        assert_eq!(executor.context().get("name"), Some(&json!("Alice")));
    }

    #[test]
    fn test_executor_llm_condition_resume() {
        use std::sync::mpsc;

        let mut branches = HashMap::new();
        branches.insert("yes".to_string(), "good".to_string());
        branches.insert("no".to_string(), "bad".to_string());

        let llm_cond = DecisionNode::new_llm_condition(
            "check",
            "Is this acceptable?",
            branches,
            Some("bad".to_string()),
        );
        let good = DecisionNode::new_terminal("good", json!("accepted"), None);
        let bad = DecisionNode::new_terminal("bad", json!("rejected"), None);

        let tree = DecisionTreeBuilder::new("llm_exec", "LLM Condition Executor Test")
            .root("check")
            .node(llm_cond)
            .node(good)
            .node(bad)
            .build();

        let (tx, _rx) = mpsc::channel();
        let mut executor = TreeExecutor::new(tree, tx);
        executor.run();

        assert!(matches!(executor.state(), ExecutorState::WaitingForLlm { .. }));

        // Resume with "yes" → should go to "good" terminal
        executor.resume_with_response("yes");
        assert_eq!(*executor.state(), ExecutorState::Completed);
    }

    #[test]
    fn test_executor_subtree() {
        use std::sync::mpsc;

        // Build a simple subtree
        let sub_terminal = DecisionNode::new_terminal("sub_end", json!("sub_result"), None);
        let sub_tree = DecisionTreeBuilder::new("sub", "SubTree")
            .root("sub_end")
            .node(sub_terminal)
            .build();

        // Main tree references the subtree
        let subtree_node = DecisionNode::new_subtree(
            "call_sub",
            "sub",
            HashMap::new(),
            Some("sub_output".to_string()),
            Some("main_end".to_string()),
        );
        let main_end = DecisionNode::new_terminal("main_end", json!("main_done"), None);

        let main_tree = DecisionTreeBuilder::new("main", "Main Tree")
            .root("call_sub")
            .node(subtree_node)
            .node(main_end)
            .build();

        let (tx, _rx) = mpsc::channel();
        let mut executor = TreeExecutor::new(main_tree, tx);
        executor.register_subtree(sub_tree);
        executor.run();

        assert_eq!(*executor.state(), ExecutorState::Completed);
        assert_eq!(executor.context().get("sub_output"), Some(&json!("sub_result")));
    }

    #[test]
    fn test_builder_convenience_methods() {
        let tree = DecisionTreeBuilder::new("conv_tree", "Convenience Test")
            .root("start")
            .terminal_node("start", json!("ok"), Some("Start".to_string()))
            .build();

        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.terminal_count(), 1);

        let path = tree.evaluate(&HashMap::new());
        assert!(path.complete);
        assert_eq!(path.result, Some(json!("ok")));
    }

    #[test]
    fn test_validate_new_node_types() {
        let tree = DecisionTreeBuilder::new("val_tree", "Validation Test")
            .root("seq")
            .sequence_node("seq", vec!["t1".to_string(), "t2".to_string()])
            .terminal_node("t1", json!("a"), None)
            .terminal_node("t2", json!("b"), None)
            .build();

        let issues = tree.validate();
        assert!(issues.is_empty(), "Expected no issues, got: {:?}", issues);
    }

    #[test]
    fn test_validate_detects_missing_children() {
        let tree = DecisionTreeBuilder::new("val_tree2", "Missing Child Test")
            .root("seq")
            .sequence_node("seq", vec!["missing".to_string()])
            .build();

        let issues = tree.validate();
        assert!(issues.iter().any(|i| i.contains("missing")));
    }

    #[test]
    fn test_mermaid_new_nodes() {
        let tree = DecisionTreeBuilder::new("mermaid_tree", "Mermaid Test")
            .root("p")
            .prompt_node("p", "Ask something", "out", Some("end".to_string()))
            .terminal_node("end", json!("done"), None)
            .build();

        let mermaid = tree.to_mermaid();
        assert!(mermaid.contains("flowchart TD"));
        assert!(mermaid.contains("p"));
        assert!(mermaid.contains("end"));
    }

    #[test]
    fn test_json_roundtrip_new_types() {
        let tree = DecisionTreeBuilder::new("json_tree", "JSON Test")
            .root("fn")
            .function_node("fn", "test_fn", HashMap::new(), Some("out".to_string()), Some("end".to_string()))
            .terminal_node("end", json!("ok"), None)
            .build();

        let json_str = tree.to_json();
        let restored = DecisionTree::from_json(&json_str).expect("Failed to deserialize");
        assert_eq!(restored.node_count(), tree.node_count());

        // Verify function node preserved
        let fn_node = restored.get_node("fn").unwrap();
        assert!(matches!(fn_node.node_type, DecisionNodeType::Function { .. }));
    }
}
