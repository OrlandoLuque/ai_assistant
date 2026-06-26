//! Hierarchical routing DAG with rule-based and bandit nodes.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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

    pub fn set_bandit(
        &mut self,
        node_id: &str,
        bandit: BanditRouter,
    ) -> Result<(), AdvancedRoutingError> {
        if !self.nodes.contains_key(node_id) {
            return Err(AdvancedRoutingError::NodeNotFound {
                node_id: node_id.to_string(),
            });
        }
        self.bandit_instances.insert(node_id.to_string(), bandit);
        Ok(())
    }

    pub fn set_dfa(&mut self, node_id: &str, dfa: DfaRouter) -> Result<(), AdvancedRoutingError> {
        if !self.nodes.contains_key(node_id) {
            return Err(AdvancedRoutingError::NodeNotFound {
                node_id: node_id.to_string(),
            });
        }
        self.dfa_instances.insert(node_id.to_string(), dfa);
        Ok(())
    }

    /// Route through the DAG from root to a leaf.
    pub fn route(
        &mut self,
        features: &QueryFeatures,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.validate()?;

        let start = std::time::Instant::now();
        let mut current_id = self.root_id.clone();
        let mut path: Vec<String> = Vec::new();

        loop {
            if path.len() > self.nodes.len() {
                return Err(AdvancedRoutingError::CycleDetected);
            }
            path.push(current_id.clone());

            let node = self
                .nodes
                .get(&current_id)
                .ok_or_else(|| AdvancedRoutingError::NodeNotFound {
                    node_id: current_id.clone(),
                })?
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
                RoutingDagNodeType::RuleBased {
                    feature,
                    threshold,
                    high_branch,
                    low_branch,
                } => {
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

        let mut queue: VecDeque<&str> = in_degree
            .iter()
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

    fn find_successor(
        &self,
        node: &RoutingDagNode,
        arm: &str,
    ) -> Result<String, AdvancedRoutingError> {
        // Try to find a successor matching the arm name
        for succ in &node.successors {
            if succ == arm {
                return Ok(succ.clone());
            }
        }
        // If no exact match, use first successor
        node.successors
            .first()
            .cloned()
            .ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
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
        "has_code" => {
            if features.has_code {
                1.0
            } else {
                0.0
            }
        }
        "is_question" => {
            if features.is_question {
                1.0
            } else {
                0.0
            }
        }
        "avg_word_length" => features.avg_word_length,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ROUTING DAG TESTS
    // =========================================================================

    #[test]
    fn test_dag_single_leaf() {
        let mut dag = RoutingDag::new("root");
        dag.add_node(RoutingDagNode {
            id: "root".to_string(),
            label: "Root".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "model-a".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

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
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "powerful".to_string(),
            label: "Powerful".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "gpt-4".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "cheap".to_string(),
            label: "Cheap".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "gpt-3.5".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

        let simple = test_features("general", 0.3);
        assert_eq!(dag.route(&simple).unwrap().selected_arm, "gpt-3.5");

        let complex = test_features("general", 0.8);
        assert_eq!(dag.route(&complex).unwrap().selected_arm, "gpt-4");
    }

    #[test]
    fn test_dag_bandit_node() {
        let mut dag = RoutingDag::new("bandit_root");
        dag.add_node(RoutingDagNode {
            id: "bandit_root".to_string(),
            label: "Bandit".to_string(),
            node_type: RoutingDagNodeType::Bandit(BanditConfig::default()),
            successors: vec!["leaf".to_string()],
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "leaf".to_string(),
            label: "Leaf".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "final-model".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

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
            id: "a".to_string(),
            label: "A".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "x".to_string(),
            },
            successors: vec!["b".to_string()],
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "b".to_string(),
            label: "B".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "y".to_string(),
            },
            successors: vec!["a".to_string()],
        })
        .unwrap();

        assert!(dag.validate().is_err());
    }

    #[test]
    fn test_dag_node_not_found() {
        let mut dag = RoutingDag::new("nonexistent");
        assert!(dag
            .set_bandit("missing", BanditRouter::new(BanditConfig::default()))
            .is_err());
    }

    #[test]
    fn test_dag_topological_order() {
        let mut dag = RoutingDag::new("root");
        dag.add_node(RoutingDagNode {
            id: "root".to_string(),
            label: "R".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "x".to_string(),
            },
            successors: vec!["child".to_string()],
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "child".to_string(),
            label: "C".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "y".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

        let order = dag.topological_order().unwrap();
        assert_eq!(order.len(), 2);
        assert!(
            order.iter().position(|x| x == "root").unwrap()
                < order.iter().position(|x| x == "child").unwrap()
        );
    }

    #[test]
    fn test_dag_validate_acyclic() {
        let mut dag = RoutingDag::new("a");
        dag.add_node(RoutingDagNode {
            id: "a".to_string(),
            label: "A".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "x".to_string(),
            },
            successors: vec!["b".to_string()],
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "b".to_string(),
            label: "B".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "y".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

        assert!(dag.validate().is_ok());
    }

    #[test]
    fn test_dag_record_outcome() {
        let mut dag = RoutingDag::new("b");
        dag.add_node(RoutingDagNode {
            id: "b".to_string(),
            label: "B".to_string(),
            node_type: RoutingDagNodeType::Bandit(BanditConfig::default()),
            successors: Vec::new(),
        })
        .unwrap();
        let mut bandit = BanditRouter::new(BanditConfig::default());
        bandit.add_arm("test");
        dag.set_bandit("b", bandit).unwrap();

        dag.record_outcome(
            "b",
            &ArmFeedback {
                arm_id: "test".to_string(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: None,
            },
        );
        // Bandit should have updated
        assert_eq!(dag.bandit_instances["b"].all_arms(None)[0].pull_count, 1);
    }

    #[test]
    fn test_dag_multi_level() {
        let mut dag = RoutingDag::new("gate");
        dag.add_node(RoutingDagNode {
            id: "gate".to_string(),
            label: "Gate".to_string(),
            node_type: RoutingDagNodeType::RuleBased {
                feature: "has_code".to_string(),
                threshold: 0.5,
                high_branch: "code_leaf".to_string(),
                low_branch: "gen_leaf".to_string(),
            },
            successors: vec!["code_leaf".to_string(), "gen_leaf".to_string()],
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "code_leaf".to_string(),
            label: "Code".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "coder".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();
        dag.add_node(RoutingDagNode {
            id: "gen_leaf".to_string(),
            label: "General".to_string(),
            node_type: RoutingDagNodeType::Leaf {
                arm_id: "general".to_string(),
            },
            successors: Vec::new(),
        })
        .unwrap();

        let code_f = test_features_code();
        assert_eq!(dag.route(&code_f).unwrap().selected_arm, "coder");

        let gen_f = test_features("general", 0.3);
        assert_eq!(dag.route(&gen_f).unwrap().selected_arm, "general");
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
}
