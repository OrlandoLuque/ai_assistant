//! Task decomposition
//!
//! Break down complex tasks into smaller subtasks.

use std::collections::HashMap;

/// Task node in the decomposition tree
#[derive(Debug, Clone)]
pub struct TaskNode {
    pub id: String,
    pub description: String,
    pub subtasks: Vec<TaskNode>,
    pub dependencies: Vec<String>,
    pub estimated_complexity: f64,
    pub required_capabilities: Vec<String>,
    pub status: DecompositionStatus,
}

/// Decomposition status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecompositionStatus {
    NotStarted,
    InProgress,
    Completed,
    Blocked,
    Skipped,
}

impl TaskNode {
    pub fn new(id: &str, description: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            subtasks: Vec::new(),
            dependencies: Vec::new(),
            estimated_complexity: 1.0,
            required_capabilities: Vec::new(),
            status: DecompositionStatus::NotStarted,
        }
    }

    pub fn with_subtask(mut self, subtask: TaskNode) -> Self {
        self.subtasks.push(subtask);
        self
    }

    pub fn with_dependency(mut self, dep_id: &str) -> Self {
        self.dependencies.push(dep_id.to_string());
        self
    }

    pub fn with_complexity(mut self, complexity: f64) -> Self {
        self.estimated_complexity = complexity;
        self
    }

    pub fn with_capability(mut self, capability: &str) -> Self {
        self.required_capabilities.push(capability.to_string());
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.subtasks.is_empty()
    }

    pub fn total_complexity(&self) -> f64 {
        if self.is_leaf() {
            self.estimated_complexity
        } else {
            self.subtasks.iter().map(|s| s.total_complexity()).sum()
        }
    }

    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.subtasks.iter().map(|s| s.leaf_count()).sum()
        }
    }

    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            0
        } else {
            1 + self.subtasks.iter().map(|s| s.depth()).max().unwrap_or(0)
        }
    }
}

/// Decomposition strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecompositionStrategy {
    /// Break down by steps
    Sequential,
    /// Break down by components
    Functional,
    /// Break down by priority
    Priority,
    /// Break down by time
    Temporal,
}

/// Task decomposer
pub struct TaskDecomposer {
    strategy: DecompositionStrategy,
    max_depth: usize,
    min_complexity: f64,
    keyword_patterns: HashMap<String, Vec<String>>,
}

impl TaskDecomposer {
    pub fn new(strategy: DecompositionStrategy) -> Self {
        let mut decomposer = Self {
            strategy,
            max_depth: 5,
            min_complexity: 0.1,
            keyword_patterns: HashMap::new(),
        };

        decomposer.init_keyword_patterns();
        decomposer
    }

    fn init_keyword_patterns(&mut self) {
        // Sequential keywords
        self.keyword_patterns.insert(
            "sequential".to_string(),
            vec![
                "first".to_string(),
                "then".to_string(),
                "after".to_string(),
                "finally".to_string(),
                "next".to_string(),
            ],
        );

        // Functional keywords
        self.keyword_patterns.insert(
            "functional".to_string(),
            vec![
                "component".to_string(),
                "module".to_string(),
                "feature".to_string(),
                "function".to_string(),
                "part".to_string(),
            ],
        );

        // Action keywords that often need subtasks
        self.keyword_patterns.insert(
            "actions".to_string(),
            vec![
                "create".to_string(),
                "implement".to_string(),
                "build".to_string(),
                "develop".to_string(),
                "design".to_string(),
                "test".to_string(),
                "deploy".to_string(),
            ],
        );
    }

    pub fn decompose(&self, description: &str) -> TaskNode {
        let root_id = uuid::Uuid::new_v4().to_string();
        let mut root = TaskNode::new(&root_id, description);

        let subtasks = self.generate_subtasks(description, 0);
        root.subtasks = subtasks;

        self.calculate_complexities(&mut root);
        root
    }

    fn generate_subtasks(&self, description: &str, depth: usize) -> Vec<TaskNode> {
        if depth >= self.max_depth {
            return Vec::new();
        }

        let lower = description.to_lowercase();

        match self.strategy {
            DecompositionStrategy::Sequential => self.decompose_sequential(&lower, depth),
            DecompositionStrategy::Functional => self.decompose_functional(&lower, depth),
            DecompositionStrategy::Priority => self.decompose_by_priority(&lower, depth),
            DecompositionStrategy::Temporal => self.decompose_temporal(&lower, depth),
        }
    }

    fn decompose_sequential(&self, description: &str, _depth: usize) -> Vec<TaskNode> {
        let mut subtasks = Vec::new();
        let actions = self.extract_actions(description);

        // Standard phases for most tasks
        let phases = [
            ("analyze", "Analyze requirements and context"),
            ("plan", "Create detailed plan"),
            ("implement", "Implement the solution"),
            ("test", "Test and validate"),
            ("finalize", "Finalize and document"),
        ];

        let mut prev_id: Option<String> = None;

        for (phase_key, phase_desc) in phases {
            if actions.iter().any(|a| a.contains(phase_key)) || actions.is_empty() {
                let id = uuid::Uuid::new_v4().to_string();
                let mut task = TaskNode::new(&id, phase_desc);

                if let Some(ref prev) = prev_id {
                    task.dependencies.push(prev.clone());
                }

                prev_id = Some(id);
                subtasks.push(task);
            }
        }

        subtasks
    }

    fn decompose_functional(&self, description: &str, _depth: usize) -> Vec<TaskNode> {
        let mut subtasks = Vec::new();

        // Identify functional components
        let components = [
            ("data", "Data handling component"),
            ("logic", "Business logic component"),
            ("interface", "User interface component"),
            ("integration", "Integration component"),
            ("validation", "Validation component"),
        ];

        for (keyword, comp_desc) in components {
            if description.contains(keyword)
                || description.contains("full")
                || description.contains("complete")
            {
                let id = uuid::Uuid::new_v4().to_string();
                subtasks.push(TaskNode::new(&id, comp_desc));
            }
        }

        // If no specific components found, create generic ones
        if subtasks.is_empty() {
            subtasks.push(TaskNode::new(
                &uuid::Uuid::new_v4().to_string(),
                "Core functionality",
            ));
            subtasks.push(TaskNode::new(
                &uuid::Uuid::new_v4().to_string(),
                "Supporting features",
            ));
        }

        subtasks
    }

    fn decompose_by_priority(&self, _description: &str, _depth: usize) -> Vec<TaskNode> {
        let mut subtasks = Vec::new();

        // Create priority buckets
        let priorities = [
            ("critical", "Critical: Must-have features", 1.0),
            ("high", "High: Important features", 0.8),
            ("medium", "Medium: Nice-to-have features", 0.5),
            ("low", "Low: Optional enhancements", 0.3),
        ];

        for (_, priority_desc, complexity) in priorities {
            let id = uuid::Uuid::new_v4().to_string();
            let mut task = TaskNode::new(&id, priority_desc);
            task.estimated_complexity = complexity;
            subtasks.push(task);
        }

        subtasks
    }

    fn decompose_temporal(&self, _description: &str, _depth: usize) -> Vec<TaskNode> {
        let mut subtasks = Vec::new();

        let phases = [
            "Phase 1: Initial setup",
            "Phase 2: Core development",
            "Phase 3: Testing & refinement",
            "Phase 4: Deployment & monitoring",
        ];

        let mut prev_id: Option<String> = None;

        for phase in phases {
            let id = uuid::Uuid::new_v4().to_string();
            let mut task = TaskNode::new(&id, phase);

            if let Some(ref prev) = prev_id {
                task.dependencies.push(prev.clone());
            }

            prev_id = Some(id.clone());
            subtasks.push(task);
        }

        subtasks
    }

    fn extract_actions(&self, description: &str) -> Vec<String> {
        let actions = self
            .keyword_patterns
            .get("actions")
            .cloned()
            .unwrap_or_default();

        actions
            .into_iter()
            .filter(|a| description.contains(a))
            .collect()
    }

    fn calculate_complexities(&self, node: &mut TaskNode) {
        if node.is_leaf() {
            // Base complexity on description length and keywords
            let word_count = node.description.split_whitespace().count();
            node.estimated_complexity = (word_count as f64 / 10.0).max(self.min_complexity);
        } else {
            for subtask in &mut node.subtasks {
                self.calculate_complexities(subtask);
            }
            // Parent complexity is sum of children
            node.estimated_complexity = node.subtasks.iter().map(|s| s.estimated_complexity).sum();
        }
    }

    pub fn flatten(&self, root: &TaskNode) -> Vec<FlatTask> {
        let mut tasks = Vec::new();
        self.flatten_recursive(root, None, &mut tasks, 0);
        tasks
    }

    fn flatten_recursive(
        &self,
        node: &TaskNode,
        parent_id: Option<&str>,
        tasks: &mut Vec<FlatTask>,
        order: usize,
    ) {
        let flat = FlatTask {
            id: node.id.clone(),
            description: node.description.clone(),
            parent_id: parent_id.map(|s| s.to_string()),
            dependencies: node.dependencies.clone(),
            complexity: node.estimated_complexity,
            order,
            is_leaf: node.is_leaf(),
        };

        tasks.push(flat);

        for (i, subtask) in node.subtasks.iter().enumerate() {
            self.flatten_recursive(subtask, Some(&node.id), tasks, i);
        }
    }

    pub fn get_execution_order(&self, root: &TaskNode) -> Vec<String> {
        let flat = self.flatten(root);
        let mut order = Vec::new();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Topological sort considering dependencies
        while order.len() < flat.len() {
            for task in &flat {
                if completed.contains(&task.id) {
                    continue;
                }

                let deps_met = task.dependencies.iter().all(|d| completed.contains(d));

                if deps_met && task.is_leaf {
                    order.push(task.id.clone());
                    completed.insert(task.id.clone());
                } else if deps_met && !task.is_leaf {
                    // Skip non-leaf nodes, their children will be executed
                    completed.insert(task.id.clone());
                }
            }

            // Prevent infinite loop
            if order.len() == completed.len() - flat.iter().filter(|t| !t.is_leaf).count() {
                break;
            }
        }

        order
    }
}

impl Default for TaskDecomposer {
    fn default() -> Self {
        Self::new(DecompositionStrategy::Sequential)
    }
}

/// Flat task representation
#[derive(Debug, Clone)]
pub struct FlatTask {
    pub id: String,
    pub description: String,
    pub parent_id: Option<String>,
    pub dependencies: Vec<String>,
    pub complexity: f64,
    pub order: usize,
    pub is_leaf: bool,
}

/// Decomposition analysis
#[derive(Debug, Clone)]
pub struct DecompositionAnalysis {
    pub total_tasks: usize,
    pub leaf_tasks: usize,
    pub max_depth: usize,
    pub total_complexity: f64,
    pub avg_complexity: f64,
    pub parallelizable: usize,
}

impl TaskDecomposer {
    pub fn analyze(&self, root: &TaskNode) -> DecompositionAnalysis {
        let flat = self.flatten(root);
        let leaf_tasks: Vec<_> = flat.iter().filter(|t| t.is_leaf).collect();

        // Count parallelizable tasks (no dependencies on incomplete tasks)
        let parallelizable = leaf_tasks
            .iter()
            .filter(|t| t.dependencies.is_empty())
            .count();

        let total_complexity: f64 = leaf_tasks.iter().map(|t| t.complexity).sum();
        let avg_complexity = if !leaf_tasks.is_empty() {
            total_complexity / leaf_tasks.len() as f64
        } else {
            0.0
        };

        DecompositionAnalysis {
            total_tasks: flat.len(),
            leaf_tasks: leaf_tasks.len(),
            max_depth: root.depth(),
            total_complexity,
            avg_complexity,
            parallelizable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Sequential);
        let root = decomposer.decompose("Build a web application");

        // The root should exist and have an ID
        assert!(!root.id.is_empty());
        assert!(!root.description.is_empty());
    }

    #[test]
    fn test_flatten() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Sequential);
        let root = decomposer.decompose("Create a REST API");

        let flat = decomposer.flatten(&root);
        assert!(!flat.is_empty());
    }

    #[test]
    fn test_complexity() {
        let mut node = TaskNode::new("1", "Complex task")
            .with_subtask(TaskNode::new("1.1", "Subtask 1").with_complexity(2.0))
            .with_subtask(TaskNode::new("1.2", "Subtask 2").with_complexity(3.0));

        // Recalculate
        node.estimated_complexity = node.subtasks.iter().map(|s| s.estimated_complexity).sum();

        assert_eq!(node.total_complexity(), 5.0);
    }

    #[test]
    fn test_analysis() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Functional);
        let root = decomposer.decompose("Implement complete data processing pipeline");

        let analysis = decomposer.analyze(&root);
        assert!(analysis.total_tasks > 0);
    }

    #[test]
    fn test_task_node_leaf() {
        let leaf = TaskNode::new("1", "Leaf task");
        assert!(leaf.is_leaf());
        assert_eq!(leaf.leaf_count(), 1);
        assert_eq!(leaf.depth(), 0);

        let parent = TaskNode::new("2", "Parent")
            .with_subtask(TaskNode::new("2.1", "Child"));
        assert!(!parent.is_leaf());
        assert_eq!(parent.leaf_count(), 1);
        assert_eq!(parent.depth(), 1);
    }

    #[test]
    fn test_task_node_depth() {
        let deep = TaskNode::new("r", "Root")
            .with_subtask(
                TaskNode::new("a", "Level 1")
                    .with_subtask(
                        TaskNode::new("b", "Level 2")
                            .with_subtask(TaskNode::new("c", "Level 3")),
                    ),
            );
        assert_eq!(deep.depth(), 3);
        assert_eq!(deep.leaf_count(), 1);
    }

    #[test]
    fn test_task_node_builders() {
        let node = TaskNode::new("1", "Task")
            .with_dependency("dep1")
            .with_dependency("dep2")
            .with_capability("rust")
            .with_complexity(3.5);
        assert_eq!(node.dependencies.len(), 2);
        assert_eq!(node.required_capabilities, vec!["rust"]);
        assert!((node.estimated_complexity - 3.5).abs() < f64::EPSILON);
        assert_eq!(node.status, DecompositionStatus::NotStarted);
    }

    #[test]
    fn test_functional_decomposition() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Functional);
        let root = decomposer.decompose("Build a simple tool");
        // Functional decomposition with no matching keywords yields generic subtasks
        assert!(root.subtasks.len() >= 2);
    }

    #[test]
    fn test_priority_decomposition() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Priority);
        let root = decomposer.decompose("Create a system");
        // Priority decomposition always produces 4 buckets
        assert_eq!(root.subtasks.len(), 4);
        // First bucket has highest complexity (1.0), last has lowest (0.3)
        assert!(root.subtasks[0].estimated_complexity >= root.subtasks[3].estimated_complexity);
    }

    #[test]
    fn test_temporal_decomposition() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::Temporal);
        let root = decomposer.decompose("Deploy application");
        // Temporal decomposition produces 4 phases with sequential dependencies
        assert_eq!(root.subtasks.len(), 4);
        // Phases 2-4 should each depend on the previous phase
        assert!(root.subtasks[1].dependencies.len() >= 1);
        assert!(root.subtasks[2].dependencies.len() >= 1);
        assert!(root.subtasks[3].dependencies.len() >= 1);
    }
}
