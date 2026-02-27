//! Distributed agents — multi-node agent execution
//!
//! Runs autonomous agents across distributed nodes using the existing
//! DHT, CRDT, and coordinator infrastructure.

use crate::distributed::NodeId;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Types
// =============================================================================

/// Information about an agent node in the distributed network.
#[derive(Debug)]
pub struct AgentNodeInfo {
    pub node_id: NodeId,
    pub agent_profile: String,
    pub status: String,
    pub current_task: Option<String>,
    pub load: f64,
    pub last_heartbeat: u64,
}

/// A task submitted for distributed execution.
#[derive(Debug)]
pub struct DistributedTask {
    pub id: String,
    pub description: String,
    pub profile: String,
    pub priority: u32,
    pub assigned_node: Option<NodeId>,
    pub status: TaskDistributionStatus,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub result: Option<String>,
}

/// Status of a distributed task.
#[derive(Debug)]
pub enum TaskDistributionStatus {
    Queued,
    Assigned,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// A map-reduce style job that fans out work to multiple agents and then
/// reduces the collected results with a single agent.
#[derive(Debug)]
pub struct MapReduceAgentJob {
    pub id: String,
    pub map_tasks: Vec<String>,
    pub map_profile: String,
    pub reduce_prompt: String,
    pub reduce_profile: String,
    pub status: MapReduceStatus,
    pub map_results: HashMap<String, String>,
    pub reduce_result: Option<String>,
}

/// Status of a map-reduce agent job.
#[derive(Debug)]
pub enum MapReduceStatus {
    Pending,
    Mapping,
    Reducing,
    Completed,
    Failed(String),
}

// =============================================================================
// Helpers
// =============================================================================

/// Return current wall-clock time in milliseconds since UNIX epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Convert a `NodeId` to a hex string suitable for use as a `HashMap` key.
fn node_hex(id: &NodeId) -> String {
    id.to_hex()
}

// =============================================================================
// DistributedAgentManager
// =============================================================================

/// Manages distributed agent tasks, worker nodes, and map-reduce jobs.
#[derive(Debug)]
pub struct DistributedAgentManager {
    local_node_id: NodeId,
    tasks: HashMap<String, DistributedTask>,
    nodes: HashMap<String, AgentNodeInfo>,
    mr_jobs: HashMap<String, MapReduceAgentJob>,
    next_task_id: u64,
    next_mr_id: u64,
}

impl DistributedAgentManager {
    /// Create a new manager for the given local node.
    pub fn new(local_node_id: NodeId) -> Self {
        Self {
            local_node_id,
            tasks: HashMap::new(),
            nodes: HashMap::new(),
            mr_jobs: HashMap::new(),
            next_task_id: 0,
            next_mr_id: 0,
        }
    }

    // -------------------------------------------------------------------------
    // Task management
    // -------------------------------------------------------------------------

    /// Submit a new task and return its unique id.
    pub fn submit_task(&mut self, description: &str, profile: &str, priority: u32) -> String {
        self.next_task_id += 1;
        let id = format!("task-{}", self.next_task_id);
        let task = DistributedTask {
            id: id.clone(),
            description: description.to_string(),
            profile: profile.to_string(),
            priority,
            assigned_node: None,
            status: TaskDistributionStatus::Queued,
            created_at: now_ms(),
            started_at: None,
            completed_at: None,
            result: None,
        };
        self.tasks.insert(id.clone(), task);
        id
    }

    /// Cancel a queued or assigned task. Returns `true` if the task was found
    /// and successfully cancelled.
    pub fn cancel_task(&mut self, task_id: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(task_id) {
            match task.status {
                TaskDistributionStatus::Queued | TaskDistributionStatus::Assigned => {
                    task.status = TaskDistributionStatus::Cancelled;
                    task.completed_at = Some(now_ms());
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Look up a task by id.
    pub fn get_task(&self, task_id: &str) -> Option<&DistributedTask> {
        self.tasks.get(task_id)
    }

    /// Return all tasks.
    pub fn list_tasks(&self) -> Vec<&DistributedTask> {
        self.tasks.values().collect()
    }

    /// Return tasks that are still queued.
    pub fn pending_tasks(&self) -> Vec<&DistributedTask> {
        self.tasks
            .values()
            .filter(|t| matches!(t.status, TaskDistributionStatus::Queued))
            .collect()
    }

    /// Return tasks that have completed successfully.
    pub fn completed_tasks(&self) -> Vec<&DistributedTask> {
        self.tasks
            .values()
            .filter(|t| matches!(t.status, TaskDistributionStatus::Completed))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Task claiming (worker nodes)
    // -------------------------------------------------------------------------

    /// Claim the highest-priority queued task for the given node.
    /// Returns a mutable reference to the task so the caller can inspect it.
    pub fn claim_task(&mut self, node_id: &NodeId) -> Option<&mut DistributedTask> {
        // Find the id of the highest-priority Queued task.
        let best_id = self
            .tasks
            .values()
            .filter(|t| matches!(t.status, TaskDistributionStatus::Queued))
            .max_by_key(|t| t.priority)
            .map(|t| t.id.clone());

        if let Some(id) = best_id {
            let task = self.tasks.get_mut(&id).expect("task id just verified");
            task.status = TaskDistributionStatus::Assigned;
            task.assigned_node = Some(*node_id);
            task.started_at = Some(now_ms());
            Some(task)
        } else {
            None
        }
    }

    /// Mark a task as completed with the given result string.
    pub fn complete_task(&mut self, task_id: &str, result: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(task_id) {
            match task.status {
                TaskDistributionStatus::Assigned | TaskDistributionStatus::Running => {
                    task.status = TaskDistributionStatus::Completed;
                    task.completed_at = Some(now_ms());
                    task.result = Some(result.to_string());
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Mark a task as failed with the given error message.
    pub fn fail_task(&mut self, task_id: &str, error: &str) -> bool {
        if let Some(task) = self.tasks.get_mut(task_id) {
            match task.status {
                TaskDistributionStatus::Assigned | TaskDistributionStatus::Running => {
                    task.status = TaskDistributionStatus::Failed(error.to_string());
                    task.completed_at = Some(now_ms());
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    // -------------------------------------------------------------------------
    // Node management
    // -------------------------------------------------------------------------

    /// Register (or update) an agent node.
    pub fn register_node(&mut self, info: AgentNodeInfo) {
        let key = node_hex(&info.node_id);
        self.nodes.insert(key, info);
    }

    /// Refresh the heartbeat timestamp for a node to `now`.
    pub fn update_heartbeat(&mut self, node_id: &NodeId) {
        let key = node_hex(node_id);
        if let Some(info) = self.nodes.get_mut(&key) {
            info.last_heartbeat = now_ms();
        }
    }

    /// Look up a node by its id.
    pub fn get_node(&self, node_id: &NodeId) -> Option<&AgentNodeInfo> {
        let key = node_hex(node_id);
        self.nodes.get(&key)
    }

    /// Return nodes whose status is not `"offline"`.
    pub fn active_nodes(&self) -> Vec<&AgentNodeInfo> {
        self.nodes
            .values()
            .filter(|n| n.status != "offline")
            .collect()
    }

    /// Return nodes whose last heartbeat is older than `timeout_ms` milliseconds.
    pub fn stale_nodes(&self, timeout_ms: u64) -> Vec<&AgentNodeInfo> {
        let now = now_ms();
        self.nodes
            .values()
            .filter(|n| now.saturating_sub(n.last_heartbeat) > timeout_ms)
            .collect()
    }

    // -------------------------------------------------------------------------
    // MapReduce
    // -------------------------------------------------------------------------

    /// Submit a new map-reduce agent job and return its id.
    pub fn submit_map_reduce(
        &mut self,
        map_tasks: Vec<String>,
        map_profile: &str,
        reduce_prompt: &str,
        reduce_profile: &str,
    ) -> String {
        self.next_mr_id += 1;
        let id = format!("mr-{}", self.next_mr_id);
        let job = MapReduceAgentJob {
            id: id.clone(),
            map_tasks,
            map_profile: map_profile.to_string(),
            reduce_prompt: reduce_prompt.to_string(),
            reduce_profile: reduce_profile.to_string(),
            status: MapReduceStatus::Pending,
            map_results: HashMap::new(),
            reduce_result: None,
        };
        self.mr_jobs.insert(id.clone(), job);
        id
    }

    /// Record a result for one map task inside a map-reduce job.
    /// If all map tasks now have results the job status transitions to `Reducing`.
    /// Returns `true` if the result was recorded.
    pub fn record_map_result(&mut self, job_id: &str, task_desc: &str, result: &str) -> bool {
        if let Some(job) = self.mr_jobs.get_mut(job_id) {
            // Only accept results while still in Pending or Mapping phase.
            match job.status {
                MapReduceStatus::Pending => {
                    job.status = MapReduceStatus::Mapping;
                }
                MapReduceStatus::Mapping => {}
                _ => return false,
            }

            job.map_results
                .insert(task_desc.to_string(), result.to_string());

            // Transition to Reducing once every map task has a result.
            if job.map_results.len() == job.map_tasks.len() {
                job.status = MapReduceStatus::Reducing;
            }
            true
        } else {
            false
        }
    }

    /// Check whether all map tasks in a job have produced results.
    pub fn is_map_phase_complete(&self, job_id: &str) -> bool {
        if let Some(job) = self.mr_jobs.get(job_id) {
            job.map_results.len() == job.map_tasks.len()
        } else {
            false
        }
    }

    /// Store the reduce result and mark the job as completed.
    pub fn complete_reduce(&mut self, job_id: &str, result: &str) -> bool {
        if let Some(job) = self.mr_jobs.get_mut(job_id) {
            job.reduce_result = Some(result.to_string());
            job.status = MapReduceStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Look up a map-reduce job by id.
    pub fn get_mr_job(&self, job_id: &str) -> Option<&MapReduceAgentJob> {
        self.mr_jobs.get(job_id)
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    /// Total number of tasks (all statuses).
    /// Get the local node ID.
    pub fn local_node_id(&self) -> &NodeId {
        &self.local_node_id
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Total number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of tasks currently in `Assigned` or `Running` state.
    pub fn active_task_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|t| {
                matches!(
                    t.status,
                    TaskDistributionStatus::Assigned | TaskDistributionStatus::Running
                )
            })
            .count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> DistributedAgentManager {
        DistributedAgentManager::new(NodeId::from_string("local"))
    }

    // 1. test_submit_task
    #[test]
    fn test_submit_task() {
        let mut mgr = make_manager();
        let id = mgr.submit_task("Summarize document", "researcher", 5);
        assert!(!id.is_empty());

        let pending = mgr.pending_tasks();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].description, "Summarize document");
    }

    // 2. test_claim_task
    #[test]
    fn test_claim_task() {
        let mut mgr = make_manager();
        mgr.submit_task("Do work", "worker", 1);

        let worker = NodeId::from_string("worker-1");
        let task = mgr.claim_task(&worker).unwrap();
        assert!(matches!(task.status, TaskDistributionStatus::Assigned));
        assert_eq!(task.assigned_node.unwrap(), worker);
        assert!(task.started_at.is_some());
    }

    // 3. test_claim_highest_priority
    #[test]
    fn test_claim_highest_priority() {
        let mut mgr = make_manager();
        mgr.submit_task("low", "worker", 1);
        mgr.submit_task("high", "worker", 10);
        mgr.submit_task("medium", "worker", 5);

        let worker = NodeId::from_string("w");
        let task = mgr.claim_task(&worker).unwrap();
        assert_eq!(task.description, "high");
        assert_eq!(task.priority, 10);
    }

    // 4. test_complete_task
    #[test]
    fn test_complete_task() {
        let mut mgr = make_manager();
        let id = mgr.submit_task("Analyze data", "analyst", 3);

        let worker = NodeId::from_string("w");
        mgr.claim_task(&worker);

        assert!(mgr.complete_task(&id, "analysis done"));
        let task = mgr.get_task(&id).unwrap();
        assert!(matches!(task.status, TaskDistributionStatus::Completed));
        assert_eq!(task.result.as_deref(), Some("analysis done"));
        assert!(task.completed_at.is_some());
    }

    // 5. test_fail_task
    #[test]
    fn test_fail_task() {
        let mut mgr = make_manager();
        let id = mgr.submit_task("Risky job", "agent", 2);

        let worker = NodeId::from_string("w");
        mgr.claim_task(&worker);

        assert!(mgr.fail_task(&id, "timeout"));
        let task = mgr.get_task(&id).unwrap();
        match &task.status {
            TaskDistributionStatus::Failed(e) => assert_eq!(e, "timeout"),
            _ => panic!("expected Failed status"),
        }
    }

    // 6. test_cancel_task
    #[test]
    fn test_cancel_task() {
        let mut mgr = make_manager();
        let id = mgr.submit_task("Cancel me", "worker", 1);
        assert!(mgr.cancel_task(&id));

        let task = mgr.get_task(&id).unwrap();
        assert!(matches!(task.status, TaskDistributionStatus::Cancelled));
    }

    // 7. test_register_node
    #[test]
    fn test_register_node() {
        let mut mgr = make_manager();
        let nid = NodeId::from_string("node-alpha");
        mgr.register_node(AgentNodeInfo {
            node_id: nid,
            agent_profile: "researcher".to_string(),
            status: "idle".to_string(),
            current_task: None,
            load: 0.0,
            last_heartbeat: now_ms(),
        });

        let node = mgr.get_node(&nid).unwrap();
        assert_eq!(node.agent_profile, "researcher");
        assert_eq!(node.status, "idle");
    }

    // 8. test_stale_nodes
    #[test]
    fn test_stale_nodes() {
        let mut mgr = make_manager();
        let nid = NodeId::from_string("old-node");
        // Heartbeat far in the past.
        mgr.register_node(AgentNodeInfo {
            node_id: nid,
            agent_profile: "worker".to_string(),
            status: "idle".to_string(),
            current_task: None,
            load: 0.1,
            last_heartbeat: 1000, // very old
        });

        let stale = mgr.stale_nodes(5000);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].node_id, nid);
    }

    // 9. test_map_reduce_submit
    #[test]
    fn test_map_reduce_submit() {
        let mut mgr = make_manager();
        let id = mgr.submit_map_reduce(
            vec!["chunk-1".into(), "chunk-2".into()],
            "mapper",
            "Combine results",
            "reducer",
        );
        assert!(!id.is_empty());

        let job = mgr.get_mr_job(&id).unwrap();
        assert_eq!(job.map_tasks.len(), 2);
        assert!(matches!(job.status, MapReduceStatus::Pending));
    }

    // 10. test_map_reduce_results
    #[test]
    fn test_map_reduce_results() {
        let mut mgr = make_manager();
        let id = mgr.submit_map_reduce(vec!["a".into(), "b".into()], "mapper", "reduce", "reducer");

        assert!(!mgr.is_map_phase_complete(&id));

        assert!(mgr.record_map_result(&id, "a", "result-a"));
        assert!(!mgr.is_map_phase_complete(&id));

        assert!(mgr.record_map_result(&id, "b", "result-b"));
        assert!(mgr.is_map_phase_complete(&id));

        let job = mgr.get_mr_job(&id).unwrap();
        assert!(matches!(job.status, MapReduceStatus::Reducing));
    }

    // 11. test_map_reduce_reduce
    #[test]
    fn test_map_reduce_reduce() {
        let mut mgr = make_manager();
        let id = mgr.submit_map_reduce(vec!["x".into()], "mapper", "combine", "reducer");

        mgr.record_map_result(&id, "x", "mapped-x");
        assert!(mgr.complete_reduce(&id, "final-result"));

        let job = mgr.get_mr_job(&id).unwrap();
        assert!(matches!(job.status, MapReduceStatus::Completed));
        assert_eq!(job.reduce_result.as_deref(), Some("final-result"));
    }

    // 12. test_task_stats
    #[test]
    fn test_task_stats() {
        let mut mgr = make_manager();
        assert_eq!(mgr.task_count(), 0);
        assert_eq!(mgr.active_task_count(), 0);

        mgr.submit_task("t1", "p", 1);
        mgr.submit_task("t2", "p", 2);
        assert_eq!(mgr.task_count(), 2);

        let worker = NodeId::from_string("w");
        mgr.claim_task(&worker);
        assert_eq!(mgr.active_task_count(), 1);

        // Register a node for node_count check.
        mgr.register_node(AgentNodeInfo {
            node_id: worker,
            agent_profile: "p".to_string(),
            status: "busy".to_string(),
            current_task: None,
            load: 0.5,
            last_heartbeat: now_ms(),
        });
        assert_eq!(mgr.node_count(), 1);
    }

    // 13. test_list_tasks_all
    #[test]
    fn test_list_tasks_all() {
        let mut mgr = make_manager();
        mgr.submit_task("alpha", "worker", 1);
        mgr.submit_task("beta", "worker", 2);
        mgr.submit_task("gamma", "worker", 3);

        let all = mgr.list_tasks();
        assert_eq!(all.len(), 3);

        let descriptions: Vec<&str> = all.iter().map(|t| t.description.as_str()).collect();
        assert!(descriptions.contains(&"alpha"));
        assert!(descriptions.contains(&"beta"));
        assert!(descriptions.contains(&"gamma"));
    }

    // 14. test_completed_tasks_filter
    #[test]
    fn test_completed_tasks_filter() {
        let mut mgr = make_manager();
        let _id1 = mgr.submit_task("task-a", "worker", 1);
        let _id2 = mgr.submit_task("task-b", "worker", 2);
        let _id3 = mgr.submit_task("task-c", "worker", 3);

        let worker = NodeId::from_string("w");

        // Claim and complete task-b (highest priority among remaining queued)
        mgr.claim_task(&worker);
        // task-b has priority 2, task-c has priority 3 — claim gets highest first
        // After first claim: task-c is assigned (priority 3)
        // Complete whichever was claimed
        let claimed_id = mgr
            .list_tasks()
            .iter()
            .find(|t| matches!(t.status, TaskDistributionStatus::Assigned))
            .map(|t| t.id.clone())
            .unwrap();
        mgr.complete_task(&claimed_id, "done");

        // Claim and complete another
        mgr.claim_task(&worker);
        let claimed_id2 = mgr
            .list_tasks()
            .iter()
            .find(|t| matches!(t.status, TaskDistributionStatus::Assigned))
            .map(|t| t.id.clone())
            .unwrap();
        mgr.complete_task(&claimed_id2, "also done");

        // Two tasks completed, one still queued
        let completed = mgr.completed_tasks();
        assert_eq!(completed.len(), 2);
        for t in &completed {
            assert!(matches!(t.status, TaskDistributionStatus::Completed));
        }

        // One task still pending
        let pending = mgr.pending_tasks();
        assert_eq!(pending.len(), 1);
    }

    // 15. test_update_heartbeat
    #[test]
    fn test_update_heartbeat() {
        let mut mgr = make_manager();
        let nid = NodeId::from_string("heartbeat-node");

        // Register with an old heartbeat
        mgr.register_node(AgentNodeInfo {
            node_id: nid,
            agent_profile: "worker".to_string(),
            status: "idle".to_string(),
            current_task: None,
            load: 0.0,
            last_heartbeat: 1000, // very old timestamp
        });

        let old_hb = mgr.get_node(&nid).unwrap().last_heartbeat;
        assert_eq!(old_hb, 1000);

        // Update heartbeat
        mgr.update_heartbeat(&nid);

        let new_hb = mgr.get_node(&nid).unwrap().last_heartbeat;
        assert!(new_hb > old_hb);
        // The new heartbeat should be a recent wall-clock value (way larger than 1000)
        assert!(new_hb > 1_000_000);
    }

    // 16. test_active_nodes
    #[test]
    fn test_active_nodes() {
        let mut mgr = make_manager();

        mgr.register_node(AgentNodeInfo {
            node_id: NodeId::from_string("online-1"),
            agent_profile: "worker".to_string(),
            status: "idle".to_string(),
            current_task: None,
            load: 0.1,
            last_heartbeat: now_ms(),
        });
        mgr.register_node(AgentNodeInfo {
            node_id: NodeId::from_string("online-2"),
            agent_profile: "researcher".to_string(),
            status: "busy".to_string(),
            current_task: Some("task-1".into()),
            load: 0.8,
            last_heartbeat: now_ms(),
        });
        mgr.register_node(AgentNodeInfo {
            node_id: NodeId::from_string("gone"),
            agent_profile: "worker".to_string(),
            status: "offline".to_string(),
            current_task: None,
            load: 0.0,
            last_heartbeat: 500,
        });

        assert_eq!(mgr.node_count(), 3);

        let active = mgr.active_nodes();
        assert_eq!(active.len(), 2);
        for node in &active {
            assert_ne!(node.status, "offline");
        }
    }

    // 17. test_local_node_id
    #[test]
    fn test_local_node_id() {
        let nid = NodeId::from_string("my-node");
        let mgr = DistributedAgentManager::new(nid);
        assert_eq!(*mgr.local_node_id(), nid);
    }
}
