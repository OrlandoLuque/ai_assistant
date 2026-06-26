//! Distributed bandit training (state merging) and snapshot export/import.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
            global_arms: router
                .global_bandit
                .iter()
                .filter(|a| !router.private_arms.contains(&a.id))
                .cloned()
                .collect(),
            task_bandits: router
                .bandits
                .iter()
                .map(|(k, arms)| {
                    let filtered: Vec<_> = arms
                        .iter()
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

        let global_arms: Vec<BanditArm> = global_map
            .into_iter()
            .map(|(id, arms)| Self::merge_arms(&id, &arms, n, prior_alpha, prior_beta))
            .collect();

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
            let merged: Vec<BanditArm> = arm_map
                .into_iter()
                .map(|(id, arms)| {
                    Self::merge_arms(&id, &arms, contributing_nodes, prior_alpha, prior_beta)
                })
                .collect();
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
            if let Some(local_arm) = router
                .global_bandit
                .iter_mut()
                .find(|a| a.id == remote_arm.id)
            {
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
                    let merged =
                        Self::merge_arm_pair(local_arm, remote_arm, prior_alpha, prior_beta);
                    *local_arm = merged;
                } else {
                    local_arms.push(remote_arm.clone());
                }
            }
        }

        router.total_pulls += remote.total_pulls;
        Ok(())
    }

    fn merge_arms(
        id: &str,
        arms: &[&BanditArm],
        n: usize,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> BanditArm {
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

    fn merge_arm_pair(
        local: &BanditArm,
        remote: &BanditArm,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> BanditArm {
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
        serde_json::to_string_pretty(&snapshot).map_err(|e| {
            AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
            }
        })
    }

    /// Import from JSON string.
    pub fn from_json(json: &str) -> Result<Self, AdvancedRoutingError> {
        let snapshot: BanditSnapshot =
            serde_json::from_str(json).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "JSON".to_string(),
                reason: e.to_string(),
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
        let json =
            std::str::from_utf8(bytes).map_err(|e| AdvancedRoutingError::SerializationFailed {
                format: "UTF-8".to_string(),
                reason: e.to_string(),
            })?;
        Self::from_json(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                arm_id: "a".to_string(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: None,
            });

            let mut r2 = BanditRouter::new(BanditConfig::default());
            r2.add_arm("a");
            r2.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(),
                success: true,
                quality: Some(0.6),
                latency_ms: None,
                cost: None,
                task_type: None,
            });

            let s1 = BanditStateMerger::extract_state(&r1, "n1");
            let s2 = BanditStateMerger::extract_state(&r2, "n2");

            let merged = BanditStateMerger::merge(&[s1, s2], 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms.len(), 1);
            assert_eq!(merged.global_arms[0].pull_count, 2);
        }

        #[test]
        fn test_merge_three_nodes() {
            let states: Vec<DistributedBanditState> = (0..3)
                .map(|i| {
                    let mut r = BanditRouter::new(BanditConfig::default());
                    r.add_arm("shared");
                    r.record_outcome(&ArmFeedback {
                        arm_id: "shared".to_string(),
                        success: true,
                        quality: Some(0.7),
                        latency_ms: None,
                        cost: None,
                        task_type: None,
                    });
                    BanditStateMerger::extract_state(&r, &format!("node-{}", i))
                })
                .collect();

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
                arm_id: "a".to_string(),
                success: true,
                quality: Some(0.9),
                latency_ms: None,
                cost: None,
                task_type: None,
            });

            let mut remote = BanditRouter::new(BanditConfig::default());
            remote.add_arm("a");
            remote.record_outcome(&ArmFeedback {
                arm_id: "a".to_string(),
                success: true,
                quality: Some(0.7),
                latency_ms: None,
                cost: None,
                task_type: None,
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
            let states: Vec<DistributedBanditState> = (0..5)
                .map(|i| {
                    let mut r = BanditRouter::new(BanditConfig::default());
                    r.add_arm("x");
                    for _ in 0..3 {
                        r.record_outcome(&ArmFeedback {
                            arm_id: "x".to_string(),
                            success: true,
                            quality: Some(0.8),
                            latency_ms: None,
                            cost: None,
                            task_type: None,
                        });
                    }
                    BanditStateMerger::extract_state(&r, &format!("n{}", i))
                })
                .collect();

            let merged = BanditStateMerger::merge(&states, 1.0, 1.0).unwrap();
            assert_eq!(merged.global_arms[0].pull_count, 15); // 5 nodes * 3 pulls
        }

        #[test]
        fn test_merge_idempotent() {
            let mut r = BanditRouter::new(BanditConfig::default());
            r.add_arm("x");
            r.record_outcome(&ArmFeedback {
                arm_id: "x".to_string(),
                success: true,
                quality: Some(0.8),
                latency_ms: None,
                cost: None,
                task_type: None,
            });

            let state = BanditStateMerger::extract_state(&r, "n1");
            // Merging single state = itself (with prior correction = no change)
            let merged = BanditStateMerger::merge(std::slice::from_ref(&state), 1.0, 1.0).unwrap();
            assert_eq!(
                merged.global_arms[0].pull_count,
                state.global_arms[0].pull_count
            );
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
            arm_id: "test-model".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: None,
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
}
