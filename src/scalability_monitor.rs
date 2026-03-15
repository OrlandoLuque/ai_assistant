//! # Scalability Monitor (v32)
//!
//! Runtime monitoring for data structures that degrade beyond their optimal
//! operating range. Emits `log::warn!` with actionable recommendations when
//! thresholds are approached or exceeded.
//!
//! ## Usage
//!
//! Each monitored module calls [`check_scalability`] after mutating operations:
//! ```ignore
//! #[cfg(feature = "analytics")]
//! crate::scalability_monitor::check_scalability(
//!     crate::scalability_monitor::Subsystem::VectorDbInMemory,
//!     self.vectors.len(),
//! );
//! ```
//!
//! The function uses `thread_local!` cooldown to avoid log spam (max 1 warning
//! per subsystem per 60 seconds).
//!
//! For batch health checks, build a [`ScalabilitySnapshot`] and call
//! [`audit_snapshot`] to get all warnings at once.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ============================================================================
// Subsystem identification
// ============================================================================

/// Identifies a monitored subsystem with potential scalability limits.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Subsystem {
    /// In-memory vector DB (HashMap-based, optimal < 10K vectors).
    VectorDbInMemory,
    /// HNSW approximate nearest-neighbor index (all nodes in memory).
    VectorDbHnsw,
    /// Embedding cache (LRU, all embeddings in RAM).
    EmbeddingCache,
    /// SQLite-backed knowledge graph (PageRank/algorithms load full graph).
    KnowledgeGraph,
    /// In-memory multi-layer graph for RAG.
    MultiLayerGraph,
    /// Chat session store (unbounded Vec).
    SessionStore,
    /// Fact store (unbounded Vec + HashMap indices).
    FactStore,
    /// LLM response cache.
    ResponseCache,
    /// Episodic memory store.
    EpisodicMemory,
    /// Entity store in advanced memory (unbounded HashMap).
    EntityStore,
    /// ORSet CRDT tombstones (never garbage-collected).
    CrdtOrSetTombstones,
    /// DHT key-value storage (unbounded HashMap).
    DhtStorage,
}

impl fmt::Display for Subsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VectorDbInMemory => write!(f, "VectorDb (InMemory)"),
            Self::VectorDbHnsw => write!(f, "HNSW Index"),
            Self::EmbeddingCache => write!(f, "Embedding Cache"),
            Self::KnowledgeGraph => write!(f, "Knowledge Graph"),
            Self::MultiLayerGraph => write!(f, "Multi-Layer Graph"),
            Self::SessionStore => write!(f, "Session Store"),
            Self::FactStore => write!(f, "Fact Store"),
            Self::ResponseCache => write!(f, "Response Cache"),
            Self::EpisodicMemory => write!(f, "Episodic Memory"),
            Self::EntityStore => write!(f, "Entity Store"),
            Self::CrdtOrSetTombstones => write!(f, "ORSet Tombstones"),
            Self::DhtStorage => write!(f, "DHT Storage"),
        }
    }
}

// ============================================================================
// Warning severity
// ============================================================================

/// Severity level for scalability warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WarningSeverity {
    /// 60-79% of optimal limit — informational.
    Info,
    /// 80-94% of optimal limit — should take action soon.
    Warning,
    /// 95%+ of optimal limit or unbounded growth — immediate action needed.
    Critical,
}

impl fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ============================================================================
// Recommendations
// ============================================================================

/// An actionable recommendation for addressing a scalability issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScalabilityAction {
    /// Switch to a different backend.
    SwitchBackend {
        from: String,
        to: String,
        reason: String,
    },
    /// Enable a feature or capability.
    EnableFeature {
        feature: String,
        description: String,
    },
    /// Add a size limit where none exists.
    AddSizeLimit {
        suggested_max: usize,
        reason: String,
    },
    /// Run a maintenance operation.
    RunMaintenance {
        operation: String,
        description: String,
    },
    /// Reduce a configuration parameter.
    ReduceConfig {
        field: String,
        current: String,
        suggested: String,
        reason: String,
    },
    /// Generic recommendation.
    Custom { action: String, details: String },
}

/// Format a recommendation as a human-readable string.
pub fn format_action(action: &ScalabilityAction) -> String {
    match action {
        ScalabilityAction::SwitchBackend { from, to, reason } => {
            format!("Switch from {} to {} ({})", from, to, reason)
        }
        ScalabilityAction::EnableFeature {
            feature,
            description,
        } => {
            format!("Enable '{}': {}", feature, description)
        }
        ScalabilityAction::AddSizeLimit {
            suggested_max,
            reason,
        } => {
            format!("Add max size limit of {} ({})", suggested_max, reason)
        }
        ScalabilityAction::RunMaintenance {
            operation,
            description,
        } => {
            format!("Run '{}': {}", operation, description)
        }
        ScalabilityAction::ReduceConfig {
            field,
            suggested,
            reason,
            ..
        } => {
            format!("Reduce '{}' to {} ({})", field, suggested, reason)
        }
        ScalabilityAction::Custom { action, details } => {
            format!("{}: {}", action, details)
        }
    }
}

// ============================================================================
// Warning
// ============================================================================

/// A scalability warning with context and actionable recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityWarning {
    /// Which subsystem triggered the warning.
    pub subsystem: Subsystem,
    /// How urgent.
    pub severity: WarningSeverity,
    /// Current number of elements.
    pub current_size: usize,
    /// Optimal maximum for this backend/structure.
    pub optimal_max: usize,
    /// Utilization as a fraction (0.0-1.0+).
    pub utilization_pct: f64,
    /// Human-readable message.
    pub message: String,
    /// What to do about it.
    pub recommendations: Vec<ScalabilityAction>,
}

// ============================================================================
// Threshold definitions
// ============================================================================

/// Threshold configuration for a subsystem.
#[derive(Debug, Clone)]
struct ScalabilityThreshold {
    /// Beyond this, the structure degrades.
    optimal_max: usize,
    /// True if the structure has no hard limit (unbounded growth).
    unbounded: bool,
    /// For unbounded structures, emit warning after this many entries.
    soft_limit: Option<usize>,
}

/// Get the default threshold for a subsystem.
fn default_threshold(subsystem: &Subsystem) -> ScalabilityThreshold {
    match subsystem {
        Subsystem::VectorDbInMemory => ScalabilityThreshold {
            optimal_max: 10_000,
            unbounded: false,
            soft_limit: None,
        },
        Subsystem::VectorDbHnsw => ScalabilityThreshold {
            optimal_max: 100_000,
            unbounded: true,
            soft_limit: Some(100_000),
        },
        Subsystem::EmbeddingCache => ScalabilityThreshold {
            optimal_max: 10_000,
            unbounded: false,
            soft_limit: None,
        },
        Subsystem::KnowledgeGraph => ScalabilityThreshold {
            optimal_max: 100_000,
            unbounded: true,
            soft_limit: Some(100_000),
        },
        Subsystem::MultiLayerGraph => ScalabilityThreshold {
            optimal_max: 50_000,
            unbounded: true,
            soft_limit: Some(50_000),
        },
        Subsystem::SessionStore => ScalabilityThreshold {
            optimal_max: 1_000,
            unbounded: true,
            soft_limit: Some(1_000),
        },
        Subsystem::FactStore => ScalabilityThreshold {
            optimal_max: 10_000,
            unbounded: true,
            soft_limit: Some(10_000),
        },
        Subsystem::ResponseCache => ScalabilityThreshold {
            optimal_max: 1_000,
            unbounded: false,
            soft_limit: None,
        },
        Subsystem::EpisodicMemory => ScalabilityThreshold {
            optimal_max: 1_000,
            unbounded: false,
            soft_limit: None,
        },
        Subsystem::EntityStore => ScalabilityThreshold {
            optimal_max: 5_000,
            unbounded: true,
            soft_limit: Some(5_000),
        },
        Subsystem::CrdtOrSetTombstones => ScalabilityThreshold {
            optimal_max: 10_000,
            unbounded: true,
            soft_limit: Some(10_000),
        },
        Subsystem::DhtStorage => ScalabilityThreshold {
            optimal_max: 50_000,
            unbounded: true,
            soft_limit: Some(50_000),
        },
    }
}

// ============================================================================
// Per-subsystem recommendations
// ============================================================================

/// Build recommendations for a subsystem based on severity.
fn build_recommendations(subsystem: &Subsystem, severity: WarningSeverity) -> Vec<ScalabilityAction> {
    match subsystem {
        Subsystem::VectorDbInMemory => {
            let mut recs = vec![ScalabilityAction::SwitchBackend {
                from: "InMemory".into(),
                to: "LanceDB".into(),
                reason: "Disk-backed index, optimal for 10K-10M vectors".into(),
            }];
            if severity >= WarningSeverity::Critical {
                recs.push(ScalabilityAction::SwitchBackend {
                    from: "InMemory".into(),
                    to: "Qdrant or PgVector".into(),
                    reason: "Dedicated vector DB for 10M+ vectors with clustering support".into(),
                });
            }
            recs
        }
        Subsystem::VectorDbHnsw => {
            let mut recs = vec![ScalabilityAction::RunMaintenance {
                operation: "hnsw_compaction".into(),
                description: "Remove deleted nodes and rebuild connections to reclaim memory".into(),
            }];
            if severity >= WarningSeverity::Warning {
                recs.push(ScalabilityAction::SwitchBackend {
                    from: "In-memory HNSW".into(),
                    to: "Qdrant (dedicated HNSW server)".into(),
                    reason: "Qdrant manages HNSW internally with optimized memory mapping".into(),
                });
            }
            recs
        }
        Subsystem::EmbeddingCache => {
            vec![
                ScalabilityAction::EnableFeature {
                    feature: "persistent_cache".into(),
                    description: "Enable disk-backed embedding cache to reduce RAM usage".into(),
                },
                ScalabilityAction::ReduceConfig {
                    field: "ttl".into(),
                    current: "24h".into(),
                    suggested: "4h".into(),
                    reason: "Shorter TTL reduces cache size at the cost of more recomputations"
                        .into(),
                },
            ]
        }
        Subsystem::KnowledgeGraph => {
            let mut recs = vec![ScalabilityAction::ReduceConfig {
                field: "max_traversal_depth".into(),
                current: "2".into(),
                suggested: "1".into(),
                reason: "Depth-2 traversal is O(n^2) on dense graphs; depth-1 is much cheaper"
                    .into(),
            }];
            if severity >= WarningSeverity::Warning {
                recs.push(ScalabilityAction::Custom {
                    action: "Avoid full-graph algorithms".into(),
                    details: "PageRank, ConnectedComponents load entire graph into RAM. \
                        Use targeted BFS queries instead."
                        .into(),
                });
            }
            if severity >= WarningSeverity::Critical {
                recs.push(ScalabilityAction::SwitchBackend {
                    from: "SQLite KnowledgeGraph".into(),
                    to: "Distributed graph or dedicated graph DB".into(),
                    reason: "100K+ entities benefit from a graph database with native indexing"
                        .into(),
                });
            }
            recs
        }
        Subsystem::MultiLayerGraph => {
            vec![
                ScalabilityAction::Custom {
                    action: "Archive old session layers".into(),
                    details: "Session-layer entities are temporary; \
                        archive completed sessions to reduce in-memory size."
                        .into(),
                },
                ScalabilityAction::ReduceConfig {
                    field: "entity_retention".into(),
                    current: "unlimited".into(),
                    suggested: "10000".into(),
                    reason: "Cap per-layer entity count to prevent unbounded growth".into(),
                },
            ]
        }
        Subsystem::SessionStore => {
            vec![
                ScalabilityAction::AddSizeLimit {
                    suggested_max: 500,
                    reason: "Unbounded session store consumes growing RAM; \
                        add max_sessions config"
                        .into(),
                },
                ScalabilityAction::EnableFeature {
                    feature: "session_ttl".into(),
                    description: "Auto-archive sessions older than 24h to disk".into(),
                },
            ]
        }
        Subsystem::FactStore => {
            vec![
                ScalabilityAction::AddSizeLimit {
                    suggested_max: 5_000,
                    reason: "Unbounded fact store grows linearly; add max_facts with LRU eviction"
                        .into(),
                },
                ScalabilityAction::Custom {
                    action: "Archive old facts".into(),
                    details: "Persist low-confidence or old facts to disk, keep only recent/high-confidence in memory".into(),
                },
            ]
        }
        Subsystem::ResponseCache => {
            vec![
                ScalabilityAction::EnableFeature {
                    feature: "redis-backend".into(),
                    description: "Switch to Redis-backed cache for shared/persistent caching".into(),
                },
            ]
        }
        Subsystem::EpisodicMemory => {
            vec![ScalabilityAction::Custom {
                action: "Archive old episodes".into(),
                details: "Persist oldest episodes to disk; increase max_episodes if RAM allows"
                    .into(),
            }]
        }
        Subsystem::EntityStore => {
            vec![ScalabilityAction::AddSizeLimit {
                suggested_max: 3_000,
                reason: "Unbounded entity store; add max_entities with merge/eviction strategy"
                    .into(),
            }]
        }
        Subsystem::CrdtOrSetTombstones => {
            vec![
                ScalabilityAction::RunMaintenance {
                    operation: "tombstone_compaction".into(),
                    description: "ORSet tombstones are never garbage-collected. \
                        Implement periodic compaction to remove tombstones \
                        for elements that no longer exist in any replica."
                        .into(),
                },
                ScalabilityAction::Custom {
                    action: "Consider CmRDT".into(),
                    details: "Operation-based CRDTs (CmRDT) avoid tombstone accumulation \
                        by transmitting operations instead of state."
                        .into(),
                },
            ]
        }
        Subsystem::DhtStorage => {
            vec![
                ScalabilityAction::EnableFeature {
                    feature: "dht_ttl_cleanup".into(),
                    description: "Enable background TTL cleanup task to evict expired entries"
                        .into(),
                },
                ScalabilityAction::Custom {
                    action: "Distribute data".into(),
                    details: "Add more DHT nodes to spread storage load via consistent hashing"
                        .into(),
                },
            ]
        }
    }
}

// ============================================================================
// Core check logic
// ============================================================================

thread_local! {
    static LAST_WARNED: RefCell<HashMap<Subsystem, Instant>> = RefCell::new(HashMap::new());
}

/// Cooldown between repeated warnings for the same subsystem (seconds).
const COOLDOWN_SECS: u64 = 60;

/// Build a warning with message and recommendations.
fn build_warning(
    subsystem: Subsystem,
    severity: WarningSeverity,
    current_size: usize,
    optimal_max: usize,
    pct: f64,
) -> ScalabilityWarning {
    let recommendations = build_recommendations(&subsystem, severity);
    let message = format!(
        "{}: {} entries ({:.0}% of {} optimal limit)",
        subsystem,
        current_size,
        pct * 100.0,
        optimal_max
    );
    ScalabilityWarning {
        subsystem,
        severity,
        current_size,
        optimal_max,
        utilization_pct: pct,
        message,
        recommendations,
    }
}

/// Compute effective max from a threshold.
fn effective_max(threshold: &ScalabilityThreshold) -> usize {
    if threshold.unbounded {
        threshold.soft_limit.unwrap_or(usize::MAX)
    } else {
        threshold.optimal_max
    }
}

/// Compute severity from utilization percentage.
fn compute_severity(pct: f64) -> Option<WarningSeverity> {
    if pct >= 0.95 {
        Some(WarningSeverity::Critical)
    } else if pct >= 0.80 {
        Some(WarningSeverity::Warning)
    } else if pct >= 0.60 {
        Some(WarningSeverity::Info)
    } else {
        None
    }
}

/// Check a subsystem's current size and emit a warning if over threshold.
///
/// Uses `thread_local!` cooldown to avoid log spam (max 1 warning per
/// subsystem per 60 seconds).
///
/// Returns the warning if generated, `None` if below threshold or on cooldown.
pub fn check_scalability(subsystem: Subsystem, current_size: usize) -> Option<ScalabilityWarning> {
    let threshold = default_threshold(&subsystem);
    let max = effective_max(&threshold);

    if max == 0 || max == usize::MAX {
        return None;
    }

    let pct = current_size as f64 / max as f64;
    let severity = compute_severity(pct)?;

    // Cooldown: skip if we warned for this subsystem recently
    let should_warn = LAST_WARNED.with(|lw| {
        let mut map = lw.borrow_mut();
        if let Some(last) = map.get(&subsystem) {
            if last.elapsed().as_secs() < COOLDOWN_SECS {
                return false;
            }
        }
        map.insert(subsystem.clone(), Instant::now());
        true
    });

    if !should_warn {
        return None;
    }

    let warning = build_warning(subsystem, severity, current_size, max, pct);

    log::warn!(
        "[scalability] {} — {}",
        warning.message,
        warning
            .recommendations
            .first()
            .map(format_action)
            .unwrap_or_default()
    );

    Some(warning)
}

/// Check without cooldown (for testing and batch audits).
/// Always returns the warning if over threshold, never suppresses.
pub fn check_scalability_no_cooldown(
    subsystem: Subsystem,
    current_size: usize,
) -> Option<ScalabilityWarning> {
    let threshold = default_threshold(&subsystem);
    let max = effective_max(&threshold);

    if max == 0 || max == usize::MAX {
        return None;
    }

    let pct = current_size as f64 / max as f64;
    let severity = compute_severity(pct)?;

    Some(build_warning(subsystem, severity, current_size, max, pct))
}

// ============================================================================
// Snapshot-based batch audit
// ============================================================================

/// A point-in-time snapshot of all monitored subsystem sizes.
///
/// Used by [`audit_snapshot`] for a comprehensive scalability health check.
/// Fields are `Option` — only set if the subsystem is active.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalabilitySnapshot {
    pub vector_db_in_memory_count: Option<usize>,
    pub hnsw_node_count: Option<usize>,
    pub embedding_cache_entries: Option<usize>,
    pub knowledge_graph_entities: Option<usize>,
    pub multi_layer_graph_entities: Option<usize>,
    pub session_count: Option<usize>,
    pub fact_store_count: Option<usize>,
    pub response_cache_entries: Option<usize>,
    pub episodic_memory_episodes: Option<usize>,
    pub entity_store_count: Option<usize>,
    pub crdt_orset_tombstones: Option<usize>,
    pub dht_storage_entries: Option<usize>,
}

/// Run a full audit over a snapshot, returning all warnings.
///
/// Unlike [`check_scalability`], this does NOT apply cooldown — it always
/// reports all issues found in the snapshot.
pub fn audit_snapshot(snapshot: &ScalabilitySnapshot) -> Vec<ScalabilityWarning> {
    let mut warnings = Vec::new();

    let checks: Vec<(Subsystem, Option<usize>)> = vec![
        (Subsystem::VectorDbInMemory, snapshot.vector_db_in_memory_count),
        (Subsystem::VectorDbHnsw, snapshot.hnsw_node_count),
        (Subsystem::EmbeddingCache, snapshot.embedding_cache_entries),
        (Subsystem::KnowledgeGraph, snapshot.knowledge_graph_entities),
        (Subsystem::MultiLayerGraph, snapshot.multi_layer_graph_entities),
        (Subsystem::SessionStore, snapshot.session_count),
        (Subsystem::FactStore, snapshot.fact_store_count),
        (Subsystem::ResponseCache, snapshot.response_cache_entries),
        (Subsystem::EpisodicMemory, snapshot.episodic_memory_episodes),
        (Subsystem::EntityStore, snapshot.entity_store_count),
        (Subsystem::CrdtOrSetTombstones, snapshot.crdt_orset_tombstones),
        (Subsystem::DhtStorage, snapshot.dht_storage_entries),
    ];

    for (subsystem, count) in checks {
        if let Some(size) = count {
            if let Some(warning) = check_scalability_no_cooldown(subsystem, size) {
                warnings.push(warning);
            }
        }
    }

    warnings
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_warning_below_60pct() {
        // VectorDbInMemory optimal_max = 10,000 → 50% = 5,000
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 5_000);
        assert!(result.is_none());
    }

    #[test]
    fn test_info_at_60pct() {
        // VectorDbInMemory optimal_max = 10,000 → 60% = 6,000
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 6_000);
        assert!(result.is_some());
        assert_eq!(result.unwrap().severity, WarningSeverity::Info);
    }

    #[test]
    fn test_warning_at_80pct() {
        // VectorDbInMemory optimal_max = 10,000 → 80% = 8,000
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 8_000);
        assert!(result.is_some());
        assert_eq!(result.unwrap().severity, WarningSeverity::Warning);
    }

    #[test]
    fn test_critical_at_95pct() {
        // VectorDbInMemory optimal_max = 10,000 → 95% = 9,500
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 9_500);
        assert!(result.is_some());
        assert_eq!(result.unwrap().severity, WarningSeverity::Critical);
    }

    #[test]
    fn test_over_100pct_is_critical() {
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 15_000);
        assert!(result.is_some());
        let w = result.unwrap();
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert!(w.utilization_pct > 1.0);
    }

    #[test]
    fn test_cooldown_suppresses_repeat() {
        // First call should succeed
        let r1 = check_scalability(Subsystem::EmbeddingCache, 8_000);
        assert!(r1.is_some());

        // Second call within 60s should be suppressed
        let r2 = check_scalability(Subsystem::EmbeddingCache, 8_500);
        assert!(r2.is_none());
    }

    #[test]
    fn test_unbounded_soft_limit() {
        // SessionStore is unbounded with soft_limit = 1,000
        // At 800 (80%) should trigger Warning
        let result = check_scalability_no_cooldown(Subsystem::SessionStore, 800);
        assert!(result.is_some());
        assert_eq!(result.unwrap().severity, WarningSeverity::Warning);
    }

    #[test]
    fn test_unbounded_below_soft_limit() {
        // SessionStore soft_limit = 1,000 → 40% = 400, below threshold
        let result = check_scalability_no_cooldown(Subsystem::SessionStore, 400);
        assert!(result.is_none());
    }

    #[test]
    fn test_vector_db_recommendation_lancedb() {
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 8_000).unwrap();
        assert!(!result.recommendations.is_empty());
        match &result.recommendations[0] {
            ScalabilityAction::SwitchBackend { to, .. } => {
                assert!(to.contains("LanceDB"));
            }
            other => panic!("Expected SwitchBackend, got {:?}", other),
        }
    }

    #[test]
    fn test_vector_db_recommendation_qdrant_at_critical() {
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 9_800).unwrap();
        assert!(result.recommendations.len() >= 2);
        match &result.recommendations[1] {
            ScalabilityAction::SwitchBackend { to, .. } => {
                assert!(to.contains("Qdrant"));
            }
            other => panic!("Expected SwitchBackend to Qdrant, got {:?}", other),
        }
    }

    #[test]
    fn test_session_store_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::SessionStore, 900).unwrap();
        assert!(!result.recommendations.is_empty());
        match &result.recommendations[0] {
            ScalabilityAction::AddSizeLimit { .. } => {}
            other => panic!("Expected AddSizeLimit, got {:?}", other),
        }
    }

    #[test]
    fn test_orset_tombstone_recommendation() {
        let result =
            check_scalability_no_cooldown(Subsystem::CrdtOrSetTombstones, 9_000).unwrap();
        match &result.recommendations[0] {
            ScalabilityAction::RunMaintenance { operation, .. } => {
                assert!(operation.contains("compaction"));
            }
            other => panic!("Expected RunMaintenance, got {:?}", other),
        }
    }

    #[test]
    fn test_knowledge_graph_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::KnowledgeGraph, 85_000).unwrap();
        match &result.recommendations[0] {
            ScalabilityAction::ReduceConfig { field, .. } => {
                assert!(field.contains("traversal_depth"));
            }
            other => panic!("Expected ReduceConfig, got {:?}", other),
        }
    }

    #[test]
    fn test_fact_store_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::FactStore, 8_000).unwrap();
        match &result.recommendations[0] {
            ScalabilityAction::AddSizeLimit { .. } => {}
            other => panic!("Expected AddSizeLimit, got {:?}", other),
        }
    }

    #[test]
    fn test_dht_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::DhtStorage, 40_000).unwrap();
        match &result.recommendations[0] {
            ScalabilityAction::EnableFeature { feature, .. } => {
                assert!(feature.contains("ttl"));
            }
            other => panic!("Expected EnableFeature, got {:?}", other),
        }
    }

    #[test]
    fn test_severity_ordering() {
        assert!(WarningSeverity::Info < WarningSeverity::Warning);
        assert!(WarningSeverity::Warning < WarningSeverity::Critical);
        assert!(WarningSeverity::Info < WarningSeverity::Critical);
    }

    #[test]
    fn test_warning_serialization() {
        let warning = ScalabilityWarning {
            subsystem: Subsystem::VectorDbInMemory,
            severity: WarningSeverity::Warning,
            current_size: 8_000,
            optimal_max: 10_000,
            utilization_pct: 0.8,
            message: "test warning".into(),
            recommendations: vec![ScalabilityAction::SwitchBackend {
                from: "InMemory".into(),
                to: "LanceDB".into(),
                reason: "test".into(),
            }],
        };
        let json = serde_json::to_string(&warning).unwrap();
        let parsed: ScalabilityWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.subsystem, Subsystem::VectorDbInMemory);
        assert_eq!(parsed.severity, WarningSeverity::Warning);
        assert_eq!(parsed.current_size, 8_000);
    }

    #[test]
    fn test_snapshot_audit_multiple_warnings() {
        let snapshot = ScalabilitySnapshot {
            vector_db_in_memory_count: Some(9_000),
            session_count: Some(900),
            crdt_orset_tombstones: Some(9_500),
            ..Default::default()
        };
        let warnings = audit_snapshot(&snapshot);
        assert_eq!(warnings.len(), 3);
    }

    #[test]
    fn test_snapshot_audit_all_healthy() {
        let snapshot = ScalabilitySnapshot {
            vector_db_in_memory_count: Some(100),
            session_count: Some(10),
            fact_store_count: Some(50),
            ..Default::default()
        };
        let warnings = audit_snapshot(&snapshot);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_zero_size_no_panic() {
        let result = check_scalability_no_cooldown(Subsystem::VectorDbInMemory, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_format_action_all_variants() {
        let actions = vec![
            ScalabilityAction::SwitchBackend {
                from: "A".into(),
                to: "B".into(),
                reason: "test".into(),
            },
            ScalabilityAction::EnableFeature {
                feature: "x".into(),
                description: "y".into(),
            },
            ScalabilityAction::AddSizeLimit {
                suggested_max: 100,
                reason: "test".into(),
            },
            ScalabilityAction::RunMaintenance {
                operation: "gc".into(),
                description: "test".into(),
            },
            ScalabilityAction::ReduceConfig {
                field: "depth".into(),
                current: "3".into(),
                suggested: "1".into(),
                reason: "test".into(),
            },
            ScalabilityAction::Custom {
                action: "do".into(),
                details: "something".into(),
            },
        ];
        for action in &actions {
            let s = format_action(action);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_subsystem_display() {
        assert_eq!(Subsystem::VectorDbInMemory.to_string(), "VectorDb (InMemory)");
        assert_eq!(Subsystem::CrdtOrSetTombstones.to_string(), "ORSet Tombstones");
        assert_eq!(Subsystem::DhtStorage.to_string(), "DHT Storage");
    }

    #[test]
    fn test_snapshot_serialization() {
        let snapshot = ScalabilitySnapshot {
            vector_db_in_memory_count: Some(5_000),
            session_count: Some(100),
            ..Default::default()
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: ScalabilitySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.vector_db_in_memory_count, Some(5_000));
        assert_eq!(parsed.session_count, Some(100));
        assert!(parsed.hnsw_node_count.is_none());
    }

    #[test]
    fn test_snapshot_audit_partial() {
        // Only some fields set, others None — should only check set fields
        let snapshot = ScalabilitySnapshot {
            hnsw_node_count: Some(85_000),
            ..Default::default()
        };
        let warnings = audit_snapshot(&snapshot);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].subsystem, Subsystem::VectorDbHnsw);
        assert_eq!(warnings[0].severity, WarningSeverity::Warning);
    }

    #[test]
    fn test_hnsw_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::VectorDbHnsw, 95_000).unwrap();
        assert!(result.recommendations.len() >= 2);
        match &result.recommendations[0] {
            ScalabilityAction::RunMaintenance { operation, .. } => {
                assert!(operation.contains("compaction"));
            }
            other => panic!("Expected RunMaintenance, got {:?}", other),
        }
    }

    #[test]
    fn test_episodic_memory_recommendation() {
        let result = check_scalability_no_cooldown(Subsystem::EpisodicMemory, 800).unwrap();
        match &result.recommendations[0] {
            ScalabilityAction::Custom { action, .. } => {
                assert!(action.contains("Archive"));
            }
            other => panic!("Expected Custom archive recommendation, got {:?}", other),
        }
    }
}
