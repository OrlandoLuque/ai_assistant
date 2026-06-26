//! Advanced routing system for intelligent model selection.
//!
//! Provides multi-armed bandit routing (Thompson Sampling, UCB1, epsilon-greedy),
//! NFA/DFA graph-based routing, hierarchical routing DAGs, ensemble voting,
//! adaptive per-query routing, eval-to-runtime feedback, distributed bandit
//! training, and export/import of bandit state.

pub use crate::error::AdvancedRoutingError;
use serde::{Deserialize, Serialize};

mod automata;
mod bandit;
mod bootstrap;
mod contextual;
mod distributed;
mod ensemble;
mod hierarchical;
mod mcp_tools;
mod pipeline;

pub use automata::*;
pub use bandit::*;
#[cfg(feature = "eval-suite")]
pub use bootstrap::*;
pub use contextual::*;
pub use distributed::*;
pub use ensemble::*;
pub use hierarchical::*;
pub use mcp_tools::*;
pub use pipeline::*;

// =============================================================================
// SHARED TYPES
// =============================================================================

/// Unique identifier for a bandit arm (typically a model ID).
pub type ArmId = String;

/// Features extracted from a query for routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Approximate token count
    pub token_count: usize,
    /// Number of sentences
    pub sentence_count: usize,
    /// Detected domain (e.g., "coding", "math", "creative", "general")
    pub domain: String,
    /// Complexity score 0.0..1.0
    pub complexity: f64,
    /// Entity count (names, numbers, code blocks)
    pub entity_count: usize,
    /// Whether the query contains code
    pub has_code: bool,
    /// Whether the query asks a question
    pub is_question: bool,
    /// Average word length (proxy for vocabulary complexity)
    pub avg_word_length: f64,
    /// Raw feature vector for ML-style routing
    pub feature_vector: Vec<f64>,
}

/// Result of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOutcome {
    /// Selected model/arm
    pub selected_arm: ArmId,
    /// Confidence in the selection (0.0..1.0)
    pub confidence: f64,
    /// Reason for the selection
    pub reason: String,
    /// Alternative arms considered (ranked by score descending)
    pub alternatives: Vec<(ArmId, f64)>,
    /// Which router made the decision
    pub router_id: String,
    /// Time taken for the routing decision in microseconds
    pub decision_time_us: u64,
}

/// Outcome feedback after a model invocation completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmFeedback {
    /// Which arm was used
    pub arm_id: ArmId,
    /// Whether the invocation was successful
    pub success: bool,
    /// Quality score if available (0.0..1.0)
    pub quality: Option<f64>,
    /// Latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Cost incurred
    pub cost: Option<f64>,
    /// Task type context
    pub task_type: Option<String>,
}

#[cfg(test)]
pub(crate) fn test_features(domain: &str, complexity: f64) -> QueryFeatures {
    QueryFeatures {
        token_count: 50,
        sentence_count: 3,
        domain: domain.to_string(),
        complexity,
        entity_count: 2,
        has_code: false,
        is_question: true,
        avg_word_length: 5.0,
        feature_vector: vec![50.0, 3.0, complexity, 2.0, 0.0, 1.0, 5.0],
    }
}

#[cfg(test)]
pub(crate) fn test_features_code() -> QueryFeatures {
    QueryFeatures {
        token_count: 100,
        sentence_count: 5,
        domain: "coding".to_string(),
        complexity: 0.8,
        entity_count: 5,
        has_code: true,
        is_question: false,
        avg_word_length: 6.0,
        feature_vector: vec![100.0, 5.0, 0.8, 5.0, 1.0, 0.0, 6.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ERROR TESTS
    // =========================================================================

    #[test]
    fn test_error_display() {
        let e = AdvancedRoutingError::ArmNotFound {
            arm_id: "test".to_string(),
        };
        assert!(format!("{}", e).contains("test"));

        let e = AdvancedRoutingError::CycleDetected;
        assert!(format!("{}", e).contains("Cycle"));

        let e = AdvancedRoutingError::EmptyEnsemble;
        assert!(format!("{}", e).contains("no sub-routers"));
    }

    #[test]
    fn test_error_suggestion() {
        let e = AdvancedRoutingError::ArmNotFound {
            arm_id: "x".to_string(),
        };
        assert!(e.suggestion().is_some());

        let e = AdvancedRoutingError::CycleDetected;
        assert!(e.suggestion().unwrap().contains("acyclic"));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(AdvancedRoutingError::NoRoutingPath {
            query: "q".to_string(),
            reason: "r".to_string()
        }
        .is_recoverable());
        assert!(AdvancedRoutingError::ArmNotFound {
            arm_id: "a".to_string()
        }
        .is_recoverable());
        assert!(!AdvancedRoutingError::CycleDetected.is_recoverable());
        assert!(!AdvancedRoutingError::EmptyEnsemble.is_recoverable());
    }

    #[test]
    fn test_error_from_conversion() {
        let e: crate::error::AiError = AdvancedRoutingError::CycleDetected.into();
        assert_eq!(e.code(), "ADVANCED_ROUTING");
    }

    #[test]
    fn test_error_code() {
        let e = crate::error::AiError::AdvancedRouting(AdvancedRoutingError::EmptyEnsemble);
        assert_eq!(e.code(), "ADVANCED_ROUTING");
        assert!(e.suggestion().is_some());
    }
}
