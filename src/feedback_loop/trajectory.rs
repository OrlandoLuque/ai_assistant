//! `TrajectoryRecord` — one execution trace with reward components.
//!
//! Produced by `butler` at end-of-run and consumed by the `FeedbackDispatcher`.
//! All reward fields are optional because early-stopped runs may only have
//! partial signal. A `TrajectoryRecord` is immutable once built.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Ids + enums
// =============================================================================

/// Globally unique trajectory id. Backed by a UUID so concurrent principals
/// never collide. Newtype so callers can't accidentally mix with other ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrajectoryId(String);

impl TrajectoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
    pub fn from_raw(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TrajectoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TrajectoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Coarse outcome of the run. Derived by the caller — the dispatcher does
/// not infer it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Outcome {
    Success,
    Failure,
    Abandoned,
    Unknown,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => f.write_str("success"),
            Self::Failure => f.write_str("failure"),
            Self::Abandoned => f.write_str("abandoned"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// Privacy tier of the trajectory. Controls which sinks receive the record
/// (e.g. `Confidential` is never sent to the dataset writer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrivacyTier {
    /// No restrictions beyond normal ACL.
    Public,
    /// Internal team scope — skip external dataset export.
    Internal,
    /// Never leaves the node; dispatcher drops everything except ledger entry.
    Confidential,
}

impl std::fmt::Display for PrivacyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public => f.write_str("public"),
            Self::Internal => f.write_str("internal"),
            Self::Confidential => f.write_str("confidential"),
        }
    }
}

// =============================================================================
// Reward components
// =============================================================================

/// Granular reward signals. Each field is optional — missing signal is
/// preserved instead of defaulted so downstream consumers (like the
/// Fragment bandit) can treat it differently from an explicit zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RewardComponents {
    pub success: Option<f32>,
    pub latency_norm: Option<f32>,
    pub faithfulness: Option<f32>,
    pub user_feedback: Option<f32>,
}

impl RewardComponents {
    /// Count of non-`None` fields. Used by `minimum_sources` gates.
    pub fn source_count(&self) -> usize {
        let mut n = 0;
        if self.success.is_some() {
            n += 1;
        }
        if self.latency_norm.is_some() {
            n += 1;
        }
        if self.faithfulness.is_some() {
            n += 1;
        }
        if self.user_feedback.is_some() {
            n += 1;
        }
        n
    }

    /// Fill missing fields with neutral defaults: success=0, latency=1 (worst),
    /// faithfulness=0, user=0. Matches `prompt_synthesis::RewardSignal`.
    pub fn fill_neutral(&self) -> (f32, f32, f32, f32) {
        (
            self.success.unwrap_or(0.0),
            self.latency_norm.unwrap_or(1.0),
            self.faithfulness.unwrap_or(0.0),
            self.user_feedback.unwrap_or(0.0),
        )
    }
}

// =============================================================================
// TrajectoryRecord
// =============================================================================

/// One execution trace. Built by the caller, frozen once dispatched. The
/// dispatcher may redact fields in place before forwarding to sinks but the
/// ledger entry uses the redacted form.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrajectoryRecord {
    pub id: TrajectoryId,
    pub principal: String,
    pub timestamp: DateTime<Utc>,
    pub fragment_arm_id: Option<String>,
    pub skill_ids_used: Vec<String>,
    pub strategy_id: Option<String>,
    pub intent_cluster: Option<u32>,
    pub outcome: Outcome,
    pub reward: RewardComponents,
    pub privacy_tier: PrivacyTier,
    pub steps: u32,
    pub notes: String,
}

impl TrajectoryRecord {
    /// Convenience constructor with sane defaults; callers fill what they have.
    pub fn new(principal: impl Into<String>) -> Self {
        Self {
            id: TrajectoryId::new(),
            principal: principal.into(),
            timestamp: Utc::now(),
            fragment_arm_id: None,
            skill_ids_used: Vec::new(),
            strategy_id: None,
            intent_cluster: None,
            outcome: Outcome::Unknown,
            reward: RewardComponents::default(),
            privacy_tier: PrivacyTier::Internal,
            steps: 0,
            notes: String::new(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_ids_are_unique() {
        let a = TrajectoryId::new();
        let b = TrajectoryId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn reward_source_count_matches_filled_fields() {
        let r = RewardComponents {
            success: Some(1.0),
            latency_norm: None,
            faithfulness: Some(0.5),
            user_feedback: None,
        };
        assert_eq!(r.source_count(), 2);
    }

    #[test]
    fn reward_fill_neutral_uses_worst_latency() {
        let r = RewardComponents::default();
        let (s, l, f, u) = r.fill_neutral();
        assert_eq!(s, 0.0);
        assert_eq!(l, 1.0);
        assert_eq!(f, 0.0);
        assert_eq!(u, 0.0);
    }

    #[test]
    fn outcome_display_stable() {
        assert_eq!(Outcome::Success.to_string(), "success");
        assert_eq!(Outcome::Abandoned.to_string(), "abandoned");
    }

    #[test]
    fn privacy_tier_display_stable() {
        assert_eq!(PrivacyTier::Public.to_string(), "public");
        assert_eq!(PrivacyTier::Confidential.to_string(), "confidential");
    }

    #[test]
    fn new_trajectory_has_fresh_id_and_unknown_outcome() {
        let t = TrajectoryRecord::new("alice");
        assert!(!t.id.as_str().is_empty());
        assert_eq!(t.principal, "alice");
        assert_eq!(t.outcome, Outcome::Unknown);
    }
}
