//! Ensemble router combining multiple sub-routers via voting strategies.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// ENSEMBLE ROUTER WITH VOTING
// =============================================================================

/// Strategy for combining votes from multiple sub-routers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum EnsembleStrategy {
    /// Simple majority: most-voted arm wins
    MajorityVote,
    /// Weighted average of confidence scores per arm
    WeightedAverage,
    /// All sub-routers must agree on the same arm
    Unanimous,
    /// Highest individual confidence wins
    MaxConfidence,
}

/// A vote from a single sub-router.
#[derive(Debug, Clone)]
pub struct SubRouterVote {
    pub router_id: String,
    pub outcome: RoutingOutcome,
    pub weight: f64,
}

/// Trait for any component that can participate in ensemble voting.
pub trait RoutingVoter: std::fmt::Debug + Send + Sync {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError>;
    fn router_id(&self) -> &str;
    fn record_outcome(&mut self, feedback: &ArmFeedback);
}

impl RoutingVoter for BanditRouter {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.select(Some(&features.domain))
    }
    fn router_id(&self) -> &str {
        "bandit"
    }
    fn record_outcome(&mut self, feedback: &ArmFeedback) {
        BanditRouter::record_outcome(self, feedback);
    }
}

impl RoutingVoter for AdaptivePerQueryRouter {
    fn vote(&mut self, features: &QueryFeatures) -> Result<RoutingOutcome, AdvancedRoutingError> {
        self.route_with_features(features)
    }
    fn router_id(&self) -> &str {
        "adaptive"
    }
    fn record_outcome(&mut self, feedback: &ArmFeedback) {
        // No raw query available, record for the task_type domain if present
        if let Some(ref domain) = feedback.task_type {
            if let Some(bandit) = self.domain_bandits.get_mut(domain) {
                bandit.record_outcome(feedback);
            }
        }
    }
}

/// Ensemble router that combines multiple sub-routers via voting.
pub struct EnsembleRouter {
    sub_routers: Vec<(Box<dyn RoutingVoter>, f64)>,
    strategy: EnsembleStrategy,
    id: String,
}

impl std::fmt::Debug for EnsembleRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleRouter")
            .field("strategy", &self.strategy)
            .field("id", &self.id)
            .field("voter_count", &self.sub_routers.len())
            .finish()
    }
}

impl EnsembleRouter {
    pub fn new(strategy: EnsembleStrategy) -> Self {
        Self {
            sub_routers: Vec::new(),
            strategy,
            id: "ensemble".to_string(),
        }
    }

    pub fn add_voter(&mut self, voter: Box<dyn RoutingVoter>, weight: f64) {
        self.sub_routers.push((voter, weight));
    }

    pub fn voter_count(&self) -> usize {
        self.sub_routers.len()
    }

    /// Route by collecting votes and tallying.
    pub fn route(
        &mut self,
        features: &QueryFeatures,
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        if self.sub_routers.is_empty() {
            return Err(AdvancedRoutingError::EmptyEnsemble);
        }

        let start = std::time::Instant::now();

        let mut votes = Vec::new();
        for (router, weight) in &mut self.sub_routers {
            if let Ok(outcome) = router.vote(features) {
                votes.push(SubRouterVote {
                    router_id: router.router_id().to_string(),
                    outcome,
                    weight: *weight,
                });
            }
        }

        if votes.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: features.domain.clone(),
                reason: "All sub-routers failed".to_string(),
            });
        }

        let mut result = self.tally_votes(&votes)?;
        result.decision_time_us = start.elapsed().as_micros() as u64;
        result.router_id = self.id.clone();
        Ok(result)
    }

    /// Propagate outcome feedback to all sub-routers.
    pub fn record_outcome(&mut self, feedback: &ArmFeedback) {
        for (router, _) in &mut self.sub_routers {
            router.record_outcome(feedback);
        }
    }

    fn tally_votes(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        // Defensive: every tally strategy below assumes at least one vote.
        // `route()` already guards this, but guarding here too makes the
        // private tally helpers panic-free regardless of caller.
        if votes.is_empty() {
            return Err(AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "no votes to tally".to_string(),
            });
        }
        match self.strategy {
            EnsembleStrategy::MajorityVote => self.majority_vote(votes),
            EnsembleStrategy::WeightedAverage => self.weighted_average(votes),
            EnsembleStrategy::Unanimous => self.unanimous(votes),
            EnsembleStrategy::MaxConfidence => self.max_confidence(votes),
        }
    }

    fn majority_vote(
        &self,
        votes: &[SubRouterVote],
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let mut counts: HashMap<&str, (usize, f64)> = HashMap::new(); // (count, max_confidence)
        for vote in votes {
            let entry = counts.entry(&vote.outcome.selected_arm).or_insert((0, 0.0));
            entry.0 += 1;
            if vote.outcome.confidence > entry.1 {
                entry.1 = vote.outcome.confidence;
            }
        }

        let winner = counts
            .iter()
            .max_by(|a, b| {
                a.1 .0.cmp(&b.1 .0).then(
                    a.1 .1
                        .partial_cmp(&b.1 .1)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            })
            .map(|(arm, (count, conf))| (arm.to_string(), *count, *conf))
            .ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "majority_vote: no votes to tally".to_string(),
            })?;

        let alternatives: Vec<(ArmId, f64)> = counts
            .iter()
            .filter(|(arm, _)| **arm != winner.0)
            .map(|(arm, (_, conf))| (arm.to_string(), *conf))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: winner.0,
            confidence: winner.2,
            reason: format!("Majority vote: {}/{} votes", winner.1, votes.len()),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }

    fn weighted_average(
        &self,
        votes: &[SubRouterVote],
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let mut scores: HashMap<&str, f64> = HashMap::new();
        for vote in votes {
            *scores.entry(&vote.outcome.selected_arm).or_insert(0.0) +=
                vote.weight * vote.outcome.confidence;
        }

        let winner = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, score)| (arm.to_string(), *score))
            .ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "weighted_average: no votes to tally".to_string(),
            })?;

        let total_weight: f64 = scores.values().sum();
        let confidence = if total_weight > 0.0 {
            winner.1 / total_weight
        } else {
            0.5
        };

        let alternatives: Vec<(ArmId, f64)> = scores
            .iter()
            .filter(|(arm, _)| **arm != winner.0)
            .map(|(arm, score)| (arm.to_string(), *score))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: winner.0,
            confidence,
            reason: format!("Weighted score: {:.3}", winner.1),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }

    fn unanimous(&self, votes: &[SubRouterVote]) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let first_arm = &votes
            .first()
            .ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "unanimous: no votes to tally".to_string(),
            })?
            .outcome
            .selected_arm;
        if votes.iter().all(|v| v.outcome.selected_arm == *first_arm) {
            let max_conf = votes
                .iter()
                .map(|v| v.outcome.confidence)
                .fold(0.0f64, f64::max);
            Ok(RoutingOutcome {
                selected_arm: first_arm.clone(),
                confidence: max_conf,
                reason: format!("Unanimous: all {} routers agree", votes.len()),
                alternatives: Vec::new(),
                router_id: "ensemble".to_string(),
                decision_time_us: 0,
            })
        } else {
            Err(AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "No unanimous agreement among sub-routers".to_string(),
            })
        }
    }

    fn max_confidence(
        &self,
        votes: &[SubRouterVote],
    ) -> Result<RoutingOutcome, AdvancedRoutingError> {
        let best = votes
            .iter()
            .max_by(|a, b| {
                a.outcome
                    .confidence
                    .partial_cmp(&b.outcome.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| AdvancedRoutingError::NoRoutingPath {
                query: "ensemble".to_string(),
                reason: "max_confidence: no votes to tally".to_string(),
            })?;

        let alternatives: Vec<(ArmId, f64)> = votes
            .iter()
            .filter(|v| v.router_id != best.router_id)
            .map(|v| (v.outcome.selected_arm.clone(), v.outcome.confidence))
            .collect();

        Ok(RoutingOutcome {
            selected_arm: best.outcome.selected_arm.clone(),
            confidence: best.outcome.confidence,
            reason: format!("Max confidence from router '{}'", best.router_id),
            alternatives,
            router_id: "ensemble".to_string(),
            decision_time_us: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ENSEMBLE ROUTER TESTS
    // =========================================================================

    #[test]
    fn test_ensemble_majority_vote() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        // Add 3 bandits that will vote for the same arm
        for i in 0..3 {
            let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42 + i);
            bandit.add_arm("winner");
            ensemble.add_voter(Box::new(bandit), 1.0);
        }

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "winner");
    }

    #[test]
    fn test_ensemble_weighted_average() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::WeightedAverage);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("model-a");
        let mut b2 = BanditRouter::with_seed(BanditConfig::default(), 99);
        b2.add_arm("model-a");

        ensemble.add_voter(Box::new(b1), 10.0);
        ensemble.add_voter(Box::new(b2), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_ensemble_unanimous_success() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::Unanimous);
        for i in 0..3 {
            let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42 + i);
            bandit.add_arm("same-model");
            ensemble.add_voter(Box::new(bandit), 1.0);
        }

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "same-model");
    }

    #[test]
    fn test_ensemble_unanimous_failure() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::Unanimous);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("model-a");
        let mut b2 = BanditRouter::with_seed(BanditConfig::default(), 99);
        b2.add_arm("model-b"); // Different arm!
        ensemble.add_voter(Box::new(b1), 1.0);
        ensemble.add_voter(Box::new(b2), 1.0);

        let features = test_features("general", 0.5);
        assert!(ensemble.route(&features).is_err());
    }

    #[test]
    fn test_ensemble_max_confidence() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MaxConfidence);
        let mut b1 = BanditRouter::with_seed(BanditConfig::default(), 42);
        b1.add_arm("a");
        ensemble.add_voter(Box::new(b1), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }

    #[test]
    fn test_ensemble_empty_error() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let features = test_features("general", 0.5);
        assert!(ensemble.route(&features).is_err());
    }

    #[test]
    fn test_tally_votes_empty_is_error_not_panic() {
        // Regression (V161): every tally strategy used to `.unwrap()` the
        // max_by result (or index `votes[0]`) and would panic on an empty
        // slice. They must now return Err instead of panicking, for every
        // strategy, even though `route()` already guards the public path.
        for strategy in [
            EnsembleStrategy::MajorityVote,
            EnsembleStrategy::WeightedAverage,
            EnsembleStrategy::Unanimous,
            EnsembleStrategy::MaxConfidence,
        ] {
            let ensemble = EnsembleRouter::new(strategy);
            assert!(
                ensemble.tally_votes(&[]).is_err(),
                "empty votes under {strategy:?} must be Err, not a panic"
            );
        }
    }

    #[test]
    fn test_ensemble_single_router() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("only");
        ensemble.add_voter(Box::new(bandit), 1.0);

        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert_eq!(result.selected_arm, "only");
    }

    #[test]
    fn test_ensemble_record_outcome_propagates() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);
        let mut b = BanditRouter::new(BanditConfig::default());
        b.add_arm("test");
        ensemble.add_voter(Box::new(b), 1.0);

        ensemble.record_outcome(&ArmFeedback {
            arm_id: "test".to_string(),
            success: true,
            quality: Some(0.9),
            latency_ms: None,
            cost: None,
            task_type: None,
        });
        // Should not panic
        assert_eq!(ensemble.voter_count(), 1);
    }

    #[test]
    fn test_ensemble_routing_voter_impl() {
        // BanditRouter implements RoutingVoter
        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("x");
        let voter: &mut dyn RoutingVoter = &mut bandit;
        let features = test_features("general", 0.5);
        let result = voter.vote(&features);
        assert!(result.is_ok());
        assert_eq!(voter.router_id(), "bandit");
    }

    #[test]
    fn test_ensemble_mixed_types() {
        let mut ensemble = EnsembleRouter::new(EnsembleStrategy::MajorityVote);

        let mut bandit = BanditRouter::with_seed(BanditConfig::default(), 42);
        bandit.add_arm("shared");
        ensemble.add_voter(Box::new(bandit), 1.0);

        let adaptive = AdaptivePerQueryRouter::new("shared", BanditConfig::default());
        ensemble.add_voter(Box::new(adaptive), 1.0);

        assert_eq!(ensemble.voter_count(), 2);
        let features = test_features("general", 0.5);
        let result = ensemble.route(&features).unwrap();
        assert!(!result.selected_arm.is_empty());
    }
}
