//! `RewardPolicy` — how raw signals map to a scalar reward in `[0, 1]`.
//!
//! v1 uses fixed weights (per the plan — adaptive weights are deferred to
//! V97+). Signals are:
//! - `success`: task completed, user did not correct — in `[0, 1]`
//! - `latency_norm`: normalized (smaller is better), in `[0, 1]`
//! - `faithfulness`: grounding score from `anti_hallucination`, in `[0, 1]`
//! - `user_feedback`: explicit thumbs, in `[-1, +1]` — mapped to `[0, 1]`
//!
//! Formula:
//!     r = w_s·success + w_l·(1-latency_norm) + w_f·faithfulness + w_u·(user+1)/2
//!
//! Weights sum to 1. Defaults: success 0.5, latency 0.1, faithfulness 0.25,
//! user 0.15. These lean on success but give faithfulness serious weight so
//! ungrounded answers do not win the bandit.

use serde::{Deserialize, Serialize};

/// Raw inputs from a single trajectory. Missing fields are filled with
/// neutral defaults (success=0, latency_norm=1 (worst), faithfulness=0,
/// user=0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardSignal {
    pub success: f32,
    pub latency_norm: f32,
    pub faithfulness: f32,
    pub user_feedback: f32,
}

impl Default for RewardSignal {
    fn default() -> Self {
        Self {
            success: 0.0,
            latency_norm: 1.0,
            faithfulness: 0.0,
            user_feedback: 0.0,
        }
    }
}

/// Configuration for `RewardPolicy`. Each weight is a non-negative scalar.
/// They are normalized at construction so the output stays in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct RewardPolicyConfig {
    pub w_success: f32,
    pub w_latency: f32,
    pub w_faithfulness: f32,
    pub w_user: f32,
}

impl Default for RewardPolicyConfig {
    fn default() -> Self {
        Self {
            w_success: 0.5,
            w_latency: 0.1,
            w_faithfulness: 0.25,
            w_user: 0.15,
        }
    }
}

/// Fixed-weight reward policy. Stateless — safe to share.
#[derive(Debug, Clone)]
pub struct RewardPolicy {
    w: [f32; 4],
}

impl RewardPolicy {
    pub fn new(cfg: RewardPolicyConfig) -> Self {
        let raw = [
            cfg.w_success.max(0.0),
            cfg.w_latency.max(0.0),
            cfg.w_faithfulness.max(0.0),
            cfg.w_user.max(0.0),
        ];
        let sum: f32 = raw.iter().sum();
        let w = if sum > f32::EPSILON {
            [raw[0] / sum, raw[1] / sum, raw[2] / sum, raw[3] / sum]
        } else {
            [0.25, 0.25, 0.25, 0.25]
        };
        Self { w }
    }

    pub fn weights(&self) -> [f32; 4] {
        self.w
    }

    /// Map a raw signal to a scalar reward in `[0, 1]`. Input components are
    /// clamped — callers do not need to pre-validate.
    pub fn score(&self, signal: &RewardSignal) -> f32 {
        let success = signal.success.clamp(0.0, 1.0);
        let lat = signal.latency_norm.clamp(0.0, 1.0);
        let faith = signal.faithfulness.clamp(0.0, 1.0);
        let user = ((signal.user_feedback.clamp(-1.0, 1.0)) + 1.0) * 0.5;
        let r =
            self.w[0] * success + self.w[1] * (1.0 - lat) + self.w[2] * faith + self.w[3] * user;
        r.clamp(0.0, 1.0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_sum_to_one() {
        let p = RewardPolicy::new(RewardPolicyConfig::default());
        let sum: f32 = p.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn all_zero_signal_scores_nonzero_for_latency_bonus() {
        // Default signal has latency_norm=1 (worst), user=0 → (user+1)/2=0.5.
        // Reward = 0.5*0 + 0.1*0 + 0.25*0 + 0.15*0.5 = 0.075.
        let p = RewardPolicy::new(RewardPolicyConfig::default());
        let r = p.score(&RewardSignal::default());
        assert!((r - 0.075).abs() < 1e-4, "got {r}");
    }

    #[test]
    fn perfect_signal_scores_one() {
        let p = RewardPolicy::new(RewardPolicyConfig::default());
        let r = p.score(&RewardSignal {
            success: 1.0,
            latency_norm: 0.0,
            faithfulness: 1.0,
            user_feedback: 1.0,
        });
        assert!((r - 1.0).abs() < 1e-5, "got {r}");
    }

    #[test]
    fn clamps_out_of_range_inputs() {
        let p = RewardPolicy::new(RewardPolicyConfig::default());
        let r = p.score(&RewardSignal {
            success: 5.0,
            latency_norm: -1.0,
            faithfulness: 9.0,
            user_feedback: 99.0,
        });
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_weights_fall_back_to_uniform() {
        let p = RewardPolicy::new(RewardPolicyConfig {
            w_success: 0.0,
            w_latency: 0.0,
            w_faithfulness: 0.0,
            w_user: 0.0,
        });
        let w = p.weights();
        assert!(w.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn negative_weights_treated_as_zero() {
        let p = RewardPolicy::new(RewardPolicyConfig {
            w_success: 1.0,
            w_latency: -2.0,
            w_faithfulness: 1.0,
            w_user: -5.0,
        });
        let w = p.weights();
        // Sum of non-negative = 2.0; normalized: 0.5, 0, 0.5, 0.
        assert!((w[0] - 0.5).abs() < 1e-5);
        assert_eq!(w[1], 0.0);
        assert!((w[2] - 0.5).abs() < 1e-5);
        assert_eq!(w[3], 0.0);
    }
}
