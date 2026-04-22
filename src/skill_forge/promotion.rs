//! Promotion pipeline — 6 gates + 1 transversal security gate that govern
//! the `Exploring → Exploited` transition for a skill.
//!
//! Gates are evaluated in order:
//! 1. **Integrity** — content hash and Ed25519 signature verify.
//! 2. **Shadow** — candidate output matches baseline on a shadow corpus.
//! 3. **Property** — declared invariants hold over property-test samples.
//! 4. **Canary** — a small fraction of traffic routed to the candidate;
//!    error rate stays under threshold.
//! 5. **JudgePeer** — LLM judge + multi-agent peer review both rate ≥ τ.
//! 6. **UserSignals** — aggregated user feedback ≥ τ_user.
//! 7. **Security** (transversal) — no unsafe source, capability audit passes,
//!    no forbidden syscalls.
//!
//! Each gate produces a `GateOutcome` with a verdict and evidence strings.
//! The overall `PromotionDecision` is either `Promote` (all gates passed) or
//! `Block` (at least one gate failed), with the aggregated evidence.
//!
//! Gates are caller-supplied via the `PromotionGate` trait. The pipeline
//! itself enforces ordering, short-circuit-on-block semantics (configurable),
//! and evidence aggregation.

use super::capability::CapabilitySet;
use super::registry::{SkillDefinition, SkillId, SkillVersion};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Verdicts + outcomes
// =============================================================================

/// Result of one gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GateVerdict {
    /// Gate passed.
    Pass,
    /// Gate failed — skill must not be promoted.
    Fail,
    /// Insufficient data to decide (e.g. too few canary samples). Treated as
    /// a "block" by default but the pipeline config can elect to promote on
    /// Skip for low-stakes gates.
    Skip,
}

impl fmt::Display for GateVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => f.write_str("pass"),
            Self::Fail => f.write_str("fail"),
            Self::Skip => f.write_str("skip"),
        }
    }
}

/// Verdict + evidence from one gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GateOutcome {
    pub gate_name: String,
    pub verdict: GateVerdict,
    /// Human-readable evidence strings. Joined in the promotion ledger event.
    pub evidence: Vec<String>,
    /// Optional numeric score (0..=1), if the gate has one.
    pub score: Option<f32>,
}

impl GateOutcome {
    pub fn pass(name: impl Into<String>) -> Self {
        Self {
            gate_name: name.into(),
            verdict: GateVerdict::Pass,
            evidence: Vec::new(),
            score: None,
        }
    }
    pub fn fail(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            gate_name: name.into(),
            verdict: GateVerdict::Fail,
            evidence: vec![reason.into()],
            score: None,
        }
    }
    pub fn skip(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            gate_name: name.into(),
            verdict: GateVerdict::Skip,
            evidence: vec![reason.into()],
            score: None,
        }
    }
    pub fn with_score(mut self, s: f32) -> Self {
        self.score = Some(s);
        self
    }
    pub fn with_evidence(mut self, e: impl Into<String>) -> Self {
        self.evidence.push(e.into());
        self
    }
}

// =============================================================================
// Gate trait
// =============================================================================

/// Input bundle passed to every gate.
#[derive(Debug)]
#[non_exhaustive]
pub struct PromotionContext<'a> {
    pub skill: &'a SkillDefinition,
    /// Caller-granted capability set (used by the Security gate).
    pub caller_capabilities: &'a CapabilitySet,
    /// Recent trajectory count for this skill. Used by canary + user-signal gates.
    pub recent_invocations: u64,
}

/// A single gate. Implementations are caller-supplied so the crate stays
/// policy-agnostic. Provide implementations of common gates (e.g.
/// `IntegrityGate`, `ThresholdGate`) in this module.
pub trait PromotionGate: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, ctx: &PromotionContext<'_>) -> GateOutcome;
}

// =============================================================================
// Built-in gates
// =============================================================================

/// Integrity gate: recompute content hash and compare.
pub struct IntegrityGate;

impl PromotionGate for IntegrityGate {
    fn name(&self) -> &'static str {
        "integrity"
    }
    fn evaluate(&self, ctx: &PromotionContext<'_>) -> GateOutcome {
        let expected = ctx.skill.compute_content_hash();
        if expected == ctx.skill.content_hash_hex {
            GateOutcome::pass(self.name()).with_evidence(format!("hash={expected}"))
        } else {
            GateOutcome::fail(
                self.name(),
                format!(
                    "hash mismatch: stored={}, recomputed={}",
                    ctx.skill.content_hash_hex, expected
                ),
            )
        }
    }
}

/// Security gate: validates that the skill's capabilities do not exceed
/// what the caller has granted, and that no `ToolCall` capabilities target
/// denylisted tools.
pub struct SecurityGate {
    /// Tools that are never allowed to be invoked by a skill.
    pub denylisted_tools: Vec<String>,
}

impl SecurityGate {
    pub fn new(denylisted_tools: Vec<String>) -> Self {
        Self { denylisted_tools }
    }
}

impl PromotionGate for SecurityGate {
    fn name(&self) -> &'static str {
        "security"
    }
    fn evaluate(&self, ctx: &PromotionContext<'_>) -> GateOutcome {
        use super::capability::Capability;
        if let Some(missing) = ctx
            .skill
            .capabilities
            .first_missing(ctx.caller_capabilities)
        {
            return GateOutcome::fail(self.name(), format!("capability not granted: {missing}"));
        }
        for cap in ctx.skill.capabilities.iter() {
            if let Capability::ToolCall(name) = cap {
                if self.denylisted_tools.iter().any(|d| d == name) {
                    return GateOutcome::fail(self.name(), format!("tool '{name}' is on denylist"));
                }
            }
        }
        GateOutcome::pass(self.name()).with_evidence(format!(
            "caps={}, denylist_size={}",
            ctx.skill.capabilities.len(),
            self.denylisted_tools.len()
        ))
    }
}

/// Generic threshold gate: supply a name, a predicate over `PromotionContext`,
/// and a reason. Useful for canary / user-signal gates wired by the caller.
pub struct ThresholdGate {
    name: &'static str,
    threshold: f32,
    actual: f32,
    min_samples: u64,
    description: String,
}

impl ThresholdGate {
    pub fn new(
        name: &'static str,
        threshold: f32,
        actual: f32,
        min_samples: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name,
            threshold,
            actual,
            min_samples,
            description: description.into(),
        }
    }
}

impl PromotionGate for ThresholdGate {
    fn name(&self) -> &'static str {
        self.name
    }
    fn evaluate(&self, ctx: &PromotionContext<'_>) -> GateOutcome {
        if ctx.recent_invocations < self.min_samples {
            return GateOutcome::skip(
                self.name,
                format!(
                    "insufficient samples: {} < {}",
                    ctx.recent_invocations, self.min_samples
                ),
            );
        }
        if self.actual >= self.threshold {
            GateOutcome::pass(self.name)
                .with_score(self.actual)
                .with_evidence(format!(
                    "{}: {:.3} >= {:.3}",
                    self.description, self.actual, self.threshold
                ))
        } else {
            GateOutcome::fail(
                self.name,
                format!(
                    "{}: {:.3} < {:.3}",
                    self.description, self.actual, self.threshold
                ),
            )
            .with_score(self.actual)
        }
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// Why promotion was blocked (or "Promote" if it wasn't).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PromotionReason {
    AllGatesPassed,
    GateFailed { gate: String, reason: String },
    GateSkipped { gate: String, reason: String },
}

impl fmt::Display for PromotionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllGatesPassed => f.write_str("all gates passed"),
            Self::GateFailed { gate, reason } => write!(f, "gate '{gate}' failed: {reason}"),
            Self::GateSkipped { gate, reason } => {
                write!(f, "gate '{gate}' skipped: {reason}")
            }
        }
    }
}

/// Decision at the end of a promotion run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromotionDecision {
    pub skill: SkillId,
    pub version: SkillVersion,
    /// True = promote to `Exploited`. False = keep current status.
    pub promote: bool,
    pub reason: PromotionReason,
    pub gate_outcomes: Vec<GateOutcome>,
}

/// Pipeline config + gates. Short-circuits on the first Fail by default;
/// Skip is treated as Block unless `allow_skip` is true.
pub struct PromotionPipeline {
    gates: Vec<Box<dyn PromotionGate>>,
    pub allow_skip: bool,
    pub short_circuit_on_fail: bool,
}

impl PromotionPipeline {
    pub fn new(gates: Vec<Box<dyn PromotionGate>>) -> Self {
        Self {
            gates,
            allow_skip: false,
            short_circuit_on_fail: true,
        }
    }

    /// Builder: allow a gate Skip verdict to count as Pass. Useful for
    /// low-stakes gates when there's not yet enough data.
    pub fn with_allow_skip(mut self, v: bool) -> Self {
        self.allow_skip = v;
        self
    }

    /// Run all gates and return a decision.
    pub fn evaluate(&self, ctx: &PromotionContext<'_>) -> PromotionDecision {
        let mut outcomes = Vec::with_capacity(self.gates.len());
        let mut failure: Option<(String, String)> = None;
        let mut skipped_block: Option<(String, String)> = None;

        for gate in &self.gates {
            let outcome = gate.evaluate(ctx);
            match outcome.verdict {
                GateVerdict::Pass => {}
                GateVerdict::Fail => {
                    let reason_str = outcome
                        .evidence
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "no evidence".into());
                    failure = Some((outcome.gate_name.clone(), reason_str));
                    outcomes.push(outcome);
                    if self.short_circuit_on_fail {
                        return PromotionDecision {
                            skill: ctx.skill.id.clone(),
                            version: ctx.skill.version,
                            promote: false,
                            reason: {
                                let (g, r) = failure.expect("set just above");
                                PromotionReason::GateFailed { gate: g, reason: r }
                            },
                            gate_outcomes: outcomes,
                        };
                    }
                    continue;
                }
                GateVerdict::Skip => {
                    if !self.allow_skip {
                        let reason_str = outcome
                            .evidence
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "no evidence".into());
                        skipped_block = Some((outcome.gate_name.clone(), reason_str));
                    }
                }
            }
            outcomes.push(outcome);
        }

        if let Some((g, r)) = failure {
            return PromotionDecision {
                skill: ctx.skill.id.clone(),
                version: ctx.skill.version,
                promote: false,
                reason: PromotionReason::GateFailed { gate: g, reason: r },
                gate_outcomes: outcomes,
            };
        }
        if let Some((g, r)) = skipped_block {
            return PromotionDecision {
                skill: ctx.skill.id.clone(),
                version: ctx.skill.version,
                promote: false,
                reason: PromotionReason::GateSkipped { gate: g, reason: r },
                gate_outcomes: outcomes,
            };
        }
        PromotionDecision {
            skill: ctx.skill.id.clone(),
            version: ctx.skill.version,
            promote: true,
            reason: PromotionReason::AllGatesPassed,
            gate_outcomes: outcomes,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_forge::capability::{Capability, CapabilitySet};
    use crate::skill_forge::declarative::{SkillStep, StepKind};
    use crate::skill_forge::registry::{
        SkillDefinition, SkillId, SkillMode, SkillStatus, SkillVersion,
    };

    fn mk_def() -> SkillDefinition {
        let mut def = SkillDefinition {
            id: SkillId::new("test"),
            version: SkillVersion(1),
            name: "test".into(),
            description: "desc".into(),
            mode: SkillMode::Declarative(vec![SkillStep {
                kind: StepKind::Plan { prompt: "p".into() },
                bind: None,
            }]),
            capabilities: CapabilitySet::empty(),
            content_hash_hex: String::new(),
            status: SkillStatus::Exploring,
            tenant: "t".into(),
            shared_cross_tenant: false,
        };
        def.content_hash_hex = def.compute_content_hash();
        def
    }

    #[test]
    fn integrity_gate_passes_on_valid_hash() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 0,
        };
        let outcome = IntegrityGate.evaluate(&ctx);
        assert_eq!(outcome.verdict, GateVerdict::Pass);
    }

    #[test]
    fn integrity_gate_fails_on_tampered_hash() {
        let mut def = mk_def();
        def.content_hash_hex = "tampered".into();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 0,
        };
        let outcome = IntegrityGate.evaluate(&ctx);
        assert_eq!(outcome.verdict, GateVerdict::Fail);
    }

    #[test]
    fn security_gate_fails_on_missing_capability() {
        let mut def = mk_def();
        def.capabilities = CapabilitySet::empty().with(Capability::Random);
        def.content_hash_hex = def.compute_content_hash();
        let granted = CapabilitySet::empty(); // nothing granted
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &granted,
            recent_invocations: 0,
        };
        let outcome = SecurityGate::new(vec![]).evaluate(&ctx);
        assert_eq!(outcome.verdict, GateVerdict::Fail);
    }

    #[test]
    fn security_gate_fails_on_denylisted_tool() {
        let mut def = mk_def();
        def.capabilities = CapabilitySet::empty().with(Capability::ToolCall("dangerous".into()));
        def.content_hash_hex = def.compute_content_hash();
        let granted = CapabilitySet::empty().with(Capability::ToolCall("dangerous".into()));
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &granted,
            recent_invocations: 0,
        };
        let outcome = SecurityGate::new(vec!["dangerous".into()]).evaluate(&ctx);
        assert_eq!(outcome.verdict, GateVerdict::Fail);
    }

    #[test]
    fn threshold_gate_skips_when_samples_below_min() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 5,
        };
        let g = ThresholdGate::new("canary", 0.95, 0.99, 50, "error rate");
        let outcome = g.evaluate(&ctx);
        assert_eq!(outcome.verdict, GateVerdict::Skip);
    }

    #[test]
    fn threshold_gate_passes_above() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 100,
        };
        let g = ThresholdGate::new("canary", 0.95, 0.99, 50, "error rate");
        assert_eq!(g.evaluate(&ctx).verdict, GateVerdict::Pass);
    }

    #[test]
    fn pipeline_short_circuits_on_first_fail() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 100,
        };
        let pipeline = PromotionPipeline::new(vec![
            Box::new(IntegrityGate),
            Box::new(ThresholdGate::new("canary", 0.99, 0.5, 50, "error")),
            Box::new(ThresholdGate::new("judge", 0.8, 0.9, 50, "judge")),
        ]);
        let decision = pipeline.evaluate(&ctx);
        assert!(!decision.promote);
        // Short-circuit: only integrity + canary outcomes recorded, not judge.
        assert_eq!(decision.gate_outcomes.len(), 2);
    }

    #[test]
    fn pipeline_promotes_when_all_pass() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 100,
        };
        let pipeline = PromotionPipeline::new(vec![
            Box::new(IntegrityGate),
            Box::new(SecurityGate::new(vec![])),
            Box::new(ThresholdGate::new("canary", 0.5, 0.9, 50, "error rate")),
            Box::new(ThresholdGate::new("judge", 0.5, 0.8, 50, "judge")),
        ]);
        let decision = pipeline.evaluate(&ctx);
        assert!(decision.promote);
        assert!(matches!(decision.reason, PromotionReason::AllGatesPassed));
    }

    #[test]
    fn pipeline_skip_blocks_without_allow_skip() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 5, // below min_samples = 50
        };
        let pipeline = PromotionPipeline::new(vec![
            Box::new(IntegrityGate),
            Box::new(ThresholdGate::new("canary", 0.95, 0.99, 50, "error rate")),
        ]);
        let decision = pipeline.evaluate(&ctx);
        assert!(!decision.promote);
        assert!(matches!(
            decision.reason,
            PromotionReason::GateSkipped { .. }
        ));
    }

    #[test]
    fn pipeline_skip_passes_with_allow_skip() {
        let def = mk_def();
        let caps = CapabilitySet::empty();
        let ctx = PromotionContext {
            skill: &def,
            caller_capabilities: &caps,
            recent_invocations: 5,
        };
        let pipeline = PromotionPipeline::new(vec![
            Box::new(IntegrityGate),
            Box::new(ThresholdGate::new("canary", 0.95, 0.99, 50, "error rate")),
        ])
        .with_allow_skip(true);
        let decision = pipeline.evaluate(&ctx);
        assert!(decision.promote);
    }
}
