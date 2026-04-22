//! Skill Forge — LLM-authored skills with Declarative DSL + WASM-Rust execution.
//!
//! A skill is a reusable, composable operation the agent can invoke by id.
//! Skills come in two execution modes:
//! - **Declarative**: a sequence of `SkillStep` (Plan / ToolCall / Transform / Branch)
//!   authored by the LLM as structured data. Safe-by-construction.
//! - **WASM-Rust**: the LLM authors Rust source, the crate compiles it to
//!   `wasm32-wasip1`, and it runs inside a wasmtime sandbox with fuel limits,
//!   memory caps, and capability gating. Source + WASM artifact + hash + signature
//!   are persisted for audit.
//!
//! Every skill carries a content hash (Blake3) and an Ed25519 signature.
//! Registry operations emit events to a hash-chained `SkillLedger` for
//! tamper-evident history.
//!
//! Auditor binaries `ai_skills` (CLI) and `ai_skills_gui` (desktop) ship with
//! the crate and cover list/filter, detail, verify integrity, show promotion
//! timeline, and export signed evidence bundles.
//!
//! # Freeze
//!
//! `LearningFreezeConfig::freeze_skill_forge` is the runtime switch the admin
//! toggles to pause skill promotion. It is independent from the per-skill
//! `SkillStatus::Frozen` (which is a lifecycle state of a single skill). See
//! `learning_control::LearningSubsystem::SkillForge`.

pub mod capability;
pub mod declarative;
pub mod ledger;
pub mod promotion;
pub mod registry;

#[cfg(feature = "skill-forge")]
pub mod wasm;

pub use capability::{Capability, CapabilityError, CapabilitySet, NetAllowList, PathGlob};
pub use declarative::{DeclarativeExecutor, SkillStep, StepKind};
pub use ledger::{LedgerEvent, LedgerEventKind, LedgerVerifyError, SkillLedger};
pub use promotion::{
    GateOutcome, GateVerdict, PromotionDecision, PromotionGate, PromotionPipeline, PromotionReason,
};
pub use registry::{
    SkillDefinition, SkillError, SkillId, SkillInputs, SkillMode, SkillOutput, SkillRegistry,
    SkillRegistryError, SkillStatus, SkillVersion, WasmArtifact,
};

// Re-export core limits + constants so callers can configure without opening submodules.
pub use wasm_limits::{DEFAULT_WASM_FUEL, DEFAULT_WASM_MEMORY_BYTES, DEFAULT_WASM_TIMEOUT_SECS};

/// Default resource limits applied to WASM skill execution.
///
/// These values are intentionally conservative. Callers that need higher
/// limits must set them explicitly on `WasmRunConfig` — the defaults protect
/// against runaway skills authored by an imperfect LLM.
pub mod wasm_limits {
    /// Default fuel units: roughly 1 second of execution on a 1GHz core.
    /// Wasmtime fuel is approximately 1 unit per executed instruction.
    pub const DEFAULT_WASM_FUEL: u64 = 1_000_000_000;

    /// Default memory cap: 64 MiB per skill invocation.
    pub const DEFAULT_WASM_MEMORY_BYTES: usize = 64 * 1024 * 1024;

    /// Default wall-clock timeout: 30 seconds per skill invocation.
    pub const DEFAULT_WASM_TIMEOUT_SECS: u64 = 30;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_conservative() {
        assert!(DEFAULT_WASM_FUEL > 0);
        assert!(DEFAULT_WASM_MEMORY_BYTES <= 256 * 1024 * 1024);
        assert!(DEFAULT_WASM_TIMEOUT_SECS > 0 && DEFAULT_WASM_TIMEOUT_SECS <= 300);
    }
}
