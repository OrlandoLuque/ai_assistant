//! DSPy-style prompt signatures, optimization, and self-reflection.
//!
//! This module provides a declarative approach to prompt engineering inspired by
//! DSPy's signature system. Instead of hand-crafting prompts, you declare the
//! input/output fields and let optimizers find the best prompt formulation.
//!
//! # Components
//!
//! - **Signatures**: Declarative input/output specifications (`Signature`, `SignatureField`)
//! - **Compilation**: Convert signatures into executable prompts (`CompiledPrompt`)
//! - **Metrics**: Evaluate prompt quality (`EvalMetric`, `ExactMatch`, `F1Score`, `ContainsAnswer`)
//! - **Optimizers**: Search for better prompts (`BootstrapFewShot`, `GridSearchOptimizer`,
//!   `RandomSearchOptimizer`, `BayesianOptimizer`)
//! - **Self-Reflection**: Analyze results and suggest improvements (`SelfReflector`)
//!
//! Feature-gated behind the `prompt-signatures` feature flag.

mod types;
mod optimizers;
mod reflector;
mod gepa;
mod miprov2;
mod assertions;
mod adapters;
mod simba;
mod reasoning;
mod judge;

#[cfg(test)]
mod tests;

// Re-export all public types so they remain accessible as prompt_signature::TypeName

pub use types::*;
pub use optimizers::*;
pub use reflector::*;
pub use gepa::*;
pub use miprov2::*;
pub use assertions::*;
pub use adapters::*;
pub use simba::*;
pub use reasoning::*;
pub use judge::*;
