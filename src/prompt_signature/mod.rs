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

mod adapters;
mod assertions;
mod gepa;
mod judge;
mod miprov2;
mod optimizers;
mod reasoning;
mod reflector;
mod simba;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types so they remain accessible as prompt_signature::TypeName

pub use adapters::*;
pub use assertions::*;
pub use gepa::*;
pub use judge::*;
pub use miprov2::*;
pub use optimizers::*;
pub use reasoning::*;
pub use reflector::*;
pub use simba::*;
pub use types::*;
