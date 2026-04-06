//! Advanced memory system: episodic, procedural, entity, consolidation, and more.
//!
//! This module provides a multi-tier memory architecture for AI assistants,
//! including episodic memory, procedural memory, entity tracking,
//! memory consolidation, temporal graphs, and self-evolving procedures.
//!
//! Feature-gated behind the `advanced-memory` feature flag.

mod consolidation;
mod entity;
mod episodic;
mod evolution;
mod extraction;
mod helpers;
mod manager;
mod persistence;
mod procedural;
mod scheduler;
mod search;
mod sharing;
mod temporal;

#[cfg(test)]
mod tests;

// Re-export all public types so they remain accessible as advanced_memory::TypeName

pub use consolidation::*;
pub use entity::*;
pub use episodic::*;
pub use evolution::*;
pub use extraction::*;
pub use helpers::*;
pub use manager::*;
pub use persistence::*;
pub use procedural::*;
pub use scheduler::*;
pub use search::*;
pub use sharing::*;
pub use temporal::*;
