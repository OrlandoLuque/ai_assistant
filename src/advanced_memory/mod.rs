//! Advanced memory system: episodic, procedural, entity, consolidation, and more.
//!
//! This module provides a multi-tier memory architecture for AI assistants,
//! including episodic memory, procedural memory, entity tracking,
//! memory consolidation, temporal graphs, and self-evolving procedures.
//!
//! Feature-gated behind the `advanced-memory` feature flag.

mod helpers;
mod episodic;
mod procedural;
mod entity;
mod consolidation;
mod manager;
mod temporal;
mod evolution;
mod extraction;
mod scheduler;
mod sharing;
mod search;
mod persistence;

#[cfg(test)]
mod tests;

// Re-export all public types so they remain accessible as advanced_memory::TypeName

pub use helpers::*;
pub use episodic::*;
pub use procedural::*;
pub use entity::*;
pub use consolidation::*;
pub use manager::*;
pub use temporal::*;
pub use evolution::*;
pub use extraction::*;
pub use scheduler::*;
pub use sharing::*;
pub use search::*;
pub use persistence::*;
