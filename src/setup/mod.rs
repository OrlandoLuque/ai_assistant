// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Setup and management utilities shared between CLI and GUI.
//!
//! Provides prerequisite detection, configuration operations, Docker management,
//! node lifecycle control, and backup/restore — all usable from any frontend.

pub mod backup;
pub mod config_ops;
pub mod docker_ops;
pub mod node_manager;
pub mod prereq;
