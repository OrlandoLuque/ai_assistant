//! MCP v2 Tool Annotations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::McpTool;

/// Tool annotations indicating behavior characteristics (MCP v2).
///
/// These complement the existing `McpToolAnnotation` (hint-based, v4 spec) with
/// a more structured, boolean-based model suited for programmatic policy checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAnnotations {
    /// If true, the tool only reads data and does not modify any state.
    #[serde(default)]
    pub read_only: bool,
    /// If true, the tool may cause destructive / irreversible side effects.
    #[serde(default = "default_true")]
    pub destructive: bool,
    /// If true, calling with the same arguments always produces the same result.
    #[serde(default)]
    pub idempotent: bool,
    /// If true, the tool may interact with external systems not described in its schema.
    #[serde(default = "default_true")]
    pub open_world: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        }
    }
}

impl ToolAnnotations {
    /// A tool is considered "safe" if it is read-only and not destructive.
    pub fn is_safe(&self) -> bool {
        self.read_only && !self.destructive
    }

    /// A tool "needs confirmation" if it is destructive or interacts with the open world.
    pub fn needs_confirmation(&self) -> bool {
        self.destructive || self.open_world
    }
}

/// Wrapper that pairs an `McpTool` with `ToolAnnotations`.
///
/// This avoids modifying the existing `McpTool` struct while still allowing
/// annotation data to travel alongside tool definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedTool {
    pub tool: McpTool,
    pub annotations: ToolAnnotations,
}

impl AnnotatedTool {
    /// Create an `AnnotatedTool` with default annotations.
    pub fn from_tool(tool: McpTool) -> Self {
        Self {
            tool,
            annotations: ToolAnnotations::default(),
        }
    }

    /// Create an `AnnotatedTool` with explicit annotations.
    pub fn with_annotations(tool: McpTool, annotations: ToolAnnotations) -> Self {
        Self { tool, annotations }
    }
}

/// Registry that maps tool names to their `ToolAnnotations`.
pub struct ToolAnnotationRegistry {
    annotations: HashMap<String, ToolAnnotations>,
}

impl ToolAnnotationRegistry {
    pub fn new() -> Self {
        Self {
            annotations: HashMap::new(),
        }
    }

    /// Register annotations for a tool by name.
    pub fn register(&mut self, tool_name: &str, annotations: ToolAnnotations) {
        self.annotations.insert(tool_name.to_string(), annotations);
    }

    /// Get the annotations for a tool by name.
    pub fn get(&self, tool_name: &str) -> Option<&ToolAnnotations> {
        self.annotations.get(tool_name)
    }

    /// Check if a tool needs human approval based on its annotations.
    ///
    /// Returns `true` if annotations exist and `needs_confirmation()` is true,
    /// or if no annotations are registered (conservative default).
    pub fn needs_approval(&self, tool_name: &str) -> bool {
        match self.annotations.get(tool_name) {
            Some(ann) => ann.needs_confirmation(),
            None => true, // Unknown tool -- require approval by default
        }
    }
}

impl Default for ToolAnnotationRegistry {
    fn default() -> Self {
        Self::new()
    }
}
