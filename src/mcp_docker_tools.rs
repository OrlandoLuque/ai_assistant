//! MCP tool registration for Docker container management.
//!
//! Provides 8 MCP tools for creating, starting, stopping, removing, and
//! monitoring Docker containers via the MCP protocol. Uses the bollard-based
//! `ContainerExecutor` for production-grade container management.
//!
//! This module is gated behind `containers` + `tools` feature flags.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::container_executor::{ContainerExecutor, CreateOptions};
use crate::mcp_protocol::{McpServer, McpTool, McpToolAnnotation};

/// Register all Docker management tools on an MCP server.
///
/// # Arguments
///
/// - `server` — MCP server to register tools on.
/// - `executor` — Shared container executor.
pub fn register_mcp_docker_tools(
    server: &mut McpServer,
    executor: Arc<RwLock<ContainerExecutor>>,
) {
    register_list_containers(server, Arc::clone(&executor));
    register_create_container(server, Arc::clone(&executor));
    register_start_container(server, Arc::clone(&executor));
    register_stop_container(server, Arc::clone(&executor));
    register_remove_container(server, Arc::clone(&executor));
    register_exec(server, Arc::clone(&executor));
    register_logs(server, Arc::clone(&executor));
    register_container_status(server, executor);
}

// =============================================================================
// Tool 1: docker_list_containers
// =============================================================================

fn register_list_containers(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_list_containers".into(),
        description: "List all managed Docker containers with their IDs, names, images, and status".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("List Docker Containers".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |_params| {
        let guard = executor.read().map_err(|e| format!("Lock poisoned: {}", e))?;
        let containers: Vec<serde_json::Value> = guard
            .list()
            .iter()
            .map(|r| {
                serde_json::json!({
                    "container_id": r.container_id,
                    "name": r.name,
                    "image": r.image,
                    "status": format!("{}", r.status),
                    "created_at": r.created_at,
                })
            })
            .collect();
        Ok(serde_json::json!({ "containers": containers }))
    });
}

// =============================================================================
// Tool 2: docker_create_container
// =============================================================================

fn register_create_container(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_create_container".into(),
        description: "Create a new Docker container from an image".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Docker image to use (e.g. 'busybox:latest', 'python:3.12-slim')"
                },
                "name": {
                    "type": "string",
                    "description": "Optional container name"
                },
                "cmd": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional command to run (overrides image CMD)"
                }
            },
            "required": ["image"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Create Docker Container".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
        }),
    };

    server.register_tool(tool, move |params| {
        let image = params
            .get("image")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: image".to_string())?;

        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp_container");

        let opts = if let Some(cmd_arr) = params.get("cmd").and_then(|v| v.as_array()) {
            let cmd: Vec<String> = cmd_arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            CreateOptions {
                cmd: Some(cmd),
                ..Default::default()
            }
        } else {
            CreateOptions::default()
        };

        let mut guard = executor.write().map_err(|e| format!("Lock poisoned: {}", e))?;
        let container_id = guard
            .create(image, name, opts)
            .map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "container_id": container_id,
            "image": image,
            "name": name,
            "status": "created"
        }))
    });
}

// =============================================================================
// Tool 3: docker_start_container
// =============================================================================

fn register_start_container(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_start_container".into(),
        description: "Start a created Docker container".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to start"
                }
            },
            "required": ["container_id"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Start Container".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let mut guard = executor.write().map_err(|e| format!("Lock poisoned: {}", e))?;
        guard.start(id).map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "container_id": id,
            "status": "running"
        }))
    });
}

// =============================================================================
// Tool 4: docker_stop_container
// =============================================================================

fn register_stop_container(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_stop_container".into(),
        description: "Stop a running Docker container".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to stop"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds before force-killing (default: 10)",
                    "default": 10
                }
            },
            "required": ["container_id"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Stop Container".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let timeout = params
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;

        let mut guard = executor.write().map_err(|e| format!("Lock poisoned: {}", e))?;
        guard.stop(id, timeout).map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "container_id": id,
            "status": "stopped"
        }))
    });
}

// =============================================================================
// Tool 5: docker_remove_container
// =============================================================================

fn register_remove_container(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_remove_container".into(),
        description: "Remove a Docker container. Use force=true to remove running containers.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to remove"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force removal even if running (default: false)",
                    "default": false
                }
            },
            "required": ["container_id"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Remove Container".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let force = params
            .get("force")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut guard = executor.write().map_err(|e| format!("Lock poisoned: {}", e))?;
        guard.remove(id, force).map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "container_id": id,
            "removed": true
        }))
    });
}

// =============================================================================
// Tool 6: docker_exec
// =============================================================================

fn register_exec(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_exec".into(),
        description: "Execute a command inside a running Docker container".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to execute in"
                },
                "command": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Command and arguments (e.g. [\"python\", \"-c\", \"print('hi')\"])"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)",
                    "default": 60
                }
            },
            "required": ["container_id", "command"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Execute in Container".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let command_arr = params
            .get("command")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "Missing required parameter: command (array of strings)".to_string())?;

        let command: Vec<String> = command_arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if command.is_empty() {
            return Err("command array must not be empty".to_string());
        }

        let timeout_secs = params
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);

        let cmd_refs: Vec<&str> = command.iter().map(|s| s.as_str()).collect();
        let timeout = Duration::from_secs(timeout_secs);

        let guard = executor.read().map_err(|e| format!("Lock poisoned: {}", e))?;
        let result = guard.exec(id, &cmd_refs, timeout).map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "duration_ms": result.duration.as_millis() as u64,
            "success": result.success(),
        }))
    });
}

// =============================================================================
// Tool 7: docker_logs
// =============================================================================

fn register_logs(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_logs".into(),
        description: "Get logs from a Docker container".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to get logs from"
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of tail lines to return (default: 100)",
                    "default": 100
                }
            },
            "required": ["container_id"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Container Logs".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let tail = params
            .get("tail")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        let guard = executor.read().map_err(|e| format!("Lock poisoned: {}", e))?;
        let logs = guard.logs(id, tail).map_err(|e| format!("{}", e))?;

        Ok(serde_json::json!({
            "container_id": id,
            "logs": logs
        }))
    });
}

// =============================================================================
// Tool 8: docker_container_status
// =============================================================================

fn register_container_status(server: &mut McpServer, executor: Arc<RwLock<ContainerExecutor>>) {
    let tool = McpTool {
        name: "docker_container_status".into(),
        description: "Get the current status of a Docker container".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Container ID to check"
                }
            },
            "required": ["container_id"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Container Status".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    };

    server.register_tool(tool, move |params| {
        let id = params
            .get("container_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: container_id".to_string())?;

        let guard = executor.read().map_err(|e| format!("Lock poisoned: {}", e))?;
        match guard.status(id) {
            Some(status) => Ok(serde_json::json!({
                "container_id": id,
                "status": format!("{}", status),
            })),
            None => Err(format!("Container not found: {}", id)),
        }
    });
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server_with_tools() -> McpServer {
        let mut server = McpServer::new("test_docker", "0.1.0");
        let config = crate::container_executor::ContainerConfig::default();
        // ContainerExecutor::new requires Docker — skip if unavailable
        let executor = match ContainerExecutor::new(config) {
            Ok(e) => e,
            Err(_) => return server, // Return empty server if Docker unavailable
        };
        let arc = Arc::new(RwLock::new(executor));
        register_mcp_docker_tools(&mut server, arc);
        server
    }

    /// Verify that register_mcp_docker_tools registers exactly 8 tools.
    #[test]
    fn test_register_mcp_docker_tools_count() {
        let server = make_server_with_tools();
        // If Docker is unavailable, 0 tools are registered (graceful skip)
        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array();
        // Either 8 tools (Docker available) or 0 (Docker unavailable)
        if let Some(t) = tools {
            if !t.is_empty() {
                assert_eq!(t.len(), 8, "Expected 8 Docker MCP tools");
            }
        }
    }

    /// Verify docker_list_containers has correct schema.
    #[test]
    fn test_mcp_tool_list_schema() {
        let server = make_server_with_tools();
        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array();
        if let Some(t) = tools {
            let list_tool = t.iter().find(|t| t["name"] == "docker_list_containers");
            if let Some(tool) = list_tool {
                assert_eq!(tool["inputSchema"]["type"], "object");
                // No required params for list
                assert!(tool["inputSchema"].get("required").is_none());
            }
        }
    }

    /// Verify docker_create_container requires 'image' parameter.
    #[test]
    fn test_mcp_tool_create_schema() {
        let server = make_server_with_tools();
        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array();
        if let Some(t) = tools {
            let create_tool = t.iter().find(|t| t["name"] == "docker_create_container");
            if let Some(tool) = create_tool {
                let required = tool["inputSchema"]["required"].as_array().unwrap();
                assert!(required.iter().any(|r| r == "image"));
            }
        }
    }

    /// Verify read-only annotations on list, logs, and status tools.
    #[test]
    fn test_mcp_tool_annotations() {
        let server = make_server_with_tools();
        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array();
        if let Some(t) = tools {
            let read_only_tools = ["docker_list_containers", "docker_logs", "docker_container_status"];
            for name in &read_only_tools {
                if let Some(tool) = t.iter().find(|t| t["name"] == *name) {
                    if let Some(ann) = tool.get("annotations") {
                        assert_eq!(ann["readOnlyHint"], true, "{} should be read-only", name);
                        assert_eq!(ann["destructiveHint"], false, "{} should not be destructive", name);
                    }
                }
            }
            // docker_remove_container should be destructive
            if let Some(tool) = t.iter().find(|t| t["name"] == "docker_remove_container") {
                if let Some(ann) = tool.get("annotations") {
                    assert_eq!(ann["destructiveHint"], true, "remove should be destructive");
                }
            }
        }
    }

    /// Verify docker_exec requires command as array type.
    #[test]
    fn test_mcp_tool_exec_schema() {
        let server = make_server_with_tools();
        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array();
        if let Some(t) = tools {
            let exec_tool = t.iter().find(|t| t["name"] == "docker_exec");
            if let Some(tool) = exec_tool {
                let required = tool["inputSchema"]["required"].as_array().unwrap();
                assert!(required.iter().any(|r| r == "container_id"));
                assert!(required.iter().any(|r| r == "command"));
                assert_eq!(tool["inputSchema"]["properties"]["command"]["type"], "array");
            }
        }
    }
}
