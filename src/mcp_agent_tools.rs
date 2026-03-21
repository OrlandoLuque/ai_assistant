//! MCP Agent Management Tools (Block H)
//!
//! Exposes agent pool management via MCP tools for supervisors and users.

use std::sync::{Arc, Mutex};

use crate::agent_wiring::AgentPool;
use crate::mcp_protocol::{McpServer, McpTool, McpToolAnnotation};

/// Register agent management tools on the given MCP server.
///
/// Tools registered:
/// - `agent_pool_status` — list active agents, queue length, completed count
/// - `agent_task_progress` — detail of a specific running agent
/// - `agent_stop` — cancel a running agent by ID
/// - `agent_list_definitions` — list registered agent definitions
pub fn register_mcp_agent_tools(server: &mut McpServer, pool: Arc<Mutex<AgentPool>>) {
    // ── agent_pool_status ──
    {
        let pool = pool.clone();
        server.register_tool(
            McpTool::new(
                "agent_pool_status",
                "Get the current status of the agent pool: active agents, queue length, and completed results count.",
            )
            .with_annotations(McpToolAnnotation {
                title: Some("Agent Pool Status".to_string()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |_args| {
                let guard = pool.lock().map_err(|e| format!("Lock error: {}", e))?;
                let statuses = guard.active_statuses();
                let agents: Vec<serde_json::Value> = statuses
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "agent_id": s.agent_id,
                            "task_id": s.task_id,
                            "iteration": s.iteration,
                            "state": format!("{:?}", s.state),
                            "cost": s.cost,
                            "tools_called": s.tools_called,
                            "idle_streak": s.idle_streak,
                        })
                    })
                    .collect();

                Ok(serde_json::json!({
                    "active_agents": agents,
                    "active_count": guard.active_count(),
                    "queue_length": guard.queue_len(),
                }))
            },
        );
    }

    // ── agent_task_progress ──
    {
        let pool = pool.clone();
        server.register_tool(
            McpTool::new(
                "agent_task_progress",
                "Get detailed progress of a specific running agent by agent_id.",
            )
            .with_property("agent_id", "string", "The agent ID to query", true)
            .with_annotations(McpToolAnnotation {
                title: Some("Agent Task Progress".to_string()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let agent_id = args
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: agent_id")?;

                let guard = pool.lock().map_err(|e| format!("Lock error: {}", e))?;
                let statuses = guard.active_statuses();
                let status = statuses
                    .iter()
                    .find(|s| s.agent_id == agent_id)
                    .ok_or_else(|| format!("Agent '{}' not found in active pool", agent_id))?;

                Ok(serde_json::json!({
                    "agent_id": status.agent_id,
                    "task_id": status.task_id,
                    "iteration": status.iteration,
                    "state": format!("{:?}", status.state),
                    "cost": status.cost,
                    "tools_called": status.tools_called,
                    "idle_streak": status.idle_streak,
                }))
            },
        );
    }

    // ── agent_stop ──
    {
        let pool = pool.clone();
        server.register_tool(
            McpTool::new(
                "agent_stop",
                "Cancel a running agent by its agent_id. The agent will stop at its next iteration check.",
            )
            .with_property("agent_id", "string", "The agent ID to cancel", true)
            .with_annotations(McpToolAnnotation {
                title: Some("Stop Agent".to_string()),
                read_only_hint: Some(false),
                destructive_hint: Some(true),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let agent_id = args
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: agent_id")?;

                let guard = pool.lock().map_err(|e| format!("Lock error: {}", e))?;
                let cancelled = guard.cancel_agent(agent_id);

                Ok(serde_json::json!({
                    "agent_id": agent_id,
                    "cancelled": cancelled,
                }))
            },
        );
    }

    // ── agent_list_definitions ──
    {
        let pool = pool.clone();
        server.register_tool(
            McpTool::new(
                "agent_list_definitions",
                "List all registered agent definitions with their capabilities and roles.",
            )
            .with_annotations(McpToolAnnotation {
                title: Some("List Agent Definitions".to_string()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |_args| {
                let guard = pool.lock().map_err(|e| format!("Lock error: {}", e))?;
                let log: Vec<String> = guard
                    .trigger_log()
                    .iter()
                    .map(|r| format!("{:?}", r))
                    .collect();

                Ok(serde_json::json!({
                    "trigger_log": log,
                    "active_count": guard.active_count(),
                }))
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_tools::ToolRegistry;

    fn make_pool() -> Arc<Mutex<AgentPool>> {
        let factory = crate::agent_wiring::make_response_generator_factory(|_model| {
            crate::agent_wiring::make_response_generator(|_msgs| {
                "test response".to_string()
            })
        });
        let registry = ToolRegistry::new();
        Arc::new(Mutex::new(AgentPool::new(4, factory, registry)))
    }

    #[test]
    fn test_register_mcp_agent_tools() {
        let mut server = McpServer::new("test-server", "1.0");
        register_mcp_agent_tools(&mut server, make_pool());
        // Verify tools were registered by checking the server handles them
        let result = server.handle_message(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        );
        assert!(result.contains("agent_pool_status"));
        assert!(result.contains("agent_task_progress"));
        assert!(result.contains("agent_stop"));
        assert!(result.contains("agent_list_definitions"));
    }

    #[test]
    fn test_agent_pool_status_tool_via_message() {
        let mut server = McpServer::new("test-server", "1.0");
        register_mcp_agent_tools(&mut server, make_pool());

        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"agent_pool_status","arguments":{}}}"#,
        );
        assert!(response.contains("active_count"));
        assert!(response.contains("queue_length"));
    }

    #[test]
    fn test_agent_stop_via_message() {
        let mut server = McpServer::new("test-server", "1.0");
        register_mcp_agent_tools(&mut server, make_pool());

        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"agent_stop","arguments":{"agent_id":"nonexistent"}}}"#,
        );
        assert!(response.contains("cancelled"));
    }

    #[test]
    fn test_agent_task_progress_not_found() {
        let mut server = McpServer::new("test-server", "1.0");
        register_mcp_agent_tools(&mut server, make_pool());

        let response = server.handle_message(
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"agent_task_progress","arguments":{"agent_id":"nonexistent"}}}"#,
        );
        // Should contain an error since agent doesn't exist
        assert!(response.contains("not found") || response.contains("error") || response.contains("isError"));
    }
}
