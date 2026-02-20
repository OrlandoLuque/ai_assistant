//! MCP (Model Context Protocol) server example.
//!
//! Run with: cargo run --example mcp_server --features tools
//!
//! Demonstrates creating an MCP server with tools and resources.

use ai_assistant::mcp_protocol::McpToolAnnotation;
use ai_assistant::{McpRequest, McpResource, McpResourceContent, McpServer, McpTool};

fn main() {
    // Create an MCP server
    let mut server = McpServer::new("demo-server", "1.0.0");

    // Register a tool with annotations (2025-03-26 spec)
    let weather_tool = McpTool {
        name: "get_weather".to_string(),
        description: "Get current weather for a city".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
        annotations: Some(McpToolAnnotation {
            title: Some("Weather Lookup".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(true),
        }),
    };

    server.register_tool(weather_tool, |params| {
        let city = params
            .get("city")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        Ok(serde_json::json!({
            "city": city,
            "temperature": 22,
            "unit": "celsius",
            "condition": "sunny"
        }))
    });

    // Register a resource
    let readme_resource = McpResource {
        uri: "file:///README.md".to_string(),
        name: "Project README".to_string(),
        description: Some("The project documentation".to_string()),
        mime_type: Some("text/markdown".to_string()),
    };
    server.register_resource(readme_resource, |_uri| {
        Ok(McpResourceContent {
            uri: "file:///README.md".to_string(),
            text: Some("# Demo Project\nThis is a demo.".to_string()),
            blob: None,
            mime_type: Some("text/markdown".to_string()),
        })
    });

    // Handle a tools/list request
    let list_request = McpRequest::new("tools/list").with_id(1);

    let response = server.handle_request(list_request);
    println!("Tools list response:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Handle a tools/call request
    let call_request = McpRequest::new("tools/call")
        .with_id(2)
        .with_params(serde_json::json!({
            "name": "get_weather",
            "arguments": { "city": "Madrid" }
        }));

    let response = server.handle_request(call_request);
    println!("\nTool call response:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());
}
