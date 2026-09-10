//! Registering the research tools on an MCP server.
//!
//! The definitions and the dispatcher already lived in
//! [`crate::mcp_research_tools`]; what was missing was the bridge that puts them
//! on a server, so `ai_mcp_server` served every other tool set and not this one.
//! A client could see `search_papers` in the catalogue built by
//! `ResearchToolRegistry` and never find it over the wire.

use super::types::{McpTool, McpToolAnnotation};
use super::McpServer;
use crate::mcp_research_tools::{dispatch_tool, ResearchToolRegistry};
use std::collections::HashMap;

/// Register every research tool on `server`.
///
/// Each tool keeps the schema declared in `ResearchToolRegistry`, so the
/// catalogue and the wire cannot drift apart: there is one definition, used
/// twice.
pub fn register_research_tools(server: &mut McpServer) {
    let registry = ResearchToolRegistry::new();

    for tool in registry.tools() {
        let name = tool.name.clone();
        // Everything here reads: nothing writes to disk or mutates state, and
        // `search_papers` / `literature_review` reach five public APIs, which is
        // what `open_world_hint` is for.
        let reaches_network = matches!(
            name.as_str(),
            "search_papers" | "get_paper_metadata" | "export_bibtex" | "literature_review"
        );
        server.register_tool(
            McpTool::new(&name, &tool.description)
                .with_schema(tool.input_schema.clone())
                .with_annotations(McpToolAnnotation {
                    title: Some(tool.name.clone()),
                    read_only_hint: Some(true),
                    destructive_hint: Some(false),
                    // Network calls are not idempotent in the sense that
                    // matters here: the corpora change under you.
                    idempotent_hint: Some(!reaches_network),
                    open_world_hint: Some(reaches_network),
                }),
            move |args: serde_json::Value| {
                // The wire hands over a JSON object; `dispatch_tool` reads a map.
                // A call with no arguments is legal (some tools need none), so an
                // absent or non-object payload becomes an empty map rather than
                // an error.
                let map: HashMap<String, serde_json::Value> = args
                    .as_object()
                    .map(|o| o.clone().into_iter().collect())
                    .unwrap_or_default();
                dispatch_tool(&name, &map)
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalogued_tool_reaches_the_wire() {
        // The bug this guards: the registry listed six tools and the server
        // served none of them, so the catalogue and the wire disagreed.
        let mut server = McpServer::new("test", "0");
        register_research_tools(&mut server);

        // Enumerated over the wire, deliberately: asking the server's own
        // `tools/list` is what a client does, and it is the only view that
        // proves the tool is reachable rather than merely stored.
        let response =
            server.handle_request(crate::mcp_protocol::types::McpRequest::new("tools/list"));
        let served = serde_json::to_string(&response).expect("tools/list must serialise");
        for name in ResearchToolRegistry::new().tool_names() {
            assert!(
                served.contains(name),
                "{name} is in the catalogue but tools/list does not offer it"
            );
        }
    }
}
