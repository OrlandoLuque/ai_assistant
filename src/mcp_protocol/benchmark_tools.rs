//! MCP tools for dataset hallucination / faithfulness benchmarks.
//!
//! Read-only tools that expose the `eval_benchmarks` registry through the
//! MCP protocol. No downloads, no runs — just metadata lookup, so the tools
//! are safe to call from any MCP client without triggering network or LLM
//! activity.

use super::server::McpServer;
use super::types::{McpTool, McpToolAnnotation};
use crate::eval_benchmarks::BenchmarkLoader;

/// Register MCP benchmark tools on `server`.
///
/// # Tools registered
/// - `list_benchmarks` — Enumerate all registered benchmark loaders.
/// - `get_benchmark` — Return metadata for a specific benchmark by name.
pub fn register_benchmark_tools(server: &mut McpServer) {
    register_list(server);
    register_get(server);
}

fn loader_json(l: &dyn BenchmarkLoader) -> serde_json::Value {
    serde_json::json!({
        "name": l.name(),
        "description": l.description(),
        "license": l.license(),
        "citation": l.citation(),
        "sample_type": format!("{:?}", l.sample_type()),
        "requires_opt_in": l.requires_opt_in(),
        "download_urls": l.download_urls(),
    })
}

fn register_list(server: &mut McpServer) {
    server.register_tool(
        McpTool::new(
            "list_benchmarks",
            "List all dataset-based hallucination / faithfulness benchmarks registered \
             in eval_benchmarks. Read-only: returns metadata only (name, description, \
             license, citation, sample type, opt-in status, download URLs).",
        )
        .with_annotations(McpToolAnnotation {
            title: Some("List Hallucination Benchmarks".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |_args| {
            let items: Vec<serde_json::Value> = crate::eval_benchmarks::all_loaders()
                .iter()
                .map(|l| loader_json(l.as_ref()))
                .collect();
            Ok(serde_json::json!({
                "total": items.len(),
                "benchmarks": items,
            }))
        },
    );
}

fn register_get(server: &mut McpServer) {
    server.register_tool(
        McpTool::new(
            "get_benchmark",
            "Get metadata for a single hallucination benchmark by its registered name \
             (e.g. 'truthfulqa', 'halueval_qa', 'factscore', 'ragas_wikiqa', 'fever').",
        )
        .with_property("name", "string", "Registered benchmark name", true)
        .with_annotations(McpToolAnnotation {
            title: Some("Get Benchmark Metadata".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |args| {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: name")?;
            match crate::eval_benchmarks::get_loader(name) {
                Some(l) => Ok(serde_json::json!({
                    "found": true,
                    "benchmark": loader_json(l.as_ref()),
                })),
                None => Ok(serde_json::json!({
                    "found": false,
                    "name": name,
                })),
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_protocol::server::McpServer;
    use crate::mcp_protocol::types::McpRequest;

    fn call(server: &McpServer, tool: &str, args: serde_json::Value) -> serde_json::Value {
        let req = McpRequest::new("tools/call")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "name": tool,
                "arguments": args,
            }));
        let resp = server.handle_request(req);
        assert!(resp.error.is_none(), "tool error: {:?}", resp.error);
        let result = resp.result.expect("result present");
        // tools/call wraps the handler result as `{ "content": [ {"type":"text", "text": "<json>"} ] }`.
        let text = result["content"][0]["text"].as_str().expect("text payload");
        serde_json::from_str(text).expect("json payload")
    }

    #[test]
    fn list_benchmarks_returns_registered_loaders() {
        let mut server = McpServer::new("test", "0.0.0");
        register_benchmark_tools(&mut server);
        let body = call(&server, "list_benchmarks", serde_json::json!({}));
        let total = body["total"].as_u64().expect("total field");
        assert!(total >= 5, "expected at least 5 benchmarks, got {}", total);
        let benchmarks = body["benchmarks"].as_array().expect("benchmarks array");
        assert!(benchmarks.iter().any(|b| b["name"] == "truthfulqa"));
    }

    #[test]
    fn get_benchmark_by_name_resolves_and_errors_cleanly() {
        let mut server = McpServer::new("test", "0.0.0");
        register_benchmark_tools(&mut server);

        let hit = call(
            &server,
            "get_benchmark",
            serde_json::json!({ "name": "truthfulqa" }),
        );
        assert_eq!(hit["found"], true);
        assert_eq!(hit["benchmark"]["name"], "truthfulqa");
        assert!(hit["benchmark"]["license"].is_string());

        let miss = call(
            &server,
            "get_benchmark",
            serde_json::json!({ "name": "__nope__" }),
        );
        assert_eq!(miss["found"], false);
    }
}
