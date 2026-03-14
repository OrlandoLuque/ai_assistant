//! MCP tools for knowledge search and graph queries.
//!
//! Registers read-only tools for searching the RAG knowledge base and
//! querying the knowledge graph via the MCP protocol.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::server::McpServer;
use super::types::{McpTool, McpToolAnnotation};
use crate::knowledge_graph::{KnowledgeGraph, PatternEntityExtractor};
use crate::rag::RagDb;

/// Register knowledge-related MCP tools on a server.
///
/// # Arguments
/// * `server` — MCP server to register tools on
/// * `rag_db_path` — Path to the RAG SQLite database (a separate connection is opened)
/// * `knowledge_graph` — Optional knowledge graph (enables graph-specific tools)
///
/// # Tools registered
/// - `search_knowledge` — BM25 full-text search over indexed chunks
/// - `list_knowledge_sources` — List all indexed document sources
/// - `query_graph` — Query the knowledge graph for entities and relations (if graph available)
/// - `get_entity` — Look up a specific entity by name (if graph available)
pub fn register_knowledge_tools(
    server: &mut McpServer,
    rag_db_path: PathBuf,
    knowledge_graph: Option<Arc<KnowledgeGraph>>,
) {
    register_search_knowledge(server, rag_db_path.clone());
    register_list_sources(server, rag_db_path);

    if let Some(kg) = knowledge_graph {
        register_query_graph(server, kg.clone());
        register_get_entity(server, kg);
    }
}

/// Lazy-opening RagDb wrapper for MCP tool handlers.
/// Opens the database connection on first use and caches it.
fn get_or_open_db(
    db: &Mutex<Option<RagDb>>,
    path: &Path,
) -> Result<(), String> {
    let mut lock = db.lock().map_err(|e| format!("Lock error: {}", e))?;
    if lock.is_none() {
        *lock = Some(RagDb::open(path).map_err(|e| format!("Failed to open RAG DB: {}", e))?);
    }
    Ok(())
}

fn register_search_knowledge(server: &mut McpServer, db_path: PathBuf) {
    let db: Arc<Mutex<Option<RagDb>>> = Arc::new(Mutex::new(None));
    let path = db_path;

    server.register_tool(
        McpTool::new(
            "search_knowledge",
            "Search indexed knowledge chunks using BM25 full-text search. \
             Returns matching document chunks with source, section, content, and token count.",
        )
        .with_property("query", "string", "Search query text", true)
        .with_property(
            "max_results",
            "integer",
            "Maximum number of chunks to return (default: 5)",
            false,
        )
        .with_property(
            "max_tokens",
            "integer",
            "Maximum total tokens in results (default: 2000)",
            false,
        )
        .with_annotations(McpToolAnnotation {
            title: Some("Search Knowledge Base".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |args| {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: query")?;
            let max_results = args
                .get("max_results")
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
            let max_tokens = args
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(2000) as usize;

            get_or_open_db(&db, &path)?;
            let lock = db.lock().map_err(|e| format!("Lock error: {}", e))?;
            let rag = lock.as_ref().ok_or("RAG DB not available")?;

            let chunks = rag
                .search_knowledge(query, max_tokens, max_results)
                .map_err(|e| format!("Search error: {}", e))?;

            Ok(serde_json::json!({
                "chunks": chunks.iter().map(|c| serde_json::json!({
                    "source": c.source,
                    "section": c.section,
                    "content": c.content,
                    "tokens": c.token_count,
                })).collect::<Vec<_>>(),
                "total_chunks": chunks.len(),
                "total_tokens": chunks.iter().map(|c| c.token_count).sum::<usize>(),
            }))
        },
    );
}

fn register_list_sources(server: &mut McpServer, db_path: PathBuf) {
    let db: Arc<Mutex<Option<RagDb>>> = Arc::new(Mutex::new(None));
    let path = db_path;

    server.register_tool(
        McpTool::new(
            "list_knowledge_sources",
            "List all indexed knowledge document sources in the RAG database.",
        )
        .with_annotations(McpToolAnnotation {
            title: Some("List Knowledge Sources".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |_args| {
            get_or_open_db(&db, &path)?;
            let lock = db.lock().map_err(|e| format!("Lock error: {}", e))?;
            let rag = lock.as_ref().ok_or("RAG DB not available")?;

            let sources = rag
                .get_knowledge_sources()
                .map_err(|e| format!("Error listing sources: {}", e))?;

            Ok(serde_json::json!({
                "sources": sources,
                "count": sources.len(),
            }))
        },
    );
}

fn register_query_graph(server: &mut McpServer, kg: Arc<KnowledgeGraph>) {
    server.register_tool(
        McpTool::new(
            "query_graph",
            "Query the knowledge graph for entities, relations, and related document chunks. \
             Extracts entities from the query and finds matching nodes in the graph.",
        )
        .with_property("query", "string", "Natural language query to search entities", true)
        .with_annotations(McpToolAnnotation {
            title: Some("Query Knowledge Graph".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
        move |args| {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("Missing required parameter: query")?;

            let extractor = PatternEntityExtractor::new();
            let result = kg
                .query(query, &extractor)
                .map_err(|e| format!("Graph query error: {}", e))?;

            // Build context from retrieved chunks
            let context: String = result
                .chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");

            Ok(serde_json::json!({
                "entities": result.entities_found.iter().map(|e| serde_json::json!({
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.as_str(),
                    "aliases": e.aliases,
                })).collect::<Vec<_>>(),
                "relations": result.relations.iter().map(|r| serde_json::json!({
                    "from": r.from,
                    "to": r.to,
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                })).collect::<Vec<_>>(),
                "chunks_count": result.chunks.len(),
                "context": context,
                "processing_time_ms": result.processing_time_ms,
            }))
        },
    );
}

fn register_get_entity(server: &mut McpServer, kg: Arc<KnowledgeGraph>) {
    server.register_tool(
        McpTool::new(
            "get_entity",
            "Get details about a specific entity in the knowledge graph by name.",
        )
        .with_property("name", "string", "Entity name to look up", true)
        .with_annotations(McpToolAnnotation {
            title: Some("Get Entity Details".to_string()),
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

            let entity = kg
                .get_entity_by_name(name)
                .map_err(|e| format!("Entity lookup error: {}", e))?;

            match entity {
                Some(e) => Ok(serde_json::json!({
                    "found": true,
                    "entity": {
                        "id": e.id,
                        "name": e.name,
                        "type": e.entity_type.as_str(),
                        "aliases": e.aliases,
                        "metadata": e.metadata,
                        "created_at": e.created_at,
                        "updated_at": e.updated_at,
                    }
                })),
                None => Ok(serde_json::json!({
                    "found": false,
                    "name": name,
                })),
            }
        },
    );
}
