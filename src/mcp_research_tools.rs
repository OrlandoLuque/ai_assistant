//! MCP tool definitions for research operations
//!
//! Provides Model Context Protocol (MCP) tool schemas for academic search,
//! BibTeX operations, literature review, and paper metadata extraction.
//! These tools are registered in the MCP server and exposed to AI clients.

use std::collections::HashMap;

// =============================================================================
// Tool Definitions
// =============================================================================

/// MCP tool definition for research operations.
#[derive(Debug, Clone)]
pub struct ResearchTool {
    /// Tool name (MCP tool ID)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for the tool's input parameters
    pub input_schema: serde_json::Value,
    /// Category for grouping
    pub category: ResearchToolCategory,
}

/// Categories of research tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResearchToolCategory {
    /// Academic paper search
    Search,
    /// BibTeX operations
    Bibliography,
    /// Literature review
    Review,
    /// Metadata extraction
    Metadata,
}

impl ResearchToolCategory {
    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Search => "Search",
            Self::Bibliography => "Bibliography",
            Self::Review => "Review",
            Self::Metadata => "Metadata",
        }
    }
}

impl std::fmt::Display for ResearchToolCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Registry of all research MCP tools.
pub struct ResearchToolRegistry {
    tools: Vec<ResearchTool>,
}

impl ResearchToolRegistry {
    /// Create a new registry with all research tools.
    pub fn new() -> Self {
        let tools = vec![
            Self::search_papers_tool(),
            Self::get_paper_metadata_tool(),
            Self::import_bibtex_tool(),
            Self::export_bibtex_tool(),
            Self::literature_review_tool(),
            Self::extract_paper_metadata_tool(),
        ];

        Self { tools }
    }

    /// Get all tool definitions.
    pub fn tools(&self) -> &[ResearchTool] {
        &self.tools
    }

    /// Get a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&ResearchTool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Get tools by category.
    pub fn tools_by_category(&self, category: ResearchToolCategory) -> Vec<&ResearchTool> {
        self.tools
            .iter()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Get all tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    // =========================================================================
    // Tool Definitions
    // =========================================================================

    fn search_papers_tool() -> ResearchTool {
        ResearchTool {
            name: "search_papers".to_string(),
            description: "Search academic databases (arXiv, Semantic Scholar, PubMed) for papers matching a query.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords, paper title, author name)"
                    },
                    "providers": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["arxiv", "semantic_scholar", "pubmed", "openalex", "crossref"] },
                        "description": "Which providers to search (default: all)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results per provider (default: 10)",
                        "minimum": 1,
                        "maximum": 50
                    },
                    "year_range": {
                        "type": "object",
                        "properties": {
                            "min": { "type": "integer" },
                            "max": { "type": "integer" }
                        },
                        "description": "Filter by publication year range"
                    }
                },
                "required": ["query"]
            }),
            category: ResearchToolCategory::Search,
        }
    }

    fn get_paper_metadata_tool() -> ResearchTool {
        ResearchTool {
            name: "get_paper_metadata".to_string(),
            description: "Get detailed metadata for a specific paper by its ID (arXiv ID, Semantic Scholar ID, PMID, OpenAlex ID, or DOI).".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper ID (e.g., '2301.01234' for arXiv, PMID for PubMed, 'W2741809807' or a DOI for OpenAlex, a DOI for Crossref)"
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["arxiv", "semantic_scholar", "pubmed", "openalex", "crossref"],
                        "description": "Which provider to query"
                    }
                },
                "required": ["paper_id", "provider"]
            }),
            category: ResearchToolCategory::Metadata,
        }
    }

    fn import_bibtex_tool() -> ResearchTool {
        ResearchTool {
            name: "import_bibtex".to_string(),
            description:
                "Parse a BibTeX string or file and import entries into the knowledge base."
                    .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "bibtex": {
                        "type": "string",
                        "description": "BibTeX content to parse"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to .bib file (alternative to bibtex content)"
                    }
                },
                "oneOf": [
                    { "required": ["bibtex"] },
                    { "required": ["file_path"] }
                ]
            }),
            category: ResearchToolCategory::Bibliography,
        }
    }

    fn export_bibtex_tool() -> ResearchTool {
        ResearchTool {
            name: "export_bibtex".to_string(),
            description: "Export citations from recent searches or knowledge base as BibTeX."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Specific paper IDs to export (default: all recent)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to write .bib file (default: return as text)"
                    }
                }
            }),
            category: ResearchToolCategory::Bibliography,
        }
    }

    fn literature_review_tool() -> ResearchTool {
        ResearchTool {
            name: "literature_review".to_string(),
            description:
                "Generate a structured literature review on a topic using academic search."
                    .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic to review"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["narrative", "systematic", "annotated", "comparative"],
                        "description": "Review style (default: narrative)"
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "Maximum papers to include (default: 50)",
                        "minimum": 5,
                        "maximum": 100
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["quick", "standard", "deep"],
                        "description": "Search depth (default: standard)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["bibtex", "apa", "mla", "chicago", "ieee"],
                        "description": "Bibliography format (default: bibtex)"
                    }
                },
                "required": ["topic"]
            }),
            category: ResearchToolCategory::Review,
        }
    }

    fn extract_paper_metadata_tool() -> ResearchTool {
        ResearchTool {
            name: "extract_paper_metadata".to_string(),
            description: "Extract structured metadata (title, authors, abstract, sections) from paper text or PDF content.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Plain text content of the paper"
                    },
                    "extract_sections": {
                        "type": "boolean",
                        "description": "Whether to detect and extract sections (default: true)"
                    },
                    "extract_references": {
                        "type": "boolean",
                        "description": "Whether to extract reference strings (default: true)"
                    }
                },
                "required": ["text"]
            }),
            category: ResearchToolCategory::Metadata,
        }
    }
}

impl Default for ResearchToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatch a research tool call.
///
/// Returns a JSON result or error message. This is the main entry point
/// for MCP tool execution.
pub fn dispatch_tool(
    tool_name: &str,
    _args: &HashMap<String, serde_json::Value>,
) -> Result<serde_json::Value, String> {
    match tool_name {
        "search_papers" => Ok(serde_json::json!({
            "status": "requires_runtime",
            "message": "search_papers requires runtime academic search providers"
        })),
        "get_paper_metadata" => Ok(serde_json::json!({
            "status": "requires_runtime",
            "message": "get_paper_metadata requires runtime academic search provider"
        })),
        "import_bibtex" => {
            if let Some(bibtex) = _args.get("bibtex").and_then(|v| v.as_str()) {
                match super::bibtex::BibParser::parse(bibtex) {
                    Ok(entries) => Ok(serde_json::json!({
                        "status": "ok",
                        "entries_parsed": entries.len(),
                        "cite_keys": entries.iter().map(|e| e.cite_key.as_str()).collect::<Vec<_>>(),
                    })),
                    Err(e) => Err(format!("BibTeX parse error: {}", e)),
                }
            } else {
                Err("Missing 'bibtex' parameter".to_string())
            }
        }
        "export_bibtex" => Ok(serde_json::json!({
            "status": "requires_runtime",
            "message": "export_bibtex requires paper data from previous search"
        })),
        "literature_review" => Ok(serde_json::json!({
            "status": "requires_runtime",
            "message": "literature_review requires runtime search providers"
        })),
        "extract_paper_metadata" => {
            if let Some(text) = _args.get("text").and_then(|v| v.as_str()) {
                let extractor = super::paper_metadata::PaperMetadataExtractor::new();
                let meta = extractor.extract(text);
                Ok(serde_json::json!({
                    "title": meta.title,
                    "abstract": meta.abstract_text,
                    "doi": meta.doi,
                    "year": meta.year,
                    "keywords": meta.keywords,
                    "sections": meta.sections.iter().map(|s| {
                        serde_json::json!({
                            "heading": s.title,
                            "type": s.section_type.display_name(),
                            "word_count": s.word_count(),
                        })
                    }).collect::<Vec<_>>(),
                    "references_count": meta.references_raw.len(),
                    "page_count": meta.page_count,
                    "extraction_confidence": meta.extraction_confidence,
                }))
            } else {
                Err("Missing 'text' parameter".to_string())
            }
        }
        _ => Err(format!("Unknown research tool: {}", tool_name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ResearchToolRegistry::new();
        assert_eq!(registry.tools().len(), 6);
    }

    #[test]
    fn test_tool_names() {
        let registry = ResearchToolRegistry::new();
        let names = registry.tool_names();
        assert!(names.contains(&"search_papers"));
        assert!(names.contains(&"get_paper_metadata"));
        assert!(names.contains(&"import_bibtex"));
        assert!(names.contains(&"export_bibtex"));
        assert!(names.contains(&"literature_review"));
        assert!(names.contains(&"extract_paper_metadata"));
    }

    #[test]
    fn test_get_tool_by_name() {
        let registry = ResearchToolRegistry::new();
        let tool = registry.get_tool("search_papers").unwrap();
        assert_eq!(tool.category, ResearchToolCategory::Search);
        assert!(tool.description.contains("academic"));
    }

    #[test]
    fn test_tools_by_category() {
        let registry = ResearchToolRegistry::new();
        let bib_tools = registry.tools_by_category(ResearchToolCategory::Bibliography);
        assert_eq!(bib_tools.len(), 2); // import + export
    }

    #[test]
    fn test_tool_category_display() {
        assert_eq!(ResearchToolCategory::Search.display_name(), "Search");
        assert_eq!(
            ResearchToolCategory::Bibliography.display_name(),
            "Bibliography"
        );
        assert_eq!(ResearchToolCategory::Review.display_name(), "Review");
        assert_eq!(ResearchToolCategory::Metadata.display_name(), "Metadata");
    }

    #[test]
    fn test_dispatch_import_bibtex() {
        let mut args = HashMap::new();
        args.insert(
            "bibtex".to_string(),
            serde_json::json!("@article{test2024, title = {Test}, year = {2024}}"),
        );
        let result = dispatch_tool("import_bibtex", &args).unwrap();
        assert_eq!(result["entries_parsed"], 1);
    }

    #[test]
    fn test_dispatch_extract_metadata() {
        let mut args = HashMap::new();
        args.insert(
            "text".to_string(),
            serde_json::json!("Test Paper Title\nDOI: 10.1234/test\n\nAbstract\nWe present a novel approach.\n\nIntroduction\nContent here."),
        );
        let result = dispatch_tool("extract_paper_metadata", &args).unwrap();
        assert!(result["title"].as_str().is_some());
    }

    #[test]
    fn test_dispatch_unknown_tool() {
        let args = HashMap::new();
        let result = dispatch_tool("nonexistent_tool", &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_search_papers_stub() {
        let mut args = HashMap::new();
        args.insert("query".to_string(), serde_json::json!("transformers"));
        let result = dispatch_tool("search_papers", &args).unwrap();
        assert_eq!(result["status"], "requires_runtime");
    }

    #[test]
    fn test_dispatch_import_bibtex_missing_param() {
        let args = HashMap::new();
        let result = dispatch_tool("import_bibtex", &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_input_schemas_valid_json() {
        let registry = ResearchToolRegistry::new();
        for tool in registry.tools() {
            assert!(
                tool.input_schema.is_object(),
                "Tool {} schema is not an object",
                tool.name
            );
        }
    }
}
