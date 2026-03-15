//! RAG pipeline example with knowledge graph integration.
//!
//! Run with: cargo run --example rag_pipeline --features rag
//!
//! Demonstrates building a KnowledgeGraph, indexing documents with a
//! pattern-based entity extractor, and wiring it into a RagPipeline.

use ai_assistant::{
    KGEntityType, KnowledgeGraphBuilder, KnowledgeGraphConfig, LlmCallback, PatternEntityExtractor,
    RagPipeline, RagPipelineConfig, RagPipelineError, RagPipelineResult, RetrievalCallback,
    RetrievedChunk,
};

// -- Minimal LLM stub (returns the prompt back) --

struct StubLlm;

impl LlmCallback for StubLlm {
    fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String, String> {
        Ok(format!(
            "(LLM would process: {})",
            &prompt[..prompt.len().min(80)]
        ))
    }
    fn model_name(&self) -> &str {
        "stub"
    }
}

// -- Minimal retrieval stub --

struct StubRetrieval;

impl RetrievalCallback for StubRetrieval {
    fn keyword_search(&self, query: &str, _limit: usize) -> Result<Vec<RetrievedChunk>, String> {
        Ok(vec![RetrievedChunk {
            chunk_id: "chunk-1".into(),
            content: format!("Result for: {}", query),
            source: "demo-doc".into(),
            section: None,
            score: 0.9,
            keyword_score: Some(0.9),
            semantic_score: None,
            token_count: 10,
            position: None,
            metadata: Default::default(),
        }])
    }
    fn semantic_search(
        &self,
        _embedding: &[f32],
        _limit: usize,
    ) -> Result<Vec<RetrievedChunk>, String> {
        Ok(vec![])
    }
    fn get_chunk(&self, _id: &str) -> Result<Option<RetrievedChunk>, String> {
        Ok(None)
    }
}

fn main() {
    // 1. Build an in-memory knowledge graph with known entities
    let graph = KnowledgeGraphBuilder::new()
        .with_config({
            let mut c = KnowledgeGraphConfig::default();
            c.max_traversal_depth = 3;
            c
        })
        .add_entity("Aegis Dynamics", KGEntityType::Organization)
        .add_entity("Sabre", KGEntityType::Product)
        .add_entity("Gladius", KGEntityType::Product)
        .add_alias("Aegis", "Aegis Dynamics")
        .build_in_memory()
        .expect("Failed to build knowledge graph");

    // 2. Create a pattern-based extractor and index a document
    let extractor = PatternEntityExtractor::new()
        .add_entity("Aegis Dynamics", KGEntityType::Organization)
        .add_entity("Sabre", KGEntityType::Product)
        .add_entity("Gladius", KGEntityType::Product);

    let mut graph = graph;
    let result = graph.index_document(
        "ships-overview",
        "Aegis Dynamics manufactures the Sabre and the Gladius. \
         The Sabre is a stealth fighter. The Gladius is a light fighter.",
        &extractor,
    );
    match result {
        Ok(r) => println!(
            "Indexed: {} chunks, {} entities, {} relations",
            r.chunks_processed, r.entities_extracted, r.relations_extracted
        ),
        Err(e) => println!("Indexing error: {}", e),
    }

    // 3. Print graph stats
    let stats = graph.stats().expect("stats failed");
    println!(
        "Graph: {} entities, {} relations, {} chunks",
        stats.total_entities, stats.total_relations, stats.total_chunks
    );

    // 4. Export the graph to JSON
    let json = graph.export_json().expect("export failed");
    println!(
        "Exported JSON keys: {:?}",
        json.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    // 5. Wire up a RAG pipeline with the graph callback
    let graph_cb = graph.as_graph_callback(&extractor);
    let llm = StubLlm;
    let retrieval = StubRetrieval;

    let config = RagPipelineConfig::default();
    let mut pipeline = RagPipeline::with_config(config);

    let result: Result<RagPipelineResult, RagPipelineError> = pipeline.process(
        "What ships does Aegis make?",
        &llm,
        None,
        &retrieval,
        Some(&graph_cb),
    );

    match result {
        Ok(r) => {
            println!("\nPipeline result:");
            println!("  Chunks retrieved: {}", r.chunks.len());
            println!("  Context length:   {} chars", r.context.len());
        }
        Err(e) => println!("Pipeline error: {:?}", e),
    }
}
