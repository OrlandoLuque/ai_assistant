# Graph RAG Guide

## What is Graph RAG?

**Graph RAG** (Graph-based Retrieval-Augmented Generation) enhances traditional RAG by building and querying a **knowledge graph** from your documents. Instead of just searching for text chunks, it:

1. Extracts **entities** (people, products, organizations, concepts)
2. Identifies **relationships** between entities
3. Traverses the graph to find related information
4. Provides context that spans multiple documents through connections

```
Traditional RAG:
  Query -> Search Chunks -> Return Matching Text

Graph RAG:
  Query -> Extract Entities -> Traverse Graph -> Find Related Entities -> Gather Context
```

## Why Use Graph RAG?

### Strengths
- **Multi-hop reasoning**: Find connections across documents (e.g., "What ships share the same manufacturer?")
- **Relationship queries**: Answer questions about how things relate
- **Contextual understanding**: The Aurora MR is connected to RSI, which is connected to other ships
- **Better for complex domains**: Ideal for knowledge bases with many interconnected entities

### When to Use
- Complex domains with many related entities
- Questions involving relationships ("Which ships are made by RSI?")
- Multi-step queries that need to gather info from multiple sources
- Domains where context from relationships improves answers

### When NOT to Use
- Simple keyword/factual lookups
- Small document collections
- Time-sensitive applications (graph building adds latency)
- When relationships aren't important

## How Graph RAG Works

### Phase 1: Knowledge Graph Construction (Indexing Time)

```
Document -> Entity Extraction -> Relationship Extraction -> Graph Storage
```

#### 1.1 Entity Extraction

The system identifies named entities in your documents:

```rust
use ai_assistant::rag_methods::{GraphRagRetriever, GraphRagConfig};

let config = GraphRagConfig {
    max_depth: 2,
    max_entities: 50,
    entity_types: vec![
        "SHIP".into(),
        "MANUFACTURER".into(),
        "COMPONENT".into(),
        "LOCATION".into(),
    ],
};

let retriever = GraphRagRetriever::new(config);

// Extract entities from text
let result = retriever.extract_entities(
    "The Aurora MR is manufactured by Roberts Space Industries (RSI).
     It has a quantum drive and twin S1 weapon hardpoints.",
    &llm,
)?;

// Result:
// Entity { name: "Aurora MR", type: "SHIP" }
// Entity { name: "Roberts Space Industries", type: "MANUFACTURER" }
// Entity { name: "RSI", type: "MANUFACTURER" }
// Entity { name: "quantum drive", type: "COMPONENT" }
```

#### 1.2 Relationship Extraction

Connections between entities are identified:

```
Aurora MR --[manufactured_by]--> RSI
Aurora MR --[has_component]--> quantum drive
Aurora MR --[has_component]--> S1 weapon hardpoints
RSI --[also_known_as]--> Roberts Space Industries
```

Relationship types include:
- `manufactured_by` - Who makes it
- `has_component` - Parts and equipment
- `located_at` - Physical locations
- `part_of` - Hierarchical relationships
- `similar_to` - Related entities
- `derived_from` - Variants and versions

#### 1.3 Graph Storage

Entities and relationships are stored in a graph structure:

```rust
pub trait GraphDatabase {
    /// Add entity to graph
    fn add_entity(&mut self, entity: &Entity) -> Result<(), String>;

    /// Add relationship between entities
    fn add_relationship(&mut self, relationship: &Relationship) -> Result<(), String>;

    /// Find entities matching text
    fn find_entities(&self, text: &str) -> Result<Vec<Entity>, String>;

    /// Get relationships for an entity (with depth limit)
    fn get_relationships(&self, entity: &str, max_depth: usize) -> Result<Vec<Relationship>, String>;

    /// Get entities related to a given entity
    fn get_related_entities(&self, entity: &str, max_depth: usize) -> Result<Vec<Entity>, String>;
}
```

### Phase 2: Query Processing (Runtime)

```
Query -> Entity Recognition -> Graph Traversal -> Context Gathering -> Response
```

#### 2.1 Entity Recognition in Query

When a user asks a question, entities are extracted from the query:

```
Query: "What weapons does the Aurora MR have?"

Extracted entities: ["Aurora MR"]
Intent: Find weapon components
```

#### 2.2 Graph Traversal

Starting from recognized entities, the graph is traversed:

```
Step 1: Find "Aurora MR" node
Step 2: Get relationships where Aurora MR is the source
Step 3: Filter by relationship type ("has_component", "has_weapon")
Step 4: Optionally expand to related entities (depth > 1)
```

Configuration controls traversal:

```rust
pub struct GraphRagConfig {
    /// Maximum traversal depth (1 = direct connections only)
    pub max_depth: usize,

    /// Maximum entities to extract per query
    pub max_entities: usize,

    /// Entity types to look for
    pub entity_types: Vec<String>,
}
```

#### 2.3 Context Gathering

Related information is collected from:
1. Entity descriptions
2. Relationship descriptions
3. Source chunks where relationships were found
4. Transitively related entities (if depth > 1)

```
Aurora MR
├── manufactured_by: RSI
│   └── other RSI ships: Constellation, Polaris...
├── has_component: quantum drive (QD-S1)
├── has_weapon: 2x S1 hardpoints
│   └── compatible weapons: Bulldog, Badger...
└── similar_to: Aurora LN, Aurora CL
```

## Implementation with KnowledgeGraph Module

The `ai_assistant` crate includes a complete `knowledge_graph` module for building and querying knowledge graphs. This module provides:

- **SQLite-backed storage** with full-text search (FTS5)
- **Thread-safe operations** via `Mutex<Connection>`
- **Entity extraction** via LLM or pattern matching
- **Multi-hop graph traversal**
- **RagPipeline integration** via `GraphCallback` trait

### Quick Start with KnowledgeGraph

```rust
use ai_assistant::{
    KnowledgeGraph, KnowledgeGraphConfig, KnowledgeGraphBuilder,
    PatternEntityExtractor, KGEntityType, KGEntityExtractor,
};

// 1. Create a knowledge graph with pre-configured Star Citizen entities
let graph = KnowledgeGraphBuilder::new()
    .with_star_citizen_entities()  // Pre-populated manufacturers
    .add_entity("Aurora MR", KGEntityType::Product)
    .add_entity("Stanton", KGEntityType::Location)
    .build("knowledge.db")?;

// 2. Create an entity extractor (pattern-based, no LLM needed)
let extractor = PatternEntityExtractor::new()
    .add_entity("Aegis", KGEntityType::Organization)
    .add_entity("Sabre", KGEntityType::Product)
    .add_alias("Aegis Dynamics", "Aegis");

// 3. Index a document
let result = graph.index_document(
    "ships_doc",
    "Aegis Dynamics manufactures the Sabre fighter, a stealth spacecraft.",
    &extractor,
)?;
println!("Extracted {} entities, {} relations",
    result.entities_extracted, result.relations_extracted);

// 4. Query the graph
let query_result = graph.query("What does Aegis make?", &extractor)?;
for entity in &query_result.entities_found {
    println!("Found: {} ({})", entity.name, entity.entity_type.as_str());
}
for chunk in &query_result.chunks {
    println!("Context: {}", chunk.content);
}
```

### Using with RagPipeline (GraphCallback)

```rust
use ai_assistant::{KnowledgeGraph, KnowledgeGraphCallback};

// Create graph and extractor
let graph = KnowledgeGraph::open("knowledge.db", KnowledgeGraphConfig::default())?;
let extractor = PatternEntityExtractor::new()...;

// Get GraphCallback for RagPipeline integration
let callback = graph.as_graph_callback(&extractor);

// Use with RagPipeline
pipeline.with_graph_callback(callback);
```

### LLM-Based Entity Extraction

For higher accuracy, use the LLM-based extractor:

```rust
use ai_assistant::LlmEntityExtractor;

// LLM function: (system_prompt, user_prompt) -> response
let llm_fn = |system: &str, user: &str| -> Result<String> {
    // Call your LLM here
    llm_client.generate(system, user)
};

let extractor = LlmEntityExtractor::new(llm_fn)
    .with_entity_types(vec![
        KGEntityType::Organization,
        KGEntityType::Product,
    ]);

// The extractor uses a structured prompt to extract entities and relations
graph.index_document("doc_id", content, &extractor)?;
```

### Configuration Options

```rust
let config = KnowledgeGraphConfig {
    max_traversal_depth: 2,        // How deep to traverse relations
    max_entities_per_query: 50,    // Max entities returned per query
    max_chunks_per_entity: 5,      // Max chunks per entity
    min_relation_confidence: 0.5,  // Filter low-confidence relations
    chunk_size: 1000,              // Document chunk size
    chunk_overlap: 200,            // Overlap between chunks
    resolve_aliases: true,         // Enable alias resolution
    ..Default::default()
};
```

## Legacy Implementation Example

### Basic Graph RAG Setup

```rust
use ai_assistant::rag_tiers::{RagConfig, RagTier};
use ai_assistant::rag_methods::{GraphRagRetriever, GraphRagConfig, Entity, Relationship};

// 1. Configure RAG with Graph tier
let config = RagConfig::with_tier(RagTier::Graph)
    .with_max_chunks(20)
    .with_max_calls(10);

// Check requirements
let reqs = config.check_requirements();
// Will include RagRequirement::GraphDatabase

// 2. Create graph retriever
let graph_config = GraphRagConfig {
    max_depth: 2,
    max_entities: 30,
    entity_types: vec!["SHIP".into(), "MANUFACTURER".into()],
};
let retriever = GraphRagRetriever::new(graph_config);

// 3. During indexing: extract entities from each document
for doc in documents {
    let entities = retriever.extract_entities(&doc.content, &llm)?;

    for entity in entities.result {
        graph_db.add_entity(&entity)?;
    }

    // Extract relationships (would need additional LLM call)
    let relationships = extract_relationships(&doc.content, &entities.result, &llm)?;

    for rel in relationships {
        graph_db.add_relationship(&rel)?;
    }
}

// 4. At query time: use graph for retrieval
fn query_with_graph(query: &str, graph_db: &impl GraphDatabase, llm: &impl LlmGenerate) -> Vec<String> {
    // Extract entities from query
    let query_entities = retriever.extract_entities(query, llm)?;

    let mut context_chunks = Vec::new();

    for entity in query_entities.result {
        // Find in graph
        if let Ok(found) = graph_db.find_entities(&entity.name) {
            for e in found {
                // Get related entities
                let related = graph_db.get_related_entities(&e.name, 2)?;

                // Get relationships
                let rels = graph_db.get_relationships(&e.name, 2)?;

                // Build context from relationships
                for rel in rels {
                    if let Some(chunk) = &rel.source_chunk {
                        context_chunks.push(chunk.clone());
                    }
                }
            }
        }
    }

    context_chunks
}
```

### Custom Entity Extraction Prompt

For better domain-specific extraction:

```rust
let custom_prompt = r#"
Extract entities from this Star Citizen documentation.

Entity types to find:
- SHIP: Spacecraft names (Aurora, Constellation, etc.)
- MANUFACTURER: Companies (RSI, Aegis, Drake, etc.)
- COMPONENT: Ship parts (quantum drive, shields, weapons)
- LOCATION: Planets, stations, systems

Text:
{text}

Format: TYPE: Name (one per line)
Example: SHIP: Aurora MR
"#;
```

### Relationship Extraction

After entities are found, extract relationships:

```rust
let relationship_prompt = r#"
Given these entities found in the text:
{entities}

Text:
{text}

List relationships between entities.
Format: ENTITY1 | RELATIONSHIP_TYPE | ENTITY2
Example: Aurora MR | manufactured_by | RSI

Valid relationship types:
- manufactured_by
- has_component
- variant_of
- located_at
- similar_to
"#;
```

## Graph RAG vs Other RAG Tiers

| Feature | Semantic RAG | Graph RAG |
|---------|--------------|-----------|
| Query type | Direct matching | Relationship-aware |
| Multi-hop | No | Yes |
| Setup complexity | Low | High |
| Indexing time | Fast | Slower (entity extraction) |
| Query latency | ~50ms | ~200ms+ |
| Best for | Simple queries | Complex domains |

## Debugging Graph RAG

Use the debug system to trace graph operations:

```rust
use ai_assistant::rag_debug::{RagDebugLogger, RagDebugLevel, RagDebugStep};

let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

// Log graph traversal
logger.log_step(RagDebugStep::GraphTraversal {
    start_entities: vec!["Aurora MR".into()],
    traversal_depth: 2,
    nodes_visited: 15,
    relationships_found: 8,
    duration_ms: 45,
});
```

Debug output shows:
```
[RAG] Graph: 15 nodes, 8 relationships (45ms)
```

## Best Practices

### 1. Entity Type Design
- Define clear, non-overlapping entity types
- Use consistent naming (SHIP vs Ship vs ship)
- Include common variants (RSI, Roberts Space Industries)

### 2. Relationship Granularity
- Keep relationship types meaningful but not too specific
- Use inverse relationships (manufactured_by ↔ manufactures)
- Store source context for later retrieval

### 3. Graph Maintenance
- Re-index when documents change significantly
- Merge duplicate entities (RSI = Roberts Space Industries)
- Prune orphan nodes periodically

### 4. Query Optimization
- Limit traversal depth (usually 2-3 is enough)
- Cache frequently accessed subgraphs
- Pre-compute common paths

### 5. Hybrid Approach
Graph RAG works best combined with traditional retrieval:

```rust
// 1. Traditional semantic search for direct matches
let semantic_results = semantic_search(query)?;

// 2. Graph traversal for related context
let graph_results = graph_search(query)?;

// 3. Fuse results using RRF
let fusion = RrfFusion::new();
let combined = fusion.fuse(vec![semantic_results, graph_results]);
```

## Entity Types for Common Domains

### Software Documentation
```
CODE_ELEMENT: Classes, functions, modules
CONFIGURATION: Config keys, environment variables
DEPENDENCY: External packages
ERROR: Error codes, exceptions
```

### Product Catalogs
```
PRODUCT: Individual products
CATEGORY: Product categories
BRAND: Manufacturers
FEATURE: Product features
```

### Gaming/Virtual Worlds
```
CHARACTER: NPCs, players
ITEM: Equipment, consumables
LOCATION: Areas, zones
QUEST: Missions, objectives
```

## Limitations

1. **Entity extraction quality**: Depends on LLM accuracy
2. **Scale**: Large graphs can be slow to traverse
3. **Maintenance**: Graphs need updating as content changes
4. **Cold start**: Need sufficient data to build meaningful graph
5. **Cost**: Additional LLM calls for entity/relationship extraction

## Further Reading

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Reference implementation
- [Knowledge Graphs for RAG](https://arxiv.org/abs/2403.xxxxx) - Academic paper
- [Neo4j + LLM Guide](https://neo4j.com/labs/genai/) - Production graph DB integration
