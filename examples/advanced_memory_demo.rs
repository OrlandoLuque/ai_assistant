//! Advanced memory demo — episodic, procedural, and entity memory stores.
//!
//! Demonstrates the three core memory types: episodic (time-stamped
//! experiences), procedural (learned routines), and entity (named
//! entities with relations), plus helper utilities.
//!
//! Run with: cargo run --example advanced_memory_demo --features "advanced-memory"

use std::collections::HashMap;

use ai_assistant::{
    EntityRecord, EntityRelation, EntityStore, EpisodicStore, Procedure,
    ProceduralStore, memory_cosine_similarity, new_episode,
};

fn main() {
    println!("=== Advanced Memory Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Episodic memory — recording and recalling experiences
    // -----------------------------------------------------------------------
    println!("--- Episodic Memory ---");
    let mut episodic = EpisodicStore::new(100, 0.001);
    println!("Created episodic store (capacity=100, decay=0.001)");

    // Add episodes using the helper function
    let ep1 = new_episode(
        "User asked about Rust ownership rules",
        "programming discussion",
        0.9,
        vec!["rust".to_string(), "ownership".to_string()],
        vec![0.8, 0.2, 0.5, 0.1], // simplified 4-dim embedding
    );
    let ep1_id = ep1.id.clone();
    episodic.add(ep1);

    let ep2 = new_episode(
        "User requested help with Python data analysis",
        "data science session",
        0.7,
        vec!["python".to_string(), "pandas".to_string()],
        vec![0.3, 0.9, 0.1, 0.4],
    );
    episodic.add(ep2);

    let ep3 = new_episode(
        "Explained borrow checker error messages",
        "debugging session",
        0.85,
        vec!["rust".to_string(), "debugging".to_string()],
        vec![0.7, 0.3, 0.6, 0.2],
    );
    episodic.add(ep3);

    println!("Stored {} episodes", episodic.len());

    // Recall episodes similar to a query about Rust
    let query_embedding = vec![0.75, 0.25, 0.55, 0.15];
    let recalled = episodic.recall(&query_embedding, 2);
    println!("Recalled top 2 episodes for Rust-related query:");
    for ep in &recalled {
        println!("  - [{}] \"{}\" (importance={:.1})", &ep.id[..8], ep.content, ep.importance);
    }

    // Access a specific episode by id
    if let Some(ep) = episodic.get(&ep1_id) {
        println!("Accessed episode '{}': access_count={}", &ep.id[..8], ep.access_count);
    }

    // -----------------------------------------------------------------------
    // 2. Procedural memory — learned routines
    // -----------------------------------------------------------------------
    println!("\n--- Procedural Memory ---");
    let mut procedural = ProceduralStore::new(50);
    println!("Created procedural store (capacity=50)");

    procedural.add(Procedure {
        id: "proc-001".to_string(),
        name: "Deploy Rust app".to_string(),
        condition: "deploy rust application production".to_string(),
        steps: vec![
            "Run cargo test --all-features".to_string(),
            "Build release binary with cargo build --release".to_string(),
            "Run integration tests against staging".to_string(),
            "Deploy binary to production server".to_string(),
            "Verify health checks pass".to_string(),
        ],
        success_count: 12,
        failure_count: 1,
        confidence: 0.92,
        created_from: vec!["ep-001".to_string(), "ep-015".to_string()],
        tags: vec!["deployment".to_string(), "rust".to_string()],
    });

    procedural.add(Procedure {
        id: "proc-002".to_string(),
        name: "Debug memory leak".to_string(),
        condition: "memory leak debugging performance".to_string(),
        steps: vec![
            "Enable RUST_BACKTRACE=1".to_string(),
            "Run with valgrind or heaptrack".to_string(),
            "Identify allocation hotspots".to_string(),
            "Check for Arc cycles or forgotten drops".to_string(),
        ],
        success_count: 5,
        failure_count: 2,
        confidence: 0.71,
        created_from: vec!["ep-003".to_string()],
        tags: vec!["debugging".to_string(), "performance".to_string()],
    });

    procedural.add(Procedure {
        id: "proc-003".to_string(),
        name: "Setup Python venv".to_string(),
        condition: "python virtual environment setup".to_string(),
        steps: vec![
            "python -m venv .venv".to_string(),
            "source .venv/bin/activate".to_string(),
            "pip install -r requirements.txt".to_string(),
        ],
        success_count: 20,
        failure_count: 0,
        confidence: 1.0,
        created_from: vec![],
        tags: vec!["python".to_string(), "setup".to_string()],
    });

    println!("Stored {} procedures", procedural.len());

    // Find procedures matching a context
    let matches = procedural.find_by_condition("deploy rust app to production");
    println!("Procedures matching 'deploy rust app to production':");
    for proc in &matches {
        println!("  - {} (confidence={:.2}, steps={})", proc.name, proc.confidence, proc.steps.len());
        for (i, step) in proc.steps.iter().enumerate() {
            println!("      {}. {}", i + 1, step);
        }
    }

    // Update an outcome
    let _ = procedural.update_outcome("proc-002", true);
    println!("Updated proc-002 with success outcome");

    // -----------------------------------------------------------------------
    // 3. Entity memory — named entities with relations
    // -----------------------------------------------------------------------
    println!("\n--- Entity Memory ---");
    let mut entities = EntityStore::new();
    println!("Created entity store");

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Add entities
    let _ = entities.add(EntityRecord {
        id: "ent-001".to_string(),
        name: "Rust".to_string(),
        entity_type: "programming_language".to_string(),
        attributes: {
            let mut a = HashMap::new();
            a.insert("paradigm".to_string(), serde_json::json!("systems"));
            a.insert("first_release".to_string(), serde_json::json!("2015"));
            a.insert("creator".to_string(), serde_json::json!("Graydon Hoare"));
            a
        },
        relations: vec![
            EntityRelation {
                relation_type: "influenced_by".to_string(),
                target_entity_id: "ent-003".to_string(),
                confidence: 0.95,
            },
        ],
        first_seen: now,
        last_updated: now,
        mention_count: 15,
    });

    let _ = entities.add(EntityRecord {
        id: "ent-002".to_string(),
        name: "Python".to_string(),
        entity_type: "programming_language".to_string(),
        attributes: {
            let mut a = HashMap::new();
            a.insert("paradigm".to_string(), serde_json::json!("multi-paradigm"));
            a.insert("first_release".to_string(), serde_json::json!("1991"));
            a
        },
        relations: vec![],
        first_seen: now,
        last_updated: now,
        mention_count: 8,
    });

    let _ = entities.add(EntityRecord {
        id: "ent-003".to_string(),
        name: "C++".to_string(),
        entity_type: "programming_language".to_string(),
        attributes: {
            let mut a = HashMap::new();
            a.insert("paradigm".to_string(), serde_json::json!("multi-paradigm"));
            a
        },
        relations: vec![],
        first_seen: now,
        last_updated: now,
        mention_count: 5,
    });

    println!("Stored {} entities", entities.len());

    // Look up by name (case-insensitive)
    if let Some(rust) = entities.find_by_name("rust") {
        println!("Found '{}' (type: {})", rust.name, rust.entity_type);
        println!("  Attributes: {:?}", rust.attributes);
        println!("  Relations: {}", rust.relations.len());
        for rel in &rust.relations {
            println!("    -> {} (target: {}, confidence: {:.2})", rel.relation_type, rel.target_entity_id, rel.confidence);
        }
        println!("  Mentions: {}", rust.mention_count);
    }

    // Update entity attributes
    let mut new_attrs = HashMap::new();
    new_attrs.insert("latest_edition".to_string(), serde_json::json!("2024"));
    let _ = entities.update("ent-001", new_attrs);
    println!("Updated Rust entity with latest_edition attribute");

    // -----------------------------------------------------------------------
    // 4. Cosine similarity helper
    // -----------------------------------------------------------------------
    println!("\n--- Cosine Similarity ---");
    let a = [1.0_f32, 0.0, 0.0];
    let b = [1.0_f32, 0.0, 0.0];
    let c = [0.0_f32, 1.0, 0.0];
    let d = [0.707_f32, 0.707, 0.0];

    println!("sim(a, a) = {:.4} (identical vectors)", memory_cosine_similarity(&a, &b));
    println!("sim(a, c) = {:.4} (orthogonal vectors)", memory_cosine_similarity(&a, &c));
    println!("sim(a, d) = {:.4} (45-degree angle)", memory_cosine_similarity(&a, &d));

    println!("\nAdvanced memory demo complete.");
}
