//! Example: vector_lancedb_demo -- Demonstrates the LanceDB vector database backend.
//!
//! Run with: cargo run --example vector_lancedb_demo --features "vector-lancedb"
//!
//! This example showcases LanceVectorDb for embedded, persistent vector storage:
//! creating a database, inserting vectors, searching by similarity, and
//! batch operations. Uses a temporary directory — no external services needed.

use ai_assistant::{LanceVectorDb, VectorDb, VectorDbConfig};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- LanceDB Vector Database Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Configuration
    // ------------------------------------------------------------------
    println!("--- 1. VectorDbConfig ---\n");

    let config = VectorDbConfig {
        dimensions: 128,
        collection_name: "demo_vectors".to_string(),
        ..Default::default()
    };

    println!("  dimensions:      {}", config.dimensions);
    println!("  collection_name: {}", config.collection_name);

    // ------------------------------------------------------------------
    // 2. Create database in a temp directory
    // ------------------------------------------------------------------
    println!("\n--- 2. Create LanceVectorDb ---\n");

    let tmp_dir = std::env::temp_dir().join("lance_demo");
    let path = tmp_dir.to_string_lossy().to_string();
    println!("  Database path: {}", path);

    let mut db = match LanceVectorDb::new(&path, config) {
        Ok(db) => {
            println!("  Database created successfully");
            db
        }
        Err(e) => {
            println!("  Error creating database: {}", e);
            return;
        }
    };

    // ------------------------------------------------------------------
    // 3. Insert vectors
    // ------------------------------------------------------------------
    println!("\n--- 3. Insert Vectors ---\n");

    let topics = [
        ("doc_rust", "Rust programming language"),
        ("doc_python", "Python data science"),
        ("doc_ml", "Machine learning models"),
        ("doc_web", "Web development frameworks"),
        ("doc_db", "Database optimization"),
    ];

    for (id, topic) in &topics {
        // Generate a deterministic embedding based on the topic string
        let embedding: Vec<f32> = (0..128)
            .map(|d| {
                let seed = topic.as_bytes().iter().map(|&b| b as f32).sum::<f32>();
                ((seed + d as f32) * 0.01).sin()
            })
            .collect();

        let metadata = serde_json::json!({ "topic": topic });
        match db.insert(id, embedding, metadata) {
            Ok(()) => println!("  Inserted: {} ({})", id, topic),
            Err(e) => println!("  Error inserting {}: {}", id, e),
        }
    }

    println!("  Total vectors: {}", db.count());

    // ------------------------------------------------------------------
    // 4. Search by similarity
    // ------------------------------------------------------------------
    println!("\n--- 4. Similarity Search ---\n");

    // Create a query vector similar to "Rust programming language"
    let query: Vec<f32> = (0..128)
        .map(|d| {
            let seed = "Rust programming language"
                .as_bytes()
                .iter()
                .map(|&b| b as f32)
                .sum::<f32>();
            ((seed + d as f32) * 0.01).sin() + 0.001 // slight offset
        })
        .collect();

    match db.search(&query, 3, None) {
        Ok(results) => {
            println!("  Query: 'Rust programming' (top 3 results):");
            for (i, result) in results.iter().enumerate() {
                println!(
                    "    {}. id={}, score={:.4}, metadata={}",
                    i + 1,
                    result.id,
                    result.score,
                    result.metadata
                );
            }
        }
        Err(e) => println!("  Search error: {}", e),
    }

    // ------------------------------------------------------------------
    // 5. Get by ID
    // ------------------------------------------------------------------
    println!("\n--- 5. Get by ID ---\n");

    match db.get("doc_ml") {
        Ok(Some(stored)) => {
            println!("  Found: id={}", stored.id);
            println!("  Vector dims: {}", stored.vector.len());
            println!("  Metadata: {}", stored.metadata);
        }
        Ok(None) => println!("  Not found"),
        Err(e) => println!("  Error: {}", e),
    }

    // ------------------------------------------------------------------
    // 6. Batch insert
    // ------------------------------------------------------------------
    println!("\n--- 6. Batch Insert ---\n");

    let batch: Vec<(String, Vec<f32>, serde_json::Value)> = (0..10)
        .map(|i| {
            let id = format!("batch_{}", i);
            let vec: Vec<f32> = (0..128).map(|d| ((i * 128 + d) as f32 * 0.01).sin()).collect();
            let meta = serde_json::json!({ "batch": true, "index": i });
            (id, vec, meta)
        })
        .collect();

    match db.insert_batch(batch) {
        Ok(count) => println!("  Batch inserted: {} vectors", count),
        Err(e) => println!("  Batch error: {}", e),
    }

    println!("  Total vectors after batch: {}", db.count());

    // ------------------------------------------------------------------
    // 7. Delete
    // ------------------------------------------------------------------
    println!("\n--- 7. Delete ---\n");

    match db.delete("doc_web") {
        Ok(true) => println!("  Deleted 'doc_web'"),
        Ok(false) => println!("  'doc_web' not found"),
        Err(e) => println!("  Delete error: {}", e),
    }

    println!("  Total vectors after delete: {}", db.count());

    // ------------------------------------------------------------------
    // 8. Backend info
    // ------------------------------------------------------------------
    println!("\n--- 8. Backend Info ---\n");

    let info = db.backend_info();
    println!("  Name:        {}", info.name);
    println!("  Persistent:  {}", info.persistent);
    println!("  Distributed: {}", info.distributed);
    println!("  Max vectors: {:?}", info.max_vectors);

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  LanceDB demo complete.");
    println!("  Capabilities: insert, batch insert, similarity search,");
    println!("    get by ID, delete, persistent storage.");
    println!("==========================================================");
}
