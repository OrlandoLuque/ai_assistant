//! Vector search example using the in-memory vector database.
//!
//! Run with: cargo run --example vector_search --features embeddings
//!
//! No external services required.

use ai_assistant::{DistanceMetric, InMemoryVectorDb, VectorDb, VectorDbConfig};

fn main() {
    // Configure vector database
    let mut config = VectorDbConfig::default();
    config.dimensions = 3;
    config.collection_name = "demo".to_string();

    let mut db = InMemoryVectorDb::new(config);

    // Insert some vectors with metadata
    db.insert(
        "doc-1",
        vec![1.0, 0.0, 0.0],
        serde_json::json!({"title": "Rust Programming", "category": "tech"}),
    )
    .expect("insert failed");

    db.insert(
        "doc-2",
        vec![0.0, 1.0, 0.0],
        serde_json::json!({"title": "Machine Learning", "category": "ai"}),
    )
    .expect("insert failed");

    db.insert(
        "doc-3",
        vec![0.9, 0.1, 0.0],
        serde_json::json!({"title": "Systems Programming", "category": "tech"}),
    )
    .expect("insert failed");

    println!("Inserted {} vectors", db.count());

    // Search for similar vectors (cosine similarity)
    let query = vec![1.0, 0.0, 0.0]; // Should match "Rust Programming" best
    let results = db.search(&query, 2, None).expect("search failed");

    println!("\nTop 2 results for query {:?}:", query);
    for result in &results {
        let stored = db.get(&result.id).expect("get failed").unwrap();
        println!(
            "  {} (score: {:.3}) - metadata: {}",
            result.id, result.score, stored.metadata
        );
    }

    // Demonstrate distance metrics
    let a = [1.0, 0.0, 0.0];
    let b = [0.9, 0.1, 0.0];
    println!("\nDistance metrics between {:?} and {:?}:", a, b);
    println!(
        "  Cosine:    {:.4}",
        DistanceMetric::Cosine.calculate(&a, &b)
    );
    println!(
        "  Euclidean: {:.4}",
        DistanceMetric::Euclidean.calculate(&a, &b)
    );
    println!(
        "  DotProduct: {:.4}",
        DistanceMetric::DotProduct.calculate(&a, &b)
    );

    // Health check
    println!("\nDB healthy: {}", db.health_check().unwrap_or(false));
    println!("Backend: {}", db.backend_info().name);
}
