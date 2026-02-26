//! Example: vector_pgvector_demo -- Demonstrates pgvector SQL generation.
//!
//! Run with: cargo run --example vector_pgvector_demo --features vector-pgvector
//!
//! This example showcases PgVectorConfig, PgVectorDb SQL generation for
//! table creation, HNSW indexing, upsert, cosine/L2 search, and vector formatting.
//! No PostgreSQL connection is required — all operations generate SQL strings.

use ai_assistant::{PgVectorConfig, PgVectorDb};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- pgvector SQL Generation Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Configuration
    // ------------------------------------------------------------------
    println!("--- 1. PgVectorConfig ---\n");

    let config = PgVectorConfig {
        connection_string: "host=localhost dbname=vectors user=postgres password=secret".to_string(),
        table_name: "embeddings".to_string(),
        dimensions: 384,
    };

    println!("  connection_string: {}", config.connection_string);
    println!("  table_name:        {}", config.table_name);
    println!("  dimensions:        {}", config.dimensions);

    // Custom config for a different use case
    let config_1536 = PgVectorConfig {
        connection_string: "host=db.example.com dbname=ai port=5432".to_string(),
        table_name: "openai_embeddings".to_string(),
        dimensions: 1536,
    };
    println!("\n  Custom config: table={}, dims={}", config_1536.table_name, config_1536.dimensions);

    let db = PgVectorDb::new(config);

    // ------------------------------------------------------------------
    // 2. Schema SQL
    // ------------------------------------------------------------------
    println!("\n--- 2. Schema Creation SQL ---\n");

    println!("  -- Enable extension:");
    println!("  {}\n", db.create_extension_sql());

    println!("  -- Create table:");
    println!("  {}\n", db.create_table_sql());

    println!("  -- Create HNSW index (cosine):");
    println!("  {}", db.create_index_sql());

    // ------------------------------------------------------------------
    // 3. CRUD SQL
    // ------------------------------------------------------------------
    println!("\n--- 3. CRUD Operations SQL ---\n");

    println!("  -- Upsert:");
    println!("  {}\n", db.upsert_sql());

    println!("  -- Get by ID:");
    println!("  {}\n", db.get_sql());

    println!("  -- Delete by ID:");
    println!("  {}\n", db.delete_sql());

    println!("  -- Count:");
    println!("  {}\n", db.count_sql());

    println!("  -- Clear (truncate):");
    println!("  {}", db.clear_sql());

    // ------------------------------------------------------------------
    // 4. Search SQL
    // ------------------------------------------------------------------
    println!("\n--- 4. Vector Search SQL ---\n");

    println!("  -- Cosine similarity search (<=>):");
    println!("  {}\n", db.search_sql());

    println!("  -- L2 distance search (<->):");
    println!("  {}", db.search_l2_sql());

    // ------------------------------------------------------------------
    // 5. Vector Formatting
    // ------------------------------------------------------------------
    println!("\n--- 5. Vector Formatting ---\n");

    let embedding = vec![0.1, 0.25, -0.5, 0.0, 0.75];
    let formatted = PgVectorDb::format_vector(&embedding);
    println!("  Input:     {:?}", embedding);
    println!("  Formatted: {}", formatted);

    // Parse back
    let parsed = PgVectorDb::parse_vector(&formatted);
    println!("  Parsed:    {:?}", parsed);
    println!("  Round-trip: {}", embedding == parsed);

    // Larger embedding
    let large_vec: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
    let large_formatted = PgVectorDb::format_vector(&large_vec);
    println!("\n  384-dim vector: {} chars formatted", large_formatted.len());
    println!("  First 60 chars: {}...", &large_formatted[..60]);

    // ------------------------------------------------------------------
    // 6. Integration Pattern
    // ------------------------------------------------------------------
    println!("\n--- 6. Integration Pattern ---\n");

    println!("  // Step 1: Generate SQL from PgVectorDb");
    println!("  let db = PgVectorDb::new(config);");
    println!("  let sql = db.upsert_sql();");
    println!("  let vec_str = PgVectorDb::format_vector(&embedding);");
    println!();
    println!("  // Step 2: Execute with your preferred PostgreSQL client");
    println!("  // e.g., tokio-postgres, sqlx, diesel, or ureq + REST API");
    println!("  // client.execute(&sql, &[&id, &vec_str, &metadata])?;");
    println!();
    println!("  // This design avoids coupling to any specific async runtime");
    println!("  // or database driver, keeping the library lightweight.");

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  pgvector demo complete.");
    println!("  Capabilities: schema generation, HNSW indexing,");
    println!("    cosine/L2 search, vector formatting.");
    println!("==========================================================");
}
