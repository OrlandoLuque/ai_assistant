//! Knowledge graph with query builder and graph algorithms.
//!
//! Run with: cargo run --example knowledge_graph --features rag
//!
//! Demonstrates creating a KnowledgeGraphStore, populating it with entities
//! and relations, querying with the Cypher-like GraphQuery builder,
//! and running graph algorithms (shortest path, PageRank).

use ai_assistant::knowledge_graph::{
    EntityType, GraphAlgorithms, GraphQuery, KnowledgeGraphConfig, KnowledgeGraphStore,
};

fn main() {
    // 1. Create an in-memory knowledge graph store
    let config = KnowledgeGraphConfig::default();
    let store = KnowledgeGraphStore::in_memory(config).expect("Failed to create store");

    // 2. Populate with entities
    let aegis_id = store
        .get_or_create_entity(
            "Aegis Dynamics",
            EntityType::Organization,
            &["Aegis".to_string()],
        )
        .unwrap();
    let anvil_id = store
        .get_or_create_entity("Anvil Aerospace", EntityType::Organization, &[])
        .unwrap();
    let sabre_id = store
        .get_or_create_entity("Sabre", EntityType::Product, &["Aegis Sabre".to_string()])
        .unwrap();
    let gladius_id = store
        .get_or_create_entity("Gladius", EntityType::Product, &[])
        .unwrap();
    let hornet_id = store
        .get_or_create_entity("Hornet", EntityType::Product, &["F7C Hornet".to_string()])
        .unwrap();
    let stanton_id = store
        .get_or_create_entity("Stanton", EntityType::Location, &[])
        .unwrap();

    println!("Created 6 entities");

    // 3. Add relations
    store
        .add_relation(
            aegis_id,
            sabre_id,
            "manufactures",
            1.0,
            Some("Aegis makes the Sabre"),
            None,
        )
        .unwrap();
    store
        .add_relation(
            aegis_id,
            gladius_id,
            "manufactures",
            1.0,
            Some("Aegis makes the Gladius"),
            None,
        )
        .unwrap();
    store
        .add_relation(
            anvil_id,
            hornet_id,
            "manufactures",
            1.0,
            Some("Anvil makes the Hornet"),
            None,
        )
        .unwrap();
    store
        .add_relation(aegis_id, stanton_id, "operates_in", 0.8, None, None)
        .unwrap();
    store
        .add_relation(anvil_id, stanton_id, "operates_in", 0.8, None, None)
        .unwrap();

    // 4. Print graph statistics
    let stats = store.get_stats().expect("stats failed");
    println!("\nGraph statistics:");
    println!("  Entities:  {}", stats.total_entities);
    println!("  Relations: {}", stats.total_relations);
    println!("  By type:   {:?}", stats.entities_by_type);

    // 5. Use the GraphQuery builder (Cypher-like syntax)
    let query = GraphQuery::new()
        .match_node("mfg", Some("organization"))
        .match_node("ship", Some("product"))
        .match_relationship("r", Some("manufactures"), "mfg", "ship")
        .return_fields(&["mfg.name", "ship.name", "r.relation_type"])
        .limit(10);

    println!("\nGenerated SQL:\n  {}", query.to_sql());

    let result = query.execute(&store).expect("query execute failed");
    println!(
        "Query returned {} rows (columns: {:?}):",
        result.rows.len(),
        result.columns
    );
    for row in &result.rows {
        println!("  {}", row.join(" | "));
    }

    // 6. Filtered query: find entities containing "Dynamics"
    let name_query = GraphQuery::new()
        .match_node("e", None)
        .where_contains("e.name", "Dynamics")
        .return_fields(&["e.name", "e.entity_type"])
        .limit(5);

    let name_result = name_query.execute(&store).expect("name query failed");
    println!(
        "\nEntities containing 'Dynamics': {} found",
        name_result.rows.len()
    );
    for row in &name_result.rows {
        println!("  {}", row.join(" | "));
    }

    // 7. Graph algorithms: shortest path (Sabre -> Hornet)
    println!("\n--- Shortest Path (Sabre -> Hornet) ---");
    match GraphAlgorithms::shortest_path(&store, sabre_id, hornet_id) {
        Ok(Some(path)) => {
            let names: Vec<String> = path
                .iter()
                .map(|id| {
                    store
                        .get_entity(*id)
                        .ok()
                        .flatten()
                        .map(|e| e.name.clone())
                        .unwrap_or_else(|| format!("id:{}", id))
                })
                .collect();
            println!("  Path ({} hops): {}", path.len() - 1, names.join(" -> "));
        }
        Ok(None) => println!("  No path found."),
        Err(e) => println!("  Error: {}", e),
    }

    // 8. Graph algorithms: PageRank
    println!("\n--- PageRank (damping=0.85, 20 iterations) ---");
    let ranks = GraphAlgorithms::page_rank(&store, 0.85, 20).expect("pagerank failed");
    let mut sorted: Vec<(i64, f64)> = ranks.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (id, score) in &sorted {
        let name = store
            .get_entity(*id)
            .ok()
            .flatten()
            .map(|e| e.name.clone())
            .unwrap_or_else(|| format!("id:{}", id));
        println!("  {:<20} {:.6}", name, score);
    }

    // 9. Connected components
    let components = GraphAlgorithms::connected_components(&store).expect("components failed");
    let unique_components: std::collections::HashSet<i64> = components.values().copied().collect();
    println!("\n--- Connected Components ---");
    println!("  {} component(s) found", unique_components.len());
}
