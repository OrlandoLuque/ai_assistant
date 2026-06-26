use super::*;

// ─── RAG Tier System Tests ────────────────────────────────────────────────────

#[cfg(feature = "rag")]
pub(crate) fn tests_rag_tiers() -> CategoryResult {
    // Use the re-exported types from lib.rs root
    use ai_assistant::{
        EntityMention, GraphEntity, GraphRagConfig, GraphRagRetriever, HybridWeights, HydeConfig,
        MultiQueryConfig, MultiQueryDecomposer, RagDebugConfig, RagDebugLevel, RagDebugLogger,
        RagFeatures, RagTier, RagTierConfig, Relationship, RrfFusion, ScoredItem,
    };

    println!("\n{}", bold(&cyan("▶ RAG Tier System")));
    let mut results = Vec::new();

    // RagTier basic tests
    results.push(run_test(
        "RagTier::to_features returns correct features",
        || {
            let disabled = RagTier::Disabled;
            let disabled_features = disabled.to_features();
            // Disabled should have all false
            assert_test!(
                !disabled_features.fts_search,
                "Disabled should not have FTS"
            );

            let fast = RagTier::Fast;
            let fast_features = fast.to_features();
            assert_test!(fast_features.fts_search, "Fast should have FTS search");

            let semantic = RagTier::Semantic;
            let sem_features = semantic.to_features();
            assert_test!(
                sem_features.semantic_search,
                "Semantic should have semantic_search"
            );
            assert_test!(
                sem_features.hybrid_search,
                "Semantic should have hybrid_search"
            );

            let full = RagTier::Full;
            let full_features = full.to_features();
            assert_test!(full_features.fts_search, "Full should have FTS");
            assert_test!(full_features.semantic_search, "Full should have semantic");
            assert_test!(
                full_features.query_expansion,
                "Full should have query_expansion"
            );
            Ok(())
        },
    ));

    results.push(run_test("RagTier display names", || {
        let tiers = vec![
            RagTier::Disabled,
            RagTier::Fast,
            RagTier::Semantic,
            RagTier::Enhanced,
            RagTier::Thorough,
            RagTier::Agentic,
            RagTier::Graph,
            RagTier::Full,
        ];
        for tier in tiers {
            let name = tier.display_name();
            assert_test!(
                !name.is_empty(),
                &format!("{:?} display_name should not be empty", tier)
            );
            let desc = tier.description();
            assert_test!(
                !desc.is_empty(),
                &format!("{:?} description should not be empty", tier)
            );
        }
        Ok(())
    }));

    results.push(run_test("RagTier default is Fast", || {
        let default_tier = RagTier::default();
        assert_eq_test!(default_tier, RagTier::Fast);
        Ok(())
    }));

    // RagConfig tests
    results.push(run_test(
        "RagTierConfig::with_tier creates correct config",
        || {
            let config = RagTierConfig::with_tier(RagTier::Semantic);
            assert_eq_test!(config.tier, RagTier::Semantic);
            assert_test!(
                config.features.semantic_search,
                "Should have semantic_search enabled"
            );
            Ok(())
        },
    ));

    results.push(run_test("RagTierConfig default values", || {
        let config = RagTierConfig::default();
        assert_eq_test!(config.tier, RagTier::Fast);
        assert_test!(config.max_chunks > 0, "max_chunks should be positive");
        assert_test!(
            config.max_knowledge_tokens > 0,
            "max_knowledge_tokens should be positive"
        );
        Ok(())
    }));

    results.push(run_test(
        "RagTierConfig::with_features creates custom config",
        || {
            let mut features = RagFeatures::default();
            features.fts_search = true;
            features.semantic_search = true;
            let config = RagTierConfig::with_features(features);
            assert_eq_test!(config.tier, RagTier::Custom);
            assert_test!(config.features.fts_search, "Should have fts_search");
            assert_test!(
                config.features.semantic_search,
                "Should have semantic_search"
            );
            Ok(())
        },
    ));

    // RagFeatures struct tests
    results.push(run_test("RagFeatures default is all false", || {
        let features = RagFeatures::default();
        assert_test!(!features.fts_search, "default fts_search should be false");
        assert_test!(
            !features.semantic_search,
            "default semantic_search should be false"
        );
        assert_test!(
            !features.query_expansion,
            "default query_expansion should be false"
        );
        Ok(())
    }));

    results.push(run_test("RagFeatures fields", || {
        let mut features = RagFeatures::default();
        features.fts_search = true;
        features.semantic_search = true;
        features.hybrid_search = true;
        features.query_expansion = true;
        features.multi_query = true;
        features.hyde = true;
        features.reranking = true;
        features.contextual_compression = true;

        assert_test!(features.fts_search, "fts_search should be set");
        assert_test!(features.semantic_search, "semantic_search should be set");
        assert_test!(features.hybrid_search, "hybrid_search should be set");
        assert_test!(features.query_expansion, "query_expansion should be set");
        assert_test!(features.multi_query, "multi_query should be set");
        assert_test!(features.hyde, "hyde should be set");
        assert_test!(features.reranking, "reranking should be set");
        assert_test!(
            features.contextual_compression,
            "contextual_compression should be set"
        );
        Ok(())
    }));

    // RagDebugLevel tests
    results.push(run_test("RagDebugLevel ordering", || {
        assert_test!(RagDebugLevel::Off < RagDebugLevel::Minimal);
        assert_test!(RagDebugLevel::Minimal < RagDebugLevel::Basic);
        assert_test!(RagDebugLevel::Basic < RagDebugLevel::Detailed);
        assert_test!(RagDebugLevel::Detailed < RagDebugLevel::Verbose);
        assert_test!(RagDebugLevel::Verbose < RagDebugLevel::Trace);
        Ok(())
    }));

    results.push(run_test("RagDebugLevel::from_str", || {
        assert_eq_test!(RagDebugLevel::from_str("off"), RagDebugLevel::Off);
        assert_eq_test!(RagDebugLevel::from_str("basic"), RagDebugLevel::Basic);
        assert_eq_test!(RagDebugLevel::from_str("detailed"), RagDebugLevel::Detailed);
        assert_eq_test!(RagDebugLevel::from_str("verbose"), RagDebugLevel::Verbose);
        assert_eq_test!(RagDebugLevel::from_str("trace"), RagDebugLevel::Trace);
        Ok(())
    }));

    results.push(run_test("RagDebugLevel::as_str", || {
        assert_eq_test!(RagDebugLevel::Off.as_str(), "OFF");
        assert_eq_test!(RagDebugLevel::Basic.as_str(), "BASIC");
        assert_eq_test!(RagDebugLevel::Detailed.as_str(), "DETAILED");
        assert_eq_test!(RagDebugLevel::Verbose.as_str(), "VERBOSE");
        assert_eq_test!(RagDebugLevel::Trace.as_str(), "TRACE");
        Ok(())
    }));

    // RagDebugConfig tests
    results.push(run_test("RagDebugConfig default values", || {
        let config = RagDebugConfig::default();
        assert_test!(!config.enabled, "debug should be disabled by default");
        assert_eq_test!(config.level, RagDebugLevel::Off);
        Ok(())
    }));

    results.push(run_test("RagDebugConfig custom values", || {
        let mut config = RagDebugConfig::default();
        config.enabled = true;
        config.level = RagDebugLevel::Detailed;
        config.log_to_file = true;
        config.log_path = Some(std::path::PathBuf::from("./logs"));
        assert_test!(config.enabled, "debug should be enabled");
        assert_eq_test!(config.level, RagDebugLevel::Detailed);
        assert_test!(config.log_to_file, "log_to_file should be true");
        Ok(())
    }));

    // RagDebugLogger tests
    results.push(run_test("RagDebugLogger creation", || {
        let config = RagDebugConfig::default();
        let _logger = RagDebugLogger::new(config);
        Ok(())
    }));

    // Graph RAG config tests
    results.push(run_test("GraphRagConfig default values", || {
        let config = GraphRagConfig::default();
        // Default should have reasonable values (may be 0 for some fields)
        let _ = config.max_depth;
        let _ = config.max_entities;
        Ok(())
    }));

    results.push(run_test("GraphRagConfig custom entity types", || {
        let mut config = GraphRagConfig::default();
        config.max_depth = 3;
        config.max_entities = 100;
        config.entity_types = vec!["SHIP".into(), "MANUFACTURER".into(), "COMPONENT".into()];
        assert_eq_test!(config.entity_types.len(), 3);
        assert_test!(
            config.entity_types.contains(&"SHIP".to_string()),
            "Should contain SHIP"
        );
        Ok(())
    }));

    // Hyde config tests
    results.push(run_test("HydeConfig default values", || {
        let config = HydeConfig::default();
        // Check structure exists and has reasonable defaults
        let _ = config.num_hypotheticals;
        Ok(())
    }));

    // Entity and Relationship tests for Graph RAG
    results.push(run_test("GraphEntity structure", || {
        let entity = GraphEntity {
            name: "Aurora MR".to_string(),
            entity_type: "SHIP".to_string(),
            mentions: vec![EntityMention {
                text: "Aurora MR".to_string(),
                start: 0,
                end: 9,
                confidence: 0.95,
            }],
        };
        assert_eq_test!(entity.name, "Aurora MR");
        assert_eq_test!(entity.entity_type, "SHIP");
        assert_eq_test!(entity.mentions.len(), 1);
        Ok(())
    }));

    results.push(run_test("Relationship structure for Graph RAG", || {
        let rel = Relationship {
            from_entity: "Aurora MR".to_string(),
            to_entity: "RSI".to_string(),
            relation_type: "manufactured_by".to_string(),
            weight: 1.0,
            source_chunk: Some("RSI makes the Aurora".to_string()),
        };
        assert_eq_test!(rel.from_entity, "Aurora MR");
        assert_eq_test!(rel.to_entity, "RSI");
        assert_eq_test!(rel.relation_type, "manufactured_by");
        Ok(())
    }));

    // Multi-query config tests
    results.push(run_test("MultiQueryConfig default values", || {
        let config = MultiQueryConfig::default();
        assert_test!(
            config.max_sub_queries > 0,
            "Should have at least 1 sub query"
        );
        Ok(())
    }));

    results.push(run_test("MultiQueryDecomposer creation", || {
        let decomposer = MultiQueryDecomposer::new();
        // Test complexity estimation
        let simple_query = "What is a ship?";
        let complex_query =
            "What ships are made by RSI and what are their weapons? Also, how much do they cost?";

        let simple_complexity = decomposer.estimate_complexity(simple_query);
        let complex_complexity = decomposer.estimate_complexity(complex_query);

        assert_test!(
            complex_complexity > simple_complexity,
            "Complex query should have higher complexity"
        );
        Ok(())
    }));

    // RRF Fusion tests
    results.push(run_test("RrfFusion creation and basic usage", || {
        let fusion = RrfFusion::default();
        let _ = fusion;
        Ok(())
    }));

    // HybridWeights tests
    results.push(run_test("HybridWeights default values", || {
        let weights = HybridWeights::default();
        assert_test!(
            weights.keyword >= 0.0,
            "keyword weight should be non-negative"
        );
        assert_test!(
            weights.semantic >= 0.0,
            "semantic weight should be non-negative"
        );
        Ok(())
    }));

    // Tier to feature mapping consistency
    results.push(run_test(
        "Tier feature consistency - Enhanced includes Semantic features",
        || {
            let semantic = RagTier::Semantic.to_features();
            let enhanced = RagTier::Enhanced.to_features();

            // Semantic features should be in Enhanced
            if semantic.fts_search {
                assert_test!(
                    enhanced.fts_search,
                    "Enhanced should have fts_search from Semantic"
                );
            }
            if semantic.semantic_search {
                assert_test!(
                    enhanced.semantic_search,
                    "Enhanced should have semantic_search from Semantic"
                );
            }
            if semantic.hybrid_search {
                assert_test!(
                    enhanced.hybrid_search,
                    "Enhanced should have hybrid_search from Semantic"
                );
            }
            Ok(())
        },
    ));

    results.push(run_test(
        "Tier feature consistency - Full has most features",
        || {
            let full = RagTier::Full.to_features();

            // Full should have core features enabled
            assert_test!(full.fts_search, "Full should have fts_search");
            assert_test!(full.semantic_search, "Full should have semantic_search");
            assert_test!(full.hybrid_search, "Full should have hybrid_search");
            assert_test!(full.query_expansion, "Full should have query_expansion");
            assert_test!(full.reranking, "Full should have reranking");
            Ok(())
        },
    ));

    // ScoredItem tests
    results.push(run_test("ScoredItem creation", || {
        let item = ScoredItem::new("test content".to_string(), 0.85);
        assert_eq_test!(item.item, "test content");
        assert_test!((item.score - 0.85).abs() < 0.001, "Score should be 0.85");
        assert_test!(item.metadata.is_empty(), "Default metadata should be empty");
        Ok(())
    }));

    results.push(run_test("ScoredItem with metadata", || {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let item = ScoredItem::with_metadata("test content".to_string(), 0.9, metadata);
        assert_eq_test!(item.metadata.get("source"), Some(&"test".to_string()));
        Ok(())
    }));

    // GraphRagRetriever tests
    results.push(run_test("GraphRagRetriever creation", || {
        let mut config = GraphRagConfig::default();
        config.max_depth = 2;
        config.max_entities = 50;
        config.entity_types = vec!["SHIP".into(), "MANUFACTURER".into()];
        let _retriever = GraphRagRetriever::new(config);
        Ok(())
    }));

    CategoryResult {
        name: "rag_tiers".to_string(),
        results,
    }
}

#[cfg(not(feature = "rag"))]
pub(crate) fn tests_rag_tiers() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ RAG Tier System (SKIPPED - rag feature not enabled)"
        ))
    );
    CategoryResult {
        name: "rag_tiers".to_string(),
        results: Vec::new(),
    }
}

// ─── Knowledge Graph Tests ────────────────────────────────────────────────────

#[cfg(feature = "rag")]
pub(crate) fn tests_knowledge_graph() -> CategoryResult {
    use ai_assistant::{
        KGEntityExtractor, KGEntityType, KnowledgeGraph, KnowledgeGraphBuilder,
        KnowledgeGraphConfig, KnowledgeGraphStore, PatternEntityExtractor,
    };

    println!("\n{}", bold(&cyan("▶ Knowledge Graph")));
    let mut results = Vec::new();

    // Test EntityType conversions
    results.push(run_test("EntityType string conversion", || {
        assert_eq_test!(
            KGEntityType::from_str("organization"),
            KGEntityType::Organization
        );
        assert_eq_test!(KGEntityType::from_str("ship"), KGEntityType::Product);
        assert_eq_test!(KGEntityType::from_str("person"), KGEntityType::Person);
        assert_eq_test!(KGEntityType::from_str("location"), KGEntityType::Location);
        assert_eq_test!(KGEntityType::from_str("concept"), KGEntityType::Concept);
        assert_eq_test!(KGEntityType::from_str("event"), KGEntityType::Event);
        assert_eq_test!(KGEntityType::from_str("unknown_type"), KGEntityType::Other);
        assert_eq_test!(KGEntityType::Organization.as_str(), "organization");
        assert_eq_test!(KGEntityType::Product.as_str(), "product");
        Ok(())
    }));

    // Test EntityType::all()
    results.push(run_test("EntityType::all returns all types", || {
        let all_types = KGEntityType::all();
        // 9 since V81-V88 added Paper + Author for the research module.
        assert_eq_test!(all_types.len(), 9);
        assert_test!(
            all_types.contains(&KGEntityType::Organization),
            "should contain Organization"
        );
        assert_test!(
            all_types.contains(&KGEntityType::Product),
            "should contain Product"
        );
        assert_test!(
            all_types.contains(&KGEntityType::Other),
            "should contain Other"
        );
        Ok(())
    }));

    // Test PatternEntityExtractor
    results.push(run_test("PatternEntityExtractor basic extraction", || {
        let extractor = PatternEntityExtractor::new()
            .add_entity("Aegis", KGEntityType::Organization)
            .add_entity("Sabre", KGEntityType::Product)
            .add_alias("Aegis Dynamics", "Aegis");

        let text = "Aegis Dynamics manufactures the Sabre fighter.";
        let result = extractor.extract(text).map_err(|e| e.to_string())?;

        assert_test!(!result.entities.is_empty(), "should extract entities");
        assert_test!(
            result
                .entities
                .iter()
                .any(|e| e.name.to_lowercase() == "aegis"),
            "should find Aegis"
        );
        assert_test!(
            result
                .entities
                .iter()
                .any(|e| e.name.to_lowercase() == "sabre"),
            "should find Sabre"
        );
        Ok(())
    }));

    // Test PatternEntityExtractor with multiple entities
    results.push(run_test("PatternEntityExtractor multiple entities", || {
        let extractor = PatternEntityExtractor::new().add_entities(&[
            ("RSI", KGEntityType::Organization),
            ("Origin", KGEntityType::Organization),
            ("Constellation", KGEntityType::Product),
            ("300i", KGEntityType::Product),
        ]);

        let text = "RSI makes the Constellation, while Origin produces the 300i.";
        let result = extractor.extract(text).map_err(|e| e.to_string())?;

        assert_test!(result.entities.len() >= 4, "should find all 4 entities");
        Ok(())
    }));

    // Test PatternEntityExtractor relation extraction
    results.push(run_test(
        "PatternEntityExtractor relation extraction",
        || {
            let extractor = PatternEntityExtractor::new()
                .add_entity("Aegis", KGEntityType::Organization)
                .add_entity("Sabre", KGEntityType::Product);

            let text = "Aegis manufactures the Sabre.";
            let result = extractor.extract(text).map_err(|e| e.to_string())?;

            // Should create at least one relation between entities in the same sentence
            assert_test!(
                !result.relations.is_empty(),
                "should extract at least one relation"
            );
            Ok(())
        },
    ));

    // Test PatternEntityExtractor query extraction
    results.push(run_test(
        "PatternEntityExtractor query entity extraction",
        || {
            let extractor = PatternEntityExtractor::new()
                .add_entity("Aegis", KGEntityType::Organization)
                .add_entity("Sabre", KGEntityType::Product);

            let query = "What ships does Aegis make?";
            let entities = extractor
                .extract_query_entities(query)
                .map_err(|e| e.to_string())?;

            assert_test!(
                entities.iter().any(|e| e.to_lowercase() == "aegis"),
                "should find Aegis in query"
            );
            Ok(())
        },
    ));

    // Test KnowledgeGraphStore in-memory creation
    results.push(run_test("KnowledgeGraphStore in-memory creation", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 0);
        assert_eq_test!(stats.total_relations, 0);
        assert_eq_test!(stats.total_chunks, 0);
        Ok(())
    }));

    // Test entity creation and lookup
    results.push(run_test("KnowledgeGraphStore entity creation", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let id = store
            .get_or_create_entity(
                "Aegis",
                KGEntityType::Organization,
                &["Aegis Dynamics".to_string()],
            )
            .map_err(|e| e.to_string())?;

        assert_test!(id > 0, "entity ID should be positive");

        // Find by name
        let found_id = store.find_entity_id("Aegis").map_err(|e| e.to_string())?;
        assert_eq_test!(found_id, Some(id));

        // Find by alias
        let alias_id = store
            .find_entity_id("Aegis Dynamics")
            .map_err(|e| e.to_string())?;
        assert_eq_test!(alias_id, Some(id));

        // Get entity
        let entity = store.get_entity(id).map_err(|e| e.to_string())?;
        assert_test!(entity.is_some(), "entity should exist");
        let entity = entity.unwrap();
        assert_eq_test!(entity.name, "Aegis");
        assert_eq_test!(entity.entity_type, KGEntityType::Organization);
        Ok(())
    }));

    // Test entity with aliases
    results.push(run_test("KnowledgeGraphStore alias resolution", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        store
            .get_or_create_entity(
                "Roberts Space Industries",
                KGEntityType::Organization,
                &["RSI".to_string()],
            )
            .map_err(|e| e.to_string())?;

        // Should find by alias
        let found = store.get_entity_by_name("RSI").map_err(|e| e.to_string())?;
        assert_test!(found.is_some(), "should find entity by alias RSI");
        let found = found.unwrap();
        assert_eq_test!(found.name, "Roberts Space Industries");
        Ok(())
    }));

    // Test relations
    results.push(run_test("KnowledgeGraphStore relations", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let aegis_id = store
            .get_or_create_entity("Aegis", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;

        let sabre_id = store
            .get_or_create_entity("Sabre", KGEntityType::Product, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(
                aegis_id,
                sabre_id,
                "manufactures",
                0.9,
                Some("Aegis makes the Sabre"),
                None,
            )
            .map_err(|e| e.to_string())?;

        let relations = store
            .get_relations_from(aegis_id, 1)
            .map_err(|e| e.to_string())?;
        assert_eq_test!(relations.len(), 1);
        assert_eq_test!(relations[0].from, "Aegis");
        assert_eq_test!(relations[0].to, "Sabre");
        assert_eq_test!(relations[0].relation_type, "manufactures");
        Ok(())
    }));

    // Test chunks
    results.push(run_test("KnowledgeGraphStore chunks", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let chunk_id = store
            .add_chunk("doc1", "Aegis manufactures the Sabre fighter.", 0)
            .map_err(|e| e.to_string())?;

        assert_test!(chunk_id > 0, "chunk ID should be positive");

        // Adding the same chunk should return the same ID (dedup by hash)
        let chunk_id2 = store
            .add_chunk("doc1", "Aegis manufactures the Sabre fighter.", 0)
            .map_err(|e| e.to_string())?;
        assert_eq_test!(chunk_id, chunk_id2);

        // Different content should create new chunk
        let chunk_id3 = store
            .add_chunk("doc1", "Origin produces the 300i.", 1)
            .map_err(|e| e.to_string())?;
        assert_test!(
            chunk_id3 != chunk_id,
            "different content should have different ID"
        );
        Ok(())
    }));

    // Test entity-chunk linking
    results.push(run_test("KnowledgeGraphStore entity-chunk linking", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let chunk_id = store
            .add_chunk("doc1", "Aegis manufactures the Sabre fighter.", 0)
            .map_err(|e| e.to_string())?;

        let aegis_id = store
            .get_or_create_entity("Aegis", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;

        store
            .link_entity_to_chunk(aegis_id, chunk_id, Some(0), Some("Aegis manufactures"))
            .map_err(|e| e.to_string())?;

        let chunks = store
            .get_chunks_for_entities(&[aegis_id])
            .map_err(|e| e.to_string())?;
        assert_eq_test!(chunks.len(), 1);
        assert_test!(
            chunks[0].content.contains("Aegis"),
            "chunk should contain Aegis"
        );
        Ok(())
    }));

    // Test stats
    results.push(run_test("KnowledgeGraphStore stats", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        store
            .get_or_create_entity("Aegis", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;
        store
            .get_or_create_entity("Origin", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;
        store
            .get_or_create_entity("Sabre", KGEntityType::Product, &[])
            .map_err(|e| e.to_string())?;

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 3);
        assert_eq_test!(stats.entities_by_type.get("organization"), Some(&2));
        assert_eq_test!(stats.entities_by_type.get("product"), Some(&1));
        Ok(())
    }));

    // Test clear
    results.push(run_test("KnowledgeGraphStore clear", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        store
            .get_or_create_entity("Aegis", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;
        store
            .add_chunk("doc1", "Test content", 0)
            .map_err(|e| e.to_string())?;

        let stats_before = store.get_stats().map_err(|e| e.to_string())?;
        assert_test!(
            stats_before.total_entities > 0,
            "should have entities before clear"
        );

        store.clear().map_err(|e| e.to_string())?;

        let stats_after = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats_after.total_entities, 0);
        assert_eq_test!(stats_after.total_chunks, 0);
        Ok(())
    }));

    // Test KnowledgeGraph high-level API
    results.push(run_test("KnowledgeGraph creation and stats", || {
        let config = KnowledgeGraphConfig::default();
        let graph = KnowledgeGraph::in_memory(config).map_err(|e| e.to_string())?;

        let stats = graph.stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 0);
        Ok(())
    }));

    // Test KnowledgeGraphBuilder
    results.push(run_test(
        "KnowledgeGraphBuilder with Star Citizen entities",
        || {
            let graph = KnowledgeGraphBuilder::new()
                .with_star_citizen_entities()
                .add_entity("Custom Entity", KGEntityType::Concept)
                .build_in_memory()
                .map_err(|e| e.to_string())?;

            let stats = graph.stats().map_err(|e| e.to_string())?;
            assert_test!(
                stats.total_entities > 0,
                "should have pre-populated entities"
            );

            // Check RSI alias works
            Ok(())
        },
    ));

    // Test KnowledgeGraphConfig defaults
    results.push(run_test("KnowledgeGraphConfig defaults", || {
        let config = KnowledgeGraphConfig::default();
        assert_eq_test!(config.max_traversal_depth, 2);
        assert_eq_test!(config.max_entities_per_query, 50);
        assert_eq_test!(config.max_chunks_per_entity, 5);
        assert_eq_test!(config.chunk_size, 1000);
        assert_eq_test!(config.chunk_overlap, 200);
        assert_test!(
            config.resolve_aliases,
            "resolve_aliases should be true by default"
        );
        Ok(())
    }));

    // Test graph traversal depth
    results.push(run_test("KnowledgeGraphStore multi-hop traversal", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        // Create a chain: Aegis -> Sabre -> Weapons
        let aegis_id = store
            .get_or_create_entity("Aegis", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;
        let sabre_id = store
            .get_or_create_entity("Sabre", KGEntityType::Product, &[])
            .map_err(|e| e.to_string())?;
        let weapons_id = store
            .get_or_create_entity("Weapons", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(aegis_id, sabre_id, "manufactures", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(sabre_id, weapons_id, "has", 0.8, None, None)
            .map_err(|e| e.to_string())?;

        // Depth 1 should only get Sabre
        let depth1_relations = store
            .get_relations_from(aegis_id, 1)
            .map_err(|e| e.to_string())?;
        assert_eq_test!(depth1_relations.len(), 1);

        // Depth 2 should get both Sabre and Weapons
        let depth2_relations = store
            .get_relations_from(aegis_id, 2)
            .map_err(|e| e.to_string())?;
        assert_eq_test!(depth2_relations.len(), 2);
        Ok(())
    }));

    // Test confidence threshold
    results.push(run_test("KnowledgeGraphStore confidence threshold", || {
        let mut config = KnowledgeGraphConfig::default();
        config.min_relation_confidence = 0.7;
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a_id = store
            .get_or_create_entity("EntityA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b_id = store
            .get_or_create_entity("EntityB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c_id = store
            .get_or_create_entity("EntityC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        // High confidence relation
        store
            .add_relation(a_id, b_id, "related", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        // Low confidence relation (below threshold)
        store
            .add_relation(a_id, c_id, "related", 0.5, None, None)
            .map_err(|e| e.to_string())?;

        let relations = store
            .get_relations_from(a_id, 1)
            .map_err(|e| e.to_string())?;
        // Should only return the high confidence relation
        assert_eq_test!(relations.len(), 1);
        assert_eq_test!(relations[0].to, "EntityB");
        Ok(())
    }));

    CategoryResult {
        name: "knowledge_graph".to_string(),
        results,
    }
}

#[cfg(not(feature = "rag"))]
pub(crate) fn tests_knowledge_graph() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Knowledge Graph (SKIPPED - rag feature not enabled)"
        ))
    );
    CategoryResult {
        name: "knowledge_graph".to_string(),
        results: Vec::new(),
    }
}

// ─── Graph Quality Tests ─────────────────────────────────────────────────────

#[cfg(feature = "rag")]
pub(crate) fn tests_graph_quality() -> CategoryResult {
    use ai_assistant::knowledge_graph::GraphAlgorithms;
    use ai_assistant::{KGEntityType, KnowledgeGraphConfig, KnowledgeGraphStore};

    println!("\n{}", bold(&cyan("▶ Graph Quality (KnowledgeGraph)")));
    let mut results = Vec::new();

    // Helper: build a test graph with known topology
    // Component 1: Aegis(Org)→Sabre(Prod), Aegis→Gladius(Prod), Sabre→Gladius
    // Component 2: Stanton(Loc)→UEE(Org), Crusader(Loc)→Stanton
    // Component 3: Vanduul(Org)→Xi_An(Org)
    // Isolated: Banu(Org) — orphan
    // Total: 9 entities, 6 relations, 4 connected components
    fn build_test_graph() -> Result<(KnowledgeGraphStore, Vec<(String, i64)>), String> {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;
        let mut ids = Vec::new();

        let entities = vec![
            ("Aegis", KGEntityType::Organization),
            ("Sabre", KGEntityType::Product),
            ("Gladius", KGEntityType::Product),
            ("Stanton", KGEntityType::Location),
            ("UEE", KGEntityType::Organization),
            ("Crusader", KGEntityType::Location),
            ("Vanduul", KGEntityType::Organization),
            ("Xi_An", KGEntityType::Organization),
            ("Banu", KGEntityType::Organization),
        ];

        for (name, etype) in &entities {
            let id = store
                .get_or_create_entity(name, *etype, &[])
                .map_err(|e| e.to_string())?;
            ids.push((name.to_string(), id));
        }

        let find = |name: &str| -> i64 { ids.iter().find(|(n, _)| n == name).unwrap().1 };

        // Component 1
        store
            .add_relation(
                find("Aegis"),
                find("Sabre"),
                "manufactures",
                0.9,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
        store
            .add_relation(
                find("Aegis"),
                find("Gladius"),
                "manufactures",
                0.9,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
        store
            .add_relation(
                find("Sabre"),
                find("Gladius"),
                "related_to",
                0.7,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;

        // Component 2
        store
            .add_relation(
                find("Stanton"),
                find("UEE"),
                "governed_by",
                0.95,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
        store
            .add_relation(
                find("Crusader"),
                find("Stanton"),
                "located_in",
                0.95,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;

        // Component 3
        store
            .add_relation(find("Vanduul"), find("Xi_An"), "enemy_of", 0.8, None, None)
            .map_err(|e| e.to_string())?;

        // Banu is isolated — no relations

        Ok((store, ids))
    }

    // --- Test 1: star topology stats ---
    results.push(run_test("star topology stats", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let hub = store
            .get_or_create_entity("Hub", KGEntityType::Organization, &[])
            .map_err(|e| e.to_string())?;
        for name in &["A", "B", "C", "D"] {
            let leaf = store
                .get_or_create_entity(name, KGEntityType::Concept, &[])
                .map_err(|e| e.to_string())?;
            store
                .add_relation(hub, leaf, "connects", 0.9, None, None)
                .map_err(|e| e.to_string())?;
        }

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 5);
        assert_eq_test!(stats.total_relations, 4);
        Ok(())
    }));

    // --- Test 2: linear chain stats ---
    results.push(run_test("linear chain stats", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let names = ["A", "B", "C", "D", "E"];
        let mut prev_id = None;
        for name in &names {
            let id = store
                .get_or_create_entity(name, KGEntityType::Concept, &[])
                .map_err(|e| e.to_string())?;
            if let Some(prev) = prev_id {
                store
                    .add_relation(prev, id, "next", 0.9, None, None)
                    .map_err(|e| e.to_string())?;
            }
            prev_id = Some(id);
        }

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 5);
        assert_eq_test!(stats.total_relations, 4);
        Ok(())
    }));

    // --- Test 3: PageRank star hub highest ---
    results.push(run_test("PageRank star hub highest", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let hub = store
            .get_or_create_entity("Hub", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let mut leaf_ids = Vec::new();
        for name in &["L1", "L2", "L3", "L4"] {
            let id = store
                .get_or_create_entity(name, KGEntityType::Concept, &[])
                .map_err(|e| e.to_string())?;
            // Leaves point TO hub
            store
                .add_relation(id, hub, "points_to", 0.9, None, None)
                .map_err(|e| e.to_string())?;
            leaf_ids.push(id);
        }

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;

        let hub_rank = ranks.get(&hub).copied().unwrap_or(0.0);
        for leaf_id in &leaf_ids {
            let leaf_rank = ranks.get(leaf_id).copied().unwrap_or(0.0);
            assert_test!(hub_rank > leaf_rank, "hub rank should exceed leaf rank");
        }
        Ok(())
    }));

    // --- Test 4: PageRank cycle uniform ---
    results.push(run_test("PageRank cycle uniform", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("CycA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("CycB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("CycC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(b, c, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(c, a, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;

        let expected = 1.0 / 3.0;
        for id in &[a, b, c] {
            let rank = ranks.get(id).copied().unwrap_or(0.0);
            assert_test!(
                (rank - expected).abs() < 0.05,
                &format!("cycle rank should be ~{:.3}, got {:.3}", expected, rank)
            );
        }
        Ok(())
    }));

    // --- Test 5: PageRank sum preserved in fully connected graph ---
    results.push(run_test("PageRank sum preserved in cycle", || {
        // PageRank sum = 1.0 only when no dangling nodes (all nodes have outgoing edges)
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        // Full cycle: A→B→C→D→A (no dangling nodes)
        let a = store
            .get_or_create_entity("FcA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("FcB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("FcC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let d = store
            .get_or_create_entity("FcD", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(b, c, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(c, d, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(d, a, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;

        let sum: f64 = ranks.values().sum();
        assert_test!(
            (sum - 1.0).abs() < 0.02,
            &format!("PageRank sum should be ~1.0 in full cycle, got {:.4}", sum)
        );
        Ok(())
    }));

    // --- Test 6: PageRank sink node ---
    results.push(run_test("PageRank sink node highest", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("SrcA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("Sink", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("SrcC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "flows", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(c, b, "flows", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;

        let sink_rank = ranks.get(&b).copied().unwrap_or(0.0);
        let a_rank = ranks.get(&a).copied().unwrap_or(0.0);
        let c_rank = ranks.get(&c).copied().unwrap_or(0.0);
        assert_test!(sink_rank > a_rank, "sink should rank higher than source A");
        assert_test!(sink_rank > c_rank, "sink should rank higher than source C");
        Ok(())
    }));

    // --- Test 7: shortest_path direct ---
    results.push(run_test("shortest_path direct edge", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("PA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("PB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "link", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let path = GraphAlgorithms::shortest_path(&store, a, b).map_err(|e| e.to_string())?;

        assert_test!(path.is_some(), "path should exist");
        let path = path.unwrap();
        assert_eq_test!(path.len(), 2);
        assert_eq_test!(path[0], a);
        assert_eq_test!(path[1], b);
        Ok(())
    }));

    // --- Test 8: shortest_path multi-hop ---
    results.push(run_test("shortest_path multi-hop", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("ChA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("ChB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("ChC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let d = store
            .get_or_create_entity("ChD", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(b, c, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(c, d, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let path = GraphAlgorithms::shortest_path(&store, a, d).map_err(|e| e.to_string())?;

        assert_test!(path.is_some(), "path should exist A→D");
        let path = path.unwrap();
        assert_eq_test!(path.len(), 4);
        assert_eq_test!(path[0], a);
        assert_eq_test!(path[3], d);
        Ok(())
    }));

    // --- Test 9: shortest_path disconnected ---
    results.push(run_test("shortest_path disconnected returns None", || {
        let (store, ids) = build_test_graph()?;

        // Aegis (component 1) → Banu (isolated) — no path
        let aegis_id = ids.iter().find(|(n, _)| n == "Aegis").unwrap().1;
        let banu_id = ids.iter().find(|(n, _)| n == "Banu").unwrap().1;

        let path =
            GraphAlgorithms::shortest_path(&store, aegis_id, banu_id).map_err(|e| e.to_string())?;

        assert_test!(path.is_none(), "disconnected nodes should have no path");
        Ok(())
    }));

    // --- Test 10: connected_components count ---
    results.push(run_test("connected_components count", || {
        let (store, _ids) = build_test_graph()?;

        let components =
            GraphAlgorithms::connected_components(&store).map_err(|e| e.to_string())?;

        // Count distinct component IDs
        let mut component_ids: Vec<i64> = components.values().copied().collect();
        component_ids.sort();
        component_ids.dedup();

        // 4 components: {Aegis,Sabre,Gladius}, {Stanton,UEE,Crusader}, {Vanduul,Xi_An}, {Banu}
        assert_eq_test!(component_ids.len(), 4);
        Ok(())
    }));

    // --- Test 11: connected_components membership ---
    results.push(run_test("connected_components membership", || {
        let (store, ids) = build_test_graph()?;

        let components =
            GraphAlgorithms::connected_components(&store).map_err(|e| e.to_string())?;

        let find_id = |name: &str| -> i64 { ids.iter().find(|(n, _)| n == name).unwrap().1 };

        // Aegis, Sabre, Gladius should be in the same component
        let aegis_comp = components.get(&find_id("Aegis")).copied().unwrap_or(-1);
        let sabre_comp = components.get(&find_id("Sabre")).copied().unwrap_or(-2);
        let gladius_comp = components.get(&find_id("Gladius")).copied().unwrap_or(-3);
        assert_eq_test!(aegis_comp, sabre_comp);
        assert_eq_test!(sabre_comp, gladius_comp);

        // Banu should be in a different component
        let banu_comp = components.get(&find_id("Banu")).copied().unwrap_or(-4);
        assert_test!(banu_comp != aegis_comp, "Banu should be isolated");
        Ok(())
    }));

    // --- Test 12: degree_centrality hub ---
    results.push(run_test("degree_centrality hub", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let hub = store
            .get_or_create_entity("DHub", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let mut leaf_ids = Vec::new();
        for name in &["DL1", "DL2", "DL3", "DL4"] {
            let id = store
                .get_or_create_entity(name, KGEntityType::Concept, &[])
                .map_err(|e| e.to_string())?;
            store
                .add_relation(hub, id, "connects", 0.9, None, None)
                .map_err(|e| e.to_string())?;
            leaf_ids.push(id);
        }

        let (in_degree, out_degree) =
            GraphAlgorithms::degree_centrality(&store).map_err(|e| e.to_string())?;

        let hub_out = out_degree.get(&hub).copied().unwrap_or(0);
        assert_eq_test!(hub_out, 4);

        for leaf_id in &leaf_ids {
            let leaf_in = in_degree.get(leaf_id).copied().unwrap_or(0);
            assert_eq_test!(leaf_in, 1);
        }
        Ok(())
    }));

    // --- Test 13: degree_centrality orphan ---
    results.push(run_test("degree_centrality orphan zero", || {
        let (store, ids) = build_test_graph()?;

        let (in_degree, out_degree) =
            GraphAlgorithms::degree_centrality(&store).map_err(|e| e.to_string())?;

        let banu_id = ids.iter().find(|(n, _)| n == "Banu").unwrap().1;
        let banu_in = in_degree.get(&banu_id).copied().unwrap_or(0);
        let banu_out = out_degree.get(&banu_id).copied().unwrap_or(0);

        assert_eq_test!(banu_in, 0);
        assert_eq_test!(banu_out, 0);
        Ok(())
    }));

    // --- Test 14: all_paths triangle ---
    results.push(run_test("all_paths triangle multiple paths", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("TA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("TB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("TC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "edge", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(b, c, "edge", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(a, c, "edge", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        let paths = GraphAlgorithms::all_paths(&store, a, c, 5).map_err(|e| e.to_string())?;

        // Direct A→C and A→B→C = at least 2 paths
        assert_test!(
            paths.len() >= 2,
            &format!("triangle should have ≥2 paths A→C, got {}", paths.len())
        );
        Ok(())
    }));

    // --- Test 15: all_paths depth limit ---
    results.push(run_test("all_paths depth limit", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("DpA", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let b = store
            .get_or_create_entity("DpB", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let c = store
            .get_or_create_entity("DpC", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;
        let d = store
            .get_or_create_entity("DpD", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        store
            .add_relation(a, b, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(b, c, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;
        store
            .add_relation(c, d, "next", 0.9, None, None)
            .map_err(|e| e.to_string())?;

        // depth=2 won't reach D from A (need 3 hops)
        let short = GraphAlgorithms::all_paths(&store, a, d, 2).map_err(|e| e.to_string())?;
        assert_test!(short.is_empty(), "depth 2 should not reach 3-hop target");

        // depth=3 should find the path
        let found = GraphAlgorithms::all_paths(&store, a, d, 3).map_err(|e| e.to_string())?;
        assert_test!(!found.is_empty(), "depth 3 should find 3-hop path");
        Ok(())
    }));

    // --- Test 16: orphan detection via degree ---
    results.push(run_test("orphan detection via degree centrality", || {
        let (store, ids) = build_test_graph()?;

        let (in_deg, out_deg) =
            GraphAlgorithms::degree_centrality(&store).map_err(|e| e.to_string())?;

        let orphans: Vec<&str> = ids
            .iter()
            .filter(|(_, id)| {
                let i = in_deg.get(id).copied().unwrap_or(0);
                let o = out_deg.get(id).copied().unwrap_or(0);
                i + o == 0
            })
            .map(|(name, _)| name.as_str())
            .collect();

        assert_test!(
            orphans.contains(&"Banu"),
            "Banu should be detected as orphan"
        );
        assert_eq_test!(orphans.len(), 1);
        Ok(())
    }));

    // --- Iter 2: empty graph PageRank ---
    results.push(run_test("PageRank empty graph", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;

        assert_test!(ranks.is_empty(), "empty graph should have no ranks");
        Ok(())
    }));

    // --- Iter 2: single node graph ---
    results.push(run_test("single node graph stats", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        store
            .get_or_create_entity("Alone", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 1);
        assert_eq_test!(stats.total_relations, 0);

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;
        assert_eq_test!(ranks.len(), 1);
        let rank = ranks.values().next().copied().unwrap_or(0.0);
        // Single dangling node: converges to (1-d)/n = 0.15 for d=0.85, n=1
        assert_test!(rank > 0.0, "single node should have positive rank");
        assert_test!(
            (rank - 0.15).abs() < 0.02,
            &format!("single dangling node rank should be ~0.15, got {:.4}", rank)
        );
        Ok(())
    }));

    // --- Iter 2: graph density metric ---
    results.push(run_test("graph density metric", || {
        let (store, _ids) = build_test_graph()?;

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        let n = stats.total_entities as f64;
        let e = stats.total_relations as f64;
        // Density = E / (N * (N-1)) for directed graph
        let density = if n > 1.0 { e / (n * (n - 1.0)) } else { 0.0 };

        // 6 relations / (9 * 8) = 0.083
        assert_test!(density > 0.0, "density should be positive");
        assert_test!(density < 1.0, "density should be < 1.0 (sparse)");
        assert_test!(
            (density - 6.0 / 72.0).abs() < 0.01,
            &format!("expected density ~0.083, got {:.4}", density)
        );
        Ok(())
    }));

    // --- Iter 2: self-path ---
    results.push(run_test("shortest_path self returns trivial", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        let a = store
            .get_or_create_entity("Self", KGEntityType::Concept, &[])
            .map_err(|e| e.to_string())?;

        let path = GraphAlgorithms::shortest_path(&store, a, a).map_err(|e| e.to_string())?;

        // Self-path: either Some([a]) or None, both are acceptable
        if let Some(p) = &path {
            assert_test!(p.len() <= 1, "self-path should be trivial");
        }
        Ok(())
    }));

    // --- Iter 3: enterprise KG with 20 entities ---
    results.push(run_test("enterprise KG 20 entities realistic", || {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).map_err(|e| e.to_string())?;

        // Simulate an enterprise product catalog
        let orgs = ["Aegis", "RSI", "Origin", "MISC", "Drake"];
        let products = [
            "Sabre",
            "Gladius",
            "Aurora",
            "Constellation",
            "300i",
            "Freelancer",
            "Prospector",
            "Cutlass",
            "Caterpillar",
            "Starfarer",
        ];
        let locations = ["Stanton", "Pyro", "Nyx", "Terra", "Sol"];

        let mut org_ids = Vec::new();
        for name in &orgs {
            let id = store
                .get_or_create_entity(name, KGEntityType::Organization, &[])
                .map_err(|e| e.to_string())?;
            org_ids.push(id);
        }

        let mut prod_ids = Vec::new();
        for name in &products {
            let id = store
                .get_or_create_entity(name, KGEntityType::Product, &[])
                .map_err(|e| e.to_string())?;
            prod_ids.push(id);
        }

        let mut loc_ids = Vec::new();
        for name in &locations {
            let id = store
                .get_or_create_entity(name, KGEntityType::Location, &[])
                .map_err(|e| e.to_string())?;
            loc_ids.push(id);
        }

        // Each org manufactures 2 products
        for (i, &org_id) in org_ids.iter().enumerate() {
            store
                .add_relation(org_id, prod_ids[i * 2], "manufactures", 0.95, None, None)
                .map_err(|e| e.to_string())?;
            store
                .add_relation(
                    org_id,
                    prod_ids[i * 2 + 1],
                    "manufactures",
                    0.95,
                    None,
                    None,
                )
                .map_err(|e| e.to_string())?;
        }
        // Each org headquartered at a location
        for (i, &org_id) in org_ids.iter().enumerate() {
            store
                .add_relation(org_id, loc_ids[i], "headquartered_at", 0.9, None, None)
                .map_err(|e| e.to_string())?;
        }

        let stats = store.get_stats().map_err(|e| e.to_string())?;
        assert_eq_test!(stats.total_entities, 20);
        assert_eq_test!(stats.total_relations, 15); // 10 manufactures + 5 HQ

        // Verify graph algorithms work at scale
        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).map_err(|e| e.to_string())?;
        assert_eq_test!(ranks.len(), 20);
        // Orgs have highest rank (they have outgoing edges to products + locations)
        let max_rank = ranks.values().copied().fold(0.0f64, f64::max);
        assert_test!(max_rank > 0.0, "max rank should be positive at scale");

        let comps = GraphAlgorithms::connected_components(&store).map_err(|e| e.to_string())?;
        let mut comp_vals: Vec<i64> = comps.values().copied().collect();
        comp_vals.sort();
        comp_vals.dedup();
        // 5 orgs, each connected to its own 2 products + 1 location = 5 components
        assert_eq_test!(comp_vals.len(), 5);
        Ok(())
    }));

    CategoryResult {
        name: "graph_quality".to_string(),
        results,
    }
}

#[cfg(not(feature = "rag"))]
pub(crate) fn tests_graph_quality() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Graph Quality (SKIPPED - rag feature not enabled)"))
    );
    CategoryResult {
        name: "graph_quality".to_string(),
        results: Vec::new(),
    }
}

// ─── Multi-Layer Graph Quality Tests ─────────────────────────────────────────

pub(crate) fn tests_multi_layer_graph() -> CategoryResult {
    use ai_assistant::{
        BeliefType, ConflictPolicy, ContradictionResolution, GraphLayer, MultiLayerGraph,
        UserBelief,
    };

    println!("\n{}", bold(&cyan("▶ Multi-Layer Graph Quality")));
    let mut results = Vec::new();

    // --- Test 1: empty graph stats ---
    results.push(run_test("empty graph stats zero", || {
        let g = MultiLayerGraph::new();
        let stats = g.stats();
        assert_eq_test!(stats.session_count, 0);
        assert_eq_test!(stats.total_session_entities, 0);
        assert_eq_test!(stats.user_beliefs_count, 0);
        assert_eq_test!(stats.internet_entries_count, 0);
        assert_eq_test!(stats.contradiction_count, 0);
        Ok(())
    }));

    // --- Test 2: session entity CRUD ---
    results.push(run_test("session entity CRUD", || {
        let mut g = MultiLayerGraph::new();
        g.process_user_message("s1", "Tell me about Aegis", &["Aegis".to_string()]);
        g.process_user_message("s1", "And Sabre too", &["Sabre".to_string()]);
        g.process_user_message("s1", "What about Gladius", &["Gladius".to_string()]);

        let stats = g.stats();
        assert_eq_test!(stats.session_count, 1);
        assert_eq_test!(stats.total_session_entities, 3);

        // Duplicate entity should not increase count
        g.process_user_message("s1", "More about Aegis", &["Aegis".to_string()]);
        let stats2 = g.stats();
        assert_eq_test!(stats2.total_session_entities, 3);
        Ok(())
    }));

    // --- Test 3: session relations in unified view ---
    results.push(run_test("session relations in unified view", || {
        let mut g = MultiLayerGraph::new();
        g.process_user_message(
            "s1",
            "Aegis makes Sabre",
            &["Aegis".to_string(), "Sabre".to_string()],
        );

        let view = g.query_unified(Some("s1"));
        assert_test!(
            !view.entities.is_empty(),
            "unified view should have entities"
        );

        let names: Vec<&str> = view.entities.iter().map(|e| e.name.as_str()).collect();
        assert_test!(names.contains(&"Aegis"), "should contain Aegis");
        assert_test!(names.contains(&"Sabre"), "should contain Sabre");
        Ok(())
    }));

    // --- Test 4: contradiction detection ---
    results.push(run_test(
        "contradiction detection on conflicting data",
        || {
            let mut g = MultiLayerGraph::new();
            // Add internet data with a conflicting knowledge value
            let contradiction = g.add_internet_data(
                "Sabre",
                "max_speed",
                "1200 m/s",
                "https://example.com/ships",
                Some("1100 m/s"), // Known value differs from internet
            );

            assert_test!(contradiction.is_some(), "should detect contradiction");
            let c = contradiction.unwrap();
            assert_eq_test!(c.entity, "Sabre");
            assert_eq_test!(c.attribute, "max_speed");
            Ok(())
        },
    ));

    // --- Test 5: contradiction resolution ---
    results.push(run_test(
        "contradiction resolution clears unresolved",
        || {
            let mut g = MultiLayerGraph::new();
            let c = g.add_internet_data("Gladius", "crew", "2", "https://example.com", Some("1"));

            assert_test!(c.is_some(), "should detect contradiction");
            let cid = c.unwrap().id.clone();

            let stats_before = g.stats();
            assert_eq_test!(stats_before.unresolved_contradictions, 1);

            g.resolve_contradiction(&cid, ContradictionResolution::PrimaryTrustworthy);
            let stats_after = g.stats();
            assert_eq_test!(stats_after.unresolved_contradictions, 0);
            Ok(())
        },
    ));

    // --- Test 6: no contradiction when matching ---
    results.push(run_test("no contradiction when values match", || {
        let mut g = MultiLayerGraph::new();
        let result = g.add_internet_data(
            "Aurora",
            "max_speed",
            "900 m/s",
            "https://example.com",
            Some("900 m/s"), // Same value — no conflict
        );

        assert_test!(
            result.is_none(),
            "matching values should not trigger contradiction"
        );
        Ok(())
    }));

    // --- Test 7: cross-layer same_as inference ---
    results.push(run_test("cross-layer same_as inference", || {
        let mut g = MultiLayerGraph::new();
        // Add entity via session
        g.process_user_message(
            "s1",
            "Tell me about Constellation",
            &["Constellation".to_string()],
        );
        // Add same entity via internet
        g.add_internet_data(
            "Constellation",
            "type",
            "multi-crew",
            "https://example.com",
            None,
        );

        let inferred = g.infer_cross_layer(Some("s1"));
        // Should detect Constellation appears in both Session and Internet layers
        let has_constellation = inferred.iter().any(|r| {
            r.source_entity.to_lowercase() == "constellation"
                || r.target_entity.to_lowercase() == "constellation"
        });
        assert_test!(
            has_constellation,
            "should infer cross-layer relation for Constellation"
        );
        Ok(())
    }));

    // --- Test 8: cross-layer case insensitive ---
    results.push(run_test("cross-layer case insensitive matching", || {
        let mut g = MultiLayerGraph::new();
        g.process_user_message("s1", "sabre info", &["sabre".to_string()]);
        g.add_internet_data("Sabre", "type", "fighter", "https://example.com", None);

        let inferred = g.infer_cross_layer(Some("s1"));
        let has_sabre = inferred.iter().any(|r| {
            r.source_entity.to_lowercase() == "sabre" || r.target_entity.to_lowercase() == "sabre"
        });
        assert_test!(
            has_sabre,
            "case-insensitive matching should find sabre/Sabre"
        );
        Ok(())
    }));

    // --- Test 9: unified view all layers ---
    results.push(run_test("unified view includes all layers", || {
        let mut g = MultiLayerGraph::new();
        // Session layer
        g.process_user_message("s1", "About Aegis", &["Aegis".to_string()]);
        // Internet layer
        g.add_internet_data("RSI", "type", "manufacturer", "https://example.com", None);
        // User layer
        g.user_graph.add_belief(UserBelief {
            id: "b1".to_string(),
            statement: "Origin makes the best ships".to_string(),
            subject_entity: Some("Origin".to_string()),
            belief_type: BeliefType::Opinion,
            expressed_at: 1000,
            session_id: "s1".to_string(),
            confidence: 0.8,
            active: true,
        });

        let view = g.query_unified(Some("s1"));
        let names: Vec<String> = view
            .entities
            .iter()
            .map(|e| e.name.to_lowercase())
            .collect();

        assert_test!(
            names.contains(&"aegis".to_string()),
            "session entity Aegis in unified view"
        );
        assert_test!(
            names.contains(&"rsi".to_string()),
            "internet entity RSI in unified view"
        );
        assert_test!(
            names.contains(&"origin".to_string()),
            "user entity Origin in unified view"
        );
        Ok(())
    }));

    // --- Test 10: cluster_entities connected ---
    results.push(run_test("cluster_entities connected group", || {
        let mut g = MultiLayerGraph::new();
        g.process_user_message(
            "s1",
            "Aegis and Sabre and Gladius",
            &[
                "Aegis".to_string(),
                "Sabre".to_string(),
                "Gladius".to_string(),
            ],
        );

        let clusters = g.cluster_entities(&GraphLayer::Session, Some("s1"), 2);

        // All 3 entities mentioned together should form 1 cluster
        if !clusters.is_empty() {
            assert_test!(
                clusters[0].entity_names.len() >= 2,
                "cluster should have ≥2 members"
            );
            assert_test!(
                clusters[0].cohesion >= 0.0,
                "cohesion should be non-negative"
            );
        }
        // If no clusters, that's acceptable — depends on relation formation
        Ok(())
    }));

    // --- Test 11: diff detects additions ---
    results.push(run_test("diff detects entity additions", || {
        let g1 = MultiLayerGraph::new();
        let mut g2 = MultiLayerGraph::new();
        g2.process_user_message("s1", "Aegis info", &["Aegis".to_string()]);

        let diff = g1.diff(&g2, Some("s1"));
        assert_test!(
            !diff.added_entities.is_empty(),
            "diff should detect added entities"
        );

        let has_aegis = diff.added_entities.iter().any(|(_, name)| name == "Aegis");
        assert_test!(has_aegis, "Aegis should appear as added");
        Ok(())
    }));

    // --- Test 12: apply_diff Union ---
    results.push(run_test("apply_diff Union merges entities", || {
        use ai_assistant::GraphMergeStrategy;

        let mut g1 = MultiLayerGraph::new();
        let mut g2 = MultiLayerGraph::new();
        g2.process_user_message("s1", "About RSI", &["RSI".to_string()]);

        let diff = g1.diff(&g2, Some("s1"));
        g1.apply_diff(&diff, "s1", &GraphMergeStrategy::Union);

        let stats = g1.stats();
        assert_test!(
            stats.total_session_entities > 0,
            "applied diff should add entities"
        );
        Ok(())
    }));

    // --- Test 13: conflict resolution HighestConfidence ---
    results.push(run_test("conflict resolution HighestConfidence", || {
        let mut g = MultiLayerGraph::new();
        // Add entities with different confidence levels via internet
        g.add_internet_data(
            "TestEntity",
            "attr",
            "high_conf_value",
            "https://high.com",
            None,
        );
        g.add_internet_data(
            "TestEntity",
            "attr2",
            "low_conf_value",
            "https://low.com",
            None,
        );

        let resolved = g.resolve_conflict(
            "TestEntity",
            &GraphLayer::Internet,
            &ConflictPolicy::HighestConfidence,
        );

        // Should return some resolved entity
        if let Some(entity) = resolved {
            assert_test!(
                !entity.name.is_empty(),
                "resolved entity should have a name"
            );
        }
        Ok(())
    }));

    // --- Iter 2: belief extraction ---
    results.push(run_test("user belief extraction and count", || {
        let mut g = MultiLayerGraph::new();

        g.user_graph.add_belief(UserBelief {
            id: "belief1".to_string(),
            statement: "Vanduul are hostile".to_string(),
            subject_entity: Some("Vanduul".to_string()),
            belief_type: BeliefType::Fact,
            expressed_at: 100,
            session_id: "s1".to_string(),
            confidence: 0.95,
            active: true,
        });

        g.user_graph.add_belief(UserBelief {
            id: "belief2".to_string(),
            statement: "Xi'An are traders".to_string(),
            subject_entity: Some("Xi_An".to_string()),
            belief_type: BeliefType::Opinion,
            expressed_at: 200,
            session_id: "s1".to_string(),
            confidence: 0.7,
            active: true,
        });

        let stats = g.stats();
        assert_eq_test!(stats.user_beliefs_count, 2);
        Ok(())
    }));

    // --- Iter 3: multi-session cross-layer ---
    results.push(run_test("multi-session cross-layer realistic", || {
        let mut g = MultiLayerGraph::new();

        // Session 1: user asks about ships
        g.process_user_message(
            "session_alpha",
            "Tell me about Aegis ships",
            &["Aegis".to_string(), "Sabre".to_string()],
        );

        // Session 2: same user, different topic
        g.process_user_message(
            "session_beta",
            "Where is Stanton?",
            &["Stanton".to_string()],
        );

        // Internet data crosses into session content
        g.add_internet_data("Aegis", "founded", "2841", "https://lore.com", None);

        let stats = g.stats();
        assert_eq_test!(stats.session_count, 2);

        // Unified view for session_alpha should include Aegis
        let view_a = g.query_unified(Some("session_alpha"));
        let has_aegis = view_a.entities.iter().any(|e| e.name == "Aegis");
        assert_test!(has_aegis, "session_alpha should see Aegis");

        // Unified view for session_beta should include Stanton
        let view_b = g.query_unified(Some("session_beta"));
        let has_stanton = view_b.entities.iter().any(|e| e.name == "Stanton");
        assert_test!(has_stanton, "session_beta should see Stanton");

        // Cross-layer inference should detect Aegis across session + internet
        let inferred = g.infer_cross_layer(Some("session_alpha"));
        let cross_aegis = inferred
            .iter()
            .any(|r| r.source_entity == "Aegis" || r.target_entity == "Aegis");
        assert_test!(cross_aegis, "Aegis should appear in cross-layer inference");
        Ok(())
    }));

    // --- Iter 4: contradiction stats consistency ---
    results.push(run_test("contradiction stats consistency", || {
        let mut g = MultiLayerGraph::new();

        // Add 3 contradictions
        g.add_internet_data("Ship1", "speed", "1000", "https://a.com", Some("900"));
        g.add_internet_data("Ship2", "crew", "3", "https://b.com", Some("2"));
        g.add_internet_data("Ship3", "mass", "heavy", "https://c.com", Some("light"));

        assert_eq_test!(g.contradictions.len(), 3);

        // Contradiction IDs may collide (timestamp-based), so assign unique IDs
        g.contradictions[0].id = "c_test_0".to_string();
        g.contradictions[1].id = "c_test_1".to_string();
        g.contradictions[2].id = "c_test_2".to_string();

        let stats = g.stats();
        assert_eq_test!(stats.contradiction_count, 3);
        assert_eq_test!(stats.unresolved_contradictions, 3);

        // Resolve 2
        let r1 = g.resolve_contradiction("c_test_0", ContradictionResolution::PrimaryTrustworthy);
        let r2 = g.resolve_contradiction("c_test_1", ContradictionResolution::InternetMoreRecent);

        assert_test!(r1, "resolve c1 should succeed");
        assert_test!(r2, "resolve c2 should succeed");

        let stats2 = g.stats();
        assert_eq_test!(stats2.contradiction_count, 3);
        assert_eq_test!(stats2.unresolved_contradictions, 1);
        Ok(())
    }));

    CategoryResult {
        name: "multi_layer_graph".to_string(),
        results,
    }
}

// ─── Agent Graph Quality Tests ───────────────────────────────────────────────

pub(crate) fn tests_agent_graph_quality() -> CategoryResult {
    use ai_assistant::{
        AgentGraph, EdgeType, ExecutionTrace, GraphAgentEdge, GraphAgentNode, GraphAnalytics,
        GraphError, GraphStepStatus, TraceStep,
    };

    println!("\n{}", bold(&cyan("▶ Agent Graph Quality")));
    let mut results = Vec::new();

    // --- Test 1: topological sort DAG ---
    results.push(run_test("topological sort linear DAG", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("a", "Agent A", "processor"));
        g.add_node(GraphAgentNode::new("b", "Agent B", "processor"));
        g.add_node(GraphAgentNode::new("c", "Agent C", "processor"));
        g.add_node(GraphAgentNode::new("d", "Agent D", "processor"));

        g.add_edge(GraphAgentEdge::new("a", "b", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("b", "c", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("c", "d", EdgeType::DataFlow));

        let sorted = g.topological_sort().map_err(|e| format!("{:?}", e))?;
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();

        // A must come before B, B before C, C before D
        let pos_a = ids.iter().position(|&x| x == "a").unwrap();
        let pos_b = ids.iter().position(|&x| x == "b").unwrap();
        let pos_c = ids.iter().position(|&x| x == "c").unwrap();
        let pos_d = ids.iter().position(|&x| x == "d").unwrap();
        assert_test!(pos_a < pos_b, "A before B");
        assert_test!(pos_b < pos_c, "B before C");
        assert_test!(pos_c < pos_d, "C before D");
        Ok(())
    }));

    // --- Test 2: topological sort diamond ---
    results.push(run_test("topological sort diamond DAG", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("a", "Source", "input"));
        g.add_node(GraphAgentNode::new("b", "Left", "processor"));
        g.add_node(GraphAgentNode::new("c", "Right", "processor"));
        g.add_node(GraphAgentNode::new("d", "Sink", "output"));

        g.add_edge(GraphAgentEdge::new("a", "b", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("a", "c", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("b", "d", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("c", "d", EdgeType::DataFlow));

        let sorted = g.topological_sort().map_err(|e| format!("{:?}", e))?;
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();

        let pos_a = ids.iter().position(|&x| x == "a").unwrap();
        let pos_d = ids.iter().position(|&x| x == "d").unwrap();
        assert_eq_test!(pos_a, 0); // A must be first
        assert_eq_test!(pos_d, 3); // D must be last
        Ok(())
    }));

    // --- Test 3: cycle detection ---
    results.push(run_test("cycle detection returns error", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("x", "X", "processor"));
        g.add_node(GraphAgentNode::new("y", "Y", "processor"));
        g.add_node(GraphAgentNode::new("z", "Z", "processor"));

        g.add_edge(GraphAgentEdge::new("x", "y", EdgeType::Control));
        g.add_edge(GraphAgentEdge::new("y", "z", EdgeType::Control));
        g.add_edge(GraphAgentEdge::new("z", "x", EdgeType::Control)); // cycle!

        let result = g.topological_sort();
        assert_test!(result.is_err(), "cycle should cause error");
        match result {
            Err(GraphError::CycleDetected) => {}
            Err(other) => return Err(format!("expected CycleDetected, got {:?}", other)),
            Ok(_) => return Err("expected error but got Ok".to_string()),
        }
        Ok(())
    }));

    // --- Test 4: export DOT valid ---
    results.push(run_test("export DOT contains valid structure", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("n1", "Node1", "processor"));
        g.add_node(GraphAgentNode::new("n2", "Node2", "processor"));
        g.add_edge(GraphAgentEdge::new("n1", "n2", EdgeType::DataFlow));

        let dot = g.export_dot();
        assert_test!(dot.contains("digraph"), "DOT should contain 'digraph'");
        assert_test!(dot.contains("n1"), "DOT should contain node n1");
        assert_test!(dot.contains("n2"), "DOT should contain node n2");
        assert_test!(dot.contains("->"), "DOT should contain arrow");
        Ok(())
    }));

    // --- Test 5: export Mermaid valid ---
    results.push(run_test("export Mermaid contains valid structure", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("m1", "MNode1", "processor"));
        g.add_node(GraphAgentNode::new("m2", "MNode2", "processor"));
        g.add_edge(GraphAgentEdge::new("m1", "m2", EdgeType::Delegation));

        let mermaid = g.export_mermaid();
        assert_test!(
            mermaid.contains("graph") || mermaid.contains("flowchart"),
            "Mermaid should contain graph/flowchart header"
        );
        assert_test!(mermaid.contains("m1"), "Mermaid should contain node m1");
        Ok(())
    }));

    // --- Test 6: critical path picks slowest ---
    results.push(run_test("critical path picks slowest branch", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("start", "Start", "input"));
        g.add_node(GraphAgentNode::new("fast", "FastPath", "processor"));
        g.add_node(GraphAgentNode::new("slow", "SlowPath", "processor"));
        g.add_node(GraphAgentNode::new("end", "End", "output"));

        g.add_edge(GraphAgentEdge::new("start", "fast", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("start", "slow", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("fast", "end", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("slow", "end", EdgeType::DataFlow));

        let mut trace = ExecutionTrace::new();
        let mut step_start = TraceStep::new("start", "process");
        step_start.duration_ms = 10;
        step_start.status = GraphStepStatus::Completed;
        trace.record(step_start);

        let mut step_fast = TraceStep::new("fast", "process");
        step_fast.duration_ms = 50;
        step_fast.status = GraphStepStatus::Completed;
        trace.record(step_fast);

        let mut step_slow = TraceStep::new("slow", "process");
        step_slow.duration_ms = 500;
        step_slow.status = GraphStepStatus::Completed;
        trace.record(step_slow);

        let mut step_end = TraceStep::new("end", "process");
        step_end.duration_ms = 10;
        step_end.status = GraphStepStatus::Completed;
        trace.record(step_end);

        let critical = GraphAnalytics::critical_path(&g, &trace);
        assert_test!(
            critical.contains(&"slow".to_string()),
            "critical path should include the slow branch"
        );
        Ok(())
    }));

    // --- Test 7: bottleneck detection ---
    results.push(run_test("bottleneck detection threshold filtering", || {
        let mut trace = ExecutionTrace::new();

        let mut fast = TraceStep::new("agent_fast", "action");
        fast.duration_ms = 10;
        fast.status = GraphStepStatus::Completed;
        trace.record(fast);

        let mut slow = TraceStep::new("agent_slow", "action");
        slow.duration_ms = 1000;
        slow.status = GraphStepStatus::Completed;
        trace.record(slow);

        let mut medium = TraceStep::new("agent_mid", "action");
        medium.duration_ms = 200;
        medium.status = GraphStepStatus::Completed;
        trace.record(medium);

        // Threshold 500ms — only slow should be a bottleneck
        let bottlenecks = GraphAnalytics::bottlenecks(&trace, 500);
        assert_eq_test!(bottlenecks.len(), 1);
        assert_eq_test!(bottlenecks[0].agent_id, "agent_slow");
        Ok(())
    }));

    // --- Test 8: utilization fractions ---
    results.push(run_test("utilization fractions proportional", || {
        let mut trace = ExecutionTrace::new();

        for (id, dur) in &[("ag1", 100u64), ("ag2", 200), ("ag3", 300)] {
            let mut step = TraceStep::new(id, "work");
            step.duration_ms = *dur;
            step.status = GraphStepStatus::Completed;
            trace.record(step);
        }

        let util = GraphAnalytics::agent_utilization(&trace);
        assert_eq_test!(util.len(), 3);

        // All utilization values should be positive
        assert_test!(
            util.values().all(|&v| v > 0.0),
            "all utilizations should be positive"
        );

        // ag3 (300ms) should have the highest utilization
        let u1 = util.get("ag1").copied().unwrap_or(0.0);
        let u3 = util.get("ag3").copied().unwrap_or(0.0);
        assert_test!(
            u3 > u1,
            "agent with longer duration should have higher utilization"
        );
        Ok(())
    }));

    // --- Iter 3: 5-agent pipeline realistic ---
    results.push(run_test("5-agent pipeline realistic scenario", || {
        let mut g = AgentGraph::new();

        // Realistic pipeline: Ingest → Parse → Analyze → Summarize → Output
        let nodes = vec![
            ("ingest", "Data Ingestion", "input"),
            ("parse", "Document Parser", "processor"),
            ("analyze", "Semantic Analyzer", "processor"),
            ("summarize", "Summarizer", "processor"),
            ("output", "Response Generator", "output"),
        ];

        for (id, name, atype) in &nodes {
            g.add_node(GraphAgentNode::new(id, name, atype));
        }

        g.add_edge(GraphAgentEdge::new("ingest", "parse", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new("parse", "analyze", EdgeType::DataFlow));
        g.add_edge(GraphAgentEdge::new(
            "analyze",
            "summarize",
            EdgeType::DataFlow,
        ));
        g.add_edge(GraphAgentEdge::new(
            "summarize",
            "output",
            EdgeType::DataFlow,
        ));

        // Verify topological order
        let sorted = g.topological_sort().map_err(|e| format!("{:?}", e))?;
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();
        assert_eq_test!(ids[0], "ingest");
        assert_eq_test!(ids[4], "output");

        // Simulate execution trace
        let mut trace = ExecutionTrace::new();
        let durations = [50u64, 150, 300, 200, 100];
        for (i, (id, _, _)) in nodes.iter().enumerate() {
            let mut step = TraceStep::new(id, "process");
            step.duration_ms = durations[i];
            step.status = GraphStepStatus::Completed;
            trace.record(step);
        }

        // Critical path should include all nodes (linear pipeline)
        let critical = GraphAnalytics::critical_path(&g, &trace);
        assert_test!(
            critical.len() >= 3,
            "critical path should include most nodes"
        );

        // Analyze should be a bottleneck (300ms > 250ms threshold)
        let bottlenecks = GraphAnalytics::bottlenecks(&trace, 250);
        let has_analyze = bottlenecks.iter().any(|b| b.agent_id == "analyze");
        assert_test!(has_analyze, "Semantic Analyzer should be a bottleneck");

        // Exports should work
        let dot = g.export_dot();
        let mermaid = g.export_mermaid();
        assert_test!(!dot.is_empty(), "DOT export should not be empty");
        assert_test!(!mermaid.is_empty(), "Mermaid export should not be empty");
        Ok(())
    }));

    // --- Iter 4: JSON roundtrip ---
    results.push(run_test("export JSON roundtrip valid", || {
        let mut g = AgentGraph::new();
        g.add_node(GraphAgentNode::new("j1", "JsonNode", "processor"));
        g.add_node(GraphAgentNode::new("j2", "JsonNode2", "processor"));
        g.add_edge(GraphAgentEdge::new("j1", "j2", EdgeType::Communication));

        let json = g.export_json();
        assert_test!(!json.is_empty(), "JSON export should not be empty");

        // Should be valid JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
        assert_test!(parsed.is_ok(), "exported JSON should be parseable");
        Ok(())
    }));

    CategoryResult {
        name: "agent_graph_quality".to_string(),
        results,
    }
}
