//! Comprehensive tests for the RAG Tier System
//!
//! Tests cover: rag_tiers, rag_debug, rag_pipeline, rag_methods

// ============================================================================
// RAG TIERS TESTS
// ============================================================================

#[cfg(feature = "rag")]
mod rag_tiers_tests {
    use ai_assistant::rag_tiers::*;

    // --- RagFeatures Tests ---

    #[test]
    fn test_rag_features_none() {
        let features = RagFeatures::none();
        assert_eq!(features.enabled_count(), 0);
        assert!(features.enabled_features().is_empty());
    }

    #[test]
    fn test_rag_features_all() {
        let features = RagFeatures::all();
        assert_eq!(features.enabled_count(), 20);
        assert!(features.fts_search);
        assert!(features.semantic_search);
        assert!(features.hybrid_search);
        assert!(features.query_expansion);
        assert!(features.multi_query);
        assert!(features.hyde);
        assert!(features.reranking);
        assert!(features.agentic_mode);
        assert!(features.graph_rag);
        assert!(features.raptor);
        assert!(features.multimodal);
    }

    #[test]
    fn test_rag_features_partial() {
        let mut features = RagFeatures::none();
        features.fts_search = true;
        features.semantic_search = true;
        features.hybrid_search = true;

        assert_eq!(features.enabled_count(), 3);
        let enabled = features.enabled_features();
        assert!(enabled.contains(&"fts_search"));
        assert!(enabled.contains(&"semantic_search"));
        assert!(enabled.contains(&"hybrid_search"));
        assert!(!enabled.contains(&"reranking"));
    }

    // --- RagTier Tests ---

    #[test]
    fn test_rag_tier_display_names() {
        assert_eq!(RagTier::Disabled.display_name(), "Disabled");
        assert_eq!(RagTier::Fast.display_name(), "Fast");
        assert_eq!(RagTier::Semantic.display_name(), "Semantic");
        assert_eq!(RagTier::Enhanced.display_name(), "Enhanced");
        assert_eq!(RagTier::Thorough.display_name(), "Thorough");
        assert_eq!(RagTier::Agentic.display_name(), "Agentic");
        assert_eq!(RagTier::Graph.display_name(), "Graph");
        assert_eq!(RagTier::Full.display_name(), "Full");
        assert_eq!(RagTier::Custom.display_name(), "Custom");
    }

    #[test]
    fn test_rag_tier_descriptions() {
        // Each tier should have a non-empty description
        for tier in RagTier::all_tiers() {
            let desc = tier.description();
            assert!(!desc.is_empty(), "Tier {:?} has empty description", tier);
        }
    }

    #[test]
    fn test_tier_to_features_disabled() {
        let features = RagTier::Disabled.to_features();
        assert_eq!(features.enabled_count(), 0);
    }

    #[test]
    fn test_tier_to_features_fast() {
        let features = RagTier::Fast.to_features();
        assert!(features.fts_search);
        assert!(!features.semantic_search);
        assert!(!features.reranking);
        assert_eq!(features.enabled_count(), 1);
    }

    #[test]
    fn test_tier_to_features_semantic() {
        let features = RagTier::Semantic.to_features();
        assert!(features.fts_search);
        assert!(features.semantic_search);
        assert!(features.hybrid_search);
        assert!(features.fusion_rrf);
        assert!(!features.reranking);
        assert!(!features.query_expansion);
    }

    #[test]
    fn test_tier_to_features_enhanced() {
        let features = RagTier::Enhanced.to_features();
        assert!(features.fts_search);
        assert!(features.semantic_search);
        assert!(features.hybrid_search);
        assert!(features.query_expansion);
        assert!(features.reranking);
        assert!(features.sentence_window);
        assert!(!features.multi_query);
        assert!(!features.agentic_mode);
    }

    #[test]
    fn test_tier_to_features_thorough() {
        let features = RagTier::Thorough.to_features();
        assert!(features.multi_query);
        assert!(features.contextual_compression);
        assert!(features.self_reflection);
        assert!(features.corrective_rag);
        assert!(!features.agentic_mode);
    }

    #[test]
    fn test_tier_to_features_agentic() {
        let features = RagTier::Agentic.to_features();
        assert!(features.agentic_mode);
        assert!(features.adaptive_strategy);
        assert!(!features.graph_rag);
    }

    #[test]
    fn test_tier_to_features_graph() {
        let features = RagTier::Graph.to_features();
        assert!(features.graph_rag);
        assert!(features.agentic_mode);
        assert!(features.parent_document);
    }

    #[test]
    fn test_tier_to_features_full() {
        let features = RagTier::Full.to_features();
        assert_eq!(features.enabled_count(), 20);
        assert!(features.graph_rag);
        assert!(features.raptor);
        assert!(features.multimodal);
    }

    #[test]
    fn test_tier_estimated_calls() {
        // Disabled and Fast should have 0 calls
        assert_eq!(RagTier::Disabled.estimated_extra_calls(), (0, Some(0)));
        assert_eq!(RagTier::Fast.estimated_extra_calls(), (0, Some(0)));
        assert_eq!(RagTier::Semantic.estimated_extra_calls(), (0, Some(0)));

        // Enhanced should have 1-2 calls
        let (min, max) = RagTier::Enhanced.estimated_extra_calls();
        assert!(min >= 1);
        assert!(max.unwrap() >= min);

        // Agentic should be unbounded
        let (_, max) = RagTier::Agentic.estimated_extra_calls();
        assert!(max.is_none());

        // Full should be unbounded
        let (_, max) = RagTier::Full.estimated_extra_calls();
        assert!(max.is_none());
    }

    #[test]
    fn test_tier_requires_embeddings() {
        assert!(!RagTier::Disabled.requires_embeddings());
        assert!(!RagTier::Fast.requires_embeddings());
        assert!(RagTier::Semantic.requires_embeddings());
        assert!(RagTier::Enhanced.requires_embeddings());
        assert!(RagTier::Full.requires_embeddings());
    }

    #[test]
    fn test_all_tiers_list() {
        let tiers = RagTier::all_tiers();
        assert_eq!(tiers.len(), 9);
        assert!(tiers.contains(&RagTier::Disabled));
        assert!(tiers.contains(&RagTier::Full));
        assert!(tiers.contains(&RagTier::Custom));
    }

    // --- RagConfig Tests ---

    #[test]
    fn test_rag_config_default() {
        let config = RagConfig::default();
        assert_eq!(config.tier, RagTier::Fast);
        assert!(!config.use_custom_features);
        assert_eq!(config.max_extra_llm_calls, 5);
        assert_eq!(config.max_chunks, 10);
        assert!(!config.debug_enabled);
    }

    #[test]
    fn test_rag_config_with_tier() {
        let config = RagConfig::with_tier(RagTier::Enhanced);
        assert_eq!(config.tier, RagTier::Enhanced);
        assert!(!config.use_custom_features);

        let features = config.effective_features();
        assert!(features.reranking);
        assert!(features.query_expansion);
    }

    #[test]
    fn test_rag_config_with_features() {
        let mut features = RagFeatures::none();
        features.fts_search = true;
        features.reranking = true;

        let config = RagConfig::with_features(features);
        assert_eq!(config.tier, RagTier::Custom);
        assert!(config.use_custom_features);

        let effective = config.effective_features();
        assert!(effective.fts_search);
        assert!(effective.reranking);
        assert!(!effective.semantic_search);
    }

    #[test]
    fn test_rag_config_effective_features_tier_mode() {
        let config = RagConfig::with_tier(RagTier::Semantic);
        // use_custom_features is false, so tier features are used
        let effective = config.effective_features();
        assert!(effective.semantic_search);
        assert!(effective.hybrid_search);
    }

    #[test]
    fn test_rag_config_effective_features_custom_mode() {
        let mut config = RagConfig::with_tier(RagTier::Semantic);
        config.features = RagFeatures::none();
        config.features.fts_search = true;
        config.use_custom_features = true;

        let effective = config.effective_features();
        // Should use custom features, not tier
        assert!(effective.fts_search);
        assert!(!effective.semantic_search);
    }

    #[test]
    fn test_rag_config_estimate_extra_calls() {
        let fast = RagConfig::with_tier(RagTier::Fast);
        assert_eq!(fast.estimate_extra_calls(), (0, Some(0)));

        let enhanced = RagConfig::with_tier(RagTier::Enhanced);
        let (min, max) = enhanced.estimate_extra_calls();
        assert!(min >= 1);
        assert!(max.is_some());

        let agentic = RagConfig::with_tier(RagTier::Agentic);
        let (_, max) = agentic.estimate_extra_calls();
        assert!(max.is_none()); // Unbounded
    }

    #[test]
    fn test_rag_config_check_requirements() {
        let fast = RagConfig::with_tier(RagTier::Fast);
        assert!(fast.check_requirements().is_empty());

        let semantic = RagConfig::with_tier(RagTier::Semantic);
        let reqs = semantic.check_requirements();
        assert!(reqs.contains(&RagRequirement::EmbeddingModel));

        let graph = RagConfig::with_tier(RagTier::Graph);
        let reqs = graph.check_requirements();
        assert!(reqs.contains(&RagRequirement::GraphDatabase));
        assert!(reqs.contains(&RagRequirement::EmbeddingModel));
    }

    #[test]
    fn test_rag_config_is_feature_enabled() {
        let config = RagConfig::with_tier(RagTier::Enhanced);
        assert!(config.is_feature_enabled("fts_search"));
        assert!(config.is_feature_enabled("semantic_search"));
        assert!(config.is_feature_enabled("reranking"));
        assert!(!config.is_feature_enabled("agentic_mode"));
        assert!(!config.is_feature_enabled("nonexistent_feature"));
    }

    #[test]
    fn test_rag_config_builder_pattern() {
        let config = RagConfig::with_tier(RagTier::Enhanced)
            .with_debug("./logs")
            .with_max_calls(10)
            .with_max_chunks(20)
            .with_embedding_model("nomic-embed");

        assert!(config.debug_enabled);
        assert_eq!(config.debug_log_path, Some("./logs".to_string()));
        assert_eq!(config.max_extra_llm_calls, 10);
        assert_eq!(config.max_chunks, 20);
        assert_eq!(config.embedding_model, Some("nomic-embed".to_string()));
    }

    #[test]
    fn test_rag_config_summary() {
        let config = RagConfig::with_tier(RagTier::Enhanced);
        let summary = config.summary();
        assert!(summary.contains("Enhanced"));
        assert!(summary.contains("features enabled"));
        assert!(summary.contains("LLM calls"));
    }

    // --- RagRequirement Tests ---

    #[test]
    fn test_rag_requirement_display_names() {
        assert_eq!(
            RagRequirement::EmbeddingModel.display_name(),
            "Embedding Model"
        );
        assert_eq!(
            RagRequirement::GraphDatabase.display_name(),
            "Graph Database"
        );
        assert_eq!(RagRequirement::VisionModel.display_name(), "Vision Model");
    }

    #[test]
    fn test_rag_requirement_descriptions() {
        let desc = RagRequirement::EmbeddingModel.description();
        assert!(desc.contains("vector"));

        let desc = RagRequirement::GraphDatabase.description();
        assert!(desc.contains("graph"));
    }

    // --- HybridWeights Tests ---

    #[test]
    fn test_hybrid_weights_default() {
        let weights = HybridWeights::default();
        assert_eq!(weights.keyword, 0.4);
        assert_eq!(weights.semantic, 0.5);
        assert!(weights.recency > 0.0);
    }

    #[test]
    fn test_hybrid_weights_presets() {
        let balanced = HybridWeights::balanced();
        assert_eq!(balanced.keyword, 0.5);
        assert_eq!(balanced.semantic, 0.5);

        let keyword_heavy = HybridWeights::keyword_heavy();
        assert!(keyword_heavy.keyword > keyword_heavy.semantic);

        let semantic_heavy = HybridWeights::semantic_heavy();
        assert!(semantic_heavy.semantic > semantic_heavy.keyword);
    }

    #[test]
    fn test_hybrid_weights_normalize() {
        let mut weights = HybridWeights {
            keyword: 2.0,
            semantic: 2.0,
            recency: 1.0,
            priority: 1.0,
        };
        weights.normalize();

        let total = weights.keyword + weights.semantic + weights.recency + weights.priority;
        assert!((total - 1.0).abs() < 0.001);
    }

    // --- RagStats Tests ---

    #[test]
    fn test_rag_stats_recording() {
        let mut stats = RagStats::default();

        stats.record_query();
        stats.record_query();
        assert_eq!(stats.queries_processed, 2);

        stats.record_llm_calls(3, 500);
        assert_eq!(stats.llm_calls, 3);
        assert_eq!(stats.llm_time_ms, 500);

        stats.record_retrieval(20, 10, 100);
        assert_eq!(stats.chunks_retrieved, 20);
        assert_eq!(stats.chunks_used, 10);
        assert_eq!(stats.retrieval_time_ms, 100);

        stats.record_feature("reranking");
        stats.record_feature("reranking");
        stats.record_feature("expansion");
        assert_eq!(*stats.feature_usage.get("reranking").unwrap(), 2);
        assert_eq!(*stats.feature_usage.get("expansion").unwrap(), 1);
    }

    #[test]
    fn test_rag_stats_calculations() {
        let mut stats = RagStats::default();
        stats.cache_hits = 30;
        stats.cache_misses = 70;

        assert!((stats.cache_hit_rate() - 0.3).abs() < 0.001);

        stats.queries_processed = 5;
        stats.llm_calls = 15;
        assert!((stats.avg_llm_calls_per_query() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_rag_stats_summary() {
        let mut stats = RagStats::default();
        stats.queries_processed = 10;
        stats.llm_calls = 25;
        stats.chunks_retrieved = 100;

        let summary = stats.summary();
        assert!(summary.contains("10 queries"));
        assert!(summary.contains("25 LLM calls"));
    }

    // --- Auto Selection Tests ---

    #[test]
    fn test_auto_select_tier_speed_preference() {
        let hints = TierSelectionHints {
            preference: UserPreference::Speed,
            has_embeddings: true,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&hints), RagTier::Fast);
    }

    #[test]
    fn test_auto_select_tier_quality_preference() {
        let hints = TierSelectionHints {
            preference: UserPreference::Quality,
            has_embeddings: true,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&hints), RagTier::Enhanced);
    }

    #[test]
    fn test_auto_select_tier_max_quality_preference() {
        let hints = TierSelectionHints {
            preference: UserPreference::MaxQuality,
            has_embeddings: true,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&hints), RagTier::Thorough);
    }

    #[test]
    fn test_auto_select_tier_no_embeddings_fallback() {
        let hints = TierSelectionHints {
            preference: UserPreference::Quality,
            has_embeddings: false, // No embeddings!
            ..Default::default()
        };
        // Should fall back to Fast since Quality requires embeddings
        assert_eq!(auto_select_tier(&hints), RagTier::Fast);
    }

    #[test]
    fn test_auto_select_tier_complexity_upgrade() {
        let hints = TierSelectionHints {
            preference: UserPreference::Balanced,
            has_embeddings: true,
            query_complexity: QueryComplexity::Reasoning,
            ..Default::default()
        };
        // Reasoning should upgrade to at least Thorough
        assert_eq!(auto_select_tier(&hints), RagTier::Thorough);
    }
}

// ============================================================================
// RAG DEBUG TESTS
// ============================================================================

#[cfg(feature = "rag")]
mod rag_debug_tests {
    use ai_assistant::rag_debug::*;
    use std::time::Duration;

    // --- RagDebugLevel Tests ---

    #[test]
    fn test_debug_level_ordering() {
        assert!(RagDebugLevel::Trace > RagDebugLevel::Verbose);
        assert!(RagDebugLevel::Verbose > RagDebugLevel::Detailed);
        assert!(RagDebugLevel::Detailed > RagDebugLevel::Basic);
        assert!(RagDebugLevel::Basic > RagDebugLevel::Minimal);
        assert!(RagDebugLevel::Minimal > RagDebugLevel::Off);
    }

    #[test]
    fn test_debug_level_from_str() {
        assert_eq!(RagDebugLevel::from_str("off"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("MINIMAL"), RagDebugLevel::Minimal);
        assert_eq!(RagDebugLevel::from_str("basic"), RagDebugLevel::Basic);
        assert_eq!(RagDebugLevel::from_str("detailed"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("verbose"), RagDebugLevel::Verbose);
        assert_eq!(RagDebugLevel::from_str("trace"), RagDebugLevel::Trace);

        // Numeric parsing
        assert_eq!(RagDebugLevel::from_str("0"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("3"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("5"), RagDebugLevel::Trace);
    }

    #[test]
    fn test_debug_level_as_str() {
        assert_eq!(RagDebugLevel::Off.as_str(), "OFF");
        assert_eq!(RagDebugLevel::Detailed.as_str(), "DETAILED");
        assert_eq!(RagDebugLevel::Trace.as_str(), "TRACE");
    }

    // --- RagDebugConfig Tests ---

    #[test]
    fn test_debug_config_default() {
        let config = RagDebugConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.level, RagDebugLevel::Off);
        assert!(!config.log_to_file);
        assert!(config.include_timestamps);
        assert_eq!(config.max_entries, 1000);
    }

    #[test]
    fn test_debug_config_presets() {
        let minimal = RagDebugConfig::minimal();
        assert!(minimal.enabled);
        assert_eq!(minimal.level, RagDebugLevel::Minimal);
        assert!(minimal.log_to_stderr);

        let detailed = RagDebugConfig::detailed();
        assert!(detailed.enabled);
        assert_eq!(detailed.level, RagDebugLevel::Detailed);
        assert!(detailed.log_chunks);
        assert!(detailed.log_scores);

        let file_based = RagDebugConfig::file_based("./rag_logs");
        assert!(file_based.log_to_file);
        assert!(file_based.log_llm_details);
        assert!(file_based.log_path.is_some());

        let verbose = RagDebugConfig::verbose("./verbose_logs");
        assert_eq!(verbose.level, RagDebugLevel::Verbose);
        assert!(verbose.log_to_stderr);
    }

    // --- RagDebugSession Tests ---

    #[test]
    fn test_debug_session_creation() {
        let session = RagDebugSession::new("test_id", "test query");
        assert_eq!(session.session_id, "test_id");
        assert_eq!(session.query, "test query");
        assert!(session.start_time_ms > 0);
        assert!(session.end_time_ms.is_none());
        assert!(session.steps.is_empty());
    }

    #[test]
    fn test_debug_session_add_step() {
        let mut session = RagDebugSession::new("test", "query");

        session.add_step(RagDebugStep::KeywordSearch {
            query: "test".into(),
            results_count: 10,
            top_score: Some(0.9),
            duration_ms: 50,
        });

        assert_eq!(session.steps.len(), 1);
        assert_eq!(session.stats.chunks_retrieved, 10);
        assert_eq!(session.stats.retrieval_time_ms, 50);
    }

    #[test]
    fn test_debug_session_llm_stats() {
        let mut session = RagDebugSession::new("test", "query");

        session.add_step(RagDebugStep::LlmCall {
            purpose: "expansion".into(),
            model: "test-model".into(),
            input_tokens: 100,
            output_tokens: 50,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 200,
        });

        session.add_step(RagDebugStep::LlmCall {
            purpose: "reranking".into(),
            model: "test-model".into(),
            input_tokens: 200,
            output_tokens: 100,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 300,
        });

        assert_eq!(session.stats.llm_calls, 2);
        assert_eq!(session.stats.llm_input_tokens, 300);
        assert_eq!(session.stats.llm_output_tokens, 150);
        assert_eq!(session.stats.llm_time_ms, 500);
    }

    #[test]
    fn test_debug_session_error_tracking() {
        let mut session = RagDebugSession::new("test", "query");

        session.add_step(RagDebugStep::Error {
            step: "retrieval".into(),
            message: "Connection timeout".into(),
            recoverable: true,
        });

        session.add_step(RagDebugStep::Warning {
            step: "reranking".into(),
            message: "Low confidence scores".into(),
        });

        assert_eq!(session.errors.len(), 1);
        assert_eq!(session.warnings.len(), 1);
        assert!(session.errors[0].contains("timeout"));
    }

    #[test]
    fn test_debug_session_completion() {
        let mut session = RagDebugSession::new("test", "query");
        assert!(session.end_time_ms.is_none());
        assert!(session.total_duration_ms.is_none());

        session.complete(Some("Response text".into()));

        assert!(session.end_time_ms.is_some());
        assert!(session.total_duration_ms.is_some());
        assert_eq!(session.final_response, Some("Response text".into()));
    }

    #[test]
    fn test_debug_session_set_metadata() {
        let mut session = RagDebugSession::new("test", "query");

        session.set_tier("Enhanced");
        session.set_features(vec!["fts".into(), "semantic".into()]);
        session.set_context("Test context");

        assert_eq!(session.rag_tier, Some("Enhanced".into()));
        assert_eq!(session.features_enabled.len(), 2);
        assert_eq!(session.final_context, Some("Test context".into()));
    }

    #[test]
    fn test_debug_session_summary() {
        let mut session = RagDebugSession::new("test_session", "test query");

        session.add_step(RagDebugStep::LlmCall {
            purpose: "test".into(),
            model: "test".into(),
            input_tokens: 100,
            output_tokens: 50,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 100,
        });

        session.complete(None);

        let summary = session.summary();
        assert!(summary.contains("test_session"));
        assert!(summary.contains("1 LLM calls"));
        assert!(summary.contains("100 in"));
        assert!(summary.contains("50 out"));
    }

    // --- RagDebugLogger Tests ---

    #[test]
    fn test_debug_logger_creation() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        assert!(!logger.is_enabled());

        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        assert!(logger.is_enabled());
        assert_eq!(logger.level(), RagDebugLevel::Detailed);
    }

    #[test]
    fn test_debug_logger_enable_disable() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        assert!(!logger.is_enabled());

        logger.set_enabled(true);
        assert!(logger.is_enabled());

        logger.set_enabled(false);
        assert!(!logger.is_enabled());
    }

    #[test]
    fn test_debug_logger_set_level() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        assert_eq!(logger.level(), RagDebugLevel::Off);

        logger.set_level(RagDebugLevel::Detailed);
        assert_eq!(logger.level(), RagDebugLevel::Detailed);
        assert!(logger.is_enabled());

        logger.set_level(RagDebugLevel::Off);
        assert!(!logger.is_enabled());
    }

    #[test]
    fn test_debug_logger_level_enabled_check() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        assert!(logger.is_level_enabled(RagDebugLevel::Off));
        assert!(logger.is_level_enabled(RagDebugLevel::Minimal));
        assert!(logger.is_level_enabled(RagDebugLevel::Basic));
        assert!(logger.is_level_enabled(RagDebugLevel::Detailed));
        assert!(!logger.is_level_enabled(RagDebugLevel::Verbose));
        assert!(!logger.is_level_enabled(RagDebugLevel::Trace));
    }

    #[test]
    fn test_debug_logger_session_lifecycle() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        // Start query
        let session = logger.start_query("test query");
        let session_id = session.session_id().to_string();

        // Log some steps
        session.log_keyword_search("test", 5, Some(0.8), Duration::from_millis(50));
        session.log_semantic_search("test", "model", 10, Some(0.9), Duration::from_millis(100));
        session.log_llm_call("expansion", "test", 100, 50, Duration::from_millis(200));

        // Complete
        session.complete_with_response("Test response");

        // Check session was stored
        let sessions = logger.all_sessions();
        assert_eq!(sessions.len(), 1);

        let stored = logger.get_session(&session_id);
        assert!(stored.is_some());

        let s = stored.unwrap();
        assert_eq!(s.query, "test query");
        assert_eq!(s.steps.len(), 3);
        assert_eq!(s.stats.llm_calls, 1);
        assert!(s.total_duration_ms.is_some());
    }

    #[test]
    fn test_debug_logger_multiple_sessions() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        for i in 0..5 {
            let session = logger.start_query(format!("query {}", i));
            session.complete_no_response();
        }

        assert_eq!(logger.all_sessions().len(), 5);

        let recent = logger.recent_sessions(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_debug_logger_aggregate_stats() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        let s1 = logger.start_query("q1");
        s1.log_llm_call("test", "model", 100, 50, Duration::from_millis(100));
        s1.complete_no_response();

        let s2 = logger.start_query("q2");
        s2.log_llm_call("test", "model", 200, 100, Duration::from_millis(200));
        s2.log_llm_call("test", "model", 50, 25, Duration::from_millis(50));
        s2.complete_no_response();

        let stats = logger.aggregate_stats();
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_llm_calls, 3);
        assert_eq!(stats.total_input_tokens, 350);
        assert_eq!(stats.total_output_tokens, 175);
        assert!((stats.avg_llm_calls_per_session - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_debug_logger_clear_sessions() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        let s = logger.start_query("test");
        s.complete_no_response();

        assert_eq!(logger.all_sessions().len(), 1);

        logger.clear_sessions();
        assert!(logger.all_sessions().is_empty());
    }

    #[test]
    fn test_debug_logger_configure() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        assert!(!logger.is_enabled());

        logger.configure(RagDebugConfig::detailed());
        assert!(logger.is_enabled());
        assert_eq!(logger.level(), RagDebugLevel::Detailed);
    }

    // --- RagQuerySession Tests ---

    #[test]
    fn test_query_session_logging_methods() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("test query");

        session.log_query_received("test query");
        session.log_expansion(
            "original",
            vec!["expanded1".into()],
            "llm",
            Duration::from_millis(50),
        );
        session.log_keyword_search("test", 5, Some(0.8), Duration::from_millis(30));
        session.log_semantic_search("test", "model", 10, Some(0.9), Duration::from_millis(40));
        session.log_llm_call("test", "model", 100, 50, Duration::from_millis(100));
        session.log_error("step", "error message", true);
        session.log_warning("step", "warning message");

        session.complete_no_response();

        let stored = logger.all_sessions();
        assert_eq!(stored[0].steps.len(), 7);
        assert_eq!(stored[0].errors.len(), 1);
        assert_eq!(stored[0].warnings.len(), 1);
    }

    #[test]
    fn test_query_session_metadata() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("test");

        session.set_tier("Enhanced");
        session.set_features(vec!["fts".into(), "semantic".into()]);
        session.set_context("Context text");

        session.complete_no_response();

        let stored = &logger.all_sessions()[0];
        assert_eq!(stored.rag_tier, Some("Enhanced".into()));
        assert_eq!(stored.features_enabled.len(), 2);
        assert_eq!(stored.final_context, Some("Context text".into()));
    }

    #[test]
    fn test_query_session_elapsed_time() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("test");

        std::thread::sleep(Duration::from_millis(10));

        let elapsed = session.elapsed();
        assert!(elapsed.as_millis() >= 10);

        session.complete_no_response();
    }

    // --- RagDebugStep Tests ---

    #[test]
    fn test_debug_step_variants() {
        // Test that all step variants can be created
        let _steps = vec![
            RagDebugStep::QueryReceived {
                query: "test".into(),
                timestamp_ms: 12345,
            },
            RagDebugStep::QueryAnalysis {
                query: "test".into(),
                intent: Some("search".into()),
                complexity: Some("simple".into()),
                keywords: vec!["keyword".into()],
                duration_ms: 10,
            },
            RagDebugStep::QueryExpansion {
                original: "test".into(),
                expanded: vec!["test1".into()],
                method: "llm".into(),
                duration_ms: 50,
            },
            RagDebugStep::MultiQuery {
                original: "test".into(),
                sub_queries: vec!["q1".into(), "q2".into()],
                duration_ms: 100,
            },
            RagDebugStep::HyDE {
                query: "test".into(),
                hypothetical_doc: "doc".into(),
                duration_ms: 200,
            },
            RagDebugStep::KeywordSearch {
                query: "test".into(),
                results_count: 10,
                top_score: Some(0.9),
                duration_ms: 30,
            },
            RagDebugStep::SemanticSearch {
                query: "test".into(),
                embedding_model: "model".into(),
                results_count: 15,
                top_similarity: Some(0.85),
                duration_ms: 50,
            },
            RagDebugStep::HybridFusion {
                keyword_results: 10,
                semantic_results: 15,
                fused_results: 20,
                method: "rrf".into(),
                weights: None,
                duration_ms: 5,
            },
            RagDebugStep::Reranking {
                input_count: 20,
                output_count: 10,
                method: "llm".into(),
                score_changes: vec![],
                duration_ms: 150,
            },
            RagDebugStep::ContextualCompression {
                input_chunks: 10,
                input_tokens: 2000,
                output_chunks: 10,
                output_tokens: 1000,
                compression_ratio: 2.0,
                duration_ms: 300,
            },
            RagDebugStep::SelfReflection {
                query: "test".into(),
                context_summary: "summary".into(),
                is_sufficient: true,
                confidence: 0.85,
                reason: Some("reason".into()),
                duration_ms: 100,
            },
            RagDebugStep::AgenticIteration {
                iteration: 1,
                action: "search".into(),
                observation: "found results".into(),
                is_complete: false,
                duration_ms: 200,
            },
            RagDebugStep::GraphTraversal {
                start_entities: vec!["entity1".into()],
                traversal_depth: 2,
                nodes_visited: 10,
                relationships_found: 5,
                duration_ms: 50,
            },
            RagDebugStep::Error {
                step: "retrieval".into(),
                message: "error".into(),
                recoverable: true,
            },
            RagDebugStep::Warning {
                step: "rerank".into(),
                message: "warning".into(),
            },
        ];

        assert!(!_steps.is_empty());
    }
}

// ============================================================================
// RAG METHODS TESTS
// ============================================================================

#[cfg(feature = "rag")]
mod rag_methods_tests {
    use ai_assistant::rag_methods::*;
    use std::collections::HashMap;

    // --- Mock LLM for testing ---

    struct MockLlm {
        response: String,
    }

    impl MockLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.into(),
            }
        }
    }

    impl LlmGenerate for MockLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
            Ok(self.response.clone())
        }
        fn model_name(&self) -> &str {
            "mock-llm"
        }
    }

    // --- ScoredItem Tests ---

    #[test]
    fn test_scored_item_creation() {
        let item = ScoredItem::new("test".to_string(), 0.85);
        assert_eq!(item.item, "test");
        assert_eq!(item.score, 0.85);
        assert!(item.metadata.is_empty());
    }

    #[test]
    fn test_scored_item_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test.md".to_string());
        meta.insert("chunk_id".to_string(), "123".to_string());

        let item = ScoredItem::with_metadata("test".to_string(), 0.9, meta);
        assert_eq!(item.metadata.get("source"), Some(&"test.md".to_string()));
        assert_eq!(item.metadata.get("chunk_id"), Some(&"123".to_string()));
    }

    // --- MethodResult Tests ---

    #[test]
    fn test_method_result_creation() {
        let result = MethodResult::new(vec![1, 2, 3], std::time::Duration::from_millis(100));
        assert_eq!(result.result, vec![1, 2, 3]);
        assert_eq!(result.duration_ms, 100);
    }

    #[test]
    fn test_method_result_with_details() {
        let result = MethodResult::new("test", std::time::Duration::from_millis(50))
            .with_details("method", "llm")
            .with_details("model", "test-model");

        assert_eq!(result.details.get("method"), Some(&"llm".to_string()));
        assert_eq!(result.details.get("model"), Some(&"test-model".to_string()));
    }

    // --- QueryExpander Tests ---

    #[test]
    fn test_query_expander_synonym_expand() {
        let expander = QueryExpander::new();

        let expanded = expander.synonym_expand("What is the ship price?");
        assert!(!expanded.is_empty());
        // Should contain cost/vessel variants
        assert!(expanded
            .iter()
            .any(|s| s.contains("cost") || s.contains("vessel")));
    }

    #[test]
    fn test_query_expander_synonym_expand_no_match() {
        let expander = QueryExpander::new();

        let expanded = expander.synonym_expand("random words here");
        // May or may not have expansions depending on words
        assert!(expanded.len() <= 3);
    }

    #[test]
    fn test_query_expander_config() {
        let config = QueryExpanderConfig {
            max_expansions: 3,
            use_synonyms: false,
            prompt_template: Some("Custom: {query}".into()),
        };
        let expander = QueryExpander::with_config(config);

        // With synonyms disabled, synonym_expand should still work but expand() won't use it
        let synonyms = expander.synonym_expand("test ship");
        assert!(!synonyms.is_empty()); // synonym_expand ignores use_synonyms flag
    }

    #[test]
    fn test_query_expander_with_llm() {
        let expander = QueryExpander::new();
        let llm = MockLlm::new("Alternative 1\nAlternative 2\nAlternative 3");

        let result = expander.expand("test query", &llm).unwrap();
        assert!(!result.result.is_empty());
        // duration_ms is u64 (always >= 0), just verify the field is accessible
        let _ = result.duration_ms;
    }

    // --- MultiQueryDecomposer Tests ---

    #[test]
    fn test_multi_query_complexity_simple() {
        let decomposer = MultiQueryDecomposer::new();

        let simple = decomposer.estimate_complexity("What is Rust?");
        assert!(simple < 0.3);
    }

    #[test]
    fn test_multi_query_complexity_complex() {
        let decomposer = MultiQueryDecomposer::new();

        let complex = decomposer.estimate_complexity(
            "Compare Rust and Python, also tell me which is better for systems programming and web development?"
        );
        assert!(complex > 0.3);
    }

    #[test]
    fn test_multi_query_complexity_factors() {
        let decomposer = MultiQueryDecomposer::new();

        // Multiple question marks increase complexity
        let multi_q = decomposer.estimate_complexity("What is A? What is B?");
        let single_q = decomposer.estimate_complexity("What is A");
        assert!(multi_q > single_q);

        // Comparisons increase complexity
        let compare = decomposer.estimate_complexity("Compare A versus B");
        let no_compare = decomposer.estimate_complexity("Tell me about A");
        assert!(compare > no_compare);
    }

    #[test]
    fn test_multi_query_decompose_simple() {
        let decomposer = MultiQueryDecomposer::new();
        let llm = MockLlm::new("Sub-query 1\nSub-query 2");

        let result = decomposer.decompose("Simple query", &llm).unwrap();
        // Simple query should not be decomposed (below threshold)
        assert!(result.details.contains_key("skipped") || result.result.len() == 1);
    }

    #[test]
    fn test_multi_query_config() {
        let config = MultiQueryConfig {
            max_sub_queries: 2,
            min_complexity_threshold: 0.1, // Very low threshold
            prompt_template: None,
        };
        let decomposer = MultiQueryDecomposer::with_config(config);
        let llm = MockLlm::new("Sub 1\nSub 2\nSub 3");

        let result = decomposer.decompose("test and more test", &llm).unwrap();
        // Should include original + at most 2 sub-queries
        assert!(result.result.len() <= 3);
    }

    // --- HyDE Tests ---

    #[test]
    fn test_hyde_generator() {
        let hyde = HydeGenerator::new();
        let llm = MockLlm::new("This is a hypothetical answer to the question.");

        let result = hyde
            .generate("What is the capital of France?", &llm)
            .unwrap();
        assert!(!result.result.is_empty());
        assert_eq!(
            result.result[0],
            "This is a hypothetical answer to the question."
        );
    }

    #[test]
    fn test_hyde_config() {
        let config = HydeConfig {
            target_length: 100,
            num_hypotheticals: 3,
            prompt_template: Some("Custom: {query}".into()),
        };
        let hyde = HydeGenerator::with_config(config);
        let llm = MockLlm::new("Hypothetical doc");

        let result = hyde.generate("test", &llm).unwrap();
        assert_eq!(result.result.len(), 3);
    }

    // --- RRF Fusion Tests ---

    #[test]
    fn test_rrf_fusion_basic() {
        let fusion = RrfFusion::new();

        let list1 = vec![
            ScoredItem::new("doc1".to_string(), 0.9),
            ScoredItem::new("doc2".to_string(), 0.8),
            ScoredItem::new("doc3".to_string(), 0.7),
        ];
        let list2 = vec![
            ScoredItem::new("doc2".to_string(), 0.95),
            ScoredItem::new("doc4".to_string(), 0.85),
            ScoredItem::new("doc1".to_string(), 0.75),
        ];

        let result = fusion.fuse_strings(vec![list1, list2]);
        assert!(!result.result.is_empty());

        // doc2 appears first in list2 and second in list1, should have high score
        let doc2 = result.result.iter().find(|i| i.item == "doc2");
        assert!(doc2.is_some());
    }

    #[test]
    fn test_rrf_fusion_single_list() {
        let fusion = RrfFusion::new();

        let list = vec![
            ScoredItem::new("doc1".to_string(), 0.9),
            ScoredItem::new("doc2".to_string(), 0.8),
        ];

        let result = fusion.fuse_strings(vec![list]);
        assert_eq!(result.result.len(), 2);
    }

    #[test]
    fn test_rrf_fusion_empty() {
        let fusion = RrfFusion::new();
        let result = fusion.fuse_strings(vec![]);
        assert!(result.result.is_empty());
    }

    #[test]
    fn test_rrf_config() {
        let config = RrfConfig {
            k: 30.0, // Different k value
            max_results: 5,
        };
        let fusion = RrfFusion::with_config(config);

        let list1 = vec![
            ScoredItem::new("doc1".to_string(), 0.9),
            ScoredItem::new("doc2".to_string(), 0.8),
            ScoredItem::new("doc3".to_string(), 0.7),
        ];
        let list2 = vec![
            ScoredItem::new("doc4".to_string(), 0.95),
            ScoredItem::new("doc5".to_string(), 0.85),
            ScoredItem::new("doc6".to_string(), 0.75),
        ];

        let result = fusion.fuse_strings(vec![list1, list2]);
        assert!(result.result.len() <= 5);
    }

    // --- Contextual Compression Tests ---

    #[test]
    fn test_contextual_compressor() {
        let compressor = ContextualCompressor::new();
        let llm = MockLlm::new("The relevant extract with important details.");

        let result = compressor
            .compress(
                "What is X?",
                "Long content about X and many other things...",
                &llm,
            )
            .unwrap();
        assert!(!result.result.is_empty());
    }

    #[test]
    fn test_contextual_compressor_config() {
        let config = CompressionConfig {
            target_tokens: 50,
            min_tokens: 10,
            prompt_template: Some("Extract relevant: {query}\n{content}".into()),
        };
        let compressor = ContextualCompressor::with_config(config);
        let llm = MockLlm::new("Short compressed result");

        let result = compressor.compress("query", "content", &llm).unwrap();
        assert!(!result.result.is_empty());
    }

    // --- Self-RAG Evaluator Tests ---

    #[test]
    fn test_self_rag_evaluator() {
        let evaluator = SelfRagEvaluator::new();
        let llm = MockLlm::new("YES|85|Context contains relevant information");

        let result = evaluator
            .evaluate("What is X?", "X is a programming language.", &llm)
            .unwrap();
        assert!(result.result.is_sufficient);
        assert!((result.result.confidence - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_self_rag_evaluator_insufficient() {
        let evaluator = SelfRagEvaluator::new();
        let llm = MockLlm::new("NO|25|Context lacks specific details");

        let result = evaluator
            .evaluate("What is X?", "Irrelevant content", &llm)
            .unwrap();
        assert!(!result.result.is_sufficient);
        assert!(result.result.confidence < 0.5);
    }

    #[test]
    fn test_self_reflection_actions() {
        let evaluator = SelfRagEvaluator::with_config(SelfRagConfig {
            confidence_threshold: 0.6,
            context_preview_length: 500,
        });

        // High confidence -> UseAsIs
        let llm_high = MockLlm::new("YES|90|Good");
        let result = evaluator.evaluate("q", "c", &llm_high).unwrap();
        assert!(matches!(
            result.result.suggested_action,
            SelfReflectionAction::UseAsIs
        ));

        // Low confidence -> ExpandSearch
        let llm_low = MockLlm::new("NO|20|Bad");
        let result = evaluator.evaluate("q", "c", &llm_low).unwrap();
        assert!(matches!(
            result.result.suggested_action,
            SelfReflectionAction::ExpandSearch
        ));
    }

    // --- CRAG Evaluator Tests ---

    #[test]
    fn test_crag_evaluator() {
        let evaluator = CragEvaluator::new();
        let llm = MockLlm::new("80|Documents are highly relevant");

        let result = evaluator
            .evaluate("query", &["doc1", "doc2"], &llm)
            .unwrap();
        assert!((result.result.quality_score - 0.8).abs() < 0.01);
        assert!(matches!(result.result.action, CragAction::Correct));
    }

    #[test]
    fn test_crag_evaluator_actions() {
        let config = CragConfig {
            correct_threshold: 0.7,
            ambiguous_threshold: 0.4,
        };
        let evaluator = CragEvaluator::with_config(config);

        // High quality -> Correct
        let llm = MockLlm::new("85|Good");
        let result = evaluator.evaluate("q", &["d"], &llm).unwrap();
        assert!(matches!(result.result.action, CragAction::Correct));

        // Medium quality -> Ambiguous
        let llm = MockLlm::new("50|Mediocre");
        let result = evaluator.evaluate("q", &["d"], &llm).unwrap();
        assert!(matches!(result.result.action, CragAction::Ambiguous));

        // Low quality -> Incorrect
        let llm = MockLlm::new("20|Poor");
        let result = evaluator.evaluate("q", &["d"], &llm).unwrap();
        assert!(matches!(result.result.action, CragAction::Incorrect));
    }

    // --- Adaptive Strategy Selector Tests ---

    #[test]
    fn test_adaptive_strategy_heuristic_technical() {
        let selector = AdaptiveStrategySelector::new();

        let query = "Aurora MR specifications and stats";
        let strategy = selector.select_heuristic(query);
        assert_eq!(strategy, RetrievalStrategy::HybridKeywordHeavy);
    }

    #[test]
    fn test_adaptive_strategy_heuristic_conceptual() {
        let selector = AdaptiveStrategySelector::new();

        let query = "How does the quantum drive work?";
        let strategy = selector.select_heuristic(query);
        assert_eq!(strategy, RetrievalStrategy::HybridSemanticHeavy);
    }

    #[test]
    fn test_adaptive_strategy_heuristic_comparison() {
        let selector = AdaptiveStrategySelector::new();

        let query = "Compare Aurora versus Mustang";
        let strategy = selector.select_heuristic(query);
        assert_eq!(strategy, RetrievalStrategy::AgenticIterative);
    }

    #[test]
    fn test_adaptive_strategy_heuristic_complex() {
        let selector = AdaptiveStrategySelector::new();

        let query = "This is a very long and complex query that contains multiple aspects \
                    and different considerations about the topic at hand and also has \
                    comparisons between different options and requires multi-step reasoning";
        let strategy = selector.select_heuristic(query);
        assert_eq!(strategy, RetrievalStrategy::MultiQuery);
    }

    #[test]
    fn test_adaptive_strategy_with_llm() {
        let selector = AdaptiveStrategySelector::new();
        let llm = MockLlm::new("HYBRID");

        let result = selector.select_with_llm("test query", &llm).unwrap();
        assert_eq!(result.result, RetrievalStrategy::HybridBalanced);
    }

    #[test]
    fn test_adaptive_strategy_with_llm_disabled() {
        let config = AdaptiveStrategyConfig { use_llm: false };
        let selector = AdaptiveStrategySelector::with_config(config);
        let llm = MockLlm::new("KEYWORD"); // Should be ignored

        let result = selector.select_with_llm("How does X work?", &llm).unwrap();
        // Should use heuristic, which detects conceptual query
        assert_eq!(result.result, RetrievalStrategy::HybridSemanticHeavy);
        assert_eq!(result.details.get("method"), Some(&"heuristic".to_string()));
    }

    // --- LLM Reranker Tests ---

    #[test]
    fn test_llm_reranker() {
        let reranker = LlmReranker::new();
        let llm = MockLlm::new("3, 1, 2");

        let items = vec![
            ScoredItem::new("doc1".to_string(), 0.9),
            ScoredItem::new("doc2".to_string(), 0.8),
            ScoredItem::new("doc3".to_string(), 0.7),
        ];

        let result = reranker.rerank("query", items, &llm).unwrap();
        assert_eq!(result.result.len(), 3);
        // First item should be doc3 (ranked 1st by LLM)
        assert_eq!(result.result[0].item, "doc3");
    }

    #[test]
    fn test_llm_reranker_empty() {
        let reranker = LlmReranker::new();
        let llm = MockLlm::new("");

        let result = reranker.rerank::<String>("query", vec![], &llm).unwrap();
        assert!(result.result.is_empty());
    }

    #[test]
    fn test_llm_reranker_config() {
        let config = LlmRerankerConfig {
            max_chunks: 5,
            chunk_preview_length: 100,
            prompt_template: Some("Custom prompt".into()),
        };
        let reranker = LlmReranker::with_config(config);
        let llm = MockLlm::new("1");

        let items: Vec<ScoredItem<String>> = (0..10)
            .map(|i| ScoredItem::new(format!("doc{}", i), 1.0 - i as f32 * 0.1))
            .collect();

        let result = reranker.rerank("query", items, &llm).unwrap();
        // Should only process max_chunks items
        assert!(result.result.len() <= 5);
    }

    // --- Graph RAG Types Tests ---

    #[test]
    fn test_graph_rag_entity() {
        let entity = Entity {
            name: "Aurora MR".into(),
            entity_type: "SHIP".into(),
            mentions: vec![EntityMention {
                text: "Aurora MR".into(),
                start: 0,
                end: 9,
                confidence: 0.95,
            }],
        };

        assert_eq!(entity.name, "Aurora MR");
        assert_eq!(entity.entity_type, "SHIP");
        assert_eq!(entity.mentions.len(), 1);
    }

    #[test]
    fn test_graph_rag_relationship() {
        let rel = Relationship {
            from_entity: "Aurora MR".into(),
            to_entity: "RSI".into(),
            relation_type: "manufactured_by".into(),
            weight: 1.0,
            source_chunk: Some("chunk_123".into()),
        };

        assert_eq!(rel.from_entity, "Aurora MR");
        assert_eq!(rel.to_entity, "RSI");
        assert_eq!(rel.relation_type, "manufactured_by");
    }

    #[test]
    fn test_graph_rag_retriever_extract_entities() {
        let config = GraphRagConfig {
            max_depth: 2,
            max_entities: 10,
            entity_types: vec!["SHIP".into(), "MANUFACTURER".into()],
        };
        let retriever = GraphRagRetriever::new(config);
        let llm = MockLlm::new("SHIP: Aurora MR\nMANUFACTURER: RSI\nSHIP: Mustang Alpha");

        let result = retriever
            .extract_entities("Text about Aurora MR made by RSI and Mustang Alpha", &llm)
            .unwrap();
        assert_eq!(result.result.len(), 3);
        assert!(result.result.iter().any(|e| e.name == "Aurora MR"));
        assert!(result.result.iter().any(|e| e.name == "RSI"));
    }

    // --- RAPTOR Tests ---

    #[test]
    fn test_raptor_config() {
        let config = RaptorConfig {
            max_levels: 4,
            chunks_per_summary: 10,
            summary_length: 300,
        };

        assert_eq!(config.max_levels, 4);
        assert_eq!(config.chunks_per_summary, 10);
        assert_eq!(config.summary_length, 300);
    }

    #[test]
    fn test_raptor_retriever_summarize() {
        let retriever = RaptorRetriever::new(RaptorConfig::default());
        let llm = MockLlm::new("Summary of the combined chunks.");

        let chunks = vec!["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"];
        let result = retriever.summarize_group(&chunks, &llm).unwrap();

        assert_eq!(result.result, "Summary of the combined chunks.");
    }

    #[test]
    fn test_raptor_node() {
        let node = RaptorNode {
            id: "node_1".into(),
            level: 2,
            content: "Summary of children".into(),
            children: vec!["child_1".into(), "child_2".into()],
            embedding: Some(vec![0.1, 0.2, 0.3]),
        };

        assert_eq!(node.level, 2);
        assert_eq!(node.children.len(), 2);
        assert!(node.embedding.is_some());
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(feature = "rag")]
mod rag_integration_tests {
    use ai_assistant::rag_debug::*;
    use ai_assistant::rag_tiers::*;
    use std::time::Duration;

    #[test]
    fn test_full_rag_workflow_simulation() {
        // Setup debug logger
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        // Create config
        let config = RagConfig::with_tier(RagTier::Enhanced).with_max_chunks(10);

        // Verify config
        assert!(config.is_feature_enabled("fts_search"));
        assert!(config.is_feature_enabled("semantic_search"));
        assert!(config.is_feature_enabled("reranking"));

        // Start debug session
        let session = logger.start_query("What ships are good for cargo?");
        session.set_tier("Enhanced");
        session.set_features(
            config
                .effective_features()
                .enabled_features()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        );

        // Simulate query expansion
        session.log_expansion(
            "What ships are good for cargo?",
            vec!["cargo ships".into(), "freight vessels".into()],
            "synonym",
            Duration::from_millis(5),
        );

        // Simulate keyword search
        session.log_keyword_search("cargo ships", 15, Some(0.85), Duration::from_millis(30));

        // Simulate semantic search
        session.log_semantic_search(
            "cargo ships",
            "nomic-embed",
            20,
            Some(0.92),
            Duration::from_millis(50),
        );

        // Simulate LLM reranking
        session.log_llm_call("reranking", "llama3", 500, 50, Duration::from_millis(200));

        // Complete session
        session.complete_with_response("The Hull series ships are best for cargo...");

        // Verify session was recorded
        let sessions = logger.all_sessions();
        assert_eq!(sessions.len(), 1);

        let s = &sessions[0];
        assert_eq!(s.stats.llm_calls, 1);
        assert_eq!(s.steps.len(), 4);
        assert!(s.final_response.is_some());
    }

    #[test]
    fn test_tier_progression() {
        // Test that higher tiers include features from lower tiers
        let fast_features = RagTier::Fast.to_features();
        let semantic_features = RagTier::Semantic.to_features();
        let enhanced_features = RagTier::Enhanced.to_features();

        // Semantic should have all Fast features
        assert!(fast_features.fts_search);
        assert!(semantic_features.fts_search);

        // Enhanced should have all Semantic features
        assert!(enhanced_features.fts_search);
        assert!(enhanced_features.semantic_search);
        assert!(enhanced_features.hybrid_search);

        // Plus additional features
        assert!(!semantic_features.reranking);
        assert!(enhanced_features.reranking);
    }

    #[test]
    fn test_custom_tier_flexibility() {
        // Create a custom config with specific features
        let mut features = RagFeatures::none();
        features.fts_search = true;
        features.agentic_mode = true; // Skip straight to agentic
        features.graph_rag = true; // Add graph

        let config = RagConfig::with_features(features);

        // Check requirements
        let reqs = config.check_requirements();
        assert!(reqs.contains(&RagRequirement::GraphDatabase));
        // Should not require embeddings since semantic_search is off
        assert!(!reqs.contains(&RagRequirement::EmbeddingModel));

        // Estimate calls should be unbounded due to agentic
        let (_, max) = config.estimate_extra_calls();
        assert!(max.is_none());
    }

    #[test]
    fn test_stats_accumulation() {
        let mut stats = RagStats::default();

        // Simulate multiple queries
        for i in 0..10 {
            stats.record_query();
            stats.record_llm_calls(i % 3 + 1, (i * 100) as u64);
            stats.record_retrieval((i + 1) * 5, (i + 1) * 3, (i * 50) as u64);
            stats.record_feature("reranking");
        }

        assert_eq!(stats.queries_processed, 10);
        assert!(stats.llm_calls > 10); // Multiple calls per query
        assert!(stats.chunks_retrieved > 0);
        assert_eq!(*stats.feature_usage.get("reranking").unwrap(), 10);

        let summary = stats.summary();
        assert!(summary.contains("10 queries"));
    }
}
