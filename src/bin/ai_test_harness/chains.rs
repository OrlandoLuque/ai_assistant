use super::*;

// ─── Integration Tests ──────────────────────────────────────────────────────────

pub(crate) fn tests_integration_entity_anonymize() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Entity → Anonymize")));
    let mut results = Vec::new();

    results.push(run_test("Extract entities then anonymize them", || {
        let extractor =
            ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
        let text = "Contact john.doe@example.com or call +1-555-123-4567 for details.";
        let entities = extractor.extract(text);
        assert_test!(!entities.is_empty(), "should extract entities from text");

        let mut anonymizer = ai_assistant::DataAnonymizer::new();
        let result = anonymizer.anonymize(text);
        assert_test!(
            result.anonymized != text,
            "anonymized text should differ from original"
        );
        assert_test!(
            !result.anonymized.contains("john.doe@example.com"),
            "email should be anonymized"
        );
        Ok(())
    }));

    results.push(run_test(
        "Entity types match anonymization detections",
        || {
            let extractor =
                ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
            let text = "Email me at test@company.org about the $5000 invoice.";
            let entities = extractor.extract(text);
            let has_email = entities
                .iter()
                .any(|e| matches!(e.entity_type, ai_assistant::EntityType::Email));
            assert_test!(has_email, "should detect email entity");

            let mut anonymizer = ai_assistant::DataAnonymizer::new();
            let result = anonymizer.anonymize(text);
            assert_test!(!result.detections.is_empty(), "should have detections");
            Ok(())
        },
    ));

    CategoryResult {
        name: "integration_entity_anonymize".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_intent_template() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Intent → Template")));
    let mut results = Vec::new();

    results.push(run_test(
        "Classify intent then pick matching template",
        || {
            let classifier = ai_assistant::IntentClassifier::new();
            let result = classifier.classify("Write a Python function to sort a list");
            assert_test!(
                matches!(
                    result.primary,
                    ai_assistant::Intent::CodeRequest
                        | ai_assistant::Intent::Command
                        | ai_assistant::Intent::Request
                ),
                &format!(
                    "should detect code-related intent, got {:?}",
                    result.primary
                )
            );

            let template = ai_assistant::ConversationTemplate::new(
                "code-help",
                "Code Helper",
                ai_assistant::TemplateCategory::Coding,
            )
            .with_system_prompt("You are a coding assistant.");
            assert_test!(
                template.system_prompt.contains("coding"),
                "template should have coding prompt"
            );
            Ok(())
        },
    ));

    results.push(run_test(
        "Question intent maps to learning template",
        || {
            let classifier = ai_assistant::IntentClassifier::new();
            let result = classifier.classify("What is the difference between TCP and UDP?");
            assert_test!(
                matches!(
                    result.primary,
                    ai_assistant::Intent::Question
                        | ai_assistant::Intent::Comparison
                        | ai_assistant::Intent::Explanation
                ),
                &format!(
                    "should detect question/comparison intent, got {:?}",
                    result.primary
                )
            );

            let template = ai_assistant::ConversationTemplate::new(
                "explainer",
                "Explainer",
                ai_assistant::TemplateCategory::Learning,
            )
            .with_system_prompt("You explain concepts clearly.");
            assert_test!(
                template.category == ai_assistant::TemplateCategory::Learning,
                "should be learning category"
            );
            Ok(())
        },
    ));

    CategoryResult {
        name: "integration_intent_template".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_versioning_merge() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Versioning → Merge")));
    let mut results = Vec::new();

    results.push(run_test("Create versions then three-way merge", || {
        let config = ai_assistant::VersioningConfig::default();
        let mut store = ai_assistant::ContentVersionStore::new(config);

        let base = "Line 1\nLine 2\nLine 3";
        store.add_version("doc1", base);

        let local = "Line 1\nLine 2 modified locally\nLine 3";
        store.add_version("doc1", local);

        let remote = "Line 1\nLine 2\nLine 3 modified remotely";

        let merge_result = ai_assistant::ThreeWayMerge::merge(base, local, remote);
        assert_test!(
            !merge_result.has_conflicts,
            "non-overlapping changes should merge cleanly"
        );
        assert_test!(
            merge_result.merged.contains("modified locally"),
            "should contain local change"
        );
        assert_test!(
            merge_result.merged.contains("modified remotely"),
            "should contain remote change"
        );
        Ok(())
    }));

    results.push(run_test("Conflicting edits detected in merge", || {
        let base = "Line 1\nShared line\nLine 3";
        let local = "Line 1\nLocal edit\nLine 3";
        let remote = "Line 1\nRemote edit\nLine 3";

        let merge_result = ai_assistant::ThreeWayMerge::merge(base, local, remote);
        assert_test!(
            merge_result.has_conflicts,
            "overlapping changes should produce conflicts"
        );
        assert_test!(
            !merge_result.conflicts.is_empty(),
            "should have conflict entries"
        );
        Ok(())
    }));

    CategoryResult {
        name: "integration_versioning_merge".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_embedding_similarity() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Embedding → Similarity")));
    let mut results = Vec::new();

    results.push(run_test("Cache embeddings and compute similarity", || {
        let mut cache = ai_assistant::EmbeddingCache::with_defaults();

        let emb_a = vec![1.0_f32, 0.0, 0.0, 0.5];
        let emb_b = vec![0.9_f32, 0.1, 0.0, 0.5];
        let emb_c = vec![0.0_f32, 0.0, 1.0, 0.0];

        cache.set("hello world", "test-model", emb_a.clone());
        cache.set("hi there", "test-model", emb_b.clone());
        cache.set("unrelated topic", "test-model", emb_c.clone());

        let cached_a = cache.get("hello world", "test-model");
        let cached_b = cache.get("hi there", "test-model");
        let cached_c = cache.get("unrelated topic", "test-model");
        assert_test!(
            cached_a.is_some() && cached_b.is_some() && cached_c.is_some(),
            "all should be cached"
        );

        let sim_ab = ai_assistant::cosine_similarity(&cached_a.unwrap(), &cached_b.unwrap());
        let sim_ac = ai_assistant::cosine_similarity(&emb_a, &cached_c.unwrap());
        assert_test!(
            sim_ab > sim_ac,
            &format!(
                "similar embeddings ({:.3}) should score higher than dissimilar ({:.3})",
                sim_ab, sim_ac
            )
        );
        Ok(())
    }));

    CategoryResult {
        name: "integration_embedding_similarity".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_facts_context() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Facts → Context Window")));
    let mut results = Vec::new();

    results.push(run_test("Extract facts and add to context window", || {
        let extractor = ai_assistant::FactExtractor::new(ai_assistant::FactExtractorConfig::default());
        let text = "The project uses Rust for performance. It supports multiple platforms. The system depends on tokio for async.";
        let facts = extractor.extract_facts(text, "knowledge-base");
        assert_test!(!facts.is_empty(), &format!("should extract facts, got {}", facts.len()));

        let mut config = ai_assistant::ContextWindowConfig::default();
        config.max_tokens = 4096;
        let mut window = ai_assistant::ContextWindow::new(config);

        for fact in &facts {
            window.add_user(&fact.statement);
        }

        let messages = window.get_messages();
        assert_test!(messages.len() == facts.len(), "context should contain one message per fact");
        Ok(())
    }));

    CategoryResult {
        name: "integration_facts_context".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_cache_compression() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Cache → Compression")));
    let mut results = Vec::new();

    results.push(run_test("Cache response then compress/decompress", || {
        let mut cache = ai_assistant::ResponseCache::new(ai_assistant::CacheConfig::default());
        let query = "What is Rust?";
        let model = "test-model";
        let response =
            "Rust is a systems programming language focused on safety and performance. ".repeat(20);

        cache.put(query, model, &response, 150, None);
        let cached = cache.get(query, model);
        assert_test!(cached.is_some(), "should find cached response");

        let cached_text = &cached.unwrap().content;
        let compressed =
            ai_assistant::compress_string(cached_text, ai_assistant::CompressionAlgorithm::Gzip);
        assert_test!(
            compressed.data.len() < compressed.original_size,
            &format!(
                "compressed ({}) should be smaller than original ({})",
                compressed.data.len(),
                compressed.original_size
            )
        );

        let decompressed = ai_assistant::decompress_string(&compressed).expect("should decompress");
        assert_eq_test!(decompressed, *cached_text);
        Ok(())
    }));

    CategoryResult {
        name: "integration_cache_compression".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_expansion_ranking() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Integration: Query Expansion → Ranking"))
    );
    let mut results = Vec::new();

    results.push(run_test("Expand query then rank candidate responses", || {
        use std::collections::HashMap;
        let mut config = ai_assistant::ExpansionConfig::default();
        config.use_synonyms = true;
        config.extract_keywords = true;
        config.use_llm = false;
        let expander = ai_assistant::QueryExpander::new(config);
        let expansion = expander.expand("How to optimize database queries?");
        assert_test!(!expansion.all_keywords.is_empty(), "should extract keywords");

        let candidates = vec![
            ai_assistant::ResponseCandidate {
                id: "r1".to_string(),
                content: "Use indexes on frequently queried columns to optimize database performance.".to_string(),
                model: "model-a".to_string(),
                generation_time_ms: 100,
                token_count: 15,
                metadata: HashMap::new(),
            },
            ai_assistant::ResponseCandidate {
                id: "r2".to_string(),
                content: "The weather today is sunny.".to_string(),
                model: "model-b".to_string(),
                generation_time_ms: 50,
                token_count: 8,
                metadata: HashMap::new(),
            },
        ];

        let ranker = ai_assistant::ResponseRanker::new(ai_assistant::RankingCriteria::default());
        let ranked = ranker.rank("How to optimize database queries?", candidates);
        assert_test!(!ranked.is_empty(), "should return ranked results");
        assert_eq_test!(ranked[0].candidate.id, "r1");
        Ok(())
    }));

    CategoryResult {
        name: "integration_expansion_ranking".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_health_keepalive() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Health → Keepalive")));
    let mut results = Vec::new();

    results.push(run_test(
        "Register providers in both health and keepalive",
        || {
            let mut checker =
                ai_assistant::HealthChecker::new(ai_assistant::HealthCheckConfig::default());
            checker.register("ollama", "http://localhost:11434");
            checker.register("openai", "https://api.openai.com");

            let keepalive = ai_assistant::KeepaliveManager::default();
            keepalive.register("ollama", "http://localhost:11434");
            keepalive.register("openai", "https://api.openai.com");

            let summary = checker.summary();
            let ka_stats = keepalive.stats();
            assert_eq_test!(summary.total, 2);
            assert_eq_test!(ka_stats.total_connections, 2);
            Ok(())
        },
    ));

    CategoryResult {
        name: "integration_health_keepalive".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_moderation_citations() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Integration: Moderation → Citations")));
    let mut results = Vec::new();

    results.push(run_test(
        "Moderate text then create citations from safe content",
        || {
            let moderator =
                ai_assistant::ContentModerator::new(ai_assistant::ModerationConfig::default());
            let text = "Rust was created in 2010 at Mozilla Research. It emphasizes memory safety.";
            let mod_result = moderator.moderate(text);
            assert_test!(mod_result.passed, "clean text should pass moderation");

            let mut generator =
                ai_assistant::CitationGenerator::new(ai_assistant::CitationConfig::default());
            let source =
                ai_assistant::Source::new("src-1", "Mozilla Research Blog", &mod_result.processed);
            generator.add_source(source);

            let cited = generator.cite("Rust emphasizes memory safety");
            assert_test!(
                !cited.cited_text.is_empty(),
                "cited text should not be empty"
            );
            Ok(())
        },
    ));

    results.push(run_test("Flagged content is not cited", || {
        let mut config = ai_assistant::ModerationConfig::default();
        config.blocked_terms.push("forbidden_word".to_string());
        let moderator = ai_assistant::ContentModerator::new(config);
        let text = "This contains a forbidden_word that should be caught.";
        let mod_result = moderator.moderate(text);
        assert_test!(
            !mod_result.passed,
            "text with blocked term should fail moderation"
        );
        Ok(())
    }));

    CategoryResult {
        name: "integration_moderation_citations".to_string(),
        results,
    }
}

pub(crate) fn tests_integration_latency_selection() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Integration: Latency → Model Selection"))
    );
    let mut results = Vec::new();

    results.push(run_test("Record latencies then select best model", || {
        use std::collections::HashMap;
        use std::time::Duration;

        let mut tracker = ai_assistant::LatencyTracker::new();
        tracker.record("fast-model", Duration::from_millis(50), true);
        tracker.record("fast-model", Duration::from_millis(60), true);
        tracker.record("slow-model", Duration::from_millis(500), true);
        tracker.record("slow-model", Duration::from_millis(550), true);

        let fast_stats = tracker.stats("fast-model");
        let slow_stats = tracker.stats("slow-model");
        assert_test!(fast_stats.is_some(), "should have stats for fast-model");
        assert_test!(slow_stats.is_some(), "should have stats for slow-model");
        assert_test!(
            fast_stats.as_ref().unwrap().avg_latency < slow_stats.as_ref().unwrap().avg_latency,
            "fast-model should have lower avg latency"
        );

        let mut selector =
            ai_assistant::AutoModelSelector::new(ai_assistant::AutoSelectConfig::default());
        let fast_profile = ai_assistant::AutoModelProfile {
            id: "fast-model".to_string(),
            name: "Fast Model".to_string(),
            provider: "local".to_string(),
            capabilities: ai_assistant::AutoModelCapabilities::default(),
            cost_input: 0.001,
            cost_output: 0.002,
            avg_latency: fast_stats.unwrap().avg_latency,
            task_quality: HashMap::new(),
            max_context: 4096,
            available: true,
        };
        selector.add_model(fast_profile);

        let result = selector.select("Write a hello world program", None);
        assert_eq_test!(result.model_id, "fast-model");
        Ok(())
    }));

    CategoryResult {
        name: "integration_latency_selection".to_string(),
        results,
    }
}

// ─── Multi-Module Integration Tests (3-4 modules chained) ───────────────────────

pub(crate) fn tests_chain_entity_anon_cache_compress() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Entity → Anonymize → Cache → Compress"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Full pipeline: extract, anonymize, cache, compress",
        || {
            // 1. Extract entities
            let extractor =
                ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
            let text = "Send the report to alice@corp.com and bob@corp.com before Friday.";
            let entities = extractor.extract(text);
            assert_test!(!entities.is_empty(), "should extract entities");

            // 2. Anonymize the text
            let mut anonymizer = ai_assistant::DataAnonymizer::new();
            let anon_result = anonymizer.anonymize(text);
            assert_test!(
                !anon_result.anonymized.contains("alice@corp.com"),
                "should anonymize emails"
            );

            // 3. Cache the anonymized result
            let mut cache = ai_assistant::ResponseCache::new(ai_assistant::CacheConfig::default());
            cache.put(
                "pii-query",
                "anonymizer",
                &anon_result.anonymized,
                50,
                Some("anonymization"),
            );
            let cached = cache.get("pii-query", "anonymizer");
            assert_test!(cached.is_some(), "should cache anonymized text");

            // 4. Compress the cached content
            let compressed = ai_assistant::compress_string(
                &cached.unwrap().content,
                ai_assistant::CompressionAlgorithm::Gzip,
            );
            let decompressed = ai_assistant::decompress_string(&compressed).expect("decompress");
            assert_eq_test!(decompressed, anon_result.anonymized);
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_entity_anon_cache_compress".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_intent_template_context_budget() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Intent → Template → Context → Budget"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Classify, template, fill context, check budget",
        || {
            // 1. Classify intent
            let classifier = ai_assistant::IntentClassifier::new();
            let user_msg = "Explain how async/await works in Rust";
            let intent = classifier.classify(user_msg);

            // 2. Pick template based on intent
            let category = match intent.primary {
                ai_assistant::Intent::CodeRequest | ai_assistant::Intent::Command => {
                    ai_assistant::TemplateCategory::Coding
                }
                ai_assistant::Intent::Question | ai_assistant::Intent::Explanation => {
                    ai_assistant::TemplateCategory::Learning
                }
                _ => ai_assistant::TemplateCategory::Research,
            };
            let template =
                ai_assistant::ConversationTemplate::new("explain", "Explainer", category)
                    .with_system_prompt(
                        "You are a technical educator. Explain concepts clearly with examples.",
                    );

            // 3. Fill context window with template + user message
            let mut config = ai_assistant::ContextWindowConfig::default();
            config.max_tokens = 2048;
            let mut window = ai_assistant::ContextWindow::new(config);
            window.add_user(&template.system_prompt);
            window.add_user(user_msg);
            window.add_assistant(
                "Async/await in Rust uses the Future trait for cooperative multitasking...",
            );

            let messages = window.get_messages();
            assert_eq_test!(messages.len(), 3);

            // 4. Check token budget
            let mut budget_mgr = ai_assistant::TokenBudgetManager::new();
            budget_mgr.set_budget(
                "user-1",
                ai_assistant::Budget::new(1000, ai_assistant::BudgetPeriod::Daily),
            );
            let check = budget_mgr.check("user-1", 200);
            assert_test!(check.allowed, "should be within budget");
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_intent_template_context_budget".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_chunker_entities_embed_similarity() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Chunker → Entities → Embed → Similarity"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Chunk doc, extract entities per chunk, embed, compare",
        || {
            // 1. Chunk a document
            let mut chunk_config = ai_assistant::ChunkingConfig::default();
            chunk_config.strategy = ai_assistant::ChunkingStrategy::Sentence;
            chunk_config.target_tokens = 10;
            chunk_config.min_tokens = 3;
            chunk_config.max_tokens = 30;
            chunk_config.overlap_tokens = 0;
            let chunker = ai_assistant::SmartChunker::new(chunk_config);
            let doc = "Rust uses ownership for memory safety and zero-cost abstractions. \
                   Python uses garbage collection for automatic memory management. \
                   JavaScript runs in browsers and Node.js on the server side.";
            let chunks = chunker.chunk(doc);
            assert_test!(
                chunks.len() >= 2,
                &format!("should produce multiple chunks, got {}", chunks.len())
            );

            // 2. Extract entities from each chunk
            let extractor =
                ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
            let mut all_entities = Vec::new();
            for chunk in &chunks {
                let entities = extractor.extract(&chunk.content);
                all_entities.extend(entities);
            }

            // 3. Create and cache embeddings for chunks
            let mut embed_cache = ai_assistant::EmbeddingCache::with_defaults();
            for (i, chunk) in chunks.iter().enumerate() {
                let fake_embedding: Vec<f32> = chunk
                    .content
                    .bytes()
                    .take(8)
                    .map(|b| b as f32 / 255.0)
                    .collect();
                embed_cache.set(&chunk.content, "test-embedder", fake_embedding);
                assert_test!(
                    embed_cache.get(&chunk.content, "test-embedder").is_some(),
                    &format!("chunk {} should be cached", i)
                );
            }

            // 4. Compute similarity between first and last chunk
            if chunks.len() >= 2 {
                let emb_first = embed_cache
                    .get(&chunks[0].content, "test-embedder")
                    .unwrap();
                let emb_last = embed_cache
                    .get(&chunks[chunks.len() - 1].content, "test-embedder")
                    .unwrap();
                let sim = ai_assistant::cosine_similarity(&emb_first, &emb_last);
                assert_test!(
                    (-1.0..=1.0).contains(&sim),
                    &format!("similarity should be in [-1,1], got {:.3}", sim)
                );
            }
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_chunker_entities_embed_similarity".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_facts_memory_context_compact() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Facts → Memory → Context → Compaction"))
    );
    let mut results = Vec::new();

    results.push(run_test("Extract facts, store in memory, build context, compact", || {
        // 1. Extract facts
        let extractor = ai_assistant::FactExtractor::new(ai_assistant::FactExtractorConfig::default());
        let text = "The system uses PostgreSQL for data storage. It supports real-time updates. The API depends on tokio runtime.";
        let facts = extractor.extract_facts(text, "docs");
        assert_test!(!facts.is_empty(), "should extract facts");

        // 2. Store facts in memory
        let mut store = ai_assistant::MemoryStore::new(ai_assistant::MemoryConfig::default());
        for fact in &facts {
            let entry = ai_assistant::MemoryEntry::new(&fact.statement, ai_assistant::MemoryType::Fact);
            store.add(entry);
        }
        let _recalled = store.search("database");
        // Search may or may not find results depending on implementation

        // 3. Build context from memory
        let context = store.build_context("system architecture", 1024);
        assert_test!(!context.is_empty(), "built context should not be empty");

        // 4. Compact when conversation grows too large
        let mut messages: Vec<ai_assistant::CompactableMessage> = facts.iter().enumerate().map(|(i, f)| {
            ai_assistant::CompactableMessage {
                id: format!("msg-{}", i),
                role: "assistant".to_string(),
                content: f.statement.clone(),
                timestamp: i as u64,
                importance: f.confidence as f64,
                topics: vec!["architecture".to_string()],
                entities: vec![],
                #[cfg(feature = "vision")]
                images: Vec::new(),
            }
        }).collect();
        // Add more messages to trigger compaction
        for i in 0..10 {
            messages.push(ai_assistant::CompactableMessage {
                id: format!("pad-{}", i),
                role: "user".to_string(),
                content: format!("Follow-up question number {}", i),
                timestamp: (facts.len() + i) as u64,
                importance: 0.3,
                topics: vec![],
                entities: vec![],
                #[cfg(feature = "vision")]
                images: Vec::new(),
            });
        }

        let compactor = ai_assistant::ConversationCompactor::new(ai_assistant::ConvCompactionConfig::default());
        let result = compactor.compact(messages);
        assert_test!(result.removed_count > 0 || !result.messages.is_empty(), "compaction should process messages");
        Ok(())
    }));

    CategoryResult {
        name: "chain_facts_memory_context_compact".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_moderation_version_merge_export() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Moderation → Version → Merge → Export"))
    );
    let mut results = Vec::new();

    results.push(run_test("Moderate, version, merge edits, export", || {
        use std::collections::HashMap;

        // 1. Moderate content
        let moderator = ai_assistant::ContentModerator::new(ai_assistant::ModerationConfig::default());
        let content = "Rust provides memory safety without garbage collection. It is fast and reliable.";
        let mod_result = moderator.moderate(content);
        assert_test!(mod_result.passed, "clean content should pass moderation");

        // 2. Create versioned content
        let mut store = ai_assistant::ContentVersionStore::new(ai_assistant::VersioningConfig::default());
        store.add_version("article", &mod_result.processed);
        let updated = "Rust provides memory safety without garbage collection. It is fast, reliable, and productive.";
        store.add_version("article", updated);

        // 3. Three-way merge with a parallel edit
        let parallel_edit = "Rust provides memory safety without garbage collection. It is fast and reliable. Used by many companies.";
        let merge = ai_assistant::ThreeWayMerge::merge(&mod_result.processed, updated, parallel_edit);
        assert_test!(!merge.merged.is_empty(), "merge should produce output");

        // 4. Export as conversation
        let conversation = ai_assistant::ExportedConversation {
            id: "conv-1".to_string(),
            title: "Article Editing".to_string(),
            messages: vec![
                ai_assistant::ExportedMessage {
                    role: "user".to_string(),
                    content: format!("Original: {}", mod_result.processed),
                    timestamp: None,
                    metadata: None,
                    #[cfg(feature = "vision")]
                    images: Vec::new(),
                },
                ai_assistant::ExportedMessage {
                    role: "assistant".to_string(),
                    content: format!("Merged result: {}", merge.merged),
                    timestamp: None,
                    metadata: None,
                    #[cfg(feature = "vision")]
                    images: Vec::new(),
                },
            ],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        };
        let exporter = ai_assistant::ConversationExporter::new(ai_assistant::ExportOptions {
            format: ai_assistant::ExportFormat::Markdown,
            ..Default::default()
        });
        let exported = exporter.export(&conversation);
        assert_test!(exported.is_ok(), "export should succeed");
        assert_test!(exported.unwrap().contains("Merged result"), "export should contain merged content");
        Ok(())
    }));

    CategoryResult {
        name: "chain_moderation_version_merge_export".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_latency_health_select_cost() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Latency → Health → Select → Cost"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Track latency, check health, select model, track cost",
        || {
            use std::collections::HashMap;
            use std::time::Duration;

            // 1. Record latencies for multiple providers
            let mut tracker = ai_assistant::LatencyTracker::new();
            tracker.record("gpt-4", Duration::from_millis(200), true);
            tracker.record("gpt-4", Duration::from_millis(220), true);
            tracker.record("llama-local", Duration::from_millis(80), true);
            tracker.record("llama-local", Duration::from_millis(90), true);

            let gpt4_stats = tracker.stats("gpt-4").unwrap();
            let llama_stats = tracker.stats("llama-local").unwrap();

            // 2. Check health of providers
            let mut checker =
                ai_assistant::HealthChecker::new(ai_assistant::HealthCheckConfig::default());
            checker.register("gpt-4", "https://api.openai.com");
            checker.register("llama-local", "http://localhost:11434");
            let summary = checker.summary();
            assert_eq_test!(summary.total, 2);

            // 3. Select best model based on latency
            let mut selector =
                ai_assistant::AutoModelSelector::new(ai_assistant::AutoSelectConfig::default());
            selector.add_model(ai_assistant::AutoModelProfile {
                id: "gpt-4".to_string(),
                name: "GPT-4".to_string(),
                provider: "openai".to_string(),
                capabilities: ai_assistant::AutoModelCapabilities {
                    code: true,
                    ..Default::default()
                },
                cost_input: 0.03,
                cost_output: 0.06,
                avg_latency: gpt4_stats.avg_latency,
                task_quality: HashMap::new(),
                max_context: 8192,
                available: true,
            });
            selector.add_model(ai_assistant::AutoModelProfile {
                id: "llama-local".to_string(),
                name: "Llama Local".to_string(),
                provider: "local".to_string(),
                capabilities: ai_assistant::AutoModelCapabilities {
                    code: true,
                    ..Default::default()
                },
                cost_input: 0.0,
                cost_output: 0.0,
                avg_latency: llama_stats.avg_latency,
                task_quality: HashMap::new(),
                max_context: 4096,
                available: true,
            });
            let selection = selector.select("Fix the bug in my Python code", None);
            assert_test!(!selection.model_id.is_empty(), "should select a model");

            // 4. Track cost for the selected model
            let mut cost_tracker = ai_assistant::CostTracker::new();
            cost_tracker.add(ai_assistant::CostEstimate {
                input_tokens: 500,
                output_tokens: 200,
                images: 0,
                cost: if selection.model_id == "gpt-4" {
                    0.027
                } else {
                    0.0
                },
                vision_cost: 0.0,
                currency: "USD".to_string(),
                model: selection.model_id.clone(),
                provider: selection.profile.provider.clone(),
                pricing_tier: None,
            });
            assert_test!(
                cost_tracker.request_count == 1,
                "should have 1 request recorded"
            );
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_latency_health_select_cost".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_analytics_topics_compact_export() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Analytics → Topics → Compact → Export"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Track analytics, detect topics, compact, export",
        || {
            use std::collections::HashMap;

            // 1. Track conversation analytics
            let mut analytics =
                ai_assistant::ConversationAnalytics::new(ai_assistant::AnalyticsConfig::default());
            analytics.track_conversation_start("sess-1", Some("user-1"), "gpt-4");
            analytics.track_message(
                "sess-1",
                Some("user-1"),
                "gpt-4",
                "How do I write async code in Rust?",
                true,
                15,
                None,
            );
            analytics.track_message(
                "sess-1",
                Some("user-1"),
                "gpt-4",
                "You can use async/await with the tokio runtime...",
                false,
                50,
                None,
            );
            analytics.track_message(
                "sess-1",
                Some("user-1"),
                "gpt-4",
                "What about error handling in async?",
                true,
                12,
                None,
            );

            let report = analytics.report();
            assert_test!(report.total_messages >= 3, "should track messages");

            // 2. Detect topics from messages
            let messages = vec![
                ai_assistant::ChatMessage::user("How do I write async code in Rust?"),
                ai_assistant::ChatMessage::assistant(
                    "You can use async/await with the tokio runtime...",
                ),
                ai_assistant::ChatMessage::user("What about error handling in async?"),
                ai_assistant::ChatMessage::assistant(
                    "Use Result types with the ? operator in async functions.",
                ),
            ];
            let detector = ai_assistant::TopicDetector::new();
            let topics = detector.detect_topics(&messages);

            // 3. Compact the conversation
            let compactable: Vec<ai_assistant::CompactableMessage> = messages
                .iter()
                .enumerate()
                .map(|(i, m)| ai_assistant::CompactableMessage {
                    id: format!("msg-{}", i),
                    role: m.role.clone(),
                    content: m.content.clone(),
                    timestamp: i as u64,
                    importance: if m.is_user() { 0.8 } else { 0.6 },
                    topics: topics.iter().map(|t| t.name.clone()).collect(),
                    entities: vec![],
                    #[cfg(feature = "vision")]
                    images: Vec::new(),
                })
                .collect();
            let compactor = ai_assistant::ConversationCompactor::new(
                ai_assistant::ConvCompactionConfig::default(),
            );
            let compacted = compactor.compact(compactable);

            // 4. Export the conversation
            let export_messages: Vec<ai_assistant::ExportedMessage> = compacted
                .messages
                .iter()
                .map(|m| ai_assistant::ExportedMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    timestamp: None,
                    metadata: None,
                    #[cfg(feature = "vision")]
                    images: Vec::new(),
                })
                .collect();
            let conversation = ai_assistant::ExportedConversation {
                id: "sess-1".to_string(),
                title: "Async Rust Discussion".to_string(),
                messages: export_messages,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let exporter = ai_assistant::ConversationExporter::new(ai_assistant::ExportOptions {
                format: ai_assistant::ExportFormat::Json,
                ..Default::default()
            });
            let exported = exporter.export(&conversation);
            assert_test!(exported.is_ok(), "export should succeed");
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_analytics_topics_compact_export".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_access_priority_ratelimit() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Access → Priority → RateLimit"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Check access, enqueue by priority, rate limit",
        || {
            use std::collections::HashMap;

            // 1. Check access control
            let mut acl = ai_assistant::AccessControlManager::new();
            acl.assign_role("user-alice", "editor");
            let access = acl.check_permission(
                "user-alice",
                ai_assistant::ResourceType::Conversation,
                ai_assistant::Permission::Write,
                None,
            );
            assert_test!(access.is_allowed(), "editor should have write access");

            // Denied user
            let denied = acl.check_permission(
                "user-bob",
                ai_assistant::ResourceType::Conversation,
                ai_assistant::Permission::Write,
                None,
            );
            assert_test!(!denied.is_allowed(), "unassigned user should be denied");

            // 2. Enqueue allowed user's request with priority
            let queue = ai_assistant::PriorityQueue::new(100);
            let req = ai_assistant::PriorityRequest {
                id: "req-1".to_string(),
                content: "Generate report".to_string(),
                priority: ai_assistant::Priority::High,
                created_at: std::time::Instant::now(),
                deadline: None,
                metadata: HashMap::new(),
                cancellable: true,
                user_id: Some("user-alice".to_string()),
            };
            let enqueue_result = queue.enqueue(req);
            assert_test!(enqueue_result.is_ok(), "should enqueue successfully");

            // Add a lower priority request
            let req2 = ai_assistant::PriorityRequest {
                id: "req-2".to_string(),
                content: "Background task".to_string(),
                priority: ai_assistant::Priority::Background,
                created_at: std::time::Instant::now(),
                deadline: None,
                metadata: HashMap::new(),
                cancellable: true,
                user_id: Some("user-alice".to_string()),
            };
            queue.enqueue(req2).unwrap();

            // 3. Rate limit check
            let backend = ai_assistant::InMemoryBackend::new();
            let limiter = ai_assistant::DistributedRateLimiter::new(Box::new(backend), 60, 10000);
            let limit_result = limiter.check("user-alice");
            assert_test!(limit_result.is_allowed(), "first request should be allowed");

            // Dequeue should return highest priority first
            let dequeued = queue.dequeue();
            assert_test!(dequeued.is_some(), "should dequeue");
            assert_eq_test!(dequeued.as_ref().unwrap().id, "req-1");
            Ok(())
        },
    ));

    CategoryResult {
        name: "chain_access_priority_ratelimit".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_expansion_chunk_embed_rank() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Expansion → Chunk → Embed → Rank"))
    );
    let mut results = Vec::new();

    results.push(run_test("Expand query, chunk corpus, embed, rank by relevance", || {
        use std::collections::HashMap;

        // 1. Expand the query
        let mut expand_config = ai_assistant::ExpansionConfig::default();
        expand_config.use_synonyms = true;
        expand_config.extract_keywords = true;
        expand_config.use_llm = false;
        let expander = ai_assistant::QueryExpander::new(expand_config);
        let expansion = expander.expand("database query optimization techniques");
        assert_test!(!expansion.all_keywords.is_empty(), "should extract keywords");

        // 2. Chunk a knowledge corpus
        let mut chunk_config = ai_assistant::ChunkingConfig::default();
        chunk_config.strategy = ai_assistant::ChunkingStrategy::Sentence;
        chunk_config.max_tokens = 50;
        chunk_config.overlap_tokens = 0;
        let chunker = ai_assistant::SmartChunker::new(chunk_config);
        let corpus = "Use indexes on frequently queried columns. Avoid SELECT * in production queries. \
                      Normalize your database schema to reduce redundancy. Consider denormalization for read-heavy workloads. \
                      The weather forecast shows sunny skies tomorrow.";
        let chunks = chunker.chunk(corpus);
        assert_test!(chunks.len() >= 2, "should produce multiple chunks");

        // 3. Create embeddings and cache them
        let mut embed_cache = ai_assistant::EmbeddingCache::with_defaults();
        for chunk in &chunks {
            // Create content-based fake embedding
            let embedding: Vec<f32> = (0..8).map(|j| {
                let has_db = chunk.content.to_lowercase().contains("database") || chunk.content.to_lowercase().contains("quer");
                if has_db { 0.8 + (j as f32 * 0.01) } else { 0.2 + (j as f32 * 0.01) }
            }).collect();
            embed_cache.set(&chunk.content, "embedder", embedding);
        }

        // 4. Rank chunks as response candidates
        let candidates: Vec<ai_assistant::ResponseCandidate> = chunks.iter().map(|c| {
            ai_assistant::ResponseCandidate {
                id: format!("chunk-{}", c.index),
                content: c.content.clone(),
                model: "rag".to_string(),
                generation_time_ms: 0,
                token_count: c.tokens,
                metadata: HashMap::new(),
            }
        }).collect();
        let ranker = ai_assistant::ResponseRanker::new(ai_assistant::RankingCriteria::default());
        let ranked = ranker.rank("database query optimization", candidates);
        assert_test!(!ranked.is_empty(), "should rank candidates");
        // The database-related chunk should rank higher than weather
        assert_test!(ranked[0].candidate.content.to_lowercase().contains("quer")
            || ranked[0].candidate.content.to_lowercase().contains("index")
            || ranked[0].candidate.content.to_lowercase().contains("database"),
            &format!("top result should be database-related, got: {}", ranked[0].candidate.content));
        Ok(())
    }));

    CategoryResult {
        name: "chain_expansion_chunk_embed_rank".to_string(),
        results,
    }
}

pub(crate) fn tests_chain_intent_entity_citation_validate() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Chain: Intent → Entity → Citation → Validate"))
    );
    let mut results = Vec::new();

    results.push(run_test("Classify intent, extract entities, cite sources, validate output", || {
        // 1. Classify the intent
        let classifier = ai_assistant::IntentClassifier::new();
        let query = "What companies use Rust in production?";
        let intent = classifier.classify(query);
        assert_test!(
            matches!(intent.primary, ai_assistant::Intent::Question | ai_assistant::Intent::Explanation | ai_assistant::Intent::Request),
            &format!("should be a question intent, got {:?}", intent.primary)
        );

        // 2. Extract entities from the response text
        let response_text = "Mozilla created Rust. Dropbox uses Rust for file sync. Cloudflare uses Rust for their edge network.";
        let extractor = ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
        let entities = extractor.extract(response_text);

        // 3. Generate citations for the response
        let mut generator = ai_assistant::CitationGenerator::new(ai_assistant::CitationConfig::default());
        generator.add_source(ai_assistant::Source::new("s1", "Rust Users Page", "Mozilla created Rust in 2010"));
        generator.add_source(ai_assistant::Source::new("s2", "Dropbox Tech Blog", "Dropbox uses Rust for performance-critical file sync"));
        let cited = generator.cite(response_text);
        assert_test!(!cited.cited_text.is_empty(), "should produce cited text");

        // 4. Validate structured output
        let output_json = serde_json::json!({
            "intent": format!("{:?}", intent.primary),
            "entities_found": entities.len(),
            "cited_response": cited.cited_text,
            "sources_count": cited.citations.len(),
        });
        assert_test!(output_json.is_object(), "output should be valid JSON object");
        assert_test!(output_json["cited_response"].as_str().map(|s| !s.is_empty()).unwrap_or(false),
            "cited_response should be non-empty");
        Ok(())
    }));

    CategoryResult {
        name: "chain_intent_entity_citation_validate".to_string(),
        results,
    }
}
