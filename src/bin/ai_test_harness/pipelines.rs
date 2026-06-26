use super::*;

// ─── End-to-End Pipeline Tests (5-6 modules) ─────────────────────────────────────

pub(crate) fn tests_pipeline_rag() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Pipeline: Full RAG (6 modules)")));
    let mut results = Vec::new();

    results.push(run_test(
        "Chunk → Embed → Expand → Similarity → Rank → Context",
        || {
            use std::collections::HashMap;

            // 1. Chunk corpus into searchable pieces
            let mut chunk_config = ai_assistant::ChunkingConfig::default();
            chunk_config.strategy = ai_assistant::ChunkingStrategy::Sentence;
            chunk_config.target_tokens = 10;
            chunk_config.min_tokens = 3;
            chunk_config.max_tokens = 25;
            chunk_config.overlap_tokens = 0;
            let chunker = ai_assistant::SmartChunker::new(chunk_config);
            let corpus = "Rust ownership prevents data races at compile time. \
                      The borrow checker ensures references are always valid. \
                      Lifetimes annotate how long references live. \
                      Traits define shared behavior across types. \
                      Generics allow writing code that works with many types.";
            let chunks = chunker.chunk(corpus);
            assert_test!(
                chunks.len() >= 2,
                &format!("should produce 2+ chunks, got {}", chunks.len())
            );

            // 2. Embed all chunks
            let mut embed_cache = ai_assistant::EmbeddingCache::with_defaults();
            for chunk in &chunks {
                let has_ownership = chunk.content.to_lowercase().contains("ownership")
                    || chunk.content.to_lowercase().contains("borrow");
                let embedding: Vec<f32> = (0..16)
                    .map(|j| {
                        if has_ownership {
                            0.9 - (j as f32 * 0.01)
                        } else {
                            0.3 + (j as f32 * 0.02)
                        }
                    })
                    .collect();
                embed_cache.set(&chunk.content, "embedder", embedding);
            }

            // 3. Expand user query
            let expander = ai_assistant::QueryExpander::new({
                let mut c = ai_assistant::ExpansionConfig::default();
                c.use_synonyms = true;
                c.extract_keywords = true;
                c.use_llm = false;
                c
            });
            let expansion = expander.expand("How does Rust prevent memory bugs?");
            assert_test!(!expansion.all_keywords.is_empty(), "should expand query");

            // 4. Compute similarity between query embedding and chunk embeddings
            let query_emb: Vec<f32> = (0..16).map(|j| 0.85 - (j as f32 * 0.01)).collect();
            let mut similarities: Vec<(usize, f32)> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let chunk_emb = embed_cache.get(&chunk.content, "embedder").unwrap();
                    let sim = ai_assistant::cosine_similarity(&query_emb, &chunk_emb);
                    (i, sim)
                })
                .collect();
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // 5. Rank top chunks
            let top_chunks: Vec<ai_assistant::ResponseCandidate> = similarities
                .iter()
                .take(3)
                .map(|(i, _)| ai_assistant::ResponseCandidate {
                    id: format!("chunk-{}", i),
                    content: chunks[*i].content.clone(),
                    model: "rag".to_string(),
                    generation_time_ms: 0,
                    token_count: chunks[*i].tokens,
                    metadata: HashMap::new(),
                })
                .collect();
            let ranker =
                ai_assistant::ResponseRanker::new(ai_assistant::RankingCriteria::default());
            let ranked = ranker.rank("How does Rust prevent memory bugs?", top_chunks);
            assert_test!(!ranked.is_empty(), "should produce ranked results");

            // 6. Build context window from ranked results
            let mut config = ai_assistant::ContextWindowConfig::default();
            config.max_tokens = 2048;
            let mut window = ai_assistant::ContextWindow::new(config);
            window.add_user("How does Rust prevent memory bugs?");
            for r in &ranked {
                window.add_assistant(&format!("[Source] {}", r.candidate.content));
            }
            let ctx_messages = window.get_messages();
            assert_test!(
                ctx_messages.len() >= 2,
                "context should have query + sources"
            );
            Ok(())
        },
    ));

    CategoryResult {
        name: "pipeline_rag".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_content_safety() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Content Safety (6 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Intent → Moderate → Anonymize → Entity → Cite → Export",
        || {
            use std::collections::HashMap;

            // 1. Classify intent
            let classifier = ai_assistant::IntentClassifier::new();
            let user_input =
                "Tell me about the security vulnerabilities reported by john@security.org";
            let _intent = classifier.classify(user_input);

            // 2. Moderate the content
            let moderator =
                ai_assistant::ContentModerator::new(ai_assistant::ModerationConfig::default());
            let mod_result = moderator.moderate(user_input);
            assert_test!(mod_result.passed, "query should pass moderation");

            // 3. Anonymize PII in the response
            let response =
                "John Smith (john@security.org) reported CVE-2024-1234. Contact: +1-555-0123.";
            let mut anonymizer = ai_assistant::DataAnonymizer::new();
            let anon = anonymizer.anonymize(response);
            assert_test!(
                !anon.anonymized.contains("john@security.org"),
                "should anonymize email"
            );

            // 4. Extract entities from anonymized text
            let extractor =
                ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
            let _entities = extractor.extract(&anon.anonymized);

            // 5. Cite the sources
            let mut generator =
                ai_assistant::CitationGenerator::new(ai_assistant::CitationConfig::default());
            generator.add_source(ai_assistant::Source::new(
                "cve-db",
                "CVE Database",
                "Security vulnerability records",
            ));
            let cited = generator.cite(&anon.anonymized);
            assert_test!(!cited.cited_text.is_empty(), "should produce cited text");

            // 6. Export as conversation
            let conversation = ai_assistant::ExportedConversation {
                id: "safe-conv-1".to_string(),
                title: "Security Query (Anonymized)".to_string(),
                messages: vec![
                    ai_assistant::ExportedMessage {
                        role: "user".to_string(),
                        content: mod_result.processed.clone(),
                        timestamp: None,
                        metadata: None,
                        #[cfg(feature = "vision")]
                        images: Vec::new(),
                    },
                    ai_assistant::ExportedMessage {
                        role: "assistant".to_string(),
                        content: cited.cited_text.clone(),
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
                format: ai_assistant::ExportFormat::Json,
                ..Default::default()
            });
            let exported = exporter.export(&conversation);
            assert_test!(exported.is_ok(), "should export successfully");
            Ok(())
        },
    ));

    CategoryResult {
        name: "pipeline_content_safety".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_session_lifecycle() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Session Lifecycle (5 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test("Analytics → Topics → Summarize → Compact → Context", || {
        // 1. Build conversation and track analytics
        let mut analytics = ai_assistant::ConversationAnalytics::new(ai_assistant::AnalyticsConfig::default());
        let messages = vec![
            ai_assistant::ChatMessage::user("What is Rust's ownership system?"),
            ai_assistant::ChatMessage::assistant("Ownership is Rust's core memory management concept. Each value has a single owner."),
            ai_assistant::ChatMessage::user("How do borrowing rules work?"),
            ai_assistant::ChatMessage::assistant("You can have either one mutable reference or multiple immutable references."),
            ai_assistant::ChatMessage::user("What are lifetimes?"),
            ai_assistant::ChatMessage::assistant("Lifetimes ensure references don't outlive the data they point to."),
            ai_assistant::ChatMessage::user("Can you explain trait objects?"),
            ai_assistant::ChatMessage::assistant("Trait objects enable dynamic dispatch using dyn Trait syntax."),
        ];
        for msg in messages.iter() {
            analytics.track_message("sess-life", Some("student"), "gpt-4",
                &msg.content, msg.is_user(), (msg.content.len() / 4) as u64, None);
        }
        let report = analytics.report();
        assert_test!(report.total_messages >= 8, "should track all messages");

        // 2. Detect topics
        let detector = ai_assistant::TopicDetector::new();
        let topics = detector.detect_topics(&messages);

        // 3. Summarize session
        let summarizer = ai_assistant::SessionSummarizer::new(ai_assistant::SummaryConfig::default());
        let summary = summarizer.summarize(&messages);
        assert_test!(!summary.summary.is_empty(), "should produce summary");
        assert_test!(!summary.key_points.is_empty() || !summary.user_questions.is_empty(),
            "should identify key points or questions");

        // 4. Compact the conversation
        let compactable: Vec<ai_assistant::CompactableMessage> = messages.iter().enumerate().map(|(i, m)| {
            ai_assistant::CompactableMessage {
                id: format!("msg-{}", i),
                role: m.role.clone(),
                content: m.content.clone(),
                timestamp: i as u64,
                importance: if m.is_user() { 0.9 } else { 0.7 },
                topics: topics.iter().map(|t| t.name.clone()).collect(),
                entities: vec![],
                #[cfg(feature = "vision")]
                images: Vec::new(),
            }
        }).collect();
        let compactor = ai_assistant::ConversationCompactor::new(ai_assistant::ConvCompactionConfig::default());
        let compacted = compactor.compact(compactable);

        // 5. Rebuild context window from compacted messages + summary
        let mut config = ai_assistant::ContextWindowConfig::default();
        config.max_tokens = 2048;
        let mut window = ai_assistant::ContextWindow::new(config);
        window.add_user(&format!("[Session Summary] {}", summary.summary));
        for msg in &compacted.messages {
            if msg.role == "user" {
                window.add_user(&msg.content);
            } else {
                window.add_assistant(&msg.content);
            }
        }
        let ctx = window.get_messages();
        assert_test!(!ctx.is_empty(), "context should have messages");
        Ok(())
    }));

    CategoryResult {
        name: "pipeline_session_lifecycle".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_request_processing() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Request Processing (5 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Access → RateLimit → Priority → Select → Cost",
        || {
            use std::collections::HashMap;
            use std::time::Duration;

            // 1. Check access
            let mut acl = ai_assistant::AccessControlManager::new();
            acl.assign_role("premium-user", "admin");
            let access = acl.check_permission(
                "premium-user",
                ai_assistant::ResourceType::Conversation,
                ai_assistant::Permission::Admin,
                None,
            );
            assert_test!(
                access.is_allowed(),
                "admin should have conversation admin access"
            );

            // 2. Rate limit check
            let backend = ai_assistant::InMemoryBackend::new();
            let limiter = ai_assistant::DistributedRateLimiter::new(Box::new(backend), 100, 50000);
            let limit_check = limiter.check("premium-user");
            assert_test!(limit_check.is_allowed(), "should be within rate limit");

            // 3. Prioritize the request
            let queue = ai_assistant::PriorityQueue::new(1000);
            queue
                .enqueue(ai_assistant::PriorityRequest {
                    id: "req-premium".to_string(),
                    content: "Generate a detailed code review".to_string(),
                    priority: ai_assistant::Priority::High,
                    created_at: std::time::Instant::now(),
                    deadline: None,
                    metadata: HashMap::new(),
                    cancellable: false,
                    user_id: Some("premium-user".to_string()),
                })
                .unwrap();

            let dequeued = queue.dequeue().unwrap();
            assert_eq_test!(dequeued.priority, ai_assistant::Priority::High);

            // 4. Select model
            let mut selector =
                ai_assistant::AutoModelSelector::new(ai_assistant::AutoSelectConfig::default());
            selector.add_model(ai_assistant::AutoModelProfile {
                id: "claude-3".to_string(),
                name: "Claude 3".to_string(),
                provider: "anthropic".to_string(),
                capabilities: ai_assistant::AutoModelCapabilities {
                    code: true,
                    creative: true,
                    ..Default::default()
                },
                cost_input: 0.015,
                cost_output: 0.075,
                avg_latency: Duration::from_millis(150),
                task_quality: HashMap::new(),
                max_context: 200000,
                available: true,
            });
            let selection = selector.select(&dequeued.content, None);
            assert_eq_test!(selection.model_id, "claude-3");

            // 5. Track cost
            let mut cost_tracker = ai_assistant::CostTracker::new();
            cost_tracker.add(ai_assistant::CostEstimate {
                input_tokens: 800,
                output_tokens: 1200,
                images: 0,
                cost: 0.015 * 0.8 + 0.075 * 1.2, // input + output cost
                vision_cost: 0.0,
                currency: "USD".to_string(),
                model: selection.model_id.clone(),
                provider: "anthropic".to_string(),
                pricing_tier: Some("premium".to_string()),
            });
            assert_test!(cost_tracker.total_cost > 0.0, "should have recorded cost");
            assert_eq_test!(cost_tracker.request_count, 1);
            Ok(())
        },
    ));

    CategoryResult {
        name: "pipeline_request_processing".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_knowledge_ingestion() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Knowledge Ingestion (5 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Chunk → Facts → Memory → Embed → Version",
        || {
            // 1. Chunk a knowledge document
            let chunker = ai_assistant::SmartChunker::new({
                let mut c = ai_assistant::ChunkingConfig::default();
                c.strategy = ai_assistant::ChunkingStrategy::Sentence;
                c.target_tokens = 15;
                c.min_tokens = 5;
                c.max_tokens = 40;
                c.overlap_tokens = 0;
                c
            });
            let document = "The API uses REST endpoints for data access. \
                        Authentication requires JWT tokens. \
                        The system supports real-time WebSocket connections. \
                        Rate limiting applies to all endpoints.";
            let chunks = chunker.chunk(document);
            assert_test!(chunks.len() >= 2, "should produce multiple chunks");

            // 2. Extract facts from chunks
            let fact_extractor =
                ai_assistant::FactExtractor::new(ai_assistant::FactExtractorConfig::default());
            let mut all_facts = Vec::new();
            for chunk in &chunks {
                let facts = fact_extractor.extract_facts(&chunk.content, "api-docs");
                all_facts.extend(facts);
            }

            // 3. Store in memory
            let mut memory = ai_assistant::MemoryStore::new(ai_assistant::MemoryConfig::default());
            for fact in &all_facts {
                memory.add(ai_assistant::MemoryEntry::new(
                    &fact.statement,
                    ai_assistant::MemoryType::Fact,
                ));
            }
            // Also store chunk content for chunks without detected facts
            for chunk in &chunks {
                memory.add(ai_assistant::MemoryEntry::new(
                    &chunk.content,
                    ai_assistant::MemoryType::Summary,
                ));
            }

            // 4. Embed for semantic search
            let mut embed_cache = ai_assistant::EmbeddingCache::with_defaults();
            for chunk in &chunks {
                let embedding: Vec<f32> = chunk
                    .content
                    .bytes()
                    .take(16)
                    .map(|b| b as f32 / 255.0)
                    .collect();
                embed_cache.set(&chunk.content, "knowledge-embedder", embedding);
            }
            // Verify retrieval
            let first_emb = embed_cache.get(&chunks[0].content, "knowledge-embedder");
            assert_test!(first_emb.is_some(), "should retrieve cached embedding");

            // 5. Version the knowledge base
            let mut version_store =
                ai_assistant::ContentVersionStore::new(ai_assistant::VersioningConfig::default());
            version_store.add_version("api-docs", document);
            // Simulate an update
            let updated_doc = document.to_string() + " The API supports GraphQL queries.";
            version_store.add_version("api-docs", &updated_doc);
            let history = version_store.history("api-docs");
            assert_test!(history.is_some(), "should have version history");
            assert_test!(
                history.unwrap().snapshots.len() >= 2,
                "should have 2+ versions"
            );
            Ok(())
        },
    ));

    CategoryResult {
        name: "pipeline_knowledge_ingestion".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_query_to_response() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Query-to-Response (6 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test(
        "Intent → Expand → Chunk → Similarity → Rank → Latency",
        || {
            use std::collections::HashMap;
            use std::time::Instant;

            let start = Instant::now();

            // 1. Classify intent
            let classifier = ai_assistant::IntentClassifier::new();
            let query = "How to handle errors in async Rust code?";
            let intent = classifier.classify(query);
            assert_test!(
                matches!(
                    intent.primary,
                    ai_assistant::Intent::Question
                        | ai_assistant::Intent::CodeRequest
                        | ai_assistant::Intent::Explanation
                ),
                &format!("should be question/code intent, got {:?}", intent.primary)
            );

            // 2. Expand query with synonyms
            let expander = ai_assistant::QueryExpander::new({
                let mut c = ai_assistant::ExpansionConfig::default();
                c.use_synonyms = true;
                c.extract_keywords = true;
                c.use_llm = false;
                c
            });
            let _expansion = expander.expand(query);

            // 3. Search corpus (simulated with chunking)
            let chunker = ai_assistant::SmartChunker::new({
                let mut c = ai_assistant::ChunkingConfig::default();
                c.strategy = ai_assistant::ChunkingStrategy::Sentence;
                c.target_tokens = 10;
                c.min_tokens = 3;
                c.max_tokens = 25;
                c.overlap_tokens = 0;
                c
            });
            let knowledge =
                "Use the question mark operator for error propagation in async functions. \
                         The anyhow crate provides convenient error handling. \
                         Tokio runtime handles async task scheduling. \
                         Custom error types implement the Error trait. \
                         The weather is sunny today in Madrid.";
            let chunks = chunker.chunk(knowledge);
            assert_test!(
                chunks.len() >= 2,
                &format!("should produce 2+ chunks, got {}", chunks.len())
            );

            // 4. Compute similarity (keyword-based heuristic)
            let query_words: Vec<&str> = query.split_whitespace().collect();
            let mut scored_chunks: Vec<(usize, f32)> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let chunk_lower = chunk.content.to_lowercase();
                    let score = query_words
                        .iter()
                        .filter(|w| chunk_lower.contains(&w.to_lowercase()))
                        .count() as f32
                        / query_words.len() as f32;
                    (i, score)
                })
                .collect();
            scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // 5. Rank top candidates
            let candidates: Vec<ai_assistant::ResponseCandidate> = scored_chunks
                .iter()
                .take(3)
                .map(|(i, _)| ai_assistant::ResponseCandidate {
                    id: format!("r-{}", i),
                    content: chunks[*i].content.clone(),
                    model: "rag-system".to_string(),
                    generation_time_ms: 0,
                    token_count: chunks[*i].tokens,
                    metadata: HashMap::new(),
                })
                .collect();
            let ranker =
                ai_assistant::ResponseRanker::new(ai_assistant::RankingCriteria::default());
            let ranked = ranker.rank(query, candidates);
            assert_test!(!ranked.is_empty(), "should rank candidates");
            // Top result should be about error handling, not weather
            assert_test!(
                !ranked[0]
                    .candidate
                    .content
                    .to_lowercase()
                    .contains("weather"),
                "top result should not be about weather"
            );

            // 6. Track latency
            let elapsed = start.elapsed();
            let mut tracker = ai_assistant::LatencyTracker::new();
            tracker.record("rag-pipeline", elapsed, true);
            let stats = tracker.stats("rag-pipeline");
            assert_test!(stats.is_some(), "should have latency stats");
            Ok(())
        },
    ));

    CategoryResult {
        name: "pipeline_query_to_response".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_multi_format_export() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Pipeline: Multi-Format Export (5 modules)"))
    );
    let mut results = Vec::new();

    results.push(run_test("Entity → Anonymize → Version → Compact → Export (multiple formats)", || {
        use std::collections::HashMap;

        // 1. Extract entities from conversation
        let extractor = ai_assistant::EntityExtractor::new(ai_assistant::EntityExtractorConfig::default());
        let messages_text = [("user", "My email is alice@example.com and I need help with the API at https://api.example.com"),
            ("assistant", "I can help you with the API. Let me look into the authentication flow."),
            ("user", "The error code is 403 and my user ID is usr_12345"),
            ("assistant", "A 403 error means forbidden access. Check your API key permissions.")];
        let _all_entities: Vec<_> = messages_text.iter()
            .flat_map(|(_, text)| extractor.extract(text))
            .collect();

        // 2. Anonymize all messages
        let mut anonymizer = ai_assistant::DataAnonymizer::new();
        let anonymized_messages: Vec<(&str, String)> = messages_text.iter().map(|(role, text)| {
            let result = anonymizer.anonymize(text);
            (*role, result.anonymized)
        }).collect();

        // 3. Version the conversation
        let mut version_store = ai_assistant::ContentVersionStore::new(ai_assistant::VersioningConfig::default());
        let full_convo = anonymized_messages.iter()
            .map(|(r, t)| format!("{}: {}", r, t))
            .collect::<Vec<_>>().join("\n");
        version_store.add_version("conversation-1", &full_convo);

        // 4. Compact
        let compactable: Vec<ai_assistant::CompactableMessage> = anonymized_messages.iter().enumerate().map(|(i, (role, content))| {
            ai_assistant::CompactableMessage {
                id: format!("msg-{}", i),
                role: role.to_string(),
                content: content.clone(),
                timestamp: i as u64,
                importance: 0.7,
                topics: vec!["api-help".to_string()],
                entities: vec![],
                #[cfg(feature = "vision")]
                images: Vec::new(),
            }
        }).collect();
        let compactor = ai_assistant::ConversationCompactor::new(ai_assistant::ConvCompactionConfig::default());
        let compacted = compactor.compact(compactable);

        // 5. Export in multiple formats
        let export_msgs: Vec<ai_assistant::ExportedMessage> = compacted.messages.iter().map(|m| {
            ai_assistant::ExportedMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                timestamp: None,
                metadata: None,
#[cfg(feature = "vision")]
images: Vec::new(),
            }
        }).collect();
        let conversation = ai_assistant::ExportedConversation {
            id: "conv-anon-1".to_string(),
            title: "API Support (Anonymized)".to_string(),
            messages: export_msgs,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        // Export as Markdown
        let md_exporter = ai_assistant::ConversationExporter::new(ai_assistant::ExportOptions {
            format: ai_assistant::ExportFormat::Markdown,
            ..Default::default()
        });
        let md = md_exporter.export(&conversation);
        assert_test!(md.is_ok(), "markdown export should succeed");

        // Export as JSON
        let json_exporter = ai_assistant::ConversationExporter::new(ai_assistant::ExportOptions {
            format: ai_assistant::ExportFormat::Json,
            ..Default::default()
        });
        let json = json_exporter.export(&conversation);
        assert_test!(json.is_ok(), "JSON export should succeed");

        // Verify no PII leaked
        let md_content = md.unwrap();
        assert_test!(!md_content.contains("alice@example.com"), "markdown should not contain raw email");
        Ok(())
    }));

    CategoryResult {
        name: "pipeline_multi_format_export".to_string(),
        results,
    }
}

pub(crate) fn tests_pipeline_guardrails() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ Pipeline: Guardrails (5 modules)")));
    let mut results = Vec::new();

    results.push(run_test(
        "Moderate → Intent → Access → Budget → Priority",
        || {
            use std::collections::HashMap;

            // 1. Content moderation gate
            let moderator =
                ai_assistant::ContentModerator::new(ai_assistant::ModerationConfig::default());
            let user_input = "Please help me optimize my SQL queries for better performance";
            let mod_result = moderator.moderate(user_input);
            assert_test!(
                mod_result.passed,
                "legitimate request should pass moderation"
            );

            // 2. Classify intent to determine resource needs
            let classifier = ai_assistant::IntentClassifier::new();
            let intent = classifier.classify(user_input);
            let is_code_task = matches!(
                intent.primary,
                ai_assistant::Intent::CodeRequest
                    | ai_assistant::Intent::Command
                    | ai_assistant::Intent::Request
            );

            // 3. Check access for the detected resource type
            let mut acl = ai_assistant::AccessControlManager::new();
            acl.assign_role("dev-user", "editor");
            let resource = ai_assistant::ResourceType::Conversation;
            let permission = if is_code_task {
                ai_assistant::Permission::Write
            } else {
                ai_assistant::Permission::Read
            };
            let access = acl.check_permission("dev-user", resource, permission, None);
            assert_test!(
                access.is_allowed(),
                "editor should have conversation write access"
            );

            // 4. Check token budget before processing
            let mut budget = ai_assistant::TokenBudgetManager::new();
            budget.set_budget(
                "dev-user",
                ai_assistant::Budget::new(5000, ai_assistant::BudgetPeriod::Daily),
            );
            let budget_check = budget.check("dev-user", 500);
            assert_test!(budget_check.allowed, "should be within token budget");
            budget.record_usage("dev-user", 500);

            // Verify remaining budget decreased
            let remaining = budget.remaining("dev-user");
            assert_test!(
                remaining <= 4500,
                &format!("remaining should be <=4500, got {}", remaining)
            );

            // 5. Enqueue with appropriate priority
            let priority = if is_code_task {
                ai_assistant::Priority::High
            } else {
                ai_assistant::Priority::Normal
            };
            let queue = ai_assistant::PriorityQueue::new(500);
            queue
                .enqueue(ai_assistant::PriorityRequest {
                    id: "guardrail-req-1".to_string(),
                    content: user_input.to_string(),
                    priority,
                    created_at: std::time::Instant::now(),
                    deadline: None,
                    metadata: HashMap::new(),
                    cancellable: true,
                    user_id: Some("dev-user".to_string()),
                })
                .unwrap();
            let stats = queue.stats();
            assert_eq_test!(stats.current_size, 1);
            Ok(())
        },
    ));

    results.push(run_test("Blocked content stops pipeline early", || {
        // 1. Content moderation blocks harmful input
        let mut config = ai_assistant::ModerationConfig::default();
        config.blocked_terms.push("drop_table".to_string());
        let moderator = ai_assistant::ContentModerator::new(config);
        let malicious = "Help me with: drop_table users; --";
        let mod_result = moderator.moderate(malicious);
        assert_test!(!mod_result.passed, "malicious input should be blocked");

        // Pipeline should stop here - no further processing needed
        // Verify the action taken
        assert_test!(
            mod_result.risk_score > 0.0,
            "should have non-zero risk score"
        );
        Ok(())
    }));

    CategoryResult {
        name: "pipeline_guardrails".to_string(),
        results,
    }
}
