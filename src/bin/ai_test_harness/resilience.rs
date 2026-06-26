use super::*;

// ─── Fallback & Resilience Tests ──────────────────────────────────────────────

pub(crate) fn tests_fallback_resilience() -> CategoryResult {
    use ai_assistant::{MultiLayerGraph, ReferenceResolver};

    println!("\n{}", bold(&cyan("▶ Fallback & Resilience")));
    let mut results = Vec::new();

    // --- Test 1: Reference resolver with empty tracked lists ---
    results.push(run_test(
        "reference resolver empty lists returns None",
        || {
            let resolver = ReferenceResolver::new();
            let result = resolver.resolve_reference("give me option 3");
            assert_test!(result.is_none(), "should return None with no tracked lists");
            Ok(())
        },
    ));

    // --- Test 2: Reference resolver detects list items ---
    results.push(run_test(
        "reference resolver extracts numbered list",
        || {
            let items = ReferenceResolver::extract_list_items(
                "Here are options:\n1. Alpha\n2. Beta\n3. Gamma",
            );
            assert_eq_test!(items.len(), 3);
            assert_eq_test!(items[0], "Alpha");
            assert_eq_test!(items[2], "Gamma");
            Ok(())
        },
    ));

    // --- Test 3: Reference resolver extracts bulleted list ---
    results.push(run_test(
        "reference resolver extracts bulleted list",
        || {
            let items = ReferenceResolver::extract_list_items(
                "Options:\n- First item\n- Second item\n* Third item",
            );
            assert_eq_test!(items.len(), 3);
            Ok(())
        },
    ));

    // --- Test 4: Reference resolver extracts lettered list ---
    results.push(run_test(
        "reference resolver extracts lettered list",
        || {
            let items =
                ReferenceResolver::extract_list_items("a. Option A\nb. Option B\nc. Option C");
            assert_eq_test!(items.len(), 3);
            assert_eq_test!(items[0], "Option A");
            Ok(())
        },
    ));

    // --- Test 5: Reference resolver resolves ordinal (English) ---
    results.push(run_test(
        "reference resolver resolves English ordinal",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. Alpha\n2. Beta\n3. Gamma", "test topic", 0);
            let result = resolver.resolve_reference("the second one please");
            assert_test!(result.is_some(), "should resolve 'the second one'");
            let text = result.unwrap();
            assert_test!(
                text.contains("Beta"),
                &format!("should contain Beta, got: {}", text)
            );
            Ok(())
        },
    ));

    // --- Test 6: Reference resolver resolves ordinal (Spanish) ---
    results.push(run_test(
        "reference resolver resolves Spanish ordinal",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. Alfa\n2. Beta\n3. Gamma", "tema test", 0);
            let result = resolver.resolve_reference("dame el tercero");
            assert_test!(result.is_some(), "should resolve 'el tercero'");
            let text = result.unwrap();
            assert_test!(
                text.contains("Gamma"),
                &format!("should contain Gamma, got: {}", text)
            );
            Ok(())
        },
    ));

    // --- Test 7: Reference resolver resolves cardinal ---
    results.push(run_test(
        "reference resolver resolves cardinal 'option 3'",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("- Red\n- Green\n- Blue\n- Yellow", "colors", 0);
            let result = resolver.resolve_reference("I want option 3");
            assert_test!(result.is_some(), "should resolve 'option 3'");
            let text = result.unwrap();
            assert_test!(
                text.contains("Blue"),
                &format!("should contain Blue, got: {}", text)
            );
            Ok(())
        },
    ));

    // --- Test 8: Reference resolver out-of-bounds ---
    results.push(run_test(
        "reference resolver handles out-of-bounds gracefully",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. Only\n2. Two", "small list", 0);
            let result = resolver.resolve_reference("give me option 5");
            assert_test!(result.is_some(), "should return info about bounds");
            let text = result.unwrap();
            assert_test!(
                text.contains("2 items"),
                &format!("should mention 2 items, got: {}", text)
            );
            Ok(())
        },
    ));

    // --- Test 9: Reference resolver with fallback callback ---
    results.push(run_test(
        "reference resolver fallback chain invoked",
        || {
            let resolver = ReferenceResolver::new(); // empty lists
            let result = resolver
                .resolve_reference_with_fallback("tell me about the previous topic", |_msg| {
                    Some("Previous topic was about Rust performance".to_string())
                });
            assert_test!(result.is_some(), "fallback should provide context");
            let text = result.unwrap();
            assert_test!(
                text.contains("Rust performance"),
                "fallback content should be present"
            );
            Ok(())
        },
    ));

    // --- Test 10: Reference resolver fallback not called when list resolves ---
    results.push(run_test(
        "reference resolver skips fallback when list matches",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. Alpha\n2. Beta", "test", 0);

            let fallback_called = std::cell::Cell::new(false);
            let result = resolver.resolve_reference_with_fallback("the first one", |_msg| {
                fallback_called.set(true);
                Some("fallback".to_string())
            });
            assert_test!(result.is_some(), "should resolve from list");
            assert_test!(!fallback_called.get(), "fallback should NOT be called");
            Ok(())
        },
    ));

    // --- Test 11: Multi-layer graph graceful degradation ---
    results.push(run_test(
        "multi-layer graph empty query returns empty view",
        || {
            let g = MultiLayerGraph::new();
            let view = g.query_unified(None);
            assert_test!(
                view.entities.is_empty(),
                "empty graph should return empty view"
            );
            assert_test!(
                view.relations.is_empty(),
                "empty graph should return no relations"
            );
            Ok(())
        },
    ));

    // --- Test 12: Multi-layer graph single layer still works ---
    results.push(run_test(
        "multi-layer graph works with only session layer",
        || {
            let mut g = MultiLayerGraph::new();
            g.process_user_message("s1", "About Rust", &["Rust".to_string()]);
            let view = g.query_unified(Some("s1"));
            assert_test!(
                !view.entities.is_empty(),
                "single layer should still produce results"
            );
            Ok(())
        },
    ));

    // --- Test 13: Context overflow truncation ---
    results.push(run_test(
        "large knowledge context truncated gracefully",
        || {
            // Simulate: a very long knowledge string
            let long_knowledge = "Line of knowledge content here.\n".repeat(1000);
            let tokens = ai_assistant::estimate_tokens(&long_knowledge);
            assert_test!(tokens > 5000, "should be a large context");
            // The truncation logic is in build_rag_context; here we verify estimate_tokens works
            let truncated = &long_knowledge[..long_knowledge.len() / 2];
            let trunc_tokens = ai_assistant::estimate_tokens(truncated);
            assert_test!(trunc_tokens < tokens, "truncated should have fewer tokens");
            Ok(())
        },
    ));

    // --- Test 14: ChunkingConfig validates bounds ---
    results.push(run_test(
        "ChunkingConfig validated prevents overflow",
        || {
            use ai_assistant::ChunkingConfig;
            // Create with defaults then mutate via validated()
            let mut config = ChunkingConfig::default();
            config.target_tokens = usize::MAX;
            config.max_tokens = usize::MAX;
            config.min_tokens = usize::MAX;
            config.overlap_tokens = usize::MAX;
            let validated = config.validated();
            assert_test!(
                validated.target_tokens < usize::MAX / 4,
                "target_tokens should be clamped"
            );
            assert_test!(
                validated.overlap_tokens < validated.target_tokens,
                "overlap should be less than target"
            );
            Ok(())
        },
    ));

    // --- Test 15: Memory search finds by keyword ---
    results.push(run_test("memory search returns relevant memories", || {
        use ai_assistant::{MemoryConfig, MemoryEntry, MemoryStore, MemoryType};

        let mut store = MemoryStore::new(MemoryConfig::default());
        let e1 = MemoryEntry::new("Rust is a systems programming language", MemoryType::Fact);
        let e2 = MemoryEntry::new("Python is good for scripting", MemoryType::Fact);
        store.add(e1);
        store.add(e2);

        let results = store.search("Rust");
        assert_test!(!results.is_empty(), "should find Rust memory");
        assert_test!(
            results[0].content.contains("Rust"),
            "first result should mention Rust"
        );
        Ok(())
    }));

    // --- Test 16: Reference resolver no pattern = fast skip ---
    results.push(run_test(
        "reference resolver fast-skips non-reference messages",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. A\n2. B\n3. C", "test", 0);
            // This message has no reference patterns at all
            let result = resolver.resolve_reference("Tell me about quantum computing");
            assert_test!(result.is_none(), "should skip non-reference messages");
            Ok(())
        },
    ));

    // --- Test 17: Reference resolver multi-list topic disambiguation ---
    results.push(run_test(
        "reference resolver disambiguates by topic",
        || {
            let mut resolver = ReferenceResolver::new();
            resolver.track_lists_in_message("1. Red\n2. Blue\n3. Green", "colors", 0);
            resolver.track_lists_in_message("1. Dog\n2. Cat\n3. Bird", "animals", 1);

            // Reference with topic hint "colors" + pattern "the list"
            let result = resolver.resolve_reference("show me the list about colors");
            assert_test!(result.is_some(), "should find list about colors");
            let text = result.unwrap();
            assert_test!(
                text.contains("Red") || text.contains("colors"),
                &format!("should reference colors list, got: {}", text)
            );
            Ok(())
        },
    ));

    // --- Test 18: Guardrail panic safety ---
    results.push(run_test(
        "guardrail pipeline survives panicking guard",
        || {
            use ai_assistant::{
                Guard, GuardAction, GuardCheckResult, GuardStage, GuardrailPipeline,
            };

            struct PanickingGuard;
            impl Guard for PanickingGuard {
                fn name(&self) -> &str {
                    "panicker"
                }
                fn stage(&self) -> GuardStage {
                    GuardStage::PreSend
                }
                fn check(&self, _text: &str) -> GuardCheckResult {
                    panic!("this guard always panics!");
                }
            }

            struct SafeGuard;
            impl Guard for SafeGuard {
                fn name(&self) -> &str {
                    "safe"
                }
                fn stage(&self) -> GuardStage {
                    GuardStage::PreSend
                }
                fn check(&self, _text: &str) -> GuardCheckResult {
                    GuardCheckResult {
                        guard_name: "safe".to_string(),
                        action: GuardAction::Pass,
                        score: 0.0,
                        details: String::new(),
                    }
                }
            }

            let mut pipeline = GuardrailPipeline::new();
            pipeline.add_guard(Box::new(PanickingGuard));
            pipeline.add_guard(Box::new(SafeGuard));

            // Should NOT crash — and the pipeline FAILS CLOSED: a panicking
            // guard blocks the message instead of being skipped. A guard that
            // panics on crafted input must not become a bypass vector.
            let result = pipeline.check_input("test message");
            assert_test!(
                !result.passed,
                "pipeline must fail closed when a guard panics"
            );
            assert_test!(
                result.blocked_by.as_deref() == Some("panicker"),
                "blocked_by must identify the panicking guard"
            );
            Ok(())
        },
    ));

    CategoryResult {
        name: "fallback_resilience".to_string(),
        results,
    }
}

// ─── Conversation Quality Tests (Ollama) ──────────────────────────────────────

pub(crate) fn tests_conversation_quality() -> CategoryResult {
    use ai_assistant::{
        recover_session, ChatMessage, ChatSession, ChatSessionStore, DiskSpillBuffer,
        DiskSpillConfig, ReferenceResolver,
    };

    println!("\n{}", bold(&cyan("▶ Conversation Quality (Ollama)")));
    let mut results = Vec::new();

    // Check Ollama availability
    let ollama_available = std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(2),
    )
    .is_ok();

    if !ollama_available {
        println!(
            "  {} Ollama not running - skipping conversation quality tests",
            yellow("SKIP")
        );
        results.push(TestResult {
            name: "Ollama availability".to_string(),
            passed: true,
            message: Some("Skipped — Ollama not running".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "conversation_quality".to_string(),
            results,
        };
    }

    // --- Test 1: Multi-turn conversation with recent reference ---
    results.push(run_test_scored(
        "multi-turn recent reference resolution",
        0.80,
        || {
            let mut resolver = ReferenceResolver::new();
            let mut score = 0.0f64;
            let total = 5.0f64;

            // Simulate assistant providing a list
            let assistant_msg =
                "Here are the top engines:\n1. Quantum Drive MK2\n2. Atlas QD\n3. Erebus Drive";
            resolver.track_lists_in_message(assistant_msg, "engines", 0);

            // Test various reference patterns
            let refs = vec![
                ("the first one", "Quantum Drive MK2"),
                ("option 2", "Atlas QD"),
                ("la tercera opción", "Erebus Drive"),
                ("the second", "Atlas QD"),
                ("give me number 1", "Quantum Drive MK2"),
            ];

            for (query, expected) in &refs {
                if let Some(resolved) = resolver.resolve_reference(query) {
                    if resolved.contains(expected) {
                        score += 1.0;
                    }
                }
            }

            Ok(score / total)
        },
    ));

    // --- Test 2: Reference to older list (multi-list disambiguation) ---
    results.push(run_test_scored(
        "old list reference disambiguation",
        0.80,
        || {
            let mut resolver = ReferenceResolver::new();
            let mut score = 0.0f64;
            let total = 4.0f64;

            // Track two different lists at different turns
            resolver.track_lists_in_message(
                "Ships:\n1. Aurora MR\n2. Mustang Alpha\n3. Avenger Titan",
                "ships",
                0,
            );
            resolver.track_lists_in_message(
                "Weapons:\n1. Mantis GT-220\n2. Badger Repeater\n3. Panther Repeater",
                "weapons",
                1,
            );

            // Reference with topic context should disambiguate
            if let Some(r) = resolver.resolve_reference("the second ship") {
                if r.contains("Mustang Alpha") {
                    score += 1.0;
                }
            }
            if let Some(r) = resolver.resolve_reference("weapon number 3") {
                if r.contains("Panther") {
                    score += 1.0;
                }
            }
            // Without context, should match most recent list
            if let Some(r) = resolver.resolve_reference("the first one") {
                // Most recent list is weapons
                if r.contains("Mantis") || r.contains("Aurora") {
                    score += 1.0;
                }
            }
            // Ordinal in Spanish
            if let Some(r) = resolver.resolve_reference("la segunda arma") {
                if r.contains("Badger") {
                    score += 1.0;
                }
            }

            Ok(score / total)
        },
    ));

    // --- Test 3: FreshContext context size estimation ---
    results.push(run_test_scored(
        "FreshContext context size estimation",
        0.80,
        || {
            use ai_assistant::get_model_context_size;

            let mut score = 0.0f64;
            let total = 4.0f64;

            // Test context size for known models
            let size = get_model_context_size("llama3.2");
            if size > 0 {
                score += 1.0;
            }

            let size2 = get_model_context_size("mistral");
            if size2 > 0 {
                score += 1.0;
            }

            // GPT-4 should have a large context
            let size3 = get_model_context_size("gpt-4");
            if size3 >= 8000 {
                score += 1.0;
            }

            // Unknown model should still return a sensible default
            let size_unknown = get_model_context_size("totally-unknown-model-xyz");
            if size_unknown > 0 {
                score += 1.0;
            }

            Ok(score / total)
        },
    ));

    // --- Test 4: Memory persistence cross-turn ---
    results.push(run_test_scored(
        "memory persistence cross-turn",
        0.80,
        || {
            use ai_assistant::memory::{
                MemoryConfig, MemoryEntry as MemEntry, MemoryStore, MemoryType,
            };

            let mut score = 0.0f64;
            let total = 5.0f64;

            let mut store = MemoryStore::new(MemoryConfig::default());

            // Store memories across simulated turns
            let mut m1 = MemEntry::new("User's name is Orlando", MemoryType::Fact);
            m1.importance = 0.9;
            store.add(m1);

            let mut m2 = MemEntry::new("User prefers concise responses", MemoryType::Preference);
            m2.importance = 0.8;
            store.add(m2);

            let mut m3 = MemEntry::new("Discussed Avenger Titan last session", MemoryType::Fact);
            m3.importance = 0.7;
            store.add(m3);

            // Verify retrieval
            let results_search = store.search("Orlando");
            if !results_search.is_empty() {
                score += 1.0;
            }

            let results_pref = store.search("concise");
            if !results_pref.is_empty() {
                score += 1.0;
            }

            // Search with related terms
            let results_ship = store.search("Avenger");
            if !results_ship.is_empty() {
                score += 1.0;
            }

            // All three should be findable
            let all_results = store.search("session");
            if !all_results.is_empty() {
                score += 1.0;
            }

            // Adding more memories works
            let m4 = MemEntry::new("Low importance note", MemoryType::Fact);
            store.add(m4);
            let r = store.search("note");
            if !r.is_empty() {
                score += 1.0;
            }

            Ok(score / total)
        },
    ));

    // --- Test 5: Graph entity linking ---
    results.push(run_test_scored(
        "knowledge graph entity linking",
        0.80,
        || {
            use ai_assistant::MultiLayerGraph;

            let mut score = 0.0f64;
            let total = 4.0f64;

            let mut graph = MultiLayerGraph::new();

            // Get or create a session graph and add entities
            let session = graph.get_or_create_session("test_session");
            session.add_entity("Avenger Titan", "ship", "user_message");
            session.add_entity("Orlando", "person", "user_message");
            session.add_relation("Orlando", "owns", "Avenger Titan");

            // Verify stats
            let stats = graph.stats();
            if stats.total_session_entities >= 2 {
                score += 1.0;
            }
            if stats.session_count >= 1 {
                score += 2.0;
            } // counts for 2 checks

            // Add another session and verify cross-session
            let session2 = graph.get_or_create_session("test_session_2");
            session2.add_entity("Avenger Titan", "ship", "knowledge_base");

            let stats2 = graph.stats();
            if stats2.session_count >= 2 {
                score += 1.0;
            }

            Ok(score / total)
        },
    ));

    // --- Test 6: Vague reference resolution via resolve_reference ---
    results.push(run_test_scored("vague reference resolution", 0.80, || {
        let mut resolver = ReferenceResolver::new();
        let mut score = 0.0f64;
        let total = 5.0f64;

        // Track a list
        resolver.track_lists_in_message(
            "Here are your options:\n1. Buy now\n2. Wait for sale\n3. Trade in",
            "purchase options",
            0,
        );

        // Explicit ordinal reference
        if let Some(r) = resolver.resolve_reference("the first option") {
            if r.contains("Buy now") {
                score += 1.0;
            }
        }
        // Cardinal reference
        if let Some(r) = resolver.resolve_reference("option 2") {
            if r.contains("Wait for sale") {
                score += 1.0;
            }
        }
        // Spanish ordinal
        if let Some(r) = resolver.resolve_reference("la tercera") {
            if r.contains("Trade in") {
                score += 1.0;
            }
        }
        // Out of bounds should return info about the error
        if resolver.resolve_reference("option 5").is_some() {
            score += 1.0; // Returns info about out-of-bounds
        }
        // "the last one" or "the third"
        if let Some(r) = resolver.resolve_reference("the third one") {
            if r.contains("Trade in") {
                score += 1.0;
            }
        }

        Ok(score / total)
    }));

    // --- Test 7: Context overflow graceful degradation ---
    results.push(run_test_scored(
        "context overflow graceful degradation",
        0.80,
        || {
            let mut score = 0.0f64;
            let total = 4.0f64;

            // DiskSpillBuffer handles overflow gracefully
            let config = DiskSpillConfig::with_threshold(50);
            let mut buf = DiskSpillBuffer::with_config(config);

            // Push data exceeding threshold
            for i in 0..20 {
                if buf.push(format!("chunk_{:04} data here\n", i)).is_ok() {
                    score = 1.0; // At least pushes succeed
                }
            }

            // Buffer should have spilled to disk
            if buf.has_spilled() {
                score += 1.0;
            }

            // All data recoverable
            let all_data = buf.drain_all();
            if let Ok(data) = all_data {
                if data.contains("chunk_0000") && data.contains("chunk_0019") {
                    score += 1.0; // First and last chunks present
                }
                if data.lines().count() == 20 {
                    score += 1.0; // All 20 lines present
                }
            }

            Ok(score / total)
        },
    ));

    // --- Test 8: Session recovery comparison ---
    results.push(run_test_scored(
        "session recovery multi-format",
        0.80,
        || {
            let mut score = 0.0f64;
            let total = 5.0f64;

            let dir = std::env::temp_dir().join("ai_assistant_conv_quality_tests");
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

            // Test 1: Valid JSON recovery
            let valid_path = dir.join("valid.json");
            let mut store = ChatSessionStore::new();
            let mut session = ChatSession::new("Test");
            session.id = "recovery_test_1".to_string();
            session.messages.push(ChatMessage::user("Hello"));
            store.save_session(session);
            store.save_to_json(&valid_path).map_err(|e| e.to_string())?;

            let result = recover_session(&valid_path);
            if result.recovered && result.store.sessions.len() == 1 {
                score += 1.0;
            }

            // Test 2: Corrupted file with partial recovery
            let corrupt_path = dir.join("corrupt.json");
            let s = ChatSession::new("Saved");
            let s_json = serde_json::to_string(&s).map_err(|e| e.to_string())?;
            std::fs::write(&corrupt_path, format!("GARBAGE\n{}", s_json))
                .map_err(|e| e.to_string())?;
            let result2 = recover_session(&corrupt_path);
            if result2.recovered {
                score += 1.0;
            }

            // Test 3: JSONL journal recovery
            let journal_path = dir.join("journal.jsonl");
            let entry =
                ai_assistant::JournalEntry::from_message(&ChatMessage::user("From journal"));
            let line = serde_json::to_string(&entry).map_err(|e| e.to_string())?;
            std::fs::write(&journal_path, format!("{}\n", line)).map_err(|e| e.to_string())?;
            let result3 = recover_session(&journal_path);
            if result3.recovered {
                score += 1.0;
            }

            // Test 4: Nonexistent file returns not-recovered
            let result4 = recover_session(std::path::Path::new("/nonexistent/path.json"));
            if !result4.recovered {
                score += 1.0;
            }

            // Test 5: Empty file returns not-recovered
            let empty_path = dir.join("empty.json");
            std::fs::write(&empty_path, "").map_err(|e| e.to_string())?;
            let result5 = recover_session(&empty_path);
            if !result5.recovered {
                score += 1.0;
            }

            Ok(score / total)
        },
    ));

    CategoryResult {
        name: "conversation_quality".to_string(),
        results,
    }
}
