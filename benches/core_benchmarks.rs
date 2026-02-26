use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

use ai_assistant::{
    ChatMessage, CompactableMessage, CompressionAlgorithm, ContentLengthGuard, ContextMessage,
    ContextWindow, ContextWindowConfig, ConversationCompactor, ConversationTemplate,
    GuardrailPipeline, HtmlExtractionConfig, HtmlExtractor, IntentClassifier, PatternGuard,
    PromptShortener, RequestSigner, SentimentAnalyzer, ServerRateLimiter, SignatureAlgorithm,
    TemplateCategory, TextTransformer, Transform, WsFrame,
    // v14 benchmark imports
    ChunkingConfig, SmartChunker,
    RagDb, KnowledgeChunk, KnowledgeUsage,
    EmbeddingConfig, LocalEmbedder,
    HnswConfig, HnswIndex,
    KnowledgeGraphStore, KnowledgeGraphConfig, KGEntityType,
    parse_tool_calls,
    PiiDetector, PiiConfig,
    JsonSchema, SchemaProperty,
};

fn bench_intent_classification(c: &mut Criterion) {
    let classifier = IntentClassifier::new();
    let sentences = [
        "What is the capital of France?",
        "Write a function to sort an array",
        "Hello, how are you?",
        "This doesn't work, it's broken",
        "Thank you so much for the help!",
        "Compare Python vs Rust for systems programming",
        "Please explain how async works",
        "goodbye, see you later",
    ];

    c.bench_function("intent_classification", |b| {
        b.iter(|| {
            for sentence in &sentences {
                let _ = classifier.classify(sentence);
            }
        });
    });
}

fn bench_conversation_compaction(c: &mut Criterion) {
    let config = ai_assistant::ConvCompactionConfig {
        max_messages: 50,
        target_messages: 20,
        preserve_recent: 10,
        preserve_first: 2,
        min_importance: 0.8,
    };
    let compactor = ConversationCompactor::new(config);

    c.bench_function("conversation_compaction_100_msgs", |b| {
        b.iter(|| {
            let messages: Vec<CompactableMessage> = (0..100)
                .map(|i| {
                    let role = if i % 2 == 0 { "user" } else { "assistant" };
                    CompactableMessage::new(role, &format!("Message number {} with some content to make it realistic.", i))
                        .with_importance(0.3 + (i as f64 % 10.0) * 0.07)
                })
                .collect();
            let _ = compactor.compact(messages);
        });
    });
}

fn bench_prompt_shortener(c: &mut Criterion) {
    let shortener = PromptShortener::new();
    let long_prompt = "Please kindly explain in order to understand the concept, \
        I would really very much like to basically know due to the fact that \
        I am quite curious. In the event that you could simply provide \
        a detailed explanation at this point in time, that would actually \
        be really helpful. Please just tell me about it.";

    c.bench_function("prompt_shortener", |b| {
        b.iter(|| {
            let _ = shortener.shorten(long_prompt);
        });
    });
}

fn bench_sentiment_analysis(c: &mut Criterion) {
    let analyzer = SentimentAnalyzer::new();
    let sentences = [
        "This is great! Thank you so much for the help!",
        "This is terrible and broken. Nothing works at all.",
        "What time is it?",
        "I am extremely frustrated with this annoying bug.",
        "The performance is absolutely fantastic and impressive!",
        "It's not bad, but it could be better.",
        "I really love how fast and efficient this is.",
        "The worst experience I have ever had, completely useless.",
    ];

    c.bench_function("sentiment_analysis", |b| {
        b.iter(|| {
            for sentence in &sentences {
                let _ = analyzer.analyze_message(sentence);
            }
        });
    });
}

fn bench_sha256_signing(c: &mut Criterion) {
    let signer = RequestSigner::new(b"benchmark-secret-key-256", SignatureAlgorithm::HmacSha256);
    let payload = "Hello, this is a benchmark payload for HMAC-SHA256 signing.";

    c.bench_function("request_signing_hmac_sha256", |b| {
        b.iter(|| {
            let _ = signer.sign(payload);
        });
    });
}

fn bench_template_rendering(c: &mut Criterion) {
    let template = ConversationTemplate::new("bench", "Benchmark Template", TemplateCategory::Learning)
        .with_system_prompt("You are a tutor teaching {topic} at {level} level using {language}.")
        .with_starter("Explain {topic} for a {level} student in {language}.")
        .with_starter("Give a {language} example of {topic}.");

    let mut vars = HashMap::new();
    vars.insert("topic".to_string(), "algorithms".to_string());
    vars.insert("level".to_string(), "intermediate".to_string());
    vars.insert("language".to_string(), "Rust".to_string());

    c.bench_function("template_rendering", |b| {
        b.iter(|| {
            let _ = template.apply(&vars);
        });
    });
}

// ---------------------------------------------------------------------------
// New benchmarks (7-16)
// ---------------------------------------------------------------------------

fn bench_cosine_similarity(c: &mut Criterion) {
    // Two 384-dimension vectors (typical embedding size)
    let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.02).cos()).collect();

    c.bench_function("cosine_similarity_384d", |b_iter| {
        b_iter.iter(|| {
            let _ = ai_assistant::cosine_similarity(&a, &b);
        });
    });
}

fn bench_guardrail_check(c: &mut Criterion) {
    let mut pipeline = GuardrailPipeline::new().with_threshold(0.8);
    pipeline.add_guard(Box::new(ContentLengthGuard::new(10_000)));
    pipeline.add_guard(Box::new(PatternGuard::new(vec![
        "DROP TABLE".to_string(),
        "rm -rf".to_string(),
        "<script>".to_string(),
    ])));

    c.bench_function("guardrail_check_input", |b| {
        b.iter(|| {
            let _ = pipeline.check_input("Hello, can you explain how Rust borrow checker works?");
        });
    });
}

fn bench_html_extract_text(c: &mut Criterion) {
    let extractor = HtmlExtractor::new(HtmlExtractionConfig::default());
    let html = r#"<!DOCTYPE html><html><head><title>Benchmark Page</title>
        <meta name="description" content="A test page for benchmarking">
        <style>body { font-size: 14px; }</style>
        <script>console.log("ignored");</script></head>
        <body><h1>Welcome to the Benchmark</h1>
        <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        <ul><li>First item</li><li>Second item</li><li>Third item</li></ul>
        <div class="content"><p>Another paragraph with a <a href="https://example.com">link</a>.</p>
        <table><tr><th>Name</th><th>Value</th></tr><tr><td>Alpha</td><td>100</td></tr></table>
        </div><footer><p>Copyright 2026</p></footer></body></html>"#;

    c.bench_function("html_extract_text", |b| {
        b.iter(|| {
            let _ = extractor.extract_text(html);
        });
    });
}

fn bench_rate_limiter(c: &mut Criterion) {
    // High limit so the benchmark does not get throttled
    let limiter = ServerRateLimiter::new(1_000_000);

    c.bench_function("rate_limiter_check", |b| {
        b.iter(|| {
            let _ = limiter.check_rate_limit();
        });
    });
}

fn bench_ws_frame_encode(c: &mut Criterion) {
    let frame = WsFrame::text("hello world");

    c.bench_function("ws_frame_encode", |b| {
        b.iter(|| {
            let _ = frame.encode();
        });
    });
}

fn bench_gzip_compress(c: &mut Criterion) {
    // ~1 KB JSON payload
    let json_payload = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions about programming."},
            {"role": "user", "content": "Explain the difference between stack and heap memory allocation in systems programming languages like Rust and C++."},
            {"role": "assistant", "content": "Stack memory is automatically managed, allocated and deallocated in LIFO order. Heap memory is dynamically allocated and must be explicitly freed or managed by a garbage collector or ownership system."},
            {"role": "user", "content": "Can you give me a concrete example in Rust showing both stack and heap allocation?"}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": false
    })
    .to_string();

    c.bench_function("gzip_compress_1kb_json", |b| {
        b.iter(|| {
            let _ = ai_assistant::compress_string(&json_payload, CompressionAlgorithm::Gzip);
        });
    });
}

fn bench_text_transform(c: &mut Criterion) {
    let input = "Hello World! This is a benchmark test for the text transformation pipeline. \
        It includes UPPERCASE words, lowercase words, and Mixed Case words. \
        Special chars: @#$%^&*() and numbers: 12345. End of input.";

    c.bench_function("text_transform_pipeline", |b| {
        b.iter(|| {
            let mut transformer = TextTransformer::new(input);
            transformer.apply(Transform::ToLowerCase);
            transformer.apply(Transform::ReplaceAll {
                find: "benchmark".to_string(),
                replace: "perf".to_string(),
                case_sensitive: false,
            });
            transformer.apply(Transform::Trim);
            let _ = transformer.into_text();
        });
    });
}

fn bench_json_serialize(c: &mut Criterion) {
    let messages: Vec<ChatMessage> = vec![
        ChatMessage::system("You are a helpful coding assistant."),
        ChatMessage::user("Write a function to compute fibonacci numbers."),
        ChatMessage::assistant("Here is a Rust function:\n```rust\nfn fib(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fib(n-1) + fib(n-2),\n    }\n}\n```"),
        ChatMessage::user("Can you make it iterative?"),
    ];

    c.bench_function("json_serialize_chat_messages", |b| {
        b.iter(|| {
            let _ = serde_json::to_string(&messages).unwrap();
        });
    });
}

fn bench_intent_classify_single(c: &mut Criterion) {
    let classifier = IntentClassifier::new();

    c.bench_function("intent_classify_single_msg", |b| {
        b.iter(|| {
            let _ = classifier.classify(
                "Please search the web for the latest Rust async runtime benchmarks and summarize the results",
            );
        });
    });
}

fn bench_context_window_trim(c: &mut Criterion) {
    // Small window that forces eviction
    let config = ContextWindowConfig {
        max_tokens: 200,
        response_reserve: 50,
        min_messages: 2,
        preserve_system: true,
        ..ContextWindowConfig::default()
    };

    c.bench_function("context_window_trim_50_msgs", |b| {
        b.iter(|| {
            let mut window = ContextWindow::new(config.clone());
            window.set_system("You are a helpful assistant.");
            for i in 0..50 {
                window.add(ContextMessage::new(
                    if i % 2 == 0 { "user" } else { "assistant" },
                    &format!("This is message number {} with enough content to use tokens.", i),
                ));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// v14 benchmarks (17-28): RAG, embeddings, vector, knowledge graph, tools, PII
// ---------------------------------------------------------------------------

fn bench_smart_chunker(c: &mut Criterion) {
    let chunker = SmartChunker::new(ChunkingConfig::default());

    let document = "Rust is a systems programming language focused on safety and performance. \
        It achieves memory safety without garbage collection through its ownership system. \
        Each value in Rust has a single owner, and when that owner goes out of scope, the \
        value is automatically dropped. This prevents common bugs like use-after-free and \
        double-free errors at compile time rather than runtime.\n\n\
        The borrow checker is the key component of the Rust compiler that enforces ownership \
        rules. It statically analyzes code to ensure that references do not outlive the data \
        they point to, and that mutable references are exclusive. This eliminates data races \
        and memory safety issues without runtime overhead.\n\n\
        Traits in Rust are similar to interfaces in other languages but more powerful. They \
        enable polymorphism, operator overloading, and generic programming. Trait objects allow \
        for dynamic dispatch when needed, while monomorphization provides zero-cost abstractions \
        for generics at compile time.";

    c.bench_function("smart_chunker_1k_chars", |b| {
        b.iter(|| {
            let _ = chunker.chunk(&document);
        });
    });
}

fn bench_rag_fts_search(c: &mut Criterion) {
    // Build a temp RAG DB and populate it with 100 documents
    let tmp = std::env::temp_dir().join("bench_rag_fts.db");
    let db = RagDb::open(&tmp).expect("open RAG DB");

    for i in 0..100 {
        let source = format!("doc_{}", i);
        let content = format!(
            "Document {} discusses topic {} which covers aspects of machine learning, \
             natural language processing, and computer vision applied to real-world \
             problems in healthcare, finance, and autonomous systems. Keywords: \
             transformer, attention, gradient, backpropagation, neural network.",
            i,
            i % 10
        );
        let _ = db.index_document(&source, &content);
    }

    c.bench_function("rag_fts_search_100_docs", |b| {
        b.iter(|| {
            let _ = db.search_knowledge("machine learning transformer attention", 2000, 10);
        });
    });
}

fn bench_rag_build_context(c: &mut Criterion) {
    // Pre-create a batch of KnowledgeChunks
    let chunks: Vec<KnowledgeChunk> = (0..20)
        .map(|i| KnowledgeChunk {
            id: i,
            source: format!("source_{}", i % 5),
            section: format!("section_{}", i),
            content: format!(
                "This is chunk {} containing relevant information about the query topic. \
                 It includes details about implementation, architecture, and design patterns \
                 used in production systems.",
                i
            ),
            token_count: 50,
        })
        .collect();

    c.bench_function("rag_build_context_20_chunks", |b| {
        b.iter(|| {
            let _ = KnowledgeUsage::from_chunks("machine learning query", &chunks);
        });
    });
}

fn bench_cosine_similarity_multi_dim(c: &mut Criterion) {
    // 1536-dimension vectors (OpenAI ada-002 embedding size)
    let a: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.002).cos()).collect();

    c.bench_function("cosine_similarity_1536d", |b_iter| {
        b_iter.iter(|| {
            let _ = ai_assistant::cosine_similarity(&a, &b);
        });
    });
}

fn bench_hnsw_vector_search(c: &mut Criterion) {
    let config = HnswConfig {
        m: 16,
        m_max: 32,
        ef_construction: 100,
        ef_search: 50,
        ..HnswConfig::default()
    };
    let mut index = HnswIndex::new(config);

    // Insert 1000 128-dim vectors
    for i in 0..1000 {
        let vec: Vec<f32> = (0..128)
            .map(|d| ((i * 128 + d) as f32 * 0.01).sin())
            .collect();
        index.insert(
            &format!("vec_{}", i),
            vec,
            serde_json::json!({"idx": i}),
        );
    }

    let query: Vec<f32> = (0..128).map(|d| (d as f32 * 0.05).cos()).collect();

    c.bench_function("hnsw_search_1k_vectors_128d", |b| {
        b.iter(|| {
            let _ = index.search(&query, 10);
        });
    });
}

fn bench_knowledge_graph_traversal(c: &mut Criterion) {
    let config = KnowledgeGraphConfig::default();
    let store = KnowledgeGraphStore::in_memory(config)
        .expect("create in-memory knowledge graph store");

    // Build a graph with 100 nodes and ~200 edges
    let mut entity_ids = Vec::new();
    for i in 0..100 {
        let id = store
            .get_or_create_entity(
                &format!("Entity_{}", i),
                KGEntityType::Concept,
                &[],
            )
            .expect("create entity");
        entity_ids.push(id);
    }

    // Connect each node to 2 neighbors (chain + skip)
    for i in 0..100 {
        let next = (i + 1) % 100;
        let skip = (i + 7) % 100;
        let _ = store.add_relation(
            entity_ids[i],
            entity_ids[next],
            "relates_to",
            0.9,
            None,
            None,
        );
        let _ = store.add_relation(
            entity_ids[i],
            entity_ids[skip],
            "connects_to",
            0.8,
            None,
            None,
        );
    }

    c.bench_function("knowledge_graph_query_relations_100", |b| {
        b.iter(|| {
            // Traverse from entity 0 with depth 2
            let _ = store.get_relations_from(entity_ids[0], 2);
        });
    });
}

fn bench_tool_call_parsing(c: &mut Criterion) {
    let response_json = r#"```json
{"name": "search", "arguments": {"query": "Rust programming", "limit": 10}}
```

Based on the search results, here is some information.

```tool
{"name": "calculate", "arguments": {"expression": "2 + 2 * 3"}}
```"#;

    c.bench_function("tool_call_parsing_markdown", |b| {
        b.iter(|| {
            let _ = parse_tool_calls(response_json);
        });
    });
}

fn bench_pii_detection(c: &mut Criterion) {
    let detector = PiiDetector::new(PiiConfig::default());

    let text = "Please contact John Smith at john.smith@example.com or call 555-123-4567. \
        His SSN is 123-45-6789 and he lives at 123 Main Street, Springfield, IL 62701. \
        Credit card: 4111-1111-1111-1111. IP address: 192.168.1.100. \
        Meeting scheduled for tomorrow at the office. No PII in this last sentence.";

    c.bench_function("pii_detection_1k_chars", |b| {
        b.iter(|| {
            let _ = detector.detect(text);
        });
    });
}

fn bench_embedding_train_and_embed(c: &mut Criterion) {
    let config = EmbeddingConfig {
        dimensions: 128,
        min_word_freq: 1,
        max_vocab_size: 5000,
        use_subwords: false,
        ngram_range: (1, 2),
    };
    let mut embedder = LocalEmbedder::new(config);

    let docs = vec![
        "Rust is a systems programming language focused on safety and concurrency.",
        "Python is widely used for machine learning and data science applications.",
        "JavaScript runs in the browser and on the server with Node.js runtime.",
        "Go was designed at Google for scalable network services and cloud infrastructure.",
        "TypeScript adds static typing to JavaScript for better developer experience.",
    ];
    embedder.train(&docs);

    c.bench_function("embedding_generate_128d", |b| {
        b.iter(|| {
            let _ = embedder.embed("systems programming language safety performance");
        });
    });
}

fn bench_json_schema_validation(c: &mut Criterion) {
    // Build a complex schema
    let schema = JsonSchema::new("UserProfile")
        .with_description("A user profile object")
        .with_property(
            "name",
            SchemaProperty::string().with_description("Full name"),
        )
        .with_property(
            "age",
            SchemaProperty::integer()
                .with_minimum(0.0)
                .with_maximum(150.0),
        )
        .with_property(
            "email",
            SchemaProperty::string().with_pattern(r"^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"),
        )
        .with_property(
            "tags",
            SchemaProperty::array(SchemaProperty::string()),
        )
        .with_property("address", SchemaProperty::object());

    c.bench_function("json_schema_to_prompt", |b| {
        b.iter(|| {
            let _ = schema.to_prompt();
        });
    });
}

fn bench_embedding_batch(c: &mut Criterion) {
    let config = EmbeddingConfig {
        dimensions: 128,
        min_word_freq: 1,
        max_vocab_size: 5000,
        use_subwords: false,
        ngram_range: (1, 2),
    };
    let mut embedder = LocalEmbedder::new(config);

    let docs = vec![
        "Machine learning models require large datasets for training.",
        "Neural networks use backpropagation to update weights.",
        "Transfer learning allows reusing pretrained model weights.",
        "Attention mechanisms improve sequence-to-sequence models.",
        "Reinforcement learning agents learn from environment rewards.",
    ];
    embedder.train(&docs);

    let queries: Vec<&str> = vec![
        "deep learning neural network training",
        "natural language processing transformers",
        "computer vision image classification",
        "reinforcement learning policy gradient",
        "generative adversarial networks",
    ];

    c.bench_function("embedding_batch_5x128d", |b| {
        b.iter(|| {
            let _ = embedder.embed_batch(&queries);
        });
    });
}

fn bench_rag_index_document(c: &mut Criterion) {
    let content = "Artificial intelligence (AI) refers to the simulation of human intelligence \
        in machines that are programmed to think and learn. The field encompasses machine learning, \
        neural networks, natural language processing, computer vision, and robotics. Modern AI \
        systems can perform tasks such as image recognition, speech processing, decision making, \
        and language translation with increasing accuracy. Deep learning, a subset of machine \
        learning, uses multi-layered neural networks to learn complex patterns from large datasets. \
        Transformer architectures have revolutionized NLP tasks including text generation, \
        summarization, and question answering. ";

    // Use an atomic counter to give each iteration a unique source name
    let counter = std::sync::atomic::AtomicU64::new(0);
    let tmp = std::env::temp_dir().join("bench_rag_index.db");
    let db = RagDb::open(&tmp).expect("open RAG DB");

    c.bench_function("rag_index_document_2k_chars", |b| {
        b.iter(|| {
            let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = db.index_document(&format!("bench_doc_{}", n), &content.repeat(3));
        });
    });
}

criterion_group!(
    benches,
    // Original 6
    bench_intent_classification,
    bench_conversation_compaction,
    bench_prompt_shortener,
    bench_sentiment_analysis,
    bench_sha256_signing,
    bench_template_rendering,
    // v12 benchmarks (7-16)
    bench_cosine_similarity,
    bench_guardrail_check,
    bench_html_extract_text,
    bench_rate_limiter,
    bench_ws_frame_encode,
    bench_gzip_compress,
    bench_text_transform,
    bench_json_serialize,
    bench_intent_classify_single,
    bench_context_window_trim,
    // v14 benchmarks (17-28)
    bench_smart_chunker,
    bench_rag_fts_search,
    bench_rag_build_context,
    bench_cosine_similarity_multi_dim,
    bench_hnsw_vector_search,
    bench_knowledge_graph_traversal,
    bench_tool_call_parsing,
    bench_pii_detection,
    bench_embedding_train_and_embed,
    bench_json_schema_validation,
    bench_embedding_batch,
    bench_rag_index_document,
);
criterion_main!(benches);
