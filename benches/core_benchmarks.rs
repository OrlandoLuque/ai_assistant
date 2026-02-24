use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

use ai_assistant::{
    ChatMessage, CompactableMessage, CompressionAlgorithm, ContentLengthGuard, ContextMessage,
    ContextWindow, ContextWindowConfig, ConversationCompactor, ConversationTemplate,
    GuardrailPipeline, HtmlExtractionConfig, HtmlExtractor, IntentClassifier, PatternGuard,
    PromptShortener, RequestSigner, SentimentAnalyzer, ServerRateLimiter, SignatureAlgorithm,
    TemplateCategory, TextTransformer, Transform, WsFrame,
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

criterion_group!(
    benches,
    // Original 6
    bench_intent_classification,
    bench_conversation_compaction,
    bench_prompt_shortener,
    bench_sentiment_analysis,
    bench_sha256_signing,
    bench_template_rendering,
    // New 10
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
);
criterion_main!(benches);
