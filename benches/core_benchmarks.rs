use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

use ai_assistant::{
    CompactableMessage, ConversationCompactor, ConversationTemplate, IntentClassifier,
    PromptShortener, RequestSigner, SentimentAnalyzer, SignatureAlgorithm, TemplateCategory,
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

criterion_group!(
    benches,
    bench_intent_classification,
    bench_conversation_compaction,
    bench_prompt_shortener,
    bench_sentiment_analysis,
    bench_sha256_signing,
    bench_template_rendering,
);
criterion_main!(benches);
