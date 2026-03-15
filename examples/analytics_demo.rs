//! Example: analytics_demo -- Demonstrates the analytics capabilities of ai_assistant.
//!
//! Run with: cargo run --example analytics_demo --features analytics
//!
//! This example showcases conversation analytics, sentiment analysis,
//! latency tracking, response quality analysis, and response comparison.

use std::time::Duration;

use ai_assistant::{
    // Conversation analytics
    AnalyticsConfig, ConversationAnalytics,
    // Sentiment analysis
    Sentiment, SentimentAnalyzer,
    // Latency tracking
    LatencyRecord, LatencyTracker, RequestTimer,
    // Quality analysis & comparison
    QualityAnalyzer, QualityConfig, compare_responses,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Analytics Capabilities Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Conversation Analytics
    // ------------------------------------------------------------------
    println!("--- 1. Conversation Analytics ---\n");

    let mut config = AnalyticsConfig::default();
    config.detailed_tracking = true;
    config.track_patterns = true;
    config.track_quality = true;
    config.max_events = 5000;
    config.aggregation_interval = Duration::from_secs(3600);
    let mut analytics = ConversationAnalytics::new(config);

    // Simulate two conversation sessions
    analytics.track_conversation_start("sess-001", Some("alice"), "llama3");
    analytics.track_message(
        "sess-001", Some("alice"), "llama3",
        "How do I read a file in Rust?", true, 12, None,
    );
    analytics.track_message(
        "sess-001", Some("alice"), "llama3",
        "You can use std::fs::read_to_string for text files.", false, 18,
        Some(Duration::from_millis(320)),
    );
    analytics.track_feedback("sess-001", 0.9, Some("Very helpful!"));

    analytics.track_conversation_start("sess-002", Some("bob"), "mistral");
    analytics.track_message(
        "sess-002", Some("bob"), "mistral",
        "Explain ownership in Rust", true, 8, None,
    );
    analytics.track_message(
        "sess-002", Some("bob"), "mistral",
        "Ownership is Rust's system for managing memory...", false, 45,
        Some(Duration::from_millis(580)),
    );
    analytics.track_error(Some("sess-002"), Some("mistral"), "Connection reset");
    analytics.track_feedback("sess-002", 0.7, None);

    // Report
    let report = analytics.report();
    println!("  Conversations     : {}", report.total_conversations);
    println!("  Total messages    : {}", report.total_messages);
    println!("  Total tokens      : {}", report.total_tokens);
    println!("  Avg msgs/conv     : {:.1}", report.avg_messages_per_conversation);
    println!("  Error rate        : {:.1}%", report.error_rate * 100.0);
    println!("  Avg query length  : {:.0} chars", report.avg_query_length);
    if let Some(sat) = report.avg_satisfaction {
        println!("  Avg satisfaction  : {:.1}%", sat * 100.0);
    }
    println!("  Top models:");
    for (model, count) in &report.top_models {
        println!("    - {}: {} messages", model, count);
    }

    // Aggregated stats
    let stats = analytics.stats();
    println!("  Model usage map   : {:?}", stats.model_usage);

    // Exported events
    let exported = analytics.export_events();
    println!("  Exported events   : {}", exported.len());
    println!();

    // ------------------------------------------------------------------
    // 2. Sentiment Analysis
    // ------------------------------------------------------------------
    println!("--- 2. Sentiment Analysis ---\n");

    let analyzer = SentimentAnalyzer::new();

    let messages = [
        "This is absolutely amazing, thank you so much!",
        "The response was okay, nothing special.",
        "This is terrible and completely useless. Very disappointed.",
        "Great work, the solution is perfect and efficient!",
        "I'm confused and frustrated, this doesn't help at all.",
    ];

    for msg in &messages {
        let result = analyzer.analyze_message(msg);
        println!(
            "  {} {:<15} (score={:+.2}, confidence={:.0}%)",
            result.sentiment.emoji(),
            format!("{}", result.sentiment),
            result.score,
            result.confidence * 100.0,
        );
        if !result.positive_indicators.is_empty() {
            println!("      + {}", result.positive_indicators.join(", "));
        }
        if !result.negative_indicators.is_empty() {
            println!("      - {}", result.negative_indicators.join(", "));
        }
    }

    // Direct Sentiment enum usage
    let s = Sentiment::from_score(0.75);
    println!("\n  Sentiment::from_score(0.75) = {} (numeric={})\n", s, s.score());

    // ------------------------------------------------------------------
    // 3. Latency Tracking
    // ------------------------------------------------------------------
    println!("--- 3. Latency Tracking ---\n");

    let mut tracker = LatencyTracker::new();

    // Simulate requests to two providers
    let ollama_latencies = [150, 200, 180, 220, 170, 190, 210, 160, 250, 300];
    for ms in &ollama_latencies {
        tracker.record("ollama", Duration::from_millis(*ms), true);
    }
    // One failed request
    tracker.record("ollama", Duration::from_millis(5000), false);

    let openai_latencies = [400, 350, 420, 380, 500, 450, 370, 410, 390, 360];
    for ms in &openai_latencies {
        tracker.record("openai", Duration::from_millis(*ms), true);
    }

    // Use RequestTimer for a timed operation
    let timer = RequestTimer::start();
    // ... simulate some work ...
    std::thread::sleep(Duration::from_millis(5));
    let record = timer.finish(true);
    tracker.record_full("ollama", record);

    // Use LatencyRecord builder
    let detailed = LatencyRecord::new(Duration::from_millis(175), true)
        .with_model("llama3")
        .with_tokens(256);
    tracker.record_full("ollama", detailed);

    // Print stats
    for provider in &["ollama", "openai"] {
        if let Some(stats) = tracker.stats(provider) {
            println!("  Provider: {}", stats.provider);
            println!("    Total requests   : {}", stats.total_requests);
            println!("    Success rate     : {:.0}%", stats.success_rate * 100.0);
            println!("    Avg latency      : {:?}", stats.avg_latency);
            println!("    P50 (median)     : {:?}", stats.p50);
            println!("    P95              : {:?}", stats.p95);
            println!("    P99              : {:?}", stats.p99);
            if let Some(tps) = stats.avg_tokens_per_second {
                println!("    Avg tok/s        : {:.1}", tps);
            }
            println!();
        }
    }

    // ------------------------------------------------------------------
    // 4. Response Quality Analysis
    // ------------------------------------------------------------------
    println!("--- 4. Response Quality Analysis ---\n");

    let qa = QualityAnalyzer::new(QualityConfig::default());

    let query = "What is the difference between Vec and array in Rust?";
    let good_response = "In Rust, a Vec<T> is a growable, heap-allocated collection, \
        while an array [T; N] has a fixed size known at compile time and is \
        stack-allocated. Vec provides methods like push and pop for dynamic \
        sizing, whereas arrays offer better cache locality for small, \
        fixed-size datasets.";
    let poor_response = "Vec is a thing in Rust.";

    let good_score = qa.analyze(query, good_response, None);
    let poor_score = qa.analyze(query, poor_response, None);

    println!("  Query: \"{}\"", query);
    println!();
    println!("  Good response quality:");
    println!("    Overall      : {:.2} ({})", good_score.overall, good_score.quality_level());
    println!("    Relevance    : {:.2}", good_score.relevance);
    println!("    Coherence    : {:.2}", good_score.coherence);
    println!("    Fluency      : {:.2}", good_score.fluency);
    println!("    Completeness : {:.2}", good_score.completeness);
    if !good_score.strengths.is_empty() {
        println!("    Strengths    : {}", good_score.strengths.join(", "));
    }
    println!();
    println!("  Poor response quality:");
    println!("    Overall      : {:.2} ({})", poor_score.overall, poor_score.quality_level());
    println!("    Issues       : {}", poor_score.issues.len());
    for issue in &poor_score.issues {
        println!("      - {:?}: {}", issue.issue_type, issue.description);
    }
    println!();

    // ------------------------------------------------------------------
    // 5. Response Comparison
    // ------------------------------------------------------------------
    println!("--- 5. Response Comparison ---\n");

    let cmp_query = "How do I handle errors in Rust?";
    let candidates = [
        "Use Result<T, E> and the ? operator for propagation. \
         Match on Ok/Err for explicit handling. The anyhow crate \
         simplifies error management in applications.",
        "Rust has errors.",
        "In Rust, error handling revolves around the Result and Option \
         types. The Result enum carries Ok(T) for success and Err(E) \
         for failure. You can use match, unwrap_or, or the ? operator.",
    ];
    let refs: Vec<&str> = candidates.iter().map(|s| s.as_ref()).collect();
    let comparison = compare_responses(cmp_query, &refs, None);

    println!("  Query: \"{}\"", cmp_query);
    println!("  Best response  : #{}", comparison.best_index + 1);
    println!("  Summary        : {}", comparison.summary);
    println!("  Scores:");
    for (i, score) in comparison.scores.iter().enumerate() {
        println!(
            "    Response #{}: {:.2} ({})",
            i + 1,
            score.overall,
            score.quality_level(),
        );
    }

    println!("\n==========================================================");
    println!("  Analytics demo completed successfully.");
    println!("==========================================================");
}
