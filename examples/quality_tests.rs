//! Example: Running conversation quality tests
//!
//! This example demonstrates how to create and run quality tests for AI conversations.
//! Tests can verify that responses contain expected keywords, meet time requirements,
//! and retrieve relevant knowledge sources.
//!
//! Run with: cargo run --example quality_tests --features rag

use ai_assistant::{
    AiAssistant, ConversationTestCase, MessageMetrics, TestSuite, TestSuiteResults,
};
use std::path::Path;

fn main() {
    println!("AI Assistant Quality Test Example\n");

    // Create a test suite for Star Citizen knowledge
    let mut suite = TestSuite::new(
        "Star Citizen Knowledge Tests",
        "Tests for verifying the AI correctly retrieves and uses Star Citizen game knowledge",
    );

    // Test 1: Basic ship query
    suite.add_test(ConversationTestCase {
        name: "Aurora ship information".to_string(),
        query: "What is the Aurora ship in Star Citizen?".to_string(),
        expected_keywords: vec!["Aurora".to_string(), "starter".to_string()],
        forbidden_keywords: vec!["error".to_string(), "I don't know".to_string()],
        expected_sources: vec!["ships".to_string()],
        max_response_time_ms: Some(10000),
        min_output_tokens: Some(50),
        max_output_tokens: Some(500),
    });

    // Test 2: Gameplay mechanics
    suite.add_test(ConversationTestCase {
        name: "Quantum travel explanation".to_string(),
        query: "How does quantum travel work?".to_string(),
        expected_keywords: vec!["quantum".to_string(), "drive".to_string()],
        forbidden_keywords: vec![],
        expected_sources: vec!["mechanics".to_string()],
        max_response_time_ms: Some(15000),
        min_output_tokens: Some(30),
        max_output_tokens: None,
    });

    // Test 3: Localization question
    suite.add_test(ConversationTestCase {
        name: "Localization process".to_string(),
        query: "How do I install Spanish translations?".to_string(),
        expected_keywords: vec!["global.ini".to_string()],
        forbidden_keywords: vec![],
        expected_sources: vec!["localization".to_string()],
        max_response_time_ms: Some(10000),
        min_output_tokens: Some(40),
        max_output_tokens: None,
    });

    // Test 4: Verify response quality
    suite.add_test(ConversationTestCase {
        name: "Response is helpful".to_string(),
        query: "I'm new to Star Citizen, where should I start?".to_string(),
        expected_keywords: vec!["tutorial".to_string()],
        forbidden_keywords: vec![
            "error".to_string(),
            "cannot".to_string(),
            "unable".to_string(),
        ],
        expected_sources: vec![],
        max_response_time_ms: Some(12000),
        min_output_tokens: Some(100),
        max_output_tokens: Some(800),
    });

    // Print test suite as JSON (for saving/loading)
    println!("Test Suite JSON:");
    println!("{}\n", suite.to_json());

    // Example of evaluating results manually (in a real scenario, you'd run the AI)
    println!("=== Simulated Test Results ===\n");

    // Simulate some test results
    let results = vec![
        simulate_test_result(&suite.test_cases[0], true, "The Aurora is a popular starter ship in Star Citizen, known for being affordable and versatile.", 150, 1200),
        simulate_test_result(&suite.test_cases[1], true, "Quantum travel uses a quantum drive to propel your ship at faster-than-light speeds.", 80, 2500),
        simulate_test_result(&suite.test_cases[2], false, "You can change the language in the settings.", 20, 800),
        simulate_test_result(&suite.test_cases[3], true, "Welcome to Star Citizen! I recommend starting with the tutorial missions to learn the basics.", 120, 3000),
    ];

    // Create summary
    let summary = suite.summarize_results(results);

    // Print results
    print_results(&summary);
}

fn simulate_test_result(
    test: &ConversationTestCase,
    _passed: bool,
    response: &str,
    output_tokens: usize,
    response_time_ms: u64,
) -> ai_assistant::TestCaseResult {
    let metrics = MessageMetrics {
        timestamp: chrono::Utc::now().to_rfc3339(),
        time_to_first_token_ms: Some(response_time_ms / 3),
        total_response_time_ms: response_time_ms,
        input_tokens: 20,
        output_tokens,
        context_tokens: 2000,
        knowledge_tokens: 500,
        conversation_tokens: 0,
        knowledge_chunks_retrieved: 3,
        conversation_messages_retrieved: 0,
        context_near_limit: false,
        model: "llama3:8b".to_string(),
    };

    test.evaluate(
        response,
        &metrics,
        &["ships".to_string(), "mechanics".to_string()],
    )
}

fn print_results(results: &TestSuiteResults) {
    println!("Suite: {}", results.suite_name);
    println!("Total: {} tests", results.total_tests);
    println!(
        "Passed: {} ({:.0}%)",
        results.passed,
        results.pass_rate * 100.0
    );
    println!("Failed: {}", results.failed);
    println!("Avg Response Time: {:.0}ms", results.avg_response_time_ms);
    println!();

    for result in &results.results {
        let status = if result.passed { "✓" } else { "✗" };
        println!("{} {}", status, result.name);

        if !result.passed {
            for reason in &result.failure_reasons {
                println!("  - {}", reason);
            }
        }

        println!(
            "  Response time: {}ms, Tokens: {}",
            result.metrics.total_response_time_ms, result.metrics.output_tokens
        );
    }
}
