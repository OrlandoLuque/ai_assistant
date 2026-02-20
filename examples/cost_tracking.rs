//! Example: Cost Tracking & Budget Management
//!
//! Demonstrates using CostEstimator and CostDashboard for session-level
//! cost monitoring, model cost breakdowns, and CSV export.
//!
//! Run with: cargo run --example cost_tracking --features full

use ai_assistant::{
    CostDashboard, CostEstimator, BudgetManager, RequestType,
};

fn main() {
    println!("=== Cost Tracking Demo ===\n");

    // 1. Direct cost estimation
    let estimator = CostEstimator::new();
    let estimate = estimator.estimate("gpt-4", "api", 1000, 500);
    println!("--- Cost Estimator ---");
    println!("Model: gpt-4, Input: 1000 tokens, Output: 500 tokens");
    println!("  Estimated cost: ${:.6} {}", estimate.cost, estimate.currency);
    println!("  Provider:       {}", estimate.provider);

    let estimate2 = estimator.estimate("gpt-3.5-turbo", "api", 1000, 500);
    println!("\nModel: gpt-3.5-turbo, Input: 1000 tokens, Output: 500 tokens");
    println!("  Estimated cost: ${:.6}", estimate2.cost);

    // 2. Session-level cost dashboard with budget
    println!("\n--- Cost Dashboard ---");
    let budget = BudgetManager::new()
        .with_daily_limit(5.00)
        .with_monthly_limit(100.00);
    let mut dashboard = CostDashboard::with_budget(budget);

    // Simulate several requests across different models
    dashboard.record("gpt-4", 2000, 1000, RequestType::Chat);
    dashboard.record("gpt-4", 1500, 800, RequestType::Chat);
    dashboard.record("gpt-3.5-turbo", 3000, 1200, RequestType::Chat);
    dashboard.record("gpt-3.5-turbo", 2500, 900, RequestType::Completion);
    dashboard.record("claude-3-sonnet", 1800, 600, RequestType::Chat);
    dashboard.record("gpt-4", 500, 0, RequestType::Embedding);

    println!("Recorded {} requests", dashboard.total_requests());

    // 3. Cost by model breakdown
    println!("\n--- Cost by Model ---");
    let by_model = dashboard.cost_by_model();
    let mut models: Vec<_> = by_model.iter().collect();
    models.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (model, cost) in &models {
        println!("  {}: ${:.6}", model, cost);
    }

    // 4. Most expensive requests
    println!("\n--- Top 3 Most Expensive Requests ---");
    for (i, entry) in dashboard.most_expensive(3).iter().enumerate() {
        println!(
            "  {}. {} ({}) - {} in / {} out = ${:.6}",
            i + 1, entry.model, entry.request_type,
            entry.input_tokens, entry.output_tokens, entry.cost_usd
        );
    }

    // 5. Full report
    println!("\n{}", dashboard.format_report());

    // 6. CSV export
    println!("\n--- CSV Export (first 3 lines) ---");
    let csv = dashboard.export_csv();
    for (i, line) in csv.lines().take(3).enumerate() {
        println!("  {}: {}", i, line);
    }

    println!("\nDone! Total session cost: ${:.6}", dashboard.total_cost());
}
