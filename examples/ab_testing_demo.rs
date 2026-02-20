//! Example: A/B Testing Framework
//!
//! Demonstrates creating experiments, assigning users to variants,
//! recording metrics, and analyzing results for statistical significance.
//!
//! Run with: cargo run --example ab_testing_demo --features full

use ai_assistant::{ExperimentManager, ExperimentVariant};

fn main() {
    println!("=== A/B Testing Framework Demo ===\n");

    let mut mgr = ExperimentManager::new();

    // 1. Create an experiment with two prompt variants
    let variants = vec![
        ExperimentVariant::new("control", "Original system prompt", 0.5),
        ExperimentVariant::new("treatment", "Improved system prompt with CoT", 0.5),
    ];
    let exp_id = mgr
        .create_experiment("Prompt Quality Test", variants)
        .expect("failed to create experiment");
    println!("Created experiment: {}\n", exp_id);

    // 2. Assign users to variants (deterministic — same user always gets same variant)
    for i in 0..6 {
        let user = format!("user_{}", i);
        let assignment = mgr.assign_user(&exp_id, &user).unwrap();
        println!(
            "  {} -> variant {} ({})",
            user, assignment.variant_index, assignment.variant_name
        );
    }

    // 3. Simulate metric data: control scores lower, treatment scores higher
    println!("\nRecording metrics...");
    let control_scores = [6.2, 5.8, 6.5, 6.0, 5.9, 6.3, 6.1, 5.7, 6.4, 6.0];
    let treatment_scores = [8.1, 7.9, 8.5, 8.0, 7.8, 8.3, 8.2, 7.6, 8.4, 8.0];

    for &v in &control_scores {
        mgr.record_metric(&exp_id, 0, v);
    }
    for &v in &treatment_scores {
        mgr.record_metric(&exp_id, 1, v);
    }

    // 4. Record some conversions
    for _ in 0..7 {
        mgr.record_exposure(&exp_id, 0);
    }
    for _ in 0..3 {
        mgr.record_conversion(&exp_id, 0); // 3/10 = 30% conversion
    }
    for _ in 0..3 {
        mgr.record_exposure(&exp_id, 1);
    }
    for _ in 0..7 {
        mgr.record_conversion(&exp_id, 1); // 7/10 = 70% conversion
    }

    // 5. Get results at 95% confidence
    let result = mgr.get_results(&exp_id, 0.95).expect("failed to get results");
    println!("\n--- Experiment Results ---");
    println!("Significant: {}", result.is_significant);
    println!("p-value:     {:.6}", result.p_value);
    println!("Confidence:  {:.0}%", result.confidence_level * 100.0);
    if let Some(ref winner) = result.recommended_variant {
        println!("Winner:      {}", winner);
    }

    for vs in &result.variants {
        println!(
            "\n  Variant: {} (n={})",
            vs.variant_name, vs.sample_size
        );
        println!("    Mean: {:.2}, StdDev: {:.2}", vs.mean, vs.std_dev);
        println!("    Min: {:.2}, Max: {:.2}", vs.min, vs.max);
        println!(
            "    Conversions: {} ({:.0}%)",
            vs.conversion_count,
            vs.conversion_rate * 100.0
        );
    }

    // 6. Check early stopping
    match mgr.check_early_stopping(&exp_id, 0.05) {
        Ok(Some(winner)) => println!("\nEarly stopping: clear winner is '{}'", winner),
        Ok(None) => println!("\nEarly stopping: no clear winner yet"),
        Err(e) => println!("\nEarly stopping check failed: {}", e),
    }

    // 7. Stop the experiment
    mgr.stop_experiment(&exp_id).unwrap();
    let experiments = mgr.list_experiments();
    println!(
        "\nExperiment '{}' stopped. Total experiments: {}",
        exp_id,
        experiments.len()
    );
}
