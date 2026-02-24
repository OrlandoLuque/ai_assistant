//! Distillation pipeline demo.
//!
//! Run with: cargo run --example distillation_demo --features distillation
//!
//! Demonstrates trajectory collection, scoring, dataset building,
//! and the data flywheel for model distillation.

use ai_assistant::{
    TrajectoryCollector, TrajectoryStep, DistillationStepType, TrajectoryOutcome,
    TrajectoryDataset, DatasetBuilder, DatasetConfig, DatasetFormat,
};

fn main() {
    println!("=== Distillation Pipeline Demo ===\n");

    // 1. Create a trajectory collector and record a trajectory
    let mut collector = TrajectoryCollector::new();

    let tid = collector.start_trajectory("agent-gpt4", "Answer: What is the capital of France?");
    println!("Started trajectory: {}", tid);

    // Add steps to the trajectory
    collector.add_step(&tid, TrajectoryStep {
        step_number: 1,
        step_type: DistillationStepType::LlmCall {
            model: "gpt-4".to_string(),
            temperature: 0.7,
        },
        input: "What is the capital of France?".to_string(),
        output: "The capital of France is Paris.".to_string(),
        tokens_used: 42,
        latency_ms: 350,
        timestamp: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    }).unwrap();

    // Finish the trajectory with a successful outcome
    collector.finish_trajectory(&tid, TrajectoryOutcome::Success { score: 0.95 }).unwrap();

    // Add a second trajectory
    let tid2 = collector.start_trajectory("agent-gpt4", "Translate: Hello world to Spanish");
    collector.add_step(&tid2, TrajectoryStep {
        step_number: 1,
        step_type: DistillationStepType::LlmCall {
            model: "gpt-4".to_string(),
            temperature: 0.3,
        },
        input: "Translate to Spanish: Hello world".to_string(),
        output: "Hola mundo".to_string(),
        tokens_used: 28,
        latency_ms: 200,
        timestamp: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    }).unwrap();
    collector.finish_trajectory(&tid2, TrajectoryOutcome::Success { score: 0.90 }).unwrap();

    let trajectories = collector.list_trajectories();
    println!("Completed trajectories: {}", trajectories.len());

    // 2. Build a dataset from trajectories
    let owned: Vec<_> = trajectories.into_iter().cloned().collect();
    let dataset = TrajectoryDataset::new(owned);
    println!("Dataset size: {} trajectories", dataset.len());

    // 3. Configure and build fine-tuning entries
    let config = DatasetConfig::new(DatasetFormat::OpenAIJsonl);
    let entries = DatasetBuilder::build(&dataset, &config);
    match entries {
        Ok(entries) => {
            println!("\nDataset entries: {}", entries.len());
            for (i, entry) in entries.iter().enumerate() {
                println!("  Entry {}: input='{}' output='{}'",
                    i, &entry.input[..entry.input.len().min(40)],
                    &entry.output[..entry.output.len().min(40)]);
            }
        }
        Err(e) => println!("Build error: {}", e),
    }

    println!("\n=== Done ===");
}
