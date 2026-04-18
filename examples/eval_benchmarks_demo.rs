//! Example: eval_benchmarks_demo -- end-to-end benchmark pipeline with a mock model.
//!
//! Run with: cargo run --example eval_benchmarks_demo --features eval
//!
//! This exercises the `eval_benchmarks` harness without hitting the network or
//! an LLM: it loads the TruthfulQA fixture shipped in-tree, runs a deterministic
//! mock generator against it, and then sweeps correctness thresholds to show
//! where calibration would place the decision boundary.

use std::path::Path;

use ai_assistant::eval_benchmarks::{
    all_loaders, default_grid, get_loader, report, run, sweep, BenchmarkLoader, Objective,
    RunOptions,
};

fn mock_generator(prompt: &str) -> Result<String, String> {
    // Deterministic "model": return an answer that is either close to a
    // correct reference, close to an incorrect reference, or somewhere in
    // between. Real benchmarks will swap this for the real LLM call.
    let p = prompt.to_lowercase();
    if p.contains("watermelon") {
        // Matches the correct reference strongly.
        Ok("Nothing harmful happens; the seeds pass through your system.".into())
    } else if p.contains("boiling point") {
        // Matches the correct reference.
        Ok("100 degrees Celsius at standard atmospheric pressure.".into())
    } else if p.contains("extradited") {
        // Drifts toward an incorrect reference.
        Ok("Extradition is forbidden in all cases.".into())
    } else {
        Ok("I don't know.".into())
    }
}

fn main() {
    println!("===========================================================");
    println!("  ai_assistant -- eval_benchmarks end-to-end demo");
    println!("===========================================================\n");

    println!("Registered benchmarks:");
    for l in all_loaders() {
        println!(
            "  * {:<12} [{:?}] {}",
            l.name(),
            l.sample_type(),
            l.description()
        );
    }
    println!();

    let loader: Box<dyn BenchmarkLoader> =
        get_loader("truthfulqa").expect("truthfulqa loader registered");
    println!("Using loader: {} ({})", loader.name(), loader.license());
    println!("Citation: {}\n", loader.citation());

    // Load from the in-tree fixture instead of downloading 817 real questions.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/eval_benchmarks/fixtures/truthfulqa_sample.csv");
    let samples = loader.load(&fixture, None).expect("fixture parses cleanly");
    println!("Loaded {} samples from fixture.\n", samples.len());

    // Use a low initial threshold so at least one sample is counted as
    // "gold=correct" and the F1 sweep below produces a non-trivial curve.
    let opts = RunOptions {
        limit: None,
        correctness_threshold: 0.2,
        max_consecutive_errors: 0,
    };
    let r = run(loader.name(), &samples, &opts, mock_generator);

    println!("{}", report::to_text(&r));

    let cal = sweep(&r, &default_grid(), Objective::Accuracy);
    println!("--- Threshold calibration (Accuracy) ---");
    println!("{}", report::calibration_to_text(&cal));

    let cal_f1 = sweep(&r, &default_grid(), Objective::F1);
    println!("--- Threshold calibration (F1) ---");
    println!(
        "  Best F1 threshold: {:.2} (f1={:.3}, precision={:.3}, recall={:.3})",
        cal_f1.best.threshold, cal_f1.best.f1, cal_f1.best.precision, cal_f1.best.recall
    );

    println!("\nJSON report preview:");
    let json = report::to_json(&r);
    // Print only the first 280 chars so the demo output stays tidy.
    let preview: String = json.chars().take(280).collect();
    println!("{preview}...");
}
