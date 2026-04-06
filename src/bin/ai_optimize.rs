// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_optimize — Configuration optimizer CLI.
//!
//! Run with: `cargo run --bin ai_optimize --features full -- <command>`
//!
//! ## Commands
//!
//! ```text
//! ai_optimize run [--rounds N]         Run N optimization rounds (default: 10)
//! ai_optimize status                   Show phase, best config, arm stats
//! ai_optimize report [--html FILE] [--json]  Generate report
//! ai_optimize reset                    Clear optimizer state
//! ai_optimize arms                     List all bandit arms
//! ai_optimize cache clear              Clear LLM cache
//! ai_optimize version                  Show version info
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use ai_assistant::config_optimizer::{
    get_code_version, ConfigOptimizer, ConfigPoint, ConfigValue, OptimizationPhase, OptimizerConfig,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("run") => cmd_run(&args[2..]),
        Some("status") => cmd_status(&args[2..]),
        Some("report") => cmd_report(&args[2..]),
        Some("reset") => cmd_reset(&args[2..]),
        Some("arms") => cmd_arms(&args[2..]),
        Some("cache") => cmd_cache(&args[2..]),
        Some("version") => cmd_version(),
        Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!(
                "{}Unknown command: {}. Run ai_optimize --help{}",
                RED, other, RESET
            );
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// ANSI color helpers
// =============================================================================

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn state_path() -> PathBuf {
    let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    dir.join(".ai_optimizer_state.json")
}

fn load_or_create() -> ConfigOptimizer {
    let path = state_path();
    if path.exists() {
        match ConfigOptimizer::load(&path) {
            Ok(opt) => opt,
            Err(e) => {
                eprintln!(
                    "{}Warning: Failed to load state: {}. Starting fresh.{}",
                    YELLOW, e, RESET
                );
                ConfigOptimizer::new(OptimizerConfig::default())
            }
        }
    } else {
        ConfigOptimizer::new(OptimizerConfig::default())
    }
}

fn save_state(optimizer: &ConfigOptimizer) {
    let path = state_path();
    if let Err(e) = optimizer.save(&path) {
        eprintln!("{}Error saving state: {}{}", RED, e, RESET);
    }
}

/// Built-in benchmark function used when no external benchmark is available.
/// Evaluates a config point by scoring common parameter choices heuristically.
fn builtin_benchmark(point: &ConfigPoint) -> f64 {
    let mut score = 0.5; // baseline

    // Temperature: 0.7 is a common sweet spot
    if let Some(ConfigValue::Float(t)) = point.get("temperature") {
        let ideal = 0.7;
        score += 0.15 * (1.0 - (t - ideal).abs().min(1.0));
    }

    // top_p: 0.9 is common
    if let Some(ConfigValue::Float(p)) = point.get("top_p") {
        let ideal = 0.9;
        score += 0.10 * (1.0 - (p - ideal).abs().min(1.0));
    }

    // Cache: enabled is better
    if let Some(ConfigValue::Bool(b)) = point.get("use_cache") {
        if *b {
            score += 0.05;
        }
    }

    // Streaming: enabled is better for latency
    if let Some(ConfigValue::Bool(b)) = point.get("use_streaming") {
        if *b {
            score += 0.03;
        }
    }

    // Guardrails: safety is a plus
    if let Some(ConfigValue::Bool(b)) = point.get("use_guardrails") {
        if *b {
            score += 0.05;
        }
    }

    // Max tokens: more is usually better (up to a point)
    if let Some(ConfigValue::Uint(n)) = point.get("max_tokens") {
        let n = *n as f64;
        score += 0.07 * (n / (n + 1024.0));
    }

    // RAG: helps quality
    if let Some(ConfigValue::Bool(b)) = point.get("use_rag") {
        if *b {
            score += 0.05;
        }
    }

    score.clamp(0.0, 1.0)
}

// =============================================================================
// Commands
// =============================================================================

fn cmd_run(args: &[String]) -> ExitCode {
    let mut rounds = 10usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_ref() {
            "--rounds" | "-n" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                    rounds = n;
                    i += 1;
                } else {
                    eprintln!("{}--rounds requires a number{}", RED, RESET);
                    return ExitCode::from(1);
                }
            }
            _ => {
                eprintln!("{}Unknown flag: {}{}", RED, args[i], RESET);
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let mut optimizer = load_or_create();
    optimizer.check_version_change();

    println!("{}{}=== Configuration Optimizer ==={}\n", BOLD, CYAN, RESET);
    println!("{}Phase:{} {:?}", DIM, RESET, optimizer.phase());
    println!(
        "{}Running {} optimization round(s)...{}\n",
        GREEN, rounds, RESET
    );

    for r in 1..=rounds {
        let phase = optimizer.phase();
        if phase == OptimizationPhase::Done {
            println!(
                "{}Optimization complete (Done phase reached).{}",
                GREEN, RESET
            );
            break;
        }

        let result = optimizer.run_round(&builtin_benchmark);

        // Progress indicator
        let phase_str = match result.phase {
            OptimizationPhase::Ablation => format!("{}ABL{}", YELLOW, RESET),
            OptimizationPhase::BayesianSearch => format!("{}BAY{}", CYAN, RESET),
            OptimizationPhase::FineTuning => format!("{}FIN{}", GREEN, RESET),
            OptimizationPhase::Done => format!("{}DONE{}", DIM, RESET),
        };

        println!(
            "  [{}/{}] {} score={:.4} bench={}",
            r, rounds, phase_str, result.quality_score, result.benchmark_name
        );
    }

    // Check regressions
    let regressions = optimizer.detect_regressions();
    if !regressions.is_empty() {
        println!("\n{}Regressions detected:{}", RED, RESET);
        for (id, old, new) in &regressions {
            println!(
                "  {} {} {:.4} -> {:.4} ({:.1}% drop){}",
                RED,
                id,
                old,
                new,
                (old - new) / old * 100.0,
                RESET
            );
        }
    }

    // Summary
    if let Some((_cfg, score)) = optimizer.best_config() {
        println!("\n{}Best score: {:.4}{}", BOLD, score, RESET);
    }

    save_state(&optimizer);
    println!("\n{}State saved.{}", DIM, RESET);

    ExitCode::SUCCESS
}

fn cmd_status(_args: &[String]) -> ExitCode {
    let optimizer = load_or_create();

    println!("{}{}=== Optimizer Status ==={}\n", BOLD, CYAN, RESET);
    println!("{}Phase:{}       {:?}", DIM, RESET, optimizer.phase());
    println!("{}Rounds:{}      {}", DIM, RESET, optimizer.total_rounds());
    println!(
        "{}Evaluations:{} {}",
        DIM,
        RESET,
        optimizer.all_evaluations().len()
    );
    println!("{}Arms:{}        {}", DIM, RESET, optimizer.arms().len());

    if let Some((_, score)) = optimizer.best_config() {
        println!("{}Best score:{} {}{:.4}{}", DIM, RESET, GREEN, score, RESET);
    } else {
        println!("{}Best score:{} (none yet)", DIM, RESET);
    }

    let fi = optimizer.feature_importance();
    if !fi.is_empty() {
        println!("\n{}Feature Importance:{}", BOLD, RESET);
        for r in fi.iter().take(10) {
            let color = if r.impact > 0.0 { GREEN } else { RED };
            println!(
                "  #{:<2} {:<25} impact: {}{:+.4}{}",
                r.importance_rank, r.feature_name, color, r.impact, RESET
            );
        }
    }

    ExitCode::SUCCESS
}

fn cmd_report(args: &[String]) -> ExitCode {
    let optimizer = load_or_create();
    let mut html_file: Option<String> = None;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_ref() {
            "--html" => {
                html_file = args.get(i + 1).map(|s| s.to_string());
                i += 1;
            }
            "--json" => {
                json_output = true;
            }
            _ => {}
        }
        i += 1;
    }

    if json_output {
        match serde_json::to_string_pretty(&optimizer.all_evaluations()) {
            Ok(json) => {
                println!("{}", json);
            }
            Err(e) => {
                eprintln!("{}Failed to generate JSON: {}{}", RED, e, RESET);
                return ExitCode::from(1);
            }
        }
        return ExitCode::SUCCESS;
    }

    if let Some(path) = html_file {
        let html = optimizer.report_html();
        match std::fs::write(&path, &html) {
            Ok(_) => {
                println!("{}HTML report written to: {}{}", GREEN, path, RESET);
            }
            Err(e) => {
                eprintln!("{}Failed to write HTML: {}{}", RED, e, RESET);
                return ExitCode::from(1);
            }
        }
    } else {
        println!("{}", optimizer.report());
    }

    ExitCode::SUCCESS
}

fn cmd_reset(_args: &[String]) -> ExitCode {
    let path = state_path();
    if path.exists() {
        match std::fs::remove_file(&path) {
            Ok(_) => {
                println!("{}Optimizer state cleared.{}", GREEN, RESET);
            }
            Err(e) => {
                eprintln!("{}Failed to delete state file: {}{}", RED, e, RESET);
                return ExitCode::from(1);
            }
        }
    } else {
        println!("{}No state file found — nothing to reset.{}", YELLOW, RESET);
    }
    ExitCode::SUCCESS
}

fn cmd_arms(_args: &[String]) -> ExitCode {
    let optimizer = load_or_create();

    let arms = optimizer.arms();
    if arms.is_empty() {
        println!(
            "{}No arms registered yet. Run some optimization rounds first.{}",
            YELLOW, RESET
        );
        return ExitCode::SUCCESS;
    }

    println!(
        "{}{}=== Bandit Arms ({}) ==={}\n",
        BOLD,
        CYAN,
        arms.len(),
        RESET
    );

    println!(
        "  {}{:<20} {:>6} {:>10} {:>10} {:>8}{}",
        DIM, "ID", "Pulls", "Mean", "Thompson", "Status", RESET
    );
    println!("  {}", "-".repeat(60));

    for arm in arms {
        let status = if arm.available {
            format!("{}OK{}", GREEN, RESET)
        } else {
            format!("{}OFF{}", RED, RESET)
        };

        println!(
            "  {:<20} {:>6} {:>10.4} {:>10.4} {:>8}",
            arm.id,
            arm.pull_count,
            arm.mean_reward(),
            arm.thompson_score(),
            status
        );
    }

    ExitCode::SUCCESS
}

fn cmd_cache(args: &[String]) -> ExitCode {
    match args.first().map(|s| s.as_str()) {
        Some("clear") => {
            println!(
                "{}LLM cache cleared (no persistent LLM cache in optimizer).{}",
                GREEN, RESET
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{}Usage: ai_optimize cache clear{}", RED, RESET);
            ExitCode::from(1)
        }
    }
}

fn cmd_version() -> ExitCode {
    let version = env!("CARGO_PKG_VERSION");
    let code_version = get_code_version();

    println!(
        "{}{}ai_optimize{} v{} (ai_assistant)",
        BOLD, CYAN, RESET, version
    );
    println!("{}Code version:{} {}", DIM, RESET, code_version);
    println!("{}Platform:{}     {}", DIM, RESET, std::env::consts::OS);
    println!("{}Arch:{}         {}", DIM, RESET, std::env::consts::ARCH);

    ExitCode::SUCCESS
}

fn print_help() {
    println!(
        "{}{}ai_optimize{} — Configuration Optimizer for AI Assistant\n",
        BOLD, CYAN, RESET
    );
    println!("{}USAGE:{}", BOLD, RESET);
    println!("  ai_optimize <COMMAND> [OPTIONS]\n");
    println!("{}COMMANDS:{}", BOLD, RESET);
    println!(
        "  {}run{}    [--rounds N]         Run N optimization rounds (default: 10)",
        GREEN, RESET
    );
    println!(
        "  {}status{}                      Show phase, best config, arm stats",
        GREEN, RESET
    );
    println!(
        "  {}report{} [--html FILE] [--json]  Generate report",
        GREEN, RESET
    );
    println!(
        "  {}reset{}                       Clear optimizer state",
        GREEN, RESET
    );
    println!(
        "  {}arms{}                        List all bandit arms",
        GREEN, RESET
    );
    println!(
        "  {}cache{}  clear                Clear LLM cache",
        GREEN, RESET
    );
    println!(
        "  {}version{}                     Show version info",
        GREEN, RESET
    );
    println!("\n{}EXAMPLES:{}", BOLD, RESET);
    println!("  ai_optimize run --rounds 50");
    println!("  ai_optimize report --html optimizer_report.html");
    println!("  ai_optimize status");
}
