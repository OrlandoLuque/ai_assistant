//! `ai_breeder` — PromptBreeder V97 auditor CLI.
//!
//! Read-only by design (see `feedback_auditable_subsystems`). Inspects the
//! checkpoint files (`*.ckpt`) and ledger JSONL files produced by a
//! `PromptBreeder` run.
//!
//! # Usage
//!
//! ```text
//! ai_breeder list-runs <DIR>
//! ai_breeder show-run <CKPT_FILE>
//! ai_breeder ledger-verify <LEDGER_JSONL>
//! ai_breeder ledger-show <LEDGER_JSONL> [--last N]
//! ai_breeder export-population <CKPT_FILE> <OUT_JSON>
//! ai_breeder compare-runs <CKPT_A> <CKPT_B>
//! ```
//!
//! Read-only. Requires `--features prompt-breeder` at build time.

use ai_assistant::prompt_breeder::checkpoint as ckpt;
use ai_assistant::prompt_breeder::{BreederEvent, Checkpoint, LedgerEntry, MutationOperator, Unit};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    match args[1].as_str() {
        "list-runs" => {
            let Some(dir) = args.get(2) else {
                eprintln!("Usage: ai_breeder list-runs <DIR>");
                return ExitCode::from(2);
            };
            cmd_list_runs(Path::new(dir))
        }
        "show-run" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_breeder show-run <CKPT_FILE>");
                return ExitCode::from(2);
            };
            cmd_show_run(Path::new(file))
        }
        "ledger-verify" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_breeder ledger-verify <LEDGER_JSONL>");
                return ExitCode::from(2);
            };
            cmd_ledger_verify(Path::new(file))
        }
        "ledger-show" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_breeder ledger-show <LEDGER_JSONL> [--last N]");
                return ExitCode::from(2);
            };
            let last: Option<usize> = get_arg(&args, "--last").and_then(|s| s.parse().ok());
            cmd_ledger_show(Path::new(file), last)
        }
        "export-population" => {
            let (Some(src), Some(dst)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: ai_breeder export-population <CKPT_FILE> <OUT_JSON>");
                return ExitCode::from(2);
            };
            cmd_export_population(Path::new(src), Path::new(dst))
        }
        "compare-runs" => {
            let (Some(a), Some(b)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: ai_breeder compare-runs <CKPT_A> <CKPT_B>");
                return ExitCode::from(2);
            };
            cmd_compare_runs(Path::new(a), Path::new(b))
        }
        other => {
            eprintln!("Unknown command: {other}. Use --help.");
            ExitCode::from(2)
        }
    }
}

// =============================================================================
// Subcommands
// =============================================================================

fn cmd_list_runs(dir: &Path) -> ExitCode {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("read-dir {}: {e}", dir.display());
            return ExitCode::from(1);
        }
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("ckpt"))
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        println!("(no *.ckpt files in {})", dir.display());
        return ExitCode::SUCCESS;
    }
    let (h_run, h_gen, h_pop, h_tip, h_file) = ("RUN_ID", "GEN", "POP", "LEDGER_TIP", "FILE");
    println!("{h_run:<40} {h_gen:>5} {h_pop:>5} {h_tip:<16} {h_file}");
    for p in &files {
        match ckpt::read(p) {
            Ok(c) => {
                let tip = short_hash(&c.ledger_tip_hash_hex);
                println!(
                    "{:<40} {:>5} {:>5} {tip:<16} {}",
                    c.run_id,
                    c.generation,
                    c.population.len(),
                    p.display(),
                );
            }
            Err(e) => {
                println!(
                    "{:<40} {:>5} {:>5} {:<16} {} [ERR: {e}]",
                    "?",
                    "?",
                    "?",
                    "?",
                    p.display(),
                );
            }
        }
    }
    ExitCode::SUCCESS
}

fn cmd_show_run(file: &Path) -> ExitCode {
    let c = match ckpt::read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("read {}: {e}", file.display());
            return ExitCode::from(1);
        }
    };
    print_checkpoint_summary(&c);
    ExitCode::SUCCESS
}

fn cmd_ledger_verify(file: &Path) -> ExitCode {
    let entries = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load {}: {e}", file.display());
            return ExitCode::from(1);
        }
    };
    println!("Loaded {} entries from {}", entries.len(), file.display());
    for (i, ev) in entries.iter().enumerate() {
        if ev.seq != i as u64 {
            eprintln!("FAIL seq gap at index {i}: got seq {}", ev.seq);
            return ExitCode::from(1);
        }
        if !ev.verify_self_hash() {
            eprintln!("FAIL self-hash mismatch at seq {}", ev.seq);
            return ExitCode::from(1);
        }
        if i == 0 {
            if !ev.prev_hash_hex.is_empty() {
                eprintln!(
                    "FAIL first entry must have empty prev_hash (seq {})",
                    ev.seq
                );
                return ExitCode::from(1);
            }
        } else if ev.prev_hash_hex != entries[i - 1].self_hash_hex {
            eprintln!("FAIL chain break before seq {}", ev.seq);
            return ExitCode::from(1);
        }
    }
    println!("Chain integrity: OK");
    ExitCode::SUCCESS
}

fn cmd_ledger_show(file: &Path, last: Option<usize>) -> ExitCode {
    let entries = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load {}: {e}", file.display());
            return ExitCode::from(1);
        }
    };
    let slice: &[LedgerEntry] = match last {
        Some(n) if n < entries.len() => &entries[entries.len() - n..],
        _ => &entries,
    };
    for ev in slice {
        println!(
            "[{:>6}] {} signer={} {}",
            ev.seq,
            ev.timestamp.to_rfc3339(),
            ev.signer,
            summarize_event(&ev.event)
        );
    }
    ExitCode::SUCCESS
}

fn cmd_export_population(src: &Path, dst: &Path) -> ExitCode {
    let c = match ckpt::read(src) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("read {}: {e}", src.display());
            return ExitCode::from(1);
        }
    };
    let payload = match serde_json::to_vec_pretty(c.population.units()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("serialize: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(parent) = dst.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("mkdir {}: {e}", parent.display());
                return ExitCode::from(1);
            }
        }
    }
    if let Err(e) = fs::write(dst, &payload) {
        eprintln!("write {}: {e}", dst.display());
        return ExitCode::from(1);
    }
    println!(
        "Exported {} units ({} bytes) to {}",
        c.population.len(),
        payload.len(),
        dst.display()
    );
    ExitCode::SUCCESS
}

fn cmd_compare_runs(a_path: &Path, b_path: &Path) -> ExitCode {
    let a = match ckpt::read(a_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("read {}: {e}", a_path.display());
            return ExitCode::from(1);
        }
    };
    let b = match ckpt::read(b_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("read {}: {e}", b_path.display());
            return ExitCode::from(1);
        }
    };
    let (h_f, h_a, h_b) = ("FIELD", "A", "B");
    println!("{h_f:<24} {h_a:<40} {h_b}");
    println!("{:<24} {:<40} {}", "run_id", a.run_id, b.run_id);
    println!("{:<24} {:<40} {}", "generation", a.generation, b.generation);
    println!(
        "{:<24} {:<40} {}",
        "population_size",
        a.population.len(),
        b.population.len()
    );
    println!(
        "{:<24} {:<40} {}",
        "mean_fitness",
        format_f64(a.population.mean_fitness()),
        format_f64(b.population.mean_fitness())
    );
    let a_best = a
        .population
        .best()
        .map(|u| u.fitness_value())
        .unwrap_or(0.0);
    let b_best = b
        .population
        .best()
        .map(|u| u.fitness_value())
        .unwrap_or(0.0);
    println!(
        "{:<24} {:<40} {}",
        "best_fitness",
        format_f64(a_best),
        format_f64(b_best)
    );
    println!(
        "{:<24} {:<40} {}",
        "config_hash",
        short_hash(&a.config_hash_hex),
        short_hash(&b.config_hash_hex)
    );
    println!(
        "{:<24} {:<40} {}",
        "ledger_tip",
        short_hash(&a.ledger_tip_hash_hex),
        short_hash(&b.ledger_tip_hash_hex)
    );
    if a.config_hash_hex != b.config_hash_hex {
        println!();
        println!("NOTE: config hashes differ — fitness comparison may not be apples-to-apples.");
    }
    ExitCode::SUCCESS
}

// =============================================================================
// Helpers
// =============================================================================

fn print_checkpoint_summary(c: &Checkpoint) {
    println!("Run ID:            {}", c.run_id);
    println!("Generation:        {}", c.generation);
    println!("Population size:   {}", c.population.len());
    println!("Config hash:       {}", c.config_hash_hex);
    println!("Ledger tip hash:   {}", c.ledger_tip_hash_hex);
    println!(
        "Mean fitness:      {}",
        format_f64(c.population.mean_fitness())
    );
    if let Some(best) = c.population.best() {
        println!(
            "Best unit:         {} (fitness {})",
            best.id,
            format_f64(best.fitness_value())
        );
        println!("  task_prompt:     {}", truncate(&best.task_prompt, 100));
        println!(
            "  mutation_prompt: {}",
            truncate(&best.mutation_prompt, 100)
        );
    } else {
        println!("Best unit:         (none evaluated)");
    }

    // Operator histogram across the population.
    let mut op_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut seeds = 0u64;
    for u in c.population.iter() {
        match &u.operator_born {
            Some(op) => *op_counts.entry(operator_label(op).to_string()).or_insert(0) += 1,
            None => seeds += 1,
        }
    }
    println!();
    println!("Provenance:");
    println!("  seeds:           {seeds}");
    if op_counts.is_empty() {
        println!("  (no mutation-born units)");
    } else {
        for (k, v) in &op_counts {
            println!("  {k:<20} {v}");
        }
    }

    // Fitness distribution — simple bucket summary.
    let mut evaluated = 0usize;
    let mut fit_sum = 0.0f64;
    let mut fit_min = f64::INFINITY;
    let mut fit_max = f64::NEG_INFINITY;
    for u in c.population.iter() {
        if let Some(f) = &u.fitness {
            evaluated += 1;
            fit_sum += f.aggregate;
            if f.aggregate < fit_min {
                fit_min = f.aggregate;
            }
            if f.aggregate > fit_max {
                fit_max = f.aggregate;
            }
        }
    }
    println!();
    println!("Fitness:");
    println!("  evaluated:       {evaluated}/{}", c.population.len());
    if evaluated > 0 {
        println!(
            "  min / mean / max: {} / {} / {}",
            format_f64(fit_min),
            format_f64(fit_sum / evaluated as f64),
            format_f64(fit_max)
        );
    }

    // Lineage DAG sizing.
    println!();
    println!("Lineage DAG:");
    println!("  nodes (parents): {}", c.lineage.parents.len());
    println!("  edges:           {}", count_edges(&c.lineage));
}

fn count_edges(dag: &ai_assistant::prompt_breeder::LineageDag) -> usize {
    dag.parents.values().map(|v| v.len()).sum()
}

fn operator_label(op: &MutationOperator) -> &'static str {
    match op {
        MutationOperator::ZeroOrder => "ZeroOrder",
        MutationOperator::FirstOrder => "FirstOrder",
        MutationOperator::Eda => "Eda",
        MutationOperator::EdaRankAndIndex => "EdaRankAndIndex",
        MutationOperator::LineageBased => "LineageBased",
        MutationOperator::HyperMutationZeroOrder => "HyperMutationZeroOrder",
        MutationOperator::HyperMutationFirstOrder => "HyperMutationFirstOrder",
        MutationOperator::Lamarckian => "Lamarckian",
        MutationOperator::PromptCrossover => "PromptCrossover",
        _ => "Unknown",
    }
}

fn summarize_event(ev: &BreederEvent) -> String {
    match ev {
        BreederEvent::RunStarted {
            run_id,
            fingerprint,
            ..
        } => format!("RunStarted run_id={run_id} fp={fingerprint}"),
        BreederEvent::SeedBootstrapped { n, source } => {
            format!("SeedBootstrapped n={n} source={source}")
        }
        BreederEvent::SeedInserted { unit_id, .. } => format!("SeedInserted unit_id={unit_id}"),
        BreederEvent::GenerationStarted { generation } => {
            format!("GenerationStarted generation={generation}")
        }
        BreederEvent::MutationApplied {
            parent_id,
            child_id,
            operator,
        } => format!(
            "MutationApplied op={} parent={parent_id} child={child_id}",
            operator_label(operator)
        ),
        BreederEvent::MutationRejected {
            parent_id,
            operator,
            reason,
        } => format!(
            "MutationRejected op={} parent={parent_id} reason={reason:?}",
            operator_label(operator)
        ),
        BreederEvent::FitnessEvaluated {
            unit_id,
            score,
            cached,
        } => format!(
            "FitnessEvaluated unit={unit_id} agg={} cached={cached}",
            format_f64(score.aggregate)
        ),
        BreederEvent::SelectionPerformed {
            strategy,
            survivors,
        } => format!(
            "SelectionPerformed strategy={strategy:?} survivors={}",
            survivors.len()
        ),
        BreederEvent::DiversityComputed { generation, score } => {
            format!(
                "DiversityComputed gen={generation} score={}",
                format_f64(*score)
            )
        }
        BreederEvent::EvalAugmented {
            n_added,
            augmenter_kind,
        } => format!("EvalAugmented n_added={n_added} kind={augmenter_kind}"),
        BreederEvent::LineageNarrated { unit_id, .. } => {
            format!("LineageNarrated unit_id={unit_id}")
        }
        BreederEvent::SmoothingSampled { unit_id, samples } => {
            format!("SmoothingSampled unit_id={unit_id} samples={samples}")
        }
        BreederEvent::BudgetExhausted { kind, value } => {
            format!("BudgetExhausted kind={kind:?} value={}", format_f64(*value))
        }
        BreederEvent::CheckpointWritten { path, tip_hash_hex } => format!(
            "CheckpointWritten path={path} tip={}",
            short_hash(tip_hash_hex)
        ),
        BreederEvent::FreezeChanged { frozen } => format!("FreezeChanged frozen={frozen}"),
        BreederEvent::SafetyFilterApplied { filter_kind } => {
            format!("SafetyFilterApplied filter_kind={filter_kind}")
        }
        BreederEvent::RunCompleted {
            run_id,
            best_id,
            generations,
        } => format!("RunCompleted run_id={run_id} best={best_id} generations={generations}"),
        BreederEvent::RunAborted { run_id, reason } => {
            format!("RunAborted run_id={run_id} reason={reason:?}")
        }
        _ => "UnknownEvent".to_string(),
    }
}

fn load_ledger(file: &Path) -> Result<Vec<LedgerEntry>, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: LedgerEntry =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        out.push(ev);
    }
    Ok(out)
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn short_hash(h: &str) -> String {
    if h.len() > 12 {
        h[..12].to_string()
    } else {
        h.to_string()
    }
}

fn format_f64(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    format!("{x:.4}")
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

// Keep `Unit` in the import list used for live type-checking; not all
// subcommands touch units directly but exports/compares need them.
#[allow(dead_code)]
fn _type_check_unit(_: &Unit) {}

fn print_help() {
    println!(
        "ai_breeder — PromptBreeder V97 auditor (read-only)\n\n\
USAGE:\n  \
ai_breeder list-runs <DIR>                        # list *.ckpt files in DIR\n  \
ai_breeder show-run <CKPT_FILE>                   # summary of a checkpoint\n  \
ai_breeder ledger-verify <LEDGER_JSONL>           # verify hash chain\n  \
ai_breeder ledger-show <LEDGER_JSONL> [--last N]  # print events\n  \
ai_breeder export-population <CKPT> <OUT_JSON>    # dump units as pretty JSON\n  \
ai_breeder compare-runs <CKPT_A> <CKPT_B>         # side-by-side metrics\n\n\
Read-only. Requires --features prompt-breeder at build time."
    );
}
