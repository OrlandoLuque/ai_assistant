// `ai_corrections` — audit the self-correction subsystem: the JSONL run ledger
// and the review queue of work the engine could not make correct.
//
// The ledger records what HAPPENED; the quarantine records what is OUTSTANDING.
// They answer different questions, so this tool reads both.
//
// Verbs:
//   pending  [--dir D]           List work awaiting human review.
//   show     <ID> [--dir D]      Evidence + artifact for one queued item.
//   resolve  <ID> [--dir D]      Move an item out of the queue (kept, not deleted).
//   log      [--file F] [--tail N]  Read the run ledger.
//
// Exit codes: 0 clean, 1 items awaiting review (so it can gate a pipeline),
// 2 usage error.

#![cfg(feature = "self-correction")]

use std::path::PathBuf;
use std::process::ExitCode;

use ai_assistant::self_correction::ledger::CorrectionLedger;
use ai_assistant::self_correction::Quarantine;

const DEFAULT_DIR: &str = ".ai_assistant/corrections";
const DEFAULT_LEDGER: &str = ".ai_assistant/corrections/ledger.jsonl";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "pending" => cmd_pending(&args[1..]),
        "show" => cmd_show(&args[1..]),
        "resolve" => cmd_resolve(&args[1..]),
        "log" => cmd_log(&args[1..]),
        "--help" | "-h" => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("ai_corrections: unknown verb '{other}'\n\n{}", usage());
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "ai_corrections <verb> [args]\n\
     \n\
     Verbs:\n\
       pending [--dir D]              Work awaiting review (exit 1 if any)\n\
       show    <ID> [--dir D]         Evidence and artifact for one item\n\
       resolve <ID> [--dir D]         Move an item out of the queue\n\
       log     [--file F] [--tail N]  Read the run ledger\n\
       --help, -h                     Show this message\n\
     \n\
     Default --dir:  .ai_assistant/corrections\n\
     Default --file: .ai_assistant/corrections/ledger.jsonl\n\
     \n\
     A quarantined artifact is work the engine could NOT verify. It is kept\n\
     exactly as produced and never fed back into a pipeline: the point is that\n\
     unfinished work must not be able to pass as finished."
}

/// `--dir D`, else the default.
fn dir_arg(args: &[String]) -> PathBuf {
    flag_value(args, "--dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR))
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// First argument that is not a flag or a flag's value.
fn positional(args: &[String]) -> Option<String> {
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a.starts_with("--") {
            skip_next = true;
            continue;
        }
        return Some(a.clone());
    }
    None
}

fn cmd_pending(args: &[String]) -> ExitCode {
    let dir = dir_arg(args);
    let q = match Quarantine::open(&dir) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("ai_corrections: cannot open {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    let pending = match q.pending() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ai_corrections: {e}");
            return ExitCode::from(2);
        }
    };
    if pending.is_empty() {
        println!("Nothing awaiting review in {}.", dir.display());
        return ExitCode::SUCCESS;
    }
    println!(
        "{} item(s) awaiting review in {}:\n",
        pending.len(),
        dir.display()
    );
    for rec in &pending {
        println!("  {}", rec.summary());
    }
    println!("\nInspect one with: ai_corrections show <ID>");
    // Non-zero so this can gate a pipeline: outstanding unverified work is
    // exactly the thing that should stop a release.
    ExitCode::from(1)
}

fn cmd_show(args: &[String]) -> ExitCode {
    let Some(id) = positional(args) else {
        eprintln!("ai_corrections show: missing <ID>\n\n{}", usage());
        return ExitCode::from(2);
    };
    let dir = dir_arg(args);
    let q = match Quarantine::open(&dir) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("ai_corrections: cannot open {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    let all = q
        .pending()
        .unwrap_or_default()
        .into_iter()
        .chain(q.resolved().unwrap_or_default())
        .collect::<Vec<_>>();
    let Some(rec) = all.into_iter().find(|r| r.id == id) else {
        eprintln!("ai_corrections: no item with id '{id}'");
        return ExitCode::from(2);
    };

    println!("id:          {}", rec.id);
    println!("task:        {}", rec.evidence.task_name);
    println!("stopped:     {}", rec.evidence.stop_reason);
    println!("attempts:    {}", rec.evidence.attempts.len());
    println!("tokens:      {}", rec.evidence.total_tokens);
    println!("cost (USD):  {:.4}", rec.evidence.total_cost_usd);
    println!("elapsed(ms): {}", rec.evidence.total_elapsed_ms);

    println!("\n── why it was not accepted ──");
    for a in &rec.evidence.attempts {
        println!(
            "  attempt {}: quality {:.2}",
            a.attempt_num, a.quality_score
        );
        for issue in &a.issues {
            for line in issue.lines() {
                println!("      {line}");
            }
        }
    }

    println!("\n── the artifact, as produced ──");
    match rec.read_artifact() {
        Ok(text) => println!("{text}"),
        Err(e) => println!("  <could not read {}: {e}>", rec.artifact_path.display()),
    }
    ExitCode::SUCCESS
}

fn cmd_resolve(args: &[String]) -> ExitCode {
    let Some(id) = positional(args) else {
        eprintln!("ai_corrections resolve: missing <ID>\n\n{}", usage());
        return ExitCode::from(2);
    };
    let dir = dir_arg(args);
    let q = match Quarantine::open(&dir) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("ai_corrections: cannot open {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    match q.resolve(&id) {
        Ok(()) => {
            println!("Resolved '{id}' — moved to resolved/, not deleted.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ai_corrections: {e}");
            ExitCode::from(2)
        }
    }
}

fn cmd_log(args: &[String]) -> ExitCode {
    let file = flag_value(args, "--file").unwrap_or_else(|| DEFAULT_LEDGER.to_string());
    let tail: usize = flag_value(args, "--tail")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    let ledger = match CorrectionLedger::open(&file) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ai_corrections: cannot open {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let (entries, skipped) = match ledger.read_all() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ai_corrections: {e}");
            return ExitCode::from(2);
        }
    };
    if entries.is_empty() {
        println!("No runs recorded in {file}.");
        return ExitCode::SUCCESS;
    }
    let start = entries.len().saturating_sub(tail);
    println!(
        "{} run(s) in {file}; showing last {}:\n",
        entries.len(),
        entries.len() - start
    );
    for e in &entries[start..] {
        println!(
            "  {}  {:<22} {:<10} {} attempt(s)  {} tok  {:.4} USD  ({})",
            e.timestamp,
            e.task_name,
            if e.succeeded { "ok" } else { "NEEDS-FIX" },
            e.attempts.len(),
            e.total_tokens,
            e.total_cost_usd,
            e.stop_reason
        );
    }
    if skipped > 0 {
        // Surfaced rather than swallowed: a silently skipped line is a hole in
        // an audit trail, which is worse than a noisy one.
        println!("\n{skipped} malformed line(s) skipped.");
    }
    ExitCode::SUCCESS
}
