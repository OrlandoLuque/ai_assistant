//! `ai_feedback` — Feedback-Loop auditor CLI.
//!
//! Read-only by design (see `feedback_auditable_subsystems`). Inspects the
//! `DispatchLedger` / `RetractionLedger` JSONL files produced by the
//! `FeedbackDispatcher`.
//!
//! # Usage
//!
//! ```text
//! ai_feedback ledger-show <DISPATCH_JSONL> [--last N]
//! ai_feedback ledger-verify <DISPATCH_JSONL>
//! ai_feedback retractions <RETRACTION_JSONL>
//! ai_feedback stats <DISPATCH_JSONL>
//! ```

use ai_assistant::{DispatchEvent, DispatchEventKind};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    let command = args[1].as_str();
    match command {
        "ledger-show" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_feedback ledger-show <DISPATCH_JSONL> [--last N]");
                return ExitCode::from(2);
            };
            let last: Option<usize> = get_arg(&args, "--last").and_then(|s| s.parse().ok());
            cmd_ledger_show(Path::new(file), last)
        }
        "ledger-verify" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_feedback ledger-verify <DISPATCH_JSONL>");
                return ExitCode::from(2);
            };
            cmd_ledger_verify(Path::new(file))
        }
        "retractions" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_feedback retractions <RETRACTION_JSONL>");
                return ExitCode::from(2);
            };
            cmd_retractions(Path::new(file))
        }
        "stats" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_feedback stats <DISPATCH_JSONL>");
                return ExitCode::from(2);
            };
            cmd_stats(Path::new(file))
        }
        other => {
            eprintln!("Unknown command: {other}. Use --help.");
            ExitCode::from(2)
        }
    }
}

fn cmd_ledger_show(file: &Path, last: Option<usize>) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let slice: &[DispatchEvent] = match last {
        Some(n) if n < events.len() => &events[events.len() - n..],
        _ => &events,
    };
    for ev in slice {
        println!(
            "[{:>6}] {} signer={} {}",
            ev.seq,
            ev.timestamp.to_rfc3339(),
            ev.signer,
            summarize_kind(&ev.kind)
        );
    }
    ExitCode::SUCCESS
}

fn cmd_ledger_verify(file: &Path) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    println!("Loaded {} events from {}", events.len(), file.display());
    for (i, ev) in events.iter().enumerate() {
        if ev.seq != i as u64 {
            eprintln!("FAIL seq gap at {i}: got {}", ev.seq);
            return ExitCode::from(1);
        }
        if !ev.verify_self_hash() {
            eprintln!("FAIL self-hash at seq {}", ev.seq);
            return ExitCode::from(1);
        }
        if i > 0 && ev.prev_hash_hex != events[i - 1].self_hash_hex {
            eprintln!("FAIL chain break before seq {}", ev.seq);
            return ExitCode::from(1);
        }
    }
    println!("Chain integrity: OK");
    ExitCode::SUCCESS
}

fn cmd_retractions(file: &Path) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let mut requested = 0u64;
    let mut propagated = 0u64;
    for ev in &events {
        match &ev.kind {
            DispatchEventKind::RetractionRequested { trajectory, reason } => {
                requested += 1;
                println!(
                    "[{:>6}] REQUEST {} {} reason={}",
                    ev.seq,
                    ev.timestamp.to_rfc3339(),
                    trajectory.as_str(),
                    reason
                );
            }
            DispatchEventKind::RetractionPropagated { trajectory, sink } => {
                propagated += 1;
                println!(
                    "[{:>6}] PROPAGATE {} {} sink={}",
                    ev.seq,
                    ev.timestamp.to_rfc3339(),
                    trajectory.as_str(),
                    sink
                );
            }
            _ => {}
        }
    }
    println!();
    println!("Retractions: requested={requested} propagated={propagated}");
    ExitCode::SUCCESS
}

fn cmd_stats(file: &Path) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let mut received = 0u64;
    let mut dispatched = 0u64;
    let mut failed = 0u64;
    let mut dropped_by_reason: BTreeMap<String, u64> = BTreeMap::new();
    let mut per_sink_dispatched: BTreeMap<String, u64> = BTreeMap::new();
    let mut per_sink_failed: BTreeMap<String, u64> = BTreeMap::new();
    let mut freeze_events = 0u64;
    for ev in &events {
        match &ev.kind {
            DispatchEventKind::TrajectoryReceived { .. } => received += 1,
            DispatchEventKind::SinkDispatched { sink, .. } => {
                dispatched += 1;
                *per_sink_dispatched.entry(sink.clone()).or_insert(0) += 1;
            }
            DispatchEventKind::SinkFailed { sink, .. } => {
                failed += 1;
                *per_sink_failed.entry(sink.clone()).or_insert(0) += 1;
            }
            DispatchEventKind::TrajectoryDropped { reason, .. } => {
                *dropped_by_reason.entry(reason.clone()).or_insert(0) += 1;
            }
            DispatchEventKind::FreezeChanged { .. } => freeze_events += 1,
            _ => {}
        }
    }
    println!("Total events:      {}", events.len());
    println!("Received:          {received}");
    println!("Dispatched (sink): {dispatched}");
    println!("Failed (sink):     {failed}");
    println!("Freeze toggles:    {freeze_events}");
    println!();
    println!("Drops by reason:");
    if dropped_by_reason.is_empty() {
        println!("  (none)");
    } else {
        for (k, v) in &dropped_by_reason {
            println!("  {k:<24} {v}");
        }
    }
    println!();
    println!("Per-sink dispatch:");
    let all_sinks: std::collections::BTreeSet<String> = per_sink_dispatched
        .keys()
        .chain(per_sink_failed.keys())
        .cloned()
        .collect();
    if all_sinks.is_empty() {
        println!("  (none)");
    } else {
        let (h_sink, h_disp, h_fail) = ("SINK", "DISPATCHED", "FAILED");
        println!("  {h_sink:<32} {h_disp:<12} {h_fail}");
        for sink in &all_sinks {
            let d = per_sink_dispatched.get(sink).copied().unwrap_or(0);
            let f = per_sink_failed.get(sink).copied().unwrap_or(0);
            println!("  {sink:<32} {d:<12} {f}");
        }
    }
    ExitCode::SUCCESS
}

fn summarize_kind(kind: &DispatchEventKind) -> String {
    match kind {
        DispatchEventKind::TrajectoryReceived {
            trajectory,
            principal,
            privacy,
            outcome,
        } => format!(
            "TrajectoryReceived {} principal={principal} privacy={privacy} outcome={outcome}",
            trajectory.as_str()
        ),
        DispatchEventKind::SinkDispatched { trajectory, sink } => {
            format!("SinkDispatched {} sink={sink}", trajectory.as_str())
        }
        DispatchEventKind::SinkFailed {
            trajectory,
            sink,
            reason,
        } => format!(
            "SinkFailed {} sink={sink} reason={reason}",
            trajectory.as_str()
        ),
        DispatchEventKind::TrajectoryDropped { trajectory, reason } => {
            format!("TrajectoryDropped {} reason={reason}", trajectory.as_str())
        }
        DispatchEventKind::RetractionRequested { trajectory, reason } => format!(
            "RetractionRequested {} reason={reason}",
            trajectory.as_str()
        ),
        DispatchEventKind::RetractionPropagated { trajectory, sink } => {
            format!("RetractionPropagated {} sink={sink}", trajectory.as_str())
        }
        DispatchEventKind::FreezeChanged { frozen } => {
            format!("FreezeChanged {frozen}")
        }
        _ => "Unknown".to_string(),
    }
}

fn load_ledger(file: &Path) -> Result<Vec<DispatchEvent>, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: DispatchEvent =
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

fn print_help() {
    println!(
        "ai_feedback — Feedback Loop auditor\n\n\
USAGE:\n  \
ai_feedback ledger-show <DISPATCH_JSONL> [--last N]\n  \
ai_feedback ledger-verify <DISPATCH_JSONL>\n  \
ai_feedback retractions <RETRACTION_JSONL>\n  \
ai_feedback stats <DISPATCH_JSONL>\n\n\
Read-only. Requires --features feedback-loop at build time."
    );
}
