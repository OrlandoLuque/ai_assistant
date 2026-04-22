//! `ai_prompt_synth` — Prompt Synthesis auditor CLI.
//!
//! Inspect the fragment-synthesis ledger produced by `FragmentLedger`.
//! Read-only by design (see `feedback_auditable_subsystems`).
//!
//! # Usage
//!
//! ```text
//! ai_prompt_synth ledger-show <LEDGER_JSONL> [--last N]
//! ai_prompt_synth ledger-verify <LEDGER_JSONL>
//! ai_prompt_synth arms-summary <LEDGER_JSONL>     # aggregate arm stats
//! ```

use ai_assistant::{FragmentEvent, FragmentEventKind};
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
                eprintln!("Usage: ai_prompt_synth ledger-show <LEDGER_JSONL> [--last N]");
                return ExitCode::from(2);
            };
            let last: Option<usize> = get_arg(&args, "--last").and_then(|s| s.parse().ok());
            cmd_ledger_show(Path::new(file), last)
        }
        "ledger-verify" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_prompt_synth ledger-verify <LEDGER_JSONL>");
                return ExitCode::from(2);
            };
            cmd_ledger_verify(Path::new(file))
        }
        "arms-summary" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_prompt_synth arms-summary <LEDGER_JSONL>");
                return ExitCode::from(2);
            };
            cmd_arms_summary(Path::new(file))
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
    let slice: &[FragmentEvent] = match last {
        Some(n) if n < events.len() => &events[events.len() - n..],
        _ => &events,
    };
    for ev in slice {
        let kind = summarize_kind(&ev.kind);
        println!(
            "[{:>6}] {} signer={} {}",
            ev.seq,
            ev.timestamp.to_rfc3339(),
            ev.signer,
            kind
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

fn cmd_arms_summary(file: &Path) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let mut selections: BTreeMap<String, u64> = BTreeMap::new();
    let mut rewards: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    let mut retirements: u64 = 0;
    let mut creations: u64 = 0;
    for ev in &events {
        match &ev.kind {
            FragmentEventKind::ArmCreated { arm, .. } => {
                creations += 1;
                selections.entry(arm.as_str().to_string()).or_insert(0);
            }
            FragmentEventKind::ArmSelected { arm, .. } => {
                *selections.entry(arm.as_str().to_string()).or_insert(0) += 1;
            }
            FragmentEventKind::RewardRecorded { arm, reward, .. } => {
                let entry = rewards.entry(arm.as_str().to_string()).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += *reward as f64;
            }
            FragmentEventKind::ArmRetired { .. } => retirements += 1,
            _ => {}
        }
    }
    println!("Total events:  {}", events.len());
    println!("Creations:     {creations}");
    println!("Retirements:   {retirements}");
    println!();
    let (h_arm, h_sel, h_samples, h_mean) = ("ARM", "SELECTED", "SAMPLES", "MEAN_REWARD");
    println!("{h_arm:<40} {h_sel:<10} {h_samples:<10} {h_mean}");
    for (arm, sel) in &selections {
        let (samples, sum) = rewards.get(arm).copied().unwrap_or((0, 0.0));
        let mean = if samples > 0 {
            sum / samples as f64
        } else {
            0.0
        };
        println!("{arm:<40} {sel:<10} {samples:<10} {mean:.3}");
    }
    ExitCode::SUCCESS
}

fn summarize_kind(kind: &FragmentEventKind) -> String {
    match kind {
        FragmentEventKind::ArmCreated {
            cluster,
            arm,
            origin,
            ..
        } => format!("ArmCreated {cluster} {arm} origin={origin}"),
        FragmentEventKind::ArmSelected {
            cluster,
            arm,
            reason,
            score,
            ..
        } => format!("ArmSelected {cluster} {arm} reason={reason:?} score={score:.3}"),
        FragmentEventKind::RewardRecorded {
            cluster,
            arm,
            reward,
            ..
        } => format!("RewardRecorded {cluster} {arm} reward={reward:.3}"),
        FragmentEventKind::ArmRetired {
            cluster,
            arm,
            reason,
            ..
        } => {
            format!("ArmRetired {cluster} {arm} reason={reason}")
        }
        FragmentEventKind::ClusterResized {
            before,
            after,
            removed,
        } => {
            format!("ClusterResized {before}->{after} removed={removed}")
        }
        FragmentEventKind::FreezeChanged { frozen } => format!("FreezeChanged {frozen}"),
        _ => "Unknown".to_string(),
    }
}

fn load_ledger(file: &Path) -> Result<Vec<FragmentEvent>, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: FragmentEvent =
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
        "ai_prompt_synth — Fragment Synthesis auditor\n\n\
USAGE:\n  \
ai_prompt_synth ledger-show <LEDGER_JSONL> [--last N]\n  \
ai_prompt_synth ledger-verify <LEDGER_JSONL>\n  \
ai_prompt_synth arms-summary <LEDGER_JSONL>\n\n\
Read-only. Requires --features prompt-synthesis at build time."
    );
}
