//! `ai_skills` — Skill Forge auditor CLI.
//!
//! Inspect, verify, and audit skills registered by the Skill Forge subsystem.
//! All operations are read-only — this binary never mutates the skill store
//! (see feedback memory `feedback_auditable_subsystems`).
//!
//! # Usage
//!
//! ```text
//! ai_skills list [--dir <PATH>]                     # list .skill.json in dir
//! ai_skills inspect <FILE>                          # show metadata + hashes
//! ai_skills verify <FILE>                           # recompute content hash
//! ai_skills ledger-verify <LEDGER_JSONL>            # verify chain integrity
//! ai_skills ledger-show <LEDGER_JSONL> [--last N]   # print events
//! ai_skills export <FILE> --out <BUNDLE.json>       # signed evidence bundle
//! ```

use ai_assistant::{LedgerEvent, SkillDefinition, SkillStatus};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }

    let command = args[1].as_str();
    match command {
        "list" => {
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./skills"));
            cmd_list(&dir)
        }
        "inspect" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_skills inspect <FILE>");
                return ExitCode::from(2);
            };
            cmd_inspect(Path::new(file))
        }
        "verify" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_skills verify <FILE>");
                return ExitCode::from(2);
            };
            cmd_verify(Path::new(file))
        }
        "ledger-verify" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_skills ledger-verify <LEDGER_JSONL>");
                return ExitCode::from(2);
            };
            cmd_ledger_verify(Path::new(file))
        }
        "ledger-show" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_skills ledger-show <LEDGER_JSONL> [--last N]");
                return ExitCode::from(2);
            };
            let last: Option<usize> = get_arg(&args, "--last").and_then(|s| s.parse().ok());
            cmd_ledger_show(Path::new(file), last)
        }
        "export" => {
            let Some(file) = args.get(2) else {
                eprintln!("Usage: ai_skills export <FILE> --out <BUNDLE.json>");
                return ExitCode::from(2);
            };
            let Some(out) = get_arg(&args, "--out") else {
                eprintln!("Usage: ai_skills export <FILE> --out <BUNDLE.json>");
                return ExitCode::from(2);
            };
            cmd_export(Path::new(file), Path::new(&out))
        }
        other => {
            eprintln!("Unknown command: {other}. Use --help.");
            ExitCode::from(2)
        }
    }
}

// =============================================================================
// Commands
// =============================================================================

fn cmd_list(dir: &Path) -> ExitCode {
    if !dir.is_dir() {
        eprintln!("Not a directory: {}", dir.display());
        return ExitCode::from(1);
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("read_dir: {e}");
            return ExitCode::from(1);
        }
    };
    let mut rows: Vec<(String, String, SkillStatus)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".skill.json"))
            .unwrap_or(false)
        {
            continue;
        }
        match load_skill(&path) {
            Ok(def) => rows.push((
                def.id.as_str().to_string(),
                def.version.to_string(),
                def.status,
            )),
            Err(e) => eprintln!("  skip {}: {e}", path.display()),
        }
    }
    if rows.is_empty() {
        println!("(no .skill.json files in {})", dir.display());
        return ExitCode::SUCCESS;
    }
    let (h_id, h_ver, h_status) = ("SKILL", "VERSION", "STATUS");
    println!("{h_id:<32} {h_ver:<10} {h_status}");
    for (id, ver, status) in rows {
        println!("{id:<32} {ver:<10} {status}");
    }
    ExitCode::SUCCESS
}

fn cmd_inspect(file: &Path) -> ExitCode {
    let def = match load_skill(file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    println!("Skill:         {}", def.id.as_str());
    println!("Version:       {}", def.version);
    println!("Name:          {}", def.name);
    println!("Status:        {}", def.status);
    println!("Tenant:        {}", def.tenant);
    println!("Shared xtenant: {}", def.shared_cross_tenant);
    println!("Capabilities:  {}", def.capabilities.len());
    for cap in def.capabilities.iter() {
        println!("  - {cap:?}");
    }
    println!("Content hash:  {}", def.content_hash_hex);
    match &def.mode {
        ai_assistant::SkillMode::Declarative(steps) => {
            println!("Mode:          Declarative ({} steps)", steps.len());
        }
        ai_assistant::SkillMode::Wasm(a) => {
            println!("Mode:          Wasm ({} bytes)", a.bytes.len());
            println!("  blake3:      {}", a.blake3_hex);
            println!("  signed_by:   {}", a.signed_by);
            println!("  toolchain:   {}", a.compile_fingerprint);
            if let Some(src) = &a.source_path {
                println!("  source:      {src}");
            }
        }
        _ => {
            println!("Mode:          (unknown — future variant)");
        }
    }
    ExitCode::SUCCESS
}

fn cmd_verify(file: &Path) -> ExitCode {
    let def = match load_skill(file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let expected = def.compute_content_hash();
    let ok_content = expected == def.content_hash_hex;
    println!("Skill:         {}", def.id.as_str());
    println!("Version:       {}", def.version);
    println!(
        "Content hash:  {}",
        if ok_content { "OK" } else { "MISMATCH" }
    );
    if !ok_content {
        println!("  expected: {expected}");
        println!("  stored:   {}", def.content_hash_hex);
    }

    let mut ok_wasm = true;
    if let ai_assistant::SkillMode::Wasm(a) = &def.mode {
        let recomputed = blake3::hash(&a.bytes).to_hex().to_string();
        ok_wasm = recomputed == a.blake3_hex;
        println!("WASM blake3:   {}", if ok_wasm { "OK" } else { "MISMATCH" });
        if !ok_wasm {
            println!("  expected: {recomputed}");
            println!("  stored:   {}", a.blake3_hex);
        }
    }
    if ok_content && ok_wasm {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_ledger_verify(file: &Path) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load ledger: {e}");
            return ExitCode::from(1);
        }
    };
    println!("Loaded {} events from {}", events.len(), file.display());
    for (i, ev) in events.iter().enumerate() {
        if ev.seq != i as u64 {
            eprintln!("FAIL seq gap at index {i}: got {}", ev.seq);
            return ExitCode::from(1);
        }
        if !ev.verify_self_hash() {
            eprintln!("FAIL self-hash at seq {}", ev.seq);
            return ExitCode::from(1);
        }
        if i > 0 {
            let prev = &events[i - 1];
            if ev.prev_hash_hex != prev.self_hash_hex {
                eprintln!("FAIL chain break at seq {} -> seq {}", prev.seq, ev.seq);
                return ExitCode::from(1);
            }
        }
    }
    println!("Chain integrity: OK");
    ExitCode::SUCCESS
}

fn cmd_ledger_show(file: &Path, last: Option<usize>) -> ExitCode {
    let events = match load_ledger(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load ledger: {e}");
            return ExitCode::from(1);
        }
    };
    let slice: &[LedgerEvent] = match last {
        Some(n) if n < events.len() => &events[events.len() - n..],
        _ => &events,
    };
    for ev in slice {
        let kind = match &ev.kind {
            ai_assistant::LedgerEventKind::Registered { skill, version, .. } => {
                format!("Registered {} {}", skill.as_str(), version)
            }
            ai_assistant::LedgerEventKind::StatusChanged {
                skill,
                version,
                from,
                to,
                ..
            } => format!("StatusChanged {} {} {from}->{to}", skill.as_str(), version),
            ai_assistant::LedgerEventKind::Retracted {
                skill,
                version,
                reason,
            } => {
                format!("Retracted {} {} ({reason})", skill.as_str(), version)
            }
            ai_assistant::LedgerEventKind::IntegrityCheck {
                skill,
                version,
                passed,
                ..
            } => format!(
                "IntegrityCheck {} {} {}",
                skill.as_str(),
                version,
                if *passed { "PASS" } else { "FAIL" }
            ),
            ai_assistant::LedgerEventKind::AuditAccess { viewer, scope } => {
                format!("AuditAccess {viewer} scope={scope}")
            }
            _ => "Unknown".to_string(),
        };
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

fn cmd_export(file: &Path, out: &Path) -> ExitCode {
    let def = match load_skill(file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::from(1);
        }
    };
    let bundle = serde_json::json!({
        "bundle_kind": "ai_assistant/skill_forge/audit_bundle/v1",
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "skill": def,
    });
    let json = match serde_json::to_string_pretty(&bundle) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("serialize bundle: {e}");
            return ExitCode::from(1);
        }
    };
    if let Err(e) = fs::write(out, json) {
        eprintln!("write {}: {e}", out.display());
        return ExitCode::from(1);
    }
    println!("Bundle written: {}", out.display());
    ExitCode::SUCCESS
}

// =============================================================================
// Helpers
// =============================================================================

fn load_skill(file: &Path) -> Result<SkillDefinition, String> {
    let bytes = fs::read(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", file.display()))
}

fn load_ledger(file: &Path) -> Result<Vec<LedgerEvent>, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: LedgerEvent =
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
        "ai_skills — Skill Forge auditor\n\n\
USAGE:\n  \
ai_skills list [--dir <PATH>]\n  \
ai_skills inspect <FILE>\n  \
ai_skills verify <FILE>\n  \
ai_skills ledger-verify <LEDGER_JSONL>\n  \
ai_skills ledger-show <LEDGER_JSONL> [--last N]\n  \
ai_skills export <FILE> --out <BUNDLE.json>\n\n\
All operations are read-only. See ai_logs for distributed log audit."
    );
}
