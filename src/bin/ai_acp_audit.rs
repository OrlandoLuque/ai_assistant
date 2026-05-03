// `ai_acp_audit` — read SLO log files emitted by `ai_acp serve` and check
// SLO compliance. Exit code: 0 if all targets met, 1 otherwise.
//
// Verbs:
//   list   [--dir D]               List discovered log files with summary.
//   show   <FILE>                  Pretty-print a single log file's records.
//   audit  [--dir D] [--strict]    Aggregate audit. --strict treats any SLO
//                                  miss as failure.
//
// Default --dir: ./.ai_assistant/acp_logs.

#![cfg(feature = "acp")]

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::acp::SloRecord;

const DEFAULT_DIR: &str = ".ai_assistant/acp_logs";
const SLO_HANDSHAKE_MS: u64 = 200;
const SLO_FIRST_CHUNK_MS: u64 = 1000;
const SLO_MIN_CHUNKS_PER_SEC: f64 = 30.0;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "list" => cmd_list(&args[1..]),
        "show" => cmd_show(&args[1..]),
        "audit" => cmd_audit(&args[1..]),
        "--help" | "-h" => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("ai_acp_audit: unknown verb '{}'\n\n{}", other, usage());
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "ai_acp_audit <verb> [args]\n\
     \n\
     Verbs:\n\
       list   [--dir D]              List logs in dir with per-file summary\n\
       show   <FILE>                 Pretty-print a log file's records\n\
       audit  [--dir D] [--strict]   Aggregate audit; non-zero exit on breach\n\
       --help, -h                    Show this message\n\
     \n\
     Default --dir: ./.ai_assistant/acp_logs"
}

fn cmd_list(args: &[String]) -> ExitCode {
    let dir = arg(args, "--dir").unwrap_or_else(|| DEFAULT_DIR.into());
    let files = match discover(&dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ai_acp_audit list: {}", e);
            return ExitCode::from(1);
        }
    };
    if files.is_empty() {
        println!("No ACP logs in {}", dir);
        return ExitCode::SUCCESS;
    }
    println!("{:<48} {:>8} {:>10}", "FILE", "RECORDS", "SESSIONS");
    for f in &files {
        let recs = read_log(f).unwrap_or_default();
        let sessions: std::collections::HashSet<&str> = recs
            .iter()
            .filter_map(|r| r.session_id.as_deref())
            .collect();
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        println!("{:<48} {:>8} {:>10}", name, recs.len(), sessions.len());
    }
    ExitCode::SUCCESS
}

fn cmd_show(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("ai_acp_audit show: missing FILE\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    let recs = match read_log(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ai_acp_audit show: {}", e);
            return ExitCode::from(1);
        }
    };
    println!(
        "{:<14} {:<14} {:>9} {:>7} {:>10}",
        "KIND", "SESSION", "ELAPSED", "CHUNKS", "CHUNKS/S"
    );
    for r in &recs {
        let session = r.session_id.as_deref().unwrap_or("-");
        println!(
            "{:<14} {:<14} {:>7}ms {:>7} {:>10.1}",
            r.kind, session, r.elapsed_ms, r.chunks, r.chunks_per_sec
        );
    }
    ExitCode::SUCCESS
}

fn cmd_audit(args: &[String]) -> ExitCode {
    let dir = arg(args, "--dir").unwrap_or_else(|| DEFAULT_DIR.into());
    let strict = args.iter().any(|a| a == "--strict");
    let files = match discover(&dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ai_acp_audit audit: {}", e);
            return ExitCode::from(1);
        }
    };
    let mut all: Vec<SloRecord> = Vec::new();
    for f in &files {
        if let Ok(recs) = read_log(f) {
            all.extend(recs);
        }
    }
    let total = all.len();
    let handshakes: Vec<&SloRecord> = all.iter().filter(|r| r.kind == "handshake").collect();
    let prompts: Vec<&SloRecord> = all.iter().filter(|r| r.kind == "prompt").collect();
    let first_chunks: Vec<&SloRecord> = all.iter().filter(|r| r.kind == "first_chunk").collect();
    let h_breach: Vec<_> = handshakes
        .iter()
        .filter(|r| r.elapsed_ms > SLO_HANDSHAKE_MS)
        .collect();
    let f_breach: Vec<_> = first_chunks
        .iter()
        .filter(|r| r.elapsed_ms > SLO_FIRST_CHUNK_MS)
        .collect();
    let p_breach: Vec<_> = prompts
        .iter()
        .filter(|r| r.chunks > 0 && r.chunks_per_sec < SLO_MIN_CHUNKS_PER_SEC)
        .collect();

    println!("ACP audit ({})", dir);
    println!("  Files:                     {}", files.len());
    println!("  Records:                   {}", total);
    println!(
        "  Handshakes:                {} (breach >{}ms: {})",
        handshakes.len(),
        SLO_HANDSHAKE_MS,
        h_breach.len()
    );
    println!(
        "  Prompts:                   {} (breach <{:.0} chunks/s: {})",
        prompts.len(),
        SLO_MIN_CHUNKS_PER_SEC,
        p_breach.len()
    );
    println!(
        "  First-chunk records:       {} (breach >{}ms: {})",
        first_chunks.len(),
        SLO_FIRST_CHUNK_MS,
        f_breach.len()
    );

    let any_breach = !h_breach.is_empty() || !f_breach.is_empty() || !p_breach.is_empty();
    if any_breach {
        if strict {
            eprintln!("FAIL: SLO breaches detected and --strict is set");
            return ExitCode::from(1);
        } else {
            println!("WARN: SLO breaches detected (run with --strict to fail)");
        }
    } else {
        println!("OK: all records within SLO targets");
    }
    ExitCode::SUCCESS
}

fn discover(dir: &str) -> std::io::Result<Vec<PathBuf>> {
    let p = Path::new(dir);
    if !p.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(p)? {
        let e = entry?;
        let pth = e.path();
        if pth.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            files.push(pth);
        }
    }
    files.sort();
    Ok(files)
}

fn read_log(path: &Path) -> std::io::Result<Vec<SloRecord>> {
    let f = fs::File::open(path)?;
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<SloRecord>(&line) {
            out.push(rec);
        }
    }
    Ok(out)
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
