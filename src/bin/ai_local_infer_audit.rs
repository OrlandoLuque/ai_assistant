// `ai_local_infer_audit` — read SloRecord JSONL files emitted by
// `ai_local_infer` and check SLO compliance.
//
// Verbs:
//   list   [--dir D]               List discovered log files with summary.
//   show   <FILE>                  Pretty-print a single log file's records.
//   audit  [--dir D] [--strict]    Aggregate audit. --strict treats any SLO
//                                  miss as failure.
//
// SLO targets:
//   load_ms        < 30000   (model load + backend init within 30 s)
//   first_chunk_ms <  1000   (first token within 1 s)
//   tokens_per_sec ≥     5   (CPU baseline; raise once GPU backends land)
//
// Default --dir: ./.ai_assistant/local_infer_logs.

#![cfg(feature = "local-inference")]

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::local_inference::SloRecord;

const DEFAULT_DIR: &str = ".ai_assistant/local_infer_logs";
pub(crate) const SLO_LOAD_MS: u64 = 30_000;
pub(crate) const SLO_FIRST_CHUNK_MS: u64 = 1_000;
pub(crate) const SLO_MIN_TPS: f64 = 5.0;

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
            eprintln!(
                "ai_local_infer_audit: unknown verb '{}'\n\n{}",
                other,
                usage()
            );
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "ai_local_infer_audit <verb> [args]\n\
     \n\
     Verbs:\n\
       list   [--dir D]              List logs in dir with per-file summary\n\
       show   <FILE>                 Pretty-print a log file's records\n\
       audit  [--dir D] [--strict]   Aggregate audit; non-zero exit on breach\n\
       --help, -h                    Show this message\n\
     \n\
     Default --dir: ./.ai_assistant/local_infer_logs"
}

fn cmd_list(args: &[String]) -> ExitCode {
    let dir = arg(args, "--dir").unwrap_or_else(|| DEFAULT_DIR.into());
    let files = match discover(&dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ai_local_infer_audit list: {}", e);
            return ExitCode::from(1);
        }
    };
    if files.is_empty() {
        println!("No local-inference logs in {}", dir);
        return ExitCode::SUCCESS;
    }
    println!("{:<48} {:>8} {:>10}", "FILE", "RECORDS", "BACKENDS");
    for f in &files {
        let recs = read_log(f).unwrap_or_default();
        let backends: std::collections::HashSet<&str> =
            recs.iter().map(|r| r.backend.as_str()).collect();
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        println!("{:<48} {:>8} {:>10}", name, recs.len(), backends.len());
    }
    ExitCode::SUCCESS
}

fn cmd_show(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("ai_local_infer_audit show: missing FILE\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    let recs = match read_log(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ai_local_infer_audit show: {}", e);
            return ExitCode::from(1);
        }
    };
    println!(
        "{:<12} {:>8} {:>10} {:>10} {:>9} {:>9}",
        "BACKEND", "LOAD_MS", "FIRST_MS", "TOTAL_MS", "GEN_TOK", "TOK/S"
    );
    for r in &recs {
        println!(
            "{:<12} {:>8} {:>10} {:>10} {:>9} {:>9.1}",
            r.backend,
            r.load_ms,
            r.first_chunk_ms,
            r.total_ms,
            r.generated_tokens,
            r.tokens_per_sec
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
            eprintln!("ai_local_infer_audit audit: {}", e);
            return ExitCode::from(1);
        }
    };
    let mut all: Vec<SloRecord> = Vec::new();
    for f in &files {
        if let Ok(recs) = read_log(f) {
            all.extend(recs);
        }
    }
    let breaches = count_breaches(&all);

    println!("Local-inference audit ({})", dir);
    println!("  Files:                     {}", files.len());
    println!("  Records:                   {}", all.len());
    println!(
        "  load_ms breaches >{}ms:    {}",
        SLO_LOAD_MS, breaches.load
    );
    println!(
        "  first_chunk breaches >{}ms: {}",
        SLO_FIRST_CHUNK_MS, breaches.first_chunk
    );
    println!(
        "  tokens/sec breaches <{:.1}:  {}",
        SLO_MIN_TPS, breaches.tps
    );

    if breaches.any() {
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

pub(crate) struct BreachCounts {
    pub load: usize,
    pub first_chunk: usize,
    pub tps: usize,
}

impl BreachCounts {
    pub fn any(&self) -> bool {
        self.load + self.first_chunk + self.tps > 0
    }
}

pub(crate) fn count_breaches(records: &[SloRecord]) -> BreachCounts {
    let mut load = 0;
    let mut first_chunk = 0;
    let mut tps = 0;
    for r in records {
        if r.load_ms > SLO_LOAD_MS {
            load += 1;
        }
        if r.first_chunk_ms > SLO_FIRST_CHUNK_MS {
            first_chunk += 1;
        }
        if r.generated_tokens > 0 && r.tokens_per_sec < SLO_MIN_TPS {
            tps += 1;
        }
    }
    BreachCounts {
        load,
        first_chunk,
        tps,
    }
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
