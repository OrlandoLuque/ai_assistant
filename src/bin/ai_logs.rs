//! `ai_logs` — Distributed log viewer and monitor.
//!
//! Read, search, tail, and export distributed log traces from local JSONL files
//! or remote nodes.
//!
//! # Usage
//!
//! ```text
//! ai_logs list                              # list traces from ./logs/
//! ai_logs list --source /var/log/ai/        # list from custom dir
//! ai_logs show <trace_id>                   # show entries for a trace
//! ai_logs show <trace_id> --level warn      # filter by level
//! ai_logs tail                              # watch for new entries
//! ai_logs tail --source ./logs/             # tail specific dir
//! ai_logs export <trace_id> -o trace.json   # export to file
//! ```

use ai_assistant::distributed_log::{
    colorize_level, parse_log_level, ExportFormat, LogLevel, LogReader, LogTailer,
};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    let command = args[1].as_str();
    let source = get_arg(&args, "--source")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./logs"));
    let level = get_arg(&args, "--level")
        .and_then(|s| parse_log_level(&s))
        .unwrap_or(LogLevel::Trace);
    let format = get_arg(&args, "--format").unwrap_or_else(|| "text".to_string());
    let no_color = args.iter().any(|a| a == "--no-color");

    match command {
        "list" => cmd_list(&source, level),
        "show" => {
            if args.len() < 3 {
                eprintln!("Usage: ai_logs show <trace_id> [--level <LEVEL>] [--source <PATH>]");
                std::process::exit(1);
            }
            cmd_show(&args[2], &source, level, no_color);
        }
        "tail" => {
            let interval: u64 = get_arg(&args, "--interval")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            cmd_tail(&source, level, interval, no_color);
        }
        "export" => {
            if args.len() < 3 {
                eprintln!("Usage: ai_logs export <trace_id> [--format json|text|csv] [-o <FILE>]");
                std::process::exit(1);
            }
            let output = get_arg(&args, "-o");
            cmd_export(&args[2], &source, &format, output.as_deref());
        }
        _ => {
            eprintln!("Unknown command: {}. Use --help for usage.", command);
            std::process::exit(1);
        }
    }
}

fn cmd_list(source: &Path, _min_level: LogLevel) {
    let traces = match LogReader::list_traces(source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", source.display(), e);
            std::process::exit(1);
        }
    };

    if traces.is_empty() {
        println!("No traces found in {}", source.display());
        return;
    }

    println!(
        "{:<36} {:>6} {:>8} {:>8} {:<20} LEVELS",
        "TRACE ID", "ENTRIES", "FIRST", "LAST", "NODES"
    );
    println!("{}", "-".repeat(100));

    for t in &traces {
        let levels_str: String = t
            .levels
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        let nodes_str = t.nodes.join(",");
        let first = format_ts(t.first_timestamp_ms);
        let last = format_ts(t.last_timestamp_ms);

        println!(
            "{:<36} {:>6} {:>8} {:>8} {:<20} {}",
            truncate(&t.trace_id, 36),
            t.entry_count,
            first,
            last,
            truncate(&nodes_str, 20),
            levels_str,
        );
    }

    println!("\n{} trace(s) found", traces.len());
}

fn cmd_show(trace_id: &str, source: &Path, min_level: LogLevel, no_color: bool) {
    let entries = match read_entries(source) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let filtered: Vec<_> = entries
        .iter()
        .filter(|e| e.trace_id == trace_id && e.level >= min_level)
        .collect();

    if filtered.is_empty() {
        eprintln!(
            "No entries found for trace '{}' (level >= {})",
            trace_id, min_level
        );
        std::process::exit(1);
    }

    println!("Trace: {} ({} entries)\n", trace_id, filtered.len());

    for e in &filtered {
        let line = e.to_text();
        if no_color {
            println!("{}", line);
        } else {
            println!("{}", colorize_level(e.level, &line));
        }
    }
}

fn cmd_tail(source: &PathBuf, min_level: LogLevel, interval_secs: u64, no_color: bool) {
    // Find all .jsonl files to tail
    let files: Vec<PathBuf> = if source.is_file() {
        vec![source.clone()]
    } else if source.is_dir() {
        std::fs::read_dir(source)
            .unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", source.display(), e);
                std::process::exit(1);
            })
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
            .collect()
    } else {
        eprintln!("Source not found: {}", source.display());
        std::process::exit(1);
    };

    if files.is_empty() {
        eprintln!("No .jsonl files found in {}", source.display());
        std::process::exit(1);
    }

    let mut tailers: Vec<LogTailer> = files
        .iter()
        .filter_map(|f| LogTailer::new(f).ok())
        .collect();

    println!(
        "Tailing {} file(s) in {} (Ctrl+C to stop, interval: {}s)\n",
        tailers.len(),
        source.display(),
        interval_secs,
    );

    loop {
        let mut found_new = false;
        for tailer in &mut tailers {
            if let Ok(entries) = tailer.next_entries() {
                for e in &entries {
                    if e.level >= min_level {
                        let line = e.to_text();
                        if no_color {
                            println!("{}", line);
                        } else {
                            println!("{}", colorize_level(e.level, &line));
                        }
                        found_new = true;
                    }
                }
            }
        }

        if found_new {
            // Flush stdout for immediate display
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }

        std::thread::sleep(std::time::Duration::from_secs(interval_secs));
    }
}

fn cmd_export(trace_id: &str, source: &Path, format_str: &str, output: Option<&str>) {
    let entries = match read_entries(source) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| e.trace_id == trace_id)
        .collect();

    if filtered.is_empty() {
        eprintln!("No entries found for trace '{}'", trace_id);
        std::process::exit(1);
    }

    let format = match format_str {
        "json" => ExportFormat::Json,
        "csv" => ExportFormat::Csv,
        _ => ExportFormat::Text,
    };

    // Use LogCollector's export via a temporary collector
    let mut collector =
        ai_assistant::distributed_log::LogCollector::new("export", Default::default());
    for e in filtered {
        collector.add_entry(&e.trace_id.clone(), e);
    }
    let exported = collector.export_trace(trace_id, format);

    if let Some(path) = output {
        match std::fs::write(path, &exported) {
            Ok(_) => println!("Exported to {}", path),
            Err(e) => {
                eprintln!("Error writing {}: {}", path, e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", exported);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn read_entries(
    source: &Path,
) -> Result<Vec<ai_assistant::distributed_log::DistributedLogEntry>, String> {
    if source.is_dir() {
        LogReader::read_dir(source).map_err(|e| format!("{}: {}", source.display(), e))
    } else if source.is_file() {
        LogReader::read_file(source).map_err(|e| format!("{}: {}", source.display(), e))
    } else {
        Err(format!("Source not found: {}", source.display()))
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn format_ts(ms: u64) -> String {
    let secs = ms / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

fn print_help() {
    println!("ai_logs — Distributed log viewer and monitor");
    println!();
    println!("USAGE:");
    println!("    ai_logs <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    list                    List available traces");
    println!("    show <trace_id>         Show log entries for a trace");
    println!("    tail                    Watch for new log entries");
    println!("    export <trace_id>       Export a trace to file");
    println!();
    println!("OPTIONS:");
    println!("    --source <PATH>         Log file or directory (default: ./logs/)");
    println!("    --level <LEVEL>         Min level: trace, debug, info, warn, error");
    println!("    --format <FMT>          Export format: text, json, csv");
    println!("    --interval <SECS>       Tail poll interval (default: 1)");
    println!("    --no-color              Disable colored output");
    println!("    -o <FILE>               Output file for export");
    println!("    -h, --help              Show this help");
    println!();
    println!("EXAMPLES:");
    println!("    ai_logs list");
    println!("    ai_logs show abc123def456 --level warn");
    println!("    ai_logs tail --source /var/log/ai/ --interval 2");
    println!("    ai_logs export abc123 --format json -o trace.json");
}
