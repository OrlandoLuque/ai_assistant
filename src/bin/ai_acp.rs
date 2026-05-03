// `ai_acp` — Agent Client Protocol server bin.
//
// Two verbs:
//   serve            Read JSON-RPC frames from stdin, write to stdout.
//                    Use this when the editor/client launches us as a
//                    subprocess (Zed, VS Code, JetBrains via ACP).
//   probe <cmd>...   Spawn another ACP server, drive a handshake, and
//                    print SLO timings. Diagnostic only.
//
// Built only with `--features acp`. The protocol is decoupled from the
// underlying model: we wire AiAssistant as the LLM callback at startup.

#![cfg(feature = "acp")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use ai_assistant::acp::{
    parse_frame, serve, AcpChunk, AcpServer, AcpServerConfig, CancelToken, ChunkSender, Inbound,
};
use ai_assistant::{AiAssistant, AiProvider, AiResponse};
use serde_json::{json, Value};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "serve" => cmd_serve(&args[1..]),
        "probe" => cmd_probe(&args[1..]),
        "--help" | "-h" => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("ai_acp: unknown verb '{}'\n\n{}", other, usage());
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "ai_acp <verb> [args]\n\
     \n\
     Verbs:\n\
       serve [--provider PROV --model MODEL [--url URL]]\n\
              Run the ACP server on stdio. Defaults: provider=Ollama, model=$ACP_MODEL.\n\
       probe <cmd> [args...]\n\
              Spawn the given command as an ACP server, run a handshake +\n\
              one prompt, print SLO timings.\n\
       --help, -h\n\
              Show this message."
}

// ────────────────────────────────────────────────────────────────────────────
// serve
// ────────────────────────────────────────────────────────────────────────────

fn cmd_serve(args: &[String]) -> ExitCode {
    let provider = arg(args, "--provider").map(|s| provider_from_name(&s));
    let model = arg(args, "--model").or_else(|| std::env::var("ACP_MODEL").ok());
    let url = arg(args, "--url");
    let log_dir = arg(args, "--log-dir").unwrap_or_else(|| ".ai_assistant/acp_logs".into());

    let llm = build_llm(provider, model, url);
    let mut server = AcpServer::new(AcpServerConfig::default()).with_llm(llm);
    if let Some(sink) = open_slo_sink(&log_dir) {
        server = server.with_slo_sink(sink);
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    if let Err(e) = serve(server, BufReader::new(stdin.lock()), stdout) {
        eprintln!("ai_acp serve error: {}", e);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Build a SLO sink that appends JSONL records to a per-process log file.
/// Returns None on filesystem errors (we still serve — logs are best-effort).
fn open_slo_sink(
    dir: &str,
) -> Option<impl Fn(&ai_assistant::acp::SloRecord) + Send + Sync + 'static> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("ai_acp: cannot create log dir '{}': {}", dir, e);
        return None;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path =
        std::path::PathBuf::from(dir).join(format!("acp_{}_{}.jsonl", ts, std::process::id()));
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ai_acp: cannot open log '{}': {}", path.display(), e);
            return None;
        }
    };
    let file = std::sync::Mutex::new(file);
    Some(move |rec: &ai_assistant::acp::SloRecord| {
        let line = match serde_json::to_string(rec) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(&mut *f, "{}", line);
        }
    })
}

fn provider_from_name(name: &str) -> AiProvider {
    match name.to_lowercase().as_str() {
        "ollama" => AiProvider::Ollama,
        "lmstudio" | "lm_studio" | "lm-studio" => AiProvider::LMStudio,
        _ => AiProvider::Ollama,
    }
}

/// Build the LLM callback. Each call spins up a fresh `AiAssistant` configured
/// with the chosen provider/model/url, sends the prompt, and forwards chunks.
fn build_llm(
    provider: Option<AiProvider>,
    model: Option<String>,
    url: Option<String>,
) -> impl Fn(String, CancelToken, ChunkSender) + Send + Sync + 'static {
    let provider = provider.unwrap_or(AiProvider::Ollama);
    let model = model.unwrap_or_default();
    let url = url.unwrap_or_default();
    move |prompt, cancel, tx| {
        let mut a = AiAssistant::new();
        a.config.provider = provider.clone();
        if !model.is_empty() {
            a.config.selected_model = model.clone();
        }
        if !url.is_empty() {
            match a.config.provider {
                AiProvider::Ollama => a.config.ollama_url = url.clone(),
                AiProvider::LMStudio => a.config.lm_studio_url = url.clone(),
                _ => a.config.custom_url = url.clone(),
            }
        }
        if a.config.selected_model.is_empty() {
            let _ = tx.send(AcpChunk::Error(
                "ai_acp: no model configured (set --model or $ACP_MODEL)".into(),
            ));
            let _ = tx.send(AcpChunk::Done);
            return;
        }
        a.send_message(prompt, "");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if cancel.is_cancelled() {
                a.cancel_generation();
                break;
            }
            if Instant::now() > deadline {
                a.cancel_generation();
                let _ = tx.send(AcpChunk::Error("ai_acp: generation timeout".into()));
                break;
            }
            if let Some(resp) = a.poll_response() {
                match resp {
                    AiResponse::Chunk(t) => {
                        if tx.send(AcpChunk::Delta(t)).is_err() {
                            break;
                        }
                    }
                    AiResponse::Complete(t) => {
                        if !t.is_empty() {
                            let _ = tx.send(AcpChunk::Delta(t));
                        }
                        break;
                    }
                    AiResponse::Cancelled(_) => break,
                    AiResponse::Error(e) => {
                        let _ = tx.send(AcpChunk::Error(e));
                        break;
                    }
                    _ => {}
                }
            } else {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let _ = tx.send(AcpChunk::Done);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// probe — diagnostic client
// ────────────────────────────────────────────────────────────────────────────

fn cmd_probe(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("ai_acp probe: missing command\n\n{}", usage());
        return ExitCode::from(2);
    }
    let mut child = match Command::new(&args[0])
        .args(&args[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ai_acp probe: spawn '{}' failed: {}", args[0], e);
            return ExitCode::from(1);
        }
    };
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);

    let started = Instant::now();
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": 1, "clientCapabilities": {} }
    });
    if let Err(e) = writeln!(stdin, "{}", init) {
        eprintln!("ai_acp probe: write initialize failed: {}", e);
        return ExitCode::from(1);
    }
    let init_resp = match read_response(&mut reader) {
        Some(v) => v,
        None => {
            eprintln!("ai_acp probe: no initialize response");
            return ExitCode::from(1);
        }
    };
    let handshake_ms = started.elapsed().as_millis();
    println!("handshake: {} ms", handshake_ms);
    println!("server response: {}", init_resp);

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".into());
    let new = json!({
        "jsonrpc": "2.0", "id": 2, "method": "session/new",
        "params": { "cwd": cwd, "mcpServers": [] }
    });
    let _ = writeln!(stdin, "{}", new);
    let _ = read_response(&mut reader);

    let prompt_started = Instant::now();
    let prompt = json!({
        "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
        "params": {
            "sessionId": "sess_1",
            "prompt": [ { "type": "text", "text": "Reply with exactly the word: ok" } ]
        }
    });
    let _ = writeln!(stdin, "{}", prompt);
    drop(stdin);

    let mut chunk_count = 0u64;
    let mut first_chunk: Option<Duration> = None;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = match reader.read_line(&mut buf) {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        let line = buf.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        if let Ok(Inbound::Notification(n)) = parse_frame(line) {
            if n.method == "session/update" {
                chunk_count += 1;
                if first_chunk.is_none() {
                    first_chunk = Some(prompt_started.elapsed());
                }
            }
        } else if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v.get("id") == Some(&json!(3)) {
                let total = prompt_started.elapsed();
                let cps = if total.as_secs_f64() > 0.0 {
                    chunk_count as f64 / total.as_secs_f64()
                } else {
                    0.0
                };
                println!(
                    "first_chunk: {} ms",
                    first_chunk.map(|d| d.as_millis()).unwrap_or(0)
                );
                println!("chunks: {}", chunk_count);
                println!("chunks_per_sec: {:.1}", cps);
                println!("total: {} ms", total.as_millis());
                println!("prompt response: {}", v);
                break;
            }
        }
    }

    let _ = child.wait();
    ExitCode::SUCCESS
}

fn read_response<R: BufRead>(reader: &mut R) -> Option<Value> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        let line = buf.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).ok()?;
        if v.get("id").is_some() {
            return Some(v);
        }
        // Skip notifications.
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
