//! # ai_assistant_standalone — All-in-One Binary
//!
//! The simplest deployment: a single binary that combines:
//! - HTTP API server (axum-based, port 8090)
//! - Interactive REPL on stdin (optional, via --repl)
//! - Butler auto-configuration (optional, via --auto-config)
//!
//! ## Usage
//! ```bash
//! # Start server only
//! ai_assistant_standalone --port 8090
//!
//! # Start server + interactive REPL
//! ai_assistant_standalone --port 8090 --repl
//!
//! # Auto-detect environment and configure optimally
//! ai_assistant_standalone --auto-config
//! ```
//!
//! ## Required features
//! `full,server-axum`

use std::process::ExitCode;

use ai_assistant::AiResponse;
use ai_assistant::server::{AuthConfig, ServerConfig, TlsConfig};
use ai_assistant::server_axum::AxumServer;

// ============================================================================
// CLI argument types
// ============================================================================

#[derive(Debug)]
struct CliArgs {
    host: Option<String>,
    port: Option<u16>,
    config_path: Option<String>,
    api_key: Option<String>,
    tls_cert: Option<String>,
    tls_key: Option<String>,
    repl: bool,
    auto_config: bool,
    dry_run: bool,
    help: bool,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let cli = match parse_args(&args[1..]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    if cli.help {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let config = match build_config(&cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Butler auto-config: detect environment and recommend settings
    #[cfg(feature = "butler")]
    if cli.auto_config {
        eprintln!("Running Butler environment scan...");
        let mut butler = ai_assistant::butler::Butler::new();
        let report = butler.scan();

        // Print detections
        if !report.llm_providers.is_empty() {
            for p in &report.llm_providers {
                eprintln!("  [Butler] Detected: {} ({})", p.name, p.url);
            }
        }
        for cap in &report.capabilities {
            eprintln!("  [Butler] Capability: {}", cap);
        }

        // Print recommendations
        let advice = butler.advise(&report);
        for rec in &advice.recommendations {
            eprintln!("  [Butler] {}: {}", rec.category, rec.message);
        }
        eprintln!();
    }

    #[cfg(not(feature = "butler"))]
    if cli.auto_config {
        eprintln!("Warning: --auto-config requires the 'butler' feature. Ignoring.");
    }

    if cli.dry_run {
        match serde_json::to_string_pretty(&config) {
            Ok(json) => {
                println!("{}", json);
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Error serializing config: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }

    // Build the runtime and start the server
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Failed to create tokio runtime: {}", e);
            std::process::exit(1);
        });

    let server = AxumServer::new(config);
    let addr = server.config().bind_address();

    eprintln!("AI Assistant Standalone v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Server: http://{}", addr);

    if cli.repl {
        eprintln!("REPL: enabled (type messages on stdin, Ctrl+C to quit)");

        // Run server in background, REPL on main thread
        let state = server.state();
        rt.spawn(async move {
            if let Err(e) = server.run().await {
                eprintln!("Server error: {}", e);
            }
        });

        // Simple REPL: read from stdin, send to assistant
        run_repl(state, &rt);
    } else {
        rt.block_on(async {
            if let Err(e) = server.run().await {
                eprintln!("Server error: {}", e);
                std::process::exit(1);
            }
        });
    }

    eprintln!("Standalone server stopped.");
    ExitCode::SUCCESS
}

/// Simple interactive REPL: read lines from stdin, send to assistant, print responses.
fn run_repl(state: ai_assistant::server_axum::AppState, rt: &tokio::runtime::Runtime) {
    use std::io::{self, BufRead, Write};

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    eprintln!("Type a message and press Enter. Type 'quit' or Ctrl+C to exit.\n");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "quit" || trimmed == "exit" {
            break;
        }

        // Send message via the assistant
        let assistant = state.assistant.clone();
        let response = rt.block_on(async {
            let mut ass = assistant.lock().await;
            ass.send_message_simple(trimmed);
            let model = ass.config.selected_model.clone();
            loop {
                match ass.poll_response() {
                    Some(AiResponse::Complete(text)) => {
                        return Ok::<_, String>(format!("[{}] {}", model, text));
                    }
                    Some(AiResponse::Error(e)) => {
                        return Err(e);
                    }
                    Some(AiResponse::Cancelled(partial)) => {
                        return Ok(format!("[{}] {}", model, partial));
                    }
                    Some(_) => continue,
                    None => {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                }
            }
        });

        match response {
            Ok(text) => {
                println!("{}", text);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        print!("> ");
        let _ = stdout.flush();
    }
}

// ============================================================================
// Config building
// ============================================================================

fn build_config(cli: &CliArgs) -> Result<ServerConfig, String> {
    let mut config = if let Some(ref path) = cli.config_path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file '{}': {}", path, e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file '{}': {}", path, e))?
    } else {
        ServerConfig::default()
    };

    if let Some(ref host) = cli.host {
        config.host = host.clone();
    }
    if let Some(port) = cli.port {
        config.port = port;
    }
    if let Some(ref key) = cli.api_key {
        config.auth = AuthConfig {
            enabled: true,
            bearer_tokens: vec![key.clone()],
            api_keys: vec![],
            exempt_paths: vec!["/health".to_string()],
        };
    }

    match (&cli.tls_cert, &cli.tls_key) {
        (Some(cert), Some(key)) => {
            config.tls = Some(TlsConfig {
                cert_path: cert.clone(),
                key_path: key.clone(),
            });
        }
        (Some(_), None) => return Err("--tls-cert requires --tls-key".to_string()),
        (None, Some(_)) => return Err("--tls-key requires --tls-cert".to_string()),
        (None, None) => {}
    }

    Ok(config)
}

// ============================================================================
// Argument parsing
// ============================================================================

fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    let mut cli = CliArgs {
        host: None,
        port: None,
        config_path: None,
        api_key: None,
        tls_cert: None,
        tls_key: None,
        repl: false,
        auto_config: false,
        dry_run: false,
        help: false,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => { i += 1; cli.host = Some(next_val(args, i, "--host")?); }
            "--port" => {
                i += 1;
                let val = next_val(args, i, "--port")?;
                cli.port = Some(val.parse().map_err(|_| format!("Invalid port: '{}'", val))?);
            }
            "--config" => { i += 1; cli.config_path = Some(next_val(args, i, "--config")?); }
            "--api-key" => { i += 1; cli.api_key = Some(next_val(args, i, "--api-key")?); }
            "--tls-cert" => { i += 1; cli.tls_cert = Some(next_val(args, i, "--tls-cert")?); }
            "--tls-key" => { i += 1; cli.tls_key = Some(next_val(args, i, "--tls-key")?); }
            "--repl" => cli.repl = true,
            "--auto-config" => cli.auto_config = true,
            "--dry-run" => cli.dry_run = true,
            "-h" | "--help" => cli.help = true,
            other => return Err(format!("Unknown argument: '{}'", other)),
        }
        i += 1;
    }
    Ok(cli)
}

fn next_val(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index).cloned().ok_or_else(|| format!("{} requires a value", flag))
}

fn print_usage() {
    eprintln!("AI Assistant Standalone — All-in-One Binary");
    eprintln!();
    eprintln!("Usage: ai_assistant_standalone [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --host <HOST>      Host to bind to (default: 127.0.0.1)");
    eprintln!("  --port <PORT>      Port to bind to (default: 8090)");
    eprintln!("  --config <PATH>    Path to JSON config file");
    eprintln!("  --api-key <KEY>    API key for Bearer authentication");
    eprintln!("  --tls-cert <PATH>  PEM certificate file (requires --tls-key)");
    eprintln!("  --tls-key <PATH>   PEM private key file (requires --tls-cert)");
    eprintln!("  --repl             Enable interactive REPL on stdin");
    eprintln!("  --auto-config      Run Butler environment scan and auto-configure");
    eprintln!("  --dry-run          Print resolved config and exit");
    eprintln!("  -h, --help         Print this help message");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_parse_args_defaults() {
        let a = args(&[]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.host.is_none());
        assert!(!cli.repl);
        assert!(!cli.auto_config);
    }

    #[test]
    fn test_parse_args_repl() {
        let a = args(&["--repl"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.repl);
    }

    #[test]
    fn test_parse_args_auto_config() {
        let a = args(&["--auto-config"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.auto_config);
    }

    #[test]
    fn test_parse_args_all() {
        let a = args(&["--host", "0.0.0.0", "--port", "3000", "--repl", "--auto-config", "--dry-run"]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(cli.port, Some(3000));
        assert!(cli.repl);
        assert!(cli.auto_config);
        assert!(cli.dry_run);
    }

    #[test]
    fn test_parse_args_help() {
        let a = args(&["-h"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.help);
    }

    #[test]
    fn test_parse_args_unknown() {
        let a = args(&["--unknown"]);
        assert!(parse_args(&a).is_err());
    }

    #[test]
    fn test_build_config_defaults() {
        let cli = CliArgs {
            host: None, port: None, config_path: None, api_key: None,
            tls_cert: None, tls_key: None, repl: false, auto_config: false,
            dry_run: false, help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8090);
    }

    #[test]
    fn test_build_config_overrides() {
        let cli = CliArgs {
            host: Some("0.0.0.0".to_string()), port: Some(3000),
            config_path: None, api_key: Some("key".to_string()),
            tls_cert: None, tls_key: None, repl: false, auto_config: false,
            dry_run: false, help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3000);
        assert!(config.auth.enabled);
    }

    #[test]
    fn test_build_config_tls_mismatch() {
        let cli = CliArgs {
            host: None, port: None, config_path: None, api_key: None,
            tls_cert: Some("c.pem".to_string()), tls_key: None,
            repl: false, auto_config: false, dry_run: false, help: false,
        };
        assert!(build_config(&cli).is_err());
    }
}
