//! AI Assistant HTTP Server — standalone binary.
//!
//! Usage:
//!   ai_assistant_server [OPTIONS]
//!
//! Options:
//!   --host <HOST>      Host to bind to (default: 127.0.0.1)
//!   --port <PORT>      Port to bind to (default: 8090)
//!   --config <PATH>    Path to JSON config file
//!   --api-key <KEY>    API key for Bearer authentication
//!   --tls-cert <PATH>  Path to PEM certificate file
//!   --tls-key <PATH>   Path to PEM private key file
//!   --dry-run          Validate config and print it, then exit
//!   -h, --help         Print this help message

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ai_assistant::server::{AiServer, AuthConfig, ServerConfig, TlsConfig};

// ============================================================================
// CLI argument types
// ============================================================================

/// Parsed CLI arguments.
#[derive(Debug)]
struct CliArgs {
    host: Option<String>,
    port: Option<u16>,
    config_path: Option<String>,
    api_key: Option<String>,
    tls_cert: Option<String>,
    tls_key: Option<String>,
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

    // Check for updates (spawn early so it has time to complete)
    let update_rx = ai_assistant::update_checker::check_for_update_bg(env!("CARGO_PKG_VERSION"));

    let config = match build_config(&cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if cli.dry_run {
        // Print the resolved config as pretty JSON (excluding non-serializable fields).
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

    // Start server with graceful shutdown on Ctrl-C.
    let server = AiServer::new(config);

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);

    // Register Ctrl-C handler.
    if let Err(e) = ctrlc_register(shutdown_flag, &server) {
        eprintln!("Warning: failed to register Ctrl-C handler: {}", e);
    }

    if let Ok(info) = update_rx.try_recv() {
        eprintln!("  Update available: v{} \u{2192} v{}", info.current, info.latest);
        eprintln!("  Download: {}", info.url);
        eprintln!();
    }

    eprintln!("Starting AI Assistant server...");
    if let Err(e) = server.run_blocking() {
        // If shutdown was requested, exit cleanly.
        if shutdown.load(Ordering::Relaxed) {
            eprintln!("Server stopped.");
            return ExitCode::SUCCESS;
        }
        eprintln!("Server error: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Server stopped.");
    ExitCode::SUCCESS
}

// ============================================================================
// Ctrl-C registration
// ============================================================================

/// Register a Ctrl-C handler that sets the shutdown flag and calls server.shutdown().
///
/// Uses a simple `std::thread` + platform signal approach via a spawned thread
/// that waits on `std::io::stdin` close, but more practically we rely on the
/// server's own shutdown mechanism.  The simplest cross-platform approach without
/// extra deps is to spawn a thread that polls the shutdown flag.
///
/// Since AiServer already has a `shutdown()` method that sets its internal
/// AtomicBool, we hook Ctrl-C by spawning a thread that detects the signal.
fn ctrlc_register(shutdown_flag: Arc<AtomicBool>, server: &AiServer) -> Result<(), String> {
    // We need a 'static reference to call server.shutdown(), so we use a
    // one-shot approach: clone what we need into the closure.
    // AiServer doesn't implement Clone, but we can get a handle to its
    // shutdown mechanism by calling shutdown() from outside.
    // Instead, we'll rely on a background thread + platform-specific signal.

    // Cross-platform: use std's ctrl-c on Windows/Unix via a polling thread.
    let _ = server; // We'll use the shutdown_flag to signal the main loop.

    // Spawn a thread that waits for Ctrl-C.
    std::thread::spawn(move || {
        // Block until Ctrl-C.  On Windows, SetConsoleCtrlHandler; on Unix, sigaction.
        // Since we have no deps, we use a simple busy-wait on the flag, but the
        // actual Ctrl-C will cause the TcpListener::accept to return an error,
        // which the server handles.  So we just set the flag for clean messaging.
        wait_for_ctrlc();
        eprintln!("\nCtrl-C received, shutting down...");
        shutdown_flag.store(true, Ordering::Relaxed);
    });

    Ok(())
}

/// Platform-specific Ctrl-C wait.
#[cfg(windows)]
fn wait_for_ctrlc() {
    use std::sync::atomic::{AtomicBool, Ordering};

    static CTRLC_RECEIVED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
        CTRLC_RECEIVED.store(true, Ordering::Relaxed);
        1 // TRUE — handled
    }

    extern "system" {
        fn SetConsoleCtrlHandler(
            handler: unsafe extern "system" fn(u32) -> i32,
            add: i32,
        ) -> i32;
    }

    unsafe {
        SetConsoleCtrlHandler(handler, 1);
    }

    while !CTRLC_RECEIVED.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[cfg(not(windows))]
fn wait_for_ctrlc() {
    // On Unix, use sigwait on SIGINT.
    // Simplified: sleep-poll approach.  The OS will deliver SIGINT to the process
    // which will terminate it if unhandled; the default Rust handler will cause
    // the TcpListener to return an error, breaking the server loop.
    // We just keep this thread alive.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

// ============================================================================
// Config building
// ============================================================================

/// Build a `ServerConfig` from CLI args, optionally loading a JSON config file first.
fn build_config(cli: &CliArgs) -> Result<ServerConfig, String> {
    // Start with file config or defaults.
    let mut config = if let Some(ref path) = cli.config_path {
        load_config_file(path)?
    } else {
        ServerConfig::default()
    };

    // CLI overrides.
    if let Some(ref host) = cli.host {
        config.host = host.clone();
    }
    if let Some(port) = cli.port {
        config.port = port;
    }
    if let Some(ref key) = cli.api_key {
        let mut auth = AuthConfig::default();
        auth.enabled = true;
        auth.bearer_tokens = vec![key.clone()];
        auth.exempt_paths = vec!["/health".to_string()];
        config.auth = auth;
    }

    // TLS: both cert and key must be provided together.
    match (&cli.tls_cert, &cli.tls_key) {
        (Some(cert), Some(key)) => {
            config.tls = Some(TlsConfig::new(cert.clone(), key.clone()));
        }
        (Some(_), None) => {
            return Err("--tls-cert requires --tls-key".to_string());
        }
        (None, Some(_)) => {
            return Err("--tls-key requires --tls-cert".to_string());
        }
        (None, None) => {}
    }

    Ok(config)
}

/// Load a `ServerConfig` from a JSON file.
fn load_config_file(path: &str) -> Result<ServerConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config file '{}': {}", path, e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse config file '{}': {}", path, e))
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
        dry_run: false,
        help: false,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                let val = next_value(args, i, "--host")?;
                cli.host = Some(val);
            }
            "--port" => {
                i += 1;
                let val = next_value(args, i, "--port")?;
                let port: u16 = val
                    .parse()
                    .map_err(|_| format!("Invalid port number: '{}'", val))?;
                cli.port = Some(port);
            }
            "--config" => {
                i += 1;
                let val = next_value(args, i, "--config")?;
                cli.config_path = Some(val);
            }
            "--api-key" => {
                i += 1;
                let val = next_value(args, i, "--api-key")?;
                cli.api_key = Some(val);
            }
            "--tls-cert" => {
                i += 1;
                let val = next_value(args, i, "--tls-cert")?;
                cli.tls_cert = Some(val);
            }
            "--tls-key" => {
                i += 1;
                let val = next_value(args, i, "--tls-key")?;
                cli.tls_key = Some(val);
            }
            "--dry-run" => {
                cli.dry_run = true;
            }
            "-h" | "--help" => {
                cli.help = true;
            }
            other => {
                return Err(format!("Unknown argument: '{}'", other));
            }
        }
        i += 1;
    }

    Ok(cli)
}

/// Get the next positional value for a flag, returning an error if missing.
fn next_value(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    if index >= args.len() {
        return Err(format!("{} requires a value", flag));
    }
    Ok(args[index].clone())
}

// ============================================================================
// Usage
// ============================================================================

fn print_usage() {
    eprintln!("AI Assistant HTTP Server");
    eprintln!();
    eprintln!("Usage: ai_assistant_server [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --host <HOST>      Host to bind to (default: 127.0.0.1)");
    eprintln!("  --port <PORT>      Port to bind to (default: 8090)");
    eprintln!("  --config <PATH>    Path to JSON config file");
    eprintln!("  --api-key <KEY>    API key for Bearer authentication");
    eprintln!("  --tls-cert <PATH>  Path to PEM certificate file");
    eprintln!("  --tls-key <PATH>   Path to PEM private key file");
    eprintln!("  --dry-run          Validate config and print it, then exit");
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
        assert!(cli.port.is_none());
        assert!(cli.config_path.is_none());
        assert!(cli.api_key.is_none());
        assert!(cli.tls_cert.is_none());
        assert!(cli.tls_key.is_none());
        assert!(!cli.dry_run);
        assert!(!cli.help);
    }

    #[test]
    fn test_parse_args_host_port() {
        let a = args(&["--host", "0.0.0.0", "--port", "9090"]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(cli.port, Some(9090));
    }

    #[test]
    fn test_parse_args_api_key() {
        let a = args(&["--api-key", "secret123"]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.api_key.as_deref(), Some("secret123"));
    }

    #[test]
    fn test_parse_args_dry_run() {
        let a = args(&["--dry-run"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.dry_run);
    }

    #[test]
    fn test_parse_args_help() {
        let a = args(&["--help"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.help);

        let a2 = args(&["-h"]);
        let cli2 = parse_args(&a2).unwrap();
        assert!(cli2.help);
    }

    #[test]
    fn test_parse_args_tls() {
        let a = args(&["--tls-cert", "cert.pem", "--tls-key", "key.pem"]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.tls_cert.as_deref(), Some("cert.pem"));
        assert_eq!(cli.tls_key.as_deref(), Some("key.pem"));
    }

    #[test]
    fn test_parse_args_invalid_port() {
        let a = args(&["--port", "abc"]);
        let err = parse_args(&a).unwrap_err();
        assert!(err.contains("Invalid port"), "got: {}", err);
    }

    #[test]
    fn test_parse_args_missing_value() {
        let a = args(&["--host"]);
        let err = parse_args(&a).unwrap_err();
        assert!(err.contains("requires a value"), "got: {}", err);
    }

    #[test]
    fn test_parse_args_unknown_flag() {
        let a = args(&["--unknown"]);
        let err = parse_args(&a).unwrap_err();
        assert!(err.contains("Unknown argument"), "got: {}", err);
    }

    #[test]
    fn test_build_config_defaults() {
        let cli = CliArgs {
            host: None,
            port: None,
            config_path: None,
            api_key: None,
            tls_cert: None,
            tls_key: None,
            dry_run: false,
            help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8090);
        assert!(!config.auth.enabled);
        assert!(config.tls.is_none());
    }

    #[test]
    fn test_build_config_overrides() {
        let cli = CliArgs {
            host: Some("0.0.0.0".to_string()),
            port: Some(3000),
            config_path: None,
            api_key: Some("mykey".to_string()),
            tls_cert: Some("c.pem".to_string()),
            tls_key: Some("k.pem".to_string()),
            dry_run: false,
            help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3000);
        assert!(config.auth.enabled);
        assert_eq!(config.auth.bearer_tokens, vec!["mykey"]);
        let tls = config.tls.unwrap();
        assert_eq!(tls.cert_path, "c.pem");
        assert_eq!(tls.key_path, "k.pem");
    }

    #[test]
    fn test_build_config_tls_cert_without_key() {
        let cli = CliArgs {
            host: None,
            port: None,
            config_path: None,
            api_key: None,
            tls_cert: Some("c.pem".to_string()),
            tls_key: None,
            dry_run: false,
            help: false,
        };
        let err = build_config(&cli).unwrap_err();
        assert!(err.contains("--tls-cert requires --tls-key"), "got: {}", err);
    }

    #[test]
    fn test_build_config_tls_key_without_cert() {
        let cli = CliArgs {
            host: None,
            port: None,
            config_path: None,
            api_key: None,
            tls_cert: None,
            tls_key: Some("k.pem".to_string()),
            dry_run: false,
            help: false,
        };
        let err = build_config(&cli).unwrap_err();
        assert!(err.contains("--tls-key requires --tls-cert"), "got: {}", err);
    }

    #[test]
    fn test_load_config_file_not_found() {
        let err = load_config_file("/nonexistent/path.json").unwrap_err();
        assert!(err.contains("Failed to read config file"), "got: {}", err);
    }

    #[test]
    fn test_load_config_file_invalid_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("ai_server_test_invalid.json");
        std::fs::write(&path, "not json").unwrap();
        let err = load_config_file(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("Failed to parse config file"), "got: {}", err);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_config_file_valid() {
        let dir = std::env::temp_dir();
        let path = dir.join("ai_server_test_valid.json");
        let json = r#"{
            "host": "10.0.0.1",
            "port": 4000,
            "max_body_size": 2048,
            "read_timeout_secs": 10,
            "auth": { "enabled": false, "bearer_tokens": [], "api_keys": [], "exempt_paths": [] },
            "cors": { "allowed_origins": ["*"], "allowed_methods": ["GET","POST"], "allowed_headers": ["Content-Type"], "allow_credentials": false, "max_age_secs": 3600 },
            "max_headers": 50,
            "max_header_line": 4096,
            "body_read_timeout_ms": 15000,
            "max_message_length": 50000
        }"#;
        std::fs::write(&path, json).unwrap();
        let config = load_config_file(path.to_str().unwrap()).unwrap();
        assert_eq!(config.host, "10.0.0.1");
        assert_eq!(config.port, 4000);
        assert_eq!(config.max_body_size, 2048);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_build_config_from_file_with_cli_override() {
        let dir = std::env::temp_dir();
        let path = dir.join("ai_server_test_override.json");
        let json = r#"{
            "host": "10.0.0.1",
            "port": 4000,
            "max_body_size": 1048576,
            "read_timeout_secs": 30,
            "auth": { "enabled": false, "bearer_tokens": [], "api_keys": [], "exempt_paths": [] },
            "cors": { "allowed_origins": ["*"], "allowed_methods": ["GET","POST"], "allowed_headers": ["Content-Type"], "allow_credentials": false, "max_age_secs": 3600 },
            "max_headers": 100,
            "max_header_line": 8192,
            "body_read_timeout_ms": 30000,
            "max_message_length": 100000
        }"#;
        std::fs::write(&path, json).unwrap();

        let cli = CliArgs {
            host: Some("0.0.0.0".to_string()),
            port: None, // keep file value
            config_path: Some(path.to_str().unwrap().to_string()),
            api_key: None,
            tls_cert: None,
            tls_key: None,
            dry_run: false,
            help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "0.0.0.0"); // CLI override
        assert_eq!(config.port, 4000); // from file
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_parse_args_all_combined() {
        let a = args(&[
            "--host", "192.168.1.1",
            "--port", "8080",
            "--config", "server.json",
            "--api-key", "tok123",
            "--tls-cert", "my.crt",
            "--tls-key", "my.key",
            "--dry-run",
        ]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.host.as_deref(), Some("192.168.1.1"));
        assert_eq!(cli.port, Some(8080));
        assert_eq!(cli.config_path.as_deref(), Some("server.json"));
        assert_eq!(cli.api_key.as_deref(), Some("tok123"));
        assert_eq!(cli.tls_cert.as_deref(), Some("my.crt"));
        assert_eq!(cli.tls_key.as_deref(), Some("my.key"));
        assert!(cli.dry_run);
        assert!(!cli.help);
    }
}
