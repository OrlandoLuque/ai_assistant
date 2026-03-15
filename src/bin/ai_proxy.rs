//! # ai_proxy — Lightweight API Gateway
//!
//! Minimal binary that routes OpenAI-compatible API requests to backend
//! ai_assistant nodes. No AiAssistant, no RAG, no local LLM — just routing.
//!
//! Features:
//! - Round-robin load balancing across backend nodes
//! - Session affinity (sticky sessions via X-Session-Id header)
//! - Health checks on backends
//! - Request forwarding with minimal overhead
//!
//! ## Usage
//! ```bash
//! # Route to 3 backend nodes
//! ai_proxy --port 8080 --backends 10.0.0.1:8090,10.0.0.2:8090,10.0.0.3:8090
//!
//! # With health check interval
//! ai_proxy --port 8080 --backends node1:8090,node2:8090 --health-interval 10
//! ```
//!
//! ## Required features
//! `server-axum` (lightest possible — no `full` needed)

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;
use dashmap::DashMap;
use serde::Serialize;

// ============================================================================
// CLI argument types
// ============================================================================

#[derive(Debug)]
struct CliArgs {
    port: Option<u16>,
    backends: Option<String>,
    health_interval: Option<u64>,
    api_key: Option<String>,
    dry_run: bool,
    help: bool,
}

// ============================================================================
// Proxy state
// ============================================================================

#[derive(Clone)]
struct ProxyState {
    backends: Arc<Vec<Backend>>,
    next_index: Arc<AtomicUsize>,
    session_affinity: Arc<DashMap<String, usize>>,
    api_key: Option<String>,
}

struct Backend {
    addr: String,
    healthy: AtomicBool,
}

impl Backend {
    fn new(addr: String) -> Self {
        Self {
            addr,
            healthy: AtomicBool::new(true),
        }
    }
}

#[derive(Serialize)]
struct ProxyHealthResponse {
    status: String,
    backends: Vec<BackendStatus>,
}

#[derive(Serialize)]
struct BackendStatus {
    addr: String,
    healthy: bool,
}

#[derive(Serialize)]
struct ProxyError {
    error: String,
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

    if cli.backends.is_none() {
        eprintln!("Error: --backends is required");
        eprintln!();
        print_usage();
        return ExitCode::FAILURE;
    }

    let port = cli.port.unwrap_or(8080);
    let health_interval = cli.health_interval.unwrap_or(30);
    let backend_addrs: Vec<String> = cli
        .backends
        .unwrap()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if backend_addrs.is_empty() {
        eprintln!("Error: --backends must specify at least one backend address");
        return ExitCode::FAILURE;
    }

    if cli.dry_run {
        println!("AI Proxy Configuration:");
        println!("  port: {}", port);
        println!("  backends: {:?}", backend_addrs);
        println!("  health-interval: {}s", health_interval);
        println!("  api-key: {}", if cli.api_key.is_some() { "(set)" } else { "(none)" });
        return ExitCode::SUCCESS;
    }

    let backends: Vec<Backend> = backend_addrs.iter().map(|a| Backend::new(a.clone())).collect();

    let state = ProxyState {
        backends: Arc::new(backends),
        next_index: Arc::new(AtomicUsize::new(0)),
        session_affinity: Arc::new(DashMap::new()),
        api_key: cli.api_key,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Failed to create tokio runtime: {}", e);
            std::process::exit(1);
        });

    if let Ok(info) = update_rx.try_recv() {
        eprintln!("  Update available: v{} \u{2192} v{}", info.current, info.latest);
        eprintln!("  Download: {}", info.url);
        eprintln!();
    }

    eprintln!("AI Proxy v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Listening on: http://0.0.0.0:{}", port);
    eprintln!("Backends: {}", backend_addrs.join(", "));
    eprintln!("Health check interval: {}s", health_interval);

    rt.block_on(async {
        // Spawn health check loop
        let hc_state = state.clone();
        tokio::spawn(async move {
            health_check_loop(hc_state, Duration::from_secs(health_interval)).await;
        });

        let app = build_proxy_router(state);

        let addr = format!("127.0.0.1:{}", port);
        let listener = match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Failed to bind to {}: {}", addr, e);
                std::process::exit(1);
            }
        };

        eprintln!("Proxy ready. Forwarding requests...");

        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
        {
            eprintln!("Proxy error: {}", e);
            std::process::exit(1);
        }
    });

    eprintln!("Proxy stopped.");
    ExitCode::SUCCESS
}

// ============================================================================
// Router
// ============================================================================

fn build_proxy_router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(proxy_health_handler))
        .fallback(proxy_forward_handler)
        .with_state(state)
}

// ============================================================================
// Handlers
// ============================================================================

async fn proxy_health_handler(State(state): State<ProxyState>) -> Json<ProxyHealthResponse> {
    let backends: Vec<BackendStatus> = state
        .backends
        .iter()
        .map(|b| BackendStatus {
            addr: b.addr.clone(),
            healthy: b.healthy.load(Ordering::Relaxed),
        })
        .collect();

    let all_healthy = backends.iter().any(|b| b.healthy);
    Json(ProxyHealthResponse {
        status: if all_healthy { "ok".to_string() } else { "degraded".to_string() },
        backends,
    })
}

async fn proxy_forward_handler(
    State(state): State<ProxyState>,
    req: Request,
) -> Response {
    // Check API key if configured
    if let Some(ref expected_key) = state.api_key {
        let auth_ok = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|v| {
                if let Some(token) = v.strip_prefix("Bearer ") {
                    // Constant-time comparison to prevent timing side-channel attacks
                    let a = token.as_bytes();
                    let b = expected_key.as_bytes();
                    if a.len() != b.len() {
                        return false;
                    }
                    let mut diff = 0u8;
                    for (x, y) in a.iter().zip(b.iter()) {
                        diff |= x ^ y;
                    }
                    diff == 0
                } else {
                    false
                }
            })
            .unwrap_or(false);

        if !auth_ok {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ProxyError { error: "Unauthorized".to_string() }),
            ).into_response();
        }
    }

    // Determine backend: session affinity or round-robin
    let session_id = req
        .headers()
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let backend_idx = if let Some(ref sid) = session_id {
        if let Some(idx) = state.session_affinity.get(sid).map(|r| *r) {
            // Verify the affinity target is healthy
            if state.backends[idx].healthy.load(Ordering::Relaxed) {
                idx
            } else {
                pick_healthy_backend(&state)
            }
        } else {
            let idx = pick_healthy_backend(&state);
            state.session_affinity.insert(sid.clone(), idx);
            idx
        }
    } else {
        pick_healthy_backend(&state)
    };

    let backend = &state.backends[backend_idx];
    if !backend.healthy.load(Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ProxyError { error: "No healthy backends available".to_string() }),
        ).into_response();
    }

    // Forward the request
    let (parts, body) = req.into_parts();
    let path = parts.uri.path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    let target_url = format!("http://{}{}", backend.addr, path);

    // Build forwarded request
    let client = reqwest::Client::new();
    let mut builder = match parts.method {
        axum::http::Method::GET => client.get(&target_url),
        axum::http::Method::POST => client.post(&target_url),
        axum::http::Method::PUT => client.put(&target_url),
        axum::http::Method::DELETE => client.delete(&target_url),
        axum::http::Method::PATCH => client.patch(&target_url),
        axum::http::Method::HEAD => client.head(&target_url),
        _ => {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                Json(ProxyError { error: "Method not allowed".to_string() }),
            ).into_response();
        }
    };

    // Copy headers (except host)
    for (name, value) in parts.headers.iter() {
        if name != header::HOST {
            if let Ok(v) = value.to_str() {
                builder = builder.header(name.as_str(), v);
            }
        }
    }

    // Forward body
    let body_bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProxyError { error: format!("Failed to read request body: {}", e) }),
            ).into_response();
        }
    };

    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes.to_vec());
    }

    // Send to backend
    match builder.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

            let mut response_builder = Response::builder().status(status);
            for (name, value) in resp.headers().iter() {
                if let Ok(v) = value.to_str() {
                    response_builder = response_builder.header(name.as_str(), v);
                }
            }

            match resp.bytes().await {
                Ok(bytes) => {
                    response_builder
                        .body(Body::from(bytes))
                        .unwrap_or_else(|_| {
                            (StatusCode::INTERNAL_SERVER_ERROR, "Internal error").into_response()
                        })
                }
                Err(e) => {
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(ProxyError { error: format!("Backend read error: {}", e) }),
                    ).into_response()
                }
            }
        }
        Err(e) => {
            // Mark backend as unhealthy on connection error
            if e.is_connect() || e.is_timeout() {
                backend.healthy.store(false, Ordering::Relaxed);
            }
            (
                StatusCode::BAD_GATEWAY,
                Json(ProxyError { error: format!("Backend error: {}", e) }),
            ).into_response()
        }
    }
}

// ============================================================================
// Backend selection
// ============================================================================

fn pick_healthy_backend(state: &ProxyState) -> usize {
    let len = state.backends.len();
    for _ in 0..len {
        let idx = state.next_index.fetch_add(1, Ordering::Relaxed) % len;
        if state.backends[idx].healthy.load(Ordering::Relaxed) {
            return idx;
        }
    }
    // All unhealthy — return first anyway (handler will check)
    0
}

// ============================================================================
// Health check loop
// ============================================================================

async fn health_check_loop(state: ProxyState, interval: Duration) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    loop {
        tokio::time::sleep(interval).await;

        for backend in state.backends.iter() {
            let url = format!("http://{}/health", backend.addr);
            let healthy = match client.get(&url).send().await {
                Ok(resp) => resp.status().is_success(),
                Err(_) => false,
            };
            let was_healthy = backend.healthy.swap(healthy, Ordering::Relaxed);
            if was_healthy != healthy {
                if healthy {
                    eprintln!("[health] Backend {} is now HEALTHY", backend.addr);
                } else {
                    eprintln!("[health] Backend {} is now UNHEALTHY", backend.addr);
                }
            }
        }
    }
}

// ============================================================================
// Graceful shutdown
// ============================================================================

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { eprintln!("\nReceived Ctrl+C, shutting down..."); },
        _ = terminate => { eprintln!("\nReceived SIGTERM, shutting down..."); },
    }
}

// ============================================================================
// Argument parsing
// ============================================================================

fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    let mut cli = CliArgs {
        port: None,
        backends: None,
        health_interval: None,
        api_key: None,
        dry_run: false,
        help: false,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                let val = next_val(args, i, "--port")?;
                cli.port = Some(val.parse().map_err(|_| format!("Invalid port: '{}'", val))?);
            }
            "--backends" => {
                i += 1;
                cli.backends = Some(next_val(args, i, "--backends")?);
            }
            "--health-interval" => {
                i += 1;
                let val = next_val(args, i, "--health-interval")?;
                cli.health_interval = Some(
                    val.parse().map_err(|_| format!("Invalid health-interval: '{}'", val))?,
                );
            }
            "--api-key" => {
                i += 1;
                cli.api_key = Some(next_val(args, i, "--api-key")?);
            }
            "--dry-run" => cli.dry_run = true,
            "-h" | "--help" => cli.help = true,
            other => return Err(format!("Unknown argument: '{}'", other)),
        }
        i += 1;
    }
    Ok(cli)
}

fn next_val(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("{} requires a value", flag))
}

fn print_usage() {
    eprintln!("AI Proxy — Lightweight API Gateway");
    eprintln!();
    eprintln!("Usage: ai_proxy [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --port <PORT>                 Port to listen on (default: 8080)");
    eprintln!("  --backends <ADDR1,ADDR2,...>   Backend addresses (REQUIRED)");
    eprintln!("  --health-interval <SECS>      Health check interval (default: 30)");
    eprintln!("  --api-key <KEY>               API key for Bearer auth");
    eprintln!("  --dry-run                     Print config and exit");
    eprintln!("  -h, --help                    Print this help message");
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
        assert!(cli.port.is_none());
        assert!(cli.backends.is_none());
        assert!(!cli.dry_run);
    }

    #[test]
    fn test_parse_args_full() {
        let a = args(&[
            "--port", "9090",
            "--backends", "10.0.0.1:8090,10.0.0.2:8090",
            "--health-interval", "15",
            "--api-key", "secret",
            "--dry-run",
        ]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.port, Some(9090));
        assert!(cli.backends.as_ref().unwrap().contains("10.0.0.1"));
        assert_eq!(cli.health_interval, Some(15));
        assert_eq!(cli.api_key.as_deref(), Some("secret"));
        assert!(cli.dry_run);
    }

    #[test]
    fn test_parse_args_help() {
        let a = args(&["--help"]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.help);
    }

    #[test]
    fn test_parse_args_unknown() {
        let a = args(&["--unknown"]);
        assert!(parse_args(&a).is_err());
    }

    #[test]
    fn test_pick_healthy_backend_round_robin() {
        let state = ProxyState {
            backends: Arc::new(vec![
                Backend::new("a:8090".to_string()),
                Backend::new("b:8090".to_string()),
                Backend::new("c:8090".to_string()),
            ]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        let idx0 = pick_healthy_backend(&state);
        let idx1 = pick_healthy_backend(&state);
        let idx2 = pick_healthy_backend(&state);
        // Should cycle through 0, 1, 2
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn test_pick_healthy_backend_skips_unhealthy() {
        let state = ProxyState {
            backends: Arc::new(vec![
                Backend::new("a:8090".to_string()),
                Backend::new("b:8090".to_string()),
                Backend::new("c:8090".to_string()),
            ]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        // Mark first backend as unhealthy
        state.backends[0].healthy.store(false, Ordering::Relaxed);
        let idx = pick_healthy_backend(&state);
        assert_eq!(idx, 1); // Skips 0, picks 1
    }

    #[test]
    fn test_build_proxy_router() {
        let state = ProxyState {
            backends: Arc::new(vec![Backend::new("localhost:8090".to_string())]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        let _router = build_proxy_router(state);
        // Should not panic
    }
}
