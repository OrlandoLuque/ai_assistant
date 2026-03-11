//! # ai_cluster_node — Cluster Node Binary
//!
//! Dedicated binary for running an ai_assistant node in a distributed cluster.
//! Joins an existing cluster via bootstrap peers, starts QUIC mesh networking,
//! and synchronizes state via CRDTs.
//!
//! ## Usage
//! ```bash
//! # First node (seed node)
//! ai_cluster_node --node-id node1 --port 8091 --quic-port 9001
//!
//! # Additional nodes (join existing cluster)
//! ai_cluster_node --node-id node2 --port 8092 --quic-port 9002 \
//!   --bootstrap-peers 192.168.1.10:9001 --join-token <TOKEN>
//!
//! # With P2P knowledge sharing (requires --features p2p)
//! ai_cluster_node --node-id node1 --port 8091 --quic-port 9001 \
//!   --enable-p2p --p2p-port 12345
//! ```
//!
//! ## Required features
//! `full,server-cluster` (add `p2p` for P2P knowledge sharing)

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use ai_assistant::cluster::{ClusterConfig, ClusterManager};
use ai_assistant::server::{AuthConfig, ServerConfig, TlsConfig};
use ai_assistant::server_axum::AxumServer;

#[cfg(feature = "p2p")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "p2p")]
use ai_assistant::p2p::{P2PConfig, P2PManager, P2PTransport};

// ============================================================================
// CLI argument types
// ============================================================================

#[derive(Debug)]
struct CliArgs {
    // Server config
    host: Option<String>,
    port: Option<u16>,
    config_path: Option<String>,
    api_key: Option<String>,
    tls_cert: Option<String>,
    tls_key: Option<String>,
    // Cluster config
    node_id: Option<String>,
    bootstrap_peers: Option<String>,
    join_token: Option<String>,
    quic_port: Option<u16>,
    data_dir: Option<String>,
    // P2P config
    enable_p2p: bool,
    p2p_port: Option<u16>,
    p2p_bootstrap: Option<String>,
    // General
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

    // Validate required cluster args
    if cli.node_id.is_none() {
        eprintln!("Error: --node-id is required for cluster nodes");
        return ExitCode::FAILURE;
    }

    let server_config = match build_config(&cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let node_id = cli.node_id.as_deref().unwrap_or("node1").to_string();
    let quic_port = cli.quic_port.unwrap_or(9001);
    let data_dir = cli.data_dir.clone().unwrap_or_else(|| "./data".to_string());
    let bootstrap_peers: Vec<String> = cli
        .bootstrap_peers
        .as_ref()
        .map(|p| p.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    if cli.dry_run {
        match serde_json::to_string_pretty(&server_config) {
            Ok(json) => {
                println!("{}", json);
                println!();
                println!("Cluster config:");
                println!("  node-id: {}", node_id);
                println!("  bootstrap-peers: {}", if bootstrap_peers.is_empty() { "(seed node)".to_string() } else { bootstrap_peers.join(", ") });
                println!("  quic-port: {}", quic_port);
                println!("  data-dir: {}", data_dir);
                if cli.enable_p2p {
                    println!("  p2p-port: {}", cli.p2p_port.unwrap_or(12345));
                    println!("  p2p-bootstrap: {}", cli.p2p_bootstrap.as_deref().unwrap_or("(none)"));
                }
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }

    // ========================================================================
    // Initialize ClusterManager (sync context — NetworkNode creates its own runtime)
    // ========================================================================

    let quic_addr: SocketAddr = match format!("127.0.0.1:{}", quic_port).parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: invalid QUIC address: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let cluster_config = ClusterConfig {
        node_id: node_id.clone(),
        quic_addr,
        bootstrap_peers: bootstrap_peers
            .iter()
            .filter_map(|p| p.parse().ok())
            .collect(),
        join_token: cli.join_token.clone(),
        data_dir: PathBuf::from(&data_dir),
        ..Default::default()
    };

    let cluster = match ClusterManager::new(cluster_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to initialize cluster: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // ========================================================================
    // Initialize P2P transport (if enabled)
    // ========================================================================

    #[cfg(feature = "p2p")]
    let p2p_transport: Option<P2PTransport> = if cli.enable_p2p {
        let p2p_port_val = cli.p2p_port.unwrap_or(12345);
        let p2p_listen: SocketAddr = match format!("127.0.0.1:{}", p2p_port_val).parse() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error: invalid P2P address: {}", e);
                return ExitCode::FAILURE;
            }
        };

        let p2p_bootstrap_addrs: Vec<String> = cli
            .p2p_bootstrap
            .as_ref()
            .map(|p| p.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();

        let p2p_config = P2PConfig {
            enabled: true,
            listen_port: p2p_port_val,
            bootstrap_nodes: p2p_bootstrap_addrs,
            enable_upnp: false,  // server node, no UPnP needed
            enable_nat_pmp: false,
            stun_servers: vec![],
            ..Default::default()
        };

        let mut manager = P2PManager::new(p2p_config);
        if let Err(e) = manager.start() {
            eprintln!("Warning: P2P bootstrap failed: {}", e);
        }

        let transport = P2PTransport::new(
            Arc::new(Mutex::new(manager)),
            p2p_listen,
        );
        Some(transport)
    } else {
        None
    };

    #[cfg(not(feature = "p2p"))]
    if cli.enable_p2p {
        eprintln!("Warning: --enable-p2p requires the 'p2p' feature flag. Ignoring.");
    }

    // ========================================================================
    // Create tokio runtime and run everything
    // ========================================================================

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Failed to create tokio runtime: {}", e);
            std::process::exit(1);
        });

    let server = AxumServer::new(server_config);
    let addr = server.config().bind_address();

    eprintln!("AI Assistant Cluster Node v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Node ID: {}", node_id);
    eprintln!("HTTP server: http://{}", addr);
    eprintln!("QUIC mesh: 0.0.0.0:{}", quic_port);
    if bootstrap_peers.is_empty() {
        eprintln!("Mode: seed node (waiting for peers to join)");
    } else {
        eprintln!("Bootstrap peers: {}", bootstrap_peers.join(", "));
    }

    // Start P2P transport threads (before entering async context)
    #[cfg(feature = "p2p")]
    if let Some(ref transport) = p2p_transport {
        transport.start();
        eprintln!("P2P transport: 0.0.0.0:{}", cli.p2p_port.unwrap_or(12345));
    }

    eprintln!("---");

    rt.block_on(async {
        // Start cluster background tasks (heartbeat, CRDT sync, anti-entropy, persistence)
        cluster.start_background_tasks().await;
        eprintln!("Cluster background tasks started.");

        // Connect to bootstrap peers (QUIC)
        for peer_addr in &bootstrap_peers {
            if let Ok(addr) = peer_addr.parse::<SocketAddr>() {
                match cluster.network_node().connect(addr) {
                    Ok(peer_id) => {
                        eprintln!("Connected to bootstrap peer: {} ({})", peer_id, addr);
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to connect to {}: {}", addr, e);
                    }
                }
            }
        }

        let peer_count = cluster.peer_count();
        if peer_count > 0 {
            eprintln!("Connected to {} peer(s).", peer_count);
        }

        // Run the HTTP server (blocks until shutdown signal)
        let server_result = server.run().await;

        // Graceful shutdown
        eprintln!("Shutting down...");

        // Stop P2P transport
        #[cfg(feature = "p2p")]
        if let Some(ref transport) = p2p_transport {
            transport.shutdown();
            eprintln!("P2P transport stopped.");
        }

        // Shutdown cluster (persists CRDTs, drains health, stops tasks)
        cluster.shutdown().await;
        eprintln!("Cluster shutdown complete.");

        if let Err(e) = server_result {
            eprintln!("Server error: {}", e);
        }
    });

    eprintln!("Cluster node stopped.");
    ExitCode::SUCCESS
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

    if let Some(ref host) = cli.host { config.host = host.clone(); }
    if let Some(port) = cli.port { config.port = port; }
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
            config.tls = Some(TlsConfig { cert_path: cert.clone(), key_path: key.clone() });
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
        host: None, port: None, config_path: None, api_key: None,
        tls_cert: None, tls_key: None, node_id: None, bootstrap_peers: None,
        join_token: None, quic_port: None, data_dir: None,
        enable_p2p: false, p2p_port: None, p2p_bootstrap: None,
        dry_run: false, help: false,
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
            "--node-id" => { i += 1; cli.node_id = Some(next_val(args, i, "--node-id")?); }
            "--bootstrap-peers" => { i += 1; cli.bootstrap_peers = Some(next_val(args, i, "--bootstrap-peers")?); }
            "--join-token" => { i += 1; cli.join_token = Some(next_val(args, i, "--join-token")?); }
            "--quic-port" => {
                i += 1;
                let val = next_val(args, i, "--quic-port")?;
                cli.quic_port = Some(val.parse().map_err(|_| format!("Invalid quic-port: '{}'", val))?);
            }
            "--data-dir" => { i += 1; cli.data_dir = Some(next_val(args, i, "--data-dir")?); }
            "--enable-p2p" => cli.enable_p2p = true,
            "--p2p-port" => {
                i += 1;
                let val = next_val(args, i, "--p2p-port")?;
                cli.p2p_port = Some(val.parse().map_err(|_| format!("Invalid p2p-port: '{}'", val))?);
            }
            "--p2p-bootstrap" => { i += 1; cli.p2p_bootstrap = Some(next_val(args, i, "--p2p-bootstrap")?); }
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
    eprintln!("AI Assistant Cluster Node — Distributed AI Processing");
    eprintln!();
    eprintln!("Usage: ai_cluster_node [OPTIONS]");
    eprintln!();
    eprintln!("Server Options:");
    eprintln!("  --host <HOST>               Host to bind HTTP server (default: 127.0.0.1)");
    eprintln!("  --port <PORT>               HTTP port (default: 8090)");
    eprintln!("  --config <PATH>             JSON config file");
    eprintln!("  --api-key <KEY>             API key for authentication");
    eprintln!("  --tls-cert <PATH>           PEM certificate (requires --tls-key)");
    eprintln!("  --tls-key <PATH>            PEM private key (requires --tls-cert)");
    eprintln!();
    eprintln!("Cluster Options:");
    eprintln!("  --node-id <ID>              Node identifier (REQUIRED)");
    eprintln!("  --bootstrap-peers <ADDRS>   Comma-separated peer addresses (host:port)");
    eprintln!("  --join-token <TOKEN>        Cluster admission token");
    eprintln!("  --quic-port <PORT>          QUIC mesh port (default: 9001)");
    eprintln!("  --data-dir <PATH>           CRDT persistence directory (default: ./data)");
    eprintln!();
    eprintln!("P2P Options (requires --features p2p):");
    eprintln!("  --enable-p2p                Enable P2P knowledge sharing layer");
    eprintln!("  --p2p-port <PORT>           P2P TCP listen port (default: 12345)");
    eprintln!("  --p2p-bootstrap <ADDRS>     Comma-separated P2P bootstrap addresses");
    eprintln!();
    eprintln!("General:");
    eprintln!("  --dry-run                   Print config and exit");
    eprintln!("  -h, --help                  Print this help message");
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
        assert!(cli.node_id.is_none());
        assert!(cli.bootstrap_peers.is_none());
        assert!(!cli.enable_p2p);
        assert!(cli.p2p_port.is_none());
    }

    #[test]
    fn test_parse_args_cluster() {
        let a = args(&[
            "--node-id", "node1",
            "--port", "8091",
            "--quic-port", "9001",
            "--bootstrap-peers", "10.0.0.1:9001,10.0.0.2:9001",
            "--join-token", "abc123",
            "--data-dir", "/data/node1",
        ]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(cli.node_id.as_deref(), Some("node1"));
        assert_eq!(cli.port, Some(8091));
        assert_eq!(cli.quic_port, Some(9001));
        assert!(cli.bootstrap_peers.as_ref().unwrap().contains("10.0.0.1"));
        assert_eq!(cli.join_token.as_deref(), Some("abc123"));
        assert_eq!(cli.data_dir.as_deref(), Some("/data/node1"));
    }

    #[test]
    fn test_parse_args_p2p() {
        let a = args(&[
            "--node-id", "n1",
            "--enable-p2p",
            "--p2p-port", "13000",
            "--p2p-bootstrap", "10.0.0.5:12345,10.0.0.6:12345",
        ]);
        let cli = parse_args(&a).unwrap();
        assert!(cli.enable_p2p);
        assert_eq!(cli.p2p_port, Some(13000));
        assert!(cli.p2p_bootstrap.as_ref().unwrap().contains("10.0.0.5"));
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
    fn test_build_config_defaults() {
        let cli = CliArgs {
            host: None, port: None, config_path: None, api_key: None,
            tls_cert: None, tls_key: None, node_id: Some("n1".to_string()),
            bootstrap_peers: None, join_token: None, quic_port: None,
            data_dir: None, enable_p2p: false, p2p_port: None, p2p_bootstrap: None,
            dry_run: false, help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_build_config_overrides() {
        let cli = CliArgs {
            host: Some("0.0.0.0".to_string()), port: Some(8091),
            config_path: None, api_key: None,
            tls_cert: None, tls_key: None, node_id: Some("n1".to_string()),
            bootstrap_peers: Some("10.0.0.1:9001".to_string()),
            join_token: Some("tok".to_string()), quic_port: Some(9001),
            data_dir: Some("/tmp/data".to_string()),
            enable_p2p: false, p2p_port: None, p2p_bootstrap: None,
            dry_run: false, help: false,
        };
        let config = build_config(&cli).unwrap();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8091);
    }
}
