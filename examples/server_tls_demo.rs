//! Example: server_tls_demo -- Demonstrates HTTPS server configuration.
//!
//! Run with: cargo run --example server_tls_demo --features "server-tls"
//!
//! This example showcases TLS configuration for the embedded HTTP server:
//! TlsConfig, load_tls_config, ServerConfig with TLS, AuthConfig, and CorsConfig.
//! No real TLS certificates are required — the example demonstrates the
//! configuration API and validates error handling for missing cert files.

use ai_assistant::{load_tls_config, AiServer, AuthConfig, CorsConfig, ServerConfig, TlsConfig};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Server TLS Configuration Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. TLS Configuration
    // ------------------------------------------------------------------
    println!("--- 1. TlsConfig ---\n");

    let tls_config = TlsConfig {
        cert_path: "/path/to/server.crt".to_string(),
        key_path: "/path/to/server.key".to_string(),
    };

    println!("  cert_path: {}", tls_config.cert_path);
    println!("  key_path:  {}", tls_config.key_path);

    // Attempt to load TLS config — will fail gracefully since files don't exist
    match load_tls_config(&tls_config) {
        Ok(_config) => println!("  TLS config loaded successfully"),
        Err(e) => println!("  Expected error (no cert files): {}", e),
    }

    // ------------------------------------------------------------------
    // 2. ServerConfig with TLS
    // ------------------------------------------------------------------
    println!("\n--- 2. ServerConfig with TLS ---\n");

    let config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8443,
        tls: Some(TlsConfig {
            cert_path: "./certs/server.crt".to_string(),
            key_path: "./certs/server.key".to_string(),
        }),
        auth: AuthConfig {
            enabled: true,
            bearer_tokens: vec!["my-secret-token".to_string()],
            api_keys: vec!["api-key-12345".to_string()],
            exempt_paths: vec!["/health".to_string()],
        },
        cors: CorsConfig {
            allowed_origins: vec!["https://myapp.example.com".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
            ],
            max_age_secs: 3600,
            allow_credentials: true,
        },
        ..Default::default()
    };

    println!("  Bind address: {}", config.bind_address());
    println!("  TLS enabled:  {}", config.tls.is_some());
    println!("  Auth enabled:  {}", config.auth.enabled);
    println!("  CORS origins:  {:?}", config.cors.allowed_origins);
    println!("  Max body size: {} bytes", config.max_body_size);

    // ------------------------------------------------------------------
    // 3. Plain HTTP vs HTTPS comparison
    // ------------------------------------------------------------------
    println!("\n--- 3. Plain HTTP vs HTTPS ---\n");

    let http_config = ServerConfig::default();
    println!("  HTTP  -> {}  (TLS: {})", http_config.bind_address(), http_config.tls.is_some());

    let https_config = ServerConfig {
        port: 8443,
        tls: Some(TlsConfig {
            cert_path: "./certs/server.crt".to_string(),
            key_path: "./certs/server.key".to_string(),
        }),
        ..Default::default()
    };
    println!("  HTTPS -> {}  (TLS: {})", https_config.bind_address(), https_config.tls.is_some());

    // ------------------------------------------------------------------
    // 4. Server creation (does not start — just instantiates)
    // ------------------------------------------------------------------
    println!("\n--- 4. AiServer instantiation ---\n");

    // Create with plain HTTP (no TLS) so it succeeds without cert files
    let plain_config = ServerConfig::default();
    let server = AiServer::new(plain_config);
    println!("  Server created (plain HTTP, not started)");

    // Demonstrate shutdown on a server that hasn't started
    server.shutdown();
    println!("  Server shutdown called (no-op since not running)");

    // ------------------------------------------------------------------
    // 5. Self-signed certificate generation guide
    // ------------------------------------------------------------------
    println!("\n--- 5. Generating Self-Signed Certificates ---\n");

    println!("  To create self-signed certs for testing:\n");
    println!("  openssl req -x509 -newkey rsa:4096 -keyout server.key \\");
    println!("    -out server.crt -days 365 -nodes \\");
    println!("    -subj '/CN=localhost'\n");
    println!("  Then set TlsConfig {{ cert_path: \"server.crt\", key_path: \"server.key\" }}");
    println!("  and the server will accept HTTPS connections.");

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  server-tls demo complete.");
    println!("  Capabilities: TLS config, cert loading, auth,");
    println!("    CORS, HTTPS server creation.");
    println!("==========================================================");
}
