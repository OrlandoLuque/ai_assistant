// server.rs — Lightweight embedded HTTP server for exposing the AI assistant as a REST API.
//
// Uses std::net::TcpListener with manual HTTP parsing — zero extra dependencies.
// Intended for local development, inter-process communication, and simple integrations.
//
// Endpoints:
//   GET  /health       — Health check
//   GET  /models       — List available models
//   POST /chat         — Send a message (non-streaming)
//   POST /config       — Update configuration
//   GET  /config       — Get current configuration
//
// # Example
//
// ```rust,no_run
// use ai_assistant::server::{ServerConfig, AiServer};
//
// let config = ServerConfig::default(); // localhost:8090
// let server = AiServer::new(config);
// server.run_blocking(); // blocks
// ```

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::assistant::AiAssistant;

// ============================================================================
// Server Configuration
// ============================================================================

/// Configuration for the embedded HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to (default: "127.0.0.1").
    pub host: String,
    /// Port to bind to (default: 8090).
    pub port: u16,
    /// Maximum request body size in bytes (default: 1MB).
    pub max_body_size: usize,
    /// Read timeout in seconds (default: 30).
    pub read_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8090,
            max_body_size: 1_048_576,
            read_timeout_secs: 30,
        }
    }
}

impl ServerConfig {
    /// Get the bind address as "host:port".
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ============================================================================
// Request / Response Types
// ============================================================================

/// Parsed HTTP request.
#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    #[allow(dead_code)]
    headers: Vec<(String, String)>,
    body: String,
}

/// Chat request body.
#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
    #[serde(default)]
    system_prompt: String,
    #[serde(default)]
    knowledge_context: String,
}

/// Chat response body.
#[derive(Debug, Serialize)]
struct ChatResponse {
    content: String,
    model: String,
}

/// Health check response.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    model: String,
    provider: String,
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// ============================================================================
// AI Server
// ============================================================================

/// Embedded HTTP server wrapping an AiAssistant instance.
///
/// Thread-safe: the assistant is behind an `Arc<Mutex<_>>`, allowing
/// concurrent request handling.
pub struct AiServer {
    config: ServerConfig,
    assistant: Arc<Mutex<AiAssistant>>,
}

impl AiServer {
    /// Create a new server with the given configuration.
    ///
    /// Initializes an `AiAssistant` with default `AiConfig`.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            assistant: Arc::new(Mutex::new(AiAssistant::new())),
        }
    }

    /// Create a server with a pre-configured assistant.
    pub fn with_assistant(config: ServerConfig, assistant: AiAssistant) -> Self {
        Self {
            config,
            assistant: Arc::new(Mutex::new(assistant)),
        }
    }

    /// Get a reference to the server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Run the server, blocking the current thread.
    ///
    /// Listens for HTTP connections and dispatches requests to handlers.
    /// Each connection is handled in its own thread.
    pub fn run_blocking(&self) -> std::io::Result<()> {
        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr)?;
        log::info!("AI Assistant server listening on http://{}", addr);

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let assistant = self.assistant.clone();
                    let max_body = self.config.max_body_size;
                    let timeout = self.config.read_timeout_secs;
                    std::thread::spawn(move || {
                        if let Err(e) = handle_connection(stream, &assistant, max_body, timeout) {
                            log::debug!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    log::error!("Accept error: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Start the server in a background thread, returning a handle.
    ///
    /// The returned `ServerHandle` can be used to get the actual bound address
    /// (useful when binding to port 0 for automatic allocation).
    pub fn start_background(self) -> std::io::Result<ServerHandle> {
        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr)?;
        let local_addr = listener.local_addr()?;
        let assistant = self.assistant.clone();
        let max_body = self.config.max_body_size;
        let timeout = self.config.read_timeout_secs;

        let handle = std::thread::spawn(move || {
            log::info!("AI Assistant server listening on http://{}", local_addr);
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let assistant = assistant.clone();
                        std::thread::spawn(move || {
                            if let Err(e) = handle_connection(stream, &assistant, max_body, timeout)
                            {
                                log::debug!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        log::error!("Accept error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(ServerHandle {
            address: local_addr,
            _thread: handle,
        })
    }
}

/// Handle to a running server.
pub struct ServerHandle {
    /// The actual bound address.
    pub address: std::net::SocketAddr,
    _thread: std::thread::JoinHandle<()>,
}

impl ServerHandle {
    /// Get the server URL.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }
}

// ============================================================================
// Request Handling
// ============================================================================

fn handle_connection(
    mut stream: TcpStream,
    assistant: &Arc<Mutex<AiAssistant>>,
    max_body_size: usize,
    timeout_secs: u64,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(timeout_secs)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(timeout_secs)))?;

    let request = parse_request(&stream, max_body_size)?;

    let (status, body) = route_request(&request, assistant);
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );

    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn parse_request(stream: &TcpStream, max_body_size: usize) -> std::io::Result<HttpRequest> {
    let mut reader = BufReader::new(stream);

    // Read request line
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid request line",
        ));
    }

    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Read headers
    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim().to_lowercase();
            let value = value.trim().to_string();
            if key == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((key, value));
        }
    }

    // Read body
    let mut body = String::new();
    if content_length > 0 {
        let read_size = content_length.min(max_body_size);
        let mut buf = vec![0u8; read_size];
        reader.read_exact(&mut buf)?;
        body = String::from_utf8_lossy(&buf).to_string();
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn route_request(request: &HttpRequest, assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    // Handle CORS preflight
    if request.method == "OPTIONS" {
        return ("204 No Content".to_string(), String::new());
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => handle_health(assistant),
        ("GET", "/models") => handle_list_models(assistant),
        ("POST", "/chat") => handle_chat(request, assistant),
        ("GET", "/config") => handle_get_config(assistant),
        ("POST", "/config") => handle_set_config(request, assistant),
        _ => (
            "404 Not Found".to_string(),
            serde_json::to_string(&ErrorResponse {
                error: format!("Unknown endpoint: {} {}", request.method, request.path),
            })
            .unwrap_or_default(),
        ),
    }
}

fn handle_health(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: ass.config.selected_model.clone(),
        provider: ass.config.provider.display_name().to_string(),
    };
    (
        "200 OK".to_string(),
        serde_json::to_string(&resp).unwrap_or_default(),
    )
}

fn handle_list_models(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let models: Vec<serde_json::Value> = ass
        .available_models
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m.name,
                "provider": format!("{:?}", m.provider),
                "size": m.size,
            })
        })
        .collect();
    (
        "200 OK".to_string(),
        serde_json::to_string(&models).unwrap_or_default(),
    )
}

fn handle_chat(request: &HttpRequest, assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let chat_req: ChatRequest = match serde_json::from_str(&request.body) {
        Ok(req) => req,
        Err(e) => {
            return (
                "400 Bad Request".to_string(),
                serde_json::to_string(&ErrorResponse {
                    error: format!("Invalid JSON: {}", e),
                })
                .unwrap_or_default(),
            );
        }
    };

    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());

    // send_message_with_notes(user_message: String, knowledge_context, session_notes, knowledge_notes)
    ass.send_message_with_notes(
        chat_req.message.clone(),
        &chat_req.knowledge_context,
        &chat_req.system_prompt,
        "",
    );

    // Poll until complete
    let model = ass.config.selected_model.clone();
    loop {
        match ass.poll_response() {
            Some(crate::messages::AiResponse::Complete(text)) => {
                let resp = ChatResponse {
                    content: text,
                    model,
                };
                return (
                    "200 OK".to_string(),
                    serde_json::to_string(&resp).unwrap_or_default(),
                );
            }
            Some(crate::messages::AiResponse::Error(e)) => {
                return (
                    "500 Internal Server Error".to_string(),
                    serde_json::to_string(&ErrorResponse { error: e }).unwrap_or_default(),
                );
            }
            Some(crate::messages::AiResponse::Cancelled(partial)) => {
                let resp = ChatResponse {
                    content: partial,
                    model,
                };
                return (
                    "200 OK".to_string(),
                    serde_json::to_string(&resp).unwrap_or_default(),
                );
            }
            Some(_) => {
                // Chunk — accumulate would be needed for streaming, skip for now
                continue;
            }
            None => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
}

fn handle_get_config(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    // Don't expose the api_key
    let safe_config = serde_json::json!({
        "provider": format!("{:?}", ass.config.provider),
        "selected_model": ass.config.selected_model,
        "base_url": ass.config.get_base_url(),
        "temperature": ass.config.temperature,
        "max_history_messages": ass.config.max_history_messages,
        "has_api_key": !ass.config.api_key.is_empty(),
    });
    (
        "200 OK".to_string(),
        serde_json::to_string(&safe_config).unwrap_or_default(),
    )
}

fn handle_set_config(
    request: &HttpRequest,
    assistant: &Arc<Mutex<AiAssistant>>,
) -> (String, String) {
    let updates: serde_json::Value = match serde_json::from_str(&request.body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request".to_string(),
                serde_json::to_string(&ErrorResponse {
                    error: format!("Invalid JSON: {}", e),
                })
                .unwrap_or_default(),
            );
        }
    };

    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(model) = updates.get("model").and_then(|m| m.as_str()) {
        ass.config.selected_model = model.to_string();
    }
    if let Some(temp) = updates.get("temperature").and_then(|t| t.as_f64()) {
        ass.config.temperature = temp as f32;
    }

    (
        "200 OK".to_string(),
        serde_json::to_string(&serde_json::json!({"status": "updated"})).unwrap_or_default(),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8090);
        assert_eq!(config.bind_address(), "127.0.0.1:8090");
        assert_eq!(config.max_body_size, 1_048_576);
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 3000,
            ..Default::default()
        };
        assert_eq!(config.bind_address(), "0.0.0.0:3000");
    }

    #[test]
    fn test_health_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_health(&assistant);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
        assert!(parsed["version"].is_string());
    }

    #[test]
    fn test_get_config_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_get_config(&assistant);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed["provider"].is_string());
        assert!(parsed["temperature"].is_number());
        // API key should not be exposed
        assert!(parsed.get("api_key").is_none());
    }

    #[test]
    fn test_set_config_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/config".to_string(),
            headers: vec![],
            body: r#"{"model": "test-model", "temperature": 0.5}"#.to_string(),
        };

        let (status, body) = handle_set_config(&request, &assistant);
        assert_eq!(status, "200 OK");
        assert!(body.contains("updated"));

        let ass = assistant.lock().unwrap();
        assert_eq!(ass.config.selected_model, "test-model");
        assert!((ass.config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_endpoint_invalid_json() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: "not json".to_string(),
        };

        let (status, body) = handle_chat(&request, &assistant);
        assert_eq!(status, "400 Bad Request");
        assert!(body.contains("Invalid JSON"));
    }

    #[test]
    fn test_route_not_found() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/unknown".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant);
        assert_eq!(status, "404 Not Found");
        assert!(body.contains("Unknown endpoint"));
    }

    #[test]
    fn test_route_options_cors() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));

        let request = HttpRequest {
            method: "OPTIONS".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, _) = route_request(&request, &assistant);
        assert_eq!(status, "204 No Content");
    }

    #[test]
    fn test_server_start_background() {
        // Bind to port 0 for automatic allocation
        let config = ServerConfig {
            port: 0,
            ..Default::default()
        };
        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();

        // Verify it got a port
        assert!(handle.address.port() > 0);
        assert!(handle.url().starts_with("http://"));

        // Make a health check request
        let url = format!("{}/health", handle.url());
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                let body: serde_json::Value = resp.into_json().unwrap();
                assert_eq!(body["status"], "ok");
            }
            Err(_) => {
                // Server might not be ready yet in very fast test execution
                std::thread::sleep(std::time::Duration::from_millis(100));
                let resp = ureq::get(&url)
                    .timeout(std::time::Duration::from_secs(2))
                    .call()
                    .unwrap();
                let body: serde_json::Value = resp.into_json().unwrap();
                assert_eq!(body["status"], "ok");
            }
        }
    }
}
