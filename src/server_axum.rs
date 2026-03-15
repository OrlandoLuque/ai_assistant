//! # axum-based HTTP Server (v30)
//!
//! Production-ready async HTTP server using the axum framework.
//! Replaces the `std::net::TcpListener` server with:
//! - HTTP/1.1 and HTTP/2 support
//! - Native WebSocket and SSE streaming
//! - Tower middleware stack (auth, CORS, compression, rate limiting, etc.)
//! - Graceful shutdown via tokio signals
//! - Optional TLS via axum-server + rustls (`server-axum-tls`)
//! - Optional cluster mode via `server-cluster` feature
//!
//! ## Feature flags
//! - `server-axum` — enables this module
//! - `server-axum-tls` — adds HTTPS/TLS support
//! - `server-cluster` — adds distributed cluster capabilities
//! - `server-openapi` — adds Swagger UI at `/swagger-ui`
//!
//! ## Usage
//! ```rust,no_run
//! use ai_assistant::server_axum::AxumServer;
//! use ai_assistant::server::ServerConfig;
//!
//! let config = ServerConfig::default();
//! let server = AxumServer::new(config);
//! // server.run().await; // async entrypoint
//! ```

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive as SseKeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::{body::Body, Router};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::assistant::AiAssistant;
use crate::server::{
    AuthResult, CorsConfig, ServerConfig,
    authenticate_request,
};

// ============================================================================
// AppState — Granular Concurrency (Phase 2)
// ============================================================================

/// Shared application state for all axum handlers.
///
/// Uses granular concurrency instead of a single `Arc<Mutex<AiAssistant>>`:
/// - `assistant`: tokio Mutex (held briefly for LLM calls)
/// - `server_config`: tokio RwLock (read-heavy, rarely written)
/// - `sessions`: DashMap (lock-free per-session access)
/// - `metrics`: AtomicU64 counters (lock-free)
/// - `rate_limiter`: DashMap-based per-IP rate limiting
#[derive(Clone)]
pub struct AppState {
    /// The AI assistant instance (tokio Mutex for async-safe access).
    pub assistant: Arc<tokio::sync::Mutex<AiAssistant>>,
    /// Server configuration (RwLock for read-heavy access).
    pub server_config: Arc<tokio::sync::RwLock<ServerConfig>>,
    /// Per-session metadata (lock-free via DashMap).
    pub sessions: Arc<DashMap<String, SessionData>>,
    /// Atomic metrics counters.
    pub metrics: Arc<AxumServerMetrics>,
    /// Per-IP sliding-window rate limiter (None = disabled).
    pub rate_limiter: Option<Arc<AxumRateLimiter>>,
    /// Guardrail pipeline (cloned from ServerConfig for independent access).
    pub guardrail_pipeline: Option<Arc<Mutex<crate::guardrail_pipeline::GuardrailPipeline>>>,
    /// Cost budget manager (cloned from ServerConfig).
    pub budget_manager: Option<Arc<Mutex<crate::cost::BudgetManager>>>,
    /// Model registry for virtual models and publish control.
    pub model_registry: Arc<crate::virtual_model::ModelRegistry>,
    /// Optional cluster manager (behind `server-cluster` feature).
    #[cfg(feature = "server-cluster")]
    pub cluster: Option<Arc<crate::cluster::ClusterManager>>,
    /// Optional MCP server for Docker container tools (behind `containers` + `tools`).
    #[cfg(all(feature = "containers", feature = "tools"))]
    pub mcp_server: Option<Arc<std::sync::RwLock<crate::mcp_protocol::McpServer>>>,
}

/// Per-session metadata stored in the DashMap.
#[derive(Debug, Clone, Serialize)]
pub struct SessionData {
    /// Session identifier.
    pub id: String,
    /// Timestamp of the last request in this session.
    pub last_active: u64,
    /// Number of messages in this session.
    pub message_count: usize,
    /// Optional affinity node (for cluster mode).
    pub affinity_node: Option<String>,
}

// ============================================================================
// AxumServerMetrics — Lock-Free Atomic Counters (Phase 2)
// ============================================================================

/// Atomic counters for server-level metrics (Prometheus-compatible).
pub struct AxumServerMetrics {
    pub requests_total: AtomicU64,
    pub requests_2xx: AtomicU64,
    pub requests_4xx: AtomicU64,
    pub requests_5xx: AtomicU64,
    pub request_duration_us_total: AtomicU64,
    request_id_counter: AtomicU64,
    pub started_at: Instant,
    /// Per-endpoint request counts.
    pub endpoint_counts: DashMap<String, u64>,
}

impl AxumServerMetrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_2xx: AtomicU64::new(0),
            requests_4xx: AtomicU64::new(0),
            requests_5xx: AtomicU64::new(0),
            request_duration_us_total: AtomicU64::new(0),
            request_id_counter: AtomicU64::new(0),
            started_at: Instant::now(),
            endpoint_counts: DashMap::new(),
        }
    }

    /// Generate a unique request ID (hex timestamp + counter).
    pub fn generate_request_id(&self) -> String {
        let counter = self.request_id_counter.fetch_add(1, Ordering::Relaxed);
        let nanos = self.started_at.elapsed().as_nanos() as u64;
        format!("{:08x}-{:04x}", nanos, counter & 0xFFFF)
    }

    /// Record a completed request with its status code and duration.
    pub fn record_request(&self, status: u16, duration: Duration) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.request_duration_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        match status {
            200..=299 => { self.requests_2xx.fetch_add(1, Ordering::Relaxed); }
            400..=499 => { self.requests_4xx.fetch_add(1, Ordering::Relaxed); }
            500..=599 => { self.requests_5xx.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
    }

    /// Render metrics in Prometheus text format.
    pub fn render_prometheus(&self) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let ok = self.requests_2xx.load(Ordering::Relaxed);
        let client_err = self.requests_4xx.load(Ordering::Relaxed);
        let server_err = self.requests_5xx.load(Ordering::Relaxed);
        let dur_us = self.request_duration_us_total.load(Ordering::Relaxed);
        let uptime = self.started_at.elapsed().as_secs();
        let avg_ms = if total > 0 {
            (dur_us as f64 / total as f64) / 1000.0
        } else {
            0.0
        };

        format!(
            "# HELP ai_axum_requests_total Total HTTP requests.\n\
             # TYPE ai_axum_requests_total counter\n\
             ai_axum_requests_total {}\n\
             # HELP ai_axum_requests_2xx Successful requests.\n\
             # TYPE ai_axum_requests_2xx counter\n\
             ai_axum_requests_2xx {}\n\
             # HELP ai_axum_requests_4xx Client error requests.\n\
             # TYPE ai_axum_requests_4xx counter\n\
             ai_axum_requests_4xx {}\n\
             # HELP ai_axum_requests_5xx Server error requests.\n\
             # TYPE ai_axum_requests_5xx counter\n\
             ai_axum_requests_5xx {}\n\
             # HELP ai_axum_request_duration_avg_ms Average request duration.\n\
             # TYPE ai_axum_request_duration_avg_ms gauge\n\
             ai_axum_request_duration_avg_ms {:.3}\n\
             # HELP ai_axum_uptime_seconds Server uptime.\n\
             # TYPE ai_axum_uptime_seconds gauge\n\
             ai_axum_uptime_seconds {}\n",
            total, ok, client_err, server_err, avg_ms, uptime,
        )
    }
}

impl Default for AxumServerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AxumRateLimiter — DashMap Per-IP Sliding Window (Phase 2)
// ============================================================================

/// Entry in the sliding-window rate limiter.
struct SlidingWindowEntry {
    /// Timestamps of requests within the current window.
    timestamps: Vec<Instant>,
}

/// Per-IP sliding-window rate limiter using DashMap.
pub struct AxumRateLimiter {
    /// Max requests per window.
    pub max_requests: u32,
    /// Window duration.
    pub window: Duration,
    /// Per-IP request timestamps.
    entries: DashMap<IpAddr, SlidingWindowEntry>,
}

impl AxumRateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window: Duration::from_secs(window_secs),
            entries: DashMap::new(),
        }
    }

    /// Check if a request from `ip` is allowed. Returns true if allowed.
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window;

        let mut entry = self.entries.entry(ip).or_insert_with(|| SlidingWindowEntry {
            timestamps: Vec::new(),
        });

        // Remove expired timestamps
        entry.timestamps.retain(|t| *t > cutoff);

        if entry.timestamps.len() < self.max_requests as usize {
            entry.timestamps.push(now);
            true
        } else {
            false
        }
    }

    /// Get the number of seconds until the next request slot opens for `ip`.
    pub fn retry_after(&self, ip: IpAddr) -> u64 {
        if let Some(entry) = self.entries.get(&ip) {
            if let Some(oldest) = entry.timestamps.first() {
                let elapsed = oldest.elapsed();
                if elapsed < self.window {
                    return (self.window - elapsed).as_secs() + 1;
                }
            }
        }
        1
    }

    /// Cleanup expired entries across all IPs (call periodically).
    pub fn cleanup(&self) {
        let cutoff = Instant::now() - self.window;
        self.entries.retain(|_, entry| {
            entry.timestamps.retain(|t| *t > cutoff);
            !entry.timestamps.is_empty()
        });
    }
}

// ============================================================================
// AppError — axum Error Response (Phase 2)
// ============================================================================

/// Application error type that implements `IntoResponse`.
#[derive(Debug)]
#[non_exhaustive]
pub enum AppError {
    /// Bad request (400).
    BadRequest(String),
    /// Unauthorized (401).
    Unauthorized(String),
    /// Not found (404).
    NotFound(String),
    /// Unprocessable entity (422).
    UnprocessableEntity(String),
    /// Too many requests (429).
    TooManyRequests { message: String, retry_after: u64 },
    /// Internal server error (500).
    Internal(String),
    /// Service unavailable (503).
    ServiceUnavailable(String),
}

/// JSON error body.
#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

/// OpenAI-format error body.
#[derive(Serialize)]
struct OpenAIErrorBody {
    error: OpenAIErrorDetail,
}

#[derive(Serialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody { error: msg }),
            ).into_response(),
            AppError::Unauthorized(msg) => (
                StatusCode::UNAUTHORIZED,
                Json(ErrorBody { error: msg }),
            ).into_response(),
            AppError::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody { error: msg }),
            ).into_response(),
            AppError::UnprocessableEntity(msg) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorBody { error: msg }),
            ).into_response(),
            AppError::TooManyRequests { message, retry_after } => {
                let mut resp = (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(ErrorBody { error: message }),
                ).into_response();
                resp.headers_mut().insert(
                    "Retry-After",
                    retry_after.to_string().parse().unwrap_or_else(|_| "60".parse().unwrap()),
                );
                resp
            }
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody { error: msg }),
            ).into_response(),
            AppError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody { error: msg }),
            ).into_response(),
        }
    }
}

/// Build an OpenAI-format error response.
fn openai_error_response(
    status: StatusCode,
    message: &str,
    error_type: &str,
    code: &str,
) -> Response {
    (
        status,
        Json(OpenAIErrorBody {
            error: OpenAIErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: code.to_string(),
            },
        }),
    )
        .into_response()
}

// ============================================================================
// Session Affinity — SessionId Extractor (Phase 6)
// ============================================================================

/// Session ID extracted from request headers, cookies, or auto-generated.
///
/// Extraction order:
/// 1. `X-Session-Id` header
/// 2. `session_id` cookie
/// 3. Auto-generated UUID
#[derive(Debug, Clone)]
pub struct SessionId(pub String);

impl<S> axum::extract::FromRequestParts<S> for SessionId
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        // 1. Check X-Session-Id header
        if let Some(val) = parts.headers.get("x-session-id") {
            if let Ok(s) = val.to_str() {
                if !s.is_empty() {
                    return Ok(SessionId(s.to_string()));
                }
            }
        }

        // 2. Check cookie
        if let Some(cookie_header) = parts.headers.get(header::COOKIE) {
            if let Ok(cookies) = cookie_header.to_str() {
                for cookie in cookies.split(';') {
                    let cookie = cookie.trim();
                    if let Some(val) = cookie.strip_prefix("session_id=") {
                        if !val.is_empty() {
                            return Ok(SessionId(val.to_string()));
                        }
                    }
                }
            }
        }

        // 3. Auto-generate UUID-like ID
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let id = format!("sess-{:016x}", ts);
        Ok(SessionId(id))
    }
}

/// Session affinity manager for cluster mode.
///
/// Maps session IDs to preferred node indices using a DashMap.
/// Falls back to consistent hash ring when the assigned node is down.
pub struct SessionAffinityManager {
    /// Map: session_id → node_index.
    affinity: DashMap<String, usize>,
    /// TTL for session affinity entries.
    ttl: Duration,
    /// Last access time per session.
    last_access: DashMap<String, Instant>,
}

impl SessionAffinityManager {
    /// Create a new session affinity manager.
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            affinity: DashMap::new(),
            ttl: Duration::from_secs(ttl_secs),
            last_access: DashMap::new(),
        }
    }

    /// Get the node index for a session, or None if not assigned/expired.
    pub fn get_node(&self, session_id: &str) -> Option<usize> {
        // Check TTL — read and drop guard before any mutation
        let expired = self
            .last_access
            .get(session_id)
            .map(|last| last.elapsed() > self.ttl)
            .unwrap_or(false);

        if expired {
            self.affinity.remove(session_id);
            self.last_access.remove(session_id);
            return None;
        }
        self.affinity.get(session_id).map(|v| *v)
    }

    /// Assign a session to a node.
    pub fn set_node(&self, session_id: &str, node_index: usize) {
        self.affinity.insert(session_id.to_string(), node_index);
        self.last_access.insert(session_id.to_string(), Instant::now());
    }

    /// Refresh the last-access time for a session.
    pub fn touch(&self, session_id: &str) {
        self.last_access.insert(session_id.to_string(), Instant::now());
    }

    /// Remove expired affinity entries.
    pub fn cleanup_expired(&self) {
        let now = Instant::now();
        self.last_access.retain(|_, last| now.duration_since(*last) < self.ttl);
        // Remove affinity entries without last_access
        self.affinity.retain(|k, _| self.last_access.contains_key(k));
    }
}

// ============================================================================
// Middleware — Auth Layer (Phase 3)
// ============================================================================

/// Authentication middleware using the existing `authenticate_request` logic.
async fn auth_middleware(
    State(state): State<AppState>,
    req: axum::http::Request<Body>,
    next: Next,
) -> Response {
    let config = state.server_config.read().await;
    let auth_config = &config.auth;

    if !auth_config.enabled {
        drop(config);
        return next.run(req).await;
    }

    let path = req.uri().path().to_string();

    // Convert axum headers to the format expected by authenticate_request
    let headers: Vec<(String, String)> = req
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_lowercase(), v.to_str().unwrap_or("").to_string()))
        .collect();

    let result = authenticate_request(&headers, &path, auth_config);
    drop(config);

    match result {
        AuthResult::Authenticated | AuthResult::Exempt => next.run(req).await,
        AuthResult::Rejected(reason) => {
            log::debug!("Auth rejected: {}", reason);
            state.metrics.record_request(401, Duration::ZERO);
            AppError::Unauthorized("Unauthorized".to_string()).into_response()
        }
    }
}

/// Rate limiting middleware.
async fn rate_limit_middleware(
    State(state): State<AppState>,
    req: axum::http::Request<Body>,
    next: Next,
) -> Response {
    if let Some(ref limiter) = state.rate_limiter {
        // Extract client IP from ConnectInfo or X-Forwarded-For
        let ip = extract_client_ip(&req);

        if !limiter.check(ip) {
            let retry_after = limiter.retry_after(ip);
            state.metrics.record_request(429, Duration::ZERO);
            return AppError::TooManyRequests {
                message: "Too Many Requests".to_string(),
                retry_after,
            }
            .into_response();
        }
    }
    next.run(req).await
}

/// Metrics recording middleware.
async fn metrics_middleware(
    State(state): State<AppState>,
    req: axum::http::Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let path = req.uri().path().to_string();
    let method = req.method().to_string();

    let response = next.run(req).await;

    let status = response.status().as_u16();
    let duration = start.elapsed();
    state.metrics.record_request(status, duration);

    // Track per-endpoint counts
    let key = format!("{} {}", method, path);
    state.metrics.endpoint_counts
        .entry(key)
        .and_modify(|c| *c += 1)
        .or_insert(1);

    let elapsed_ms = duration.as_secs_f64() * 1000.0;
    log::info!("{} {} → {} ({:.1}ms)", method, path, status, elapsed_ms);

    response
}

/// Extract client IP from the request (X-Forwarded-For, X-Real-IP, or remote addr).
fn extract_client_ip(req: &axum::http::Request<Body>) -> IpAddr {
    // Try X-Forwarded-For header
    if let Some(xff) = req.headers().get("x-forwarded-for") {
        if let Ok(s) = xff.to_str() {
            if let Some(first) = s.split(',').next() {
                if let Ok(ip) = first.trim().parse::<IpAddr>() {
                    return ip;
                }
            }
        }
    }

    // Try X-Real-IP header
    if let Some(xri) = req.headers().get("x-real-ip") {
        if let Ok(s) = xri.to_str() {
            if let Ok(ip) = s.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // Fallback to loopback
    IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
}

// ============================================================================
// Request/Response Types (Phase 4)
// ============================================================================

/// Chat request body (mirrors server.rs ChatRequest).
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "server-openapi", derive(utoipa::ToSchema))]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub knowledge_context: String,
    /// Optional model name — can refer to a physical or virtual model.
    #[serde(default)]
    pub model: Option<String>,
}

// ============================================================================
// StreamEvent — Bridge between blocking LLM thread and async handlers (v31)
// ============================================================================

/// Events sent from the blocking LLM thread to async streaming handlers.
///
/// Used by SSE, OpenAI streaming, and WebSocket handlers to forward
/// real tokens as they arrive from the provider, instead of waiting
/// for the complete response.
enum StreamEvent {
    /// A streaming token from the LLM provider.
    Token(String),
    /// Generation is complete.
    Done,
    /// An error occurred during generation.
    Error(String),
}

/// Chat response body.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub content: String,
    pub model: String,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model: String,
    pub provider: String,
    pub uptime_secs: u64,
    pub active_sessions: usize,
    pub conversation_messages: usize,
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

/// A single message in the OpenAI chat format.
#[derive(Debug, Deserialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

// ============================================================================
// Router Builder (Phase 3/4)
// ============================================================================

/// Build the complete axum router with all middleware and endpoints.
///
/// Takes a snapshot of the `ServerConfig` for middleware setup.
/// Call this from an async context after reading the config.
pub fn build_router(state: AppState, config: &ServerConfig) -> Router {
    // Build CORS layer from CorsConfig
    let cors_layer = build_cors_layer(&config.cors);

    // Build the API routes (shared between / and /api/v1)
    let api_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/models", get(models_handler))
        .route("/chat", post(chat_handler))
        .route("/chat/stream", post(chat_stream_handler))
        .route("/config", get(get_config_handler).post(set_config_handler))
        .route("/metrics", get(metrics_handler))
        .route("/sessions", get(list_sessions_handler))
        .route("/sessions/{id}", get(get_session_handler).delete(delete_session_handler))
        .route("/openapi.json", get(openapi_handler))
        .route("/ws", get(ws_handler))
        .route("/chat/completions", post(openai_completions_handler));

    // OpenAI v1 routes
    let v1_routes = Router::new()
        .route("/v1/chat/completions", post(openai_completions_handler))
        .route("/v1/models", get(openai_models_handler));

    // Admin routes (virtual models + publish control)
    let admin_routes = Router::new()
        .route("/admin/virtual-models", get(admin_list_virtual_models).post(admin_create_virtual_model))
        .route("/admin/virtual-models/{name}", get(admin_get_virtual_model).put(admin_update_virtual_model).delete(admin_delete_virtual_model))
        .route("/admin/models", get(admin_list_models))
        .route("/admin/models/{name}/publish", post(admin_publish_model))
        .route("/admin/models/{name}/unpublish", post(admin_unpublish_model));

    // Compose the full router
    #[allow(unused_mut)]
    let mut app = Router::new()
        .merge(api_routes.clone())
        .merge(v1_routes)
        .merge(admin_routes)
        .nest("/api/v1", api_routes);

    // MCP endpoint for Docker container tools (behind containers + tools)
    #[cfg(all(feature = "containers", feature = "tools"))]
    {
        app = app.route("/mcp", post(mcp_handler));
    }

    // Swagger UI (behind server-openapi feature)
    #[cfg(feature = "server-openapi")]
    {
        use utoipa::OpenApi;
        use utoipa_swagger_ui::SwaggerUi;

        #[derive(OpenApi)]
        #[openapi(
            info(
                title = "AI Assistant API",
                version = "1.0.0",
                description = "OpenAI-compatible AI Assistant HTTP API with enrichment pipeline"
            ),
            paths(
                health_handler,
                models_handler,
                chat_handler,
                metrics_handler,
                list_sessions_handler,
                openapi_handler,
            ),
        )]
        struct ApiDoc;

        app = app.merge(SwaggerUi::new("/swagger-ui").url("/swagger-api.json", ApiDoc::openapi()));
    }

    let app = app
        .fallback(fallback_handler)
        .layer(middleware::from_fn_with_state(state.clone(), metrics_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .layer(cors_layer)
        .layer(CompressionLayer::new())
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(config.read_timeout_secs)))
        .with_state(state);

    app
}

/// Build a `CorsLayer` from the `CorsConfig`.
fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    let mut cors = CorsLayer::new();

    // Origins
    if config.allowed_origins.contains(&"*".to_string()) {
        cors = cors.allow_origin(tower_http::cors::Any);
    } else {
        let origins: Vec<_> = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        cors = cors.allow_origin(origins);
    }

    // Methods
    let methods: Vec<Method> = config
        .allowed_methods
        .iter()
        .filter_map(|m| m.parse().ok())
        .collect();
    cors = cors.allow_methods(methods);

    // Headers
    let headers: Vec<_> = config
        .allowed_headers
        .iter()
        .filter_map(|h| h.parse().ok())
        .collect();
    cors = cors.allow_headers(headers);

    // Max age
    cors = cors.max_age(Duration::from_secs(config.max_age_secs));

    // Credentials
    if config.allow_credentials {
        cors = cors.allow_credentials(true);
    }

    cors
}

/// Fallback handler for unmatched routes.
async fn fallback_handler() -> Response {
    AppError::NotFound("Not Found".to_string()).into_response()
}

// ============================================================================
// Handlers — Phase 4
// ============================================================================

/// GET /health — Health check.
#[cfg_attr(feature = "server-openapi", utoipa::path(get, path = "/health", responses((status = 200, description = "Health check response"))))]
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let ass = state.assistant.lock().await;
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: ass.config.selected_model.clone(),
        provider: ass.config.provider.display_name().to_string(),
        uptime_secs: state.metrics.started_at.elapsed().as_secs(),
        active_sessions: ass.session_store.sessions.len(),
        conversation_messages: ass.conversation.len(),
    };
    drop(ass);
    Json(resp)
}

/// GET /models — List available models (internal format).
#[cfg_attr(feature = "server-openapi", utoipa::path(get, path = "/models", responses((status = 200, description = "List of available models"))))]
async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ass = state.assistant.lock().await;
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
    drop(ass);
    Json(serde_json::json!(models))
}

/// POST /chat — Send a message (non-streaming).
#[cfg_attr(feature = "server-openapi", utoipa::path(post, path = "/chat", responses((status = 200, description = "Chat response"))))]
async fn chat_handler(
    State(state): State<AppState>,
    Json(chat_req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    // Validate message length
    let max_len = {
        let config = state.server_config.read().await;
        config.max_message_length
    };
    if chat_req.message.len() > max_len {
        return Err(AppError::UnprocessableEntity(format!(
            "Message too long: {} characters (max {})",
            chat_req.message.len(),
            max_len,
        )));
    }

    // Run the blocking LLM call in spawn_blocking
    let assistant = state.assistant.clone();
    let message = chat_req.message;
    let system_prompt = chat_req.system_prompt;
    let knowledge_context = chat_req.knowledge_context;

    let result = tokio::task::spawn_blocking(move || {
        let mut ass = assistant.blocking_lock();
        ass.send_message_with_notes(
            message,
            &knowledge_context,
            &system_prompt,
            "",
        );

        let model = ass.config.selected_model.clone();
        loop {
            match ass.poll_response() {
                Some(crate::messages::AiResponse::Complete(text)) => {
                    return Ok(ChatResponse { content: text, model });
                }
                Some(crate::messages::AiResponse::Error(e)) => {
                    return Err(e);
                }
                Some(crate::messages::AiResponse::Cancelled(partial)) => {
                    return Ok(ChatResponse { content: partial, model });
                }
                Some(_) => continue,
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    })
    .await
    .map_err(|e| AppError::Internal(format!("Task join error: {}", e)))?;

    result
        .map(Json)
        .map_err(|e| AppError::Internal(format!("Generation error: {}", e)))
}

/// POST /chat/stream — SSE streaming with real token-by-token output.
///
/// Forwards `AiResponse::Chunk` events from the LLM provider as individual
/// SSE events, giving clients real-time token streaming instead of waiting
/// for the complete response.
///
/// ## SSE event format
/// - Token: `data: {"token":"Hello"}`
/// - Error: `data: {"error":"message"}`
/// - Done:  `data: [DONE]`
async fn chat_stream_handler(
    State(state): State<AppState>,
    Json(chat_req): Json<ChatRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>>, AppError> {
    // Validate message length
    let max_len = {
        let config = state.server_config.read().await;
        config.max_message_length
    };
    if chat_req.message.len() > max_len {
        return Err(AppError::UnprocessableEntity(format!(
            "Message too long: {} characters (max {})",
            chat_req.message.len(),
            max_len,
        )));
    }

    // Create channel to bridge blocking LLM thread → async SSE stream
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    // Spawn blocking LLM call — forwards Chunk events through channel
    let assistant = state.assistant.clone();
    let message = chat_req.message;
    let system_prompt = chat_req.system_prompt;
    let knowledge_context = chat_req.knowledge_context;

    tokio::task::spawn_blocking(move || {
        let mut ass = assistant.blocking_lock();
        ass.send_message_with_notes(message, &knowledge_context, &system_prompt, "");
        loop {
            match ass.poll_response() {
                Some(crate::messages::AiResponse::Chunk(token)) => {
                    if tx.blocking_send(StreamEvent::Token(token)).is_err() {
                        break; // receiver dropped (client disconnected)
                    }
                }
                Some(crate::messages::AiResponse::Complete(_)) => {
                    let _ = tx.blocking_send(StreamEvent::Done);
                    break;
                }
                Some(crate::messages::AiResponse::Error(e)) => {
                    let _ = tx.blocking_send(StreamEvent::Error(e));
                    break;
                }
                Some(crate::messages::AiResponse::Cancelled(_)) => {
                    let _ = tx.blocking_send(StreamEvent::Done);
                    break;
                }
                Some(_) => continue,
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    });

    // Async SSE stream — yields events as real tokens arrive
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(token) => {
                    let token_json = serde_json::json!({"token": token});
                    yield Ok(SseEvent::default().data(token_json.to_string()));
                }
                StreamEvent::Done => {
                    yield Ok(SseEvent::default().data("[DONE]"));
                    break;
                }
                StreamEvent::Error(e) => {
                    let err_json = serde_json::json!({"error": e});
                    yield Ok(SseEvent::default().data(err_json.to_string()));
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(SseKeepAlive::default()))
}

/// GET /config — Get current configuration.
async fn get_config_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ass = state.assistant.lock().await;
    let safe_config = serde_json::json!({
        "provider": format!("{:?}", ass.config.provider),
        "selected_model": ass.config.selected_model,
        "base_url": ass.config.get_base_url(),
        "temperature": ass.config.temperature,
        "max_history_messages": ass.config.max_history_messages,
        "has_api_key": !ass.config.api_key.is_empty(),
    });
    drop(ass);
    Json(safe_config)
}

/// POST /config — Update configuration.
async fn set_config_handler(
    State(state): State<AppState>,
    Json(updates): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut ass = state.assistant.lock().await;

    if let Some(model) = updates.get("model").and_then(|m| m.as_str()) {
        ass.config.selected_model = model.to_string();
    }
    if let Some(temp) = updates.get("temperature").and_then(|t| t.as_f64()) {
        ass.config.temperature = temp as f32;
    }

    drop(ass);
    Ok(Json(serde_json::json!({"status": "updated"})))
}

/// GET /metrics — Prometheus-style metrics.
#[cfg_attr(feature = "server-openapi", utoipa::path(get, path = "/metrics", responses((status = 200, description = "Prometheus metrics"))))]
async fn metrics_handler(State(state): State<AppState>) -> Response {
    let body = state.metrics.render_prometheus();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

/// GET /sessions — List active sessions.
#[cfg_attr(feature = "server-openapi", utoipa::path(get, path = "/sessions", responses((status = 200, description = "List of active sessions"))))]
async fn list_sessions_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ass = state.assistant.lock().await;
    let sessions: Vec<serde_json::Value> = ass
        .session_store
        .sessions
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "messages": s.messages.len(),
            })
        })
        .collect();
    drop(ass);
    Json(serde_json::json!(sessions))
}

/// GET /sessions/{id} — Get a specific session.
async fn get_session_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let ass = state.assistant.lock().await;
    match ass.session_store.sessions.iter().find(|s| s.id == id) {
        Some(session) => {
            let msgs: Vec<serde_json::Value> = session
                .messages
                .iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect();
            drop(ass);
            Ok(Json(serde_json::json!(msgs)))
        }
        None => {
            drop(ass);
            Err(AppError::NotFound(format!("Session not found: {}", id)))
        }
    }
}

/// DELETE /sessions/{id} — Delete a session.
async fn delete_session_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut ass = state.assistant.lock().await;
    let before = ass.session_store.sessions.len();
    ass.delete_session(&id);
    let deleted = ass.session_store.sessions.len() < before;
    drop(ass);

    if deleted {
        // Also remove from session affinity DashMap
        state.sessions.remove(&id);
        Ok(Json(serde_json::json!({"deleted": true})))
    } else {
        Err(AppError::NotFound(format!("Session not found: {}", id)))
    }
}

/// POST /v1/chat/completions — OpenAI-compatible chat completions (stream + non-stream).
async fn openai_completions_handler(
    State(state): State<AppState>,
    Json(oai_req): Json<OpenAIChatRequest>,
) -> Response {
    // Validate messages
    if oai_req.messages.is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
            "invalid_messages",
        );
    }

    // Extract system prompt and user message
    let system_prompt: String = oai_req
        .messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let user_message = oai_req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if user_message.is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "No user message found",
            "invalid_request_error",
            "missing_user_message",
        );
    }

    // Read config
    let config = state.server_config.read().await.clone();

    // Validate message length
    if user_message.len() > config.max_message_length {
        return openai_error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Message too long ({} chars, max {})",
                user_message.len(),
                config.max_message_length
            ),
            "invalid_request_error",
            "message_too_long",
        );
    }

    // -- Input guardrails --
    if config.enrichment.enable_guardrails {
        if let Some(ref pipeline) = state.guardrail_pipeline {
            let mut gp = pipeline.lock().unwrap_or_else(|e| e.into_inner());
            let result = gp.check_input(&user_message);
            drop(gp);
            if !result.passed && config.enrichment.block_on_input_violation {
                let guard_name = result.blocked_by.unwrap_or_else(|| "guardrail".to_string());
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("Request blocked by {}", guard_name),
                    "content_policy_violation",
                    "guardrail_blocked",
                );
            }
        }
    }

    // -- Cost budget pre-check --
    if let Some(ref bm) = state.budget_manager {
        let bm_guard = bm.lock().unwrap_or_else(|e| e.into_inner());
        if bm_guard.check(0.0).is_exceeded() {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "Budget exceeded",
                "rate_limit_error",
                "budget_exceeded",
            );
        }
    }

    // -- Virtual model resolution --
    let is_stream = oai_req.stream;
    let raw_model_name = oai_req.model.clone().unwrap_or_default();
    let resolution = state.model_registry.resolve(&raw_model_name);

    let (enrichment, requested_model, system_prompt) = match resolution {
        crate::virtual_model::ModelResolution::Virtual(ref vmodel) => {
            let mut sys = system_prompt;
            if let Some(ref vsp) = vmodel.system_prompt {
                if sys.is_empty() {
                    sys = vsp.clone();
                } else {
                    sys = format!("{}\n{}", vsp, sys);
                }
            }
            (vmodel.enrichment.clone(), Some(vmodel.base_model.clone()), sys)
        }
        crate::virtual_model::ModelResolution::Physical { ref name, .. } => {
            (config.enrichment.clone(), Some(name.clone()), system_prompt)
        }
        crate::virtual_model::ModelResolution::PassThrough { ref name } => {
            let model = if name.is_empty() { None } else { Some(name.clone()) };
            (config.enrichment.clone(), model, system_prompt)
        }
    };

    // -- RAG enrichment + Generate response --
    let assistant = state.assistant.clone();

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if is_stream {
        // ── Real streaming path: forward AiResponse::Chunk tokens in real-time ──
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

        let stream_user_msg = user_message.clone();
        let stream_sys_prompt = system_prompt.clone();

        // Spawn blocking enrichment + LLM call — sends chunks through channel
        tokio::task::spawn_blocking(move || {
            // RAG enrichment
            let mut knowledge_context = String::new();
            {
                let mut ass = assistant.blocking_lock();
                apply_enrichment_config(&mut ass, &enrichment);
                if enrichment.enable_rag {
                    let rconf = &enrichment.rag;
                    apply_rag_config(&mut ass, rconf);
                    let (knowledge, _conversation) = ass.build_rag_context(&stream_user_msg);
                    knowledge_context = knowledge;
                }
            }

            // Generate with streaming
            let mut ass = assistant.blocking_lock();
            ass.send_message_with_notes(
                stream_user_msg,
                &knowledge_context,
                &stream_sys_prompt,
                "",
            );
            loop {
                match ass.poll_response() {
                    Some(crate::messages::AiResponse::Chunk(token)) => {
                        if tx.blocking_send(StreamEvent::Token(token)).is_err() {
                            break;
                        }
                    }
                    Some(crate::messages::AiResponse::Complete(_)) => {
                        let _ = tx.blocking_send(StreamEvent::Done);
                        break;
                    }
                    Some(crate::messages::AiResponse::Error(e)) => {
                        let _ = tx.blocking_send(StreamEvent::Error(e));
                        break;
                    }
                    Some(crate::messages::AiResponse::Cancelled(_)) => {
                        let _ = tx.blocking_send(StreamEvent::Done);
                        break;
                    }
                    Some(_) => continue,
                    None => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        });

        let model = requested_model.unwrap_or_default();
        let id = format!("chatcmpl-{:016x}", created);

        // Async SSE stream — real tokens from the LLM provider
        let stream = async_stream::stream! {
            // Role announcement chunk
            let first_chunk = crate::openai_adapter::StreamChunk {
                id: Some(id.clone()),
                object: Some("chat.completion.chunk".to_string()),
                created: Some(created),
                model: Some(model.clone()),
                choices: vec![crate::openai_adapter::StreamChoice {
                    index: Some(0),
                    delta: crate::openai_adapter::StreamDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            yield Ok::<_, std::convert::Infallible>(
                SseEvent::default().data(serde_json::to_string(&first_chunk).unwrap_or_default())
            );

            // Real token chunks from LLM provider
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(token) => {
                        let chunk = crate::openai_adapter::StreamChunk {
                            id: Some(id.clone()),
                            object: Some("chat.completion.chunk".to_string()),
                            created: Some(created),
                            model: Some(model.clone()),
                            choices: vec![crate::openai_adapter::StreamChoice {
                                index: Some(0),
                                delta: crate::openai_adapter::StreamDelta {
                                    role: None,
                                    content: Some(token),
                                },
                                finish_reason: None,
                            }],
                        };
                        yield Ok(SseEvent::default().data(
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                    }
                    StreamEvent::Done => {
                        // finish_reason: "stop" chunk
                        let stop_chunk = crate::openai_adapter::StreamChunk {
                            id: Some(id.clone()),
                            object: Some("chat.completion.chunk".to_string()),
                            created: Some(created),
                            model: Some(model.clone()),
                            choices: vec![crate::openai_adapter::StreamChoice {
                                index: Some(0),
                                delta: crate::openai_adapter::StreamDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop".to_string()),
                            }],
                        };
                        yield Ok(SseEvent::default().data(
                            serde_json::to_string(&stop_chunk).unwrap_or_default()
                        ));
                        yield Ok(SseEvent::default().data("[DONE]"));
                        break;
                    }
                    StreamEvent::Error(e) => {
                        let err_chunk = serde_json::json!({"error": {"message": e, "type": "server_error"}});
                        yield Ok(SseEvent::default().data(err_chunk.to_string()));
                        break;
                    }
                }
            }
        };

        return Sse::new(stream).keep_alive(SseKeepAlive::default()).into_response();
    }

    // ── Non-streaming path (unchanged logic) ──
    let result = tokio::task::spawn_blocking(move || {
        // RAG enrichment
        let mut knowledge_context = String::new();
        {
            let mut ass = assistant.blocking_lock();
            apply_enrichment_config(&mut ass, &enrichment);
            if enrichment.enable_rag {
                let rconf = &enrichment.rag;
                apply_rag_config(&mut ass, rconf);
                let (knowledge, _conversation) = ass.build_rag_context(&user_message);
                knowledge_context = knowledge;
            }
        }

        // Generate response
        let mut ass = assistant.blocking_lock();
        let model = requested_model
            .unwrap_or_else(|| ass.config.selected_model.clone());

        ass.send_message_with_notes(user_message.clone(), &knowledge_context, &system_prompt, "");

        let response_text = loop {
            match ass.poll_response() {
                Some(crate::messages::AiResponse::Complete(text)) => break Ok(text),
                Some(crate::messages::AiResponse::Error(e)) => break Err(e),
                Some(crate::messages::AiResponse::Cancelled(partial)) => break Ok(partial),
                Some(_) => continue,
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        };

        response_text.map(|text| (text, model, user_message, system_prompt, knowledge_context))
    })
    .await;

    let (response_text, model, user_msg, sys_prompt, knowledge_ctx) = match result {
        Ok(Ok(tuple)) => tuple,
        Ok(Err(e)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Generation error: {}", e),
                "server_error",
                "generation_failed",
            );
        }
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Task join error: {}", e),
                "server_error",
                "internal_error",
            );
        }
    };

    // -- Output guardrails (only for non-streaming) --
    let final_text = if config.enrichment.enable_guardrails {
        if let Some(ref pipeline) = state.guardrail_pipeline {
            let mut gp = pipeline.lock().unwrap_or_else(|e| e.into_inner());
            let result = gp.check_output(&response_text);
            drop(gp);

            if !result.passed && config.enrichment.redact_output_pii {
                let gconf = &config.enrichment.guardrails;
                let action = if gconf.output_pii_action == "block" {
                    crate::guardrail_pipeline::PiiAction::Block
                } else {
                    crate::guardrail_pipeline::PiiAction::Redact(gconf.output_pii_redact_char)
                };
                let output_pii = crate::guardrail_pipeline::OutputPiiGuard::new(
                    crate::guardrail_pipeline::OutputPiiConfig {
                        action,
                        check_emails: gconf.output_pii_check_emails,
                        check_phones: gconf.output_pii_check_phones,
                        check_ssns: gconf.output_pii_check_ssns,
                        check_credit_cards: gconf.output_pii_check_credit_cards,
                        check_ip_addresses: gconf.output_pii_check_ip_addresses,
                    },
                );
                output_pii.redact(&response_text)
            } else {
                response_text
            }
        } else {
            response_text
        }
    } else {
        response_text
    };

    // Token estimation
    let prompt_tokens = crate::context::estimate_tokens(&user_msg)
        + crate::context::estimate_tokens(&sys_prompt)
        + crate::context::estimate_tokens(&knowledge_ctx);
    let completion_tokens = crate::context::estimate_tokens(&final_text);

    let id = format!("chatcmpl-{:016x}", created);

    // Return non-streaming response
    let response = crate::openai_adapter::OpenAIResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![crate::openai_adapter::OpenAIChoice {
            index: 0,
            message: crate::openai_adapter::OpenAIResponseMessage {
                role: "assistant".to_string(),
                content: Some(final_text),
                function_call: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(crate::openai_adapter::OpenAIUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
    };

    Json(response).into_response()
}

/// Apply enrichment sub-configs (compaction, thinking) to the assistant.
fn apply_enrichment_config(
    ass: &mut AiAssistant,
    enrichment: &crate::server::EnrichmentConfig,
) {
    if enrichment.compaction.enabled {
        let cconf = &enrichment.compaction;
        ass.set_compaction_config(crate::conversation_compaction::CompactionConfig {
            max_messages: cconf.max_messages,
            target_messages: cconf.target_messages,
            preserve_recent: cconf.preserve_recent,
            preserve_first: cconf.preserve_first,
            min_importance: cconf.min_importance,
        });
    }
    if enrichment.thinking.enabled {
        let tconf = &enrichment.thinking;
        ass.adaptive_thinking.enabled = true;
        ass.adaptive_thinking.min_depth = tconf.min_depth;
        ass.adaptive_thinking.max_depth = tconf.max_depth;
        ass.adaptive_thinking.inject_cot_instructions = tconf.inject_cot_instructions;
        ass.adaptive_thinking.parse_thinking_tags = tconf.parse_thinking_tags;
        ass.adaptive_thinking.strip_thinking_from_response = tconf.strip_thinking_from_response;
        ass.adaptive_thinking.adjust_temperature = tconf.adjust_temperature;
    }
}

/// Apply RAG enrichment config fields to the assistant.
fn apply_rag_config(
    ass: &mut AiAssistant,
    rconf: &crate::server::RagEnrichmentConfig,
) {
    ass.rag_config.knowledge_rag_enabled = rconf.knowledge_rag;
    ass.rag_config.conversation_rag_enabled = rconf.conversation_rag;
    ass.rag_config.max_knowledge_tokens = rconf.max_knowledge_tokens;
    ass.rag_config.max_conversation_tokens = rconf.max_conversation_tokens;
    ass.rag_config.top_k_chunks = rconf.top_k_chunks;
    ass.rag_config.min_relevance_score = rconf.min_relevance_score;
    ass.rag_config.dynamic_context_enabled = rconf.dynamic_context;
    ass.rag_config.auto_store_messages = rconf.auto_store_messages;
}

/// GET /v1/models — OpenAI-compatible model listing.
///
/// Returns published models from the ModelRegistry. If the registry has
/// published models, only those are shown. Otherwise falls back to the
/// raw available_models list (backward compatible).
async fn openai_models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ass = state.assistant.lock().await;
    let client_models = state.model_registry.list_client_visible(&ass.available_models);

    let data: Vec<serde_json::Value> = if client_models.is_empty() {
        // Fallback: no models published → show raw available models (backward compat)
        let models: Vec<serde_json::Value> = ass
            .available_models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": format!("{:?}", m.provider).to_lowercase(),
                })
            })
            .collect();
        if models.is_empty() {
            vec![serde_json::json!({
                "id": ass.config.selected_model,
                "object": "model",
                "created": 0,
                "owned_by": format!("{:?}", ass.config.provider).to_lowercase(),
            })]
        } else {
            models
        }
    } else {
        // Registry has published models → use them
        client_models
            .iter()
            .map(|m| serde_json::to_value(m).unwrap_or_default())
            .collect()
    };

    drop(ass);
    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

/// GET /openapi.json — OpenAPI specification.
#[cfg_attr(feature = "server-openapi", utoipa::path(get, path = "/openapi.json", responses((status = 200, description = "OpenAPI 3.0 specification"))))]
async fn openapi_handler() -> Json<serde_json::Value> {
    Json(crate::openapi_export::generate_server_api_spec())
}

/// GET /ws — WebSocket upgrade handler.
async fn ws_handler(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws_session(socket, state))
}

/// Handle a WebSocket session after upgrade with real token streaming.
///
/// ## Message format
/// - Client → Server: `{"message":"Hello","system_prompt":"...","knowledge_context":"..."}`
/// - Server → Client (token): `{"type":"chunk","content":"token"}`
/// - Server → Client (done):  `{"type":"complete","model":"llama3"}`
/// - Server → Client (error): `{"type":"error","error":"message"}`
async fn handle_ws_session(mut socket: WebSocket, state: AppState) {
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            WsMessage::Text(text) => {
                // Parse as ChatRequest
                let chat_req: ChatRequest = match serde_json::from_str(&text) {
                    Ok(r) => r,
                    Err(e) => {
                        let err_json = serde_json::json!({"type": "error", "error": format!("Invalid JSON: {}", e)});
                        let _ = socket.send(WsMessage::Text(err_json.to_string().into())).await;
                        continue;
                    }
                };

                // Create channel for streaming
                let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

                let assistant = state.assistant.clone();
                let message = chat_req.message;
                let system_prompt = chat_req.system_prompt;
                let knowledge_context = chat_req.knowledge_context;

                // Spawn blocking LLM call — sends chunks through channel
                tokio::task::spawn_blocking(move || {
                    let mut ass = assistant.blocking_lock();
                    ass.send_message_with_notes(
                        message,
                        &knowledge_context,
                        &system_prompt,
                        "",
                    );
                    loop {
                        match ass.poll_response() {
                            Some(crate::messages::AiResponse::Chunk(token)) => {
                                if tx.blocking_send(StreamEvent::Token(token)).is_err() {
                                    break;
                                }
                            }
                            Some(crate::messages::AiResponse::Complete(_)) => {
                                let _ = tx.blocking_send(StreamEvent::Done);
                                break;
                            }
                            Some(crate::messages::AiResponse::Error(e)) => {
                                let _ = tx.blocking_send(StreamEvent::Error(e));
                                break;
                            }
                            Some(crate::messages::AiResponse::Cancelled(_)) => {
                                let _ = tx.blocking_send(StreamEvent::Done);
                                break;
                            }
                            Some(_) => continue,
                            None => std::thread::sleep(Duration::from_millis(10)),
                        }
                    }
                });

                // Forward token events as individual WS messages
                let mut had_error = false;
                while let Some(event) = rx.recv().await {
                    let ws_msg = match event {
                        StreamEvent::Token(token) => {
                            serde_json::json!({"type": "chunk", "content": token})
                        }
                        StreamEvent::Done => {
                            let model = {
                                let ass = state.assistant.lock().await;
                                ass.config.selected_model.clone()
                            };
                            let msg = serde_json::json!({"type": "complete", "model": model});
                            if socket.send(WsMessage::Text(msg.to_string().into())).await.is_err() {
                                had_error = true;
                            }
                            break;
                        }
                        StreamEvent::Error(e) => {
                            serde_json::json!({"type": "error", "error": e})
                        }
                    };
                    if socket.send(WsMessage::Text(ws_msg.to_string().into())).await.is_err() {
                        had_error = true;
                        break;
                    }
                }
                if had_error {
                    break;
                }
            }
            WsMessage::Ping(data) => {
                if socket.send(WsMessage::Pong(data)).await.is_err() {
                    break;
                }
            }
            WsMessage::Close(_) => break,
            _ => {}
        }
    }
}

// ============================================================================
// Admin Endpoints — Virtual Models + Publish Control (v31 Phase 6)
// ============================================================================

/// Request body for creating a virtual model.
#[derive(Debug, Deserialize)]
struct CreateVirtualModelRequest {
    name: String,
    description: String,
    base_model: String,
    #[serde(default)]
    base_provider: Option<crate::config::AiProvider>,
    #[serde(default)]
    enrichment: Option<crate::server::EnrichmentConfig>,
    #[serde(default)]
    profile: Option<crate::profiles::ModelProfile>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    published: Option<bool>,
    #[serde(default)]
    tags: Option<Vec<String>>,
}

/// Request body for publishing a physical model.
#[derive(Debug, Deserialize)]
struct PublishModelRequest {
    #[serde(default)]
    provider: Option<crate::config::AiProvider>,
    #[serde(default)]
    display_name: Option<String>,
}

/// POST /admin/virtual-models — Create a virtual model.
async fn admin_create_virtual_model(
    State(state): State<AppState>,
    Json(req): Json<CreateVirtualModelRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let config = state.server_config.read().await;
    let default_enrichment = config.enrichment.clone();
    drop(config);

    let vmodel = crate::virtual_model::VirtualModel {
        name: req.name,
        description: req.description,
        base_model: req.base_model,
        base_provider: req.base_provider,
        enrichment: req.enrichment.unwrap_or(default_enrichment),
        profile: req.profile,
        system_prompt: req.system_prompt,
        published: req.published.unwrap_or(false),
        created_at: created,
        tags: req.tags.unwrap_or_default(),
    };

    let name = vmodel.name.clone();
    state
        .model_registry
        .register_virtual(vmodel)
        .map_err(|e| AppError::BadRequest(e))?;

    Ok(Json(serde_json::json!({"created": name})))
}

/// GET /admin/virtual-models — List all virtual models (including unpublished).
async fn admin_list_virtual_models(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let models = state.model_registry.list_virtual();
    Json(serde_json::json!({"virtual_models": models}))
}

/// GET /admin/virtual-models/{name} — Get a specific virtual model.
async fn admin_get_virtual_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let model = state
        .model_registry
        .get_virtual(&name)
        .ok_or_else(|| AppError::NotFound(format!("Virtual model '{}' not found", name)))?;
    Ok(Json(serde_json::to_value(&model).unwrap_or_default()))
}

/// PUT /admin/virtual-models/{name} — Update a virtual model.
async fn admin_update_virtual_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<CreateVirtualModelRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Get the existing model to preserve defaults
    let existing = state
        .model_registry
        .get_virtual(&name)
        .ok_or_else(|| AppError::NotFound(format!("Virtual model '{}' not found", name)))?;

    let updated = crate::virtual_model::VirtualModel {
        name: name.clone(),
        description: req.description,
        base_model: req.base_model,
        base_provider: req.base_provider.or(existing.base_provider),
        enrichment: req.enrichment.unwrap_or(existing.enrichment),
        profile: req.profile.or(existing.profile),
        system_prompt: req.system_prompt.or(existing.system_prompt),
        published: req.published.unwrap_or(existing.published),
        created_at: existing.created_at,
        tags: req.tags.unwrap_or(existing.tags),
    };

    if state.model_registry.update_virtual(updated) {
        Ok(Json(serde_json::json!({"updated": name})))
    } else {
        Err(AppError::NotFound(format!("Virtual model '{}' not found", name)))
    }
}

/// DELETE /admin/virtual-models/{name} — Delete a virtual model.
async fn admin_delete_virtual_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if state.model_registry.unregister_virtual(&name) {
        Ok(Json(serde_json::json!({"deleted": name})))
    } else {
        Err(AppError::NotFound(format!("Virtual model '{}' not found", name)))
    }
}

/// GET /admin/models — List all models with their publish status.
async fn admin_list_models(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ass = state.assistant.lock().await;
    let physical: Vec<serde_json::Value> = ass
        .available_models
        .iter()
        .map(|m| {
            let published = state.model_registry.is_published(&m.name);
            let info = state.model_registry.get_published(&m.name);
            serde_json::json!({
                "name": m.name,
                "provider": format!("{:?}", m.provider),
                "published": published,
                "display_name": info.and_then(|i| i.display_name),
                "type": "physical",
            })
        })
        .collect();
    drop(ass);

    let virtual_models: Vec<serde_json::Value> = state
        .model_registry
        .list_virtual()
        .iter()
        .map(|v| {
            serde_json::json!({
                "name": v.name,
                "base_model": v.base_model,
                "published": v.published,
                "description": v.description,
                "type": "virtual",
            })
        })
        .collect();

    Json(serde_json::json!({
        "physical": physical,
        "virtual": virtual_models,
    }))
}

/// POST /admin/models/{name}/publish — Publish a physical model.
async fn admin_publish_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Option<Json<PublishModelRequest>>,
) -> Json<serde_json::Value> {
    let req = body.map(|b| b.0);
    let provider = req
        .as_ref()
        .and_then(|r| r.provider.clone())
        .unwrap_or(crate::config::AiProvider::Ollama);
    let display_name = req.as_ref().and_then(|r| r.display_name.clone());

    state.model_registry.set_published_with_display_name(
        &name,
        provider,
        true,
        display_name,
    );

    Json(serde_json::json!({"published": name}))
}

/// POST /admin/models/{name}/unpublish — Unpublish a physical model.
async fn admin_unpublish_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Json<serde_json::Value> {
    state.model_registry.set_published(
        &name,
        crate::config::AiProvider::Ollama,
        false,
    );
    Json(serde_json::json!({"unpublished": name}))
}

// ============================================================================
// MCP Docker Handler (containers + tools)
// ============================================================================

#[cfg(all(feature = "containers", feature = "tools"))]
async fn mcp_handler(
    State(state): State<AppState>,
    body: String,
) -> impl IntoResponse {
    match &state.mcp_server {
        Some(server) => {
            let guard = match server.read() {
                Ok(g) => g,
                Err(_) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"error":"MCP server lock poisoned"}"#.to_string(),
                    );
                }
            };
            let response = guard.handle_message(&body);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                response,
            )
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "application/json")],
            r#"{"error":"MCP Docker tools not enabled (Docker unavailable)"}"#.to_string(),
        ),
    }
}

// ============================================================================
// AxumServer — Entrypoint + Graceful Shutdown (Phase 5)
// ============================================================================

/// Production-ready axum HTTP server.
///
/// Wraps `AppState` and manages the server lifecycle including
/// TLS, graceful shutdown, and background tasks.
pub struct AxumServer {
    config: ServerConfig,
    state: AppState,
}

impl AxumServer {
    /// Create a new AxumServer with default AiAssistant.
    pub fn new(config: ServerConfig) -> Self {
        let config = init_enrichment(config);
        let state = build_app_state(config.clone(), AiAssistant::new());
        Self { config, state }
    }

    /// Create a new AxumServer with a pre-configured AiAssistant.
    pub fn with_assistant(config: ServerConfig, assistant: AiAssistant) -> Self {
        let config = init_enrichment(config);
        let state = build_app_state(config.clone(), assistant);
        Self { config, state }
    }

    /// Get a reference to the server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Get a clone of the AppState (for testing or external use).
    pub fn state(&self) -> AppState {
        self.state.clone()
    }

    /// Run the server, binding to the configured address.
    ///
    /// This method blocks until a shutdown signal (SIGINT/SIGTERM or Ctrl+C) is received.
    /// On Windows, only Ctrl+C is supported.
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = self.config.bind_address();
        let app = build_router(self.state.clone(), &self.config);

        // Spawn background rate limiter cleanup task
        if let Some(ref limiter) = self.state.rate_limiter {
            let limiter = limiter.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(60));
                loop {
                    interval.tick().await;
                    limiter.cleanup();
                }
            });
        }

        log::info!("AI Assistant axum server listening on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app.into_make_service())
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        log::info!("Server shut down gracefully");
        Ok(())
    }

    /// Run the server with TLS (requires `server-axum-tls` feature).
    #[cfg(feature = "server-axum-tls")]
    pub async fn run_tls(
        self,
        cert_path: &str,
        key_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use axum_server::tls_rustls::RustlsConfig;

        let addr = self.config.bind_address();
        let app = build_router(self.state.clone(), &self.config);

        // Spawn background rate limiter cleanup task
        if let Some(ref limiter) = self.state.rate_limiter {
            let limiter = limiter.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(60));
                loop {
                    interval.tick().await;
                    limiter.cleanup();
                }
            });
        }

        let tls_config = RustlsConfig::from_pem_file(cert_path, key_path).await?;

        log::info!("AI Assistant axum server listening on https://{}", addr);

        let addr: std::net::SocketAddr = addr.parse()?;
        axum_server::bind_rustls(addr, tls_config)
            .serve(app.into_make_service())
            .await?;

        Ok(())
    }
}

/// Build AppState from config and assistant.
fn build_app_state(config: ServerConfig, assistant: AiAssistant) -> AppState {
    let rate_limiter = config.rate_limiter.as_ref().map(|rl| {
        Arc::new(AxumRateLimiter::new(rl.requests_per_minute, 60))
    });

    AppState {
        assistant: Arc::new(tokio::sync::Mutex::new(assistant)),
        guardrail_pipeline: config.guardrail_pipeline.clone(),
        budget_manager: config.budget_manager.clone(),
        server_config: Arc::new(tokio::sync::RwLock::new(config)),
        sessions: Arc::new(DashMap::new()),
        metrics: Arc::new(AxumServerMetrics::new()),
        rate_limiter,
        model_registry: Arc::new(crate::virtual_model::ModelRegistry::new()),
        #[cfg(feature = "server-cluster")]
        cluster: None,
        #[cfg(all(feature = "containers", feature = "tools"))]
        mcp_server: build_mcp_docker_server(),
    }
}

/// Build an MCP server with Docker tools if Docker is available.
#[cfg(all(feature = "containers", feature = "tools"))]
fn build_mcp_docker_server() -> Option<Arc<std::sync::RwLock<crate::mcp_protocol::McpServer>>> {
    use crate::container_executor::{ContainerConfig, ContainerExecutor};

    if !ContainerExecutor::is_docker_available() {
        log::info!("MCP Docker tools: Docker not available, skipping");
        return None;
    }

    let executor = match ContainerExecutor::new(ContainerConfig::default()) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("MCP Docker tools: executor init failed: {}", e);
            return None;
        }
    };

    let mut mcp = crate::mcp_protocol::McpServer::new(
        "ai_assistant_docker",
        env!("CARGO_PKG_VERSION"),
    );
    let exec_arc = std::sync::Arc::new(std::sync::RwLock::new(executor));
    crate::mcp_docker_tools::register_mcp_docker_tools(&mut mcp, exec_arc);

    log::info!("MCP Docker tools: 8 tools registered on /mcp endpoint");
    Some(Arc::new(std::sync::RwLock::new(mcp)))
}

/// Initialize enrichment subsystems (guardrails, budget manager).
fn init_enrichment(config: ServerConfig) -> ServerConfig {
    let config = init_guardrail_pipeline(config);
    init_budget_manager(config)
}

/// Build a `GuardrailPipeline` if enrichment.enable_guardrails is set.
fn init_guardrail_pipeline(mut config: ServerConfig) -> ServerConfig {
    if config.enrichment.enable_guardrails && config.guardrail_pipeline.is_none() {
        use crate::guardrail_pipeline::{
            AttackGuard, ContentLengthGuard, GuardrailPipeline, OutputPiiConfig,
            OutputPiiGuard, OutputToxicityConfig, OutputToxicityGuard, PatternGuard,
            PiiAction, PiiGuard, RateLimitGuard, ToxicityGuard,
        };

        let gconf = &config.enrichment.guardrails;
        let mut pipeline =
            GuardrailPipeline::new().with_threshold(config.enrichment.guardrail_threshold as f64);

        if gconf.attack_guard {
            pipeline.add_guard(Box::new(AttackGuard::new()));
        }
        if gconf.pii_guard {
            pipeline.add_guard(Box::new(PiiGuard::new()));
        }
        if gconf.toxicity_guard {
            pipeline.add_guard(Box::new(ToxicityGuard::new()));
        }
        if gconf.content_length_guard {
            pipeline.add_guard(Box::new(ContentLengthGuard::new(config.max_message_length)));
        }
        if gconf.rate_limit_guard {
            pipeline.add_guard(Box::new(RateLimitGuard::new(
                gconf.rate_limit_max_requests,
                gconf.rate_limit_window_secs,
            )));
        }
        if !gconf.blocked_patterns.is_empty() {
            pipeline.add_guard(Box::new(PatternGuard::new(gconf.blocked_patterns.clone())));
        }
        if gconf.output_pii_guard {
            let action = if gconf.output_pii_action == "block" {
                PiiAction::Block
            } else {
                PiiAction::Redact(gconf.output_pii_redact_char)
            };
            pipeline.add_guard(Box::new(OutputPiiGuard::new(OutputPiiConfig {
                action,
                check_emails: gconf.output_pii_check_emails,
                check_phones: gconf.output_pii_check_phones,
                check_ssns: gconf.output_pii_check_ssns,
                check_credit_cards: gconf.output_pii_check_credit_cards,
                check_ip_addresses: gconf.output_pii_check_ip_addresses,
            })));
        }
        if gconf.output_toxicity_guard {
            pipeline.add_guard(Box::new(OutputToxicityGuard::new(OutputToxicityConfig {
                severity_threshold: gconf.output_toxicity_threshold,
            })));
        }

        config.guardrail_pipeline = Some(Arc::new(Mutex::new(pipeline)));
    }
    config
}

/// Build a `BudgetManager` if enrichment.cost.enabled is set.
fn init_budget_manager(mut config: ServerConfig) -> ServerConfig {
    if config.enrichment.cost.enabled && config.budget_manager.is_none() {
        let mut bm = crate::cost::BudgetManager::new();
        if let Some(dl) = config.enrichment.cost.daily_limit {
            bm = bm.with_daily_limit(dl);
        }
        if let Some(ml) = config.enrichment.cost.monthly_limit {
            bm = bm.with_monthly_limit(ml);
        }
        if let Some(rl) = config.enrichment.cost.per_request_limit {
            bm = bm.with_request_limit(rl);
        }
        config.budget_manager = Some(Arc::new(Mutex::new(bm)));
    }
    config
}

/// Shutdown signal handler (Ctrl+C / SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => log::info!("Received Ctrl+C, shutting down..."),
        _ = terminate => log::info!("Received SIGTERM, shutting down..."),
    }
}

// ============================================================================
// Tests (Phases 2-6)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::EnrichmentConfig;

    // ── Phase 2: AppState Tests ──────────────────────────────────────────

    #[test]
    fn test_app_state_construction() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);
        assert!(state.sessions.is_empty());
        assert!(state.rate_limiter.is_none());
    }

    #[test]
    fn test_app_state_with_rate_limiter() {
        let mut config = ServerConfig::default();
        config.rate_limiter = Some(Arc::new(crate::server::ServerRateLimiter::new(100)));
        let state = build_app_state(config, AiAssistant::new());
        assert!(state.rate_limiter.is_some());
    }

    #[test]
    fn test_session_data_creation() {
        let session = SessionData {
            id: "test-session".to_string(),
            last_active: 1000,
            message_count: 5,
            affinity_node: None,
        };
        assert_eq!(session.id, "test-session");
        assert_eq!(session.message_count, 5);
        assert!(session.affinity_node.is_none());
    }

    #[test]
    fn test_session_data_with_affinity() {
        let session = SessionData {
            id: "s1".to_string(),
            last_active: 2000,
            message_count: 3,
            affinity_node: Some("node-1".to_string()),
        };
        assert_eq!(session.affinity_node, Some("node-1".to_string()));
    }

    #[test]
    fn test_dashmap_session_crud() {
        let sessions: DashMap<String, SessionData> = DashMap::new();

        // Create
        sessions.insert("s1".to_string(), SessionData {
            id: "s1".to_string(),
            last_active: 0,
            message_count: 0,
            affinity_node: None,
        });
        assert_eq!(sessions.len(), 1);

        // Read
        assert!(sessions.get("s1").is_some());
        assert!(sessions.get("s2").is_none());

        // Update
        sessions.entry("s1".to_string()).and_modify(|s| s.message_count = 10);
        assert_eq!(sessions.get("s1").unwrap().message_count, 10);

        // Delete
        sessions.remove("s1");
        assert!(sessions.is_empty());
    }

    // ── Phase 2: Metrics Tests ───────────────────────────────────────────

    #[test]
    fn test_metrics_new() {
        let metrics = AxumServerMetrics::new();
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_2xx.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_4xx.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_5xx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_record_2xx() {
        let metrics = AxumServerMetrics::new();
        metrics.record_request(200, Duration::from_millis(50));
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_2xx.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_4xx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_record_4xx() {
        let metrics = AxumServerMetrics::new();
        metrics.record_request(404, Duration::from_millis(5));
        metrics.record_request(400, Duration::from_millis(3));
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.requests_4xx.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_metrics_record_5xx() {
        let metrics = AxumServerMetrics::new();
        metrics.record_request(500, Duration::from_millis(100));
        assert_eq!(metrics.requests_5xx.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_request_id_uniqueness() {
        let metrics = AxumServerMetrics::new();
        let id1 = metrics.generate_request_id();
        let id2 = metrics.generate_request_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_metrics_request_id_format() {
        let metrics = AxumServerMetrics::new();
        let id = metrics.generate_request_id();
        assert!(id.contains('-'));
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_metrics_prometheus_format() {
        let metrics = AxumServerMetrics::new();
        metrics.record_request(200, Duration::from_millis(10));
        let output = metrics.render_prometheus();
        assert!(output.contains("ai_axum_requests_total 1"));
        assert!(output.contains("ai_axum_requests_2xx 1"));
        assert!(output.contains("ai_axum_uptime_seconds"));
    }

    #[test]
    fn test_metrics_prometheus_avg_duration() {
        let metrics = AxumServerMetrics::new();
        metrics.record_request(200, Duration::from_millis(100));
        metrics.record_request(200, Duration::from_millis(200));
        let output = metrics.render_prometheus();
        assert!(output.contains("ai_axum_request_duration_avg_ms"));
    }

    #[test]
    fn test_metrics_endpoint_counts() {
        let metrics = AxumServerMetrics::new();
        metrics.endpoint_counts.insert("GET /health".to_string(), 5);
        metrics.endpoint_counts.entry("GET /health".to_string()).and_modify(|c| *c += 1);
        assert_eq!(*metrics.endpoint_counts.get("GET /health").unwrap(), 6);
    }

    // ── Phase 2: Rate Limiter Tests ──────────────────────────────────────

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let limiter = AxumRateLimiter::new(5, 60);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        for _ in 0..5 {
            assert!(limiter.check(ip));
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let limiter = AxumRateLimiter::new(3, 60);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(limiter.check(ip));
        assert!(limiter.check(ip));
        assert!(limiter.check(ip));
        assert!(!limiter.check(ip)); // 4th request blocked
    }

    #[test]
    fn test_rate_limiter_different_ips_independent() {
        let limiter = AxumRateLimiter::new(2, 60);
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(limiter.check(ip1));
        assert!(limiter.check(ip1));
        assert!(!limiter.check(ip1)); // ip1 blocked
        assert!(limiter.check(ip2));  // ip2 still allowed
    }

    #[test]
    fn test_rate_limiter_retry_after() {
        let limiter = AxumRateLimiter::new(1, 60);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        limiter.check(ip);
        limiter.check(ip); // blocked
        let retry = limiter.retry_after(ip);
        assert!(retry > 0);
        assert!(retry <= 61);
    }

    #[test]
    fn test_rate_limiter_cleanup() {
        let limiter = AxumRateLimiter::new(100, 1); // 1-second window
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        limiter.check(ip);
        assert!(!limiter.entries.is_empty());
        std::thread::sleep(Duration::from_millis(1100));
        limiter.cleanup();
        assert!(limiter.entries.is_empty());
    }

    // ── Phase 2: AppError Tests ──────────────────────────────────────────

    #[test]
    fn test_app_error_variants() {
        // Just ensure they can be constructed (IntoResponse tested via axum)
        let _ = AppError::BadRequest("bad".to_string());
        let _ = AppError::Unauthorized("unauth".to_string());
        let _ = AppError::NotFound("not found".to_string());
        let _ = AppError::UnprocessableEntity("invalid".to_string());
        let _ = AppError::TooManyRequests { message: "slow down".to_string(), retry_after: 60 };
        let _ = AppError::Internal("oops".to_string());
        let _ = AppError::ServiceUnavailable("down".to_string());
    }

    // ── Phase 3: Middleware Tests ─────────────────────────────────────────

    #[test]
    fn test_build_cors_layer_wildcard() {
        let config = CorsConfig::default();
        let _layer = build_cors_layer(&config); // should not panic
    }

    #[test]
    fn test_build_cors_layer_specific_origins() {
        let config = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string()],
            allowed_headers: vec!["Content-Type".to_string()],
            max_age_secs: 3600,
            allow_credentials: true,
        };
        let _layer = build_cors_layer(&config); // should not panic
    }

    // ── Phase 4: Request/Response Type Tests ─────────────────────────────

    #[test]
    fn test_chat_request_deserialization() {
        let json = r#"{"message": "Hello"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hello");
        assert_eq!(req.system_prompt, "");
        assert_eq!(req.knowledge_context, "");
    }

    #[test]
    fn test_chat_request_with_all_fields() {
        let json = r#"{"message": "Hi", "system_prompt": "You are helpful", "knowledge_context": "docs"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hi");
        assert_eq!(req.system_prompt, "You are helpful");
        assert_eq!(req.knowledge_context, "docs");
    }

    #[test]
    fn test_chat_response_serialization() {
        let resp = ChatResponse {
            content: "Hello!".to_string(),
            model: "test-model".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Hello!"));
        assert!(json.contains("test-model"));
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            model: "test".to_string(),
            provider: "ollama".to_string(),
            uptime_secs: 60,
            active_sessions: 2,
            conversation_messages: 10,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["status"], "ok");
        assert_eq!(parsed["uptime_secs"], 60);
    }

    #[test]
    fn test_openai_chat_request_deserialization() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "stream": false
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, Some("gpt-4".to_string()));
        assert_eq!(req.messages.len(), 2);
        assert!(!req.stream);
    }

    #[test]
    fn test_openai_chat_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.model.is_none());
        assert!(req.temperature.is_none());
        assert!(!req.stream);
    }

    #[test]
    fn test_openai_chat_request_stream_true() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}], "stream": true}"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_error_body_serialization() {
        let body = ErrorBody { error: "test error".to_string() };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_openai_error_body_serialization() {
        let body = OpenAIErrorBody {
            error: OpenAIErrorDetail {
                message: "msg".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: "bad".to_string(),
            },
        };
        let json = serde_json::to_string(&body).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["error"]["message"], "msg");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
    }

    // ── Phase 5: Server Construction Tests ───────────────────────────────

    #[test]
    fn test_axum_server_new() {
        let config = ServerConfig::default();
        let server = AxumServer::new(config);
        assert_eq!(server.config().host, "127.0.0.1");
        assert_eq!(server.config().port, 8090);
    }

    #[test]
    fn test_axum_server_with_assistant() {
        let config = ServerConfig::default();
        let assistant = AiAssistant::new();
        let server = AxumServer::with_assistant(config, assistant);
        assert_eq!(server.config().port, 8090);
    }

    #[test]
    fn test_axum_server_custom_port() {
        let config = ServerConfig {
            port: 3000,
            ..Default::default()
        };
        let server = AxumServer::new(config);
        assert_eq!(server.config().port, 3000);
    }

    #[test]
    fn test_axum_server_state_accessible() {
        let config = ServerConfig::default();
        let server = AxumServer::new(config);
        let state = server.state();
        assert!(state.sessions.is_empty());
    }

    #[test]
    fn test_axum_server_enrichment_init_guardrails() {
        let config = ServerConfig {
            enrichment: EnrichmentConfig {
                enable_guardrails: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let server = AxumServer::new(config);
        assert!(server.state().guardrail_pipeline.is_some());
    }

    #[test]
    fn test_axum_server_enrichment_init_budget() {
        let config = ServerConfig {
            enrichment: EnrichmentConfig {
                cost: crate::server::CostEnrichmentConfig {
                    enabled: true,
                    daily_limit: Some(10.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let server = AxumServer::new(config);
        assert!(server.state().budget_manager.is_some());
    }

    // ── Phase 6: Session Affinity Tests ──────────────────────────────────

    #[test]
    fn test_session_affinity_set_and_get() {
        let mgr = SessionAffinityManager::new(3600);
        mgr.set_node("s1", 2);
        assert_eq!(mgr.get_node("s1"), Some(2));
    }

    #[test]
    fn test_session_affinity_unknown_session() {
        let mgr = SessionAffinityManager::new(3600);
        assert_eq!(mgr.get_node("unknown"), None);
    }

    #[test]
    fn test_session_affinity_multiple_sessions() {
        let mgr = SessionAffinityManager::new(3600);
        mgr.set_node("s1", 0);
        mgr.set_node("s2", 1);
        mgr.set_node("s3", 2);
        assert_eq!(mgr.get_node("s1"), Some(0));
        assert_eq!(mgr.get_node("s2"), Some(1));
        assert_eq!(mgr.get_node("s3"), Some(2));
    }

    #[test]
    fn test_session_affinity_reassign() {
        let mgr = SessionAffinityManager::new(3600);
        mgr.set_node("s1", 0);
        assert_eq!(mgr.get_node("s1"), Some(0));
        mgr.set_node("s1", 3);
        assert_eq!(mgr.get_node("s1"), Some(3));
    }

    #[test]
    fn test_session_affinity_touch() {
        let mgr = SessionAffinityManager::new(3600);
        mgr.set_node("s1", 0);
        mgr.touch("s1");
        assert_eq!(mgr.get_node("s1"), Some(0));
    }

    #[test]
    fn test_session_affinity_expiry() {
        let mgr = SessionAffinityManager::new(1); // 1-second TTL
        mgr.set_node("s1", 0);
        assert_eq!(mgr.get_node("s1"), Some(0));
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(mgr.get_node("s1"), None); // expired
    }

    #[test]
    fn test_session_affinity_cleanup() {
        let mgr = SessionAffinityManager::new(1); // 1-second TTL
        mgr.set_node("s1", 0);
        mgr.set_node("s2", 1);
        std::thread::sleep(Duration::from_millis(1100));
        mgr.cleanup_expired();
        assert!(mgr.affinity.is_empty());
        assert!(mgr.last_access.is_empty());
    }

    // ── Phase 3: Router Tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_build_router_does_not_panic() {
        let config = ServerConfig::default();
        let state = build_app_state(config.clone(), AiAssistant::new());
        let _router = build_router(state, &config);
    }

    // ── Phase 4: Handler Integration Tests (using axum test utilities) ───

    #[tokio::test]
    async fn test_health_handler_returns_ok() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = health_handler(State(state)).await;
        assert_eq!(resp.status, "ok");
        assert!(!resp.version.is_empty());
    }

    #[tokio::test]
    async fn test_models_handler_returns_array() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = models_handler(State(state)).await;
        assert!(resp.is_array());
    }

    #[tokio::test]
    async fn test_get_config_handler() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = get_config_handler(State(state)).await;
        assert!(resp["provider"].is_string());
        assert!(resp["temperature"].is_number());
        // API key should not be exposed
        assert!(resp.get("api_key").is_none());
    }

    #[tokio::test]
    async fn test_set_config_handler_update_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let updates = serde_json::json!({"model": "test-model", "temperature": 0.5});
        let result = set_config_handler(State(state.clone()), Json(updates)).await;
        assert!(result.is_ok());

        let ass = state.assistant.lock().await;
        assert_eq!(ass.config.selected_model, "test-model");
        assert!((ass.config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_metrics_handler_prometheus_format() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        state.metrics.record_request(200, Duration::from_millis(10));

        let resp = metrics_handler(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_sessions_handler_empty() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = list_sessions_handler(State(state)).await;
        assert!(resp.is_array());
    }

    #[tokio::test]
    async fn test_get_session_not_found() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let result = get_session_handler(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_session_not_found() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let result = delete_session_handler(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_openai_models_handler() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = openai_models_handler(State(state)).await;
        assert_eq!(resp["object"], "list");
        assert!(resp["data"].is_array());
    }

    #[tokio::test]
    async fn test_openapi_handler_returns_spec() {
        let Json(resp) = openapi_handler().await;
        assert!(resp.get("openapi").is_some() || resp.get("paths").is_some());
    }

    // ── Phase 4: Chat Handler Error Tests ────────────────────────────────

    #[tokio::test]
    async fn test_chat_handler_message_too_long() {
        let config = ServerConfig {
            max_message_length: 10,
            ..Default::default()
        };
        let state = build_app_state(config, AiAssistant::new());
        let req = ChatRequest {
            message: "a".repeat(100),
            system_prompt: String::new(),
            knowledge_context: String::new(),
            model: None,
        };
        let result = chat_handler(State(state), Json(req)).await;
        assert!(result.is_err());
    }

    // ── Phase 5: Init Functions Tests ────────────────────────────────────

    #[test]
    fn test_init_enrichment_no_guardrails() {
        let config = ServerConfig::default();
        let config = init_enrichment(config);
        assert!(config.guardrail_pipeline.is_none());
        assert!(config.budget_manager.is_none());
    }

    #[test]
    fn test_init_enrichment_with_guardrails() {
        let config = ServerConfig {
            enrichment: EnrichmentConfig {
                enable_guardrails: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let config = init_enrichment(config);
        assert!(config.guardrail_pipeline.is_some());
    }

    #[test]
    fn test_init_enrichment_with_budget() {
        let config = ServerConfig {
            enrichment: EnrichmentConfig {
                cost: crate::server::CostEnrichmentConfig {
                    enabled: true,
                    daily_limit: Some(50.0),
                    monthly_limit: Some(500.0),
                    per_request_limit: Some(1.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let config = init_enrichment(config);
        assert!(config.budget_manager.is_some());
    }

    #[test]
    fn test_init_guardrail_pipeline_idempotent() {
        let config = ServerConfig {
            enrichment: EnrichmentConfig {
                enable_guardrails: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let config = init_guardrail_pipeline(config);
        assert!(config.guardrail_pipeline.is_some());
        // Second call should not replace existing pipeline
        let config = init_guardrail_pipeline(config);
        assert!(config.guardrail_pipeline.is_some());
    }

    // ── Phase 6: SessionId Tests ─────────────────────────────────────────

    #[test]
    fn test_session_id_struct() {
        let sid = SessionId("test-123".to_string());
        assert_eq!(sid.0, "test-123");
    }

    // ── Misc Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_client_ip_fallback() {
        let req = axum::http::Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let ip = extract_client_ip(&req);
        assert_eq!(ip, IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    }

    #[test]
    fn test_extract_client_ip_xff() {
        let req = axum::http::Request::builder()
            .uri("/test")
            .header("x-forwarded-for", "192.168.1.100, 10.0.0.1")
            .body(Body::empty())
            .unwrap();
        let ip = extract_client_ip(&req);
        assert_eq!(ip, "192.168.1.100".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn test_extract_client_ip_xri() {
        let req = axum::http::Request::builder()
            .uri("/test")
            .header("x-real-ip", "10.0.0.50")
            .body(Body::empty())
            .unwrap();
        let ip = extract_client_ip(&req);
        assert_eq!(ip, "10.0.0.50".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn test_server_config_bind_address() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 3000,
            ..Default::default()
        };
        assert_eq!(config.bind_address(), "0.0.0.0:3000");
    }

    #[test]
    fn test_metrics_default_impl() {
        let metrics = AxumServerMetrics::default();
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_rate_limiter_zero_window() {
        let limiter = AxumRateLimiter::new(10, 0);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        // With zero window, all timestamps are "expired"
        assert!(limiter.check(ip));
    }

    #[test]
    fn test_openai_error_body_format() {
        let body = OpenAIErrorBody {
            error: OpenAIErrorDetail {
                message: "test".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: "test_code".to_string(),
            },
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["error"]["message"], "test");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "test_code");
    }

    // ── Multi-threaded safety tests ──────────────────────────────────────

    #[test]
    fn test_metrics_concurrent_writes() {
        let metrics = Arc::new(AxumServerMetrics::new());
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = metrics.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        m.record_request(200, Duration::from_millis(1));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_rate_limiter_concurrent_access() {
        let limiter = Arc::new(AxumRateLimiter::new(1000, 60));
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let l = limiter.clone();
                std::thread::spawn(move || {
                    let ip: IpAddr = format!("10.0.0.{}", i).parse().unwrap();
                    for _ in 0..50 {
                        l.check(ip);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_session_affinity_concurrent_access() {
        let mgr = Arc::new(SessionAffinityManager::new(3600));
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = mgr.clone();
                std::thread::spawn(move || {
                    let id = format!("session-{}", i);
                    m.set_node(&id, i);
                    assert_eq!(m.get_node(&id), Some(i));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    // ── v31: StreamEvent Tests ──────────────────────────────────────────

    #[test]
    fn test_stream_event_token() {
        let event = StreamEvent::Token("Hello".to_string());
        match event {
            StreamEvent::Token(t) => assert_eq!(t, "Hello"),
            _ => panic!("Expected Token"),
        }
    }

    #[test]
    fn test_stream_event_done() {
        let event = StreamEvent::Done;
        assert!(matches!(event, StreamEvent::Done));
    }

    #[test]
    fn test_stream_event_error() {
        let event = StreamEvent::Error("fail".to_string());
        match event {
            StreamEvent::Error(e) => assert_eq!(e, "fail"),
            _ => panic!("Expected Error"),
        }
    }

    // ── v31: Real Streaming Channel Tests ───────────────────────────────

    #[tokio::test]
    async fn test_stream_channel_sends_tokens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        tokio::task::spawn_blocking(move || {
            tx.blocking_send(StreamEvent::Token("Hello".to_string())).unwrap();
            tx.blocking_send(StreamEvent::Token(" world".to_string())).unwrap();
            tx.blocking_send(StreamEvent::Done).unwrap();
        });

        let mut tokens = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(t) => tokens.push(t),
                StreamEvent::Done => break,
                StreamEvent::Error(_) => panic!("Unexpected error"),
            }
        }
        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    #[tokio::test]
    async fn test_stream_channel_sends_error() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        tokio::task::spawn_blocking(move || {
            tx.blocking_send(StreamEvent::Token("partial".to_string())).unwrap();
            tx.blocking_send(StreamEvent::Error("connection lost".to_string())).unwrap();
        });

        let mut got_error = false;
        let mut got_token = false;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(_) => got_token = true,
                StreamEvent::Error(e) => {
                    assert_eq!(e, "connection lost");
                    got_error = true;
                    break;
                }
                StreamEvent::Done => break,
            }
        }
        assert!(got_token);
        assert!(got_error);
    }

    #[tokio::test]
    async fn test_stream_channel_receiver_dropped() {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        drop(rx); // Simulate client disconnect

        let result = tokio::task::spawn_blocking(move || {
            tx.blocking_send(StreamEvent::Token("test".to_string()))
        }).await.unwrap();

        assert!(result.is_err()); // Channel closed
    }

    #[tokio::test]
    async fn test_stream_channel_multiple_tokens_ordering() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        let expected: Vec<String> = (0..10).map(|i| format!("token_{}", i)).collect();
        let expected_clone = expected.clone();

        tokio::task::spawn_blocking(move || {
            for token in expected_clone {
                tx.blocking_send(StreamEvent::Token(token)).unwrap();
            }
            tx.blocking_send(StreamEvent::Done).unwrap();
        });

        let mut received = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(t) => received.push(t),
                StreamEvent::Done => break,
                _ => {}
            }
        }
        assert_eq!(received, expected);
    }

    // ── v31: ChatRequest model field Tests ──────────────────────────────

    #[test]
    fn test_chat_request_with_model() {
        let json = r#"{"message": "Hello", "model": "my-rag-assistant"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hello");
        assert_eq!(req.model, Some("my-rag-assistant".to_string()));
    }

    #[test]
    fn test_chat_request_without_model() {
        let json = r#"{"message": "Hello"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.model.is_none());
    }

    // ── v31: Enrichment helper Tests ────────────────────────────────────

    #[test]
    fn test_apply_enrichment_config_compaction() {
        let mut ass = AiAssistant::new();
        let enrichment = EnrichmentConfig {
            compaction: crate::server::CompactionEnrichmentConfig {
                enabled: true,
                max_messages: 50,
                target_messages: 30,
                preserve_recent: 5,
                preserve_first: 2,
                min_importance: 0.5,
            },
            ..Default::default()
        };
        apply_enrichment_config(&mut ass, &enrichment);
        // Verify compaction was applied (compaction config is internal)
        // Just verify no panic
    }

    #[test]
    fn test_apply_enrichment_config_thinking() {
        let mut ass = AiAssistant::new();
        let enrichment = EnrichmentConfig {
            thinking: crate::server::ThinkingEnrichmentConfig {
                enabled: true,
                inject_cot_instructions: true,
                parse_thinking_tags: true,
                strip_thinking_from_response: true,
                adjust_temperature: false,
                ..Default::default()
            },
            ..Default::default()
        };
        apply_enrichment_config(&mut ass, &enrichment);
        assert!(ass.adaptive_thinking.enabled);
        assert!(ass.adaptive_thinking.inject_cot_instructions);
    }

    #[test]
    fn test_apply_rag_config() {
        let mut ass = AiAssistant::new();
        let rconf = crate::server::RagEnrichmentConfig {
            knowledge_rag: true,
            conversation_rag: false,
            max_knowledge_tokens: 2000,
            max_conversation_tokens: 1000,
            top_k_chunks: 5,
            min_relevance_score: 0.8,
            dynamic_context: true,
            auto_store_messages: false,
        };
        apply_rag_config(&mut ass, &rconf);
        assert!(ass.rag_config.knowledge_rag_enabled);
        assert!(!ass.rag_config.conversation_rag_enabled);
        assert_eq!(ass.rag_config.max_knowledge_tokens, 2000);
        assert_eq!(ass.rag_config.top_k_chunks, 5);
    }

    // ── v31: WebSocket streaming format Tests ───────────────────────────

    #[test]
    fn test_ws_chunk_message_format() {
        let msg = serde_json::json!({"type": "chunk", "content": "Hello"});
        assert_eq!(msg["type"], "chunk");
        assert_eq!(msg["content"], "Hello");
    }

    #[test]
    fn test_ws_complete_message_format() {
        let msg = serde_json::json!({"type": "complete", "model": "llama3:8b"});
        assert_eq!(msg["type"], "complete");
        assert_eq!(msg["model"], "llama3:8b");
    }

    #[test]
    fn test_ws_error_message_format() {
        let msg = serde_json::json!({"type": "error", "error": "timeout"});
        assert_eq!(msg["type"], "error");
        assert_eq!(msg["error"], "timeout");
    }

    // ── v31: OpenAI streaming chunk format Tests ────────────────────────

    #[test]
    fn test_openai_stream_chunk_role_announcement() {
        let chunk = crate::openai_adapter::StreamChunk {
            id: Some("chatcmpl-test".to_string()),
            object: Some("chat.completion.chunk".to_string()),
            created: Some(1000),
            model: Some("test-model".to_string()),
            choices: vec![crate::openai_adapter::StreamChoice {
                index: Some(0),
                delta: crate::openai_adapter::StreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["delta"]["role"], "assistant");
        assert!(json["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn test_openai_stream_chunk_content_token() {
        let chunk = crate::openai_adapter::StreamChunk {
            id: Some("chatcmpl-test".to_string()),
            object: Some("chat.completion.chunk".to_string()),
            created: Some(1000),
            model: Some("test-model".to_string()),
            choices: vec![crate::openai_adapter::StreamChoice {
                index: Some(0),
                delta: crate::openai_adapter::StreamDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_openai_stream_chunk_stop() {
        let chunk = crate::openai_adapter::StreamChunk {
            id: Some("chatcmpl-test".to_string()),
            object: Some("chat.completion.chunk".to_string()),
            created: Some(1000),
            model: Some("test-model".to_string()),
            choices: vec![crate::openai_adapter::StreamChoice {
                index: Some(0),
                delta: crate::openai_adapter::StreamDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    // ── v31 Phase 5: Virtual Model Resolution Tests ─────────────────────

    #[test]
    fn test_app_state_has_model_registry() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        // Registry should be empty by default
        assert!(state.model_registry.list_virtual().is_empty());
    }

    #[test]
    fn test_model_registry_virtual_registration() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let vmodel = crate::virtual_model::VirtualModel {
            name: "my-rag".to_string(),
            description: "RAG assistant".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: Some("You are helpful.".to_string()),
            published: true,
            created_at: 0,
            tags: vec![],
        };
        state.model_registry.register_virtual(vmodel).unwrap();
        assert_eq!(state.model_registry.list_virtual().len(), 1);
    }

    #[test]
    fn test_model_resolution_virtual_enrichment() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());

        let mut enrichment = EnrichmentConfig::default();
        enrichment.enable_rag = true;
        enrichment.rag.knowledge_rag = true;
        enrichment.rag.top_k_chunks = 10;

        let vmodel = crate::virtual_model::VirtualModel {
            name: "rag-model".to_string(),
            description: "RAG model".to_string(),
            base_model: "gpt-4o".to_string(),
            base_provider: None,
            enrichment,
            profile: None,
            system_prompt: None,
            published: true,
            created_at: 0,
            tags: vec![],
        };
        state.model_registry.register_virtual(vmodel).unwrap();

        let resolution = state.model_registry.resolve("rag-model");
        match resolution {
            crate::virtual_model::ModelResolution::Virtual(v) => {
                assert_eq!(v.base_model, "gpt-4o");
                assert!(v.enrichment.enable_rag);
                assert_eq!(v.enrichment.rag.top_k_chunks, 10);
            }
            _ => panic!("Expected Virtual"),
        }
    }

    #[test]
    fn test_model_resolution_system_prompt_prepend() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());

        let vmodel = crate::virtual_model::VirtualModel {
            name: "custom".to_string(),
            description: "Custom".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: Some("Always respond in Spanish.".to_string()),
            published: true,
            created_at: 0,
            tags: vec![],
        };
        state.model_registry.register_virtual(vmodel).unwrap();

        let resolution = state.model_registry.resolve("custom");
        if let crate::virtual_model::ModelResolution::Virtual(v) = resolution {
            // Simulate what the handler does
            let user_sys = "Be concise.";
            let sys = format!("{}\n{}", v.system_prompt.unwrap(), user_sys);
            assert!(sys.contains("Always respond in Spanish."));
            assert!(sys.contains("Be concise."));
        } else {
            panic!("Expected Virtual");
        }
    }

    #[test]
    fn test_model_resolution_passthrough_empty() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let resolution = state.model_registry.resolve("");
        assert!(matches!(
            resolution,
            crate::virtual_model::ModelResolution::PassThrough { .. }
        ));
    }

    #[tokio::test]
    async fn test_openai_models_handler_fallback() {
        // No published models → should fall back to available_models
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = openai_models_handler(State(state)).await;
        assert_eq!(resp["object"], "list");
        assert!(resp["data"].is_array());
    }

    #[tokio::test]
    async fn test_openai_models_handler_with_published() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());

        // Register a published virtual model
        let vmodel = crate::virtual_model::VirtualModel {
            name: "my-model".to_string(),
            description: "Test".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: None,
            published: true,
            created_at: 1000,
            tags: vec!["test".to_string()],
        };
        state.model_registry.register_virtual(vmodel).unwrap();

        let Json(resp) = openai_models_handler(State(state)).await;
        let data = resp["data"].as_array().unwrap();
        assert!(!data.is_empty());
        // Should contain our virtual model
        let has_virtual = data.iter().any(|m| m["id"] == "my-model");
        assert!(has_virtual);
    }

    #[test]
    fn test_model_publish_controls_visibility() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());

        // Publish one physical model
        state.model_registry.set_published(
            "llama3:8b",
            crate::config::AiProvider::Ollama,
            true,
        );

        let models = vec![
            crate::models::ModelInfo {
                name: "llama3:8b".to_string(),
                provider: crate::config::AiProvider::Ollama,
                size: None,
                modified_at: None,
                capabilities: None,
            },
            crate::models::ModelInfo {
                name: "mistral:7b".to_string(),
                provider: crate::config::AiProvider::Ollama,
                size: None,
                modified_at: None,
                capabilities: None,
            },
        ];

        let visible = state.model_registry.list_client_visible(&models);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "llama3:8b");
    }

    // ── v31 Phase 6: Admin Endpoint Tests ───────────────────────────────

    #[tokio::test]
    async fn test_admin_create_virtual_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let req = CreateVirtualModelRequest {
            name: "test-model".to_string(),
            description: "A test model".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: None,
            profile: None,
            system_prompt: Some("Be helpful.".to_string()),
            published: Some(true),
            tags: Some(vec!["test".to_string()]),
        };
        let result = admin_create_virtual_model(State(state.clone()), Json(req)).await;
        assert!(result.is_ok());
        let Json(resp) = result.unwrap();
        assert_eq!(resp["created"], "test-model");

        // Verify it exists
        assert!(state.model_registry.get_virtual("test-model").is_some());
    }

    #[tokio::test]
    async fn test_admin_create_duplicate_virtual_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let req = CreateVirtualModelRequest {
            name: "dup".to_string(),
            description: "First".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: None,
            profile: None,
            system_prompt: None,
            published: None,
            tags: None,
        };
        admin_create_virtual_model(State(state.clone()), Json(req)).await.unwrap();

        let req2 = CreateVirtualModelRequest {
            name: "dup".to_string(),
            description: "Second".to_string(),
            base_model: "gpt-4o".to_string(),
            base_provider: None,
            enrichment: None,
            profile: None,
            system_prompt: None,
            published: None,
            tags: None,
        };
        let result = admin_create_virtual_model(State(state), Json(req2)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_admin_list_virtual_models() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        state.model_registry.register_virtual(crate::virtual_model::VirtualModel {
            name: "m1".to_string(),
            description: "M1".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: None,
            published: true,
            created_at: 0,
            tags: vec![],
        }).unwrap();

        let Json(resp) = admin_list_virtual_models(State(state)).await;
        let models = resp["virtual_models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
    }

    #[tokio::test]
    async fn test_admin_get_virtual_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        state.model_registry.register_virtual(crate::virtual_model::VirtualModel {
            name: "my-model".to_string(),
            description: "Test".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: None,
            published: true,
            created_at: 0,
            tags: vec![],
        }).unwrap();

        let result = admin_get_virtual_model(State(state), Path("my-model".to_string())).await;
        assert!(result.is_ok());
        let Json(resp) = result.unwrap();
        assert_eq!(resp["name"], "my-model");
    }

    #[tokio::test]
    async fn test_admin_get_virtual_model_not_found() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let result = admin_get_virtual_model(State(state), Path("nope".to_string())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_admin_delete_virtual_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        state.model_registry.register_virtual(crate::virtual_model::VirtualModel {
            name: "del-me".to_string(),
            description: "Delete me".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: None,
            published: false,
            created_at: 0,
            tags: vec![],
        }).unwrap();

        let result = admin_delete_virtual_model(State(state.clone()), Path("del-me".to_string())).await;
        assert!(result.is_ok());
        assert!(state.model_registry.get_virtual("del-me").is_none());
    }

    #[tokio::test]
    async fn test_admin_delete_virtual_model_not_found() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let result = admin_delete_virtual_model(State(state), Path("nope".to_string())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_admin_publish_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let Json(resp) = admin_publish_model(
            State(state.clone()),
            Path("llama3:8b".to_string()),
            None,
        ).await;
        assert_eq!(resp["published"], "llama3:8b");
        assert!(state.model_registry.is_published("llama3:8b"));
    }

    #[tokio::test]
    async fn test_admin_unpublish_model() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        state.model_registry.set_published("llama3:8b", crate::config::AiProvider::Ollama, true);
        assert!(state.model_registry.is_published("llama3:8b"));

        let Json(resp) = admin_unpublish_model(State(state.clone()), Path("llama3:8b".to_string())).await;
        assert_eq!(resp["unpublished"], "llama3:8b");
        assert!(!state.model_registry.is_published("llama3:8b"));
    }

    #[tokio::test]
    async fn test_admin_list_models() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());

        // Add a virtual model
        state.model_registry.register_virtual(crate::virtual_model::VirtualModel {
            name: "v1".to_string(),
            description: "V1".to_string(),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: None,
            published: true,
            created_at: 0,
            tags: vec![],
        }).unwrap();

        let Json(resp) = admin_list_models(State(state)).await;
        assert!(resp["physical"].is_array());
        assert!(resp["virtual"].is_array());
        let virtuals = resp["virtual"].as_array().unwrap();
        assert_eq!(virtuals.len(), 1);
        assert_eq!(virtuals[0]["name"], "v1");
    }

    #[tokio::test]
    async fn test_admin_publish_with_display_name() {
        let config = ServerConfig::default();
        let state = build_app_state(config, AiAssistant::new());
        let req = PublishModelRequest {
            provider: Some(crate::config::AiProvider::Ollama),
            display_name: Some("My Llama".to_string()),
        };
        admin_publish_model(
            State(state.clone()),
            Path("llama3:8b".to_string()),
            Some(Json(req)),
        ).await;

        let info = state.model_registry.get_published("llama3:8b").unwrap();
        assert_eq!(info.display_name, Some("My Llama".to_string()));
    }
}
