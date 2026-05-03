// Agent Client Protocol (ACP) — server implementation.
//
// ACP is the open JSON-RPC-2.0-over-stdio protocol used by editors
// (Zed, VS Code, JetBrains) to drive embedded coding agents
// (OpenHands, Goose, Hermes, Gemini, …). Spec: https://agentclientprotocol.com.
//
// We implement protocol version 1, server side only. Wire format and method
// names follow the spec verbatim so any conformant client can drive us.
//
// Pluggable execution: the server is decoupled from the LLM via a callback
// (see `AcpServer::with_llm`). The same pattern as `recipes::RecipeEngine` and
// the V89 CoVe verifier — keeps the protocol layer testable without spinning
// up a real model.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Protocol version we implement. Bumped only on breaking changes.
pub const PROTOCOL_VERSION: u16 = 1;

/// Hard caps to keep the trust surface small.
#[derive(Debug, Clone)]
pub struct AcpServerConfig {
    /// Maximum bytes of a single inbound JSON-RPC frame.
    pub max_frame_bytes: usize,
    /// Maximum simultaneous live sessions.
    pub max_sessions: usize,
    /// Per-session idle timeout (no activity → session evicted).
    pub session_idle: Duration,
    /// Generation timeout per `session/prompt`.
    pub prompt_timeout: Duration,
    /// SLO targets — informational; surfaced via `SloRecord`.
    pub slo_handshake: Duration,
    pub slo_first_chunk: Duration,
    pub slo_min_chunks_per_sec: f64,
}

impl Default for AcpServerConfig {
    fn default() -> Self {
        Self {
            max_frame_bytes: 4 * 1024 * 1024,
            max_sessions: 32,
            session_idle: Duration::from_secs(15 * 60),
            prompt_timeout: Duration::from_secs(120),
            slo_handshake: Duration::from_millis(200),
            slo_first_chunk: Duration::from_secs(1),
            slo_min_chunks_per_sec: 30.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Wire envelope (JSON-RPC 2.0)
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Num(i64),
    Str(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: RequestId,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: RequestId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self {
            code: -32700,
            message: msg.into(),
            data: None,
        }
    }
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: msg.into(),
            data: None,
        }
    }
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("Method not found: {}", method),
            data: None,
        }
    }
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: msg.into(),
            data: None,
        }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: msg.into(),
            data: None,
        }
    }
    pub fn auth_required() -> Self {
        Self {
            code: -32000,
            message: "Authentication required".into(),
            data: None,
        }
    }
    pub fn resource_not_found(msg: impl Into<String>) -> Self {
        Self {
            code: -32002,
            message: msg.into(),
            data: None,
        }
    }
}

/// Inbound frame parsed from a single newline-delimited JSON line.
#[derive(Debug, Clone)]
pub enum Inbound {
    Request(Request),
    Notification(Notification),
}

pub fn parse_frame(line: &str) -> Result<Inbound, RpcError> {
    let v: Value = serde_json::from_str(line)
        .map_err(|e| RpcError::parse_error(format!("invalid JSON: {}", e)))?;
    let obj = v
        .as_object()
        .ok_or_else(|| RpcError::invalid_request("JSON-RPC frame must be an object"))?;
    if obj.get("jsonrpc").and_then(|j| j.as_str()) != Some("2.0") {
        return Err(RpcError::invalid_request(
            "missing or wrong 'jsonrpc' (must be \"2.0\")",
        ));
    }
    if obj.contains_key("id") {
        let req: Request = serde_json::from_value(v)
            .map_err(|e| RpcError::invalid_request(format!("bad request shape: {}", e)))?;
        Ok(Inbound::Request(req))
    } else {
        let n: Notification = serde_json::from_value(v)
            .map_err(|e| RpcError::invalid_request(format!("bad notification shape: {}", e)))?;
        Ok(Inbound::Notification(n))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ACP domain types (subset — what we need for prompt/stream/cancel)
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientFsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: ClientFsCapabilities,
    #[serde(default)]
    pub terminal: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    pub name: Option<String>,
    pub title: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    #[serde(default)]
    pub image: bool,
    #[serde(default)]
    pub audio: bool,
    #[serde(default)]
    pub embedded_context: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpCapabilities {
    #[serde(default)]
    pub http: bool,
    #[serde(default)]
    pub sse: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    #[serde(default)]
    pub load_session: bool,
    #[serde(default)]
    pub prompt_capabilities: PromptCapabilities,
    #[serde(default)]
    pub mcp_capabilities: McpCapabilities,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentInfo {
    pub name: Option<String>,
    pub title: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    pub protocol_version: u16,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    #[serde(default)]
    pub client_info: Option<ClientInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResponse {
    pub protocol_version: u16,
    pub agent_capabilities: AgentCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_info: Option<AgentInfo>,
    #[serde(default)]
    pub auth_methods: Vec<Value>,
}

/// `ContentBlock` — tagged on `type`. Baseline mandatory: `text`, `resource_link`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        annotations: Option<Value>,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
    },
    Audio {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    ResourceLink {
        uri: String,
        name: String,
        #[serde(default, rename = "mimeType", skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    Resource {
        resource: Value,
    },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock::Text {
            text: s.into(),
            annotations: None,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        if let ContentBlock::Text { text, .. } = self {
            Some(text.as_str())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionRequest {
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptRequest {
    pub session_id: String,
    pub prompt: Vec<ContentBlock>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResponse {
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

/// `SessionUpdate` — tagged on `sessionUpdate` (NOT `type`, per spec).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    UserMessageChunk { content: ContentBlock },
    AgentMessageChunk { content: ContentBlock },
    AgentThoughtChunk { content: ContentBlock },
}

// ────────────────────────────────────────────────────────────────────────────
// SLO recording — fed to auditor binaries
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloRecord {
    pub kind: String,
    pub session_id: Option<String>,
    pub method: Option<String>,
    pub elapsed_ms: u64,
    pub chunks: u64,
    pub chunks_per_sec: f64,
    pub timestamp_ms: u128,
}

// ────────────────────────────────────────────────────────────────────────────
// LLM callback contract
// ────────────────────────────────────────────────────────────────────────────

/// Streaming chunk emitted by the LLM callback.
pub enum AcpChunk {
    Delta(String),
    Done,
    Error(String),
}

/// Cancellation flag handed to the LLM callback. The callback should poll
/// `is_cancelled()` between chunks and stop emission promptly.
#[derive(Clone, Default)]
pub struct CancelToken(pub Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Streaming sink the LLM callback writes into. Server reads from the
/// matching receiver and emits ACP `session/update` notifications.
pub type ChunkSender = std::sync::mpsc::Sender<AcpChunk>;
pub type ChunkReceiver = std::sync::mpsc::Receiver<AcpChunk>;

/// LLM callback signature.
///
/// Args: full prompt text, cancellation token, sender for streaming chunks.
/// The callback runs on a worker thread spawned by the server. It MUST
/// eventually emit either `AcpChunk::Done` or `AcpChunk::Error`, even if
/// cancellation was observed (in which case `Done` is fine — the server
/// translates to `StopReason::Cancelled`).
pub type LlmFn = dyn Fn(String, CancelToken, ChunkSender) + Send + Sync + 'static;

// ────────────────────────────────────────────────────────────────────────────
// Server state
// ────────────────────────────────────────────────────────────────────────────

struct SessionState {
    cancel: CancelToken,
    last_activity: Instant,
}

/// External sink for SLO records — fires once per record, in order.
/// Used by `ai_acp serve` to persist records to a JSONL file as they are
/// produced. `Send + Sync` because the server is shared across threads.
pub type SloSink = dyn Fn(&SloRecord) + Send + Sync + 'static;

pub struct AcpServer {
    config: AcpServerConfig,
    agent_info: Option<AgentInfo>,
    capabilities: AgentCapabilities,
    llm: Option<Arc<LlmFn>>,
    slo_log: Arc<Mutex<Vec<SloRecord>>>,
    slo_sink: Option<Arc<SloSink>>,
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    next_session: AtomicU64,
    initialized: AtomicBool,
}

impl AcpServer {
    pub fn new(config: AcpServerConfig) -> Self {
        let mut caps = AgentCapabilities::default();
        // We accept embedded resources in prompts; the spec calls this
        // capability `embeddedContext`. Image/audio default off until
        // wired through the multimodal pipeline.
        caps.prompt_capabilities.embedded_context = true;
        Self {
            config,
            agent_info: Some(AgentInfo {
                name: Some("ai_assistant".into()),
                title: Some("ai_assistant ACP server".into()),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
            capabilities: caps,
            llm: None,
            slo_log: Arc::new(Mutex::new(Vec::new())),
            slo_sink: None,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session: AtomicU64::new(1),
            initialized: AtomicBool::new(false),
        }
    }

    pub fn with_llm<F>(mut self, f: F) -> Self
    where
        F: Fn(String, CancelToken, ChunkSender) + Send + Sync + 'static,
    {
        self.llm = Some(Arc::new(f));
        self
    }

    pub fn with_capabilities(mut self, caps: AgentCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Install a callback fired once per SLO record. Records are still kept
    /// in-memory (`slo_records()`); the sink is purely additive.
    pub fn with_slo_sink<F>(mut self, f: F) -> Self
    where
        F: Fn(&SloRecord) + Send + Sync + 'static,
    {
        self.slo_sink = Some(Arc::new(f));
        self
    }

    pub fn slo_records(&self) -> Vec<SloRecord> {
        self.slo_log.lock().map(|g| g.clone()).unwrap_or_default()
    }

    fn record_slo(&self, rec: SloRecord) {
        if let Some(sink) = &self.slo_sink {
            sink(&rec);
        }
        if let Ok(mut g) = self.slo_log.lock() {
            g.push(rec);
        }
    }

    fn evict_stale(&self) {
        if let Ok(mut sessions) = self.sessions.lock() {
            let now = Instant::now();
            let idle = self.config.session_idle;
            sessions.retain(|_, s| now.duration_since(s.last_activity) < idle);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Method handlers — return either result JSON or an RpcError
// ────────────────────────────────────────────────────────────────────────────

fn handle_initialize(server: &AcpServer, params: Option<Value>) -> Result<Value, RpcError> {
    let started = Instant::now();
    let req: InitializeRequest = match params {
        Some(p) => serde_json::from_value(p)
            .map_err(|e| RpcError::invalid_params(format!("bad initialize params: {}", e)))?,
        None => return Err(RpcError::invalid_params("initialize requires params")),
    };
    // Version negotiation: client sends its preferred version. We echo it if
    // we support it; otherwise we return our own (the client may then drop).
    let chosen = if req.protocol_version == PROTOCOL_VERSION {
        PROTOCOL_VERSION
    } else {
        PROTOCOL_VERSION
    };
    let resp = InitializeResponse {
        protocol_version: chosen,
        agent_capabilities: server.capabilities.clone(),
        agent_info: server.agent_info.clone(),
        auth_methods: vec![],
    };
    server.initialized.store(true, Ordering::SeqCst);
    let elapsed = started.elapsed();
    server.record_slo(SloRecord {
        kind: "handshake".into(),
        session_id: None,
        method: Some("initialize".into()),
        elapsed_ms: elapsed.as_millis() as u64,
        chunks: 0,
        chunks_per_sec: 0.0,
        timestamp_ms: ts_now_ms(),
    });
    Ok(serde_json::to_value(resp).expect("serialize initialize response"))
}

fn handle_session_new(server: &AcpServer, params: Option<Value>) -> Result<Value, RpcError> {
    if !server.initialized.load(Ordering::SeqCst) {
        return Err(RpcError::invalid_request(
            "initialize must precede session/new",
        ));
    }
    let req: NewSessionRequest = match params {
        Some(p) => serde_json::from_value(p)
            .map_err(|e| RpcError::invalid_params(format!("bad session/new params: {}", e)))?,
        None => return Err(RpcError::invalid_params("session/new requires params")),
    };
    if !std::path::Path::new(&req.cwd).is_absolute() {
        return Err(RpcError::invalid_params("cwd must be an absolute path"));
    }
    server.evict_stale();
    let id = format!(
        "sess_{}",
        server.next_session.fetch_add(1, Ordering::SeqCst)
    );
    let cap = server.config.max_sessions;
    if let Ok(mut sessions) = server.sessions.lock() {
        if sessions.len() >= cap {
            return Err(RpcError::internal(format!(
                "session cap reached ({} live)",
                cap
            )));
        }
        sessions.insert(
            id.clone(),
            SessionState {
                cancel: CancelToken::new(),
                last_activity: Instant::now(),
            },
        );
    }
    Ok(json!({ "sessionId": id }))
}

/// Drive a `session/prompt`: run the LLM, stream `agent_message_chunk`s as
/// `session/update` notifications, then return `{ stopReason }`. The writer
/// is shared with the dispatcher loop and locked per-frame.
fn handle_session_prompt(
    server: &AcpServer,
    params: Option<Value>,
    out: &Arc<Mutex<Box<dyn Write + Send>>>,
) -> Result<Value, RpcError> {
    if !server.initialized.load(Ordering::SeqCst) {
        return Err(RpcError::invalid_request(
            "initialize must precede session/prompt",
        ));
    }
    let req: PromptRequest = match params {
        Some(p) => serde_json::from_value(p)
            .map_err(|e| RpcError::invalid_params(format!("bad session/prompt params: {}", e)))?,
        None => return Err(RpcError::invalid_params("session/prompt requires params")),
    };
    let cancel = {
        let mut sessions = server
            .sessions
            .lock()
            .map_err(|_| RpcError::internal("sessions mutex poisoned"))?;
        let s = sessions.get_mut(&req.session_id).ok_or_else(|| {
            RpcError::resource_not_found(format!("unknown session {}", req.session_id))
        })?;
        s.last_activity = Instant::now();
        s.cancel.0.store(false, Ordering::SeqCst);
        s.cancel.clone()
    };
    let llm = server
        .llm
        .clone()
        .ok_or_else(|| RpcError::internal("ACP server has no LLM callback wired"))?;

    // Concatenate text content blocks. Image/audio handling deferred until
    // the relevant prompt capabilities are advertised + wired.
    let prompt_text = req
        .prompt
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n");

    let (tx, rx): (ChunkSender, ChunkReceiver) = std::sync::mpsc::channel();
    let cancel_for_llm = cancel.clone();
    std::thread::spawn(move || {
        llm(prompt_text, cancel_for_llm, tx);
    });

    let started = Instant::now();
    let mut first_chunk: Option<Instant> = None;
    let mut chunks: u64 = 0;
    let timeout = server.config.prompt_timeout;
    let session_id = req.session_id.clone();

    let mut stop = StopReason::EndTurn;
    loop {
        if cancel.is_cancelled() {
            stop = StopReason::Cancelled;
            break;
        }
        if started.elapsed() > timeout {
            stop = StopReason::MaxTurnRequests;
            cancel.cancel();
            break;
        }
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(AcpChunk::Delta(text)) => {
                if first_chunk.is_none() {
                    first_chunk = Some(Instant::now());
                }
                chunks += 1;
                let upd = SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::text(text),
                };
                let note = Notification {
                    jsonrpc: "2.0".into(),
                    method: "session/update".into(),
                    params: Some(json!({
                        "sessionId": session_id,
                        "update": upd,
                    })),
                };
                if let Err(e) = write_notification(out, &note) {
                    let _ = e;
                    stop = StopReason::Refusal;
                    cancel.cancel();
                    break;
                }
            }
            Ok(AcpChunk::Done) => break,
            Ok(AcpChunk::Error(_)) => {
                stop = StopReason::Refusal;
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if cancel.is_cancelled() {
                    stop = StopReason::Cancelled;
                }
                break;
            }
        }
    }

    let elapsed = started.elapsed();
    let cps = if elapsed.as_secs_f64() > 0.0 {
        chunks as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    server.record_slo(SloRecord {
        kind: "prompt".into(),
        session_id: Some(session_id.clone()),
        method: Some("session/prompt".into()),
        elapsed_ms: elapsed.as_millis() as u64,
        chunks,
        chunks_per_sec: cps,
        timestamp_ms: ts_now_ms(),
    });
    if let Some(fc) = first_chunk {
        server.record_slo(SloRecord {
            kind: "first_chunk".into(),
            session_id: Some(session_id.clone()),
            method: Some("session/prompt".into()),
            elapsed_ms: fc.duration_since(started).as_millis() as u64,
            chunks: 1,
            chunks_per_sec: 0.0,
            timestamp_ms: ts_now_ms(),
        });
    }

    if let Ok(mut sessions) = server.sessions.lock() {
        if let Some(s) = sessions.get_mut(&session_id) {
            s.last_activity = Instant::now();
        }
    }

    Ok(serde_json::to_value(PromptResponse { stop_reason: stop })
        .expect("serialize prompt response"))
}

fn handle_session_cancel(server: &AcpServer, params: Option<Value>) {
    let p: CancelParams = match params {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(_) => return,
        },
        None => return,
    };
    if let Ok(sessions) = server.sessions.lock() {
        if let Some(s) = sessions.get(&p.session_id) {
            s.cancel.cancel();
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dispatch + framed I/O
// ────────────────────────────────────────────────────────────────────────────

fn write_notification(
    out: &Arc<Mutex<Box<dyn Write + Send>>>,
    n: &Notification,
) -> std::io::Result<()> {
    let mut g = out
        .lock()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "writer mutex poisoned"))?;
    let s = serde_json::to_string(n)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    if s.contains('\n') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame contains newline (NDJSON violation)",
        ));
    }
    g.write_all(s.as_bytes())?;
    g.write_all(b"\n")?;
    g.flush()?;
    Ok(())
}

fn write_response(out: &Arc<Mutex<Box<dyn Write + Send>>>, r: &Response) -> std::io::Result<()> {
    let mut g = out
        .lock()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "writer mutex poisoned"))?;
    let s = serde_json::to_string(r)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    if s.contains('\n') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame contains newline (NDJSON violation)",
        ));
    }
    g.write_all(s.as_bytes())?;
    g.write_all(b"\n")?;
    g.flush()?;
    Ok(())
}

/// Pump frames from `reader` into the server until EOF. Responses and
/// streaming notifications go to `writer`. Returns when the reader is
/// closed or a fatal frame error occurs.
pub fn serve<R, W>(server: AcpServer, mut reader: R, writer: W) -> std::io::Result<()>
where
    R: BufRead,
    W: Write + Send + 'static,
{
    let server = Arc::new(server);
    let out: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(Box::new(writer)));
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        if buf.len() > server.config.max_frame_bytes {
            let err = Response {
                jsonrpc: "2.0".into(),
                id: RequestId::Num(0),
                result: None,
                error: Some(RpcError::invalid_request(format!(
                    "frame exceeds max_frame_bytes ({} > {})",
                    buf.len(),
                    server.config.max_frame_bytes
                ))),
            };
            write_response(&out, &err)?;
            continue;
        }
        let line = buf.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        match parse_frame(line) {
            Ok(Inbound::Request(req)) => {
                let id = req.id.clone();
                let result = match req.method.as_str() {
                    "initialize" => handle_initialize(&server, req.params),
                    "session/new" => handle_session_new(&server, req.params),
                    "session/prompt" => handle_session_prompt(&server, req.params, &out),
                    other => Err(RpcError::method_not_found(other)),
                };
                let resp = match result {
                    Ok(v) => Response {
                        jsonrpc: "2.0".into(),
                        id,
                        result: Some(v),
                        error: None,
                    },
                    Err(e) => Response {
                        jsonrpc: "2.0".into(),
                        id,
                        result: None,
                        error: Some(e),
                    },
                };
                write_response(&out, &resp)?;
            }
            Ok(Inbound::Notification(n)) => {
                if n.method == "session/cancel" {
                    handle_session_cancel(&server, n.params);
                }
                // Other notifications: ignored silently per JSON-RPC.
            }
            Err(e) => {
                let resp = Response {
                    jsonrpc: "2.0".into(),
                    id: RequestId::Num(0),
                    result: None,
                    error: Some(e),
                };
                write_response(&out, &resp)?;
            }
        }
    }
    Ok(())
}

fn ts_now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only echo LLM: emits the prompt back as 5 chunks then Done.
    fn echo_llm(prompt: String, cancel: CancelToken, tx: ChunkSender) {
        let chunks: Vec<String> = prompt
            .split_whitespace()
            .take(5)
            .map(|w| w.to_string())
            .collect();
        for c in chunks {
            if cancel.is_cancelled() {
                break;
            }
            let _ = tx.send(AcpChunk::Delta(c));
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = tx.send(AcpChunk::Done);
    }

    /// Cleaner test driver that returns the writer's bytes.
    fn drive(server: AcpServer, input: &str) -> (String, Arc<AcpServer>) {
        use std::io::Cursor;
        let server = Arc::new(server);
        let cursor = Cursor::new(input.as_bytes().to_vec());
        let mut reader = std::io::BufReader::new(cursor);
        let raw: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                let mut g = self.0.lock().unwrap();
                g.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let writer: Arc<Mutex<Box<dyn Write + Send>>> =
            Arc::new(Mutex::new(Box::new(Sink(raw.clone()))));
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = reader.read_line(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            let line = buf.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            match parse_frame(line) {
                Ok(Inbound::Request(req)) => {
                    let id = req.id.clone();
                    let result = match req.method.as_str() {
                        "initialize" => handle_initialize(&server, req.params),
                        "session/new" => handle_session_new(&server, req.params),
                        "session/prompt" => handle_session_prompt(&server, req.params, &writer),
                        other => Err(RpcError::method_not_found(other)),
                    };
                    let resp = match result {
                        Ok(v) => Response {
                            jsonrpc: "2.0".into(),
                            id,
                            result: Some(v),
                            error: None,
                        },
                        Err(e) => Response {
                            jsonrpc: "2.0".into(),
                            id,
                            result: None,
                            error: Some(e),
                        },
                    };
                    write_response(&writer, &resp).unwrap();
                }
                Ok(Inbound::Notification(n)) => {
                    if n.method == "session/cancel" {
                        handle_session_cancel(&server, n.params);
                    }
                }
                Err(e) => {
                    let resp = Response {
                        jsonrpc: "2.0".into(),
                        id: RequestId::Num(0),
                        result: None,
                        error: Some(e),
                    };
                    write_response(&writer, &resp).unwrap();
                }
            }
        }
        let bytes = raw.lock().unwrap().clone();
        (String::from_utf8(bytes).unwrap(), server)
    }

    #[test]
    fn parse_request_frame() {
        let f = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let parsed = parse_frame(f).unwrap();
        match parsed {
            Inbound::Request(r) => {
                assert_eq!(r.method, "initialize");
                assert!(matches!(r.id, RequestId::Num(1)));
            }
            _ => panic!("expected request"),
        }
    }

    #[test]
    fn parse_notification_frame() {
        let f = r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s1"}}"#;
        let parsed = parse_frame(f).unwrap();
        assert!(matches!(parsed, Inbound::Notification(_)));
    }

    #[test]
    fn reject_missing_jsonrpc() {
        let f = r#"{"id":1,"method":"x"}"#;
        let err = parse_frame(f).unwrap_err();
        assert_eq!(err.code, -32600);
    }

    #[test]
    fn reject_non_object_frame() {
        let err = parse_frame("123").unwrap_err();
        assert_eq!(err.code, -32600);
    }

    #[test]
    fn reject_malformed_json() {
        let err = parse_frame("not json").unwrap_err();
        assert_eq!(err.code, -32700);
    }

    #[test]
    fn handshake_completes() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#
        );
        let (out, server) = drive(server, &input);
        assert!(out.contains("\"protocolVersion\":1"));
        // Field is serialized camelCase per ACP wire format.
        assert!(
            out.contains("\"agentCapabilities\""),
            "missing agentCapabilities: {}",
            out
        );
        let recs = server.slo_records();
        assert!(recs.iter().any(|r| r.kind == "handshake"));
    }

    #[test]
    fn version_negotiation_returns_ours() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":99,"clientCapabilities":{}}}"#
        );
        let (out, _) = drive(server, &input);
        assert!(out.contains(&format!("\"protocolVersion\":{}", PROTOCOL_VERSION)));
    }

    #[test]
    fn session_new_requires_initialize_first() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}"#
        );
        let (out, _) = drive(server, &input);
        assert!(out.contains("initialize must precede"));
    }

    #[test]
    fn session_new_requires_absolute_cwd() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"relative/path","mcpServers":[]}}"#
        );
        let (out, _) = drive(server, &input);
        assert!(out.contains("cwd must be an absolute path"));
    }

    #[test]
    fn prompt_streams_chunks_and_returns_end_turn() {
        let server = AcpServer::new(AcpServerConfig::default()).with_llm(echo_llm);
        let cwd = if cfg!(windows) { "C:\\tmp" } else { "/tmp" };
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#;
        let new = format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{}","mcpServers":[]}}}}"#,
            cwd.replace('\\', "\\\\")
        );
        let prompt = r#"{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"sess_1","prompt":[{"type":"text","text":"hello world from acp test"}]}}"#;
        let input = format!("{}\n{}\n{}\n", init, new, prompt);
        let (out, server) = drive(server, &input);
        // We should see at least one session/update notification for chunks.
        assert!(
            out.contains("\"method\":\"session/update\""),
            "missing chunks: {}",
            out
        );
        assert!(out.contains("\"sessionUpdate\":\"agent_message_chunk\""));
        // And a final response with stopReason end_turn.
        assert!(
            out.contains("\"stopReason\":\"end_turn\""),
            "missing stopReason: {}",
            out
        );
        let recs = server.slo_records();
        assert!(recs.iter().any(|r| r.kind == "first_chunk"));
        assert!(recs.iter().any(|r| r.kind == "prompt"));
    }

    #[test]
    fn prompt_without_session_returns_resource_not_found() {
        let server = AcpServer::new(AcpServerConfig::default()).with_llm(echo_llm);
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#;
        let prompt = r#"{"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{"sessionId":"nope","prompt":[{"type":"text","text":"x"}]}}"#;
        let input = format!("{}\n{}\n", init, prompt);
        let (out, _) = drive(server, &input);
        assert!(out.contains("\"code\":-32002"), "expected -32002: {}", out);
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"made/up","params":{}}"#
        );
        let (out, _) = drive(server, &input);
        assert!(out.contains("\"code\":-32601"));
    }

    #[test]
    fn cancel_notification_short_circuits_prompt() {
        // LLM that sleeps for a long time without sending Done — server
        // must observe the cancel flag and return Cancelled on its own.
        fn slow_llm(_prompt: String, cancel: CancelToken, tx: ChunkSender) {
            // Emit one chunk so we know the LLM thread is alive, then idle.
            let _ = tx.send(AcpChunk::Delta("first".into()));
            for _ in 0..1000 {
                if cancel.is_cancelled() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            // Intentionally do NOT send Done — exercise server-side cancel.
        }
        let server = Arc::new(AcpServer::new(AcpServerConfig::default()).with_llm(slow_llm));
        let _ = handle_initialize(
            &server,
            Some(json!({"protocolVersion":1,"clientCapabilities":{}})),
        )
        .unwrap();
        let cwd = if cfg!(windows) { "C:\\tmp" } else { "/tmp" };
        let _ = handle_session_new(&server, Some(json!({"cwd": cwd, "mcpServers": []}))).unwrap();
        // Spawn the prompt on its own thread and cancel from this thread.
        let raw: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let writer: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(Box::new(Sink(raw))));
        let s2 = server.clone();
        let w2 = writer.clone();
        let handle = std::thread::spawn(move || {
            handle_session_prompt(
                &s2,
                Some(json!({
                    "sessionId": "sess_1",
                    "prompt": [{"type":"text","text":"slow"}]
                })),
                &w2,
            )
        });
        // Give the prompt a brief moment to start streaming, then cancel.
        std::thread::sleep(Duration::from_millis(80));
        handle_session_cancel(&server, Some(json!({ "sessionId": "sess_1" })));
        let resp = handle.join().expect("prompt thread panicked").unwrap();
        let parsed: PromptResponse = serde_json::from_value(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::Cancelled);
    }

    #[test]
    fn content_block_text_serializes_with_type_tag() {
        let cb = ContentBlock::text("hi");
        let s = serde_json::to_string(&cb).unwrap();
        assert!(s.contains("\"type\":\"text\""));
        assert!(s.contains("\"text\":\"hi\""));
    }

    #[test]
    fn handshake_meets_slo_target() {
        let server = AcpServer::new(AcpServerConfig::default());
        let input = format!(
            "{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#
        );
        let (_out, server) = drive(server, &input);
        let recs = server.slo_records();
        let h = recs
            .iter()
            .find(|r| r.kind == "handshake")
            .expect("no handshake record");
        // SLO: handshake <200ms. Pure data shuffling — should be sub-ms.
        assert!(
            h.elapsed_ms < 200,
            "handshake exceeded SLO: {}ms",
            h.elapsed_ms
        );
    }

    #[test]
    fn streaming_meets_chunks_per_sec_target() {
        // Stub LLM that bursts 100 chunks back-to-back, then Done.
        // We expect server-side throughput well above 30/s.
        fn burst_llm(_p: String, _cancel: CancelToken, tx: ChunkSender) {
            for _ in 0..100 {
                let _ = tx.send(AcpChunk::Delta("c".into()));
            }
            let _ = tx.send(AcpChunk::Done);
        }
        let server = AcpServer::new(AcpServerConfig::default()).with_llm(burst_llm);
        let cwd = if cfg!(windows) { "C:\\tmp" } else { "/tmp" };
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}"#;
        let new = format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{}","mcpServers":[]}}}}"#,
            cwd.replace('\\', "\\\\")
        );
        let prompt = r#"{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"sess_1","prompt":[{"type":"text","text":"go"}]}}"#;
        let input = format!("{}\n{}\n{}\n", init, new, prompt);
        let (_out, server) = drive(server, &input);
        let recs = server.slo_records();
        let p = recs
            .iter()
            .find(|r| r.kind == "prompt")
            .expect("no prompt record");
        assert!(p.chunks >= 100, "wrong chunk count: {}", p.chunks);
        assert!(
            p.chunks_per_sec >= 30.0,
            "throughput below SLO: {:.1} chunks/s",
            p.chunks_per_sec
        );
        let fc = recs
            .iter()
            .find(|r| r.kind == "first_chunk")
            .expect("no first_chunk record");
        assert!(
            fc.elapsed_ms < 1000,
            "first_chunk exceeded SLO: {}ms",
            fc.elapsed_ms
        );
    }

    #[test]
    fn session_update_uses_session_update_discriminator() {
        let upd = SessionUpdate::AgentMessageChunk {
            content: ContentBlock::text("x"),
        };
        let s = serde_json::to_string(&upd).unwrap();
        assert!(s.contains("\"sessionUpdate\":\"agent_message_chunk\""));
    }
}
