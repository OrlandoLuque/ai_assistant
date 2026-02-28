//! Browser tools — Chrome DevTools Protocol automation
//!
//! Provides browser control tools for web scraping, testing, and interaction.
//! Tools are registered into the existing ToolRegistry system.
//!
//! This module implements real CDP communication over a minimal WebSocket client.
//! It can launch Chrome in headless mode or connect to an existing DevTools endpoint.

use crate::agent_policy::{ActionDescriptor, ActionType};
use crate::agent_sandbox::SandboxValidator;
use crate::unified_tools::{ToolBuilder, ToolCall, ToolError, ToolOutput, ToolRegistry};
use std::sync::{Arc, RwLock};

// ============================================================================
// Types
// ============================================================================

/// Represents the content of a page after navigation.
#[derive(Debug)]
pub struct PageContent {
    pub url: String,
    pub title: String,
    pub html: String,
    pub text: String,
    pub screenshot: Option<String>,
}

/// Errors that can occur during browser operations.
#[derive(Debug, Clone)]
pub enum BrowserError {
    /// Chrome binary not found on the system.
    ChromeNotFound,
    /// Failed to launch Chrome process.
    LaunchFailed(String),
    /// Not connected to a browser.
    NotConnected,
    /// WebSocket communication error.
    WebSocketError(String),
    /// CDP protocol error.
    CdpError(String),
    /// Operation timed out.
    Timeout(String),
    /// Element not found.
    ElementNotFound(String),
}

impl std::fmt::Display for BrowserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChromeNotFound => write!(
                f,
                "Chrome/Chromium not found. Set CHROME_BIN env var or install Chrome."
            ),
            Self::LaunchFailed(e) => write!(f, "Failed to launch Chrome: {}", e),
            Self::NotConnected => write!(f, "Not connected to a browser"),
            Self::WebSocketError(e) => write!(f, "WebSocket error: {}", e),
            Self::CdpError(e) => write!(f, "CDP error: {}", e),
            Self::Timeout(e) => write!(f, "Timeout: {}", e),
            Self::ElementNotFound(s) => write!(f, "Element not found: {}", s),
        }
    }
}

impl std::error::Error for BrowserError {}

// ============================================================================
// Chrome process management
// ============================================================================

/// Find Chrome/Chromium binary path.
pub fn find_chrome_binary() -> Option<std::path::PathBuf> {
    // Check env var first
    if let Ok(path) = std::env::var("CHROME_BIN") {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(path) = std::env::var("CHROMIUM_BIN") {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Common paths by platform
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Chromium\Application\chrome.exe",
        ];
        for c in &candidates {
            let p = std::path::PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        let candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ];
        for c in &candidates {
            let p = std::path::PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
        ];
        for c in &candidates {
            let p = std::path::PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Launch Chrome in headless mode with remote debugging.
/// Returns the child process and the WebSocket debugger URL.
fn launch_chrome(port: u16) -> Result<(std::process::Child, String), BrowserError> {
    let chrome_path = find_chrome_binary().ok_or(BrowserError::ChromeNotFound)?;

    let user_data_dir = std::env::temp_dir().join(format!("ai_assistant_chrome_{}", port));
    let _ = std::fs::create_dir_all(&user_data_dir);

    let child = std::process::Command::new(&chrome_path)
        .args([
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            &format!("--remote-debugging-port={}", port),
            &format!("--user-data-dir={}", user_data_dir.display()),
            "about:blank",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| BrowserError::LaunchFailed(format!("Failed to spawn Chrome: {}", e)))?;

    // Wait for Chrome to start and get the WS URL from the /json/version endpoint
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(10);
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();

    loop {
        if start.elapsed() > timeout {
            return Err(BrowserError::LaunchFailed(
                "Chrome startup timed out".into(),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Try to connect to the debug endpoint
        if let Ok(resp) = client
            .get(&format!("http://127.0.0.1:{}/json/version", port))
            .call()
        {
            if let Ok(body) = resp.into_string() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(ws_url) = val.get("webSocketDebuggerUrl").and_then(|v| v.as_str()) {
                        return Ok((child, ws_url.to_string()));
                    }
                }
            }
        }
    }
}

// ============================================================================
// WebSocket + CDP communication
// ============================================================================

/// Generate pseudo-random bytes (not cryptographic, just for WS masking key).
fn random_bytes(n: usize) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut bytes = Vec::with_capacity(n);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut hasher = DefaultHasher::new();
    for i in 0..n {
        (seed + i as u128).hash(&mut hasher);
        bytes.push((hasher.finish() & 0xFF) as u8);
    }
    bytes
}

/// Base64 encode bytes (standard alphabet, with padding).
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Connect via WebSocket to a CDP endpoint.
/// Parses the ws:// URL, performs TCP connect, sends HTTP upgrade handshake.
fn ws_connect(ws_url: &str) -> Result<std::net::TcpStream, BrowserError> {
    // Parse ws://host:port/path
    let url = ws_url.strip_prefix("ws://").ok_or_else(|| {
        BrowserError::WebSocketError("Invalid WS URL: must start with ws://".into())
    })?;
    let (host_port, _path) = url.split_once('/').unwrap_or((url, ""));

    let mut stream = std::net::TcpStream::connect(host_port)
        .map_err(|e| BrowserError::WebSocketError(format!("TCP connect failed: {}", e)))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|e| BrowserError::WebSocketError(format!("Set timeout failed: {}", e)))?;
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .map_err(|e| BrowserError::WebSocketError(format!("Set timeout failed: {}", e)))?;

    // WebSocket handshake
    use std::io::Write;
    let key = base64_encode(&random_bytes(16));
    let path_part = url.split_once('/').map(|(_, p)| p).unwrap_or("");
    let handshake = format!(
        "GET /{} HTTP/1.1\r\nHost: {}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 13\r\n\r\n",
        path_part,
        host_port,
        key
    );
    stream
        .write_all(handshake.as_bytes())
        .map_err(|e| BrowserError::WebSocketError(format!("Handshake write failed: {}", e)))?;

    // Read response (verify 101 status)
    use std::io::Read;
    let mut buf = [0u8; 4096];
    let n = stream
        .read(&mut buf)
        .map_err(|e| BrowserError::WebSocketError(format!("Handshake read failed: {}", e)))?;
    let response = String::from_utf8_lossy(&buf[..n]);
    if !response.contains("101") {
        return Err(BrowserError::WebSocketError(format!(
            "Handshake failed: {}",
            response.lines().next().unwrap_or("")
        )));
    }

    Ok(stream)
}

/// Send a WebSocket text frame (masked, per RFC 6455).
fn ws_send(stream: &mut std::net::TcpStream, data: &str) -> Result<(), BrowserError> {
    use std::io::Write;
    let payload = data.as_bytes();
    let len = payload.len();

    let mut frame = Vec::new();
    frame.push(0x81); // FIN + text opcode

    let mask_key: [u8; 4] = {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        [
            (t & 0xFF) as u8,
            ((t >> 8) & 0xFF) as u8,
            ((t >> 16) & 0xFF) as u8,
            ((t >> 24) & 0xFF) as u8,
        ]
    };

    if len < 126 {
        frame.push((len as u8) | 0x80); // mask bit set
    } else if len < 65536 {
        frame.push(126 | 0x80);
        frame.push((len >> 8) as u8);
        frame.push((len & 0xFF) as u8);
    } else {
        frame.push(127 | 0x80);
        for i in (0..8).rev() {
            frame.push(((len >> (8 * i)) & 0xFF) as u8);
        }
    }

    frame.extend_from_slice(&mask_key);
    for (i, b) in payload.iter().enumerate() {
        frame.push(b ^ mask_key[i % 4]);
    }

    stream
        .write_all(&frame)
        .map_err(|e| BrowserError::WebSocketError(format!("Send failed: {}", e)))
}

/// Read a WebSocket frame (server sends unmasked).
fn ws_recv(stream: &mut std::net::TcpStream) -> Result<String, BrowserError> {
    use std::io::Read;
    let mut header = [0u8; 2];
    stream
        .read_exact(&mut header)
        .map_err(|e| BrowserError::WebSocketError(format!("Read header failed: {}", e)))?;

    let _fin = header[0] & 0x80 != 0;
    let masked = header[1] & 0x80 != 0;
    let mut payload_len = (header[1] & 0x7F) as u64;

    if payload_len == 126 {
        let mut ext = [0u8; 2];
        stream
            .read_exact(&mut ext)
            .map_err(|e| BrowserError::WebSocketError(format!("Read ext len failed: {}", e)))?;
        payload_len = u16::from_be_bytes(ext) as u64;
    } else if payload_len == 127 {
        let mut ext = [0u8; 8];
        stream
            .read_exact(&mut ext)
            .map_err(|e| BrowserError::WebSocketError(format!("Read ext len failed: {}", e)))?;
        payload_len = u64::from_be_bytes(ext);
    }

    let mask_key = if masked {
        let mut mk = [0u8; 4];
        stream
            .read_exact(&mut mk)
            .map_err(|e| BrowserError::WebSocketError(format!("Read mask failed: {}", e)))?;
        Some(mk)
    } else {
        None
    };

    let mut payload = vec![0u8; payload_len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| BrowserError::WebSocketError(format!("Read payload failed: {}", e)))?;

    if let Some(mk) = mask_key {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= mk[i % 4];
        }
    }

    String::from_utf8(payload)
        .map_err(|e| BrowserError::WebSocketError(format!("UTF-8 decode failed: {}", e)))
}

/// Send a CDP command and wait for the matching response by ID.
fn send_cdp_command(
    stream: &mut std::net::TcpStream,
    id: u64,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, BrowserError> {
    let msg = serde_json::json!({
        "id": id,
        "method": method,
        "params": params,
    });
    ws_send(stream, &msg.to_string())?;

    // Read responses until we find the one matching our ID.
    // Other responses (events) are discarded.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if std::time::Instant::now() > deadline {
            return Err(BrowserError::Timeout(format!(
                "CDP command {} timed out",
                method
            )));
        }
        let raw = ws_recv(stream)?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
            if val.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(err) = val.get("error") {
                    return Err(BrowserError::CdpError(err.to_string()));
                }
                return Ok(val
                    .get("result")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null));
            }
            // Otherwise it's an event -- skip it
        }
    }
}

// ============================================================================
// BrowserSession
// ============================================================================

/// A browser session over CDP (Chrome DevTools Protocol).
///
/// Manages a WebSocket connection to a Chrome instance and sends CDP commands
/// for navigation, DOM interaction, JavaScript evaluation, and screenshots.
pub struct BrowserSession {
    /// WebSocket URL for CDP connection (e.g., ws://127.0.0.1:9222/devtools/page/xxx)
    ws_url: Option<String>,
    /// TCP stream for WebSocket communication
    ws_stream: Option<std::net::TcpStream>,
    /// Current page URL
    current_url: Option<String>,
    /// Current page title
    current_title: Option<String>,
    /// Chrome subprocess handle
    process: Option<std::process::Child>,
    /// Monotonically increasing request ID for CDP JSON-RPC
    next_request_id: std::sync::atomic::AtomicU64,
    /// Whether we're connected
    connected: bool,
}

impl BrowserSession {
    /// Create a new disconnected browser session.
    pub fn new() -> Self {
        Self {
            ws_url: None,
            ws_stream: None,
            current_url: None,
            current_title: None,
            process: None,
            next_request_id: std::sync::atomic::AtomicU64::new(1),
            connected: false,
        }
    }

    fn next_id(&self) -> u64 {
        self.next_request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Connect to an existing Chrome DevTools endpoint.
    pub fn connect(&mut self, ws_url: &str) -> Result<(), BrowserError> {
        let stream = ws_connect(ws_url)?;
        self.ws_url = Some(ws_url.to_string());
        self.ws_stream = Some(stream);
        self.connected = true;
        Ok(())
    }

    /// Launch Chrome and connect to it.
    pub fn launch(&mut self, port: u16) -> Result<(), BrowserError> {
        let (child, ws_url) = launch_chrome(port)?;
        self.process = Some(child);
        self.connect(&ws_url)
    }

    /// Navigate to a URL and return the page content.
    pub fn navigate(&mut self, url: &str) -> Result<PageContent, BrowserError> {
        // Pre-allocate all command IDs before borrowing the stream mutably
        let nav_id = self.next_id();
        let eval_id = self.next_id();
        let html_id = self.next_id();
        let text_id = self.next_id();

        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;

        // Navigate
        send_cdp_command(
            stream,
            nav_id,
            "Page.navigate",
            serde_json::json!({"url": url}),
        )?;

        // Wait a bit for page load
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Get title
        let title_result = send_cdp_command(
            stream,
            eval_id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": "document.title"
            }),
        )?;
        let title = title_result
            .get("result")
            .and_then(|r| r.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Get HTML
        let html_result = send_cdp_command(
            stream,
            html_id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": "document.documentElement.outerHTML"
            }),
        )?;
        let html = html_result
            .get("result")
            .and_then(|r| r.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Get text
        let text_result = send_cdp_command(
            stream,
            text_id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": "document.body ? document.body.innerText : ''"
            }),
        )?;
        let text = text_result
            .get("result")
            .and_then(|r| r.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        self.current_url = Some(url.to_string());
        self.current_title = Some(title.clone());

        Ok(PageContent {
            url: url.to_string(),
            title,
            html,
            text,
            screenshot: None,
        })
    }

    /// Click on an element matching the CSS selector.
    pub fn click(&mut self, selector: &str) -> Result<(), BrowserError> {
        let id = self.next_id();
        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;
        let js = format!(
            "(() => {{ const el = document.querySelector('{}'); if (!el) throw new Error('not found'); el.click(); return 'clicked'; }})()",
            selector.replace('\'', "\\'")
        );
        let result = send_cdp_command(
            stream,
            id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": js,
                "awaitPromise": false,
            }),
        )?;
        // Check for exception
        if let Some(ex) = result.get("exceptionDetails") {
            return Err(BrowserError::ElementNotFound(format!(
                "{}: {}",
                selector, ex
            )));
        }
        Ok(())
    }

    /// Type text into an element matching the CSS selector.
    pub fn type_text(&mut self, selector: &str, text: &str) -> Result<(), BrowserError> {
        let id = self.next_id();
        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;
        let js = format!(
            "(() => {{ const el = document.querySelector('{}'); if (!el) throw new Error('not found'); el.focus(); el.value = '{}'; el.dispatchEvent(new Event('input', {{bubbles: true}})); return 'typed'; }})()",
            selector.replace('\'', "\\'"),
            text.replace('\'', "\\'")
        );
        let result = send_cdp_command(
            stream,
            id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": js,
            }),
        )?;
        if let Some(ex) = result.get("exceptionDetails") {
            return Err(BrowserError::ElementNotFound(format!(
                "{}: {}",
                selector, ex
            )));
        }
        Ok(())
    }

    /// Get the text content of an element matching the CSS selector.
    pub fn get_text(&mut self, selector: &str) -> Result<String, BrowserError> {
        let id = self.next_id();
        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;
        let js = format!(
            "(() => {{ const el = document.querySelector('{}'); if (!el) throw new Error('not found'); return el.textContent || ''; }})()",
            selector.replace('\'', "\\'")
        );
        let result = send_cdp_command(
            stream,
            id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": js,
            }),
        )?;
        if let Some(ex) = result.get("exceptionDetails") {
            return Err(BrowserError::ElementNotFound(format!(
                "{}: {}",
                selector, ex
            )));
        }
        Ok(result
            .get("result")
            .and_then(|r| r.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
    }

    /// Execute JavaScript and return the result as a string.
    pub fn evaluate(&mut self, js: &str) -> Result<String, BrowserError> {
        let id = self.next_id();
        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;
        let result = send_cdp_command(
            stream,
            id,
            "Runtime.evaluate",
            serde_json::json!({
                "expression": js,
                "returnByValue": true,
            }),
        )?;
        if let Some(ex) = result.get("exceptionDetails") {
            return Err(BrowserError::CdpError(format!("JS error: {}", ex)));
        }
        let val = result.get("result").and_then(|r| r.get("value"));
        match val {
            Some(serde_json::Value::String(s)) => Ok(s.clone()),
            Some(v) => Ok(v.to_string()),
            None => Ok(String::new()),
        }
    }

    /// Take a screenshot and return the PNG data as a base64-encoded string.
    pub fn screenshot(&mut self) -> Result<String, BrowserError> {
        let id = self.next_id();
        let stream = self.ws_stream.as_mut().ok_or(BrowserError::NotConnected)?;
        let result = send_cdp_command(
            stream,
            id,
            "Page.captureScreenshot",
            serde_json::json!({
                "format": "png",
            }),
        )?;
        Ok(result
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
    }

    /// Wait for an element matching the selector to appear in the DOM.
    pub fn wait_for(&mut self, selector: &str, timeout_ms: u64) -> Result<(), BrowserError> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);
        loop {
            if start.elapsed() > timeout {
                return Err(BrowserError::Timeout(format!(
                    "Element '{}' not found within {}ms",
                    selector, timeout_ms
                )));
            }
            match self.get_text(selector) {
                Ok(_) => return Ok(()),
                Err(BrowserError::ElementNotFound(_)) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Close the browser session.
    pub fn close(&mut self) -> Result<(), BrowserError> {
        let id = self.next_id();
        if let Some(ref mut stream) = self.ws_stream {
            let _ = send_cdp_command(stream, id, "Browser.close", serde_json::json!({}));
        }
        self.ws_stream = None;
        self.connected = false;
        if let Some(ref mut child) = self.process {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.process = None;
        Ok(())
    }

    /// Check whether the session is connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get the current page URL, if any.
    pub fn current_url(&self) -> Option<&str> {
        self.current_url.as_deref()
    }

    /// Get the current page title, if any.
    pub fn current_title(&self) -> Option<&str> {
        self.current_title.as_deref()
    }
}

impl Default for BrowserSession {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BrowserSession {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// ============================================================================
// Tool Registration
// ============================================================================

/// Register browser tools into a ToolRegistry.
///
/// Registers the following tools:
/// - `browser_navigate` -- navigate to a URL
/// - `browser_click` -- click an element by CSS selector
/// - `browser_type` -- type text into an element
/// - `browser_get_text` -- get text content of an element
/// - `browser_evaluate` -- evaluate JavaScript in the page
/// - `browser_screenshot` -- capture a screenshot
/// - `browser_close` -- close the browser session
pub fn register_browser_tools(
    registry: &mut ToolRegistry,
    session: Arc<RwLock<BrowserSession>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    // 1. browser_navigate
    {
        let def = ToolBuilder::new("browser_navigate", "Navigate to a URL in the browser")
            .required_string("url", "The URL to navigate to")
            .category("browser")
            .build();

        let sess = Arc::clone(&session);
        let sbx = Arc::clone(&sandbox);

        registry.register(
            def,
            Arc::new(move |call: &ToolCall| {
                let url = call
                    .get_string("url")
                    .ok_or_else(|| ToolError::MissingParameter("url".to_string()))?;

                // Validate URL with sandbox
                {
                    let mut sandbox_guard = sbx
                        .write()
                        .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?;
                    let action = ActionDescriptor::new(ActionType::HttpRequest, url);
                    sandbox_guard.validate(&action).map_err(|e| {
                        ToolError::ExecutionFailed(format!("Sandbox denied navigation: {}", e))
                    })?;
                }

                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                let content = session_guard
                    .navigate(url)
                    .map_err(|e| ToolError::ExecutionFailed(format!("Navigation failed: {}", e)))?;

                Ok(ToolOutput::text(format!(
                    "Navigated to {}\nTitle: {}\nText: {}",
                    content.url, content.title, content.text
                )))
            }),
        );
    }

    // 2. browser_click
    {
        let def = ToolBuilder::new("browser_click", "Click an element by CSS selector")
            .required_string("selector", "CSS selector of the element to click")
            .category("browser")
            .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |call: &ToolCall| {
                let selector = call
                    .get_string("selector")
                    .ok_or_else(|| ToolError::MissingParameter("selector".to_string()))?;

                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                session_guard
                    .click(selector)
                    .map_err(|e| ToolError::ExecutionFailed(format!("Click failed: {}", e)))?;

                Ok(ToolOutput::text(format!("Clicked element: {}", selector)))
            }),
        );
    }

    // 3. browser_type
    {
        let def = ToolBuilder::new("browser_type", "Type text into an element by CSS selector")
            .required_string("selector", "CSS selector of the input element")
            .required_string("text", "Text to type into the element")
            .category("browser")
            .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |call: &ToolCall| {
                let selector = call
                    .get_string("selector")
                    .ok_or_else(|| ToolError::MissingParameter("selector".to_string()))?;
                let text = call
                    .get_string("text")
                    .ok_or_else(|| ToolError::MissingParameter("text".to_string()))?;

                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                session_guard
                    .type_text(selector, text)
                    .map_err(|e| ToolError::ExecutionFailed(format!("Type text failed: {}", e)))?;

                Ok(ToolOutput::text(format!(
                    "Typed '{}' into element: {}",
                    text, selector
                )))
            }),
        );
    }

    // 4. browser_get_text
    {
        let def = ToolBuilder::new(
            "browser_get_text",
            "Get text content of an element by CSS selector",
        )
        .required_string("selector", "CSS selector of the element")
        .category("browser")
        .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |call: &ToolCall| {
                let selector = call
                    .get_string("selector")
                    .ok_or_else(|| ToolError::MissingParameter("selector".to_string()))?;

                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                let text = session_guard
                    .get_text(selector)
                    .map_err(|e| ToolError::ExecutionFailed(format!("Get text failed: {}", e)))?;

                Ok(ToolOutput::text(text))
            }),
        );
    }

    // 5. browser_evaluate
    {
        let def = ToolBuilder::new(
            "browser_evaluate",
            "Evaluate JavaScript in the browser page",
        )
        .required_string("js", "JavaScript code to evaluate")
        .category("browser")
        .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |call: &ToolCall| {
                let js = call
                    .get_string("js")
                    .ok_or_else(|| ToolError::MissingParameter("js".to_string()))?;

                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                let result = session_guard
                    .evaluate(js)
                    .map_err(|e| ToolError::ExecutionFailed(format!("Evaluate failed: {}", e)))?;

                Ok(ToolOutput::text(result))
            }),
        );
    }

    // 6. browser_screenshot
    {
        let def = ToolBuilder::new(
            "browser_screenshot",
            "Capture a screenshot of the current page",
        )
        .category("browser")
        .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |_call: &ToolCall| {
                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                let data = session_guard
                    .screenshot()
                    .map_err(|e| ToolError::ExecutionFailed(format!("Screenshot failed: {}", e)))?;

                Ok(ToolOutput::text(data))
            }),
        );
    }

    // 7. browser_close
    {
        let def = ToolBuilder::new("browser_close", "Close the browser session")
            .category("browser")
            .build();

        let sess = Arc::clone(&session);

        registry.register(
            def,
            Arc::new(move |_call: &ToolCall| {
                let mut session_guard = sess
                    .write()
                    .map_err(|_| ToolError::ExecutionFailed("session lock poisoned".into()))?;
                session_guard
                    .close()
                    .map_err(|e| ToolError::ExecutionFailed(format!("Close failed: {}", e)))?;
                Ok(ToolOutput::text("Browser session closed".to_string()))
            }),
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_policy::{AgentPolicyBuilder, InternetMode};

    #[test]
    fn test_find_chrome_binary_does_not_crash() {
        // Should return None or Some depending on the machine, but never panic.
        let _result = find_chrome_binary();
    }

    #[test]
    fn test_base64_encode_empty() {
        assert_eq!(base64_encode(&[]), "");
    }

    #[test]
    fn test_base64_encode_known_values() {
        // "Hello" -> "SGVsbG8="
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        // "f" -> "Zg=="
        assert_eq!(base64_encode(b"f"), "Zg==");
        // "fo" -> "Zm8="
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        // "foo" -> "Zm9v"
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        // "foobar" -> "Zm9vYmFy"
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_random_bytes_length() {
        assert_eq!(random_bytes(0).len(), 0);
        assert_eq!(random_bytes(1).len(), 1);
        assert_eq!(random_bytes(16).len(), 16);
        assert_eq!(random_bytes(100).len(), 100);
    }

    #[test]
    fn test_browser_error_display() {
        let err = BrowserError::ChromeNotFound;
        let s = format!("{}", err);
        assert!(s.contains("Chrome"));
        assert!(s.contains("not found"));

        let err = BrowserError::LaunchFailed("test reason".into());
        let s = format!("{}", err);
        assert!(s.contains("test reason"));

        let err = BrowserError::NotConnected;
        let s = format!("{}", err);
        assert!(s.contains("Not connected"));

        let err = BrowserError::WebSocketError("ws broke".into());
        let s = format!("{}", err);
        assert!(s.contains("ws broke"));

        let err = BrowserError::CdpError("bad method".into());
        let s = format!("{}", err);
        assert!(s.contains("bad method"));

        let err = BrowserError::Timeout("5s elapsed".into());
        let s = format!("{}", err);
        assert!(s.contains("5s elapsed"));

        let err = BrowserError::ElementNotFound("#missing".into());
        let s = format!("{}", err);
        assert!(s.contains("#missing"));
    }

    #[test]
    fn test_browser_session_new_state() {
        let session = BrowserSession::new();
        assert!(!session.is_connected());
        assert!(session.current_url().is_none());
        assert!(session.current_title().is_none());
        assert!(session.ws_url.is_none());
        assert!(session.ws_stream.is_none());
        assert!(session.process.is_none());
    }

    #[test]
    fn test_ws_connect_invalid_url() {
        let result = ws_connect("http://not-a-ws-url");
        assert!(result.is_err());
        match result {
            Err(BrowserError::WebSocketError(msg)) => {
                assert!(msg.contains("Invalid WS URL"));
            }
            other => panic!("Expected WebSocketError, got: {:?}", other),
        }
    }

    #[test]
    fn test_ws_connect_unreachable() {
        // Valid ws:// prefix but unreachable host:port
        let result = ws_connect("ws://127.0.0.1:1");
        assert!(result.is_err());
        match result {
            Err(BrowserError::WebSocketError(msg)) => {
                assert!(
                    msg.contains("TCP connect failed") || msg.contains("refused") || msg.len() > 0
                );
            }
            other => panic!("Expected WebSocketError, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_navigate_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.navigate("https://example.com");
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_click_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.click("#btn");
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_type_text_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.type_text("#input", "hello");
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_get_text_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.get_text("#el");
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_evaluate_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.evaluate("1 + 1");
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_screenshot_not_connected() {
        let mut session = BrowserSession::new();
        let result = session.screenshot();
        assert!(result.is_err());
        match result {
            Err(BrowserError::NotConnected) => {}
            other => panic!("Expected NotConnected, got: {:?}", other),
        }
    }

    #[test]
    fn test_session_close_when_not_connected() {
        let mut session = BrowserSession::new();
        // close on a non-connected session should succeed cleanly
        let result = session.close();
        assert!(result.is_ok());
        assert!(!session.is_connected());
    }

    #[test]
    fn test_page_content_creation() {
        let pc = PageContent {
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            html: "<html></html>".to_string(),
            text: "Hello".to_string(),
            screenshot: Some("base64data".to_string()),
        };
        assert_eq!(pc.url, "https://example.com");
        assert_eq!(pc.title, "Example");
        assert_eq!(pc.html, "<html></html>");
        assert_eq!(pc.text, "Hello");
        assert_eq!(pc.screenshot.as_deref(), Some("base64data"));
    }

    #[test]
    fn test_page_content_no_screenshot() {
        let pc = PageContent {
            url: "https://test.com".to_string(),
            title: "Test".to_string(),
            html: "".to_string(),
            text: "".to_string(),
            screenshot: None,
        };
        assert!(pc.screenshot.is_none());
    }

    #[test]
    fn test_session_default() {
        let session = BrowserSession::default();
        assert!(!session.is_connected());
    }

    #[test]
    fn test_register_browser_tools() {
        let mut registry = ToolRegistry::new();
        let session = Arc::new(RwLock::new(BrowserSession::new()));
        let policy = AgentPolicyBuilder::new()
            .internet(InternetMode::FullAccess)
            .build();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));

        let before = registry.len();
        register_browser_tools(&mut registry, session, sandbox);
        let after = registry.len();

        assert_eq!(after - before, 7);
        assert!(registry.get("browser_navigate").is_some());
        assert!(registry.get("browser_click").is_some());
        assert!(registry.get("browser_type").is_some());
        assert!(registry.get("browser_get_text").is_some());
        assert!(registry.get("browser_evaluate").is_some());
        assert!(registry.get("browser_screenshot").is_some());
        assert!(registry.get("browser_close").is_some());
    }
}
