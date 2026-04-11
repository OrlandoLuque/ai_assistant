//! # ai_proxy — API Gateway
//!
//! Binary that routes OpenAI-compatible API requests to backend
//! `ai_assistant` nodes with optional guardrails, budgeting, auditing,
//! caching, and rate limiting (V78 Gateway Hardening).
//!
//! Features:
//! - Round-robin load balancing across backend nodes
//! - Session affinity (sticky sessions via `X-Session-Id` header)
//! - Health checks on backends
//! - Request forwarding with minimal overhead
//! - **(V78, with `security` feature)** PII redaction, toxicity / attack
//!   filters, per-key rate limiting, response cache, append-only JSONL
//!   audit log with rotation, budget enforcement via V75 `CostDashboard`,
//!   TOML config file support
//!
//! ## Usage
//! ```bash
//! # Minimal: CLI flags only
//! ai_proxy --port 8080 --backends 10.0.0.1:8090,10.0.0.2:8090
//!
//! # With config file (V78)
//! ai_proxy --config /etc/ai_proxy.toml
//!
//! # With API key via env var (preferred over --api-key flag)
//! AI_PROXY_API_KEY=secret ai_proxy --config /etc/ai_proxy.toml
//! ```
//!
//! ## Required features
//! - `server-axum` — baseline routing, LB, session affinity, health checks,
//!   TOML config parsing (middleware sections parsed but inactive)
//! - `server-axum,security` — full gateway with guardrails, cache, audit,
//!   budget, and per-key rate limiting (V78)

use std::path::{Path, PathBuf};
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
use serde::{Deserialize, Serialize};

// ============================================================================
// CLI argument types
// ============================================================================

#[derive(Debug, Default)]
struct CliArgs {
    port: Option<u16>,
    backends: Option<String>,
    health_interval: Option<u64>,
    api_key: Option<String>,
    config: Option<PathBuf>,
    // WS-6: optional middleware overrides (CLI wins over config file).
    audit_log: Option<PathBuf>,
    audit_max_files: Option<u32>,
    enable_pii_redaction: bool,
    disable_cache: bool,
    cost_snapshot: Option<PathBuf>,
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
// Config file (V78) — TOML schema
// ============================================================================
//
// A typed mirror of `examples/ai_proxy.toml`. Loaded via `load_config()` and
// merged with CLI flags by `merge_cli_and_config()`. CLI flags always win on
// conflict.
//
// All sections are optional; unknown fields cause a parse error (fail loud)
// so typos in production configs are caught at startup.

/// Hard cap on the size of an `ai_proxy.toml` file, in bytes. Prevents a
/// malicious or accidental multi-GiB config from exhausting the process
/// memory during parsing.
const MAX_CONFIG_SIZE: u64 = 1024 * 1024; // 1 MiB

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ProxyConfig {
    #[serde(default)]
    server: ServerSection,
    #[serde(default)]
    backends: Vec<BackendSection>,
    #[serde(default)]
    middleware: MiddlewareSection,
    #[serde(default)]
    audit: AuditSection,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ServerSection {
    /// Bind address, e.g. `"0.0.0.0:8080"`. Only the port is used today;
    /// the listening host is still hard-coded (see `main()`).
    #[serde(default)]
    bind: Option<String>,
    #[serde(default)]
    health_check_interval_secs: Option<u64>,
    /// Name of the env var to read the API key from. Default:
    /// `AI_PROXY_API_KEY` (resolved unconditionally in the merger).
    #[serde(default)]
    api_key_env: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct BackendSection {
    addr: String,
    /// Weight is accepted but ignored in V78 (plain round-robin). Weighted
    /// round-robin is tracked for V80.
    #[serde(default = "default_backend_weight")]
    #[allow(dead_code)]
    weight: u32,
}

fn default_backend_weight() -> u32 {
    1
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // Fields consumed by WS-2/WS-3/WS-5 in follow-up workstreams.
struct MiddlewareSection {
    #[serde(default)]
    enable_rate_limit: bool,
    #[serde(default)]
    rate_limit_rpm: Option<u32>,
    #[serde(default)]
    rate_limit_burst: Option<u32>,

    #[serde(default)]
    enable_pii_input: bool,
    #[serde(default)]
    pii_input_strategy: Option<String>,
    #[serde(default)]
    pii_sensitivity: Option<String>,

    #[serde(default)]
    enable_pii_output: bool,
    #[serde(default)]
    pii_output_strategy: Option<String>,

    #[serde(default)]
    enable_toxicity_input: bool,
    #[serde(default)]
    enable_toxicity_output: bool,
    #[serde(default)]
    toxicity_threshold: Option<f64>,

    #[serde(default)]
    enable_attack_filter: bool,

    #[serde(default)]
    enable_budget: bool,
    #[serde(default)]
    cost_snapshot_path: Option<String>,
    #[serde(default)]
    monthly_budget_usd: Option<f64>,
    #[serde(default)]
    per_request_limit_usd: Option<f64>,

    #[serde(default)]
    enable_cache: bool,
    #[serde(default)]
    cache_max_entries: Option<usize>,
    #[serde(default)]
    cache_ttl_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // Fields consumed by WS-4 (audit log writer).
struct AuditSection {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    max_files: Option<u32>,
    #[serde(default)]
    max_bytes_per_file: Option<u64>,
    #[serde(default)]
    on_middleware_error: Option<String>,
}

/// Load and parse an `ai_proxy.toml` file.
///
/// Security:
/// - Rejects files larger than [`MAX_CONFIG_SIZE`] to prevent memory exhaustion.
/// - Canonicalizes the path (follows the caller-supplied path once, then reads
///   from the canonical result) so relative-path confusion is impossible.
/// - Unknown TOML fields are rejected (`deny_unknown_fields`) to catch typos.
fn load_config(path: &Path) -> Result<ProxyConfig, String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("Failed to stat config file '{}': {}", path.display(), e))?;
    if !meta.is_file() {
        return Err(format!(
            "Config path '{}' is not a regular file",
            path.display()
        ));
    }
    if meta.len() > MAX_CONFIG_SIZE {
        return Err(format!(
            "Config file '{}' is {} bytes, exceeds limit of {} bytes",
            path.display(),
            meta.len(),
            MAX_CONFIG_SIZE
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to canonicalize '{}': {}", path.display(), e))?;
    let text = std::fs::read_to_string(&canonical)
        .map_err(|e| format!("Failed to read '{}': {}", canonical.display(), e))?;
    let config: ProxyConfig = toml::from_str(&text)
        .map_err(|e| format!("Failed to parse TOML '{}': {}", canonical.display(), e))?;
    Ok(config)
}

/// Final, resolved runtime settings after merging CLI flags with an optional
/// loaded config file. CLI flags override any equivalent field from the file.
///
/// Precedence (lowest → highest):
///   built-in defaults → config file → `AI_PROXY_API_KEY` env var → CLI flags
#[derive(Debug)]
struct Effective {
    port: u16,
    backend_addrs: Vec<String>,
    health_interval: u64,
    api_key: Option<String>,
    /// Consumed by WS-2/WS-3/WS-5 in follow-up workstreams.
    #[allow(dead_code)]
    middleware: MiddlewareSection,
    /// Consumed by WS-4 (audit log writer).
    #[allow(dead_code)]
    audit: AuditSection,
}

/// Merge CLI flags and an optional loaded config file into the final
/// [`Effective`] settings used by `main()`.
///
/// Returns an error if the final backend list is empty.
fn merge_cli_and_config(cli: &CliArgs, file: Option<ProxyConfig>) -> Result<Effective, String> {
    // Built-in defaults (used when neither CLI nor file specifies a value).
    let mut port: u16 = 8080;
    let mut backend_addrs: Vec<String> = Vec::new();
    let mut health_interval: u64 = 30;
    let mut api_key: Option<String> = None;
    let mut middleware = MiddlewareSection::default();
    let mut audit = AuditSection::default();

    if let Some(cfg) = file {
        // Server section
        if let Some(bind) = cfg.server.bind.as_ref() {
            // Extract trailing :PORT. Host is ignored (we still bind 127.0.0.1).
            if let Some(p) = bind.rsplit(':').next() {
                match p.parse::<u16>() {
                    Ok(n) => port = n,
                    Err(_) => {
                        return Err(format!(
                            "Invalid [server].bind port in config file: '{}'",
                            bind
                        ));
                    }
                }
            }
        }
        if let Some(hi) = cfg.server.health_check_interval_secs {
            health_interval = hi;
        }
        // Config-file-selected env var for the API key (default: AI_PROXY_API_KEY).
        let env_name = cfg
            .server
            .api_key_env
            .clone()
            .unwrap_or_else(|| "AI_PROXY_API_KEY".to_string());
        if let Ok(v) = std::env::var(&env_name) {
            if !v.is_empty() {
                api_key = Some(v);
            }
        }

        // Backends section
        for b in cfg.backends.iter() {
            if !b.addr.is_empty() {
                backend_addrs.push(b.addr.clone());
            }
        }

        middleware = cfg.middleware;
        audit = cfg.audit;
    } else {
        // No config file: still honor AI_PROXY_API_KEY as a convenience.
        if let Ok(v) = std::env::var("AI_PROXY_API_KEY") {
            if !v.is_empty() {
                api_key = Some(v);
            }
        }
    }

    // CLI flags override any file-provided value.
    if let Some(p) = cli.port {
        port = p;
    }
    if let Some(ref b) = cli.backends {
        backend_addrs = b
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(hi) = cli.health_interval {
        health_interval = hi;
    }
    if cli.api_key.is_some() {
        api_key = cli.api_key.clone();
    }

    // WS-6: middleware-level CLI overrides.
    if let Some(ref p) = cli.audit_log {
        audit.enabled = true;
        audit.path = Some(p.to_string_lossy().into_owned());
    }
    if let Some(n) = cli.audit_max_files {
        audit.max_files = Some(n);
    }
    if cli.enable_pii_redaction {
        middleware.enable_pii_input = true;
        middleware.pii_input_strategy = Some("redact".to_string());
    }
    if cli.disable_cache {
        middleware.enable_cache = false;
    }
    if let Some(ref p) = cli.cost_snapshot {
        middleware.cost_snapshot_path = Some(p.to_string_lossy().into_owned());
    }

    if backend_addrs.is_empty() {
        return Err(
            "At least one backend must be specified (via --backends or [[backends]] in the config file)"
                .to_string(),
        );
    }

    Ok(Effective {
        port,
        backend_addrs,
        health_interval,
        api_key,
        middleware,
        audit,
    })
}

// ============================================================================
// V78 / WS-3: Response cache (LRU, PII-safe)
// ============================================================================

#[cfg(feature = "security")]
#[allow(dead_code)] // All items consumed by WS-2 (request path wiring).
mod cache {
    use dashmap::DashMap;
    use parking_lot::Mutex;
    use sha2::{Digest, Sha256};
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Hard size limit (in bytes) for cached response bodies. Oversized
    /// responses are not cached (prevents a giant payload from occupying an
    /// entire LRU slot).
    pub(super) const MAX_BODY_SIZE: usize = 1024 * 1024; // 1 MiB

    /// Cache key. Uses SHA-256 of the full scanned prompt plus quantized
    /// sampling parameters so that floating-point `temperature` values never
    /// cause false cache misses.
    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    pub(super) struct CacheKey {
        pub model: String,
        /// `(temperature * 1000).round() as u32` — quantized to integer
        /// milli-units to keep the key hashable.
        pub temperature_milli: u32,
        pub max_tokens: u32,
        pub prompt_sha256: [u8; 32],
    }

    impl CacheKey {
        pub fn new(model: &str, temperature: f64, max_tokens: u32, prompt: &str) -> Self {
            let mut hasher = Sha256::new();
            hasher.update(prompt.as_bytes());
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&hasher.finalize());
            let temp_clamped = temperature.clamp(0.0, 1000.0);
            Self {
                model: model.to_string(),
                temperature_milli: (temp_clamped * 1000.0).round() as u32,
                max_tokens,
                prompt_sha256: digest,
            }
        }
    }

    /// A cached backend response. `pii_free` gates whether the entry may be
    /// stored at all — see [`ResponseCache::put`].
    #[derive(Clone, Debug)]
    #[allow(dead_code)] // `status` consumed by WS-2 (request handler wiring).
    pub(super) struct CachedResponse {
        pub body: Vec<u8>,
        pub status: u16,
        pub stored_at: Instant,
        pub pii_free: bool,
    }

    /// Bounded LRU cache.
    ///
    /// - `put` **refuses** entries with `pii_free = false` (prevents leaking
    ///   redacted content on a subsequent cache hit).
    /// - `get` enforces a TTL and removes expired entries lazily.
    /// - Eviction is oldest-first when the capacity is reached.
    pub(super) struct ResponseCache {
        entries: DashMap<CacheKey, CachedResponse>,
        order: Mutex<VecDeque<CacheKey>>,
        max_entries: usize,
        ttl: Duration,
    }

    impl ResponseCache {
        pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
            Self {
                entries: DashMap::new(),
                order: Mutex::new(VecDeque::new()),
                max_entries: max_entries.max(1),
                ttl: Duration::from_secs(ttl_secs.max(1)),
            }
        }

        pub fn get(&self, key: &CacheKey) -> Option<CachedResponse> {
            let entry = self.entries.get(key)?;
            if entry.stored_at.elapsed() > self.ttl {
                drop(entry);
                self.entries.remove(key);
                self.order.lock().retain(|k| k != key);
                return None;
            }
            Some(entry.clone())
        }

        /// Insert a response into the cache. Returns `true` on success,
        /// `false` if the entry was rejected (e.g. `pii_free = false` or
        /// body too large).
        pub fn put(&self, key: CacheKey, response: CachedResponse) -> bool {
            if !response.pii_free {
                return false;
            }
            if response.body.len() > MAX_BODY_SIZE {
                return false;
            }
            let mut order = self.order.lock();
            // If re-inserting an existing key, drop the old order slot first.
            order.retain(|k| k != &key);
            // Evict until there is room for the new entry.
            while self.entries.len() >= self.max_entries {
                match order.pop_front() {
                    Some(evicted) => {
                        self.entries.remove(&evicted);
                    }
                    None => break,
                }
            }
            order.push_back(key.clone());
            self.entries.insert(key, response);
            true
        }

        pub fn len(&self) -> usize {
            self.entries.len()
        }

        #[cfg(test)]
        pub fn clear(&self) {
            self.entries.clear();
            self.order.lock().clear();
        }
    }
}

// ============================================================================
// V78 / WS-3: Per-key sliding-window rate limiter
// ============================================================================

#[cfg(feature = "security")]
#[allow(dead_code)] // All items consumed by WS-2 (request path wiring).
mod rate_limit {
    use dashmap::DashMap;
    use parking_lot::Mutex;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Global cap on the number of live rate-limit buckets. Prevents memory
    /// exhaustion from an attacker spraying unique client IPs.
    pub(super) const MAX_BUCKETS: usize = 100_000;

    /// Sliding-window rate limiter with one bucket per caller-supplied key.
    ///
    /// `ai_proxy` uses `key:sha256(api_key)` / `sess:session_id` / `ip:addr`
    /// in that priority order so that anonymous traffic still gets a bucket.
    pub(super) struct KeyRateLimiter {
        buckets: DashMap<String, Mutex<VecDeque<Instant>>>,
        window: Duration,
        max_requests: u32,
    }

    impl KeyRateLimiter {
        pub fn new(window_secs: u64, max_requests: u32) -> Self {
            Self {
                buckets: DashMap::new(),
                window: Duration::from_secs(window_secs.max(1)),
                max_requests: max_requests.max(1),
            }
        }

        /// Attempt to acquire a slot for `key`. On success, records the
        /// timestamp in the bucket and returns `Ok(())`. On rejection,
        /// returns the approximate `retry_in` duration.
        pub fn try_acquire(&self, key: &str) -> Result<(), Duration> {
            // Bound the number of buckets. If we're at the cap and this key
            // is new, drop some arbitrary existing bucket to make room.
            if self.buckets.len() >= MAX_BUCKETS && !self.buckets.contains_key(key) {
                if let Some(stale) = self.buckets.iter().next().map(|e| e.key().clone()) {
                    self.buckets.remove(&stale);
                }
            }

            let bucket = self
                .buckets
                .entry(key.to_string())
                .or_insert_with(|| Mutex::new(VecDeque::new()));
            let mut q = bucket.lock();
            let now = Instant::now();
            let cutoff = now.checked_sub(self.window).unwrap_or(now);

            // Drop expired timestamps.
            while let Some(&front) = q.front() {
                if front < cutoff {
                    q.pop_front();
                } else {
                    break;
                }
            }

            if (q.len() as u32) < self.max_requests {
                q.push_back(now);
                Ok(())
            } else {
                let oldest = q.front().copied().unwrap_or(now);
                let retry_in = (oldest + self.window).saturating_duration_since(now);
                Err(retry_in)
            }
        }

        pub fn bucket_count(&self) -> usize {
            self.buckets.len()
        }

        /// Drop buckets whose most-recent timestamp is older than
        /// `window * 2`. Intended to be called periodically from a background
        /// task so the map doesn't grow without bound.
        #[allow(dead_code)] // Background cleanup task is wired in WS-2.
        pub fn cleanup_stale(&self) {
            let cutoff = Instant::now()
                .checked_sub(self.window * 2)
                .unwrap_or_else(Instant::now);
            self.buckets.retain(|_, v| match v.lock().back() {
                Some(&t) => t >= cutoff,
                None => false,
            });
        }
    }
}

// ============================================================================
// V78 / WS-4: Audit log writer (JSONL, append-only, rotated, symlink-safe)
// ============================================================================

#[cfg(feature = "security")]
#[allow(dead_code)] // All items consumed by WS-2 (request path wiring).
mod audit {
    use chrono::Utc;
    use parking_lot::Mutex;
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};

    /// Tagged audit outcome. Serializes as `{"kind":"blocked","reason":"pii"}`.
    #[derive(Serialize, Debug, Clone)]
    #[serde(tag = "kind", content = "reason", rename_all = "snake_case")]
    #[allow(dead_code)] // Variants consumed by WS-2 (request path wiring).
    pub(super) enum AuditOutcome {
        Ok,
        Blocked(String),
        BudgetBlock(String),
        OutputBlocked(String),
        CacheHit,
        Streamed,
        Error(String),
    }

    /// One line of the JSONL audit log.
    #[derive(Serialize, Debug)]
    pub(super) struct AuditEntry<'a> {
        pub ts: String,
        pub request_id: &'a str,
        pub client: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub key_hash: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub session_id: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<&'a str>,
        pub status: u16,
        pub latency_ms: u64,
        pub prompt_sha256: String,
        pub prompt_tokens_est: u32,
        pub outcome: AuditOutcome,
    }

    /// SHA-256 hex digest of `input`.
    pub(super) fn sha256_hex(input: &str) -> String {
        let mut h = Sha256::new();
        h.update(input.as_bytes());
        hex_encode(&h.finalize()[..])
    }

    /// Hash an API key for logging. NEVER log the raw key.
    pub(super) fn hash_api_key(key: &str) -> String {
        sha256_hex(key)
    }

    /// Short (16-char) SHA-256 prefix of a prompt, for audit compactness.
    pub(super) fn hash_prompt_short(prompt: &str) -> String {
        let full = sha256_hex(prompt);
        full[..16].to_string()
    }

    pub(super) fn rfc3339_now() -> String {
        Utc::now().to_rfc3339()
    }

    fn hex_encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0xf) as usize] as char);
        }
        out
    }

    /// Append-only JSONL audit log writer with size- and count-based rotation.
    ///
    /// Security posture:
    /// - **Symlink rejection**: the target path is checked with
    ///   `symlink_metadata` before opening, and on Unix the file is opened
    ///   with `O_NOFOLLOW` for race-free enforcement.
    /// - **Bounded growth**: rotates on size overflow, keeps at most
    ///   `max_files` archives.
    /// - **No raw secrets**: callers are expected to hash API keys with
    ///   [`hash_api_key`] before putting them in `AuditEntry::key_hash`.
    pub(super) struct AuditWriter {
        path: PathBuf,
        max_files: u32,
        max_bytes: u64,
        inner: Mutex<Inner>,
    }

    struct Inner {
        file: BufWriter<File>,
        current_size: u64,
        entries_since_flush: u32,
    }

    impl AuditWriter {
        pub fn open(
            path: impl Into<PathBuf>,
            max_files: u32,
            max_bytes: u64,
        ) -> std::io::Result<Self> {
            let path = path.into();
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            reject_symlink(&path)?;
            let file = open_nofollow_append(&path)?;
            let current_size = file.metadata()?.len();
            Ok(Self {
                path,
                max_files: max_files.max(1),
                max_bytes: max_bytes.max(1024),
                inner: Mutex::new(Inner {
                    file: BufWriter::new(file),
                    current_size,
                    entries_since_flush: 0,
                }),
            })
        }

        /// Append a serialized entry with a trailing newline. Rotates
        /// automatically if the next write would exceed `max_bytes`.
        /// Flushes every 16 entries and on rotation.
        pub fn write_entry(&self, entry: &AuditEntry<'_>) -> std::io::Result<()> {
            let json = serde_json::to_string(entry)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let line_len = json.len() as u64 + 1;

            let mut inner = self.inner.lock();

            if inner.current_size + line_len > self.max_bytes {
                inner.file.flush()?;
                drop(inner);
                self.rotate()?;
                // Re-open the (now truncated) current file.
                let file = open_nofollow_append(&self.path)?;
                let current_size = file.metadata()?.len();
                let mut fresh = self.inner.lock();
                fresh.file = BufWriter::new(file);
                fresh.current_size = current_size;
                fresh.entries_since_flush = 0;
                inner = fresh;
            }

            inner.file.write_all(json.as_bytes())?;
            inner.file.write_all(b"\n")?;
            inner.current_size += line_len;
            inner.entries_since_flush += 1;
            if inner.entries_since_flush >= 16 {
                inner.file.flush()?;
                inner.entries_since_flush = 0;
            }
            Ok(())
        }

        pub fn flush(&self) -> std::io::Result<()> {
            let mut inner = self.inner.lock();
            inner.file.flush()?;
            inner.entries_since_flush = 0;
            Ok(())
        }

        #[cfg(test)]
        #[allow(dead_code)]
        pub fn path(&self) -> &Path {
            &self.path
        }

        fn rotate(&self) -> std::io::Result<()> {
            // Remove oldest archive (audit.jsonl.N), shift the others down
            // by one, then rename audit.jsonl -> audit.jsonl.1.
            let base = &self.path;
            let oldest = numbered(base, self.max_files);
            if oldest.exists() {
                std::fs::remove_file(&oldest)?;
            }
            for i in (1..self.max_files).rev() {
                let src = numbered(base, i);
                let dst = numbered(base, i + 1);
                if src.exists() {
                    std::fs::rename(&src, &dst)?;
                }
            }
            if base.exists() {
                std::fs::rename(base, numbered(base, 1))?;
            }
            Ok(())
        }
    }

    fn numbered(base: &Path, n: u32) -> PathBuf {
        let mut buf = base.as_os_str().to_os_string();
        buf.push(format!(".{}", n));
        PathBuf::from(buf)
    }

    #[cfg(unix)]
    fn open_nofollow_append(path: &Path) -> std::io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .create(true)
            .append(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }

    #[cfg(not(unix))]
    fn open_nofollow_append(path: &Path) -> std::io::Result<File> {
        OpenOptions::new().create(true).append(true).open(path)
    }

    fn reject_symlink(path: &Path) -> std::io::Result<()> {
        match std::fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_symlink() => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("refusing to follow symlink at {}", path.display()),
            )),
            _ => Ok(()),
        }
    }
}

// ============================================================================
// V78 / WS-5: Budget middleware wrapper
// ============================================================================

#[cfg(feature = "security")]
#[allow(dead_code)] // All items consumed by WS-2 (request path wiring).
mod budget {
    use ai_assistant::cost_integration::{
        CostAwareConfig, CostDecision, CostMiddleware, DefaultCostMiddleware,
    };
    use parking_lot::Mutex;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// Result of calling [`BudgetGate::pre_request`].
    #[derive(Debug, Clone)]
    #[allow(dead_code)] // Consumed by WS-2 (request path wiring).
    pub(super) enum BudgetCheck {
        Allow,
        Warn(String),
        Block(String),
    }

    impl From<CostDecision> for BudgetCheck {
        fn from(d: CostDecision) -> Self {
            match d {
                CostDecision::Allow => BudgetCheck::Allow,
                CostDecision::Warn(m) => BudgetCheck::Warn(m),
                CostDecision::Block(m) => BudgetCheck::Block(m),
                _ => BudgetCheck::Allow, // forward-compatible fallback
            }
        }
    }

    /// Shared, lockable budget gate used by the `ai_proxy` request path.
    ///
    /// The underlying `DefaultCostMiddleware::post_response` takes `&mut self`
    /// so access is serialized through a `parking_lot::Mutex`. For realistic
    /// gateway loads (5–50 req/s) this is fine; V80 can split per-worker.
    pub(super) struct BudgetGate {
        inner: Mutex<DefaultCostMiddleware>,
        snapshot_path: Option<PathBuf>,
    }

    impl BudgetGate {
        pub fn new(config: CostAwareConfig, snapshot_path: Option<PathBuf>) -> Self {
            Self {
                inner: Mutex::new(DefaultCostMiddleware::new(config)),
                snapshot_path,
            }
        }

        /// Estimated wait on budget — `Allow` for green, `Warn` for soft, `Block` for hard.
        pub fn pre_request(&self, model: &str, estimated_input_tokens: usize) -> BudgetCheck {
            let guard = self.inner.lock();
            guard.pre_request(model, estimated_input_tokens).into()
        }

        /// Record an actual response's token usage. Idempotent per request.
        #[allow(dead_code)] // Consumed by WS-2.
        pub fn post_response(&self, model: &str, input_tokens: usize, output_tokens: usize) {
            let mut guard = self.inner.lock();
            let _entry = guard.post_response(model, input_tokens, output_tokens);
        }

        /// Return the optional snapshot path for diagnostics. WS-5 intentionally
        /// does NOT auto-load or auto-flush snapshots — that's wired in WS-2
        /// (restore at startup + periodic 60s flush task).
        #[allow(dead_code)]
        pub fn snapshot_path(&self) -> Option<&Path> {
            self.snapshot_path.as_deref()
        }
    }

    /// Build a [`CostAwareConfig`] from the parsed middleware section.
    /// Returns `None` if `enable_budget = false` (caller skips construction).
    ///
    /// `CostAwareConfig` is `#[non_exhaustive]`, so we start from `default()`
    /// and mutate the fields we care about. Any new fields added upstream
    /// will therefore inherit their default values.
    pub(super) fn config_from_middleware(m: &super::MiddlewareSection) -> Option<CostAwareConfig> {
        if !m.enable_budget {
            return None;
        }
        let mut cfg = CostAwareConfig::default();
        cfg.enabled = true;
        cfg.daily_budget = None;
        cfg.monthly_budget = m.monthly_budget_usd;
        cfg.per_request_limit = m.per_request_limit_usd;
        cfg.alert_threshold_pct = 0.8;
        cfg.track_by_model = true;
        Some(cfg)
    }

    /// Construct an `Arc<BudgetGate>` from the merged config, or `None` if
    /// budgeting is disabled. Exposed as a free function so the call site in
    /// `main()` is a single line.
    pub(super) fn build_gate(m: &super::MiddlewareSection) -> Option<Arc<BudgetGate>> {
        let config = config_from_middleware(m)?;
        let snap_path = m.cost_snapshot_path.as_ref().map(PathBuf::from);
        Some(Arc::new(BudgetGate::new(config, snap_path)))
    }
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

    // Load config file if --config was given.
    let file_config: Option<ProxyConfig> = match cli.config.as_ref() {
        Some(p) => match load_config(p) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                eprintln!("Error: {}", e);
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    // Merge CLI + file into the final Effective settings.
    let effective = match merge_cli_and_config(&cli, file_config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let port = effective.port;
    let health_interval = effective.health_interval;
    let backend_addrs = effective.backend_addrs.clone();

    if cli.dry_run {
        println!("AI Proxy Configuration:");
        println!("  port: {}", port);
        println!("  backends: {:?}", backend_addrs);
        println!("  health-interval: {}s", health_interval);
        println!(
            "  api-key: {}",
            if effective.api_key.is_some() {
                "(set)"
            } else {
                "(none)"
            }
        );
        if cli.config.is_some() {
            println!("  config: {}", cli.config.as_ref().unwrap().display());
            println!(
                "  middleware.enabled_flags: rate_limit={} pii_in={} pii_out={} tox_in={} tox_out={} attack={} budget={} cache={} audit={}",
                effective.middleware.enable_rate_limit,
                effective.middleware.enable_pii_input,
                effective.middleware.enable_pii_output,
                effective.middleware.enable_toxicity_input,
                effective.middleware.enable_toxicity_output,
                effective.middleware.enable_attack_filter,
                effective.middleware.enable_budget,
                effective.middleware.enable_cache,
                effective.audit.enabled,
            );
        }
        return ExitCode::SUCCESS;
    }

    let backends: Vec<Backend> = backend_addrs
        .iter()
        .map(|a| Backend::new(a.clone()))
        .collect();

    let state = ProxyState {
        backends: Arc::new(backends),
        next_index: Arc::new(AtomicUsize::new(0)),
        session_affinity: Arc::new(DashMap::new()),
        api_key: effective.api_key,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Failed to create tokio runtime: {}", e);
            std::process::exit(1);
        });

    if let Ok(info) = update_rx.try_recv() {
        eprintln!(
            "  Update available: v{} \u{2192} v{}",
            info.current, info.latest
        );
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

        // Choose router: if the `security` feature is enabled AND at least one
        // middleware is turned on in the effective config, run the full
        // guardrail gateway. Otherwise fall back to the V77 plain router
        // (routing, load balancing, health checks, session affinity only).
        #[cfg(feature = "security")]
        let app: Router = if any_middleware_enabled(&effective.middleware, &effective.audit) {
            match build_gateway_context(
                state.clone(),
                &effective.middleware,
                &effective.audit,
            ) {
                Ok(ctx) => {
                    eprintln!(
                        "Gateway middleware: rate_limit={} pii_in={} pii_out={} tox_in={} tox_out={} attack={} budget={} cache={} audit={}",
                        effective.middleware.enable_rate_limit,
                        effective.middleware.enable_pii_input,
                        effective.middleware.enable_pii_output,
                        effective.middleware.enable_toxicity_input,
                        effective.middleware.enable_toxicity_output,
                        effective.middleware.enable_attack_filter,
                        effective.middleware.enable_budget,
                        effective.middleware.enable_cache,
                        effective.audit.enabled,
                    );
                    build_gateway_router(ctx)
                }
                Err(e) => {
                    eprintln!("Failed to build gateway context: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            build_proxy_router(state)
        };

        #[cfg(not(feature = "security"))]
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
        status: if all_healthy {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        backends,
    })
}

/// Constant-time Bearer-token check. Returns `true` iff the `Authorization`
/// header matches `expected_key` exactly.
fn check_bearer_auth(req: &Request, expected_key: &str) -> bool {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            if let Some(token) = v.strip_prefix("Bearer ") {
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
        .unwrap_or(false)
}

async fn proxy_forward_handler(State(state): State<ProxyState>, req: Request) -> Response {
    // Check API key if configured
    if let Some(ref expected_key) = state.api_key {
        if !check_bearer_auth(&req, expected_key) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ProxyError {
                    error: "Unauthorized".to_string(),
                }),
            )
                .into_response();
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
            Json(ProxyError {
                error: "No healthy backends available".to_string(),
            }),
        )
            .into_response();
    }

    // Forward the request
    let (parts, body) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
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
                Json(ProxyError {
                    error: "Method not allowed".to_string(),
                }),
            )
                .into_response();
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
                Json(ProxyError {
                    error: format!("Failed to read request body: {}", e),
                }),
            )
                .into_response();
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
                Ok(bytes) => response_builder
                    .body(Body::from(bytes))
                    .unwrap_or_else(|_| {
                        (StatusCode::INTERNAL_SERVER_ERROR, "Internal error").into_response()
                    }),
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(ProxyError {
                        error: format!("Backend read error: {}", e),
                    }),
                )
                    .into_response(),
            }
        }
        Err(e) => {
            // Mark backend as unhealthy on connection error
            if e.is_connect() || e.is_timeout() {
                backend.healthy.store(false, Ordering::Relaxed);
            }
            (
                StatusCode::BAD_GATEWAY,
                Json(ProxyError {
                    error: format!("Backend error: {}", e),
                }),
            )
                .into_response()
        }
    }
}

// ============================================================================
// V78 / WS-2: Gateway context + hardened request path
// ============================================================================

/// Aggregated shared state used by the hardened request path. Only built
/// when `security` feature is on AND at least one middleware is enabled;
/// otherwise `main()` falls back to the lightweight `ProxyState` router.
#[cfg(feature = "security")]
#[derive(Clone)]
struct GatewayContext {
    proxy: ProxyState,
    pipeline: Arc<parking_lot::Mutex<ai_assistant::guardrail_pipeline::GuardrailPipeline>>,
    cache: Option<Arc<cache::ResponseCache>>,
    rate_limiter: Option<Arc<rate_limit::KeyRateLimiter>>,
    budget: Option<Arc<budget::BudgetGate>>,
    audit: Option<Arc<audit::AuditWriter>>,
    middleware_cfg: Arc<MiddlewareSection>,
}

/// Header name used for the echoed request ID.
#[cfg(feature = "security")]
const X_REQUEST_ID: &str = "x-request-id";
/// Header name for the cache hit/miss indicator.
#[cfg(feature = "security")]
const X_CACHE: &str = "x-cache";
/// Header name for the block reason (on 4xx / 5xx hardened responses).
#[cfg(feature = "security")]
const X_REASON: &str = "x-reason";

/// Maximum request body size read by the gateway handler (16 MiB).
#[cfg(feature = "security")]
const MAX_REQUEST_BODY: usize = 16 * 1024 * 1024;

/// Build a fully-configured [`GatewayContext`] from the effective settings.
#[cfg(feature = "security")]
fn build_gateway_context(
    proxy: ProxyState,
    m: &MiddlewareSection,
    audit_cfg: &AuditSection,
) -> Result<GatewayContext, String> {
    use ai_assistant::guardrail_pipeline::{
        AttackGuard, ContentLengthGuard, GuardrailPipeline, PiiGuard, ToxicityGuard,
    };

    let mut pipeline = GuardrailPipeline::new().with_threshold(0.8);
    pipeline.add_guard(Box::new(ContentLengthGuard::new(65_536)));
    if m.enable_pii_input || m.enable_pii_output {
        pipeline.add_guard(Box::new(PiiGuard::new()));
    }
    if m.enable_toxicity_input || m.enable_toxicity_output {
        pipeline.add_guard(Box::new(ToxicityGuard::new()));
    }
    if m.enable_attack_filter {
        pipeline.add_guard(Box::new(AttackGuard::new()));
    }

    let cache = if m.enable_cache {
        Some(Arc::new(cache::ResponseCache::new(
            m.cache_max_entries.unwrap_or(10_000),
            m.cache_ttl_secs.unwrap_or(3600),
        )))
    } else {
        None
    };

    let rate_limiter = if m.enable_rate_limit {
        Some(Arc::new(rate_limit::KeyRateLimiter::new(
            60,
            m.rate_limit_rpm.unwrap_or(60),
        )))
    } else {
        None
    };

    let budget_gate = budget::build_gate(m);

    let audit_writer = if audit_cfg.enabled {
        let path = audit_cfg
            .path
            .clone()
            .unwrap_or_else(|| "./ai_proxy_audit.jsonl".to_string());
        let max_files = audit_cfg.max_files.unwrap_or(10);
        let max_bytes = audit_cfg.max_bytes_per_file.unwrap_or(10 * 1024 * 1024);
        match audit::AuditWriter::open(&path, max_files, max_bytes) {
            Ok(w) => Some(Arc::new(w)),
            Err(e) => {
                return Err(format!("failed to open audit log at '{}': {}", path, e));
            }
        }
    } else {
        None
    };

    Ok(GatewayContext {
        proxy,
        pipeline: Arc::new(parking_lot::Mutex::new(pipeline)),
        cache,
        rate_limiter,
        budget: budget_gate,
        audit: audit_writer,
        middleware_cfg: Arc::new(m.clone()),
    })
}

/// Returns `true` if any middleware in the section is enabled. Used by
/// `main()` to decide whether to build a gateway router or fall back to the
/// plain passthrough router.
#[cfg(feature = "security")]
fn any_middleware_enabled(m: &MiddlewareSection, a: &AuditSection) -> bool {
    m.enable_rate_limit
        || m.enable_pii_input
        || m.enable_pii_output
        || m.enable_toxicity_input
        || m.enable_toxicity_output
        || m.enable_attack_filter
        || m.enable_budget
        || m.enable_cache
        || a.enabled
}

/// Build the axum router for the hardened gateway path. `/health` is served
/// locally, `/v1/chat/completions` goes through the full pipeline, everything
/// else falls through to a passthrough handler that still writes audit.
#[cfg(feature = "security")]
fn build_gateway_router(ctx: GatewayContext) -> Router {
    use axum::routing::{any, post};
    Router::new()
        .route("/health", get(gateway_health_handler))
        .route("/v1/chat/completions", post(gateway_chat_handler))
        .fallback(any(gateway_passthrough_handler))
        .with_state(ctx)
}

#[cfg(feature = "security")]
async fn gateway_health_handler(State(ctx): State<GatewayContext>) -> Json<ProxyHealthResponse> {
    let backends: Vec<BackendStatus> = ctx
        .proxy
        .backends
        .iter()
        .map(|b| BackendStatus {
            addr: b.addr.clone(),
            healthy: b.healthy.load(Ordering::Relaxed),
        })
        .collect();
    let all_healthy = backends.iter().any(|b| b.healthy);
    Json(ProxyHealthResponse {
        status: if all_healthy {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        backends,
    })
}

/// Fallback handler used for every path other than `/v1/chat/completions`.
/// Runs the SAME auth + rate-limit checks as the chat handler, then forwards
/// the request unmodified (no guardrails, no cache).
#[cfg(feature = "security")]
async fn gateway_passthrough_handler(State(ctx): State<GatewayContext>, req: Request) -> Response {
    // Auth
    if let Some(ref key) = ctx.proxy.api_key {
        if !check_bearer_auth(&req, key) {
            return unauthorized();
        }
    }
    // Rate limit using whichever key dimension exists.
    let rate_key = pick_rate_limit_key(&req, ctx.proxy.api_key.as_deref());
    if let Some(ref rl) = ctx.rate_limiter {
        if let Err(retry) = rl.try_acquire(&rate_key) {
            return rate_limited(retry);
        }
    }

    let request_id = uuid::Uuid::new_v4().to_string();
    let start = std::time::Instant::now();
    let client_ip = extract_client_ip(&req);
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b.to_vec(),
        Err(e) => {
            return with_request_id_header(
                bad_request(format!("Failed to read request body: {e}")),
                &request_id,
            );
        }
    };

    let (status, headers, resp_body) = match forward_core(&ctx.proxy, &parts, body_bytes).await {
        Ok(tuple) => tuple,
        Err(resp) => return with_request_id_header(resp, &request_id),
    };

    // Audit entry (best-effort).
    write_audit(
        &ctx,
        &request_id,
        &client_ip,
        None, // session id omitted on passthrough
        None, // model unknown on passthrough
        status.as_u16(),
        start.elapsed().as_millis() as u64,
        "",
        0,
        audit::AuditOutcome::Ok,
    );

    let mut builder = Response::builder().status(status);
    for (k, v) in headers.iter() {
        builder = builder.header(k, v);
    }
    builder = builder.header(X_REQUEST_ID, &request_id);
    builder
        .body(Body::from(resp_body))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Internal error").into_response())
}

/// Hardened handler for `/v1/chat/completions`.
#[cfg(feature = "security")]
async fn gateway_chat_handler(State(ctx): State<GatewayContext>, req: Request) -> Response {
    // Auth
    if let Some(ref key) = ctx.proxy.api_key {
        if !check_bearer_auth(&req, key) {
            return unauthorized();
        }
    }

    let request_id = uuid::Uuid::new_v4().to_string();
    let start = std::time::Instant::now();
    let client_ip = extract_client_ip(&req);
    let session_id = req
        .headers()
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Rate-limit first so abusers don't exercise the JSON parser.
    let rate_key = pick_rate_limit_key(&req, ctx.proxy.api_key.as_deref());
    if let Some(ref rl) = ctx.rate_limiter {
        if let Err(retry) = rl.try_acquire(&rate_key) {
            write_audit(
                &ctx,
                &request_id,
                &client_ip,
                session_id.as_deref(),
                None,
                429,
                0,
                "",
                0,
                audit::AuditOutcome::Blocked("rate_limit".to_string()),
            );
            return with_request_id_header(rate_limited(retry), &request_id);
        }
    }

    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b.to_vec(),
        Err(e) => {
            return with_request_id_header(
                bad_request(format!("Failed to read request body: {e}")),
                &request_id,
            );
        }
    };

    // Parse body as JSON; if it fails, reject with 400.
    let json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return with_request_id_header(bad_request(format!("Invalid JSON: {e}")), &request_id);
        }
    };

    // Streaming: pass through unmodified (V78 policy).
    let is_stream = json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if is_stream {
        let (status, headers, resp_body) = match forward_core(&ctx.proxy, &parts, body_bytes).await
        {
            Ok(t) => t,
            Err(resp) => return with_request_id_header(resp, &request_id),
        };
        write_audit(
            &ctx,
            &request_id,
            &client_ip,
            session_id.as_deref(),
            json.get("model").and_then(|v| v.as_str()),
            status.as_u16(),
            start.elapsed().as_millis() as u64,
            "",
            0,
            audit::AuditOutcome::Streamed,
        );
        return build_response(status, headers, resp_body, &request_id, None);
    }

    // Build scan_text from messages[].content for roles in {user, system}.
    let scan_text = extract_scan_text(&json);
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let prompt_hash = audit::hash_prompt_short(&scan_text);
    let prompt_tokens_est = (scan_text.chars().count() as u32 / 4).max(1);

    // Input pipeline check.
    {
        let mut pipeline = ctx.pipeline.lock();
        let result = pipeline.check_input(&scan_text);
        if !result.passed {
            let reason = result
                .blocked_by
                .unwrap_or_else(|| "input_guard".to_string());
            write_audit(
                &ctx,
                &request_id,
                &client_ip,
                session_id.as_deref(),
                Some(model),
                403,
                start.elapsed().as_millis() as u64,
                &prompt_hash,
                prompt_tokens_est,
                audit::AuditOutcome::Blocked(reason.clone()),
            );
            return with_request_id_header(blocked(&reason), &request_id);
        }
    }

    // Budget check.
    if let Some(ref gate) = ctx.budget {
        match gate.pre_request(model, prompt_tokens_est as usize) {
            budget::BudgetCheck::Block(r) => {
                write_audit(
                    &ctx,
                    &request_id,
                    &client_ip,
                    session_id.as_deref(),
                    Some(model),
                    429,
                    start.elapsed().as_millis() as u64,
                    &prompt_hash,
                    prompt_tokens_est,
                    audit::AuditOutcome::BudgetBlock(r.clone()),
                );
                return with_request_id_header(budget_exceeded(&r), &request_id);
            }
            budget::BudgetCheck::Warn(_) | budget::BudgetCheck::Allow => {}
        }
    }

    // Cache lookup. Build a CacheKey from model + sampling params + scan_text.
    let temperature = json
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let max_tokens = json
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .unwrap_or(0);
    let cache_key = cache::CacheKey::new(model, temperature, max_tokens, &scan_text);

    let respect_no_cache = parts
        .headers
        .get("cache-control")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("no-cache") || v.contains("no-store"))
        .unwrap_or(false);

    if !respect_no_cache {
        if let Some(ref c) = ctx.cache {
            if let Some(hit) = c.get(&cache_key) {
                write_audit(
                    &ctx,
                    &request_id,
                    &client_ip,
                    session_id.as_deref(),
                    Some(model),
                    hit.status,
                    start.elapsed().as_millis() as u64,
                    &prompt_hash,
                    prompt_tokens_est,
                    audit::AuditOutcome::CacheHit,
                );
                return build_response(
                    StatusCode::from_u16(hit.status).unwrap_or(StatusCode::OK),
                    axum::http::HeaderMap::new(),
                    hit.body,
                    &request_id,
                    Some("HIT"),
                );
            }
        }
    }

    // Forward to backend.
    let (status, headers, resp_body) = match forward_core(&ctx.proxy, &parts, body_bytes).await {
        Ok(t) => t,
        Err(resp) => return with_request_id_header(resp, &request_id),
    };

    // Parse response body to run output pipeline and extract usage.
    let (output_text, usage_in, usage_out) = extract_response_text_and_usage(&resp_body);

    // Output pipeline check. When the pipeline blocks, we return 503 and never
    // reach the cache-store branch, so no `pii_free=false` flag needs to be
    // propagated.
    let pii_free = true;
    {
        let mut pipeline = ctx.pipeline.lock();
        let result = pipeline.check_output(&output_text);
        if !result.passed {
            let reason = result
                .blocked_by
                .unwrap_or_else(|| "output_guard".to_string());
            write_audit(
                &ctx,
                &request_id,
                &client_ip,
                session_id.as_deref(),
                Some(model),
                503,
                start.elapsed().as_millis() as u64,
                &prompt_hash,
                prompt_tokens_est,
                audit::AuditOutcome::OutputBlocked(reason.clone()),
            );
            return with_request_id_header(output_blocked(&reason), &request_id);
        }
    }

    // Update budget with actual usage.
    if let Some(ref gate) = ctx.budget {
        gate.post_response(model, usage_in, usage_out);
    }

    // Store in cache (only if everything looks clean).
    if !respect_no_cache && pii_free {
        if let Some(ref c) = ctx.cache {
            let entry = cache::CachedResponse {
                body: resp_body.clone(),
                status: status.as_u16(),
                stored_at: std::time::Instant::now(),
                pii_free: true,
            };
            c.put(cache_key, entry);
        }
    }

    write_audit(
        &ctx,
        &request_id,
        &client_ip,
        session_id.as_deref(),
        Some(model),
        status.as_u16(),
        start.elapsed().as_millis() as u64,
        &prompt_hash,
        prompt_tokens_est,
        audit::AuditOutcome::Ok,
    );

    build_response(status, headers, resp_body, &request_id, Some("MISS"))
}

// --- V78 / WS-2 helpers ----------------------------------------------------

/// Forward the request to a healthy backend using the existing routing logic
/// (session affinity → round-robin). Returns the tuple `(status, headers,
/// body)` on success, or an already-built error [`Response`] on failure.
#[cfg(feature = "security")]
async fn forward_core(
    state: &ProxyState,
    parts: &axum::http::request::Parts,
    body_bytes: Vec<u8>,
) -> Result<(StatusCode, axum::http::HeaderMap, Vec<u8>), Response> {
    let session_id = parts
        .headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let backend_idx = if let Some(ref sid) = session_id {
        if let Some(idx) = state.session_affinity.get(sid).map(|r| *r) {
            if state.backends[idx].healthy.load(Ordering::Relaxed) {
                idx
            } else {
                pick_healthy_backend(state)
            }
        } else {
            let idx = pick_healthy_backend(state);
            state.session_affinity.insert(sid.clone(), idx);
            idx
        }
    } else {
        pick_healthy_backend(state)
    };
    let backend = &state.backends[backend_idx];
    if !backend.healthy.load(Ordering::Relaxed) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ProxyError {
                error: "No healthy backends available".to_string(),
            }),
        )
            .into_response());
    }

    let path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let target_url = format!("http://{}{}", backend.addr, path);
    let client = reqwest::Client::new();
    let mut builder = match parts.method {
        axum::http::Method::GET => client.get(&target_url),
        axum::http::Method::POST => client.post(&target_url),
        axum::http::Method::PUT => client.put(&target_url),
        axum::http::Method::DELETE => client.delete(&target_url),
        axum::http::Method::PATCH => client.patch(&target_url),
        axum::http::Method::HEAD => client.head(&target_url),
        _ => {
            return Err((
                StatusCode::METHOD_NOT_ALLOWED,
                Json(ProxyError {
                    error: "Method not allowed".to_string(),
                }),
            )
                .into_response());
        }
    };
    for (name, value) in parts.headers.iter() {
        if name != header::HOST {
            if let Ok(v) = value.to_str() {
                builder = builder.header(name.as_str(), v);
            }
        }
    }
    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    match builder.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut headers = axum::http::HeaderMap::new();
            for (name, value) in resp.headers().iter() {
                if let (Ok(hn), Ok(hv)) = (
                    axum::http::HeaderName::from_bytes(name.as_ref()),
                    axum::http::HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    headers.insert(hn, hv);
                }
            }
            match resp.bytes().await {
                Ok(bytes) => Ok((status, headers, bytes.to_vec())),
                Err(e) => Err((
                    StatusCode::BAD_GATEWAY,
                    Json(ProxyError {
                        error: format!("Backend read error: {e}"),
                    }),
                )
                    .into_response()),
            }
        }
        Err(e) => {
            if e.is_connect() || e.is_timeout() {
                backend.healthy.store(false, Ordering::Relaxed);
            }
            Err((
                StatusCode::BAD_GATEWAY,
                Json(ProxyError {
                    error: format!("Backend error: {e}"),
                }),
            )
                .into_response())
        }
    }
}

#[cfg(feature = "security")]
fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ProxyError {
            error: "Unauthorized".to_string(),
        }),
    )
        .into_response()
}

#[cfg(feature = "security")]
fn rate_limited(retry_in: std::time::Duration) -> Response {
    let retry_secs = retry_in.as_secs().max(1);
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(X_REASON, "rate_limit"), ("retry-after", "")],
        Json(ProxyError {
            error: format!("Rate limit exceeded, retry in {retry_secs}s"),
        }),
    )
        .into_response()
}

#[cfg(feature = "security")]
fn blocked(reason: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        [(X_REASON, "input_guard")],
        Json(ProxyError {
            error: format!("Blocked by input guard: {reason}"),
        }),
    )
        .into_response()
}

#[cfg(feature = "security")]
fn budget_exceeded(reason: &str) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(X_REASON, "budget_exceeded")],
        Json(ProxyError {
            error: format!("Budget exceeded: {reason}"),
        }),
    )
        .into_response()
}

#[cfg(feature = "security")]
fn output_blocked(reason: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(X_REASON, "output_guard")],
        Json(ProxyError {
            error: format!("Blocked by output guard: {reason}"),
        }),
    )
        .into_response()
}

#[cfg(feature = "security")]
fn bad_request(msg: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(ProxyError { error: msg })).into_response()
}

#[cfg(feature = "security")]
fn with_request_id_header(resp: Response, request_id: &str) -> Response {
    let (mut parts, body) = resp.into_parts();
    if let Ok(v) = axum::http::HeaderValue::from_str(request_id) {
        parts.headers.insert(X_REQUEST_ID, v);
    }
    Response::from_parts(parts, body)
}

#[cfg(feature = "security")]
fn build_response(
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: Vec<u8>,
    request_id: &str,
    cache_marker: Option<&'static str>,
) -> Response {
    let mut builder = Response::builder().status(status);
    for (k, v) in headers.iter() {
        builder = builder.header(k, v);
    }
    builder = builder.header(X_REQUEST_ID, request_id);
    if let Some(marker) = cache_marker {
        builder = builder.header(X_CACHE, marker);
    }
    builder
        .body(Body::from(body))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Internal error").into_response())
}

/// Extract the scan text from a chat/completions JSON body: concatenates
/// `messages[].content` for roles in {user, system}.
#[cfg(feature = "security")]
fn extract_scan_text(json: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(messages) = json.get("messages").and_then(|v| v.as_array()) {
        for m in messages {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role == "user" || role == "system" {
                if let Some(content) = m.get("content").and_then(|v| v.as_str()) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(content);
                }
            }
        }
    } else if let Some(prompt) = json.get("prompt").and_then(|v| v.as_str()) {
        out.push_str(prompt);
    }
    out
}

/// Extract `choices[].message.content` text for output pipeline scan, plus
/// `usage.prompt_tokens` and `usage.completion_tokens` for budget updates.
#[cfg(feature = "security")]
fn extract_response_text_and_usage(body: &[u8]) -> (String, usize, usize) {
    let json: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return (String::new(), 0, 0),
    };
    let mut out = String::new();
    if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
        for c in choices {
            if let Some(txt) = c
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
            {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(txt);
            } else if let Some(txt) = c.get("text").and_then(|v| v.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(txt);
            }
        }
    }
    let usage_in = json
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let usage_out = json
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    (out, usage_in, usage_out)
}

/// Pick the rate-limit bucket key. Priority: API key hash > session id > client IP.
#[cfg(feature = "security")]
fn pick_rate_limit_key(req: &Request, api_key: Option<&str>) -> String {
    // Hash any Bearer token the client sent (since we may be running without
    // an expected api_key but still want fairness).
    let bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").map(|s| s.to_string()));
    if let Some(token) = bearer.as_deref() {
        // Only use the Bearer key if it matches the expected one, OR if
        // we have no expected key (free mode). Prevents a random token from
        // creating an anonymous bucket that differs across requests.
        if api_key.map(|k| k == token).unwrap_or(true) {
            return format!("key:{}", audit::hash_api_key(token));
        }
    }
    if let Some(sid) = req
        .headers()
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
    {
        return format!("sess:{sid}");
    }
    format!("ip:{}", extract_client_ip(req))
}

#[cfg(feature = "security")]
fn extract_client_ip(req: &Request) -> String {
    req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next().map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Best-effort audit write. Failures are logged but do NOT affect the
/// response to the client.
#[cfg(feature = "security")]
#[allow(clippy::too_many_arguments)]
fn write_audit(
    ctx: &GatewayContext,
    request_id: &str,
    client_ip: &str,
    session_id: Option<&str>,
    model: Option<&str>,
    status: u16,
    latency_ms: u64,
    prompt_sha256: &str,
    prompt_tokens_est: u32,
    outcome: audit::AuditOutcome,
) {
    let Some(ref writer) = ctx.audit else {
        return;
    };
    let entry = audit::AuditEntry {
        ts: audit::rfc3339_now(),
        request_id,
        client: client_ip,
        key_hash: ctx.proxy.api_key.as_ref().map(|k| audit::hash_api_key(k)),
        session_id,
        model,
        status,
        latency_ms,
        prompt_sha256: prompt_sha256.to_string(),
        prompt_tokens_est,
        outcome,
    };
    if let Err(e) = writer.write_entry(&entry) {
        eprintln!("[audit] write failed: {e}");
    }
    // Ensure the compiler believes middleware_cfg is used (WS-6 CLI toggles
    // may read from it later). Also keeps it alive for V79.
    let _ = &ctx.middleware_cfg;
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
    let mut cli = CliArgs::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                let val = next_val(args, i, "--port")?;
                cli.port = Some(
                    val.parse()
                        .map_err(|_| format!("Invalid port: '{}'", val))?,
                );
            }
            "--backends" => {
                i += 1;
                cli.backends = Some(next_val(args, i, "--backends")?);
            }
            "--health-interval" => {
                i += 1;
                let val = next_val(args, i, "--health-interval")?;
                cli.health_interval = Some(
                    val.parse()
                        .map_err(|_| format!("Invalid health-interval: '{}'", val))?,
                );
            }
            "--api-key" => {
                i += 1;
                eprintln!(
                    "warning: --api-key is deprecated and leaks via the process list; \
                     prefer the AI_PROXY_API_KEY environment variable"
                );
                cli.api_key = Some(next_val(args, i, "--api-key")?);
            }
            "--config" => {
                i += 1;
                let val = next_val(args, i, "--config")?;
                cli.config = Some(PathBuf::from(val));
            }
            "--audit-log" => {
                i += 1;
                let val = next_val(args, i, "--audit-log")?;
                cli.audit_log = Some(PathBuf::from(val));
            }
            "--audit-max-files" => {
                i += 1;
                let val = next_val(args, i, "--audit-max-files")?;
                cli.audit_max_files = Some(
                    val.parse()
                        .map_err(|_| format!("Invalid audit-max-files: '{}'", val))?,
                );
            }
            "--enable-pii-redaction" => cli.enable_pii_redaction = true,
            "--disable-cache" => cli.disable_cache = true,
            "--cost-snapshot" => {
                i += 1;
                let val = next_val(args, i, "--cost-snapshot")?;
                cli.cost_snapshot = Some(PathBuf::from(val));
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
    eprintln!("AI Proxy — API Gateway (V78)");
    eprintln!();
    eprintln!("Usage: ai_proxy [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --config <PATH>               TOML config file (see examples/ai_proxy.toml)");
    eprintln!("  --port <PORT>                 Port to listen on (default: 8080)");
    eprintln!("  --backends <ADDR1,ADDR2,...>  Backend addresses (required if no config)");
    eprintln!("  --health-interval <SECS>      Health check interval (default: 30)");
    eprintln!("  --api-key <KEY>               [DEPRECATED] prefer AI_PROXY_API_KEY env variable");
    eprintln!("  --audit-log <PATH>            Path to audit log (JSONL, append-only)");
    eprintln!("  --audit-max-files <N>         Max rotated audit files to keep");
    eprintln!("  --enable-pii-redaction        Force PII input redaction on");
    eprintln!("  --disable-cache               Force response cache off");
    eprintln!("  --cost-snapshot <PATH>        Path to cost-dashboard snapshot (budget mw)");
    eprintln!("  --dry-run                     Print config and exit");
    eprintln!("  -h, --help                    Print this help message");
    eprintln!();
    eprintln!("Environment:");
    eprintln!(
        "  AI_PROXY_API_KEY              API key; takes precedence over config file and --api-key"
    );
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
            "--port",
            "9090",
            "--backends",
            "10.0.0.1:8090,10.0.0.2:8090",
            "--health-interval",
            "15",
            "--api-key",
            "secret",
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

    // ------------------------------------------------------------------
    // V78 / WS-1: config loader + merger
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_args_config_flag() {
        let a = args(&["--config", "/etc/ai_proxy.toml"]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(
            cli.config.as_deref(),
            Some(std::path::Path::new("/etc/ai_proxy.toml"))
        );
    }

    // ------------------------------------------------------------------
    // V78 / WS-6: middleware-level CLI flags
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_args_ws6_flags() {
        let a = args(&[
            "--audit-log",
            "/tmp/audit.jsonl",
            "--audit-max-files",
            "7",
            "--enable-pii-redaction",
            "--disable-cache",
            "--cost-snapshot",
            "/var/lib/ai_proxy/cost.json",
        ]);
        let cli = parse_args(&a).unwrap();
        assert_eq!(
            cli.audit_log.as_deref(),
            Some(std::path::Path::new("/tmp/audit.jsonl"))
        );
        assert_eq!(cli.audit_max_files, Some(7));
        assert!(cli.enable_pii_redaction);
        assert!(cli.disable_cache);
        assert_eq!(
            cli.cost_snapshot.as_deref(),
            Some(std::path::Path::new("/var/lib/ai_proxy/cost.json"))
        );
    }

    #[test]
    fn test_parse_args_audit_max_files_rejects_invalid() {
        let a = args(&["--audit-max-files", "notanumber"]);
        let err = parse_args(&a).unwrap_err();
        assert!(err.contains("audit-max-files"));
    }

    #[test]
    fn test_merge_ws6_cli_overrides_apply() {
        let cli = CliArgs {
            backends: Some("10.0.0.9:8090".to_string()),
            audit_log: Some(PathBuf::from("/tmp/from_cli.jsonl")),
            audit_max_files: Some(3),
            enable_pii_redaction: true,
            disable_cache: true,
            cost_snapshot: Some(PathBuf::from("/tmp/snap.json")),
            ..Default::default()
        };
        // Feed a config file that enables caching so we can prove --disable-cache
        // flips it back off.
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:8080"

            [[backends]]
            addr = "10.0.0.1:8090"

            [middleware]
            enable_cache = true
        "#;
        let file_cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        let eff = merge_cli_and_config(&cli, Some(file_cfg)).unwrap();
        assert!(eff.audit.enabled);
        assert_eq!(eff.audit.path.as_deref(), Some("/tmp/from_cli.jsonl"));
        assert_eq!(eff.audit.max_files, Some(3));
        assert!(eff.middleware.enable_pii_input);
        assert_eq!(eff.middleware.pii_input_strategy.as_deref(), Some("redact"));
        // File said enable_cache = true; CLI --disable-cache must win.
        assert!(!eff.middleware.enable_cache);
        assert_eq!(
            eff.middleware.cost_snapshot_path.as_deref(),
            Some("/tmp/snap.json")
        );
    }

    #[test]
    fn test_parse_config_minimal_ok() {
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:9090"

            [[backends]]
            addr = "10.0.0.1:8090"
        "#;
        let cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        assert_eq!(cfg.server.bind.as_deref(), Some("0.0.0.0:9090"));
        assert_eq!(cfg.backends.len(), 1);
        assert_eq!(cfg.backends[0].addr, "10.0.0.1:8090");
    }

    #[test]
    fn test_parse_config_full_ok() {
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:8080"
            health_check_interval_secs = 10

            [[backends]]
            addr = "10.0.0.1:8090"

            [[backends]]
            addr = "10.0.0.2:8090"

            [middleware]
            enable_rate_limit = true
            rate_limit_rpm = 60
            enable_pii_input = true
            pii_input_strategy = "redact"
            enable_cache = true
            cache_max_entries = 1000
            cache_ttl_secs = 60

            [audit]
            enabled = true
            path = "/var/log/ai_proxy/audit.jsonl"
            max_files = 10
            max_bytes_per_file = 10485760
        "#;
        let cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        assert_eq!(cfg.backends.len(), 2);
        assert!(cfg.middleware.enable_rate_limit);
        assert_eq!(cfg.middleware.rate_limit_rpm, Some(60));
        assert!(cfg.middleware.enable_pii_input);
        assert_eq!(cfg.middleware.pii_input_strategy.as_deref(), Some("redact"));
        assert!(cfg.middleware.enable_cache);
        assert!(cfg.audit.enabled);
        assert_eq!(cfg.audit.max_files, Some(10));
    }

    #[test]
    fn test_parse_config_unknown_field_rejected() {
        // `deny_unknown_fields` must reject typos inside `[middleware]`.
        let toml_text = r#"
            [[backends]]
            addr = "10.0.0.1:8090"

            [middleware]
            enable_cash = true   # typo: should be enable_cache
        "#;
        let err = toml::from_str::<ProxyConfig>(toml_text).unwrap_err();
        assert!(
            err.to_string().contains("enable_cash") || err.to_string().contains("unknown"),
            "expected unknown-field error, got: {err}"
        );
    }

    #[test]
    fn test_parse_config_invalid_toml() {
        let toml_text = "this is not = [ valid toml";
        assert!(toml::from_str::<ProxyConfig>(toml_text).is_err());
    }

    #[test]
    fn test_load_config_file_ok() {
        let dir = std::env::temp_dir().join(format!("ai_proxy_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ok.toml");
        std::fs::write(
            &path,
            r#"
                [[backends]]
                addr = "127.0.0.1:9100"
            "#,
        )
        .unwrap();
        let cfg = load_config(&path).expect("should load");
        assert_eq!(cfg.backends[0].addr, "127.0.0.1:9100");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_config_missing_file() {
        let missing = std::path::Path::new("/nonexistent/ai_proxy_xyz_12345.toml");
        let err = load_config(missing).unwrap_err();
        assert!(err.contains("stat") || err.contains("canonicalize"));
    }

    #[test]
    fn test_load_config_size_cap() {
        let dir = std::env::temp_dir().join(format!("ai_proxy_cap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.toml");
        // Write 2 MiB of a valid key = value repeated.
        let mut body = String::from("addrs = [\n");
        let row = "  \"10.0.0.1:8090\",\n";
        while body.len() < (2 * 1024 * 1024) {
            body.push_str(row);
        }
        body.push_str("]\n");
        std::fs::write(&path, &body).unwrap();
        let err = load_config(&path).unwrap_err();
        assert!(err.contains("exceeds limit"), "got: {err}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_merge_defaults_no_file_requires_backends() {
        let cli = CliArgs::default();
        let err = merge_cli_and_config(&cli, None).unwrap_err();
        assert!(err.contains("backend"));
    }

    #[test]
    fn test_merge_cli_only() {
        let cli = CliArgs {
            port: Some(9090),
            backends: Some("a:1,b:2".to_string()),
            health_interval: Some(15),
            api_key: Some("secret".to_string()),
            ..CliArgs::default()
        };
        let eff = merge_cli_and_config(&cli, None).unwrap();
        assert_eq!(eff.port, 9090);
        assert_eq!(eff.backend_addrs, vec!["a:1", "b:2"]);
        assert_eq!(eff.health_interval, 15);
        assert_eq!(eff.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn test_merge_file_only() {
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:7000"
            health_check_interval_secs = 5

            [[backends]]
            addr = "10.0.0.1:8090"

            [[backends]]
            addr = "10.0.0.2:8090"
        "#;
        let cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        let cli = CliArgs::default();
        let eff = merge_cli_and_config(&cli, Some(cfg)).unwrap();
        assert_eq!(eff.port, 7000);
        assert_eq!(eff.health_interval, 5);
        assert_eq!(eff.backend_addrs.len(), 2);
    }

    #[test]
    fn test_merge_cli_overrides_file() {
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:7000"
            health_check_interval_secs = 5

            [[backends]]
            addr = "10.0.0.1:8090"
        "#;
        let cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        let cli = CliArgs {
            port: Some(9999),
            backends: Some("override:1".to_string()),
            health_interval: Some(60),
            api_key: Some("cli-key".to_string()),
            ..CliArgs::default()
        };
        let eff = merge_cli_and_config(&cli, Some(cfg)).unwrap();
        // CLI wins
        assert_eq!(eff.port, 9999);
        assert_eq!(eff.backend_addrs, vec!["override:1"]);
        assert_eq!(eff.health_interval, 60);
        assert_eq!(eff.api_key.as_deref(), Some("cli-key"));
    }

    // ------------------------------------------------------------------
    // V78 / WS-3: cache + per-key rate limiter
    // ------------------------------------------------------------------

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_key_stable_for_same_inputs() {
        use super::cache::CacheKey;
        let a = CacheKey::new("gpt-3.5", 0.7, 256, "hello world");
        let b = CacheKey::new("gpt-3.5", 0.7, 256, "hello world");
        assert_eq!(a, b);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_key_differs_for_different_temperature() {
        use super::cache::CacheKey;
        let a = CacheKey::new("m", 0.7, 128, "p");
        let b = CacheKey::new("m", 0.8, 128, "p");
        assert_ne!(a, b);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_key_quantizes_temperature() {
        use super::cache::CacheKey;
        // Tiny floating-point noise below 0.5 milli-units should not flip
        // the quantized value.
        let a = CacheKey::new("m", 0.7000001, 128, "p");
        let b = CacheKey::new("m", 0.6999999, 128, "p");
        assert_eq!(a, b);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_put_rejects_pii_tainted() {
        use super::cache::{CacheKey, CachedResponse, ResponseCache};
        use std::time::Instant;
        let c = ResponseCache::new(10, 60);
        let key = CacheKey::new("m", 0.5, 128, "prompt");
        let tainted = CachedResponse {
            body: b"{}".to_vec(),
            status: 200,
            stored_at: Instant::now(),
            pii_free: false,
        };
        assert!(!c.put(key.clone(), tainted));
        assert!(c.get(&key).is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_put_rejects_oversize_body() {
        use super::cache::{CacheKey, CachedResponse, ResponseCache, MAX_BODY_SIZE};
        use std::time::Instant;
        let c = ResponseCache::new(10, 60);
        let key = CacheKey::new("m", 0.5, 128, "prompt");
        let huge = CachedResponse {
            body: vec![0u8; MAX_BODY_SIZE + 1],
            status: 200,
            stored_at: Instant::now(),
            pii_free: true,
        };
        assert!(!c.put(key.clone(), huge));
        assert!(c.get(&key).is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_get_ok_and_ttl_expiry() {
        use super::cache::{CacheKey, CachedResponse, ResponseCache};
        use std::time::Instant;
        let c = ResponseCache::new(10, 1);
        let key = CacheKey::new("m", 0.5, 128, "prompt");
        let entry = CachedResponse {
            body: b"{\"ok\":true}".to_vec(),
            status: 200,
            stored_at: Instant::now(),
            pii_free: true,
        };
        assert!(c.put(key.clone(), entry));
        assert!(c.get(&key).is_some());

        // Simulate expiry by re-inserting with a stale stored_at.
        c.clear();
        let stale = CachedResponse {
            body: b"{}".to_vec(),
            status: 200,
            stored_at: Instant::now() - std::time::Duration::from_secs(5),
            pii_free: true,
        };
        assert!(c.put(key.clone(), stale));
        // get() should detect the TTL breach and evict lazily.
        assert!(c.get(&key).is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_cache_lru_eviction() {
        use super::cache::{CacheKey, CachedResponse, ResponseCache};
        use std::time::Instant;
        let c = ResponseCache::new(2, 60);
        let mk = |prompt: &str| CachedResponse {
            body: prompt.as_bytes().to_vec(),
            status: 200,
            stored_at: Instant::now(),
            pii_free: true,
        };
        let k1 = CacheKey::new("m", 0.5, 128, "p1");
        let k2 = CacheKey::new("m", 0.5, 128, "p2");
        let k3 = CacheKey::new("m", 0.5, 128, "p3");
        assert!(c.put(k1.clone(), mk("p1")));
        assert!(c.put(k2.clone(), mk("p2")));
        assert_eq!(c.len(), 2);
        assert!(c.put(k3.clone(), mk("p3")));
        assert_eq!(c.len(), 2);
        // Oldest (k1) should have been evicted.
        assert!(c.get(&k1).is_none());
        assert!(c.get(&k2).is_some());
        assert!(c.get(&k3).is_some());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_rate_limiter_allows_up_to_max() {
        use super::rate_limit::KeyRateLimiter;
        let rl = KeyRateLimiter::new(60, 3);
        assert!(rl.try_acquire("key-a").is_ok());
        assert!(rl.try_acquire("key-a").is_ok());
        assert!(rl.try_acquire("key-a").is_ok());
        // 4th within the window should fail.
        let err = rl.try_acquire("key-a");
        assert!(err.is_err());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_rate_limiter_independent_keys() {
        use super::rate_limit::KeyRateLimiter;
        let rl = KeyRateLimiter::new(60, 1);
        assert!(rl.try_acquire("alice").is_ok());
        // alice is blocked but bob isn't.
        assert!(rl.try_acquire("alice").is_err());
        assert!(rl.try_acquire("bob").is_ok());
    }

    // ------------------------------------------------------------------
    // V78 / WS-4: audit log writer
    // ------------------------------------------------------------------

    #[cfg(feature = "security")]
    #[test]
    fn test_audit_hash_api_key_stable_and_not_plaintext() {
        use super::audit::hash_api_key;
        let k = "sk-very-secret-value";
        let h1 = hash_api_key(k);
        let h2 = hash_api_key(k);
        assert_eq!(h1, h2);
        assert_ne!(h1, k);
        assert_eq!(h1.len(), 64); // SHA-256 hex
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_audit_hash_prompt_short_length() {
        use super::audit::hash_prompt_short;
        let h = hash_prompt_short("hello world");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_audit_entry_serializes_to_jsonl() {
        use super::audit::{AuditEntry, AuditOutcome};
        let entry = AuditEntry {
            ts: "2026-04-11T00:00:00Z".to_string(),
            request_id: "req-123",
            client: "127.0.0.1",
            key_hash: Some("abcd".to_string()),
            session_id: Some("sess-1"),
            model: Some("gpt-3.5"),
            status: 200,
            latency_ms: 42,
            prompt_sha256: "deadbeefcafebabe".to_string(),
            prompt_tokens_est: 16,
            outcome: AuditOutcome::Ok,
        };
        let json = serde_json::to_string(&entry).unwrap();
        // Must be a single-line JSON object (no embedded newlines).
        assert!(!json.contains('\n'));
        // Sanity-check key fields are present.
        assert!(json.contains("\"request_id\":\"req-123\""));
        assert!(json.contains("\"outcome\":{\"kind\":\"ok\"}"));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_audit_writer_append_and_read_back() {
        use super::audit::{AuditEntry, AuditOutcome, AuditWriter};
        let dir = std::env::temp_dir().join(format!("ai_proxy_audit_a_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.jsonl");
        let _ = std::fs::remove_file(&path);

        let w = AuditWriter::open(&path, 3, 10 * 1024 * 1024).unwrap();
        let entry = AuditEntry {
            ts: "t".to_string(),
            request_id: "r1",
            client: "1.2.3.4",
            key_hash: None,
            session_id: None,
            model: Some("m"),
            status: 200,
            latency_ms: 5,
            prompt_sha256: "x".to_string(),
            prompt_tokens_est: 0,
            outcome: AuditOutcome::Ok,
        };
        w.write_entry(&entry).unwrap();
        w.flush().unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let line = contents.lines().next().expect("must have one line");
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(parsed["request_id"], "r1");
        assert_eq!(parsed["status"], 200);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_audit_writer_rotation_on_size() {
        use super::audit::{AuditEntry, AuditOutcome, AuditWriter};
        let dir = std::env::temp_dir().join(format!("ai_proxy_audit_r_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.jsonl");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(dir.join("audit.jsonl.1"));
        let _ = std::fs::remove_file(dir.join("audit.jsonl.2"));

        // Max 2 archives, very small max_bytes so every write rotates.
        let w = AuditWriter::open(&path, 2, 1024).unwrap();
        for i in 0..10 {
            let req = format!("r{}", i);
            let entry = AuditEntry {
                ts: "t".to_string(),
                request_id: &req,
                client: "1.2.3.4",
                key_hash: None,
                session_id: None,
                model: None,
                status: 200,
                latency_ms: 0,
                prompt_sha256: "x".repeat(200), // big enough to force rotation
                prompt_tokens_est: 0,
                outcome: AuditOutcome::Ok,
            };
            w.write_entry(&entry).unwrap();
        }
        w.flush().unwrap();

        // The current file should still exist, and audit.jsonl.1 should
        // exist. audit.jsonl.3 should NOT exist (max_files = 2).
        assert!(path.exists());
        assert!(dir.join("audit.jsonl.1").exists());
        assert!(!dir.join("audit.jsonl.3").exists());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(dir.join("audit.jsonl.1"));
        let _ = std::fs::remove_file(dir.join("audit.jsonl.2"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(all(feature = "security", unix))]
    #[test]
    fn test_audit_writer_rejects_symlink() {
        use super::audit::AuditWriter;
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!("ai_proxy_audit_sym_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("real.jsonl");
        let link = dir.join("audit.jsonl");
        std::fs::write(&target, b"").unwrap();
        let _ = std::fs::remove_file(&link);
        symlink(&target, &link).unwrap();

        let err = AuditWriter::open(&link, 3, 1024 * 1024).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);

        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // V78 / WS-5: budget middleware wrapper
    // ------------------------------------------------------------------

    #[cfg(feature = "security")]
    #[test]
    fn test_budget_disabled_returns_none() {
        use super::budget::build_gate;
        let m = MiddlewareSection::default(); // enable_budget = false
        assert!(build_gate(&m).is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_budget_enabled_builds_gate_and_allows_under_limit() {
        use super::budget::{build_gate, BudgetCheck};
        let mut m = MiddlewareSection::default();
        m.enable_budget = true;
        m.monthly_budget_usd = Some(1000.0);
        m.per_request_limit_usd = Some(100.0);
        let gate = build_gate(&m).expect("gate should be built");
        match gate.pre_request("gpt-3.5-turbo", 100) {
            BudgetCheck::Allow | BudgetCheck::Warn(_) => {}
            BudgetCheck::Block(r) => panic!("expected allow, got Block({r})"),
        }
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_budget_config_from_middleware_picks_fields() {
        use super::budget::config_from_middleware;
        let mut m = MiddlewareSection::default();
        m.enable_budget = true;
        m.monthly_budget_usd = Some(500.0);
        m.per_request_limit_usd = Some(2.5);
        let cfg = config_from_middleware(&m).expect("cfg should be Some");
        assert!(cfg.enabled);
        assert_eq!(cfg.monthly_budget, Some(500.0));
        assert_eq!(cfg.per_request_limit, Some(2.5));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_rate_limiter_cleanup_stale_buckets() {
        use super::rate_limit::KeyRateLimiter;
        let rl = KeyRateLimiter::new(1, 5);
        assert!(rl.try_acquire("ephemeral").is_ok());
        assert_eq!(rl.bucket_count(), 1);
        // Immediate cleanup should keep it (bucket is fresh).
        rl.cleanup_stale();
        assert_eq!(rl.bucket_count(), 1);
    }

    #[test]
    fn test_merge_invalid_bind_port_rejected() {
        let toml_text = r#"
            [server]
            bind = "0.0.0.0:not_a_port"

            [[backends]]
            addr = "10.0.0.1:8090"
        "#;
        let cfg: ProxyConfig = toml::from_str(toml_text).unwrap();
        let cli = CliArgs::default();
        let err = merge_cli_and_config(&cli, Some(cfg)).unwrap_err();
        assert!(err.contains("bind"), "got: {err}");
    }

    // ------------------------------------------------------------------
    // V78 / WS-7: WS-2 helper unit tests (gateway extraction logic)
    // ------------------------------------------------------------------

    #[cfg(feature = "security")]
    #[test]
    fn test_any_middleware_enabled_false_when_all_off() {
        let m = MiddlewareSection::default();
        let a = AuditSection::default();
        assert!(!any_middleware_enabled(&m, &a));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_any_middleware_enabled_true_when_any_on() {
        let mut m = MiddlewareSection::default();
        let a = AuditSection::default();
        m.enable_cache = true;
        assert!(any_middleware_enabled(&m, &a));
        m.enable_cache = false;
        let mut a2 = a.clone();
        a2.enabled = true;
        assert!(any_middleware_enabled(&m, &a2));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_scan_text_messages_filters_roles() {
        let body = serde_json::json!({
            "model": "gpt-x",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user",   "content": "hello"},
                {"role": "assistant", "content": "prior response, must be ignored"},
                {"role": "user",   "content": "second question"},
            ]
        });
        let text = extract_scan_text(&body);
        assert!(text.contains("you are helpful"));
        assert!(text.contains("hello"));
        assert!(text.contains("second question"));
        assert!(!text.contains("prior response"));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_scan_text_legacy_prompt_field() {
        let body = serde_json::json!({ "prompt": "legacy completion prompt" });
        let text = extract_scan_text(&body);
        assert_eq!(text, "legacy completion prompt");
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_response_text_and_usage_chat() {
        let body = br#"{
            "choices": [{"message": {"content": "hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;
        let (text, tin, tout) = extract_response_text_and_usage(body);
        assert_eq!(text, "hi there");
        assert_eq!(tin, 10);
        assert_eq!(tout, 5);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_response_text_and_usage_legacy_completion() {
        let body = br#"{
            "choices": [{"text": "classic completion"}]
        }"#;
        let (text, tin, tout) = extract_response_text_and_usage(body);
        assert_eq!(text, "classic completion");
        assert_eq!(tin, 0);
        assert_eq!(tout, 0);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_response_text_and_usage_invalid_json() {
        let (text, tin, tout) = extract_response_text_and_usage(b"not json");
        assert_eq!(text, "");
        assert_eq!(tin, 0);
        assert_eq!(tout, 0);
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_pick_rate_limit_key_matching_bearer_hashes_key() {
        let req = Request::builder()
            .uri("/v1/chat/completions")
            .header(header::AUTHORIZATION, "Bearer the-secret")
            .body(Body::empty())
            .unwrap();
        let key = pick_rate_limit_key(&req, Some("the-secret"));
        assert!(key.starts_with("key:"));
        assert!(!key.contains("the-secret"));
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_pick_rate_limit_key_falls_back_to_session() {
        let req = Request::builder()
            .uri("/v1/chat/completions")
            .header("x-session-id", "sess-42")
            .body(Body::empty())
            .unwrap();
        let key = pick_rate_limit_key(&req, Some("expected"));
        assert_eq!(key, "sess:sess-42");
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_pick_rate_limit_key_falls_back_to_ip() {
        let req = Request::builder()
            .uri("/v1/chat/completions")
            .header("x-forwarded-for", "203.0.113.7, 10.0.0.1")
            .body(Body::empty())
            .unwrap();
        let key = pick_rate_limit_key(&req, Some("expected"));
        assert_eq!(key, "ip:203.0.113.7");
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_extract_client_ip_unknown_when_missing() {
        let req = Request::builder()
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req), "unknown");
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_build_gateway_context_all_middlewares_off() {
        let proxy = ProxyState {
            backends: Arc::new(vec![Backend::new("10.0.0.1:8090".to_string())]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        let m = MiddlewareSection::default();
        let a = AuditSection::default();
        let ctx = build_gateway_context(proxy, &m, &a).unwrap();
        assert!(ctx.cache.is_none());
        assert!(ctx.rate_limiter.is_none());
        assert!(ctx.budget.is_none());
        assert!(ctx.audit.is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_build_gateway_context_cache_and_rate_limiter_built() {
        let proxy = ProxyState {
            backends: Arc::new(vec![Backend::new("10.0.0.1:8090".to_string())]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        let mut m = MiddlewareSection::default();
        m.enable_cache = true;
        m.cache_max_entries = Some(64);
        m.cache_ttl_secs = Some(30);
        m.enable_rate_limit = true;
        m.rate_limit_rpm = Some(120);
        let a = AuditSection::default();
        let ctx = build_gateway_context(proxy, &m, &a).unwrap();
        assert!(ctx.cache.is_some());
        assert!(ctx.rate_limiter.is_some());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_build_gateway_router_has_health_route() {
        let proxy = ProxyState {
            backends: Arc::new(vec![Backend::new("10.0.0.1:8090".to_string())]),
            next_index: Arc::new(AtomicUsize::new(0)),
            session_affinity: Arc::new(DashMap::new()),
            api_key: None,
        };
        let ctx = build_gateway_context(
            proxy,
            &MiddlewareSection::default(),
            &AuditSection::default(),
        )
        .unwrap();
        let _router = build_gateway_router(ctx);
        // Should not panic; full HTTP-level testing is left for a follow-up
        // integration-test file with a mock upstream backend.
    }

    // ------------------------------------------------------------------
    // V78.1: end-to-end tests with an in-process mock upstream backend
    // ------------------------------------------------------------------
    //
    // We spin up a tiny axum app on 127.0.0.1:0 as a fake upstream
    // `ai_assistant_server`, build a real `GatewayContext` pointing at it,
    // and drive `build_gateway_router` via `tower::ServiceExt::oneshot` so
    // the tests exercise the full request path without needing to bind the
    // gateway itself. The mock backend IS a real HTTP server (because the
    // gateway forwards via `reqwest`), so each test uses a
    // `tokio::runtime::Runtime` and a oneshot shutdown channel.
    #[cfg(feature = "security")]
    mod gateway_e2e {
        use super::*;
        use axum::body::to_bytes;
        use axum::response::Response as AxumResponse;
        use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;
        use tower::ServiceExt;

        fn rt() -> tokio::runtime::Runtime {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
        }

        /// Spawn a minimal axum app as a fake upstream backend. The closure
        /// builds a response for every request. Returns the `host:port`
        /// address string (fed directly to `Backend::new`) plus a shutdown
        /// sender that must be kept alive for the duration of the test.
        async fn spawn_mock_backend<F>(responder: F) -> (String, oneshot::Sender<()>)
        where
            F: Fn() -> AxumResponse + Clone + Send + Sync + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let (tx, rx) = oneshot::channel::<()>();
            let app = axum::Router::new().fallback(axum::routing::any(
                move |_req: axum::extract::Request| {
                    let r = responder.clone();
                    async move { r() }
                },
            ));
            tokio::spawn(async move {
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = rx.await;
                    })
                    .await;
            });
            // Give the listener a moment to be fully ready.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            (format!("127.0.0.1:{port}"), tx)
        }

        /// Counting variant — increments a shared counter on every request
        /// so tests can assert that the cache prevented a second backend hit.
        async fn spawn_mock_backend_counting<F>(
            responder: F,
        ) -> (String, Arc<StdAtomicUsize>, oneshot::Sender<()>)
        where
            F: Fn() -> AxumResponse + Clone + Send + Sync + 'static,
        {
            let counter = Arc::new(StdAtomicUsize::new(0));
            let counter_cl = counter.clone();
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let (tx, rx) = oneshot::channel::<()>();
            let app = axum::Router::new().fallback(axum::routing::any(
                move |_req: axum::extract::Request| {
                    let r = responder.clone();
                    let c = counter_cl.clone();
                    async move {
                        c.fetch_add(1, StdOrdering::SeqCst);
                        r()
                    }
                },
            ));
            tokio::spawn(async move {
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = rx.await;
                    })
                    .await;
            });
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            (format!("127.0.0.1:{port}"), counter, tx)
        }

        fn make_state(backend_addr: &str, api_key: Option<&str>) -> ProxyState {
            ProxyState {
                backends: Arc::new(vec![Backend::new(backend_addr.to_string())]),
                next_index: Arc::new(AtomicUsize::new(0)),
                session_affinity: Arc::new(DashMap::new()),
                api_key: api_key.map(|s| s.to_string()),
            }
        }

        fn chat_ok_response() -> AxumResponse {
            AxumResponse::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"id":"x","choices":[{"message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":4,"completion_tokens":1}}"#,
                ))
                .unwrap()
        }

        fn embeddings_ok_response() -> AxumResponse {
            AxumResponse::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#))
                .unwrap()
        }

        fn chat_body(prompt: &str) -> String {
            format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":"{prompt}"}}]}}"#
            )
        }

        fn chat_req(body: String) -> axum::http::Request<Body> {
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        }

        async fn read_body_bytes(resp: AxumResponse) -> Vec<u8> {
            let (_p, body) = resp.into_parts();
            let b = to_bytes(body, 1024 * 1024).await.unwrap();
            b.to_vec()
        }

        #[test]
        fn test_gateway_e2e_forward_ok_and_headers() {
            rt().block_on(async {
                let (addr, _shutdown) = spawn_mock_backend(chat_ok_response).await;
                let mut mw = MiddlewareSection::default();
                mw.enable_cache = true;
                mw.cache_max_entries = Some(16);
                mw.cache_ttl_secs = Some(60);
                let state = make_state(&addr, None);
                let ctx = build_gateway_context(state, &mw, &AuditSection::default()).unwrap();
                let router = build_gateway_router(ctx);

                let resp = router.oneshot(chat_req(chat_body("hello"))).await.unwrap();
                assert_eq!(resp.status(), 200);
                assert!(
                    resp.headers().contains_key("x-request-id"),
                    "X-Request-Id must be set"
                );
                assert_eq!(
                    resp.headers().get("x-cache").and_then(|v| v.to_str().ok()),
                    Some("MISS")
                );
                let body = read_body_bytes(resp).await;
                let text = String::from_utf8_lossy(&body);
                assert!(text.contains("pong"), "body: {text}");
            });
        }

        #[test]
        fn test_gateway_e2e_cache_hit_on_second_request() {
            rt().block_on(async {
                let (addr, counter, _shutdown) =
                    spawn_mock_backend_counting(chat_ok_response).await;
                let mut mw = MiddlewareSection::default();
                mw.enable_cache = true;
                mw.cache_max_entries = Some(16);
                mw.cache_ttl_secs = Some(60);
                let state = make_state(&addr, None);
                let ctx = build_gateway_context(state, &mw, &AuditSection::default()).unwrap();
                let router = build_gateway_router(ctx);

                let body = chat_body("same-prompt");
                let r1 = router
                    .clone()
                    .oneshot(chat_req(body.clone()))
                    .await
                    .unwrap();
                assert_eq!(r1.status(), 200);
                assert_eq!(
                    r1.headers().get("x-cache").and_then(|v| v.to_str().ok()),
                    Some("MISS")
                );

                let r2 = router.oneshot(chat_req(body)).await.unwrap();
                assert_eq!(r2.status(), 200);
                assert_eq!(
                    r2.headers().get("x-cache").and_then(|v| v.to_str().ok()),
                    Some("HIT")
                );
                // Backend saw exactly one request; the cache served the second.
                assert_eq!(counter.load(StdOrdering::SeqCst), 1);
            });
        }

        #[test]
        fn test_gateway_e2e_rate_limit_returns_429() {
            rt().block_on(async {
                let (addr, _shutdown) = spawn_mock_backend(chat_ok_response).await;
                let mut mw = MiddlewareSection::default();
                mw.enable_rate_limit = true;
                mw.rate_limit_rpm = Some(2); // 2 req per 60s window
                let state = make_state(&addr, None);
                let ctx = build_gateway_context(state, &mw, &AuditSection::default()).unwrap();
                let router = build_gateway_router(ctx);

                // Same session id → same bucket. First 2 OK, 3rd rejected.
                let mk = || {
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .header("x-session-id", "sess-abc")
                        .body(Body::from(chat_body("ping")))
                        .unwrap()
                };
                let r1 = router.clone().oneshot(mk()).await.unwrap();
                assert_eq!(r1.status(), 200);
                let r2 = router.clone().oneshot(mk()).await.unwrap();
                assert_eq!(r2.status(), 200);
                let r3 = router.oneshot(mk()).await.unwrap();
                assert_eq!(r3.status(), 429);
                assert_eq!(
                    r3.headers().get("x-reason").and_then(|v| v.to_str().ok()),
                    Some("rate_limit")
                );
            });
        }

        #[test]
        fn test_gateway_e2e_embeddings_passthrough_bypasses_pipeline() {
            rt().block_on(async {
                let (addr, counter, _shutdown) =
                    spawn_mock_backend_counting(embeddings_ok_response).await;
                // Even with PII+toxicity+attack guards enabled, /v1/embeddings
                // should go through the fallback passthrough handler.
                let mut mw = MiddlewareSection::default();
                mw.enable_pii_input = true;
                mw.enable_toxicity_input = true;
                mw.enable_attack_filter = true;
                let state = make_state(&addr, None);
                let ctx = build_gateway_context(state, &mw, &AuditSection::default()).unwrap();
                let router = build_gateway_router(ctx);

                let req = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"emb","input":"anything"}"#))
                    .unwrap();
                let resp = router.oneshot(req).await.unwrap();
                assert_eq!(resp.status(), 200);
                assert_eq!(counter.load(StdOrdering::SeqCst), 1);
                let body = read_body_bytes(resp).await;
                assert!(String::from_utf8_lossy(&body).contains("embedding"));
            });
        }

        #[test]
        fn test_gateway_e2e_unauthorized_when_api_key_mismatch() {
            rt().block_on(async {
                let (addr, _shutdown) = spawn_mock_backend(chat_ok_response).await;
                let mw = MiddlewareSection::default();
                let state = make_state(&addr, Some("expected-key"));
                let ctx = build_gateway_context(state, &mw, &AuditSection::default()).unwrap();
                let router = build_gateway_router(ctx);

                // Missing Authorization header → 401.
                let resp = router
                    .clone()
                    .oneshot(chat_req(chat_body("hi")))
                    .await
                    .unwrap();
                assert_eq!(resp.status(), 401);

                // Wrong Bearer token → 401.
                let bad = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer wrong-key")
                    .body(Body::from(chat_body("hi")))
                    .unwrap();
                let resp2 = router.oneshot(bad).await.unwrap();
                assert_eq!(resp2.status(), 401);
            });
        }

        #[test]
        fn test_gateway_e2e_audit_entry_written() {
            rt().block_on(async {
                let (addr, _shutdown) = spawn_mock_backend(chat_ok_response).await;
                let tmpdir = tempfile::tempdir().unwrap();
                let audit_path = tmpdir.path().join("audit.jsonl");

                let mw = MiddlewareSection::default();
                let mut audit_cfg = AuditSection::default();
                audit_cfg.enabled = true;
                audit_cfg.path = Some(audit_path.to_string_lossy().into_owned());
                audit_cfg.max_files = Some(3);
                audit_cfg.max_bytes_per_file = Some(10 * 1024);

                let state = make_state(&addr, None);
                let ctx = build_gateway_context(state, &mw, &audit_cfg).unwrap();
                let router = build_gateway_router(ctx.clone());

                let resp = router.oneshot(chat_req(chat_body("hello"))).await.unwrap();
                assert_eq!(resp.status(), 200);
                // Drop the context to force the AuditWriter to flush on the
                // file handle going out of scope (writer is buffered).
                drop(ctx);

                // Allow a few ms for the tokio task to actually write.
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                let contents = std::fs::read_to_string(&audit_path).unwrap_or_default();
                assert!(
                    !contents.is_empty(),
                    "audit log should contain at least one entry"
                );
                assert!(
                    contents.contains("\"request_id\""),
                    "entry must include request_id; got: {contents}"
                );
            });
        }
    }
}
