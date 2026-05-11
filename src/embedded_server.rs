// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Embedded `llama-server` launcher.
//!
//! Spawns and supervises a local `llama-server` (or compatible binary —
//! `koboldcpp`, `vllm` with OpenAI-compat shim) so callers no longer have
//! to manage the process out-of-band. Pairs with [`crate::mmproj`] for
//! vision setups: a validated `MultimodalProjector` is passed straight
//! through to `--mmproj` on the spawned argv.
//!
//! ## Lifecycle
//!
//! 1. [`LlamaServerConfig::builder`] composes a config (binary, model,
//!    optional mmproj, host, port, ctx-size, GPU layers, extra args).
//! 2. [`EmbeddedLlamaServer::start`] validates everything, picks a free
//!    port if requested (port `0`), spawns the child, and returns a
//!    handle.
//! 3. [`EmbeddedLlamaServer::wait_until_ready`] polls `/health` until
//!    the server answers `200` or the deadline passes.
//! 4. The handle exposes [`EmbeddedLlamaServer::base_url`] for downstream
//!    HTTP traffic.
//! 5. [`EmbeddedLlamaServer::stop`] sends a polite kill and waits; the
//!    same path runs on `Drop`, so a panicking caller does not leak the
//!    child.
//!
//! ## What this module does *not* do
//!
//! * It does not download binaries or model files.
//! * It does not parse `llama-server` stdout for tokens — every model
//!   request still goes over HTTP through [`crate::providers`].
//! * It does not retry on crash; restart policy is the caller's
//!   responsibility.
//!
//! ## Security
//!
//! `extra_args` is a flat `Vec<String>` passed directly to
//! `Command::args` — no shell is invoked, so meta-characters like `;`,
//! `&&`, `|`, `` ` `` cannot escape into a shell. We additionally reject
//! arguments that contain NUL bytes (which `Command` would refuse with
//! a less helpful error). Binary, model, and mmproj paths reject `..`
//! components before canonicalize as defense-in-depth against
//! symlink-race substitution.

use std::ffi::OsString;
use std::fmt;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::mmproj::{MmprojValidationError, MultimodalProjector};

/// Host / port range for `llama-server`.
const LOWEST_USER_PORT: u16 = 1024;

/// Default health-check timeout when callers pass `None`.
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(60);

/// Default poll interval while waiting for `/health`.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Configuration for a single embedded `llama-server` instance.
///
/// Build with [`LlamaServerConfig::builder`].
#[derive(Debug, Clone)]
pub struct LlamaServerConfig {
    binary_path: PathBuf,
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    host: String,
    port: u16,
    ctx_size: Option<u32>,
    n_gpu_layers: Option<u32>,
    extra_args: Vec<String>,
    ready_timeout: Duration,
    capture_output: bool,
}

impl LlamaServerConfig {
    /// Start a builder bound to the required `binary_path` + `model_path`.
    pub fn builder(
        binary_path: impl Into<PathBuf>,
        model_path: impl Into<PathBuf>,
    ) -> LlamaServerConfigBuilder {
        LlamaServerConfigBuilder {
            binary_path: binary_path.into(),
            model_path: model_path.into(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 0,
            ctx_size: None,
            n_gpu_layers: None,
            extra_args: Vec::new(),
            ready_timeout: DEFAULT_READY_TIMEOUT,
            capture_output: true,
        }
    }

    /// Path to the `llama-server` (or compatible) executable.
    pub fn binary_path(&self) -> &Path {
        &self.binary_path
    }

    /// Path to the base GGUF model.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Optional projector path; `None` means a text-only server.
    pub fn mmproj_path(&self) -> Option<&Path> {
        self.mmproj_path.as_deref()
    }

    /// Bind host. Default `127.0.0.1`.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Bind port. `0` means "auto-pick a free port at start time".
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Fluent builder for [`LlamaServerConfig`].
#[derive(Debug, Clone)]
pub struct LlamaServerConfigBuilder {
    binary_path: PathBuf,
    model_path: PathBuf,
    mmproj_path: Option<PathBuf>,
    host: String,
    port: u16,
    ctx_size: Option<u32>,
    n_gpu_layers: Option<u32>,
    extra_args: Vec<String>,
    ready_timeout: Duration,
    capture_output: bool,
}

impl LlamaServerConfigBuilder {
    /// Set the multimodal projector path.
    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj_path = Some(path.into());
        self
    }

    /// Set the bind host (default `127.0.0.1`).
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the bind port. Pass `0` to auto-pick a free port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the context window size (`--ctx-size`).
    pub fn ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = Some(ctx_size);
        self
    }

    /// Set the number of GPU layers (`--n-gpu-layers`).
    pub fn n_gpu_layers(mut self, layers: u32) -> Self {
        self.n_gpu_layers = Some(layers);
        self
    }

    /// Append extra raw arguments. Validated for NUL bytes; not parsed.
    pub fn extra_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Override the ready-probe timeout (default 60 s).
    pub fn ready_timeout(mut self, timeout: Duration) -> Self {
        self.ready_timeout = timeout;
        self
    }

    /// Toggle stdout/stderr capture. When `false`, the child inherits
    /// the parent's streams, which is occasionally useful for debugging.
    pub fn capture_output(mut self, capture: bool) -> Self {
        self.capture_output = capture;
        self
    }

    /// Finalize the config without validating paths or spawning.
    pub fn build(self) -> LlamaServerConfig {
        LlamaServerConfig {
            binary_path: self.binary_path,
            model_path: self.model_path,
            mmproj_path: self.mmproj_path,
            host: self.host,
            port: self.port,
            ctx_size: self.ctx_size,
            n_gpu_layers: self.n_gpu_layers,
            extra_args: self.extra_args,
            ready_timeout: self.ready_timeout,
            capture_output: self.capture_output,
        }
    }
}

/// Construct the argv for a given config. Pure function: no I/O, no
/// validation. Exposed so callers (and tests) can inspect the command
/// without spawning.
pub fn build_command_args(config: &LlamaServerConfig, resolved_port: u16) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        OsString::from("--model"),
        config.model_path.clone().into_os_string(),
        OsString::from("--host"),
        OsString::from(&config.host),
        OsString::from("--port"),
        OsString::from(resolved_port.to_string()),
    ];
    if let Some(ref mmproj) = config.mmproj_path {
        args.push(OsString::from("--mmproj"));
        args.push(mmproj.clone().into_os_string());
    }
    if let Some(ctx) = config.ctx_size {
        args.push(OsString::from("--ctx-size"));
        args.push(OsString::from(ctx.to_string()));
    }
    if let Some(layers) = config.n_gpu_layers {
        args.push(OsString::from("--n-gpu-layers"));
        args.push(OsString::from(layers.to_string()));
    }
    for extra in &config.extra_args {
        args.push(OsString::from(extra));
    }
    args
}

/// Live handle to a spawned `llama-server` child.
pub struct EmbeddedLlamaServer {
    child: Option<Child>,
    base_url: String,
    resolved_port: u16,
    binary_filename: String,
    ready_timeout: Duration,
}

impl fmt::Debug for EmbeddedLlamaServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddedLlamaServer")
            .field("binary", &self.binary_filename)
            .field("base_url", &self.base_url)
            .field("port", &self.resolved_port)
            .field("running", &self.child.is_some())
            .finish()
    }
}

impl EmbeddedLlamaServer {
    /// Validate `config` and spawn the child process. The handle returned
    /// owns the child; dropping it (or calling [`Self::stop`]) terminates
    /// the server.
    pub fn start(config: LlamaServerConfig) -> Result<Self, LaunchError> {
        Self::start_with_validation(config, /* skip_path_validation = */ false)
    }

    /// Test-only entry point that bypasses path-existence checks. Used by
    /// integration tests that supply a stand-in mock binary and an empty
    /// dummy model file. Argument validation, port allocation, NUL-byte
    /// rejection, and traversal rejection still run.
    #[doc(hidden)]
    pub fn start_for_testing(config: LlamaServerConfig) -> Result<Self, LaunchError> {
        Self::start_with_validation(config, /* skip_path_validation = */ true)
    }

    fn start_with_validation(
        config: LlamaServerConfig,
        skip_path_validation: bool,
    ) -> Result<Self, LaunchError> {
        // Path traversal defense always runs — even in test mode.
        reject_traversal(&config.binary_path, "binary_path")?;
        reject_traversal(&config.model_path, "model_path")?;
        if let Some(ref m) = config.mmproj_path {
            reject_traversal(m, "mmproj_path")?;
        }
        for arg in &config.extra_args {
            if arg.as_bytes().contains(&0) {
                return Err(LaunchError::ArgContainsNul { arg: arg.clone() });
            }
        }
        if config.host.is_empty() || config.host.as_bytes().contains(&0) {
            return Err(LaunchError::InvalidHost {
                host: config.host.clone(),
            });
        }

        if !skip_path_validation {
            let bmeta =
                std::fs::metadata(&config.binary_path).map_err(|_| LaunchError::BinaryNotFound)?;
            if !bmeta.is_file() {
                return Err(LaunchError::BinaryNotFound);
            }
            let mmeta =
                std::fs::metadata(&config.model_path).map_err(|_| LaunchError::ModelNotFound)?;
            if !mmeta.is_file() {
                return Err(LaunchError::ModelNotFound);
            }
            if let Some(ref mmproj) = config.mmproj_path {
                MultimodalProjector::from_path(mmproj).map_err(LaunchError::MmprojValidation)?;
            }
        }

        let resolved_port = if config.port == 0 {
            pick_free_port(&config.host)?
        } else if config.port < LOWEST_USER_PORT {
            return Err(LaunchError::PortTooLow { port: config.port });
        } else {
            config.port
        };

        let argv = build_command_args(&config, resolved_port);
        let mut command = Command::new(&config.binary_path);
        command.args(&argv);
        if config.capture_output {
            command.stdout(Stdio::piped());
            command.stderr(Stdio::piped());
        }
        let child = command
            .spawn()
            .map_err(|e| LaunchError::SpawnFailed(e.to_string()))?;

        let binary_filename = config
            .binary_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unnamed>".to_string());

        Ok(Self {
            child: Some(child),
            base_url: format!("http://{}:{}", config.host, resolved_port),
            resolved_port,
            binary_filename,
            ready_timeout: config.ready_timeout,
        })
    }

    /// HTTP base URL of the running server (e.g. `http://127.0.0.1:42171`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Resolved bind port (after auto-pick if the config used `0`).
    pub fn port(&self) -> u16 {
        self.resolved_port
    }

    /// Filename of the spawned binary. Safe to log (no directory layout).
    pub fn binary_filename(&self) -> &str {
        &self.binary_filename
    }

    /// Process id of the live child, or `None` after [`Self::stop`].
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(|c| c.id())
    }

    /// Cheap liveness check: returns `true` if the child has not exited.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(c) => matches!(c.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// Poll `/health` until it returns `200`, or the timeout elapses. The
    /// poll is bounded by the `ready_timeout` configured on the
    /// [`LlamaServerConfig`] unless `timeout` overrides it.
    pub fn wait_until_ready(&mut self, timeout: Option<Duration>) -> Result<(), LaunchError> {
        let deadline = Instant::now() + timeout.unwrap_or(self.ready_timeout);
        let url = format!("{}/health", self.base_url);
        loop {
            if !self.is_running() {
                return Err(LaunchError::ChildExitedEarly);
            }
            if probe_health(&url) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(LaunchError::Timeout);
            }
            std::thread::sleep(DEFAULT_POLL_INTERVAL);
        }
    }

    /// Terminate the child and wait for it to exit. Idempotent: calling
    /// twice (or after [`Drop`]) is a no-op.
    pub fn stop(&mut self) -> Result<(), LaunchError> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(())
    }
}

impl Drop for EmbeddedLlamaServer {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Reject `..` components in `path` with a typed error.
fn reject_traversal(path: &Path, field: &'static str) -> Result<(), LaunchError> {
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(LaunchError::PathTraversal { field });
        }
    }
    Ok(())
}

fn pick_free_port(host: &str) -> Result<u16, LaunchError> {
    let bind: SocketAddr = format!("{}:0", host)
        .parse()
        .map_err(|_| LaunchError::InvalidHost {
            host: host.to_string(),
        })?;
    let listener = TcpListener::bind(bind).map_err(|e| LaunchError::SpawnFailed(e.to_string()))?;
    let port = listener
        .local_addr()
        .map_err(|e| LaunchError::SpawnFailed(e.to_string()))?
        .port();
    drop(listener);
    Ok(port)
}

/// Best-effort `/health` probe using a raw `TcpStream` so we don't pull
/// `reqwest` into the type signature here. A 200 OK or any 2xx counts as
/// ready; everything else is "not ready yet".
fn probe_health(url: &str) -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    // Parse url: http://host:port/path
    let rest = match url.strip_prefix("http://") {
        Some(r) => r,
        None => return false,
    };
    let (authority, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };

    let mut stream = match TcpStream::connect_timeout(
        &match authority.parse() {
            Ok(addr) => addr,
            Err(_) => return false,
        },
        Duration::from_millis(500),
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

    let req = format!(
        "GET {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, authority
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.0 2") || head.starts_with("HTTP/1.1 2")
}

/// Typed errors produced by the embedded launcher.
#[derive(Debug)]
pub enum LaunchError {
    /// `binary_path` did not exist or was not a regular file.
    BinaryNotFound,
    /// `model_path` did not exist or was not a regular file.
    ModelNotFound,
    /// The configured projector failed [`MultimodalProjector`] validation.
    MmprojValidation(MmprojValidationError),
    /// One of the path inputs contained a `..` component.
    PathTraversal { field: &'static str },
    /// `extra_args` contained a NUL byte (rejected before `Command`).
    ArgContainsNul { arg: String },
    /// Configured `host` was empty or contained a NUL byte.
    InvalidHost { host: String },
    /// Explicit non-auto port below `LOWEST_USER_PORT` (1024).
    PortTooLow { port: u16 },
    /// `Command::spawn` failed (binary not executable, permission, …).
    SpawnFailed(String),
    /// The child exited before `/health` returned 200.
    ChildExitedEarly,
    /// `/health` did not return 200 within the configured timeout.
    Timeout,
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BinaryNotFound => write!(f, "embedded server binary not found"),
            Self::ModelNotFound => write!(f, "embedded server model file not found"),
            Self::MmprojValidation(e) => write!(f, "embedded server mmproj invalid: {}", e),
            Self::PathTraversal { field } => {
                write!(f, "embedded server {} contains a `..` component", field)
            }
            Self::ArgContainsNul { arg } => write!(
                f,
                "embedded server extra_arg contains NUL byte ({} bytes shown)",
                arg.len()
            ),
            Self::InvalidHost { host } => {
                write!(f, "embedded server host is invalid (len {})", host.len())
            }
            Self::PortTooLow { port } => write!(
                f,
                "embedded server port {} is below {} (use 0 for auto-pick)",
                port, LOWEST_USER_PORT
            ),
            Self::SpawnFailed(msg) => write!(f, "embedded server spawn failed: {}", msg),
            Self::ChildExitedEarly => {
                write!(f, "embedded server child exited before becoming ready")
            }
            Self::Timeout => write!(f, "embedded server did not become ready in time"),
        }
    }
}

impl std::error::Error for LaunchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MmprojValidation(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_command_args_minimal() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .host("127.0.0.1")
            .port(8080)
            .build();
        let argv = build_command_args(&cfg, 8080);
        let strs: Vec<String> = argv
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert!(strs.contains(&"--model".to_string()));
        assert!(strs.contains(&"/m/model.gguf".to_string()));
        assert!(strs.contains(&"--host".to_string()));
        assert!(strs.contains(&"127.0.0.1".to_string()));
        assert!(strs.contains(&"--port".to_string()));
        assert!(strs.contains(&"8080".to_string()));
        assert!(!strs.contains(&"--mmproj".to_string()));
    }

    #[test]
    fn build_command_args_with_mmproj_and_extras() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .mmproj("/m/mmproj.gguf")
            .ctx_size(4096)
            .n_gpu_layers(35)
            .extra_args(vec!["--log-disable", "--mlock"])
            .build();
        let argv = build_command_args(&cfg, 17171);
        let strs: Vec<String> = argv
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert!(strs.contains(&"--mmproj".to_string()));
        assert!(strs.contains(&"/m/mmproj.gguf".to_string()));
        assert!(strs.contains(&"--ctx-size".to_string()));
        assert!(strs.contains(&"4096".to_string()));
        assert!(strs.contains(&"--n-gpu-layers".to_string()));
        assert!(strs.contains(&"35".to_string()));
        assert!(strs.contains(&"--log-disable".to_string()));
        assert!(strs.contains(&"--mlock".to_string()));
        assert!(strs.contains(&"17171".to_string()));
    }

    #[test]
    fn rejects_binary_path_traversal() {
        let cfg = LlamaServerConfig::builder("../sneaky/llama-server", "/m/model.gguf").build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must reject traversal");
        assert!(matches!(
            err,
            LaunchError::PathTraversal {
                field: "binary_path"
            }
        ));
    }

    #[test]
    fn rejects_model_path_traversal() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "../sneaky/model.gguf").build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must reject traversal");
        assert!(matches!(
            err,
            LaunchError::PathTraversal {
                field: "model_path"
            }
        ));
    }

    #[test]
    fn rejects_mmproj_path_traversal() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .mmproj("../sneaky/mmproj.gguf")
            .build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must reject traversal");
        assert!(matches!(
            err,
            LaunchError::PathTraversal {
                field: "mmproj_path"
            }
        ));
    }

    #[test]
    fn rejects_extra_arg_with_nul() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .extra_args(vec!["--ok", "bad\0arg"])
            .build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must reject NUL");
        assert!(matches!(err, LaunchError::ArgContainsNul { .. }));
    }

    #[test]
    fn rejects_empty_host() {
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .host("")
            .build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must reject empty host");
        assert!(matches!(err, LaunchError::InvalidHost { .. }));
    }

    #[test]
    fn rejects_explicit_low_port() {
        // We use start_for_testing to get past path checks but still hit
        // the port check.
        let cfg = LlamaServerConfig::builder("/bin/llama-server", "/m/model.gguf")
            .port(80)
            .build();
        let err = EmbeddedLlamaServer::start_for_testing(cfg).expect_err("must reject low port");
        assert!(matches!(err, LaunchError::PortTooLow { port: 80 }));
    }

    #[test]
    fn missing_binary_yields_binary_not_found() {
        let cfg =
            LlamaServerConfig::builder("/definitely/not/a/real/binary_xyzzy", "/m/model.gguf")
                .build();
        let err = EmbeddedLlamaServer::start(cfg).expect_err("must fail");
        assert!(matches!(err, LaunchError::BinaryNotFound));
    }

    #[test]
    fn launch_error_display_contains_actionable_text() {
        assert!(format!("{}", LaunchError::BinaryNotFound).contains("binary"));
        assert!(format!("{}", LaunchError::ModelNotFound).contains("model"));
        assert!(format!("{}", LaunchError::Timeout).contains("ready"));
        assert!(format!("{}", LaunchError::ChildExitedEarly).contains("exited"));
        assert!(format!("{}", LaunchError::PortTooLow { port: 80 }).contains("80"));
    }
}
