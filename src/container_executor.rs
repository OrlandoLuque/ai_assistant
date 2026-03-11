//! Docker container execution engine.
//!
//! Provides a managed Docker container lifecycle for isolated code execution.
//! Uses the bollard crate for async Docker API operations, wrapped in a tokio
//! runtime for a synchronous public API.
//!
//! Architecture: A tokio runtime is stored in the struct and `runtime.block_on()`
//! bridges async bollard calls to sync. This matches the pattern used by
//! `LanceVectorDb` and `NetworkNode` in this crate.
//!
//! This module is gated behind the `containers` feature flag.
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::container_executor::{ContainerExecutor, ContainerConfig, CreateOptions};
//!
//! let config = ContainerConfig::default();
//! let mut executor = ContainerExecutor::new(config).unwrap();
//!
//! let id = executor.create("python:3.11-slim", "my_sandbox", CreateOptions::default()).unwrap();
//! executor.start(&id).unwrap();
//!
//! let result = executor.exec(&id, &["python", "-c", "print('hello')"], std::time::Duration::from_secs(30)).unwrap();
//! assert_eq!(result.exit_code, 0);
//! assert!(result.stdout.contains("hello"));
//!
//! executor.stop(&id, 10).unwrap();
//! executor.remove(&id, false).unwrap();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bollard::container::{
    Config as BollardConfig, CreateContainerOptions, DownloadFromContainerOptions,
    LogOutput, LogsOptions, RemoveContainerOptions, StartContainerOptions,
    StopContainerOptions, UploadToContainerOptions,
};
use bollard::exec::{CreateExecOptions, StartExecResults};
use bollard::models::{HostConfig, PortBinding};
use bollard::Docker;
use futures::StreamExt;
use tokio::runtime::Runtime;

// =============================================================================
// Types and enums
// =============================================================================

/// Network mode for container isolation.
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkMode {
    /// Full network isolation (default for sandboxed execution).
    None,
    /// Standard Docker bridge networking.
    Bridge,
    /// Host networking (shares host network stack — use with caution).
    Host,
    /// Named custom Docker network.
    Custom(String),
}

impl NetworkMode {
    /// Convert to the Docker API string representation.
    pub fn to_docker_string(&self) -> String {
        match self {
            NetworkMode::None => "none".to_string(),
            NetworkMode::Bridge => "bridge".to_string(),
            NetworkMode::Host => "host".to_string(),
            NetworkMode::Custom(name) => name.clone(),
        }
    }
}

impl std::fmt::Display for NetworkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkMode::None => write!(f, "none"),
            NetworkMode::Bridge => write!(f, "bridge"),
            NetworkMode::Host => write!(f, "host"),
            NetworkMode::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Status of a managed container.
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerStatus {
    /// Container has been created but not started.
    Created,
    /// Container is currently running.
    Running,
    /// Container is paused.
    Paused,
    /// Container has been stopped.
    Stopped,
    /// Container has been removed.
    Removed,
}

impl std::fmt::Display for ContainerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerStatus::Created => write!(f, "created"),
            ContainerStatus::Running => write!(f, "running"),
            ContainerStatus::Paused => write!(f, "paused"),
            ContainerStatus::Stopped => write!(f, "stopped"),
            ContainerStatus::Removed => write!(f, "removed"),
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Policy for automatic container cleanup.
#[derive(Debug, Clone)]
pub struct ContainerCleanupPolicy {
    /// Maximum containers allowed per session (default: 5).
    pub max_per_session: usize,
    /// Maximum total managed containers (default: 20).
    pub max_total: usize,
    /// Automatically remove containers after this many seconds (default: 3600 = 1 hour).
    pub auto_remove_after_secs: Option<u64>,
    /// Whether to remove all session containers when the session ends (default: true).
    pub cleanup_on_session_end: bool,
}

impl Default for ContainerCleanupPolicy {
    fn default() -> Self {
        Self {
            max_per_session: 5,
            max_total: 20,
            auto_remove_after_secs: Some(3600),
            cleanup_on_session_end: true,
        }
    }
}

/// Configuration for the container execution engine.
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Docker daemon host URI (None = local defaults).
    pub docker_host: Option<String>,
    /// Default timeout for container operations (default: 60s).
    pub default_timeout: Duration,
    /// Default memory limit in bytes (default: 512 MB).
    pub default_memory_limit: u64,
    /// Default CPU quota in microseconds per 100ms period (default: 100_000 = 1 core).
    pub default_cpu_quota: i64,
    /// Allowed host path prefixes for bind mounts. Empty = deny all bind mounts.
    pub allowed_bind_mount_prefixes: Vec<std::path::PathBuf>,
    /// Default network mode for new containers (default: None = isolated).
    pub default_network_mode: NetworkMode,
    /// Whether to automatically pull images if not found locally (default: true).
    pub auto_pull: bool,
    /// Container cleanup policy.
    pub cleanup_policy: ContainerCleanupPolicy,
    /// Prefix for managed container names (default: "ai_assistant_").
    pub container_name_prefix: String,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            docker_host: None,
            default_timeout: Duration::from_secs(60),
            default_memory_limit: 512 * 1024 * 1024, // 512 MB
            default_cpu_quota: 100_000, // 1 core (100ms per 100ms period)
            allowed_bind_mount_prefixes: Vec::new(),
            default_network_mode: NetworkMode::None,
            auto_pull: true,
            cleanup_policy: ContainerCleanupPolicy::default(),
            container_name_prefix: "ai_assistant_".to_string(),
        }
    }
}

// =============================================================================
// Container record
// =============================================================================

/// Record of a managed container tracked by the executor.
#[derive(Debug, Clone)]
pub struct ContainerRecord {
    /// Docker container ID (hex string).
    pub container_id: String,
    /// Human-readable container name.
    pub name: String,
    /// Docker image used.
    pub image: String,
    /// Agent that created this container (if any).
    pub created_by_agent: Option<String>,
    /// Session that owns this container (if any).
    pub created_by_session: Option<String>,
    /// Unix timestamp when the container was created.
    pub created_at: u64,
    /// Current container status.
    pub status: ContainerStatus,
    /// Port mappings: (host_port, container_port).
    pub ports: Vec<(u16, u16)>,
    /// Bind mount mappings: (host_path, container_path).
    pub bind_mounts: Vec<(String, String)>,
}

// =============================================================================
// Create options
// =============================================================================

/// Options for creating a container.
#[derive(Debug, Clone, Default)]
pub struct CreateOptions {
    /// Override memory limit (bytes). None = use config default.
    pub memory_limit: Option<u64>,
    /// Override CPU quota. None = use config default.
    pub cpu_quota: Option<i64>,
    /// Override network mode. None = use config default.
    pub network_mode: Option<NetworkMode>,
    /// Environment variables as (key, value) pairs.
    pub env_vars: Vec<(String, String)>,
    /// Port mappings: (host_port, container_port).
    pub ports: Vec<(u16, u16)>,
    /// Bind mounts: (host_path, container_path).
    pub bind_mounts: Vec<(String, String)>,
    /// Working directory inside the container.
    pub working_dir: Option<String>,
    /// Override the image entrypoint.
    pub entrypoint: Option<Vec<String>>,
    /// Command to run (overrides CMD).
    pub cmd: Option<Vec<String>>,
    /// Labels to apply to the container.
    pub labels: HashMap<String, String>,
    /// Session ID for cleanup tracking.
    pub session_id: Option<String>,
    /// Agent ID for tracking which agent created the container.
    pub agent_id: Option<String>,
}

// =============================================================================
// Execution result
// =============================================================================

/// Result of executing a command inside a container.
#[derive(Debug, Clone)]
pub struct ExecResult {
    /// Standard output captured from the command.
    pub stdout: String,
    /// Standard error captured from the command.
    pub stderr: String,
    /// Exit code returned by the command.
    pub exit_code: i64,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
    /// Whether the command was killed due to timeout.
    pub timed_out: bool,
}

impl ExecResult {
    /// Returns true if the command exited successfully (exit code 0, no timeout).
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }

    /// Returns combined stdout + stderr output.
    pub fn combined_output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Errors from container operations.
#[derive(Debug)]
pub enum ContainerError {
    /// Docker daemon is not reachable.
    DockerNotAvailable(String),
    /// The requested image was not found (and auto-pull is disabled or failed).
    ImageNotFound(String),
    /// The specified container was not found.
    ContainerNotFound(String),
    /// A Docker API operation failed.
    OperationFailed(String),
    /// The operation exceeded its timeout.
    Timeout(String),
    /// The operation violates a cleanup or resource policy.
    PolicyViolation(String),
    /// An I/O error occurred (e.g., file read/write for copy operations).
    IoError(std::io::Error),
}

impl std::fmt::Display for ContainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerError::DockerNotAvailable(msg) => {
                write!(f, "Docker not available: {}", msg)
            }
            ContainerError::ImageNotFound(img) => {
                write!(f, "Image not found: {}", img)
            }
            ContainerError::ContainerNotFound(id) => {
                write!(f, "Container not found: {}", id)
            }
            ContainerError::OperationFailed(msg) => {
                write!(f, "Operation failed: {}", msg)
            }
            ContainerError::Timeout(msg) => {
                write!(f, "Timeout: {}", msg)
            }
            ContainerError::PolicyViolation(msg) => {
                write!(f, "Policy violation: {}", msg)
            }
            ContainerError::IoError(e) => {
                write!(f, "I/O error: {}", e)
            }
        }
    }
}

impl std::error::Error for ContainerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContainerError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ContainerError {
    fn from(err: std::io::Error) -> Self {
        ContainerError::IoError(err)
    }
}

impl From<bollard::errors::Error> for ContainerError {
    fn from(err: bollard::errors::Error) -> Self {
        ContainerError::OperationFailed(err.to_string())
    }
}

// =============================================================================
// Helper: generate management labels
// =============================================================================

/// Generate the standard set of labels applied to all managed containers.
fn management_labels(
    session_id: Option<&str>,
    agent_id: Option<&str>,
    extra: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert("ai.assistant.managed".to_string(), "true".to_string());
    if let Some(sid) = session_id {
        labels.insert("ai.assistant.session".to_string(), sid.to_string());
    }
    if let Some(aid) = agent_id {
        labels.insert("ai.assistant.agent".to_string(), aid.to_string());
    }
    // Merge user-supplied labels (user labels take precedence for non-management keys).
    for (k, v) in extra {
        labels.insert(k.clone(), v.clone());
    }
    labels
}

/// Get current Unix timestamp in seconds.
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// =============================================================================
// ContainerExecutor
// =============================================================================

/// Docker container execution engine.
///
/// Manages Docker containers for isolated code execution. Uses bollard for the
/// async Docker API, wrapped in a tokio runtime for a synchronous interface.
///
/// All containers created through this executor are labeled with
/// `ai.assistant.managed=true` for easy identification and cleanup.
pub struct ContainerExecutor {
    /// Tokio runtime for async -> sync bridge.
    runtime: Runtime,
    /// Bollard Docker client.
    docker: Docker,
    /// In-memory registry of managed containers (keyed by container ID).
    containers: HashMap<String, ContainerRecord>,
    /// Engine configuration.
    config: ContainerConfig,
}

impl std::fmt::Debug for ContainerExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerExecutor")
            .field("containers", &self.containers.len())
            .field("config", &self.config)
            .finish()
    }
}

impl ContainerExecutor {
    /// Create a new ContainerExecutor, connecting to the Docker daemon.
    ///
    /// The connection uses local defaults (Unix socket on Linux/macOS,
    /// named pipe on Windows). If the Docker daemon is not reachable,
    /// returns `ContainerError::DockerNotAvailable`.
    pub fn new(config: ContainerConfig) -> Result<Self, ContainerError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| ContainerError::DockerNotAvailable(format!("tokio runtime: {}", e)))?;

        let docker = if let Some(ref host) = config.docker_host {
            runtime
                .block_on(async {
                    Docker::connect_with_http(host, 120, bollard::API_DEFAULT_VERSION)
                })
                .map_err(|e| ContainerError::DockerNotAvailable(e.to_string()))?
        } else {
            Docker::connect_with_defaults()
                .map_err(|e| ContainerError::DockerNotAvailable(e.to_string()))?
        };

        Ok(Self {
            runtime,
            docker,
            containers: HashMap::new(),
            config,
        })
    }

    /// Check if the Docker daemon is available and responding.
    ///
    /// Creates a temporary runtime and attempts to ping the daemon.
    /// Returns `false` if Docker is not installed, not running, or not reachable.
    pub fn is_docker_available() -> bool {
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            if let Ok(docker) = Docker::connect_with_defaults() {
                return rt.block_on(async { docker.ping().await.is_ok() });
            }
        }
        false
    }

    /// Get the executor configuration.
    pub fn config(&self) -> &ContainerConfig {
        &self.config
    }

    // =========================================================================
    // Container Lifecycle
    // =========================================================================

    /// Create a new container from the specified image.
    ///
    /// Enforces cleanup policy limits (max_per_session, max_total) before creating.
    /// The container is labeled with `ai.assistant.managed=true` and optional
    /// session/agent tracking labels.
    ///
    /// Returns the Docker container ID on success.
    pub fn create(
        &mut self,
        image: &str,
        name: &str,
        opts: CreateOptions,
    ) -> Result<String, ContainerError> {
        // === Enforce cleanup policy ===
        let policy = &self.config.cleanup_policy;

        // Check max_total
        let active_count = self
            .containers
            .values()
            .filter(|r| r.status != ContainerStatus::Removed)
            .count();
        if active_count >= policy.max_total {
            return Err(ContainerError::PolicyViolation(format!(
                "Maximum total containers ({}) reached. Remove some containers first.",
                policy.max_total
            )));
        }

        // Check max_per_session
        if let Some(ref session_id) = opts.session_id {
            let session_count = self
                .containers
                .values()
                .filter(|r| {
                    r.created_by_session.as_deref() == Some(session_id.as_str())
                        && r.status != ContainerStatus::Removed
                })
                .count();
            if session_count >= policy.max_per_session {
                return Err(ContainerError::PolicyViolation(format!(
                    "Maximum containers per session ({}) reached for session '{}'.",
                    policy.max_per_session, session_id
                )));
            }
        }

        // === Build labels ===
        let labels = management_labels(
            opts.session_id.as_deref(),
            opts.agent_id.as_deref(),
            &opts.labels,
        );

        // === Build environment variables ===
        let env: Vec<String> = opts
            .env_vars
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();

        // === Build port bindings and exposed ports ===
        let mut port_bindings: HashMap<String, Option<Vec<PortBinding>>> = HashMap::new();
        let mut exposed_ports: HashMap<String, HashMap<(), ()>> = HashMap::new();
        for &(host_port, container_port) in &opts.ports {
            let container_key = format!("{}/tcp", container_port);
            exposed_ports.insert(container_key.clone(), HashMap::new());
            port_bindings.insert(
                container_key,
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".to_string()),
                    host_port: Some(host_port.to_string()),
                }]),
            );
        }

        // === Validate bind mounts (H8) ===
        // Reject dangerous host paths and enforce prefix whitelist
        const DANGEROUS_MOUNTS: &[&str] = &["/", "/etc", "/var", "/home", "/root", "/proc", "/sys", "/dev"];
        for (host_path, _) in &opts.bind_mounts {
            let normalized = host_path.replace('\\', "/");
            let trimmed = normalized.trim_end_matches('/');
            if DANGEROUS_MOUNTS.iter().any(|d| trimmed == *d) {
                return Err(ContainerError::PolicyViolation(format!(
                    "Bind mount to dangerous host path '{}' is not allowed",
                    host_path
                )));
            }
            if !self.config.allowed_bind_mount_prefixes.is_empty() {
                let hp = std::path::Path::new(host_path);
                let hp_canon = std::fs::canonicalize(hp).unwrap_or_else(|_| hp.to_path_buf());
                let allowed = self.config.allowed_bind_mount_prefixes.iter().any(|prefix| {
                    let prefix_canon = std::fs::canonicalize(prefix).unwrap_or_else(|_| prefix.clone());
                    hp_canon.starts_with(&prefix_canon)
                });
                if !allowed {
                    return Err(ContainerError::PolicyViolation(format!(
                        "Bind mount host path '{}' is not in allowed prefixes",
                        host_path
                    )));
                }
            }
        }
        let binds: Vec<String> = opts
            .bind_mounts
            .iter()
            .map(|(host, container)| format!("{}:{}", host, container))
            .collect();

        // === Determine resource limits ===
        let memory = opts
            .memory_limit
            .unwrap_or(self.config.default_memory_limit);
        let cpu_quota = opts.cpu_quota.unwrap_or(self.config.default_cpu_quota);
        let resolved_network = opts
            .network_mode
            .as_ref()
            .unwrap_or(&self.config.default_network_mode);
        // Security: warn when Host networking is used — shares host network stack (M9)
        if *resolved_network == NetworkMode::Host {
            log::warn!(
                "Container '{}' using Host networking — shares host network stack. \
                 This bypasses network isolation.",
                name
            );
        }
        let network_mode = resolved_network.to_docker_string();

        // === Build HostConfig ===
        let host_config = HostConfig {
            memory: Some(memory as i64),
            cpu_quota: if cpu_quota > 0 {
                Some(cpu_quota)
            } else {
                None
            },
            network_mode: Some(network_mode),
            port_bindings: if port_bindings.is_empty() {
                None
            } else {
                Some(port_bindings)
            },
            binds: if binds.is_empty() {
                None
            } else {
                Some(binds)
            },
            ..Default::default()
        };

        // === Build container config ===
        let full_name = format!("{}{}", self.config.container_name_prefix, name);

        let container_config = BollardConfig {
            image: Some(image.to_string()),
            hostname: None,
            env: if env.is_empty() { None } else { Some(env) },
            exposed_ports: if exposed_ports.is_empty() {
                None
            } else {
                Some(exposed_ports)
            },
            host_config: Some(host_config),
            labels: Some(labels),
            working_dir: opts.working_dir.clone(),
            entrypoint: opts.entrypoint.clone(),
            cmd: opts.cmd.clone(),
            tty: Some(false),
            attach_stdin: Some(false),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            ..Default::default()
        };

        let create_opts = CreateContainerOptions {
            name: full_name.as_str(),
            platform: None,
        };

        // === Create the container ===
        let response = self.runtime.block_on(async {
            self.docker
                .create_container(Some(create_opts), container_config)
                .await
        });

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                // Check if this is an image-not-found error and auto_pull is enabled
                let err_str = e.to_string();
                if self.config.auto_pull && err_str.contains("No such image") {
                    // Attempt to pull the image
                    log::info!("Image '{}' not found locally, pulling...", image);
                    self.pull_image(image)?;
                    // Retry create
                    let retry_config = BollardConfig {
                        image: Some(image.to_string()),
                        env: if opts.env_vars.is_empty() {
                            None
                        } else {
                            Some(
                                opts.env_vars
                                    .iter()
                                    .map(|(k, v)| format!("{}={}", k, v))
                                    .collect(),
                            )
                        },
                        working_dir: opts.working_dir.clone(),
                        entrypoint: opts.entrypoint.clone(),
                        cmd: opts.cmd.clone(),
                        tty: Some(false),
                        attach_stdin: Some(false),
                        attach_stdout: Some(true),
                        attach_stderr: Some(true),
                        ..Default::default()
                    };
                    let retry_opts = CreateContainerOptions {
                        name: full_name.as_str(),
                        platform: None,
                    };
                    self.runtime
                        .block_on(async {
                            self.docker
                                .create_container(Some(retry_opts), retry_config)
                                .await
                        })
                        .map_err(|e2| ContainerError::OperationFailed(e2.to_string()))?
                } else {
                    return Err(ContainerError::OperationFailed(err_str));
                }
            }
        };

        let container_id = response.id;

        // === Store the record ===
        let record = ContainerRecord {
            container_id: container_id.clone(),
            name: full_name,
            image: image.to_string(),
            created_by_agent: opts.agent_id.clone(),
            created_by_session: opts.session_id.clone(),
            created_at: unix_timestamp(),
            status: ContainerStatus::Created,
            ports: opts.ports.clone(),
            bind_mounts: opts.bind_mounts.clone(),
        };
        self.containers.insert(container_id.clone(), record);

        log::info!(
            "Created container '{}' (ID: {})",
            name,
            &container_id[..12.min(container_id.len())]
        );
        Ok(container_id)
    }

    /// Pull a Docker image from the registry.
    fn pull_image(&self, image: &str) -> Result<(), ContainerError> {
        use bollard::image::CreateImageOptions;

        let pull_opts = Some(CreateImageOptions {
            from_image: image,
            ..Default::default()
        });

        self.runtime.block_on(async {
            let mut stream = self.docker.create_image(pull_opts, None, None);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(info) => {
                        if let Some(status) = info.status {
                            log::debug!("Pull: {}", status);
                        }
                    }
                    Err(e) => {
                        return Err(ContainerError::ImageNotFound(format!(
                            "Failed to pull '{}': {}",
                            image, e
                        )));
                    }
                }
            }
            Ok(())
        })
    }

    /// Start a created container.
    pub fn start(&mut self, container_id: &str) -> Result<(), ContainerError> {
        self.runtime
            .block_on(async {
                self.docker
                    .start_container(container_id, None::<StartContainerOptions<String>>)
                    .await
            })
            .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

        // Update record status
        if let Some(record) = self.containers.get_mut(container_id) {
            record.status = ContainerStatus::Running;
        }

        log::info!(
            "Started container {}",
            &container_id[..12.min(container_id.len())]
        );
        Ok(())
    }

    /// Stop a running container with a grace period.
    ///
    /// The `timeout_secs` parameter controls how long Docker waits for the container
    /// to stop gracefully before sending SIGKILL.
    pub fn stop(&mut self, container_id: &str, timeout_secs: u32) -> Result<(), ContainerError> {
        let stop_opts = StopContainerOptions {
            t: timeout_secs as i64,
        };

        self.runtime
            .block_on(async {
                self.docker
                    .stop_container(container_id, Some(stop_opts))
                    .await
            })
            .map_err(|e| {
                let err_str = e.to_string();
                // "not running" is acceptable — the container is already stopped
                if err_str.contains("is not running") || err_str.contains("Not Modified") {
                    log::debug!("Container {} already stopped", container_id);
                    return ContainerError::OperationFailed(String::new());
                }
                ContainerError::OperationFailed(err_str)
            })
            .or_else(|e| {
                if e.to_string().is_empty() {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;

        // Update record status
        if let Some(record) = self.containers.get_mut(container_id) {
            record.status = ContainerStatus::Stopped;
        }

        log::info!(
            "Stopped container {}",
            &container_id[..12.min(container_id.len())]
        );
        Ok(())
    }

    /// Remove a container.
    ///
    /// If `force` is true, the container will be killed and removed even if running.
    pub fn remove(&mut self, container_id: &str, force: bool) -> Result<(), ContainerError> {
        let remove_opts = Some(RemoveContainerOptions {
            force,
            v: true, // Remove anonymous volumes
            ..Default::default()
        });

        self.runtime
            .block_on(async {
                self.docker
                    .remove_container(container_id, remove_opts)
                    .await
            })
            .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

        // Update record status
        if let Some(record) = self.containers.get_mut(container_id) {
            record.status = ContainerStatus::Removed;
        }

        log::info!(
            "Removed container {}",
            &container_id[..12.min(container_id.len())]
        );
        Ok(())
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute a command inside a running container.
    ///
    /// Captures stdout and stderr, enforces the specified timeout. If the timeout
    /// expires, the exec process is abandoned and `ExecResult::timed_out` is set.
    pub fn exec(
        &self,
        container_id: &str,
        cmd: &[&str],
        timeout: Duration,
    ) -> Result<ExecResult, ContainerError> {
        let start = Instant::now();

        let exec_config = CreateExecOptions {
            cmd: Some(cmd.iter().map(|s| s.to_string()).collect()),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            attach_stdin: Some(false),
            tty: Some(false),
            ..Default::default()
        };

        let result = self.runtime.block_on(async {
            // Create exec instance
            let exec = self
                .docker
                .create_exec(container_id, exec_config)
                .await
                .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

            // Start exec and attach to output
            let start_result = self
                .docker
                .start_exec(&exec.id, None)
                .await
                .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

            let mut stdout = String::new();
            let mut stderr = String::new();

            match start_result {
                StartExecResults::Attached { mut output, .. } => {
                    // Collect output with timeout
                    let collect_result = tokio::time::timeout(timeout, async {
                        while let Some(chunk) = output.next().await {
                            match chunk {
                                Ok(LogOutput::StdOut { message }) => {
                                    stdout
                                        .push_str(&String::from_utf8_lossy(&message));
                                }
                                Ok(LogOutput::StdErr { message }) => {
                                    stderr
                                        .push_str(&String::from_utf8_lossy(&message));
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    return Err(ContainerError::OperationFailed(
                                        e.to_string(),
                                    ));
                                }
                            }
                        }
                        Ok(())
                    })
                    .await;

                    match collect_result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => return Err(e),
                        Err(_) => {
                            // Timeout
                            return Ok(ExecResult {
                                stdout,
                                stderr,
                                exit_code: -1,
                                duration: start.elapsed(),
                                timed_out: true,
                            });
                        }
                    }
                }
                StartExecResults::Detached => {
                    return Err(ContainerError::OperationFailed(
                        "Exec started in detached mode unexpectedly".to_string(),
                    ));
                }
            }

            // Inspect exec to get exit code
            let inspect = self
                .docker
                .inspect_exec(&exec.id)
                .await
                .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

            let exit_code = inspect.exit_code.unwrap_or(-1);

            Ok(ExecResult {
                stdout,
                stderr,
                exit_code,
                duration: start.elapsed(),
                timed_out: false,
            })
        });

        result
    }

    // =========================================================================
    // Inspection
    // =========================================================================

    /// Get container logs (last `tail` lines).
    pub fn logs(&self, container_id: &str, tail: usize) -> Result<String, ContainerError> {
        let log_opts = Some(LogsOptions::<String> {
            stdout: true,
            stderr: true,
            tail: tail.to_string(),
            ..Default::default()
        });

        self.runtime.block_on(async {
            let mut stream = self.docker.logs(container_id, log_opts);
            let mut output = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(LogOutput::StdOut { message }) => {
                        output.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(LogOutput::StdErr { message }) => {
                        output.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        return Err(ContainerError::OperationFailed(e.to_string()));
                    }
                }
            }
            Ok(output)
        })
    }

    /// List all managed containers (returns references to records).
    pub fn list(&self) -> Vec<&ContainerRecord> {
        self.containers.values().collect()
    }

    /// Get the status of a specific container.
    pub fn status(&self, container_id: &str) -> Option<&ContainerStatus> {
        self.containers.get(container_id).map(|r| &r.status)
    }

    // =========================================================================
    // File Transfer
    // =========================================================================

    /// Copy a file from the host filesystem into a container.
    ///
    /// The file is packed into a tar archive and uploaded via the Docker API.
    pub fn copy_to(
        &self,
        container_id: &str,
        src: &std::path::Path,
        dest: &str,
    ) -> Result<(), ContainerError> {
        // Read the source file
        let file_data = std::fs::read(src)?;
        let file_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");

        // Build a tar archive in memory
        let tar_bytes = build_tar_archive(file_name, &file_data)?;

        let upload_opts = UploadToContainerOptions {
            path: dest.to_string(),
            ..Default::default()
        };

        self.runtime
            .block_on(async {
                self.docker
                    .upload_to_container(
                        container_id,
                        Some(upload_opts),
                        tar_bytes.into(),
                    )
                    .await
            })
            .map_err(|e| ContainerError::OperationFailed(e.to_string()))?;

        log::debug!("Copied {} -> {}:{}", src.display(), container_id, dest);
        Ok(())
    }

    /// Copy a file from a container to the host filesystem.
    ///
    /// Downloads a tar archive from the Docker API and extracts the file.
    pub fn copy_from(
        &self,
        container_id: &str,
        src: &str,
        dest: &std::path::Path,
    ) -> Result<(), ContainerError> {
        let download_opts = Some(DownloadFromContainerOptions { path: src });

        let data = self.runtime.block_on(async {
            let mut stream = self
                .docker
                .download_from_container(container_id, download_opts);
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(data) => bytes.extend_from_slice(&data),
                    Err(e) => {
                        return Err(ContainerError::OperationFailed(e.to_string()));
                    }
                }
            }
            Ok(bytes)
        })?;

        // Extract first file from the tar archive
        let extracted = extract_first_from_tar(&data)?;

        // Ensure parent directory exists
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, &extracted)?;

        log::debug!("Copied {}:{} -> {}", container_id, src, dest.display());
        Ok(())
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    /// Remove all containers associated with a specific session.
    ///
    /// Returns the number of containers removed.
    pub fn cleanup_session(&mut self, session_id: &str) -> usize {
        let ids: Vec<String> = self
            .containers
            .values()
            .filter(|r| {
                r.created_by_session.as_deref() == Some(session_id)
                    && r.status != ContainerStatus::Removed
            })
            .map(|r| r.container_id.clone())
            .collect();

        let mut removed = 0;
        for id in &ids {
            if self.remove(id, true).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    /// Remove containers that have exceeded `auto_remove_after_secs`.
    ///
    /// Returns the number of containers removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let now = unix_timestamp();
        let max_age = match self.config.cleanup_policy.auto_remove_after_secs {
            Some(secs) => secs,
            None => return 0, // No expiry policy
        };

        let ids: Vec<String> = self
            .containers
            .values()
            .filter(|r| {
                r.status != ContainerStatus::Removed && now.saturating_sub(r.created_at) >= max_age
            })
            .map(|r| r.container_id.clone())
            .collect();

        let mut removed = 0;
        for id in &ids {
            if self.remove(id, true).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    /// Remove all managed containers tracked by this executor.
    ///
    /// Returns the number of containers removed.
    pub fn cleanup_all(&mut self) -> usize {
        let ids: Vec<String> = self
            .containers
            .values()
            .filter(|r| r.status != ContainerStatus::Removed)
            .map(|r| r.container_id.clone())
            .collect();

        let mut removed = 0;
        for id in &ids {
            if self.remove(id, true).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

// =============================================================================
// Tar helpers (in-memory, no external dep — flate2 is already in this crate)
// =============================================================================

/// Build a minimal tar archive containing a single file.
///
/// The tar format is simple: a 512-byte header followed by file data padded to
/// 512-byte blocks, followed by two zero blocks as end-of-archive marker.
fn build_tar_archive(file_name: &str, data: &[u8]) -> Result<Vec<u8>, ContainerError> {
    let mut archive = Vec::new();

    // Build tar header (512 bytes)
    let mut header = [0u8; 512];

    // File name (0..100)
    let name_bytes = file_name.as_bytes();
    let name_len = name_bytes.len().min(99);
    header[..name_len].copy_from_slice(&name_bytes[..name_len]);

    // File mode (100..108) - "0000644\0"
    header[100..108].copy_from_slice(b"0000644\0");

    // Owner ID (108..116) - "0000000\0"
    header[108..116].copy_from_slice(b"0000000\0");

    // Group ID (116..124) - "0000000\0"
    header[116..124].copy_from_slice(b"0000000\0");

    // File size in octal (124..136)
    let size_str = format!("{:011o}\0", data.len());
    header[124..136].copy_from_slice(size_str.as_bytes());

    // Modification time (136..148) - current time in octal
    let mtime = unix_timestamp();
    let mtime_str = format!("{:011o}\0", mtime);
    header[136..148].copy_from_slice(mtime_str.as_bytes());

    // Checksum placeholder (148..156) - spaces for calculation
    header[148..156].copy_from_slice(b"        ");

    // Type flag (156) - '0' = regular file
    header[156] = b'0';

    // USTAR magic (257..263)
    header[257..263].copy_from_slice(b"ustar\0");

    // USTAR version (263..265)
    header[263..265].copy_from_slice(b"00");

    // Calculate checksum
    let checksum: u32 = header.iter().map(|&b| b as u32).sum();
    let checksum_str = format!("{:06o}\0 ", checksum);
    header[148..156].copy_from_slice(checksum_str.as_bytes());

    archive.extend_from_slice(&header);

    // File data, padded to 512-byte boundary
    archive.extend_from_slice(data);
    let padding = (512 - (data.len() % 512)) % 512;
    archive.extend_from_slice(&vec![0u8; padding]);

    // End of archive: two 512-byte zero blocks
    archive.extend_from_slice(&[0u8; 1024]);

    Ok(archive)
}

/// Extract the first file's contents from a tar archive.
fn extract_first_from_tar(tar_data: &[u8]) -> Result<Vec<u8>, ContainerError> {
    if tar_data.len() < 512 {
        return Err(ContainerError::OperationFailed(
            "Tar archive too small".to_string(),
        ));
    }

    // Parse header to get file size
    let header = &tar_data[..512];

    // Size field is at offset 124, 12 bytes, octal, null-terminated
    let size_field = &header[124..136];
    let size_str = std::str::from_utf8(size_field)
        .map_err(|_| ContainerError::OperationFailed("Invalid tar header".to_string()))?
        .trim_matches('\0')
        .trim();

    let file_size = usize::from_str_radix(size_str, 8)
        .map_err(|_| ContainerError::OperationFailed("Invalid file size in tar".to_string()))?;

    if tar_data.len() < 512 + file_size {
        return Err(ContainerError::OperationFailed(
            "Tar archive truncated".to_string(),
        ));
    }

    Ok(tar_data[512..512 + file_size].to_vec())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper: skip integration tests when Docker is not available
    // =========================================================================

    fn skip_if_no_docker() -> bool {
        !ContainerExecutor::is_docker_available()
    }

    // =========================================================================
    // Configuration tests
    // =========================================================================

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.docker_host, None);
        assert_eq!(config.default_timeout, Duration::from_secs(60));
        assert_eq!(config.default_memory_limit, 512 * 1024 * 1024);
        assert_eq!(config.default_cpu_quota, 100_000); // 1 CPU core (M8: safe default)
        assert_eq!(config.default_network_mode, NetworkMode::None);
        assert!(config.auto_pull);
        assert_eq!(config.container_name_prefix, "ai_assistant_");
    }

    #[test]
    fn test_container_config_custom() {
        let config = ContainerConfig {
            docker_host: Some("tcp://localhost:2375".to_string()),
            default_timeout: Duration::from_secs(120),
            default_memory_limit: 1024 * 1024 * 1024,
            default_cpu_quota: 50000,
            default_network_mode: NetworkMode::Bridge,
            auto_pull: false,
            cleanup_policy: ContainerCleanupPolicy {
                max_per_session: 10,
                max_total: 50,
                auto_remove_after_secs: None,
                cleanup_on_session_end: false,
            },
            container_name_prefix: "custom_".to_string(),
            allowed_bind_mount_prefixes: Vec::new(),
        };
        assert_eq!(
            config.docker_host,
            Some("tcp://localhost:2375".to_string())
        );
        assert_eq!(config.default_timeout, Duration::from_secs(120));
        assert_eq!(config.default_memory_limit, 1024 * 1024 * 1024);
        assert_eq!(config.default_cpu_quota, 50000);
        assert_eq!(config.default_network_mode, NetworkMode::Bridge);
        assert!(!config.auto_pull);
        assert_eq!(config.container_name_prefix, "custom_");
    }

    #[test]
    fn test_cleanup_policy_default() {
        let policy = ContainerCleanupPolicy::default();
        assert_eq!(policy.max_per_session, 5);
        assert_eq!(policy.max_total, 20);
        assert_eq!(policy.auto_remove_after_secs, Some(3600));
        assert!(policy.cleanup_on_session_end);
    }

    // =========================================================================
    // NetworkMode tests
    // =========================================================================

    #[test]
    fn test_network_mode_variants() {
        assert_eq!(NetworkMode::None, NetworkMode::None);
        assert_eq!(NetworkMode::Bridge, NetworkMode::Bridge);
        assert_eq!(NetworkMode::Host, NetworkMode::Host);
        assert_eq!(
            NetworkMode::Custom("my_net".to_string()),
            NetworkMode::Custom("my_net".to_string())
        );
        assert_ne!(NetworkMode::None, NetworkMode::Bridge);
    }

    #[test]
    fn test_network_mode_to_string() {
        assert_eq!(NetworkMode::None.to_docker_string(), "none");
        assert_eq!(NetworkMode::Bridge.to_docker_string(), "bridge");
        assert_eq!(NetworkMode::Host.to_docker_string(), "host");
        assert_eq!(
            NetworkMode::Custom("my_net".to_string()).to_docker_string(),
            "my_net"
        );
    }

    #[test]
    fn test_network_mode_display() {
        assert_eq!(format!("{}", NetworkMode::None), "none");
        assert_eq!(format!("{}", NetworkMode::Bridge), "bridge");
        assert_eq!(format!("{}", NetworkMode::Host), "host");
        assert_eq!(
            format!("{}", NetworkMode::Custom("overlay_net".to_string())),
            "custom(overlay_net)"
        );
    }

    // =========================================================================
    // ContainerStatus tests
    // =========================================================================

    #[test]
    fn test_container_status_transitions() {
        // Verify expected lifecycle transitions
        let mut status = ContainerStatus::Created;
        assert_eq!(status, ContainerStatus::Created);

        status = ContainerStatus::Running;
        assert_eq!(status, ContainerStatus::Running);

        status = ContainerStatus::Paused;
        assert_eq!(status, ContainerStatus::Paused);

        status = ContainerStatus::Running;
        assert_eq!(status, ContainerStatus::Running);

        status = ContainerStatus::Stopped;
        assert_eq!(status, ContainerStatus::Stopped);

        status = ContainerStatus::Removed;
        assert_eq!(status, ContainerStatus::Removed);
    }

    #[test]
    fn test_container_status_display() {
        assert_eq!(format!("{}", ContainerStatus::Created), "created");
        assert_eq!(format!("{}", ContainerStatus::Running), "running");
        assert_eq!(format!("{}", ContainerStatus::Paused), "paused");
        assert_eq!(format!("{}", ContainerStatus::Stopped), "stopped");
        assert_eq!(format!("{}", ContainerStatus::Removed), "removed");
    }

    // =========================================================================
    // ContainerRecord tests
    // =========================================================================

    fn make_test_record(id: &str, session: Option<&str>) -> ContainerRecord {
        ContainerRecord {
            container_id: id.to_string(),
            name: format!("ai_assistant_test_{}", id),
            image: "alpine:latest".to_string(),
            created_by_agent: Some("test_agent".to_string()),
            created_by_session: session.map(|s| s.to_string()),
            created_at: unix_timestamp(),
            status: ContainerStatus::Created,
            ports: vec![(8080, 80)],
            bind_mounts: vec![("/tmp/host".to_string(), "/data".to_string())],
        }
    }

    #[test]
    fn test_container_record_creation() {
        let record = make_test_record("abc123", Some("session_1"));
        assert_eq!(record.container_id, "abc123");
        assert_eq!(record.name, "ai_assistant_test_abc123");
        assert_eq!(record.image, "alpine:latest");
        assert_eq!(
            record.created_by_agent,
            Some("test_agent".to_string())
        );
        assert_eq!(
            record.created_by_session,
            Some("session_1".to_string())
        );
        assert_eq!(record.status, ContainerStatus::Created);
        assert_eq!(record.ports, vec![(8080, 80)]);
        assert_eq!(
            record.bind_mounts,
            vec![("/tmp/host".to_string(), "/data".to_string())]
        );
    }

    #[test]
    fn test_container_record_clone() {
        let record = make_test_record("def456", Some("session_2"));
        let cloned = record.clone();
        assert_eq!(record.container_id, cloned.container_id);
        assert_eq!(record.name, cloned.name);
        assert_eq!(record.image, cloned.image);
        assert_eq!(record.created_by_agent, cloned.created_by_agent);
        assert_eq!(record.created_by_session, cloned.created_by_session);
        assert_eq!(record.created_at, cloned.created_at);
        assert_eq!(record.status, cloned.status);
        assert_eq!(record.ports, cloned.ports);
        assert_eq!(record.bind_mounts, cloned.bind_mounts);
    }

    // =========================================================================
    // CreateOptions tests
    // =========================================================================

    #[test]
    fn test_create_options_default() {
        let opts = CreateOptions::default();
        assert!(opts.memory_limit.is_none());
        assert!(opts.cpu_quota.is_none());
        assert!(opts.network_mode.is_none());
        assert!(opts.env_vars.is_empty());
        assert!(opts.ports.is_empty());
        assert!(opts.bind_mounts.is_empty());
        assert!(opts.working_dir.is_none());
        assert!(opts.entrypoint.is_none());
        assert!(opts.cmd.is_none());
        assert!(opts.labels.is_empty());
        assert!(opts.session_id.is_none());
        assert!(opts.agent_id.is_none());
    }

    #[test]
    fn test_create_options_with_env() {
        let opts = CreateOptions {
            env_vars: vec![
                ("PATH".to_string(), "/usr/bin".to_string()),
                ("HOME".to_string(), "/root".to_string()),
            ],
            ..Default::default()
        };
        assert_eq!(opts.env_vars.len(), 2);
        assert_eq!(opts.env_vars[0], ("PATH".to_string(), "/usr/bin".to_string()));
        assert_eq!(opts.env_vars[1], ("HOME".to_string(), "/root".to_string()));
    }

    #[test]
    fn test_create_options_with_ports() {
        let opts = CreateOptions {
            ports: vec![(8080, 80), (4433, 443)],
            ..Default::default()
        };
        assert_eq!(opts.ports.len(), 2);
        assert_eq!(opts.ports[0], (8080, 80));
        assert_eq!(opts.ports[1], (4433, 443));
    }

    #[test]
    fn test_create_options_with_mounts() {
        let opts = CreateOptions {
            bind_mounts: vec![
                ("/host/data".to_string(), "/container/data".to_string()),
                ("/host/config".to_string(), "/etc/app".to_string()),
            ],
            ..Default::default()
        };
        assert_eq!(opts.bind_mounts.len(), 2);
        assert_eq!(
            opts.bind_mounts[0],
            ("/host/data".to_string(), "/container/data".to_string())
        );
    }

    #[test]
    fn test_create_options_labels() {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), "my_app".to_string());
        labels.insert("version".to_string(), "1.0".to_string());

        let opts = CreateOptions {
            labels: labels.clone(),
            ..Default::default()
        };
        assert_eq!(opts.labels.len(), 2);
        assert_eq!(opts.labels.get("app"), Some(&"my_app".to_string()));
        assert_eq!(opts.labels.get("version"), Some(&"1.0".to_string()));
    }

    // =========================================================================
    // ExecResult tests
    // =========================================================================

    #[test]
    fn test_exec_result_success() {
        let result = ExecResult {
            stdout: "hello world\n".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration: Duration::from_millis(150),
            timed_out: false,
        };
        assert!(result.success());
        assert_eq!(result.exit_code, 0);
        assert!(!result.timed_out);
        assert_eq!(result.stdout, "hello world\n");
        assert!(result.stderr.is_empty());
    }

    #[test]
    fn test_exec_result_timeout() {
        let result = ExecResult {
            stdout: "partial output".to_string(),
            stderr: String::new(),
            exit_code: -1,
            duration: Duration::from_secs(30),
            timed_out: true,
        };
        assert!(!result.success());
        assert!(result.timed_out);
        assert_eq!(result.exit_code, -1);
    }

    #[test]
    fn test_exec_result_combined_output() {
        // Stdout only
        let result = ExecResult {
            stdout: "out".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration: Duration::from_millis(10),
            timed_out: false,
        };
        assert_eq!(result.combined_output(), "out");

        // Stderr only
        let result = ExecResult {
            stdout: String::new(),
            stderr: "err".to_string(),
            exit_code: 1,
            duration: Duration::from_millis(10),
            timed_out: false,
        };
        assert_eq!(result.combined_output(), "err");

        // Both
        let result = ExecResult {
            stdout: "out".to_string(),
            stderr: "err".to_string(),
            exit_code: 0,
            duration: Duration::from_millis(10),
            timed_out: false,
        };
        assert_eq!(result.combined_output(), "out\nerr");
    }

    #[test]
    fn test_exec_result_failure_not_timeout() {
        let result = ExecResult {
            stdout: String::new(),
            stderr: "command not found\n".to_string(),
            exit_code: 127,
            duration: Duration::from_millis(5),
            timed_out: false,
        };
        assert!(!result.success());
        assert!(!result.timed_out);
        assert_eq!(result.exit_code, 127);
    }

    // =========================================================================
    // ContainerError tests
    // =========================================================================

    #[test]
    fn test_container_error_display() {
        let e = ContainerError::DockerNotAvailable("connection refused".to_string());
        assert_eq!(
            format!("{}", e),
            "Docker not available: connection refused"
        );

        let e = ContainerError::ImageNotFound("foo:latest".to_string());
        assert_eq!(format!("{}", e), "Image not found: foo:latest");

        let e = ContainerError::ContainerNotFound("abc123".to_string());
        assert_eq!(format!("{}", e), "Container not found: abc123");

        let e = ContainerError::OperationFailed("something broke".to_string());
        assert_eq!(format!("{}", e), "Operation failed: something broke");

        let e = ContainerError::Timeout("30s exceeded".to_string());
        assert_eq!(format!("{}", e), "Timeout: 30s exceeded");

        let e = ContainerError::PolicyViolation("max containers".to_string());
        assert_eq!(format!("{}", e), "Policy violation: max containers");
    }

    #[test]
    fn test_container_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let container_err: ContainerError = io_err.into();
        match container_err {
            ContainerError::IoError(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_container_error_is_error_trait() {
        // Verify std::error::Error is implemented
        let e = ContainerError::OperationFailed("test".to_string());
        let _dyn_err: &dyn std::error::Error = &e;
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_container_error_source() {
        // IoError variant should have a source
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let container_err = ContainerError::IoError(io_err);
        assert!(
            std::error::Error::source(&container_err).is_some(),
            "IoError should provide source"
        );

        // Other variants should not
        let other = ContainerError::Timeout("t".to_string());
        assert!(
            std::error::Error::source(&other).is_none(),
            "Timeout should not provide source"
        );
    }

    // =========================================================================
    // Labels tests
    // =========================================================================

    #[test]
    fn test_labels_generation() {
        let extra = HashMap::new();
        let labels = management_labels(Some("sess_1"), Some("agent_a"), &extra);
        assert_eq!(labels.get("ai.assistant.managed"), Some(&"true".to_string()));
        assert_eq!(
            labels.get("ai.assistant.session"),
            Some(&"sess_1".to_string())
        );
        assert_eq!(
            labels.get("ai.assistant.agent"),
            Some(&"agent_a".to_string())
        );
    }

    #[test]
    fn test_labels_without_session_and_agent() {
        let extra = HashMap::new();
        let labels = management_labels(None, None, &extra);
        assert_eq!(labels.get("ai.assistant.managed"), Some(&"true".to_string()));
        assert!(labels.get("ai.assistant.session").is_none());
        assert!(labels.get("ai.assistant.agent").is_none());
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_labels_with_extra() {
        let mut extra = HashMap::new();
        extra.insert("custom.key".to_string(), "custom.value".to_string());
        let labels = management_labels(Some("s1"), None, &extra);
        assert_eq!(labels.get("ai.assistant.managed"), Some(&"true".to_string()));
        assert_eq!(
            labels.get("custom.key"),
            Some(&"custom.value".to_string())
        );
    }

    // =========================================================================
    // Policy enforcement tests (unit logic, no Docker needed)
    // =========================================================================

    #[test]
    fn test_container_name_prefix() {
        let config = ContainerConfig {
            container_name_prefix: "test_prefix_".to_string(),
            ..Default::default()
        };
        let full_name = format!("{}my_container", config.container_name_prefix);
        assert_eq!(full_name, "test_prefix_my_container");
    }

    #[test]
    fn test_policy_max_per_session() {
        let policy = ContainerCleanupPolicy {
            max_per_session: 3,
            ..Default::default()
        };

        // Simulate counting containers for a session
        let containers: Vec<ContainerRecord> = (0..3)
            .map(|i| ContainerRecord {
                container_id: format!("id_{}", i),
                name: format!("test_{}", i),
                image: "alpine".to_string(),
                created_by_agent: None,
                created_by_session: Some("sess_a".to_string()),
                created_at: unix_timestamp(),
                status: ContainerStatus::Running,
                ports: vec![],
                bind_mounts: vec![],
            })
            .collect();

        let session_count = containers
            .iter()
            .filter(|r| {
                r.created_by_session.as_deref() == Some("sess_a")
                    && r.status != ContainerStatus::Removed
            })
            .count();
        assert_eq!(session_count, 3);
        assert!(session_count >= policy.max_per_session);
    }

    #[test]
    fn test_policy_max_total() {
        let policy = ContainerCleanupPolicy {
            max_total: 2,
            ..Default::default()
        };

        let containers: Vec<ContainerRecord> = (0..3)
            .map(|i| ContainerRecord {
                container_id: format!("id_{}", i),
                name: format!("test_{}", i),
                image: "alpine".to_string(),
                created_by_agent: None,
                created_by_session: None,
                created_at: unix_timestamp(),
                status: if i < 2 {
                    ContainerStatus::Running
                } else {
                    ContainerStatus::Removed
                },
                ports: vec![],
                bind_mounts: vec![],
            })
            .collect();

        let active_count = containers
            .iter()
            .filter(|r| r.status != ContainerStatus::Removed)
            .count();
        assert_eq!(active_count, 2);
        assert!(active_count >= policy.max_total);
    }

    // =========================================================================
    // Cleanup logic tests (no Docker needed — tests time comparison)
    // =========================================================================

    #[test]
    fn test_cleanup_session_empty() {
        // With no containers, cleanup should remove 0
        let containers: HashMap<String, ContainerRecord> = HashMap::new();
        let ids: Vec<String> = containers
            .values()
            .filter(|r| {
                r.created_by_session.as_deref() == Some("nonexistent")
                    && r.status != ContainerStatus::Removed
            })
            .map(|r| r.container_id.clone())
            .collect();
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_cleanup_expired_logic() {
        let max_age: u64 = 3600;
        let now = unix_timestamp();

        // Container created 2 hours ago — should be expired
        let old_record = ContainerRecord {
            container_id: "old_id".to_string(),
            name: "old_container".to_string(),
            image: "alpine".to_string(),
            created_by_agent: None,
            created_by_session: None,
            created_at: now.saturating_sub(7200), // 2 hours ago
            status: ContainerStatus::Stopped,
            ports: vec![],
            bind_mounts: vec![],
        };

        // Container created 30 minutes ago — should NOT be expired
        let new_record = ContainerRecord {
            container_id: "new_id".to_string(),
            name: "new_container".to_string(),
            image: "alpine".to_string(),
            created_by_agent: None,
            created_by_session: None,
            created_at: now.saturating_sub(1800), // 30 minutes ago
            status: ContainerStatus::Running,
            ports: vec![],
            bind_mounts: vec![],
        };

        // Already removed — should NOT be counted
        let removed_record = ContainerRecord {
            container_id: "removed_id".to_string(),
            name: "removed_container".to_string(),
            image: "alpine".to_string(),
            created_by_agent: None,
            created_by_session: None,
            created_at: now.saturating_sub(7200),
            status: ContainerStatus::Removed,
            ports: vec![],
            bind_mounts: vec![],
        };

        let containers = vec![&old_record, &new_record, &removed_record];

        let expired_ids: Vec<&str> = containers
            .iter()
            .filter(|r| {
                r.status != ContainerStatus::Removed
                    && now.saturating_sub(r.created_at) >= max_age
            })
            .map(|r| r.container_id.as_str())
            .collect();

        assert_eq!(expired_ids.len(), 1);
        assert_eq!(expired_ids[0], "old_id");
    }

    #[test]
    fn test_cleanup_expired_no_policy() {
        // When auto_remove_after_secs is None, nothing expires
        let policy = ContainerCleanupPolicy {
            auto_remove_after_secs: None,
            ..Default::default()
        };
        assert!(policy.auto_remove_after_secs.is_none());
        // The executor's cleanup_expired returns 0 in this case (tested by logic).
    }

    // =========================================================================
    // Tar archive helper tests
    // =========================================================================

    #[test]
    fn test_tar_archive_roundtrip() {
        let data = b"Hello, container world!";
        let archive = build_tar_archive("test.txt", data).unwrap();

        // Archive should be at least: 512 (header) + 512 (data padded) + 1024 (end marker)
        assert!(archive.len() >= 2048);

        // Extract and verify
        let extracted = extract_first_from_tar(&archive).unwrap();
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_tar_archive_empty_file() {
        let data = b"";
        let archive = build_tar_archive("empty.txt", data).unwrap();

        // Should have header + end marker (no data block since file is empty)
        assert!(archive.len() >= 512 + 1024);

        let extracted = extract_first_from_tar(&archive).unwrap();
        assert!(extracted.is_empty());
    }

    #[test]
    fn test_tar_extract_invalid() {
        let result = extract_first_from_tar(&[0u8; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tar_archive_large_file() {
        let data = vec![0xAB; 2048]; // Larger than one 512-byte block
        let archive = build_tar_archive("large.bin", &data).unwrap();
        let extracted = extract_first_from_tar(&archive).unwrap();
        assert_eq!(extracted.len(), 2048);
        assert_eq!(extracted, data);
    }

    // =========================================================================
    // Unix timestamp helper
    // =========================================================================

    #[test]
    fn test_unix_timestamp() {
        let ts = unix_timestamp();
        // Should be a reasonable recent timestamp (after 2024-01-01)
        assert!(ts > 1_704_067_200);
        // Should not be too far in the future
        assert!(ts < 2_000_000_000);
    }

    // =========================================================================
    // Debug formatting
    // =========================================================================

    #[test]
    fn test_container_config_debug() {
        let config = ContainerConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ContainerConfig"));
        assert!(debug.contains("auto_pull"));
    }

    #[test]
    fn test_container_error_debug() {
        let err = ContainerError::PolicyViolation("too many containers".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("PolicyViolation"));
        assert!(debug.contains("too many containers"));
    }
}
