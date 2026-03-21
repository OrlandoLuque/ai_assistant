//! Container Backend Abstraction (Block C)
//!
//! Defines a common trait `ContainerBackend` and shared types
//! that unify `container_tools::ContainerExecutor` (Docker CLI)
//! and `container_executor::ContainerExecutor` (Bollard API).

use std::collections::HashMap;
use std::time::Duration;

/// Errors from container operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BackendError {
    /// Docker daemon not available.
    NotAvailable(String),
    /// Requested image not found.
    ImageNotFound(String),
    /// Requested container not found.
    ContainerNotFound(String),
    /// Operation failed with a message.
    OperationFailed(String),
    /// Operation timed out.
    Timeout,
    /// Policy violation (e.g., disallowed mount).
    PolicyViolation(String),
    /// I/O error.
    Io(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable(msg) => write!(f, "Docker not available: {}", msg),
            Self::ImageNotFound(img) => write!(f, "Image not found: {}", img),
            Self::ContainerNotFound(id) => write!(f, "Container not found: {}", id),
            Self::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::PolicyViolation(msg) => write!(f, "Policy violation: {}", msg),
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

impl Default for BackendError {
    fn default() -> Self {
        Self::OperationFailed("unknown error".to_string())
    }
}

/// Options for creating a container.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct BackendCreateOptions {
    /// Environment variables (key → value).
    pub env_vars: HashMap<String, String>,
    /// Port mappings (host_port → container_port).
    pub ports: Vec<(u16, u16)>,
    /// Bind mounts ("host_path:container_path").
    pub bind_mounts: Vec<String>,
    /// Working directory inside the container.
    pub working_dir: Option<String>,
    /// Command to run.
    pub cmd: Option<Vec<String>>,
    /// Memory limit in bytes (0 = unlimited).
    pub memory_limit: u64,
    /// CPU quota (0 = unlimited).
    pub cpu_quota: i64,
    /// Labels for the container.
    pub labels: HashMap<String, String>,
}

/// Result of executing a command inside a container.
#[derive(Debug, Clone)]
pub struct BackendExecResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Process exit code.
    pub exit_code: i64,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Whether the command timed out.
    pub timed_out: bool,
}

impl BackendExecResult {
    /// Whether the command exited successfully (code 0, no timeout).
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }

    /// Combined stdout + stderr.
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

/// Summary of a container's state.
#[derive(Debug, Clone)]
pub struct BackendContainerInfo {
    /// Container ID.
    pub id: String,
    /// Container name.
    pub name: String,
    /// Image used.
    pub image: String,
    /// Current status description.
    pub status: String,
}

/// Trait for container backends.
///
/// Both the Docker CLI backend (`container_tools`) and the Bollard API
/// backend (`container_executor`) implement this trait, providing a
/// unified interface for container lifecycle operations.
pub trait ContainerBackend {
    /// Check whether the container runtime is available.
    fn is_available(&self) -> Result<bool, BackendError>;

    /// Create a container from an image. Returns the container ID.
    fn create(
        &mut self,
        image: &str,
        name: Option<&str>,
        options: &BackendCreateOptions,
    ) -> Result<String, BackendError>;

    /// Start a created container.
    fn start(&mut self, container_id: &str) -> Result<(), BackendError>;

    /// Stop a running container.
    fn stop(&mut self, container_id: &str, timeout_secs: u64) -> Result<(), BackendError>;

    /// Remove a container (optionally force-remove running containers).
    fn remove(&mut self, container_id: &str, force: bool) -> Result<(), BackendError>;

    /// Execute a command inside a running container.
    fn exec(
        &self,
        container_id: &str,
        cmd: &[&str],
        timeout: Duration,
    ) -> Result<BackendExecResult, BackendError>;

    /// Get recent logs from a container.
    fn logs(&self, container_id: &str, tail: u64) -> Result<String, BackendError>;

    /// List known containers.
    fn list(&self) -> Result<Vec<BackendContainerInfo>, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_exec_result_success() {
        let result = BackendExecResult {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration: Duration::from_millis(100),
            timed_out: false,
        };
        assert!(result.success());
        assert_eq!(result.combined_output(), "hello");
    }

    #[test]
    fn test_backend_exec_result_failure() {
        let result = BackendExecResult {
            stdout: String::new(),
            stderr: "error".to_string(),
            exit_code: 1,
            duration: Duration::from_millis(50),
            timed_out: false,
        };
        assert!(!result.success());
    }

    #[test]
    fn test_backend_exec_result_timeout() {
        let result = BackendExecResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            duration: Duration::from_secs(30),
            timed_out: true,
        };
        assert!(!result.success());
    }

    #[test]
    fn test_backend_create_options_default() {
        let opts = BackendCreateOptions::default();
        assert!(opts.env_vars.is_empty());
        assert!(opts.ports.is_empty());
        assert!(opts.cmd.is_none());
        assert_eq!(opts.memory_limit, 0);
    }

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::Timeout;
        assert_eq!(err.to_string(), "Operation timed out");

        let err = BackendError::ImageNotFound("alpine:latest".to_string());
        assert!(err.to_string().contains("alpine:latest"));
    }
}
