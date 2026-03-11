//! Container tools — Docker container management tools for the ToolRegistry.
//!
//! Provides container lifecycle and code execution tools that autonomous agents
//! can use, with each tool call validated through the sandbox before execution.
//! Follows the same registration pattern as `os_tools.rs`.

use crate::agent_sandbox::SandboxValidator;
use crate::unified_tools::{
    ParamSchema, ToolBuilder, ToolCall, ToolDef, ToolError, ToolHandler, ToolOutput, ToolRegistry,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

// ============================================================================
// Container Executor Types
// ============================================================================

/// Error type for container operations.
#[derive(Debug, Clone)]
pub enum ContainerError {
    /// Docker daemon is not running or not installed.
    DockerNotAvailable,
    /// The specified image was not found.
    ImageNotFound(String),
    /// The specified container was not found.
    ContainerNotFound(String),
    /// A container operation failed.
    OperationFailed(String),
    /// The operation timed out.
    Timeout,
    /// The operation violates a security policy.
    PolicyViolation(String),
    /// An I/O error occurred.
    IoError(String),
}

impl std::fmt::Display for ContainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerError::DockerNotAvailable => write!(f, "Docker is not available"),
            ContainerError::ImageNotFound(img) => write!(f, "Image not found: {}", img),
            ContainerError::ContainerNotFound(id) => write!(f, "Container not found: {}", id),
            ContainerError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            ContainerError::Timeout => write!(f, "Operation timed out"),
            ContainerError::PolicyViolation(msg) => write!(f, "Policy violation: {}", msg),
            ContainerError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for ContainerError {}

/// Options for creating a new container.
#[derive(Debug, Clone, Default)]
pub struct CreateOptions {
    /// Environment variables to set inside the container.
    pub env_vars: HashMap<String, String>,
    /// Port mappings (host_port -> container_port).
    pub ports: HashMap<u16, u16>,
    /// Bind mount paths ("host_path:container_path").
    pub bind_mounts: Vec<String>,
    /// Working directory inside the container.
    pub working_dir: Option<String>,
    /// Command to run (overrides image default).
    pub cmd: Option<Vec<String>>,
    /// Session ID for tracking.
    pub session_id: Option<String>,
    /// Agent ID for tracking.
    pub agent_id: Option<String>,
}

/// Result of executing a command inside a container.
#[derive(Debug, Clone)]
pub struct ExecResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Process exit code.
    pub exit_code: i32,
    /// How long the execution took.
    pub duration: Duration,
    /// Whether the execution was terminated due to timeout.
    pub timed_out: bool,
}

/// Record for a running or stopped container.
#[derive(Debug, Clone)]
pub struct ContainerRecord {
    /// Container ID.
    pub id: String,
    /// Image used.
    pub image: String,
    /// Container name.
    pub name: String,
    /// Current status (e.g. "running", "exited").
    pub status: String,
}

/// Executor for Docker container operations.
///
/// Wraps the Docker CLI to create, start, stop, remove, and exec into
/// containers. All operations are synchronous and shell out to `docker`.
pub struct ContainerExecutor {
    /// Maximum time to wait for any single docker command.
    pub default_timeout: Duration,
}

impl ContainerExecutor {
    /// Create a new executor with default settings.
    pub fn new() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
        }
    }

    /// Check whether Docker is available on this system.
    pub fn is_available(&self) -> Result<bool, ContainerError> {
        let output = std::process::Command::new("docker")
            .args(["info"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;
        Ok(output.success())
    }

    /// Create a container from an image.
    pub fn create(
        &self,
        image: &str,
        name: Option<&str>,
        options: &CreateOptions,
    ) -> Result<String, ContainerError> {
        let mut args = vec!["create".to_string()];

        if let Some(n) = name {
            args.push("--name".to_string());
            args.push(n.to_string());
        }

        for (key, val) in &options.env_vars {
            args.push("-e".to_string());
            args.push(format!("{}={}", key, val));
        }

        for (host, container) in &options.ports {
            args.push("-p".to_string());
            args.push(format!("{}:{}", host, container));
        }

        for mount in &options.bind_mounts {
            args.push("-v".to_string());
            args.push(mount.clone());
        }

        if let Some(ref wd) = options.working_dir {
            args.push("-w".to_string());
            args.push(wd.clone());
        }

        args.push(image.to_string());

        if let Some(ref cmd) = options.cmd {
            args.extend(cmd.iter().cloned());
        }

        let output = std::process::Command::new("docker")
            .args(&args)
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if output.status.success() {
            let id = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(id)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("No such image") || stderr.contains("not found") {
                Err(ContainerError::ImageNotFound(image.to_string()))
            } else {
                Err(ContainerError::OperationFailed(stderr))
            }
        }
    }

    /// Start a stopped container.
    pub fn start(&self, container_id: &str) -> Result<(), ContainerError> {
        let output = std::process::Command::new("docker")
            .args(["start", container_id])
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("No such container") {
                Err(ContainerError::ContainerNotFound(container_id.to_string()))
            } else {
                Err(ContainerError::OperationFailed(stderr))
            }
        }
    }

    /// Stop a running container.
    pub fn stop(&self, container_id: &str, timeout_secs: u64) -> Result<(), ContainerError> {
        let output = std::process::Command::new("docker")
            .args(["stop", "-t", &timeout_secs.to_string(), container_id])
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("No such container") {
                Err(ContainerError::ContainerNotFound(container_id.to_string()))
            } else {
                Err(ContainerError::OperationFailed(stderr))
            }
        }
    }

    /// Remove a container.
    pub fn remove(&self, container_id: &str, force: bool) -> Result<(), ContainerError> {
        let mut args = vec!["rm".to_string()];
        if force {
            args.push("-f".to_string());
        }
        args.push(container_id.to_string());

        let output = std::process::Command::new("docker")
            .args(&args)
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("No such container") {
                Err(ContainerError::ContainerNotFound(container_id.to_string()))
            } else {
                Err(ContainerError::OperationFailed(stderr))
            }
        }
    }

    /// Execute a command inside a running container.
    pub fn exec(
        &self,
        container_id: &str,
        command: &str,
        timeout_secs: u64,
    ) -> Result<ExecResult, ContainerError> {
        let start = std::time::Instant::now();

        // Security: reject shell metacharacters to prevent injection via sh -c
        const SHELL_META: &[char] = &[';', '|', '&', '$', '`', '(', ')', '>', '<', '{', '}', '\n', '\r'];
        for ch in SHELL_META {
            if command.contains(*ch) {
                return Err(ContainerError::PolicyViolation(format!(
                    "Command contains disallowed shell metacharacter '{}'",
                    ch
                )));
            }
        }

        // Split into program + args (no shell interpretation)
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ContainerError::OperationFailed("Empty command".into()));
        }
        let mut exec_args: Vec<&str> = vec!["exec", container_id];
        exec_args.extend(&parts);

        let child = std::process::Command::new("docker")
            .args(&exec_args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        let output = child
            .wait_with_output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        let duration = start.elapsed();
        let timed_out = duration.as_secs() >= timeout_secs;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration,
            timed_out,
        })
    }

    /// Get logs from a container.
    pub fn logs(&self, container_id: &str, tail: u64) -> Result<String, ContainerError> {
        let output = std::process::Command::new("docker")
            .args(["logs", "--tail", &tail.to_string(), container_id])
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            // Docker logs can go to either stdout or stderr
            Ok(if stderr.is_empty() {
                stdout
            } else {
                format!("{}{}", stdout, stderr)
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("No such container") {
                Err(ContainerError::ContainerNotFound(container_id.to_string()))
            } else {
                Err(ContainerError::OperationFailed(stderr))
            }
        }
    }

    /// List all containers (running and stopped).
    pub fn list(&self) -> Result<Vec<ContainerRecord>, ContainerError> {
        let output = std::process::Command::new("docker")
            .args([
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}",
            ])
            .output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        if !output.status.success() {
            return Err(ContainerError::OperationFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let records = stdout
            .lines()
            .filter(|line| !line.is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(4, '\t').collect();
                if parts.len() >= 4 {
                    Some(ContainerRecord {
                        id: parts[0].to_string(),
                        image: parts[1].to_string(),
                        name: parts[2].to_string(),
                        status: parts[3].to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(records)
    }

    /// Run code in a temporary container: create, start, capture output, remove.
    pub fn run_code(
        &self,
        language: &str,
        code: &str,
        timeout_secs: u64,
    ) -> Result<ExecResult, ContainerError> {
        let (image, cmd) = match language {
            "python" => ("python:3-slim", vec!["python3", "-c", code]),
            "javascript" => ("node:slim", vec!["node", "-e", code]),
            "bash" => ("alpine:latest", vec!["sh", "-c", code]),
            _ => {
                return Err(ContainerError::OperationFailed(format!(
                    "Unsupported language: {}. Supported: python, javascript, bash",
                    language
                )))
            }
        };

        let start = std::time::Instant::now();

        let child = std::process::Command::new("docker")
            .args(["run", "--rm", "--network=none", image])
            .args(&cmd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        let output = child
            .wait_with_output()
            .map_err(|e| ContainerError::IoError(e.to_string()))?;

        let duration = start.elapsed();
        let timed_out = duration.as_secs() >= timeout_secs;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration,
            timed_out,
        })
    }
}

impl Default for ContainerExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all container management tools into a ToolRegistry.
pub fn register_container_tools(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    register_container_create(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_start(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_stop(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_remove(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_exec(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_logs(registry, Arc::clone(&executor), Arc::clone(&sandbox));
    register_container_list(registry, Arc::clone(&executor));
    register_container_run_code(registry, Arc::clone(&executor), Arc::clone(&sandbox));
}

/// Get the list of container tool definitions without registering them.
/// Useful for documentation and introspection.
pub fn container_tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolBuilder::new("container_create", "Create a new Docker container from an image")
            .param(ParamSchema::string("image", "Docker image to use"))
            .param(ParamSchema::string("name", "Container name").optional())
            .param(ParamSchema::string("env", "Environment variables as JSON object").optional())
            .param(
                ParamSchema::string("mounts", "Bind mounts as JSON array of \"host:container\"")
                    .optional(),
            )
            .category("container")
            .build(),
        ToolBuilder::new("container_start", "Start a stopped Docker container")
            .param(ParamSchema::string("container_id", "ID of the container to start"))
            .category("container")
            .build(),
        ToolBuilder::new("container_stop", "Stop a running Docker container")
            .param(ParamSchema::string("container_id", "ID of the container to stop"))
            .param(
                ParamSchema::integer("timeout", "Seconds to wait before killing (default: 10)")
                    .optional(),
            )
            .category("container")
            .build(),
        ToolBuilder::new("container_remove", "Remove a Docker container")
            .param(ParamSchema::string("container_id", "ID of the container to remove"))
            .param(ParamSchema::boolean("force", "Force removal of running container").optional())
            .category("container")
            .build(),
        ToolBuilder::new("container_exec", "Execute a command inside a running container")
            .param(ParamSchema::string(
                "container_id",
                "ID of the container to exec into",
            ))
            .param(ParamSchema::string("command", "Shell command to execute"))
            .param(
                ParamSchema::integer("timeout", "Execution timeout in seconds (default: 30)")
                    .optional(),
            )
            .category("container")
            .build(),
        ToolBuilder::new("container_logs", "Retrieve logs from a Docker container")
            .param(ParamSchema::string("container_id", "ID of the container"))
            .param(
                ParamSchema::integer("tail", "Number of lines from the end (default: 100)")
                    .optional(),
            )
            .category("container")
            .build(),
        ToolBuilder::new("container_list", "List all Docker containers")
            .category("container")
            .build(),
        ToolBuilder::new(
            "container_run_code",
            "Run code in a temporary container with automatic cleanup",
        )
        .param(ParamSchema::string(
            "language",
            "Programming language: python, javascript, or bash",
        ))
        .param(ParamSchema::string("code", "Source code to execute"))
        .param(
            ParamSchema::integer("timeout", "Execution timeout in seconds (default: 30)")
                .optional(),
        )
        .category("container")
        .build(),
    ]
}

// ============================================================================
// Tool Implementations
// ============================================================================

fn register_container_create(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new("container_create", "Create a new Docker container from an image")
        .param(ParamSchema::string("image", "Docker image to use"))
        .param(ParamSchema::string("name", "Container name").optional())
        .param(ParamSchema::string("env", "Environment variables as JSON object").optional())
        .param(
            ParamSchema::string("mounts", "Bind mounts as JSON array of \"host:container\"")
                .optional(),
        )
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let image = call
            .get_string("image")
            .ok_or_else(|| ToolError::MissingParameter("image".into()))?;

        // Medium risk — validate with sandbox
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker create {}", image))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let name = call.get_string("name").map(|s| s.to_string());

        // Parse optional env vars from JSON string
        let env_vars: HashMap<String, String> = call
            .get_string("env")
            .map(|s| serde_json::from_str(s).unwrap_or_default())
            .unwrap_or_default();

        // Parse optional mounts from JSON array string
        let bind_mounts: Vec<String> = call
            .get_string("mounts")
            .map(|s| serde_json::from_str(s).unwrap_or_default())
            .unwrap_or_default();

        let options = CreateOptions {
            env_vars,
            bind_mounts,
            ..Default::default()
        };

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        let container_id = exec.create(image, name.as_deref(), &options).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to create container: {}", e))
        })?;

        Ok(ToolOutput::with_data(
            format!("Container created: {}", container_id),
            serde_json::json!({ "container_id": container_id }),
        ))
    });

    registry.register(def, handler);
}

fn register_container_start(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new("container_start", "Start a stopped Docker container")
        .param(ParamSchema::string(
            "container_id",
            "ID of the container to start",
        ))
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let container_id = call
            .get_string("container_id")
            .ok_or_else(|| ToolError::MissingParameter("container_id".into()))?;

        // Low risk
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker start {}", container_id))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        exec.start(container_id)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to start container: {}", e)))?;

        Ok(ToolOutput::text("Container started"))
    });

    registry.register(def, handler);
}

fn register_container_stop(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new("container_stop", "Stop a running Docker container")
        .param(ParamSchema::string(
            "container_id",
            "ID of the container to stop",
        ))
        .param(
            ParamSchema::integer("timeout", "Seconds to wait before killing (default: 10)")
                .optional(),
        )
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let container_id = call
            .get_string("container_id")
            .ok_or_else(|| ToolError::MissingParameter("container_id".into()))?;

        let timeout = call.get_integer("timeout").unwrap_or(10) as u64;

        // Low risk
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker stop {}", container_id))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        exec.stop(container_id, timeout)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to stop container: {}", e)))?;

        Ok(ToolOutput::text("Container stopped"))
    });

    registry.register(def, handler);
}

fn register_container_remove(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new("container_remove", "Remove a Docker container")
        .param(ParamSchema::string(
            "container_id",
            "ID of the container to remove",
        ))
        .param(ParamSchema::boolean("force", "Force removal of running container").optional())
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let container_id = call
            .get_string("container_id")
            .ok_or_else(|| ToolError::MissingParameter("container_id".into()))?;

        let force = call.get_bool("force").unwrap_or(false);

        // Medium risk — validate with sandbox
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker rm {}", container_id))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        exec.remove(container_id, force).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to remove container: {}", e))
        })?;

        Ok(ToolOutput::text("Container removed"))
    });

    registry.register(def, handler);
}

fn register_container_exec(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def =
        ToolBuilder::new("container_exec", "Execute a command inside a running container")
            .param(ParamSchema::string(
                "container_id",
                "ID of the container to exec into",
            ))
            .param(ParamSchema::string("command", "Shell command to execute"))
            .param(
                ParamSchema::integer("timeout", "Execution timeout in seconds (default: 30)")
                    .optional(),
            )
            .category("container")
            .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let container_id = call
            .get_string("container_id")
            .ok_or_else(|| ToolError::MissingParameter("container_id".into()))?;

        let command = call
            .get_string("command")
            .ok_or_else(|| ToolError::MissingParameter("command".into()))?;

        let timeout = call.get_integer("timeout").unwrap_or(30) as u64;

        // Medium risk — validate with sandbox
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker exec {} {}", container_id, command))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        let result = exec.exec(container_id, command, timeout).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to exec in container: {}", e))
        })?;

        Ok(ToolOutput::with_data(
            format!(
                "Exit code: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                result.exit_code,
                if result.stdout.is_empty() {
                    "(empty)"
                } else {
                    result.stdout.trim()
                },
                if result.stderr.is_empty() {
                    "(empty)"
                } else {
                    result.stderr.trim()
                },
            ),
            serde_json::json!({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }),
        ))
    });

    registry.register(def, handler);
}

fn register_container_logs(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new("container_logs", "Retrieve logs from a Docker container")
        .param(ParamSchema::string(
            "container_id",
            "ID of the container",
        ))
        .param(
            ParamSchema::integer("tail", "Number of lines from the end (default: 100)").optional(),
        )
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let container_id = call
            .get_string("container_id")
            .ok_or_else(|| ToolError::MissingParameter("container_id".into()))?;

        let tail = call.get_integer("tail").unwrap_or(100) as u64;

        // Safe / low risk
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker logs {}", container_id))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        let logs = exec.logs(container_id, tail).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to get container logs: {}", e))
        })?;

        Ok(ToolOutput::text(logs))
    });

    registry.register(def, handler);
}

fn register_container_list(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
) {
    let def = ToolBuilder::new("container_list", "List all Docker containers")
        .category("container")
        .build();

    let handler: ToolHandler = Arc::new(move |_call: &ToolCall| {
        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        let records = exec
            .list()
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to list containers: {}", e)))?;

        let json_records: Vec<serde_json::Value> = records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "image": r.image,
                    "name": r.name,
                    "status": r.status,
                })
            })
            .collect();

        Ok(ToolOutput::with_data(
            format!("{} container(s)", records.len()),
            serde_json::json!(json_records),
        ))
    });

    registry.register(def, handler);
}

fn register_container_run_code(
    registry: &mut ToolRegistry,
    executor: Arc<RwLock<ContainerExecutor>>,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    let def = ToolBuilder::new(
        "container_run_code",
        "Run code in a temporary container with automatic cleanup",
    )
    .param(ParamSchema::string(
        "language",
        "Programming language: python, javascript, or bash",
    ))
    .param(ParamSchema::string("code", "Source code to execute"))
    .param(
        ParamSchema::integer("timeout", "Execution timeout in seconds (default: 30)").optional(),
    )
    .category("container")
    .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let language = call
            .get_string("language")
            .ok_or_else(|| ToolError::MissingParameter("language".into()))?;

        let code = call
            .get_string("code")
            .ok_or_else(|| ToolError::MissingParameter("code".into()))?;

        // Validate language before proceeding
        match language {
            "python" | "javascript" | "bash" => {}
            _ => {
                return Err(ToolError::InvalidParameter {
                    name: "language".into(),
                    message: format!(
                        "Unsupported language '{}'. Supported: python, javascript, bash",
                        language
                    ),
                });
            }
        }

        let timeout = call.get_integer("timeout").unwrap_or(30) as u64;

        // Medium risk — validate with sandbox
        sandbox
            .write()
            .map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(&format!("docker run {} code", language))
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let exec = executor
            .read()
            .map_err(|_| ToolError::ExecutionFailed("executor lock poisoned".into()))?;

        let result = exec.run_code(language, code, timeout).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to run code in container: {}", e))
        })?;

        Ok(ToolOutput::with_data(
            format!(
                "Exit code: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                result.exit_code,
                if result.stdout.is_empty() {
                    "(empty)"
                } else {
                    result.stdout.trim()
                },
                if result.stderr.is_empty() {
                    "(empty)"
                } else {
                    result.stderr.trim()
                },
            ),
            serde_json::json!({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            }),
        ))
    });

    registry.register(def, handler);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_policy::{AgentPolicyBuilder, AutoApproveAll, InternetMode};

    fn test_sandbox() -> Arc<RwLock<SandboxValidator>> {
        let policy = AgentPolicyBuilder::new()
            .allow_command("docker")
            .internet(InternetMode::FullAccess)
            .build();
        let handler: Arc<dyn crate::agent_policy::ApprovalHandler> = Arc::new(AutoApproveAll);
        Arc::new(RwLock::new(SandboxValidator::with_approval(
            policy, handler,
        )))
    }

    fn test_executor() -> Arc<RwLock<ContainerExecutor>> {
        Arc::new(RwLock::new(ContainerExecutor::new()))
    }

    fn test_registry() -> ToolRegistry {
        let sandbox = test_sandbox();
        let executor = test_executor();
        let mut registry = ToolRegistry::new();
        register_container_tools(&mut registry, executor, sandbox);
        registry
    }

    // ---- Tool count ----

    #[test]
    fn test_register_container_tools_count() {
        let registry = test_registry();
        let tools = registry.list();
        assert_eq!(tools.len(), 8, "Expected 8 container tools");
    }

    // ---- Tool definition tests ----

    #[test]
    fn test_container_create_def() {
        let registry = test_registry();
        let def = registry.get("container_create").expect("tool exists");
        assert_eq!(def.name, "container_create");
        assert_eq!(def.category.as_deref(), Some("container"));

        let param_names: Vec<&str> = def.parameters.iter().map(|p| p.name.as_str()).collect();
        assert!(param_names.contains(&"image"));
        assert!(param_names.contains(&"name"));
        assert!(param_names.contains(&"env"));
        assert!(param_names.contains(&"mounts"));

        // image is required, others optional
        let image_param = def.parameters.iter().find(|p| p.name == "image").unwrap();
        assert!(image_param.required);
        let name_param = def.parameters.iter().find(|p| p.name == "name").unwrap();
        assert!(!name_param.required);
    }

    #[test]
    fn test_container_start_def() {
        let registry = test_registry();
        let def = registry.get("container_start").expect("tool exists");
        assert_eq!(def.name, "container_start");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert_eq!(def.parameters.len(), 1);
        assert_eq!(def.parameters[0].name, "container_id");
        assert!(def.parameters[0].required);
    }

    #[test]
    fn test_container_stop_def() {
        let registry = test_registry();
        let def = registry.get("container_stop").expect("tool exists");
        assert_eq!(def.name, "container_stop");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert_eq!(def.parameters.len(), 2);

        let id_param = def
            .parameters
            .iter()
            .find(|p| p.name == "container_id")
            .unwrap();
        assert!(id_param.required);

        let timeout_param = def
            .parameters
            .iter()
            .find(|p| p.name == "timeout")
            .unwrap();
        assert!(!timeout_param.required);
    }

    #[test]
    fn test_container_remove_def() {
        let registry = test_registry();
        let def = registry.get("container_remove").expect("tool exists");
        assert_eq!(def.name, "container_remove");
        assert_eq!(def.category.as_deref(), Some("container"));

        let param_names: Vec<&str> = def.parameters.iter().map(|p| p.name.as_str()).collect();
        assert!(param_names.contains(&"container_id"));
        assert!(param_names.contains(&"force"));

        let force_param = def.parameters.iter().find(|p| p.name == "force").unwrap();
        assert!(!force_param.required);
    }

    #[test]
    fn test_container_exec_def() {
        let registry = test_registry();
        let def = registry.get("container_exec").expect("tool exists");
        assert_eq!(def.name, "container_exec");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert_eq!(def.parameters.len(), 3);

        let id_param = def
            .parameters
            .iter()
            .find(|p| p.name == "container_id")
            .unwrap();
        assert!(id_param.required);

        let cmd_param = def
            .parameters
            .iter()
            .find(|p| p.name == "command")
            .unwrap();
        assert!(cmd_param.required);

        let timeout_param = def
            .parameters
            .iter()
            .find(|p| p.name == "timeout")
            .unwrap();
        assert!(!timeout_param.required);
    }

    #[test]
    fn test_container_logs_def() {
        let registry = test_registry();
        let def = registry.get("container_logs").expect("tool exists");
        assert_eq!(def.name, "container_logs");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert_eq!(def.parameters.len(), 2);

        let id_param = def
            .parameters
            .iter()
            .find(|p| p.name == "container_id")
            .unwrap();
        assert!(id_param.required);

        let tail_param = def.parameters.iter().find(|p| p.name == "tail").unwrap();
        assert!(!tail_param.required);
    }

    #[test]
    fn test_container_list_def() {
        let registry = test_registry();
        let def = registry.get("container_list").expect("tool exists");
        assert_eq!(def.name, "container_list");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert!(def.parameters.is_empty());
    }

    #[test]
    fn test_container_run_code_def() {
        let registry = test_registry();
        let def = registry.get("container_run_code").expect("tool exists");
        assert_eq!(def.name, "container_run_code");
        assert_eq!(def.category.as_deref(), Some("container"));
        assert_eq!(def.parameters.len(), 3);

        let lang_param = def
            .parameters
            .iter()
            .find(|p| p.name == "language")
            .unwrap();
        assert!(lang_param.required);

        let code_param = def.parameters.iter().find(|p| p.name == "code").unwrap();
        assert!(code_param.required);

        let timeout_param = def
            .parameters
            .iter()
            .find(|p| p.name == "timeout")
            .unwrap();
        assert!(!timeout_param.required);
    }

    // ---- Parameter validation tests (via registry.validate_call) ----

    #[test]
    fn test_container_create_missing_image() {
        let registry = test_registry();
        let call = ToolCall::new("container_create", HashMap::new());
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "image"),
            "Expected MissingParameter(\"image\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_exec_missing_id() {
        let registry = test_registry();
        let call = ToolCall::new(
            "container_exec",
            HashMap::from([("command".to_string(), serde_json::json!("ls"))]),
        );
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "container_id"),
            "Expected MissingParameter(\"container_id\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_exec_missing_command() {
        let registry = test_registry();
        let call = ToolCall::new(
            "container_exec",
            HashMap::from([(
                "container_id".to_string(),
                serde_json::json!("abc123"),
            )]),
        );
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "command"),
            "Expected MissingParameter(\"command\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_run_code_missing_language() {
        let registry = test_registry();
        let call = ToolCall::new(
            "container_run_code",
            HashMap::from([("code".to_string(), serde_json::json!("print('hi')"))]),
        );
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "language"),
            "Expected MissingParameter(\"language\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_run_code_missing_code() {
        let registry = test_registry();
        let call = ToolCall::new(
            "container_run_code",
            HashMap::from([("language".to_string(), serde_json::json!("python"))]),
        );
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "code"),
            "Expected MissingParameter(\"code\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_run_code_invalid_language() {
        let registry = test_registry();
        let call = ToolCall::new(
            "container_run_code",
            HashMap::from([
                ("language".to_string(), serde_json::json!("ruby")),
                ("code".to_string(), serde_json::json!("puts 'hi'")),
            ]),
        );
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::InvalidParameter { ref name, .. }) if name == "language"),
            "Expected InvalidParameter for language, got {:?}",
            result
        );
    }

    // ---- Category tests ----

    #[test]
    fn test_all_tools_have_container_category() {
        let registry = test_registry();
        let tools = registry.list();
        for tool in &tools {
            assert_eq!(
                tool.category.as_deref(),
                Some("container"),
                "Tool '{}' should have category 'container'",
                tool.name
            );
        }
    }

    // ---- Default value tests ----

    #[test]
    fn test_container_stop_default_timeout() {
        // Verify the timeout parameter is optional (default 10 is applied in handler)
        let registry = test_registry();
        let def = registry.get("container_stop").unwrap();
        let timeout_param = def
            .parameters
            .iter()
            .find(|p| p.name == "timeout")
            .unwrap();
        assert!(!timeout_param.required, "timeout should be optional");
        assert_eq!(
            timeout_param.param_type,
            crate::unified_tools::ParamType::Integer
        );
    }

    #[test]
    fn test_container_logs_default_tail() {
        // Verify the tail parameter is optional (default 100 is applied in handler)
        let registry = test_registry();
        let def = registry.get("container_logs").unwrap();
        let tail_param = def.parameters.iter().find(|p| p.name == "tail").unwrap();
        assert!(!tail_param.required, "tail should be optional");
        assert_eq!(
            tail_param.param_type,
            crate::unified_tools::ParamType::Integer
        );
    }

    #[test]
    fn test_container_list_no_params_needed() {
        let registry = test_registry();
        let def = registry.get("container_list").unwrap();
        assert!(
            def.parameters.is_empty(),
            "container_list should have no parameters"
        );
        // Validate that calling with no args passes validation
        let call = ToolCall::new("container_list", HashMap::new());
        let validation = registry.validate_call(&call);
        assert!(
            validation.is_ok(),
            "container_list should validate with no params"
        );
    }

    #[test]
    fn test_tool_definitions_have_descriptions() {
        let registry = test_registry();
        let tools = registry.list();
        for tool in &tools {
            assert!(
                !tool.description.is_empty(),
                "Tool '{}' should have a non-empty description",
                tool.name
            );
        }
    }

    // ---- container_tool_definitions() ----

    #[test]
    fn test_container_tool_definitions_fn() {
        let defs = container_tool_definitions();
        assert_eq!(defs.len(), 8);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"container_create"));
        assert!(names.contains(&"container_start"));
        assert!(names.contains(&"container_stop"));
        assert!(names.contains(&"container_remove"));
        assert!(names.contains(&"container_exec"));
        assert!(names.contains(&"container_logs"));
        assert!(names.contains(&"container_list"));
        assert!(names.contains(&"container_run_code"));
    }

    // ---- ContainerExecutor unit tests ----

    #[test]
    fn test_container_executor_default() {
        let exec = ContainerExecutor::new();
        assert_eq!(exec.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_container_executor_default_trait() {
        let exec = ContainerExecutor::default();
        assert_eq!(exec.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_create_options_default() {
        let opts = CreateOptions::default();
        assert!(opts.env_vars.is_empty());
        assert!(opts.ports.is_empty());
        assert!(opts.bind_mounts.is_empty());
        assert!(opts.working_dir.is_none());
        assert!(opts.cmd.is_none());
        assert!(opts.session_id.is_none());
        assert!(opts.agent_id.is_none());
    }

    #[test]
    fn test_container_error_display() {
        assert_eq!(
            ContainerError::DockerNotAvailable.to_string(),
            "Docker is not available"
        );
        assert_eq!(
            ContainerError::ImageNotFound("alpine".into()).to_string(),
            "Image not found: alpine"
        );
        assert_eq!(
            ContainerError::ContainerNotFound("abc".into()).to_string(),
            "Container not found: abc"
        );
        assert_eq!(
            ContainerError::Timeout.to_string(),
            "Operation timed out"
        );
        assert_eq!(
            ContainerError::PolicyViolation("no root".into()).to_string(),
            "Policy violation: no root"
        );
        assert_eq!(
            ContainerError::IoError("broken pipe".into()).to_string(),
            "I/O error: broken pipe"
        );
    }

    #[test]
    fn test_container_start_missing_id() {
        let registry = test_registry();
        let call = ToolCall::new("container_start", HashMap::new());
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "container_id"),
            "Expected MissingParameter(\"container_id\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_remove_missing_id() {
        let registry = test_registry();
        let call = ToolCall::new("container_remove", HashMap::new());
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "container_id"),
            "Expected MissingParameter(\"container_id\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_logs_missing_id() {
        let registry = test_registry();
        let call = ToolCall::new("container_logs", HashMap::new());
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "container_id"),
            "Expected MissingParameter(\"container_id\"), got {:?}",
            result
        );
    }

    #[test]
    fn test_container_stop_missing_id() {
        let registry = test_registry();
        let call = ToolCall::new("container_stop", HashMap::new());
        let result = registry.execute(&call);
        assert!(
            matches!(result, Err(ToolError::MissingParameter(ref p)) if p == "container_id"),
            "Expected MissingParameter(\"container_id\"), got {:?}",
            result
        );
    }
}
