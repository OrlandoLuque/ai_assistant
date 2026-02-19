//! OS tools — filesystem, shell, git, and network tools for the ToolRegistry
//!
//! Provides system-level tools that autonomous agents can use, with each
//! tool call validated through the sandbox before execution.

use crate::agent_sandbox::SandboxValidator;
use crate::unified_tools::{
    ParamSchema, ToolBuilder, ToolCall, ToolDef, ToolError, ToolHandler, ToolOutput, ToolRegistry,
};
use std::fs;
use std::io::Read as IoRead;
use std::sync::{Arc, RwLock};

// ============================================================================
// Registration
// ============================================================================

/// Register all OS tools into a ToolRegistry with sandbox validation.
pub fn register_os_tools(
    registry: &mut ToolRegistry,
    sandbox: Arc<RwLock<SandboxValidator>>,
) {
    // Filesystem
    register_read_file(registry, Arc::clone(&sandbox));
    register_write_file(registry, Arc::clone(&sandbox));
    register_list_dir(registry, Arc::clone(&sandbox));
    register_create_dir(registry, Arc::clone(&sandbox));
    register_delete_file(registry, Arc::clone(&sandbox));
    register_file_info(registry, Arc::clone(&sandbox));

    // Shell
    register_run_command(registry, Arc::clone(&sandbox));

    // Git
    register_git_status(registry, Arc::clone(&sandbox));
    register_git_diff(registry, Arc::clone(&sandbox));
    register_git_log(registry, Arc::clone(&sandbox));

    // Network
    register_http_get(registry, Arc::clone(&sandbox));
}

// ============================================================================
// Filesystem Tools
// ============================================================================

fn register_read_file(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("read_file", "Read the contents of a file")
        .param(ParamSchema::string("path", "Absolute path to the file"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_file_read(path)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let content = fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read {}: {}", path, e)))?;
        Ok(ToolOutput::text(content))
    });

    registry.register(def, handler);
}

fn register_write_file(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("write_file", "Write content to a file")
        .param(ParamSchema::string("path", "Absolute path to the file"))
        .param(ParamSchema::string("content", "Content to write"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;
        let content = call.arguments.get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("content".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_file_write(path)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        fs::write(path, content)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to write {}: {}", path, e)))?;
        Ok(ToolOutput::text(format!("Written {} bytes to {}", content.len(), path)))
    });

    registry.register(def, handler);
}

fn register_list_dir(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("list_dir", "List contents of a directory")
        .param(ParamSchema::string("path", "Absolute path to the directory"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_file_read(path)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let entries = fs::read_dir(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read dir {}: {}", path, e)))?;

        let mut lines = Vec::new();
        for entry in entries.flatten() {
            let meta = entry.metadata().ok();
            let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let name = entry.file_name().to_string_lossy().to_string();
            if is_dir {
                lines.push(format!("  [DIR] {}/", name));
            } else {
                lines.push(format!("  {:>8} {}", format_size(size), name));
            }
        }
        lines.sort();
        Ok(ToolOutput::text(lines.join("\n")))
    });

    registry.register(def, handler);
}

fn register_create_dir(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("create_dir", "Create a directory (and parents)")
        .param(ParamSchema::string("path", "Absolute path for the new directory"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_file_write(path)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        fs::create_dir_all(path)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to create dir {}: {}", path, e)))?;
        Ok(ToolOutput::text(format!("Created directory: {}", path)))
    });

    registry.register(def, handler);
}

fn register_delete_file(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("delete_file", "Delete a file")
        .param(ParamSchema::string("path", "Absolute path to the file to delete"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path_str = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;

        {
            use crate::agent_policy::{ActionDescriptor, ActionType};
            let action = ActionDescriptor::new(ActionType::FileDelete, path_str);
            sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
                .validate(&action)
                .map_err(|e| ToolError::ExecutionFailed(e.message))?;
        }

        fs::remove_file(path_str)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to delete {}: {}", path_str, e)))?;
        Ok(ToolOutput::text(format!("Deleted: {}", path_str)))
    });

    registry.register(def, handler);
}

fn register_file_info(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("file_info", "Get metadata about a file or directory")
        .param(ParamSchema::string("path", "Absolute path"))
        .category("filesystem")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let path_str = call.arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("path".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_file_read(path_str)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let meta = fs::metadata(path_str)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to stat {}: {}", path_str, e)))?;

        let info = format!(
            "Path: {}\nType: {}\nSize: {}\nReadonly: {}",
            path_str,
            if meta.is_dir() { "directory" } else if meta.is_file() { "file" } else { "other" },
            format_size(meta.len()),
            meta.permissions().readonly(),
        );
        Ok(ToolOutput::text(info))
    });

    registry.register(def, handler);
}

// ============================================================================
// Shell Tools
// ============================================================================

fn register_run_command(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("run_command", "Execute a shell command")
        .param(ParamSchema::string("command", "The command to execute"))
        .category("shell")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let cmd = call.arguments.get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("command".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_command(cmd)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        #[cfg(target_os = "windows")]
        let output = std::process::Command::new("cmd")
            .args(["/C", cmd])
            .output()
            .map_err(|e| ToolError::ExecutionFailed(format!("Command failed: {}", e)))?;

        #[cfg(not(target_os = "windows"))]
        let output = std::process::Command::new("sh")
            .args(["-c", cmd])
            .output()
            .map_err(|e| ToolError::ExecutionFailed(format!("Command failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        let result = format!(
            "Exit code: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            exit_code,
            if stdout.is_empty() { "(empty)" } else { stdout.trim() },
            if stderr.is_empty() { "(empty)" } else { stderr.trim() }
        );
        Ok(ToolOutput::text(result))
    });

    registry.register(def, handler);
}

// ============================================================================
// Git Tools
// ============================================================================

fn run_git(args: &[&str], sandbox: &Arc<RwLock<SandboxValidator>>) -> Result<String, ToolError> {
    let cmd = format!("git {}", args.join(" "));
    sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
        .validate_command(&cmd)
        .map_err(|e| ToolError::ExecutionFailed(e.message))?;

    let output = std::process::Command::new("git")
        .args(args)
        .output()
        .map_err(|e| ToolError::ExecutionFailed(format!("git failed: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(stdout)
    } else {
        Err(ToolError::ExecutionFailed(format!(
            "git {} failed: {}",
            args.join(" "),
            if stderr.is_empty() { &stdout } else { &stderr }
        )))
    }
}

fn register_git_status(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("git_status", "Show git working tree status")
        .category("git")
        .build();

    let handler: ToolHandler = Arc::new(move |_call: &ToolCall| {
        let output = run_git(&["status", "--short"], &sandbox)?;
        Ok(ToolOutput::text(if output.is_empty() { "Clean working tree".to_string() } else { output }))
    });

    registry.register(def, handler);
}

fn register_git_diff(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("git_diff", "Show git diff of changes")
        .category("git")
        .build();

    let handler: ToolHandler = Arc::new(move |_call: &ToolCall| {
        let output = run_git(&["diff"], &sandbox)?;
        Ok(ToolOutput::text(if output.is_empty() { "No unstaged changes".to_string() } else { output }))
    });

    registry.register(def, handler);
}

fn register_git_log(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("git_log", "Show recent git commits")
        .category("git")
        .build();

    let handler: ToolHandler = Arc::new(move |_call: &ToolCall| {
        let output = run_git(&["log", "--oneline", "-10"], &sandbox)?;
        Ok(ToolOutput::text(output))
    });

    registry.register(def, handler);
}

// ============================================================================
// Network Tools
// ============================================================================

fn register_http_get(registry: &mut ToolRegistry, sandbox: Arc<RwLock<SandboxValidator>>) {
    let def = ToolBuilder::new("http_get", "Fetch content from a URL via HTTP GET")
        .param(ParamSchema::string("url", "The URL to fetch"))
        .category("network")
        .build();

    let handler: ToolHandler = Arc::new(move |call: &ToolCall| {
        let url = call.arguments.get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter("url".into()))?;

        sandbox.write().map_err(|_| ToolError::ExecutionFailed("sandbox lock poisoned".into()))?
            .validate_url(url)
            .map_err(|e| ToolError::ExecutionFailed(e.message))?;

        let resp = ureq::get(url)
            .call()
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP GET failed: {}", e)))?;

        let status = resp.status();
        let mut body = String::new();
        resp.into_reader()
            .take(1_000_000) // 1MB limit
            .read_to_string(&mut body)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read body: {}", e)))?;

        Ok(ToolOutput::text(format!("Status: {}\n\n{}", status, body)))
    });

    registry.register(def, handler);
}

// ============================================================================
// Helpers
// ============================================================================

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Get the list of tool definitions without registering them.
/// Useful for documentation/introspection.
pub fn os_tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolBuilder::new("read_file", "Read the contents of a file")
            .param(ParamSchema::string("path", "Absolute path to the file"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("write_file", "Write content to a file")
            .param(ParamSchema::string("path", "Absolute path to the file"))
            .param(ParamSchema::string("content", "Content to write"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("list_dir", "List contents of a directory")
            .param(ParamSchema::string("path", "Absolute path to the directory"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("create_dir", "Create a directory (and parents)")
            .param(ParamSchema::string("path", "Absolute path for the new directory"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("delete_file", "Delete a file")
            .param(ParamSchema::string("path", "Absolute path to the file to delete"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("file_info", "Get metadata about a file or directory")
            .param(ParamSchema::string("path", "Absolute path"))
            .category("filesystem")
            .build(),
        ToolBuilder::new("run_command", "Execute a shell command")
            .param(ParamSchema::string("command", "The command to execute"))
            .category("shell")
            .build(),
        ToolBuilder::new("git_status", "Show git working tree status")
            .category("git")
            .build(),
        ToolBuilder::new("git_diff", "Show git diff of changes")
            .category("git")
            .build(),
        ToolBuilder::new("git_log", "Show recent git commits")
            .category("git")
            .build(),
        ToolBuilder::new("http_get", "Fetch content from a URL via HTTP GET")
            .param(ParamSchema::string("url", "The URL to fetch"))
            .category("network")
            .build(),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_policy::{AgentPolicyBuilder, AutoApproveAll, InternetMode};
    use std::collections::HashMap;

    fn test_sandbox() -> Arc<RwLock<SandboxValidator>> {
        let policy = AgentPolicyBuilder::new()
            .allow_path(std::env::temp_dir())
            .allow_command("git")
            .allow_command("echo")
            .internet(InternetMode::FullAccess)
            .build();
        let handler: Arc<dyn crate::agent_policy::ApprovalHandler> = Arc::new(AutoApproveAll);
        Arc::new(RwLock::new(SandboxValidator::with_approval(policy, handler)))
    }

    #[test]
    fn test_register_os_tools() {
        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let tools = registry.list();
        assert!(tools.iter().any(|t| t.name == "read_file"));
        assert!(tools.iter().any(|t| t.name == "write_file"));
        assert!(tools.iter().any(|t| t.name == "list_dir"));
        assert!(tools.iter().any(|t| t.name == "create_dir"));
        assert!(tools.iter().any(|t| t.name == "delete_file"));
        assert!(tools.iter().any(|t| t.name == "file_info"));
        assert!(tools.iter().any(|t| t.name == "run_command"));
        assert!(tools.iter().any(|t| t.name == "git_status"));
        assert!(tools.iter().any(|t| t.name == "git_diff"));
        assert!(tools.iter().any(|t| t.name == "git_log"));
        assert!(tools.iter().any(|t| t.name == "http_get"));
    }

    #[test]
    fn test_read_file_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_read");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("test.txt");
        fs::write(&file_path, "hello world").unwrap();

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("read_file", HashMap::from([
            ("path".to_string(), serde_json::Value::String(file_path.to_string_lossy().to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert_eq!(result.content, "hello world");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_file_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_write");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("output.txt");

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("write_file", HashMap::from([
            ("path".to_string(), serde_json::Value::String(file_path.to_string_lossy().to_string())),
            ("content".to_string(), serde_json::Value::String("test content".to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("12 bytes"));

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_dir_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_list");
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("a.txt"), "").unwrap();
        fs::write(dir.join("b.txt"), "").unwrap();

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("list_dir", HashMap::from([
            ("path".to_string(), serde_json::Value::String(dir.to_string_lossy().to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("a.txt"));
        assert!(result.content.contains("b.txt"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_create_dir_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_mkdir").join("sub").join("dir");
        let _ = fs::remove_dir_all(std::env::temp_dir().join("os_tools_test_mkdir"));

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("create_dir", HashMap::from([
            ("path".to_string(), serde_json::Value::String(dir.to_string_lossy().to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("Created directory"));
        assert!(dir.exists());

        let _ = fs::remove_dir_all(std::env::temp_dir().join("os_tools_test_mkdir"));
    }

    #[test]
    fn test_delete_file_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_delete");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("to_delete.txt");
        fs::write(&file, "delete me").unwrap();

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("delete_file", HashMap::from([
            ("path".to_string(), serde_json::Value::String(file.to_string_lossy().to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("Deleted"));
        assert!(!file.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_file_info_tool() {
        let dir = std::env::temp_dir().join("os_tools_test_info");
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("info.txt"), "12345").unwrap();

        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("file_info", HashMap::from([
            ("path".to_string(), serde_json::Value::String(dir.join("info.txt").to_string_lossy().to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("file"));
        assert!(result.content.contains("5B"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sandbox_denies_unauthorized_path() {
        let policy = AgentPolicyBuilder::new()
            .allow_path(std::env::temp_dir())
            .build();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("read_file", HashMap::from([
            ("path".to_string(), serde_json::Value::String("/etc/passwd".to_string())),
        ]));
        let result = registry.execute(&call);
        assert!(result.is_err());
    }

    #[test]
    fn test_os_tool_definitions() {
        let defs = os_tool_definitions();
        assert_eq!(defs.len(), 11);
        assert!(defs.iter().any(|d| d.name == "read_file"));
        assert!(defs.iter().any(|d| d.name == "run_command"));
        assert!(defs.iter().any(|d| d.name == "http_get"));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(100), "100B");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(2_097_152), "2.0MB");
    }

    #[test]
    fn test_run_command_tool() {
        let sandbox = test_sandbox();
        let mut registry = ToolRegistry::new();
        register_os_tools(&mut registry, sandbox);

        let call = ToolCall::new("run_command", HashMap::from([
            ("command".to_string(), serde_json::Value::String("echo hello_test".to_string())),
        ]));
        let result = registry.execute(&call).unwrap();
        assert!(result.content.contains("hello_test"));
        assert!(result.content.contains("Exit code: 0"));
    }
}
