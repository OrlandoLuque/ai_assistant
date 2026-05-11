// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Node lifecycle management: start, stop, and status checks.
//!
//! Manages the ai_assistant server process, supporting both foreground and
//! background (daemon) modes.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// Information about a running (or stopped) node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Whether the node process is currently running.
    pub running: bool,
    /// Process ID (0 if not running).
    pub pid: u32,
    /// Port the server is listening on (0 if not running).
    pub port: u16,
    /// Uptime in seconds (0 if not running).
    pub uptime_secs: u64,
    /// Health endpoint result (e.g. "ok", "degraded", "unreachable").
    pub health: String,
}

/// Default port for the ai_assistant server.
const DEFAULT_PORT: u16 = 3000;

/// PID file location within the config directory.
const PID_FILENAME: &str = "ai_assistant.pid";

/// Start the ai_assistant server node.
///
/// If `foreground` is true, the process replaces the current one (exec).
/// If false, it spawns in the background and writes a PID file.
pub fn start_node(config_path: &Path, foreground: bool) -> Result<String, String> {
    // Determine the binary path (same directory as the current executable)
    let current_exe =
        std::env::current_exe().map_err(|e| format!("Cannot determine executable path: {}", e))?;
    let exe_dir = current_exe.parent().unwrap_or(Path::new("."));

    let server_bin = if cfg!(target_os = "windows") {
        exe_dir.join("ai_assistant_server.exe")
    } else {
        exe_dir.join("ai_assistant_server")
    };

    if !server_bin.exists() {
        return Err(format!(
            "Server binary not found at {}. Build it with: cargo build --bin ai_assistant_server --features full",
            server_bin.display()
        ));
    }

    if foreground {
        // Run in foreground — blocks until the server exits
        let status = Command::new(&server_bin)
            .args(["--config", &config_path.display().to_string()])
            .status()
            .map_err(|e| format!("Failed to start server: {}", e))?;

        if status.success() {
            Ok("Server exited normally".to_string())
        } else {
            Err(format!("Server exited with status: {}", status))
        }
    } else {
        // Spawn in background
        let child = Command::new(&server_bin)
            .args(["--config", &config_path.display().to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to spawn server: {}", e))?;

        let pid = child.id();

        // Write PID file next to config
        let pid_path = config_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(PID_FILENAME);
        let _ = std::fs::write(&pid_path, pid.to_string());

        Ok(format!("Server started in background (PID: {})", pid))
    }
}

/// Stop the ai_assistant server node.
///
/// Reads the PID file, sends a termination signal, and cleans up.
pub fn stop_node() -> Result<(), String> {
    // Try to find PID from the default locations
    let pid = find_server_pid()?;

    // Send termination signal
    kill_process(pid)?;

    // Clean up PID file
    let config_dir = default_config_dir();
    let pid_path = config_dir.join(PID_FILENAME);
    let _ = std::fs::remove_file(&pid_path);

    Ok(())
}

/// Check the status of the ai_assistant server node.
pub fn node_status() -> Result<NodeInfo, String> {
    // Try health endpoint first
    let port = DEFAULT_PORT;
    let health_url = format!("http://127.0.0.1:{}/health", port);

    let start = Instant::now();

    match ureq::get(&health_url)
        .timeout(Duration::from_secs(2))
        .call()
    {
        Ok(resp) => {
            let status_code = resp.status();
            // Drain the response body so the connection can be released.
            let _ = resp.into_string();
            let response_ms = start.elapsed().as_millis() as u64;
            let health = if status_code == 200 {
                format!("ok ({}ms)", response_ms)
            } else {
                format!("HTTP {}", status_code)
            };

            // Try to get PID
            let pid = find_server_pid().unwrap_or(0);

            Ok(NodeInfo {
                running: true,
                pid,
                port,
                uptime_secs: 0, // Would need process start time to calculate
                health,
            })
        }
        Err(_) => {
            // Health endpoint not reachable — check if process is running via PID
            let pid = find_server_pid().unwrap_or(0);
            let running = pid > 0 && is_process_running(pid);

            Ok(NodeInfo {
                running,
                pid,
                port,
                uptime_secs: 0,
                health: if running {
                    "starting".to_string()
                } else {
                    "unreachable".to_string()
                },
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn default_config_dir() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("ai_assistant")
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("XDG_CONFIG_HOME")
            .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.config", h)))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("ai_assistant")
    }
}

fn find_server_pid() -> Result<u32, String> {
    let config_dir = default_config_dir();
    let pid_path = config_dir.join(PID_FILENAME);

    if pid_path.exists() {
        let content = std::fs::read_to_string(&pid_path)
            .map_err(|e| format!("Cannot read PID file: {}", e))?;
        content
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("Invalid PID in {}: {}", pid_path.display(), e))
    } else {
        Err("No PID file found. Server may not be running.".to_string())
    }
}

fn kill_process(pid: u32) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output()
            .map_err(|e| format!("Failed to kill process {}: {}", pid, e))?;
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        Command::new("kill")
            .arg(pid.to_string())
            .output()
            .map_err(|e| format!("Failed to kill process {}: {}", pid, e))?;
        Ok(())
    }
}

fn is_process_running(pid: u32) -> bool {
    #[cfg(target_os = "windows")]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid), "/NH"])
            .output()
            .map(|o| {
                let s = String::from_utf8_lossy(&o.stdout);
                s.contains(&pid.to_string())
            })
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "windows"))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_info_default() {
        let info = NodeInfo {
            running: false,
            pid: 0,
            port: 3000,
            uptime_secs: 0,
            health: "unreachable".to_string(),
        };
        assert!(!info.running);
        assert_eq!(info.port, 3000);
    }

    #[test]
    fn test_default_config_dir_is_not_empty() {
        let dir = default_config_dir();
        // On any platform, this should produce a non-empty path
        assert!(!dir.as_os_str().is_empty());
    }
}
