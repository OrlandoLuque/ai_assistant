// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Docker CLI wrapper for build, compose, and container management.
//!
//! Shells out to the `docker` command rather than using a Docker API client,
//! keeping this module lightweight and always available without feature gates.

use std::process::Command;

/// Status of a single Docker container.
#[derive(Debug, Clone)]
pub struct ContainerStatus {
    /// Container name.
    pub name: String,
    /// Status string (e.g. "Up 2 hours", "Exited (0) 5 minutes ago").
    pub status: String,
    /// Health status (e.g. "healthy", "unhealthy", "none").
    pub health: String,
    /// Published ports (e.g. "0.0.0.0:8080->80/tcp").
    pub ports: String,
    /// Uptime string from docker ps.
    pub uptime: String,
}

/// Check whether the `docker` CLI is available on PATH.
pub fn docker_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `docker build` with the given feature flags.
///
/// Expects a Dockerfile in the project root. The `features` string is passed
/// as a build arg (`--build-arg FEATURES=...`).
pub fn docker_build(features: &str) -> Result<String, String> {
    let output = Command::new("docker")
        .args(["build", "--build-arg", &format!("FEATURES={}", features), "-t", "ai_assistant", "."])
        .output()
        .map_err(|e| format!("Failed to run docker build: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "docker build failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Run `docker compose up -d` with optional profiles.
pub fn docker_compose_up(profiles: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new("docker");
    cmd.args(["compose", "up", "-d"]);

    for profile in profiles {
        cmd.args(["--profile", profile]);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run docker compose: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "docker compose up failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Run `docker compose down`.
pub fn docker_compose_down() -> Result<String, String> {
    let output = Command::new("docker")
        .args(["compose", "down"])
        .output()
        .map_err(|e| format!("Failed to run docker compose: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "docker compose down failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Parse `docker ps` output into structured container statuses.
pub fn docker_status() -> Result<Vec<ContainerStatus>, String> {
    let output = Command::new("docker")
        .args([
            "ps",
            "--format",
            "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.RunningFor}}",
            "--no-trunc",
        ])
        .output()
        .map_err(|e| format!("Failed to run docker ps: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "docker ps failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_docker_ps_output(&stdout))
}

/// Fetch logs from a specific container.
pub fn docker_logs(container: &str, tail: usize) -> Result<String, String> {
    let output = Command::new("docker")
        .args(["logs", "--tail", &tail.to_string(), container])
        .output()
        .map_err(|e| format!("Failed to get logs for {}: {}", container, e))?;

    if output.status.success() {
        // Docker logs go to both stdout and stderr
        let mut result = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            result.push_str(&stderr);
        }
        Ok(result)
    } else {
        Err(format!(
            "docker logs failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse the tab-separated output of `docker ps --format`.
fn parse_docker_ps_output(output: &str) -> Vec<ContainerStatus> {
    let mut containers = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(4, '\t').collect();
        if parts.len() < 2 {
            continue;
        }

        let name = parts[0].to_string();
        let status = parts.get(1).unwrap_or(&"").to_string();
        let ports = parts.get(2).unwrap_or(&"").to_string();
        let uptime = parts.get(3).unwrap_or(&"").to_string();

        // Extract health from status string (e.g. "Up 2 hours (healthy)")
        let health = if status.contains("(healthy)") {
            "healthy".to_string()
        } else if status.contains("(unhealthy)") {
            "unhealthy".to_string()
        } else if status.contains("(health: starting)") {
            "starting".to_string()
        } else {
            "none".to_string()
        };

        containers.push(ContainerStatus {
            name,
            status,
            health,
            ports,
            uptime,
        });
    }

    containers
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_available_does_not_panic() {
        // Just verify it returns a bool without panicking
        let _available = docker_available();
    }

    #[test]
    fn test_parse_docker_ps_output() {
        let output = "ai_assistant\tUp 2 hours (healthy)\t0.0.0.0:8080->80/tcp\t2 hours\n\
                       redis\tUp 30 minutes\t6379/tcp\t30 minutes\n\
                       postgres\tUp 2 hours (unhealthy)\t5432/tcp\t2 hours\n";

        let containers = parse_docker_ps_output(output);
        assert_eq!(containers.len(), 3);

        assert_eq!(containers[0].name, "ai_assistant");
        assert_eq!(containers[0].health, "healthy");
        assert!(containers[0].ports.contains("8080"));

        assert_eq!(containers[1].name, "redis");
        assert_eq!(containers[1].health, "none");

        assert_eq!(containers[2].name, "postgres");
        assert_eq!(containers[2].health, "unhealthy");
    }
}
