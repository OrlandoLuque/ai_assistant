// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Prerequisites detection and install command generation.
//!
//! Checks for required and optional dependencies (Ollama, Docker, GPU, API keys)
//! and generates platform-specific installation instructions.

use std::collections::HashMap;
use std::process::Command;

/// Status of a single prerequisite.
#[derive(Debug, Clone)]
pub struct PrereqStatus {
    /// Human-readable name (e.g. "Ollama", "Docker", "GPU").
    pub name: String,
    /// Whether the prerequisite is installed / available.
    pub installed: bool,
    /// Detected version string, if available.
    pub version: Option<String>,
    /// Additional details (e.g. model list, GPU name).
    pub details: String,
}

/// Platform-specific installation instructions.
#[derive(Debug, Clone)]
pub struct InstallInstructions {
    /// Shell command to run (may be empty if manual-only).
    pub command: String,
    /// Human-readable manual steps (empty if command is sufficient).
    pub manual_steps: String,
    /// URL with more information.
    pub url: String,
}

/// Check all prerequisites and return a status list.
///
/// This is a lightweight check that shells out to detect installed tools.
/// It does NOT use Butler (which requires the `butler` feature) — instead
/// it performs simple direct checks so the setup module is always available.
pub fn check_prerequisites() -> Vec<PrereqStatus> {
    let mut results = Vec::new();

    // 1. Ollama
    results.push(check_ollama());

    // 2. Docker
    results.push(check_docker());

    // 3. GPU
    results.push(check_gpu());

    // 4. OpenAI API key
    results.push(check_env_key("OpenAI API Key", "OPENAI_API_KEY"));

    // 5. Anthropic API key
    results.push(check_env_key("Anthropic API Key", "ANTHROPIC_API_KEY"));

    results
}

/// Generate platform-specific install instructions for a target.
///
/// Supported targets: `"ollama"`, `"docker"`, `"model <name>"`.
pub fn install_command(target: &str) -> Result<InstallInstructions, String> {
    let target_lower = target.trim().to_lowercase();

    if target_lower == "ollama" {
        return Ok(install_ollama());
    }

    if target_lower == "docker" {
        return Ok(install_docker());
    }

    if let Some(model_name) = target_lower.strip_prefix("model ") {
        let model_name = model_name.trim();
        if model_name.is_empty() {
            return Err("Model name cannot be empty. Usage: install model <name>".to_string());
        }
        return Ok(InstallInstructions {
            command: format!("ollama pull {}", model_name),
            manual_steps: String::new(),
            url: format!("https://ollama.com/library/{}", model_name),
        });
    }

    Err(format!(
        "Unknown install target: '{}'. Supported: ollama, docker, model <name>",
        target
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn check_ollama() -> PrereqStatus {
    // Try `ollama --version` or check if the API is reachable
    let version = Command::new("ollama")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    let installed = version.is_some();

    let details = if installed {
        // Try to list models
        let models = Command::new("ollama")
            .arg("list")
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()
                } else {
                    None
                }
            });
        match models {
            Some(m) => {
                let count = m.lines().count().saturating_sub(1); // header line
                format!("{} model(s) installed", count)
            }
            None => "Running but could not list models".to_string(),
        }
    } else {
        "Not installed".to_string()
    };

    PrereqStatus {
        name: "Ollama".to_string(),
        installed,
        version,
        details,
    }
}

fn check_docker() -> PrereqStatus {
    let version = Command::new("docker")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    let installed = version.is_some();
    let details = if installed {
        "Docker CLI available".to_string()
    } else {
        "Not installed".to_string()
    };

    PrereqStatus {
        name: "Docker".to_string(),
        installed,
        version,
        details,
    }
}

fn check_gpu() -> PrereqStatus {
    // NVIDIA: nvidia-smi
    let nvidia = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    if let Some(info) = nvidia {
        return PrereqStatus {
            name: "GPU".to_string(),
            installed: true,
            version: Some("NVIDIA".to_string()),
            details: info,
        };
    }

    // Apple Silicon: check sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        let apple = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    let s = String::from_utf8(o.stdout).ok()?.trim().to_string();
                    if s.contains("Apple") {
                        Some(s)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

        if let Some(info) = apple {
            return PrereqStatus {
                name: "GPU".to_string(),
                installed: true,
                version: Some("Apple Silicon".to_string()),
                details: info,
            };
        }
    }

    PrereqStatus {
        name: "GPU".to_string(),
        installed: false,
        version: None,
        details: "No NVIDIA or Apple GPU detected".to_string(),
    }
}

fn check_env_key(display_name: &str, env_var: &str) -> PrereqStatus {
    let value = std::env::var(env_var).ok();
    let installed = value.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
    let details = if installed {
        format!("${} is set", env_var)
    } else {
        format!("${} not set", env_var)
    };

    PrereqStatus {
        name: display_name.to_string(),
        installed,
        version: None,
        details,
    }
}

fn install_ollama() -> InstallInstructions {
    let os = std::env::consts::OS;
    match os {
        "windows" => InstallInstructions {
            command: "winget install Ollama.Ollama".to_string(),
            manual_steps: "Or download from https://ollama.com/download/windows".to_string(),
            url: "https://ollama.com/download".to_string(),
        },
        "macos" => InstallInstructions {
            command: "brew install ollama".to_string(),
            manual_steps: "Or download from https://ollama.com/download/mac".to_string(),
            url: "https://ollama.com/download".to_string(),
        },
        "linux" => InstallInstructions {
            command: "curl -fsSL https://ollama.com/install.sh | sh".to_string(),
            manual_steps: String::new(),
            url: "https://ollama.com/download".to_string(),
        },
        _ => InstallInstructions {
            command: String::new(),
            manual_steps: format!("See https://ollama.com/download for {} instructions", os),
            url: "https://ollama.com/download".to_string(),
        },
    }
}

fn install_docker() -> InstallInstructions {
    let os = std::env::consts::OS;
    match os {
        "windows" => InstallInstructions {
            command: String::new(),
            manual_steps: "Download Docker Desktop from https://www.docker.com/products/docker-desktop/\nRequires WSL 2 backend.".to_string(),
            url: "https://docs.docker.com/desktop/install/windows-install/".to_string(),
        },
        "macos" => InstallInstructions {
            command: "brew install --cask docker".to_string(),
            manual_steps: "Or download Docker Desktop from https://www.docker.com/products/docker-desktop/".to_string(),
            url: "https://docs.docker.com/desktop/install/mac-install/".to_string(),
        },
        "linux" => InstallInstructions {
            command: "curl -fsSL https://get.docker.com | sh".to_string(),
            manual_steps: "After install, add your user to the docker group:\n  sudo usermod -aG docker $USER".to_string(),
            url: "https://docs.docker.com/engine/install/".to_string(),
        },
        _ => InstallInstructions {
            command: String::new(),
            manual_steps: format!("See https://docs.docker.com/engine/install/ for {} instructions", os),
            url: "https://docs.docker.com/engine/install/".to_string(),
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_prerequisites_returns_all_items() {
        let statuses = check_prerequisites();
        assert!(statuses.len() >= 5, "Should check at least 5 prerequisites");

        let names: Vec<&str> = statuses.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Ollama"));
        assert!(names.contains(&"Docker"));
        assert!(names.contains(&"GPU"));
        assert!(names.contains(&"OpenAI API Key"));
        assert!(names.contains(&"Anthropic API Key"));
    }

    #[test]
    fn test_install_command_known_targets() {
        let ollama = install_command("ollama").expect("ollama should be a valid target");
        assert!(!ollama.url.is_empty());

        let docker = install_command("docker").expect("docker should be a valid target");
        assert!(!docker.url.is_empty());

        let model = install_command("model llama3").expect("model should be a valid target");
        assert!(model.command.contains("ollama pull llama3"));
        assert!(model.url.contains("llama3"));
    }

    #[test]
    fn test_install_command_unknown_target() {
        let result = install_command("foobar");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown install target"));

        // Empty model name
        let result = install_command("model ");
        assert!(result.is_err());
    }
}
