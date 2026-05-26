// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_setup — Setup and management CLI for AI Assistant.
//!
//! Run with: `cargo run --bin ai_setup --features full -- <command>`
//!
//! Provides interactive initialization, prerequisite checking, configuration
//! management, Docker orchestration, node lifecycle, and backup/restore.
//!
//! ## Commands
//!
//! ```text
//! ai_setup init [--yes]              Interactive setup wizard
//! ai_setup check                     Check prerequisites
//! ai_setup validate <config>         Validate a config file
//! ai_setup version                   Show version and platform info
//! ai_setup config show [--redact]    Show configuration
//! ai_setup config get <key>          Get a config value
//! ai_setup config set <key> <value>  Set a config value
//! ai_setup config diff <a> <b>       Compare two configs
//! ai_setup export <format> <output>  Export config to TOML/JSON
//! ai_setup import <input> [output]   Import config with validation
//! ai_setup start [--foreground]      Start the server node
//! ai_setup stop                      Stop the server node
//! ai_setup status                    Show node status
//! ai_setup docker build [features]   Build Docker image
//! ai_setup docker up [profiles...]   Start Docker Compose services
//! ai_setup docker down               Stop Docker Compose services
//! ai_setup docker status             Show container status
//! ai_setup docker logs <name> [n]    Show container logs
//! ai_setup backup [--include-models] Create a backup
//! ai_setup restore <archive>         Restore from backup
//! ai_setup install <target>          Install a prerequisite
//! ```

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use ai_assistant::config_file::{default_config_path, ConfigFile};
use ai_assistant::setup::{backup, config_ops, docker_ops, node_manager, prereq};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("init") => cmd_init(&args[2..]),
        Some("check") => cmd_check(&args[2..]),
        Some("validate") => cmd_validate(&args[2..]),
        Some("version") => cmd_version(),
        Some("config") => cmd_config(&args[2..]),
        Some("export") => cmd_export(&args[2..]),
        Some("import") => cmd_import(&args[2..]),
        Some("start") => cmd_start(&args[2..]),
        Some("stop") => cmd_stop(&args[2..]),
        Some("status") => cmd_status(&args[2..]),
        Some("docker") => cmd_docker(&args[2..]),
        Some("backup") => cmd_backup(&args[2..]),
        Some("restore") => cmd_restore(&args[2..]),
        Some("install") => cmd_install(&args[2..]),
        Some("recommend") => cmd_recommend(&args[2..]),
        Some("hardware") => cmd_hardware(&args[2..]),
        Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!(
                "{}Unknown command: {}. Run ai_setup --help{}",
                RED, other, RESET
            );
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// ANSI color helpers
// =============================================================================

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn ok(msg: &str) {
    println!("  {} \u{2713} {}{}", GREEN, RESET, msg);
}

fn fail(msg: &str) {
    println!("  {} \u{2717} {}{}", RED, RESET, msg);
}

fn warn(msg: &str) {
    println!("  {} \u{26A0} {}{}", YELLOW, RESET, msg);
}

// =============================================================================
// help
// =============================================================================

fn print_help() -> ExitCode {
    println!(
        "{}ai_setup{} — AI Assistant Setup & Management CLI",
        BOLD, RESET
    );
    println!();
    println!("{}Usage:{} ai_setup <command> [options]", BOLD, RESET);
    println!();
    println!("{}Commands:{}", BOLD, RESET);
    println!(
        "  {}init{}     [--yes]                Interactive setup wizard",
        CYAN, RESET
    );
    println!(
        "  {}check{}                           Check prerequisites (Ollama, Docker, GPU, API keys)",
        CYAN, RESET
    );
    println!(
        "  {}validate{} <config>               Validate a configuration file",
        CYAN, RESET
    );
    println!(
        "  {}version{}                         Show version, features, and platform info",
        CYAN, RESET
    );
    println!(
        "  {}config{}   show [--redact] [file]  Show configuration",
        CYAN, RESET
    );
    println!(
        "  {}config{}   get <key> [file]        Get a specific config value",
        CYAN, RESET
    );
    println!(
        "  {}config{}   set <key> <value> [file] Set a config value",
        CYAN, RESET
    );
    println!(
        "  {}config{}   diff <file_a> <file_b>  Compare two config files",
        CYAN, RESET
    );
    println!(
        "  {}export{}   <format> <output> [file] Export config (toml/json)",
        CYAN, RESET
    );
    println!(
        "  {}import{}   <input> [output]        Import config with validation",
        CYAN, RESET
    );
    println!(
        "  {}start{}    [--foreground] [--config <file>] Start the server node",
        CYAN, RESET
    );
    println!(
        "  {}stop{}                             Stop the server node",
        CYAN, RESET
    );
    println!(
        "  {}status{}                           Show node status",
        CYAN, RESET
    );
    println!(
        "  {}docker{}   build [features]        Build Docker image",
        CYAN, RESET
    );
    println!(
        "  {}docker{}   up [profiles...]        Start Docker Compose services",
        CYAN, RESET
    );
    println!(
        "  {}docker{}   down                    Stop Docker Compose services",
        CYAN, RESET
    );
    println!(
        "  {}docker{}   status                  Show container statuses",
        CYAN, RESET
    );
    println!(
        "  {}docker{}   logs <container> [tail]  Show container logs",
        CYAN, RESET
    );
    println!(
        "  {}backup{}   [--include-models] [--output <file>] Create backup",
        CYAN, RESET
    );
    println!(
        "  {}restore{}  <archive> [--target <dir>] Restore from backup",
        CYAN, RESET
    );
    println!("  {}install{}  <target>                Install a prerequisite (ollama, docker, vllm, llamacpp, model <name>)", CYAN, RESET);
    println!(
        "  {}recommend{} [--workload <kind>]       Recommend a local inference runtime (Ollama / vLLM / llama.cpp / LM Studio)",
        CYAN, RESET
    );
    println!(
        "  {}hardware{}  [--json]                  Probe and print host hardware (CPU/RAM/GPU/OS)",
        CYAN, RESET
    );
    println!();
    println!("{}Examples:{}", BOLD, RESET);
    println!("  ai_setup init");
    println!("  ai_setup check");
    println!("  ai_setup config show --redact");
    println!("  ai_setup config set provider.model mistral");
    println!("  ai_setup docker status");
    println!("  ai_setup install ollama");
    println!("  ai_setup install vllm");
    println!("  ai_setup install llamacpp");
    println!("  ai_setup install model llama3");
    println!("  ai_setup recommend --workload multi-agent");
    println!("  ai_setup backup --output my_backup.gz");
    ExitCode::SUCCESS
}

// =============================================================================
// init — Interactive setup wizard
// =============================================================================

fn cmd_init(args: &[String]) -> ExitCode {
    let auto_yes = args.iter().any(|a| a == "--yes" || a == "-y");

    println!();
    println!("{}{}AI Assistant — Setup Wizard{}", BOLD, CYAN, RESET);
    println!("{}={}{}", DIM, "=".repeat(40), RESET);
    println!();

    // Step 1: Check prerequisites
    println!("{}Step 1: Checking prerequisites...{}", BOLD, RESET);
    let statuses = prereq::check_prerequisites();
    for s in &statuses {
        if s.installed {
            ok(&format!(
                "{} — {}{}",
                s.name,
                s.version.as_deref().unwrap_or("detected"),
                if s.details.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", s.details)
                }
            ));
        } else {
            fail(&format!("{} — {}", s.name, s.details));
        }
    }
    println!();

    // Step 2: Determine config path
    let config_path = default_config_path();
    println!("{}Step 2: Configuration{}", BOLD, RESET);
    println!("  Config path: {}{}{}", DIM, config_path.display(), RESET);

    let config_exists = config_path.exists();

    if config_exists {
        println!("  Existing config found.");
        if !auto_yes {
            print!("  Overwrite with new config? [y/N] ");
            let _ = std::io::stdout().flush();
            let mut answer = String::new();
            let _ = std::io::stdin().lock().read_line(&mut answer);
            if !answer.trim().eq_ignore_ascii_case("y") {
                println!("  Keeping existing config.");
                println!();
                println!("{}Setup complete.{}", GREEN, RESET);
                return ExitCode::SUCCESS;
            }
        }
    }

    // Step 3: Generate config
    println!();
    println!("{}Step 3: Generating configuration...{}", BOLD, RESET);

    // Determine provider based on what's available
    let ollama_ok = statuses.iter().any(|s| s.name == "Ollama" && s.installed);
    let openai_ok = statuses
        .iter()
        .any(|s| s.name == "OpenAI API Key" && s.installed);

    let provider_type = if ollama_ok {
        "ollama"
    } else if openai_ok {
        "openai"
    } else {
        "ollama" // default
    };

    let model = if ollama_ok {
        "llama3"
    } else if openai_ok {
        "gpt-4o-mini"
    } else {
        "llama3"
    };

    if !auto_yes {
        println!("  Provider: {} (auto-detected)", provider_type);
        println!("  Model: {}", model);
        print!("  Accept defaults? [Y/n] ");
        let _ = std::io::stdout().flush();
        let mut answer = String::new();
        let _ = std::io::stdin().lock().read_line(&mut answer);
        if answer.trim().eq_ignore_ascii_case("n") {
            println!(
                "  Please edit the config manually at: {}",
                config_path.display()
            );
            return ExitCode::SUCCESS;
        }
    }

    // Create config
    let config_content = format!(
        r#"# AI Assistant configuration — generated by ai_setup
# See: https://ai-assistant.runawaybrains.com/docs/config

[provider]
type = "{}"
model = "{}"

[urls]
ollama = "http://localhost:11434"
lm_studio = "http://localhost:1234"

[generation]
temperature = 0.7
max_history = 20

[rag]
enabled = true
knowledge_tokens = 2000
conversation_tokens = 1500

[logging]
level = "info"
"#,
        provider_type, model
    );

    if let Some(parent) = config_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match std::fs::write(&config_path, &config_content) {
        Ok(()) => ok(&format!("Config written to {}", config_path.display())),
        Err(e) => {
            fail(&format!("Failed to write config: {}", e));
            return ExitCode::from(1);
        }
    }

    println!();
    println!("{}Setup complete!{}", GREEN, RESET);
    println!(
        "  Run {}ai_setup check{} to verify your environment.",
        BOLD, RESET
    );
    println!("  Run {}ai_setup start{} to start the server.", BOLD, RESET);

    ExitCode::SUCCESS
}

// =============================================================================
// check — prerequisite checklist
// =============================================================================

fn cmd_check(_args: &[String]) -> ExitCode {
    println!();
    println!("{}Prerequisite Check{}", BOLD, RESET);
    println!("{}{}{}", DIM, "-".repeat(40), RESET);
    println!();

    let statuses = prereq::check_prerequisites();
    let mut all_ok = true;

    for s in &statuses {
        if s.installed {
            ok(&format!(
                "{:<20} {}",
                s.name,
                s.version.as_deref().unwrap_or(&s.details)
            ));
        } else {
            fail(&format!("{:<20} {}", s.name, s.details));
            all_ok = false;
        }
    }

    println!();
    if all_ok {
        println!("{}All prerequisites satisfied.{}", GREEN, RESET);
    } else {
        println!(
            "{}Some prerequisites are missing.{} Run {}ai_setup install <target>{} to install them.",
            YELLOW, RESET, BOLD, RESET
        );
    }

    ExitCode::SUCCESS
}

// =============================================================================
// validate — config validation
// =============================================================================

fn cmd_validate(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: ai_setup validate <config-file>");
            return ExitCode::from(1);
        }
    };

    match ConfigFile::load(&path) {
        Ok(config) => match config.validate_detailed() {
            Ok(()) => {
                ok(&format!("{} is valid", path.display()));
                ExitCode::SUCCESS
            }
            Err(errors) => {
                println!();
                println!("{}Validation errors in {}:{}", RED, path.display(), RESET);
                for err in &errors {
                    fail(&format!("{}", err));
                }
                ExitCode::from(1)
            }
        },
        Err(e) => {
            fail(&format!("Failed to load {}: {}", path.display(), e));
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// version
// =============================================================================

fn cmd_version() -> ExitCode {
    println!(
        "{}ai_assistant{} v{}",
        BOLD,
        RESET,
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "  Platform:  {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  Features:  full");

    #[cfg(debug_assertions)]
    println!("  Profile:   debug");
    #[cfg(not(debug_assertions))]
    println!("  Profile:   release");

    println!("  Rust:      {}", env!("CARGO_PKG_RUST_VERSION", "unknown"));

    ExitCode::SUCCESS
}

// =============================================================================
// config — sub-subcommands: show, get, set, diff
// =============================================================================

fn cmd_config(args: &[String]) -> ExitCode {
    match args.first().map(|s| s.as_str()) {
        Some("show") => {
            let redact = args.iter().any(|a| a == "--redact");
            let path = args
                .iter()
                .filter(|a| !a.starts_with('-'))
                .nth(1)
                .map(PathBuf::from)
                .unwrap_or_else(default_config_path);

            match config_ops::show_config(&path, redact) {
                Ok(content) => {
                    println!("{}", content);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("get") => {
            let key = match args.get(1) {
                Some(k) => k,
                None => {
                    eprintln!("Usage: ai_setup config get <key> [config-file]");
                    return ExitCode::from(1);
                }
            };
            let path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(default_config_path);

            match config_ops::get_config_value(&path, key) {
                Ok(val) => {
                    println!("{}", val);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("set") => {
            let key = match args.get(1) {
                Some(k) => k,
                None => {
                    eprintln!("Usage: ai_setup config set <key> <value> [config-file]");
                    return ExitCode::from(1);
                }
            };
            let value = match args.get(2) {
                Some(v) => v,
                None => {
                    eprintln!("Usage: ai_setup config set <key> <value> [config-file]");
                    return ExitCode::from(1);
                }
            };
            let path = args
                .get(3)
                .map(PathBuf::from)
                .unwrap_or_else(default_config_path);

            match config_ops::set_config_value(&path, key, value) {
                Ok(result) => {
                    if result.old_value.is_empty() {
                        ok(&format!("Set {} = \"{}\"", key, result.new_value));
                    } else {
                        ok(&format!(
                            "Changed {} = \"{}\" -> \"{}\"",
                            key, result.old_value, result.new_value
                        ));
                    }
                    if result.needs_restart {
                        warn("This change requires a server restart to take effect.");
                    } else {
                        println!("  {}(hot-reloadable — no restart needed){}", DIM, RESET);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("diff") => {
            let path_a = match args.get(1) {
                Some(p) => PathBuf::from(p),
                None => {
                    eprintln!("Usage: ai_setup config diff <file_a> <file_b>");
                    return ExitCode::from(1);
                }
            };
            let path_b = match args.get(2) {
                Some(p) => PathBuf::from(p),
                None => {
                    eprintln!("Usage: ai_setup config diff <file_a> <file_b>");
                    return ExitCode::from(1);
                }
            };

            let diffs = config_ops::diff_configs(&path_a, &path_b);
            if diffs.is_empty() {
                println!("  {}No differences found.{}", DIM, RESET);
            } else {
                println!();
                println!(
                    "  {}{:<20} {:<15} {:<20} {}",
                    BOLD, "Section.Key", "File A", "File B", RESET
                );
                println!("  {}{}", DIM, "-".repeat(55));
                for d in &diffs {
                    let full_key = if d.section.is_empty() {
                        d.key.clone()
                    } else {
                        format!("{}.{}", d.section, d.key)
                    };
                    println!(
                        "  {:<20} {}{:<20}{} {}{:<20}{}",
                        full_key,
                        RED,
                        if d.value_a.is_empty() {
                            "(absent)"
                        } else {
                            &d.value_a
                        },
                        RESET,
                        GREEN,
                        if d.value_b.is_empty() {
                            "(absent)"
                        } else {
                            &d.value_b
                        },
                        RESET
                    );
                }
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("Usage: ai_setup config <show|get|set|diff> [args]");
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// export
// =============================================================================

fn cmd_export(args: &[String]) -> ExitCode {
    let format = match args.first() {
        Some(f) => f.as_str(),
        None => {
            eprintln!("Usage: ai_setup export <toml|json> <output-file> [config-file]");
            return ExitCode::from(1);
        }
    };
    let output = match args.get(1) {
        Some(o) => PathBuf::from(o),
        None => {
            eprintln!("Usage: ai_setup export <toml|json> <output-file> [config-file]");
            return ExitCode::from(1);
        }
    };
    let source = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(default_config_path);

    match config_ops::export_config(&source, format, &output) {
        Ok(()) => {
            ok(&format!("Exported to {}", output.display()));
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// import
// =============================================================================

fn cmd_import(args: &[String]) -> ExitCode {
    let input = match args.first() {
        Some(i) => PathBuf::from(i),
        None => {
            eprintln!("Usage: ai_setup import <input-file> [output-file]");
            return ExitCode::from(1);
        }
    };
    let output = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_config_path);

    match config_ops::import_config(&input, &output) {
        Ok(warnings) => {
            ok(&format!("Imported to {}", output.display()));
            if !warnings.is_empty() {
                println!();
                println!("  {}Warnings:{}", YELLOW, RESET);
                for w in &warnings {
                    warn(w);
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// start / stop / status
// =============================================================================

fn cmd_start(args: &[String]) -> ExitCode {
    let foreground = args.iter().any(|a| a == "--foreground" || a == "-f");

    // Find --config <path>
    let config_path = args
        .windows(2)
        .find(|w| w[0] == "--config")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(default_config_path);

    match node_manager::start_node(&config_path, foreground) {
        Ok(msg) => {
            ok(&msg);
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

fn cmd_stop(_args: &[String]) -> ExitCode {
    match node_manager::stop_node() {
        Ok(()) => {
            ok("Server stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

fn cmd_status(_args: &[String]) -> ExitCode {
    match node_manager::node_status() {
        Ok(info) => {
            println!();
            println!("{}Node Status{}", BOLD, RESET);
            println!("{}{}{}", DIM, "-".repeat(30), RESET);

            if info.running {
                ok(&format!("Running (PID: {}, port: {})", info.pid, info.port));
                println!(
                    "  Health: {}{}{}",
                    if info.health == "ok" { GREEN } else { YELLOW },
                    info.health,
                    RESET
                );
                if info.uptime_secs > 0 {
                    println!("  Uptime: {} seconds", info.uptime_secs);
                }
            } else {
                fail("Not running");
                println!("  Run {}ai_setup start{} to start the server.", BOLD, RESET);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// docker — sub-subcommands: build, up, down, status, logs
// =============================================================================

fn cmd_docker(args: &[String]) -> ExitCode {
    match args.first().map(|s| s.as_str()) {
        Some("build") => {
            let features = args.get(1).map(|s| s.as_str()).unwrap_or("full");
            println!("  Building Docker image with features: {}", features);
            match docker_ops::docker_build(features) {
                Ok(output) => {
                    ok("Docker build succeeded");
                    if !output.is_empty() {
                        println!("{}", output);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("up") => {
            let profiles: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
            match docker_ops::docker_compose_up(&profiles) {
                Ok(output) => {
                    ok("Docker Compose services started");
                    if !output.is_empty() {
                        println!("{}", output);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("down") => match docker_ops::docker_compose_down() {
            Ok(_) => {
                ok("Docker Compose services stopped");
                ExitCode::SUCCESS
            }
            Err(e) => {
                fail(&e);
                ExitCode::from(1)
            }
        },
        Some("status") => {
            if !docker_ops::docker_available() {
                fail("Docker is not available on this system");
                return ExitCode::from(1);
            }

            match docker_ops::docker_status() {
                Ok(containers) => {
                    if containers.is_empty() {
                        println!("  {}No running containers.{}", DIM, RESET);
                    } else {
                        println!();
                        println!(
                            "  {}{:<25} {:<20} {:<12} {}",
                            BOLD, "Name", "Status", "Health", RESET
                        );
                        println!("  {}{}{}", DIM, "-".repeat(57), RESET);
                        for c in &containers {
                            let health_color = match c.health.as_str() {
                                "healthy" => GREEN,
                                "unhealthy" => RED,
                                _ => YELLOW,
                            };
                            println!(
                                "  {:<25} {:<20} {}{:<12}{}",
                                c.name, c.status, health_color, c.health, RESET
                            );
                        }
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        Some("logs") => {
            let container = match args.get(1) {
                Some(c) => c.as_str(),
                None => {
                    eprintln!("Usage: ai_setup docker logs <container> [tail-lines]");
                    return ExitCode::from(1);
                }
            };
            let tail: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

            match docker_ops::docker_logs(container, tail) {
                Ok(logs) => {
                    println!("{}", logs);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    fail(&e);
                    ExitCode::from(1)
                }
            }
        }
        _ => {
            eprintln!("Usage: ai_setup docker <build|up|down|status|logs> [args]");
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// backup / restore
// =============================================================================

fn cmd_backup(args: &[String]) -> ExitCode {
    let include_models = args.iter().any(|a| a == "--include-models");

    // Find --output <path>
    let output = args
        .windows(2)
        .find(|w| w[0] == "--output" || w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| {
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            PathBuf::from(format!("ai_assistant_backup_{}.gz", ts))
        });

    let config_dir = default_config_path()
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();

    if !config_dir.exists() {
        fail(&format!(
            "Config directory does not exist: {}",
            config_dir.display()
        ));
        return ExitCode::from(1);
    }

    match backup::create_backup(&config_dir, &output, include_models) {
        Ok(info) => {
            ok(&format!(
                "Backup created: {} ({} files, {} bytes)",
                info.path.display(),
                info.files_count,
                info.size_bytes
            ));
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

fn cmd_restore(args: &[String]) -> ExitCode {
    let archive = match args.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: ai_setup restore <archive> [--target <dir>]");
            return ExitCode::from(1);
        }
    };

    // Find --target <path>
    let target = args
        .windows(2)
        .find(|w| w[0] == "--target")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| {
            default_config_path()
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        });

    match backup::restore_backup(&archive, &target) {
        Ok(()) => {
            ok(&format!("Restored to {}", target.display()));
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// install — prerequisite installer
// =============================================================================

fn cmd_install(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_setup install <ollama|docker|vllm|llamacpp|model name>");
        return ExitCode::from(1);
    }

    let target = args.join(" ");
    match prereq::install_command(&target) {
        Ok(instructions) => {
            println!();
            if !instructions.command.is_empty() {
                println!("  {}Command:{}", BOLD, RESET);
                println!("    {}{}{}", CYAN, instructions.command, RESET);
            }
            if !instructions.manual_steps.is_empty() {
                println!();
                println!("  {}Manual steps:{}", BOLD, RESET);
                for line in instructions.manual_steps.lines() {
                    println!("    {}", line);
                }
            }
            if !instructions.url.is_empty() {
                println!();
                println!("  {}More info:{} {}", DIM, RESET, instructions.url);
            }
            println!();
            ExitCode::SUCCESS
        }
        Err(e) => {
            fail(&e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// recommend — runtime recommendation (Ollama / vLLM / llama.cpp / LM Studio)
// =============================================================================

#[cfg(feature = "butler")]
fn cmd_recommend(args: &[String]) -> ExitCode {
    use ai_assistant::{Butler, RuntimeKind, WorkloadHint};

    // Parse --workload <kind>
    let mut workload = WorkloadHint::Auto;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workload" | "-w" => {
                let v = match args.get(i + 1) {
                    Some(v) => v.clone(),
                    None => {
                        eprintln!("--workload requires a value");
                        return ExitCode::from(1);
                    }
                };
                workload = match v.as_str() {
                    "auto" => WorkloadHint::Auto,
                    "chat" | "interactive" => WorkloadHint::InteractiveChat,
                    "code" | "code-assist" => WorkloadHint::CodeAssist,
                    "agentic" | "agentic-coding" => WorkloadHint::AgenticCoding,
                    "research" => WorkloadHint::ResearchPipeline,
                    "multi-agent" => WorkloadHint::MultiAgent {
                        concurrent_agents: 4,
                    },
                    "eval" | "batch" => WorkloadHint::EvalBatch { prompt_count: 100 },
                    "autonomous" | "scheduler" => WorkloadHint::AutonomousScheduler,
                    other => {
                        eprintln!(
                            "Unknown workload: {}. Expected: auto, chat, code, agentic, research, multi-agent, eval, autonomous",
                            other
                        );
                        return ExitCode::from(1);
                    }
                };
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "{}ai_setup recommend{} — Recommend a local inference runtime",
                    BOLD, RESET
                );
                println!();
                println!(
                    "{}Usage:{} ai_setup recommend [--workload <kind>]",
                    BOLD, RESET
                );
                println!();
                println!("{}Workload kinds:{}", BOLD, RESET);
                println!("  auto            Let Butler decide from the environment (default)");
                println!("  chat            Single-user interactive chat");
                println!("  code            IDE-integrated coding assistant");
                println!("  agentic         Autonomous coding agent (Aider/Cline-style)");
                println!("  research        Research pipeline (many sequential queries)");
                println!("  multi-agent     Multi-agent orchestration (N concurrent agents)");
                println!("  eval            Eval / benchmark batch over many prompts");
                println!("  autonomous      Autonomous scheduler running cron-style jobs");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                return ExitCode::from(1);
            }
        }
    }

    println!("{}Scanning environment...{}", DIM, RESET);
    let mut butler = Butler::new();
    let report = butler.scan();
    let rec = butler.recommend_runtime(&report, workload);

    println!();
    println!("{}Recommended runtime:{} {}", BOLD, RESET, rec.preferred);
    if let Some(fb) = rec.fallback {
        println!("  {}Fallback:{} {}", DIM, RESET, fb);
    }
    println!();
    println!("  {}Reason:{}", BOLD, RESET);
    for line in textwrap_simple(&rec.reason, 76) {
        println!("    {}", line);
    }
    if !rec.estimated_speedup.is_empty() {
        println!();
        println!("  {}Speedup:{} {}", BOLD, RESET, rec.estimated_speedup);
    }
    if let Some(tp) = rec.suggested_tensor_parallel_size {
        println!();
        println!(
            "  {}Tensor parallelism:{} shard across {} GPU{} \
             ({}--tensor-parallel-size {}{})",
            BOLD,
            RESET,
            tp,
            if tp == 1 { "" } else { "s" },
            CYAN,
            tp,
            RESET
        );
    }
    if !rec.caveats.is_empty() {
        println!();
        println!("  {}Caveats:{}", BOLD, RESET);
        for c in &rec.caveats {
            for line in textwrap_simple(c, 74) {
                println!("    - {}", line);
            }
        }
    }
    if let Some(hint) = &rec.install_hint {
        println!();
        println!("  {}Install:{} {}{}{}", BOLD, RESET, CYAN, hint, RESET);
    }
    println!();

    let running = report.llm_providers.iter().any(|p| match rec.preferred {
        RuntimeKind::Ollama => matches!(p.provider_type, ai_assistant::AiProvider::Ollama),
        RuntimeKind::LmStudio => matches!(p.provider_type, ai_assistant::AiProvider::LMStudio),
        RuntimeKind::LlamaCpp => matches!(p.provider_type, ai_assistant::AiProvider::LlamaCpp),
        RuntimeKind::VLlm => matches!(p.provider_type, ai_assistant::AiProvider::VLLM),
        _ => false,
    });
    if running {
        ok(&format!("{} is already running.", rec.preferred));
    } else {
        warn(&format!(
            "{} is not currently running — install or start it.",
            rec.preferred
        ));
    }

    ExitCode::SUCCESS
}

#[cfg(not(feature = "butler"))]
fn cmd_recommend(_args: &[String]) -> ExitCode {
    eprintln!(
        "{}`recommend` requires the `butler` feature. Rebuild with `cargo build --features butler`.{}",
        RED, RESET
    );
    ExitCode::from(1)
}

#[cfg(feature = "butler")]
fn textwrap_simple(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            out.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

// =============================================================================
// hardware — V139 host hardware probe
// =============================================================================

fn cmd_hardware(args: &[String]) -> ExitCode {
    let mut as_json = false;
    for a in args {
        match a.as_str() {
            "--json" | "-j" => as_json = true,
            "--help" | "-h" => {
                println!(
                    "{}ai_setup hardware{} — Probe host CPU, RAM, GPUs, OS",
                    BOLD, RESET
                );
                println!();
                println!("{}Usage:{} ai_setup hardware [--json]", BOLD, RESET);
                println!();
                println!("  --json    Emit machine-readable JSON instead of the table.");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                return ExitCode::from(1);
            }
        }
    }

    let info = match ai_assistant::hardware_info::detect() {
        Ok(info) => info,
        Err(e) => {
            eprintln!("{}hardware probe failed: {}{}", RED, e, RESET);
            return ExitCode::from(1);
        }
    };
    if as_json {
        match serde_json::to_string_pretty(&info) {
            Ok(s) => println!("{}", s),
            Err(e) => {
                eprintln!("{}JSON encode failed: {}{}", RED, e, RESET);
                return ExitCode::from(1);
            }
        }
    } else {
        println!("{}Host hardware{}", BOLD, RESET);
        println!("{}", info.pretty_summary());
    }
    ExitCode::SUCCESS
}
