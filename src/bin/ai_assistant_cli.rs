//! Interactive CLI for the AI assistant.
//!
//! Run with: cargo run --bin ai_assistant_cli --features full
//!
//! This binary creates a REPL engine and runs an interactive input loop,
//! processing commands and messages from stdin.
//!
//! ## Docker support
//!
//! With `--containers` flag and `containers` feature:
//! ```bash
//! cargo run --bin ai_assistant_cli --features "full,containers" -- --containers
//! ```
//! Then use `/docker help` for container management commands.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use ai_assistant::repl::{ReplAction, ReplCommand, ReplConfig, ReplEngine};

// Docker handle type — conditional on feature
#[cfg(feature = "containers")]
type DockerHandle = Option<std::sync::Arc<std::sync::RwLock<ai_assistant::ContainerExecutor>>>;

#[cfg(not(feature = "containers"))]
type DockerHandle = ();

fn main() -> ExitCode {
    let containers_flag = std::env::args().any(|a| a == "--containers");

    // Docker executor init
    #[cfg(feature = "containers")]
    let docker_handle: DockerHandle = if containers_flag {
        match ai_assistant::ContainerExecutor::is_docker_available() {
            true => {
                let config = ai_assistant::ContainerConfig::default();
                match ai_assistant::ContainerExecutor::new(config) {
                    Ok(exec) => {
                        eprintln!("[docker] Docker available. /docker commands enabled.");
                        Some(std::sync::Arc::new(std::sync::RwLock::new(exec)))
                    }
                    Err(e) => {
                        eprintln!("[docker] WARNING: Failed to init executor: {}", e);
                        None
                    }
                }
            }
            false => {
                eprintln!("[docker] WARNING: Docker not available. /docker commands will fail.");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "containers"))]
    let docker_handle: DockerHandle = {
        if containers_flag {
            eprintln!("Warning: --containers requires the 'containers' feature. Ignoring.");
        }
        ()
    };

    let config = ReplConfig::default();
    let mut engine = ReplEngine::new(config);

    println!("AI Assistant CLI");
    println!("Type /help for available commands, /exit to quit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        print!("{}", engine.config().prompt_string);
        if stdout.flush().is_err() {
            break;
        }

        // Read one line
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                return ExitCode::FAILURE;
            }
        }

        // Docker commands — intercept before ReplEngine
        let trimmed = line.trim();
        if trimmed.starts_with("/docker") {
            #[cfg(feature = "containers")]
            {
                if let Some(ref exec) = docker_handle {
                    let output = handle_docker_command(trimmed, exec);
                    println!("{}", output);
                } else {
                    eprintln!("Docker not available. Start with --containers and ensure Docker is running.");
                }
            }
            #[cfg(not(feature = "containers"))]
            {
                let _ = &docker_handle;
                eprintln!("Docker requires --features containers");
            }
            continue;
        }

        let action = engine.process_input(&line);

        match action {
            ReplAction::SendMessage(msg) => {
                engine.add_message("user", &msg);
                println!("[AI response would go here - connect to a provider]");
                engine.add_message("assistant", "[AI response would go here - connect to a provider]");
            }
            ReplAction::ExecuteCommand(cmd) => match cmd {
                ReplCommand::Help => {
                    println!("{}", ReplEngine::format_help());
                    #[cfg(feature = "containers")]
                    println!("  /docker <cmd>  Docker container management (try /docker help)");
                }
                ReplCommand::Models => {
                    println!("Current model: {}", engine.current_model());
                    println!("(Connect a provider to list available models)");
                }
                ReplCommand::Config => {
                    println!("{}", engine.format_config());
                }
                ReplCommand::Clear => {
                    engine.clear_history();
                    println!("History cleared.");
                }
                ReplCommand::History => {
                    let hist = engine.history();
                    if hist.is_empty() {
                        println!("(no messages yet)");
                    } else {
                        for (role, content) in hist {
                            println!("{}: {}", role, content);
                        }
                    }
                }
                ReplCommand::Save(path) => {
                    if path.is_empty() {
                        println!("Usage: /save <path>");
                    } else {
                        match engine.save_session(&path) {
                            Ok(()) => println!("Session saved to {}", path),
                            Err(e) => eprintln!("Error saving session: {}", e),
                        }
                    }
                }
                ReplCommand::Load(path) => {
                    if path.is_empty() {
                        println!("Usage: /load <path>");
                    } else {
                        match engine.load_session(&path) {
                            Ok(()) => println!(
                                "Session loaded from {} ({} messages)",
                                path,
                                engine.history().len()
                            ),
                            Err(e) => eprintln!("Error loading session: {}", e),
                        }
                    }
                }
                ReplCommand::Model(name) => {
                    if name.is_empty() {
                        println!("Current model: {}", engine.current_model());
                    } else {
                        engine.set_model(&name);
                        println!("Model set to: {}", name);
                    }
                }
                ReplCommand::Template(name) => {
                    if name.is_empty() {
                        match engine.current_template() {
                            Some(t) => println!("Current template: {}", t),
                            None => println!("No template set."),
                        }
                    } else {
                        engine.set_template(&name);
                        println!("Template set to: {}", name);
                    }
                }
                ReplCommand::Cost => {
                    println!("(Cost tracking not connected in this demo CLI. Use AiAssistant::init_cost_tracking())");
                }
                ReplCommand::Unknown(cmd) => {
                    println!("Unknown command: /{}. Type /help for available commands.", cmd);
                }
                ReplCommand::Exit => unreachable!("Exit is handled by ReplAction::Exit"),
            },
            ReplAction::Continue => {
                // Empty input, just show prompt again
            }
            ReplAction::Exit => {
                println!("Goodbye!");
                break;
            }
        }
    }

    ExitCode::SUCCESS
}

// =============================================================================
// Docker REPL command handler
// =============================================================================

#[cfg(feature = "containers")]
fn handle_docker_command(
    input: &str,
    executor: &std::sync::Arc<std::sync::RwLock<ai_assistant::ContainerExecutor>>,
) -> String {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let subcmd = parts.get(1).copied().unwrap_or("help");

    match subcmd {
        "list" | "ls" => {
            let guard = match executor.read() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            let containers = guard.list();
            if containers.is_empty() {
                return "No managed containers.".to_string();
            }
            let mut out = format!("{:<16} {:<20} {:<25} {:<10}\n", "ID", "NAME", "IMAGE", "STATUS");
            out.push_str(&"-".repeat(71));
            out.push('\n');
            for r in containers {
                let short_id = if r.container_id.len() > 12 {
                    &r.container_id[..12]
                } else {
                    &r.container_id
                };
                out.push_str(&format!(
                    "{:<16} {:<20} {:<25} {:<10}\n",
                    short_id, r.name, r.image, r.status,
                ));
            }
            out
        }

        "create" => {
            let image = match parts.get(2) {
                Some(img) => *img,
                None => return "Usage: /docker create <image> [--name NAME] [--cmd CMD...]".to_string(),
            };
            let mut name = "mcp_container".to_string();
            let mut cmd: Option<Vec<String>> = None;
            let mut i = 3;
            while i < parts.len() {
                match parts[i] {
                    "--name" => {
                        if let Some(n) = parts.get(i + 1) {
                            name = n.to_string();
                            i += 2;
                        } else {
                            return "--name requires a value".to_string();
                        }
                    }
                    "--cmd" => {
                        cmd = Some(parts[i + 1..].iter().map(|s| s.to_string()).collect());
                        break;
                    }
                    _ => { i += 1; }
                }
            }
            let opts = ai_assistant::CreateOptions {
                cmd,
                ..Default::default()
            };
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.create(image, &name, opts) {
                Ok(id) => format!("Created container {} (image: {}, name: {})", &id[..12.min(id.len())], image, name),
                Err(e) => format!("Error: {}", e),
            }
        }

        "start" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker start <container_id>".to_string(),
            };
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.start(id) {
                Ok(()) => format!("Started container {}", id),
                Err(e) => format!("Error: {}", e),
            }
        }

        "stop" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker stop <container_id> [--timeout N]".to_string(),
            };
            let mut timeout: u32 = 10;
            if parts.get(3).copied() == Some("--timeout") {
                if let Some(t) = parts.get(4).and_then(|s| s.parse().ok()) {
                    timeout = t;
                }
            }
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.stop(id, timeout) {
                Ok(()) => format!("Stopped container {}", id),
                Err(e) => format!("Error: {}", e),
            }
        }

        "rm" | "remove" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker rm <container_id> [--force]".to_string(),
            };
            let force = parts.iter().any(|p| *p == "--force");
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.remove(id, force) {
                Ok(()) => format!("Removed container {}", id),
                Err(e) => format!("Error: {}", e),
            }
        }

        "exec" => {
            if parts.len() < 4 {
                return "Usage: /docker exec <container_id> <command...>".to_string();
            }
            let id = parts[2];
            let cmd: Vec<&str> = parts[3..].to_vec();
            let guard = match executor.read() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.exec(id, &cmd, std::time::Duration::from_secs(60)) {
                Ok(result) => {
                    let mut out = String::new();
                    if !result.stdout.is_empty() {
                        out.push_str(&result.stdout);
                    }
                    if !result.stderr.is_empty() {
                        if !out.is_empty() { out.push('\n'); }
                        out.push_str("[stderr] ");
                        out.push_str(&result.stderr);
                    }
                    if result.timed_out {
                        out.push_str("\n[timed out]");
                    }
                    out.push_str(&format!("\n[exit code: {}]", result.exit_code));
                    out
                }
                Err(e) => format!("Error: {}", e),
            }
        }

        "logs" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker logs <container_id> [--tail N]".to_string(),
            };
            let mut tail: usize = 100;
            if parts.get(3).copied() == Some("--tail") {
                if let Some(t) = parts.get(4).and_then(|s| s.parse().ok()) {
                    tail = t;
                }
            }
            let guard = match executor.read() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.logs(id, tail) {
                Ok(logs) => {
                    if logs.is_empty() {
                        "(no logs)".to_string()
                    } else {
                        logs
                    }
                }
                Err(e) => format!("Error: {}", e),
            }
        }

        "status" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker status <container_id>".to_string(),
            };
            let guard = match executor.read() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            match guard.status(id) {
                Some(status) => format!("Container {}: {}", id, status),
                None => format!("Container {} not found", id),
            }
        }

        "cleanup" => {
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {}", e),
            };
            let count = guard.cleanup_all();
            format!("Cleaned up {} container(s)", count)
        }

        "help" | _ => {
            "Docker commands:\n\
             \x20 /docker list              List all containers\n\
             \x20 /docker create <image>    Create container (--name NAME, --cmd CMD...)\n\
             \x20 /docker start <id>        Start a container\n\
             \x20 /docker stop <id>         Stop a container (--timeout N)\n\
             \x20 /docker rm <id>           Remove a container (--force)\n\
             \x20 /docker exec <id> <cmd>   Execute command in container\n\
             \x20 /docker logs <id>         Show container logs (--tail N)\n\
             \x20 /docker status <id>       Show container status\n\
             \x20 /docker cleanup           Remove all managed containers\n\
             \x20 /docker help              Show this help"
                .to_string()
        }
    }
}
