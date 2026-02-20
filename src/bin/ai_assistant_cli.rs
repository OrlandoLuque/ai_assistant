//! Interactive CLI for the AI assistant.
//!
//! Run with: cargo run --bin ai_assistant_cli --features full
//!
//! This binary creates a REPL engine and runs an interactive input loop,
//! processing commands and messages from stdin.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use ai_assistant::repl::{ReplAction, ReplCommand, ReplConfig, ReplEngine};

fn main() -> ExitCode {
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
