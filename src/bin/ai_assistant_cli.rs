//! Interactive CLI for the AI assistant.
//!
//! Run with: cargo run --bin ai_assistant_cli --features "full,butler"
//!
//! This binary creates an AiAssistant with Butler auto-detection of local
//! LLM providers (Ollama, LM Studio, cloud APIs), then runs an interactive
//! REPL loop with real streaming responses.
//!
//! ## Flags
//!
//! - `--provider <name>`: Override provider (ollama, lmstudio, openai, anthropic, etc.)
//! - `--model <name>`: Override model name
//! - `--url <url>`: Override provider URL
//! - `--no-butler`: Skip auto-detection, use defaults
//!
//! ## Docker support
//!
//! With `--containers` flag and `containers` feature:
//! ```bash
//! cargo run --bin ai_assistant_cli --features "full,butler,containers" -- --containers
//! ```
//! Then use `/docker help` for container management commands.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ai_assistant::repl::{ReplAction, ReplCommand, ReplConfig, ReplEngine};
use ai_assistant::{AiAssistant, AiResponse};
#[cfg(feature = "butler")]
use ai_assistant::ModelInfo;

#[cfg(feature = "butler")]
use ai_assistant::butler::{Butler, EnvironmentReport};

// Docker handle type — conditional on feature
#[cfg(feature = "containers")]
type DockerHandle = Option<std::sync::Arc<std::sync::RwLock<ai_assistant::ContainerExecutor>>>;

#[cfg(not(feature = "containers"))]
type DockerHandle = ();

/// Parsed CLI arguments.
struct CliArgs {
    provider_override: Option<String>,
    model_override: Option<String>,
    url_override: Option<String>,
    no_butler: bool,
    containers: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut cli = CliArgs {
        provider_override: None,
        model_override: None,
        url_override: None,
        no_butler: false,
        containers: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" => {
                cli.provider_override = args.get(i + 1).cloned();
                i += 2;
            }
            "--model" => {
                cli.model_override = args.get(i + 1).cloned();
                i += 2;
            }
            "--url" => {
                cli.url_override = args.get(i + 1).cloned();
                i += 2;
            }
            "--no-butler" => {
                cli.no_butler = true;
                i += 1;
            }
            "--containers" => {
                cli.containers = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                i += 1;
            }
        }
    }
    cli
}

fn print_usage() {
    println!("AI Assistant CLI — Interactive chat with local/cloud LLM providers\n");
    println!("Usage: ai_assistant_cli [OPTIONS]\n");
    println!("Options:");
    println!("  --provider <name>  Override provider (ollama, lmstudio, openai, anthropic, gemini, groq, deepseek, mistral)");
    println!("  --model <name>     Override model name");
    println!("  --url <url>        Override provider base URL");
    println!("  --no-butler        Skip auto-detection, use Ollama defaults");
    println!("  --containers       Enable Docker commands (/docker help)");
    println!("  -h, --help         Show this help");
    println!();
    println!("Examples:");
    println!("  ai_assistant_cli                         # Auto-detect providers");
    println!("  ai_assistant_cli --provider ollama       # Force Ollama");
    println!("  ai_assistant_cli --model llama3.1:8b     # Specific model");
    println!("  ai_assistant_cli --provider openai --model gpt-4o");
}

fn provider_from_name(name: &str) -> ai_assistant::AiProvider {
    match name.to_lowercase().as_str() {
        "ollama" => ai_assistant::AiProvider::Ollama,
        "lmstudio" | "lm-studio" | "lm_studio" => ai_assistant::AiProvider::LMStudio,
        "openai" => ai_assistant::AiProvider::OpenAI,
        "anthropic" => ai_assistant::AiProvider::Anthropic,
        "gemini" => ai_assistant::AiProvider::Gemini,
        "groq" => ai_assistant::AiProvider::Groq,
        "together" => ai_assistant::AiProvider::Together,
        "fireworks" => ai_assistant::AiProvider::Fireworks,
        "deepseek" => ai_assistant::AiProvider::DeepSeek,
        "mistral" => ai_assistant::AiProvider::Mistral,
        "perplexity" => ai_assistant::AiProvider::Perplexity,
        "openrouter" => ai_assistant::AiProvider::OpenRouter,
        _ => ai_assistant::AiProvider::Ollama,
    }
}

fn main() -> ExitCode {
    let cli = parse_args();

    // Background update check
    let update_rx = ai_assistant::update_checker::check_for_update_bg(env!("CARGO_PKG_VERSION"));

    // =========================================================================
    // Docker executor init
    // =========================================================================

    #[cfg(feature = "containers")]
    let docker_handle: DockerHandle = if cli.containers {
        match ai_assistant::ContainerExecutor::is_docker_available() {
            true => {
                let config = ai_assistant::ContainerConfig::default();
                match ai_assistant::ContainerExecutor::new(config) {
                    Ok(exec) => {
                        eprintln!("[docker] Docker available. /docker commands enabled.");
                        Some(std::sync::Arc::new(std::sync::RwLock::new(exec)))
                    }
                    Err(e) => {
                        eprintln!("[docker] WARNING: Failed to init executor: {e}");
                        None
                    }
                }
            }
            false => {
                eprintln!("[docker] WARNING: Docker not available.");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "containers"))]
    let docker_handle: DockerHandle = {
        if cli.containers {
            eprintln!("Warning: --containers requires the 'containers' feature. Ignoring.");
        }
    };

    // =========================================================================
    // Provider detection & AiAssistant init
    // =========================================================================

    let mut assistant = AiAssistant::new();

    #[cfg(feature = "butler")]
    let mut scan_report: Option<EnvironmentReport> = if !cli.no_butler {
        eprint!("Scanning for LLM providers...");
        let mut butler = Butler::new();
        let report = butler.scan();
        let config = butler.suggest_config(&report);
        assistant.load_config(config);
        eprintln!(" done.");
        Some(report)
    } else {
        None
    };

    #[cfg(not(feature = "butler"))]
    let scan_report: Option<()> = None;

    // Apply CLI overrides
    if let Some(ref provider_name) = cli.provider_override {
        assistant.config.provider = provider_from_name(provider_name);
    }
    if let Some(ref model) = cli.model_override {
        assistant.config.selected_model = model.clone();
    }
    if let Some(ref url) = cli.url_override {
        match assistant.config.provider {
            ai_assistant::AiProvider::Ollama => assistant.config.ollama_url = url.clone(),
            ai_assistant::AiProvider::LMStudio => assistant.config.lm_studio_url = url.clone(),
            _ => assistant.config.custom_url = url.clone(),
        }
    }

    // Print detected environment & build unified model list
    #[cfg(feature = "butler")]
    {
        if let Some(ref report) = scan_report {
            print_environment_summary(report);

            // Build unified model list from ALL detected providers
            let unified = build_unified_model_list(report);
            if !unified.is_empty() {
                // Auto-select first model and set its provider
                if assistant.config.selected_model.is_empty() {
                    assistant.config.selected_model = unified[0].name.clone();
                    assistant.config.provider = unified[0].provider.clone();
                    apply_provider_url(&mut assistant, report, &unified[0].provider);
                }
                assistant.available_models = unified;
                eprintln!("{} model(s) across {} provider(s).",
                    assistant.available_models.len(),
                    report.llm_providers.len(),
                );
                println!("Using model: {} {} ({})\n",
                    assistant.config.provider.icon(),
                    assistant.config.selected_model,
                    assistant.config.provider.display_name(),
                );
            } else {
                // Butler found providers but no models — try fetch
                fetch_models_blocking(&mut assistant);
            }
        } else {
            // No butler scan — fetch from configured provider
            fetch_models_blocking(&mut assistant);
        }
    }

    #[cfg(not(feature = "butler"))]
    {
        let _ = &scan_report;
        println!("Provider: {} ({})", assistant.config.provider.display_name(), assistant.config.get_base_url());
        println!("(Compile with 'butler' feature for auto-detection)\n");
        fetch_models_blocking(&mut assistant);
    }

    // =========================================================================
    // REPL loop
    // =========================================================================

    let config = ReplConfig::default();
    let mut engine = ReplEngine::new(config);

    // Sync REPL engine model with assistant
    if !assistant.config.selected_model.is_empty() {
        engine.set_model(&assistant.config.selected_model);
    }

    println!("AI Assistant CLI");
    println!("Type /help for commands, /models to list models, /exit to quit.\n");

    if let Ok(info) = update_rx.try_recv() {
        println!("  Update available: v{} \u{2192} v{} \u{2014} {}", info.current, info.latest, info.url);
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("{}", engine.config().prompt_string);
        if stdout.flush().is_err() {
            break;
        }

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {e}");
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
                    println!("{output}");
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

        // Provider scan command
        if trimmed == "/scan" {
            #[cfg(feature = "butler")]
            {
                eprint!("Re-scanning providers...");
                let mut butler = Butler::new();
                let report = butler.scan();
                let config = butler.suggest_config(&report);
                assistant.load_config(config);
                eprintln!(" done.");
                print_environment_summary(&report);

                // Build unified model list
                let unified = build_unified_model_list(&report);
                if !unified.is_empty() {
                    assistant.config.selected_model = unified[0].name.clone();
                    assistant.config.provider = unified[0].provider.clone();
                    apply_provider_url(&mut assistant, &report, &unified[0].provider);
                    assistant.available_models = unified;
                    eprintln!("{} model(s) across {} provider(s).",
                        assistant.available_models.len(),
                        report.llm_providers.len(),
                    );
                    engine.set_model(&assistant.config.selected_model);
                }
                scan_report = Some(report);
            }
            #[cfg(not(feature = "butler"))]
            {
                eprintln!("/scan requires the 'butler' feature.");
            }
            continue;
        }

        // List detected providers
        if trimmed == "/providers" {
            #[cfg(feature = "butler")]
            {
                if let Some(ref report) = scan_report {
                    if report.llm_providers.is_empty() {
                        println!("No providers detected. Use /scan to re-detect.");
                    } else {
                        println!("Detected providers:");
                        for (i, p) in report.llm_providers.iter().enumerate() {
                            let current = if p.provider_type == assistant.config.provider { " <-- active" } else { "" };
                            println!("  [{}] {} {} @ {} ({} models){}", i + 1, p.provider_type.icon(), p.name, p.url, p.available_models.len(), current);
                        }
                        println!("\nUse /use <number> to switch provider.");
                    }
                } else {
                    println!("No scan data. Use /scan first.");
                }
            }
            #[cfg(not(feature = "butler"))]
            {
                println!("Requires 'butler' feature.");
            }
            continue;
        }

        // Switch provider by index
        if trimmed.starts_with("/use ") {
            let arg = trimmed[5..].trim();
            #[cfg(feature = "butler")]
            {
                if let Some(ref report) = scan_report {
                    if let Ok(idx) = arg.parse::<usize>() {
                        if idx >= 1 && idx <= report.llm_providers.len() {
                            let p = &report.llm_providers[idx - 1];
                            assistant.config.provider = p.provider_type.clone();
                            match p.provider_type {
                                ai_assistant::AiProvider::Ollama => assistant.config.ollama_url = p.url.clone(),
                                ai_assistant::AiProvider::LMStudio => assistant.config.lm_studio_url = p.url.clone(),
                                _ => assistant.config.custom_url = p.url.clone(),
                            }
                            println!("Switched to {} {} @ {}", p.provider_type.icon(), p.name, p.url);

                            // Re-fetch models for this provider
                            assistant.config.selected_model.clear();
                            eprint!("Fetching models...");
                            assistant.fetch_models();
                            let start = Instant::now();
                            loop {
                                if assistant.poll_models() { break; }
                                if start.elapsed() > Duration::from_secs(10) {
                                    eprintln!(" timeout.");
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(50));
                            }
                            if !assistant.available_models.is_empty() {
                                eprintln!(" {} model(s).", assistant.available_models.len());
                                println!("Using model: {}", assistant.config.selected_model);
                                engine.set_model(&assistant.config.selected_model);
                            } else {
                                eprintln!(" no models found.");
                            }
                        } else {
                            println!("Invalid index. Use /providers to see the list.");
                        }
                    } else {
                        // Try as provider name
                        let provider = provider_from_name(arg);
                        assistant.config.provider = provider;
                        println!("Switched to {}", assistant.config.provider.display_name());
                        eprint!("Fetching models...");
                        assistant.fetch_models();
                        let start = Instant::now();
                        loop {
                            if assistant.poll_models() { break; }
                            if start.elapsed() > Duration::from_secs(10) {
                                eprintln!(" timeout.");
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        if !assistant.available_models.is_empty() {
                            eprintln!(" {} model(s).", assistant.available_models.len());
                            engine.set_model(&assistant.config.selected_model);
                        }
                    }
                } else {
                    // No scan report — switch by name
                    let provider = provider_from_name(arg);
                    assistant.config.provider = provider;
                    println!("Switched to {}", assistant.config.provider.display_name());
                }
            }
            #[cfg(not(feature = "butler"))]
            {
                let provider = provider_from_name(arg);
                assistant.config.provider = provider;
                println!("Switched to {}", assistant.config.provider.display_name());
            }
            continue;
        }

        // Provider info command
        if trimmed == "/provider" {
            println!("Provider:  {} {}", assistant.config.provider.icon(), assistant.config.provider.display_name());
            println!("URL:       {}", assistant.config.get_base_url());
            println!("Model:     {}", assistant.config.selected_model);
            println!("Temp:      {}", assistant.config.temperature);
            println!("History:   {} messages max", assistant.config.max_history_messages);
            continue;
        }

        let action = engine.process_input(&line);

        match action {
            ReplAction::SendMessage(msg) => {
                if assistant.config.selected_model.is_empty() {
                    eprintln!("No model selected. Use /model <name> or /scan to detect providers.");
                    continue;
                }

                engine.add_message("user", &msg);
                assistant.send_message_simple(msg);

                // Stream response to terminal
                let start = Instant::now();
                let mut full_response = String::new();
                let mut first_token = true;

                while assistant.is_generating {
                    if let Some(response) = assistant.poll_response() {
                        match response {
                            AiResponse::Chunk(text) => {
                                if first_token {
                                    first_token = false;
                                }
                                print!("{text}");
                                let _ = stdout.flush();
                                full_response.push_str(&text);
                            }
                            AiResponse::Complete(text) => {
                                // If we got chunks, the response was streamed
                                if full_response.is_empty() {
                                    // Non-streaming: print complete response
                                    print!("{text}");
                                    full_response = text;
                                } else {
                                    // Streaming finished — text may be the full
                                    // response or empty depending on provider
                                    if !text.is_empty() && full_response.is_empty() {
                                        full_response = text;
                                    }
                                }
                                break;
                            }
                            AiResponse::Error(err) => {
                                eprintln!("\n[Error] {err}");
                                break;
                            }
                            AiResponse::Cancelled(partial) => {
                                if !partial.is_empty() {
                                    full_response = partial;
                                }
                                eprintln!("\n[Cancelled]");
                                break;
                            }
                            _ => {}
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }

                // Drain any remaining messages after is_generating goes false
                while let Some(response) = assistant.poll_response() {
                    match response {
                        AiResponse::Chunk(text) => {
                            print!("{text}");
                            let _ = stdout.flush();
                            full_response.push_str(&text);
                        }
                        AiResponse::Complete(text) => {
                            if full_response.is_empty() && !text.is_empty() {
                                print!("{text}");
                                full_response = text;
                            }
                        }
                        AiResponse::Error(err) => {
                            eprintln!("\n[Error] {err}");
                        }
                        _ => {}
                    }
                }

                println!(); // newline after response

                let elapsed = start.elapsed();
                if !full_response.is_empty() {
                    engine.add_message("assistant", &full_response);
                    // Show timing
                    eprintln!("[{:.1}s]", elapsed.as_secs_f64());
                }
            }
            ReplAction::ExecuteCommand(cmd) => match cmd {
                ReplCommand::Help => {
                    println!("{}", ReplEngine::format_help());
                    println!("  /provider          Show current provider info");
                    println!("  /providers         List all detected providers");
                    println!("  /use <N|name>      Switch to provider by number or name");
                    println!("  /scan              Re-scan for providers (butler)");
                    #[cfg(feature = "containers")]
                    println!("  /docker <cmd>      Docker container management (/docker help)");
                }
                ReplCommand::Models => {
                    if assistant.available_models.is_empty() {
                        println!("No models loaded. Use /scan to detect providers.");
                    } else {
                        // Group by provider for clearer display
                        let mut by_provider: std::collections::BTreeMap<String, Vec<&ai_assistant::ModelInfo>> =
                            std::collections::BTreeMap::new();
                        for m in &assistant.available_models {
                            by_provider.entry(format!("{} {}", m.provider.icon(), m.provider.display_name()))
                                .or_default()
                                .push(m);
                        }
                        println!("Available models ({} total):", assistant.available_models.len());
                        for (provider_label, models) in &by_provider {
                            println!("  {provider_label}:");
                            for m in models {
                                let current = if m.name == assistant.config.selected_model { " <--" } else { "" };
                                let size = m.size.as_deref().unwrap_or("");
                                println!("    {} {}{}", m.name, size, current);
                            }
                        }
                        println!("\nUse /model <name> to switch (provider adjusts automatically).");
                    }
                }
                ReplCommand::Config => {
                    println!("{}", engine.format_config());
                    println!("provider = {} ({})", assistant.config.provider.display_name(), assistant.config.get_base_url());
                    println!("temperature = {}", assistant.config.temperature);
                }
                ReplCommand::Clear => {
                    engine.clear_history();
                    assistant.conversation.clear();
                    println!("History cleared.");
                }
                ReplCommand::History => {
                    let hist = engine.history();
                    if hist.is_empty() {
                        println!("(no messages yet)");
                    } else {
                        for (role, content) in hist {
                            let prefix = if role == "user" { "You" } else { "AI" };
                            // Truncate long messages in history view
                            let display = if content.len() > 200 {
                                format!("{}...", &content[..200])
                            } else {
                                content.to_string()
                            };
                            println!("{prefix}: {display}");
                        }
                    }
                }
                ReplCommand::Save(path) => {
                    if path.is_empty() {
                        println!("Usage: /save <path>");
                    } else {
                        match engine.save_session(&path) {
                            Ok(()) => println!("Session saved to {path}"),
                            Err(e) => eprintln!("Error saving session: {e}"),
                        }
                    }
                }
                ReplCommand::Load(path) => {
                    if path.is_empty() {
                        println!("Usage: /load <path>");
                    } else {
                        match engine.load_session(&path) {
                            Ok(()) => println!(
                                "Session loaded from {path} ({} messages)",
                                engine.history().len()
                            ),
                            Err(e) => eprintln!("Error loading session: {e}"),
                        }
                    }
                }
                ReplCommand::Model(name) => {
                    if name.is_empty() {
                        println!("Current model: {} {} ({})",
                            assistant.config.provider.icon(),
                            assistant.config.selected_model,
                            assistant.config.provider.display_name(),
                        );
                        println!("Use /models to list available models.");
                    } else {
                        // Check if model exists in available list
                        let found = assistant.available_models.iter().find(|m| m.name == name).cloned();
                        if let Some(m) = found {
                            assistant.config.provider = m.provider.clone();
                            assistant.config.selected_model = m.name.clone();
                            // Apply provider URL from scan data
                            #[cfg(feature = "butler")]
                            if let Some(ref report) = scan_report {
                                apply_provider_url(&mut assistant, report, &m.provider);
                            }
                            engine.set_model(&m.name);
                            println!("Model set to: {} {} ({})", m.provider.icon(), m.name, m.provider.display_name());
                        } else {
                            // Allow setting unknown models (user might know what they're doing)
                            assistant.config.selected_model = name.clone();
                            engine.set_model(&name);
                            println!("Model set to: {name} (not in available list — may fail)");
                        }
                    }
                }
                ReplCommand::Template(name) => {
                    if name.is_empty() {
                        match engine.current_template() {
                            Some(t) => println!("Current template: {t}"),
                            None => println!("No template set."),
                        }
                    } else {
                        engine.set_template(&name);
                        println!("Template set to: {name}");
                    }
                }
                ReplCommand::Cost => {
                    let session = assistant.metrics.get_session_metrics();
                    println!("Session metrics:");
                    println!("  Messages:      {}", session.message_count);
                    println!("  Input tokens:  {}", session.total_input_tokens);
                    println!("  Output tokens: {}", session.total_output_tokens);
                    println!("  Avg latency:   {:.0}ms", session.avg_response_time_ms);
                }
                ReplCommand::Unknown(cmd) => {
                    println!("Unknown command: /{cmd}. Type /help for available commands.");
                }
                ReplCommand::Exit => unreachable!("Exit handled by ReplAction::Exit"),
                _ => { eprintln!("Unknown command"); }
            },
            ReplAction::Continue => {}
            ReplAction::Exit => {
                println!("Goodbye!");
                break;
            }
            _ => { eprintln!("Unknown action"); }
        }
    }

    ExitCode::SUCCESS
}

// =============================================================================
// Blocking model fetch (for non-butler mode or fallback)
// =============================================================================

fn fetch_models_blocking(assistant: &mut AiAssistant) {
    eprint!("Fetching models...");
    assistant.fetch_models();
    let start = Instant::now();
    loop {
        if assistant.poll_models() {
            break;
        }
        if start.elapsed() > Duration::from_secs(10) {
            eprintln!(" timeout (provider may be offline).");
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if assistant.available_models.is_empty() {
        eprintln!("\nNo models found. Check that your provider is running.");
        eprintln!("  Ollama:    ollama serve    (default: http://localhost:11434)");
        eprintln!("  LM Studio: start LM Studio (default: http://localhost:1234)");
        eprintln!("  Cloud:     set API key env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)");
        eprintln!("\nYou can still chat if you set a model manually with /model <name>\n");
    } else {
        eprintln!(" {} model(s) available.", assistant.available_models.len());
        println!("Using model: {}\n", assistant.config.selected_model);
    }
}

// =============================================================================
// Unified model list from all detected providers
// =============================================================================

#[cfg(feature = "butler")]
fn build_unified_model_list(report: &EnvironmentReport) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    for provider in &report.llm_providers {
        for model_name in &provider.available_models {
            models.push(ModelInfo::new(model_name.clone(), provider.provider_type.clone()));
        }
    }
    models
}

/// Apply the correct URL for a provider based on Butler scan data.
#[cfg(feature = "butler")]
fn apply_provider_url(assistant: &mut AiAssistant, report: &EnvironmentReport, provider: &ai_assistant::AiProvider) {
    for p in &report.llm_providers {
        if p.provider_type == *provider {
            match p.provider_type {
                ai_assistant::AiProvider::Ollama => assistant.config.ollama_url = p.url.clone(),
                ai_assistant::AiProvider::LMStudio => assistant.config.lm_studio_url = p.url.clone(),
                _ => assistant.config.custom_url = p.url.clone(),
            }
            break;
        }
    }
}

// =============================================================================
// Butler environment summary
// =============================================================================

#[cfg(feature = "butler")]
fn print_environment_summary(report: &EnvironmentReport) {
    println!();
    println!("--- Environment ---");
    println!("OS:      {} ({})", report.runtime.os, report.runtime.arch);
    println!("CPUs:    {}", report.runtime.cpus);
    println!("GPU:     {}", if report.runtime.has_gpu { "detected" } else { "not detected" });
    println!("Docker:  {}", if report.runtime.has_docker { "available" } else { "not available" });

    if report.llm_providers.is_empty() {
        println!("LLM:     no providers detected");
    } else {
        println!("LLM providers:");
        for p in &report.llm_providers {
            let model_count = p.available_models.len();
            let models_str = if model_count > 0 {
                let preview: Vec<&str> = p.available_models.iter().take(3).map(|s| s.as_str()).collect();
                let suffix = if model_count > 3 { format!(" +{} more", model_count - 3) } else { String::new() };
                format!(" ({} models: {}{})", model_count, preview.join(", "), suffix)
            } else {
                String::new()
            };
            println!("  {} {} @ {}{}", p.provider_type.icon(), p.name, p.url, models_str);
        }
    }
    println!("-------------------\n");
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.create(image, &name, opts) {
                Ok(id) => format!("Created container {} (image: {image}, name: {name})", &id[..12.min(id.len())]),
                Err(e) => format!("Error: {e}"),
            }
        }

        "start" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker start <container_id>".to_string(),
            };
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.start(id) {
                Ok(()) => format!("Started container {id}"),
                Err(e) => format!("Error: {e}"),
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.stop(id, timeout) {
                Ok(()) => format!("Stopped container {id}"),
                Err(e) => format!("Error: {e}"),
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.remove(id, force) {
                Ok(()) => format!("Removed container {id}"),
                Err(e) => format!("Error: {e}"),
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
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
                Err(e) => format!("Error: {e}"),
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
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.logs(id, tail) {
                Ok(logs) => {
                    if logs.is_empty() { "(no logs)".to_string() } else { logs }
                }
                Err(e) => format!("Error: {e}"),
            }
        }

        "status" => {
            let id = match parts.get(2) {
                Some(id) => *id,
                None => return "Usage: /docker status <container_id>".to_string(),
            };
            let guard = match executor.read() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            match guard.status(id) {
                Some(status) => format!("Container {id}: {status}"),
                None => format!("Container {id} not found"),
            }
        }

        "cleanup" => {
            let mut guard = match executor.write() {
                Ok(g) => g,
                Err(e) => return format!("Error: lock poisoned: {e}"),
            };
            let count = guard.cleanup_all();
            format!("Cleaned up {count} container(s)")
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
