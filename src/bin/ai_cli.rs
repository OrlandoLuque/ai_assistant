//! Non-interactive CLI tool for the AI assistant.
//!
//! Run with: `cargo run --bin ai_cli --features "full,butler" -- <command>`
//!
//! Unlike `ai_assistant_cli` (interactive REPL), this binary executes a single
//! command and exits — suitable for scripting, CI/CD, and quick one-shot queries.
//!
//! ## Commands
//!
//! ```text
//! ai_cli scan                           Detect providers and show environment
//! ai_cli providers                      List detected LLM providers
//! ai_cli models [--provider <name>]     List available models
//! ai_cli config show [<file>]           Show current or file-based config
//! ai_cli config check <file>            Validate a config file
//! ai_cli config set <file> [options]    Modify a config file
//! ai_cli butler [--config <file>]       Run Butler advisor scan
//! ai_cli query [options] <prompt>       One-shot LLM query
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{ExitCode, Stdio};
use std::time::{Duration, Instant};

use ai_assistant::{AiAssistant, AiConfig, AiResponse};

#[cfg(feature = "butler")]
use ai_assistant::butler::{Butler, ButlerAdvisor, EnvironmentReport};
#[cfg(feature = "butler")]
use ai_assistant::ModelInfo;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_usage();
        return ExitCode::from(1);
    }

    match args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "scan" => cmd_scan(),
        "providers" => cmd_providers(),
        "models" => cmd_models(&args[1..]),
        "config" => cmd_config(&args[1..]),
        "butler" => cmd_butler(&args[1..]),
        "query" => cmd_query(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "test" => cmd_test(&args[1..]),
        other => {
            eprintln!("Error: unknown command '{}'\n", other);
            print_usage();
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// Usage
// =============================================================================

fn print_usage() {
    println!("ai_cli — Non-interactive CLI for AI Assistant\n");
    println!("Usage: ai_cli <command> [options]\n");
    println!("Commands:");
    println!("  scan                           Detect LLM providers, show environment info");
    println!("  providers                      List detected LLM providers with model counts");
    println!("  models [--provider <name>]     List available models (optionally filtered)");
    println!("  config show [<file>]           Show config (defaults or from JSON file)");
    println!("  config check <file>            Validate a JSON config file");
    println!("  config set <file> [options]    Modify config values in a JSON file");
    println!("    --provider <name>              Set the provider");
    println!("    --model <name>                 Set the model");
    println!("    --url <url>                    Set provider URL");
    println!("    --temperature <float>          Set temperature (0.0-2.0)");
    println!("    --max-history <n>              Set max history messages");
    println!("  butler [--config <file>]       Run Butler advisor (optimization recommendations)");
    println!("  query [options] <prompt>       Send a one-shot query to an LLM");
    println!("  bench [options]                Run Criterion benchmarks (44 benchmarks)");
    println!("    --filter <pattern>             Filter benchmarks by name");
    println!("    --list                         List available benchmarks");
    println!("    --output <dir>                 Output directory (default: results/)");
    println!("  test [options]                 Run tests (lib or harness), save results");
    println!("    --all                          Run test harness (all categories)");
    println!("    --category <name>              Run specific harness category");
    println!("    --list                         List harness categories");
    println!("    --lib                          Run cargo test --lib (default)");
    println!("    --filter <pattern>             Filter test names");
    println!("    --features <flags>             Override feature flags");
    println!("    --output <dir>                 Output directory (default: results/)");
    println!("    --provider <name>              Provider (ollama, openai, anthropic, gemini, ...)");
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!("    --config <file>                Load config from JSON file");
    println!("    --system <prompt>              System prompt");
    println!("    --file <path>                  Read user prompt from file instead of argument");
    println!("    --knowledge <path>             Inject file content as knowledge context");
    println!("    --json                         Output response as JSON");
    println!("    --temperature <float>          Temperature (0.0-2.0)");
    println!();
    println!("Examples:");
    println!("  ai_cli scan");
    println!("  ai_cli models --provider ollama");
    println!("  ai_cli config show");
    println!("  ai_cli config set myconfig.json --provider openai --model gpt-4o");
    println!("  ai_cli butler");
    println!("  ai_cli query \"What is Rust?\"");
    println!("  ai_cli query --provider openai --model gpt-4o \"Explain ownership\"");
    println!("  ai_cli query --config myconfig.json --file prompt.txt");
    println!("  ai_cli query --system \"You are a Rust expert\" \"How do lifetimes work?\"");
    println!("  ai_cli bench");
    println!("  ai_cli bench --filter rag");
    println!("  ai_cli test --all");
    println!("  ai_cli test --category security");
    println!("  ai_cli test --list");
}

// =============================================================================
// scan — detect environment
// =============================================================================

fn cmd_scan() -> ExitCode {
    #[cfg(feature = "butler")]
    {
        let mut butler = Butler::new();
        let report = butler.scan();
        print_environment(&report);
        ExitCode::SUCCESS
    }
    #[cfg(not(feature = "butler"))]
    {
        eprintln!("Error: 'scan' requires the 'butler' feature.");
        eprintln!("  cargo run --bin ai_cli --features \"full,butler\" -- scan");
        ExitCode::from(1)
    }
}

// =============================================================================
// providers — list providers
// =============================================================================

fn cmd_providers() -> ExitCode {
    #[cfg(feature = "butler")]
    {
        let mut butler = Butler::new();
        let report = butler.scan();

        if report.llm_providers.is_empty() {
            println!("No LLM providers detected.");
            println!();
            println!("Install one:");
            println!("  Ollama:    https://ollama.com  (then: ollama serve)");
            println!("  LM Studio: https://lmstudio.ai");
            println!("  Cloud:     set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.");
            return ExitCode::from(1);
        }

        println!("Detected {} provider(s):\n", report.llm_providers.len());
        for (i, p) in report.llm_providers.iter().enumerate() {
            println!(
                "  [{}] {} {} @ {}",
                i + 1,
                p.provider_type.icon(),
                p.name,
                p.url,
            );
            if p.available_models.is_empty() {
                println!("      Models: (none detected)");
            } else {
                println!("      Models: {} available", p.available_models.len());
                for m in p.available_models.iter().take(5) {
                    println!("        - {}", m);
                }
                if p.available_models.len() > 5 {
                    println!("        ... +{} more", p.available_models.len() - 5);
                }
            }
            println!();
        }
        ExitCode::SUCCESS
    }
    #[cfg(not(feature = "butler"))]
    {
        eprintln!("Error: 'providers' requires the 'butler' feature.");
        ExitCode::from(1)
    }
}

// =============================================================================
// models — list models
// =============================================================================

fn cmd_models(args: &[String]) -> ExitCode {
    let mut provider_filter: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --provider requires a value");
                    return ExitCode::from(1);
                }
                provider_filter = Some(args[i].to_lowercase());
            }
            _ => {
                eprintln!("Error: unknown option '{}' for 'models'", args[i]);
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    #[cfg(feature = "butler")]
    {
        let mut butler = Butler::new();
        let report = butler.scan();

        if report.llm_providers.is_empty() {
            println!("No providers detected. Run 'ai_cli providers' for setup instructions.");
            return ExitCode::from(1);
        }

        let models = build_unified_model_list(&report);
        let filtered: Vec<&ModelInfo> = if let Some(ref filter) = provider_filter {
            models
                .iter()
                .filter(|m| {
                    let pname = format!("{:?}", m.provider).to_lowercase();
                    pname.contains(filter)
                })
                .collect()
        } else {
            models.iter().collect()
        };

        if filtered.is_empty() {
            if let Some(ref filter) = provider_filter {
                println!("No models found for provider '{}'.", filter);
            } else {
                println!("Providers detected but no models installed.");
                println!("  Ollama: ollama pull llama3.2");
                println!("  LM Studio: download a model from the app");
            }
            return ExitCode::from(1);
        }

        println!("{} model(s) available:\n", filtered.len());
        for (i, m) in filtered.iter().enumerate() {
            println!("  [{}] {} ({})", i + 1, m.name, m.provider.display_name());
        }
        println!();
        ExitCode::SUCCESS
    }
    #[cfg(not(feature = "butler"))]
    {
        let _ = provider_filter;
        // Without butler, try direct fetch
        let mut assistant = AiAssistant::new();
        fetch_models_blocking(&mut assistant);
        if assistant.available_models.is_empty() {
            return ExitCode::from(1);
        }
        println!("{} model(s) available:\n", assistant.available_models.len());
        for (i, m) in assistant.available_models.iter().enumerate() {
            println!("  [{}] {} ({})", i + 1, m.name, m.provider.display_name());
        }
        println!();
        ExitCode::SUCCESS
    }
}

// =============================================================================
// config — show / check / set
// =============================================================================

fn cmd_config(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_cli config <show|check|set> [options]");
        return ExitCode::from(1);
    }

    match args[0].as_str() {
        "show" => cmd_config_show(args.get(1).map(|s| s.as_str())),
        "check" => {
            if args.len() < 2 {
                eprintln!("Usage: ai_cli config check <file.json>");
                return ExitCode::from(1);
            }
            cmd_config_check(&args[1])
        }
        "set" => {
            if args.len() < 2 {
                eprintln!("Usage: ai_cli config set <file.json> [--provider X] [--model Y] ...");
                return ExitCode::from(1);
            }
            cmd_config_set(&args[1], &args[2..])
        }
        other => {
            eprintln!("Error: unknown config subcommand '{}'. Use show, check, or set.", other);
            ExitCode::from(1)
        }
    }
}

fn cmd_config_show(file: Option<&str>) -> ExitCode {
    let config = if let Some(path) = file {
        match load_config(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        AiConfig::default()
    };

    println!("Configuration{}:\n", file.map(|f| format!(" ({})", f)).unwrap_or_default());
    println!("  provider:          {:?}", config.provider);
    println!("  selected_model:    {}", config.selected_model);
    println!("  temperature:       {}", config.temperature);
    println!("  max_history:       {}", config.max_history_messages);
    println!("  ollama_url:        {}", config.ollama_url);
    println!("  lm_studio_url:     {}", config.lm_studio_url);
    println!("  custom_url:        {}", config.custom_url);
    println!(
        "  api_key:           {}",
        if config.api_key.is_empty() {
            "(not set — will use env vars)"
        } else {
            "(set)"
        }
    );
    println!();
    ExitCode::SUCCESS
}

fn cmd_config_check(path: &str) -> ExitCode {
    match load_config(path) {
        Ok(config) => {
            println!("Config '{}': VALID\n", path);
            let mut warnings = Vec::new();

            if config.selected_model.is_empty() {
                warnings.push("  - No model selected (selected_model is empty)");
            }
            if config.temperature < 0.0 || config.temperature > 2.0 {
                warnings.push("  - Temperature out of range (expected 0.0-2.0)");
            }
            if config.max_history_messages == 0 {
                warnings.push("  - max_history_messages is 0 (no conversation history)");
            }

            // Check provider-specific config
            match config.provider {
                ai_assistant::AiProvider::OpenAI
                | ai_assistant::AiProvider::Anthropic
                | ai_assistant::AiProvider::Gemini
                | ai_assistant::AiProvider::Groq
                | ai_assistant::AiProvider::DeepSeek
                | ai_assistant::AiProvider::Mistral => {
                    if config.api_key.is_empty() {
                        warnings.push(
                            "  - Cloud provider selected but api_key is empty (will try env vars)",
                        );
                    }
                }
                _ => {}
            }

            if warnings.is_empty() {
                println!("  No warnings.");
            } else {
                println!("  Warnings:");
                for w in &warnings {
                    println!("{}", w);
                }
            }
            println!();
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Config '{}': INVALID\n", path);
            eprintln!("  Error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_config_set(path: &str, args: &[String]) -> ExitCode {
    // Load existing or create default
    let mut config = if PathBuf::from(path).exists() {
        match load_config(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        AiConfig::default()
    };

    let mut changes = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --provider requires a value");
                    return ExitCode::from(1);
                }
                let new_provider = provider_from_name(&args[i]);
                changes.push(format!("provider: {:?} -> {:?}", config.provider, new_provider));
                config.provider = new_provider;
            }
            "--model" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --model requires a value");
                    return ExitCode::from(1);
                }
                changes.push(format!(
                    "model: '{}' -> '{}'",
                    config.selected_model, args[i]
                ));
                config.selected_model = args[i].clone();
            }
            "--url" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --url requires a value");
                    return ExitCode::from(1);
                }
                changes.push(format!("custom_url: '{}' -> '{}'", config.custom_url, args[i]));
                config.custom_url = args[i].clone();
            }
            "--temperature" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --temperature requires a value");
                    return ExitCode::from(1);
                }
                match args[i].parse::<f32>() {
                    Ok(t) => {
                        changes.push(format!("temperature: {} -> {}", config.temperature, t));
                        config.temperature = t;
                    }
                    Err(_) => {
                        eprintln!("Error: invalid temperature '{}'", args[i]);
                        return ExitCode::from(1);
                    }
                }
            }
            "--max-history" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --max-history requires a value");
                    return ExitCode::from(1);
                }
                match args[i].parse::<usize>() {
                    Ok(n) => {
                        changes.push(format!(
                            "max_history: {} -> {}",
                            config.max_history_messages, n
                        ));
                        config.max_history_messages = n;
                    }
                    Err(_) => {
                        eprintln!("Error: invalid number '{}'", args[i]);
                        return ExitCode::from(1);
                    }
                }
            }
            other => {
                eprintln!("Error: unknown option '{}' for 'config set'", other);
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    if changes.is_empty() {
        eprintln!("No changes specified. Use --provider, --model, --url, --temperature, or --max-history.");
        return ExitCode::from(1);
    }

    match save_config(path, &config) {
        Ok(()) => {
            println!("Updated '{}':", path);
            for c in &changes {
                println!("  {}", c);
            }
            println!();
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error saving '{}': {}", path, e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// butler — advisor scan
// =============================================================================

fn cmd_butler(args: &[String]) -> ExitCode {
    #[cfg(feature = "butler")]
    {
        let mut config_path: Option<&str> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--config" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("Error: --config requires a file path");
                        return ExitCode::from(1);
                    }
                    config_path = Some(&args[i]);
                }
                other => {
                    eprintln!("Error: unknown option '{}' for 'butler'", other);
                    return ExitCode::from(1);
                }
            }
            i += 1;
        }

        eprint!("Scanning environment...");
        let mut butler = Butler::new();
        let report = butler.scan();
        eprintln!(" done.");

        let advisor_report = if let Some(path) = config_path {
            match load_config(path) {
                Ok(config) => {
                    let advisor_config = ai_assistant::butler::AdvisorConfig::default();
                    let _ = config; // Config loaded for validation; advisor uses its own config
                    ButlerAdvisor::with_config(&report, &advisor_config).analyze()
                }
                Err(e) => {
                    eprintln!("Warning: could not load '{}': {}. Using defaults.", path, e);
                    ButlerAdvisor::new(&report).analyze()
                }
            }
        } else {
            ButlerAdvisor::new(&report).analyze()
        };

        // Print environment summary
        print_environment(&report);

        // Print recommendations
        let pending = advisor_report.pending();
        let summary = &advisor_report.summary;

        println!("--- Butler Advisor ---");
        println!(
            "Recommendations: {} total, {} already enabled, {} pending\n",
            summary.total,
            summary.already_enabled,
            pending.len(),
        );

        if pending.is_empty() {
            println!("  All recommendations satisfied.");
        } else {
            for (i, rec) in pending.iter().enumerate() {
                println!(
                    "  [{}] [{:?}] {:?}: {}",
                    i + 1,
                    rec.priority,
                    rec.category,
                    rec.title,
                );
                println!("       {}", rec.description);
                println!("       Action: {}", rec.action);
                if let Some(ref flag) = rec.feature_flag {
                    println!("       Feature: {}", flag);
                }
                println!();
            }
        }
        println!("---------------------\n");
        ExitCode::SUCCESS
    }
    #[cfg(not(feature = "butler"))]
    {
        let _ = args;
        eprintln!("Error: 'butler' requires the 'butler' feature.");
        eprintln!("  cargo run --bin ai_cli --features \"full,butler\" -- butler");
        ExitCode::from(1)
    }
}

// =============================================================================
// query — one-shot LLM query
// =============================================================================

fn cmd_query(args: &[String]) -> ExitCode {
    let mut provider_name: Option<String> = None;
    let mut model_name: Option<String> = None;
    let mut url_override: Option<String> = None;
    let mut config_file: Option<String> = None;
    let mut system_prompt: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut knowledge_path: Option<String> = None;
    let mut json_output = false;
    let mut temperature: Option<f32> = None;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --provider requires a value");
                    return ExitCode::from(1);
                }
                provider_name = Some(args[i].clone());
            }
            "--model" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --model requires a value");
                    return ExitCode::from(1);
                }
                model_name = Some(args[i].clone());
            }
            "--url" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --url requires a value");
                    return ExitCode::from(1);
                }
                url_override = Some(args[i].clone());
            }
            "--config" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --config requires a file path");
                    return ExitCode::from(1);
                }
                config_file = Some(args[i].clone());
            }
            "--system" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --system requires a value");
                    return ExitCode::from(1);
                }
                system_prompt = Some(args[i].clone());
            }
            "--file" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --file requires a file path");
                    return ExitCode::from(1);
                }
                prompt_file = Some(args[i].clone());
            }
            "--knowledge" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --knowledge requires a path");
                    return ExitCode::from(1);
                }
                knowledge_path = Some(args[i].clone());
            }
            "--json" => {
                json_output = true;
            }
            "--temperature" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --temperature requires a value");
                    return ExitCode::from(1);
                }
                match args[i].parse::<f32>() {
                    Ok(t) => temperature = Some(t),
                    Err(_) => {
                        eprintln!("Error: invalid temperature '{}'", args[i]);
                        return ExitCode::from(1);
                    }
                }
            }
            arg if arg.starts_with('-') => {
                eprintln!("Error: unknown option '{}' for 'query'", arg);
                return ExitCode::from(1);
            }
            _ => {
                prompt_parts.push(args[i].clone());
            }
        }
        i += 1;
    }

    // Resolve the user prompt
    let user_prompt = if let Some(ref path) = prompt_file {
        match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading prompt file '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    } else if !prompt_parts.is_empty() {
        prompt_parts.join(" ")
    } else {
        eprintln!("Error: no prompt provided. Pass it as argument or use --file <path>.");
        return ExitCode::from(1);
    };

    // Build assistant
    let mut assistant = if let Some(ref sp) = system_prompt {
        AiAssistant::with_system_prompt(sp)
    } else {
        AiAssistant::new()
    };

    // Load config file if provided
    if let Some(ref path) = config_file {
        match load_config(path) {
            Ok(config) => assistant.load_config(config),
            Err(e) => {
                eprintln!("Error loading config '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    }

    // Apply overrides (take precedence over config file)
    if let Some(ref name) = provider_name {
        assistant.config.provider = provider_from_name(name);
    }
    if let Some(ref name) = model_name {
        assistant.config.selected_model = name.clone();
    }
    if let Some(ref url) = url_override {
        match assistant.config.provider {
            ai_assistant::AiProvider::Ollama => assistant.config.ollama_url = url.clone(),
            ai_assistant::AiProvider::LMStudio => assistant.config.lm_studio_url = url.clone(),
            _ => assistant.config.custom_url = url.clone(),
        }
    }
    if let Some(t) = temperature {
        assistant.config.temperature = t;
    }

    // If no model set, try auto-detection
    if assistant.config.selected_model.is_empty() {
        #[cfg(feature = "butler")]
        {
            eprint!("Auto-detecting providers...");
            let mut butler = Butler::new();
            let report = butler.scan();
            eprintln!(" done.");

            let models = build_unified_model_list(&report);
            if !models.is_empty() {
                assistant.config.selected_model = models[0].name.clone();
                assistant.config.provider = models[0].provider.clone();
                apply_provider_url(&mut assistant, &report, &models[0].provider);
                eprintln!(
                    "Using: {} ({})",
                    models[0].name,
                    models[0].provider.display_name()
                );
            } else if !report.llm_providers.is_empty() {
                // Providers found but no models — try HTTP fetch
                fetch_models_blocking(&mut assistant);
            }
        }

        #[cfg(not(feature = "butler"))]
        {
            fetch_models_blocking(&mut assistant);
        }

        if assistant.config.selected_model.is_empty() {
            eprintln!("Error: no model available. Specify --provider and --model, or install a model.");
            return ExitCode::from(1);
        }
    }

    // Load knowledge context
    let knowledge = if let Some(ref path) = knowledge_path {
        match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading knowledge file '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        String::new()
    };

    // Send query
    let start = Instant::now();
    assistant.send_message(user_prompt, &knowledge);

    // Poll for response
    let mut full_response = String::new();
    let mut errored = false;

    loop {
        if let Some(response) = assistant.poll_response() {
            match response {
                AiResponse::Chunk(text) => {
                    if !json_output {
                        print!("{}", text);
                        let _ = std::io::stdout().flush();
                    }
                    full_response.push_str(&text);
                }
                AiResponse::Complete(text) => {
                    if !json_output {
                        if !text.is_empty() && full_response.is_empty() {
                            print!("{}", text);
                        }
                        println!();
                    }
                    if full_response.is_empty() {
                        full_response = text;
                    }
                    break;
                }
                AiResponse::Error(e) => {
                    eprintln!("\nError: {}", e);
                    errored = true;
                    break;
                }
                AiResponse::Cancelled(partial) => {
                    full_response = partial;
                    break;
                }
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    // Drain any remaining messages
    for _ in 0..50 {
        if assistant.poll_response().is_none() {
            break;
        }
    }

    let elapsed = start.elapsed();

    if json_output {
        let json = serde_json::json!({
            "provider": format!("{:?}", assistant.config.provider),
            "model": assistant.config.selected_model,
            "response": full_response,
            "elapsed_ms": elapsed.as_millis(),
            "error": errored,
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
    } else {
        eprintln!(
            "\n[{:?} / {} / {:.1}s]",
            assistant.config.provider,
            assistant.config.selected_model,
            elapsed.as_secs_f64()
        );
    }

    if errored {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

// =============================================================================
// bench — run Criterion benchmarks with output capture
// =============================================================================

fn cmd_bench(args: &[String]) -> ExitCode {
    let mut cargo_args = vec![
        "bench".to_string(),
        "--bench".to_string(),
        "core_benchmarks".to_string(),
        "--features".to_string(),
        "full,constrained-decoding,multi-agent,distributed".to_string(),
    ];
    let mut output_dir = PathBuf::from("results");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--filter" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --filter requires a pattern");
                    return ExitCode::from(1);
                }
                cargo_args.push("--".to_string());
                cargo_args.push(args[i].clone());
            }
            "--list" => {
                cargo_args.push("--".to_string());
                cargo_args.push("--list".to_string());
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a directory");
                    return ExitCode::from(1);
                }
                output_dir = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("Error: unknown option '{}' for 'bench'", other);
                eprintln!("Usage: ai_cli bench [--filter <pattern>] [--list] [--output <dir>]");
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let cmd_str = format!("cargo {}", cargo_args.join(" "));
    run_and_capture("bench", &cmd_str, "cargo", &cargo_args, &output_dir)
}

// =============================================================================
// test — run test harness or cargo test with output capture
// =============================================================================

fn cmd_test(args: &[String]) -> ExitCode {
    let mut harness_mode = false;
    let mut harness_args: Vec<String> = Vec::new();
    let mut cargo_features =
        "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite".to_string();
    let mut test_filter: Option<String> = None;
    let mut output_dir = PathBuf::from("results");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => {
                harness_mode = true;
                harness_args.push("--all".to_string());
            }
            "--category" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --category requires a name");
                    return ExitCode::from(1);
                }
                harness_mode = true;
                harness_args.push(format!("--category={}", args[i]));
            }
            "--list" => {
                harness_mode = true;
                harness_args.push("--list".to_string());
            }
            "--no-color" => {
                harness_args.push("--no-color".to_string());
            }
            "--json" => {
                harness_args.push("--json".to_string());
            }
            "--harness" => {
                harness_mode = true;
            }
            "--lib" => {
                harness_mode = false;
            }
            "--features" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --features requires a value");
                    return ExitCode::from(1);
                }
                cargo_features = args[i].clone();
            }
            "--filter" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --filter requires a pattern");
                    return ExitCode::from(1);
                }
                test_filter = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --output requires a directory");
                    return ExitCode::from(1);
                }
                output_dir = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("Error: unknown option '{}' for 'test'", other);
                eprintln!("Usage: ai_cli test [--all|--category <name>|--list|--lib] [--filter <pat>] [--features <f>] [--output <dir>]");
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let cargo_args = if harness_mode {
        let mut a = vec![
            "run".to_string(),
            "--bin".to_string(),
            "ai_test_harness".to_string(),
            "--features".to_string(),
            cargo_features,
            "--".to_string(),
        ];
        // Force no-color in harness for clean log files
        if !harness_args.contains(&"--no-color".to_string()) {
            a.push("--no-color".to_string());
        }
        a.extend(harness_args);
        a
    } else {
        let mut a = vec![
            "test".to_string(),
            "--features".to_string(),
            cargo_features,
            "--lib".to_string(),
        ];
        if let Some(ref filter) = test_filter {
            a.push("--".to_string());
            a.push(filter.clone());
        }
        a
    };

    let label = if harness_mode { "test-harness" } else { "test-lib" };
    let cmd_str = format!("cargo {}", cargo_args.join(" "));
    run_and_capture(label, &cmd_str, "cargo", &cargo_args, &output_dir)
}

// =============================================================================
// Shared: run a command, tee output to terminal + log file
// =============================================================================

fn run_and_capture(
    label: &str,
    display_cmd: &str,
    program: &str,
    args: &[String],
    output_dir: &PathBuf,
) -> ExitCode {
    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output dir '{}': {}", output_dir.display(), e);
        return ExitCode::from(1);
    }

    // Generate timestamped filename
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Format as YYYYMMDD_HHMMSS (UTC-approximation from epoch)
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;
    // Approximate date from epoch days (good enough for filenames)
    let (year, month, day) = epoch_days_to_date(days);
    let timestamp = format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        year, month, day, hours, minutes, seconds
    );

    let log_file = output_dir.join(format!("{}_{}.log", label, timestamp));

    println!("Command: {}", display_cmd);
    println!("Log:     {}\n", log_file.display());

    // Spawn process with piped stdout+stderr
    let mut child = match std::process::Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error spawning '{}': {}", program, e);
            return ExitCode::from(1);
        }
    };

    let start = Instant::now();
    let mut all_output = Vec::new();

    // Header for log file
    let header = format!(
        "# ai_cli {} — {}\n# Command: {}\n# Date: {}\n# Platform: {} {}\n{}\n",
        label,
        timestamp,
        display_cmd,
        timestamp,
        std::env::consts::OS,
        std::env::consts::ARCH,
        "=".repeat(78),
    );
    all_output.extend_from_slice(header.as_bytes());

    // Read stdout in a separate thread, stderr in main thread
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let stdout_handle = std::thread::spawn(move || {
        let mut lines = Vec::new();
        if let Some(out) = stdout {
            let reader = BufReader::new(out);
            for line in reader.lines() {
                if let Ok(l) = line {
                    println!("{}", l);
                    lines.push(l);
                }
            }
        }
        lines
    });

    let stderr_handle = std::thread::spawn(move || {
        let mut lines = Vec::new();
        if let Some(err) = stderr {
            let reader = BufReader::new(err);
            for line in reader.lines() {
                if let Ok(l) = line {
                    eprintln!("{}", l);
                    lines.push(l);
                }
            }
        }
        lines
    });

    let status = child.wait();
    let elapsed = start.elapsed();

    let stdout_lines = stdout_handle.join().unwrap_or_default();
    let stderr_lines = stderr_handle.join().unwrap_or_default();

    // Build log content
    if !stdout_lines.is_empty() {
        all_output.extend_from_slice(b"\n--- STDOUT ---\n");
        for line in &stdout_lines {
            all_output.extend_from_slice(line.as_bytes());
            all_output.push(b'\n');
        }
    }
    if !stderr_lines.is_empty() {
        all_output.extend_from_slice(b"\n--- STDERR ---\n");
        for line in &stderr_lines {
            all_output.extend_from_slice(line.as_bytes());
            all_output.push(b'\n');
        }
    }

    let exit_code = match &status {
        Ok(s) => s.code().unwrap_or(-1),
        Err(_) => -1,
    };

    // Footer with summary
    let footer = format!(
        "\n{}\n# Finished: exit code {}, elapsed {:.2}s\n# Total output: {} stdout lines, {} stderr lines\n",
        "=".repeat(78),
        exit_code,
        elapsed.as_secs_f64(),
        stdout_lines.len(),
        stderr_lines.len(),
    );
    all_output.extend_from_slice(footer.as_bytes());

    // Write log file
    match std::fs::write(&log_file, &all_output) {
        Ok(()) => {
            let size_kb = all_output.len() / 1024;
            eprintln!(
                "\nResults saved to: {} ({} KB, {:.1}s)",
                log_file.display(),
                size_kb,
                elapsed.as_secs_f64(),
            );
        }
        Err(e) => {
            eprintln!("Warning: could not write log file '{}': {}", log_file.display(), e);
        }
    }

    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("Error waiting for process: {}", e);
            ExitCode::from(1)
        }
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn epoch_days_to_date(mut days: u64) -> (u64, u64, u64) {
    // Simplified Gregorian calendar calculation
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &months_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// =============================================================================
// Helper functions
// =============================================================================

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
        other => {
            eprintln!("Warning: unknown provider '{}', defaulting to Ollama", other);
            ai_assistant::AiProvider::Ollama
        }
    }
}

fn load_config(path: &str) -> Result<AiConfig, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {}", e))
}

fn save_config(path: &str, config: &AiConfig) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(config).map_err(|e| format!("serialize error: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("cannot write file: {}", e))
}

fn fetch_models_blocking(assistant: &mut AiAssistant) {
    eprint!("Fetching models...");
    assistant.fetch_models();
    let start = Instant::now();
    loop {
        if assistant.poll_models() {
            break;
        }
        if start.elapsed() > Duration::from_secs(10) {
            eprintln!(" timeout.");
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if assistant.available_models.is_empty() {
        eprintln!(" no models found.");
    } else {
        eprintln!(" {} model(s).", assistant.available_models.len());
    }
}

#[cfg(feature = "butler")]
fn build_unified_model_list(report: &EnvironmentReport) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    for provider in &report.llm_providers {
        for model_name in &provider.available_models {
            models.push(ModelInfo::new(
                model_name.clone(),
                provider.provider_type.clone(),
            ));
        }
    }
    models
}

#[cfg(feature = "butler")]
fn apply_provider_url(
    assistant: &mut AiAssistant,
    report: &EnvironmentReport,
    provider: &ai_assistant::AiProvider,
) {
    for p in &report.llm_providers {
        if p.provider_type == *provider {
            match p.provider_type {
                ai_assistant::AiProvider::Ollama => {
                    assistant.config.ollama_url = p.url.clone();
                }
                ai_assistant::AiProvider::LMStudio => {
                    assistant.config.lm_studio_url = p.url.clone();
                }
                _ => {
                    assistant.config.custom_url = p.url.clone();
                }
            }
            break;
        }
    }
}

#[cfg(feature = "butler")]
fn print_environment(report: &EnvironmentReport) {
    println!();
    println!("--- Environment ---");
    println!(
        "OS:      {} ({})",
        report.runtime.os, report.runtime.arch
    );
    println!("CPUs:    {}", report.runtime.cpus);
    println!(
        "GPU:     {}",
        if report.runtime.has_gpu {
            "detected"
        } else {
            "not detected"
        }
    );
    println!(
        "Docker:  {}",
        if report.runtime.has_docker {
            "available"
        } else {
            "not available"
        }
    );

    if report.llm_providers.is_empty() {
        println!("LLM:     no providers detected");
    } else {
        println!("LLM providers:");
        for p in &report.llm_providers {
            let model_count = p.available_models.len();
            let models_str = if model_count > 0 {
                let preview: Vec<&str> = p
                    .available_models
                    .iter()
                    .take(3)
                    .map(|s| s.as_str())
                    .collect();
                let suffix = if model_count > 3 {
                    format!(" +{} more", model_count - 3)
                } else {
                    String::new()
                };
                format!(" ({} models: {}{})", model_count, preview.join(", "), suffix)
            } else {
                " (no models)".to_string()
            };
            println!(
                "  {} {} @ {}{}",
                p.provider_type.icon(),
                p.name,
                p.url,
                models_str
            );
        }
    }
    println!("-------------------\n");
}
