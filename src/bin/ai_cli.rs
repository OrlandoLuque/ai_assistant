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

    // Parse global flags before command dispatch
    let mut verbose_level = 0u8;
    let mut log_file: Option<String> = None;
    let mut command_args: Vec<String> = Vec::new();
    let mut found_command = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-v" | "--verbose" if !found_command => verbose_level = verbose_level.max(1),
            "-vv" if !found_command => verbose_level = verbose_level.max(2),
            "-vvv" if !found_command => verbose_level = verbose_level.max(3),
            "--debug" if !found_command => verbose_level = verbose_level.max(2),
            "--log-file" if !found_command => {
                i += 1;
                if i < args.len() {
                    log_file = Some(args[i].clone());
                }
            }
            _ => {
                found_command = true;
                command_args.push(args[i].clone());
            }
        }
        i += 1;
    }

    // Initialize log backend (requires diagnostic-logging feature for env_logger)
    #[cfg(feature = "diagnostic-logging")]
    {
        let log_level = match verbose_level {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        let mut log_builder =
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level));
        if let Some(ref path) = log_file {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path);
            match file {
                Ok(file) => {
                    log_builder.target(env_logger::Target::Pipe(Box::new(file)));
                }
                Err(e) => {
                    eprintln!("Warning: could not open log file '{}': {}", path, e);
                }
            }
        }
        log_builder.init();
        log::debug!(
            "ai_cli starting, log_level={}, args={:?}",
            log_level,
            command_args
        );
    }
    #[cfg(not(feature = "diagnostic-logging"))]
    {
        let _ = verbose_level;
        let _ = log_file;
    }

    if command_args.is_empty() {
        print_usage();
        return ExitCode::from(1);
    }

    // Background update check
    let update_rx = ai_assistant::update_checker::check_for_update_bg(env!("CARGO_PKG_VERSION"));

    let result = match command_args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "scan" => cmd_scan(),
        "providers" => cmd_providers(),
        "models" => cmd_models(&command_args[1..]),
        "config" => cmd_config(&command_args[1..]),
        "butler" => cmd_butler(&command_args[1..]),
        "query" => cmd_query(&command_args[1..]),
        "bench" => cmd_bench(&command_args[1..]),
        "test" => cmd_test(&command_args[1..]),
        "cost" => cmd_cost(&command_args[1..]),
        "verify" => cmd_verify(&command_args[1..]),
        #[cfg(feature = "vision")]
        "vision-check" => cmd_vision_check(&command_args[1..]),
        #[cfg(feature = "research")]
        "research" => cmd_research(&command_args[1..]),
        "quality" => cmd_quality(&command_args[1..]),
        "tool" => cmd_tool(&command_args[1..]),
        "workflow" => cmd_workflow(&command_args[1..]),
        "benchmark" => cmd_benchmark(&command_args[1..]),
        "recipes" => cmd_recipes(&command_args[1..]),
        other => {
            eprintln!("Error: unknown command '{}'\n", other);
            print_usage();
            ExitCode::from(1)
        }
    };

    // Check for updates before exit
    if let Ok(info) = update_rx.try_recv() {
        eprintln!();
        eprintln!(
            "  Update available: v{} \u{2192} v{}",
            info.current, info.latest
        );
        eprintln!("  Download: {}", info.url);
        eprintln!();
    }

    result
}

// =============================================================================
// Usage
// =============================================================================

fn print_usage() {
    println!("ai_cli — Non-interactive CLI for AI Assistant\n");
    println!("Usage: ai_cli [global-options] <command> [options]\n");
    println!("Global options:");
    println!("  -v, --verbose                    Info-level logging");
    println!("  -vv, --debug                     Debug-level logging");
    println!("  -vvv                             Trace-level logging (full prompts, contexts)");
    println!("  --log-file <path>                Write log output to file");
    println!();
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
    println!("  cost <subcommand>              Inspect Cost Intelligence snapshots (V75)");
    println!("    report --snapshot <path>         Human-readable dashboard report");
    println!("    budget --snapshot <path>         Budget status as JSON");
    println!("    savings --snapshot <path>        Token savings summary");
    println!("    projection --snapshot <path>     Daily/monthly cost projections");
    println!("    export --snapshot <path> --output <csv> [--force]");
    println!("                                     Export entries as CSV (formula-safe)");
    println!("    help                             Show cost-subcommand help");
    println!("  verify [options] <prompt>      One-shot query with anti-hallucination pipeline");
    println!(
        "    --provider <name>              Provider (ollama, openai, anthropic, gemini, ...)"
    );
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!(
        "    --strategy <name>              Strategy: mark, omit, warn, footnote (default: mark)"
    );
    println!("    --min-confidence <0.0-1.0>     Minimum confidence threshold (default: 0.3)");
    println!("    --knowledge <path>             Reference document for grounding");
    println!("    --faithfulness                 Enable faithfulness scoring");
    println!("    --cove                         Enable Chain-of-Verification");
    println!("    --quality-gates                Run quality gates on output");
    println!(
        "  vision-check [options]         Pre-flight vision pipeline (transport+model+mmproj)"
    );
    println!("    --provider <name>              Provider (llamacpp, lmstudio, ollama, ...)");
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!("    --mmproj <path>                Path to mmproj.gguf (validates magic+size)");
    println!("    --json                         Output JSON status");
    println!("  research <query>               Search academic papers (requires research feature)");
    println!(
        "    --providers <list>             Providers: arxiv, scholar, pubmed (comma-separated)"
    );
    println!("    --max-results <N>              Max results (default: 10)");
    println!("    --bibtex                       Output in BibTeX format");
    println!("  quality <subcommand>           Quality gate operations");
    println!("    gates list                     List configured quality gates");
    println!("    gates check <text>             Run quality gates on text");
    println!("  tool <name> [options]          Delegated tool invocation (best-effort NL bridge)");
    println!("    --args <json>                  JSON arguments for the tool (default: {{}})");
    println!("    --provider <name>              Provider (ollama, openai, anthropic, ...)");
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!("  workflow <id> [options]        Delegated workflow invocation (best-effort)");
    println!("    --provider <name>              Provider");
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!(
        "  benchmark <subcommand>         Dataset hallucination / faithfulness benchmarks (V90)"
    );
    println!("    list                             List available benchmarks");
    println!("    info <name>                      Show benchmark metadata and citation");
    println!("    download <name> [--accept-license] [--cache-dir <path>]");
    println!("                                     Fetch the benchmark dataset into the cache");
    println!("    run <name> --provider X --model Y [--limit N] [--threshold 0.5]");
    println!("               [--cache-dir <path>] [--json]");
    println!("                                     Run the model against the benchmark and report");
    println!("    calibrate <name> --provider X --model Y [--limit N] [--objective accuracy|f1]");
    println!("               [--cache-dir <path>] [--json]");
    println!(
        "                                     Run the benchmark and sweep correctness threshold"
    );
    println!("  recipes <subcommand>           Recipes — declarative YAML workflows (Phase A.1)");
    println!("    list [--dir <path>]              List discovered recipes");
    println!("    show <name>                      Show recipe definition");
    println!("    validate <name|path>             Validate schema (recipes/v1)");
    println!("    init <name> [--out <path>]       Scaffold a new recipe template");
    println!("    run <name> [--var k=v ...] [--provider X] [--model Y]");
    println!("                                     Execute a recipe end-to-end");
    println!("    share <name> [--out <path>]      Produce a portable single-file bundle");
    println!("    --user-dir <path>                Override user-global recipe dir");
    println!("    --project-dir <path>             Override project recipe dir");
    println!("  test [options]                 Run tests (lib or harness), save results");
    println!("    --all                          Run test harness (all categories)");
    println!("    --category <name>              Run specific harness category");
    println!("    --list                         List harness categories");
    println!("    --lib                          Run cargo test --lib (default)");
    println!("    --filter <pattern>             Filter test names");
    println!("    --features <flags>             Override feature flags");
    println!("    --output <dir>                 Output directory (default: results/)");
    println!(
        "    --provider <name>              Provider (ollama, openai, anthropic, gemini, ...)"
    );
    println!("    --model <name>                 Model name");
    println!("    --url <url>                    Provider URL");
    println!("    --config <file>                Load config from JSON file");
    println!("    --system <prompt>              System prompt");
    println!("    --file <path>                  Read user prompt from file instead of argument");
    println!("    --knowledge <path>             Inject file content as knowledge context");
    println!("    --rag-tier <tier>              RAG tier (fast, semantic, enhanced, thorough, graph, full)");
    println!("    --list-tiers                   List available RAG tiers");
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
            eprintln!(
                "Error: unknown config subcommand '{}'. Use show, check, or set.",
                other
            );
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

    println!(
        "Configuration{}:\n",
        file.map(|f| format!(" ({})", f)).unwrap_or_default()
    );
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
                changes.push(format!(
                    "provider: {:?} -> {:?}",
                    config.provider, new_provider
                ));
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
                changes.push(format!(
                    "custom_url: '{}' -> '{}'",
                    config.custom_url, args[i]
                ));
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
        // Subcommand dispatch: `butler recommend-prompt ...`
        if let Some(first) = args.first() {
            if first == "recommend-prompt" {
                return cmd_butler_recommend_prompt(&args[1..]);
            }
        }

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

#[cfg(all(feature = "butler", feature = "prompt-fragments"))]
fn cmd_butler_recommend_prompt(args: &[String]) -> ExitCode {
    let mut intent: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--intent" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --intent requires a value");
                    return ExitCode::from(1);
                }
                intent = Some(args[i].clone());
            }
            "-h" | "--help" => {
                println!("Usage: ai_cli butler recommend-prompt --intent \"<natural language>\"");
                println!();
                println!("Recommends a PromptPreset and overlay fragments based on a natural");
                println!("language intent and the current environment.");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!(
                    "Error: unknown option '{}' for 'butler recommend-prompt'",
                    other
                );
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let intent = match intent {
        Some(s) if !s.trim().is_empty() => s,
        _ => {
            eprintln!("Error: --intent \"<description>\" is required.");
            eprintln!(
                "  Example: ai_cli butler recommend-prompt --intent \"help me refactor Rust code\""
            );
            return ExitCode::from(1);
        }
    };

    eprint!("Scanning environment...");
    let mut butler = Butler::new();
    let report = butler.scan();
    eprintln!(" done.");

    let rec = butler.recommend_prompt_fragments(&intent, &report);

    println!("--- Prompt Fragments Recommendation ---");
    println!("Intent:        {}", intent);
    println!("Recommended preset: {:?}", rec.preset);
    if rec.extra_fragment_keys.is_empty() {
        println!("Extra fragments:    (none)");
    } else {
        println!("Extra fragments:");
        for key in &rec.extra_fragment_keys {
            println!("  - {}", key);
        }
    }
    println!();
    println!("Justification:");
    for line in rec.justification.lines() {
        println!("  {}", line);
    }
    println!("---------------------------------------\n");
    ExitCode::SUCCESS
}

#[cfg(all(feature = "butler", not(feature = "prompt-fragments")))]
fn cmd_butler_recommend_prompt(_args: &[String]) -> ExitCode {
    eprintln!("Error: 'butler recommend-prompt' requires the 'prompt-fragments' feature.");
    eprintln!(
        "  cargo run --bin ai_cli --features \"full,butler,prompt-fragments\" -- butler recommend-prompt --intent \"...\""
    );
    ExitCode::from(1)
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
    let mut rag_tier_name: Option<String> = None;
    let mut json_output = false;
    let mut temperature: Option<f32> = None;
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut image_paths: Vec<String> = Vec::new();

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
            "--image" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --image requires a path or URL");
                    return ExitCode::from(1);
                }
                image_paths.push(args[i].clone());
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
            "--rag-tier" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --rag-tier requires a tier name");
                    return ExitCode::from(1);
                }
                rag_tier_name = Some(args[i].clone());
            }
            "--list-tiers" => {
                let store = ai_assistant::RagTierStore::new();
                println!("Available RAG tiers:");
                for tier in store.list() {
                    let marker = if tier.builtin { " (builtin)" } else { "" };
                    println!("  {:12} — {}{}", tier.name, tier.description, marker);
                }
                return ExitCode::SUCCESS;
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
            eprintln!(
                "Error: no model available. Specify --provider and --model, or install a model."
            );
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

    // Vision path: when --image is supplied, bypass the streaming assistant
    // and call the vision dispatcher directly. Knowledge (if any) is appended
    // to the user prompt because the dispatcher takes a single user message.
    #[cfg(feature = "vision")]
    if !image_paths.is_empty() {
        let start = Instant::now();
        let combined_prompt = if knowledge.is_empty() {
            user_prompt.clone()
        } else {
            format!("Reference context:\n{}\n\n{}", knowledge, user_prompt)
        };

        let images = match load_images(&image_paths) {
            Ok(imgs) => imgs,
            Err(e) => {
                eprintln!("Error loading images: {}", e);
                return ExitCode::from(1);
            }
        };

        let vmsg = ai_assistant::VisionMessage::user(&combined_prompt, images);
        let sys_prompt = system_prompt.unwrap_or_default();
        let result =
            ai_assistant::generate_vision_response(&assistant.config, &[vmsg], &sys_prompt);

        let elapsed = start.elapsed();
        match result {
            Ok(text) => {
                if json_output {
                    let json = serde_json::json!({
                        "provider": format!("{:?}", assistant.config.provider),
                        "model": assistant.config.selected_model,
                        "images": image_paths,
                        "response": text,
                        "elapsed_ms": elapsed.as_millis(),
                        "error": false,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                } else {
                    println!("{}", text);
                    eprintln!(
                        "\n[{:?} / {} / {} image(s) / {:.1}s]",
                        assistant.config.provider,
                        assistant.config.selected_model,
                        image_paths.len(),
                        elapsed.as_secs_f64()
                    );
                }
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Vision request failed: {}", e);
                return ExitCode::from(1);
            }
        }
    }

    #[cfg(not(feature = "vision"))]
    if !image_paths.is_empty() {
        eprintln!("Error: --image requires the `vision` feature. Rebuild with --features vision.");
        return ExitCode::from(1);
    }

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
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
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
// tool / workflow — delegated NL-bridge to the LLM
//
// These are best-effort "delegated runtime" bridges used by `ai_jobs` when
// runtime="delegated". The LLM receives a natural-language instruction to
// run the tool/workflow. A real MCP tool dispatcher is available only in
// the embedded runtime (ai_jobs with feature="full"). When precise tool
// invocation is required, prefer runtime="embedded".
// =============================================================================

fn cmd_tool(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_cli tool <name> [--args <json>] [--provider X] [--model Y] [--url Z]");
        return ExitCode::from(1);
    }
    let tool_name = args[0].clone();
    let rest = &args[1..];

    let args_json = find_flag_value(rest, "--args").unwrap_or("{}").to_string();
    let prompt = format!(
        "Use the tool `{}` with these JSON arguments:\n{}\n\nReturn the tool's result.",
        tool_name, args_json
    );
    run_delegated_llm(rest, &prompt)
}

fn cmd_workflow(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_cli workflow <id> [--provider X] [--model Y] [--url Z]");
        return ExitCode::from(1);
    }
    let workflow_id = args[0].clone();
    let rest = &args[1..];
    let prompt = format!("Run workflow `{}`.", workflow_id);
    run_delegated_llm(rest, &prompt)
}

fn run_delegated_llm(args: &[String], prompt: &str) -> ExitCode {
    let provider_name = find_flag_value(args, "--provider").map(String::from);
    let model_name = find_flag_value(args, "--model").map(String::from);
    let url_override = find_flag_value(args, "--url").map(String::from);

    let mut assistant = AiAssistant::new();
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

    if assistant.config.selected_model.is_empty() {
        eprintln!(
            "Error: no model set. Pass --provider and --model (e.g. --provider ollama \
             --model mistral:7b-instruct)."
        );
        return ExitCode::from(1);
    }

    assistant.send_message(prompt.to_string(), "");
    let start = Instant::now();
    let deadline = Duration::from_secs(120);
    let mut out = String::new();
    let mut errored = false;

    loop {
        if start.elapsed() > deadline {
            eprintln!(
                "\nError: delegated LLM query exceeded {}s",
                deadline.as_secs()
            );
            errored = true;
            break;
        }
        if let Some(response) = assistant.poll_response() {
            match response {
                AiResponse::Chunk(text) => {
                    print!("{}", text);
                    let _ = std::io::stdout().flush();
                    out.push_str(&text);
                }
                AiResponse::Complete(text) => {
                    if out.is_empty() && !text.is_empty() {
                        print!("{}", text);
                    }
                    println!();
                    break;
                }
                AiResponse::Error(e) => {
                    eprintln!("\nError: {}", e);
                    errored = true;
                    break;
                }
                AiResponse::Cancelled(_) => break,
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(10));
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

    let label = if harness_mode {
        "test-harness"
    } else {
        "test-lib"
    };
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
        eprintln!(
            "Error creating output dir '{}': {}",
            output_dir.display(),
            e
        );
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
            eprintln!(
                "Warning: could not write log file '{}': {}",
                log_file.display(),
                e
            );
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
        "llamacpp" | "llama-cpp" | "llama_cpp" | "llama.cpp" => ai_assistant::AiProvider::LlamaCpp,
        "vllm" => ai_assistant::AiProvider::VLLM,
        "kobold" | "koboldcpp" | "kobold-cpp" | "kobold_cpp" => ai_assistant::AiProvider::KoboldCpp,
        "localai" | "local-ai" | "local_ai" => ai_assistant::AiProvider::LocalAI,
        "textgen" | "textgen-webui" | "text-gen-webui" => ai_assistant::AiProvider::TextGenWebUI,
        other => {
            eprintln!(
                "Warning: unknown provider '{}', defaulting to Ollama",
                other
            );
            ai_assistant::AiProvider::Ollama
        }
    }
}

// =============================================================================
// vision-check — pre-flight for the vision pipeline (V90.26)
// =============================================================================

#[cfg(feature = "vision")]
fn cmd_vision_check(args: &[String]) -> ExitCode {
    let mut provider_name: Option<String> = None;
    let mut model_name: Option<String> = None;
    let mut url_override: Option<String> = None;
    let mut mmproj: Option<String> = None;
    let mut as_json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" if i + 1 < args.len() => {
                i += 1;
                provider_name = Some(args[i].clone());
            }
            "--model" if i + 1 < args.len() => {
                i += 1;
                model_name = Some(args[i].clone());
            }
            "--url" if i + 1 < args.len() => {
                i += 1;
                url_override = Some(args[i].clone());
            }
            "--mmproj" if i + 1 < args.len() => {
                i += 1;
                mmproj = Some(args[i].clone());
            }
            "--json" => as_json = true,
            other => {
                eprintln!("Error: unknown vision-check option '{}'", other);
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let mut config = ai_assistant::AiConfig::default();
    if let Some(name) = &provider_name {
        config.provider = provider_from_name(name);
    }
    if let Some(model) = &model_name {
        config.selected_model = model.clone();
    }
    if let Some(url) = &url_override {
        match config.provider {
            ai_assistant::AiProvider::LlamaCpp => config.llamacpp_url = url.clone(),
            ai_assistant::AiProvider::LMStudio => config.lm_studio_url = url.clone(),
            ai_assistant::AiProvider::Ollama => config.ollama_url = url.clone(),
            ai_assistant::AiProvider::KoboldCpp => config.kobold_url = url.clone(),
            ai_assistant::AiProvider::LocalAI => config.local_ai_url = url.clone(),
            ai_assistant::AiProvider::VLLM => config.vllm_url = url.clone(),
            ai_assistant::AiProvider::TextGenWebUI => config.text_gen_webui_url = url.clone(),
            _ => {}
        }
    }
    if let Some(p) = &mmproj {
        config.mmproj_path = Some(std::path::PathBuf::from(p));
    }

    // 1. Static transport check.
    let transport_ok = ai_assistant::vision::agent_bridge::vision_supported_for(&config);

    // 2. Static model-capability check.
    let caps = ai_assistant::VisionCapabilities::default();
    let model_ok = caps.supports_vision(&config.selected_model);

    // 3. mmproj validation, if a path was provided.
    let mmproj_status = config.validated_mmproj().map(|res| match res {
        Ok(p) => Ok(p.filename().into_owned()),
        Err(e) => Err(e.to_string()),
    });

    // 4. llama.cpp /props probe (best effort, only when applicable).
    let probe = if matches!(config.provider, ai_assistant::AiProvider::LlamaCpp) {
        Some(ai_assistant::llamacpp_capability::probe_llamacpp(
            &config.llamacpp_url,
        ))
    } else {
        None
    };
    let multimodal_reported = probe
        .as_ref()
        .and_then(|r| r.as_ref().ok())
        .and_then(|c| c.multimodal);

    if as_json {
        let mmproj_validation_json = mmproj_status.as_ref().map(|r| match r {
            Ok(name) => serde_json::json!({"ok": true, "filename": name}),
            Err(msg) => serde_json::json!({"ok": false, "error": msg}),
        });
        let json = serde_json::json!({
            "provider": format!("{:?}", config.provider),
            "model": config.selected_model,
            "transport_ok": transport_ok,
            "model_in_known_set": model_ok,
            "mmproj_configured": mmproj_status.is_some(),
            "mmproj_validation": mmproj_validation_json,
            "llamacpp_probe_ran": probe.is_some(),
            "llamacpp_multimodal_loaded": multimodal_reported,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!("vision-check");
        println!("  provider              : {:?}", config.provider);
        println!("  model                 : {}", config.selected_model);
        println!(
            "  transport supported   : {}",
            if transport_ok { "yes" } else { "NO" }
        );
        println!(
            "  model in known set    : {}",
            if model_ok { "yes" } else { "NO" }
        );
        match &mmproj_status {
            None => println!("  mmproj                : (not configured)"),
            Some(Ok(name)) => println!("  mmproj                : OK ({})", name),
            Some(Err(msg)) => println!("  mmproj                : INVALID — {}", msg),
        }
        if let Some(probe_res) = probe.as_ref() {
            match probe_res {
                Ok(cap) => match cap.multimodal {
                    Some(true) => println!("  llama-server projector: loaded"),
                    Some(false) => println!(
                        "  llama-server projector: NOT loaded — start with --mmproj <path>"
                    ),
                    None => println!("  llama-server projector: unknown (probe inconclusive)"),
                },
                Err(e) => println!("  llama-server probe    : failed ({})", e),
            }
        }
    }

    let any_error = !transport_ok
        || !model_ok
        || matches!(&mmproj_status, Some(Err(_)))
        || matches!(multimodal_reported, Some(false));
    if any_error {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

// =============================================================================
// cost — inspect Cost Intelligence snapshots (V77)
// =============================================================================

fn cmd_cost(args: &[String]) -> ExitCode {
    if args.is_empty() {
        print_cost_usage();
        return ExitCode::from(1);
    }
    match args[0].as_str() {
        "help" | "-h" | "--help" => {
            print_cost_usage();
            ExitCode::SUCCESS
        }
        "report" => cost_subcommand_report(&args[1..]),
        "budget" => cost_subcommand_budget(&args[1..]),
        "savings" => cost_subcommand_savings(&args[1..]),
        "projection" => cost_subcommand_projection(&args[1..]),
        "export" => cost_subcommand_export(&args[1..]),
        other => {
            eprintln!("Error: unknown cost subcommand '{}'\n", other);
            print_cost_usage();
            ExitCode::from(1)
        }
    }
}

fn print_cost_usage() {
    println!("ai_cli cost — inspect Cost Intelligence snapshots\n");
    println!("Usage: ai_cli cost <subcommand> [options]\n");
    println!("Subcommands:");
    println!("  report --snapshot <path>                    Human-readable dashboard report");
    println!("  budget --snapshot <path>                    Budget status as JSON");
    println!("  savings --snapshot <path>                   Token savings summary (informational)");
    println!("  projection --snapshot <path>                Daily/monthly/per-request projections");
    println!("  export --snapshot <path> --output <csv> [--force]");
    println!("                                              Export entries as CSV");
    println!();
    println!("Examples:");
    println!("  ai_cli cost report --snapshot ~/.ai_assistant/cost.json");
    println!("  ai_cli cost budget --snapshot cost.json");
    println!("  ai_cli cost export --snapshot cost.json --output out.csv --force");
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].as_str());
        }
        i += 1;
    }
    None
}

fn load_cost_snapshot(path: &str) -> Result<ai_assistant::cost_integration::CostDashboard, String> {
    use ai_assistant::cost_integration::{CostDashboard, CostDashboardSnapshot};
    let canon = std::path::Path::new(path)
        .canonicalize()
        .map_err(|e| format!("cannot resolve snapshot path '{}': {}", path, e))?;
    eprintln!("[ai_cli cost] loading snapshot: {}", canon.display());
    let content =
        std::fs::read_to_string(&canon).map_err(|e| format!("cannot read snapshot: {}", e))?;
    let snapshot: CostDashboardSnapshot =
        serde_json::from_str(&content).map_err(|e| format!("invalid snapshot JSON: {}", e))?;
    let mut dashboard = CostDashboard::new();
    dashboard.restore(snapshot);
    Ok(dashboard)
}

fn cost_subcommand_report(args: &[String]) -> ExitCode {
    let snapshot_path = match find_flag_value(args, "--snapshot") {
        Some(p) => p,
        None => {
            eprintln!(
                "ai_cli cost report: no --snapshot provided.\n\
                 Persist a CostDashboardSnapshot to disk (via API) and pass \
                 its path here."
            );
            return ExitCode::from(1);
        }
    };
    match load_cost_snapshot(snapshot_path) {
        Ok(dashboard) => {
            println!("{}", dashboard.format_report());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cost_subcommand_budget(args: &[String]) -> ExitCode {
    let snapshot_path = match find_flag_value(args, "--snapshot") {
        Some(p) => p,
        None => {
            eprintln!("ai_cli cost budget: missing --snapshot <path>");
            return ExitCode::from(1);
        }
    };
    match load_cost_snapshot(snapshot_path) {
        Ok(dashboard) => {
            let projected_daily = dashboard.projected_daily_cost();
            let projected_monthly = dashboard.projected_monthly_cost();
            let total: f64 = dashboard
                .format_report()
                .lines()
                .find_map(|l| l.strip_prefix("Total cost: $"))
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(0.0);
            let json = serde_json::json!({
                "total_usd": total,
                "projected_daily_usd": projected_daily,
                "projected_monthly_usd": projected_monthly,
                "snapshot": snapshot_path,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cost_subcommand_savings(args: &[String]) -> ExitCode {
    let snapshot_path = match find_flag_value(args, "--snapshot") {
        Some(p) => p,
        None => {
            eprintln!("ai_cli cost savings: missing --snapshot <path>");
            return ExitCode::from(1);
        }
    };
    let dashboard = match load_cost_snapshot(snapshot_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::from(1);
        }
    };

    let total = dashboard.total_cost();
    let requests = dashboard.total_requests();
    let avg = dashboard.average_cost_per_request();
    let by_model = dashboard.cost_by_model();
    let top = dashboard.most_expensive(5);

    println!("Cost Savings Analysis");
    println!("=====================");
    println!("Total spend:          ${:.4}", total);
    println!("Requests:             {}", requests);
    println!("Avg cost / request:   ${:.6}", avg);
    println!();

    if by_model.is_empty() {
        println!("(no recorded entries — savings analysis unavailable)");
        return ExitCode::SUCCESS;
    }

    let mut models: Vec<(String, f64, usize)> = by_model
        .iter()
        .map(|(m, c)| {
            let count = dashboard.entries().iter().filter(|e| &e.model == m).count();
            (m.clone(), *c, count)
        })
        .collect();
    models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("By model:");
    for (m, c, n) in &models {
        let avg_model = if *n > 0 { c / *n as f64 } else { 0.0 };
        println!(
            "  {:<32} ${:>10.4}  ({} req, avg ${:.6})",
            m, c, n, avg_model
        );
    }
    println!();

    // Hypothetical-max savings: cost if every request had used the most
    // expensive model (by average cost/request).
    if let Some(most_expensive) = models.iter().filter(|(_, _, n)| *n > 0).max_by(|a, b| {
        let ea = a.1 / a.2 as f64;
        let eb = b.1 / b.2 as f64;
        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        let worst_avg = most_expensive.1 / most_expensive.2 as f64;
        let hypothetical = worst_avg * requests as f64;
        let saved = (hypothetical - total).max(0.0);
        let pct = if hypothetical > 0.0 {
            saved / hypothetical * 100.0
        } else {
            0.0
        };
        println!("Hypothetical single-model cost (worst avg):");
        println!(
            "  If all {} requests had used '{}' (avg ${:.6}/req):",
            requests, most_expensive.0, worst_avg
        );
        println!("    Hypothetical total: ${:.4}", hypothetical);
        println!("    Actual total:       ${:.4}", total);
        println!("    Savings:            ${:.4} ({:.1}%)", saved, pct);
        println!();
    }

    if !top.is_empty() {
        println!("Top {} most expensive requests:", top.len());
        for (i, e) in top.iter().enumerate() {
            println!(
                "  {}. {} — ${:.6} ({}/{} tok, {})",
                i + 1,
                e.model,
                e.cost_usd,
                e.input_tokens,
                e.output_tokens,
                e.timestamp
            );
        }
    }

    ExitCode::SUCCESS
}

fn cost_subcommand_projection(args: &[String]) -> ExitCode {
    let snapshot_path = match find_flag_value(args, "--snapshot") {
        Some(p) => p,
        None => {
            eprintln!("ai_cli cost projection: missing --snapshot <path>");
            return ExitCode::from(1);
        }
    };
    match load_cost_snapshot(snapshot_path) {
        Ok(dashboard) => {
            let daily = dashboard
                .projected_daily_cost()
                .map(|v| format!("${:.4}", v))
                .unwrap_or_else(|| "n/a".into());
            let monthly = dashboard
                .projected_monthly_cost()
                .map(|v| format!("${:.2}", v))
                .unwrap_or_else(|| "n/a".into());
            let per_1k = dashboard.projected_cost_for_requests(1000);
            println!("Projections:");
            println!("  Daily   : {}", daily);
            println!("  Monthly : {}", monthly);
            println!("  Per 1k  : ${:.4}", per_1k);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cost_subcommand_export(args: &[String]) -> ExitCode {
    let snapshot_path = match find_flag_value(args, "--snapshot") {
        Some(p) => p,
        None => {
            eprintln!("ai_cli cost export: missing --snapshot <path>");
            return ExitCode::from(1);
        }
    };
    let output = match find_flag_value(args, "--output") {
        Some(p) => p,
        None => {
            eprintln!("ai_cli cost export: missing --output <file.csv>");
            return ExitCode::from(1);
        }
    };
    let force = args.iter().any(|a| a == "--force");
    if std::path::Path::new(output).exists() && !force {
        eprintln!(
            "ai_cli cost export: output file '{}' already exists. Use --force to overwrite.",
            output
        );
        return ExitCode::from(1);
    }
    match load_cost_snapshot(snapshot_path) {
        Ok(dashboard) => {
            let csv = dashboard.export_csv();
            if let Err(e) = std::fs::write(output, csv) {
                eprintln!("Error writing CSV: {}", e);
                return ExitCode::from(1);
            }
            println!("Wrote CSV to {}", output);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn load_config(path: &str) -> Result<AiConfig, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("cannot read file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {}", e))
}

/// Load images for a vision request from a list of paths or http(s) URLs.
///
/// Validates: file size ≤ 20 MB, supported extension (.jpg, .jpeg, .png,
/// .gif, .webp, .bmp). URLs are passed through without download — the
/// provider fetches them server-side.
#[cfg(feature = "vision")]
fn load_images(paths: &[String]) -> Result<Vec<ai_assistant::ImageInput>, String> {
    const MAX_BYTES: u64 = 20 * 1024 * 1024;
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        if p.starts_with("http://") || p.starts_with("https://") {
            out.push(ai_assistant::ImageInput::from_url(p));
            continue;
        }
        let pb = std::path::PathBuf::from(p);
        if !pb.exists() {
            return Err(format!("image not found: {}", p));
        }
        let meta = std::fs::metadata(&pb).map_err(|e| format!("stat {}: {}", p, e))?;
        if meta.len() > MAX_BYTES {
            return Err(format!("image {} is {} bytes (>20 MB max)", p, meta.len()));
        }
        let img =
            ai_assistant::ImageInput::from_file(&pb).map_err(|e| format!("load {}: {}", p, e))?;
        out.push(img);
    }
    Ok(out)
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
    println!("OS:      {} ({})", report.runtime.os, report.runtime.arch);
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
                format!(
                    " ({} models: {}{})",
                    model_count,
                    preview.join(", "),
                    suffix
                )
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

// =============================================================================
// verify — one-shot query with anti-hallucination (V88)
// =============================================================================

fn cmd_verify(args: &[String]) -> ExitCode {
    let mut provider_name: Option<String> = None;
    let mut model_name: Option<String> = None;
    let mut url_override: Option<String> = None;
    let mut strategy = "mark".to_string();
    let mut min_confidence: f64 = 0.3;
    let mut faithfulness = false;
    let mut cove = false;
    let mut quality_gates = false;
    let mut knowledge_path: Option<String> = None;
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut image_paths: Vec<String> = Vec::new();

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
            "--image" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --image requires a path or URL");
                    return ExitCode::from(1);
                }
                image_paths.push(args[i].clone());
            }
            "--strategy" if i + 1 < args.len() => {
                i += 1;
                strategy = args[i].clone();
            }
            "--min-confidence" if i + 1 < args.len() => {
                i += 1;
                min_confidence = args[i].parse().unwrap_or(0.3);
            }
            "--faithfulness" => faithfulness = true,
            "--cove" => cove = true,
            "--quality-gates" => quality_gates = true,
            "--knowledge" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --knowledge requires a path");
                    return ExitCode::from(1);
                }
                knowledge_path = Some(args[i].clone());
            }
            other => prompt_parts.push(other.to_string()),
        }
        i += 1;
    }

    if prompt_parts.is_empty() {
        eprintln!("Error: verify requires a prompt");
        return ExitCode::from(1);
    }

    let user_prompt = prompt_parts.join(" ");

    // Load knowledge context for grounding
    let knowledge = if let Some(ref path) = knowledge_path {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                eprintln!("Loaded knowledge: {} chars from {}", content.len(), path);
                content
            }
            Err(e) => {
                eprintln!("Error reading knowledge file '{}': {}", path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        String::new()
    };

    eprintln!("Verify mode:");
    eprintln!("  Strategy:       {}", strategy);
    eprintln!("  Min confidence: {:.2}", min_confidence);
    eprintln!("  Faithfulness:   {}", faithfulness);
    eprintln!("  CoVe:           {}", cove);
    eprintln!("  Quality gates:  {}", quality_gates);
    if !knowledge.is_empty() {
        eprintln!("  Knowledge:      {} chars", knowledge.len());
    }
    eprintln!();

    // --- Build assistant and query LLM ---
    let mut assistant = AiAssistant::new();

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

    // Auto-detect model if not specified
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
                fetch_models_blocking(&mut assistant);
            }
        }

        #[cfg(not(feature = "butler"))]
        {
            fetch_models_blocking(&mut assistant);
        }

        if assistant.config.selected_model.is_empty() {
            eprintln!(
                "Error: no model available. Specify --provider and --model, or install a model."
            );
            return ExitCode::from(1);
        }
    }

    eprintln!(
        "Querying {:?} / {} ...",
        assistant.config.provider, assistant.config.selected_model
    );

    let start = Instant::now();

    // Vision short-circuit: if --image was supplied, run a non-streaming
    // vision request and continue into the anti-hallucination pipeline with
    // the resulting `full_response`.
    #[cfg(feature = "vision")]
    let mut __vision_response: Option<String> = None;
    #[cfg(feature = "vision")]
    if !image_paths.is_empty() {
        let combined = if knowledge.is_empty() {
            user_prompt.clone()
        } else {
            format!("Reference context:\n{}\n\n{}", knowledge, user_prompt)
        };
        let images = match load_images(&image_paths) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error loading images: {}", e);
                return ExitCode::from(1);
            }
        };
        let vmsg = ai_assistant::VisionMessage::user(&combined, images);
        match ai_assistant::generate_vision_response(&assistant.config, &[vmsg], "") {
            Ok(text) => {
                println!("{}", text);
                __vision_response = Some(text);
            }
            Err(e) => {
                eprintln!("Vision request failed: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        assistant.send_message(user_prompt.clone(), &knowledge);
    }
    #[cfg(not(feature = "vision"))]
    if !image_paths.is_empty() {
        eprintln!("Error: --image requires the `vision` feature.");
        return ExitCode::from(1);
    } else {
        assistant.send_message(user_prompt.clone(), &knowledge);
    }

    // Poll for response
    let mut full_response = String::new();
    let mut errored = false;

    #[cfg(feature = "vision")]
    let skip_polling = __vision_response.is_some();
    #[cfg(not(feature = "vision"))]
    let skip_polling = false;

    #[cfg(feature = "vision")]
    if let Some(text) = __vision_response.take() {
        full_response = text;
    }

    if !skip_polling {
        loop {
            if let Some(response) = assistant.poll_response() {
                match response {
                    AiResponse::Chunk(text) => {
                        print!("{}", text);
                        let _ = std::io::stdout().flush();
                        full_response.push_str(&text);
                    }
                    AiResponse::Complete(text) => {
                        if !text.is_empty() && full_response.is_empty() {
                            print!("{}", text);
                        }
                        println!();
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

        // Drain remaining messages
        for _ in 0..50 {
            if assistant.poll_response().is_none() {
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\n[{:?} / {} / {:.1}s]",
        assistant.config.provider,
        assistant.config.selected_model,
        elapsed.as_secs_f64()
    );

    if errored || full_response.is_empty() {
        return ExitCode::from(1);
    }

    // --- Run anti-hallucination pipeline ---
    #[cfg(feature = "eval")]
    {
        use ai_assistant::anti_hallucination::{
            AntiHallucinationConfig, AntiHallucinationPipeline,
        };

        let strategy_enum = match strategy.as_str() {
            "omit" => ai_assistant::anti_hallucination::UngroundedClaimStrategy::Omit,
            "warn" => ai_assistant::anti_hallucination::UngroundedClaimStrategy::Warn,
            "footnote" => ai_assistant::anti_hallucination::UngroundedClaimStrategy::Footnote,
            "verify_then_mark" => {
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::VerifyThenMark
            }
            "verify_then_omit" => {
                ai_assistant::anti_hallucination::UngroundedClaimStrategy::VerifyThenOmit
            }
            "ask" => ai_assistant::anti_hallucination::UngroundedClaimStrategy::Ask,
            _ => ai_assistant::anti_hallucination::UngroundedClaimStrategy::Mark,
        };

        let mut ah_config = AntiHallucinationConfig::production();
        ah_config.ungrounded_strategy = strategy_enum;
        ah_config.min_confidence_for_output = min_confidence;

        let pipeline = AntiHallucinationPipeline::new(ah_config);
        // Only use knowledge for grounding; without it, fall back to confidence-based
        let grounding_context = if !knowledge.is_empty() {
            Some(knowledge.as_str())
        } else {
            None
        };
        let ah_result = pipeline.process(&full_response, grounding_context);

        println!();
        println!("--- Anti-Hallucination Analysis ---");
        println!("  Strategy:        {:?}", ah_result.strategy_applied);
        println!("  Confidence:      {:.2}", ah_result.overall_confidence);
        println!("  Grounding ratio: {:.2}", ah_result.grounding_ratio());
        println!("  Ungrounded:      {} claims", ah_result.ungrounded_count);
        println!("  Abstained:       {}", ah_result.abstained);
        if ah_result.abstained {
            if let Some(ref reason) = ah_result.abstention_reason {
                println!("  Reason:          {}", reason);
            }
        }
        println!("  Claims found:    {}", ah_result.claims.len());

        // Faithfulness scoring
        let mut faith_score: Option<f64> = None;
        if faithfulness {
            use ai_assistant::faithfulness::{FaithfulnessConfig, FaithfulnessScorer};

            // Split context into individual sentences for better Jaccard matching.
            // WordOverlap uses Jaccard(claim_words, chunk_words) — a single large
            // chunk dilutes the ratio, so we split into per-sentence chunks.
            let source = if !knowledge.is_empty() {
                &knowledge
            } else {
                &user_prompt
            };
            let context_sentences: Vec<&str> = source
                .split(|c: char| c == '.' || c == '\n')
                .map(|s| s.trim())
                .filter(|s| s.len() > 5)
                .collect();

            let f_scorer = FaithfulnessScorer::new(FaithfulnessConfig::default());
            let f_report = f_scorer.score(&full_response, &context_sentences);

            println!();
            println!("--- Faithfulness Report ---");
            println!("  Overall score:   {:.2}", f_report.overall_score);
            println!(
                "  Entailed: {} | Contradicted: {} | Neutral: {}",
                f_report.entailed_count, f_report.contradicted_count, f_report.neutral_count
            );
            faith_score = Some(f_report.overall_score);
        }

        // Chain-of-Verification
        if cove {
            use ai_assistant::chain_of_verification::{
                ChainOfVerification, CoVeConfig, VerificationContext, VerificationSource,
            };

            let cove_source = if !knowledge.is_empty() {
                &knowledge
            } else {
                &user_prompt
            };
            let reliability = if !knowledge.is_empty() { 0.9 } else { 0.5 };
            let source_type = if !knowledge.is_empty() {
                "file"
            } else {
                "user_query"
            };
            let cove_contexts: Vec<VerificationContext> = cove_source
                .split(|c: char| c == '.' || c == '\n')
                .map(|s| s.trim())
                .filter(|s| s.len() > 5)
                .enumerate()
                .map(|(i, sentence)| VerificationContext {
                    source_id: format!("ctx-{}", i),
                    source_type: source_type.to_string(),
                    content: sentence.to_string(),
                    reliability,
                })
                .collect();

            // Configure CoVe with Both source (accepts any source_type)
            let mut cove_config = CoVeConfig::default();
            cove_config.verification_source = VerificationSource::Both;

            // Build LLM verifier closure
            let v_provider = assistant.config.provider.clone();
            let v_model = assistant.config.selected_model.clone();
            let v_ollama_url = assistant.config.ollama_url.clone();
            let v_lm_url = assistant.config.lm_studio_url.clone();
            let v_custom_url = assistant.config.custom_url.clone();

            let llm_verify = move |prompt: &str| -> Option<String> {
                let mut verifier = AiAssistant::new();
                verifier.config.provider = v_provider.clone();
                verifier.config.selected_model = v_model.clone();
                verifier.config.ollama_url = v_ollama_url.clone();
                verifier.config.lm_studio_url = v_lm_url.clone();
                verifier.config.custom_url = v_custom_url.clone();
                verifier.config.temperature = 0.1;

                verifier.send_message(prompt.to_string(), "");

                let mut full = String::new();
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    if Instant::now() > deadline {
                        return None;
                    }
                    if let Some(resp) = verifier.poll_response() {
                        match resp {
                            AiResponse::Chunk(t) => full.push_str(&t),
                            AiResponse::Complete(t) => {
                                if full.is_empty() {
                                    full = t;
                                }
                                return Some(full);
                            }
                            AiResponse::Error(_) => return None,
                            AiResponse::Cancelled(t) => return Some(t),
                            _ => {}
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            };

            eprintln!("Running Chain-of-Verification with LLM...");
            let cove_engine = ChainOfVerification::new(cove_config).with_llm_verifier(llm_verify);
            let cove_result = cove_engine.verify(&full_response, &cove_contexts);

            // Count by status
            let supported = cove_result
                .verified_claims
                .iter()
                .filter(|c| {
                    c.status
                        == ai_assistant::chain_of_verification::ClaimVerificationStatus::Supported
                })
                .count();
            let contradicted = cove_result
                .verified_claims
                .iter()
                .filter(|c| {
                    c.status
                        == ai_assistant::chain_of_verification::ClaimVerificationStatus::Contradicted
                })
                .count();
            let unverifiable = cove_result.verified_claims.len() - supported - contradicted;

            println!();
            println!("--- Chain-of-Verification ---");
            println!("  Claims verified: {}", cove_result.verified_claims.len());
            println!(
                "  Supported: {} | Contradicted: {} | Unverifiable: {}",
                supported, contradicted, unverifiable
            );
            println!("  Accuracy:        {:.2}", cove_result.overall_accuracy);
            println!("  Corrections:     {}", cove_result.corrections_made);
            if cove_result.corrections_made > 0 {
                println!();
                println!("  Corrected response:");
                println!("  {}", cove_result.corrected_response);
            }
        }

        // Quality gates
        if quality_gates {
            use ai_assistant::quality_gates::{QualityGateRunner, QualityScores};

            let scores = QualityScores {
                faithfulness: faith_score,
                confidence: Some(ah_result.overall_confidence),
                grounding_ratio: Some(ah_result.grounding_ratio()),
                consistency_score: None,
                citation_coverage: None,
            };
            let runner = QualityGateRunner::production_defaults();
            let gate_result = runner.run(&scores);

            println!();
            println!("--- Quality Gates ---");
            println!(
                "  {} ({}/{})",
                gate_result.summary(),
                gate_result.passed_count(),
                gate_result.total_checked()
            );
            if !gate_result.warnings.is_empty() {
                for w in &gate_result.warnings {
                    println!("  Warning: {}", w);
                }
            }
            if !gate_result.passed {
                for f in &gate_result.failing_gates {
                    println!("  FAILED: {}", f);
                }
            }
        }
    }

    #[cfg(not(feature = "eval"))]
    {
        let _ = (
            strategy,
            min_confidence,
            faithfulness,
            cove,
            quality_gates,
            knowledge,
            user_prompt,
        );
        eprintln!("Anti-hallucination pipeline requires the 'eval' feature");
    }

    ExitCode::SUCCESS
}

// =============================================================================
// research — academic paper search (V88, gated)
// =============================================================================

#[cfg(feature = "research")]
fn cmd_research(args: &[String]) -> ExitCode {
    let mut providers = vec!["arxiv".to_string(), "semantic_scholar".to_string()];
    let mut max_results: usize = 10;
    let mut bibtex = false;
    let mut query_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--providers" if i + 1 < args.len() => {
                i += 1;
                providers = args[i].split(',').map(|s| s.trim().to_string()).collect();
            }
            "--max-results" if i + 1 < args.len() => {
                i += 1;
                max_results = args[i].parse().unwrap_or(10);
            }
            "--bibtex" => bibtex = true,
            other => query_parts.push(other.to_string()),
        }
        i += 1;
    }

    if query_parts.is_empty() {
        eprintln!("Error: research requires a query");
        return ExitCode::from(1);
    }

    let query = query_parts.join(" ");
    println!("Academic search:");
    println!("  Query:       {}", query);
    println!("  Providers:   {}", providers.join(", "));
    println!("  Max results: {}", max_results);
    println!("  BibTeX:      {}", bibtex);
    println!();

    // Use providers to search
    use ai_assistant::academic_search::AcademicSearchProvider;
    let mut config = ai_assistant::academic_search::AcademicSearchConfig::default();
    config.max_results = max_results;

    let display_papers = |papers: &[ai_assistant::academic_search::AcademicPaper], bibtex: bool| {
        if bibtex {
            println!(
                "{}",
                ai_assistant::bibtex::BibGenerator::from_papers(papers)
            );
        } else {
            for (idx, p) in papers.iter().enumerate() {
                println!("  {}. {} ({})", idx + 1, p.title, p.year.unwrap_or(0));
                if let Some(ref doi) = p.doi {
                    println!("     DOI: {}", doi);
                }
                if let Some(ref url) = p.url {
                    println!("     URL: {}", url);
                }
            }
        }
        println!("  Found {} papers", papers.len());
    };

    for provider_name in &providers {
        println!("--- {} ---", provider_name);
        match provider_name.as_str() {
            "arxiv" => {
                let provider = ai_assistant::academic_search::ArxivProvider::new();
                match provider.search_papers(&query, &config) {
                    Ok(papers) => display_papers(&papers, bibtex),
                    Err(e) => println!("  Error: {}", e),
                }
            }
            "scholar" | "semantic_scholar" => {
                let provider = ai_assistant::academic_search::SemanticScholarProvider::new();
                match provider.search_papers(&query, &config) {
                    Ok(papers) => display_papers(&papers, bibtex),
                    Err(e) => println!("  Error: {}", e),
                }
            }
            "pubmed" => {
                let provider = ai_assistant::academic_search::PubMedProvider::new();
                match provider.search_papers(&query, &config) {
                    Ok(papers) => display_papers(&papers, bibtex),
                    Err(e) => println!("  Error: {}", e),
                }
            }
            other => {
                println!(
                    "  Unknown provider '{}'. Available: arxiv, scholar, pubmed",
                    other
                );
            }
        }
    }

    ExitCode::SUCCESS
}

// =============================================================================
// quality — quality gate operations (V88)
// =============================================================================

fn cmd_quality(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_cli quality <subcommand>");
        eprintln!("  gates list    — List configured quality gates");
        eprintln!("  gates check   — Run quality gates on text");
        return ExitCode::from(1);
    }

    match args[0].as_str() {
        "gates" => {
            if args.len() < 2 {
                eprintln!("Usage: ai_cli quality gates <list|check>");
                return ExitCode::from(1);
            }
            match args[1].as_str() {
                "list" => {
                    #[cfg(feature = "eval")]
                    {
                        let runner =
                            ai_assistant::quality_gates::QualityGateRunner::production_defaults();
                        println!("Configured quality gates:");
                        for gate in runner.gates() {
                            println!(
                                "  {} — {:?} >= {:.2} (action: {:?})",
                                gate.name, gate.metric, gate.threshold, gate.action
                            );
                        }
                    }
                    #[cfg(not(feature = "eval"))]
                    {
                        eprintln!("Quality gates require the 'eval' feature");
                        return ExitCode::from(1);
                    }
                }
                "check" => {
                    #[cfg(feature = "eval")]
                    {
                        let text = args[2..].join(" ");
                        if text.is_empty() {
                            eprintln!("Usage: ai_cli quality gates check <text>");
                            return ExitCode::from(1);
                        }

                        // Compute real confidence score
                        let conf_scorer = ai_assistant::confidence_scoring::ConfidenceScorer::new(
                            ai_assistant::confidence_scoring::ConfidenceConfig::default(),
                        );
                        let conf_score = conf_scorer.score(&text, None);

                        // Compute faithfulness (self-reference baseline)
                        let faith_scorer = ai_assistant::faithfulness::FaithfulnessScorer::new(
                            ai_assistant::faithfulness::FaithfulnessConfig::default(),
                        );
                        let faith_report = faith_scorer.score(&text, &[&text]);

                        let scores = ai_assistant::quality_gates::QualityScores {
                            faithfulness: Some(faith_report.overall_score),
                            confidence: Some(conf_score.overall),
                            grounding_ratio: Some(faith_report.grounding_ratio()),
                            consistency_score: None,
                            citation_coverage: None,
                        };

                        let runner =
                            ai_assistant::quality_gates::QualityGateRunner::production_defaults();
                        let result = runner.run(&scores);

                        println!("Quality check on {} chars of text:", text.len());
                        println!("  Confidence:      {:.2}", conf_score.overall);
                        println!("  Faithfulness:    {:.2}", faith_report.overall_score);
                        println!("  Grounding ratio: {:.2}", faith_report.grounding_ratio());
                        println!(
                            "  Result: {} ({}/{})",
                            result.summary(),
                            result.passed_count(),
                            result.total_checked()
                        );
                        if !result.warnings.is_empty() {
                            for w in &result.warnings {
                                println!("  Warning: {}", w);
                            }
                        }
                        if !result.passed {
                            for f in &result.failing_gates {
                                println!("  FAILED: {}", f);
                            }
                        }
                    }
                    #[cfg(not(feature = "eval"))]
                    {
                        eprintln!("Quality gates require the 'eval' feature");
                        return ExitCode::from(1);
                    }
                }
                other => {
                    eprintln!("Unknown quality gates subcommand: {}", other);
                    return ExitCode::from(1);
                }
            }
        }
        other => {
            eprintln!("Unknown quality subcommand: {}", other);
            return ExitCode::from(1);
        }
    }

    ExitCode::SUCCESS
}

// =============================================================================
// benchmark — dataset hallucination / faithfulness benchmarks (V90)
// =============================================================================

fn cmd_benchmark(args: &[String]) -> ExitCode {
    #[cfg(not(feature = "eval"))]
    {
        let _ = args;
        eprintln!("Error: 'benchmark' requires the 'eval' feature.");
        eprintln!("  cargo run --bin ai_cli --features \"full,eval\" -- benchmark list");
        ExitCode::from(1)
    }
    #[cfg(feature = "eval")]
    {
        if args.is_empty() {
            print_benchmark_usage();
            return ExitCode::from(1);
        }
        match args[0].as_str() {
            "help" | "-h" | "--help" => {
                print_benchmark_usage();
                ExitCode::SUCCESS
            }
            "list" => cmd_benchmark_list(),
            "info" => cmd_benchmark_info(&args[1..]),
            "download" => cmd_benchmark_download(&args[1..]),
            "run" => cmd_benchmark_run(&args[1..]),
            "calibrate" => cmd_benchmark_calibrate(&args[1..]),
            other => {
                eprintln!("Error: unknown benchmark subcommand '{}'\n", other);
                print_benchmark_usage();
                ExitCode::from(1)
            }
        }
    }
}

#[cfg(feature = "eval")]
fn print_benchmark_usage() {
    println!("ai_cli benchmark — dataset hallucination / faithfulness benchmarks\n");
    println!("Usage: ai_cli benchmark <subcommand> [options]\n");
    println!("Subcommands:");
    println!("  list                                 List available benchmarks");
    println!("  info <name>                          Show metadata (license, citation, URL)");
    println!("  download <name> [--accept-license] [--cache-dir <path>]");
    println!("                                       Fetch the dataset into the cache");
    println!("  run <name> --provider X --model Y [--limit N] [--threshold 0.5]");
    println!("             [--cache-dir <path>] [--json]");
    println!("                                       Run the model against the benchmark");
    println!("  calibrate <name> --provider X --model Y [--limit N]");
    println!("                   [--objective accuracy|f1] [--cache-dir <path>] [--json]");
    println!("                                       Sweep the correctness threshold post-hoc");
    println!();
    println!("Notes:");
    println!("  * FEVER and other datasets with share-alike terms require --accept-license");
    println!("    on 'download' before anything will be fetched.");
    println!("  * --cache-dir defaults to target/eval_benchmarks (respects CARGO_TARGET_DIR).");
    println!();
    println!("Examples:");
    println!("  ai_cli benchmark list");
    println!("  ai_cli benchmark info truthfulqa");
    println!("  ai_cli benchmark download fever --accept-license");
    println!("  ai_cli benchmark run truthfulqa --provider ollama --model mistral:7b --limit 20");
    println!("  ai_cli benchmark calibrate halueval --provider ollama --model llama3.2 \\");
    println!("      --limit 50 --objective f1 --json");
}

#[cfg(feature = "eval")]
fn cmd_benchmark_list() -> ExitCode {
    let loaders = ai_assistant::eval_benchmarks::all_loaders();
    println!("Available benchmarks ({}):\n", loaders.len());
    for l in &loaders {
        let opt = if l.requires_opt_in() { " [opt-in]" } else { "" };
        println!(
            "  {:<12} {} — {}{}",
            l.name(),
            format!("[{:?}]", l.sample_type()),
            l.description(),
            opt
        );
    }
    println!();
    ExitCode::SUCCESS
}

#[cfg(feature = "eval")]
fn cmd_benchmark_info(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: ai_cli benchmark info <name>");
        return ExitCode::from(1);
    }
    let name = &args[0];
    let loader = match ai_assistant::eval_benchmarks::get_loader(name) {
        Some(l) => l,
        None => {
            eprintln!(
                "Error: unknown benchmark '{}'. Try: ai_cli benchmark list",
                name
            );
            return ExitCode::from(1);
        }
    };
    println!("Benchmark: {}", loader.name());
    println!("  Type:       {:?}", loader.sample_type());
    println!("  Description: {}", loader.description());
    println!("  License:    {}", loader.license());
    println!("  Citation:   {}", loader.citation());
    println!("  Opt-in:     {}", loader.requires_opt_in());
    println!("  URLs:");
    for u in loader.download_urls() {
        println!("    - {}", u);
    }
    println!();
    ExitCode::SUCCESS
}

#[cfg(feature = "eval")]
fn resolve_cache_dir(args: &[String]) -> std::path::PathBuf {
    if let Some(p) = find_flag_value(args, "--cache-dir") {
        std::path::PathBuf::from(p)
    } else {
        ai_assistant::eval_benchmarks::BenchmarkCache::default_root()
            .root()
            .to_path_buf()
    }
}

#[cfg(feature = "eval")]
fn cmd_benchmark_download(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!(
            "Usage: ai_cli benchmark download <name> [--accept-license] [--cache-dir <path>]"
        );
        return ExitCode::from(1);
    }
    let name = &args[0];
    let rest = &args[1..];
    let accept = rest.iter().any(|a| a == "--accept-license");

    let loader = match ai_assistant::eval_benchmarks::get_loader(name) {
        Some(l) => l,
        None => {
            eprintln!(
                "Error: unknown benchmark '{}'. Try: ai_cli benchmark list",
                name
            );
            return ExitCode::from(1);
        }
    };

    if loader.requires_opt_in() && !accept {
        eprintln!(
            "Error: benchmark '{}' requires explicit license acceptance.",
            loader.name()
        );
        eprintln!("  License: {}", loader.license());
        eprintln!("  Re-run with --accept-license to proceed.");
        return ExitCode::from(1);
    }

    let cache = ai_assistant::eval_benchmarks::BenchmarkCache::with_root(resolve_cache_dir(rest));
    let dir = match cache.dir_for(loader.name()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: cannot prepare cache dir: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!("[benchmark] cache dir: {}", dir.display());
    eprintln!("[benchmark] downloading {} ...", loader.name());
    match loader.download(&dir) {
        Ok(path) => {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("Downloaded: {}", path.display());
            println!("  Size: {} bytes", size);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error downloading benchmark: {}", e);
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "eval")]
struct BenchmarkRunSetup {
    loader: Box<dyn ai_assistant::eval_benchmarks::BenchmarkLoader>,
    samples: Vec<ai_assistant::eval_benchmarks::BenchmarkSample>,
    limit: Option<usize>,
    threshold: f64,
    json: bool,
    provider: ai_assistant::AiProvider,
    model: String,
    ollama_url: String,
    lm_studio_url: String,
    custom_url: String,
}

#[cfg(feature = "eval")]
fn prepare_benchmark_run(args: &[String]) -> Result<BenchmarkRunSetup, ExitCode> {
    if args.is_empty() {
        eprintln!("Error: missing benchmark name.");
        return Err(ExitCode::from(1));
    }
    let name = &args[0];
    let rest = &args[1..];

    let loader = match ai_assistant::eval_benchmarks::get_loader(name) {
        Some(l) => l,
        None => {
            eprintln!(
                "Error: unknown benchmark '{}'. Try: ai_cli benchmark list",
                name
            );
            return Err(ExitCode::from(1));
        }
    };

    let provider_name = match find_flag_value(rest, "--provider") {
        Some(v) => v.to_string(),
        None => {
            eprintln!("Error: --provider is required (e.g. --provider ollama).");
            return Err(ExitCode::from(1));
        }
    };
    let model = match find_flag_value(rest, "--model") {
        Some(v) => v.to_string(),
        None => {
            eprintln!("Error: --model is required.");
            return Err(ExitCode::from(1));
        }
    };
    let url_override = find_flag_value(rest, "--url").map(String::from);
    let limit = find_flag_value(rest, "--limit").and_then(|s| s.parse::<usize>().ok());
    let threshold = find_flag_value(rest, "--threshold")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.5);
    let json = rest.iter().any(|a| a == "--json");

    let cache = ai_assistant::eval_benchmarks::BenchmarkCache::with_root(resolve_cache_dir(rest));
    let dir = cache.dir_for(loader.name()).map_err(|e| {
        eprintln!("Error: cannot prepare cache dir: {}", e);
        ExitCode::from(1)
    })?;

    if loader.requires_opt_in() {
        eprintln!(
            "Note: benchmark '{}' is opt-in; cached data reuses a previous --accept-license.",
            loader.name()
        );
    }

    let dataset_path = loader.download(&dir).map_err(|e| {
        eprintln!("Error loading benchmark data: {}", e);
        eprintln!(
            "  Hint: run 'ai_cli benchmark download {}{}' first.",
            loader.name(),
            if loader.requires_opt_in() {
                " --accept-license"
            } else {
                ""
            }
        );
        ExitCode::from(1)
    })?;

    let samples = loader.load(&dataset_path, limit).map_err(|e| {
        eprintln!("Error parsing benchmark dataset: {}", e);
        ExitCode::from(1)
    })?;

    let provider = provider_from_name(&provider_name);
    let (ollama_url, lm_studio_url, custom_url) = {
        let defaults = AiConfig::default();
        let mut o = defaults.ollama_url.clone();
        let mut l = defaults.lm_studio_url.clone();
        let mut c = defaults.custom_url.clone();
        if let Some(u) = url_override {
            match provider {
                ai_assistant::AiProvider::Ollama => o = u,
                ai_assistant::AiProvider::LMStudio => l = u,
                _ => c = u,
            }
        }
        (o, l, c)
    };

    Ok(BenchmarkRunSetup {
        loader,
        samples,
        limit,
        threshold,
        json,
        provider,
        model,
        ollama_url,
        lm_studio_url,
        custom_url,
    })
}

#[cfg(feature = "eval")]
fn query_llm_for_benchmark(
    provider: &ai_assistant::AiProvider,
    model: &str,
    ollama_url: &str,
    lm_studio_url: &str,
    custom_url: &str,
    prompt: &str,
) -> Result<String, String> {
    let mut assistant = AiAssistant::new();
    assistant.config.provider = provider.clone();
    assistant.config.selected_model = model.to_string();
    assistant.config.ollama_url = ollama_url.to_string();
    assistant.config.lm_studio_url = lm_studio_url.to_string();
    assistant.config.custom_url = custom_url.to_string();
    assistant.config.temperature = 0.1;

    assistant.send_message(prompt.to_string(), "");
    let start = Instant::now();
    let deadline = Duration::from_secs(90);
    let mut out = String::new();
    loop {
        if start.elapsed() > deadline {
            return Err(format!("timeout after {}s", deadline.as_secs()));
        }
        if let Some(response) = assistant.poll_response() {
            match response {
                AiResponse::Chunk(t) => out.push_str(&t),
                AiResponse::Complete(t) => {
                    if out.is_empty() {
                        out = t;
                    }
                    return Ok(out);
                }
                AiResponse::Error(e) => return Err(e),
                AiResponse::Cancelled(t) => return Ok(t),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(feature = "eval")]
fn cmd_benchmark_run(args: &[String]) -> ExitCode {
    let setup = match prepare_benchmark_run(args) {
        Ok(s) => s,
        Err(code) => return code,
    };

    let opts = ai_assistant::eval_benchmarks::RunOptions {
        limit: setup.limit,
        correctness_threshold: setup.threshold,
        max_consecutive_errors: 10,
    };

    eprintln!(
        "[benchmark] running {} with {} samples (limit={:?})",
        setup.loader.name(),
        setup.samples.len(),
        setup.limit
    );

    let provider = setup.provider.clone();
    let model = setup.model.clone();
    let ollama = setup.ollama_url.clone();
    let lm = setup.lm_studio_url.clone();
    let custom = setup.custom_url.clone();
    let report = ai_assistant::eval_benchmarks::run(
        setup.loader.name(),
        &setup.samples,
        &opts,
        move |prompt: &str| {
            query_llm_for_benchmark(&provider, &model, &ollama, &lm, &custom, prompt)
        },
    );

    if setup.json {
        println!(
            "{}",
            ai_assistant::eval_benchmarks::report::to_json(&report)
        );
    } else {
        println!(
            "{}",
            ai_assistant::eval_benchmarks::report::to_text(&report)
        );
    }
    ExitCode::SUCCESS
}

#[cfg(feature = "eval")]
fn cmd_benchmark_calibrate(args: &[String]) -> ExitCode {
    let objective = match find_flag_value(args, "--objective").unwrap_or("accuracy") {
        "accuracy" => ai_assistant::eval_benchmarks::Objective::Accuracy,
        "f1" => ai_assistant::eval_benchmarks::Objective::F1,
        other => {
            eprintln!(
                "Error: --objective must be 'accuracy' or 'f1' (got '{}')",
                other
            );
            return ExitCode::from(1);
        }
    };

    let setup = match prepare_benchmark_run(args) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let json = setup.json;

    let opts = ai_assistant::eval_benchmarks::RunOptions {
        limit: setup.limit,
        correctness_threshold: 0.5,
        max_consecutive_errors: 10,
    };

    eprintln!(
        "[benchmark] running {} for calibration with {} samples",
        setup.loader.name(),
        setup.samples.len()
    );

    let provider = setup.provider.clone();
    let model = setup.model.clone();
    let ollama = setup.ollama_url.clone();
    let lm = setup.lm_studio_url.clone();
    let custom = setup.custom_url.clone();
    let report = ai_assistant::eval_benchmarks::run(
        setup.loader.name(),
        &setup.samples,
        &opts,
        move |prompt: &str| {
            query_llm_for_benchmark(&provider, &model, &ollama, &lm, &custom, prompt)
        },
    );

    let grid = ai_assistant::eval_benchmarks::default_grid();
    let calibration = ai_assistant::eval_benchmarks::sweep(&report, &grid, objective);

    if json {
        println!(
            "{}",
            ai_assistant::eval_benchmarks::report::calibration_to_json(&calibration)
        );
    } else {
        println!(
            "{}",
            ai_assistant::eval_benchmarks::report::calibration_to_text(&calibration)
        );
    }
    ExitCode::SUCCESS
}

// =============================================================================
// Recipes (Phase A.1) — declarative YAML workflows
// =============================================================================

fn cmd_recipes(args: &[String]) -> ExitCode {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        eprintln!("Usage: ai_cli recipes <list|show|validate|init|run|share> [options]");
        eprintln!();
        eprintln!("Subcommands:");
        eprintln!("  list                          List discovered recipes");
        eprintln!("  show <name>                   Show recipe definition");
        eprintln!("  validate <name|path>          Validate schema");
        eprintln!("  init <name> [--out <path>]    Scaffold a new recipe template");
        eprintln!("  run <name> [--var k=v ...]    Execute a recipe end-to-end");
        eprintln!("  share <name> [--out <path>]   Produce a portable bundle");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --user-dir <path>             Override user-global recipes dir");
        eprintln!("  --project-dir <path>          Override project recipes dir");
        eprintln!("  --provider <name>             Provider for prompt steps (default: ollama)");
        eprintln!("  --model <name>                Model name");
        eprintln!("  --url <url>                   Provider URL");
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "list" => cmd_recipes_list(&args[1..]),
        "show" => cmd_recipes_show(&args[1..]),
        "validate" => cmd_recipes_validate(&args[1..]),
        "init" => cmd_recipes_init(&args[1..]),
        "run" => cmd_recipes_run(&args[1..]),
        "share" => cmd_recipes_share(&args[1..]),
        other => {
            eprintln!("Unknown recipes subcommand: '{}'", other);
            ExitCode::from(2)
        }
    }
}

fn recipe_roots(args: &[String]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(p) = get_arg(args, "--user-dir") {
        roots.push(PathBuf::from(p));
    } else if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
    {
        let mut p = PathBuf::from(home);
        p.push(".config");
        p.push("ai_assistant");
        p.push("recipes");
        roots.push(p);
    }
    if let Some(p) = get_arg(args, "--project-dir") {
        roots.push(PathBuf::from(p));
    } else {
        let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        p.push(".ai_assistant");
        p.push("recipes");
        roots.push(p);
    }
    roots
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            return iter.next().cloned();
        }
    }
    None
}

fn cmd_recipes_list(args: &[String]) -> ExitCode {
    let roots = recipe_roots(args);
    let cfg = ai_assistant::RecipeConfig::default();
    let reg = ai_assistant::discover_recipes(&roots, &cfg);
    if reg.is_empty() {
        println!("No recipes found in:");
        for r in &roots {
            println!("  {}", r.display());
        }
        if !reg.load_errors.is_empty() {
            println!("\nLoad errors:");
            for (p, e) in &reg.load_errors {
                println!("  {} — {}", p.display(), e);
            }
        }
        return ExitCode::SUCCESS;
    }
    println!("{:<24} {:<10} {}", "NAME", "VERSION", "DESCRIPTION");
    for (name, r) in reg.iter() {
        println!(
            "{:<24} {:<10} {}",
            name,
            r.version.as_deref().unwrap_or("-"),
            r.description.as_deref().unwrap_or("")
        );
    }
    if !reg.load_errors.is_empty() {
        eprintln!(
            "\n  {} recipe(s) failed to load (use --debug)",
            reg.load_errors.len()
        );
    }
    ExitCode::SUCCESS
}

fn cmd_recipes_show(args: &[String]) -> ExitCode {
    let Some(name) = args.first() else {
        eprintln!("Usage: ai_cli recipes show <name>");
        return ExitCode::from(2);
    };
    let roots = recipe_roots(args);
    let cfg = ai_assistant::RecipeConfig::default();
    let reg = ai_assistant::discover_recipes(&roots, &cfg);
    let Some(r) = reg.get(name) else {
        eprintln!("Recipe not found: '{}'", name);
        return ExitCode::from(1);
    };
    println!("Name:        {}", r.name);
    println!("Version:     {}", r.version.as_deref().unwrap_or("-"));
    println!("API:         {}", r.api_version);
    if let Some(d) = &r.description {
        println!("Description: {}", d);
    }
    if let Some(a) = &r.author {
        println!("Author:      {}", a);
    }
    if !r.tags.is_empty() {
        println!("Tags:        {}", r.tags.join(", "));
    }
    if let Some(m) = &r.model {
        println!("Model:       {}", m);
    }
    println!("Source:      {}", r.source_path.display());
    println!();
    if !r.variables.is_empty() {
        println!("Variables:");
        for v in &r.variables {
            println!(
                "  {} (required={}, default={})",
                v.name,
                v.required,
                v.default.as_deref().unwrap_or("-")
            );
            if let Some(d) = &v.description {
                println!("    {}", d);
            }
        }
        println!();
    }
    println!("Steps ({}):", r.steps.len());
    for s in &r.steps {
        let kind = match &s.kind {
            ai_assistant::StepKind::Prompt { .. } => "prompt",
            ai_assistant::StepKind::Tool { tool, .. } => &format!("tool:{}", tool)[..],
            ai_assistant::StepKind::Recipe { recipe, .. } => &format!("recipe:{}", recipe)[..],
            ai_assistant::StepKind::Shell { .. } => "shell",
        };
        println!("  - {} ({})", s.id, kind);
    }
    ExitCode::SUCCESS
}

fn cmd_recipes_validate(args: &[String]) -> ExitCode {
    let Some(target) = args.first() else {
        eprintln!("Usage: ai_cli recipes validate <name|path>");
        return ExitCode::from(2);
    };
    let cfg = ai_assistant::RecipeConfig::default();
    let r = if target.ends_with(".yaml") || target.ends_with(".yml") {
        let p = PathBuf::from(target);
        let text = match std::fs::read_to_string(&p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("read error: {}", e);
                return ExitCode::from(1);
            }
        };
        match ai_assistant::parse_recipe(&text, &p) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("parse error: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        let roots = recipe_roots(args);
        let reg = ai_assistant::discover_recipes(&roots, &cfg);
        match reg.get(target) {
            Some(r) => r.clone(),
            None => {
                eprintln!("Recipe not found: '{}'", target);
                return ExitCode::from(1);
            }
        }
    };
    match ai_assistant::validate_recipe(&r, &cfg) {
        Ok(()) => {
            println!(
                "OK: '{}' (apiVersion={}, steps={})",
                r.name,
                r.api_version,
                r.steps.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("INVALID: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_recipes_init(args: &[String]) -> ExitCode {
    let Some(name) = args.first() else {
        eprintln!("Usage: ai_cli recipes init <name> [--out <path>]");
        return ExitCode::from(2);
    };
    let body = ai_assistant::scaffold_recipe(name);
    let out = get_arg(args, "--out").unwrap_or_else(|| format!("{}.yaml", name.to_lowercase()));
    if std::path::Path::new(&out).exists() {
        eprintln!("Refusing to overwrite existing file: {}", out);
        return ExitCode::from(1);
    }
    if let Err(e) = std::fs::write(&out, body) {
        eprintln!("write error: {}", e);
        return ExitCode::from(1);
    }
    println!("Scaffolded: {}", out);
    ExitCode::SUCCESS
}

fn cmd_recipes_run(args: &[String]) -> ExitCode {
    let Some(name) = args.first() else {
        eprintln!("Usage: ai_cli recipes run <name> [--var k=v ...]");
        return ExitCode::from(2);
    };
    let roots = recipe_roots(args);
    let cfg = ai_assistant::RecipeConfig::default();
    let reg = ai_assistant::discover_recipes(&roots, &cfg);
    let Some(r) = reg.get(name) else {
        eprintln!("Recipe not found: '{}'", name);
        return ExitCode::from(1);
    };

    // Collect --var k=v
    let mut bindings: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "--var" && i + 1 < args.len() {
            if let Some((k, v)) = args[i + 1].split_once('=') {
                bindings.insert(k.trim().to_string(), v.trim().to_string());
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    // Build LLM callback
    let provider_name = get_arg(args, "--provider")
        .or_else(|| r.provider.clone())
        .unwrap_or_else(|| "ollama".into());
    let model_name = get_arg(args, "--model")
        .or_else(|| r.model.clone())
        .unwrap_or_default();
    let url_override = get_arg(args, "--url");

    let llm_provider = provider_name.clone();
    let llm_model = model_name.clone();
    let llm_url = url_override.clone();

    let engine = ai_assistant::RecipeEngine::default()
        .with_llm(move |prompt| {
            let mut a = AiAssistant::new();
            a.config.provider = provider_from_name(&llm_provider);
            if !llm_model.is_empty() {
                a.config.selected_model = llm_model.clone();
            }
            if let Some(u) = &llm_url {
                a.config.ollama_url = u.clone();
                a.config.lm_studio_url = u.clone();
                a.config.custom_url = u.clone();
            }
            a.config.temperature = 0.2;
            a.send_message(prompt.to_string(), "");
            let mut full = String::new();
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                if Instant::now() > deadline {
                    return None;
                }
                if let Some(resp) = a.poll_response() {
                    match resp {
                        AiResponse::Chunk(t) => full.push_str(&t),
                        AiResponse::Complete(t) => {
                            if full.is_empty() {
                                full = t;
                            }
                            return Some(full);
                        }
                        AiResponse::Error(_) => return None,
                        AiResponse::Cancelled(t) => return Some(t),
                        _ => {}
                    }
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        })
        .with_tool(|tool, args_map| {
            // Built-in minimal tool registry: file_read, echo
            match tool {
                "echo" => Some(args_map.get("msg").cloned().unwrap_or_default()),
                "file_read" => {
                    let p = args_map.get("path")?;
                    std::fs::read_to_string(p).ok()
                }
                _ => None,
            }
        });

    match engine.run(r, &bindings, &reg) {
        Ok(result) => {
            println!("--- Recipe '{}' ---", result.recipe_name);
            for s in &result.steps {
                println!("[{}]", s.step_id);
                println!("{}", s.output.trim_end());
                println!();
            }
            println!("--- Output ---");
            println!("{}", result.final_output);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Recipe failed: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_recipes_share(args: &[String]) -> ExitCode {
    let Some(name) = args.first() else {
        eprintln!("Usage: ai_cli recipes share <name> [--out <path>]");
        return ExitCode::from(2);
    };
    let roots = recipe_roots(args);
    let cfg = ai_assistant::RecipeConfig::default();
    let reg = ai_assistant::discover_recipes(&roots, &cfg);
    let Some(r) = reg.get(name) else {
        eprintln!("Recipe not found: '{}'", name);
        return ExitCode::from(1);
    };
    let body = match std::fs::read_to_string(&r.source_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read error: {}", e);
            return ExitCode::from(1);
        }
    };
    let out = get_arg(args, "--out").unwrap_or_else(|| format!("{}.shared.yaml", name));
    if let Err(e) = std::fs::write(&out, body) {
        eprintln!("write error: {}", e);
        return ExitCode::from(1);
    }
    println!("Shared bundle: {}", out);
    ExitCode::SUCCESS
}

// =============================================================================
// Tests (V77 — cost subcommand helpers)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_flag_value_present() {
        let args: Vec<String> = vec!["report".into(), "--snapshot".into(), "/tmp/x.json".into()];
        assert_eq!(find_flag_value(&args, "--snapshot"), Some("/tmp/x.json"));
    }

    #[test]
    fn test_find_flag_value_absent() {
        let args: Vec<String> = vec!["report".into()];
        assert_eq!(find_flag_value(&args, "--snapshot"), None);
    }

    #[test]
    fn test_find_flag_value_dangling_flag() {
        // Flag with no value should return None, not panic.
        let args: Vec<String> = vec!["export".into(), "--snapshot".into()];
        assert_eq!(find_flag_value(&args, "--snapshot"), None);
    }

    #[test]
    fn test_find_flag_value_multiple_flags() {
        let args: Vec<String> = vec![
            "export".into(),
            "--snapshot".into(),
            "a.json".into(),
            "--output".into(),
            "b.csv".into(),
            "--force".into(),
        ];
        assert_eq!(find_flag_value(&args, "--snapshot"), Some("a.json"));
        assert_eq!(find_flag_value(&args, "--output"), Some("b.csv"));
        assert_eq!(find_flag_value(&args, "--force"), None);
    }

    #[test]
    fn test_load_cost_snapshot_invalid_path() {
        let res = load_cost_snapshot("/__definitely_nowhere__/snap.json");
        assert!(res.is_err());
        let msg = res.err().unwrap();
        assert!(msg.contains("cannot resolve"));
    }

    #[test]
    fn test_provider_from_name_roundtrip() {
        use ai_assistant::AiProvider;
        assert!(matches!(provider_from_name("openai"), AiProvider::OpenAI));
        assert!(matches!(
            provider_from_name("anthropic"),
            AiProvider::Anthropic
        ));
        assert!(matches!(
            provider_from_name("unknown_xyz"),
            AiProvider::Ollama
        ));
    }
}
