//! `ai_recipes` — Recipes auditor CLI.
//!
//! Inspect, validate, and audit recipe definitions discovered under the
//! standard recipe roots. Read-only — never mutates recipes
//! (per memory rule `feedback_auditable_subsystems`).
//!
//! # Usage
//!
//! ```text
//! ai_recipes list [--dir <PATH>]                List all recipes in dir
//! ai_recipes inspect <FILE|NAME> [--dir <PATH>] Show full metadata + steps
//! ai_recipes validate <FILE|NAME> [--dir <PATH>] Validate schema
//! ai_recipes graph [--dir <PATH>]               Show sub-recipe call graph
//! ai_recipes audit [--dir <PATH>]               Aggregate audit (counts, issues)
//! ```

use ai_assistant::{
    discover_recipes, parse_recipe, validate_recipe, RecipeConfig, RecipeError, StepKind,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    let command = args[1].as_str();
    match command {
        "list" => {
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(default_dir);
            cmd_list(&dir)
        }
        "inspect" => {
            let Some(target) = args.get(2) else {
                eprintln!("Usage: ai_recipes inspect <FILE|NAME>");
                return ExitCode::from(2);
            };
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(default_dir);
            cmd_inspect(target, &dir)
        }
        "validate" => {
            let Some(target) = args.get(2) else {
                eprintln!("Usage: ai_recipes validate <FILE|NAME>");
                return ExitCode::from(2);
            };
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(default_dir);
            cmd_validate(target, &dir)
        }
        "graph" => {
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(default_dir);
            cmd_graph(&dir)
        }
        "audit" => {
            let dir = get_arg(&args, "--dir")
                .map(PathBuf::from)
                .unwrap_or_else(default_dir);
            cmd_audit(&dir)
        }
        other => {
            eprintln!("Unknown command: {other}. Use --help.");
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    println!("ai_recipes — Recipes auditor CLI (read-only)\n");
    println!("Usage:");
    println!("  ai_recipes list [--dir PATH]");
    println!("  ai_recipes inspect <FILE|NAME> [--dir PATH]");
    println!("  ai_recipes validate <FILE|NAME> [--dir PATH]");
    println!("  ai_recipes graph [--dir PATH]");
    println!("  ai_recipes audit [--dir PATH]");
    println!();
    println!("Default --dir: ./.ai_assistant/recipes");
}

fn default_dir() -> PathBuf {
    let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    p.push(".ai_assistant");
    p.push("recipes");
    p
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

fn cmd_list(dir: &Path) -> ExitCode {
    if !dir.is_dir() {
        eprintln!("Not a directory: {}", dir.display());
        return ExitCode::from(1);
    }
    let cfg = RecipeConfig::default();
    let reg = discover_recipes(&[dir.to_path_buf()], &cfg);
    if reg.is_empty() {
        println!("No recipes found in {}", dir.display());
        if !reg.load_errors.is_empty() {
            for (p, e) in &reg.load_errors {
                eprintln!("  load error: {} — {}", p.display(), e);
            }
        }
        return ExitCode::SUCCESS;
    }
    println!(
        "{:<24} {:<10} {:<6} DESCRIPTION",
        "NAME", "VERSION", "STEPS"
    );
    for (name, r) in reg.iter() {
        println!(
            "{:<24} {:<10} {:<6} {}",
            name,
            r.version.as_deref().unwrap_or("-"),
            r.steps.len(),
            r.description.as_deref().unwrap_or("")
        );
    }
    ExitCode::SUCCESS
}

fn cmd_inspect(target: &str, dir: &Path) -> ExitCode {
    let cfg = RecipeConfig::default();
    let recipe = if target.ends_with(".yaml") || target.ends_with(".yml") {
        let p = PathBuf::from(target);
        match fs::read_to_string(&p)
            .ok()
            .and_then(|t| parse_recipe(&t, &p).ok())
        {
            Some(r) => r,
            None => {
                eprintln!("Failed to read or parse {}", p.display());
                return ExitCode::from(1);
            }
        }
    } else {
        let reg = discover_recipes(&[dir.to_path_buf()], &cfg);
        match reg.get(target) {
            Some(r) => r.clone(),
            None => {
                eprintln!("Recipe not found: {}", target);
                return ExitCode::from(1);
            }
        }
    };
    println!("Name:        {}", recipe.name);
    println!("API:         {}", recipe.api_version);
    println!("Version:     {}", recipe.version.as_deref().unwrap_or("-"));
    println!("Author:      {}", recipe.author.as_deref().unwrap_or("-"));
    if let Some(d) = &recipe.description {
        println!("Description: {}", d);
    }
    if !recipe.tags.is_empty() {
        println!("Tags:        {}", recipe.tags.join(", "));
    }
    if let Some(m) = &recipe.model {
        println!("Model:       {}", m);
    }
    if let Some(p) = &recipe.provider {
        println!("Provider:    {}", p);
    }
    println!("Source:      {}", recipe.source_path.display());
    println!();
    if !recipe.variables.is_empty() {
        println!("Variables ({}):", recipe.variables.len());
        for v in &recipe.variables {
            println!(
                "  {:<16} required={:<5} default={}",
                v.name,
                v.required,
                v.default.as_deref().unwrap_or("-")
            );
        }
        println!();
    }
    println!("Steps ({}):", recipe.steps.len());
    for s in &recipe.steps {
        let (kind, detail) = match &s.kind {
            StepKind::Prompt { prompt } => ("prompt", short(prompt)),
            StepKind::Tool { tool, args } => ("tool", format!("{} {} args", tool, args.len())),
            StepKind::Recipe { recipe, args } => {
                ("recipe", format!("{} {} args", recipe, args.len()))
            }
            StepKind::Shell { command } => ("shell", short(command)),
        };
        println!("  - {:<16} [{:<6}] {}", s.id, kind, detail);
    }
    if let Some(o) = &recipe.output {
        println!("\nOutput template: {}", short(o));
    }
    ExitCode::SUCCESS
}

fn short(s: &str) -> String {
    let one = s.replace('\n', " ");
    if one.len() > 72 {
        format!("{}…", &one[..72])
    } else {
        one
    }
}

fn cmd_validate(target: &str, dir: &Path) -> ExitCode {
    let cfg = RecipeConfig::default();
    let recipe = if target.ends_with(".yaml") || target.ends_with(".yml") {
        let p = PathBuf::from(target);
        let text = match fs::read_to_string(&p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("read error: {}", e);
                return ExitCode::from(1);
            }
        };
        match parse_recipe(&text, &p) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("INVALID parse: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        let reg = discover_recipes(&[dir.to_path_buf()], &cfg);
        match reg.get(target) {
            Some(r) => r.clone(),
            None => {
                eprintln!("Recipe not found: {}", target);
                return ExitCode::from(1);
            }
        }
    };
    match validate_recipe(&recipe, &cfg) {
        Ok(()) => {
            println!(
                "OK: {} (apiVersion={}, steps={})",
                recipe.name,
                recipe.api_version,
                recipe.steps.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("INVALID: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_graph(dir: &Path) -> ExitCode {
    let cfg = RecipeConfig::default();
    let reg = discover_recipes(&[dir.to_path_buf()], &cfg);
    if reg.is_empty() {
        println!("No recipes in {}", dir.display());
        return ExitCode::SUCCESS;
    }
    println!("Sub-recipe call graph:");
    let mut callees: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, r) in reg.iter() {
        for s in &r.steps {
            if let StepKind::Recipe { recipe, .. } = &s.kind {
                callees
                    .entry(name.clone())
                    .or_default()
                    .push(recipe.clone());
            }
        }
    }
    for (caller, called) in &callees {
        for c in called {
            let exists = if reg.get(c).is_some() {
                ""
            } else {
                "  [MISSING]"
            };
            println!("  {} -> {}{}", caller, c, exists);
        }
    }
    if callees.is_empty() {
        println!("  (no sub-recipe calls)");
    }
    ExitCode::SUCCESS
}

fn cmd_audit(dir: &Path) -> ExitCode {
    let cfg = RecipeConfig::default();
    let reg = discover_recipes(&[dir.to_path_buf()], &cfg);
    let total = reg.len();
    let mut valid = 0usize;
    let mut invalid = 0usize;
    let mut shell_count = 0usize;
    let mut step_total = 0usize;
    let mut errors: Vec<String> = Vec::new();
    for (_, r) in reg.iter() {
        step_total += r.steps.len();
        for s in &r.steps {
            if matches!(s.kind, StepKind::Shell { .. }) {
                shell_count += 1;
            }
        }
        match validate_recipe(r, &cfg) {
            Ok(()) => valid += 1,
            Err(e) => {
                invalid += 1;
                errors.push(format!("  {} — {}", r.name, e));
            }
        }
    }
    println!("Recipes audit ({})", dir.display());
    println!("  Discovered:    {}", total);
    println!("  Valid:         {}", valid);
    println!("  Invalid:       {}", invalid);
    println!("  Total steps:   {}", step_total);
    println!("  Shell steps:   {} (security-sensitive)", shell_count);
    if !reg.load_errors.is_empty() {
        println!("\nLoad errors ({}):", reg.load_errors.len());
        for (p, e) in &reg.load_errors {
            println!("  {} — {}", p.display(), e);
        }
    }
    if !errors.is_empty() {
        println!("\nValidation errors:");
        for e in &errors {
            println!("{}", e);
        }
    }
    if invalid > 0 || !reg.load_errors.is_empty() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

// suppress dead_code warning if RecipeError is unused via direct path
#[allow(dead_code)]
fn _force_link_recipe_error(_e: RecipeError) {}
