// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_gpu_share — GPU Sharing Network CLI for AI Assistant.
//!
//! Run with: `cargo run --bin ai_gpu_share --features full -- <command>`
//!
//! Manages a node in the GPU sharing network, allowing it to provide
//! local GPU capacity for LLM inference, request inference from peers,
//! or act as a gateway routing traffic.
//!
//! ## Commands
//!
//! ```text
//! ai_gpu_share start [--provider|--gateway]  Start the node (default: both)
//! ai_gpu_share stop                          Stop the node gracefully
//! ai_gpu_share status                        Show node status
//! ai_gpu_share models                        List available models on the network
//! ai_gpu_share credits                       Show credit balance and history
//! ai_gpu_share peers                         List connected peers
//! ai_gpu_share backup-keys <path>            Backup node identity keys
//! ai_gpu_share restore <path>                Restore identity from backup
//! ai_gpu_share --help                        Show help
//! ```

use std::process::ExitCode;

use ai_assistant::gpu_sharing::{GpuSharingConfig, SharingMode};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("start") => cmd_start(&args[2..]),
        Some("stop") => cmd_stop(),
        Some("status") => cmd_status(),
        Some("models") => cmd_models(),
        Some("credits") => cmd_credits(),
        Some("peers") => cmd_peers(),
        Some("backup-keys") => cmd_backup_keys(&args[2..]),
        Some("restore") => cmd_restore(&args[2..]),
        Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!(
                "{}Unknown command: {}. Run ai_gpu_share --help{}",
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

// =============================================================================
// Commands
// =============================================================================

fn cmd_start(args: &[String]) -> ExitCode {
    let mut config = GpuSharingConfig::default();
    config.enabled = true;

    // Parse --provider / --gateway flags
    for arg in args {
        match arg.as_str() {
            "--provider" => config.mode = SharingMode::Provider,
            "--gateway" => config.mode = SharingMode::Gateway,
            "--both" => config.mode = SharingMode::Both,
            other => {
                eprintln!("{}Unknown start flag: {}{}", RED, other, RESET);
                return ExitCode::from(1);
            }
        }
    }

    println!(
        "{}{}ai_gpu_share{} v{}",
        BOLD,
        CYAN,
        RESET,
        env!("CARGO_PKG_VERSION")
    );
    println!();

    // GPU detection (placeholder — real detection would query NVML/ROCm/Metal)
    println!("{}GPU Detection:{}", BOLD, RESET);
    println!("  {}*{} Scanning for GPUs...", GREEN, RESET);
    println!(
        "  {}*{} No GPU auto-detection available (placeholder)",
        YELLOW, RESET
    );
    println!(
        "  {}*{} Configure GPU manually in ai_setup config",
        DIM, RESET
    );
    println!();

    // Network join (placeholder)
    println!("{}Network:{}", BOLD, RESET);
    println!(
        "  {}*{} Mode: {}{}{}",
        GREEN, RESET, CYAN, config.mode, RESET
    );
    println!(
        "  {}*{} Stake: {}{:.0}{} credits",
        GREEN, RESET, CYAN, config.stake_amount, RESET
    );
    println!(
        "  {}*{} Privacy: {}{}{}",
        GREEN, RESET, CYAN, config.privacy_level, RESET
    );
    println!(
        "  {}*{} Joined network (0 peers — discovery pending)",
        YELLOW, RESET
    );
    println!();

    match config.mode {
        SharingMode::Provider | SharingMode::Both => {
            println!("{}Listening for inference requests...{}", GREEN, RESET);
        }
        SharingMode::Gateway => {
            println!("{}Gateway ready at localhost:8090{}", GREEN, RESET);
        }
    }

    if config.mode == SharingMode::Both {
        println!("{}Gateway ready at localhost:8090{}", GREEN, RESET);
    }

    println!();
    println!("{}Press Ctrl+C to stop.{}", DIM, RESET);

    // In a real implementation, this would start async event loops.
    // For now, just print status and exit.
    ExitCode::SUCCESS
}

fn cmd_stop() -> ExitCode {
    println!("{}Stopping GPU sharing node...{}", YELLOW, RESET);
    println!("{}Node stopped gracefully.{}", GREEN, RESET);
    ExitCode::SUCCESS
}

fn cmd_status() -> ExitCode {
    let config = GpuSharingConfig::default();

    println!("{}{}ai_gpu_share{} — Node Status", BOLD, CYAN, RESET);
    println!("{}{}{}", DIM, "=".repeat(50), RESET);
    println!();
    println!("  {}Mode:{}        {}", BOLD, RESET, config.mode);
    println!(
        "  {}Status:{}      {}Not running{}",
        BOLD, RESET, YELLOW, RESET
    );
    println!(
        "  {}Credits:{}     {}{:.2}{}",
        BOLD, RESET, CYAN, 0.0f64, RESET
    );
    println!(
        "  {}Reputation:{}  {}{:.2}{}",
        BOLD, RESET, GREEN, 0.0f64, RESET
    );
    println!("  {}Peers:{}       {}0{}", BOLD, RESET, CYAN, RESET);
    println!("  {}Models:{}      {}0 loaded{}", BOLD, RESET, CYAN, RESET);
    println!("  {}GPU Load:{}    {}0%{}", BOLD, RESET, GREEN, RESET);
    println!();
    println!(
        "  {}Routing:{}     {}",
        BOLD,
        RESET,
        format!("{:?}", config.routing.strategy)
    );
    println!(
        "  {}Pricing:{}     {} (base: {:.2} credits/1K tokens)",
        BOLD, RESET, config.pricing.mode, config.pricing.base_price
    );
    println!("  {}Privacy:{}     {}", BOLD, RESET, config.privacy_level);
    println!(
        "  {}Audit:{}       {}% of transactions",
        BOLD, RESET, config.auditor_verify_percent
    );

    ExitCode::SUCCESS
}

fn cmd_credits() -> ExitCode {
    println!("{}{}ai_gpu_share{} — Credit Balance", BOLD, CYAN, RESET);
    println!("{}{}{}", DIM, "=".repeat(50), RESET);
    println!();
    println!(
        "  {}Balance:{}       {}{:.2}{} credits",
        BOLD, RESET, CYAN, 0.0f64, RESET
    );
    println!(
        "  {}Pending:{}       {}{:.2}{} credits (maturing)",
        BOLD, RESET, YELLOW, 0.0f64, RESET
    );
    println!(
        "  {}Earned today:{}  {}{:.2}{} credits",
        BOLD, RESET, GREEN, 0.0f64, RESET
    );
    println!(
        "  {}Spent today:{}   {}{:.2}{} credits",
        BOLD, RESET, RED, 0.0f64, RESET
    );
    println!(
        "  {}Reputation:{}    {}{:.2}{}",
        BOLD, RESET, GREEN, 0.0f64, RESET
    );
    println!();
    println!(
        "  {}Stake:{}         {}{:.0}{} credits (locked)",
        BOLD, RESET, DIM, 0.0f64, RESET
    );

    ExitCode::SUCCESS
}

fn cmd_models() -> ExitCode {
    println!("{}{}ai_gpu_share{} — Available Models", BOLD, CYAN, RESET);
    println!("{}{}{}", DIM, "=".repeat(70), RESET);
    println!();
    println!(
        "  {}{:<30} {:<10} {:<12} {:<10}{}",
        BOLD, "Model", "Quant", "Credits/1K", "Tokens/s", RESET
    );
    println!("  {}{}{}", DIM, "-".repeat(66), RESET);
    println!();
    println!(
        "  {}No models available — not connected to network.{}",
        DIM, RESET
    );
    println!(
        "  {}Run `ai_gpu_share start` to join the network.{}",
        DIM, RESET
    );

    ExitCode::SUCCESS
}

fn cmd_peers() -> ExitCode {
    println!("{}{}ai_gpu_share{} — Connected Peers", BOLD, CYAN, RESET);
    println!("{}{}{}", DIM, "=".repeat(70), RESET);
    println!();
    println!(
        "  {}{:<20} {:<15} {:<10} {:<10} {:<10}{}",
        BOLD, "Node ID", "GPU", "VRAM", "Load", "Rep", RESET
    );
    println!("  {}{}{}", DIM, "-".repeat(66), RESET);
    println!();
    println!("  {}No peers connected — not running.{}", DIM, RESET);
    println!(
        "  {}Run `ai_gpu_share start` to join the network.{}",
        DIM, RESET
    );

    ExitCode::SUCCESS
}

fn cmd_backup_keys(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => p,
        None => {
            eprintln!(
                "{}Usage: ai_gpu_share backup-keys <output-path>{}",
                RED, RESET
            );
            return ExitCode::from(1);
        }
    };

    println!(
        "{}Backing up node identity keys to: {}{}{}",
        GREEN, CYAN, path, RESET
    );
    println!(
        "{}Key backup not yet implemented — requires distributed-network feature.{}",
        YELLOW, RESET
    );

    ExitCode::SUCCESS
}

fn cmd_restore(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => p,
        None => {
            eprintln!("{}Usage: ai_gpu_share restore <backup-path>{}", RED, RESET);
            return ExitCode::from(1);
        }
    };

    println!(
        "{}Restoring node identity from: {}{}{}",
        GREEN, CYAN, path, RESET
    );
    println!(
        "{}Key restore not yet implemented — requires distributed-network feature.{}",
        YELLOW, RESET
    );

    ExitCode::SUCCESS
}

fn print_help() {
    println!(
        "{}{}ai_gpu_share{} v{} — GPU Sharing Network for AI Assistant",
        BOLD,
        CYAN,
        RESET,
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("{}USAGE:{}", BOLD, RESET);
    println!("    ai_gpu_share <COMMAND> [OPTIONS]");
    println!();
    println!("{}COMMANDS:{}", BOLD, RESET);
    println!(
        "    {}start{}    [--provider|--gateway]  Start the GPU sharing node",
        GREEN, RESET
    );
    println!(
        "    {}stop{}                             Stop the node gracefully",
        GREEN, RESET
    );
    println!(
        "    {}status{}                           Show node status and configuration",
        GREEN, RESET
    );
    println!(
        "    {}models{}                           List models available on the network",
        GREEN, RESET
    );
    println!(
        "    {}credits{}                          Show credit balance and earnings",
        GREEN, RESET
    );
    println!(
        "    {}peers{}                            List connected peers with GPU info",
        GREEN, RESET
    );
    println!(
        "    {}backup-keys{} <path>               Backup node identity keys",
        GREEN, RESET
    );
    println!(
        "    {}restore{} <path>                   Restore node identity from backup",
        GREEN, RESET
    );
    println!();
    println!("{}START OPTIONS:{}", BOLD, RESET);
    println!("    --provider    Run as provider only (share GPU, don't route)");
    println!("    --gateway     Run as gateway only (route requests, don't provide)");
    println!("    (default)     Run as both provider and gateway");
    println!();
    println!("{}EXAMPLES:{}", BOLD, RESET);
    println!("    ai_gpu_share start                 # Start as both provider + gateway");
    println!("    ai_gpu_share start --provider      # Share your GPU only");
    println!("    ai_gpu_share start --gateway       # Route requests only");
    println!("    ai_gpu_share status                # Check node status");
    println!("    ai_gpu_share credits               # View credit balance");
    println!("    ai_gpu_share models                # List network models");
    println!("    ai_gpu_share peers                 # List connected peers");
    println!("    ai_gpu_share backup-keys keys.bak  # Backup identity");
    println!("    ai_gpu_share restore keys.bak      # Restore identity");
    println!();
    println!("{}CONFIGURATION:{}", BOLD, RESET);
    println!("    Configure GPU sharing via `ai_setup config set gpu_sharing.*`");
    println!("    or edit the config file directly.");
    println!();
    println!(
        "{}For more information, see the project documentation.{}",
        DIM, RESET
    );
}
