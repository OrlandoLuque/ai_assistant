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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::gpu_sharing::{GpuSharingConfig, SharingMode};
use ai_assistant::node_security::CertificateManager;

const NODE_IDENTITY_DIR: &str = "./node_identity";

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
    let mut config = GpuSharingConfig {
        enabled: true,
        ..Default::default()
    };

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

    // GPU detection — subprocess checks (nvidia-smi / CUDA env / Apple sysctl).
    // Parallel impl of butler::GpuDetector to avoid requiring the `butler` feature
    // from gpu-sharing. Keep in sync with src/butler.rs:GpuDetector.
    println!("{}GPU Detection:{}", BOLD, RESET);
    println!("  {}*{} Scanning for GPUs...", GREEN, RESET);
    match detect_gpu() {
        Some(info) => {
            println!("  {}*{} {}", GREEN, RESET, info);
        }
        None => {
            println!(
                "  {}*{} No GPU detected — configure manually via ai_setup",
                YELLOW, RESET
            );
        }
    }
    println!();

    // Node identity — load-or-create so the node has a stable ID
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
    match CertificateManager::load_or_create(Path::new(NODE_IDENTITY_DIR)) {
        Ok((identity, is_new)) => {
            let tag = if is_new { "generated" } else { "loaded" };
            println!(
                "  {}*{} Node identity {} ({}{:?}{})",
                GREEN, RESET, tag, CYAN, identity.node_id, RESET
            );
            println!(
                "  {}*{} Peer discovery pending — full async bootstrap available via `ai_cluster_node`",
                YELLOW, RESET
            );
        }
        Err(e) => {
            println!("  {}*{} Identity init failed: {}", RED, RESET, e);
        }
    }
    println!();

    // What this command genuinely does is validate the configuration and
    // load-or-create the node identity above. It does NOT serve.
    //
    // It used to print "Listening for inference requests…" / "Gateway ready at
    // localhost:8090" and "Press Ctrl+C to stop", then exit(0) immediately —
    // nothing was ever bound to that port. Announcing a service that is not
    // running is worse than not implementing it: the operator has no way to
    // tell, and will find out from whatever depends on it.
    println!();
    println!(
        "{}Configuration is valid and the node identity is ready.{}",
        GREEN, RESET
    );
    println!(
        "  {}!{} This command does NOT start a server — nothing is listening on \
         {}. Run the async node with `ai_cluster_node` to actually serve.",
        YELLOW,
        RESET,
        match config.mode {
            SharingMode::Provider => "the provider endpoint",
            _ => "localhost:8090",
        }
    );
    ExitCode::SUCCESS
}

fn cmd_stop() -> ExitCode {
    // Likewise: this used to report "Node stopped gracefully" unconditionally,
    // having stopped nothing.
    println!(
        "{}Nothing to stop — `ai_gpu_share start` does not run a server.{}",
        YELLOW, RESET
    );
    println!(
        "  {}!{} If you started a node with `ai_cluster_node`, stop that process instead.",
        DIM, RESET
    );
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
        "  {}Routing:{}     {:?}",
        BOLD, RESET, config.routing.strategy
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
                "{}Usage: ai_gpu_share backup-keys <output-dir>{}",
                RED, RESET
            );
            return ExitCode::from(1);
        }
    };

    let src = PathBuf::from(NODE_IDENTITY_DIR);
    if !src.exists() {
        eprintln!(
            "{}No node identity to back up. Run `ai_gpu_share start` first to generate one.{}",
            RED, RESET
        );
        return ExitCode::from(1);
    }

    let identity = match CertificateManager::load_identity(&src) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("{}Failed to load identity: {}{}", RED, e, RESET);
            return ExitCode::from(1);
        }
    };

    let dst = PathBuf::from(path);
    println!(
        "{}Backing up node identity ({}{:?}{}) to: {}{}{}",
        GREEN,
        CYAN,
        identity.node_id,
        RESET,
        CYAN,
        dst.display(),
        RESET
    );

    if let Err(e) = CertificateManager::save_identity(&identity, &dst) {
        eprintln!("{}Failed to save backup: {}{}", RED, e, RESET);
        return ExitCode::from(1);
    }

    println!(
        "{}*{} Backup written: cert.der, key.der, ca.der",
        GREEN, RESET
    );
    println!(
        "{}WARNING: the private key is stored in plain DER. Protect the backup directory.{}",
        YELLOW, RESET
    );
    ExitCode::SUCCESS
}

fn cmd_restore(args: &[String]) -> ExitCode {
    let path = match args.first() {
        Some(p) => p,
        None => {
            eprintln!("{}Usage: ai_gpu_share restore <backup-dir>{}", RED, RESET);
            return ExitCode::from(1);
        }
    };

    let src = PathBuf::from(path);
    if !src.exists() {
        eprintln!(
            "{}Backup directory not found: {}{}",
            RED,
            src.display(),
            RESET
        );
        return ExitCode::from(1);
    }

    let identity = match CertificateManager::load_identity(&src) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("{}Failed to load backup: {}{}", RED, e, RESET);
            return ExitCode::from(1);
        }
    };

    let dst = PathBuf::from(NODE_IDENTITY_DIR);
    println!(
        "{}Restoring node identity ({}{:?}{}) to: {}{}{}",
        GREEN,
        CYAN,
        identity.node_id,
        RESET,
        CYAN,
        dst.display(),
        RESET
    );

    if let Err(e) = CertificateManager::save_identity(&identity, &dst) {
        eprintln!("{}Failed to write identity: {}{}", RED, e, RESET);
        return ExitCode::from(1);
    }

    println!("{}*{} Identity restored.", GREEN, RESET);
    ExitCode::SUCCESS
}

fn detect_gpu() -> Option<String> {
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let gpus: Vec<&str> = stdout.trim().lines().collect();
                if !gpus.is_empty() {
                    return Some(format!(
                        "NVIDIA: {} GPU(s), first: {}",
                        gpus.len(),
                        gpus[0].trim()
                    ));
                }
            }
        }
    }

    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || std::env::var("CUDA_HOME").is_ok() {
        return Some("CUDA env vars present (no nvidia-smi output)".into());
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                let cpu = String::from_utf8_lossy(&output.stdout).to_string();
                if cpu.contains("Apple") {
                    return Some(format!("Apple Silicon: {}", cpu.trim()));
                }
            }
        }
    }

    None
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
