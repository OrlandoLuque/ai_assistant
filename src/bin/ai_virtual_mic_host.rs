//! ai_virtual_mic_host — slot-assignment server for Group Queue protocol.
//!
//! Usage:
//! ```text
//! ai_virtual_mic_host --port 9876 --slots 8 --preset flat
//! ai_virtual_mic_host --preset squad --callouts 3 --slots 8
//! ai_virtual_mic_host --preset meeting
//! ```
//!
//! Clients connect with `ai_virtual_mic` (Group Queue tab → Connect to host).
//! The host is the authoritative source of slot/priority assignments. It
//! requires only TCP reachability (LAN or port-forwarded).

use ai_assistant::group_queue_host::{GroupQueueHost, HostConfig, HostPreset};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut port: u16 = 9876;
    let mut slots: u8 = 8;
    let mut preset = HostPreset::Flat;
    let mut bind_host = "0.0.0.0".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(9876);
            }
            "--bind" => {
                i += 1;
                bind_host = args.get(i).cloned().unwrap_or_else(|| "0.0.0.0".into());
            }
            "--slots" => {
                i += 1;
                slots = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8).min(8);
            }
            "--preset" => {
                i += 1;
                let p = args.get(i).cloned().unwrap_or_else(|| "flat".into());
                preset = match p.as_str() {
                    "flat" => HostPreset::Flat,
                    "squad" => HostPreset::Squad { callouts: 3 },
                    "meeting" => HostPreset::Meeting,
                    _ => {
                        eprintln!("Unknown preset '{}'. Valid: flat, squad, meeting", p);
                        std::process::exit(1);
                    }
                };
            }
            "--callouts" => {
                i += 1;
                let c = args.get(i).and_then(|s| s.parse::<u8>().ok()).unwrap_or(3);
                if let HostPreset::Squad { ref mut callouts } = preset {
                    *callouts = c;
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown arg: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let bind_addr: SocketAddr = match format!("{}:{}", bind_host, port).parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Invalid bind address '{}:{}': {}", bind_host, port, e);
            std::process::exit(1);
        }
    };

    let config = HostConfig {
        bind_addr,
        slot_count: slots,
        preset,
        ..HostConfig::default()
    };
    let host = Arc::new(GroupQueueHost::new(config));

    // Ctrl+C handler
    let host_sig = host.clone();
    ctrlc_handler(move || {
        eprintln!("\n[host] shutdown signal received");
        host_sig.shutdown();
    });

    println!("ai_virtual_mic_host ready — press Ctrl+C to stop");
    if let Err(e) = host.run() {
        eprintln!("[host] fatal: {}", e);
        std::process::exit(1);
    }
    println!("[host] stopped");
}

fn print_help() {
    println!("ai_virtual_mic_host — Group Queue slot-assignment server");
    println!();
    println!("Usage:");
    println!("  ai_virtual_mic_host [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --port <N>          TCP port to listen on (default: 9876)");
    println!("  --bind <HOST>       Interface to bind (default: 0.0.0.0)");
    println!("  --slots <N>         Max concurrent slots (default: 8, max: 8)");
    println!("  --preset <NAME>     Priority preset: flat | squad | meeting (default: flat)");
    println!("  --callouts <N>      For squad preset: number of callout slots (default: 3)");
    println!("  -h, --help          Print this help");
}

/// Simple cross-platform Ctrl+C handler using a polling thread.
/// (Avoids adding the `ctrlc` crate just for this binary.)
fn ctrlc_handler<F: Fn() + Send + 'static>(f: F) {
    #[cfg(windows)]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static FIRED: AtomicBool = AtomicBool::new(false);
        unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
            FIRED.store(true, Ordering::Relaxed);
            1 // TRUE — handled
        }
        // Win32 API type names kept verbatim for FFI readability.
        #[allow(clippy::upper_case_acronyms)]
        type BOOL = i32;
        #[allow(clippy::upper_case_acronyms)]
        type DWORD = u32;
        extern "system" {
            fn SetConsoleCtrlHandler(
                handler: Option<unsafe extern "system" fn(DWORD) -> BOOL>,
                add: BOOL,
            ) -> BOOL;
        }
        unsafe {
            SetConsoleCtrlHandler(Some(handler), 1);
        }
        std::thread::spawn(move || loop {
            if FIRED.load(Ordering::Relaxed) {
                f();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        });
    }
    #[cfg(not(windows))]
    {
        // Unix: use a signal-polling thread (minimal, avoids nix/signal-hook deps)
        std::thread::spawn(move || {
            // No-op — host will block on accept. Users can SIGKILL or close terminal.
            // Full SIGINT handling would need `signal-hook` crate; skipped here.
            let _ = f;
        });
    }
}
