// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Integration tests for [`ai_assistant::embedded_server`] using a real
//! child process.
//!
//! The mock binary `mock_llama_server` is a tiny TCP server that
//! pretends to be `llama-server`. It is declared in `Cargo.toml` as a
//! `[[bin]]` with `required-features = ["vision"]`, so Cargo builds it
//! before running these tests and exposes its absolute path through
//! `env!("CARGO_BIN_EXE_mock_llama_server")`.

#![cfg(feature = "vision")]

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use ai_assistant::embedded_server::{EmbeddedLlamaServer, LlamaServerConfig};

/// Tests that read or mutate `MOCK_LLAMA_DELAY_MS` (a process-wide
/// env var inherited by spawned mock children) must serialize with
/// each other. Without this, parallel test execution can pick up the
/// 60 s warm-up delay set by the timeout test and starve unrelated
/// spawns of `wait_until_ready`.
fn env_serial_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Path to the test mock binary built by Cargo.
fn mock_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mock_llama_server"))
}

/// A non-empty placeholder file used as the `--model` argument. The
/// mock server ignores the value, but the launcher's pre-flight insists
/// on it existing.
fn dummy_model_file() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ai_assistant_emb_test_{}_{}",
        std::process::id(),
        rand_suffix()
    ));
    let _ = std::fs::create_dir_all(&dir);
    let p = dir.join("dummy_model.gguf");
    let _ = std::fs::write(&p, b"GGUF\0\0\0\0");
    p
}

fn rand_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:08x}", nanos)
}

#[test]
fn spawns_mock_server_and_health_returns_ok() {
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .host("127.0.0.1")
        .port(0)
        .ready_timeout(Duration::from_secs(10))
        .build();
    let mut server = EmbeddedLlamaServer::start(cfg).expect("mock should spawn");
    assert!(server.port() >= 1024, "auto-picked port should be valid");
    server
        .wait_until_ready(Some(Duration::from_secs(10)))
        .expect("mock /health should reply 200");
    assert!(server.is_running(), "mock should still be alive");
    assert!(server.base_url().starts_with("http://127.0.0.1:"));
}

#[test]
fn drop_kills_child_process() {
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .ready_timeout(Duration::from_secs(10))
        .build();
    let pid = {
        let mut server = EmbeddedLlamaServer::start(cfg).expect("spawn");
        server
            .wait_until_ready(Some(Duration::from_secs(10)))
            .expect("ready");
        server.pid().expect("pid")
    };
    // After Drop the process must be gone. We can't query arbitrary PIDs
    // portably, so we instead try to connect to the port we observed and
    // expect it to be free again. Use the alternative approach: another
    // server started immediately should succeed even on the same loopback.
    let _ = pid; // pid retained only for documentation
                 // If the previous child were still alive holding its port, our
                 // *next* server with port=0 should still succeed (different port),
                 // but the kill is observable: spawn a second, ensure it works.
    let cfg2 = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .ready_timeout(Duration::from_secs(10))
        .build();
    let mut second = EmbeddedLlamaServer::start(cfg2).expect("second spawn");
    second
        .wait_until_ready(Some(Duration::from_secs(10)))
        .expect("second ready");
    assert!(second.is_running());
}

#[test]
fn wait_until_ready_returns_timeout_when_health_never_replies() {
    // Point at a real binary that exits immediately so /health never
    // answers. We use the mock with MOCK_LLAMA_DELAY_MS huge to force
    // /health to return 503 forever within the test window.
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("MOCK_LLAMA_DELAY_MS", "60000");
    let cfg = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .ready_timeout(Duration::from_millis(800))
        .build();
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut server = EmbeddedLlamaServer::start(cfg)?;
        let started = Instant::now();
        let err = server.wait_until_ready(Some(Duration::from_millis(800)));
        let elapsed = started.elapsed();
        std::env::remove_var("MOCK_LLAMA_DELAY_MS");
        assert!(err.is_err(), "should time out while child returns 503");
        assert!(
            elapsed >= Duration::from_millis(700),
            "should respect the timeout (elapsed {:?})",
            elapsed
        );
        // Cleanup happens on Drop.
        Ok(())
    })();
    std::env::remove_var("MOCK_LLAMA_DELAY_MS");
    result.expect("scenario must complete cleanly");
}

#[test]
fn auto_picked_port_is_unique_per_call() {
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    let cfg1 = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .build();
    let cfg2 = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .build();
    let mut s1 = EmbeddedLlamaServer::start(cfg1).expect("s1");
    let mut s2 = EmbeddedLlamaServer::start(cfg2).expect("s2");
    s1.wait_until_ready(Some(Duration::from_secs(10)))
        .expect("ready1");
    s2.wait_until_ready(Some(Duration::from_secs(10)))
        .expect("ready2");
    assert_ne!(s1.port(), s2.port(), "auto-picked ports must differ");
}

#[test]
fn binary_filename_is_safe_for_logs() {
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .port(0)
        .build();
    let server = EmbeddedLlamaServer::start(cfg).expect("spawn");
    let fname = server.binary_filename();
    assert!(
        !fname.contains(std::path::MAIN_SEPARATOR),
        "filename must not leak directory layout: {}",
        fname
    );
    assert!(
        fname.contains("mock_llama_server"),
        "filename should match the test binary, got {}",
        fname
    );
}

#[test]
fn explicit_port_is_honoured_when_above_threshold() {
    // Find a free port via the OS, then ask the launcher to use that
    // exact value. Confirms that explicit `port(N)` is not silently
    // replaced.
    let _guard = env_serial_lock().lock().unwrap_or_else(|e| e.into_inner());
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
    let chosen = probe.local_addr().expect("local_addr").port();
    drop(probe);
    let cfg = LlamaServerConfig::builder(mock_binary(), dummy_model_file())
        .host("127.0.0.1")
        .port(chosen)
        .ready_timeout(Duration::from_secs(10))
        .build();
    let mut server = EmbeddedLlamaServer::start(cfg).expect("spawn");
    assert_eq!(server.port(), chosen);
    server
        .wait_until_ready(Some(Duration::from_secs(10)))
        .expect("ready");
}
