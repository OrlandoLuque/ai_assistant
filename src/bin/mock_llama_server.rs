// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Tiny stand-in for `llama-server` used by integration tests of
//! `crate::embedded_server`. It binds to the `--host` / `--port` it is
//! given on the command line and answers a hard-coded set of routes:
//!
//! * `GET /health` → `200 OK` after an optional `MOCK_LLAMA_DELAY_MS`
//!   ramp-up so tests can exercise the wait-loop.
//! * `GET /v1/models` → `200 OK` with a single dummy model entry.
//! * Anything else → `404 Not Found`.
//!
//! Reads only the args it cares about (`--host`, `--port`,
//! `--mmproj` for argv echo, plus arbitrary tail). Everything else is
//! discarded so test fixtures can pass realistic flags.
//!
//! Not part of the public surface — only built when running tests:
//! `Cargo.toml` declares it as `[[bin]] required-features = ["vision"]`.

#![cfg(feature = "vision")]

use std::env;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

fn parse_args() -> (String, u16) {
    let args: Vec<String> = env::args().collect();
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" if i + 1 < args.len() => {
                host = args[i + 1].clone();
                i += 2;
            }
            "--port" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            _ => i += 1,
        }
    }
    (host, port)
}

fn main() {
    let (host, port) = parse_args();
    let listener = TcpListener::bind((host.as_str(), port))
        .unwrap_or_else(|e| panic!("mock_llama_server: bind {}:{} failed: {}", host, port, e));

    // Optional ramp-up: simulate a slow boot so tests can exercise
    // wait_until_ready more realistically. Defaults to 0 ms.
    let ramp_ms: u64 = env::var("MOCK_LLAMA_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ready_at = Instant::now() + Duration::from_millis(ramp_ms);

    // Marker line for the parent test to detect "started" without
    // depending on /health timing.
    println!(
        "mock_llama_server listening on {}",
        listener.local_addr().expect("local_addr")
    );

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Tiny request parser: read until \r\n\r\n or 1 KiB.
        let mut buf = [0u8; 1024];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let req = String::from_utf8_lossy(&buf[..n]);
        let first_line = req.lines().next().unwrap_or("");
        let mut parts = first_line.split_whitespace();
        let _method = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("");

        let (status, body) = match path {
            "/health" => {
                if Instant::now() < ready_at {
                    (
                        "503 Service Unavailable",
                        "{\"status\":\"warming-up\"}".to_string(),
                    )
                } else {
                    ("200 OK", "{\"status\":\"ok\"}".to_string())
                }
            }
            "/v1/models" => (
                "200 OK",
                "{\"object\":\"list\",\"data\":[{\"id\":\"mock\",\"object\":\"model\"}]}"
                    .to_string(),
            ),
            "/__shutdown" => {
                let _ = write_response(&mut stream, "200 OK", "{\"bye\":true}");
                std::process::exit(0);
            }
            _ => ("404 Not Found", "{\"error\":\"not-found\"}".to_string()),
        };
        let _ = write_response(&mut stream, status, &body);
    }
}

fn write_response(
    stream: &mut std::net::TcpStream,
    status: &str,
    body: &str,
) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}
