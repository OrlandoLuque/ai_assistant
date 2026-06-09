// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Tiny stand-in for `llama-server` used by integration tests of
//! `crate::embedded_server`. It binds to the `--host` / `--port` it is
//! given on the command line and answers a hard-coded set of routes:
//!
//! * `GET /health` → `200 OK` after an optional `MOCK_LLAMA_DELAY_MS`
//!   ramp-up so tests can exercise the wait-loop.
//! * `GET /v1/models` → `200 OK` with a single dummy model entry.
//! * `GET /sse-test?chunks=N&gap_ms=M&stall_after=K` → emits an
//!   `text/event-stream` body with `N` chunks separated by `M` ms.
//!   If `stall_after=K` is set, the stream emits `K` chunks then hangs
//!   forever (drives V150 per-chunk timeout tests).
//! * `POST /v1/chat/completions` → `text/event-stream` SSE when the
//!   body contains `"stream":true` (env knobs reused from `/sse-test`),
//!   else a single JSON chat response.
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
        // Tiny request parser: read until \r\n\r\n or 4 KiB.
        let mut buf = [0u8; 4096];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let req = String::from_utf8_lossy(&buf[..n]);
        let first_line = req.lines().next().unwrap_or("");
        let mut parts = first_line.split_whitespace();
        let method = parts.next().unwrap_or("");
        let raw_path = parts.next().unwrap_or("");
        let (path, query) = match raw_path.find('?') {
            Some(i) => (&raw_path[..i], &raw_path[i + 1..]),
            None => (raw_path, ""),
        };

        // Streaming endpoints handled inline (they write bytes directly).
        if path == "/sse-test" {
            stream_sse(&mut stream, query);
            continue;
        }
        if path == "/v1/chat/completions" && method == "POST" {
            let stream_requested =
                req.contains("\"stream\":true") || req.contains("\"stream\": true");
            if stream_requested {
                stream_sse(&mut stream, query);
                continue;
            }
            let body = "{\"id\":\"mock\",\"object\":\"chat.completion\",\"created\":0,\
                \"model\":\"mock\",\"choices\":[{\"index\":0,\"message\":\
                {\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}";
            let _ = write_response(&mut stream, "200 OK", body);
            continue;
        }

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

fn parse_query_u64(query: &str, key: &str) -> Option<u64> {
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        if k == key {
            return v.parse().ok();
        }
    }
    None
}

fn env_or(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Emit an SSE response on `stream`. Knobs (query > env > default):
///
/// * `chunks` / `MOCK_SSE_CHUNKS` (default 3)
/// * `gap_ms` / `MOCK_SSE_GAP_MS` (default 50)
/// * `stall_after` / `MOCK_SSE_STALL_AFTER` — if set, emit that many
///   chunks then sleep forever, without `[DONE]` or close.
///
/// HTTP framing: `Connection: close`, no `Content-Length`. The body is
/// terminated by EOF (TCP close), which `reqwest::Response::bytes_stream`
/// handles naturally.
fn stream_sse(stream: &mut std::net::TcpStream, query: &str) {
    let chunks = parse_query_u64(query, "chunks").unwrap_or_else(|| env_or("MOCK_SSE_CHUNKS", 3));
    let gap_ms = parse_query_u64(query, "gap_ms").unwrap_or_else(|| env_or("MOCK_SSE_GAP_MS", 50));
    let stall_after = parse_query_u64(query, "stall_after").or_else(|| {
        env::var("MOCK_SSE_STALL_AFTER")
            .ok()
            .and_then(|v| v.parse().ok())
    });

    let header = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: close\r\n\
        \r\n";
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    let _ = stream.flush();

    for i in 0..chunks {
        if let Some(k) = stall_after {
            if i >= k {
                // Hang forever (test should drive the proxy's per-chunk
                // timeout). 1h is effectively "forever" for any test.
                std::thread::sleep(Duration::from_secs(3600));
                return;
            }
        }
        let line = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"chunk-{}\"}},\"index\":0}}]}}\n\n",
            i
        );
        if stream.write_all(line.as_bytes()).is_err() {
            return;
        }
        let _ = stream.flush();
        if gap_ms > 0 {
            std::thread::sleep(Duration::from_millis(gap_ms));
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
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
