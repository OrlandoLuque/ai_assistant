//! `ai_mcp_server` — an MCP server over stdio, speaking JSON-RPC 2.0.
//!
//! This is what `examples/mcp_server.rs` was not. That example builds a server, registers
//! a made-up weather tool, hands it two canned requests and prints the answers. It
//! demonstrates the API; it never reads a byte from stdin, so nothing can connect to it.
//! This binary is the transport: it reads newline-delimited JSON-RPC from stdin and
//! writes replies to stdout, which is how Claude Desktop, Claude Code and the rest of the
//! MCP ecosystem launch a local server.
//!
//! # stdout belongs to the protocol
//!
//! On a stdio transport, **anything written to stdout that is not a JSON-RPC frame
//! corrupts the session**. The client is parsing that stream. A stray `println!`, a
//! progress bar, a warning from a dependency — any of them desynchronises the peer, and
//! the failure surfaces as "the server disconnected", nowhere near its cause.
//!
//! So: every human-facing byte this binary emits goes to **stderr**. That is a constraint
//! on the whole call tree, not just on this file, which is why the tool sets registered
//! below are chosen from the ones that do not print. Document parsing is deliberately
//! absent: `pdf-extract` writes "Unicode mismatch" straight to stdout (issue N52), and
//! under this transport that is not cosmetic noise, it is a protocol violation.
//!
//! # Usage
//!
//! ```text
//! ai_mcp_server                     # serve on stdio (what a client does)
//! ai_mcp_server --list-tools        # print the registered tools and exit
//! ai_mcp_server --db tasks.sqlite   # task store location
//! ai_mcp_server --allow-config-writes
//! ```
//!
//! Registering it with a client:
//!
//! ```json
//! { "mcpServers": { "ai_assistant": { "command": "ai_mcp_server", "args": [] } } }
//! ```

use ai_assistant::mcp_protocol::McpServer;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const NAME: &str = "ai_assistant";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Options {
    list_tools: bool,
    db_path: PathBuf,
    allow_config_writes: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            list_tools: false,
            db_path: PathBuf::from("ai_assistant_tasks.sqlite"),
            allow_config_writes: false,
        }
    }
}

fn print_help() {
    // stderr, like everything else here: --help on stdout would be the first thing to
    // corrupt the stream if a client ever passed the flag through.
    eprintln!(
        "\
ai_mcp_server {VERSION} — MCP server over stdio (JSON-RPC 2.0)

USAGE:
    ai_mcp_server [OPTIONS]

OPTIONS:
    --list-tools             Print the registered tool names and exit
    --db <PATH>              Task store database (default: ai_assistant_tasks.sqlite)
    --allow-config-writes    Let the config tools modify configuration (default: read-only)
    -h, --help               Show this help

With no options it serves on stdio: newline-delimited JSON-RPC on stdin, replies on
stdout. Diagnostics always go to stderr, because stdout belongs to the protocol."
    );
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut opts = Options::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list-tools" => opts.list_tools = true,
            "--allow-config-writes" => opts.allow_config_writes = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--db" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--db needs a path".to_string())?
                    .clone();
                opts.db_path = PathBuf::from(value);
            }
            other => {
                // An unknown flag is an error, not something to fold into a positional
                // argument. The CLI-drift audit (N-documented_cli) found exactly that
                // failure mode elsewhere: the command "works" and answers a different
                // question than the one asked.
                return Err(format!("unknown argument: {other}"));
            }
        }
        i += 1;
    }
    Ok(opts)
}

/// Build the server with every tool set this build can offer.
///
/// Returns the server and the list of tool sets that were *not* registered, each with the
/// reason. Saying "the research tools are absent because this build lacks `research`" is
/// worth more than silently serving a shorter list: a client that cannot find a tool
/// otherwise has no way to tell a missing feature from a bug.
fn build_server(opts: &Options) -> (McpServer, Vec<String>) {
    let mut server = McpServer::new(NAME, VERSION);
    let mut absent: Vec<String> = Vec::new();

    // --- Configuration tools (always available under `tools`) ---
    {
        let config = ai_assistant::config_file::default_config_path();
        let loaded = if config.exists() {
            ai_assistant::config_file::load_config(&config).unwrap_or_default()
        } else {
            Default::default()
        };
        ai_assistant::register_config_tools(
            &mut server,
            Arc::new(Mutex::new(loaded)),
            opts.allow_config_writes,
        );
        if !opts.allow_config_writes {
            absent.push(
                "config writes are disabled (pass --allow-config-writes to enable)".to_string(),
            );
        }
    }

    // --- Task tools ---
    match ai_assistant::mcp_task_tools::UserTaskStore::open(&opts.db_path) {
        Ok(store) => {
            ai_assistant::mcp_task_tools::register_task_tools(
                &mut server,
                Arc::new(Mutex::new(store)),
            );
        }
        Err(e) => {
            // Do not abort: a broken task store is a reason to serve fewer tools, not a
            // reason to serve none. Reported so it is not mistaken for a shorter build.
            absent.push(format!(
                "task tools: could not open {} ({e})",
                opts.db_path.display()
            ));
        }
    }

    // --- Benchmark tools ---
    #[cfg(feature = "eval")]
    ai_assistant::mcp_protocol::register_benchmark_tools(&mut server);
    #[cfg(not(feature = "eval"))]
    absent.push("benchmark tools: built without the `eval` feature".to_string());

    // --- Knowledge tools ---
    #[cfg(feature = "rag")]
    {
        let rag_db = PathBuf::from("ai_assistant_rag.db");
        ai_assistant::mcp_protocol::knowledge_tools::register_knowledge_tools(
            &mut server,
            rag_db,
            None,
        );
    }
    #[cfg(not(feature = "rag"))]
    absent.push("knowledge tools: built without the `rag` feature".to_string());

    (server, absent)
}

/// Serve JSON-RPC over the given reader/writer pair.
///
/// Split out from `main` so it can be driven by a test over in-memory buffers: a stdio
/// loop that can only be exercised by launching the process is a stdio loop whose framing
/// bugs are found by users.
fn serve<R: BufRead, W: Write>(server: &McpServer, input: R, mut output: W) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // `handle_stream_message` returns None for notifications, which JSON-RPC says
        // must not be answered. Writing a frame there would put bytes on the wire the
        // client never asked for.
        if let Some(reply) = server.handle_stream_message(&line) {
            writeln!(output, "{reply}")?;
            // Flush per message. A client is blocked waiting for this reply; leaving it
            // in the buffer until the next one turns a working server into a hung one.
            output.flush()?;
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ai_mcp_server: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    let (server, absent) = build_server(&opts);

    if opts.list_tools {
        // The one place stdout is not the protocol: --list-tools is not a server run.
        let request = ai_assistant::McpRequest::new("tools/list");
        let response = server.handle_request(request);
        match serde_json::to_string_pretty(&response) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("ai_mcp_server: could not render the tool list: {e}");
                std::process::exit(1);
            }
        }
        for note in &absent {
            eprintln!("note: {note}");
        }
        return;
    }

    for note in &absent {
        eprintln!("ai_mcp_server: {note}");
    }
    eprintln!("ai_mcp_server {VERSION} serving on stdio");

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    if let Err(e) = serve(&server, stdin.lock(), stdout.lock()) {
        eprintln!("ai_mcp_server: transport error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare server. Deliberately registers nothing: these tests are about the
    /// **framing** — one line in, zero or one lines out — which is a property of the
    /// transport, not of which tool sets a given feature combination happens to provide.
    /// Depending on `eval` here would make the transport tests silently vanish from the
    /// builds that do not enable it.
    fn test_server() -> McpServer {
        McpServer::new(NAME, VERSION)
    }

    #[test]
    fn a_request_gets_exactly_one_line_back() {
        let server = test_server();
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n";
        let mut out: Vec<u8> = Vec::new();
        serve(&server, input.as_bytes(), &mut out).expect("serve");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.lines().count(), 1, "got: {text}");
        assert!(text.contains("\"id\":1"), "got: {text}");
    }

    #[test]
    fn a_notification_produces_no_output_at_all() {
        // The reason this binary exists rather than a loop around `handle_message`:
        // every MCP client sends `notifications/initialized` right after `initialize`,
        // and a reply to it is a frame the client is not reading for.
        let server = test_server();
        let input = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
        let mut out: Vec<u8> = Vec::new();
        serve(&server, input.as_bytes(), &mut out).expect("serve");
        assert!(out.is_empty(), "got: {}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn blank_lines_between_frames_are_skipped_not_answered() {
        let server = test_server();
        let input = "\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n\n  \n";
        let mut out: Vec<u8> = Vec::new();
        serve(&server, input.as_bytes(), &mut out).expect("serve");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.lines().count(), 1, "got: {text}");
    }

    #[test]
    fn a_malformed_frame_is_answered_and_the_session_continues() {
        // A parse error must not kill the loop. A client that sends one bad frame and
        // then a good one should get two replies, not a dead server.
        let server = test_server();
        let input = "{ not json\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n";
        let mut out: Vec<u8> = Vec::new();
        serve(&server, input.as_bytes(), &mut out).expect("serve");
        let text = String::from_utf8(out).expect("utf8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "got: {text}");
        assert!(lines[0].contains("-32700"), "got: {}", lines[0]);
        assert!(lines[1].contains("\"id\":2"), "got: {}", lines[1]);
    }

    #[test]
    fn every_reply_is_one_line_of_valid_json() {
        // The framing contract: one JSON value per line. A reply containing a raw newline
        // would split into two frames and desynchronise the peer.
        let server = test_server();
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n\
                     {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n";
        let mut out: Vec<u8> = Vec::new();
        serve(&server, input.as_bytes(), &mut out).expect("serve");
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.lines().count(), 2, "got: {text}");
        for line in text.lines() {
            serde_json::from_str::<serde_json::Value>(line)
                .unwrap_or_else(|e| panic!("not a JSON frame ({e}): {line}"));
        }
    }

    #[test]
    fn an_unknown_flag_is_rejected_rather_than_absorbed() {
        assert!(parse_args(&["--nope".to_string()]).is_err());
        assert!(
            parse_args(&["--db".to_string()]).is_err(),
            "--db needs a value"
        );
        let ok = parse_args(&["--db".to_string(), "x.sqlite".to_string()]).expect("valid");
        assert_eq!(ok.db_path, PathBuf::from("x.sqlite"));
    }

    #[test]
    fn config_writes_are_off_unless_asked_for() {
        assert!(!Options::default().allow_config_writes);
        let opts = parse_args(&["--allow-config-writes".to_string()]).expect("valid");
        assert!(opts.allow_config_writes);
    }
}
