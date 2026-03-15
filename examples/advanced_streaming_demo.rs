//! Example: advanced_streaming_demo -- Demonstrates advanced streaming capabilities.
//!
//! Run with: cargo run --example advanced_streaming_demo --features advanced-streaming
//!
//! This example showcases SSE events, WebSocket frames, streaming compression,
//! and resumable streaming with checkpoints.

use ai_assistant::{
    // SSE
    SseEvent, SseWriter,
    // WebSocket
    WsFrame, WsOpcode, WsCloseCode, WsAiMessage, WsUsage, WsStreamHandler, WsState,
    // Compression
    StreamCompressor, StreamCompressionConfig, StreamCompressionAlgorithm,
    // Resumable streaming
    ResumableStream, ResumableStreamConfig, format_sse_event,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Advanced Streaming Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. SSE Events
    // ------------------------------------------------------------------
    println!("--- 1. Server-Sent Events (SSE) ---\n");

    let event = SseEvent::new("Hello from the AI assistant!")
        .with_event("message")
        .with_id("evt-001");

    println!("  Event type:  {:?}", "message");
    println!("  Event ID:    evt-001");
    println!("  Wire format:");
    for line in event.to_wire_format().lines() {
        println!("    {}", line);
    }

    // SSE Writer
    let mut writer = SseWriter::new();
    let wire1 = writer.write_event(
        SseEvent::new("First chunk of response")
            .with_event("delta")
            .with_id("1"),
    );
    let wire2 = writer.write_event(
        SseEvent::new("Second chunk of response")
            .with_event("delta")
            .with_id("2"),
    );
    let wire3 = writer.write_data("[DONE]");
    let keepalive = writer.write_keepalive();
    let comment = writer.write_comment("heartbeat");

    println!("\n  Writer output ({} events):", 3);
    println!("    chunk 1: {} bytes", wire1.len());
    println!("    chunk 2: {} bytes", wire2.len());
    println!("    done:    {} bytes", wire3.len());
    println!("    keepalive: {:?}", keepalive.trim());
    println!("    comment:   {:?}", comment.trim());

    // Parse back
    if let Some(parsed) = SseEvent::from_wire_format(&wire1) {
        println!("\n  Parsed back: data={:?}", parsed.data);
    }

    // ------------------------------------------------------------------
    // 2. WebSocket Frames (RFC 6455)
    // ------------------------------------------------------------------
    println!("\n--- 2. WebSocket Frames (RFC 6455) ---\n");

    let text_frame = WsFrame::text("Hello, WebSocket!");
    let binary_frame = WsFrame::binary(&[0x01, 0x02, 0x03, 0x04]);
    let ping_frame = WsFrame::ping(b"keepalive");
    let pong_frame = WsFrame::pong(b"keepalive");
    let close_frame = WsFrame::close(1000, "Normal closure");

    println!("  Text frame:   opcode={:?}, payload={:?}",
        WsOpcode::Text, text_frame.as_text().unwrap_or(""));
    println!("  Binary frame: {} bytes encoded", binary_frame.encode().len());
    println!("  Ping frame:   {} bytes encoded", ping_frame.encode().len());
    println!("  Pong frame:   {} bytes encoded", pong_frame.encode().len());
    println!("  Close frame:  code={}, {} bytes encoded",
        1000, close_frame.encode().len());

    // Close codes
    println!("\n  Close codes:");
    let codes = [
        (WsCloseCode::Normal, "Normal"),
        (WsCloseCode::GoingAway, "GoingAway"),
        (WsCloseCode::ProtocolError, "ProtocolError"),
    ];
    for (code, name) in &codes {
        println!("    {}: {:?}", name, code);
    }

    // ------------------------------------------------------------------
    // 3. WebSocket AI Messages
    // ------------------------------------------------------------------
    println!("\n--- 3. WebSocket AI Protocol Messages ---\n");

    let messages: Vec<WsAiMessage> = vec![
        WsAiMessage::StreamStart {
            id: "msg-001".to_string(),
            model: "llama3-8b".to_string(),
        },
        WsAiMessage::StreamChunk {
            id: "msg-001".to_string(),
            content: "Hello, ".to_string(),
            index: Some(0),
        },
        WsAiMessage::StreamChunk {
            id: "msg-001".to_string(),
            content: "world!".to_string(),
            index: Some(1),
        },
        WsAiMessage::StreamEnd {
            id: "msg-001".to_string(),
            finish_reason: Some("stop".to_string()),
            usage: Some(WsUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        },
    ];

    for msg in &messages {
        let json = serde_json::to_string(msg).unwrap_or_default();
        println!("  {}", json);
    }

    // WsStreamHandler
    let _handler = WsStreamHandler::new();
    println!("\n  Handler state: {:?}", WsState::Connecting);

    // ------------------------------------------------------------------
    // 4. Streaming Compression
    // ------------------------------------------------------------------
    println!("\n--- 4. Streaming Compression ---\n");

    let configs = vec![
        ("fast", StreamCompressionConfig::fast()),
        ("max", StreamCompressionConfig::max_compression()),
        ("none", StreamCompressionConfig::none()),
    ];

    let payload = "This is a test payload for compression benchmarking. ".repeat(20);
    println!("  Original payload: {} bytes\n", payload.len());

    for (name, config) in configs {
        let compressor = StreamCompressor::new(config);
        let result = compressor.compress(payload.as_bytes());
        println!("  Config '{}': {} -> {} bytes (ratio: {:.1}%, saved: {} bytes, beneficial: {})",
            name,
            result.original_size,
            result.compressed_size,
            result.ratio * 100.0,
            result.bytes_saved(),
            result.is_beneficial(),
        );
    }

    // Algorithm content-encoding headers
    println!("\n  Content-Encoding headers:");
    for alg in &[
        StreamCompressionAlgorithm::Gzip,
        StreamCompressionAlgorithm::Deflate,
        StreamCompressionAlgorithm::None,
    ] {
        println!("    {:?} -> \"{}\"", alg, alg.content_encoding());
    }

    // Parse back from header
    let parsed = StreamCompressionAlgorithm::from_content_encoding("gzip");
    println!("    from_content_encoding(\"gzip\") -> {:?}", parsed);

    // ------------------------------------------------------------------
    // 5. Resumable Streaming
    // ------------------------------------------------------------------
    println!("\n--- 5. Resumable Streaming ---\n");

    let mut config = ResumableStreamConfig::default();
    config.checkpoint_interval = 3;
    config.max_checkpoints = 10;
    config.max_replay_buffer = 100;
    let stream = ResumableStream::new(config);

    // Simulate streaming chunks
    let chunks = [
        "The Rust programming language ",
        "is designed for performance ",
        "and memory safety. ",
        "It uses an ownership model ",
        "to prevent data races ",
        "at compile time. ",
        "This makes it ideal ",
        "for systems programming ",
        "and WebAssembly targets.",
    ];

    for chunk in &chunks {
        let seq_id = stream.push(chunk);
        println!("  Pushed seq={}: {:?}", seq_id, chunk);
    }

    println!("\n  Stream stats:");
    println!("    Chunks:      {}", stream.chunk_count());
    println!("    Checkpoints: {}", stream.checkpoint_count());
    println!("    Current seq: {}", stream.current_sequence_id());

    // Checkpoint
    if let Some(cp) = stream.latest_checkpoint() {
        println!("    Latest checkpoint: seq={}, tokens={}, text_len={}",
            cp.sequence_id, cp.token_count, cp.accumulated_text.len());
    }

    // Resume from sequence 5
    let resumed = stream.resume_from(5);
    println!("\n  Resuming from seq=5: {} chunks replayed", resumed.len());
    for chunk in &resumed {
        println!("    seq={}: {:?}", chunk.sequence_id, chunk.text);
    }

    // SSE format
    if let Some(chunk) = stream.get_chunk(1) {
        let sse = format_sse_event(&chunk);
        println!("\n  SSE format for seq=1:");
        for line in sse.lines() {
            println!("    {}", line);
        }
    }

    // Finish stream
    stream.finish();
    println!("\n  Stream finished: {}", stream.is_finished());

    // Accumulated text
    let full_text = stream.accumulated_text();
    println!("  Full text: {:?}...", &full_text[..60.min(full_text.len())]);

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  Advanced streaming demo complete.");
    println!("  Capabilities: SSE, WebSocket RFC 6455, compression,");
    println!("    and resumable streaming with checkpoints.");
    println!("==========================================================");
}
