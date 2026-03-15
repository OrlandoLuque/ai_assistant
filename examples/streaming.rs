//! Streaming with backpressure example.
//!
//! Run with: cargo run --example streaming
//!
//! Demonstrates the streaming buffer, backpressure stream with
//! producer/consumer handles, chunking, and metrics tracking.
//! No external services or feature flags required.

use std::thread;
use std::time::Duration;

use ai_assistant::{BackpressureStream, Chunker, StreamBuffer, StreamingConfig};

fn main() {
    // 1. Basic StreamBuffer usage
    println!("=== StreamBuffer basics ===");
    let buffer = StreamBuffer::new(4096);

    buffer.push("Hello ".to_string()).expect("push failed");
    buffer.push("streaming ".to_string()).expect("push failed");
    buffer.push("world!".to_string()).expect("push failed");

    println!("Buffer fill: {:.1}%", buffer.fill_percentage());
    println!("Backpressure active: {}", buffer.is_backpressure_active());

    while let Some(chunk) = buffer.try_pop() {
        print!("{}", chunk);
    }
    println!();

    let metrics = buffer.metrics();
    println!(
        "Metrics: {} chunks produced, {} bytes produced",
        metrics.chunks_produced, metrics.bytes_produced
    );

    // 2. BackpressureStream with producer/consumer in separate threads
    println!("\n=== BackpressureStream (threaded) ===");
    let mut config = StreamingConfig::default();
    config.buffer_size = 1024;
    config.high_water_mark = 768;
    config.low_water_mark = 256;
    config.backpressure_timeout = Duration::from_secs(5);
    config.chunk_size = 64;
    config.auto_chunk = true;

    let stream = BackpressureStream::<String>::new(config);
    let producer = stream.producer();
    let consumer = stream.consumer();

    // Producer thread: sends several chunks
    let producer_handle = thread::spawn(move || {
        let sentences = [
            "Rust is a systems programming language.",
            "It focuses on safety and performance.",
            "The borrow checker prevents data races.",
            "Async/await enables concurrent I/O.",
        ];
        for sentence in &sentences {
            producer.send(sentence.to_string()).expect("send failed");
        }
        producer.close();
        println!("  Producer: sent {} messages, closed.", sentences.len());
    });

    // Consumer: read until done
    let consumer_handle = thread::spawn(move || {
        let mut count = 0;
        while let Some(chunk) = consumer.recv_timeout(Duration::from_secs(2)) {
            count += 1;
            println!("  Consumer[{}]: {}", count, chunk);
        }
        println!("  Consumer: received {} chunks total.", count);
    });

    producer_handle.join().expect("producer panicked");
    consumer_handle.join().expect("consumer panicked");

    // Final metrics
    let final_metrics = stream.metrics();
    println!(
        "Stream metrics: produced={} bytes, consumed={} bytes, throughput={:.0} B/s",
        final_metrics.bytes_produced,
        final_metrics.bytes_consumed,
        final_metrics.throughput_bps()
    );

    // 3. Chunker for splitting large text
    println!("\n=== Chunker ===");
    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs. \
                How vexingly quick daft zebras jump.";

    let chunker = Chunker::new(50);
    let chunks = chunker.chunk(text);
    println!(
        "Input length: {} chars -> {} chunks (word-preserving)",
        text.len(),
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  chunk[{}]: \"{}\" ({} chars)",
            i,
            chunk.trim(),
            chunk.len()
        );
    }

    let exact_chunker = Chunker::new(50).exact();
    let exact_chunks = exact_chunker.chunk(text);
    println!("Exact chunking: {} chunks", exact_chunks.len());
}
