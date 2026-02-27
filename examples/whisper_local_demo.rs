//! Example: whisper_local_demo -- Demonstrates local Whisper STT configuration.
//!
//! Run with: cargo run --example whisper_local_demo --features "whisper-local"
//!
//! This example showcases WhisperLocalProvider for local speech-to-text:
//! configuration, builder pattern, audio format support, and transcription API.
//! No actual model file is required — the example demonstrates the API surface
//! and validates behavior with synthetic audio data.

use ai_assistant::{AudioFormat, SpeechProvider, WhisperLocalProvider};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Whisper Local STT Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Provider configuration
    // ------------------------------------------------------------------
    println!("--- 1. WhisperLocalProvider Configuration ---\n");

    // Basic creation with model path
    let provider = WhisperLocalProvider::new("/path/to/ggml-base.en.bin");
    println!("  Provider name: {}", provider.name());
    println!("  Supports STT:  {}", provider.supports_stt());
    println!("  Supports TTS:  {}", provider.supports_tts());
    println!("  TTS voices:    {:?}", provider.tts_voices());

    // Builder pattern with language hint and thread control
    let provider_es = WhisperLocalProvider::new("/path/to/ggml-large-v3.bin")
        .with_language("es")
        .with_threads(4);
    println!("\n  Spanish provider: {}", provider_es.name());
    println!("  Supports STT: {}", provider_es.supports_stt());

    // ------------------------------------------------------------------
    // 2. Audio format support
    // ------------------------------------------------------------------
    println!("\n--- 2. AudioFormat Support ---\n");

    let formats = [
        ("audio.wav", AudioFormat::from_extension("wav")),
        ("audio.pcm", AudioFormat::from_extension("pcm")),
        ("audio.mp3", AudioFormat::from_extension("mp3")),
        ("audio.ogg", AudioFormat::from_extension("ogg")),
        ("audio.flac", AudioFormat::from_extension("flac")),
        ("audio.xyz", AudioFormat::from_extension("xyz")),
    ];

    for (file, format) in &formats {
        match format {
            Some(fmt) => println!("  {} -> {:?} (ext: {})", file, fmt, fmt.extension()),
            None => println!("  {} -> unsupported", file),
        }
    }

    // ------------------------------------------------------------------
    // 3. Transcription attempt with synthetic WAV
    // ------------------------------------------------------------------
    println!("\n--- 3. Transcription API ---\n");

    // Create a minimal synthetic WAV file header (44 bytes RIFF/WAVE header + silence)
    let sample_rate: u32 = 16000;
    let bits_per_sample: u16 = 16;
    let num_channels: u16 = 1;
    let num_samples: u32 = sample_rate; // 1 second of audio
    let data_size: u32 = num_samples * (bits_per_sample as u32 / 8) * num_channels as u32;
    let file_size: u32 = 36 + data_size;

    let mut wav_data: Vec<u8> = Vec::new();
    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&file_size.to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");
    // fmt subchunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // subchunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav_data.extend_from_slice(&num_channels.to_le_bytes());
    wav_data.extend_from_slice(&sample_rate.to_le_bytes());
    wav_data.extend_from_slice(&(sample_rate * num_channels as u32 * bits_per_sample as u32 / 8).to_le_bytes());
    wav_data.extend_from_slice(&(num_channels * bits_per_sample / 8).to_le_bytes());
    wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data subchunk (1 second of silence)
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&data_size.to_le_bytes());
    wav_data.extend_from_slice(&vec![0u8; data_size as usize]);

    println!("  Synthetic WAV: {} bytes ({} samples @ {} Hz)",
        wav_data.len(), num_samples, sample_rate);

    // Attempt transcription — will fail because model file doesn't exist,
    // but demonstrates the API
    match provider.transcribe(&wav_data, AudioFormat::Wav, Some("en")) {
        Ok(result) => {
            println!("  Transcription: '{}'", result.text);
            println!("  Language: {:?}", result.language);
            println!("  Duration: {:.2}s", result.duration_secs);
            println!("  Segments: {}", result.segments.len());
            if let Some(conf) = result.confidence {
                println!("  Confidence: {:.2}", conf);
            }
        }
        Err(e) => {
            println!("  Expected error (no model file): {}", e);
            println!("  To run with real transcription:");
            println!("    1. Download a GGML model from:");
            println!("       https://huggingface.co/ggerganov/whisper.cpp/tree/main");
            println!("    2. Use WhisperLocalProvider::new(\"path/to/ggml-base.en.bin\")");
        }
    }

    // ------------------------------------------------------------------
    // 4. Model selection guide
    // ------------------------------------------------------------------
    println!("\n--- 4. Available Whisper Models ---\n");

    let models = [
        ("ggml-tiny.en.bin", "~75 MB", "Fastest, English only"),
        ("ggml-base.en.bin", "~142 MB", "Good balance, English only"),
        ("ggml-small.en.bin", "~466 MB", "Better accuracy, English only"),
        ("ggml-medium.bin", "~1.5 GB", "High accuracy, multilingual"),
        ("ggml-large-v3.bin", "~3.1 GB", "Best accuracy, multilingual"),
    ];

    println!("  {:<25} {:<12} {}", "Model", "Size", "Notes");
    println!("  {}", "-".repeat(60));
    for (name, size, notes) in &models {
        println!("  {:<25} {:<12} {}", name, size, notes);
    }

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  whisper-local demo complete.");
    println!("  Capabilities: local STT, language detection, WAV/PCM");
    println!("    input, configurable threads, GGML model support.");
    println!("==========================================================");
}
