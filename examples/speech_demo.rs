//! Speech module demo: STT (speech-to-text) and TTS (text-to-speech) providers.
//!
//! Run with: cargo run --example speech_demo --features audio
//!
//! Demonstrates:
//! - Creating providers (OpenAI, Piper, Coqui, LocalSpeechProvider composite)
//! - Transcription (STT) with OpenAI Whisper
//! - Synthesis (TTS) with OpenAI or local Piper/Coqui
//! - Using `with_base_url` for a local whisper.cpp server
//! - Butler-based auto-detection (requires features: audio,butler)

use ai_assistant::{
    AudioFormat, CoquiTtsProvider, LocalSpeechProvider, OpenAISpeechProvider,
    PiperTtsProvider, SpeechProvider, SynthesisOptions,
};

fn main() -> anyhow::Result<()> {
    println!("=== ai_assistant speech module demo ===\n");

    // ── 1. Creating providers ──────────────────────────────────────────

    // OpenAI: reads OPENAI_API_KEY from the environment.
    println!("--- Provider creation ---");
    match OpenAISpeechProvider::from_env() {
        Ok(p) => println!("[ok] OpenAI provider created (voices: {:?})", p.tts_voices()),
        Err(e) => println!("[skip] OpenAI: {} (set OPENAI_API_KEY to enable)", e),
    }

    // Piper TTS: local neural TTS on port 5000 (no API key needed).
    let piper = PiperTtsProvider::default_local();
    println!(
        "[{}] Piper TTS (localhost:5000): supports_tts={}",
        if piper.is_available() { "ok" } else { "skip" },
        piper.supports_tts()
    );

    // Coqui TTS: local deep-learning TTS on port 5002.
    let coqui = CoquiTtsProvider::default_local()
        .with_speaker_id("default")
        .with_language_id("en");
    println!(
        "[{}] Coqui TTS (localhost:5002): supports_tts={}",
        if coqui.is_available() { "ok" } else { "skip" },
        coqui.supports_tts()
    );

    // LocalSpeechProvider: composite that delegates STT and TTS separately.
    // Combine a whisper.cpp HTTP server for STT + Piper for TTS.
    let whisper_http = OpenAISpeechProvider::new("not-needed")
        .with_base_url("http://localhost:8080");
    let local_combo = LocalSpeechProvider::new(
        Box::new(whisper_http),
        Box::new(PiperTtsProvider::default_local()),
    );
    println!(
        "[info] LocalSpeechProvider composite: stt={}, tts={}",
        local_combo.supports_stt(),
        local_combo.supports_tts()
    );

    // ── 2. Transcription (STT) example ─────────────────────────────────

    println!("\n--- Transcription (STT) ---");
    // In a real app you would read from a .wav file:
    //   let audio = std::fs::read("recording.wav")?;
    // Here we use a minimal valid WAV (44-byte header + silence) for demonstration.
    let wav_header = build_silent_wav(16000, 1.0);

    match OpenAISpeechProvider::from_env() {
        Ok(provider) => {
            println!("Sending 1s silent WAV to OpenAI Whisper...");
            match provider.transcribe(&wav_header, AudioFormat::Wav, Some("en")) {
                Ok(result) => {
                    println!("  Text: {:?}", result.text);
                    println!("  Language: {:?}", result.language);
                    println!("  Duration: {:.2}s", result.duration_secs);
                    println!("  Segments: {}", result.segments.len());
                }
                Err(e) => println!("  Transcription failed: {}", e),
            }
        }
        Err(_) => println!("[skip] Set OPENAI_API_KEY to test cloud transcription."),
    }

    // ── 3. Synthesis (TTS) example ─────────────────────────────────────

    println!("\n--- Synthesis (TTS) ---");
    let tts_text = "Hello from the ai_assistant speech module!";
    let opts = SynthesisOptions {
        voice: Some("alloy".into()),
        format: AudioFormat::Mp3,
        speed: 1.0,
        sample_rate: Some(24000),
    };

    match OpenAISpeechProvider::from_env() {
        Ok(provider) => {
            println!("Synthesizing with OpenAI TTS (voice=alloy)...");
            match provider.synthesize(tts_text, &opts) {
                Ok(result) => {
                    println!("  Format: {:?}", result.format);
                    println!("  Audio size: {} bytes", result.audio.len());
                    println!("  Estimated duration: {:.2}s", result.duration_secs);
                    println!("  Sample rate: {} Hz", result.sample_rate);
                    // To save: std::fs::write("output.mp3", &result.audio)?;
                }
                Err(e) => println!("  Synthesis failed: {}", e),
            }
        }
        Err(_) => println!("[skip] Set OPENAI_API_KEY to test cloud synthesis."),
    }

    // Try local Piper TTS as a fallback.
    if piper.is_available() {
        println!("Synthesizing with Piper TTS (local)...");
        match piper.synthesize(tts_text, &SynthesisOptions::default()) {
            Ok(result) => println!(
                "  Got {} bytes WAV, ~{:.2}s",
                result.audio.len(),
                result.duration_secs,
            ),
            Err(e) => println!("  Piper synthesis failed: {}", e),
        }
    }

    // ── 4. Using with_base_url for a local whisper.cpp server ──────────

    println!("\n--- Local whisper.cpp server (with_base_url) ---");
    println!("If you run whisper.cpp server on port 8080:");
    println!("  ./server -m ggml-base.en.bin --port 8080");
    let local_whisper = OpenAISpeechProvider::new("not-needed")
        .with_base_url("http://localhost:8080");
    println!(
        "  Provider: name={}, stt={}, base_url overridden to localhost:8080",
        local_whisper.name(),
        local_whisper.supports_stt(),
    );
    // Transcription would work the same way:
    //   local_whisper.transcribe(&audio_bytes, AudioFormat::Wav, Some("en"))?;

    // ── 5. Butler-based auto-detection ─────────────────────────────────

    butler_demo();

    println!("\n=== Done ===");
    Ok(())
}

/// Butler auto-detection of speech capabilities.
/// Requires both `audio` and `butler` feature flags.
#[cfg(all(feature = "audio", feature = "butler"))]
fn butler_demo() {
    use ai_assistant::butler::Butler;

    println!("\n--- Butler auto-detection ---");
    let mut butler = Butler::new();
    let _report = butler.scan();
    let (stt, tts) = butler.suggest_speech_config();
    println!("  Suggested STT provider: {:?}", stt);
    println!("  Suggested TTS provider: {:?}", tts);

    if let Some(ref stt_name) = stt {
        println!("  -> To create: create_speech_provider({:?})", stt_name);
    }
    if let Some(ref tts_name) = tts {
        println!("  -> To create: create_speech_provider({:?})", tts_name);
    }
    if stt.is_none() && tts.is_none() {
        println!("  No speech services detected. Start Piper/Coqui or set OPENAI_API_KEY.");
    }
}

#[cfg(not(all(feature = "audio", feature = "butler")))]
fn butler_demo() {
    println!("\n--- Butler auto-detection ---");
    println!("  [skip] Requires: cargo run --example speech_demo --features audio,butler");
}

/// Build a minimal valid WAV file containing silence.
/// `sample_rate` in Hz, `duration_secs` in seconds, 16-bit mono PCM.
fn build_silent_wav(sample_rate: u32, duration_secs: f32) -> Vec<u8> {
    let num_samples = (sample_rate as f32 * duration_secs) as u32;
    let data_size = num_samples * 2; // 16-bit = 2 bytes per sample
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size as usize + 8);
    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf.resize(buf.len() + data_size as usize, 0); // silence
    buf
}
