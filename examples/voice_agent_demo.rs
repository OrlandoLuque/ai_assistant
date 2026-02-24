//! Real-time voice agent demo.
//!
//! Run with: cargo run --example voice_agent_demo --features voice-agent
//!
//! Demonstrates voice agent configuration, VAD (Voice Activity Detection),
//! session management, and the voice pipeline.
//! Note: actual audio I/O requires a running audio system.

use ai_assistant::{
    VoiceAgentConfig, VoiceAgent, VadConfig,
    InterruptionPolicy, TurnPolicy,
};

fn main() {
    println!("=== Real-Time Voice Agent Demo ===\n");

    // 1. Configure VAD (Voice Activity Detection)
    let vad_config = VadConfig {
        energy_threshold: 0.03,
        silence_duration_ms: 600,
        min_speech_duration_ms: 150,
        frame_size_ms: 20,
    };

    println!("VAD Configuration:");
    println!("  Energy threshold: {}", vad_config.energy_threshold);
    println!("  Silence duration: {}ms", vad_config.silence_duration_ms);
    println!("  Min speech duration: {}ms", vad_config.min_speech_duration_ms);
    println!("  Frame size: {}ms", vad_config.frame_size_ms);

    // 2. Configure the voice agent
    let config = VoiceAgentConfig {
        model_stt: "whisper-1".to_string(),
        model_tts: "tts-1".to_string(),
        vad_config,
        interruption_policy: InterruptionPolicy::Immediate,
        voice_id: "alloy".to_string(),
        sample_rate: 16000,
        chunk_size_ms: 20,
        turn_policy: TurnPolicy::NaturalOverlap,
    };

    println!("\nVoice Agent Config:");
    println!("  STT model: {}", config.model_stt);
    println!("  TTS model: {}", config.model_tts);
    println!("  Voice: {}", config.voice_id);
    println!("  Sample rate: {}Hz", config.sample_rate);
    println!("  Interruption: {:?}", config.interruption_policy);
    println!("  Turn policy: {:?}", config.turn_policy);

    // 3. Create the agent
    let mut agent = VoiceAgent::new(config).expect("valid config");
    println!("\nVoice agent created.");

    // 4. Start a session
    let session = agent.start_session().expect("session started");
    println!("Session ID: {}", session.session_id);
    println!("Session state: {:?}", session.state());

    // 5. Show default configuration
    let default_config = VoiceAgentConfig::default();
    println!("\nDefault config:");
    println!("  STT: {}", default_config.model_stt);
    println!("  TTS: {}", default_config.model_tts);
    println!("  Voice: {}", default_config.voice_id);
    println!("  Chunk size: {}ms", default_config.chunk_size_ms);

    println!("\n=== Done (no audio hardware needed for demo) ===");
}
