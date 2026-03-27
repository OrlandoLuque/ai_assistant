//! ai_virtual_mic — Real-time voice transformation GUI application.
//!
//! A desktop app for real-time audio processing with 4 modes:
//! - **Transform**: Mic -> Effects -> STT -> mood -> TTS (cloned voice) -> output
//! - **Direct**: Same as Transform, output to normal speaker
//! - **Passthrough**: Mic -> Effects chain only -> output
//! - **Monitor**: Mic -> display levels + speech detection (no output)
//!
//! Run: `cargo run --bin ai_virtual_mic --features audio-io`

use ai_assistant::{
    AiAssistant, AiConfig, AiProvider, AiResponse,
    AudioEffectChain, AudioModelCategory, AudioModelInfo, AudioModelRegistry, AutoGainControl,
    DiarizationResult, MfccSpeakerVerifier, ModelStatus, NoiseGate, SpeakerDiarizer,
    SpeakerGate, SpeakerIdentification, SpeakerVerifier,
    VoiceAgent, VoiceAgentConfig, VoiceLlmCallback,
    emotion_detection::{EmotionDetector, KeywordEmotionDetector},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

// ============================================================================
// Mode
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum Mode {
    Monitor,
    Passthrough,
    Transform,
    Direct,
    Agent,
}

impl Mode {
    fn label(&self) -> &str {
        match self {
            Self::Monitor => "Monitor",
            Self::Passthrough => "Passthrough",
            Self::Transform => "Transform",
            Self::Direct => "Direct Output",
            Self::Agent => "Agent",
        }
    }

    fn description(&self) -> &str {
        match self {
            Self::Monitor => "Display audio levels and speech detection (no output)",
            Self::Passthrough => "Mic -> Effects (noise gate, AGC) -> Output",
            Self::Transform => "Mic -> Effects -> STT -> Mood -> TTS (voice clone) -> Virtual Mic",
            Self::Direct => "Same as Transform but output to normal speaker",
            Self::Agent => "Mic -> STT -> Mood -> AI Agent (RAG) -> Mood -> TTS -> Output",
        }
    }

    fn all() -> &'static [Mode] {
        &[Mode::Monitor, Mode::Passthrough, Mode::Transform, Mode::Direct, Mode::Agent]
    }
}

// ============================================================================
// Agent Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum ConnectionMode {
    /// AiAssistant runs in-process with local LLM (Ollama, LM Studio)
    Local,
    /// Connect to a remote ai_assistant_server node (OpenAI-compatible API)
    RemoteNode,
    /// Direct LLM provider, no RAG
    DirectProvider,
    /// Full manual configuration
    Custom,
}

impl ConnectionMode {
    fn label(&self) -> &str {
        match self {
            Self::Local => "Local",
            Self::RemoteNode => "Remote Node",
            Self::DirectProvider => "Direct LLM",
            Self::Custom => "Custom",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AgentConfig {
    connection: ConnectionMode,
    provider: String,
    provider_url: String,
    #[serde(skip)]
    api_key: String,
    model: String,
    rag_tier: String,
    system_prompt: String,
    stt_provider: String,
    tts_provider: String,
    mood_aware: bool,
    /// Name the agent responds to (e.g. "Luna", "Jarvis"). Others can call
    /// this name to address the agent.  If empty the agent listens to everything.
    agent_name: String,
    /// Only respond when the agent's name is mentioned (or context suggests
    /// the speaker is addressing it). When false the agent responds to all speech.
    respond_only_when_addressed: bool,
    /// Wait for silence before playing TTS so the agent doesn't talk over people.
    /// Value is the silence gap in milliseconds that must be detected before the
    /// agent starts speaking its response.
    wait_for_silence_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            connection: ConnectionMode::Local,
            provider: "ollama".to_string(),
            provider_url: "http://localhost:11434".to_string(),
            api_key: String::new(),
            model: "llama3.2".to_string(),
            rag_tier: "fast".to_string(),
            system_prompt: "You are a helpful voice assistant. Keep responses concise and natural for speech.".to_string(),
            stt_provider: "openai".to_string(),
            tts_provider: "piper".to_string(),
            mood_aware: true,
            agent_name: "Luna".to_string(),
            respond_only_when_addressed: true,
            wait_for_silence_ms: 800,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PipelineState {
    Idle,
    Listening,
    Transcribing,
    Thinking,
    /// Agent has a response ready but is waiting for others to stop talking.
    WaitingForSilence,
    Speaking,
    Error,
}

impl PipelineState {
    fn label(&self) -> &str {
        match self {
            Self::Idle => "Idle",
            Self::Listening => "Listening...",
            Self::Transcribing => "Transcribing...",
            Self::Thinking => "Thinking...",
            Self::WaitingForSilence => "Waiting to speak...",
            Self::Speaking => "Speaking...",
            Self::Error => "Error",
        }
    }

    fn color(&self) -> egui::Color32 {
        match self {
            Self::Idle => egui::Color32::GRAY,
            Self::Listening => egui::Color32::from_rgb(0, 200, 80),
            Self::Transcribing => egui::Color32::from_rgb(80, 150, 255),
            Self::Thinking => egui::Color32::from_rgb(255, 180, 50),
            Self::WaitingForSilence => egui::Color32::from_rgb(255, 220, 100),
            Self::Speaking => egui::Color32::from_rgb(180, 100, 255),
            Self::Error => egui::Color32::RED,
        }
    }
}

#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    speaker: String,
    text: String,
    mood: String,
    mood_color: egui::Color32,
    timestamp: String,
}

fn mood_color(mood: &str) -> egui::Color32 {
    match mood.to_lowercase().as_str() {
        "happy" | "excited" => egui::Color32::from_rgb(80, 220, 80),
        "sad" | "fearful" => egui::Color32::from_rgb(100, 150, 255),
        "angry" | "frustrated" => egui::Color32::from_rgb(255, 80, 80),
        "confused" => egui::Color32::from_rgb(255, 220, 50),
        "calm" => egui::Color32::from_rgb(80, 220, 220),
        "bored" => egui::Color32::from_rgb(180, 180, 180),
        _ => egui::Color32::GRAY,
    }
}

/// Check if a transcript is addressed to the agent.
///
/// Looks for the agent name (case-insensitive) or contextual cues like
/// "hey", "oye", "tell me", "dime", direct questions after silence, etc.
fn is_addressed_to_agent(transcript: &str, agent_name: &str) -> bool {
    let lower = transcript.to_lowercase();
    let name_lower = agent_name.to_lowercase();

    // Direct name mention
    if !name_lower.is_empty() && lower.contains(&name_lower) {
        return true;
    }

    // Common address patterns (EN + ES)
    let prefixes = [
        "hey ", "oye ", "eh ", "a ver ", "escucha ",
        "tell me", "dime", "cuéntame", "explica",
    ];
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    // If the transcript ends with '?' it's likely directed at someone
    // (but only if respond_only_when_addressed is on, caller handles this)

    false
}

/// Try to extract a self-introduction from a transcript.
///
/// Detects patterns like "soy Carlos", "me llamo Ana", "I'm John", "my name is Sarah",
/// "call me Bob", "llámame Pedro". Returns the extracted name if found.
fn extract_self_introduction(transcript: &str) -> Option<String> {
    let lower = transcript.to_lowercase();

    let patterns: &[&str] = &[
        "me llamo ", "soy ", "mi nombre es ",
        "i'm ", "i am ", "my name is ", "call me ",
        "llámame ", "llamame ", "dime ", "puedes llamarme ",
    ];

    for pattern in patterns {
        if let Some(pos) = lower.find(pattern) {
            let after = &transcript[pos + pattern.len()..];
            // Take the first word(s) as name — up to comma, period, or 2 words
            let name: String = after
                .split(|c: char| c == ',' || c == '.' || c == '!' || c == '?')
                .next()
                .unwrap_or("")
                .trim()
                .split_whitespace()
                .take(2) // first + optional last name
                .collect::<Vec<_>>()
                .join(" ");
            if !name.is_empty() && name.len() < 30 {
                // Capitalize first letter
                let mut chars = name.chars();
                let capitalized = match chars.next() {
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                    None => name,
                };
                return Some(capitalized);
            }
        }
    }
    None
}

/// Resolve a diarization label ("Speaker 1") to a known name if available.
fn resolve_speaker_name(
    label: &str,
    aliases: &std::collections::HashMap<String, String>,
) -> String {
    if let Some(name) = aliases.get(label) {
        format!("{} ({})", name, label)
    } else {
        label.to_string()
    }
}

struct AgentState {
    config: AgentConfig,
    initialized: bool,
    assistant: Option<Arc<Mutex<AiAssistant>>>,
    conversation: Vec<ChatMessage>,
    pipeline_state: Arc<Mutex<PipelineState>>,
    session_cost: f32,
    error: Option<String>,
    show_advanced: bool,
    /// Maps diarization labels ("Speaker 1") to real names ("Carlos").
    /// Updated when someone says "soy X", "I'm X", etc.
    speaker_aliases: Arc<Mutex<std::collections::HashMap<String, String>>>,
}

// ============================================================================
// Audio Device Info
// ============================================================================

#[derive(Clone)]
struct DeviceInfo {
    name: String,
    index: usize,
    config_desc: String,
}

fn list_input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devs| {
            devs.enumerate()
                .map(|(i, d)| {
                    let name = d.name().unwrap_or_else(|_| "Unknown".to_string());
                    let desc = d
                        .default_input_config()
                        .map(|c| format!("{}Hz {}ch", c.sample_rate().0, c.channels()))
                        .unwrap_or_else(|_| "N/A".to_string());
                    DeviceInfo { name, index: i, config_desc: desc }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn list_output_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    host.output_devices()
        .map(|devs| {
            devs.enumerate()
                .map(|(i, d)| {
                    let name = d.name().unwrap_or_else(|_| "Unknown".to_string());
                    let desc = d
                        .default_output_config()
                        .map(|c| format!("{}Hz {}ch", c.sample_rate().0, c.channels()))
                        .unwrap_or_else(|_| "N/A".to_string());
                    DeviceInfo { name, index: i, config_desc: desc }
                })
                .collect()
        })
        .unwrap_or_default()
}

// ============================================================================
// Shared Audio State (updated by audio thread, read by GUI)
// ============================================================================

struct AudioState {
    rms: f32,
    peak: f32,
    db: f32,
    is_speech: bool,
    frames_processed: u64,
    /// Current speaker identification/diarization result.
    speaker_name: String,
    speaker_confidence: f32,
    /// Number of distinct speakers detected (diarization mode).
    diarized_count: usize,
    /// Processing latency: time from frame capture to processing complete (microseconds).
    latency_us: u64,
    /// Min/max/avg latency over a rolling window.
    latency_min_us: u64,
    latency_max_us: u64,
    latency_avg_us: u64,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            rms: 0.0,
            peak: 0.0,
            db: -60.0,
            is_speech: false,
            frames_processed: 0,
            speaker_name: String::new(),
            speaker_confidence: 0.0,
            diarized_count: 0,
            latency_us: 0,
            latency_min_us: u64::MAX,
            latency_max_us: 0,
            latency_avg_us: 0,
        }
    }
}

// ============================================================================
// Tab for bottom panel
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum BottomTab {
    Models,
    Speakers,
    AgentConfig,
}

// ============================================================================
// Model download state
// ============================================================================

struct DownloadProgress {
    model_id: String,
    bytes_downloaded: Arc<Mutex<u64>>,
    total_bytes: u64,
    finished: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
}

// ============================================================================
// App State
// ============================================================================

struct VirtualMicApp {
    // Mode
    mode: Mode,

    // Devices
    input_devices: Vec<DeviceInfo>,
    output_devices: Vec<DeviceInfo>,
    selected_input: usize,
    selected_output: usize,

    // Audio state (shared with audio thread)
    audio_state: Arc<Mutex<AudioState>>,

    // Stream control
    is_running: Arc<AtomicBool>,
    _input_stream: Option<cpal::Stream>,
    _output_stream: Option<cpal::Stream>,

    // Effects
    noise_gate_enabled: bool,
    noise_gate_threshold: f32,
    agc_enabled: bool,
    agc_target: f32,

    // Models
    model_registry: AudioModelRegistry,
    active_stt_model: Option<String>,
    active_tts_model: Option<String>,
    active_downloads: Vec<DownloadProgress>,

    // Speaker identification
    speaker_gate: Arc<Mutex<SpeakerGate>>,
    diarizer: Arc<Mutex<SpeakerDiarizer>>,
    /// true = diarization (auto), false = identification (enrolled profiles)
    use_diarization: bool,
    enrollment_name: String,
    enrollment_seconds: f32,
    is_enrolling: Arc<AtomicBool>,
    enrollment_buffer: Arc<Mutex<Vec<i16>>>,

    // Agent
    agent: AgentState,
    /// Shared conversation updated by processing thread (Agent mode)
    agent_conversation_shared: Arc<Mutex<Vec<ChatMessage>>>,

    // UI
    bottom_tab: Option<BottomTab>,
    status_message: String,
}

impl VirtualMicApp {
    fn new() -> Self {
        let input_devices = list_input_devices();
        let output_devices = list_output_devices();
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let speaker_gate = Arc::new(Mutex::new(SpeakerGate::new(verifier, 0.65)));
        let diarizer_verifier = Box::new(MfccSpeakerVerifier::new());
        let diarizer = Arc::new(Mutex::new(SpeakerDiarizer::with_defaults(diarizer_verifier)));

        let mut result = Self {
            mode: Mode::Monitor,
            input_devices,
            output_devices,
            selected_input: 0,
            selected_output: 0,
            audio_state: Arc::new(Mutex::new(AudioState::default())),
            is_running: Arc::new(AtomicBool::new(false)),
            _input_stream: None,
            _output_stream: None,
            noise_gate_enabled: true,
            noise_gate_threshold: -50.0,
            agc_enabled: true,
            agc_target: -18.0,
            model_registry: AudioModelRegistry::new(),
            active_stt_model: None,
            active_tts_model: None,
            active_downloads: Vec::new(),
            speaker_gate,
            diarizer,
            use_diarization: true,
            enrollment_name: String::new(),
            enrollment_seconds: 0.0,
            is_enrolling: Arc::new(AtomicBool::new(false)),
            enrollment_buffer: Arc::new(Mutex::new(Vec::new())),
            agent_conversation_shared: Arc::new(Mutex::new(Vec::new())),
            agent: AgentState {
                config: AgentConfig::default(),
                initialized: false,
                assistant: None,
                conversation: Vec::new(),
                pipeline_state: Arc::new(Mutex::new(PipelineState::Idle)),
                session_cost: 0.0,
                error: None,
                show_advanced: false,
                speaker_aliases: Arc::new(Mutex::new(std::collections::HashMap::new())),
            },
            bottom_tab: None,
            status_message: "Ready. Select devices and click Start.".to_string(),
        };
        result.load_config();
        result
    }

    /// Auto-detect virtual audio devices and configure for Discord AI mode.
    ///
    /// Detects VB-Cable (A/B) or VoiceMeeter (Banana/Potato) devices:
    /// - Input: captures Discord Web audio (VB-Cable A Output or VoiceMeeter Output)
    /// - Output: sends AI voice (VB-Cable B Input or VoiceMeeter Aux Input)
    fn apply_discord_preset(&mut self) {
        // ── Find INPUT device (captures what Discord Web plays) ──
        // Priority: VoiceMeeter Output > VB-Cable Output
        let vm_input = self.input_devices.iter().position(|d| {
            let lower = d.name.to_lowercase();
            lower.contains("voicemeeter output") && !lower.contains("aux")
        });
        let cable_a_input = self.input_devices.iter().position(|d| {
            let lower = d.name.to_lowercase();
            lower.contains("cable output") && !lower.contains("cable-b") && !lower.contains("cable-c")
        });
        let cable_any_input = self.input_devices.iter().position(|d| {
            d.name.to_lowercase().contains("cable output")
        });
        let best_input = vm_input.or(cable_a_input).or(cable_any_input);

        // ── Find OUTPUT device (sends AI voice to Discord Web mic) ──
        // Priority: VoiceMeeter Aux Input > VB-Cable B Input > any Cable Input
        let vm_output = self.output_devices.iter().position(|d| {
            d.name.to_lowercase().contains("voicemeeter aux input")
        });
        let cable_b_output = self.output_devices.iter().position(|d| {
            let lower = d.name.to_lowercase();
            lower.contains("cable-b input") || lower.contains("cable b input")
        });
        let cable_any_output = self.output_devices.iter().position(|d| {
            d.name.to_lowercase().contains("cable input")
        });
        let best_output = vm_output.or(cable_b_output).or(cable_any_output);

        let mut found = Vec::new();
        let mut missing = Vec::new();

        if let Some(idx) = best_input {
            self.selected_input = idx;
            found.push(format!("Input: {}", self.input_devices[idx].name));
        } else {
            missing.push("Virtual output (VoiceMeeter Output / VB-Cable Output)");
        }

        if let Some(idx) = best_output {
            self.selected_output = idx;
            found.push(format!("Output: {}", self.output_devices[idx].name));
        } else {
            missing.push("Virtual input (VoiceMeeter Aux Input / VB-Cable B Input)");
        }

        self.mode = Mode::Transform;
        self.use_diarization = true;

        if missing.is_empty() {
            self.status_message = format!(
                "Discord AI Mode! {}. Configure Discord Web: output → first virtual device, mic → second. Click Start.",
                found.join(", ")
            );
        } else {
            self.status_message = format!(
                "Partially configured. Found: [{}]. Missing: [{}]. Install VoiceMeeter Banana (free) or VB-Cable A+B.",
                if found.is_empty() { "none".to_string() } else { found.join(", ") },
                missing.join(", ")
            );
        }
    }

    fn start_audio(&mut self) {
        if self.is_running.load(Ordering::Relaxed) {
            return;
        }

        let host = cpal::default_host();
        let input_device = match host.input_devices() {
            Ok(mut devs) => devs.nth(self.selected_input),
            Err(e) => { self.status_message = format!("Error: {}", e); return; }
        };
        let input_device = match input_device {
            Some(d) => d,
            None => { self.status_message = "No input device found".to_string(); return; }
        };
        let input_config = match input_device.default_input_config() {
            Ok(c) => c,
            Err(e) => { self.status_message = format!("No config: {}", e); return; }
        };

        let sample_rate = input_config.sample_rate().0;
        let channels = input_config.channels() as usize;

        let rb = HeapRb::<f32>::new(sample_rate as usize * 2);
        let (mut producer, mut consumer) = rb.split();

        // Timestamp ring buffer: one timestamp per input callback invocation
        let ts_rb = HeapRb::<std::time::Instant>::new(1024);
        let (mut ts_producer, mut ts_consumer) = ts_rb.split();

        let state = self.audio_state.clone();
        let running = self.is_running.clone();
        let gate = self.speaker_gate.clone();
        let diarizer = self.diarizer.clone();
        let use_diarization = self.use_diarization;
        let enrolling = self.is_enrolling.clone();
        let enroll_buf = self.enrollment_buffer.clone();

        // Agent mode resources
        let agent_assistant = self.agent.assistant.clone();
        let agent_pipeline_state = self.agent.pipeline_state.clone();
        let agent_name = self.agent.config.agent_name.clone();
        let agent_respond_only = self.agent.config.respond_only_when_addressed;
        let agent_wait_silence = self.agent.config.wait_for_silence_ms;
        let agent_conv_ref = self.agent_conversation_shared.clone();
        let agent_aliases = self.agent.speaker_aliases.clone();

        // Build effect chain
        let mut effect_chain = AudioEffectChain::new();
        if self.noise_gate_enabled {
            effect_chain.add_effect(Box::new(NoiseGate::new(self.noise_gate_threshold)));
        }
        if self.agc_enabled {
            effect_chain.add_effect(Box::new(AutoGainControl::new(self.agc_target)));
        }
        let effect_chain = Arc::new(Mutex::new(effect_chain));
        let mode = self.mode;

        // Input stream — stamps a timestamp per callback for latency measurement
        let input_stream = input_device
            .build_input_stream(
                &input_config.config(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let _ = ts_producer.try_push(std::time::Instant::now());
                    for &sample in data {
                        let _ = producer.try_push(sample);
                    }
                },
                |err| eprintln!("Input error: {}", err),
                None,
            )
            .ok();

        if let Some(ref stream) = input_stream {
            let _ = stream.play();
        }

        // Processing thread
        let process_chain = effect_chain;
        std::thread::spawn(move || {
            let frame_size = (sample_rate as usize) / 50; // 20ms
            let mut frame_buf = Vec::with_capacity(frame_size * channels);
            // Accumulate ~1 second for speaker ID (run every 50 frames = 1s)
            let mut speaker_buf: Vec<i16> = Vec::new();
            let mut frame_counter = 0u32;
            // Latency rolling window (last 100 measurements)
            let mut latency_window: Vec<u64> = Vec::with_capacity(100);

            while running.load(Ordering::Relaxed) {
                // Grab the most recent capture timestamp (best-effort)
                let mut capture_ts: Option<std::time::Instant> = None;
                while let Some(ts) = ts_consumer.try_pop() {
                    capture_ts = Some(ts);
                }

                frame_buf.clear();
                let target = frame_size * channels;
                while frame_buf.len() < target {
                    if let Some(s) = consumer.try_pop() {
                        frame_buf.push(s);
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        if !running.load(Ordering::Relaxed) { return; }
                    }
                }

                // Mono mix
                let mono: Vec<f32> = if channels > 1 {
                    frame_buf.chunks(channels)
                        .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                        .collect()
                } else {
                    frame_buf.clone()
                };

                // Convert to i16
                let samples_i16: Vec<i16> = mono.iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                // RMS + peak
                let rms = if mono.is_empty() { 0.0 } else {
                    (mono.iter().map(|&s| s * s).sum::<f32>() / mono.len() as f32).sqrt()
                };
                let peak = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                let db = if rms > 0.0 { 20.0 * rms.log10() } else { -60.0 };
                let is_speech = db > -35.0;

                // Collect samples for enrollment if active
                if enrolling.load(Ordering::Relaxed) {
                    if let Ok(mut buf) = enroll_buf.lock() {
                        buf.extend_from_slice(&samples_i16);
                    }
                }

                // Apply effects if not Monitor
                if mode != Mode::Monitor {
                    let mut processed = samples_i16.clone();
                    if let Ok(mut chain) = process_chain.lock() {
                        chain.process_frame(&mut processed, sample_rate);
                    }
                }

                // Accumulate for speaker identification (every ~1 second)
                if is_speech {
                    speaker_buf.extend_from_slice(&samples_i16);
                }
                frame_counter += 1;

                let mut speaker_name = String::new();
                let mut speaker_confidence = 0.0f32;
                let mut diarized_count = 0usize;

                if frame_counter >= 50 && !speaker_buf.is_empty() {
                    if use_diarization {
                        // Diarization: auto-detect distinct speakers
                        if let Ok(mut d) = diarizer.lock() {
                            match d.process(&speaker_buf, sample_rate) {
                                DiarizationResult::Assigned { label, confidence, .. } => {
                                    speaker_name = label;
                                    speaker_confidence = confidence;
                                }
                                DiarizationResult::NewSpeaker { label, .. } => {
                                    speaker_name = format!("{} (new!)", label);
                                    speaker_confidence = 1.0;
                                }
                                DiarizationResult::Inconclusive => {}
                            }
                            diarized_count = d.speaker_count();
                        }
                    } else {
                        // Identification: match against enrolled profiles
                        if let Ok(mut g) = gate.lock() {
                            match g.identify(&speaker_buf, sample_rate) {
                                SpeakerIdentification::Identified { name, confidence, .. } => {
                                    speaker_name = name;
                                    speaker_confidence = confidence;
                                }
                                SpeakerIdentification::Unknown { confidence } => {
                                    speaker_name = "Unknown".to_string();
                                    speaker_confidence = confidence;
                                }
                                SpeakerIdentification::NoProfiles => {
                                    speaker_name = "(no profiles enrolled)".to_string();
                                }
                            }
                        }
                    }
                    speaker_buf.clear();
                    frame_counter = 0;
                }

                // ── Agent mode: process speech segments ──────────────
                if mode == Mode::Agent {
                    // When we have a speaker identification result + accumulated speech,
                    // send to the LLM agent
                    if frame_counter == 0 && !speaker_name.is_empty()
                        && !speaker_name.starts_with('(')
                    {
                        // Resolve speaker name via aliases
                        let display_name = {
                            let aliases = agent_aliases.lock().unwrap_or_else(|e| e.into_inner());
                            resolve_speaker_name(&speaker_name, &aliases)
                        };

                        // Build a pseudo-transcript from the speaker label
                        // (real STT will come when VoiceAgent is fully wired)
                        let transcript = format!("[{} is speaking]", display_name);

                        // Try to detect self-introduction ("soy Carlos", "I'm John")
                        if let Some(real_name) = extract_self_introduction(&transcript) {
                            // Strip "(new!)" suffix from speaker label
                            let clean_label = speaker_name.replace(" (new!)", "");
                            if let Ok(mut aliases) = agent_aliases.lock() {
                                aliases.insert(clean_label, real_name);
                            }
                        }

                        // Check if addressed to agent
                        let should_respond = !agent_respond_only
                            || is_addressed_to_agent(&transcript, &agent_name);

                        if should_respond {
                            if let Some(ref assistant) = agent_assistant {
                                if let Ok(mut ps) = agent_pipeline_state.lock() {
                                    *ps = PipelineState::Thinking;
                                }

                                if let Ok(mut ast) = assistant.lock() {
                                    // Build prompt with resolved speaker name
                                    let prompt = format!(
                                        "[{} says:] {}",
                                        display_name, transcript
                                    );
                                    ast.send_message_simple(prompt);

                                    // Poll for response (blocking, with timeout)
                                    let mut response_text = String::new();
                                    let start = std::time::Instant::now();
                                    loop {
                                        if let Some(resp) = ast.poll_response() {
                                            match resp {
                                                AiResponse::Chunk(chunk) => {
                                                    response_text.push_str(&chunk);
                                                }
                                                AiResponse::Complete(text) => {
                                                    if response_text.is_empty() {
                                                        response_text = text;
                                                    }
                                                    break;
                                                }
                                                AiResponse::Error(_) => break,
                                                _ => {}
                                            }
                                        }
                                        if start.elapsed().as_secs() > 10 {
                                            break;
                                        }
                                        std::thread::sleep(std::time::Duration::from_millis(50));
                                    }

                                    if !response_text.is_empty() {
                                        let now = chrono::Local::now().format("%H:%M:%S").to_string();
                                        if let Ok(mut conv) = agent_conv_ref.lock() {
                                            conv.push(ChatMessage {
                                                role: "user".to_string(),
                                                speaker: display_name.clone(),
                                                text: transcript.clone(),
                                                mood: String::new(),
                                                mood_color: egui::Color32::GRAY,
                                                timestamp: now.clone(),
                                            });
                                            conv.push(ChatMessage {
                                                role: "agent".to_string(),
                                                speaker: agent_name.clone(),
                                                text: response_text,
                                                mood: String::new(),
                                                mood_color: egui::Color32::GRAY,
                                                timestamp: now,
                                            });
                                        }
                                    }
                                }

                                if let Ok(mut ps) = agent_pipeline_state.lock() {
                                    *ps = PipelineState::Listening;
                                }
                            }
                        }
                    }
                }

                // Measure processing latency
                let latency_us = capture_ts
                    .map(|ts| ts.elapsed().as_micros() as u64)
                    .unwrap_or(0);

                if latency_us > 0 {
                    latency_window.push(latency_us);
                    if latency_window.len() > 100 {
                        latency_window.remove(0);
                    }
                }

                let (lat_min, lat_max, lat_avg) = if latency_window.is_empty() {
                    (0, 0, 0)
                } else {
                    let min = *latency_window.iter().min().unwrap_or(&0);
                    let max = *latency_window.iter().max().unwrap_or(&0);
                    let avg = latency_window.iter().sum::<u64>() / latency_window.len() as u64;
                    (min, max, avg)
                };

                // Update shared state
                if let Ok(mut st) = state.lock() {
                    st.rms = rms;
                    st.peak = peak;
                    st.db = db;
                    st.is_speech = is_speech;
                    st.frames_processed += 1;
                    st.latency_us = latency_us;
                    st.latency_min_us = lat_min;
                    st.latency_max_us = lat_max;
                    st.latency_avg_us = lat_avg;
                    if !speaker_name.is_empty() {
                        st.speaker_name = speaker_name;
                        st.speaker_confidence = speaker_confidence;
                    }
                    if diarized_count > 0 {
                        st.diarized_count = diarized_count;
                    }
                }
            }
        });

        self.is_running.store(true, Ordering::Relaxed);
        self._input_stream = input_stream;
        self.status_message = format!(
            "Running: {} @ {}Hz {}ch",
            input_device.name().unwrap_or_default(), sample_rate, channels
        );
    }

    fn stop_audio(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        self._input_stream = None;
        self._output_stream = None;
        self.status_message = "Stopped.".to_string();
        self.save_config();
    }

    fn refresh_devices(&mut self) {
        self.input_devices = list_input_devices();
        self.output_devices = list_output_devices();
    }

    fn start_enrollment(&mut self) {
        if !self.is_running.load(Ordering::Relaxed) {
            self.status_message = "Start audio capture first before enrolling.".to_string();
            return;
        }
        if self.enrollment_name.trim().is_empty() {
            self.status_message = "Enter a name for the speaker first.".to_string();
            return;
        }
        // Clear buffer and start collecting
        if let Ok(mut buf) = self.enrollment_buffer.lock() {
            buf.clear();
        }
        self.is_enrolling.store(true, Ordering::Relaxed);
        self.enrollment_seconds = 0.0;
        self.status_message = format!("Recording voice for '{}'... Speak now!", self.enrollment_name);

        // Auto-stop after 5 seconds
        let enrolling = self.is_enrolling.clone();
        let enroll_buf = self.enrollment_buffer.clone();
        let gate = self.speaker_gate.clone();
        let name = self.enrollment_name.clone();
        let state_msg = Arc::new(Mutex::new(String::new()));
        let state_msg_clone = state_msg.clone();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(5));
            enrolling.store(false, Ordering::Relaxed);

            let samples = if let Ok(buf) = enroll_buf.lock() {
                buf.clone()
            } else {
                Vec::new()
            };

            if samples.len() < 16000 {
                if let Ok(mut msg) = state_msg_clone.lock() {
                    *msg = "Enrollment failed: not enough audio captured.".to_string();
                }
                return;
            }

            if let Ok(mut g) = gate.lock() {
                match g.enroll(&samples, 16000, &name, false) {
                    Ok(id) => {
                        if let Ok(mut msg) = state_msg_clone.lock() {
                            *msg = format!("Enrolled '{}' (id: {})", name, id);
                        }
                    }
                    Err(e) => {
                        if let Ok(mut msg) = state_msg_clone.lock() {
                            *msg = format!("Enrollment failed: {}", e);
                        }
                    }
                }
            }
        });
    }

    fn start_model_download(&mut self, model: AudioModelInfo) {
        let model_id = model.id.clone();
        let bytes_downloaded = Arc::new(Mutex::new(0u64));
        let finished = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));

        let progress = DownloadProgress {
            model_id: model_id.clone(),
            bytes_downloaded: bytes_downloaded.clone(),
            total_bytes: model.size_bytes,
            finished: finished.clone(),
            error: error.clone(),
        };
        self.active_downloads.push(progress);

        let registry_dir = self.model_registry.model_dir().to_string();
        std::thread::spawn(move || {
            let registry = AudioModelRegistry::with_directory(&registry_dir);
            let bd = bytes_downloaded;
            match registry.download_model(&model, |downloaded, _total| {
                if let Ok(mut b) = bd.lock() { *b = downloaded; }
            }) {
                Ok(_path) => {}
                Err(e) => {
                    if let Ok(mut err) = error.lock() { *err = Some(e); }
                }
            }
            finished.store(true, Ordering::Relaxed);
        });

        self.status_message = format!("Downloading {}...", model_id);
    }

    fn delete_model(&self, model: &AudioModelInfo) {
        let path = self.model_registry.model_path(model);
        let _ = std::fs::remove_file(&path);
    }

    // ========================================================================
    // UI Panels
    // ========================================================================

    fn render_models_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Audio Models");
        ui.small(format!("Directory: {}", self.model_registry.model_dir()));
        ui.add_space(4.0);

        // Active model selectors
        ui.horizontal(|ui| {
            ui.label("Active STT:");
            let stt_models = self.model_registry.models_by_category(AudioModelCategory::Stt);
            let stt_label = self.active_stt_model.as_deref().unwrap_or("(none)");
            egui::ComboBox::from_id_source("active_stt")
                .selected_text(stt_label)
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.active_stt_model, None, "(none)").clicked() {}
                    for m in &stt_models {
                        if matches!(self.model_registry.model_status(m), ModelStatus::Installed { .. }) {
                            let val = Some(m.id.clone());
                            ui.selectable_value(&mut self.active_stt_model, val, &m.name);
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Active TTS:");
            let tts_models = self.model_registry.models_by_category(AudioModelCategory::Tts);
            let tts_label = self.active_tts_model.as_deref().unwrap_or("(none)");
            egui::ComboBox::from_id_source("active_tts")
                .selected_text(tts_label)
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.active_tts_model, None, "(none)").clicked() {}
                    for m in &tts_models {
                        if matches!(self.model_registry.model_status(m), ModelStatus::Installed { .. }) {
                            let val = Some(m.id.clone());
                            ui.selectable_value(&mut self.active_tts_model, val, &m.name);
                        }
                    }
                });
        });

        ui.add_space(8.0);

        // Clean up finished downloads
        self.active_downloads.retain(|d| !d.finished.load(Ordering::Relaxed));

        // Model catalog table
        egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
            egui::Grid::new("models_grid").striped(true).min_col_width(80.0).show(ui, |ui| {
                ui.strong("Status");
                ui.strong("Name");
                ui.strong("Category");
                ui.strong("Size");
                ui.strong("Action");
                ui.end_row();

                // Collect actions to perform after the grid (can't mutate during iteration)
                let mut to_download: Option<AudioModelInfo> = None;
                let mut to_delete: Option<AudioModelInfo> = None;

                for model in self.model_registry.catalog().to_vec() {
                    let status = self.model_registry.model_status(&model);

                    // Check if downloading
                    let downloading = self.active_downloads.iter().find(|d| d.model_id == model.id);

                    match &status {
                        ModelStatus::Installed { .. } => {
                            ui.label(egui::RichText::new("Installed").color(egui::Color32::GREEN));
                        }
                        ModelStatus::Corrupted { .. } => {
                            ui.label(egui::RichText::new("Corrupted!").color(egui::Color32::RED));
                        }
                        ModelStatus::NotInstalled => {
                            if let Some(dl) = downloading {
                                let bytes = dl.bytes_downloaded.lock().map(|b| *b).unwrap_or(0);
                                let pct = if dl.total_bytes > 0 {
                                    (bytes as f32 / dl.total_bytes as f32 * 100.0) as u32
                                } else { 0 };
                                ui.label(format!("{}%", pct));
                            } else {
                                ui.label("Not installed");
                            }
                        }
                    }

                    ui.label(&model.name);
                    ui.label(model.category.to_string());
                    ui.label(&model.size_estimate);

                    match &status {
                        ModelStatus::Installed { .. } => {
                            if ui.small_button("Delete").clicked() {
                                to_delete = Some(model.clone());
                            }
                        }
                        ModelStatus::NotInstalled if downloading.is_none() => {
                            let has_url = !model.url.is_empty();
                            let btn = ui.add_enabled(has_url, egui::Button::new("Download"));
                            if btn.clicked() {
                                to_download = Some(model.clone());
                            }
                            if !has_url {
                                btn.on_hover_text("No download URL (install manually)");
                            }
                        }
                        _ => { ui.label(""); }
                    }
                    ui.end_row();
                }

                // Apply deferred actions
                if let Some(m) = to_download {
                    // We'll handle this after the grid
                    self.start_model_download(m);
                }
                if let Some(m) = to_delete {
                    self.delete_model(&m);
                }
            });
        });
    }

    fn render_speakers_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Speaker Detection");
        ui.add_space(4.0);

        // Mode toggle
        ui.horizontal(|ui| {
            ui.label("Mode:");
            if ui.selectable_label(self.use_diarization, "Diarization (auto)").clicked() {
                self.use_diarization = true;
            }
            if ui.selectable_label(!self.use_diarization, "Identification (enrolled)").clicked() {
                self.use_diarization = false;
            }
        });
        ui.small(if self.use_diarization {
            "Auto-detects distinct speakers without enrollment. Labels: Speaker 1, Speaker 2..."
        } else {
            "Identifies speakers by name. Requires enrollment first."
        });
        ui.add_space(4.0);

        if self.use_diarization {
            // Diarization info
            ui.group(|ui| {
                if let Ok(d) = self.diarizer.lock() {
                    ui.label(format!("Distinct speakers detected: {}", d.speaker_count()));
                    if !d.clusters().is_empty() {
                        egui::Grid::new("diarization_grid").striped(true).show(ui, |ui| {
                            ui.strong("Label");
                            ui.strong("Segments");
                            ui.end_row();
                            for cluster in d.clusters() {
                                ui.label(&cluster.label);
                                ui.label(format!("{}", cluster.segment_count));
                                ui.end_row();
                            }
                        });
                    }
                }
                if ui.button("Reset Clusters").clicked() {
                    if let Ok(mut d) = self.diarizer.lock() {
                        d.reset();
                    }
                }
            });

            ui.add_space(8.0);
        }

        if !self.use_diarization {
        // Enrollment section
        ui.group(|ui| {
            ui.label("Enroll New Speaker");
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut self.enrollment_name);
            });

            let is_enrolling = self.is_enrolling.load(Ordering::Relaxed);
            if is_enrolling {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Recording... speak clearly for 5 seconds");
                });
            } else if ui.button("Record & Enroll (5 seconds)").clicked() {
                self.start_enrollment();
            }

            ui.small("Speak clearly for 5 seconds. Audio capture must be running.");
        });

        ui.add_space(8.0);

        // Enrolled profiles list
        let mut to_remove: Option<String> = None;
        if let Ok(gate) = self.speaker_gate.lock() {
            let profiles = gate.profiles();
            if profiles.is_empty() {
                ui.label("No speakers enrolled yet.");
            } else {
                egui::Grid::new("speakers_grid").striped(true).show(ui, |ui| {
                    ui.strong("Name");
                    ui.strong("Owner");
                    ui.strong("Samples");
                    ui.strong("Action");
                    ui.end_row();

                    for profile in profiles {
                        ui.label(&profile.name);
                        ui.label(if profile.is_owner { "Yes" } else { "No" });
                        ui.label(format!("{}", profile.embeddings.len()));
                        if ui.small_button("Remove").clicked() {
                            to_remove = Some(profile.name.clone());
                        }
                        ui.end_row();
                    }
                });
            }
        }

        // Deferred remove
        if let Some(name) = to_remove {
            if let Ok(mut gate) = self.speaker_gate.lock() {
                gate.remove_profile(&name);
            }
        }

        } // end if !self.use_diarization

        ui.add_space(8.0);

        // Live speaker display (both modes)
        ui.separator();
        ui.heading("Live Speaker");
        if let Ok(st) = self.audio_state.lock() {
            if st.speaker_name.is_empty() {
                ui.label("Waiting for speech...");
            } else {
                let color = if st.speaker_name == "Unknown" || st.speaker_name.starts_with('(') {
                    egui::Color32::GRAY
                } else {
                    egui::Color32::from_rgb(100, 255, 100)
                };
                ui.horizontal(|ui| {
                    ui.label("Speaker:");
                    ui.label(egui::RichText::new(&st.speaker_name).color(color).size(18.0).strong());
                    if st.speaker_confidence > 0.0 {
                        ui.label(format!("({:.0}%)", st.speaker_confidence * 100.0));
                    }
                });
            }
        }

        // Show known name aliases
        if let Ok(aliases) = self.agent.speaker_aliases.lock() {
            if !aliases.is_empty() {
                ui.add_space(4.0);
                ui.separator();
                ui.label("Known Names:");
                for (label, name) in aliases.iter() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{} = {}", label, name));
                    });
                }
            }
        }
    }

    // ========================================================================
    // Agent Config Panel (bottom tab)
    // ========================================================================

    fn render_agent_config_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Agent Configuration");

        // Quick presets
        ui.horizontal(|ui| {
            ui.label("Preset:");
            if ui.small_button("Local (Ollama)").clicked() {
                self.agent.config.connection = ConnectionMode::Local;
                self.agent.config.provider = "ollama".to_string();
                self.agent.config.provider_url = "http://localhost:11434".to_string();
                self.agent.config.model = "llama3.2".to_string();
                self.agent.config.stt_provider = "openai".to_string();
                self.agent.config.tts_provider = "piper".to_string();
                self.agent.config.rag_tier = "fast".to_string();
            }
            if ui.small_button("Cloud (OpenAI)").clicked() {
                self.agent.config.connection = ConnectionMode::DirectProvider;
                self.agent.config.provider = "openai".to_string();
                self.agent.config.model = "gpt-4o-mini".to_string();
                self.agent.config.stt_provider = "openai".to_string();
                self.agent.config.tts_provider = "openai-expressive".to_string();
            }
            if ui.small_button("Remote Node").clicked() {
                self.agent.config.connection = ConnectionMode::RemoteNode;
                self.agent.config.provider_url = "http://localhost:3000".to_string();
                self.agent.config.model = "default".to_string();
            }
        });

        ui.add_space(4.0);

        // Connection mode selector
        ui.horizontal(|ui| {
            ui.label("Connection:");
            for mode in [ConnectionMode::Local, ConnectionMode::RemoteNode, ConnectionMode::DirectProvider, ConnectionMode::Custom] {
                if ui.selectable_label(self.agent.config.connection == mode, mode.label()).clicked() {
                    self.agent.config.connection = mode;
                }
            }
        });

        ui.add_space(4.0);

        match self.agent.config.connection {
            ConnectionMode::Local => {
                ui.horizontal(|ui| {
                    ui.label("Provider URL:");
                    ui.text_edit_singleline(&mut self.agent.config.provider_url);
                });
                ui.horizontal(|ui| {
                    ui.label("Model:");
                    ui.text_edit_singleline(&mut self.agent.config.model);
                });
                ui.horizontal(|ui| {
                    ui.label("RAG Tier:");
                    egui::ComboBox::from_id_source("rag_tier")
                        .selected_text(&self.agent.config.rag_tier)
                        .show_ui(ui, |ui| {
                            for tier in ["disabled", "fast", "semantic", "enhanced", "thorough", "agentic", "graph", "full"] {
                                ui.selectable_value(&mut self.agent.config.rag_tier, tier.to_string(), tier);
                            }
                        });
                });
            }
            ConnectionMode::RemoteNode => {
                ui.horizontal(|ui| {
                    ui.label("Node URL:");
                    ui.text_edit_singleline(&mut self.agent.config.provider_url);
                });
                ui.small("Connect to a running ai_assistant_server (OpenAI-compatible API)");
            }
            ConnectionMode::DirectProvider => {
                ui.horizontal(|ui| {
                    ui.label("Provider:");
                    egui::ComboBox::from_id_source("direct_provider")
                        .selected_text(&self.agent.config.provider)
                        .show_ui(ui, |ui| {
                            for p in ["ollama", "openai", "anthropic", "lm-studio", "groq"] {
                                ui.selectable_value(&mut self.agent.config.provider, p.to_string(), p);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Model:");
                    ui.text_edit_singleline(&mut self.agent.config.model);
                });
                if self.agent.config.provider != "ollama" && self.agent.config.provider != "lm-studio" {
                    ui.horizontal(|ui| {
                        ui.label("API Key:");
                        ui.add(egui::TextEdit::singleline(&mut self.agent.config.api_key).password(true));
                    });
                }
                ui.small("Direct LLM query without RAG or knowledge base");
            }
            ConnectionMode::Custom => {
                ui.horizontal(|ui| {
                    ui.label("Provider:");
                    ui.text_edit_singleline(&mut self.agent.config.provider);
                });
                ui.horizontal(|ui| {
                    ui.label("URL:");
                    ui.text_edit_singleline(&mut self.agent.config.provider_url);
                });
                ui.horizontal(|ui| {
                    ui.label("API Key:");
                    ui.add(egui::TextEdit::singleline(&mut self.agent.config.api_key).password(true));
                });
                ui.horizontal(|ui| {
                    ui.label("Model:");
                    ui.text_edit_singleline(&mut self.agent.config.model);
                });
                ui.horizontal(|ui| {
                    ui.label("RAG Tier:");
                    ui.text_edit_singleline(&mut self.agent.config.rag_tier);
                });
            }
        }

        ui.add_space(4.0);
        ui.separator();

        // STT / TTS / Mood
        ui.horizontal(|ui| {
            ui.label("STT:");
            egui::ComboBox::from_id_source("stt_sel")
                .width(120.0)
                .selected_text(&self.agent.config.stt_provider)
                .show_ui(ui, |ui| {
                    for p in ["openai", "google", "piper", "coqui"] {
                        ui.selectable_value(&mut self.agent.config.stt_provider, p.to_string(), p);
                    }
                    #[cfg(feature = "whisper-local")]
                    ui.selectable_value(&mut self.agent.config.stt_provider, "whisper-local".to_string(), "whisper-local");
                });

            ui.label("TTS:");
            egui::ComboBox::from_id_source("tts_sel")
                .width(120.0)
                .selected_text(&self.agent.config.tts_provider)
                .show_ui(ui, |ui| {
                    for p in ["piper", "coqui", "openai", "openai-expressive", "elevenlabs"] {
                        ui.selectable_value(&mut self.agent.config.tts_provider, p.to_string(), p);
                    }
                });

            ui.checkbox(&mut self.agent.config.mood_aware, "Mood-aware");
        });

        // Agent identity & behavior
        ui.add_space(4.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Agent name:");
            ui.add(egui::TextEdit::singleline(&mut self.agent.config.agent_name).desired_width(100.0));
            ui.checkbox(&mut self.agent.config.respond_only_when_addressed, "Only respond when addressed");
        });
        ui.horizontal(|ui| {
            ui.label("Wait for silence:");
            ui.add(egui::Slider::new(&mut self.agent.config.wait_for_silence_ms, 0..=3000).suffix(" ms"));
            if self.agent.config.wait_for_silence_ms > 0 {
                ui.small("(won't talk over others)");
            }
        });

        // System prompt
        ui.horizontal(|ui| {
            ui.label("System prompt:");
        });
        let name = &self.agent.config.agent_name;
        let default_prompt = format!(
            "You are {}, a voice assistant in a group conversation. \
             Keep responses concise (1-3 sentences) and natural for speech. \
             You can hear who is talking (labeled as Speaker 1, Speaker 2, etc.).",
            if name.is_empty() { "a helpful assistant" } else { name.as_str() }
        );
        if self.agent.config.system_prompt.is_empty() {
            self.agent.config.system_prompt = default_prompt;
        }
        ui.add(
            egui::TextEdit::multiline(&mut self.agent.config.system_prompt)
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );

        ui.add_space(4.0);

        // Initialize / status
        ui.horizontal(|ui| {
            if self.agent.initialized {
                ui.label(egui::RichText::new("Agent initialized").color(egui::Color32::GREEN));
                if ui.button("Reinitialize").clicked() {
                    self.initialize_agent();
                }
            } else if ui.button(egui::RichText::new("Initialize Agent").size(16.0)).clicked() {
                self.initialize_agent();
            }

            if let Some(ref err) = self.agent.error {
                ui.label(egui::RichText::new(err).color(egui::Color32::RED));
            }
        });
    }

    // ========================================================================
    // Agent Chat View (central panel when mode==Agent)
    // ========================================================================

    fn render_agent_chat(&mut self, ui: &mut egui::Ui) {
        // Pipeline status bar
        let ps = self.agent.pipeline_state.lock()
            .map(|s| *s)
            .unwrap_or(PipelineState::Idle);

        ui.horizontal(|ui| {
            let steps = [
                ("Listening", PipelineState::Listening),
                ("STT", PipelineState::Transcribing),
                ("LLM", PipelineState::Thinking),
                ("Wait", PipelineState::WaitingForSilence),
                ("TTS", PipelineState::Speaking),
            ];
            for (label, step) in steps {
                let active = ps == step;
                let color = if active { step.color() } else { egui::Color32::from_gray(60) };
                ui.label(egui::RichText::new(label).color(color).strong());
                if step != PipelineState::Speaking {
                    ui.label(egui::RichText::new("→").color(egui::Color32::from_gray(80)));
                }
            }
            ui.separator();
            ui.label(egui::RichText::new(ps.label()).color(ps.color()));

            if self.agent.session_cost > 0.001 {
                ui.separator();
                ui.small(format!("~${:.3}", self.agent.session_cost));
            }
        });

        ui.separator();

        // Sync conversation from shared thread state
        if let Ok(conv) = self.agent_conversation_shared.lock() {
            if conv.len() != self.agent.conversation.len() {
                self.agent.conversation = conv.clone();
            }
        }

        // Chat messages
        if self.agent.conversation.is_empty() {
            ui.centered_and_justified(|ui| {
                if !self.agent.initialized {
                    ui.label(egui::RichText::new("Configure and initialize the agent in the bottom panel")
                        .color(egui::Color32::GRAY).size(16.0));
                } else if !self.is_running.load(Ordering::Relaxed) {
                    ui.label(egui::RichText::new("Agent ready. Click Start to begin listening")
                        .color(egui::Color32::YELLOW).size(16.0));
                } else {
                    ui.label(egui::RichText::new("Listening... speak to the agent")
                        .color(egui::Color32::from_rgb(0, 200, 80)).size(16.0));
                }
            });
        } else {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.agent.conversation {
                        let is_user = msg.role == "user";
                        let bg = if is_user {
                            egui::Color32::from_rgb(40, 50, 70)
                        } else {
                            egui::Color32::from_rgb(30, 45, 30)
                        };

                        egui::Frame::none()
                            .fill(bg)
                            .inner_margin(8.0)
                            .outer_margin(egui::Margin::symmetric(4.0, 2.0))
                            .rounding(6.0)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    // Speaker / role label
                                    let role_icon = if is_user { "🎤" } else { "🤖" };
                                    let speaker_label = if msg.speaker.is_empty() {
                                        if is_user { "You".to_string() } else { "Agent".to_string() }
                                    } else {
                                        msg.speaker.clone()
                                    };
                                    ui.label(egui::RichText::new(format!("{} {}", role_icon, speaker_label)).strong());

                                    // Mood badge
                                    if !msg.mood.is_empty() && msg.mood != "neutral" {
                                        ui.label(
                                            egui::RichText::new(&msg.mood)
                                                .color(msg.mood_color)
                                                .small()
                                                .strong(),
                                        );
                                    }

                                    // Timestamp
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.label(egui::RichText::new(&msg.timestamp).small().color(egui::Color32::GRAY));
                                    });
                                });
                                ui.label(&msg.text);
                            });
                    }
                });
        }
    }

    fn config_path(&self) -> String {
        format!("{}/virtual_mic_config.json", self.model_registry.model_dir())
    }

    fn save_config(&self) {
        let cfg = serde_json::json!({
            "mode": self.mode,
            "selected_input": self.selected_input,
            "selected_output": self.selected_output,
            "noise_gate_enabled": self.noise_gate_enabled,
            "noise_gate_threshold": self.noise_gate_threshold,
            "agc_enabled": self.agc_enabled,
            "agc_target": self.agc_target,
            "use_diarization": self.use_diarization,
            "agent": self.agent.config,
        });
        let path = self.config_path();
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&cfg) {
            let _ = std::fs::write(&path, json);
        }
    }

    fn load_config(&mut self) {
        let path = self.config_path();
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(m) = cfg.get("mode").and_then(|v| serde_json::from_value::<Mode>(v.clone()).ok()) {
                    self.mode = m;
                }
                if let Some(v) = cfg.get("selected_input").and_then(|v| v.as_u64()) {
                    self.selected_input = v as usize;
                }
                if let Some(v) = cfg.get("selected_output").and_then(|v| v.as_u64()) {
                    self.selected_output = v as usize;
                }
                if let Some(v) = cfg.get("noise_gate_enabled").and_then(|v| v.as_bool()) {
                    self.noise_gate_enabled = v;
                }
                if let Some(v) = cfg.get("noise_gate_threshold").and_then(|v| v.as_f64()) {
                    self.noise_gate_threshold = v as f32;
                }
                if let Some(v) = cfg.get("agc_enabled").and_then(|v| v.as_bool()) {
                    self.agc_enabled = v;
                }
                if let Some(v) = cfg.get("agc_target").and_then(|v| v.as_f64()) {
                    self.agc_target = v as f32;
                }
                if let Some(v) = cfg.get("use_diarization").and_then(|v| v.as_bool()) {
                    self.use_diarization = v;
                }
                if let Some(agent_val) = cfg.get("agent") {
                    if let Ok(ac) = serde_json::from_value::<AgentConfig>(agent_val.clone()) {
                        self.agent.config = ac;
                    }
                }
                self.status_message = "Config loaded.".to_string();
            }
        }
    }

    fn initialize_agent(&mut self) {
        self.agent.error = None;

        let mut assistant = AiAssistant::new();
        let url = self.agent.config.provider_url.clone();

        match self.agent.config.connection {
            ConnectionMode::Local => {
                assistant.config.provider = AiProvider::Ollama;
                assistant.config.ollama_url = url;
                assistant.config.selected_model = self.agent.config.model.clone();
            }
            ConnectionMode::RemoteNode => {
                assistant.config.provider = AiProvider::OpenAICompatible {
                    base_url: url,
                };
                assistant.config.selected_model = self.agent.config.model.clone();
            }
            ConnectionMode::DirectProvider => {
                assistant.config.provider = match self.agent.config.provider.as_str() {
                    "openai" => AiProvider::OpenAI,
                    "anthropic" => AiProvider::Anthropic,
                    "lm-studio" => AiProvider::LMStudio,
                    "groq" => AiProvider::Groq,
                    _ => AiProvider::Ollama,
                };
                assistant.config.selected_model = self.agent.config.model.clone();
                assistant.config.api_key = self.agent.config.api_key.clone();
            }
            ConnectionMode::Custom => {
                assistant.config.provider = AiProvider::OpenAICompatible {
                    base_url: url,
                };
                assistant.config.selected_model = self.agent.config.model.clone();
                assistant.config.api_key = self.agent.config.api_key.clone();
            }
        }

        // Set system prompt
        if !self.agent.config.system_prompt.is_empty() {
            assistant.set_system_prompt(&self.agent.config.system_prompt);
        }

        self.agent.assistant = Some(Arc::new(Mutex::new(assistant)));
        self.agent.initialized = true;
        self.agent.conversation.clear();
        self.status_message = "Agent initialized. Click Start to begin.".to_string();
        self.save_config();
    }
}

// ============================================================================
// egui rendering
// ============================================================================

impl eframe::App for VirtualMicApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running.load(Ordering::Relaxed) {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("ai_virtual_mic");
                ui.separator();
                ui.label(&self.status_message);
            });
        });

        // Bottom panel — always present so side panel respects its space
        let has_tab = self.bottom_tab.is_some();
        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(has_tab)
            .min_height(if has_tab { 150.0 } else { 28.0 })
            .default_height(if has_tab { 300.0 } else { 28.0 })
            .show(ctx, |ui| {
                // Tab bar (always visible)
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.bottom_tab == Some(BottomTab::Models), "Models").clicked() {
                        self.bottom_tab = if self.bottom_tab == Some(BottomTab::Models) { None } else { Some(BottomTab::Models) };
                    }
                    if ui.selectable_label(self.bottom_tab == Some(BottomTab::Speakers), "Speakers").clicked() {
                        self.bottom_tab = if self.bottom_tab == Some(BottomTab::Speakers) { None } else { Some(BottomTab::Speakers) };
                    }
                    if ui.selectable_label(self.bottom_tab == Some(BottomTab::AgentConfig), "Agent").clicked() {
                        self.bottom_tab = if self.bottom_tab == Some(BottomTab::AgentConfig) { None } else { Some(BottomTab::AgentConfig) };
                    }
                    if has_tab {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Close").clicked() {
                                self.bottom_tab = None;
                            }
                        });
                    }
                });

                if has_tab {
                    ui.separator();
                    match self.bottom_tab {
                        Some(BottomTab::Models) => self.render_models_panel(ui),
                        Some(BottomTab::Speakers) => self.render_speakers_panel(ui),
                        Some(BottomTab::AgentConfig) => self.render_agent_config_panel(ui),
                        None => {}
                    }
                }
            });

        // Left panel — controls
        egui::SidePanel::left("controls").min_width(280.0).show(ctx, |ui| {
            ui.add_space(8.0);

            // Mode selector
            ui.heading("Mode");
            for &mode in Mode::all() {
                if ui.selectable_label(self.mode == mode, mode.label()).clicked() {
                    self.mode = mode;
                    if mode == Mode::Agent && self.bottom_tab != Some(BottomTab::AgentConfig) {
                        self.bottom_tab = Some(BottomTab::AgentConfig);
                    }
                }
            }
            ui.small(self.mode.description());
            ui.add_space(8.0);

            // Preset
            if ui.button("Discord AI Mode").on_hover_text(
                "Auto-detect virtual audio devices and configure:\n\
                 - Input: VoiceMeeter Output / VB-Cable Output (captures Discord Web)\n\
                 - Output: VoiceMeeter Aux Input / VB-Cable B Input (AI voice)\n\
                 - Mode: Transform + Diarization\n\n\
                 Requires VoiceMeeter Banana (free) or VB-Cable A+B"
            ).clicked() {
                self.refresh_devices();
                self.apply_discord_preset();
            }

            ui.add_space(8.0);
            ui.separator();

            // Devices
            ui.add_space(8.0);
            ui.heading("Input Device");
            egui::ComboBox::from_id_source("input_dev")
                .width(250.0)
                .selected_text(
                    self.input_devices.get(self.selected_input)
                        .map(|d| format!("{} ({})", d.name, d.config_desc))
                        .unwrap_or_else(|| "None".to_string()),
                )
                .show_ui(ui, |ui| {
                    for (i, dev) in self.input_devices.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_input, i, format!("{} ({})", dev.name, dev.config_desc));
                    }
                });

            ui.add_space(4.0);
            ui.heading("Output Device");
            egui::ComboBox::from_id_source("output_dev")
                .width(250.0)
                .selected_text(
                    self.output_devices.get(self.selected_output)
                        .map(|d| format!("{} ({})", d.name, d.config_desc))
                        .unwrap_or_else(|| "None".to_string()),
                )
                .show_ui(ui, |ui| {
                    for (i, dev) in self.output_devices.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_output, i, format!("{} ({})", dev.name, dev.config_desc));
                    }
                });

            if ui.button("Refresh Devices").clicked() {
                self.refresh_devices();
            }

            ui.add_space(12.0);
            ui.separator();

            // Start / Stop
            ui.add_space(8.0);
            let running = self.is_running.load(Ordering::Relaxed);
            if running {
                if ui.button(egui::RichText::new("Stop").color(egui::Color32::RED).size(18.0)).clicked() {
                    self.stop_audio();
                }
            } else if ui.button(egui::RichText::new("Start").color(egui::Color32::GREEN).size(18.0)).clicked() {
                self.start_audio();
            }

            ui.add_space(12.0);
            ui.separator();

            // Effects
            ui.add_space(8.0);
            ui.heading("Effects Chain");
            ui.checkbox(&mut self.noise_gate_enabled, "Noise Gate");
            if self.noise_gate_enabled {
                ui.add(egui::Slider::new(&mut self.noise_gate_threshold, -80.0..=-20.0).text("Threshold dB"));
            }
            ui.checkbox(&mut self.agc_enabled, "Auto Gain Control");
            if self.agc_enabled {
                ui.add(egui::Slider::new(&mut self.agc_target, -30.0..=-6.0).text("Target dB"));
            }

        });

        // Central panel — VU meter + visualization (or Agent chat)
        egui::CentralPanel::default().show(ctx, |ui| {
            let state = self.audio_state.lock().unwrap_or_else(|e| e.into_inner());

            if self.mode == Mode::Agent {
                // Compact VU + speaker on one line
                ui.horizontal(|ui| {
                    let db_n = ((state.db + 60.0) / 60.0).clamp(0.0, 1.0);
                    let color = if state.db > -6.0 { egui::Color32::RED }
                        else if state.db > -20.0 { egui::Color32::YELLOW }
                        else { egui::Color32::from_rgb(0, 200, 80) };
                    let (r, _) = ui.allocate_exact_size(egui::vec2(120.0, 14.0), egui::Sense::hover());
                    ui.painter().rect_filled(r, 2.0, egui::Color32::from_gray(30));
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(r.min, egui::vec2(db_n * 120.0, 14.0)),
                        2.0, color,
                    );
                    ui.label(egui::RichText::new(format!("{:.0}dB", state.db)).monospace().small());

                    if !state.speaker_name.is_empty() && !state.speaker_name.starts_with('(') {
                        ui.separator();
                        ui.label(egui::RichText::new(&state.speaker_name)
                            .color(egui::Color32::from_rgb(100, 255, 100)).small());
                    }

                    if state.latency_avg_us > 0 {
                        ui.separator();
                        ui.label(egui::RichText::new(format_latency(state.latency_avg_us)).monospace().small());
                    }
                });
                ui.separator();

                drop(state); // release lock before rendering chat
                self.render_agent_chat(ui);
                return;
            }

            ui.add_space(16.0);
            ui.heading("Audio Levels");
            ui.add_space(8.0);

            // VU meter bar
            let available_width = ui.available_width() - 20.0;
            let db_normalized = ((state.db + 60.0) / 60.0).clamp(0.0, 1.0);
            let bar_width = db_normalized * available_width;

            let (rect, _) = ui.allocate_exact_size(egui::vec2(available_width, 30.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(40));

            let color = if state.db > -6.0 { egui::Color32::RED }
                else if state.db > -20.0 { egui::Color32::YELLOW }
                else { egui::Color32::from_rgb(0, 200, 80) };
            let bar_rect = egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, 30.0));
            ui.painter().rect_filled(bar_rect, 4.0, color);
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER,
                format!("{:.1} dB", state.db), egui::FontId::monospace(16.0), egui::Color32::WHITE);

            ui.add_space(12.0);

            let peak_db = if state.peak > 0.0 { 20.0 * state.peak.log10() } else { -60.0 };
            ui.horizontal(|ui| {
                ui.label("RMS:"); ui.monospace(format!("{:.1} dB", state.db));
                ui.separator();
                ui.label("Peak:"); ui.monospace(format!("{:.1} dB", peak_db));
                ui.separator();
                ui.label("Frames:"); ui.monospace(format!("{}", state.frames_processed));
            });

            ui.add_space(8.0);

            // Latency display
            ui.heading("Latency");
            ui.horizontal(|ui| {
                let lat_color = if state.latency_avg_us < 5000 {
                    egui::Color32::from_rgb(0, 200, 80) // green < 5ms
                } else if state.latency_avg_us < 20000 {
                    egui::Color32::YELLOW // yellow < 20ms
                } else {
                    egui::Color32::RED // red >= 20ms
                };

                ui.label("Current:");
                ui.label(egui::RichText::new(format_latency(state.latency_us)).color(lat_color).monospace());
                ui.separator();
                ui.label("Avg:");
                ui.label(egui::RichText::new(format_latency(state.latency_avg_us)).color(lat_color).monospace());
                ui.separator();
                ui.label("Min:");
                ui.monospace(format_latency(state.latency_min_us));
                ui.separator();
                ui.label("Max:");
                ui.monospace(format_latency(state.latency_max_us));
            });

            // Latency bar (target: 20ms frame budget)
            let budget_ms = 20.0f32;
            let actual_ms = state.latency_avg_us as f32 / 1000.0;
            let bar_pct = (actual_ms / budget_ms).clamp(0.0, 2.0) / 2.0; // 0-100% maps to 0-40ms
            let bar_w = bar_pct * (ui.available_width() - 20.0);

            let (bar_rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width() - 20.0, 12.0), egui::Sense::hover());
            ui.painter().rect_filled(bar_rect, 2.0, egui::Color32::from_gray(30));

            // Budget marker at 50% (= 20ms)
            let budget_x = bar_rect.min.x + bar_rect.width() * 0.5;
            ui.painter().line_segment(
                [egui::pos2(budget_x, bar_rect.min.y), egui::pos2(budget_x, bar_rect.max.y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
            );

            let lat_bar_color = if actual_ms < budget_ms * 0.5 {
                egui::Color32::from_rgb(0, 200, 80)
            } else if actual_ms < budget_ms {
                egui::Color32::YELLOW
            } else {
                egui::Color32::RED
            };
            let filled = egui::Rect::from_min_size(bar_rect.min, egui::vec2(bar_w, 12.0));
            ui.painter().rect_filled(filled, 2.0, lat_bar_color);

            ui.small(format!("Budget: {:.0}ms per frame | Bar: 0-40ms range", budget_ms));

            ui.add_space(8.0);

            // Speech + speaker indicator
            ui.horizontal(|ui| {
                if state.is_speech {
                    ui.label(egui::RichText::new("SPEECH").color(egui::Color32::GREEN).size(16.0).strong());
                } else {
                    ui.label(egui::RichText::new("Silence").color(egui::Color32::GRAY).size(14.0));
                }

                if !state.speaker_name.is_empty() && state.speaker_name != "(no profiles enrolled)" {
                    ui.separator();
                    let speaker_color = if state.speaker_name == "Unknown" {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::from_rgb(100, 255, 100)
                    };
                    ui.label(egui::RichText::new(format!("Speaker: {}", state.speaker_name))
                        .color(speaker_color).size(16.0).strong());
                    if state.speaker_confidence > 0.0 {
                        ui.label(format!("({:.0}%)", state.speaker_confidence * 100.0));
                    }
                }
            });

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(8.0);

            // Mode info
            ui.heading(format!("Mode: {}", self.mode.label()));
            ui.label(self.mode.description());

            if self.mode == Mode::Transform || self.mode == Mode::Direct {
                ui.add_space(8.0);
                ui.group(|ui| {
                    ui.label("Voice Transform Pipeline:");
                    ui.label("  1. Audio Effects (noise gate, AGC)");
                    ui.label("  2. STT (speech-to-text)");
                    ui.label("  3. Mood detection");
                    ui.label("  4. TTS with cloned voice");
                });
            }
        });
    }
}

// ============================================================================
// Main
// ============================================================================

/// Format microseconds as human-readable latency.
fn format_latency(us: u64) -> String {
    if us == 0 || us == u64::MAX {
        return "--".to_string();
    }
    if us < 1000 {
        format!("{}us", us)
    } else if us < 1_000_000 {
        format!("{:.1}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 650.0])
            .with_min_inner_size([700.0, 450.0])
            .with_title("ai_virtual_mic"),
        ..Default::default()
    };

    eframe::run_native(
        "ai_virtual_mic",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::new(VirtualMicApp::new())
        }),
    )
}
