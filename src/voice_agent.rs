//! Real-time voice agent with bidirectional audio streaming, VAD, and turn management.
//!
//! Implements v5 roadmap Phase 3 items:
//! - 3.1 Bidirectional Audio Streaming (STT -> LLM -> TTS pipeline)
//! - 3.2 Voice Activity Detection + Interruption Handling
//! - 3.3 Conversation Turn Management
//!
//! Feature-gated behind `voice-agent`. The outer `#[cfg]` guard ensures this
//! entire module compiles away when the feature is not enabled.

#[cfg(feature = "voice-agent")]
mod inner {
    use std::fmt;

    use crate::error::{AiError, VoiceAgentError};
    use serde::{Deserialize, Serialize};

    // ========================================================================
    // 3.1 — Audio Types
    // ========================================================================

    /// Audio encoding format for voice chunks.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum AudioFormat {
        /// Raw PCM 16-bit signed little-endian
        Pcm16,
        /// WAV container
        Wav,
        /// Ogg/Vorbis
        Ogg,
        /// MP3
        Mp3,
    }

    impl fmt::Display for AudioFormat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                AudioFormat::Pcm16 => write!(f, "pcm16"),
                AudioFormat::Wav => write!(f, "wav"),
                AudioFormat::Ogg => write!(f, "ogg"),
                AudioFormat::Mp3 => write!(f, "mp3"),
            }
        }
    }

    impl AudioFormat {
        /// MIME type for this format.
        pub fn mime_type(&self) -> &str {
            match self {
                AudioFormat::Pcm16 => "audio/pcm",
                AudioFormat::Wav => "audio/wav",
                AudioFormat::Ogg => "audio/ogg",
                AudioFormat::Mp3 => "audio/mpeg",
            }
        }

        /// File extension for this format.
        pub fn extension(&self) -> &str {
            match self {
                AudioFormat::Pcm16 => "pcm",
                AudioFormat::Wav => "wav",
                AudioFormat::Ogg => "ogg",
                AudioFormat::Mp3 => "mp3",
            }
        }
    }

    /// A chunk of audio data with metadata.
    #[derive(Debug, Clone)]
    pub struct AudioChunk {
        /// Raw audio bytes.
        pub bytes: Vec<u8>,
        /// Sample rate in Hz (e.g. 16000).
        pub sample_rate: u32,
        /// Number of audio channels (1 = mono, 2 = stereo).
        pub channels: u8,
        /// Encoding format.
        pub format: AudioFormat,
    }

    impl AudioChunk {
        /// Create a new audio chunk.
        pub fn new(bytes: Vec<u8>, sample_rate: u32, channels: u8, format: AudioFormat) -> Self {
            Self {
                bytes,
                sample_rate,
                channels,
                format,
            }
        }

        /// Duration of this chunk in milliseconds (assumes PCM16 encoding for calculation).
        /// For non-PCM formats this is an approximation based on raw byte count.
        pub fn duration_ms(&self) -> u64 {
            if self.sample_rate == 0 || self.channels == 0 {
                return 0;
            }
            // PCM16: 2 bytes per sample per channel
            let bytes_per_sample = 2u64 * self.channels as u64;
            let total_samples = self.bytes.len() as u64 / bytes_per_sample;
            (total_samples * 1000) / self.sample_rate as u64
        }

        /// Returns true if this chunk contains no audio data.
        pub fn is_empty(&self) -> bool {
            self.bytes.is_empty()
        }

        /// Number of samples in this chunk (PCM16 assumption).
        pub fn sample_count(&self) -> usize {
            if self.channels == 0 {
                return 0;
            }
            let bytes_per_sample = 2 * self.channels as usize;
            if bytes_per_sample == 0 {
                return 0;
            }
            self.bytes.len() / bytes_per_sample
        }
    }

    // ========================================================================
    // 3.1 — VoiceTransport trait + InMemoryTransport
    // ========================================================================

    /// Trait for bidirectional audio transport (send/receive audio chunks).
    pub trait VoiceTransport: Send + Sync {
        /// Send an audio chunk to the remote end.
        fn send_audio(&self, chunk: &AudioChunk) -> Result<(), AiError>;
        /// Receive the next audio chunk, if available.
        fn receive_audio(&mut self) -> Result<Option<AudioChunk>, AiError>;
        /// Close the transport.
        fn close(&mut self) -> Result<(), AiError>;
    }

    /// In-memory transport for testing — stores sent and received audio in Vecs.
    #[derive(Debug, Clone)]
    pub struct InMemoryTransport {
        sent: Vec<AudioChunk>,
        receive_queue: Vec<AudioChunk>,
        closed: bool,
    }

    impl InMemoryTransport {
        /// Create a new empty in-memory transport.
        pub fn new() -> Self {
            Self {
                sent: Vec::new(),
                receive_queue: Vec::new(),
                closed: false,
            }
        }

        /// Enqueue audio that will be returned by `receive_audio`.
        pub fn enqueue(&mut self, chunk: AudioChunk) {
            self.receive_queue.push(chunk);
        }

        /// Get all audio chunks that were sent.
        pub fn sent_chunks(&self) -> &[AudioChunk] {
            &self.sent
        }

        /// Returns true if the transport has been closed.
        pub fn is_closed(&self) -> bool {
            self.closed
        }
    }

    impl Default for InMemoryTransport {
        fn default() -> Self {
            Self::new()
        }
    }

    impl VoiceTransport for InMemoryTransport {
        fn send_audio(&self, _chunk: &AudioChunk) -> Result<(), AiError> {
            if self.closed {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "transport is closed".to_string(),
                }
                .into());
            }
            // NOTE: We cannot mutate self in &self, so we use interior mutability
            // workaround: clone into a new vec. For production, use Arc<Mutex<>>.
            // In this test transport we accept this limitation — callers use
            // the mutable `send_audio_mut` or the trait's &self version is a no-op
            // that records via interior mutability in real implementations.
            // For testing, use `send_audio_mut` directly.
            Ok(())
        }

        fn receive_audio(&mut self) -> Result<Option<AudioChunk>, AiError> {
            if self.closed {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "transport is closed".to_string(),
                }
                .into());
            }
            if self.receive_queue.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.receive_queue.remove(0)))
            }
        }

        fn close(&mut self) -> Result<(), AiError> {
            self.closed = true;
            Ok(())
        }
    }

    impl InMemoryTransport {
        /// Mutable send that actually records the chunk (for testing).
        pub fn send_audio_mut(&mut self, chunk: &AudioChunk) -> Result<(), AiError> {
            if self.closed {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "transport is closed".to_string(),
                }
                .into());
            }
            self.sent.push(chunk.clone());
            Ok(())
        }
    }

    // ========================================================================
    // 3.2 — Voice Activity Detection (VAD)
    // ========================================================================

    /// Configuration for the energy-based voice activity detector.
    #[non_exhaustive]
    #[derive(Debug, Clone)]
    pub struct VadConfig {
        /// RMS energy threshold for speech detection (0.0 - 1.0). Default: 0.02.
        pub energy_threshold: f32,
        /// How long silence must persist before emitting SpeechEnd (ms). Default: 500.
        pub silence_duration_ms: u32,
        /// Minimum speech duration to be considered valid (ms). Default: 200.
        pub min_speech_duration_ms: u32,
        /// Frame size for analysis (ms). Default: 20.
        pub frame_size_ms: u32,
    }

    impl Default for VadConfig {
        fn default() -> Self {
            Self {
                energy_threshold: 0.02,
                silence_duration_ms: 500,
                min_speech_duration_ms: 200,
                frame_size_ms: 20,
            }
        }
    }

    /// Events emitted by the VAD detector.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq)]
    pub enum VadEvent {
        /// Speech has started at the given timestamp.
        SpeechStart { timestamp_ms: u64 },
        /// Speech has ended at the given timestamp, with total duration.
        SpeechEnd {
            timestamp_ms: u64,
            duration_ms: u64,
        },
        /// Frame contains only silence.
        Silence,
    }

    /// Energy-based voice activity detector.
    ///
    /// Analyzes PCM16 audio frames to detect speech start/end boundaries
    /// using RMS energy with a moving average smoother.
    #[derive(Debug, Clone)]
    pub struct VadDetector {
        config: VadConfig,
        /// Whether we are currently in a speech segment.
        in_speech: bool,
        /// Timestamp (ms) when current speech started.
        speech_start_ms: u64,
        /// Running timestamp counter (ms).
        current_ms: u64,
        /// Consecutive silence frames counter.
        silence_frames: u32,
        /// Moving average of energy values (exponential).
        moving_avg_energy: f32,
        /// Smoothing factor for moving average (0..1).
        alpha: f32,
    }

    impl VadDetector {
        /// Create a new VAD detector with the given configuration.
        pub fn new(config: VadConfig) -> Self {
            Self {
                config,
                in_speech: false,
                speech_start_ms: 0,
                current_ms: 0,
                silence_frames: 0,
                moving_avg_energy: 0.0,
                alpha: 0.3,
            }
        }

        /// Create a VAD detector with default configuration.
        pub fn with_defaults() -> Self {
            Self::new(VadConfig::default())
        }

        /// Process a single frame of PCM16 audio samples.
        ///
        /// Returns a `VadEvent` indicating whether speech started, ended, or
        /// the frame is silence.
        pub fn process_frame(&mut self, frame: &[i16]) -> VadEvent {
            let rms = Self::compute_rms(frame);
            self.moving_avg_energy =
                self.alpha * rms + (1.0 - self.alpha) * self.moving_avg_energy;

            let is_speech = self.moving_avg_energy > self.config.energy_threshold;
            let frame_duration_ms = self.config.frame_size_ms as u64;

            let event = if is_speech {
                self.silence_frames = 0;
                if !self.in_speech {
                    // Transition: silence -> speech
                    self.in_speech = true;
                    self.speech_start_ms = self.current_ms;
                    VadEvent::SpeechStart {
                        timestamp_ms: self.current_ms,
                    }
                } else {
                    // Continuing speech — report silence (no boundary event)
                    VadEvent::Silence
                }
            } else {
                // Silent frame
                if self.in_speech {
                    self.silence_frames += 1;
                    let silence_ms =
                        self.silence_frames as u64 * self.config.frame_size_ms as u64;
                    if silence_ms >= self.config.silence_duration_ms as u64 {
                        // Enough silence to end speech segment
                        let duration = self.current_ms - self.speech_start_ms;
                        self.in_speech = false;
                        self.silence_frames = 0;
                        if duration >= self.config.min_speech_duration_ms as u64 {
                            VadEvent::SpeechEnd {
                                timestamp_ms: self.current_ms,
                                duration_ms: duration,
                            }
                        } else {
                            // Too short — discard as noise
                            VadEvent::Silence
                        }
                    } else {
                        VadEvent::Silence
                    }
                } else {
                    VadEvent::Silence
                }
            };

            self.current_ms += frame_duration_ms;
            event
        }

        /// Compute RMS energy of a PCM16 frame, normalized to [0, 1].
        fn compute_rms(frame: &[i16]) -> f32 {
            if frame.is_empty() {
                return 0.0;
            }
            let sum_sq: f64 = frame
                .iter()
                .map(|&s| {
                    let normalized = s as f64 / i16::MAX as f64;
                    normalized * normalized
                })
                .sum();
            (sum_sq / frame.len() as f64).sqrt() as f32
        }

        /// Reset the detector state.
        pub fn reset(&mut self) {
            self.in_speech = false;
            self.speech_start_ms = 0;
            self.current_ms = 0;
            self.silence_frames = 0;
            self.moving_avg_energy = 0.0;
        }

        /// Returns true if the detector currently considers speech is active.
        pub fn is_in_speech(&self) -> bool {
            self.in_speech
        }

        /// Current timestamp counter in milliseconds.
        pub fn current_timestamp_ms(&self) -> u64 {
            self.current_ms
        }
    }

    // ========================================================================
    // 3.2 — Interruption Handling
    // ========================================================================

    /// Policy for handling user interruptions while the agent is speaking.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq)]
    pub enum InterruptionPolicy {
        /// Stop speaking instantly when user interrupts.
        Immediate,
        /// Finish the current sentence before yielding.
        EndSentence,
        /// Never allow interruptions — agent speaks to completion.
        Never,
    }

    impl fmt::Display for InterruptionPolicy {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                InterruptionPolicy::Immediate => write!(f, "immediate"),
                InterruptionPolicy::EndSentence => write!(f, "end_sentence"),
                InterruptionPolicy::Never => write!(f, "never"),
            }
        }
    }

    /// An interruption event triggered when the user speaks while the agent is responding.
    #[derive(Debug, Clone)]
    pub struct InterruptionEvent {
        /// Timestamp when the interruption was detected (ms).
        pub timestamp_ms: u64,
        /// Partial agent response that was spoken before interruption.
        pub partial_response: String,
        /// Which policy was applied to handle this interruption.
        pub policy_applied: InterruptionPolicy,
    }

    // ========================================================================
    // 3.3 — Conversation Turn Management
    // ========================================================================

    /// Who is speaking in a conversation turn.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq)]
    pub enum TurnSpeaker {
        /// The human user.
        User,
        /// The AI agent.
        Agent,
    }

    impl fmt::Display for TurnSpeaker {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TurnSpeaker::User => write!(f, "user"),
                TurnSpeaker::Agent => write!(f, "agent"),
            }
        }
    }

    /// A single turn in a voice conversation.
    #[derive(Debug, Clone)]
    pub struct ConversationTurn {
        /// Who spoke during this turn.
        pub speaker: TurnSpeaker,
        /// Transcribed text of what was said.
        pub transcript: String,
        /// Duration of the audio for this turn (ms).
        pub audio_duration_ms: u64,
        /// When this turn occurred.
        pub timestamp: chrono::DateTime<chrono::Utc>,
        /// Sequential turn number (1-based).
        pub turn_number: usize,
    }

    /// Policy governing how turns are managed.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq)]
    pub enum TurnPolicy {
        /// Speakers must strictly alternate (user, agent, user, agent, ...).
        StrictAlternating,
        /// Natural overlap is allowed — same speaker can have consecutive turns.
        NaturalOverlap,
        /// Push-to-talk: turns are explicitly signalled.
        PushToTalk,
    }

    impl fmt::Display for TurnPolicy {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TurnPolicy::StrictAlternating => write!(f, "strict_alternating"),
                TurnPolicy::NaturalOverlap => write!(f, "natural_overlap"),
                TurnPolicy::PushToTalk => write!(f, "push_to_talk"),
            }
        }
    }

    /// Manages conversation turns, enforcing turn policies and tracking history.
    #[derive(Debug, Clone)]
    pub struct TurnManager {
        turns: Vec<ConversationTurn>,
        current_turn: Option<TurnSpeaker>,
        policy: TurnPolicy,
    }

    impl TurnManager {
        /// Create a new turn manager with the given policy.
        pub fn new(policy: TurnPolicy) -> Self {
            Self {
                turns: Vec::new(),
                current_turn: None,
                policy,
            }
        }

        /// Start a new turn for the given speaker.
        ///
        /// Under `StrictAlternating` policy, returns an error if the same speaker
        /// attempts two consecutive turns.
        pub fn start_turn(&mut self, speaker: TurnSpeaker) -> Result<(), AiError> {
            if self.policy == TurnPolicy::StrictAlternating {
                if let Some(ref current) = self.current_turn {
                    if *current == speaker {
                        return Err(VoiceAgentError::InvalidSessionState {
                            current: format!("{} speaking", current),
                            attempted: format!("{} start_turn", speaker),
                        }
                        .into());
                    }
                }
            }
            self.current_turn = Some(speaker);
            Ok(())
        }

        /// End the current turn, recording the transcript and duration.
        ///
        /// Returns an error if no turn is active.
        pub fn end_turn(
            &mut self,
            transcript: String,
            duration_ms: u64,
        ) -> Result<&ConversationTurn, AiError> {
            let speaker = self.current_turn.take().ok_or_else(|| {
                VoiceAgentError::InvalidSessionState {
                    current: "no active turn".to_string(),
                    attempted: "end_turn".to_string(),
                }
            })?;

            let turn_number = self.turns.len() + 1;
            let turn = ConversationTurn {
                speaker,
                transcript,
                audio_duration_ms: duration_ms,
                timestamp: chrono::Utc::now(),
                turn_number,
            };
            self.turns.push(turn);
            Ok(self.turns.last().expect("just pushed"))
        }

        /// Returns who is currently speaking, if anyone.
        pub fn current_speaker(&self) -> Option<&TurnSpeaker> {
            self.current_turn.as_ref()
        }

        /// Total number of completed turns.
        pub fn total_turns(&self) -> usize {
            self.turns.len()
        }

        /// Full conversation history.
        pub fn get_history(&self) -> &[ConversationTurn] {
            &self.turns
        }

        /// Get the last `n` turns (or fewer if history is shorter).
        pub fn get_last_n(&self, n: usize) -> Vec<&ConversationTurn> {
            let start = self.turns.len().saturating_sub(n);
            self.turns[start..].iter().collect()
        }

        /// Total audio duration across all turns (ms).
        pub fn total_duration_ms(&self) -> u64 {
            self.turns.iter().map(|t| t.audio_duration_ms).sum()
        }

        /// Get the turn policy.
        pub fn policy(&self) -> &TurnPolicy {
            &self.policy
        }
    }

    // ========================================================================
    // 3.1 — Voice Session
    // ========================================================================

    /// State machine for a voice session.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq)]
    pub enum VoiceSessionState {
        /// Session created but not active.
        Idle,
        /// Actively listening to user audio.
        Listening,
        /// Processing audio (STT / LLM inference).
        Processing,
        /// Agent is speaking (TTS playback).
        Speaking,
        /// User interrupted agent speech.
        Interrupted,
        /// Session has been closed.
        Closed,
    }

    impl fmt::Display for VoiceSessionState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                VoiceSessionState::Idle => write!(f, "idle"),
                VoiceSessionState::Listening => write!(f, "listening"),
                VoiceSessionState::Processing => write!(f, "processing"),
                VoiceSessionState::Speaking => write!(f, "speaking"),
                VoiceSessionState::Interrupted => write!(f, "interrupted"),
                VoiceSessionState::Closed => write!(f, "closed"),
            }
        }
    }

    impl VoiceSessionState {
        /// Check whether a transition from the current state to `target` is valid.
        pub fn can_transition_to(&self, target: &VoiceSessionState) -> bool {
            matches!(
                (self, target),
                (VoiceSessionState::Idle, VoiceSessionState::Listening)
                    | (VoiceSessionState::Listening, VoiceSessionState::Processing)
                    | (VoiceSessionState::Processing, VoiceSessionState::Speaking)
                    | (VoiceSessionState::Processing, VoiceSessionState::Listening)
                    | (VoiceSessionState::Speaking, VoiceSessionState::Listening)
                    | (VoiceSessionState::Speaking, VoiceSessionState::Interrupted)
                    | (VoiceSessionState::Interrupted, VoiceSessionState::Listening)
                    // Any state can transition to Closed
                    | (VoiceSessionState::Idle, VoiceSessionState::Closed)
                    | (VoiceSessionState::Listening, VoiceSessionState::Closed)
                    | (VoiceSessionState::Processing, VoiceSessionState::Closed)
                    | (VoiceSessionState::Speaking, VoiceSessionState::Closed)
                    | (VoiceSessionState::Interrupted, VoiceSessionState::Closed)
            )
        }
    }

    /// A voice conversation session.
    #[derive(Debug, Clone)]
    pub struct VoiceSession {
        /// Unique session identifier.
        pub session_id: String,
        /// Current state of the session.
        state: VoiceSessionState,
        /// Conversation history.
        history: Vec<ConversationTurn>,
        /// When the session was created.
        pub created_at: chrono::DateTime<chrono::Utc>,
        /// Configuration snapshot for this session.
        config: VoiceAgentConfig,
        /// Turn manager for this session.
        turn_manager: TurnManager,
        /// Interruption events that occurred during this session.
        interruptions: Vec<InterruptionEvent>,
    }

    impl VoiceSession {
        /// Create a new session with a generated ID.
        fn new(config: VoiceAgentConfig) -> Self {
            let session_id = format!("vs-{}", Self::generate_id());
            let turn_policy = config.turn_policy.clone();
            Self {
                session_id,
                state: VoiceSessionState::Idle,
                history: Vec::new(),
                created_at: chrono::Utc::now(),
                config,
                turn_manager: TurnManager::new(turn_policy),
                interruptions: Vec::new(),
            }
        }

        /// Generate a simple unique ID using timestamp + counter.
        fn generate_id() -> String {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let count = COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("{:x}-{:04x}", ts, count)
        }

        /// Get the current session state.
        pub fn state(&self) -> &VoiceSessionState {
            &self.state
        }

        /// Attempt a state transition. Returns an error if the transition is invalid.
        pub fn transition_to(&mut self, target: VoiceSessionState) -> Result<(), AiError> {
            if !self.state.can_transition_to(&target) {
                return Err(VoiceAgentError::InvalidSessionState {
                    current: self.state.to_string(),
                    attempted: target.to_string(),
                }
                .into());
            }
            self.state = target;
            Ok(())
        }

        /// Get the conversation history.
        pub fn history(&self) -> &[ConversationTurn] {
            &self.history
        }

        /// Get a mutable reference to the turn manager.
        pub fn turn_manager_mut(&mut self) -> &mut TurnManager {
            &mut self.turn_manager
        }

        /// Get a reference to the turn manager.
        pub fn turn_manager(&self) -> &TurnManager {
            &self.turn_manager
        }

        /// Record an interruption event.
        pub fn record_interruption(&mut self, event: InterruptionEvent) {
            self.interruptions.push(event);
        }

        /// Get all interruption events.
        pub fn interruptions(&self) -> &[InterruptionEvent] {
            &self.interruptions
        }

        /// Push a completed turn into the session history.
        pub fn push_turn(&mut self, turn: ConversationTurn) {
            self.history.push(turn);
        }

        /// Get the session config.
        pub fn config(&self) -> &VoiceAgentConfig {
            &self.config
        }
    }

    // ========================================================================
    // 3.1 — VoiceAgentConfig
    // ========================================================================

    /// Configuration for a voice agent instance.
    #[non_exhaustive]
    #[derive(Debug, Clone)]
    pub struct VoiceAgentConfig {
        /// STT model identifier (e.g. "whisper-1", "google-chirp").
        pub model_stt: String,
        /// TTS model identifier (e.g. "tts-1", "google-wavenet").
        pub model_tts: String,
        /// Voice Activity Detection configuration.
        pub vad_config: VadConfig,
        /// How interruptions should be handled.
        pub interruption_policy: InterruptionPolicy,
        /// Voice identifier for TTS (e.g. "alloy", "nova").
        pub voice_id: String,
        /// Audio sample rate in Hz. Default: 16000.
        pub sample_rate: u32,
        /// Chunk size for streaming in milliseconds. Default: 20.
        pub chunk_size_ms: u32,
        /// Turn management policy.
        pub turn_policy: TurnPolicy,
    }

    impl Default for VoiceAgentConfig {
        fn default() -> Self {
            Self {
                model_stt: "whisper-1".to_string(),
                model_tts: "tts-1".to_string(),
                vad_config: VadConfig::default(),
                interruption_policy: InterruptionPolicy::Immediate,
                voice_id: "alloy".to_string(),
                sample_rate: 16000,
                chunk_size_ms: 20,
                turn_policy: TurnPolicy::NaturalOverlap,
            }
        }
    }

    impl VoiceAgentConfig {
        /// Validate the configuration. Returns an error if values are out of range.
        pub fn validate(&self) -> Result<(), AiError> {
            if self.model_stt.is_empty() {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "model_stt cannot be empty".to_string(),
                }
                .into());
            }
            if self.model_tts.is_empty() {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "model_tts cannot be empty".to_string(),
                }
                .into());
            }
            if self.sample_rate == 0 {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "sample_rate must be > 0".to_string(),
                }
                .into());
            }
            if self.chunk_size_ms == 0 {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "chunk_size_ms must be > 0".to_string(),
                }
                .into());
            }
            if self.vad_config.energy_threshold < 0.0
                || self.vad_config.energy_threshold > 1.0
            {
                return Err(VoiceAgentError::VadError {
                    reason: "energy_threshold must be in [0.0, 1.0]".to_string(),
                }
                .into());
            }
            Ok(())
        }
    }

    // ========================================================================
    // 3.1 — VoiceAgent
    // ========================================================================

    /// Main voice agent — orchestrates the STT -> LLM -> TTS pipeline.
    ///
    /// The `VoiceAgent` manages sessions, processes incoming audio through VAD,
    /// sends detected speech to STT, feeds transcripts through the LLM, and
    /// synthesises responses via TTS.
    #[derive(Debug)]
    pub struct VoiceAgent {
        config: VoiceAgentConfig,
        vad: VadDetector,
        /// Audio buffer for accumulating frames before STT.
        audio_buffer: Vec<i16>,
        /// Active sessions (session_id -> session).
        sessions: std::collections::HashMap<String, VoiceSession>,
    }

    impl VoiceAgent {
        /// Create a new voice agent with the given configuration.
        ///
        /// Returns an error if the configuration is invalid.
        pub fn new(config: VoiceAgentConfig) -> Result<Self, AiError> {
            config.validate()?;
            let vad = VadDetector::new(config.vad_config.clone());
            Ok(Self {
                config,
                vad,
                audio_buffer: Vec::new(),
                sessions: std::collections::HashMap::new(),
            })
        }

        /// Create a voice agent with default configuration.
        pub fn with_defaults() -> Result<Self, AiError> {
            Self::new(VoiceAgentConfig::default())
        }

        /// Start a new voice session. Returns the session.
        pub fn start_session(&mut self) -> Result<VoiceSession, AiError> {
            let session = VoiceSession::new(self.config.clone());
            let id = session.session_id.clone();
            self.sessions.insert(id, session.clone());
            Ok(session)
        }

        /// Process an incoming audio chunk for a session.
        ///
        /// Runs VAD on the chunk, and when a speech segment ends, simulates
        /// the STT -> LLM -> TTS pipeline and returns a response audio chunk.
        /// Returns `None` if no complete response is ready yet.
        pub fn process_audio(
            &mut self,
            session: &mut VoiceSession,
            chunk: &AudioChunk,
        ) -> Result<Option<AudioChunk>, AiError> {
            // Validate session state — must be Idle or Listening
            match session.state() {
                VoiceSessionState::Idle => {
                    session.transition_to(VoiceSessionState::Listening)?;
                }
                VoiceSessionState::Listening => { /* already listening */ }
                VoiceSessionState::Speaking => {
                    // User is interrupting
                    return self.handle_interruption(session, chunk);
                }
                other => {
                    return Err(VoiceAgentError::InvalidSessionState {
                        current: other.to_string(),
                        attempted: "process_audio".to_string(),
                    }
                    .into());
                }
            }

            // Convert bytes to PCM16 samples for VAD
            let samples = Self::bytes_to_samples(&chunk.bytes);

            // Feed into VAD frame by frame
            let frame_samples = (self.config.sample_rate as usize
                * self.config.chunk_size_ms as usize)
                / 1000;
            let frame_size = if frame_samples > 0 {
                frame_samples
            } else {
                samples.len().max(1)
            };

            let mut response = None;

            for frame_start in (0..samples.len()).step_by(frame_size) {
                let frame_end = (frame_start + frame_size).min(samples.len());
                let frame = &samples[frame_start..frame_end];
                let event = self.vad.process_frame(frame);

                match event {
                    VadEvent::SpeechStart { .. } => {
                        self.audio_buffer.clear();
                        self.audio_buffer.extend_from_slice(frame);
                    }
                    VadEvent::Silence if self.vad.is_in_speech() => {
                        self.audio_buffer.extend_from_slice(frame);
                    }
                    VadEvent::SpeechEnd { duration_ms, .. } => {
                        self.audio_buffer.extend_from_slice(frame);

                        // Transition to Processing
                        session.transition_to(VoiceSessionState::Processing)?;

                        // Simulate STT: create a transcript from audio length
                        let transcript = format!(
                            "[user speech: {}ms, {} samples]",
                            duration_ms,
                            self.audio_buffer.len()
                        );

                        // Record user turn
                        session.turn_manager_mut().start_turn(TurnSpeaker::User)?;
                        if let Ok(turn) = session
                            .turn_manager_mut()
                            .end_turn(transcript.clone(), duration_ms)
                        {
                            let turn_clone = ConversationTurn {
                                speaker: turn.speaker.clone(),
                                transcript: turn.transcript.clone(),
                                audio_duration_ms: turn.audio_duration_ms,
                                timestamp: turn.timestamp,
                                turn_number: turn.turn_number,
                            };
                            session.push_turn(turn_clone);
                        }

                        // Simulate LLM response
                        let agent_response =
                            format!("[agent response to: {}]", transcript);

                        // Record agent turn
                        if session.turn_manager().policy() != &TurnPolicy::StrictAlternating
                            || session.turn_manager().current_speaker()
                                != Some(&TurnSpeaker::Agent)
                        {
                            let _ =
                                session.turn_manager_mut().start_turn(TurnSpeaker::Agent);
                        }
                        let response_duration = duration_ms / 2; // simulated
                        if let Ok(turn) = session.turn_manager_mut().end_turn(
                            agent_response.clone(),
                            response_duration,
                        ) {
                            let turn_clone = ConversationTurn {
                                speaker: turn.speaker.clone(),
                                transcript: turn.transcript.clone(),
                                audio_duration_ms: turn.audio_duration_ms,
                                timestamp: turn.timestamp,
                                turn_number: turn.turn_number,
                            };
                            session.push_turn(turn_clone);
                        }

                        // Simulate TTS: create response audio
                        let response_bytes =
                            agent_response.as_bytes().to_vec();
                        let response_chunk = AudioChunk::new(
                            response_bytes,
                            self.config.sample_rate,
                            1,
                            AudioFormat::Pcm16,
                        );

                        // Transition to Speaking, then back to Listening
                        session.transition_to(VoiceSessionState::Speaking)?;
                        session.transition_to(VoiceSessionState::Listening)?;

                        self.audio_buffer.clear();
                        response = Some(response_chunk);
                    }
                    _ => {
                        // Silence, not in speech — nothing to do
                    }
                }
            }

            Ok(response)
        }

        /// Handle an interruption while the agent is speaking.
        fn handle_interruption(
            &self,
            session: &mut VoiceSession,
            _chunk: &AudioChunk,
        ) -> Result<Option<AudioChunk>, AiError> {
            match &self.config.interruption_policy {
                InterruptionPolicy::Immediate => {
                    session.transition_to(VoiceSessionState::Interrupted)?;
                    session.record_interruption(InterruptionEvent {
                        timestamp_ms: self.vad.current_timestamp_ms(),
                        partial_response: String::new(),
                        policy_applied: InterruptionPolicy::Immediate,
                    });
                    session.transition_to(VoiceSessionState::Listening)?;
                    Ok(None)
                }
                InterruptionPolicy::EndSentence => {
                    // In a real implementation, we would finish the current sentence.
                    // For now, treat similarly to Immediate but record the policy.
                    session.transition_to(VoiceSessionState::Interrupted)?;
                    session.record_interruption(InterruptionEvent {
                        timestamp_ms: self.vad.current_timestamp_ms(),
                        partial_response: String::new(),
                        policy_applied: InterruptionPolicy::EndSentence,
                    });
                    session.transition_to(VoiceSessionState::Listening)?;
                    Ok(None)
                }
                InterruptionPolicy::Never => {
                    // Ignore the interruption — remain speaking
                    Ok(None)
                }
            }
        }

        /// End a session gracefully.
        pub fn end_session(&mut self, session: &mut VoiceSession) -> Result<(), AiError> {
            session.transition_to(VoiceSessionState::Closed)?;
            self.sessions.remove(&session.session_id);
            self.audio_buffer.clear();
            self.vad.reset();
            Ok(())
        }

        /// Get the transcript history for a session.
        pub fn get_transcript<'a>(&self, session: &'a VoiceSession) -> Vec<&'a ConversationTurn> {
            session.history().iter().collect()
        }

        /// Get the voice agent configuration.
        pub fn config(&self) -> &VoiceAgentConfig {
            &self.config
        }

        /// Get the current VAD detector state.
        pub fn vad(&self) -> &VadDetector {
            &self.vad
        }

        /// Number of active sessions.
        pub fn active_session_count(&self) -> usize {
            self.sessions.len()
        }

        /// Convert raw bytes to PCM16 samples (little-endian).
        fn bytes_to_samples(bytes: &[u8]) -> Vec<i16> {
            bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect()
        }
    }

    // ========================================================================
    // 9.1 — WebRTC Voice Transport
    // ========================================================================

    #[cfg(feature = "webrtc")]
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WebRtcConfig {
        pub stun_servers: Vec<String>,
        pub turn_servers: Vec<TurnServer>,
        pub audio_codec: WebRtcAudioCodec,
        pub sample_rate: u32,
        pub enable_dtls: bool,
        pub ice_timeout_ms: u64,
    }

    #[cfg(feature = "webrtc")]
    impl Default for WebRtcConfig {
        fn default() -> Self {
            Self {
                stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
                turn_servers: Vec::new(),
                audio_codec: WebRtcAudioCodec::Opus,
                sample_rate: 48000,
                enable_dtls: true,
                ice_timeout_ms: 5000,
            }
        }
    }

    #[cfg(feature = "webrtc")]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TurnServer {
        pub url: String,
        pub username: String,
        pub credential: String,
    }

    #[cfg(feature = "webrtc")]
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum WebRtcAudioCodec {
        Opus,
        Pcm16,
        G711,
    }

    #[cfg(feature = "webrtc")]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SdpOffer {
        pub sdp: String,
        pub session_id: String,
    }

    #[cfg(feature = "webrtc")]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SdpAnswer {
        pub sdp: String,
        pub session_id: String,
    }

    #[cfg(feature = "webrtc")]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WebRtcIceCandidate {
        pub candidate: String,
        pub sdp_mid: Option<String>,
        pub sdp_m_line_index: Option<u32>,
        pub priority: u32,
        pub candidate_type: IceCandidateType,
    }

    #[cfg(feature = "webrtc")]
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum IceCandidateType {
        Host,
        ServerReflexive,
        PeerReflexive,
        Relay,
    }

    #[cfg(feature = "webrtc")]
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RtpStreamConfig {
        pub ssrc: u32,
        pub payload_type: u8,
        pub clock_rate: u32,
        pub jitter_buffer_ms: u32,
    }

    #[cfg(feature = "webrtc")]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WebRtcSessionStats {
        pub latency_ms: f64,
        pub packet_loss_pct: f64,
        pub jitter_ms: f64,
        pub bytes_sent: u64,
        pub bytes_received: u64,
    }

    #[cfg(feature = "webrtc")]
    impl Default for WebRtcSessionStats {
        fn default() -> Self {
            Self {
                latency_ms: 0.0,
                packet_loss_pct: 0.0,
                jitter_ms: 0.0,
                bytes_sent: 0,
                bytes_received: 0,
            }
        }
    }

    #[cfg(feature = "webrtc")]
    pub struct WebRtcTransport {
        config: WebRtcConfig,
        local_sdp: Option<SdpOffer>,
        remote_sdp: Option<SdpAnswer>,
        ice_candidates: Vec<WebRtcIceCandidate>,
        connected: bool,
        stats: WebRtcSessionStats,
    }

    #[cfg(feature = "webrtc")]
    impl WebRtcTransport {
        /// Create a new WebRTC transport with the given configuration.
        pub fn new(config: WebRtcConfig) -> Self {
            Self {
                config,
                local_sdp: None,
                remote_sdp: None,
                ice_candidates: Vec::new(),
                connected: false,
                stats: WebRtcSessionStats::default(),
            }
        }

        /// Create a WebRTC transport with default configuration.
        pub fn with_defaults() -> Self {
            Self::new(WebRtcConfig::default())
        }

        /// Generate a mock SDP offer. In production, this would delegate to a
        /// real WebRTC stack (e.g. libwebrtc). The SDP string follows the
        /// general structure of a real offer for testing purposes.
        pub fn create_offer(&mut self) -> SdpOffer {
            let session_id = format!("webrtc-{}", {
                use std::sync::atomic::{AtomicU64, Ordering};
                static COUNTER: AtomicU64 = AtomicU64::new(0);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                let count = COUNTER.fetch_add(1, Ordering::Relaxed);
                format!("{:x}-{:04x}", ts, count)
            });

            let codec_name = match &self.config.audio_codec {
                WebRtcAudioCodec::Opus => "opus",
                WebRtcAudioCodec::Pcm16 => "PCMU",
                WebRtcAudioCodec::G711 => "PCMA",
            };

            let sdp = format!(
                "v=0\r\n\
                 o=- {session_id} 2 IN IP4 127.0.0.1\r\n\
                 s=-\r\n\
                 t=0 0\r\n\
                 m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
                 a=rtpmap:111 {codec}/{rate}/2\r\n\
                 a=fmtp:111 minptime=10;useinbandfec=1\r\n\
                 a=sendrecv\r\n",
                session_id = session_id,
                codec = codec_name,
                rate = self.config.sample_rate,
            );

            let offer = SdpOffer {
                sdp,
                session_id: session_id.clone(),
            };
            self.local_sdp = Some(offer.clone());
            offer
        }

        /// Set the remote SDP answer received from the peer.
        pub fn set_remote_answer(&mut self, answer: SdpAnswer) {
            self.remote_sdp = Some(answer);
        }

        /// Add an ICE candidate gathered during the negotiation process.
        pub fn add_ice_candidate(&mut self, candidate: WebRtcIceCandidate) {
            self.ice_candidates.push(candidate);
        }

        /// Get all ICE candidates gathered so far.
        pub fn ice_candidates(&self) -> &[WebRtcIceCandidate] {
            &self.ice_candidates
        }

        /// Attempt to establish the WebRTC connection.
        ///
        /// Requires both local SDP offer and remote SDP answer to be set.
        /// In production this would perform the full ICE connectivity check
        /// and DTLS handshake. Here we simulate a successful connection.
        pub fn connect(&mut self) -> Result<(), VoiceAgentError> {
            if self.local_sdp.is_none() {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "No local SDP offer — call create_offer() first".to_string(),
                });
            }
            if self.remote_sdp.is_none() {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "No remote SDP answer — call set_remote_answer() first".to_string(),
                });
            }
            self.connected = true;
            // Simulate initial stats from a successful connection
            self.stats = WebRtcSessionStats {
                latency_ms: 25.0,
                packet_loss_pct: 0.0,
                jitter_ms: 2.0,
                bytes_sent: 0,
                bytes_received: 0,
            };
            Ok(())
        }

        /// Disconnect the transport, resetting connection state.
        pub fn disconnect(&mut self) {
            self.connected = false;
        }

        /// Whether the transport is currently connected.
        pub fn is_connected(&self) -> bool {
            self.connected
        }

        /// Get the current session statistics.
        pub fn stats(&self) -> &WebRtcSessionStats {
            &self.stats
        }

        /// Get the transport configuration.
        pub fn config(&self) -> &WebRtcConfig {
            &self.config
        }
    }

    #[cfg(feature = "webrtc")]
    impl VoiceTransport for WebRtcTransport {
        fn send_audio(&self, chunk: &AudioChunk) -> Result<(), AiError> {
            if !self.connected {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "WebRTC transport is not connected".to_string(),
                }
                .into());
            }
            // In production, this would encode and send via RTP.
            // For simulation, we just validate the chunk is non-empty.
            if chunk.is_empty() {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "Cannot send empty audio chunk".to_string(),
                }
                .into());
            }
            Ok(())
        }

        fn receive_audio(&mut self) -> Result<Option<AudioChunk>, AiError> {
            if !self.connected {
                return Err(VoiceAgentError::StreamFailed {
                    reason: "WebRTC transport is not connected".to_string(),
                }
                .into());
            }
            // In production, this would dequeue from a jitter buffer.
            // Simulated: no data available.
            Ok(None)
        }

        fn close(&mut self) -> Result<(), AiError> {
            self.disconnect();
            self.local_sdp = None;
            self.remote_sdp = None;
            self.ice_candidates.clear();
            Ok(())
        }
    }

    // ========================================================================
    // 9.2 — Speech-to-Speech Pipeline
    // ========================================================================

    /// Describes the audio I/O capabilities of a model.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum AudioModelCapability {
        /// Model only accepts and produces text.
        TextOnly,
        /// Model can accept audio input but produces text output.
        AudioInput,
        /// Model produces audio output but only accepts text input.
        AudioOutput,
        /// Model can accept audio input and produce audio output (native S2S).
        AudioInputOutput,
    }

    /// Configuration for a Speech-to-Speech pipeline.
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct S2SConfig {
        /// Model identifier for the S2S model.
        pub model_id: String,
        /// Expected input audio format.
        pub input_format: AudioFormat,
        /// Desired output audio format.
        pub output_format: AudioFormat,
        /// Optional voice identifier for the output voice.
        pub voice_id: Option<String>,
        /// The audio capability of the model.
        pub capability: AudioModelCapability,
    }

    /// Fallback strategy when the model does not support direct audio I/O.
    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum S2SFallbackMode {
        /// Standard fallback: STT -> LLM -> TTS
        SttLlmTts,
        /// No fallback — return an error if model does not support direct S2S.
        Disabled,
    }

    /// A transcript entry recorded by the S2S pipeline for each processed audio.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct S2STranscript {
        /// Transcribed input text (if available via STT or model transcription).
        pub input_text: Option<String>,
        /// Generated output text (if available via model or TTS input).
        pub output_text: Option<String>,
        /// The model that processed this interaction.
        pub model_id: String,
        /// Whether the fallback STT->LLM->TTS pipeline was used instead of direct S2S.
        pub used_fallback: bool,
        /// End-to-end latency in milliseconds.
        pub latency_ms: u64,
        /// Unix timestamp in milliseconds when this was processed.
        pub timestamp: u64,
    }

    /// Speech-to-Speech pipeline that processes audio input and produces audio output.
    ///
    /// When the model supports `AudioInputOutput`, audio is processed directly (native S2S).
    /// Otherwise, depending on the fallback mode, the pipeline falls back to the
    /// traditional STT -> LLM -> TTS chain.
    pub struct SpeechToSpeechPipeline {
        config: S2SConfig,
        fallback: S2SFallbackMode,
        transcripts: Vec<S2STranscript>,
    }

    impl SpeechToSpeechPipeline {
        /// Create a new S2S pipeline with the given configuration and fallback mode.
        pub fn new(config: S2SConfig, fallback: S2SFallbackMode) -> Self {
            Self {
                config,
                fallback,
                transcripts: Vec::new(),
            }
        }

        /// Returns true if the configured model supports direct audio-to-audio processing.
        pub fn supports_direct_audio(&self) -> bool {
            self.config.capability == AudioModelCapability::AudioInputOutput
        }

        /// Process an audio input chunk and return an audio output chunk.
        ///
        /// If the model supports `AudioInputOutput`, the audio is processed directly
        /// (simulated here). Otherwise, falls back to STT -> LLM -> TTS if fallback
        /// mode is `SttLlmTts`, or returns an error if fallback is `Disabled`.
        pub fn process_audio(
            &mut self,
            input: &AudioChunk,
        ) -> Result<AudioChunk, VoiceAgentError> {
            let start = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let (output, used_fallback, input_text, output_text) =
                if self.supports_direct_audio() {
                    // Native S2S: simulate direct audio processing
                    // In production, this sends audio to the model and receives audio back.
                    let response_bytes = vec![0u8; input.bytes.len()];
                    let output_chunk = AudioChunk::new(
                        response_bytes,
                        input.sample_rate,
                        input.channels,
                        self.config.output_format.clone(),
                    );
                    (
                        output_chunk,
                        false,
                        Some("[direct audio input]".to_string()),
                        Some("[direct audio output]".to_string()),
                    )
                } else {
                    // Model does not support direct S2S
                    match &self.fallback {
                        S2SFallbackMode::SttLlmTts => {
                            // Simulate STT -> LLM -> TTS fallback
                            let stt_text = format!(
                                "[stt: {} bytes, {}Hz]",
                                input.bytes.len(),
                                input.sample_rate
                            );
                            let llm_response = format!("[llm response to: {}]", stt_text);
                            // Simulate TTS output
                            let tts_bytes = llm_response.as_bytes().to_vec();
                            let output_chunk = AudioChunk::new(
                                tts_bytes,
                                input.sample_rate,
                                input.channels,
                                self.config.output_format.clone(),
                            );
                            (
                                output_chunk,
                                true,
                                Some(stt_text),
                                Some(llm_response),
                            )
                        }
                        S2SFallbackMode::Disabled => {
                            return Err(VoiceAgentError::StreamFailed {
                                reason: format!(
                                    "Model '{}' does not support AudioInputOutput and fallback is disabled",
                                    self.config.model_id
                                ),
                            });
                        }
                    }
                };

            let end = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let transcript = S2STranscript {
                input_text,
                output_text,
                model_id: self.config.model_id.clone(),
                used_fallback,
                latency_ms: end.saturating_sub(start),
                timestamp: end,
            };
            self.transcripts.push(transcript);

            Ok(output)
        }

        /// Get all recorded transcripts.
        pub fn transcripts(&self) -> &[S2STranscript] {
            &self.transcripts
        }

        /// Get the pipeline configuration.
        pub fn config(&self) -> &S2SConfig {
            &self.config
        }

        /// Get the current fallback mode.
        pub fn fallback_mode(&self) -> &S2SFallbackMode {
            &self.fallback
        }

        /// Clear all recorded transcripts.
        pub fn clear_transcripts(&mut self) {
            self.transcripts.clear();
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        // ----------------------------------------------------------------
        // AudioFormat tests
        // ----------------------------------------------------------------

        #[test]
        fn test_audio_format_display() {
            assert_eq!(AudioFormat::Pcm16.to_string(), "pcm16");
            assert_eq!(AudioFormat::Wav.to_string(), "wav");
            assert_eq!(AudioFormat::Ogg.to_string(), "ogg");
            assert_eq!(AudioFormat::Mp3.to_string(), "mp3");
        }

        #[test]
        fn test_audio_format_mime_type() {
            assert_eq!(AudioFormat::Pcm16.mime_type(), "audio/pcm");
            assert_eq!(AudioFormat::Wav.mime_type(), "audio/wav");
            assert_eq!(AudioFormat::Ogg.mime_type(), "audio/ogg");
            assert_eq!(AudioFormat::Mp3.mime_type(), "audio/mpeg");
        }

        #[test]
        fn test_audio_format_extension() {
            assert_eq!(AudioFormat::Pcm16.extension(), "pcm");
            assert_eq!(AudioFormat::Wav.extension(), "wav");
            assert_eq!(AudioFormat::Ogg.extension(), "ogg");
            assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        }

        #[test]
        fn test_audio_format_equality() {
            assert_eq!(AudioFormat::Pcm16, AudioFormat::Pcm16);
            assert_ne!(AudioFormat::Pcm16, AudioFormat::Wav);
        }

        // ----------------------------------------------------------------
        // AudioChunk tests
        // ----------------------------------------------------------------

        #[test]
        fn test_audio_chunk_new() {
            let chunk = AudioChunk::new(vec![0u8; 3200], 16000, 1, AudioFormat::Pcm16);
            assert_eq!(chunk.sample_rate, 16000);
            assert_eq!(chunk.channels, 1);
            assert_eq!(chunk.format, AudioFormat::Pcm16);
            assert_eq!(chunk.bytes.len(), 3200);
        }

        #[test]
        fn test_audio_chunk_duration_ms() {
            // 16000 Hz, mono, PCM16 = 2 bytes per sample
            // 3200 bytes = 1600 samples = 100ms at 16kHz
            let chunk = AudioChunk::new(vec![0u8; 3200], 16000, 1, AudioFormat::Pcm16);
            assert_eq!(chunk.duration_ms(), 100);
        }

        #[test]
        fn test_audio_chunk_duration_ms_stereo() {
            // 16000 Hz, stereo, PCM16 = 4 bytes per sample-pair
            // 6400 bytes = 1600 sample-pairs = 100ms at 16kHz
            let chunk = AudioChunk::new(vec![0u8; 6400], 16000, 2, AudioFormat::Pcm16);
            assert_eq!(chunk.duration_ms(), 100);
        }

        #[test]
        fn test_audio_chunk_duration_zero_rate() {
            let chunk = AudioChunk::new(vec![0u8; 100], 0, 1, AudioFormat::Pcm16);
            assert_eq!(chunk.duration_ms(), 0);
        }

        #[test]
        fn test_audio_chunk_duration_zero_channels() {
            let chunk = AudioChunk::new(vec![0u8; 100], 16000, 0, AudioFormat::Pcm16);
            assert_eq!(chunk.duration_ms(), 0);
        }

        #[test]
        fn test_audio_chunk_is_empty() {
            let empty = AudioChunk::new(vec![], 16000, 1, AudioFormat::Pcm16);
            assert!(empty.is_empty());

            let non_empty = AudioChunk::new(vec![0, 0], 16000, 1, AudioFormat::Pcm16);
            assert!(!non_empty.is_empty());
        }

        #[test]
        fn test_audio_chunk_sample_count() {
            // 3200 bytes, mono PCM16 = 1600 samples
            let chunk = AudioChunk::new(vec![0u8; 3200], 16000, 1, AudioFormat::Pcm16);
            assert_eq!(chunk.sample_count(), 1600);
        }

        #[test]
        fn test_audio_chunk_sample_count_zero_channels() {
            let chunk = AudioChunk::new(vec![0u8; 100], 16000, 0, AudioFormat::Pcm16);
            assert_eq!(chunk.sample_count(), 0);
        }

        // ----------------------------------------------------------------
        // VadConfig + VadDetector tests
        // ----------------------------------------------------------------

        #[test]
        fn test_vad_config_default() {
            let cfg = VadConfig::default();
            assert!((cfg.energy_threshold - 0.02).abs() < f32::EPSILON);
            assert_eq!(cfg.silence_duration_ms, 500);
            assert_eq!(cfg.min_speech_duration_ms, 200);
            assert_eq!(cfg.frame_size_ms, 20);
        }

        #[test]
        fn test_vad_detector_silence_detection() {
            let mut vad = VadDetector::with_defaults();
            // Silent frame (all zeros)
            let frame = vec![0i16; 320]; // 20ms at 16kHz
            let event = vad.process_frame(&frame);
            assert_eq!(event, VadEvent::Silence);
            assert!(!vad.is_in_speech());
        }

        #[test]
        fn test_vad_detector_speech_start() {
            let mut vad = VadDetector::new(VadConfig {
                energy_threshold: 0.01,
                silence_duration_ms: 100,
                min_speech_duration_ms: 10,
                frame_size_ms: 20,
            });

            // Loud frame (high energy)
            let frame: Vec<i16> = (0..320).map(|i| ((i % 50) * 500) as i16).collect();
            let event = vad.process_frame(&frame);
            assert_eq!(event, VadEvent::SpeechStart { timestamp_ms: 0 });
            assert!(vad.is_in_speech());
        }

        #[test]
        fn test_vad_detector_speech_end() {
            let mut vad = VadDetector::new(VadConfig {
                energy_threshold: 0.01,
                silence_duration_ms: 60, // 3 frames of silence at 20ms each
                min_speech_duration_ms: 10,
                frame_size_ms: 20,
            });

            // Start speech with loud frames
            let loud_frame: Vec<i16> =
                (0..320).map(|i| ((i % 50) * 500) as i16).collect();
            let event = vad.process_frame(&loud_frame);
            assert_eq!(event, VadEvent::SpeechStart { timestamp_ms: 0 });

            // Send more loud frames to build up duration
            for _ in 0..15 {
                vad.process_frame(&loud_frame);
            }

            // Now send silence frames to trigger speech end
            let silent_frame = vec![0i16; 320];
            let mut found_end = false;
            for _ in 0..20 {
                let ev = vad.process_frame(&silent_frame);
                if matches!(ev, VadEvent::SpeechEnd { .. }) {
                    found_end = true;
                    break;
                }
            }
            assert!(found_end, "Expected SpeechEnd after silence");
            assert!(!vad.is_in_speech());
        }

        #[test]
        fn test_vad_detector_short_speech_discarded() {
            let mut vad = VadDetector::new(VadConfig {
                energy_threshold: 0.01,
                silence_duration_ms: 40, // 2 frames
                min_speech_duration_ms: 200,
                frame_size_ms: 20,
            });

            // One loud frame, then immediate silence
            let loud_frame: Vec<i16> =
                (0..320).map(|i| ((i % 50) * 500) as i16).collect();
            let silent_frame = vec![0i16; 320];

            let _ = vad.process_frame(&loud_frame);
            // Send silence frames — the speech is too short, should be discarded
            let mut events = Vec::new();
            for _ in 0..10 {
                events.push(vad.process_frame(&silent_frame));
            }
            // No SpeechEnd should be emitted because duration < min_speech_duration_ms
            for ev in &events {
                assert!(
                    !matches!(ev, VadEvent::SpeechEnd { .. }),
                    "Short speech should be discarded"
                );
            }
        }

        #[test]
        fn test_vad_detector_reset() {
            let mut vad = VadDetector::with_defaults();
            let loud_frame: Vec<i16> =
                (0..320).map(|i| ((i % 50) * 500) as i16).collect();
            vad.process_frame(&loud_frame);
            assert!(vad.current_timestamp_ms() > 0);

            vad.reset();
            assert_eq!(vad.current_timestamp_ms(), 0);
            assert!(!vad.is_in_speech());
        }

        #[test]
        fn test_vad_detector_empty_frame() {
            let mut vad = VadDetector::with_defaults();
            let event = vad.process_frame(&[]);
            assert_eq!(event, VadEvent::Silence);
        }

        #[test]
        fn test_vad_event_equality() {
            assert_eq!(
                VadEvent::SpeechStart { timestamp_ms: 100 },
                VadEvent::SpeechStart { timestamp_ms: 100 }
            );
            assert_ne!(
                VadEvent::SpeechStart { timestamp_ms: 100 },
                VadEvent::SpeechStart { timestamp_ms: 200 }
            );
            assert_ne!(VadEvent::Silence, VadEvent::SpeechStart { timestamp_ms: 0 });
        }

        // ----------------------------------------------------------------
        // InterruptionPolicy tests
        // ----------------------------------------------------------------

        #[test]
        fn test_interruption_policy_display() {
            assert_eq!(InterruptionPolicy::Immediate.to_string(), "immediate");
            assert_eq!(InterruptionPolicy::EndSentence.to_string(), "end_sentence");
            assert_eq!(InterruptionPolicy::Never.to_string(), "never");
        }

        #[test]
        fn test_interruption_policy_equality() {
            assert_eq!(InterruptionPolicy::Immediate, InterruptionPolicy::Immediate);
            assert_ne!(InterruptionPolicy::Immediate, InterruptionPolicy::Never);
        }

        #[test]
        fn test_interruption_event_creation() {
            let event = InterruptionEvent {
                timestamp_ms: 5000,
                partial_response: "Hello, I was saying...".to_string(),
                policy_applied: InterruptionPolicy::EndSentence,
            };
            assert_eq!(event.timestamp_ms, 5000);
            assert_eq!(event.policy_applied, InterruptionPolicy::EndSentence);
            assert!(event.partial_response.contains("saying"));
        }

        // ----------------------------------------------------------------
        // TurnSpeaker + TurnPolicy tests
        // ----------------------------------------------------------------

        #[test]
        fn test_turn_speaker_display() {
            assert_eq!(TurnSpeaker::User.to_string(), "user");
            assert_eq!(TurnSpeaker::Agent.to_string(), "agent");
        }

        #[test]
        fn test_turn_policy_display() {
            assert_eq!(
                TurnPolicy::StrictAlternating.to_string(),
                "strict_alternating"
            );
            assert_eq!(TurnPolicy::NaturalOverlap.to_string(), "natural_overlap");
            assert_eq!(TurnPolicy::PushToTalk.to_string(), "push_to_talk");
        }

        // ----------------------------------------------------------------
        // TurnManager tests
        // ----------------------------------------------------------------

        #[test]
        fn test_turn_manager_basic_flow() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            assert_eq!(tm.total_turns(), 0);
            assert!(tm.current_speaker().is_none());

            tm.start_turn(TurnSpeaker::User).unwrap();
            assert_eq!(*tm.current_speaker().unwrap(), TurnSpeaker::User);

            let turn = tm.end_turn("Hello!".to_string(), 1200).unwrap();
            assert_eq!(turn.speaker, TurnSpeaker::User);
            assert_eq!(turn.transcript, "Hello!");
            assert_eq!(turn.audio_duration_ms, 1200);
            assert_eq!(turn.turn_number, 1);
            assert_eq!(tm.total_turns(), 1);
        }

        #[test]
        fn test_turn_manager_strict_alternating_rejects_same_speaker() {
            let mut tm = TurnManager::new(TurnPolicy::StrictAlternating);
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("Hi".to_string(), 500).unwrap();

            // Same speaker again should fail under StrictAlternating
            // First we need user to be "current" — start another user turn
            tm.start_turn(TurnSpeaker::User).unwrap();
            let result = tm.start_turn(TurnSpeaker::User);
            assert!(result.is_err());
        }

        #[test]
        fn test_turn_manager_strict_alternating_allows_different_speaker() {
            let mut tm = TurnManager::new(TurnPolicy::StrictAlternating);
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("Hi".to_string(), 500).unwrap();

            tm.start_turn(TurnSpeaker::Agent).unwrap();
            tm.end_turn("Hello!".to_string(), 600).unwrap();
            assert_eq!(tm.total_turns(), 2);
        }

        #[test]
        fn test_turn_manager_natural_overlap_allows_same_speaker() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("Part 1".to_string(), 300).unwrap();

            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("Part 2".to_string(), 200).unwrap();
            assert_eq!(tm.total_turns(), 2);
        }

        #[test]
        fn test_turn_manager_end_turn_without_start() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            let result = tm.end_turn("orphan".to_string(), 100);
            assert!(result.is_err());
        }

        #[test]
        fn test_turn_manager_get_history() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("First".to_string(), 100).unwrap();
            tm.start_turn(TurnSpeaker::Agent).unwrap();
            tm.end_turn("Second".to_string(), 200).unwrap();
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("Third".to_string(), 300).unwrap();

            let history = tm.get_history();
            assert_eq!(history.len(), 3);
            assert_eq!(history[0].transcript, "First");
            assert_eq!(history[2].transcript, "Third");
        }

        #[test]
        fn test_turn_manager_get_last_n() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            for i in 1..=5 {
                tm.start_turn(TurnSpeaker::User).unwrap();
                tm.end_turn(format!("Turn {}", i), i as u64 * 100)
                    .unwrap();
            }

            let last_2 = tm.get_last_n(2);
            assert_eq!(last_2.len(), 2);
            assert_eq!(last_2[0].transcript, "Turn 4");
            assert_eq!(last_2[1].transcript, "Turn 5");

            // Request more than available
            let last_10 = tm.get_last_n(10);
            assert_eq!(last_10.len(), 5);
        }

        #[test]
        fn test_turn_manager_total_duration() {
            let mut tm = TurnManager::new(TurnPolicy::NaturalOverlap);
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("A".to_string(), 100).unwrap();
            tm.start_turn(TurnSpeaker::Agent).unwrap();
            tm.end_turn("B".to_string(), 200).unwrap();
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("C".to_string(), 300).unwrap();

            assert_eq!(tm.total_duration_ms(), 600);
        }

        #[test]
        fn test_turn_manager_push_to_talk() {
            let mut tm = TurnManager::new(TurnPolicy::PushToTalk);
            // PushToTalk allows any sequence — no restrictions
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("ptt1".to_string(), 50).unwrap();
            tm.start_turn(TurnSpeaker::User).unwrap();
            tm.end_turn("ptt2".to_string(), 60).unwrap();
            assert_eq!(tm.total_turns(), 2);
        }

        // ----------------------------------------------------------------
        // VoiceSessionState tests
        // ----------------------------------------------------------------

        #[test]
        fn test_session_state_display() {
            assert_eq!(VoiceSessionState::Idle.to_string(), "idle");
            assert_eq!(VoiceSessionState::Listening.to_string(), "listening");
            assert_eq!(VoiceSessionState::Processing.to_string(), "processing");
            assert_eq!(VoiceSessionState::Speaking.to_string(), "speaking");
            assert_eq!(VoiceSessionState::Interrupted.to_string(), "interrupted");
            assert_eq!(VoiceSessionState::Closed.to_string(), "closed");
        }

        #[test]
        fn test_session_state_valid_transitions() {
            assert!(VoiceSessionState::Idle.can_transition_to(&VoiceSessionState::Listening));
            assert!(
                VoiceSessionState::Listening
                    .can_transition_to(&VoiceSessionState::Processing)
            );
            assert!(
                VoiceSessionState::Processing
                    .can_transition_to(&VoiceSessionState::Speaking)
            );
            assert!(
                VoiceSessionState::Processing
                    .can_transition_to(&VoiceSessionState::Listening)
            );
            assert!(
                VoiceSessionState::Speaking
                    .can_transition_to(&VoiceSessionState::Listening)
            );
            assert!(
                VoiceSessionState::Speaking
                    .can_transition_to(&VoiceSessionState::Interrupted)
            );
            assert!(
                VoiceSessionState::Interrupted
                    .can_transition_to(&VoiceSessionState::Listening)
            );
        }

        #[test]
        fn test_session_state_closed_transitions() {
            // Any state can go to Closed
            assert!(VoiceSessionState::Idle.can_transition_to(&VoiceSessionState::Closed));
            assert!(
                VoiceSessionState::Listening.can_transition_to(&VoiceSessionState::Closed)
            );
            assert!(
                VoiceSessionState::Processing.can_transition_to(&VoiceSessionState::Closed)
            );
            assert!(
                VoiceSessionState::Speaking.can_transition_to(&VoiceSessionState::Closed)
            );
            assert!(
                VoiceSessionState::Interrupted.can_transition_to(&VoiceSessionState::Closed)
            );
        }

        #[test]
        fn test_session_state_invalid_transitions() {
            // Closed cannot transition anywhere
            assert!(
                !VoiceSessionState::Closed
                    .can_transition_to(&VoiceSessionState::Listening)
            );
            assert!(
                !VoiceSessionState::Closed.can_transition_to(&VoiceSessionState::Idle)
            );

            // Idle cannot go directly to Processing or Speaking
            assert!(
                !VoiceSessionState::Idle
                    .can_transition_to(&VoiceSessionState::Processing)
            );
            assert!(
                !VoiceSessionState::Idle.can_transition_to(&VoiceSessionState::Speaking)
            );

            // Listening cannot go directly to Speaking
            assert!(
                !VoiceSessionState::Listening
                    .can_transition_to(&VoiceSessionState::Speaking)
            );
        }

        // ----------------------------------------------------------------
        // VoiceSession tests
        // ----------------------------------------------------------------

        #[test]
        fn test_voice_session_creation() {
            let config = VoiceAgentConfig::default();
            let session = VoiceSession::new(config);
            assert!(session.session_id.starts_with("vs-"));
            assert_eq!(*session.state(), VoiceSessionState::Idle);
            assert!(session.history().is_empty());
        }

        #[test]
        fn test_voice_session_transition() {
            let config = VoiceAgentConfig::default();
            let mut session = VoiceSession::new(config);

            session
                .transition_to(VoiceSessionState::Listening)
                .unwrap();
            assert_eq!(*session.state(), VoiceSessionState::Listening);

            session
                .transition_to(VoiceSessionState::Processing)
                .unwrap();
            assert_eq!(*session.state(), VoiceSessionState::Processing);
        }

        #[test]
        fn test_voice_session_invalid_transition() {
            let config = VoiceAgentConfig::default();
            let mut session = VoiceSession::new(config);
            // Idle -> Speaking should fail
            let result = session.transition_to(VoiceSessionState::Speaking);
            assert!(result.is_err());
        }

        #[test]
        fn test_voice_session_unique_ids() {
            let config = VoiceAgentConfig::default();
            let s1 = VoiceSession::new(config.clone());
            let s2 = VoiceSession::new(config);
            assert_ne!(s1.session_id, s2.session_id);
        }

        // ----------------------------------------------------------------
        // VoiceAgentConfig tests
        // ----------------------------------------------------------------

        #[test]
        fn test_voice_agent_config_default() {
            let cfg = VoiceAgentConfig::default();
            assert_eq!(cfg.model_stt, "whisper-1");
            assert_eq!(cfg.model_tts, "tts-1");
            assert_eq!(cfg.sample_rate, 16000);
            assert_eq!(cfg.chunk_size_ms, 20);
            assert_eq!(cfg.voice_id, "alloy");
            assert_eq!(cfg.interruption_policy, InterruptionPolicy::Immediate);
        }

        #[test]
        fn test_voice_agent_config_validate_ok() {
            let cfg = VoiceAgentConfig::default();
            assert!(cfg.validate().is_ok());
        }

        #[test]
        fn test_voice_agent_config_validate_empty_stt() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.model_stt = String::new();
            assert!(cfg.validate().is_err());
        }

        #[test]
        fn test_voice_agent_config_validate_empty_tts() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.model_tts = String::new();
            assert!(cfg.validate().is_err());
        }

        #[test]
        fn test_voice_agent_config_validate_zero_sample_rate() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.sample_rate = 0;
            assert!(cfg.validate().is_err());
        }

        #[test]
        fn test_voice_agent_config_validate_zero_chunk_size() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.chunk_size_ms = 0;
            assert!(cfg.validate().is_err());
        }

        #[test]
        fn test_voice_agent_config_validate_bad_threshold() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.vad_config.energy_threshold = -0.1;
            assert!(cfg.validate().is_err());

            cfg.vad_config.energy_threshold = 1.5;
            assert!(cfg.validate().is_err());
        }

        // ----------------------------------------------------------------
        // InMemoryTransport tests
        // ----------------------------------------------------------------

        #[test]
        fn test_in_memory_transport_new() {
            let transport = InMemoryTransport::new();
            assert!(transport.sent_chunks().is_empty());
            assert!(!transport.is_closed());
        }

        #[test]
        fn test_in_memory_transport_default() {
            let transport = InMemoryTransport::default();
            assert!(!transport.is_closed());
        }

        #[test]
        fn test_in_memory_transport_send_receive() {
            let mut transport = InMemoryTransport::new();

            // Enqueue a chunk to be received
            let chunk = AudioChunk::new(vec![1, 2, 3, 4], 16000, 1, AudioFormat::Pcm16);
            transport.enqueue(chunk.clone());

            // Receive it
            let received = transport.receive_audio().unwrap();
            assert!(received.is_some());
            assert_eq!(received.unwrap().bytes, vec![1, 2, 3, 4]);

            // Nothing more to receive
            let empty = transport.receive_audio().unwrap();
            assert!(empty.is_none());
        }

        #[test]
        fn test_in_memory_transport_send_mut() {
            let mut transport = InMemoryTransport::new();
            let chunk = AudioChunk::new(vec![5, 6], 16000, 1, AudioFormat::Pcm16);
            transport.send_audio_mut(&chunk).unwrap();
            assert_eq!(transport.sent_chunks().len(), 1);
            assert_eq!(transport.sent_chunks()[0].bytes, vec![5, 6]);
        }

        #[test]
        fn test_in_memory_transport_close() {
            let mut transport = InMemoryTransport::new();
            assert!(!transport.is_closed());
            transport.close().unwrap();
            assert!(transport.is_closed());
        }

        #[test]
        fn test_in_memory_transport_closed_errors() {
            let mut transport = InMemoryTransport::new();
            transport.close().unwrap();

            let chunk = AudioChunk::new(vec![0, 0], 16000, 1, AudioFormat::Pcm16);
            assert!(transport.send_audio_mut(&chunk).is_err());
            assert!(transport.receive_audio().is_err());
        }

        // ----------------------------------------------------------------
        // VoiceAgent tests
        // ----------------------------------------------------------------

        #[test]
        fn test_voice_agent_creation() {
            let agent = VoiceAgent::with_defaults().unwrap();
            assert_eq!(agent.config().model_stt, "whisper-1");
            assert_eq!(agent.active_session_count(), 0);
        }

        #[test]
        fn test_voice_agent_invalid_config() {
            let mut cfg = VoiceAgentConfig::default();
            cfg.sample_rate = 0;
            let result = VoiceAgent::new(cfg);
            assert!(result.is_err());
        }

        #[test]
        fn test_voice_agent_start_session() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let session = agent.start_session().unwrap();
            assert_eq!(*session.state(), VoiceSessionState::Idle);
            assert_eq!(agent.active_session_count(), 1);
        }

        #[test]
        fn test_voice_agent_end_session() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let mut session = agent.start_session().unwrap();
            assert_eq!(agent.active_session_count(), 1);

            agent.end_session(&mut session).unwrap();
            assert_eq!(*session.state(), VoiceSessionState::Closed);
            assert_eq!(agent.active_session_count(), 0);
        }

        #[test]
        fn test_voice_agent_process_silent_audio() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let mut session = agent.start_session().unwrap();

            // Silent chunk — all zeros, 20ms at 16kHz mono = 640 bytes
            let chunk = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);
            let result = agent.process_audio(&mut session, &chunk).unwrap();
            assert!(result.is_none(), "Silent audio should not produce a response");
        }

        #[test]
        fn test_voice_agent_get_transcript_empty() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let session = agent.start_session().unwrap();
            let transcript = agent.get_transcript(&session);
            assert!(transcript.is_empty());
        }

        #[test]
        fn test_voice_agent_session_lifecycle() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let mut session = agent.start_session().unwrap();

            // Process some silence
            let chunk = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);
            let _ = agent.process_audio(&mut session, &chunk);

            // Session should be in Listening state now
            assert_eq!(*session.state(), VoiceSessionState::Listening);

            // End session
            agent.end_session(&mut session).unwrap();
            assert_eq!(*session.state(), VoiceSessionState::Closed);
        }

        #[test]
        fn test_voice_agent_process_audio_closed_session() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let mut session = agent.start_session().unwrap();
            agent.end_session(&mut session).unwrap();

            let chunk = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);
            let result = agent.process_audio(&mut session, &chunk);
            assert!(result.is_err(), "Processing on closed session should fail");
        }

        #[test]
        fn test_voice_agent_bytes_to_samples() {
            // Little-endian PCM16: [0x01, 0x00] = 1, [0xFF, 0x7F] = 32767
            let bytes = vec![0x01, 0x00, 0xFF, 0x7F];
            let samples = VoiceAgent::bytes_to_samples(&bytes);
            assert_eq!(samples.len(), 2);
            assert_eq!(samples[0], 1);
            assert_eq!(samples[1], 32767);
        }

        #[test]
        fn test_voice_agent_bytes_to_samples_odd_length() {
            // Odd number of bytes — last byte is discarded by chunks_exact
            let bytes = vec![0x01, 0x00, 0xFF];
            let samples = VoiceAgent::bytes_to_samples(&bytes);
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0], 1);
        }

        // ----------------------------------------------------------------
        // ConversationTurn tests
        // ----------------------------------------------------------------

        #[test]
        fn test_conversation_turn_creation() {
            let turn = ConversationTurn {
                speaker: TurnSpeaker::User,
                transcript: "Hello there".to_string(),
                audio_duration_ms: 1500,
                timestamp: chrono::Utc::now(),
                turn_number: 1,
            };
            assert_eq!(turn.speaker, TurnSpeaker::User);
            assert_eq!(turn.transcript, "Hello there");
            assert_eq!(turn.audio_duration_ms, 1500);
            assert_eq!(turn.turn_number, 1);
        }

        // ----------------------------------------------------------------
        // Integration-style tests
        // ----------------------------------------------------------------

        #[test]
        fn test_full_vad_speech_cycle() {
            let mut vad = VadDetector::new(VadConfig {
                energy_threshold: 0.01,
                silence_duration_ms: 60,
                min_speech_duration_ms: 40,
                frame_size_ms: 20,
            });

            let loud_frame: Vec<i16> =
                (0..320).map(|i| ((i % 50) * 600) as i16).collect();
            let silent_frame = vec![0i16; 320];

            // Phase 1: silence
            for _ in 0..5 {
                let ev = vad.process_frame(&silent_frame);
                assert_eq!(ev, VadEvent::Silence);
            }

            // Phase 2: speech starts
            let ev = vad.process_frame(&loud_frame);
            assert_eq!(ev, VadEvent::SpeechStart { timestamp_ms: 100 });

            // Phase 3: continue speech
            for _ in 0..10 {
                vad.process_frame(&loud_frame);
            }
            assert!(vad.is_in_speech());

            // Phase 4: silence -> speech ends
            let mut got_end = false;
            for _ in 0..20 {
                let ev = vad.process_frame(&silent_frame);
                if let VadEvent::SpeechEnd {
                    duration_ms, ..
                } = ev
                {
                    assert!(duration_ms >= 40);
                    got_end = true;
                    break;
                }
            }
            assert!(got_end, "Expected speech end");
        }

        #[test]
        fn test_transport_trait_object() {
            // Verify VoiceTransport can be used as a trait object
            let mut transport: Box<dyn VoiceTransport> =
                Box::new(InMemoryTransport::new());
            let chunk = AudioChunk::new(vec![0, 0], 16000, 1, AudioFormat::Pcm16);
            assert!(transport.send_audio(&chunk).is_ok());
            assert!(transport.receive_audio().unwrap().is_none());
            assert!(transport.close().is_ok());
        }

        #[test]
        fn test_multiple_sessions() {
            let mut agent = VoiceAgent::with_defaults().unwrap();
            let _s1 = agent.start_session().unwrap();
            let _s2 = agent.start_session().unwrap();
            let _s3 = agent.start_session().unwrap();
            assert_eq!(agent.active_session_count(), 3);
        }

        #[test]
        fn test_voice_session_interruption_recording() {
            let config = VoiceAgentConfig::default();
            let mut session = VoiceSession::new(config);
            assert!(session.interruptions().is_empty());

            session.record_interruption(InterruptionEvent {
                timestamp_ms: 1000,
                partial_response: "I was...".to_string(),
                policy_applied: InterruptionPolicy::Immediate,
            });
            assert_eq!(session.interruptions().len(), 1);
            assert_eq!(session.interruptions()[0].timestamp_ms, 1000);
        }

        // ----------------------------------------------------------------
        // 9.1 — WebRTC Voice Transport tests
        // ----------------------------------------------------------------

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_config_defaults() {
            let cfg = WebRtcConfig::default();
            assert_eq!(cfg.stun_servers, vec!["stun:stun.l.google.com:19302"]);
            assert!(cfg.turn_servers.is_empty());
            assert_eq!(cfg.audio_codec, WebRtcAudioCodec::Opus);
            assert_eq!(cfg.sample_rate, 48000);
            assert!(cfg.enable_dtls);
            assert_eq!(cfg.ice_timeout_ms, 5000);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_new() {
            let cfg = WebRtcConfig::default();
            let transport = WebRtcTransport::new(cfg);
            assert!(!transport.is_connected());
            assert!(transport.ice_candidates().is_empty());
            assert_eq!(transport.stats().bytes_sent, 0);
            assert_eq!(transport.config().sample_rate, 48000);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_create_offer() {
            let mut transport = WebRtcTransport::with_defaults();
            let offer = transport.create_offer();
            assert!(!offer.sdp.is_empty());
            assert!(offer.sdp.contains("v=0"));
            assert!(offer.sdp.contains("opus"));
            assert!(offer.sdp.contains("48000"));
            assert!(offer.session_id.starts_with("webrtc-"));
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_set_remote_answer() {
            let mut transport = WebRtcTransport::with_defaults();
            let answer = SdpAnswer {
                sdp: "v=0\r\no=- answer 2 IN IP4 127.0.0.1\r\n".to_string(),
                session_id: "answer-001".to_string(),
            };
            transport.set_remote_answer(answer);
            // Verify we can still access the transport after setting the answer
            assert!(!transport.is_connected());
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_add_ice_candidate() {
            let mut transport = WebRtcTransport::with_defaults();
            assert!(transport.ice_candidates().is_empty());

            let candidate = WebRtcIceCandidate {
                candidate: "candidate:1 1 udp 2130706431 192.168.1.1 5000 typ host".to_string(),
                sdp_mid: Some("0".to_string()),
                sdp_m_line_index: Some(0),
                priority: 2130706431,
                candidate_type: IceCandidateType::Host,
            };
            transport.add_ice_candidate(candidate);
            assert_eq!(transport.ice_candidates().len(), 1);
            assert_eq!(transport.ice_candidates()[0].priority, 2130706431);

            // Add a second candidate
            let relay_candidate = WebRtcIceCandidate {
                candidate: "candidate:2 1 udp 1 10.0.0.1 3478 typ relay".to_string(),
                sdp_mid: None,
                sdp_m_line_index: None,
                priority: 1,
                candidate_type: IceCandidateType::Relay,
            };
            transport.add_ice_candidate(relay_candidate);
            assert_eq!(transport.ice_candidates().len(), 2);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_connect_without_sdp_fails() {
            let mut transport = WebRtcTransport::with_defaults();
            // No offer created, no answer set — should fail
            let result = transport.connect();
            assert!(result.is_err());
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_connect_with_offer_no_answer_fails() {
            let mut transport = WebRtcTransport::with_defaults();
            transport.create_offer();
            // Offer created but no answer — should fail
            let result = transport.connect();
            assert!(result.is_err());
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_connect_success() {
            let mut transport = WebRtcTransport::with_defaults();
            transport.create_offer();
            transport.set_remote_answer(SdpAnswer {
                sdp: "v=0\r\n".to_string(),
                session_id: "remote-1".to_string(),
            });
            let result = transport.connect();
            assert!(result.is_ok());
            assert!(transport.is_connected());
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_disconnect() {
            let mut transport = WebRtcTransport::with_defaults();
            transport.create_offer();
            transport.set_remote_answer(SdpAnswer {
                sdp: "v=0\r\n".to_string(),
                session_id: "remote-2".to_string(),
            });
            transport.connect().unwrap();
            assert!(transport.is_connected());

            transport.disconnect();
            assert!(!transport.is_connected());
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_transport_stats() {
            let mut transport = WebRtcTransport::with_defaults();
            // Before connection, stats are default/zeroed
            assert_eq!(transport.stats().latency_ms, 0.0);
            assert_eq!(transport.stats().bytes_sent, 0);

            transport.create_offer();
            transport.set_remote_answer(SdpAnswer {
                sdp: "v=0\r\n".to_string(),
                session_id: "remote-3".to_string(),
            });
            transport.connect().unwrap();

            // After connection, stats are populated with initial values
            assert!(transport.stats().latency_ms > 0.0);
            assert!((transport.stats().packet_loss_pct - 0.0).abs() < f64::EPSILON);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_sdp_offer_construction() {
            let offer = SdpOffer {
                sdp: "v=0\r\no=- session 2 IN IP4 0.0.0.0\r\n".to_string(),
                session_id: "test-offer-1".to_string(),
            };
            assert!(offer.sdp.contains("v=0"));
            assert_eq!(offer.session_id, "test-offer-1");
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_sdp_answer_construction() {
            let answer = SdpAnswer {
                sdp: "v=0\r\no=- answer 2 IN IP4 0.0.0.0\r\n".to_string(),
                session_id: "test-answer-1".to_string(),
            };
            assert!(answer.sdp.contains("v=0"));
            assert_eq!(answer.session_id, "test-answer-1");
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_ice_candidate_construction() {
            let candidate = WebRtcIceCandidate {
                candidate: "candidate:1 1 udp 2130706431 192.168.1.1 5000 typ host".to_string(),
                sdp_mid: Some("audio".to_string()),
                sdp_m_line_index: Some(0),
                priority: 2130706431,
                candidate_type: IceCandidateType::Host,
            };
            assert!(candidate.candidate.contains("host"));
            assert_eq!(candidate.sdp_mid, Some("audio".to_string()));
            assert_eq!(candidate.sdp_m_line_index, Some(0));
            assert_eq!(candidate.priority, 2130706431);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_ice_candidate_type_all_variants() {
            let types = vec![
                IceCandidateType::Host,
                IceCandidateType::ServerReflexive,
                IceCandidateType::PeerReflexive,
                IceCandidateType::Relay,
            ];
            // Ensure all variants are distinct
            for (i, t1) in types.iter().enumerate() {
                for (j, t2) in types.iter().enumerate() {
                    if i == j {
                        assert_eq!(t1, t2);
                    } else {
                        assert_ne!(t1, t2);
                    }
                }
            }
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_audio_codec_all_variants() {
            let codecs = vec![
                WebRtcAudioCodec::Opus,
                WebRtcAudioCodec::Pcm16,
                WebRtcAudioCodec::G711,
            ];
            assert_eq!(codecs.len(), 3);
            assert_ne!(WebRtcAudioCodec::Opus, WebRtcAudioCodec::G711);
            assert_ne!(WebRtcAudioCodec::Pcm16, WebRtcAudioCodec::G711);
            assert_eq!(WebRtcAudioCodec::Opus, WebRtcAudioCodec::Opus);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_turn_server_construction() {
            let server = TurnServer {
                url: "turn:turn.example.com:3478".to_string(),
                username: "user".to_string(),
                credential: "pass123".to_string(),
            };
            assert!(server.url.starts_with("turn:"));
            assert_eq!(server.username, "user");
            assert_eq!(server.credential, "pass123");
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_rtp_stream_config_construction() {
            let config = RtpStreamConfig {
                ssrc: 12345,
                payload_type: 111,
                clock_rate: 48000,
                jitter_buffer_ms: 50,
            };
            assert_eq!(config.ssrc, 12345);
            assert_eq!(config.payload_type, 111);
            assert_eq!(config.clock_rate, 48000);
            assert_eq!(config.jitter_buffer_ms, 50);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_session_stats_defaults() {
            let stats = WebRtcSessionStats::default();
            assert!((stats.latency_ms - 0.0).abs() < f64::EPSILON);
            assert!((stats.packet_loss_pct - 0.0).abs() < f64::EPSILON);
            assert!((stats.jitter_ms - 0.0).abs() < f64::EPSILON);
            assert_eq!(stats.bytes_sent, 0);
            assert_eq!(stats.bytes_received, 0);
        }

        #[cfg(feature = "webrtc")]
        #[test]
        fn test_webrtc_voice_transport_impl() {
            let mut transport = WebRtcTransport::with_defaults();
            transport.create_offer();
            transport.set_remote_answer(SdpAnswer {
                sdp: "v=0\r\n".to_string(),
                session_id: "vt-test".to_string(),
            });
            transport.connect().unwrap();

            // send_audio with valid data should succeed
            let chunk = AudioChunk::new(vec![1, 2, 3, 4], 48000, 1, AudioFormat::Pcm16);
            assert!(transport.send_audio(&chunk).is_ok());

            // send_audio with empty data should fail
            let empty_chunk = AudioChunk::new(vec![], 48000, 1, AudioFormat::Pcm16);
            assert!(transport.send_audio(&empty_chunk).is_err());

            // receive_audio returns None (simulated)
            let received = transport.receive_audio().unwrap();
            assert!(received.is_none());

            // close disconnects and clears state
            transport.close().unwrap();
            assert!(!transport.is_connected());
            assert!(transport.ice_candidates().is_empty());

            // Operations on closed/disconnected transport should fail
            let chunk2 = AudioChunk::new(vec![5, 6], 48000, 1, AudioFormat::Pcm16);
            assert!(transport.send_audio(&chunk2).is_err());
            assert!(transport.receive_audio().is_err());
        }

        // ----------------------------------------------------------------
        // 9.2 — Speech-to-Speech Pipeline tests
        // ----------------------------------------------------------------

        #[test]
        fn test_audio_model_capability_all_variants() {
            let capabilities = vec![
                AudioModelCapability::TextOnly,
                AudioModelCapability::AudioInput,
                AudioModelCapability::AudioOutput,
                AudioModelCapability::AudioInputOutput,
            ];
            assert_eq!(capabilities.len(), 4);
            assert_ne!(AudioModelCapability::TextOnly, AudioModelCapability::AudioInputOutput);
            assert_ne!(AudioModelCapability::AudioInput, AudioModelCapability::AudioOutput);
            assert_eq!(AudioModelCapability::TextOnly, AudioModelCapability::TextOnly);
        }

        #[test]
        fn test_s2s_config_construction() {
            let config = S2SConfig {
                model_id: "gpt-4o-audio".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: Some("alloy".to_string()),
                capability: AudioModelCapability::AudioInputOutput,
            };
            assert_eq!(config.model_id, "gpt-4o-audio");
            assert_eq!(config.input_format, AudioFormat::Pcm16);
            assert_eq!(config.output_format, AudioFormat::Pcm16);
            assert_eq!(config.voice_id, Some("alloy".to_string()));
            assert_eq!(config.capability, AudioModelCapability::AudioInputOutput);
        }

        #[test]
        fn test_s2s_fallback_mode_variants() {
            assert_eq!(S2SFallbackMode::SttLlmTts, S2SFallbackMode::SttLlmTts);
            assert_eq!(S2SFallbackMode::Disabled, S2SFallbackMode::Disabled);
            assert_ne!(S2SFallbackMode::SttLlmTts, S2SFallbackMode::Disabled);
        }

        #[test]
        fn test_s2s_pipeline_new() {
            let config = S2SConfig {
                model_id: "test-model".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Wav,
                voice_id: None,
                capability: AudioModelCapability::AudioInputOutput,
            };
            let pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);
            assert!(pipeline.transcripts().is_empty());
            assert_eq!(pipeline.config().model_id, "test-model");
            assert_eq!(*pipeline.fallback_mode(), S2SFallbackMode::Disabled);
        }

        #[test]
        fn test_s2s_pipeline_supports_direct_audio_true() {
            let config = S2SConfig {
                model_id: "native-s2s".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::AudioInputOutput,
            };
            let pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);
            assert!(pipeline.supports_direct_audio());
        }

        #[test]
        fn test_s2s_pipeline_supports_direct_audio_false() {
            let text_only = S2SConfig {
                model_id: "text-only".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::TextOnly,
            };
            let p1 = SpeechToSpeechPipeline::new(text_only, S2SFallbackMode::SttLlmTts);
            assert!(!p1.supports_direct_audio());

            let audio_in = S2SConfig {
                model_id: "audio-in".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::AudioInput,
            };
            let p2 = SpeechToSpeechPipeline::new(audio_in, S2SFallbackMode::SttLlmTts);
            assert!(!p2.supports_direct_audio());

            let audio_out = S2SConfig {
                model_id: "audio-out".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::AudioOutput,
            };
            let p3 = SpeechToSpeechPipeline::new(audio_out, S2SFallbackMode::SttLlmTts);
            assert!(!p3.supports_direct_audio());
        }

        #[test]
        fn test_s2s_pipeline_process_audio_direct() {
            let config = S2SConfig {
                model_id: "native-s2s".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: Some("echo".to_string()),
                capability: AudioModelCapability::AudioInputOutput,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);
            let input = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);

            let output = pipeline.process_audio(&input).unwrap();
            assert_eq!(output.bytes.len(), 640);
            assert_eq!(output.sample_rate, 16000);
            assert_eq!(output.format, AudioFormat::Pcm16);
            assert_eq!(pipeline.transcripts().len(), 1);
            assert!(!pipeline.transcripts()[0].used_fallback);
            assert_eq!(pipeline.transcripts()[0].model_id, "native-s2s");
        }

        #[test]
        fn test_s2s_pipeline_transcripts() {
            let config = S2SConfig {
                model_id: "s2s-model".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::AudioInputOutput,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);

            let input1 = AudioChunk::new(vec![0u8; 320], 16000, 1, AudioFormat::Pcm16);
            let input2 = AudioChunk::new(vec![1u8; 640], 16000, 1, AudioFormat::Pcm16);

            pipeline.process_audio(&input1).unwrap();
            pipeline.process_audio(&input2).unwrap();

            assert_eq!(pipeline.transcripts().len(), 2);
            assert!(pipeline.transcripts()[0].input_text.is_some());
            assert!(pipeline.transcripts()[1].output_text.is_some());
            assert!(pipeline.transcripts()[0].timestamp > 0);
        }

        #[test]
        fn test_s2s_pipeline_clear_transcripts() {
            let config = S2SConfig {
                model_id: "model".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::AudioInputOutput,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);
            let input = AudioChunk::new(vec![0u8; 100], 16000, 1, AudioFormat::Pcm16);
            pipeline.process_audio(&input).unwrap();
            assert_eq!(pipeline.transcripts().len(), 1);

            pipeline.clear_transcripts();
            assert!(pipeline.transcripts().is_empty());
        }

        #[test]
        fn test_s2s_transcript_construction() {
            let transcript = S2STranscript {
                input_text: Some("Hello".to_string()),
                output_text: Some("Hi there!".to_string()),
                model_id: "test-model".to_string(),
                used_fallback: false,
                latency_ms: 42,
                timestamp: 1700000000000,
            };
            assert_eq!(transcript.input_text, Some("Hello".to_string()));
            assert_eq!(transcript.output_text, Some("Hi there!".to_string()));
            assert_eq!(transcript.model_id, "test-model");
            assert!(!transcript.used_fallback);
            assert_eq!(transcript.latency_ms, 42);
            assert_eq!(transcript.timestamp, 1700000000000);
        }

        #[test]
        fn test_s2s_pipeline_text_only_with_fallback() {
            let config = S2SConfig {
                model_id: "text-llm".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Wav,
                voice_id: None,
                capability: AudioModelCapability::TextOnly,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::SttLlmTts);
            let input = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);

            let output = pipeline.process_audio(&input).unwrap();
            // Fallback produces TTS audio bytes
            assert!(!output.bytes.is_empty());
            assert_eq!(output.format, AudioFormat::Wav);
            assert_eq!(pipeline.transcripts().len(), 1);
            assert!(pipeline.transcripts()[0].used_fallback);
            assert!(pipeline.transcripts()[0].input_text.is_some());
            assert!(pipeline.transcripts()[0].output_text.is_some());
        }

        #[test]
        fn test_s2s_pipeline_text_only_disabled_fallback_errors() {
            let config = S2SConfig {
                model_id: "text-llm".to_string(),
                input_format: AudioFormat::Pcm16,
                output_format: AudioFormat::Pcm16,
                voice_id: None,
                capability: AudioModelCapability::TextOnly,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::Disabled);
            let input = AudioChunk::new(vec![0u8; 640], 16000, 1, AudioFormat::Pcm16);

            let result = pipeline.process_audio(&input);
            assert!(result.is_err());
            // No transcript should be recorded on error
            assert!(pipeline.transcripts().is_empty());
        }

        #[test]
        fn test_s2s_pipeline_audio_input_output_direct() {
            let config = S2SConfig {
                model_id: "gpt-4o-audio-preview".to_string(),
                input_format: AudioFormat::Wav,
                output_format: AudioFormat::Mp3,
                voice_id: Some("nova".to_string()),
                capability: AudioModelCapability::AudioInputOutput,
            };
            let mut pipeline = SpeechToSpeechPipeline::new(config, S2SFallbackMode::SttLlmTts);
            let input = AudioChunk::new(vec![42u8; 1024], 24000, 1, AudioFormat::Wav);

            let output = pipeline.process_audio(&input).unwrap();
            // Direct S2S: output should have same byte count as input (simulated)
            assert_eq!(output.bytes.len(), 1024);
            assert_eq!(output.format, AudioFormat::Mp3);
            assert_eq!(output.sample_rate, 24000);

            let t = &pipeline.transcripts()[0];
            assert!(!t.used_fallback);
            assert_eq!(t.model_id, "gpt-4o-audio-preview");
        }
    }
}

// Re-export everything from the inner module when the feature is enabled.
#[cfg(feature = "voice-agent")]
pub use inner::*;
