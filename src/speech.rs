//! Speech-to-text (STT) and text-to-speech (TTS) capabilities.
//!
//! Provides a unified `SpeechProvider` trait with implementations for:
//! - OpenAI (Whisper for STT, TTS API for synthesis)
//! - Google Cloud (Speech-to-Text v1, Text-to-Speech v1)
//!
//! Feature-gated behind the `audio` feature flag (handled by lib.rs).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read as IoRead;
use std::time::Duration;

#[cfg(feature = "whisper-local")]
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// ============================================================================
// Audio Types
// ============================================================================

/// Audio encoding format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AudioFormat {
    Wav,
    Mp3,
    Ogg,
    Flac,
    Pcm,
    Opus,
    Aac,
}

impl AudioFormat {
    /// File extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Flac => "flac",
            AudioFormat::Pcm => "pcm",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
        }
    }

    /// MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Ogg => "audio/ogg",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Pcm => "audio/pcm",
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Aac => "audio/aac",
        }
    }

    /// Parse an audio format from a file extension string.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(AudioFormat::Wav),
            "mp3" => Some(AudioFormat::Mp3),
            "ogg" => Some(AudioFormat::Ogg),
            "flac" => Some(AudioFormat::Flac),
            "pcm" => Some(AudioFormat::Pcm),
            "opus" => Some(AudioFormat::Opus),
            "aac" | "m4a" => Some(AudioFormat::Aac),
            _ => None,
        }
    }
}

/// Configuration for a speech provider.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpeechConfig {
    /// API key for authentication.
    pub api_key: Option<String>,
    /// Base URL override for the API endpoint.
    pub base_url: Option<String>,
    /// Model name (e.g., "whisper-1", "tts-1").
    pub model: Option<String>,
    /// Language hint (ISO 639-1 code).
    pub language: Option<String>,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            model: None,
            language: None,
            timeout: Duration::from_secs(60),
        }
    }
}

/// Result of speech-to-text transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Transcribed text.
    pub text: String,
    /// Detected language (ISO 639-1).
    pub language: Option<String>,
    /// Audio duration in seconds.
    pub duration_secs: f64,
    /// Timed segments (if available).
    pub segments: Vec<TranscriptionSegment>,
    /// Overall confidence score (0.0 - 1.0).
    pub confidence: Option<f64>,
}

/// A timed segment within a transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
    /// Segment text.
    pub text: String,
}

/// Options for text-to-speech synthesis.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Voice name or ID.
    pub voice: Option<String>,
    /// Output audio format.
    pub format: AudioFormat,
    /// Speed multiplier (0.5 - 2.0, default 1.0).
    pub speed: f64,
    /// Sample rate in Hz (default is provider-specific).
    pub sample_rate: Option<u32>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            voice: None,
            format: AudioFormat::Mp3,
            speed: 1.0,
            sample_rate: None,
        }
    }
}

/// Result of text-to-speech synthesis.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Raw audio data bytes.
    pub audio: Vec<u8>,
    /// Audio format of the returned data.
    pub format: AudioFormat,
    /// Estimated duration in seconds.
    pub duration_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

// ============================================================================
// SpeechProvider Trait
// ============================================================================

/// Unified trait for speech-to-text and text-to-speech providers.
///
/// # Stability
///
/// New methods may be added to this trait in minor versions with default
/// implementations. Required methods will only change in major versions.
pub trait SpeechProvider: Send + Sync {
    /// Provider name (e.g., "openai", "google").
    fn name(&self) -> &str;

    /// Whether this provider supports speech-to-text.
    fn supports_stt(&self) -> bool;

    /// Whether this provider supports text-to-speech.
    fn supports_tts(&self) -> bool;

    /// Available TTS voices for this provider.
    fn tts_voices(&self) -> Vec<String>;

    /// Transcribe audio to text (STT).
    ///
    /// # Arguments
    /// * `audio` - Raw audio bytes
    /// * `format` - Audio encoding format
    /// * `language` - Optional language hint (ISO 639-1)
    fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<&str>,
    ) -> Result<TranscriptionResult>;

    /// Synthesize text to audio (TTS).
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `options` - Synthesis configuration (voice, format, speed)
    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult>;
}

// ============================================================================
// OpenAI Speech Provider (Whisper STT + TTS)
// ============================================================================

/// OpenAI speech provider using Whisper for STT and TTS API for synthesis.
///
/// - STT: `POST https://api.openai.com/v1/audio/transcriptions`
/// - TTS: `POST https://api.openai.com/v1/audio/speech`
#[derive(Debug)]
pub struct OpenAISpeechProvider {
    config: SpeechConfig,
}

impl OpenAISpeechProvider {
    /// Create with an explicit API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            config: SpeechConfig {
                api_key: Some(api_key.to_string()),
                base_url: Some("https://api.openai.com".into()),
                model: Some("whisper-1".into()),
                ..Default::default()
            },
        }
    }

    /// Create from the `OPENAI_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let key =
            std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY not set")?;
        Ok(Self::new(&key))
    }

    /// Set the TTS model (default: "tts-1", also available: "tts-1-hd").
    pub fn with_tts_model(mut self, model: &str) -> Self {
        self.config.model = Some(model.into());
        self
    }

    /// Override the base URL (e.g., for local whisper.cpp server or LM Studio).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.config.base_url = Some(url.to_string());
        self
    }

    fn api_key(&self) -> Result<&str> {
        self.config
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key not configured"))
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com")
    }
}

impl SpeechProvider for OpenAISpeechProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supports_stt(&self) -> bool {
        true
    }

    fn supports_tts(&self) -> bool {
        true
    }

    fn tts_voices(&self) -> Vec<String> {
        vec![
            "alloy".into(),
            "echo".into(),
            "fable".into(),
            "onyx".into(),
            "nova".into(),
            "shimmer".into(),
        ]
    }

    fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let api_key = self.api_key()?;
        let url = format!("{}/v1/audio/transcriptions", self.base_url());

        // Build multipart form manually (ureq v2 does not have native multipart)
        let boundary = format!(
            "----FormBoundary{}",
            uuid::Uuid::new_v4().to_string().replace('-', "")
        );
        let filename = format!("audio.{}", format.extension());

        let mut body = Vec::new();

        // File part
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
                filename
            )
            .as_bytes(),
        );
        body.extend_from_slice(
            format!("Content-Type: {}\r\n\r\n", format.mime_type()).as_bytes(),
        );
        body.extend_from_slice(audio);
        body.extend_from_slice(b"\r\n");

        // Model part
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"model\"\r\n\r\n",
        );
        body.extend_from_slice(b"whisper-1\r\n");

        // Response format part (verbose_json gives us segments)
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n",
        );
        body.extend_from_slice(b"verbose_json\r\n");

        // Language part (optional)
        if let Some(lang) = language {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(
                b"Content-Disposition: form-data; name=\"language\"\r\n\r\n",
            );
            body.extend_from_slice(lang.as_bytes());
            body.extend_from_slice(b"\r\n");
        }

        // End boundary
        body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", api_key))
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", boundary),
            )
            .timeout(self.config.timeout)
            .send_bytes(&body)
            .context("OpenAI transcription request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse OpenAI transcription response")?;

        let text = json["text"].as_str().unwrap_or("").to_string();
        let language_detected = json["language"].as_str().map(|s| s.to_string());
        let duration = json["duration"].as_f64().unwrap_or(0.0);

        let segments = json["segments"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|s| TranscriptionSegment {
                        start_secs: s["start"].as_f64().unwrap_or(0.0),
                        end_secs: s["end"].as_f64().unwrap_or(0.0),
                        text: s["text"].as_str().unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(TranscriptionResult {
            text,
            language: language_detected,
            duration_secs: duration,
            segments,
            confidence: None, // OpenAI does not return overall confidence
        })
    }

    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let api_key = self.api_key()?;
        let url = format!("{}/v1/audio/speech", self.base_url());

        let voice = options.voice.as_deref().unwrap_or("alloy");
        let model = self.config.model.as_deref().unwrap_or("tts-1");

        // Map AudioFormat to OpenAI response_format parameter
        let response_format = match options.format {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
            AudioFormat::Wav => "wav",
            AudioFormat::Pcm => "pcm",
            AudioFormat::Ogg => "mp3", // OpenAI doesn't support raw ogg; fallback
        };

        let body = serde_json::json!({
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": options.speed,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", api_key))
            .set("Content-Type", "application/json")
            .timeout(self.config.timeout)
            .send_json(body)
            .context("OpenAI TTS request failed")?;

        let mut audio = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut audio)
            .context("Failed to read TTS audio response")?;

        // Estimate duration: ~150 words per minute, ~5 chars per word
        let estimated_duration = (text.len() as f64) / (5.0 * 150.0 / 60.0);

        Ok(SynthesisResult {
            audio,
            format: options.format.clone(),
            duration_secs: estimated_duration,
            sample_rate: options.sample_rate.unwrap_or(24000),
        })
    }
}

// ============================================================================
// Google Cloud Speech Provider
// ============================================================================

/// Google Cloud speech provider.
///
/// - STT: `POST https://speech.googleapis.com/v1/speech:recognize`
/// - TTS: `POST https://texttospeech.googleapis.com/v1/text:synthesize`
#[derive(Debug)]
pub struct GoogleSpeechProvider {
    config: SpeechConfig,
}

impl GoogleSpeechProvider {
    /// Create with an explicit API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            config: SpeechConfig {
                api_key: Some(api_key.to_string()),
                ..Default::default()
            },
        }
    }

    /// Create from the `GOOGLE_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let key =
            std::env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
        Ok(Self::new(&key))
    }

    fn api_key(&self) -> Result<&str> {
        self.config
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Google API key not configured"))
    }

    /// Parse Google's duration format (e.g., "1.500s" -> 1.5).
    fn parse_google_duration(s: &str) -> f64 {
        s.trim_end_matches('s').parse().unwrap_or(0.0)
    }
}

impl SpeechProvider for GoogleSpeechProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn supports_stt(&self) -> bool {
        true
    }

    fn supports_tts(&self) -> bool {
        true
    }

    fn tts_voices(&self) -> Vec<String> {
        vec![
            "en-US-Standard-A".into(),
            "en-US-Standard-B".into(),
            "en-US-Standard-C".into(),
            "en-US-Standard-D".into(),
            "en-US-Wavenet-A".into(),
            "en-US-Wavenet-B".into(),
            "en-US-Neural2-A".into(),
            "en-US-Neural2-C".into(),
            "es-ES-Standard-A".into(),
            "es-ES-Standard-B".into(),
        ]
    }

    fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let api_key = self.api_key()?;
        let url = "https://speech.googleapis.com/v1/speech:recognize";

        // Google expects base64-encoded audio content
        let audio_b64 = base64_encode(audio);

        // Map AudioFormat to Google Cloud encoding string
        let encoding = match format {
            AudioFormat::Wav => "LINEAR16",
            AudioFormat::Mp3 => "MP3",
            AudioFormat::Flac => "FLAC",
            AudioFormat::Ogg => "OGG_OPUS",
            AudioFormat::Opus => "OGG_OPUS",
            AudioFormat::Pcm => "LINEAR16",
            AudioFormat::Aac => "MP3", // Google doesn't support AAC; closest
        };

        let lang = language.unwrap_or("en-US");

        let body = serde_json::json!({
            "config": {
                "encoding": encoding,
                "languageCode": lang,
                "enableWordTimeOffsets": true,
                "enableAutomaticPunctuation": true,
            },
            "audio": {
                "content": audio_b64,
            }
        });

        let response = ureq::post(url)
            .set("Content-Type", "application/json")
            .set("x-goog-api-key", &api_key)
            .timeout(self.config.timeout)
            .send_json(body)
            .context("Google STT request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse Google STT response")?;

        // Parse Google's results array
        let results = json["results"].as_array();
        let mut text = String::new();
        let mut segments = Vec::new();
        let mut confidence = None;

        if let Some(results) = results {
            for result in results {
                if let Some(alternatives) = result["alternatives"].as_array() {
                    if let Some(best) = alternatives.first() {
                        let alt_text =
                            best["transcript"].as_str().unwrap_or("");
                        text.push_str(alt_text);
                        text.push(' ');
                        if confidence.is_none() {
                            confidence = best["confidence"].as_f64();
                        }

                        // Parse word-level timestamps into segments
                        if let Some(words) = best["words"].as_array() {
                            for word in words {
                                let start =
                                    Self::parse_google_duration(
                                        word["startTime"]
                                            .as_str()
                                            .unwrap_or("0s"),
                                    );
                                let end = Self::parse_google_duration(
                                    word["endTime"].as_str().unwrap_or("0s"),
                                );
                                segments.push(TranscriptionSegment {
                                    start_secs: start,
                                    end_secs: end,
                                    text: word["word"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: Some(lang.to_string()),
            duration_secs: segments.last().map(|s| s.end_secs).unwrap_or(0.0),
            segments,
            confidence,
        })
    }

    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        let api_key = self.api_key()?;
        let url = "https://texttospeech.googleapis.com/v1/text:synthesize";

        let voice_name =
            options.voice.as_deref().unwrap_or("en-US-Standard-A");

        // Extract language code from voice name (e.g., "en-US" from "en-US-Standard-A")
        let lang_code = if voice_name.len() >= 5 {
            &voice_name[..5]
        } else {
            "en-US"
        };

        // Map AudioFormat to Google Cloud audio encoding
        let audio_encoding = match options.format {
            AudioFormat::Mp3 => "MP3",
            AudioFormat::Wav => "LINEAR16",
            AudioFormat::Ogg => "OGG_OPUS",
            AudioFormat::Opus => "OGG_OPUS",
            AudioFormat::Flac => "LINEAR16",
            AudioFormat::Pcm => "LINEAR16",
            AudioFormat::Aac => "MP3",
        };

        let body = serde_json::json!({
            "input": { "text": text },
            "voice": {
                "languageCode": lang_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": audio_encoding,
                "speakingRate": options.speed,
                "sampleRateHertz": options.sample_rate.unwrap_or(24000),
            }
        });

        let response = ureq::post(url)
            .set("Content-Type", "application/json")
            .set("x-goog-api-key", &api_key)
            .timeout(self.config.timeout)
            .send_json(body)
            .context("Google TTS request failed")?;

        let json: serde_json::Value = response
            .into_json()
            .context("Failed to parse Google TTS response")?;

        let audio_b64 = json["audioContent"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No audioContent in response"))?;
        let audio = base64_decode(audio_b64)?;

        // Estimate duration: ~150 words per minute, ~5 chars per word
        let estimated_duration = (text.len() as f64) / (5.0 * 150.0 / 60.0);

        Ok(SynthesisResult {
            audio,
            format: options.format.clone(),
            duration_secs: estimated_duration,
            sample_rate: options.sample_rate.unwrap_or(24000),
        })
    }
}

// ============================================================================
// Piper TTS Provider (local HTTP server)
// ============================================================================

/// TTS-only provider connecting to a local Piper HTTP server.
///
/// Piper is a fast, local neural text-to-speech system.
/// Expected endpoint: `POST /` with `Content-Type: text/plain`, returns WAV audio.
pub struct PiperTtsProvider {
    base_url: String,
    timeout: Duration,
}

impl PiperTtsProvider {
    /// Create with an explicit base URL.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Create with the default local URL (`http://localhost:5000`).
    pub fn default_local() -> Self {
        Self::new("http://localhost:5000")
    }

    /// Set the HTTP request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if the Piper server is reachable.
    pub fn is_available(&self) -> bool {
        match ureq::get(&self.base_url).timeout(Duration::from_secs(2)).call() {
            Ok(_) => true,
            Err(ureq::Error::Status(400, _)) => true, // Server is up, just rejects empty GET
            Err(_) => false,
        }
    }
}

impl SpeechProvider for PiperTtsProvider {
    fn name(&self) -> &str {
        "piper"
    }

    fn supports_stt(&self) -> bool {
        false
    }

    fn supports_tts(&self) -> bool {
        true
    }

    fn tts_voices(&self) -> Vec<String> {
        vec!["default".into()]
    }

    fn transcribe(
        &self,
        _audio: &[u8],
        _format: AudioFormat,
        _language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        anyhow::bail!("Piper does not support speech-to-text")
    }

    fn synthesize(
        &self,
        text: &str,
        _options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        if text.is_empty() {
            anyhow::bail!("Cannot synthesize empty text");
        }

        let response = ureq::post(&self.base_url)
            .set("Content-Type", "text/plain")
            .timeout(self.timeout)
            .send_string(text)
            .context("Piper TTS request failed")?;

        let mut audio = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut audio)
            .context("Failed to read Piper TTS audio response")?;

        // WAV header is 44 bytes; estimate duration from PCM data size
        // Assuming 16-bit mono at 22050 Hz (Piper default)
        let sample_rate = 22050u32;
        let data_size = if audio.len() > 44 { audio.len() - 44 } else { 0 };
        let duration_secs = data_size as f64 / (sample_rate as f64 * 2.0);

        Ok(SynthesisResult {
            audio,
            format: AudioFormat::Wav,
            duration_secs,
            sample_rate,
        })
    }
}

// ============================================================================
// Coqui TTS Provider (local HTTP server)
// ============================================================================

/// TTS-only provider connecting to a local Coqui TTS server.
///
/// Coqui TTS is a deep-learning toolkit for text-to-speech.
/// Expected endpoint: `GET /api/tts?text=...` returns WAV audio.
pub struct CoquiTtsProvider {
    base_url: String,
    timeout: Duration,
    speaker_id: Option<String>,
    language_id: Option<String>,
}

impl CoquiTtsProvider {
    /// Create with an explicit base URL.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(30),
            speaker_id: None,
            language_id: None,
        }
    }

    /// Create with the default local URL (`http://localhost:5002`).
    pub fn default_local() -> Self {
        Self::new("http://localhost:5002")
    }

    /// Set the speaker ID for multi-speaker models.
    pub fn with_speaker_id(mut self, speaker_id: &str) -> Self {
        self.speaker_id = Some(speaker_id.to_string());
        self
    }

    /// Set the language ID for multi-language models.
    pub fn with_language_id(mut self, language_id: &str) -> Self {
        self.language_id = Some(language_id.to_string());
        self
    }

    /// Set the HTTP request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if the Coqui TTS server is reachable.
    pub fn is_available(&self) -> bool {
        ureq::get(&self.base_url)
            .timeout(Duration::from_secs(2))
            .call()
            .is_ok()
    }
}

impl SpeechProvider for CoquiTtsProvider {
    fn name(&self) -> &str {
        "coqui"
    }

    fn supports_stt(&self) -> bool {
        false
    }

    fn supports_tts(&self) -> bool {
        true
    }

    fn tts_voices(&self) -> Vec<String> {
        vec!["default".into()]
    }

    fn transcribe(
        &self,
        _audio: &[u8],
        _format: AudioFormat,
        _language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        anyhow::bail!("Coqui TTS does not support speech-to-text")
    }

    fn synthesize(
        &self,
        text: &str,
        _options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        if text.is_empty() {
            anyhow::bail!("Cannot synthesize empty text");
        }

        let encoded_text = urlencoding::encode(text);
        let mut url = format!("{}/api/tts?text={}", self.base_url, encoded_text);

        if let Some(ref speaker) = self.speaker_id {
            url.push_str(&format!("&speaker_id={}", urlencoding::encode(speaker)));
        }
        if let Some(ref language) = self.language_id {
            url.push_str(&format!("&language_id={}", urlencoding::encode(language)));
        }

        let response = ureq::get(&url)
            .timeout(self.timeout)
            .call()
            .context("Coqui TTS request failed")?;

        let mut audio = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut audio)
            .context("Failed to read Coqui TTS audio response")?;

        // WAV header is 44 bytes; estimate duration from PCM data size
        // Assuming 16-bit mono at 22050 Hz (Coqui default)
        let sample_rate = 22050u32;
        let data_size = if audio.len() > 44 { audio.len() - 44 } else { 0 };
        let duration_secs = data_size as f64 / (sample_rate as f64 * 2.0);

        Ok(SynthesisResult {
            audio,
            format: AudioFormat::Wav,
            duration_secs,
            sample_rate,
        })
    }
}

// ============================================================================
// Local Speech Provider (composite STT + TTS)
// ============================================================================

/// Composite provider that delegates STT and TTS to separate local providers.
///
/// Useful for combining e.g. Whisper (STT) with Piper (TTS) into a single
/// `SpeechProvider` interface.
pub struct LocalSpeechProvider {
    stt: Option<Box<dyn SpeechProvider>>,
    tts: Option<Box<dyn SpeechProvider>>,
}

impl LocalSpeechProvider {
    /// Create with both STT and TTS providers.
    pub fn new(stt: Box<dyn SpeechProvider>, tts: Box<dyn SpeechProvider>) -> Self {
        Self {
            stt: Some(stt),
            tts: Some(tts),
        }
    }

    /// Create with only an STT provider (no TTS).
    pub fn stt_only(stt: Box<dyn SpeechProvider>) -> Self {
        Self {
            stt: Some(stt),
            tts: None,
        }
    }

    /// Create with only a TTS provider (no STT).
    pub fn tts_only(tts: Box<dyn SpeechProvider>) -> Self {
        Self {
            stt: None,
            tts: Some(tts),
        }
    }
}

impl SpeechProvider for LocalSpeechProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn supports_stt(&self) -> bool {
        self.stt.as_ref().map_or(false, |p| p.supports_stt())
    }

    fn supports_tts(&self) -> bool {
        self.tts.as_ref().map_or(false, |p| p.supports_tts())
    }

    fn tts_voices(&self) -> Vec<String> {
        self.tts
            .as_ref()
            .map_or_else(Vec::new, |p| p.tts_voices())
    }

    fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        match &self.stt {
            Some(provider) => provider.transcribe(audio, format, language),
            None => anyhow::bail!("No STT provider configured in LocalSpeechProvider"),
        }
    }

    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        match &self.tts {
            Some(provider) => provider.synthesize(text, options),
            None => anyhow::bail!("No TTS provider configured in LocalSpeechProvider"),
        }
    }
}

// ============================================================================
// Whisper Local Provider (native whisper-rs, feature-gated)
// ============================================================================

#[cfg(feature = "whisper-local")]
/// STT-only provider using whisper-rs for local Whisper model inference.
///
/// Requires a GGML model file (`.bin`). Download from:
/// <https://huggingface.co/ggerganov/whisper.cpp/tree/main>
pub struct WhisperLocalProvider {
    model_path: String,
    language: Option<String>,
    n_threads: i32,
}

#[cfg(feature = "whisper-local")]
impl WhisperLocalProvider {
    /// Create with a path to the GGML model file.
    pub fn new(model_path: &str) -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8);
        Self {
            model_path: model_path.to_string(),
            language: None,
            n_threads,
        }
    }

    /// Set the language hint (ISO 639-1 code, e.g. "en", "es").
    pub fn with_language(mut self, lang: &str) -> Self {
        self.language = Some(lang.to_string());
        self
    }

    /// Set the number of threads for inference.
    pub fn with_threads(mut self, n: i32) -> Self {
        self.n_threads = n;
        self
    }

    /// Convert raw audio bytes to f32 PCM samples normalized to [-1.0, 1.0].
    fn audio_to_f32_pcm(audio: &[u8], format: AudioFormat) -> Result<Vec<f32>> {
        match format {
            AudioFormat::Wav => {
                // Validate RIFF/WAVE header
                if audio.len() < 44 {
                    anyhow::bail!("WAV file too short (less than 44 bytes)");
                }
                if &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
                    anyhow::bail!("Invalid WAV header (missing RIFF/WAVE markers)");
                }

                // Find the "data" subchunk by scanning for the marker
                let mut data_offset = None;
                for i in 12..audio.len().saturating_sub(8) {
                    if &audio[i..i + 4] == b"data" {
                        data_offset = Some(i + 8); // skip "data" + 4-byte size
                        break;
                    }
                }
                let offset = data_offset
                    .ok_or_else(|| anyhow::anyhow!("WAV data subchunk not found"))?;

                let pcm_data = &audio[offset..];
                let samples: Vec<f32> = pcm_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect();
                Ok(samples)
            }
            AudioFormat::Pcm => {
                // Raw 16-bit little-endian PCM
                let samples: Vec<f32> = audio
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect();
                Ok(samples)
            }
            _ => anyhow::bail!(
                "WhisperLocalProvider only supports WAV and PCM formats, got {:?}",
                format
            ),
        }
    }
}

#[cfg(feature = "whisper-local")]
impl SpeechProvider for WhisperLocalProvider {
    fn name(&self) -> &str {
        "whisper-local"
    }

    fn supports_stt(&self) -> bool {
        true
    }

    fn supports_tts(&self) -> bool {
        false
    }

    fn tts_voices(&self) -> Vec<String> {
        vec![]
    }

    fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let samples = Self::audio_to_f32_pcm(audio, format)?;

        let ctx = WhisperContext::new_with_params(
            &self.model_path,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(self.n_threads);

        let lang = language.or(self.language.as_deref());
        if let Some(lang) = lang {
            params.set_language(Some(lang));
        }

        state
            .full(params, &samples)
            .map_err(|e| anyhow::anyhow!("Whisper inference failed: {}", e))?;

        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segment count: {}", e))?;

        let mut text = String::new();
        let mut segments = Vec::new();

        for i in 0..num_segments {
            let seg_text = state
                .full_get_segment_text(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment {} text: {}", i, e))?;
            let t0 = state
                .full_get_segment_t0(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment {} start time: {}", i, e))?;
            let t1 = state
                .full_get_segment_t1(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment {} end time: {}", i, e))?;

            text.push_str(&seg_text);

            segments.push(TranscriptionSegment {
                start_secs: t0 as f64 / 100.0,
                end_secs: t1 as f64 / 100.0,
                text: seg_text,
            });
        }

        let duration_secs = segments.last().map(|s| s.end_secs).unwrap_or(0.0);

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: lang.map(|s| s.to_string()),
            duration_secs,
            segments,
            confidence: None,
        })
    }

    fn synthesize(
        &self,
        _text: &str,
        _options: &SynthesisOptions,
    ) -> Result<SynthesisResult> {
        anyhow::bail!("WhisperLocalProvider does not support text-to-speech")
    }
}

// ============================================================================
// Factory
// ============================================================================

/// Create a speech provider by name, resolving API keys from environment.
///
/// Supported providers: `"openai"`, `"google"`, `"piper"`, `"coqui"`,
/// `"whisper"` / `"whisper-local"` (requires `whisper-local` feature),
/// `"local"` (auto-detect best available local providers).
pub fn create_speech_provider(
    name: &str,
) -> Result<Box<dyn SpeechProvider>> {
    match name.to_lowercase().as_str() {
        "openai" => Ok(Box::new(OpenAISpeechProvider::from_env()?)),
        "google" => Ok(Box::new(GoogleSpeechProvider::from_env()?)),
        "piper" => Ok(Box::new(PiperTtsProvider::default_local())),
        "coqui" => Ok(Box::new(CoquiTtsProvider::default_local())),
        #[cfg(feature = "whisper-local")]
        "whisper" | "whisper-local" => {
            let model_path = std::env::var("WHISPER_MODEL_PATH")
                .context("WHISPER_MODEL_PATH not set (path to .bin model file)")?;
            Ok(Box::new(WhisperLocalProvider::new(&model_path)))
        }
        #[cfg(not(feature = "whisper-local"))]
        "whisper" | "whisper-local" => {
            anyhow::bail!(
                "Whisper local provider requires the 'whisper-local' feature flag. \
                 Rebuild with: cargo build --features whisper-local"
            )
        }
        "local" => {
            let tts: Box<dyn SpeechProvider> = if PiperTtsProvider::default_local().is_available() {
                Box::new(PiperTtsProvider::default_local())
            } else if CoquiTtsProvider::default_local().is_available() {
                Box::new(CoquiTtsProvider::default_local())
            } else {
                anyhow::bail!(
                    "No local TTS server found. Start Piper (port 5000) or Coqui (port 5002)."
                )
            };
            #[cfg(feature = "whisper-local")]
            {
                if let Ok(model_path) = std::env::var("WHISPER_MODEL_PATH") {
                    let stt = Box::new(WhisperLocalProvider::new(&model_path));
                    return Ok(Box::new(LocalSpeechProvider::new(stt, tts)));
                }
            }
            Ok(Box::new(LocalSpeechProvider::tts_only(tts)))
        }
        _ => anyhow::bail!(
            "Unknown speech provider: {}. Available: openai, google, piper, coqui, whisper, local",
            name
        ),
    }
}

// ============================================================================
// Base64 helpers (inline, no external dependency)
// ============================================================================

const B64_CHARS: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(data: &str) -> Result<Vec<u8>> {
    let data = data.trim();
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(data.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;

    for c in data.bytes() {
        let val = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => {
                anyhow::bail!("Invalid base64 character: {}", c as char)
            }
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // AudioFormat tests
    // ------------------------------------------------------------------

    #[test]
    fn test_audio_format_extension() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Ogg.extension(), "ogg");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Pcm.extension(), "pcm");
        assert_eq!(AudioFormat::Opus.extension(), "opus");
        assert_eq!(AudioFormat::Aac.extension(), "aac");
    }

    #[test]
    fn test_audio_format_mime_type() {
        assert_eq!(AudioFormat::Wav.mime_type(), "audio/wav");
        assert_eq!(AudioFormat::Mp3.mime_type(), "audio/mpeg");
        assert_eq!(AudioFormat::Ogg.mime_type(), "audio/ogg");
        assert_eq!(AudioFormat::Flac.mime_type(), "audio/flac");
        assert_eq!(AudioFormat::Pcm.mime_type(), "audio/pcm");
        assert_eq!(AudioFormat::Opus.mime_type(), "audio/opus");
        assert_eq!(AudioFormat::Aac.mime_type(), "audio/aac");
    }

    #[test]
    fn test_audio_format_from_extension() {
        assert_eq!(AudioFormat::from_extension("wav"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::from_extension("mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_extension("ogg"), Some(AudioFormat::Ogg));
        assert_eq!(
            AudioFormat::from_extension("flac"),
            Some(AudioFormat::Flac)
        );
        assert_eq!(AudioFormat::from_extension("pcm"), Some(AudioFormat::Pcm));
        assert_eq!(
            AudioFormat::from_extension("opus"),
            Some(AudioFormat::Opus)
        );
        assert_eq!(AudioFormat::from_extension("aac"), Some(AudioFormat::Aac));
        // m4a maps to Aac
        assert_eq!(AudioFormat::from_extension("m4a"), Some(AudioFormat::Aac));
        // Case-insensitive
        assert_eq!(AudioFormat::from_extension("WAV"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::from_extension("MP3"), Some(AudioFormat::Mp3));
    }

    #[test]
    fn test_audio_format_from_extension_invalid() {
        assert_eq!(AudioFormat::from_extension("txt"), None);
        assert_eq!(AudioFormat::from_extension("pdf"), None);
        assert_eq!(AudioFormat::from_extension(""), None);
        assert_eq!(AudioFormat::from_extension("avi"), None);
        assert_eq!(AudioFormat::from_extension("wmv"), None);
    }

    #[test]
    fn test_audio_format_clone_eq() {
        let fmt = AudioFormat::Opus;
        let cloned = fmt.clone();
        assert_eq!(fmt, cloned);
        assert_ne!(AudioFormat::Wav, AudioFormat::Mp3);
    }

    #[test]
    fn test_audio_format_debug() {
        let dbg = format!("{:?}", AudioFormat::Flac);
        assert_eq!(dbg, "Flac");
    }

    // ------------------------------------------------------------------
    // SpeechConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn test_speech_config_default() {
        let config = SpeechConfig::default();
        assert!(config.api_key.is_none());
        assert!(config.base_url.is_none());
        assert!(config.model.is_none());
        assert!(config.language.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_speech_config_custom() {
        let config = SpeechConfig {
            api_key: Some("test-key".into()),
            base_url: Some("http://localhost:8080".into()),
            model: Some("whisper-1".into()),
            language: Some("es".into()),
            timeout: Duration::from_secs(30),
        };
        assert_eq!(config.api_key.as_deref(), Some("test-key"));
        assert_eq!(config.base_url.as_deref(), Some("http://localhost:8080"));
        assert_eq!(config.model.as_deref(), Some("whisper-1"));
        assert_eq!(config.language.as_deref(), Some("es"));
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    // ------------------------------------------------------------------
    // SynthesisOptions tests
    // ------------------------------------------------------------------

    #[test]
    fn test_synthesis_options_default() {
        let opts = SynthesisOptions::default();
        assert!(opts.voice.is_none());
        assert_eq!(opts.format, AudioFormat::Mp3);
        assert!((opts.speed - 1.0).abs() < f64::EPSILON);
        assert!(opts.sample_rate.is_none());
    }

    #[test]
    fn test_synthesis_options_clone() {
        let opts = SynthesisOptions {
            voice: Some("alloy".into()),
            format: AudioFormat::Wav,
            speed: 1.5,
            sample_rate: Some(16000),
        };
        let cloned = opts.clone();
        assert_eq!(cloned.voice.as_deref(), Some("alloy"));
        assert_eq!(cloned.format, AudioFormat::Wav);
        assert!((cloned.speed - 1.5).abs() < f64::EPSILON);
        assert_eq!(cloned.sample_rate, Some(16000));
    }

    // ------------------------------------------------------------------
    // TranscriptionResult / TranscriptionSegment tests
    // ------------------------------------------------------------------

    #[test]
    fn test_transcription_result_fields() {
        let result = TranscriptionResult {
            text: "Hello world".into(),
            language: Some("en".into()),
            duration_secs: 2.5,
            segments: vec![],
            confidence: Some(0.95),
        };
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!((result.duration_secs - 2.5).abs() < f64::EPSILON);
        assert!(result.segments.is_empty());
        assert!((result.confidence.unwrap() - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transcription_segment_fields() {
        let seg = TranscriptionSegment {
            start_secs: 0.5,
            end_secs: 1.2,
            text: "hello".into(),
        };
        assert!((seg.start_secs - 0.5).abs() < f64::EPSILON);
        assert!((seg.end_secs - 1.2).abs() < f64::EPSILON);
        assert_eq!(seg.text, "hello");
    }

    #[test]
    fn test_transcription_result_serialization() {
        let result = TranscriptionResult {
            text: "Test".into(),
            language: None,
            duration_secs: 1.0,
            segments: vec![TranscriptionSegment {
                start_secs: 0.0,
                end_secs: 1.0,
                text: "Test".into(),
            }],
            confidence: Some(0.9),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: TranscriptionResult =
            serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "Test");
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].text, "Test");
    }

    // ------------------------------------------------------------------
    // SynthesisResult tests
    // ------------------------------------------------------------------

    #[test]
    fn test_synthesis_result_fields() {
        let result = SynthesisResult {
            audio: vec![0xFF, 0xD8, 0x00],
            format: AudioFormat::Mp3,
            duration_secs: 3.14,
            sample_rate: 44100,
        };
        assert_eq!(result.audio.len(), 3);
        assert_eq!(result.format, AudioFormat::Mp3);
        assert!((result.duration_secs - 3.14).abs() < f64::EPSILON);
        assert_eq!(result.sample_rate, 44100);
    }

    #[test]
    fn test_synthesis_result_clone() {
        let result = SynthesisResult {
            audio: vec![1, 2, 3, 4],
            format: AudioFormat::Wav,
            duration_secs: 1.0,
            sample_rate: 16000,
        };
        let cloned = result.clone();
        assert_eq!(cloned.audio, vec![1, 2, 3, 4]);
        assert_eq!(cloned.format, AudioFormat::Wav);
        assert_eq!(cloned.sample_rate, 16000);
    }

    // ------------------------------------------------------------------
    // OpenAI provider tests
    // ------------------------------------------------------------------

    #[test]
    fn test_openai_provider_name() {
        let provider = OpenAISpeechProvider::new("test-key");
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_openai_supports_stt() {
        let provider = OpenAISpeechProvider::new("test-key");
        assert!(provider.supports_stt());
    }

    #[test]
    fn test_openai_supports_tts() {
        let provider = OpenAISpeechProvider::new("test-key");
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_openai_tts_voices() {
        let provider = OpenAISpeechProvider::new("test-key");
        let voices = provider.tts_voices();
        assert_eq!(voices.len(), 6);
        assert!(voices.contains(&"alloy".to_string()));
        assert!(voices.contains(&"echo".to_string()));
        assert!(voices.contains(&"fable".to_string()));
        assert!(voices.contains(&"onyx".to_string()));
        assert!(voices.contains(&"nova".to_string()));
        assert!(voices.contains(&"shimmer".to_string()));
    }

    #[test]
    fn test_openai_with_tts_model() {
        let provider =
            OpenAISpeechProvider::new("key").with_tts_model("tts-1-hd");
        assert_eq!(
            provider.config.model.as_deref(),
            Some("tts-1-hd")
        );
    }

    #[test]
    fn test_openai_base_url_default() {
        let provider = OpenAISpeechProvider::new("key");
        assert_eq!(provider.base_url(), "https://api.openai.com");
    }

    #[test]
    fn test_openai_api_key_configured() {
        let provider = OpenAISpeechProvider::new("my-secret-key");
        assert_eq!(provider.api_key().unwrap(), "my-secret-key");
    }

    #[test]
    fn test_openai_from_env_no_key() {
        // Ensure the env var is not set for this test
        std::env::remove_var("OPENAI_API_KEY");
        let result = OpenAISpeechProvider::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("OPENAI_API_KEY"),
            "Error should mention OPENAI_API_KEY: {}",
            err
        );
    }

    // ------------------------------------------------------------------
    // Google provider tests
    // ------------------------------------------------------------------

    #[test]
    fn test_google_provider_name() {
        let provider = GoogleSpeechProvider::new("test-key");
        assert_eq!(provider.name(), "google");
    }

    #[test]
    fn test_google_supports_stt() {
        let provider = GoogleSpeechProvider::new("test-key");
        assert!(provider.supports_stt());
    }

    #[test]
    fn test_google_supports_tts() {
        let provider = GoogleSpeechProvider::new("test-key");
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_google_tts_voices() {
        let provider = GoogleSpeechProvider::new("test-key");
        let voices = provider.tts_voices();
        assert_eq!(voices.len(), 10);
        assert!(voices.contains(&"en-US-Standard-A".to_string()));
        assert!(voices.contains(&"en-US-Standard-B".to_string()));
        assert!(voices.contains(&"en-US-Standard-C".to_string()));
        assert!(voices.contains(&"en-US-Standard-D".to_string()));
        assert!(voices.contains(&"en-US-Wavenet-A".to_string()));
        assert!(voices.contains(&"en-US-Wavenet-B".to_string()));
        assert!(voices.contains(&"en-US-Neural2-A".to_string()));
        assert!(voices.contains(&"en-US-Neural2-C".to_string()));
        assert!(voices.contains(&"es-ES-Standard-A".to_string()));
        assert!(voices.contains(&"es-ES-Standard-B".to_string()));
    }

    #[test]
    fn test_google_parse_duration() {
        assert!(
            (GoogleSpeechProvider::parse_google_duration("1.500s") - 1.5)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (GoogleSpeechProvider::parse_google_duration("0s") - 0.0).abs()
                < f64::EPSILON
        );
        assert!(
            (GoogleSpeechProvider::parse_google_duration("123.456s")
                - 123.456)
                .abs()
                < 1e-10
        );
        assert!(
            (GoogleSpeechProvider::parse_google_duration("0.001s") - 0.001)
                .abs()
                < f64::EPSILON
        );
        // Edge case: no "s" suffix
        assert!(
            (GoogleSpeechProvider::parse_google_duration("5.0") - 5.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_google_parse_duration_invalid() {
        // Non-numeric should fall back to 0.0
        assert!(
            (GoogleSpeechProvider::parse_google_duration("invalid") - 0.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (GoogleSpeechProvider::parse_google_duration("") - 0.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_google_api_key_configured() {
        let provider = GoogleSpeechProvider::new("google-key");
        assert_eq!(provider.api_key().unwrap(), "google-key");
    }

    #[test]
    fn test_google_from_env_no_key() {
        std::env::remove_var("GOOGLE_API_KEY");
        let result = GoogleSpeechProvider::from_env();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("GOOGLE_API_KEY"),
            "Error should mention GOOGLE_API_KEY: {}",
            err
        );
    }

    // ------------------------------------------------------------------
    // Factory tests
    // ------------------------------------------------------------------

    #[test]
    fn test_create_speech_provider_unknown() {
        let result = create_speech_provider("unknown-provider");
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("Unknown speech provider"));
        assert!(err.contains("unknown-provider"));
    }

    #[test]
    fn test_create_speech_provider_openai_no_key() {
        std::env::remove_var("OPENAI_API_KEY");
        let result = create_speech_provider("openai");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_speech_provider_google_no_key() {
        std::env::remove_var("GOOGLE_API_KEY");
        let result = create_speech_provider("google");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_speech_provider_case_insensitive() {
        std::env::remove_var("OPENAI_API_KEY");
        // Should recognize "OpenAI" (mixed case) and fail on missing key,
        // not on unknown provider
        let result = create_speech_provider("OpenAI");
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        // Should not say "Unknown speech provider"
        assert!(!err.contains("Unknown speech provider"));
    }

    // ------------------------------------------------------------------
    // Base64 tests
    // ------------------------------------------------------------------

    #[test]
    fn test_base64_encode_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn test_base64_encode_hello() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_base64_encode_padding() {
        // 1 byte -> 2 chars + "==" padding
        assert_eq!(base64_encode(b"A"), "QQ==");
        // 2 bytes -> 3 chars + "=" padding
        assert_eq!(base64_encode(b"AB"), "QUI=");
        // 3 bytes -> 4 chars, no padding
        assert_eq!(base64_encode(b"ABC"), "QUJD");
    }

    #[test]
    fn test_base64_encode_binary() {
        // All zeros
        assert_eq!(base64_encode(&[0, 0, 0]), "AAAA");
        // All 0xFF
        assert_eq!(base64_encode(&[255, 255, 255]), "////");
    }

    #[test]
    fn test_base64_decode_empty() {
        let decoded = base64_decode("").unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_base64_decode_hello() {
        let decoded = base64_decode("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_roundtrip() {
        let inputs: &[&[u8]] = &[
            b"",
            b"a",
            b"ab",
            b"abc",
            b"Hello, World!",
            b"\x00\x01\x02\xFF\xFE\xFD",
            b"The quick brown fox jumps over the lazy dog",
        ];
        for input in inputs {
            let encoded = base64_encode(input);
            let decoded = base64_decode(&encoded).unwrap();
            assert_eq!(
                &decoded, input,
                "Roundtrip failed for input of length {}",
                input.len()
            );
        }
    }

    #[test]
    fn test_base64_decode_invalid_char() {
        let result = base64_decode("SGVs!G8=");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid base64 character"));
    }

    #[test]
    fn test_base64_decode_whitespace_ignored() {
        // Whitespace should be skipped
        let decoded = base64_decode("SGVs\nbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_decode_padding_stripped() {
        // Whitespace around input
        let decoded = base64_decode("  SGVsbG8=  ").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    // ------------------------------------------------------------------
    // OpenAI with_base_url tests
    // ------------------------------------------------------------------

    #[test]
    fn test_openai_with_base_url() {
        let provider = OpenAISpeechProvider::new("key")
            .with_base_url("http://localhost:8080");
        assert_eq!(
            provider.config.base_url.as_deref(),
            Some("http://localhost:8080")
        );
    }

    #[test]
    fn test_openai_with_base_url_chain() {
        let provider = OpenAISpeechProvider::new("key")
            .with_base_url("http://localhost:8080")
            .with_tts_model("tts-1-hd");
        assert_eq!(
            provider.config.base_url.as_deref(),
            Some("http://localhost:8080")
        );
        assert_eq!(provider.config.model.as_deref(), Some("tts-1-hd"));
    }

    #[test]
    fn test_openai_with_base_url_used_in_url() {
        let provider = OpenAISpeechProvider::new("key")
            .with_base_url("http://localhost:8080");
        assert_eq!(provider.base_url(), "http://localhost:8080");
    }

    // ------------------------------------------------------------------
    // PiperTtsProvider tests
    // ------------------------------------------------------------------

    #[test]
    fn test_piper_provider_name() {
        let provider = PiperTtsProvider::default_local();
        assert_eq!(provider.name(), "piper");
    }

    #[test]
    fn test_piper_supports_stt_false() {
        let provider = PiperTtsProvider::default_local();
        assert!(!provider.supports_stt());
    }

    #[test]
    fn test_piper_supports_tts_true() {
        let provider = PiperTtsProvider::default_local();
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_piper_tts_voices() {
        let provider = PiperTtsProvider::default_local();
        assert_eq!(provider.tts_voices(), vec!["default".to_string()]);
    }

    #[test]
    fn test_piper_transcribe_errors() {
        let provider = PiperTtsProvider::default_local();
        let result = provider.transcribe(b"audio", AudioFormat::Wav, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_piper_synthesize_empty_text() {
        let provider = PiperTtsProvider::default_local();
        let result = provider.synthesize("", &SynthesisOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty text"));
    }

    #[test]
    fn test_piper_default_local_url() {
        let provider = PiperTtsProvider::default_local();
        assert!(
            provider.base_url.contains("5000"),
            "Piper default URL should use port 5000, got: {}",
            provider.base_url
        );
    }

    // ------------------------------------------------------------------
    // CoquiTtsProvider tests
    // ------------------------------------------------------------------

    #[test]
    fn test_coqui_provider_name() {
        let provider = CoquiTtsProvider::default_local();
        assert_eq!(provider.name(), "coqui");
    }

    #[test]
    fn test_coqui_supports_stt_false() {
        let provider = CoquiTtsProvider::default_local();
        assert!(!provider.supports_stt());
    }

    #[test]
    fn test_coqui_supports_tts_true() {
        let provider = CoquiTtsProvider::default_local();
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_coqui_tts_voices() {
        let provider = CoquiTtsProvider::default_local();
        assert_eq!(provider.tts_voices(), vec!["default".to_string()]);
    }

    #[test]
    fn test_coqui_transcribe_errors() {
        let provider = CoquiTtsProvider::default_local();
        let result = provider.transcribe(b"audio", AudioFormat::Wav, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_coqui_synthesize_empty_text() {
        let provider = CoquiTtsProvider::default_local();
        let result = provider.synthesize("", &SynthesisOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty text"));
    }

    #[test]
    fn test_coqui_default_local_url() {
        let provider = CoquiTtsProvider::default_local();
        assert!(
            provider.base_url.contains("5002"),
            "Coqui default URL should use port 5002, got: {}",
            provider.base_url
        );
    }

    #[test]
    fn test_coqui_with_speaker_and_language() {
        let provider = CoquiTtsProvider::default_local()
            .with_speaker_id("speaker-1")
            .with_language_id("en");
        assert_eq!(provider.speaker_id.as_deref(), Some("speaker-1"));
        assert_eq!(provider.language_id.as_deref(), Some("en"));
    }

    // ------------------------------------------------------------------
    // LocalSpeechProvider tests
    // ------------------------------------------------------------------

    struct MockSttProvider;
    impl SpeechProvider for MockSttProvider {
        fn name(&self) -> &str {
            "mock-stt"
        }
        fn supports_stt(&self) -> bool {
            true
        }
        fn supports_tts(&self) -> bool {
            false
        }
        fn tts_voices(&self) -> Vec<String> {
            vec![]
        }
        fn transcribe(
            &self,
            _audio: &[u8],
            _format: AudioFormat,
            _language: Option<&str>,
        ) -> Result<TranscriptionResult> {
            Ok(TranscriptionResult {
                text: "hello world".to_string(),
                language: Some("en".to_string()),
                duration_secs: 1.5,
                segments: vec![],
                confidence: Some(0.95),
            })
        }
        fn synthesize(
            &self,
            _text: &str,
            _options: &SynthesisOptions,
        ) -> Result<SynthesisResult> {
            anyhow::bail!("Mock STT does not support TTS")
        }
    }

    struct MockTtsProvider;
    impl SpeechProvider for MockTtsProvider {
        fn name(&self) -> &str {
            "mock-tts"
        }
        fn supports_stt(&self) -> bool {
            false
        }
        fn supports_tts(&self) -> bool {
            true
        }
        fn tts_voices(&self) -> Vec<String> {
            vec!["mock-voice".to_string()]
        }
        fn transcribe(
            &self,
            _audio: &[u8],
            _format: AudioFormat,
            _language: Option<&str>,
        ) -> Result<TranscriptionResult> {
            anyhow::bail!("Mock TTS does not support STT")
        }
        fn synthesize(
            &self,
            _text: &str,
            _options: &SynthesisOptions,
        ) -> Result<SynthesisResult> {
            Ok(SynthesisResult {
                audio: vec![0u8; 100],
                format: AudioFormat::Wav,
                duration_secs: 2.0,
                sample_rate: 22050,
            })
        }
    }

    #[test]
    fn test_local_provider_name() {
        let provider = LocalSpeechProvider::new(
            Box::new(MockSttProvider),
            Box::new(MockTtsProvider),
        );
        assert_eq!(provider.name(), "local");
    }

    #[test]
    fn test_local_stt_support() {
        let provider = LocalSpeechProvider::stt_only(Box::new(MockSttProvider));
        assert!(provider.supports_stt());
        assert!(!provider.supports_tts());
    }

    #[test]
    fn test_local_tts_support() {
        let provider = LocalSpeechProvider::tts_only(Box::new(MockTtsProvider));
        assert!(!provider.supports_stt());
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_local_both_support() {
        let provider = LocalSpeechProvider::new(
            Box::new(MockSttProvider),
            Box::new(MockTtsProvider),
        );
        assert!(provider.supports_stt());
        assert!(provider.supports_tts());
    }

    #[test]
    fn test_local_tts_voices_delegates() {
        let provider = LocalSpeechProvider::tts_only(Box::new(MockTtsProvider));
        let voices = provider.tts_voices();
        assert_eq!(voices, vec!["mock-voice".to_string()]);
    }

    #[test]
    fn test_local_transcribe_delegates() {
        let provider = LocalSpeechProvider::stt_only(Box::new(MockSttProvider));
        let result = provider
            .transcribe(b"audio", AudioFormat::Wav, None)
            .unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!((result.duration_secs - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_local_synthesize_delegates() {
        let provider = LocalSpeechProvider::tts_only(Box::new(MockTtsProvider));
        let result = provider
            .synthesize("test", &SynthesisOptions::default())
            .unwrap();
        assert_eq!(result.audio.len(), 100);
        assert_eq!(result.format, AudioFormat::Wav);
        assert_eq!(result.sample_rate, 22050);
    }

    #[test]
    fn test_local_no_stt_errors() {
        let provider = LocalSpeechProvider::tts_only(Box::new(MockTtsProvider));
        let result = provider.transcribe(b"audio", AudioFormat::Wav, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No STT provider"));
    }

    #[test]
    fn test_local_no_tts_errors() {
        let provider = LocalSpeechProvider::stt_only(Box::new(MockSttProvider));
        let result = provider.synthesize("test", &SynthesisOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No TTS provider"));
    }

    // ------------------------------------------------------------------
    // WhisperLocalProvider tests (feature-gated)
    // ------------------------------------------------------------------

    #[cfg(feature = "whisper-local")]
    #[test]
    fn test_whisper_local_name() {
        let provider = WhisperLocalProvider::new("/tmp/model.bin");
        assert_eq!(provider.name(), "whisper-local");
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn test_whisper_local_supports_stt() {
        let provider = WhisperLocalProvider::new("/tmp/model.bin");
        assert!(provider.supports_stt());
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn test_whisper_local_no_tts() {
        let provider = WhisperLocalProvider::new("/tmp/model.bin");
        assert!(!provider.supports_tts());
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn test_whisper_local_synthesize_errors() {
        let provider = WhisperLocalProvider::new("/tmp/model.bin");
        let result = provider.synthesize("hello", &SynthesisOptions::default());
        assert!(result.is_err());
    }

    #[cfg(feature = "whisper-local")]
    #[test]
    fn test_whisper_local_pcm_to_f32() {
        // Create synthetic PCM data: two 16-bit LE samples
        // Sample 1: 0x0040 = 16384 -> 16384/32768 = 0.5
        // Sample 2: 0xC000 = -16384 (signed) -> -16384/32768 = -0.5
        let pcm_data: Vec<u8> = vec![0x00, 0x40, 0x00, 0xC0];
        let samples =
            WhisperLocalProvider::audio_to_f32_pcm(&pcm_data, AudioFormat::Pcm)
                .unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 1e-4);
        assert!((samples[1] - (-0.5)).abs() < 1e-4);
    }

    // ------------------------------------------------------------------
    // Updated factory tests
    // ------------------------------------------------------------------

    #[test]
    fn test_create_speech_provider_piper() {
        let result = create_speech_provider("piper");
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.name(), "piper");
    }

    #[test]
    fn test_create_speech_provider_coqui() {
        let result = create_speech_provider("coqui");
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.name(), "coqui");
    }

    #[cfg(not(feature = "whisper-local"))]
    #[test]
    fn test_create_speech_provider_whisper_without_feature() {
        let result = create_speech_provider("whisper");
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("whisper-local"),
            "Error should mention whisper-local feature: {}",
            err
        );
    }

    #[test]
    fn test_create_speech_provider_updated_available_list() {
        let result = create_speech_provider("nonexistent");
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("piper"), "Error should list piper: {}", err);
        assert!(err.contains("coqui"), "Error should list coqui: {}", err);
        assert!(
            err.contains("whisper"),
            "Error should list whisper: {}",
            err
        );
    }
}
