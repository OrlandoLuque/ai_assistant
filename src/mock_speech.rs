//! Mock speech provider for testing STT/TTS pipeline without real audio services.
//!
//! `MockSpeechProvider` implements `SpeechProvider` with deterministic behavior:
//! - STT: returns audio duration and sample count as text
//! - TTS: generates valid PCM16 sine wave audio
//!
//! Feature-gated behind `audio`.

#[cfg(feature = "audio")]
use crate::speech::{
    AudioFormat, SpeechProvider, SynthesisOptions, SynthesisResult, TranscriptionResult,
    TranscriptionSegment,
};

/// Mock speech provider for testing. No external services needed.
#[cfg(feature = "audio")]
pub struct MockSpeechProvider {
    /// Fixed transcript to return for any STT call. If None, generates from audio metadata.
    pub fixed_transcript: Option<String>,
    /// Sample rate for generated TTS audio.
    pub tts_sample_rate: u32,
    /// Duration in seconds for generated TTS audio.
    pub tts_duration_secs: f64,
}

#[cfg(feature = "audio")]
impl Default for MockSpeechProvider {
    fn default() -> Self {
        Self {
            fixed_transcript: None,
            tts_sample_rate: 16000,
            tts_duration_secs: 1.0,
        }
    }
}

#[cfg(feature = "audio")]
impl MockSpeechProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a fixed transcript for STT.
    pub fn with_transcript(transcript: &str) -> Self {
        Self {
            fixed_transcript: Some(transcript.to_string()),
            ..Default::default()
        }
    }

    /// Generate a sine wave PCM16 audio buffer.
    fn generate_sine_wave(&self, frequency: f32, duration_secs: f64, sample_rate: u32) -> Vec<u8> {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        let mut bytes = Vec::with_capacity(num_samples * 2);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
            let pcm = (sample * 16000.0) as i16;
            bytes.extend_from_slice(&pcm.to_le_bytes());
        }
        bytes
    }
}

#[cfg(feature = "audio")]
impl SpeechProvider for MockSpeechProvider {
    fn transcribe(
        &self,
        audio: &[u8],
        _format: AudioFormat,
        _language: Option<&str>,
    ) -> anyhow::Result<TranscriptionResult> {
        let num_samples = audio.len() / 2; // PCM16
        let duration = num_samples as f64 / 16000.0;

        let text = self.fixed_transcript.clone().unwrap_or_else(|| {
            format!(
                "Mock transcript ({:.1}s, {} samples)",
                duration, num_samples
            )
        });

        Ok(TranscriptionResult {
            text: text.clone(),
            language: Some("en".to_string()),
            duration_secs: duration,
            segments: vec![TranscriptionSegment {
                start_secs: 0.0,
                end_secs: duration,
                text,
            }],
            confidence: Some(0.95),
        })
    }

    fn synthesize(
        &self,
        text: &str,
        _options: &SynthesisOptions,
    ) -> anyhow::Result<SynthesisResult> {
        // Generate a valid sine wave (440Hz = A4 note)
        let audio = self.generate_sine_wave(440.0, self.tts_duration_secs, self.tts_sample_rate);

        Ok(SynthesisResult {
            audio,
            format: AudioFormat::Pcm,
            duration_secs: self.tts_duration_secs,
            sample_rate: self.tts_sample_rate,
        })
    }

    fn supports_stt(&self) -> bool {
        true
    }

    fn supports_tts(&self) -> bool {
        true
    }

    fn tts_voices(&self) -> Vec<String> {
        vec!["mock-voice".to_string()]
    }

    fn name(&self) -> &str {
        "MockSpeechProvider"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "audio"))]
mod tests {
    use super::*;

    #[test]
    fn test_mock_stt_default() {
        let mock = MockSpeechProvider::new();
        let audio = vec![0u8; 32000]; // 1 second at 16kHz PCM16
        let result = mock.transcribe(&audio, AudioFormat::Pcm, None).unwrap();
        assert!(result.text.contains("Mock transcript"));
        assert!(result.duration_secs > 0.9);
        assert_eq!(result.language, Some("en".to_string()));
    }

    #[test]
    fn test_mock_stt_fixed_transcript() {
        let mock = MockSpeechProvider::with_transcript("Hello Luna");
        let audio = vec![0u8; 16000];
        let result = mock.transcribe(&audio, AudioFormat::Pcm, None).unwrap();
        assert_eq!(result.text, "Hello Luna");
    }

    #[test]
    fn test_mock_tts() {
        let mock = MockSpeechProvider::new();
        let opts = SynthesisOptions::default();
        let result = mock.synthesize("Hello world", &opts).unwrap();
        assert!(!result.audio.is_empty());
        assert_eq!(result.format, AudioFormat::Pcm);
        assert_eq!(result.sample_rate, 16000);
        // PCM16: 2 bytes per sample, 16000 samples/sec, 1 sec
        assert_eq!(result.audio.len(), 16000 * 2);
    }

    #[test]
    fn test_mock_provider_metadata() {
        let mock = MockSpeechProvider::new();
        assert!(mock.supports_stt());
        assert!(mock.supports_tts());
        assert_eq!(mock.name(), "MockSpeechProvider");
        assert_eq!(mock.tts_voices(), vec!["mock-voice"]);
    }
}
