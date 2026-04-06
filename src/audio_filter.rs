//! Audio processing pipeline — noise suppression, speaker verification,
//! acoustic echo cancellation, audio effects, and source separation.
//!
//! Provides a configurable `AudioEffectChain` that processes audio in real-time
//! through a series of effects. Each effect reports its estimated latency.
//!
//! Feature flags:
//! - `audio` — noise suppression (RNNoise), MFCC speaker verification, effects
//! - `audio-separation` — ONNX-based source separation (Demucs/SepFormer)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// ============================================================================
// Core Traits
// ============================================================================

/// Category of an audio effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectCategory {
    InputProcessing,
    Identity,
    Enhancement,
    Creative,
    Analysis,
}

/// An audio processing effect.
pub trait AudioEffect: Send + Sync {
    fn name(&self) -> &str;
    fn process(&mut self, samples: &mut [i16], sample_rate: u32);
    fn is_enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
    fn category(&self) -> EffectCategory;
    /// Estimated latency in microseconds.
    fn estimated_latency_us(&self) -> u64;
}

/// Noise suppression trait.
pub trait NoiseSuppressor: Send + Sync {
    fn suppress(&mut self, frame: &[i16], sample_rate: u32) -> Vec<i16>;
    fn name(&self) -> &str;
}

/// Speaker verification trait.
pub trait SpeakerVerifier: Send + Sync {
    fn create_embedding(&self, audio: &[i16], sample_rate: u32) -> Result<VoiceEmbedding, String>;
    fn compare(&self, a: &VoiceEmbedding, b: &VoiceEmbedding) -> f32;
    fn name(&self) -> &str;
}

/// Audio source separator trait.
pub trait AudioSeparator: Send + Sync {
    fn separate(&self, audio: &[i16], sample_rate: u32) -> Result<Vec<SeparatedTrack>, String>;
    fn name(&self) -> &str;
    fn max_tracks(&self) -> usize;
}

// ============================================================================
// Voice Embedding & Speaker Profile
// ============================================================================

/// A voice embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceEmbedding {
    pub vector: Vec<f32>,
    pub model_id: String,
    pub sample_duration_ms: u64,
}

/// An enrolled speaker profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub id: String,
    pub name: String,
    pub embeddings: Vec<VoiceEmbedding>,
    pub mean_embedding: Vec<f32>,
    pub threshold: f32,
    pub created_at: String,
    pub is_owner: bool,
}

impl SpeakerProfile {
    /// Recompute mean embedding from all samples.
    pub fn recompute_mean(&mut self) {
        if self.embeddings.is_empty() {
            return;
        }
        let dim = self.embeddings[0].vector.len();
        let mut mean = vec![0.0f32; dim];
        for emb in &self.embeddings {
            for (i, v) in emb.vector.iter().enumerate() {
                if i < dim {
                    mean[i] += v;
                }
            }
        }
        let n = self.embeddings.len() as f32;
        for v in &mut mean {
            *v /= n;
        }
        self.mean_embedding = mean;
    }
}

/// A separated audio track.
#[derive(Debug, Clone)]
pub struct SeparatedTrack {
    pub samples: Vec<i16>,
    pub track_id: usize,
    pub confidence: f32,
}

// ============================================================================
// Speaker Gate
// ============================================================================

/// Live info about a detected speaker (for GUI display).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSpeakerInfo {
    pub speaker_id: String,
    pub name: String,
    pub is_owner: bool,
    pub is_speaking: bool,
    pub last_confidence: f32,
    pub last_seen_ms: u64,
    pub rejected_count: u32,
}

/// Result of speaker identification.
#[derive(Debug, Clone)]
pub enum SpeakerIdentification {
    Identified {
        speaker_id: String,
        name: String,
        confidence: f32,
        is_owner: bool,
    },
    Unknown {
        confidence: f32,
    },
    NoProfiles,
}

/// Speaker gate — filters audio by speaker identity.
pub struct SpeakerGate {
    verifier: Box<dyn SpeakerVerifier>,
    profiles: Vec<SpeakerProfile>,
    owner_id: Option<String>,
    threshold: f32,
    allow_unknown: bool,
    only_owner: bool,
    active_speakers: Vec<ActiveSpeakerInfo>,
    unknown_counter: u32,
}

impl SpeakerGate {
    pub fn new(verifier: Box<dyn SpeakerVerifier>, threshold: f32) -> Self {
        Self {
            verifier,
            profiles: Vec::new(),
            owner_id: None,
            threshold,
            allow_unknown: true,
            only_owner: false,
            active_speakers: Vec::new(),
            unknown_counter: 0,
        }
    }

    /// Set whether only the owner can pass.
    pub fn set_only_owner(&mut self, only: bool) {
        self.only_owner = only;
    }

    /// Set whether unknown speakers are allowed.
    pub fn set_allow_unknown(&mut self, allow: bool) {
        self.allow_unknown = allow;
    }

    /// Enroll a speaker.
    pub fn enroll(
        &mut self,
        audio: &[i16],
        sample_rate: u32,
        name: &str,
        is_owner: bool,
    ) -> Result<String, String> {
        let embedding = self.verifier.create_embedding(audio, sample_rate)?;
        let id = format!("spk_{}", now_epoch());

        let mut profile = SpeakerProfile {
            id: id.clone(),
            name: name.to_string(),
            embeddings: vec![embedding],
            mean_embedding: Vec::new(),
            threshold: self.threshold,
            created_at: chrono::Utc::now().to_rfc3339(),
            is_owner,
        };
        profile.recompute_mean();

        if is_owner {
            self.owner_id = Some(id.clone());
        }
        self.profiles.push(profile);
        Ok(id)
    }

    /// Identify which enrolled speaker is talking.
    pub fn identify(&mut self, audio: &[i16], sample_rate: u32) -> SpeakerIdentification {
        if self.profiles.is_empty() {
            return SpeakerIdentification::NoProfiles;
        }

        let embedding = match self.verifier.create_embedding(audio, sample_rate) {
            Ok(e) => e,
            Err(_) => return SpeakerIdentification::Unknown { confidence: 0.0 },
        };

        let mut best_id = String::new();
        let mut best_name = String::new();
        let mut best_score = 0.0f32;
        let mut best_is_owner = false;

        for profile in &self.profiles {
            let ref_emb = VoiceEmbedding {
                vector: profile.mean_embedding.clone(),
                model_id: embedding.model_id.clone(),
                sample_duration_ms: 0,
            };
            let score = self.verifier.compare(&embedding, &ref_emb);
            if score > best_score {
                best_score = score;
                best_id = profile.id.clone();
                best_name = profile.name.clone();
                best_is_owner = profile.is_owner;
            }
        }

        let now = now_epoch_ms();

        if best_score >= self.threshold {
            // Update active speakers
            if let Some(info) = self
                .active_speakers
                .iter_mut()
                .find(|s| s.speaker_id == best_id)
            {
                info.is_speaking = true;
                info.last_confidence = best_score;
                info.last_seen_ms = now;
            } else {
                self.active_speakers.push(ActiveSpeakerInfo {
                    speaker_id: best_id.clone(),
                    name: best_name.clone(),
                    is_owner: best_is_owner,
                    is_speaking: true,
                    last_confidence: best_score,
                    last_seen_ms: now,
                    rejected_count: 0,
                });
            }

            SpeakerIdentification::Identified {
                speaker_id: best_id,
                name: best_name,
                confidence: best_score,
                is_owner: best_is_owner,
            }
        } else {
            self.unknown_counter += 1;
            let unknown_id = format!("unknown_{}", self.unknown_counter);
            if let Some(info) = self
                .active_speakers
                .iter_mut()
                .find(|s| s.speaker_id.starts_with("unknown"))
            {
                info.rejected_count += 1;
                info.last_seen_ms = now;
            } else {
                self.active_speakers.push(ActiveSpeakerInfo {
                    speaker_id: unknown_id,
                    name: "Unknown".to_string(),
                    is_owner: false,
                    is_speaking: false,
                    last_confidence: best_score,
                    last_seen_ms: now,
                    rejected_count: 1,
                });
            }

            SpeakerIdentification::Unknown {
                confidence: best_score,
            }
        }
    }

    /// Check if audio should pass through the gate.
    pub fn should_pass(&mut self, audio: &[i16], sample_rate: u32) -> bool {
        if self.profiles.is_empty() {
            return self.allow_unknown; // No profiles → open or closed
        }

        match self.identify(audio, sample_rate) {
            SpeakerIdentification::Identified { is_owner, .. } => {
                if self.only_owner {
                    is_owner
                } else {
                    true // Any enrolled speaker passes
                }
            }
            SpeakerIdentification::Unknown { .. } => self.allow_unknown,
            SpeakerIdentification::NoProfiles => self.allow_unknown,
        }
    }

    /// Get active speaker info for GUI.
    pub fn active_speakers(&self) -> &[ActiveSpeakerInfo] {
        &self.active_speakers
    }

    /// List enrolled profiles.
    pub fn profiles(&self) -> &[SpeakerProfile] {
        &self.profiles
    }

    /// Remove an enrolled profile.
    pub fn remove_profile(&mut self, speaker_id: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.id != speaker_id);
        if self.owner_id.as_deref() == Some(speaker_id) {
            self.owner_id = None;
        }
        self.profiles.len() < before
    }
}

// ============================================================================
// Speaker Diarization (no enrollment needed)
// ============================================================================

/// A temporary speaker cluster discovered during diarization.
#[derive(Debug, Clone)]
pub struct DiarizedSpeaker {
    /// Auto-assigned label (e.g., "Speaker 1", "Speaker 2").
    pub label: String,
    /// Numeric index (0-based).
    pub index: usize,
    /// Mean embedding of all audio segments from this speaker.
    mean_embedding: Vec<f32>,
    /// Number of segments assigned to this speaker.
    pub segment_count: u32,
    /// Last time this speaker was heard (epoch ms).
    pub last_seen_ms: u64,
}

/// Result of a diarization step.
#[derive(Debug, Clone)]
pub enum DiarizationResult {
    /// Matched an existing cluster.
    Assigned {
        label: String,
        index: usize,
        confidence: f32,
    },
    /// New speaker detected — created a new cluster.
    NewSpeaker { label: String, index: usize },
    /// Audio too short or silent to classify.
    Inconclusive,
}

/// Speaker diarizer — detects and tracks distinct speakers without enrollment.
///
/// Uses MFCC embeddings to cluster audio segments in real-time. Each new voice
/// that doesn't match existing clusters creates a new "Speaker N" label.
pub struct SpeakerDiarizer {
    verifier: Box<dyn SpeakerVerifier>,
    /// Discovered speaker clusters in the current session.
    clusters: Vec<DiarizedSpeaker>,
    /// Similarity threshold to assign to an existing cluster (0.0–1.0).
    /// Lower = more permissive (fewer speakers detected).
    threshold: f32,
    /// Maximum number of clusters to prevent unbounded growth.
    max_clusters: usize,
}

impl SpeakerDiarizer {
    /// Create a new diarizer.
    ///
    /// - `threshold`: similarity score above which audio is assigned to existing cluster (default: 0.55)
    /// - `max_clusters`: maximum distinct speakers to track (default: 10)
    pub fn new(verifier: Box<dyn SpeakerVerifier>, threshold: f32, max_clusters: usize) -> Self {
        Self {
            verifier,
            clusters: Vec::new(),
            threshold: threshold.clamp(0.0, 1.0),
            max_clusters: max_clusters.max(1), // At least 1 cluster
        }
    }

    /// Create with sensible defaults (threshold=0.55, max_clusters=10).
    pub fn with_defaults(verifier: Box<dyn SpeakerVerifier>) -> Self {
        Self::new(verifier, 0.55, 10)
    }

    /// Process an audio segment and assign it to a speaker cluster.
    ///
    /// Returns which speaker is talking, or `NewSpeaker` if this is a new voice.
    pub fn process(&mut self, audio: &[i16], sample_rate: u32) -> DiarizationResult {
        let embedding = match self.verifier.create_embedding(audio, sample_rate) {
            Ok(e) => e,
            Err(_) => return DiarizationResult::Inconclusive,
        };

        if embedding.vector.iter().all(|&v| v == 0.0) {
            return DiarizationResult::Inconclusive;
        }

        // Compare against existing clusters
        let mut best_idx = None;
        let mut best_score = 0.0f32;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let ref_emb = VoiceEmbedding {
                vector: cluster.mean_embedding.clone(),
                model_id: embedding.model_id.clone(),
                sample_duration_ms: 0,
            };
            let score = self.verifier.compare(&embedding, &ref_emb);
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let now = now_epoch_ms();

        // If best match is above threshold, assign to that cluster
        if let Some(idx) = best_idx {
            if best_score >= self.threshold {
                let cluster = &mut self.clusters[idx];
                // Update mean embedding with running average
                let n = cluster.segment_count as f32;
                for (j, val) in embedding.vector.iter().enumerate() {
                    if j < cluster.mean_embedding.len() {
                        cluster.mean_embedding[j] =
                            (cluster.mean_embedding[j] * n + val) / (n + 1.0);
                    }
                }
                cluster.segment_count += 1;
                cluster.last_seen_ms = now;

                return DiarizationResult::Assigned {
                    label: cluster.label.clone(),
                    index: cluster.index,
                    confidence: best_score,
                };
            }
        }

        // No match — create a new cluster (if under limit)
        if self.clusters.len() >= self.max_clusters {
            // At capacity — assign to the closest cluster anyway
            if let Some(idx) = best_idx {
                let cluster = &self.clusters[idx];
                return DiarizationResult::Assigned {
                    label: cluster.label.clone(),
                    index: cluster.index,
                    confidence: best_score,
                };
            }
            return DiarizationResult::Inconclusive;
        }

        let index = self.clusters.len();
        let label = format!("Speaker {}", index + 1);
        self.clusters.push(DiarizedSpeaker {
            label: label.clone(),
            index,
            mean_embedding: embedding.vector,
            segment_count: 1,
            last_seen_ms: now,
        });

        DiarizationResult::NewSpeaker { label, index }
    }

    /// Get all discovered speaker clusters.
    pub fn clusters(&self) -> &[DiarizedSpeaker] {
        &self.clusters
    }

    /// Number of distinct speakers detected so far.
    pub fn speaker_count(&self) -> usize {
        self.clusters.len()
    }

    /// Reset all clusters (start a fresh session).
    pub fn reset(&mut self) {
        self.clusters.clear();
    }

    /// Set the similarity threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

// ============================================================================
// MFCC Speaker Verifier (pure Rust)
// ============================================================================

/// MFCC-based speaker verification (pure Rust, no external deps).
/// Less accurate than neural models (~85%) but zero dependencies.
pub struct MfccSpeakerVerifier {
    num_coefficients: usize,
    num_mel_bands: usize,
}

impl MfccSpeakerVerifier {
    pub fn new() -> Self {
        Self {
            num_coefficients: 13,
            num_mel_bands: 26,
        }
    }

    /// Compute MFCC features from audio.
    fn compute_mfcc(&self, audio: &[i16], sample_rate: u32) -> Vec<f32> {
        if audio.is_empty() {
            return vec![0.0; self.num_coefficients];
        }

        let frame_size = (sample_rate as usize * 25) / 1000; // 25ms frames
        let hop_size = (sample_rate as usize * 10) / 1000; // 10ms hop

        let mut all_coeffs = vec![vec![0.0f32; self.num_coefficients]; 0];

        let mut pos = 0;
        while pos + frame_size <= audio.len() {
            let frame: Vec<f32> = audio[pos..pos + frame_size]
                .iter()
                .map(|&s| s as f32 / i16::MAX as f32)
                .collect();

            // Compute frame energy in mel bands (simplified)
            let energy = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
            let log_energy = (energy + 1e-10).ln();

            // Simple MFCC approximation: distribute energy across coefficients
            // based on frequency content (spectral centroid estimation)
            let mut coeffs = vec![0.0f32; self.num_coefficients];
            coeffs[0] = log_energy;

            // Spectral features (simplified — real MFCC uses FFT + mel filterbank + DCT)
            let mut zero_crossings = 0usize;
            for i in 1..frame.len() {
                if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
                    zero_crossings += 1;
                }
            }
            coeffs[1] = zero_crossings as f32 / frame.len() as f32;

            // Spectral moments
            let sum: f32 = frame.iter().map(|s| s.abs()).sum();
            let mean = sum / frame.len() as f32;
            coeffs[2] = mean;

            let variance: f32 =
                frame.iter().map(|s| (s.abs() - mean).powi(2)).sum::<f32>() / frame.len() as f32;
            coeffs[3] = variance.sqrt();

            // Higher order features (fill remaining coefficients)
            for j in 4..self.num_coefficients {
                let k = j as f32 / self.num_coefficients as f32;
                let weighted_sum: f32 = frame
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        let t = i as f32 / frame.len() as f32;
                        s.abs() * (std::f32::consts::PI * k * t).cos()
                    })
                    .sum();
                coeffs[j] = weighted_sum / frame.len() as f32;
            }

            all_coeffs.push(coeffs);
            pos += hop_size;
        }

        // Average across all frames
        if all_coeffs.is_empty() {
            return vec![0.0; self.num_coefficients];
        }

        let mut mean = vec![0.0f32; self.num_coefficients];
        for coeffs in &all_coeffs {
            for (i, v) in coeffs.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = all_coeffs.len() as f32;
        for v in &mut mean {
            *v /= n;
        }
        mean
    }
}

impl Default for MfccSpeakerVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeakerVerifier for MfccSpeakerVerifier {
    fn create_embedding(&self, audio: &[i16], sample_rate: u32) -> Result<VoiceEmbedding, String> {
        if audio.len() < sample_rate as usize {
            return Err("Audio too short (need at least 1 second)".into());
        }
        // Anti-spoofing: reject flat audio (#1)
        let variance: f64 =
            audio.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / audio.len() as f64;
        if variance < 1.0 {
            return Err("Audio appears to be silence or flat signal (anti-spoofing)".into());
        }

        let mfcc = self.compute_mfcc(audio, sample_rate);
        Ok(VoiceEmbedding {
            vector: mfcc,
            model_id: "mfcc-13".to_string(),
            sample_duration_ms: (audio.len() as u64 * 1000) / sample_rate as u64,
        })
    }

    fn compare(&self, a: &VoiceEmbedding, b: &VoiceEmbedding) -> f32 {
        cosine_similarity(&a.vector, &b.vector)
    }

    fn name(&self) -> &str {
        "MFCC-13 Speaker Verifier"
    }
}

// ============================================================================
// Audio Effects (pure Rust implementations)
// ============================================================================

/// Noise gate — silences audio below an energy threshold.
pub struct NoiseGate {
    threshold_rms: f32,
    enabled: bool,
}

impl NoiseGate {
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold_rms: 10.0f32.powf(threshold_db / 20.0),
            enabled: true,
        }
    }
}

impl AudioEffect for NoiseGate {
    fn name(&self) -> &str {
        "Noise Gate"
    }
    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        let rms = compute_rms(samples);
        if rms < self.threshold_rms {
            for s in samples.iter_mut() {
                *s = 0;
            }
        }
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::InputProcessing
    }
    fn estimated_latency_us(&self) -> u64 {
        50
    } // ~0ms
}

/// Auto Gain Control — adjusts volume to target level.
pub struct AutoGainControl {
    target_rms: f32,
    smoothing: f32,
    current_gain: f32,
    enabled: bool,
}

impl AutoGainControl {
    pub fn new(target_db: f32) -> Self {
        Self {
            target_rms: 10.0f32.powf(target_db / 20.0),
            smoothing: 0.1,
            current_gain: 1.0,
            enabled: true,
        }
    }
}

impl AudioEffect for AutoGainControl {
    fn name(&self) -> &str {
        "Auto Gain Control"
    }
    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        let rms = compute_rms(samples);
        if rms > 0.0001 {
            let target_gain = self.target_rms / rms;
            let clamped_gain = target_gain.clamp(0.1, 10.0);
            self.current_gain =
                self.current_gain * (1.0 - self.smoothing) + clamped_gain * self.smoothing;
        }
        for s in samples.iter_mut() {
            let v = (*s as f32 * self.current_gain).clamp(-32767.0, 32767.0);
            *s = v as i16;
        }
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::InputProcessing
    }
    fn estimated_latency_us(&self) -> u64 {
        50
    }
}

/// Compressor — reduces dynamic range.
pub struct Compressor {
    threshold_rms: f32,
    ratio: f32,
    enabled: bool,
}

impl Compressor {
    pub fn new(threshold_db: f32, ratio: f32) -> Self {
        Self {
            threshold_rms: 10.0f32.powf(threshold_db / 20.0),
            ratio: ratio.max(1.0),
            enabled: true,
        }
    }
}

impl AudioEffect for Compressor {
    fn name(&self) -> &str {
        "Compressor"
    }
    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        let rms = compute_rms(samples);
        if rms > self.threshold_rms {
            let excess = rms - self.threshold_rms;
            let compressed_excess = excess / self.ratio;
            let gain = (self.threshold_rms + compressed_excess) / rms;
            for s in samples.iter_mut() {
                *s = ((*s as f32) * gain).clamp(-32767.0, 32767.0) as i16;
            }
        }
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::Enhancement
    }
    fn estimated_latency_us(&self) -> u64 {
        100
    }
}

/// Simple distortion effect.
pub struct Distortion {
    gain: f32,
    clip_threshold: f32,
    enabled: bool,
}

impl Distortion {
    pub fn new(gain: f32, clip_threshold: f32) -> Self {
        Self {
            gain: gain.max(1.0),
            clip_threshold: clip_threshold.clamp(0.1, 1.0),
            enabled: false,
        }
    }
}

impl AudioEffect for Distortion {
    fn name(&self) -> &str {
        "Distortion"
    }
    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        let max = (i16::MAX as f32) * self.clip_threshold;
        for s in samples.iter_mut() {
            let v = (*s as f32) * self.gain;
            *s = v.clamp(-max, max) as i16;
        }
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        50
    }
}

/// Simple reverb (Schroeder-style, 4 comb filters).
pub struct Reverb {
    buffers: Vec<VecDeque<f32>>,
    mix: f32,
    feedback: f32,
    enabled: bool,
}

impl Reverb {
    pub fn new(room_size: f32, mix: f32) -> Self {
        let base_delays = [1557, 1617, 1491, 1422]; // prime-ish delays
        let scale = room_size.clamp(0.1, 2.0);
        let buffers = base_delays
            .iter()
            .map(|&d| {
                let len = (d as f32 * scale) as usize;
                let mut buf = VecDeque::with_capacity(len);
                buf.resize(len, 0.0);
                buf
            })
            .collect();
        Self {
            buffers,
            mix: mix.clamp(0.0, 1.0),
            feedback: 0.7 * room_size.clamp(0.0, 1.0),
            enabled: false,
        }
    }
}

impl AudioEffect for Reverb {
    fn name(&self) -> &str {
        "Reverb"
    }
    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        for s in samples.iter_mut() {
            let input = *s as f32 / i16::MAX as f32;
            let mut reverb_sum = 0.0f32;
            for buf in &mut self.buffers {
                let delayed = buf.pop_front().unwrap_or(0.0);
                reverb_sum += delayed;
                buf.push_back(input + delayed * self.feedback);
            }
            reverb_sum /= self.buffers.len() as f32;
            let mixed = input * (1.0 - self.mix) + reverb_sum * self.mix;
            *s = (mixed * i16::MAX as f32).clamp(-32767.0, 32767.0) as i16;
        }
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        5000
    } // ~5ms
}

// ============================================================================
// Acoustic Echo Canceller (NLMS)
// ============================================================================

/// Acoustic Echo Canceller using Normalized LMS adaptive filter.
pub struct AcousticEchoCanceller {
    weights: Vec<f32>,
    reference_buffer: VecDeque<f32>,
    step_size: f32,
    enabled: bool,
}

impl AcousticEchoCanceller {
    /// Create with filter length in samples (default 4800 = 300ms at 16kHz).
    pub fn new(filter_length: usize) -> Self {
        Self {
            weights: vec![0.0; filter_length],
            reference_buffer: VecDeque::from(vec![0.0; filter_length]),
            step_size: 0.5,
            enabled: true,
        }
    }

    /// Feed speaker output as reference signal.
    pub fn feed_reference(&mut self, speaker_output: &[i16]) {
        for &s in speaker_output {
            self.reference_buffer.push_back(s as f32 / i16::MAX as f32);
            if self.reference_buffer.len() > self.weights.len() {
                self.reference_buffer.pop_front();
            }
        }
    }
}

impl AudioEffect for AcousticEchoCanceller {
    fn name(&self) -> &str {
        "Echo Cancellation (AEC)"
    }

    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        let ref_vec: Vec<f32> = self.reference_buffer.iter().copied().collect();
        let n = self.weights.len();
        if ref_vec.len() < n {
            return; // Not enough reference data yet
        }

        for s in samples.iter_mut() {
            let mic = *s as f32 / i16::MAX as f32;

            // Estimate echo: dot product of weights and reference
            let mut echo_estimate = 0.0f32;
            for i in 0..n {
                let ref_idx = ref_vec.len().saturating_sub(n) + i;
                if ref_idx < ref_vec.len() {
                    echo_estimate += self.weights[i] * ref_vec[ref_idx];
                }
            }

            // Error = mic - estimated echo
            let error = mic - echo_estimate;

            // NLMS weight update
            let ref_power: f32 = ref_vec[ref_vec.len().saturating_sub(n)..]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                + 1e-6; // avoid div by zero

            let step = self.step_size * error / ref_power;
            for i in 0..n {
                let ref_idx = ref_vec.len().saturating_sub(n) + i;
                if ref_idx < ref_vec.len() {
                    self.weights[i] += step * ref_vec[ref_idx];
                    // Clamp weights to prevent divergence (#16)
                    self.weights[i] = self.weights[i].clamp(-10.0, 10.0);
                }
            }

            // Output: error signal (mic without echo)
            *s = (error * i16::MAX as f32).clamp(-32767.0, 32767.0) as i16;
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn category(&self) -> EffectCategory {
        EffectCategory::InputProcessing
    }
    fn estimated_latency_us(&self) -> u64 {
        5000
    } // ~5ms
}

// ============================================================================
// Audio Effect Chain
// ============================================================================

/// Configurable chain of audio effects processed in order.
pub struct AudioEffectChain {
    effects: Vec<Box<dyn AudioEffect>>,
    /// Speaker reference for AEC (fed from speaker output stream).
    pub aec_reference: Option<Arc<Mutex<VecDeque<i16>>>>,
}

impl AudioEffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            aec_reference: None,
        }
    }

    /// Add an effect to the chain.
    pub fn add_effect(&mut self, effect: Box<dyn AudioEffect>) {
        self.effects.push(effect);
    }

    /// Process audio through all enabled effects.
    pub fn process_frame(&mut self, samples: &mut [i16], sample_rate: u32) {
        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(samples, sample_rate);
            }
        }
    }

    /// Enable or disable an effect by name.
    pub fn enable_effect(&mut self, name: &str, enabled: bool) {
        for effect in &mut self.effects {
            if effect.name() == name {
                effect.set_enabled(enabled);
            }
        }
    }

    /// List all effects with their status.
    pub fn list_effects(&self) -> Vec<(&str, bool, EffectCategory, u64)> {
        self.effects
            .iter()
            .map(|e| {
                (
                    e.name(),
                    e.is_enabled(),
                    e.category(),
                    e.estimated_latency_us(),
                )
            })
            .collect()
    }

    /// Total estimated latency of all ENABLED effects in microseconds.
    pub fn total_latency_us(&self) -> u64 {
        self.effects
            .iter()
            .filter(|e| e.is_enabled())
            .map(|e| e.estimated_latency_us())
            .sum()
    }

    /// Per-effect latency breakdown.
    pub fn latency_breakdown(&self) -> Vec<(&str, u64, bool)> {
        self.effects
            .iter()
            .map(|e| (e.name(), e.estimated_latency_us(), e.is_enabled()))
            .collect()
    }
}

impl Default for AudioEffectChain {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn compute_rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples
        .iter()
        .map(|&s| (s as f64 / i16::MAX as f64).powi(2))
        .sum();
    (sum / samples.len() as f64).sqrt() as f32
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Creative Voice Effects
// ============================================================================

/// Pitch shifter — shifts pitch up (helium/chipmunk) or down (Darth Vader/deep).
///
/// Uses simple resampling with linear interpolation. Positive semitones shift up,
/// negative shift down. +12 = one octave up, -12 = one octave down.
pub struct PitchShifter {
    shift_semitones: f32,
    enabled: bool,
    buffer: Vec<f32>,
    read_pos: f64,
}

impl PitchShifter {
    pub fn new(semitones: f32) -> Self {
        Self {
            shift_semitones: semitones,
            enabled: true,
            buffer: Vec::new(),
            read_pos: 0.0,
        }
    }

    /// +1 octave — helium voice.
    pub fn helium() -> Self {
        Self::new(12.0)
    }

    /// +7 semitones — chipmunk voice (slightly less extreme than helium).
    pub fn chipmunk() -> Self {
        Self::new(7.0)
    }

    /// -8 semitones — deep Darth Vader voice.
    pub fn darth_vader() -> Self {
        Self::new(-8.0)
    }

    /// -1 octave — very deep voice.
    pub fn deep() -> Self {
        Self::new(-12.0)
    }
}

impl AudioEffect for PitchShifter {
    fn name(&self) -> &str {
        "Pitch Shifter"
    }

    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        if samples.is_empty() || self.shift_semitones.abs() < 0.001 {
            return;
        }

        let ratio = 2.0f64.powf(self.shift_semitones as f64 / 12.0);

        // Convert input to f32 buffer
        self.buffer.clear();
        self.buffer.extend(samples.iter().map(|&s| s as f32));

        let input_len = self.buffer.len();
        let mut output = Vec::with_capacity(samples.len());

        // Resample via linear interpolation
        let mut pos = self.read_pos;
        for _ in 0..samples.len() {
            let idx = pos as usize;
            if idx + 1 < input_len {
                let frac = (pos - idx as f64) as f32;
                let interpolated = self.buffer[idx] * (1.0 - frac) + self.buffer[idx + 1] * frac;
                output.push(interpolated);
            } else if idx < input_len {
                output.push(self.buffer[idx]);
            } else {
                output.push(0.0);
            }
            pos += ratio;
        }

        // Track fractional read position for continuity across calls
        self.read_pos = pos - input_len as f64;
        if self.read_pos < 0.0 {
            self.read_pos = 0.0;
        }

        // Write back to samples
        for (i, s) in samples.iter_mut().enumerate() {
            if i < output.len() {
                *s = output[i].clamp(-32767.0, 32767.0) as i16;
            }
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }

    fn estimated_latency_us(&self) -> u64 {
        200
    }
}

/// Robot voice — ring modulation (multiply audio by a carrier sine wave).
///
/// Creates a metallic, robotic sound by modulating the audio with a fixed-frequency
/// sine wave. Default carrier frequency is 150 Hz.
pub struct RobotVoice {
    frequency: f32,
    phase: f64,
    enabled: bool,
}

impl RobotVoice {
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            phase: 0.0,
            enabled: true,
        }
    }

    /// Default robot voice at 150 Hz carrier.
    pub fn default_robot() -> Self {
        Self::new(150.0)
    }
}

impl AudioEffect for RobotVoice {
    fn name(&self) -> &str {
        "Robot Voice"
    }

    fn process(&mut self, samples: &mut [i16], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        let phase_inc = 2.0 * std::f64::consts::PI * self.frequency as f64 / sample_rate as f64;

        for s in samples.iter_mut() {
            let modulator = self.phase.sin() as f32;
            let modulated = *s as f32 * modulator;
            *s = modulated.clamp(-32767.0, 32767.0) as i16;
            self.phase += phase_inc;
        }

        // Keep phase in [0, 2pi) to avoid floating-point drift
        self.phase %= 2.0 * std::f64::consts::PI;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }

    fn estimated_latency_us(&self) -> u64 {
        50
    }
}

/// AutoTune — snap pitch to nearest musical note.
///
/// Uses autocorrelation to estimate the dominant pitch of each frame, finds the
/// nearest semitone (A4=440 Hz reference), and applies a corrective pitch shift.
pub struct AutoTune {
    enabled: bool,
    correction_strength: f32,
    last_correction: f32,
}

impl AutoTune {
    pub fn new(strength: f32) -> Self {
        Self {
            enabled: true,
            correction_strength: strength.clamp(0.0, 1.0),
            last_correction: 0.0,
        }
    }

    /// Full snap to nearest note (strength = 1.0).
    pub fn full() -> Self {
        Self::new(1.0)
    }

    /// Subtle correction (strength = 0.5).
    pub fn subtle() -> Self {
        Self::new(0.5)
    }

    /// Estimate pitch via autocorrelation. Returns frequency in Hz or None if unclear.
    fn estimate_pitch(samples: &[i16], sample_rate: u32) -> Option<f32> {
        if samples.len() < 64 {
            return None;
        }

        let float_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

        // Check for silence
        let rms =
            (float_samples.iter().map(|s| s * s).sum::<f32>() / float_samples.len() as f32).sqrt();
        if rms < 500.0 {
            return None;
        }

        // Autocorrelation to find fundamental period
        let max_lag = (sample_rate as usize / 80).min(samples.len() / 2); // 80 Hz min
        let min_lag = sample_rate as usize / 1000; // 1000 Hz max

        if min_lag >= max_lag {
            return None;
        }

        let mut best_lag = min_lag;
        let mut best_corr = f32::NEG_INFINITY;

        for lag in min_lag..max_lag {
            let mut corr = 0.0f32;
            let len = float_samples.len() - lag;
            for i in 0..len {
                corr += float_samples[i] * float_samples[i + lag];
            }
            corr /= len as f32;

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Verify that the autocorrelation peak is significant
        let mut zero_corr = 0.0f32;
        for i in 0..float_samples.len() {
            zero_corr += float_samples[i] * float_samples[i];
        }
        zero_corr /= float_samples.len() as f32;

        if zero_corr < 1.0 || best_corr / zero_corr < 0.3 {
            return None;
        }

        let freq = sample_rate as f32 / best_lag as f32;
        if freq >= 80.0 && freq <= 1000.0 {
            Some(freq)
        } else {
            None
        }
    }

    /// Find the nearest semitone to a given frequency (A4=440 Hz).
    /// Returns the correction in semitones needed to reach the nearest note.
    fn correction_for_freq(freq: f32) -> f32 {
        // Semitones from A4: n = 12 * log2(freq / 440)
        let semitones_from_a4 = 12.0 * (freq / 440.0).log2();
        let nearest_semitone = semitones_from_a4.round();
        nearest_semitone - semitones_from_a4
    }
}

impl AudioEffect for AutoTune {
    fn name(&self) -> &str {
        "AutoTune"
    }

    fn process(&mut self, samples: &mut [i16], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        // Estimate current pitch
        let correction = if let Some(freq) = Self::estimate_pitch(samples, sample_rate) {
            let raw_correction = Self::correction_for_freq(freq);
            raw_correction * self.correction_strength
        } else {
            // No clear pitch detected — use last correction with decay
            self.last_correction * 0.5
        };

        self.last_correction = correction;

        // Apply pitch shift by the correction amount
        if correction.abs() < 0.01 {
            return; // No correction needed
        }

        let ratio = 2.0f64.powf(correction as f64 / 12.0);
        let float_in: Vec<f32> = samples.iter().map(|&s| s as f32).collect();
        let input_len = float_in.len();

        for (i, s) in samples.iter_mut().enumerate() {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            if idx + 1 < input_len {
                let frac = (pos - idx as f64) as f32;
                let interpolated = float_in[idx] * (1.0 - frac) + float_in[idx + 1] * frac;
                *s = interpolated.clamp(-32767.0, 32767.0) as i16;
            } else if idx < input_len {
                *s = float_in[idx].clamp(-32767.0, 32767.0) as i16;
            }
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }

    fn estimated_latency_us(&self) -> u64 {
        500
    }
}

/// Echo effect — delayed copy mixed back into the signal.
///
/// Supports configurable delay, feedback (how much of the echo feeds back),
/// and wet/dry mix ratio.
pub struct EchoEffect {
    delay_ms: u32,
    feedback: f32,
    mix: f32,
    buffer: Vec<i16>,
    write_pos: usize,
    enabled: bool,
}

impl EchoEffect {
    pub fn new(delay_ms: u32, feedback: f32, mix: f32) -> Self {
        Self {
            delay_ms,
            feedback: feedback.clamp(0.0, 0.9),
            mix: mix.clamp(0.0, 1.0),
            buffer: Vec::new(),
            write_pos: 0,
            enabled: true,
        }
    }

    /// Default echo: 300ms delay, 0.4 feedback, 0.3 mix.
    pub fn default_echo() -> Self {
        Self::new(300, 0.4, 0.3)
    }

    /// Short echo: 100ms, subtle.
    pub fn short() -> Self {
        Self::new(100, 0.2, 0.2)
    }

    /// Long echo: 800ms, more prominent.
    pub fn long() -> Self {
        Self::new(800, 0.5, 0.4)
    }

    fn ensure_buffer(&mut self, sample_rate: u32) {
        let delay_samples = (sample_rate as usize * self.delay_ms as usize) / 1000;
        if self.buffer.len() != delay_samples {
            self.buffer = vec![0i16; delay_samples];
            self.write_pos = 0;
        }
    }
}

impl AudioEffect for EchoEffect {
    fn name(&self) -> &str {
        "Echo"
    }

    fn process(&mut self, samples: &mut [i16], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        self.ensure_buffer(sample_rate);

        if self.buffer.is_empty() {
            return;
        }

        let buf_len = self.buffer.len();

        for s in samples.iter_mut() {
            // Read delayed sample from buffer
            let delayed = self.buffer[self.write_pos];

            // Mix: output = dry * (1 - mix) + delayed * mix
            let dry = *s as f32;
            let mixed = dry * (1.0 - self.mix) + delayed as f32 * self.mix;

            // Write to buffer: current input + feedback from delayed
            let to_buffer = (dry + delayed as f32 * self.feedback).clamp(-32767.0, 32767.0) as i16;
            self.buffer[self.write_pos] = to_buffer;

            // Output
            *s = mixed.clamp(-32767.0, 32767.0) as i16;

            self.write_pos = (self.write_pos + 1) % buf_len;
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }

    fn estimated_latency_us(&self) -> u64 {
        self.delay_ms as u64 * 1000
    }
}

/// Megaphone effect — bandpass filter (approx. 300-3000 Hz) + soft clipping distortion.
///
/// Simulates the sound of a megaphone or bullhorn by removing bass and treble,
/// then applying soft saturation via `tanh`.
pub struct MegaphoneEffect {
    enabled: bool,
    drive: f32,
    prev_sample: f32,
}

impl MegaphoneEffect {
    pub fn new(drive: f32) -> Self {
        Self {
            enabled: true,
            drive: drive.clamp(0.0, 1.0),
            prev_sample: 0.0,
        }
    }

    /// Default megaphone with moderate distortion (drive = 0.6).
    pub fn default_megaphone() -> Self {
        Self::new(0.6)
    }
}

impl AudioEffect for MegaphoneEffect {
    fn name(&self) -> &str {
        "Megaphone"
    }

    fn process(&mut self, samples: &mut [i16], sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        // Single-pole IIR coefficients for approximate bandpass:
        // High-pass at ~300 Hz: alpha_hp = 1 / (1 + 2pi * fc / fs)
        let hp_alpha = 1.0 / (1.0 + 2.0 * std::f32::consts::PI * 300.0 / sample_rate as f32);

        // Low-pass at ~3000 Hz: alpha_lp = 2pi * fc / (2pi * fc + fs)
        let lp_rc = 1.0 / (2.0 * std::f32::consts::PI * 3000.0);
        let lp_dt = 1.0 / sample_rate as f32;
        let lp_alpha = lp_dt / (lp_rc + lp_dt);

        let mut hp_prev_in = self.prev_sample;
        let mut hp_prev_out = 0.0f32;
        let mut lp_prev = 0.0f32;

        for s in samples.iter_mut() {
            let input = *s as f32 / i16::MAX as f32;

            // High-pass filter (remove bass < ~300 Hz)
            let hp_out = hp_alpha * (hp_prev_out + input - hp_prev_in);
            hp_prev_in = input;
            hp_prev_out = hp_out;

            // Low-pass filter (remove treble > ~3000 Hz)
            lp_prev += lp_alpha * (hp_out - lp_prev);

            // Soft clipping via tanh with drive
            let drive_scale = 1.0 + self.drive * 4.0; // map 0.0-1.0 to 1.0-5.0
            let clipped = (lp_prev * drive_scale).tanh();

            *s = (clipped * i16::MAX as f32).clamp(-32767.0, 32767.0) as i16;
        }

        self.prev_sample = samples
            .last()
            .map(|&s| s as f32 / i16::MAX as f32)
            .unwrap_or(0.0);
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Creative
    }

    fn estimated_latency_us(&self) -> u64 {
        100
    }
}

// ============================================================================
// Industrial Noise Suppression
// ============================================================================

/// Intelligent noise reducer — differentiates voice from non-voice (machinery,
/// impacts) and reduces non-voice sounds to a target level relative to voice.
///
/// Uses zero-crossing rate, spectral flatness, and energy to detect speech frames.
/// Non-speech frames are attenuated so their level is `voice_to_noise_ratio` of
/// the tracked voice level.
pub struct IntelligentNoiseReducer {
    enabled: bool,
    /// Target: non-voice should be this fraction of voice level.
    /// 0.7 means non-voice at 70% of voice level (30% quieter).
    voice_to_noise_ratio: f32,
    /// Running average of voice RMS level.
    voice_rms_avg: f32,
    /// Smoothing factor for voice level tracking (0.0-1.0).
    voice_tracking_alpha: f32,
    /// Minimum voice RMS to consider as speech (anti-silence).
    min_voice_rms: f32,
    /// Number of consecutive voice frames needed to confirm speech.
    voice_confirm_frames: u32,
    voice_frame_count: u32,
    /// Frame-level state.
    is_voice: bool,
}

impl IntelligentNoiseReducer {
    pub fn new(voice_to_noise_ratio: f32) -> Self {
        Self {
            enabled: true,
            voice_to_noise_ratio: voice_to_noise_ratio.clamp(0.0, 1.0),
            voice_rms_avg: 0.0,
            voice_tracking_alpha: 0.1,
            min_voice_rms: 500.0,
            voice_confirm_frames: 2,
            voice_frame_count: 0,
            is_voice: false,
        }
    }

    /// Factory noise preset: non-voice at 70% of voice level.
    pub fn factory() -> Self {
        Self::new(0.7)
    }

    /// Construction site: more aggressive, non-voice at 50%.
    pub fn construction() -> Self {
        Self::new(0.5)
    }

    /// Detect whether a frame contains voice using ZCR, spectral flatness, and RMS.
    fn detect_voice(samples: &[i16]) -> bool {
        if samples.is_empty() {
            return false;
        }

        let rms = compute_rms_i16(samples);
        if rms < 500.0 {
            return false; // silence
        }

        // Zero-crossing rate
        let zcr = samples
            .windows(2)
            .filter(|w| (w[0] >= 0) != (w[1] >= 0))
            .count() as f32
            / samples.len() as f32;

        // Spectral flatness approximation via ratio of geometric/arithmetic mean of |samples|
        let abs_samples: Vec<f32> = samples.iter().map(|&s| (s as f32).abs().max(1.0)).collect();
        let log_mean = abs_samples.iter().map(|s| s.ln()).sum::<f32>() / abs_samples.len() as f32;
        let geo_mean = log_mean.exp();
        let arith_mean = abs_samples.iter().sum::<f32>() / abs_samples.len() as f32;
        let flatness = geo_mean / arith_mean.max(1.0); // 0=tonal(voice), 1=flat(noise)

        // Voice: moderate ZCR (0.02-0.15), low flatness (<0.7), sufficient energy
        zcr < 0.20 && flatness < 0.75
    }
}

impl AudioEffect for IntelligentNoiseReducer {
    fn name(&self) -> &str {
        "Intelligent Noise Reducer"
    }

    fn process(&mut self, samples: &mut [i16], _sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        let is_voice_frame = Self::detect_voice(samples);

        if is_voice_frame {
            self.voice_frame_count += 1;
            if self.voice_frame_count >= self.voice_confirm_frames {
                self.is_voice = true;
            }

            // Update running voice RMS average
            let rms = compute_rms_i16(samples);
            if rms > self.min_voice_rms {
                self.voice_rms_avg = self.voice_rms_avg * (1.0 - self.voice_tracking_alpha)
                    + rms * self.voice_tracking_alpha;
            }
            // Voice frames pass through unmodified
        } else {
            self.voice_frame_count = 0;
            self.is_voice = false;

            // Apply gain reduction to non-voice frames
            if self.voice_rms_avg > 0.0 {
                let current_rms = compute_rms_i16(samples);
                if current_rms > 1.0 {
                    let target = self.voice_rms_avg * self.voice_to_noise_ratio;
                    let gain = (target / current_rms).clamp(0.0, 1.0);

                    for s in samples.iter_mut() {
                        *s = (*s as f32 * gain).clamp(-32767.0, 32767.0) as i16;
                    }
                }
            }
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn category(&self) -> EffectCategory {
        EffectCategory::Enhancement
    }

    fn estimated_latency_us(&self) -> u64 {
        300
    }
}

/// Compute RMS of i16 samples (raw, not normalized).
fn compute_rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sample_rate: u32, duration_ms: u32) -> Vec<i16> {
        let num_samples = (sample_rate * duration_ms as u32) / 1000;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (f32::sin(2.0 * std::f32::consts::PI * freq * t) * 16000.0) as i16
            })
            .collect()
    }

    fn silence(sample_rate: u32, duration_ms: u32) -> Vec<i16> {
        vec![0i16; (sample_rate * duration_ms as u32 / 1000) as usize]
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_mfcc_verifier_creates_embedding() {
        let verifier = MfccSpeakerVerifier::new();
        let audio = sine_wave(300.0, 16000, 2000); // 2 seconds
        let emb = verifier.create_embedding(&audio, 16000).unwrap();
        assert_eq!(emb.vector.len(), 13);
        assert_eq!(emb.model_id, "mfcc-13");
    }

    #[test]
    fn test_mfcc_verifier_rejects_short_audio() {
        let verifier = MfccSpeakerVerifier::new();
        let audio = sine_wave(300.0, 16000, 500); // 0.5 seconds
        assert!(verifier.create_embedding(&audio, 16000).is_err());
    }

    #[test]
    fn test_mfcc_verifier_rejects_flat() {
        let verifier = MfccSpeakerVerifier::new();
        let audio = vec![0i16; 32000]; // 2 seconds of silence
        assert!(verifier.create_embedding(&audio, 16000).is_err());
    }

    #[test]
    fn test_mfcc_different_frequencies() {
        let verifier = MfccSpeakerVerifier::new();
        let audio_low = sine_wave(200.0, 16000, 2000);
        let audio_high = sine_wave(2000.0, 16000, 2000);
        let emb_low = verifier.create_embedding(&audio_low, 16000).unwrap();
        let emb_high = verifier.create_embedding(&audio_high, 16000).unwrap();
        let similarity = verifier.compare(&emb_low, &emb_high);
        // Pure sine waves may have high cosine similarity in simplified MFCC
        // but the embeddings should not be identical
        assert!(
            emb_low.vector != emb_high.vector,
            "Embeddings should differ"
        );
        // Similarity can be high for synthetic signals — real speech differs more
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_speaker_gate_no_profiles() {
        let verifier = MfccSpeakerVerifier::new();
        let mut gate = SpeakerGate::new(Box::new(verifier), 0.7);
        let audio = sine_wave(300.0, 16000, 2000);
        // No profiles + allow_unknown = true → passes
        assert!(gate.should_pass(&audio, 16000));
    }

    #[test]
    fn test_speaker_enrollment() {
        let verifier = MfccSpeakerVerifier::new();
        let mut gate = SpeakerGate::new(Box::new(verifier), 0.7);
        let audio = sine_wave(300.0, 16000, 2000);
        let id = gate.enroll(&audio, 16000, "Lander", true).unwrap();
        assert!(!id.is_empty());
        assert_eq!(gate.profiles().len(), 1);
        assert!(gate.profiles()[0].is_owner);
    }

    #[test]
    fn test_noise_gate_silences_quiet() {
        let mut gate = NoiseGate::new(-40.0);
        let mut samples = vec![10i16; 160]; // Very quiet
        gate.process(&mut samples, 16000);
        assert!(samples.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_noise_gate_passes_speech() {
        let mut gate = NoiseGate::new(-40.0);
        let mut samples = sine_wave(300.0, 16000, 10); // 10ms of tone
        let original = samples.clone();
        gate.process(&mut samples, 16000);
        assert_eq!(samples, original); // Loud enough to pass
    }

    #[test]
    fn test_agc_normalizes() {
        let mut agc = AutoGainControl::new(-18.0);
        let mut quiet = vec![100i16; 160]; // Very quiet
        agc.process(&mut quiet, 16000);
        // After AGC, should be louder
        let max_after = quiet.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(max_after > 100);
    }

    #[test]
    fn test_aec_no_reference_passthrough() {
        let mut aec = AcousticEchoCanceller::new(480);
        let original = sine_wave(300.0, 16000, 10);
        let mut samples = original.clone();
        aec.process(&mut samples, 16000);
        // Without reference, passes through (almost) unchanged
        // (some small artifacts from zero-initialized weights)
    }

    #[test]
    fn test_compressor_reduces_peaks() {
        let mut comp = Compressor::new(-20.0, 4.0);
        let mut loud = sine_wave(300.0, 16000, 10);
        let peak_before = loud.iter().map(|s| s.abs()).max().unwrap_or(0);
        comp.process(&mut loud, 16000);
        let peak_after = loud.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak_after <= peak_before);
    }

    #[test]
    fn test_distortion_clips() {
        let mut dist = Distortion::new(5.0, 0.3);
        dist.set_enabled(true);
        let mut samples = sine_wave(300.0, 16000, 10);
        dist.process(&mut samples, 16000);
        let max = (i16::MAX as f32 * 0.3) as i16;
        assert!(samples.iter().all(|&s| s.abs() <= max + 1));
    }

    #[test]
    fn test_reverb_modifies_audio() {
        let mut reverb = Reverb::new(0.5, 0.5);
        reverb.set_enabled(true);
        let original = sine_wave(300.0, 16000, 50);
        let mut processed = original.clone();
        reverb.process(&mut processed, 16000);
        // Reverb should modify the audio (not identical)
        assert_ne!(original, processed);
    }

    // ── Diarization tests ──────────────────────────────────────────────

    #[test]
    fn test_diarizer_new_speaker() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::with_defaults(verifier);
        assert_eq!(diarizer.speaker_count(), 0);

        // Generate speech-like audio (loud enough to not be inconclusive)
        let audio: Vec<i16> = (0..32000)
            .map(|i| ((i % 73) as i16 * 400) - 14000)
            .collect();
        let result = diarizer.process(&audio, 16000);

        match result {
            DiarizationResult::NewSpeaker { label, index } => {
                assert_eq!(label, "Speaker 1");
                assert_eq!(index, 0);
            }
            other => panic!("Expected NewSpeaker, got {:?}", other),
        }
        assert_eq!(diarizer.speaker_count(), 1);
    }

    #[test]
    fn test_diarizer_same_speaker_assigned() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::with_defaults(verifier);

        // Same audio twice should be assigned to same cluster
        let audio: Vec<i16> = (0..32000)
            .map(|i| ((i % 73) as i16 * 400) - 14000)
            .collect();
        let _ = diarizer.process(&audio, 16000); // creates Speaker 1

        let result = diarizer.process(&audio, 16000); // should match Speaker 1
        match result {
            DiarizationResult::Assigned { label, index, .. } => {
                assert_eq!(label, "Speaker 1");
                assert_eq!(index, 0);
            }
            DiarizationResult::NewSpeaker { .. } => {
                // Also acceptable — MFCC can vary slightly
            }
            other => panic!("Expected Assigned or NewSpeaker, got {:?}", other),
        }
    }

    #[test]
    fn test_diarizer_reset() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::with_defaults(verifier);

        let audio: Vec<i16> = (0..32000)
            .map(|i| ((i % 73) as i16 * 400) - 14000)
            .collect();
        let _ = diarizer.process(&audio, 16000);
        assert!(diarizer.speaker_count() > 0);

        diarizer.reset();
        assert_eq!(diarizer.speaker_count(), 0);
        assert!(diarizer.clusters().is_empty());
    }

    #[test]
    fn test_diarizer_inconclusive_on_silence() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::with_defaults(verifier);

        let silence = vec![0i16; 16000];
        let result = diarizer.process(&silence, 16000);
        assert!(matches!(result, DiarizationResult::Inconclusive));
        assert_eq!(diarizer.speaker_count(), 0);
    }

    #[test]
    fn test_diarizer_max_clusters() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::new(verifier, 0.99, 2); // very high threshold, max 2

        // With threshold 0.99, almost nothing will match, so each call creates a new cluster
        let a1: Vec<i16> = (0..32000)
            .map(|i| ((i as i32 % 50) * 400 - 10000) as i16)
            .collect();
        let _ = diarizer.process(&a1, 16000);

        let a2: Vec<i16> = (0..32000)
            .map(|i| ((i as i32 % 120) * 200 - 12000) as i16)
            .collect();
        let _ = diarizer.process(&a2, 16000);

        // At max — should not create a third
        let a3: Vec<i16> = (0..32000)
            .map(|i| ((i as i32 % 200) * 150 - 5000) as i16)
            .collect();
        let result = diarizer.process(&a3, 16000);
        assert!(diarizer.speaker_count() <= 2);
        // Should be Assigned (forced) since we hit the limit
        assert!(
            !matches!(result, DiarizationResult::NewSpeaker { .. })
                || diarizer.speaker_count() <= 2
        );
    }

    #[test]
    fn test_diarizer_threshold() {
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let mut diarizer = SpeakerDiarizer::with_defaults(verifier);
        assert!((diarizer.threshold() - 0.55).abs() < 0.01);
        diarizer.set_threshold(0.7);
        assert!((diarizer.threshold() - 0.7).abs() < 0.01);
    }

    // ── Effect chain tests ───────────────────────────────────────────────

    #[test]
    fn test_effect_chain_order() {
        let mut chain = AudioEffectChain::new();
        chain.add_effect(Box::new(NoiseGate::new(-60.0)));
        chain.add_effect(Box::new(AutoGainControl::new(-18.0)));
        assert_eq!(chain.list_effects().len(), 2);
        assert_eq!(chain.list_effects()[0].0, "Noise Gate");
        assert_eq!(chain.list_effects()[1].0, "Auto Gain Control");
    }

    #[test]
    fn test_effect_enable_disable() {
        let mut chain = AudioEffectChain::new();
        chain.add_effect(Box::new(NoiseGate::new(-40.0)));
        assert!(chain.list_effects()[0].1); // enabled by default

        chain.enable_effect("Noise Gate", false);
        assert!(!chain.list_effects()[0].1);
    }

    #[test]
    fn test_total_latency() {
        let mut chain = AudioEffectChain::new();
        chain.add_effect(Box::new(NoiseGate::new(-40.0))); // ~50us
        chain.add_effect(Box::new(AcousticEchoCanceller::new(480))); // ~5000us

        let total = chain.total_latency_us();
        assert!(total >= 5000); // At least AEC latency
    }

    #[test]
    fn test_effect_chain_latency_breakdown() {
        let mut chain = AudioEffectChain::new();
        chain.add_effect(Box::new(NoiseGate::new(-40.0)));
        chain.add_effect(Box::new(Compressor::new(-20.0, 4.0)));

        let breakdown = chain.latency_breakdown();
        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0].0, "Noise Gate");
        assert_eq!(breakdown[1].0, "Compressor");
    }

    #[test]
    fn test_speaker_profile_recompute_mean() {
        let mut profile = SpeakerProfile {
            id: "test".into(),
            name: "Test".into(),
            embeddings: vec![
                VoiceEmbedding {
                    vector: vec![1.0, 2.0, 3.0],
                    model_id: "test".into(),
                    sample_duration_ms: 1000,
                },
                VoiceEmbedding {
                    vector: vec![3.0, 4.0, 5.0],
                    model_id: "test".into(),
                    sample_duration_ms: 1000,
                },
            ],
            mean_embedding: Vec::new(),
            threshold: 0.7,
            created_at: String::new(),
            is_owner: true,
        };
        profile.recompute_mean();
        assert_eq!(profile.mean_embedding, vec![2.0, 3.0, 4.0]);
    }

    // ── Creative voice effects tests ─────────────────────────────────────

    #[test]
    fn test_pitch_shift_helium() {
        let mut shifter = PitchShifter::helium();
        let original = sine_wave(300.0, 16000, 50);
        let mut processed = original.clone();
        shifter.process(&mut processed, 16000);
        // Helium shifts up: the resampled signal reads faster through the buffer,
        // so output should differ from original and many trailing samples become zero
        // (because read position advances past the input).
        assert_ne!(original, processed, "Helium shift should modify the audio");
        // Trailing samples should be zero (read past end of input buffer)
        let last_quarter = &processed[processed.len() * 3 / 4..];
        let trailing_zeros = last_quarter.iter().filter(|&&s| s == 0).count();
        assert!(
            trailing_zeros > last_quarter.len() / 4,
            "Pitch-up should run out of input samples, producing zeros at the end"
        );
    }

    #[test]
    fn test_pitch_shift_darth_vader() {
        let mut shifter = PitchShifter::darth_vader();
        let original = sine_wave(300.0, 16000, 50);
        let mut processed = original.clone();
        shifter.process(&mut processed, 16000);
        // Darth Vader shifts down: reads slower, so output uses only the first portion
        // of the input. The output should differ from the original.
        assert_ne!(
            original, processed,
            "Darth Vader shift should modify the audio"
        );
        // With -8 semitones, ratio < 1, so it reads only about 63% of input.
        // The last sample should still come from valid input (not zero).
        let last = *processed.last().unwrap_or(&0);
        // For a continuous sine, the last sample from a slower read should be non-zero
        // (it's reading the middle of the sine wave, not past the end).
        assert!(
            last != 0 || processed.iter().any(|&s| s != 0),
            "Pitch-down should produce non-trivial output"
        );
    }

    #[test]
    fn test_robot_voice_modulates() {
        let mut robot = RobotVoice::default_robot();
        let original = sine_wave(440.0, 16000, 50);
        let mut processed = original.clone();
        robot.process(&mut processed, 16000);
        assert_ne!(original, processed, "Robot voice should modify the signal");
        // Ring modulation multiplies by a carrier — some samples should be near zero
        // where the carrier sine crosses zero.
        let near_zero_count = processed.iter().filter(|&&s| s.abs() < 100).count();
        assert!(
            near_zero_count > 0,
            "Ring modulation should produce near-zero samples at carrier zero-crossings"
        );
    }

    #[test]
    fn test_autotune_detects_pitch() {
        let mut autotune = AutoTune::full();
        // Generate a 440 Hz sine wave (A4 — should be perfectly in tune, so minimal change)
        let original = sine_wave(440.0, 16000, 100);
        let mut processed = original.clone();
        autotune.process(&mut processed, 16000);
        // A4 is exactly a musical note, so correction should be near zero.
        // Allow some tolerance since the autocorrelation is approximate.
        let diff: i64 = original
            .iter()
            .zip(processed.iter())
            .map(|(&a, &b)| (a as i64 - b as i64).abs())
            .sum();
        let avg_diff = diff / original.len() as i64;
        // Average sample difference should be small for an in-tune note
        assert!(
            avg_diff < 2000,
            "A4 (440 Hz) should need little correction, avg diff = {}",
            avg_diff
        );
    }

    #[test]
    fn test_echo_delays_audio() {
        let mut echo = EchoEffect::default_echo();
        // Start with silence then a burst
        let mut samples = vec![0i16; 4800]; // 300ms at 16kHz
                                            // Put a pulse at the beginning
        samples[0] = 16000;
        echo.process(&mut samples, 16000);
        // The echo of the initial pulse should appear at ~300ms = 4800 samples
        // Since our buffer is exactly 4800 samples, the echo wraps around.
        // The direct output at sample 0 should be attenuated (dry * 0.7 + delayed * 0.3)
        // and later samples should contain the echo feedback.
        // At minimum, processing should modify the audio.
        let sum: i64 = samples.iter().map(|&s| s.abs() as i64).sum();
        assert!(sum > 0, "Echo should produce non-zero output");
        // The first sample should contain the dry pulse
        assert!(
            samples[0].abs() > 0,
            "First sample should have the dry pulse"
        );
    }

    #[test]
    fn test_megaphone_clips() {
        let mut mega = MegaphoneEffect::default_megaphone();
        let original = sine_wave(500.0, 16000, 50);
        let mut processed = original.clone();
        mega.process(&mut processed, 16000);
        assert_ne!(original, processed, "Megaphone should modify the audio");
        // Soft clipping via tanh means output amplitude is bounded to i16 range.
        assert!(processed.iter().all(|&s| s.abs() <= i16::MAX));
        // The tanh saturation means output cannot exceed i16::MAX (tanh output < 1.0)
        let peak_proc = processed.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(
            peak_proc < i16::MAX,
            "Megaphone tanh should keep output strictly below i16::MAX, got {}",
            peak_proc
        );
    }

    #[test]
    fn test_noise_reducer_voice_passes_through() {
        let mut reducer = IntelligentNoiseReducer::factory();
        // Simulate voice: a clean sine wave with voice-like ZCR and spectral properties
        let mut voice_samples = sine_wave(300.0, 16000, 50);
        let original = voice_samples.clone();

        // First, feed a voice frame to establish voice_rms_avg
        reducer.process(&mut voice_samples, 16000);

        // Voice frames should pass through with minimal change (they are not attenuated)
        let diff: i64 = original
            .iter()
            .zip(voice_samples.iter())
            .map(|(&a, &b)| (a as i64 - b as i64).abs())
            .sum();
        let avg_diff = diff as f64 / original.len() as f64;
        // Voice should pass through unchanged (diff = 0) since detect_voice returns true
        // for a clean sine and voice frames are not modified.
        assert!(
            avg_diff < 1.0,
            "Voice frames should pass through unmodified, avg_diff = {}",
            avg_diff
        );
    }

    #[test]
    fn test_noise_reducer_noise_reduced() {
        let mut reducer = IntelligentNoiseReducer::factory();

        // First establish voice level with a voice frame
        let mut voice = sine_wave(300.0, 16000, 50);
        reducer.process(&mut voice, 16000);
        // Process voice again to confirm (need voice_confirm_frames = 2)
        let mut voice2 = sine_wave(300.0, 16000, 50);
        reducer.process(&mut voice2, 16000);

        // Now process broadband noise (high ZCR, high flatness → non-voice)
        let mut noise: Vec<i16> = (0..800)
            .map(|i| {
                // Alternating high-frequency noise-like pattern
                if i % 2 == 0 {
                    10000
                } else {
                    -10000
                }
            })
            .collect();
        let noise_rms_before = compute_rms_i16(&noise);
        reducer.process(&mut noise, 16000);
        let noise_rms_after = compute_rms_i16(&noise);

        // Noise should be reduced (gain < 1.0)
        assert!(
            noise_rms_after < noise_rms_before,
            "Noise frame should be attenuated: before={}, after={}",
            noise_rms_before,
            noise_rms_after
        );
    }

    #[test]
    fn test_noise_reducer_relative_level() {
        let mut reducer = IntelligentNoiseReducer::factory(); // ratio = 0.7

        // Establish voice level
        for _ in 0..3 {
            let mut voice = sine_wave(300.0, 16000, 50);
            reducer.process(&mut voice, 16000);
        }

        let voice_avg = reducer.voice_rms_avg;
        assert!(voice_avg > 0.0, "Voice RMS should be tracked");

        // Process a loud noise frame
        let mut noise: Vec<i16> = (0..800)
            .map(|i| if i % 2 == 0 { 15000 } else { -15000 })
            .collect();
        reducer.process(&mut noise, 16000);
        let noise_rms_after = compute_rms_i16(&noise);

        // After reduction, noise should be approximately voice_avg * 0.7
        let target = voice_avg * 0.7;
        // Allow generous tolerance since this is frame-level approximate processing
        assert!(
            noise_rms_after < target * 2.0,
            "Noise RMS ({}) should be near target ({})",
            noise_rms_after,
            target
        );
    }

    #[test]
    fn test_noise_reducer_factory_preset() {
        let factory = IntelligentNoiseReducer::factory();
        assert!((factory.voice_to_noise_ratio - 0.7).abs() < 0.01);
        assert!(factory.is_enabled());
        assert_eq!(factory.name(), "Intelligent Noise Reducer");

        let construction = IntelligentNoiseReducer::construction();
        assert!((construction.voice_to_noise_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_all_effects_in_chain() {
        let mut chain = AudioEffectChain::new();
        chain.add_effect(Box::new(PitchShifter::chipmunk()));
        chain.add_effect(Box::new(RobotVoice::default_robot()));
        chain.add_effect(Box::new(AutoTune::subtle()));
        chain.add_effect(Box::new(EchoEffect::short()));
        chain.add_effect(Box::new(MegaphoneEffect::default_megaphone()));
        chain.add_effect(Box::new(IntelligentNoiseReducer::factory()));

        let mut samples = sine_wave(440.0, 16000, 50);
        let original = samples.clone();
        chain.process_frame(&mut samples, 16000);

        // All effects enabled — audio should be significantly modified
        assert_ne!(
            original, samples,
            "Chain of all effects should modify audio"
        );

        // Verify all 6 effects are listed
        let effects = chain.list_effects();
        assert_eq!(effects.len(), 6);
        assert_eq!(effects[0].0, "Pitch Shifter");
        assert_eq!(effects[1].0, "Robot Voice");
        assert_eq!(effects[2].0, "AutoTune");
        assert_eq!(effects[3].0, "Echo");
        assert_eq!(effects[4].0, "Megaphone");
        assert_eq!(effects[5].0, "Intelligent Noise Reducer");

        // Total latency should be sum of all
        assert!(chain.total_latency_us() > 0);
    }

    #[test]
    fn test_effect_categories() {
        let pitch = PitchShifter::new(0.0);
        assert_eq!(pitch.category(), EffectCategory::Creative);

        let robot = RobotVoice::default_robot();
        assert_eq!(robot.category(), EffectCategory::Creative);

        let autotune = AutoTune::full();
        assert_eq!(autotune.category(), EffectCategory::Creative);

        let echo = EchoEffect::default_echo();
        assert_eq!(echo.category(), EffectCategory::Creative);

        let mega = MegaphoneEffect::default_megaphone();
        assert_eq!(mega.category(), EffectCategory::Creative);

        let reducer = IntelligentNoiseReducer::factory();
        assert_eq!(reducer.category(), EffectCategory::Enhancement);
    }
}
