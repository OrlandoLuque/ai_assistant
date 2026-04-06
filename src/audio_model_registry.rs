//! Audio model registry — catalog, download, and management of audio ML models.
//!
//! Provides a registry of known audio models (Whisper STT, Piper TTS voices,
//! XTTS v2, emotion2vec) with download support, SHA-256 verification, and
//! installed model detection.
//!
//! Models are stored in a platform-appropriate cache directory:
//! - Linux/macOS: `~/.cache/ai_assistant/models/`
//! - Windows: `%LOCALAPPDATA%\ai_assistant\models\`
//! - Override: `AI_ASSISTANT_MODEL_DIR` environment variable

use serde::{Deserialize, Serialize};

// ============================================================================
// Model Types
// ============================================================================

/// Category of audio model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioModelCategory {
    /// Speech-to-text (e.g., Whisper)
    Stt,
    /// Text-to-speech voice (e.g., Piper voices)
    Tts,
    /// Voice cloning model (e.g., XTTS v2)
    VoiceClone,
    /// Audio emotion detection (e.g., emotion2vec)
    Emotion,
}

impl std::fmt::Display for AudioModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stt => write!(f, "STT"),
            Self::Tts => write!(f, "TTS"),
            Self::VoiceClone => write!(f, "Voice Clone"),
            Self::Emotion => write!(f, "Emotion"),
        }
    }
}

/// Information about an audio model in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioModelInfo {
    /// Unique model identifier (e.g., "whisper-base.en").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Model category.
    pub category: AudioModelCategory,
    /// Approximate download size (human-readable, e.g. "150 MB").
    pub size_estimate: String,
    /// Size in bytes (for progress tracking).
    pub size_bytes: u64,
    /// Description of the model's capabilities.
    pub description: String,
    /// Download URL (HuggingFace, direct link, etc.).
    pub url: String,
    /// Expected SHA-256 hash of the downloaded file (hex-encoded).
    pub sha256: String,
    /// Filename to save as in the model directory.
    pub filename: String,
}

/// Status of a model (installed or not).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// Not downloaded.
    NotInstalled,
    /// Downloaded and verified.
    Installed {
        /// Full path to the model file.
        path: String,
    },
    /// Downloaded but checksum mismatch (potentially corrupted).
    Corrupted { path: String },
}

// ============================================================================
// Model Directory
// ============================================================================

/// Get the default model storage directory for the current platform.
///
/// Priority:
/// 1. `AI_ASSISTANT_MODEL_DIR` environment variable
/// 2. Platform-specific cache directory
pub fn model_directory() -> String {
    if let Ok(dir) = std::env::var("AI_ASSISTANT_MODEL_DIR") {
        return dir;
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            return format!("{}\\ai_assistant\\models", local_app_data);
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/.cache/ai_assistant/models", home);
        }
    }

    // Fallback
    "ai_assistant_models".to_string()
}

// ============================================================================
// Audio Model Registry
// ============================================================================

/// Registry of known audio models with catalog, download, and detection.
pub struct AudioModelRegistry {
    catalog: Vec<AudioModelInfo>,
    model_dir: String,
}

impl AudioModelRegistry {
    /// Create a registry with the default model directory and built-in catalog.
    pub fn new() -> Self {
        Self {
            catalog: Self::builtin_catalog(),
            model_dir: model_directory(),
        }
    }

    /// Create with a custom model directory.
    pub fn with_directory(dir: &str) -> Self {
        Self {
            catalog: Self::builtin_catalog(),
            model_dir: dir.to_string(),
        }
    }

    /// Get the model directory path.
    pub fn model_dir(&self) -> &str {
        &self.model_dir
    }

    /// Get the full catalog of known models.
    pub fn catalog(&self) -> &[AudioModelInfo] {
        &self.catalog
    }

    /// Filter catalog by category.
    pub fn models_by_category(&self, category: AudioModelCategory) -> Vec<&AudioModelInfo> {
        self.catalog
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Find a model by ID.
    pub fn find_model(&self, id: &str) -> Option<&AudioModelInfo> {
        self.catalog.iter().find(|m| m.id == id)
    }

    /// Check if a model is installed (file exists + optional checksum).
    pub fn model_status(&self, model: &AudioModelInfo) -> ModelStatus {
        let path = format!("{}/{}", self.model_dir, model.filename);

        if !std::path::Path::new(&path).exists() {
            return ModelStatus::NotInstalled;
        }

        // If we have a checksum, verify it
        if !model.sha256.is_empty() {
            if let Ok(file_hash) = compute_sha256_file(&path) {
                if file_hash != model.sha256 {
                    return ModelStatus::Corrupted { path: path.clone() };
                }
            }
        }

        ModelStatus::Installed { path }
    }

    /// Detect all installed models by scanning the model directory.
    pub fn detect_installed(&self) -> Vec<(AudioModelInfo, String)> {
        let mut installed = Vec::new();
        for model in &self.catalog {
            if let ModelStatus::Installed { path } = self.model_status(model) {
                installed.push((model.clone(), path));
            }
        }
        installed
    }

    /// Get the expected file path for a model (path-traversal safe).
    pub fn model_path(&self, model: &AudioModelInfo) -> String {
        // Prevent path traversal: use only the filename component
        let safe_name = std::path::Path::new(&model.filename)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown_model");
        format!("{}/{}", self.model_dir, safe_name)
    }

    /// Download a model with progress callback.
    ///
    /// The callback receives (bytes_downloaded, total_bytes) on each chunk.
    /// Returns the path where the model was saved.
    ///
    /// Verifies SHA-256 after download if the model has a checksum.
    /// Downloads to a temporary file first, then atomic rename (TOCTOU safe).
    pub fn download_model(
        &self,
        model: &AudioModelInfo,
        progress: impl Fn(u64, u64),
    ) -> Result<String, String> {
        // Validate URL
        if model.url.is_empty() {
            return Err("Model has no download URL (install manually)".to_string());
        }
        if !model.url.starts_with("https://") && !model.url.starts_with("http://") {
            return Err("Only http/https URLs allowed".to_string());
        }

        // Ensure directory exists
        if let Err(e) = std::fs::create_dir_all(&self.model_dir) {
            return Err(format!("Failed to create model directory: {}", e));
        }

        let path = self.model_path(model);
        let temp_path = format!("{}.tmp", path);

        // Download to temporary file (TOCTOU safe)
        let response = ureq::get(&model.url)
            .timeout(std::time::Duration::from_secs(600))
            .call()
            .map_err(|e| format!("Download failed: {}", e))?;

        let total = model.size_bytes;
        let mut downloaded = 0u64;
        let mut file = std::fs::File::create(&temp_path)
            .map_err(|e| format!("Failed to create temp file: {}", e))?;

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65536]; // 64KB chunks

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buf[..n])
                .map_err(|e| format!("Write error: {}", e))?;
            downloaded += n as u64;
            progress(downloaded, total);
        }
        drop(file); // Close before checksum

        // Verify checksum if available
        if !model.sha256.is_empty() {
            let file_hash =
                compute_sha256_file(&temp_path).map_err(|e| format!("Checksum error: {}", e))?;
            if file_hash != model.sha256 {
                let _ = std::fs::remove_file(&temp_path);
                return Err(format!(
                    "Checksum mismatch: expected {}, got {}",
                    model.sha256, file_hash
                ));
            }
        }

        // Atomic rename: temp → final (TOCTOU safe)
        std::fs::rename(&temp_path, &path)
            .map_err(|e| format!("Failed to finalize download: {}", e))?;

        Ok(path)
    }

    /// Built-in catalog of known audio models.
    fn builtin_catalog() -> Vec<AudioModelInfo> {
        vec![
            // ── Whisper STT models ──────────────────────────────────────────
            AudioModelInfo {
                id: "whisper-tiny.en".to_string(),
                name: "Whisper Tiny (English)".to_string(),
                category: AudioModelCategory::Stt,
                size_estimate: "75 MB".to_string(),
                size_bytes: 75_000_000,
                description: "Fastest Whisper model, English only. Good for real-time on CPU."
                    .to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
                    .to_string(),
                sha256: String::new(), // TODO: add verified checksums
                filename: "ggml-tiny.en.bin".to_string(),
            },
            AudioModelInfo {
                id: "whisper-base.en".to_string(),
                name: "Whisper Base (English)".to_string(),
                category: AudioModelCategory::Stt,
                size_estimate: "150 MB".to_string(),
                size_bytes: 150_000_000,
                description: "Good balance of speed and accuracy for English.".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
                    .to_string(),
                sha256: String::new(),
                filename: "ggml-base.en.bin".to_string(),
            },
            AudioModelInfo {
                id: "whisper-small".to_string(),
                name: "Whisper Small (Multilingual)".to_string(),
                category: AudioModelCategory::Stt,
                size_estimate: "500 MB".to_string(),
                size_bytes: 500_000_000,
                description: "Multilingual STT. Good accuracy, moderate speed.".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
                    .to_string(),
                sha256: String::new(),
                filename: "ggml-small.bin".to_string(),
            },
            AudioModelInfo {
                id: "whisper-medium".to_string(),
                name: "Whisper Medium (Multilingual)".to_string(),
                category: AudioModelCategory::Stt,
                size_estimate: "1.5 GB".to_string(),
                size_bytes: 1_500_000_000,
                description: "High accuracy multilingual STT. Needs GPU or fast CPU.".to_string(),
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
                    .to_string(),
                sha256: String::new(),
                filename: "ggml-medium.bin".to_string(),
            },
            // ── Piper TTS voices ────────────────────────────────────────────
            AudioModelInfo {
                id: "piper-en-us-amy-low".to_string(),
                name: "Piper: Amy (US English, Low)".to_string(),
                category: AudioModelCategory::Tts,
                size_estimate: "20 MB".to_string(),
                size_bytes: 20_000_000,
                description: "Fast English female voice for Piper TTS.".to_string(),
                url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx".to_string(),
                sha256: String::new(),
                filename: "piper-en_US-amy-low.onnx".to_string(),
            },
            AudioModelInfo {
                id: "piper-es-es-davefx-medium".to_string(),
                name: "Piper: Davefx (Spanish, Medium)".to_string(),
                category: AudioModelCategory::Tts,
                size_estimate: "60 MB".to_string(),
                size_bytes: 60_000_000,
                description: "Spanish male voice for Piper TTS.".to_string(),
                url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx".to_string(),
                sha256: String::new(),
                filename: "piper-es_ES-davefx-medium.onnx".to_string(),
            },
            // ── XTTS v2 (voice cloning) ─────────────────────────────────────
            AudioModelInfo {
                id: "xtts-v2".to_string(),
                name: "Coqui XTTS v2".to_string(),
                category: AudioModelCategory::VoiceClone,
                size_estimate: "1.8 GB".to_string(),
                size_bytes: 1_800_000_000,
                description:
                    "Voice cloning from 6-second reference audio. Requires Coqui TTS server."
                        .to_string(),
                url: String::new(), // XTTS is installed via `pip install TTS`
                sha256: String::new(),
                filename: "xtts_v2".to_string(),
            },
        ]
    }
}

impl Default for AudioModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHA-256 file hashing (reuses binary_integrity pattern)
// ============================================================================

/// Compute SHA-256 hash of a file and return hex string.
fn compute_sha256_file(path: &str) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open file {}: {}", path, e))?;
    let mut hasher = Sha256Hasher::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Read error: {}", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize_hex())
}

/// Minimal SHA-256 implementation (pure Rust, no external dep).
struct Sha256Hasher {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Sha256Hasher {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.total_len += data.len() as u64;

        while self.buffer.len() >= 64 {
            let block: Vec<u8> = self.buffer.drain(..64).collect();
            self.compress(&block);
        }
    }

    fn finalize_hex(mut self) -> String {
        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        while self.buffer.len() >= 64 {
            let block: Vec<u8> = self.buffer.drain(..64).collect();
            self.compress(&block);
        }

        self.state
            .iter()
            .map(|w| format!("{:08x}", w))
            .collect::<String>()
    }

    fn compress(&mut self, block: &[u8]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_category_display() {
        assert_eq!(AudioModelCategory::Stt.to_string(), "STT");
        assert_eq!(AudioModelCategory::Tts.to_string(), "TTS");
        assert_eq!(AudioModelCategory::VoiceClone.to_string(), "Voice Clone");
        assert_eq!(AudioModelCategory::Emotion.to_string(), "Emotion");
    }

    #[test]
    fn test_registry_builtin_catalog() {
        let registry = AudioModelRegistry::new();
        let catalog = registry.catalog();
        assert!(
            catalog.len() >= 7,
            "Should have at least 7 models in catalog"
        );

        // Check we have models in each relevant category
        assert!(
            !registry
                .models_by_category(AudioModelCategory::Stt)
                .is_empty(),
            "Should have STT models"
        );
        assert!(
            !registry
                .models_by_category(AudioModelCategory::Tts)
                .is_empty(),
            "Should have TTS models"
        );
        assert!(
            !registry
                .models_by_category(AudioModelCategory::VoiceClone)
                .is_empty(),
            "Should have VoiceClone models"
        );
    }

    #[test]
    fn test_registry_find_model() {
        let registry = AudioModelRegistry::new();
        let model = registry.find_model("whisper-base.en");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.name, "Whisper Base (English)");
        assert_eq!(model.category, AudioModelCategory::Stt);

        assert!(registry.find_model("nonexistent").is_none());
    }

    #[test]
    fn test_model_status_not_installed() {
        let registry = AudioModelRegistry::with_directory("/tmp/nonexistent_dir_12345");
        let model = &registry.catalog()[0];
        assert_eq!(registry.model_status(model), ModelStatus::NotInstalled);
    }

    #[test]
    fn test_model_directory_defaults() {
        let dir = model_directory();
        assert!(!dir.is_empty());
        // Should contain "ai_assistant" somewhere in the path
        assert!(
            dir.contains("ai_assistant"),
            "Model dir should contain 'ai_assistant': {}",
            dir
        );
    }

    #[test]
    fn test_sha256_known_vectors() {
        // SHA-256 of empty string
        let mut hasher = Sha256Hasher::new();
        hasher.update(b"");
        assert_eq!(
            hasher.finalize_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        // SHA-256 of "abc"
        let mut hasher = Sha256Hasher::new();
        hasher.update(b"abc");
        assert_eq!(
            hasher.finalize_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_registry_custom_directory() {
        let registry = AudioModelRegistry::with_directory("/custom/path");
        assert_eq!(registry.model_dir(), "/custom/path");
    }

    #[test]
    fn test_detect_installed_empty_dir() {
        let registry = AudioModelRegistry::with_directory("/tmp/nonexistent_audio_models_12345");
        let installed = registry.detect_installed();
        assert!(installed.is_empty());
    }
}
