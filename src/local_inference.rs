//! In-process local LLM inference (V108 / Phase A.3, base scaffolding).
//!
//! Defines the [`Backend`] trait, [`LocalInferenceConfig`] builder, VRAM
//! detection helpers, and the GPU-layer auto-clamp policy. Backend
//! implementations (Candle, llama-cpp-2) live behind sub-features so users
//! who only need the trait surface (e.g. for stub testing) don't pull the
//! native deps.
//!
//! Out of scope here (deferred): real Candle / llama-cpp-2 integration,
//! provider dispatch wiring, dedicated bin + auditor pair. Each of those is
//! a separate task that builds on this module.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Candle native Rust backend. Safetensors models (HuggingFace format).
    Candle,
    /// llama-cpp-2 backend. GGUF models, broad model coverage, ABI-fragile.
    LlamaCpp,
    /// In-tree stub. Echoes prompts. For tests that exercise the trait
    /// surface without pulling native deps.
    Stub,
}

impl BackendKind {
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Candle => "candle",
            BackendKind::LlamaCpp => "llama-cpp-2",
            BackendKind::Stub => "stub",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalInferenceConfig {
    pub kind: BackendKind,
    pub model_path: PathBuf,
    pub ctx_size: u32,
    pub n_gpu_layers: u32,
    /// If true, [`Backend::load`] consults VRAM and reduces `n_gpu_layers`
    /// rather than letting the backend OOM. Defaults to true.
    pub allow_gpu_clamp: bool,
    /// Optional model size override in MiB. When `None`, the backend is
    /// expected to read it from the file. Used by the clamp policy.
    pub model_size_mib: Option<u64>,
}

impl LocalInferenceConfig {
    pub fn builder<P: Into<PathBuf>>(
        kind: BackendKind,
        model_path: P,
    ) -> LocalInferenceConfigBuilder {
        LocalInferenceConfigBuilder {
            kind,
            model_path: model_path.into(),
            ctx_size: 4096,
            n_gpu_layers: 0,
            allow_gpu_clamp: true,
            model_size_mib: None,
        }
    }
}

pub struct LocalInferenceConfigBuilder {
    kind: BackendKind,
    model_path: PathBuf,
    ctx_size: u32,
    n_gpu_layers: u32,
    allow_gpu_clamp: bool,
    model_size_mib: Option<u64>,
}

impl LocalInferenceConfigBuilder {
    pub fn ctx_size(mut self, n: u32) -> Self {
        self.ctx_size = n;
        self
    }
    pub fn n_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = n;
        self
    }
    pub fn allow_gpu_clamp(mut self, on: bool) -> Self {
        self.allow_gpu_clamp = on;
        self
    }
    pub fn model_size_mib(mut self, n: u64) -> Self {
        self.model_size_mib = Some(n);
        self
    }
    pub fn build(self) -> LocalInferenceConfig {
        LocalInferenceConfig {
            kind: self.kind,
            model_path: self.model_path,
            ctx_size: self.ctx_size,
            n_gpu_layers: self.n_gpu_layers,
            allow_gpu_clamp: self.allow_gpu_clamp,
            model_size_mib: self.model_size_mib,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stop: Vec<String>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            stop: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub time_ms: u64,
    pub tokens_per_sec: f64,
    pub peak_vram_mib: Option<u64>,
}

#[derive(Debug)]
pub enum BackendError {
    NotImplemented(&'static str),
    ModelNotFound(PathBuf),
    Io(std::io::Error),
    Backend(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotImplemented(name) => write!(f, "backend not compiled in: {name}"),
            BackendError::ModelNotFound(p) => write!(f, "model not found: {}", p.display()),
            BackendError::Io(e) => write!(f, "io error: {e}"),
            BackendError::Backend(s) => write!(f, "backend error: {s}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<std::io::Error> for BackendError {
    fn from(e: std::io::Error) -> Self {
        BackendError::Io(e)
    }
}

pub trait Backend: Send {
    /// Generate from a single prompt. Streams chunks via `on_chunk`.
    /// Returns aggregate stats on success.
    fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError>;

    /// Loaded backend kind (mirrors the config). Useful for SLO records.
    fn kind(&self) -> BackendKind;

    /// Optional explicit unload. Defaults to no-op; `Drop` typically suffices.
    fn unload(&mut self) {}
}

/// Load a backend from a config. Until concrete backends are wired, only
/// [`BackendKind::Stub`] returns a working backend; the others surface
/// `NotImplemented` so callers can detect missing features at runtime.
pub fn load(config: &LocalInferenceConfig) -> Result<Box<dyn Backend>, BackendError> {
    if !config.model_path.as_os_str().is_empty()
        && config.kind != BackendKind::Stub
        && !config.model_path.exists()
    {
        return Err(BackendError::ModelNotFound(config.model_path.clone()));
    }
    match config.kind {
        BackendKind::Stub => Ok(Box::new(StubBackend::new())),
        BackendKind::Candle => {
            #[cfg(feature = "local-inference-candle")]
            {
                crate::local_inference_candle::load_candle(config)
            }
            #[cfg(not(feature = "local-inference-candle"))]
            {
                Err(BackendError::NotImplemented("candle"))
            }
        }
        BackendKind::LlamaCpp => Err(BackendError::NotImplemented("llama-cpp-2")),
    }
}

/// Echoes a fixed prefix + the prompt back as one chunk. Lets tests exercise
/// the trait + SLO-record machinery without native deps.
pub struct StubBackend {
    loaded_at: std::time::Instant,
}

impl StubBackend {
    pub fn new() -> Self {
        Self {
            loaded_at: std::time::Instant::now(),
        }
    }
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for StubBackend {
    fn generate(
        &mut self,
        prompt: &str,
        _params: &GenParams,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        let start = std::time::Instant::now();
        let response = format!("[stub] {prompt}");
        on_chunk(&response);
        let elapsed = start.elapsed();
        let tokens = response.split_whitespace().count() as u32;
        let secs = elapsed.as_secs_f64().max(1e-9);
        Ok(GenStats {
            prompt_tokens: prompt.split_whitespace().count() as u32,
            generated_tokens: tokens,
            time_ms: elapsed.as_millis() as u64,
            tokens_per_sec: tokens as f64 / secs,
            peak_vram_mib: None,
        })
    }
    fn kind(&self) -> BackendKind {
        BackendKind::Stub
    }
}

impl StubBackend {
    /// How long the stub has been alive — useful for tests that want to
    /// distinguish a freshly loaded stub from a long-lived one.
    pub fn age_ms(&self) -> u64 {
        self.loaded_at.elapsed().as_millis() as u64
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SloRecord {
    pub ts_unix_ms: u64,
    pub backend: String,
    pub model_path: String,
    pub load_ms: u64,
    pub first_chunk_ms: u64,
    pub total_ms: u64,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub tokens_per_sec: f64,
    pub n_gpu_layers_requested: u32,
    pub n_gpu_layers_used: u32,
    pub peak_vram_mib: Option<u64>,
}

impl SloRecord {
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

pub mod vram {
    //! VRAM detection + GPU-layer clamp policy.
    //!
    //! Detection is best-effort: we shell out to `nvidia-smi` when present
    //! and otherwise return `None`. The clamp policy is pure — it takes the
    //! detected VRAM as input and is fully unit-testable.

    use std::process::Command;

    /// Total + free VRAM on the first NVIDIA device (MiB), if any.
    /// Returns `(total, free)`. None if no NVIDIA device or no nvidia-smi.
    pub fn detect_nvidia_mib() -> Option<(u64, u64)> {
        let out = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().next()?;
        let mut parts = line.split(',').map(|s| s.trim());
        let total: u64 = parts.next()?.parse().ok()?;
        let free: u64 = parts.next()?.parse().ok()?;
        Some((total, free))
    }

    /// Available VRAM on the primary GPU (MiB). None if no GPU detected.
    pub fn detect_available_mib() -> Option<u64> {
        detect_nvidia_mib().map(|(_, free)| free)
    }

    /// GPU-layer auto-clamp policy.
    ///
    /// Reduces `requested_layers` so the model fits in `available_mib`
    /// instead of OOMing. Assumes layer cost is approximately linear in the
    /// total (a reasonable approximation for transformer models, where the
    /// dominant per-layer cost is the attention + MLP weights).
    ///
    /// Inputs:
    /// - `model_size_mib`: total model weight size on disk.
    /// - `requested_layers`: layers the user asked to offload.
    /// - `total_layers`: layers in the model (e.g. 32 for Llama-3 8B).
    /// - `available_mib`: free VRAM, with a safety margin already removed.
    ///
    /// Edge cases:
    /// - `total_layers == 0` returns `requested_layers` unchanged (caller
    ///   probably hasn't loaded the model yet — nothing to clamp against).
    /// - `requested_layers > total_layers` is treated as `total_layers`.
    /// - `available_mib == 0` clamps to 0 (CPU only).
    pub fn clamp_gpu_layers(
        model_size_mib: u64,
        requested_layers: u32,
        total_layers: u32,
        available_mib: u64,
    ) -> u32 {
        if total_layers == 0 {
            return requested_layers;
        }
        let req = requested_layers.min(total_layers);
        if req == 0 {
            return 0;
        }
        if available_mib == 0 {
            return 0;
        }
        let per_layer_mib = model_size_mib as f64 / total_layers as f64;
        if per_layer_mib <= 0.0 {
            return req;
        }
        let max_fittable = (available_mib as f64 / per_layer_mib).floor() as u32;
        req.min(max_fittable.min(total_layers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_defaults() {
        let cfg = LocalInferenceConfig::builder(BackendKind::Stub, "/tmp/m").build();
        assert_eq!(cfg.kind, BackendKind::Stub);
        assert_eq!(cfg.ctx_size, 4096);
        assert_eq!(cfg.n_gpu_layers, 0);
        assert!(cfg.allow_gpu_clamp);
        assert_eq!(cfg.model_size_mib, None);
    }

    #[test]
    fn config_builder_chains() {
        let cfg = LocalInferenceConfig::builder(BackendKind::Candle, "/tmp/m")
            .ctx_size(8192)
            .n_gpu_layers(20)
            .allow_gpu_clamp(false)
            .model_size_mib(4500)
            .build();
        assert_eq!(cfg.ctx_size, 8192);
        assert_eq!(cfg.n_gpu_layers, 20);
        assert!(!cfg.allow_gpu_clamp);
        assert_eq!(cfg.model_size_mib, Some(4500));
    }

    #[test]
    fn stub_backend_generates_and_streams() {
        let mut backend = StubBackend::new();
        let mut chunks: Vec<String> = Vec::new();
        let stats = backend
            .generate("hello world", &GenParams::default(), &mut |c| {
                chunks.push(c.to_string())
            })
            .unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].contains("hello world"));
        assert_eq!(stats.prompt_tokens, 2);
        assert!(stats.generated_tokens >= 2);
    }

    #[test]
    fn load_stub_succeeds() {
        let cfg = LocalInferenceConfig::builder(BackendKind::Stub, "").build();
        let backend = load(&cfg).expect("stub loads");
        assert_eq!(backend.kind(), BackendKind::Stub);
    }

    #[test]
    fn load_candle_unimplemented() {
        let cfg = LocalInferenceConfig::builder(BackendKind::Candle, "/tmp/x.safetensors").build();
        let r = load(&cfg);
        assert!(matches!(
            r,
            Err(BackendError::NotImplemented("candle")) | Err(BackendError::ModelNotFound(_))
        ));
    }

    #[test]
    fn load_missing_real_model_errors() {
        let cfg =
            LocalInferenceConfig::builder(BackendKind::Candle, "/nonexistent/model.bin").build();
        let r = load(&cfg);
        assert!(matches!(r, Err(BackendError::ModelNotFound(_))));
    }

    #[test]
    fn clamp_zero_total_layers_passthrough() {
        assert_eq!(vram::clamp_gpu_layers(1000, 32, 0, 8000), 32);
    }

    #[test]
    fn clamp_zero_requested_returns_zero() {
        assert_eq!(vram::clamp_gpu_layers(4000, 0, 32, 8000), 0);
    }

    #[test]
    fn clamp_zero_vram_returns_zero() {
        assert_eq!(vram::clamp_gpu_layers(4000, 32, 32, 0), 0);
    }

    #[test]
    fn clamp_fits_returns_requested() {
        // 4000 MiB model, 32 layers → 125 MiB/layer. 8 GiB free → all 32 fit.
        assert_eq!(vram::clamp_gpu_layers(4000, 32, 32, 8000), 32);
    }

    #[test]
    fn clamp_undersized_reduces() {
        // 8000 MiB model, 32 layers → 250 MiB/layer. 2000 MiB free → 8 layers.
        assert_eq!(vram::clamp_gpu_layers(8000, 32, 32, 2000), 8);
    }

    #[test]
    fn clamp_request_above_total_treated_as_total() {
        // Asked for more layers than the model has — caps to total first.
        assert_eq!(vram::clamp_gpu_layers(4000, 99, 32, 8000), 32);
    }

    #[test]
    fn clamp_partial_fit_below_request() {
        // 16 GiB model / 80 layers → 200 MiB/layer. 4 GiB free → 20 layers.
        // User requested 40 → clamp to 20.
        assert_eq!(vram::clamp_gpu_layers(16000, 40, 80, 4000), 20);
    }

    #[test]
    fn slo_record_serializes() {
        let r = SloRecord {
            ts_unix_ms: 123,
            backend: "stub".into(),
            model_path: "/tmp/m".into(),
            load_ms: 10,
            first_chunk_ms: 20,
            total_ms: 30,
            prompt_tokens: 5,
            generated_tokens: 7,
            tokens_per_sec: 70.0,
            n_gpu_layers_requested: 32,
            n_gpu_layers_used: 16,
            peak_vram_mib: Some(2048),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"backend\":\"stub\""));
        assert!(json.contains("\"n_gpu_layers_used\":16"));
        let back: SloRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generated_tokens, 7);
    }
}
