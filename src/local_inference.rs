//! In-process local LLM inference (V108 / Phase A.3, base scaffolding).
//!
//! Defines the [`Backend`] trait, [`LocalInferenceConfig`] builder, VRAM
//! detection helpers, and the GPU-layer auto-clamp policy. Backend
//! implementations (Candle, llama-cpp-2) live behind sub-features so users
//! who only need the trait surface (e.g. for stub testing) don't pull the
//! native deps.
//!
//! # What is done and what is not, as of 2026-09-19
//!
//! This header used to say the Candle and llama-cpp-2 integrations were
//! deferred. A sentence that says "not yet" about something that exists hides a
//! capability — which is the same defect as claiming one that does not, with
//! the cost on the other side. Someone reading this module concluded the crate
//! could not run a model in process, when it can. Keeping the list honest is
//! therefore part of the module, not bookkeeping around it.
//!
//! * **Candle backend** — done in V110 (`local_inference_candle`).
//! * **llama-cpp-2 backend** — done in V112 (`local_inference_llama_cpp`),
//!   real bindings, CPU only.
//! * **Provider dispatch wiring** — **done in V330**
//!   ([`crate::LocalInferenceProvider`]). It had been the gap that mattered for
//!   eighteen versions: loading worked and nobody could ask for it through the
//!   normal path, so in-process inference sat outside every decorator the crate
//!   wraps generation in. It is now an [`crate::llm_provider::LlmProvider`],
//!   injected with [`crate::AiAssistant::set_llm_provider`].
//! * **Dedicated bin + auditor pair** — **not done**. Every other subsystem
//!   that stores or runs things has one.

use std::path::PathBuf;
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
    /// CPU threads for generation. `None` means "ask the machine" — the
    /// backend counts cores and applies [`thread_policy`].
    ///
    /// This exists because llama.cpp's own default is a **fixed 4**, whatever
    /// the machine is. Until V332 nothing overrode it, so a sixteen-core desktop
    /// and a four-core laptop got the same four threads and neither was told.
    pub n_threads: Option<u32>,
}

/// How many CPU threads to generate with, given what the machine has.
///
/// Pure on purpose: the probe that counts cores lives in the llama.cpp backend,
/// because counting *physical* cores needs a dependency and this module's
/// umbrella feature is deliberately dependency-free. Keeping the policy here
/// means it can be tested on machines it was never run on.
///
/// # Why physical and not logical
///
/// `llama_context_default_params()` hardcodes 4 threads whatever the machine
/// is — a fact about llama.cpp's defaults, not about the computer in front of
/// the user. The obvious correction is "use every core", and it is wrong.
///
/// Measured 2026-09-19 on a 4-physical / 8-logical laptop, Bonsai-4B `Q1_0`,
/// same prompt, rounds interleaved so ordering could not favour either:
///
/// | threads | generation |
/// |---|---|
/// | 4 (physical) | 5.16 · 3.46 · 3.47 s |
/// | 8 (logical)  | 16.77 · 16.91 · 16.18 s |
///
/// Filling the SMT siblings made it **five times slower**. So the rule is
/// physical cores, clamped into `[1, logical]` — which is also what llama.cpp's
/// own CLI does for its default, and the measurement agrees with it.
pub fn thread_policy(physical: u32, logical: u32) -> u32 {
    physical.clamp(1, logical.max(1))
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
            n_threads: None,
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
    n_threads: Option<u32>,
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
    /// Pin the CPU thread count. Leave unset and the backend counts the
    /// machine's cores and applies [`thread_policy`].
    pub fn n_threads(mut self, n: u32) -> Self {
        self.n_threads = Some(n);
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
            n_threads: self.n_threads,
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

/// What a backend measured about one generation.
///
/// # Two regimes, two numbers (V334)
///
/// Reading the prompt and writing the answer are **not the same workload**:
/// prompt processing runs the whole prompt through the model in parallel and is
/// compute-bound, while generation produces one token at a time and is
/// memory-bandwidth-bound. llama.cpp's own tooling reports them as `pp` and
/// `tg` for exactly that reason.
///
/// Until V334 this struct had a single `tokens_per_sec` = generated ÷ *total*
/// time, which mixed them. Measured 2026-09-19: 32 prompt tokens, **2**
/// generated, 3.5 s → 0.58 "tok/s" — a number that describes neither half. The
/// auditor's SLO (`≥ 5 tok/s`) was checked against it, so a healthy 3.5 s reply
/// read as a breach, and a slow machine answering at length would have passed.
/// A threshold on a quantity that means two things cannot be met or missed.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// Wall-clock for the **whole** call: reading the prompt and writing the
    /// answer. `prompt_ms + generation_ms` plus whatever sits between them.
    pub time_ms: u64,
    /// Time spent evaluating the prompt, before the first token is sampled.
    pub prompt_ms: u64,
    /// Time spent generating, from the first sampled token to the last.
    pub generation_ms: u64,
    /// `prompt_tokens / prompt_ms` — llama.cpp's `pp`. How fast it *reads*.
    pub prompt_tokens_per_sec: f64,
    /// `generated_tokens / generation_ms` — llama.cpp's `tg`. How fast it
    /// *writes*, and the one an SLO on responsiveness should use.
    pub generation_tokens_per_sec: f64,
    pub peak_vram_mib: Option<u64>,
}

impl GenStats {
    /// Build the derived rates from the counts and the two durations, so the
    /// three backends cannot each divide slightly differently.
    ///
    /// Zero elapsed is floored rather than propagated as infinity: a rate of
    /// `inf` in a log reads as a sensor fault, and the honest reading of "too
    /// fast to time" is "we cannot say", not "infinitely fast".
    pub fn with_timings(
        prompt_tokens: u32,
        generated_tokens: u32,
        prompt: std::time::Duration,
        generation: std::time::Duration,
    ) -> Self {
        let rate = |tokens: u32, d: std::time::Duration| -> f64 {
            if tokens == 0 {
                return 0.0;
            }
            tokens as f64 / d.as_secs_f64().max(1e-6)
        };
        Self {
            prompt_tokens,
            generated_tokens,
            time_ms: (prompt + generation).as_millis() as u64,
            prompt_ms: prompt.as_millis() as u64,
            generation_ms: generation.as_millis() as u64,
            prompt_tokens_per_sec: rate(prompt_tokens, prompt),
            generation_tokens_per_sec: rate(generated_tokens, generation),
            peak_vram_mib: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("backend not compiled in: {0}")]
    NotImplemented(&'static str),
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("backend error: {0}")]
    Backend(String),
}

impl crate::error_taxonomy::ErrorCode for BackendError {
    fn code(&self) -> &'static str {
        match self {
            BackendError::NotImplemented(_) => "LOCAL_INFER_NOT_IMPLEMENTED",
            BackendError::ModelNotFound(_) => "LOCAL_INFER_MODEL_NOT_FOUND",
            BackendError::Io(_) => "LOCAL_INFER_IO",
            BackendError::Backend(_) => "LOCAL_INFER_BACKEND",
        }
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            BackendError::NotImplemented(name) => vec![("backend", (*name).to_string())],
            BackendError::ModelNotFound(p) => vec![("path", p.display().to_string())],
            BackendError::Io(e) => vec![("io_kind", format!("{:?}", e.kind()))],
            BackendError::Backend(s) => vec![("detail", s.clone())],
        }
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

/// Load a backend from a config.
///
/// This used to say "until concrete backends are wired, only `Stub` returns a
/// working backend". **They were wired** -- Candle in V110, `llama-cpp-2` in
/// V112 -- and the sentence stayed. It is the second stale claim in this file:
/// a header that says "not yet" about something that exists hides a capability,
/// which costs as much as claiming one that does not.
///
/// What is still true: a `BackendKind` whose feature was not compiled in
/// surfaces `NotImplemented` naming the missing one, so a caller can tell
/// "you did not build this" from "this does not exist".
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
        BackendKind::LlamaCpp => {
            #[cfg(feature = "local-inference-llama-cpp")]
            {
                crate::local_inference_llama_cpp::load_llama_cpp(config)
            }
            #[cfg(not(feature = "local-inference-llama-cpp"))]
            {
                Err(BackendError::NotImplemented("llama-cpp-2"))
            }
        }
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
        // The stub has the same two phases as a real backend, even though both
        // are trivial: it "reads" the prompt and then "writes" the answer.
        // Timing them separately keeps it an honest stand-in — a stub that
        // reported zero for one phase would let a consumer that divides by it
        // pass here and fail against a real model.
        let read_start = std::time::Instant::now();
        let prompt_tokens = prompt.split_whitespace().count() as u32;
        let response = format!("[stub] {prompt}");
        let prompt_elapsed = read_start.elapsed();

        let write_start = std::time::Instant::now();
        on_chunk(&response);
        let generation_elapsed = write_start.elapsed();

        Ok(GenStats::with_timings(
            prompt_tokens,
            response.split_whitespace().count() as u32,
            prompt_elapsed,
            generation_elapsed,
        ))
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

/// One durable line of the local-inference log.
///
/// # Reading records written before V334
///
/// This is persisted as JSONL and there are records on disk from earlier runs,
/// so the wire format stays backward-compatible: the old `tokens_per_sec` key
/// is still read and written, and the four fields V334 adds default to zero on
/// an old line.
///
/// The Rust field was renamed to [`Self::tokens_per_sec_mixed`] even though the
/// JSON key did not change, because the number was never a generation rate —
/// it is generated tokens over *total* time, prompt evaluation included. A
/// rename makes the compiler point at every place that reads it; redefining
/// what the old name means would have let the misreading continue silently,
/// which is how it survived this long.
///
/// **Telling a legacy record apart from a slow one:** since V334 a record with
/// `generated_tokens > 0` always carries a positive
/// `generation_tokens_per_sec`. So zero-with-tokens means "written before the
/// split", not "infinitely slow", and an auditor must say so rather than
/// report a breach it cannot actually evidence.
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
    /// Legacy: generated tokens over **total** time. Kept under its original
    /// JSON key so old logs still parse. Prefer
    /// [`Self::generation_tokens_per_sec`].
    #[serde(rename = "tokens_per_sec")]
    pub tokens_per_sec_mixed: f64,
    /// Zero on records written before V334.
    #[serde(default)]
    pub prompt_ms: u64,
    /// Zero on records written before V334.
    #[serde(default)]
    pub generation_ms: u64,
    /// Zero on records written before V334.
    #[serde(default)]
    pub prompt_tokens_per_sec: f64,
    /// Zero on records written before V334. The rate an SLO on responsiveness
    /// should be checked against.
    #[serde(default)]
    pub generation_tokens_per_sec: f64,
    pub n_gpu_layers_requested: u32,
    pub n_gpu_layers_used: u32,
    pub peak_vram_mib: Option<u64>,
}

impl SloRecord {
    /// Whether this line predates the prompt/generation split, and therefore
    /// cannot answer "how fast did it write?" at all.
    pub fn predates_phase_split(&self) -> bool {
        self.generated_tokens > 0 && self.generation_tokens_per_sec == 0.0
    }
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
    fn the_thread_policy_prefers_physical_cores_over_filling_smt_siblings() {
        // The measured case: 4 physical, 8 logical. Answering 8 here would be
        // the "obvious" reading of available_parallelism and it was five times
        // slower on the machine this was measured on.
        assert_eq!(thread_policy(4, 8), 4);
        // No SMT: physical == logical, nothing to hold back.
        assert_eq!(thread_policy(16, 16), 16);
        // A probe that over-reports physical cannot exceed what exists.
        assert_eq!(thread_policy(32, 8), 8);
        // And it never returns zero, whatever the probe says.
        assert_eq!(thread_policy(0, 0), 1);
        assert_eq!(thread_policy(0, 4), 1);
    }

    #[test]
    fn threads_are_unset_until_asked_for() {
        // `None` is meaningful: it is what tells the backend to look at the
        // machine instead of taking llama.cpp's fixed 4.
        let cfg = LocalInferenceConfig::builder(BackendKind::Stub, "/tmp/m").build();
        assert_eq!(cfg.n_threads, None);
        let pinned = LocalInferenceConfig::builder(BackendKind::Stub, "/tmp/m")
            .n_threads(6)
            .build();
        assert_eq!(pinned.n_threads, Some(6));
    }

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
            tokens_per_sec_mixed: 70.0,
            prompt_ms: 4,
            generation_ms: 20,
            prompt_tokens_per_sec: 1250.0,
            generation_tokens_per_sec: 350.0,
            n_gpu_layers_requested: 32,
            n_gpu_layers_used: 16,
            peak_vram_mib: Some(2048),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"backend\":\"stub\""));
        assert!(json.contains("\"n_gpu_layers_used\":16"));
        // The Rust field was renamed; the wire key must not move, or every log
        // written before V334 stops parsing.
        assert!(json.contains("\"tokens_per_sec\":70.0"), "{json}");
        let back: SloRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generated_tokens, 7);
        assert_eq!(back.generation_tokens_per_sec, 350.0);
        assert!(!back.predates_phase_split());
    }

    #[test]
    fn a_record_written_before_the_split_still_parses_and_says_so() {
        // Copied verbatim from .ai_assistant/local_infer_logs, written in May
        // 2026. The point of keeping the old JSON key is that this line keeps
        // working; the point of `predates_phase_split` is that nobody mistakes
        // its missing generation rate for a measurement of zero.
        let old = r#"{"ts_unix_ms":1777934566481,"backend":"candle","model_path":"../TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q2_K.gguf","load_ms":1192,"first_chunk_ms":1858,"total_ms":6828,"prompt_tokens":6,"generated_tokens":16,"tokens_per_sec":2.3430714702990207,"n_gpu_layers_requested":0,"n_gpu_layers_used":0,"peak_vram_mib":null}"#;

        let r: SloRecord = serde_json::from_str(old).expect("old records must keep parsing");
        assert_eq!(r.generated_tokens, 16);
        assert_eq!(r.tokens_per_sec_mixed, 2.3430714702990207);
        // Absent in the file, so zero -- and that is exactly what must NOT be
        // read as "it generated at zero tokens per second".
        assert_eq!(r.generation_tokens_per_sec, 0.0);
        assert_eq!(r.prompt_ms, 0);
        assert!(
            r.predates_phase_split(),
            "a record with tokens but no generation rate predates the split"
        );
    }

    #[test]
    fn a_new_record_with_tokens_is_never_mistaken_for_a_legacy_one() {
        // The discriminator only works if the backends always populate the
        // rate when they generated anything. This is that guarantee, checked
        // on the one backend that runs everywhere.
        let mut b = StubBackend::new();
        let stats = b
            .generate("hola que tal", &GenParams::default(), &mut |_| {})
            .expect("stub generates");
        assert!(stats.generated_tokens > 0);
        assert!(
            stats.generation_tokens_per_sec > 0.0,
            "a backend that produced tokens must report a rate, or the legacy              discriminator turns fresh records into old ones: {stats:?}"
        );
        assert!(stats.prompt_tokens_per_sec > 0.0, "{stats:?}");
    }
}
