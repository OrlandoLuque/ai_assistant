//! Candle CPU backend for in-process Llama inference (V110 / Phase A.3 iter 3).
//!
//! Activated by the `local-inference-candle` sub-feature. Loads a HuggingFace
//! Llama-format checkpoint from a directory containing:
//!   - `config.json`     — Llama config (hidden_size, n_layers, GQA params, …)
//!   - `tokenizer.json`  — HuggingFace tokenizer
//!   - `model.safetensors` — model weights (single file; sharded loaders TBD)
//!
//! and runs the forward pass on CPU via
//! [`candle_transformers::models::llama::Llama`]. CPU only — CUDA / Metal /
//! Accelerate are intentionally deferred until the CPU baseline is verified
//! end-to-end against TinyLlama 1.1B.

#![cfg(feature = "local-inference-candle")]

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use tokenizers::Tokenizer;

use crate::local_inference::{
    Backend, BackendError, BackendKind, GenParams, GenStats, LocalInferenceConfig,
};

const SAFETENSORS_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Load a CandleBackend from a HuggingFace-format Llama directory.
pub(crate) fn load_candle(cfg: &LocalInferenceConfig) -> Result<Box<dyn Backend>, BackendError> {
    let dir: &Path = &cfg.model_path;
    if !dir.is_dir() {
        return Err(BackendError::Backend(format!(
            "candle backend requires a directory containing {CONFIG_FILE}, \
             {TOKENIZER_FILE}, {SAFETENSORS_FILE}; got: {}",
            dir.display()
        )));
    }
    let config_path = dir.join(CONFIG_FILE);
    let tokenizer_path = dir.join(TOKENIZER_FILE);
    let safetensors_path = dir.join(SAFETENSORS_FILE);
    for required in [&config_path, &tokenizer_path, &safetensors_path] {
        if !required.exists() {
            return Err(BackendError::ModelNotFound(required.clone()));
        }
    }

    let config_json = std::fs::read_to_string(&config_path)?;
    let llama_cfg_raw: LlamaConfig = serde_json::from_str(&config_json)
        .map_err(|e| BackendError::Backend(format!("parse {CONFIG_FILE}: {e}")))?;
    let llama_cfg = llama_cfg_raw.into_config(false);

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| BackendError::Backend(format!("load tokenizer: {e}")))?;

    let device = Device::Cpu;
    // CPU + Llama: f32 keeps the math stable. bf16 would halve memory but
    // most CPU kernels in candle 0.10 are f32-only — cast at load time.
    let dtype = DType::F32;

    // SAFETY: mmaping a file the user pointed at. The file must outlive the
    // VarBuilder, which it does because we keep the model alive in self.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &device)
            .map_err(|e| BackendError::Backend(format!("load safetensors: {e}")))?
    };
    let cache = Cache::new(true, dtype, &llama_cfg, &device)
        .map_err(|e| BackendError::Backend(format!("build kv cache: {e}")))?;
    let model = Llama::load(vb, &llama_cfg)
        .map_err(|e| BackendError::Backend(format!("load llama: {e}")))?;

    let eos_id = match &llama_cfg.eos_token_id {
        Some(LlamaEosToks::Single(id)) => Some(*id),
        Some(LlamaEosToks::Multiple(ids)) => ids.first().copied(),
        None => None,
    };

    Ok(Box::new(CandleBackend {
        model,
        tokenizer,
        cache,
        device,
        eos_id,
    }))
}

struct CandleBackend {
    model: Llama,
    tokenizer: Tokenizer,
    cache: Cache,
    device: Device,
    eos_id: Option<u32>,
}

impl Backend for CandleBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Candle
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        let start = Instant::now();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| BackendError::Backend(format!("encode prompt: {e}")))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_tokens = tokens.len() as u32;

        // Sampling: temperature 0 (or near-zero) → greedy; otherwise top-p.
        // candle's LogitsProcessor expects f64 and treats `Some(0.0)` as greedy.
        let temperature = params.temperature as f64;
        let top_p = if params.top_p > 0.0 && params.top_p < 1.0 {
            Some(params.top_p as f64)
        } else {
            None
        };
        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), top_p);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
        let mut prev_decoded = String::new();
        let mut index_pos = 0usize;

        for i in 0..params.max_tokens {
            // First step: feed the whole prompt. After that: just the new
            // token; the KV cache holds the rest.
            let ctx_slice: &[u32] = if i == 0 {
                &tokens
            } else {
                &tokens[tokens.len() - 1..]
            };
            let input = Tensor::new(ctx_slice, &self.device)
                .map_err(|e| BackendError::Backend(format!("build input: {e}")))?
                .unsqueeze(0)
                .map_err(|e| BackendError::Backend(format!("unsqueeze: {e}")))?;
            let logits = self
                .model
                .forward(&input, index_pos, &mut self.cache)
                .map_err(|e| BackendError::Backend(format!("forward: {e}")))?;
            index_pos += ctx_slice.len();
            let logits = logits
                .squeeze(0)
                .map_err(|e| BackendError::Backend(format!("squeeze: {e}")))?;
            let next = logits_processor
                .sample(&logits)
                .map_err(|e| BackendError::Backend(format!("sample: {e}")))?;
            tokens.push(next);
            generated_tokens.push(next);

            if Some(next) == self.eos_id {
                break;
            }

            // Incremental decode: render the full generated sequence, emit
            // the suffix vs. last frame. Llama BPE leaves partial UTF-8 if
            // we emit per-token, so decoding the cumulative buffer and
            // diffing avoids broken multi-byte glyphs.
            let decoded = self
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| BackendError::Backend(format!("decode: {e}")))?;
            if decoded.len() > prev_decoded.len() {
                let suffix = &decoded[prev_decoded.len()..];
                on_chunk(suffix);
                prev_decoded = decoded;
            }

            if !params.stop.is_empty() && params.stop.iter().any(|s| prev_decoded.ends_with(s)) {
                break;
            }
        }

        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64().max(1e-9);
        let n_gen = generated_tokens.len() as u32;
        Ok(GenStats {
            prompt_tokens,
            generated_tokens: n_gen,
            time_ms: elapsed.as_millis() as u64,
            tokens_per_sec: n_gen as f64 / secs,
            peak_vram_mib: None,
        })
    }
}
