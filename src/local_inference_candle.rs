//! Candle CPU backend for in-process Llama inference.
//!
//! Activated by the `local-inference-candle` sub-feature. Two input formats:
//!
//! 1. **Safetensors directory** (V110) — HuggingFace-style Llama checkpoint:
//!      - `config.json`     — Llama config (hidden_size, n_layers, GQA params, …)
//!      - `tokenizer.json`  — HuggingFace tokenizer
//!      - `model.safetensors` — model weights (single file)
//!    Loaded via [`candle_transformers::models::llama::Llama`] in F32.
//!
//! 2. **GGUF file** (V111) — single-file quantized format used by llama.cpp,
//!    Ollama, LM Studio, etc. Path must end in `.gguf`. Tokenizer is read
//!    from a sibling `tokenizer.json` (the original HF tokenizer); GGUF
//!    metadata-embedded tokenizers are not yet decoded by candle 0.10.
//!    Loaded via [`candle_transformers::models::quantized_llama::ModelWeights`]
//!    which keeps weights in their original quantization (Q4_K_M, Q5_K_M, …).
//!
//! CPU only — CUDA / Metal / Accelerate are intentionally deferred until the
//! CPU baseline is verified end-to-end.

#![cfg(feature = "local-inference-candle")]

use std::path::Path;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use tokenizers::Tokenizer;

use crate::local_inference::{
    Backend, BackendError, BackendKind, GenParams, GenStats, LocalInferenceConfig,
};

const SAFETENSORS_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const GGUF_EXT: &str = "gguf";

/// Load a CandleBackend, dispatching by `cfg.model_path`:
/// - file ending in `.gguf` → quantized loader (V111).
/// - directory             → safetensors loader (V110).
pub(crate) fn load_candle(cfg: &LocalInferenceConfig) -> Result<Box<dyn Backend>, BackendError> {
    let path: &Path = &cfg.model_path;
    let is_gguf = path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(GGUF_EXT))
            .unwrap_or(false);

    if is_gguf {
        load_gguf(path)
    } else if path.is_dir() {
        load_safetensors_dir(path)
    } else {
        Err(BackendError::Backend(format!(
            "candle backend requires either a .gguf file or a directory \
             containing {CONFIG_FILE}, {TOKENIZER_FILE}, {SAFETENSORS_FILE}; got: {}",
            path.display()
        )))
    }
}

fn load_safetensors_dir(dir: &Path) -> Result<Box<dyn Backend>, BackendError> {
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
        inner: LoadedModel::Safetensors { model, cache },
        tokenizer,
        device,
        eos_id,
    }))
}

fn load_gguf(gguf_path: &Path) -> Result<Box<dyn Backend>, BackendError> {
    // Tokenizer: candle 0.10's quantized_llama doesn't decode the GGUF
    // metadata tokenizer. Require a sibling tokenizer.json (the standard
    // HF tokenizer the GGUF was built from — Ollama / LM Studio ship it
    // alongside their downloads, or it can be copied from the source repo).
    let tokenizer_path = gguf_path
        .parent()
        .map(|p| p.join(TOKENIZER_FILE))
        .ok_or_else(|| {
            BackendError::Backend(format!(
                "cannot resolve parent dir of {} to find tokenizer",
                gguf_path.display()
            ))
        })?;
    if !tokenizer_path.exists() {
        return Err(BackendError::ModelNotFound(tokenizer_path));
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| BackendError::Backend(format!("load tokenizer: {e}")))?;

    let device = Device::Cpu;
    let mut file = std::fs::File::open(gguf_path)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| BackendError::Backend(format!("read GGUF metadata: {e}")))?;

    // EOS id is in GGUF metadata as `tokenizer.ggml.eos_token_id` (u32).
    // Best-effort: missing → no early stop on EOS, generation just runs to
    // max_tokens or hits a stop string.
    let eos_id = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok());

    let model = QuantizedLlama::from_gguf(content, &mut file, &device)
        .map_err(|e| BackendError::Backend(format!("load quantized llama: {e}")))?;

    Ok(Box::new(CandleBackend {
        inner: LoadedModel::Gguf { model },
        tokenizer,
        device,
        eos_id,
    }))
}

enum LoadedModel {
    Safetensors { model: Llama, cache: Cache },
    Gguf { model: QuantizedLlama },
}

impl LoadedModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Safetensors { model, cache } => model.forward(input, index_pos, cache),
            Self::Gguf { model } => model.forward(input, index_pos),
        }
    }
}

struct CandleBackend {
    inner: LoadedModel,
    tokenizer: Tokenizer,
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
                .inner
                .forward(&input, index_pos)
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
