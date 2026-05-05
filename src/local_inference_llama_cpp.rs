//! llama-cpp-2 backend for in-process GGUF inference (V112).
//!
//! Activated by the `local-inference-llama-cpp` sub-feature. Wraps the safe
//! Rust bindings around llama.cpp (`llama-cpp-2` crate). Input is always a
//! `.gguf` file — the GGUF metadata-embedded tokenizer is used directly, so
//! unlike the Candle GGUF backend (V111) no sibling `tokenizer.json` is
//! required.
//!
//! ## Why a second GGUF backend?
//!
//! Candle GGUF (V111) is fine for **single-stream, CPU-only** generation.
//! llama-cpp-2 unlocks two capabilities that are deferred there:
//!
//! 1. **Continuous batching** — N concurrent sequences sharing one model
//!    load on a single GPU. Critical for multi-agent throughput.
//! 2. **Tensor-split** across multiple GPUs.
//!
//! This first iteration is **CPU baseline** only. GPU offload (`cuda` /
//! `metal` upstream features) is wired through `n_gpu_layers` but takes
//! effect only when the upstream crate is built with the matching feature.
//! The VRAM auto-clamp policy from V108 (`vram::clamp_gpu_layers`) is
//! applied end-to-end so a request like `--n-gpu-layers 999` automatically
//! reduces to what fits, never OOMing.
//!
//! ## Lifetime model
//!
//! `LlamaModel::new_context` borrows `&self` so the returned `LlamaContext`
//! is tied to the model. We can't store both in the same struct without
//! self-referential gymnastics, so we keep only the model + the
//! resolved context params and create a fresh context per `generate()`
//! call. The KV cache is per-context anyway, so this matches the typical
//! pattern (one generation = one context).

#![cfg(feature = "local-inference-llama-cpp")]

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::local_inference::{
    vram, Backend, BackendError, BackendKind, GenParams, GenStats, LocalInferenceConfig,
};

const GGUF_EXT: &str = "gguf";
/// GGUF metadata key holding the transformer layer count. All llama-family
/// GGUFs use this key (`llama.block_count`); other architectures use their
/// own (e.g. `qwen2.block_count`). Fallback when missing: 32 (Llama-3 8B
/// shape), conservative for the clamp policy.
const META_BLOCK_COUNT_LLAMA: &str = "llama.block_count";
const FALLBACK_BLOCK_COUNT: u32 = 32;

/// Process-wide singleton. `LlamaBackend::init()` errors with
/// `BackendAlreadyInitialized` on the second call, so we wrap it in a
/// `OnceLock`. The backend has internal `'static` lifetime tied to the
/// underlying llama.cpp init — leaking a `&'static` reference is the
/// idiomatic way to hand it to model loaders.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend, BackendError> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaBackend::init()
        .map_err(|e| BackendError::Backend(format!("init llama backend: {e}")))?;
    Ok(BACKEND.get_or_init(|| b))
}

pub(crate) fn load_llama_cpp(cfg: &LocalInferenceConfig) -> Result<Box<dyn Backend>, BackendError> {
    let path: &Path = &cfg.model_path;
    let is_gguf = path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(GGUF_EXT))
            .unwrap_or(false);
    if !is_gguf {
        return Err(BackendError::Backend(format!(
            "llama-cpp-2 backend requires a .gguf file; got: {}",
            path.display()
        )));
    }

    let backend = backend()?;

    // Peek the GGUF header for the transformer layer count. Used by the
    // VRAM clamp policy when the user asked for GPU offload.
    let total_layers = read_block_count(path).unwrap_or(FALLBACK_BLOCK_COUNT);

    // Resolve the effective n_gpu_layers via the V108 clamp policy. If
    // VRAM detection is unavailable, just clamp to total_layers so a
    // bogus "999" request doesn't reach the C side.
    let requested = cfg.n_gpu_layers;
    let n_gpu_layers_used = if requested == 0 {
        0
    } else if cfg.allow_gpu_clamp {
        match vram::detect_available_mib() {
            Some(free) => {
                let model_mib = cfg
                    .model_size_mib
                    .unwrap_or_else(|| file_size_mib(path).unwrap_or(0));
                vram::clamp_gpu_layers(model_mib, requested, total_layers, free)
            }
            None => requested.min(total_layers),
        }
    } else {
        requested.min(total_layers)
    };

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers_used);
    let model = LlamaModel::load_from_file(backend, path, &model_params)
        .map_err(|e| BackendError::Backend(format!("load gguf via llama-cpp-2: {e}")))?;

    let ctx_size = cfg.ctx_size.max(1);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(ctx_size));

    Ok(Box::new(LlamaCppBackend {
        backend,
        model,
        ctx_params,
        n_gpu_layers_requested: requested,
        n_gpu_layers_used,
    }))
}

fn read_block_count(path: &Path) -> Option<u32> {
    let g = GgufContext::from_file(path)?;
    let idx = g.find_key(META_BLOCK_COUNT_LLAMA);
    if idx < 0 {
        return None;
    }
    Some(g.val_u32(idx))
}

fn file_size_mib(path: &Path) -> Option<u64> {
    let m = std::fs::metadata(path).ok()?;
    Some(m.len() / (1024 * 1024))
}

struct LlamaCppBackend {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    ctx_params: LlamaContextParams,
    n_gpu_layers_requested: u32,
    n_gpu_layers_used: u32,
}

impl Backend for LlamaCppBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::LlamaCpp
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        let start = Instant::now();

        // Fresh per-generation context — KV cache is per-context. Cloning
        // the params keeps the public API of `generate` non-consuming on
        // the backend (subsequent calls reuse the resolved ctx_size).
        let mut ctx = self
            .model
            .new_context(self.backend, self.ctx_params.clone())
            .map_err(|e| BackendError::Backend(format!("new context: {e}")))?;

        let prompt_tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| BackendError::Backend(format!("tokenize prompt: {e}")))?;
        if prompt_tokens.is_empty() {
            return Err(BackendError::Backend(
                "tokenized prompt is empty".to_string(),
            ));
        }
        let n_prompt = prompt_tokens.len() as u32;

        // Single-sequence generation. Allocate a batch big enough for the
        // prompt (we send it in one shot) and reuse it for one-token
        // increments afterwards. n_seq_max=1 — multi-agent batching is the
        // job of a future iteration.
        let batch_capacity = prompt_tokens.len().max(1);
        let mut batch = LlamaBatch::new(batch_capacity, 1);

        // Fill the batch with the prompt. Only the LAST token gets its
        // logits computed — all we need to start sampling is the next
        // distribution.
        let last_idx = prompt_tokens.len() - 1;
        for (i, tok) in prompt_tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == last_idx)
                .map_err(|e| BackendError::Backend(format!("batch.add prompt: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| BackendError::Backend(format!("decode prompt: {e}")))?;

        // Sampler chain: temp ≤ 0 → greedy. Otherwise temp + top_p + dist.
        // `dist` is the actual stochastic sampler at the tail; without it
        // the chain is purely deterministic re-shaping of the distribution.
        let mut sampler = if params.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            let mut chain: Vec<LlamaSampler> = Vec::with_capacity(3);
            chain.push(LlamaSampler::temp(params.temperature));
            if params.top_p > 0.0 && params.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(params.top_p, 1));
            }
            chain.push(LlamaSampler::dist(42));
            LlamaSampler::chain_simple(chain)
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let eos = self.model.token_eos();

        let mut next_pos = prompt_tokens.len() as i32;
        let mut generated: u32 = 0;
        let mut tail_buffer = String::new();

        for _ in 0..params.max_tokens {
            // Sample from the most recent logits. After the prompt decode
            // that's the last token of the prompt; after each subsequent
            // single-token decode that's index 0 of the new batch.
            let next = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(next);

            if self.model.is_eog_token(next) || next == eos {
                break;
            }

            let piece = self
                .model
                .token_to_piece(next, &mut decoder, false, None)
                .map_err(|e| BackendError::Backend(format!("decode token: {e}")))?;
            if !piece.is_empty() {
                on_chunk(&piece);
                tail_buffer.push_str(&piece);
                // Trim to the longest stop-string prefix-keeping size
                // so we don't grow the buffer unboundedly when no stops
                // are configured. 64 chars covers all realistic stops.
                if tail_buffer.len() > 64 {
                    let cut = tail_buffer.len() - 64;
                    let cut = floor_char_boundary(&tail_buffer, cut);
                    tail_buffer.drain(..cut);
                }
            }
            generated += 1;

            if !params.stop.is_empty() && params.stop.iter().any(|s| tail_buffer.ends_with(s)) {
                break;
            }

            // Feed the sampled token back as a 1-token batch and decode
            // again. This is the standard incremental loop — the KV cache
            // grows by 1 each step, so we don't reprocess the prompt.
            batch.clear();
            batch
                .add(next, next_pos, &[0], true)
                .map_err(|e| BackendError::Backend(format!("batch.add token: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::Backend(format!("decode token: {e}")))?;
            next_pos += 1;
        }

        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64().max(1e-9);
        Ok(GenStats {
            prompt_tokens: n_prompt,
            generated_tokens: generated,
            time_ms: elapsed.as_millis() as u64,
            tokens_per_sec: generated as f64 / secs,
            peak_vram_mib: None,
        })
    }
}

impl LlamaCppBackend {
    /// VRAM clamp diagnostics, useful for SLO records and tests. Not part
    /// of the trait yet; each backend exposes its own status surface.
    #[allow(dead_code)]
    pub(crate) fn n_gpu_layers_requested(&self) -> u32 {
        self.n_gpu_layers_requested
    }
    #[allow(dead_code)]
    pub(crate) fn n_gpu_layers_used(&self) -> u32 {
        self.n_gpu_layers_used
    }
}

/// `String::floor_char_boundary` is unstable; this is the same operation
/// scoped to the buffer we own.
fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}
