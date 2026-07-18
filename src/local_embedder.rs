//! In-process semantic embedder (candle BERT, e.g. `all-MiniLM-L6-v2`).
//!
//! Produces sentence embeddings for semantic knowledge retrieval **without** an
//! Ollama / embedding server — bundleable and mobile-friendly (~22M params,
//! CPU, milliseconds). Model files (`config.json`, `tokenizer.json`,
//! `model.safetensors`) are loaded from a local directory; [`ensure_model`]
//! downloads them once from HuggingFace.
//!
//! Gated behind the `embeddings-local` feature.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};

/// Default HuggingFace repo for the bundled small embedder.
pub const DEFAULT_MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// A loaded sentence-transformer (BERT) embedder running in-process on CPU.
pub struct LocalEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl LocalEmbedder {
    /// Load from a directory containing `config.json`, `tokenizer.json` and
    /// `model.safetensors`.
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let config: Config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).context("read config.json")?,
        )
        .context("parse BERT config.json")?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer.json: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        let weights = dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)
                .context("load model.safetensors")?
        };
        let model = BertModel::load(vb, &config).context("build BERT model")?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Download the model files if missing, then load. `dir` is the cache
    /// directory (e.g. `<data>/embedders/all-MiniLM-L6-v2`).
    pub fn load_or_download(dir: &Path, repo: &str) -> Result<Self> {
        ensure_model(dir, repo)?;
        Self::load(dir)
    }

    /// Embed a batch of texts → one L2-normalized, mean-pooled vector each.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let n = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut ids = Vec::with_capacity(n * seq);
        let mut mask = Vec::with_capacity(n * seq);
        for e in &encodings {
            ids.extend(e.get_ids().iter().copied());
            mask.extend(e.get_attention_mask().iter().copied());
        }
        let input_ids = Tensor::from_vec(ids, (n, seq), &self.device)?;
        let attn = Tensor::from_vec(mask, (n, seq), &self.device)?;
        let token_type_ids = input_ids.zeros_like()?;

        // (n, seq, hidden)
        let out = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attn))?;

        // Mean pooling over tokens, weighted by the attention mask.
        let mask_f = attn.to_dtype(DType::F32)?.unsqueeze(2)?; // (n, seq, 1)
        let summed = out.broadcast_mul(&mask_f)?.sum(1)?; // (n, hidden)
        let counts = mask_f.sum(1)?; // (n, 1)
        let mean = summed.broadcast_div(&counts)?; // (n, hidden)
                                                   // L2 normalize so cosine == dot product.
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean.broadcast_div(&norm)?;
        Ok(normalized.to_vec2::<f32>()?)
    }
}

/// Default on-disk cache directory for the embedder.
pub fn default_model_dir() -> PathBuf {
    let base = dirs_data_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("ai_assistant")
        .join("embedders")
        .join("all-MiniLM-L6-v2")
}

fn dirs_data_dir() -> Option<PathBuf> {
    // Avoid a `dirs` dependency: use platform env vars.
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    }
}

/// Ensure the model files exist in `dir`, downloading any that are missing from
/// `https://huggingface.co/<repo>/resolve/main/<file>`.
pub fn ensure_model(dir: &Path, repo: &str) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    for f in FILES {
        let target = dir.join(f);
        if target.exists() && std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0) > 0 {
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{f}");
        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(600))
            .call()
            .with_context(|| format!("download {url}"))?;
        let mut reader = resp.into_reader();
        let mut out = std::fs::File::create(&target)
            .with_context(|| format!("create {}", target.display()))?;
        std::io::copy(&mut reader, &mut out).with_context(|| format!("write {f}"))?;
    }
    Ok(())
}
