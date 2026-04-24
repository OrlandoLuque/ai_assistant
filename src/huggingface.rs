//! V103: HuggingFace Hub model metadata resolver.
//!
//! vLLM loads models by HuggingFace repo ID (e.g. `Qwen/Qwen2.5-7B-Instruct`)
//! instead of local GGUF files. Before we recommend a launch command we
//! should be able to tell the user a few things about that repo:
//!
//! - Is the repo gated (requires `huggingface-cli login`)?
//! - Is it a private repo (same — requires auth)?
//! - What's its approximate on-disk size? (sum of `siblings[].size`)
//! - What pipeline type is it? (`text-generation`, `feature-extraction`, …)
//!
//! This module only hits the public `https://huggingface.co/api/models/{repo}`
//! endpoint. No auth is required for metadata of public repos.

use crate::retry::{retry_with_config, RetryConfig};
use serde::{Deserialize, Serialize};

/// Metadata about a HuggingFace repository, as used by vLLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HfModelInfo {
    /// Repo ID, e.g. `Qwen/Qwen2.5-7B-Instruct`.
    pub id: String,
    /// Pipeline tag declared by the repo (`text-generation`, `feature-extraction`,
    /// `text-classification`, …). Useful to reject non-chat models for vLLM
    /// serving.
    pub pipeline_tag: Option<String>,
    /// Whether the repo is gated (requires explicit license acceptance +
    /// auth token to download). vLLM launch will fail without a token.
    pub gated: bool,
    /// Whether the repo is private. Functionally equivalent to `gated` from
    /// the user's perspective (needs auth).
    pub private: bool,
    /// Approximate total size of the repo in bytes. Sum of all `siblings`
    /// with a known `size`. Returns `None` when no sibling has size data.
    pub total_size_bytes: Option<u64>,
    /// Free-form tags the repo declares (`conversational`, `chat`, …).
    pub tags: Vec<String>,
    /// Approximate download count (best-effort; some repos report 0).
    pub downloads: Option<u64>,
    /// Like count (social signal, ignore for production decisions).
    pub likes: Option<u64>,
}

impl HfModelInfo {
    /// Is this repo suitable for vLLM serving?
    ///
    /// vLLM serves `text-generation` and `conversational` pipelines. Other
    /// pipelines (embeddings, classification, ASR) still work via vLLM in
    /// some cases but aren't what the chat CLI expects.
    pub fn is_text_generation(&self) -> bool {
        matches!(
            self.pipeline_tag.as_deref(),
            Some("text-generation") | Some("conversational")
        ) || self
            .tags
            .iter()
            .any(|t| t == "conversational" || t == "text-generation")
    }

    /// Can this repo be downloaded without authentication?
    pub fn is_public(&self) -> bool {
        !self.gated && !self.private
    }

    /// Approximate size in GiB, rounded to one decimal. `None` if size
    /// metadata is unavailable.
    pub fn approx_size_gib(&self) -> Option<f64> {
        self.total_size_bytes
            .map(|b| (b as f64) / 1024.0 / 1024.0 / 1024.0)
            .map(|g| (g * 10.0).round() / 10.0)
    }
}

/// Fetch metadata for a HuggingFace repo.
///
/// `repo_id` is e.g. `Qwen/Qwen2.5-7B-Instruct`. Uses `RetryConfig::fast`
/// (2 retries) against `https://huggingface.co/api/models/{repo}`.
///
/// Returns `Err(String)` on network failure or non-2xx status (including
/// 401/403 for gated repos when no token is set — this is intentional so
/// the caller can surface the auth requirement to the user).
pub fn huggingface_model_info(repo_id: &str) -> Result<HfModelInfo, String> {
    let url = format!("https://huggingface.co/api/models/{}", repo_id);
    let body: serde_json::Value = retry_with_config(RetryConfig::fast(), || {
        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .call()?;
        let v: serde_json::Value = resp.into_json()?;
        Ok(v)
    })
    .map_err(|e| format!("GET {}: {}", url, e))?;

    Ok(parse_hf_response(repo_id, &body))
}

/// Parse a `/api/models/{repo}` JSON body into an `HfModelInfo`.
///
/// Split out from `huggingface_model_info` so tests can exercise parsing
/// without network access.
pub fn parse_hf_response(repo_id: &str, body: &serde_json::Value) -> HfModelInfo {
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| repo_id.to_string());

    let pipeline_tag = body
        .get("pipeline_tag")
        .and_then(|v| v.as_str())
        .map(String::from);

    let gated = body
        .get("gated")
        .map(|v| match v {
            serde_json::Value::Bool(b) => *b,
            // HF sometimes returns "manual" or "auto" strings for gated.
            serde_json::Value::String(s) => !s.is_empty() && s != "false",
            _ => false,
        })
        .unwrap_or(false);

    let private = body
        .get("private")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let tags: Vec<String> = body
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let downloads = body.get("downloads").and_then(|v| v.as_u64());
    let likes = body.get("likes").and_then(|v| v.as_u64());

    let total_size_bytes = body
        .get("siblings")
        .and_then(|v| v.as_array())
        .map(|arr| {
            let sum: u64 = arr
                .iter()
                .filter_map(|s| s.get("size").and_then(|v| v.as_u64()))
                .sum();
            sum
        })
        .filter(|sum| *sum > 0);

    HfModelInfo {
        id,
        pipeline_tag,
        gated,
        private,
        total_size_bytes,
        tags,
        downloads,
        likes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_public_text_generation_repo() {
        let body = json!({
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "pipeline_tag": "text-generation",
            "gated": false,
            "private": false,
            "tags": ["conversational", "text-generation", "transformers"],
            "downloads": 1_500_000,
            "likes": 500
        });
        let info = parse_hf_response("Qwen/Qwen2.5-7B-Instruct", &body);
        assert_eq!(info.id, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(info.pipeline_tag.as_deref(), Some("text-generation"));
        assert!(!info.gated);
        assert!(!info.private);
        assert!(info.is_text_generation());
        assert!(info.is_public());
        assert_eq!(info.downloads, Some(1_500_000));
    }

    #[test]
    fn detects_gated_llama_repo() {
        let body = json!({
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "pipeline_tag": "text-generation",
            "gated": "manual",
            "private": false
        });
        let info = parse_hf_response("meta-llama/Llama-3.1-8B-Instruct", &body);
        assert!(info.gated);
        assert!(!info.is_public());
    }

    #[test]
    fn detects_private_repo() {
        let body = json!({
            "id": "my-org/internal",
            "gated": false,
            "private": true
        });
        let info = parse_hf_response("my-org/internal", &body);
        assert!(info.private);
        assert!(!info.is_public());
    }

    #[test]
    fn sums_sibling_sizes_for_total() {
        let body = json!({
            "id": "x/y",
            "siblings": [
                { "rfilename": "config.json", "size": 1024 },
                { "rfilename": "model.safetensors", "size": 4_000_000_000u64 },
                { "rfilename": "README.md" }
            ]
        });
        let info = parse_hf_response("x/y", &body);
        assert_eq!(info.total_size_bytes, Some(4_000_001_024));
        let gib = info.approx_size_gib().expect("size present");
        assert!((gib - 3.7).abs() < 0.1, "unexpected gib={}", gib);
    }

    #[test]
    fn returns_none_size_when_no_siblings_have_size() {
        let body = json!({
            "id": "x/y",
            "siblings": [ { "rfilename": "README.md" } ]
        });
        let info = parse_hf_response("x/y", &body);
        assert!(info.total_size_bytes.is_none());
        assert!(info.approx_size_gib().is_none());
    }

    #[test]
    fn falls_back_to_requested_id_when_response_missing_id() {
        let body = json!({ "pipeline_tag": "text-generation" });
        let info = parse_hf_response("requested/repo", &body);
        assert_eq!(info.id, "requested/repo");
    }

    #[test]
    fn is_text_generation_via_tags_when_pipeline_missing() {
        let body = json!({
            "id": "x/y",
            "tags": ["conversational"]
        });
        let info = parse_hf_response("x/y", &body);
        assert!(info.is_text_generation());
    }

    #[test]
    fn is_not_text_generation_for_embedding_models() {
        let body = json!({
            "id": "BAAI/bge-m3",
            "pipeline_tag": "feature-extraction",
            "tags": ["sentence-transformers"]
        });
        let info = parse_hf_response("BAAI/bge-m3", &body);
        assert!(!info.is_text_generation());
    }

    #[test]
    fn gated_as_string_manual_is_gated() {
        let body = json!({ "id": "x/y", "gated": "manual" });
        assert!(parse_hf_response("x/y", &body).gated);
    }

    #[test]
    fn gated_as_string_false_is_not_gated() {
        let body = json!({ "id": "x/y", "gated": "false" });
        assert!(!parse_hf_response("x/y", &body).gated);
    }
}
