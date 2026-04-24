//! V103: vLLM OpenAI-compatible server capability probe.
//!
//! vLLM (<https://github.com/vllm-project/vllm>) exposes a set of OpenAI-compatible
//! endpoints plus a small number of vLLM-specific ones that let us detect the
//! engine version, the models currently loaded, and whether LoRA hot-swap is
//! enabled:
//!
//! - `/v1/models` — OpenAI-compatible list of served models.
//! - `/version` — vLLM engine version string (vLLM-specific).
//! - `/health` — simple healthcheck (HTTP 200 when ready).
//! - `/v1/load_lora_adapter` — only present when vLLM was launched with
//!   `--enable-lora`. We don't call it; we just detect whether probing it
//!   returns something other than 404.
//!
//! The probe is intentionally *passive*: it never loads adapters, never
//! triggers inference, never changes server state.

use crate::retry::{retry_with_config, RetryConfig};
use serde::{Deserialize, Serialize};

/// A single model currently served by vLLM (as reported by `/v1/models`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLlmServedModel {
    /// Model identifier (typically a HuggingFace repo ID, e.g.
    /// `Qwen/Qwen2.5-7B-Instruct`).
    pub id: String,
    /// Who owns/served this model (`vllm` for base models, user-chosen for
    /// LoRA adapters).
    pub owned_by: Option<String>,
    /// Maximum context length the engine was launched with (`--max-model-len`).
    /// vLLM reports this under the non-standard `max_model_len` field when
    /// available.
    pub max_model_len: Option<u32>,
    /// Root model that a LoRA adapter is attached to, when this entry is a
    /// LoRA adapter rather than a base model.
    pub parent: Option<String>,
}

/// What a vLLM probe tells us about the running server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLlmCapability {
    /// vLLM engine version string reported by `/version`, if reachable.
    pub engine_version: Option<String>,
    /// Models currently loaded and served by the engine.
    pub served_models: Vec<VLlmServedModel>,
    /// Whether the `/health` endpoint returned success.
    pub healthy: bool,
    /// Whether vLLM was launched with LoRA support (`--enable-lora`).
    ///
    /// Detected by whether the `/v1/load_lora_adapter` endpoint responds
    /// with anything other than 404 to a HEAD/OPTIONS probe — we do not
    /// actually issue a load.
    pub supports_lora: bool,
}

impl VLlmCapability {
    /// Is the given model ID currently loaded in the engine?
    pub fn has_model(&self, id: &str) -> bool {
        self.served_models.iter().any(|m| m.id == id)
    }

    /// Return the maximum context length across all served base models.
    /// Useful for UI to cap the slider without querying per-model.
    pub fn max_context_length(&self) -> Option<u32> {
        self.served_models
            .iter()
            .filter_map(|m| m.max_model_len)
            .max()
    }
}

/// Probe a running vLLM server at `base_url` (e.g. `http://localhost:8000`).
///
/// Tries `/version`, `/v1/models`, `/health`, and `/v1/load_lora_adapter` in
/// sequence. Each probe is independent: a failed `/version` does not block
/// reading the model list, and a healthy server without LoRA support still
/// yields a valid capability. Returns `Err(String)` only if `/v1/models`
/// itself cannot be reached (the authoritative "is this a vLLM?" signal).
pub fn probe_vllm(base_url: &str) -> Result<VLlmCapability, String> {
    let base = base_url.trim_end_matches('/');

    // `/v1/models` is the authoritative check — if this fails, the server is
    // either down or not OpenAI-compatible.
    let models_url = format!("{}/v1/models", base);
    let models_body: serde_json::Value = retry_with_config(RetryConfig::fast(), || {
        let resp = ureq::get(&models_url)
            .timeout(std::time::Duration::from_secs(5))
            .call()?;
        let v: serde_json::Value = resp.into_json()?;
        Ok(v)
    })
    .map_err(|e| format!("GET {}: {}", models_url, e))?;
    let served_models = parse_models_response(&models_body);

    // `/version` is vLLM-specific; absence just means we don't know the
    // version. Don't retry hard — a single try is enough.
    let engine_version = {
        let url = format!("{}/version", base);
        ureq::get(&url)
            .timeout(std::time::Duration::from_secs(3))
            .call()
            .ok()
            .and_then(|resp| resp.into_json::<serde_json::Value>().ok())
            .and_then(|v| parse_version_response(&v))
    };

    // `/health` is OpenAI-compatible-ish — 200 means ready.
    let healthy = {
        let url = format!("{}/health", base);
        ureq::get(&url)
            .timeout(std::time::Duration::from_secs(3))
            .call()
            .is_ok()
    };

    // LoRA detection: HEAD `/v1/load_lora_adapter`. A 404 means LoRA wasn't
    // enabled at launch; anything else (405, 400, 200) means the route
    // exists. We use OPTIONS because HEAD isn't guaranteed to work on POST
    // endpoints.
    let supports_lora = {
        let url = format!("{}/v1/load_lora_adapter", base);
        match ureq::request("OPTIONS", &url)
            .timeout(std::time::Duration::from_secs(3))
            .call()
        {
            Ok(_) => true,
            Err(ureq::Error::Status(404, _)) => false,
            Err(ureq::Error::Status(_, _)) => true, // route exists, method not allowed etc.
            Err(_) => false,                        // network error — assume absent
        }
    };

    Ok(VLlmCapability {
        engine_version,
        served_models,
        healthy,
        supports_lora,
    })
}

/// Parse a `/v1/models` response body into a list of served models.
///
/// OpenAI's contract is `{ "object": "list", "data": [{ "id": "...", ... }] }`.
/// vLLM augments each entry with `max_model_len` and, for LoRA adapters, a
/// `parent` field pointing at the base model.
pub fn parse_models_response(body: &serde_json::Value) -> Vec<VLlmServedModel> {
    body.get("data")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id").and_then(|v| v.as_str()).map(String::from)?;
                    Some(VLlmServedModel {
                        id,
                        owned_by: m.get("owned_by").and_then(|v| v.as_str()).map(String::from),
                        max_model_len: m
                            .get("max_model_len")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                        parent: m.get("parent").and_then(|v| v.as_str()).map(String::from),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Block until a vLLM server at `base_url` is ready or `timeout` elapses.
///
/// Polls `/health` every `interval` with a short per-request timeout. Returns
/// the capability probe on first success, or `Err` with an elapsed-timeout
/// message. Never hits the network after success.
///
/// Typical use after `vllm serve …`: give vLLM 30–120 s to load weights into
/// VRAM before the CLI returns control. No-op when vLLM is already up.
pub fn vllm_wait_until_ready(
    base_url: &str,
    timeout: std::time::Duration,
    interval: std::time::Duration,
) -> Result<VLlmCapability, String> {
    let base = base_url.trim_end_matches('/');
    let health_url = format!("{}/health", base);
    let deadline = std::time::Instant::now() + timeout;

    loop {
        let healthy = ureq::get(&health_url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .is_ok();
        if healthy {
            return probe_vllm(base);
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "vLLM at {} not ready after {:.1}s",
                base_url,
                timeout.as_secs_f32()
            ));
        }
        std::thread::sleep(interval);
    }
}

/// Parse a `/version` response into a version string.
///
/// vLLM returns `{ "version": "0.6.3" }`. Some older builds return the raw
/// string or nothing at all. Handles both.
pub fn parse_version_response(body: &serde_json::Value) -> Option<String> {
    if let Some(s) = body.as_str() {
        return Some(s.to_string());
    }
    body.get("version")
        .and_then(|v| v.as_str())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_minimal_models_list() {
        let body = json!({
            "object": "list",
            "data": [
                { "id": "Qwen/Qwen2.5-7B-Instruct", "owned_by": "vllm" }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(models[0].owned_by.as_deref(), Some("vllm"));
        assert!(models[0].max_model_len.is_none());
        assert!(models[0].parent.is_none());
    }

    #[test]
    fn parses_vllm_specific_max_model_len() {
        let body = json!({
            "data": [
                {
                    "id": "meta-llama/Llama-3.1-8B-Instruct",
                    "owned_by": "vllm",
                    "max_model_len": 8192
                }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models[0].max_model_len, Some(8192));
    }

    #[test]
    fn parses_lora_adapter_with_parent() {
        let body = json!({
            "data": [
                { "id": "base-model", "owned_by": "vllm", "max_model_len": 4096 },
                {
                    "id": "my-lora",
                    "owned_by": "user",
                    "parent": "base-model"
                }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[1].parent.as_deref(), Some("base-model"));
    }

    #[test]
    fn parses_empty_models_list() {
        let body = json!({ "object": "list", "data": [] });
        assert!(parse_models_response(&body).is_empty());
    }

    #[test]
    fn parses_missing_data_field_as_empty() {
        let body = json!({ "object": "list" });
        assert!(parse_models_response(&body).is_empty());
    }

    #[test]
    fn skips_entries_without_id() {
        let body = json!({ "data": [ { "owned_by": "vllm" }, { "id": "ok" } ] });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "ok");
    }

    #[test]
    fn parses_version_object() {
        let body = json!({ "version": "0.6.3" });
        assert_eq!(parse_version_response(&body).as_deref(), Some("0.6.3"));
    }

    #[test]
    fn parses_version_as_bare_string() {
        let body = json!("0.5.0");
        assert_eq!(parse_version_response(&body).as_deref(), Some("0.5.0"));
    }

    #[test]
    fn returns_none_when_no_version() {
        let body = json!({ "other": "field" });
        assert!(parse_version_response(&body).is_none());
    }

    #[test]
    fn capability_has_model_lookup() {
        let cap = VLlmCapability {
            engine_version: Some("0.6.3".into()),
            served_models: vec![VLlmServedModel {
                id: "Qwen/Qwen2.5-7B-Instruct".into(),
                owned_by: Some("vllm".into()),
                max_model_len: Some(32768),
                parent: None,
            }],
            healthy: true,
            supports_lora: false,
        };
        assert!(cap.has_model("Qwen/Qwen2.5-7B-Instruct"));
        assert!(!cap.has_model("not-loaded"));
    }

    #[test]
    fn capability_max_context_length_picks_largest() {
        let cap = VLlmCapability {
            engine_version: None,
            served_models: vec![
                VLlmServedModel {
                    id: "a".into(),
                    owned_by: None,
                    max_model_len: Some(4096),
                    parent: None,
                },
                VLlmServedModel {
                    id: "b".into(),
                    owned_by: None,
                    max_model_len: Some(32768),
                    parent: None,
                },
                VLlmServedModel {
                    id: "c".into(),
                    owned_by: None,
                    max_model_len: None,
                    parent: None,
                },
            ],
            healthy: true,
            supports_lora: true,
        };
        assert_eq!(cap.max_context_length(), Some(32768));
    }

    #[test]
    fn wait_until_ready_times_out_fast_when_unreachable() {
        let start = std::time::Instant::now();
        let res = vllm_wait_until_ready(
            "http://127.0.0.1:1", // unused port — connection will fail fast
            std::time::Duration::from_millis(100),
            std::time::Duration::from_millis(50),
        );
        assert!(res.is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn capability_max_context_none_when_no_data() {
        let cap = VLlmCapability {
            engine_version: None,
            served_models: vec![],
            healthy: false,
            supports_lora: false,
        };
        assert_eq!(cap.max_context_length(), None);
    }
}
