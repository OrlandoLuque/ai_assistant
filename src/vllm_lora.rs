//! V103.1: vLLM LoRA hot-swap client.
//!
//! vLLM launched with `--enable-lora` exposes two endpoints to mount and
//! unmount LoRA adapters without restarting the engine:
//!
//! - `POST /v1/load_lora_adapter` — body `{"lora_name": "...", "lora_path": "..."}`
//! - `POST /v1/unload_lora_adapter` — body `{"lora_name": "..."}`
//!
//! After a successful load, the adapter appears in `/v1/models` with the
//! supplied `lora_name` as its `id` and the base model as its `parent`. Use
//! that `lora_name` as the `model` field in chat/completions requests to
//! route inference through the adapter.
//!
//! This module is a thin JSON-over-HTTP client — no caching, no retries
//! beyond the single call. Detect whether LoRA is enabled via
//! [`crate::vllm_capability::VLlmCapability::supports_lora`] first.

use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/load_lora_adapter`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadLoraRequest {
    /// Name to register the adapter under. Used as the `model` field in
    /// subsequent chat/completions requests.
    pub lora_name: String,
    /// Absolute path (or HuggingFace repo) to the adapter weights. Must be
    /// reachable from the vLLM process (inside the container if Dockerised).
    pub lora_path: String,
}

/// Request body for `POST /v1/unload_lora_adapter`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnloadLoraRequest {
    /// Name previously registered via [`LoadLoraRequest`].
    pub lora_name: String,
}

/// Load a LoRA adapter into a running vLLM server.
///
/// `base_url` is the vLLM root (e.g. `http://localhost:8000`). Returns
/// `Err` if the server responds non-2xx or cannot be reached.
pub fn load_lora_adapter(base_url: &str, lora_name: &str, lora_path: &str) -> Result<(), String> {
    let url = format!("{}/v1/load_lora_adapter", base_url.trim_end_matches('/'));
    let body = LoadLoraRequest {
        lora_name: lora_name.to_string(),
        lora_path: lora_path.to_string(),
    };
    ureq::post(&url)
        .timeout(std::time::Duration::from_secs(60))
        .send_json(&body)
        .map_err(|e| format!("POST {}: {}", url, e))?;
    Ok(())
}

/// Unload a previously loaded LoRA adapter.
pub fn unload_lora_adapter(base_url: &str, lora_name: &str) -> Result<(), String> {
    let url = format!("{}/v1/unload_lora_adapter", base_url.trim_end_matches('/'));
    let body = UnloadLoraRequest {
        lora_name: lora_name.to_string(),
    };
    ureq::post(&url)
        .timeout(std::time::Duration::from_secs(30))
        .send_json(&body)
        .map_err(|e| format!("POST {}: {}", url, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_lora_request_serializes_expected_fields() {
        let req = LoadLoraRequest {
            lora_name: "math-tutor".into(),
            lora_path: "/models/math-tutor-lora".into(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["lora_name"], "math-tutor");
        assert_eq!(json["lora_path"], "/models/math-tutor-lora");
    }

    #[test]
    fn unload_lora_request_serializes_expected_fields() {
        let req = UnloadLoraRequest {
            lora_name: "math-tutor".into(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["lora_name"], "math-tutor");
        assert_eq!(json.as_object().unwrap().len(), 1);
    }

    #[test]
    fn load_returns_err_when_server_unreachable() {
        let res = load_lora_adapter("http://127.0.0.1:1", "x", "/tmp/x");
        assert!(res.is_err());
    }

    #[test]
    fn unload_returns_err_when_server_unreachable() {
        let res = unload_lora_adapter("http://127.0.0.1:1", "x");
        assert!(res.is_err());
    }
}
