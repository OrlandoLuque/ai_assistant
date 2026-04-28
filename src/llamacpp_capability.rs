//! V102: `llama-server` capability probe.
//!
//! `llama.cpp`'s `llama-server` exposes a `/props` endpoint that returns
//! build info and the list of quantization kernels the binary was compiled
//! with. V102 uses this to detect whether the running server is the
//! PrismML fork (`PrismML-Eng/llama.cpp`) — the only build that ships the
//! `Q1_0` kernel used by the Bonsai 1-bit models.
//!
//! The probe is intentionally *passive*: it never modifies server state.

use crate::retry::{retry_with_config, RetryConfig};
use serde::{Deserialize, Serialize};

/// What a `/props` probe tells us about the running `llama-server`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlamaCppCapability {
    /// Raw build string reported by the server, if any.
    pub build_info: Option<String>,
    /// Whether the binary looks like the PrismML fork (name contains
    /// `prism` or advertises `Q1_0` / ternary kernels).
    pub is_prismml_fork: bool,
    /// Whether the server advertises `Q1_0` 1.125-bpw quantization.
    pub supports_q1_0: bool,
    /// Whether the server advertises ternary (`{-1, 0, 1}`) quantization.
    pub supports_ternary: bool,
    /// Default context window reported by the server, if any.
    pub default_ctx: Option<u32>,
    /// Whether the server reports a multimodal projector (mmproj) loaded.
    /// `Some(true)` if `/props` advertised any of `multimodal`, `has_clip`,
    /// `mmproj_loaded`, `mmproj`, or `clip_model`. `Some(false)` if the
    /// endpoint replied without any of those fields. `None` if no probe
    /// has run yet (different from "probe says no projector"). The field
    /// names vary across forks, so detection is best-effort.
    #[serde(default)]
    pub multimodal: Option<bool>,
}

impl LlamaCppCapability {
    /// Can the server run a model with the given quantization tag?
    ///
    /// Known tags: `Q1_0`, `Ternary`, `Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`,
    /// `F32`. Standard quantizations are assumed present on any build.
    pub fn can_run_quantization(&self, quant: &str) -> bool {
        let q = quant.trim().to_ascii_lowercase();
        if q == "q1_0" {
            self.supports_q1_0
        } else if q.contains("ternary") {
            self.supports_ternary
        } else {
            // Standard quants ship with every llama.cpp build.
            true
        }
    }
}

/// Probe a running `llama-server` at `base_url` (e.g.
/// `http://localhost:8080`).
///
/// On success returns the parsed capability. On network error returns
/// `Err(String)` with a human-readable message. Uses `RetryConfig::fast`
/// (2 retries) so a single transient TCP hiccup doesn't fail the probe.
pub fn probe_llamacpp(base_url: &str) -> Result<LlamaCppCapability, String> {
    let url = format!("{}/props", base_url.trim_end_matches('/'));
    let body: serde_json::Value = retry_with_config(RetryConfig::fast(), || {
        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()?;
        let v: serde_json::Value = resp.into_json()?;
        Ok(v)
    })
    .map_err(|e| format!("GET {}: {}", url, e))?;

    Ok(parse_props(&body))
}

/// Parse a `/props` JSON body into a `LlamaCppCapability`.
///
/// Split out from `probe_llamacpp` so tests don't need a live server.
pub fn parse_props(body: &serde_json::Value) -> LlamaCppCapability {
    let build_info = body
        .get("build_info")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            body.get("system_info")
                .and_then(|v| v.as_str())
                .map(String::from)
        });

    let default_ctx = body
        .get("default_generation_settings")
        .and_then(|v| v.get("n_ctx"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .or_else(|| body.get("n_ctx").and_then(|v| v.as_u64()).map(|v| v as u32));

    // Heuristics: PrismML fork advertises itself either via the build
    // string ("PrismML" or "prism-ml") or via the `quantizations` array
    // listing `Q1_0` / ternary.
    let lower_build = build_info
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let quant_list: Vec<String> = body
        .get("quantizations")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_ascii_lowercase()))
                .collect()
        })
        .unwrap_or_default();

    let supports_q1_0 = quant_list.iter().any(|q| q == "q1_0");
    let supports_ternary = quant_list.iter().any(|q| q.contains("ternary"));

    let is_prismml_fork = lower_build.contains("prism")
        || lower_build.contains("prism-ml")
        || supports_q1_0
        || supports_ternary;

    let multimodal = detect_multimodal(body);

    LlamaCppCapability {
        build_info,
        is_prismml_fork,
        supports_q1_0,
        supports_ternary,
        default_ctx,
        multimodal,
    }
}

/// Heuristic detection of mmproj-loaded state from a `/props` body.
/// Different `llama.cpp` forks advertise the same fact under different
/// keys, so we accept any of: `multimodal` (bool), `has_clip` (bool),
/// `mmproj_loaded` (bool), or a non-empty string in `mmproj` /
/// `clip_model`. Returns `Some(false)` if none of those fields are
/// present — the server is reachable, just not reporting vision.
fn detect_multimodal(body: &serde_json::Value) -> Option<bool> {
    for key in ["multimodal", "has_clip", "mmproj_loaded"] {
        if let Some(b) = body.get(key).and_then(|v| v.as_bool()) {
            if b {
                return Some(true);
            }
        }
    }
    for key in ["mmproj", "clip_model", "clip_model_path"] {
        if let Some(s) = body.get(key).and_then(|v| v.as_str()) {
            if !s.trim().is_empty() {
                return Some(true);
            }
        }
    }
    Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_upstream_minimal_props() {
        let body = json!({
            "build_info": "b2345",
            "n_ctx": 4096,
        });
        let cap = parse_props(&body);
        assert_eq!(cap.build_info.as_deref(), Some("b2345"));
        assert_eq!(cap.default_ctx, Some(4096));
        assert!(!cap.supports_q1_0);
        assert!(!cap.supports_ternary);
        assert!(!cap.is_prismml_fork);
    }

    #[test]
    fn parses_prismml_build_string() {
        let body = json!({
            "build_info": "PrismML llama.cpp fork b4001",
            "quantizations": ["Q4_K_M", "Q1_0", "F16"],
        });
        let cap = parse_props(&body);
        assert!(cap.is_prismml_fork);
        assert!(cap.supports_q1_0);
        assert!(!cap.supports_ternary);
    }

    #[test]
    fn parses_ternary_quant_only() {
        let body = json!({
            "build_info": "some-build",
            "quantizations": ["Q4_K_M", "Ternary"],
        });
        let cap = parse_props(&body);
        assert!(cap.supports_ternary);
        assert!(cap.is_prismml_fork, "ternary kernel implies fork");
    }

    #[test]
    fn reads_default_ctx_from_nested_settings() {
        let body = json!({
            "build_info": "b2345",
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let cap = parse_props(&body);
        assert_eq!(cap.default_ctx, Some(8192));
    }

    #[test]
    fn can_run_quantization_handles_unknown_tags() {
        let cap = LlamaCppCapability {
            build_info: None,
            is_prismml_fork: false,
            supports_q1_0: false,
            supports_ternary: false,
            default_ctx: None,
            multimodal: None,
        };
        assert!(cap.can_run_quantization("Q4_K_M"));
        assert!(cap.can_run_quantization("F16"));
        assert!(!cap.can_run_quantization("Q1_0"));
        assert!(!cap.can_run_quantization("Ternary"));
    }

    #[test]
    fn can_run_quantization_on_prismml_fork() {
        let cap = LlamaCppCapability {
            build_info: Some("PrismML fork".into()),
            is_prismml_fork: true,
            supports_q1_0: true,
            supports_ternary: true,
            default_ctx: None,
            multimodal: None,
        };
        assert!(cap.can_run_quantization("Q1_0"));
        assert!(cap.can_run_quantization("Ternary"));
        assert!(cap.can_run_quantization("Q4_K_M"));
    }

    #[test]
    fn system_info_fallback_build_string() {
        let body = json!({ "system_info": "prism-ml build xyz" });
        let cap = parse_props(&body);
        assert_eq!(cap.build_info.as_deref(), Some("prism-ml build xyz"));
        assert!(cap.is_prismml_fork);
    }

    #[test]
    fn detects_multimodal_via_explicit_bool() {
        let body = json!({ "build_info": "b1", "multimodal": true });
        let cap = parse_props(&body);
        assert_eq!(cap.multimodal, Some(true));
    }

    #[test]
    fn detects_multimodal_via_clip_path() {
        let body = json!({
            "build_info": "b1",
            "clip_model_path": "/models/llava/mmproj.gguf"
        });
        let cap = parse_props(&body);
        assert_eq!(cap.multimodal, Some(true));
    }

    #[test]
    fn no_multimodal_fields_yields_some_false() {
        let body = json!({ "build_info": "b1" });
        let cap = parse_props(&body);
        assert_eq!(
            cap.multimodal,
            Some(false),
            "reachable server with no mmproj fields means probe answered, no projector"
        );
    }

    #[test]
    fn empty_clip_path_does_not_count() {
        let body = json!({ "build_info": "b1", "mmproj": "" });
        let cap = parse_props(&body);
        assert_eq!(cap.multimodal, Some(false));
    }
}
