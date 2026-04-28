// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Cross-module integration tests for the mmproj support introduced in
//! V90.26. Exercises the full chain:
//!
//! ```text
//! AiConfig (mmproj_path) → MultimodalProjector::from_path
//!                       → vision_runtime_ready_for(config, capability)
//!                       → providers::looks_like_mmproj_error (via dispatch)
//! ```
//!
//! All tests are vision-gated. They use synthetic GGUF-prefixed bytes
//! written to a per-pid tempdir; no real `llama-server` is touched.

#![cfg(feature = "vision")]

use std::path::PathBuf;

use ai_assistant::llamacpp_capability::{parse_props, LlamaCppCapability};
use ai_assistant::vision::agent_bridge::{vision_runtime_ready_for, vision_supported_for};
use ai_assistant::{AiConfig, AiProvider, GGUF_MAGIC, MIN_PROJECTOR_BYTES};

fn tmpdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ai_assistant_mmproj_integration_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn write_synthetic_projector(name: &str) -> PathBuf {
    let mut bytes = Vec::with_capacity(MIN_PROJECTOR_BYTES as usize + 16);
    bytes.extend_from_slice(&GGUF_MAGIC);
    bytes.resize(MIN_PROJECTOR_BYTES as usize + 16, 0xAB);
    let p = tmpdir().join(name);
    std::fs::write(&p, &bytes).expect("write synthetic gguf");
    p
}

#[test]
fn ai_config_persists_mmproj_path() {
    let p = write_synthetic_projector("persist.gguf");
    let mut cfg = AiConfig::default();
    cfg.mmproj_path = Some(p.clone());
    let validated = cfg.validated_mmproj().expect("path was set");
    let proj = validated.expect("synthetic file must validate");
    assert!(proj.path().is_absolute());
    assert!(proj.size_bytes() >= MIN_PROJECTOR_BYTES);
}

#[test]
fn ai_config_no_mmproj_path_yields_none() {
    let cfg = AiConfig::default();
    assert!(cfg.validated_mmproj().is_none());
}

#[test]
fn ai_config_invalid_mmproj_path_returns_typed_error() {
    let mut cfg = AiConfig::default();
    cfg.mmproj_path = Some(PathBuf::from("/__definitely_not_a_path__/nope.gguf"));
    let res = cfg.validated_mmproj().expect("path was set");
    assert!(res.is_err());
}

#[test]
fn runtime_ready_check_blocks_when_probe_says_no_mmproj() {
    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LlamaCpp;
    cfg.selected_model = "llava".to_string();

    let cap = LlamaCppCapability {
        build_info: Some("b1".into()),
        is_prismml_fork: false,
        supports_q1_0: false,
        supports_ternary: false,
        default_ctx: Some(4096),
        multimodal: Some(false),
    };

    let err = vision_runtime_ready_for(&cfg, Some(&cap)).expect_err("must reject");
    assert!(err.to_string().contains("--mmproj"));
}

#[test]
fn runtime_ready_check_passes_when_probe_reports_mmproj() {
    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LlamaCpp;
    cfg.selected_model = "llava".to_string();

    let cap = LlamaCppCapability {
        build_info: Some("b1".into()),
        is_prismml_fork: false,
        supports_q1_0: false,
        supports_ternary: false,
        default_ctx: Some(4096),
        multimodal: Some(true),
    };
    assert!(vision_runtime_ready_for(&cfg, Some(&cap)).is_ok());
}

#[test]
fn runtime_ready_passes_when_no_probe_available() {
    // Even with `mmproj_path` set, lack of a runtime probe must not
    // block: we don't try to spawn the server ourselves, and rejecting
    // would block valid LM Studio sessions.
    let p = write_synthetic_projector("no_probe.gguf");
    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LMStudio;
    cfg.selected_model = "llava".into();
    cfg.mmproj_path = Some(p);
    assert!(vision_runtime_ready_for(&cfg, None).is_ok());
}

#[test]
fn props_parse_to_capability_then_block_on_no_multimodal() {
    let body = serde_json::json!({
        "build_info": "b1",
    });
    let cap = parse_props(&body);
    assert_eq!(cap.multimodal, Some(false));

    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LlamaCpp;
    cfg.selected_model = "llava".into();
    let err = vision_runtime_ready_for(&cfg, Some(&cap)).expect_err("must reject");
    assert!(err.to_string().contains("multimodal projector"));
}

#[test]
fn props_with_clip_path_unblocks_runtime_check() {
    let body = serde_json::json!({
        "build_info": "b1",
        "clip_model_path": "/models/llava/mmproj.gguf",
    });
    let cap = parse_props(&body);
    assert_eq!(cap.multimodal, Some(true));

    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LlamaCpp;
    cfg.selected_model = "llava".into();
    assert!(vision_runtime_ready_for(&cfg, Some(&cap)).is_ok());
}

#[test]
fn vision_supported_for_includes_llamacpp_and_lmstudio() {
    let mut cfg = AiConfig::default();
    cfg.provider = AiProvider::LlamaCpp;
    assert!(vision_supported_for(&cfg));
    cfg.provider = AiProvider::LMStudio;
    assert!(vision_supported_for(&cfg));
}

#[test]
fn projector_traversal_path_rejected_via_config() {
    let mut cfg = AiConfig::default();
    cfg.mmproj_path = Some(PathBuf::from("..").join("evil.gguf"));
    let res = cfg.validated_mmproj().expect("path set");
    assert!(matches!(
        res,
        Err(ai_assistant::MmprojValidationError::PathTraversal)
    ));
}

#[test]
fn projector_too_small_rejected_via_config() {
    let p = tmpdir().join("tiny.gguf");
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(&GGUF_MAGIC);
    bytes.resize(64, 0);
    std::fs::write(&p, &bytes).expect("write tiny");
    let mut cfg = AiConfig::default();
    cfg.mmproj_path = Some(p);
    let res = cfg.validated_mmproj().expect("path set");
    assert!(matches!(
        res,
        Err(ai_assistant::MmprojValidationError::TooSmall { .. })
    ));
}
