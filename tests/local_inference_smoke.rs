// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! End-to-end smoke test for [`ai_assistant::local_inference`].
//!
//! Two layers:
//!
//! 1. **Always-on**: drives the in-tree `StubBackend` through the public
//!    `Backend` trait. Verifies the load → generate → stats path works
//!    end-to-end and produces a `SloRecord` with sensible fields. No
//!    model file required.
//!
//! 2. **Tiny-model gated** (env var `AI_LOCAL_INFER_TINY_MODEL`): when
//!    set to a path, drives the chosen backend (env var
//!    `AI_LOCAL_INFER_BACKEND`, default `candle`) against the file. This
//!    iteration only validates the request/error surface — real backends
//!    surface `NotImplemented` until #319 (Candle CPU) and #314
//!    (llama-cpp-2 GGUF) land. Keeping the test in tree means the smoke
//!    test starts asserting real generation the moment those backends
//!    flip from stub to real, with no test-side change.

#![cfg(feature = "local-inference")]

use std::time::Instant;

use ai_assistant::local_inference::{
    load, vram, Backend, BackendError, BackendKind, GenParams, LocalInferenceConfig, SloRecord,
};

const SLO_LOAD_MS: u64 = 30_000;
const SLO_FIRST_CHUNK_MS: u64 = 5_000; // CPU-only; production budget is 1 s
const SLO_MIN_TPS: f64 = 1.0; // CPU baseline; production budget is 5/s

#[test]
fn stub_backend_full_roundtrip() {
    let cfg = LocalInferenceConfig::builder(BackendKind::Stub, "")
        .ctx_size(2048)
        .n_gpu_layers(0)
        .build();

    let load_start = Instant::now();
    let mut backend: Box<dyn Backend> = load(&cfg).expect("stub loads");
    let load_ms = load_start.elapsed().as_millis() as u64;

    assert_eq!(backend.kind(), BackendKind::Stub);

    let gen_start = Instant::now();
    let mut first_chunk_ms: Option<u64> = None;
    let mut sink: Vec<String> = Vec::new();
    let mut on_chunk = |c: &str| {
        if first_chunk_ms.is_none() {
            first_chunk_ms = Some(gen_start.elapsed().as_millis() as u64);
        }
        sink.push(c.to_string());
    };
    let stats = backend
        .generate("Hello, smoke", &GenParams::default(), &mut on_chunk)
        .expect("generate succeeds");
    let total_ms = gen_start.elapsed().as_millis() as u64;

    assert!(!sink.is_empty(), "expected at least one chunk");
    assert!(stats.generated_tokens > 0);
    assert!(stats.prompt_tokens >= 2);

    let rec = SloRecord {
        ts_unix_ms: SloRecord::now_ms(),
        backend: backend.kind().name().to_string(),
        model_path: String::new(),
        load_ms,
        first_chunk_ms: first_chunk_ms.unwrap_or(total_ms),
        total_ms,
        prompt_tokens: stats.prompt_tokens,
        generated_tokens: stats.generated_tokens,
        tokens_per_sec: stats.tokens_per_sec,
        n_gpu_layers_requested: 0,
        n_gpu_layers_used: 0,
        peak_vram_mib: stats.peak_vram_mib,
    };
    let json = serde_json::to_string(&rec).expect("record serializes");
    assert!(json.contains("\"backend\":\"stub\""));
}

#[test]
fn vram_detection_returns_consistent_shape() {
    // Best-effort detection — just assert the shape if anything is reported.
    if let Some((total, free)) = vram::detect_nvidia_mib() {
        assert!(total > 0, "reported total VRAM is zero");
        assert!(free <= total, "free > total VRAM ({} > {})", free, total);
    }
}

#[test]
fn vram_clamp_policy_under_realistic_inputs() {
    // 4 GiB model, 32 layers → 128 MiB/layer. With 2 GiB free, expect
    // 2048/128 = 16 layers.
    assert_eq!(vram::clamp_gpu_layers(4096, 32, 32, 2048), 16);
    // Asking for more layers than the model has should cap to total.
    assert_eq!(vram::clamp_gpu_layers(4096, 99, 32, 8192), 32);
    // No VRAM → 0 (CPU only).
    assert_eq!(vram::clamp_gpu_layers(4096, 32, 32, 0), 0);
}

/// Tiny-model gate. Skips silently when the env var is not set so CI stays
/// hermetic. When set, drives the chosen backend against the model file.
#[test]
fn tiny_model_smoke() {
    let model = match std::env::var("AI_LOCAL_INFER_TINY_MODEL") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("skip: AI_LOCAL_INFER_TINY_MODEL not set");
            return;
        }
    };
    let backend_name = std::env::var("AI_LOCAL_INFER_BACKEND").unwrap_or_else(|_| "candle".into());
    let backend = match backend_name.as_str() {
        "stub" => BackendKind::Stub,
        "candle" => BackendKind::Candle,
        "llama-cpp" | "llama-cpp-2" => BackendKind::LlamaCpp,
        other => panic!("unknown AI_LOCAL_INFER_BACKEND: {}", other),
    };

    let cfg = LocalInferenceConfig::builder(backend, &model)
        .ctx_size(2048)
        .n_gpu_layers(0)
        .build();

    let load_start = Instant::now();
    let mut backend_box: Box<dyn Backend> = match load(&cfg) {
        Ok(b) => b,
        Err(BackendError::NotImplemented(name)) => {
            eprintln!(
                "skip: backend '{}' not compiled in (env requested it)",
                name
            );
            return;
        }
        Err(e) => panic!("load failed for tiny model: {}", e),
    };
    let load_ms = load_start.elapsed().as_millis() as u64;
    assert!(
        load_ms < SLO_LOAD_MS,
        "load_ms exceeded SLO: {} > {}",
        load_ms,
        SLO_LOAD_MS
    );

    let gen_start = Instant::now();
    let mut first_chunk_ms: Option<u64> = None;
    let mut chunks = 0usize;
    let mut on_chunk = |_c: &str| {
        if first_chunk_ms.is_none() {
            first_chunk_ms = Some(gen_start.elapsed().as_millis() as u64);
        }
        chunks += 1;
    };
    let stats = backend_box
        .generate(
            "The quick brown fox",
            &GenParams {
                max_tokens: 32,
                ..GenParams::default()
            },
            &mut on_chunk,
        )
        .expect("generate succeeds");
    let total_ms = gen_start.elapsed().as_millis() as u64;

    assert!(chunks > 0, "expected at least one chunk");
    assert!(stats.generated_tokens > 0);
    if let Some(fc) = first_chunk_ms {
        assert!(
            fc < SLO_FIRST_CHUNK_MS,
            "first_chunk_ms exceeded SLO: {} > {}",
            fc,
            SLO_FIRST_CHUNK_MS
        );
    }
    assert!(
        stats.tokens_per_sec >= SLO_MIN_TPS,
        "tokens_per_sec below SLO: {} < {}",
        stats.tokens_per_sec,
        SLO_MIN_TPS
    );
    eprintln!(
        "tiny_model_smoke: backend={:?} load_ms={} first={}ms total={}ms tok/s={:.1}",
        backend,
        load_ms,
        first_chunk_ms.unwrap_or(total_ms),
        total_ms,
        stats.tokens_per_sec
    );
}
