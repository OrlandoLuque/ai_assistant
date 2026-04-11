// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! V79 FFI cross-crate integration test.
//!
//! Exercises the `extern "C"` symbols from `ai_assistant::ffi` in a
//! separate compilation unit. Catches any accidentally-private type or
//! missing `pub use` re-export that the in-module tests would miss.

#![cfg(feature = "ffi")]

use ai_assistant::ffi::{
    ai_assistant_abi_version, ai_assistant_free, ai_assistant_free_string, ai_assistant_last_error,
    ai_assistant_new, ai_assistant_new_with_prompt, ai_assistant_set_max_history,
    ai_assistant_set_model, ai_assistant_set_provider, ai_assistant_set_system_prompt,
    ai_assistant_set_temperature, ai_assistant_version, AiProviderKind,
};
use std::ffi::{CStr, CString};
use std::ptr;

#[test]
fn roundtrip_handle_from_another_crate() {
    unsafe {
        let h = ai_assistant_new();
        assert!(!h.is_null());
        let prompt = CString::new("hello from tests/ffi_integration.rs").unwrap();
        assert_eq!(ai_assistant_set_system_prompt(h, prompt.as_ptr()), 0);
        assert_eq!(ai_assistant_set_provider(h, AiProviderKind::Ollama), 0);
        let model = CString::new("llama3.2:3b").unwrap();
        assert_eq!(ai_assistant_set_model(h, model.as_ptr()), 0);
        assert_eq!(ai_assistant_set_temperature(h, 0.5), 0);
        assert_eq!(ai_assistant_set_max_history(h, 10), 0);
        ai_assistant_free(h);
    }
}

#[test]
fn last_error_populated_after_null_input() {
    unsafe {
        let h = ai_assistant_new_with_prompt(ptr::null());
        assert!(h.is_null());
        let err = ai_assistant_last_error();
        assert!(!err.is_null());
        let msg = CStr::from_ptr(err).to_string_lossy();
        assert!(msg.contains("null"));
    }
}

#[test]
fn version_and_abi_probes() {
    let v = ai_assistant_version();
    assert!(!v.is_null());
    let vs = unsafe { CStr::from_ptr(v) }.to_string_lossy();
    assert!(vs.contains('.'), "version: {vs}");
    assert_eq!(ai_assistant_abi_version(), 1);
}

#[test]
fn free_string_null_is_safe() {
    unsafe { ai_assistant_free_string(ptr::null_mut()) };
}

#[test]
fn temperature_strict_reject_from_another_crate() {
    unsafe {
        let h = ai_assistant_new();
        assert_eq!(ai_assistant_set_temperature(h, f32::NAN), -5);
        assert_eq!(ai_assistant_set_temperature(h, 3.0), -5);
        assert_eq!(ai_assistant_set_temperature(h, 0.9), 0);
        ai_assistant_free(h);
    }
}
