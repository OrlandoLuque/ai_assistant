// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_assistant — V79 C FFI bindings.
//!
//! Exposes 20 `extern "C"` entry points wrapping [`AiAssistant`] so the
//! library can be consumed from C, C++, C#, Unity, Unreal, Bevy, and any
//! language with a C FFI bridge.
//!
//! # Threading contract
//!
//! Each handle is **single-threaded** (SQLite / libcurl-style). The C
//! caller must keep all calls on the same OS thread. In debug builds this
//! is enforced via an atomic thread-pin — the first call records the
//! caller's thread ID and subsequent calls from other threads panic
//! (caught by [`guard`] and reported as [`AI_ERR_PANIC`]). Release builds
//! trust the caller for zero overhead.
//!
//! This pattern is chosen over a `Mutex<AiAssistant>` because
//! `AiAssistant` holds feature-gated trait objects and third-party types
//! whose `Send` status cannot be universally guaranteed.
//!
//! # Memory model
//!
//! - Input strings (`*const c_char`): borrowed for the duration of the
//!   call. Must be NUL-terminated, valid UTF-8, and non-null (unless
//!   explicitly documented).
//! - Output strings (`*mut c_char`): allocated by Rust via
//!   [`CString::into_raw`]. **The C caller must free them via
//!   [`ai_assistant_free_string`]** and MUST NOT call `free(3)` directly.
//! - Handles (`*mut AiAssistantHandle`): allocated by Rust via
//!   `Box::into_raw`. Free via [`ai_assistant_free`], null-safe.
//!
//! Double-free is not runtime-detectable without sanitizers — the C
//! caller should set its pointer to NULL after free.
//!
//! # Error reporting
//!
//! Most entry points return an `int` status. On non-zero return, the
//! thread-local [`ai_assistant_last_error`] gives a human-readable hint.
//! The error buffer is valid until the next FFI call on that thread.
//!
//! # Panic boundary
//!
//! Every entry point wraps its body in [`std::panic::catch_unwind`]. A
//! caught panic sets the last-error slot and returns [`AI_ERR_PANIC`].
//!
//! **Caveat**: the default `release` Cargo profile uses `panic = "abort"`,
//! which turns `catch_unwind` into a no-op. FFI consumers almost certainly
//! want `--profile release-fast` (which keeps `panic = "unwind"`).
//! `build.rs` emits a `cargo:warning` when it detects the dangerous combo.

#![allow(clippy::missing_safety_doc)]

use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::AiProvider;
use crate::messages::AiResponse;
use crate::AiAssistant;

// ---------------------------------------------------------------------------
// Return codes
// ---------------------------------------------------------------------------

/// Success.
#[no_mangle]
pub static AI_OK: c_int = 0;
/// A required pointer argument was NULL.
#[no_mangle]
pub static AI_ERR_NULL_PTR: c_int = -1;
/// A string argument was not valid UTF-8.
#[no_mangle]
pub static AI_ERR_INVALID_UTF8: c_int = -2;
/// A Rust panic was caught at the FFI boundary.
#[no_mangle]
pub static AI_ERR_PANIC: c_int = -3;
/// An internal synchronization primitive was poisoned (should not happen).
#[no_mangle]
pub static AI_ERR_POISONED: c_int = -4;
/// Internal error — see `ai_assistant_last_error()` for details.
#[no_mangle]
pub static AI_ERR_INTERNAL: c_int = -5;
/// Unknown or unsupported provider (e.g. `OpenAICompatible` without its URL).
#[no_mangle]
pub static AI_ERR_UNKNOWN_PROVIDER: c_int = -6;
/// Upstream provider call failed (network, auth, timeout, ...).
#[no_mangle]
pub static AI_ERR_SEND_FAILED: c_int = -7;
/// No response was produced (empty stream).
#[no_mangle]
pub static AI_ERR_NO_RESPONSE: c_int = -8;

// Duplicated as `const` for use in match arms, since `static` can't be
// used in patterns.
const OK: c_int = 0;
const E_NULL_PTR: c_int = -1;
const E_INVALID_UTF8: c_int = -2;
const E_PANIC: c_int = -3;
const _E_POISONED: c_int = -4;
const E_INTERNAL: c_int = -5;
const E_UNKNOWN_PROVIDER: c_int = -6;
const E_SEND_FAILED: c_int = -7;
const _E_NO_RESPONSE: c_int = -8;

// ---------------------------------------------------------------------------
// Flat provider enum (C-visible)
// ---------------------------------------------------------------------------

/// Flat C enum mirroring Rust's `AiProvider` (17 variants).
///
/// Two Rust variants carry data (`OpenAICompatible { base_url }` and
/// `Bedrock { region }`). They're represented as unit values here and
/// configured via companion setters:
/// - [`ai_assistant_set_openai_compatible_url`]
/// - [`ai_assistant_set_bedrock_region`]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AiProviderKind {
    Ollama,
    LMStudio,
    TextGenWebUI,
    KoboldCpp,
    LocalAI,
    OpenAICompatible,
    OpenAI,
    Anthropic,
    Gemini,
    Bedrock,
    Groq,
    Together,
    Fireworks,
    DeepSeek,
    Mistral,
    Perplexity,
    OpenRouter,
}

// ---------------------------------------------------------------------------
// Handle / Inner
// ---------------------------------------------------------------------------

/// Opaque handle type seen by C callers.
///
/// Callers must treat the pointer as opaque and only manipulate it via
/// the `ai_assistant_*` functions.
#[repr(C)]
pub struct AiAssistantHandle {
    _private: [u8; 0],
}

/// Internal state behind the opaque handle.
///
/// # Safety
///
/// `Inner` contains `UnsafeCell`, `Cell`, and `RefCell` — all `!Sync`.
/// The `unsafe impl Send + Sync` is upheld by the **single-thread
/// contract** (debug-enforced via `owner_thread`): the C caller promises
/// to use each handle from one thread only, so there is never aliasing
/// access from multiple threads.
struct Inner {
    assistant: UnsafeCell<AiAssistant>,
    provider_kind: Cell<AiProviderKind>,
    openai_compatible_url: RefCell<Option<String>>,
    bedrock_region: RefCell<Option<String>>,
    /// Debug-only thread pin. 0 = unpinned; any other value is the
    /// per-thread ID stamped by [`check_thread`] on first use.
    #[cfg(debug_assertions)]
    owner_thread: AtomicU64,
}

// SAFETY: See the `Inner` doc comment — single-thread contract.
unsafe impl Send for Inner {}
// SAFETY: See the `Inner` doc comment — single-thread contract.
unsafe impl Sync for Inner {}

/// Compile-time witness that we intentionally do NOT require
/// `AiAssistant: Send`. Flipping the `any()` below to `all()` would
/// re-check the Send bound under the current feature set; today we
/// cannot guarantee it in full-feature build, hence the `UnsafeCell`
/// + single-thread model.
#[cfg(any())]
const _ASSISTANT_SEND_PROBE: fn() = || {
    fn probe<T: Send>() {}
    probe::<AiAssistant>();
};

// ---------------------------------------------------------------------------
// Thread-local last error
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    // Replace interior NULs (CString rejects them) with '?'.
    let cleaned: String = msg
        .chars()
        .map(|c| if c == '\0' { '?' } else { c })
        .collect();
    if let Ok(c) = CString::new(cleaned) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
    }
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

// ---------------------------------------------------------------------------
// Thread pin (debug-only)
// ---------------------------------------------------------------------------

/// Returns a unique, non-zero u64 for the current OS thread. Each thread
/// gets the next counter value the first time it calls this function.
#[cfg(debug_assertions)]
fn current_thread_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static THIS: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    THIS.with(|id| *id)
}

/// Debug-only single-thread enforcement. On the first call, CAS the
/// slot from 0 to the caller's thread ID. On subsequent calls, panic if
/// the caller is on a different thread.
#[cfg(debug_assertions)]
fn check_thread(inner: &Inner) {
    let me = current_thread_id();
    match inner
        .owner_thread
        .compare_exchange(0, me, Ordering::AcqRel, Ordering::Acquire)
    {
        Ok(_) => { /* first use — pinned */ }
        Err(existing) => {
            if existing != me {
                panic!(
                    "ai_assistant: handle used from wrong thread (pinned to {existing}, now {me})"
                );
            }
        }
    }
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn check_thread(_inner: &Inner) {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts a formatted panic message from a `catch_unwind` payload.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Wraps an FFI body in a panic boundary. Caught panics set the
/// last-error slot and return `AI_ERR_PANIC`.
fn guard<F: FnOnce() -> c_int>(name: &'static str, f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(rc) => rc,
        Err(e) => {
            let msg = panic_message(e);
            set_last_error(&format!("panic in {name}: {msg}"));
            E_PANIC
        }
    }
}

/// Converts a raw handle pointer back into an `&Inner`.
///
/// Returns `None` if `handle` is null. The caller must ensure `handle`
/// came from `ai_assistant_new*` and hasn't been freed.
///
/// # Safety
///
/// `handle` must either be null or a pointer produced by `Box::into_raw`
/// on a `Box<Inner>` that has not been freed.
unsafe fn handle_to_inner<'a>(handle: *mut AiAssistantHandle) -> Option<&'a Inner> {
    if handle.is_null() {
        None
    } else {
        Some(&*(handle as *const Inner))
    }
}

/// Validates a `*const c_char` input: non-null + valid UTF-8. Returns
/// the borrowed `&str` on success, or a negative error code on failure.
///
/// # Safety
///
/// `p` must either be null or a pointer to a NUL-terminated C string
/// whose bytes are readable for the duration of the call.
unsafe fn cstr_to_str<'a>(p: *const c_char) -> Result<&'a str, c_int> {
    if p.is_null() {
        return Err(E_NULL_PTR);
    }
    CStr::from_ptr(p).to_str().map_err(|_| E_INVALID_UTF8)
}

/// Builds an `AiProvider` from the handle's stored kind + companion
/// setters. **Exhaustive match** — adding a new Rust variant causes a
/// compile error here (and correctly so).
///
/// Note: `AiProvider` is `#[non_exhaustive]`, but that only affects
/// external crates. Because `ffi.rs` is a module of the defining crate,
/// we can match exhaustively. If FFI is ever split into a sibling crate,
/// this match must grow a `_ =>` arm returning `AI_ERR_UNKNOWN_PROVIDER`.
fn build_provider(inner: &Inner) -> Result<AiProvider, c_int> {
    use AiProviderKind::*;
    match inner.provider_kind.get() {
        Ollama => Ok(AiProvider::Ollama),
        LMStudio => Ok(AiProvider::LMStudio),
        TextGenWebUI => Ok(AiProvider::TextGenWebUI),
        KoboldCpp => Ok(AiProvider::KoboldCpp),
        LocalAI => Ok(AiProvider::LocalAI),
        OpenAICompatible => match inner.openai_compatible_url.borrow().clone() {
            Some(url) => Ok(AiProvider::OpenAICompatible { base_url: url }),
            None => {
                set_last_error(
                    "OpenAICompatible requires prior ai_assistant_set_openai_compatible_url",
                );
                Err(E_UNKNOWN_PROVIDER)
            }
        },
        OpenAI => Ok(AiProvider::OpenAI),
        Anthropic => Ok(AiProvider::Anthropic),
        Gemini => Ok(AiProvider::Gemini),
        Bedrock => match inner.bedrock_region.borrow().clone() {
            Some(region) => Ok(AiProvider::Bedrock { region }),
            None => {
                set_last_error("Bedrock requires prior ai_assistant_set_bedrock_region");
                Err(E_UNKNOWN_PROVIDER)
            }
        },
        Groq => Ok(AiProvider::Groq),
        Together => Ok(AiProvider::Together),
        Fireworks => Ok(AiProvider::Fireworks),
        DeepSeek => Ok(AiProvider::DeepSeek),
        Mistral => Ok(AiProvider::Mistral),
        Perplexity => Ok(AiProvider::Perplexity),
        OpenRouter => Ok(AiProvider::OpenRouter),
    }
}

/// Allocates a fresh handle by boxing `assistant` with default state.
fn make_handle(assistant: AiAssistant) -> *mut AiAssistantHandle {
    let inner = Inner {
        assistant: UnsafeCell::new(assistant),
        provider_kind: Cell::new(AiProviderKind::Ollama),
        openai_compatible_url: RefCell::new(None),
        bedrock_region: RefCell::new(None),
        #[cfg(debug_assertions)]
        owner_thread: AtomicU64::new(0),
    };
    Box::into_raw(Box::new(inner)) as *mut AiAssistantHandle
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Creates a new assistant handle with default settings.
///
/// Returns `NULL` only if a Rust panic was caught. The C caller should
/// check for NULL and, if so, call `ai_assistant_last_error()`.
#[no_mangle]
pub extern "C" fn ai_assistant_new() -> *mut AiAssistantHandle {
    let r = catch_unwind(AssertUnwindSafe(|| {
        clear_last_error();
        make_handle(AiAssistant::new())
    }));
    match r {
        Ok(h) => h,
        Err(e) => {
            set_last_error(&format!("panic in ai_assistant_new: {}", panic_message(e)));
            ptr::null_mut()
        }
    }
}

/// Creates a new assistant handle with a custom system prompt.
///
/// Returns `NULL` on null / non-UTF-8 prompt or on caught panic.
///
/// # Safety
///
/// `prompt` must be null or a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_new_with_prompt(
    prompt: *const c_char,
) -> *mut AiAssistantHandle {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let s = match cstr_to_str(prompt) {
            Ok(s) => s,
            Err(E_NULL_PTR) => {
                set_last_error("prompt is null");
                return ptr::null_mut();
            }
            Err(_) => {
                set_last_error("prompt is not valid UTF-8");
                return ptr::null_mut();
            }
        };
        clear_last_error();
        make_handle(AiAssistant::with_system_prompt(s))
    }));
    match r {
        Ok(p) => p,
        Err(e) => {
            set_last_error(&format!(
                "panic in ai_assistant_new_with_prompt: {}",
                panic_message(e)
            ));
            ptr::null_mut()
        }
    }
}

/// Releases an assistant handle. **Null-safe.**
///
/// After this call the C caller must NOT dereference the pointer and
/// should set its local to NULL to avoid accidental double-free.
///
/// # Safety
///
/// `handle` must either be null or a pointer produced by
/// `ai_assistant_new*` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_free(handle: *mut AiAssistantHandle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: non-null + from Box::into_raw per the doc comment.
        drop(Box::from_raw(handle as *mut Inner));
    }));
}

// ---------------------------------------------------------------------------
// Configuration setters
// ---------------------------------------------------------------------------

/// Updates the base system prompt.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_system_prompt(
    handle: *mut AiAssistantHandle,
    prompt: *const c_char,
) -> c_int {
    guard("ai_assistant_set_system_prompt", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(prompt) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "prompt is null"
                } else {
                    "prompt is not valid UTF-8"
                });
                return rc;
            }
        };
        // SAFETY: single-thread contract — no aliasing borrow exists.
        let a = &mut *inner.assistant.get();
        a.set_system_prompt(s);
        clear_last_error();
        OK
    })
}

/// Selects the provider. For `OpenAICompatible` and `Bedrock`, the
/// companion setter MUST also be called before the next `send_message`.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_provider(
    handle: *mut AiAssistantHandle,
    kind: AiProviderKind,
) -> c_int {
    guard("ai_assistant_set_provider", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        inner.provider_kind.set(kind);
        clear_last_error();
        OK
    })
}

/// Sets the base URL for `OpenAICompatible`. Order-independent vs.
/// `ai_assistant_set_provider`.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_openai_compatible_url(
    handle: *mut AiAssistantHandle,
    url: *const c_char,
) -> c_int {
    guard("ai_assistant_set_openai_compatible_url", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(url) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "url is null"
                } else {
                    "url is not valid UTF-8"
                });
                return rc;
            }
        };
        *inner.openai_compatible_url.borrow_mut() = Some(s.to_string());
        clear_last_error();
        OK
    })
}

/// Sets the AWS region for `Bedrock`. Order-independent vs.
/// `ai_assistant_set_provider`.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_bedrock_region(
    handle: *mut AiAssistantHandle,
    region: *const c_char,
) -> c_int {
    guard("ai_assistant_set_bedrock_region", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(region) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "region is null"
                } else {
                    "region is not valid UTF-8"
                });
                return rc;
            }
        };
        *inner.bedrock_region.borrow_mut() = Some(s.to_string());
        clear_last_error();
        OK
    })
}

/// Sets the selected model name.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_model(
    handle: *mut AiAssistantHandle,
    model: *const c_char,
) -> c_int {
    guard("ai_assistant_set_model", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(model) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "model is null"
                } else {
                    "model is not valid UTF-8"
                });
                return rc;
            }
        };
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.selected_model = s.to_string();
        clear_last_error();
        OK
    })
}

/// Sets the API key (empty string clears it).
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_api_key(
    handle: *mut AiAssistantHandle,
    key: *const c_char,
) -> c_int {
    guard("ai_assistant_set_api_key", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(key) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "api key is null"
                } else {
                    "api key is not valid UTF-8"
                });
                return rc;
            }
        };
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.api_key = s.to_string();
        clear_last_error();
        OK
    })
}

/// Overrides the Ollama base URL (default `http://localhost:11434`).
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_ollama_url(
    handle: *mut AiAssistantHandle,
    url: *const c_char,
) -> c_int {
    guard("ai_assistant_set_ollama_url", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let s = match cstr_to_str(url) {
            Ok(s) => s,
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "url is null"
                } else {
                    "url is not valid UTF-8"
                });
                return rc;
            }
        };
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.ollama_url = s.to_string();
        clear_last_error();
        OK
    })
}

/// Sets the sampling temperature. **Strict reject** on NaN, ±Inf, and
/// any finite value outside `[0.0, 2.0]`. Clamping is intentionally not
/// performed — it hides caller bugs.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_temperature(
    handle: *mut AiAssistantHandle,
    temperature: f32,
) -> c_int {
    guard("ai_assistant_set_temperature", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        if !temperature.is_finite() {
            set_last_error("temperature must be finite");
            return E_INTERNAL;
        }
        if !(0.0..=2.0).contains(&temperature) {
            set_last_error("temperature must be in [0.0, 2.0]");
            return E_INTERNAL;
        }
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.temperature = temperature;
        clear_last_error();
        OK
    })
}

/// Sets the conversation history retention cap.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_set_max_history(
    handle: *mut AiAssistantHandle,
    max_history: usize,
) -> c_int {
    guard("ai_assistant_set_max_history", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.max_history_messages = max_history;
        clear_last_error();
        OK
    })
}

// ---------------------------------------------------------------------------
// Messaging
// ---------------------------------------------------------------------------

/// Sends a user message and blocks until the full response is available.
///
/// On success, `*out` is set to a Rust-allocated NUL-terminated UTF-8
/// string. The caller **must** free it via `ai_assistant_free_string`.
///
/// When built with `--features "ffi,rag"`, the call automatically
/// dispatches to `generate_sync_with_rag` (which builds context from the
/// indexed corpus). Otherwise the plain `generate_sync(msg, "")` path is
/// used — the core blocking primitive, not RAG-aware.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_send_message(
    handle: *mut AiAssistantHandle,
    prompt: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard("ai_assistant_send_message", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        if out.is_null() {
            set_last_error("out pointer is null");
            return E_NULL_PTR;
        }
        *out = ptr::null_mut();
        let msg = match cstr_to_str(prompt) {
            Ok(s) => s.to_string(),
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "prompt is null"
                } else {
                    "prompt is not valid UTF-8"
                });
                return rc;
            }
        };
        let provider = match build_provider(inner) {
            Ok(p) => p,
            Err(rc) => return rc,
        };
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.provider = provider;

        #[cfg(feature = "rag")]
        let result = a.generate_sync_with_rag(msg);
        #[cfg(not(feature = "rag"))]
        let result = a.generate_sync(msg, "");

        match result {
            Ok(text) => match CString::new(text) {
                Ok(c) => {
                    *out = c.into_raw();
                    clear_last_error();
                    OK
                }
                Err(_) => {
                    set_last_error("response contained interior NUL");
                    E_INTERNAL
                }
            },
            Err(e) => {
                set_last_error(&format!("send failed: {e}"));
                E_SEND_FAILED
            }
        }
    })
}

/// Sends a user message and dispatches partial chunks to `callback` as
/// they arrive. Blocks until the terminal event has been delivered.
///
/// The `callback` parameter is a C function pointer of signature:
/// `void (*)(const char* chunk, bool is_final, void* user_data)`.
/// It **must not be NULL** — Rust function pointer types are non-nullable.
///
/// `chunk` passed to the callback is a borrowed NUL-terminated UTF-8
/// string valid only for the duration of the callback invocation — copy
/// it if you need to keep it. `is_final = true` indicates the terminal
/// event (after which no further callbacks will be dispatched for this
/// call).
///
/// Re-entrancy into any other `ai_assistant_*` function from within the
/// callback on the same handle is **undefined behavior**.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_send_message_stream(
    handle: *mut AiAssistantHandle,
    prompt: *const c_char,
    callback: unsafe extern "C" fn(
        chunk: *const c_char,
        is_final: bool,
        user_data: *mut std::os::raw::c_void,
    ),
    user_data: *mut std::os::raw::c_void,
) -> c_int {
    guard("ai_assistant_send_message_stream", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        let cb = callback;
        let msg = match cstr_to_str(prompt) {
            Ok(s) => s.to_string(),
            Err(rc) => {
                set_last_error(if rc == E_NULL_PTR {
                    "prompt is null"
                } else {
                    "prompt is not valid UTF-8"
                });
                return rc;
            }
        };
        let provider = match build_provider(inner) {
            Ok(p) => p,
            Err(rc) => return rc,
        };
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.config.provider = provider;
        a.send_message_simple(msg);

        // Dispatch helper — converts chunk to CString and invokes cb.
        let dispatch = |text: &str, is_final: bool| -> bool {
            match CString::new(text) {
                Ok(c) => {
                    cb(c.as_ptr(), is_final, user_data);
                    true
                }
                Err(_) => false,
            }
        };

        let mut got_any = false;
        let mut iterations: u32 = 0;
        const MAX_ITERATIONS: u32 = 60_000; // ~10 min at 10 ms/poll
        loop {
            if iterations >= MAX_ITERATIONS {
                let _ = dispatch("stream poll limit reached", true);
                set_last_error("stream poll limit reached");
                return E_SEND_FAILED;
            }
            iterations += 1;
            match a.poll_response() {
                Some(AiResponse::Chunk(s)) => {
                    got_any = true;
                    let _ = dispatch(&s, false);
                }
                Some(AiResponse::Complete(s)) => {
                    let _ = dispatch(&s, true);
                    clear_last_error();
                    return OK;
                }
                Some(AiResponse::Cancelled(s)) => {
                    let _ = dispatch(&s, true);
                    set_last_error("stream cancelled");
                    return E_SEND_FAILED;
                }
                Some(AiResponse::Error(e)) => {
                    let _ = dispatch(&e, true);
                    set_last_error(&format!("stream error: {e}"));
                    return E_SEND_FAILED;
                }
                Some(_) => {
                    // ModelsLoaded or other non-terminal non-chunk — skip.
                }
                None => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            if !got_any && iterations > 1 {
                // Still waiting; continue polling.
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Clears the current conversation history.
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_clear_conversation(handle: *mut AiAssistantHandle) -> c_int {
    guard("ai_assistant_clear_conversation", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.clear_conversation();
        clear_last_error();
        OK
    })
}

/// Starts a fresh session (new conversation + session store entry).
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_new_session(handle: *mut AiAssistantHandle) -> c_int {
    guard("ai_assistant_new_session", || {
        let Some(inner) = handle_to_inner(handle) else {
            set_last_error("handle is null");
            return E_NULL_PTR;
        };
        check_thread(inner);
        // SAFETY: single-thread contract.
        let a = &mut *inner.assistant.get();
        a.new_session();
        clear_last_error();
        OK
    })
}

// ---------------------------------------------------------------------------
// Memory & diagnostics
// ---------------------------------------------------------------------------

/// Frees a string previously produced by `ai_assistant_send_message`.
/// **Null-safe.**
///
/// # Safety
///
/// `s` must either be null or a pointer produced by
/// `CString::into_raw` (i.e. the `*out` of `ai_assistant_send_message`).
#[no_mangle]
pub unsafe extern "C" fn ai_assistant_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(CString::from_raw(s));
    }));
}

/// Returns a borrowed pointer to the last error message on the current
/// thread, or NULL if there is no pending error.
///
/// The returned pointer is valid until the next FFI call on this thread.
/// The C caller must **not** free it.
#[no_mangle]
pub extern "C" fn ai_assistant_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

/// Returns the library version as a static NUL-terminated C string
/// ("0.2.11"). The C caller must **not** free it.
#[no_mangle]
pub extern "C" fn ai_assistant_version() -> *const c_char {
    // Static version string with trailing NUL.
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const c_char
}

/// Returns the FFI ABI version. Currently `1`. Bumps on breaking
/// changes to any of the `extern "C"` signatures or the return-code enum.
#[no_mangle]
pub extern "C" fn ai_assistant_abi_version() -> c_int {
    1
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn make_cstr(s: &str) -> CString {
        CString::new(s).expect("test input contains NUL")
    }

    #[test]
    fn test_new_and_free_roundtrip() {
        let h = ai_assistant_new();
        assert!(!h.is_null());
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_free_null_is_safe() {
        unsafe { ai_assistant_free(ptr::null_mut()) };
    }

    #[test]
    fn test_new_with_prompt_ok() {
        let p = make_cstr("you are a test bot");
        let h = unsafe { ai_assistant_new_with_prompt(p.as_ptr()) };
        assert!(!h.is_null());
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_new_with_prompt_null_returns_null_and_sets_error() {
        let h = unsafe { ai_assistant_new_with_prompt(ptr::null()) };
        assert!(h.is_null());
        let err = ai_assistant_last_error();
        assert!(!err.is_null());
        let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy();
        assert!(msg.contains("null"));
    }

    #[test]
    fn test_set_system_prompt_happy_path() {
        let h = ai_assistant_new();
        let p = make_cstr("hello");
        let rc = unsafe { ai_assistant_set_system_prompt(h, p.as_ptr()) };
        assert_eq!(rc, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_system_prompt_null_prompt_returns_err() {
        let h = ai_assistant_new();
        let rc = unsafe { ai_assistant_set_system_prompt(h, ptr::null()) };
        assert_eq!(rc, E_NULL_PTR);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_system_prompt_invalid_utf8() {
        let h = ai_assistant_new();
        // 0xFF is not valid UTF-8; build a NUL-terminated byte buffer
        // manually so CString's NUL check doesn't reject it.
        let bad: [u8; 3] = [0xFF, 0xFE, 0x00];
        let rc = unsafe { ai_assistant_set_system_prompt(h, bad.as_ptr() as *const c_char) };
        assert_eq!(rc, E_INVALID_UTF8);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_provider_every_unit_variant() {
        let h = ai_assistant_new();
        let unit_variants = [
            AiProviderKind::Ollama,
            AiProviderKind::LMStudio,
            AiProviderKind::TextGenWebUI,
            AiProviderKind::KoboldCpp,
            AiProviderKind::LocalAI,
            AiProviderKind::OpenAI,
            AiProviderKind::Anthropic,
            AiProviderKind::Gemini,
            AiProviderKind::Groq,
            AiProviderKind::Together,
            AiProviderKind::Fireworks,
            AiProviderKind::DeepSeek,
            AiProviderKind::Mistral,
            AiProviderKind::Perplexity,
            AiProviderKind::OpenRouter,
        ];
        assert_eq!(unit_variants.len(), 15);
        for k in unit_variants {
            let rc = unsafe { ai_assistant_set_provider(h, k) };
            assert_eq!(rc, OK, "provider {:?} failed", k);
        }
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_provider_openai_compatible_orderings() {
        // Order 1: setter first, then provider — build_provider OK.
        let h = ai_assistant_new();
        let url = make_cstr("http://localhost:1234/v1");
        assert_eq!(
            unsafe { ai_assistant_set_openai_compatible_url(h, url.as_ptr()) },
            OK
        );
        assert_eq!(
            unsafe { ai_assistant_set_provider(h, AiProviderKind::OpenAICompatible) },
            OK
        );
        let inner: &Inner = unsafe { &*(h as *const Inner) };
        assert!(build_provider(inner).is_ok());
        unsafe { ai_assistant_free(h) };

        // Order 2: provider first, then setter — also OK (lazy build).
        let h = ai_assistant_new();
        assert_eq!(
            unsafe { ai_assistant_set_provider(h, AiProviderKind::OpenAICompatible) },
            OK
        );
        assert_eq!(
            unsafe { ai_assistant_set_openai_compatible_url(h, url.as_ptr()) },
            OK
        );
        let inner: &Inner = unsafe { &*(h as *const Inner) };
        assert!(build_provider(inner).is_ok());
        unsafe { ai_assistant_free(h) };

        // Order 3: provider without setter — build_provider fails.
        let h = ai_assistant_new();
        assert_eq!(
            unsafe { ai_assistant_set_provider(h, AiProviderKind::OpenAICompatible) },
            OK
        );
        let inner: &Inner = unsafe { &*(h as *const Inner) };
        assert_eq!(build_provider(inner).unwrap_err(), E_UNKNOWN_PROVIDER);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_provider_bedrock_orderings() {
        let region = make_cstr("us-east-1");

        let h = ai_assistant_new();
        assert_eq!(
            unsafe { ai_assistant_set_bedrock_region(h, region.as_ptr()) },
            OK
        );
        assert_eq!(
            unsafe { ai_assistant_set_provider(h, AiProviderKind::Bedrock) },
            OK
        );
        let inner: &Inner = unsafe { &*(h as *const Inner) };
        assert!(build_provider(inner).is_ok());
        unsafe { ai_assistant_free(h) };

        let h = ai_assistant_new();
        assert_eq!(
            unsafe { ai_assistant_set_provider(h, AiProviderKind::Bedrock) },
            OK
        );
        let inner: &Inner = unsafe { &*(h as *const Inner) };
        assert_eq!(build_provider(inner).unwrap_err(), E_UNKNOWN_PROVIDER);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_model_happy_path() {
        let h = ai_assistant_new();
        let m = make_cstr("llama3.2:3b");
        assert_eq!(unsafe { ai_assistant_set_model(h, m.as_ptr()) }, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_api_key_happy_path() {
        let h = ai_assistant_new();
        let k = make_cstr("sk-test-key-not-real");
        assert_eq!(unsafe { ai_assistant_set_api_key(h, k.as_ptr()) }, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_temperature_valid_range() {
        let h = ai_assistant_new();
        for t in [0.0f32, 0.7, 1.0, 2.0] {
            assert_eq!(
                unsafe { ai_assistant_set_temperature(h, t) },
                OK,
                "temp {t} should be accepted"
            );
        }
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_temperature_rejects_nan() {
        let h = ai_assistant_new();
        let rc = unsafe { ai_assistant_set_temperature(h, f32::NAN) };
        assert_eq!(rc, E_INTERNAL);
        let err = ai_assistant_last_error();
        assert!(!err.is_null());
        let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy();
        assert!(msg.contains("finite"));
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_temperature_rejects_out_of_range() {
        let h = ai_assistant_new();
        for bad in [-0.1f32, 2.5, f32::INFINITY, f32::NEG_INFINITY] {
            let rc = unsafe { ai_assistant_set_temperature(h, bad) };
            assert_eq!(rc, E_INTERNAL, "temp {bad} should be rejected");
        }
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_set_max_history_happy_path() {
        let h = ai_assistant_new();
        assert_eq!(unsafe { ai_assistant_set_max_history(h, 42) }, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_clear_conversation_happy_path() {
        let h = ai_assistant_new();
        assert_eq!(unsafe { ai_assistant_clear_conversation(h) }, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_new_session_happy_path() {
        let h = ai_assistant_new();
        assert_eq!(unsafe { ai_assistant_new_session(h) }, OK);
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_free_string_null_is_safe() {
        unsafe { ai_assistant_free_string(ptr::null_mut()) };
    }

    #[test]
    fn test_last_error_null_after_success() {
        let h = ai_assistant_new();
        // Trigger an error.
        let _ = unsafe { ai_assistant_set_system_prompt(h, ptr::null()) };
        assert!(!ai_assistant_last_error().is_null());
        // Now succeed — should clear.
        let p = make_cstr("hi");
        let _ = unsafe { ai_assistant_set_system_prompt(h, p.as_ptr()) };
        assert!(ai_assistant_last_error().is_null());
        unsafe { ai_assistant_free(h) };
    }

    #[test]
    fn test_last_error_thread_local_isolation() {
        use std::sync::{Arc, Barrier};

        let barrier = Arc::new(Barrier::new(2));
        let b1 = barrier.clone();
        let t1 = std::thread::spawn(move || {
            let h = ai_assistant_new();
            let _ = unsafe { ai_assistant_set_system_prompt(h, ptr::null()) };
            b1.wait();
            // Other thread sets its own error here.
            b1.wait();
            // Our error slot must still contain OUR message.
            let err = ai_assistant_last_error();
            assert!(!err.is_null());
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { ai_assistant_free(h) };
            msg
        });
        let b2 = barrier.clone();
        let t2 = std::thread::spawn(move || {
            let h = ai_assistant_new();
            b2.wait();
            let bad: [u8; 2] = [0xFF, 0x00];
            let _ = unsafe { ai_assistant_set_system_prompt(h, bad.as_ptr() as *const c_char) };
            b2.wait();
            let err = ai_assistant_last_error();
            assert!(!err.is_null());
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { ai_assistant_free(h) };
            msg
        });
        let m1 = t1.join().unwrap();
        let m2 = t2.join().unwrap();
        assert!(m1.contains("null"), "t1 saw: {m1}");
        assert!(m2.contains("UTF-8") || m2.contains("utf"), "t2 saw: {m2}");
    }

    #[test]
    fn test_abi_version_is_one() {
        assert_eq!(ai_assistant_abi_version(), 1);
    }

    #[test]
    fn test_version_returns_static_nonnull() {
        let p = ai_assistant_version();
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
        assert!(!s.is_empty());
        // Sanity: looks like semver.
        assert!(s.contains('.'), "version: {s}");
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_wrong_thread_use_panics() {
        let h = ai_assistant_new();
        // Pin on this thread.
        let p = make_cstr("pin");
        let rc = unsafe { ai_assistant_set_system_prompt(h, p.as_ptr()) };
        assert_eq!(rc, OK);
        // Raw pointer is Send-able (we opt in via unsafe impl Send for Inner).
        let h_addr = h as usize;
        let other = std::thread::spawn(move || {
            let h = h_addr as *mut AiAssistantHandle;
            let q = CString::new("wrong").unwrap();
            unsafe { ai_assistant_set_system_prompt(h, q.as_ptr()) }
        });
        let rc = other.join().expect("thread joined");
        assert_eq!(rc, E_PANIC, "wrong-thread use should return panic code");
        let err = ai_assistant_last_error();
        // Last-error is thread-local, so on THIS thread it should still
        // reflect the successful pin (i.e. be null). The panic message
        // was stashed on the other thread's local.
        assert!(err.is_null());
        unsafe { ai_assistant_free(h) };
    }

    // --- Live smoke tests (ignored by default) -----------------------------

    /// Requires a running Ollama with at least one pulled model.
    #[test]
    #[ignore]
    fn test_send_message_live_ollama() {
        let h = ai_assistant_new();
        let prompt = make_cstr("Say 'ok' and nothing else.");
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { ai_assistant_send_message(h, prompt.as_ptr(), &mut out) };
        assert_eq!(rc, OK);
        assert!(!out.is_null());
        let text = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        println!("live ollama reply: {text}");
        unsafe { ai_assistant_free_string(out) };
        unsafe { ai_assistant_free(h) };
    }

    /// Requires a running Ollama with at least one pulled model.
    #[test]
    #[ignore]
    fn test_send_message_stream_live_ollama() {
        use std::os::raw::c_void;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CHUNKS: AtomicUsize = AtomicUsize::new(0);
        static FINAL: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "C" fn cb(_chunk: *const c_char, is_final: bool, _user_data: *mut c_void) {
            if is_final {
                FINAL.fetch_add(1, Ordering::SeqCst);
            } else {
                CHUNKS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let h = ai_assistant_new();
        let prompt = make_cstr("Count to three.");
        let rc =
            unsafe { ai_assistant_send_message_stream(h, prompt.as_ptr(), cb, ptr::null_mut()) };
        assert_eq!(rc, OK);
        assert!(FINAL.load(Ordering::SeqCst) >= 1);
        unsafe { ai_assistant_free(h) };
    }

    /// Double-free is UB and not runtime-detectable — documented only.
    #[test]
    #[ignore]
    fn test_double_free_is_documented_not_safe() {
        // Intentionally never runs: would be UB.
    }
}
