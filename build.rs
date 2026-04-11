// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_assistant — build script.
//!
//! Two responsibilities:
//!
//! 1. **Windows icon embedding** (always): embeds per-binary `.ico` resources
//!    into each Windows executable. No-op on non-Windows targets.
//!
//! 2. **V79 FFI header generation** (only when `--features ffi`): runs
//!    `cbindgen` over `src/ffi.rs` to (re)generate `include/ai_assistant.h`.
//!    The generated header is committed to the repo so downstream C/C++/C#
//!    consumers can use it without having to run `cargo` themselves.

fn main() {
    embed_windows_icons();
    generate_ffi_header();
}

/// Embeds per-binary Windows icons into each executable.
/// On non-Windows targets this is a no-op.
fn embed_windows_icons() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Map: icon name → binaries that use it
    let icon_map: &[(&str, &[&str])] = &[
        ("cli", &["ai_cli", "ai_assistant_cli"]),
        ("server", &["ai_assistant_server", "ai_proxy"]),
        ("gui", &["ai_gui", "ai_gui-pro"]),
        ("kpkg", &["kpkg_tool"]),
        ("cluster", &["ai_cluster_node"]),
        ("home", &["ai_test_harness", "ai_assistant_standalone"]),
        ("virtual_mic", &["ai_virtual_mic"]),
    ];

    for (icon_name, bins) in icon_map {
        let ico_path = format!("{}\\assets\\icons\\{}.ico", manifest_dir, icon_name);
        if !std::path::Path::new(&ico_path).exists() {
            continue;
        }
        let rc_path = format!("{}\\{}.rc", out_dir, icon_name);
        let ico_escaped = ico_path.replace('\\', "\\\\");
        std::fs::write(&rc_path, format!("1 ICON \"{}\"", ico_escaped)).unwrap();
        let _ = embed_resource::compile_for(&rc_path, *bins, embed_resource::NONE);
    }
}

/// V79: regenerate `include/ai_assistant.h` from `src/ffi.rs` via cbindgen.
///
/// Fast no-op when the `ffi` feature is off. On FFI builds this (re)writes
/// the header and emits `cargo:warning` diagnostics on any failure instead
/// of panicking — a broken cbindgen run should never block a regular build.
fn generate_ffi_header() {
    // Re-run whenever the FFI feature flag toggles.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FFI");

    // Fast path: nothing to do unless `ffi` is enabled. cargo sets
    // `CARGO_FEATURE_FFI=1` automatically when --features ffi is passed
    // (directly or transitively).
    if std::env::var("CARGO_FEATURE_FFI").is_err() {
        return;
    }

    // Panic-strategy warning: the default `release` profile has
    // `panic = "abort"`, which turns `catch_unwind` into a no-op in
    // src/ffi.rs. FFI consumers almost certainly want `--profile
    // release-fast` (which keeps `panic = "unwind"`). `CARGO_CFG_PANIC` is
    // set by cargo to the active panic strategy and is reliable per the
    // Cargo book.
    if std::env::var("PROFILE").as_deref() == Ok("release")
        && std::env::var("CARGO_CFG_PANIC").as_deref() == Ok("abort")
    {
        println!(
            "cargo:warning=ai_assistant ffi: panic=abort in release; \
             catch_unwind is a no-op, panics abort the process. \
             Prefer --profile release-fast for FFI consumers."
        );
    }

    let crate_dir: std::path::PathBuf = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(e) => {
            println!("cargo:warning=CARGO_MANIFEST_DIR not set: {e}");
            return;
        }
    };
    let include_dir = crate_dir.join("include");
    if let Err(e) = std::fs::create_dir_all(&include_dir) {
        println!("cargo:warning=failed to create include/: {e}");
        return;
    }
    let out = include_dir.join("ai_assistant.h");

    let cfg_path = crate_dir.join("cbindgen.toml");
    let config = match cbindgen::Config::from_file(&cfg_path) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "cargo:warning=failed to read cbindgen.toml at {}: {e}",
                cfg_path.display()
            );
            return;
        }
    };
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(header) => {
            header.write_to_file(&out);
        }
        Err(e) => {
            println!("cargo:warning=cbindgen failed: {e}");
        }
    }

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
