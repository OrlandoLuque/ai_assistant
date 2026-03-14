// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

/// Embeds per-binary Windows icons into each executable.
/// On non-Windows targets this is a no-op.
fn main() {
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
