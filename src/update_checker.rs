// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Lightweight GitHub release update checker.
//!
//! Checks the latest release tag from the GitHub API and compares it
//! with the current binary version. Designed to run in a background
//! thread so it never blocks startup.

use std::sync::mpsc;
use std::time::Duration;

/// Information about an available update.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    /// Currently running version (e.g. "0.1.1").
    pub current: String,
    /// Latest release version on GitHub (e.g. "0.2.0").
    pub latest: String,
    /// URL to the release page.
    pub url: String,
}

const GITHUB_API_URL: &str =
    "https://api.github.com/repos/OrlandoLuque/ai_assistant/releases/latest";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(4);

/// Check GitHub releases for a newer version (blocking, with timeout).
///
/// Returns `Some(UpdateInfo)` if a newer version exists, `None` otherwise.
/// Silently returns `None` on any network or parsing error.
pub fn check_for_update(current_version: &str) -> Option<UpdateInfo> {
    let resp = ureq::get(GITHUB_API_URL)
        .set("User-Agent", "ai_assistant-update-checker")
        .timeout(REQUEST_TIMEOUT)
        .call()
        .ok()?;

    let body: serde_json::Value = resp.into_json().ok()?;
    let tag = body["tag_name"].as_str()?;
    let url = body["html_url"].as_str()?;

    let latest = tag.strip_prefix('v').unwrap_or(tag);

    if is_newer(latest, current_version) {
        Some(UpdateInfo {
            current: current_version.to_string(),
            latest: latest.to_string(),
            url: url.to_string(),
        })
    } else {
        None
    }
}

/// Spawn a background thread that checks for updates.
///
/// Returns a receiver that will contain an [`UpdateInfo`] if a newer
/// version is found. The receiver can be polled with `try_recv()`.
pub fn check_for_update_bg(current_version: &str) -> mpsc::Receiver<UpdateInfo> {
    let (tx, rx) = mpsc::channel();
    let version = current_version.to_string();
    std::thread::spawn(move || {
        if let Some(info) = check_for_update(&version) {
            let _ = tx.send(info);
        }
    });
    rx
}

/// Simple semver comparison: returns `true` if `latest` is strictly newer than `current`.
fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |s: &str| -> (u32, u32, u32) {
        let parts: Vec<u32> = s.split('.').filter_map(|p| p.parse().ok()).collect();
        (
            parts.first().copied().unwrap_or(0),
            parts.get(1).copied().unwrap_or(0),
            parts.get(2).copied().unwrap_or(0),
        )
    };
    parse(latest) > parse(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.2.0", "0.1.1"));
        assert!(is_newer("0.1.2", "0.1.1"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.1.1", "0.1.1"));
        assert!(!is_newer("0.1.0", "0.1.1"));
        assert!(!is_newer("0.0.9", "0.1.0"));
    }

    #[test]
    fn test_is_newer_partial_versions() {
        assert!(is_newer("1.0", "0.9.9"));
        assert!(is_newer("2", "1.5.3"));
        assert!(!is_newer("0.1", "0.1.0"));
    }
}
