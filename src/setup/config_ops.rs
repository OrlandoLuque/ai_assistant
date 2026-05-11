// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Configuration file operations: show, get, set, diff, export, import.
//!
//! Works with the project's TOML/JSON configuration files and provides
//! redaction of secrets, hot-reload awareness, and cross-format export.

use std::collections::BTreeMap;
use std::path::Path;

use crate::config_file::{ConfigFile, ConfigFormat};

/// Result of setting a config value.
#[derive(Debug, Clone)]
pub struct SetResult {
    /// Previous value (empty if the key was new).
    pub old_value: String,
    /// New value that was written.
    pub new_value: String,
    /// Whether the change requires a restart to take effect.
    pub needs_restart: bool,
}

/// A single difference between two config files.
#[derive(Debug, Clone)]
pub struct ConfigDiff {
    /// TOML section (e.g. "provider", "generation").
    pub section: String,
    /// Key within the section.
    pub key: String,
    /// Value in file A (empty if absent).
    pub value_a: String,
    /// Value in file B (empty if absent).
    pub value_b: String,
}

// Keys whose changes can be applied without restart (hot-reloadable).
const HOT_RELOAD_KEYS: &[&str] = &[
    "generation.temperature",
    "generation.max_tokens",
    "generation.top_p",
    "generation.max_history",
    "logging.level",
    "cache.enabled",
    "cache.max_entries",
];

// Keys that contain secrets and should be redacted.
const SECRET_KEYS: &[&str] = &["api_key", "secret", "password", "token"];

/// Load and display a configuration file with optional secret redaction.
///
/// Returns a human-readable formatted string of the configuration.
pub fn show_config(path: &Path, redact: bool) -> Result<String, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    if redact {
        Ok(redact_secrets(&content))
    } else {
        Ok(content)
    }
}

/// Read a specific dotted key from a config file (e.g. "provider.model").
pub fn get_config_value(path: &Path, key: &str) -> Result<String, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let pairs = parse_flat_pairs(&content);
    pairs
        .get(key)
        .cloned()
        .ok_or_else(|| format!("Key '{}' not found in {}", key, path.display()))
}

/// Set a specific dotted key in a config file.
///
/// The file is read, the key is updated (or added), and the file is rewritten.
/// Returns a [`SetResult`] indicating the old value and whether a restart is needed.
pub fn set_config_value(path: &Path, key: &str, value: &str) -> Result<SetResult, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut pairs = parse_flat_pairs(&content);
    let old_value = pairs.get(key).cloned().unwrap_or_default();

    pairs.insert(key.to_string(), value.to_string());

    // Rebuild a minimal TOML-like file
    let new_content = rebuild_config(&content, key, value);
    std::fs::write(path, &new_content)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;

    let needs_restart = !HOT_RELOAD_KEYS.contains(&key);

    Ok(SetResult {
        old_value,
        new_value: value.to_string(),
        needs_restart,
    })
}

/// Compare two config files and return the differences.
pub fn diff_configs(path_a: &Path, path_b: &Path) -> Vec<ConfigDiff> {
    let pairs_a = std::fs::read_to_string(path_a)
        .map(|c| parse_flat_pairs(&c))
        .unwrap_or_default();
    let pairs_b = std::fs::read_to_string(path_b)
        .map(|c| parse_flat_pairs(&c))
        .unwrap_or_default();

    let mut diffs = Vec::new();

    // All keys from both sides
    let mut all_keys: Vec<String> = pairs_a.keys().chain(pairs_b.keys()).cloned().collect();
    all_keys.sort();
    all_keys.dedup();

    for key in &all_keys {
        let va = pairs_a.get(key.as_str()).cloned().unwrap_or_default();
        let vb = pairs_b.get(key.as_str()).cloned().unwrap_or_default();

        if va != vb {
            let (section, subkey) = split_key(key);
            diffs.push(ConfigDiff {
                section,
                key: subkey,
                value_a: va,
                value_b: vb,
            });
        }
    }

    diffs
}

/// Export a config file to a different format.
///
/// Supported formats: `"toml"`, `"json"`.
pub fn export_config(path: &Path, format: &str, output: &Path) -> Result<(), String> {
    let config = ConfigFile::load(path).map_err(|e| format!("Failed to load config: {}", e))?;

    let target_format = match format.to_lowercase().as_str() {
        "toml" => ConfigFormat::Toml,
        "json" => ConfigFormat::Json,
        _ => {
            return Err(format!(
                "Unsupported export format: '{}'. Use 'toml' or 'json'",
                format
            ))
        }
    };

    let content = config
        .serialize(target_format)
        .map_err(|e| format!("Serialization failed: {}", e))?;

    std::fs::write(output, content)
        .map_err(|e| format!("Failed to write {}: {}", output.display(), e))?;

    Ok(())
}

/// Import a config file with validation warnings.
///
/// Loads the source, validates it, and writes it to the output path.
/// Returns a list of validation warnings (empty if valid).
pub fn import_config(input: &Path, output: &Path) -> Result<Vec<String>, String> {
    let config = ConfigFile::load(input)
        .map_err(|e| format!("Failed to load {}: {}", input.display(), e))?;

    let mut warnings = Vec::new();

    match config.validate_detailed() {
        Ok(()) => {}
        Err(errors) => {
            for err in &errors {
                warnings.push(format!("{}", err));
            }
        }
    }

    config
        .save(output)
        .map_err(|e| format!("Failed to save {}: {}", output.display(), e))?;

    Ok(warnings)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse a TOML/JSON-like config into flat "section.key" -> "value" pairs.
fn parse_flat_pairs(content: &str) -> BTreeMap<String, String> {
    let mut pairs = BTreeMap::new();
    let mut current_section = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        // TOML section header
        if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.contains("[[") {
            current_section = trimmed[1..trimmed.len() - 1].trim().to_string();
            continue;
        }

        // Key = value
        if let Some(eq_pos) = trimmed.find('=') {
            let key = trimmed[..eq_pos].trim().to_string();
            let val = trimmed[eq_pos + 1..].trim().to_string();
            // Strip surrounding quotes
            let val = val
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(&val)
                .to_string();

            let full_key = if current_section.is_empty() {
                key
            } else {
                format!("{}.{}", current_section, key)
            };
            pairs.insert(full_key, val);
        }
    }

    pairs
}

/// Rebuild config content after setting a key.
fn rebuild_config(original: &str, changed_key: &str, new_value: &str) -> String {
    let (section, bare_key) = split_key(changed_key);

    let mut result = String::new();
    let mut found = false;
    let mut in_target_section = section.is_empty();

    for line in original.lines() {
        let trimmed = line.trim();

        // Track section
        if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.contains("[[") {
            let sec = trimmed[1..trimmed.len() - 1].trim();
            in_target_section = sec == section;
        }

        // Replace the matching key
        if in_target_section && !found {
            if let Some(eq_pos) = trimmed.find('=') {
                let key = trimmed[..eq_pos].trim();
                if key == bare_key {
                    // Preserve leading whitespace
                    let indent = line.len() - line.trim_start().len();
                    let prefix = &line[..indent];
                    result.push_str(&format!("{}{} = \"{}\"\n", prefix, bare_key, new_value));
                    found = true;
                    continue;
                }
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    // If the key wasn't found, append it
    if !found {
        if !section.is_empty() {
            // Check if section exists
            let section_header = format!("[{}]", section);
            if !original.contains(&section_header) {
                result.push_str(&format!("\n{}\n", section_header));
            }
        }
        result.push_str(&format!("{} = \"{}\"\n", bare_key, new_value));
    }

    result
}

/// Split "section.key" into ("section", "key"). If no dot, section is "".
fn split_key(key: &str) -> (String, String) {
    if let Some(dot) = key.rfind('.') {
        (key[..dot].to_string(), key[dot + 1..].to_string())
    } else {
        (String::new(), key.to_string())
    }
}

/// Redact secret values in a config string.
fn redact_secrets(content: &str) -> String {
    let mut result = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        let is_secret = SECRET_KEYS.iter().any(|k| {
            trimmed
                .split('=')
                .next()
                .map(|lhs| lhs.trim().to_lowercase().contains(k))
                .unwrap_or(false)
        });

        if is_secret && trimmed.contains('=') {
            if let Some(eq_pos) = line.find('=') {
                result.push_str(&line[..eq_pos + 1]);
                result.push_str(" \"***REDACTED***\"");
            } else {
                result.push_str(line);
            }
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_config(content: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_setup_test_{}.toml", uuid::Uuid::new_v4()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_show_config_redact() {
        let cfg = temp_config(
            "[provider]\ntype = \"ollama\"\nmodel = \"llama3\"\napi_key = \"sk-secret123\"\n",
        );
        let output = show_config(&cfg, true).unwrap();
        assert!(
            output.contains("***REDACTED***"),
            "api_key should be redacted"
        );
        assert!(!output.contains("sk-secret123"), "Secret should not appear");
        assert!(output.contains("ollama"), "Non-secret values should remain");
        let _ = std::fs::remove_file(&cfg);
    }

    #[test]
    fn test_get_and_set_config_value() {
        let cfg = temp_config(
            "[provider]\ntype = \"ollama\"\nmodel = \"llama3\"\n\n[generation]\ntemperature = 0.7\n",
        );

        let val = get_config_value(&cfg, "provider.model").unwrap();
        assert_eq!(val, "llama3");

        let result = set_config_value(&cfg, "provider.model", "mistral").unwrap();
        assert_eq!(result.old_value, "llama3");
        assert_eq!(result.new_value, "mistral");
        assert!(result.needs_restart, "provider.model is not hot-reloadable");

        // Verify the change persisted
        let val2 = get_config_value(&cfg, "provider.model").unwrap();
        assert_eq!(val2, "mistral");

        let _ = std::fs::remove_file(&cfg);
    }

    #[test]
    fn test_diff_configs() {
        let cfg_a = temp_config(
            "[provider]\ntype = \"ollama\"\nmodel = \"llama3\"\n\n[generation]\ntemperature = 0.7\n",
        );
        let cfg_b = temp_config(
            "[provider]\ntype = \"ollama\"\nmodel = \"mistral\"\n\n[generation]\ntemperature = 0.9\n",
        );

        let diffs = diff_configs(&cfg_a, &cfg_b);
        assert!(
            diffs.len() >= 2,
            "Should have at least 2 diffs, got {}",
            diffs.len()
        );

        let model_diff = diffs.iter().find(|d| d.key == "model");
        assert!(model_diff.is_some(), "Should find model diff");
        let md = model_diff.unwrap();
        assert_eq!(md.value_a, "llama3");
        assert_eq!(md.value_b, "mistral");

        let _ = std::fs::remove_file(&cfg_a);
        let _ = std::fs::remove_file(&cfg_b);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let cfg = temp_config(
            "[provider]\ntype = \"ollama\"\nmodel = \"llama3\"\n\n[generation]\ntemperature = 0.7\n",
        );
        let json_out = {
            let mut p = std::env::temp_dir();
            p.push(format!("ai_setup_export_{}.json", uuid::Uuid::new_v4()));
            p
        };

        export_config(&cfg, "json", &json_out).unwrap();
        let json_content = std::fs::read_to_string(&json_out).unwrap();
        assert!(
            json_content.contains("ollama"),
            "JSON export should contain provider type"
        );

        // Import back
        let reimport = {
            let mut p = std::env::temp_dir();
            p.push(format!("ai_setup_reimport_{}.json", uuid::Uuid::new_v4()));
            p
        };
        let _warnings = import_config(&json_out, &reimport).unwrap();
        // Warnings may or may not be empty depending on defaults — just check it succeeded

        let _ = std::fs::remove_file(&cfg);
        let _ = std::fs::remove_file(&json_out);
        let _ = std::fs::remove_file(&reimport);
    }
}
