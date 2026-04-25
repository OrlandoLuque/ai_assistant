//! Layered (8-tier) JSON configuration merge (V104.6)
//!
//! OpenCode-style config loading: build a final `serde_json::Value` by
//! deep-merging up to eight ordered layers. Highest-precedence wins on
//! key collision; the result can be deserialized into any caller type.
//!
//! ## Layer precedence (low → high)
//!
//! 1. **Built-in defaults** (caller-provided JSON object).
//! 2. **System config** — `/etc/ai_assistant/config.json` (or any path
//!    listed in [`LayerLoadConfig::system_files`]).
//! 3. **User config** — `<config-dir>/ai_assistant/config.json` (XDG /
//!    APPDATA / `Library/Application Support`).
//! 4. **Per-machine override** — `<config-dir>/ai_assistant/config.<hostname>.json`
//!    (skip if hostname unknown).
//! 5. **Project ancestor configs** — walk from `project_root` up to filesystem
//!    root, picking the first match of [`LayerLoadConfig::project_filenames`]
//!    in each ancestor. Furthest ancestor has lowest precedence.
//! 6. **Project root config** — same filenames in `project_root` itself
//!    (highest of the file layers).
//! 7. **Environment overrides** — variables matching
//!    `<env_prefix>__<dotted.path>=<value>` are spliced in. Value is
//!    parsed as JSON; if that fails, treated as a string.
//! 8. **Explicit overrides** — caller-provided JSON object (typically
//!    parsed CLI flags). Highest precedence.
//!
//! ## Merge semantics
//!
//! - **Objects**: deep-merge. Higher layer's keys win.
//! - **Arrays**: higher layer **replaces** lower (no concat). Surprising
//!   concat behavior is opt-in via [`LayerLoadConfig::array_strategy`].
//! - **Scalars / null**: higher layer wins.
//!
//! ## Security baseline
//!
//! - Per-layer size cap (default 1 MiB).
//! - Project-ancestor walk depth cap (default 16) — protects against
//!   pathological directory chains.
//! - Symlinks rejected for config files unless `follow_symlinks` is on.
//! - Env vars are NOT eval'd: bad JSON falls back to a literal string.
//! - File reads are best-effort: a missing file is not an error, but a
//!   present-but-invalid file IS surfaced.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

// ============================================================================
// Configuration
// ============================================================================

/// What to do when both a low-precedence and high-precedence layer
/// define the same JSON array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayMergeStrategy {
    /// Higher layer fully replaces lower (default — predictable).
    Replace,
    /// Concatenate lower then higher (use sparingly — order-sensitive).
    Concat,
}

impl Default for ArrayMergeStrategy {
    fn default() -> Self {
        Self::Replace
    }
}

/// Configuration for [`load_layered_config`].
#[derive(Debug, Clone)]
pub struct LayerLoadConfig {
    /// System-wide config files to try (layer 2). Default empty.
    pub system_files: Vec<PathBuf>,
    /// User config file (layer 3). Default `<config-dir>/ai_assistant/config.json`.
    pub user_file: Option<PathBuf>,
    /// Per-machine override file (layer 4). Default
    /// `<config-dir>/ai_assistant/config.<hostname>.json`.
    pub machine_file: Option<PathBuf>,
    /// Project root for layers 5/6. If `None`, those layers are skipped.
    pub project_root: Option<PathBuf>,
    /// Filenames to probe in each ancestor directory.
    /// Default: `[".ai_assistant.json", ".ai_assistant/config.json"]`.
    pub project_filenames: Vec<String>,
    /// Maximum ancestors walked above `project_root`. Default 16.
    pub max_ancestor_walk: usize,
    /// Env-var prefix for layer 7. Default `"AI_ASSISTANT"`.
    /// Vars must look like `AI_ASSISTANT__a__b__c=value` →
    /// `{ "a": { "b": { "c": value } } }`.
    pub env_prefix: String,
    /// Whether to follow symlinks when reading config files. Default false.
    pub follow_symlinks: bool,
    /// Maximum bytes per layer file. Default 1 MiB.
    pub max_layer_size: u64,
    /// Strategy for merging arrays. Default `Replace`.
    pub array_strategy: ArrayMergeStrategy,
    /// If true, layer 7 reads from `std::env`. If false, only the
    /// `env_overrides` map below is used (good for tests).
    pub read_process_env: bool,
    /// Pre-supplied env var map (used when `read_process_env` is false,
    /// or merged on top of `std::env` when true).
    pub env_overrides: HashMap<String, String>,
}

impl Default for LayerLoadConfig {
    fn default() -> Self {
        Self {
            system_files: vec![],
            user_file: None,
            machine_file: None,
            project_root: None,
            project_filenames: vec![
                ".ai_assistant.json".into(),
                ".ai_assistant/config.json".into(),
            ],
            max_ancestor_walk: 16,
            env_prefix: "AI_ASSISTANT".into(),
            follow_symlinks: false,
            max_layer_size: 1024 * 1024,
            array_strategy: ArrayMergeStrategy::Replace,
            read_process_env: true,
            env_overrides: HashMap::new(),
        }
    }
}

// ============================================================================
// Result types
// ============================================================================

/// One step in the merge process — for debugging / `explain` output.
#[derive(Debug, Clone)]
pub struct LayerStep {
    /// Human-readable layer label (e.g. `"system: /etc/ai_assistant/config.json"`).
    pub label: String,
    /// Was a value actually loaded for this layer?
    pub loaded: bool,
    /// Number of top-level keys this layer contributed (object) or 0
    /// (scalar/array/missing).
    pub top_level_keys: usize,
    /// If loading failed (file present but invalid), the error message.
    pub error: Option<String>,
}

/// Successful result of [`load_layered_config`].
#[derive(Debug, Clone)]
pub struct LayeredConfig {
    /// Final merged JSON value (typically an object).
    pub merged: Value,
    /// Per-layer trace (in precedence order: low → high).
    pub trace: Vec<LayerStep>,
}

impl LayeredConfig {
    /// Try to deserialize the merged value into a strongly-typed config.
    pub fn deserialize<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.merged.clone())
    }
}

/// Errors raised by the layered loader. A *missing* file is not an error;
/// only present-but-invalid layers surface here.
#[derive(Debug)]
pub enum LayeredError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    InvalidJson {
        path: PathBuf,
        message: String,
    },
    LayerTooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    NotRegularFile(PathBuf),
    SymlinkRejected(PathBuf),
}

impl std::fmt::Display for LayeredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "I/O error reading {}: {}", path.display(), source)
            }
            Self::InvalidJson { path, message } => {
                write!(f, "invalid JSON in {}: {}", path.display(), message)
            }
            Self::LayerTooLarge { path, size, limit } => write!(
                f,
                "config layer too large: {} ({} > {} bytes)",
                path.display(),
                size,
                limit
            ),
            Self::NotRegularFile(p) => {
                write!(f, "config layer is not a regular file: {}", p.display())
            }
            Self::SymlinkRejected(p) => {
                write!(f, "config layer is a symlink (rejected): {}", p.display())
            }
        }
    }
}

impl std::error::Error for LayeredError {}

// ============================================================================
// Public entry point
// ============================================================================

/// Build the layered configuration.
///
/// `defaults` is layer 1; `cli_overrides` (if any) is layer 8.
pub fn load_layered_config(
    defaults: Value,
    cli_overrides: Option<Value>,
    cfg: &LayerLoadConfig,
) -> Result<LayeredConfig, LayeredError> {
    let mut merged = defaults.clone();
    let mut trace: Vec<LayerStep> = Vec::new();

    // Layer 1: defaults
    trace.push(LayerStep {
        label: "defaults".into(),
        loaded: !is_null_or_missing(&defaults),
        top_level_keys: top_level_keys(&defaults),
        error: None,
    });

    // Layers 2: system files
    for path in &cfg.system_files {
        let (val, step) = load_optional_layer(path, &format!("system: {}", path.display()), cfg)?;
        merge_into(&mut merged, val, cfg.array_strategy);
        trace.push(step);
    }

    // Layer 3: user file
    if let Some(path) = &cfg.user_file {
        let (val, step) = load_optional_layer(path, &format!("user: {}", path.display()), cfg)?;
        merge_into(&mut merged, val, cfg.array_strategy);
        trace.push(step);
    }

    // Layer 4: machine override
    if let Some(path) = &cfg.machine_file {
        let (val, step) = load_optional_layer(path, &format!("machine: {}", path.display()), cfg)?;
        merge_into(&mut merged, val, cfg.array_strategy);
        trace.push(step);
    }

    // Layers 5+6: project ancestors then project root
    if let Some(root) = &cfg.project_root {
        // Collect ancestors furthest-first.
        let ancestors = collect_ancestors(root, cfg.max_ancestor_walk);
        for (depth, dir) in ancestors.iter().enumerate() {
            if let Some(path) = first_existing(dir, &cfg.project_filenames) {
                let (val, step) = load_optional_layer(
                    &path,
                    &format!(
                        "project ancestor [depth -{}]: {}",
                        depth + 1,
                        path.display()
                    ),
                    cfg,
                )?;
                merge_into(&mut merged, val, cfg.array_strategy);
                trace.push(step);
            }
        }
        // Project root itself.
        if let Some(path) = first_existing(root, &cfg.project_filenames) {
            let (val, step) =
                load_optional_layer(&path, &format!("project root: {}", path.display()), cfg)?;
            merge_into(&mut merged, val, cfg.array_strategy);
            trace.push(step);
        }
    }

    // Layer 7: env vars
    let env_value = collect_env_layer(&cfg.env_prefix, cfg.read_process_env, &cfg.env_overrides);
    let env_keys = top_level_keys(&env_value);
    let env_loaded = !is_null_or_missing(&env_value);
    merge_into(&mut merged, env_value, cfg.array_strategy);
    trace.push(LayerStep {
        label: format!("env (prefix {})", cfg.env_prefix),
        loaded: env_loaded,
        top_level_keys: env_keys,
        error: None,
    });

    // Layer 8: explicit CLI overrides
    if let Some(ov) = cli_overrides {
        let keys = top_level_keys(&ov);
        let loaded = !is_null_or_missing(&ov);
        merge_into(&mut merged, ov, cfg.array_strategy);
        trace.push(LayerStep {
            label: "cli overrides".into(),
            loaded,
            top_level_keys: keys,
            error: None,
        });
    }

    Ok(LayeredConfig { merged, trace })
}

// ============================================================================
// Helpers — file loading
// ============================================================================

fn load_optional_layer(
    path: &Path,
    label: &str,
    cfg: &LayerLoadConfig,
) -> Result<(Value, LayerStep), LayeredError> {
    let meta = match if cfg.follow_symlinks {
        fs::metadata(path)
    } else {
        fs::symlink_metadata(path)
    } {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok((
                Value::Null,
                LayerStep {
                    label: label.into(),
                    loaded: false,
                    top_level_keys: 0,
                    error: None,
                },
            ));
        }
        Err(e) => {
            return Err(LayeredError::Io {
                path: path.to_path_buf(),
                source: e,
            });
        }
    };

    if meta.file_type().is_symlink() && !cfg.follow_symlinks {
        return Err(LayeredError::SymlinkRejected(path.to_path_buf()));
    }
    if !meta.is_file() {
        return Err(LayeredError::NotRegularFile(path.to_path_buf()));
    }
    let size = meta.len();
    if size > cfg.max_layer_size {
        return Err(LayeredError::LayerTooLarge {
            path: path.to_path_buf(),
            size,
            limit: cfg.max_layer_size,
        });
    }

    let bytes = fs::read(path).map_err(|e| LayeredError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let val: Value = serde_json::from_slice(&bytes).map_err(|e| LayeredError::InvalidJson {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let keys = top_level_keys(&val);
    Ok((
        val,
        LayerStep {
            label: label.into(),
            loaded: true,
            top_level_keys: keys,
            error: None,
        },
    ))
}

fn first_existing(dir: &Path, names: &[String]) -> Option<PathBuf> {
    for name in names {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn collect_ancestors(root: &Path, max: usize) -> Vec<PathBuf> {
    // Returns ancestors in furthest-first order (lowest precedence first).
    let mut stack: Vec<PathBuf> = Vec::new();
    let mut cur = root.parent();
    let mut count = 0usize;
    while let Some(p) = cur {
        if count >= max {
            break;
        }
        stack.push(p.to_path_buf());
        cur = p.parent();
        count += 1;
    }
    stack.reverse();
    stack
}

// ============================================================================
// Helpers — env layer
// ============================================================================

fn collect_env_layer(
    prefix: &str,
    read_process_env: bool,
    extra: &HashMap<String, String>,
) -> Value {
    let prefix_marker = format!("{}__", prefix);
    let mut entries: Vec<(String, String)> = Vec::new();

    if read_process_env {
        for (k, v) in std::env::vars() {
            if k.starts_with(&prefix_marker) {
                entries.push((k, v));
            }
        }
    }
    for (k, v) in extra {
        if k.starts_with(&prefix_marker) {
            entries.push((k.clone(), v.clone()));
        }
    }

    if entries.is_empty() {
        return Value::Null;
    }

    let mut root = Value::Object(Map::new());
    for (k, v) in entries {
        let path = &k[prefix_marker.len()..];
        // Split on `__` for path segments; allow `_` literal in keys.
        let segs: Vec<&str> = path.split("__").collect();
        if segs.is_empty() || segs.iter().any(|s| s.is_empty()) {
            continue;
        }
        // Try JSON-parse the value, fall back to string.
        let parsed = serde_json::from_str::<Value>(&v).unwrap_or(Value::String(v));
        insert_at_path(&mut root, &segs, parsed);
    }
    root
}

fn insert_at_path(value: &mut Value, path: &[&str], leaf: Value) {
    if path.is_empty() {
        *value = leaf;
        return;
    }
    if !value.is_object() {
        *value = Value::Object(Map::new());
    }
    let obj = value.as_object_mut().unwrap();
    let key = path[0].to_lowercase();
    let rest = &path[1..];
    if rest.is_empty() {
        obj.insert(key, leaf);
    } else {
        let entry = obj.entry(key).or_insert_with(|| Value::Object(Map::new()));
        insert_at_path(entry, rest, leaf);
    }
}

// ============================================================================
// Helpers — deep merge
// ============================================================================

fn merge_into(dst: &mut Value, src: Value, arr: ArrayMergeStrategy) {
    match (dst, src) {
        (Value::Object(dst_map), Value::Object(src_map)) => {
            for (k, v) in src_map {
                match dst_map.get_mut(&k) {
                    Some(existing) => merge_into(existing, v, arr),
                    None => {
                        dst_map.insert(k, v);
                    }
                }
            }
        }
        (Value::Array(dst_arr), Value::Array(src_arr)) => match arr {
            ArrayMergeStrategy::Replace => {
                *dst_arr = src_arr;
            }
            ArrayMergeStrategy::Concat => {
                dst_arr.extend(src_arr);
            }
        },
        // Higher-layer null is "absent" → don't overwrite.
        (_dst, Value::Null) => {}
        // Anything else: src wins.
        (dst_slot, src_other) => {
            *dst_slot = src_other;
        }
    }
}

fn is_null_or_missing(v: &Value) -> bool {
    matches!(v, Value::Null)
}

fn top_level_keys(v: &Value) -> usize {
    match v {
        Value::Object(m) => m.len(),
        _ => 0,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ai_assistant_layered_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_json(path: &Path, json: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
    }

    fn no_env_cfg() -> LayerLoadConfig {
        LayerLoadConfig {
            read_process_env: false,
            ..LayerLoadConfig::default()
        }
    }

    // ---------- basic merge ----------

    #[test]
    fn defaults_only() {
        let defaults = serde_json::json!({"a": 1, "b": "x"});
        let res = load_layered_config(defaults.clone(), None, &no_env_cfg()).unwrap();
        assert_eq!(res.merged, defaults);
    }

    #[test]
    fn cli_overrides_win() {
        let defaults = serde_json::json!({"a": 1});
        let cli = serde_json::json!({"a": 99});
        let res = load_layered_config(defaults, Some(cli), &no_env_cfg()).unwrap();
        assert_eq!(res.merged["a"], 99);
    }

    #[test]
    fn deep_object_merge() {
        let defaults = serde_json::json!({"db": {"host": "a", "port": 5}});
        let cli = serde_json::json!({"db": {"port": 9}});
        let res = load_layered_config(defaults, Some(cli), &no_env_cfg()).unwrap();
        assert_eq!(res.merged["db"]["host"], "a");
        assert_eq!(res.merged["db"]["port"], 9);
    }

    #[test]
    fn arrays_replace_by_default() {
        let defaults = serde_json::json!({"list": [1, 2, 3]});
        let cli = serde_json::json!({"list": [9]});
        let res = load_layered_config(defaults, Some(cli), &no_env_cfg()).unwrap();
        assert_eq!(res.merged["list"], serde_json::json!([9]));
    }

    #[test]
    fn arrays_concat_when_opted_in() {
        let defaults = serde_json::json!({"list": [1, 2]});
        let cli = serde_json::json!({"list": [3, 4]});
        let mut cfg = no_env_cfg();
        cfg.array_strategy = ArrayMergeStrategy::Concat;
        let res = load_layered_config(defaults, Some(cli), &cfg).unwrap();
        assert_eq!(res.merged["list"], serde_json::json!([1, 2, 3, 4]));
    }

    #[test]
    fn null_in_higher_layer_does_not_overwrite() {
        let defaults = serde_json::json!({"a": 1});
        let cli = serde_json::json!({"a": null});
        let res = load_layered_config(defaults, Some(cli), &no_env_cfg()).unwrap();
        assert_eq!(res.merged["a"], 1);
    }

    // ---------- file layers ----------

    #[test]
    fn user_file_layer_loaded() {
        let dir = tmpdir("user");
        let file = dir.join("config.json");
        write_json(&file, r#"{"a": "from-user"}"#);
        let mut cfg = no_env_cfg();
        cfg.user_file = Some(file.clone());
        let res = load_layered_config(serde_json::json!({"a": "default"}), None, &cfg).unwrap();
        assert_eq!(res.merged["a"], "from-user");
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let mut cfg = no_env_cfg();
        cfg.user_file = Some(PathBuf::from("/nonexistent/zzz/cfg.json"));
        let res = load_layered_config(serde_json::json!({"a": 1}), None, &cfg).unwrap();
        assert_eq!(res.merged["a"], 1);
    }

    #[test]
    fn invalid_json_surfaces_error() {
        let dir = tmpdir("invalid");
        let file = dir.join("bad.json");
        write_json(&file, "{not json}");
        let mut cfg = no_env_cfg();
        cfg.user_file = Some(file);
        let err = load_layered_config(serde_json::json!({}), None, &cfg).unwrap_err();
        match err {
            LayeredError::InvalidJson { .. } => {}
            other => panic!("expected InvalidJson, got {:?}", other),
        }
    }

    #[test]
    fn size_cap_rejects_oversized() {
        let dir = tmpdir("size");
        let file = dir.join("big.json");
        let body = format!("{{\"x\":\"{}\"}}", "y".repeat(2000));
        write_json(&file, &body);
        let mut cfg = no_env_cfg();
        cfg.user_file = Some(file);
        cfg.max_layer_size = 500;
        let err = load_layered_config(serde_json::json!({}), None, &cfg).unwrap_err();
        match err {
            LayeredError::LayerTooLarge { .. } => {}
            other => panic!("expected LayerTooLarge, got {:?}", other),
        }
    }

    #[test]
    fn project_ancestors_then_root_precedence() {
        let dir = tmpdir("ancestors");
        let parent = dir.clone();
        let child = parent.join("child");
        let grand = child.join("grand");
        fs::create_dir_all(&grand).unwrap();
        write_json(&parent.join(".ai_assistant.json"), r#"{"a": "parent"}"#);
        write_json(&child.join(".ai_assistant.json"), r#"{"a": "child"}"#);
        write_json(&grand.join(".ai_assistant.json"), r#"{"a": "grand"}"#);
        let mut cfg = no_env_cfg();
        cfg.project_root = Some(grand);
        let res = load_layered_config(serde_json::json!({"a": "default"}), None, &cfg).unwrap();
        // Walk picks: parent (depth -2), child (depth -1), then grand (root).
        // Root itself wins.
        assert_eq!(res.merged["a"], "grand");
    }

    #[test]
    fn ancestor_walk_depth_capped() {
        let mut cfg = no_env_cfg();
        cfg.project_root = Some(PathBuf::from("/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q"));
        cfg.max_ancestor_walk = 3;
        let res = load_layered_config(serde_json::json!({}), None, &cfg).unwrap();
        // Just verifies it doesn't blow up walking.
        assert!(res.merged.is_object());
    }

    // ---------- env layer ----------

    #[test]
    fn env_overrides_via_extra_map() {
        let mut env = HashMap::new();
        env.insert("AI_ASSISTANT__db__port".into(), "9999".into());
        env.insert("AI_ASSISTANT__db__host".into(), "\"prod\"".into());
        let cfg = LayerLoadConfig {
            read_process_env: false,
            env_overrides: env,
            ..LayerLoadConfig::default()
        };
        let res = load_layered_config(
            serde_json::json!({"db": {"host": "a", "port": 5}}),
            None,
            &cfg,
        )
        .unwrap();
        assert_eq!(res.merged["db"]["port"], 9999);
        assert_eq!(res.merged["db"]["host"], "prod");
    }

    #[test]
    fn env_value_falls_back_to_string_when_not_json() {
        let mut env = HashMap::new();
        env.insert("AI_ASSISTANT__name".into(), "no-quotes".into());
        let cfg = LayerLoadConfig {
            read_process_env: false,
            env_overrides: env,
            ..LayerLoadConfig::default()
        };
        let res = load_layered_config(serde_json::json!({}), None, &cfg).unwrap();
        assert_eq!(res.merged["name"], "no-quotes");
    }

    #[test]
    fn env_ignores_keys_without_prefix() {
        let mut env = HashMap::new();
        env.insert("OTHER__foo".into(), "1".into());
        let cfg = LayerLoadConfig {
            read_process_env: false,
            env_overrides: env,
            ..LayerLoadConfig::default()
        };
        let res = load_layered_config(serde_json::json!({"k": "v"}), None, &cfg).unwrap();
        assert!(res.merged.get("foo").is_none());
        assert_eq!(res.merged["k"], "v");
    }

    #[test]
    fn env_lowercases_segments() {
        let mut env = HashMap::new();
        env.insert("AI_ASSISTANT__DB__PORT".into(), "42".into());
        let cfg = LayerLoadConfig {
            read_process_env: false,
            env_overrides: env,
            ..LayerLoadConfig::default()
        };
        let res = load_layered_config(serde_json::json!({}), None, &cfg).unwrap();
        assert_eq!(res.merged["db"]["port"], 42);
    }

    // ---------- precedence integration ----------

    #[test]
    fn full_precedence_chain() {
        let dir = tmpdir("full");
        let user_file = dir.join("user.json");
        let proj_root = dir.join("proj");
        fs::create_dir_all(&proj_root).unwrap();
        let proj_file = proj_root.join(".ai_assistant.json");

        write_json(&user_file, r#"{"a": "user", "b": "user"}"#);
        write_json(&proj_file, r#"{"b": "proj", "c": "proj"}"#);

        let mut env = HashMap::new();
        env.insert("AI_ASSISTANT__c".into(), "\"env\"".into());
        env.insert("AI_ASSISTANT__d".into(), "\"env\"".into());

        let cfg = LayerLoadConfig {
            user_file: Some(user_file),
            project_root: Some(proj_root),
            read_process_env: false,
            env_overrides: env,
            ..LayerLoadConfig::default()
        };
        let cli = serde_json::json!({"d": "cli", "e": "cli"});

        let res = load_layered_config(
            serde_json::json!({"a": "def", "b": "def", "c": "def", "d": "def", "e": "def"}),
            Some(cli),
            &cfg,
        )
        .unwrap();
        assert_eq!(res.merged["a"], "user");
        assert_eq!(res.merged["b"], "proj");
        assert_eq!(res.merged["c"], "env");
        assert_eq!(res.merged["d"], "cli");
        assert_eq!(res.merged["e"], "cli");
    }

    // ---------- trace ----------

    #[test]
    fn trace_records_each_layer() {
        let dir = tmpdir("trace");
        let file = dir.join("u.json");
        write_json(&file, r#"{"x": 1, "y": 2}"#);
        let cfg = LayerLoadConfig {
            user_file: Some(file),
            read_process_env: false,
            ..LayerLoadConfig::default()
        };
        let res = load_layered_config(serde_json::json!({"a": 1}), None, &cfg).unwrap();
        assert!(res.trace.iter().any(|s| s.label == "defaults"));
        let user_step = res
            .trace
            .iter()
            .find(|s| s.label.starts_with("user:"))
            .unwrap();
        assert!(user_step.loaded);
        assert_eq!(user_step.top_level_keys, 2);
    }

    // ---------- typed deserialize ----------

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct DummyCfg {
        a: String,
        port: u16,
    }

    #[test]
    fn deserialize_into_typed_config() {
        let res = load_layered_config(
            serde_json::json!({"a": "hello", "port": 8080}),
            None,
            &no_env_cfg(),
        )
        .unwrap();
        let c: DummyCfg = res.deserialize().unwrap();
        assert_eq!(
            c,
            DummyCfg {
                a: "hello".into(),
                port: 8080
            }
        );
    }
}
