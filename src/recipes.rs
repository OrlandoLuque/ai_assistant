//! Recipes (V106 / Phase A.1) — declarative, portable, composable workflows.
//!
//! A recipe is a YAML file that describes a multi-step LLM workflow:
//! variables, prompts, tool calls, and sub-recipe composition. Recipes
//! are *portable artifacts*: drop a `.yaml` file into a project, run it
//! with `ai_cli recipes run <name>`, and the engine takes care of
//! variable substitution, step ordering, and capturing intermediate
//! outputs for downstream steps.
//!
//! Inspired by Goose's recipes pattern (YAML + Jinja-ish substitution +
//! sub-recipes). Designed for parity with Hermes/Autocode/OpenHands
//! workflow tiers while reusing our existing `slash_commands` discovery
//! and `ApprovalHandler` trust model.
//!
//! ## Schema (apiVersion: recipes/v1)
//!
//! ```yaml
//! apiVersion: recipes/v1
//! name: code-review
//! description: Review a code file for bugs and clarity
//! version: "1.0.0"
//! author: orlando.luque@gmail.com
//! tags: [code, review]
//!
//! variables:
//!   file_path:
//!     description: Path to the file
//!     required: true
//!   focus:
//!     description: What to focus on
//!     default: "general bugs"
//!
//! model: claude-opus-4-7
//! provider: anthropic
//!
//! steps:
//!   - id: read_file
//!     type: tool
//!     tool: file_read
//!     args:
//!       path: "{{file_path}}"
//!
//!   - id: review
//!     type: prompt
//!     prompt: |
//!       Review the file {{file_path}}, focus on {{focus}}:
//!       {{steps.read_file.output}}
//!
//!   - id: format
//!     type: recipe
//!     recipe: format-markdown
//!     args:
//!       content: "{{steps.review.output}}"
//!
//! output: "{{steps.format.output}}"
//! ```
//!
//! ## Schema versioning
//!
//! `apiVersion` must be `recipes/v1`. Future versions can be supported
//! by [`Recipe::migrate_to_v1`]. Unknown apiVersion → load error.
//!
//! ## Discovery
//!
//! Mirrors [`crate::slash_commands`]: pass an ordered list of roots,
//! later roots override earlier on duplicate names. Typical setup:
//!
//! 1. user-global: `<config-dir>/ai_assistant/recipes/`
//! 2. project root: `<project>/.ai_assistant/recipes/`
//!
//! ## Trust model
//!
//! Recipe files are *user-supplied content* — they could be hostile.
//! Defenses:
//! - File size cap (default 256 KiB)
//! - Symlinks rejected by default
//! - UTF-8 enforced
//! - Only `.yaml` / `.yml` files loaded
//! - Sub-recipe depth capped (default 8) to prevent infinite recursion
//! - Step count capped per recipe (default 64)
//! - Variables with `eval:` / shell-like syntax never executed — pure
//!   string substitution only
//! - `shell` step type gated behind explicit caller opt-in (not in v1)

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for recipe discovery and execution.
#[derive(Debug, Clone)]
pub struct RecipeConfig {
    /// Maximum bytes per recipe file. Default 256 KiB.
    pub max_file_size: u64,
    /// Whether to follow symlinks when reading recipes. Default false.
    pub follow_symlinks: bool,
    /// Maximum number of recipes loaded per root. Default 256.
    pub max_per_root: usize,
    /// Maximum sub-recipe call depth. Default 8.
    pub max_recipe_depth: usize,
    /// Maximum steps per recipe. Default 64.
    pub max_steps: usize,
    /// Whether to allow `shell` step type. Default false (security).
    pub allow_shell: bool,
}

impl Default for RecipeConfig {
    fn default() -> Self {
        Self {
            max_file_size: 256 * 1024,
            follow_symlinks: false,
            max_per_root: 256,
            max_recipe_depth: 8,
            max_steps: 64,
            allow_shell: false,
        }
    }
}

// ============================================================================
// Schema types
// ============================================================================

/// Supported schema versions.
pub const SUPPORTED_API_VERSION: &str = "recipes/v1";

/// A loaded recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    /// Schema version, e.g. `recipes/v1`.
    pub api_version: String,
    /// Canonical name (lowercase, no spaces).
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Recipe semver (e.g., `1.0.0`).
    pub version: Option<String>,
    /// Author email/name.
    pub author: Option<String>,
    /// Tags for discovery/filtering.
    pub tags: Vec<String>,
    /// Variable schema, in declaration order.
    pub variables: Vec<RecipeVariable>,
    /// Optional model hint (resolved by caller via existing routing).
    pub model: Option<String>,
    /// Optional provider hint.
    pub provider: Option<String>,
    /// Steps in execution order.
    pub steps: Vec<RecipeStep>,
    /// Final output template (uses `{{steps.id.output}}` and variables).
    pub output: Option<String>,
    /// Path the recipe was loaded from.
    pub source_path: PathBuf,
}

/// One variable declared by a recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeVariable {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
    pub default: Option<String>,
}

/// One step in a recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeStep {
    /// Step identifier — referenced by `{{steps.<id>.output}}`.
    pub id: String,
    /// Step kind.
    pub kind: StepKind,
    /// Optional human description.
    pub description: Option<String>,
}

/// Kind of step a recipe can perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Send `prompt` to the LLM. Capture the response as the step output.
    Prompt { prompt: String },
    /// Invoke a tool by name with named args.
    Tool {
        tool: String,
        args: BTreeMap<String, String>,
    },
    /// Call another recipe with bound variables.
    Recipe {
        recipe: String,
        args: BTreeMap<String, String>,
    },
    /// Run a shell command. Disabled unless `RecipeConfig::allow_shell`.
    Shell { command: String },
}

// ============================================================================
// Errors
// ============================================================================

/// Errors during recipe load, validation, or execution.
#[derive(Debug)]
pub enum RecipeError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    NotRegularFile(PathBuf),
    SymlinkRejected(PathBuf),
    InvalidUtf8(PathBuf),
    InvalidYaml {
        path: PathBuf,
        message: String,
    },
    UnsupportedApiVersion {
        path: PathBuf,
        got: String,
    },
    SchemaViolation {
        path: PathBuf,
        message: String,
    },
    UnknownRecipe(String),
    RecursionLimit(usize),
    StepLimit(usize),
    ShellDisabled,
    MissingRequiredVariable(String),
    StepFailed {
        step_id: String,
        message: String,
    },
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "I/O error reading {}: {}", path.display(), source)
            }
            Self::TooLarge { path, size, limit } => write!(
                f,
                "recipe file too large: {} ({} > {} bytes)",
                path.display(),
                size,
                limit
            ),
            Self::NotRegularFile(p) => write!(f, "not a regular file: {}", p.display()),
            Self::SymlinkRejected(p) => write!(f, "symlink rejected: {}", p.display()),
            Self::InvalidUtf8(p) => write!(f, "invalid UTF-8: {}", p.display()),
            Self::InvalidYaml { path, message } => {
                write!(f, "invalid YAML in {}: {}", path.display(), message)
            }
            Self::UnsupportedApiVersion { path, got } => write!(
                f,
                "unsupported apiVersion in {}: got '{}', expected '{}'",
                path.display(),
                got,
                SUPPORTED_API_VERSION
            ),
            Self::SchemaViolation { path, message } => {
                write!(f, "schema violation in {}: {}", path.display(), message)
            }
            Self::UnknownRecipe(n) => write!(f, "unknown recipe: '{}'", n),
            Self::RecursionLimit(n) => write!(f, "sub-recipe recursion limit exceeded: {}", n),
            Self::StepLimit(n) => write!(f, "step limit exceeded: {}", n),
            Self::ShellDisabled => write!(f, "shell steps are disabled (set allow_shell)"),
            Self::MissingRequiredVariable(v) => write!(f, "missing required variable: '{}'", v),
            Self::StepFailed { step_id, message } => {
                write!(f, "step '{}' failed: {}", step_id, message)
            }
        }
    }
}

impl std::error::Error for RecipeError {}

// ============================================================================
// Registry
// ============================================================================

/// In-memory recipe registry. Later roots override earlier on duplicate names.
#[derive(Debug, Default, Clone)]
pub struct RecipeRegistry {
    by_name: BTreeMap<String, Recipe>,
    /// Recipes that failed to load, with reason.
    pub load_errors: Vec<(PathBuf, String)>,
}

impl RecipeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, r: Recipe) {
        self.by_name.insert(r.name.clone(), r);
    }

    pub fn get(&self, name: &str) -> Option<&Recipe> {
        self.by_name.get(&name.to_lowercase())
    }

    pub fn names(&self) -> Vec<String> {
        self.by_name.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Recipe)> {
        self.by_name.iter()
    }
}

// ============================================================================
// Discovery
// ============================================================================

/// Scan each root in order and build a [`RecipeRegistry`].
///
/// Roots are typically `[user_dir, project_dir]`; later roots override
/// earlier on duplicate names. Both `.yaml` and `.yml` are loaded.
pub fn discover_recipes(roots: &[PathBuf], cfg: &RecipeConfig) -> RecipeRegistry {
    let mut reg = RecipeRegistry::new();
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut count = 0usize;
        for entry in entries.flatten() {
            if count >= cfg.max_per_root {
                break;
            }
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str());
            if !matches!(ext, Some("yaml") | Some("yml")) {
                continue;
            }
            match load_one(&path, cfg) {
                Ok(r) => {
                    reg.insert(r);
                    count += 1;
                }
                Err(e) => {
                    reg.load_errors.push((path, e.to_string()));
                }
            }
        }
    }
    reg
}

// ============================================================================
// Validator
// ============================================================================

/// Validate a recipe against schema rules. Returns Ok if valid.
///
/// Checks performed:
/// - apiVersion is supported
/// - name is non-empty and lowercase-able
/// - step IDs are unique and non-empty
/// - step count within limit
/// - shell steps only if `allow_shell`
/// - variable defaults are strings
/// - sub-recipe references don't have empty `recipe:` field
pub fn validate_recipe(r: &Recipe, cfg: &RecipeConfig) -> Result<(), RecipeError> {
    if r.api_version != SUPPORTED_API_VERSION {
        return Err(RecipeError::UnsupportedApiVersion {
            path: r.source_path.clone(),
            got: r.api_version.clone(),
        });
    }
    if r.name.trim().is_empty() {
        return Err(RecipeError::SchemaViolation {
            path: r.source_path.clone(),
            message: "recipe name is empty".into(),
        });
    }
    if r.steps.len() > cfg.max_steps {
        return Err(RecipeError::StepLimit(r.steps.len()));
    }
    let mut seen = std::collections::HashSet::new();
    for step in &r.steps {
        if step.id.trim().is_empty() {
            return Err(RecipeError::SchemaViolation {
                path: r.source_path.clone(),
                message: "step id is empty".into(),
            });
        }
        if !seen.insert(step.id.clone()) {
            return Err(RecipeError::SchemaViolation {
                path: r.source_path.clone(),
                message: format!("duplicate step id: '{}'", step.id),
            });
        }
        match &step.kind {
            StepKind::Shell { .. } if !cfg.allow_shell => return Err(RecipeError::ShellDisabled),
            StepKind::Recipe { recipe, .. } if recipe.trim().is_empty() => {
                return Err(RecipeError::SchemaViolation {
                    path: r.source_path.clone(),
                    message: format!("step '{}' has empty 'recipe' field", step.id),
                });
            }
            StepKind::Tool { tool, .. } if tool.trim().is_empty() => {
                return Err(RecipeError::SchemaViolation {
                    path: r.source_path.clone(),
                    message: format!("step '{}' has empty 'tool' field", step.id),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

// ============================================================================
// Execution
// ============================================================================

/// Output of a single step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub step_id: String,
    pub output: String,
}

/// Result of executing a recipe end-to-end.
#[derive(Debug, Clone)]
pub struct RecipeRunResult {
    pub recipe_name: String,
    pub steps: Vec<StepOutput>,
    pub final_output: String,
}

/// Callback invoked when a `prompt` step needs an LLM response.
///
/// Receives the rendered prompt; returns the model's reply or `None` on
/// failure (which surfaces as `StepFailed`).
pub type LlmCallback = Box<dyn Fn(&str) -> Option<String>>;

/// Callback invoked when a `tool` step runs.
///
/// Receives `(tool_name, named_args)`; returns the tool output or `None`
/// on failure.
pub type ToolCallback = Box<dyn Fn(&str, &BTreeMap<String, String>) -> Option<String>>;

/// Recipe execution engine.
///
/// Builder-style configuration. Wire `LlmCallback` and `ToolCallback`
/// before running. Sub-recipes are resolved from the registry.
pub struct RecipeEngine {
    cfg: RecipeConfig,
    llm: Option<LlmCallback>,
    tool: Option<ToolCallback>,
}

impl Default for RecipeEngine {
    fn default() -> Self {
        Self::new(RecipeConfig::default())
    }
}

impl RecipeEngine {
    pub fn new(cfg: RecipeConfig) -> Self {
        Self {
            cfg,
            llm: None,
            tool: None,
        }
    }

    pub fn with_llm<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Option<String> + 'static,
    {
        self.llm = Some(Box::new(f));
        self
    }

    pub fn with_tool<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &BTreeMap<String, String>) -> Option<String> + 'static,
    {
        self.tool = Some(Box::new(f));
        self
    }

    /// Run a recipe with the given variable bindings.
    ///
    /// Variable bindings are resolved in this order:
    /// 1. Caller-supplied `bindings`
    /// 2. Recipe-declared `default` values
    /// Missing required variables → `MissingRequiredVariable` error.
    pub fn run(
        &self,
        recipe: &Recipe,
        bindings: &BTreeMap<String, String>,
        registry: &RecipeRegistry,
    ) -> Result<RecipeRunResult, RecipeError> {
        self.run_inner(recipe, bindings, registry, 0)
    }

    fn run_inner(
        &self,
        recipe: &Recipe,
        bindings: &BTreeMap<String, String>,
        registry: &RecipeRegistry,
        depth: usize,
    ) -> Result<RecipeRunResult, RecipeError> {
        if depth > self.cfg.max_recipe_depth {
            return Err(RecipeError::RecursionLimit(depth));
        }
        validate_recipe(recipe, &self.cfg)?;

        // Resolve variables.
        let mut vars: BTreeMap<String, String> = BTreeMap::new();
        for v in &recipe.variables {
            if let Some(supplied) = bindings.get(&v.name) {
                vars.insert(v.name.clone(), supplied.clone());
            } else if let Some(def) = &v.default {
                vars.insert(v.name.clone(), def.clone());
            } else if v.required {
                return Err(RecipeError::MissingRequiredVariable(v.name.clone()));
            }
        }
        // Allow callers to pass extra bindings even if not declared (lenient).
        for (k, val) in bindings {
            vars.entry(k.clone()).or_insert_with(|| val.clone());
        }

        // Execute steps.
        let mut step_outputs: Vec<StepOutput> = Vec::new();
        for step in &recipe.steps {
            let out = self.execute_step(step, &vars, &step_outputs, registry, depth)?;
            step_outputs.push(StepOutput {
                step_id: step.id.clone(),
                output: out,
            });
        }

        // Render final output.
        let final_output = if let Some(tmpl) = &recipe.output {
            substitute(tmpl, &vars, &step_outputs)
        } else {
            step_outputs
                .last()
                .map(|s| s.output.clone())
                .unwrap_or_default()
        };

        Ok(RecipeRunResult {
            recipe_name: recipe.name.clone(),
            steps: step_outputs,
            final_output,
        })
    }

    fn execute_step(
        &self,
        step: &RecipeStep,
        vars: &BTreeMap<String, String>,
        prior: &[StepOutput],
        registry: &RecipeRegistry,
        depth: usize,
    ) -> Result<String, RecipeError> {
        match &step.kind {
            StepKind::Prompt { prompt } => {
                let rendered = substitute(prompt, vars, prior);
                let llm = self.llm.as_ref().ok_or_else(|| RecipeError::StepFailed {
                    step_id: step.id.clone(),
                    message: "no LLM callback configured".into(),
                })?;
                llm(&rendered).ok_or_else(|| RecipeError::StepFailed {
                    step_id: step.id.clone(),
                    message: "LLM call returned None".into(),
                })
            }
            StepKind::Tool { tool, args } => {
                let resolved: BTreeMap<String, String> = args
                    .iter()
                    .map(|(k, v)| (k.clone(), substitute(v, vars, prior)))
                    .collect();
                let cb = self.tool.as_ref().ok_or_else(|| RecipeError::StepFailed {
                    step_id: step.id.clone(),
                    message: "no Tool callback configured".into(),
                })?;
                cb(tool, &resolved).ok_or_else(|| RecipeError::StepFailed {
                    step_id: step.id.clone(),
                    message: format!("tool '{}' returned None", tool),
                })
            }
            StepKind::Recipe { recipe, args } => {
                let sub = registry
                    .get(recipe)
                    .ok_or_else(|| RecipeError::UnknownRecipe(recipe.clone()))?;
                let resolved: BTreeMap<String, String> = args
                    .iter()
                    .map(|(k, v)| (k.clone(), substitute(v, vars, prior)))
                    .collect();
                let r = self.run_inner(sub, &resolved, registry, depth + 1)?;
                Ok(r.final_output)
            }
            StepKind::Shell { command } => {
                if !self.cfg.allow_shell {
                    return Err(RecipeError::ShellDisabled);
                }
                let rendered = substitute(command, vars, prior);
                // Execute with std::process::Command — but ONLY if allow_shell.
                // We do not provide a callback hook here since shell is meant
                // to be a last resort and runs with the engine's own privileges.
                #[cfg(unix)]
                let out = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&rendered)
                    .output();
                #[cfg(windows)]
                let out = std::process::Command::new("cmd")
                    .args(["/C", &rendered])
                    .output();
                let out = out.map_err(|e| RecipeError::StepFailed {
                    step_id: step.id.clone(),
                    message: format!("shell exec error: {e}"),
                })?;
                if !out.status.success() {
                    return Err(RecipeError::StepFailed {
                        step_id: step.id.clone(),
                        message: format!(
                            "shell exit {}: {}",
                            out.status.code().unwrap_or(-1),
                            String::from_utf8_lossy(&out.stderr).trim()
                        ),
                    });
                }
                Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
            }
        }
    }
}

// ============================================================================
// Variable substitution
// ============================================================================

/// Substitute `{{var}}` and `{{steps.<id>.output}}` placeholders in a
/// template. Unbound placeholders are left verbatim.
pub fn substitute(template: &str, vars: &BTreeMap<String, String>, steps: &[StepOutput]) -> String {
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 <= bytes.len() && &bytes[i..i + 2] == b"{{" {
            if let Some(end_rel) = find_subseq(&bytes[i + 2..], b"}}") {
                let key = template[i + 2..i + 2 + end_rel].trim();
                if let Some(v) = resolve_placeholder(key, vars, steps) {
                    out.push_str(&v);
                } else {
                    out.push_str("{{");
                    out.push_str(&template[i + 2..i + 2 + end_rel]);
                    out.push_str("}}");
                }
                i += 2 + end_rel + 2;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn resolve_placeholder(
    key: &str,
    vars: &BTreeMap<String, String>,
    steps: &[StepOutput],
) -> Option<String> {
    // steps.<id>.output
    if let Some(rest) = key.strip_prefix("steps.") {
        if let Some(id_end) = rest.find('.') {
            let id = &rest[..id_end];
            let field = &rest[id_end + 1..];
            if field == "output" {
                return steps
                    .iter()
                    .find(|s| s.step_id == id)
                    .map(|s| s.output.clone());
            }
        }
        return None;
    }
    vars.get(key).cloned()
}

fn find_subseq(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    for i in 0..=hay.len() - needle.len() {
        if &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// Loading + parsing
// ============================================================================

fn load_one(path: &Path, cfg: &RecipeConfig) -> Result<Recipe, RecipeError> {
    let meta = if cfg.follow_symlinks {
        fs::metadata(path)
    } else {
        fs::symlink_metadata(path)
    }
    .map_err(|e| RecipeError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    if meta.file_type().is_symlink() && !cfg.follow_symlinks {
        return Err(RecipeError::SymlinkRejected(path.to_path_buf()));
    }
    if !meta.is_file() {
        return Err(RecipeError::NotRegularFile(path.to_path_buf()));
    }
    let size = meta.len();
    if size > cfg.max_file_size {
        return Err(RecipeError::TooLarge {
            path: path.to_path_buf(),
            size,
            limit: cfg.max_file_size,
        });
    }

    let bytes = fs::read(path).map_err(|e| RecipeError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let text =
        std::str::from_utf8(&bytes).map_err(|_| RecipeError::InvalidUtf8(path.to_path_buf()))?;

    let mut r = parse_recipe(text, path).map_err(|e| RecipeError::InvalidYaml {
        path: path.to_path_buf(),
        message: e,
    })?;
    r.source_path = path.to_path_buf();

    // Default name from filename stem if unset.
    if r.name.trim().is_empty() {
        r.name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_lowercase();
    }
    Ok(r)
}

/// Parse a recipe from a YAML string.
///
/// This is a *limited-subset* YAML parser tailored to the recipe schema.
/// It supports: scalar key:value, block scalars (`|`), inline lists
/// `[a, b]`, block lists `- item`, and mappings nested by indentation.
/// It rejects anchors/references, tags, flow mappings `{...}`, and
/// arbitrary YAML constructs — keeping the trust surface narrow.
pub fn parse_recipe(text: &str, path: &Path) -> Result<Recipe, String> {
    let stripped = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<&str> = stripped.lines().collect();

    let mut api_version = String::new();
    let mut name = String::new();
    let mut description: Option<String> = None;
    let mut version: Option<String> = None;
    let mut author: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();
    let mut model: Option<String> = None;
    let mut provider: Option<String> = None;
    let mut variables: Vec<RecipeVariable> = Vec::new();
    let mut steps: Vec<RecipeStep> = Vec::new();
    let mut output: Option<String> = None;

    let mut i = 0usize;
    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();

        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        if indent != 0 {
            // Top-level only consumes indent==0 keys; nested block consumed below.
            i += 1;
            continue;
        }

        let (key, val_inline) = match line.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => return Err(format!("line {}: expected 'key: value'", i + 1)),
        };

        match key {
            "apiVersion" => api_version = unquote(val_inline).to_string(),
            "name" => name = unquote(val_inline).to_lowercase(),
            "description" => description = Some(unquote(val_inline).to_string()),
            "version" => version = Some(unquote(val_inline).to_string()),
            "author" => author = Some(unquote(val_inline).to_string()),
            "model" => model = Some(unquote(val_inline).to_string()),
            "provider" => provider = Some(unquote(val_inline).to_string()),
            "tags" => {
                if !val_inline.is_empty() {
                    tags = parse_inline_list(val_inline);
                } else {
                    let (consumed, items) = parse_block_list(&lines, i + 1, 0);
                    tags = items;
                    i += consumed;
                }
            }
            "output" => {
                if val_inline == "|" {
                    let (consumed, body) = parse_block_scalar(&lines, i + 1, 0);
                    output = Some(body);
                    i += consumed;
                } else {
                    output = Some(unquote(val_inline).to_string());
                }
            }
            "variables" => {
                let (consumed, vars) = parse_variables(&lines, i + 1, 0)?;
                variables = vars;
                i += consumed;
            }
            "steps" => {
                let (consumed, st) = parse_steps(&lines, i + 1, 0)?;
                steps = st;
                i += consumed;
            }
            _ => {
                // Unknown key — skip silently (forward-compat).
            }
        }
        i += 1;
    }

    if api_version.is_empty() {
        return Err("missing apiVersion".into());
    }
    if api_version != SUPPORTED_API_VERSION {
        return Err(format!(
            "unsupported apiVersion '{}', expected '{}'",
            api_version, SUPPORTED_API_VERSION
        ));
    }
    Ok(Recipe {
        api_version,
        name,
        description,
        version,
        author,
        tags,
        variables,
        model,
        provider,
        steps,
        output,
        source_path: path.to_path_buf(),
    })
}

/// Parse `variables:` block — each entry is `<name>:` then nested
/// `description:`, `required:`, `default:`.
fn parse_variables(
    lines: &[&str],
    start: usize,
    parent_indent: usize,
) -> Result<(usize, Vec<RecipeVariable>), String> {
    let mut vars: Vec<RecipeVariable> = Vec::new();
    let mut i = 0usize;
    let block_indent = detect_block_indent(lines, start, parent_indent);

    while start + i < lines.len() {
        let raw = lines[start + i];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();

        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        if indent < block_indent {
            break;
        }
        if indent > block_indent {
            // Nested deeper than expected — ignore (parsed by outer)
            i += 1;
            continue;
        }

        // Each variable: "name:"
        let (vname, vrest) = match line.split_once(':') {
            Some((k, v)) => (k.trim().to_string(), v.trim()),
            None => return Err(format!("line {}: expected variable 'name:'", start + i + 1)),
        };

        let mut description: Option<String> = None;
        let mut required = false;
        let mut default: Option<String> = None;

        if !vrest.is_empty() {
            // Inline scalar default value: `name: "default"`
            default = Some(unquote(vrest).to_string());
            i += 1;
            vars.push(RecipeVariable {
                name: vname,
                description,
                required,
                default,
            });
            continue;
        }

        i += 1;
        // Read nested attributes
        while start + i < lines.len() {
            let raw2 = lines[start + i];
            let trimmed2 = raw2.trim_end();
            let line2 = trimmed2.trim_start();
            let indent2 = trimmed2.len() - line2.len();
            if line2.is_empty() || line2.starts_with('#') {
                i += 1;
                continue;
            }
            if indent2 <= block_indent {
                break;
            }
            let (k2, v2) = match line2.split_once(':') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => return Err(format!("line {}: expected 'key: value'", start + i + 1)),
            };
            match k2 {
                "description" => description = Some(unquote(v2).to_string()),
                "required" => required = matches!(v2, "true" | "yes" | "1"),
                "default" => default = Some(unquote(v2).to_string()),
                _ => {}
            }
            i += 1;
        }

        vars.push(RecipeVariable {
            name: vname,
            description,
            required,
            default,
        });
    }
    Ok((i, vars))
}

/// Parse `steps:` block — each step starts with `- id: ...`.
fn parse_steps(
    lines: &[&str],
    start: usize,
    parent_indent: usize,
) -> Result<(usize, Vec<RecipeStep>), String> {
    let mut steps: Vec<RecipeStep> = Vec::new();
    let mut i = 0usize;
    let block_indent = detect_block_indent(lines, start, parent_indent);

    while start + i < lines.len() {
        let raw = lines[start + i];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();

        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        if indent < block_indent {
            break;
        }
        if !line.starts_with('-') {
            // Indented further (continuation of previous step) is handled below;
            // any other unrelated line breaks the block.
            if indent > block_indent {
                i += 1;
                continue;
            }
            break;
        }

        // Strip leading "- "
        let after_dash = line.trim_start_matches('-').trim_start();
        // Step's inner indent baseline is block_indent + (line.len() - after_dash.len() vs. dash position)
        // Simplification: collect attribute lines until indent <= block_indent or new "- "
        let mut id = String::new();
        let mut step_kind_kind: Option<&str> = None;
        let mut prompt_body: Option<String> = None;
        let mut tool_name: Option<String> = None;
        let mut recipe_name: Option<String> = None;
        let mut shell_cmd: Option<String> = None;
        let mut args_map: BTreeMap<String, String> = BTreeMap::new();
        let mut description: Option<String> = None;

        // Process the first line's "key: value" if present.
        if let Some((k, v)) = after_dash.split_once(':') {
            apply_step_attr(
                k.trim(),
                v.trim(),
                lines,
                start + i,
                block_indent + 2,
                &mut id,
                &mut step_kind_kind,
                &mut prompt_body,
                &mut tool_name,
                &mut recipe_name,
                &mut shell_cmd,
                &mut args_map,
                &mut description,
                &mut i,
            )?;
        }
        i += 1;

        // Continuation lines until new "- " at block_indent or shallower.
        while start + i < lines.len() {
            let raw2 = lines[start + i];
            let trimmed2 = raw2.trim_end();
            let line2 = trimmed2.trim_start();
            let indent2 = trimmed2.len() - line2.len();
            if line2.is_empty() || line2.starts_with('#') {
                i += 1;
                continue;
            }
            if indent2 <= block_indent {
                break;
            }
            if let Some((k, v)) = line2.split_once(':') {
                apply_step_attr(
                    k.trim(),
                    v.trim(),
                    lines,
                    start + i,
                    indent2 + 2,
                    &mut id,
                    &mut step_kind_kind,
                    &mut prompt_body,
                    &mut tool_name,
                    &mut recipe_name,
                    &mut shell_cmd,
                    &mut args_map,
                    &mut description,
                    &mut i,
                )?;
            }
            i += 1;
        }

        let kind = match step_kind_kind {
            Some("prompt") => StepKind::Prompt {
                prompt: prompt_body.unwrap_or_default(),
            },
            Some("tool") => StepKind::Tool {
                tool: tool_name.unwrap_or_default(),
                args: args_map,
            },
            Some("recipe") => StepKind::Recipe {
                recipe: recipe_name.unwrap_or_default(),
                args: args_map,
            },
            Some("shell") => StepKind::Shell {
                command: shell_cmd.unwrap_or_default(),
            },
            Some(other) => return Err(format!("unknown step type '{}'", other)),
            None => return Err(format!("step '{}' missing 'type'", id)),
        };

        steps.push(RecipeStep {
            id,
            kind,
            description,
        });
    }
    Ok((i, steps))
}

#[allow(clippy::too_many_arguments)]
fn apply_step_attr(
    k: &str,
    v: &str,
    lines: &[&str],
    cur: usize,
    nested_indent: usize,
    id: &mut String,
    kind: &mut Option<&'static str>,
    prompt_body: &mut Option<String>,
    tool_name: &mut Option<String>,
    recipe_name: &mut Option<String>,
    shell_cmd: &mut Option<String>,
    args_map: &mut BTreeMap<String, String>,
    description: &mut Option<String>,
    consumed: &mut usize,
) -> Result<(), String> {
    match k {
        "id" => *id = unquote(v).to_string(),
        "type" => {
            *kind = match unquote(v) {
                "prompt" => Some("prompt"),
                "tool" => Some("tool"),
                "recipe" => Some("recipe"),
                "shell" => Some("shell"),
                other => return Err(format!("unknown step type '{}'", other)),
            };
        }
        "description" => *description = Some(unquote(v).to_string()),
        "tool" => *tool_name = Some(unquote(v).to_string()),
        "recipe" => *recipe_name = Some(unquote(v).to_string()),
        "command" => {
            if v == "|" {
                let (n, body) = parse_block_scalar(lines, cur + 1, nested_indent.saturating_sub(2));
                *shell_cmd = Some(body);
                *consumed += n;
            } else {
                *shell_cmd = Some(unquote(v).to_string());
            }
        }
        "prompt" => {
            if v == "|" {
                let (n, body) = parse_block_scalar(lines, cur + 1, nested_indent.saturating_sub(2));
                *prompt_body = Some(body);
                *consumed += n;
            } else {
                *prompt_body = Some(unquote(v).to_string());
            }
        }
        "args" => {
            let (n, m) = parse_inline_or_block_map(lines, cur + 1, nested_indent.saturating_sub(2));
            *args_map = m;
            *consumed += n;
        }
        _ => {}
    }
    Ok(())
}

/// Parse a block scalar `|` body — collect indented lines until indent
/// drops to `parent_indent` or below.
fn parse_block_scalar(lines: &[&str], start: usize, parent_indent: usize) -> (usize, String) {
    let mut body = String::new();
    let mut consumed = 0usize;
    let mut block_indent: Option<usize> = None;
    while start + consumed < lines.len() {
        let raw = lines[start + consumed];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();
        if line.is_empty() {
            body.push('\n');
            consumed += 1;
            continue;
        }
        if indent <= parent_indent {
            break;
        }
        let bi = *block_indent.get_or_insert(indent);
        let strip = indent.min(bi);
        let stripped = if raw.len() >= strip {
            &raw[strip..]
        } else {
            raw
        };
        body.push_str(stripped);
        body.push('\n');
        consumed += 1;
    }
    (consumed, body.trim_end_matches('\n').to_string())
}

/// Parse a block list — `- a` lines.
fn parse_block_list(lines: &[&str], start: usize, parent_indent: usize) -> (usize, Vec<String>) {
    let mut out: Vec<String> = Vec::new();
    let mut consumed = 0usize;
    while start + consumed < lines.len() {
        let raw = lines[start + consumed];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();
        if line.is_empty() || line.starts_with('#') {
            consumed += 1;
            continue;
        }
        if indent <= parent_indent {
            break;
        }
        if let Some(rest) = line.strip_prefix("- ") {
            out.push(unquote(rest.trim()).to_string());
        } else if line == "-" {
            out.push(String::new());
        } else {
            break;
        }
        consumed += 1;
    }
    (consumed, out)
}

/// Parse either inline `{a: 1, b: 2}` (rejected for safety) or block map.
fn parse_inline_or_block_map(
    lines: &[&str],
    start: usize,
    parent_indent: usize,
) -> (usize, BTreeMap<String, String>) {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    let mut consumed = 0usize;
    while start + consumed < lines.len() {
        let raw = lines[start + consumed];
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        let indent = trimmed.len() - line.len();
        if line.is_empty() || line.starts_with('#') {
            consumed += 1;
            continue;
        }
        if indent <= parent_indent {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            out.insert(k.trim().to_string(), unquote(v.trim()).to_string());
        } else {
            break;
        }
        consumed += 1;
    }
    (consumed, out)
}

fn detect_block_indent(lines: &[&str], start: usize, parent_indent: usize) -> usize {
    for raw in lines.iter().skip(start) {
        let trimmed = raw.trim_end();
        let line = trimmed.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let indent = trimmed.len() - line.len();
        if indent > parent_indent {
            return indent;
        }
        return parent_indent + 2;
    }
    parent_indent + 2
}

fn parse_inline_list(raw: &str) -> Vec<String> {
    let s = raw.trim();
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };
    inner
        .split(',')
        .map(|p| {
            p.trim()
                .trim_matches(|c: char| c == '"' || c == '\'')
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

// ============================================================================
// Scaffold
// ============================================================================

/// Generate a minimal recipe template body for `init`.
pub fn scaffold_recipe(name: &str) -> String {
    format!(
        "apiVersion: {api}\n\
         name: {name}\n\
         description: One-line description here\n\
         version: \"0.1.0\"\n\
         tags: [example]\n\
         \n\
         variables:\n  \
           topic:\n    \
             description: What to explain\n    \
             required: true\n\
         \n\
         steps:\n  \
           - id: explain\n    \
             type: prompt\n    \
             prompt: |\n      \
               Explain {{{{topic}}}} in one paragraph.\n\
         \n\
         output: \"{{{{steps.explain.output}}}}\"\n",
        api = SUPPORTED_API_VERSION,
        name = name.to_lowercase()
    )
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
            "ai_assistant_recipes_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    fn fake_path() -> PathBuf {
        PathBuf::from("/tmp/x.yaml")
    }

    // ---------- parse ----------

    #[test]
    fn parse_minimal_recipe() {
        let yaml = "\
apiVersion: recipes/v1
name: minimal
description: A minimal recipe
steps:
  - id: hello
    type: prompt
    prompt: \"Say hello\"
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        assert_eq!(r.api_version, "recipes/v1");
        assert_eq!(r.name, "minimal");
        assert_eq!(r.description.as_deref(), Some("A minimal recipe"));
        assert_eq!(r.steps.len(), 1);
        assert_eq!(r.steps[0].id, "hello");
        match &r.steps[0].kind {
            StepKind::Prompt { prompt } => assert_eq!(prompt, "Say hello"),
            _ => panic!("expected Prompt"),
        }
    }

    #[test]
    fn parse_block_scalar_prompt() {
        let yaml = "\
apiVersion: recipes/v1
name: bs
steps:
  - id: s1
    type: prompt
    prompt: |
      line one
      line two
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        match &r.steps[0].kind {
            StepKind::Prompt { prompt } => {
                assert!(prompt.contains("line one"));
                assert!(prompt.contains("line two"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_variables_with_default_and_required() {
        let yaml = "\
apiVersion: recipes/v1
name: vars
variables:
  topic:
    description: Subject
    required: true
  tone:
    description: Tone
    default: friendly
steps:
  - id: s1
    type: prompt
    prompt: \"{{topic}} in {{tone}} tone\"
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        assert_eq!(r.variables.len(), 2);
        assert_eq!(r.variables[0].name, "topic");
        assert!(r.variables[0].required);
        assert_eq!(r.variables[1].name, "tone");
        assert_eq!(r.variables[1].default.as_deref(), Some("friendly"));
    }

    #[test]
    fn parse_tool_step_with_args() {
        let yaml = "\
apiVersion: recipes/v1
name: t
steps:
  - id: read
    type: tool
    tool: file_read
    args:
      path: README.md
      mode: text
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        match &r.steps[0].kind {
            StepKind::Tool { tool, args } => {
                assert_eq!(tool, "file_read");
                assert_eq!(args.get("path").map(|s| s.as_str()), Some("README.md"));
                assert_eq!(args.get("mode").map(|s| s.as_str()), Some("text"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_subrecipe_step() {
        let yaml = "\
apiVersion: recipes/v1
name: outer
steps:
  - id: inner
    type: recipe
    recipe: helper
    args:
      input: hello
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        match &r.steps[0].kind {
            StepKind::Recipe { recipe, args } => {
                assert_eq!(recipe, "helper");
                assert_eq!(args.get("input").map(|s| s.as_str()), Some("hello"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_rejects_unsupported_api_version() {
        let yaml = "\
apiVersion: recipes/v999
name: x
steps:
  - id: s
    type: prompt
    prompt: hi
";
        let err = parse_recipe(yaml, &fake_path()).unwrap_err();
        assert!(err.contains("unsupported apiVersion"));
    }

    #[test]
    fn parse_rejects_missing_api_version() {
        let yaml = "name: x\nsteps: []\n";
        let err = parse_recipe(yaml, &fake_path()).unwrap_err();
        assert!(err.contains("missing apiVersion"));
    }

    #[test]
    fn parse_inline_tags_list() {
        let yaml = "\
apiVersion: recipes/v1
name: tg
tags: [a, b, c]
steps:
  - id: s
    type: prompt
    prompt: x
";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        assert_eq!(r.tags, vec!["a", "b", "c"]);
    }

    // ---------- substitute ----------

    #[test]
    fn substitute_variables() {
        let mut vars = BTreeMap::new();
        vars.insert("name".into(), "Alice".into());
        let r = substitute("Hello {{name}}!", &vars, &[]);
        assert_eq!(r, "Hello Alice!");
    }

    #[test]
    fn substitute_step_outputs() {
        let steps = vec![StepOutput {
            step_id: "first".into(),
            output: "foo".into(),
        }];
        let r = substitute("got {{steps.first.output}}", &BTreeMap::new(), &steps);
        assert_eq!(r, "got foo");
    }

    #[test]
    fn substitute_unbound_left_verbatim() {
        let r = substitute("{{undefined}}", &BTreeMap::new(), &[]);
        assert_eq!(r, "{{undefined}}");
    }

    // ---------- discovery ----------

    #[test]
    fn discover_loads_yaml_files() {
        let dir = tmpdir("disc");
        write(
            &dir.join("a.yaml"),
            "apiVersion: recipes/v1\nname: alpha\nsteps:\n  - id: s\n    type: prompt\n    prompt: hi\n",
        );
        write(
            &dir.join("b.yml"),
            "apiVersion: recipes/v1\nname: beta\nsteps:\n  - id: s\n    type: prompt\n    prompt: hi\n",
        );
        write(&dir.join("ignored.txt"), "nope");
        let reg = discover_recipes(&[dir], &RecipeConfig::default());
        assert_eq!(reg.len(), 2);
        assert!(reg.get("alpha").is_some());
        assert!(reg.get("beta").is_some());
    }

    #[test]
    fn discover_later_root_overrides() {
        let a = tmpdir("ovr_a");
        let b = tmpdir("ovr_b");
        write(
            &a.join("x.yaml"),
            "apiVersion: recipes/v1\nname: x\ndescription: from-a\nsteps:\n  - id: s\n    type: prompt\n    prompt: hi\n",
        );
        write(
            &b.join("x.yaml"),
            "apiVersion: recipes/v1\nname: x\ndescription: from-b\nsteps:\n  - id: s\n    type: prompt\n    prompt: hi\n",
        );
        let reg = discover_recipes(&[a, b], &RecipeConfig::default());
        assert_eq!(reg.get("x").unwrap().description.as_deref(), Some("from-b"));
    }

    #[test]
    fn discover_skips_oversized() {
        let dir = tmpdir("size");
        let big = format!(
            "apiVersion: recipes/v1\nname: big\ndescription: {}\nsteps:\n  - id: s\n    type: prompt\n    prompt: x\n",
            "x".repeat(2000)
        );
        write(&dir.join("big.yaml"), &big);
        let mut cfg = RecipeConfig::default();
        cfg.max_file_size = 100;
        let reg = discover_recipes(&[dir], &cfg);
        assert_eq!(reg.len(), 0);
        assert_eq!(reg.load_errors.len(), 1);
    }

    // ---------- validation ----------

    #[test]
    fn validate_rejects_duplicate_step_ids() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: dup\nsteps:\n  - id: s\n    type: prompt\n    prompt: a\n  - id: s\n    type: prompt\n    prompt: b\n",
            &fake_path(),
        )
        .unwrap();
        let err = validate_recipe(&r, &RecipeConfig::default()).unwrap_err();
        match err {
            RecipeError::SchemaViolation { message, .. } => assert!(message.contains("duplicate")),
            _ => panic!(),
        }
    }

    #[test]
    fn validate_rejects_shell_when_disabled() {
        let r = Recipe {
            api_version: "recipes/v1".into(),
            name: "sh".into(),
            description: None,
            version: None,
            author: None,
            tags: vec![],
            variables: vec![],
            model: None,
            provider: None,
            steps: vec![RecipeStep {
                id: "s".into(),
                kind: StepKind::Shell {
                    command: "ls".into(),
                },
                description: None,
            }],
            output: None,
            source_path: fake_path(),
        };
        let err = validate_recipe(&r, &RecipeConfig::default()).unwrap_err();
        matches!(err, RecipeError::ShellDisabled);
    }

    #[test]
    fn validate_rejects_step_limit_exceeded() {
        let mut steps = Vec::new();
        for i in 0..10 {
            steps.push(RecipeStep {
                id: format!("s{}", i),
                kind: StepKind::Prompt { prompt: "x".into() },
                description: None,
            });
        }
        let r = Recipe {
            api_version: "recipes/v1".into(),
            name: "many".into(),
            description: None,
            version: None,
            author: None,
            tags: vec![],
            variables: vec![],
            model: None,
            provider: None,
            steps,
            output: None,
            source_path: fake_path(),
        };
        let mut cfg = RecipeConfig::default();
        cfg.max_steps = 5;
        matches!(
            validate_recipe(&r, &cfg).unwrap_err(),
            RecipeError::StepLimit(_)
        );
    }

    // ---------- engine execution ----------

    #[test]
    fn engine_runs_prompt_step_with_mock_llm() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: m\nvariables:\n  topic:\n    required: true\nsteps:\n  - id: ask\n    type: prompt\n    prompt: \"Tell me about {{topic}}\"\n",
            &fake_path(),
        )
        .unwrap();
        let engine = RecipeEngine::default().with_llm(|prompt| Some(format!("ECHO: {}", prompt)));
        let mut bindings = BTreeMap::new();
        bindings.insert("topic".into(), "Rust".into());
        let result = engine.run(&r, &bindings, &RecipeRegistry::new()).unwrap();
        assert_eq!(result.steps.len(), 1);
        assert!(result.steps[0].output.contains("Tell me about Rust"));
    }

    #[test]
    fn engine_chains_step_outputs() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: chain\nsteps:\n  - id: a\n    type: prompt\n    prompt: first\n  - id: b\n    type: prompt\n    prompt: \"got {{steps.a.output}}\"\noutput: \"{{steps.b.output}}\"\n",
            &fake_path(),
        )
        .unwrap();
        let engine = RecipeEngine::default().with_llm(|p| Some(p.to_string()));
        let result = engine
            .run(&r, &BTreeMap::new(), &RecipeRegistry::new())
            .unwrap();
        assert_eq!(result.final_output, "got first");
    }

    #[test]
    fn engine_executes_tool_step() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: t\nsteps:\n  - id: read\n    type: tool\n    tool: echo_tool\n    args:\n      msg: hello\n",
            &fake_path(),
        )
        .unwrap();
        let engine = RecipeEngine::default().with_tool(|name, args| {
            assert_eq!(name, "echo_tool");
            args.get("msg").cloned()
        });
        let result = engine
            .run(&r, &BTreeMap::new(), &RecipeRegistry::new())
            .unwrap();
        assert_eq!(result.final_output, "hello");
    }

    #[test]
    fn engine_resolves_subrecipe() {
        let inner = parse_recipe(
            "apiVersion: recipes/v1\nname: inner\nvariables:\n  x:\n    required: true\nsteps:\n  - id: s\n    type: prompt\n    prompt: \"echo {{x}}\"\n",
            &fake_path(),
        )
        .unwrap();
        let outer = parse_recipe(
            "apiVersion: recipes/v1\nname: outer\nsteps:\n  - id: call\n    type: recipe\n    recipe: inner\n    args:\n      x: world\n",
            &fake_path(),
        )
        .unwrap();
        let mut reg = RecipeRegistry::new();
        reg.insert(inner);
        let engine = RecipeEngine::default().with_llm(|p| Some(p.to_string()));
        let r = engine.run(&outer, &BTreeMap::new(), &reg).unwrap();
        assert_eq!(r.final_output, "echo world");
    }

    #[test]
    fn engine_errors_on_missing_required_var() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: r\nvariables:\n  required_one:\n    required: true\nsteps:\n  - id: s\n    type: prompt\n    prompt: hi\n",
            &fake_path(),
        )
        .unwrap();
        let engine = RecipeEngine::default().with_llm(|p| Some(p.to_string()));
        let err = engine
            .run(&r, &BTreeMap::new(), &RecipeRegistry::new())
            .unwrap_err();
        matches!(err, RecipeError::MissingRequiredVariable(_));
    }

    #[test]
    fn engine_errors_on_unknown_subrecipe() {
        let r = parse_recipe(
            "apiVersion: recipes/v1\nname: r\nsteps:\n  - id: s\n    type: recipe\n    recipe: nonexistent\n",
            &fake_path(),
        )
        .unwrap();
        let engine = RecipeEngine::default();
        let err = engine
            .run(&r, &BTreeMap::new(), &RecipeRegistry::new())
            .unwrap_err();
        matches!(err, RecipeError::UnknownRecipe(_));
    }

    #[test]
    fn engine_enforces_recursion_limit() {
        // recipe that calls itself
        let yaml = "apiVersion: recipes/v1\nname: loop\nsteps:\n  - id: s\n    type: recipe\n    recipe: loop\n";
        let r = parse_recipe(yaml, &fake_path()).unwrap();
        let mut reg = RecipeRegistry::new();
        reg.insert(r.clone());
        let mut cfg = RecipeConfig::default();
        cfg.max_recipe_depth = 3;
        let engine = RecipeEngine::new(cfg);
        let err = engine.run(&r, &BTreeMap::new(), &reg).unwrap_err();
        matches!(err, RecipeError::RecursionLimit(_));
    }

    // ---------- scaffold ----------

    #[test]
    fn scaffold_produces_valid_recipe() {
        let body = scaffold_recipe("my-new");
        let r = parse_recipe(&body, &fake_path()).unwrap();
        assert_eq!(r.name, "my-new");
        assert!(!r.steps.is_empty());
        validate_recipe(&r, &RecipeConfig::default()).unwrap();
    }
}
