//! Declarative Agent Definitions (v5 Roadmap 7.3)
//!
//! Allows agents to be defined in JSON (or a basic TOML subset) instead of code,
//! loaded and validated at runtime. This module is always available (no feature gate).
//!
//! # Example (JSON)
//!
//! ```json
//! {
//!   "agent": { "name": "research_assistant", "role": "Analyst", "model": "openai/gpt-4o" },
//!   "tools": [{ "name": "web_search", "needs_approval": true }],
//!   "memory": { "memory_type": "episodic", "max_episodes": 1000 },
//!   "guardrails": { "max_tokens_per_response": 2000, "block_pii": true },
//!   "metadata": { "author": "Lander", "version": "1.0" }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{AiError, AiResult, SerializationError};

// ── Known valid values ──────────────────────────────────────────────────────

const KNOWN_ROLES: &[&str] = &["Analyst", "Manager", "Worker", "Expert", "Validator"];
const KNOWN_AUTONOMY_LEVELS: &[&str] = &[
    "manual",
    "assisted",
    "delegated",
    "independent",
    "proactive",
];
const KNOWN_MEMORY_TYPES: &[&str] = &["episodic", "procedural", "entity", "all"];

// ── Core types ──────────────────────────────────────────────────────────────

/// Top-level declarative agent definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentDefinition {
    /// Core agent specification (name, model, parameters).
    pub agent: AgentSpec,
    /// Tool references available to this agent.
    #[serde(default)]
    pub tools: Vec<ToolRef>,
    /// Optional memory configuration.
    #[serde(default)]
    pub memory: Option<MemorySpec>,
    /// Optional guardrail configuration.
    #[serde(default)]
    pub guardrails: Option<GuardrailSpec>,
    /// Arbitrary string metadata (author, version, tags, etc.).
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

/// Specification of the agent itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentSpec {
    /// Unique agent name (must be non-empty).
    pub name: String,
    /// Role: "Analyst", "Manager", "Worker", "Expert", or "Validator".
    #[serde(default)]
    pub role: Option<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// System prompt injected at the start of every conversation.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Model in `provider/model` format, e.g. `"openai/gpt-4o"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Sampling temperature (0.0 ..= 2.0).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Nucleus sampling parameter (0.0 ..= 1.0).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Autonomy level: "manual", "assisted", "delegated", "independent", "proactive".
    #[serde(default)]
    pub autonomy_level: Option<String>,
}

/// Reference to a tool the agent may invoke.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolRef {
    /// Tool name (must be non-empty).
    pub name: String,
    /// Whether the tool requires human approval before execution.
    #[serde(default)]
    pub needs_approval: bool,
    /// Optional human-readable description of the tool.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional timeout in milliseconds for tool execution.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// Memory subsystem configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySpec {
    /// Type of memory: "episodic", "procedural", "entity", or "all".
    pub memory_type: String,
    /// Maximum number of episodic memories to retain.
    #[serde(default)]
    pub max_episodes: Option<usize>,
    /// Maximum number of procedural memories to retain.
    #[serde(default)]
    pub max_procedures: Option<usize>,
    /// Maximum number of entity memories to retain.
    #[serde(default)]
    pub max_entities: Option<usize>,
    /// Whether periodic memory consolidation is enabled.
    #[serde(default)]
    pub consolidation_enabled: bool,
}

/// Guardrail constraints applied to agent output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GuardrailSpec {
    /// Maximum tokens allowed in a single response.
    #[serde(default)]
    pub max_tokens_per_response: Option<usize>,
    /// Whether to block responses that contain PII.
    #[serde(default)]
    pub block_pii: bool,
    /// Maximum conversation turns before the agent stops.
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Regex patterns that must not appear in output.
    #[serde(default)]
    pub blocked_patterns: Vec<String>,
    /// Whether destructive tool calls require explicit approval.
    #[serde(default = "default_true")]
    pub require_approval_for_destructive: bool,
}

fn default_true() -> bool {
    true
}

// ── Validation types ────────────────────────────────────────────────────────

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WarningSeverity {
    /// Informational note, not a problem.
    Info,
    /// Potential issue that may cause unexpected behaviour.
    Warning,
    /// Hard error; the definition cannot be used as-is.
    Error,
}

/// A single validation finding produced by [`AgentDefinitionLoader::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Dotted field path, e.g. `"agent.temperature"`.
    pub field: String,
    /// Human-readable explanation.
    pub message: String,
    /// Severity level.
    pub severity: WarningSeverity,
}

// ── Loader ──────────────────────────────────────────────────────────────────

/// Loads and validates declarative agent definitions from JSON or a basic TOML
/// subset.
pub struct AgentDefinitionLoader;

impl AgentDefinitionLoader {
    /// Parse an [`AgentDefinition`] from a JSON string.
    pub fn from_json(content: &str) -> AiResult<AgentDefinition> {
        serde_json::from_str(content).map_err(|e| {
            AiError::Serialization(SerializationError {
                format: "JSON".to_string(),
                operation: "deserialize".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Parse an [`AgentDefinition`] from a basic TOML subset.
    ///
    /// Supported TOML features:
    /// - `[table]` headers
    /// - `[[array_of_tables]]` headers
    /// - String values (`key = "value"`)
    /// - Integer values (`key = 123`)
    /// - Float values (`key = 0.3`)
    /// - Boolean values (`key = true` / `key = false`)
    ///
    /// Inline tables, multi-line strings, dates, and other advanced TOML
    /// features are **not** supported.
    pub fn from_toml(content: &str) -> AiResult<AgentDefinition> {
        let value = Self::parse_toml_to_json_value(content)?;
        serde_json::from_value(value).map_err(|e| {
            AiError::Serialization(SerializationError {
                format: "TOML".to_string(),
                operation: "deserialize".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Parse from a file, auto-detecting the format by extension.
    ///
    /// `.json` is parsed as JSON; `.toml` is parsed as the basic TOML subset.
    /// Other extensions produce an error.
    pub fn from_file(path: &str) -> AiResult<AgentDefinition> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AiError::Serialization(SerializationError {
                format: "file".to_string(),
                operation: "read".to_string(),
                reason: e.to_string(),
            })
        })?;

        if path.ends_with(".json") {
            Self::from_json(&content)
        } else if path.ends_with(".toml") {
            Self::from_toml(&content)
        } else {
            Err(AiError::Serialization(SerializationError {
                format: "file".to_string(),
                operation: "detect_format".to_string(),
                reason: format!(
                    "unsupported file extension for '{}': expected .json or .toml",
                    path
                ),
            }))
        }
    }

    /// Validate an [`AgentDefinition`] and return a list of findings.
    ///
    /// If any finding has [`WarningSeverity::Error`] the definition should be
    /// considered unusable. Warnings and info-level findings are advisory.
    pub fn validate(def: &AgentDefinition) -> AiResult<Vec<ValidationWarning>> {
        let mut warnings: Vec<ValidationWarning> = Vec::new();

        // ── agent.name ──────────────────────────────────────────────────
        if def.agent.name.trim().is_empty() {
            warnings.push(ValidationWarning {
                field: "agent.name".to_string(),
                message: "agent name must be non-empty".to_string(),
                severity: WarningSeverity::Error,
            });
        }

        // ── agent.temperature ───────────────────────────────────────────
        if let Some(t) = def.agent.temperature {
            if !(0.0..=2.0).contains(&t) {
                warnings.push(ValidationWarning {
                    field: "agent.temperature".to_string(),
                    message: format!("temperature {} is out of range [0.0, 2.0]", t),
                    severity: WarningSeverity::Error,
                });
            }
        }

        // ── agent.top_p ────────────────────────────────────────────────
        if let Some(p) = def.agent.top_p {
            if !(0.0..=1.0).contains(&p) {
                warnings.push(ValidationWarning {
                    field: "agent.top_p".to_string(),
                    message: format!("top_p {} is out of range [0.0, 1.0]", p),
                    severity: WarningSeverity::Error,
                });
            }
        }

        // ── agent.max_tokens ───────────────────────────────────────────
        if let Some(mt) = def.agent.max_tokens {
            if mt == 0 {
                warnings.push(ValidationWarning {
                    field: "agent.max_tokens".to_string(),
                    message: "max_tokens must be > 0".to_string(),
                    severity: WarningSeverity::Error,
                });
            }
        }

        // ── agent.role ─────────────────────────────────────────────────
        if let Some(ref role) = def.agent.role {
            if !KNOWN_ROLES.contains(&role.as_str()) {
                warnings.push(ValidationWarning {
                    field: "agent.role".to_string(),
                    message: format!(
                        "unknown role '{}'; known roles: {}",
                        role,
                        KNOWN_ROLES.join(", ")
                    ),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        // ── agent.autonomy_level ───────────────────────────────────────
        if let Some(ref level) = def.agent.autonomy_level {
            if !KNOWN_AUTONOMY_LEVELS.contains(&level.as_str()) {
                warnings.push(ValidationWarning {
                    field: "agent.autonomy_level".to_string(),
                    message: format!(
                        "unknown autonomy level '{}'; known levels: {}",
                        level,
                        KNOWN_AUTONOMY_LEVELS.join(", ")
                    ),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        // ── tools ──────────────────────────────────────────────────────
        for (i, tool) in def.tools.iter().enumerate() {
            if tool.name.trim().is_empty() {
                warnings.push(ValidationWarning {
                    field: format!("tools[{}].name", i),
                    message: "tool name must be non-empty".to_string(),
                    severity: WarningSeverity::Error,
                });
            }
        }

        // ── memory ─────────────────────────────────────────────────────
        if let Some(ref mem) = def.memory {
            if !KNOWN_MEMORY_TYPES.contains(&mem.memory_type.as_str()) {
                warnings.push(ValidationWarning {
                    field: "memory.memory_type".to_string(),
                    message: format!(
                        "unknown memory type '{}'; known types: {}",
                        mem.memory_type,
                        KNOWN_MEMORY_TYPES.join(", ")
                    ),
                    severity: WarningSeverity::Error,
                });
            }
        }

        // ── guardrails ─────────────────────────────────────────────────
        if let Some(ref gr) = def.guardrails {
            if let Some(mt) = gr.max_tokens_per_response {
                if mt == 0 {
                    warnings.push(ValidationWarning {
                        field: "guardrails.max_tokens_per_response".to_string(),
                        message: "max_tokens_per_response must be > 0".to_string(),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        Ok(warnings)
    }

    // ── Minimal TOML-to-JSON parser ─────────────────────────────────────

    /// Convert the basic TOML subset into a [`serde_json::Value`] tree.
    fn parse_toml_to_json_value(content: &str) -> AiResult<serde_json::Value> {
        use serde_json::{Map, Value};

        let mut root = Map::new();
        let mut current_table: Option<String> = None;
        let mut current_array_table: Option<String> = None;

        for (line_no, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            // [[array_of_tables]]
            if line.starts_with("[[") && line.ends_with("]]") {
                let table_name = line[2..line.len() - 2].trim().to_string();
                current_table = None;
                current_array_table = Some(table_name.clone());

                // Ensure the array exists in root.
                if !root.contains_key(&table_name) {
                    root.insert(table_name.clone(), Value::Array(Vec::new()));
                }
                // Push a new empty object into the array.
                if let Some(Value::Array(arr)) = root.get_mut(&table_name) {
                    arr.push(Value::Object(Map::new()));
                }
                continue;
            }

            // [table]
            if line.starts_with('[') && line.ends_with(']') {
                let table_name = line[1..line.len() - 1].trim().to_string();
                current_array_table = None;
                current_table = Some(table_name.clone());

                if !root.contains_key(&table_name) {
                    root.insert(table_name, Value::Object(Map::new()));
                }
                continue;
            }

            // key = value
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim().to_string();
                let val_str = line[eq_pos + 1..].trim();
                let value = Self::parse_toml_value(val_str, line_no)?;

                // Determine the target map.
                if let Some(ref arr_name) = current_array_table {
                    if let Some(Value::Array(arr)) = root.get_mut(arr_name) {
                        if let Some(Value::Object(obj)) = arr.last_mut() {
                            obj.insert(key, value);
                        }
                    }
                } else if let Some(ref tbl_name) = current_table {
                    if let Some(Value::Object(obj)) = root.get_mut(tbl_name) {
                        obj.insert(key, value);
                    }
                } else {
                    root.insert(key, value);
                }
                continue;
            }

            // Unrecognised line.
            return Err(AiError::Serialization(SerializationError {
                format: "TOML".to_string(),
                operation: "parse".to_string(),
                reason: format!("unrecognised syntax at line {}: {}", line_no + 1, raw_line),
            }));
        }

        Ok(Value::Object(root))
    }

    /// Parse a single TOML value token into a [`serde_json::Value`].
    fn parse_toml_value(s: &str, line_no: usize) -> AiResult<serde_json::Value> {
        use serde_json::Value;

        // String literal: "..."
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            let inner = &s[1..s.len() - 1];
            // Basic escape processing.
            let unescaped = inner
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\t", "\t");
            return Ok(Value::String(unescaped));
        }

        // Boolean
        if s == "true" {
            return Ok(Value::Bool(true));
        }
        if s == "false" {
            return Ok(Value::Bool(false));
        }

        // Inline array: ["a", "b"]
        if s.starts_with('[') && s.ends_with(']') {
            let inner = s[1..s.len() - 1].trim();
            if inner.is_empty() {
                return Ok(Value::Array(Vec::new()));
            }
            let mut items = Vec::new();
            for item in Self::split_respecting_quotes(inner) {
                items.push(Self::parse_toml_value(item.trim(), line_no)?);
            }
            return Ok(Value::Array(items));
        }

        // Integer (try before float so "42" is parsed as Number(42) not 42.0)
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Value::Number(serde_json::Number::from(i)));
        }

        // Float
        if let Ok(f) = s.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(f) {
                return Ok(Value::Number(n));
            }
        }

        Err(AiError::Serialization(SerializationError {
            format: "TOML".to_string(),
            operation: "parse_value".to_string(),
            reason: format!("cannot parse value at line {}: {}", line_no + 1, s),
        }))
    }

    /// Split a string by commas while respecting double-quoted strings.
    fn split_respecting_quotes(s: &str) -> Vec<&str> {
        let mut parts = Vec::new();
        let mut start = 0;
        let mut in_quotes = false;
        for (i, ch) in s.char_indices() {
            match ch {
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => {
                    parts.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        parts.push(&s[start..]);
        parts
    }
}

// ── Display implementations ─────────────────────────────────────────────────

impl std::fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarningSeverity::Info => write!(f, "INFO"),
            WarningSeverity::Warning => write!(f, "WARNING"),
            WarningSeverity::Error => write!(f, "ERROR"),
        }
    }
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

// ── Deployment Profiles (v6 Phase 10.2) ─────────────────────────────────────

/// Deployment profile for an agent, describing the sandbox backend,
/// resource limits, networking policy, health checks, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentProfile {
    /// Profile name (e.g. "development", "staging", "production").
    pub name: String,
    /// Sandbox backend to use (e.g. "docker", "podman", "wasm", "process").
    pub sandbox_backend: String,
    /// Resource limits for the deployed agent.
    pub resources: AgentResourceLimits,
    /// Networking constraints.
    pub networking: DeploymentNetworking,
    /// Optional health check configuration.
    pub health_check: Option<AgentHealthCheck>,
    /// Environment variables injected into the sandbox.
    pub environment: HashMap<String, String>,
    /// Arbitrary labels/tags for the deployment.
    pub labels: HashMap<String, String>,
}

/// Resource limits for a deployed agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResourceLimits {
    /// Maximum memory in megabytes.
    pub max_memory_mb: u64,
    /// Maximum CPU usage as a percentage (0.0 ..= 100.0).
    pub max_cpu_percent: f64,
    /// Maximum disk usage in megabytes.
    pub max_disk_mb: u64,
    /// Maximum runtime in seconds.
    pub max_runtime_secs: u64,
}

impl Default for AgentResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            max_cpu_percent: 100.0,
            max_disk_mb: 1024,
            max_runtime_secs: 3600,
        }
    }
}

/// Networking constraints for a deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentNetworking {
    /// List of allowed destination hosts (empty = no restrictions).
    pub allowed_hosts: Vec<String>,
    /// DNS resolution policy.
    pub dns_policy: DnsPolicy,
    /// Optional HTTP proxy URL.
    pub proxy: Option<String>,
}

/// DNS resolution policy for deployed agents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DnsPolicy {
    /// Use the host's DNS settings.
    Default,
    /// Resolve via cluster DNS first, then fall back to host DNS.
    ClusterFirst,
    /// No DNS resolution (fully isolated).
    None,
}

/// Health check configuration for a deployed agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealthCheck {
    /// HTTP endpoint to probe (e.g. "/health").
    pub endpoint: String,
    /// Interval between probes in seconds.
    pub interval_secs: u64,
    /// Per-probe timeout in seconds.
    pub timeout_secs: u64,
    /// Number of consecutive failures before marking unhealthy.
    pub unhealthy_threshold: u32,
}

impl Default for AgentHealthCheck {
    fn default() -> Self {
        Self {
            endpoint: "/health".to_string(),
            interval_secs: 30,
            timeout_secs: 5,
            unhealthy_threshold: 3,
        }
    }
}

/// Loads, serializes, and validates [`DeploymentProfile`]s.
pub struct ProfileLoader;

impl ProfileLoader {
    /// Deserialize a [`DeploymentProfile`] from a JSON string.
    pub fn from_json(json: &str) -> Result<DeploymentProfile, AiError> {
        serde_json::from_str(json).map_err(|e| {
            AiError::Serialization(SerializationError {
                format: "JSON".to_string(),
                operation: "deserialize".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Serialize a [`DeploymentProfile`] to a JSON string.
    pub fn to_json(profile: &DeploymentProfile) -> Result<String, AiError> {
        serde_json::to_string_pretty(profile).map_err(|e| {
            AiError::Serialization(SerializationError {
                format: "JSON".to_string(),
                operation: "serialize".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Validate a [`DeploymentProfile`] and return a list of human-readable
    /// error messages. An empty list means the profile is valid.
    pub fn validate(profile: &DeploymentProfile) -> Vec<String> {
        let mut errors = Vec::new();

        if profile.resources.max_memory_mb == 0 {
            errors.push("max_memory_mb must be > 0".to_string());
        }
        if profile.resources.max_cpu_percent <= 0.0
            || profile.resources.max_cpu_percent > 100.0
        {
            errors.push(
                "max_cpu_percent must be > 0 and <= 100".to_string(),
            );
        }
        if profile.resources.max_runtime_secs == 0 {
            errors.push("max_runtime_secs must be > 0".to_string());
        }
        if let Some(ref hc) = profile.health_check {
            if hc.interval_secs <= hc.timeout_secs {
                errors.push(
                    "health_check interval_secs must be greater than timeout_secs".to_string(),
                );
            }
        }

        errors
    }

    /// Create a [`DeploymentProfile`] populated with sensible defaults.
    pub fn with_defaults() -> DeploymentProfile {
        DeploymentProfile {
            name: "default".to_string(),
            sandbox_backend: "docker".to_string(),
            resources: AgentResourceLimits::default(),
            networking: DeploymentNetworking {
                allowed_hosts: Vec::new(),
                dns_policy: DnsPolicy::Default,
                proxy: None,
            },
            health_check: Some(AgentHealthCheck::default()),
            environment: HashMap::new(),
            labels: HashMap::new(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a full JSON definition used by several tests.
    fn full_json() -> &'static str {
        r#"{
            "agent": {
                "name": "research_assistant",
                "role": "Analyst",
                "description": "A research analyst that finds and synthesizes information",
                "system_prompt": "You are a research analyst...",
                "model": "openai/gpt-4o",
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.9,
                "autonomy_level": "assisted"
            },
            "tools": [
                {
                    "name": "web_search",
                    "needs_approval": true,
                    "description": "Search the web for information",
                    "timeout_ms": 30000
                },
                {
                    "name": "code_execute",
                    "needs_approval": false
                }
            ],
            "memory": {
                "memory_type": "episodic",
                "max_episodes": 1000,
                "consolidation_enabled": true
            },
            "guardrails": {
                "max_tokens_per_response": 2000,
                "block_pii": true,
                "max_turns": 50,
                "blocked_patterns": ["password", "secret"],
                "require_approval_for_destructive": true
            },
            "metadata": {
                "author": "Lander",
                "version": "1.0",
                "tags": "research, analysis"
            }
        }"#
    }

    // ── 1. Parse JSON agent definition (complete) ───────────────────────

    #[test]
    fn test_parse_json_complete() {
        let def = AgentDefinitionLoader::from_json(full_json()).expect("parse complete JSON");
        assert_eq!(def.agent.name, "research_assistant");
        assert_eq!(def.agent.role.as_deref(), Some("Analyst"));
        assert_eq!(def.agent.model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(def.agent.temperature, Some(0.3));
        assert_eq!(def.agent.max_tokens, Some(4096));
        assert_eq!(def.agent.top_p, Some(0.9));
        assert_eq!(def.agent.autonomy_level.as_deref(), Some("assisted"));
        assert_eq!(def.tools.len(), 2);
        assert!(def.tools[0].needs_approval);
        assert!(!def.tools[1].needs_approval);
        assert!(def.memory.is_some());
        assert!(def.guardrails.is_some());
        assert!(def.metadata.is_some());
    }

    // ── 2. Parse JSON with minimal fields (only name) ───────────────────

    #[test]
    fn test_parse_json_minimal() {
        let json = r#"{ "agent": { "name": "tiny" } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse minimal JSON");
        assert_eq!(def.agent.name, "tiny");
        assert!(def.agent.role.is_none());
        assert!(def.agent.model.is_none());
        assert!(def.tools.is_empty());
        assert!(def.memory.is_none());
        assert!(def.guardrails.is_none());
        assert!(def.metadata.is_none());
    }

    // ── 3. Parse JSON with all optional fields ──────────────────────────

    #[test]
    fn test_parse_json_all_optional_fields() {
        let def = AgentDefinitionLoader::from_json(full_json()).expect("parse JSON for all_optional_fields");
        // Check all optional fields on agent
        assert!(def.agent.description.is_some());
        assert!(def.agent.system_prompt.is_some());
        // Check memory optional fields
        let mem = def.memory.as_ref().expect("memory section in all_optional_fields");
        assert_eq!(mem.memory_type, "episodic");
        assert_eq!(mem.max_episodes, Some(1000));
        assert!(mem.consolidation_enabled);
        // Check guardrails optional fields
        let gr = def.guardrails.as_ref().expect("guardrails section in all_optional_fields");
        assert_eq!(gr.max_tokens_per_response, Some(2000));
        assert!(gr.block_pii);
        assert_eq!(gr.max_turns, Some(50));
        assert_eq!(gr.blocked_patterns, vec!["password", "secret"]);
        assert!(gr.require_approval_for_destructive);
    }

    // ── 4. Validate valid definition (no warnings) ──────────────────────

    #[test]
    fn test_validate_valid_definition() {
        let def = AgentDefinitionLoader::from_json(full_json()).expect("parse for validate_valid_definition");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate valid definition");
        // Should produce zero errors or warnings (all fields are valid).
        assert!(
            warnings.is_empty(),
            "expected no warnings, got: {:?}",
            warnings
        );
    }

    // ── 5. Validate invalid temperature (out of range) ──────────────────

    #[test]
    fn test_validate_invalid_temperature() {
        let json = r#"{ "agent": { "name": "hot", "temperature": 3.5 } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse invalid temperature JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate invalid temperature");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].field, "agent.temperature");
        assert_eq!(warnings[0].severity, WarningSeverity::Error);
    }

    // ── 6. Validate empty name (error) ──────────────────────────────────

    #[test]
    fn test_validate_empty_name() {
        let json = r#"{ "agent": { "name": "" } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse empty name JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate empty name");
        assert!(warnings.iter().any(|w| w.field == "agent.name"
            && w.severity == WarningSeverity::Error));
    }

    // ── 7. Validate unknown role (warning) ──────────────────────────────

    #[test]
    fn test_validate_unknown_role() {
        let json = r#"{ "agent": { "name": "x", "role": "Wizard" } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse unknown role JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate unknown role");
        assert!(warnings.iter().any(|w| w.field == "agent.role"
            && w.severity == WarningSeverity::Warning));
    }

    // ── 8. Validate unknown autonomy level (warning) ────────────────────

    #[test]
    fn test_validate_unknown_autonomy_level() {
        let json = r#"{ "agent": { "name": "x", "autonomy_level": "rogue" } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse unknown autonomy JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate unknown autonomy");
        assert!(warnings.iter().any(|w| w.field == "agent.autonomy_level"
            && w.severity == WarningSeverity::Warning));
    }

    // ── 9. Validate tool with empty name (error) ────────────────────────

    #[test]
    fn test_validate_tool_empty_name() {
        let json = r#"{
            "agent": { "name": "x" },
            "tools": [{ "name": "" }]
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse tool empty name JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate tool empty name");
        assert!(warnings.iter().any(|w| w.field == "tools[0].name"
            && w.severity == WarningSeverity::Error));
    }

    // ── 10. Validate memory with unknown type (error) ───────────────────

    #[test]
    fn test_validate_memory_unknown_type() {
        let json = r#"{
            "agent": { "name": "x" },
            "memory": { "memory_type": "quantum" }
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse unknown memory type JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate unknown memory type");
        assert!(warnings.iter().any(|w| w.field == "memory.memory_type"
            && w.severity == WarningSeverity::Error));
    }

    // ── 11. Round-trip: serialize then deserialize ───────────────────────

    #[test]
    fn test_roundtrip_serialize_deserialize() {
        let original = AgentDefinitionLoader::from_json(full_json()).expect("parse for roundtrip");
        let serialized = serde_json::to_string(&original).expect("serialize AgentDefinition");
        let restored: AgentDefinition =
            serde_json::from_str(&serialized).expect("deserialize AgentDefinition roundtrip");
        assert_eq!(original, restored);
    }

    // ── 12. from_file with .json extension ──────────────────────────────

    #[test]
    fn test_from_file_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("agent_def_test_12.json");
        std::fs::write(&path, full_json()).expect("write temp JSON file for from_file test");

        let def =
            AgentDefinitionLoader::from_file(path.to_str().expect("temp path to str"))
                .expect("load agent definition from file");
        assert_eq!(def.agent.name, "research_assistant");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    // ── 13. Multiple tools with approval settings ───────────────────────

    #[test]
    fn test_multiple_tools_approval() {
        let json = r#"{
            "agent": { "name": "multi" },
            "tools": [
                { "name": "read", "needs_approval": false },
                { "name": "write", "needs_approval": true },
                { "name": "delete", "needs_approval": true, "timeout_ms": 5000 }
            ]
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse multiple tools JSON");
        assert_eq!(def.tools.len(), 3);
        assert!(!def.tools[0].needs_approval);
        assert!(def.tools[1].needs_approval);
        assert!(def.tools[2].needs_approval);
        assert_eq!(def.tools[2].timeout_ms, Some(5000));
    }

    // ── 14. Guardrail defaults ──────────────────────────────────────────

    #[test]
    fn test_guardrail_defaults() {
        let json = r#"{
            "agent": { "name": "x" },
            "guardrails": {}
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse empty guardrails JSON");
        let gr = def.guardrails.expect("guardrails section in defaults test");
        assert!(gr.max_tokens_per_response.is_none());
        assert!(!gr.block_pii);
        assert!(gr.max_turns.is_none());
        assert!(gr.blocked_patterns.is_empty());
        assert!(gr.require_approval_for_destructive); // defaults to true
    }

    // ── 15. Metadata handling ───────────────────────────────────────────

    #[test]
    fn test_metadata_handling() {
        let json = r#"{
            "agent": { "name": "meta" },
            "metadata": {
                "author": "Lander",
                "version": "2.0",
                "custom_key": "custom_value"
            }
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse metadata JSON");
        let meta = def.metadata.expect("metadata section in handling test");
        assert_eq!(meta.get("author").map(|s| s.as_str()), Some("Lander"));
        assert_eq!(meta.get("version").map(|s| s.as_str()), Some("2.0"));
        assert_eq!(
            meta.get("custom_key").map(|s| s.as_str()),
            Some("custom_value")
        );
        assert_eq!(meta.len(), 3);
    }

    // ── Bonus: TOML parser smoke test ───────────────────────────────────

    #[test]
    fn test_parse_toml_basic() {
        let toml = r#"
[agent]
name = "research_assistant"
role = "Analyst"
description = "A research analyst"
system_prompt = "You are a research analyst..."
model = "openai/gpt-4o"
temperature = 0.3
max_tokens = 4096
top_p = 0.9

[[tools]]
name = "web_search"
needs_approval = true
description = "Search the web"

[[tools]]
name = "code_execute"
needs_approval = false

[memory]
memory_type = "episodic"
max_episodes = 1000
consolidation_enabled = true

[guardrails]
max_tokens_per_response = 2000
block_pii = true
max_turns = 50
blocked_patterns = ["password", "secret"]

[metadata]
author = "Lander"
version = "1.0"
tags = "research, analysis"
"#;
        let def = AgentDefinitionLoader::from_toml(toml).expect("parse TOML basic definition");
        assert_eq!(def.agent.name, "research_assistant");
        assert_eq!(def.agent.role.as_deref(), Some("Analyst"));
        assert_eq!(def.agent.temperature, Some(0.3));
        assert_eq!(def.tools.len(), 2);
        assert!(def.tools[0].needs_approval);
        assert!(!def.tools[1].needs_approval);
        let mem = def.memory.as_ref().expect("memory section in TOML test");
        assert_eq!(mem.memory_type, "episodic");
        assert_eq!(mem.max_episodes, Some(1000));
        let gr = def.guardrails.as_ref().expect("guardrails section in TOML test");
        assert_eq!(gr.max_tokens_per_response, Some(2000));
        assert!(gr.block_pii);
    }

    // ── Bonus: validate guardrails max_tokens_per_response = 0 ──────────

    #[test]
    fn test_validate_guardrail_zero_max_tokens() {
        let json = r#"{
            "agent": { "name": "x" },
            "guardrails": { "max_tokens_per_response": 0 }
        }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse zero max_tokens guardrail JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate zero max_tokens guardrail");
        assert!(warnings
            .iter()
            .any(|w| w.field == "guardrails.max_tokens_per_response"
                && w.severity == WarningSeverity::Error));
    }

    // ── Bonus: validate top_p out of range ──────────────────────────────

    #[test]
    fn test_validate_top_p_out_of_range() {
        let json = r#"{ "agent": { "name": "x", "top_p": 1.5 } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse top_p out of range JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate top_p out of range");
        assert!(warnings
            .iter()
            .any(|w| w.field == "agent.top_p" && w.severity == WarningSeverity::Error));
    }

    // ── Bonus: validate max_tokens = 0 ──────────────────────────────────

    #[test]
    fn test_validate_max_tokens_zero() {
        let json = r#"{ "agent": { "name": "x", "max_tokens": 0 } }"#;
        let def = AgentDefinitionLoader::from_json(json).expect("parse zero max_tokens JSON");
        let warnings = AgentDefinitionLoader::validate(&def).expect("validate zero max_tokens");
        assert!(warnings
            .iter()
            .any(|w| w.field == "agent.max_tokens" && w.severity == WarningSeverity::Error));
    }

    // ── v6 Phase 10.2: Deployment Profiles tests ─────────────────────────

    #[test]
    fn test_deployment_profile_construction() {
        let profile = DeploymentProfile {
            name: "staging".to_string(),
            sandbox_backend: "docker".to_string(),
            resources: AgentResourceLimits::default(),
            networking: DeploymentNetworking {
                allowed_hosts: vec!["api.example.com".to_string()],
                dns_policy: DnsPolicy::Default,
                proxy: None,
            },
            health_check: None,
            environment: HashMap::new(),
            labels: HashMap::new(),
        };
        assert_eq!(profile.name, "staging");
        assert_eq!(profile.sandbox_backend, "docker");
        assert!(profile.health_check.is_none());
        assert_eq!(profile.networking.allowed_hosts.len(), 1);
    }

    #[test]
    fn test_agent_resource_limits_defaults() {
        let limits = AgentResourceLimits::default();
        assert_eq!(limits.max_memory_mb, 512);
        assert!((limits.max_cpu_percent - 100.0).abs() < f64::EPSILON);
        assert_eq!(limits.max_disk_mb, 1024);
        assert_eq!(limits.max_runtime_secs, 3600);
    }

    #[test]
    fn test_deployment_networking_construction() {
        let net = DeploymentNetworking {
            allowed_hosts: vec!["a.com".to_string(), "b.com".to_string()],
            dns_policy: DnsPolicy::ClusterFirst,
            proxy: Some("http://proxy:8080".to_string()),
        };
        assert_eq!(net.allowed_hosts.len(), 2);
        assert_eq!(net.dns_policy, DnsPolicy::ClusterFirst);
        assert_eq!(net.proxy.as_deref(), Some("http://proxy:8080"));
    }

    #[test]
    fn test_dns_policy_all_variants() {
        assert_eq!(DnsPolicy::Default, DnsPolicy::Default);
        assert_eq!(DnsPolicy::ClusterFirst, DnsPolicy::ClusterFirst);
        assert_eq!(DnsPolicy::None, DnsPolicy::None);
        assert_ne!(DnsPolicy::Default, DnsPolicy::ClusterFirst);
        assert_ne!(DnsPolicy::ClusterFirst, DnsPolicy::None);
    }

    #[test]
    fn test_agent_health_check_defaults() {
        let hc = AgentHealthCheck::default();
        assert_eq!(hc.endpoint, "/health");
        assert_eq!(hc.interval_secs, 30);
        assert_eq!(hc.timeout_secs, 5);
        assert_eq!(hc.unhealthy_threshold, 3);
    }

    #[test]
    fn test_profile_loader_from_json_valid() {
        let json = r#"{
            "name": "test",
            "sandbox_backend": "docker",
            "resources": { "max_memory_mb": 256, "max_cpu_percent": 50.0, "max_disk_mb": 512, "max_runtime_secs": 1800 },
            "networking": { "allowed_hosts": [], "dns_policy": "Default", "proxy": null },
            "health_check": null,
            "environment": {},
            "labels": {}
        }"#;
        let profile = ProfileLoader::from_json(json).expect("parse profile JSON");
        assert_eq!(profile.name, "test");
        assert_eq!(profile.resources.max_memory_mb, 256);
    }

    #[test]
    fn test_profile_loader_to_json_roundtrip() {
        let original = ProfileLoader::with_defaults();
        let json = ProfileLoader::to_json(&original).expect("serialize DeploymentProfile");
        let restored = ProfileLoader::from_json(&json).expect("deserialize DeploymentProfile roundtrip");
        assert_eq!(restored.name, original.name);
        assert_eq!(restored.sandbox_backend, original.sandbox_backend);
        assert_eq!(
            restored.resources.max_memory_mb,
            original.resources.max_memory_mb
        );
    }

    #[test]
    fn test_profile_loader_validate_valid() {
        let profile = ProfileLoader::with_defaults();
        let errors = ProfileLoader::validate(&profile);
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_profile_loader_validate_invalid() {
        let mut profile = ProfileLoader::with_defaults();
        profile.resources.max_memory_mb = 0;
        profile.resources.max_cpu_percent = 0.0;
        profile.resources.max_runtime_secs = 0;
        profile.health_check = Some(AgentHealthCheck {
            endpoint: "/health".to_string(),
            interval_secs: 5,
            timeout_secs: 10, // interval <= timeout → error
            unhealthy_threshold: 3,
        });
        let errors = ProfileLoader::validate(&profile);
        assert!(errors.iter().any(|e| e.contains("max_memory_mb")));
        assert!(errors.iter().any(|e| e.contains("max_cpu_percent")));
        assert!(errors.iter().any(|e| e.contains("max_runtime_secs")));
        assert!(errors.iter().any(|e| e.contains("interval_secs")));
    }

    #[test]
    fn test_profile_loader_with_defaults_valid() {
        let profile = ProfileLoader::with_defaults();
        assert_eq!(profile.name, "default");
        assert_eq!(profile.sandbox_backend, "docker");
        assert!(profile.health_check.is_some());
        assert!(profile.environment.is_empty());
        assert!(profile.labels.is_empty());
        // The default profile must pass validation
        let errors = ProfileLoader::validate(&profile);
        assert!(errors.is_empty());
    }
}
