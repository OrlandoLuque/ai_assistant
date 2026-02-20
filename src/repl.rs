//! Interactive REPL/CLI for the AI assistant.
//!
//! Provides a command-line interface for interacting with the assistant.
//! This module is pure Rust with no external dependencies beyond `std` and `serde`/`serde_json`
//! (already crate dependencies) for session serialization.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

// =============================================================================
// ReplCommand
// =============================================================================

/// Parsed REPL command from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    /// Show help text listing all available commands.
    Help,
    /// Exit the REPL session.
    Exit,
    /// List available models.
    Models,
    /// Show or modify current configuration.
    Config,
    /// Clear conversation history.
    Clear,
    /// Save the current session to a file.
    Save(String),
    /// Load a session from a file.
    Load(String),
    /// Set or show the current conversation template.
    Template(String),
    /// Switch the active model.
    Model(String),
    /// Show conversation history.
    History,
    /// Show cost tracking dashboard report.
    Cost,
    /// Unrecognized command.
    Unknown(String),
}

impl ReplCommand {
    /// Parse user input into a `ReplCommand`.
    ///
    /// Commands start with `/`. Anything else is not a command
    /// and should be handled as a regular message by the caller.
    pub fn parse(input: &str) -> Option<ReplCommand> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        let mut parts = trimmed[1..].splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim().to_string();

        Some(match cmd.to_lowercase().as_str() {
            "help" | "h" | "?" => ReplCommand::Help,
            "exit" | "quit" | "q" => ReplCommand::Exit,
            "models" => ReplCommand::Models,
            "config" => ReplCommand::Config,
            "clear" => ReplCommand::Clear,
            "save" => ReplCommand::Save(arg),
            "load" => ReplCommand::Load(arg),
            "template" => ReplCommand::Template(arg),
            "model" => ReplCommand::Model(arg),
            "history" => ReplCommand::History,
            "cost" => ReplCommand::Cost,
            _ => ReplCommand::Unknown(cmd.to_string()),
        })
    }
}

// =============================================================================
// ReplAction
// =============================================================================

/// Action to take after processing user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplAction {
    /// User wants to send this message to the AI.
    SendMessage(String),
    /// A command was parsed and should be executed.
    ExecuteCommand(ReplCommand),
    /// Empty input or continuation — do nothing.
    Continue,
    /// User wants to quit.
    Exit,
}

// =============================================================================
// ReplError
// =============================================================================

/// Errors that can occur during REPL operations.
#[derive(Debug)]
pub enum ReplError {
    /// An I/O error occurred.
    IoError(io::Error),
    /// Serialization or deserialization failed.
    SerializationError(String),
    /// The requested session file was not found.
    SessionNotFound(String),
}

impl fmt::Display for ReplError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplError::IoError(e) => write!(f, "I/O error: {}", e),
            ReplError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            ReplError::SessionNotFound(path) => write!(f, "Session not found: {}", path),
        }
    }
}

impl std::error::Error for ReplError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReplError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ReplError {
    fn from(err: io::Error) -> Self {
        ReplError::IoError(err)
    }
}

impl From<serde_json::Error> for ReplError {
    fn from(err: serde_json::Error) -> Self {
        ReplError::SerializationError(err.to_string())
    }
}

// =============================================================================
// ReplConfig
// =============================================================================

/// Configuration for the REPL engine.
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// The prompt string displayed before user input (e.g., "> ").
    pub prompt_string: String,
    /// Maximum number of entries to keep in command history.
    pub max_history: usize,
    /// Whether to display performance metrics after responses.
    pub show_metrics: bool,
    /// Whether to display timestamps alongside messages.
    pub show_timestamps: bool,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt_string: "> ".to_string(),
            max_history: 1000,
            show_metrics: false,
            show_timestamps: false,
        }
    }
}

// =============================================================================
// ReplSession (serializable)
// =============================================================================

/// A serializable conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplSession {
    /// Messages as (role, content) pairs.
    pub messages: Vec<(String, String)>,
    /// The currently active model identifier.
    pub model: String,
    /// An optional conversation template name.
    pub template: Option<String>,
}

impl Default for ReplSession {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            model: "default".to_string(),
            template: None,
        }
    }
}

// =============================================================================
// ReplEngine
// =============================================================================

/// The core REPL engine that processes input and manages session state.
pub struct ReplEngine {
    /// Current configuration.
    config: ReplConfig,
    /// Current conversation session.
    session: ReplSession,
    /// History of raw command inputs for recall.
    command_history: Vec<String>,
}

impl ReplEngine {
    /// Create a new REPL engine with the given configuration.
    pub fn new(config: ReplConfig) -> Self {
        Self {
            config,
            session: ReplSession::default(),
            command_history: Vec::new(),
        }
    }

    /// Process a line of user input and return the appropriate action.
    ///
    /// - Empty or whitespace-only input returns `Continue`.
    /// - Input starting with `/` is parsed as a command.
    /// - `/exit` and `/quit` return `Exit` directly.
    /// - Other commands return `ExecuteCommand`.
    /// - Everything else returns `SendMessage`.
    pub fn process_input(&mut self, input: &str) -> ReplAction {
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return ReplAction::Continue;
        }

        // Record in command history
        if self.command_history.len() >= self.config.max_history {
            self.command_history.remove(0);
        }
        self.command_history.push(trimmed.to_string());

        // Try parsing as a command
        if let Some(cmd) = ReplCommand::parse(trimmed) {
            match cmd {
                ReplCommand::Exit => ReplAction::Exit,
                other => ReplAction::ExecuteCommand(other),
            }
        } else {
            ReplAction::SendMessage(trimmed.to_string())
        }
    }

    /// Add a message to the conversation session.
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.session
            .messages
            .push((role.to_string(), content.to_string()));
    }

    /// Return a slice of all messages in the current session.
    pub fn history(&self) -> &[(String, String)] {
        &self.session.messages
    }

    /// Clear the conversation history (messages only, not the model or template).
    pub fn clear_history(&mut self) {
        self.session.messages.clear();
    }

    /// Save the current session to a JSON file at the given path.
    pub fn save_session(&self, path: &str) -> Result<(), ReplError> {
        let json = serde_json::to_string_pretty(&self.session)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a session from a JSON file, replacing the current session.
    pub fn load_session(&mut self, path: &str) -> Result<(), ReplError> {
        if !Path::new(path).exists() {
            return Err(ReplError::SessionNotFound(path.to_string()));
        }
        let json = fs::read_to_string(path)?;
        self.session = serde_json::from_str(&json)?;
        Ok(())
    }

    /// Return a formatted help string listing all available commands.
    pub fn format_help() -> String {
        let mut help = String::new();
        help.push_str("Available commands:\n");
        help.push_str("  /help, /h, /?       Show this help message\n");
        help.push_str("  /exit, /quit, /q    Exit the REPL\n");
        help.push_str("  /models             List available models\n");
        help.push_str("  /model <name>       Switch to a different model\n");
        help.push_str("  /template <name>    Set a conversation template\n");
        help.push_str("  /config             Show current configuration\n");
        help.push_str("  /history            Show conversation history\n");
        help.push_str("  /clear              Clear conversation history\n");
        help.push_str("  /save <path>        Save session to a file\n");
        help.push_str("  /load <path>        Load session from a file\n");
        help.push_str("  /cost               Show cost tracking report\n");
        help.push_str("\nAnything else is sent as a message to the AI.");
        help
    }

    /// Return a formatted string showing the current configuration.
    pub fn format_config(&self) -> String {
        let mut out = String::new();
        out.push_str("Current configuration:\n");
        out.push_str(&format!("  Prompt:          {}\n", self.config.prompt_string));
        out.push_str(&format!("  Max history:     {}\n", self.config.max_history));
        out.push_str(&format!("  Show metrics:    {}\n", self.config.show_metrics));
        out.push_str(&format!(
            "  Show timestamps: {}\n",
            self.config.show_timestamps
        ));
        out.push_str(&format!("  Model:           {}\n", self.session.model));
        out.push_str(&format!(
            "  Template:        {}",
            self.session
                .template
                .as_deref()
                .unwrap_or("(none)")
        ));
        out
    }

    /// Set the active model.
    pub fn set_model(&mut self, model: &str) {
        self.session.model = model.to_string();
    }

    /// Get the currently active model identifier.
    pub fn current_model(&self) -> &str {
        &self.session.model
    }

    /// Set the active template.
    pub fn set_template(&mut self, template: &str) {
        self.session.template = Some(template.to_string());
    }

    /// Get the currently active template, if any.
    pub fn current_template(&self) -> Option<&str> {
        self.session.template.as_deref()
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &ReplConfig {
        &self.config
    }

    /// Get the raw command history.
    pub fn command_history(&self) -> &[String] {
        &self.command_history
    }

    /// Get a reference to the underlying session.
    pub fn session(&self) -> &ReplSession {
        &self.session
    }
}

// =============================================================================
// Formatting helpers
// =============================================================================

/// Format a single message for display.
///
/// If `show_timestamp` is true, a human-readable timestamp is prepended.
pub fn format_message(role: &str, content: &str, show_timestamp: bool) -> String {
    let mut out = String::new();

    if show_timestamp {
        let timestamp = format_system_time(SystemTime::now());
        out.push_str(&format!("[{}] ", timestamp));
    }

    out.push_str(&format!("{}: {}", role, content));
    out
}

/// Format a `SystemTime` into a simple `YYYY-MM-DD HH:MM:SS` string.
///
/// Uses a lightweight approach without pulling in `chrono` — computes from
/// the Unix epoch duration. Accuracy is sufficient for display purposes.
fn format_system_time(time: SystemTime) -> String {
    let secs = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Simple date/time calculation from Unix timestamp
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Compute year/month/day from days since epoch (1970-01-01)
    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm based on Howard Hinnant's civil_from_days
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    (y as u64, m, d)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Command parsing tests ---

    #[test]
    fn test_parse_help() {
        assert_eq!(ReplCommand::parse("/help"), Some(ReplCommand::Help));
        assert_eq!(ReplCommand::parse("/h"), Some(ReplCommand::Help));
        assert_eq!(ReplCommand::parse("/?"), Some(ReplCommand::Help));
    }

    #[test]
    fn test_parse_exit() {
        assert_eq!(ReplCommand::parse("/exit"), Some(ReplCommand::Exit));
        assert_eq!(ReplCommand::parse("/quit"), Some(ReplCommand::Exit));
        assert_eq!(ReplCommand::parse("/q"), Some(ReplCommand::Exit));
    }

    #[test]
    fn test_parse_save() {
        assert_eq!(
            ReplCommand::parse("/save path.json"),
            Some(ReplCommand::Save("path.json".to_string()))
        );
        // Save with no argument yields empty string
        assert_eq!(
            ReplCommand::parse("/save"),
            Some(ReplCommand::Save(String::new()))
        );
    }

    #[test]
    fn test_parse_load() {
        assert_eq!(
            ReplCommand::parse("/load path.json"),
            Some(ReplCommand::Load("path.json".to_string()))
        );
    }

    #[test]
    fn test_parse_model() {
        assert_eq!(
            ReplCommand::parse("/model gpt-4"),
            Some(ReplCommand::Model("gpt-4".to_string()))
        );
    }

    #[test]
    fn test_parse_template() {
        assert_eq!(
            ReplCommand::parse("/template coding"),
            Some(ReplCommand::Template("coding".to_string()))
        );
    }

    #[test]
    fn test_parse_unknown() {
        assert_eq!(
            ReplCommand::parse("/foo"),
            Some(ReplCommand::Unknown("foo".to_string()))
        );
        assert_eq!(
            ReplCommand::parse("/blah something"),
            Some(ReplCommand::Unknown("blah".to_string()))
        );
    }

    #[test]
    fn test_parse_regular_message() {
        // Regular text is not a command
        assert_eq!(ReplCommand::parse("hello"), None);
        assert_eq!(ReplCommand::parse("tell me about Rust"), None);
    }

    // --- Engine processing tests ---

    #[test]
    fn test_process_input_message() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        let action = engine.process_input("hello world");
        assert_eq!(action, ReplAction::SendMessage("hello world".to_string()));
    }

    #[test]
    fn test_process_input_command() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        let action = engine.process_input("/help");
        assert_eq!(action, ReplAction::ExecuteCommand(ReplCommand::Help));
    }

    #[test]
    fn test_process_input_empty() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        assert_eq!(engine.process_input(""), ReplAction::Continue);
        assert_eq!(engine.process_input("   "), ReplAction::Continue);
    }

    #[test]
    fn test_process_input_exit() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        assert_eq!(engine.process_input("/exit"), ReplAction::Exit);
        assert_eq!(engine.process_input("/quit"), ReplAction::Exit);
    }

    // --- Session management tests ---

    #[test]
    fn test_add_and_history() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        assert!(engine.history().is_empty());

        engine.add_message("user", "Hello");
        engine.add_message("assistant", "Hi there!");

        let hist = engine.history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], ("user".to_string(), "Hello".to_string()));
        assert_eq!(
            hist[1],
            ("assistant".to_string(), "Hi there!".to_string())
        );
    }

    #[test]
    fn test_clear_history() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.add_message("user", "test");
        engine.add_message("assistant", "response");
        assert_eq!(engine.history().len(), 2);

        engine.clear_history();
        assert!(engine.history().is_empty());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("repl_test_session.json");
        let path_str = path.to_string_lossy().to_string();

        // Create and save a session
        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.set_model("llama3");
        engine.set_template("coding");
        engine.add_message("user", "What is Rust?");
        engine.add_message("assistant", "Rust is a systems programming language.");
        engine.save_session(&path_str).expect("save should succeed");

        // Load into a fresh engine
        let mut engine2 = ReplEngine::new(ReplConfig::default());
        engine2
            .load_session(&path_str)
            .expect("load should succeed");

        assert_eq!(engine2.current_model(), "llama3");
        assert_eq!(engine2.current_template(), Some("coding"));
        assert_eq!(engine2.history().len(), 2);
        assert_eq!(engine2.history()[0].0, "user");
        assert_eq!(engine2.history()[0].1, "What is Rust?");
        assert_eq!(engine2.history()[1].0, "assistant");
        assert_eq!(engine2.history()[1].1, "Rust is a systems programming language.");

        // Clean up
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_session() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        let result = engine.load_session("/nonexistent/path/session.json");
        assert!(result.is_err());
        match result.unwrap_err() {
            ReplError::SessionNotFound(p) => {
                assert_eq!(p, "/nonexistent/path/session.json");
            }
            other => panic!("Expected SessionNotFound, got: {:?}", other),
        }
    }

    // --- Formatting tests ---

    #[test]
    fn test_format_help() {
        let help = ReplEngine::format_help();
        assert!(help.contains("/help"));
        assert!(help.contains("/exit"));
        assert!(help.contains("/quit"));
        assert!(help.contains("/models"));
        assert!(help.contains("/model"));
        assert!(help.contains("/template"));
        assert!(help.contains("/config"));
        assert!(help.contains("/history"));
        assert!(help.contains("/clear"));
        assert!(help.contains("/save"));
        assert!(help.contains("/load"));
        assert!(help.contains("/cost"));
    }

    #[test]
    fn test_set_model() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        assert_eq!(engine.current_model(), "default");

        engine.set_model("gpt-4");
        assert_eq!(engine.current_model(), "gpt-4");

        engine.set_model("claude-3");
        assert_eq!(engine.current_model(), "claude-3");
    }

    #[test]
    fn test_format_message_without_timestamp() {
        let msg = format_message("user", "Hello!", false);
        assert_eq!(msg, "user: Hello!");
    }

    #[test]
    fn test_format_message_with_timestamp() {
        let msg = format_message("assistant", "Hi!", true);
        // Should contain the role and content
        assert!(msg.contains("assistant: Hi!"));
        // Should have a bracketed timestamp prefix
        assert!(msg.starts_with('['));
        assert!(msg.contains(']'));
    }

    #[test]
    fn test_format_config() {
        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.set_model("llama3");
        let cfg = engine.format_config();
        assert!(cfg.contains("llama3"));
        assert!(cfg.contains("(none)")); // no template set
        assert!(cfg.contains("Prompt"));
    }
}
