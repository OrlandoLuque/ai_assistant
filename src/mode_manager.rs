//! Mode manager — operation mode escalation and de-escalation
//!
//! Manages the progression between operation modes:
//! Chat → Assistant → Programming → AssemblyLine → Autonomous

use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// OperationMode
// ============================================================================

/// Operation modes from least to most capable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationMode {
    /// Simple conversation, no tools.
    Chat = 0,
    /// Basic tools (search, calculator, datetime).
    Assistant = 1,
    /// Filesystem + shell + git access.
    Programming = 2,
    /// Multi-agent coordinated execution.
    AssemblyLine = 3,
    /// Full autonomous loop with sandbox.
    Autonomous = 4,
}

impl OperationMode {
    /// Get all modes in order.
    pub fn all() -> &'static [OperationMode] {
        &[
            OperationMode::Chat,
            OperationMode::Assistant,
            OperationMode::Programming,
            OperationMode::AssemblyLine,
            OperationMode::Autonomous,
        ]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            OperationMode::Chat => "Chat",
            OperationMode::Assistant => "Assistant",
            OperationMode::Programming => "Programming",
            OperationMode::AssemblyLine => "Assembly Line",
            OperationMode::Autonomous => "Autonomous",
        }
    }

    /// Description of capabilities at this mode.
    pub fn description(&self) -> &'static str {
        match self {
            OperationMode::Chat => "Simple conversation, no tools",
            OperationMode::Assistant => "Basic tools: search, calculator, datetime",
            OperationMode::Programming => "Filesystem, shell, git access",
            OperationMode::AssemblyLine => "Multi-agent coordinated execution",
            OperationMode::Autonomous => "Full autonomous loop with sandbox",
        }
    }

    /// Get the list of capabilities for this mode.
    pub fn capabilities(&self) -> Vec<&'static str> {
        match self {
            OperationMode::Chat => vec!["conversation", "context"],
            OperationMode::Assistant => vec![
                "conversation",
                "context",
                "web_search",
                "calculator",
                "datetime",
            ],
            OperationMode::Programming => vec![
                "conversation",
                "context",
                "web_search",
                "calculator",
                "datetime",
                "file_read",
                "file_write",
                "shell",
                "git",
            ],
            OperationMode::AssemblyLine => vec![
                "conversation",
                "context",
                "web_search",
                "calculator",
                "datetime",
                "file_read",
                "file_write",
                "shell",
                "git",
                "multi_agent",
                "task_board",
                "delegation",
            ],
            OperationMode::Autonomous => vec![
                "conversation",
                "context",
                "web_search",
                "calculator",
                "datetime",
                "file_read",
                "file_write",
                "shell",
                "git",
                "multi_agent",
                "task_board",
                "delegation",
                "autonomous_loop",
                "mcp",
                "browser",
                "scheduler",
            ],
        }
    }

    /// Numeric level (0-4).
    pub fn level(&self) -> u8 {
        *self as u8
    }

    /// Parse from string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "chat" => Some(Self::Chat),
            "assistant" => Some(Self::Assistant),
            "programming" | "coding" | "dev" => Some(Self::Programming),
            "assembly" | "assemblyline" | "assembly_line" | "assembly-line" => {
                Some(Self::AssemblyLine)
            }
            "autonomous" | "auto" | "full" => Some(Self::Autonomous),
            _ => None,
        }
    }
}

impl std::fmt::Display for OperationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// ModeManager
// ============================================================================

/// Manages operation mode transitions with history and constraints.
pub struct ModeManager {
    current: OperationMode,
    /// Maximum mode the user allows.
    allowed_max: OperationMode,
    /// Whether to auto-escalate based on context analysis.
    auto_escalate: bool,
    /// History of mode changes: (mode, timestamp_ms).
    history: Vec<(OperationMode, u64)>,
}

impl ModeManager {
    /// Create a new manager starting at Chat with Autonomous as max.
    pub fn new() -> Self {
        let now = now_millis();
        Self {
            current: OperationMode::Chat,
            allowed_max: OperationMode::Autonomous,
            auto_escalate: false,
            history: vec![(OperationMode::Chat, now)],
        }
    }

    /// Create with specific starting mode and ceiling.
    pub fn with_config(start: OperationMode, max: OperationMode, auto_escalate: bool) -> Self {
        let effective_start = if start > max { max } else { start };
        let now = now_millis();
        Self {
            current: effective_start,
            allowed_max: max,
            auto_escalate,
            history: vec![(effective_start, now)],
        }
    }

    /// Get the current mode.
    pub fn current(&self) -> OperationMode {
        self.current
    }

    /// Get the maximum allowed mode.
    pub fn allowed_max(&self) -> OperationMode {
        self.allowed_max
    }

    /// Whether auto-escalation is enabled.
    pub fn auto_escalate(&self) -> bool {
        self.auto_escalate
    }

    /// Set auto-escalation.
    pub fn set_auto_escalate(&mut self, enabled: bool) {
        self.auto_escalate = enabled;
    }

    /// Set the maximum allowed mode.
    pub fn set_allowed_max(&mut self, max: OperationMode) {
        self.allowed_max = max;
        // If current exceeds new max, de-escalate
        if self.current > self.allowed_max {
            self.set_mode(self.allowed_max).ok();
        }
    }

    /// Escalate one level up. Returns Err if already at max.
    pub fn escalate(&mut self) -> Result<OperationMode, String> {
        let next_level = self.current.level() + 1;
        let next = OperationMode::all()
            .get(next_level as usize)
            .copied()
            .ok_or_else(|| "Already at maximum mode".to_string())?;

        if next > self.allowed_max {
            return Err(format!(
                "Cannot escalate to {} — max allowed is {}",
                next.name(),
                self.allowed_max.name()
            ));
        }

        self.current = next;
        self.history.push((next, now_millis()));
        Ok(next)
    }

    /// De-escalate one level down. Returns current if already at Chat.
    pub fn de_escalate(&mut self) -> OperationMode {
        if self.current == OperationMode::Chat {
            return self.current;
        }
        let prev_level = self.current.level().saturating_sub(1);
        let prev = OperationMode::all()[prev_level as usize];
        self.current = prev;
        self.history.push((prev, now_millis()));
        prev
    }

    /// Set mode directly (respecting allowed_max).
    pub fn set_mode(&mut self, mode: OperationMode) -> Result<(), String> {
        if mode > self.allowed_max {
            return Err(format!(
                "Mode {} exceeds allowed maximum {}",
                mode.name(),
                self.allowed_max.name()
            ));
        }
        self.current = mode;
        self.history.push((mode, now_millis()));
        Ok(())
    }

    /// Suggest the best mode based on context keywords.
    pub fn suggest_mode(&self, context: &str) -> OperationMode {
        let lower = context.to_lowercase();

        if contains_any(
            &lower,
            &["run", "execute", "deploy", "build and test", "autonomous"],
        ) {
            return cap(OperationMode::Autonomous, self.allowed_max);
        }
        if contains_any(
            &lower,
            &[
                "coordinate",
                "delegate",
                "team",
                "parallel agents",
                "assembly",
            ],
        ) {
            return cap(OperationMode::AssemblyLine, self.allowed_max);
        }
        if contains_any(
            &lower,
            &[
                "file", "code", "edit", "commit", "compile", "shell", "git", "mkdir", "write",
            ],
        ) {
            return cap(OperationMode::Programming, self.allowed_max);
        }
        if contains_any(
            &lower,
            &["search", "calculate", "what time", "look up", "find"],
        ) {
            return cap(OperationMode::Assistant, self.allowed_max);
        }
        OperationMode::Chat
    }

    /// Auto-escalate if enabled and context suggests a higher mode.
    pub fn auto_suggest_and_escalate(&mut self, context: &str) -> Option<OperationMode> {
        if !self.auto_escalate {
            return None;
        }
        let suggested = self.suggest_mode(context);
        if suggested > self.current {
            self.current = suggested;
            self.history.push((suggested, now_millis()));
            Some(suggested)
        } else {
            None
        }
    }

    /// Get mode transition history.
    pub fn history(&self) -> &[(OperationMode, u64)] {
        &self.history
    }

    /// Number of mode changes (excluding initial).
    pub fn change_count(&self) -> usize {
        self.history.len().saturating_sub(1)
    }
}

impl Default for ModeManager {
    fn default() -> Self {
        Self::new()
    }
}

fn contains_any(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|k| text.contains(k))
}

fn cap(mode: OperationMode, max: OperationMode) -> OperationMode {
    if mode > max {
        max
    } else {
        mode
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_manager() {
        let mgr = ModeManager::new();
        assert_eq!(mgr.current(), OperationMode::Chat);
        assert_eq!(mgr.allowed_max(), OperationMode::Autonomous);
        assert!(!mgr.auto_escalate());
    }

    #[test]
    fn test_escalate() {
        let mut mgr = ModeManager::new();
        assert_eq!(mgr.escalate().unwrap(), OperationMode::Assistant);
        assert_eq!(mgr.escalate().unwrap(), OperationMode::Programming);
        assert_eq!(mgr.escalate().unwrap(), OperationMode::AssemblyLine);
        assert_eq!(mgr.escalate().unwrap(), OperationMode::Autonomous);
        assert!(mgr.escalate().is_err());
    }

    #[test]
    fn test_escalate_capped() {
        let mut mgr =
            ModeManager::with_config(OperationMode::Chat, OperationMode::Programming, false);
        assert_eq!(mgr.escalate().unwrap(), OperationMode::Assistant);
        assert_eq!(mgr.escalate().unwrap(), OperationMode::Programming);
        assert!(mgr.escalate().is_err());
    }

    #[test]
    fn test_de_escalate() {
        let mut mgr =
            ModeManager::with_config(OperationMode::Autonomous, OperationMode::Autonomous, false);
        assert_eq!(mgr.de_escalate(), OperationMode::AssemblyLine);
        assert_eq!(mgr.de_escalate(), OperationMode::Programming);
        assert_eq!(mgr.de_escalate(), OperationMode::Assistant);
        assert_eq!(mgr.de_escalate(), OperationMode::Chat);
        assert_eq!(mgr.de_escalate(), OperationMode::Chat); // stays at Chat
    }

    #[test]
    fn test_set_mode() {
        let mut mgr = ModeManager::new();
        assert!(mgr.set_mode(OperationMode::Programming).is_ok());
        assert_eq!(mgr.current(), OperationMode::Programming);
    }

    #[test]
    fn test_set_mode_exceeds_max() {
        let mut mgr =
            ModeManager::with_config(OperationMode::Chat, OperationMode::Assistant, false);
        assert!(mgr.set_mode(OperationMode::Programming).is_err());
        assert_eq!(mgr.current(), OperationMode::Chat);
    }

    #[test]
    fn test_suggest_mode() {
        let mgr = ModeManager::new();
        assert_eq!(mgr.suggest_mode("hello, how are you?"), OperationMode::Chat);
        assert_eq!(
            mgr.suggest_mode("search for rust docs"),
            OperationMode::Assistant
        );
        assert_eq!(
            mgr.suggest_mode("edit the file main.rs"),
            OperationMode::Programming
        );
        assert_eq!(
            mgr.suggest_mode("coordinate a team of agents"),
            OperationMode::AssemblyLine
        );
        assert_eq!(
            mgr.suggest_mode("run the full test suite and deploy"),
            OperationMode::Autonomous
        );
    }

    #[test]
    fn test_auto_escalate() {
        let mut mgr =
            ModeManager::with_config(OperationMode::Chat, OperationMode::Autonomous, true);
        let result = mgr.auto_suggest_and_escalate("please edit the file config.rs");
        assert_eq!(result, Some(OperationMode::Programming));
        assert_eq!(mgr.current(), OperationMode::Programming);

        // Already at Programming, suggesting Programming again → no change
        let result = mgr.auto_suggest_and_escalate("edit another file");
        assert_eq!(result, None);
    }

    #[test]
    fn test_mode_capabilities() {
        assert!(OperationMode::Chat.capabilities().contains(&"conversation"));
        assert!(!OperationMode::Chat.capabilities().contains(&"shell"));
        assert!(OperationMode::Programming.capabilities().contains(&"shell"));
        assert!(OperationMode::Autonomous
            .capabilities()
            .contains(&"autonomous_loop"));
    }

    #[test]
    fn test_mode_ordering() {
        assert!(OperationMode::Chat < OperationMode::Assistant);
        assert!(OperationMode::Assistant < OperationMode::Programming);
        assert!(OperationMode::Programming < OperationMode::AssemblyLine);
        assert!(OperationMode::AssemblyLine < OperationMode::Autonomous);
    }

    #[test]
    fn test_history_tracking() {
        let mut mgr = ModeManager::new();
        mgr.escalate().unwrap();
        mgr.escalate().unwrap();
        mgr.de_escalate();

        assert_eq!(mgr.change_count(), 3);
        assert_eq!(mgr.history().len(), 4); // initial + 3 changes
        assert_eq!(mgr.history()[0].0, OperationMode::Chat);
        assert_eq!(mgr.history()[1].0, OperationMode::Assistant);
        assert_eq!(mgr.history()[2].0, OperationMode::Programming);
        assert_eq!(mgr.history()[3].0, OperationMode::Assistant);
    }

    #[test]
    fn test_from_str_loose() {
        assert_eq!(
            OperationMode::from_str_loose("chat"),
            Some(OperationMode::Chat)
        );
        assert_eq!(
            OperationMode::from_str_loose("AUTONOMOUS"),
            Some(OperationMode::Autonomous)
        );
        assert_eq!(
            OperationMode::from_str_loose("coding"),
            Some(OperationMode::Programming)
        );
        assert_eq!(
            OperationMode::from_str_loose("assembly-line"),
            Some(OperationMode::AssemblyLine)
        );
        assert_eq!(OperationMode::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_set_allowed_max_deescalates() {
        let mut mgr =
            ModeManager::with_config(OperationMode::Autonomous, OperationMode::Autonomous, false);
        assert_eq!(mgr.current(), OperationMode::Autonomous);

        mgr.set_allowed_max(OperationMode::Assistant);
        assert_eq!(mgr.current(), OperationMode::Assistant);
    }

    #[test]
    fn test_mode_description() {
        for mode in OperationMode::all() {
            let desc = mode.description();
            assert!(
                !desc.is_empty(),
                "Mode {:?} should have a non-empty description",
                mode
            );
        }
    }

    #[test]
    fn test_mode_level_ordering() {
        assert!(OperationMode::Chat.level() < OperationMode::Assistant.level());
        assert!(OperationMode::Assistant.level() < OperationMode::Programming.level());
        assert!(OperationMode::Programming.level() < OperationMode::AssemblyLine.level());
        assert!(OperationMode::AssemblyLine.level() < OperationMode::Autonomous.level());
    }

    #[test]
    fn test_set_auto_escalate() {
        let mut mgr = ModeManager::new();
        assert!(!mgr.auto_escalate());

        mgr.set_auto_escalate(true);
        assert!(mgr.auto_escalate());

        mgr.set_auto_escalate(false);
        assert!(!mgr.auto_escalate());
    }

    #[test]
    fn test_mode_history() {
        let mut mgr = ModeManager::new();
        // Initial history has one entry (Chat)
        assert_eq!(mgr.history().len(), 1);
        assert_eq!(mgr.history()[0].0, OperationMode::Chat);

        mgr.set_mode(OperationMode::Programming).unwrap();
        mgr.set_mode(OperationMode::Assistant).unwrap();
        mgr.set_mode(OperationMode::Autonomous).unwrap();

        // Should have 4 entries: initial Chat + 3 set_mode calls
        assert_eq!(mgr.history().len(), 4);
        assert_eq!(mgr.history()[0].0, OperationMode::Chat);
        assert_eq!(mgr.history()[1].0, OperationMode::Programming);
        assert_eq!(mgr.history()[2].0, OperationMode::Assistant);
        assert_eq!(mgr.history()[3].0, OperationMode::Autonomous);
    }
}
