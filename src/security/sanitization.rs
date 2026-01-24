//! Input sanitization for cleaning user messages

use serde::{Deserialize, Serialize};

/// Configuration for input sanitization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationConfig {
    /// Maximum input length (characters)
    pub max_input_length: usize,
    /// Maximum consecutive newlines
    pub max_consecutive_newlines: usize,
    /// Strip invisible/control characters
    pub strip_control_chars: bool,
    /// Normalize unicode
    pub normalize_unicode: bool,
    /// Block potential prompt injections
    pub block_prompt_injection: bool,
}

impl Default for SanitizationConfig {
    fn default() -> Self {
        Self {
            max_input_length: 50000,
            max_consecutive_newlines: 5,
            strip_control_chars: true,
            normalize_unicode: true,
            block_prompt_injection: false, // Conservative default
        }
    }
}

/// Input sanitizer for cleaning user messages
pub struct InputSanitizer {
    config: SanitizationConfig,
    injection_patterns: Vec<&'static str>,
}

impl InputSanitizer {
    pub fn new(config: SanitizationConfig) -> Self {
        Self {
            config,
            injection_patterns: vec![
                "ignore previous instructions",
                "ignore all previous",
                "disregard previous",
                "forget your instructions",
                "new instructions:",
                "system prompt:",
                "you are now",
                "act as if",
                "pretend you are",
                "roleplay as",
                "jailbreak",
                "DAN mode",
                "developer mode",
            ],
        }
    }

    /// Sanitize user input
    pub fn sanitize(&self, input: &str) -> SanitizationResult {
        let mut warnings = Vec::new();
        let mut output = input.to_string();

        // Check and truncate length
        if output.len() > self.config.max_input_length {
            output.truncate(self.config.max_input_length);
            warnings.push(SanitizationWarning::Truncated {
                original_length: input.len(),
                max_length: self.config.max_input_length,
            });
        }

        // Strip control characters (except newlines and tabs)
        if self.config.strip_control_chars {
            let before_len = output.len();
            output = output.chars()
                .filter(|c| !c.is_control() || *c == '\n' || *c == '\t' || *c == '\r')
                .collect();
            if output.len() != before_len {
                warnings.push(SanitizationWarning::ControlCharsRemoved);
            }
        }

        // Normalize consecutive newlines
        if self.config.max_consecutive_newlines > 0 {
            let max_newlines = "\n".repeat(self.config.max_consecutive_newlines);
            let too_many = "\n".repeat(self.config.max_consecutive_newlines + 1);
            while output.contains(&too_many) {
                output = output.replace(&too_many, &max_newlines);
            }
        }

        // Normalize unicode (NFKC normalization simulation - basic)
        if self.config.normalize_unicode {
            // Replace common lookalike characters
            output = output
                .replace('\u{200B}', "") // zero-width space
                .replace('\u{200C}', "") // zero-width non-joiner
                .replace('\u{200D}', "") // zero-width joiner
                .replace('\u{FEFF}', "") // BOM
                .replace('\u{00A0}', " "); // non-breaking space
        }

        // Check for prompt injection patterns
        if self.config.block_prompt_injection {
            let lower = output.to_lowercase();
            for pattern in &self.injection_patterns {
                if lower.contains(pattern) {
                    return SanitizationResult::Blocked {
                        reason: format!("Potential prompt injection detected: '{}'", pattern),
                    };
                }
            }
        }

        if warnings.is_empty() {
            SanitizationResult::Clean { output }
        } else {
            SanitizationResult::Sanitized { output, warnings }
        }
    }

    /// Quick check if input needs sanitization
    pub fn needs_sanitization(&self, input: &str) -> bool {
        if input.len() > self.config.max_input_length {
            return true;
        }
        if self.config.strip_control_chars && input.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
            return true;
        }
        false
    }
}

/// Result of input sanitization
#[derive(Debug, Clone)]
pub enum SanitizationResult {
    /// Input was already clean
    Clean { output: String },
    /// Input was sanitized with warnings
    Sanitized { output: String, warnings: Vec<SanitizationWarning> },
    /// Input was blocked entirely
    Blocked { reason: String },
}

impl SanitizationResult {
    pub fn is_blocked(&self) -> bool {
        matches!(self, SanitizationResult::Blocked { .. })
    }

    pub fn get_output(&self) -> Option<&str> {
        match self {
            SanitizationResult::Clean { output } | SanitizationResult::Sanitized { output, .. } => Some(output),
            SanitizationResult::Blocked { .. } => None,
        }
    }

    pub fn into_output(self) -> Option<String> {
        match self {
            SanitizationResult::Clean { output } | SanitizationResult::Sanitized { output, .. } => Some(output),
            SanitizationResult::Blocked { .. } => None,
        }
    }
}

/// Warning from sanitization process
#[derive(Debug, Clone)]
pub enum SanitizationWarning {
    Truncated { original_length: usize, max_length: usize },
    ControlCharsRemoved,
    UnicodeNormalized,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_sanitizer() {
        let config = SanitizationConfig {
            max_input_length: 100,
            ..Default::default()
        };
        let sanitizer = InputSanitizer::new(config);

        // Test truncation
        let long_input = "a".repeat(200);
        let result = sanitizer.sanitize(&long_input);
        match result {
            SanitizationResult::Sanitized { output, warnings } => {
                assert_eq!(output.len(), 100);
                assert!(warnings.iter().any(|w| matches!(w, SanitizationWarning::Truncated { .. })));
            }
            _ => panic!("Expected sanitized result"),
        }

        // Test clean input
        let clean = "Hello, world!";
        let result = sanitizer.sanitize(clean);
        assert!(matches!(result, SanitizationResult::Clean { .. }));
    }
}
