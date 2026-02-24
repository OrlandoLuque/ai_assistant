//! Prompt assertions and constraints for output validation.

use super::types::Signature;

// ============================================================================
// 1.3 Prompt Assertions & Constraints
// ============================================================================

/// Result of checking a prompt assertion.
#[derive(Debug, Clone, PartialEq)]
pub enum AssertionResult {
    /// The assertion passed.
    Pass,
    /// The assertion failed with a reason.
    Fail { reason: String },
    /// The assertion raised a warning (soft failure).
    Warn { reason: String },
}

/// Trait for programmatic constraints on prompt outputs.
pub trait PromptAssertion: Send + Sync {
    /// Check the assertion against the given output string.
    fn check(&self, output: &str) -> AssertionResult;
    /// The name of this assertion.
    fn name(&self) -> &str;
}

/// Asserts that the output length is within specified bounds.
pub struct LengthAssertion {
    /// Minimum number of characters (inclusive)
    pub min_chars: Option<usize>,
    /// Maximum number of characters (inclusive)
    pub max_chars: Option<usize>,
    /// Minimum number of whitespace-separated tokens
    pub min_tokens: Option<usize>,
    /// Maximum number of whitespace-separated tokens
    pub max_tokens: Option<usize>,
}

impl PromptAssertion for LengthAssertion {
    fn check(&self, output: &str) -> AssertionResult {
        let char_count = output.len();
        let token_count = output.split_whitespace().count();

        if let Some(min) = self.min_chars {
            if char_count < min {
                return AssertionResult::Fail {
                    reason: format!(
                        "Output has {} chars, below minimum of {}",
                        char_count, min
                    ),
                };
            }
        }
        if let Some(max) = self.max_chars {
            if char_count > max {
                return AssertionResult::Fail {
                    reason: format!(
                        "Output has {} chars, exceeds maximum of {}",
                        char_count, max
                    ),
                };
            }
        }
        if let Some(min) = self.min_tokens {
            if token_count < min {
                return AssertionResult::Fail {
                    reason: format!(
                        "Output has {} tokens, below minimum of {}",
                        token_count, min
                    ),
                };
            }
        }
        if let Some(max) = self.max_tokens {
            if token_count > max {
                return AssertionResult::Fail {
                    reason: format!(
                        "Output has {} tokens, exceeds maximum of {}",
                        token_count, max
                    ),
                };
            }
        }
        AssertionResult::Pass
    }

    fn name(&self) -> &str {
        "length_assertion"
    }
}

/// Asserts that the output matches a given pattern (simple substring/pattern match).
pub struct FormatAssertion {
    /// Pattern that the output must match (substring search)
    pub pattern: String,
}

impl PromptAssertion for FormatAssertion {
    fn check(&self, output: &str) -> AssertionResult {
        // Simple pattern matching: check if output contains the pattern
        // For more complex regex, one would use the `regex` crate, but we keep
        // dependencies minimal by using a simple contains check with basic
        // wildcard support (pattern as literal substring).
        if output.contains(&self.pattern) {
            AssertionResult::Pass
        } else {
            AssertionResult::Fail {
                reason: format!(
                    "Output does not match pattern '{}'",
                    self.pattern
                ),
            }
        }
    }

    fn name(&self) -> &str {
        "format_assertion"
    }
}

/// Asserts that the output contains all specified keywords.
pub struct ContainsAssertion {
    /// Keywords that must be present in the output
    pub required_keywords: Vec<String>,
    /// Whether the keyword check is case-sensitive
    pub case_sensitive: bool,
}

impl PromptAssertion for ContainsAssertion {
    fn check(&self, output: &str) -> AssertionResult {
        let output_normalized = if self.case_sensitive {
            output.to_string()
        } else {
            output.to_lowercase()
        };

        for keyword in &self.required_keywords {
            let kw = if self.case_sensitive {
                keyword.clone()
            } else {
                keyword.to_lowercase()
            };
            if !output_normalized.contains(&kw) {
                return AssertionResult::Fail {
                    reason: format!("Output is missing required keyword '{}'", keyword),
                };
            }
        }
        AssertionResult::Pass
    }

    fn name(&self) -> &str {
        "contains_assertion"
    }
}

/// Asserts that the output is valid JSON.
pub struct JsonSchemaAssertion;

impl PromptAssertion for JsonSchemaAssertion {
    fn check(&self, output: &str) -> AssertionResult {
        match serde_json::from_str::<serde_json::Value>(output) {
            Ok(_) => AssertionResult::Pass,
            Err(e) => AssertionResult::Fail {
                reason: format!("Output is not valid JSON: {}", e),
            },
        }
    }

    fn name(&self) -> &str {
        "json_schema_assertion"
    }
}

/// Custom assertion using a user-provided closure.
pub struct CustomAssertion {
    /// Name for this custom assertion
    assertion_name: String,
    /// The check function
    check_fn: Box<dyn Fn(&str) -> AssertionResult + Send + Sync>,
}

impl CustomAssertion {
    /// Create a new custom assertion with the given name and check function.
    pub fn new(
        name: impl Into<String>,
        check_fn: Box<dyn Fn(&str) -> AssertionResult + Send + Sync>,
    ) -> Self {
        Self {
            assertion_name: name.into(),
            check_fn,
        }
    }
}

impl PromptAssertion for CustomAssertion {
    fn check(&self, output: &str) -> AssertionResult {
        (self.check_fn)(output)
    }

    fn name(&self) -> &str {
        &self.assertion_name
    }
}

/// A signature paired with programmatic assertions on its outputs.
pub struct AssertedSignature {
    /// The underlying signature
    pub signature: Signature,
    /// Assertions that must hold on the output
    pub assertions: Vec<Box<dyn PromptAssertion + Send + Sync>>,
}

impl AssertedSignature {
    /// Create a new asserted signature.
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            assertions: Vec::new(),
        }
    }

    /// Add an assertion to this signature.
    pub fn add_assertion(&mut self, assertion: Box<dyn PromptAssertion + Send + Sync>) {
        self.assertions.push(assertion);
    }

    /// Check all assertions against the given output.
    ///
    /// Returns a list of (assertion_name, result) pairs.
    pub fn check_output(&self, output: &str) -> Vec<(String, AssertionResult)> {
        self.assertions
            .iter()
            .map(|a| (a.name().to_string(), a.check(output)))
            .collect()
    }

    /// Compute an assertion penalty score.
    ///
    /// Returns 0.0 if all assertions pass, 1.0 if all fail.
    /// Warnings count as 0.5 weight.
    pub fn assertion_penalty(&self, output: &str) -> f64 {
        if self.assertions.is_empty() {
            return 0.0;
        }

        let mut penalty_sum = 0.0;
        for assertion in &self.assertions {
            match assertion.check(output) {
                AssertionResult::Pass => {}
                AssertionResult::Fail { .. } => penalty_sum += 1.0,
                AssertionResult::Warn { .. } => penalty_sum += 0.5,
            }
        }

        penalty_sum / self.assertions.len() as f64
    }
}
