//! Chain-of-thought parsing
//!
//! This module provides utilities for extracting and analyzing
//! chain-of-thought reasoning from model responses.
//!
//! # Features
//!
//! - **Step extraction**: Parse reasoning steps from responses
//! - **Reasoning validation**: Verify logical consistency
//! - **Answer extraction**: Extract final answers from CoT responses
//! - **Thought summarization**: Summarize reasoning chains

/// Configuration for CoT parsing
#[derive(Debug, Clone)]
pub struct CotConfig {
    /// Markers that indicate reasoning steps
    pub step_markers: Vec<String>,
    /// Markers that indicate final answer
    pub answer_markers: Vec<String>,
    /// Whether to extract numbered steps
    pub extract_numbered: bool,
    /// Minimum words for a valid step
    pub min_step_words: usize,
    /// Maximum steps to extract
    pub max_steps: usize,
}

impl Default for CotConfig {
    fn default() -> Self {
        Self {
            step_markers: vec![
                "First,".to_string(),
                "Second,".to_string(),
                "Third,".to_string(),
                "Then,".to_string(),
                "Next,".to_string(),
                "Finally,".to_string(),
                "Step".to_string(),
                "Let's".to_string(),
                "We can".to_string(),
                "To solve".to_string(),
            ],
            answer_markers: vec![
                "Therefore,".to_string(),
                "Thus,".to_string(),
                "So,".to_string(),
                "The answer is".to_string(),
                "In conclusion,".to_string(),
                "Hence,".to_string(),
                "Final answer:".to_string(),
            ],
            extract_numbered: true,
            min_step_words: 3,
            max_steps: 20,
        }
    }
}

/// A single reasoning step
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step number (1-indexed)
    pub number: usize,
    /// The step content
    pub content: String,
    /// Step type/category
    pub step_type: StepType,
    /// Entities mentioned in this step
    pub entities: Vec<String>,
    /// Mathematical operations if any
    pub math_operations: Vec<String>,
}

/// Types of reasoning steps
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StepType {
    /// Problem understanding
    Understanding,
    /// Breaking down the problem
    Decomposition,
    /// Applying knowledge/rules
    Application,
    /// Calculation/computation
    Calculation,
    /// Logical inference
    Inference,
    /// Verification/checking
    Verification,
    /// Final conclusion
    Conclusion,
    /// General reasoning
    General,
}

impl StepType {
    fn from_content(content: &str) -> Self {
        let lower = content.to_lowercase();

        if lower.contains("let's understand")
            || lower.contains("the problem")
            || lower.contains("given that")
        {
            StepType::Understanding
        } else if lower.contains("break down")
            || lower.contains("split")
            || lower.contains("divide")
        {
            StepType::Decomposition
        } else if lower.contains("apply")
            || lower.contains("using the")
            || lower.contains("according to")
        {
            StepType::Application
        } else if lower.contains("calculate")
            || lower.contains("compute")
            || contains_calculation(&lower)
        {
            StepType::Calculation
        } else if lower.contains("therefore")
            || lower.contains("implies")
            || lower.contains("conclude")
        {
            StepType::Inference
        } else if lower.contains("verify")
            || lower.contains("check")
            || lower.contains("confirm")
        {
            StepType::Verification
        } else if lower.contains("final answer")
            || lower.contains("the answer is")
            || lower.contains("in conclusion")
        {
            StepType::Conclusion
        } else {
            StepType::General
        }
    }
}

fn contains_calculation(s: &str) -> bool {
    s.contains(" = ") || s.contains(" + ") || s.contains(" - ") || s.contains(" * ") || s.contains(" / ")
}

/// Result of CoT parsing
#[derive(Debug, Clone)]
pub struct CotParseResult {
    /// Original response
    pub original: String,
    /// Extracted reasoning steps
    pub steps: Vec<ReasoningStep>,
    /// Final answer if found
    pub answer: Option<String>,
    /// Whether response appears to use CoT
    pub is_cot_response: bool,
    /// Overall reasoning quality score (0-1)
    pub quality_score: f64,
    /// Summary of the reasoning
    pub summary: Option<String>,
}

/// Chain-of-thought parser
pub struct CotParser {
    config: CotConfig,
}

impl CotParser {
    /// Create a new CoT parser
    pub fn new(config: CotConfig) -> Self {
        Self { config }
    }

    /// Parse a response for chain-of-thought reasoning
    pub fn parse(&self, response: &str) -> CotParseResult {
        let mut steps = Vec::new();
        #[allow(unused_assignments)]
        let mut answer = None;

        // Try numbered extraction first
        if self.config.extract_numbered {
            steps = self.extract_numbered_steps(response);
        }

        // Fall back to marker-based extraction
        if steps.is_empty() {
            steps = self.extract_marker_steps(response);
        }

        // Extract final answer
        answer = self.extract_answer(response);

        // Calculate quality score
        let quality_score = self.calculate_quality(&steps);

        // Generate summary
        let summary = if !steps.is_empty() {
            Some(self.summarize_reasoning(&steps))
        } else {
            None
        };

        CotParseResult {
            original: response.to_string(),
            steps: steps.clone(),
            answer,
            is_cot_response: !steps.is_empty(),
            quality_score,
            summary,
        }
    }

    /// Extract numbered steps (1., 2., etc.)
    fn extract_numbered_steps(&self, response: &str) -> Vec<ReasoningStep> {
        let mut steps = Vec::new();
        // Use a simpler regex without lookahead (not supported by regex crate)
        let re = regex::Regex::new(r"(?m)^\s*(\d+)[.\)]\s*(.+)$").expect("valid regex");

        for (idx, cap) in re.captures_iter(response).enumerate() {
            if idx >= self.config.max_steps {
                break;
            }

            let content = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if content.split_whitespace().count() >= self.config.min_step_words {
                steps.push(ReasoningStep {
                    number: idx + 1,
                    content: content.to_string(),
                    step_type: StepType::from_content(content),
                    entities: extract_entities(content),
                    math_operations: extract_math_ops(content),
                });
            }
        }

        steps
    }

    /// Extract steps based on markers
    fn extract_marker_steps(&self, response: &str) -> Vec<ReasoningStep> {
        let mut steps = Vec::new();
        let lines: Vec<&str> = response.lines().collect();

        let mut current_step = String::new();
        let mut step_number = 0;

        for line in lines {
            let trimmed = line.trim();

            // Check if line starts with a marker
            let is_new_step = self.config.step_markers.iter().any(|m| {
                trimmed.to_lowercase().starts_with(&m.to_lowercase())
            });

            if is_new_step {
                // Save previous step
                if !current_step.is_empty()
                    && current_step.split_whitespace().count() >= self.config.min_step_words
                {
                    step_number += 1;
                    steps.push(ReasoningStep {
                        number: step_number,
                        content: current_step.trim().to_string(),
                        step_type: StepType::from_content(&current_step),
                        entities: extract_entities(&current_step),
                        math_operations: extract_math_ops(&current_step),
                    });

                    if steps.len() >= self.config.max_steps {
                        break;
                    }
                }
                current_step = trimmed.to_string();
            } else if !current_step.is_empty() {
                current_step.push(' ');
                current_step.push_str(trimmed);
            }
        }

        // Don't forget the last step
        if !current_step.is_empty()
            && current_step.split_whitespace().count() >= self.config.min_step_words
            && steps.len() < self.config.max_steps
        {
            step_number += 1;
            steps.push(ReasoningStep {
                number: step_number,
                content: current_step.trim().to_string(),
                step_type: StepType::from_content(&current_step),
                entities: extract_entities(&current_step),
                math_operations: extract_math_ops(&current_step),
            });
        }

        steps
    }

    /// Extract final answer from response
    fn extract_answer(&self, response: &str) -> Option<String> {
        let lower = response.to_lowercase();

        for marker in &self.config.answer_markers {
            let marker_lower = marker.to_lowercase();
            if let Some(pos) = lower.find(&marker_lower) {
                let start = pos + marker.len();
                let rest = &response[start..];

                // Take until end of sentence or paragraph
                let end = rest
                    .find(|c: char| c == '.' || c == '\n')
                    .unwrap_or(rest.len());

                let answer = rest[..end].trim();
                if !answer.is_empty() {
                    return Some(answer.to_string());
                }
            }
        }

        // Try to extract answer from last line
        let last_line = response.lines().last()?.trim();
        if last_line.len() < 200 {
            return Some(last_line.to_string());
        }

        None
    }

    /// Calculate reasoning quality score
    fn calculate_quality(&self, steps: &[ReasoningStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let step_count = steps.len() as f64;

        // Score for number of steps (more is generally better, up to a point)
        score += (step_count / 5.0).min(1.0) * 0.3;

        // Score for step diversity
        let unique_types: std::collections::HashSet<_> =
            steps.iter().map(|s| &s.step_type).collect();
        score += (unique_types.len() as f64 / 4.0).min(1.0) * 0.2;

        // Score for conclusion step
        if steps.iter().any(|s| s.step_type == StepType::Conclusion) {
            score += 0.2;
        }

        // Score for verification step
        if steps.iter().any(|s| s.step_type == StepType::Verification) {
            score += 0.15;
        }

        // Score for calculation steps if math operations found
        if steps.iter().any(|s| !s.math_operations.is_empty()) {
            score += 0.15;
        }

        score.min(1.0)
    }

    /// Summarize the reasoning chain
    fn summarize_reasoning(&self, steps: &[ReasoningStep]) -> String {
        let step_count = steps.len();
        let types: Vec<_> = steps.iter().map(|s| format!("{:?}", s.step_type)).collect();

        format!(
            "Reasoning with {} steps: {}",
            step_count,
            types.join(" -> ")
        )
    }
}

impl Default for CotParser {
    fn default() -> Self {
        Self::new(CotConfig::default())
    }
}

/// Extract entities mentioned in text
fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();

    // Extract quoted strings
    let quote_re = regex::Regex::new(r#""([^"]+)""#).expect("valid regex");
    for cap in quote_re.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            entities.push(m.as_str().to_string());
        }
    }

    // Extract capitalized phrases (simple NER)
    let cap_re = regex::Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b").expect("valid regex");
    for m in cap_re.find_iter(text) {
        let entity = m.as_str().to_string();
        if !entities.contains(&entity) {
            entities.push(entity);
        }
    }

    entities
}

/// Extract mathematical operations
fn extract_math_ops(text: &str) -> Vec<String> {
    let mut ops = Vec::new();

    // Match mathematical expressions
    let math_re = regex::Regex::new(r"\d+\s*[+\-*/×÷=]\s*\d+(?:\s*[+\-*/×÷=]\s*\d+)*").expect("valid regex");
    for m in math_re.find_iter(text) {
        ops.push(m.as_str().to_string());
    }

    ops
}

/// Validator for chain-of-thought reasoning
pub struct CotValidator {
    /// Required step types for valid reasoning
    required_types: Vec<StepType>,
    /// Minimum steps required
    min_steps: usize,
}

impl CotValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            required_types: Vec::new(),
            min_steps: 1,
        }
    }

    /// Require specific step types
    pub fn require_types(mut self, types: Vec<StepType>) -> Self {
        self.required_types = types;
        self
    }

    /// Set minimum steps
    pub fn min_steps(mut self, n: usize) -> Self {
        self.min_steps = n;
        self
    }

    /// Validate a CoT parse result
    pub fn validate(&self, result: &CotParseResult) -> ValidationResult {
        let mut issues = Vec::new();

        // Check step count
        if result.steps.len() < self.min_steps {
            issues.push(format!(
                "Insufficient steps: {} < {}",
                result.steps.len(),
                self.min_steps
            ));
        }

        // Check required types
        for required in &self.required_types {
            if !result.steps.iter().any(|s| &s.step_type == required) {
                issues.push(format!("Missing required step type: {:?}", required));
            }
        }

        // Check for answer
        if result.answer.is_none() {
            issues.push("No final answer found".to_string());
        }

        ValidationResult {
            valid: issues.is_empty(),
            issues,
            score: result.quality_score,
        }
    }
}

impl Default for CotValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of CoT validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether reasoning is valid
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<String>,
    /// Quality score
    pub score: f64,
}

/// Builder for CoT configuration
pub struct CotConfigBuilder {
    config: CotConfig,
}

impl CotConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CotConfig::default(),
        }
    }

    /// Add step markers
    pub fn step_markers(mut self, markers: Vec<String>) -> Self {
        self.config.step_markers = markers;
        self
    }

    /// Add answer markers
    pub fn answer_markers(mut self, markers: Vec<String>) -> Self {
        self.config.answer_markers = markers;
        self
    }

    /// Enable/disable numbered extraction
    pub fn extract_numbered(mut self, enabled: bool) -> Self {
        self.config.extract_numbered = enabled;
        self
    }

    /// Set minimum step words
    pub fn min_step_words(mut self, n: usize) -> Self {
        self.config.min_step_words = n;
        self
    }

    /// Set maximum steps
    pub fn max_steps(mut self, n: usize) -> Self {
        self.config.max_steps = n;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CotConfig {
        self.config
    }
}

impl Default for CotConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = CotParser::default();
        assert!(!parser.config.step_markers.is_empty());
    }

    #[test]
    fn test_numbered_step_extraction() {
        let parser = CotParser::default();
        let response = r#"
1. First, let's understand the problem.
2. Then, we calculate 2 + 2 = 4.
3. Finally, the answer is 4.
"#;

        let result = parser.parse(response);
        assert!(result.is_cot_response);
        assert_eq!(result.steps.len(), 3);
    }

    #[test]
    fn test_marker_step_extraction() {
        let parser = CotParser::default();
        let response = r#"
Let's think about this step by step.
First, we need to understand what we're solving.
Then, we apply the formula to get our result.
Therefore, the answer is 42.
"#;

        let result = parser.parse(response);
        assert!(result.is_cot_response);
        assert!(!result.steps.is_empty());
    }

    #[test]
    fn test_answer_extraction() {
        let parser = CotParser::default();
        let response = "After careful analysis, the answer is 42.";

        let result = parser.parse(response);
        assert!(result.answer.is_some());
        assert!(result.answer.unwrap().contains("42"));
    }

    #[test]
    fn test_step_type_detection() {
        assert_eq!(
            StepType::from_content("Let's understand the problem first"),
            StepType::Understanding
        );
        assert_eq!(
            StepType::from_content("Calculate 2 + 2 = 4"),
            StepType::Calculation
        );
        assert_eq!(
            StepType::from_content("Therefore, the result is"),
            StepType::Inference
        );
    }

    #[test]
    fn test_math_extraction() {
        let ops = extract_math_ops("We compute 5 + 3 = 8 and then 8 * 2 = 16");
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_validator() {
        let validator = CotValidator::new()
            .min_steps(2)
            .require_types(vec![StepType::Conclusion]);

        let result = CotParseResult {
            original: String::new(),
            steps: vec![
                ReasoningStep {
                    number: 1,
                    content: "Step 1".to_string(),
                    step_type: StepType::General,
                    entities: Vec::new(),
                    math_operations: Vec::new(),
                },
                ReasoningStep {
                    number: 2,
                    content: "Final answer".to_string(),
                    step_type: StepType::Conclusion,
                    entities: Vec::new(),
                    math_operations: Vec::new(),
                },
            ],
            answer: Some("42".to_string()),
            is_cot_response: true,
            quality_score: 0.8,
            summary: None,
        };

        let validation = validator.validate(&result);
        assert!(validation.valid);
    }
}
