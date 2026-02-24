//! Self-reflection: analyze prompt results and suggest improvements.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::{CompiledPrompt, FieldType, Signature, TrainingExample};

// ============================================================================
// Self-Reflector
// ============================================================================

/// Analyzes prompt compilation results and suggests improvement rules.
pub struct SelfReflector {
    /// Maximum number of reflection iterations
    pub max_iterations: usize,
    /// Minimum score improvement to consider a change worthwhile
    pub improvement_threshold: f64,
}

/// A suggested improvement rule from the self-reflector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementRule {
    /// Condition that triggered this rule (e.g., "low_precision")
    pub condition: String,
    /// Suggested action (e.g., "Add more examples")
    pub action: String,
    /// Whether this rule has been applied
    pub applied: bool,
}

impl SelfReflector {
    /// Create a new self-reflector.
    pub fn new(max_iterations: usize, improvement_threshold: f64) -> Self {
        Self {
            max_iterations,
            improvement_threshold,
        }
    }

    /// Analyze a compiled prompt against training examples and suggest improvements.
    ///
    /// Returns a list of improvement rules based on detected weaknesses.
    pub fn reflect(
        &self,
        signature: &Signature,
        compiled: &CompiledPrompt,
        examples: &[TrainingExample],
    ) -> Vec<ImprovementRule> {
        let mut rules: Vec<ImprovementRule> = Vec::new();

        // Rule 1: Check if the prompt mentions all output fields
        for output_field in &signature.outputs {
            if !compiled.system_prompt.contains(&output_field.name) {
                rules.push(ImprovementRule {
                    condition: format!(
                        "missing_output_field_reference:{}",
                        output_field.name
                    ),
                    action: format!(
                        "Add explicit reference to output field '{}' in the system prompt",
                        output_field.name
                    ),
                    applied: false,
                });
            }
        }

        // Rule 2: Check example coverage
        if examples.is_empty() {
            rules.push(ImprovementRule {
                condition: "no_examples".to_string(),
                action: "Add at least 3-5 few-shot examples to improve consistency".to_string(),
                applied: false,
            });
        } else if examples.len() < 3 {
            rules.push(ImprovementRule {
                condition: "few_examples".to_string(),
                action: format!(
                    "Only {} examples provided; consider adding more for better coverage",
                    examples.len()
                ),
                applied: false,
            });
        }

        // Rule 3: Check for missing instructions
        if signature.instructions.is_none() {
            rules.push(ImprovementRule {
                condition: "no_instructions".to_string(),
                action: "Add explicit instructions to guide the model's behavior".to_string(),
                applied: false,
            });
        }

        // Rule 4: Check input field coverage in examples
        for input_field in &signature.inputs {
            if input_field.required {
                let all_covered = examples.iter().all(|ex| {
                    ex.inputs.contains_key(&input_field.name)
                });
                if !all_covered {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "incomplete_input_coverage:{}",
                            input_field.name
                        ),
                        action: format!(
                            "Some examples are missing required input field '{}'; ensure all examples include it",
                            input_field.name
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 5: Check output field coverage in examples
        for output_field in &signature.outputs {
            if output_field.required {
                let all_covered = examples.iter().all(|ex| {
                    ex.expected_outputs.contains_key(&output_field.name)
                });
                if !all_covered {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "incomplete_output_coverage:{}",
                            output_field.name
                        ),
                        action: format!(
                            "Some examples are missing expected output field '{}'; ensure all examples include it",
                            output_field.name
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 6: Check if the system prompt is too short (may lack context)
        if compiled.system_prompt.len() < 50 {
            rules.push(ImprovementRule {
                condition: "short_system_prompt".to_string(),
                action: "System prompt is very short; consider adding more context and constraints".to_string(),
                applied: false,
            });
        }

        // Rule 7: Check for typed output fields without format guidance
        for output_field in &signature.outputs {
            if output_field.field_type != FieldType::Text {
                let type_str = output_field.field_type.to_string();
                if !compiled.system_prompt.contains(&type_str) {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "missing_type_guidance:{}",
                            output_field.name
                        ),
                        action: format!(
                            "Output field '{}' is type '{}' but the prompt lacks format guidance for this type",
                            output_field.name, type_str
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 8: Check for potential ambiguity (multiple output fields with same type)
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for output_field in &signature.outputs {
            let key = output_field.field_type.to_string();
            *type_counts.entry(key).or_insert(0) += 1;
        }
        for (type_name, count) in &type_counts {
            if *count > 1 {
                rules.push(ImprovementRule {
                    condition: format!("ambiguous_output_types:{}", type_name),
                    action: format!(
                        "Multiple output fields share type '{}'; add distinct prefixes or instructions to disambiguate",
                        type_name
                    ),
                    applied: false,
                });
            }
        }

        // Limit to max_iterations worth of rules
        rules.truncate(self.max_iterations);

        rules
    }

    /// Apply improvement rules by modifying a signature and recompiling.
    ///
    /// Returns the updated signature with improvements applied.
    pub fn apply_rules(
        &self,
        signature: &Signature,
        rules: &[ImprovementRule],
    ) -> Signature {
        let mut improved = signature.clone();

        // Collect actions into instructions
        let mut extra_instructions: Vec<String> = Vec::new();

        for rule in rules {
            if rule.condition == "no_instructions" {
                extra_instructions.push(
                    "Follow the field descriptions carefully.".to_string(),
                );
            } else if rule.condition.starts_with("missing_type_guidance:") {
                let field_name = rule.condition.strip_prefix("missing_type_guidance:")
                    .unwrap_or("");
                for output_field in &signature.outputs {
                    if output_field.name == field_name {
                        extra_instructions.push(format!(
                            "The '{}' field must be formatted as {}.",
                            field_name, output_field.field_type
                        ));
                    }
                }
            }
        }

        if !extra_instructions.is_empty() {
            let combined = match &improved.instructions {
                Some(existing) => format!("{}\n{}", existing, extra_instructions.join("\n")),
                None => extra_instructions.join("\n"),
            };
            improved.instructions = Some(combined);
        }

        improved
    }
}
