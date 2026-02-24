//! Automatic memory extraction: applies rules to conversational text to extract
//! facts, entities, preferences, and procedures.

use serde::{Deserialize, Serialize};

use super::consolidation::SemanticFact;

/// A memory extraction result from analyzing conversational text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryExtraction {
    /// A new semantic fact was extracted.
    NewFact { fact: SemanticFact },
    /// An entity attribute was updated.
    EntityUpdate {
        entity_name: String,
        attribute: String,
        value: String,
    },
    /// A new procedure was identified.
    NewProcedure {
        name: String,
        steps: Vec<String>,
        confidence: f64,
    },
    /// A correction to a previously stored value.
    Correction {
        original_id: String,
        corrected_value: String,
    },
    /// A user preference.
    Preference { key: String, value: String },
}

/// Configuration for the automatic memory extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Minimum confidence for an extraction to be accepted.
    pub min_confidence: f64,
    /// Maximum number of extractions to return per invocation.
    pub max_extractions_per_turn: usize,
    /// Whether to extract facts (subject-predicate-object).
    pub extract_facts: bool,
    /// Whether to extract entity updates.
    pub extract_entities: bool,
    /// Whether to extract procedures.
    pub extract_procedures: bool,
    /// Whether to extract preferences.
    pub extract_preferences: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_extractions_per_turn: 10,
            extract_facts: true,
            extract_entities: true,
            extract_procedures: true,
            extract_preferences: true,
        }
    }
}

/// The type of pattern an extraction rule matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionRuleType {
    /// Matches fact patterns (subject-predicate-object).
    FactPattern,
    /// Matches entity patterns (names, attributes).
    EntityPattern,
    /// Matches preference patterns ("I prefer X").
    PreferencePattern,
    /// Matches date patterns (YYYY-MM-DD, etc.).
    DatePattern,
    /// Matches name patterns ("my name is X").
    NamePattern,
}

/// A single extraction rule that maps a regex pattern to an extraction type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Regex pattern string.
    pub pattern: String,
    /// The type of extraction this rule produces.
    pub extraction_type: ExtractionRuleType,
    /// Confidence level for extractions produced by this rule.
    pub confidence: f64,
}

/// Automatic memory extractor that applies rules to conversational text.
pub struct MemoryExtractor {
    config: ExtractionConfig,
    rules: Vec<ExtractionRule>,
}

impl MemoryExtractor {
    /// Create a new extractor with the given configuration and no rules.
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
        }
    }

    /// Create an extractor with default configuration and a set of built-in rules
    /// for common conversational patterns.
    pub fn with_defaults() -> Self {
        let config = ExtractionConfig::default();
        let rules = vec![
            ExtractionRule {
                name: "name_introduction".to_string(),
                pattern: r"(?i)my name is (\w+)".to_string(),
                extraction_type: ExtractionRuleType::NamePattern,
                confidence: 0.9,
            },
            ExtractionRule {
                name: "preference_over".to_string(),
                pattern: r"(?i)I prefer (\w[\w\s]*?) over (\w[\w\s]*?)$".to_string(),
                extraction_type: ExtractionRuleType::PreferencePattern,
                confidence: 0.85,
            },
            ExtractionRule {
                name: "preference_simple".to_string(),
                pattern: r"(?i)I prefer (\w[\w\s]*)".to_string(),
                extraction_type: ExtractionRuleType::PreferencePattern,
                confidence: 0.8,
            },
            ExtractionRule {
                name: "fact_is".to_string(),
                pattern: r"(?i)(\w[\w\s]*?) (?:is|are) (\w[\w\s]*)".to_string(),
                extraction_type: ExtractionRuleType::FactPattern,
                confidence: 0.7,
            },
            ExtractionRule {
                name: "remember_that".to_string(),
                pattern: r"(?i)remember that (.+)".to_string(),
                extraction_type: ExtractionRuleType::FactPattern,
                confidence: 0.85,
            },
            ExtractionRule {
                name: "date_iso".to_string(),
                pattern: r"(\d{4}-\d{2}-\d{2})".to_string(),
                extraction_type: ExtractionRuleType::DatePattern,
                confidence: 0.9,
            },
            ExtractionRule {
                name: "date_weekday".to_string(),
                pattern: r"(?i)on (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)".to_string(),
                extraction_type: ExtractionRuleType::DatePattern,
                confidence: 0.75,
            },
            ExtractionRule {
                name: "email_pattern".to_string(),
                pattern: r"[\w.+-]+@[\w-]+\.[\w.-]+".to_string(),
                extraction_type: ExtractionRuleType::EntityPattern,
                confidence: 0.95,
            },
        ];
        Self { config, rules }
    }

    /// Add a custom extraction rule.
    pub fn add_rule(&mut self, rule: ExtractionRule) {
        self.rules.push(rule);
    }

    /// Return the number of rules currently loaded.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get a reference to the current extraction configuration.
    pub fn config(&self) -> &ExtractionConfig {
        &self.config
    }

    /// Extract memory items from the given text by applying all loaded rules.
    ///
    /// Returns at most `max_extractions_per_turn` extractions, filtered by the
    /// configuration flags (extract_facts, extract_entities, etc.) and
    /// `min_confidence`.
    pub fn extract(&self, text: &str) -> Vec<MemoryExtraction> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<MemoryExtraction> = Vec::new();

        for rule in &self.rules {
            if rule.confidence < self.config.min_confidence {
                continue;
            }

            // Check whether this rule type is enabled in the config
            let enabled = match &rule.extraction_type {
                ExtractionRuleType::FactPattern | ExtractionRuleType::DatePattern => {
                    self.config.extract_facts
                }
                ExtractionRuleType::EntityPattern | ExtractionRuleType::NamePattern => {
                    self.config.extract_entities
                }
                ExtractionRuleType::PreferencePattern => self.config.extract_preferences,
            };
            if !enabled {
                continue;
            }

            // Try to match the rule's regex pattern against the text
            if let Some(extraction) = self.apply_rule(rule, text) {
                results.push(extraction);
                if results.len() >= self.config.max_extractions_per_turn {
                    break;
                }
            }
        }

        results
    }

    /// Apply a single rule to text and return an extraction if the pattern matches.
    fn apply_rule(&self, rule: &ExtractionRule, text: &str) -> Option<MemoryExtraction> {
        // We use a simple regex-like approach based on the rule type.
        // For patterns that contain capture groups, we extract them manually
        // since we do not pull in the `regex` crate in this module.
        match &rule.extraction_type {
            ExtractionRuleType::NamePattern => {
                // "my name is X"
                let text_lower = text.to_lowercase();
                if let Some(pos) = text_lower.find("my name is ") {
                    let after = &text[pos + 11..];
                    let name: String = after
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                        .collect();
                    if !name.is_empty() {
                        return Some(MemoryExtraction::EntityUpdate {
                            entity_name: "user".to_string(),
                            attribute: "name".to_string(),
                            value: name,
                        });
                    }
                }
                None
            }
            ExtractionRuleType::PreferencePattern => {
                let text_lower = text.to_lowercase();
                if let Some(pos) = text_lower.find("i prefer ") {
                    let after = &text[pos + 9..];
                    let after_trimmed = after.trim();
                    // Check for "X over Y" pattern
                    if let Some(over_pos) = after_trimmed.to_lowercase().find(" over ") {
                        let preferred = after_trimmed[..over_pos].trim().to_string();
                        let other = after_trimmed[over_pos + 6..].trim().to_string();
                        if !preferred.is_empty() {
                            return Some(MemoryExtraction::Preference {
                                key: format!("preference:{}", preferred.to_lowercase()),
                                value: format!("{} over {}", preferred, other),
                            });
                        }
                    } else {
                        // Simple preference
                        let preferred: String = after_trimmed
                            .chars()
                            .take_while(|c| *c != '.' && *c != '!' && *c != '?')
                            .collect();
                        let preferred = preferred.trim().to_string();
                        if !preferred.is_empty() {
                            return Some(MemoryExtraction::Preference {
                                key: format!("preference:{}", preferred.to_lowercase()),
                                value: preferred,
                            });
                        }
                    }
                }
                None
            }
            ExtractionRuleType::FactPattern => {
                let text_lower = text.to_lowercase();
                // "remember that X"
                if let Some(pos) = text_lower.find("remember that ") {
                    let content = text[pos + 14..].trim();
                    if !content.is_empty() {
                        let now = chrono::Utc::now();
                        return Some(MemoryExtraction::NewFact {
                            fact: SemanticFact {
                                id: uuid::Uuid::new_v4().to_string(),
                                subject: "user".to_string(),
                                predicate: "stated".to_string(),
                                object: content.to_string(),
                                confidence: rule.confidence,
                                source_episodes: Vec::new(),
                                created_at: now,
                                last_confirmed: now,
                            },
                        });
                    }
                }
                // "X is Y" / "X are Y"
                for verb in &[" is ", " are "] {
                    if let Some(pos) = text_lower.find(verb) {
                        let subject = text[..pos].trim();
                        let object = text[pos + verb.len()..].trim();
                        // Filter out very short subjects/objects
                        if subject.len() >= 2 && object.len() >= 2 {
                            let now = chrono::Utc::now();
                            return Some(MemoryExtraction::NewFact {
                                fact: SemanticFact {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    subject: subject.to_string(),
                                    predicate: verb.trim().to_string(),
                                    object: object.to_string(),
                                    confidence: rule.confidence,
                                    source_episodes: Vec::new(),
                                    created_at: now,
                                    last_confirmed: now,
                                },
                            });
                        }
                    }
                }
                None
            }
            ExtractionRuleType::DatePattern => {
                // ISO date: YYYY-MM-DD
                let mut i = 0;
                let bytes = text.as_bytes();
                while i + 10 <= bytes.len() {
                    if bytes[i].is_ascii_digit()
                        && bytes[i + 4] == b'-'
                        && bytes[i + 7] == b'-'
                        && bytes[i + 1].is_ascii_digit()
                        && bytes[i + 2].is_ascii_digit()
                        && bytes[i + 3].is_ascii_digit()
                        && bytes[i + 5].is_ascii_digit()
                        && bytes[i + 6].is_ascii_digit()
                        && bytes[i + 8].is_ascii_digit()
                        && bytes[i + 9].is_ascii_digit()
                    {
                        let date_str = &text[i..i + 10];
                        let now = chrono::Utc::now();
                        return Some(MemoryExtraction::NewFact {
                            fact: SemanticFact {
                                id: uuid::Uuid::new_v4().to_string(),
                                subject: "date_reference".to_string(),
                                predicate: "mentioned".to_string(),
                                object: date_str.to_string(),
                                confidence: rule.confidence,
                                source_episodes: Vec::new(),
                                created_at: now,
                                last_confirmed: now,
                            },
                        });
                    }
                    i += 1;
                }
                // Weekday pattern
                let text_lower = text.to_lowercase();
                for day in &[
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ] {
                    let pattern = format!("on {}", day);
                    if text_lower.contains(&pattern) {
                        let now = chrono::Utc::now();
                        return Some(MemoryExtraction::NewFact {
                            fact: SemanticFact {
                                id: uuid::Uuid::new_v4().to_string(),
                                subject: "date_reference".to_string(),
                                predicate: "mentioned".to_string(),
                                object: day.to_string(),
                                confidence: rule.confidence,
                                source_episodes: Vec::new(),
                                created_at: now,
                                last_confirmed: now,
                            },
                        });
                    }
                }
                None
            }
            ExtractionRuleType::EntityPattern => {
                // Email pattern
                let mut email_start = None;
                let chars: Vec<char> = text.chars().collect();
                for (i, ch) in chars.iter().enumerate() {
                    if *ch == '@' {
                        // Walk backward to find start of local part
                        let mut start = i;
                        while start > 0 {
                            let prev = chars[start - 1];
                            if prev.is_alphanumeric() || prev == '.' || prev == '+' || prev == '-' || prev == '_' {
                                start -= 1;
                            } else {
                                break;
                            }
                        }
                        // Walk forward to find end of domain
                        let mut end = i + 1;
                        while end < chars.len() {
                            let next = chars[end];
                            if next.is_alphanumeric() || next == '.' || next == '-' {
                                end += 1;
                            } else {
                                break;
                            }
                        }
                        if start < i && end > i + 1 {
                            // Verify domain has a dot
                            let domain: String = chars[i + 1..end].iter().collect();
                            if domain.contains('.') {
                                email_start = Some((start, end));
                                break;
                            }
                        }
                    }
                }
                if let Some((start, end)) = email_start {
                    let email: String = chars[start..end].iter().collect();
                    return Some(MemoryExtraction::EntityUpdate {
                        entity_name: "user".to_string(),
                        attribute: "email".to_string(),
                        value: email,
                    });
                }
                None
            }
        }
    }
}
