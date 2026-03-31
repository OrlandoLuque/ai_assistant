//! Automatic memory extraction: applies rules to conversational text to extract
//! facts, entities, preferences, and procedures.

use serde::{Deserialize, Serialize};

use super::consolidation::SemanticFact;

/// A memory extraction result from analyzing conversational text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// Use LLM to enhance entity extraction with richer NER.
    /// When false (default), uses heuristic pattern matching.
    pub llm_enhanced: bool,
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
            llm_enhanced: false,
        }
    }
}

/// The type of pattern an extraction rule matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
                // "my name is X" / "mi nombre es X" / "me llamo X" / "soy X"
                let text_lower = text.to_lowercase();

                // English + Spanish name prefixes
                let name_prefixes: &[(&str, usize)] = &[
                    ("my name is ", 11),
                    ("mi nombre es ", 14),
                    ("me llamo ", 9),
                ];
                for &(prefix, skip) in name_prefixes {
                    if let Some(pos) = text_lower.find(prefix) {
                        let after = &text[pos + skip..];
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
                }

                // Spanish: "soy X" — only when followed by a capitalized word
                if let Some(pos) = text_lower.find("soy ") {
                    let after = &text[pos + 4..];
                    if let Some(first_char) = after.chars().next() {
                        if first_char.is_uppercase() {
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
                    }
                }

                None
            }
            ExtractionRuleType::PreferencePattern => {
                let text_lower = text.to_lowercase();

                // English: "I prefer X [over Y]"
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

                // Spanish preference patterns: "prefiero X", "me gusta X", "no me gusta X"
                let es_pref_patterns: &[(&str, usize, bool)] = &[
                    ("no me gusta ", 12, true),   // negative preference (check first)
                    ("prefiero ", 9, false),
                    ("me gusta ", 9, false),
                ];
                for &(prefix, skip, is_negative) in es_pref_patterns {
                    if let Some(pos) = text_lower.find(prefix) {
                        let after = &text[pos + skip..];
                        let preferred: String = after
                            .trim()
                            .chars()
                            .take_while(|c| *c != '.' && *c != '!' && *c != '?')
                            .collect();
                        let preferred = preferred.trim().to_string();
                        if !preferred.is_empty() {
                            let value = if is_negative {
                                format!("dislikes {}", preferred)
                            } else {
                                preferred.clone()
                            };
                            return Some(MemoryExtraction::Preference {
                                key: format!("preference:{}", preferred.to_lowercase()),
                                value,
                            });
                        }
                    }
                }

                None
            }
            ExtractionRuleType::FactPattern => {
                let text_lower = text.to_lowercase();
                // "remember that X" / "recuerda que X" / "ten en cuenta que X"
                let fact_prefixes: &[(&str, usize)] = &[
                    ("remember that ", 14),
                    ("recuerda que ", 13),
                    ("ten en cuenta que ", 18),
                ];
                for &(prefix, skip) in fact_prefixes {
                    if let Some(pos) = text_lower.find(prefix) {
                        let content = text[pos + skip..].trim();
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
                // Weekday pattern (English + Spanish)
                let text_lower = text.to_lowercase();

                // English weekdays with "on" prefix
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

                // Spanish weekdays (standalone, no prefix needed)
                let es_days: &[(&str, &str)] = &[
                    ("lunes", "lunes"),
                    ("martes", "martes"),
                    ("miércoles", "miércoles"),
                    ("jueves", "jueves"),
                    ("viernes", "viernes"),
                    ("sábado", "sábado"),
                    ("domingo", "domingo"),
                ];
                for &(day_lower, day_label) in es_days {
                    if text_lower.contains(day_lower) {
                        let now = chrono::Utc::now();
                        return Some(MemoryExtraction::NewFact {
                            fact: SemanticFact {
                                id: uuid::Uuid::new_v4().to_string(),
                                subject: "date_reference".to_string(),
                                predicate: "mentioned".to_string(),
                                object: day_label.to_string(),
                                confidence: rule.confidence,
                                source_episodes: Vec::new(),
                                created_at: now,
                                last_confirmed: now,
                            },
                        });
                    }
                }

                // Spanish relative dates and phrases
                let es_relative: &[(&str, &str)] = &[
                    ("la semana que viene", "next week"),
                    ("el mes que viene", "next month"),
                    ("mañana", "tomorrow"),
                    ("ayer", "yesterday"),
                    ("hoy", "today"),
                ];
                for &(phrase, label) in es_relative {
                    if text_lower.contains(phrase) {
                        let now = chrono::Utc::now();
                        return Some(MemoryExtraction::NewFact {
                            fact: SemanticFact {
                                id: uuid::Uuid::new_v4().to_string(),
                                subject: "date_reference".to_string(),
                                predicate: "mentioned".to_string(),
                                object: label.to_string(),
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

// ============================================================================
// LLM Enhancement: Entity Extraction (V68)
// ============================================================================

/// A named entity extracted by LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name/value.
    pub name: String,
    /// Entity type (person, organization, location, date, concept).
    pub entity_type: String,
}

impl MemoryExtractor {
    /// Build a prompt for LLM-based entity extraction.
    ///
    /// Returns None if LLM enhancement is disabled or text is empty.
    pub fn build_entity_prompt(&self, text: &str) -> Option<String> {
        if !self.config.llm_enhanced || text.is_empty() {
            return None;
        }

        Some(format!(
            "Extract named entities from this text. Return JSON array: \
             [{{\"name\":\"X\",\"type\":\"person|organization|location|date|concept\"}}]\n\n\
             Examples:\n\
             Input: \"Alice works at Google in New York.\"\n\
             Output: [{{\"name\":\"Alice\",\"type\":\"person\"}},{{\"name\":\"Google\",\"type\":\"organization\"}},{{\"name\":\"New York\",\"type\":\"location\"}}]\n\n\
             Input: \"The meeting is on 2026-03-15 about machine learning.\"\n\
             Output: [{{\"name\":\"2026-03-15\",\"type\":\"date\"}},{{\"name\":\"machine learning\",\"type\":\"concept\"}}]\n\n\
             Now extract entities from this:\n{}",
            crate::llm_enhance::prompt_wrap(text)
        ))
    }

    /// Parse LLM response for entity extraction.
    pub fn parse_entity_response(response: &str) -> Vec<ExtractedEntity> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(entities) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
                return entities
                    .iter()
                    .filter_map(|v| {
                        let name = v.get("name")?.as_str()?.to_string();
                        let entity_type = v.get("type")?.as_str()?.to_string();
                        if name.is_empty() {
                            return None;
                        }
                        Some(ExtractedEntity { name, entity_type })
                    })
                    .collect();
            }
        }
        Vec::new()
    }

    /// Extract entities with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for richer
    /// named entity recognition. Otherwise falls back to heuristic extraction.
    pub fn extract_entities_with_llm(
        &self,
        text: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> Vec<ExtractedEntity> {
        // Try LLM enhancement first
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_entity_prompt(text) {
                    if let Ok(response) = enhancer.generate(&prompt, 500) {
                        let entities = Self::parse_entity_response(&response);
                        if !entities.is_empty() {
                            return entities;
                        }
                    }
                }
            }
        }

        // Fallback: convert heuristic extractions to ExtractedEntity
        self.extract(text)
            .into_iter()
            .filter_map(|extraction| match extraction {
                MemoryExtraction::EntityUpdate {
                    entity_name,
                    attribute,
                    value,
                } => Some(ExtractedEntity {
                    name: value,
                    entity_type: if attribute == "name" {
                        "person".to_string()
                    } else if attribute == "email" {
                        "concept".to_string()
                    } else {
                        entity_name
                    },
                }),
                MemoryExtraction::NewFact { fact } => Some(ExtractedEntity {
                    name: fact.object,
                    entity_type: "concept".to_string(),
                }),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_entities_heuristic_only() {
        // with_defaults() has llm_enhanced=false and built-in rules
        let extractor = MemoryExtractor::with_defaults();
        let entities = extractor.extract_entities_with_llm("My name is Alice", None);
        assert!(
            entities.iter().any(|e| e.name == "Alice"),
            "Should extract Alice from heuristic, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_entities_with_mock_llm() {
        let mut config = ExtractionConfig::default();
        config.llm_enhanced = true;
        let extractor = MemoryExtractor::new(config);
        let mock = crate::llm_enhance::MockLlm::new(
            "[{\"name\":\"OpenAI\",\"type\":\"organization\"},{\"name\":\"San Francisco\",\"type\":\"location\"}]",
        );
        let entities = extractor.extract_entities_with_llm(
            "OpenAI is based in San Francisco",
            Some(&mock),
        );
        assert_eq!(entities.len(), 2);
        assert!(entities.iter().any(|e| e.name == "OpenAI" && e.entity_type == "organization"));
        assert!(entities.iter().any(|e| e.name == "San Francisco" && e.entity_type == "location"));
    }

    #[test]
    fn test_extract_entities_llm_fallback_on_failure() {
        // Build extractor with llm_enhanced=true and default rules
        let config = ExtractionConfig {
            llm_enhanced: true,
            ..ExtractionConfig::default()
        };
        let mut extractor = MemoryExtractor::new(config);
        // Manually add the name rule so heuristic fallback works
        extractor.add_rule(ExtractionRule {
            name: "name_introduction".to_string(),
            pattern: r"(?i)my name is (\w+)".to_string(),
            extraction_type: ExtractionRuleType::NamePattern,
            confidence: 0.9,
        });
        let failing = crate::llm_enhance::FailingMockLlm;
        let entities = extractor.extract_entities_with_llm("My name is Bob", Some(&failing));
        // Should fall back to heuristic (not crash), and find Bob via NamePattern
        assert!(
            entities.iter().any(|e| e.name == "Bob"),
            "Fallback should extract Bob, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_build_entity_prompt() {
        let mut config = ExtractionConfig::default();
        config.llm_enhanced = true;
        let extractor = MemoryExtractor::new(config);
        let prompt = extractor.build_entity_prompt("Hello world");
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("Extract named entities"));
    }

    #[test]
    fn test_build_entity_prompt_disabled() {
        let config = ExtractionConfig::default(); // llm_enhanced = false
        let extractor = MemoryExtractor::new(config);
        assert!(extractor.build_entity_prompt("Hello").is_none());
    }

    #[test]
    fn test_parse_entity_response_valid() {
        let response = "[{\"name\":\"John\",\"type\":\"person\"}]";
        let entities = MemoryExtractor::parse_entity_response(response);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "John");
        assert_eq!(entities[0].entity_type, "person");
    }

    #[test]
    fn test_parse_entity_response_invalid() {
        let response = "Not valid JSON at all";
        let entities = MemoryExtractor::parse_entity_response(response);
        assert!(entities.is_empty());
    }

    // ── V69 Phase B: Multilingual extraction tests ──────────────────

    #[test]
    fn test_spanish_name_extraction() {
        let extractor = MemoryExtractor::with_defaults();

        // "mi nombre es X"
        let results = extractor.extract("mi nombre es Carlos");
        assert!(
            results.iter().any(|e| matches!(e,
                MemoryExtraction::EntityUpdate { value, attribute, .. }
                if value == "Carlos" && attribute == "name"
            )),
            "Should extract 'Carlos' from 'mi nombre es Carlos', got: {:?}",
            results
        );

        // "me llamo X"
        let results2 = extractor.extract("me llamo Ana");
        assert!(
            results2.iter().any(|e| matches!(e,
                MemoryExtraction::EntityUpdate { value, attribute, .. }
                if value == "Ana" && attribute == "name"
            )),
            "Should extract 'Ana' from 'me llamo Ana', got: {:?}",
            results2
        );
    }

    #[test]
    fn test_spanish_preference_extraction() {
        let extractor = MemoryExtractor::with_defaults();

        // "prefiero X"
        let results = extractor.extract("prefiero el modo oscuro");
        assert!(
            results.iter().any(|e| matches!(e, MemoryExtraction::Preference { .. })),
            "Should extract preference from 'prefiero el modo oscuro', got: {:?}",
            results
        );

        // "no me gusta X"
        let results2 = extractor.extract("no me gusta la verbosidad");
        assert!(
            results2.iter().any(|e| matches!(e, MemoryExtraction::Preference { value, .. } if value.contains("dislikes"))),
            "Should extract negative preference from 'no me gusta', got: {:?}",
            results2
        );
    }

    #[test]
    fn test_spanish_date_extraction() {
        let extractor = MemoryExtractor::with_defaults();

        // Spanish weekday
        let results = extractor.extract("La reunión es el lunes");
        assert!(
            results.iter().any(|e| matches!(e,
                MemoryExtraction::NewFact { fact }
                if fact.subject == "date_reference" && fact.object == "lunes"
            )),
            "Should extract 'lunes' as date reference, got: {:?}",
            results
        );

        // Spanish relative date
        let results2 = extractor.extract("Lo haré mañana");
        assert!(
            results2.iter().any(|e| matches!(e,
                MemoryExtraction::NewFact { fact }
                if fact.subject == "date_reference" && fact.object == "tomorrow"
            )),
            "Should extract 'mañana' as 'tomorrow', got: {:?}",
            results2
        );
    }
}
