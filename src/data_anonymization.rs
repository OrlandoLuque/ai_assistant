//! Data anonymization
//!
//! Anonymize personal and sensitive data.

use std::collections::HashMap;
use regex::Regex;

/// Anonymization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymizationStrategy {
    Redact,
    Hash,
    Pseudonymize,
    Generalize,
    Mask,
}

/// Data type for anonymization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Email,
    Phone,
    Name,
    Address,
    CreditCard,
    Ssn,
    IpAddress,
    Date,
    Custom,
}

/// Anonymization rule
#[derive(Debug, Clone)]
pub struct AnonymizationRule {
    pub data_type: DataType,
    pub strategy: AnonymizationStrategy,
    pub pattern: Option<String>,
    pub replacement: Option<String>,
}

impl AnonymizationRule {
    pub fn new(data_type: DataType, strategy: AnonymizationStrategy) -> Self {
        Self {
            data_type,
            strategy,
            pattern: None,
            replacement: None,
        }
    }

    pub fn with_pattern(mut self, pattern: &str) -> Self {
        self.pattern = Some(pattern.to_string());
        self
    }

    pub fn with_replacement(mut self, replacement: &str) -> Self {
        self.replacement = Some(replacement.to_string());
        self
    }
}

/// Anonymization result
#[derive(Debug, Clone)]
pub struct AnonymizationResult {
    pub original: String,
    pub anonymized: String,
    pub detections: Vec<Detection>,
    pub mapping: HashMap<String, String>,
}

/// Detection of sensitive data
#[derive(Debug, Clone)]
pub struct Detection {
    pub data_type: DataType,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}

/// Data anonymizer
pub struct DataAnonymizer {
    rules: Vec<AnonymizationRule>,
    patterns: HashMap<DataType, Regex>,
    pseudonym_counter: u64,
    pseudonym_mapping: HashMap<String, String>,
}

impl DataAnonymizer {
    pub fn new() -> Self {
        let mut anonymizer = Self {
            rules: Vec::new(),
            patterns: HashMap::new(),
            pseudonym_counter: 0,
            pseudonym_mapping: HashMap::new(),
        };

        // Add default patterns
        anonymizer.add_default_patterns();
        anonymizer
    }

    fn add_default_patterns(&mut self) {
        // Email pattern
        if let Ok(re) = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}") {
            self.patterns.insert(DataType::Email, re);
        }

        // Phone pattern (various formats)
        if let Ok(re) = Regex::new(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}") {
            self.patterns.insert(DataType::Phone, re);
        }

        // Credit card pattern
        if let Ok(re) = Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b") {
            self.patterns.insert(DataType::CreditCard, re);
        }

        // SSN pattern
        if let Ok(re) = Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b") {
            self.patterns.insert(DataType::Ssn, re);
        }

        // IP address pattern
        if let Ok(re) = Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b") {
            self.patterns.insert(DataType::IpAddress, re);
        }

        // Date pattern
        if let Ok(re) = Regex::new(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b") {
            self.patterns.insert(DataType::Date, re);
        }
    }

    pub fn add_rule(&mut self, rule: AnonymizationRule) {
        if let Some(ref pattern) = rule.pattern {
            if let Ok(re) = Regex::new(pattern) {
                self.patterns.insert(rule.data_type, re);
            }
        }
        self.rules.push(rule);
    }

    pub fn anonymize(&mut self, text: &str) -> AnonymizationResult {
        let mut result = text.to_string();
        let mut detections = Vec::new();
        let mut mapping = HashMap::new();

        // Collect pattern matches first
        let matches: Vec<_> = self.patterns.iter()
            .flat_map(|(data_type, pattern)| {
                let strategy = self.get_strategy(*data_type);
                pattern.find_iter(text)
                    .map(move |cap| (*data_type, strategy, cap.as_str().to_string(), cap.start(), cap.end()))
            })
            .collect();

        // Now process matches with mutable access
        for (data_type, strategy, value, start, end) in matches {
            let replacement = self.anonymize_value(&value, data_type, strategy);

            detections.push(Detection {
                data_type,
                value: value.clone(),
                start,
                end,
                replacement: replacement.clone(),
            });

            mapping.insert(value, replacement);
        }

        // Sort detections by position (reverse order for replacement)
        detections.sort_by(|a, b| b.start.cmp(&a.start));

        // Apply replacements
        for detection in &detections {
            if let Some(replacement) = mapping.get(&detection.value) {
                result = result.replacen(&detection.value, replacement, 1);
            }
        }

        // Re-sort for output
        detections.sort_by(|a, b| a.start.cmp(&b.start));

        AnonymizationResult {
            original: text.to_string(),
            anonymized: result,
            detections,
            mapping,
        }
    }

    fn get_strategy(&self, data_type: DataType) -> AnonymizationStrategy {
        self.rules.iter()
            .find(|r| r.data_type == data_type)
            .map(|r| r.strategy)
            .unwrap_or(AnonymizationStrategy::Redact)
    }

    fn anonymize_value(&mut self, value: &str, data_type: DataType, strategy: AnonymizationStrategy) -> String {
        match strategy {
            AnonymizationStrategy::Redact => {
                self.get_redaction(data_type)
            }
            AnonymizationStrategy::Hash => {
                format!("HASH_{:08x}", self.simple_hash(value))
            }
            AnonymizationStrategy::Pseudonymize => {
                self.get_pseudonym(value, data_type)
            }
            AnonymizationStrategy::Generalize => {
                self.generalize_value(value, data_type)
            }
            AnonymizationStrategy::Mask => {
                self.mask_value(value)
            }
        }
    }

    fn get_redaction(&self, data_type: DataType) -> String {
        match data_type {
            DataType::Email => "[EMAIL REDACTED]".to_string(),
            DataType::Phone => "[PHONE REDACTED]".to_string(),
            DataType::Name => "[NAME REDACTED]".to_string(),
            DataType::Address => "[ADDRESS REDACTED]".to_string(),
            DataType::CreditCard => "[CARD REDACTED]".to_string(),
            DataType::Ssn => "[SSN REDACTED]".to_string(),
            DataType::IpAddress => "[IP REDACTED]".to_string(),
            DataType::Date => "[DATE REDACTED]".to_string(),
            DataType::Custom => "[REDACTED]".to_string(),
        }
    }

    fn simple_hash(&self, value: &str) -> u32 {
        let mut hash: u32 = 0;
        for byte in value.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn get_pseudonym(&mut self, value: &str, data_type: DataType) -> String {
        if let Some(existing) = self.pseudonym_mapping.get(value) {
            return existing.clone();
        }

        self.pseudonym_counter += 1;
        let pseudonym = match data_type {
            DataType::Email => format!("user{}@example.com", self.pseudonym_counter),
            DataType::Phone => format!("555-000-{:04}", self.pseudonym_counter % 10000),
            DataType::Name => format!("Person_{}", self.pseudonym_counter),
            DataType::IpAddress => format!("10.0.0.{}", self.pseudonym_counter % 256),
            _ => format!("PSEUDO_{}", self.pseudonym_counter),
        };

        self.pseudonym_mapping.insert(value.to_string(), pseudonym.clone());
        pseudonym
    }

    fn generalize_value(&self, value: &str, data_type: DataType) -> String {
        match data_type {
            DataType::Date => {
                // Generalize to year only
                if let Some(year) = value.split(&['/', '-'][..]).last() {
                    if year.len() == 4 {
                        return format!("YEAR_{}", year);
                    }
                }
                "[DATE]".to_string()
            }
            DataType::IpAddress => {
                // Generalize to subnet
                let parts: Vec<_> = value.split('.').collect();
                if parts.len() == 4 {
                    return format!("{}.{}.x.x", parts[0], parts[1]);
                }
                "[IP]".to_string()
            }
            _ => self.mask_value(value),
        }
    }

    fn mask_value(&self, value: &str) -> String {
        let len = value.len();
        if len <= 4 {
            "*".repeat(len)
        } else {
            format!("{}{}{}",
                &value[..2],
                "*".repeat(len - 4),
                &value[len - 2..])
        }
    }

    pub fn deanonymize(&self, text: &str, mapping: &HashMap<String, String>) -> String {
        let mut result = text.to_string();
        for (original, replacement) in mapping {
            result = result.replace(replacement, original);
        }
        result
    }

    pub fn reset_pseudonyms(&mut self) {
        self.pseudonym_mapping.clear();
        self.pseudonym_counter = 0;
    }
}

impl Default for DataAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch anonymizer for multiple texts
pub struct BatchAnonymizer {
    anonymizer: DataAnonymizer,
}

impl BatchAnonymizer {
    pub fn new() -> Self {
        Self {
            anonymizer: DataAnonymizer::new(),
        }
    }

    pub fn anonymize_all(&mut self, texts: &[&str]) -> Vec<AnonymizationResult> {
        texts.iter().map(|t| self.anonymizer.anonymize(t)).collect()
    }
}

impl Default for BatchAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_anonymization() {
        let mut anonymizer = DataAnonymizer::new();
        let result = anonymizer.anonymize("Contact me at john@example.com");

        assert!(!result.anonymized.contains("john@example.com"));
        assert!(result.detections.len() == 1);
    }

    #[test]
    fn test_phone_anonymization() {
        let mut anonymizer = DataAnonymizer::new();
        let result = anonymizer.anonymize("Call 555-123-4567 for info");

        assert!(!result.anonymized.contains("555-123-4567"));
    }

    #[test]
    fn test_pseudonymization() {
        let mut anonymizer = DataAnonymizer::new();
        anonymizer.add_rule(AnonymizationRule::new(DataType::Email, AnonymizationStrategy::Pseudonymize));

        let result1 = anonymizer.anonymize("Email: john@test.com");
        let result2 = anonymizer.anonymize("Contact: john@test.com");

        // Same email should get same pseudonym
        assert!(result1.mapping.get("john@test.com") == result2.mapping.get("john@test.com"));
    }

    #[test]
    fn test_multiple_types() {
        let mut anonymizer = DataAnonymizer::new();
        let text = "Email: user@test.com, Phone: 555-123-4567, IP: 192.168.1.1";
        let result = anonymizer.anonymize(text);

        assert!(result.detections.len() >= 3);
        assert!(!result.anonymized.contains("user@test.com"));
        assert!(!result.anonymized.contains("555-123-4567"));
        assert!(!result.anonymized.contains("192.168.1.1"));
    }
}
