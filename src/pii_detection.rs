//! PII Detection and Redaction
//!
//! This module provides Personal Identifiable Information (PII) detection
//! and redaction capabilities for AI inputs and outputs.
//!
//! # Features
//!
//! - **Pattern-based detection**: Detect emails, phones, SSN, etc.
//! - **Context-aware detection**: Use context to identify PII
//! - **Configurable redaction**: Replace, mask, or hash PII
//! - **Audit logging**: Track what was redacted

use regex::Regex;
use std::collections::HashMap;

/// Configuration for PII detection
#[derive(Debug, Clone)]
pub struct PiiConfig {
    /// PII types to detect
    pub detect_types: Vec<PiiType>,
    /// Redaction strategy
    pub redaction: RedactionStrategy,
    /// Sensitivity level (higher = more aggressive detection)
    pub sensitivity: SensitivityLevel,
    /// Log detected PII (for audit)
    pub log_detections: bool,
    /// Custom patterns
    pub custom_patterns: Vec<CustomPiiPattern>,
}

impl Default for PiiConfig {
    fn default() -> Self {
        Self {
            detect_types: vec![
                PiiType::Email,
                PiiType::Phone,
                PiiType::Ssn,
                PiiType::CreditCard,
                PiiType::IpAddress,
                PiiType::Name,
            ],
            redaction: RedactionStrategy::Replace,
            sensitivity: SensitivityLevel::Medium,
            log_detections: true,
            custom_patterns: Vec::new(),
        }
    }
}

/// Types of PII
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiType {
    /// Email addresses
    Email,
    /// Phone numbers
    Phone,
    /// Social Security Numbers
    Ssn,
    /// Credit card numbers
    CreditCard,
    /// IP addresses
    IpAddress,
    /// Names (requires context)
    Name,
    /// Addresses
    Address,
    /// Date of birth
    DateOfBirth,
    /// Passport numbers
    Passport,
    /// Driver's license
    DriversLicense,
    /// Bank account numbers
    BankAccount,
    /// Medical record numbers
    MedicalId,
    /// Custom type
    Custom,
}

impl PiiType {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            PiiType::Email => "Email",
            PiiType::Phone => "Phone",
            PiiType::Ssn => "SSN",
            PiiType::CreditCard => "Credit Card",
            PiiType::IpAddress => "IP Address",
            PiiType::Name => "Name",
            PiiType::Address => "Address",
            PiiType::DateOfBirth => "Date of Birth",
            PiiType::Passport => "Passport",
            PiiType::DriversLicense => "Driver's License",
            PiiType::BankAccount => "Bank Account",
            PiiType::MedicalId => "Medical ID",
            PiiType::Custom => "Custom",
        }
    }
}

/// Redaction strategies
#[derive(Debug, Clone)]
pub enum RedactionStrategy {
    /// Replace with placeholder like [EMAIL]
    Replace,
    /// Mask with asterisks: j***@example.com
    Mask,
    /// Hash the value
    Hash,
    /// Remove entirely
    Remove,
    /// Custom replacement
    Custom(String),
}

/// Sensitivity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityLevel {
    /// Only high-confidence matches
    Low,
    /// Balanced detection
    Medium,
    /// Aggressive detection (may have false positives)
    High,
}

/// Custom PII pattern
#[derive(Debug, Clone)]
pub struct CustomPiiPattern {
    /// Pattern name
    pub name: String,
    /// Regex pattern
    pub pattern: String,
    /// Replacement text
    pub replacement: String,
}

/// A detected PII item
#[derive(Debug, Clone)]
pub struct DetectedPii {
    /// Type of PII
    pub pii_type: PiiType,
    /// Original value
    pub value: String,
    /// Redacted value
    pub redacted: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Confidence score (0-1)
    pub confidence: f64,
}

/// Result of PII detection
#[derive(Debug, Clone)]
pub struct PiiResult {
    /// Original text
    pub original: String,
    /// Redacted text
    pub redacted: String,
    /// List of detected PII
    pub detections: Vec<DetectedPii>,
    /// Count by type
    pub counts: HashMap<PiiType, usize>,
    /// Whether any PII was found
    pub has_pii: bool,
}

/// PII detector
pub struct PiiDetector {
    config: PiiConfig,
    /// Compiled patterns
    patterns: HashMap<PiiType, Regex>,
    /// Common first names (for name detection)
    common_names: Vec<String>,
}

impl PiiDetector {
    /// Create a new PII detector
    pub fn new(config: PiiConfig) -> Self {
        let mut detector = Self {
            config,
            patterns: HashMap::new(),
            common_names: Vec::new(),
        };
        detector.compile_patterns();
        detector.load_common_names();
        detector
    }

    fn compile_patterns(&mut self) {
        // Email pattern
        self.patterns.insert(
            PiiType::Email,
            Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").expect("valid regex"),
        );

        // Phone patterns (various formats)
        self.patterns.insert(
            PiiType::Phone,
            Regex::new(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}")
                .expect("valid regex"),
        );

        // SSN pattern
        self.patterns.insert(
            PiiType::Ssn,
            Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b").expect("valid regex"),
        );

        // Credit card pattern (simplified)
        self.patterns.insert(
            PiiType::CreditCard,
            Regex::new(r"\b(?:\d{4}[-\s]?){3}\d{4}\b").expect("valid regex"),
        );

        // IP address pattern (IPv4)
        self.patterns.insert(
            PiiType::IpAddress,
            Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").expect("valid regex"),
        );

        // Date of birth patterns
        self.patterns.insert(
            PiiType::DateOfBirth,
            Regex::new(r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")
                .expect("valid regex"),
        );
    }

    fn load_common_names(&mut self) {
        // Common first names for name detection
        let names = [
            "james",
            "john",
            "robert",
            "michael",
            "william",
            "david",
            "richard",
            "joseph",
            "thomas",
            "charles",
            "mary",
            "patricia",
            "jennifer",
            "linda",
            "elizabeth",
            "barbara",
            "susan",
            "jessica",
            "sarah",
            "karen",
            "alice",
            "bob",
            "charlie",
        ];
        self.common_names = names.iter().map(|s| s.to_string()).collect();
    }

    /// Detect and redact PII in text
    pub fn detect(&self, text: &str) -> PiiResult {
        #[allow(unused_assignments)]
        let mut detections = Vec::new();
        let mut redacted = text.to_string();
        let mut counts: HashMap<PiiType, usize> = HashMap::new();

        // Sort detections by position (descending) for proper redaction
        let mut all_detections = Vec::new();

        for pii_type in &self.config.detect_types {
            if let Some(pattern) = self.patterns.get(pii_type) {
                for m in pattern.find_iter(text) {
                    let value = m.as_str().to_string();
                    let confidence = self.calculate_confidence(*pii_type, &value);

                    if confidence >= self.min_confidence() {
                        all_detections.push(DetectedPii {
                            pii_type: *pii_type,
                            value: value.clone(),
                            redacted: self.redact_value(*pii_type, &value),
                            start: m.start(),
                            end: m.end(),
                            confidence,
                        });
                    }
                }
            }
        }

        // Name detection (context-based)
        if self.config.detect_types.contains(&PiiType::Name) {
            let name_detections = self.detect_names(text);
            all_detections.extend(name_detections);
        }

        // Custom patterns
        for custom in &self.config.custom_patterns {
            if let Ok(re) = Regex::new(&custom.pattern) {
                for m in re.find_iter(text) {
                    all_detections.push(DetectedPii {
                        pii_type: PiiType::Custom,
                        value: m.as_str().to_string(),
                        redacted: custom.replacement.clone(),
                        start: m.start(),
                        end: m.end(),
                        confidence: 1.0,
                    });
                }
            }
        }

        // Sort by start position descending for proper replacement
        all_detections.sort_by(|a, b| b.start.cmp(&a.start));

        // Apply redactions
        for detection in &all_detections {
            redacted.replace_range(detection.start..detection.end, &detection.redacted);
            *counts.entry(detection.pii_type).or_insert(0) += 1;
        }

        // Sort back by position ascending for output
        all_detections.sort_by_key(|d| d.start);
        detections = all_detections;

        PiiResult {
            original: text.to_string(),
            redacted,
            detections: detections.clone(),
            counts,
            has_pii: !detections.is_empty(),
        }
    }

    /// Detect names using context and common name list
    fn detect_names(&self, text: &str) -> Vec<DetectedPii> {
        let mut detections = Vec::new();

        // Look for "Mr.", "Mrs.", "Ms.", "Dr." patterns
        let title_pattern =
            Regex::new(r"\b(?:Mr|Mrs|Ms|Dr|Miss)\.?\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?")
                .expect("valid regex");

        for cap in title_pattern.captures_iter(text) {
            let full_match = cap.get(0).expect("capture group 0");
            let name = full_match.as_str();

            detections.push(DetectedPii {
                pii_type: PiiType::Name,
                value: name.to_string(),
                redacted: self.redact_value(PiiType::Name, name),
                start: full_match.start(),
                end: full_match.end(),
                confidence: 0.9,
            });
        }

        // Look for common first names followed by capitalized word
        if self.config.sensitivity == SensitivityLevel::High {
            let words: Vec<_> = text.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let word_lower = word.to_lowercase();
                let word_clean = word_lower.trim_matches(|c: char| !c.is_alphabetic());

                if self.common_names.contains(&word_clean.to_string()) {
                    // Check if next word is also capitalized (potential last name)
                    if let Some(next) = words.get(i + 1) {
                        if next
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false)
                        {
                            if let Some(start) = text.find(*word) {
                                let full_name = format!("{} {}", word, next);
                                let end = start + full_name.len();

                                detections.push(DetectedPii {
                                    pii_type: PiiType::Name,
                                    value: full_name.clone(),
                                    redacted: self.redact_value(PiiType::Name, &full_name),
                                    start,
                                    end,
                                    confidence: 0.6,
                                });
                            }
                        }
                    }
                }
            }
        }

        detections
    }

    /// Calculate confidence for a detection
    fn calculate_confidence(&self, pii_type: PiiType, value: &str) -> f64 {
        match pii_type {
            PiiType::Email => {
                // Higher confidence for common email domains
                if value.contains("@gmail.com")
                    || value.contains("@yahoo.com")
                    || value.contains("@hotmail.com")
                {
                    0.95
                } else {
                    0.85
                }
            }
            PiiType::Phone => {
                // Check if it looks like a valid phone number
                let digits: String = value.chars().filter(|c| c.is_numeric()).collect();
                if digits.len() >= 10 && digits.len() <= 11 {
                    0.9
                } else {
                    0.6
                }
            }
            PiiType::Ssn => {
                // SSN validation: area number, group number, serial number
                let digits: String = value.chars().filter(|c| c.is_numeric()).collect();
                if digits.len() == 9 && !digits.starts_with("000") && !digits.starts_with("666") {
                    0.85
                } else {
                    0.5
                }
            }
            PiiType::CreditCard => {
                // Luhn check
                let digits: String = value.chars().filter(|c| c.is_numeric()).collect();
                if digits.len() >= 13 && digits.len() <= 19 && self.luhn_check(&digits) {
                    0.95
                } else {
                    0.4
                }
            }
            PiiType::IpAddress => {
                // Validate octets
                let parts: Vec<_> = value.split('.').collect();
                if parts.len() == 4 {
                    let valid = parts.iter().all(|p| p.parse::<u8>().is_ok());
                    if valid {
                        0.9
                    } else {
                        0.3
                    }
                } else {
                    0.3
                }
            }
            _ => 0.7,
        }
    }

    /// Luhn algorithm for credit card validation
    fn luhn_check(&self, number: &str) -> bool {
        let digits: Vec<u32> = number.chars().filter_map(|c| c.to_digit(10)).collect();
        if digits.is_empty() {
            return false;
        }

        let checksum: u32 = digits
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &d)| {
                if i % 2 == 1 {
                    let doubled = d * 2;
                    if doubled > 9 {
                        doubled - 9
                    } else {
                        doubled
                    }
                } else {
                    d
                }
            })
            .sum();

        checksum % 10 == 0
    }

    /// Get minimum confidence based on sensitivity
    fn min_confidence(&self) -> f64 {
        match self.config.sensitivity {
            SensitivityLevel::Low => 0.9,
            SensitivityLevel::Medium => 0.7,
            SensitivityLevel::High => 0.5,
        }
    }

    /// Redact a value based on strategy
    fn redact_value(&self, pii_type: PiiType, value: &str) -> String {
        match &self.config.redaction {
            RedactionStrategy::Replace => format!("[{}]", pii_type.display_name().to_uppercase()),
            RedactionStrategy::Mask => self.mask_value(value),
            RedactionStrategy::Hash => self.hash_value(value),
            RedactionStrategy::Remove => String::new(),
            RedactionStrategy::Custom(replacement) => replacement.clone(),
        }
    }

    /// Mask a value with asterisks
    fn mask_value(&self, value: &str) -> String {
        let len = value.len();
        if len <= 4 {
            "*".repeat(len)
        } else {
            let show = (len / 4).max(1);
            format!(
                "{}{}{}",
                &value[..show],
                "*".repeat(len - show * 2),
                &value[len - show..]
            )
        }
    }

    /// Hash a value
    fn hash_value(&self, value: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        format!("[HASH:{}]", hasher.finish())
    }
}

impl Default for PiiDetector {
    fn default() -> Self {
        Self::new(PiiConfig::default())
    }
}

/// Builder for PII configuration
pub struct PiiConfigBuilder {
    config: PiiConfig,
}

impl PiiConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PiiConfig::default(),
        }
    }

    /// Set detection types
    pub fn detect(mut self, types: Vec<PiiType>) -> Self {
        self.config.detect_types = types;
        self
    }

    /// Set redaction strategy
    pub fn redaction(mut self, strategy: RedactionStrategy) -> Self {
        self.config.redaction = strategy;
        self
    }

    /// Set sensitivity level
    pub fn sensitivity(mut self, level: SensitivityLevel) -> Self {
        self.config.sensitivity = level;
        self
    }

    /// Add custom pattern
    pub fn custom_pattern(mut self, name: &str, pattern: &str, replacement: &str) -> Self {
        self.config.custom_patterns.push(CustomPiiPattern {
            name: name.to_string(),
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
        });
        self
    }

    /// Build the configuration
    pub fn build(self) -> PiiConfig {
        self.config
    }
}

impl Default for PiiConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_detection() {
        let detector = PiiDetector::default();
        let result = detector.detect("Contact me at john@example.com");

        assert!(result.has_pii);
        assert!(result
            .detections
            .iter()
            .any(|d| d.pii_type == PiiType::Email));
    }

    #[test]
    fn test_phone_detection() {
        let detector = PiiDetector::default();
        let result = detector.detect("Call me at 555-123-4567");

        assert!(result.has_pii);
        assert!(result
            .detections
            .iter()
            .any(|d| d.pii_type == PiiType::Phone));
    }

    #[test]
    fn test_ssn_detection() {
        let detector = PiiDetector::default();
        let result = detector.detect("My SSN is 123-45-6789");

        assert!(result.has_pii);
        assert!(result.detections.iter().any(|d| d.pii_type == PiiType::Ssn));
    }

    #[test]
    fn test_redaction_replace() {
        let detector = PiiDetector::default();
        let result = detector.detect("Email: test@example.com");

        assert!(result.redacted.contains("[EMAIL]"));
        assert!(!result.redacted.contains("test@example.com"));
    }

    #[test]
    fn test_redaction_mask() {
        let config = PiiConfig {
            redaction: RedactionStrategy::Mask,
            ..Default::default()
        };
        let detector = PiiDetector::new(config);
        let result = detector.detect("Email: test@example.com");

        assert!(result.redacted.contains("*"));
    }

    #[test]
    fn test_credit_card_luhn() {
        let detector = PiiDetector::default();

        // Valid test card number
        assert!(detector.luhn_check("4532015112830366"));
        // Invalid number
        assert!(!detector.luhn_check("1234567890123456"));
    }

    #[test]
    fn test_no_pii() {
        let detector = PiiDetector::default();
        let result = detector.detect("This is a normal sentence.");

        assert!(!result.has_pii);
        assert!(result.detections.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = PiiConfigBuilder::new()
            .detect(vec![PiiType::Email, PiiType::Phone])
            .redaction(RedactionStrategy::Hash)
            .sensitivity(SensitivityLevel::High)
            .build();

        assert_eq!(config.detect_types.len(), 2);
    }
}
