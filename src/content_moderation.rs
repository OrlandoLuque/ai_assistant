//! Content moderation for AI inputs and outputs
//!
//! This module provides content moderation capabilities to detect
//! and filter inappropriate, harmful, or policy-violating content.
//!
//! # Features
//!
//! - **Category-based detection**: Detect various content categories
//! - **Configurable thresholds**: Adjust sensitivity per category
//! - **Multi-language support**: Detect content in multiple languages
//! - **Custom blocklists**: Add custom terms to block

use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Configuration for content moderation
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModerationConfig {
    /// Categories to moderate
    pub categories: Vec<ModerationCategory>,
    /// Action to take on violation
    pub action: ModerationAction,
    /// Confidence threshold for flagging (0-1)
    pub threshold: f64,
    /// Custom blocked terms
    pub blocked_terms: Vec<String>,
    /// Terms to allow (override detection)
    pub allowed_terms: Vec<String>,
    /// Enable fuzzy matching
    pub fuzzy_matching: bool,
}

impl Default for ModerationConfig {
    fn default() -> Self {
        Self {
            categories: vec![
                ModerationCategory::Hate,
                ModerationCategory::Violence,
                ModerationCategory::SelfHarm,
                ModerationCategory::Sexual,
                ModerationCategory::Harassment,
                ModerationCategory::Weapons,
                ModerationCategory::Drugs,
                ModerationCategory::Fraud,
                ModerationCategory::Illicit,
            ],
            action: ModerationAction::Flag,
            threshold: 0.7,
            blocked_terms: Vec::new(),
            allowed_terms: Vec::new(),
            fuzzy_matching: true,
        }
    }
}

/// Categories of content to moderate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModerationCategory {
    /// Hate speech
    Hate,
    /// Violence and threats
    Violence,
    /// Self-harm content
    SelfHarm,
    /// Sexual content
    Sexual,
    /// Harassment and bullying
    Harassment,
    /// Misinformation
    Misinformation,
    /// Spam
    Spam,
    /// Personal attacks
    PersonalAttack,
    /// Profanity
    Profanity,
    /// Drugs/controlled substances
    Drugs,
    /// Weapons
    Weapons,
    /// Financial fraud
    Fraud,
    /// Instructions for illegal activity (hacking, forgery, burglary,
    /// malware, evading detection). Distinct from Violence/Weapons: the
    /// harm is the *how-to seeking*, not the act described.
    Illicit,
    /// Custom category
    Custom,
}

impl ModerationCategory {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            ModerationCategory::Hate => "Hate Speech",
            ModerationCategory::Violence => "Violence",
            ModerationCategory::SelfHarm => "Self-Harm",
            ModerationCategory::Sexual => "Sexual Content",
            ModerationCategory::Harassment => "Harassment",
            ModerationCategory::Misinformation => "Misinformation",
            ModerationCategory::Spam => "Spam",
            ModerationCategory::PersonalAttack => "Personal Attack",
            ModerationCategory::Profanity => "Profanity",
            ModerationCategory::Drugs => "Drug Content",
            ModerationCategory::Weapons => "Weapons",
            ModerationCategory::Fraud => "Fraud",
            ModerationCategory::Illicit => "Illegal Activity",
            ModerationCategory::Custom => "Custom",
        }
    }
}

/// Actions to take on moderation violation
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModerationAction {
    /// Just flag the content
    Flag,
    /// Block the content entirely
    Block,
    /// Replace with placeholder
    Replace(String),
    /// Filter out specific parts
    Filter,
    /// Log only (no action)
    LogOnly,
}

/// A moderation flag
#[derive(Debug, Clone)]
pub struct ModerationFlag {
    /// Category of the flag
    pub category: ModerationCategory,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Matched text
    pub matched_text: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Explanation
    pub reason: String,
}

/// Result of moderation check
#[derive(Debug, Clone)]
pub struct ModerationResult {
    /// Original text
    pub original: String,
    /// Processed text (if action taken)
    pub processed: String,
    /// Whether content passed moderation
    pub passed: bool,
    /// Flags raised
    pub flags: Vec<ModerationFlag>,
    /// Scores by category
    pub category_scores: HashMap<ModerationCategory, f64>,
    /// Overall risk score
    pub risk_score: f64,
    /// Action taken
    pub action_taken: Option<ModerationAction>,
}

/// Content moderator
pub struct ContentModerator {
    config: ModerationConfig,
    /// Category patterns
    patterns: HashMap<ModerationCategory, Vec<PatternEntry>>,
    /// Blocked terms set (for fast lookup)
    blocked_set: HashSet<String>,
    /// Allowed terms set
    allowed_set: HashSet<String>,
}

struct PatternEntry {
    pattern: Regex,
    weight: f64,
    reason: String,
}

impl ContentModerator {
    /// Create a new content moderator
    pub fn new(config: ModerationConfig) -> Self {
        let blocked_set: HashSet<_> = config
            .blocked_terms
            .iter()
            .map(|t| t.to_lowercase())
            .collect();
        let allowed_set: HashSet<_> = config
            .allowed_terms
            .iter()
            .map(|t| t.to_lowercase())
            .collect();

        let mut moderator = Self {
            config,
            patterns: HashMap::new(),
            blocked_set,
            allowed_set,
        };
        moderator.init_patterns();
        moderator
    }

    fn init_patterns(&mut self) {
        // Violence patterns
        self.add_pattern(
            ModerationCategory::Violence,
            r"\b(?:kill|murder|attack|shoot|stab|bomb)\b",
            0.7,
            "Violent action term",
        );
        self.add_pattern(
            ModerationCategory::Violence,
            r"\b(?:i will|going to|gonna)\s+(?:kill|hurt|attack)\b",
            0.9,
            "Threat of violence",
        );

        // Hate speech patterns
        self.add_pattern(
            ModerationCategory::Hate,
            r"\b(?:all|every)\s+\w+\s+(?:are|is)\s+(?:bad|evil|terrible)\b",
            0.6,
            "Generalization about group",
        );

        // Self-harm patterns
        self.add_pattern(
            ModerationCategory::SelfHarm,
            r"\b(?:suicide|self-harm|cut myself|end my life)\b",
            0.8,
            "Self-harm reference",
        );
        self.add_pattern(
            ModerationCategory::SelfHarm,
            r"\b(?:want to|going to)\s+(?:die|end it|hurt myself)\b",
            0.9,
            "Self-harm intent",
        );

        // Harassment patterns
        self.add_pattern(
            ModerationCategory::Harassment,
            r"\b(?:you're|you are)\s+(?:stupid|idiot|moron|dumb)\b",
            0.7,
            "Personal insult",
        );

        // Spam patterns
        self.add_pattern(
            ModerationCategory::Spam,
            r"(?:buy now|click here|limited time|act now|free money)",
            0.6,
            "Spam terminology",
        );
        self.add_pattern(
            ModerationCategory::Spam,
            r"(.)\1{5,}",
            0.5,
            "Repeated characters",
        );

        // Profanity (basic - would need more comprehensive list)
        self.add_pattern(
            ModerationCategory::Profanity,
            r"\b(?:damn|hell|crap)\b",
            0.3,
            "Mild profanity",
        );

        // Harmful-instruction seeking. The patterns key on the *how-to
        // framing* combined with a harm noun — "how to make a bomb",
        // "ways to poison someone" — not on the nouns alone, so neutral
        // mentions ("the bomb squad arrived") stay clean.
        let howto = r"(?:how\s+to|ways?\s+to|methods?\s+(?:for|of)|guide\s+to|instructions?\s+for|techniques?\s+for|tips\s+for|detailed\s+guide\s+to)";

        // Weapons construction
        self.add_pattern(
            ModerationCategory::Weapons,
            &format!(
                r"{howto}\s+(?:\w+\s+){{0,3}}?(?:make|making|build|building|create|creating)?\s*(?:a\s+|an\s+)?(?:bomb|explosive|weapon|gun|silencer|firearm)"
            ),
            0.85,
            "Weapon construction instructions",
        );
        self.add_pattern(
            ModerationCategory::Weapons,
            r"\b(?:make|build|construct|assemble)\s+(?:a\s+|an\s+)?(?:bomb|explosive|pipe\s+bomb|silencer)\b",
            0.85,
            "Weapon construction phrasing",
        );

        // Drug synthesis / manufacture
        self.add_pattern(
            ModerationCategory::Drugs,
            r"\b(?:synthesiz(?:e|ing)|cook(?:ing)?|manufactur(?:e|ing)|produc(?:e|ing))\s+(?:\w+\s+){0,2}?(?:illegal\s+)?(?:drugs?|meth(?:amphetamine)?|narcotics|fentanyl)\b",
            0.85,
            "Drug synthesis instructions",
        );

        // Fraud / forgery / unauthorized access
        self.add_pattern(
            ModerationCategory::Fraud,
            r"\b(?:forg(?:e|ing)|counterfeit(?:ing)?|fak(?:e|ing))\s+(?:\w+\s+){0,2}?(?:documents?|id|identity|passports?|currency|money|signatures?)\b",
            0.8,
            "Forgery instructions",
        );
        self.add_pattern(
            ModerationCategory::Fraud,
            r"\bhack(?:ing)?\s+(?:into\s+)?(?:\w+(?:'s)?\s+){0,3}?(?:account|email|phone|computer|network|password|system)\b",
            0.8,
            "Unauthorized access instructions",
        );
        self.add_pattern(
            ModerationCategory::Fraud,
            r"\bidentity\s+theft\b",
            0.75,
            "Identity theft reference",
        );

        // Other illegal-activity instruction seeking
        self.add_pattern(
            ModerationCategory::Illicit,
            r"\b(?:create|creating|make|making|write|writing)\s+(?:a\s+)?(?:computer\s+)?(?:virus|malware|ransomware|worm|trojan)\b",
            0.85,
            "Malware creation instructions",
        );
        self.add_pattern(
            ModerationCategory::Illicit,
            r"\bbreak(?:ing)?\s+into\s+(?:a\s+|an\s+)?(?:house|home|building|car|apartment)\b",
            0.8,
            "Burglary instructions",
        );
        self.add_pattern(
            ModerationCategory::Illicit,
            r"\b(?:poison(?:ing)?|drug(?:ging)?)\s+(?:someone|somebody|a\s+person|people)\b",
            0.85,
            "Poisoning reference",
        );
        self.add_pattern(
            ModerationCategory::Illicit,
            r"\b(?:stalk(?:ing)?|track(?:ing)?)\s+(?:someone|somebody|a\s+person)\b",
            0.75,
            "Stalking reference",
        );
        // Deliberately below the default 0.7 threshold: detection-evasion
        // framing appears in legitimate text ("merged without being
        // noticed"), so alone it contributes to risk_score but never flags.
        self.add_pattern(
            ModerationCategory::Illicit,
            r"\bwithout\s+(?:being\s+)?(?:detect(?:ed|ion)|caught|noticed|a\s+trace)\b",
            0.65,
            "Detection-evasion framing",
        );
    }

    fn add_pattern(
        &mut self,
        category: ModerationCategory,
        pattern: &str,
        weight: f64,
        reason: &str,
    ) {
        // Security: limit regex DFA size to prevent ReDoS (H7)
        match regex::RegexBuilder::new(&format!("(?i){}", pattern))
            .size_limit(1_000_000)
            .dfa_size_limit(1_000_000)
            .build()
        {
            Ok(re) => {
                self.patterns
                    .entry(category)
                    .or_insert_with(Vec::new)
                    .push(PatternEntry {
                        pattern: re,
                        weight,
                        reason: reason.to_string(),
                    });
            }
            Err(_) => {
                // Reject patterns that are too complex or invalid
            }
        }
    }

    /// Moderate content
    pub fn moderate(&self, text: &str) -> ModerationResult {
        let mut flags = Vec::new();
        let mut category_scores: HashMap<ModerationCategory, f64> = HashMap::new();

        // Check blocked terms first
        for term in &self.blocked_set {
            if let Some((start, end)) = crate::text_util::find_ci_range(text, term) {
                // Check if in allowed list
                if !self.allowed_set.contains(term) {
                    flags.push(ModerationFlag {
                        category: ModerationCategory::Custom,
                        confidence: 1.0,
                        matched_text: text[start..end].to_string(),
                        start,
                        end,
                        reason: "Blocked term".to_string(),
                    });
                    *category_scores
                        .entry(ModerationCategory::Custom)
                        .or_insert(0.0) = 1.0;
                }
            }
        }

        // Check patterns for each category
        for category in &self.config.categories {
            if let Some(patterns) = self.patterns.get(category) {
                let mut max_score = 0.0;

                for entry in patterns {
                    for m in entry.pattern.find_iter(text) {
                        let matched = m.as_str().to_string();

                        // Check if matched term is allowed
                        if self.allowed_set.contains(&matched.to_lowercase()) {
                            continue;
                        }

                        let confidence = entry.weight;

                        if confidence > max_score {
                            max_score = confidence;
                        }

                        if confidence >= self.config.threshold {
                            flags.push(ModerationFlag {
                                category: *category,
                                confidence,
                                matched_text: matched,
                                start: m.start(),
                                end: m.end(),
                                reason: entry.reason.clone(),
                            });
                        }
                    }
                }

                if max_score > 0.0 {
                    category_scores.insert(*category, max_score);
                }
            }
        }

        // Calculate risk score
        let risk_score = if category_scores.is_empty() {
            0.0
        } else {
            category_scores.values().sum::<f64>() / category_scores.len() as f64
        };

        // Determine if passed
        let passed = flags.is_empty() || flags.iter().all(|f| f.confidence < self.config.threshold);

        // Process text based on action
        let (processed, action_taken) = if !passed {
            match &self.config.action {
                ModerationAction::Block => (
                    "[CONTENT BLOCKED]".to_string(),
                    Some(ModerationAction::Block),
                ),
                ModerationAction::Replace(replacement) => (
                    replacement.clone(),
                    Some(ModerationAction::Replace(replacement.clone())),
                ),
                ModerationAction::Filter => {
                    let filtered = self.filter_content(text, &flags);
                    (filtered, Some(ModerationAction::Filter))
                }
                ModerationAction::Flag | ModerationAction::LogOnly => {
                    (text.to_string(), Some(self.config.action.clone()))
                }
            }
        } else {
            (text.to_string(), None)
        };

        ModerationResult {
            original: text.to_string(),
            processed,
            passed,
            flags,
            category_scores,
            risk_score,
            action_taken,
        }
    }

    /// Filter out flagged content
    fn filter_content(&self, text: &str, flags: &[ModerationFlag]) -> String {
        let mut result = text.to_string();
        let mut sorted_flags = flags.to_vec();
        sorted_flags.sort_by_key(|e| std::cmp::Reverse(e.start));

        for flag in sorted_flags {
            let replacement = format!("[{}]", flag.category.display_name().to_uppercase());
            result.replace_range(flag.start..flag.end, &replacement);
        }

        result
    }

    /// Add blocked term
    pub fn add_blocked_term(&mut self, term: impl Into<String>) {
        let term = term.into().to_lowercase();
        self.blocked_set.insert(term);
    }

    /// Add allowed term
    pub fn add_allowed_term(&mut self, term: impl Into<String>) {
        let term = term.into().to_lowercase();
        self.allowed_set.insert(term);
    }

    /// Check if text would pass moderation
    pub fn would_pass(&self, text: &str) -> bool {
        self.moderate(text).passed
    }

    /// Get risk level description
    pub fn risk_level(score: f64) -> &'static str {
        if score < 0.3 {
            "Low"
        } else if score < 0.6 {
            "Medium"
        } else if score < 0.8 {
            "High"
        } else {
            "Critical"
        }
    }

    /// Moderate an image based on its referenced URL and (optionally)
    /// caller-supplied OCR text.
    ///
    /// The library does not bundle OCR; pair this with an external OCR
    /// pipeline (tesseract subprocess, cloud vision, on-device) and pass
    /// the extracted text via `ocr_text`. With `ocr_text = None` only the
    /// URL is moderated (catches malicious link domains, blocked terms in
    /// signed URLs); base64-encoded images contribute nothing on their own.
    #[cfg(feature = "vision")]
    pub fn moderate_image(
        &self,
        image: &crate::vision::ImageInput,
        ocr_text: Option<&str>,
    ) -> ModerationResult {
        let mut combined = String::new();
        if let crate::vision::ImageData::Url(url) = &image.data {
            combined.push_str(url);
            combined.push(' ');
        }
        if let Some(text) = ocr_text {
            combined.push_str(text);
        }
        self.moderate(&combined)
    }
}

impl Default for ContentModerator {
    fn default() -> Self {
        Self::new(ModerationConfig::default())
    }
}

/// Moderation statistics
#[derive(Debug, Clone, Default)]
pub struct ModerationStats {
    /// Total checks performed
    pub total_checks: usize,
    /// Checks that passed
    pub passed: usize,
    /// Checks that failed
    pub failed: usize,
    /// Counts by category
    pub category_counts: HashMap<ModerationCategory, usize>,
    /// Average risk score
    pub avg_risk_score: f64,
}

impl ModerationStats {
    /// Record a moderation result
    pub fn record(&mut self, result: &ModerationResult) {
        self.total_checks += 1;

        if result.passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }

        for flag in &result.flags {
            *self.category_counts.entry(flag.category).or_insert(0) += 1;
        }

        // Update average
        self.avg_risk_score = (self.avg_risk_score * (self.total_checks - 1) as f64
            + result.risk_score)
            / self.total_checks as f64;
    }

    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_checks == 0 {
            1.0
        } else {
            self.passed as f64 / self.total_checks as f64
        }
    }
}

/// Builder for moderation configuration
#[non_exhaustive]
pub struct ModerationConfigBuilder {
    config: ModerationConfig,
}

impl ModerationConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ModerationConfig::default(),
        }
    }

    /// Set categories to moderate
    pub fn categories(mut self, categories: Vec<ModerationCategory>) -> Self {
        self.config.categories = categories;
        self
    }

    /// Set action on violation
    pub fn action(mut self, action: ModerationAction) -> Self {
        self.config.action = action;
        self
    }

    /// Set threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Add blocked term
    pub fn block_term(mut self, term: impl Into<String>) -> Self {
        self.config.blocked_terms.push(term.into());
        self
    }

    /// Add allowed term
    pub fn allow_term(mut self, term: impl Into<String>) -> Self {
        self.config.allowed_terms.push(term.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> ModerationConfig {
        self.config
    }
}

impl Default for ModerationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_content() {
        let moderator = ContentModerator::default();
        let result = moderator.moderate("Hello, how are you today?");

        assert!(result.passed);
        assert!(result.flags.is_empty());
    }

    #[test]
    fn test_violence_detection() {
        let moderator = ContentModerator::default();
        let result = moderator.moderate("I will attack the server with requests");

        // "attack" alone might not trigger, depends on context
        // But "I will attack" pattern should match
        assert!(!result.flags.is_empty() || result.risk_score > 0.0);
    }

    #[test]
    fn test_blocked_terms() {
        let config = ModerationConfig {
            blocked_terms: vec!["badword".to_string()],
            ..Default::default()
        };
        let moderator = ContentModerator::new(config);
        let result = moderator.moderate("This contains badword in it");

        assert!(!result.passed);
        assert!(result
            .flags
            .iter()
            .any(|f| f.category == ModerationCategory::Custom));
    }

    #[test]
    fn test_allowed_terms() {
        let config = ModerationConfig {
            blocked_terms: vec!["kill".to_string()],
            allowed_terms: vec!["kill".to_string()], // Allow overrides block
            ..Default::default()
        };
        let moderator = ContentModerator::new(config);
        let result = moderator.moderate("The process will kill the zombie process");

        // Should pass because "kill" is allowed
        assert!(result.flags.is_empty() || result.passed);
    }

    #[test]
    fn test_filter_action() {
        let config = ModerationConfig {
            action: ModerationAction::Filter,
            threshold: 0.5,
            ..Default::default()
        };
        let moderator = ContentModerator::new(config);
        let result = moderator.moderate("You're stupid and an idiot");

        if !result.passed {
            assert!(result.processed.contains("["));
        }
    }

    #[test]
    fn test_risk_level() {
        assert_eq!(ContentModerator::risk_level(0.1), "Low");
        assert_eq!(ContentModerator::risk_level(0.5), "Medium");
        assert_eq!(ContentModerator::risk_level(0.7), "High");
        assert_eq!(ContentModerator::risk_level(0.9), "Critical");
    }

    #[test]
    fn test_stats() {
        let mut stats = ModerationStats::default();
        let moderator = ContentModerator::default();

        let result1 = moderator.moderate("Hello world");
        stats.record(&result1);

        assert_eq!(stats.total_checks, 1);
        assert_eq!(stats.passed, 1);
    }

    #[test]
    fn test_config_builder() {
        let config = ModerationConfigBuilder::new()
            .categories(vec![ModerationCategory::Violence, ModerationCategory::Hate])
            .action(ModerationAction::Block)
            .threshold(0.8)
            .block_term("forbidden")
            .build();

        assert_eq!(config.categories.len(), 2);
        assert_eq!(config.action, ModerationAction::Block);
    }

    #[test]
    fn test_category_display_names() {
        assert_eq!(ModerationCategory::Hate.display_name(), "Hate Speech");
        assert_eq!(ModerationCategory::Violence.display_name(), "Violence");
        assert_eq!(ModerationCategory::SelfHarm.display_name(), "Self-Harm");
        assert_eq!(ModerationCategory::Sexual.display_name(), "Sexual Content");
        assert_eq!(ModerationCategory::Harassment.display_name(), "Harassment");
        assert_eq!(
            ModerationCategory::Misinformation.display_name(),
            "Misinformation"
        );
        assert_eq!(ModerationCategory::Spam.display_name(), "Spam");
        assert_eq!(
            ModerationCategory::PersonalAttack.display_name(),
            "Personal Attack"
        );
        assert_eq!(ModerationCategory::Profanity.display_name(), "Profanity");
        assert_eq!(ModerationCategory::Drugs.display_name(), "Drug Content");
        assert_eq!(ModerationCategory::Weapons.display_name(), "Weapons");
        assert_eq!(ModerationCategory::Fraud.display_name(), "Fraud");
        assert_eq!(ModerationCategory::Custom.display_name(), "Custom");
    }

    #[test]
    fn test_stats_pass_rate_and_multi_record() {
        let mut stats = ModerationStats::default();

        // Initially pass rate should be 1.0 (no checks)
        assert!((stats.pass_rate() - 1.0).abs() < f64::EPSILON);

        let moderator = ContentModerator::default();

        // Record a passing result
        let pass_result = moderator.moderate("Hello, how are you?");
        stats.record(&pass_result);
        assert_eq!(stats.total_checks, 1);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.failed, 0);
        assert!((stats.pass_rate() - 1.0).abs() < f64::EPSILON);

        // Record a failing result (blocked term)
        let config = ModerationConfig {
            blocked_terms: vec!["forbidden".to_string()],
            ..Default::default()
        };
        let strict_moderator = ContentModerator::new(config);
        let fail_result = strict_moderator.moderate("This is forbidden content");
        stats.record(&fail_result);

        assert_eq!(stats.total_checks, 2);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.failed, 1);
        assert!((stats.pass_rate() - 0.5).abs() < f64::EPSILON);
        assert!(stats.avg_risk_score > 0.0);
    }

    // === Batch 13 — image-aware moderation surface ===

    #[cfg(feature = "vision")]
    #[test]
    fn test_moderate_image_blocks_url_with_blocked_term() {
        let cfg = ModerationConfig {
            blocked_terms: vec!["malware".to_string()],
            ..Default::default()
        };
        let mod_ = ContentModerator::new(cfg);
        let img = crate::vision::ImageInput::from_url("https://evil.example.com/malware.png");
        let result = mod_.moderate_image(&img, None);
        assert!(
            !result.passed,
            "URL containing blocked term must be flagged"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_moderate_image_with_ocr_text_flagged() {
        let cfg = ModerationConfig {
            blocked_terms: vec!["forbidden".to_string()],
            ..Default::default()
        };
        let mod_ = ContentModerator::new(cfg);
        let img = crate::vision::ImageInput::from_base64("AAAA", "image/png");
        let result = mod_.moderate_image(&img, Some("contains forbidden content"));
        assert!(!result.passed);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_moderate_image_clean_passes() {
        let mod_ = ContentModerator::new(ModerationConfig::default());
        let img = crate::vision::ImageInput::from_base64("AAAA", "image/png");
        let result = mod_.moderate_image(&img, Some("a sunny landscape with mountains"));
        assert!(result.passed);
    }
}
