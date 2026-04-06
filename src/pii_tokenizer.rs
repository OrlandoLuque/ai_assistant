// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Reversible PII masking system.
//!
//! Replaces real PII with numbered placeholders (e.g. `[email1]`, `[nombre2]`),
//! keeps the mapping locally, and can restore the original values in LLM responses.
//!
//! The token map **MUST NEVER** be sent over the network — it contains the real data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PiiCategory
// ---------------------------------------------------------------------------

/// Categories of personally-identifiable information that the tokenizer can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PiiCategory {
    Name,
    Email,
    Phone,
    Address,
    City,
    Date,
    Number,
    CreditCard,
    Custom,
}

impl std::fmt::Display for PiiCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            PiiCategory::Name => "nombre",
            PiiCategory::Email => "email",
            PiiCategory::Phone => "telefono",
            PiiCategory::Address => "direccion",
            PiiCategory::City => "ciudad",
            PiiCategory::Date => "fecha",
            PiiCategory::Number => "numero",
            PiiCategory::CreditCard => "tarjeta",
            PiiCategory::Custom => "custom",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// Privacy level & config
// ---------------------------------------------------------------------------

/// How aggressively to mask PII.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TokenizerPrivacyLevel {
    /// No masking at all.
    None,
    /// Mask names, emails, phones, addresses, credit cards (default).
    Tokenize,
    /// Everything in `Tokenize` plus numbers and dates.
    Aggressive,
    /// Everything in `Aggressive` plus noise added to placeholder counters.
    Paranoid,
}

/// Configuration for the [`PiiTokenizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Privacy level that controls which categories are detected.
    pub level: TokenizerPrivacyLevel,
    /// Additional user-supplied regex patterns mapped to a category.
    /// Each entry is `(regex_pattern, category)`.
    pub custom_patterns: Vec<(String, PiiCategory)>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            level: TokenizerPrivacyLevel::Tokenize,
            custom_patterns: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Token map alias
// ---------------------------------------------------------------------------

/// Maps placeholders such as `"[nombre1]"` to the real value `"Alfredo"`.
///
/// **NEVER** send this map over the network — it contains the real PII.
pub type PiiTokenMap = HashMap<String, String>;

// ---------------------------------------------------------------------------
// Stopwords — common English words that happen to be capitalised at sentence start
// ---------------------------------------------------------------------------

const STOP_WORDS: &[&str] = &[
    "The", "This", "That", "These", "Those", "There", "Their", "Then", "Than", "They", "What",
    "When", "Where", "Which", "While", "With", "Will", "Would", "Should", "Could", "About",
    "After", "Also", "Because", "Before", "Between", "Both", "Each", "Even", "Every", "From",
    "Have", "Here", "How", "Into", "Just", "Like", "Make", "Many", "More", "Most", "Much", "Must",
    "Never", "Only", "Other", "Over", "Same", "Some", "Such", "Sure", "Take", "Very", "Were",
    "Your", "Being", "Does", "Done", "Good", "Great", "Help", "Keep", "Know", "Last", "Long",
    "Look", "Made", "Need", "Next", "Part", "Real", "Right", "Said", "Still", "Think", "Time",
    "Under", "Used", "Want", "Well", "Work", "Year", "Call", "Come", "Find", "First", "Give",
    "High", "Little", "Live", "Name", "New", "Now", "Old", "Open", "Our", "Own", "See", "Set",
    "She", "Show", "Small", "Start", "Tell", "Three", "Try", "Turn", "Use", "Why", "And", "But",
    "For", "Not", "You", "All", "Any", "Are", "Can", "Had", "Has", "Her", "Him", "His", "Its",
    "Let", "May", "Nor", "Off", "One", "Out", "Put", "Run", "Say", "Too", "Two", "Was", "Way",
    "Who", "Yes", "Yet", // Spanish common words
    "El", "La", "Los", "Las", "Un", "Una", "Del", "Al", "Con", "Sin", "Por", "Para", "Como",
    "Pero", "Que", "Es", "En", "De", "No", "Se",
];

// ---------------------------------------------------------------------------
// PiiTokenizer
// ---------------------------------------------------------------------------

/// A reversible PII masking engine.
///
/// # Example
///
/// ```
/// use ai_assistant::pii_tokenizer::{PiiTokenizer, PiiTokenMap};
///
/// let mut tok = PiiTokenizer::with_default();
/// let (masked, map) = tok.mask("Contact alfredo@mail.com");
/// assert!(masked.contains("[email1]"));
/// let restored = PiiTokenizer::unmask(&masked, &map);
/// assert_eq!(restored, "Contact alfredo@mail.com");
/// ```
pub struct PiiTokenizer {
    config: TokenizerConfig,
    counter: HashMap<PiiCategory, usize>,
}

impl PiiTokenizer {
    /// Create a tokenizer with a custom configuration.
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            counter: HashMap::new(),
        }
    }

    /// Create a tokenizer with the default `Tokenize` privacy level.
    pub fn with_default() -> Self {
        Self::new(TokenizerConfig::default())
    }

    /// Mask PII in `text`, returning the masked text and the token map.
    ///
    /// The token map maps each placeholder (e.g. `[email1]`) to the real value.
    /// **NEVER send the token map over the network.**
    pub fn mask(&mut self, text: &str) -> (String, PiiTokenMap) {
        if text.is_empty() {
            return (String::new(), PiiTokenMap::new());
        }

        if self.config.level == TokenizerPrivacyLevel::None {
            return (text.to_string(), PiiTokenMap::new());
        }

        let mut result = text.to_string();
        let mut token_map = PiiTokenMap::new();

        // Order matters: longer/more specific patterns first to avoid partial matches.

        // 1. Credit cards  (always in Tokenize+)
        self.mask_pattern(
            &mut result,
            &mut token_map,
            PiiCategory::CreditCard,
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        );

        // 2. Emails
        self.mask_pattern(
            &mut result,
            &mut token_map,
            PiiCategory::Email,
            r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        );

        // 3. Phones
        self.mask_pattern(
            &mut result,
            &mut token_map,
            PiiCategory::Phone,
            r"\+?\d[\d\s\-()]{7,}\d",
        );

        // 4. Dates (only in Aggressive / Paranoid)
        if matches!(
            self.config.level,
            TokenizerPrivacyLevel::Aggressive | TokenizerPrivacyLevel::Paranoid
        ) {
            self.mask_pattern(
                &mut result,
                &mut token_map,
                PiiCategory::Date,
                r"\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}",
            );
        }

        // 5. Custom patterns
        let custom = self.config.custom_patterns.clone();
        for (pattern, category) in &custom {
            self.mask_pattern(&mut result, &mut token_map, *category, pattern);
        }

        // 6. Names — capitalised-word heuristic (Tokenize+)
        self.mask_names(&mut result, &mut token_map);

        // 7. Standalone numbers (only Aggressive / Paranoid)
        if matches!(
            self.config.level,
            TokenizerPrivacyLevel::Aggressive | TokenizerPrivacyLevel::Paranoid
        ) {
            self.mask_pattern(
                &mut result,
                &mut token_map,
                PiiCategory::Number,
                r"\b\d{3,}\b",
            );
        }

        (result, token_map)
    }

    /// Restore all placeholders in `text` using the given `token_map`.
    pub fn unmask(text: &str, token_map: &PiiTokenMap) -> String {
        let mut result = text.to_string();
        for (placeholder, real_value) in token_map {
            result = result.replace(placeholder.as_str(), real_value);
        }
        result
    }

    /// Reset the internal counters so the next `mask()` call starts from 1.
    pub fn reset(&mut self) {
        self.counter.clear();
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn next_placeholder(&mut self, cat: PiiCategory) -> String {
        let cnt = self.counter.entry(cat).or_insert(0);
        *cnt += 1;
        let idx = *cnt;

        if self.config.level == TokenizerPrivacyLevel::Paranoid {
            // Add a small noise offset so placeholder numbers are less predictable.
            let noise = (idx.wrapping_mul(7) ^ 0x5a) % 3;
            format!("[{}{}]", cat, idx + noise)
        } else {
            format!("[{}{}]", cat, idx)
        }
    }

    fn mask_pattern(
        &mut self,
        text: &mut String,
        map: &mut PiiTokenMap,
        cat: PiiCategory,
        pattern: &str,
    ) {
        let re = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return,
        };

        // Collect all matches first to avoid borrow issues.
        let matches: Vec<String> = re
            .find_iter(text.as_str())
            .map(|m| m.as_str().to_string())
            .collect();

        for mat in matches {
            let placeholder = self.next_placeholder(cat);
            map.insert(placeholder.clone(), mat.clone());
            // Replace only the first occurrence (it may appear multiple times).
            if let Some(pos) = text.find(&mat) {
                let end = pos + mat.len();
                text.replace_range(pos..end, &placeholder);
            }
        }
    }

    /// Heuristic name detection: consecutive capitalised words that are not
    /// common stop words and are not at the very start of a sentence.
    fn mask_names(&mut self, text: &mut String, map: &mut PiiTokenMap) {
        // Split into words, find runs of capitalised non-stop words.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return;
        }

        let mut names_to_mask: Vec<String> = Vec::new();
        let mut i = 0;

        while i < words.len() {
            let word = words[i];
            if self.is_name_candidate(word, i == 0) {
                // Accumulate consecutive capitalised name candidates.
                let start = i;
                let mut end = i + 1;
                while end < words.len() && self.is_name_candidate(words[end], false) {
                    end += 1;
                }
                // Only consider single capitalised words that are NOT at position 0,
                // or multi-word runs.
                if end - start >= 2 || (end - start == 1 && start > 0) {
                    for j in start..end {
                        let clean = words[j].trim_end_matches(|c: char| c.is_ascii_punctuation());
                        if !clean.is_empty() {
                            names_to_mask.push(clean.to_string());
                        }
                    }
                }
                i = end;
            } else {
                i += 1;
            }
        }

        // Deduplicate preserving order.
        let mut seen = std::collections::HashSet::new();
        let unique: Vec<String> = names_to_mask
            .into_iter()
            .filter(|n| seen.insert(n.clone()))
            .collect();

        for name in unique {
            let placeholder = self.next_placeholder(PiiCategory::Name);
            map.insert(placeholder.clone(), name.clone());
            // Replace all occurrences of this name as a whole word.
            let escaped = regex::escape(&name);
            let pattern = format!(r"\b{}\b", escaped);
            if let Ok(re) = regex::Regex::new(&pattern) {
                *text = re.replace_all(text, placeholder.as_str()).to_string();
            }
        }
    }

    /// Returns `true` if `word` looks like a proper name.
    fn is_name_candidate(&self, word: &str, _is_sentence_start: bool) -> bool {
        let clean = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
        if clean.is_empty() || clean.len() < 2 {
            return false;
        }
        let first = clean.chars().next().unwrap_or(' ');
        if !first.is_uppercase() {
            return false;
        }
        // Rest should be mostly lowercase.
        let rest_lower = clean[first.len_utf8()..]
            .chars()
            .all(|c| c.is_lowercase() || c == '-' || c == '\'');
        if !rest_lower {
            return false;
        }
        // Reject stop words (case-sensitive match).
        if STOP_WORDS.contains(&clean) {
            return false;
        }
        true
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_email() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("contact alfredo@mail.com");
        assert!(masked.contains("[email1]"), "masked = {}", masked);
        assert!(!masked.contains("alfredo@mail.com"));
        assert_eq!(map.get("[email1]").unwrap(), "alfredo@mail.com");
    }

    #[test]
    fn test_mask_phone() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("call +34 612 345 678");
        assert!(masked.contains("[telefono1]"), "masked = {}", masked);
        assert_eq!(map.get("[telefono1]").unwrap(), "+34 612 345 678");
    }

    #[test]
    fn test_mask_multiple_names() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("Hello John and Mary went");
        // Both names should be masked.
        assert!(masked.contains("[nombre"), "masked = {}", masked);
        assert!(!masked.contains("John"), "masked = {}", masked);
        assert!(!masked.contains("Mary"), "masked = {}", masked);
        // We should have at least 2 name entries.
        let name_entries: Vec<_> = map.keys().filter(|k| k.starts_with("[nombre")).collect();
        assert!(name_entries.len() >= 2, "map = {:?}", map);
    }

    #[test]
    fn test_mask_roundtrip() {
        let mut tok = PiiTokenizer::with_default();
        let original = "Email alfredo@mail.com, phone +34 612 345 678";
        let (masked, map) = tok.mask(original);
        let restored = PiiTokenizer::unmask(&masked, &map);
        assert_eq!(restored, original);
    }

    #[test]
    fn test_unmask_preserves_non_pii() {
        let text = "hello world, no PII here";
        let empty_map = PiiTokenMap::new();
        assert_eq!(PiiTokenizer::unmask(text, &empty_map), text);
    }

    #[test]
    fn test_mask_empty() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("");
        assert_eq!(masked, "");
        assert!(map.is_empty());
    }

    #[test]
    fn test_mask_no_pii() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("hello world");
        assert_eq!(masked, "hello world");
        assert!(map.is_empty());
    }

    #[test]
    fn test_mask_level_none() {
        let config = TokenizerConfig {
            level: TokenizerPrivacyLevel::None,
            custom_patterns: Vec::new(),
        };
        let mut tok = PiiTokenizer::new(config);
        let original = "alfredo@mail.com +34 612 345 678";
        let (masked, map) = tok.mask(original);
        assert_eq!(masked, original);
        assert!(map.is_empty());
    }

    #[test]
    fn test_mask_credit_card() {
        let mut tok = PiiTokenizer::with_default();
        let (masked, map) = tok.mask("card 4111-1111-1111-1111");
        assert!(masked.contains("[tarjeta1]"), "masked = {}", masked);
        assert_eq!(map.get("[tarjeta1]").unwrap(), "4111-1111-1111-1111");
    }

    #[test]
    fn test_mask_date() {
        let config = TokenizerConfig {
            level: TokenizerPrivacyLevel::Aggressive,
            custom_patterns: Vec::new(),
        };
        let mut tok = PiiTokenizer::new(config);
        let (masked, map) = tok.mask("born 15/03/1990");
        assert!(masked.contains("[fecha1]"), "masked = {}", masked);
        assert_eq!(map.get("[fecha1]").unwrap(), "15/03/1990");
    }
}
