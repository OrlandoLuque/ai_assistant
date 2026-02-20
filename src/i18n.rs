//! Internationalization and multi-language support
//!
//! This module provides language detection, translation hints, and
//! localization utilities for AI conversations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Language Detection
// ============================================================================

/// Detected language with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLanguage {
    /// ISO 639-1 language code
    pub code: String,
    /// Full language name
    pub name: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Detected script (Latin, Cyrillic, etc.)
    pub script: Option<String>,
}

/// Language detector using character and word patterns
pub struct LanguageDetector {
    /// Language-specific patterns
    patterns: HashMap<String, LanguagePatterns>,
}

#[derive(Debug, Clone)]
struct LanguagePatterns {
    /// Common words in this language
    common_words: Vec<&'static str>,
    /// Unique characters or patterns
    unique_chars: Vec<char>,
    /// Script type
    script: &'static str,
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageDetector {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // English
        patterns.insert(
            "en".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "the", "is", "are", "was", "were", "have", "has", "been", "be", "to", "of",
                    "and", "in", "that", "it", "with", "for", "on", "this", "can",
                ],
                unique_chars: vec![],
                script: "Latin",
            },
        );

        // Spanish
        patterns.insert(
            "es".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "el", "la", "los", "las", "un", "una", "es", "son", "que", "de", "en", "por",
                    "para", "con", "como", "pero", "si", "se", "su", "yo",
                ],
                unique_chars: vec!['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡'],
                script: "Latin",
            },
        );

        // French
        patterns.insert(
            "fr".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "le", "la", "les", "un", "une", "est", "sont", "que", "de", "et", "dans",
                    "pour", "avec", "comme", "mais", "si", "ce", "cette", "je", "vous",
                ],
                unique_chars: vec!['é', 'è', 'ê', 'à', 'ù', 'ç', 'œ', 'î', 'ô'],
                script: "Latin",
            },
        );

        // German
        patterns.insert(
            "de".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "der", "die", "das", "ist", "sind", "ein", "eine", "und", "in", "zu", "mit",
                    "für", "auf", "von", "nicht", "es", "ich", "sie", "wir", "haben",
                ],
                unique_chars: vec!['ä', 'ö', 'ü', 'ß'],
                script: "Latin",
            },
        );

        // Italian
        patterns.insert(
            "it".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "il", "la", "lo", "le", "gli", "un", "una", "è", "sono", "che", "di", "in",
                    "per", "con", "come", "ma", "se", "ci", "questo", "questa",
                ],
                unique_chars: vec!['à', 'è', 'é', 'ì', 'ò', 'ù'],
                script: "Latin",
            },
        );

        // Portuguese
        patterns.insert(
            "pt".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "o", "a", "os", "as", "um", "uma", "é", "são", "que", "de", "em", "para",
                    "com", "como", "mas", "se", "eu", "você", "não", "isso",
                ],
                unique_chars: vec!['ã', 'õ', 'ç', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô'],
                script: "Latin",
            },
        );

        // Russian
        patterns.insert(
            "ru".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "и", "в", "не", "на", "я", "что", "он", "как", "это", "она", "по", "но",
                ],
                unique_chars: vec![
                    'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п',
                    'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
                ],
                script: "Cyrillic",
            },
        );

        // Chinese (simplified)
        patterns.insert(
            "zh".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "的", "是", "在", "有", "和", "与", "了", "不", "我", "他", "这", "为",
                ],
                unique_chars: vec![],
                script: "Han",
            },
        );

        // Japanese
        patterns.insert(
            "ja".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "の", "は", "が", "を", "に", "で", "と", "も", "た", "です", "ます",
                ],
                unique_chars: vec!['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ'],
                script: "Japanese",
            },
        );

        // Korean
        patterns.insert(
            "ko".to_string(),
            LanguagePatterns {
                common_words: vec![
                    "이",
                    "가",
                    "은",
                    "는",
                    "을",
                    "를",
                    "의",
                    "에",
                    "로",
                    "와",
                    "과",
                    "입니다",
                ],
                unique_chars: vec!['가', '나', '다', '라', '마', '바', '사', '아', '자', '차'],
                script: "Hangul",
            },
        );

        Self { patterns }
    }

    /// Detect the language of a text
    pub fn detect(&self, text: &str) -> DetectedLanguage {
        if text.trim().is_empty() {
            return DetectedLanguage {
                code: "en".to_string(),
                name: "English".to_string(),
                confidence: 0.0,
                script: Some("Latin".to_string()),
            };
        }

        let text_lower = text.to_lowercase();
        let mut scores: Vec<(String, f32)> = Vec::new();

        for (code, patterns) in &self.patterns {
            let score = self.calculate_score(&text_lower, patterns);
            scores.push((code.clone(), score));
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_code, best_score) = scores
            .first()
            .cloned()
            .unwrap_or_else(|| ("en".to_string(), 0.5));

        let (name, script) = self.get_language_info(&best_code);

        DetectedLanguage {
            code: best_code,
            name: name.to_string(),
            confidence: best_score.min(1.0),
            script: Some(script.to_string()),
        }
    }

    fn calculate_score(&self, text: &str, patterns: &LanguagePatterns) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len().max(1);

        // Word matching
        let word_matches: usize = words
            .iter()
            .filter(|w| patterns.common_words.contains(w))
            .count();

        let word_score = word_matches as f32 / total_words.min(20) as f32;

        // Character matching
        let char_matches: usize = text
            .chars()
            .filter(|c| patterns.unique_chars.contains(c))
            .count();

        let char_score = if patterns.unique_chars.is_empty() {
            0.0
        } else {
            (char_matches as f32 / text.len().max(1) as f32 * 10.0).min(1.0)
        };

        // Script detection
        let script_score = self.detect_script(text, patterns.script);

        word_score * 0.5 + char_score * 0.3 + script_score * 0.2
    }

    fn detect_script(&self, text: &str, expected_script: &str) -> f32 {
        let mut latin_count = 0;
        let mut cyrillic_count = 0;
        let mut han_count = 0;
        let mut hiragana_count = 0;
        let mut hangul_count = 0;
        let mut total = 0;

        for c in text.chars() {
            if !c.is_alphabetic() {
                continue;
            }
            total += 1;

            match c {
                'a'..='z' | 'A'..='Z' | 'à'..='ÿ' | 'À'..='ß' => latin_count += 1,
                '\u{0400}'..='\u{04FF}' => cyrillic_count += 1,
                '\u{4E00}'..='\u{9FFF}' => han_count += 1,
                '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' => hiragana_count += 1,
                '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' => hangul_count += 1,
                _ => {}
            }
        }

        if total == 0 {
            return 0.5;
        }

        let detected_script = if latin_count > cyrillic_count && latin_count > han_count {
            "Latin"
        } else if cyrillic_count > latin_count && cyrillic_count > han_count {
            "Cyrillic"
        } else if han_count > 0 && hiragana_count > 0 {
            "Japanese"
        } else if han_count > latin_count && han_count > cyrillic_count {
            "Han"
        } else if hangul_count > latin_count {
            "Hangul"
        } else {
            "Latin"
        };

        if detected_script == expected_script {
            1.0
        } else {
            0.3
        }
    }

    fn get_language_info(&self, code: &str) -> (&'static str, &'static str) {
        match code {
            "en" => ("English", "Latin"),
            "es" => ("Spanish", "Latin"),
            "fr" => ("French", "Latin"),
            "de" => ("German", "Latin"),
            "it" => ("Italian", "Latin"),
            "pt" => ("Portuguese", "Latin"),
            "ru" => ("Russian", "Cyrillic"),
            "zh" => ("Chinese", "Han"),
            "ja" => ("Japanese", "Japanese"),
            "ko" => ("Korean", "Hangul"),
            _ => ("Unknown", "Unknown"),
        }
    }

    /// Detect multiple languages in text (for mixed content)
    pub fn detect_multiple(&self, text: &str) -> Vec<DetectedLanguage> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut seen_languages: HashMap<String, f32> = HashMap::new();

        for sentence in sentences {
            let detected = self.detect(sentence);
            let entry = seen_languages.entry(detected.code.clone()).or_insert(0.0);
            *entry = (*entry + detected.confidence) / 2.0;
        }

        let mut results: Vec<DetectedLanguage> = seen_languages
            .iter()
            .map(|(code, confidence)| {
                let (name, script) = self.get_language_info(code);
                DetectedLanguage {
                    code: code.clone(),
                    name: name.to_string(),
                    confidence: *confidence,
                    script: Some(script.to_string()),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ============================================================================
// Language Preferences
// ============================================================================

/// User language preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePreferences {
    /// Preferred language code
    pub preferred_language: String,
    /// Fallback language
    pub fallback_language: String,
    /// Auto-detect language from messages
    pub auto_detect: bool,
    /// Match response language to query
    pub match_query_language: bool,
    /// Languages the user understands
    pub understood_languages: Vec<String>,
}

impl Default for LanguagePreferences {
    fn default() -> Self {
        Self {
            preferred_language: "en".to_string(),
            fallback_language: "en".to_string(),
            auto_detect: true,
            match_query_language: true,
            understood_languages: vec!["en".to_string()],
        }
    }
}

// ============================================================================
// Localization Strings
// ============================================================================

/// Localized string collection
#[derive(Debug, Clone, Default)]
pub struct LocalizedStrings {
    strings: HashMap<String, HashMap<String, String>>,
}

impl LocalizedStrings {
    pub fn new() -> Self {
        let mut strings = Self::default();
        strings.add_defaults();
        strings
    }

    fn add_defaults(&mut self) {
        // Common UI strings
        self.add("greeting", "en", "Hello! How can I help you?");
        self.add("greeting", "es", "¡Hola! ¿Cómo puedo ayudarte?");
        self.add("greeting", "fr", "Bonjour! Comment puis-je vous aider?");
        self.add("greeting", "de", "Hallo! Wie kann ich Ihnen helfen?");

        self.add("thinking", "en", "Thinking...");
        self.add("thinking", "es", "Pensando...");
        self.add("thinking", "fr", "Réflexion...");
        self.add("thinking", "de", "Nachdenken...");

        self.add("error", "en", "An error occurred");
        self.add("error", "es", "Ocurrió un error");
        self.add("error", "fr", "Une erreur s'est produite");
        self.add("error", "de", "Ein Fehler ist aufgetreten");

        self.add("retry", "en", "Retry");
        self.add("retry", "es", "Reintentar");
        self.add("retry", "fr", "Réessayer");
        self.add("retry", "de", "Wiederholen");

        self.add("cancel", "en", "Cancel");
        self.add("cancel", "es", "Cancelar");
        self.add("cancel", "fr", "Annuler");
        self.add("cancel", "de", "Abbrechen");

        self.add("send", "en", "Send");
        self.add("send", "es", "Enviar");
        self.add("send", "fr", "Envoyer");
        self.add("send", "de", "Senden");

        self.add("copy", "en", "Copy");
        self.add("copy", "es", "Copiar");
        self.add("copy", "fr", "Copier");
        self.add("copy", "de", "Kopieren");

        self.add("clear_chat", "en", "Clear chat");
        self.add("clear_chat", "es", "Limpiar chat");
        self.add("clear_chat", "fr", "Effacer le chat");
        self.add("clear_chat", "de", "Chat löschen");
    }

    /// Add a localized string
    pub fn add(&mut self, key: &str, lang: &str, value: &str) {
        self.strings
            .entry(key.to_string())
            .or_default()
            .insert(lang.to_string(), value.to_string());
    }

    /// Get a localized string
    pub fn get(&self, key: &str, lang: &str) -> Option<&str> {
        self.strings
            .get(key)
            .and_then(|langs| langs.get(lang))
            .map(|s| s.as_str())
    }

    /// Get a localized string with fallback
    pub fn get_or_fallback(&self, key: &str, lang: &str, fallback: &str) -> String {
        self.get(key, lang)
            .or_else(|| self.get(key, fallback))
            .or_else(|| self.get(key, "en"))
            .unwrap_or(key)
            .to_string()
    }

    /// List available languages for a key
    pub fn available_languages(&self, key: &str) -> Vec<&str> {
        self.strings
            .get(key)
            .map(|langs| langs.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }
}

// ============================================================================
// Language-Aware Prompt Builder
// ============================================================================

/// Build prompts with language context
pub struct MultilingualPromptBuilder {
    detector: LanguageDetector,
    preferences: LanguagePreferences,
}

impl MultilingualPromptBuilder {
    pub fn new(preferences: LanguagePreferences) -> Self {
        Self {
            detector: LanguageDetector::new(),
            preferences,
        }
    }

    /// Build a system prompt with language instructions
    pub fn build_system_prompt(&self, base_prompt: &str, target_language: Option<&str>) -> String {
        let lang = target_language.unwrap_or(&self.preferences.preferred_language);
        let lang_name = self.get_language_name(lang);

        format!(
            "{}\n\nIMPORTANT: Respond in {}. Match the language of the user's message when appropriate.",
            base_prompt,
            lang_name
        )
    }

    /// Detect language and add context to message
    pub fn enrich_message(&self, message: &str) -> (String, DetectedLanguage) {
        let detected = self.detector.detect(message);

        if self.preferences.match_query_language {
            let enriched = format!("[User message in {}]\n{}", detected.name, message);
            (enriched, detected)
        } else {
            (message.to_string(), detected)
        }
    }

    fn get_language_name(&self, code: &str) -> &'static str {
        match code {
            "en" => "English",
            "es" => "Spanish",
            "fr" => "French",
            "de" => "German",
            "it" => "Italian",
            "pt" => "Portuguese",
            "ru" => "Russian",
            "zh" => "Chinese",
            "ja" => "Japanese",
            "ko" => "Korean",
            _ => "English",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_detection() {
        let detector = LanguageDetector::new();
        let result = detector.detect("The quick brown fox jumps over the lazy dog");

        assert_eq!(result.code, "en");
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_spanish_detection() {
        let detector = LanguageDetector::new();
        let result = detector.detect("Hola, ¿cómo estás? Me gustaría hablar contigo.");

        assert_eq!(result.code, "es");
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_german_detection() {
        let detector = LanguageDetector::new();
        // Use more distinctive German with umlauts
        let result = detector.detect("Ich möchte ein Buch über die deutsche Geschichte lesen.");

        assert_eq!(result.code, "de");
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_localized_strings() {
        let strings = LocalizedStrings::new();

        assert_eq!(
            strings.get("greeting", "en"),
            Some("Hello! How can I help you?")
        );
        assert_eq!(
            strings.get("greeting", "es"),
            Some("¡Hola! ¿Cómo puedo ayudarte?")
        );
    }

    #[test]
    fn test_language_preferences_default() {
        let prefs = LanguagePreferences::default();
        assert_eq!(prefs.preferred_language, "en");
        assert!(prefs.auto_detect);
    }

    #[test]
    fn test_multilingual_prompt_builder() {
        let prefs = LanguagePreferences::default();
        let builder = MultilingualPromptBuilder::new(prefs);

        let prompt = builder.build_system_prompt("You are a helpful assistant.", Some("es"));
        assert!(prompt.contains("Spanish"));
    }
}
