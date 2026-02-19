//! AI-powered translation quality analysis module.
//!
//! Provides paragraph alignment, glossary checking, issue detection,
//! and LLM-based semantic comparison prompt generation for evaluating
//! translation quality between source and target texts.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in a translation glossary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryEntry {
    /// The source language term.
    pub source: String,
    /// The expected target language translation.
    pub target: String,
    /// Optional context describing when this translation applies.
    pub context: Option<String>,
    /// Whether matching should be case-sensitive.
    pub case_sensitive: bool,
}

/// A collection of glossary entries with optional language metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Glossary {
    /// The list of glossary entries.
    pub entries: Vec<GlossaryEntry>,
    /// The source language code (e.g. "en").
    pub source_language: Option<String>,
    /// The target language code (e.g. "es").
    pub target_language: Option<String>,
}

impl Glossary {
    /// Create a new empty glossary.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            source_language: None,
            target_language: None,
        }
    }

    /// Add a simple source-target pair (case-insensitive, no context).
    pub fn add(&mut self, source: impl Into<String>, target: impl Into<String>) {
        self.entries.push(GlossaryEntry {
            source: source.into(),
            target: target.into(),
            context: None,
            case_sensitive: false,
        });
    }

    /// Add a source-target pair with context.
    pub fn add_with_context(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        context: impl Into<String>,
    ) {
        self.entries.push(GlossaryEntry {
            source: source.into(),
            target: target.into(),
            context: Some(context.into()),
            case_sensitive: false,
        });
    }

    /// Look up a source term in the glossary, returning the first matching entry.
    pub fn lookup(&self, source_term: &str) -> Option<&GlossaryEntry> {
        self.entries.iter().find(|entry| {
            if entry.case_sensitive {
                entry.source == source_term
            } else {
                entry.source.to_lowercase() == source_term.to_lowercase()
            }
        })
    }

    /// Deserialize a glossary from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let glossary: Glossary = serde_json::from_str(json)?;
        Ok(glossary)
    }

    /// Serialize the glossary to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

impl Default for Glossary {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue Types
// ─────────────────────────────────────────────────────────────────────────────

/// Categories of translation issues that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TranslationIssueType {
    /// Content present in source but missing in target.
    MissingContent,
    /// Content present in target but not in source.
    AddedContent,
    /// Glossary term translated inconsistently.
    InconsistentTerminology,
    /// Numerical values differ between source and target.
    NumberMismatch,
    /// Formality or register level mismatch.
    RegisterMismatch,
    /// Segment left untranslated.
    Untranslated,
    /// Formatting (punctuation, whitespace, tags) differs.
    FormattingDifference,
    /// Potential semantic error in translation.
    PotentialMistranslation,
}

/// A single detected translation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationIssue {
    /// The type/category of this issue.
    pub issue_type: TranslationIssueType,
    /// Severity score from 0.0 (minor) to 1.0 (critical).
    pub severity: f32,
    /// Human-readable description of the issue.
    pub description: String,
    /// The relevant source text segment, if applicable.
    pub source_segment: Option<String>,
    /// The relevant target text segment, if applicable.
    pub target_segment: Option<String>,
    /// Index of the aligned segment where this issue was found.
    pub segment_index: usize,
    /// Optional suggested fix.
    pub suggestion: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Alignment and Results
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of aligned source and target text segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedSegment {
    /// The source text paragraph.
    pub source: String,
    /// The aligned target text paragraph.
    pub target: String,
    /// Confidence score (0.0 to 1.0) for this alignment.
    pub confidence: f32,
    /// Position index of this segment.
    pub index: usize,
}

/// Statistics about the translation pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationStats {
    /// Word count in the source text.
    pub source_words: usize,
    /// Word count in the target text.
    pub target_words: usize,
    /// Ratio of target words to source words.
    pub word_ratio: f64,
    /// Number of glossary terms found in source text.
    pub glossary_terms_found: usize,
    /// Number of glossary terms correctly translated.
    pub glossary_terms_correct: usize,
    /// Count of numbers found in source text.
    pub source_numbers: usize,
    /// Count of numbers found in target text.
    pub target_numbers: usize,
}

/// Complete result of a translation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationAnalysisResult {
    /// Aligned source-target segment pairs.
    pub alignments: Vec<AlignedSegment>,
    /// All detected issues.
    pub issues: Vec<TranslationIssue>,
    /// Overall quality score (0.0 to 1.0).
    pub quality_score: f32,
    /// Glossary adherence score (0.0 to 1.0).
    pub glossary_score: f32,
    /// Number of segments analyzed.
    pub segments_analyzed: usize,
    /// Detailed statistics.
    pub stats: TranslationStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the translation analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationAnalysisConfig {
    /// Glossary to use for terminology checks.
    pub glossary: Glossary,
    /// Minimum severity threshold for reporting issues.
    pub min_severity: f32,
    /// Whether to check for number consistency.
    pub check_numbers: bool,
    /// Whether to check glossary term consistency.
    pub check_glossary: bool,
    /// Whether to check translation completeness.
    pub check_completeness: bool,
    /// Whether to check register/formality consistency.
    pub check_register: bool,
    /// Whether to enable LLM-based analysis prompt generation.
    pub enable_llm_analysis: bool,
    /// Maximum number of segments to analyze (0 = unlimited).
    pub max_segments: usize,
}

impl Default for TranslationAnalysisConfig {
    fn default() -> Self {
        Self {
            glossary: Glossary::new(),
            min_severity: 0.3,
            check_numbers: true,
            check_glossary: true,
            check_completeness: true,
            check_register: false,
            enable_llm_analysis: false,
            max_segments: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLM Comparison Prompts
// ─────────────────────────────────────────────────────────────────────────────

/// A prompt to send to an LLM for semantic translation comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonPrompt {
    /// The system message describing the LLM's role.
    pub system: String,
    /// The user message with source and target for comparison.
    pub user: String,
    /// Index of the segment being compared.
    pub segment_index: usize,
}

/// Parsed response from an LLM comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResponse {
    /// Quality score assigned by the LLM (0.0 to 1.0).
    pub score: f32,
    /// Issues identified by the LLM.
    pub issues: Vec<TranslationIssue>,
    /// Free-text explanation from the LLM.
    pub explanation: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all number-like tokens (integers, decimals, percentages) from text.
fn extract_numbers(text: &str) -> Vec<String> {
    let re = Regex::new(r"\d+(?:\.\d+)?%?").expect("valid regex");
    re.find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Calculate an alignment confidence score based on length ratio similarity.
/// Returns a value between 0.0 and 1.0.
fn alignment_score(source: &str, target: &str) -> f32 {
    let src_len = source.len().max(1) as f64;
    let tgt_len = target.len().max(1) as f64;
    let ratio = if src_len > tgt_len {
        tgt_len / src_len
    } else {
        src_len / tgt_len
    };
    // A ratio near 1.0 is ideal; we also accept some variance for
    // languages with different word lengths. Apply a gentle curve.
    let score = (ratio * 1.2).min(1.0);
    score as f32
}

/// Split text into paragraphs (on double newlines, or single newlines if no doubles exist).
fn split_paragraphs(text: &str) -> Vec<String> {
    // First try splitting on double newlines
    let paragraphs: Vec<String> = text
        .split("\n\n")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if paragraphs.len() > 1 {
        return paragraphs;
    }

    // Fall back to single newlines
    text.split('\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Check whether `text` contains the given `term`, respecting case sensitivity.
fn contains_term(text: &str, term: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        text.contains(term)
    } else {
        text.to_lowercase().contains(&term.to_lowercase())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analyzer
// ─────────────────────────────────────────────────────────────────────────────

/// The main translation quality analyzer.
#[derive(Debug, Clone)]
pub struct TranslationAnalyzer {
    /// Configuration controlling which checks to run.
    pub config: TranslationAnalysisConfig,
}

impl TranslationAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: TranslationAnalysisConfig) -> Self {
        Self { config }
    }

    /// Perform a full translation analysis between source and target texts.
    ///
    /// Steps:
    /// 1. Align paragraphs between source and target.
    /// 2. Run configured checks (numbers, glossary, completeness).
    /// 3. Calculate quality and glossary scores.
    pub fn analyze(&self, source: &str, target: &str) -> TranslationAnalysisResult {
        let alignments = self.align_paragraphs(source, target);
        let mut issues: Vec<TranslationIssue> = Vec::new();

        if self.config.check_numbers {
            issues.extend(self.check_numbers(&alignments));
        }
        if self.config.check_glossary {
            issues.extend(self.check_glossary(&alignments));
        }
        if self.config.check_completeness {
            issues.extend(self.check_completeness(&alignments));
        }

        // Filter by minimum severity
        issues.retain(|issue| issue.severity >= self.config.min_severity);

        // Calculate statistics
        let source_words = source.split_whitespace().count();
        let target_words = target.split_whitespace().count();
        let word_ratio = if source_words > 0 {
            target_words as f64 / source_words as f64
        } else {
            0.0
        };

        let all_source_numbers = extract_numbers(source);
        let all_target_numbers = extract_numbers(target);

        let glossary_terms_found = self.count_glossary_terms_found(&alignments);
        let glossary_terms_correct = self.count_glossary_terms_correct(&alignments);

        let stats = TranslationStats {
            source_words,
            target_words,
            word_ratio,
            glossary_terms_found,
            glossary_terms_correct,
            source_numbers: all_source_numbers.len(),
            target_numbers: all_target_numbers.len(),
        };

        // Calculate quality score: start at 1.0, subtract based on issues
        let quality_score = self.calculate_quality_score(&issues, alignments.len());

        // Calculate glossary score
        let glossary_score = if glossary_terms_found > 0 {
            glossary_terms_correct as f32 / glossary_terms_found as f32
        } else {
            1.0
        };

        TranslationAnalysisResult {
            segments_analyzed: alignments.len(),
            alignments,
            issues,
            quality_score,
            glossary_score,
            stats,
        }
    }

    /// Align source and target paragraphs by position and length ratio.
    pub fn align_paragraphs(&self, source: &str, target: &str) -> Vec<AlignedSegment> {
        let source_paras = split_paragraphs(source);
        let target_paras = split_paragraphs(target);

        let max_len = source_paras.len().max(target_paras.len());
        let limit = if self.config.max_segments > 0 {
            max_len.min(self.config.max_segments)
        } else {
            max_len
        };

        let mut alignments = Vec::new();

        for i in 0..limit {
            let src = source_paras.get(i).cloned().unwrap_or_default();
            let tgt = target_paras.get(i).cloned().unwrap_or_default();
            let confidence = if src.is_empty() || tgt.is_empty() {
                0.0
            } else {
                alignment_score(&src, &tgt)
            };

            alignments.push(AlignedSegment {
                source: src,
                target: tgt,
                confidence,
                index: i,
            });
        }

        alignments
    }

    /// Check glossary term consistency across aligned segments.
    ///
    /// For each aligned segment, finds glossary source terms in the source text
    /// and checks whether the expected target term appears in the target text.
    pub fn check_glossary(&self, alignments: &[AlignedSegment]) -> Vec<TranslationIssue> {
        let mut issues = Vec::new();

        for segment in alignments {
            for entry in &self.config.glossary.entries {
                if contains_term(&segment.source, &entry.source, entry.case_sensitive) {
                    if !contains_term(&segment.target, &entry.target, entry.case_sensitive) {
                        issues.push(TranslationIssue {
                            issue_type: TranslationIssueType::InconsistentTerminology,
                            severity: 0.7,
                            description: format!(
                                "Glossary term '{}' should be translated as '{}' but was not found in target",
                                entry.source, entry.target
                            ),
                            source_segment: Some(segment.source.clone()),
                            target_segment: Some(segment.target.clone()),
                            segment_index: segment.index,
                            suggestion: Some(format!(
                                "Use '{}' as the translation for '{}'",
                                entry.target, entry.source
                            )),
                        });
                    }
                }
            }
        }

        issues
    }

    /// Check for number consistency between source and target segments.
    ///
    /// Extracts all number patterns and compares the sorted sets.
    pub fn check_numbers(&self, alignments: &[AlignedSegment]) -> Vec<TranslationIssue> {
        let mut issues = Vec::new();

        for segment in alignments {
            if segment.source.is_empty() || segment.target.is_empty() {
                continue;
            }

            let mut src_numbers = extract_numbers(&segment.source);
            let mut tgt_numbers = extract_numbers(&segment.target);

            src_numbers.sort();
            tgt_numbers.sort();

            if src_numbers != tgt_numbers {
                let missing: Vec<_> = src_numbers
                    .iter()
                    .filter(|n| !tgt_numbers.contains(n))
                    .collect();
                let added: Vec<_> = tgt_numbers
                    .iter()
                    .filter(|n| !src_numbers.contains(n))
                    .collect();

                let mut desc = String::from("Number mismatch between source and target.");
                if !missing.is_empty() {
                    desc.push_str(&format!(" Missing in target: {:?}.", missing));
                }
                if !added.is_empty() {
                    desc.push_str(&format!(" Added in target: {:?}.", added));
                }

                issues.push(TranslationIssue {
                    issue_type: TranslationIssueType::NumberMismatch,
                    severity: 0.8,
                    description: desc,
                    source_segment: Some(segment.source.clone()),
                    target_segment: Some(segment.target.clone()),
                    segment_index: segment.index,
                    suggestion: Some("Ensure all numbers from the source appear in the target translation.".to_string()),
                });
            }
        }

        issues
    }

    /// Check translation completeness by comparing segment word counts.
    ///
    /// Flags segments where the target is much shorter (< 0.5 ratio) or
    /// much longer (> 2.0 ratio) than the source.
    pub fn check_completeness(&self, alignments: &[AlignedSegment]) -> Vec<TranslationIssue> {
        let mut issues = Vec::new();

        for segment in alignments {
            let src_words = segment.source.split_whitespace().count();
            let tgt_words = segment.target.split_whitespace().count();

            // Handle empty segments
            if segment.source.is_empty() && !segment.target.is_empty() {
                issues.push(TranslationIssue {
                    issue_type: TranslationIssueType::AddedContent,
                    severity: 0.6,
                    description: "Target segment has content with no corresponding source.".to_string(),
                    source_segment: None,
                    target_segment: Some(segment.target.clone()),
                    segment_index: segment.index,
                    suggestion: Some("Verify this content should be present.".to_string()),
                });
                continue;
            }

            if !segment.source.is_empty() && segment.target.is_empty() {
                issues.push(TranslationIssue {
                    issue_type: TranslationIssueType::MissingContent,
                    severity: 0.9,
                    description: "Source segment has no corresponding translation.".to_string(),
                    source_segment: Some(segment.source.clone()),
                    target_segment: None,
                    segment_index: segment.index,
                    suggestion: Some("Translate this segment.".to_string()),
                });
                continue;
            }

            if src_words == 0 {
                continue;
            }

            let ratio = tgt_words as f64 / src_words as f64;

            if ratio < 0.5 {
                issues.push(TranslationIssue {
                    issue_type: TranslationIssueType::MissingContent,
                    severity: 0.6,
                    description: format!(
                        "Target segment is significantly shorter than source (ratio: {:.2}). Possible incomplete translation.",
                        ratio
                    ),
                    source_segment: Some(segment.source.clone()),
                    target_segment: Some(segment.target.clone()),
                    segment_index: segment.index,
                    suggestion: Some("Review for missing content.".to_string()),
                });
            } else if ratio > 2.0 {
                issues.push(TranslationIssue {
                    issue_type: TranslationIssueType::AddedContent,
                    severity: 0.4,
                    description: format!(
                        "Target segment is significantly longer than source (ratio: {:.2}). Possible added content.",
                        ratio
                    ),
                    source_segment: Some(segment.source.clone()),
                    target_segment: Some(segment.target.clone()),
                    segment_index: segment.index,
                    suggestion: Some("Review for extraneous content.".to_string()),
                });
            }
        }

        issues
    }

    /// Generate LLM comparison prompts for semantic analysis of aligned segments.
    pub fn generate_comparison_prompts(&self, alignments: &[AlignedSegment]) -> Vec<ComparisonPrompt> {
        let system = "You are a translation quality expert. Analyze the following source and target \
            text segments for translation accuracy. Identify any semantic differences, omissions, \
            additions, or mistranslations. Respond in JSON format with fields: \
            \"score\" (0.0-1.0), \"issues\" (array of {\"type\", \"description\", \"severity\"}), \
            \"explanation\" (string).".to_string();

        alignments
            .iter()
            .filter(|seg| !seg.source.is_empty() && !seg.target.is_empty())
            .map(|seg| {
                let user = format!(
                    "Compare the following translation pair:\n\n\
                     Source: {}\n\n\
                     Target: {}\n\n\
                     Identify any translation issues, rate the quality, and explain your assessment.",
                    seg.source, seg.target
                );

                ComparisonPrompt {
                    system: system.clone(),
                    user,
                    segment_index: seg.index,
                }
            })
            .collect()
    }

    /// Parse a JSON response from an LLM comparison into a structured result.
    pub fn parse_comparison_response(&self, response: &str, segment_index: usize) -> ComparisonResponse {
        // Attempt to parse as JSON
        #[derive(Deserialize)]
        struct RawResponse {
            score: Option<f32>,
            issues: Option<Vec<RawIssue>>,
            explanation: Option<String>,
        }

        #[derive(Deserialize)]
        struct RawIssue {
            #[serde(rename = "type")]
            issue_type: Option<String>,
            description: Option<String>,
            severity: Option<f32>,
        }

        let parsed: Option<RawResponse> = serde_json::from_str(response).ok();

        match parsed {
            Some(raw) => {
                let issues = raw
                    .issues
                    .unwrap_or_default()
                    .into_iter()
                    .map(|ri| {
                        let issue_type = match ri.issue_type.as_deref() {
                            Some("MissingContent") => TranslationIssueType::MissingContent,
                            Some("AddedContent") => TranslationIssueType::AddedContent,
                            Some("InconsistentTerminology") => TranslationIssueType::InconsistentTerminology,
                            Some("NumberMismatch") => TranslationIssueType::NumberMismatch,
                            Some("RegisterMismatch") => TranslationIssueType::RegisterMismatch,
                            Some("Untranslated") => TranslationIssueType::Untranslated,
                            Some("FormattingDifference") => TranslationIssueType::FormattingDifference,
                            _ => TranslationIssueType::PotentialMistranslation,
                        };

                        TranslationIssue {
                            issue_type,
                            severity: ri.severity.unwrap_or(0.5),
                            description: ri.description.unwrap_or_default(),
                            source_segment: None,
                            target_segment: None,
                            segment_index,
                            suggestion: None,
                        }
                    })
                    .collect();

                ComparisonResponse {
                    score: raw.score.unwrap_or(0.5),
                    issues,
                    explanation: raw.explanation.unwrap_or_else(|| "No explanation provided.".to_string()),
                }
            }
            None => ComparisonResponse {
                score: 0.5,
                issues: Vec::new(),
                explanation: format!("Failed to parse LLM response as JSON: {}", response),
            },
        }
    }

    /// Heuristic language detection based on Unicode script block frequency analysis.
    ///
    /// Returns a language code hint (e.g., "zh", "ja", "ko", "ru", "ar", "en", "es", etc.)
    /// or None if detection is inconclusive.
    pub fn detect_language(&self, text: &str) -> Option<String> {
        if text.is_empty() {
            return None;
        }

        let mut counts: HashMap<&str, usize> = HashMap::new();
        let total_chars = text.chars().filter(|c| !c.is_whitespace()).count();

        if total_chars == 0 {
            return None;
        }

        for ch in text.chars() {
            if ch.is_whitespace() {
                continue;
            }
            let script = classify_char(ch);
            *counts.entry(script).or_insert(0) += 1;
        }

        // Determine dominant script
        let dominant = counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&script, &count)| (script, count));

        match dominant {
            Some(("han", count)) if count > total_chars / 3 => {
                // Could be Chinese or Japanese with Kanji
                if counts.get("hiragana").copied().unwrap_or(0) > 0
                    || counts.get("katakana").copied().unwrap_or(0) > 0
                {
                    Some("ja".to_string())
                } else {
                    Some("zh".to_string())
                }
            }
            Some(("hiragana", _)) | Some(("katakana", _)) => Some("ja".to_string()),
            Some(("hangul", _)) => Some("ko".to_string()),
            Some(("cyrillic", _)) => Some("ru".to_string()),
            Some(("arabic", _)) => Some("ar".to_string()),
            Some(("devanagari", _)) => Some("hi".to_string()),
            Some(("thai", _)) => Some("th".to_string()),
            Some(("latin", count)) if count > total_chars / 2 => {
                // Distinguish among Latin-script languages by accent/character frequency
                self.detect_latin_language(text)
            }
            _ => None,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private Methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Attempt to distinguish Latin-script languages by character frequency heuristics.
    fn detect_latin_language(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();

        // Spanish indicators
        let _spanish_chars = ['n', 'a', 'e', 'i', 'o', 'u'];
        let spanish_markers = ["que", "los", "las", "del", "por", "como", "una", "esta"];
        let spanish_score: usize = spanish_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // French indicators
        let french_markers = ["les", "des", "une", "est", "pas", "que", "dans", "pour"];
        let french_score: usize = french_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // German indicators
        let german_markers = ["der", "die", "das", "und", "ist", "ein", "nicht", "auf"];
        let german_score: usize = german_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // Portuguese indicators
        let portuguese_markers = ["que", "nao", "uma", "para", "com", "mais", "como", "dos"];
        let portuguese_score: usize = portuguese_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // Italian indicators
        let italian_markers = ["che", "non", "una", "con", "per", "sono", "della", "questo"];
        let italian_score: usize = italian_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // English indicators
        let english_markers = ["the", "and", "that", "have", "for", "not", "with", "you"];
        let english_score: usize = english_markers
            .iter()
            .filter(|m| lower.contains(*m))
            .count();

        // Check for accented characters common in specific languages
        let has_tilde_n = lower.contains('\u{00f1}'); // n with tilde (Spanish)
        let has_cedilla = lower.contains('\u{00e7}'); // c with cedilla (French/Portuguese)
        let has_umlaut = lower.contains('\u{00fc}') || lower.contains('\u{00f6}') || lower.contains('\u{00e4}');
        let has_circumflex = lower.contains('\u{00ea}') || lower.contains('\u{00ee}') || lower.contains('\u{00f4}');

        let mut scores: Vec<(&str, usize)> = vec![
            ("es", spanish_score + if has_tilde_n { 3 } else { 0 }),
            ("fr", french_score + if has_cedilla || has_circumflex { 2 } else { 0 }),
            ("de", german_score + if has_umlaut { 3 } else { 0 }),
            ("pt", portuguese_score + if has_cedilla { 1 } else { 0 }),
            ("it", italian_score),
            ("en", english_score),
        ];

        scores.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some((lang, score)) = scores.first() {
            if *score >= 2 {
                return Some(lang.to_string());
            }
        }

        // Default to English for Latin script if inconclusive
        Some("en".to_string())
    }

    /// Calculate an overall quality score based on detected issues.
    fn calculate_quality_score(&self, issues: &[TranslationIssue], segment_count: usize) -> f32 {
        if segment_count == 0 {
            return 1.0;
        }

        let total_penalty: f32 = issues.iter().map(|i| i.severity * 0.1).sum();
        let score = (1.0 - total_penalty).max(0.0).min(1.0);
        score
    }

    /// Count how many glossary source terms appear in the aligned source segments.
    fn count_glossary_terms_found(&self, alignments: &[AlignedSegment]) -> usize {
        let mut count = 0;
        for segment in alignments {
            for entry in &self.config.glossary.entries {
                if contains_term(&segment.source, &entry.source, entry.case_sensitive) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Count how many glossary terms are correctly translated in the target.
    fn count_glossary_terms_correct(&self, alignments: &[AlignedSegment]) -> usize {
        let mut count = 0;
        for segment in alignments {
            for entry in &self.config.glossary.entries {
                if contains_term(&segment.source, &entry.source, entry.case_sensitive)
                    && contains_term(&segment.target, &entry.target, entry.case_sensitive)
                {
                    count += 1;
                }
            }
        }
        count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unicode Script Classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classify a character into its Unicode script block category.
fn classify_char(ch: char) -> &'static str {
    let code = ch as u32;
    match code {
        // CJK Unified Ideographs and extensions
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0xF900..=0xFAFF => "han",
        // Hiragana
        0x3040..=0x309F => "hiragana",
        // Katakana
        0x30A0..=0x30FF | 0x31F0..=0x31FF => "katakana",
        // Hangul
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => "hangul",
        // Cyrillic
        0x0400..=0x04FF | 0x0500..=0x052F => "cyrillic",
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => "arabic",
        // Devanagari
        0x0900..=0x097F => "devanagari",
        // Thai
        0x0E00..=0x0E7F => "thai",
        // Basic Latin and Latin Extended
        0x0041..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => "latin",
        // Common punctuation, digits
        0x0030..=0x0039 => "digit",
        _ => "other",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glossary_lookup_and_serialization() {
        let mut glossary = Glossary::new();
        glossary.add("ship", "nave");
        glossary.add_with_context("quantum drive", "propulsor cuantico", "spaceship component");

        // Lookup
        let entry = glossary.lookup("ship").unwrap();
        assert_eq!(entry.target, "nave");

        let entry2 = glossary.lookup("Quantum Drive").unwrap(); // case insensitive
        assert_eq!(entry2.target, "propulsor cuantico");

        assert!(glossary.lookup("nonexistent").is_none());

        // Serialization round-trip
        let json = glossary.to_json();
        let restored = Glossary::from_json(&json).unwrap();
        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.entries[0].source, "ship");
    }

    #[test]
    fn test_number_check_detects_mismatch() {
        let mut config = TranslationAnalysisConfig::default();
        config.check_numbers = true;
        config.check_glossary = false;
        config.check_completeness = false;

        let analyzer = TranslationAnalyzer::new(config);

        let source = "The ship has 3 engines and travels at 0.2c speed with 99.5% efficiency.";
        let target = "La nave tiene 3 motores y viaja a velocidad 0.3c con 99.5% de eficiencia.";

        let result = analyzer.analyze(source, target);

        // Should detect a number mismatch (0.2 vs 0.3)
        let number_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.issue_type == TranslationIssueType::NumberMismatch)
            .collect();
        assert!(
            !number_issues.is_empty(),
            "Should detect number mismatch between 0.2 and 0.3"
        );
    }

    #[test]
    fn test_completeness_check_flags_short_translation() {
        let mut config = TranslationAnalysisConfig::default();
        config.check_numbers = false;
        config.check_glossary = false;
        config.check_completeness = true;

        let analyzer = TranslationAnalyzer::new(config);

        let source = "The quantum drive is an advanced propulsion system that enables faster-than-light travel across star systems in the universe.";
        let target = "Propulsor cuantico.";

        let result = analyzer.analyze(source, target);

        let missing_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.issue_type == TranslationIssueType::MissingContent)
            .collect();
        assert!(
            !missing_issues.is_empty(),
            "Should flag significantly shorter translation as missing content"
        );
    }

    #[test]
    fn test_glossary_check_detects_inconsistency() {
        let mut config = TranslationAnalysisConfig::default();
        config.check_numbers = false;
        config.check_completeness = false;
        config.check_glossary = true;
        config.glossary.add("quantum drive", "propulsor cuantico");
        config.glossary.add("shield", "escudo");

        let analyzer = TranslationAnalyzer::new(config);

        let source = "Activate the quantum drive and raise the shield.";
        let target = "Activar el motor cuantico y levantar la barrera.";

        let result = analyzer.analyze(source, target);

        let terminology_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.issue_type == TranslationIssueType::InconsistentTerminology)
            .collect();

        // Should detect both "quantum drive" and "shield" inconsistencies
        assert!(
            terminology_issues.len() >= 2,
            "Should detect glossary inconsistencies for 'quantum drive' and 'shield', got {}",
            terminology_issues.len()
        );
    }

    #[test]
    fn test_language_detection_heuristic() {
        let analyzer = TranslationAnalyzer::new(TranslationAnalysisConfig::default());

        // Chinese
        let zh = analyzer.detect_language("这是一个测试句子");
        assert_eq!(zh, Some("zh".to_string()));

        // Japanese (mix of kanji and hiragana)
        let ja = analyzer.detect_language("これはテストです");
        assert_eq!(ja, Some("ja".to_string()));

        // Korean
        let ko = analyzer.detect_language("이것은 테스트입니다");
        assert_eq!(ko, Some("ko".to_string()));

        // Russian
        let ru = analyzer.detect_language("Это тестовое предложение");
        assert_eq!(ru, Some("ru".to_string()));

        // Arabic
        let ar = analyzer.detect_language("هذه جملة اختبار");
        assert_eq!(ar, Some("ar".to_string()));
    }

    #[test]
    fn test_paragraph_alignment_and_confidence() {
        let analyzer = TranslationAnalyzer::new(TranslationAnalysisConfig::default());

        let source = "First paragraph here.\n\nSecond paragraph with more words.\n\nThird.";
        let target = "Primer parrafo aqui.\n\nSegundo parrafo con mas palabras.\n\nTercero.";

        let alignments = analyzer.align_paragraphs(source, target);

        assert_eq!(alignments.len(), 3);
        assert_eq!(alignments[0].index, 0);
        assert_eq!(alignments[1].index, 1);
        assert_eq!(alignments[2].index, 2);

        // All confidence scores should be positive since lengths are similar
        for seg in &alignments {
            assert!(
                seg.confidence > 0.5,
                "Expected confidence > 0.5 for similar-length segments, got {}",
                seg.confidence
            );
        }
    }

    #[test]
    fn test_comparison_prompt_generation_and_response_parsing() {
        let analyzer = TranslationAnalyzer::new(TranslationAnalysisConfig::default());

        let alignments = vec![
            AlignedSegment {
                source: "The ship is ready.".to_string(),
                target: "La nave esta lista.".to_string(),
                confidence: 0.9,
                index: 0,
            },
            AlignedSegment {
                source: "".to_string(),
                target: "Extra content.".to_string(),
                confidence: 0.0,
                index: 1,
            },
        ];

        let prompts = analyzer.generate_comparison_prompts(&alignments);

        // Only non-empty source+target segments get prompts
        assert_eq!(prompts.len(), 1);
        assert!(prompts[0].system.contains("translation quality expert"));
        assert!(prompts[0].user.contains("The ship is ready."));
        assert!(prompts[0].user.contains("La nave esta lista."));

        // Test response parsing
        let json_response = r#"{"score": 0.85, "issues": [{"type": "FormattingDifference", "description": "Missing accent", "severity": 0.3}], "explanation": "Good translation with minor accent issue."}"#;
        let parsed = analyzer.parse_comparison_response(json_response, 0);

        assert_eq!(parsed.score, 0.85);
        assert_eq!(parsed.issues.len(), 1);
        assert_eq!(parsed.issues[0].issue_type, TranslationIssueType::FormattingDifference);
        assert!(parsed.explanation.contains("Good translation"));

        // Test invalid JSON fallback
        let bad_response = "This is not valid JSON at all.";
        let fallback = analyzer.parse_comparison_response(bad_response, 5);
        assert_eq!(fallback.score, 0.5);
        assert!(fallback.issues.is_empty());
        assert!(fallback.explanation.contains("Failed to parse"));
    }
}
