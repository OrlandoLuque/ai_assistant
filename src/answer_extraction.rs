//! Answer extraction
//!
//! Extract direct answers from long text responses.

use std::collections::HashMap;

/// Extraction configuration
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Maximum length of extracted answer
    pub max_answer_length: usize,
    /// Whether to include context
    pub include_context: bool,
    /// Context sentences before/after
    pub context_sentences: usize,
    /// Confidence threshold
    pub min_confidence: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_answer_length: 500,
            include_context: true,
            context_sentences: 1,
            min_confidence: 0.5,
        }
    }
}

/// Extracted answer
#[derive(Debug, Clone)]
pub struct ExtractedAnswer {
    /// The direct answer
    pub answer: String,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Position in source text
    pub position: usize,
    /// Context around the answer
    pub context: Option<String>,
    /// Type of answer
    pub answer_type: AnswerType,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Type of answer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerType {
    /// Direct factual answer
    Factual,
    /// Yes/No answer
    Boolean,
    /// Numeric answer
    Numeric,
    /// List of items
    List,
    /// Explanation
    Explanation,
    /// Definition
    Definition,
    /// Uncertain/hedged answer
    Uncertain,
}

/// Answer extractor
pub struct AnswerExtractor {
    config: ExtractionConfig,
    answer_indicators: Vec<&'static str>,
    uncertainty_markers: Vec<&'static str>,
}

impl AnswerExtractor {
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            config,
            answer_indicators: vec![
                "the answer is",
                "it is",
                "they are",
                "this is",
                "yes,",
                "no,",
                "in short",
                "to summarize",
                "specifically",
                "the result is",
                "is defined as",
            ],
            uncertainty_markers: vec![
                "might",
                "could",
                "possibly",
                "perhaps",
                "maybe",
                "uncertain",
                "not sure",
                "depends on",
            ],
        }
    }

    /// Extract answer from text for a given question
    pub fn extract(&self, question: &str, text: &str) -> Option<ExtractedAnswer> {
        let question_type = self.classify_question(question);
        let sentences = self.split_sentences(text);

        match question_type {
            QuestionType::YesNo => self.extract_boolean(question, &sentences),
            QuestionType::What => self.extract_definition(question, &sentences),
            QuestionType::How => self.extract_explanation(question, &sentences),
            QuestionType::Why => self.extract_explanation(question, &sentences),
            QuestionType::When | QuestionType::Where => self.extract_factual(question, &sentences),
            QuestionType::HowMany => self.extract_numeric(question, &sentences),
            QuestionType::List => self.extract_list(question, text),
            QuestionType::Other => self.extract_best_match(question, &sentences),
        }
    }

    /// Extract multiple potential answers
    pub fn extract_all(&self, question: &str, text: &str) -> Vec<ExtractedAnswer> {
        let mut answers = Vec::new();
        let sentences = self.split_sentences(text);

        // Try different extraction methods
        if let Some(answer) = self.extract_with_indicators(&sentences) {
            answers.push(answer);
        }

        if let Some(answer) = self.extract_first_relevant(question, &sentences) {
            if !answers.iter().any(|a| a.answer == answer.answer) {
                answers.push(answer);
            }
        }

        if let Some(answer) = self.extract_from_conclusion(&sentences) {
            if !answers.iter().any(|a| a.answer == answer.answer) {
                answers.push(answer);
            }
        }

        // Sort by confidence
        answers.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        answers
    }

    fn classify_question(&self, question: &str) -> QuestionType {
        let lower = question.to_lowercase();

        if lower.starts_with("is ") || lower.starts_with("are ") ||
           lower.starts_with("do ") || lower.starts_with("does ") ||
           lower.starts_with("can ") || lower.starts_with("will ") {
            return QuestionType::YesNo;
        }

        if lower.starts_with("what") {
            if lower.contains("list") || lower.contains("types") || lower.contains("kinds") {
                return QuestionType::List;
            }
            return QuestionType::What;
        }

        if lower.starts_with("how many") || lower.starts_with("how much") {
            return QuestionType::HowMany;
        }

        if lower.starts_with("how") {
            return QuestionType::How;
        }

        if lower.starts_with("why") {
            return QuestionType::Why;
        }

        if lower.starts_with("when") {
            return QuestionType::When;
        }

        if lower.starts_with("where") {
            return QuestionType::Where;
        }

        QuestionType::Other
    }

    fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn extract_boolean(&self, _question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            // Check for explicit yes/no
            if lower.starts_with("yes") || lower.contains(" yes,") || lower.contains(" yes.") {
                return Some(ExtractedAnswer {
                    answer: "Yes".to_string(),
                    confidence: 0.9,
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Boolean,
                    evidence: vec![sentence.clone()],
                });
            }

            if lower.starts_with("no") || lower.contains(" no,") || lower.contains(" no.") {
                return Some(ExtractedAnswer {
                    answer: "No".to_string(),
                    confidence: 0.9,
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Boolean,
                    evidence: vec![sentence.clone()],
                });
            }
        }

        // Look for implicit yes/no
        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            if lower.contains("it is") || lower.contains("they are") || lower.contains("this is correct") {
                return Some(ExtractedAnswer {
                    answer: "Yes (implied)".to_string(),
                    confidence: 0.7,
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Boolean,
                    evidence: vec![sentence.clone()],
                });
            }

            if lower.contains("it is not") || lower.contains("cannot") || lower.contains("isn't") {
                return Some(ExtractedAnswer {
                    answer: "No (implied)".to_string(),
                    confidence: 0.7,
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Boolean,
                    evidence: vec![sentence.clone()],
                });
            }
        }

        None
    }

    fn extract_definition(&self, question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        // Look for "is defined as", "is a", "refers to"
        let patterns = ["is defined as", "is a ", "refers to", "means", "is the"];

        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            for pattern in patterns {
                if lower.contains(pattern) {
                    return Some(ExtractedAnswer {
                        answer: self.truncate_answer(sentence),
                        confidence: 0.8,
                        position: i,
                        context: self.get_context(sentences, i),
                        answer_type: AnswerType::Definition,
                        evidence: vec![sentence.clone()],
                    });
                }
            }
        }

        // Fall back to first relevant sentence
        self.extract_first_relevant(question, sentences)
    }

    fn extract_explanation(&self, question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        // Look for explanation indicators
        let patterns = ["because", "the reason", "this is due to", "in order to", "by"];

        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            for pattern in patterns {
                if lower.contains(pattern) {
                    return Some(ExtractedAnswer {
                        answer: self.truncate_answer(sentence),
                        confidence: 0.75,
                        position: i,
                        context: self.get_context(sentences, i),
                        answer_type: AnswerType::Explanation,
                        evidence: vec![sentence.clone()],
                    });
                }
            }
        }

        self.extract_first_relevant(question, sentences)
    }

    fn extract_factual(&self, question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        // Extract keywords from question
        let question_lower = question.to_lowercase();
        let keywords: Vec<&str> = question_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let mut best_match: Option<(usize, String, f64)> = None;

        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();
            let matches = keywords.iter().filter(|k| lower.contains(*k)).count();
            let score: f64 = matches as f64 / keywords.len().max(1) as f64;

            if score > best_match.as_ref().map(|(_, _, s)| *s).unwrap_or(0.0) {
                best_match = Some((i, sentence.clone(), score));
            }
        }

        best_match.map(|(i, sentence, score)| ExtractedAnswer {
            answer: self.truncate_answer(&sentence),
            confidence: score * 0.8,
            position: i,
            context: self.get_context(sentences, i),
            answer_type: AnswerType::Factual,
            evidence: vec![sentence],
        })
    }

    fn extract_numeric(&self, _question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        // Look for numbers
        let number_pattern = regex::Regex::new(r"\d+(?:\.\d+)?").ok()?;

        for (i, sentence) in sentences.iter().enumerate() {
            if let Some(m) = number_pattern.find(sentence) {
                return Some(ExtractedAnswer {
                    answer: m.as_str().to_string(),
                    confidence: 0.8,
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Numeric,
                    evidence: vec![sentence.clone()],
                });
            }
        }

        None
    }

    fn extract_list(&self, _question: &str, text: &str) -> Option<ExtractedAnswer> {
        let mut items = Vec::new();

        // Look for numbered or bulleted lists
        let list_pattern = regex::Regex::new(r"(?m)^[\s]*(?:\d+[.\)]|[-•*])\s*(.+)$").ok()?;

        for cap in list_pattern.captures_iter(text) {
            if let Some(item) = cap.get(1) {
                items.push(item.as_str().trim().to_string());
            }
        }

        if !items.is_empty() {
            return Some(ExtractedAnswer {
                answer: items.join("; "),
                confidence: 0.85,
                position: 0,
                context: None,
                answer_type: AnswerType::List,
                evidence: items,
            });
        }

        None
    }

    fn extract_with_indicators(&self, sentences: &[String]) -> Option<ExtractedAnswer> {
        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            for indicator in &self.answer_indicators {
                if lower.contains(indicator) {
                    // Extract the part after the indicator
                    if let Some(pos) = lower.find(indicator) {
                        let answer_part = &sentence[pos + indicator.len()..];
                        if !answer_part.trim().is_empty() {
                            return Some(ExtractedAnswer {
                                answer: self.truncate_answer(answer_part.trim()),
                                confidence: 0.85,
                                position: i,
                                context: self.get_context(sentences, i),
                                answer_type: AnswerType::Factual,
                                evidence: vec![sentence.clone()],
                            });
                        }
                    }
                }
            }
        }

        None
    }

    fn extract_first_relevant(&self, question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        let question_lower = question.to_lowercase();
        let keywords: Vec<&str> = question_lower
            .split_whitespace()
            .filter(|w| w.len() > 3 && !["what", "when", "where", "which", "how"].contains(w))
            .collect();

        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();
            let has_keyword = keywords.iter().any(|k| lower.contains(*k));

            if has_keyword && sentence.len() > 20 {
                let mut confidence: f64 = 0.6;

                // Check for uncertainty
                for marker in &self.uncertainty_markers {
                    if lower.contains(marker) {
                        confidence -= 0.1;
                    }
                }

                return Some(ExtractedAnswer {
                    answer: self.truncate_answer(sentence),
                    confidence: confidence.max(0.3),
                    position: i,
                    context: self.get_context(sentences, i),
                    answer_type: AnswerType::Factual,
                    evidence: vec![sentence.clone()],
                });
            }
        }

        None
    }

    fn extract_from_conclusion(&self, sentences: &[String]) -> Option<ExtractedAnswer> {
        let conclusion_markers = ["in conclusion", "to summarize", "in summary", "therefore", "thus"];

        for (i, sentence) in sentences.iter().enumerate() {
            let lower = sentence.to_lowercase();

            for marker in conclusion_markers {
                if lower.contains(marker) {
                    return Some(ExtractedAnswer {
                        answer: self.truncate_answer(sentence),
                        confidence: 0.75,
                        position: i,
                        context: self.get_context(sentences, i),
                        answer_type: AnswerType::Factual,
                        evidence: vec![sentence.clone()],
                    });
                }
            }
        }

        None
    }

    fn extract_best_match(&self, question: &str, sentences: &[String]) -> Option<ExtractedAnswer> {
        self.extract_first_relevant(question, sentences)
    }

    fn get_context(&self, sentences: &[String], pos: usize) -> Option<String> {
        if !self.config.include_context {
            return None;
        }

        let start = pos.saturating_sub(self.config.context_sentences);
        let end = (pos + self.config.context_sentences + 1).min(sentences.len());

        let context: Vec<_> = sentences[start..end].to_vec();
        Some(context.join(". "))
    }

    fn truncate_answer(&self, text: &str) -> String {
        if text.len() <= self.config.max_answer_length {
            text.to_string()
        } else {
            format!("{}...", &text[..self.config.max_answer_length])
        }
    }
}

impl Default for AnswerExtractor {
    fn default() -> Self {
        Self::new(ExtractionConfig::default())
    }
}

#[derive(Debug, Clone, Copy)]
enum QuestionType {
    YesNo,
    What,
    How,
    Why,
    When,
    Where,
    HowMany,
    List,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_extraction() {
        let extractor = AnswerExtractor::default();

        let text = "Yes, Python is a programming language. It was created in 1991.";
        let answer = extractor.extract("Is Python a programming language?", text);

        assert!(answer.is_some());
        assert_eq!(answer.unwrap().answer, "Yes");
    }

    #[test]
    fn test_definition_extraction() {
        let extractor = AnswerExtractor::default();

        let text = "Python is a high-level programming language. It is known for readability.";
        let answer = extractor.extract("What is Python?", text);

        assert!(answer.is_some());
        assert!(answer.unwrap().answer.contains("programming language"));
    }

    #[test]
    fn test_numeric_extraction() {
        let extractor = AnswerExtractor::default();

        let text = "The population is approximately 8 million people.";
        let answer = extractor.extract("How many people?", text);

        assert!(answer.is_some());
        assert_eq!(answer.unwrap().answer_type, AnswerType::Numeric);
    }

    #[test]
    fn test_list_extraction() {
        let extractor = AnswerExtractor::default();

        // Use proper list format with newlines
        let text = "The main types include:\n1. Easy syntax\n2. Large library\n3. Cross-platform";
        let answer = extractor.extract("What types of features are there?", text);

        // The extractor should find the list
        assert!(answer.is_some());
        assert_eq!(answer.unwrap().answer_type, AnswerType::List);
    }
}
