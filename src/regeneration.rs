//! Response regeneration with feedback
//!
//! Allows regenerating responses with user feedback to improve results.

use std::collections::HashMap;
use std::time::Instant;

/// Feedback for regeneration
#[derive(Debug, Clone)]
pub struct RegenerationFeedback {
    /// What was wrong
    pub issue: RegenerationIssue,
    /// Additional instructions
    pub instructions: Option<String>,
    /// Preferred style
    pub style: Option<ResponseStyle>,
    /// Length preference
    pub length: Option<LengthPreference>,
}

/// Types of issues with responses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegenerationIssue {
    /// Too long
    TooLong,
    /// Too short
    TooShort,
    /// Off topic
    OffTopic,
    /// Incorrect information
    Incorrect,
    /// Too technical
    TooTechnical,
    /// Too simple
    TooSimple,
    /// Wrong format
    WrongFormat,
    /// Other
    Other,
}

/// Response style preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStyle {
    Formal,
    Casual,
    Technical,
    Simple,
    Creative,
    Factual,
}

/// Length preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthPreference {
    VeryShort,
    Short,
    Medium,
    Long,
    VeryLong,
}

impl LengthPreference {
    pub fn to_instruction(&self) -> &'static str {
        match self {
            Self::VeryShort => "Be extremely brief, 1-2 sentences max.",
            Self::Short => "Keep the response short and concise.",
            Self::Medium => "Provide a moderately detailed response.",
            Self::Long => "Provide a comprehensive, detailed response.",
            Self::VeryLong => "Provide an exhaustive, thorough response.",
        }
    }
}

/// Regeneration request
#[derive(Debug, Clone)]
pub struct RegenerationRequest {
    pub original_prompt: String,
    pub original_response: String,
    pub feedback: RegenerationFeedback,
    pub attempt: u32,
    pub created_at: Instant,
}

impl RegenerationRequest {
    pub fn new(prompt: &str, response: &str, feedback: RegenerationFeedback) -> Self {
        Self {
            original_prompt: prompt.to_string(),
            original_response: response.to_string(),
            feedback,
            attempt: 1,
            created_at: Instant::now(),
        }
    }

    /// Build improved prompt incorporating feedback
    pub fn build_improved_prompt(&self) -> String {
        let mut parts = Vec::new();

        // Add issue-specific instruction
        parts.push(match self.feedback.issue {
            RegenerationIssue::TooLong => "Please provide a shorter, more concise response.",
            RegenerationIssue::TooShort => "Please provide a more detailed response.",
            RegenerationIssue::OffTopic => "Please focus directly on the question asked.",
            RegenerationIssue::Incorrect => "Please verify the information and correct any errors.",
            RegenerationIssue::TooTechnical => "Please explain in simpler terms.",
            RegenerationIssue::TooSimple => "Please provide more technical depth.",
            RegenerationIssue::WrongFormat => "Please use proper formatting.",
            RegenerationIssue::Other => "",
        }.to_string());

        // Add length preference
        if let Some(length) = self.feedback.length {
            parts.push(length.to_instruction().to_string());
        }

        // Add style preference
        if let Some(style) = self.feedback.style {
            parts.push(match style {
                ResponseStyle::Formal => "Use a formal, professional tone.",
                ResponseStyle::Casual => "Use a casual, friendly tone.",
                ResponseStyle::Technical => "Use technical terminology.",
                ResponseStyle::Simple => "Use simple, easy to understand language.",
                ResponseStyle::Creative => "Be creative and engaging.",
                ResponseStyle::Factual => "Focus on facts and be objective.",
            }.to_string());
        }

        // Add custom instructions
        if let Some(instructions) = &self.feedback.instructions {
            parts.push(instructions.clone());
        }

        // Combine
        let instructions = parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");

        format!(
            "{}\n\n[Instructions for improvement: {}]\n\nOriginal question: {}",
            instructions,
            if instructions.is_empty() { "Please try again with a better response".to_string() } else { instructions.clone() },
            self.original_prompt
        )
    }
}

/// Manager for regeneration attempts
pub struct RegenerationManager {
    history: HashMap<String, Vec<RegenerationRequest>>,
    max_attempts: u32,
}

impl RegenerationManager {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            history: HashMap::new(),
            max_attempts,
        }
    }

    pub fn request_regeneration(
        &mut self,
        conversation_id: &str,
        prompt: &str,
        response: &str,
        feedback: RegenerationFeedback,
    ) -> Option<RegenerationRequest> {
        let history = self.history.entry(conversation_id.to_string()).or_default();

        let attempt = history.len() as u32 + 1;
        if attempt > self.max_attempts {
            return None;
        }

        let mut request = RegenerationRequest::new(prompt, response, feedback);
        request.attempt = attempt;
        history.push(request.clone());

        Some(request)
    }

    pub fn get_history(&self, conversation_id: &str) -> Vec<&RegenerationRequest> {
        self.history.get(conversation_id)
            .map(|h| h.iter().collect())
            .unwrap_or_default()
    }

    pub fn clear_history(&mut self, conversation_id: &str) {
        self.history.remove(conversation_id);
    }
}

impl Default for RegenerationManager {
    fn default() -> Self {
        Self::new(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regeneration_request() {
        let feedback = RegenerationFeedback {
            issue: RegenerationIssue::TooLong,
            instructions: Some("Focus on key points".to_string()),
            style: Some(ResponseStyle::Simple),
            length: Some(LengthPreference::Short),
        };

        let request = RegenerationRequest::new("What is AI?", "Long response...", feedback);
        let prompt = request.build_improved_prompt();

        assert!(prompt.contains("shorter"));
        assert!(prompt.contains("simple"));
    }

    #[test]
    fn test_manager() {
        let mut manager = RegenerationManager::new(3);

        let feedback = RegenerationFeedback {
            issue: RegenerationIssue::TooShort,
            instructions: None,
            style: None,
            length: None,
        };

        let r1 = manager.request_regeneration("conv1", "Q", "A", feedback.clone());
        assert!(r1.is_some());
        assert_eq!(r1.unwrap().attempt, 1);

        let r2 = manager.request_regeneration("conv1", "Q", "A", feedback.clone());
        assert_eq!(r2.unwrap().attempt, 2);
    }
}
