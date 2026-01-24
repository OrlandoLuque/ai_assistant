//! Message types for AI conversations

use serde::{Deserialize, Serialize};
use crate::models::ModelInfo;

/// Represents a chat message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp when the message was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Check if this is a user message
    pub fn is_user(&self) -> bool {
        self.role == "user"
    }

    /// Check if this is an assistant message
    pub fn is_assistant(&self) -> bool {
        self.role == "assistant"
    }

    /// Check if this is a system message
    pub fn is_system(&self) -> bool {
        self.role == "system"
    }
}

/// Response variants from AI generation
#[derive(Debug)]
pub enum AiResponse {
    /// A streaming chunk of text
    Chunk(String),
    /// Complete response text
    Complete(String),
    /// Response was cancelled (contains partial response)
    Cancelled(String),
    /// Error message
    Error(String),
    /// List of available models (from model discovery)
    ModelsLoaded(Vec<ModelInfo>),
}

impl AiResponse {
    /// Check if this is a terminal response (Complete, Cancelled, or Error)
    pub fn is_terminal(&self) -> bool {
        matches!(self, AiResponse::Complete(_) | AiResponse::Cancelled(_) | AiResponse::Error(_))
    }

    /// Get the text content if this is a Chunk, Complete, or Cancelled response
    pub fn text(&self) -> Option<&str> {
        match self {
            AiResponse::Chunk(s) | AiResponse::Complete(s) | AiResponse::Cancelled(s) => Some(s),
            _ => None,
        }
    }

    /// Check if this is an error
    pub fn is_error(&self) -> bool {
        matches!(self, AiResponse::Error(_))
    }

    /// Check if response was cancelled
    pub fn is_cancelled(&self) -> bool {
        matches!(self, AiResponse::Cancelled(_))
    }

    /// Get the error message if this is an error
    pub fn error(&self) -> Option<&str> {
        match self {
            AiResponse::Error(e) => Some(e),
            _ => None,
        }
    }

    /// Get the partial response if cancelled
    pub fn partial(&self) -> Option<&str> {
        match self {
            AiResponse::Cancelled(s) => Some(s),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_user() {
        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
        assert!(msg.is_user());
        assert!(!msg.is_assistant());
        assert!(!msg.is_system());
    }

    #[test]
    fn test_chat_message_assistant() {
        let msg = ChatMessage::assistant("Hi there");
        assert_eq!(msg.role, "assistant");
        assert!(msg.is_assistant());
        assert!(!msg.is_user());
    }

    #[test]
    fn test_chat_message_system() {
        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert!(msg.is_system());
    }

    #[test]
    fn test_ai_response_terminal() {
        assert!(AiResponse::Complete("done".to_string()).is_terminal());
        assert!(AiResponse::Cancelled("partial".to_string()).is_terminal());
        assert!(AiResponse::Error("oops".to_string()).is_terminal());
        assert!(!AiResponse::Chunk("chunk".to_string()).is_terminal());
    }

    #[test]
    fn test_ai_response_text() {
        assert_eq!(AiResponse::Chunk("hello".to_string()).text(), Some("hello"));
        assert_eq!(AiResponse::Complete("done".to_string()).text(), Some("done"));
        assert_eq!(AiResponse::Cancelled("part".to_string()).text(), Some("part"));
        assert!(AiResponse::ModelsLoaded(vec![]).text().is_none());
    }

    #[test]
    fn test_ai_response_error() {
        let err = AiResponse::Error("connection failed".to_string());
        assert!(err.is_error());
        assert_eq!(err.error(), Some("connection failed"));
        assert!(!AiResponse::Complete("ok".to_string()).is_error());
    }

    #[test]
    fn test_ai_response_cancelled() {
        let cancelled = AiResponse::Cancelled("partial output".to_string());
        assert!(cancelled.is_cancelled());
        assert_eq!(cancelled.partial(), Some("partial output"));
    }
}
