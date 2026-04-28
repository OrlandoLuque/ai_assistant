//! Message types for AI conversations

use crate::models::ModelInfo;
use serde::{Deserialize, Serialize};

/// Represents a chat message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp when the message was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional image attachments. Empty by default; populated by
    /// vision-aware surfaces (CLI `--image`, `send_message_with_images`,
    /// browser screenshot tool, etc.). `#[serde(default)]` only — bincode
    /// (the binary-storage format) ignores `skip_serializing_if` and would
    /// mis-align field positions on read if it were used.
    #[cfg(feature = "vision")]
    #[serde(default)]
    pub images: Vec<crate::vision::ImageInput>,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
            #[cfg(feature = "vision")]
            images: Vec::new(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
            #[cfg(feature = "vision")]
            images: Vec::new(),
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
            #[cfg(feature = "vision")]
            images: Vec::new(),
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

    /// Attach an image. Builder pattern for vision-aware callers.
    #[cfg(feature = "vision")]
    pub fn with_image(mut self, image: crate::vision::ImageInput) -> Self {
        self.images.push(image);
        self
    }

    /// Attach multiple images.
    #[cfg(feature = "vision")]
    pub fn with_images(mut self, images: Vec<crate::vision::ImageInput>) -> Self {
        self.images.extend(images);
        self
    }

    /// Whether this message carries any image attachments.
    #[cfg(feature = "vision")]
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

/// Response variants from AI generation
#[derive(Debug)]
#[non_exhaustive]
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
    /// An image emitted by the model — surfaces image-out from providers
    /// such as Gemini image generation and OpenAI image-out variants
    /// through the canonical response channel. Carries the same
    /// `ImageData` representation used elsewhere in the vision pipeline.
    #[cfg(feature = "vision")]
    Image(crate::vision::ImageData),
}

impl AiResponse {
    /// Check if this is a terminal response (Complete, Cancelled, or Error)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AiResponse::Complete(_) | AiResponse::Cancelled(_) | AiResponse::Error(_)
        )
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

    /// Borrow the contained [`crate::vision::ImageData`] when this is an
    /// [`AiResponse::Image`] (image-out from providers like Gemini /
    /// GPT-4o image variants). Returns `None` for any other variant.
    #[cfg(feature = "vision")]
    pub fn image(&self) -> Option<&crate::vision::ImageData> {
        match self {
            AiResponse::Image(img) => Some(img),
            _ => None,
        }
    }

    /// Convenience: return all images carried by this response. Currently
    /// returns 0 or 1 image (the variant carries a single payload), but
    /// the slice-shape mirrors the multi-image fields used elsewhere
    /// (e.g. [`ChatMessage::images`]) so future multi-image variants are
    /// non-breaking.
    #[cfg(feature = "vision")]
    pub fn images(&self) -> &[crate::vision::ImageData] {
        match self {
            AiResponse::Image(img) => std::slice::from_ref(img),
            _ => &[],
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
        assert_eq!(
            AiResponse::Complete("done".to_string()).text(),
            Some("done")
        );
        assert_eq!(
            AiResponse::Cancelled("part".to_string()).text(),
            Some("part")
        );
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

    #[test]
    fn test_chunk_is_not_terminal() {
        let chunk = AiResponse::Chunk("partial".to_string());
        assert!(!chunk.is_terminal());
        assert_eq!(chunk.text(), Some("partial"));
    }

    #[test]
    fn test_complete_is_terminal() {
        let complete = AiResponse::Complete("done".to_string());
        assert!(complete.is_terminal());
        assert_eq!(complete.text(), Some("done"));
    }

    #[test]
    fn test_message_role_checks() {
        let sys = ChatMessage::system("You are a helper");
        assert!(sys.is_system());
        assert!(!sys.is_user());
        assert!(!sys.is_assistant());
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_ai_response_image_variant_accessors() {
        let img = crate::vision::ImageData::Base64("AAAA".to_string());
        let resp = AiResponse::Image(img);
        // Not text-typed, not terminal-typed text-bearing.
        assert!(resp.text().is_none());
        // image() / images() expose the payload.
        assert!(resp.image().is_some());
        assert_eq!(resp.images().len(), 1);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_ai_response_image_url_variant() {
        let img = crate::vision::ImageData::Url("https://example.com/x.png".to_string());
        let resp = AiResponse::Image(img);
        match resp.image() {
            Some(crate::vision::ImageData::Url(u)) => {
                assert_eq!(u, "https://example.com/x.png");
            }
            other => panic!("expected Url, got {:?}", other),
        }
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_ai_response_non_image_returns_no_images() {
        let resp = AiResponse::Complete("text only".to_string());
        assert!(resp.image().is_none());
        assert!(resp.images().is_empty());
    }
}
