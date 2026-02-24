//! Prelude module for convenient imports
//!
//! This module re-exports the most commonly used types so users can get started
//! with a single `use ai_assistant::prelude::*;` import.
//!
//! # Example
//!
//! ```no_run
//! use ai_assistant::prelude::*;
//!
//! let assistant = AiAssistant::new();
//! ```

// Core types
pub use crate::assistant::AiAssistant;
pub use crate::config::{AiConfig, AiProvider};
pub use crate::error::{AiError, AiResult, ContextualError, ResultExt};
pub use crate::events::{AiEvent, EventBus, EventHandler};
pub use crate::messages::{AiResponse, ChatMessage};
pub use crate::progress::{Progress, ProgressCallback};

// Configuration
pub use crate::config_file::ConfigFile;

// Server
pub use crate::server::{AiServer, AuthConfig, CorsConfig, ServerConfig, ServerHandle};

// Providers
pub use crate::providers::generate_response;

// Guardrails
pub use crate::guardrail_pipeline::{Guard, GuardAction, GuardStage, GuardrailPipeline};

#[cfg(test)]
mod tests {
    #[test]
    fn test_prelude_imports_core() {
        // Verify that core prelude types are accessible
        use super::*;
        let _ = AiAssistant::new();
        let _ = AiConfig::default();
        let err = AiError::other("test error");
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_prelude_error_types() {
        use super::*;
        let err: AiResult<()> = Err(AiError::other("test"));
        assert!(err.is_err());
    }

    #[test]
    fn test_prelude_config_default() {
        use super::*;
        let config = AiConfig::default();
        assert!(!format!("{:?}", config).is_empty());
    }

    #[test]
    fn test_prelude_chat_message() {
        use super::*;
        let msg = ChatMessage::user("hello");
        assert_eq!(msg.content, "hello");
        assert_eq!(msg.role, "user");
    }
}
