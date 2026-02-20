//! Security, rate limiting, and audit logging module
//!
//! This module provides security-related functionality for the AI assistant:
//!
//! - **Rate limiting**: Control request frequency to prevent abuse
//! - **Input sanitization**: Clean and validate user input
//! - **Audit logging**: Track all operations for compliance and debugging
//! - **Message hooks**: Pre/post processing of messages
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::security::{
//!     RateLimiter, RateLimitConfig,
//!     InputSanitizer, SanitizationConfig,
//!     AuditLogger, AuditConfig, AuditEvent, AuditEventType,
//!     HookManager, HookResult,
//! };
//!
//! // Rate limiting
//! let mut limiter = RateLimiter::new(RateLimitConfig::default());
//! if limiter.check_allowed().is_allowed() {
//!     limiter.record_request_start();
//!     // ... make request ...
//!     limiter.record_request_end(100); // tokens used
//! }
//!
//! // Input sanitization
//! let sanitizer = InputSanitizer::new(SanitizationConfig::default());
//! let result = sanitizer.sanitize("user input here");
//! if let Some(clean_input) = result.get_output() {
//!     // use clean_input
//! }
//!
//! // Audit logging
//! let mut logger = AuditLogger::new(AuditConfig::default());
//! logger.log(AuditEvent::new(AuditEventType::MessageSent)
//!     .with_user("user123")
//!     .with_detail("tokens", "150"));
//! ```

mod audit;
mod hooks;
mod rate_limiting;
mod sanitization;

// Re-export all public types
pub use rate_limiting::{
    RateLimitConfig, RateLimitReason, RateLimitResult, RateLimitStatus, RateLimitUsage, RateLimiter,
};

pub use sanitization::{
    InputSanitizer, SanitizationConfig, SanitizationResult, SanitizationWarning,
};

pub use audit::{AuditConfig, AuditEvent, AuditEventType, AuditLogger, AuditStats};

pub use hooks::{HookChainResult, HookManager, HookResult, PostMessageHook, PreMessageHook};
