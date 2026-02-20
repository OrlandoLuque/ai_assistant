//! Log redaction module for stripping sensitive data before output.
//!
//! Provides pattern-based redaction of API keys, Bearer tokens, passwords
//! in URLs, PEM keys, and other sensitive patterns. Always available (no
//! feature flag required).
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::log_redaction::redact;
//!
//! let text = "Authorization: Bearer sk-abc123xyz";
//! let safe = redact(text);
//! assert!(!safe.contains("sk-abc123xyz"));
//! assert!(safe.contains("***"));
//! ```

use regex::Regex;
use std::sync::OnceLock;

/// Configuration for log redaction behavior.
#[derive(Debug, Clone)]
pub struct RedactionConfig {
    /// Whether redaction is enabled. Default: true.
    pub enabled: bool,
    /// Redact API key patterns (sk-*, key-*, api_key=*). Default: true.
    pub redact_api_keys: bool,
    /// Redact Bearer/Basic auth tokens. Default: true.
    pub redact_auth_tokens: bool,
    /// Redact passwords in URLs (://user:pass@host). Default: true.
    pub redact_url_passwords: bool,
    /// Redact PEM-encoded keys/certificates. Default: true.
    pub redact_pem_blocks: bool,
    /// Redact email addresses. Default: false (usually not sensitive in logs).
    pub redact_emails: bool,
    /// Custom patterns to redact (compiled regexes).
    pub custom_patterns: Vec<String>,
    /// Replacement text for redacted content. Default: "***REDACTED***".
    pub replacement: String,
}

impl Default for RedactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            redact_api_keys: true,
            redact_auth_tokens: true,
            redact_url_passwords: true,
            redact_pem_blocks: true,
            redact_emails: false,
            custom_patterns: Vec::new(),
            replacement: "***REDACTED***".to_string(),
        }
    }
}

// ============================================================================
// Compiled regex patterns (initialized once)
// ============================================================================

struct RedactionPatterns {
    api_keys: Vec<Regex>,
    auth_tokens: Vec<Regex>,
    url_passwords: Regex,
    pem_blocks: Regex,
    emails: Regex,
}

fn patterns() -> &'static RedactionPatterns {
    static PATTERNS: OnceLock<RedactionPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        RedactionPatterns {
            api_keys: vec![
                // OpenAI-style: sk-...
                Regex::new(r"(?i)\bsk-[a-zA-Z0-9]{20,}").expect("valid regex"),
                // Generic key-...
                Regex::new(r"(?i)\bkey-[a-zA-Z0-9]{10,}").expect("valid regex"),
                // api_key=VALUE or api-key=VALUE in query strings or headers
                Regex::new(r#"(?i)(api[_-]?key\s*[=:]\s*)[^\s&"',]+"#).expect("valid regex"),
                // x-api-key: VALUE header
                Regex::new(r"(?i)(x-api-key:\s*)[^\s]+").expect("valid regex"),
                // Generic secret/token assignment patterns
                Regex::new(r#"(?i)((?:secret|token|password|passwd|pwd)\s*[=:]\s*)"?[^\s&"',]+"#).expect("valid regex"),
            ],
            auth_tokens: vec![
                // Bearer TOKEN
                Regex::new(r"(?i)(Bearer\s+)[^\s]+").expect("valid regex"),
                // Basic BASE64
                Regex::new(r"(?i)(Basic\s+)[^\s]+").expect("valid regex"),
                // Authorization: ...
                Regex::new(r"(?i)(Authorization:\s*(?:Bearer|Basic|Token)\s+)[^\s]+").expect("valid regex"),
            ],
            // ://user:password@host
            url_passwords: Regex::new(r"(://[^:@\s]+:)[^@\s]+(@)").expect("valid regex"),
            // PEM blocks
            pem_blocks: Regex::new(
                r"-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----"
            ).expect("valid regex"),
            // Emails (from pii_detection.rs pattern)
            emails: Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").expect("valid regex"),
        }
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Redact sensitive patterns from text using default configuration.
///
/// Replaces API keys, Bearer tokens, URL passwords, and PEM keys with
/// `***REDACTED***`. This is the most common entry point for log safety.
pub fn redact(text: &str) -> String {
    redact_with_config(text, &RedactionConfig::default())
}

/// Redact sensitive patterns from text using custom configuration.
///
/// Allows fine-grained control over which patterns are redacted and what
/// replacement string is used.
pub fn redact_with_config(text: &str, config: &RedactionConfig) -> String {
    if !config.enabled || text.is_empty() {
        return text.to_string();
    }

    let p = patterns();
    let mut result = text.to_string();
    let replacement = &config.replacement;

    if config.redact_pem_blocks {
        result = p
            .pem_blocks
            .replace_all(&result, "***PEM_KEY***")
            .to_string();
    }

    if config.redact_api_keys {
        for pattern in &p.api_keys {
            // For patterns with capture groups (prefix + value), keep prefix
            if pattern.captures_len() > 1 {
                result = pattern
                    .replace_all(&result, |caps: &regex::Captures| {
                        format!("{}{}", &caps[1], replacement)
                    })
                    .to_string();
            } else {
                result = pattern
                    .replace_all(&result, replacement.as_str())
                    .to_string();
            }
        }
    }

    if config.redact_auth_tokens {
        for pattern in &p.auth_tokens {
            result = pattern
                .replace_all(&result, |caps: &regex::Captures| {
                    format!("{}{}", &caps[1], replacement)
                })
                .to_string();
        }
    }

    if config.redact_url_passwords {
        result = p
            .url_passwords
            .replace_all(&result, |caps: &regex::Captures| {
                format!("{}***{}", &caps[1], &caps[2])
            })
            .to_string();
    }

    if config.redact_emails {
        result = p
            .emails
            .replace_all(&result, replacement.as_str())
            .to_string();
    }

    // Custom patterns
    for custom in &config.custom_patterns {
        if let Ok(re) = Regex::new(custom) {
            result = re.replace_all(&result, replacement.as_str()).to_string();
        }
    }

    result
}

/// Check if a text contains any sensitive patterns that would be redacted.
///
/// Useful for conditional logging: only call `redact()` when needed.
pub fn contains_sensitive(text: &str) -> bool {
    let p = patterns();
    for pattern in &p.api_keys {
        if pattern.is_match(text) {
            return true;
        }
    }
    for pattern in &p.auth_tokens {
        if pattern.is_match(text) {
            return true;
        }
    }
    if p.url_passwords.is_match(text) || p.pem_blocks.is_match(text) {
        return true;
    }
    false
}

/// Macro for safe logging that automatically redacts sensitive data.
///
/// Uses `log::warn!` internally and passes output through `redact()` first.
///
/// # Example
///
/// ```rust
/// use ai_assistant::safe_log;
///
/// let api_key = "sk-abc123xyz456";
/// safe_log!("Connecting with key: {}", api_key);
/// // Logs at warn level: "Connecting with key: ***REDACTED***"
/// ```
#[macro_export]
macro_rules! safe_log {
    ($($arg:tt)*) => {
        log::warn!("{}", $crate::log_redaction::redact(&format!($($arg)*)));
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_openai_api_key() {
        let text = "Using key sk-abc123456789012345678901234567890";
        let result = redact(text);
        assert!(!result.contains("sk-abc123456789012345678901234567890"));
        assert!(result.contains("***REDACTED***"));
    }

    #[test]
    fn test_redact_generic_key() {
        let text = "api_key=mySecretKeyValue1234";
        let result = redact(text);
        assert!(!result.contains("mySecretKeyValue1234"));
    }

    #[test]
    fn test_redact_bearer_token() {
        let text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature";
        let result = redact(text);
        assert!(!result.contains("eyJhbGciOiJ"));
        assert!(result.contains("Bearer"));
        assert!(result.contains("***REDACTED***"));
    }

    #[test]
    fn test_redact_basic_auth() {
        let text = "Basic dXNlcjpwYXNzd29yZA==";
        let result = redact(text);
        assert!(!result.contains("dXNlcjpwYXNzd29yZA=="));
        assert!(result.contains("Basic"));
    }

    #[test]
    fn test_redact_url_password() {
        let text = "Connecting to https://admin:supersecret123@db.example.com:5432/mydb";
        let result = redact(text);
        assert!(!result.contains("supersecret123"));
        assert!(result.contains("admin:***@db.example.com"));
    }

    #[test]
    fn test_redact_pem_key() {
        let text = "Key: -----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg...\n-----END PRIVATE KEY-----";
        let result = redact(text);
        assert!(!result.contains("MIIEvgIBADANBg"));
        assert!(result.contains("***PEM_KEY***"));
    }

    #[test]
    fn test_redact_x_api_key_header() {
        let text = "x-api-key: my-secret-api-key-12345";
        let result = redact(text);
        assert!(!result.contains("my-secret-api-key-12345"));
    }

    #[test]
    fn test_no_sensitive_data_unchanged() {
        let text = "Hello, this is a normal log message about model llama-3.2";
        let result = redact(text);
        assert_eq!(text, result);
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(redact(""), "");
    }

    #[test]
    fn test_disabled_config() {
        let config = RedactionConfig {
            enabled: false,
            ..Default::default()
        };
        let text = "Bearer secret123";
        let result = redact_with_config(text, &config);
        assert_eq!(text, result);
    }

    #[test]
    fn test_custom_replacement() {
        let config = RedactionConfig {
            replacement: "[HIDDEN]".to_string(),
            ..Default::default()
        };
        let text = "sk-abc123456789012345678901";
        let result = redact_with_config(text, &config);
        assert!(result.contains("[HIDDEN]"));
    }

    #[test]
    fn test_custom_patterns() {
        let config = RedactionConfig {
            custom_patterns: vec![r"session_id=[a-f0-9]+".to_string()],
            ..Default::default()
        };
        let text = "Request with session_id=abc123def456";
        let result = redact_with_config(text, &config);
        assert!(!result.contains("abc123def456"));
    }

    #[test]
    fn test_email_redaction_opt_in() {
        let text = "Contact user@example.com for details";

        // Default: emails NOT redacted
        let result = redact(text);
        assert!(result.contains("user@example.com"));

        // Opt-in: emails redacted
        let config = RedactionConfig {
            redact_emails: true,
            ..Default::default()
        };
        let result = redact_with_config(text, &config);
        assert!(!result.contains("user@example.com"));
    }

    #[test]
    fn test_contains_sensitive_true() {
        assert!(contains_sensitive("Bearer mytoken123"));
        assert!(contains_sensitive("sk-abc12345678901234567890"));
        assert!(contains_sensitive("https://user:pass@host.com"));
    }

    #[test]
    fn test_contains_sensitive_false() {
        assert!(!contains_sensitive("Hello world"));
        assert!(!contains_sensitive("Just a normal URL https://example.com"));
    }

    #[test]
    fn test_multiple_sensitive_items_in_one_string() {
        let text = "Bearer token123 and api_key=secret456 with sk-keyvalue78901234567890";
        let result = redact(text);
        assert!(!result.contains("token123"));
        assert!(!result.contains("secret456"));
        assert!(!result.contains("keyvalue789"));
    }

    #[test]
    fn test_password_secret_token_patterns() {
        let text = "password=mypass123 and secret=topsecret and token=abc123";
        let result = redact(text);
        assert!(!result.contains("mypass123"));
        assert!(!result.contains("topsecret"));
        assert!(!result.contains("abc123"));
    }
}
