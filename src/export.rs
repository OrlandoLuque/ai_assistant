//! Advanced export and import functionality
//!
//! This module provides comprehensive export/import capabilities for
//! conversations, knowledge bases, and configurations.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

// ============================================================================
// Export Formats
// ============================================================================

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExportFormat {
    /// JSON format (human-readable)
    Json,
    /// Compressed JSON
    JsonCompressed,
    /// Markdown format (for conversations)
    Markdown,
    /// CSV format (for data)
    Csv,
    /// HTML format (for viewing)
    Html,
}

impl ExportFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::JsonCompressed => "json.gz",
            ExportFormat::Markdown => "md",
            ExportFormat::Csv => "csv",
            ExportFormat::Html => "html",
        }
    }

    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::Json | ExportFormat::JsonCompressed => "application/json",
            ExportFormat::Markdown => "text/markdown",
            ExportFormat::Csv => "text/csv",
            ExportFormat::Html => "text/html",
        }
    }
}

// ============================================================================
// Export Options
// ============================================================================

/// Options for export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Pretty print (for JSON)
    pub pretty_print: bool,
    /// Filter by date range
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    /// Maximum items to export
    pub max_items: Option<usize>,
    /// Include system messages
    pub include_system_messages: bool,
    /// Redact sensitive information
    pub redact_sensitive: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_metadata: true,
            include_timestamps: true,
            pretty_print: true,
            date_from: None,
            date_to: None,
            max_items: None,
            include_system_messages: false,
            redact_sensitive: false,
        }
    }
}

// ============================================================================
// Conversation Export
// ============================================================================

/// Exported conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedConversation {
    /// Conversation ID
    pub id: String,
    /// Title or summary
    pub title: String,
    /// Messages
    pub messages: Vec<ExportedMessage>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Exported message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMessage {
    /// Role (user/assistant/system)
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: Option<DateTime<Utc>>,
    /// Additional data
    pub metadata: Option<HashMap<String, String>>,
}

/// Conversation exporter
pub struct ConversationExporter {
    options: ExportOptions,
}

impl ConversationExporter {
    pub fn new(options: ExportOptions) -> Self {
        Self { options }
    }

    /// Export a single conversation to the specified format
    pub fn export(&self, conversation: &ExportedConversation) -> Result<String> {
        match self.options.format {
            ExportFormat::Json => self.to_json(conversation),
            ExportFormat::JsonCompressed => self.to_json(conversation), // Compression done at file level
            ExportFormat::Markdown => self.to_markdown(conversation),
            ExportFormat::Csv => self.to_csv(conversation),
            ExportFormat::Html => self.to_html(conversation),
        }
    }

    /// Export multiple conversations
    pub fn export_all(&self, conversations: &[ExportedConversation]) -> Result<String> {
        match self.options.format {
            ExportFormat::Json | ExportFormat::JsonCompressed => {
                let filtered = self.filter_conversations(conversations);
                if self.options.pretty_print {
                    Ok(serde_json::to_string_pretty(&filtered)?)
                } else {
                    Ok(serde_json::to_string(&filtered)?)
                }
            }
            ExportFormat::Markdown => {
                let mut output = String::new();
                for conv in self.filter_conversations(conversations) {
                    output.push_str(&self.to_markdown(&conv)?);
                    output.push_str("\n---\n\n");
                }
                Ok(output)
            }
            _ => {
                // For other formats, export first conversation
                if let Some(first) = conversations.first() {
                    self.export(first)
                } else {
                    Ok(String::new())
                }
            }
        }
    }

    /// Export to file
    pub fn export_to_file(&self, conversation: &ExportedConversation, path: &Path) -> Result<()> {
        let content = self.export(conversation)?;

        if self.options.format == ExportFormat::JsonCompressed {
            use flate2::write::GzEncoder;
            use flate2::Compression;

            let file = std::fs::File::create(path)?;
            let mut encoder = GzEncoder::new(file, Compression::default());
            encoder.write_all(content.as_bytes())?;
            encoder.finish()?;
        } else {
            std::fs::write(path, content)?;
        }

        Ok(())
    }

    fn filter_conversations(
        &self,
        conversations: &[ExportedConversation],
    ) -> Vec<ExportedConversation> {
        let mut result: Vec<ExportedConversation> = conversations
            .iter()
            .filter(|c| {
                if let Some(from) = self.options.date_from {
                    if c.created_at < from {
                        return false;
                    }
                }
                if let Some(to) = self.options.date_to {
                    if c.created_at > to {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        if let Some(max) = self.options.max_items {
            result.truncate(max);
        }

        result
    }

    fn to_json(&self, conversation: &ExportedConversation) -> Result<String> {
        if self.options.pretty_print {
            Ok(serde_json::to_string_pretty(conversation)?)
        } else {
            Ok(serde_json::to_string(conversation)?)
        }
    }

    fn to_markdown(&self, conversation: &ExportedConversation) -> Result<String> {
        let mut output = String::new();

        // Title
        output.push_str(&format!("# {}\n\n", conversation.title));

        // Metadata
        if self.options.include_metadata {
            output.push_str(&format!(
                "*Created: {}*\n\n",
                conversation.created_at.format("%Y-%m-%d %H:%M")
            ));
        }

        // Messages
        for msg in &conversation.messages {
            if !self.options.include_system_messages && msg.role == "system" {
                continue;
            }

            let role_emoji = match msg.role.as_str() {
                "user" => "**User:**",
                "assistant" => "**Assistant:**",
                "system" => "*System:*",
                _ => "**Unknown:**",
            };

            output.push_str(role_emoji);
            output.push('\n');

            if self.options.include_timestamps {
                if let Some(ts) = msg.timestamp {
                    output.push_str(&format!("*{}*\n", ts.format("%H:%M:%S")));
                }
            }

            let content = if self.options.redact_sensitive {
                self.redact_content(&msg.content)
            } else {
                msg.content.clone()
            };

            output.push_str(&content);
            output.push_str("\n\n");
        }

        Ok(output)
    }

    fn to_csv(&self, conversation: &ExportedConversation) -> Result<String> {
        let mut output = String::new();

        // Header
        output.push_str("timestamp,role,content\n");

        // Messages
        for msg in &conversation.messages {
            if !self.options.include_system_messages && msg.role == "system" {
                continue;
            }

            let timestamp = msg.timestamp.map(|ts| ts.to_rfc3339()).unwrap_or_default();

            let content = if self.options.redact_sensitive {
                self.redact_content(&msg.content)
            } else {
                msg.content.clone()
            };

            // Escape CSV
            let escaped_content =
                format!("\"{}\"", content.replace('"', "\"\"").replace('\n', "\\n"));

            output.push_str(&format!("{},{},{}\n", timestamp, msg.role, escaped_content));
        }

        Ok(output)
    }

    fn to_html(&self, conversation: &ExportedConversation) -> Result<String> {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html><head>\n");
        output.push_str("<meta charset=\"UTF-8\">\n");
        output.push_str(&format!("<title>{}</title>\n", conversation.title));
        output.push_str("<style>\n");
        output.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n");
        output.push_str(".message { margin: 10px 0; padding: 10px; border-radius: 8px; }\n");
        output.push_str(".user { background: #e3f2fd; }\n");
        output.push_str(".assistant { background: #f5f5f5; }\n");
        output.push_str(".system { background: #fff3e0; font-style: italic; }\n");
        output.push_str(".role { font-weight: bold; margin-bottom: 5px; }\n");
        output.push_str(".timestamp { font-size: 0.8em; color: #666; }\n");
        output.push_str("pre { background: #282c34; color: #abb2bf; padding: 10px; border-radius: 4px; overflow-x: auto; }\n");
        output.push_str("</style>\n</head><body>\n");

        output.push_str(&format!("<h1>{}</h1>\n", conversation.title));

        if self.options.include_metadata {
            output.push_str(&format!(
                "<p class=\"timestamp\">Created: {}</p>\n",
                conversation.created_at.format("%Y-%m-%d %H:%M")
            ));
        }

        for msg in &conversation.messages {
            if !self.options.include_system_messages && msg.role == "system" {
                continue;
            }

            let class = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                "system" => "system",
                _ => "message",
            };

            output.push_str(&format!("<div class=\"message {}\">\n", class));
            output.push_str(&format!(
                "<div class=\"role\">{}</div>\n",
                msg.role
                    .chars()
                    .next()
                    .map(|c| c.to_uppercase().to_string())
                    .unwrap_or_default()
                    + &msg.role[1..]
            ));

            if self.options.include_timestamps {
                if let Some(ts) = msg.timestamp {
                    output.push_str(&format!(
                        "<div class=\"timestamp\">{}</div>\n",
                        ts.format("%H:%M:%S")
                    ));
                }
            }

            let content = if self.options.redact_sensitive {
                self.redact_content(&msg.content)
            } else {
                msg.content.clone()
            };

            // Convert markdown code blocks to HTML
            let html_content = self.markdown_to_html(&content);
            output.push_str(&format!("<div class=\"content\">{}</div>\n", html_content));
            output.push_str("</div>\n");
        }

        output.push_str("</body></html>\n");
        Ok(output)
    }

    fn markdown_to_html(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Convert code blocks
        while let Some(start) = result.find("```") {
            if let Some(end) = result[start + 3..].find("```") {
                let code = &result[start + 3..start + 3 + end];
                let (lang, code_content) = if let Some(newline) = code.find('\n') {
                    (&code[..newline], &code[newline + 1..])
                } else {
                    ("", code)
                };

                let html = format!(
                    "<pre><code class=\"language-{}\">{}</code></pre>",
                    lang,
                    html_escape(code_content)
                );

                result = format!(
                    "{}{}{}",
                    &result[..start],
                    html,
                    &result[start + 3 + end + 3..]
                );
            } else {
                break;
            }
        }

        // Convert inline code
        while result.contains('`') {
            if let (Some(start), Some(end)) = (
                result.find('`'),
                result[result.find('`').expect("char verified above") + 1..].find('`'),
            ) {
                let code = &result[start + 1..start + 1 + end];
                let html = format!("<code>{}</code>", html_escape(code));
                result = format!(
                    "{}{}{}",
                    &result[..start],
                    html,
                    &result[start + 1 + end + 1..]
                );
            } else {
                break;
            }
        }

        // Convert newlines to <br>
        result = result.replace("\n", "<br>\n");

        result
    }

    fn redact_content(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Redact email addresses
        let email_pattern = |s: &str| s.contains('@') && s.contains('.');

        for word in content.split_whitespace() {
            if email_pattern(word) {
                result = result.replace(word, "[EMAIL REDACTED]");
            }
        }

        // Redact potential API keys (long alphanumeric strings)
        for word in content.split_whitespace() {
            if word.len() > 20
                && word
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                result = result.replace(word, "[KEY REDACTED]");
            }
        }

        result
    }
}

// Helper function
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// ============================================================================
// Import
// ============================================================================

/// Import options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    /// Merge with existing data
    pub merge: bool,
    /// Skip duplicates
    pub skip_duplicates: bool,
    /// Validate before import
    pub validate: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            merge: true,
            skip_duplicates: true,
            validate: true,
        }
    }
}

/// Import result
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Number of items imported
    pub imported: usize,
    /// Number of items skipped
    pub skipped: usize,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Conversation importer
pub struct ConversationImporter {
    options: ImportOptions,
}

impl ConversationImporter {
    pub fn new(options: ImportOptions) -> Self {
        Self { options }
    }

    /// Get import options.
    pub fn options(&self) -> &ImportOptions {
        &self.options
    }

    /// Import from JSON string
    pub fn import_json(&self, json: &str) -> Result<(Vec<ExportedConversation>, ImportResult)> {
        let mut result = ImportResult {
            imported: 0,
            skipped: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Try to parse as array first
        let conversations: Vec<ExportedConversation> = match serde_json::from_str(json) {
            Ok(convs) => convs,
            Err(_) => {
                // Try as single conversation
                match serde_json::from_str::<ExportedConversation>(json) {
                    Ok(conv) => vec![conv],
                    Err(e) => {
                        result.errors.push(format!("Failed to parse JSON: {}", e));
                        return Ok((Vec::new(), result));
                    }
                }
            }
        };

        result.imported = conversations.len();
        Ok((conversations, result))
    }

    /// Import from file
    pub fn import_file(&self, path: &Path) -> Result<(Vec<ExportedConversation>, ImportResult)> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let content = if extension == "gz" {
            use flate2::read::GzDecoder;
            let file = std::fs::File::open(path)?;
            let mut decoder = GzDecoder::new(file);
            let mut content = String::new();
            decoder.read_to_string(&mut content)?;
            content
        } else {
            std::fs::read_to_string(path)?
        };

        self.import_json(&content)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_conversation() -> ExportedConversation {
        ExportedConversation {
            id: "test-123".to_string(),
            title: "Test Conversation".to_string(),
            messages: vec![
                ExportedMessage {
                    role: "user".to_string(),
                    content: "Hello, how are you?".to_string(),
                    timestamp: Some(Utc::now()),
                    metadata: None,
                },
                ExportedMessage {
                    role: "assistant".to_string(),
                    content: "I'm doing well, thank you!".to_string(),
                    timestamp: Some(Utc::now()),
                    metadata: None,
                },
            ],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_export_json() {
        let exporter = ConversationExporter::new(ExportOptions::default());
        let conv = create_test_conversation();

        let json = exporter.export(&conv).unwrap();
        assert!(json.contains("Test Conversation"));
        assert!(json.contains("Hello, how are you?"));
    }

    #[test]
    fn test_export_markdown() {
        let mut options = ExportOptions::default();
        options.format = ExportFormat::Markdown;

        let exporter = ConversationExporter::new(options);
        let conv = create_test_conversation();

        let md = exporter.export(&conv).unwrap();
        assert!(md.contains("# Test Conversation"));
        assert!(md.contains("**User:**"));
    }

    #[test]
    fn test_export_html() {
        let mut options = ExportOptions::default();
        options.format = ExportFormat::Html;

        let exporter = ConversationExporter::new(options);
        let conv = create_test_conversation();

        let html = exporter.export(&conv).unwrap();
        assert!(html.contains("<html>"));
        assert!(html.contains("Test Conversation"));
    }

    #[test]
    fn test_import_json() {
        let importer = ConversationImporter::new(ImportOptions::default());
        let conv = create_test_conversation();

        let json = serde_json::to_string(&conv).unwrap();
        let (imported, result) = importer.import_json(&json).unwrap();

        assert_eq!(imported.len(), 1);
        assert_eq!(result.imported, 1);
    }

    #[test]
    fn test_redaction() {
        let exporter = ConversationExporter::new(ExportOptions {
            redact_sensitive: true,
            ..Default::default()
        });

        let content = exporter.redact_content("Contact me at test@example.com");
        assert!(content.contains("[EMAIL REDACTED]"));
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::JsonCompressed.extension(), "json.gz");
        assert_eq!(ExportFormat::Markdown.extension(), "md");
        assert_eq!(ExportFormat::Csv.extension(), "csv");
        assert_eq!(ExportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_format_mime_types() {
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
        assert_eq!(ExportFormat::Markdown.mime_type(), "text/markdown");
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
        assert_eq!(ExportFormat::Html.mime_type(), "text/html");
    }

    #[test]
    fn test_export_csv() {
        let mut options = ExportOptions::default();
        options.format = ExportFormat::Csv;

        let exporter = ConversationExporter::new(options);
        let conv = create_test_conversation();

        let csv = exporter.export(&conv).unwrap();
        assert!(csv.contains("timestamp,role,content"));
        assert!(csv.contains("user"));
    }

    #[test]
    fn test_export_options_defaults() {
        let options = ExportOptions::default();
        assert_eq!(options.format, ExportFormat::Json);
        assert!(options.include_metadata);
        assert!(options.include_timestamps);
        assert!(options.pretty_print);
        assert!(options.date_from.is_none());
        assert!(options.max_items.is_none());
        assert!(!options.redact_sensitive);
    }

    #[test]
    fn test_export_json_pretty() {
        let options = ExportOptions {
            pretty_print: true,
            ..Default::default()
        };
        let exporter = ConversationExporter::new(options);
        let conv = create_test_conversation();

        let json = exporter.export(&conv).unwrap();
        assert!(json.contains('\n')); // Pretty-printed has newlines
        assert!(json.contains("Test Conversation"));
    }
}
