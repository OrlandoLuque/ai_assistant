//! Code-specific editing operations
//!
//! This module provides code-aware editing functionality that understands
//! programming constructs and can make intelligent edits.
//!
//! # Features
//!
//! - **Search and replace in code**: Scope-aware replacements
//! - **Comment handling**: Add, remove, toggle comments
//! - **Indentation management**: Smart indent/dedent
//! - **Code block operations**: Extract, move, duplicate blocks
//! - **Edit suggestions**: Generate edit suggestions with explanations

use crate::edit_operations::{TextEditor, EditError};

/// Language configuration for code editing
#[derive(Debug, Clone)]
pub struct LanguageConfig {
    /// Language identifier
    pub language: String,
    /// Single-line comment prefix
    pub line_comment: Option<String>,
    /// Block comment start
    pub block_comment_start: Option<String>,
    /// Block comment end
    pub block_comment_end: Option<String>,
    /// String delimiters
    pub string_delimiters: Vec<char>,
    /// Indent size
    pub indent_size: usize,
    /// Use tabs for indentation
    pub use_tabs: bool,
    /// File extensions
    pub extensions: Vec<String>,
}

impl LanguageConfig {
    /// Get configuration for Rust
    pub fn rust() -> Self {
        Self {
            language: "rust".to_string(),
            line_comment: Some("//".to_string()),
            block_comment_start: Some("/*".to_string()),
            block_comment_end: Some("*/".to_string()),
            string_delimiters: vec!['"'],
            indent_size: 4,
            use_tabs: false,
            extensions: vec!["rs".to_string()],
        }
    }

    /// Get configuration for Python
    pub fn python() -> Self {
        Self {
            language: "python".to_string(),
            line_comment: Some("#".to_string()),
            block_comment_start: Some("'''".to_string()),
            block_comment_end: Some("'''".to_string()),
            string_delimiters: vec!['"', '\''],
            indent_size: 4,
            use_tabs: false,
            extensions: vec!["py".to_string()],
        }
    }

    /// Get configuration for JavaScript/TypeScript
    pub fn javascript() -> Self {
        Self {
            language: "javascript".to_string(),
            line_comment: Some("//".to_string()),
            block_comment_start: Some("/*".to_string()),
            block_comment_end: Some("*/".to_string()),
            string_delimiters: vec!['"', '\'', '`'],
            indent_size: 2,
            use_tabs: false,
            extensions: vec!["js".to_string(), "ts".to_string(), "jsx".to_string(), "tsx".to_string()],
        }
    }

    /// Get configuration for HTML
    pub fn html() -> Self {
        Self {
            language: "html".to_string(),
            line_comment: None,
            block_comment_start: Some("<!--".to_string()),
            block_comment_end: Some("-->".to_string()),
            string_delimiters: vec!['"', '\''],
            indent_size: 2,
            use_tabs: false,
            extensions: vec!["html".to_string(), "htm".to_string()],
        }
    }

    /// Get configuration for CSS
    pub fn css() -> Self {
        Self {
            language: "css".to_string(),
            line_comment: None,
            block_comment_start: Some("/*".to_string()),
            block_comment_end: Some("*/".to_string()),
            string_delimiters: vec!['"', '\''],
            indent_size: 2,
            use_tabs: false,
            extensions: vec!["css".to_string(), "scss".to_string(), "less".to_string()],
        }
    }

    /// Get configuration for SQL
    pub fn sql() -> Self {
        Self {
            language: "sql".to_string(),
            line_comment: Some("--".to_string()),
            block_comment_start: Some("/*".to_string()),
            block_comment_end: Some("*/".to_string()),
            string_delimiters: vec!['\''],
            indent_size: 2,
            use_tabs: false,
            extensions: vec!["sql".to_string()],
        }
    }

    /// Get configuration for shell/bash
    pub fn shell() -> Self {
        Self {
            language: "shell".to_string(),
            line_comment: Some("#".to_string()),
            block_comment_start: None,
            block_comment_end: None,
            string_delimiters: vec!['"', '\''],
            indent_size: 2,
            use_tabs: false,
            extensions: vec!["sh".to_string(), "bash".to_string(), "zsh".to_string()],
        }
    }

    /// Plain text (no code features)
    pub fn plain_text() -> Self {
        Self {
            language: "text".to_string(),
            line_comment: None,
            block_comment_start: None,
            block_comment_end: None,
            string_delimiters: vec![],
            indent_size: 4,
            use_tabs: false,
            extensions: vec!["txt".to_string()],
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        let ext = ext.trim_start_matches('.').to_lowercase();
        match ext.as_str() {
            "rs" => Self::rust(),
            "py" | "pyw" => Self::python(),
            "js" | "ts" | "jsx" | "tsx" | "mjs" => Self::javascript(),
            "html" | "htm" | "xml" => Self::html(),
            "css" | "scss" | "sass" | "less" => Self::css(),
            "sql" => Self::sql(),
            "sh" | "bash" | "zsh" | "fish" => Self::shell(),
            _ => Self::plain_text(),
        }
    }

    /// Get the indent string
    pub fn indent_string(&self) -> String {
        if self.use_tabs {
            "\t".to_string()
        } else {
            " ".repeat(self.indent_size)
        }
    }
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self::plain_text()
    }
}

/// A suggested code edit with explanation
#[derive(Debug, Clone)]
pub struct EditSuggestion {
    /// Start line (1-indexed)
    pub start_line: usize,
    /// End line (1-indexed, inclusive)
    pub end_line: usize,
    /// Original code being replaced
    pub original: String,
    /// Suggested replacement
    pub replacement: String,
    /// Explanation of why this edit is suggested
    pub explanation: String,
    /// Category of edit
    pub category: EditCategory,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl EditSuggestion {
    /// Create a new edit suggestion
    pub fn new(
        start_line: usize,
        end_line: usize,
        original: impl Into<String>,
        replacement: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            start_line,
            end_line,
            original: original.into(),
            replacement: replacement.into(),
            explanation: explanation.into(),
            category: EditCategory::Improvement,
            confidence: 0.8,
        }
    }

    /// Set the category
    pub fn with_category(mut self, category: EditCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Format as a displayable string
    pub fn format(&self) -> String {
        format!(
            "Lines {}-{} ({:?}, {:.0}% confidence):\n\
             Explanation: {}\n\
             \n\
             Original:\n{}\n\
             \n\
             Suggested:\n{}",
            self.start_line,
            self.end_line,
            self.category,
            self.confidence * 100.0,
            self.explanation,
            self.original,
            self.replacement
        )
    }
}

/// Category of edit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditCategory {
    /// Bug fix
    BugFix,
    /// Performance improvement
    Performance,
    /// Code style/formatting
    Style,
    /// Simplification/refactoring
    Refactoring,
    /// Security improvement
    Security,
    /// Documentation improvement
    Documentation,
    /// General improvement
    Improvement,
}

/// Code editor with language-aware features
pub struct CodeEditor {
    editor: TextEditor,
    config: LanguageConfig,
}

impl CodeEditor {
    /// Create a new code editor
    pub fn new(code: impl Into<String>, config: LanguageConfig) -> Self {
        Self {
            editor: TextEditor::new(code),
            config,
        }
    }

    /// Create with auto-detected language from extension
    pub fn from_extension(code: impl Into<String>, ext: &str) -> Self {
        Self::new(code, LanguageConfig::from_extension(ext))
    }

    /// Get the current code
    pub fn code(&self) -> &str {
        self.editor.text()
    }

    /// Get the language config
    pub fn config(&self) -> &LanguageConfig {
        &self.config
    }

    /// Comment out a line
    pub fn comment_line(&mut self, line_num: usize) -> Result<(), EditError> {
        if let Some(ref comment) = self.config.line_comment {
            let lines: Vec<&str> = self.editor.text().lines().collect();
            if line_num >= lines.len() {
                return Err(EditError::OutOfBounds {
                    offset: line_num,
                    text_len: lines.len(),
                });
            }

            let line = lines[line_num];
            let new_line = format!("{} {}", comment, line);
            self.editor.replace_line(line_num, new_line)
        } else {
            Ok(())
        }
    }

    /// Uncomment a line
    pub fn uncomment_line(&mut self, line_num: usize) -> Result<(), EditError> {
        if let Some(ref comment) = self.config.line_comment {
            let lines: Vec<&str> = self.editor.text().lines().collect();
            if line_num >= lines.len() {
                return Err(EditError::OutOfBounds {
                    offset: line_num,
                    text_len: lines.len(),
                });
            }

            let line = lines[line_num];
            let trimmed = line.trim_start();

            if trimmed.starts_with(comment) {
                let after_comment = trimmed[comment.len()..].trim_start();
                let leading_ws: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                let new_line = format!("{}{}", leading_ws, after_comment);
                self.editor.replace_line(line_num, new_line)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Toggle comment on a line
    pub fn toggle_comment(&mut self, line_num: usize) -> Result<(), EditError> {
        if let Some(ref comment) = self.config.line_comment {
            let lines: Vec<&str> = self.editor.text().lines().collect();
            if line_num >= lines.len() {
                return Err(EditError::OutOfBounds {
                    offset: line_num,
                    text_len: lines.len(),
                });
            }

            let line = lines[line_num];
            if line.trim_start().starts_with(comment) {
                self.uncomment_line(line_num)
            } else {
                self.comment_line(line_num)
            }
        } else {
            Ok(())
        }
    }

    /// Comment a range of lines
    pub fn comment_lines(&mut self, start_line: usize, end_line: usize) -> Result<(), EditError> {
        // Process in reverse to maintain line numbers
        for line_num in (start_line..=end_line).rev() {
            self.comment_line(line_num)?;
        }
        Ok(())
    }

    /// Uncomment a range of lines
    pub fn uncomment_lines(&mut self, start_line: usize, end_line: usize) -> Result<(), EditError> {
        for line_num in (start_line..=end_line).rev() {
            self.uncomment_line(line_num)?;
        }
        Ok(())
    }

    /// Indent a line
    pub fn indent_line(&mut self, line_num: usize) -> Result<(), EditError> {
        let indent = self.config.indent_string();
        let lines: Vec<&str> = self.editor.text().lines().collect();

        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        let new_line = format!("{}{}", indent, lines[line_num]);
        self.editor.replace_line(line_num, new_line)
    }

    /// Dedent a line
    pub fn dedent_line(&mut self, line_num: usize) -> Result<(), EditError> {
        let lines: Vec<&str> = self.editor.text().lines().collect();

        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        let line = lines[line_num];
        let indent_size = self.config.indent_size;

        // Count leading whitespace
        let leading: String = line.chars().take_while(|c| c.is_whitespace()).collect();
        let to_remove = if self.config.use_tabs {
            if leading.starts_with('\t') { 1 } else { 0 }
        } else {
            leading.chars().take(indent_size).count()
        };

        let new_line = line.chars().skip(to_remove).collect::<String>();
        self.editor.replace_line(line_num, new_line)
    }

    /// Indent a range of lines
    pub fn indent_lines(&mut self, start_line: usize, end_line: usize) -> Result<(), EditError> {
        for line_num in (start_line..=end_line).rev() {
            self.indent_line(line_num)?;
        }
        Ok(())
    }

    /// Dedent a range of lines
    pub fn dedent_lines(&mut self, start_line: usize, end_line: usize) -> Result<(), EditError> {
        for line_num in (start_line..=end_line).rev() {
            self.dedent_line(line_num)?;
        }
        Ok(())
    }

    /// Duplicate a line
    pub fn duplicate_line(&mut self, line_num: usize) -> Result<(), EditError> {
        let lines: Vec<&str> = self.editor.text().lines().collect();

        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        let line_content = lines[line_num].to_string();
        self.editor.insert_line(line_num + 1, line_content)
    }

    /// Move a line up
    pub fn move_line_up(&mut self, line_num: usize) -> Result<(), EditError> {
        if line_num == 0 {
            return Ok(()); // Can't move first line up
        }

        let lines: Vec<&str> = self.editor.text().lines().collect();
        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        // Swap with previous line
        let current = lines[line_num].to_string();
        let previous = lines[line_num - 1].to_string();

        self.editor.replace_line(line_num - 1, current)?;
        self.editor.replace_line(line_num, previous)?;

        Ok(())
    }

    /// Move a line down
    pub fn move_line_down(&mut self, line_num: usize) -> Result<(), EditError> {
        let lines: Vec<&str> = self.editor.text().lines().collect();

        if line_num >= lines.len() - 1 {
            return Ok(()); // Can't move last line down
        }

        // Swap with next line
        let current = lines[line_num].to_string();
        let next = lines[line_num + 1].to_string();

        self.editor.replace_line(line_num, next)?;
        self.editor.replace_line(line_num + 1, current)?;

        Ok(())
    }

    /// Insert a blank line below
    pub fn insert_blank_line_below(&mut self, line_num: usize) -> Result<(), EditError> {
        self.editor.insert_line(line_num + 1, "")
    }

    /// Insert a blank line above
    pub fn insert_blank_line_above(&mut self, line_num: usize) -> Result<(), EditError> {
        self.editor.insert_line(line_num, "")
    }

    /// Get the indentation level of a line
    pub fn get_indentation(&self, line_num: usize) -> usize {
        let lines: Vec<&str> = self.editor.text().lines().collect();
        if line_num >= lines.len() {
            return 0;
        }

        let line = lines[line_num];
        let leading_ws = line.chars().take_while(|c| c.is_whitespace()).count();

        if self.config.use_tabs {
            line.chars().take_while(|c| *c == '\t').count()
        } else {
            leading_ws / self.config.indent_size
        }
    }

    /// Wrap code in a block comment
    pub fn wrap_in_block_comment(&mut self, start_line: usize, end_line: usize) -> Result<(), EditError> {
        let (start_marker, end_marker) = match (&self.config.block_comment_start, &self.config.block_comment_end) {
            (Some(s), Some(e)) => (s.clone(), e.clone()),
            _ => return Ok(()), // No block comments supported
        };

        let lines: Vec<&str> = self.editor.text().lines().collect();
        if end_line >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: end_line,
                text_len: lines.len(),
            });
        }

        // Insert in reverse order to maintain line numbers
        self.editor.insert_line(end_line + 1, end_marker)?;
        self.editor.insert_line(start_line, start_marker)?;

        Ok(())
    }

    /// Apply an edit suggestion
    pub fn apply_suggestion(&mut self, suggestion: &EditSuggestion) -> Result<(), EditError> {
        // Find the offset of start_line
        let mut offset = 0;
        let lines: Vec<&str> = self.editor.text().lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if i == suggestion.start_line - 1 {
                break;
            }
            offset += line.len() + 1; // +1 for newline
        }

        // Calculate end offset
        let mut end_offset = offset;
        for i in (suggestion.start_line - 1)..suggestion.end_line.min(lines.len()) {
            end_offset += lines.get(i).map(|l| l.len() + 1).unwrap_or(0);
        }

        // Remove trailing newline from end_offset calculation if needed
        if end_offset > 0 {
            end_offset -= 1;
        }

        // Apply the replacement
        self.editor.replace_range(offset..end_offset, &suggestion.replacement)
    }

    /// Undo the last edit
    pub fn undo(&mut self) -> Result<(), EditError> {
        self.editor.undo()
    }

    /// Redo the last undone edit
    pub fn redo(&mut self) -> Result<(), EditError> {
        self.editor.redo()
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        self.editor.can_undo()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        self.editor.can_redo()
    }
}

/// Simple search and replace for code
#[derive(Debug)]
pub struct CodeSearch {
    /// Whether to match whole words only
    pub whole_word: bool,
    /// Whether to be case sensitive
    pub case_sensitive: bool,
    /// Whether to use regex
    pub regex: bool,
    /// Scope to search in
    pub scope: SearchScope,
}

/// Scope for code search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScope {
    /// Search everywhere
    All,
    /// Search only in comments
    Comments,
    /// Search only in strings
    Strings,
    /// Search only in code (not comments or strings)
    Code,
}

impl Default for CodeSearch {
    fn default() -> Self {
        Self {
            whole_word: false,
            case_sensitive: true,
            regex: false,
            scope: SearchScope::All,
        }
    }
}

impl CodeSearch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn whole_word(mut self) -> Self {
        self.whole_word = true;
        self
    }

    pub fn case_insensitive(mut self) -> Self {
        self.case_sensitive = false;
        self
    }

    pub fn with_regex(mut self) -> Self {
        self.regex = true;
        self
    }

    pub fn in_scope(mut self, scope: SearchScope) -> Self {
        self.scope = scope;
        self
    }

    /// Find all occurrences in text
    pub fn find_all(&self, text: &str, pattern: &str) -> Vec<(usize, usize)> {
        let mut results = Vec::new();

        if self.regex {
            if let Ok(re) = regex::Regex::new(pattern) {
                for m in re.find_iter(text) {
                    results.push((m.start(), m.end()));
                }
            }
        } else {
            let (search_text, search_pattern) = if self.case_sensitive {
                (text.to_string(), pattern.to_string())
            } else {
                (text.to_lowercase(), pattern.to_lowercase())
            };

            let mut start = 0;
            while let Some(pos) = search_text[start..].find(&search_pattern) {
                let abs_pos = start + pos;

                if self.whole_word {
                    let before_ok = abs_pos == 0 ||
                        !text.chars().nth(abs_pos - 1).map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
                    let after_ok = abs_pos + pattern.len() >= text.len() ||
                        !text.chars().nth(abs_pos + pattern.len()).map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);

                    if before_ok && after_ok {
                        results.push((abs_pos, abs_pos + pattern.len()));
                    }
                } else {
                    results.push((abs_pos, abs_pos + pattern.len()));
                }

                start = abs_pos + 1;
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(LanguageConfig::from_extension("rs").language, "rust");
        assert_eq!(LanguageConfig::from_extension("py").language, "python");
        assert_eq!(LanguageConfig::from_extension("js").language, "javascript");
        assert_eq!(LanguageConfig::from_extension("unknown").language, "text");
    }

    #[test]
    fn test_comment_line() {
        let mut editor = CodeEditor::new("let x = 5;", LanguageConfig::rust());
        editor.comment_line(0).unwrap();
        assert_eq!(editor.code(), "// let x = 5;");
    }

    #[test]
    fn test_uncomment_line() {
        let mut editor = CodeEditor::new("// let x = 5;", LanguageConfig::rust());
        editor.uncomment_line(0).unwrap();
        assert_eq!(editor.code(), "let x = 5;");
    }

    #[test]
    fn test_toggle_comment() {
        let mut editor = CodeEditor::new("let x = 5;", LanguageConfig::rust());

        editor.toggle_comment(0).unwrap();
        assert_eq!(editor.code(), "// let x = 5;");

        editor.toggle_comment(0).unwrap();
        assert_eq!(editor.code(), "let x = 5;");
    }

    #[test]
    fn test_indent_dedent() {
        let mut editor = CodeEditor::new("let x = 5;", LanguageConfig::rust());

        editor.indent_line(0).unwrap();
        assert_eq!(editor.code(), "    let x = 5;");

        editor.dedent_line(0).unwrap();
        assert_eq!(editor.code(), "let x = 5;");
    }

    #[test]
    fn test_duplicate_line() {
        let mut editor = CodeEditor::new("line1\nline2", LanguageConfig::rust());
        editor.duplicate_line(0).unwrap();
        assert_eq!(editor.code(), "line1\nline1\nline2");
    }

    #[test]
    fn test_move_line() {
        let mut editor = CodeEditor::new("line1\nline2\nline3", LanguageConfig::rust());

        editor.move_line_down(0).unwrap();
        assert_eq!(editor.code(), "line2\nline1\nline3");

        editor.move_line_up(1).unwrap();
        assert_eq!(editor.code(), "line1\nline2\nline3");
    }

    #[test]
    fn test_code_search() {
        let text = "foo bar foo baz Foo";

        let search = CodeSearch::new();
        assert_eq!(search.find_all(text, "foo").len(), 2);

        let search = CodeSearch::new().case_insensitive();
        assert_eq!(search.find_all(text, "foo").len(), 3);

        let search = CodeSearch::new().whole_word();
        assert_eq!(search.find_all(text, "foo").len(), 2);
    }

    #[test]
    fn test_edit_suggestion() {
        let suggestion = EditSuggestion::new(
            1, 1,
            "let x = 5",
            "let x: i32 = 5",
            "Add explicit type annotation"
        ).with_category(EditCategory::Style);

        assert!(suggestion.format().contains("Style"));
        assert!(suggestion.format().contains("type annotation"));
    }

    #[test]
    fn test_python_comment() {
        let mut editor = CodeEditor::new("x = 5", LanguageConfig::python());
        editor.comment_line(0).unwrap();
        assert_eq!(editor.code(), "# x = 5");
    }
}
