//! Text transformation utilities
//!
//! This module provides utilities for transforming text content,
//! including find/replace, regex operations, and formatting.
//!
//! # Features
//!
//! - **Find and replace**: Simple and regex-based replacements
//! - **Bulk operations**: Apply multiple transformations efficiently
//! - **Case conversion**: Various case transformation options
//! - **Whitespace handling**: Normalize, trim, indent operations
//! - **Line operations**: Sort, dedupe, filter lines
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::text_transform::{TextTransformer, Transform};
//!
//! let text = "Hello World";
//! let mut transformer = TextTransformer::new(text);
//!
//! transformer.apply(Transform::Replace {
//!     find: "World".to_string(),
//!     replace: "Rust".to_string(),
//!     case_sensitive: true,
//! });
//!
//! assert_eq!(transformer.text(), "Hello Rust");
//! ```

use regex::Regex;
use std::collections::HashSet;

/// A text transformation operation
#[derive(Debug, Clone)]
pub enum Transform {
    /// Simple find and replace
    Replace {
        find: String,
        replace: String,
        case_sensitive: bool,
    },

    /// Regex find and replace
    RegexReplace { pattern: String, replace: String },

    /// Replace all occurrences
    ReplaceAll {
        find: String,
        replace: String,
        case_sensitive: bool,
    },

    /// Convert to uppercase
    ToUpperCase,

    /// Convert to lowercase
    ToLowerCase,

    /// Convert to title case
    ToTitleCase,

    /// Convert to sentence case
    ToSentenceCase,

    /// Trim whitespace from start and end
    Trim,

    /// Trim whitespace from start
    TrimStart,

    /// Trim whitespace from end
    TrimEnd,

    /// Normalize whitespace (collapse multiple spaces)
    NormalizeWhitespace,

    /// Add prefix to each line
    PrefixLines(String),

    /// Add suffix to each line
    SuffixLines(String),

    /// Indent lines by N spaces
    Indent(usize),

    /// Remove N spaces of indentation
    Dedent(usize),

    /// Sort lines alphabetically
    SortLines,

    /// Sort lines in reverse
    SortLinesReverse,

    /// Remove duplicate lines
    DeduplicateLines,

    /// Filter lines matching pattern (keep matches)
    FilterLines(String),

    /// Filter lines matching pattern (remove matches)
    FilterLinesInvert(String),

    /// Remove empty lines
    RemoveEmptyLines,

    /// Remove lines containing text
    RemoveLinesContaining(String),

    /// Keep only lines containing text
    KeepLinesContaining(String),

    /// Wrap lines at column
    WrapLines(usize),

    /// Join lines with separator
    JoinLines(String),

    /// Split into lines by separator
    SplitLines(String),

    /// Reverse line order
    ReverseLines,

    /// Number lines
    NumberLines { start: usize, separator: String },

    /// Remove line numbers
    RemoveLineNumbers,

    /// Convert tabs to spaces
    TabsToSpaces(usize),

    /// Convert spaces to tabs
    SpacesToTabs(usize),

    /// Remove trailing whitespace from lines
    RemoveTrailingWhitespace,

    /// Ensure file ends with newline
    EnsureTrailingNewline,

    /// Remove all blank lines at end
    TrimTrailingBlankLines,

    /// Custom transformation function (for advanced use)
    Custom(String),
}

impl Transform {
    /// Get a human-readable description of this transform
    pub fn description(&self) -> String {
        match self {
            Self::Replace { find, replace, .. } => format!("Replace '{}' with '{}'", find, replace),
            Self::RegexReplace { pattern, replace } => {
                format!("Regex replace '{}' with '{}'", pattern, replace)
            }
            Self::ReplaceAll { find, replace, .. } => {
                format!("Replace all '{}' with '{}'", find, replace)
            }
            Self::ToUpperCase => "Convert to uppercase".to_string(),
            Self::ToLowerCase => "Convert to lowercase".to_string(),
            Self::ToTitleCase => "Convert to title case".to_string(),
            Self::ToSentenceCase => "Convert to sentence case".to_string(),
            Self::Trim => "Trim whitespace".to_string(),
            Self::TrimStart => "Trim leading whitespace".to_string(),
            Self::TrimEnd => "Trim trailing whitespace".to_string(),
            Self::NormalizeWhitespace => "Normalize whitespace".to_string(),
            Self::PrefixLines(p) => format!("Prefix lines with '{}'", p),
            Self::SuffixLines(s) => format!("Suffix lines with '{}'", s),
            Self::Indent(n) => format!("Indent by {} spaces", n),
            Self::Dedent(n) => format!("Remove {} spaces indent", n),
            Self::SortLines => "Sort lines".to_string(),
            Self::SortLinesReverse => "Sort lines (reverse)".to_string(),
            Self::DeduplicateLines => "Remove duplicate lines".to_string(),
            Self::FilterLines(p) => format!("Keep lines matching '{}'", p),
            Self::FilterLinesInvert(p) => format!("Remove lines matching '{}'", p),
            Self::RemoveEmptyLines => "Remove empty lines".to_string(),
            Self::RemoveLinesContaining(s) => format!("Remove lines containing '{}'", s),
            Self::KeepLinesContaining(s) => format!("Keep lines containing '{}'", s),
            Self::WrapLines(w) => format!("Wrap lines at column {}", w),
            Self::JoinLines(s) => format!("Join lines with '{}'", s),
            Self::SplitLines(s) => format!("Split by '{}'", s),
            Self::ReverseLines => "Reverse line order".to_string(),
            Self::NumberLines { start, separator } => {
                format!("Number lines from {} with '{}'", start, separator)
            }
            Self::RemoveLineNumbers => "Remove line numbers".to_string(),
            Self::TabsToSpaces(n) => format!("Convert tabs to {} spaces", n),
            Self::SpacesToTabs(n) => format!("Convert {} spaces to tabs", n),
            Self::RemoveTrailingWhitespace => "Remove trailing whitespace".to_string(),
            Self::EnsureTrailingNewline => "Ensure trailing newline".to_string(),
            Self::TrimTrailingBlankLines => "Trim trailing blank lines".to_string(),
            Self::Custom(desc) => desc.clone(),
        }
    }
}

/// Result of a transformation
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// The transformed text
    pub text: String,
    /// Number of changes made
    pub changes: usize,
    /// Description of what was done
    pub description: String,
}

/// Text transformer for applying transformations
pub struct TextTransformer {
    text: String,
    history: Vec<String>,
}

impl TextTransformer {
    /// Create a new transformer with the given text
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            history: Vec::new(),
        }
    }

    /// Get the current text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Take ownership of the text
    pub fn into_text(self) -> String {
        self.text
    }

    /// Apply a transformation
    pub fn apply(&mut self, transform: Transform) -> TransformResult {
        self.history.push(self.text.clone());
        let original_len = self.text.len();

        match transform {
            Transform::Replace {
                find,
                replace,
                case_sensitive,
            } => self.replace_first(&find, &replace, case_sensitive),
            Transform::RegexReplace { pattern, replace } => self.regex_replace(&pattern, &replace),
            Transform::ReplaceAll {
                find,
                replace,
                case_sensitive,
            } => self.replace_all(&find, &replace, case_sensitive),
            Transform::ToUpperCase => {
                self.text = self.text.to_uppercase();
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Converted to uppercase".to_string(),
                }
            }
            Transform::ToLowerCase => {
                self.text = self.text.to_lowercase();
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Converted to lowercase".to_string(),
                }
            }
            Transform::ToTitleCase => {
                self.text = self.to_title_case(&self.text);
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Converted to title case".to_string(),
                }
            }
            Transform::ToSentenceCase => {
                self.text = self.to_sentence_case(&self.text);
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Converted to sentence case".to_string(),
                }
            }
            Transform::Trim => {
                self.text = self.text.trim().to_string();
                let changed = self.text.len() != original_len;
                TransformResult {
                    text: self.text.clone(),
                    changes: if changed { 1 } else { 0 },
                    description: "Trimmed whitespace".to_string(),
                }
            }
            Transform::TrimStart => {
                self.text = self.text.trim_start().to_string();
                let changed = self.text.len() != original_len;
                TransformResult {
                    text: self.text.clone(),
                    changes: if changed { 1 } else { 0 },
                    description: "Trimmed leading whitespace".to_string(),
                }
            }
            Transform::TrimEnd => {
                self.text = self.text.trim_end().to_string();
                let changed = self.text.len() != original_len;
                TransformResult {
                    text: self.text.clone(),
                    changes: if changed { 1 } else { 0 },
                    description: "Trimmed trailing whitespace".to_string(),
                }
            }
            Transform::NormalizeWhitespace => {
                let re = Regex::new(r"\s+").expect("valid regex");
                self.text = re.replace_all(&self.text, " ").to_string();
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Normalized whitespace".to_string(),
                }
            }
            Transform::PrefixLines(prefix) => {
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .map(|l| format!("{}{}", prefix, l))
                    .collect();
                let count = lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Prefixed {} lines", count),
                }
            }
            Transform::SuffixLines(suffix) => {
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .map(|l| format!("{}{}", l, suffix))
                    .collect();
                let count = lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Suffixed {} lines", count),
                }
            }
            Transform::Indent(spaces) => {
                let indent = " ".repeat(spaces);
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .map(|l| {
                        if l.is_empty() {
                            l.to_string()
                        } else {
                            format!("{}{}", indent, l)
                        }
                    })
                    .collect();
                let count = lines.iter().filter(|l| !l.is_empty()).count();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Indented {} lines by {} spaces", count, spaces),
                }
            }
            Transform::Dedent(spaces) => {
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .map(|l| {
                        let trimmed = l
                            .chars()
                            .skip_while(|c| c.is_whitespace())
                            .collect::<String>();
                        let leading: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                        let to_remove = leading.len().min(spaces);
                        format!("{}{}", &leading[to_remove..], trimmed)
                    })
                    .collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: format!("Removed {} spaces of indent", spaces),
                }
            }
            Transform::SortLines => {
                let mut lines: Vec<_> = self.text.lines().collect();
                lines.sort();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Sorted lines".to_string(),
                }
            }
            Transform::SortLinesReverse => {
                let mut lines: Vec<_> = self.text.lines().collect();
                lines.sort();
                lines.reverse();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Sorted lines (reverse)".to_string(),
                }
            }
            Transform::DeduplicateLines => {
                let mut seen = HashSet::new();
                let lines: Vec<_> = self.text.lines().filter(|l| seen.insert(*l)).collect();
                let removed = self.text.lines().count() - lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: removed,
                    description: format!("Removed {} duplicate lines", removed),
                }
            }
            Transform::FilterLines(pattern) => {
                if let Ok(re) = Regex::new(&pattern) {
                    let lines: Vec<_> = self.text.lines().filter(|l| re.is_match(l)).collect();
                    let kept = lines.len();
                    self.text = lines.join("\n");
                    TransformResult {
                        text: self.text.clone(),
                        changes: kept,
                        description: format!("Kept {} matching lines", kept),
                    }
                } else {
                    TransformResult {
                        text: self.text.clone(),
                        changes: 0,
                        description: "Invalid regex pattern".to_string(),
                    }
                }
            }
            Transform::FilterLinesInvert(pattern) => {
                if let Ok(re) = Regex::new(&pattern) {
                    let original_count = self.text.lines().count();
                    let lines: Vec<_> = self.text.lines().filter(|l| !re.is_match(l)).collect();
                    let removed = original_count - lines.len();
                    self.text = lines.join("\n");
                    TransformResult {
                        text: self.text.clone(),
                        changes: removed,
                        description: format!("Removed {} matching lines", removed),
                    }
                } else {
                    TransformResult {
                        text: self.text.clone(),
                        changes: 0,
                        description: "Invalid regex pattern".to_string(),
                    }
                }
            }
            Transform::RemoveEmptyLines => {
                let original_count = self.text.lines().count();
                let lines: Vec<_> = self.text.lines().filter(|l| !l.trim().is_empty()).collect();
                let removed = original_count - lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: removed,
                    description: format!("Removed {} empty lines", removed),
                }
            }
            Transform::RemoveLinesContaining(needle) => {
                let original_count = self.text.lines().count();
                let lines: Vec<_> = self.text.lines().filter(|l| !l.contains(&needle)).collect();
                let removed = original_count - lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: removed,
                    description: format!("Removed {} lines containing '{}'", removed, needle),
                }
            }
            Transform::KeepLinesContaining(needle) => {
                let lines: Vec<_> = self.text.lines().filter(|l| l.contains(&needle)).collect();
                let kept = lines.len();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: kept,
                    description: format!("Kept {} lines containing '{}'", kept, needle),
                }
            }
            Transform::WrapLines(width) => {
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .flat_map(|l| self.wrap_line(l, width))
                    .collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: format!("Wrapped lines at column {}", width),
                }
            }
            Transform::JoinLines(separator) => {
                self.text = self.text.lines().collect::<Vec<_>>().join(&separator);
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Joined lines".to_string(),
                }
            }
            Transform::SplitLines(separator) => {
                self.text = self.text.split(&separator).collect::<Vec<_>>().join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Split into lines".to_string(),
                }
            }
            Transform::ReverseLines => {
                let lines: Vec<_> = self.text.lines().rev().collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Reversed line order".to_string(),
                }
            }
            Transform::NumberLines { start, separator } => {
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .enumerate()
                    .map(|(i, l)| format!("{}{}{}", start + i, separator, l))
                    .collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Added line numbers".to_string(),
                }
            }
            Transform::RemoveLineNumbers => {
                let re = Regex::new(r"^\s*\d+[.:\-)\]\s]+").expect("valid regex");
                let lines: Vec<_> = self
                    .text
                    .lines()
                    .map(|l| re.replace(l, "").to_string())
                    .collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Removed line numbers".to_string(),
                }
            }
            Transform::TabsToSpaces(n) => {
                let spaces = " ".repeat(n);
                let count = self.text.matches('\t').count();
                self.text = self.text.replace('\t', &spaces);
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Converted {} tabs to spaces", count),
                }
            }
            Transform::SpacesToTabs(n) => {
                let spaces = " ".repeat(n);
                let count = self.text.matches(&spaces).count();
                self.text = self.text.replace(&spaces, "\t");
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Converted {} space groups to tabs", count),
                }
            }
            Transform::RemoveTrailingWhitespace => {
                let lines: Vec<_> = self.text.lines().map(|l| l.trim_end()).collect();
                self.text = lines.join("\n");
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Removed trailing whitespace".to_string(),
                }
            }
            Transform::EnsureTrailingNewline => {
                if !self.text.ends_with('\n') {
                    self.text.push('\n');
                    TransformResult {
                        text: self.text.clone(),
                        changes: 1,
                        description: "Added trailing newline".to_string(),
                    }
                } else {
                    TransformResult {
                        text: self.text.clone(),
                        changes: 0,
                        description: "Already has trailing newline".to_string(),
                    }
                }
            }
            Transform::TrimTrailingBlankLines => {
                while self.text.ends_with("\n\n") {
                    self.text.pop();
                }
                TransformResult {
                    text: self.text.clone(),
                    changes: 1,
                    description: "Trimmed trailing blank lines".to_string(),
                }
            }
            Transform::Custom(desc) => TransformResult {
                text: self.text.clone(),
                changes: 0,
                description: format!("Custom: {}", desc),
            },
        }
    }

    /// Apply multiple transformations in sequence
    pub fn apply_all(&mut self, transforms: &[Transform]) -> Vec<TransformResult> {
        transforms.iter().map(|t| self.apply(t.clone())).collect()
    }

    /// Undo the last transformation
    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.history.pop() {
            self.text = prev;
            true
        } else {
            false
        }
    }

    fn replace_first(
        &mut self,
        find: &str,
        replace: &str,
        case_sensitive: bool,
    ) -> TransformResult {
        let (new_text, count) = if case_sensitive {
            if let Some(pos) = self.text.find(find) {
                let mut new = self.text[..pos].to_string();
                new.push_str(replace);
                new.push_str(&self.text[pos + find.len()..]);
                (new, 1)
            } else {
                (self.text.clone(), 0)
            }
        } else {
            let lower = self.text.to_lowercase();
            let find_lower = find.to_lowercase();
            if let Some(pos) = lower.find(&find_lower) {
                let mut new = self.text[..pos].to_string();
                new.push_str(replace);
                new.push_str(&self.text[pos + find.len()..]);
                (new, 1)
            } else {
                (self.text.clone(), 0)
            }
        };

        self.text = new_text;
        TransformResult {
            text: self.text.clone(),
            changes: count,
            description: format!("Replaced {} occurrence(s)", count),
        }
    }

    fn replace_all(&mut self, find: &str, replace: &str, case_sensitive: bool) -> TransformResult {
        let count = if case_sensitive {
            self.text.matches(find).count()
        } else {
            self.text
                .to_lowercase()
                .matches(&find.to_lowercase())
                .count()
        };

        self.text = if case_sensitive {
            self.text.replace(find, replace)
        } else {
            // Use (?i) flag for case-insensitive matching
            let pattern = format!("(?i){}", regex::escape(find));
            let re = Regex::new(&pattern).expect("valid regex");
            re.replace_all(&self.text, replace).to_string()
        };

        TransformResult {
            text: self.text.clone(),
            changes: count,
            description: format!("Replaced {} occurrence(s)", count),
        }
    }

    fn regex_replace(&mut self, pattern: &str, replace: &str) -> TransformResult {
        match Regex::new(pattern) {
            Ok(re) => {
                let count = re.find_iter(&self.text).count();
                self.text = re.replace_all(&self.text, replace).to_string();
                TransformResult {
                    text: self.text.clone(),
                    changes: count,
                    description: format!("Regex replaced {} match(es)", count),
                }
            }
            Err(e) => TransformResult {
                text: self.text.clone(),
                changes: 0,
                description: format!("Invalid regex: {}", e),
            },
        }
    }

    fn to_title_case(&self, s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first
                        .to_uppercase()
                        .chain(chars.flat_map(|c| c.to_lowercase()))
                        .collect(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn to_sentence_case(&self, s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for c in s.chars() {
            if capitalize_next && c.is_alphabetic() {
                result.extend(c.to_uppercase());
                capitalize_next = false;
            } else {
                result.extend(c.to_lowercase());
            }

            if c == '.' || c == '!' || c == '?' {
                capitalize_next = true;
            }
        }

        result
    }

    fn wrap_line(&self, line: &str, width: usize) -> Vec<String> {
        if line.len() <= width {
            return vec![line.to_string()];
        }

        let mut result = Vec::new();
        let mut current = String::new();

        for word in line.split_whitespace() {
            if current.is_empty() {
                current = word.to_string();
            } else if current.len() + 1 + word.len() <= width {
                current.push(' ');
                current.push_str(word);
            } else {
                result.push(current);
                current = word.to_string();
            }
        }

        if !current.is_empty() {
            result.push(current);
        }

        result
    }
}

/// Builder for creating transformation pipelines
#[derive(Default)]
pub struct TransformPipeline {
    transforms: Vec<Transform>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, transform: Transform) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn replace(self, find: &str, replace: &str) -> Self {
        self.add(Transform::ReplaceAll {
            find: find.to_string(),
            replace: replace.to_string(),
            case_sensitive: true,
        })
    }

    pub fn regex(self, pattern: &str, replace: &str) -> Self {
        self.add(Transform::RegexReplace {
            pattern: pattern.to_string(),
            replace: replace.to_string(),
        })
    }

    pub fn trim(self) -> Self {
        self.add(Transform::Trim)
    }

    pub fn uppercase(self) -> Self {
        self.add(Transform::ToUpperCase)
    }

    pub fn lowercase(self) -> Self {
        self.add(Transform::ToLowerCase)
    }

    pub fn indent(self, spaces: usize) -> Self {
        self.add(Transform::Indent(spaces))
    }

    pub fn sort_lines(self) -> Self {
        self.add(Transform::SortLines)
    }

    pub fn dedupe(self) -> Self {
        self.add(Transform::DeduplicateLines)
    }

    pub fn apply(&self, text: &str) -> String {
        let mut transformer = TextTransformer::new(text);
        for t in &self.transforms {
            transformer.apply(t.clone());
        }
        transformer.into_text()
    }

    pub fn apply_with_results(&self, text: &str) -> (String, Vec<TransformResult>) {
        let mut transformer = TextTransformer::new(text);
        let results = transformer.apply_all(&self.transforms);
        (transformer.into_text(), results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace() {
        let mut t = TextTransformer::new("Hello World");
        t.apply(Transform::ReplaceAll {
            find: "World".to_string(),
            replace: "Rust".to_string(),
            case_sensitive: true,
        });
        assert_eq!(t.text(), "Hello Rust");
    }

    #[test]
    fn test_case_insensitive_replace() {
        let mut t = TextTransformer::new("Hello WORLD");
        t.apply(Transform::ReplaceAll {
            find: "world".to_string(),
            replace: "Rust".to_string(),
            case_sensitive: false,
        });
        assert_eq!(t.text(), "Hello Rust");
    }

    #[test]
    fn test_regex_replace() {
        let mut t = TextTransformer::new("foo123bar456");
        t.apply(Transform::RegexReplace {
            pattern: r"\d+".to_string(),
            replace: "X".to_string(),
        });
        assert_eq!(t.text(), "fooXbarX");
    }

    #[test]
    fn test_title_case() {
        let mut t = TextTransformer::new("hello world");
        t.apply(Transform::ToTitleCase);
        assert_eq!(t.text(), "Hello World");
    }

    #[test]
    fn test_sort_lines() {
        let mut t = TextTransformer::new("banana\napple\ncherry");
        t.apply(Transform::SortLines);
        assert_eq!(t.text(), "apple\nbanana\ncherry");
    }

    #[test]
    fn test_deduplicate() {
        let mut t = TextTransformer::new("a\nb\na\nc\nb");
        t.apply(Transform::DeduplicateLines);
        assert_eq!(t.text(), "a\nb\nc");
    }

    #[test]
    fn test_indent() {
        let mut t = TextTransformer::new("line1\nline2");
        t.apply(Transform::Indent(4));
        assert_eq!(t.text(), "    line1\n    line2");
    }

    #[test]
    fn test_filter_lines() {
        let mut t = TextTransformer::new("foo\nbar\nfoobar\nbaz");
        t.apply(Transform::FilterLines("foo".to_string()));
        assert_eq!(t.text(), "foo\nfoobar");
    }

    #[test]
    fn test_number_lines() {
        let mut t = TextTransformer::new("a\nb\nc");
        t.apply(Transform::NumberLines {
            start: 1,
            separator: ": ".to_string(),
        });
        assert_eq!(t.text(), "1: a\n2: b\n3: c");
    }

    #[test]
    fn test_pipeline() {
        let result = TransformPipeline::new()
            .trim()
            .lowercase()
            .replace("world", "rust")
            .apply("  Hello World  ");
        assert_eq!(result, "hello rust");
    }

    #[test]
    fn test_undo() {
        let mut t = TextTransformer::new("Hello");
        t.apply(Transform::ToUpperCase);
        assert_eq!(t.text(), "HELLO");

        t.undo();
        assert_eq!(t.text(), "Hello");
    }

    #[test]
    fn test_wrap_lines() {
        let mut t = TextTransformer::new("This is a very long line that should be wrapped");
        t.apply(Transform::WrapLines(20));
        let lines: Vec<_> = t.text().lines().collect();
        assert!(lines.iter().all(|l| l.len() <= 20));
    }
}
