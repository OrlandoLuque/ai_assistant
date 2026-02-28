//! Patch application for applying diffs to text
//!
//! This module provides functionality to parse and apply unified diffs
//! (git-style patches) to text content.
//!
//! # Features
//!
//! - **Unified diff parsing**: Parse standard unified diff format
//! - **Patch application**: Apply patches with context validation
//! - **Fuzzy matching**: Apply patches even when context is slightly off
//! - **Conflict detection**: Detect when patches cannot be applied cleanly
//! - **Reverse patches**: Generate and apply reverse patches
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::patch_application::{Patch, PatchApplicator};
//!
//! let original = "Line 1\nLine 2\nLine 3";
//! let patch_text = r#"@@ -1,3 +1,3 @@
//!  Line 1
//! -Line 2
//! +Modified Line 2
//!  Line 3"#;
//!
//! let patch = Patch::parse(patch_text).unwrap();
//! let applicator = PatchApplicator::new();
//! let result = applicator.apply(&original, &patch).unwrap();
//! assert_eq!(result, "Line 1\nModified Line 2\nLine 3");
//! ```

use std::fmt;

/// A single line in a patch hunk
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchLine {
    /// Context line (unchanged)
    Context(String),
    /// Line to add
    Add(String),
    /// Line to remove
    Remove(String),
}

impl PatchLine {
    /// Get the line content without the prefix
    pub fn content(&self) -> &str {
        match self {
            PatchLine::Context(s) | PatchLine::Add(s) | PatchLine::Remove(s) => s,
        }
    }

    /// Check if this is a context line
    pub fn is_context(&self) -> bool {
        matches!(self, PatchLine::Context(_))
    }

    /// Check if this is an add line
    pub fn is_add(&self) -> bool {
        matches!(self, PatchLine::Add(_))
    }

    /// Check if this is a remove line
    pub fn is_remove(&self) -> bool {
        matches!(self, PatchLine::Remove(_))
    }
}

/// A hunk in a patch (one contiguous set of changes)
#[derive(Debug, Clone)]
pub struct PatchHunk {
    /// Original file start line (1-indexed)
    pub old_start: usize,
    /// Original file line count
    pub old_count: usize,
    /// New file start line (1-indexed)
    pub new_start: usize,
    /// New file line count
    pub new_count: usize,
    /// Lines in this hunk
    pub lines: Vec<PatchLine>,
}

impl PatchHunk {
    /// Create a new empty hunk
    pub fn new(old_start: usize, old_count: usize, new_start: usize, new_count: usize) -> Self {
        Self {
            old_start,
            old_count,
            new_start,
            new_count,
            lines: Vec::new(),
        }
    }

    /// Get context lines before the changes
    pub fn leading_context(&self) -> Vec<&str> {
        self.lines
            .iter()
            .take_while(|l| l.is_context())
            .map(|l| l.content())
            .collect()
    }

    /// Get context lines after the changes
    pub fn trailing_context(&self) -> Vec<&str> {
        self.lines
            .iter()
            .rev()
            .take_while(|l| l.is_context())
            .map(|l| l.content())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Get lines to remove
    pub fn removals(&self) -> Vec<&str> {
        self.lines
            .iter()
            .filter(|l| l.is_remove())
            .map(|l| l.content())
            .collect()
    }

    /// Get lines to add
    pub fn additions(&self) -> Vec<&str> {
        self.lines
            .iter()
            .filter(|l| l.is_add())
            .map(|l| l.content())
            .collect()
    }

    /// Reverse this hunk (swap additions and removals)
    pub fn reverse(&self) -> Self {
        let lines = self
            .lines
            .iter()
            .map(|l| match l {
                PatchLine::Context(s) => PatchLine::Context(s.clone()),
                PatchLine::Add(s) => PatchLine::Remove(s.clone()),
                PatchLine::Remove(s) => PatchLine::Add(s.clone()),
            })
            .collect();

        Self {
            old_start: self.new_start,
            old_count: self.new_count,
            new_start: self.old_start,
            new_count: self.old_count,
            lines,
        }
    }
}

/// A complete patch (may contain multiple hunks)
#[derive(Debug, Clone)]
pub struct Patch {
    /// Original file name
    pub old_file: Option<String>,
    /// New file name
    pub new_file: Option<String>,
    /// Hunks in this patch
    pub hunks: Vec<PatchHunk>,
}

impl Patch {
    /// Create an empty patch
    pub fn new() -> Self {
        Self {
            old_file: None,
            new_file: None,
            hunks: Vec::new(),
        }
    }

    /// Parse a unified diff string into a Patch
    pub fn parse(diff: &str) -> Result<Self, PatchParseError> {
        let mut patch = Patch::new();
        let mut current_hunk: Option<PatchHunk> = None;
        let lines: Vec<&str> = diff.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];

            // Parse file headers
            if line.starts_with("--- ") {
                patch.old_file = Some(line[4..].trim().to_string());
                i += 1;
                continue;
            }

            if line.starts_with("+++ ") {
                patch.new_file = Some(line[4..].trim().to_string());
                i += 1;
                continue;
            }

            // Parse hunk header
            if line.starts_with("@@") {
                // Save previous hunk
                if let Some(hunk) = current_hunk.take() {
                    patch.hunks.push(hunk);
                }

                // Parse: @@ -old_start,old_count +new_start,new_count @@
                let hunk = Self::parse_hunk_header(line)?;
                current_hunk = Some(hunk);
                i += 1;
                continue;
            }

            // Parse hunk content
            if let Some(ref mut hunk) = current_hunk {
                if line.starts_with(' ') || line.is_empty() {
                    let content = if line.is_empty() { "" } else { &line[1..] };
                    hunk.lines.push(PatchLine::Context(content.to_string()));
                } else if line.starts_with('+') {
                    hunk.lines.push(PatchLine::Add(line[1..].to_string()));
                } else if line.starts_with('-') {
                    hunk.lines.push(PatchLine::Remove(line[1..].to_string()));
                } else if line.starts_with('\\') {
                    // "\ No newline at end of file" - skip
                } else {
                    // Treat as context if no prefix (some diffs omit space for context)
                    hunk.lines.push(PatchLine::Context(line.to_string()));
                }
            }

            i += 1;
        }

        // Save last hunk
        if let Some(hunk) = current_hunk {
            patch.hunks.push(hunk);
        }

        if patch.hunks.is_empty() {
            return Err(PatchParseError::NoHunks);
        }

        Ok(patch)
    }

    fn parse_hunk_header(line: &str) -> Result<PatchHunk, PatchParseError> {
        // Format: @@ -old_start,old_count +new_start,new_count @@ optional_context
        let line = line.trim_start_matches('@').trim();

        // Find the closing @@
        let end_marker = line.find("@@").unwrap_or(line.len());
        let range_part = &line[..end_marker].trim();

        let parts: Vec<&str> = range_part.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(PatchParseError::InvalidHunkHeader(line.to_string()));
        }

        let old_range = Self::parse_range(parts[0].trim_start_matches('-'))?;
        let new_range = Self::parse_range(parts[1].trim_start_matches('+'))?;

        Ok(PatchHunk::new(
            old_range.0,
            old_range.1,
            new_range.0,
            new_range.1,
        ))
    }

    fn parse_range(s: &str) -> Result<(usize, usize), PatchParseError> {
        if let Some(comma_pos) = s.find(',') {
            let start: usize = s[..comma_pos]
                .parse()
                .map_err(|_| PatchParseError::InvalidRange(s.to_string()))?;
            let count: usize = s[comma_pos + 1..]
                .parse()
                .map_err(|_| PatchParseError::InvalidRange(s.to_string()))?;
            Ok((start, count))
        } else {
            let start: usize = s
                .parse()
                .map_err(|_| PatchParseError::InvalidRange(s.to_string()))?;
            Ok((start, 1))
        }
    }

    /// Create a simple patch from old and new text
    pub fn from_texts(old: &str, new: &str) -> Self {
        use crate::diff::{diff, ChangeType};

        let diff_result = diff(old, new);
        let mut patch = Patch::new();

        // Convert each DiffHunk to a PatchHunk
        for diff_hunk in &diff_result.hunks {
            let mut patch_hunk = PatchHunk::new(
                diff_hunk.old_start,
                diff_hunk.old_count,
                diff_hunk.new_start,
                diff_hunk.new_count,
            );

            for diff_line in &diff_hunk.lines {
                match diff_line.change_type {
                    ChangeType::Equal => {
                        patch_hunk
                            .lines
                            .push(PatchLine::Context(diff_line.content.clone()));
                    }
                    ChangeType::Added => {
                        patch_hunk
                            .lines
                            .push(PatchLine::Add(diff_line.content.clone()));
                    }
                    ChangeType::Removed => {
                        patch_hunk
                            .lines
                            .push(PatchLine::Remove(diff_line.content.clone()));
                    }
                    ChangeType::Modified => {
                        // Modified lines appear as remove + add in unified diff format
                        patch_hunk
                            .lines
                            .push(PatchLine::Remove(diff_line.content.clone()));
                    }
                }
            }

            if !patch_hunk.lines.is_empty() {
                patch.hunks.push(patch_hunk);
            }
        }

        patch
    }

    /// Reverse this patch (swap additions and removals)
    pub fn reverse(&self) -> Self {
        Patch {
            old_file: self.new_file.clone(),
            new_file: self.old_file.clone(),
            hunks: self.hunks.iter().map(|h| h.reverse()).collect(),
        }
    }

    /// Format as unified diff string
    pub fn to_unified(&self) -> String {
        let mut result = String::new();

        if let Some(ref old) = self.old_file {
            result.push_str(&format!("--- {}\n", old));
        }
        if let Some(ref new) = self.new_file {
            result.push_str(&format!("+++ {}\n", new));
        }

        for hunk in &self.hunks {
            result.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            ));

            for line in &hunk.lines {
                match line {
                    PatchLine::Context(s) => result.push_str(&format!(" {}\n", s)),
                    PatchLine::Add(s) => result.push_str(&format!("+{}\n", s)),
                    PatchLine::Remove(s) => result.push_str(&format!("-{}\n", s)),
                }
            }
        }

        result
    }
}

impl Default for Patch {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_unified())
    }
}

/// Error during patch parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchParseError {
    /// No hunks found in patch
    NoHunks,
    /// Invalid hunk header
    InvalidHunkHeader(String),
    /// Invalid range in hunk header
    InvalidRange(String),
}

impl fmt::Display for PatchParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoHunks => write!(f, "No hunks found in patch"),
            Self::InvalidHunkHeader(h) => write!(f, "Invalid hunk header: {}", h),
            Self::InvalidRange(r) => write!(f, "Invalid range: {}", r),
        }
    }
}

impl std::error::Error for PatchParseError {}

/// Error during patch application
#[derive(Debug, Clone)]
pub enum PatchApplyError {
    /// Context doesn't match
    ContextMismatch {
        hunk: usize,
        expected: String,
        found: String,
        line: usize,
    },
    /// Line to remove not found
    RemovalMismatch {
        hunk: usize,
        expected: String,
        found: String,
        line: usize,
    },
    /// Hunk offset is out of bounds
    OutOfBounds {
        hunk: usize,
        line: usize,
        file_lines: usize,
    },
    /// Already applied
    AlreadyApplied { hunk: usize },
}

impl fmt::Display for PatchApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch {
                hunk,
                expected,
                found,
                line,
            } => {
                write!(
                    f,
                    "Hunk {} context mismatch at line {}: expected '{}', found '{}'",
                    hunk, line, expected, found
                )
            }
            Self::RemovalMismatch {
                hunk,
                expected,
                found,
                line,
            } => {
                write!(
                    f,
                    "Hunk {} removal mismatch at line {}: expected '{}', found '{}'",
                    hunk, line, expected, found
                )
            }
            Self::OutOfBounds {
                hunk,
                line,
                file_lines,
            } => {
                write!(
                    f,
                    "Hunk {} line {} out of bounds (file has {} lines)",
                    hunk, line, file_lines
                )
            }
            Self::AlreadyApplied { hunk } => {
                write!(f, "Hunk {} appears to be already applied", hunk)
            }
        }
    }
}

impl std::error::Error for PatchApplyError {}

/// Result of patch application
#[derive(Debug, Clone)]
pub struct PatchResult {
    /// The resulting text
    pub text: String,
    /// Number of hunks applied successfully
    pub hunks_applied: usize,
    /// Number of hunks that needed offset adjustment
    pub hunks_offset: usize,
    /// Number of hunks that failed
    pub hunks_failed: usize,
    /// Warnings generated during application
    pub warnings: Vec<String>,
}

/// Configuration for patch application
#[derive(Debug, Clone)]
pub struct PatchConfig {
    /// Allow fuzzy matching (ignore whitespace differences)
    pub fuzzy: bool,
    /// Maximum lines to search for context
    pub fuzz_lines: usize,
    /// Ignore whitespace in context matching
    pub ignore_whitespace: bool,
    /// Continue on hunk failure
    pub continue_on_error: bool,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            fuzzy: true,
            fuzz_lines: 3,
            ignore_whitespace: false,
            continue_on_error: false,
        }
    }
}

/// Applies patches to text
pub struct PatchApplicator {
    config: PatchConfig,
}

impl PatchApplicator {
    pub fn new() -> Self {
        Self {
            config: PatchConfig::default(),
        }
    }

    pub fn with_config(config: PatchConfig) -> Self {
        Self { config }
    }

    /// Apply a patch to text
    pub fn apply(&self, text: &str, patch: &Patch) -> Result<String, PatchApplyError> {
        let result = self.apply_with_result(text, patch)?;
        Ok(result.text)
    }

    /// Apply a patch and return detailed result
    pub fn apply_with_result(
        &self,
        text: &str,
        patch: &Patch,
    ) -> Result<PatchResult, PatchApplyError> {
        let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let mut offset: isize = 0;
        let mut hunks_applied = 0;
        let mut hunks_offset = 0;
        let mut hunks_failed = 0;
        let mut warnings = Vec::new();

        for (hunk_idx, hunk) in patch.hunks.iter().enumerate() {
            let target_line = ((hunk.old_start as isize - 1) + offset) as usize;

            match self.apply_hunk(&mut lines, hunk, target_line) {
                Ok((applied_at, line_delta)) => {
                    if applied_at != target_line {
                        hunks_offset += 1;
                        warnings.push(format!(
                            "Hunk {} applied with offset {} (at line {} instead of {})",
                            hunk_idx + 1,
                            applied_at as isize - target_line as isize,
                            applied_at + 1,
                            target_line + 1
                        ));
                    }
                    offset += line_delta;
                    hunks_applied += 1;
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        hunks_failed += 1;
                        warnings.push(format!("Hunk {} failed: {}", hunk_idx + 1, e));
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(PatchResult {
            text: lines.join("\n"),
            hunks_applied,
            hunks_offset,
            hunks_failed,
            warnings,
        })
    }

    fn apply_hunk(
        &self,
        lines: &mut Vec<String>,
        hunk: &PatchHunk,
        target_line: usize,
    ) -> Result<(usize, isize), PatchApplyError> {
        // Try exact match first
        if let Ok(delta) = self.try_apply_at(lines, hunk, target_line) {
            return Ok((target_line, delta));
        }

        // Try fuzzy matching
        if self.config.fuzzy {
            for offset in 1..=self.config.fuzz_lines {
                // Try after
                if target_line + offset < lines.len() {
                    if let Ok(delta) = self.try_apply_at(lines, hunk, target_line + offset) {
                        return Ok((target_line + offset, delta));
                    }
                }
                // Try before
                if target_line >= offset {
                    if let Ok(delta) = self.try_apply_at(lines, hunk, target_line - offset) {
                        return Ok((target_line - offset, delta));
                    }
                }
            }
        }

        Err(PatchApplyError::ContextMismatch {
            hunk: 0,
            expected: hunk.leading_context().join("\n"),
            found: lines.get(target_line).cloned().unwrap_or_default(),
            line: target_line,
        })
    }

    fn try_apply_at(
        &self,
        lines: &mut Vec<String>,
        hunk: &PatchHunk,
        start_line: usize,
    ) -> Result<isize, PatchApplyError> {
        let mut line_idx = start_line;
        let mut new_lines: Vec<String> = Vec::new();
        let mut removals = 0;
        let mut additions = 0;

        // Collect lines before the hunk
        for i in 0..start_line {
            new_lines.push(lines.get(i).cloned().unwrap_or_default());
        }

        // Process hunk lines
        for patch_line in &hunk.lines {
            match patch_line {
                PatchLine::Context(expected) => {
                    let actual = lines.get(line_idx).map(|s| s.as_str()).unwrap_or("");
                    if !self.lines_match(expected, actual) {
                        return Err(PatchApplyError::ContextMismatch {
                            hunk: 0,
                            expected: expected.clone(),
                            found: actual.to_string(),
                            line: line_idx,
                        });
                    }
                    new_lines.push(actual.to_string());
                    line_idx += 1;
                }
                PatchLine::Remove(expected) => {
                    let actual = lines.get(line_idx).map(|s| s.as_str()).unwrap_or("");
                    if !self.lines_match(expected, actual) {
                        return Err(PatchApplyError::RemovalMismatch {
                            hunk: 0,
                            expected: expected.clone(),
                            found: actual.to_string(),
                            line: line_idx,
                        });
                    }
                    // Don't add to new_lines (removing)
                    line_idx += 1;
                    removals += 1;
                }
                PatchLine::Add(content) => {
                    new_lines.push(content.clone());
                    additions += 1;
                }
            }
        }

        // Add remaining lines
        while line_idx < lines.len() {
            new_lines.push(lines[line_idx].clone());
            line_idx += 1;
        }

        *lines = new_lines;
        Ok(additions as isize - removals as isize)
    }

    fn lines_match(&self, expected: &str, actual: &str) -> bool {
        if self.config.ignore_whitespace {
            expected.split_whitespace().collect::<Vec<_>>()
                == actual.split_whitespace().collect::<Vec<_>>()
        } else {
            expected == actual
        }
    }

    /// Check if a patch can be applied cleanly
    pub fn can_apply(&self, text: &str, patch: &Patch) -> bool {
        let lines: Vec<&str> = text.lines().collect();

        for hunk in &patch.hunks {
            let target = hunk.old_start.saturating_sub(1);
            if !self.hunk_matches(&lines, hunk, target) {
                return false;
            }
        }

        true
    }

    fn hunk_matches(&self, lines: &[&str], hunk: &PatchHunk, start: usize) -> bool {
        let mut idx = start;

        for patch_line in &hunk.lines {
            match patch_line {
                PatchLine::Context(expected) | PatchLine::Remove(expected) => {
                    if let Some(actual) = lines.get(idx) {
                        if !self.lines_match(expected, actual) {
                            return false;
                        }
                        idx += 1;
                    } else {
                        return false;
                    }
                }
                PatchLine::Add(_) => {
                    // Additions don't need to match existing content
                }
            }
        }

        true
    }
}

impl Default for PatchApplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_patch() {
        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified Line 2
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        assert_eq!(patch.hunks.len(), 1);
        assert_eq!(patch.hunks[0].old_start, 1);
        assert_eq!(patch.hunks[0].old_count, 3);
        assert_eq!(patch.hunks[0].new_start, 1);
        assert_eq!(patch.hunks[0].new_count, 3);
    }

    #[test]
    fn test_apply_simple_patch() {
        let original = "Line 1\nLine 2\nLine 3";
        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified Line 2
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();
        let result = applicator.apply(original, &patch).unwrap();

        assert_eq!(result, "Line 1\nModified Line 2\nLine 3");
    }

    #[test]
    fn test_apply_add_lines() {
        let original = "Line 1\nLine 3";
        let patch_text = r#"@@ -1,2 +1,3 @@
 Line 1
+Line 2
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();
        let result = applicator.apply(original, &patch).unwrap();

        assert_eq!(result, "Line 1\nLine 2\nLine 3");
    }

    #[test]
    fn test_apply_delete_lines() {
        let original = "Line 1\nLine 2\nLine 3";
        let patch_text = r#"@@ -1,3 +1,2 @@
 Line 1
-Line 2
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();
        let result = applicator.apply(original, &patch).unwrap();

        assert_eq!(result, "Line 1\nLine 3");
    }

    #[test]
    fn test_reverse_patch() {
        let original = "Line 1\nLine 2\nLine 3";
        let modified = "Line 1\nModified\nLine 3";

        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();

        // Apply forward
        let result = applicator.apply(original, &patch).unwrap();
        assert_eq!(result, modified);

        // Apply reverse
        let reversed = patch.reverse();
        let result = applicator.apply(modified, &reversed).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_patch_to_unified() {
        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified
 Line 3
"#;

        let patch = Patch::parse(patch_text).unwrap();
        let output = patch.to_unified();

        assert!(output.contains("@@ -1,3 +1,3 @@"));
        assert!(output.contains("-Line 2"));
        assert!(output.contains("+Modified"));
    }

    #[test]
    fn test_can_apply() {
        let original = "Line 1\nLine 2\nLine 3";
        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();

        assert!(applicator.can_apply(original, &patch));

        // Already modified - shouldn't apply
        let modified = "Line 1\nModified\nLine 3";
        assert!(!applicator.can_apply(modified, &patch));
    }

    #[test]
    fn test_fuzzy_apply() {
        // Original has different start position
        let original = "Extra line\nLine 1\nLine 2\nLine 3";
        let patch_text = r#"@@ -1,3 +1,3 @@
 Line 1
-Line 2
+Modified
 Line 3"#;

        let patch = Patch::parse(patch_text).unwrap();
        let config = PatchConfig {
            fuzzy: true,
            fuzz_lines: 3,
            ..Default::default()
        };
        let applicator = PatchApplicator::with_config(config);
        let result = applicator.apply_with_result(original, &patch).unwrap();

        assert_eq!(result.text, "Extra line\nLine 1\nModified\nLine 3");
        assert_eq!(result.hunks_offset, 1);
    }

    #[test]
    fn test_empty_patch() {
        let patch_text = "@@ -1,1 +1,1 @@\n Line 1";
        let patch = Patch::parse(patch_text).unwrap();
        let applicator = PatchApplicator::new();
        let result = applicator.apply("Line 1", &patch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_patch_config_defaults() {
        let config = PatchConfig::default();
        assert!(!config.continue_on_error);
        assert!(!config.ignore_whitespace);
    }
}
