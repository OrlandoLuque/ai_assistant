//! Core edit operations for text manipulation
//!
//! This module provides fundamental editing primitives that can be used
//! to build more complex editing operations.
//!
//! # Features
//!
//! - **Position-based edits**: Insert, delete, replace at specific positions
//! - **Line-based edits**: Operations on whole lines
//! - **Edit batching**: Combine multiple edits into atomic operations
//! - **Edit validation**: Ensure edits don't conflict or go out of bounds
//! - **Undo/Redo support**: Track edit history for reversal
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::edit_operations::{EditBuilder, TextEditor};
//!
//! let mut editor = TextEditor::new("Hello World");
//!
//! // Single edit
//! editor.replace_range(0..5, "Hi");
//! assert_eq!(editor.text(), "Hi World");
//!
//! // Batched edits
//! let edits = EditBuilder::new()
//!     .insert(3, " Beautiful")
//!     .delete(9..14)
//!     .build();
//! editor.apply_batch(&edits);
//! ```

use std::collections::VecDeque;
use std::ops::Range;

/// A position in text (line, column) - both 0-indexed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

impl Position {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    /// Create position at start of document
    pub fn start() -> Self {
        Self { line: 0, column: 0 }
    }

    /// Convert to byte offset in text
    pub fn to_offset(&self, text: &str) -> Option<usize> {
        let mut current_line = 0;
        let mut offset = 0;

        for (i, c) in text.char_indices() {
            if current_line == self.line {
                let col_offset = text[offset..]
                    .chars()
                    .take(self.column)
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
                return Some(offset + col_offset);
            }
            if c == '\n' {
                current_line += 1;
                offset = i + 1;
            }
        }

        if current_line == self.line {
            let col_offset = text[offset..]
                .chars()
                .take(self.column)
                .map(|c| c.len_utf8())
                .sum::<usize>();
            Some(offset + col_offset)
        } else {
            None
        }
    }

    /// Create from byte offset
    pub fn from_offset(text: &str, offset: usize) -> Self {
        let mut line = 0;
        let mut col = 0;

        for (i, c) in text.char_indices() {
            if i >= offset {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }

        Self { line, column: col }
    }
}

/// A range in text using positions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextRange {
    pub start: Position,
    pub end: Position,
}

impl TextRange {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    /// Convert to byte range
    pub fn to_byte_range(&self, text: &str) -> Option<Range<usize>> {
        let start = self.start.to_offset(text)?;
        let end = self.end.to_offset(text)?;
        Some(start..end)
    }

    /// Check if range is valid (start <= end)
    pub fn is_valid(&self) -> bool {
        self.start.line < self.end.line
            || (self.start.line == self.end.line && self.start.column <= self.end.column)
    }

    /// Check if this range contains a position
    pub fn contains(&self, pos: &Position) -> bool {
        if pos.line < self.start.line || pos.line > self.end.line {
            return false;
        }
        if pos.line == self.start.line && pos.column < self.start.column {
            return false;
        }
        if pos.line == self.end.line && pos.column > self.end.column {
            return false;
        }
        true
    }

    /// Check if two ranges overlap
    pub fn overlaps(&self, other: &TextRange) -> bool {
        self.contains(&other.start)
            || self.contains(&other.end)
            || other.contains(&self.start)
            || other.contains(&self.end)
    }
}

/// Type of edit operation
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditKind {
    /// Insert text at position
    Insert,
    /// Delete text in range
    Delete,
    /// Replace text in range with new text
    Replace,
}

/// A single edit operation
#[derive(Debug, Clone)]
pub struct Edit {
    /// Kind of edit
    pub kind: EditKind,
    /// Byte offset where edit starts
    pub offset: usize,
    /// Number of bytes to delete (0 for insert)
    pub delete_len: usize,
    /// Text to insert (empty for delete)
    pub insert_text: String,
    /// Optional description of the edit
    pub description: Option<String>,
}

impl Edit {
    /// Create an insert edit
    pub fn insert(offset: usize, text: impl Into<String>) -> Self {
        Self {
            kind: EditKind::Insert,
            offset,
            delete_len: 0,
            insert_text: text.into(),
            description: None,
        }
    }

    /// Create a delete edit
    pub fn delete(range: Range<usize>) -> Self {
        Self {
            kind: EditKind::Delete,
            offset: range.start,
            delete_len: range.end - range.start,
            insert_text: String::new(),
            description: None,
        }
    }

    /// Create a replace edit
    pub fn replace(range: Range<usize>, text: impl Into<String>) -> Self {
        Self {
            kind: EditKind::Replace,
            offset: range.start,
            delete_len: range.end - range.start,
            insert_text: text.into(),
            description: None,
        }
    }

    /// Add description to edit
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Calculate the change in length this edit causes
    pub fn length_delta(&self) -> isize {
        self.insert_text.len() as isize - self.delete_len as isize
    }

    /// Apply this edit to text
    pub fn apply(&self, text: &mut String) -> Result<(), EditError> {
        if self.offset > text.len() {
            return Err(EditError::OutOfBounds {
                offset: self.offset,
                text_len: text.len(),
            });
        }

        if self.offset + self.delete_len > text.len() {
            return Err(EditError::OutOfBounds {
                offset: self.offset + self.delete_len,
                text_len: text.len(),
            });
        }

        // Perform the edit
        let end = self.offset + self.delete_len;
        text.replace_range(self.offset..end, &self.insert_text);

        Ok(())
    }

    /// Create the inverse edit (for undo)
    pub fn inverse(&self, original_text: &str) -> Self {
        let deleted_text = &original_text[self.offset..self.offset + self.delete_len];
        Self {
            kind: match self.kind {
                EditKind::Insert => EditKind::Delete,
                EditKind::Delete => EditKind::Insert,
                EditKind::Replace => EditKind::Replace,
            },
            offset: self.offset,
            delete_len: self.insert_text.len(),
            insert_text: deleted_text.to_string(),
            description: self.description.clone().map(|d| format!("Undo: {}", d)),
        }
    }
}

/// Error during edit operation
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditError {
    /// Edit position is out of bounds
    OutOfBounds { offset: usize, text_len: usize },
    /// Edits overlap and cannot be applied together
    OverlappingEdits { edit1: usize, edit2: usize },
    /// Edit validation failed
    ValidationFailed(String),
    /// No edits to undo
    NothingToUndo,
    /// No edits to redo
    NothingToRedo,
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds { offset, text_len } => {
                write!(
                    f,
                    "Edit offset {} is out of bounds (text length: {})",
                    offset, text_len
                )
            }
            Self::OverlappingEdits { edit1, edit2 } => {
                write!(f, "Edits {} and {} overlap", edit1, edit2)
            }
            Self::ValidationFailed(msg) => write!(f, "Edit validation failed: {}", msg),
            Self::NothingToUndo => write!(f, "Nothing to undo"),
            Self::NothingToRedo => write!(f, "Nothing to redo"),
        }
    }
}

impl std::error::Error for EditError {}

/// Builder for creating multiple edits
#[derive(Debug, Default)]
pub struct EditBuilder {
    edits: Vec<Edit>,
}

impl EditBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an insert edit
    pub fn insert(mut self, offset: usize, text: impl Into<String>) -> Self {
        self.edits.push(Edit::insert(offset, text));
        self
    }

    /// Add a delete edit
    pub fn delete(mut self, range: Range<usize>) -> Self {
        self.edits.push(Edit::delete(range));
        self
    }

    /// Add a replace edit
    pub fn replace(mut self, range: Range<usize>, text: impl Into<String>) -> Self {
        self.edits.push(Edit::replace(range, text));
        self
    }

    /// Add any edit
    pub fn edit(mut self, edit: Edit) -> Self {
        self.edits.push(edit);
        self
    }

    /// Build the list of edits, sorted by offset (descending for safe application)
    pub fn build(mut self) -> Vec<Edit> {
        // Sort by offset descending so we can apply from end to start
        // This prevents offset invalidation
        self.edits.sort_by(|a, b| b.offset.cmp(&a.offset));
        self.edits
    }

    /// Build and validate edits don't overlap
    pub fn build_validated(self) -> Result<Vec<Edit>, EditError> {
        let edits = self.build();

        // Check for overlaps (since sorted descending, check consecutive pairs)
        for i in 0..edits.len().saturating_sub(1) {
            let current = &edits[i];
            let next = &edits[i + 1];

            // Current starts at higher offset, next at lower
            // They overlap if next's end > current's start
            let next_end = next.offset + next.delete_len;
            if next_end > current.offset {
                return Err(EditError::OverlappingEdits {
                    edit1: i,
                    edit2: i + 1,
                });
            }
        }

        Ok(edits)
    }
}

/// A text editor with undo/redo support
pub struct TextEditor {
    text: String,
    undo_stack: VecDeque<Vec<Edit>>,
    redo_stack: VecDeque<Vec<Edit>>,
    max_history: usize,
}

impl TextEditor {
    /// Create a new editor with the given text
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Get the current text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the current text length
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Check if text is empty
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Set maximum undo history size
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
        while self.undo_stack.len() > max {
            self.undo_stack.pop_front();
        }
    }

    /// Apply a single edit
    pub fn apply(&mut self, edit: Edit) -> Result<(), EditError> {
        let inverse = edit.inverse(&self.text);
        edit.apply(&mut self.text)?;

        self.push_undo(vec![inverse]);
        self.redo_stack.clear();

        Ok(())
    }

    /// Apply multiple edits as a single undoable operation
    pub fn apply_batch(&mut self, edits: &[Edit]) -> Result<(), EditError> {
        if edits.is_empty() {
            return Ok(());
        }

        // Create inverses before applying
        let mut inverses: Vec<Edit> = Vec::new();
        let mut temp_text = self.text.clone();

        // Apply edits (should be sorted by offset descending)
        for edit in edits {
            inverses.push(edit.inverse(&temp_text));
            edit.apply(&mut temp_text)?;
        }

        // If all succeeded, commit
        self.text = temp_text;

        // Reverse inverses so undo applies them in correct order
        inverses.reverse();
        self.push_undo(inverses);
        self.redo_stack.clear();

        Ok(())
    }

    /// Insert text at offset
    pub fn insert(&mut self, offset: usize, text: impl Into<String>) -> Result<(), EditError> {
        self.apply(Edit::insert(offset, text))
    }

    /// Delete text in range
    pub fn delete(&mut self, range: Range<usize>) -> Result<(), EditError> {
        self.apply(Edit::delete(range))
    }

    /// Replace text in range
    pub fn replace_range(
        &mut self,
        range: Range<usize>,
        text: impl Into<String>,
    ) -> Result<(), EditError> {
        self.apply(Edit::replace(range, text))
    }

    /// Insert text at position
    pub fn insert_at(&mut self, pos: Position, text: impl Into<String>) -> Result<(), EditError> {
        let offset = pos.to_offset(&self.text).ok_or(EditError::OutOfBounds {
            offset: 0,
            text_len: self.text.len(),
        })?;
        self.insert(offset, text)
    }

    /// Replace line content (keeps line ending)
    pub fn replace_line(
        &mut self,
        line_num: usize,
        new_content: impl Into<String>,
    ) -> Result<(), EditError> {
        let lines: Vec<&str> = self.text.lines().collect();

        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        // Find line start and end
        let mut offset = 0;
        for (i, line) in self.text.lines().enumerate() {
            if i == line_num {
                let line_end = offset + line.len();
                return self.replace_range(offset..line_end, new_content);
            }
            offset += line.len() + 1; // +1 for newline
        }

        Err(EditError::OutOfBounds {
            offset: line_num,
            text_len: lines.len(),
        })
    }

    /// Insert a new line at the given line number
    pub fn insert_line(
        &mut self,
        line_num: usize,
        content: impl Into<String>,
    ) -> Result<(), EditError> {
        let content = content.into();
        let lines: Vec<&str> = self.text.lines().collect();

        let offset = if line_num == 0 {
            0
        } else if line_num >= lines.len() {
            self.text.len()
        } else {
            let mut off = 0;
            for (i, line) in self.text.lines().enumerate() {
                if i == line_num {
                    break;
                }
                off += line.len() + 1;
            }
            off
        };

        let insert_text = if line_num >= lines.len() {
            format!("\n{}", content)
        } else {
            format!("{}\n", content)
        };

        self.insert(offset, insert_text)
    }

    /// Delete a line
    pub fn delete_line(&mut self, line_num: usize) -> Result<(), EditError> {
        let lines: Vec<&str> = self.text.lines().collect();

        if line_num >= lines.len() {
            return Err(EditError::OutOfBounds {
                offset: line_num,
                text_len: lines.len(),
            });
        }

        let mut offset = 0;
        for (i, line) in self.text.lines().enumerate() {
            if i == line_num {
                let mut end = offset + line.len();
                // Include the newline if not the last line
                if end < self.text.len() {
                    end += 1;
                } else if offset > 0 {
                    // Last line - remove preceding newline instead
                    offset -= 1;
                }
                return self.delete(offset..end);
            }
            offset += line.len() + 1;
        }

        Ok(())
    }

    /// Undo the last operation
    pub fn undo(&mut self) -> Result<(), EditError> {
        let edits = self.undo_stack.pop_back().ok_or(EditError::NothingToUndo)?;

        // Create redo edits
        let mut redo_edits = Vec::new();
        for edit in &edits {
            redo_edits.push(edit.inverse(&self.text));
            edit.clone().apply(&mut self.text)?;
        }

        redo_edits.reverse();
        self.redo_stack.push_back(redo_edits);

        Ok(())
    }

    /// Redo the last undone operation
    pub fn redo(&mut self) -> Result<(), EditError> {
        let edits = self.redo_stack.pop_back().ok_or(EditError::NothingToRedo)?;

        let mut undo_edits = Vec::new();
        for edit in &edits {
            undo_edits.push(edit.inverse(&self.text));
            edit.clone().apply(&mut self.text)?;
        }

        undo_edits.reverse();
        self.undo_stack.push_back(undo_edits);

        Ok(())
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get number of available undos
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get number of available redos
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clear undo/redo history
    pub fn clear_history(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    fn push_undo(&mut self, edits: Vec<Edit>) {
        self.undo_stack.push_back(edits);
        while self.undo_stack.len() > self.max_history {
            self.undo_stack.pop_front();
        }
    }
}

/// Line-based editor for simpler operations
pub struct LineEditor {
    lines: Vec<String>,
    undo_stack: VecDeque<Vec<String>>,
    redo_stack: VecDeque<Vec<String>>,
}

impl LineEditor {
    pub fn new(text: &str) -> Self {
        Self {
            lines: text.lines().map(|s| s.to_string()).collect(),
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
        }
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub fn get_line(&self, index: usize) -> Option<&str> {
        self.lines.get(index).map(|s| s.as_str())
    }

    pub fn set_line(&mut self, index: usize, content: impl Into<String>) -> bool {
        if index < self.lines.len() {
            self.save_state();
            self.lines[index] = content.into();
            true
        } else {
            false
        }
    }

    pub fn insert_line(&mut self, index: usize, content: impl Into<String>) {
        self.save_state();
        let index = index.min(self.lines.len());
        self.lines.insert(index, content.into());
    }

    pub fn delete_line(&mut self, index: usize) -> Option<String> {
        if index < self.lines.len() {
            self.save_state();
            Some(self.lines.remove(index))
        } else {
            None
        }
    }

    pub fn append_line(&mut self, content: impl Into<String>) {
        self.save_state();
        self.lines.push(content.into());
    }

    pub fn to_string(&self) -> String {
        self.lines.join("\n")
    }

    pub fn undo(&mut self) -> bool {
        if let Some(state) = self.undo_stack.pop_back() {
            self.redo_stack.push_back(self.lines.clone());
            self.lines = state;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(state) = self.redo_stack.pop_back() {
            self.undo_stack.push_back(self.lines.clone());
            self.lines = state;
            true
        } else {
            false
        }
    }

    fn save_state(&mut self) {
        self.undo_stack.push_back(self.lines.clone());
        self.redo_stack.clear();
        if self.undo_stack.len() > 50 {
            self.undo_stack.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_to_offset() {
        let text = "Hello\nWorld\nTest";

        assert_eq!(Position::new(0, 0).to_offset(text), Some(0));
        assert_eq!(Position::new(0, 5).to_offset(text), Some(5));
        assert_eq!(Position::new(1, 0).to_offset(text), Some(6));
        assert_eq!(Position::new(1, 5).to_offset(text), Some(11));
        assert_eq!(Position::new(2, 0).to_offset(text), Some(12));
    }

    #[test]
    fn test_position_from_offset() {
        let text = "Hello\nWorld\nTest";

        assert_eq!(Position::from_offset(text, 0), Position::new(0, 0));
        assert_eq!(Position::from_offset(text, 5), Position::new(0, 5));
        assert_eq!(Position::from_offset(text, 6), Position::new(1, 0));
        assert_eq!(Position::from_offset(text, 11), Position::new(1, 5));
    }

    #[test]
    fn test_edit_insert() {
        let mut text = "Hello World".to_string();
        let edit = Edit::insert(5, " Beautiful");
        edit.apply(&mut text).unwrap();
        assert_eq!(text, "Hello Beautiful World");
    }

    #[test]
    fn test_edit_delete() {
        let mut text = "Hello Beautiful World".to_string();
        let edit = Edit::delete(5..15);
        edit.apply(&mut text).unwrap();
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_edit_replace() {
        let mut text = "Hello World".to_string();
        let edit = Edit::replace(6..11, "Rust");
        edit.apply(&mut text).unwrap();
        assert_eq!(text, "Hello Rust");
    }

    #[test]
    fn test_edit_inverse() {
        let original = "Hello World";
        let edit = Edit::replace(6..11, "Rust");
        let inverse = edit.inverse(original);

        let mut text = original.to_string();
        edit.apply(&mut text).unwrap();
        assert_eq!(text, "Hello Rust");

        inverse.apply(&mut text).unwrap();
        assert_eq!(text, original);
    }

    #[test]
    fn test_text_editor_undo_redo() {
        let mut editor = TextEditor::new("Hello World");

        editor.replace_range(6..11, "Rust").unwrap();
        assert_eq!(editor.text(), "Hello Rust");

        editor.undo().unwrap();
        assert_eq!(editor.text(), "Hello World");

        editor.redo().unwrap();
        assert_eq!(editor.text(), "Hello Rust");
    }

    #[test]
    fn test_text_editor_batch() {
        let mut editor = TextEditor::new("Line 1\nLine 2\nLine 3");

        let edits = EditBuilder::new()
            .replace(0..6, "First")
            .replace(7..13, "Second")
            .build();

        editor.apply_batch(&edits).unwrap();
        assert_eq!(editor.text(), "First\nSecond\nLine 3");

        editor.undo().unwrap();
        assert_eq!(editor.text(), "Line 1\nLine 2\nLine 3");
    }

    #[test]
    fn test_line_editor() {
        let mut editor = LineEditor::new("Line 1\nLine 2\nLine 3");

        assert_eq!(editor.line_count(), 3);
        assert_eq!(editor.get_line(1), Some("Line 2"));

        editor.set_line(1, "Modified");
        assert_eq!(editor.get_line(1), Some("Modified"));

        editor.undo();
        assert_eq!(editor.get_line(1), Some("Line 2"));
    }

    #[test]
    fn test_edit_builder_validation() {
        // Non-overlapping edits should succeed
        let result = EditBuilder::new()
            .replace(0..5, "A")
            .replace(10..15, "B")
            .build_validated();
        assert!(result.is_ok());

        // Overlapping edits should fail
        let result = EditBuilder::new()
            .replace(0..10, "A")
            .replace(5..15, "B")
            .build_validated();
        assert!(matches!(result, Err(EditError::OverlappingEdits { .. })));
    }

    #[test]
    fn test_replace_line() {
        let mut editor = TextEditor::new("Line 1\nLine 2\nLine 3");

        editor.replace_line(1, "Modified Line").unwrap();
        assert_eq!(editor.text(), "Line 1\nModified Line\nLine 3");
    }

    #[test]
    fn test_insert_line() {
        let mut editor = TextEditor::new("Line 1\nLine 3");

        editor.insert_line(1, "Line 2").unwrap();
        assert_eq!(editor.text(), "Line 1\nLine 2\nLine 3");
    }

    #[test]
    fn test_delete_line() {
        let mut editor = TextEditor::new("Line 1\nLine 2\nLine 3");

        editor.delete_line(1).unwrap();
        assert_eq!(editor.text(), "Line 1\nLine 3");
    }
}
