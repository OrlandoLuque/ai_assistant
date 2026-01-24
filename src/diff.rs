//! Diff Viewer - Compare responses and text differences
//!
//! This module provides tools for comparing text, responses,
//! and generating visual diffs.


/// Type of change in a diff
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// No change
    Equal,
    /// Line was added
    Added,
    /// Line was removed
    Removed,
    /// Line was modified
    Modified,
}

/// A single line in a diff
#[derive(Debug, Clone)]
pub struct DiffLine {
    /// Line number in the old version (None if added)
    pub old_line_num: Option<usize>,
    /// Line number in the new version (None if removed)
    pub new_line_num: Option<usize>,
    /// The content of the line
    pub content: String,
    /// Type of change
    pub change_type: ChangeType,
}

/// A chunk/hunk of changes
#[derive(Debug, Clone)]
pub struct DiffHunk {
    /// Starting line in old version
    pub old_start: usize,
    /// Number of lines in old version
    pub old_count: usize,
    /// Starting line in new version
    pub new_start: usize,
    /// Number of lines in new version
    pub new_count: usize,
    /// Lines in this hunk
    pub lines: Vec<DiffLine>,
}

/// Result of a diff operation
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// All hunks
    pub hunks: Vec<DiffHunk>,
    /// Total lines added
    pub additions: usize,
    /// Total lines removed
    pub deletions: usize,
    /// Whether the texts are identical
    pub identical: bool,
}

impl DiffResult {
    /// Generate a unified diff format string
    pub fn to_unified(&self, old_name: &str, new_name: &str) -> String {
        if self.identical {
            return String::new();
        }

        let mut output = String::new();
        output.push_str(&format!("--- {}\n", old_name));
        output.push_str(&format!("+++ {}\n", new_name));

        for hunk in &self.hunks {
            output.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.old_start, hunk.old_count,
                hunk.new_start, hunk.new_count
            ));

            for line in &hunk.lines {
                let prefix = match line.change_type {
                    ChangeType::Equal => ' ',
                    ChangeType::Added => '+',
                    ChangeType::Removed => '-',
                    ChangeType::Modified => '~',
                };
                output.push_str(&format!("{}{}\n", prefix, line.content));
            }
        }

        output
    }

    /// Generate a side-by-side diff format
    pub fn to_side_by_side(&self, width: usize) -> String {
        let half_width = width / 2 - 2;
        let mut output = String::new();

        output.push_str(&format!("{:^half_width$} | {:^half_width$}\n", "OLD", "NEW"));
        output.push_str(&format!("{:-<half_width$}-+-{:-<half_width$}\n", "", ""));

        for hunk in &self.hunks {
            for line in &hunk.lines {
                let (left, right) = match line.change_type {
                    ChangeType::Equal => {
                        let truncated = truncate_str(&line.content, half_width);
                        (truncated.clone(), truncated)
                    }
                    ChangeType::Added => {
                        (String::new(), truncate_str(&line.content, half_width))
                    }
                    ChangeType::Removed => {
                        (truncate_str(&line.content, half_width), String::new())
                    }
                    ChangeType::Modified => {
                        (truncate_str(&line.content, half_width), "~".to_string())
                    }
                };

                let marker = match line.change_type {
                    ChangeType::Equal => ' ',
                    ChangeType::Added => '+',
                    ChangeType::Removed => '-',
                    ChangeType::Modified => '~',
                };

                output.push_str(&format!(
                    "{:half_width$} {} {:half_width$}\n",
                    left, marker, right
                ));
            }
        }

        output
    }

    /// Generate an HTML diff
    pub fn to_html(&self) -> String {
        let mut html = String::from(r#"<div class="diff">"#);

        for hunk in &self.hunks {
            html.push_str(r#"<div class="hunk">"#);
            html.push_str(&format!(
                r#"<div class="hunk-header">@@ -{},{} +{},{} @@</div>"#,
                hunk.old_start, hunk.old_count,
                hunk.new_start, hunk.new_count
            ));

            for line in &hunk.lines {
                let class = match line.change_type {
                    ChangeType::Equal => "equal",
                    ChangeType::Added => "added",
                    ChangeType::Removed => "removed",
                    ChangeType::Modified => "modified",
                };
                let escaped = html_escape(&line.content);
                html.push_str(&format!(
                    r#"<div class="line {}"><pre>{}</pre></div>"#,
                    class, escaped
                ));
            }

            html.push_str("</div>");
        }

        html.push_str("</div>");
        html
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        if self.identical {
            "No changes".to_string()
        } else {
            format!(
                "{} addition(s), {} deletion(s), {} hunk(s)",
                self.additions, self.deletions, self.hunks.len()
            )
        }
    }
}

/// Simple diff algorithm (Myers-like)
pub fn diff(old: &str, new: &str) -> DiffResult {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    if old_lines == new_lines {
        return DiffResult {
            hunks: Vec::new(),
            additions: 0,
            deletions: 0,
            identical: true,
        };
    }

    // Simple LCS-based diff
    let lcs = longest_common_subsequence(&old_lines, &new_lines);

    let mut hunks = Vec::new();
    let mut current_hunk: Option<DiffHunk> = None;
    let mut additions = 0;
    let mut deletions = 0;

    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut lcs_idx = 0;

    let context_lines = 3;

    while old_idx < old_lines.len() || new_idx < new_lines.len() {
        // Check if current lines match the LCS
        let in_lcs = lcs_idx < lcs.len() &&
            old_idx < old_lines.len() &&
            new_idx < new_lines.len() &&
            old_lines[old_idx] == lcs[lcs_idx] &&
            new_lines[new_idx] == lcs[lcs_idx];

        if in_lcs {
            // Equal line
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(DiffLine {
                    old_line_num: Some(old_idx + 1),
                    new_line_num: Some(new_idx + 1),
                    content: old_lines[old_idx].to_string(),
                    change_type: ChangeType::Equal,
                });
                hunk.old_count += 1;
                hunk.new_count += 1;
            }
            old_idx += 1;
            new_idx += 1;
            lcs_idx += 1;
        } else {
            // Difference found
            if current_hunk.is_none() {
                // Start new hunk with context
                let start_old = old_idx.saturating_sub(context_lines);
                let start_new = new_idx.saturating_sub(context_lines);

                let mut hunk = DiffHunk {
                    old_start: start_old + 1,
                    old_count: 0,
                    new_start: start_new + 1,
                    new_count: 0,
                    lines: Vec::new(),
                };

                // Add context before
                for i in start_old..old_idx {
                    if i < old_lines.len() {
                        hunk.lines.push(DiffLine {
                            old_line_num: Some(i + 1),
                            new_line_num: Some(start_new + (i - start_old) + 1),
                            content: old_lines[i].to_string(),
                            change_type: ChangeType::Equal,
                        });
                        hunk.old_count += 1;
                        hunk.new_count += 1;
                    }
                }

                current_hunk = Some(hunk);
            }

            // Check what kind of change
            let old_in_lcs = lcs_idx < lcs.len() &&
                old_idx < old_lines.len() &&
                old_lines[old_idx] == lcs[lcs_idx];
            let new_in_lcs = lcs_idx < lcs.len() &&
                new_idx < new_lines.len() &&
                new_lines[new_idx] == lcs[lcs_idx];

            if !old_in_lcs && old_idx < old_lines.len() {
                // Line removed
                if let Some(ref mut hunk) = current_hunk {
                    hunk.lines.push(DiffLine {
                        old_line_num: Some(old_idx + 1),
                        new_line_num: None,
                        content: old_lines[old_idx].to_string(),
                        change_type: ChangeType::Removed,
                    });
                    hunk.old_count += 1;
                }
                deletions += 1;
                old_idx += 1;
            } else if !new_in_lcs && new_idx < new_lines.len() {
                // Line added
                if let Some(ref mut hunk) = current_hunk {
                    hunk.lines.push(DiffLine {
                        old_line_num: None,
                        new_line_num: Some(new_idx + 1),
                        content: new_lines[new_idx].to_string(),
                        change_type: ChangeType::Added,
                    });
                    hunk.new_count += 1;
                }
                additions += 1;
                new_idx += 1;
            } else {
                // Both exhausted
                break;
            }
        }

        // Check if we should close the current hunk
        if current_hunk.is_some() {
            let consecutive_equals = current_hunk.as_ref().unwrap().lines.iter()
                .rev()
                .take_while(|l| l.change_type == ChangeType::Equal)
                .count();

            if consecutive_equals > context_lines * 2 && old_idx < old_lines.len() {
                // Close hunk and start fresh
                let mut hunk = current_hunk.take().unwrap();
                // Trim trailing context
                while hunk.lines.len() > 1 &&
                    hunk.lines.last().map(|l| l.change_type == ChangeType::Equal).unwrap_or(false) &&
                    hunk.lines.iter().rev().take_while(|l| l.change_type == ChangeType::Equal).count() > context_lines
                {
                    hunk.lines.pop();
                    hunk.old_count -= 1;
                    hunk.new_count -= 1;
                }
                hunks.push(hunk);
            }
        }
    }

    // Add final hunk
    if let Some(hunk) = current_hunk {
        hunks.push(hunk);
    }

    DiffResult {
        hunks,
        additions,
        deletions,
        identical: false,
    }
}

/// Find longest common subsequence
fn longest_common_subsequence<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<&'a str> {
    let m = a.len();
    let n = b.len();

    if m == 0 || n == 0 {
        return Vec::new();
    }

    // Build LCS table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find LCS
    let mut lcs = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            lcs.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    lcs.reverse();
    lcs
}

/// Response comparison result
#[derive(Debug, Clone)]
pub struct ResponseComparison {
    /// Text diff
    pub diff: DiffResult,
    /// Word-level statistics
    pub word_stats: WordStats,
    /// Similarity score (0-1)
    pub similarity: f64,
    /// Common phrases
    pub common_phrases: Vec<String>,
}

/// Word-level statistics
#[derive(Debug, Clone, Default)]
pub struct WordStats {
    /// Words in first response
    pub words_a: usize,
    /// Words in second response
    pub words_b: usize,
    /// Common words
    pub common_words: usize,
    /// Unique to first
    pub unique_a: usize,
    /// Unique to second
    pub unique_b: usize,
}

/// Compare two responses
pub fn compare_responses(response_a: &str, response_b: &str) -> ResponseComparison {
    let diff = diff(response_a, response_b);

    // Word analysis
    let words_a: Vec<&str> = response_a.split_whitespace().collect();
    let words_b: Vec<&str> = response_b.split_whitespace().collect();

    let set_a: std::collections::HashSet<&str> = words_a.iter().copied().collect();
    let set_b: std::collections::HashSet<&str> = words_b.iter().copied().collect();

    let common: std::collections::HashSet<&str> = set_a.intersection(&set_b).copied().collect();

    let word_stats = WordStats {
        words_a: words_a.len(),
        words_b: words_b.len(),
        common_words: common.len(),
        unique_a: set_a.len() - common.len(),
        unique_b: set_b.len() - common.len(),
    };

    // Calculate similarity
    let similarity = if set_a.is_empty() && set_b.is_empty() {
        1.0
    } else {
        let union_size = set_a.union(&set_b).count();
        if union_size == 0 {
            0.0
        } else {
            common.len() as f64 / union_size as f64
        }
    };

    // Find common phrases (3+ word sequences)
    let common_phrases = find_common_phrases(&words_a, &words_b, 3);

    ResponseComparison {
        diff,
        word_stats,
        similarity,
        common_phrases,
    }
}

/// Find common phrases between two word sequences
fn find_common_phrases(words_a: &[&str], words_b: &[&str], min_length: usize) -> Vec<String> {
    let mut phrases = Vec::new();

    for window_size in (min_length..=10).rev() {
        if words_a.len() < window_size || words_b.len() < window_size {
            continue;
        }

        for window_a in words_a.windows(window_size) {
            for window_b in words_b.windows(window_size) {
                if window_a == window_b {
                    let phrase = window_a.join(" ");
                    // Don't add if it's a substring of existing phrase
                    if !phrases.iter().any(|p: &String| p.contains(&phrase)) {
                        phrases.push(phrase);
                    }
                }
            }
        }
    }

    phrases.truncate(10);
    phrases
}

/// Inline word diff for a single line
#[derive(Debug, Clone)]
pub struct InlineWordDiff {
    /// Segments of the diff
    pub segments: Vec<InlineDiffSegment>,
}

/// A segment in an inline diff
#[derive(Debug, Clone)]
pub struct InlineDiffSegment {
    /// The text
    pub text: String,
    /// Type of change
    pub change_type: ChangeType,
}

/// Generate word-level inline diff
pub fn inline_word_diff(old_line: &str, new_line: &str) -> InlineWordDiff {
    let old_words: Vec<&str> = old_line.split_whitespace().collect();
    let new_words: Vec<&str> = new_line.split_whitespace().collect();

    let lcs = longest_common_subsequence(&old_words, &new_words);

    let mut segments = Vec::new();
    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut lcs_idx = 0;

    while old_idx < old_words.len() || new_idx < new_words.len() {
        let in_lcs = lcs_idx < lcs.len() &&
            old_idx < old_words.len() &&
            new_idx < new_words.len() &&
            old_words[old_idx] == lcs[lcs_idx] &&
            new_words[new_idx] == lcs[lcs_idx];

        if in_lcs {
            segments.push(InlineDiffSegment {
                text: old_words[old_idx].to_string(),
                change_type: ChangeType::Equal,
            });
            old_idx += 1;
            new_idx += 1;
            lcs_idx += 1;
        } else {
            // Check for removed words
            while old_idx < old_words.len() &&
                (lcs_idx >= lcs.len() || old_words[old_idx] != lcs[lcs_idx])
            {
                segments.push(InlineDiffSegment {
                    text: old_words[old_idx].to_string(),
                    change_type: ChangeType::Removed,
                });
                old_idx += 1;
            }

            // Check for added words
            while new_idx < new_words.len() &&
                (lcs_idx >= lcs.len() || new_words[new_idx] != lcs[lcs_idx])
            {
                segments.push(InlineDiffSegment {
                    text: new_words[new_idx].to_string(),
                    change_type: ChangeType::Added,
                });
                new_idx += 1;
            }
        }
    }

    InlineWordDiff { segments }
}

/// Truncate string to width
fn truncate_str(s: &str, max_width: usize) -> String {
    if s.len() <= max_width {
        s.to_string()
    } else if max_width > 3 {
        format!("{}...", &s[..max_width - 3])
    } else {
        s[..max_width].to_string()
    }
}

/// HTML escape
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts() {
        let result = diff("hello\nworld", "hello\nworld");
        assert!(result.identical);
        assert_eq!(result.additions, 0);
        assert_eq!(result.deletions, 0);
    }

    #[test]
    fn test_simple_addition() {
        let result = diff("hello", "hello\nworld");
        assert!(!result.identical);
        assert_eq!(result.additions, 1);
        assert_eq!(result.deletions, 0);
    }

    #[test]
    fn test_simple_deletion() {
        let result = diff("hello\nworld", "hello");
        assert!(!result.identical);
        assert_eq!(result.additions, 0);
        assert_eq!(result.deletions, 1);
    }

    #[test]
    fn test_unified_format() {
        let result = diff("hello\nworld", "hello\nuniverse");
        let unified = result.to_unified("old.txt", "new.txt");

        assert!(unified.contains("--- old.txt"));
        assert!(unified.contains("+++ new.txt"));
        assert!(unified.contains("-world"));
        assert!(unified.contains("+universe"));
    }

    #[test]
    fn test_html_format() {
        let result = diff("hello", "hello\nworld");
        let html = result.to_html();

        assert!(html.contains("class=\"diff\""));
        // The diff includes lines with equal and added classes
        assert!(html.contains("class=\"line"));
    }

    #[test]
    fn test_compare_responses() {
        let response_a = "The quick brown fox jumps over the lazy dog";
        let response_b = "The quick red fox jumps over the sleepy dog";

        let comparison = compare_responses(response_a, response_b);

        assert!(comparison.similarity > 0.5);
        assert!(comparison.word_stats.common_words > 0);
    }

    #[test]
    fn test_inline_word_diff() {
        let diff = inline_word_diff("hello world", "hello universe");

        assert!(!diff.segments.is_empty());
        let equal_count = diff.segments.iter()
            .filter(|s| s.change_type == ChangeType::Equal)
            .count();
        assert!(equal_count > 0);
    }

    #[test]
    fn test_common_phrases() {
        let words_a: Vec<&str> = "the quick brown fox".split_whitespace().collect();
        let words_b: Vec<&str> = "the quick brown dog".split_whitespace().collect();

        let phrases = find_common_phrases(&words_a, &words_b, 3);
        assert!(!phrases.is_empty());
        assert!(phrases[0].contains("the quick brown"));
    }

    #[test]
    fn test_lcs() {
        let a = vec!["a", "b", "c", "d"];
        let b = vec!["a", "c", "d"];

        let lcs = longest_common_subsequence(&a, &b);
        assert_eq!(lcs, vec!["a", "c", "d"]);
    }

    #[test]
    fn test_summary() {
        let result = diff("hello", "hello\nworld\nuniverse");
        let summary = result.summary();

        assert!(summary.contains("addition"));
    }

    #[test]
    fn test_empty_diff() {
        let result = diff("", "");
        assert!(result.identical);
    }

    #[test]
    fn test_word_stats() {
        let comparison = compare_responses("apple banana cherry", "banana cherry date");

        assert_eq!(comparison.word_stats.words_a, 3);
        assert_eq!(comparison.word_stats.words_b, 3);
        assert_eq!(comparison.word_stats.common_words, 2); // banana, cherry
    }
}
