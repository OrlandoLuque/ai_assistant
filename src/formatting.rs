//! Response formatting and parsing utilities
//!
//! This module provides tools for parsing and extracting structured content from
//! AI responses, including JSON, code blocks, lists, and tables.

use serde::{Deserialize, Serialize};

/// A parsed response with extracted components
#[derive(Debug, Clone, Default)]
pub struct ParsedResponse {
    /// The original full response
    pub raw: String,
    /// Plain text content (without code blocks, etc.)
    pub text: String,
    /// Extracted code blocks
    pub code_blocks: Vec<CodeBlock>,
    /// Extracted JSON objects
    pub json_objects: Vec<serde_json::Value>,
    /// Extracted lists
    pub lists: Vec<ParsedList>,
    /// Extracted tables
    pub tables: Vec<ParsedTable>,
    /// Extracted links
    pub links: Vec<ParsedLink>,
    /// Headings found in the response
    pub headings: Vec<Heading>,
}

/// A code block extracted from the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    /// The language identifier (if specified)
    pub language: Option<String>,
    /// The code content
    pub code: String,
    /// Line number in original response where block starts
    pub start_line: usize,
    /// Whether this appears to be a complete code snippet
    pub is_complete: bool,
}

/// A list extracted from the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedList {
    /// List items
    pub items: Vec<ListItem>,
    /// Whether it's an ordered list
    pub ordered: bool,
    /// Nesting level (0 = top level)
    pub level: usize,
}

/// An item in a list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListItem {
    /// The text content
    pub text: String,
    /// Item number (for ordered lists)
    pub number: Option<usize>,
    /// Checkbox state (for task lists)
    pub checkbox: Option<bool>,
    /// Nested list
    pub nested: Option<ParsedList>,
}

/// A table extracted from the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedTable {
    /// Table headers
    pub headers: Vec<String>,
    /// Table rows
    pub rows: Vec<Vec<String>>,
    /// Column alignments
    pub alignments: Vec<TableAlignment>,
}

/// Table column alignment
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TableAlignment {
    Left,
    Center,
    Right,
    None,
}

/// A link extracted from the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedLink {
    /// Link text
    pub text: String,
    /// URL
    pub url: String,
    /// Title (if any)
    pub title: Option<String>,
}

/// A heading found in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heading {
    /// Heading level (1-6)
    pub level: u8,
    /// Heading text
    pub text: String,
    /// Line number
    pub line: usize,
}

/// Response parser
pub struct ResponseParser {
    /// Configuration
    config: ParserConfig,
}

/// Parser configuration
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Extract code blocks
    pub extract_code: bool,
    /// Extract JSON
    pub extract_json: bool,
    /// Extract lists
    pub extract_lists: bool,
    /// Extract tables
    pub extract_tables: bool,
    /// Extract links
    pub extract_links: bool,
    /// Extract headings
    pub extract_headings: bool,
    /// Parse nested JSON
    pub parse_nested_json: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            extract_code: true,
            extract_json: true,
            extract_lists: true,
            extract_tables: true,
            extract_links: true,
            extract_headings: true,
            parse_nested_json: true,
        }
    }
}

impl ResponseParser {
    /// Create a new parser with default config
    pub fn new() -> Self {
        Self {
            config: ParserConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ParserConfig) -> Self {
        Self { config }
    }

    /// Parse a response string
    pub fn parse(&self, response: &str) -> ParsedResponse {
        let mut result = ParsedResponse {
            raw: response.to_string(),
            text: String::new(),
            ..Default::default()
        };

        // Extract code blocks first (they might contain JSON, lists, etc.)
        if self.config.extract_code {
            result.code_blocks = self.extract_code_blocks(response);
        }

        // Remove code blocks from text for further parsing
        let text_without_code = self.remove_code_blocks(response);

        // Extract JSON objects
        if self.config.extract_json {
            result.json_objects = self.extract_json(&text_without_code);

            // Also check code blocks for JSON
            for block in &result.code_blocks {
                if block.language.as_deref() == Some("json") {
                    if let Ok(json) = serde_json::from_str(&block.code) {
                        result.json_objects.push(json);
                    }
                }
            }
        }

        // Extract lists
        if self.config.extract_lists {
            result.lists = self.extract_lists(&text_without_code);
        }

        // Extract tables
        if self.config.extract_tables {
            result.tables = self.extract_tables(&text_without_code);
        }

        // Extract links
        if self.config.extract_links {
            result.links = self.extract_links(&text_without_code);
        }

        // Extract headings
        if self.config.extract_headings {
            result.headings = self.extract_headings(response);
        }

        // Set plain text (without markdown formatting)
        result.text = self.strip_markdown(&text_without_code);

        result
    }

    /// Extract code blocks from response
    fn extract_code_blocks(&self, text: &str) -> Vec<CodeBlock> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Check for fenced code block
            if line.starts_with("```") {
                let language = if line.len() > 3 {
                    Some(line[3..].trim().to_string())
                } else {
                    None
                };

                let start_line = i;
                let mut code_lines = Vec::new();
                i += 1;

                // Collect code until closing fence
                while i < lines.len() && !lines[i].trim().starts_with("```") {
                    code_lines.push(lines[i]);
                    i += 1;
                }

                let code = code_lines.join("\n");
                let is_complete = i < lines.len(); // Has closing fence

                blocks.push(CodeBlock {
                    language: language.filter(|l| !l.is_empty()),
                    code,
                    start_line,
                    is_complete,
                });
            }

            i += 1;
        }

        blocks
    }

    /// Remove code blocks from text
    fn remove_code_blocks(&self, text: &str) -> String {
        let mut result = String::new();
        let mut in_code_block = false;

        for line in text.lines() {
            if line.trim().starts_with("```") {
                in_code_block = !in_code_block;
                continue;
            }

            if !in_code_block {
                result.push_str(line);
                result.push('\n');
            }
        }

        result
    }

    /// Extract JSON objects from text
    fn extract_json(&self, text: &str) -> Vec<serde_json::Value> {
        let mut objects = Vec::new();

        // Try to find JSON objects
        let mut depth = 0;
        let mut start: Option<usize> = None;

        for (i, c) in text.char_indices() {
            match c {
                '{' | '[' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start_idx) = start {
                            let json_str = &text[start_idx..=i];
                            if let Ok(json) = serde_json::from_str(json_str) {
                                objects.push(json);
                            }
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }

        objects
    }

    /// Extract lists from text
    fn extract_lists(&self, text: &str) -> Vec<ParsedList> {
        let mut lists = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim_start();

            // Check for list item
            if let Some((is_ordered, number, checkbox, content)) = self.parse_list_item(trimmed) {
                let indent = line.len() - trimmed.len();
                let level = indent / 2; // Assume 2-space indent

                let mut items = vec![ListItem {
                    text: content,
                    number,
                    checkbox,
                    nested: None,
                }];

                i += 1;

                // Continue collecting items at same level
                while i < lines.len() {
                    let next_line = lines[i];
                    let next_trimmed = next_line.trim_start();
                    let next_indent = next_line.len() - next_trimmed.len();

                    if let Some((_, num, cb, cont)) = self.parse_list_item(next_trimmed) {
                        if next_indent / 2 == level {
                            items.push(ListItem {
                                text: cont,
                                number: num,
                                checkbox: cb,
                                nested: None,
                            });
                            i += 1;
                        } else {
                            break;
                        }
                    } else if next_trimmed.is_empty() {
                        i += 1;
                    } else {
                        break;
                    }
                }

                lists.push(ParsedList {
                    items,
                    ordered: is_ordered,
                    level,
                });
            } else {
                i += 1;
            }
        }

        lists
    }

    /// Parse a single list item
    fn parse_list_item(&self, line: &str) -> Option<(bool, Option<usize>, Option<bool>, String)> {
        // Ordered list: 1. item, 1) item
        if let Some(_rest) = line.strip_prefix(|c: char| c.is_ascii_digit()) {
            let mut num_end = 0;
            for (i, c) in line.char_indices() {
                if c.is_ascii_digit() {
                    num_end = i + 1;
                } else {
                    break;
                }
            }

            let after_num = &line[num_end..];
            if after_num.starts_with(". ") || after_num.starts_with(") ") {
                let number: usize = line[..num_end].parse().ok()?;
                let content = after_num[2..].trim().to_string();

                // Check for checkbox
                let (checkbox, final_content) = self.parse_checkbox(&content);

                return Some((true, Some(number), checkbox, final_content));
            }
        }

        // Unordered list: - item, * item, + item
        for prefix in &["- ", "* ", "+ "] {
            if let Some(content) = line.strip_prefix(prefix) {
                let (checkbox, final_content) = self.parse_checkbox(content.trim());
                return Some((false, None, checkbox, final_content));
            }
        }

        None
    }

    /// Parse checkbox from list item content
    fn parse_checkbox(&self, content: &str) -> (Option<bool>, String) {
        if let Some(rest) = content.strip_prefix("[ ] ") {
            (Some(false), rest.to_string())
        } else if let Some(rest) = content
            .strip_prefix("[x] ")
            .or(content.strip_prefix("[X] "))
        {
            (Some(true), rest.to_string())
        } else {
            (None, content.to_string())
        }
    }

    /// Extract tables from text
    fn extract_tables(&self, text: &str) -> Vec<ParsedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            // Look for table header (line with |)
            if lines[i].contains('|') && i + 1 < lines.len() {
                // Check if next line is separator
                let next_line = lines[i + 1].trim();
                if next_line.contains('-') && next_line.contains('|') {
                    // Parse header
                    let headers: Vec<String> = lines[i]
                        .split('|')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();

                    // Parse alignments
                    let alignments: Vec<TableAlignment> = next_line
                        .split('|')
                        .filter(|s| !s.trim().is_empty())
                        .map(|s| {
                            let s = s.trim();
                            let left_colon = s.starts_with(':');
                            let right_colon = s.ends_with(':');
                            match (left_colon, right_colon) {
                                (true, true) => TableAlignment::Center,
                                (true, false) => TableAlignment::Left,
                                (false, true) => TableAlignment::Right,
                                _ => TableAlignment::None,
                            }
                        })
                        .collect();

                    // Parse rows
                    let mut rows = Vec::new();
                    i += 2;

                    while i < lines.len() && lines[i].contains('|') {
                        let row: Vec<String> = lines[i]
                            .split('|')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();

                        if !row.is_empty() {
                            rows.push(row);
                        }
                        i += 1;
                    }

                    if !headers.is_empty() {
                        tables.push(ParsedTable {
                            headers,
                            rows,
                            alignments,
                        });
                    }
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        tables
    }

    /// Extract links from text
    fn extract_links(&self, text: &str) -> Vec<ParsedLink> {
        let mut links = Vec::new();

        // Markdown links: [text](url "title")
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '[' {
                let mut link_text = String::new();
                let mut found_link = false;

                // Get link text
                while let Some(&next) = chars.peek() {
                    if next == ']' {
                        chars.next();
                        found_link = true;
                        break;
                    }
                    link_text.push(chars.next().expect("char verified by peek"));
                }

                if found_link && chars.peek() == Some(&'(') {
                    chars.next(); // consume (
                    let mut url = String::new();
                    let mut title = None;
                    let mut in_title = false;
                    let mut title_buf = String::new();

                    while let Some(next) = chars.next() {
                        if next == ')' {
                            break;
                        } else if next == '"' && !in_title {
                            in_title = true;
                        } else if next == '"' && in_title {
                            title = Some(title_buf.clone());
                            in_title = false;
                        } else if in_title {
                            title_buf.push(next);
                        } else if next != ' ' || url.is_empty() {
                            url.push(next);
                        }
                    }

                    if !url.is_empty() {
                        links.push(ParsedLink {
                            text: link_text,
                            url: url.trim().to_string(),
                            title,
                        });
                    }
                }
            }
        }

        links
    }

    /// Extract headings from text
    fn extract_headings(&self, text: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        for (line_num, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();

            if trimmed.starts_with('#') {
                let mut level = 0u8;
                for c in trimmed.chars() {
                    if c == '#' {
                        level += 1;
                    } else {
                        break;
                    }
                }

                if level <= 6 {
                    let text = trimmed[level as usize..].trim().to_string();
                    if !text.is_empty() {
                        headings.push(Heading {
                            level,
                            text,
                            line: line_num,
                        });
                    }
                }
            }
        }

        headings
    }

    /// Strip markdown formatting from text
    fn strip_markdown(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove bold/italic
        result = result.replace("**", "");
        result = result.replace("__", "");
        result = result.replace("*", "");
        result = result.replace("_", "");

        // Remove code formatting
        result = result.replace("`", "");

        // Remove headers
        let lines: Vec<&str> = result.lines().collect();
        result = lines
            .iter()
            .map(|line| {
                let trimmed = line.trim_start();
                if trimmed.starts_with('#') {
                    trimmed.trim_start_matches('#').trim()
                } else {
                    *line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Remove link syntax
        // [text](url) -> text
        let mut final_result = String::new();
        let mut in_link_text = false;
        let mut in_link_url = false;
        let mut link_text = String::new();

        for c in result.chars() {
            match c {
                '[' if !in_link_text => {
                    in_link_text = true;
                    link_text.clear();
                }
                ']' if in_link_text => {
                    in_link_text = false;
                }
                '(' if !in_link_text && !link_text.is_empty() => {
                    in_link_url = true;
                    final_result.push_str(&link_text);
                    link_text.clear();
                }
                ')' if in_link_url => {
                    in_link_url = false;
                }
                _ if in_link_text => {
                    link_text.push(c);
                }
                _ if in_link_url => {
                    // Skip URL characters
                }
                _ => {
                    final_result.push(c);
                }
            }
        }

        final_result
    }
}

impl Default for ResponseParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common parsing tasks

/// Extract the first code block from a response
pub fn extract_first_code(response: &str) -> Option<CodeBlock> {
    let parser = ResponseParser::new();
    let parsed = parser.parse(response);
    parsed.code_blocks.into_iter().next()
}

/// Extract all code blocks of a specific language
pub fn extract_code_by_language(response: &str, language: &str) -> Vec<CodeBlock> {
    let parser = ResponseParser::new();
    let parsed = parser.parse(response);
    parsed
        .code_blocks
        .into_iter()
        .filter(|b| b.language.as_deref() == Some(language))
        .collect()
}

/// Extract the first JSON object from a response
pub fn extract_first_json(response: &str) -> Option<serde_json::Value> {
    let parser = ResponseParser::new();
    let parsed = parser.parse(response);
    parsed.json_objects.into_iter().next()
}

/// Try to parse the entire response as JSON
pub fn parse_as_json<T: for<'de> Deserialize<'de>>(response: &str) -> Result<T, serde_json::Error> {
    // First try direct parsing
    if let Ok(result) = serde_json::from_str(response) {
        return Ok(result);
    }

    // Try extracting JSON from code block
    let parser = ResponseParser::new();
    let parsed = parser.parse(response);

    for json in parsed.json_objects {
        if let Ok(result) = serde_json::from_value(json) {
            return Ok(result);
        }
    }

    // Try from code blocks
    for block in parsed.code_blocks {
        if let Ok(result) = serde_json::from_str(&block.code) {
            return Ok(result);
        }
    }

    serde_json::from_str(response) // Return the original error
}

/// Extract plain text from markdown response
pub fn to_plain_text(response: &str) -> String {
    let parser = ResponseParser::new();
    parser.parse(response).text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_blocks() {
        let response = r#"Here's some code:

```rust
fn main() {
    println!("Hello");
}
```

And some Python:

```python
print("Hello")
```
"#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert_eq!(parsed.code_blocks.len(), 2);
        assert_eq!(parsed.code_blocks[0].language, Some("rust".to_string()));
        assert!(parsed.code_blocks[0].code.contains("println!"));
        assert_eq!(parsed.code_blocks[1].language, Some("python".to_string()));
    }

    #[test]
    fn test_extract_json() {
        let response = r#"Here's the data:
{"name": "John", "age": 30}
And more text."#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert_eq!(parsed.json_objects.len(), 1);
        assert_eq!(parsed.json_objects[0]["name"], "John");
    }

    #[test]
    fn test_extract_lists() {
        let response = r#"Todo list:
- Item 1
- Item 2
- Item 3

Numbered list:
1. First
2. Second
3. Third
"#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert!(parsed.lists.len() >= 2);
        assert!(!parsed.lists[0].ordered);
        assert!(parsed.lists.iter().any(|l| l.ordered));
    }

    #[test]
    fn test_extract_table() {
        let response = r#"Here's a table:

| Name | Age |
|------|-----|
| John | 30  |
| Jane | 25  |
"#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert_eq!(parsed.tables.len(), 1);
        assert_eq!(parsed.tables[0].headers, vec!["Name", "Age"]);
        assert_eq!(parsed.tables[0].rows.len(), 2);
    }

    #[test]
    fn test_extract_links() {
        let response = r#"Check out [Google](https://google.com) and [GitHub](https://github.com "Git hosting")."#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert_eq!(parsed.links.len(), 2);
        assert_eq!(parsed.links[0].text, "Google");
        assert_eq!(parsed.links[1].title, Some("Git hosting".to_string()));
    }

    #[test]
    fn test_extract_headings() {
        let response = r#"# Main Title

Some text.

## Section 1

More text.

### Subsection

Even more text.
"#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert_eq!(parsed.headings.len(), 3);
        assert_eq!(parsed.headings[0].level, 1);
        assert_eq!(parsed.headings[1].level, 2);
        assert_eq!(parsed.headings[2].level, 3);
    }

    #[test]
    fn test_checkbox_list() {
        let response = r#"Tasks:
- [ ] Incomplete task
- [x] Completed task
"#;

        let parser = ResponseParser::new();
        let parsed = parser.parse(response);

        assert!(!parsed.lists.is_empty());
        let list = &parsed.lists[0];
        assert_eq!(list.items[0].checkbox, Some(false));
        assert_eq!(list.items[1].checkbox, Some(true));
    }

    #[test]
    fn test_convenience_functions() {
        let response = r#"```json
{"key": "value"}
```"#;

        let json = extract_first_json(response).unwrap();
        assert_eq!(json["key"], "value");

        let code = extract_first_code(response).unwrap();
        assert_eq!(code.language, Some("json".to_string()));
    }

    #[test]
    fn test_to_plain_text() {
        let response = r#"# Hello **World**

This is a [link](https://example.com) and some `code`.
"#;

        let plain = to_plain_text(response);
        assert!(plain.contains("Hello World"));
        assert!(plain.contains("link"));
        assert!(!plain.contains("https://"));
        assert!(!plain.contains("**"));
    }
}
