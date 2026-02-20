//! Table extraction module for detecting and parsing tables from text.
//!
//! Supports multiple table formats including Markdown, ASCII art, HTML,
//! and delimited (CSV/TSV) tables. Extracted tables can be exported to
//! CSV, JSON, Markdown, or accessed as structured data.

use regex::Regex;
use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// Represents a single cell in an extracted table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// The text content of the cell.
    pub content: String,
    /// Number of columns this cell spans.
    pub colspan: usize,
    /// Number of rows this cell spans.
    pub rowspan: usize,
    /// Whether this cell is a header cell.
    pub is_header: bool,
}

impl TableCell {
    /// Create a new regular (non-header) table cell.
    pub fn new(content: &str) -> Self {
        Self {
            content: content.trim().to_string(),
            colspan: 1,
            rowspan: 1,
            is_header: false,
        }
    }

    /// Create a new header table cell.
    pub fn header(content: &str) -> Self {
        Self {
            content: content.trim().to_string(),
            colspan: 1,
            rowspan: 1,
            is_header: true,
        }
    }
}

/// The source format from which a table was extracted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TableSourceFormat {
    /// Markdown pipe-delimited table.
    Markdown,
    /// ASCII art table with box-drawing characters.
    Ascii,
    /// HTML `<table>` element.
    Html,
    /// Tab-separated values.
    Tsv,
    /// Comma-separated values.
    Csv,
}

/// A table extracted from text, containing headers, rows, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTable {
    /// Column headers (empty strings if no headers detected).
    pub headers: Vec<String>,
    /// Data rows, each containing a vector of cells.
    pub rows: Vec<Vec<TableCell>>,
    /// The number of columns in the table.
    pub column_count: usize,
    /// Optional table caption or title.
    pub caption: Option<String>,
    /// The format the table was parsed from.
    pub source_format: TableSourceFormat,
    /// Byte offset where the table starts in the source text.
    pub start_offset: usize,
    /// Byte offset where the table ends in the source text.
    pub end_offset: usize,
}

// ============================================================================
// ExtractedTable Implementation
// ============================================================================

impl ExtractedTable {
    /// Returns the number of data rows (excluding headers).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the content of a specific cell by row and column index.
    pub fn cell(&self, row: usize, col: usize) -> Option<&str> {
        self.rows
            .get(row)
            .and_then(|r| r.get(col))
            .map(|c| c.content.as_str())
    }

    /// Returns all cell contents for a given column index.
    pub fn column(&self, col: usize) -> Vec<&str> {
        self.rows
            .iter()
            .filter_map(|row| row.get(col).map(|c| c.content.as_str()))
            .collect()
    }

    /// Returns all cell contents for a column identified by header name.
    pub fn column_by_name(&self, header: &str) -> Option<Vec<&str>> {
        let col_idx = self.headers.iter().position(|h| h == header)?;
        Some(self.column(col_idx))
    }

    /// Exports the table to CSV format with proper quoting.
    pub fn to_csv(&self) -> String {
        let mut output = String::new();

        // Write headers
        if !self.headers.is_empty() {
            let header_line: Vec<String> = self.headers.iter().map(|h| csv_quote(h)).collect();
            output.push_str(&header_line.join(","));
            output.push('\n');
        }

        // Write data rows
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|c| csv_quote(&c.content)).collect();
            output.push_str(&cells.join(","));
            output.push('\n');
        }

        output
    }

    /// Exports the table to JSON format as an array of objects using headers as keys.
    pub fn to_json(&self) -> String {
        let mut records: Vec<serde_json::Map<String, serde_json::Value>> = Vec::new();

        for row in &self.rows {
            let mut obj = serde_json::Map::new();
            for (i, cell) in row.iter().enumerate() {
                let key = if i < self.headers.len() {
                    self.headers[i].clone()
                } else {
                    format!("column_{}", i)
                };
                obj.insert(key, serde_json::Value::String(cell.content.clone()));
            }
            records.push(obj);
        }

        serde_json::to_string_pretty(&records).unwrap_or_else(|_| "[]".to_string())
    }

    /// Exports the table to Markdown format.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Determine column widths
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len().max(3)).collect();
        // Extend widths if rows have more columns
        while widths.len() < self.column_count {
            widths.push(3);
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.content.len());
                }
            }
        }

        // Write header row
        output.push('|');
        for (i, header) in self.headers.iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(3);
            output.push_str(&format!(" {:<width$} |", header, width = w));
        }
        output.push('\n');

        // Write separator row
        output.push('|');
        for &w in &widths {
            output.push_str(&format!(" {} |", "-".repeat(w)));
        }
        output.push('\n');

        // Write data rows
        for row in &self.rows {
            output.push('|');
            for (i, cell) in row.iter().enumerate() {
                let w = widths.get(i).copied().unwrap_or(3);
                output.push_str(&format!(" {:<width$} |", cell.content, width = w));
            }
            output.push('\n');
        }

        output
    }

    /// Returns a simple 2D grid of strings including headers as the first row.
    pub fn to_grid(&self) -> Vec<Vec<String>> {
        let mut grid: Vec<Vec<String>> = Vec::new();

        if !self.headers.is_empty() {
            grid.push(self.headers.clone());
        }

        for row in &self.rows {
            let string_row: Vec<String> = row.iter().map(|c| c.content.clone()).collect();
            grid.push(string_row);
        }

        grid
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the table extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableExtractorConfig {
    /// Whether to detect Markdown pipe tables.
    pub detect_markdown: bool,
    /// Whether to detect ASCII art tables.
    pub detect_ascii: bool,
    /// Whether to detect HTML tables.
    pub detect_html: bool,
    /// Whether to detect delimited (CSV/TSV) tables.
    pub detect_delimited: bool,
    /// Minimum number of columns for a valid table.
    pub min_columns: usize,
    /// Minimum number of data rows for a valid table.
    pub min_rows: usize,
    /// Whether to treat the first row as a header row.
    pub first_row_is_header: bool,
}

impl Default for TableExtractorConfig {
    fn default() -> Self {
        Self {
            detect_markdown: true,
            detect_ascii: true,
            detect_html: true,
            detect_delimited: false,
            min_columns: 2,
            min_rows: 1,
            first_row_is_header: true,
        }
    }
}

// ============================================================================
// TableExtractor
// ============================================================================

/// Extracts and parses tables from text in various formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableExtractor {
    /// Configuration controlling detection behavior.
    pub config: TableExtractorConfig,
}

impl TableExtractor {
    /// Create a new `TableExtractor` with the given configuration.
    pub fn new(config: TableExtractorConfig) -> Self {
        Self { config }
    }

    /// Detect and extract all tables from the given text.
    ///
    /// Scans for Markdown, ASCII, HTML, and delimited tables based on
    /// the current configuration.
    pub fn extract_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();

        if self.config.detect_markdown {
            tables.extend(self.extract_markdown_tables(text));
        }

        if self.config.detect_ascii {
            tables.extend(self.extract_ascii_tables(text));
        }

        if self.config.detect_html {
            tables.extend(self.extract_html_tables(text));
        }

        if self.config.detect_delimited {
            tables.extend(self.extract_delimited_tables(text));
        }

        // Sort by start offset
        tables.sort_by_key(|t| t.start_offset);

        tables
    }

    /// Extract tables specifically from HTML content.
    pub fn extract_html_tables(&self, html: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let table_re = Regex::new(r"(?is)<table[^>]*>(.*?)</table>").expect("valid regex");

        for cap in table_re.captures_iter(html) {
            let full_match = cap.get(0).expect("capture group 0");
            let table_content = &cap[1];
            if let Some(table) =
                self.parse_html_table(table_content, full_match.start(), full_match.end())
            {
                if table.column_count >= self.config.min_columns
                    && table.rows.len() >= self.config.min_rows
                {
                    tables.push(table);
                }
            }
        }

        tables
    }

    /// Parse a single Markdown table from text.
    ///
    /// Expects the input to be the raw lines of a Markdown table including
    /// the header, separator, and data rows.
    pub fn parse_markdown_table(&self, table_text: &str) -> Option<ExtractedTable> {
        self.parse_markdown_table_at(table_text, 0)
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    /// Extract all Markdown tables from text.
    fn extract_markdown_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            // Look for a line containing pipes that could be a table row
            if self.is_markdown_table_row(lines[i]) {
                let start_line = i;
                let mut table_lines = Vec::new();
                table_lines.push(lines[i]);
                i += 1;

                // Collect consecutive table rows
                while i < lines.len() && self.is_markdown_table_row(lines[i]) {
                    table_lines.push(lines[i]);
                    i += 1;
                }

                // Skip if surrounded by ASCII borders (belongs to an ASCII table)
                let has_border_before =
                    start_line > 0 && self.is_ascii_border_row(lines[start_line - 1]);
                let has_border_after = i < lines.len() && self.is_ascii_border_row(lines[i]);

                // Need at least 2 lines (header + separator, or header + data)
                if table_lines.len() >= 2 && !has_border_before && !has_border_after {
                    let table_text = table_lines.join("\n");
                    let start_offset = text
                        .lines()
                        .take(start_line)
                        .map(|l| l.len() + 1)
                        .sum::<usize>();
                    let end_offset = start_offset + table_text.len();

                    if let Some(table) = self.parse_markdown_table_at(&table_text, start_offset) {
                        if table.column_count >= self.config.min_columns
                            && table.rows.len() >= self.config.min_rows
                        {
                            let mut t = table;
                            t.end_offset = end_offset;
                            tables.push(t);
                        }
                    }
                }
            } else {
                i += 1;
            }
        }

        tables
    }

    /// Parse a Markdown table starting at a given byte offset.
    fn parse_markdown_table_at(
        &self,
        table_text: &str,
        start_offset: usize,
    ) -> Option<ExtractedTable> {
        let lines: Vec<&str> = table_text.lines().collect();
        if lines.len() < 2 {
            return None;
        }

        let mut headers = Vec::new();
        let mut rows: Vec<Vec<TableCell>> = Vec::new();
        let mut data_start = 0;

        // Check if second line is a separator row
        let has_separator = lines.len() > 1 && self.is_separator_row(lines[1]);

        if has_separator && self.config.first_row_is_header {
            // First line is headers
            headers = self.parse_pipe_row(lines[0]);
            data_start = 2;
        } else if self.config.first_row_is_header {
            // Treat first row as header even without separator
            headers = self.parse_pipe_row(lines[0]);
            data_start = 1;
        }

        let column_count = if !headers.is_empty() {
            headers.len()
        } else {
            // Determine from first data row
            self.count_pipe_columns(lines[data_start])
        };

        // Parse data rows (skip separator rows)
        for line in &lines[data_start..] {
            if self.is_separator_row(line) {
                continue;
            }
            let cells: Vec<TableCell> = self
                .parse_pipe_row(line)
                .into_iter()
                .map(|s| TableCell::new(&s))
                .collect();
            if !cells.is_empty() {
                rows.push(cells);
            }
        }

        if rows.is_empty() && headers.is_empty() {
            return None;
        }

        Some(ExtractedTable {
            headers,
            rows,
            column_count,
            caption: None,
            source_format: TableSourceFormat::Markdown,
            start_offset,
            end_offset: start_offset + table_text.len(),
        })
    }

    /// Extract ASCII art tables from text.
    fn extract_ascii_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            if self.is_ascii_border_row(lines[i]) {
                let start_line = i;
                let mut table_lines = Vec::new();
                table_lines.push(lines[i]);
                i += 1;

                // Collect lines until we run out of table-like lines (borders or pipe rows)
                let mut last_border_idx = 0; // index within table_lines of last border
                while i < lines.len() {
                    let line = lines[i];
                    if self.is_ascii_border_row(line) {
                        table_lines.push(line);
                        last_border_idx = table_lines.len() - 1;
                        i += 1;
                        // If we've seen data rows between borders, check if next line continues the table
                        if last_border_idx >= 2 {
                            // Peek: if next line is neither a border nor a pipe row, stop
                            if i >= lines.len()
                                || (!self.is_ascii_border_row(lines[i]) && !lines[i].contains('|'))
                            {
                                break;
                            }
                        }
                    } else if line.contains('|') {
                        table_lines.push(line);
                        i += 1;
                    } else {
                        break;
                    }
                }
                // Trim to the last border row (discard trailing non-border rows)
                table_lines.truncate(last_border_idx + 1);

                if table_lines.len() >= 3 {
                    let start_offset = text
                        .lines()
                        .take(start_line)
                        .map(|l| l.len() + 1)
                        .sum::<usize>();
                    let table_text = table_lines.join("\n");
                    let end_offset = start_offset + table_text.len();

                    if let Some(table) =
                        self.parse_ascii_table(&table_lines, start_offset, end_offset)
                    {
                        if table.column_count >= self.config.min_columns
                            && table.rows.len() >= self.config.min_rows
                        {
                            tables.push(table);
                        }
                    }
                }
            } else {
                i += 1;
            }
        }

        tables
    }

    /// Parse an ASCII art table from collected lines.
    fn parse_ascii_table(
        &self,
        lines: &[&str],
        start_offset: usize,
        end_offset: usize,
    ) -> Option<ExtractedTable> {
        let mut headers = Vec::new();
        let mut rows: Vec<Vec<TableCell>> = Vec::new();
        let mut found_first_separator = false;
        let mut header_done = false;

        for line in lines {
            if self.is_ascii_border_row(line) {
                if found_first_separator && !header_done && !rows.is_empty() {
                    // The rows collected so far are actually headers
                    if self.config.first_row_is_header {
                        if let Some(first_row) = rows.pop() {
                            headers = first_row.into_iter().map(|c| c.content).collect();
                        }
                    }
                    header_done = true;
                }
                found_first_separator = true;
                continue;
            }

            // Parse cell content from lines like "| foo | bar |"
            let cells = self.parse_ascii_data_row(line);
            if !cells.is_empty() {
                rows.push(cells);
            }
        }

        // If we never found a middle separator, treat the first data row as headers
        if !header_done && self.config.first_row_is_header && !rows.is_empty() {
            let first_row = rows.remove(0);
            headers = first_row.into_iter().map(|c| c.content).collect();
        }

        let column_count = if !headers.is_empty() {
            headers.len()
        } else if let Some(first) = rows.first() {
            first.len()
        } else {
            return None;
        };

        Some(ExtractedTable {
            headers,
            rows,
            column_count,
            caption: None,
            source_format: TableSourceFormat::Ascii,
            start_offset,
            end_offset,
        })
    }

    /// Parse a single HTML table element's inner content.
    fn parse_html_table(
        &self,
        table_html: &str,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<ExtractedTable> {
        let mut headers = Vec::new();
        let mut rows: Vec<Vec<TableCell>> = Vec::new();

        // Extract caption if present
        let caption_re = Regex::new(r"(?is)<caption[^>]*>(.*?)</caption>").expect("valid regex");
        let caption = caption_re
            .captures(table_html)
            .map(|c| strip_html_tags(&c[1]).trim().to_string());

        // Find all table rows
        let tr_re = Regex::new(r"(?is)<tr[^>]*>(.*?)</tr>").expect("valid regex");
        let th_re = Regex::new(r"(?is)<th[^>]*>(.*?)</th>").expect("valid regex");
        let td_re = Regex::new(r"(?is)<td[^>]*>(.*?)</td>").expect("valid regex");
        let colspan_re = Regex::new(r#"(?i)colspan\s*=\s*["']?(\d+)["']?"#).expect("valid regex");
        let rowspan_re = Regex::new(r#"(?i)rowspan\s*=\s*["']?(\d+)["']?"#).expect("valid regex");

        for tr_cap in tr_re.captures_iter(table_html) {
            let row_content = &tr_cap[1];
            let _tr_tag = tr_cap.get(0).expect("capture group 0").as_str();

            // Check if this row contains <th> elements
            let has_th = th_re.is_match(row_content);

            if has_th && headers.is_empty() {
                // This is a header row
                for th_cap in th_re.captures_iter(row_content) {
                    let content = strip_html_tags(&th_cap[1]).trim().to_string();
                    headers.push(content);
                }
            } else {
                // This is a data row - parse <td> and <th> cells
                let mut cells = Vec::new();
                let td_cell_re = Regex::new(r"(?is)<td([^>]*)>(.*?)</td>").expect("valid regex");
                let th_cell_re = Regex::new(r"(?is)<th([^>]*)>(.*?)</th>").expect("valid regex");

                // Collect all cell matches with their positions for ordering
                let mut cell_matches: Vec<(usize, String, String, bool)> = Vec::new();

                for cell_cap in td_cell_re.captures_iter(row_content) {
                    let pos = cell_cap.get(0).expect("capture group 0").start();
                    let attrs = cell_cap[1].to_string();
                    let content = strip_html_tags(&cell_cap[2]).trim().to_string();
                    cell_matches.push((pos, attrs, content, false));
                }
                for cell_cap in th_cell_re.captures_iter(row_content) {
                    let pos = cell_cap.get(0).expect("capture group 0").start();
                    let attrs = cell_cap[1].to_string();
                    let content = strip_html_tags(&cell_cap[2]).trim().to_string();
                    cell_matches.push((pos, attrs, content, true));
                }

                // Sort by position to maintain document order
                cell_matches.sort_by_key(|(pos, _, _, _)| *pos);

                for (_pos, attrs, content, is_header) in &cell_matches {
                    let colspan = colspan_re
                        .captures(attrs)
                        .and_then(|c| c[1].parse::<usize>().ok())
                        .unwrap_or(1);
                    let rowspan = rowspan_re
                        .captures(attrs)
                        .and_then(|c| c[1].parse::<usize>().ok())
                        .unwrap_or(1);

                    let mut cell = if *is_header {
                        TableCell::header(content)
                    } else {
                        TableCell::new(content)
                    };
                    cell.colspan = colspan;
                    cell.rowspan = rowspan;
                    cells.push(cell);
                }

                // If no cells found with td/th regex, try just td
                if cells.is_empty() {
                    for td_cap in td_re.captures_iter(row_content) {
                        let content = strip_html_tags(&td_cap[1]).trim().to_string();
                        cells.push(TableCell::new(&content));
                    }
                }

                if !cells.is_empty() {
                    rows.push(cells);
                }
            }
        }

        // If no explicit headers found but first_row_is_header, promote first row
        if headers.is_empty() && self.config.first_row_is_header && !rows.is_empty() {
            let first_row = rows.remove(0);
            headers = first_row.into_iter().map(|c| c.content).collect();
        }

        let column_count = if !headers.is_empty() {
            headers.len()
        } else if let Some(first) = rows.first() {
            first.len()
        } else {
            return None;
        };

        Some(ExtractedTable {
            headers,
            rows,
            column_count,
            caption,
            source_format: TableSourceFormat::Html,
            start_offset,
            end_offset,
        })
    }

    /// Extract delimited (CSV/TSV) tables from text.
    fn extract_delimited_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        if lines.is_empty() {
            return tables;
        }

        // Detect delimiter: tab or comma
        let first_line = lines[0];
        let (delimiter, format) = if first_line.contains('\t') {
            ('\t', TableSourceFormat::Tsv)
        } else if first_line.contains(',') {
            (',', TableSourceFormat::Csv)
        } else {
            return tables;
        };

        let mut headers = Vec::new();
        let mut rows: Vec<Vec<TableCell>> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let fields = split_delimited(line, delimiter);
            if i == 0 && self.config.first_row_is_header {
                headers = fields;
            } else {
                let cells: Vec<TableCell> =
                    fields.into_iter().map(|f| TableCell::new(&f)).collect();
                if !cells.is_empty() {
                    rows.push(cells);
                }
            }
        }

        let column_count = if !headers.is_empty() {
            headers.len()
        } else if let Some(first) = rows.first() {
            first.len()
        } else {
            return tables;
        };

        if column_count >= self.config.min_columns && rows.len() >= self.config.min_rows {
            tables.push(ExtractedTable {
                headers,
                rows,
                column_count,
                caption: None,
                source_format: format,
                start_offset: 0,
                end_offset: text.len(),
            });
        }

        tables
    }

    /// Determine whether a line is a Markdown separator row like `|---|---|`.
    fn is_separator_row(&self, line: &str) -> bool {
        let trimmed = line.trim();
        if !trimmed.contains('|') {
            return false;
        }
        // After removing pipes, spaces, colons, and dashes, should be empty
        let cleaned: String = trimmed
            .chars()
            .filter(|c| *c != '|' && *c != '-' && *c != ':' && *c != ' ')
            .collect();
        cleaned.is_empty() && trimmed.contains('-')
    }

    /// Detect headers from the first row of data if not already set.
    #[allow(dead_code)]
    fn detect_headers(&self, rows: &mut Vec<Vec<TableCell>>) -> Vec<String> {
        if self.config.first_row_is_header && !rows.is_empty() {
            let first_row = rows.remove(0);
            first_row.into_iter().map(|c| c.content).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if a line looks like a Markdown table row (contains pipes).
    fn is_markdown_table_row(&self, line: &str) -> bool {
        let trimmed = line.trim();
        trimmed.contains('|') && !trimmed.starts_with("```")
    }

    /// Parse a pipe-delimited row into a vector of cell contents.
    fn parse_pipe_row(&self, line: &str) -> Vec<String> {
        let trimmed = line.trim();
        // Remove leading and trailing pipes
        let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
        inner.split('|').map(|s| s.trim().to_string()).collect()
    }

    /// Count the number of columns in a pipe-delimited line.
    fn count_pipe_columns(&self, line: &str) -> usize {
        self.parse_pipe_row(line).len()
    }

    /// Check if a line is an ASCII table border (e.g., `+---+---+` or `╔═══╗`).
    fn is_ascii_border_row(&self, line: &str) -> bool {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return false;
        }

        // Check for +---+---+ style
        if trimmed.starts_with('+') && trimmed.ends_with('+') {
            let inner: String = trimmed
                .chars()
                .filter(|c| *c != '+' && *c != '-' && *c != '=' && *c != ' ')
                .collect();
            if inner.is_empty() {
                return true;
            }
        }

        // Check for Unicode box-drawing borders
        let box_chars = [
            '╔', '╗', '╚', '╝', '═', '╤', '╧', '╟', '╢', '║', '─', '┌', '┐', '└', '┘', '├', '┤',
            '┬', '┴', '┼', '╠', '╣', '╦', '╩', '╬',
        ];
        let first_char = trimmed.chars().next().unwrap_or(' ');
        if box_chars.contains(&first_char) {
            return true;
        }

        false
    }

    /// Parse a data row from an ASCII table (between borders).
    fn parse_ascii_data_row(&self, line: &str) -> Vec<TableCell> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // Remove leading/trailing box-drawing or pipe characters
        let stripped = trimmed
            .trim_start_matches(|c: char| c == '|' || c == '║' || c == '│')
            .trim_end_matches(|c: char| c == '|' || c == '║' || c == '│');

        if stripped.is_empty() {
            return Vec::new();
        }

        // Split on pipe or box-drawing vertical separators
        let parts: Vec<&str> = stripped
            .split(|c: char| c == '|' || c == '║' || c == '│')
            .collect();

        parts
            .into_iter()
            .map(|s| TableCell::new(s))
            .filter(|c| !c.content.is_empty() || true) // keep all cells
            .collect()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Quote a field for CSV output. Wraps in double-quotes if the field contains
/// commas, double-quotes, or newlines.
fn csv_quote(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}

/// Strip HTML tags from a string, leaving only text content.
fn strip_html_tags(html: &str) -> String {
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    let result = tag_re.replace_all(html, "");
    // Also decode basic HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Split a line by a delimiter, respecting quoted fields.
fn split_delimited(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '"' {
            if in_quotes {
                // Check for escaped quote
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
        } else if c == delimiter && !in_quotes {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current.trim().to_string());

    fields
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_markdown_table() {
        let input = "| Name   | Age | City     |\n\
                     |--------|-----|----------|\n\
                     | Alice  | 30  | New York |\n\
                     | Bob    | 25  | London   |\n\
                     | Charlie| 35  | Paris    |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let table = extractor.parse_markdown_table(input).unwrap();

        assert_eq!(table.headers, vec!["Name", "Age", "City"]);
        assert_eq!(table.row_count(), 3);
        assert_eq!(table.column_count, 3);
        assert_eq!(table.cell(0, 0), Some("Alice"));
        assert_eq!(table.cell(1, 1), Some("25"));
        assert_eq!(table.cell(2, 2), Some("Paris"));
        assert_eq!(table.source_format, TableSourceFormat::Markdown);

        // Test column access
        let ages = table.column(1);
        assert_eq!(ages, vec!["30", "25", "35"]);

        // Test column by name
        let cities = table.column_by_name("City").unwrap();
        assert_eq!(cities, vec!["New York", "London", "Paris"]);
    }

    #[test]
    fn test_parse_ascii_table() {
        let input = "+--------+-----+----------+\n\
                     | Name   | Age | City     |\n\
                     +--------+-----+----------+\n\
                     | Alice  | 30  | New York |\n\
                     | Bob    | 25  | London   |\n\
                     +--------+-----+----------+";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let tables = extractor.extract_tables(input);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.source_format, TableSourceFormat::Ascii);
        assert_eq!(table.headers, vec!["Name", "Age", "City"]);
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.cell(0, 0), Some("Alice"));
        assert_eq!(table.cell(1, 2), Some("London"));
    }

    #[test]
    fn test_parse_html_table() {
        let input = r#"<table>
            <caption>Employee List</caption>
            <tr><th>Name</th><th>Department</th><th>Salary</th></tr>
            <tr><td>Alice</td><td>Engineering</td><td>100000</td></tr>
            <tr><td>Bob</td><td>Marketing</td><td>85000</td></tr>
            <tr><td colspan="2">Charlie (Manager)</td><td>120000</td></tr>
        </table>"#;

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let tables = extractor.extract_html_tables(input);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.source_format, TableSourceFormat::Html);
        assert_eq!(table.headers, vec!["Name", "Department", "Salary"]);
        assert_eq!(table.row_count(), 3);
        assert_eq!(table.caption, Some("Employee List".to_string()));
        assert_eq!(table.cell(0, 0), Some("Alice"));
        assert_eq!(table.cell(1, 1), Some("Marketing"));

        // Check colspan on the third row
        assert_eq!(table.rows[2][0].colspan, 2);
        assert_eq!(table.rows[2][0].content, "Charlie (Manager)");
    }

    #[test]
    fn test_csv_export() {
        let input = "| Product  | Price | Quantity |\n\
                     |----------|-------|----------|\n\
                     | Widget   | 9.99  | 100      |\n\
                     | Gadget   | 24.99 | 50       |\n\
                     | Thing, A | 5.00  | 200      |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let table = extractor.parse_markdown_table(input).unwrap();
        let csv = table.to_csv();

        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "Product,Price,Quantity");
        assert_eq!(lines[1], "Widget,9.99,100");
        assert_eq!(lines[2], "Gadget,24.99,50");
        // Field containing a comma should be quoted
        assert_eq!(lines[3], "\"Thing, A\",5.00,200");
    }

    #[test]
    fn test_json_export() {
        let input = "| Name  | Score |\n\
                     |-------|-------|\n\
                     | Alice | 95    |\n\
                     | Bob   | 87    |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let table = extractor.parse_markdown_table(input).unwrap();
        let json = table.to_json();

        let parsed: Vec<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["Name"], "Alice");
        assert_eq!(parsed[0]["Score"], "95");
        assert_eq!(parsed[1]["Name"], "Bob");
        assert_eq!(parsed[1]["Score"], "87");
    }

    #[test]
    fn test_to_markdown_roundtrip() {
        let input = "| Name  | Age |\n\
                     |-------|-----|\n\
                     | Alice | 30  |\n\
                     | Bob   | 25  |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let table = extractor.parse_markdown_table(input).unwrap();
        let md = table.to_markdown();

        // Parse the output again
        let table2 = extractor.parse_markdown_table(&md).unwrap();
        assert_eq!(table2.headers, table.headers);
        assert_eq!(table2.row_count(), table.row_count());
        assert_eq!(table2.cell(0, 0), table.cell(0, 0));
        assert_eq!(table2.cell(1, 1), table.cell(1, 1));
    }

    #[test]
    fn test_to_grid() {
        let input = "| A | B |\n\
                     |---|---|\n\
                     | 1 | 2 |\n\
                     | 3 | 4 |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let table = extractor.parse_markdown_table(input).unwrap();
        let grid = table.to_grid();

        assert_eq!(grid.len(), 3); // 1 header + 2 data rows
        assert_eq!(grid[0], vec!["A", "B"]);
        assert_eq!(grid[1], vec!["1", "2"]);
        assert_eq!(grid[2], vec!["3", "4"]);
    }

    #[test]
    fn test_extract_multiple_tables() {
        let input = "Some text here.\n\n\
                     | Col1 | Col2 |\n\
                     |------|------|\n\
                     | A    | B    |\n\n\
                     More text in between.\n\n\
                     | X | Y | Z |\n\
                     |---|---|---|\n\
                     | 1 | 2 | 3 |\n\
                     | 4 | 5 | 6 |";

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let tables = extractor.extract_tables(input);

        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].column_count, 2);
        assert_eq!(tables[1].column_count, 3);
        assert_eq!(tables[1].row_count(), 2);
    }

    #[test]
    fn test_delimited_csv_extraction() {
        let input = "Name,Age,City\n\
                     Alice,30,New York\n\
                     Bob,25,London";

        let mut config = TableExtractorConfig::default();
        config.detect_delimited = true;
        config.detect_markdown = false;
        let extractor = TableExtractor::new(config);
        let tables = extractor.extract_tables(input);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.source_format, TableSourceFormat::Csv);
        assert_eq!(table.headers, vec!["Name", "Age", "City"]);
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.cell(0, 0), Some("Alice"));
    }

    #[test]
    fn test_html_with_entities_and_spans() {
        let input = r#"<table>
            <tr><th>Item</th><th>Price</th></tr>
            <tr><td>Books &amp; Magazines</td><td>$19.99</td></tr>
            <tr><td rowspan="2">Electronics</td><td>$299</td></tr>
            <tr><td>$499</td></tr>
        </table>"#;

        let extractor = TableExtractor::new(TableExtractorConfig::default());
        let tables = extractor.extract_html_tables(input);

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.cell(0, 0), Some("Books & Magazines"));
        assert_eq!(table.rows[1][0].rowspan, 2);
    }
}
