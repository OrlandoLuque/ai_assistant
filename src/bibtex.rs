//! BibTeX parser and generator
//!
//! Provides parsing of `.bib` files into structured entries and generation of
//! BibTeX output from academic papers. Includes security sanitization against
//! LaTeX injection attacks.
//!
//! # Security
//!
//! All BibTeX field values are sanitized to strip dangerous LaTeX commands:
//! `\input`, `\include`, `\write18`, `\immediate`, `\openout`, `\csname`.
//!
//! # Limits
//!
//! - Max file size: 10 MB
//! - Max entries: 10,000
//! - Max field value length: 10,000 characters

use std::collections::HashMap;

// =============================================================================
// Types
// =============================================================================

/// BibTeX entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BibEntryType {
    /// Journal article
    Article,
    /// Book
    Book,
    /// Conference/workshop paper
    InProceedings,
    /// Chapter in a collection
    InCollection,
    /// PhD or Master's thesis
    Thesis,
    /// Technical report
    TechReport,
    /// Miscellaneous
    Misc,
    /// Unpublished work
    Unpublished,
    /// Online resource
    Online,
}

impl BibEntryType {
    /// Parse from a BibTeX type string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "article" => Self::Article,
            "book" => Self::Book,
            "inproceedings" | "conference" => Self::InProceedings,
            "incollection" => Self::InCollection,
            "phdthesis" | "mastersthesis" | "thesis" => Self::Thesis,
            "techreport" => Self::TechReport,
            "unpublished" => Self::Unpublished,
            "online" | "electronic" => Self::Online,
            _ => Self::Misc,
        }
    }

    /// BibTeX type name string.
    pub fn as_bibtex_str(&self) -> &'static str {
        match self {
            Self::Article => "article",
            Self::Book => "book",
            Self::InProceedings => "inproceedings",
            Self::InCollection => "incollection",
            Self::Thesis => "phdthesis",
            Self::TechReport => "techreport",
            Self::Misc => "misc",
            Self::Unpublished => "unpublished",
            Self::Online => "online",
        }
    }
}

impl std::fmt::Display for BibEntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_bibtex_str())
    }
}

/// A single BibTeX entry with type, citation key, and fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BibEntry {
    /// Entry type (article, book, etc.)
    pub entry_type: BibEntryType,
    /// Citation key (e.g., "vaswani2017attention")
    pub cite_key: String,
    /// Fields (e.g., "title" -> "Attention Is All You Need")
    pub fields: HashMap<String, String>,
}

impl BibEntry {
    /// Create a new entry with the given type and key.
    pub fn new(entry_type: BibEntryType, cite_key: &str) -> Self {
        Self {
            entry_type,
            cite_key: cite_key.to_string(),
            fields: HashMap::new(),
        }
    }

    /// Set a field value (sanitized).
    pub fn set_field(&mut self, key: &str, value: &str) {
        let sanitized = sanitize_latex(value);
        if sanitized.len() <= MAX_FIELD_LENGTH {
            self.fields.insert(key.to_lowercase(), sanitized);
        }
    }

    /// Get a field value.
    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.fields.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Get the title field.
    pub fn title(&self) -> Option<&str> {
        self.get_field("title")
    }

    /// Get the author field.
    pub fn author(&self) -> Option<&str> {
        self.get_field("author")
    }

    /// Get the year field.
    pub fn year(&self) -> Option<&str> {
        self.get_field("year")
    }

    /// Generate BibTeX string for this entry.
    pub fn to_bibtex(&self) -> String {
        let mut result = format!("@{}{{{},\n", self.entry_type, self.cite_key);

        // Sort fields for deterministic output
        let mut sorted_fields: Vec<_> = self.fields.iter().collect();
        sorted_fields.sort_by_key(|(k, _)| (*k).clone());

        for (i, (key, value)) in sorted_fields.iter().enumerate() {
            result.push_str(&format!("  {} = {{{}}}", key, value));
            if i < sorted_fields.len() - 1 {
                result.push(',');
            }
            result.push('\n');
        }
        result.push_str("}\n");
        result
    }
}

// =============================================================================
// Parser
// =============================================================================

/// Maximum .bib file size in bytes (10 MB).
const MAX_FILE_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of entries in a single .bib file.
const MAX_ENTRIES: usize = 10_000;

/// Maximum length of a single field value.
const MAX_FIELD_LENGTH: usize = 10_000;

/// Errors from BibTeX parsing.
#[derive(Debug)]
#[non_exhaustive]
pub enum BibParseError {
    /// File too large
    FileTooLarge(usize),
    /// Too many entries
    TooManyEntries(usize),
    /// Syntax error at position
    SyntaxError(String),
}

impl std::fmt::Display for BibParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileTooLarge(size) => {
                write!(
                    f,
                    "BibTeX file too large: {} bytes (max {})",
                    size, MAX_FILE_SIZE
                )
            }
            Self::TooManyEntries(count) => {
                write!(
                    f,
                    "Too many BibTeX entries: {} (max {})",
                    count, MAX_ENTRIES
                )
            }
            Self::SyntaxError(msg) => write!(f, "BibTeX syntax error: {}", msg),
        }
    }
}

impl std::error::Error for BibParseError {}

/// BibTeX parser.
///
/// Parses `.bib` file content into structured `BibEntry` values.
/// Handles brace nesting, `#` concatenation, and LaTeX accent commands.
pub struct BibParser;

impl BibParser {
    /// Parse a BibTeX string into entries.
    pub fn parse(input: &str) -> Result<Vec<BibEntry>, BibParseError> {
        if input.len() > MAX_FILE_SIZE {
            return Err(BibParseError::FileTooLarge(input.len()));
        }

        let mut entries = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        let mut pos = 0;

        while pos < len {
            // Skip until we find @
            if chars[pos] != '@' {
                pos += 1;
                continue;
            }
            pos += 1; // skip @

            // Skip whitespace
            while pos < len && chars[pos].is_whitespace() {
                pos += 1;
            }

            // Read entry type
            let type_start = pos;
            while pos < len && chars[pos].is_alphanumeric() {
                pos += 1;
            }
            let entry_type_str: String = chars[type_start..pos].iter().collect();

            // Skip comments and preambles
            let lower_type = entry_type_str.to_lowercase();
            if lower_type == "comment" || lower_type == "preamble" || lower_type == "string" {
                // Skip until matching brace
                if let Some(end) = Self::skip_braced_block(&chars, pos) {
                    pos = end;
                }
                continue;
            }

            // Skip whitespace
            while pos < len && chars[pos].is_whitespace() {
                pos += 1;
            }

            // Expect opening brace or paren
            if pos >= len || (chars[pos] != '{' && chars[pos] != '(') {
                pos += 1;
                continue;
            }
            let close_char = if chars[pos] == '{' { '}' } else { ')' };
            pos += 1;

            // Skip whitespace
            while pos < len && chars[pos].is_whitespace() {
                pos += 1;
            }

            // Read citation key (until comma or whitespace)
            let key_start = pos;
            while pos < len
                && chars[pos] != ','
                && chars[pos] != close_char
                && !chars[pos].is_whitespace()
            {
                pos += 1;
            }
            let cite_key: String = chars[key_start..pos].iter().collect();

            if cite_key.is_empty() {
                continue;
            }

            // Skip comma
            if pos < len && chars[pos] == ',' {
                pos += 1;
            }

            let entry_type = BibEntryType::from_str_loose(&entry_type_str);
            let mut entry = BibEntry::new(entry_type, &cite_key);

            // Parse fields until closing brace/paren
            loop {
                // Skip whitespace
                while pos < len && chars[pos].is_whitespace() {
                    pos += 1;
                }

                if pos >= len || chars[pos] == close_char {
                    pos += 1;
                    break;
                }

                // Read field name
                let fname_start = pos;
                while pos < len
                    && chars[pos] != '='
                    && !chars[pos].is_whitespace()
                    && chars[pos] != close_char
                {
                    pos += 1;
                }
                let field_name: String = chars[fname_start..pos].iter().collect();
                let field_name = field_name.trim_matches(',').to_string();

                if field_name.is_empty() {
                    pos += 1;
                    continue;
                }

                // Skip whitespace and =
                while pos < len && (chars[pos].is_whitespace() || chars[pos] == '=') {
                    pos += 1;
                }

                // Read field value
                if pos >= len {
                    break;
                }

                let value = if chars[pos] == '{' {
                    // Braced value — handle nesting
                    pos += 1;
                    let mut depth = 1;
                    let val_start = pos;
                    while pos < len && depth > 0 {
                        if chars[pos] == '{' {
                            depth += 1;
                        } else if chars[pos] == '}' {
                            depth -= 1;
                        }
                        if depth > 0 {
                            pos += 1;
                        }
                    }
                    let val: String = chars[val_start..pos].iter().collect();
                    pos += 1; // skip closing }
                    val
                } else if chars[pos] == '"' {
                    // Quoted value
                    pos += 1;
                    let val_start = pos;
                    while pos < len && chars[pos] != '"' {
                        pos += 1;
                    }
                    let val: String = chars[val_start..pos].iter().collect();
                    if pos < len {
                        pos += 1; // skip closing "
                    }
                    val
                } else {
                    // Bare value (number or string name)
                    let val_start = pos;
                    while pos < len
                        && chars[pos] != ','
                        && chars[pos] != close_char
                        && !chars[pos].is_whitespace()
                    {
                        pos += 1;
                    }
                    let val: String = chars[val_start..pos].iter().collect();
                    val
                };

                // Skip trailing comma
                while pos < len && (chars[pos] == ',' || chars[pos].is_whitespace()) {
                    pos += 1;
                }

                if !field_name.is_empty() && !value.is_empty() {
                    entry.set_field(&field_name, &value);
                }
            }

            entries.push(entry);

            if entries.len() > MAX_ENTRIES {
                return Err(BibParseError::TooManyEntries(entries.len()));
            }
        }

        Ok(entries)
    }

    /// Skip a braced block starting at `pos`, returning the position after the closing brace.
    fn skip_braced_block(chars: &[char], mut pos: usize) -> Option<usize> {
        let len = chars.len();
        while pos < len && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= len {
            return None;
        }
        pos += 1;
        let mut depth = 1;
        while pos < len && depth > 0 {
            if chars[pos] == '{' {
                depth += 1;
            } else if chars[pos] == '}' {
                depth -= 1;
            }
            pos += 1;
        }
        Some(pos)
    }
}

// =============================================================================
// Generator
// =============================================================================

/// BibTeX generator — creates `.bib` content from entries or academic papers.
pub struct BibGenerator;

impl BibGenerator {
    /// Generate BibTeX string from a list of entries.
    pub fn generate(entries: &[BibEntry]) -> String {
        entries
            .iter()
            .map(|e| e.to_bibtex())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate a BibEntry from an AcademicPaper.
    #[cfg(feature = "research")]
    pub fn from_paper(paper: &super::academic_search::AcademicPaper) -> BibEntry {
        use super::academic_search::AcademicSource;

        let entry_type = match paper.source {
            AcademicSource::ArXiv => BibEntryType::Misc, // preprints
            AcademicSource::PubMed => BibEntryType::Article,
            _ => {
                if paper
                    .venue
                    .as_ref()
                    .map(|v| v.contains("Proceedings") || v.contains("Conference"))
                    .unwrap_or(false)
                {
                    BibEntryType::InProceedings
                } else {
                    BibEntryType::Article
                }
            }
        };

        // Generate cite key: first author last name + year + first title word
        let first_author = paper
            .authors
            .first()
            .map(|a| {
                a.name
                    .split_whitespace()
                    .last()
                    .unwrap_or(&a.name)
                    .to_lowercase()
            })
            .unwrap_or_else(|| "unknown".to_string());

        let year_str = paper.year.map(|y| y.to_string()).unwrap_or_default();

        let first_word = paper
            .title
            .split_whitespace()
            .find(|w| w.len() > 3)
            .unwrap_or("paper")
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>();

        let cite_key = format!("{}{}{}", first_author, year_str, first_word);

        let mut entry = BibEntry::new(entry_type, &cite_key);

        entry.set_field("title", &paper.title);

        if !paper.authors.is_empty() {
            let authors_str = paper
                .authors
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(" and ");
            entry.set_field("author", &authors_str);
        }

        if let Some(year) = paper.year {
            entry.set_field("year", &year.to_string());
        }

        if let Some(venue) = &paper.venue {
            if entry_type == BibEntryType::InProceedings {
                entry.set_field("booktitle", venue);
            } else {
                entry.set_field("journal", venue);
            }
        }

        if let Some(doi) = &paper.doi {
            entry.set_field("doi", doi);
        }

        if let Some(url) = &paper.url {
            entry.set_field("url", url);
        }

        if let Some(abstract_text) = &paper.abstract_text {
            entry.set_field("abstract", abstract_text);
        }

        entry
    }

    /// Generate BibTeX string from a list of academic papers.
    #[cfg(feature = "research")]
    pub fn from_papers(papers: &[super::academic_search::AcademicPaper]) -> String {
        let entries: Vec<BibEntry> = papers.iter().map(|p| Self::from_paper(p)).collect();
        Self::generate(&entries)
    }
}

// =============================================================================
// LaTeX Sanitization
// =============================================================================

/// Dangerous LaTeX commands to strip from BibTeX field values.
const DANGEROUS_COMMANDS: &[&str] = &[
    "\\input",
    "\\include",
    "\\write18",
    "\\immediate",
    "\\openout",
    "\\openin",
    "\\csname",
    "\\newwrite",
    "\\closeout",
    "\\read",
    "\\write",
    "\\catcode",
    "\\def",
    "\\edef",
    "\\gdef",
    "\\xdef",
    "\\directlua",
    "\\luadirect",
    "\\luaexec",
    "\\ShellEscape",
];

/// Sanitize a string for safe inclusion in BibTeX.
///
/// Strips dangerous LaTeX commands that could execute arbitrary code when
/// the .bib file is processed by LaTeX.
fn sanitize_latex(input: &str) -> String {
    let mut result = input.to_string();

    for cmd in DANGEROUS_COMMANDS {
        // Remove the command and any braced argument following it
        loop {
            // Case-insensitive match on the ORIGINAL string so `pos`/`end`
            // are valid char boundaries of `result` (a lowercased-copy offset
            // panics on multibyte input — the sanitizer runs on untrusted .bib).
            if let Some((pos, end)) = crate::text_util::find_ci_range(&result, cmd) {
                // If followed by a braced argument, remove that too
                let after = &result[end..];
                let skip = if after.starts_with('{') {
                    // Find matching close brace
                    let mut depth = 0;
                    let mut i = 0;
                    for ch in after.chars() {
                        if ch == '{' {
                            depth += 1;
                        } else if ch == '}' {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        i += ch.len_utf8();
                    }
                    i
                } else {
                    0
                };
                result = format!("{}{}", &result[..pos], &result[end + skip..]);
            } else {
                break;
            }
        }
    }

    result
}

/// Convert common LaTeX accent commands to Unicode equivalents.
pub fn latex_to_unicode(input: &str) -> String {
    let mut result = input.to_string();

    // Common accent replacements
    let replacements = [
        (r#"\"{o}"#, "ö"),
        (r#"\"{u}"#, "ü"),
        (r#"\"{a}"#, "ä"),
        (r#"\"{O}"#, "Ö"),
        (r#"\"{U}"#, "Ü"),
        (r#"\"{A}"#, "Ä"),
        (r"\'e", "é"),
        (r"\'a", "á"),
        (r"\'i", "í"),
        (r"\'o", "ó"),
        (r"\'u", "ú"),
        (r"\`e", "è"),
        (r"\`a", "à"),
        (r"\^e", "ê"),
        (r"\^o", "ô"),
        (r"\~n", "ñ"),
        (r"\~a", "ã"),
        (r"\c{c}", "ç"),
        (r"\c{C}", "Ç"),
        (r"\ss{}", "ß"),
        (r"\ss ", "ß"),
        (r"\o{}", "ø"),
        (r"\O{}", "Ø"),
        (r"\aa{}", "å"),
        (r"\AA{}", "Å"),
        (r"\ae{}", "æ"),
        (r"\AE{}", "Æ"),
    ];

    for (latex, unicode) in &replacements {
        result = result.replace(latex, unicode);
    }

    // Remove remaining braces that were only used for grouping
    // E.g., {T}itle -> Title (but keep nested braces like {{Title}})
    let chars: Vec<char> = result.chars().collect();
    let mut cleaned = String::with_capacity(result.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            // Check if this is a single-char brace group like {T}
            if i + 2 < chars.len() && chars[i + 2] == '}' && chars[i + 1] != '{' {
                cleaned.push(chars[i + 1]);
                i += 3;
                continue;
            }
        }
        cleaned.push(chars[i]);
        i += 1;
    }

    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_type_from_str() {
        assert_eq!(
            BibEntryType::from_str_loose("article"),
            BibEntryType::Article
        );
        assert_eq!(
            BibEntryType::from_str_loose("ARTICLE"),
            BibEntryType::Article
        );
        assert_eq!(
            BibEntryType::from_str_loose("inproceedings"),
            BibEntryType::InProceedings
        );
        assert_eq!(
            BibEntryType::from_str_loose("conference"),
            BibEntryType::InProceedings
        );
        assert_eq!(
            BibEntryType::from_str_loose("phdthesis"),
            BibEntryType::Thesis
        );
        assert_eq!(
            BibEntryType::from_str_loose("mastersthesis"),
            BibEntryType::Thesis
        );
        assert_eq!(
            BibEntryType::from_str_loose("unknown_type"),
            BibEntryType::Misc
        );
    }

    #[test]
    fn test_entry_type_display() {
        assert_eq!(BibEntryType::Article.as_bibtex_str(), "article");
        assert_eq!(BibEntryType::InProceedings.as_bibtex_str(), "inproceedings");
        assert_eq!(BibEntryType::Thesis.as_bibtex_str(), "phdthesis");
    }

    #[test]
    fn test_bib_entry_creation() {
        let mut entry = BibEntry::new(BibEntryType::Article, "smith2024");
        entry.set_field("title", "Test Paper");
        entry.set_field("author", "John Smith");
        entry.set_field("year", "2024");

        assert_eq!(entry.cite_key, "smith2024");
        assert_eq!(entry.title(), Some("Test Paper"));
        assert_eq!(entry.author(), Some("John Smith"));
        assert_eq!(entry.year(), Some("2024"));
    }

    #[test]
    fn test_bib_entry_to_bibtex() {
        let mut entry = BibEntry::new(BibEntryType::Article, "key1");
        entry.set_field("title", "My Title");
        entry.set_field("year", "2024");

        let bib = entry.to_bibtex();
        assert!(bib.starts_with("@article{key1,"));
        assert!(bib.contains("title = {My Title}"));
        assert!(bib.contains("year = {2024}"));
        assert!(bib.ends_with("}\n"));
    }

    #[test]
    fn test_parse_simple_entry() {
        let input = r#"
@article{smith2024test,
  title = {A Test Paper},
  author = {John Smith and Jane Doe},
  year = {2024},
  journal = {Nature},
}
"#;
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.entry_type, BibEntryType::Article);
        assert_eq!(e.cite_key, "smith2024test");
        assert_eq!(e.title(), Some("A Test Paper"));
        assert_eq!(e.author(), Some("John Smith and Jane Doe"));
        assert_eq!(e.year(), Some("2024"));
        assert_eq!(e.get_field("journal"), Some("Nature"));
    }

    #[test]
    fn test_parse_multiple_entries() {
        let input = r#"
@article{a1, title = {Paper 1}, year = {2023}}
@inproceedings{b2, title = {Paper 2}, year = {2024}}
@book{c3, title = {Book 1}, year = {2022}}
"#;
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].entry_type, BibEntryType::Article);
        assert_eq!(entries[1].entry_type, BibEntryType::InProceedings);
        assert_eq!(entries[2].entry_type, BibEntryType::Book);
    }

    #[test]
    fn test_parse_nested_braces() {
        let input = r#"
@article{k1,
  title = {{Deep Learning} for {NLP}},
  year = {2024}
}
"#;
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title(), Some("{Deep Learning} for {NLP}"));
    }

    #[test]
    fn test_parse_quoted_values() {
        let input = r#"
@article{k1,
  title = "A Quoted Title",
  year = "2024"
}
"#;
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title(), Some("A Quoted Title"));
    }

    #[test]
    fn test_parse_skips_comments() {
        let input = r#"
@comment{This is a comment}
@article{k1, title = {Real Entry}, year = {2024}}
@preamble{"Some preamble"}
@string{jnl = "Journal Name"}
"#;
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].cite_key, "k1");
    }

    #[test]
    fn test_parse_empty() {
        let entries = BibParser::parse("").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_bare_number_value() {
        let input = "@article{k1, title = {Title}, year = 2024}";
        let entries = BibParser::parse(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].year(), Some("2024"));
    }

    #[test]
    fn test_sanitize_latex_input_command() {
        let sanitized = sanitize_latex(r"Safe text \input{malicious.tex} more text");
        assert!(!sanitized.contains("\\input"));
        assert!(!sanitized.contains("malicious"));
        assert!(sanitized.contains("Safe text"));
        assert!(sanitized.contains("more text"));
    }

    #[test]
    fn test_sanitize_latex_write18() {
        let sanitized = sanitize_latex(r"\write18{rm -rf /}");
        assert!(!sanitized.contains("\\write18"));
        assert!(!sanitized.contains("rm -rf"));
    }

    #[test]
    fn test_sanitize_latex_immediate() {
        let sanitized = sanitize_latex(r"\immediate\write18{evil}");
        assert!(!sanitized.contains("\\immediate"));
        assert!(!sanitized.contains("\\write18"));
    }

    #[test]
    fn test_sanitize_latex_safe_text() {
        let input = "Normal title with math $x^2 + y^2 = z^2$";
        let sanitized = sanitize_latex(input);
        assert_eq!(sanitized, input);
    }

    #[test]
    fn test_sanitize_latex_include() {
        let sanitized = sanitize_latex(r"\include{evil.tex}");
        assert!(!sanitized.contains("\\include"));
    }

    #[test]
    fn test_latex_to_unicode_accents() {
        assert_eq!(latex_to_unicode(r#"\"{o}"#), "ö");
        assert_eq!(latex_to_unicode(r"\'e"), "é");
        assert_eq!(latex_to_unicode(r"\~n"), "ñ");
        assert_eq!(latex_to_unicode(r"\c{c}"), "ç");
    }

    #[test]
    fn test_latex_to_unicode_no_change() {
        let input = "Plain text without accents";
        assert_eq!(latex_to_unicode(input), input);
    }

    #[test]
    fn test_generator_roundtrip() {
        let mut entry = BibEntry::new(BibEntryType::Article, "test2024");
        entry.set_field("title", "Roundtrip Test");
        entry.set_field("author", "Alice");
        entry.set_field("year", "2024");

        let bibtex = BibGenerator::generate(&[entry]);
        let parsed = BibParser::parse(&bibtex).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].cite_key, "test2024");
        assert_eq!(parsed[0].title(), Some("Roundtrip Test"));
    }

    #[cfg(feature = "research")]
    #[test]
    fn test_from_paper() {
        use super::super::academic_search::{AcademicPaper, AcademicSource, Author};

        let mut paper = AcademicPaper::new(
            "123",
            "Attention Is All You Need",
            AcademicSource::SemanticScholar,
        );
        paper.authors = vec![Author::new("Ashish Vaswani"), Author::new("Noam Shazeer")];
        paper.year = Some(2017);
        paper.venue = Some("NeurIPS".to_string());
        paper.doi = Some("10.1234/test".to_string());

        let entry = BibGenerator::from_paper(&paper);
        assert_eq!(entry.entry_type, BibEntryType::Article);
        assert!(entry.cite_key.contains("vaswani"));
        assert!(entry.cite_key.contains("2017"));
        assert_eq!(entry.title(), Some("Attention Is All You Need"));
        assert!(entry.author().unwrap().contains("Ashish Vaswani"));
        assert!(entry.author().unwrap().contains("Noam Shazeer"));
        assert_eq!(entry.get_field("doi"), Some("10.1234/test"));
    }

    #[cfg(feature = "research")]
    #[test]
    fn test_from_papers_multiple() {
        use super::super::academic_search::{AcademicPaper, AcademicSource, Author};

        let p1 = {
            let mut p = AcademicPaper::new("1", "Paper One", AcademicSource::ArXiv);
            p.authors = vec![Author::new("Alice")];
            p.year = Some(2023);
            p
        };
        let p2 = {
            let mut p = AcademicPaper::new("2", "Paper Two", AcademicSource::PubMed);
            p.authors = vec![Author::new("Bob")];
            p.year = Some(2024);
            p
        };

        let bibtex = BibGenerator::from_papers(&[p1, p2]);
        assert!(bibtex.contains("Paper One"));
        assert!(bibtex.contains("Paper Two"));

        let parsed = BibParser::parse(&bibtex).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_file_too_large() {
        let huge = "x".repeat(MAX_FILE_SIZE + 1);
        assert!(matches!(
            BibParser::parse(&huge),
            Err(BibParseError::FileTooLarge(_))
        ));
    }

    #[test]
    fn test_parse_error_display() {
        let e = BibParseError::FileTooLarge(999);
        assert!(e.to_string().contains("999"));
        let e2 = BibParseError::TooManyEntries(100);
        assert!(e2.to_string().contains("100"));
        let e3 = BibParseError::SyntaxError("bad".to_string());
        assert!(e3.to_string().contains("bad"));
    }
}
