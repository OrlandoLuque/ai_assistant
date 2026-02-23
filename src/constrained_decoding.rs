//! Grammar-guided constrained decoding for structured LLM output.
//!
//! This module implements Phase 8 of the v5 roadmap:
//! - **8.1 Grammar-Guided Generation**: GBNF format parsing and serialization
//! - **8.2 JSON Schema to Grammar Compiler**: Convert JSON Schema to GBNF grammars
//! - **8.3 Streaming Structured Validation**: Incremental token validation against schemas
//!
//! GBNF (GGML BNF) is the grammar format used by llama.cpp and Ollama for
//! constraining model output to match specific syntactic patterns (JSON objects,
//! SQL queries, function calls, etc.).
//!
//! Feature-gated behind `constrained-decoding`.

#[cfg(feature = "constrained-decoding")]
mod inner {
    use std::collections::HashMap;
    use std::fmt;

    use crate::error::{AiError, ConstrainedDecodingError};

    // ========================================================================
    // 8.1 — Grammar-Guided Generation (GBNF format)
    // ========================================================================

    /// How a grammar element is repeated.
    #[derive(Debug, Clone, PartialEq)]
    pub enum RepeatKind {
        /// Zero or more repetitions (`*`).
        ZeroOrMore,
        /// One or more repetitions (`+`).
        OneOrMore,
    }

    impl fmt::Display for RepeatKind {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                RepeatKind::ZeroOrMore => write!(f, "*"),
                RepeatKind::OneOrMore => write!(f, "+"),
            }
        }
    }

    /// A single element in a grammar alternative.
    #[derive(Debug, Clone, PartialEq)]
    pub enum GrammarElement {
        /// Exact string literal match, e.g. `"true"`.
        Literal(String),
        /// Character range, e.g. `[a-z]`.
        CharRange(char, char),
        /// Character set, e.g. `[abc]`.
        CharSet(Vec<char>),
        /// Negated character set, e.g. `[^abc]`.
        NegatedCharSet(Vec<char>),
        /// Reference to another rule by name.
        RuleRef(String),
        /// Repeated element.
        Repeat(Box<GrammarElement>, RepeatKind),
        /// Grouped sequence of elements.
        Group(Vec<GrammarElement>),
        /// Optional element (shorthand for `element?`).
        Optional(Box<GrammarElement>),
    }

    impl GrammarElement {
        /// Serialize this element to a GBNF string fragment.
        fn to_gbnf(&self) -> String {
            match self {
                GrammarElement::Literal(s) => {
                    let escaped = s
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\t', "\\t");
                    format!("\"{}\"", escaped)
                }
                GrammarElement::CharRange(from, to) => {
                    format!("[{}-{}]", escape_char_for_charset(*from), escape_char_for_charset(*to))
                }
                GrammarElement::CharSet(chars) => {
                    let inner: String = chars.iter().map(|c| escape_char_for_charset(*c)).collect();
                    format!("[{}]", inner)
                }
                GrammarElement::NegatedCharSet(chars) => {
                    let inner: String = chars.iter().map(|c| escape_char_for_charset(*c)).collect();
                    format!("[^{}]", inner)
                }
                GrammarElement::RuleRef(name) => name.clone(),
                GrammarElement::Repeat(elem, kind) => {
                    let inner = elem.to_gbnf();
                    let needs_parens = matches!(
                        elem.as_ref(),
                        GrammarElement::Group(_)
                            | GrammarElement::Literal(_)
                            | GrammarElement::CharRange(_, _)
                            | GrammarElement::CharSet(_)
                            | GrammarElement::NegatedCharSet(_)
                            | GrammarElement::RuleRef(_)
                    );
                    if needs_parens {
                        format!("{}{}", inner, kind)
                    } else {
                        format!("({}){}", inner, kind)
                    }
                }
                GrammarElement::Group(elems) => {
                    let parts: Vec<String> = elems.iter().map(|e| e.to_gbnf()).collect();
                    format!("({})", parts.join(" "))
                }
                GrammarElement::Optional(elem) => {
                    let inner = elem.to_gbnf();
                    format!("{}?", inner)
                }
            }
        }
    }

    /// Escape a character for use inside a GBNF character set (`[...]`).
    fn escape_char_for_charset(c: char) -> String {
        match c {
            '\\' => "\\\\".to_string(),
            ']' => "\\]".to_string(),
            '-' => "\\-".to_string(),
            '^' => "\\^".to_string(),
            '"' => "\\\"".to_string(),
            '\n' => "\\n".to_string(),
            '\t' => "\\t".to_string(),
            ' ' => " ".to_string(),
            other => other.to_string(),
        }
    }

    /// One alternative in a grammar rule (a sequence of elements).
    #[derive(Debug, Clone, PartialEq)]
    pub struct GrammarAlternative {
        /// The ordered sequence of elements that make up this alternative.
        pub elements: Vec<GrammarElement>,
    }

    impl GrammarAlternative {
        /// Create a new empty alternative.
        pub fn new() -> Self {
            Self {
                elements: Vec::new(),
            }
        }

        /// Serialize this alternative to a GBNF fragment.
        fn to_gbnf(&self) -> String {
            self.elements
                .iter()
                .map(|e| e.to_gbnf())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    impl Default for GrammarAlternative {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A single named rule in the grammar.
    #[derive(Debug, Clone, PartialEq)]
    pub struct GrammarRule {
        /// The rule name (e.g. `"root"`, `"object"`, `"string"`).
        pub name: String,
        /// The alternatives for this rule (separated by `|` in GBNF).
        pub alternatives: Vec<GrammarAlternative>,
    }

    impl GrammarRule {
        /// Create a new rule with a name.
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                alternatives: Vec::new(),
            }
        }

        /// Serialize this rule to a GBNF line.
        fn to_gbnf(&self) -> String {
            let alts: Vec<String> = self.alternatives.iter().map(|a| a.to_gbnf()).collect();
            format!("{} ::= {}", self.name, alts.join(" | "))
        }
    }

    /// A complete grammar consisting of named rules.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Grammar {
        /// The rules that make up this grammar.
        pub rules: Vec<GrammarRule>,
        /// The name of the root rule (entry point).
        pub root_rule: String,
    }

    impl Grammar {
        /// Create a new grammar with the specified root rule name.
        pub fn new(root_rule: impl Into<String>) -> Self {
            Self {
                rules: Vec::new(),
                root_rule: root_rule.into(),
            }
        }

        /// Serialize this grammar to a GBNF-format string.
        pub fn to_gbnf(&self) -> String {
            // Put root rule first, then others in order.
            let mut lines = Vec::new();
            for rule in &self.rules {
                if rule.name == self.root_rule {
                    lines.insert(0, rule.to_gbnf());
                } else {
                    lines.push(rule.to_gbnf());
                }
            }
            lines.join("\n")
        }

        /// Parse a GBNF-format string into a Grammar.
        ///
        /// The first rule encountered becomes the root rule.
        pub fn from_gbnf(input: &str) -> Result<Grammar, AiError> {
            let mut rules = Vec::new();
            let mut root_rule = String::new();

            for (line_idx, raw_line) in input.lines().enumerate() {
                let line = raw_line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let sep_pos = line.find("::=").ok_or_else(|| {
                    AiError::ConstrainedDecoding(ConstrainedDecodingError::GrammarSyntaxError {
                        line: line_idx + 1,
                        message: format!("expected '::=' separator in: {}", line),
                    })
                })?;

                let name = line[..sep_pos].trim().to_string();
                if name.is_empty() {
                    return Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::GrammarSyntaxError {
                            line: line_idx + 1,
                            message: "rule name is empty".to_string(),
                        },
                    ));
                }

                let rhs = line[sep_pos + 3..].trim();
                let alternatives = parse_alternatives(rhs, line_idx + 1)?;

                if root_rule.is_empty() {
                    root_rule.clone_from(&name);
                }
                rules.push(GrammarRule {
                    name,
                    alternatives,
                });
            }

            if rules.is_empty() {
                return Err(AiError::ConstrainedDecoding(
                    ConstrainedDecodingError::GrammarCompilationFailed {
                        reason: "grammar has no rules".to_string(),
                    },
                ));
            }

            Ok(Grammar { rules, root_rule })
        }
    }

    // ------------------------------------------------------------------------
    // GBNF parser helpers
    // ------------------------------------------------------------------------

    /// Split the right-hand side of a rule by top-level `|` and parse each alternative.
    fn parse_alternatives(
        rhs: &str,
        line_num: usize,
    ) -> Result<Vec<GrammarAlternative>, AiError> {
        let parts = split_top_level(rhs, '|');
        let mut alts = Vec::new();
        for part in &parts {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let elements = parse_elements(trimmed, line_num)?;
            alts.push(GrammarAlternative { elements });
        }
        if alts.is_empty() {
            alts.push(GrammarAlternative::new());
        }
        Ok(alts)
    }

    /// Split a string by a delimiter character, but only at the top level
    /// (respecting quotes, brackets, and parentheses).
    fn split_top_level(s: &str, delimiter: char) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut chars = s.chars().peekable();
        let mut depth = 0i32; // tracks (), []
        let mut in_quote = false;

        while let Some(c) = chars.next() {
            if c == '\\' && in_quote {
                current.push(c);
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
                continue;
            }
            if c == '"' {
                in_quote = !in_quote;
                current.push(c);
                continue;
            }
            if in_quote {
                current.push(c);
                continue;
            }
            if c == '(' || c == '[' {
                depth += 1;
                current.push(c);
                continue;
            }
            if c == ')' || c == ']' {
                depth -= 1;
                current.push(c);
                continue;
            }
            if c == delimiter && depth == 0 {
                parts.push(current.clone());
                current.clear();
                continue;
            }
            current.push(c);
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }

    /// Parse a sequence of GBNF elements from a string (one alternative).
    fn parse_elements(s: &str, line_num: usize) -> Result<Vec<GrammarElement>, AiError> {
        let mut elements = Vec::new();
        let mut chars = s.chars().peekable();

        while chars.peek().is_some() {
            // Skip whitespace
            while chars.peek().map_or(false, |c| c.is_whitespace()) {
                chars.next();
            }
            if chars.peek().is_none() {
                break;
            }

            let elem = parse_single_element(&mut chars, line_num)?;

            // Check for repetition suffix (*, +, ?)
            if let Some(&suffix) = chars.peek() {
                match suffix {
                    '*' => {
                        chars.next();
                        elements.push(GrammarElement::Repeat(
                            Box::new(elem),
                            RepeatKind::ZeroOrMore,
                        ));
                        continue;
                    }
                    '+' => {
                        chars.next();
                        elements.push(GrammarElement::Repeat(
                            Box::new(elem),
                            RepeatKind::OneOrMore,
                        ));
                        continue;
                    }
                    '?' => {
                        chars.next();
                        elements.push(GrammarElement::Optional(Box::new(elem)));
                        continue;
                    }
                    _ => {}
                }
            }

            elements.push(elem);
        }

        Ok(elements)
    }

    /// Parse a single GBNF element from a character stream.
    fn parse_single_element(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        line_num: usize,
    ) -> Result<GrammarElement, AiError> {
        match chars.peek() {
            Some('"') => parse_literal(chars, line_num),
            Some('[') => parse_charset(chars, line_num),
            Some('(') => parse_group(chars, line_num),
            Some(c) if c.is_alphanumeric() || *c == '_' || *c == '-' => {
                parse_rule_ref(chars)
            }
            Some(c) => Err(AiError::ConstrainedDecoding(
                ConstrainedDecodingError::GrammarSyntaxError {
                    line: line_num,
                    message: format!("unexpected character: '{}'", c),
                },
            )),
            None => Err(AiError::ConstrainedDecoding(
                ConstrainedDecodingError::GrammarSyntaxError {
                    line: line_num,
                    message: "unexpected end of input".to_string(),
                },
            )),
        }
    }

    /// Parse a quoted literal like `"true"`.
    fn parse_literal(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        line_num: usize,
    ) -> Result<GrammarElement, AiError> {
        chars.next(); // consume opening '"'
        let mut value = String::new();
        loop {
            match chars.next() {
                Some('\\') => match chars.next() {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('"') => value.push('"'),
                    Some('\\') => value.push('\\'),
                    Some(c) => {
                        value.push('\\');
                        value.push(c);
                    }
                    None => {
                        return Err(AiError::ConstrainedDecoding(
                            ConstrainedDecodingError::GrammarSyntaxError {
                                line: line_num,
                                message: "unexpected end of escape in literal".to_string(),
                            },
                        ))
                    }
                },
                Some('"') => break,
                Some(c) => value.push(c),
                None => {
                    return Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::GrammarSyntaxError {
                            line: line_num,
                            message: "unterminated string literal".to_string(),
                        },
                    ))
                }
            }
        }
        Ok(GrammarElement::Literal(value))
    }

    /// Parse a character set like `[a-z]`, `[abc]`, or `[^"\\]`.
    fn parse_charset(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        line_num: usize,
    ) -> Result<GrammarElement, AiError> {
        chars.next(); // consume '['
        let negated = chars.peek() == Some(&'^');
        if negated {
            chars.next();
        }

        let mut set_chars = Vec::new();
        let mut range_from: Option<char> = None;
        let mut expect_range_end = false;

        loop {
            match chars.next() {
                Some(']') => break,
                Some('\\') => {
                    let esc = match chars.next() {
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('\\') => '\\',
                        Some('"') => '"',
                        Some(']') => ']',
                        Some('-') => '-',
                        Some('^') => '^',
                        Some(c) => c,
                        None => {
                            return Err(AiError::ConstrainedDecoding(
                                ConstrainedDecodingError::GrammarSyntaxError {
                                    line: line_num,
                                    message: "unexpected end of escape in charset".to_string(),
                                },
                            ))
                        }
                    };
                    if expect_range_end {
                        if let Some(from) = range_from.take() {
                            // This is a range: from-esc
                            return if negated {
                                // Negated ranges not directly representable; store as NegatedCharSet
                                // with expanded chars
                                let mut expanded = set_chars;
                                for c in from..=esc {
                                    expanded.push(c);
                                }
                                // Consume remaining chars in charset
                                consume_remaining_charset(chars, &mut expanded, line_num)?;
                                Ok(GrammarElement::NegatedCharSet(expanded))
                            } else {
                                let mut expanded = set_chars;
                                for c in from..=esc {
                                    expanded.push(c);
                                }
                                consume_remaining_charset(chars, &mut expanded, line_num)?;
                                // Check if it's actually a single range
                                if expanded.len() == (esc as u32 - from as u32 + 1) as usize {
                                    Ok(GrammarElement::CharRange(from, esc))
                                } else {
                                    Ok(GrammarElement::CharSet(expanded))
                                }
                            };
                        }
                        expect_range_end = false;
                    }
                    range_from = Some(esc);
                    set_chars.push(esc);
                }
                Some('-') if range_from.is_some() => {
                    // Potential range: remove the last char from set and mark range start
                    set_chars.pop(); // will be re-added as part of range
                    expect_range_end = true;
                }
                Some(c) => {
                    if expect_range_end {
                        if let Some(from) = range_from.take() {
                            // Range from..=c
                            if negated {
                                for rc in from..=c {
                                    set_chars.push(rc);
                                }
                                consume_remaining_charset(chars, &mut set_chars, line_num)?;
                                return Ok(GrammarElement::NegatedCharSet(set_chars));
                            }
                            // Non-negated range
                            for rc in from..=c {
                                set_chars.push(rc);
                            }
                            consume_remaining_charset(chars, &mut set_chars, line_num)?;
                            if set_chars.len() == (c as u32 - from as u32 + 1) as usize {
                                return Ok(GrammarElement::CharRange(from, c));
                            }
                            return Ok(GrammarElement::CharSet(set_chars));
                        }
                        expect_range_end = false;
                    }
                    range_from = Some(c);
                    set_chars.push(c);
                }
                None => {
                    return Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::GrammarSyntaxError {
                            line: line_num,
                            message: "unterminated character set".to_string(),
                        },
                    ))
                }
            }
        }

        if negated {
            Ok(GrammarElement::NegatedCharSet(set_chars))
        } else if set_chars.is_empty() {
            Err(AiError::ConstrainedDecoding(
                ConstrainedDecodingError::GrammarSyntaxError {
                    line: line_num,
                    message: "empty character set".to_string(),
                },
            ))
        } else {
            Ok(GrammarElement::CharSet(set_chars))
        }
    }

    /// Consume the rest of a charset until `]` and add chars to `out`.
    fn consume_remaining_charset(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        out: &mut Vec<char>,
        line_num: usize,
    ) -> Result<(), AiError> {
        loop {
            match chars.next() {
                Some(']') => return Ok(()),
                Some('\\') => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some('"') => out.push('"'),
                    Some(c) => out.push(c),
                    None => {
                        return Err(AiError::ConstrainedDecoding(
                            ConstrainedDecodingError::GrammarSyntaxError {
                                line: line_num,
                                message: "unexpected end in charset".to_string(),
                            },
                        ))
                    }
                },
                Some(c) => out.push(c),
                None => {
                    return Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::GrammarSyntaxError {
                            line: line_num,
                            message: "unterminated charset".to_string(),
                        },
                    ))
                }
            }
        }
    }

    /// Parse a parenthesized group like `("," ws value)`.
    fn parse_group(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
        line_num: usize,
    ) -> Result<GrammarElement, AiError> {
        chars.next(); // consume '('
        let mut depth = 1i32;
        let mut inner = String::new();
        loop {
            match chars.next() {
                Some('(') => {
                    depth += 1;
                    inner.push('(');
                }
                Some(')') => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    inner.push(')');
                }
                Some('"') => {
                    inner.push('"');
                    // Consume string contents verbatim
                    loop {
                        match chars.next() {
                            Some('\\') => {
                                inner.push('\\');
                                if let Some(c) = chars.next() {
                                    inner.push(c);
                                }
                            }
                            Some('"') => {
                                inner.push('"');
                                break;
                            }
                            Some(c) => inner.push(c),
                            None => break,
                        }
                    }
                }
                Some(c) => inner.push(c),
                None => {
                    return Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::GrammarSyntaxError {
                            line: line_num,
                            message: "unterminated group".to_string(),
                        },
                    ))
                }
            }
        }
        let elements = parse_elements(inner.trim(), line_num)?;
        Ok(GrammarElement::Group(elements))
    }

    /// Parse a rule reference (identifier).
    fn parse_rule_ref(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    ) -> Result<GrammarElement, AiError> {
        let mut name = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                name.push(c);
                chars.next();
            } else {
                break;
            }
        }
        Ok(GrammarElement::RuleRef(name))
    }

    // ========================================================================
    // Grammar Builder (Fluent API)
    // ========================================================================

    /// Builder for constructing a single alternative in a rule.
    pub struct AltBuilder<'a> {
        elements: Vec<GrammarElement>,
        rule_builder: &'a mut RuleBuilderInner,
    }

    impl<'a> AltBuilder<'a> {
        /// Add a literal string element.
        pub fn literal(mut self, s: &str) -> Self {
            self.elements.push(GrammarElement::Literal(s.to_string()));
            self
        }

        /// Add a character range element `[from-to]`.
        pub fn char_range(mut self, from: char, to: char) -> Self {
            self.elements
                .push(GrammarElement::CharRange(from, to));
            self
        }

        /// Add a rule reference element.
        pub fn rule_ref(mut self, name: &str) -> Self {
            self.elements
                .push(GrammarElement::RuleRef(name.to_string()));
            self
        }

        /// Add a repeated element.
        pub fn repeat(mut self, element: GrammarElement, kind: RepeatKind) -> Self {
            self.elements
                .push(GrammarElement::Repeat(Box::new(element), kind));
            self
        }

        /// Add an optional element.
        pub fn optional(mut self, element: GrammarElement) -> Self {
            self.elements
                .push(GrammarElement::Optional(Box::new(element)));
            self
        }

        /// Add a character set element `[chars]`.
        pub fn char_set(mut self, chars: Vec<char>) -> Self {
            self.elements.push(GrammarElement::CharSet(chars));
            self
        }

        /// Add a negated character set element `[^chars]`.
        pub fn negated_char_set(mut self, chars: Vec<char>) -> Self {
            self.elements.push(GrammarElement::NegatedCharSet(chars));
            self
        }

        /// Add a group of elements.
        pub fn group(mut self, elements: Vec<GrammarElement>) -> Self {
            self.elements.push(GrammarElement::Group(elements));
            self
        }

        /// Finish this alternative and return to the rule builder.
        pub fn done(self) -> RuleBuilder {
            self.rule_builder.alternatives.push(GrammarAlternative {
                elements: self.elements,
            });
            RuleBuilder {
                inner: self.rule_builder as *mut RuleBuilderInner,
            }
        }
    }

    /// Internal storage for a rule being built.
    struct RuleBuilderInner {
        name: String,
        alternatives: Vec<GrammarAlternative>,
    }

    /// Builder for a single grammar rule.
    pub struct RuleBuilder {
        inner: *mut RuleBuilderInner,
    }

    impl RuleBuilder {
        /// Start building a new alternative for this rule.
        pub fn alt(self) -> AltBuilder<'static> {
            // Safety: The inner pointer is valid for the lifetime of the GrammarBuilder
            // that created it. We use 'static as a simplification; the builder API is
            // designed to be used in a single fluent chain.
            let inner = unsafe { &mut *self.inner };
            AltBuilder {
                elements: Vec::new(),
                rule_builder: inner,
            }
        }

        /// Finish this rule and return to the grammar builder.
        pub fn done(self) -> GrammarRule {
            let inner = unsafe { &mut *self.inner };
            GrammarRule {
                name: inner.name.clone(),
                alternatives: std::mem::take(&mut inner.alternatives),
            }
        }
    }

    /// Fluent API builder for constructing grammars programmatically.
    #[allow(clippy::vec_box)] // Box needed: raw pointers require stable heap addresses
    pub struct GrammarBuilder {
        root_name: String,
        rules: Vec<GrammarRule>,
        // Keep alive for rule builders to reference
        _rule_builders: Vec<Box<RuleBuilderInner>>,
    }

    impl GrammarBuilder {
        /// Create a new grammar builder with the specified root rule name.
        pub fn new(root_name: &str) -> Self {
            Self {
                root_name: root_name.to_string(),
                rules: Vec::new(),
                _rule_builders: Vec::new(),
            }
        }

        /// Start building a new rule with the given name.
        pub fn rule(&mut self, name: &str) -> RuleBuilder {
            let rb = Box::new(RuleBuilderInner {
                name: name.to_string(),
                alternatives: Vec::new(),
            });
            let ptr = &*rb as *const RuleBuilderInner as *mut RuleBuilderInner;
            self._rule_builders.push(rb);
            RuleBuilder { inner: ptr }
        }

        /// Add a previously built rule to the grammar.
        pub fn add_rule(&mut self, rule: GrammarRule) -> &mut Self {
            self.rules.push(rule);
            self
        }

        /// Build the final Grammar.
        pub fn build(self) -> Grammar {
            Grammar {
                rules: self.rules,
                root_rule: self.root_name,
            }
        }
    }

    // ========================================================================
    // 8.2 — JSON Schema → Grammar Compiler
    // ========================================================================

    /// Format of grammar output for different providers.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ProviderGrammarFormat {
        /// GBNF format (Ollama / llama.cpp).
        GBNF,
        /// Regex format (vLLM).
        Regex,
        /// JSON Schema format (OpenAI structured outputs).
        JsonSchema,
    }

    impl fmt::Display for ProviderGrammarFormat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ProviderGrammarFormat::GBNF => write!(f, "gbnf"),
                ProviderGrammarFormat::Regex => write!(f, "regex"),
                ProviderGrammarFormat::JsonSchema => write!(f, "json_schema"),
            }
        }
    }

    /// A grammar constraint that can be sent to a provider for constrained generation.
    #[derive(Debug, Clone)]
    pub struct GrammarConstraint {
        /// The underlying grammar.
        pub grammar: Grammar,
        /// The target provider format.
        pub provider_format: ProviderGrammarFormat,
    }

    impl GrammarConstraint {
        /// Create a new grammar constraint.
        pub fn new(grammar: Grammar, provider_format: ProviderGrammarFormat) -> Self {
            Self {
                grammar,
                provider_format,
            }
        }

        /// Format the grammar for a specific provider.
        ///
        /// Supported providers: `"ollama"`, `"llama.cpp"`, `"lmstudio"`, `"vllm"`, `"openai"`.
        pub fn for_provider(grammar: &Grammar, provider: &str) -> Result<String, AiError> {
            match provider.to_lowercase().as_str() {
                "ollama" | "llama.cpp" | "llamacpp" | "lmstudio" => {
                    Ok(grammar.to_gbnf())
                }
                "vllm" => {
                    // vLLM uses regex; produce a simplified regex from the grammar root
                    Ok(grammar_to_simple_regex(grammar))
                }
                "openai" => {
                    // OpenAI uses JSON Schema pass-through; not truly grammar-based
                    Err(AiError::ConstrainedDecoding(
                        ConstrainedDecodingError::ProviderUnsupported {
                            provider: provider.to_string(),
                        },
                    ))
                }
                _ => Err(AiError::ConstrainedDecoding(
                    ConstrainedDecodingError::ProviderUnsupported {
                        provider: provider.to_string(),
                    }),
                ),
            }
        }
    }

    /// Produce a simple regex approximation of the grammar root rule.
    /// This is a best-effort translation; complex grammars may not translate perfectly.
    fn grammar_to_simple_regex(grammar: &Grammar) -> String {
        let rule_map: HashMap<String, &GrammarRule> = grammar
            .rules
            .iter()
            .map(|r| (r.name.clone(), r))
            .collect();
        if let Some(root) = rule_map.get(&grammar.root_rule) {
            let alts: Vec<String> = root
                .alternatives
                .iter()
                .map(|alt| {
                    alt.elements
                        .iter()
                        .map(|e| element_to_regex(e, &rule_map, 0))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .collect();
            if alts.len() == 1 {
                alts.into_iter().next().unwrap_or_default()
            } else {
                format!("({})", alts.join("|"))
            }
        } else {
            ".*".to_string()
        }
    }

    /// Convert a single grammar element to a regex fragment (best-effort, depth-limited).
    fn element_to_regex(
        elem: &GrammarElement,
        rules: &HashMap<String, &GrammarRule>,
        depth: usize,
    ) -> String {
        if depth > 10 {
            return ".*".to_string(); // prevent infinite recursion
        }
        match elem {
            GrammarElement::Literal(s) => regex::escape(s),
            GrammarElement::CharRange(a, b) => format!("[{}-{}]", a, b),
            GrammarElement::CharSet(cs) => {
                let inner: String = cs.iter().map(|c| regex::escape(&c.to_string())).collect();
                format!("[{}]", inner)
            }
            GrammarElement::NegatedCharSet(cs) => {
                let inner: String = cs.iter().map(|c| regex::escape(&c.to_string())).collect();
                format!("[^{}]", inner)
            }
            GrammarElement::RuleRef(name) => {
                if let Some(rule) = rules.get(name) {
                    let alts: Vec<String> = rule
                        .alternatives
                        .iter()
                        .map(|alt| {
                            alt.elements
                                .iter()
                                .map(|e| element_to_regex(e, rules, depth + 1))
                                .collect::<Vec<_>>()
                                .join("")
                        })
                        .collect();
                    if alts.len() == 1 {
                        alts.into_iter().next().unwrap_or_default()
                    } else {
                        format!("({})", alts.join("|"))
                    }
                } else {
                    ".*".to_string()
                }
            }
            GrammarElement::Repeat(inner, kind) => {
                let r = element_to_regex(inner, rules, depth);
                match kind {
                    RepeatKind::ZeroOrMore => format!("({})*", r),
                    RepeatKind::OneOrMore => format!("({})+", r),
                }
            }
            GrammarElement::Group(elems) => {
                let inner: String = elems
                    .iter()
                    .map(|e| element_to_regex(e, rules, depth))
                    .collect();
                format!("({})", inner)
            }
            GrammarElement::Optional(inner) => {
                let r = element_to_regex(inner, rules, depth);
                format!("({})?", r)
            }
        }
    }

    /// Compiler that converts a JSON Schema (`serde_json::Value`) into a `Grammar`.
    pub struct SchemaToGrammar;

    impl SchemaToGrammar {
        /// Compile a JSON Schema value into a GBNF grammar.
        pub fn compile(schema: &serde_json::Value) -> Result<Grammar, AiError> {
            let mut rules = Vec::new();
            let mut counter = Counter::new();

            // Always add the whitespace rule
            rules.push(GrammarRule {
                name: "ws".to_string(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![GrammarElement::Repeat(
                        Box::new(GrammarElement::CharSet(vec![' ', '\t', '\n'])),
                        RepeatKind::ZeroOrMore,
                    )],
                }],
            });

            let root_rule_name =
                Self::compile_schema(schema, "root", &mut rules, &mut counter)?;

            // If the root was generated with a different name, alias it
            if root_rule_name != "root" {
                rules.push(GrammarRule {
                    name: "root".to_string(),
                    alternatives: vec![GrammarAlternative {
                        elements: vec![GrammarElement::RuleRef(root_rule_name)],
                    }],
                });
            }

            Ok(Grammar {
                rules,
                root_rule: "root".to_string(),
            })
        }

        /// Recursively compile a schema node into grammar rules.
        /// Returns the rule name for this node.
        fn compile_schema(
            schema: &serde_json::Value,
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
            counter: &mut Counter,
        ) -> Result<String, AiError> {
            // Handle enum first (can appear on any type)
            if let Some(enum_values) = schema.get("enum") {
                return Self::compile_enum(enum_values, name_hint, rules);
            }

            let type_val = schema.get("type").and_then(|t| t.as_str());

            match type_val {
                Some("string") => Self::compile_string(name_hint, rules),
                Some("number") => Self::compile_number(name_hint, rules),
                Some("integer") => Self::compile_integer(name_hint, rules),
                Some("boolean") => Self::compile_boolean(name_hint, rules),
                Some("null") => Self::compile_null(name_hint, rules),
                Some("object") => {
                    Self::compile_object(schema, name_hint, rules, counter)
                }
                Some("array") => {
                    Self::compile_array(schema, name_hint, rules, counter)
                }
                Some(unknown) => Err(AiError::ConstrainedDecoding(
                    ConstrainedDecodingError::SchemaConversionFailed {
                        path: name_hint.to_string(),
                        reason: format!("unsupported type: {}", unknown),
                    },
                )),
                None => {
                    // No type specified: default to any JSON value
                    Self::compile_any_value(name_hint, rules, counter)
                }
            }
        }

        fn compile_string(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-string", name_hint);
            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![
                        GrammarElement::Literal("\"".to_string()),
                        GrammarElement::Repeat(
                            Box::new(GrammarElement::NegatedCharSet(vec!['"', '\\'])),
                            RepeatKind::ZeroOrMore,
                        ),
                        GrammarElement::Literal("\"".to_string()),
                    ],
                }],
            });
            Ok(rule_name)
        }

        fn compile_number(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-number", name_hint);
            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![
                        GrammarElement::Optional(Box::new(GrammarElement::Literal(
                            "-".to_string(),
                        ))),
                        GrammarElement::Repeat(
                            Box::new(GrammarElement::CharRange('0', '9')),
                            RepeatKind::OneOrMore,
                        ),
                        GrammarElement::Optional(Box::new(GrammarElement::Group(vec![
                            GrammarElement::Literal(".".to_string()),
                            GrammarElement::Repeat(
                                Box::new(GrammarElement::CharRange('0', '9')),
                                RepeatKind::OneOrMore,
                            ),
                        ]))),
                    ],
                }],
            });
            Ok(rule_name)
        }

        fn compile_integer(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-integer", name_hint);
            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![
                        GrammarElement::Optional(Box::new(GrammarElement::Literal(
                            "-".to_string(),
                        ))),
                        GrammarElement::Repeat(
                            Box::new(GrammarElement::CharRange('0', '9')),
                            RepeatKind::OneOrMore,
                        ),
                    ],
                }],
            });
            Ok(rule_name)
        }

        fn compile_boolean(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-boolean", name_hint);
            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![
                    GrammarAlternative {
                        elements: vec![GrammarElement::Literal("true".to_string())],
                    },
                    GrammarAlternative {
                        elements: vec![GrammarElement::Literal("false".to_string())],
                    },
                ],
            });
            Ok(rule_name)
        }

        fn compile_null(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-null", name_hint);
            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![GrammarElement::Literal("null".to_string())],
                }],
            });
            Ok(rule_name)
        }

        fn compile_enum(
            values: &serde_json::Value,
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
        ) -> Result<String, AiError> {
            let arr = values.as_array().ok_or_else(|| {
                AiError::ConstrainedDecoding(ConstrainedDecodingError::SchemaConversionFailed {
                    path: name_hint.to_string(),
                    reason: "enum must be an array".to_string(),
                })
            })?;

            let rule_name = format!("{}-enum", name_hint);
            let mut alternatives = Vec::new();

            for val in arr {
                let literal = match val {
                    serde_json::Value::String(s) => format!("\"{}\"", s),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Null => "null".to_string(),
                    _ => val.to_string(),
                };
                alternatives.push(GrammarAlternative {
                    elements: vec![GrammarElement::Literal(literal)],
                });
            }

            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives,
            });
            Ok(rule_name)
        }

        fn compile_object(
            schema: &serde_json::Value,
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
            counter: &mut Counter,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-object", name_hint);
            let properties = schema
                .get("properties")
                .and_then(|p| p.as_object())
                .cloned()
                .unwrap_or_default();

            let required: Vec<String> = schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if properties.is_empty() {
                // Empty object
                rules.push(GrammarRule {
                    name: rule_name.clone(),
                    alternatives: vec![GrammarAlternative {
                        elements: vec![
                            GrammarElement::Literal("{".to_string()),
                            GrammarElement::RuleRef("ws".to_string()),
                            GrammarElement::Literal("}".to_string()),
                        ],
                    }],
                });
                return Ok(rule_name);
            }

            // Build property rules
            let mut prop_elements = Vec::new();
            let mut first = true;

            // Sort properties: required first, then optional
            let mut prop_names: Vec<String> = properties.keys().cloned().collect();
            prop_names.sort_by(|a, b| {
                let a_req = required.contains(a);
                let b_req = required.contains(b);
                b_req.cmp(&a_req).then(a.cmp(b))
            });

            for prop_name in &prop_names {
                if let Some(prop_schema) = properties.get(prop_name) {
                    let prop_hint = format!("{}-{}", name_hint, prop_name);
                    let prop_rule =
                        Self::compile_schema(prop_schema, &prop_hint, rules, counter)?;

                    let mut pair_elements = Vec::new();
                    if !first {
                        pair_elements.push(GrammarElement::Literal(",".to_string()));
                        pair_elements.push(GrammarElement::RuleRef("ws".to_string()));
                    }
                    pair_elements.push(GrammarElement::Literal(format!(
                        "\"{}\"",
                        prop_name
                    )));
                    pair_elements.push(GrammarElement::RuleRef("ws".to_string()));
                    pair_elements.push(GrammarElement::Literal(":".to_string()));
                    pair_elements.push(GrammarElement::RuleRef("ws".to_string()));
                    pair_elements.push(GrammarElement::RuleRef(prop_rule));

                    let is_required = required.contains(prop_name);
                    if is_required || first {
                        prop_elements.extend(pair_elements);
                    } else {
                        prop_elements.push(GrammarElement::Optional(Box::new(
                            GrammarElement::Group(pair_elements),
                        )));
                    }
                    first = false;
                }
            }

            let mut elements = vec![
                GrammarElement::Literal("{".to_string()),
                GrammarElement::RuleRef("ws".to_string()),
            ];
            elements.extend(prop_elements);
            elements.push(GrammarElement::RuleRef("ws".to_string()));
            elements.push(GrammarElement::Literal("}".to_string()));

            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![GrammarAlternative { elements }],
            });
            Ok(rule_name)
        }

        fn compile_array(
            schema: &serde_json::Value,
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
            counter: &mut Counter,
        ) -> Result<String, AiError> {
            let rule_name = format!("{}-array", name_hint);

            let item_rule = if let Some(items_schema) = schema.get("items") {
                let item_hint = format!("{}-item", name_hint);
                Self::compile_schema(items_schema, &item_hint, rules, counter)?
            } else {
                // No items schema: any value
                let item_hint = format!("{}-item", name_hint);
                Self::compile_any_value(&item_hint, rules, counter)?
            };

            // Build the repeating part: ("," ws item)*
            let repeat_name = format!("{}-repeat", rule_name);
            rules.push(GrammarRule {
                name: repeat_name.clone(),
                alternatives: vec![GrammarAlternative {
                    elements: vec![
                        GrammarElement::Literal(",".to_string()),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::RuleRef(item_rule.clone()),
                    ],
                }],
            });

            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![
                    // Non-empty array
                    GrammarAlternative {
                        elements: vec![
                            GrammarElement::Literal("[".to_string()),
                            GrammarElement::RuleRef("ws".to_string()),
                            GrammarElement::RuleRef(item_rule),
                            GrammarElement::Repeat(
                                Box::new(GrammarElement::RuleRef(repeat_name)),
                                RepeatKind::ZeroOrMore,
                            ),
                            GrammarElement::RuleRef("ws".to_string()),
                            GrammarElement::Literal("]".to_string()),
                        ],
                    },
                    // Empty array
                    GrammarAlternative {
                        elements: vec![
                            GrammarElement::Literal("[".to_string()),
                            GrammarElement::RuleRef("ws".to_string()),
                            GrammarElement::Literal("]".to_string()),
                        ],
                    },
                ],
            });
            Ok(rule_name)
        }

        fn compile_any_value(
            name_hint: &str,
            rules: &mut Vec<GrammarRule>,
            counter: &mut Counter,
        ) -> Result<String, AiError> {
            let idx = counter.next();
            let rule_name = format!("{}-value-{}", name_hint, idx);

            // A generic JSON value: string | number | boolean | null
            let str_rule = Self::compile_string(&format!("{}-s{}", name_hint, idx), rules)?;
            let num_rule = Self::compile_number(&format!("{}-n{}", name_hint, idx), rules)?;
            let bool_rule =
                Self::compile_boolean(&format!("{}-b{}", name_hint, idx), rules)?;
            let null_rule = Self::compile_null(&format!("{}-nl{}", name_hint, idx), rules)?;

            rules.push(GrammarRule {
                name: rule_name.clone(),
                alternatives: vec![
                    GrammarAlternative {
                        elements: vec![GrammarElement::RuleRef(str_rule)],
                    },
                    GrammarAlternative {
                        elements: vec![GrammarElement::RuleRef(num_rule)],
                    },
                    GrammarAlternative {
                        elements: vec![GrammarElement::RuleRef(bool_rule)],
                    },
                    GrammarAlternative {
                        elements: vec![GrammarElement::RuleRef(null_rule)],
                    },
                ],
            });
            Ok(rule_name)
        }
    }

    /// Simple monotonic counter for generating unique rule names.
    struct Counter(usize);

    impl Counter {
        fn new() -> Self {
            Self(0)
        }

        fn next(&mut self) -> usize {
            let val = self.0;
            self.0 += 1;
            val
        }
    }

    // ========================================================================
    // 8.3 — Streaming Structured Validation
    // ========================================================================

    /// Outcome of feeding a token to the streaming validator.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ValidationState {
        /// The accumulated output so far is valid according to the schema.
        Valid,
        /// The output is partially valid; more tokens are expected.
        Partial {
            /// Human-readable description of what is expected next.
            expected_next: String,
        },
        /// A validation error was detected.
        Invalid {
            /// Byte position in the accumulated output where the error was found.
            position: usize,
            /// Explanation of the validation failure.
            reason: String,
        },
        /// The accumulated output is complete and valid.
        Complete,
    }

    /// Configuration for the streaming validator.
    #[derive(Debug, Clone)]
    pub struct StreamingValidationConfig {
        /// Maximum number of recovery attempts before giving up.
        pub max_recovery_attempts: usize,
        /// Whether to immediately abort on the first invalid token.
        pub abort_on_invalid: bool,
    }

    impl Default for StreamingValidationConfig {
        fn default() -> Self {
            Self {
                max_recovery_attempts: 3,
                abort_on_invalid: true,
            }
        }
    }

    /// What JSON construct we are currently inside.
    #[derive(Debug, Clone, PartialEq)]
    #[allow(dead_code)]
    enum JsonContext {
        /// Top level (nothing consumed yet or expecting a value).
        TopLevel,
        /// Inside an object, expecting a key or `}`.
        ObjectStart,
        /// Inside an object after seeing a key, expecting `:`.
        ObjectColon { key: String },
        /// Inside an object after `:`, expecting a value.
        ObjectValue { key: String },
        /// Inside an object after a value, expecting `,` or `}`.
        ObjectNext,
        /// Inside an array, expecting a value or `]`.
        ArrayStart,
        /// Inside an array after a value, expecting `,` or `]`.
        ArrayNext,
        /// Inside a string literal (accumulating chars).
        InString,
        /// Accumulating a keyword or number literal.
        InLiteral,
    }

    /// JSON schema type expectations for validation.
    #[derive(Debug, Clone)]
    enum ExpectedType {
        /// Any JSON value.
        Any,
        /// A string value.
        String,
        /// A numeric value.
        Number,
        /// An integer value.
        Integer,
        /// A boolean value.
        Boolean,
        /// A null value.
        Null,
        /// An object with expected properties.
        Object {
            properties: HashMap<String, serde_json::Value>,
            required: Vec<String>,
        },
        /// An array with expected item schema.
        Array {
            items: Option<Box<serde_json::Value>>,
        },
        /// One of the enumerated values.
        Enum(Vec<serde_json::Value>),
    }

    /// Incremental streaming validator for JSON output against a schema.
    ///
    /// Feed tokens one at a time and receive feedback about whether the partial
    /// output is still valid, needs more, or has become invalid.
    pub struct StreamingValidator {
        /// The original schema.
        schema: serde_json::Value,
        /// Accumulated raw text.
        buffer: String,
        /// Stack of JSON contexts (nesting).
        context_stack: Vec<JsonContext>,
        /// Stack of expected types (mirrors context for schema validation).
        type_stack: Vec<ExpectedType>,
        /// Configuration.
        config: StreamingValidationConfig,
        /// Number of recovery attempts used.
        recovery_attempts: usize,
        /// Whether validation has been completed successfully.
        completed: bool,
        /// Whether an unrecoverable error occurred.
        errored: bool,
    }

    impl StreamingValidator {
        /// Create a new streaming validator for the given JSON Schema.
        pub fn new(schema: serde_json::Value) -> Self {
            let expected = Self::schema_to_expected(&schema);
            Self {
                schema,
                buffer: String::new(),
                context_stack: vec![JsonContext::TopLevel],
                type_stack: vec![expected],
                config: StreamingValidationConfig::default(),
                recovery_attempts: 0,
                completed: false,
                errored: false,
            }
        }

        /// Create a new streaming validator with custom configuration.
        pub fn with_config(
            schema: serde_json::Value,
            config: StreamingValidationConfig,
        ) -> Self {
            let expected = Self::schema_to_expected(&schema);
            Self {
                schema,
                buffer: String::new(),
                context_stack: vec![JsonContext::TopLevel],
                type_stack: vec![expected],
                config,
                recovery_attempts: 0,
                completed: false,
                errored: false,
            }
        }

        /// Reset the validator state for reuse with the same schema.
        pub fn reset(&mut self) {
            let expected = Self::schema_to_expected(&self.schema);
            self.buffer.clear();
            self.context_stack = vec![JsonContext::TopLevel];
            self.type_stack = vec![expected];
            self.recovery_attempts = 0;
            self.completed = false;
            self.errored = false;
        }

        /// Feed a token (a string fragment) into the validator.
        pub fn feed_token(&mut self, token: &str) -> ValidationState {
            if self.completed {
                return ValidationState::Complete;
            }
            if self.errored {
                return ValidationState::Invalid {
                    position: self.buffer.len(),
                    reason: "validator is in error state; call reset() to retry".to_string(),
                };
            }

            self.buffer.push_str(token);

            // Try to validate the accumulated buffer
            self.validate_buffer()
        }

        /// Convert a JSON Schema value to our internal ExpectedType.
        fn schema_to_expected(schema: &serde_json::Value) -> ExpectedType {
            if let Some(enum_vals) = schema.get("enum") {
                if let Some(arr) = enum_vals.as_array() {
                    return ExpectedType::Enum(arr.clone());
                }
            }

            match schema.get("type").and_then(|t| t.as_str()) {
                Some("string") => ExpectedType::String,
                Some("number") => ExpectedType::Number,
                Some("integer") => ExpectedType::Integer,
                Some("boolean") => ExpectedType::Boolean,
                Some("null") => ExpectedType::Null,
                Some("object") => {
                    let properties = schema
                        .get("properties")
                        .and_then(|p| p.as_object())
                        .map(|obj| {
                            obj.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    let required = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    ExpectedType::Object {
                        properties,
                        required,
                    }
                }
                Some("array") => {
                    let items = schema.get("items").map(|i| Box::new(i.clone()));
                    ExpectedType::Array { items }
                }
                _ => ExpectedType::Any,
            }
        }

        /// Validate the current buffer contents.
        fn validate_buffer(&mut self) -> ValidationState {
            let buf = self.buffer.trim().to_string();

            if buf.is_empty() {
                return ValidationState::Partial {
                    expected_next: self.describe_expected(),
                };
            }

            // Check top-level type expectation against what we see
            let expected = self.type_stack.last().cloned().unwrap_or(ExpectedType::Any);

            match &expected {
                ExpectedType::String => self.validate_string_buffer(&buf),
                ExpectedType::Number | ExpectedType::Integer => {
                    self.validate_number_buffer(&buf, matches!(expected, ExpectedType::Integer))
                }
                ExpectedType::Boolean => self.validate_boolean_buffer(&buf),
                ExpectedType::Null => self.validate_null_buffer(&buf),
                ExpectedType::Object { properties, required } => {
                    self.validate_object_buffer(&buf, properties.clone(), required.clone())
                }
                ExpectedType::Array { items } => {
                    self.validate_array_buffer(&buf, items.clone())
                }
                ExpectedType::Enum(values) => self.validate_enum_buffer(&buf, values.clone()),
                ExpectedType::Any => self.validate_any_buffer(&buf),
            }
        }

        fn validate_string_buffer(&mut self, buf: &str) -> ValidationState {
            if !buf.starts_with('"') {
                self.mark_error();
                return ValidationState::Invalid {
                    position: 0,
                    reason: "expected string to start with '\"'".to_string(),
                };
            }
            // Check if string is complete (ends with unescaped ")
            if buf.len() >= 2 && is_string_complete(buf) {
                self.completed = true;
                return ValidationState::Complete;
            }
            ValidationState::Partial {
                expected_next: "string content or closing '\"'".to_string(),
            }
        }

        fn validate_number_buffer(
            &mut self,
            buf: &str,
            integer_only: bool,
        ) -> ValidationState {
            // A number can be: -?[0-9]+(\.[0-9]+)?
            let mut valid_so_far = true;
            let mut has_digit = false;
            let mut has_dot = false;
            let mut has_digit_after_dot = false;

            for (i, c) in buf.chars().enumerate() {
                match c {
                    '-' if i == 0 => {}
                    '0'..='9' => {
                        has_digit = true;
                        if has_dot {
                            has_digit_after_dot = true;
                        }
                    }
                    '.' if !integer_only && has_digit && !has_dot => {
                        has_dot = true;
                    }
                    _ => {
                        valid_so_far = false;
                        break;
                    }
                }
            }

            if !valid_so_far {
                self.mark_error();
                return ValidationState::Invalid {
                    position: buf.len().saturating_sub(1),
                    reason: if integer_only {
                        "invalid integer character".to_string()
                    } else {
                        "invalid number character".to_string()
                    },
                };
            }

            if has_digit && (!has_dot || has_digit_after_dot) {
                // Could be complete or more digits could follow
                self.completed = true;
                ValidationState::Complete
            } else {
                ValidationState::Partial {
                    expected_next: if has_dot {
                        "digits after decimal point".to_string()
                    } else {
                        "digits".to_string()
                    },
                }
            }
        }

        fn validate_boolean_buffer(&mut self, buf: &str) -> ValidationState {
            if "true".starts_with(buf) || "false".starts_with(buf) {
                if buf == "true" || buf == "false" {
                    self.completed = true;
                    return ValidationState::Complete;
                }
                return ValidationState::Partial {
                    expected_next: "remaining boolean characters".to_string(),
                };
            }
            self.mark_error();
            ValidationState::Invalid {
                position: buf.len().saturating_sub(1),
                reason: "expected 'true' or 'false'".to_string(),
            }
        }

        fn validate_null_buffer(&mut self, buf: &str) -> ValidationState {
            if "null".starts_with(buf) {
                if buf == "null" {
                    self.completed = true;
                    return ValidationState::Complete;
                }
                return ValidationState::Partial {
                    expected_next: "remaining 'null' characters".to_string(),
                };
            }
            self.mark_error();
            ValidationState::Invalid {
                position: buf.len().saturating_sub(1),
                reason: "expected 'null'".to_string(),
            }
        }

        fn validate_object_buffer(
            &mut self,
            buf: &str,
            properties: HashMap<String, serde_json::Value>,
            required: Vec<String>,
        ) -> ValidationState {
            if !buf.starts_with('{') {
                self.mark_error();
                return ValidationState::Invalid {
                    position: 0,
                    reason: "expected object to start with '{'".to_string(),
                };
            }

            // Try to parse as complete JSON
            match serde_json::from_str::<serde_json::Value>(buf) {
                Ok(val) => {
                    if let Some(obj) = val.as_object() {
                        // Check required properties
                        for req_key in &required {
                            if !obj.contains_key(req_key) {
                                self.mark_error();
                                return ValidationState::Invalid {
                                    position: buf.len(),
                                    reason: format!(
                                        "missing required property '{}'",
                                        req_key
                                    ),
                                };
                            }
                        }
                        // Validate property types
                        for (key, value) in obj {
                            if let Some(prop_schema) = properties.get(key) {
                                if !value_matches_schema(value, prop_schema) {
                                    self.mark_error();
                                    return ValidationState::Invalid {
                                        position: buf.len(),
                                        reason: format!(
                                            "property '{}' does not match schema",
                                            key
                                        ),
                                    };
                                }
                            }
                        }
                        self.completed = true;
                        ValidationState::Complete
                    } else {
                        self.mark_error();
                        ValidationState::Invalid {
                            position: 0,
                            reason: "expected an object".to_string(),
                        }
                    }
                }
                Err(_) => {
                    // Not complete yet — check basic structure validity
                    if is_partial_json_valid(buf) {
                        let prop_names: Vec<&str> =
                            properties.keys().map(|s| s.as_str()).collect();
                        ValidationState::Partial {
                            expected_next: if prop_names.is_empty() {
                                "object content or '}'".to_string()
                            } else {
                                format!(
                                    "property (one of: {}) or '}}'",
                                    prop_names.join(", ")
                                )
                            },
                        }
                    } else {
                        self.mark_error();
                        ValidationState::Invalid {
                            position: buf.len(),
                            reason: "malformed JSON object".to_string(),
                        }
                    }
                }
            }
        }

        fn validate_array_buffer(
            &mut self,
            buf: &str,
            items: Option<Box<serde_json::Value>>,
        ) -> ValidationState {
            if !buf.starts_with('[') {
                self.mark_error();
                return ValidationState::Invalid {
                    position: 0,
                    reason: "expected array to start with '['".to_string(),
                };
            }

            match serde_json::from_str::<serde_json::Value>(buf) {
                Ok(val) => {
                    if let Some(arr) = val.as_array() {
                        // Validate item types if schema provided
                        if let Some(item_schema) = &items {
                            for (i, item) in arr.iter().enumerate() {
                                if !value_matches_schema(item, item_schema) {
                                    self.mark_error();
                                    return ValidationState::Invalid {
                                        position: buf.len(),
                                        reason: format!(
                                            "array item {} does not match schema",
                                            i
                                        ),
                                    };
                                }
                            }
                        }
                        self.completed = true;
                        ValidationState::Complete
                    } else {
                        self.mark_error();
                        ValidationState::Invalid {
                            position: 0,
                            reason: "expected an array".to_string(),
                        }
                    }
                }
                Err(_) => {
                    if is_partial_json_valid(buf) {
                        ValidationState::Partial {
                            expected_next: "array item or ']'".to_string(),
                        }
                    } else {
                        self.mark_error();
                        ValidationState::Invalid {
                            position: buf.len(),
                            reason: "malformed JSON array".to_string(),
                        }
                    }
                }
            }
        }

        fn validate_enum_buffer(
            &mut self,
            buf: &str,
            values: Vec<serde_json::Value>,
        ) -> ValidationState {
            // Check if buf matches any enum value completely
            for val in &values {
                let serialized = match val {
                    serde_json::Value::String(s) => format!("\"{}\"", s),
                    other => other.to_string(),
                };
                if serialized == buf {
                    self.completed = true;
                    return ValidationState::Complete;
                }
                if serialized.starts_with(buf) {
                    return ValidationState::Partial {
                        expected_next: format!("one of enum values: {:?}", values),
                    };
                }
            }

            // Could be a partial match in progress
            let any_prefix = values.iter().any(|val| {
                let serialized = match val {
                    serde_json::Value::String(s) => format!("\"{}\"", s),
                    other => other.to_string(),
                };
                serialized.starts_with(buf)
            });

            if any_prefix {
                ValidationState::Partial {
                    expected_next: format!("one of enum values: {:?}", values),
                }
            } else {
                self.mark_error();
                ValidationState::Invalid {
                    position: buf.len().saturating_sub(1),
                    reason: format!("value does not match any enum variant: {:?}", values),
                }
            }
        }

        fn validate_any_buffer(&mut self, buf: &str) -> ValidationState {
            // Try to parse as any JSON
            match serde_json::from_str::<serde_json::Value>(buf) {
                Ok(_) => {
                    self.completed = true;
                    ValidationState::Complete
                }
                Err(_) => {
                    if is_partial_json_valid(buf) {
                        ValidationState::Partial {
                            expected_next: "JSON value".to_string(),
                        }
                    } else {
                        self.mark_error();
                        ValidationState::Invalid {
                            position: buf.len(),
                            reason: "invalid JSON".to_string(),
                        }
                    }
                }
            }
        }

        fn describe_expected(&self) -> String {
            match self.type_stack.last() {
                Some(ExpectedType::String) => "a JSON string".to_string(),
                Some(ExpectedType::Number) => "a JSON number".to_string(),
                Some(ExpectedType::Integer) => "a JSON integer".to_string(),
                Some(ExpectedType::Boolean) => "'true' or 'false'".to_string(),
                Some(ExpectedType::Null) => "'null'".to_string(),
                Some(ExpectedType::Object { .. }) => "a JSON object".to_string(),
                Some(ExpectedType::Array { .. }) => "a JSON array".to_string(),
                Some(ExpectedType::Enum(vals)) => {
                    format!("one of: {:?}", vals)
                }
                Some(ExpectedType::Any) | None => "a JSON value".to_string(),
            }
        }

        fn mark_error(&mut self) {
            self.recovery_attempts += 1;
            if self.config.abort_on_invalid
                || self.recovery_attempts > self.config.max_recovery_attempts
            {
                self.errored = true;
            }
        }
    }

    // ========================================================================
    // Validation helper functions
    // ========================================================================

    /// Check if a string literal is complete (ends with unescaped `"`).
    fn is_string_complete(s: &str) -> bool {
        if s.len() < 2 || !s.starts_with('"') {
            return false;
        }
        if !s.ends_with('"') {
            return false;
        }
        // Count trailing backslashes before the final quote
        let inner = &s[1..s.len() - 1];
        let trailing_backslashes = inner.chars().rev().take_while(|c| *c == '\\').count();
        trailing_backslashes % 2 == 0
    }

    /// Check if a partial JSON string is structurally plausible
    /// (balanced brackets so far, valid characters, etc.).
    fn is_partial_json_valid(s: &str) -> bool {
        let mut depth_brace = 0i32;
        let mut depth_bracket = 0i32;
        let mut in_string = false;
        let mut escape_next = false;

        for c in s.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if in_string {
                if c == '\\' {
                    escape_next = true;
                } else if c == '"' {
                    in_string = false;
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                '{' => depth_brace += 1,
                '}' => {
                    depth_brace -= 1;
                    if depth_brace < 0 {
                        return false;
                    }
                }
                '[' => depth_bracket += 1,
                ']' => {
                    depth_bracket -= 1;
                    if depth_bracket < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }

        // Depth must be >= 0 (can be > 0 since it's partial)
        depth_brace >= 0 && depth_bracket >= 0
    }

    /// Check if a JSON value matches a JSON Schema type definition.
    fn value_matches_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
        let type_str = schema.get("type").and_then(|t| t.as_str());

        // Handle enum
        if let Some(enum_vals) = schema.get("enum") {
            if let Some(arr) = enum_vals.as_array() {
                return arr.contains(value);
            }
        }

        match type_str {
            Some("string") => value.is_string(),
            Some("number") => value.is_number(),
            Some("integer") => value.is_i64() || value.is_u64(),
            Some("boolean") => value.is_boolean(),
            Some("null") => value.is_null(),
            Some("object") => {
                if let Some(obj) = value.as_object() {
                    // Validate required properties
                    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                        for req in required {
                            if let Some(key) = req.as_str() {
                                if !obj.contains_key(key) {
                                    return false;
                                }
                            }
                        }
                    }
                    // Validate property types
                    if let Some(properties) =
                        schema.get("properties").and_then(|p| p.as_object())
                    {
                        for (key, val) in obj {
                            if let Some(prop_schema) = properties.get(key) {
                                if !value_matches_schema(val, prop_schema) {
                                    return false;
                                }
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            Some("array") => {
                if let Some(arr) = value.as_array() {
                    if let Some(item_schema) = schema.get("items") {
                        arr.iter().all(|item| value_matches_schema(item, item_schema))
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            _ => true, // unknown type => accept
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        // ----------------------------------------------------------------
        // GrammarBuilder tests
        // ----------------------------------------------------------------

        #[test]
        fn test_grammar_builder_simple() {
            let mut builder = GrammarBuilder::new("root");
            let rule = builder
                .rule("root")
                .alt()
                .literal("hello")
                .done()
                .done();
            builder.add_rule(rule);
            let grammar = builder.build();

            assert_eq!(grammar.root_rule, "root");
            assert_eq!(grammar.rules.len(), 1);
            assert_eq!(grammar.rules[0].name, "root");
            assert_eq!(grammar.rules[0].alternatives.len(), 1);
            assert_eq!(
                grammar.rules[0].alternatives[0].elements,
                vec![GrammarElement::Literal("hello".to_string())]
            );
        }

        #[test]
        fn test_grammar_builder_complex_nested() {
            let mut builder = GrammarBuilder::new("root");

            let root_rule = builder
                .rule("root")
                .alt()
                .rule_ref("object")
                .done()
                .done();
            builder.add_rule(root_rule);

            let object_rule = builder
                .rule("object")
                .alt()
                .literal("{")
                .rule_ref("ws")
                .rule_ref("members")
                .rule_ref("ws")
                .literal("}")
                .done()
                .done();
            builder.add_rule(object_rule);

            let ws_rule = builder
                .rule("ws")
                .alt()
                .repeat(
                    GrammarElement::CharSet(vec![' ', '\t', '\n']),
                    RepeatKind::ZeroOrMore,
                )
                .done()
                .done();
            builder.add_rule(ws_rule);

            let grammar = builder.build();
            assert_eq!(grammar.rules.len(), 3);
            assert_eq!(grammar.root_rule, "root");

            // Verify the object rule has the expected structure
            let obj = &grammar.rules[1];
            assert_eq!(obj.name, "object");
            assert_eq!(obj.alternatives[0].elements.len(), 5);
        }

        #[test]
        fn test_grammar_builder_empty() {
            let builder = GrammarBuilder::new("root");
            let grammar = builder.build();
            assert_eq!(grammar.rules.len(), 0);
            assert_eq!(grammar.root_rule, "root");
        }

        // ----------------------------------------------------------------
        // Grammar::to_gbnf tests
        // ----------------------------------------------------------------

        #[test]
        fn test_to_gbnf_simple_rule() {
            let grammar = Grammar {
                rules: vec![GrammarRule {
                    name: "root".to_string(),
                    alternatives: vec![GrammarAlternative {
                        elements: vec![GrammarElement::Literal("hello".to_string())],
                    }],
                }],
                root_rule: "root".to_string(),
            };
            let gbnf = grammar.to_gbnf();
            assert_eq!(gbnf, "root ::= \"hello\"");
        }

        #[test]
        fn test_to_gbnf_multiple_rules() {
            let grammar = Grammar {
                rules: vec![
                    GrammarRule {
                        name: "root".to_string(),
                        alternatives: vec![GrammarAlternative {
                            elements: vec![GrammarElement::RuleRef("value".to_string())],
                        }],
                    },
                    GrammarRule {
                        name: "value".to_string(),
                        alternatives: vec![
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("true".to_string())],
                            },
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("false".to_string())],
                            },
                        ],
                    },
                ],
                root_rule: "root".to_string(),
            };
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("root ::= value"));
            assert!(gbnf.contains("value ::= \"true\" | \"false\""));
        }

        #[test]
        fn test_to_gbnf_special_characters() {
            let grammar = Grammar {
                rules: vec![GrammarRule {
                    name: "root".to_string(),
                    alternatives: vec![GrammarAlternative {
                        elements: vec![
                            GrammarElement::Literal("\"".to_string()),
                            GrammarElement::NegatedCharSet(vec!['"', '\\']),
                            GrammarElement::Literal("\"".to_string()),
                        ],
                    }],
                }],
                root_rule: "root".to_string(),
            };
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\\\""));
            assert!(gbnf.contains("[^"));
        }

        #[test]
        fn test_to_gbnf_round_trip() {
            let original = Grammar {
                rules: vec![
                    GrammarRule {
                        name: "root".to_string(),
                        alternatives: vec![GrammarAlternative {
                            elements: vec![GrammarElement::RuleRef("bool".to_string())],
                        }],
                    },
                    GrammarRule {
                        name: "bool".to_string(),
                        alternatives: vec![
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("true".to_string())],
                            },
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("false".to_string())],
                            },
                        ],
                    },
                ],
                root_rule: "root".to_string(),
            };

            let gbnf_str = original.to_gbnf();
            let parsed = Grammar::from_gbnf(&gbnf_str).expect("should parse");

            assert_eq!(parsed.root_rule, "root");
            assert_eq!(parsed.rules.len(), 2);
            // Re-serialize and compare
            let round_trip = parsed.to_gbnf();
            assert_eq!(gbnf_str, round_trip);
        }

        // ----------------------------------------------------------------
        // Grammar::from_gbnf tests
        // ----------------------------------------------------------------

        #[test]
        fn test_from_gbnf_valid() {
            let input = r#"root ::= "hello" | "world""#;
            let grammar = Grammar::from_gbnf(input).expect("should parse");
            assert_eq!(grammar.root_rule, "root");
            assert_eq!(grammar.rules.len(), 1);
            assert_eq!(grammar.rules[0].alternatives.len(), 2);
        }

        #[test]
        fn test_from_gbnf_char_ranges() {
            let input = "digit ::= [0-9]";
            let grammar = Grammar::from_gbnf(input).expect("should parse");
            assert_eq!(grammar.rules[0].name, "digit");
            let elem = &grammar.rules[0].alternatives[0].elements[0];
            assert_eq!(*elem, GrammarElement::CharRange('0', '9'));
        }

        #[test]
        fn test_from_gbnf_repetition() {
            let input = "digits ::= [0-9]+";
            let grammar = Grammar::from_gbnf(input).expect("should parse");
            let elem = &grammar.rules[0].alternatives[0].elements[0];
            match elem {
                GrammarElement::Repeat(inner, RepeatKind::OneOrMore) => {
                    assert_eq!(**inner, GrammarElement::CharRange('0', '9'));
                }
                other => panic!("expected Repeat, got {:?}", other),
            }
        }

        #[test]
        fn test_from_gbnf_error_handling() {
            // Missing ::= separator
            let result = Grammar::from_gbnf("root = hello");
            assert!(result.is_err());

            // Empty grammar
            let result = Grammar::from_gbnf("");
            assert!(result.is_err());

            // Comment-only grammar
            let result = Grammar::from_gbnf("# just a comment");
            assert!(result.is_err());
        }

        // ----------------------------------------------------------------
        // SchemaToGrammar tests
        // ----------------------------------------------------------------

        #[test]
        fn test_schema_string_type() {
            let schema = serde_json::json!({"type": "string"});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("root"));
            assert!(gbnf.contains("string"));
            assert!(gbnf.contains("\\\""));
        }

        #[test]
        fn test_schema_number_type() {
            let schema = serde_json::json!({"type": "number"});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("number"));
            assert!(gbnf.contains("[0-9]"));
        }

        #[test]
        fn test_schema_integer_type() {
            let schema = serde_json::json!({"type": "integer"});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("integer"));
            assert!(gbnf.contains("[0-9]"));
            // Integer should not have a decimal point rule
            assert!(!gbnf.contains("\".\""));
        }

        #[test]
        fn test_schema_boolean_type() {
            let schema = serde_json::json!({"type": "boolean"});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"true\""));
            assert!(gbnf.contains("\"false\""));
        }

        #[test]
        fn test_schema_null_type() {
            let schema = serde_json::json!({"type": "null"});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"null\""));
        }

        #[test]
        fn test_schema_enum_type() {
            let schema = serde_json::json!({"enum": ["red", "green", "blue"]});
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"\\\"red\\\"\""));
            assert!(gbnf.contains("\"\\\"green\\\"\""));
            assert!(gbnf.contains("\"\\\"blue\\\"\""));
        }

        #[test]
        fn test_schema_simple_object() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"{\""));
            assert!(gbnf.contains("\"}\""));
            assert!(gbnf.contains("\"\\\"name\\\"\""));
        }

        #[test]
        fn test_schema_nested_object() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"}
                        }
                    }
                }
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            // Should contain rules for the nested object
            assert!(gbnf.contains("address"));
            assert!(gbnf.contains("street"));
            assert!(gbnf.contains("city"));
        }

        #[test]
        fn test_schema_array_of_strings() {
            let schema = serde_json::json!({
                "type": "array",
                "items": {"type": "string"}
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"[\""));
            assert!(gbnf.contains("\"]\""));
            assert!(gbnf.contains("string"));
        }

        #[test]
        fn test_schema_array_of_objects() {
            let schema = serde_json::json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"}
                    },
                    "required": ["id"]
                }
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("array"));
            assert!(gbnf.contains("object"));
            assert!(gbnf.contains("id"));
        }

        #[test]
        fn test_schema_required_fields() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "number"},
                    "c": {"type": "boolean"}
                },
                "required": ["a", "b"]
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            // Grammar should compile without error and contain all fields
            assert!(!grammar.rules.is_empty());
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"\\\"a\\\"\""));
            assert!(gbnf.contains("\"\\\"b\\\"\""));
        }

        #[test]
        fn test_schema_mixed_types() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "active": {"type": "boolean"},
                    "metadata": {"type": "null"}
                }
            });
            let grammar = SchemaToGrammar::compile(&schema).expect("should compile");
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("string"));
            assert!(gbnf.contains("integer"));
            assert!(gbnf.contains("number"));
            assert!(gbnf.contains("boolean"));
            assert!(gbnf.contains("null"));
        }

        // ----------------------------------------------------------------
        // GrammarConstraint tests
        // ----------------------------------------------------------------

        #[test]
        fn test_constraint_for_gbnf_provider() {
            let grammar = Grammar {
                rules: vec![GrammarRule {
                    name: "root".to_string(),
                    alternatives: vec![GrammarAlternative {
                        elements: vec![GrammarElement::Literal("test".to_string())],
                    }],
                }],
                root_rule: "root".to_string(),
            };

            let result = GrammarConstraint::for_provider(&grammar, "ollama");
            assert!(result.is_ok());
            let gbnf = result.unwrap();
            assert!(gbnf.contains("root ::="));
        }

        #[test]
        fn test_constraint_for_unsupported_provider() {
            let grammar = Grammar {
                rules: vec![],
                root_rule: "root".to_string(),
            };

            let result = GrammarConstraint::for_provider(&grammar, "unknown_provider");
            assert!(result.is_err());
            if let Err(AiError::ConstrainedDecoding(
                ConstrainedDecodingError::ProviderUnsupported { provider },
            )) = &result
            {
                assert_eq!(provider, "unknown_provider");
            } else {
                panic!("expected ProviderUnsupported error");
            }
        }

        // ----------------------------------------------------------------
        // StreamingValidator tests
        // ----------------------------------------------------------------

        #[test]
        fn test_streaming_valid_object_token_by_token() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            });

            let mut validator = StreamingValidator::new(schema);

            // Feed token by token
            let state = validator.feed_token("{");
            assert!(
                matches!(state, ValidationState::Partial { .. }),
                "expected Partial, got {:?}",
                state
            );

            let state = validator.feed_token("\"name\"");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token(": ");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token("\"Alice\"");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token("}");
            assert_eq!(state, ValidationState::Complete);
        }

        #[test]
        fn test_streaming_valid_array() {
            let schema = serde_json::json!({
                "type": "array",
                "items": {"type": "number"}
            });

            let mut validator = StreamingValidator::new(schema);

            let state = validator.feed_token("[1, 2, ");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token("3]");
            assert_eq!(state, ValidationState::Complete);
        }

        #[test]
        fn test_streaming_partial_validation() {
            let schema = serde_json::json!({"type": "string"});
            let mut validator = StreamingValidator::new(schema);

            let state = validator.feed_token("\"hel");
            assert!(
                matches!(state, ValidationState::Partial { .. }),
                "got {:?}",
                state
            );
        }

        #[test]
        fn test_streaming_invalid_token() {
            let schema = serde_json::json!({"type": "boolean"});
            let mut validator = StreamingValidator::new(schema);

            let state = validator.feed_token("xyz");
            assert!(
                matches!(state, ValidationState::Invalid { .. }),
                "got {:?}",
                state
            );
        }

        #[test]
        fn test_streaming_recovery_attempt() {
            let schema = serde_json::json!({"type": "number"});
            let mut validator = StreamingValidator::with_config(
                schema,
                StreamingValidationConfig {
                    max_recovery_attempts: 3,
                    abort_on_invalid: false,
                },
            );

            // Feed something partially valid then invalid
            let state = validator.feed_token("12abc");
            assert!(
                matches!(state, ValidationState::Invalid { .. }),
                "got {:?}",
                state
            );

            // Since abort_on_invalid is false and we haven't exceeded max_recovery,
            // we can still feed tokens (though the buffer is corrupted).
            // The validator should track attempts.
            assert_eq!(validator.recovery_attempts, 1);
        }

        #[test]
        fn test_streaming_complete_detection() {
            let schema = serde_json::json!({"type": "boolean"});
            let mut validator = StreamingValidator::new(schema);

            let state = validator.feed_token("true");
            assert_eq!(state, ValidationState::Complete);

            // Feeding more tokens after completion should return Complete
            let state = validator.feed_token("extra");
            assert_eq!(state, ValidationState::Complete);
        }

        #[test]
        fn test_streaming_nested_object() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {
                            "inner": {"type": "string"}
                        }
                    }
                }
            });

            let mut validator = StreamingValidator::new(schema);

            let state = validator.feed_token("{\"outer\":");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token("{\"inner\":\"val\"}}");
            assert_eq!(state, ValidationState::Complete);
        }

        #[test]
        fn test_streaming_reset_and_reuse() {
            let schema = serde_json::json!({"type": "boolean"});
            let mut validator = StreamingValidator::new(schema);

            // First use
            let state = validator.feed_token("true");
            assert_eq!(state, ValidationState::Complete);

            // Reset
            validator.reset();

            // Second use with different value
            let state = validator.feed_token("fals");
            assert!(matches!(state, ValidationState::Partial { .. }));

            let state = validator.feed_token("e");
            assert_eq!(state, ValidationState::Complete);
        }

        // ----------------------------------------------------------------
        // Additional tests for coverage
        // ----------------------------------------------------------------

        #[test]
        fn test_constraint_for_vllm_provider() {
            let grammar = Grammar {
                rules: vec![
                    GrammarRule {
                        name: "root".to_string(),
                        alternatives: vec![GrammarAlternative {
                            elements: vec![GrammarElement::RuleRef("bool".to_string())],
                        }],
                    },
                    GrammarRule {
                        name: "bool".to_string(),
                        alternatives: vec![
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("true".to_string())],
                            },
                            GrammarAlternative {
                                elements: vec![GrammarElement::Literal("false".to_string())],
                            },
                        ],
                    },
                ],
                root_rule: "root".to_string(),
            };

            let result = GrammarConstraint::for_provider(&grammar, "vllm");
            assert!(result.is_ok());
            let regex_str = result.unwrap();
            // Should contain regex alternation for true|false
            assert!(
                regex_str.contains("true") && regex_str.contains("false"),
                "regex should contain both alternatives: {}",
                regex_str
            );
        }

        #[test]
        fn test_grammar_builder_multiple_alternatives() {
            let mut builder = GrammarBuilder::new("root");
            let rule = builder
                .rule("root")
                .alt()
                .literal("yes")
                .done()
                .alt()
                .literal("no")
                .done()
                .alt()
                .literal("maybe")
                .done()
                .done();
            builder.add_rule(rule);
            let grammar = builder.build();

            assert_eq!(grammar.rules[0].alternatives.len(), 3);
            let gbnf = grammar.to_gbnf();
            assert!(gbnf.contains("\"yes\""));
            assert!(gbnf.contains("\"no\""));
            assert!(gbnf.contains("\"maybe\""));
            assert!(gbnf.contains(" | "));
        }

        #[test]
        fn test_streaming_integer_validation() {
            let schema = serde_json::json!({"type": "integer"});
            let mut validator = StreamingValidator::new(schema);

            // Valid integer
            let state = validator.feed_token("42");
            assert_eq!(state, ValidationState::Complete);

            validator.reset();

            // Negative integer
            let state = validator.feed_token("-7");
            assert_eq!(state, ValidationState::Complete);

            validator.reset();

            // Decimal not allowed for integer
            let state = validator.feed_token("3.14");
            assert!(
                matches!(state, ValidationState::Invalid { .. }),
                "decimals should be invalid for integer type, got {:?}",
                state
            );
        }
    }
}

#[cfg(feature = "constrained-decoding")]
pub use inner::*;
