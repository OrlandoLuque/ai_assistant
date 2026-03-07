//! Entity extraction and fact tracking
//!
//! This module provides lightweight NER (Named Entity Recognition) and fact
//! tracking for conversations without requiring external ML models.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Entity Types
// ============================================================================

/// Types of entities that can be extracted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person name
    Person,
    /// Organization or company
    Organization,
    /// Location (city, country, address)
    Location,
    /// Date or time reference
    DateTime,
    /// Monetary amount
    Money,
    /// Percentage value
    Percentage,
    /// Email address
    Email,
    /// URL
    Url,
    /// Phone number
    Phone,
    /// Programming language
    ProgrammingLanguage,
    /// File path or name
    FilePath,
    /// Code/technical term
    TechnicalTerm,
    /// Version number (e.g., v1.2.3)
    Version,
    /// Custom entity type
    Custom(u32),
}

impl EntityType {
    /// Get human-readable name for this entity type
    pub fn display_name(&self) -> &'static str {
        match self {
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::DateTime => "Date/Time",
            EntityType::Money => "Money",
            EntityType::Percentage => "Percentage",
            EntityType::Email => "Email",
            EntityType::Url => "URL",
            EntityType::Phone => "Phone",
            EntityType::ProgrammingLanguage => "Language",
            EntityType::FilePath => "File",
            EntityType::TechnicalTerm => "Technical",
            EntityType::Version => "Version",
            EntityType::Custom(_) => "Custom",
        }
    }
}

/// An extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// The entity text as found
    pub text: String,
    /// Normalized form (lowercase, trimmed)
    pub normalized: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Character offset in source text
    pub start_offset: usize,
    /// End offset
    pub end_offset: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Entity {
    pub fn new(
        text: &str,
        entity_type: EntityType,
        confidence: f32,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            text: text.to_string(),
            normalized: text.to_lowercase().trim().to_string(),
            entity_type,
            confidence,
            start_offset: start,
            end_offset: end,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ============================================================================
// Entity Extractor
// ============================================================================

/// Configuration for entity extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractorConfig {
    /// Minimum confidence to keep an entity
    pub min_confidence: f32,
    /// Enable person name extraction
    pub extract_persons: bool,
    /// Enable organization extraction
    pub extract_organizations: bool,
    /// Enable location extraction
    pub extract_locations: bool,
    /// Enable technical terms
    pub extract_technical: bool,
    /// Enable URLs and emails
    pub extract_links: bool,
    /// Custom patterns to match (regex-like simple patterns)
    pub custom_patterns: Vec<CustomPattern>,
}

impl Default for EntityExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            extract_persons: true,
            extract_organizations: true,
            extract_locations: true,
            extract_technical: true,
            extract_links: true,
            custom_patterns: Vec::new(),
        }
    }
}

/// A custom pattern for entity extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    /// Pattern name
    pub name: String,
    /// Keywords that trigger this pattern
    pub keywords: Vec<String>,
    /// Entity type to assign
    pub entity_type: EntityType,
    /// Base confidence for matches
    pub confidence: f32,
}

/// Lightweight entity extractor using pattern matching
pub struct EntityExtractor {
    config: EntityExtractorConfig,
    /// Common programming languages
    programming_languages: Vec<&'static str>,
    /// Common organization suffixes
    org_suffixes: Vec<&'static str>,
    /// Title prefixes that indicate a person
    person_titles: Vec<&'static str>,
}

impl EntityExtractor {
    pub fn new(config: EntityExtractorConfig) -> Self {
        Self {
            config,
            programming_languages: vec![
                "rust",
                "python",
                "javascript",
                "typescript",
                "java",
                "c++",
                "c#",
                "go",
                "golang",
                "ruby",
                "php",
                "swift",
                "kotlin",
                "scala",
                "haskell",
                "lua",
                "perl",
                "r",
                "julia",
                "dart",
                "elixir",
                "clojure",
                "sql",
                "html",
                "css",
                "bash",
                "shell",
                "powershell",
            ],
            org_suffixes: vec![
                "inc",
                "inc.",
                "corp",
                "corp.",
                "corporation",
                "llc",
                "ltd",
                "ltd.",
                "gmbh",
                "co",
                "co.",
                "company",
                "group",
                "holdings",
                "enterprises",
                "technologies",
                "tech",
                "software",
                "systems",
                "solutions",
            ],
            person_titles: vec![
                "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof.", "sir",
                "madam", "lord", "lady", "captain", "cpt", "general", "gen",
            ],
        }
    }

    /// Extract all entities from text
    pub fn extract(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Extract different entity types
        if self.config.extract_links {
            entities.extend(self.extract_emails(text));
            entities.extend(self.extract_urls(text));
            entities.extend(self.extract_phones(text));
        }

        if self.config.extract_technical {
            entities.extend(self.extract_programming_languages(text));
            entities.extend(self.extract_file_paths(text));
            entities.extend(self.extract_versions(text));
        }

        entities.extend(self.extract_money(text));
        entities.extend(self.extract_percentages(text));

        if self.config.extract_persons {
            entities.extend(self.extract_persons(text));
        }

        if self.config.extract_organizations {
            entities.extend(self.extract_organizations(text));
        }

        // Apply custom patterns
        for pattern in &self.config.custom_patterns {
            entities.extend(self.apply_custom_pattern(text, pattern));
        }

        // Filter by confidence and deduplicate
        entities.retain(|e| e.confidence >= self.config.min_confidence);
        self.deduplicate_entities(entities)
    }

    /// Extract email addresses
    fn extract_emails(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Simple email pattern: word@word.word
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if word.contains('@') && word.contains('.') {
                let at_pos = word.find('@').expect("char verified in format check");
                let dot_pos = word.rfind('.').expect("char verified in format check");

                if at_pos > 0 && dot_pos > at_pos + 1 && dot_pos < word.len() - 1 {
                    // Clean up the word
                    let clean = word.trim_matches(|c: char| {
                        !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-'
                    });
                    if let Some(start) = text.find(clean) {
                        entities.push(Entity::new(
                            clean,
                            EntityType::Email,
                            0.95,
                            start,
                            start + clean.len(),
                        ));
                    }
                }
            }
        }

        entities
    }

    /// Extract URLs
    fn extract_urls(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        let prefixes = ["http://", "https://", "www."];

        for prefix in prefixes {
            let mut search_start = 0;
            while let Some(pos) = text[search_start..].find(prefix) {
                let abs_pos = search_start + pos;
                // Find end of URL (whitespace or common punctuation at end)
                let remaining = &text[abs_pos..];
                let end = remaining
                    .find(|c: char| {
                        c.is_whitespace()
                            || c == '"'
                            || c == '\''
                            || c == '>'
                            || c == ')'
                            || c == ']'
                    })
                    .unwrap_or(remaining.len());

                let url = &remaining[..end];
                if url.len() > prefix.len() + 3 {
                    entities.push(Entity::new(
                        url,
                        EntityType::Url,
                        0.95,
                        abs_pos,
                        abs_pos + end,
                    ));
                }

                search_start = abs_pos + end;
            }
        }

        entities
    }

    /// Extract phone numbers
    fn extract_phones(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Look for phone number patterns
            if chars[i] == '+' || chars[i].is_ascii_digit() || chars[i] == '(' {
                let start = i;
                let mut digit_count = 0;
                let mut len = 0;

                while i < chars.len() && len < 20 {
                    let c = chars[i];
                    if c.is_ascii_digit() {
                        digit_count += 1;
                        len += 1;
                        i += 1;
                    } else if c == ' ' || c == '-' || c == '(' || c == ')' || c == '+' || c == '.' {
                        len += 1;
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Valid phone numbers have 7-15 digits
                if digit_count >= 7 && digit_count <= 15 {
                    let phone: String = chars[start..i].iter().collect();
                    let phone = phone.trim();
                    if !phone.is_empty() {
                        entities.push(Entity::new(phone, EntityType::Phone, 0.8, start, i));
                    }
                }
            } else {
                i += 1;
            }
        }

        entities
    }

    /// Extract programming languages
    fn extract_programming_languages(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();

        for lang in &self.programming_languages {
            let mut search_start = 0;
            while let Some(pos) = text_lower[search_start..].find(lang) {
                let abs_pos = search_start + pos;
                // Check word boundaries
                let before_ok = abs_pos == 0
                    || !text
                        .chars()
                        .nth(abs_pos - 1)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let after_ok = abs_pos + lang.len() >= text.len()
                    || !text
                        .chars()
                        .nth(abs_pos + lang.len())
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if before_ok && after_ok {
                    let original = &text[abs_pos..abs_pos + lang.len()];
                    entities.push(Entity::new(
                        original,
                        EntityType::ProgrammingLanguage,
                        0.85,
                        abs_pos,
                        abs_pos + lang.len(),
                    ));
                }

                search_start = abs_pos + lang.len();
            }
        }

        entities
    }

    /// Extract file paths
    fn extract_file_paths(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Windows paths: C:\..., D:\...
        // Unix paths: /home/..., ./..., ../...
        // File extensions: .rs, .py, .js, etc.

        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            let is_path = word.contains('/')
                && (word.starts_with('/') || word.starts_with("./") || word.starts_with("../"))
                || (word.len() >= 3
                    && word.chars().nth(1) == Some(':')
                    && word.chars().nth(2) == Some('\\'))
                || word.ends_with(".rs")
                || word.ends_with(".py")
                || word.ends_with(".js")
                || word.ends_with(".ts")
                || word.ends_with(".json")
                || word.ends_with(".toml")
                || word.ends_with(".yaml")
                || word.ends_with(".yml")
                || word.ends_with(".md")
                || word.ends_with(".txt")
                || word.ends_with(".html")
                || word.ends_with(".css");

            if is_path {
                if let Some(start) = text.find(word) {
                    entities.push(Entity::new(
                        word,
                        EntityType::FilePath,
                        0.85,
                        start,
                        start + word.len(),
                    ));
                }
            }
        }

        entities
    }

    /// Extract version numbers
    fn extract_versions(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Pattern: v1.2.3, 1.2.3, version 1.2
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check for 'v' prefix or start of number
            let has_v = chars[i] == 'v' || chars[i] == 'V';
            let start = i;

            if has_v {
                i += 1;
            }

            if i < chars.len() && chars[i].is_ascii_digit() {
                let _num_start = i;
                let mut has_dot = false;
                let mut digit_count = 0;

                while i < chars.len() {
                    if chars[i].is_ascii_digit() {
                        digit_count += 1;
                        i += 1;
                    } else if chars[i] == '.'
                        && i + 1 < chars.len()
                        && chars[i + 1].is_ascii_digit()
                    {
                        has_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Valid version: has dots and multiple digits
                if has_dot && digit_count >= 2 {
                    let version: String = chars[start..i].iter().collect();
                    let confidence = if has_v { 0.9 } else { 0.7 };
                    entities.push(Entity::new(
                        &version,
                        EntityType::Version,
                        confidence,
                        start,
                        i,
                    ));
                }
            } else if !has_v {
                i += 1;
            }
        }

        entities
    }

    /// Extract money amounts
    fn extract_money(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        let currency_symbols = ['$', '€', '£', '¥'];
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if currency_symbols.contains(&chars[i]) {
                let start = i;
                i += 1;

                // Skip space
                while i < chars.len() && chars[i] == ' ' {
                    i += 1;
                }

                // Collect number
                let mut has_digits = false;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || chars[i] == ',' || chars[i] == '.')
                {
                    if chars[i].is_ascii_digit() {
                        has_digits = true;
                    }
                    i += 1;
                }

                if has_digits {
                    let money: String = chars[start..i].iter().collect();
                    entities.push(Entity::new(&money, EntityType::Money, 0.9, start, i));
                }
            } else {
                i += 1;
            }
        }

        entities
    }

    /// Extract percentages
    fn extract_percentages(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                let start = i;

                // Collect number
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }

                // Check for % sign
                if i < chars.len() && chars[i] == '%' {
                    i += 1;
                    let pct: String = chars[start..i].iter().collect();
                    entities.push(Entity::new(&pct, EntityType::Percentage, 0.95, start, i));
                }
            } else {
                i += 1;
            }
        }

        entities
    }

    /// Extract person names (heuristic)
    fn extract_persons(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();

            // Check for title prefix
            if self.person_titles.contains(&word_lower.as_str()) {
                // Next word(s) might be a name
                if i + 1 < words.len() {
                    let next = words[i + 1];
                    if next
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        let name = format!("{} {}", word, next);
                        if let Some(start) = text.find(&name) {
                            entities.push(Entity::new(
                                &name,
                                EntityType::Person,
                                0.8,
                                start,
                                start + name.len(),
                            ));
                        }
                    }
                }
            }
        }

        entities
    }

    /// Extract organizations (heuristic)
    fn extract_organizations(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();

        for suffix in &self.org_suffixes {
            let pattern = format!(" {}", suffix);
            let mut search_start = 0;

            while let Some(pos) = text_lower[search_start..].find(&pattern) {
                let abs_pos = search_start + pos;

                // Find start of organization name (previous capitalized words)
                let before = &text[..abs_pos];
                let words: Vec<&str> = before.split_whitespace().collect();

                if let Some(last_word) = words.last() {
                    if last_word
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        let org_name = format!(
                            "{} {}",
                            last_word,
                            &text[abs_pos + 1..abs_pos + 1 + suffix.len()]
                        );
                        if let Some(start) = text.find(&org_name) {
                            entities.push(Entity::new(
                                &org_name,
                                EntityType::Organization,
                                0.75,
                                start,
                                start + org_name.len(),
                            ));
                        }
                    }
                }

                search_start = abs_pos + suffix.len();
            }
        }

        entities
    }

    /// Apply a custom pattern
    fn apply_custom_pattern(&self, text: &str, pattern: &CustomPattern) -> Vec<Entity> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();

        for keyword in &pattern.keywords {
            let keyword_lower = keyword.to_lowercase();
            let mut search_start = 0;

            while let Some(pos) = text_lower[search_start..].find(&keyword_lower) {
                let abs_pos = search_start + pos;
                let original = &text[abs_pos..abs_pos + keyword.len()];

                entities.push(
                    Entity::new(
                        original,
                        pattern.entity_type,
                        pattern.confidence,
                        abs_pos,
                        abs_pos + keyword.len(),
                    )
                    .with_metadata("pattern", &pattern.name),
                );

                search_start = abs_pos + keyword.len();
            }
        }

        entities
    }

    /// Remove duplicate entities (keep highest confidence)
    fn deduplicate_entities(&self, mut entities: Vec<Entity>) -> Vec<Entity> {
        // Sort by start position and confidence
        entities.sort_by(|a, b| {
            a.start_offset.cmp(&b.start_offset).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        let mut result = Vec::new();
        let mut last_end = 0;

        for entity in entities {
            // Skip overlapping entities
            if entity.start_offset >= last_end {
                last_end = entity.end_offset;
                result.push(entity);
            }
        }

        result
    }
}

// ============================================================================
// Fact Tracking
// ============================================================================

/// A fact extracted from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// Unique identifier
    pub id: String,
    /// The fact statement
    pub statement: String,
    /// Subject entity (if identified)
    pub subject: Option<String>,
    /// Predicate/relation
    pub predicate: String,
    /// Object/value
    pub object: String,
    /// Source (message ID or context)
    pub source: String,
    /// When this fact was extracted
    pub extracted_at: DateTime<Utc>,
    /// Confidence score
    pub confidence: f32,
    /// Times this fact was reinforced
    pub reinforcement_count: u32,
    /// Related entities
    pub related_entities: Vec<String>,
}

impl Fact {
    pub fn new(
        statement: &str,
        predicate: &str,
        object: &str,
        source: &str,
        confidence: f32,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            statement: statement.to_string(),
            subject: None,
            predicate: predicate.to_string(),
            object: object.to_string(),
            source: source.to_string(),
            extracted_at: Utc::now(),
            confidence,
            reinforcement_count: 1,
            related_entities: Vec::new(),
        }
    }

    pub fn with_subject(mut self, subject: &str) -> Self {
        self.subject = Some(subject.to_string());
        self
    }

    pub fn add_related_entity(&mut self, entity: &str) {
        if !self.related_entities.contains(&entity.to_string()) {
            self.related_entities.push(entity.to_string());
        }
    }

    pub fn reinforce(&mut self) {
        self.reinforcement_count += 1;
        // Increase confidence slightly with reinforcement
        self.confidence = (self.confidence + 0.05).min(1.0);
    }
}

/// Types of facts that can be tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactType {
    /// User preference (likes/dislikes)
    Preference,
    /// User characteristic or attribute
    Attribute,
    /// User's stated goal or intention
    Goal,
    /// Technical fact about a project
    Technical,
    /// Relationship between entities
    Relationship,
    /// Event or action
    Event,
    /// Definition or explanation
    Definition,
}

/// Configuration for fact extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactExtractorConfig {
    /// Minimum confidence to keep a fact
    pub min_confidence: f32,
    /// Enable preference extraction
    pub extract_preferences: bool,
    /// Enable technical fact extraction
    pub extract_technical: bool,
    /// Enable goal extraction
    pub extract_goals: bool,
    /// Maximum facts to store per session
    pub max_facts_per_session: usize,
}

impl Default for FactExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            extract_preferences: true,
            extract_technical: true,
            extract_goals: true,
            max_facts_per_session: 100,
        }
    }
}

/// Fact extractor using pattern matching
pub struct FactExtractor {
    config: FactExtractorConfig,
    /// Preference indicators
    preference_patterns: Vec<(&'static str, f32)>,
    /// Goal indicators
    goal_patterns: Vec<(&'static str, f32)>,
    /// Technical fact indicators
    technical_patterns: Vec<(&'static str, f32)>,
}

impl FactExtractor {
    pub fn new(config: FactExtractorConfig) -> Self {
        Self {
            config,
            preference_patterns: vec![
                ("i like", 0.85),
                ("i love", 0.9),
                ("i prefer", 0.85),
                ("i hate", 0.85),
                ("i don't like", 0.85),
                ("i dislike", 0.85),
                ("i enjoy", 0.8),
                ("i always", 0.75),
                ("i never", 0.75),
                ("i usually", 0.7),
                ("my favorite", 0.9),
            ],
            goal_patterns: vec![
                ("i want to", 0.85),
                ("i need to", 0.85),
                ("i'm trying to", 0.8),
                ("i plan to", 0.85),
                ("i'd like to", 0.8),
                ("my goal is", 0.9),
                ("i'm working on", 0.75),
                ("i'm building", 0.75),
                ("i'm creating", 0.75),
            ],
            technical_patterns: vec![
                ("uses", 0.7),
                ("uses a", 0.7),
                ("is written in", 0.85),
                ("is built with", 0.85),
                ("requires", 0.75),
                ("depends on", 0.8),
                ("is configured", 0.75),
                ("runs on", 0.75),
                ("supports", 0.7),
            ],
        }
    }

    /// Extract facts from a message
    pub fn extract_facts(&self, text: &str, source: &str) -> Vec<Fact> {
        let mut facts = Vec::new();
        let _text_lower = text.to_lowercase();
        let sentences = self.split_sentences(text);

        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();

            // Extract preferences
            if self.config.extract_preferences {
                for (pattern, confidence) in &self.preference_patterns {
                    if sentence_lower.contains(pattern) {
                        if let Some(fact) =
                            self.extract_preference_fact(sentence, pattern, *confidence, source)
                        {
                            facts.push(fact);
                        }
                    }
                }
            }

            // Extract goals
            if self.config.extract_goals {
                for (pattern, confidence) in &self.goal_patterns {
                    if sentence_lower.contains(pattern) {
                        if let Some(fact) =
                            self.extract_goal_fact(sentence, pattern, *confidence, source)
                        {
                            facts.push(fact);
                        }
                    }
                }
            }

            // Extract technical facts
            if self.config.extract_technical {
                for (pattern, confidence) in &self.technical_patterns {
                    if sentence_lower.contains(pattern) {
                        if let Some(fact) =
                            self.extract_technical_fact(sentence, pattern, *confidence, source)
                        {
                            facts.push(fact);
                        }
                    }
                }
            }
        }

        // Filter by confidence
        facts.retain(|f| f.confidence >= self.config.min_confidence);
        facts
    }

    fn split_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        // Simple sentence splitting
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn extract_preference_fact(
        &self,
        sentence: &str,
        pattern: &str,
        confidence: f32,
        source: &str,
    ) -> Option<Fact> {
        let sentence_lower = sentence.to_lowercase();
        if let Some(pos) = sentence_lower.find(pattern) {
            let after = &sentence[pos + pattern.len()..].trim();
            if !after.is_empty() {
                let object = self.extract_object(after);
                if !object.is_empty() {
                    let is_negative = pattern.contains("don't")
                        || pattern.contains("hate")
                        || pattern.contains("dislike")
                        || pattern.contains("never");
                    let predicate = if is_negative { "dislikes" } else { "likes" };

                    return Some(
                        Fact::new(sentence, predicate, &object, source, confidence)
                            .with_subject("user"),
                    );
                }
            }
        }
        None
    }

    fn extract_goal_fact(
        &self,
        sentence: &str,
        pattern: &str,
        confidence: f32,
        source: &str,
    ) -> Option<Fact> {
        let sentence_lower = sentence.to_lowercase();
        if let Some(pos) = sentence_lower.find(pattern) {
            let after = &sentence[pos + pattern.len()..].trim();
            if !after.is_empty() {
                let object = self.extract_object(after);
                if !object.is_empty() {
                    return Some(
                        Fact::new(sentence, "wants to", &object, source, confidence)
                            .with_subject("user"),
                    );
                }
            }
        }
        None
    }

    fn extract_technical_fact(
        &self,
        sentence: &str,
        pattern: &str,
        confidence: f32,
        source: &str,
    ) -> Option<Fact> {
        let sentence_lower = sentence.to_lowercase();
        if let Some(pos) = sentence_lower.find(pattern) {
            let before = &sentence[..pos].trim();
            let after = &sentence[pos + pattern.len()..].trim();

            if !before.is_empty() && !after.is_empty() {
                let subject = self.extract_last_noun_phrase(before);
                let object = self.extract_object(after);

                if !subject.is_empty() && !object.is_empty() {
                    let predicate = pattern.trim();
                    return Some(
                        Fact::new(sentence, predicate, &object, source, confidence)
                            .with_subject(&subject),
                    );
                }
            }
        }
        None
    }

    fn extract_object(&self, text: &str) -> String {
        // Extract the first meaningful phrase (up to preposition or conjunction)
        let stop_words = [
            "and", "or", "but", "because", "since", "when", "if", "which", "that",
        ];

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();

        for word in words {
            let word_lower = word.to_lowercase();
            let clean_word = word_lower.trim_matches(|c: char| !c.is_alphanumeric());

            if stop_words.contains(&clean_word) {
                break;
            }
            result.push(word);

            if result.len() >= 10 {
                break;
            }
        }

        result
            .join(" ")
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string()
    }

    fn extract_last_noun_phrase(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let last_words: Vec<&str> = words.iter().rev().take(5).rev().cloned().collect();
        last_words.join(" ")
    }
}

// ============================================================================
// Fact Store
// ============================================================================

/// In-memory fact store with deduplication
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactStore {
    /// All stored facts
    facts: Vec<Fact>,
    /// Index by subject
    by_subject: HashMap<String, Vec<usize>>,
    /// Index by predicate
    by_predicate: HashMap<String, Vec<usize>>,
}

impl FactStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fact (with deduplication)
    pub fn add_fact(&mut self, fact: Fact) {
        // Check for similar existing facts
        let similar_idx = self.find_similar_fact(&fact);

        if let Some(idx) = similar_idx {
            // Reinforce existing fact
            self.facts[idx].reinforce();
        } else {
            // Add new fact
            let idx = self.facts.len();

            if let Some(ref subject) = fact.subject {
                self.by_subject
                    .entry(subject.to_lowercase())
                    .or_default()
                    .push(idx);
            }
            self.by_predicate
                .entry(fact.predicate.to_lowercase())
                .or_default()
                .push(idx);

            self.facts.push(fact);

            #[cfg(feature = "analytics")]
            crate::scalability_monitor::check_scalability(
                crate::scalability_monitor::Subsystem::FactStore,
                self.facts.len(),
            );
        }
    }

    /// Find a similar existing fact
    fn find_similar_fact(&self, fact: &Fact) -> Option<usize> {
        for (idx, existing) in self.facts.iter().enumerate() {
            // Same predicate and similar object
            if existing.predicate.to_lowercase() == fact.predicate.to_lowercase() {
                let similarity = self.string_similarity(&existing.object, &fact.object);
                if similarity > 0.8 {
                    return Some(idx);
                }
            }
        }
        None
    }

    fn string_similarity(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        if a_lower == b_lower {
            return 1.0;
        }

        let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Get all facts
    pub fn all_facts(&self) -> &[Fact] {
        &self.facts
    }

    /// Get facts by subject
    pub fn facts_by_subject(&self, subject: &str) -> Vec<&Fact> {
        self.by_subject
            .get(&subject.to_lowercase())
            .map(|indices| indices.iter().map(|&i| &self.facts[i]).collect())
            .unwrap_or_default()
    }

    /// Get facts by predicate
    pub fn facts_by_predicate(&self, predicate: &str) -> Vec<&Fact> {
        self.by_predicate
            .get(&predicate.to_lowercase())
            .map(|indices| indices.iter().map(|&i| &self.facts[i]).collect())
            .unwrap_or_default()
    }

    /// Get user preferences
    pub fn user_preferences(&self) -> Vec<&Fact> {
        let likes = self.facts_by_predicate("likes");
        let dislikes = self.facts_by_predicate("dislikes");
        likes.into_iter().chain(dislikes).collect()
    }

    /// Get user goals
    pub fn user_goals(&self) -> Vec<&Fact> {
        self.facts_by_predicate("wants to")
    }

    /// Get top facts by confidence
    pub fn top_facts(&self, n: usize) -> Vec<&Fact> {
        let mut sorted: Vec<&Fact> = self.facts.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get fact count
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Clear all facts
    pub fn clear(&mut self) {
        self.facts.clear();
        self.by_subject.clear();
        self.by_predicate.clear();
    }

    /// Export facts to JSON
    pub fn export(&self) -> String {
        serde_json::to_string_pretty(&self.facts).unwrap_or_default()
    }

    /// Export facts to internal binary format (bincode+gzip when feature enabled).
    #[cfg(feature = "binary-storage")]
    pub fn export_bytes(&self) -> Result<Vec<u8>, anyhow::Error> {
        crate::internal_storage::serialize_internal(&self.facts)
    }

    /// Build a summary of known facts for context
    pub fn build_context_summary(&self, max_facts: usize) -> String {
        let mut summary = String::new();

        let preferences = self.user_preferences();
        if !preferences.is_empty() {
            summary.push_str("User preferences:\n");
            for pref in preferences.iter().take(max_facts / 3) {
                summary.push_str(&format!("- {}: {}\n", pref.predicate, pref.object));
            }
        }

        let goals = self.user_goals();
        if !goals.is_empty() {
            summary.push_str("\nUser goals:\n");
            for goal in goals.iter().take(max_facts / 3) {
                summary.push_str(&format!("- {}\n", goal.object));
            }
        }

        // Technical facts
        let technical: Vec<&Fact> = self
            .facts
            .iter()
            .filter(|f| {
                f.predicate != "likes" && f.predicate != "dislikes" && f.predicate != "wants to"
            })
            .collect();

        if !technical.is_empty() {
            summary.push_str("\nKnown facts:\n");
            for fact in technical.iter().take(max_facts / 3) {
                if let Some(ref subject) = fact.subject {
                    summary.push_str(&format!(
                        "- {} {} {}\n",
                        subject, fact.predicate, fact.object
                    ));
                }
            }
        }

        summary
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction_email() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Contact me at test@example.com for more info.");

        assert!(!entities.is_empty());
        let email = entities.iter().find(|e| e.entity_type == EntityType::Email);
        assert!(email.is_some());
        assert_eq!(email.unwrap().text, "test@example.com");
    }

    #[test]
    fn test_entity_extraction_url() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Visit https://github.com/rust-lang for more.");

        let url = entities.iter().find(|e| e.entity_type == EntityType::Url);
        assert!(url.is_some());
        assert!(url.unwrap().text.contains("github.com"));
    }

    #[test]
    fn test_entity_extraction_version() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Using v1.2.3 of the library.");

        let version = entities
            .iter()
            .find(|e| e.entity_type == EntityType::Version);
        assert!(version.is_some());
        assert_eq!(version.unwrap().text, "v1.2.3");
    }

    #[test]
    fn test_fact_extraction_preference() {
        let extractor = FactExtractor::new(FactExtractorConfig::default());
        let facts = extractor.extract_facts("I like using Rust for systems programming.", "test");

        assert!(!facts.is_empty());
        let pref = facts.iter().find(|f| f.predicate == "likes");
        assert!(pref.is_some());
    }

    #[test]
    fn test_fact_extraction_goal() {
        let extractor = FactExtractor::new(FactExtractorConfig::default());
        let facts = extractor.extract_facts("I want to build a web server.", "test");

        let goal = facts.iter().find(|f| f.predicate == "wants to");
        assert!(goal.is_some());
        assert!(goal.unwrap().object.contains("build"));
    }

    #[test]
    fn test_fact_store() {
        let mut store = FactStore::new();

        let fact1 = Fact::new("I like Rust", "likes", "Rust", "msg1", 0.9).with_subject("user");
        let fact2 =
            Fact::new("I like Python", "likes", "Python", "msg2", 0.85).with_subject("user");

        store.add_fact(fact1);
        store.add_fact(fact2);

        assert_eq!(store.len(), 2);

        let prefs = store.user_preferences();
        assert_eq!(prefs.len(), 2);
    }

    #[test]
    fn test_fact_reinforcement() {
        let mut store = FactStore::new();

        let fact1 = Fact::new("I like Rust", "likes", "Rust", "msg1", 0.8).with_subject("user");
        store.add_fact(fact1);

        // Add similar fact
        let fact2 =
            Fact::new("I really like Rust", "likes", "Rust", "msg2", 0.85).with_subject("user");
        store.add_fact(fact2);

        // Should reinforce, not add new
        assert_eq!(store.len(), 1);
        assert!(store.facts[0].confidence > 0.8);
        assert_eq!(store.facts[0].reinforcement_count, 2);
    }

    // ========================================================================
    // Phase 3 (v11): EntityType display_name coverage
    // ========================================================================

    #[test]
    fn test_entity_type_display_name_all_variants() {
        assert_eq!(EntityType::Person.display_name(), "Person");
        assert_eq!(EntityType::Organization.display_name(), "Organization");
        assert_eq!(EntityType::Location.display_name(), "Location");
        assert_eq!(EntityType::DateTime.display_name(), "Date/Time");
        assert_eq!(EntityType::Money.display_name(), "Money");
        assert_eq!(EntityType::Percentage.display_name(), "Percentage");
        assert_eq!(EntityType::Email.display_name(), "Email");
        assert_eq!(EntityType::Url.display_name(), "URL");
        assert_eq!(EntityType::Phone.display_name(), "Phone");
        assert_eq!(EntityType::ProgrammingLanguage.display_name(), "Language");
        assert_eq!(EntityType::FilePath.display_name(), "File");
        assert_eq!(EntityType::TechnicalTerm.display_name(), "Technical");
        assert_eq!(EntityType::Version.display_name(), "Version");
    }

    #[test]
    fn test_entity_type_custom_variant() {
        let c1 = EntityType::Custom(0);
        let c2 = EntityType::Custom(42);
        assert_eq!(c1.display_name(), "Custom");
        assert_eq!(c2.display_name(), "Custom");
        assert_ne!(c1, c2);
        assert_eq!(c1, EntityType::Custom(0));
    }

    // ========================================================================
    // Phase 3 (v11): Entity metadata and position
    // ========================================================================

    #[test]
    fn test_entity_with_metadata() {
        let entity = Entity::new("test", EntityType::Person, 0.9, 0, 4)
            .with_metadata("source", "manual")
            .with_metadata("lang", "en");
        assert_eq!(entity.metadata.len(), 2);
        assert_eq!(entity.metadata.get("source").unwrap(), "manual");
        assert_eq!(entity.metadata.get("lang").unwrap(), "en");
    }

    #[test]
    fn test_entity_position_tracking() {
        let entity = Entity::new("Rust", EntityType::ProgrammingLanguage, 0.85, 10, 14);
        assert_eq!(entity.start_offset, 10);
        assert_eq!(entity.end_offset, 14);
        assert_eq!(entity.text, "Rust");
        assert_eq!(entity.normalized, "rust");
    }

    #[test]
    fn test_entity_normalization() {
        let entity = Entity::new("  HELLO World  ", EntityType::TechnicalTerm, 0.7, 0, 15);
        assert_eq!(entity.normalized, "hello world");
        assert_eq!(entity.text, "  HELLO World  ");
    }

    // ========================================================================
    // Phase 3 (v11): Entity extraction by type
    // ========================================================================

    #[test]
    fn test_extract_programming_language() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("I am learning Rust and Python for my projects.");

        let langs: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::ProgrammingLanguage)
            .collect();
        assert!(
            langs.len() >= 2,
            "Expected at least 2 languages, got {}",
            langs.len()
        );
    }

    #[test]
    fn test_extract_file_path() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Edit the file main.rs and config.toml please.");

        let paths: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::FilePath)
            .collect();
        assert!(!paths.is_empty(), "Expected file path entities");
    }

    #[test]
    fn test_extract_phone_number() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Call me at +1 555-123-4567 tomorrow.");

        let phones: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Phone)
            .collect();
        assert!(!phones.is_empty(), "Expected phone entity");
    }

    #[test]
    fn test_extract_money() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("The price is $49.99 and the tax is $3.50.");

        let money: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();
        assert!(
            money.len() >= 2,
            "Expected at least 2 money entities, got {}",
            money.len()
        );
    }

    #[test]
    fn test_extract_percentage() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("Coverage went from 75% to 99.5% this week.");

        let pcts: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Percentage)
            .collect();
        assert!(
            pcts.len() >= 2,
            "Expected at least 2 percentages, got {}",
            pcts.len()
        );
    }

    // ========================================================================
    // Phase 3 (v11): Edge cases
    // ========================================================================

    #[test]
    fn test_extract_empty_text() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_whitespace_only() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let entities = extractor.extract("     \t\n   ");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_very_long_text() {
        let extractor = EntityExtractor::new(EntityExtractorConfig::default());
        let mut text = "word ".repeat(500);
        text.push_str("contact user@longtest.com here ");
        text.push_str(&"more ".repeat(500));
        let entities = extractor.extract(&text);
        let emails: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Email)
            .collect();
        assert_eq!(emails.len(), 1);
        assert_eq!(emails[0].text, "user@longtest.com");
    }

    // ========================================================================
    // Phase 3 (v11): FactStore operations
    // ========================================================================

    #[test]
    fn test_fact_store_facts_by_subject() {
        let mut store = FactStore::new();
        store.add_fact(
            Fact::new("User likes Rust", "likes", "Rust", "s1", 0.9).with_subject("user"),
        );
        store.add_fact(
            Fact::new("Bot knows Rust", "knows", "Rust", "s2", 0.8).with_subject("bot"),
        );

        let user_facts = store.facts_by_subject("user");
        assert_eq!(user_facts.len(), 1);
        assert_eq!(user_facts[0].object, "Rust");

        let bot_facts = store.facts_by_subject("bot");
        assert_eq!(bot_facts.len(), 1);

        let empty = store.facts_by_subject("nobody");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_fact_store_facts_by_predicate() {
        let mut store = FactStore::new();
        store.add_fact(
            Fact::new("Likes Rust", "likes", "Rust", "s1", 0.9).with_subject("user"),
        );
        store.add_fact(
            Fact::new("Likes Python", "likes", "Python", "s2", 0.85).with_subject("user"),
        );
        store.add_fact(
            Fact::new("Wants web", "wants to", "build a web app", "s3", 0.8)
                .with_subject("user"),
        );

        let likes = store.facts_by_predicate("likes");
        assert_eq!(likes.len(), 2);

        let goals = store.facts_by_predicate("wants to");
        assert_eq!(goals.len(), 1);

        let empty = store.facts_by_predicate("hates");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_fact_store_user_goals() {
        let mut store = FactStore::new();
        store.add_fact(
            Fact::new("Want server", "wants to", "build a server", "s1", 0.9)
                .with_subject("user"),
        );
        store.add_fact(
            Fact::new("Want deploy", "wants to", "deploy to production", "s2", 0.85)
                .with_subject("user"),
        );

        let goals = store.user_goals();
        assert_eq!(goals.len(), 2);
    }

    #[test]
    fn test_fact_store_top_facts() {
        let mut store = FactStore::new();
        store.add_fact(Fact::new("Low", "pred", "obj1", "s1", 0.5));
        store.add_fact(Fact::new("High", "pred2", "obj2", "s2", 0.99));
        store.add_fact(Fact::new("Mid", "pred3", "obj3", "s3", 0.75));

        let top2 = store.top_facts(2);
        assert_eq!(top2.len(), 2);
        assert!(top2[0].confidence >= top2[1].confidence);
    }

    #[test]
    fn test_fact_store_clear() {
        let mut store = FactStore::new();
        store.add_fact(Fact::new("A", "likes", "X", "s1", 0.9).with_subject("user"));
        store.add_fact(Fact::new("B", "likes", "Y", "s2", 0.8).with_subject("user"));
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_fact_store_export_json() {
        let mut store = FactStore::new();
        store.add_fact(
            Fact::new("Likes Rust", "likes", "Rust", "s1", 0.9).with_subject("user"),
        );

        let json = store.export();
        assert!(!json.is_empty());
        assert!(json.contains("Rust"));
        assert!(json.contains("likes"));
    }

    // ========================================================================
    // Phase 3 (v11): FactExtractor patterns
    // ========================================================================

    #[test]
    fn test_fact_extractor_working_on_pattern() {
        let extractor = FactExtractor::new(FactExtractorConfig::default());
        let facts = extractor.extract_facts("I'm working on a new compiler.", "test");
        let goal = facts.iter().find(|f| f.predicate == "wants to");
        assert!(goal.is_some(), "Expected goal from 'I'm working on'");
    }

    #[test]
    fn test_fact_extractor_building_pattern() {
        let extractor = FactExtractor::new(FactExtractorConfig::default());
        let facts = extractor.extract_facts("I'm building a distributed system.", "test");
        let goal = facts.iter().find(|f| f.predicate == "wants to");
        assert!(goal.is_some(), "Expected goal from 'I'm building'");
    }

    // ========================================================================
    // Phase 3 (v11): Context summary
    // ========================================================================

    #[test]
    fn test_build_context_summary_with_facts() {
        let mut store = FactStore::new();
        store.add_fact(
            Fact::new("Likes Rust", "likes", "Rust", "s1", 0.9).with_subject("user"),
        );
        store.add_fact(
            Fact::new("Goal", "wants to", "build APIs", "s2", 0.85).with_subject("user"),
        );

        let summary = store.build_context_summary(9);
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_build_context_summary_empty_store() {
        let store = FactStore::new();
        let summary = store.build_context_summary(10);
        assert!(summary.is_empty());
    }

    // ========================================================================
    // Phase 3 (v11): Fact struct methods
    // ========================================================================

    #[test]
    fn test_fact_add_related_entity() {
        let mut fact = Fact::new("test", "likes", "Rust", "s1", 0.9);
        fact.add_related_entity("programming");
        fact.add_related_entity("systems");
        fact.add_related_entity("programming"); // duplicate
        assert_eq!(fact.related_entities.len(), 2);
    }

    #[test]
    fn test_fact_reinforce_caps_at_one() {
        let mut fact = Fact::new("test", "likes", "Rust", "s1", 0.98);
        fact.reinforce();
        fact.reinforce();
        fact.reinforce();
        assert!(fact.confidence <= 1.0);
        assert_eq!(fact.reinforcement_count, 4);
    }
}
