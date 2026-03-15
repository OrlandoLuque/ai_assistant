//! Entity enrichment module
//!
//! Provides automatic enrichment of extracted entities by cross-referencing
//! against external sources, detecting duplicates via fuzzy matching,
//! applying auto-tagging heuristics, and offering configurable merge strategies.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::entities::EntityType;

// ============================================================================
// Data Types
// ============================================================================

/// An entity that can be enriched with external data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichableEntity {
    /// The raw text of the entity as extracted.
    pub text: String,
    /// The classified type of the entity.
    pub entity_type: EntityType,
    /// Key-value attributes associated with this entity.
    pub attributes: HashMap<String, String>,
    /// The source from which this entity was originally extracted.
    pub source: String,
    /// Timestamp when this entity was first observed.
    pub first_seen: DateTime<Utc>,
    /// Confidence score for the extraction (0.0 - 1.0).
    pub confidence: f32,
    /// Tags assigned to this entity.
    pub tags: Vec<String>,
}

/// Data obtained from an enrichment source for a given entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentData {
    /// Name of the enrichment source that provided this data.
    pub source: String,
    /// Attributes retrieved from the source.
    pub attributes: HashMap<String, String>,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Names/texts of related entities discovered.
    pub related_entities: Vec<String>,
    /// Confidence of the enrichment data (0.0 - 1.0).
    pub confidence: f32,
    /// When this enrichment data was fetched.
    pub fetched_at: DateTime<Utc>,
    /// Relevant URLs from the source.
    pub urls: Vec<String>,
}

/// An entity after enrichment processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedEntity {
    /// The original enrichable entity.
    pub entity: EnrichableEntity,
    /// All enrichment data gathered from various sources.
    pub enrichments: Vec<EnrichmentData>,
    /// Merged attributes from all enrichment sources.
    pub merged_attributes: HashMap<String, String>,
    /// Overall confidence after enrichment.
    pub enrichment_confidence: f32,
    /// If this entity is a duplicate, the text of the original.
    pub duplicate_of: Option<String>,
    /// Automatically generated tags.
    pub auto_tags: Vec<String>,
}

// ============================================================================
// Merge Strategy
// ============================================================================

/// Strategy for merging attributes when combining entities or enrichments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum MergeStrategy {
    /// Keep attributes from the first/original entity only.
    KeepFirst,
    /// Keep attributes from the latest/newest entity only.
    KeepLatest,
    /// Keep attributes from the entity with the highest confidence.
    KeepHighestConfidence,
    /// Merge all attributes (later values overwrite earlier ones).
    MergeAll,
    /// Do not auto-merge; requires manual resolution.
    Manual,
}

// ============================================================================
// Duplicate Detection
// ============================================================================

/// A detected duplicate match between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateMatch {
    /// The text of the entity being checked.
    pub entity_text: String,
    /// The text of the entity it matches against.
    pub matches_text: String,
    /// Similarity score (0.0 - 1.0).
    pub similarity: f32,
    /// The reason the duplicate was detected.
    pub reason: DuplicateReason,
}

/// Reason why two entities were considered duplicates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum DuplicateReason {
    /// Exact string match after normalization.
    ExactMatch,
    /// Fuzzy match above the configured threshold.
    FuzzyMatch,
    /// Match after text normalization (lowercase, stripped punctuation).
    NormalizedMatch,
}

// ============================================================================
// Enrichment Source Configuration
// ============================================================================

/// Configuration for a single external enrichment source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentSource {
    /// Human-readable name of the source.
    pub name: String,
    /// Base URL for API calls.
    pub base_url: String,
    /// Path appended to the base URL for search queries.
    pub search_path: String,
    /// Entity types this source supports.
    pub supported_types: Vec<EntityType>,
    /// Weight factor for this source's confidence scores.
    pub weight: f32,
    /// Whether this source is currently active.
    pub enabled: bool,
}

// ============================================================================
// Enrichment Config
// ============================================================================

/// Configuration for the entity enrichment system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EnrichmentConfig {
    /// List of enrichment sources to query.
    pub sources: Vec<EnrichmentSource>,
    /// Strategy for merging attributes from multiple sources.
    pub merge_strategy: MergeStrategy,
    /// Minimum confidence for enrichment data to be accepted.
    pub min_enrichment_confidence: f32,
    /// Similarity threshold for duplicate detection (0.0 - 1.0).
    pub duplicate_threshold: f32,
    /// Maximum number of enrichments to collect per entity.
    pub max_enrichments: usize,
    /// Whether to automatically generate tags for enriched entities.
    pub auto_tagging: bool,
    /// Whether duplicate detection is enabled.
    pub dedup_enabled: bool,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            merge_strategy: MergeStrategy::MergeAll,
            min_enrichment_confidence: 0.5,
            duplicate_threshold: 0.85,
            max_enrichments: 5,
            auto_tagging: true,
            dedup_enabled: true,
        }
    }
}

// ============================================================================
// Entity Enricher
// ============================================================================

/// Main enricher that orchestrates entity enrichment, deduplication, and tagging.
pub struct EntityEnricher {
    /// Configuration for this enricher instance.
    config: EnrichmentConfig,
    /// Registry of known entities for duplicate checking.
    known_entities: Vec<EnrichableEntity>,
}

impl EntityEnricher {
    /// Create a new `EntityEnricher` with the given configuration.
    pub fn new(config: EnrichmentConfig) -> Self {
        Self {
            config,
            known_entities: Vec::new(),
        }
    }

    /// Enrich a single entity by querying configured sources, merging results,
    /// detecting duplicates, and generating auto-tags.
    pub fn enrich(&mut self, entity: &EnrichableEntity) -> Result<EnrichedEntity> {
        let mut enrichments: Vec<EnrichmentData> = Vec::new();

        // Query each enabled source that supports this entity type
        for source in &self.config.sources {
            if !source.enabled {
                continue;
            }
            if !source.supported_types.contains(&entity.entity_type) {
                continue;
            }
            if enrichments.len() >= self.config.max_enrichments {
                break;
            }
            match self.query_source(entity, source) {
                Ok(Some(data)) => {
                    if data.confidence >= self.config.min_enrichment_confidence {
                        enrichments.push(data);
                    }
                }
                Ok(None) => {}
                Err(_) => {
                    // Source query failed; skip this source silently
                }
            }
        }

        // Merge attributes from all enrichments
        let mut merged_attributes = entity.attributes.clone();
        for enrichment in &enrichments {
            merged_attributes = merge_attributes(&merged_attributes, &enrichment.attributes);
        }

        // Compute overall enrichment confidence
        let enrichment_confidence = if enrichments.is_empty() {
            entity.confidence
        } else {
            let total_conf: f32 = enrichments.iter().map(|e| e.confidence).sum();
            let avg_conf = total_conf / enrichments.len() as f32;
            // Blend original and enrichment confidence
            (entity.confidence + avg_conf) / 2.0
        };

        // Check for duplicates against known entities
        let duplicate_of = if self.config.dedup_enabled {
            self.check_duplicate(entity).map(|m| m.matches_text)
        } else {
            None
        };

        // Build intermediate enriched entity
        let mut enriched = EnrichedEntity {
            entity: entity.clone(),
            enrichments,
            merged_attributes,
            enrichment_confidence,
            duplicate_of,
            auto_tags: Vec::new(),
        };

        // Generate auto-tags if enabled
        if self.config.auto_tagging {
            enriched.auto_tags = self.auto_tag(&enriched);
        }

        Ok(enriched)
    }

    /// Enrich a batch of entities, returning results for each.
    pub fn enrich_batch(&mut self, entities: &[EnrichableEntity]) -> Vec<Result<EnrichedEntity>> {
        entities.iter().map(|e| self.enrich(e)).collect()
    }

    /// Find all duplicate pairs among a set of entities using pairwise comparison.
    pub fn find_duplicates(&self, entities: &[EnrichableEntity]) -> Vec<DuplicateMatch> {
        let mut matches = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let a = &entities[i];
                let b = &entities[j];

                // Check exact match on normalized forms
                let norm_a = normalize(&a.text);
                let norm_b = normalize(&b.text);

                if norm_a == norm_b {
                    matches.push(DuplicateMatch {
                        entity_text: a.text.clone(),
                        matches_text: b.text.clone(),
                        similarity: 1.0,
                        reason: if a.text == b.text {
                            DuplicateReason::ExactMatch
                        } else {
                            DuplicateReason::NormalizedMatch
                        },
                    });
                    continue;
                }

                // Fuzzy comparison
                let sim = fuzzy_similarity(&norm_a, &norm_b);
                if sim >= self.config.duplicate_threshold {
                    matches.push(DuplicateMatch {
                        entity_text: a.text.clone(),
                        matches_text: b.text.clone(),
                        similarity: sim,
                        reason: DuplicateReason::FuzzyMatch,
                    });
                }
            }
        }

        matches
    }

    /// Merge two entities into one using the configured merge strategy.
    pub fn merge_entities(&self, a: &EnrichableEntity, b: &EnrichableEntity) -> EnrichableEntity {
        match self.config.merge_strategy {
            MergeStrategy::KeepFirst => {
                let mut merged = a.clone();
                merged.tags = {
                    let mut tags: Vec<String> = a.tags.clone();
                    for t in &b.tags {
                        if !tags.contains(t) {
                            tags.push(t.clone());
                        }
                    }
                    tags
                };
                merged
            }
            MergeStrategy::KeepLatest => {
                let mut merged = b.clone();
                merged.tags = {
                    let mut tags: Vec<String> = a.tags.clone();
                    for t in &b.tags {
                        if !tags.contains(t) {
                            tags.push(t.clone());
                        }
                    }
                    tags
                };
                merged
            }
            MergeStrategy::KeepHighestConfidence => {
                if a.confidence >= b.confidence {
                    let mut merged = a.clone();
                    merged.attributes = merge_attributes(&a.attributes, &b.attributes);
                    merged
                } else {
                    let mut merged = b.clone();
                    merged.attributes = merge_attributes(&a.attributes, &b.attributes);
                    merged
                }
            }
            MergeStrategy::MergeAll | MergeStrategy::Manual => {
                let mut merged = a.clone();
                merged.attributes = merge_attributes(&a.attributes, &b.attributes);
                merged.confidence = (a.confidence + b.confidence) / 2.0;
                merged.tags = {
                    let mut tags: Vec<String> = a.tags.clone();
                    for t in &b.tags {
                        if !tags.contains(t) {
                            tags.push(t.clone());
                        }
                    }
                    tags
                };
                merged
            }
        }
    }

    /// Generate automatic tags for an enriched entity based on its type and attributes.
    pub fn auto_tag(&self, entity: &EnrichedEntity) -> Vec<String> {
        generate_tags(&entity.merged_attributes, &entity.entity.entity_type)
    }

    /// Register an entity in the known entities list for future duplicate detection.
    pub fn register_entity(&mut self, entity: EnrichableEntity) {
        self.known_entities.push(entity);
    }

    /// Check whether the given entity is a duplicate of any known entity.
    pub fn check_duplicate(&self, entity: &EnrichableEntity) -> Option<DuplicateMatch> {
        let norm_entity = normalize(&entity.text);

        for known in &self.known_entities {
            let norm_known = normalize(&known.text);

            // Exact normalized match
            if norm_entity == norm_known {
                return Some(DuplicateMatch {
                    entity_text: entity.text.clone(),
                    matches_text: known.text.clone(),
                    similarity: 1.0,
                    reason: if entity.text == known.text {
                        DuplicateReason::ExactMatch
                    } else {
                        DuplicateReason::NormalizedMatch
                    },
                });
            }

            // Fuzzy match
            let sim = fuzzy_similarity(&norm_entity, &norm_known);
            if sim >= self.config.duplicate_threshold {
                return Some(DuplicateMatch {
                    entity_text: entity.text.clone(),
                    matches_text: known.text.clone(),
                    similarity: sim,
                    reason: DuplicateReason::FuzzyMatch,
                });
            }
        }

        None
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    /// Query an external enrichment source for data about an entity.
    /// Performs a GET request to `base_url + search_path + "?q=" + urlencoded(entity.text)`,
    /// and attempts to parse the JSON response as enrichment attributes.
    fn query_source(
        &self,
        entity: &EnrichableEntity,
        source: &EnrichmentSource,
    ) -> Result<Option<EnrichmentData>> {
        let encoded_query = urlencoding::encode(&entity.text);
        let url = format!(
            "{}{}?q={}",
            source.base_url, source.search_path, encoded_query
        );

        let response = ureq::get(&url).call();

        match response {
            Ok(resp) => {
                let json: serde_json::Value = resp.into_json()?;

                // Try to extract attributes from the response object
                let mut attributes = HashMap::new();
                let mut description = None;
                let mut related_entities = Vec::new();
                let mut urls = Vec::new();

                if let Some(obj) = json.as_object() {
                    for (key, value) in obj {
                        match key.as_str() {
                            "description" => {
                                description = value.as_str().map(|s| s.to_string());
                            }
                            "related" | "related_entities" => {
                                if let Some(arr) = value.as_array() {
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            related_entities.push(s.to_string());
                                        }
                                    }
                                }
                            }
                            "urls" | "links" => {
                                if let Some(arr) = value.as_array() {
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            urls.push(s.to_string());
                                        }
                                    }
                                }
                            }
                            _ => {
                                if let Some(s) = value.as_str() {
                                    attributes.insert(key.clone(), s.to_string());
                                } else if let Some(n) = value.as_f64() {
                                    attributes.insert(key.clone(), n.to_string());
                                } else if let Some(b) = value.as_bool() {
                                    attributes.insert(key.clone(), b.to_string());
                                }
                            }
                        }
                    }
                }

                // If no useful data was extracted, return None
                if attributes.is_empty() && description.is_none() && related_entities.is_empty() {
                    return Ok(None);
                }

                Ok(Some(EnrichmentData {
                    source: source.name.clone(),
                    attributes,
                    description,
                    related_entities,
                    confidence: source.weight,
                    fetched_at: Utc::now(),
                    urls,
                }))
            }
            Err(_) => Ok(None),
        }
    }
}

// ============================================================================
// Private Helper Functions
// ============================================================================

/// Compute Jaccard similarity between two strings using character bigrams.
///
/// Splits each string into overlapping pairs of characters (bigrams), then
/// computes the Jaccard index: |intersection| / |union|.
fn fuzzy_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let bigrams_a = char_bigrams(a);
    let bigrams_b = char_bigrams(b);

    if bigrams_a.is_empty() && bigrams_b.is_empty() {
        // Both strings are single characters; compare directly
        return if a == b { 1.0 } else { 0.0 };
    }

    let set_a: HashSet<&(char, char)> = bigrams_a.iter().collect();
    let set_b: HashSet<&(char, char)> = bigrams_b.iter().collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Generate character bigrams (sliding window of size 2) from a string.
fn char_bigrams(s: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Normalize text: lowercase, remove non-alphanumeric characters (keeping spaces),
/// and collapse multiple whitespace into single spaces.
fn normalize(text: &str) -> String {
    let lowered = text.to_lowercase();
    let stripped: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();

    // Collapse multiple spaces into one
    let mut result = String::with_capacity(stripped.len());
    let mut prev_space = false;
    for c in stripped.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }

    result.trim().to_string()
}

/// Generate tags from entity attributes and type.
///
/// Creates a tag from the entity type's display name, plus one tag for each
/// attribute key that has a non-empty value.
fn generate_tags(attributes: &HashMap<String, String>, entity_type: &EntityType) -> Vec<String> {
    let mut tags = Vec::new();

    // Tag from entity type
    let type_tag = format!("type:{}", entity_type.display_name().to_lowercase());
    tags.push(type_tag);

    // Tags from attribute keys with non-empty values
    let mut keys: Vec<&String> = attributes.keys().collect();
    keys.sort(); // deterministic ordering
    for key in keys {
        if let Some(value) = attributes.get(key) {
            if !value.is_empty() {
                tags.push(format!("attr:{}", key.to_lowercase()));
            }
        }
    }

    tags
}

/// Merge two attribute maps, with values from `b` overwriting those in `a`
/// for duplicate keys.
fn merge_attributes(
    a: &HashMap<String, String>,
    b: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = a.clone();
    for (key, value) in b {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    /// Helper to create a test entity with minimal fields.
    fn make_entity(text: &str, entity_type: EntityType) -> EnrichableEntity {
        EnrichableEntity {
            text: text.to_string(),
            entity_type,
            attributes: HashMap::new(),
            source: "test".to_string(),
            first_seen: Utc::now(),
            confidence: 0.9,
            tags: Vec::new(),
        }
    }

    #[test]
    fn test_fuzzy_similarity_identical_strings() {
        let sim = fuzzy_similarity("hello world", "hello world");
        assert!(
            (sim - 1.0).abs() < f32::EPSILON,
            "Identical strings should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_fuzzy_similarity_completely_different() {
        let sim = fuzzy_similarity("abc", "xyz");
        assert!(
            sim < 0.1,
            "Completely different strings should have very low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_normalize_text() {
        let result = normalize("  Hello,   World!  How's it?  ");
        assert_eq!(result, "hello world how s it");
    }

    #[test]
    fn test_normalize_preserves_alphanumeric() {
        let result = normalize("Rust v1.75.0 release");
        assert_eq!(result, "rust v1 75 0 release");
    }

    #[test]
    fn test_find_duplicates_exact_match() {
        let config = EnrichmentConfig::default();
        let enricher = EntityEnricher::new(config);

        let entities = vec![
            make_entity("Rust Language", EntityType::ProgrammingLanguage),
            make_entity("Rust Language", EntityType::ProgrammingLanguage),
        ];

        let dupes = enricher.find_duplicates(&entities);
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].reason, DuplicateReason::ExactMatch);
        assert!((dupes[0].similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_find_duplicates_normalized_match() {
        let config = EnrichmentConfig::default();
        let enricher = EntityEnricher::new(config);

        let entities = vec![
            make_entity("Rust Language", EntityType::ProgrammingLanguage),
            make_entity("rust language", EntityType::ProgrammingLanguage),
        ];

        let dupes = enricher.find_duplicates(&entities);
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].reason, DuplicateReason::NormalizedMatch);
    }

    #[test]
    fn test_merge_entities_keep_first() {
        let config = EnrichmentConfig {
            merge_strategy: MergeStrategy::KeepFirst,
            ..Default::default()
        };
        let enricher = EntityEnricher::new(config);

        let mut a = make_entity("Entity A", EntityType::Person);
        a.attributes
            .insert("role".to_string(), "developer".to_string());

        let mut b = make_entity("Entity B", EntityType::Person);
        b.attributes
            .insert("role".to_string(), "manager".to_string());
        b.attributes
            .insert("dept".to_string(), "engineering".to_string());

        let merged = enricher.merge_entities(&a, &b);
        assert_eq!(merged.text, "Entity A");
        assert_eq!(merged.attributes.get("role").unwrap(), "developer");
        // KeepFirst does not merge attributes from b
        assert!(!merged.attributes.contains_key("dept"));
    }

    #[test]
    fn test_merge_entities_merge_all() {
        let config = EnrichmentConfig {
            merge_strategy: MergeStrategy::MergeAll,
            ..Default::default()
        };
        let enricher = EntityEnricher::new(config);

        let mut a = make_entity("Entity A", EntityType::Organization);
        a.attributes.insert("country".to_string(), "US".to_string());
        a.confidence = 0.8;

        let mut b = make_entity("Entity B", EntityType::Organization);
        b.attributes.insert("country".to_string(), "UK".to_string());
        b.attributes
            .insert("sector".to_string(), "tech".to_string());
        b.confidence = 0.6;

        let merged = enricher.merge_entities(&a, &b);
        // MergeAll: b overwrites a for shared keys
        assert_eq!(merged.attributes.get("country").unwrap(), "UK");
        assert_eq!(merged.attributes.get("sector").unwrap(), "tech");
        // Confidence is average
        assert!((merged.confidence - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_check_duplicate_against_known() {
        let config = EnrichmentConfig::default();
        let mut enricher = EntityEnricher::new(config);

        let known = make_entity("Microsoft Corporation", EntityType::Organization);
        enricher.register_entity(known);

        let check = make_entity("microsoft corporation", EntityType::Organization);
        let result = enricher.check_duplicate(&check);
        assert!(result.is_some());
        let dup = result.unwrap();
        assert_eq!(dup.reason, DuplicateReason::NormalizedMatch);
        assert_eq!(dup.matches_text, "Microsoft Corporation");
    }

    #[test]
    fn test_auto_tag_generates_type_and_attribute_tags() {
        let config = EnrichmentConfig::default();
        let enricher = EntityEnricher::new(config);

        let mut attrs = HashMap::new();
        attrs.insert("language".to_string(), "Rust".to_string());
        attrs.insert("version".to_string(), "1.75".to_string());

        let enriched = EnrichedEntity {
            entity: make_entity("Rust", EntityType::ProgrammingLanguage),
            enrichments: Vec::new(),
            merged_attributes: attrs,
            enrichment_confidence: 0.9,
            duplicate_of: None,
            auto_tags: Vec::new(),
        };

        let tags = enricher.auto_tag(&enriched);
        assert!(tags.contains(&"type:language".to_string()));
        assert!(tags.contains(&"attr:language".to_string()));
        assert!(tags.contains(&"attr:version".to_string()));
    }

    #[test]
    fn test_enrich_without_sources_returns_entity_with_original_confidence() {
        let config = EnrichmentConfig::default();
        let mut enricher = EntityEnricher::new(config);

        let entity = make_entity("test@example.com", EntityType::Email);
        let result = enricher.enrich(&entity);
        assert!(result.is_ok());

        let enriched = result.unwrap();
        assert!(enriched.enrichments.is_empty());
        assert!((enriched.enrichment_confidence - entity.confidence).abs() < f32::EPSILON);
        // Auto-tags should be generated
        assert!(enriched.auto_tags.contains(&"type:email".to_string()));
    }

    #[test]
    fn test_enrich_batch_processes_all_entities() {
        let config = EnrichmentConfig::default();
        let mut enricher = EntityEnricher::new(config);

        let entities = vec![
            make_entity("Alice", EntityType::Person),
            make_entity("bob@test.com", EntityType::Email),
            make_entity("Acme Corp", EntityType::Organization),
        ];

        let results = enricher.enrich_batch(&entities);
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_merge_attributes_b_overwrites_a() {
        let mut a = HashMap::new();
        a.insert("key1".to_string(), "val_a".to_string());
        a.insert("key2".to_string(), "only_a".to_string());

        let mut b = HashMap::new();
        b.insert("key1".to_string(), "val_b".to_string());
        b.insert("key3".to_string(), "only_b".to_string());

        let merged = merge_attributes(&a, &b);
        assert_eq!(merged.get("key1").unwrap(), "val_b");
        assert_eq!(merged.get("key2").unwrap(), "only_a");
        assert_eq!(merged.get("key3").unwrap(), "only_b");
    }

    #[test]
    fn test_fuzzy_similarity_empty_strings() {
        assert!((fuzzy_similarity("", "") - 1.0).abs() < f32::EPSILON);
        assert!((fuzzy_similarity("abc", "")).abs() < f32::EPSILON);
        assert!((fuzzy_similarity("", "xyz")).abs() < f32::EPSILON);
    }

    #[test]
    fn test_enrichment_config_default() {
        let config = EnrichmentConfig::default();
        assert!(config.sources.is_empty());
        assert_eq!(config.merge_strategy, MergeStrategy::MergeAll);
        assert!((config.min_enrichment_confidence - 0.5).abs() < f32::EPSILON);
        assert!((config.duplicate_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(config.max_enrichments, 5);
        assert!(config.auto_tagging);
        assert!(config.dedup_enabled);
    }

    #[test]
    fn test_find_duplicates_fuzzy_match() {
        let config = EnrichmentConfig {
            duplicate_threshold: 0.5, // Lower threshold for testing
            ..Default::default()
        };
        let enricher = EntityEnricher::new(config);

        let entities = vec![
            make_entity("Microsoft Corporation", EntityType::Organization),
            make_entity("Microsoft Corporatin", EntityType::Organization), // typo
        ];

        let dupes = enricher.find_duplicates(&entities);
        assert!(!dupes.is_empty());
        assert_eq!(dupes[0].reason, DuplicateReason::FuzzyMatch);
        assert!(dupes[0].similarity >= 0.5);
    }

    #[test]
    fn test_merge_entities_keep_highest_confidence() {
        let config = EnrichmentConfig {
            merge_strategy: MergeStrategy::KeepHighestConfidence,
            ..Default::default()
        };
        let enricher = EntityEnricher::new(config);

        let mut a = make_entity("Low Conf", EntityType::Person);
        a.confidence = 0.3;
        a.attributes.insert("from_a".to_string(), "yes".to_string());

        let mut b = make_entity("High Conf", EntityType::Person);
        b.confidence = 0.95;
        b.attributes.insert("from_b".to_string(), "yes".to_string());

        let merged = enricher.merge_entities(&a, &b);
        // Should keep b's text since it has higher confidence
        assert_eq!(merged.text, "High Conf");
        assert_eq!(merged.confidence, 0.95);
        // Attributes should be merged (b overwrites a)
        assert!(merged.attributes.contains_key("from_a"));
        assert!(merged.attributes.contains_key("from_b"));
    }
}
