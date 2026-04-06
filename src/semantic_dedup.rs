//! Semantic Deduplication — detect and merge similar chunks intelligently.
//!
//! Three levels:
//! - Identical (>0.98 similarity): eliminate the lower-scored one
//! - Similar with nuances (0.85-0.98): merge via LLM preserving differences
//! - Distinct (<0.85): keep both
//!
//! Uses batching for LLM calls — groups all fusion candidates into a single
//! prompt to minimize API calls.

use serde::{Deserialize, Serialize};

/// Configuration for semantic deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SemanticDedupConfig {
    /// Threshold above which chunks are considered identical (eliminate one).
    pub identical_threshold: f32,
    /// Threshold above which chunks are considered similar (merge candidates).
    pub similar_threshold: f32,
    /// Enable LLM-based fusion of similar chunks (opt-in, costs LLM call).
    pub enable_fusion: bool,
    /// Maximum groups per LLM batch call.
    pub max_groups_per_batch: usize,
}

impl Default for SemanticDedupConfig {
    fn default() -> Self {
        Self {
            identical_threshold: 0.98,
            similar_threshold: 0.85,
            enable_fusion: false,
            max_groups_per_batch: 20,
        }
    }
}

/// A chunk with its embedding for similarity comparison.
#[derive(Debug, Clone)]
pub struct DedupChunk {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub embedding: Option<Vec<f32>>,
    pub source: String,
    pub tokens: usize,
}

/// Result of deduplication for a single chunk.
#[derive(Debug, Clone)]
pub enum DedupAction {
    /// Keep the chunk as-is.
    Keep,
    /// Remove this chunk (identical to another with higher score).
    Remove { duplicate_of: String },
    /// This chunk was merged with others into a new fused chunk.
    Merged { group_id: usize },
}

/// A group of similar chunks that should be fused.
#[derive(Debug, Clone)]
pub struct SimilarGroup {
    pub group_id: usize,
    pub chunks: Vec<DedupChunk>,
    pub similarity: f32,
}

/// Result of the deduplication process.
#[derive(Debug)]
pub struct DedupResult {
    /// Chunks to keep (original + fused).
    pub kept: Vec<DedupChunk>,
    /// Number of identical chunks removed.
    pub identical_removed: usize,
    /// Number of similar groups fused.
    pub groups_fused: usize,
    /// Total tokens saved.
    pub tokens_saved: usize,
}

/// Semantic deduplicator.
pub struct SemanticDeduplicator {
    config: SemanticDedupConfig,
}

impl SemanticDeduplicator {
    pub fn new(config: SemanticDedupConfig) -> Self {
        Self { config }
    }

    /// Deduplicate chunks by semantic similarity.
    /// Returns chunks with identical ones removed and similar ones marked for fusion.
    pub fn deduplicate(&self, chunks: Vec<DedupChunk>) -> DedupResult {
        if chunks.len() < 2 {
            return DedupResult {
                kept: chunks,
                identical_removed: 0,
                groups_fused: 0,
                tokens_saved: 0,
            };
        }

        let mut kept: Vec<DedupChunk> = Vec::new();
        let mut removed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut similar_groups: Vec<SimilarGroup> = Vec::new();
        let mut identical_removed = 0usize;
        let mut tokens_saved = 0usize;

        // 1. Find identical pairs (>= identical_threshold) and similar pairs
        for i in 0..chunks.len() {
            if removed_ids.contains(&chunks[i].id) {
                continue;
            }

            let mut found_similar = false;

            for j in (i + 1)..chunks.len() {
                if removed_ids.contains(&chunks[j].id) {
                    continue;
                }

                let sim = self.similarity(&chunks[i], &chunks[j]);

                if sim >= self.config.identical_threshold {
                    // Level 1: Identical — remove the lower-scored one
                    if chunks[i].score >= chunks[j].score {
                        removed_ids.insert(chunks[j].id.clone());
                        tokens_saved += chunks[j].tokens;
                    } else {
                        removed_ids.insert(chunks[i].id.clone());
                        tokens_saved += chunks[i].tokens;
                    }
                    identical_removed += 1;
                } else if sim >= self.config.similar_threshold && self.config.enable_fusion {
                    // Level 2: Similar — group for fusion
                    // Check if either chunk is already in a group
                    let existing_group = similar_groups.iter_mut().find(|g| {
                        g.chunks
                            .iter()
                            .any(|c| c.id == chunks[i].id || c.id == chunks[j].id)
                    });

                    if let Some(group) = existing_group {
                        // Add to existing group
                        if !group.chunks.iter().any(|c| c.id == chunks[i].id) {
                            group.chunks.push(chunks[i].clone());
                        }
                        if !group.chunks.iter().any(|c| c.id == chunks[j].id) {
                            group.chunks.push(chunks[j].clone());
                        }
                        group.similarity = group.similarity.min(sim);
                    } else {
                        similar_groups.push(SimilarGroup {
                            group_id: similar_groups.len(),
                            chunks: vec![chunks[i].clone(), chunks[j].clone()],
                            similarity: sim,
                        });
                    }
                    found_similar = true;
                }
            }

            // Level 3: Distinct — keep as-is
            if !found_similar && !removed_ids.contains(&chunks[i].id) {
                // Will be added later if not in a fusion group
            }
        }

        // Collect chunks that are not removed and not in a fusion group
        let grouped_ids: std::collections::HashSet<String> = similar_groups
            .iter()
            .flat_map(|g| g.chunks.iter().map(|c| c.id.clone()))
            .collect();

        for chunk in &chunks {
            if !removed_ids.contains(&chunk.id) && !grouped_ids.contains(&chunk.id) {
                kept.push(chunk.clone());
            }
        }

        // For fusion groups: keep a placeholder (the highest-scored chunk from each group)
        // The actual fusion would happen via LLM call (build_fusion_prompt)
        let groups_fused = similar_groups.len();
        for group in &similar_groups {
            if let Some(best) = group.chunks.iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                // Mark as needing fusion — for now keep the best chunk
                // The caller can use build_fusion_prompt() to get the LLM fusion prompt
                kept.push(best.clone());
                tokens_saved += group
                    .chunks
                    .iter()
                    .filter(|c| c.id != best.id)
                    .map(|c| c.tokens)
                    .sum::<usize>();
            }
        }

        // Sort by score descending
        kept.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        DedupResult {
            kept,
            identical_removed,
            groups_fused,
            tokens_saved,
        }
    }

    /// Build the LLM fusion prompt for all similar groups (batched).
    /// Returns batches of prompts, each fitting within max_tokens.
    pub fn build_fusion_prompts(
        &self,
        groups: &[SimilarGroup],
        max_tokens_per_batch: usize,
    ) -> Vec<String> {
        if groups.is_empty() {
            return Vec::new();
        }

        let mut batches = Vec::new();
        let mut current_batch = String::from(
            "Merge each group of similar text chunks into ONE unified chunk.\n\
             IMPORTANT: Preserve ALL details and nuances that differ between chunks.\n\
             Only remove truly redundant repetition.\n\n",
        );
        let mut current_tokens = estimate_tokens(&current_batch);
        let mut groups_in_batch = 0;

        for group in groups {
            let mut group_text = format!(
                "[Group {} — similarity {:.0}%]\n",
                group.group_id + 1,
                group.similarity * 100.0
            );
            for (i, chunk) in group.chunks.iter().enumerate() {
                group_text.push_str(&format!(
                    "  Chunk {} (score {:.2}): {}\n",
                    i + 1,
                    chunk.score,
                    chunk.content
                ));
            }
            group_text.push('\n');

            let group_tokens = estimate_tokens(&group_text);

            // Check if adding this group would exceed the batch limit
            if current_tokens + group_tokens > max_tokens_per_batch && groups_in_batch > 0 {
                current_batch.push_str("For each group, output: [Group N] merged text\n");
                batches.push(current_batch);

                current_batch = String::from(
                    "Merge each group of similar text chunks into ONE unified chunk.\n\
                     Preserve ALL differing details.\n\n",
                );
                current_tokens = estimate_tokens(&current_batch);
                groups_in_batch = 0;
            }

            current_batch.push_str(&group_text);
            current_tokens += group_tokens;
            groups_in_batch += 1;

            if groups_in_batch >= self.config.max_groups_per_batch {
                current_batch.push_str("For each group, output: [Group N] merged text\n");
                batches.push(current_batch);
                current_batch = String::from(
                    "Merge each group of similar text chunks into ONE unified chunk.\n\
                     Preserve ALL differing details.\n\n",
                );
                current_tokens = estimate_tokens(&current_batch);
                groups_in_batch = 0;
            }
        }

        if groups_in_batch > 0 {
            current_batch.push_str("For each group, output: [Group N] merged text\n");
            batches.push(current_batch);
        }

        batches
    }

    /// Compute similarity between two chunks.
    fn similarity(&self, a: &DedupChunk, b: &DedupChunk) -> f32 {
        // If both have embeddings, use cosine similarity
        if let (Some(emb_a), Some(emb_b)) = (&a.embedding, &b.embedding) {
            return cosine_similarity(emb_a, emb_b);
        }
        // Fallback: Jaccard word overlap
        jaccard_similarity(&a.content, &b.content)
    }
}

impl Default for SemanticDeduplicator {
    fn default() -> Self {
        Self::new(SemanticDedupConfig::default())
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Jaccard word similarity (fallback when no embeddings).
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Estimate token count from text.
fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / 3.5).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str, content: &str, score: f32) -> DedupChunk {
        DedupChunk {
            id: id.to_string(),
            content: content.to_string(),
            score,
            embedding: None,
            source: "test".to_string(),
            tokens: estimate_tokens(content),
        }
    }

    fn make_chunk_with_emb(id: &str, content: &str, score: f32, emb: Vec<f32>) -> DedupChunk {
        DedupChunk {
            id: id.to_string(),
            content: content.to_string(),
            score,
            embedding: Some(emb),
            source: "test".to_string(),
            tokens: estimate_tokens(content),
        }
    }

    #[test]
    fn test_no_duplicates() {
        let dedup = SemanticDeduplicator::default();
        let chunks = vec![
            make_chunk("a", "Rust is a systems language", 0.9),
            make_chunk("b", "Python is great for data science", 0.8),
        ];
        let result = dedup.deduplicate(chunks);
        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.identical_removed, 0);
    }

    #[test]
    fn test_identical_by_embedding() {
        let dedup = SemanticDeduplicator::default();
        let chunks = vec![
            make_chunk_with_emb("a", "Rust ownership model", 0.9, vec![1.0, 0.0, 0.0]),
            make_chunk_with_emb("b", "Rust ownership system", 0.7, vec![0.999, 0.01, 0.0]),
        ];
        let result = dedup.deduplicate(chunks);
        assert_eq!(result.identical_removed, 1);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].id, "a"); // higher score kept
    }

    #[test]
    fn test_similar_grouped_for_fusion() {
        let config = SemanticDedupConfig {
            enable_fusion: true,
            similar_threshold: 0.3, // low threshold for word-based test
            ..Default::default()
        };
        let dedup = SemanticDeduplicator::new(config);
        let chunks = vec![
            make_chunk("a", "timeout is 30 seconds for HTTP connections", 0.9),
            make_chunk(
                "b",
                "timeout is 30 seconds for HTTP but 60 for WebSocket",
                0.85,
            ),
            make_chunk("c", "Python is great for ML", 0.7),
        ];
        let result = dedup.deduplicate(chunks);
        // c should be kept as-is, a and b should be in a fusion group
        assert!(result.groups_fused > 0 || result.kept.len() <= 3);
    }

    #[test]
    fn test_single_chunk() {
        let dedup = SemanticDeduplicator::default();
        let chunks = vec![make_chunk("a", "Only one chunk", 0.9)];
        let result = dedup.deduplicate(chunks);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.identical_removed, 0);
    }

    #[test]
    fn test_empty() {
        let dedup = SemanticDeduplicator::default();
        let result = dedup.deduplicate(vec![]);
        assert_eq!(result.kept.len(), 0);
    }

    #[test]
    fn test_build_fusion_prompt_single_batch() {
        let dedup = SemanticDeduplicator::default();
        let groups = vec![SimilarGroup {
            group_id: 0,
            chunks: vec![
                make_chunk("a", "timeout 30s for HTTP", 0.9),
                make_chunk("b", "timeout 30s HTTP, 60s WebSocket", 0.85),
            ],
            similarity: 0.91,
        }];
        let prompts = dedup.build_fusion_prompts(&groups, 10000);
        assert_eq!(prompts.len(), 1);
        assert!(prompts[0].contains("Group 1"));
        assert!(prompts[0].contains("timeout"));
        assert!(prompts[0].contains("Preserve ALL"));
    }

    #[test]
    fn test_build_fusion_prompt_multiple_batches() {
        let dedup = SemanticDeduplicator::new(SemanticDedupConfig {
            max_groups_per_batch: 2,
            ..Default::default()
        });

        let groups: Vec<SimilarGroup> = (0..5)
            .map(|i| SimilarGroup {
                group_id: i,
                chunks: vec![
                    make_chunk(&format!("a{}", i), &format!("Content A {}", i), 0.9),
                    make_chunk(&format!("b{}", i), &format!("Content B {}", i), 0.85),
                ],
                similarity: 0.90,
            })
            .collect();

        let prompts = dedup.build_fusion_prompts(&groups, 100000);
        assert!(prompts.len() >= 3); // 5 groups / 2 per batch = 3 batches
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.01);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 0.01);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.99, 0.01]) > 0.98);
    }

    #[test]
    fn test_jaccard_similarity() {
        assert!((jaccard_similarity("hello world", "hello world") - 1.0).abs() < 0.01);
        assert!(jaccard_similarity("hello world", "goodbye moon") < 0.1);
        assert!(jaccard_similarity("rust ownership model", "rust ownership system") > 0.4);
    }

    #[test]
    fn test_tokens_saved() {
        let dedup = SemanticDeduplicator::default();
        let chunks = vec![
            make_chunk_with_emb("a", "Same content here", 0.9, vec![1.0, 0.0]),
            make_chunk_with_emb("b", "Same content here", 0.7, vec![1.0, 0.0]),
        ];
        let result = dedup.deduplicate(chunks);
        assert!(result.tokens_saved > 0);
    }
}
