//! Helper functions for the advanced memory system.

use super::episodic::Episode;

/// Cosine similarity between two vectors.
///
/// Returns 0.0 when either vector is zero-length or when dimensions mismatch.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute keyword overlap ratio between two strings (lowercased, whitespace-split).
pub(crate) fn keyword_overlap(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let words_a: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Create a new episode with a generated UUID.
pub fn new_episode(
    content: impl Into<String>,
    context: impl Into<String>,
    importance: f64,
    tags: Vec<String>,
    embedding: Vec<f32>,
) -> Episode {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Episode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.into(),
        context: context.into(),
        timestamp: now,
        importance: importance.clamp(0.0, 1.0),
        tags,
        embedding,
        access_count: 0,
        last_accessed: now,
    }
}
