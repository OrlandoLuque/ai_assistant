//! Episodic memory: time-stamped experiences with embedding-based similarity recall.

use serde::{Deserialize, Serialize};

use crate::error::{AdvancedMemoryError, AiError};
use super::helpers::cosine_similarity;

/// A single episodic memory entry — a recorded experience with context and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub content: String,
    pub context: String,
    pub timestamp: u64,
    pub importance: f64,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub access_count: usize,
    pub last_accessed: u64,
}

/// Store for episodic memories with capacity limits and temporal decay.
#[derive(Debug)]
pub struct EpisodicStore {
    episodes: Vec<Episode>,
    max_episodes: usize,
    decay_factor: f64,
}

impl EpisodicStore {
    /// Create a new episodic store with the given capacity and temporal decay factor.
    ///
    /// `decay_factor` controls how quickly older memories lose relevance (0.0 = no
    /// decay, 1.0 = aggressive decay). Typical values are 0.001 .. 0.01.
    pub fn new(max_episodes: usize, decay_factor: f64) -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes,
            decay_factor,
        }
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Add an episode to the store. If capacity is reached the oldest episode is
    /// evicted first.
    pub fn add(&mut self, episode: Episode) {
        if self.episodes.len() >= self.max_episodes {
            self.remove_oldest();
        }
        self.episodes.push(episode);
    }

    /// Retrieve an episode by id (mutable so we can track access).
    pub fn get(&mut self, id: &str) -> Option<&Episode> {
        if let Some(ep) = self.episodes.iter_mut().find(|e| e.id == id) {
            ep.access_count += 1;
            ep.last_accessed = Self::now();
            // Return shared ref after mutation.
            let idx = self.episodes.iter().position(|e| e.id == id);
            return idx.map(|i| &self.episodes[i]);
        }
        None
    }

    /// Recall the top-k episodes most similar to `query_embedding`, weighted by
    /// temporal decay and importance.
    pub fn recall(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<Episode> {
        let now = Self::now();
        let mut scored: Vec<(f64, usize)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(idx, ep)| {
                let sim = cosine_similarity(query_embedding, &ep.embedding);
                let age = (now.saturating_sub(ep.timestamp)) as f64;
                let decay = (-self.decay_factor * age).exp();
                let score = sim * decay * (0.5 + 0.5 * ep.importance);
                (score, idx)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(scored.len());
        let mut results = Vec::with_capacity(k);
        for &(_, idx) in scored.iter().take(k) {
            self.episodes[idx].access_count += 1;
            self.episodes[idx].last_accessed = now;
            results.push(self.episodes[idx].clone());
        }
        results
    }

    /// Recall episodes that share at least one of the given tags, returning the
    /// top-k by number of matching tags then by importance descending.
    pub fn recall_by_tags(&mut self, tags: &[String], top_k: usize) -> Vec<Episode> {
        let now = Self::now();
        let mut scored: Vec<(usize, f64, usize)> = self
            .episodes
            .iter()
            .enumerate()
            .filter_map(|(idx, ep)| {
                let matching = ep.tags.iter().filter(|t| tags.contains(t)).count();
                if matching > 0 {
                    Some((matching, ep.importance, idx))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let k = top_k.min(scored.len());
        let mut results = Vec::with_capacity(k);
        for &(_, _, idx) in scored.iter().take(k) {
            self.episodes[idx].access_count += 1;
            self.episodes[idx].last_accessed = now;
            results.push(self.episodes[idx].clone());
        }
        results
    }

    /// Remove the oldest episode (by timestamp).
    pub fn remove_oldest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }
        let oldest_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by_key(|(_, ep)| ep.timestamp)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.episodes.remove(oldest_idx);
    }

    /// Serialize the store to a JSON string.
    pub fn to_json(&self) -> Result<String, AiError> {
        serde_json::to_string(&self.episodes).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::StoreFailed {
                memory_type: "episodic".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Deserialize episodes from a JSON string, replacing current contents.
    pub fn from_json(&mut self, json: &str) -> Result<(), AiError> {
        let episodes: Vec<Episode> = serde_json::from_str(json).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::RecallFailed {
                query: "from_json".to_string(),
                reason: e.to_string(),
            })
        })?;
        self.episodes = episodes;
        Ok(())
    }

    /// Get a read-only slice of all episodes.
    pub fn all(&self) -> &[Episode] {
        &self.episodes
    }

    // Simple monotonic "now" for timestamping — in production would use
    // `std::time::SystemTime`; here we use it for sorting/decay.
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Save the episodic store to a JSON file. Uses atomic write (temp file + rename).
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<String, String> {
        let json = serde_json::to_string_pretty(&self.episodes)
            .map_err(|e| format!("Serialize error: {}", e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("Rename error: {}", e))?;
        Ok(json)
    }

    /// Load an episodic store from a JSON file.
    pub fn load_from_file(path: &std::path::Path, max_episodes: usize, decay_factor: f64) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let episodes: Vec<Episode> =
            serde_json::from_str(&data).map_err(|e| format!("Deserialize error: {}", e))?;
        Ok(Self {
            episodes,
            max_episodes,
            decay_factor,
        })
    }
}
