//! Cross-session memory sharing: shared memory pools with sync policies and filters.

use serde::{Deserialize, Serialize};

use super::consolidation::SemanticFact;
use super::episodic::Episode;

/// Policy controlling when memories are synchronized between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySyncPolicy {
    /// Synchronize immediately on every change.
    Eager,
    /// Synchronize only when queried.
    Lazy,
    /// Synchronize at a fixed interval.
    Periodic { interval_secs: u64 },
    /// Only synchronize when explicitly triggered.
    Manual,
}

/// A filter for selecting a subset of memories when querying a shared pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFilter {
    /// Minimum confidence threshold (only facts at or above this are returned).
    pub min_confidence: Option<f64>,
    /// If non-empty, only facts whose predicate matches one of these categories
    /// are returned.
    pub categories: Vec<String>,
    /// If non-empty, only facts whose subject matches one of these entity names
    /// are returned.
    pub entity_names: Vec<String>,
    /// Maximum age in seconds; facts older than this are excluded.
    pub max_age_secs: Option<u64>,
}

impl MemoryFilter {
    /// Create a new empty filter that matches all memories.
    pub fn new() -> Self {
        Self {
            min_confidence: None,
            categories: Vec::new(),
            entity_names: Vec::new(),
            max_age_secs: None,
        }
    }

    /// Create a filter that only accepts facts at or above the given confidence.
    pub fn with_min_confidence(confidence: f64) -> Self {
        Self {
            min_confidence: Some(confidence),
            categories: Vec::new(),
            entity_names: Vec::new(),
            max_age_secs: None,
        }
    }

    /// Create a filter that only accepts facts whose predicate matches one of
    /// the given categories.
    pub fn with_categories(categories: Vec<String>) -> Self {
        Self {
            min_confidence: None,
            categories,
            entity_names: Vec::new(),
            max_age_secs: None,
        }
    }

    /// Check whether a semantic fact passes this filter.
    pub fn matches_fact(&self, fact: &SemanticFact) -> bool {
        if let Some(min_conf) = self.min_confidence {
            if fact.confidence < min_conf {
                return false;
            }
        }
        if !self.categories.is_empty() {
            let pred_lower = fact.predicate.to_lowercase();
            if !self.categories.iter().any(|c| c.to_lowercase() == pred_lower) {
                return false;
            }
        }
        if !self.entity_names.is_empty() {
            let subj_lower = fact.subject.to_lowercase();
            if !self
                .entity_names
                .iter()
                .any(|n| n.to_lowercase() == subj_lower)
            {
                return false;
            }
        }
        true
    }

    /// Check whether an episode passes this filter.
    ///
    /// For episodes, `max_age_secs` is checked against the episode's timestamp,
    /// and `categories` is matched against the episode's tags.
    pub fn matches_episode(&self, episode: &Episode) -> bool {
        if let Some(max_age) = self.max_age_secs {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let age = now.saturating_sub(episode.timestamp);
            if age > max_age {
                return false;
            }
        }
        if !self.categories.is_empty() {
            let has_matching_tag = episode.tags.iter().any(|tag| {
                let tag_lower = tag.to_lowercase();
                self.categories.iter().any(|c| c.to_lowercase() == tag_lower)
            });
            if !has_matching_tag {
                return false;
            }
        }
        true
    }
}

/// A shared memory pool for cross-session and cross-agent memory sharing.
pub struct SharedMemoryPool {
    facts: Vec<(String, SemanticFact)>,
    sync_policy: MemorySyncPolicy,
    subscribers: Vec<String>,
}

impl SharedMemoryPool {
    /// Create a new shared pool with the given synchronization policy.
    pub fn new(sync_policy: MemorySyncPolicy) -> Self {
        Self {
            facts: Vec::new(),
            sync_policy,
            subscribers: Vec::new(),
        }
    }

    /// Publish a fact from a specific agent into the shared pool.
    pub fn publish(&mut self, agent_id: String, fact: SemanticFact) {
        self.facts.push((agent_id, fact));
    }

    /// Subscribe an agent to the shared pool.
    pub fn subscribe(&mut self, agent_id: String) {
        if !self.subscribers.contains(&agent_id) {
            self.subscribers.push(agent_id);
        }
    }

    /// Unsubscribe an agent from the shared pool. Returns `true` if the agent
    /// was previously subscribed.
    pub fn unsubscribe(&mut self, agent_id: &str) -> bool {
        let before = self.subscribers.len();
        self.subscribers.retain(|id| id != agent_id);
        self.subscribers.len() < before
    }

    /// Query the pool for facts matching the given filter.
    pub fn query(&self, filter: &MemoryFilter) -> Vec<&SemanticFact> {
        self.facts
            .iter()
            .map(|(_, fact)| fact)
            .filter(|fact| filter.matches_fact(fact))
            .collect()
    }

    /// Query the pool for all facts published by a specific agent.
    pub fn query_by_agent(&self, agent_id: &str) -> Vec<&SemanticFact> {
        self.facts
            .iter()
            .filter(|(aid, _)| aid == agent_id)
            .map(|(_, fact)| fact)
            .collect()
    }

    /// Return the total number of facts in the pool.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Return the number of subscribed agents.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Remove all facts and subscribers from the pool.
    pub fn clear(&mut self) {
        self.facts.clear();
        self.subscribers.clear();
    }

    /// Get a reference to the synchronization policy.
    pub fn sync_policy(&self) -> &MemorySyncPolicy {
        &self.sync_policy
    }
}
