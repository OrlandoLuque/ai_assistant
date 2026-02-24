//! Temporal memory graphs: directed temporal relationships between episodes.

use serde::{Deserialize, Serialize};

use super::episodic::Episode;
use super::helpers::keyword_overlap;

/// The type of temporal relationship between two episodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalEdgeType {
    /// Episode A happened before episode B.
    Before,
    /// Episode A happened after episode B.
    After,
    /// Episode A caused episode B.
    Causes,
    /// Episode A was enabled by episode B.
    EnabledBy,
    /// Episodes A and B co-occurred (within a small time window).
    CoOccurs,
}

/// A directed temporal edge between two episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    pub from_episode_id: String,
    pub to_episode_id: String,
    pub edge_type: TemporalEdgeType,
    pub confidence: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// A directed graph representing temporal relationships between episodes.
pub struct TemporalGraph {
    edges: Vec<TemporalEdge>,
    episode_ids: std::collections::HashSet<String>,
}

/// Specifies a temporal query type.
#[derive(Debug, Clone)]
pub enum TemporalQueryType {
    /// What caused this episode?
    WhatCaused,
    /// What followed this episode?
    WhatFollowed,
    /// What preceded this episode?
    WhatPreceded,
    /// What co-occurred with this episode?
    WhatCoOccurred,
}

/// A query against the temporal graph.
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    pub query_type: TemporalQueryType,
    pub episode_id: String,
    pub max_depth: usize,
}

impl TemporalQuery {
    /// Create a new temporal query with default max_depth of 5.
    pub fn new(query_type: TemporalQueryType, episode_id: &str) -> Self {
        Self {
            query_type,
            episode_id: episode_id.to_string(),
            max_depth: 5,
        }
    }

    /// Set the maximum traversal depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

impl TemporalGraph {
    /// Create an empty temporal graph.
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            episode_ids: std::collections::HashSet::new(),
        }
    }

    /// Register an episode in the graph (even if it has no edges yet).
    pub fn add_episode(&mut self, episode_id: &str) {
        self.episode_ids.insert(episode_id.to_string());
    }

    /// Add a directed edge to the graph.
    pub fn add_edge(&mut self, edge: TemporalEdge) {
        self.episode_ids.insert(edge.from_episode_id.clone());
        self.episode_ids.insert(edge.to_episode_id.clone());
        self.edges.push(edge);
    }

    /// Automatically create Before/After edges based on episode timestamps.
    ///
    /// Two episodes are linked if their timestamps differ by at most
    /// `max_gap_secs` (default: 3600 seconds = 1 hour).
    pub fn auto_link_temporal(&mut self, episodes: &[Episode]) {
        self.auto_link_temporal_with_gap(episodes, 3600);
    }

    /// Auto-link with a custom maximum time gap in seconds.
    pub fn auto_link_temporal_with_gap(&mut self, episodes: &[Episode], max_gap_secs: u64) {
        for ep in episodes {
            self.episode_ids.insert(ep.id.clone());
        }

        for i in 0..episodes.len() {
            for j in (i + 1)..episodes.len() {
                let a = &episodes[i];
                let b = &episodes[j];

                let diff = a.timestamp.abs_diff(b.timestamp);

                if diff <= max_gap_secs {
                    let now = chrono::Utc::now();

                    if a.timestamp < b.timestamp {
                        self.edges.push(TemporalEdge {
                            from_episode_id: a.id.clone(),
                            to_episode_id: b.id.clone(),
                            edge_type: TemporalEdgeType::Before,
                            confidence: 1.0,
                            created_at: now,
                        });
                        self.edges.push(TemporalEdge {
                            from_episode_id: b.id.clone(),
                            to_episode_id: a.id.clone(),
                            edge_type: TemporalEdgeType::After,
                            confidence: 1.0,
                            created_at: now,
                        });
                    } else if a.timestamp > b.timestamp {
                        self.edges.push(TemporalEdge {
                            from_episode_id: b.id.clone(),
                            to_episode_id: a.id.clone(),
                            edge_type: TemporalEdgeType::Before,
                            confidence: 1.0,
                            created_at: now,
                        });
                        self.edges.push(TemporalEdge {
                            from_episode_id: a.id.clone(),
                            to_episode_id: b.id.clone(),
                            edge_type: TemporalEdgeType::After,
                            confidence: 1.0,
                            created_at: now,
                        });
                    } else {
                        // Same timestamp -> co-occurs
                        self.edges.push(TemporalEdge {
                            from_episode_id: a.id.clone(),
                            to_episode_id: b.id.clone(),
                            edge_type: TemporalEdgeType::CoOccurs,
                            confidence: 1.0,
                            created_at: now,
                        });
                        self.edges.push(TemporalEdge {
                            from_episode_id: b.id.clone(),
                            to_episode_id: a.id.clone(),
                            edge_type: TemporalEdgeType::CoOccurs,
                            confidence: 1.0,
                            created_at: now,
                        });
                    }
                }
            }
        }
    }

    /// Detect potential causal edges: if episode A's content keywords appear in
    /// episode B's context (and A happened before B), create a Causes edge.
    pub fn auto_detect_causal(&mut self, episodes: &[Episode]) {
        for ep in episodes {
            self.episode_ids.insert(ep.id.clone());
        }

        for i in 0..episodes.len() {
            for j in 0..episodes.len() {
                if i == j {
                    continue;
                }
                let a = &episodes[i];
                let b = &episodes[j];

                // A must happen before B
                if a.timestamp >= b.timestamp {
                    continue;
                }

                // Check if A's content keywords appear in B's context
                let overlap = keyword_overlap(&a.content, &b.context);
                if overlap >= 0.3 {
                    let now = chrono::Utc::now();
                    self.edges.push(TemporalEdge {
                        from_episode_id: a.id.clone(),
                        to_episode_id: b.id.clone(),
                        edge_type: TemporalEdgeType::Causes,
                        confidence: overlap.min(1.0),
                        created_at: now,
                    });
                }
            }
        }
    }

    /// Get all edges originating from the given episode.
    pub fn get_edges_from(&self, episode_id: &str) -> Vec<&TemporalEdge> {
        self.edges
            .iter()
            .filter(|e| e.from_episode_id == episode_id)
            .collect()
    }

    /// Get all edges pointing to the given episode.
    pub fn get_edges_to(&self, episode_id: &str) -> Vec<&TemporalEdge> {
        self.edges
            .iter()
            .filter(|e| e.to_episode_id == episode_id)
            .collect()
    }

    /// Follow Causes edges forward from `start_id`, returning the causal chain
    /// of episode ids in order. Stops at `max_depth` or when no more Causes
    /// edges are found.
    pub fn get_causal_chain(&self, start_id: &str) -> Vec<String> {
        self.get_causal_chain_with_depth(start_id, 10)
    }

    /// Follow Causes edges with a configurable depth limit.
    pub fn get_causal_chain_with_depth(&self, start_id: &str, max_depth: usize) -> Vec<String> {
        let mut chain = Vec::new();
        let mut current = start_id.to_string();
        let mut visited = std::collections::HashSet::new();
        visited.insert(current.clone());

        for _ in 0..max_depth {
            let next = self
                .edges
                .iter()
                .filter(|e| {
                    e.from_episode_id == current && e.edge_type == TemporalEdgeType::Causes
                })
                .max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            match next {
                Some(edge) => {
                    if visited.contains(&edge.to_episode_id) {
                        break; // cycle detection
                    }
                    chain.push(edge.to_episode_id.clone());
                    visited.insert(edge.to_episode_id.clone());
                    current = edge.to_episode_id.clone();
                }
                None => break,
            }
        }
        chain
    }

    /// Get all episodes that happened Before the given episode (direct edges only).
    pub fn get_predecessors(&self, episode_id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| {
                e.to_episode_id == episode_id && e.edge_type == TemporalEdgeType::Before
            })
            .map(|e| e.from_episode_id.clone())
            .collect()
    }

    /// Get all episodes that happened After the given episode (direct edges only).
    pub fn get_successors(&self, episode_id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| {
                e.from_episode_id == episode_id && e.edge_type == TemporalEdgeType::Before
            })
            .map(|e| e.to_episode_id.clone())
            .collect()
    }

    /// Execute a temporal query and return matching episode ids.
    pub fn query_temporal(&self, query: &TemporalQuery) -> Vec<String> {
        match &query.query_type {
            TemporalQueryType::WhatCaused => {
                // Find episodes that have a Causes edge pointing to the query episode
                self.edges
                    .iter()
                    .filter(|e| {
                        e.to_episode_id == query.episode_id
                            && e.edge_type == TemporalEdgeType::Causes
                    })
                    .map(|e| e.from_episode_id.clone())
                    .collect()
            }
            TemporalQueryType::WhatFollowed => {
                self.get_causal_chain_with_depth(&query.episode_id, query.max_depth)
            }
            TemporalQueryType::WhatPreceded => self.get_predecessors(&query.episode_id),
            TemporalQueryType::WhatCoOccurred => self
                .edges
                .iter()
                .filter(|e| {
                    e.from_episode_id == query.episode_id
                        && e.edge_type == TemporalEdgeType::CoOccurs
                })
                .map(|e| e.to_episode_id.clone())
                .collect(),
        }
    }
}
