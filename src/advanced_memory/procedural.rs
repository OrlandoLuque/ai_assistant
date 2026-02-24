//! Procedural memory: learned procedures/routines with confidence tracking.

use serde::{Deserialize, Serialize};

use crate::error::{AdvancedMemoryError, AiError};

/// A learned procedure — a sequence of steps with a triggering condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub steps: Vec<String>,
    pub success_count: usize,
    pub failure_count: usize,
    pub confidence: f64,
    pub created_from: Vec<String>,
    pub tags: Vec<String>,
}

/// Store for procedural memories with capacity limits.
#[derive(Debug)]
pub struct ProceduralStore {
    pub(crate) procedures: Vec<Procedure>,
    max_procedures: usize,
}

impl ProceduralStore {
    /// Create a new procedural store with the given capacity.
    pub fn new(max_procedures: usize) -> Self {
        Self {
            procedures: Vec::new(),
            max_procedures,
        }
    }

    /// Number of stored procedures.
    pub fn len(&self) -> usize {
        self.procedures.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.procedures.is_empty()
    }

    /// Add a procedure. If capacity is reached the least-confident procedure is
    /// evicted first.
    pub fn add(&mut self, procedure: Procedure) {
        if self.procedures.len() >= self.max_procedures {
            // Evict least confident
            if let Some(idx) = self
                .procedures
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.procedures.remove(idx);
            }
        }
        self.procedures.push(procedure);
    }

    /// Find procedures whose condition keywords match the given context string.
    /// Returns matches sorted by confidence descending.
    pub fn find_by_condition(&self, context: &str) -> Vec<&Procedure> {
        let ctx_lower = context.to_lowercase();
        let ctx_words: std::collections::HashSet<&str> = ctx_lower.split_whitespace().collect();

        let mut matches: Vec<(f64, &Procedure)> = self
            .procedures
            .iter()
            .filter_map(|p| {
                let cond_lower = p.condition.to_lowercase();
                let cond_words: Vec<&str> = cond_lower.split_whitespace().collect();
                let matching = cond_words.iter().filter(|w| ctx_words.contains(*w)).count();
                if matching > 0 {
                    Some((p.confidence, p))
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches.into_iter().map(|(_, p)| p).collect()
    }

    /// Record an outcome (success or failure) for a procedure and update its
    /// confidence.
    pub fn update_outcome(&mut self, id: &str, success: bool) -> Result<(), AiError> {
        let proc = self.procedures.iter_mut().find(|p| p.id == id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        if success {
            proc.success_count += 1;
        } else {
            proc.failure_count += 1;
        }
        let total = proc.success_count + proc.failure_count;
        proc.confidence = proc.success_count as f64 / total as f64;
        Ok(())
    }

    /// Return the top-n most confident procedures.
    pub fn most_confident(&self, n: usize) -> Vec<&Procedure> {
        let mut sorted: Vec<&Procedure> = self.procedures.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Retrieve a procedure by id.
    pub fn get(&self, id: &str) -> Option<&Procedure> {
        self.procedures.iter().find(|p| p.id == id)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, AiError> {
        serde_json::to_string(&self.procedures).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::StoreFailed {
                memory_type: "procedural".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Deserialize from JSON, replacing current contents.
    pub fn from_json(&mut self, json: &str) -> Result<(), AiError> {
        let procs: Vec<Procedure> = serde_json::from_str(json).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::RecallFailed {
                query: "from_json".to_string(),
                reason: e.to_string(),
            })
        })?;
        self.procedures = procs;
        Ok(())
    }

    /// Read-only access to all procedures.
    pub fn all(&self) -> &[Procedure] {
        &self.procedures
    }

    /// Save the procedural store to a JSON file. Uses atomic write (temp file + rename).
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<String, String> {
        let json = serde_json::to_string_pretty(&self.procedures)
            .map_err(|e| format!("Serialize error: {}", e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("Rename error: {}", e))?;
        Ok(json)
    }

    /// Load a procedural store from a JSON file.
    pub fn load_from_file(path: &std::path::Path, max_procedures: usize) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let procedures: Vec<Procedure> =
            serde_json::from_str(&data).map_err(|e| format!("Deserialize error: {}", e))?;
        Ok(Self {
            procedures,
            max_procedures,
        })
    }
}
