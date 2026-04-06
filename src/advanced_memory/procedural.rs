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
    #[serde(default)]
    pub success_count: usize,
    #[serde(default)]
    pub failure_count: usize,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub created_from: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_confidence() -> f64 {
    0.5
}

impl Default for Procedure {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            condition: String::new(),
            steps: Vec::new(),
            success_count: 0,
            failure_count: 0,
            confidence: 0.5,
            created_from: Vec::new(),
            tags: Vec::new(),
        }
    }
}

/// Versioned export format for sharing procedure libraries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProcedureExport {
    /// Schema version (for forward compatibility).
    pub version: u32,
    /// ISO 8601 export timestamp.
    pub exported_at: String,
    /// Source label (e.g. "defaults", "user", "imported").
    pub source: String,
    /// Optional user ID for multi-user clarity.
    #[serde(default)]
    pub user_id: String,
    /// The procedures.
    pub procedures: Vec<Procedure>,
}

impl Default for ProcedureExport {
    fn default() -> Self {
        Self {
            version: 1,
            exported_at: chrono::Utc::now().to_rfc3339(),
            source: "user".to_string(),
            user_id: String::new(),
            procedures: Vec::new(),
        }
    }
}

/// Options for importing procedures.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProcedureImportOptions {
    /// If true, add to existing procedures. If false, replace all.
    pub merge: bool,
    /// If true and merge is true, skip procedures whose ID already exists.
    pub skip_duplicates: bool,
    /// If true, reset imported procedure confidence to 0.5.
    pub reset_confidence: bool,
}

impl Default for ProcedureImportOptions {
    fn default() -> Self {
        Self {
            merge: true,
            skip_duplicates: true,
            reset_confidence: false,
        }
    }
}

/// Result of an import operation.
#[derive(Debug, Clone, Default)]
pub struct ProcedureImportResult {
    /// Number of procedures imported.
    pub imported: usize,
    /// Number of procedures skipped (duplicate ID).
    pub skipped: usize,
    /// Number of procedures replaced (same ID, overwritten).
    pub replaced: usize,
    /// Errors encountered.
    pub errors: Vec<String>,
}

/// Procedure category for organizing defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProcedureCategory {
    Coding,
    Deployment,
    Documentation,
    Review,
    Testing,
    Custom,
}

/// Return a set of sensible default procedures covering common development workflows.
pub fn default_procedures() -> Vec<Procedure> {
    vec![
        Procedure {
            id: "default_code_review".to_string(),
            name: "Code Review Checklist".to_string(),
            condition: "review code changes pull request".to_string(),
            steps: vec![
                "Check for compiler warnings and errors".to_string(),
                "Run the full test suite".to_string(),
                "Review edge cases and error handling".to_string(),
                "Check for security vulnerabilities (injection, XSS, etc.)".to_string(),
                "Verify documentation is updated".to_string(),
            ],
            confidence: 0.8,
            tags: vec!["review".to_string(), "quality".to_string()],
            ..Default::default()
        },
        Procedure {
            id: "default_pre_commit".to_string(),
            name: "Pre-Commit Validation".to_string(),
            condition: "commit save changes push".to_string(),
            steps: vec![
                "Run compile/build check".to_string(),
                "Run linter (clippy, eslint, etc.)".to_string(),
                "Run relevant tests".to_string(),
                "Review diff for accidental changes".to_string(),
            ],
            confidence: 0.8,
            tags: vec!["commit".to_string(), "validation".to_string()],
            ..Default::default()
        },
        Procedure {
            id: "default_deploy".to_string(),
            name: "Deploy Pipeline".to_string(),
            condition: "deploy release production staging".to_string(),
            steps: vec![
                "Run full test suite".to_string(),
                "Build release/optimized binary".to_string(),
                "Run integration tests against staging".to_string(),
                "Deploy to target environment".to_string(),
                "Verify health checks pass".to_string(),
            ],
            confidence: 0.75,
            tags: vec!["deployment".to_string(), "release".to_string()],
            ..Default::default()
        },
        Procedure {
            id: "default_bug_investigation".to_string(),
            name: "Bug Investigation".to_string(),
            condition: "bug fix error crash issue debug".to_string(),
            steps: vec![
                "Reproduce the issue reliably".to_string(),
                "Identify the root cause (not just symptoms)".to_string(),
                "Write a failing test that captures the bug".to_string(),
                "Implement the fix".to_string(),
                "Verify the test passes and no regressions".to_string(),
            ],
            confidence: 0.8,
            tags: vec!["debugging".to_string(), "bugfix".to_string()],
            ..Default::default()
        },
        Procedure {
            id: "default_documentation".to_string(),
            name: "Documentation Update".to_string(),
            condition: "document docs write guide readme".to_string(),
            steps: vec![
                "Identify what changed that needs documentation".to_string(),
                "Update relevant doc files (README, GUIDE, CONCEPTS, etc.)".to_string(),
                "Add code examples where applicable".to_string(),
                "Sync HTML docs if applicable".to_string(),
            ],
            confidence: 0.75,
            tags: vec!["documentation".to_string()],
            ..Default::default()
        },
        Procedure {
            id: "default_test_writing".to_string(),
            name: "Test Writing Methodology".to_string(),
            condition: "test write tests unit integration".to_string(),
            steps: vec![
                "Identify the behavior to test (not the implementation)".to_string(),
                "Write the test first if possible (TDD)".to_string(),
                "Cover happy path, edge cases, and error cases".to_string(),
                "Check boundary conditions (empty, max, overflow)".to_string(),
                "Verify tests are deterministic (no flaky tests)".to_string(),
            ],
            confidence: 0.8,
            tags: vec!["testing".to_string(), "quality".to_string()],
            ..Default::default()
        },
    ]
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
        let proc = self
            .procedures
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| {
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

    /// Remove a procedure by ID. Returns the removed procedure if found.
    pub fn remove(&mut self, id: &str) -> Option<Procedure> {
        if let Some(idx) = self.procedures.iter().position(|p| p.id == id) {
            Some(self.procedures.remove(idx))
        } else {
            None
        }
    }

    /// Find procedures relevant to the given context with quality filtering.
    ///
    /// Unlike `find_by_condition`, this method:
    /// - Requires at least `min_match_ratio` (0.0–1.0) of condition keywords to match
    /// - Filters out procedures below `min_confidence`
    /// - Skips procedures with empty condition or empty steps
    /// - Limits results to `max_results`
    /// - Returns sorted by confidence descending
    pub fn find_relevant(
        &self,
        context: &str,
        min_match_ratio: f64,
        min_confidence: f64,
        max_results: usize,
    ) -> Vec<&Procedure> {
        let ctx_lower = context.to_lowercase();
        let ctx_words: std::collections::HashSet<&str> = ctx_lower.split_whitespace().collect();

        let mut matches: Vec<(f64, &Procedure)> = self
            .procedures
            .iter()
            .filter(|p| {
                // Skip empty conditions or empty steps
                !p.condition.trim().is_empty()
                    && !p.steps.is_empty()
                    && p.confidence >= min_confidence
            })
            .filter_map(|p| {
                let cond_lower = p.condition.to_lowercase();
                let cond_words: Vec<&str> = cond_lower.split_whitespace().collect();
                if cond_words.is_empty() {
                    return None;
                }
                let matching = cond_words.iter().filter(|w| ctx_words.contains(*w)).count();
                let ratio = matching as f64 / cond_words.len() as f64;
                if ratio >= min_match_ratio {
                    Some((p.confidence, p))
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches
            .into_iter()
            .take(max_results)
            .map(|(_, p)| p)
            .collect()
    }

    /// Create a versioned export of all procedures.
    pub fn export(&self, source: &str) -> ProcedureExport {
        ProcedureExport {
            version: 1,
            exported_at: chrono::Utc::now().to_rfc3339(),
            source: source.to_string(),
            user_id: String::new(),
            procedures: self.procedures.clone(),
        }
    }

    /// Export to a JSON file.
    pub fn export_to_file(&self, path: &std::path::Path, source: &str) -> Result<(), String> {
        let export = self.export(source);
        let json =
            serde_json::to_string_pretty(&export).map_err(|e| format!("Serialize error: {}", e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("Rename error: {}", e))?;
        Ok(())
    }

    /// Import procedures from an export, respecting the given options.
    pub fn import(
        &mut self,
        export: &ProcedureExport,
        options: &ProcedureImportOptions,
    ) -> ProcedureImportResult {
        let mut result = ProcedureImportResult::default();

        if !options.merge {
            // Replace mode: clear existing procedures
            let old_count = self.procedures.len();
            self.procedures.clear();
            result.replaced = old_count;
        }

        for mut proc in export.procedures.clone() {
            if options.reset_confidence {
                proc.confidence = 0.5;
                proc.success_count = 0;
                proc.failure_count = 0;
            }

            if options.merge {
                let exists = self.procedures.iter().any(|p| p.id == proc.id);
                if exists && options.skip_duplicates {
                    result.skipped += 1;
                    continue;
                } else if exists {
                    // Replace existing
                    if let Some(idx) = self.procedures.iter().position(|p| p.id == proc.id) {
                        self.procedures[idx] = proc;
                        result.replaced += 1;
                        continue;
                    }
                }
            }

            self.add(proc);
            result.imported += 1;
        }

        result
    }

    /// Import from a JSON file.
    pub fn import_from_file(
        &mut self,
        path: &std::path::Path,
        options: &ProcedureImportOptions,
    ) -> Result<ProcedureImportResult, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let export: ProcedureExport =
            serde_json::from_str(&data).map_err(|e| format!("Deserialize error: {}", e))?;
        Ok(self.import(&export, options))
    }

    /// Load default procedures (skip any whose ID already exists).
    pub fn load_defaults(&mut self) {
        let defaults = default_procedures();
        for proc in defaults {
            let exists = self.procedures.iter().any(|p| p.id == proc.id);
            if !exists {
                self.add(proc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_procedure(
        id: &str,
        name: &str,
        condition: &str,
        steps: Vec<&str>,
        confidence: f64,
    ) -> Procedure {
        Procedure {
            id: id.to_string(),
            name: name.to_string(),
            condition: condition.to_string(),
            steps: steps.into_iter().map(|s| s.to_string()).collect(),
            success_count: (confidence * 10.0) as usize,
            failure_count: ((1.0 - confidence) * 10.0) as usize,
            confidence,
            created_from: Vec::new(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn test_procedural_remove() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "Proc 1",
            "deploy rust",
            vec!["step1"],
            0.9,
        ));
        store.add(make_procedure(
            "p2",
            "Proc 2",
            "test code",
            vec!["step1"],
            0.8,
        ));

        let removed = store.remove("p1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "p1");
        assert_eq!(store.len(), 1);

        // Removing non-existent returns None
        assert!(store.remove("nonexistent").is_none());
    }

    #[test]
    fn test_find_relevant_min_match_ratio() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "Deploy",
            "deploy rust application production",
            vec!["step1"],
            0.9,
        ));
        store.add(make_procedure(
            "p2",
            "Test",
            "run cargo test suite",
            vec!["step1"],
            0.8,
        ));

        // "deploy rust" matches 2/4 = 50% of p1's condition words
        let results = store.find_relevant("deploy rust app", 0.3, 0.0, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");

        // With high ratio threshold, requires more keywords
        let results_strict = store.find_relevant("deploy rust app", 0.9, 0.0, 10);
        assert!(results_strict.is_empty());
    }

    #[test]
    fn test_find_relevant_min_confidence() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "High",
            "deploy code",
            vec!["step"],
            0.9,
        ));
        store.add(make_procedure(
            "p2",
            "Low",
            "deploy code",
            vec!["step"],
            0.05,
        ));

        let results = store.find_relevant("deploy code now", 0.3, 0.1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");
    }

    #[test]
    fn test_find_relevant_empty_condition_skipped() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure("p1", "Empty", "", vec!["step"], 0.9));
        store.add(make_procedure("p2", "Whitespace", "   ", vec!["step"], 0.9));

        let results = store.find_relevant("anything at all", 0.0, 0.0, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_relevant_empty_steps_skipped() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure("p1", "No steps", "deploy code", vec![], 0.9));

        let results = store.find_relevant("deploy code", 0.3, 0.0, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_relevant_max_results() {
        let mut store = ProceduralStore::new(10);
        for i in 0..8 {
            store.add(make_procedure(
                &format!("p{}", i),
                &format!("Proc {}", i),
                "deploy code application",
                vec!["step"],
                0.5 + (i as f64 * 0.05),
            ));
        }

        let results = store.find_relevant("deploy code application now", 0.3, 0.0, 3);
        assert_eq!(results.len(), 3);
        // Should be sorted by confidence descending
        assert!(results[0].confidence >= results[1].confidence);
        assert!(results[1].confidence >= results[2].confidence);
    }

    #[test]
    fn test_procedure_default_trait() {
        let p = Procedure::default();
        assert!(p.id.is_empty());
        assert!((p.confidence - 0.5).abs() < f64::EPSILON);
        assert!(p.steps.is_empty());
    }

    #[test]
    fn test_default_procedures_not_empty() {
        let defaults = default_procedures();
        assert!(
            defaults.len() >= 5,
            "should have at least 5 default procedures"
        );
        for p in &defaults {
            assert!(!p.id.is_empty(), "default procedure must have an ID");
            assert!(!p.name.is_empty(), "default procedure must have a name");
            assert!(!p.steps.is_empty(), "default procedure must have steps");
            assert!(
                !p.condition.is_empty(),
                "default procedure must have a condition"
            );
        }
    }

    #[test]
    fn test_export_import_roundtrip() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "Deploy",
            "deploy rust",
            vec!["test", "build"],
            0.9,
        ));
        store.add(make_procedure(
            "p2",
            "Review",
            "code review",
            vec!["compile", "lint"],
            0.85,
        ));

        let export = store.export("test");
        assert_eq!(export.version, 1);
        assert_eq!(export.procedures.len(), 2);

        let mut store2 = ProceduralStore::new(10);
        let result = store2.import(&export, &ProcedureImportOptions::default());
        assert_eq!(result.imported, 2);
        assert_eq!(store2.len(), 2);
    }

    #[test]
    fn test_import_merge_skip_duplicates() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "Original",
            "deploy rust",
            vec!["step1"],
            0.9,
        ));

        let export = ProcedureExport {
            procedures: vec![
                make_procedure("p1", "Duplicate", "deploy rust", vec!["step2"], 0.5),
                make_procedure("p2", "New", "test code", vec!["step3"], 0.8),
            ],
            ..Default::default()
        };

        let options = ProcedureImportOptions {
            merge: true,
            skip_duplicates: true,
            reset_confidence: false,
        };
        let result = store.import(&export, &options);
        assert_eq!(result.skipped, 1); // p1 skipped
        assert_eq!(result.imported, 1); // p2 imported
        assert_eq!(store.len(), 2);
        // p1 should still have original name
        assert_eq!(store.get("p1").unwrap().name, "Original");
    }

    #[test]
    fn test_import_replace_all() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure("p1", "Old", "old", vec!["step"], 0.9));
        store.add(make_procedure("p2", "Also Old", "old", vec!["step"], 0.8));

        let export = ProcedureExport {
            procedures: vec![make_procedure("p3", "New", "new stuff", vec!["step"], 0.7)],
            ..Default::default()
        };

        let options = ProcedureImportOptions {
            merge: false,
            skip_duplicates: false,
            reset_confidence: false,
        };
        let result = store.import(&export, &options);
        assert_eq!(result.replaced, 2); // old ones cleared
        assert_eq!(result.imported, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("p3").unwrap().name, "New");
    }

    #[test]
    fn test_import_reset_confidence() {
        let mut store = ProceduralStore::new(10);
        let export = ProcedureExport {
            procedures: vec![make_procedure("p1", "High", "deploy", vec!["step"], 0.95)],
            ..Default::default()
        };

        let options = ProcedureImportOptions {
            merge: true,
            skip_duplicates: false,
            reset_confidence: true,
        };
        let result = store.import(&export, &options);
        assert_eq!(result.imported, 1);
        assert!((store.get("p1").unwrap().confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_export_empty_store() {
        let store = ProceduralStore::new(10);
        let export = store.export("empty");
        assert_eq!(export.procedures.len(), 0);
        assert_eq!(export.version, 1);
        assert_eq!(export.source, "empty");
    }

    #[test]
    fn test_load_defaults_no_overwrite() {
        let mut store = ProceduralStore::new(50);
        // Add a custom procedure with same ID as a default
        store.add(Procedure {
            id: "default_code_review".to_string(),
            name: "My Custom Review".to_string(),
            condition: "review".to_string(),
            steps: vec!["my step".to_string()],
            ..Default::default()
        });

        store.load_defaults();

        // Custom version should NOT be overwritten
        let p = store.get("default_code_review").unwrap();
        assert_eq!(p.name, "My Custom Review");
        // But other defaults should be loaded
        assert!(store.get("default_deploy").is_some());
    }

    #[test]
    fn test_export_import_file_roundtrip() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure(
            "p1",
            "Deploy",
            "deploy",
            vec!["step1", "step2"],
            0.9,
        ));

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("procedures_export.json");

        store.export_to_file(&path, "test").expect("export");
        assert!(path.exists());

        let mut store2 = ProceduralStore::new(10);
        let result = store2
            .import_from_file(&path, &ProcedureImportOptions::default())
            .expect("import");
        assert_eq!(result.imported, 1);
        assert_eq!(store2.get("p1").unwrap().name, "Deploy");
    }
}
