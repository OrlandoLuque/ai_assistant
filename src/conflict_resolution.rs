//! Conflict resolution
//!
//! Resolve conflicts in concurrent modifications.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Conflict type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Both sides modified same content
    ConcurrentModification,
    /// One side deleted, other modified
    DeleteModify,
    /// Both sides added at same position
    DuplicateAdd,
    /// Version mismatch
    VersionConflict,
}

/// Conflict record
#[derive(Debug, Clone)]
pub struct Conflict {
    pub id: String,
    pub conflict_type: ConflictType,
    pub entity_type: String,
    pub entity_id: String,
    pub local_value: Option<String>,
    pub remote_value: Option<String>,
    pub local_version: u64,
    pub remote_version: u64,
    pub detected_at: u64,
    pub resolved: bool,
    pub resolution: Option<Resolution>,
}

impl Conflict {
    pub fn new(
        conflict_type: ConflictType,
        entity_type: &str,
        entity_id: &str,
        local_value: Option<String>,
        remote_value: Option<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            conflict_type,
            entity_type: entity_type.to_string(),
            entity_id: entity_id.to_string(),
            local_value,
            remote_value,
            local_version: 0,
            remote_version: 0,
            detected_at: now,
            resolved: false,
            resolution: None,
        }
    }

    pub fn with_versions(mut self, local: u64, remote: u64) -> Self {
        self.local_version = local;
        self.remote_version = remote;
        self
    }
}

/// Resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// Keep local version
    KeepLocal,
    /// Keep remote version
    KeepRemote,
    /// Keep newest (by timestamp/version)
    KeepNewest,
    /// Keep oldest
    KeepOldest,
    /// Merge both
    Merge,
    /// Manual resolution required
    Manual,
    /// Keep both as separate entries
    KeepBoth,
}

/// Resolution result
#[derive(Debug, Clone)]
pub struct Resolution {
    pub strategy: ResolutionStrategy,
    pub resolved_value: Option<String>,
    pub resolved_at: u64,
    pub resolved_by: String,
}

impl Resolution {
    pub fn new(strategy: ResolutionStrategy, value: Option<String>, resolved_by: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            strategy,
            resolved_value: value,
            resolved_at: now,
            resolved_by: resolved_by.to_string(),
        }
    }
}

/// Conflict resolver
pub struct ConflictResolver {
    default_strategy: ResolutionStrategy,
    type_strategies: HashMap<String, ResolutionStrategy>,
    pending_conflicts: Vec<Conflict>,
    resolved_conflicts: Vec<Conflict>,
    merge_handler: Option<Box<dyn Fn(&str, &str) -> String + Send + Sync>>,
}

impl ConflictResolver {
    pub fn new(default_strategy: ResolutionStrategy) -> Self {
        Self {
            default_strategy,
            type_strategies: HashMap::new(),
            pending_conflicts: Vec::new(),
            resolved_conflicts: Vec::new(),
            merge_handler: None,
        }
    }

    pub fn set_strategy_for_type(&mut self, entity_type: &str, strategy: ResolutionStrategy) {
        self.type_strategies
            .insert(entity_type.to_string(), strategy);
    }

    pub fn set_merge_handler<F>(&mut self, handler: F)
    where
        F: Fn(&str, &str) -> String + Send + Sync + 'static,
    {
        self.merge_handler = Some(Box::new(handler));
    }

    pub fn detect_conflict(
        &mut self,
        entity_type: &str,
        entity_id: &str,
        local_value: Option<&str>,
        remote_value: Option<&str>,
        local_version: u64,
        remote_version: u64,
    ) -> Option<Conflict> {
        let conflict_type = match (local_value, remote_value) {
            (Some(_), Some(_)) if local_version != remote_version => {
                ConflictType::ConcurrentModification
            }
            (None, Some(_)) | (Some(_), None) => ConflictType::DeleteModify,
            _ => return None,
        };

        let conflict = Conflict::new(
            conflict_type,
            entity_type,
            entity_id,
            local_value.map(|s| s.to_string()),
            remote_value.map(|s| s.to_string()),
        )
        .with_versions(local_version, remote_version);

        self.pending_conflicts.push(conflict.clone());
        Some(conflict)
    }

    pub fn resolve(&mut self, conflict_id: &str) -> Result<Resolution, ConflictError> {
        let idx = self
            .pending_conflicts
            .iter()
            .position(|c| c.id == conflict_id)
            .ok_or(ConflictError::NotFound)?;

        let conflict = &self.pending_conflicts[idx];

        let strategy = self
            .type_strategies
            .get(&conflict.entity_type)
            .copied()
            .unwrap_or(self.default_strategy);

        let resolution = self.apply_strategy(conflict, strategy)?;

        let mut resolved = self.pending_conflicts.remove(idx);
        resolved.resolved = true;
        resolved.resolution = Some(resolution.clone());
        self.resolved_conflicts.push(resolved);

        Ok(resolution)
    }

    pub fn resolve_with_strategy(
        &mut self,
        conflict_id: &str,
        strategy: ResolutionStrategy,
    ) -> Result<Resolution, ConflictError> {
        let idx = self
            .pending_conflicts
            .iter()
            .position(|c| c.id == conflict_id)
            .ok_or(ConflictError::NotFound)?;

        let conflict = &self.pending_conflicts[idx];
        let resolution = self.apply_strategy(conflict, strategy)?;

        let mut resolved = self.pending_conflicts.remove(idx);
        resolved.resolved = true;
        resolved.resolution = Some(resolution.clone());
        self.resolved_conflicts.push(resolved);

        Ok(resolution)
    }

    pub fn resolve_manual(
        &mut self,
        conflict_id: &str,
        value: &str,
        resolved_by: &str,
    ) -> Result<Resolution, ConflictError> {
        let idx = self
            .pending_conflicts
            .iter()
            .position(|c| c.id == conflict_id)
            .ok_or(ConflictError::NotFound)?;

        let resolution = Resolution::new(
            ResolutionStrategy::Manual,
            Some(value.to_string()),
            resolved_by,
        );

        let mut resolved = self.pending_conflicts.remove(idx);
        resolved.resolved = true;
        resolved.resolution = Some(resolution.clone());
        self.resolved_conflicts.push(resolved);

        Ok(resolution)
    }

    fn apply_strategy(
        &self,
        conflict: &Conflict,
        strategy: ResolutionStrategy,
    ) -> Result<Resolution, ConflictError> {
        let resolved_value = match strategy {
            ResolutionStrategy::KeepLocal => conflict.local_value.clone(),
            ResolutionStrategy::KeepRemote => conflict.remote_value.clone(),
            ResolutionStrategy::KeepNewest => {
                if conflict.local_version >= conflict.remote_version {
                    conflict.local_value.clone()
                } else {
                    conflict.remote_value.clone()
                }
            }
            ResolutionStrategy::KeepOldest => {
                if conflict.local_version <= conflict.remote_version {
                    conflict.local_value.clone()
                } else {
                    conflict.remote_value.clone()
                }
            }
            ResolutionStrategy::Merge => {
                match (&conflict.local_value, &conflict.remote_value) {
                    (Some(local), Some(remote)) => {
                        if let Some(ref handler) = self.merge_handler {
                            Some(handler(local, remote))
                        } else {
                            // Default merge: concatenate
                            Some(format!("{}\n---\n{}", local, remote))
                        }
                    }
                    (Some(v), None) | (None, Some(v)) => Some(v.clone()),
                    (None, None) => None,
                }
            }
            ResolutionStrategy::Manual => {
                return Err(ConflictError::ManualResolutionRequired);
            }
            ResolutionStrategy::KeepBoth => {
                // This would need special handling at the application level
                conflict.local_value.clone()
            }
        };

        Ok(Resolution::new(strategy, resolved_value, "auto"))
    }

    pub fn auto_resolve_all(&mut self) -> Vec<Result<Resolution, ConflictError>> {
        let conflict_ids: Vec<_> = self
            .pending_conflicts
            .iter()
            .map(|c| c.id.clone())
            .collect();

        conflict_ids
            .into_iter()
            .map(|id| self.resolve(&id))
            .collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_conflicts.len()
    }

    pub fn get_pending(&self) -> &[Conflict] {
        &self.pending_conflicts
    }

    pub fn get_resolved(&self) -> &[Conflict] {
        &self.resolved_conflicts
    }

    pub fn clear_resolved(&mut self) {
        self.resolved_conflicts.clear();
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new(ResolutionStrategy::KeepNewest)
    }
}

/// Conflict errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictError {
    NotFound,
    AlreadyResolved,
    ManualResolutionRequired,
    MergeFailure,
}

impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Conflict not found"),
            Self::AlreadyResolved => write!(f, "Conflict already resolved"),
            Self::ManualResolutionRequired => write!(f, "Manual resolution required"),
            Self::MergeFailure => write!(f, "Merge failed"),
        }
    }
}

impl std::error::Error for ConflictError {}

/// Three-way merge helper
pub struct ThreeWayMerge;

impl ThreeWayMerge {
    pub fn merge(base: &str, local: &str, remote: &str) -> MergeResult {
        // Simple line-based merge
        let base_lines: Vec<_> = base.lines().collect();
        let local_lines: Vec<_> = local.lines().collect();
        let remote_lines: Vec<_> = remote.lines().collect();

        let mut result = Vec::new();
        let mut conflicts = Vec::new();

        let max_len = base_lines
            .len()
            .max(local_lines.len())
            .max(remote_lines.len());

        for i in 0..max_len {
            let base_line = base_lines.get(i).copied();
            let local_line = local_lines.get(i).copied();
            let remote_line = remote_lines.get(i).copied();

            match (base_line, local_line, remote_line) {
                (_, Some(l), Some(r)) if l == r => {
                    result.push(l.to_string());
                }
                (Some(b), Some(l), Some(r)) if l == b => {
                    result.push(r.to_string());
                }
                (Some(b), Some(l), Some(r)) if r == b => {
                    result.push(l.to_string());
                }
                (_, Some(l), Some(r)) => {
                    conflicts.push(MergeConflictLine {
                        line: i,
                        local: l.to_string(),
                        remote: r.to_string(),
                    });
                    result.push(format!(
                        "<<<<<<< LOCAL\n{}\n=======\n{}\n>>>>>>> REMOTE",
                        l, r
                    ));
                }
                (_, Some(l), None) => {
                    result.push(l.to_string());
                }
                (_, None, Some(r)) => {
                    result.push(r.to_string());
                }
                _ => {}
            }
        }

        let has_conflicts = !conflicts.is_empty();
        MergeResult {
            merged: result.join("\n"),
            conflicts,
            has_conflicts,
        }
    }
}

/// Merge result
#[derive(Debug, Clone)]
pub struct MergeResult {
    pub merged: String,
    pub conflicts: Vec<MergeConflictLine>,
    pub has_conflicts: bool,
}

/// Merge conflict at a line
#[derive(Debug, Clone)]
pub struct MergeConflictLine {
    pub line: usize,
    pub local: String,
    pub remote: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_detection() {
        let mut resolver = ConflictResolver::default();

        let conflict = resolver.detect_conflict(
            "message",
            "1",
            Some("local content"),
            Some("remote content"),
            1,
            2,
        );

        assert!(conflict.is_some());
        assert_eq!(resolver.pending_count(), 1);
    }

    #[test]
    fn test_auto_resolve() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::KeepLocal);

        resolver.detect_conflict("message", "1", Some("local"), Some("remote"), 1, 2);

        let results = resolver.auto_resolve_all();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_manual_resolve() {
        let mut resolver = ConflictResolver::default();

        let conflict = resolver
            .detect_conflict("message", "1", Some("local"), Some("remote"), 1, 2)
            .unwrap();

        let resolution = resolver.resolve_manual(&conflict.id, "manual value", "user1");
        assert!(resolution.is_ok());
    }

    #[test]
    fn test_conflict_with_versions() {
        let c = Conflict::new(ConflictType::VersionConflict, "doc", "1", Some("a".into()), Some("b".into()))
            .with_versions(3, 5);
        assert_eq!(c.local_version, 3);
        assert_eq!(c.remote_version, 5);
        assert!(!c.resolved);
    }

    #[test]
    fn test_resolve_keep_newest() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::KeepNewest);
        let c = resolver.detect_conflict("msg", "1", Some("local_v3"), Some("remote_v5"), 3, 5).unwrap();
        let res = resolver.resolve(&c.id).unwrap();
        assert_eq!(res.strategy, ResolutionStrategy::KeepNewest);
        assert_eq!(res.resolved_value.as_deref(), Some("remote_v5"));
    }

    #[test]
    fn test_resolve_with_custom_strategy() {
        let mut resolver = ConflictResolver::default();
        let c = resolver.detect_conflict("msg", "1", Some("A"), Some("B"), 1, 2).unwrap();
        let res = resolver.resolve_with_strategy(&c.id, ResolutionStrategy::KeepLocal).unwrap();
        assert_eq!(res.resolved_value.as_deref(), Some("A"));
    }

    #[test]
    fn test_type_specific_strategy() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::KeepNewest);
        resolver.set_strategy_for_type("config", ResolutionStrategy::KeepRemote);
        let c = resolver.detect_conflict("config", "1", Some("local"), Some("remote"), 1, 2).unwrap();
        let res = resolver.resolve(&c.id).unwrap();
        assert_eq!(res.resolved_value.as_deref(), Some("remote"));
    }

    #[test]
    fn test_pending_and_resolved() {
        let mut resolver = ConflictResolver::default();
        resolver.detect_conflict("msg", "1", Some("a"), Some("b"), 1, 2);
        resolver.detect_conflict("msg", "2", Some("c"), Some("d"), 1, 2);
        assert_eq!(resolver.pending_count(), 2);
        assert_eq!(resolver.get_resolved().len(), 0);
        let id = resolver.get_pending()[0].id.clone();
        resolver.resolve(&id).unwrap();
        assert_eq!(resolver.pending_count(), 1);
        assert_eq!(resolver.get_resolved().len(), 1);
        resolver.clear_resolved();
        assert_eq!(resolver.get_resolved().len(), 0);
    }

    #[test]
    fn test_conflict_error_display() {
        assert_eq!(ConflictError::NotFound.to_string(), "Conflict not found");
        assert_eq!(ConflictError::ManualResolutionRequired.to_string(), "Manual resolution required");
    }

    #[test]
    fn test_three_way_merge() {
        let base = "line1\nline2\nline3";
        let local = "line1\nmodified local\nline3";
        let remote = "line1\nline2\nmodified remote";

        let result = ThreeWayMerge::merge(base, local, remote);
        assert!(!result.has_conflicts);
        assert!(result.merged.contains("modified local"));
        assert!(result.merged.contains("modified remote"));
    }
}
