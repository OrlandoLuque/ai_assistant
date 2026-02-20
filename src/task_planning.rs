//! # Task Planning Module
//!
//! Provides persistent task plans with steps, sub-steps, statuses, priorities,
//! and progress tracking. This module is for explicit user/system-defined plans,
//! distinct from the `task_decomposition` module which handles automatic AI decomposition.
//!
//! ## Features
//! - Hierarchical plan steps with unlimited nesting
//! - Status tracking (Pending, InProgress, Done, Blocked, Skipped)
//! - Priority levels (Critical, High, Medium, Low, Optional)
//! - Recursive progress calculation
//! - Notes and metadata on steps
//! - Dependency tracking via `blocked_by`
//! - Markdown and JSON export
//! - Builder pattern for fluent construction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a plan step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    InProgress,
    Done,
    Blocked,
    Skipped,
}

/// Priority level of a plan step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepPriority {
    Critical,
    High,
    Medium,
    Low,
    Optional,
}

// ─────────────────────────────────────────────────────────────────────────────
// StepNote
// ─────────────────────────────────────────────────────────────────────────────

/// A note attached to a plan step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepNote {
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub author: Option<String>,
}

impl StepNote {
    /// Create a new note with the given content and the current timestamp.
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            created_at: Utc::now(),
            author: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanStep
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in a task plan. Steps can be nested to form a hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub status: StepStatus,
    pub priority: StepPriority,
    pub children: Vec<PlanStep>,
    pub notes: Vec<StepNote>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub blocked_by: Vec<String>,
    pub assignee: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl PlanStep {
    /// Create a new step with the given title, generating a UUID and defaulting
    /// to Pending status and Medium priority.
    pub fn new(title: &str) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            description: None,
            status: StepStatus::Pending,
            priority: StepPriority::Medium,
            children: Vec::new(),
            notes: Vec::new(),
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
            completed_at: None,
            blocked_by: Vec::new(),
            assignee: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the description (builder pattern).
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self.updated_at = Utc::now();
        self
    }

    /// Set the priority (builder pattern).
    pub fn with_priority(mut self, p: StepPriority) -> Self {
        self.priority = p;
        self.updated_at = Utc::now();
        self
    }

    /// Add a tag (builder pattern).
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self.updated_at = Utc::now();
        self
    }

    /// Add a child step (builder pattern).
    pub fn with_child(mut self, child: PlanStep) -> Self {
        self.children.push(child);
        self.updated_at = Utc::now();
        self
    }

    /// Returns true if this step has no children (is a leaf node).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Calculate progress recursively.
    ///
    /// - Leaf nodes: Done=1.0, InProgress=0.5, Skipped=1.0, Blocked/Pending=0.0
    /// - Parent nodes: average of children's progress
    pub fn progress(&self) -> f32 {
        if self.is_leaf() {
            match self.status {
                StepStatus::Done => 1.0,
                StepStatus::Skipped => 1.0,
                StepStatus::InProgress => 0.5,
                StepStatus::Pending | StepStatus::Blocked => 0.0,
            }
        } else {
            let total: f32 = self.children.iter().map(|c| c.progress()).sum();
            total / self.children.len() as f32
        }
    }

    /// Count total steps recursively (including self).
    pub fn total_steps(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_steps()).sum::<usize>()
    }

    /// Count pending steps recursively (including self if pending).
    pub fn pending_count(&self) -> usize {
        let self_count = if self.status == StepStatus::Pending {
            1
        } else {
            0
        };
        self_count
            + self
                .children
                .iter()
                .map(|c| c.pending_count())
                .sum::<usize>()
    }

    /// Count done steps recursively (including self if done).
    pub fn done_count(&self) -> usize {
        let self_count = if self.status == StepStatus::Done {
            1
        } else {
            0
        };
        self_count + self.children.iter().map(|c| c.done_count()).sum::<usize>()
    }

    /// Recursively search for a step by ID.
    pub fn find_step(&self, id: &str) -> Option<&PlanStep> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_step(id) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively search for a step by ID (mutable reference).
    pub fn find_step_mut(&mut self, id: &str) -> Option<&mut PlanStep> {
        if self.id == id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_step_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Collect all leaf steps with Pending status recursively.
    fn collect_pending_leaves<'a>(&'a self, out: &mut Vec<&'a PlanStep>) {
        if self.is_leaf() && self.status == StepStatus::Pending {
            out.push(self);
        }
        for child in &self.children {
            child.collect_pending_leaves(out);
        }
    }

    /// Collect all steps with Blocked status recursively.
    fn collect_blocked<'a>(&'a self, out: &mut Vec<&'a PlanStep>) {
        if self.status == StepStatus::Blocked {
            out.push(self);
        }
        for child in &self.children {
            child.collect_blocked(out);
        }
    }

    /// Recursively remove a step with the given ID from children.
    /// Returns true if the step was found and removed.
    fn remove_child(&mut self, id: &str) -> bool {
        let len_before = self.children.len();
        self.children.retain(|c| c.id != id);
        if self.children.len() < len_before {
            self.updated_at = Utc::now();
            return true;
        }
        for child in &mut self.children {
            if child.remove_child(id) {
                self.updated_at = Utc::now();
                return true;
            }
        }
        false
    }

    /// Render this step and its children as markdown, with indentation.
    fn to_markdown_lines(&self, depth: usize, lines: &mut Vec<String>) {
        let indent = "  ".repeat(depth);
        let marker = match self.status {
            StepStatus::Done => "[x]",
            StepStatus::InProgress => "[~]",
            StepStatus::Blocked => "[!]",
            StepStatus::Pending | StepStatus::Skipped => "[ ]",
        };
        lines.push(format!("{}- {} {}", indent, marker, self.title));

        for note in &self.notes {
            lines.push(format!("{}  > {}", indent, note.content));
        }

        for child in &self.children {
            child.to_markdown_lines(depth + 1, lines);
        }
    }

    /// Find the first pending leaf whose blocked_by list is empty or all
    /// referenced steps are Done (resolved in the context of a full plan).
    fn first_actionable_leaf<'a>(&'a self, plan_steps: &[PlanStep]) -> Option<&'a PlanStep> {
        if self.is_leaf() && self.status == StepStatus::Pending {
            let all_resolved = self.blocked_by.iter().all(|blocker_id| {
                find_in_steps(plan_steps, blocker_id)
                    .map(|s| s.status == StepStatus::Done)
                    .unwrap_or(true)
            });
            if all_resolved {
                return Some(self);
            }
        }
        for child in &self.children {
            if let Some(found) = child.first_actionable_leaf(plan_steps) {
                return Some(found);
            }
        }
        None
    }
}

/// Helper: find a step by ID across a slice of top-level steps.
fn find_in_steps<'a>(steps: &'a [PlanStep], id: &str) -> Option<&'a PlanStep> {
    for step in steps {
        if let Some(found) = step.find_step(id) {
            return Some(found);
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanSummary
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a task plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub total_steps: usize,
    pub done: usize,
    pub in_progress: usize,
    pub pending: usize,
    pub blocked: usize,
    pub skipped: usize,
    pub progress_percent: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// TaskPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A complete task plan consisting of hierarchical steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPlan {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub steps: Vec<PlanStep>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl TaskPlan {
    /// Create a new empty plan with the given name and a generated UUID.
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: None,
            steps: Vec::new(),
            created_at: now,
            updated_at: now,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the description (builder pattern).
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self.updated_at = Utc::now();
        self
    }

    /// Add a top-level step to the plan.
    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
        self.updated_at = Utc::now();
    }

    /// Add a step as a child of the step with the given parent_id.
    /// Returns true if the parent was found and the child was added.
    pub fn add_step_to(&mut self, parent_id: &str, step: PlanStep) -> bool {
        if let Some(parent) = self.find_step_mut(parent_id) {
            parent.children.push(step);
            parent.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Remove the step with the given ID from the plan (recursive).
    /// Returns true if the step was found and removed.
    pub fn remove_step(&mut self, id: &str) -> bool {
        let len_before = self.steps.len();
        self.steps.retain(|s| s.id != id);
        if self.steps.len() < len_before {
            self.updated_at = Utc::now();
            return true;
        }
        for step in &mut self.steps {
            if step.remove_child(id) {
                self.updated_at = Utc::now();
                return true;
            }
        }
        false
    }

    /// Mark a step as Done and set its completed_at timestamp.
    pub fn complete_step(&mut self, id: &str) -> bool {
        if let Some(step) = self.find_step_mut(id) {
            step.status = StepStatus::Done;
            step.completed_at = Some(Utc::now());
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Mark a step as InProgress.
    pub fn start_step(&mut self, id: &str) -> bool {
        if let Some(step) = self.find_step_mut(id) {
            step.status = StepStatus::InProgress;
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Mark a step as Blocked and add the blocker ID to blocked_by.
    pub fn block_step(&mut self, id: &str, blocked_by: &str) -> bool {
        if let Some(step) = self.find_step_mut(id) {
            step.status = StepStatus::Blocked;
            if !step.blocked_by.contains(&blocked_by.to_string()) {
                step.blocked_by.push(blocked_by.to_string());
            }
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Mark a step as Skipped.
    pub fn skip_step(&mut self, id: &str) -> bool {
        if let Some(step) = self.find_step_mut(id) {
            step.status = StepStatus::Skipped;
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Reset a step to Pending, clearing its completed_at.
    pub fn reset_step(&mut self, id: &str) -> bool {
        if let Some(step) = self.find_step_mut(id) {
            step.status = StepStatus::Pending;
            step.completed_at = None;
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Add a note to the step with the given ID.
    pub fn add_note(&mut self, step_id: &str, note: StepNote) -> bool {
        if let Some(step) = self.find_step_mut(step_id) {
            step.notes.push(note);
            step.updated_at = Utc::now();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Recursively find a step by ID across the entire plan.
    pub fn find_step(&self, id: &str) -> Option<&PlanStep> {
        find_in_steps(&self.steps, id)
    }

    /// Recursively find a step by ID across the entire plan (mutable).
    pub fn find_step_mut(&mut self, id: &str) -> Option<&mut PlanStep> {
        for step in &mut self.steps {
            if let Some(found) = step.find_step_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Calculate overall plan progress as average of top-level step progress.
    pub fn progress(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: f32 = self.steps.iter().map(|s| s.progress()).sum();
        total / self.steps.len() as f32
    }

    /// Total number of steps (all levels) in the plan.
    pub fn total_steps(&self) -> usize {
        self.steps.iter().map(|s| s.total_steps()).sum()
    }

    /// Collect all pending leaf steps in the plan.
    pub fn pending_steps(&self) -> Vec<&PlanStep> {
        let mut result = Vec::new();
        for step in &self.steps {
            step.collect_pending_leaves(&mut result);
        }
        result
    }

    /// Collect all blocked steps in the plan.
    pub fn blocked_steps(&self) -> Vec<&PlanStep> {
        let mut result = Vec::new();
        for step in &self.steps {
            step.collect_blocked(&mut result);
        }
        result
    }

    /// Find the first actionable step: a pending leaf whose blocked_by
    /// dependencies are all Done (or empty).
    pub fn next_actionable(&self) -> Option<&PlanStep> {
        for step in &self.steps {
            if let Some(found) = step.first_actionable_leaf(&self.steps) {
                return Some(found);
            }
        }
        None
    }

    /// Returns true if all steps are Done or Skipped.
    pub fn is_complete(&self) -> bool {
        self.all_steps_complete(&self.steps)
    }

    /// Recursive helper to check if all steps are Done or Skipped.
    fn all_steps_complete(&self, steps: &[PlanStep]) -> bool {
        steps.iter().all(|s| {
            let status_ok = s.status == StepStatus::Done || s.status == StepStatus::Skipped;
            status_ok && self.all_steps_complete(&s.children)
        })
    }

    /// Generate a summary of the plan's current state.
    pub fn summary(&self) -> PlanSummary {
        let mut done = 0;
        let mut in_progress = 0;
        let mut pending = 0;
        let mut blocked = 0;
        let mut skipped = 0;
        self.count_statuses(
            &self.steps,
            &mut done,
            &mut in_progress,
            &mut pending,
            &mut blocked,
            &mut skipped,
        );
        let total_steps = done + in_progress + pending + blocked + skipped;
        let progress_percent = if total_steps > 0 {
            self.progress() * 100.0
        } else {
            0.0
        };
        PlanSummary {
            total_steps,
            done,
            in_progress,
            pending,
            blocked,
            skipped,
            progress_percent,
        }
    }

    /// Recursively count step statuses.
    fn count_statuses(
        &self,
        steps: &[PlanStep],
        done: &mut usize,
        in_progress: &mut usize,
        pending: &mut usize,
        blocked: &mut usize,
        skipped: &mut usize,
    ) {
        for step in steps {
            match step.status {
                StepStatus::Done => *done += 1,
                StepStatus::InProgress => *in_progress += 1,
                StepStatus::Pending => *pending += 1,
                StepStatus::Blocked => *blocked += 1,
                StepStatus::Skipped => *skipped += 1,
            }
            self.count_statuses(&step.children, done, in_progress, pending, blocked, skipped);
        }
    }

    /// Serialize the plan to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize a plan from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let plan: TaskPlan = serde_json::from_str(json)?;
        Ok(plan)
    }

    /// Render the plan as a markdown checklist with indentation.
    pub fn to_markdown(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("# Plan: {}", self.name));
        lines.push(String::new());

        if let Some(ref desc) = self.description {
            lines.push(desc.clone());
            lines.push(String::new());
        }

        for step in &self.steps {
            step.to_markdown_lines(0, &mut lines);
        }

        lines.push(String::new());
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for fluent construction of a `TaskPlan`.
#[derive(Debug, Clone)]
pub struct PlanBuilder {
    name: String,
    description: Option<String>,
    steps: Vec<PlanStep>,
    tags: Vec<String>,
    metadata: HashMap<String, String>,
}

impl PlanBuilder {
    /// Create a new builder with the given plan name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            steps: Vec::new(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the plan description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a step to the plan.
    pub fn step(mut self, step: PlanStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add a tag to the plan.
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Build the final TaskPlan.
    pub fn build(self) -> TaskPlan {
        let now = Utc::now();
        TaskPlan {
            id: Uuid::new_v4().to_string(),
            name: self.name,
            description: self.description,
            steps: self.steps,
            created_at: now,
            updated_at: now,
            tags: self.tags,
            metadata: self.metadata,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_creation_and_defaults() {
        let step = PlanStep::new("Setup database");
        assert_eq!(step.title, "Setup database");
        assert_eq!(step.status, StepStatus::Pending);
        assert_eq!(step.priority, StepPriority::Medium);
        assert!(step.is_leaf());
        assert!(step.description.is_none());
        assert!(step.blocked_by.is_empty());
        assert!(!step.id.is_empty());
    }

    #[test]
    fn test_step_builder_methods() {
        let step = PlanStep::new("Deploy")
            .with_description("Deploy to production")
            .with_priority(StepPriority::Critical)
            .with_tag("ops")
            .with_tag("release");

        assert_eq!(step.description.as_deref(), Some("Deploy to production"));
        assert_eq!(step.priority, StepPriority::Critical);
        assert_eq!(step.tags, vec!["ops", "release"]);
    }

    #[test]
    fn test_progress_leaf_steps() {
        let mut step_done = PlanStep::new("Done step");
        step_done.status = StepStatus::Done;
        assert_eq!(step_done.progress(), 1.0);

        let mut step_in_progress = PlanStep::new("WIP step");
        step_in_progress.status = StepStatus::InProgress;
        assert_eq!(step_in_progress.progress(), 0.5);

        let step_pending = PlanStep::new("Pending step");
        assert_eq!(step_pending.progress(), 0.0);

        let mut step_skipped = PlanStep::new("Skipped step");
        step_skipped.status = StepStatus::Skipped;
        assert_eq!(step_skipped.progress(), 1.0);

        let mut step_blocked = PlanStep::new("Blocked step");
        step_blocked.status = StepStatus::Blocked;
        assert_eq!(step_blocked.progress(), 0.0);
    }

    #[test]
    fn test_progress_parent_step() {
        let mut child1 = PlanStep::new("Child 1");
        child1.status = StepStatus::Done;

        let child2 = PlanStep::new("Child 2"); // Pending

        let parent = PlanStep::new("Parent")
            .with_child(child1)
            .with_child(child2);

        // (1.0 + 0.0) / 2 = 0.5
        assert_eq!(parent.progress(), 0.5);
    }

    #[test]
    fn test_total_steps_and_counts() {
        let child1 = PlanStep::new("Child 1");
        let child2 = PlanStep::new("Child 2");
        let parent = PlanStep::new("Parent")
            .with_child(child1)
            .with_child(child2);

        assert_eq!(parent.total_steps(), 3);
        assert_eq!(parent.pending_count(), 3); // all pending
        assert_eq!(parent.done_count(), 0);
    }

    #[test]
    fn test_find_step_recursive() {
        let grandchild = PlanStep::new("Grandchild");
        let grandchild_id = grandchild.id.clone();
        let child = PlanStep::new("Child").with_child(grandchild);
        let parent = PlanStep::new("Parent").with_child(child);

        let found = parent.find_step(&grandchild_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().title, "Grandchild");

        assert!(parent.find_step("nonexistent").is_none());
    }

    #[test]
    fn test_task_plan_add_and_remove() {
        let mut plan = TaskPlan::new("Release Plan");
        let step1 = PlanStep::new("Build");
        let step1_id = step1.id.clone();
        let step2 = PlanStep::new("Test");
        let step2_id = step2.id.clone();

        plan.add_step(step1);
        plan.add_step(step2);
        assert_eq!(plan.total_steps(), 2);

        assert!(plan.remove_step(&step1_id));
        assert_eq!(plan.total_steps(), 1);
        assert!(plan.find_step(&step2_id).is_some());
        assert!(plan.find_step(&step1_id).is_none());
    }

    #[test]
    fn test_task_plan_step_status_transitions() {
        let mut plan = TaskPlan::new("Workflow");
        let step = PlanStep::new("Task A");
        let id = step.id.clone();
        plan.add_step(step);

        assert!(plan.start_step(&id));
        assert_eq!(plan.find_step(&id).unwrap().status, StepStatus::InProgress);

        assert!(plan.complete_step(&id));
        assert_eq!(plan.find_step(&id).unwrap().status, StepStatus::Done);
        assert!(plan.find_step(&id).unwrap().completed_at.is_some());

        assert!(plan.reset_step(&id));
        assert_eq!(plan.find_step(&id).unwrap().status, StepStatus::Pending);
        assert!(plan.find_step(&id).unwrap().completed_at.is_none());

        assert!(plan.skip_step(&id));
        assert_eq!(plan.find_step(&id).unwrap().status, StepStatus::Skipped);
    }

    #[test]
    fn test_task_plan_block_step() {
        let mut plan = TaskPlan::new("Pipeline");
        let blocker = PlanStep::new("Prerequisite");
        let blocker_id = blocker.id.clone();
        let dependent = PlanStep::new("Dependent Task");
        let dep_id = dependent.id.clone();

        plan.add_step(blocker);
        plan.add_step(dependent);

        assert!(plan.block_step(&dep_id, &blocker_id));
        let step = plan.find_step(&dep_id).unwrap();
        assert_eq!(step.status, StepStatus::Blocked);
        assert!(step.blocked_by.contains(&blocker_id));
    }

    #[test]
    fn test_task_plan_add_step_to_parent() {
        let mut plan = TaskPlan::new("Nested Plan");
        let parent = PlanStep::new("Phase 1");
        let parent_id = parent.id.clone();
        plan.add_step(parent);

        let child = PlanStep::new("Sub-task 1");
        let child_id = child.id.clone();
        assert!(plan.add_step_to(&parent_id, child));

        let found_parent = plan.find_step(&parent_id).unwrap();
        assert_eq!(found_parent.children.len(), 1);
        assert!(plan.find_step(&child_id).is_some());

        // Try adding to non-existent parent
        let orphan = PlanStep::new("Orphan");
        assert!(!plan.add_step_to("bad-id", orphan));
    }

    #[test]
    fn test_plan_progress_and_completion() {
        let mut plan = TaskPlan::new("Progress Test");

        let mut s1 = PlanStep::new("Step 1");
        s1.status = StepStatus::Done;
        let mut s2 = PlanStep::new("Step 2");
        s2.status = StepStatus::Done;

        plan.add_step(s1);
        plan.add_step(s2);

        assert_eq!(plan.progress(), 1.0);
        assert!(plan.is_complete());

        // Add a pending step
        plan.add_step(PlanStep::new("Step 3"));
        assert!(!plan.is_complete());
        // (1.0 + 1.0 + 0.0) / 3
        let expected = 2.0 / 3.0;
        assert!((plan.progress() - expected).abs() < 0.001);
    }

    #[test]
    fn test_next_actionable() {
        let mut plan = TaskPlan::new("Actionable Test");

        let blocker = PlanStep::new("Blocker");
        let blocker_id = blocker.id.clone();
        plan.add_step(blocker);

        let mut blocked_step = PlanStep::new("Blocked");
        blocked_step.blocked_by.push(blocker_id.clone());
        plan.add_step(blocked_step);

        let free_step = PlanStep::new("Free Step");
        let free_id = free_step.id.clone();
        plan.add_step(free_step);

        // The first actionable should be the blocker (pending, no blockers)
        let actionable = plan.next_actionable().unwrap();
        assert_eq!(actionable.id, blocker_id);

        // Complete the blocker
        plan.complete_step(&blocker_id);

        // Now the "Blocked" step becomes actionable (its blocker is Done)
        let actionable = plan.next_actionable().unwrap();
        assert_eq!(actionable.title, "Blocked");

        // Complete all steps
        plan.complete_step(&actionable.id.clone());
        plan.complete_step(&free_id);
        assert!(plan.next_actionable().is_none());
    }

    #[test]
    fn test_pending_and_blocked_steps() {
        let mut plan = TaskPlan::new("Lists Test");

        let mut s1 = PlanStep::new("Done");
        s1.status = StepStatus::Done;
        let s2 = PlanStep::new("Pending 1");
        let s3 = PlanStep::new("Pending 2");
        let mut s4 = PlanStep::new("Blocked");
        s4.status = StepStatus::Blocked;

        plan.add_step(s1);
        plan.add_step(s2);
        plan.add_step(s3);
        plan.add_step(s4);

        let pending = plan.pending_steps();
        assert_eq!(pending.len(), 2);

        let blocked = plan.blocked_steps();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].title, "Blocked");
    }

    #[test]
    fn test_summary() {
        let mut plan = TaskPlan::new("Summary Test");

        let mut s1 = PlanStep::new("Done");
        s1.status = StepStatus::Done;
        let mut s2 = PlanStep::new("WIP");
        s2.status = StepStatus::InProgress;
        let s3 = PlanStep::new("Pending");
        let mut s4 = PlanStep::new("Blocked");
        s4.status = StepStatus::Blocked;
        let mut s5 = PlanStep::new("Skipped");
        s5.status = StepStatus::Skipped;

        plan.add_step(s1);
        plan.add_step(s2);
        plan.add_step(s3);
        plan.add_step(s4);
        plan.add_step(s5);

        let summary = plan.summary();
        assert_eq!(summary.total_steps, 5);
        assert_eq!(summary.done, 1);
        assert_eq!(summary.in_progress, 1);
        assert_eq!(summary.pending, 1);
        assert_eq!(summary.blocked, 1);
        assert_eq!(summary.skipped, 1);
        // progress: (1.0 + 0.5 + 0.0 + 0.0 + 1.0) / 5 * 100 = 50.0
        assert!((summary.progress_percent - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_json_serialization() {
        let plan = PlanBuilder::new("JSON Test")
            .description("Testing serialization")
            .step(PlanStep::new("Step 1"))
            .step(PlanStep::new("Step 2"))
            .tag("test")
            .build();

        let json = plan.to_json();
        assert!(json.contains("JSON Test"));
        assert!(json.contains("Step 1"));
        assert!(json.contains("Step 2"));

        let restored = TaskPlan::from_json(&json).unwrap();
        assert_eq!(restored.name, "JSON Test");
        assert_eq!(restored.steps.len(), 2);
        assert_eq!(restored.tags, vec!["test"]);
        assert_eq!(
            restored.description.as_deref(),
            Some("Testing serialization")
        );
    }

    #[test]
    fn test_to_markdown() {
        let mut plan = TaskPlan::new("Markdown Plan").with_description("A test plan");

        let mut done_step = PlanStep::new("Completed Task");
        done_step.status = StepStatus::Done;
        let mut wip_step = PlanStep::new("Working On");
        wip_step.status = StepStatus::InProgress;
        let pending_step = PlanStep::new("Todo");
        let mut blocked_step = PlanStep::new("Blocked Task");
        blocked_step.status = StepStatus::Blocked;

        let child = PlanStep::new("Sub-task");
        let parent = PlanStep::new("Parent").with_child(child);

        plan.add_step(done_step);
        plan.add_step(wip_step);
        plan.add_step(pending_step);
        plan.add_step(blocked_step);
        plan.add_step(parent);

        let md = plan.to_markdown();
        assert!(md.contains("# Plan: Markdown Plan"));
        assert!(md.contains("A test plan"));
        assert!(md.contains("- [x] Completed Task"));
        assert!(md.contains("- [~] Working On"));
        assert!(md.contains("- [ ] Todo"));
        assert!(md.contains("- [!] Blocked Task"));
        assert!(md.contains("  - [ ] Sub-task"));
    }

    #[test]
    fn test_add_note() {
        let mut plan = TaskPlan::new("Notes Test");
        let step = PlanStep::new("Noted step");
        let step_id = step.id.clone();
        plan.add_step(step);

        let note = StepNote::new("This is important");
        assert!(plan.add_note(&step_id, note));

        let found = plan.find_step(&step_id).unwrap();
        assert_eq!(found.notes.len(), 1);
        assert_eq!(found.notes[0].content, "This is important");
        assert!(found.notes[0].author.is_none());

        // Adding note to nonexistent step returns false
        assert!(!plan.add_note("bad-id", StepNote::new("Orphan note")));
    }

    #[test]
    fn test_plan_builder() {
        let plan = PlanBuilder::new("Builder Test")
            .description("Built with builder")
            .step(PlanStep::new("Alpha"))
            .step(PlanStep::new("Beta").with_priority(StepPriority::High))
            .tag("v1")
            .tag("sprint")
            .build();

        assert_eq!(plan.name, "Builder Test");
        assert_eq!(plan.description.as_deref(), Some("Built with builder"));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].title, "Alpha");
        assert_eq!(plan.steps[1].priority, StepPriority::High);
        assert_eq!(plan.tags, vec!["v1", "sprint"]);
        assert!(!plan.id.is_empty());
    }

    #[test]
    fn test_is_complete_with_skipped() {
        let mut plan = TaskPlan::new("Complete Check");
        let mut s1 = PlanStep::new("Done");
        s1.status = StepStatus::Done;
        let mut s2 = PlanStep::new("Skipped");
        s2.status = StepStatus::Skipped;
        plan.add_step(s1);
        plan.add_step(s2);

        assert!(plan.is_complete());
    }

    #[test]
    fn test_remove_nested_step() {
        let mut plan = TaskPlan::new("Remove Nested");
        let grandchild = PlanStep::new("Grandchild");
        let gc_id = grandchild.id.clone();
        let child = PlanStep::new("Child").with_child(grandchild);
        let parent = PlanStep::new("Parent").with_child(child);
        plan.add_step(parent);

        assert_eq!(plan.total_steps(), 3);
        assert!(plan.remove_step(&gc_id));
        assert_eq!(plan.total_steps(), 2);
        assert!(plan.find_step(&gc_id).is_none());
    }

    #[test]
    fn test_empty_plan_progress() {
        let plan = TaskPlan::new("Empty");
        assert_eq!(plan.progress(), 0.0);
        assert!(plan.is_complete()); // vacuously true
        assert!(plan.next_actionable().is_none());
        assert!(plan.pending_steps().is_empty());
    }

    #[test]
    fn test_step_note_creation() {
        let note = StepNote::new("Remember to check logs");
        assert_eq!(note.content, "Remember to check logs");
        assert!(note.author.is_none());
        // created_at should be very recent
        let now = Utc::now();
        let diff = now - note.created_at;
        assert!(diff.num_seconds() < 2);
    }

    #[test]
    fn test_deeply_nested_progress() {
        // Root -> A (Done), B -> B1 (Done), B2 (Pending)
        let mut a = PlanStep::new("A");
        a.status = StepStatus::Done;

        let mut b1 = PlanStep::new("B1");
        b1.status = StepStatus::Done;
        let b2 = PlanStep::new("B2"); // Pending

        let b = PlanStep::new("B").with_child(b1).with_child(b2);

        let mut plan = TaskPlan::new("Deep Progress");
        plan.add_step(a);
        plan.add_step(b);

        // A progress = 1.0, B progress = (1.0 + 0.0) / 2 = 0.5
        // Plan progress = (1.0 + 0.5) / 2 = 0.75
        assert!((plan.progress() - 0.75).abs() < 0.001);
    }
}
