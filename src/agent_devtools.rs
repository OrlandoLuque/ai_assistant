//! Agent DevTools — debugging, profiling, execution replay for AI agents.
//!
//! Gated on `#[cfg(feature = "devtools")]` in lib.rs.
//!
//! Provides:
//! - **ExecutionRecorder**: records agent execution events for later replay
//! - **ExecutionReplay**: step-by-step replay of recorded executions
//! - **PerformanceProfiler**: collects per-step latency, token, and cost metrics
//! - **StateInspector**: captures and diffs agent state snapshots
//! - **AgentDebugger**: unified facade combining recorder + profiler + inspector + breakpoints

use crate::error::{AiError, DevToolsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// DevToolsConfig
// =============================================================================

/// Configuration for the agent devtools subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevToolsConfig {
    /// Whether the execution recorder should capture events.
    pub enable_recording: bool,
    /// Whether the performance profiler should collect step metrics.
    pub enable_profiling: bool,
    /// Maximum number of steps the recorder will store before silently discarding.
    pub max_recording_steps: usize,
    /// Breakpoints that pause or flag execution on matching events.
    pub breakpoints: Vec<Breakpoint>,
}

impl Default for DevToolsConfig {
    fn default() -> Self {
        Self {
            enable_recording: true,
            enable_profiling: true,
            max_recording_steps: 10000,
            breakpoints: Vec::new(),
        }
    }
}

// =============================================================================
// Breakpoint
// =============================================================================

/// A condition that, when matched against a [`DebugEvent`], triggers the debugger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Breakpoint {
    /// Triggers immediately before a tool call with the given name.
    BeforeToolCall { tool_name: String },
    /// Triggers after a specific step number completes.
    AfterStep { step_number: usize },
    /// Triggers when the confidence score drops below the threshold.
    OnConfidenceBelow { threshold: f64 },
    /// Triggers on any error event.
    OnError,
    /// Triggers when cumulative cost exceeds the threshold.
    OnCostAbove { threshold: f64 },
    /// Triggers when the step count equals the given value.
    AtStepCount { count: usize },
}

impl Breakpoint {
    /// Returns `true` if this breakpoint matches the supplied event.
    pub fn matches(&self, event: &DebugEvent) -> bool {
        match self {
            Breakpoint::BeforeToolCall { tool_name } => {
                event.event_type == DebugEventType::ToolCallStart
                    && event
                        .tool_name
                        .as_ref()
                        .map(|n| n == tool_name)
                        .unwrap_or(false)
            }
            Breakpoint::AfterStep { step_number } => {
                event.event_type == DebugEventType::StepComplete
                    && event.step_number == *step_number
            }
            Breakpoint::OnConfidenceBelow { threshold } => event
                .confidence
                .map(|c| c < *threshold)
                .unwrap_or(false),
            Breakpoint::OnError => event.is_error,
            Breakpoint::OnCostAbove { threshold } => {
                event.cost.map(|c| c > *threshold).unwrap_or(false)
            }
            Breakpoint::AtStepCount { count } => event.step_number == *count,
        }
    }
}

// =============================================================================
// DebugEvent / DebugEventType
// =============================================================================

/// An event emitted by the agent that can be recorded and checked against breakpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugEvent {
    /// The category of this event.
    pub event_type: DebugEventType,
    /// The sequential step number within the agent run.
    pub step_number: usize,
    /// The tool name, if the event is related to a tool call.
    pub tool_name: Option<String>,
    /// The agent's confidence score at this point, if available.
    pub confidence: Option<f64>,
    /// Cumulative cost so far, if tracked.
    pub cost: Option<f64>,
    /// Whether this event represents an error condition.
    pub is_error: bool,
    /// Unix timestamp (seconds since epoch) when the event was created.
    pub timestamp: u64,
    /// Arbitrary key-value data attached to the event.
    pub data: HashMap<String, String>,
}

impl DebugEvent {
    /// Helper to create a new event with the current system timestamp.
    pub fn now(event_type: DebugEventType, step_number: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            event_type,
            step_number,
            tool_name: None,
            confidence: None,
            cost: None,
            is_error: false,
            timestamp,
            data: HashMap::new(),
        }
    }
}

/// Categories of debug events emitted during an agent run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DebugEventType {
    ToolCallStart,
    ToolCallEnd,
    LlmQueryStart,
    LlmQueryEnd,
    PlanningStart,
    PlanningEnd,
    StepComplete,
    Error,
}

// =============================================================================
// ExecutionRecorder
// =============================================================================

/// Records agent execution events for later replay or analysis.
#[derive(Debug)]
pub struct ExecutionRecorder {
    events: Vec<DebugEvent>,
    recording: bool,
    agent_id: String,
    max_events: usize,
    metadata: HashMap<String, String>,
}

impl ExecutionRecorder {
    /// Create a new recorder for the given agent.
    pub fn new(agent_id: impl Into<String>, max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            recording: false,
            agent_id: agent_id.into(),
            max_events,
            metadata: HashMap::new(),
        }
    }

    /// Start recording events.
    pub fn start(&mut self) {
        self.recording = true;
    }

    /// Stop recording events.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Returns `true` if the recorder is currently capturing events.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record a single event. Fails if the recorder is not currently active.
    pub fn record(&mut self, event: DebugEvent) -> Result<(), DevToolsError> {
        if !self.recording {
            return Err(DevToolsError::RecordingFailed {
                agent_id: self.agent_id.clone(),
                reason: "Recorder is not active — call start() first".to_string(),
            });
        }
        if self.events.len() < self.max_events {
            self.events.push(event);
        }
        Ok(())
    }

    /// A read-only slice of all recorded events.
    pub fn events(&self) -> &[DebugEvent] {
        &self.events
    }

    /// The number of events captured so far.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Clear all recorded events (does not change recording state).
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// The agent id this recorder is bound to.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Filter events by type.
    pub fn events_by_type(&self, event_type: &DebugEventType) -> Vec<&DebugEvent> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Filter events whose step number falls within `[start_step, end_step]` (inclusive).
    pub fn events_in_range(&self, start_step: usize, end_step: usize) -> Vec<&DebugEvent> {
        self.events
            .iter()
            .filter(|e| e.step_number >= start_step && e.step_number <= end_step)
            .collect()
    }

    /// Attach arbitrary metadata to the recording.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Serialize the entire recording (events + metadata + agent id) to JSON.
    pub fn to_json(&self) -> Result<String, AiError> {
        let payload = RecordingPayload {
            agent_id: self.agent_id.clone(),
            events: self.events.clone(),
            metadata: self.metadata.clone(),
        };
        serde_json::to_string(&payload).map_err(|e| {
            AiError::DevTools(DevToolsError::RecordingFailed {
                agent_id: self.agent_id.clone(),
                reason: format!("JSON serialization failed: {}", e),
            })
        })
    }

    /// Deserialize a recording from JSON, producing a new `ExecutionRecorder` in the stopped state.
    pub fn from_json(json: &str) -> Result<Self, AiError> {
        let payload: RecordingPayload = serde_json::from_str(json).map_err(|e| {
            AiError::DevTools(DevToolsError::RecordingFailed {
                agent_id: "unknown".to_string(),
                reason: format!("JSON deserialization failed: {}", e),
            })
        })?;
        let max_events = if payload.events.is_empty() {
            10000
        } else {
            payload.events.len().max(10000)
        };
        Ok(Self {
            agent_id: payload.agent_id,
            events: payload.events,
            recording: false,
            max_events,
            metadata: payload.metadata,
        })
    }
}

/// Internal serialization helper for the recorder.
#[derive(Serialize, Deserialize)]
struct RecordingPayload {
    agent_id: String,
    events: Vec<DebugEvent>,
    metadata: HashMap<String, String>,
}

// =============================================================================
// ExecutionReplay
// =============================================================================

/// Replays a recorded execution step by step.
#[derive(Debug)]
pub struct ExecutionReplay {
    events: Vec<DebugEvent>,
    current_index: usize,
    agent_id: String,
}

impl ExecutionReplay {
    /// Create a replay from an existing recorder.
    pub fn new(recorder: &ExecutionRecorder) -> Self {
        Self {
            events: recorder.events.clone(),
            current_index: 0,
            agent_id: recorder.agent_id.clone(),
        }
    }

    /// Create a replay from raw components.
    pub fn from_events(agent_id: String, events: Vec<DebugEvent>) -> Self {
        Self {
            events,
            current_index: 0,
            agent_id,
        }
    }

    /// Advance to the next event and return a reference to it.
    pub fn next(&mut self) -> Option<&DebugEvent> {
        if self.current_index < self.events.len() {
            let idx = self.current_index;
            self.current_index += 1;
            Some(&self.events[idx])
        } else {
            None
        }
    }

    /// Peek at the next event without advancing.
    pub fn peek(&self) -> Option<&DebugEvent> {
        if self.current_index < self.events.len() {
            Some(&self.events[self.current_index])
        } else {
            None
        }
    }

    /// The event at the current replay position (the one most recently returned by `next`).
    /// Returns `None` if `next` has not been called yet or the replay is exhausted.
    pub fn current(&self) -> Option<&DebugEvent> {
        if self.current_index == 0 {
            None
        } else {
            self.events.get(self.current_index - 1)
        }
    }

    /// Skip forward until we reach an event whose `step_number` matches `step_number`.
    /// Returns the matching event, or `None` if no such event exists ahead.
    pub fn skip_to_step(&mut self, step_number: usize) -> Option<&DebugEvent> {
        while self.current_index < self.events.len() {
            if self.events[self.current_index].step_number == step_number {
                let idx = self.current_index;
                self.current_index += 1;
                return Some(&self.events[idx]);
            }
            self.current_index += 1;
        }
        None
    }

    /// Reset replay to the beginning.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Number of events remaining (not yet consumed by `next`).
    pub fn remaining(&self) -> usize {
        self.events.len().saturating_sub(self.current_index)
    }

    /// Total number of events in the replay.
    pub fn total_events(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when all events have been consumed.
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.events.len()
    }

    /// Progress as a fraction `[0.0, 1.0]`.
    pub fn progress(&self) -> f64 {
        if self.events.is_empty() {
            return 1.0;
        }
        self.current_index as f64 / self.events.len() as f64
    }

    /// The agent id associated with this replay.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

// =============================================================================
// PerformanceProfiler / StepProfile / ProfileSummary
// =============================================================================

/// A per-step performance profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepProfile {
    pub step_number: usize,
    pub action_name: String,
    pub duration_ms: u64,
    pub token_count: usize,
    pub cost: f64,
    pub memory_delta_bytes: i64,
}

/// Collects per-step performance metrics during an agent run.
#[derive(Debug)]
pub struct PerformanceProfiler {
    step_metrics: Vec<StepProfile>,
    active: bool,
}

impl PerformanceProfiler {
    /// Create a new, inactive profiler.
    pub fn new() -> Self {
        Self {
            step_metrics: Vec::new(),
            active: false,
        }
    }

    /// Activate profiling.
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Deactivate profiling.
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Whether the profiler is currently collecting data.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Record a single step profile. Fails if the profiler is not active.
    pub fn record_step(&mut self, profile: StepProfile) -> Result<(), DevToolsError> {
        if !self.active {
            return Err(DevToolsError::ProfilingUnavailable {
                reason: "Profiler is not active — call start() first".to_string(),
            });
        }
        self.step_metrics.push(profile);
        Ok(())
    }

    /// Read-only view of all recorded step profiles.
    pub fn steps(&self) -> &[StepProfile] {
        &self.step_metrics
    }

    /// Sum of all step durations.
    pub fn total_duration_ms(&self) -> u64 {
        self.step_metrics.iter().map(|s| s.duration_ms).sum()
    }

    /// Sum of all token counts.
    pub fn total_tokens(&self) -> usize {
        self.step_metrics.iter().map(|s| s.token_count).sum()
    }

    /// Sum of all step costs.
    pub fn total_cost(&self) -> f64 {
        self.step_metrics.iter().map(|s| s.cost).sum()
    }

    /// Average step duration. Returns 0.0 when there are no steps.
    pub fn avg_step_duration_ms(&self) -> f64 {
        if self.step_metrics.is_empty() {
            return 0.0;
        }
        self.total_duration_ms() as f64 / self.step_metrics.len() as f64
    }

    /// The step with the longest duration.
    pub fn slowest_step(&self) -> Option<&StepProfile> {
        self.step_metrics.iter().max_by_key(|s| s.duration_ms)
    }

    /// The step with the highest cost.
    pub fn most_expensive_step(&self) -> Option<&StepProfile> {
        self.step_metrics
            .iter()
            .max_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Produce an aggregate summary of all recorded steps.
    pub fn summary(&self) -> ProfileSummary {
        ProfileSummary {
            total_steps: self.step_metrics.len(),
            total_duration_ms: self.total_duration_ms(),
            avg_duration_ms: self.avg_step_duration_ms(),
            total_tokens: self.total_tokens(),
            total_cost: self.total_cost(),
            slowest_step_name: self.slowest_step().map(|s| s.action_name.clone()),
            most_expensive_step_name: self.most_expensive_step().map(|s| s.action_name.clone()),
        }
    }

    /// Remove all recorded profiles (does not change active state).
    pub fn clear(&mut self) {
        self.step_metrics.clear();
    }
}

/// Aggregate performance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    pub total_steps: usize,
    pub total_duration_ms: u64,
    pub avg_duration_ms: f64,
    pub total_tokens: usize,
    pub total_cost: f64,
    pub slowest_step_name: Option<String>,
    pub most_expensive_step_name: Option<String>,
}

// =============================================================================
// StateInspector / StateSnapshot / StateDiff
// =============================================================================

/// A snapshot of agent state captured at a particular step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub step_number: usize,
    pub timestamp: u64,
    pub state_data: HashMap<String, serde_json::Value>,
    pub label: String,
}

/// The difference between two state snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    pub step_a: usize,
    pub step_b: usize,
    pub added_keys: Vec<String>,
    pub removed_keys: Vec<String>,
    pub changed_keys: Vec<String>,
}

/// Captures and compares agent state snapshots.
#[derive(Debug)]
pub struct StateInspector {
    snapshots: Vec<StateSnapshot>,
    max_snapshots: usize,
}

impl StateInspector {
    /// Create a new inspector that stores up to `max_snapshots` snapshots.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
        }
    }

    /// Capture a state snapshot. If the store is full, the oldest snapshot is discarded.
    pub fn capture(
        &mut self,
        step_number: usize,
        label: impl Into<String>,
        data: HashMap<String, serde_json::Value>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0);
        }

        self.snapshots.push(StateSnapshot {
            step_number,
            timestamp,
            state_data: data,
            label: label.into(),
        });
    }

    /// Look up the first snapshot captured at the given step number.
    pub fn snapshot_at(&self, step_number: usize) -> Option<&StateSnapshot> {
        self.snapshots
            .iter()
            .find(|s| s.step_number == step_number)
    }

    /// Read-only view of all snapshots.
    pub fn snapshots(&self) -> &[StateSnapshot] {
        &self.snapshots
    }

    /// The most recently captured snapshot.
    pub fn latest(&self) -> Option<&StateSnapshot> {
        self.snapshots.last()
    }

    /// Compute the difference between two snapshots identified by step number.
    /// Returns `None` if either snapshot cannot be found.
    pub fn diff(&self, step_a: usize, step_b: usize) -> Option<StateDiff> {
        let snap_a = self.snapshot_at(step_a)?;
        let snap_b = self.snapshot_at(step_b)?;

        let keys_a: std::collections::HashSet<&String> = snap_a.state_data.keys().collect();
        let keys_b: std::collections::HashSet<&String> = snap_b.state_data.keys().collect();

        let added_keys: Vec<String> = keys_b
            .difference(&keys_a)
            .map(|k| (*k).clone())
            .collect();
        let removed_keys: Vec<String> = keys_a
            .difference(&keys_b)
            .map(|k| (*k).clone())
            .collect();
        let changed_keys: Vec<String> = keys_a
            .intersection(&keys_b)
            .filter(|k| snap_a.state_data[**k] != snap_b.state_data[**k])
            .map(|k| (*k).clone())
            .collect();

        Some(StateDiff {
            step_a,
            step_b,
            added_keys,
            removed_keys,
            changed_keys,
        })
    }

    /// Number of snapshots currently stored.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Remove all stored snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

// =============================================================================
// AgentDebugger
// =============================================================================

/// Unified debugging facade that combines recording, profiling, state inspection, and breakpoints.
#[derive(Debug)]
pub struct AgentDebugger {
    config: DevToolsConfig,
    recorder: ExecutionRecorder,
    profiler: PerformanceProfiler,
    inspector: StateInspector,
    hit_breakpoints: Vec<(Breakpoint, DebugEvent)>,
}

impl AgentDebugger {
    /// Create a debugger with the supplied configuration.
    pub fn new(agent_id: impl Into<String>, config: DevToolsConfig) -> Self {
        let max_events = config.max_recording_steps;
        Self {
            config,
            recorder: ExecutionRecorder::new(agent_id, max_events),
            profiler: PerformanceProfiler::new(),
            inspector: StateInspector::new(1000),
            hit_breakpoints: Vec::new(),
        }
    }

    /// Create a debugger with default configuration.
    pub fn with_defaults(agent_id: impl Into<String>) -> Self {
        Self::new(agent_id, DevToolsConfig::default())
    }

    /// Start both the recorder (if enabled) and profiler (if enabled).
    pub fn start(&mut self) {
        if self.config.enable_recording {
            self.recorder.start();
        }
        if self.config.enable_profiling {
            self.profiler.start();
        }
    }

    /// Stop both the recorder and profiler.
    pub fn stop(&mut self) {
        self.recorder.stop();
        self.profiler.stop();
    }

    /// Process an event: record it, then check all breakpoints. Returns references
    /// to the breakpoints that matched this event.
    pub fn process_event(&mut self, event: DebugEvent) -> Vec<&Breakpoint> {
        // Record if enabled and active
        if self.config.enable_recording && self.recorder.is_recording() {
            // Ignore recording capacity errors silently
            let _ = self.recorder.record(event.clone());
        }

        // Check breakpoints
        let mut triggered_indices = Vec::new();
        for (i, bp) in self.config.breakpoints.iter().enumerate() {
            if bp.matches(&event) {
                triggered_indices.push(i);
            }
        }

        // Store hits
        for &i in &triggered_indices {
            self.hit_breakpoints
                .push((self.config.breakpoints[i].clone(), event.clone()));
        }

        // Return references to the triggered breakpoints
        triggered_indices
            .into_iter()
            .map(|i| &self.config.breakpoints[i])
            .collect()
    }

    /// Read-only access to the recorder.
    pub fn recorder(&self) -> &ExecutionRecorder {
        &self.recorder
    }

    /// Read-only access to the profiler.
    pub fn profiler(&self) -> &PerformanceProfiler {
        &self.profiler
    }

    /// Read-only access to the state inspector.
    pub fn inspector(&self) -> &StateInspector {
        &self.inspector
    }

    /// Mutable access to the state inspector for capturing snapshots.
    pub fn inspector_mut(&mut self) -> &mut StateInspector {
        &mut self.inspector
    }

    /// All breakpoint hits so far (breakpoint, triggering event).
    pub fn hit_breakpoints(&self) -> &[(Breakpoint, DebugEvent)] {
        &self.hit_breakpoints
    }

    /// Add a new breakpoint.
    pub fn add_breakpoint(&mut self, bp: Breakpoint) {
        self.config.breakpoints.push(bp);
    }

    /// Remove all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.config.breakpoints.clear();
    }

    /// Create a replay from the current recording.
    pub fn create_replay(&self) -> ExecutionReplay {
        ExecutionReplay::new(&self.recorder)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        event_type: DebugEventType,
        step: usize,
        tool: Option<&str>,
        confidence: Option<f64>,
        cost: Option<f64>,
        is_error: bool,
    ) -> DebugEvent {
        DebugEvent {
            event_type,
            step_number: step,
            tool_name: tool.map(|s| s.to_string()),
            confidence,
            cost,
            is_error,
            timestamp: 1000,
            data: HashMap::new(),
        }
    }

    // ---- DevToolsConfig defaults ----

    #[test]
    fn test_devtools_config_defaults() {
        let cfg = DevToolsConfig::default();
        assert!(cfg.enable_recording);
        assert!(cfg.enable_profiling);
        assert_eq!(cfg.max_recording_steps, 10000);
        assert!(cfg.breakpoints.is_empty());
    }

    // ---- Breakpoint matching ----

    #[test]
    fn test_breakpoint_before_tool_call_matches() {
        let bp = Breakpoint::BeforeToolCall {
            tool_name: "search".to_string(),
        };
        let event = make_event(
            DebugEventType::ToolCallStart,
            1,
            Some("search"),
            None,
            None,
            false,
        );
        assert!(bp.matches(&event));

        // Wrong tool name
        let event2 = make_event(
            DebugEventType::ToolCallStart,
            1,
            Some("calculate"),
            None,
            None,
            false,
        );
        assert!(!bp.matches(&event2));

        // Right name but wrong event type
        let event3 = make_event(
            DebugEventType::ToolCallEnd,
            1,
            Some("search"),
            None,
            None,
            false,
        );
        assert!(!bp.matches(&event3));
    }

    #[test]
    fn test_breakpoint_after_step_matches() {
        let bp = Breakpoint::AfterStep { step_number: 5 };
        let event = make_event(DebugEventType::StepComplete, 5, None, None, None, false);
        assert!(bp.matches(&event));

        let event2 = make_event(DebugEventType::StepComplete, 3, None, None, None, false);
        assert!(!bp.matches(&event2));

        // Correct step but wrong event type
        let event3 = make_event(DebugEventType::LlmQueryStart, 5, None, None, None, false);
        assert!(!bp.matches(&event3));
    }

    #[test]
    fn test_breakpoint_on_error_matches() {
        let bp = Breakpoint::OnError;
        let event = make_event(DebugEventType::Error, 1, None, None, None, true);
        assert!(bp.matches(&event));

        let event2 = make_event(DebugEventType::StepComplete, 1, None, None, None, false);
        assert!(!bp.matches(&event2));
    }

    #[test]
    fn test_breakpoint_on_confidence_below_matches() {
        let bp = Breakpoint::OnConfidenceBelow { threshold: 0.5 };
        let event = make_event(DebugEventType::StepComplete, 1, None, Some(0.3), None, false);
        assert!(bp.matches(&event));

        let event2 = make_event(DebugEventType::StepComplete, 1, None, Some(0.7), None, false);
        assert!(!bp.matches(&event2));

        // No confidence => no match
        let event3 = make_event(DebugEventType::StepComplete, 1, None, None, None, false);
        assert!(!bp.matches(&event3));
    }

    #[test]
    fn test_breakpoint_on_cost_above_matches() {
        let bp = Breakpoint::OnCostAbove { threshold: 1.0 };
        let event = make_event(DebugEventType::StepComplete, 1, None, None, Some(1.5), false);
        assert!(bp.matches(&event));

        let event2 = make_event(DebugEventType::StepComplete, 1, None, None, Some(0.5), false);
        assert!(!bp.matches(&event2));

        // No cost => no match
        let event3 = make_event(DebugEventType::StepComplete, 1, None, None, None, false);
        assert!(!bp.matches(&event3));
    }

    // ---- ExecutionRecorder ----

    #[test]
    fn test_recorder_new_start_stop() {
        let mut rec = ExecutionRecorder::new("agent-1", 100);
        assert!(!rec.is_recording());
        assert_eq!(rec.agent_id(), "agent-1");
        assert_eq!(rec.event_count(), 0);

        rec.start();
        assert!(rec.is_recording());

        rec.stop();
        assert!(!rec.is_recording());
    }

    #[test]
    fn test_recorder_record_not_recording_fails() {
        let mut rec = ExecutionRecorder::new("agent-1", 100);
        let event = make_event(DebugEventType::StepComplete, 0, None, None, None, false);
        let result = rec.record(event);
        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_record_events_and_retrieve() {
        let mut rec = ExecutionRecorder::new("agent-1", 100);
        rec.start();

        let e1 = make_event(DebugEventType::ToolCallStart, 0, Some("search"), None, None, false);
        let e2 = make_event(DebugEventType::ToolCallEnd, 0, Some("search"), None, None, false);
        let e3 = make_event(DebugEventType::StepComplete, 1, None, None, None, false);

        rec.record(e1).unwrap();
        rec.record(e2).unwrap();
        rec.record(e3).unwrap();

        assert_eq!(rec.event_count(), 3);
        assert_eq!(rec.events().len(), 3);

        let tool_starts = rec.events_by_type(&DebugEventType::ToolCallStart);
        assert_eq!(tool_starts.len(), 1);
        assert_eq!(tool_starts[0].tool_name.as_deref(), Some("search"));

        let range = rec.events_in_range(0, 0);
        assert_eq!(range.len(), 2); // step 0 has two events

        rec.add_metadata("run_id".to_string(), "abc".to_string());

        rec.clear();
        assert_eq!(rec.event_count(), 0);
    }

    #[test]
    fn test_recorder_json_roundtrip() {
        let mut rec = ExecutionRecorder::new("agent-rt", 100);
        rec.start();
        rec.record(make_event(
            DebugEventType::LlmQueryStart,
            0,
            None,
            Some(0.9),
            Some(0.01),
            false,
        ))
        .unwrap();
        rec.record(make_event(
            DebugEventType::LlmQueryEnd,
            0,
            None,
            Some(0.85),
            Some(0.02),
            false,
        ))
        .unwrap();
        rec.add_metadata("model".to_string(), "gpt-4".to_string());
        rec.stop();

        let json = rec.to_json().unwrap();
        let rec2 = ExecutionRecorder::from_json(&json).unwrap();

        assert_eq!(rec2.agent_id(), "agent-rt");
        assert_eq!(rec2.event_count(), 2);
        assert!(!rec2.is_recording()); // restored in stopped state
        assert_eq!(rec2.events()[0].event_type, DebugEventType::LlmQueryStart);
        assert_eq!(rec2.events()[1].confidence, Some(0.85));
    }

    // ---- ExecutionReplay ----

    #[test]
    fn test_replay_next_peek_reset_progress() {
        let mut rec = ExecutionRecorder::new("agent-r", 100);
        rec.start();
        rec.record(make_event(DebugEventType::StepComplete, 0, None, None, None, false))
            .unwrap();
        rec.record(make_event(DebugEventType::StepComplete, 1, None, None, None, false))
            .unwrap();
        rec.record(make_event(DebugEventType::StepComplete, 2, None, None, None, false))
            .unwrap();
        rec.stop();

        let mut replay = ExecutionReplay::new(&rec);
        assert_eq!(replay.total_events(), 3);
        assert_eq!(replay.remaining(), 3);
        assert!(!replay.is_complete());
        assert!((replay.progress() - 0.0).abs() < f64::EPSILON);

        // current is None before first next()
        assert!(replay.current().is_none());

        // peek does not advance
        assert_eq!(replay.peek().unwrap().step_number, 0);
        assert_eq!(replay.remaining(), 3);

        // next advances
        let first = replay.next().unwrap();
        assert_eq!(first.step_number, 0);
        assert_eq!(replay.remaining(), 2);
        assert!(replay.current().is_some());
        assert_eq!(replay.current().unwrap().step_number, 0);

        replay.next();
        replay.next();
        assert!(replay.is_complete());
        assert!((replay.progress() - 1.0).abs() < f64::EPSILON);
        assert!(replay.next().is_none());

        replay.reset();
        assert_eq!(replay.remaining(), 3);
        assert!(!replay.is_complete());
    }

    #[test]
    fn test_replay_skip_to_step() {
        let mut rec = ExecutionRecorder::new("agent-skip", 100);
        rec.start();
        for i in 0..5 {
            rec.record(make_event(
                DebugEventType::StepComplete,
                i,
                None,
                None,
                None,
                false,
            ))
            .unwrap();
        }
        rec.stop();

        let mut replay = ExecutionReplay::new(&rec);
        let found = replay.skip_to_step(3);
        assert!(found.is_some());
        assert_eq!(found.unwrap().step_number, 3);
        assert_eq!(replay.remaining(), 1); // only step 4 left

        // Skip to non-existent step
        let not_found = replay.skip_to_step(99);
        assert!(not_found.is_none());
        assert!(replay.is_complete());
    }

    // ---- PerformanceProfiler ----

    #[test]
    fn test_profiler_start_stop_record_step() {
        let mut profiler = PerformanceProfiler::new();
        assert!(!profiler.is_active());

        // Recording while inactive fails
        let result = profiler.record_step(StepProfile {
            step_number: 0,
            action_name: "plan".to_string(),
            duration_ms: 100,
            token_count: 50,
            cost: 0.01,
            memory_delta_bytes: 1024,
        });
        assert!(result.is_err());

        profiler.start();
        assert!(profiler.is_active());

        profiler
            .record_step(StepProfile {
                step_number: 0,
                action_name: "plan".to_string(),
                duration_ms: 100,
                token_count: 50,
                cost: 0.01,
                memory_delta_bytes: 1024,
            })
            .unwrap();

        profiler
            .record_step(StepProfile {
                step_number: 1,
                action_name: "execute".to_string(),
                duration_ms: 300,
                token_count: 200,
                cost: 0.05,
                memory_delta_bytes: 2048,
            })
            .unwrap();

        assert_eq!(profiler.steps().len(), 2);
        assert_eq!(profiler.total_duration_ms(), 400);
        assert_eq!(profiler.total_tokens(), 250);
        assert!((profiler.total_cost() - 0.06).abs() < 1e-9);
        assert!((profiler.avg_step_duration_ms() - 200.0).abs() < 1e-9);

        profiler.stop();
        assert!(!profiler.is_active());

        profiler.clear();
        assert_eq!(profiler.steps().len(), 0);
    }

    #[test]
    fn test_profiler_summary_slowest_most_expensive() {
        let mut profiler = PerformanceProfiler::new();
        profiler.start();

        profiler
            .record_step(StepProfile {
                step_number: 0,
                action_name: "fast_cheap".to_string(),
                duration_ms: 10,
                token_count: 5,
                cost: 0.001,
                memory_delta_bytes: 0,
            })
            .unwrap();

        profiler
            .record_step(StepProfile {
                step_number: 1,
                action_name: "slow_moderate".to_string(),
                duration_ms: 500,
                token_count: 100,
                cost: 0.02,
                memory_delta_bytes: 4096,
            })
            .unwrap();

        profiler
            .record_step(StepProfile {
                step_number: 2,
                action_name: "fast_expensive".to_string(),
                duration_ms: 20,
                token_count: 300,
                cost: 0.10,
                memory_delta_bytes: -1024,
            })
            .unwrap();

        let slowest = profiler.slowest_step().unwrap();
        assert_eq!(slowest.action_name, "slow_moderate");

        let most_expensive = profiler.most_expensive_step().unwrap();
        assert_eq!(most_expensive.action_name, "fast_expensive");

        let summary = profiler.summary();
        assert_eq!(summary.total_steps, 3);
        assert_eq!(summary.total_duration_ms, 530);
        assert_eq!(summary.total_tokens, 405);
        assert!((summary.total_cost - 0.121).abs() < 1e-9);
        assert_eq!(
            summary.slowest_step_name.as_deref(),
            Some("slow_moderate")
        );
        assert_eq!(
            summary.most_expensive_step_name.as_deref(),
            Some("fast_expensive")
        );
    }

    // ---- StateInspector ----

    #[test]
    fn test_state_inspector_capture_and_lookup() {
        let mut inspector = StateInspector::new(10);
        assert_eq!(inspector.snapshot_count(), 0);
        assert!(inspector.latest().is_none());

        let mut data1 = HashMap::new();
        data1.insert("plan".to_string(), serde_json::json!("initial"));
        inspector.capture(0, "start", data1);

        let mut data2 = HashMap::new();
        data2.insert("plan".to_string(), serde_json::json!("revised"));
        data2.insert("tool_result".to_string(), serde_json::json!(42));
        inspector.capture(1, "after_tool", data2);

        assert_eq!(inspector.snapshot_count(), 2);

        let snap0 = inspector.snapshot_at(0).unwrap();
        assert_eq!(snap0.label, "start");
        assert_eq!(snap0.state_data["plan"], serde_json::json!("initial"));

        let latest = inspector.latest().unwrap();
        assert_eq!(latest.step_number, 1);

        assert!(inspector.snapshot_at(99).is_none());
    }

    #[test]
    fn test_state_diff_added_removed_changed() {
        let mut inspector = StateInspector::new(10);

        let mut data_a = HashMap::new();
        data_a.insert("alpha".to_string(), serde_json::json!(1));
        data_a.insert("beta".to_string(), serde_json::json!("old"));
        data_a.insert("gamma".to_string(), serde_json::json!(true));
        inspector.capture(0, "step_a", data_a);

        let mut data_b = HashMap::new();
        data_b.insert("beta".to_string(), serde_json::json!("new")); // changed
        data_b.insert("gamma".to_string(), serde_json::json!(true)); // unchanged
        data_b.insert("delta".to_string(), serde_json::json!(99)); // added
        // alpha is removed
        inspector.capture(1, "step_b", data_b);

        let diff = inspector.diff(0, 1).unwrap();
        assert_eq!(diff.step_a, 0);
        assert_eq!(diff.step_b, 1);
        assert!(diff.added_keys.contains(&"delta".to_string()));
        assert!(diff.removed_keys.contains(&"alpha".to_string()));
        assert!(diff.changed_keys.contains(&"beta".to_string()));
        assert!(!diff.changed_keys.contains(&"gamma".to_string())); // same value

        // Diff with missing step returns None
        assert!(inspector.diff(0, 99).is_none());
    }

    #[test]
    fn test_state_inspector_max_snapshots() {
        let mut inspector = StateInspector::new(2);
        inspector.capture(0, "a", HashMap::new());
        inspector.capture(1, "b", HashMap::new());
        inspector.capture(2, "c", HashMap::new());
        assert_eq!(inspector.snapshot_count(), 2);
        // Oldest (step 0) should have been evicted
        assert!(inspector.snapshot_at(0).is_none());
        assert!(inspector.snapshot_at(1).is_some());
        assert!(inspector.snapshot_at(2).is_some());

        inspector.clear();
        assert_eq!(inspector.snapshot_count(), 0);
    }

    // ---- AgentDebugger ----

    #[test]
    fn test_agent_debugger_process_event_triggers_breakpoints() {
        let mut config = DevToolsConfig::default();
        config.breakpoints.push(Breakpoint::OnError);
        config
            .breakpoints
            .push(Breakpoint::OnCostAbove { threshold: 1.0 });

        let mut dbg = AgentDebugger::new("agent-dbg", config);
        dbg.start();

        // Non-matching event
        let e1 = make_event(DebugEventType::StepComplete, 0, None, None, Some(0.5), false);
        let triggered = dbg.process_event(e1);
        assert!(triggered.is_empty());

        // Error event triggers OnError
        let e2 = make_event(DebugEventType::Error, 1, None, None, None, true);
        let triggered = dbg.process_event(e2);
        assert_eq!(triggered.len(), 1);

        // High cost triggers OnCostAbove
        let e3 = make_event(DebugEventType::StepComplete, 2, None, None, Some(5.0), false);
        let triggered = dbg.process_event(e3);
        assert_eq!(triggered.len(), 1);

        // Error + high cost triggers both
        let e4 = make_event(DebugEventType::Error, 3, None, None, Some(10.0), true);
        let triggered = dbg.process_event(e4);
        assert_eq!(triggered.len(), 2);

        assert_eq!(dbg.hit_breakpoints().len(), 4); // 1 + 1 + 2
        assert_eq!(dbg.recorder().event_count(), 4);

        dbg.stop();
    }

    #[test]
    fn test_agent_debugger_create_replay_from_recording() {
        let mut dbg = AgentDebugger::with_defaults("agent-replay");
        dbg.start();

        dbg.process_event(make_event(
            DebugEventType::PlanningStart,
            0,
            None,
            None,
            None,
            false,
        ));
        dbg.process_event(make_event(
            DebugEventType::PlanningEnd,
            0,
            None,
            None,
            None,
            false,
        ));
        dbg.process_event(make_event(
            DebugEventType::ToolCallStart,
            1,
            Some("search"),
            None,
            None,
            false,
        ));

        let mut replay = dbg.create_replay();
        assert_eq!(replay.total_events(), 3);
        assert_eq!(replay.next().unwrap().event_type, DebugEventType::PlanningStart);
        assert_eq!(replay.next().unwrap().event_type, DebugEventType::PlanningEnd);
        assert_eq!(
            replay.next().unwrap().tool_name.as_deref(),
            Some("search")
        );
        assert!(replay.is_complete());
    }

    #[test]
    fn test_agent_debugger_add_clear_breakpoints() {
        let mut dbg = AgentDebugger::with_defaults("agent-bp");
        dbg.add_breakpoint(Breakpoint::OnError);
        dbg.add_breakpoint(Breakpoint::AtStepCount { count: 10 });
        dbg.start();

        // Verify breakpoints are active
        let e = make_event(DebugEventType::Error, 10, None, None, None, true);
        let triggered = dbg.process_event(e);
        assert_eq!(triggered.len(), 2); // both match

        dbg.clear_breakpoints();
        let e2 = make_event(DebugEventType::Error, 10, None, None, None, true);
        let triggered = dbg.process_event(e2);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_agent_debugger_inspector_access() {
        let mut dbg = AgentDebugger::with_defaults("agent-insp");

        let mut data = HashMap::new();
        data.insert("key".to_string(), serde_json::json!("value"));
        dbg.inspector_mut().capture(0, "initial", data);

        assert_eq!(dbg.inspector().snapshot_count(), 1);
        assert_eq!(dbg.inspector().latest().unwrap().label, "initial");
    }
}
