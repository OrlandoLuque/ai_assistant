//! Ephemeral sub-agent spawning — structural equivalent of Claude Code's
//! `Task` tool (Fork / Teammate / Explore patterns), adapted to a library.
//!
//! The main agent delegates a discrete sub-task to a short-lived sub-agent,
//! receives a single consolidated result, and discards the sub-agent. This is
//! **orthogonal** to the persistent multi-agent team that lives in
//! [`crate::multi_agent::AgentOrchestrator`]: orchestrators manage long-lived
//! roles and message passing; sub-agents are fire-and-forget delegations.
//!
//! # Scope & framing
//!
//! True filesystem/process isolation (git worktree, spawned subprocess) is a
//! **host-level concern** and stays with the caller (per the library's
//! "caller only configures, everything in-crate" policy — except host
//! resources). What lives in the crate:
//!
//! * A declarative model ([`SubAgentSpec`], [`SubAgentResult`]).
//! * A [`SubAgentRunner`] trait the caller can re-implement to plug in any
//!   isolation backend (git worktree, Docker, remote RPC, …).
//! * A default [`InProcessSubAgentRunner`] that executes the sub-task in the
//!   same process using the existing [`multi_agent::AgentOrchestrator`].
//! * Telemetry integration: counters `sub_agents_spawned_total` /
//!   `sub_agents_completed_total` and an OTel span named [`SPAN_NAME`].
//!
//! Feature-gated behind `sub-agents`. Implies `multi-agent` + `analytics` —
//! both zero-dep gates, so enabling `sub-agents` adds no new transitive deps.

use std::sync::Arc;
use std::time::Instant;

use crate::multi_agent::AgentRole;
use crate::opentelemetry_integration::OtelTracer;
use crate::telemetry::TelemetryCollector;

/// OpenTelemetry span name emitted when a sub-agent is spawned.
pub const SPAN_NAME: &str = "agent.sub_agent_spawned";

/// Kind of sub-agent — loosely mirrors Claude Code's `Task` sub-types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubAgentKind {
    /// General-purpose worker that runs a discrete task to completion.
    /// Analogous to Claude Code's `general-purpose` / "Fork" agents.
    Fork,
    /// Role-specialised worker (researcher, reviewer, writer, …).
    /// Analogous to Claude Code's "Teammate" role-based agents.
    Teammate,
    /// Read-only exploratory agent. The caller is expected to gate writes.
    /// Analogous to Claude Code's `Explore` agent.
    Explore,
}

impl SubAgentKind {
    /// String form used for telemetry and OpenTelemetry span attributes.
    pub fn as_str(&self) -> &'static str {
        match self {
            SubAgentKind::Fork => "Fork",
            SubAgentKind::Teammate => "Teammate",
            SubAgentKind::Explore => "Explore",
        }
    }
}

impl std::fmt::Display for SubAgentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Isolation level requested for a sub-agent.
///
/// Only [`IsolationLevel::InProcess`] is handled by the default runner.
/// Anything stronger requires a caller-provided [`SubAgentRunner`] that knows
/// how to create a git worktree, spawn a subprocess, or dispatch to a remote
/// worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsolationLevel {
    /// No isolation — sub-agent runs in the same process, sharing memory.
    InProcess,
    /// Logical context isolation — new conversation/context window, same
    /// process. Useful to keep sub-agent tool calls off the main transcript.
    ContextIsolated,
    /// Full host-level isolation (worktree, subprocess, container, remote).
    /// The default runner reports [`SubAgentStatus::Deferred`] for this
    /// variant — callers must provide their own runner.
    ExternalProcess,
}

impl IsolationLevel {
    /// String form for telemetry.
    pub fn as_str(&self) -> &'static str {
        match self {
            IsolationLevel::InProcess => "InProcess",
            IsolationLevel::ContextIsolated => "ContextIsolated",
            IsolationLevel::ExternalProcess => "ExternalProcess",
        }
    }
}

/// Outcome of a sub-agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubAgentStatus {
    /// Sub-agent ran to completion and produced a result.
    Completed,
    /// Sub-agent failed — see `summary` / `artifacts` for details.
    Failed,
    /// Sub-agent was cancelled by the caller or by a budget cutoff.
    Cancelled,
    /// Runner cannot handle this spec (e.g. `ExternalProcess` with the default
    /// runner). Caller should route the spec to a different runner.
    Deferred,
}

impl SubAgentStatus {
    /// String form for telemetry.
    pub fn as_str(&self) -> &'static str {
        match self {
            SubAgentStatus::Completed => "Completed",
            SubAgentStatus::Failed => "Failed",
            SubAgentStatus::Cancelled => "Cancelled",
            SubAgentStatus::Deferred => "Deferred",
        }
    }

    /// Whether the outcome counts as a successful delegation.
    pub fn is_success(&self) -> bool {
        matches!(self, SubAgentStatus::Completed)
    }
}

/// Declarative request to spawn a sub-agent.
///
/// Produced by the main agent / planner; consumed by a [`SubAgentRunner`].
/// Construction uses a fluent builder (`with_*` methods) to keep call sites
/// compact.
#[derive(Debug, Clone)]
pub struct SubAgentSpec {
    /// Stable identifier assigned by the caller (e.g. uuid, task slug).
    pub id: String,
    /// Which sub-agent family to use.
    pub kind: SubAgentKind,
    /// Role hint for Teammate-kind specs; ignored for Fork/Explore by the
    /// default runner.
    pub role: AgentRole,
    /// Plain-text description of the task (passed to the sub-agent as its
    /// prompt). MUST be trusted / sanitised by the caller.
    pub task: String,
    /// Compact summary of the parent context to pass to the sub-agent.
    /// Keeping this explicit (instead of cloning the full transcript) is the
    /// whole point of Fork/Teammate: the sub-agent gets only what it needs.
    pub context_summary: Option<String>,
    /// Requested isolation level.
    pub isolation: IsolationLevel,
    /// Optional token / iteration budget hint for the sub-agent. The runner
    /// is free to enforce or ignore this.
    pub budget_hint: Option<u32>,
    /// Vision attachments for the sub-agent. Stored as content-addressed
    /// [`crate::vision::ImageRef`]s so the spec stays small even with many
    /// images; the runner resolves them through an `ImageStore`.
    #[cfg(feature = "vision")]
    pub images: Vec<crate::vision::ImageRef>,
}

impl SubAgentSpec {
    /// Create a new spec with sensible defaults:
    /// `role = Custom`, `isolation = InProcess`, no summary, no budget.
    pub fn new(id: impl Into<String>, kind: SubAgentKind, task: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            role: AgentRole::Custom,
            task: task.into(),
            context_summary: None,
            isolation: IsolationLevel::InProcess,
            budget_hint: None,
            #[cfg(feature = "vision")]
            images: Vec::new(),
        }
    }

    /// Set the role hint (only meaningful for `SubAgentKind::Teammate`).
    pub fn with_role(mut self, role: AgentRole) -> Self {
        self.role = role;
        self
    }

    /// Attach a compact parent-context summary.
    pub fn with_context_summary(mut self, summary: impl Into<String>) -> Self {
        self.context_summary = Some(summary.into());
        self
    }

    /// Request a stronger isolation level.
    pub fn with_isolation(mut self, isolation: IsolationLevel) -> Self {
        self.isolation = isolation;
        self
    }

    /// Suggest a budget (tokens or iterations — runner-defined).
    pub fn with_budget_hint(mut self, budget: u32) -> Self {
        self.budget_hint = Some(budget);
        self
    }

    /// Attach a vision image (by ref) to this spec. Refs travel with the
    /// spec; bytes live in an `ImageStore` and are pulled by the runner.
    #[cfg(feature = "vision")]
    pub fn with_image(mut self, image: crate::vision::ImageRef) -> Self {
        self.images.push(image);
        self
    }

    /// Attach multiple image refs to this spec.
    #[cfg(feature = "vision")]
    pub fn with_images(mut self, images: Vec<crate::vision::ImageRef>) -> Self {
        self.images.extend(images);
        self
    }
}

/// Outcome returned by a [`SubAgentRunner`].
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    /// Echoes [`SubAgentSpec::id`] so the caller can correlate.
    pub spec_id: String,
    /// Outcome.
    pub status: SubAgentStatus,
    /// One-paragraph summary of what the sub-agent did. Empty on `Deferred`.
    pub summary: String,
    /// Artifacts the sub-agent produced (file paths, URLs, blob ids — runner-
    /// defined). Empty on `Deferred` / `Failed` / `Cancelled` when nothing
    /// was produced.
    pub artifacts: Vec<String>,
    /// Wall-clock duration of the run, in milliseconds. `0` on `Deferred`.
    pub duration_ms: u64,
    /// Image artifacts (screenshots, diagrams, generated images) produced by
    /// the sub-agent. Stored as content-addressed [`crate::vision::ImageRef`]s.
    #[cfg(feature = "vision")]
    pub image_artifacts: Vec<crate::vision::ImageRef>,
}

impl SubAgentResult {
    /// Helper: deferred result (default runner cannot handle this kind).
    pub fn deferred(spec_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            spec_id: spec_id.into(),
            status: SubAgentStatus::Deferred,
            summary: reason.into(),
            artifacts: Vec::new(),
            duration_ms: 0,
            #[cfg(feature = "vision")]
            image_artifacts: Vec::new(),
        }
    }
}

/// Trait implemented by anything that can execute a [`SubAgentSpec`].
///
/// The default [`InProcessSubAgentRunner`] is suitable for simple cases. For
/// worktree / subprocess / remote execution, callers implement this trait
/// themselves and plug it into their main loop.
pub trait SubAgentRunner: Send + Sync {
    /// Report whether this runner can handle a given spec. Callers should
    /// check this before `run` so they can route unsupported specs to another
    /// runner. A `false` here is equivalent to returning
    /// [`SubAgentStatus::Deferred`] from `run` — `supports` is just cheaper.
    fn supports(&self, spec: &SubAgentSpec) -> bool;

    /// Execute the sub-agent synchronously and return its result.
    ///
    /// Implementations MUST be total: never panic on a valid spec. If the
    /// runner cannot handle the spec, return [`SubAgentResult::deferred`].
    fn run(&self, spec: &SubAgentSpec) -> SubAgentResult;
}

/// Default, in-process runner.
///
/// Supports `IsolationLevel::InProcess` and `IsolationLevel::ContextIsolated`.
/// For `ExternalProcess` it returns a [`SubAgentStatus::Deferred`] result —
/// callers that need host-level isolation must plug in their own runner.
///
/// This runner does **not** call an LLM by itself: it is a thin harness that
/// returns a structured `Completed` result echoing the spec. Callers wire an
/// LLM-backed runner on top when they want real delegation. Keeping the
/// default LLM-free preserves the "no required network deps" property of the
/// core library.
#[derive(Default, Clone)]
pub struct InProcessSubAgentRunner {
    telemetry: Option<Arc<TelemetryCollector>>,
    tracer: Option<Arc<OtelTracer>>,
}

impl std::fmt::Debug for InProcessSubAgentRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcessSubAgentRunner")
            .field("telemetry", &self.telemetry.is_some())
            .field("tracer", &self.tracer.is_some())
            .finish()
    }
}

impl InProcessSubAgentRunner {
    /// Construct a new runner with no telemetry or tracer attached.
    pub fn new() -> Self {
        Self {
            telemetry: None,
            tracer: None,
        }
    }

    /// Attach a [`TelemetryCollector`]; `run()` will then record
    /// `sub_agents_spawned_total` at entry and `sub_agents_completed_total`
    /// on successful completion.
    pub fn with_telemetry(mut self, collector: Arc<TelemetryCollector>) -> Self {
        self.telemetry = Some(collector);
        self
    }

    /// Attach an [`OtelTracer`]; `run()` will then open a span named
    /// [`SPAN_NAME`] with `kind` and `isolation` attributes and end it before
    /// returning (the span is finalised as an error span for non-success
    /// statuses).
    pub fn with_tracer(mut self, tracer: Arc<OtelTracer>) -> Self {
        self.tracer = Some(tracer);
        self
    }
}

impl SubAgentRunner for InProcessSubAgentRunner {
    fn supports(&self, spec: &SubAgentSpec) -> bool {
        !matches!(spec.isolation, IsolationLevel::ExternalProcess)
    }

    fn run(&self, spec: &SubAgentSpec) -> SubAgentResult {
        let kind_str = spec.kind.as_str();
        let isolation_str = spec.isolation.as_str();

        if let Some(t) = self.telemetry.as_ref() {
            t.record_sub_agent_spawn(kind_str, isolation_str);
        }
        let span = self
            .tracer
            .as_ref()
            .map(|t| t.start_sub_agent_span(kind_str, isolation_str));

        let result = if !self.supports(spec) {
            SubAgentResult::deferred(
                &spec.id,
                "InProcessSubAgentRunner does not support ExternalProcess isolation",
            )
        } else {
            let started = Instant::now();
            let summary = format!(
                "Sub-agent {} ({}) acknowledged task: {}",
                spec.id, kind_str, spec.task
            );
            SubAgentResult {
                spec_id: spec.id.clone(),
                status: SubAgentStatus::Completed,
                summary,
                artifacts: Vec::new(),
                duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                #[cfg(feature = "vision")]
                image_artifacts: Vec::new(),
            }
        };

        if let Some(t) = self.telemetry.as_ref() {
            t.record_sub_agent_complete(
                kind_str,
                result.status.as_str(),
                result.status.is_success(),
            );
        }
        if let (Some(tracer), Some(span)) = (self.tracer.as_ref(), span) {
            if result.status.is_success() {
                tracer.end_span(span);
            } else {
                tracer.record_error(span, result.status.as_str());
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_string_round_trips() {
        assert_eq!(SubAgentKind::Fork.as_str(), "Fork");
        assert_eq!(SubAgentKind::Teammate.as_str(), "Teammate");
        assert_eq!(SubAgentKind::Explore.as_str(), "Explore");
        assert_eq!(format!("{}", SubAgentKind::Fork), "Fork");
    }

    #[test]
    fn isolation_string_round_trips() {
        assert_eq!(IsolationLevel::InProcess.as_str(), "InProcess");
        assert_eq!(IsolationLevel::ContextIsolated.as_str(), "ContextIsolated");
        assert_eq!(IsolationLevel::ExternalProcess.as_str(), "ExternalProcess");
    }

    #[test]
    fn status_string_round_trips() {
        assert_eq!(SubAgentStatus::Completed.as_str(), "Completed");
        assert_eq!(SubAgentStatus::Failed.as_str(), "Failed");
        assert_eq!(SubAgentStatus::Cancelled.as_str(), "Cancelled");
        assert_eq!(SubAgentStatus::Deferred.as_str(), "Deferred");
    }

    #[test]
    fn status_is_success_only_for_completed() {
        assert!(SubAgentStatus::Completed.is_success());
        assert!(!SubAgentStatus::Failed.is_success());
        assert!(!SubAgentStatus::Cancelled.is_success());
        assert!(!SubAgentStatus::Deferred.is_success());
    }

    #[test]
    fn spec_builder_sets_defaults() {
        let spec = SubAgentSpec::new("s1", SubAgentKind::Fork, "do the thing");
        assert_eq!(spec.id, "s1");
        assert_eq!(spec.kind, SubAgentKind::Fork);
        assert_eq!(spec.role, AgentRole::Custom);
        assert_eq!(spec.task, "do the thing");
        assert!(spec.context_summary.is_none());
        assert_eq!(spec.isolation, IsolationLevel::InProcess);
        assert!(spec.budget_hint.is_none());
    }

    #[test]
    fn spec_builder_chains() {
        let spec = SubAgentSpec::new("s2", SubAgentKind::Teammate, "review draft")
            .with_role(AgentRole::PeerReviewer)
            .with_context_summary("draft is 3 paragraphs on quantum")
            .with_isolation(IsolationLevel::ContextIsolated)
            .with_budget_hint(4_000);
        assert_eq!(spec.role, AgentRole::PeerReviewer);
        assert_eq!(
            spec.context_summary.as_deref(),
            Some("draft is 3 paragraphs on quantum")
        );
        assert_eq!(spec.isolation, IsolationLevel::ContextIsolated);
        assert_eq!(spec.budget_hint, Some(4_000));
    }

    #[test]
    fn default_runner_supports_in_process_and_context_isolated() {
        let runner = InProcessSubAgentRunner::new();
        let in_proc = SubAgentSpec::new("a", SubAgentKind::Fork, "t");
        let ctx_iso = SubAgentSpec::new("b", SubAgentKind::Fork, "t")
            .with_isolation(IsolationLevel::ContextIsolated);
        assert!(runner.supports(&in_proc));
        assert!(runner.supports(&ctx_iso));
    }

    #[test]
    fn default_runner_rejects_external_process() {
        let runner = InProcessSubAgentRunner::new();
        let external = SubAgentSpec::new("c", SubAgentKind::Fork, "t")
            .with_isolation(IsolationLevel::ExternalProcess);
        assert!(!runner.supports(&external));
    }

    #[test]
    fn default_runner_returns_deferred_for_external_process() {
        let runner = InProcessSubAgentRunner::new();
        let external = SubAgentSpec::new("c", SubAgentKind::Fork, "t")
            .with_isolation(IsolationLevel::ExternalProcess);
        let res = runner.run(&external);
        assert_eq!(res.status, SubAgentStatus::Deferred);
        assert_eq!(res.spec_id, "c");
        assert!(res.artifacts.is_empty());
        assert_eq!(res.duration_ms, 0);
        assert!(!res.summary.is_empty());
    }

    #[test]
    fn default_runner_completes_in_process_spec() {
        let runner = InProcessSubAgentRunner::new();
        let spec = SubAgentSpec::new("d", SubAgentKind::Fork, "summarise");
        let res = runner.run(&spec);
        assert_eq!(res.status, SubAgentStatus::Completed);
        assert_eq!(res.spec_id, "d");
        assert!(res.summary.contains("summarise"));
        assert!(res.summary.contains("Fork"));
    }

    #[test]
    fn default_runner_echoes_kind_in_summary() {
        let runner = InProcessSubAgentRunner::new();
        let spec = SubAgentSpec::new("e", SubAgentKind::Teammate, "review");
        let res = runner.run(&spec);
        assert!(res.summary.contains("Teammate"));
    }

    #[test]
    fn deferred_helper_builds_expected_result() {
        let res = SubAgentResult::deferred("x", "not supported");
        assert_eq!(res.status, SubAgentStatus::Deferred);
        assert_eq!(res.spec_id, "x");
        assert_eq!(res.summary, "not supported");
        assert!(res.artifacts.is_empty());
        assert_eq!(res.duration_ms, 0);
    }

    #[test]
    fn span_name_is_stable() {
        assert_eq!(SPAN_NAME, "agent.sub_agent_spawned");
    }

    #[test]
    fn runner_is_object_safe() {
        let _boxed: Box<dyn SubAgentRunner> = Box::new(InProcessSubAgentRunner::new());
    }

    #[test]
    fn runner_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InProcessSubAgentRunner>();
    }

    #[test]
    fn runner_with_telemetry_increments_counters() {
        use crate::telemetry::{TelemetryCollector, TelemetryConfig};
        let collector = Arc::new(TelemetryCollector::new(TelemetryConfig {
            enabled: true,
            ..Default::default()
        }));
        let runner = InProcessSubAgentRunner::new().with_telemetry(Arc::clone(&collector));
        let spec = SubAgentSpec::new("tel", SubAgentKind::Fork, "do it");
        let res = runner.run(&spec);
        assert_eq!(res.status, SubAgentStatus::Completed);
        let agg = collector.get_aggregated();
        assert_eq!(agg.sub_agents_spawned_total, 1);
        assert_eq!(agg.sub_agents_completed_total, 1);
    }

    #[test]
    fn runner_with_telemetry_records_deferred_without_success() {
        use crate::telemetry::{TelemetryCollector, TelemetryConfig};
        let collector = Arc::new(TelemetryCollector::new(TelemetryConfig {
            enabled: true,
            ..Default::default()
        }));
        let runner = InProcessSubAgentRunner::new().with_telemetry(Arc::clone(&collector));
        let spec = SubAgentSpec::new("x", SubAgentKind::Fork, "noop")
            .with_isolation(IsolationLevel::ExternalProcess);
        let res = runner.run(&spec);
        assert_eq!(res.status, SubAgentStatus::Deferred);
        let agg = collector.get_aggregated();
        assert_eq!(agg.sub_agents_spawned_total, 1);
        // Deferred is not a success — completed counter stays at 0.
        assert_eq!(agg.sub_agents_completed_total, 0);
    }

    #[test]
    fn runner_with_tracer_emits_span() {
        use crate::opentelemetry_integration::{OtelConfig, OtelTracer};
        let tracer = Arc::new(OtelTracer::new(OtelConfig::default()));
        let runner = InProcessSubAgentRunner::new().with_tracer(Arc::clone(&tracer));
        let spec = SubAgentSpec::new("tr", SubAgentKind::Explore, "look");
        let _res = runner.run(&spec);
        let completed = tracer.completed_spans();
        assert!(completed.iter().any(|s| s.operation == SPAN_NAME));
    }
}
