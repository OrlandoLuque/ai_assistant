//! Composable prompt fragments — conditional system prompt assembly.
//!
//! This module provides an extensible pattern for assembling system prompts from
//! small, conditional fragments. It is the structural equivalent of the ~110
//! conditional instruction strings found in Claude Code's leaked system prompt,
//! but rather than hardcoding them we let the caller compose and override.
//!
//! # Not RAG
//!
//! Fragments are **instructions** (how the model should behave), not knowledge
//! (what data the model should know). For retrieval of documents and grounding,
//! use the `rag` feature.
//!
//! # Usage
//!
//! ```no_run
//! # #[cfg(feature = "prompt-fragments")] {
//! use ai_assistant::{PromptBuilder, PromptContext, PromptPreset, Platform};
//!
//! let ctx = PromptContext::default()
//!     .with_platform(Platform::Linux)
//!     .with_locale("en");
//!
//! let prompt = PromptBuilder::new()
//!     .with_preset(PromptPreset::CodeDeveloper)
//!     .build(&ctx);
//! # }
//! ```
//!
//! # Fragment text is trusted
//!
//! Fragment text is concatenated verbatim into the system prompt. It must come
//! from trusted sources (your code or curated config files), **never** from end
//! user input — otherwise you create a prompt-injection vector.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "autonomous")]
use crate::mode_manager::OperationMode;

// =============================================================================
// Category & platform
// =============================================================================

/// Semantic category of a prompt fragment. Used for ordering conventions and
/// introspection via `build_with_trace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentCategory {
    /// Safety and compliance guidance (GDPR, content policies, red lines).
    Safety,
    /// How to call tools, interpret results, and handle tool errors.
    ToolGuidance,
    /// Situational context (available features, attached systems).
    Context,
    /// Output style, tone, formatting conventions.
    Style,
    /// Instructions specific to an `OperationMode`.
    ModeSpecific,
    /// OS / shell / environment specific notes.
    PlatformSpecific,
    /// Domain specific guidance (coding conventions, research style, etc.).
    Domain,
}

impl fmt::Display for FragmentCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FragmentCategory::Safety => "Safety",
            FragmentCategory::ToolGuidance => "ToolGuidance",
            FragmentCategory::Context => "Context",
            FragmentCategory::Style => "Style",
            FragmentCategory::ModeSpecific => "ModeSpecific",
            FragmentCategory::PlatformSpecific => "PlatformSpecific",
            FragmentCategory::Domain => "Domain",
        };
        f.write_str(s)
    }
}

/// Host OS family used by platform-specific fragments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    #[default]
    Other,
}

impl Platform {
    /// Detect the running host's platform from `std::env::consts::OS`.
    pub fn detect() -> Self {
        match std::env::consts::OS {
            "windows" => Platform::Windows,
            "linux" => Platform::Linux,
            "macos" => Platform::MacOS,
            _ => Platform::Other,
        }
    }
}

// =============================================================================
// Context
// =============================================================================

/// Runtime context used to evaluate fragment conditions.
///
/// All fields are optional / default-able; a fragment should only look at the
/// fields it cares about.
#[derive(Debug, Clone)]
pub struct PromptContext {
    /// Names of tools the agent can currently call (e.g. `["git", "retrieve", "bash"]`).
    pub tools_available: Vec<String>,
    /// Host OS family. Defaults to `Platform::Other`.
    pub platform: Platform,
    /// Current operation mode, when known.
    #[cfg(feature = "autonomous")]
    pub mode: Option<OperationMode>,
    /// BCP-47 language tag or simple 2-letter code (`"en"`, `"es"`, ...).
    pub locale: String,
    /// Region code (`"EU"`, `"US"`, ...). Used for compliance fragments.
    pub region: String,
    /// Caller-supplied key/value signals for custom fragments.
    pub custom: HashMap<String, String>,
}

impl Default for PromptContext {
    fn default() -> Self {
        Self {
            tools_available: Vec::new(),
            platform: Platform::Other,
            #[cfg(feature = "autonomous")]
            mode: None,
            locale: "en".to_string(),
            region: String::new(),
            custom: HashMap::new(),
        }
    }
}

impl PromptContext {
    /// Builder: set platform.
    pub fn with_platform(mut self, p: Platform) -> Self {
        self.platform = p;
        self
    }

    /// Builder: set locale (e.g. `"en"`, `"es"`).
    pub fn with_locale(mut self, l: impl Into<String>) -> Self {
        self.locale = l.into();
        self
    }

    /// Builder: set region (e.g. `"EU"`, `"US"`).
    pub fn with_region(mut self, r: impl Into<String>) -> Self {
        self.region = r.into();
        self
    }

    /// Builder: replace the tools list.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools_available = tools;
        self
    }

    /// Builder: set operation mode.
    #[cfg(feature = "autonomous")]
    pub fn with_mode(mut self, m: OperationMode) -> Self {
        self.mode = Some(m);
        self
    }

    /// Builder: add a custom key/value signal.
    pub fn with_custom(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.custom.insert(k.into(), v.into());
        self
    }

    /// Convenience: does `tools_available` contain `name`?
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools_available.iter().any(|t| t == name)
    }

    /// Convenience: does the locale start with the given prefix (case-insensitive)?
    pub fn locale_is(&self, prefix: &str) -> bool {
        self.locale
            .to_ascii_lowercase()
            .starts_with(&prefix.to_ascii_lowercase())
    }
}

// =============================================================================
// Fragment
// =============================================================================

/// Type alias for the condition closure. Stored behind `Arc` so fragments are
/// cheap to clone and safe to share across threads.
pub type FragmentCondition = Arc<dyn Fn(&PromptContext) -> bool + Send + Sync + 'static>;

/// A single conditional piece of prompt text.
///
/// Fragments are keyed by a `'static` string so they can be overridden or
/// removed deterministically by the caller.
#[derive(Clone)]
pub struct PromptFragment {
    pub key: &'static str,
    pub text: String,
    pub category: FragmentCategory,
    /// Lower priority = appears earlier in the built prompt. Conventional ranges:
    /// 0–9 Safety, 10–19 ToolGuidance, 20–29 Context, 30–39 Style, 40–49 Mode,
    /// 50–59 Platform, 100+ Domain.
    pub priority: u8,
    pub applies: FragmentCondition,
}

impl fmt::Debug for PromptFragment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptFragment")
            .field("key", &self.key)
            .field("category", &self.category)
            .field("priority", &self.priority)
            .field("text_len", &self.text.len())
            .finish()
    }
}

impl PromptFragment {
    /// Build a fragment with an explicit condition closure.
    pub fn new<F>(
        key: &'static str,
        text: impl Into<String>,
        category: FragmentCategory,
        priority: u8,
        applies: F,
    ) -> Self
    where
        F: Fn(&PromptContext) -> bool + Send + Sync + 'static,
    {
        Self {
            key,
            text: text.into(),
            category,
            priority,
            applies: Arc::new(applies),
        }
    }

    /// Build a fragment that always applies, regardless of context.
    pub fn always(
        key: &'static str,
        text: impl Into<String>,
        category: FragmentCategory,
        priority: u8,
    ) -> Self {
        Self::new(key, text, category, priority, |_| true)
    }
}

/// Introspection record produced by [`PromptBuilder::build_with_trace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedFragment {
    pub key: &'static str,
    pub category: FragmentCategory,
    pub priority: u8,
}

// =============================================================================
// Presets
// =============================================================================

/// Curated sets of fragments for common agent shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptPreset {
    /// Empty preset — useful as a base when fully custom.
    Minimal,
    /// Generic tool-calling chatbot.
    ToolUseChatbot,
    /// Retrieval-augmented assistant that cites sources.
    RagAssistant,
    /// Autonomous/agentic loop with repeated tool calls.
    AgenticLoop,
    /// Research assistant (arXiv / academic workflow).
    ResearchAgent,
    /// Coding agent with TDD + git conventions.
    CodeDeveloper,
}

impl PromptPreset {
    /// Materialize the fragments that make up this preset. Each call returns a
    /// fresh `Vec` so the caller can mutate without aliasing.
    pub fn fragments(&self) -> Vec<PromptFragment> {
        match self {
            PromptPreset::Minimal => Vec::new(),
            PromptPreset::ToolUseChatbot => vec![catalog::tool_use_guidance_general()],
            PromptPreset::RagAssistant => vec![
                catalog::tool_use_guidance_general(),
                catalog::rag_citation_reminder(),
            ],
            PromptPreset::AgenticLoop => vec![
                catalog::tool_use_guidance_general(),
                catalog::execute_mode_instructions(),
            ],
            PromptPreset::ResearchAgent => vec![
                catalog::tool_use_guidance_general(),
                catalog::rag_citation_reminder(),
                catalog::academic_citation_style(),
            ],
            PromptPreset::CodeDeveloper => vec![
                catalog::tool_use_guidance_general(),
                catalog::tdd_workflow(),
                catalog::git_commit_conventions(),
                catalog::rust_idioms(),
            ],
        }
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Composable builder that assembles fragments into a final system prompt.
#[derive(Clone, Default)]
pub struct PromptBuilder {
    fragments: Vec<PromptFragment>,
}

impl fmt::Debug for PromptBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptBuilder")
            .field("fragment_count", &self.fragments.len())
            .finish()
    }
}

impl PromptBuilder {
    /// Empty builder. Chain `.with_preset` / `.add_fragment` to populate.
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
        }
    }

    /// Seed the builder with all fragments from `preset`. Can be called more
    /// than once; each call appends.
    pub fn with_preset(mut self, preset: PromptPreset) -> Self {
        for f in preset.fragments() {
            self.insert_or_replace(f);
        }
        self
    }

    /// Add or replace a fragment. Fragments sharing the same `key` are replaced.
    pub fn add_fragment(mut self, f: PromptFragment) -> Self {
        self.insert_or_replace(f);
        self
    }

    /// Remove a fragment by key. No-op if absent.
    pub fn remove_fragment(mut self, key: &str) -> Self {
        self.fragments.retain(|f| f.key != key);
        self
    }

    /// Number of fragments currently registered (before filtering by context).
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Build the final prompt string for the given context.
    ///
    /// Fragments whose `applies(&ctx)` returns true are included, sorted by
    /// `(priority, insertion_order)`, joined by a blank line.
    pub fn build(&self, ctx: &PromptContext) -> String {
        let (prompt, _trace) = self.build_inner(ctx, false);
        prompt
    }

    /// Like [`build`] but also returns an `AppliedFragment` per included
    /// fragment in the exact order they appear in the output. Useful for
    /// debugging or telemetry.
    pub fn build_with_trace(&self, ctx: &PromptContext) -> (String, Vec<AppliedFragment>) {
        self.build_inner(ctx, true)
    }

    fn build_inner(
        &self,
        ctx: &PromptContext,
        collect_trace: bool,
    ) -> (String, Vec<AppliedFragment>) {
        // Enumerate to get insertion order for stable tie-breaking.
        let mut selected: Vec<(usize, &PromptFragment)> = self
            .fragments
            .iter()
            .enumerate()
            .filter(|(_, f)| (f.applies)(ctx))
            .collect();

        selected.sort_by(|(ai, a), (bi, b)| a.priority.cmp(&b.priority).then(ai.cmp(bi)));

        let mut out = String::new();
        let mut trace = if collect_trace {
            Vec::with_capacity(selected.len())
        } else {
            Vec::new()
        };

        for (_, frag) in &selected {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&frag.text);
            if collect_trace {
                trace.push(AppliedFragment {
                    key: frag.key,
                    category: frag.category,
                    priority: frag.priority,
                });
            }
        }

        (out, trace)
    }

    fn insert_or_replace(&mut self, f: PromptFragment) {
        if let Some(existing) = self.fragments.iter_mut().find(|x| x.key == f.key) {
            *existing = f;
        } else {
            self.fragments.push(f);
        }
    }
}

// =============================================================================
// Built-in fragment catalog
// =============================================================================

/// Catalog of built-in fragments. Each function returns a fresh `PromptFragment`
/// so callers can mutate freely. Text is localized via `ctx.locale` at build
/// time using the fragment's closure.
pub mod catalog {
    use super::*;

    fn localized(ctx: &PromptContext, en: &str, es: &str) -> String {
        if ctx.locale_is("es") {
            es.to_string()
        } else {
            en.to_string()
        }
    }

    /// Platform: Windows shell conventions reminder.
    pub fn windows_shell_note() -> PromptFragment {
        PromptFragment::new(
            "windows_shell_note",
            "Shell: use Unix-style syntax (forward slashes, /dev/null) even though the host \
             is Windows. The environment shims POSIX semantics.",
            FragmentCategory::PlatformSpecific,
            50,
            |ctx| ctx.platform == Platform::Windows,
        )
    }

    /// Platform: Unix shell note for non-Windows hosts.
    pub fn unix_shell_note() -> PromptFragment {
        PromptFragment::new(
            "unix_shell_note",
            "Shell: POSIX-compatible. Prefer standard Unix utilities and forward slashes in paths.",
            FragmentCategory::PlatformSpecific,
            50,
            |ctx| matches!(ctx.platform, Platform::Linux | Platform::MacOS),
        )
    }

    /// Tool guidance: generic, applies whenever tools are available.
    pub fn tool_use_guidance_general() -> PromptFragment {
        PromptFragment::new(
            "tool_use_guidance_general",
            "When tools are available, prefer calling a tool over guessing. If you are \
             unsure whether a tool applies, describe what you would look up and ask for \
             confirmation before running a destructive action.",
            FragmentCategory::ToolGuidance,
            10,
            |ctx| !ctx.tools_available.is_empty(),
        )
    }

    /// Mode: plan-mode guidance (no edits yet).
    pub fn plan_mode_instructions() -> PromptFragment {
        PromptFragment::always(
            "plan_mode_instructions",
            "You are in plan mode. Do not apply edits or run side-effecting commands yet — \
             produce a concrete plan and wait for explicit approval before executing.",
            FragmentCategory::ModeSpecific,
            40,
        )
    }

    /// Mode: autonomous execution guidance.
    pub fn execute_mode_instructions() -> PromptFragment {
        PromptFragment::always(
            "execute_mode_instructions",
            "You may execute tool calls iteratively to complete the task. After each tool \
             result, briefly assess progress before choosing the next action. Stop as soon \
             as the objective is met.",
            FragmentCategory::ModeSpecific,
            40,
        )
    }

    /// Context: RAG citation reminder, applies when a retrieval tool is present.
    pub fn rag_citation_reminder() -> PromptFragment {
        PromptFragment::new(
            "rag_citation_reminder",
            "Ground claims in retrieved sources. When you use information from a retrieval \
             result, cite the source identifier (filename, URL, or document id) in your \
             answer. Do not invent sources.",
            FragmentCategory::Context,
            20,
            |ctx| ctx.has_tool("retrieve") || ctx.has_tool("search") || ctx.has_tool("rag"),
        )
    }

    /// Safety: GDPR data-handling reminder for EU-region deployments.
    pub fn gdpr_eu_notice() -> PromptFragment {
        PromptFragment::new(
            "gdpr_eu_notice",
            "This deployment serves EU users. Do not store, transmit, or surface personal \
             data beyond what is strictly necessary for the current task. If a request \
             appears to require processing personal data, confirm the lawful basis first.",
            FragmentCategory::Safety,
            5,
            |ctx| ctx.region.eq_ignore_ascii_case("EU"),
        )
    }

    /// Domain: TDD workflow hint for coding agents.
    pub fn tdd_workflow() -> PromptFragment {
        PromptFragment::new(
            "tdd_workflow",
            "Prefer a test-driven loop: reproduce the issue with a failing test, implement \
             the minimal fix, then confirm the test passes and existing tests still pass.",
            FragmentCategory::Domain,
            100,
            |_| true,
        )
    }

    /// Domain: git commit conventions for coding agents with git in the toolset.
    pub fn git_commit_conventions() -> PromptFragment {
        PromptFragment::new(
            "git_commit_conventions",
            "When committing, write terse imperative-mood subjects (≤72 chars) and focus \
             the body on the 'why', not the 'what' — the diff already shows the 'what'.",
            FragmentCategory::Domain,
            100,
            |ctx| ctx.has_tool("git"),
        )
    }

    /// Domain: idiomatic Rust guidance, locale-aware.
    pub fn rust_idioms() -> PromptFragment {
        let text_source: FragmentCondition = Arc::new(|_| true);
        // We store the English default text and swap at build time by checking locale
        // through a fresh fragment per call; simpler: closure uses static strings.
        PromptFragment {
            key: "rust_idioms",
            text: "Follow idiomatic Rust: no `unwrap`/`expect` in production code paths, \
                   prefer `?` for error propagation, derive traits where practical, and \
                   keep error types meaningful (`thiserror` or a crate-local enum)."
                .to_string(),
            category: FragmentCategory::Domain,
            priority: 100,
            applies: text_source,
        }
    }

    /// Domain: academic citation style for research agents.
    pub fn academic_citation_style() -> PromptFragment {
        PromptFragment::always(
            "academic_citation_style",
            "Cite academic sources with author + year + venue (e.g. Smith et al., 2024, \
             ACL). Prefer primary sources and flag pre-prints vs. peer-reviewed work.",
            FragmentCategory::Domain,
            100,
        )
    }

    /// Convenience helper that returns the localized pair for [`rag_citation_reminder`].
    /// Kept as a reference for future localization work; currently unused so that the
    /// catalog stays single-language by default.
    #[allow(dead_code)]
    fn _rag_citation_reminder_localized(ctx: &PromptContext) -> String {
        localized(
            ctx,
            "Ground claims in retrieved sources; cite the source id.",
            "Fundamenta las afirmaciones en las fuentes recuperadas; cita el id de la fuente.",
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_default() -> PromptContext {
        PromptContext::default()
    }

    #[test]
    fn empty_builder_yields_empty_prompt() {
        let b = PromptBuilder::new();
        assert_eq!(b.build(&ctx_default()), "");
        assert_eq!(b.fragment_count(), 0);
    }

    #[test]
    fn always_fragment_appears_in_output() {
        let b = PromptBuilder::new().add_fragment(PromptFragment::always(
            "hello",
            "hello world",
            FragmentCategory::Style,
            30,
        ));
        let out = b.build(&ctx_default());
        assert_eq!(out, "hello world");
    }

    #[test]
    fn conditional_fragment_filters_by_context() {
        let b = PromptBuilder::new().add_fragment(catalog::windows_shell_note());
        let linux = PromptContext::default().with_platform(Platform::Linux);
        let windows = PromptContext::default().with_platform(Platform::Windows);
        assert_eq!(b.build(&linux), "");
        assert!(b.build(&windows).contains("Windows"));
    }

    #[test]
    fn priority_orders_fragments() {
        let b = PromptBuilder::new()
            .add_fragment(PromptFragment::always(
                "last",
                "LAST",
                FragmentCategory::Domain,
                100,
            ))
            .add_fragment(PromptFragment::always(
                "first",
                "FIRST",
                FragmentCategory::Safety,
                0,
            ));
        let out = b.build(&ctx_default());
        assert_eq!(out, "FIRST\n\nLAST");
    }

    #[test]
    fn tie_breaks_by_insertion_order() {
        let b = PromptBuilder::new()
            .add_fragment(PromptFragment::always(
                "a",
                "A",
                FragmentCategory::Style,
                10,
            ))
            .add_fragment(PromptFragment::always(
                "b",
                "B",
                FragmentCategory::Style,
                10,
            ));
        assert_eq!(b.build(&ctx_default()), "A\n\nB");
    }

    #[test]
    fn duplicate_key_replaces_existing() {
        let b = PromptBuilder::new()
            .add_fragment(PromptFragment::always(
                "k",
                "first",
                FragmentCategory::Style,
                30,
            ))
            .add_fragment(PromptFragment::always(
                "k",
                "second",
                FragmentCategory::Style,
                30,
            ));
        assert_eq!(b.fragment_count(), 1);
        assert_eq!(b.build(&ctx_default()), "second");
    }

    #[test]
    fn remove_fragment_is_noop_when_absent() {
        let b = PromptBuilder::new().remove_fragment("missing");
        assert_eq!(b.fragment_count(), 0);
    }

    #[test]
    fn remove_fragment_drops_the_entry() {
        let b = PromptBuilder::new()
            .add_fragment(PromptFragment::always(
                "k",
                "text",
                FragmentCategory::Style,
                30,
            ))
            .remove_fragment("k");
        assert_eq!(b.fragment_count(), 0);
        assert_eq!(b.build(&ctx_default()), "");
    }

    #[test]
    fn build_with_trace_reports_applied_fragments() {
        let b = PromptBuilder::new()
            .add_fragment(PromptFragment::always(
                "a",
                "A",
                FragmentCategory::Safety,
                0,
            ))
            .add_fragment(PromptFragment::always(
                "b",
                "B",
                FragmentCategory::Style,
                30,
            ));
        let (out, trace) = b.build_with_trace(&ctx_default());
        assert_eq!(out, "A\n\nB");
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].key, "a");
        assert_eq!(trace[0].category, FragmentCategory::Safety);
        assert_eq!(trace[1].key, "b");
    }

    #[test]
    fn has_tool_matches_exact_name() {
        let ctx = PromptContext::default().with_tools(vec!["git".into(), "bash".into()]);
        assert!(ctx.has_tool("git"));
        assert!(!ctx.has_tool("svn"));
    }

    #[test]
    fn locale_is_matches_prefix_case_insensitive() {
        let ctx = PromptContext::default().with_locale("ES-es");
        assert!(ctx.locale_is("es"));
        assert!(ctx.locale_is("ES"));
        assert!(!ctx.locale_is("fr"));
    }

    #[test]
    fn gdpr_notice_fires_for_eu_region_only() {
        let b = PromptBuilder::new().add_fragment(catalog::gdpr_eu_notice());
        let eu = PromptContext::default().with_region("EU");
        let us = PromptContext::default().with_region("US");
        assert!(b.build(&eu).contains("EU users"));
        assert_eq!(b.build(&us), "");
    }

    #[test]
    fn rag_citation_reminder_requires_retrieval_tool() {
        let b = PromptBuilder::new().add_fragment(catalog::rag_citation_reminder());
        let empty = PromptContext::default();
        let with_tool = PromptContext::default().with_tools(vec!["retrieve".into()]);
        assert_eq!(b.build(&empty), "");
        assert!(b.build(&with_tool).contains("Ground claims"));
    }

    #[test]
    fn git_commit_conventions_requires_git_tool() {
        let b = PromptBuilder::new().add_fragment(catalog::git_commit_conventions());
        let no_git = PromptContext::default();
        let with_git = PromptContext::default().with_tools(vec!["git".into()]);
        assert_eq!(b.build(&no_git), "");
        assert!(b.build(&with_git).contains("imperative-mood"));
    }

    #[test]
    fn tool_use_guidance_requires_any_tool() {
        let b = PromptBuilder::new().add_fragment(catalog::tool_use_guidance_general());
        let empty = PromptContext::default();
        let with_tool = PromptContext::default().with_tools(vec!["x".into()]);
        assert_eq!(b.build(&empty), "");
        assert!(b.build(&with_tool).contains("prefer calling a tool"));
    }

    #[test]
    fn preset_minimal_produces_no_fragments() {
        let b = PromptBuilder::new().with_preset(PromptPreset::Minimal);
        assert_eq!(b.fragment_count(), 0);
    }

    #[test]
    fn preset_code_developer_contains_expected_keys() {
        let b = PromptBuilder::new().with_preset(PromptPreset::CodeDeveloper);
        let keys: Vec<&str> = (0..b.fragment_count())
            .filter_map(|_| None)
            .collect::<Vec<_>>();
        // Use trace with a context that satisfies all conditions:
        let ctx = PromptContext::default().with_tools(vec!["git".into(), "x".into()]);
        let (_out, trace) = b.build_with_trace(&ctx);
        let trace_keys: Vec<&str> = trace.iter().map(|t| t.key).collect();
        assert!(trace_keys.contains(&"tool_use_guidance_general"));
        assert!(trace_keys.contains(&"tdd_workflow"));
        assert!(trace_keys.contains(&"git_commit_conventions"));
        assert!(trace_keys.contains(&"rust_idioms"));
        let _ = keys; // silence unused if compiler complains
    }

    #[test]
    fn preset_research_agent_contains_academic_citations() {
        let b = PromptBuilder::new().with_preset(PromptPreset::ResearchAgent);
        let ctx = PromptContext::default().with_tools(vec!["retrieve".into()]);
        let out = b.build(&ctx);
        assert!(out.contains("academic sources"));
        assert!(out.contains("Ground claims"));
    }

    #[test]
    fn override_fragment_from_preset() {
        let b = PromptBuilder::new()
            .with_preset(PromptPreset::ToolUseChatbot)
            .add_fragment(PromptFragment::always(
                "tool_use_guidance_general",
                "OVERRIDDEN",
                FragmentCategory::ToolGuidance,
                10,
            ));
        let ctx = PromptContext::default().with_tools(vec!["x".into()]);
        assert_eq!(b.build(&ctx), "OVERRIDDEN");
    }

    #[test]
    fn platform_detect_is_one_of_known_variants() {
        let p = Platform::detect();
        // Just ensures the call is stable and doesn't panic.
        matches!(
            p,
            Platform::Windows | Platform::Linux | Platform::MacOS | Platform::Other
        );
    }

    #[test]
    fn builder_is_cloneable() {
        let b = PromptBuilder::new().add_fragment(PromptFragment::always(
            "k",
            "v",
            FragmentCategory::Style,
            30,
        ));
        let cloned = b.clone();
        assert_eq!(cloned.fragment_count(), 1);
        assert_eq!(cloned.build(&ctx_default()), "v");
    }

    #[test]
    fn fragment_category_display_stable() {
        assert_eq!(FragmentCategory::Safety.to_string(), "Safety");
        assert_eq!(FragmentCategory::Domain.to_string(), "Domain");
    }

    #[test]
    fn custom_context_signals_drive_fragments() {
        let frag = PromptFragment::new(
            "feature_flag_fragment",
            "experimental mode",
            FragmentCategory::Context,
            25,
            |ctx| {
                ctx.custom
                    .get("experimental")
                    .map(|v| v == "on")
                    .unwrap_or(false)
            },
        );
        let b = PromptBuilder::new().add_fragment(frag);
        let off = PromptContext::default();
        let on = PromptContext::default().with_custom("experimental", "on");
        assert_eq!(b.build(&off), "");
        assert_eq!(b.build(&on), "experimental mode");
    }
}
