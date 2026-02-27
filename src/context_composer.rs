//! Context composition and orchestration
//!
//! This module provides tools for composing multi-section prompts within
//! a token budget, detecting overflow conditions, and applying hybrid
//! compaction strategies when the context window is exhausted.
//!
//! # Features
//!
//! - **Token budgeting**: Proportional allocation across named sections
//! - **Overflow detection**: Configurable thresholds (warning / critical / emergency)
//! - **Hybrid compaction**: Summarise and trim to keep within budget
//! - **Priority ordering**: Sections ordered by priority for prompt assembly
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::context_composer::{ContextComposer, ContextSection};
//! use std::collections::HashMap;
//!
//! let composer = ContextComposer::new(ContextComposer::default_config());
//! let mut sections = HashMap::new();
//! sections.insert(ContextSection::SystemPrompt, "You are a helpful assistant.".to_string());
//! sections.insert(ContextSection::UserPrompt, "Hello!".to_string());
//!
//! let composed = composer.compose(sections);
//! assert!(composed.total_tokens <= composed.total_budget);
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Logical sections of a composed context prompt.
///
/// Variants are ordered from highest to lowest priority so that derived
/// `Ord` gives a useful default ordering for prompt assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ContextSection {
    /// System prompt — always included, highest priority.
    SystemPrompt,
    /// Graph-derived context (knowledge graph neighbours, etc.).
    GraphContext,
    /// Retrieved RAG chunks.
    RagChunks,
    /// Long-term memory context.
    MemoryContext,
    /// Conversation history.
    Conversation,
    /// The current user prompt — always included.
    UserPrompt,
}

impl ContextSection {
    /// Return all variants in priority order.
    pub fn all() -> &'static [ContextSection] {
        &[
            ContextSection::SystemPrompt,
            ContextSection::GraphContext,
            ContextSection::RagChunks,
            ContextSection::MemoryContext,
            ContextSection::Conversation,
            ContextSection::UserPrompt,
        ]
    }
}

/// Priority class that governs how budget is allocated and whether a
/// section may be trimmed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectionPriority {
    /// Never trimmed — always gets its full content.
    Fixed,
    /// Trimmed last; gets preferential surplus redistribution.
    High,
    /// Default trimming behaviour.
    Normal,
    /// Trimmed first when budget is tight.
    Low,
}

/// Budget configuration for a single section.
#[derive(Debug, Clone)]
pub struct SectionBudget {
    /// Which section this budget applies to.
    pub section: ContextSection,
    /// Priority class.
    pub priority: SectionPriority,
    /// Percentage of the *available* budget (0.0 .. 1.0).
    pub percentage: f64,
    /// Minimum tokens that must be allocated (floor).
    pub min_tokens: usize,
    /// Hard upper-bound on allocation, if any.
    pub max_tokens: Option<usize>,
}

/// Thresholds that map a usage fraction to an [`OverflowLevel`].
#[derive(Debug, Clone)]
pub struct OverflowThresholds {
    /// Fraction at which a warning is raised (default 0.70).
    pub warning: f64,
    /// Fraction at which the situation is critical (default 0.85).
    pub critical: f64,
    /// Fraction at which emergency action is needed (default 0.95).
    pub emergency: f64,
}

impl Default for OverflowThresholds {
    fn default() -> Self {
        Self {
            warning: 0.70,
            critical: 0.85,
            emergency: 0.95,
        }
    }
}

/// Severity of a context-window overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OverflowLevel {
    /// Usage is within comfortable limits.
    Normal,
    /// Usage is approaching the budget.
    Warning,
    /// Usage is high — compaction recommended.
    Critical,
    /// Usage is at/near the hard limit.
    Emergency,
}

/// Recommended action for a given overflow level.
#[derive(Debug, Clone)]
pub enum OverflowAction {
    /// No action required.
    None,
    /// Compact (summarise / trim) to reduce usage.
    Compact {
        /// Target percentage of tokens to shed (e.g. 0.20 = 20%).
        target_reduction_pct: f64,
    },
    /// Create a checkpoint summary before continuing.
    Checkpoint {
        /// Maximum tokens the summary should occupy.
        summary_max_tokens: usize,
    },
    /// The context is beyond recovery — start a new session.
    NewSession {
        /// Human-readable reason for the reset.
        reason: String,
    },
}

/// Top-level configuration for [`ContextComposer`].
#[derive(Debug, Clone)]
pub struct ContextComposerConfig {
    /// Total token budget for the entire prompt (input side).
    pub total_budget: usize,
    /// Tokens reserved for the model's response.
    pub response_reserve: usize,
    /// Per-section budget configuration.
    pub section_budgets: Vec<SectionBudget>,
    /// Whether overflow detection is enabled.
    pub overflow_detection: bool,
    /// Thresholds for overflow detection.
    pub overflow_thresholds: OverflowThresholds,
    /// Whether hybrid compaction (summarise + trim) is enabled.
    pub hybrid_compaction: bool,
}

/// Result of budget allocation for a single section.
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// Section this allocation applies to.
    pub section: ContextSection,
    /// Tokens allocated by the budgeter.
    pub allocated_tokens: usize,
    /// Tokens actually consumed by the section content.
    pub actual_tokens: usize,
    /// Surplus = allocated - actual.  Positive means unused headroom.
    pub surplus: i64,
}

/// One section of the final composed prompt.
#[derive(Debug, Clone)]
pub struct ComposedSection {
    /// Section identifier.
    pub section: ContextSection,
    /// Content (possibly trimmed).
    pub content: String,
    /// Token count of `content`.
    pub tokens: usize,
    /// `true` if the content was trimmed to fit its budget.
    pub trimmed: bool,
}

/// The fully assembled context ready for submission to a model.
#[derive(Debug, Clone)]
pub struct ComposedContext {
    /// Ordered sections.
    pub sections: Vec<ComposedSection>,
    /// Total tokens across all sections.
    pub total_tokens: usize,
    /// The budget that was targeted.
    pub total_budget: usize,
    /// Current overflow level.
    pub overflow_level: OverflowLevel,
    /// Recommended action (if any).
    pub overflow_action: OverflowAction,
}

impl ComposedContext {
    /// Concatenate all sections into a single string, ordered by
    /// [`ContextSection`] priority.
    pub fn to_composed_string(&self) -> String {
        let mut sorted: Vec<&ComposedSection> = self.sections.iter().collect();
        sorted.sort_by_key(|s| s.section);

        let mut out = String::new();
        for (i, sec) in sorted.iter().enumerate() {
            if i > 0 && !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&sec.content);
        }
        out
    }

    /// Fraction of the budget currently used (0.0 .. 1.0+).
    pub fn usage_fraction(&self) -> f64 {
        if self.total_budget == 0 {
            return if self.total_tokens == 0 { 0.0 } else { 1.0 };
        }
        self.total_tokens as f64 / self.total_budget as f64
    }
}

// ---------------------------------------------------------------------------
// Token estimation helper
// ---------------------------------------------------------------------------

/// Rough token estimate: ~4 characters per token, rounded up.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

// ---------------------------------------------------------------------------
// Mini-summary helper (hybrid compaction)
// ---------------------------------------------------------------------------

/// Generate a very short summary of a sequence of `(role, content)` messages.
///
/// This is a *local* heuristic — it does not call an LLM.  It extracts the
/// first sentence of each message, prioritising user messages, and truncates
/// to fit within `max_tokens` (estimated).
pub fn generate_mini_summary(messages: &[(String, String)], max_tokens: usize) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let max_chars = max_tokens * 4; // inverse of estimate_tokens
    let mut summary = String::from("[Summary] ");

    for (role, content) in messages {
        if summary.len() >= max_chars {
            break;
        }
        let first_sentence = content
            .split_terminator(|c| c == '.' || c == '!' || c == '?')
            .next()
            .unwrap_or(content);

        let trimmed = first_sentence.trim();
        if trimmed.is_empty() {
            continue;
        }

        let remaining = max_chars.saturating_sub(summary.len());
        if remaining < 10 {
            break;
        }

        summary.push_str(role);
        summary.push_str(": ");
        if trimmed.len() <= remaining.saturating_sub(role.len() + 4) {
            summary.push_str(trimmed);
        } else {
            let limit = remaining.saturating_sub(role.len() + 7);
            if limit > 0 && limit <= trimmed.len() {
                summary.push_str(&trimmed[..limit]);
                summary.push_str("...");
            } else if limit > 0 {
                summary.push_str(trimmed);
            }
        }
        summary.push_str(". ");
    }

    // Final truncation safety-net
    if estimate_tokens(&summary) > max_tokens && summary.len() > max_chars {
        summary.truncate(max_chars);
    }

    summary.trim_end().to_string()
}

// ---------------------------------------------------------------------------
// TokenBudgetAllocator
// ---------------------------------------------------------------------------

/// Proportional token allocator with surplus redistribution.
#[derive(Debug)]
pub struct TokenBudgetAllocator;

impl TokenBudgetAllocator {
    /// Allocate budget to each section based on its configured percentage,
    /// then redistribute surplus from sections that are under-budget to
    /// sections that need more room.
    pub fn allocate(
        config: &ContextComposerConfig,
        contents: &HashMap<ContextSection, String>,
    ) -> Vec<BudgetAllocation> {
        let available = config.total_budget.saturating_sub(config.response_reserve);

        // Phase 1: initial proportional allocation
        let mut allocations: Vec<BudgetAllocation> = config
            .section_budgets
            .iter()
            .map(|sb| {
                let content_tokens = contents
                    .get(&sb.section)
                    .map(|c| estimate_tokens(c))
                    .unwrap_or(0);

                let mut allocated = (available as f64 * sb.percentage).round() as usize;
                allocated = allocated.max(sb.min_tokens);
                if let Some(max) = sb.max_tokens {
                    allocated = allocated.min(max);
                }

                // Fixed sections: allocate exactly what they need
                if sb.priority == SectionPriority::Fixed {
                    allocated = content_tokens.max(sb.min_tokens);
                    if let Some(max) = sb.max_tokens {
                        allocated = allocated.min(max);
                    }
                }

                BudgetAllocation {
                    section: sb.section,
                    allocated_tokens: allocated,
                    actual_tokens: content_tokens,
                    surplus: allocated as i64 - content_tokens as i64,
                }
            })
            .collect();

        // Phase 2: redistribute unused budget
        // Total allocated may be less than available (e.g., Fixed sections using less
        // than their proportional share), creating "freed" budget to redistribute.
        let total_allocated: usize = allocations.iter().map(|a| a.allocated_tokens).sum();
        let freed_budget = available.saturating_sub(total_allocated);

        // Also count explicit surplus from under-used sections
        let explicit_surplus: usize = allocations
            .iter()
            .filter(|a| a.surplus > 0)
            .map(|a| a.surplus as usize)
            .sum();

        let total_surplus = freed_budget + explicit_surplus;

        if total_surplus == 0 {
            return allocations;
        }

        // Sections that need more space
        let deficit_indices: Vec<usize> = allocations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.surplus < 0)
            .map(|(i, _)| i)
            .collect();

        if deficit_indices.is_empty() {
            return allocations;
        }

        let total_deficit: usize = deficit_indices
            .iter()
            .map(|&i| (-allocations[i].surplus) as usize)
            .sum();

        let redistributable = total_surplus.min(total_deficit);

        for &idx in &deficit_indices {
            let need = (-allocations[idx].surplus) as usize;
            let share = if total_deficit > 0 {
                (need as f64 / total_deficit as f64 * redistributable as f64).round() as usize
            } else {
                0
            };
            allocations[idx].allocated_tokens += share;

            // Re-enforce max_tokens cap after redistribution
            if let Some(sb) = config.section_budgets.iter().find(|sb| sb.section == allocations[idx].section) {
                if let Some(max) = sb.max_tokens {
                    allocations[idx].allocated_tokens = allocations[idx].allocated_tokens.min(max);
                }
            }

            allocations[idx].surplus = allocations[idx].allocated_tokens as i64
                - allocations[idx].actual_tokens as i64;
        }

        // Reduce surplus from donors proportionally
        let surplus_indices: Vec<usize> = allocations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.surplus > 0)
            .map(|(i, _)| i)
            .collect();

        let original_surplus: usize = surplus_indices
            .iter()
            .map(|&i| allocations[i].surplus.max(0) as usize)
            .sum();

        if original_surplus > 0 {
            for &idx in &surplus_indices {
                let old = allocations[idx].surplus.max(0) as usize;
                let give = if original_surplus > 0 {
                    (old as f64 / original_surplus as f64 * redistributable as f64).round() as usize
                } else {
                    0
                };
                allocations[idx].allocated_tokens =
                    allocations[idx].allocated_tokens.saturating_sub(give);
                allocations[idx].surplus = allocations[idx].allocated_tokens as i64
                    - allocations[idx].actual_tokens as i64;
            }
        }

        allocations
    }
}

// ---------------------------------------------------------------------------
// ContextOverflowDetector
// ---------------------------------------------------------------------------

/// Monitors token usage against a budget and recommends actions.
pub struct ContextOverflowDetector {
    budget: usize,
    thresholds: OverflowThresholds,
    on_overflow_callback: Option<Box<dyn Fn(OverflowLevel, usize) + Send + Sync>>,
}

impl ContextOverflowDetector {
    /// Create a detector with the given total budget.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            thresholds: OverflowThresholds::default(),
            on_overflow_callback: None,
        }
    }

    /// Override the default thresholds.
    pub fn with_thresholds(mut self, thresholds: OverflowThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Register a callback that fires whenever overflow is detected
    /// (i.e. level >= Warning).
    pub fn on_overflow<F>(mut self, f: F) -> Self
    where
        F: Fn(OverflowLevel, usize) + Send + Sync + 'static,
    {
        self.on_overflow_callback = Some(Box::new(f));
        self
    }

    /// Check the overflow level for the given token count.
    pub fn check(&self, current_tokens: usize) -> OverflowLevel {
        if self.budget == 0 {
            return if current_tokens == 0 {
                OverflowLevel::Normal
            } else {
                OverflowLevel::Emergency
            };
        }

        let fraction = current_tokens as f64 / self.budget as f64;

        let level = if fraction >= self.thresholds.emergency {
            OverflowLevel::Emergency
        } else if fraction >= self.thresholds.critical {
            OverflowLevel::Critical
        } else if fraction >= self.thresholds.warning {
            OverflowLevel::Warning
        } else {
            OverflowLevel::Normal
        };

        // Fire callback for non-normal levels
        if level != OverflowLevel::Normal {
            if let Some(ref cb) = self.on_overflow_callback {
                cb(level, current_tokens);
            }
        }

        level
    }

    /// Suggest an [`OverflowAction`] for the given level and token usage.
    pub fn get_action(&self, level: OverflowLevel, current_tokens: usize) -> OverflowAction {
        match level {
            OverflowLevel::Normal => OverflowAction::None,
            OverflowLevel::Warning => OverflowAction::Compact {
                target_reduction_pct: 0.20,
            },
            OverflowLevel::Critical => {
                let summary_max = self.budget / 5;
                OverflowAction::Checkpoint {
                    summary_max_tokens: summary_max,
                }
            }
            OverflowLevel::Emergency => OverflowAction::NewSession {
                reason: format!(
                    "Context overflow: {} tokens used of {} budget",
                    current_tokens, self.budget
                ),
            },
        }
    }

    /// Derive a sensible input budget from a model's context window and
    /// the desired response reserve.
    pub fn auto_configure_budget(model_context_window: usize, response_reserve: usize) -> usize {
        model_context_window.saturating_sub(response_reserve)
    }
}

impl fmt::Debug for ContextOverflowDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextOverflowDetector")
            .field("budget", &self.budget)
            .field("thresholds", &self.thresholds)
            .field("on_overflow_callback", &"<...>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ContextComposer
// ---------------------------------------------------------------------------

/// Orchestrates context composition: budgeting, trimming, overflow
/// detection, and prompt assembly.
#[derive(Debug)]
pub struct ContextComposer {
    config: ContextComposerConfig,
}

impl ContextComposer {
    /// Create a composer with the given configuration.
    pub fn new(config: ContextComposerConfig) -> Self {
        Self { config }
    }

    /// Sensible default configuration.
    ///
    /// - total_budget = 4096
    /// - response_reserve = 1024
    /// - SystemPrompt: 10 % Fixed
    /// - GraphContext: 10 % Normal
    /// - RagChunks: 20 % High
    /// - MemoryContext: 10 % Normal
    /// - Conversation: 40 % High
    /// - UserPrompt: 10 % Fixed
    pub fn default_config() -> ContextComposerConfig {
        ContextComposerConfig {
            total_budget: 4096,
            response_reserve: 1024,
            section_budgets: vec![
                SectionBudget {
                    section: ContextSection::SystemPrompt,
                    priority: SectionPriority::Fixed,
                    percentage: 0.10,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::GraphContext,
                    priority: SectionPriority::Normal,
                    percentage: 0.10,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::RagChunks,
                    priority: SectionPriority::High,
                    percentage: 0.20,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::MemoryContext,
                    priority: SectionPriority::Normal,
                    percentage: 0.10,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::Conversation,
                    priority: SectionPriority::High,
                    percentage: 0.40,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::UserPrompt,
                    priority: SectionPriority::Fixed,
                    percentage: 0.10,
                    min_tokens: 0,
                    max_tokens: None,
                },
            ],
            overflow_detection: true,
            overflow_thresholds: OverflowThresholds::default(),
            hybrid_compaction: true,
        }
    }

    /// Build a configuration derived from a model's context window size.
    ///
    /// The budget is set to `window - reserve` and the section percentages
    /// follow the same defaults as [`default_config`](Self::default_config).
    pub fn with_model_context_window(window: usize, reserve: usize) -> Self {
        let mut config = Self::default_config();
        config.total_budget = window;
        config.response_reserve = reserve;
        Self { config }
    }

    /// Read-only access to the current configuration.
    pub fn config(&self) -> &ContextComposerConfig {
        &self.config
    }

    /// Compose sections into a [`ComposedContext`].
    ///
    /// 1. Allocate budgets via [`TokenBudgetAllocator`].
    /// 2. Trim sections that exceed their allocation.
    /// 3. Detect overflow.
    /// 4. Order sections by priority.
    pub fn compose(&self, sections: HashMap<ContextSection, String>) -> ComposedContext {
        let allocations = TokenBudgetAllocator::allocate(&self.config, &sections);
        let available = self
            .config
            .total_budget
            .saturating_sub(self.config.response_reserve);

        let mut composed_sections: Vec<ComposedSection> = Vec::new();

        for alloc in &allocations {
            let content = match sections.get(&alloc.section) {
                Some(c) => c.clone(),
                None => String::new(),
            };

            if content.is_empty() {
                continue;
            }

            let content_tokens = estimate_tokens(&content);

            let (final_content, trimmed) = if content_tokens > alloc.allocated_tokens {
                // Trim to fit: approximate char limit from token allocation
                let char_limit = alloc.allocated_tokens * 4;
                let truncated = if char_limit < content.len() {
                    let mut end = char_limit;
                    // Avoid splitting a multi-byte character
                    while end > 0 && !content.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}...", &content[..end])
                } else {
                    content
                };
                (truncated, true)
            } else {
                (content, false)
            };

            composed_sections.push(ComposedSection {
                section: alloc.section,
                content: final_content.clone(),
                tokens: estimate_tokens(&final_content),
                trimmed,
            });
        }

        // Sort by section priority order
        composed_sections.sort_by_key(|s| s.section);

        let total_tokens: usize = composed_sections.iter().map(|s| s.tokens).sum();

        // Overflow detection
        let (overflow_level, overflow_action) = if self.config.overflow_detection {
            let detector =
                ContextOverflowDetector::new(available).with_thresholds(self.config.overflow_thresholds.clone());
            let level = detector.check(total_tokens);
            let action = detector.get_action(level, total_tokens);
            (level, action)
        } else {
            (OverflowLevel::Normal, OverflowAction::None)
        };

        ComposedContext {
            sections: composed_sections,
            total_tokens,
            total_budget: available,
            overflow_level,
            overflow_action,
        }
    }

    /// Compose and also return the recommended overflow action separately.
    pub fn compose_with_overflow_check(
        &self,
        sections: HashMap<ContextSection, String>,
    ) -> (ComposedContext, OverflowAction) {
        let composed = self.compose(sections);
        let available = self
            .config
            .total_budget
            .saturating_sub(self.config.response_reserve);

        let detector =
            ContextOverflowDetector::new(available).with_thresholds(self.config.overflow_thresholds.clone());
        let level = detector.check(composed.total_tokens);
        let action = detector.get_action(level, composed.total_tokens);

        (composed, action)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ContextCompiler - token-budget-aware segment compiler with eviction
// ---------------------------------------------------------------------------

/// Allocates token budgets per context segment type and enforces limits through
/// priority-based eviction when the total approaches the model context window.
#[derive(Debug)]
pub struct ContextCompiler {
    total_budget: usize,
    allocations: HashMap<SegmentType, SegmentAllocation>,
    segments: Vec<ContextSegment>,
}

/// Segment type for the context compiler.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum SegmentType {
    System,
    Tools,
    Memory,
    Documents,
    Conversation,
    Examples,
}

/// Budget allocation for a single segment type.
#[derive(Debug)]
pub struct SegmentAllocation {
    /// Fraction of total budget (0.0 - 1.0).
    pub percentage: f64,
    /// Higher priority segments are kept longer during eviction.
    pub priority: u32,
    /// Maximum tokens computed from percentage * total_budget.
    pub max_tokens: usize,
    /// Tokens currently consumed by segments of this type.
    pub current_tokens: usize,
}

/// A single context segment with metadata for budget management.
#[derive(Debug)]
pub struct ContextSegment {
    /// The type/category of this segment.
    pub segment_type: SegmentType,
    /// The textual content.
    pub content: String,
    /// Estimated token count (chars / 4).
    pub token_estimate: usize,
    /// Priority score - higher means keep longer.
    pub priority: u32,
    /// Timestamp for recency-based eviction (epoch millis or monotonic).
    pub timestamp: u64,
}

/// Result of compiling segments within the budget.
#[derive(Debug)]
pub struct CompiledContext {
    /// Segments that survived eviction.
    pub segments: Vec<ContextSegment>,
    /// Total tokens across surviving segments.
    pub total_tokens: usize,
    /// Number of segments evicted to meet budget.
    pub evicted_count: usize,
    /// Fraction of total budget consumed (0.0 - 1.0+).
    pub utilization: f64,
}

impl ContextCompiler {
    /// Create a new compiler with default allocations:
    /// System 15%, Tools 20%, Memory 15%, Documents 30%, Conversation 20%.
    pub fn new(total_budget: usize) -> Self {
        let mut allocations = HashMap::new();
        let defaults = [
            (SegmentType::System, 0.15, 5u32),
            (SegmentType::Tools, 0.20, 4),
            (SegmentType::Memory, 0.15, 3),
            (SegmentType::Documents, 0.30, 2),
            (SegmentType::Conversation, 0.20, 3),
            (SegmentType::Examples, 0.00, 1),
        ];
        for (seg_type, pct, prio) in defaults {
            allocations.insert(
                seg_type,
                SegmentAllocation {
                    percentage: pct,
                    priority: prio,
                    max_tokens: (total_budget as f64 * pct).round() as usize,
                    current_tokens: 0,
                },
            );
        }
        Self {
            total_budget,
            allocations,
            segments: Vec::new(),
        }
    }

    /// Override the allocation for a specific segment type (builder pattern).
    pub fn with_allocation(mut self, segment_type: SegmentType, percentage: f64, priority: u32) -> Self {
        let max_tokens = (self.total_budget as f64 * percentage).round() as usize;
        self.allocations.insert(
            segment_type,
            SegmentAllocation {
                percentage,
                priority,
                max_tokens,
                current_tokens: 0,
            },
        );
        self
    }

    /// Add a segment to the compiler.
    pub fn add_segment(&mut self, segment: ContextSegment) {
        if let Some(alloc) = self.allocations.get_mut(&segment.segment_type) {
            alloc.current_tokens += segment.token_estimate;
        }
        self.segments.push(segment);
    }

    /// Compile segments: apply budgets and evict low-priority segments if over budget.
    pub fn compile(&mut self) -> CompiledContext {
        let mut evicted_count = 0;

        // Evict while total tokens exceed the total budget
        while self.total_tokens() > self.total_budget && !self.segments.is_empty() {
            self.evict_lowest_priority();
            evicted_count += 1;
        }

        // Also evict segments that exceed their per-type budget
        loop {
            let mut over_budget_type: Option<SegmentType> = None;
            for (seg_type, alloc) in &self.allocations {
                if alloc.current_tokens > alloc.max_tokens && alloc.max_tokens > 0 {
                    over_budget_type = Some(seg_type.clone());
                    break;
                }
            }
            match over_budget_type {
                Some(seg_type) => {
                    if let Some(idx) = self.find_lowest_priority_of_type(&seg_type) {
                        let removed = self.segments.remove(idx);
                        if let Some(alloc) = self.allocations.get_mut(&removed.segment_type) {
                            alloc.current_tokens = alloc.current_tokens.saturating_sub(removed.token_estimate);
                        }
                        evicted_count += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        let total_tokens = self.total_tokens();
        let utilization = self.utilization();

        let compiled_segments: Vec<ContextSegment> = self.segments.drain(..).collect();

        for alloc in self.allocations.values_mut() {
            alloc.current_tokens = 0;
        }

        CompiledContext {
            segments: compiled_segments,
            total_tokens,
            evicted_count,
            utilization,
        }
    }

    /// Total tokens across all current segments.
    pub fn total_tokens(&self) -> usize {
        self.segments.iter().map(|s| s.token_estimate).sum()
    }

    /// Budget utilization as a fraction (total_tokens / total_budget).
    pub fn utilization(&self) -> f64 {
        if self.total_budget == 0 {
            return if self.total_tokens() == 0 { 0.0 } else { 1.0 };
        }
        self.total_tokens() as f64 / self.total_budget as f64
    }

    /// Maximum tokens allocated for a given segment type.
    pub fn budget_for(&self, segment_type: &SegmentType) -> usize {
        self.allocations
            .get(segment_type)
            .map(|a| a.max_tokens)
            .unwrap_or(0)
    }

    /// Evict the segment with the lowest priority * recency score.
    fn evict_lowest_priority(&mut self) {
        if self.segments.is_empty() {
            return;
        }
        let mut worst_idx = 0;
        let mut worst_score = f64::MAX;

        for (i, seg) in self.segments.iter().enumerate() {
            let recency_factor = 1.0 + (1.0 + seg.timestamp as f64).log2();
            let score = seg.priority as f64 * recency_factor;
            if score < worst_score {
                worst_score = score;
                worst_idx = i;
            }
        }

        let removed = self.segments.remove(worst_idx);
        if let Some(alloc) = self.allocations.get_mut(&removed.segment_type) {
            alloc.current_tokens = alloc.current_tokens.saturating_sub(removed.token_estimate);
        }
    }

    /// Find the index of the lowest-priority segment of a given type.
    fn find_lowest_priority_of_type(&self, seg_type: &SegmentType) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::MAX;

        for (i, seg) in self.segments.iter().enumerate() {
            if &seg.segment_type == seg_type {
                let recency_factor = 1.0 + (1.0 + seg.timestamp as f64).log2();
                let score = seg.priority as f64 * recency_factor;
                if score < best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }

    /// Simple token estimation: characters / 4.
    #[allow(dead_code)]
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }
}

// ---------------------------------------------------------------------------
// ConversationCompactor - automatic conversation compaction
// ---------------------------------------------------------------------------

/// Automatically compacts long conversations by summarizing older messages
/// while preserving the most recent N messages intact.
#[derive(Debug)]
pub struct ConversationCompactor {
    max_messages_before_compact: usize,
    preserve_recent: usize,
    #[allow(dead_code)]
    summary_max_tokens: usize,
    compaction_count: usize,
}

/// Result of compacting a conversation.
pub struct CompactedConversation {
    /// Summary of the older (compacted) messages.
    pub summary: String,
    /// The most recent messages, preserved intact.
    pub preserved_messages: Vec<CompactableMessage>,
    /// Number of messages in the original conversation.
    pub original_count: usize,
    /// Number of messages that were compacted into the summary.
    pub compacted_count: usize,
}

/// A single message that can be compacted.
#[derive(Debug, Clone)]
pub struct CompactableMessage {
    /// Role of the speaker (e.g. user, assistant).
    pub role: String,
    /// Message content.
    pub content: String,
    /// Estimated token count.
    pub token_estimate: usize,
}

impl ConversationCompactor {
    /// Create a compactor.
    ///
    /// - `max_before_compact`: trigger compaction when message count exceeds this.
    /// - `preserve_recent`: always keep the last N messages unchanged.
    pub fn new(max_before_compact: usize, preserve_recent: usize) -> Self {
        Self {
            max_messages_before_compact: max_before_compact,
            preserve_recent,
            summary_max_tokens: 256,
            compaction_count: 0,
        }
    }

    /// Whether the given message count warrants compaction.
    pub fn needs_compaction(&self, message_count: usize) -> bool {
        message_count > self.max_messages_before_compact
    }

    /// Compact a list of messages.
    ///
    /// Messages beyond `preserve_recent` from the end are summarised.
    /// The most recent `preserve_recent` messages are returned intact.
    pub fn compact(&mut self, messages: &[CompactableMessage]) -> CompactedConversation {
        let total = messages.len();
        let preserve = self.preserve_recent.min(total);
        let compact_end = total.saturating_sub(preserve);

        let to_compact = &messages[..compact_end];
        let to_preserve = &messages[compact_end..];

        let summary = Self::summarize_messages(to_compact);
        self.compaction_count += 1;

        CompactedConversation {
            summary,
            preserved_messages: to_preserve.to_vec(),
            original_count: total,
            compacted_count: compact_end,
        }
    }

    /// Number of times compaction has been performed.
    pub fn compaction_count(&self) -> usize {
        self.compaction_count
    }

    /// Simple extractive summary: take the first sentence of each message.
    fn summarize_messages(messages: &[CompactableMessage]) -> String {
        if messages.is_empty() {
            return String::new();
        }

        let mut summary = String::from("[Conversation Summary] ");

        for msg in messages {
            let first_sentence = msg
                .content
                .split_terminator(|c: char| c == '.' || c == '!' || c == '?')
                .next()
                .unwrap_or(&msg.content);

            let trimmed = first_sentence.trim();
            if trimmed.is_empty() {
                continue;
            }

            summary.push_str(&msg.role);
            summary.push_str(": ");
            summary.push_str(trimmed);
            summary.push_str(". ");
        }

        summary.trim_end().to_string()
    }
}

// ---------------------------------------------------------------------------
// ToolSearchIndex - TF-IDF based tool selection
// ---------------------------------------------------------------------------

/// Uses TF-IDF to select the top-K most relevant tools for a given query,
/// avoiding injecting all tool definitions into the context window.
pub struct ToolSearchIndex {
    tools: Vec<IndexedTool>,
    idf_cache: HashMap<String, f64>,
    vocab_size: usize,
}

/// A tool indexed for search with pre-computed TF vector.
pub struct IndexedTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Pre-computed term-frequency vector.
    pub tf_vector: HashMap<String, f64>,
}

/// A single search result with relevance score.
pub struct ToolSearchResult {
    /// Name of the matching tool.
    pub tool_name: String,
    /// Cosine similarity score (0.0 - 1.0).
    pub relevance_score: f64,
}

/// Common English stop words to filter out during tokenization.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their",
];

impl ToolSearchIndex {
    /// Create an empty search index.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            idf_cache: HashMap::new(),
            vocab_size: 0,
        }
    }

    /// Add a tool to the index.
    pub fn add_tool(&mut self, name: &str, description: &str) {
        let combined = format!("{} {}", name, description);
        let tokens = Self::tokenize(&combined);
        let tf = Self::compute_tf(&tokens);

        self.tools.push(IndexedTool {
            name: name.to_string(),
            description: description.to_string(),
            tf_vector: tf,
        });

        self.rebuild_idf();
    }

    /// Recalculate IDF (inverse document frequency) for all terms across all tools.
    pub fn rebuild_idf(&mut self) {
        self.idf_cache.clear();
        let n = self.tools.len() as f64;
        if n == 0.0 {
            self.vocab_size = 0;
            return;
        }

        let mut all_terms: HashMap<String, usize> = HashMap::new();
        for tool in &self.tools {
            for term in tool.tf_vector.keys() {
                *all_terms.entry(term.clone()).or_insert(0) += 1;
            }
        }

        for (term, doc_count) in &all_terms {
            let idf = (n / (*doc_count as f64)).ln();
            self.idf_cache.insert(term.clone(), idf);
        }

        self.vocab_size = all_terms.len();
    }

    /// Search for the top-K tools most relevant to the query.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<ToolSearchResult> {
        if self.tools.is_empty() {
            return Vec::new();
        }

        let query_tokens = Self::tokenize(query);
        let query_tf = Self::compute_tf(&query_tokens);

        let query_tfidf: HashMap<String, f64> = query_tf
            .iter()
            .map(|(term, tf)| {
                let idf = self.idf_cache.get(term).copied().unwrap_or(0.0);
                (term.clone(), tf * idf)
            })
            .collect();

        let mut results: Vec<ToolSearchResult> = self
            .tools
            .iter()
            .map(|tool| {
                let tool_tfidf: HashMap<String, f64> = tool
                    .tf_vector
                    .iter()
                    .map(|(term, tf)| {
                        let idf = self.idf_cache.get(term).copied().unwrap_or(0.0);
                        (term.clone(), tf * idf)
                    })
                    .collect();

                let score = Self::cosine_similarity(&query_tfidf, &tool_tfidf);
                ToolSearchResult {
                    tool_name: tool.name.clone(),
                    relevance_score: score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Number of tools in the index.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Tokenize text: lowercase, split on whitespace/punctuation, filter stop words.
    fn tokenize(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() > 1)
            .filter(|s| !STOP_WORDS.contains(s))
            .map(|s| s.to_string())
            .collect()
    }

    /// Compute term frequency for a list of tokens.
    fn compute_tf(tokens: &[String]) -> HashMap<String, f64> {
        if tokens.is_empty() {
            return HashMap::new();
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        let total = tokens.len() as f64;
        counts.into_iter().map(|(k, v)| (k, v as f64 / total)).collect()
    }

    /// Cosine similarity between two sparse TF-IDF vectors.
    fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        let dot: f64 = a
            .iter()
            .map(|(k, v)| v * b.get(k).unwrap_or(&0.0))
            .sum();

        let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // -- Ordering tests ----------------------------------------------------

    #[test]
    fn test_context_section_ordering() {
        assert!(ContextSection::SystemPrompt < ContextSection::GraphContext);
        assert!(ContextSection::GraphContext < ContextSection::RagChunks);
        assert!(ContextSection::RagChunks < ContextSection::MemoryContext);
        assert!(ContextSection::MemoryContext < ContextSection::Conversation);
        assert!(ContextSection::Conversation < ContextSection::UserPrompt);
    }

    #[test]
    fn test_overflow_level_ordering() {
        assert!(OverflowLevel::Normal < OverflowLevel::Warning);
        assert!(OverflowLevel::Warning < OverflowLevel::Critical);
        assert!(OverflowLevel::Critical < OverflowLevel::Emergency);
    }

    // -- OverflowThresholds ------------------------------------------------

    #[test]
    fn test_overflow_thresholds_defaults() {
        let t = OverflowThresholds::default();
        assert!((t.warning - 0.70).abs() < f64::EPSILON);
        assert!((t.critical - 0.85).abs() < f64::EPSILON);
        assert!((t.emergency - 0.95).abs() < f64::EPSILON);
    }

    // -- OverflowDetector --------------------------------------------------

    #[test]
    fn test_overflow_detector_normal() {
        let d = ContextOverflowDetector::new(1000);
        assert_eq!(d.check(500), OverflowLevel::Normal);
    }

    #[test]
    fn test_overflow_detector_warning() {
        let d = ContextOverflowDetector::new(1000);
        assert_eq!(d.check(750), OverflowLevel::Warning);
    }

    #[test]
    fn test_overflow_detector_critical() {
        let d = ContextOverflowDetector::new(1000);
        assert_eq!(d.check(880), OverflowLevel::Critical);
    }

    #[test]
    fn test_overflow_detector_emergency() {
        let d = ContextOverflowDetector::new(1000);
        assert_eq!(d.check(960), OverflowLevel::Emergency);
    }

    #[test]
    fn test_overflow_detector_action() {
        let d = ContextOverflowDetector::new(1000);
        assert!(matches!(
            d.get_action(OverflowLevel::Normal, 100),
            OverflowAction::None
        ));
        assert!(matches!(
            d.get_action(OverflowLevel::Warning, 750),
            OverflowAction::Compact { .. }
        ));
        assert!(matches!(
            d.get_action(OverflowLevel::Critical, 880),
            OverflowAction::Checkpoint { .. }
        ));
        assert!(matches!(
            d.get_action(OverflowLevel::Emergency, 980),
            OverflowAction::NewSession { .. }
        ));
    }

    #[test]
    fn test_overflow_detector_callback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let d = ContextOverflowDetector::new(1000)
            .on_overflow(move |_level, _tokens| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });

        d.check(500); // Normal — no callback
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        d.check(800); // Warning — callback fires
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        d.check(900); // Critical — callback fires again
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_overflow_detector_zero_budget() {
        let d = ContextOverflowDetector::new(0);
        assert_eq!(d.check(0), OverflowLevel::Normal);
        assert_eq!(d.check(1), OverflowLevel::Emergency);
    }

    #[test]
    fn test_overflow_detector_custom_thresholds() {
        let d = ContextOverflowDetector::new(100).with_thresholds(OverflowThresholds {
            warning: 0.50,
            critical: 0.75,
            emergency: 0.90,
        });
        assert_eq!(d.check(40), OverflowLevel::Normal);
        assert_eq!(d.check(55), OverflowLevel::Warning);
        assert_eq!(d.check(80), OverflowLevel::Critical);
        assert_eq!(d.check(95), OverflowLevel::Emergency);
    }

    #[test]
    fn test_auto_configure_budget() {
        assert_eq!(
            ContextOverflowDetector::auto_configure_budget(8192, 2048),
            6144
        );
        assert_eq!(
            ContextOverflowDetector::auto_configure_budget(1000, 2000),
            0
        );
    }

    // -- TokenBudgetAllocator ----------------------------------------------

    #[test]
    fn test_allocator_basic() {
        let config = ContextComposer::default_config();
        let mut contents = HashMap::new();
        contents.insert(ContextSection::SystemPrompt, "Hello".to_string());
        contents.insert(ContextSection::UserPrompt, "World".to_string());

        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        assert!(!allocs.is_empty());

        // Every section in config should appear in allocations
        assert_eq!(allocs.len(), config.section_budgets.len());
    }

    #[test]
    fn test_allocator_surplus_redistribution() {
        let config = ContextComposerConfig {
            total_budget: 1000,
            response_reserve: 0,
            section_budgets: vec![
                SectionBudget {
                    section: ContextSection::SystemPrompt,
                    priority: SectionPriority::Fixed,
                    percentage: 0.50,
                    min_tokens: 0,
                    max_tokens: None,
                },
                SectionBudget {
                    section: ContextSection::Conversation,
                    priority: SectionPriority::High,
                    percentage: 0.50,
                    min_tokens: 0,
                    max_tokens: None,
                },
            ],
            overflow_detection: false,
            overflow_thresholds: OverflowThresholds::default(),
            hybrid_compaction: false,
        };

        let mut contents = HashMap::new();
        // SystemPrompt uses only 10 tokens but gets 500 allocated (Fixed adjusts)
        contents.insert(ContextSection::SystemPrompt, "tiny".to_string());
        // Conversation needs a lot
        contents.insert(
            ContextSection::Conversation,
            "x".repeat(3000), // ~750 tokens
        );

        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        // Conversation should have gotten surplus from SystemPrompt
        let conv = allocs.iter().find(|a| a.section == ContextSection::Conversation);
        assert!(conv.is_some());
        let conv = conv.expect("conversation allocation missing");
        // It should have more than its initial 500 allocation
        assert!(conv.allocated_tokens > 500);
    }

    #[test]
    fn test_allocator_fixed_sections() {
        let config = ContextComposer::default_config();
        let mut contents = HashMap::new();
        let sys = "System prompt text here.";
        contents.insert(ContextSection::SystemPrompt, sys.to_string());

        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        let sys_alloc = allocs
            .iter()
            .find(|a| a.section == ContextSection::SystemPrompt)
            .expect("system prompt allocation missing");

        // Fixed section: allocated should equal actual (content tokens)
        assert_eq!(sys_alloc.allocated_tokens, estimate_tokens(sys));
    }

    // -- Compose -----------------------------------------------------------

    #[test]
    fn test_compose_with_all_sections() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let mut sections = HashMap::new();
        sections.insert(
            ContextSection::SystemPrompt,
            "You are a helpful assistant.".to_string(),
        );
        sections.insert(
            ContextSection::GraphContext,
            "Entity: Rust. Related: systems programming.".to_string(),
        );
        sections.insert(
            ContextSection::RagChunks,
            "Chunk 1: Rust is a systems language.".to_string(),
        );
        sections.insert(
            ContextSection::MemoryContext,
            "User prefers concise answers.".to_string(),
        );
        sections.insert(
            ContextSection::Conversation,
            "user: Hello\nassistant: Hi!".to_string(),
        );
        sections.insert(ContextSection::UserPrompt, "Tell me about Rust.".to_string());

        let composed = composer.compose(sections);
        assert_eq!(composed.sections.len(), 6);
        assert!(composed.total_tokens > 0);
        assert!(composed.total_tokens <= composed.total_budget);
        assert_eq!(composed.overflow_level, OverflowLevel::Normal);
    }

    #[test]
    fn test_compose_with_missing_sections() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let mut sections = HashMap::new();
        sections.insert(ContextSection::SystemPrompt, "System.".to_string());
        sections.insert(ContextSection::UserPrompt, "Hello.".to_string());

        let composed = composer.compose(sections);
        // Only non-empty sections are included
        assert_eq!(composed.sections.len(), 2);
    }

    #[test]
    fn test_compose_overflow_large() {
        let mut config = ContextComposer::default_config();
        config.total_budget = 100;
        config.response_reserve = 0;
        let composer = ContextComposer::new(config);

        let mut sections = HashMap::new();
        // 2000 chars ~ 500 tokens, budget = 100
        sections.insert(ContextSection::Conversation, "x".repeat(2000));
        sections.insert(ContextSection::UserPrompt, "Hi".to_string());

        let composed = composer.compose(sections);
        // Should detect overflow
        assert!(composed.overflow_level >= OverflowLevel::Warning);
    }

    #[test]
    fn test_composed_context_to_string_ordering() {
        let ctx = ComposedContext {
            sections: vec![
                ComposedSection {
                    section: ContextSection::UserPrompt,
                    content: "user prompt".to_string(),
                    tokens: 3,
                    trimmed: false,
                },
                ComposedSection {
                    section: ContextSection::SystemPrompt,
                    content: "system".to_string(),
                    tokens: 2,
                    trimmed: false,
                },
            ],
            total_tokens: 5,
            total_budget: 100,
            overflow_level: OverflowLevel::Normal,
            overflow_action: OverflowAction::None,
        };

        let s = ctx.to_composed_string();
        // SystemPrompt should come before UserPrompt
        let sys_pos = s.find("system").expect("system not found");
        let usr_pos = s.find("user prompt").expect("user prompt not found");
        assert!(sys_pos < usr_pos);
    }

    #[test]
    fn test_composed_context_usage_fraction() {
        let ctx = ComposedContext {
            sections: vec![],
            total_tokens: 500,
            total_budget: 1000,
            overflow_level: OverflowLevel::Normal,
            overflow_action: OverflowAction::None,
        };
        assert!((ctx.usage_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composed_context_usage_fraction_zero_budget() {
        let ctx = ComposedContext {
            sections: vec![],
            total_tokens: 0,
            total_budget: 0,
            overflow_level: OverflowLevel::Normal,
            overflow_action: OverflowAction::None,
        };
        assert!((ctx.usage_fraction() - 0.0).abs() < f64::EPSILON);

        let ctx2 = ComposedContext {
            sections: vec![],
            total_tokens: 10,
            total_budget: 0,
            overflow_level: OverflowLevel::Emergency,
            overflow_action: OverflowAction::None,
        };
        assert!((ctx2.usage_fraction() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composer_with_model_context_window() {
        let composer = ContextComposer::with_model_context_window(8192, 2048);
        assert_eq!(composer.config().total_budget, 8192);
        assert_eq!(composer.config().response_reserve, 2048);
    }

    #[test]
    fn test_compose_with_overflow_check() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let mut sections = HashMap::new();
        sections.insert(ContextSection::UserPrompt, "Hello".to_string());

        let (composed, action) = composer.compose_with_overflow_check(sections);
        assert_eq!(composed.overflow_level, OverflowLevel::Normal);
        assert!(matches!(action, OverflowAction::None));
    }

    // -- generate_mini_summary ---------------------------------------------

    #[test]
    fn test_generate_mini_summary() {
        let messages = vec![
            ("user".to_string(), "Tell me about Rust programming.".to_string()),
            (
                "assistant".to_string(),
                "Rust is a systems programming language. It focuses on safety.".to_string(),
            ),
        ];
        let summary = generate_mini_summary(&messages, 50);
        assert!(!summary.is_empty());
        assert!(summary.contains("[Summary]"));
        assert!(estimate_tokens(&summary) <= 60); // some slack for formatting
    }

    #[test]
    fn test_generate_mini_summary_empty() {
        let summary = generate_mini_summary(&[], 100);
        assert!(summary.is_empty());
    }

    // -- estimate_tokens ---------------------------------------------------

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // "Hi" = 2 chars => (2+3)/4 = 1
        assert_eq!(estimate_tokens("Hi"), 1);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        let text = "a".repeat(100);
        assert_eq!(estimate_tokens(&text), (100 + 3) / 4);
    }

    // -- BudgetAllocation surplus ------------------------------------------

    #[test]
    fn test_budget_allocation_surplus() {
        let alloc = BudgetAllocation {
            section: ContextSection::SystemPrompt,
            allocated_tokens: 100,
            actual_tokens: 60,
            surplus: 40,
        };
        assert_eq!(alloc.surplus, 40);
    }

    #[test]
    fn test_budget_allocation_deficit() {
        let alloc = BudgetAllocation {
            section: ContextSection::Conversation,
            allocated_tokens: 100,
            actual_tokens: 150,
            surplus: -50,
        };
        assert_eq!(alloc.surplus, -50);
    }

    // -- default_config ----------------------------------------------------

    #[test]
    fn test_default_config_has_all_sections() {
        let config = ContextComposer::default_config();
        let sections: Vec<ContextSection> =
            config.section_budgets.iter().map(|sb| sb.section).collect();
        for s in ContextSection::all() {
            assert!(sections.contains(s), "Missing section: {:?}", s);
        }
    }

    #[test]
    fn test_default_config_percentages_sum() {
        let config = ContextComposer::default_config();
        let sum: f64 = config.section_budgets.iter().map(|sb| sb.percentage).sum();
        assert!((sum - 1.0).abs() < 0.01, "Percentages sum to {}, not 1.0", sum);
    }

    #[test]
    fn test_default_config_budget_values() {
        let config = ContextComposer::default_config();
        assert_eq!(config.total_budget, 4096);
        assert_eq!(config.response_reserve, 1024);
        assert!(config.overflow_detection);
        assert!(config.hybrid_compaction);
    }

    // -- ComposedSection trimmed flag --------------------------------------

    #[test]
    fn test_composed_section_trimmed_flag() {
        let mut config = ContextComposer::default_config();
        config.total_budget = 50;
        config.response_reserve = 0;
        let composer = ContextComposer::new(config);

        let mut sections = HashMap::new();
        // Very large conversation, tiny budget
        sections.insert(ContextSection::Conversation, "word ".repeat(500));
        sections.insert(ContextSection::UserPrompt, "Hi".to_string());

        let composed = composer.compose(sections);
        let conv = composed
            .sections
            .iter()
            .find(|s| s.section == ContextSection::Conversation);
        assert!(conv.is_some());
        assert!(conv.expect("conversation section missing").trimmed);
    }

    // -- Section priority and equality -------------------------------------

    #[test]
    fn test_section_priority_fixed_eq() {
        assert_eq!(SectionPriority::Fixed, SectionPriority::Fixed);
        assert_ne!(SectionPriority::Fixed, SectionPriority::High);
    }

    #[test]
    fn test_context_section_clone_eq() {
        let s = ContextSection::RagChunks;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn test_overflow_level_clone_eq() {
        let l = OverflowLevel::Critical;
        let l2 = l;
        assert_eq!(l, l2);
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_compose_empty_sections() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let sections = HashMap::new();
        let composed = composer.compose(sections);
        assert_eq!(composed.sections.len(), 0);
        assert_eq!(composed.total_tokens, 0);
        assert_eq!(composed.overflow_level, OverflowLevel::Normal);
    }

    #[test]
    fn test_compose_single_section() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let mut sections = HashMap::new();
        sections.insert(ContextSection::UserPrompt, "Hello world".to_string());
        let composed = composer.compose(sections);
        assert_eq!(composed.sections.len(), 1);
        assert_eq!(composed.sections[0].section, ContextSection::UserPrompt);
    }

    #[test]
    fn test_allocator_all_empty_contents() {
        let config = ContextComposer::default_config();
        let contents = HashMap::new();
        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        // All sections allocated, all actual_tokens = 0
        for alloc in &allocs {
            assert_eq!(alloc.actual_tokens, 0);
        }
    }

    #[test]
    fn test_overflow_action_compact_fields() {
        let action = OverflowAction::Compact {
            target_reduction_pct: 0.20,
        };
        if let OverflowAction::Compact {
            target_reduction_pct,
        } = action
        {
            assert!((target_reduction_pct - 0.20).abs() < f64::EPSILON);
        } else {
            panic!("Expected Compact variant");
        }
    }

    #[test]
    fn test_overflow_action_checkpoint_fields() {
        let action = OverflowAction::Checkpoint {
            summary_max_tokens: 256,
        };
        if let OverflowAction::Checkpoint { summary_max_tokens } = action {
            assert_eq!(summary_max_tokens, 256);
        } else {
            panic!("Expected Checkpoint variant");
        }
    }

    #[test]
    fn test_overflow_action_new_session_fields() {
        let action = OverflowAction::NewSession {
            reason: "too big".to_string(),
        };
        if let OverflowAction::NewSession { reason } = action {
            assert_eq!(reason, "too big");
        } else {
            panic!("Expected NewSession variant");
        }
    }

    #[test]
    fn test_context_section_all_length() {
        assert_eq!(ContextSection::all().len(), 6);
    }

    #[test]
    fn test_section_budget_min_tokens_respected() {
        let config = ContextComposerConfig {
            total_budget: 100,
            response_reserve: 0,
            section_budgets: vec![SectionBudget {
                section: ContextSection::Conversation,
                priority: SectionPriority::Normal,
                percentage: 0.10,
                min_tokens: 50,
                max_tokens: None,
            }],
            overflow_detection: false,
            overflow_thresholds: OverflowThresholds::default(),
            hybrid_compaction: false,
        };

        let mut contents = HashMap::new();
        contents.insert(ContextSection::Conversation, "abc".to_string());
        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        // min_tokens = 50, but 10% of 100 = 10; min wins
        assert!(allocs[0].allocated_tokens >= 50);
    }

    #[test]
    fn test_section_budget_max_tokens_respected() {
        let config = ContextComposerConfig {
            total_budget: 10000,
            response_reserve: 0,
            section_budgets: vec![SectionBudget {
                section: ContextSection::Conversation,
                priority: SectionPriority::Normal,
                percentage: 0.90,
                min_tokens: 0,
                max_tokens: Some(100),
            }],
            overflow_detection: false,
            overflow_thresholds: OverflowThresholds::default(),
            hybrid_compaction: false,
        };

        let mut contents = HashMap::new();
        contents.insert(ContextSection::Conversation, "x".repeat(4000));
        let allocs = TokenBudgetAllocator::allocate(&config, &contents);
        assert!(allocs[0].allocated_tokens <= 100);
    }

    #[test]
    fn test_generate_mini_summary_single_message() {
        let messages = vec![("user".to_string(), "Hello there friend.".to_string())];
        let summary = generate_mini_summary(&messages, 50);
        assert!(summary.contains("user"));
        assert!(summary.contains("Hello there friend"));
    }

    #[test]
    fn test_generate_mini_summary_very_small_budget() {
        let messages = vec![
            ("user".to_string(), "A very long sentence that goes on and on.".to_string()),
        ];
        let summary = generate_mini_summary(&messages, 5);
        // Should still produce something (possibly truncated)
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_composed_string_empty_sections() {
        let ctx = ComposedContext {
            sections: vec![],
            total_tokens: 0,
            total_budget: 100,
            overflow_level: OverflowLevel::Normal,
            overflow_action: OverflowAction::None,
        };
        assert!(ctx.to_composed_string().is_empty());
    }

    #[test]
    fn test_overflow_detector_exactly_at_threshold() {
        let d = ContextOverflowDetector::new(100);
        // Exactly at warning (70%)
        assert_eq!(d.check(70), OverflowLevel::Warning);
        // Exactly at critical (85%)
        assert_eq!(d.check(85), OverflowLevel::Critical);
        // Exactly at emergency (95%)
        assert_eq!(d.check(95), OverflowLevel::Emergency);
    }

    #[test]
    fn test_overflow_detector_action_checkpoint_budget() {
        let d = ContextOverflowDetector::new(1000);
        if let OverflowAction::Checkpoint { summary_max_tokens } =
            d.get_action(OverflowLevel::Critical, 880)
        {
            // budget / 5 = 200
            assert_eq!(summary_max_tokens, 200);
        } else {
            panic!("Expected Checkpoint action for Critical level");
        }
    }

    #[test]
    fn test_compose_preserves_content_for_small_sections() {
        let composer = ContextComposer::new(ContextComposer::default_config());
        let mut sections = HashMap::new();
        sections.insert(ContextSection::SystemPrompt, "Be helpful.".to_string());
        sections.insert(ContextSection::UserPrompt, "Hi.".to_string());

        let composed = composer.compose(sections);
        let sys = composed
            .sections
            .iter()
            .find(|s| s.section == ContextSection::SystemPrompt)
            .expect("system prompt section missing");

        assert_eq!(sys.content, "Be helpful.");
        assert!(!sys.trimmed);
    }

    #[test]
    fn test_compose_with_overflow_check_returns_action() {
        let mut config = ContextComposer::default_config();
        config.total_budget = 50;
        config.response_reserve = 0;
        let composer = ContextComposer::new(config);

        let mut sections = HashMap::new();
        sections.insert(ContextSection::Conversation, "x".repeat(1000));
        sections.insert(ContextSection::UserPrompt, "Hi".to_string());

        let (_composed, action) = composer.compose_with_overflow_check(sections);
        // With such a tight budget, should get some overflow action
        assert!(!matches!(action, OverflowAction::None));
    }

    // -- ContextCompiler tests ---------------------------------------------

    #[test]
    fn test_context_compiler_new() {
        let compiler = ContextCompiler::new(4000);
        assert_eq!(compiler.total_budget, 4000);
        assert_eq!(compiler.total_tokens(), 0);
        assert!(compiler.segments.is_empty());
    }

    #[test]
    fn test_compiler_default_allocations() {
        let compiler = ContextCompiler::new(10000);
        assert_eq!(compiler.budget_for(&SegmentType::System), 1500);
        assert_eq!(compiler.budget_for(&SegmentType::Tools), 2000);
        assert_eq!(compiler.budget_for(&SegmentType::Memory), 1500);
        assert_eq!(compiler.budget_for(&SegmentType::Documents), 3000);
        assert_eq!(compiler.budget_for(&SegmentType::Conversation), 2000);
        assert_eq!(compiler.budget_for(&SegmentType::Examples), 0);
    }

    #[test]
    fn test_compiler_add_segment() {
        let mut compiler = ContextCompiler::new(4000);
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::System,
            content: "You are a helpful assistant.".to_string(),
            token_estimate: 7,
            priority: 5,
            timestamp: 100,
        });
        assert_eq!(compiler.segments.len(), 1);
        assert_eq!(compiler.total_tokens(), 7);
    }

    #[test]
    fn test_compiler_compile() {
        let mut compiler = ContextCompiler::new(4000);
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::System,
            content: "System prompt".to_string(),
            token_estimate: 50,
            priority: 5,
            timestamp: 100,
        });
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::Conversation,
            content: "User message".to_string(),
            token_estimate: 30,
            priority: 3,
            timestamp: 200,
        });

        let compiled = compiler.compile();
        assert_eq!(compiled.total_tokens, 80);
        assert_eq!(compiled.evicted_count, 0);
        assert_eq!(compiled.segments.len(), 2);
        assert!(compiled.utilization < 1.0);
    }

    #[test]
    fn test_compiler_utilization() {
        let mut compiler = ContextCompiler::new(100);
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::System,
            content: "test".to_string(),
            token_estimate: 50,
            priority: 5,
            timestamp: 1,
        });
        let util = compiler.utilization();
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compiler_eviction() {
        let mut compiler = ContextCompiler::new(100);
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::System,
            content: "high priority".to_string(),
            token_estimate: 60,
            priority: 10,
            timestamp: 100,
        });
        compiler.add_segment(ContextSegment {
            segment_type: SegmentType::Documents,
            content: "low priority".to_string(),
            token_estimate: 60,
            priority: 1,
            timestamp: 1,
        });
        let compiled = compiler.compile();
        assert!(compiled.evicted_count > 0);
        assert!(compiled.total_tokens <= 100);
    }

    #[test]
    fn test_compiler_custom_allocation() {
        let compiler = ContextCompiler::new(1000)
            .with_allocation(SegmentType::Examples, 0.25, 2);
        assert_eq!(compiler.budget_for(&SegmentType::Examples), 250);
    }

    #[test]
    fn test_compiler_budget_for() {
        let compiler = ContextCompiler::new(2000);
        assert_eq!(compiler.budget_for(&SegmentType::System), 300);
        assert_eq!(compiler.budget_for(&SegmentType::Tools), 400);
        let custom_compiler = ContextCompiler::new(1000);
        assert_eq!(custom_compiler.budget_for(&SegmentType::Examples), 0);
    }

    // -- ConversationCompactor tests ---------------------------------------

    #[test]
    fn test_compactor_new() {
        let compactor = ConversationCompactor::new(20, 5);
        assert_eq!(compactor.max_messages_before_compact, 20);
        assert_eq!(compactor.preserve_recent, 5);
        assert_eq!(compactor.compaction_count(), 0);
    }

    #[test]
    fn test_compactor_needs_compaction() {
        let compactor = ConversationCompactor::new(10, 3);
        assert!(!compactor.needs_compaction(5));
        assert!(!compactor.needs_compaction(10));
        assert!(compactor.needs_compaction(11));
        assert!(compactor.needs_compaction(100));
    }

    #[test]
    fn test_compactor_compact() {
        let mut compactor = ConversationCompactor::new(5, 2);
        let messages = vec![
            CompactableMessage { role: "user".to_string(), content: "Hello there.".to_string(), token_estimate: 3 },
            CompactableMessage { role: "assistant".to_string(), content: "Hi! How can I help?".to_string(), token_estimate: 5 },
            CompactableMessage { role: "user".to_string(), content: "Tell me about Rust.".to_string(), token_estimate: 5 },
            CompactableMessage { role: "assistant".to_string(), content: "Rust is a systems language.".to_string(), token_estimate: 6 },
            CompactableMessage { role: "user".to_string(), content: "Thanks!".to_string(), token_estimate: 2 },
        ];

        let result = compactor.compact(&messages);
        assert_eq!(result.original_count, 5);
        assert_eq!(result.compacted_count, 3);
        assert_eq!(result.preserved_messages.len(), 2);
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_compactor_preserve_recent() {
        let mut compactor = ConversationCompactor::new(3, 2);
        let messages = vec![
            CompactableMessage { role: "user".to_string(), content: "First message.".to_string(), token_estimate: 3 },
            CompactableMessage { role: "assistant".to_string(), content: "Second message.".to_string(), token_estimate: 3 },
            CompactableMessage { role: "user".to_string(), content: "Third message.".to_string(), token_estimate: 3 },
            CompactableMessage { role: "assistant".to_string(), content: "Fourth message.".to_string(), token_estimate: 3 },
        ];

        let result = compactor.compact(&messages);
        assert_eq!(result.preserved_messages.len(), 2);
        assert_eq!(result.preserved_messages[0].content, "Third message.");
        assert_eq!(result.preserved_messages[1].content, "Fourth message.");
    }

    #[test]
    fn test_compactor_summary() {
        let mut compactor = ConversationCompactor::new(2, 1);
        let messages = vec![
            CompactableMessage { role: "user".to_string(), content: "What is machine learning? It is cool.".to_string(), token_estimate: 8 },
            CompactableMessage { role: "assistant".to_string(), content: "ML is a subset of AI. It uses data.".to_string(), token_estimate: 9 },
            CompactableMessage { role: "user".to_string(), content: "Thanks!".to_string(), token_estimate: 2 },
        ];

        let result = compactor.compact(&messages);
        assert!(result.summary.contains("user:"));
        assert!(result.summary.contains("assistant:"));
        assert!(result.summary.contains("[Conversation Summary]"));
    }

    #[test]
    fn test_compactor_count() {
        let mut compactor = ConversationCompactor::new(2, 1);
        let messages = vec![
            CompactableMessage { role: "user".to_string(), content: "Hello.".to_string(), token_estimate: 2 },
            CompactableMessage { role: "assistant".to_string(), content: "Hi.".to_string(), token_estimate: 1 },
            CompactableMessage { role: "user".to_string(), content: "Bye.".to_string(), token_estimate: 1 },
        ];
        assert_eq!(compactor.compaction_count(), 0);
        compactor.compact(&messages);
        assert_eq!(compactor.compaction_count(), 1);
        compactor.compact(&messages);
        assert_eq!(compactor.compaction_count(), 2);
    }

    // -- ToolSearchIndex tests ---------------------------------------------

    #[test]
    fn test_tool_index_new() {
        let index = ToolSearchIndex::new();
        assert_eq!(index.tool_count(), 0);
        assert!(index.idf_cache.is_empty());
    }

    #[test]
    fn test_tool_index_add_tool() {
        let mut index = ToolSearchIndex::new();
        index.add_tool("calculator", "Performs arithmetic calculations and math operations");
        assert_eq!(index.tool_count(), 1);
        assert!(!index.idf_cache.is_empty());
    }

    #[test]
    fn test_tool_index_search() {
        let mut index = ToolSearchIndex::new();
        index.add_tool("calculator", "Performs arithmetic calculations and math");
        index.add_tool("weather", "Gets current weather forecast for a location");
        index.add_tool("search", "Searches the web for information");

        let results = index.search("calculate math addition", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].tool_name, "calculator");
    }

    #[test]
    fn test_tool_index_search_relevance() {
        let mut index = ToolSearchIndex::new();
        index.add_tool("file_reader", "Reads files from disk and returns content");
        index.add_tool("file_writer", "Writes content to files on disk");
        index.add_tool("calculator", "Performs math calculations");

        let results = index.search("read file content", 3);
        assert!(!results.is_empty());
        let reader_score = results.iter().find(|r| r.tool_name == "file_reader").map(|r| r.relevance_score).unwrap_or(0.0);
        let calc_score = results.iter().find(|r| r.tool_name == "calculator").map(|r| r.relevance_score).unwrap_or(0.0);
        assert!(reader_score > calc_score);
    }

    #[test]
    fn test_tool_index_top_k() {
        let mut index = ToolSearchIndex::new();
        index.add_tool("tool1", "alpha beta gamma");
        index.add_tool("tool2", "delta epsilon zeta");
        index.add_tool("tool3", "eta theta iota");
        index.add_tool("tool4", "kappa lambda mu");

        let results = index.search("alpha beta", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_tool_index_empty() {
        let index = ToolSearchIndex::new();
        let results = index.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tool_index_tokenize() {
        let tokens = ToolSearchIndex::tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_cosine_similarity() {
        let mut a = HashMap::new();
        a.insert("hello".to_string(), 1.0);
        a.insert("world".to_string(), 1.0);

        let b = a.clone();
        let sim = ToolSearchIndex::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);

        let mut c = HashMap::new();
        c.insert("foo".to_string(), 1.0);
        c.insert("bar".to_string(), 1.0);
        let sim2 = ToolSearchIndex::cosine_similarity(&a, &c);
        assert!((sim2 - 0.0).abs() < 1e-10);
    }

}
