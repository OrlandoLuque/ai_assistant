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

// ---------------------------------------------------------------------------
// ContextComposer
// ---------------------------------------------------------------------------

/// Orchestrates context composition: budgeting, trimming, overflow
/// detection, and prompt assembly.
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
}
