//! Token budget manager
//!
//! This module provides tools for managing token usage budgets,
//! helping to control costs and stay within limits.
//!
//! # Features
//!
//! - **Budget tracking**: Track usage against limits
//! - **Per-user budgets**: Individual user limits
//! - **Time-based limits**: Daily, weekly, monthly budgets
//! - **Alerts**: Notifications when approaching limits
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::token_budget::{BudgetManager, Budget, BudgetPeriod};
//!
//! let mut manager = BudgetManager::new();
//!
//! // Set a daily budget
//! manager.set_budget("default", Budget::new(100_000, BudgetPeriod::Daily));
//!
//! // Record usage
//! manager.record_usage("default", 1500);
//!
//! // Check remaining
//! let remaining = manager.remaining("default");
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Budget time period
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetPeriod {
    /// Per-hour budget
    Hourly,
    /// Per-day budget
    Daily,
    /// Per-week budget
    Weekly,
    /// Per-month budget
    Monthly,
    /// No reset (lifetime)
    Lifetime,
}

impl BudgetPeriod {
    /// Get period duration
    pub fn duration(&self) -> Option<Duration> {
        match self {
            Self::Hourly => Some(Duration::from_secs(3600)),
            Self::Daily => Some(Duration::from_secs(86400)),
            Self::Weekly => Some(Duration::from_secs(604800)),
            Self::Monthly => Some(Duration::from_secs(2592000)), // 30 days
            Self::Lifetime => None,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
            Self::Lifetime => "lifetime",
        }
    }
}

/// Budget configuration
#[derive(Debug, Clone)]
pub struct Budget {
    /// Maximum tokens for the period
    pub limit: u64,
    /// Budget period
    pub period: BudgetPeriod,
    /// Alert threshold (0-1)
    pub alert_threshold: f64,
    /// Hard limit (reject requests when exceeded)
    pub hard_limit: bool,
    /// Rollover unused tokens to next period
    pub rollover: bool,
    /// Maximum rollover amount
    pub max_rollover: Option<u64>,
}

impl Budget {
    /// Create a new budget
    pub fn new(limit: u64, period: BudgetPeriod) -> Self {
        Self {
            limit,
            period,
            alert_threshold: 0.8,
            hard_limit: true,
            rollover: false,
            max_rollover: None,
        }
    }

    /// Set alert threshold
    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.alert_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable soft limit (warn but allow)
    pub fn soft_limit(mut self) -> Self {
        self.hard_limit = false;
        self
    }

    /// Enable rollover
    pub fn with_rollover(mut self, max: Option<u64>) -> Self {
        self.rollover = true;
        self.max_rollover = max;
        self
    }
}

/// Budget usage tracking
#[derive(Debug, Clone)]
pub struct BudgetUsage {
    /// Tokens used in current period
    pub used: u64,
    /// Period start time
    pub period_start: Instant,
    /// Rollover tokens from previous period
    pub rollover_tokens: u64,
    /// Total tokens used (lifetime)
    pub lifetime_used: u64,
    /// Number of requests
    pub request_count: u64,
    /// Alert triggered
    pub alert_triggered: bool,
}

impl BudgetUsage {
    fn new() -> Self {
        Self {
            used: 0,
            period_start: Instant::now(),
            rollover_tokens: 0,
            lifetime_used: 0,
            request_count: 0,
            alert_triggered: false,
        }
    }

    /// Get effective limit (including rollover)
    pub fn effective_limit(&self, budget: &Budget) -> u64 {
        budget.limit + self.rollover_tokens
    }

    /// Get remaining tokens
    pub fn remaining(&self, budget: &Budget) -> u64 {
        let limit = self.effective_limit(budget);
        limit.saturating_sub(self.used)
    }

    /// Get usage percentage
    pub fn usage_percentage(&self, budget: &Budget) -> f64 {
        let limit = self.effective_limit(budget);
        if limit == 0 {
            1.0
        } else {
            self.used as f64 / limit as f64
        }
    }

    /// Check if period needs reset
    fn needs_reset(&self, period: &BudgetPeriod) -> bool {
        if let Some(duration) = period.duration() {
            self.period_start.elapsed() >= duration
        } else {
            false // Lifetime never resets
        }
    }

    /// Reset for new period
    fn reset(&mut self, budget: &Budget) {
        // Calculate rollover
        if budget.rollover {
            let unused = budget.limit.saturating_sub(self.used);
            self.rollover_tokens = match budget.max_rollover {
                Some(max) => unused.min(max),
                None => unused,
            };
        } else {
            self.rollover_tokens = 0;
        }

        self.used = 0;
        self.period_start = Instant::now();
        self.alert_triggered = false;
    }
}

/// Budget check result
#[derive(Debug, Clone)]
pub struct BudgetCheckResult {
    /// Is the request allowed
    pub allowed: bool,
    /// Remaining tokens after this request
    pub remaining: u64,
    /// Usage percentage after this request
    pub usage_percentage: f64,
    /// Alert triggered
    pub alert: Option<BudgetAlert>,
    /// Reason if not allowed
    pub reason: Option<String>,
}

/// Budget alert types
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BudgetAlert {
    /// Approaching limit
    ApproachingLimit { percentage: f64 },
    /// Limit exceeded (soft limit)
    LimitExceeded { over_by: u64 },
    /// Period reset occurred
    PeriodReset { rollover: u64 },
}

/// Budget manager
pub struct BudgetManager {
    budgets: HashMap<String, Budget>,
    usage: HashMap<String, BudgetUsage>,
    global_budget: Option<Budget>,
    global_usage: BudgetUsage,
}

impl BudgetManager {
    /// Create a new budget manager
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
            usage: HashMap::new(),
            global_budget: None,
            global_usage: BudgetUsage::new(),
        }
    }

    /// Set a budget for a key (user, session, etc.)
    pub fn set_budget(&mut self, key: &str, budget: Budget) {
        self.budgets.insert(key.to_string(), budget);
        if !self.usage.contains_key(key) {
            self.usage.insert(key.to_string(), BudgetUsage::new());
        }
    }

    /// Set global budget (applies to all)
    pub fn set_global_budget(&mut self, budget: Budget) {
        self.global_budget = Some(budget);
    }

    /// Remove a budget
    pub fn remove_budget(&mut self, key: &str) {
        self.budgets.remove(key);
        self.usage.remove(key);
    }

    /// Check if request is within budget
    pub fn check(&mut self, key: &str, tokens: u64) -> BudgetCheckResult {
        self.maybe_reset(key);

        let budget = match self.budgets.get(key) {
            Some(b) => b.clone(),
            None => {
                return BudgetCheckResult {
                    allowed: true,
                    remaining: u64::MAX,
                    usage_percentage: 0.0,
                    alert: None,
                    reason: None,
                };
            }
        };

        let usage = self
            .usage
            .get(key)
            .cloned()
            .unwrap_or_else(BudgetUsage::new);

        let new_used = usage.used + tokens;
        let limit = usage.effective_limit(&budget);
        let new_percentage = new_used as f64 / limit as f64;

        // Check if allowed
        let allowed = if budget.hard_limit {
            new_used <= limit
        } else {
            true
        };

        // Check for alerts
        let alert = if new_used > limit {
            Some(BudgetAlert::LimitExceeded {
                over_by: new_used - limit,
            })
        } else if new_percentage >= budget.alert_threshold && !usage.alert_triggered {
            Some(BudgetAlert::ApproachingLimit {
                percentage: new_percentage,
            })
        } else {
            None
        };

        let reason = if !allowed {
            Some(format!(
                "Budget exceeded: {} tokens used of {} limit",
                new_used, limit
            ))
        } else {
            None
        };

        BudgetCheckResult {
            allowed,
            remaining: limit.saturating_sub(new_used),
            usage_percentage: new_percentage,
            alert,
            reason,
        }
    }

    /// Record token usage
    pub fn record_usage(&mut self, key: &str, tokens: u64) -> BudgetCheckResult {
        let result = self.check(key, tokens);

        if let Some(usage) = self.usage.get_mut(key) {
            usage.used += tokens;
            usage.lifetime_used += tokens;
            usage.request_count += 1;

            if result.alert.is_some() {
                usage.alert_triggered = true;
            }
        }

        // Also record global usage
        self.global_usage.used += tokens;
        self.global_usage.lifetime_used += tokens;
        self.global_usage.request_count += 1;

        result
    }

    /// Get remaining tokens
    pub fn remaining(&mut self, key: &str) -> u64 {
        self.maybe_reset(key);

        let budget = match self.budgets.get(key) {
            Some(b) => b,
            None => return u64::MAX,
        };

        let usage = match self.usage.get(key) {
            Some(u) => u,
            None => return budget.limit,
        };

        usage.remaining(budget)
    }

    /// Get usage for a key
    pub fn get_usage(&mut self, key: &str) -> Option<BudgetUsage> {
        self.maybe_reset(key);
        self.usage.get(key).cloned()
    }

    /// Get all usage
    pub fn all_usage(&mut self) -> HashMap<String, BudgetUsage> {
        // Reset all that need it
        let keys: Vec<_> = self.usage.keys().cloned().collect();
        for key in keys {
            self.maybe_reset(&key);
        }
        self.usage.clone()
    }

    /// Get global usage
    pub fn global_usage(&self) -> &BudgetUsage {
        &self.global_usage
    }

    /// Get statistics
    pub fn stats(&self) -> BudgetStats {
        let mut stats = BudgetStats::default();

        stats.total_budgets = self.budgets.len();
        stats.total_tokens_used = self.global_usage.lifetime_used;
        stats.total_requests = self.global_usage.request_count;

        for (key, usage) in &self.usage {
            if let Some(budget) = self.budgets.get(key) {
                let percentage = usage.usage_percentage(budget);
                if percentage >= 1.0 {
                    stats.budgets_exceeded += 1;
                } else if percentage >= budget.alert_threshold {
                    stats.budgets_warning += 1;
                }
            }
        }

        stats
    }

    /// Reset a budget's usage
    pub fn reset_usage(&mut self, key: &str) {
        if let Some(usage) = self.usage.get_mut(key) {
            if let Some(budget) = self.budgets.get(key) {
                usage.reset(budget);
            }
        }
    }

    /// Reset all usage
    pub fn reset_all(&mut self) {
        for (key, usage) in &mut self.usage {
            if let Some(budget) = self.budgets.get(key) {
                usage.reset(budget);
            }
        }
        self.global_usage = BudgetUsage::new();
    }

    fn maybe_reset(&mut self, key: &str) {
        let needs_reset = self
            .usage
            .get(key)
            .and_then(|u| self.budgets.get(key).map(|b| u.needs_reset(&b.period)))
            .unwrap_or(false);

        if needs_reset {
            if let Some(budget) = self.budgets.get(key).cloned() {
                if let Some(usage) = self.usage.get_mut(key) {
                    usage.reset(&budget);
                }
            }
        }
    }
}

impl Default for BudgetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget statistics
#[derive(Debug, Clone, Default)]
pub struct BudgetStats {
    /// Total budgets configured
    pub total_budgets: usize,
    /// Budgets that have exceeded limit
    pub budgets_exceeded: usize,
    /// Budgets at warning level
    pub budgets_warning: usize,
    /// Total tokens used (all time)
    pub total_tokens_used: u64,
    /// Total requests (all time)
    pub total_requests: u64,
}

/// Token estimator for different content types
pub struct TokenEstimator;

impl TokenEstimator {
    /// Estimate tokens for text — delegates to canonical implementation.
    pub fn estimate_text(text: &str) -> usize {
        crate::context::estimate_tokens(text)
    }

    /// Estimate tokens for a chat message
    pub fn estimate_message(_role: &str, content: &str) -> usize {
        // Role overhead + content
        4 + Self::estimate_text(content)
    }

    /// Estimate tokens for chat history
    pub fn estimate_chat(messages: &[(String, String)]) -> usize {
        messages
            .iter()
            .map(|(role, content)| Self::estimate_message(role, content))
            .sum()
    }

    /// Estimate output tokens based on input
    pub fn estimate_output(input_tokens: usize, task_type: &str) -> usize {
        match task_type {
            "chat" => input_tokens / 2 + 100,
            "summary" => input_tokens / 4,
            "code" => input_tokens * 2,
            "translation" => (input_tokens as f64 * 1.2) as usize,
            _ => input_tokens,
        }
    }
}

/// Budget-aware request planner
pub struct RequestPlanner {
    manager: BudgetManager,
}

impl RequestPlanner {
    /// Create a new planner
    pub fn new(manager: BudgetManager) -> Self {
        Self { manager }
    }

    /// Plan requests within budget
    pub fn plan_requests(&mut self, key: &str, requests: &[PlannedRequest]) -> Vec<PlannedRequest> {
        let mut remaining = self.manager.remaining(key);
        let mut approved = Vec::new();

        for request in requests {
            if request.estimated_tokens <= remaining {
                approved.push(request.clone());
                remaining -= request.estimated_tokens;
            }
        }

        approved
    }

    /// Check if a request fits in budget
    pub fn can_execute(&mut self, key: &str, estimated_tokens: u64) -> bool {
        self.manager.remaining(key) >= estimated_tokens
    }

    /// Get manager reference
    pub fn manager(&self) -> &BudgetManager {
        &self.manager
    }

    /// Get mutable manager reference
    pub fn manager_mut(&mut self) -> &mut BudgetManager {
        &mut self.manager
    }
}

/// A planned request
#[derive(Debug, Clone)]
pub struct PlannedRequest {
    /// Request ID
    pub id: String,
    /// Estimated tokens
    pub estimated_tokens: u64,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Request content
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_creation() {
        let budget = Budget::new(100_000, BudgetPeriod::Daily);
        assert_eq!(budget.limit, 100_000);
        assert_eq!(budget.period, BudgetPeriod::Daily);
    }

    #[test]
    fn test_budget_manager() {
        let mut manager = BudgetManager::new();
        manager.set_budget("user1", Budget::new(1000, BudgetPeriod::Daily));

        // Record some usage
        manager.record_usage("user1", 300);
        manager.record_usage("user1", 200);

        assert_eq!(manager.remaining("user1"), 500);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut manager = BudgetManager::new();
        manager.set_budget("user1", Budget::new(1000, BudgetPeriod::Daily));

        // Use all budget
        manager.record_usage("user1", 1000);

        // Try to use more
        let result = manager.check("user1", 100);
        assert!(!result.allowed);
        assert!(result.reason.is_some());
    }

    #[test]
    fn test_soft_limit() {
        let mut manager = BudgetManager::new();
        manager.set_budget("user1", Budget::new(1000, BudgetPeriod::Daily).soft_limit());

        // Use all budget
        manager.record_usage("user1", 1000);

        // Should still be allowed (soft limit)
        let result = manager.check("user1", 100);
        assert!(result.allowed);
        assert!(matches!(
            result.alert,
            Some(BudgetAlert::LimitExceeded { .. })
        ));
    }

    #[test]
    fn test_alert_threshold() {
        let mut manager = BudgetManager::new();
        manager.set_budget(
            "user1",
            Budget::new(1000, BudgetPeriod::Daily).with_alert_threshold(0.8),
        );

        // Use 80% of budget
        let result = manager.record_usage("user1", 800);
        assert!(matches!(
            result.alert,
            Some(BudgetAlert::ApproachingLimit { .. })
        ));
    }

    #[test]
    fn test_token_estimator() {
        let text = "Hello, this is a test message with some content.";
        let estimate = TokenEstimator::estimate_text(text);
        assert!(estimate > 0);
        assert!(estimate < text.len());
    }

    #[test]
    fn test_no_budget() {
        let mut manager = BudgetManager::new();

        // No budget set - should allow unlimited
        let result = manager.check("unknown", 1_000_000);
        assert!(result.allowed);
        assert_eq!(result.remaining, u64::MAX);
    }

    #[test]
    fn test_budget_period_names() {
        assert_eq!(BudgetPeriod::Hourly.name(), "hourly");
        assert_eq!(BudgetPeriod::Daily.name(), "daily");
        assert_eq!(BudgetPeriod::Lifetime.name(), "lifetime");
    }

    #[test]
    fn test_record_usage() {
        let mut manager = BudgetManager::new();
        manager.set_budget("user1", Budget::new(1000, BudgetPeriod::Daily));
        let result = manager.record_usage("user1", 500);
        assert!(result.allowed);
        assert_eq!(result.remaining, 500);
    }

    #[test]
    fn test_budget_hard_limit_blocks() {
        let mut manager = BudgetManager::new();
        manager.set_budget("user1", Budget::new(100, BudgetPeriod::Daily));
        manager.record_usage("user1", 100);
        let result = manager.check("user1", 50);
        assert!(!result.allowed);
    }
}
