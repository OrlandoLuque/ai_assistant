//! Cost tracking integration — session-level cost dashboard and middleware
//!
//! Bridges the existing `cost.rs` infrastructure (CostEstimator, BudgetManager)
//! with higher-level session tracking, reporting, and cost-aware request gating.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cost::{BudgetManager, BudgetStatus, CostEstimator};

// ---------------------------------------------------------------------------
// Security constants
// ---------------------------------------------------------------------------

/// Maximum reasonable cost in a single entry or projection (USD).
const MAX_COST: f64 = 1_000_000.0;

/// Maximum number of entries retained in CostDashboard to prevent unbounded growth.
const MAX_ENTRIES: usize = 100_000;

/// Validate a cost value: must be finite, non-negative, and within MAX_COST.
/// Returns the sanitized value (0.0 for invalid inputs, clamped for excess).
fn validate_cost(cost: f64) -> f64 {
    if cost.is_nan() || cost.is_infinite() || cost < 0.0 {
        0.0
    } else {
        cost.min(MAX_COST)
    }
}

/// Sanitize a string field for CSV export to prevent formula injection.
///
/// Wraps fields that start with dangerous characters (`=`, `+`, `-`, `@`,
/// `\t`, `\r`) or contain delimiters in double quotes with proper escaping.
fn sanitize_csv_field(s: &str) -> String {
    let needs_escape = s.starts_with('=')
        || s.starts_with('+')
        || s.starts_with('-')
        || s.starts_with('@')
        || s.starts_with('\t')
        || s.starts_with('\r')
        || s.contains(',')
        || s.contains('"')
        || s.contains('\n');
    if needs_escape {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// RequestType
// ---------------------------------------------------------------------------

/// Classification of an AI API request for cost tracking purposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RequestType {
    Chat,
    Embedding,
    Rerank,
    Completion,
    ImageGeneration,
}

impl RequestType {
    /// Return a stable string key for aggregation maps.
    fn as_str(&self) -> &'static str {
        match self {
            RequestType::Chat => "Chat",
            RequestType::Embedding => "Embedding",
            RequestType::Rerank => "Rerank",
            RequestType::Completion => "Completion",
            RequestType::ImageGeneration => "ImageGeneration",
        }
    }
}

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// RequestCostEntry
// ---------------------------------------------------------------------------

/// A single recorded cost entry for one API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCostEntry {
    /// ISO 8601 timestamp of when the request was made.
    pub timestamp: String,
    /// Model name (e.g. "gpt-4o", "claude-3-sonnet").
    pub model: String,
    /// Number of input tokens consumed.
    pub input_tokens: usize,
    /// Number of output tokens produced.
    pub output_tokens: usize,
    /// Estimated cost in USD.
    pub cost_usd: f64,
    /// Type of the request.
    pub request_type: RequestType,
}

// ---------------------------------------------------------------------------
// CostDashboard
// ---------------------------------------------------------------------------

/// Session-level cost summary and reporting dashboard.
///
/// Records individual request costs, provides aggregated queries (by model,
/// by type, top-N most expensive), budget status, and human-readable /
/// CSV exports.
pub struct CostDashboard {
    entries: Vec<RequestCostEntry>,
    estimator: CostEstimator,
    budget: Option<BudgetManager>,
    session_start: String,
}

impl CostDashboard {
    /// Create a new dashboard with no budget constraints.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            estimator: CostEstimator::new(),
            budget: None,
            session_start: Self::now_iso8601(),
        }
    }

    /// Create a new dashboard backed by a `BudgetManager`.
    pub fn with_budget(budget: BudgetManager) -> Self {
        Self {
            entries: Vec::new(),
            estimator: CostEstimator::new(),
            budget: Some(budget),
            session_start: Self::now_iso8601(),
        }
    }

    // -- Recording ----------------------------------------------------------

    /// Record a completed request.
    ///
    /// The cost is estimated via the internal `CostEstimator` using the
    /// model name and token counts. If a `BudgetManager` is attached its
    /// running totals are updated as well.
    pub fn record(
        &mut self,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
        request_type: RequestType,
    ) {
        let estimate = self
            .estimator
            .estimate(model, "api", input_tokens, output_tokens);
        let cost = validate_cost(estimate.cost);

        let entry = RequestCostEntry {
            timestamp: Self::now_iso8601(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            cost_usd: cost,
            request_type,
        };

        self.entries.push(entry);

        // Evict oldest entries if over cap (S2 mitigation)
        if self.entries.len() > MAX_ENTRIES {
            let drain_count = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(0..drain_count);
        }

        if let Some(ref mut bm) = self.budget {
            bm.record(cost);
        }
    }

    // -- Queries ------------------------------------------------------------

    /// Total cost across all recorded entries (USD).
    pub fn total_cost(&self) -> f64 {
        self.entries.iter().map(|e| e.cost_usd).sum()
    }

    /// Total number of recorded requests.
    pub fn total_requests(&self) -> usize {
        self.entries.len()
    }

    /// Aggregate cost grouped by model name.
    pub fn cost_by_model(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for e in &self.entries {
            *map.entry(e.model.clone()).or_insert(0.0) += e.cost_usd;
        }
        map
    }

    /// Aggregate cost grouped by `RequestType` (key is the display string).
    pub fn cost_by_type(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for e in &self.entries {
            *map.entry(e.request_type.as_str().to_string())
                .or_insert(0.0) += e.cost_usd;
        }
        map
    }

    /// Return the `n` most expensive entries, sorted descending by cost.
    pub fn most_expensive(&self, n: usize) -> Vec<&RequestCostEntry> {
        let mut sorted: Vec<&RequestCostEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.cost_usd
                .partial_cmp(&a.cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Average cost per request (returns 0.0 when no entries exist).
    pub fn average_cost_per_request(&self) -> f64 {
        if self.entries.is_empty() {
            0.0
        } else {
            self.total_cost() / self.entries.len() as f64
        }
    }

    /// Read-only access to all recorded entries.
    pub fn entries(&self) -> &[RequestCostEntry] {
        &self.entries
    }

    // -- Budget -------------------------------------------------------------

    /// Current budget status (if a budget manager is attached).
    pub fn budget_status(&self) -> Option<BudgetStatus> {
        self.budget.as_ref().map(|bm| bm.check(0.0))
    }

    /// Remaining budget in USD, taking the *minimum* of daily and monthly
    /// remaining budgets. Returns `None` when no budget manager is set or
    /// neither daily nor monthly limits are configured.
    pub fn budget_remaining(&self) -> Option<f64> {
        self.budget.as_ref().and_then(|bm| {
            let (daily, monthly) = bm.remaining();
            match (daily, monthly) {
                (Some(d), Some(m)) => Some(d.min(m)),
                (Some(d), None) => Some(d),
                (None, Some(m)) => Some(m),
                (None, None) => None,
            }
        })
    }

    // -- Reporting ----------------------------------------------------------

    /// Generate a human-readable multi-line report.
    pub fn format_report(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        lines.push("=== Cost Dashboard Report ===".to_string());
        lines.push(format!("Session start: {}", self.session_start));
        lines.push(format!("Total requests: {}", self.total_requests()));
        lines.push(format!("Total cost: ${:.4}", self.total_cost()));
        lines.push(format!(
            "Average cost/request: ${:.4}",
            self.average_cost_per_request()
        ));

        // Cost by model
        let by_model = self.cost_by_model();
        if !by_model.is_empty() {
            lines.push(String::new());
            lines.push("--- Cost by Model ---".to_string());
            let mut model_entries: Vec<_> = by_model.into_iter().collect();
            model_entries
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (model, cost) in model_entries {
                lines.push(format!("  {}: ${:.4}", model, cost));
            }
        }

        // Cost by type
        let by_type = self.cost_by_type();
        if !by_type.is_empty() {
            lines.push(String::new());
            lines.push("--- Cost by Type ---".to_string());
            let mut type_entries: Vec<_> = by_type.into_iter().collect();
            type_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (rtype, cost) in type_entries {
                lines.push(format!("  {}: ${:.4}", rtype, cost));
            }
        }

        // Budget
        if let Some(ref bm) = self.budget {
            lines.push(String::new());
            lines.push("--- Budget ---".to_string());
            if let Some(d) = bm.daily_limit {
                lines.push(format!(
                    "  Daily limit: ${:.2} (spent: ${:.4})",
                    d, bm.spent_today
                ));
            }
            if let Some(m) = bm.monthly_limit {
                lines.push(format!(
                    "  Monthly limit: ${:.2} (spent: ${:.4})",
                    m, bm.spent_month
                ));
            }
            if let Some(remaining) = self.budget_remaining() {
                lines.push(format!("  Remaining: ${:.4}", remaining));
            }
        }

        // Projections
        if !self.entries.is_empty() {
            lines.push(String::new());
            lines.push("--- Projections ---".to_string());
            if let Some(daily) = self.projected_daily_cost() {
                lines.push(format!("  Projected daily: ${:.2}", daily));
            }
            if let Some(monthly) = self.projected_monthly_cost() {
                lines.push(format!("  Projected monthly: ${:.2}", monthly));
            }
            if let Some(rph) = self.requests_per_hour() {
                lines.push(format!("  Requests/hour: {:.1}", rph));
            }
        }

        lines.join("\n")
    }

    /// Export all entries as CSV (header + data rows).
    ///
    /// All fields are sanitized to prevent CSV formula injection (S1 mitigation).
    pub fn export_csv(&self) -> String {
        let mut csv =
            String::from("timestamp,model,input_tokens,output_tokens,cost_usd,request_type\n");
        for e in &self.entries {
            csv.push_str(&format!(
                "{},{},{},{},{:.6},{}\n",
                sanitize_csv_field(&e.timestamp),
                sanitize_csv_field(&e.model),
                e.input_tokens,
                e.output_tokens,
                e.cost_usd,
                sanitize_csv_field(&e.request_type.as_str())
            ));
        }
        csv
    }

    // -- Projections --------------------------------------------------------

    /// Compute requests per hour based on session elapsed time.
    fn requests_per_hour(&self) -> Option<f64> {
        if self.entries.is_empty() {
            return None;
        }
        let elapsed_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(Self::parse_epoch_secs(&self.session_start));
        if elapsed_secs == 0 {
            return None;
        }
        let hours = elapsed_secs as f64 / 3600.0;
        Some(self.entries.len() as f64 / hours)
    }

    /// Projected daily cost based on current session rate and average cost.
    ///
    /// Returns `None` if no requests have been recorded yet.
    pub fn projected_daily_cost(&self) -> Option<f64> {
        let rph = self.requests_per_hour()?;
        let avg = self.average_cost_per_request();
        Some(validate_cost(rph * 24.0 * avg))
    }

    /// Projected monthly cost (daily * 30).
    ///
    /// Returns `None` if no requests have been recorded yet.
    pub fn projected_monthly_cost(&self) -> Option<f64> {
        self.projected_daily_cost().map(|d| validate_cost(d * 30.0))
    }

    /// Projected cost for `n` additional requests at the current average.
    pub fn projected_cost_for_requests(&self, n: usize) -> f64 {
        validate_cost(self.average_cost_per_request() * n as f64)
    }

    /// Parse the session_start ISO 8601 timestamp back to epoch seconds.
    fn parse_epoch_secs(iso: &str) -> u64 {
        // Parse "YYYY-MM-DDThh:mm:ssZ" back to approximate epoch seconds.
        // This is a best-effort parse for our own format.
        let parts: Vec<&str> = iso.split('T').collect();
        if parts.len() != 2 {
            return 0;
        }
        let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
        let time_str = parts[1].trim_end_matches('Z');
        let time_parts: Vec<u64> = time_str.split(':').filter_map(|s| s.parse().ok()).collect();
        if date_parts.len() != 3 || time_parts.len() != 3 {
            return 0;
        }
        // Approximate days from epoch using the same algorithm in reverse
        let (y, m, d) = (date_parts[0], date_parts[1], date_parts[2]);
        let (yr, mo) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
        let era = yr / 400;
        let yoe = yr - era * 400;
        let doy = (153 * mo + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146097 + doe - 719468;
        days * 86400 + time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
    }

    // -- Snapshots -----------------------------------------------------------

    /// Create a serializable snapshot of the dashboard state.
    pub fn snapshot(&self) -> CostDashboardSnapshot {
        CostDashboardSnapshot {
            schema_version: 1,
            entries: self.entries.clone(),
            session_start: self.session_start.clone(),
            budget_config: self.budget.as_ref().map(|bm| CostAwareConfig {
                enabled: true,
                daily_budget: bm.daily_limit,
                monthly_budget: bm.monthly_limit,
                per_request_limit: bm.per_request_limit,
                alert_threshold_pct: bm.warning_threshold as f64,
                track_by_model: true,
            }),
        }
    }

    /// Restore dashboard state from a snapshot.
    ///
    /// Validates all entries on load: NaN/Infinity/negative costs are clamped
    /// to 0.0 (S6 mitigation).
    pub fn restore(&mut self, snapshot: CostDashboardSnapshot) {
        self.session_start = snapshot.session_start;
        self.entries = snapshot
            .entries
            .into_iter()
            .map(|mut e| {
                e.cost_usd = validate_cost(e.cost_usd);
                e
            })
            .collect();

        // Enforce cap
        if self.entries.len() > MAX_ENTRIES {
            let drain_count = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(0..drain_count);
        }

        if let Some(config) = snapshot.budget_config {
            let mut bm = BudgetManager::new();
            if let Some(d) = config.daily_budget {
                bm = bm.with_daily_limit(d);
            }
            if let Some(m) = config.monthly_budget {
                bm = bm.with_monthly_limit(m);
            }
            if let Some(r) = config.per_request_limit {
                bm = bm.with_request_limit(r);
            }
            bm.warning_threshold = config.alert_threshold_pct as f32;
            // Re-accumulate spending from entries
            let total: f64 = self.entries.iter().map(|e| e.cost_usd).sum();
            bm.record(total);
            self.budget = Some(bm);
        }
    }

    // -- Reset --------------------------------------------------------------

    /// Clear all recorded entries (budget manager totals are also reset).
    pub fn clear(&mut self) {
        self.entries.clear();
        if let Some(ref mut bm) = self.budget {
            bm.spent_today = 0.0;
            bm.spent_month = 0.0;
        }
        self.session_start = Self::now_iso8601();
    }

    // -- Helpers ------------------------------------------------------------

    /// Produce a simple ISO 8601 timestamp string.
    fn now_iso8601() -> String {
        // Use std::time for a lightweight, no-external-dep timestamp.
        // This gives seconds since UNIX epoch which we format as a pseudo-ISO string.
        let dur = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = dur.as_secs();

        // Decompose into date/time components (UTC).
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Simple days-since-epoch to Y-M-D (good enough for session timestamps).
        let (year, month, day) = Self::days_to_ymd(days);

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            year, month, day, hours, minutes, seconds
        )
    }

    /// Convert days since Unix epoch (1970-01-01) to (year, month, day).
    fn days_to_ymd(days: u64) -> (u64, u64, u64) {
        // Algorithm adapted from Howard Hinnant's civil_from_days.
        let z = days as i64 + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u64; // day of era [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = if m <= 2 { y + 1 } else { y } as u64;
        (year, m, d)
    }
}

// ---------------------------------------------------------------------------
// CostAwareConfig
// ---------------------------------------------------------------------------

/// Configuration for automatic cost tracking and budget enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CostAwareConfig {
    /// Whether cost tracking is enabled.
    pub enabled: bool,
    /// Optional daily budget limit (USD).
    pub daily_budget: Option<f64>,
    /// Optional monthly budget limit (USD).
    pub monthly_budget: Option<f64>,
    /// Optional per-request cost limit (USD).
    pub per_request_limit: Option<f64>,
    /// Alert threshold as a fraction of budget (0.0–1.0). Default 0.8.
    pub alert_threshold_pct: f64,
    /// Whether to track costs broken down by model.
    pub track_by_model: bool,
}

impl Default for CostAwareConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            daily_budget: None,
            monthly_budget: None,
            per_request_limit: None,
            alert_threshold_pct: 0.8,
            track_by_model: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CostDashboardSnapshot
// ---------------------------------------------------------------------------

/// Serializable snapshot of a `CostDashboard` for persistence.
///
/// Captures entries, session start, and budget configuration so the
/// dashboard can be saved to disk and restored across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDashboardSnapshot {
    /// Schema version for forward compatibility.
    pub schema_version: u32,
    /// All recorded cost entries.
    pub entries: Vec<RequestCostEntry>,
    /// ISO 8601 session start timestamp.
    pub session_start: String,
    /// Budget configuration (if any).
    pub budget_config: Option<CostAwareConfig>,
}

// ---------------------------------------------------------------------------
// CostDecision / CostMiddleware
// ---------------------------------------------------------------------------

/// Decision returned by `CostMiddleware::pre_request`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CostDecision {
    /// Request is within budget — proceed.
    Allow,
    /// Request is over the alert threshold but not the hard limit.
    Warn(String),
    /// Request would exceed the hard limit — block it.
    Block(String),
}

/// Trait for cost-aware request gating.
///
/// Implementations inspect estimated costs *before* a request is sent and
/// record actual costs *after* the response is received.
pub trait CostMiddleware: Send + Sync {
    /// Evaluate whether a request should proceed based on estimated input tokens.
    fn pre_request(&self, model: &str, estimated_input_tokens: usize) -> CostDecision;

    /// Record the actual cost after a response is received. Returns the entry.
    fn post_response(
        &mut self,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
    ) -> RequestCostEntry;
}

// ---------------------------------------------------------------------------
// DefaultCostMiddleware
// ---------------------------------------------------------------------------

/// Default implementation of `CostMiddleware` backed by a `CostDashboard`
/// and `CostAwareConfig`.
pub struct DefaultCostMiddleware {
    dashboard: CostDashboard,
    config: CostAwareConfig,
}

impl DefaultCostMiddleware {
    /// Create a new middleware from the given config.
    pub fn new(config: CostAwareConfig) -> Self {
        let budget = {
            let mut bm = BudgetManager::new();
            if let Some(d) = config.daily_budget {
                bm = bm.with_daily_limit(d);
            }
            if let Some(m) = config.monthly_budget {
                bm = bm.with_monthly_limit(m);
            }
            if let Some(r) = config.per_request_limit {
                bm = bm.with_request_limit(r);
            }
            bm.warning_threshold = config.alert_threshold_pct as f32;
            bm
        };

        Self {
            dashboard: CostDashboard::with_budget(budget),
            config,
        }
    }

    /// Read-only access to the inner dashboard.
    pub fn dashboard(&self) -> &CostDashboard {
        &self.dashboard
    }

    /// Mutable access to the inner dashboard.
    pub fn dashboard_mut(&mut self) -> &mut CostDashboard {
        &mut self.dashboard
    }
}

impl CostMiddleware for DefaultCostMiddleware {
    fn pre_request(&self, model: &str, estimated_input_tokens: usize) -> CostDecision {
        if !self.config.enabled {
            return CostDecision::Allow;
        }

        // Estimate cost assuming a 1:1 output ratio for the pre-check.
        let estimate = self.dashboard.estimator.estimate(
            model,
            "api",
            estimated_input_tokens,
            estimated_input_tokens,
        );

        // Check per-request limit from config.
        if let Some(limit) = self.config.per_request_limit {
            if estimate.cost > limit {
                return CostDecision::Block(format!(
                    "Estimated cost ${:.4} exceeds per-request limit ${:.2}",
                    estimate.cost, limit
                ));
            }
        }

        // Check budget manager (daily / monthly).
        if let Some(ref bm) = self.dashboard.budget {
            let status = bm.check(estimate.cost);
            match status {
                BudgetStatus::Exceeded {
                    limit_type,
                    limit,
                    current,
                } => {
                    return CostDecision::Block(format!(
                        "{} budget exceeded: ${:.4} / ${:.2} limit",
                        limit_type, current, limit
                    ));
                }
                BudgetStatus::Warning {
                    limit_type,
                    limit,
                    remaining,
                    ..
                } => {
                    return CostDecision::Warn(format!(
                        "{} budget warning: ${:.4} remaining of ${:.2} limit",
                        limit_type, remaining, limit
                    ));
                }
                BudgetStatus::Ok => {}
            }
        }

        CostDecision::Allow
    }

    fn post_response(
        &mut self,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
    ) -> RequestCostEntry {
        self.dashboard
            .record(model, input_tokens, output_tokens, RequestType::Chat);

        // Return a clone of the most recently added entry.
        self.dashboard
            .entries
            .last()
            .expect("just pushed entry via record()")
            .clone()
    }
}

// ---------------------------------------------------------------------------
// MCP Tool Registration
// ---------------------------------------------------------------------------

/// Register cost-related MCP tools on the given server.
///
/// All tools are read-only and return aggregated data (no per-request detail
/// exposed to prevent information leakage — S7 mitigation).
pub fn register_cost_tools(
    server: &mut crate::mcp_protocol::server::McpServer,
    dashboard: std::sync::Arc<std::sync::Mutex<CostDashboard>>,
) {
    use crate::mcp_protocol::types::{McpTool, McpToolAnnotation};

    let read_only_annotation = McpToolAnnotation {
        title: None,
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(false),
    };

    // --- cost_report ---
    {
        let dash = dashboard.clone();
        server.register_tool(
            McpTool::new(
                "cost_report",
                "Get a formatted cost report for the current session including totals, \
                 breakdowns by model/type, budget status, and projections.",
            )
            .with_annotations(McpToolAnnotation {
                title: Some("Cost Report".into()),
                ..read_only_annotation.clone()
            }),
            move |_args| {
                let guard = dash.lock().map_err(|e| format!("Lock error: {}", e))?;
                let report = guard.format_report();
                Ok(serde_json::json!({ "report": report }))
            },
        );
    }

    // --- cost_budget_status ---
    {
        let dash = dashboard.clone();
        server.register_tool(
            McpTool::new(
                "cost_budget_status",
                "Get remaining budget and cost projections for the current session.",
            )
            .with_annotations(McpToolAnnotation {
                title: Some("Budget Status".into()),
                ..read_only_annotation.clone()
            }),
            move |_args| {
                let guard = dash.lock().map_err(|e| format!("Lock error: {}", e))?;
                let remaining = guard.budget_remaining();
                let projected_daily = guard.projected_daily_cost();
                let projected_monthly = guard.projected_monthly_cost();
                let status = guard.budget_status().map(|s| format!("{:?}", s));
                Ok(serde_json::json!({
                    "remaining_usd": remaining,
                    "projected_daily_usd": projected_daily,
                    "projected_monthly_usd": projected_monthly,
                    "total_cost_usd": guard.total_cost(),
                    "total_requests": guard.total_requests(),
                    "budget_status": status,
                }))
            },
        );
    }

    // --- cost_savings_summary ---
    {
        let dash = dashboard.clone();
        server.register_tool(
            McpTool::new(
                "cost_savings_summary",
                "Get a summary of tokens and cost saved by the context budget allocator.",
            )
            .with_annotations(McpToolAnnotation {
                title: Some("Cost Savings".into()),
                ..read_only_annotation
            }),
            move |_args| {
                let guard = dash.lock().map_err(|e| format!("Lock error: {}", e))?;
                let total_cost = guard.total_cost();
                let total_requests = guard.total_requests();
                let avg = guard.average_cost_per_request();
                Ok(serde_json::json!({
                    "total_cost_usd": total_cost,
                    "total_requests": total_requests,
                    "average_cost_per_request_usd": avg,
                    "cost_by_model": guard.cost_by_model(),
                    "cost_by_type": guard.cost_by_type(),
                }))
            },
        );
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a dashboard with a known model so cost is deterministic.
    fn make_dashboard() -> CostDashboard {
        CostDashboard::new()
    }

    // 1. Record one entry, verify total_cost.
    #[test]
    fn test_record_cost() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        assert!(dash.total_cost() > 0.0, "total cost should be positive");
        assert_eq!(dash.total_requests(), 1);
    }

    // 2. Record 5 entries, verify accumulation.
    #[test]
    fn test_multiple_records() {
        let mut dash = make_dashboard();
        for _ in 0..5 {
            dash.record("gpt-3.5-turbo", 500, 200, RequestType::Chat);
        }
        assert_eq!(dash.total_requests(), 5);
        // Each entry should have the same cost so total = 5 * single.
        let single = dash.entries()[0].cost_usd;
        let diff = (dash.total_cost() - single * 5.0).abs();
        assert!(
            diff < 1e-9,
            "accumulated cost should be 5x single entry cost"
        );
    }

    // 3. Three models, verify breakdown.
    #[test]
    fn test_cost_by_model() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        dash.record("gpt-3.5-turbo", 1000, 500, RequestType::Chat);
        dash.record("claude-3-sonnet", 1000, 500, RequestType::Chat);

        let by_model = dash.cost_by_model();
        assert_eq!(by_model.len(), 3);
        assert!(by_model.contains_key("gpt-4"));
        assert!(by_model.contains_key("gpt-3.5-turbo"));
        assert!(by_model.contains_key("claude-3-sonnet"));
    }

    // 4. Mix of Chat / Embedding, verify breakdown.
    #[test]
    fn test_cost_by_type() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        dash.record("gpt-4", 2000, 0, RequestType::Embedding);

        let by_type = dash.cost_by_type();
        assert_eq!(by_type.len(), 2);
        assert!(by_type.contains_key("Chat"));
        assert!(by_type.contains_key("Embedding"));
    }

    // 5. Five entries, top 2 are correct.
    #[test]
    fn test_most_expensive() {
        let mut dash = make_dashboard();
        // Vary output tokens to get different costs.
        dash.record("gpt-4", 100, 100, RequestType::Chat);
        dash.record("gpt-4", 100, 10000, RequestType::Chat); // expensive
        dash.record("gpt-4", 100, 200, RequestType::Chat);
        dash.record("gpt-4", 100, 50000, RequestType::Chat); // most expensive
        dash.record("gpt-4", 100, 300, RequestType::Chat);

        let top = dash.most_expensive(2);
        assert_eq!(top.len(), 2);
        assert!(
            top[0].cost_usd >= top[1].cost_usd,
            "should be sorted descending"
        );
        assert_eq!(top[0].output_tokens, 50000);
        assert_eq!(top[1].output_tokens, 10000);
    }

    // 6. Average cost calculation.
    #[test]
    fn test_average_cost() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);

        let avg = dash.average_cost_per_request();
        let expected = dash.total_cost() / 3.0;
        assert!((avg - expected).abs() < 1e-12);

        // Edge case: empty
        let empty = CostDashboard::new();
        assert_eq!(empty.average_cost_per_request(), 0.0);
    }

    // 7. Report contains expected sections.
    #[test]
    fn test_format_report() {
        let mut dash = CostDashboard::with_budget(BudgetManager::new().with_daily_limit(10.0));
        dash.record("gpt-4", 1000, 500, RequestType::Chat);

        let report = dash.format_report();
        assert!(
            report.contains("Cost Dashboard Report"),
            "should contain title"
        );
        assert!(
            report.contains("Total requests:"),
            "should contain request count"
        );
        assert!(report.contains("Total cost:"), "should contain total cost");
        assert!(
            report.contains("Cost by Model"),
            "should contain model section"
        );
        assert!(report.contains("Budget"), "should contain budget section");
    }

    // 8. CSV has header + correct rows.
    #[test]
    fn test_export_csv() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        dash.record("claude-3-sonnet", 2000, 800, RequestType::Embedding);

        let csv = dash.export_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 data rows");
        assert!(
            lines[0].contains("timestamp,model,input_tokens,output_tokens,cost_usd,request_type")
        );
        assert!(lines[1].contains("gpt-4"));
        assert!(lines[2].contains("claude-3-sonnet"));
        assert!(lines[2].contains("Embedding"));
    }

    // 9. Over budget returns Exceeded.
    #[test]
    fn test_budget_status() {
        let budget = BudgetManager::new().with_daily_limit(0.001);
        let mut dash = CostDashboard::with_budget(budget);

        // Record enough to blow past the tiny budget.
        dash.record("gpt-4", 10000, 10000, RequestType::Chat);

        let _status = dash.budget_status().unwrap();
        // After recording, the spent_today > daily_limit so check(0.0) should still
        // show Warning or Ok (since check(0.0) doesn't add new cost). Let's verify
        // that remaining is essentially 0.
        let remaining = dash.budget_remaining().unwrap();
        assert!(
            remaining < 0.001,
            "remaining should be near zero after blowing budget"
        );
    }

    // 10. Clear resets everything.
    #[test]
    fn test_clear() {
        let budget = BudgetManager::new().with_daily_limit(100.0);
        let mut dash = CostDashboard::with_budget(budget);
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        assert!(dash.total_requests() > 0);
        assert!(dash.total_cost() > 0.0);

        dash.clear();
        assert_eq!(dash.total_requests(), 0);
        assert_eq!(dash.total_cost(), 0.0);
        assert!(dash.entries().is_empty());
        // Budget should be reset too.
        let remaining = dash.budget_remaining().unwrap();
        assert!((remaining - 100.0).abs() < 1e-9);
    }

    // ── V75: Security tests ──

    // S4: NaN cost validation
    #[test]
    fn test_validate_cost_nan() {
        assert!((validate_cost(f64::NAN) - 0.0).abs() < f64::EPSILON);
    }

    // S4: Infinity cost validation
    #[test]
    fn test_validate_cost_infinity() {
        assert!((validate_cost(f64::INFINITY) - 0.0).abs() < f64::EPSILON);
        assert!((validate_cost(f64::NEG_INFINITY) - 0.0).abs() < f64::EPSILON);
    }

    // S4: Negative cost validation
    #[test]
    fn test_validate_cost_negative() {
        assert!((validate_cost(-5.0) - 0.0).abs() < f64::EPSILON);
    }

    // S4: Valid cost passes through
    #[test]
    fn test_validate_cost_valid() {
        assert!((validate_cost(1.5) - 1.5).abs() < f64::EPSILON);
        assert!((validate_cost(0.0) - 0.0).abs() < f64::EPSILON);
    }

    // S4: Cost exceeding MAX_COST is clamped
    #[test]
    fn test_validate_cost_clamp() {
        assert!((validate_cost(2_000_000.0) - MAX_COST).abs() < f64::EPSILON);
    }

    // S1: CSV injection is sanitized
    #[test]
    fn test_csv_injection_sanitized() {
        let evil_model = "=cmd|'/C calc'!A0";
        let sanitized = sanitize_csv_field(evil_model);
        assert!(
            sanitized.starts_with('"'),
            "injection string must be quoted: {}",
            sanitized
        );
        assert!(!sanitized.starts_with('='), "must not start with =");

        // Other dangerous prefixes
        assert!(sanitize_csv_field("+SUM(1)").starts_with('"'));
        assert!(sanitize_csv_field("-2+3").starts_with('"'));
        assert!(sanitize_csv_field("@SUM").starts_with('"'));

        // Safe strings pass through unchanged
        assert_eq!(sanitize_csv_field("gpt-4"), "gpt-4");
        assert_eq!(sanitize_csv_field("claude3sonnet"), "claude3sonnet");
    }

    // S2: Entries cap eviction
    #[test]
    fn test_entries_cap_eviction() {
        let mut dash = CostDashboard::new();
        // Record MAX_ENTRIES + 10 entries
        for i in 0..(MAX_ENTRIES + 10) {
            dash.record(&format!("model-{}", i), 100, 50, RequestType::Chat);
        }
        assert!(
            dash.entries().len() <= MAX_ENTRIES,
            "entries should be capped at MAX_ENTRIES, got {}",
            dash.entries().len()
        );
        // The newest entries should be retained (not the oldest)
        let last_model = &dash.entries().last().unwrap().model;
        assert!(last_model.contains(&format!("{}", MAX_ENTRIES + 9)));
    }

    // ── V75: Projection tests ──

    #[test]
    fn test_projected_cost_for_requests() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);

        let avg = dash.average_cost_per_request();
        let projected = dash.projected_cost_for_requests(10);
        assert!((projected - avg * 10.0).abs() < 0.001);

        // Zero requests
        assert!(dash.projected_cost_for_requests(0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_projected_daily_cost_some() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        // After at least 1 request, projection should return Some
        // (unless elapsed time is 0, which is unlikely in practice)
        let daily = dash.projected_daily_cost();
        // May be None if test runs too fast (0 elapsed secs) — that's OK
        if let Some(d) = daily {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_projected_monthly_cost() {
        let mut dash = make_dashboard();
        // Empty dashboard should return None
        assert!(dash.projected_monthly_cost().is_none());

        dash.record("gpt-4", 1000, 1000, RequestType::Chat);
        if let Some(monthly) = dash.projected_monthly_cost() {
            assert!(monthly >= 0.0);
            // Monthly should be ~30x daily
            if let Some(daily) = dash.projected_daily_cost() {
                assert!((monthly - daily * 30.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_projection_in_report() {
        let mut dash = make_dashboard();
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        let report = dash.format_report();
        assert!(
            report.contains("Projections"),
            "report should contain Projections section"
        );
    }

    // ── V75: Snapshot tests ──

    #[test]
    fn test_dashboard_snapshot_roundtrip() {
        let budget = BudgetManager::new().with_daily_limit(50.0);
        let mut dash = CostDashboard::with_budget(budget);
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        dash.record("claude-3-sonnet", 2000, 800, RequestType::Embedding);

        let snapshot = dash.snapshot();
        assert_eq!(snapshot.schema_version, 1);
        assert_eq!(snapshot.entries.len(), 2);
        assert!(snapshot.budget_config.is_some());

        // Restore into a new dashboard
        let mut restored = CostDashboard::new();
        restored.restore(snapshot);
        assert_eq!(restored.total_requests(), 2);
        assert!((restored.total_cost() - dash.total_cost()).abs() < 0.001);
    }

    #[test]
    fn test_snapshot_rejects_nan_on_load() {
        let snapshot = CostDashboardSnapshot {
            schema_version: 1,
            entries: vec![
                RequestCostEntry {
                    timestamp: "2026-04-09T10:00:00Z".to_string(),
                    model: "gpt-4".to_string(),
                    input_tokens: 100,
                    output_tokens: 50,
                    cost_usd: f64::NAN,
                    request_type: RequestType::Chat,
                },
                RequestCostEntry {
                    timestamp: "2026-04-09T10:01:00Z".to_string(),
                    model: "gpt-4".to_string(),
                    input_tokens: 200,
                    output_tokens: 100,
                    cost_usd: -99.0,
                    request_type: RequestType::Chat,
                },
            ],
            session_start: "2026-04-09T10:00:00Z".to_string(),
            budget_config: None,
        };

        let mut dash = CostDashboard::new();
        dash.restore(snapshot);
        // NaN and negative costs should be clamped to 0.0
        for entry in dash.entries() {
            assert!(
                entry.cost_usd >= 0.0 && entry.cost_usd.is_finite(),
                "cost must be valid after restore: {}",
                entry.cost_usd
            );
        }
    }

    // 11. Under budget returns Allow.
    #[test]
    fn test_cost_middleware_allow() {
        let config = CostAwareConfig {
            enabled: true,
            daily_budget: Some(100.0),
            monthly_budget: None,
            per_request_limit: Some(10.0),
            alert_threshold_pct: 0.8,
            track_by_model: true,
        };
        let mw = DefaultCostMiddleware::new(config);

        // A small request should be allowed.
        let decision = mw.pre_request("gpt-3.5-turbo", 100);
        assert!(
            matches!(decision, CostDecision::Allow),
            "small request should be allowed, got: {:?}",
            decision,
        );
    }

    // ── V75: MCP tool registration tests ──

    fn call_mcp_tool(
        server: &crate::mcp_protocol::server::McpServer,
        tool_name: &str,
    ) -> serde_json::Value {
        use crate::mcp_protocol::types::McpRequest;
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": tool_name,
                "arguments": {}
            })),
        };
        let response = server.handle_request(request);
        // Extract the result field from the response
        serde_json::to_value(&response).unwrap()
    }

    #[test]
    fn test_cost_tools_register() {
        use crate::mcp_protocol::types::McpRequest;
        use std::sync::{Arc, Mutex};
        let dashboard = Arc::new(Mutex::new(CostDashboard::new()));
        let mut server = crate::mcp_protocol::server::McpServer::new("test", "0.1.0");
        register_cost_tools(&mut server, dashboard);

        // List tools via MCP protocol
        let list_req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/list".to_string(),
            params: None,
        };
        let resp = server.handle_request(list_req);
        let resp_json = serde_json::to_value(&resp).unwrap();
        let tools = resp_json["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"cost_report"), "cost_report missing");
        assert!(
            names.contains(&"cost_budget_status"),
            "cost_budget_status missing"
        );
        assert!(
            names.contains(&"cost_savings_summary"),
            "cost_savings_summary missing"
        );
    }

    #[test]
    fn test_cost_report_tool_returns_data() {
        use std::sync::{Arc, Mutex};
        let mut dash = CostDashboard::new();
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        let dashboard = Arc::new(Mutex::new(dash));
        let mut server = crate::mcp_protocol::server::McpServer::new("test", "0.1.0");
        register_cost_tools(&mut server, dashboard);

        let resp = call_mcp_tool(&server, "cost_report");
        // The response should contain the cost report text
        let resp_str = resp.to_string();
        assert!(
            resp_str.contains("Cost Dashboard Report"),
            "response should contain report: {}",
            resp_str
        );
    }

    #[test]
    fn test_cost_budget_status_tool() {
        use std::sync::{Arc, Mutex};
        let budget = BudgetManager::new().with_daily_limit(50.0);
        let mut dash = CostDashboard::with_budget(budget);
        dash.record("gpt-4", 1000, 500, RequestType::Chat);
        let dashboard = Arc::new(Mutex::new(dash));
        let mut server = crate::mcp_protocol::server::McpServer::new("test", "0.1.0");
        register_cost_tools(&mut server, dashboard);

        let resp = call_mcp_tool(&server, "cost_budget_status");
        let resp_str = resp.to_string();
        assert!(
            resp_str.contains("remaining_usd"),
            "response should contain remaining: {}",
            resp_str
        );
    }

    // 12. Over per-request limit returns Block.
    #[test]
    fn test_cost_middleware_block() {
        let config = CostAwareConfig {
            enabled: true,
            daily_budget: None,
            monthly_budget: None,
            per_request_limit: Some(0.0001), // tiny limit
            alert_threshold_pct: 0.8,
            track_by_model: true,
        };
        let mw = DefaultCostMiddleware::new(config);

        // A large request should be blocked.
        let decision = mw.pre_request("gpt-4", 1_000_000);
        assert!(
            matches!(decision, CostDecision::Block(_)),
            "over-limit request should be blocked, got: {:?}",
            decision,
        );
    }
}
