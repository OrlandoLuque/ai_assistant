//! Trigger system — event-driven action execution
//!
//! Executes actions when specific conditions are met: cron schedules,
//! file changes, feed updates, AI events, manual triggers, etc.
//! Integrates with the EventBus and Scheduler.

use crate::events::{AiEvent, EventHandler};
use crate::scheduler::{CronSchedule, ScheduledAction};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// TriggerCondition
// ============================================================================

/// What causes a trigger to fire.
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// Fire on a cron schedule.
    Cron(CronSchedule),
    /// Fire when a file changes (path, optional glob pattern).
    FileChanged {
        path: PathBuf,
        pattern: Option<String>,
    },
    /// Fire when an RSS/Atom feed has new entries.
    FeedUpdate { feed_url: String },
    /// Fire when a specific AI event is emitted.
    AiEventMatch { event_name: String },
    /// Fire on an HTTP webhook (method, path).
    WebhookReceived { method: String, path: String },
    /// Only fire manually via `fire_trigger()`.
    Manual,
}

// ============================================================================
// Trigger
// ============================================================================

/// A trigger that fires an action when its condition is met.
#[derive(Debug, Clone)]
pub struct Trigger {
    pub id: String,
    pub name: String,
    pub condition: TriggerCondition,
    pub action: ScheduledAction,
    pub enabled: bool,
    pub fire_count: u32,
    pub max_fires: Option<u32>,
    pub cooldown_ms: Option<u64>,
    pub last_fired: Option<u64>,
    pub created_at: u64,
}

impl Trigger {
    /// Create a new trigger.
    pub fn new(
        name: impl Into<String>,
        condition: TriggerCondition,
        action: ScheduledAction,
    ) -> Self {
        Self {
            id: String::new(), // set by TriggerManager
            name: name.into(),
            condition,
            action,
            enabled: true,
            fire_count: 0,
            max_fires: None,
            cooldown_ms: None,
            last_fired: None,
            created_at: now_millis(),
        }
    }

    /// Set maximum number of times this trigger can fire.
    pub fn with_max_fires(mut self, max: u32) -> Self {
        self.max_fires = Some(max);
        self
    }

    /// Set cooldown period between firings.
    pub fn with_cooldown_ms(mut self, ms: u64) -> Self {
        self.cooldown_ms = Some(ms);
        self
    }

    /// Whether this trigger has exhausted its fire limit.
    pub fn is_exhausted(&self) -> bool {
        if let Some(max) = self.max_fires {
            self.fire_count >= max
        } else {
            false
        }
    }

    /// Whether the trigger is in cooldown (fired too recently).
    pub fn is_in_cooldown(&self) -> bool {
        if let (Some(cooldown), Some(last)) = (self.cooldown_ms, self.last_fired) {
            let now = now_millis();
            now.saturating_sub(last) < cooldown
        } else {
            false
        }
    }

    /// Whether the trigger can fire right now.
    pub fn can_fire(&self) -> bool {
        self.enabled && !self.is_exhausted() && !self.is_in_cooldown()
    }
}

// ============================================================================
// FiredTrigger — record of a trigger firing
// ============================================================================

/// Record of a trigger that fired.
#[derive(Debug, Clone)]
pub struct FiredTrigger {
    pub trigger_id: String,
    pub trigger_name: String,
    pub fired_at: u64,
    pub action: ScheduledAction,
}

// ============================================================================
// TriggerManager
// ============================================================================

/// Manages triggers: registration, condition evaluation, and firing.
pub struct TriggerManager {
    triggers: Vec<Trigger>,
    next_id: u64,
    fire_log: Vec<FiredTrigger>,
    max_log: usize,
}

impl TriggerManager {
    /// Create a new trigger manager.
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
            next_id: 1,
            fire_log: Vec::new(),
            max_log: 1000,
        }
    }

    /// Register a trigger and return its ID.
    pub fn register(&mut self, mut trigger: Trigger) -> String {
        let id = format!("trigger-{}", self.next_id);
        self.next_id += 1;
        trigger.id = id.clone();
        self.triggers.push(trigger);
        id
    }

    /// Remove a trigger by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.triggers.len();
        self.triggers.retain(|t| t.id != id);
        self.triggers.len() < before
    }

    /// Get a trigger by ID.
    pub fn get(&self, id: &str) -> Option<&Trigger> {
        self.triggers.iter().find(|t| t.id == id)
    }

    /// Get a mutable trigger by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Trigger> {
        self.triggers.iter_mut().find(|t| t.id == id)
    }

    /// List all triggers.
    pub fn list(&self) -> &[Trigger] {
        &self.triggers
    }

    /// List enabled triggers.
    pub fn enabled_triggers(&self) -> Vec<&Trigger> {
        self.triggers.iter().filter(|t| t.enabled).collect()
    }

    /// Enable or disable a trigger.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> bool {
        if let Some(t) = self.get_mut(id) {
            t.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Manually fire a trigger by ID. Returns the FiredTrigger if successful.
    pub fn fire_trigger(&mut self, id: &str) -> Result<FiredTrigger, String> {
        let trigger = self
            .triggers
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or_else(|| format!("Trigger '{}' not found", id))?;

        if !trigger.can_fire() {
            if trigger.is_exhausted() {
                return Err(format!("Trigger '{}' has exhausted its fire limit", id));
            }
            if trigger.is_in_cooldown() {
                return Err(format!("Trigger '{}' is in cooldown", id));
            }
            return Err(format!("Trigger '{}' is disabled", id));
        }

        let now = now_millis();
        trigger.fire_count += 1;
        trigger.last_fired = Some(now);

        let fired = FiredTrigger {
            trigger_id: trigger.id.clone(),
            trigger_name: trigger.name.clone(),
            fired_at: now,
            action: trigger.action.clone(),
        };

        // Log
        if self.fire_log.len() >= self.max_log {
            self.fire_log.remove(0);
        }
        self.fire_log.push(fired.clone());

        Ok(fired)
    }

    /// Check which triggers should fire for a given AI event.
    /// Returns the triggers that match and can fire.
    pub fn check_event(&self, event: &AiEvent) -> Vec<&Trigger> {
        let event_name = event.name();
        self.triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::AiEventMatch { event_name: en } if en == event_name)
            })
            .collect()
    }

    /// Fire all triggers that match an AI event. Returns list of fired triggers.
    pub fn fire_for_event(&mut self, event: &AiEvent) -> Vec<FiredTrigger> {
        let event_name = event.name().to_string();
        let matching_ids: Vec<String> = self
            .triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::AiEventMatch { event_name: en } if *en == event_name)
            })
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in matching_ids {
            if let Ok(f) = self.fire_trigger(&id) {
                fired.push(f);
            }
        }
        fired
    }

    /// Check which triggers should fire for a cron tick.
    /// minute/hour/day/month/weekday follow standard cron conventions.
    pub fn check_cron(
        &self,
        minute: u32,
        hour: u32,
        day: u32,
        month: u32,
        weekday: u32,
    ) -> Vec<&Trigger> {
        self.triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::Cron(sched) if sched.matches(minute, hour, day, month, weekday))
            })
            .collect()
    }

    /// Fire all cron-matching triggers. Returns list of fired triggers.
    pub fn fire_cron(
        &mut self,
        minute: u32,
        hour: u32,
        day: u32,
        month: u32,
        weekday: u32,
    ) -> Vec<FiredTrigger> {
        let matching_ids: Vec<String> = self
            .triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::Cron(sched) if sched.matches(minute, hour, day, month, weekday))
            })
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in matching_ids {
            if let Ok(f) = self.fire_trigger(&id) {
                fired.push(f);
            }
        }
        fired
    }

    /// Check triggers that match a file change.
    pub fn check_file_change(&self, changed_path: &str) -> Vec<&Trigger> {
        self.triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                match &t.condition {
                    TriggerCondition::FileChanged { path, pattern } => {
                        let path_matches =
                            changed_path.starts_with(&path.to_string_lossy().as_ref().to_string());
                        if let Some(pat) = pattern {
                            path_matches && changed_path.contains(pat)
                        } else {
                            path_matches
                        }
                    }
                    _ => false,
                }
            })
            .collect()
    }

    /// Fire all triggers matching a file change.
    pub fn fire_for_file_change(&mut self, changed_path: &str) -> Vec<FiredTrigger> {
        let matching_ids: Vec<String> = self
            .check_file_change(changed_path)
            .iter()
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in matching_ids {
            if let Ok(f) = self.fire_trigger(&id) {
                fired.push(f);
            }
        }
        fired
    }

    /// Check triggers matching a feed update.
    pub fn check_feed_update(&self, feed_url: &str) -> Vec<&Trigger> {
        self.triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::FeedUpdate { feed_url: url } if url == feed_url)
            })
            .collect()
    }

    /// Fire all triggers matching a feed update.
    pub fn fire_for_feed_update(&mut self, feed_url: &str) -> Vec<FiredTrigger> {
        let matching_ids: Vec<String> = self
            .check_feed_update(feed_url)
            .iter()
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in matching_ids {
            if let Ok(f) = self.fire_trigger(&id) {
                fired.push(f);
            }
        }
        fired
    }

    /// Check triggers matching a webhook.
    pub fn check_webhook(&self, method: &str, path: &str) -> Vec<&Trigger> {
        self.triggers
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(&t.condition, TriggerCondition::WebhookReceived { method: m, path: p } if m == method && p == path)
            })
            .collect()
    }

    /// Get the fire log.
    pub fn fire_log(&self) -> &[FiredTrigger] {
        &self.fire_log
    }

    /// Clear the fire log.
    pub fn clear_log(&mut self) {
        self.fire_log.clear();
    }

    /// Number of triggers registered.
    pub fn trigger_count(&self) -> usize {
        self.triggers.len()
    }

    /// Total number of times any trigger has fired.
    pub fn total_fires(&self) -> u32 {
        self.triggers.iter().map(|t| t.fire_count).sum()
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TriggerEventBridge — connects EventBus to TriggerManager
// ============================================================================

/// Bridges the EventBus to the TriggerManager.
///
/// When an event is emitted on the EventBus, this handler checks all
/// registered triggers and fires matching ones.
pub struct TriggerEventBridge {
    manager: Arc<Mutex<TriggerManager>>,
}

impl TriggerEventBridge {
    /// Create a bridge that will fire triggers in the given manager.
    pub fn new(manager: Arc<Mutex<TriggerManager>>) -> Self {
        Self { manager }
    }
}

impl EventHandler for TriggerEventBridge {
    fn on_event(&self, event: &AiEvent) {
        if let Ok(mut mgr) = self.manager.lock() {
            mgr.fire_for_event(event);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// CronField (trigger-system-local cron parsing)
// ============================================================================

/// Cron field representation for the trigger-system scheduler runner.
///
/// This is the trigger-system's own cron field enum, parallel to
/// `scheduler::CronField` but with variant names aligned to the scheduler
/// runner API.
#[derive(Debug, Clone, PartialEq)]
pub enum TsCronField {
    /// Wildcard `*` — matches any value.
    Any,
    /// A specific numeric value.
    Exact(u32),
    /// An inclusive range (start-end).
    Range(u32, u32),
    /// A step value (`*/N`).
    Step(u32),
    /// A list of values (comma-separated).
    List(Vec<u32>),
}

impl TsCronField {
    /// Check whether the given value matches this field.
    pub fn matches(&self, value: u32) -> bool {
        match self {
            TsCronField::Any => true,
            TsCronField::Exact(v) => value == *v,
            TsCronField::Range(lo, hi) => value >= *lo && value <= *hi,
            TsCronField::Step(s) => {
                if *s == 0 {
                    return false;
                }
                value % s == 0
            }
            TsCronField::List(vs) => vs.contains(&value),
        }
    }

    /// Parse a single cron field token.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        if s == "*" {
            return Ok(TsCronField::Any);
        }

        // Step: */N
        if let Some(rest) = s.strip_prefix("*/") {
            let step: u32 = rest
                .parse()
                .map_err(|_| format!("invalid step value: {}", rest))?;
            if step == 0 {
                return Err("step value must be > 0".to_string());
            }
            return Ok(TsCronField::Step(step));
        }

        // List: contains comma
        if s.contains(',') {
            let values: Result<Vec<u32>, _> = s
                .split(',')
                .map(|v| {
                    v.trim()
                        .parse::<u32>()
                        .map_err(|_| format!("invalid list value: {}", v.trim()))
                })
                .collect();
            return Ok(TsCronField::List(values?));
        }

        // Range: contains dash
        if s.contains('-') {
            let parts: Vec<&str> = s.splitn(2, '-').collect();
            if parts.len() != 2 {
                return Err(format!("invalid range: {}", s));
            }
            let lo: u32 = parts[0]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range start: {}", parts[0].trim()))?;
            let hi: u32 = parts[1]
                .trim()
                .parse()
                .map_err(|_| format!("invalid range end: {}", parts[1].trim()))?;
            return Ok(TsCronField::Range(lo, hi));
        }

        // Single value
        let v: u32 = s
            .parse()
            .map_err(|_| format!("invalid cron field: {}", s))?;
        Ok(TsCronField::Exact(v))
    }
}

// ============================================================================
// CronExpression
// ============================================================================

/// A parsed five-field cron expression for the scheduler runner.
#[derive(Debug, Clone)]
pub struct CronExpression {
    pub minute: TsCronField,
    pub hour: TsCronField,
    pub day_of_month: TsCronField,
    pub month: TsCronField,
    pub day_of_week: TsCronField,
}

impl CronExpression {
    /// Parse a standard five-field cron expression (`"* * * * *"`).
    pub fn parse(expr: &str) -> Result<Self, String> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(format!(
                "expected 5 cron fields, got {} in '{}'",
                fields.len(),
                expr
            ));
        }
        Ok(CronExpression {
            minute: TsCronField::parse(fields[0])?,
            hour: TsCronField::parse(fields[1])?,
            day_of_month: TsCronField::parse(fields[2])?,
            month: TsCronField::parse(fields[3])?,
            day_of_week: TsCronField::parse(fields[4])?,
        })
    }

    /// Check whether the given time components match this expression.
    pub fn matches(&self, minute: u32, hour: u32, dom: u32, month: u32, dow: u32) -> bool {
        self.minute.matches(minute)
            && self.hour.matches(hour)
            && self.day_of_month.matches(dom)
            && self.month.matches(month)
            && self.day_of_week.matches(dow)
    }
}

// ============================================================================
// SchedulerError
// ============================================================================

/// An error recorded during scheduler execution.
#[derive(Debug, Clone)]
pub struct SchedulerError {
    pub trigger_id: String,
    pub message: String,
    pub timestamp_ms: u64,
}

// ============================================================================
// SchedulerConfig
// ============================================================================

/// Configuration for the scheduler runner.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub poll_interval_ms: u64,
    pub max_concurrent: usize,
    pub enable_cron: bool,
    pub enable_events: bool,
    pub enable_file_watch: bool,
    pub enable_feed: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,
            max_concurrent: 4,
            enable_cron: true,
            enable_events: true,
            enable_file_watch: true,
            enable_feed: true,
        }
    }
}

// ============================================================================
// SchedulerState
// ============================================================================

/// Runtime state of the scheduler runner.
#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub running: bool,
    pub tick_count: u64,
    pub last_tick_ms: u64,
    pub fired_history: Vec<FiredTrigger>,
    pub active_actions: Vec<String>,
    pub errors: Vec<SchedulerError>,
}

impl SchedulerState {
    /// Create a new default state (all zeros / empty).
    pub fn new() -> Self {
        Self {
            running: false,
            tick_count: 0,
            last_tick_ms: 0,
            fired_history: Vec::new(),
            active_actions: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SchedulerRunner
// ============================================================================

/// Drives the trigger system by periodically ticking through registered
/// triggers and collecting fired results.
pub struct SchedulerRunner {
    config: SchedulerConfig,
    manager: TriggerManager,
    state: SchedulerState,
}

impl SchedulerRunner {
    /// Create a new scheduler runner.
    pub fn new(config: SchedulerConfig, manager: TriggerManager) -> Self {
        Self {
            config,
            manager,
            state: SchedulerState::new(),
        }
    }

    /// Perform a full tick: check all enabled trigger types.
    ///
    /// For cron triggers, `current_time_ms` is decomposed into
    /// minute / hour / day / month / weekday and checked against each
    /// trigger's cron schedule.
    pub fn tick(&mut self, current_time_ms: u64) -> Vec<FiredTrigger> {
        let mut all_fired = Vec::new();

        if self.config.enable_cron {
            let cron_fired = self.tick_cron_inner(current_time_ms);
            all_fired.extend(cron_fired);
        }

        self.state.tick_count += 1;
        self.state.last_tick_ms = current_time_ms;
        self.state.fired_history.extend(all_fired.clone());
        all_fired
    }

    /// Tick only cron triggers.
    pub fn tick_cron(&mut self, current_time_ms: u64) -> Vec<FiredTrigger> {
        let fired = self.tick_cron_inner(current_time_ms);
        self.state.tick_count += 1;
        self.state.last_tick_ms = current_time_ms;
        self.state.fired_history.extend(fired.clone());
        fired
    }

    /// Internal: check cron triggers by decomposing timestamp into time fields.
    fn tick_cron_inner(&mut self, current_time_ms: u64) -> Vec<FiredTrigger> {
        // Convert ms timestamp to broken-down time components.
        // Using simple arithmetic (UTC):
        let total_secs = current_time_ms / 1000;
        let minute = ((total_secs / 60) % 60) as u32;
        let hour = ((total_secs / 3600) % 24) as u32;

        // Day-of-month, month, day-of-week from days since epoch
        let days_since_epoch = (total_secs / 86400) as i64;
        let (year, month, day) = days_to_ymd(days_since_epoch);
        let dow = ((days_since_epoch + 4) % 7) as u32; // 1970-01-01 was Thursday (4)

        let _ = year; // not used in cron matching
        self.manager.fire_cron(minute, hour, day, month, dow)
    }

    /// Tick event triggers matching the given event name.
    ///
    /// Fires all enabled triggers whose condition is `AiEventMatch` with a
    /// matching `event_name`.
    pub fn tick_events(&mut self, event_name: &str) -> Vec<FiredTrigger> {
        let matching_ids: Vec<String> = self
            .manager
            .list()
            .iter()
            .filter(|t| {
                if !t.can_fire() {
                    return false;
                }
                matches!(
                    &t.condition,
                    TriggerCondition::AiEventMatch { event_name: en } if en == event_name
                )
            })
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in matching_ids {
            if let Ok(f) = self.manager.fire_trigger(&id) {
                fired.push(f);
            }
        }
        self.state.fired_history.extend(fired.clone());
        fired
    }

    /// Tick file-change triggers for the given path.
    pub fn tick_file_changes(&mut self, path: &str, _modified_ms: u64) -> Vec<FiredTrigger> {
        let fired = self.manager.fire_for_file_change(path);
        self.state.fired_history.extend(fired.clone());
        fired
    }

    /// Tick feed-update triggers for the given feed name/URL.
    pub fn tick_feed(&mut self, feed_name: &str, _new_entry_count: usize) -> Vec<FiredTrigger> {
        let fired = self.manager.fire_for_feed_update(feed_name);
        self.state.fired_history.extend(fired.clone());
        fired
    }

    /// Run `n` ticks starting at `start_time_ms`, advancing by
    /// `config.poll_interval_ms` each iteration.
    pub fn run_n_ticks(&mut self, n: u64, start_time_ms: u64) -> Vec<FiredTrigger> {
        let mut all_fired = Vec::new();
        for i in 0..n {
            let time = start_time_ms + i * self.config.poll_interval_ms;
            let fired = self.tick(time);
            all_fired.extend(fired);
        }
        all_fired
    }

    /// Read-only access to the scheduler state.
    pub fn state(&self) -> &SchedulerState {
        &self.state
    }

    /// Reset state: clear history, errors, and tick count.
    pub fn reset_state(&mut self) {
        self.state.fired_history.clear();
        self.state.errors.clear();
        self.state.tick_count = 0;
        self.state.last_tick_ms = 0;
        self.state.active_actions.clear();
    }

    /// Export scheduler status as JSON.
    pub fn export_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": {
                "poll_interval_ms": self.config.poll_interval_ms,
                "max_concurrent": self.config.max_concurrent,
                "enable_cron": self.config.enable_cron,
                "enable_events": self.config.enable_events,
                "enable_file_watch": self.config.enable_file_watch,
                "enable_feed": self.config.enable_feed,
            },
            "state": {
                "running": self.state.running,
                "tick_count": self.state.tick_count,
                "last_tick_ms": self.state.last_tick_ms,
                "active_actions": self.state.active_actions.len(),
            },
            "fired_history_count": self.state.fired_history.len(),
            "errors_count": self.state.errors.len(),
            "trigger_count": self.manager.trigger_count(),
        })
    }

    /// Read-only access to the inner trigger manager.
    pub fn manager(&self) -> &TriggerManager {
        &self.manager
    }

    /// Mutable access to the inner trigger manager.
    pub fn manager_mut(&mut self) -> &mut TriggerManager {
        &mut self.manager
    }
}

/// Convert days since Unix epoch to (year, month 1-12, day 1-31).
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Algorithm from Howard Hinnant's `civil_from_days`
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::CronSchedule;

    fn make_action() -> ScheduledAction {
        ScheduledAction::Custom {
            action_type: "test".into(),
            payload: "payload".into(),
        }
    }

    // 1. test_register_trigger
    #[test]
    fn test_register_trigger() {
        let mut mgr = TriggerManager::new();
        let trigger = Trigger::new("my-trigger", TriggerCondition::Manual, make_action());
        let id = mgr.register(trigger);

        assert_eq!(mgr.trigger_count(), 1);
        let t = mgr.get(&id).unwrap();
        assert_eq!(t.name, "my-trigger");
        assert!(t.enabled);
        assert_eq!(t.fire_count, 0);
    }

    // 2. test_remove_trigger
    #[test]
    fn test_remove_trigger() {
        let mut mgr = TriggerManager::new();
        let id = mgr.register(Trigger::new("t1", TriggerCondition::Manual, make_action()));
        assert_eq!(mgr.trigger_count(), 1);

        assert!(mgr.remove(&id));
        assert_eq!(mgr.trigger_count(), 0);
        assert!(!mgr.remove(&id)); // already gone
    }

    // 3. test_fire_manual_trigger
    #[test]
    fn test_fire_manual_trigger() {
        let mut mgr = TriggerManager::new();
        let id = mgr.register(Trigger::new(
            "manual",
            TriggerCondition::Manual,
            make_action(),
        ));

        let fired = mgr.fire_trigger(&id).unwrap();
        assert_eq!(fired.trigger_name, "manual");
        assert_eq!(mgr.get(&id).unwrap().fire_count, 1);
        assert_eq!(mgr.fire_log().len(), 1);
    }

    // 4. test_fire_exhausted
    #[test]
    fn test_fire_exhausted() {
        let mut mgr = TriggerManager::new();
        let trigger =
            Trigger::new("once", TriggerCondition::Manual, make_action()).with_max_fires(1);
        let id = mgr.register(trigger);

        assert!(mgr.fire_trigger(&id).is_ok());
        assert!(mgr.fire_trigger(&id).is_err()); // exhausted
        assert_eq!(mgr.get(&id).unwrap().fire_count, 1);
    }

    // 5. test_fire_cooldown
    #[test]
    fn test_fire_cooldown() {
        let mut mgr = TriggerManager::new();
        let trigger =
            Trigger::new("cool", TriggerCondition::Manual, make_action()).with_cooldown_ms(60_000); // 60 seconds cooldown
        let id = mgr.register(trigger);

        assert!(mgr.fire_trigger(&id).is_ok());
        // Should be in cooldown now
        assert!(mgr.fire_trigger(&id).is_err());
    }

    // 6. test_fire_disabled
    #[test]
    fn test_fire_disabled() {
        let mut mgr = TriggerManager::new();
        let id = mgr.register(Trigger::new("dis", TriggerCondition::Manual, make_action()));

        mgr.set_enabled(&id, false);
        assert!(mgr.fire_trigger(&id).is_err());

        mgr.set_enabled(&id, true);
        assert!(mgr.fire_trigger(&id).is_ok());
    }

    // 7. test_check_ai_event
    #[test]
    fn test_check_ai_event() {
        let mut mgr = TriggerManager::new();
        let trigger = Trigger::new(
            "on-tool",
            TriggerCondition::AiEventMatch {
                event_name: "tool_executed".into(),
            },
            make_action(),
        );
        mgr.register(trigger);

        let event = AiEvent::ToolExecuted {
            name: "grep".into(),
            success: true,
            duration_ms: 100,
        };
        let matching = mgr.check_event(&event);
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].name, "on-tool");

        // Non-matching event
        let event2 = AiEvent::ResponseComplete {
            response_length: 42,
        };
        assert_eq!(mgr.check_event(&event2).len(), 0);
    }

    // 8. test_fire_for_event
    #[test]
    fn test_fire_for_event() {
        let mut mgr = TriggerManager::new();
        mgr.register(Trigger::new(
            "on-complete",
            TriggerCondition::AiEventMatch {
                event_name: "response_complete".into(),
            },
            make_action(),
        ));

        let event = AiEvent::ResponseComplete {
            response_length: 100,
        };
        let fired = mgr.fire_for_event(&event);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].trigger_name, "on-complete");
    }

    // 9. test_check_cron
    #[test]
    fn test_check_cron() {
        let mut mgr = TriggerManager::new();
        let sched = CronSchedule::parse("*/5 * * * *").unwrap();
        mgr.register(Trigger::new(
            "every-5-min",
            TriggerCondition::Cron(sched),
            make_action(),
        ));

        // minute=10 matches */5
        assert_eq!(mgr.check_cron(10, 3, 1, 1, 0).len(), 1);
        // minute=7 does not match */5
        assert_eq!(mgr.check_cron(7, 3, 1, 1, 0).len(), 0);
    }

    // 10. test_check_file_change
    #[test]
    fn test_check_file_change() {
        let mut mgr = TriggerManager::new();
        mgr.register(Trigger::new(
            "src-change",
            TriggerCondition::FileChanged {
                path: PathBuf::from("/project/src"),
                pattern: Some(".rs".into()),
            },
            make_action(),
        ));

        assert_eq!(mgr.check_file_change("/project/src/main.rs").len(), 1);
        assert_eq!(mgr.check_file_change("/project/src/data.json").len(), 0); // no .rs
        assert_eq!(mgr.check_file_change("/other/main.rs").len(), 0); // wrong path
    }

    // 11. test_check_feed_update
    #[test]
    fn test_check_feed_update() {
        let mut mgr = TriggerManager::new();
        mgr.register(Trigger::new(
            "rss-update",
            TriggerCondition::FeedUpdate {
                feed_url: "https://example.com/feed.xml".into(),
            },
            make_action(),
        ));

        assert_eq!(
            mgr.check_feed_update("https://example.com/feed.xml").len(),
            1
        );
        assert_eq!(mgr.check_feed_update("https://other.com/feed.xml").len(), 0);
    }

    // 12. test_trigger_event_bridge
    #[test]
    fn test_trigger_event_bridge() {
        let mgr = Arc::new(Mutex::new(TriggerManager::new()));
        {
            let mut m = mgr.lock().unwrap();
            m.register(Trigger::new(
                "bridge-test",
                TriggerCondition::AiEventMatch {
                    event_name: "tool_executed".into(),
                },
                make_action(),
            ));
        }

        let bridge = TriggerEventBridge::new(Arc::clone(&mgr));
        let event = AiEvent::ToolExecuted {
            name: "test".into(),
            success: true,
            duration_ms: 50,
        };

        bridge.on_event(&event);

        let m = mgr.lock().unwrap();
        assert_eq!(m.fire_log().len(), 1);
        assert_eq!(m.total_fires(), 1);
    }

    // 13. test_enabled_triggers
    #[test]
    fn test_enabled_triggers() {
        let mut mgr = TriggerManager::new();
        let id1 = mgr.register(Trigger::new("t1", TriggerCondition::Manual, make_action()));
        let _id2 = mgr.register(Trigger::new("t2", TriggerCondition::Manual, make_action()));

        assert_eq!(mgr.enabled_triggers().len(), 2);
        mgr.set_enabled(&id1, false);
        assert_eq!(mgr.enabled_triggers().len(), 1);
    }

    // 14. test_total_fires
    #[test]
    fn test_total_fires() {
        let mut mgr = TriggerManager::new();
        let id1 = mgr.register(Trigger::new("t1", TriggerCondition::Manual, make_action()));
        let id2 = mgr.register(Trigger::new("t2", TriggerCondition::Manual, make_action()));

        mgr.fire_trigger(&id1).unwrap();
        mgr.fire_trigger(&id1).unwrap();
        mgr.fire_trigger(&id2).unwrap();

        assert_eq!(mgr.total_fires(), 3);
        assert_eq!(mgr.fire_log().len(), 3);
    }

    // 15. test_fire_trigger_manually
    #[test]
    fn test_fire_trigger_manually() {
        let mut mgr = TriggerManager::new();
        let trigger = Trigger::new("manual-test", TriggerCondition::Manual, make_action());
        let id = mgr.register(trigger);

        assert_eq!(mgr.get(&id).unwrap().fire_count, 0);
        assert!(mgr.get(&id).unwrap().last_fired.is_none());

        let fired = mgr.fire_trigger(&id).unwrap();
        assert_eq!(fired.trigger_name, "manual-test");

        let t = mgr.get(&id).unwrap();
        assert_eq!(t.fire_count, 1);
        assert!(t.last_fired.is_some());
    }

    // 16. test_trigger_cooldown
    #[test]
    fn test_trigger_cooldown() {
        let mut mgr = TriggerManager::new();
        let trigger = Trigger::new("cooldown-test", TriggerCondition::Manual, make_action())
            .with_cooldown_ms(120_000); // 120 seconds — well above any test duration
        let id = mgr.register(trigger);

        // First fire succeeds
        assert!(mgr.fire_trigger(&id).is_ok());
        assert_eq!(mgr.get(&id).unwrap().fire_count, 1);

        // Immediate second fire is rejected (still in cooldown)
        let err = mgr.fire_trigger(&id).unwrap_err();
        assert!(err.contains("cooldown"));
        assert_eq!(mgr.get(&id).unwrap().fire_count, 1);
    }

    // 17. test_trigger_max_fires
    #[test]
    fn test_trigger_max_fires() {
        let mut mgr = TriggerManager::new();
        let trigger = Trigger::new("max-fires-test", TriggerCondition::Manual, make_action())
            .with_max_fires(2);
        let id = mgr.register(trigger);

        // First and second fires succeed
        assert!(mgr.fire_trigger(&id).is_ok());
        assert!(mgr.fire_trigger(&id).is_ok());
        assert_eq!(mgr.get(&id).unwrap().fire_count, 2);

        // Third fire fails — exhausted
        let err = mgr.fire_trigger(&id).unwrap_err();
        assert!(err.contains("exhausted") || err.contains("fire limit"));
        assert_eq!(mgr.get(&id).unwrap().fire_count, 2);
    }

    // 18. test_clear_trigger_log
    #[test]
    fn test_clear_trigger_log() {
        let mut mgr = TriggerManager::new();
        let id1 = mgr.register(Trigger::new("a", TriggerCondition::Manual, make_action()));
        let id2 = mgr.register(Trigger::new("b", TriggerCondition::Manual, make_action()));

        mgr.fire_trigger(&id1).unwrap();
        mgr.fire_trigger(&id2).unwrap();
        assert_eq!(mgr.fire_log().len(), 2);

        mgr.clear_log();
        assert!(mgr.fire_log().is_empty());
    }

    // 19. test_get_trigger_mut
    #[test]
    fn test_get_trigger_mut() {
        let mut mgr = TriggerManager::new();
        let id = mgr.register(Trigger::new(
            "mutable",
            TriggerCondition::Manual,
            make_action(),
        ));

        // Mutate the trigger's name via get_mut
        {
            let t = mgr.get_mut(&id).unwrap();
            t.name = "renamed".to_string();
        }

        // Verify the change persists
        assert_eq!(mgr.get(&id).unwrap().name, "renamed");
    }

    // 20. test_disable_trigger
    #[test]
    fn test_disable_trigger() {
        let mut mgr = TriggerManager::new();
        let id = mgr.register(Trigger::new(
            "disable-me",
            TriggerCondition::Manual,
            make_action(),
        ));

        // Disable the trigger
        mgr.set_enabled(&id, false);
        assert!(!mgr.get(&id).unwrap().enabled);

        // Attempt to fire — should fail because it is disabled
        let err = mgr.fire_trigger(&id).unwrap_err();
        assert!(err.contains("disabled"));
        assert_eq!(mgr.get(&id).unwrap().fire_count, 0);
    }

    // ========================================================================
    // CronExpression & SchedulerRunner tests
    // ========================================================================

    // 21. test_cron_parse_every_minute
    #[test]
    fn test_cron_parse_every_minute() {
        let expr = CronExpression::parse("* * * * *").unwrap();
        assert_eq!(expr.minute, TsCronField::Any);
        assert_eq!(expr.hour, TsCronField::Any);
        assert_eq!(expr.day_of_month, TsCronField::Any);
        assert_eq!(expr.month, TsCronField::Any);
        assert_eq!(expr.day_of_week, TsCronField::Any);
    }

    // 22. test_cron_parse_specific
    #[test]
    fn test_cron_parse_specific() {
        let expr = CronExpression::parse("30 14 * * 1").unwrap();
        assert_eq!(expr.minute, TsCronField::Exact(30));
        assert_eq!(expr.hour, TsCronField::Exact(14));
        assert_eq!(expr.day_of_month, TsCronField::Any);
        assert_eq!(expr.month, TsCronField::Any);
        assert_eq!(expr.day_of_week, TsCronField::Exact(1));
    }

    // 23. test_cron_parse_range_and_step
    #[test]
    fn test_cron_parse_range_and_step() {
        let expr = CronExpression::parse("*/15 9-17 * * *").unwrap();
        assert_eq!(expr.minute, TsCronField::Step(15));
        assert_eq!(expr.hour, TsCronField::Range(9, 17));
        assert_eq!(expr.day_of_month, TsCronField::Any);
        assert_eq!(expr.month, TsCronField::Any);
        assert_eq!(expr.day_of_week, TsCronField::Any);
    }

    // 24. test_cron_parse_list
    #[test]
    fn test_cron_parse_list() {
        let expr = CronExpression::parse("0,15,30,45 * * * *").unwrap();
        assert_eq!(expr.minute, TsCronField::List(vec![0, 15, 30, 45]));
        assert_eq!(expr.hour, TsCronField::Any);
    }

    // 25. test_cron_matching
    #[test]
    fn test_cron_matching() {
        let expr = CronExpression::parse("30 14 * * 1").unwrap();
        // Should match: minute=30, hour=14, dom=15, month=6, dow=1
        assert!(expr.matches(30, 14, 15, 6, 1));
        // Should NOT match: minute=0 (wrong minute)
        assert!(!expr.matches(0, 14, 15, 6, 1));
    }

    // 26. test_scheduler_tick
    #[test]
    fn test_scheduler_tick() {
        let mut mgr = TriggerManager::new();
        // Register a cron trigger that matches minute=0 (*/5 fires at 0,5,10...)
        let sched = CronSchedule::parse("*/5 * * * *").unwrap();
        mgr.register(Trigger::new(
            "every-5",
            TriggerCondition::Cron(sched),
            make_action(),
        ));

        let config = SchedulerConfig::default();
        let mut runner = SchedulerRunner::new(config, mgr);

        // Tick at time 0 — minute=0 matches */5
        let fired = runner.tick(0);
        assert!(!fired.is_empty(), "cron trigger should fire at minute 0");
        assert_eq!(runner.state().tick_count, 1);
        assert_eq!(runner.state().fired_history.len(), fired.len());
    }

    // 27. test_scheduler_run_n_ticks
    #[test]
    fn test_scheduler_run_n_ticks() {
        let mut mgr = TriggerManager::new();
        // Manual trigger (won't auto-fire on cron tick, but tick_count still increments)
        mgr.register(Trigger::new(
            "manual-t",
            TriggerCondition::Manual,
            make_action(),
        ));

        let config = SchedulerConfig {
            poll_interval_ms: 1000,
            ..SchedulerConfig::default()
        };
        let mut runner = SchedulerRunner::new(config, mgr);

        let _fired = runner.run_n_ticks(5, 0);
        assert_eq!(runner.state().tick_count, 5);
        // last_tick_ms should be start + 4 * interval = 4000
        assert_eq!(runner.state().last_tick_ms, 4000);
    }

    // 28. test_scheduler_export_json
    #[test]
    fn test_scheduler_export_json() {
        let mgr = TriggerManager::new();
        let config = SchedulerConfig::default();
        let runner = SchedulerRunner::new(config, mgr);

        let json = runner.export_json();
        assert!(json.get("config").is_some());
        assert!(json.get("state").is_some());
        assert!(json.get("fired_history_count").is_some());
        assert!(json.get("errors_count").is_some());
        assert!(json.get("trigger_count").is_some());

        // Verify config values
        assert_eq!(json["config"]["poll_interval_ms"], 1000);
        assert_eq!(json["config"]["max_concurrent"], 4);
        assert_eq!(json["state"]["tick_count"], 0);
        assert_eq!(json["trigger_count"], 0);
    }
}
