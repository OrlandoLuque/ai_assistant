//! Scheduler — cron-like job scheduling
//!
//! Manages scheduled jobs with cron expressions. Determines next run times.
//! This module does NOT execute jobs — it only manages schedules, parses
//! cron expressions, and determines next occurrence times.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// CronField
// =============================================================================

/// Represents a single field in a cron expression.
#[derive(Debug, Clone)]
pub enum CronField {
    /// Wildcard `*` — matches any value.
    Any,
    /// A single numeric value, e.g. `5`.
    Value(u32),
    /// An inclusive range, e.g. `1-5`.
    Range(u32, u32),
    /// A list of values, e.g. `1,3,5`.
    List(Vec<u32>),
    /// A step value, e.g. `*/5`.
    Step(u32),
}

impl CronField {
    /// Check whether the given `value` matches this field.
    pub fn matches(&self, value: u32) -> bool {
        match self {
            CronField::Any => true,
            CronField::Value(v) => value == *v,
            CronField::Range(lo, hi) => value >= *lo && value <= *hi,
            CronField::List(vs) => vs.contains(&value),
            CronField::Step(s) => {
                if *s == 0 {
                    return false;
                }
                value % s == 0
            }
        }
    }

    /// Parse a single cron field token.
    ///
    /// Supported formats:
    /// - `"*"` → `Any`
    /// - `"5"` → `Value(5)`
    /// - `"1-5"` → `Range(1, 5)`
    /// - `"1,3,5"` → `List(vec![1, 3, 5])`
    /// - `"*/5"` → `Step(5)`
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        if s == "*" {
            return Ok(CronField::Any);
        }

        // Step: */N
        if let Some(rest) = s.strip_prefix("*/") {
            let step: u32 = rest
                .parse()
                .map_err(|_| format!("invalid step value: {}", rest))?;
            if step == 0 {
                return Err("step value must be > 0".to_string());
            }
            return Ok(CronField::Step(step));
        }

        // List: contains comma
        if s.contains(',') {
            let values: Result<Vec<u32>, _> = s.split(',').map(|v| {
                v.trim()
                    .parse::<u32>()
                    .map_err(|_| format!("invalid list value: {}", v.trim()))
            }).collect();
            return Ok(CronField::List(values?));
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
            return Ok(CronField::Range(lo, hi));
        }

        // Single value
        let v: u32 = s
            .parse()
            .map_err(|_| format!("invalid cron field: {}", s))?;
        Ok(CronField::Value(v))
    }
}

// =============================================================================
// CronSchedule
// =============================================================================

/// A parsed cron schedule with five fields (minute, hour, day-of-month, month,
/// day-of-week).
#[derive(Debug, Clone)]
pub struct CronSchedule {
    /// Minute field (0-59).
    pub minute: CronField,
    /// Hour field (0-23).
    pub hour: CronField,
    /// Day-of-month field (1-31).
    pub day_of_month: CronField,
    /// Month field (1-12).
    pub month: CronField,
    /// Day-of-week field (0-6, where 0 = Sunday).
    pub day_of_week: CronField,
}

impl CronSchedule {
    /// Parse a standard five-field cron expression such as `"*/5 * * * *"`.
    pub fn parse(expr: &str) -> Result<Self, String> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(format!(
                "expected 5 cron fields, got {} in '{}'",
                fields.len(),
                expr
            ));
        }
        Ok(CronSchedule {
            minute: CronField::parse(fields[0])?,
            hour: CronField::parse(fields[1])?,
            day_of_month: CronField::parse(fields[2])?,
            month: CronField::parse(fields[3])?,
            day_of_week: CronField::parse(fields[4])?,
        })
    }

    /// Check if this schedule matches the given broken-down time.
    pub fn matches(
        &self,
        minute: u32,
        hour: u32,
        day: u32,
        month: u32,
        weekday: u32,
    ) -> bool {
        self.minute.matches(minute)
            && self.hour.matches(hour)
            && self.day_of_month.matches(day)
            && self.month.matches(month)
            && self.day_of_week.matches(weekday)
    }

    /// Return a human-readable description of this schedule.
    pub fn describe(&self) -> String {
        // Simple common-case descriptions
        match (&self.minute, &self.hour, &self.day_of_month, &self.month, &self.day_of_week) {
            // Every N minutes: */N * * * *
            (CronField::Step(m), CronField::Any, CronField::Any, CronField::Any, CronField::Any) => {
                format!("Every {} minutes", m)
            }
            // Every minute: * * * * *
            (CronField::Any, CronField::Any, CronField::Any, CronField::Any, CronField::Any) => {
                "Every minute".to_string()
            }
            // At specific minute every hour: N * * * *
            (CronField::Value(m), CronField::Any, CronField::Any, CronField::Any, CronField::Any) => {
                format!("At minute {} of every hour", m)
            }
            // Daily at midnight: 0 0 * * *
            (CronField::Value(0), CronField::Value(0), CronField::Any, CronField::Any, CronField::Any) => {
                "Daily at midnight".to_string()
            }
            // At specific hour:minute every day: M H * * *
            (CronField::Value(m), CronField::Value(h), CronField::Any, CronField::Any, CronField::Any) => {
                let period = if *h < 12 { "AM" } else { "PM" };
                let display_h = if *h == 0 {
                    12
                } else if *h > 12 {
                    h - 12
                } else {
                    *h
                };
                format!("At {}:{:02} {}", display_h, m, period)
            }
            // Fallback: reconstruct the expression
            _ => {
                format!(
                    "Cron: {} {} {} {} {}",
                    describe_field(&self.minute),
                    describe_field(&self.hour),
                    describe_field(&self.day_of_month),
                    describe_field(&self.month),
                    describe_field(&self.day_of_week),
                )
            }
        }
    }
}

/// Helper to render a single `CronField` back to its textual representation.
fn describe_field(field: &CronField) -> String {
    match field {
        CronField::Any => "*".to_string(),
        CronField::Value(v) => v.to_string(),
        CronField::Range(lo, hi) => format!("{}-{}", lo, hi),
        CronField::List(vs) => vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        CronField::Step(s) => format!("*/{}", s),
    }
}

// =============================================================================
// ScheduledAction
// =============================================================================

/// The action a scheduled job should trigger when it fires.
#[derive(Debug, Clone)]
pub enum ScheduledAction {
    /// Run an agent with the given profile and task description.
    RunAgent { profile: String, task: String },
    /// Run a named tool with parameters.
    RunTool {
        tool_name: String,
        params: HashMap<String, String>,
    },
    /// Run a workflow profile with variables.
    RunWorkflow {
        profile: String,
        variables: HashMap<String, String>,
    },
    /// Run a shell command.
    RunShell { command: String },
    /// Custom/extensible action.
    Custom { action_type: String, payload: String },
}

// =============================================================================
// ScheduledJob
// =============================================================================

/// A single scheduled job with its schedule, action, and run tracking metadata.
pub struct ScheduledJob {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The cron schedule controlling when this job fires.
    pub schedule: CronSchedule,
    /// The action to perform when the schedule fires.
    pub action: ScheduledAction,
    /// Whether this job is currently enabled.
    pub enabled: bool,
    /// Timestamp (ms since epoch) of the last run, if any.
    pub last_run: Option<u64>,
    /// Computed timestamp (ms since epoch) of the next run, if known.
    pub next_run: Option<u64>,
    /// How many times this job has been run.
    pub run_count: u32,
    /// Maximum number of runs before the job is exhausted. `None` = unlimited.
    pub max_runs: Option<u32>,
    /// Timestamp (ms since epoch) when this job was created.
    pub created_at: u64,
}

impl ScheduledJob {
    /// Create a new scheduled job with sensible defaults.
    ///
    /// The `id` field will be empty initially and is expected to be set by the
    /// `Scheduler` when the job is added.
    pub fn new(
        name: impl Into<String>,
        schedule: CronSchedule,
        action: ScheduledAction,
    ) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        ScheduledJob {
            id: String::new(),
            name: name.into(),
            schedule,
            action,
            enabled: true,
            last_run: None,
            next_run: None,
            run_count: 0,
            max_runs: None,
            created_at: now_ms,
        }
    }

    /// Returns `true` if this job has reached its `max_runs` limit.
    pub fn is_exhausted(&self) -> bool {
        match self.max_runs {
            Some(max) => self.run_count >= max,
            None => false,
        }
    }
}

// =============================================================================
// Scheduler
// =============================================================================

/// A cron-like scheduler that stores jobs and determines when they should run.
pub struct Scheduler {
    jobs: Vec<ScheduledJob>,
    next_id: u64,
}

impl Scheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Scheduler {
            jobs: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a job to the scheduler. Returns the assigned job ID.
    pub fn add_job(&mut self, mut job: ScheduledJob) -> String {
        let id = format!("job-{}", self.next_id);
        self.next_id += 1;
        job.id = id.clone();
        self.jobs.push(job);
        id
    }

    /// Remove a job by ID. Returns `true` if a job was removed.
    pub fn remove_job(&mut self, id: &str) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        self.jobs.len() < before
    }

    /// Get an immutable reference to a job by ID.
    pub fn get_job(&self, id: &str) -> Option<&ScheduledJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Get a mutable reference to a job by ID.
    pub fn get_job_mut(&mut self, id: &str) -> Option<&mut ScheduledJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Return a slice of all jobs.
    pub fn list_jobs(&self) -> &[ScheduledJob] {
        &self.jobs
    }

    /// Return references to all enabled (and non-exhausted) jobs.
    pub fn enabled_jobs(&self) -> Vec<&ScheduledJob> {
        self.jobs
            .iter()
            .filter(|j| j.enabled && !j.is_exhausted())
            .collect()
    }

    /// Determine which jobs should fire for the given broken-down time.
    ///
    /// Only returns jobs that are enabled, not exhausted, and whose schedule
    /// matches the provided time.
    pub fn due_jobs(
        &self,
        minute: u32,
        hour: u32,
        day: u32,
        month: u32,
        weekday: u32,
    ) -> Vec<&ScheduledJob> {
        self.jobs
            .iter()
            .filter(|j| {
                j.enabled
                    && !j.is_exhausted()
                    && j.schedule.matches(minute, hour, day, month, weekday)
            })
            .collect()
    }

    /// Mark a job as having been run at the current time. Increments `run_count`
    /// and sets `last_run`.
    pub fn mark_run(&mut self, id: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            job.last_run = Some(now_ms);
            job.run_count += 1;
        }
    }

    /// Enable or disable a job by ID. Returns `true` if the job was found.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> bool {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            job.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Return the total number of jobs (enabled or not).
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Remove all jobs.
    pub fn clear(&mut self) {
        self.jobs.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. CronField::Any matches everything
    #[test]
    fn test_cron_field_any_matches_all() {
        let field = CronField::Any;
        assert!(field.matches(0));
        assert!(field.matches(30));
        assert!(field.matches(59));
        assert!(field.matches(999));
    }

    // 2. CronField::Value
    #[test]
    fn test_cron_field_value() {
        let field = CronField::Value(5);
        assert!(field.matches(5));
        assert!(!field.matches(0));
        assert!(!field.matches(4));
        assert!(!field.matches(6));
    }

    // 3. CronField::Range
    #[test]
    fn test_cron_field_range() {
        let field = CronField::Range(1, 5);
        assert!(!field.matches(0));
        assert!(field.matches(1));
        assert!(field.matches(3));
        assert!(field.matches(5));
        assert!(!field.matches(6));
    }

    // 4. CronField::List
    #[test]
    fn test_cron_field_list() {
        let field = CronField::List(vec![1, 3, 5]);
        assert!(field.matches(1));
        assert!(!field.matches(2));
        assert!(field.matches(3));
        assert!(!field.matches(4));
        assert!(field.matches(5));
    }

    // 5. CronField::Step
    #[test]
    fn test_cron_field_step() {
        let field = CronField::Step(5);
        assert!(field.matches(0));
        assert!(!field.matches(1));
        assert!(field.matches(5));
        assert!(field.matches(10));
        assert!(field.matches(15));
        assert!(!field.matches(7));
    }

    // 6. CronField::parse
    #[test]
    fn test_cron_field_parse() {
        // Any
        assert!(matches!(CronField::parse("*").unwrap(), CronField::Any));

        // Value
        match CronField::parse("5").unwrap() {
            CronField::Value(5) => {}
            _ => panic!("expected Value(5)"),
        }

        // Range
        match CronField::parse("1-5").unwrap() {
            CronField::Range(1, 5) => {}
            _ => panic!("expected Range(1, 5)"),
        }

        // List
        match CronField::parse("1,3,5").unwrap() {
            CronField::List(vs) => assert_eq!(vs, vec![1, 3, 5]),
            _ => panic!("expected List"),
        }

        // Step
        match CronField::parse("*/5").unwrap() {
            CronField::Step(5) => {}
            _ => panic!("expected Step(5)"),
        }

        // Errors
        assert!(CronField::parse("abc").is_err());
        assert!(CronField::parse("*/0").is_err());
    }

    // 7. CronSchedule::parse — every 5 minutes
    #[test]
    fn test_cron_schedule_parse_every_5_min() {
        let schedule = CronSchedule::parse("*/5 * * * *").unwrap();
        assert!(matches!(schedule.minute, CronField::Step(5)));
        assert!(matches!(schedule.hour, CronField::Any));
        assert!(matches!(schedule.day_of_month, CronField::Any));
        assert!(matches!(schedule.month, CronField::Any));
        assert!(matches!(schedule.day_of_week, CronField::Any));

        // Wrong number of fields
        assert!(CronSchedule::parse("* * *").is_err());
        assert!(CronSchedule::parse("* * * * * *").is_err());
    }

    // 8. CronSchedule::matches
    #[test]
    fn test_cron_schedule_matches() {
        // Every 5 minutes: should match minute=0, 5, 10, etc.
        let every5 = CronSchedule::parse("*/5 * * * *").unwrap();
        assert!(every5.matches(0, 12, 15, 6, 3));
        assert!(every5.matches(5, 0, 1, 1, 0));
        assert!(every5.matches(10, 23, 31, 12, 6));
        assert!(!every5.matches(3, 12, 15, 6, 3));

        // At 3:00 AM daily
        let at_three = CronSchedule::parse("0 3 * * *").unwrap();
        assert!(at_three.matches(0, 3, 1, 1, 0));
        assert!(!at_three.matches(0, 4, 1, 1, 0));
        assert!(!at_three.matches(1, 3, 1, 1, 0));

        // Weekdays only (1-5 = Mon-Fri)
        let weekdays = CronSchedule::parse("0 9 * * 1-5").unwrap();
        assert!(weekdays.matches(0, 9, 15, 6, 1));  // Monday
        assert!(weekdays.matches(0, 9, 15, 6, 5));  // Friday
        assert!(!weekdays.matches(0, 9, 15, 6, 0)); // Sunday
        assert!(!weekdays.matches(0, 9, 15, 6, 6)); // Saturday
    }

    // 9. CronSchedule::describe
    #[test]
    fn test_cron_schedule_describe() {
        let every5 = CronSchedule::parse("*/5 * * * *").unwrap();
        assert_eq!(every5.describe(), "Every 5 minutes");

        let at_three = CronSchedule::parse("0 3 * * *").unwrap();
        assert_eq!(at_three.describe(), "At 3:00 AM");

        let midnight = CronSchedule::parse("0 0 * * *").unwrap();
        assert_eq!(midnight.describe(), "Daily at midnight");

        let every_min = CronSchedule::parse("* * * * *").unwrap();
        assert_eq!(every_min.describe(), "Every minute");

        let afternoon = CronSchedule::parse("30 14 * * *").unwrap();
        assert_eq!(afternoon.describe(), "At 2:30 PM");
    }

    // 10. Scheduler add/remove
    #[test]
    fn test_scheduler_add_remove() {
        let mut sched = Scheduler::new();
        assert_eq!(sched.job_count(), 0);

        let job1 = ScheduledJob::new(
            "Job 1",
            CronSchedule::parse("*/5 * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo hi".into() },
        );
        let id1 = sched.add_job(job1);
        assert_eq!(sched.job_count(), 1);
        assert!(sched.get_job(&id1).is_some());
        assert_eq!(sched.get_job(&id1).unwrap().name, "Job 1");

        let job2 = ScheduledJob::new(
            "Job 2",
            CronSchedule::parse("0 0 * * *").unwrap(),
            ScheduledAction::Custom {
                action_type: "test".into(),
                payload: "{}".into(),
            },
        );
        let id2 = sched.add_job(job2);
        assert_eq!(sched.job_count(), 2);

        // Remove first job
        assert!(sched.remove_job(&id1));
        assert_eq!(sched.job_count(), 1);
        assert!(sched.get_job(&id1).is_none());
        assert!(sched.get_job(&id2).is_some());

        // Remove non-existent
        assert!(!sched.remove_job("no-such-id"));

        // Clear
        sched.clear();
        assert_eq!(sched.job_count(), 0);
    }

    // 11. Scheduler due_jobs
    #[test]
    fn test_scheduler_due_jobs() {
        let mut sched = Scheduler::new();

        let job_every5 = ScheduledJob::new(
            "Every 5 min",
            CronSchedule::parse("*/5 * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo 5".into() },
        );
        let id_every5 = sched.add_job(job_every5);

        let job_midnight = ScheduledJob::new(
            "Midnight",
            CronSchedule::parse("0 0 * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo midnight".into() },
        );
        let _id_midnight = sched.add_job(job_midnight);

        // At minute=0, hour=0 both should fire
        let due = sched.due_jobs(0, 0, 1, 1, 3);
        assert_eq!(due.len(), 2);

        // At minute=5, hour=12 only every-5 fires
        let due = sched.due_jobs(5, 12, 1, 1, 3);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].name, "Every 5 min");

        // At minute=3, hour=12 nothing fires
        let due = sched.due_jobs(3, 12, 1, 1, 3);
        assert_eq!(due.len(), 0);

        // Disable a job — it should no longer appear
        sched.set_enabled(&id_every5, false);
        let due = sched.due_jobs(5, 12, 1, 1, 3);
        assert_eq!(due.len(), 0);
    }

    // 12. Scheduler mark_run and exhaustion
    #[test]
    fn test_scheduler_mark_run_exhausted() {
        let mut sched = Scheduler::new();

        let mut job = ScheduledJob::new(
            "Limited",
            CronSchedule::parse("* * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo limited".into() },
        );
        job.max_runs = Some(2);
        let id = sched.add_job(job);

        // Should be due initially
        let due = sched.due_jobs(0, 0, 1, 1, 0);
        assert_eq!(due.len(), 1);

        // First run
        sched.mark_run(&id);
        let j = sched.get_job(&id).unwrap();
        assert_eq!(j.run_count, 1);
        assert!(j.last_run.is_some());
        assert!(!j.is_exhausted());

        // Second run
        sched.mark_run(&id);
        let j = sched.get_job(&id).unwrap();
        assert_eq!(j.run_count, 2);
        assert!(j.is_exhausted());

        // Exhausted — should no longer be due
        let due = sched.due_jobs(0, 0, 1, 1, 0);
        assert_eq!(due.len(), 0);

        // enabled_jobs also excludes exhausted
        assert_eq!(sched.enabled_jobs().len(), 0);
    }

    // 13. get_job_mut — modify job through mutable reference
    #[test]
    fn test_get_job_mut() {
        let mut sched = Scheduler::new();

        let job = ScheduledJob::new(
            "Mutable Job",
            CronSchedule::parse("*/10 * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo hello".into() },
        );
        let id = sched.add_job(job);

        // Get mutable reference and modify the job
        let job_mut = sched.get_job_mut(&id).unwrap();
        assert_eq!(job_mut.name, "Mutable Job");
        job_mut.name = "Renamed Job".to_string();
        job_mut.enabled = false;

        // Verify change persists through immutable reference
        let job_ref = sched.get_job(&id).unwrap();
        assert_eq!(job_ref.name, "Renamed Job");
        assert!(!job_ref.enabled);
    }

    // 14. list_jobs — verify all jobs are returned
    #[test]
    fn test_list_jobs() {
        let mut sched = Scheduler::new();

        let job1 = ScheduledJob::new(
            "Job A",
            CronSchedule::parse("*/5 * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo a".into() },
        );
        let job2 = ScheduledJob::new(
            "Job B",
            CronSchedule::parse("0 0 * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo b".into() },
        );
        let job3 = ScheduledJob::new(
            "Job C",
            CronSchedule::parse("30 12 * * *").unwrap(),
            ScheduledAction::Custom {
                action_type: "notify".into(),
                payload: "{}".into(),
            },
        );

        sched.add_job(job1);
        sched.add_job(job2);
        sched.add_job(job3);

        let jobs = sched.list_jobs();
        assert_eq!(jobs.len(), 3);

        let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
        assert!(names.contains(&"Job A"));
        assert!(names.contains(&"Job B"));
        assert!(names.contains(&"Job C"));
    }

    // 15. remove_nonexistent — returns false
    #[test]
    fn test_remove_nonexistent() {
        let mut sched = Scheduler::new();

        // Empty scheduler
        assert!(!sched.remove_job("no-such-job"));

        // Add a job, try to remove a different one
        let job = ScheduledJob::new(
            "Real Job",
            CronSchedule::parse("* * * * *").unwrap(),
            ScheduledAction::RunShell { command: "echo real".into() },
        );
        let _id = sched.add_job(job);
        assert_eq!(sched.job_count(), 1);

        assert!(!sched.remove_job("fake-id"));
        assert_eq!(sched.job_count(), 1);
    }

    // 16. cron next_occurrence edge case — day 31 only matches months with 31 days
    #[test]
    fn test_cron_next_occurrence_edge() {
        // "0 0 31 * *" — midnight on the 31st of every month
        let schedule = CronSchedule::parse("0 0 31 * *").unwrap();

        // January has 31 days → should match
        assert!(schedule.matches(0, 0, 31, 1, 3));
        // March has 31 days → should match
        assert!(schedule.matches(0, 0, 31, 3, 5));

        // Day 30 should NOT match
        assert!(!schedule.matches(0, 0, 30, 1, 2));
        // Day 28 should NOT match
        assert!(!schedule.matches(0, 0, 28, 2, 0));
        // Day 31 at wrong hour should NOT match
        assert!(!schedule.matches(0, 1, 31, 1, 3));
        // Day 31 at wrong minute should NOT match
        assert!(!schedule.matches(1, 0, 31, 1, 3));
    }
}
