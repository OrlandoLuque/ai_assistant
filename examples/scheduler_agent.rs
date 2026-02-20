//! Scheduler and trigger system example.
//!
//! Run with: cargo run --example scheduler_agent --features "scheduler"
//!
//! Demonstrates building a Scheduler with cron expressions, adding jobs,
//! checking which jobs are due, and using the TriggerManager + SchedulerRunner
//! to fire triggers based on cron ticks and manual events.

use ai_assistant::scheduler::{CronSchedule, ScheduledAction, ScheduledJob, Scheduler};
use ai_assistant::trigger_system::{
    FiredTrigger, SchedulerConfig, SchedulerRunner, Trigger, TriggerCondition, TriggerManager,
};

fn main() {
    println!("=== Scheduler & Trigger System Demo ===\n");

    // -------------------------------------------------------------------------
    // Part 1: Basic Scheduler — cron parsing, job management, due-job queries
    // -------------------------------------------------------------------------

    let mut scheduler = Scheduler::new();

    // Job 1: runs every 5 minutes
    let job1 = ScheduledJob::new(
        "Health check",
        CronSchedule::parse("*/5 * * * *").expect("valid cron"),
        ScheduledAction::RunShell {
            command: "echo 'health-ok'".into(),
        },
    );
    let id1 = scheduler.add_job(job1);
    println!(
        "Added job '{}': {}",
        id1,
        scheduler.get_job(&id1).unwrap().name
    );

    // Job 2: runs daily at 3:00 AM
    let job2 = ScheduledJob::new(
        "Nightly backup",
        CronSchedule::parse("0 3 * * *").expect("valid cron"),
        ScheduledAction::RunAgent {
            profile: "backup-agent".into(),
            task: "Run nightly backup of knowledge bases".into(),
        },
    );
    let id2 = scheduler.add_job(job2);
    println!(
        "Added job '{}': {}",
        id2,
        scheduler.get_job(&id2).unwrap().name
    );

    // Job 3: limited to 3 runs
    let mut job3 = ScheduledJob::new(
        "One-time report",
        CronSchedule::parse("30 12 * * *").expect("valid cron"),
        ScheduledAction::Custom {
            action_type: "report".into(),
            payload: r#"{"format":"pdf"}"#.into(),
        },
    );
    job3.max_runs = Some(3);
    let id3 = scheduler.add_job(job3);
    println!(
        "Added job '{}': {} (max 3 runs)",
        id3,
        scheduler.get_job(&id3).unwrap().name
    );

    // Describe schedules
    println!("\nSchedule descriptions:");
    for job in scheduler.list_jobs() {
        println!("  {}: {}", job.name, job.schedule.describe());
    }

    // Check which jobs are due at minute=0, hour=3 (3:00 AM)
    let due = scheduler.due_jobs(0, 3, 15, 6, 2);
    println!("\nJobs due at 03:00 on June 15 (Tuesday):");
    for job in &due {
        println!("  - {} (id: {})", job.name, job.id);
    }

    // Mark a job as run and check exhaustion
    scheduler.mark_run(&id3);
    scheduler.mark_run(&id3);
    scheduler.mark_run(&id3);
    let j3 = scheduler.get_job(&id3).unwrap();
    println!(
        "\n'{}' run {} times, exhausted: {}",
        j3.name,
        j3.run_count,
        j3.is_exhausted()
    );

    // -------------------------------------------------------------------------
    // Part 2: TriggerManager + SchedulerRunner — event-driven execution
    // -------------------------------------------------------------------------

    println!("\n=== Trigger Manager + Scheduler Runner ===\n");

    let mut trigger_mgr = TriggerManager::new();

    // Cron-based trigger: every 5 minutes
    let cron_sched = CronSchedule::parse("*/5 * * * *").expect("valid cron");
    let cron_trigger = Trigger::new(
        "Periodic sync",
        TriggerCondition::Cron(cron_sched),
        ScheduledAction::RunShell {
            command: "sync-data --incremental".into(),
        },
    );
    let tid1 = trigger_mgr.register(cron_trigger);
    println!("Registered trigger '{}': Periodic sync", tid1);

    // Manual trigger with max 2 fires
    let manual_trigger = Trigger::new(
        "Deploy trigger",
        TriggerCondition::Manual,
        ScheduledAction::RunShell {
            command: "deploy --production".into(),
        },
    )
    .with_max_fires(2);
    let tid2 = trigger_mgr.register(manual_trigger);
    println!("Registered trigger '{}': Deploy trigger (max 2)", tid2);

    // Fire the manual trigger twice
    let fired1 = trigger_mgr.fire_trigger(&tid2).expect("first fire");
    println!("\nFired '{}' at {}", fired1.trigger_name, fired1.fired_at);

    let fired2 = trigger_mgr.fire_trigger(&tid2).expect("second fire");
    println!("Fired '{}' at {}", fired2.trigger_name, fired2.fired_at);

    // Third fire should fail (exhausted)
    match trigger_mgr.fire_trigger(&tid2) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Third fire correctly rejected: {}", e),
    }

    // Use the SchedulerRunner to tick cron triggers
    let config = SchedulerConfig {
        poll_interval_ms: 60_000, // 1 minute intervals
        ..SchedulerConfig::default()
    };
    let mut runner = SchedulerRunner::new(config, trigger_mgr);

    // Simulate 5 ticks starting at timestamp 0 (epoch)
    println!("\nRunning 5 scheduler ticks...");
    let all_fired: Vec<FiredTrigger> = runner.run_n_ticks(5, 0);
    println!("  Total triggers fired: {}", all_fired.len());
    for f in &all_fired {
        println!("    - {} (at {})", f.trigger_name, f.fired_at);
    }

    // Export state as JSON
    let status = runner.export_json();
    println!(
        "\nScheduler status: {}",
        serde_json::to_string_pretty(&status).unwrap()
    );

    println!("\nDone.");
}
