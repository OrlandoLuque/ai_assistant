//! Memory scheduler: periodic maintenance tasks for the memory system.

use serde::{Deserialize, Serialize};

/// A task that can be scheduled for periodic execution on the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SchedulerTask {
    /// Run memory consolidation.
    Consolidate,
    /// Apply temporal decay to memory relevance scores.
    Decay { decay_rate: f64 },
    /// Compress similar memories together.
    Compress { min_similarity: f64 },
    /// Garbage-collect old, infrequently accessed memories.
    GarbageCollect {
        min_age_secs: u64,
        min_access_count: usize,
    },
}

/// Configuration for the memory scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SchedulerConfig {
    /// Interval between consolidation runs, in seconds.
    pub consolidation_interval_secs: u64,
    /// Interval between decay runs, in seconds.
    pub decay_interval_secs: u64,
    /// Interval between compression runs, in seconds.
    pub compression_interval_secs: u64,
    /// Interval between garbage collection runs, in seconds.
    pub gc_interval_secs: u64,
    /// Number of items to process per batch.
    pub batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            consolidation_interval_secs: 3600,
            decay_interval_secs: 7200,
            compression_interval_secs: 86400,
            gc_interval_secs: 86400,
            batch_size: 100,
        }
    }
}

/// A scheduled job in the memory scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledJob {
    /// The task to execute.
    pub task: SchedulerTask,
    /// Timestamp of last execution (0 = never).
    pub last_run: u64,
    /// Interval between runs, in seconds.
    pub interval_secs: u64,
    /// Whether this job is enabled.
    pub enabled: bool,
}

/// Scheduler that manages periodic memory maintenance tasks.
pub struct MemoryScheduler {
    jobs: Vec<ScheduledJob>,
    config: SchedulerConfig,
}

impl MemoryScheduler {
    /// Create a scheduler with the given config and the four default jobs:
    /// Consolidate, Decay(0.01), Compress(0.85), GarbageCollect(86400, 0).
    pub fn new(config: SchedulerConfig) -> Self {
        let jobs = vec![
            ScheduledJob {
                task: SchedulerTask::Consolidate,
                last_run: 0,
                interval_secs: config.consolidation_interval_secs,
                enabled: true,
            },
            ScheduledJob {
                task: SchedulerTask::Decay { decay_rate: 0.01 },
                last_run: 0,
                interval_secs: config.decay_interval_secs,
                enabled: true,
            },
            ScheduledJob {
                task: SchedulerTask::Compress {
                    min_similarity: 0.85,
                },
                last_run: 0,
                interval_secs: config.compression_interval_secs,
                enabled: true,
            },
            ScheduledJob {
                task: SchedulerTask::GarbageCollect {
                    min_age_secs: 86400,
                    min_access_count: 0,
                },
                last_run: 0,
                interval_secs: config.gc_interval_secs,
                enabled: true,
            },
        ];
        Self { jobs, config }
    }

    /// Create a scheduler with default configuration and default jobs.
    pub fn with_defaults() -> Self {
        Self::new(SchedulerConfig::default())
    }

    /// Add a custom job with the given task and interval.
    pub fn add_job(&mut self, task: SchedulerTask, interval_secs: u64) {
        self.jobs.push(ScheduledJob {
            task,
            last_run: 0,
            interval_secs,
            enabled: true,
        });
    }

    /// Remove a job by index. Returns `true` if the index was valid and the job
    /// was removed.
    pub fn remove_job(&mut self, index: usize) -> bool {
        if index < self.jobs.len() {
            self.jobs.remove(index);
            true
        } else {
            false
        }
    }

    /// Return references to all jobs that are due for execution at
    /// `current_time` (i.e. `current_time - last_run >= interval_secs` and
    /// the job is enabled).
    pub fn due_jobs(&self, current_time: u64) -> Vec<&ScheduledJob> {
        self.jobs
            .iter()
            .filter(|job| {
                job.enabled && current_time.saturating_sub(job.last_run) >= job.interval_secs
            })
            .collect()
    }

    /// Mark a job as completed by updating its `last_run` timestamp.
    pub fn mark_completed(&mut self, index: usize, timestamp: u64) {
        if let Some(job) = self.jobs.get_mut(index) {
            job.last_run = timestamp;
        }
    }

    /// Enable a job by index.
    pub fn enable_job(&mut self, index: usize) {
        if let Some(job) = self.jobs.get_mut(index) {
            job.enabled = true;
        }
    }

    /// Disable a job by index.
    pub fn disable_job(&mut self, index: usize) {
        if let Some(job) = self.jobs.get_mut(index) {
            job.enabled = false;
        }
    }

    /// Return the total number of jobs.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Get a reference to the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}
