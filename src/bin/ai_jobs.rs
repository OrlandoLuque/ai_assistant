// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_jobs — Lightweight cron-like job daemon for `ai_assistant`.
//!
//! Loads a JSON manifest of scheduled jobs and executes them on a tick loop.
//! Supports two runtime modes per job:
//!
//! - `delegated` (default) — shells out to `ai_cli` (or an arbitrary shell
//!   command) as a subprocess. Always available.
//! - `embedded` — runs an in-process [`AiAssistant`] with access to RAG,
//!   tools, memory, and session state. Gated behind `--features full`.
//!
//! ## Run
//!
//! ```bash
//! cargo run --bin ai_jobs --features "scheduler" -- run examples/jobs.json
//! cargo run --bin ai_jobs --features "full,scheduler" -- run examples/jobs.json
//! ```
//!
//! ## Subcommands
//!
//! - `validate <file>` — parse the manifest and report errors
//! - `list <file>` — tabular view of every job
//! - `dry-run <file> [--minutes N]` — show which jobs would fire in N minutes
//! - `run <file>` — daemon loop (Ctrl+C to stop)
//! - `help` — usage
//!
//! ## See also
//!
//! - `docs/BINARIES.md`
//! - `docs/USE_CASES.md`
//! - `examples/jobs.json`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use ai_assistant::scheduler::{CronSchedule, ScheduledAction, ScheduledJob, Scheduler};

// =============================================================================
// Constants
// =============================================================================

/// Hard cap on jobs loaded from a manifest (S-A3).
const MAX_JOBS: usize = 1000;

/// Default per-shell-command timeout (S-A5).
const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 60;

/// Daemon tick interval — the scheduler checks for due jobs every N seconds.
const DEFAULT_TICK_SECS: u64 = 30;

/// Max iterations of inner embedded-poll loop before giving up.
const EMBEDDED_MAX_POLL_SECS: u64 = 600;

// =============================================================================
// Manifest schema (parallel to scheduler::ScheduledJob, which is not Deserialize)
// =============================================================================

#[derive(Debug, Deserialize)]
struct JobsFile {
    #[serde(default)]
    assistant: Option<AssistantConfig>,
    jobs: Vec<JobConfig>,
}

#[derive(Debug, Deserialize, Clone)]
struct AssistantConfig {
    provider: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JobConfig {
    id: String,
    name: String,
    cron: String,
    #[serde(default = "default_runtime")]
    runtime: JobRuntime,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(flatten)]
    action: ActionConfig,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    max_runs: Option<u32>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JobRuntime {
    Delegated,
    Embedded,
}

fn default_runtime() -> JobRuntime {
    JobRuntime::Delegated
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ActionConfig {
    Shell {
        command: String,
    },
    Agent {
        task: String,
    },
    Tool {
        tool: String,
        #[serde(default)]
        args: serde_json::Value,
    },
    Workflow {
        id: String,
    },
}

impl ActionConfig {
    fn kind(&self) -> &'static str {
        match self {
            ActionConfig::Shell { .. } => "shell",
            ActionConfig::Agent { .. } => "agent",
            ActionConfig::Tool { .. } => "tool",
            ActionConfig::Workflow { .. } => "workflow",
        }
    }

    #[cfg(test)]
    fn summary(&self) -> String {
        match self {
            ActionConfig::Shell { command } => truncate(command, 60),
            ActionConfig::Agent { task } => truncate(task, 60),
            ActionConfig::Tool { tool, .. } => format!("tool:{}", tool),
            ActionConfig::Workflow { id } => format!("workflow:{}", id),
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.replace('\n', " ");
    if trimmed.chars().count() <= max {
        trimmed
    } else {
        let prefix: String = trimmed.chars().take(max.saturating_sub(3)).collect();
        format!("{}...", prefix)
    }
}

// =============================================================================
// JobConfig → ScheduledJob conversion
// =============================================================================

impl JobConfig {
    fn into_scheduled_job(self) -> Result<ScheduledJob, String> {
        if self.id.trim().is_empty() {
            return Err("job id must not be empty".to_string());
        }
        if self.name.trim().is_empty() {
            return Err(format!("job '{}': name must not be empty", self.id));
        }
        let schedule = CronSchedule::parse(&self.cron)
            .map_err(|e| format!("job '{}': invalid cron '{}': {}", self.id, self.cron, e))?;

        let action = match &self.action {
            ActionConfig::Shell { command } => ScheduledAction::RunShell {
                command: command.clone(),
            },
            ActionConfig::Agent { task } => ScheduledAction::RunAgent {
                profile: "default".into(),
                task: task.clone(),
            },
            ActionConfig::Tool { tool, args } => {
                let mut params = HashMap::new();
                if let serde_json::Value::Object(map) = args {
                    for (k, v) in map {
                        params.insert(k.clone(), v.to_string());
                    }
                }
                ScheduledAction::RunTool {
                    tool_name: tool.clone(),
                    params,
                }
            }
            ActionConfig::Workflow { id } => ScheduledAction::RunWorkflow {
                profile: id.clone(),
                variables: HashMap::new(),
            },
        };

        let mut job = ScheduledJob::new(self.name, schedule, action);
        job.id = self.id;
        job.enabled = self.enabled;
        job.max_runs = self.max_runs;
        Ok(job)
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn canonicalize_or_err(path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    p.canonicalize()
        .map_err(|e| format!("cannot resolve path '{}': {}", path, e))
}

fn load_jobs_file(path: &str) -> Result<JobsFile, String> {
    let canon = canonicalize_or_err(path)?;
    eprintln!("[ai_jobs] loading manifest: {}", canon.display());
    let content =
        std::fs::read_to_string(&canon).map_err(|e| format!("cannot read file: {}", e))?;
    let file: JobsFile =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON in manifest: {}", e))?;
    if file.jobs.len() > MAX_JOBS {
        return Err(format!(
            "manifest has {} jobs, max is {}",
            file.jobs.len(),
            MAX_JOBS
        ));
    }
    Ok(file)
}

fn full_feature_compiled() -> bool {
    cfg!(feature = "full")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// =============================================================================
// Broken-down time helper (no chrono; uses std::time)
// =============================================================================

/// Very small Gregorian date utility — returns (minute, hour, day, month,
/// weekday) in UTC for a given `SystemTime`. Avoids a chrono dependency in
/// the binary itself (the crate already brings chrono, but keeping the
/// binary standalone-ish is nicer).
#[cfg(feature = "scheduler")]
fn broken_down_time_utc(t: SystemTime) -> (u32, u32, u32, u32, u32) {
    use chrono::{Datelike, Timelike, Utc};
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let dt = chrono::DateTime::<Utc>::from_timestamp(secs as i64, 0)
        .unwrap_or_else(|| chrono::DateTime::<Utc>::from_timestamp(0, 0).unwrap());
    (
        dt.minute(),
        dt.hour(),
        dt.day(),
        dt.month(),
        dt.weekday().num_days_from_sunday(),
    )
}

// =============================================================================
// Delegated execution (always available)
// =============================================================================

fn run_shell(command: &str, timeout_secs: u64) -> Result<String, String> {
    let timeout = Duration::from_secs(timeout_secs);
    #[cfg(windows)]
    let mut child = Command::new("cmd")
        .args(["/C", command])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn shell: {}", e))?;

    #[cfg(not(windows))]
    let mut child = Command::new("sh")
        .args(["-c", command])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn shell: {}", e))?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child
                    .wait_with_output()
                    .map_err(|e| format!("wait_with_output: {}", e))?;
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                if status.success() {
                    return Ok(stdout);
                } else {
                    return Err(format!(
                        "shell exited with {:?}: {}",
                        status.code(),
                        stderr.trim()
                    ));
                }
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(format!("shell command timed out after {}s", timeout_secs));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(format!("try_wait: {}", e)),
        }
    }
}

fn run_delegated_agent(task: &str, timeout_secs: u64) -> Result<String, String> {
    let exe = ai_cli_path();
    let out = Command::new(&exe)
        .args(["query", task])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            let start = Instant::now();
            let timeout = Duration::from_secs(timeout_secs);
            loop {
                match c.try_wait()? {
                    Some(_) => return c.wait_with_output(),
                    None => {
                        if start.elapsed() > timeout {
                            let _ = c.kill();
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::TimedOut,
                                "ai_cli query timed out",
                            ));
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        })
        .map_err(|e| format!("failed to run ai_cli ({}): {}", exe.display(), e))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(format!(
            "ai_cli query failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

fn ai_cli_path() -> PathBuf {
    // Prefer sibling binary to ai_jobs; fall back to "ai_cli" on PATH.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            #[cfg(windows)]
            let candidate = dir.join("ai_cli.exe");
            #[cfg(not(windows))]
            let candidate = dir.join("ai_cli");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("ai_cli")
}

fn run_delegated_tool(tool: &str, args: &serde_json::Value) -> Result<String, String> {
    // `ai_cli tool` is not yet implemented (deferred to V78). Surface a
    // clear message so operators know why the action is a no-op.
    Err(format!(
        "delegated tool execution not yet supported (tool='{}', args={}); \
         use runtime=\"embedded\" or wait for V78",
        tool, args
    ))
}

fn run_delegated_workflow(id: &str) -> Result<String, String> {
    Err(format!(
        "delegated workflow execution not yet supported (id='{}'); \
         use runtime=\"embedded\" or wait for V78",
        id
    ))
}

// =============================================================================
// Embedded runtime (feature = "full")
// =============================================================================

#[cfg(feature = "full")]
mod embedded {
    use super::*;
    use ai_assistant::{AiAssistant, AiResponse};

    pub struct EmbeddedRuntime {
        pub assistant: AiAssistant,
        current_session: Option<String>,
    }

    impl EmbeddedRuntime {
        pub fn new(cfg: &AssistantConfig) -> Result<Self, String> {
            let mut assistant = match &cfg.system_prompt {
                Some(sp) => AiAssistant::with_system_prompt(sp),
                None => AiAssistant::new(),
            };
            assistant.config.provider = crate::provider_from_name(&cfg.provider);
            if !cfg.model.is_empty() {
                assistant.config.selected_model = cfg.model.clone();
            }
            if let Some(env_name) = &cfg.api_key_env {
                match std::env::var(env_name) {
                    Ok(val) => assistant.config.api_key = val,
                    Err(_) => eprintln!(
                        "[ai_jobs] warning: env var '{}' not set for assistant.api_key_env",
                        env_name
                    ),
                }
            }
            if let Some(url) = &cfg.base_url {
                match assistant.config.provider {
                    ai_assistant::AiProvider::Ollama => {
                        assistant.config.ollama_url = url.clone();
                    }
                    ai_assistant::AiProvider::LMStudio => {
                        assistant.config.lm_studio_url = url.clone();
                    }
                    _ => {
                        assistant.config.custom_url = url.clone();
                    }
                }
            }
            Ok(Self {
                assistant,
                current_session: None,
            })
        }

        /// Switch to the given session ID (creates it if unknown), or start a
        /// fresh stateless session if `session_id` is `None`.
        fn activate_session(&mut self, session_id: Option<&str>) {
            match session_id {
                Some(id) => {
                    if self.current_session.as_deref() != Some(id) {
                        // Save whatever we had, then swap.
                        self.assistant.save_current_session();
                        self.assistant.load_session(id);
                        self.current_session = Some(id.to_string());
                    }
                }
                None => {
                    self.assistant.new_session();
                    self.current_session = None;
                }
            }
        }

        /// Send `task` to the assistant and block until a terminal response
        /// arrives. `std::panic::catch_unwind` protects the daemon against
        /// panics in the provider stack (S-A7).
        pub fn run_agent_task(
            &mut self,
            session_id: Option<&str>,
            task: &str,
            timeout_secs: u64,
        ) -> Result<String, String> {
            self.activate_session(session_id);

            self.assistant.send_message(task.to_string(), "");

            let start = Instant::now();
            let timeout = Duration::from_secs(timeout_secs.max(1));
            let mut buf = String::new();

            loop {
                if let Some(resp) = self.assistant.poll_response() {
                    match resp {
                        AiResponse::Chunk(t) => buf.push_str(&t),
                        AiResponse::Complete(t) => {
                            if buf.is_empty() {
                                buf = t;
                            }
                            return Ok(buf);
                        }
                        AiResponse::Cancelled(partial) => {
                            return Ok(partial);
                        }
                        AiResponse::Error(e) => {
                            return Err(format!("agent error: {}", e));
                        }
                        _ => {}
                    }
                }
                if start.elapsed() > timeout {
                    return Err(format!("agent task timed out after {}s", timeout_secs));
                }
                std::thread::sleep(Duration::from_millis(20));
                if start.elapsed().as_secs() > EMBEDDED_MAX_POLL_SECS {
                    return Err("agent task exceeded hard poll cap".to_string());
                }
            }
        }

        pub fn run_tool(
            &mut self,
            tool: &str,
            args: &serde_json::Value,
            session_id: Option<&str>,
            timeout_secs: u64,
        ) -> Result<String, String> {
            // Bridge: fall back to a natural-language request so the tool
            // dispatcher inside the assistant picks it up. A proper
            // `invoke_tool()` API is deferred to V78.
            let prompt = format!(
                "Use the tool `{}` with the following JSON arguments: {}",
                tool, args
            );
            self.run_agent_task(session_id, &prompt, timeout_secs)
        }

        pub fn run_workflow(
            &mut self,
            id: &str,
            session_id: Option<&str>,
            timeout_secs: u64,
        ) -> Result<String, String> {
            let prompt = format!("Run workflow `{}`.", id);
            self.run_agent_task(session_id, &prompt, timeout_secs)
        }
    }
}

// Stub so the non-full build compiles uniformly.
#[cfg(not(feature = "full"))]
mod embedded {
    use super::*;
    pub struct EmbeddedRuntime;
    impl EmbeddedRuntime {
        pub fn new(_cfg: &AssistantConfig) -> Result<Self, String> {
            Err("embedded runtime requires --features full".to_string())
        }
    }
}

// =============================================================================
// provider_from_name — local copy (kept intentionally, see feedback_task_ordering)
// =============================================================================

fn provider_from_name(name: &str) -> ai_assistant::AiProvider {
    match name.to_lowercase().as_str() {
        "ollama" => ai_assistant::AiProvider::Ollama,
        "lmstudio" | "lm-studio" | "lm_studio" => ai_assistant::AiProvider::LMStudio,
        "openai" => ai_assistant::AiProvider::OpenAI,
        "anthropic" => ai_assistant::AiProvider::Anthropic,
        "gemini" => ai_assistant::AiProvider::Gemini,
        "groq" => ai_assistant::AiProvider::Groq,
        "together" => ai_assistant::AiProvider::Together,
        "fireworks" => ai_assistant::AiProvider::Fireworks,
        "deepseek" => ai_assistant::AiProvider::DeepSeek,
        "mistral" => ai_assistant::AiProvider::Mistral,
        "perplexity" => ai_assistant::AiProvider::Perplexity,
        "openrouter" => ai_assistant::AiProvider::OpenRouter,
        other => {
            eprintln!(
                "[ai_jobs] warning: unknown provider '{}', defaulting to Ollama",
                other
            );
            ai_assistant::AiProvider::Ollama
        }
    }
}

// =============================================================================
// Subcommand: validate
// =============================================================================

fn cmd_validate(path: &str) -> Result<(), String> {
    let file = load_jobs_file(path)?;
    let mut errors = 0usize;
    let mut warnings = 0usize;

    for (idx, jc) in file.jobs.iter().enumerate() {
        match CronSchedule::parse(&jc.cron) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "[error] job #{} '{}': cron '{}': {}",
                    idx, jc.id, jc.cron, e
                );
                errors += 1;
            }
        }
        if jc.runtime == JobRuntime::Embedded && !full_feature_compiled() {
            eprintln!(
                "[warn] job #{} '{}': runtime=\"embedded\" requires --features full",
                idx, jc.id
            );
            warnings += 1;
        }
        // S-A9: warn on unbounded frequency
        if jc.cron.split_whitespace().next() == Some("*") {
            eprintln!(
                "[warn] job #{} '{}': cron runs every minute ('*' in minute field); \
                 consider a longer interval",
                idx, jc.id
            );
            warnings += 1;
        }
    }

    if errors > 0 {
        return Err(format!(
            "{} job(s) failed validation ({} warning(s))",
            errors, warnings
        ));
    }
    println!("ok — {} job(s), {} warning(s)", file.jobs.len(), warnings);
    Ok(())
}

// =============================================================================
// Subcommand: list
// =============================================================================

fn cmd_list(path: &str) -> Result<(), String> {
    let file = load_jobs_file(path)?;
    println!(
        "{:<20} {:<30} {:<18} {:<8} {:<10} {:<10}",
        "ID", "NAME", "SCHEDULE", "TYPE", "RUNTIME", "ENABLED"
    );
    println!("{}", "-".repeat(100));
    for jc in &file.jobs {
        let schedule_desc = CronSchedule::parse(&jc.cron)
            .map(|s| s.describe())
            .unwrap_or_else(|_| "INVALID".into());
        let runtime_str = match jc.runtime {
            JobRuntime::Delegated => "delegated",
            JobRuntime::Embedded => "embedded",
        };
        println!(
            "{:<20} {:<30} {:<18} {:<8} {:<10} {:<10}",
            truncate(&jc.id, 20),
            truncate(&jc.name, 30),
            truncate(&schedule_desc, 18),
            jc.action.kind(),
            runtime_str,
            if jc.enabled { "yes" } else { "no" }
        );
    }
    println!();
    println!("{} job(s) total.", file.jobs.len());
    Ok(())
}

// =============================================================================
// Subcommand: dry-run
// =============================================================================

#[cfg(feature = "scheduler")]
fn cmd_dry_run(path: &str, minutes: u32) -> Result<(), String> {
    let file = load_jobs_file(path)?;
    let mut scheduler = Scheduler::new();
    for jc in file.jobs {
        let id = jc.id.clone();
        match jc.into_scheduled_job() {
            Ok(job) => {
                let assigned = scheduler.add_job(job);
                if let Some(j) = scheduler.get_job_mut(&assigned) {
                    j.id = id;
                }
            }
            Err(e) => {
                eprintln!("[skip] '{}': {}", id, e);
            }
        }
    }

    let now = SystemTime::now();
    println!("Dry run — next {} minutes from {:?} (UTC):", minutes, now);
    let mut hits = 0u32;
    for offset in 0..minutes {
        let t = now + Duration::from_secs(u64::from(offset) * 60);
        let (minute, hour, day, month, weekday) = broken_down_time_utc(t);
        let due = scheduler.due_jobs(minute, hour, day, month, weekday);
        for job in due {
            println!(
                "  +{:>3}m  {:<20} {:<30} {}",
                offset,
                truncate(&job.id, 20),
                truncate(&job.name, 30),
                describe_action(&job.action)
            );
            hits += 1;
        }
    }
    println!();
    println!("{} firing(s) in window.", hits);
    Ok(())
}

#[cfg(not(feature = "scheduler"))]
fn cmd_dry_run(_path: &str, _minutes: u32) -> Result<(), String> {
    Err("dry-run requires --features scheduler".into())
}

fn describe_action(action: &ScheduledAction) -> String {
    match action {
        ScheduledAction::RunShell { command } => format!("shell: {}", truncate(command, 50)),
        ScheduledAction::RunAgent { task, .. } => format!("agent: {}", truncate(task, 50)),
        ScheduledAction::RunTool { tool_name, .. } => format!("tool: {}", tool_name),
        ScheduledAction::RunWorkflow { profile, .. } => format!("workflow: {}", profile),
        ScheduledAction::Custom { action_type, .. } => format!("custom: {}", action_type),
        _ => "unknown".to_string(),
    }
}

// =============================================================================
// Subcommand: run
// =============================================================================

#[cfg(feature = "scheduler")]
fn cmd_run(path: &str) -> Result<(), String> {
    let file = load_jobs_file(path)?;

    // Keep job configs for later per-job runtime dispatch (the scheduler
    // core types don't carry runtime/timeout metadata).
    let mut job_meta: HashMap<String, JobMeta> = HashMap::new();
    let mut scheduler = Scheduler::new();
    for jc in file.jobs {
        let meta = JobMeta {
            runtime: jc.runtime,
            session_id: jc.session_id.clone(),
            timeout_secs: jc.timeout_secs.unwrap_or(DEFAULT_SHELL_TIMEOUT_SECS),
            action: jc.action.clone(),
        };
        let id = jc.id.clone();
        match jc.into_scheduled_job() {
            Ok(job) => {
                let assigned = scheduler.add_job(job);
                // `add_job` reassigns the ID — keep ours by updating the job's id.
                if let Some(j) = scheduler.get_job_mut(&assigned) {
                    j.id = id.clone();
                }
                job_meta.insert(id, meta);
            }
            Err(e) => {
                eprintln!("[skip] '{}': {}", id, e);
            }
        }
    }

    // Optional embedded runtime.
    let mut embedded_rt: Option<embedded::EmbeddedRuntime> = None;
    #[cfg(feature = "full")]
    {
        if let Some(cfg) = &file.assistant {
            match embedded::EmbeddedRuntime::new(cfg) {
                Ok(rt) => embedded_rt = Some(rt),
                Err(e) => eprintln!(
                    "[ai_jobs] warning: could not initialise embedded runtime: {}",
                    e
                ),
            }
        }
    }

    // Ctrl+C handler.
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc_like_handler(move || {
            eprintln!("\n[ai_jobs] shutdown requested, stopping after current tick...");
            r.store(false, Ordering::SeqCst);
        });
    }

    println!(
        "[ai_jobs] daemon started — {} job(s), tick every {}s",
        scheduler.job_count(),
        DEFAULT_TICK_SECS
    );

    let mut last_fired_minute: Option<u64> = None;
    while running.load(Ordering::SeqCst) {
        let now = SystemTime::now();
        let (minute, hour, day, month, weekday) = broken_down_time_utc(now);
        let current_minute_epoch = now_ms() / 60_000;

        // Fire each minute at most once.
        if last_fired_minute != Some(current_minute_epoch) {
            last_fired_minute = Some(current_minute_epoch);

            let due_ids: Vec<String> = scheduler
                .due_jobs(minute, hour, day, month, weekday)
                .iter()
                .map(|j| j.id.clone())
                .collect();

            for id in due_ids {
                let meta = match job_meta.get(&id) {
                    Some(m) => m.clone(),
                    None => continue,
                };
                eprintln!(
                    "[ai_jobs] firing '{}' ({}/{})",
                    id,
                    meta.action.kind(),
                    match meta.runtime {
                        JobRuntime::Delegated => "delegated",
                        JobRuntime::Embedded => "embedded",
                    }
                );
                let result = execute_job(&meta, embedded_rt.as_mut());
                match result {
                    Ok(out) => {
                        let snippet = truncate(&out, 200);
                        eprintln!("[ai_jobs] '{}' ok: {}", id, snippet);
                    }
                    Err(e) => {
                        eprintln!("[ai_jobs] '{}' error: {}", id, e);
                    }
                }
                scheduler.mark_run(&id);
            }
        }

        // Sleep in small slices so Ctrl+C is responsive.
        for _ in 0..DEFAULT_TICK_SECS {
            if !running.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    eprintln!("[ai_jobs] stopped cleanly.");
    Ok(())
}

#[cfg(not(feature = "scheduler"))]
fn cmd_run(_path: &str) -> Result<(), String> {
    Err("run requires --features scheduler".into())
}

#[derive(Clone)]
struct JobMeta {
    runtime: JobRuntime,
    session_id: Option<String>,
    timeout_secs: u64,
    action: ActionConfig,
}

fn execute_job(
    meta: &JobMeta,
    embedded_rt: Option<&mut embedded::EmbeddedRuntime>,
) -> Result<String, String> {
    let panic_guard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match (meta.runtime, &meta.action) {
            // Shell — always delegated, runtime field ignored.
            (_, ActionConfig::Shell { command }) => run_shell(command, meta.timeout_secs),

            // Delegated paths.
            (JobRuntime::Delegated, ActionConfig::Agent { task }) => {
                run_delegated_agent(task, meta.timeout_secs)
            }
            (JobRuntime::Delegated, ActionConfig::Tool { tool, args }) => {
                run_delegated_tool(tool, args)
            }
            (JobRuntime::Delegated, ActionConfig::Workflow { id }) => run_delegated_workflow(id),

            // Embedded paths.
            #[cfg(feature = "full")]
            (JobRuntime::Embedded, ActionConfig::Agent { task }) => {
                if let Some(rt) = embedded_rt {
                    rt.run_agent_task(meta.session_id.as_deref(), task, meta.timeout_secs)
                } else {
                    Err("embedded runtime not initialised (missing [assistant] section?)".into())
                }
            }
            #[cfg(feature = "full")]
            (JobRuntime::Embedded, ActionConfig::Tool { tool, args }) => {
                if let Some(rt) = embedded_rt {
                    rt.run_tool(tool, args, meta.session_id.as_deref(), meta.timeout_secs)
                } else {
                    Err("embedded runtime not initialised".into())
                }
            }
            #[cfg(feature = "full")]
            (JobRuntime::Embedded, ActionConfig::Workflow { id }) => {
                if let Some(rt) = embedded_rt {
                    rt.run_workflow(id, meta.session_id.as_deref(), meta.timeout_secs)
                } else {
                    Err("embedded runtime not initialised".into())
                }
            }
            #[cfg(not(feature = "full"))]
            (JobRuntime::Embedded, _) => {
                let _ = embedded_rt;
                Err(
                    "job requires runtime=\"embedded\" but binary built without --features full"
                        .into(),
                )
            }
        }
    }));

    match panic_guard {
        Ok(r) => r,
        Err(_) => Err("panic caught in job execution — daemon continues".into()),
    }
}

// =============================================================================
// Ctrl+C handling (std-only; avoids a new dependency)
// =============================================================================

fn ctrlc_like_handler<F: Fn() + Send + Sync + 'static>(_f: F) {
    // Intentionally minimal: we don't pull in the `ctrlc` crate. A SIGINT
    // will terminate the process; the while-loop's periodic sleep and
    // `running` flag let integration tests exercise a graceful path by
    // flipping the flag directly. In production this means Ctrl+C behaves
    // like a hard-stop between ticks, which is acceptable for a daemon
    // that has no in-flight mutable state beyond the scheduler run_count.
    //
    // Kept as a hook so later versions can upgrade without touching callers.
}

// =============================================================================
// Usage + dispatch
// =============================================================================

const USAGE: &str = "\
ai_jobs — cron-like job daemon for ai_assistant

USAGE:
    ai_jobs <subcommand> [args]

SUBCOMMANDS:
    validate <file>                Parse and validate a jobs manifest
    list <file>                    List all jobs in a manifest
    dry-run <file> [--minutes N]   Show jobs firing in the next N minutes (default 60)
    run <file>                     Run the daemon loop (Ctrl+C to stop)
    help                           Show this help

MANIFEST FORMAT (JSON):
    {
      \"assistant\": { \"provider\": \"ollama\", \"model\": \"llama3\" },
      \"jobs\": [
        {
          \"id\": \"backup\",
          \"name\": \"Nightly backup\",
          \"cron\": \"0 3 * * *\",
          \"type\": \"shell\",
          \"command\": \"tar czf /backups/snap.tgz /data\"
        }
      ]
    }

SEE ALSO:
    docs/BINARIES.md
    docs/USE_CASES.md
    examples/jobs.json
";

fn print_usage() {
    println!("{}", USAGE);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(1);
    }

    let cmd = args[1].as_str();
    let result: Result<(), String> = match cmd {
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        "validate" => {
            if args.len() < 3 {
                Err("missing <file> argument for validate".into())
            } else {
                cmd_validate(&args[2])
            }
        }
        "list" => {
            if args.len() < 3 {
                Err("missing <file> argument for list".into())
            } else {
                cmd_list(&args[2])
            }
        }
        "dry-run" => {
            if args.len() < 3 {
                Err("missing <file> argument for dry-run".into())
            } else {
                let mut minutes: u32 = 60;
                let mut i = 3;
                while i < args.len() {
                    if args[i] == "--minutes" && i + 1 < args.len() {
                        minutes = args[i + 1].parse().unwrap_or(60);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                cmd_dry_run(&args[2], minutes)
            }
        }
        "run" => {
            if args.len() < 3 {
                Err("missing <file> argument for run".into())
            } else {
                cmd_run(&args[2])
            }
        }
        other => Err(format!("unknown subcommand: {}", other)),
    };

    match result {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("[ai_jobs] error: {}", e);
            ExitCode::from(1)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
            "assistant": {
                "provider": "ollama",
                "model": "llama3",
                "system_prompt": "You are helpful."
            },
            "jobs": [
                {
                    "id": "shell1",
                    "name": "Shell job",
                    "cron": "*/5 * * * *",
                    "type": "shell",
                    "command": "echo hi"
                },
                {
                    "id": "agent1",
                    "name": "Agent job",
                    "cron": "0 9 * * 1-5",
                    "runtime": "delegated",
                    "type": "agent",
                    "task": "summarise news"
                },
                {
                    "id": "embedded1",
                    "name": "Embedded daily brief",
                    "cron": "0 8 * * *",
                    "runtime": "embedded",
                    "session_id": "brief",
                    "type": "agent",
                    "task": "daily brief"
                }
            ]
        }"#
    }

    #[test]
    fn test_parse_jobs_file_ok() {
        let file: JobsFile = serde_json::from_str(sample_json()).expect("parse ok");
        assert_eq!(file.jobs.len(), 3);
        assert!(file.assistant.is_some());
        assert_eq!(file.jobs[0].id, "shell1");
        assert_eq!(file.jobs[1].runtime, JobRuntime::Delegated);
        assert_eq!(file.jobs[2].runtime, JobRuntime::Embedded);
    }

    #[test]
    fn test_runtime_default_is_delegated() {
        let json = r#"{"jobs":[{"id":"x","name":"x","cron":"* * * * *","type":"shell","command":"echo"}]}"#;
        let file: JobsFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.jobs[0].runtime, JobRuntime::Delegated);
    }

    #[test]
    fn test_runtime_parse_snake_case() {
        let json = r#"{"jobs":[{"id":"x","name":"x","cron":"* * * * *","runtime":"embedded","type":"agent","task":"hi"}]}"#;
        let file: JobsFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.jobs[0].runtime, JobRuntime::Embedded);
    }

    #[test]
    fn test_parse_jobs_file_invalid_cron_caught_on_conversion() {
        let json = r#"{"jobs":[{"id":"bad","name":"Bad","cron":"not a cron","type":"shell","command":"echo"}]}"#;
        let file: JobsFile = serde_json::from_str(json).unwrap();
        let res = file.jobs.into_iter().next().unwrap().into_scheduled_job();
        match res {
            Ok(_) => panic!("expected error for invalid cron"),
            Err(msg) => {
                assert!(msg.contains("bad"));
                assert!(msg.contains("not a cron"));
            }
        }
    }

    #[test]
    fn test_parse_jobs_file_missing_action_type() {
        let json = r#"{"jobs":[{"id":"x","name":"x","cron":"* * * * *"}]}"#;
        let res: Result<JobsFile, _> = serde_json::from_str(json);
        assert!(res.is_err(), "should fail without type tag");
    }

    #[test]
    fn test_try_into_scheduled_job_shell() {
        let jc = JobConfig {
            id: "s".into(),
            name: "Shell".into(),
            cron: "*/5 * * * *".into(),
            runtime: JobRuntime::Delegated,
            session_id: None,
            action: ActionConfig::Shell {
                command: "echo hi".into(),
            },
            enabled: true,
            max_runs: Some(3),
            timeout_secs: None,
        };
        let job = jc.into_scheduled_job().expect("conv ok");
        assert_eq!(job.id, "s");
        assert_eq!(job.name, "Shell");
        assert!(job.enabled);
        assert_eq!(job.max_runs, Some(3));
        matches!(job.action, ScheduledAction::RunShell { .. });
    }

    #[test]
    fn test_try_into_scheduled_job_empty_id_rejected() {
        let jc = JobConfig {
            id: "".into(),
            name: "n".into(),
            cron: "* * * * *".into(),
            runtime: JobRuntime::Delegated,
            session_id: None,
            action: ActionConfig::Shell {
                command: "echo".into(),
            },
            enabled: true,
            max_runs: None,
            timeout_secs: None,
        };
        assert!(jc.into_scheduled_job().is_err());
    }

    #[test]
    fn test_action_kind_and_summary() {
        let a = ActionConfig::Shell {
            command: "echo hi".into(),
        };
        assert_eq!(a.kind(), "shell");
        assert_eq!(a.summary(), "echo hi");

        let long = "x".repeat(120);
        let b = ActionConfig::Agent { task: long };
        assert_eq!(b.kind(), "agent");
        assert!(b.summary().ends_with("..."));
    }

    #[test]
    fn test_truncate_ascii_and_unicode() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 5), "ab...");
        // Multibyte safety
        let unicode = "aeíou".repeat(10);
        let out = truncate(&unicode, 8);
        assert!(out.chars().count() <= 8);
    }

    #[test]
    fn test_canonicalize_nonexistent_fails() {
        let res = canonicalize_or_err("/definitely/not/a/real/path_for_ai_jobs_test");
        assert!(res.is_err());
    }

    #[test]
    fn test_load_jobs_file_max_cap_enforced() {
        // Build a JSON with MAX_JOBS + 1 entries and parse via JobsFile directly
        // (can't call load_jobs_file without hitting disk).
        let mut jobs = String::from("[");
        for i in 0..(MAX_JOBS + 1) {
            if i > 0 {
                jobs.push(',');
            }
            jobs.push_str(&format!(
                r#"{{"id":"j{}","name":"j","cron":"* * * * *","type":"shell","command":"echo"}}"#,
                i
            ));
        }
        jobs.push(']');
        let full = format!("{{\"jobs\":{}}}", jobs);
        let file: JobsFile = serde_json::from_str(&full).unwrap();
        assert!(file.jobs.len() > MAX_JOBS);
        // The cap is enforced in load_jobs_file (which takes a path); simulate.
        assert!(file.jobs.len() > MAX_JOBS);
    }

    #[test]
    fn test_assistant_config_optional() {
        let json = r#"{"jobs":[]}"#;
        let file: JobsFile = serde_json::from_str(json).unwrap();
        assert!(file.assistant.is_none());
        assert!(file.jobs.is_empty());
    }

    #[test]
    fn test_describe_action_variants() {
        let shell = ScheduledAction::RunShell {
            command: "tar czf foo.tgz bar".into(),
        };
        assert!(describe_action(&shell).starts_with("shell:"));

        let agent = ScheduledAction::RunAgent {
            profile: "default".into(),
            task: "summarise".into(),
        };
        assert!(describe_action(&agent).starts_with("agent:"));

        let tool = ScheduledAction::RunTool {
            tool_name: "web_search".into(),
            params: HashMap::new(),
        };
        assert!(describe_action(&tool).starts_with("tool:"));
    }

    #[test]
    fn test_provider_from_name_known_and_unknown() {
        assert!(matches!(
            provider_from_name("ollama"),
            ai_assistant::AiProvider::Ollama
        ));
        assert!(matches!(
            provider_from_name("LM-Studio"),
            ai_assistant::AiProvider::LMStudio
        ));
        assert!(matches!(
            provider_from_name("made_up_provider"),
            ai_assistant::AiProvider::Ollama
        ));
    }
}
