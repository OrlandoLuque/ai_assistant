# Runbook: Scheduler missed a job

**Severity**: P2 — depends on what the job does.
**Owner**: Autonomous platform.
**Last reviewed**: 2026-05-06 (V130).

The crate's `scheduler` feature (cron-style) drives autonomous
agents and recurring jobs. A "missed" job is one whose `due_at` is
in the past and whose `last_fired_at` is older than `due_at` — i.e.
the scheduler did not fire on time.

## 1. Symptoms

* `ai_jobs list --overdue` shows entries.
* Downstream consumer alert: "no new $thing in N minutes".
* `scheduler_jobs_missed_total` metric increments.
* Audit log gap: no `JobFired` events for the expected window.

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | Scheduler process not running (crashed / never started) | high |
| 2 | System clock skew (NTP failure) — job thinks it's not yet time | medium |
| 3 | Job blocked on a long-running predecessor (queue starvation) | medium |
| 4 | Disabled flag set inadvertently by a config push | medium |
| 5 | Time-zone mismatch between cron spec and runtime | medium |
| 6 | Scheduler holds the `.scheduled_tasks.lock` from a dead worker | low |

## 3. Diagnose

```bash
# Is the scheduler alive?
ai_jobs status

# Is the cron expression what you think it is?
ai_jobs show <job_id> | head -30

# When was it last successful?
ai_jobs history <job_id> --limit 20

# Clock and TZ
date -u
timedatectl status                              # Linux
w32tm /query /status                            # Windows

# Lock file (V126+: scheduler uses a flock-style guard).
ls -la .claude/scheduled_tasks.lock 2>/dev/null
ls -la <state_dir>/scheduled_tasks.lock

# Is another job blocking the queue?
ai_jobs list --running
```

## 4. Mitigate

**A. Scheduler not running:**
- Restart the service. Verify it picks up persisted jobs:
  `ai_jobs list --enabled` should match what was there before.
- Trigger the missed job by hand:
  `ai_jobs run --id <job_id> --force`

**B. Clock skew:**
- Re-sync NTP (`systemctl restart systemd-timesyncd` /
  `w32tm /resync`). Retrigger missed jobs after.

**C. Queue starvation:**
- Identify the long-runner: `ai_jobs list --running` shows duration.
- If it's stuck in inference, see [`llama-server-down`](llama-server-down.md).
- Cancel with `ai_jobs cancel --id <id>`, then re-queue.
- Add per-job timeouts so this can't recur silently.

**D. Stale lock file:**
- *Verify the previous worker is dead* (`ps`/`Get-Process`). If you
  delete the lock with the worker still alive you will corrupt
  state. Once verified dead, `rm <path>/scheduled_tasks.lock` and
  restart.

**E. Time-zone mismatch:**
- The crate's scheduler stores cron specs in UTC by default.
  Confirm with `ai_jobs show <id> --raw`. If the spec was set in
  local time, edit it: `ai_jobs edit --id <id> --cron "<UTC spec>"`.

## 5. Resolve

* Add a synthetic heartbeat job that fires every 5 minutes and
  alerts if absent for 10 — this catches whole-scheduler outage.
* Per-job timeout (`ai_jobs edit --id <id> --timeout 600s`) so a
  hung job cannot block the queue indefinitely.
* Audit-log retention long enough to see one full week of fires —
  default is 1000 events, which can be too few for low-frequency
  jobs.
* If the scheduler holds a lock and crashes hard often: switch to
  the OS-supervised mode (systemd, Windows Service) so a kill drops
  the lock automatically via process exit.

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Job id / name | |
| Missed window | first → last expected fire |
| Cause | from §2 |
| Downstream impact | what didn't happen on time |
| Recovery action | retrigger / wait / data fix |
| Action items | owner + due date |
