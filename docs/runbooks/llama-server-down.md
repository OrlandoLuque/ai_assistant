# Runbook: `llama-server` is down

**Severity**: P1 if traffic-serving; P2 if dev-only.
**Owner**: Inference platform.
**Last reviewed**: 2026-05-06 (V130).

`llama-server` is the upstream `llama.cpp` HTTP daemon. The crate
talks to it over the OpenAI-compatible API at the URL configured
under `[llama_cpp]` (default `http://127.0.0.1:8080`). When it
goes away, every Ollama-bypass / direct-llama path returns errors.

## 1. Symptoms

* `ai_cli` — generation hangs, then errors with `inference backend
  unreachable`.
* `ai_local_infer status` — `llama-server probe: failed (...)`.
* HTTP 502 / 503 on the embedded server.
* OpenTelemetry: spike in `inference_request_failures_total{backend="llama_cpp"}`.
* Metric `llama_cpp_health_probe_success` flat-zero for ≥1 min.

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | Out-of-memory / OOM-kill (model didn't fit in VRAM/RAM) | high |
| 2 | Process crashed on a malformed prompt | medium |
| 3 | Model file moved or corrupted | medium |
| 4 | GPU driver hung (`nvidia-smi` returns "No devices") | medium |
| 5 | Port already bound (another process took 8080) | low |
| 6 | Upgraded `llama.cpp` is incompatible with the GGUF | low |

## 3. Diagnose

Run, in order, until one identifies the cause:

```bash
# Is the process even alive?
ps -ef | grep -E '[l]lama-server'                # Linux
Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue   # Windows PS

# What does the health probe say?
curl -sS http://127.0.0.1:8080/health | head -c 200
ai_local_infer status                            # crate-side probe (V108)

# OOM in dmesg / Event Viewer?
dmesg --since '15 min ago' | grep -i 'oom\|killed'   # Linux
Get-WinEvent -FilterHashtable @{LogName='System'; Level=2; StartTime=(Get-Date).AddMinutes(-15)}

# Last log lines from llama-server (path is per-deployment).
tail -n 200 /var/log/llama-server.log
ai_logs --since=15m --component=llama_cpp        # if logs are wired into ai_logs

# GPU still alive?
nvidia-smi
```

## 4. Mitigate

Pick the first that applies:

**A. OOM:**
- Restart with smaller `n_ctx` and/or fewer GPU layers.
  ```bash
  ai_local_infer start --model models/qwen-2.5-7b.gguf \
    --n-ctx 2048 --n-gpu-layers 28
  ```
- The crate's `local_inference` module has a VRAM auto-clamp (V108);
  call `ai_local_infer auto-clamp --model <path>` to recompute.

**B. Process crashed:**
- Restart it. Do not loop-restart more than 3× in 5 minutes — if it
  keeps dying, escalate to "Resolve" rather than masking the bug.

**C. Model file corrupted:**
- Verify against the SHA-256 sidecar from your model store.
  ```bash
  sha256sum models/qwen-2.5-7b.gguf
  ```
- Re-download or restore from `secure_backup` (`ai_backup restore`)
  if the file is part of a backup set.

**D. GPU driver hung:**
- `sudo nvidia-smi -r` (Linux) or reboot the box. **Never** do this
  on a shared host without checking other tenants.

**E. Port collision:**
- `ss -tlnp | grep 8080` (Linux) or `netstat -ano | findstr :8080`
  (Windows). Kill the squatter or move llama-server with `--port`.

**F. Version drift:**
- Roll back to the last-known-good `llama.cpp` binary. Document the
  GGUF + binary pair in your release notes.

While you mitigate, traffic should fall back to the configured
`fallback` provider (Ollama, cloud) automatically — verify by
watching `inference_request_total{backend="..."}`.

## 5. Resolve

* If the crash repeats: file a bug with `ai_logs --since=1h
  --component=llama_cpp --json`, the prompt that triggered it (with
  PII redacted), and the GGUF SHA-256.
* If OOM: lower the configured budget in
  `internal_storage/<env>.toml` and add an alert at
  `vram_used_bytes / vram_total_bytes > 0.9`.
* If GPU hang: open a ticket with the GPU-driver vendor with the
  `nvidia-bug-report.sh` (or AMD equivalent) attached.
* If port collision: assign llama-server a non-default port in your
  systemd unit / Windows service definition.

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Detection | how the alert fired (probe, user report, etc.) |
| Time-to-detect | first symptom → first responder |
| Time-to-mitigate | symptom → traffic restored |
| Root cause | one sentence |
| Customer impact | requests dropped, error rate %, region |
| Action items | each with owner + due date |

Update this runbook if a new failure mode was found. Bump
*Last reviewed* date.
