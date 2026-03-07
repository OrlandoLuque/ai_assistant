# v33 Improvements Changelog

> Date: 2026-03-07

v33 is the **Docker Integration Tests** release: 24 integration tests that validate the container execution engine against a real Docker daemon, covering lifecycle, exec, copy, bind mounts, network isolation, resource limits, sandbox execution, and document pipeline conversion.

---

## Summary Metrics

| Metric | v32 | v33 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | ~7,086 | ~7,086 | 0 |
| Integration tests (containers) | 0 | 24 | +24 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 1 | +1 |
| Lines added | — | ~580 | — |

---

## New Files

| File | LOC | Description |
|------|-----|-------------|
| `tests/docker_integration_tests.rs` | ~580 | 24 Docker integration tests across 6 categories |

---

## Test Infrastructure

### Skip Macros

- **`skip_if_no_docker!()`** — calls `ContainerExecutor::is_docker_available()`, returns early with diagnostic message if Docker is unavailable.
- **`skip_if_no_heavy_images!()`** — additionally checks `AI_TEST_HEAVY_IMAGES` env var for tests requiring large images (python:3.12-slim, pandoc/extra).

### RAII Cleanup Guard

`TestExecutor` struct owns a `ContainerExecutor` and tracks container IDs. On `Drop`, it force-stops and removes all tracked containers — even if the test panics.

### Unique Naming

`unique_name(prefix)` generates `dit_{prefix}_{nanos}` names to prevent collisions when tests run in parallel.

---

## Test Categories

### Executor Tests (10 tests)

| Test | Description |
|------|-------------|
| `test_docker_is_available` | Diagnostic: logs Docker availability |
| `test_docker_full_lifecycle` | create → start → exec → stop → remove with status verification |
| `test_docker_exec_stdout_stderr` | Verifies stdout/stderr stream separation |
| `test_docker_exec_nonzero_exit` | `exit 42` returns exit_code=42, `!success()` |
| `test_docker_exec_timeout` | `sleep 60` with 2s timeout → `timed_out=true`, `exit_code=-1` |
| `test_docker_copy_roundtrip` | Host → container (copy_to) → host (copy_from), byte comparison |
| `test_docker_env_vars` | `env_vars` in `CreateOptions`, verified via `printenv` |
| `test_docker_logs_capture` | Container produces output, `logs()` captures it |
| `test_docker_list_and_status` | 2 containers, 1 started — verify `list()` and `status()` |
| `test_docker_cleanup_all` | 2 running containers, `cleanup_all()` returns 2 |

### Sandbox Tests (4 tests)

| Test | Description |
|------|-------------|
| `test_sandbox_bash_execution` | `ContainerSandbox` + `Language::Bash`, run `echo` |
| `test_sandbox_warm_reuse` | `reuse_containers: true`, execute twice, both succeed |
| `test_sandbox_no_reuse` | `reuse_containers: false`, container cleaned up after execution |
| `test_execution_backend_auto` | `ExecutionBackend::auto()` selects container when Docker available |

### Shared Folder Tests (3 tests)

| Test | Description |
|------|-------------|
| `test_shared_folder_host_to_container_via_mount` | Host writes file, container reads via bind mount |
| `test_shared_folder_container_to_host` | Container writes file, host reads from SharedFolder |
| `test_shared_folder_subdirectory_mount` | Nested directory structure visible across mount boundary |

### Network & Resource Limit Tests (2 tests)

| Test | Description |
|------|-------------|
| `test_docker_network_none_blocks_access` | `NetworkMode::None` prevents `ping 8.8.8.8` |
| `test_docker_memory_limit_applied` | 128MB limit verified via cgroup files |

### Error Tests (2 tests)

| Test | Description |
|------|-------------|
| `test_docker_exec_on_stopped_container` | Exec on stopped container returns `Err` |
| `test_docker_remove_nonexistent` | Remove fake ID returns `Err` |

### Heavy Tests (3 tests, behind `AI_TEST_HEAVY_IMAGES=1`)

| Test | Description |
|------|-------------|
| `test_sandbox_python` | Python execution in python:3.12-slim |
| `test_document_pipeline_conversion` | Markdown → HTML via pandoc/extra |
| `test_docker_auto_pull_specific_tag` | Auto-pull `busybox:1.36.1` specific tag |

---

## How to Run

```bash
# All Docker integration tests (requires Docker running):
cargo test --features containers --test docker_integration_tests

# Single test:
cargo test --features containers --test docker_integration_tests test_docker_full_lifecycle

# Including heavy image tests:
AI_TEST_HEAVY_IMAGES=1 cargo test --features containers --test docker_integration_tests

# Sequential execution (if Docker contention):
cargo test --features containers --test docker_integration_tests -- --test-threads=1
```

---

## Key Design Decisions

1. **Separate integration test file** — `tests/docker_integration_tests.rs`, not mixed into unit tests. Compiled only with `--features containers`.
2. **RAII cleanup** — `TestExecutor` guard ensures containers are removed even on test panics.
3. **Graceful skip** — every test starts with `skip_if_no_docker!()`. No CI failure on Docker-less runners.
4. **Heavy image separation** — python/pandoc tests gated behind `AI_TEST_HEAVY_IMAGES` env var.
5. **Unique naming** — nanosecond timestamps in container names prevent parallel test collisions.
6. **Zero production code changes** — only adds tests, no modifications to existing modules.
