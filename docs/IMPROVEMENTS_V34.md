# v34 Improvements Changelog

> Date: 2026-03-07

v34 is the **Docker CLI + MCP Integration** release: wires the bollard-based `ContainerExecutor` to the standalone binary REPL (`/docker` commands) and exposes 8 MCP tools for remote Docker container management via `POST /mcp`.

---

## Summary Metrics

| Metric | v33 | v34 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | ~7,086 | ~7,094 | ~+8 |
| Integration tests (containers) | 24 | 24 | 0 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 1 | +1 |
| Lines added | — | ~750 | — |

---

## New Files

| File | LOC | Description |
|------|-----|-------------|
| `src/mcp_docker_tools.rs` | ~300 | 8 MCP tools for Docker container management, 5 tests |

## Modified Files

| File | Changes | Description |
|------|---------|-------------|
| `src/config_file.rs` | ~60 LOC | `ContainersConfig` struct + field on `ConfigFile`, 3 tests |
| `src/lib.rs` | ~6 LOC | `mcp_docker_tools` module registration + re-export |
| `src/bin/ai_assistant_standalone.rs` | ~300 LOC | `--containers` flag, `/docker` REPL commands, 2 tests |
| `src/server_axum.rs` | ~80 LOC | `mcp_server` in `AppState`, `POST /mcp` handler |

---

## Phase 1: Configuration (`config_file.rs`)

### `ContainersConfig` struct

```rust
pub struct ContainersConfig {
    pub enabled: bool,             // default: false
    pub default_timeout_secs: u64, // default: 60
    pub allowed_images: Vec<String>, // empty = all allowed
    pub mcp_enabled: bool,         // default: false
}
```

Added as `pub containers: ContainersConfig` field on `ConfigFile`.

### Tests: 3

- `test_containers_config_default` — defaults are disabled
- `test_containers_config_json_parse` — parse JSON with containers section
- `test_config_round_trip_with_containers` — serialize/deserialize round-trip

---

## Phase 2: MCP Docker Tools (`mcp_docker_tools.rs`, NEW)

Feature gate: `#[cfg(all(feature = "containers", feature = "tools"))]`

### 8 MCP Tools

| Tool Name | Parameters | Read-only | Destructive | Description |
|-----------|-----------|-----------|-------------|-------------|
| `docker_list_containers` | — | yes | no | List all managed containers |
| `docker_create_container` | `image` (req), `name` (opt), `cmd` (opt) | no | no | Create container from image |
| `docker_start_container` | `container_id` (req) | no | no | Start a created container |
| `docker_stop_container` | `container_id` (req), `timeout` (opt) | no | no | Stop a running container |
| `docker_remove_container` | `container_id` (req), `force` (opt) | no | yes | Remove a container |
| `docker_exec` | `container_id` (req), `command` (req array), `timeout_secs` (opt) | no | no | Execute command in container |
| `docker_logs` | `container_id` (req), `tail` (opt) | yes | no | Get container logs |
| `docker_container_status` | `container_id` (req) | yes | no | Get container status |

All tools include `McpToolAnnotation` hints per MCP 2025-03-26 spec.

### Tests: 5

- `test_register_mcp_docker_tools_count` — 8 tools registered
- `test_mcp_tool_list_schema` — verify list tool schema
- `test_mcp_tool_create_schema` — `image` is required
- `test_mcp_tool_annotations` — read-only/destructive hints correct
- `test_mcp_tool_exec_schema` — `command` is array type

---

## Phase 3: CLI REPL Docker Commands (`ai_assistant_standalone.rs`)

### `--containers` CLI flag

Enables Docker `/docker` commands in the interactive REPL.

### REPL `/docker` commands

| Command | Description |
|---------|-------------|
| `/docker list` | List all containers (ID, name, image, status) |
| `/docker create <image>` | Create container (`--name NAME`, `--cmd CMD...`) |
| `/docker start <id>` | Start a container |
| `/docker stop <id>` | Stop a container (`--timeout N`) |
| `/docker rm <id>` | Remove a container (`--force`) |
| `/docker exec <id> <cmd>` | Execute command in container |
| `/docker logs <id>` | Show container logs (`--tail N`) |
| `/docker status <id>` | Show container status |
| `/docker cleanup` | Remove all managed containers |
| `/docker help` | Show help |

### Tests: 2

- `test_parse_args_containers` — flag parsing
- `test_parse_args_all` — updated with `--containers`

---

## Phase 4: MCP HTTP Endpoint (`server_axum.rs`)

### `POST /mcp` route

- 200 OK with MCP JSON-RPC response when Docker available
- 503 Service Unavailable when Docker not available

### Auto-initialization

`build_mcp_docker_server()` detects Docker availability at startup and registers 8 tools. Returns `None` for graceful degradation.

---

## How to Use

```bash
# Start standalone with Docker support
cargo run --bin ai_assistant_standalone \
  --features "full,containers,server-axum" -- --repl --containers

# REPL commands
/docker help
/docker create busybox:latest --name test1
/docker start <id>
/docker exec <id> echo hello
/docker logs <id>
/docker cleanup

# MCP endpoint (from another terminal)
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

---

## Key Design Decisions

1. **Separate `mcp_docker_tools.rs`** — avoids `autonomous` dependency. Gated under `containers` + `tools` only.
2. **Bollard-based `ContainerExecutor`** — uses production executor, consistent with integration tests.
3. **`DockerHandle` type alias** — conditional type for clean feature-gated REPL integration.
4. **Graceful degradation** — Docker features return helpful errors when unavailable.
5. **`McpToolAnnotation` hints** — all 8 tools annotated per MCP spec.
6. **`Arc<RwLock<ContainerExecutor>>`** — shared across closures. Read lock for queries, write lock for mutations.
