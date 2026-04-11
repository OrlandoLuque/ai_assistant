# ai_assistant — Use Cases

This document wires the 20 binaries listed in
[`docs/BINARIES.md`](BINARIES.md) into **end-to-end scenarios**. Each case
answers four questions:

1. **Problem** — what the user is trying to do
2. **Binaries involved** — which executables to run
3. **Commands** — a copy-pasteable recipe
4. **Required features** — the Cargo feature flags you need

Eight canonical scenarios are documented below. Additional scenarios are
listed in the companion website page
[`ai_assistant-website/use_cases.html`](../../ai_assistant-website/use_cases.html).

---

## 1. Local chat with RAG over personal documents

**Problem.** You want a desktop chat UI that answers questions using your own
PDFs, markdown notes, and source files, 100% locally.

**Binaries.** `ai_gui`, `kpkg_tool`.

**Commands.**

```bash
# 1. Package your docs into an encrypted .kpkg bundle
cargo run --bin kpkg_tool --features rag -- create \
    --input ./my_notes \
    --output ./knowledge.kpkg \
    --password "$KPKG_PASS"

# 2. Launch the desktop GUI pointing at the bundle
cargo run --bin ai_gui --features "gui,rag" -- \
    --rag-bundle ./knowledge.kpkg
```

**Required features.** `gui`, `rag` (and optionally `full` for tools + memory).

**Security notes.** `.kpkg` files are AES-256-GCM encrypted; the password is
never stored and must be supplied at launch.

---

## 2. Cost gate in CI/CD

**Problem.** You want your CI pipeline to **fail** if the projected monthly
cost of an LLM-powered feature branch exceeds a budget.

**Binaries.** `ai_cli` (the new V77 `cost` subcommand).

**Commands.**

```bash
# In your CI step, after running the feature-branch workload:
cargo run --bin ai_cli --features full -- \
    cost budget --snapshot "$CI_ARTIFACT/cost_snapshot.json"

# The command exits non-zero if projected_monthly_usd > the snapshot's limit.
# Wire it into your job's exit gate:
if ! ai_cli cost budget --snapshot "$CI_ARTIFACT/cost_snapshot.json"; then
    echo "::error::Budget exceeded on feature branch"
    exit 1
fi
```

**Required features.** `full` (for the `CostDashboard` API).

**Security notes.** Snapshot paths are canonicalized before use. The JSON
schema rejects `NaN` / `Infinity` (validated in V75).

---

## 3. Scheduled summaries of feeds or logs

**Problem.** You want a cron-like daemon that runs LLM-backed summarization
jobs on a schedule — for example, a weekday morning briefing generated from
your internal knowledge base.

**Binaries.** `ai_jobs` (new in V77), optionally `ai_cli` for delegated jobs.

**Commands.**

```bash
# 1. Define a jobs.json manifest
cat > jobs.json <<'JSON'
{
  "assistant": {
    "provider": "ollama",
    "model": "llama3",
    "system_prompt": "You are a concise reporting assistant."
  },
  "jobs": [
    {
      "id": "rag_daily_brief",
      "name": "RAG-enhanced daily brief",
      "cron": "0 8 * * 1-5",
      "runtime": "embedded",
      "session_id": "daily_brief",
      "type": "agent",
      "task": "Generate today's daily brief using the team knowledge base"
    },
    {
      "id": "health_check",
      "name": "Quick health ping",
      "cron": "*/5 * * * *",
      "type": "shell",
      "command": "curl -sf http://localhost:8080/health"
    }
  ]
}
JSON

# 2. Validate the manifest
cargo run --bin ai_jobs --features scheduler -- validate jobs.json

# 3. Run the daemon
cargo run --bin ai_jobs --features "full,scheduler" -- run jobs.json
```

**Required features.** `scheduler` (minimum); add `full` to enable the
`embedded` runtime mode (in-process `AiAssistant` with RAG + tools).

**Security notes.** `MAX_JOBS = 1000`, per-job `timeout_secs` default 60s,
`std::panic::catch_unwind` protects the daemon, API key env vars are
referenced by **name** (never logged).

---

## 4. Private team assistant behind TLS + RBAC

**Problem.** You want a self-hosted HTTPS chat endpoint that multiple
colleagues can use, with role-based access control.

**Binaries.** `ai_assistant_server` (or `ai_assistant_standalone`).

**Commands.**

```bash
# 1. Launch the server with TLS + RBAC
cargo run --bin ai_assistant_server --features full -- \
    --bind 0.0.0.0:8443 \
    --tls-cert /etc/ssl/ai_assistant.crt \
    --tls-key  /etc/ssl/ai_assistant.key \
    --rbac-config /etc/ai_assistant/rbac.toml

# 2. Test from a client
curl -H "Authorization: Bearer $TOKEN" \
     --data '{"prompt":"Summarize yesterday's PRs"}' \
     https://ai.internal.corp/api/chat
```

**Required features.** `full`, `server-axum`.

---

## 5. Distributed RAG cluster

**Problem.** You have more documents than fit in a single node's RAM and want
to fan out RAG retrieval across a QUIC-meshed cluster.

**Binaries.** `ai_cluster_node`.

**Commands.**

```bash
# Bootstrap peer
cargo run --bin ai_cluster_node --features "full,server-cluster" -- \
    --bind 0.0.0.0:4433 --bootstrap

# Follower nodes
cargo run --bin ai_cluster_node --features "full,server-cluster" -- \
    --bind 0.0.0.0:4433 \
    --peers quic://bootstrap.internal:4433
```

**Required features.** `full`, `server-cluster`.

---

## 6. Voice assistant with emotional detection

**Problem.** You want a voice assistant that detects tone (including snoring)
from the microphone stream, pipes it through STT, and speaks the reply back.

**Binaries.** `ai_virtual_mic`, `ai_assistant_cli` or `ai_gui`.

**Commands.**

```bash
# 1. Start the virtual mic with the snore detector chain
cargo run --bin ai_virtual_mic --features audio-io -- \
    --mode transform \
    --effects snore_detector,voice_anonymizer

# 2. Point your chat client at the virtual device
cargo run --bin ai_gui --features "gui,audio-io" -- \
    --audio-input ai_virtual_mic
```

**Required features.** `audio-io` (and `gui` for the client).

---

## 7. Butler auto-configuration on a new machine

**Problem.** You just cloned the repo on a new laptop and want the assistant
to detect your environment, available providers, and GPU, then recommend a
starter config.

**Binaries.** `ai_cli butler`, `ai_setup`.

**Commands.**

```bash
# 1. Interactive butler advisor
cargo run --bin ai_cli --features full -- butler

# 2. Apply the recommendations via the terminal setup wizard
cargo run --bin ai_setup --features full
```

**Required features.** `full`.

---

## 8. MCP server as a Claude Desktop backend

**Problem.** You want Claude Desktop (or any other MCP client) to talk to
your local `ai_assistant` instance — exposing 40+ MCP tools.

**Binaries.** `ai_assistant_server` (MCP endpoint enabled by default when
built with `full`).

**Commands.**

```bash
# 1. Launch the MCP endpoint on the Unix socket Claude Desktop listens on
cargo run --bin ai_assistant_server --features full -- \
    --mcp-transport stdio

# 2. In Claude Desktop, add to ~/.config/Claude/claude_desktop_config.json:
# {
#   "mcpServers": {
#     "ai_assistant": {
#       "command": "/path/to/target/release/ai_assistant_server",
#       "args": ["--mcp-transport", "stdio", "--features", "full"]
#     }
#   }
# }
```

**Required features.** `full` (which enables MCP).

---

## Cross-references

- [`docs/BINARIES.md`](BINARIES.md) — the 20-binary authoritative inventory.
- [`docs/IMPROVEMENTS_V77.md`](IMPROVEMENTS_V77.md) — why V77 added `ai_jobs`
  and the `ai_cli cost` subcommand.
- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
