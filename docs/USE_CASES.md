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

## 4. Private team assistant behind TLS + RBAC (with gateway guardrails)

**Problem.** You want a self-hosted HTTPS chat endpoint that multiple
colleagues can use, with role-based access control **and** server-side
PII redaction, budget enforcement, and audit logging — without bolting
middlewares on top of every deployment.

**Azure OpenAI note (V80).** The upstream server supports Azure OpenAI
as a first-class provider. Set `provider = "azure_openai"` in your config,
or pass `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and
`AZURE_OPENAI_DEPLOYMENT` as environment variables. Azure uses the
`api-key` header (not Bearer) and the Azure-specific deployment URL
automatically.

**Binaries.** `ai_assistant_server` (or `ai_assistant_standalone`) as
the upstream, **`ai_proxy` (V78)** as the hardened gateway.

**Commands.**

```bash
# 1. Launch the upstream server(s) — normal RBAC + TLS as before
cargo run --bin ai_assistant_server --features full -- \
    --bind 127.0.0.1:8090 \
    --tls-cert /etc/ssl/ai_assistant.crt \
    --tls-key  /etc/ssl/ai_assistant.key \
    --rbac-config /etc/ai_assistant/rbac.toml

# 2. Put ai_proxy in front with the full middleware stack.
#    See examples/ai_proxy.toml for every knob.
export AI_PROXY_API_KEY="$(cat /etc/ai_proxy/key)"
cargo run --bin ai_proxy --features "server-axum,security" -- \
    --config /etc/ai_proxy/ai_proxy.toml

# 3. Test from a client — every request is rate-limited, PII-redacted,
#    budget-checked, and audited.
curl -H "Authorization: Bearer $AI_PROXY_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}' \
     https://ai.internal.corp/v1/chat/completions
```

**What the gateway adds over a plain `ai_assistant_server`:**

- **PII redaction** on `messages[].content` for roles {user, system}
  before the request ever reaches the upstream
- **Budget enforcement** — requests over the monthly / per-request cap
  get 429 with `X-Reason: budget-exceeded`
- **LRU response cache**, PII-safe (tainted responses never get stored)
- **Append-only JSONL audit log** with rotation and symlink-safe open;
  API keys only ever logged as SHA-256 hash
- **Per-key rate limiting** with `key:/sess:/ip:` bucket priority
- `X-Request-Id` and `X-Cache` headers on every response

**Required features.** Upstream: `full` + `server-axum` (+ `server-axum-tls`).
Gateway: `server-axum` + `security`. See `docs/IMPROVEMENTS_V78.md` for
the full design and the list of 13 security mitigations.

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

---

## 9. NPCs in games via FFI (new in V79)

**Problem.** You're building a game in Unity, Unreal, or Bevy and want
non-player characters to drive their dialogue and behavior through an
LLM — with no out-of-process HTTP requests, no tokio runtime in the
engine, and a native C ABI the game engine can consume directly.

**Binaries / libraries.** `libai_assistant.{so,dylib,dll}` +
`include/ai_assistant.h` — built with `--features ffi`.

**Commands.**

```bash
# 1. Build the shared library + auto-generated C header.
cargo build --features ffi --profile release-fast

# 2. Drop the artifacts into your engine's native plugin folder:
#    - include/ai_assistant.h
#    - target/release-fast/libai_assistant.{so,dylib,dll}
#    - target/release-fast/libai_assistant.a (optional, for Unreal
#      which prefers static linking)

# 3. Call from C (minimal NPC driver):
cat > npc_demo.c <<'C'
#include "ai_assistant.h"
#include <stdio.h>

int main(void) {
    AiAssistantHandle *h = ai_assistant_new_with_prompt(
        "You are a gruff blacksmith. Reply in <= 30 words, in-character.");
    ai_assistant_set_provider(h, AI_PROVIDER_KIND_OLLAMA);
    ai_assistant_set_model(h, "llama3.2:3b");

    char *reply = NULL;
    int rc = ai_assistant_send_message(
        h, "A stranger asks about your wares.", &reply);
    if (rc == 0) {
        printf("Blacksmith: %s\n", reply);
        ai_assistant_free_string(reply);
    }
    ai_assistant_free(h);
    return rc;
}
C

gcc -I include npc_demo.c -L target/release-fast -lai_assistant \
    -o /tmp/npc_demo
LD_LIBRARY_PATH=target/release-fast /tmp/npc_demo
```

**Required features.** `ffi` (minimum, zero-dep). Add `rag` to
automatically build RAG context from an indexed world lore corpus.
Add `full` to unlock every provider and tool the library supports.

**Security notes.**

- Each `AiAssistantHandle *` is **single-threaded** (SQLite-style).
  Pin each NPC's handle to a dedicated worker thread, or use a
  message queue to serialize access.
- Every entry point has a `catch_unwind` panic boundary — panics
  never escape into the game engine.
- Strings from `ai_assistant_send_message` must be freed via
  `ai_assistant_free_string`, **not** `free(3)`.
- Build with `--profile release-fast`, not `release` — the default
  `release` profile uses `panic = "abort"` and makes `catch_unwind`
  a no-op. `build.rs` warns when it detects the dangerous combo.

**See also.**

- [`docs/FFI.md`](FFI.md) — full API reference
- [`examples/ffi_c/`](../examples/ffi_c/) — C example with per-platform
  build instructions
- [`examples/ffi_python/`](../examples/ffi_python/) — Python example
  (ctypes, zero-dep)
- [`examples/ffi_node/`](../examples/ffi_node/) — Node.js example
  (koffi, pure-JS FFI bridge)
- [`examples/ffi_java/`](../examples/ffi_java/) — Java example (JNA)
- [`docs/BINARIES.md`](BINARIES.md#library-artifacts-v79-new) —
  library artifact naming per platform

---

## Cross-references

- [`docs/BINARIES.md`](BINARIES.md) — the 20-binary authoritative inventory.
- [`docs/FFI.md`](FFI.md) — V79 C FFI API reference (updated V80: Azure setters).
- [`docs/IMPROVEMENTS_V80.md`](IMPROVEMENTS_V80.md) — V80 Azure OpenAI provider.
- [`docs/IMPROVEMENTS_V79.md`](IMPROVEMENTS_V79.md) — V79 workstreams
  and design decisions.
- [`docs/IMPROVEMENTS_V77.md`](IMPROVEMENTS_V77.md) — why V77 added `ai_jobs`
  and the `ai_cli cost` subcommand.
- [`CHANGELOG.md`](../CHANGELOG.md) — release history.
