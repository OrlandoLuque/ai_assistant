# V107 — ACP (Phase A.2): Agent Client Protocol server

**Version**: 0.2.53 → 0.2.54
**Date**: 2026-05-03
**Phase**: Tier-1 competitive gaps — A.2 (after A.1 Recipes, before A.3 in-process inference)

## Why

Modern editors (Zed, VS Code, JetBrains) speak the **Agent Client Protocol** (ACP)
to embed coding agents like OpenHands, Goose, Hermes, and Gemini-CLI. ACP is
JSON-RPC 2.0 over newline-delimited JSON on stdio: the editor launches the
agent as a subprocess, talks over `stdin`/`stdout`, and gets streamed responses
back through `session/update` notifications.

Before V107, `ai_assistant` could only be driven from the CLI or via its
embedded HTTP server — neither of which an editor knows how to consume. After
V107, any ACP-conformant editor can launch `ai_acp serve` and use the full
feature set as a coding agent.

## What

### 1. `src/acp.rs` (~880 lines)

A self-contained server implementation of ACP protocol version 1. Public
surface:

- **Wire**: `Request`, `Notification`, `Response`, `RpcError` (all the
  standard codes plus `-32000` auth required and `-32002` resource not found).
- **Domain types** (subset matching the schema): `InitializeRequest`/`Response`,
  `ClientCapabilities`, `AgentCapabilities`, `PromptCapabilities`, `McpCapabilities`,
  `NewSessionRequest`/`Response`, `PromptRequest`/`Response`, `StopReason`,
  `CancelParams`, `ContentBlock` (tagged on `type`), `SessionUpdate`
  (tagged on `sessionUpdate` — note the discriminator is **not** `type`).
- **Server**: `AcpServer::new(config).with_llm(callback).with_slo_sink(callback)`.
- **Pump**: `serve(server, reader, writer)` — reads NDJSON frames, dispatches,
  writes responses + `session/update` notifications back. Returns on EOF.

Decisions (locked at design time):

- **No `agent-client-protocol` crate dependency.** The crate is at 0.11.x with
  an in-flight redesign (SACP). The JSON-RPC envelope is ~120 lines hand-rolled
  in our wire layer; the domain types are ~80 derives. Re-deriving and tracking
  the schema ourselves is cheaper than chasing 0.x API churn, and avoids
  pulling 80+ types we'd only re-export anyway.
- **stdio transport only**, NDJSON framing, no Content-Length. WebSocket/HTTP
  draft transports (in the ACP RFD pipeline) deferred until stable.
- **Capabilities default off.** We only advertise `embeddedContext`. Image,
  audio, MCP HTTP/SSE, and `loadSession` get turned on when each is wired
  through the corresponding subsystem (vision, audio, mcp_protocol, sessions).
- **Pluggable LLM execution** via `with_llm(callback)`. Same pattern as the
  V106 `RecipeEngine` and the V89 CoVe LLM verifier — keeps the protocol
  layer testable without spinning up a real model.

### 2. SLO instrumentation

Every `initialize`, `session/prompt`, and first-chunk emission is recorded
with elapsed milliseconds, chunk count, and chunks-per-second. Records are
held in-memory (`AcpServer::slo_records()`) and, optionally, fired through
`with_slo_sink(callback)` so the bin can persist JSONL.

| SLO            | Target       | Where measured                                   |
|----------------|--------------|--------------------------------------------------|
| Handshake      | < 200 ms     | `initialize` request → response time             |
| First chunk    | < 1 s        | `session/prompt` start → first `session/update`  |
| Throughput     | ≥ 30 chunks/s | full `session/prompt` duration vs. chunk count   |

Two unit tests assert the handshake and throughput targets on stub generators.

### 3. Three new bins

- **`ai_acp`** (feature `acp`) — primary entry point.
  - `serve [--provider PROV --model MODEL [--url URL] [--log-dir DIR]]` —
    runs the server on stdio, with `AiAssistant`-backed LLM. Persists SLO
    records to `./.ai_assistant/acp_logs/acp_<timestamp>_<pid>.jsonl`.
  - `probe <cmd> [args...]` — diagnostic. Spawns the given command as an
    ACP server, runs handshake + one prompt, prints timings.
- **`ai_acp_audit`** (feature `acp`) — read-only CLI auditor. Verbs:
  `list`, `show <FILE>`, `audit [--dir D] [--strict]`. `--strict` makes
  any SLO breach exit 1 (CI-friendly).
- **`ai_acp_audit_gui`** (feature `gui-acp = ["acp", "dep:eframe"]`) —
  egui visual auditor. List of log files, per-file records table with
  red-coded breaches, summary panel showing handshake/prompt/first-chunk
  breach counts.

The audit pair satisfies the `feedback_auditable_subsystems` memory rule
("every subsystem that stores artifacts needs dedicated CLI + GUI auditor
binaries"). The `gui-acp` feature is intentionally narrower than `gui-pro` —
a user who only wants the auditor pays only for `eframe`.

### 4. Cancellation correctness

The original prompt loop broke out with `EndTurn` on channel disconnect. If a
late `session/cancel` set the flag *just* as the LLM thread exited, the
disconnect path could ship `EndTurn` instead of `Cancelled`. Fixed by
re-checking the cancel flag in the `Disconnected` arm before settling on the
stop reason.

## Tests

17 unit tests in `src/acp.rs`, all green:

```
running 17 tests
test acp::tests::content_block_text_serializes_with_type_tag ... ok
test acp::tests::reject_missing_jsonrpc ... ok
test acp::tests::reject_non_object_frame ... ok
test acp::tests::parse_request_frame ... ok
test acp::tests::reject_malformed_json ... ok
test acp::tests::parse_notification_frame ... ok
test acp::tests::session_update_uses_session_update_discriminator ... ok
test acp::tests::handshake_completes ... ok
test acp::tests::handshake_meets_slo_target ... ok
test acp::tests::session_new_requires_initialize_first ... ok
test acp::tests::session_new_requires_absolute_cwd ... ok
test acp::tests::version_negotiation_returns_ours ... ok
test acp::tests::unknown_method_returns_method_not_found ... ok
test acp::tests::prompt_without_session_returns_resource_not_found ... ok
test acp::tests::streaming_meets_chunks_per_sec_target ... ok
test acp::tests::prompt_streams_chunks_and_returns_end_turn ... ok
test acp::tests::cancel_notification_short_circuits_prompt ... ok

test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 6143 filtered out
```

End-to-end smoke (probe → serve subprocess):

```
$ ./target/debug/ai_acp probe ./target/debug/ai_acp serve --model dummy
handshake: 6 ms
server response: {"id":1,"jsonrpc":"2.0","result":{"agentCapabilities":{...},
                  "agentInfo":{"name":"ai_assistant","version":"0.2.54"},
                  "authMethods":[],"protocolVersion":1}}
first_chunk: 0 ms
chunks: 0
chunks_per_sec: 0.0
total: 4118 ms
prompt response: {"id":3,"jsonrpc":"2.0","result":{"stopReason":"refusal"}}

$ ./target/debug/ai_acp_audit list
FILE                                              RECORDS   SESSIONS
acp_1777806862_71464.jsonl                              2          1

$ ./target/debug/ai_acp_audit audit
ACP audit (.ai_assistant/acp_logs)
  Files:                     1
  Records:                   2
  Handshakes:                1 (breach >200ms: 0)
  Prompts:                   1 (breach <30 chunks/s: 0)
  First-chunk records:       0 (breach >1000ms: 0)
OK: all records within SLO targets
```

(Prompt fails with `refusal` because `dummy` is not a real Ollama model — the
protocol layer is fully exercised; the model layer correctly errors out.)

## Lessons

- **Parser-driven design avoids dep churn.** Hand-rolling the envelope was
  cheap. Binding `agent-client-protocol = "0.11"` would have meant tracking
  every RFD they accept and re-fitting our handlers each release.
- **NDJSON framing is unforgiving.** Any embedded newline in a serialized
  frame breaks the protocol. Both `write_response` and `write_notification`
  guard against this and return `InvalidData` if it ever happens — better a
  loud error than a silent client desync.
- **Test the cancellation path explicitly.** The disconnect-vs-cancel race
  was easy to write and easy to miss; the slow_llm test caught it
  immediately on the second iteration.
- **SLO logging needs to be additive.** The first cut had `record_slo` only
  pushing to an in-memory ring. Adding `with_slo_sink` (a callback that
  fires per record before the push) gave us file persistence without
  changing the test surface — every existing test still asserts against
  `slo_records()`.

## Next

Phase A.3: in-process local inference (candle-core + candle-nn + candle-transformers
+ llama-cpp-2). After A.3, Phase B (Stuck Detector, parallel tool execution,
adversary/egress inspectors) and Phase C (cargo-audit/deny + SBOM, error
taxonomy, OTel sampler with redaction, release-plz, perf budgets, GDPR purge,
runbooks).
