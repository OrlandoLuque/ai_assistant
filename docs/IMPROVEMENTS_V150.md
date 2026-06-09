# IMPROVEMENTS_V150 — SSE streaming passthrough + per-chunk timeout

**Version:** 0.2.100 → 0.2.101
**Scope:** `src/bin/ai_proxy.rs` + `src/bin/mock_llama_server.rs`
**Feature flag:** `server-axum` (no new flags)
**Plan:** `../../ai_assistant_plans/V150_streaming_passthrough.md`

## Why

V78 wired the multi-backend `ai_proxy` to forward requests with
`resp.bytes().await` on every path — both the free path and the
security gateway. That bufferized the full upstream response before
returning anything to the client. For JSON it's harmless; for
`text/event-stream` it defeats the entire point of SSE. Clients
asking for `stream: true` received the whole token stream in one
chunk at the very end. V149 deferred fixing this because it touches
the hot path; V150 takes that risk on a clean isolated patch.

## What changed

### Forwarding core split

`forward_core` is unchanged (still buffers — needed by the non-stream
chat handler so output guards can scan the body). A new sibling
`forward_core_streamable` takes the same `parts + body_bytes +
outbound_hops` tuple but returns an axum `Response` directly. It
inspects the upstream's `content-type`:

| Upstream content-type | What `forward_core_streamable` does                                  |
|-----------------------|----------------------------------------------------------------------|
| `text/event-stream`   | Pipes `bytes_stream()` through `streaming_body_with_chunk_timeout`. |
| `application/x-ndjson`| Same — both are line-delimited streaming formats.                    |
| anything else         | Buffers via `bytes().await` (same as `forward_core`).                |

Three forwarding sites switched over:

1. **Gateway passthrough handler** (axum fallback route). Anything
   that isn't `/health`, `/metrics`, `/v1/models`, or chat-completions
   lands here — embeddings, custom endpoints, anything a backend
   exposes. Streams now propagate.
2. **Gateway chat handler — stream branch** (`json["stream"] == true`).
   The client explicitly asked for SSE; we honor it end-to-end.
3. **Free proxy path** (when `security` feature is off). Same
   stream-vs-buffer decision.

The chat handler's **non-stream branch** (no `"stream": true` in
body, output guards may be active) deliberately stays on
`forward_core`. Output guards cannot scan partial chunks, so
bufferizing is correct. But now, if the upstream returns SSE shape
anyway (some backends ignore `stream: false`), the response carries
`x-streaming-disabled: output-guard-active` and bumps
`proxy_stream_disabled_output_guard` so operators can see it.

### Per-chunk inactivity timeout

`streaming_body_with_chunk_timeout(upstream, chunk_timeout, metrics)`
wraps `upstream.bytes_stream()` in an `async_stream::stream!` block
that calls `tokio::time::timeout(chunk_timeout, s.next())` on every
chunk. Three outcomes:

- **Chunk arrives within timeout** → yield to client,
  `stream_chunks_total += 1`.
- **Upstream errors mid-stream** → yield `io::Error::other(...)`,
  `stream_aborts_upstream += 1`, break.
- **Timeout fires** → yield `io::Error::new(TimedOut, ...)`,
  `stream_aborts_chunk_timeout += 1`, break.

The default `chunk_timeout` is 30s, tunable via the new
`[routing] stream_chunk_timeout_secs` config (also in
`RoutingSection`). Lower values defend more aggressively against
slow-loris-style backends; higher values accommodate genuinely
slow generators.

### Streaming-disabled header

When the non-stream chat path observes an SSE-shaped upstream
content-type, the response carries:

```
x-streaming-disabled: output-guard-active
```

This is the only `x-streaming-disabled` reason emitted today, but
the helper takes a `&'static str` reason so adding more values
later (e.g. `cache-only`, `client-prefers-buffered`) is one-line.

### Metrics

`/metrics` (Prometheus text format) gains five `proxy_stream_*`
counters. All `AtomicU64`. All exposed identically on the free path
(`proxy_metrics_handler`) and the gateway path
(`gateway_metrics_handler` delegates to the same body builder).

| Counter                                | Meaning                                                       |
|----------------------------------------|---------------------------------------------------------------|
| `proxy_stream_chunks_total`            | Chunks emitted across all streamed responses.                 |
| `proxy_stream_aborts_chunk_timeout`    | Streams cut because a chunk gap exceeded the per-chunk limit. |
| `proxy_stream_aborts_upstream`         | Streams cut by an upstream error mid-stream.                  |
| `proxy_stream_aborts_client_close`     | Streams cut by the client closing the connection early.       |
| `proxy_stream_disabled_output_guard`   | Times an SSE upstream was buffered because guards were active.|

## Config

New `[routing]` field (all optional, default preserves V78 behavior
plus the new 30s stream timeout):

```toml
[routing]
# ... V149 fields ...

# V150
# stream_chunk_timeout_secs = 30
```

## Tests

Five new tests in `tests::gateway_e2e`:

- `test_gateway_e2e_v150_passthrough_sse_streams` — fallback route
  streams SSE end-to-end, body contains every chunk, chunk metric
  advances.
- `test_gateway_e2e_v150_chunk_timeout_aborts_stream` — backend
  emits one chunk then hangs 1h, proxy aborts within 150ms,
  `stream_aborts_chunk_timeout` increments.
- `test_gateway_e2e_v150_chat_stream_branch_streams` — chat handler
  with `stream:true` pipes SSE chunks through; metric advances.
- `test_gateway_e2e_v150_non_stream_chat_with_sse_upstream_sets_disabled_header`
  — non-stream chat path, SSE upstream → response carries
  `x-streaming-disabled: output-guard-active`, metric advances.
- `test_gateway_e2e_v150_non_stream_chat_json_no_disabled_header` —
  regression: JSON response has no `x-streaming-disabled` header.

The mock harness uses `axum::body::Body::from_stream` over
`async_stream::stream!` for the streaming responders. The
chunk-timeout test sleeps the responder 1h (much longer than any
test timeout) and asserts the metric advances within 2s of receiving
the first chunk.

`src/bin/mock_llama_server.rs` also gained a configurable SSE
endpoint and an SSE branch of `POST /v1/chat/completions` (kicks
in when the body contains `"stream":true`) for any future test that
prefers an out-of-process backend over an in-process axum mock.

## Risks & mitigations

| Risk                                         | Mitigation                                                                                            |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Hot-path regression on JSON responses        | `forward_core_streamable` inspects content-type; non-SSE upstreams still go through `bytes().await`. |
| Slow-loris-style backend pinning sockets     | Per-chunk timeout, default 30s, tunable down via config.                                              |
| Output guards silently degraded by streaming | Non-stream chat path keeps `forward_core`; explicit `x-streaming-disabled` header on SSE upstreams.   |
| Streams not cacheable                        | Cache layer was already bypassed on `stream:true` requests in V78; documented in CHANGELOG.           |
| Client closes mid-stream → upstream leaks    | `stream_aborts_client_close` counter; stream drops naturally when the body is dropped client-side.    |

## Known gaps (deliberate)

- **No backpressure metric.** A backend that's faster than the client
  drives memory growth on the `from_stream` adapter. axum/hyper handle
  the actual backpressure; we don't surface a counter for it.
- **No partial-stream cache.** Streams stay uncacheable. The plan
  acknowledges this — adding a "record-and-replay" cache for SSE
  responses is V151+ territory.
- **`stream_aborts_client_close` is best-effort.** It increments only
  when the upstream channel surfaces a send-side error; some closes
  manifest as a clean upstream EOF and stay invisible.

## Code map (key sites)

| Site                                       | What it does                                                              |
|--------------------------------------------|---------------------------------------------------------------------------|
| `src/bin/ai_proxy.rs:195`                  | `DEFAULT_STREAM_CHUNK_TIMEOUT` + `X_STREAMING_DISABLED` constants.        |
| `src/bin/ai_proxy.rs:409`                  | `streaming_body_with_chunk_timeout` (per-chunk timeout + metrics).        |
| `src/bin/ai_proxy.rs:456`                  | `inject_streaming_disabled` (header injection on guard-disabled bodies).  |
| `src/bin/ai_proxy.rs:565-580`              | `ProxyMetrics` extended with 5 V150 counters.                             |
| `src/bin/ai_proxy.rs:3135`                 | `forward_core_streamable` (stream-vs-buffer dispatcher).                  |
| `src/bin/ai_proxy.rs:2642, 2749`           | Two of the three call sites (chat stream branch + passthrough fallback). |
| `src/bin/ai_proxy.rs:2885-2898, 2966-2970` | Non-stream chat: SSE detection + header injection.                        |

## Follow-ups

- **Stream cache (record/replay).** Would let two clients streaming
  the same prompt share one upstream stream. Significant design work.
- **Per-stream tracing.** A request-id-correlated log line on
  start/end/abort would help debug stream-related incidents.
- **Connection-pool warmth metric.** Streams hold sockets longer;
  the existing reqwest pool sizing may need a knob.
