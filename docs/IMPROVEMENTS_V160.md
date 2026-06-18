# IMPROVEMENTS_V160 — Streaming output guardrails for ai_proxy

**Version:** 0.2.110 → 0.2.111
**Scope:** `src/bin/ai_proxy.rs`
**Feature:** `security` (existing)

## Why

The binary catalogue page for `ai_proxy` honestly listed "output
guardrails don't run over a live stream." When a client requested
`stream: true`, the chat path piped the SSE through unmodified and the
PII / toxicity / attack guards were bypassed (V150 set
`x-streaming-disabled: output-guard-active` to admit this). The library
already shipped a `StreamingGuardrailPipeline` (chunk-by-chunk guards
with Pass / Flag / Pause / Block) — V160 wires it into the gateway's
streaming path so guards actually run over the wire.

## What changed

### `streaming_body_with_guards`

A guarded sibling of V150's `streaming_body_with_chunk_timeout`:

1. Accumulates upstream bytes and drains complete SSE frames
   (`\n\n`-terminated) — robust to byte chunks that split or merge
   frames.
2. `extract_sse_delta_content` pulls every `choices[].delta.content`
   from a frame. Role-announcement / `[DONE]` / keep-alive frames yield
   no text and pass through unguarded.
3. The text feeds `StreamingGuardrailPipeline::process_chunk`; the
   returned action drives forwarding:
   - **Pass** (and any future non-blocking variant) → forward the frame.
   - **Flag** → forward + bump `proxy_stream_guard_flags_total`.
   - **Pause** → hold the frame; a later Pass flushes the held bytes.
     The hold is bounded (`MAX_HELD_BYTES = 256 KiB`) — exceeding it
     fails closed (Block).
   - **Block** → stop forwarding, drop held/buffered suspect bytes, emit
     a terminal `data: {"error":{…,"code":"output_guard"}}\n\ndata:
     [DONE]\n\n`, bump `proxy_stream_guard_blocks_total`.
4. Still wrapped in the V150 per-chunk inactivity timeout.

### Wiring

- `build_streaming_pipeline(&MiddlewareSection)` builds the streaming
  pipeline from the enabled **output** guards (`enable_pii_output` →
  `StreamingPiiGuard`, `enable_toxicity_output` →
  `StreamingToxicityGuard::with_defaults()`, `enable_attack_filter` →
  `StreamingPatternGuard` with common injection markers). Returns `None`
  if none are on.
- `forward_core_streamable` takes an
  `Option<StreamingGuardrailPipeline>`. The chat stream branch builds and
  passes it; the generic passthrough (embeddings, etc.) passes `None`.
- Two `/metrics` counters added.

## The honest contract

Streaming guards catch violations **mid-stream**. They can't un-send the
tokens already streamed before the trigger, but they stop the leak from
continuing and terminate the stream. The "blocked" e2e test asserts
exactly this: a secret payload placed **after** the trigger phrase never
reaches the client, even though earlier tokens did.

## Tests

- `test_extract_sse_delta_content_parses_deltas` — delta extraction,
  empty for role/`[DONE]`.
- `test_build_streaming_pipeline_toggles` — None by default, Some when an
  output guard is on.
- `test_gateway_e2e_v160_stream_blocked_by_guard` — real backend streams
  an injection phrase + a secret tail; asserts the block event is
  present, `SECRET_PAYLOAD_LEAKED` is **not**, and the block metric
  advanced.
- `test_gateway_e2e_v160_clean_stream_passes` — clean stream with guards
  on passes through, no block.

118 ai_proxy tests pass; clippy clean on `server-axum,security` and
`server-axum,security,server-axum-tls`; the non-security `server-axum`
build still compiles.

## Relationship to V150

V150's `x-streaming-disabled` header (and `stream_disabled_output_guard`
metric) still describe the **non-stream** chat path that buffers an
SSE-shaped upstream. V160 is about the **streaming** path, which now
guards instead of bypassing.
