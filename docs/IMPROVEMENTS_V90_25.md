# V90.20–V90.25 — Vision wiring closure: carriers, surfaces, integration

**Version**: 0.2.49 → 0.2.50
**Feature flag**: `vision` (additive — no new flag)
**Date**: 2026-04-28

## Scope

V90.18–V90.19 wired the centre of the message graph (canonical
`ChatMessage`, persistence, agents, FFI, plugins, embeddings). V90.20–V90.25
closes the *outer ring* — every public surface that touches a chat-like
type now carries images end-to-end, with cross-module integration tests
and bench coverage to prevent silent regressions.

This series is the final wave of `vision_wiring_full_plan.md`. After
V90.25 there are no remaining "stub" call sites in the plan.

## Change set (batches from `vision_wiring_full_plan.md`)

| Tag | Batch | File(s) | What |
|-----|-----:|---------|------|
| V90.20 | 70 | `file_references.rs` | `FileReference.image_ref: Option<ImageRef>` so file-references and image-attachments share one wire shape. |
| V90.20 | 72 | `a2a_protocol.rs` | `A2AMessage::image()` constructor + `extract_image_parts()` — closes the silent-discard bug where vision content vanished through agent hops. |
| V90.20 | 73 | `token_counter.rs` | `estimate_image_tokens(detail, count)` honoring OpenAI's per-tile math; `estimate_messages_with_images()` aggregator. |
| V90.20 | 74 | `context_budget.rs` | `ContextSource::image_token_estimate()` trait method (default 0); allocator reserves image budget *before* text packing. |
| V90.20 | 76 | `messages.rs` | `AiResponse::Image(ImageData)` variant so image-out from Gemini / GPT-4o-image arrives through the canonical channel. |
| V90.21 | 77 | `batch.rs` | `BatchRequest.images` + `with_image` / `with_images` builders. |
| V90.21 | 78 | `prompt_chaining.rs` | `ChainStep.images` + `with_step_images` builder. |
| V90.21 | 79 | `model_ensemble.rs` | `execute_with_images()` extends the ensemble closure surface to take an image slice. |
| V90.21 | 80 | `regeneration.rs` | `RegenerationRequest.images` + `with_images` builder. |
| V90.22 | 20 | `faithfulness.rs` | `VisualGroundednessReport` + `score_visual_groundedness()` — fixed visual-vocab heuristic for response/text alignment with attached images. |
| V90.22 | 22 | `sse_streaming.rs` | `SseEvent::image_chunk(media_type, base64)` + `is_image()` / `decode_image()` for `event: image` envelopes. |
| V90.23 | 19 | `agent_methodology.rs` | `TaskStep.images: Vec<ImageRef>` so methodology audit trails carry vision evidence. |
| V90.23 | 68 | `unified_persistence.rs` | SQLite migration V6 adds `session_message_attachments` table; `attach_image()` / `attachments_for_message()` round-trip with FK cascade. |
| V90.24 | 25 | `tests/vision_integration.rs` | 10 cross-module integration tests covering ChatMessage → A2A → context_budget → SQLite → AiResponse. Adds `SqliteSessionStore::message_ids_for_session()` helper. |
| V90.25 | 41 | `websocket_streaming.rs` | `WsFrame::image_binary` / `as_image_binary` / `as_image_input` with v1 self-describing envelope; `WsAiMessage::Image` text variant for SSE-parity. |
| V90.25 | 32/81 | `widgets.rs` | `drain_dropped_images()` + `chat_input_with_attachments()` — egui chat input absorbs drag-drop image files into a staged `Vec<ImageInput>`, validates against `VisionLimits`, emits `ChatInputSubmission`. |
| V90.25 | 26 | `benches/vision_benchmarks.rs` | `from_bytes` / `sha256` / `detect_media_type` / `store_round_trip` benchmark groups, gated on the vision feature. |

## SQLite migration V6 — `session_message_attachments`

```sql
CREATE TABLE session_message_attachments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL
                REFERENCES session_messages(id) ON DELETE CASCADE,
    image_key   TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
```

Foreign-key cascade is the design contract: deleting a session drops its
messages, which drops the message attachments. Tests in
`unified_persistence::tests::test_attachments_cascade_delete` enforce
this invariant; tests `test_schema_version_tracking` and
`test_migration_idempotency` were updated from 5→6 migrations.

## WS binary image envelope (v1)

`WsFrame::image_binary("image/png", &bytes)` produces a `WsOpcode::Binary`
frame with the following payload:

```
+--------+-----------------+----------------+----------------+
| 1 byte |   2 bytes BE    |   N bytes      |  remaining     |
| 0x01   | media-type len  | media-type     |  image bytes   |
| (ver)  |   (n; u16 BE)   | (UTF-8)        |                |
+--------+-----------------+----------------+----------------+
```

`as_image_binary()` decodes back to `(media_type, &[u8])`;
`as_image_input()` further wraps as a base64-encoded `ImageInput` for
direct injection into the vision pipeline. Truncated frames return
`None` rather than panicking.

## egui chat-input attachment model

The new `chat_input_with_attachments()` is **stateful**: the caller owns
a `Vec<ImageInput>` of staged attachments. Each frame the function:

1. Drains `ctx.input(|i| i.raw.dropped_files.clone())` and appends valid
   image bytes to `staged_images` (validation via
   `ImageInput::from_bytes_validated`).
2. Renders a chip row with attachment count + `clear` button.
3. Reuses `chat_input_multiline` for text + Enter handling.
4. On submit, returns `Some(ChatInputSubmission { text, images })` and
   `mem::take`s the staged vector so subsequent frames start clean.

## Test additions

- `tests/vision_integration.rs` — 10 cross-module tests (passes under
  `--features "vision rag a2a"`).
- `unified_persistence::tests` — 2 new tests for attach/cascade.
- `websocket_streaming::tests` — 5 new vision-gated tests.
- `widgets::chat_input_attachments_tests` — 3 new vision+egui-widgets
  tests.
- `messages::tests` — 3 new tests for `AiResponse::Image` accessors.
- `context_budget::tests` — 2 new tests for image-token reservation.

Total: **25 new vision-gated tests** added across this series.
