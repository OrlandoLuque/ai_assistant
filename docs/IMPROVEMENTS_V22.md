# v22 Improvements Changelog

> Date: 2026-02-28

v22 is a **code quality** release focused on eliminating compiler warnings and reducing dead-code suppressions. No new features or tests — purely structural hygiene.

---

## Summary Metrics

| Metric | v21 | v22 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,963 | 5,963 | 0 |
| Clippy warnings | 0 | 0 | 0 |
| `#[allow(dead_code)]` annotations | 13 | 1 | -12 |
| Public API accessors added | 0 | 17 | +17 |

---

## Phase 1 — Fix 4 clippy warnings in ai_test_harness.rs

| # | File | Line | Warning | Fix |
|---|------|------|---------|-----|
| 1 | `src/bin/ai_test_harness.rs` | 9884 | `collapsible_else_if` | Collapsed `else { if }` to `else if` |
| 2 | `src/bin/ai_test_harness.rs` | 10061 | `manual_range_contains` | `!(8.0..=12.0).contains(&ratio)` |
| 3 | `src/bin/ai_test_harness.rs` | 10174 | `len_zero` | `.is_empty()` |
| 4 | `src/bin/ai_test_harness.rs` | 10380 | `map_or` | `.is_none_or(...)` |

---

## Phase 2 — Remove 12 `#[allow(dead_code)]` annotations

For each annotation, a public accessor method was added so the stored field is reachable through the public API, eliminating the need for the suppression.

| # | File | Struct | Field(s) | Accessor(s) Added |
|---|------|--------|----------|-------------------|
| 1 | `src/agent.rs` | `PlanningAgent` | `config` | `config()` |
| 2 | `src/auto_model_selection.rs` | `FallbackChain` | `strategy` | `strategy()` |
| 3 | `src/context_composer.rs` | `ConversationCompactor` | `summary_max_tokens` | `summary_max_tokens()` |
| 4 | `src/conversation_flow.rs` | `FlowAnalyzer` | `topic_keywords` | `add_topic_keyword()`, `topic_keywords()` |
| 5 | `src/debug.rs` | `RequestInspector` | `capture_requests`, `capture_responses` | `captures_requests()`, `captures_responses()` |
| 6 | `src/export.rs` | `ConversationImporter` | `options` | `options()` |
| 7 | `src/fallback.rs` | `HealthChecker` | `check_interval`, `running` | `check_interval()`, `is_running()` |
| 8 | `src/sse_streaming.rs` | `SseClient` | `retry_ms` | `retry_ms()` |
| 9 | `src/user_engagement.rs` | `EngagementTracker` | `user_id` + record fields | `user_id()`, `event_count()`, `time_since_last_event()`, `last_event_metadata()` |
| 10 | `src/websocket_streaming.rs` | `BidirectionalStream` | `incoming_buffer` | `incoming_buffer()` |
| 11 | `src/p2p.rs` | `P2PManager` | `pending_ice` | `pending_ice_count()` |
| 12 | `src/vector_db.rs` | `RedisVectorClient`, `ElasticsearchClient` | `config` | `config()` (×2) |

### Kept (justified)

| File | Item | Reason |
|------|------|--------|
| `src/constrained_decoding.rs` | `enum JsonContext` | State machine enum — variants defined for completeness, used via `context_stack: Vec<JsonContext>` |

---

## Verification

- `cargo check --features full,...` — clean, 0 warnings
- `cargo test --features full,...` — 5,963 passed, 0 failed
- `cargo clippy --features full,...` — 0 warnings
