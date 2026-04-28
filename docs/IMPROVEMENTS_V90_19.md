# V90.19 — Vision wiring across persistence, agents, FFI, plugins, embeddings

**Version**: 0.2.38 → 0.2.49
**Feature flag**: `vision` (additive — no new flag)
**Date**: 2026-04-28

## Scope

V90.18 closed the dispatcher path so a caller in possession of a
`VisionMessage` could hit any vision-capable provider. V90.19 wires the
*surrounding* surface so an agent (not just the CLI) can be the caller:
the canonical `ChatMessage`, conversation persistence, agent definitions,
plugins, FFI, and the embedding stack now all carry image attachments
end-to-end.

## Change set (batches from `vision_wiring_full_plan.md`)

| Batch | File(s) | What |
|------:|---------|------|
| 16 | `embedding_providers.rs`, `lib.rs` | `VisionEmbeddingProvider` trait, `LocalHashImageEmbedding` (FNV-1a fallback when sha2 not available), `create_vision_embedding_provider("local-hash")` factory. Re-exported from crate root under `cfg(all(feature = "embeddings", feature = "vision"))`. |
| 51 | `messages.rs` | `ChatMessage.images: Vec<ImageInput>` cfg-gated, `with_image` / `with_images` builders, `has_images()`. |
| 53 | `agent_definition.rs`, `agent_graph.rs`, `agent_wiring.rs` | `AgentSpec.accepts_images`, `AgentSpec.max_images_per_request`, `AgentNode.accepts_images` + `with_image_support()`. |
| 56 | `ffi.rs` | `ai_assistant_send_message_with_image` — validates bytes via `ImagePreprocessor`, falls back to magic-byte detection when `media_type` is null, **dispatches through `vision::generate_vision_response`** (no longer text-only fallback). |
| 57 | `plugins.rs` | `PluginCapability::Vision` variant (additive — relies on `#[non_exhaustive]`). |
| 65 | `conversation_snapshot.rs` | `SnapshotMessage.images: Vec<ImageRef>` (sha256-keyed for snapshot transport). |
| 66 | `export.rs`, `bin/ai_test_harness.rs` | `ExportedMessage.images` carried through Markdown/HTML/JSON exporters. Test harness call sites updated. |
| 67 | `conversation_compaction.rs`, `context_composer.rs`, `bin/ai_test_harness.rs` | `CompactableMessage.images` (both variants). 16 test sites bulk-updated. |
| 69 | `rag.rs` | `StoredMessage.images`; SQLite `query_map` closures preserve attachments on read. |
| — | `model_integration.rs`, `ui_hooks.rs`, `wasm_hooks.rs` | All three parallel `ChatMessage` types in flight surfaces gain `.images`. |

## Critical correctness fix — bincode + cfg-gated `Vec`

The first iteration of these fields used:

```rust
#[serde(default, skip_serializing_if = "Vec::is_empty")]
pub images: Vec<...>,
```

This deserialised cleanly under JSON but **mis-aligned positional offsets
in bincode** (the binary-storage format used by sessions / snapshots /
RAG). Symptom: 4 round-trip tests in `assistant::tests::*` failed with
"Failed to deserialize binary data (tried bincode and JSON)".

Resolution: removed `skip_serializing_if` from every cfg-gated `Vec`
field added in this batch. `#[serde(default)]` only — empty vecs are
serialised as `len=0`, keeping the byte layout stable.

The decision is documented in-source on `messages.rs::ChatMessage` and
`conversation_snapshot.rs::SnapshotMessage` so future contributors do
not reintroduce the attribute.

## FFI vision — real wiring

Before V90.19, `ai_assistant_send_message_with_image` validated the image
bytes and then silently degraded to a text-only `generate_sync` call:

```rust
// Until the per-call image-aware send path is wired through every
// provider, fall back to text-only send so the FFI surface is
// stable; image bytes have been validated above.
let result = a.generate_sync(msg, "");
```

Now it builds a `VisionMessage::user(prompt, vec![image])` and calls
`vision::generate_vision_response(&a.config, &[vmsg], &system_prompt)`.
Bytes still pass `ImagePreprocessor::validate_bytes` first, so size /
magic-byte / animation rejection happens before any provider dispatch.

## What is still pending in the wiring plan

The 82-batch plan continues. Remaining categories (none of which break
the surface added here):

- Real `ImagePreprocessor::process` decoding (Batch 11) — needs the
  `image` crate; deferred to keep dep graph small.
- MCP `vision_query` / `vision_capabilities` tools (Batch 7).
- Azure deployment-aware vision routing (Batch 8) and Bedrock vision
  feature (Batch 9).
- `vision-audit` binary + Prometheus histograms (Batches 10, 42).
- Cost subsystem `with_per_image_limit` (Batch 12) and safety pipeline
  OCR-backed scanning (Batch 13).
- Streaming vision skeleton (Batch 22).
- Public API root: `AiResponse::Image` variant (Batch 76),
  `BatchRequest` (Batch 77), `prompt_chaining::ChainStep` (Batch 78),
  `model_ensemble::Ensemble::execute` (Batch 79),
  `regeneration::RegenerationRequest` (Batch 80).
- `widgets::chat_input` paste/drop (Batch 81).

## Tests

Full lib suite passes with the canonical feature combo:

```text
cargo test --features vision,security,advanced-memory,embeddings,multi-agent,rag,distributed,autonomous,research --lib
test result: ok. 6417 passed; 0 failed
```

The single previously-failing `test_save_and_load_sessions` and three
sibling round-trip tests now pass after the bincode fix above.

## Cross-batch invariants honoured

- **Zero `.unwrap()` in production** — all new error paths use
  `set_last_error` (FFI) or typed `anyhow::Result` (FFI dispatcher,
  embedding factory).
- **Zero new compiler warnings** introduced by these batches on
  `cargo check --features vision,full`.
- **Vision feature transparent** when disabled — every new field /
  variant is `#[cfg(feature = "vision")]` and has a non-vision build path.
- **Patch-level bumps** within 0.2.x — 0.2.48 → 0.2.49.
