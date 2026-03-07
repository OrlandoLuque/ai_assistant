# v31 Improvements Changelog

> Date: 2026-03-06

v31 is the **Real Streaming + Virtual Models** release: all three streaming endpoints now forward actual LLM tokens in real-time, virtual models bundle enrichment configs as named API models, and the cluster broadcasts model catalogs via CRDTs for cross-node discovery.

---

## Summary Metrics

| Metric | v30 | v31 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | ~6,900 | ~7,060 | ~+160 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 0 | 0 |
| New source files | 0 | 1 | +1 |
| Lines added | — | ~2,200 | — |

---

## New Files

| File | LOC | Description |
|------|-----|-------------|
| `src/virtual_model.rs` | ~480 | Virtual model definitions, model registry, publish control |

---

## Phase 1: Real Streaming — SSE `/chat/stream`

- **Problem**: `chat_stream_handler` waited for `AiResponse::Complete`, then split by whitespace for fake token-by-token streaming.
- **Fix**: Introduced `StreamEvent` enum (`Token`/`Done`/`Error`) and `tokio::sync::mpsc` channel to bridge the blocking LLM thread with the async SSE handler. Each `AiResponse::Chunk` is forwarded immediately as an SSE event.
- **Format**: `data: {"token":"..."}` per chunk, `data: [DONE]` at end.
- Tests: 5 new

## Phase 2: Real Streaming — OpenAI `/v1/chat/completions?stream=true`

- **Problem**: Same fake streaming pattern — waited for full response, then split by whitespace into `chat.completion.chunk` SSE events.
- **Fix**: When `stream=true`, uses the same `StreamEvent` mpsc pattern. Input guardrails and RAG enrichment run before streaming starts. Each real token is emitted as an OpenAI-format `chat.completion.chunk`.
- **Format**: Role announcement chunk → content delta chunks → `finish_reason: "stop"` chunk → `[DONE]`.
- **Helpers extracted**: `apply_enrichment_config()` and `apply_rag_config()` — reusable across streaming and non-streaming paths.
- Non-streaming path is unchanged.
- Tests: 8 new

## Phase 3: Real Streaming — WebSocket `/ws`

- **Problem**: WebSocket handler waited for `AiResponse::Complete`, then sent a single JSON message with the full response.
- **Fix**: Same `StreamEvent` mpsc pattern. Each chunk is sent as `{"type":"chunk","content":"..."}`, followed by `{"type":"complete","model":"..."}` at the end.
- Tests: 5 new

## Phase 4: Virtual Models — Data Model + Registry

New module `src/virtual_model.rs`:

- **`VirtualModel`**: Bundles `EnrichmentConfig` + `ModelProfile` + system prompt + base model + provider as a named model. Fields: `name`, `description`, `base_model`, `base_provider`, `enrichment`, `profile`, `system_prompt`, `published`, `created_at`, `tags`.
- **`PublishedModel`**: Controls visibility of physical (local) models. Fields: `name`, `provider`, `published`, `display_name`.
- **`ModelRegistry`**: DashMap-based lock-free registry for virtual models and physical model publish control. Methods: `register_virtual`, `unregister_virtual`, `get_virtual`, `list_virtual`, `list_published_virtual`, `set_published`, `is_published`, `list_published_physical`, `resolve`, `list_client_visible`, `save_to_file`, `load_from_file`.
- **`ModelResolution`**: Enum returned by `resolve()` — `Virtual(VirtualModel)`, `Physical { name, provider }`, `PassThrough { name }`.
- **`ClientModel`**: OpenAI-compatible model info for `/v1/models` responses.
- **`RegistrySnapshot`**: JSON-serializable snapshot for persistence.
- Tests: 32

## Phase 5: Virtual Model Resolution in Handlers

- Added `model_registry: Arc<ModelRegistry>` to `AppState`.
- Added `model: Option<String>` field to `ChatRequest` (backward compatible).
- `openai_completions_handler`: resolves requested model via `ModelRegistry` before enrichment setup. Virtual model overrides enrichment config, system prompt, and profile.
- `openai_models_handler`: returns published models from registry via `list_client_visible()`. Falls back to raw `available_models` when registry is empty.
- Tests: 8 new

## Phase 6: Admin Endpoints

Eight new admin endpoints (under auth middleware):

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/virtual-models` | Create a virtual model |
| GET | `/admin/virtual-models` | List all virtual models (incl. unpublished) |
| GET | `/admin/virtual-models/{name}` | Get specific virtual model |
| PUT | `/admin/virtual-models/{name}` | Update a virtual model |
| DELETE | `/admin/virtual-models/{name}` | Delete a virtual model |
| GET | `/admin/models` | List all models with publish status |
| POST | `/admin/models/{name}/publish` | Publish a physical model |
| POST | `/admin/models/{name}/unpublish` | Unpublish a physical model |

- `CreateVirtualModelRequest`: JSON body with name, description, base_model, base_provider, enrichment, profile, system_prompt, published, tags.
- `PublishModelRequest`: JSON body with provider and display_name.
- Tests: 11 new

## Phase 7: Cluster Model Broadcasting

- **`ModelAdvertisement`**: Serializable struct broadcast via CRDT (`LWWMap<String, String>`) keyed by `"{node_id}:{model_name}"`. Fields: `node_id`, `model_name`, `model_type` ("physical"/"virtual"), `provider`, `description`, `published`.
- **`ClusterState::model_catalog`**: New `Arc<RwLock<LWWMap<String, String>>>` field.
- **`ClusterManager` methods**: `advertise_model()`, `withdraw_model()`, `list_published_models()`, `list_peer_models()`, `find_model()`, `model_catalog()`.
- **CRDT sync loop**: Model catalog entries are serialized and stored in `NetworkNode` for peer access during each sync cycle.
- **Persistence loop**: Model catalog is included in periodic snapshots.
- **`ClusterDebugInfo`**: New `published_model_count` field.
- LWW conflict resolution ensures the latest publish/unpublish wins across nodes.
- Tests: 10 new

---

## Key Design Decisions

1. **`StreamEvent` enum + mpsc channel**: Clean bridge between blocking provider thread and async handler. Reusable across all 3 streaming endpoints.
2. **Output guardrails skipped during streaming**: Industry standard (OpenAI, Anthropic, etc.). Input guardrails still apply before stream starts.
3. **`ModelRegistry` uses `DashMap`**: Lock-free, consistent with existing `AppState` pattern.
4. **Virtual models stored as JSON file**: Simple persistence via `save_to_file`/`load_from_file`, no DB dependency.
5. **Cluster broadcasting via existing CRDTs**: Reuses `LWWMap` infrastructure in `ClusterState`, no new dependencies.
6. **Physical models unpublished by default**: Admin must explicitly publish. Prevents accidental exposure.
7. **`model` field added to `ChatRequest`**: `Option<String>`, backward compatible — existing clients work unchanged.
8. **`openai_models_handler` fallback**: When no models are published in registry, returns raw `available_models` (existing behavior preserved).
