# V144 — Wiring V140 (model recommender) into Butler, HTTP server, GUI (0.2.91)

**Status**: SHIPPED 2026-05-27 (0.2.91). Closes the wiring contract from
V140's "shipped as library entry point; integration callers next" and
V143-008's `/hardware` endpoint follow-up.
**Scope**: surface the V139 hardware probe and V140 model recommender
through every realistic caller boundary (Butler facade, HTTP server,
setup GUI), without inflating the public API.
**No-goals**: GUI side panel for tweaking advisor prompts (V140 advisor
is in-process Rust, no UI knobs needed yet); MCP tool exposure (no
existing Butler MCP surface to extend in this crate).

## Por qué

V140 landed `model_recommender::recommend(...)` as a free function. The
project's wiring checklist (`memory/feedback_wiring_checklist.md`)
demands that new modules be reachable from API + server + GUI + CLI.
V140 already covered CLI (`ai_setup recommend-model`, `ai_setup
hardware`). V144 closes the remaining four surfaces.

## Cambios

### 1. `Butler::recommend_model` delegate

Thin facade in `src/butler.rs`:

```rust
#[cfg(feature = "model-recommender")]
impl Butler {
    pub fn recommend_model(
        &self,
        req: &RecommendationRequest,
        registry: &ModelRegistry,
        hw: &HardwareInfo,
        advisor: Option<&dyn LlmEnhancer>,
    ) -> Result<Recommendation, RecommendError> { ... }
}
```

Stateless — does not consult `self.detectors`/`self.cache`. Kept on
Butler so callers that already hold a `Butler` don't need to know
about the lower-level module. Mirrors the existing
`recommend_runtime` / `recommend_prompt_fragments` pattern.

### 2. HTTP endpoints in `src/server.rs`

* `GET /hardware` and `GET /api/v1/hardware`
  (cfg: `hardware-detection`) — returns `HardwareInfo` JSON.
  **Auth-gated** by default: `/hardware` is NOT in
  `ServerAuthConfig::exempt_paths`, so when API-key auth is enabled
  the endpoint requires it. Closes the V143-008 contract.
* `POST /recommend-model` and `POST /api/v1/recommend-model`
  (cfg: `model-recommender`) — accepts
  `{ "request": <RecommendationRequest>, "registry_path": "..." | null }`.
  When `registry_path` is null uses `ModelRegistry::default()`.
  Returns the same `Recommendation` JSON the CLI emits.

Both handlers reuse `crate::hardware_info::detect_cached()` so the
probe cost is paid once per process.

### 3. `ai_setup_gui` Hardware tab

New `Tab::Hardware` between Models and Backup. Renders:

* "Probe host" button → `pretty_summary()` in a monospace group.
* Combo boxes for task / tier / privacy + "Recommend" button →
  formatted recommendation block.

Required-features bumped: `gui` → `gui, hardware-detection,
model-recommender`. These are already in the `full` feature set so
this is not a new dependency surface — just a tighter declaration.

## Verification

* `cargo build --lib`: clean (default features).
* `cargo build --lib --features model-recommender,models-dev-fetcher`:
  clean.
* `cargo clippy --lib --features model-recommender,models-dev-fetcher
  -- -D warnings`: clean.
* `cargo build --bin ai_setup_gui --features gui,hardware-detection,
  model-recommender`: clean (pre-existing lib warnings not introduced
  here).
* `cargo test --lib --features model-recommender,models-dev-fetcher`:
  **6300 passed, 0 failed** (was 6297 → +3 new):
  * `server::tests::test_hardware_route_returns_json`
  * `server::tests::test_recommend_model_route_empty_registry_rejects`
  * `server::tests::test_recommend_model_route_malformed_json_rejects`

## Decisiones de implementación

* **`/hardware` is auth-gated by default, not feature-gated off**.
  V143-008 said "MUST live behind authentication (at minimum an opt-in
  flag, ideally RBAC-gated)". The server already has `exempt_paths`
  — leaving `/hardware` out of it means auth applies whenever the
  operator enables it. We do not add a hard kill switch because that
  would shift the burden onto every operator who *does* want the
  endpoint.

* **POST not GET for `/recommend-model`**. The recommender takes a
  structured request body (task, tier, privacy, size cap, hint).
  Encoding that as query params would be ugly and would leak the
  optional `user_hint` into access logs. POST keeps the contract
  clean.

* **MCP tool deferred**. The library has no existing Butler MCP tool
  surface — exposing recommend-model via MCP would mean a new tool
  group (`butler_*`) with its own registration plumbing. That's a
  separate concern; the HTTP endpoint already gives external callers
  a path.

* **GUI tab is sync, not background**. The recommender is rule-based
  + optional LLM call. The rule-based path completes in milliseconds;
  the GUI does not need a worker thread for it. If the advisor LLM is
  wired in via a future config, that path can be moved to
  `bg_tx`/`bg_rx` then.
