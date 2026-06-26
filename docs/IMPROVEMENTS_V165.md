# IMPROVEMENTS_V165 — fix `server-axum` + `eval-suite` compile break

**Version:** 0.2.116 → 0.2.117
**Scope:** `src/server_axum.rs` + `.github/workflows/ci.yml`
**Feature:** none new

## Why

V164's dead-code sweep surfaced a **pre-existing** compile break: building
`server-axum` together with `eval-suite` failed because the MCP-tool
registration in `server_axum.rs` referenced a type that no longer
exists:

```rust
// before — does not compile
let generator = std::sync::Arc::new(std::sync::Mutex::new(
    crate::eval_suite::EvalGenerator::new(),   // no such type
));
crate::eval_suite::register_eval_tools(&mut mcp, generator);
```

`register_eval_tools` actually takes a **generator closure**
`Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>` (it produces a
model response for an eval prompt) — not an `Arc<Mutex<EvalGenerator>>`.
The combo was never built by CI, so the break went unnoticed.

## What changed

### The fix (`server_axum.rs`)

Wire the generator to the configured provider, consistent with the other
MCP backends registered in the same function (which use a default config):

```rust
let generator: std::sync::Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync> =
    std::sync::Arc::new(|prompt: &str| {
        let config = crate::AiConfig::default();
        let messages = [crate::ChatMessage::user(prompt)];
        crate::providers::generate_response(&config, &messages, "")
            .map_err(|e| e.to_string())
    });
crate::eval_suite::register_eval_tools(&mut mcp, generator);
```

Behaviour: each eval prompt is answered by the default-configured
provider; provider errors surface as the closure's `Err(String)`. (The
existing unit tests for `register_eval_tools` use a mock closure — this is
the production wiring.)

### Regression guard (`ci.yml`)

Added `"server-axum,eval-suite"` to the CI feature matrix (next to the
other `server-axum,*` combos), so this exact combo is now compiled +
tested on every push and can't silently re-break.

## Tests

Verified: `cargo check --features "server-axum,eval-suite"` (lib + bins)
clean — the previously-broken combo now builds; rustfmt clean. The change
is isolated to the `#[cfg(all(server-axum, eval-suite))]` block, so builds
without `eval-suite` (e.g. `FEATURES_NETWORK`) are unaffected.
