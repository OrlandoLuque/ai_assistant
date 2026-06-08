# V149 — Routing Hygiene + Model-Aware Routing

**Version:** 0.2.99
**Date:** 2026-06-08
**Scope:** `src/bin/ai_proxy.rs` + `src/server_axum.rs`
**Feature flag:** `server-axum` (and `security` for the hardened
gateway path). No new feature flags.

## Why

V78 introduced the `ai_proxy` binary as a thin load-balancer in
front of a pool of equivalent local backends. V146/V147/V148 were
maintenance patches. Re-reading `core/docs/MESH_DESIGN.md` while
closing V148 surfaced five gestures already approved in the design
doc that the implementation had never picked up — all foundation
work that turns the proxy from "n-way round-robin" into a real
federation primitive:

1. **Identity on every response.** Today the client has no idea
   which backend served a given request — debugging multi-host
   deployments means hand-correlating timing across nodes.
2. **Idempotency / replay protection.** Network retries today either
   double-charge token budgets or duplicate side effects.
3. **Loop guard.** No protection if two proxies ever point at each
   other (zero today, foundation for trusted multi-hop tomorrow).
4. **Model awareness.** Backends were treated as equivalent — the
   moment a deployment runs different models per host, routing
   silently goes wrong.
5. **Capability discovery.** Callers need to ask each backend
   individually what models it serves; no `/v1/models` aggregation.

V149 ships all five. V150 (streaming passthrough) is a follow-up
patch because the hot-path change carries different risk.

## What ships

V149 is split into four sub-phases that landed together. The phase
numbering matches the approved plan
(`ai_assistant_plans/V149_routing_hygiene.md`), which kept F2 as the
streaming work and renamed the rest F1, F3, F4, F5 — preserving
intent rather than renumbering.

### F1 — `x-mesh-served-by` + OpenAI error envelope

**Header.** Every response from `ai_proxy` (free path + security
gateway) now carries `x-mesh-served-by: <id>`. The id is one of:

- The literal `host:port` of the backend that served the request
  (default).
- A 12-hex-char SipHash digest of `addr + salt` (opaque mode), if
  `expose_served_by_addr = false`.

The header is also injected on responses that never reached a
backend (auth failure, rate limit, input guard, etc.) — in that case
the id identifies the proxy itself, so a stack trace through a
forwarding chain is always readable end-to-end.

If a backend already set `x-mesh-served-by` (it served as a proxy
itself), the proxy preserves the value rather than overwriting —
multi-hop trails stay intact, the closest backend's identity wins.

Configuration via `[routing]` in the TOML config:

```toml
[routing]
expose_served_by_addr = false           # default true
served_by_salt = "stable-salt-string"   # default: random per-process
```

**Envelope.** All `ai_proxy` errors now use the OpenAI canonical
shape so any OpenAI-compatible client can parse them uniformly:

```json
{
  "error": {
    "message": "Rate limit exceeded, retry in 5s",
    "type":    "rate_limit_error",
    "code":    "rate_limit_exceeded",
    "param":   null
  }
}
```

Five canonical types:
`invalid_request_error`, `authentication_error`, `rate_limit_error`,
`not_found_error`, `service_unavailable_error`, `server_error`.

The `x-mesh-served-by` injection is implemented as an axum
middleware in `server_axum.rs` so any axum-based entry point that
mounts the middleware participates in the V149 contract uniformly.

### F3 — Request-id dedupe + `max_forward_hops`

**Dedupe.** Replays of the same `x-request-id` from the same caller
are rejected with `409 Conflict` (OpenAI envelope) so a client
network retry never produces a duplicate charge or duplicate side
effect.

Properties:
- LRU bounded at 10k entries with a 5-min sliding TTL.
- Key: `(siphash(api_key), siphash(request_id))`. Cross-tenant
  collisions are impossible — two tenants sending the same
  `x-request-id` get fully independent dedupe windows.
- Only POST/PUT/PATCH/DELETE are deduped. GET/HEAD bypass — they're
  idempotent by definition.
- `len(x-request-id) > 128` → 400 (envelope). Defends against
  unbounded memory growth from a malicious client sending huge ids.

**Loop guard.** `x-forward-hops` is incremented on each forward.
Exceeding `routing.max_forward_hops` (default 8) returns `508 Loop
Detected` (envelope).

Properties:
- Strict parse: negative or non-numeric inbound values are treated
  as 0. A malicious client cannot fake a low value.
- Inbound from outside the mesh (header absent) starts at 0.
  Foundation for future trusted multi-hop chains — when a signed
  identity layer lands, only signed peers will be allowed to forward
  with `x-forward-hops > 0`.
- Configurable:
  ```toml
  [routing]
  max_forward_hops = 4
  ```

### F4 — Backend model registry + routing policies

**Registry.** Each backend now tracks:
- `static_models`: declared in `[[backends]].models` in TOML.
- `advertised_models`: scraped from the backend's `/v1/models`,
  refreshed by the health-check loop.

`known_models()` returns the deduped sorted union. The registry is
populated permissively — entries that don't parse are silently
skipped, supports both OpenAI shape (`{"data":[{"id":"..."}]}`) and
Ollama shape (`{"models":[{"name":"..."}]}`).

If the `/v1/models` scrape fails, the backend keeps its previous
advertised-model list. Non-2xx from `/v1/models` does NOT mark the
backend unhealthy — that responsibility stays with `/health` alone.
Failed scrapes feed an exponential backoff (capped at 30 ticks ≈ 5
minutes) so a backend that doesn't speak `/v1/models` doesn't spam
the network.

**Policies.** `RoutingPolicy`:

| Policy | Behavior | Use case |
|---|---|---|
| `round_robin` (default) | Even fanout across healthy backends, model-agnostic. | Identical backends. |
| `local_first` | Walks backends in config order, first healthy wins. Model-agnostic. | One primary + warm spares. |
| `model_aware` | Restricts candidates to backends advertising the requested model. Round-robins among them. | Heterogeneous pool. |

Selection:
```bash
ai_proxy --routing-policy model_aware
```

```toml
[routing]
policy = "model_aware"
```

`model_aware` always overrides session affinity (a sticky session
shouldn't defeat model routing). For all other policies, sticky
sessions win first as before.

When `model_aware` is active but no backend advertises the
requested model, the response is `404 Not Found` with
`code: "model_not_in_mesh"` — distinguishable from a backend's own
404 for an unknown model.

Validation: enabling `model_aware` without declaring `models` on any
backend and without `enable_model_polling = true` emits a startup
warning and auto-enables polling. Better to log+continue than to fail
boot.

**Metrics.** New `GET /metrics` endpoint, Prometheus text format
(no client lib dep):

```
# TYPE proxy_requests_by_policy counter
proxy_requests_by_policy{policy="round_robin"} 1234
proxy_requests_by_policy{policy="local_first"} 0
proxy_requests_by_policy{policy="model_aware"} 0
proxy_loop_detected_total 7
proxy_dedupe_hit_total 19
proxy_model_aware_no_match_total 0
```

**`/health` extended.** Now reports `models_advertised: Vec<String>`
per backend so dashboards can render the federation's model
topology without hitting `/v1/models`.

### F5 — Aggregated `/v1/models`

New endpoint: `GET /v1/models`. Returns the union of all backend
models with the OpenAI list shape, plus a `served_by` array per
entry pointing to every backend that advertises the model.

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3",
      "object": "model",
      "created": 0,
      "served_by": ["10.0.0.5:11434", "10.0.0.6:11434"]
    }
  ]
}
```

Properties:
- 60s TTL cache. Invalidated on any backend health transition AND
  on any change to a backend's advertised-model list — flapping
  meshes surface fresh data without waiting on the TTL.
- Respects the `api_key` auth gate. Unauth → 401 envelope. Never
  discloses model topology to anonymous callers.
- GET only. POST/PUT/etc. → 405 envelope with `Allow: GET`.
- `served_by` values honor `expose_served_by_addr` — opaque mode
  hides addrs.
- Unhealthy backends are excluded from the list. A model with all
  hosts down disappears from `/v1/models` until any host returns.

## Configuration reference

Full new `[routing]` section in `examples/ai_proxy.toml`:

```toml
[routing]
# One of: round_robin (default) | local_first | model_aware
policy = "round_robin"

# Whether x-mesh-served-by exposes the literal addr (true, default)
# or a 12-hex-char opaque id (false).
expose_served_by_addr = true

# Salt for the opaque id when expose_served_by_addr = false.
# Unset = random per-process salt. Set this if you need stable
# opaque ids across restarts.
# served_by_salt = "..."

# Hard ceiling on x-forward-hops. Default 8. Exceed → 508.
max_forward_hops = 8

# Force-enable /v1/models polling regardless of policy.
# Auto-enabled if policy = "model_aware".
# enable_model_polling = true
```

CLI:

```
--routing-policy <round_robin|local_first|model_aware>
```

CLI always wins over config file (consistent with all other proxy
flags).

## Tests

Total: 116 tests for `ai_proxy` (up from 73 at V78). Breakdown:

- F1: 5 (envelope shape, header injection, opaque mode, preserve
  backend value, self-served-by on early rejections)
- F3: 8 (replay POST → 409, GET bypass, oversize id → 400,
  cross-tenant isolation, loop guard, strict parse of garbage hops,
  negative hops, outbound increment)
- F4: 21 (policy parse, model extract, OpenAI/Ollama shape parsers,
  malformed-entry tolerance, all three policies + fallback paths,
  metrics endpoint + counter wiring, health endpoint model listing,
  backoff cap)
- F5: 6 (union with served_by, method not allowed, api_key gate,
  cache hit within TTL, invalidation on health transition, opaque
  mode in served_by)
- Plus 76 regression / pre-existing tests from V78–V148.

Run:

```bash
cargo test --bin ai_proxy --features server-axum,security
```

## Backwards compatibility

All defaults preserve V78 behavior. A V78 config and command line
keep working as-is: no `[routing]` section needed, no flags
required. The new endpoints (`/metrics`, `/v1/models`) are
purely additive — existing clients won't hit them unless they ask.

The OpenAI envelope migration changes error response *shape* but
not status codes. Any client that parsed by status code is
unaffected. Any client that parsed an ad-hoc body is now reading a
standards-conforming OpenAI envelope.

## Follow-ups (deliberately deferred)

1. **Model disambiguation across same-named different-sized
   models** — by backend id in the routing key.
2. **Mark model-unhealthy per pair** when a backend lies about
   advertising a model but 404s on the actual request.
3. **Persistent dedupe across restart** — currently in-memory.
4. **Composite policy** (`local_first` + `model_aware`).
5. **Multi-hop trusted peer chain** with Ed25519-signed identities.
   F3's hop guard is the foundation; this layer adds the trust
   model on top.
6. **Membership tri-mode** (`private` / `hybrid` / `open`) — needs
   signed identity layer first.
7. **NAT-aware mesh participation** — use the `p2p.rs` (STUN/UPnP)
   layer so a home-deployed proxy can join the mesh from behind a
   NAT.

## V150 — Streaming passthrough (next patch)

The buffering behavior in `proxy_forward_handler` and
`gateway_chat_handler` is unchanged in V149. Both still read the
entire backend response into memory before forwarding — which means
SSE streams are end-to-end broken today.

V150 will swap `resp.bytes().await` for
`reqwest::Response::bytes_stream()` + `axum::body::Body::from_stream()`,
with a per-chunk timeout to defend against slow-loris backends. It
ships as a separate patch because the hot-path change carries
different risk than the additive surface in V149.

## Design decisions worth recording

- **SipHash, not blake3, for opaque ids and dedupe keys.** The
  `std::hash::DefaultHasher` is SipHash-1-3 — fine for non-crypto
  collision resistance, and it sidesteps the `sha2` / `blake3`
  feature flag thicket. The opaque id is a debugging affordance,
  not a security token.
- **`parking_lot` over `std`.** Already used elsewhere in
  `ai_proxy.rs` (`cache::ResponseCache`), no new dep.
- **No `prometheus` crate.** The `/metrics` body is hand-rendered
  Prometheus text. Adding a dep to expose six counters is wildly
  out of scale.
- **`model_aware` overrides session affinity.** Conscious choice.
  A sticky session pinning to a backend that can't serve the
  requested model is a worse failure mode than breaking affinity
  for one request.
- **F3 dedupe is per-tenant by api_key hash.** Falls back to
  `("anonymous", request_id)` when no API key is configured. The
  fall-back collision risk is real but acceptable: callers in
  unauthenticated deployments are typically a single trusted
  consumer.
- **F5 cache invalidates on poll deltas, not just health
  transitions.** A backend that quietly drops a model from its
  `/v1/models` list (model unload) needs to vanish from the
  federation view immediately, even though `/health` stays green.

## Files touched

- `src/bin/ai_proxy.rs` — F1+F3+F4+F5 (~+1500 LoC including tests).
- `src/server_axum.rs` — `x-mesh-served-by` middleware.
- `examples/ai_proxy.toml` — `[routing]` section documenting all
  new config knobs.
- `Cargo.toml` — 0.2.98 → 0.2.99.
- `CHANGELOG.md` — V149 entry.
- `docs/IMPROVEMENTS_V149.md` — this file.
- `../ai_assistant-website/concepts.html` — V149 card.

## Known gaps (deferred deliberately)

- **`embedded_server.rs` no emite `x-mesh-served-by` con node identity.**
  El plan original lo contemplaba pero se aplazó: hoy el backend ya
  inyecta el header cuando el cliente lo manda en el request — la
  necesidad de un valor by-default desde el nodo backend es defensiva
  pero no funcional para V149. Tratado como follow-up V149.2 opcional.
- **`mock_llama_server.rs` emite un único modelo estático.** Los tests
  F4/F5 usan harness `gateway_e2e` directo (no mock backend), por lo
  que extender el mock no es bloqueante. V150 sí lo requerirá para
  tests SSE.
- **V150 — streaming passthrough.** Patch separado por riesgo hot-path.
  Plan en `ai_assistant_plans/V150_streaming_passthrough.md`.
