# V138 — HTTP fetcher in-crate + RefreshPolicy (0.2.85)

**Status**: DRAFT — pending V137 close
**Scope**: añadir descargador HTTP dentro del crate, eliminando el
"left to the caller" actual. Política de refresh configurable.
**No-goals**: extender el schema (eso es V137), recommender (V140).

## Por qué

El docstring actual de `models_dev` dice literalmente *"the actual HTTP
fetch is left to the caller"*. Esto viola tu regla
`feedback_library_framing`: la librería se configura, no se completa
fuera. V138 cierra esa contradicción.

## API propuesta

```rust
pub enum RefreshPolicy {
    Never,              // sólo lee caché; never fetches
    OnMiss,             // fetch sólo si no hay caché
    OnStale,            // fetch si caché > TTL (default)
    Background {        // refresh periódico en task tokio
        interval: Duration,
        on_error: BackoffPolicy,
    },
}

pub struct ModelsDevFetcher {
    cfg: ModelsDevConfig,
    client: Arc<dyn AsyncHttpClient>,
    policy: RefreshPolicy,
}

impl ModelsDevFetcher {
    pub fn new(cfg: ModelsDevConfig, client: Arc<dyn AsyncHttpClient>) -> Self;
    pub fn with_policy(mut self, policy: RefreshPolicy) -> Self;

    /// Returns registry — uses caché if fresh, fetches if stale,
    /// according to policy.
    pub async fn registry(&self) -> Result<ModelRegistry, ModelsDevError>;

    /// Force refresh now, ignoring policy.
    pub async fn force_refresh(&self) -> Result<(), ModelsDevError>;

    /// Start background refresh task (cancellable via returned handle).
    pub fn start_background(self: &Arc<Self>) -> BackgroundHandle;
}
```

## Reuso de infraestructura existente

- `src/http_client.rs` ya define `trait HttpClient` (sync, `UreqClient`).
  Añadir hermano async: `trait AsyncHttpClient` + impl con `reqwest`.
  Reqwest ya está en deps (lo usan otros providers).
- `src/huggingface_connector.rs` ya habla con HF — patrón a seguir
  para auth opcional y rate-limit handling.
- `models_dev.rs` ya tiene `save_cache`/`load_cache` atómicos con TTL
  — reutilizar tal cual; el fetcher sólo añade la otra mitad.

## Fuentes implementadas en V138

1. **models.dev** — endpoint `https://models.dev/api.json` (JSON
   estático, sin auth). Pull completo cada refresh.
2. **HuggingFace Hub** — `huggingface_connector` extendido con
   búsqueda paginada por filtros:
   - `family` (p.ej. `meta-llama/Llama-3.1-8B`)
   - `task` (text-generation, text-classification, …)
   - `tag` (quantized, gguf, lora, abliterated)
   - `sort` (downloads, trending)
   - Sólo se piden los top-N filtered por defecto; expansión on-demand.
   - ETag / `If-Modified-Since` para updates incrementales.
3. **Ollama library** — opcional; endpoint `https://ollama.com/library`
   parsed como HTML (no hay API oficial JSON). Detrás de sub-feature
   `catalog-ollama`.

`curated_models.rs` sigue siendo el fallback offline.

## Feature flag

`models-dev-fetcher` — opt-in. Implica `dep:reqwest` (ya en deps) y
`dep:tokio` (ya en deps). Detrás del flag para que crates que sólo
quieran el parser/cache de V137 no arrastren async runtime si no usan
fetch.

## Políticas de refresh — decisiones de diseño

- **OnStale** es default (≈ `cache_ttl` actual de 24 h).
- **Background** usa `tokio::spawn` con `CancellationToken`. Errores
  silenciosos no aceptados — se loguean vía `tracing::warn!` y, si la
  feature `metrics` está activa, se emiten como métrica
  `models_dev.refresh.failure`.
- **BackoffPolicy** = exponencial con jitter, cap 1 h. Tras N fallos
  consecutivos (default 5) pasa a estado *degraded* — sigue sirviendo
  caché vieja, pero la query expone `Registry::is_stale()`.

## Tests

- Mock async client devolviendo JSON canned → round-trip parse + cache
  write.
- TTL boundary tests (caché justo a TTL, justo después).
- Background task — cancellation test, error backoff test (mock client
  devolviendo 500 N veces).
- ETag handling — mock servidor responde 304 → no parse, no write.
- Payload bomb — fetcher respeta `max_payload_bytes` (4 MiB default).
- Concurrent registry() calls coalescen en un único fetch (no
  thundering herd).

## Riesgos / vectores de ataque (preview para V143)

- **SSRF** — usuario o config malicioso podría apuntar `cfg.endpoint`
  a `http://localhost:6379` (Redis) etc. → V143 introduce allowlist de
  hosts.
- **TLS bypass** — nada en V138 desactiva verificación. Si en futuro
  alguien añade `accept_invalid_certs(true)` para "testing", debe ir
  detrás de cfg-flag específico.
- **Cache poisoning offline** — si el atacante escribe en
  `cfg.cache_path`, sirve catálogo trucado. V143 evalúa firmar caché
  con HMAC + clave local.

## Iteraciones del plan

- **iter 1**: trait sync + fetcher sync. Rechazado: background refresh
  no encaja con `ureq`.
- **iter 2 (actual)**: trait async (`AsyncHttpClient` nuevo, sin tocar
  `HttpClient` sync existente), reqwest, tokio task para background.

## Out of scope para V138

- Hardware detection (V139).
- Cualquier wiring a Butler u otros consumers (V140).
- Catálogo firmado / TLS pinning (V143).
