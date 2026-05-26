# V143 — Security audit del flujo de catálogo + recommender (0.2.90)

**Status**: SHIPPED 2026-05-26 (0.2.90) — 10 findings analysed, 2
fixed (V143-001 SSRF, V143-002 prompt-injection wrapper escape), 2
documented as accepted (V143-003 catalog signing, V143-008
`/hardware` endpoint privacy), 6 confirmed non-exploitable. Audit
report: [`docs/SECURITY_AUDIT_V143.md`](./SECURITY_AUDIT_V143.md).
**Scope**: revisar atak surface real (no especulativa) introducido
por V137-V142. Cerrar gaps con código + tests + documentación.
**No-goals**: redesign de subsistemas — los gaps que requieran
redesign se escalan a Vxxx siguiente.

## Por qué

Pediste auditoría de seguridad explícita: GAPS, vectores de ataque,
posibles mejoras. V143 se ejecuta cuando V137-V142 están cerradas
porque audita código real, no diseño.

## Áreas a auditar

### 1. Fetcher HTTP (V138)

- **SSRF**: ¿se valida que `endpoint` apunte a un host externo y no a
  IPs privadas/loopback/link-local/UNIX sockets? — *expected gap*,
  añadir allowlist por defecto + opt-out.
- **TLS verification**: ¿algún path desactiva verify? `reqwest`
  default es seguro, comprobar que no haya `danger_*` flags.
- **Redirección abierta**: ¿el cliente sigue redirects a cualquier
  host? Limitar a host original o allowlist.
- **Auth tokens** (HF): ¿se loguean tokens en error paths? Redact.
- **Rate-limit en background refresh**: backoff exponencial correcto,
  cap razonable.

### 2. Catálogo (V137 + V138)

- **Catalog poisoning**: si el endpoint sirve JSON manipulado (e.g.
  `min_vram_bytes: 0` en un modelo que necesita 80 GB), el
  recommender lo promociona. *Mit propuesta*: catálogo firmado
  (Ed25519) con clave pública pinned in-crate; opt-out via flag.
- **JSON bomb**: `max_payload_bytes` (4 MiB) ya está; verificar que
  se aplica antes de parsear (no después).
- **Schema drift attack**: campo desconocido con nombre engañoso (e.g.
  `__proto__` no aplica a Rust pero `extra_data: <huge>` sí podría
  inflar memoria). Confirmar que `serde(deny_unknown_fields)` o
  filtrado explícito están.
- **Cache poisoning local**: si atacante escribe en `cache_path`, el
  proceso siguiente sirve catálogo trucado. *Mit*: HMAC con clave
  derivada de seed local (no protege contra atacante con acceso al
  proceso, sí contra atacantes que escriben sólo en disco).

### 3. Hardware detection (V139)

- **Driver crash**: `nvml-wrapper` puede colgar — verificar
  `spawn_blocking` + timeout.
- **Privacy leak**: si `ai_serve` expone `/hardware`, fingerprint del
  host queda accesible. *Mit*: endpoint detrás de auth, o por defecto
  desactivado.
- **Shell escape**: `system_profiler` / `rocm-smi` se invocan con
  argumentos. ¿Hay forma de inyectar? Verificar que no se interpola
  user input.

### 4. Butler recommender (V140)

- **Prompt injection en `user_hint`**: ya identificado en V140; aquí
  se verifica que sanitizer actúa antes del LLM call.
- **LLM-driven exfiltration**: el advisor recibe hardware info en su
  prompt → ¿podría un prompt inyectado hacerle "decir" la VRAM al
  caller? Sí, eso es el output. *Mit*: documentar como expected.
- **Recommendation tampering**: ¿el LLM puede devolver un
  `family_id`/`variant_id` que no está en la lista prefiltrada?
  Validar contra catálogo antes de devolver al usuario.
- **Resource exhaustion**: bucle de recomendaciones LLM-driven
  consume API. Cache + rate-limit.

### 5. Wasmtime 44 (V141)

- **Sandbox escape**: revisar config: `consume_fuel`, memory limits,
  CPU limits. ¿Hay paths que carguen sin limits aplicados?
- **WASI capabilities**: verificar que sólo se permite lo
  estrictamente necesario (no filesystem, no network, no env).
- **Trap on unbounded**: módulo malicioso con bucle infinito → trap
  test en suite.

### 6. Transversal

- **Logs**: revisar que no se loguean URLs con tokens en query
  string, ni payloads JSON enteros.
- **Métricas**: nombres no contengan PII ni IDs de modelo
  abliterated en cardinalidad alta.
- **Concurrencia**: refresh background + lookup concurrente — race
  en `OnceLock<HardwareInfo>` o en `Mutex<ModelRegistry>`?

## Entregables

- Cada gap encontrado entra como entrada en `docs/SECURITY_AUDIT_V143.md`
  con: descripción, severidad (CVSS-light), reproducción, fix
  aplicado.
- Tests nuevos cubriendo cada vector confirmado.
- Update de `docs/CONCEPTS.md` con cards sobre catálogo firmado,
  hardware probe sandboxing, etc.
- Concept cards #304+ en `ai_assistant-website/concepts.html` (mirror).

## Estructura del audit

Plantilla por vector:

```markdown
### V143-001 — SSRF en ModelsDevFetcher.endpoint

**Severity**: Medium (network egress to internal host).
**Affects**: V138 onwards.
**Repro**: `cfg.endpoint = "http://169.254.169.254/latest/meta-data"`
sin allowlist activa.
**Fix**: añadido `EndpointAllowlist` default = `[
  "models.dev", "huggingface.co", "ollama.com" ]`. Override explícito
en config para tests.
**Test**: `tests/security/test_ssrf_blocked.rs`.
```

## Iteraciones

- **iter 1**: audit en checklist exhaustivo. Rechazado: scope sin
  límites se vuelve teatro.
- **iter 2 (actual)**: 6 áreas concretas, plantilla por hallazgo,
  triage de severidad antes de fix.

## Out of scope para V143

- Pen-test externo o bug-bounty (separado, fuera del proyecto).
- Auditoría de las 50+ módulos no tocados por V137-V142 (esos siguen
  cubiertos por la revisión normal de PRs).
- Cambios al threat model global (escalado a documento aparte si
  surge).

## Verification (shipped 2026-05-26)

- `cargo build --lib`: clean (default features).
- `cargo build --lib --features model-recommender,models-dev-fetcher`:
  clean.
- `cargo clippy --lib --features model-recommender,models-dev-fetcher
  -- -D warnings`: clean.
- `cargo test --lib`: 6297 passed, 0 failed (was 6284 → +13 new for
  V143). Pre-existing flaky `http_client::test_mock_server_post_streaming`
  failed in parallel run but passed in isolation — unrelated to V143.
- Audit deliverable: [`docs/SECURITY_AUDIT_V143.md`](./SECURITY_AUDIT_V143.md).

## Decisiones de implementación (no en el plan original)

- **SSRF defense alcance: literal-IP + `localhost` only**: la versión
  exhaustiva que protege contra DNS-rebinding requiere un connector
  reqwest custom con post-resolution check. Es scope mayor y se
  marca como conocido en el audit doc (V143-001). La versión actual
  cierra los foot-guns obvios (AWS metadata, RFC 1918, IPv6 ULA/LL).
- **Sanitización `user_hint`: replace, no reject**: rechazar el hint
  cuando contiene `>>>`/`<<<` daría falsos positivos (usuarios que
  legítimamente usan triple-bracket). Reemplazo por glyphs visualmente
  similares (›››/‹‹‹) mantiene legibilidad para el LLM y bloquea el
  escape del wrapper.
- **Catalog signing (V143-003) diferido a V144+**: requiere identidad
  de publicador (¿quién firma?) y key management — no soluble en
  scope de audit interno. Defensas existentes (TLS + payload cap +
  post-filter por HardwareInfo medida) mitigan parcialmente.
- **`deny_unknown_fields` no añadido (V143-004)**: rompería
  forward-compat con cambios de schema en models.dev. El cap de
  payload de 4 MiB ya es la defensa correcta para JSON bomb.
- **No se añadieron tests E2E de WASM malicioso (V143-009)**: los
  hooks para esa suite viven en V128's `tests/skill_forge_*`; V141
  no regresionó nada que justifique recrearlos aquí.
- **`SECURITY_AUDIT_V143.md` en `docs/`, no en `docs/runbooks/`**:
  es un audit report (informe), no un runbook (procedimiento
  operacional). Sigue el formato de los `IMPROVEMENTS_V*.md`
  shipped: status + scope + findings + verification + decisions.
