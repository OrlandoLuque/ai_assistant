# Security Audit — V155 (2026-06-11)

**Alcance:** los subsistemas que V153 dejó fuera por estar fuera del
área "reciente" — `distributed_network` (QUIC/TLS mesh) + `node_security`,
el sandbox del agente autónomo (`agent_policy`/`agent_sandbox`/`inspector`),
y `browser_policy` (automatización CDP).

**Metodología:** dos auditores en paralelo por superficie, veredicto por
vector citando código. **Cada hallazgo de los agentes se verificó a mano
contra el código** antes de actuar — dos de los "RISK HIGH" reportados
resultaron mal analizados (ver Falsos positivos).

**Resultado:** 3 bugs reales corregidos (1 SSRF, 1 timing leak, 1
self-DoS) + 1 feature a medio cablear arreglada. Documentadas 2
limitaciones de diseño con su modelo de confianza.

---

## Bugs corregidos

### 1. SSRF — bypass de la comprobación de IP privada vía userinfo · MEDIUM
`browser_policy.rs::extract_host`. La función partía la URL pero no
quitaba el `userinfo`. Para `https://attacker.com@192.168.1.1/`
devolvía `attacker.com@192.168.1.1`, que **no parsea como IP**, así que
el gate de IP privada (y el de metadata endpoints) se saltaba — y el
navegador iría al host real tras el `@` (`192.168.1.1`). SSRF a la red
interna / `169.254.169.254`.

**Fix:** `extract_host` ahora toma el host tras el **último** `@`
(`rsplit_once('@')`) y maneja literales IPv6 entre corchetes
(`[::1]:443`). 5 tests de regresión nuevos
(`test_userinfo_cannot_bypass_private_ip_check`): userinfo+IP privada,
user:pass@loopback, userinfo@metadata, `[::1]`, y un userinfo legítimo
sobre host público que sigue permitido.

### 2. Timing leak en la comparación del join token · MEDIUM
`distributed_network.rs` líneas 1158 y 1421. El token de membresía del
cluster se comparaba con `t.token == token` (String `==`, no
constant-time) — fuga de timing que permite fuerza bruta
byte-a-byte del secreto. Irónico: `constant_time_eq` **ya existía**
en `node_security.rs:39` y ya se usaba para el challenge-response
(línea 519), pero no para el token.

**Fix:** ambos sitios usan `constant_time_eq(t.token.as_bytes(),
token.as_bytes())`. La función pasó de privada a `pub(crate)`.

### 3. Self-DoS — hinted handoffs nunca expiraban · MEDIUM
`distributed_network.rs`. `HintedHandoffQueue::expire_old()` estaba
definida y testeada pero **sin caller**. La cola acotada (cap 1000) se
llenaba de entradas caducas para peers que nunca volvían y dejaba de
aceptar handoffs frescos — denegación de servicio lenta contra la
propia replicación. (Relacionado con el bug que V151 ya corrigió:
`drain_handoffs_for_peer` también estaba sin cablear.)

**Fix:** `cleanup_expired` (ciclo de cleanup del event loop, cada 30s)
ahora pasa a `&mut self` y llama a `handoff_queue.expire_old()`.

### 4. Feature a medio cablear — `min_level` ignorado · Low
`server_axum.rs::get_trace_handler`. El endpoint
`GET /v1/logs/traces/{id}` aceptaba un query param `min_level` que el
handler **ignoraba** (misma clase de bug que los de V151). Cableado:
nuevo `export_trace_filtered` en `distributed_log.rs` que filtra por
nivel; el handler parsea `min_level` y lo pasa.

---

## Limitaciones de diseño documentadas (no son bugs, son el modelo)

### A. Filtro de JS del browser es defense-in-depth, NO un boundary
`browser_policy.rs::validate_js`. Los `contains_*_pattern` son
substring matching sobre código JS — bypassables por cualquier
adversario decidido (`window['fe'+'tch']`, `Function(...)`, escapes
unicode, `atob`). **No es un bug**: captura los casos obvios. El error
sería *confiar* en ellos. Añadido un doc-comment de SECURITY MODEL
explícito: para input no confiable el boundary real es el navegador
(`JsPermission::Disabled`, contexto sandboxed, o CSP restrictivo).

### B. Command deny-list: limitación con comandos encadenados
`agent_policy.rs::can_run_command`. El allow es por palabra base
(`base == allowed`); el deny es por substring (`cmd.contains(denied)`,
que **sobre-bloquea**, falla cerrado). Un comando encadenado tras una
base permitida (`ls ; foo`, con `foo` no en deny-list) pasaría. Es una
limitación conocida del modelo "allow por palabra base", mitigada por
la **segunda capa**: el `Inspector` (V123) escanea contenido antes del
sandbox. Reescribir a parsing de shell real es un cambio de diseño con
su propio riesgo de romper comandos legítimos — fuera del alcance de
este audit quirúrgico, registrado como follow-up.

### C. Modelo de confianza del mesh (peers autenticados)
`distributed_network`. Tras el TLS mutuo (verificado SOLID — sin
verifiers peligrosos, mTLS con CA por cluster), el mesh **confía en los
peers autenticados**: pueden almacenar datos sin cuota (storage
unbounded), y el `NodeId` se toma del mensaje Ping, no del certificado.
Un atacante *no autenticado* no puede entrar (el muro TLS lo para
primero). Cuotas por peer y binding NodeId↔cert son hardening para
entornos de peers no confiables — registrado como follow-up.

---

## Falsos positivos del audit (verificados y descartados)

- **"Command deny substring = RISK HIGH bypass"**: el agente razonó que
  `echo x; rm` sería *permitido*. Falso: `cmd.contains("rm")` lo
  **deniega** (sobre-bloqueo). Es fail-closed, no un bypass. Reclasificado
  a limitación menor (B).
- **"AutoApproveAll alcanzable por defecto"**: verificado SOLID — solo
  vía `AutonomyLevel::Autonomous` explícito, nunca default.

---

## Veredictos SOLID (sin cambio de código)

- **TLS/cert del mesh**: mTLS correcto, CA por cluster, sin
  `SkipServerVerification` ni accept-any-cert.
- **bincode wire**: cap de 16 MB antes de alocar (`read_message`) — sin
  OOM por length prefix forjado.
- **max_connections**: enforced antes de aceptar (default 50).
- **Ring poisoning**: `add_node` solo tras auth+token; no inyectable vía
  mensajes Put.
- **Path traversal del sandbox**: `Component::ParentDir` rechazado
  sintácticamente + `canonicalize()` resuelve symlinks.
- **SSRF (resto)**: scheme allowlist (bloquea file/data/javascript),
  rangos de IP privada RFC-correctos, metadata endpoints.

---

## Resumen

| # | Subsistema | Hallazgo | Severidad | Estado |
|---|---|---|---|---|
| 1 | browser_policy | SSRF via userinfo | Medium | **Corregido + 5 tests** |
| 2 | distributed_network | Timing leak join token | Medium | **Corregido** |
| 3 | distributed_network | Self-DoS handoffs sin expirar | Medium | **Corregido** |
| 4 | server_axum | `min_level` ignorado | Low | **Cableado** |
| A | browser_policy | Filtro JS bypassable | — | Documentado (defense-in-depth) |
| B | agent_policy | Command chain limitation | — | Documentado + follow-up |
| C | distributed_network | Trust model peers | — | Documentado + follow-up |

## Follow-ups registrados

- Parsing de shell real en `can_run_command` (vs substring/palabra base).
- Cuotas por peer + binding NodeId↔certificado en el mesh.
- Cap por target-node en la cola de hinted handoff (hoy global 1000).
