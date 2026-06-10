# Security Audit — V152 (2026-06-11)

**Alcance:** subsistemas tocados/recientes — `ai_proxy` (V149 routing +
V150 streaming), crypto (`content_encryption`, `secure_backup`,
`encrypted_knowledge`), PII (`pii_detection`), moderación
(`content_moderation`) — más `cargo audit` (RUSTSEC) transversal.

**Metodología:** dos auditores en paralelo sobre la superficie de
ataque enumerada, con veredicto por vector (SOLID / RISK / NEEDS-REVIEW)
citando código. Estilo V143 (audit-as-delivery-artifact).

**Resultado:** 1 bug explotable encontrado y corregido (panic DoS en
masking de PII). Todo lo demás SOLID. `cargo audit` limpio tras añadir
1 supresión justificada.

---

## 1. cargo audit (RUSTSEC)

4 advisories, todas en dependencias **transitivas**, ninguna con
vulnerabilidad explotable directa:

| ID | Crate | Tipo | Veredicto |
|---|---|---|---|
| RUSTSEC-2025-0141 | bincode 1.x | unmaintained | Suprimido (ya existía). Re-check 2026-08-01. |
| RUSTSEC-2024-0436 | paste | unmaintained | Suprimido (ya existía). Transitive vía egui. |
| RUSTSEC-2025-0134 | rustls-pemfile | unmaintained | Suprimido (ya existía). Transitive vía reqwest. |
| **RUSTSEC-2026-0002** | **lru 0.12.5** | **unsound (IterMut)** | **Nuevo. Suprimido en V152.** |

**RUSTSEC-2026-0002 (lru unsound IterMut):** Stacked Borrows
violation en `IterMut`. **Puramente transitivo**: `tantivy` →
`lance`/`lancedb`; verificado por grep que NO hay uso directo de `lru`
en este crate. La unsoundness requiere un patrón de llamada específico
a `IterMut` dentro de los internals de tantivy; no alcanzable desde
nuestra superficie de API. Suprimido en `deny.toml`, `ci.yml` y
`supply-chain.yml` (las 3 listas sincronizadas, verificado por el job
`audit-deny-sync`). Re-check: 2026-09-01 o cuando lancedb/tantivy
suban lru ≥ 0.16.

---

## 2. ai_proxy — V149 routing + V150 streaming

Seis vectores auditados. **Todos SOLID.**

### 2.1 Request-ID dedupe DoS — SOLID
Cache acotada a 10.000 entries con evicción FIFO
(`ai_proxy.rs:238-242`), TTL 300s con expiry perezoso en lookup
(`:228-234`), longitud de request-id validada ≤128 bytes
(`:282-290`). Clave `siphash(api_key) ⊕ domain_sep ⊕ siphash(req_id)`
(`:250-256`) con separador de dominio. Seguro para deployment
single-key (el proxy usa un único `api_key` global).

### 2.2 Forward-hops loop guard — SOLID
`parse_forward_hops` (`:314-322`) colapsa a 0 cualquier input
negativo/no-numérico/fuera de rango. `saturating_add` evita overflow
(`:331`). El header inbound se **sobrescribe siempre** con el valor del
proxy (`:2355`) — un cliente no puede forjar `x-forward-hops` para
saltarse el límite. Cubierto por tests F3 (negative, garbage,
excessive).

### 2.3 Streaming per-chunk timeout (V150) — SOLID
`tokio::time::timeout` por chunk (`:419-459`), default 30s
configurable. Abort inmediato en timeout → slow-loris no puede pinear
recursos. Sin buffering interno: `reqwest::bytes_stream()` piped
directo a `axum::Body::from_stream()`. Errores upstream incrementan
métrica y cierran limpio sin leak de internals.

### 2.4 Header injection / topology leak — SOLID
`inject_served_by` preserva el `x-mesh-served-by` del backend si ya
existe (`:373-374`, diseño multi-hop documentado). Modo opaco
(`expose_served_by_addr=false`) hashea addr con salt
(`:354-365`), por defecto salt random per-proceso. **Nota de
despliegue:** un backend malicioso puede inyectar headers que el proxy
reenvía (passthrough intencionado — el proxy no es boundary de
headers de respuesta); mitigar con modo opaco o backends de confianza.

### 2.5 /v1/models agregado — SOLID
Auth-first: rechaza con 401 si falta bearer válido (`:2070-2081`).
Solo GET (`:2082-2093`). Solo lista backends sanos (`:2128`). Cache
TTL 60s invalidada en transiciones de salud. Cliente no autenticado
NO puede enumerar topología.

### 2.6 SSRF / backend control — SOLID
Backend elegido por política de routing (`pick_by_policy`), NO por
contenido del request. URL = `addr_fijo_del_backend + path_inbound`
normalizado por axum. Cuerpo y query passthrough sin mutación. Sin
vector de inyección de URL/path.

---

## 3. Crypto — content_encryption / secure_backup / encrypted_knowledge

**Todo SOLID.**

- **Nonce AES-256-GCM**: `OsRng.fill_bytes()` (CSPRNG del OS), fresco
  por cada cifrado, en los tres módulos. Sin reuse.
- **Fallback nonce** (sin feature `aes-gcm`): hash de timestamp+thread+
  stack-ptr — **no es debilidad** porque en ese modo AES/ChaCha se
  rechaza por completo (fail-loud), solo queda XOR.
- **Enforcement de clave**: rechaza `key.len() != 32` en cifrado y
  descifrado.
- **Fail-loud**: NUNCA degrada AES-256-GCM a XOR silenciosamente —
  devuelve `EncryptionFailed`. (Este es justamente el gate que el bug
  de feature-graph de V152 dejó apagado; ahora reparado.)
- **Tamper detection**: tag AEAD de AES-GCM; descifrado falla atómico
  si se altera 1 byte (test `test_aes256gcm_tamper_detection`).
- **Key rotation**: claves viejas permanecen para descifrar legacy;
  `remove_expired_keys` limpia bajo demanda.
- **KDF**: HKDF-SHA256 con salt random per-archive en `secure_backup`;
  documentado que passphrases débiles siguen débiles (no es
  password-stretching, por diseño).

---

## 4. PII — pii_detection

### 4.1 Resolución de matches solapados — SOLID (fix de V152)
El fix de V152 (spans disjuntos pre-redacción, aplicados de atrás
hacia delante) es **correcto**: la condición
`d.end <= c.start || d.start >= c.end` (`:360`) garantiza disjunción;
el sort descendente (`:367`) mantiene índices válidos durante la
mutación.

### 4.2 **mask_value panic UTF-8 — RISK (Medium) → CORREGIDO en V152**
**Bug encontrado por el audit.** `mask_value` (`:556`) usaba
`value.len()` (bytes) y slicing por bytes `&value[..show]`. Con PII
conteniendo UTF-8 multibyte (acentos, emoji) — habitual en nombres y
emails — el corte podía caer en mitad de un carácter →
**panic `is_char_boundary` → DoS**. No es brecha de
confidencialidad/integridad pero sí denial-of-service desde input
controlado por el atacante.

**Fix:** reescrito sobre `char`s (`value.chars().collect()`),
indexando por carácter. Test de regresión
`test_mask_value_multibyte_no_panic` con cadenas
`"tök-Zürich🏔️café"`, `"tok-日本語テスト"`, `"tök-é"`.

### 4.3 Validación SSN / estrategias de redacción — SOLID
Rechaza rangos inválidos (area 000/666/9xx, group 00, serial 0000).
Replace/Hash/Remove sin issues; Mask ahora char-safe.

---

## 5. Moderación — content_moderation

**ReDoS — SOLID.** `add_pattern` fija `size_limit` y `dfa_size_limit`
a 1MB (`:274`) y rechaza patrones que excedan. Las 14 reglas nuevas de
V152 (harmful-instruction) usan cuantificadores **acotados**
(`{0,2}`, `{0,3}`) y **perezosos** (`*?`, `+?`) — sin backtracking
catastrófico. Ningún patrón puede colgar o panickear con input
adversarial.

---

## Resumen de hallazgos

| # | Subsistema | Hallazgo | Severidad | Estado |
|---|---|---|---|---|
| 1 | pii_detection | Panic UTF-8 en `mask_value` (DoS) | Medium | **Corregido V152** |
| 2 | cargo audit | lru unsound IterMut (transitivo) | Low | Suprimido + justificado |
| — | ai_proxy (6 vectores) | — | — | SOLID |
| — | crypto (3 módulos) | — | — | SOLID |
| — | moderación ReDoS | — | — | SOLID |

**Confianza:** alta para los subsistemas auditados. No se auditó en
profundidad: RBAC, guardrails de streaming, browser_policy, sandbox
del agente autónomo, distributed_network QUIC/TLS (fuera del alcance
de "subsistemas recientes"; candidatos a un audit dedicado futuro).

## Follow-ups

- Audit dedicado de `distributed_network` (QUIC/TLS 1.3, join tokens,
  hinted handoff que V151 acaba de cablear).
- Considerar un lint/test que rechace slicing por bytes sobre `&str`
  en paths que procesan input externo (la clase de bug de `mask_value`).
- El passthrough de headers de respuesta del proxy: documentar en la
  guía de despliegue que requiere backends de confianza o modo opaco.
