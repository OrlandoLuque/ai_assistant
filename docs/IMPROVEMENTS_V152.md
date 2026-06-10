# IMPROVEMENTS_V152 — Batería completa de tests: 7 bugs reales

**Version:** 0.2.102 → 0.2.103
**Trigger:** orden del autor — batería de pruebas completa de todas
las funcionalidades vía CLI. Herramienta: `ai_test_harness --all`
(585 tests, 131 categorías).
**Resultado:** 9 fallos → 0. De los 9, 7 eran bugs de código/feature
graph y 2 eran tests desactualizados.

## Los dos críticos: feature-graph breaks silenciosos

### 1. AES-256-GCM roto en builds `full`/`rag`

`Cargo.toml` declaraba:

```toml
rag = ["rusqlite", "dep:aes-gcm"]     # ← MAL
aes-gcm = ["dep:aes-gcm"]
```

Con namespaced features, `dep:aes-gcm` habilita la **dependencia**
pero NO el **feature** homónimo. Todos los gates
`cfg(feature = "aes-gcm")` de `content_encryption.rs` quedaban
apagados: cualquier `ContentEncryptor` con `Aes256Gcm` o
`ChaCha20Poly1305` devolvía `EncryptionFailed`.

**Por qué CI nunca lo vio:** los tests lib de AES están detrás del
mismo `cfg`, así que ni siquiera compilaban. La cadena completa:
feature mal referenciado → código gated off → tests gated off → CI
verde. El harness no está cfg-gateado y lo cazó a la primera.

**Mitigante:** el diseño fail-loud (declinar en vez de degradar a
XOR) evitó que esto fuera una vulnerabilidad de cifrado débil
silencioso. Los módulos `secure_backup`/`encrypted_knowledge` usan
el dep directamente (sin cfg granular) y nunca estuvieron afectados.

**Fix:** `rag = ["rusqlite", "aes-gcm"]`.

### 2. PDF parsing roto en builds `documents`

Mismo patrón exacto: `documents = ["dep:zip", "dep:pdf-extract"]`
nunca encendía `cfg(feature = "pdf-extract")` en
`document_parsing/parser.rs`. Fix análogo.

**Sweep preventivo:** script sobre Cargo.toml buscando
`"dep:X"` donde existe feature `X` + código fuente usa
`cfg(feature = "X")`. Solo estos dos casos.

## Panic: PII redaction con matches solapados

`PiiDetector::detect` ordenaba detecciones por posición descendente
y aplicaba `replace_range` secuencialmente. Con spans solapados (el
patrón de teléfono matcheando dígitos dentro de un número de
tarjeta), el segundo `replace_range` usaba índices del string
original sobre el string ya mutado → out-of-bounds panic
(`is_char_boundary`). Peor: podía dejar PII parcial sin redactar.

**Fix:** resolución de solapes pre-redacción — preferencia por
confianza, luego longitud de span, luego posición. Los índices de
los spans elegidos (disjuntos, aplicados de atrás hacia delante)
siempre son válidos.

## Calidad: tres heurísticas bajo umbral

| Heurística | Score antes | Fix | Score después |
|---|---|---|---|
| Moderación (recall harmful) | 0.125 | Capa de patrones harmful-instruction + categoría `Illicit` + Weapons/Drugs/Fraud/Illicit en el default set | 1.0 |
| Intent classification | 0.55 | Scoring sin normalizar por tamaño del set + bonus posicional; "please" fuera de Request; verbos comunes añadidos | ≥0.9 |
| Token estimation | 0.80 | ASCII: palabras×1.3 + puntuación×0.5, floor chars/4.5; no-ASCII: bytes/3.5 | 1.0 |

Notas de diseño:

- **Moderación**: los patrones nuevos clavan el *framing* how-to +
  sustantivo de daño, no los sustantivos solos ("the bomb squad
  arrived" no flaggea). El patrón de evasión de detección va a 0.65
  (bajo el umbral 0.7): señal de riesgo, nunca bloquea solo.
- **Intent**: la normalización por nº de patrones penalizaba añadir
  sinónimos (más cobertura → intent menos probable). Conteo bruto +
  0.5 de bonus si el patrón ancla al inicio del mensaje. Confianza =
  cuota relativa del total de evidencia.
- **Tokens**: bytes/3.5 sobreestimaba inglés ~30% y subestimaba
  código. La fórmula nueva queda más cerca de BPE real en los tres
  regímenes (prosa, código, no-ASCII). Cambio de comportamiento en
  presupuestos de contexto: estimaciones ~10% más bajas para prosa
  inglesa — sigue siendo conservadora frente a tokenizers reales.

## Consistencia: chunking empaquetaba a max en vez de target

`chunk_by_sentences` y `chunk_by_paragraphs` acumulaban unidades
hasta `max_tokens`; `chunk_fixed_size` usa `target_tokens`. Con
target=15/max=40, el caller recibía chunks de ~40 tokens (2.5× lo
pedido). Ambas estrategias empaquetan ahora hacia `target_tokens`;
`max_tokens` queda como trigger de "esta unidad sola es demasiado
grande, subdivídela".

## Tests desactualizados (el código tenía razón)

1. **Guardrail panicking**: el harness esperaba fail-open (skip del
   guard roto). El pipeline hace fail-closed deliberado desde el
   hardening — un guard que panickea con input crafteado no puede
   convertirse en bypass. Test actualizado para asertar fail-closed.
2. **EntityType::all**: esperaba 7; son 9 desde que V81-V88 añadió
   Paper + Author.

## Verificación final

- `ai_test_harness --all`: **585/585** (2 skipped: requieren Ollama
  vivo; 1 slow marcado).
- `cargo test --lib` con FEATURES_STD: **8.448 passed, 0 failed**
  (+3 vs antes: los tests AES ahora compilan).
- clippy: **0 warnings**. `ai_proxy`: 107/107.

## Follow-ups

- Considerar ejecutar `ai_test_harness --all` en CI (job opcional,
  ~35s) para que la cobertura no-cfg-gateada vigile el feature graph.
- El sweep `dep:` vs feature debería ser un test/lint permanente
  (regla: si existe `cfg(feature="X")` en src y un feature `X`,
  ningún otro feature debe referenciar `dep:X` directamente).
