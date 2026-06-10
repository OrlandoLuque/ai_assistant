# IMPROVEMENTS_V151 — Zero-warnings sweep + the 3 bugs the warnings were hiding

**Version:** 0.2.101 → 0.2.102
**Scope:** transversal — 18 archivos, sin features nuevos
**Trigger:** política zero-warnings de CLAUDE.md; clippy en CI es
warn-level (`-W clippy::all`), así que 36 warnings se habían ido
acumulando sin romper builds.

## La tesis

Un warning de `unused_variable` o `dead_code` no es ruido: es el
compilador señalando que el código *dice* que hace algo que en
realidad no hace. De los 36 warnings barridos, 3 eran exactamente
eso — features a medio cablear que parecían funcionar:

## Bug 1 — Streaming SSE descartaba el `system_prompt` del cliente

`src/server_axum.rs`, ambos paths de streaming (el endpoint nativo
`/chat/stream` y la rama stream del handler OpenAI-compat).

- **Síntoma del lint:** `unused variable: system_prompt` y
  `unused variable: stream_sys_prompt`.
- **Realidad:** los handlers extraían el `system_prompt` del request
  y llamaban a `send_message_cancellable(message, &knowledge)` — que
  no tiene slot para system prompt. El path NO-streaming sí lo pasa
  (`send_message_with_notes(message, &knowledge, &system_prompt, "")`,
  donde viaja por el slot `session_notes`).
- **Efecto observable:** un cliente que pidiera `stream: true` con
  `system_prompt` custom recibía respuestas generadas SIN su system
  prompt. El mismo request sin streaming lo respetaba. Inconsistencia
  silenciosa entre los dos modos.
- **Fix:** ambos paths llaman ahora a
  `send_message_cancellable_with_notes(message, &knowledge,
  &system_prompt, "")`, espejando el path no-streaming.

## Bug 2 — Hinted handoffs encolados pero jamás entregados

`src/distributed_network.rs`.

- **Síntoma del lint:** `methods select_best_peers and
  drain_handoffs_for_peer are never used`.
- **Realidad:** el replication pass encola `HintedHandoff`s cuando
  un peer está caído o el send falla (líneas ~2120/2131). El método
  que los entrega al reconectar (`drain_handoffs_for_peer`) existía,
  con su lógica completa… y ningún caller. La cola de handoffs solo
  podía crecer (bounded a 1000 entries + TTL 1h, así que no era leak
  infinito, pero las réplicas diferidas nunca llegaban).
- **Fix:** `drain_handoffs_for_peer(&peer_id)` cableado en los dos
  sitios que emiten `PeerConnected`: `connect_to_peer` (saliente) y
  `handle_incoming` (entrante).
- **`select_best_peers`** (selección por reputación): sin consumidor
  natural hoy — eliminado. Git lo preserva; volverá con la capa de
  trust del mesh design si hace falta.

## Bug 3 — El tiebreaker FIFO del AgentPool no existía

`src/agent_wiring.rs`.

- **Síntoma del lint:** `field sequence_counter is never read`.
- **Realidad:** el campo estaba documentado como "Sequence counter
  for FIFO tiebreaker in priority queue", pero el `Ord` de `PoolTask`
  solo comparaba `priority`. `BinaryHeap` no garantiza orden entre
  claves iguales → dos tareas con la misma prioridad podían salir en
  cualquier orden.
- **Fix:** wrapper privado `QueuedPoolTask { task, seq }` como entry
  del heap, ordenado por `(priority desc, seq asc)`. `submit_task`
  asigna `seq` desde el counter (con `wrapping_add`). `PoolTask` no
  cambia (API pública intacta). Test de regresión:
  `test_pool_equal_priority_dequeues_fifo`.

## Limpieza mecánica (sin cambio de comportamiento)

| Categoría | Detalle |
|---|---|
| `static mut` → atomics | `ai_test_harness`: 7 flags CLI migradas a `AtomicBool`/`AtomicU64` (f64 como bits)/`OnceLock<String>`. Cero `unsafe` en el plumbing de flags. |
| `result_large_err` | 3 helpers de `ai_proxy` con `Result<_, Response>`: `#[allow]` justificado — el Err solo se materializa en rechazos y boxear cascadearía por el hot path de forwarding. |
| Deprecated in-crate | `AutoApproveAll`: `#[allow(deprecated)]` scoped en su propio impl, el re-export de lib.rs y el import de agent_wiring (la deprecación apunta a callers externos; el wiring interno es deliberado). |
| Dead code eliminado | `MfccSpeakerVerifier.num_mel_bands` (el pipeline simplificado no tiene filterbank mel), `VoiceAnonymizer.read_pos` (el resampler reinicia por frame), `autonomous_loop.planning_hint_idx` (cleanup nunca implementado), `CategoryResult::total`. |
| Diagnóstico mejorado | `group_queue_host`: el log de eviction por heartbeat ahora incluye nombre y addr del cliente (los campos existían sin leerse). |
| Unreachable arms | `emotion_detection` (×2) y `browser_policy`: brazos `_` inalcanzables en matches in-crate sobre enums `#[non_exhaustive]` (el atributo no aplica al crate definidor). |
| FFI Win32 | `BOOL`/`DWORD` en `ai_virtual_mic_host`: `#[allow(clippy::upper_case_acronyms)]` scoped — nombres de la API de Windows verbatim. |
| `--fix` automático | `&PathBuf`→`&Path`, `contains()`, `io::Error::other`, clones de tipos `Copy`, `vec![x; 0]`→`Vec::new()`, import `Query` gated a `distributed-network`. |

## Verificación

- `cargo clippy --lib --bins` con el feature set completo de CI
  (`FEATURES_STD`): **0 warnings**.
- `cargo clippy --features server-axum --lib --bins`: **0 warnings**.
- `cargo test --lib` (feature set completo): **8.445 passed, 0 failed**.
- `cargo test --features server-axum --bin ai_proxy`: **107 passed**.
- `ai_test_harness --list` y `--filter=...` funcionan tras la
  migración de flags.

## Follow-ups

- CI corre clippy con `-W clippy::all` — considerar `-D warnings`
  ahora que la base está limpia, para que la deuda no se reacumule.
  (Decisión para V152+: requiere confirmar que las 36 combinaciones
  de features del matrix también están limpias, no solo FEATURES_STD.)
