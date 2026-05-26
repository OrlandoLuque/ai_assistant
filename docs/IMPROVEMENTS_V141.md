# V141 — wasmtime 36 → 41 (0.2.88)

**Status**: SHIPPED 2026-05-26 (0.2.88) — wasmtime 36 → 41 (NO 44 —
ver "Decisiones"). API de `src/skill_forge/wasm.rs` no requirió
cambios. Bonus: fixed long-standing `StepKind` collision in lib.rs
that broke `--features skill-forge` builds.
**Scope**: migrar el backend WASM de skill_forge de wasmtime 36 a 44.
**No-goals**: añadir nuevas capabilities WASI, cambiar el modelo de
fuel/epoch.

## Por qué

36.0.9 cerró RUSTSEC-2026-0114. No hay urgencia de seguridad. Sí hay
deuda técnica: 8 mayors entre 36 y 44 traen mejoras de Cranelift,
arranque más rápido y APIs que se han ido decantando. Mantenerse en
36 indefinidamente hace que la próxima RUSTSEC contra 36.x sea más
dolorosa de saltar.

## Estado actual

- `src/skill_forge/wasm.rs` (357 líneas) — único archivo del crate
  que importa `wasmtime`.
- APIs en uso: `Engine`, `Config`, `Module`, `Store`, `Linker`,
  `ResourceLimiter`, `Store::limiter`.
- Detrás de feature `skill-forge`.

## Plan de migración

1. **Bump Cargo.toml** `wasmtime = "36"` → `"44"`,
   `wasmtime-wasi = "36"` → `"44"`.
2. **Build con `--features skill-forge`** — diagnosticar errores de
   compilación; documentar cada diff de API.
3. **Cambios API esperados** (no exhaustivo, lista a confirmar al
   ejecutar):
   - `Engine::new` mantiene firma pero algunos `Config` flags se han
     deprecated o renombrado.
   - `ResourceLimiter` ha cambiado signature en alguna versión —
     puede requerir `&mut self` distinto.
   - `Store::limiter` puede haber pasado de closure-by-ref a
     trait-based.
   - `Linker::define_unknown_imports_as_traps` o equivalente para
     traps.
4. **Actualizar tests** de `skill_forge::wasm`.
5. **Re-medir** memoria y fuel — los defaults de wasmtime cambian
   entre mayors; documentar cualquier ajuste en bench_budget.toml.

## Verificación

```bash
cargo build --features skill-forge --release
cargo test --features skill-forge --lib -- skill_forge::wasm::
cargo clippy --features skill-forge --release -- -D warnings
cargo deny check
```

Si el bench `skill_forge_wasm_*` (si existe) se desvía, recalibrar.

## Riesgos

- **Regresión silenciosa** en sandboxing (fuel semantics, memory
  growth limits). *Mit*: tests dedicados que carguen un módulo
  malicioso (loop infinito, malloc bomb) y verifiquen trap.
- **Build-time pesado** — wasmtime 44 tarda más en compilar. *Mit*:
  verificar tiempo en CI; añadir a budget si supera +20%.
- **Plataformas** — wasmtime 44 puede haber dropped soporte para
  triples antiguos. Verificar contra la CI matrix.

## Iteraciones

- **iter 1**: migrar de un salto. Posible si el diff de API es
  pequeño.
- **iter 2 (preferido)**: salto en 2 pasos — 36→40→44 — si la
  primera tanda de errores sugiere muchos breakings.

## Out of scope para V141

- Nuevas capabilities WASI (sockets, threads).
- Cambiar el modelo de seguridad de skill-forge.
- Soporte para WASM components (component-model GA en 22+ pero
  introducir = scope nuevo).

## Verification (shipped 2026-05-26)

- `cargo build --lib`: clean (default features, sin skill-forge).
- `cargo build --lib --features skill-forge`: clean.
- `cargo build --bin ai_setup --features full`: clean.
- `cargo build --bin ai_cli --features full`: clean.
- `cargo clippy --lib --features skill-forge -- -D warnings`: clean.
- `cargo test --lib`: **6284 passed, 0 failed** (default features).
- `cargo test --lib --features skill-forge skill_forge::wasm::`:
  **4 passed, 0 failed**.
- `cargo deny check`: skipped (cargo-deny no instalado localmente —
  cubierto por V142).

## Decisiones de implementación (no en el plan original)

- **41, no 44**: wasmtime 44 requiere rustc 1.92; el proyecto está
  pinned a 1.90.0 via `rust-toolchain.toml`. wasmtime 42 requiere
  1.91. wasmtime 41 es el último compatible con 1.90. La razón del
  bump original (alejarse de 36.x para que la próxima RUSTSEC sea
  menos dolorosa) se cumple igual con 41: 5 mayors de margen vs 0.
  El salto a 44 se hará cuando el toolchain suba (decisión separada,
  no en V141).
- **No hubo cambios de API en `wasm.rs`**: `Engine`, `Config`,
  `Module`, `Store`, `Linker`, `ResourceLimiter`, `set_fuel`,
  `set_epoch_deadline`, `increment_epoch`, `Store::limiter`,
  `consume_fuel`, `epoch_interruption`, `wasm_bulk_memory` siguen con
  la misma signature en 41. El plan listaba cambios "posibles" y
  ninguno se materializó en este rango de versiones.
- **`wasi-common` pinned a 36**: el crate `wasi-common` está
  deprecated upstream (no hay versiones 41+ en crates.io para él
  como crate independiente — está absorbido por `wasmtime-wasi`).
  Como no se importa en ningún archivo `.rs` del proyecto, se deja
  en 36 sin tocar. Limpieza completa (drop de la dep + del feature
  `dep:wasi-common`) se difiere para no inflar este PR.
- **Fix colateral: `StepKind` collision en `src/lib.rs`**: el build
  con `--features skill-forge` estaba roto en master (colisión entre
  `skill_forge::StepKind` y `recipes::StepKind`). Se renombra el
  re-export de `skill_forge` a `SkillStepKind`. Pre-existente; sólo
  salió a la luz al verificar V141. `recipes::StepKind` (usado por
  `ai_cli`) queda intacto.

## Out of scope (post-V141)

- Bump final a wasmtime 44+ (depende de subir rust-toolchain a 1.92).
- Drop completo de `wasi-common` del Cargo.toml + feature
  `skill-forge`.
