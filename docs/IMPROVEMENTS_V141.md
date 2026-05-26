# V141 — wasmtime 36 → 44 (0.2.88)

**Status**: DRAFT — independiente, no bloquea ni se bloquea por otras fases
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
