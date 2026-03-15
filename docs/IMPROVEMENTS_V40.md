# V40 — Testing & Debugging Infrastructure

**Tesis**: Mejora sistemática de la infraestructura de testing: harness enriquecido con
puntuaciones, tests de precisión scored, integración CLI de los comandos de test/bench,
y detección automática de regresiones contra baselines.

**Estado**: HECHO
**Fecha**: 2026-03-15
**LOC nuevas**: ~600 (harness core + precision tests + CLI + regression)

---

## Resumen de cambios

### Fase 1: Mejoras del núcleo del Harness

Extensión de las estructuras centrales del test harness para soportar puntuaciones,
filtrado y modos de salida avanzados.

**Structs extendidos**:

| Struct | Campos/Métodos añadidos |
|--------|------------------------|
| `TestResult` | `score: Option<f64>`, `details: Vec<String>`, `skipped: bool`, `slow: bool` |
| `CategoryResult` | Métodos `skipped()`, `slow()`, `total_active()` |
| `HarnessReport` | `total_skipped`; todas las structs ahora derivan `Deserialize` |

**Nuevas funciones**:
- `run_test_scored(name, threshold, f)` — ejecuta un test con puntuación numérica y
  umbral de aprobado/suspenso
- `run_test()` ahora consulta `should_run(name)` para filtrado y marca tests lentos
- `print_summary()` mejorado: modo verbose (detalle por test), modo summary-only,
  ordenación por duración, visualización de score/slow, conteo de skips

**Nuevos flags CLI del harness** (9 flags):

| Flag | Descripción |
|------|-------------|
| `--verbose` / `-v` | Muestra detalle por test (duración, score, detalles) |
| `--filter=PATTERN` | Filtra tests por nombre (substring match) |
| `--timeout <ms>` | Timeout por test en milisegundos |
| `--summary-only` | Solo muestra resumen final, sin detalle por categoría |
| `--sort=duration` | Ordena resultados por duración (más lento primero) |
| `--retry-failed <N>` | Reintenta tests fallidos hasta N veces |
| `--save-baseline <path>` | Guarda el informe como JSON baseline |
| `--diff <baseline.json>` | Compara contra un baseline previo |
| `--regression-threshold <pct>` | Umbral de caída de score para regresión (default: 10%) |

### Fase 2: 15 Nuevos Tests de Precisión Scored

Tests que miden exactitud numérica en funcionalidades clave. Cada test tiene un umbral
mínimo de aprobado:

| ID | Test | Umbral |
|----|------|--------|
| E1 | Content moderation safe precision | >= 0.90 |
| E2 | Content moderation harmful recall | >= 0.75 |
| E3 | Intent classification accuracy | >= 0.70 |
| E4 | Sentiment analysis directional accuracy | >= 0.80 |
| E5 | Chunking content preservation scored | >= 0.95 |
| E6 | Embedding similarity ordering | >= 0.80 |
| E7 | Query expansion diversity | >= 0.50 |
| E8 | PII per-type detection rate | >= 0.80 |
| E9 | RBAC permission boundary correctness | >= 0.95 |
| E10 | ORSet concurrent convergence | = 1.00 |
| E11 | Token count estimation accuracy | >= 0.90 |
| E12 | Priority queue ordering scored | = 1.00 |
| E13 | Injection detection obfuscated | >= 0.40 |
| E14 | Summarization key-term preservation | >= 0.60 |
| E15 | DHT store/retrieve fidelity | = 1.00 |

**Total tests de precisión**: 16 existentes + 15 nuevos = 31 tests scored.

### Fase 3: Integración CLI

Nuevos comandos REPL para testing y benchmarking directamente desde la CLI del asistente.

**Variantes añadidas a `ReplCommand`**:
- `Test(String)` — ejecuta tests del harness
- `Bench(String)` — ejecuta benchmarks
- `Precision` — ejecuta tests de precisión

**Comandos parse**:

| Comando | Acción |
|---------|--------|
| `/test` | Lista categorías, ejecuta todas, o ejecuta una específica |
| `/bench` o `/benchmark` | Ejecuta `cargo bench --bench core_benchmarks` |
| `/precision` | Ejecuta harness con `--category=precision --verbose` |

Los handlers en `ai_assistant_cli.rs` lanzan subprocesos para ejecutar el harness o
cargo bench según el comando.

### Fase 4: Detección de Regresiones

Sistema completo de comparación contra baselines para detectar regresiones automáticamente.

**Structs nuevos**:

| Struct | Campos |
|--------|--------|
| `DiffReport` | `regressions`, `improvements`, `new_tests`, `removed_tests`, `summary` |
| `TestDiff` | `was_passing`/`now_passing`, cambios de duración, cambios de score |
| `DiffSummary` | Conteos: pass→fail, fail→pass, score regressions, timing regressions |

**Funciones**:
- `diff_reports(current, previous, threshold)` — compara dos informes JSON
- `print_diff()` — salida coloreada (rojo para regresiones, verde para mejoras)
- `load_baseline(path)` — deserializa informe JSON previo

**Comportamiento**:
- Exit code 1 si se detectan regresiones pass→fail o caídas de score
- Umbral configurable con `--regression-threshold` (default: 10%)

---

## Flujo de uso típico

### Guardar baseline y detectar regresiones

```bash
# 1. Guardar baseline actual
cargo run --bin ai_test_harness -- --all --save-baseline baseline_v40.json

# 2. Hacer cambios al código...

# 3. Comparar contra baseline
cargo run --bin ai_test_harness -- --all --diff baseline_v40.json

# 4. Con umbral personalizado (5% en lugar de 10%)
cargo run --bin ai_test_harness -- --all --diff baseline_v40.json --regression-threshold 5
```

### Tests de precisión

```bash
# Ejecutar todos los tests de precisión con detalle
cargo run --bin ai_test_harness -- --category=precision --verbose

# Filtrar un test específico
cargo run --bin ai_test_harness -- --all --filter="pii" --verbose

# Ordenar por duración (encontrar tests lentos)
cargo run --bin ai_test_harness -- --all --sort=duration --summary-only
```

### Desde el REPL

```
/test                    # Lista categorías disponibles
/test security           # Ejecuta la categoría "security"
/bench                   # Ejecuta benchmarks de Criterion
/precision               # Ejecuta tests de precisión scored
```

---

## Estadísticas

| Métrica | Valor |
|---------|-------|
| Archivos modificados | 3 (ai_test_harness.rs, repl.rs, ai_assistant_cli.rs) |
| LOC añadidas | ~600 |
| Nuevos flags CLI | 9 |
| Tests de precisión nuevos | 15 |
| Tests de precisión totales | 31 |
| Tests lib (`cargo test`) | 6,829 (sin cambios — tests del harness son runtime) |
| Structs nuevos (regression) | 3 (DiffReport, TestDiff, DiffSummary) |
| Comandos REPL nuevos | 3 (/test, /bench, /precision) |

---

## Próximos pasos

- [ ] Añadir exportación de resultados del harness a formatos CI (JUnit XML, TAP)
- [ ] Integrar coverage del harness con `cargo-llvm-cov` o `grcov`
- [ ] Añadir tests de precisión para nuevos módulos (eval-suite, advanced routing)
- [ ] Dashboard HTML interactivo para comparar baselines históricos
