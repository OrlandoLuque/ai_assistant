# V142 — RUSTSEC review + automatización (0.2.89)

**Status**: DRAFT — independiente, mantenimiento
**Scope**: revisar las 4 entradas vigentes de `deny.toml#advisories.ignore`,
añadir automatización para que esa lista no se quede obsoleta.
**No-goals**: parches a vulnerabilidades — las 4 ignoradas hoy están
documentadas como no-aplicables a este crate.

## Por qué

`deny.toml` ignora 4 RUSTSEC con justificación caso a caso:

- `RUSTSEC-2025-0141` — bincode 1.x unmaintained
- `RUSTSEC-2024-0436` — paste unmaintained
- `RUSTSEC-2025-0134` — rustls-pemfile unmaintained
- `RUSTSEC-2026-0002` — lru IterMut unsoundness (vía tantivy)

El riesgo no es que estén ignoradas hoy — está justificado — sino que
nadie verifica si la justificación sigue vigente cuando upstream
publica una nueva versión que la haga obsoleta.

## Acciones

### 1. Revisar las 4 entradas actuales

Por cada una, comprobar:

- ¿Hay versión nueva de la dep transitiva que evite la crate
  comprometida?
- ¿La justificación ("our query path doesn't trip the affected
  codepath") sigue siendo cierta tras los últimos commits del crate?
- ¿Tiene fecha de revisión próxima?

Resultado esperado: o se elimina la entrada (porque ya no hace falta),
o se le añade un comentario con la próxima fecha de re-revisión.

### 2. Automatizar la revisión

- **GitHub Action `rustsec-review-monthly`** — cron 1.º de mes. Corre
  `cargo audit --json` + extrae IDs vs `deny.toml#ignore` y abre un
  issue listando "ignored entries to re-review" con la lista
  actualizada y diffs si los hay.
- **Pre-merge** — el CI actual con `cargo-deny check` queda como
  está; sigue siendo gating.
- **Renovate** — el `renovate.json` ya tiene `vulnerabilityAlerts`
  con assignee. Verificar que está disparando.

### 3. Política escrita

Añadir sección "RUSTSEC handling" a `docs/runbooks/`:

- Nunca añadir un ignore sin comentario que explique *por qué* no
  aplica.
- Cada ignore lleva un *re-check by* (fecha o evento, ej.
  "when tantivy releases 0.25").
- El monthly review abre un issue obligatorio; cerrar sólo tras
  revisión efectiva.

## Tests / verificación

- Smoke test del workflow nuevo en un PR de prueba.
- Verificar que `cargo audit` sigue saliendo con exit ≠ 0 (eso es
  esperado; gating lo hace `cargo deny check` que respeta los
  ignores).

## Riesgos

- **Bot fatigue** — issues mensuales que nadie atiende. *Mit*: el
  workflow detecta y reabre si llevan 30 días abiertos, asignados a
  ti.
- **Ignore creep** — añadir excepciones por inercia. *Mit*: la
  política escrita exige justificación y fecha de re-check, sin las
  cuales el lint del workflow falla.

## Iteraciones

- **iter 1**: sólo política escrita. Rechazado: sin automatización
  vuelve a quedarse obsoleta.
- **iter 2 (actual)**: política + workflow mensual + Renovate
  verification.

## Out of scope para V142

- Cambios a `cargo-deny` config más allá de la lista de ignores.
- SBOM regen (eso ya está en V125, mantenido por Renovate).
- Política de actualización de toolchain (separada).
