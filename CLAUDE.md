# Contexto del Proyecto: ai_assistant

## Inicio de sesión — OBLIGATORIO
Al iniciar cada nueva sesión de trabajo, **lee estos archivos ANTES de hacer cualquier otra cosa**:
1. `docs/modus-operandi.md` — workflow, reglas de calidad, checklist de implementación
2. `CHANGELOG.md` (las entradas de cabecera) — **el estado actual**: qué se hizo en las
   últimas versiones y por qué
3. Si la sesión va de medir modelos: `docs/MODEL_BENCHMARKS.md` (resultados, con fecha y
   backend) y `docs/LOCAL_MODELS.md` (cuantización, KV cache, qué cabe en la tarjeta)

Esto te da el contexto completo: dónde estamos, qué se ha hecho, cómo trabajamos.

> **Los `docs/IMPROVEMENTS_V*.md` son histórico, no estado.** La serie se detuvo en
> **V167** (0.2.119) y el trabajo posterior — más de cien versiones — vive en el
> `CHANGELOG.md`. Leer «el de número más alto» daba una foto de marzo de 2026 como si
> fuera la actual, que es exactamente el error que esta sección debe evitar.
> El propio `IMPROVEMENTS_V167.md` lo dice ya en su cabecera, y `docs/README.md` es el
> índice de los 198 ficheros de `docs/`: qué es actual, qué es histórico y qué es una
> auditoría con fecha.

## Sobre el Proyecto
- **Librería Rust** para integración con LLMs locales y cloud
- **540K líneas de código**, 559 archivos fuente (.rs), 9.753 tests
  (medido el 2026-09-03 en 0.2.244; antes decía 523K/500/9.600+)
- **Autor único**: Orlando José Luque Moraira (Lander) — orlando.luque@gmail.com
- **Estado**: **el repositorio de GitHub es PÚBLICO** (`OrlandoLuque/ai_assistant`, 3 estrellas
  y 1 fork a 2026-09-03). No está en crates.io ni distribuido a terceros de otra forma.
  Esta línea decía «NO publicado en ningún sitio (ni crates.io, ni GitHub público…)», que
  llevaba tiempo siendo falso. La licencia PolyForm Noncommercial sigue aplicando y prohíbe
  el uso comercial, pero **el código es legible por cualquiera y ya se ha bifurcado una vez**:
  cualquier decisión sobre PI, secreto industrial o «aún no lo hemos enseñado» debe partir
  de ese hecho. Si la intención era mantenerlo privado, hay que cambiar la visibilidad, no
  la frase.
- **Feature flags**: 95 (multi-agent, security, analytics, rag, distributed, autonomous, gui-pro, audio-io, gpu-sharing, research, etc.). El conjunto mínimo soportado son ocho, documentado en `Cargo.toml` y verificado en CI.
- **Binarios**: 41 declarados en `Cargo.toml` (V305 añadió `ai_mcp_server`). El inventario
  es `docs/BINARIES.md`, y desde V305 CI comprueba que coincide con el manifiesto: decía 26
  de 41, y una página que se llama a sí misma exhaustiva y va al 60 % contesta «no existe»
  con seguridad a preguntas sobre quince binarios que sí existen.

## Módulos Principales
- Multi-proveedor LLM — **19 con nombre propio** más el genérico `OpenAICompatible`
  (`AiProvider` en `src/config.rs`, verificado 2026-09-03): Ollama, LM Studio,
  text-gen-webui, Kobold.cpp, LocalAI, llama.cpp, vLLM, OpenAI, Anthropic, Gemini,
  Bedrock, Groq, Together, Fireworks, DeepSeek, Mistral, Perplexity, OpenRouter,
  Azure OpenAI. *(Antes decía 18 e incluía HuggingFace, que no es una variante del enum;
  faltaban llama.cpp y vLLM.)*
- RAG 5 niveles: Self-RAG, CRAG, Graph RAG, RAPTOR — 7 backends vector DB
- Multi-agente: 5 roles, orquestación, memoria compartida
- Agente autónomo: 5 niveles autonomía, scheduler cron, browser automation (CDP)
- Distribuido: CRDTs, DHT Kademlia, MapReduce, QUIC/TLS 1.3
- Seguridad: RBAC, PII detection, guardrails constitucionales, AES-256-GCM
- Streaming: SSE, WebSocket RFC 6455, compresión, resumible
- FreshContext mode: contexto alternativo que maximiza tokens para conocimiento
- MCP protocol (40+ tools + 4 knowledge tools), WASM, egui widgets, HTTP server embebido
- Memory integration: MemoryManager con auto-inyección en FreshContext
- FreshContext Advisor API: diagnóstico programático (effectiveness, warnings)
- ContextBudgetAllocator: scoring dinámico (4 modos), intent-based, integrado con RAG tiers
- Anti-Hallucination: pipeline configurable (7 estrategias), abstención calibrada, auto-temperature, attribution
- Faithfulness scoring: NLI-based (word overlap + LLM), grounded generation, claim decomposition
- Chain-of-Verification (CoVe): fact-check con RAG/web search, self-consistency mejorado
- Quality Gates: métricas configurables (faithfulness, confidence, grounding ratio), acciones fail/warn/log
- Research académico: arXiv, Semantic Scholar, PubMed APIs, BibTeX parser/generator, literature review pipeline
- Paper metadata: extracción estructurada (título, autores, secciones, DOI, referencias), 3 agent roles research

## Decisiones Estratégicas Tomadas (Febrero 2026)

### Licencia — DECIDIDA
- **Licencia elegida**: **PolyForm Noncommercial 1.0.0** (decidida 2026-02-22)
- **Efecto**: Prohíbe TODO uso comercial sin licencia negociada. Solo permite uso personal, académico, investigación.
- **Archivos**: `LICENSE` (PolyForm Noncommercial), `Cargo.toml` usa `license-file = "LICENSE"`
- **MIT/Apache-2.0 es de `ai_assistant_core`, no de este proyecto.** Ese crate aparte lleva
  `license = "MIT OR Apache-2.0"` en su `Cargo.toml` y sus dos ficheros de licencia. Es
  deliberado y no hay nada que corregir ahí. **Este** repositorio es PolyForm y solo PolyForm.
- **Historia limpiada el 2026-09-03.** `LICENSE-MIT` y `LICENSE-APACHE` estuvieron en la
  historia de ESTE repositorio y ya no están. Lo ocurrido, por si vuelve a salir el tema:
  - El commit `9daa953` (21/02/2026) los metió dentro de un cambio de **271 ficheros y
    98 124 líneas** — andamiaje, no decisión: MIT/Apache es el par por defecto de un crate
    de Rust. `eb2ccdc` (22/02/2026) los quitó y puso PolyForm. **Un día de ventana**, y solo
    3 commits de 670 llegaron a contenerlos.
  - **El repositorio de GitHub se creó el 11/03/2026**, dos semanas y media más tarde, así
    que el repo público **nunca presentó MIT/Apache como su licencia**: su `LICENSE` ha sido
    PolyForm desde el primer día que existe.
  - Se reescribió la historia con `git-filter-repo` (`--invert-paths`) y force-push. Los dos
    commits son ahora `44f95a5` y `5b5abf6`. Verificado: 670 commits antes y después, y el
    **árbol de `master` idéntico** (`ca61fdc1…`) — no se perdió una línea de código. Copia de
    seguridad previa en un bundle completo antes de tocar nada.
  - **Lo que la reescritura NO consigue, y conviene no olvidarlo:** GitHub mantiene los
    objetos viejos accesibles por SHA exacto hasta que su Soporte haga recolección de basura
    (comprobado: `9daa953` seguía resolviendo por API después del force-push). Y el fork
    `janreges/ai_assistant` conserva la historia antigua completa. Ninguna de las dos cosas
    se arregla desde aquí.
  - Ese fork quedó **desacoplado** (repo público independiente, sin relación con este) por un
    cambio de visibilidad accidental el mismo día; el código está intacto. Ver la memoria
    `feedback_never_change_repo_visibility`.
- **Opción futura**: Considerar publicar un módulo básico pequeño (solo providers) bajo MIT en crates.io como módulo open-source bajo MIT/Apache-2.0

### Monetización
- **Modelo elegido**: PolyForm Noncommercial + negociación caso a caso con empresas
- **Motivo**: El autor es desarrollador solo, con tiempo y presupuesto limitados, con familia
- **Estrategia de precios**: Caso a caso, sin infraestructura SaaS compleja por ahora
- **Futuro**: Si crece, evolucionar a Open-Core con features premium

### Protección de Propiedad Intelectual
- **Registro PI España** (cultura.gob.es): 13,59€ — PENDIENTE
- **WIPO PROOF** (OMPI/ONU): ~20€ — PENDIENTE
- **Safe Creative**: 15-30€ — PENDIENTE
- **Depósito notarial**: 50-150€ — PENDIENTE (recomendado)
- **CLA**: Implementar antes de recibir contribuciones externas (CLA Assistant en GitHub)
- **Patentes**: DESCARTADAS (no merece la pena en Europa para software puro)

### Estrategia de Features para Monetización Futura
| Tier | Features | Precio |
|------|----------|--------|
| Gratuito | core, embeddings, streaming, tools, documents, rag | 0€ |
| Pro | multi-agent, security, analytics, eval, vision | Pago |
| Enterprise | distributed, p2p, autonomous, scheduler, browser | Personalizado |

## PARADA OBLIGATORIA — licencias, visibilidad y cualquier cosa con efecto legal

**Nada de esto se toca sin (a) petición explícita del autor y (b) confirmación suya después
de que le expliques qué hace y qué no se puede deshacer.** No basta con que «se deduzca» de
un objetivo que te haya dado. Si te pide una meta («protege esto», «corta la exposición»,
«limpia aquello»), tu trabajo es **enumerar los mecanismos y preguntar cuál**, no elegir uno.

Entra aquí, como mínimo:

- **Licencias**: `LICENSE`, `LICENSE-*`, los campos `license` / `license-file` de
  `Cargo.toml`, cabeceras de copyright, avisos de terceros, `NOTICE`, `CLA.md`.
  **Nunca añadas un fichero de licencia «por convención»** — así entraron `LICENSE-MIT` y
  `LICENSE-APACHE` en febrero de 2026, dentro de un commit de 271 ficheros, y estuvieron en
  la historia pública seis meses. Este proyecto es **PolyForm Noncommercial 1.0.0**; MIT y
  Apache-2.0 son de `ai_assistant_core`, que es otro crate.
- **Visibilidad de repositorios**: público ↔ privado, en este repo o en cualquier otro.
  Cambiarla borra estrellas y desacopla forks, y volver atrás **no** los recupera.
- **Reescrituras de historia y force-push**: `filter-repo`, `filter-branch`, `push --force`,
  borrado de ramas o tags. Cambian todos los SHA y no siempre logran lo que parecen: GitHub
  sigue sirviendo los objetos viejos por SHA exacto hasta que su Soporte haga GC.
- **Publicación y distribución**: crates.io, releases, paquetes, subir artefactos, hacer
  público un repo o una página.
- **Terceros y datos**: CLA, DPIA, textos de privacidad, condiciones de uso, cualquier cosa
  que afirme algo sobre datos personales o sobre derechos de otros.
- **Afirmaciones legales en documentación**: qué licencia aplica, qué se puede o no hacer con
  el código, si algo se distribuyó o no. Si una frase de este tipo está desfasada, **dilo y
  propón la corrección**; corregir un hecho verificable (con la evidencia delante) es
  aceptable, redefinir la estrategia no lo es nunca.

Precedente que originó esta regla (2026-09-03): «corta por ahora la exposición futura» se
interpretó como poner el repositorio en privado. Costó 2 de 3 estrellas y desacopló el fork
de un tercero, ninguna de las dos cosas recuperable. Lo que el autor quería era otra cosa.

## Reglas para Claude Code
0. **Lee la PARADA OBLIGATORIA de arriba antes de tocar licencias, visibilidad o historia.**
1. **NUNCA modificar código sin permiso explícito** — el autor es muy protector con su trabajo
2. El código ha sido desarrollado iterativamente con Claude (prompts del autor + generación asistida)
3. Respetar la estructura modular existente basada en feature flags de Cargo
4. Zero `.unwrap()` en producción — usar proper error handling siempre
5. Zero warnings del compilador — compilación limpia en todas las combinaciones de features
6. Tests para todo — el proyecto tiene 9.600+ tests y debe mantenerse así

## Plan Nocturno (Night Plan)
Definido en las **instrucciones globales de usuario** (`~/.claude/CLAUDE.md`),
aplicable a cualquier proyecto de esta máquina. Resumen: solo cuando el autor lo
pide explícitamente esa noche; safety-net `ClaudeNightSuspend` + helper
`C:\Users\Lander\.claude\night_suspend.ps1` (ubicación estable, ya no en Temp);
dos variantes **Solo** (al acabar suspende directamente) y **Cooperativo** (no
suspende, deja que la programación llegue sola). Ciclo de calidad de este
proyecto en el plan nocturno: compila, clippy `-D warnings`, tests, batería,
commit por cambio. Fuera del plan nocturno explícito, **NUNCA suspendas/apagues**
el equipo (ver memoria `feedback_no_suspend`).

## Documentos Generados
- `Informe_Viabilidad_ai_assistant.docx` — Informe completo de viabilidad v2 (monetización, PI, licenciamiento)
  - Actualizado 21/02/2026 con estrategia PolyForm Shield (recomendada) vs Noncommercial (alternativa)
  - Incluye secciones sobre evolución del código, plan de acción y costes

## Tareas Pendientes
- [x] Actualizar el informe .docx con la estrategia PolyForm (v2 completada 21/02/2026)
- [x] Decidir licencia → PolyForm Noncommercial 1.0.0 (decidida 22/02/2026)
- [x] Sustituir LICENSE-MIT y LICENSE-APACHE por LICENSE PolyForm (hecho 22/02/2026)
- [x] Actualizar campo `license` en Cargo.toml (hecho 22/02/2026)
- [ ] Registrar PI en España, WIPO PROOF y Safe Creative
- [x] Implementar CLA en el repositorio (CLA.md + CONTRIBUTING.md + GitHub Action, hecho 11/04/2026)
- [ ] Decidir si publicar módulo básico bajo MIT como módulo open-source bajo MIT/Apache-2.0
- [ ] Crear página web/landing con info de licencia comercial
