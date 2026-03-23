# V53 — SearchProviders + Web Search GUI + Block D Audit

**Estado**: COMPLETADO
**Fecha**: 2026-03-22

---

## Resumen

V53 adds 4 new search providers, integrates web search into the GUI, and completes
the Block D tool framework consolidation by deprecating unused modules and migrating
field names for interop.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **GoogleSearchProvider** — Google Custom Search API with configurable CX engine | HECHO |
| 2 | **BingSearchProvider** — Azure Cognitive Services Bing Web Search API | HECHO |
| 3 | **SerpApiProvider** — unified search API with configurable engine (Google, Bing, Yahoo) | HECHO |
| 4 | **TavilyProvider** — AI-optimized search with content extraction | HECHO |
| 5 | **GUI Web Search tab** — DuckDuckGo search in ai_gui with async thread+channel | HECHO |
| 6 | **Block D audit** — tool_use + function_calling deprecated (0 consumers) | HECHO |
| 7 | **Block D migration** — tool_calling field names unified with unified_tools (tool_name→name) | HECHO |
| 8 | **From<> conversions** — bidirectional between tool_calling::ToolCall ↔ unified_tools::ToolCall | HECHO |
| 9 | **GUI non-exhaustive fixes** — 6 pre-existing match arm errors fixed | HECHO |
