# V55 — Adaptive Context Budget Allocator

**Estado**: COMPLETADO
**Fecha**: 2026-03-23

---

## Resumen

V55 introduces intelligent context window management that maximizes the quality of
information sent to the LLM. Replaces hardcoded token budgets per source with a
score-based multi-source allocator that packs the most relevant items regardless
of origin. Adds LLM-assisted compression with domain filtering, a multi-armed
bandit for strategy learning, and shareable RAG tier definitions.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **ContextItem** — atomic unit with content, tokens, score, source type, label | HECHO |
| 2 | **ContextSource trait** — query_items(), source_name(), source_type() | HECHO |
| 3 | **ContextBudgetAllocator** — merge all items by score, greedy packing into budget | HECHO |
| 4 | **OverflowStrategy** — ScoreTruncation, ExtractiveCompression, LlmCompression, Hybrid | HECHO |
| 5 | **Extractive compression** — RECOMP-style sentence selection (Rust pure, free) | HECHO |
| 6 | **LlmCompressor trait** — compress with scores visible to the compressor LLM | HECHO |
| 7 | **build_compressor_prompt()** — items with scores + domain filtering instructions | HECHO |
| 8 | **CompressionLevel** — Light (~60-70%), Medium (~30-50%), Aggressive (~10-25%) | HECHO |
| 9 | **StrategyBandit** — UCB1 multi-armed bandit for learning best overflow strategy | HECHO |
| 10 | **LegacyStringSource** — adapter for existing String-returning sources | HECHO |
| 11 | **ClosureSource** — wrap any closure as ContextSource | HECHO |
| 12 | **build_from_items()** — allocate from pre-collected items without source trait | HECHO |
| 13 | **build_allocated_context()** — new method in AiAssistant replacing hardcoded injection | HECHO |
| 14 | **Wired into all send_message variants** — 5 variants now use the allocator | HECHO |
| 15 | **RagTierDefinition** — serializable tier config (JSON/TOML), shareable between users | HECHO |
| 16 | **RagTierStore** — manages builtin + custom tiers, import/export | HECHO |
| 17 | **GUI RAG tier selector** — horizontal tier picker with tooltips in ai_gui | HECHO |
| 18 | **CLI --rag-tier + --list-tiers** — tier selection and listing in ai_cli | HECHO |
| 19 | **ContextBudgetStatus** — Butler advisor with recommendations | HECHO |
| 20 | **REFERENCES.md** — 25+ papers on context compression, adaptive RAG | HECHO |
| 21 | **Diagram updates** — flow diagram 7 updated to show allocator pipeline | HECHO |
| 22 | **19 context_budget tests + 5 tier tests** | HECHO |

## Novel contributions (not found in published work)

1. Multi-source relevance-adaptive budget allocation
2. Score-forwarding to compressor (BM25/cosine as input signal)
3. Heterogeneous multi-source compression (RAG + Memory + Procedural)
4. Adaptive learning across compression strategies (UCB1 bandit)
5. Domain-aware contextual filtering ("Mordor vs Alps" problem)
6. Integrated 5-tier RAG + context budget allocator
7. Shareable RAG tier definitions (data, not code)

## Test count

- **Before**: 6,930 lib tests
- **After**: 6,957 lib tests (+27)
- **0 failures**

## Documentation updated

- `docs/REFERENCES.md` — 25+ papers
- `docs/TESTING.md` — test count 6,957, V55 row
- `docs/modus-operandi.md` — V55 line, What's next updated
- Website: developer_guide.html V55 section, index.html stats
