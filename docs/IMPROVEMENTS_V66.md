# Improvements V66 — Topic-Aware RAG Relevance

## RAG Relevance Enhancements

| # | Item | Estado |
|---|------|--------|
| 1 | `TopicMatcher` with Jaccard similarity for topic overlap | HECHO |
| 2 | Autocut: automatic score-gap detection for RAG result filtering | HECHO |
| 3 | LLM topic classifier for semantic topic matching | HECHO |
| 4 | Self-Query filter extraction from natural language queries | HECHO |
| 5 | ChunkRAG: sub-chunk granular scoring for fine-grained relevance | HECHO |
| 6 | 5 new RagFeatures (33 -> 38 total): semantic_dedup_fusion, distributed_search, context_budget_allocation, fresh_context, emotion_aware | HECHO |
| 7 | Butler RAG warnings for suboptimal configurations | HECHO |
| 8 | 12 attack vectors identified and mitigated | HECHO |
| 9 | Concept 235: Topic-Aware RAG | HECHO |

## Test count

- Before: 7,198 (V65)
- After: 7,218 (+20)
