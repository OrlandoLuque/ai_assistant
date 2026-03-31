# Improvements V68 — LLM-Enhanced Pipeline (17 Modules)

## Infrastructure

| # | Item | Estado |
|---|------|--------|
| 1 | `LlmEnhancer` trait (generate + model_name + is_available) | HECHO |
| 2 | `LlmEnhancementConfig` (enabled, max_calls, timeout_ms) | HECHO |
| 3 | `MockLlm` + `FailingMockLlm` for testing | HECHO |
| 4 | `prompt_wrap()` security wrapper for user content | HECHO |
| 5 | `extract_json()` parser for LLM responses | HECHO |
| 6 | `MockSpeechProvider` (deterministic STT/TTS for testing) | HECHO |

## High-Impact Modules (1-5)

| # | Module | File | Enhancement | Estado |
|---|--------|------|-------------|--------|
| 7 | Conversation Compaction | conversation_compaction.rs | LLM abstractive summarization of removed messages | HECHO |
| 8 | Entity Extraction | advanced_memory/extraction.rs | LLM-based NER (person, org, location, date, concept) | HECHO |
| 9 | Intent Classification | intent.rs | LLM semantic intent + confidence scoring | HECHO |
| 10 | Query Expansion | query_expansion.rs | Wired existing expand_with_llm() via LlmEnhancer | HECHO |
| 11 | Response Quality | quality.rs | LLM evaluation (relevance, coherence, completeness) | HECHO |

## Medium-Impact Modules (6-11)

| # | Module | File | Enhancement | Estado |
|---|--------|------|-------------|--------|
| 12 | Topic Detection | analysis.rs | LLM topic classification merged with keyword baseline | HECHO |
| 13 | Auto-Model Selection | auto_model_selection.rs | LLM task type + complexity classification | HECHO |
| 14 | Guardrail Evaluation | injection_detection.rs | LLM injection detection (defense-in-depth, never downgrades) | HECHO |
| 15 | RAG Tier Auto-Selection | rag_tiers.rs | LLM query analysis → tier recommendation | HECHO |
| 16 | Document Chunking | rag.rs | LLM-suggested chunk boundaries (character offsets) | HECHO |
| 17 | KG Enrichment | knowledge_graph.rs | LLM relation inference between entities | HECHO |

## Low-Impact Modules (12-17)

| # | Module | File | Enhancement | Estado |
|---|--------|------|-------------|--------|
| 18 | Procedural Evolution | advanced_memory/evolution.rs | LLM failure analysis + improvement suggestions | HECHO |
| 19 | Speaker Intent | emotion_detection.rs | LLM intent + urgency classification | HECHO |
| 20 | Home Automation Intent | mcp_home_tools.rs | LLM natural language command interpretation | HECHO |
| 21 | Agent Task Decomposition | agent_methodology.rs | LLM task → steps breakdown | HECHO |
| 22 | Conversation Sentiment Trend | analysis.rs | LLM sentiment trend summary | HECHO |
| 23 | Multi-Agent Consensus | multi_agent.rs | LLM response synthesis from multiple agents | HECHO |

## Pattern

All 17 modules follow the same pattern:
1. `llm_enhanced: bool` in config (default: false — zero cost unless opted in)
2. `build_*_prompt()` — pure Rust prompt builder
3. `parse_*_response()` — JSON response parser with fallback
4. `*_with_llm(input, Option<&dyn LlmEnhancer>)` — enhanced method
5. Heuristic baseline always runs first
6. LLM failures gracefully fall back to heuristic

## Security

- User content wrapped in `prompt_wrap()` delimiters to prevent injection
- Guardrail LLM enhancement uses defense-in-depth: LLM can upgrade (safe→detected) but never downgrade (detected→safe)
- All prompts instruct LLM to treat delimited content as DATA, not instructions

## Test count

- Before: 7,206 (V67)
- After: 7,275 (+69)
- Pattern: 3 tests per module (heuristic, mock LLM, fallback) + infrastructure tests

## New files

- `src/llm_enhance.rs` — Shared LLM enhancement infrastructure
- `src/mock_speech.rs` — Mock speech provider for pipeline testing
