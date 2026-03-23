# Innovations

Novel contributions in ai_assistant that are not found in existing published work.

---

## 1. Multi-Source Relevance-Adaptive Budget Allocation

**What**: Dynamically distributes context window tokens across RAG chunks, episodic memory, procedural workflows, knowledge graph entities, and conversation references — based on per-item relevance scoring.

**Why it's new**: The [Context Engineering Survey (2025)](https://arxiv.org/abs/2507.13334) formalizes the problem of assembling context from multiple components under a token budget, but provides no algorithm for dynamic allocation. Existing frameworks (LangChain, LlamaIndex, Haystack) use fixed proportions or manual orchestration. No published system performs query-adaptive, score-driven budget distribution across heterogeneous knowledge sources.

**How it works**: Each source implements `ContextSource` and returns `Vec<ContextItem>` with relevance scores. The `ContextBudgetAllocator` merges all items from all sources, sorts by score descending, and packs greedily into the available budget. The most relevant items win regardless of which source produced them.

---

## 2. Score-Forwarding to Compressor

**What**: When LLM-assisted compression is needed, the retrieval relevance score (BM25, cosine similarity, confidence) is passed *as an input* to the compressor LLM alongside the text content.

**Why it's new**: All existing compression systems (LLMLingua, RECOMP, Selective Context, xRAG) work on text alone — the compressor has no knowledge of upstream relevance scores. By forwarding scores, the compressor can preserve high-confidence items verbatim while aggressively compressing borderline ones.

**How it works**: `build_compressor_prompt()` includes `[score 0.95] [RAG] content...` for each item, so the compressor LLM can make informed priority decisions.

---

## 3. Heterogeneous Multi-Source Compression

**What**: A single compression pipeline that handles RAG chunks, memory snippets, procedural rules, graph entities, and user notes — each with different characteristics and priorities.

**Why it's new**: Existing systems compress homogeneous documents (all RAG chunks, or all conversation messages). None handle the heterogeneous case where different source types require different compression strategies and have different baseline priorities.

**How it works**: The `ContextSourceType` enum tags each item, and the compressor prompt includes source type labels so the LLM understands each item's nature.

---

## 4. Adaptive Learning Across Compression Strategies

**What**: A multi-armed bandit (`StrategyBandit`) learns which compression strategy works best for which type of query, adapting over time based on outcome feedback.

**Why it's new**: [ACON (2025)](https://arxiv.org/abs/2510.00615) learns compression guidelines for agent trajectories, but no system learns across multiple distinct strategies (score truncation vs. extractive vs. LLM Light/Medium/Aggressive) and adapts the choice per query type.

**How it works**: UCB1 bandit with arms for each strategy. After each query, the selected strategy receives a reward based on response quality. Over time, the bandit converges on the best strategy per scenario.

---

## 5. Domain-Aware Contextual Filtering

**What**: LLM-based filtering that discards items which are keyword-relevant but contextually irrelevant — the "Mordor vs Alps" problem where a query about fictional mountains retrieves real-world geography.

**Why it's new**: [ChunkRAG (2024)](https://arxiv.org/abs/2410.19572) performs multi-layer scoring with self-reflection, but doesn't explicitly address topical/domain mismatch. No system combines BM25/embedding retrieval scores with LLM-based domain disambiguation.

**How it works**: The compressor prompt instructs the LLM to "discard items that are keyword-relevant but contextually irrelevant to the question (wrong domain, wrong topic, false matches)."

---

## 6. Integrated 5-Tier RAG + Context Budget Allocator

**What**: A 9-tier RAG system (Disabled → Fast → Semantic → Enhanced → Thorough → Agentic → Graph → Full → Custom) integrated with an adaptive context budget allocator — treating RAG tier selection and context assembly as a unified optimization problem.

**Why it's new**: In the literature, adaptive RAG ([Adaptive-RAG, NAACL 2024](https://arxiv.org/abs/2403.14403)) and context compression ([LLMLingua](https://github.com/microsoft/LLMLingua)) are completely separate research areas. No system combines multi-tier RAG with Self-RAG, CRAG, and Graph RAG with a budget-aware context allocator.

---

## 7. Shareable RAG Tier Definitions

**What**: RAG tier configurations are data (JSON/TOML), not code. Users can create custom tier definitions, export them as files, and share them with others.

**Why it's new**: All existing RAG frameworks define retrieval strategies in code (Python classes, configuration objects). No system provides serializable, versioned, importable/exportable tier definitions that users can share like recipes.

**How it works**: `RagTierDefinition` is a serializable struct with name, description, author, version, and full feature flags. `RagTierStore` manages builtin + custom tiers with import/export support.

---

## Implementation

All innovations are implemented in Rust, providing:
- **Performance**: Extractive compression, bandit learning, and budget allocation run in microseconds
- **Memory safety**: Zero-cost abstractions with ownership semantics
- **No Python dependency**: Pure Rust with optional ONNX for neural models

See [REFERENCES.md](REFERENCES.md) for the full list of papers and projects that inform this work.
