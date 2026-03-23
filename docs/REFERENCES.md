# References — Papers and Projects

Research papers and open-source projects that inform the design of ai_assistant's
RAG pipeline, context management, and retrieval strategies.

## Context Compression

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [LLMLingua](https://github.com/microsoft/LLMLingua) | 2023 | EMNLP | Token-level compression via small LM surprisal, up to 20x |
| [LLMLingua-2](https://arxiv.org/abs/2403.12968) | 2024 | ACL | Data distillation from GPT-4 for BERT-level compressor, 3-6x faster |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839) | 2024 | ACL | Question-aware coarse-to-fine compression for RAG |
| [RECOMP](https://arxiv.org/abs/2310.04408) | 2024 | ICLR | Extractive + abstractive compression, empty output for irrelevant docs |
| [Selective Context](https://arxiv.org/abs/2310.06201) | 2023 | EMNLP | Self-information (surprisal) based pruning, no training required |
| [xRAG](https://arxiv.org/abs/2405.13792) | 2024 | NeurIPS | Extreme compression: entire document → 1 embedding token |
| [CCF](https://arxiv.org/abs/2509.09199) | 2025 | — | Hierarchical latent representations, 8-32x near-lossless |
| [ACON](https://arxiv.org/abs/2510.00615) | 2025 | — | Compression guideline optimization for long-horizon agents |

## Contextual Relevance Filtering

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [ChunkRAG](https://arxiv.org/abs/2410.19572) | 2024 | — | Multi-layered LLM chunk scoring with self-reflection |
| [Provence](https://arxiv.org/abs/2501.16214) | 2025 | ICLR | Lightweight DeBERTa binary keep/discard per sentence |
| [AttentionRAG](https://arxiv.org/abs/2503.10720) | 2025 | — | Attention scores across transformer layers for filtering |
| [AttnComp](https://arxiv.org/abs/2509.17486) | 2025 | EMNLP | Top-P attention-guided adaptive compression |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 2023 | — | Reflection tokens to decide when/what to retrieve |
| [CRAG](https://arxiv.org/abs/2401.15884) | 2024 | — | Corrective RAG: evaluates retrieval quality, triggers alternatives |

## Adaptive RAG

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 2024 | NAACL | Query complexity classifier → routes to appropriate retrieval |
| [MBA-RAG](https://arxiv.org/abs/2412.01572) | 2025 | COLING | Multi-armed bandit for retrieval strategy selection |
| [CtrlA](https://arxiv.org/abs/2405.18727) | 2025 | ACL | Internal representations to decide when to retrieve |
| [SARA](https://arxiv.org/abs/2507.05633) | 2025 | — | Progressive context selection, semantic + compressed vectors |
| [Dynamic-RAG](https://github.com/FUTUREEEEEE/Dynamic-RAG) | 2025 | AAAI | Multi-armed bandit on knowledge graphs |
| [CAR](https://arxiv.org/abs/2511.14769) | 2025 | — | Cluster-based adaptive retrieval document count |

## Context Engineering

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [Context Engineering Survey](https://arxiv.org/abs/2507.13334) | 2025 | — | Formalizes context as C = A(c1..cn), optimization under |C| ≤ Lmax |
| [Token-Budget-Aware Reasoning](https://aclanthology.org/2025.findings-acl.1274/) | 2025 | ACL | Dynamic output token budget based on problem complexity |
| [SelfBudgeter](https://arxiv.org/abs/2505.11274) | 2025 | — | Self-adaptive reasoning budget with RL refinement |

## Procedural Memory

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [TokMem](https://arxiv.org/abs/2510.00444) | 2025 | — | Procedures as compact trainable embedding tokens |

## Frameworks

| Project | Language | Key Features |
|---------|----------|-------------|
| [LangChain](https://python.langchain.com/) | Python | ContextualCompressionRetriever, chains, agents |
| [LlamaIndex](https://www.llamaindex.ai/) | Python | Multi-index RAG, query engines, response synthesis |
| [Haystack](https://github.com/deepset-ai/haystack) | Python | Hybrid retrieval, LLMRanker, pipeline architecture |
