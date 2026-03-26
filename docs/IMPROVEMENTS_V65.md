# Improvements V65 — Model-Aware Tokenization

## Token Precision System

| # | Item | Estado |
|---|------|--------|
| 1 | `TokenPrecision` enum (Exact, BPE, Heuristic) with automatic selection | HECHO |
| 2 | `TiktokenCounter` cfg-gated behind `precise-tokens` feature | HECHO |
| 3 | `ProviderTokenCounter` fallback chain: tiktoken → BPE → heuristic | HECHO |
| 4 | Dynamic reserve margin 10-20% based on precision level | HECHO |
| 5 | Model-aware budget allocation in `assistant.rs` | HECHO |
| 6 | Feature flag `precise-tokens` (adds ~4MB to binary) | HECHO |

## Test count

- Before: 7,179 (V64)
- After: 7,198 (+19)
