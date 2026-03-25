# Improvements V59–V61

Combined log for versions V59, V60, and V61 (implemented in session 2026-03-24/25).

## V59 — Security Hardening

| # | Item | Estado |
|---|------|--------|
| 1 | ConfigLock — 8 lockable config sections, UnlockRequirement | HECHO |
| 2 | IntegrityChecker — SHA-256 checksum tamper detection | HECHO |
| 3 | SecurityAlertManager — cooldown dedup, alert levels | HECHO |
| 4 | LearningFreezeConfig — 8 freezable subsystems | HECHO |
| 5 | NetworkPolicy — permissive/restrictive/paranoid, SSRF protection | HECHO |
| 6 | 6 concrete vulnerability fixes (server_axum, distributed_log, guardrails, os_tools, distributed, rate_limit) | HECHO |
| 7 | 256-vector exhaustive security audit | HECHO |
| 8 | 27 new tests | HECHO |

## V60 — Semantic Dedup + Distributed RAG + P2P Security

| # | Item | Estado |
|---|------|--------|
| 1 | SemanticDeduplicator — 3-level (identical/similar/distinct), batched LLM fusion | HECHO |
| 2 | DistributedRag — DocumentScope (Private/Shared), SharedChunkMeta with TTL | HECHO |
| 3 | IceCandidate/IceConfig — ICE NAT traversal types | HECHO |
| 4 | P2P Security — TrustLevel (4 levels), MessageAuthorization matrix, PeerAccessControl | HECHO |
| 5 | Concepts 210-217 | HECHO |
| 6 | 33 new tests | HECHO |

## V61 — Browser Policy + MCP Security

| # | Item | Estado |
|---|------|--------|
| 1 | BrowserPolicy — URL validation, SSRF protection, JS permission levels | HECHO |
| 2 | JsPermission — Disabled/ReadOnly/Mutating/Full with 16 dangerous patterns | HECHO |
| 3 | ToolPermission — 14 fine-grained permission categories | HECHO |
| 4 | Concepts 218-219 | HECHO |
| 5 | 17 new tests | HECHO |
