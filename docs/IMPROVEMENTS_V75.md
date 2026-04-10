# Improvements V75 — Cost Intelligence: Wire, Enhance, Position

## Motivation

Xataka article ("La economía de tokens se ha roto") documents that AI flat-rate subscriptions are mathematically unsustainable: one Claude Max subscriber generated $5,600 in costs on a $100/month plan. Agentic code workloads consume 10-100x more tokens than interactive chat. ai_assistant already had the cost infrastructure (`cost.rs`, `cost_integration.rs`, `token_budget.rs`, `auto_model_selection.rs`, `context_budget.rs`) — but wiring gaps and missing features. V75 closes them.

## Workstream A: Wire CostDashboard in LLM pipeline

| # | Item | File | Estado |
|---|------|------|--------|
| A1 | Auto-record cost in `poll_response()` after `AiResponse::Complete` | assistant.rs | HECHO |
| A2 | `with_cost_config()` builder on `AiAssistant` | assistant.rs | HECHO |
| A3 | 3 new tests (builder, disabled config, report after init) | assistant.rs | HECHO |

## Workstream B: Savings estimation in AllocationResult

| # | Item | File | Estado |
|---|------|------|--------|
| B1 | Add `total_candidate_tokens`, `tokens_saved`, `compression_ratio` fields | context_budget.rs | HECHO |
| B2 | Compute savings in `build()` and `build_from_items()` | context_budget.rs | HECHO |
| B3 | `estimated_cost_saved(input_cost_per_million)` method (clamps negative pricing) | context_budget.rs | HECHO |
| B4 | 4 new tests (savings, compression_ratio, cost saved, negative pricing guard) | context_budget.rs | HECHO |

## Workstream C: Cost projection

| # | Item | File | Estado |
|---|------|------|--------|
| C1 | `projected_daily_cost()`, `projected_monthly_cost()`, `projected_cost_for_requests()` | cost_integration.rs | HECHO |
| C2 | `requests_per_hour()` helper, `parse_epoch_secs()` timestamp helper | cost_integration.rs | HECHO |
| C3 | Projections section in `format_report()` | cost_integration.rs | HECHO |
| C4 | 4 new tests (daily, monthly, N requests, report contains projections) | cost_integration.rs | HECHO |

## Workstream D: CostDashboard persistence

| # | Item | File | Estado |
|---|------|------|--------|
| D1 | `CostDashboardSnapshot` struct with `schema_version: u32` | cost_integration.rs | HECHO |
| D2 | `snapshot()` and `restore()` methods (validates costs on restore) | cost_integration.rs | HECHO |
| D3 | Re-export `CostDashboardSnapshot` | lib.rs | HECHO |
| D4 | 2 new tests (snapshot roundtrip, rejects NaN on load) | cost_integration.rs | HECHO |

## Workstream E: MCP tools for cost

| # | Item | File | Estado |
|---|------|------|--------|
| E1 | `register_cost_tools()` function with 3 tools | cost_integration.rs | HECHO |
| E2 | `cost_report` tool (current session report, read-only) | cost_integration.rs | HECHO |
| E3 | `cost_budget_status` tool (remaining budget + projections, read-only) | cost_integration.rs | HECHO |
| E4 | `cost_savings_summary` tool (allocator savings summary, read-only) | cost_integration.rs | HECHO |
| E5 | 3 new tests using `handle_request` protocol | cost_integration.rs | HECHO |

## Workstream F: Website cost positioning

| # | Item | File | Estado |
|---|------|------|--------|
| F1 | "Token Efficiency & Cost Control" section (5 cards) | index.html | HECHO |
| F2 | "TOKEN EFFICIENCY & COST CONTROL" comparison category (4 rows) | framework_comparison.html | HECHO |

## Workstream G: Security hardening

| # | Vector | Severity | Mitigation | Estado |
|---|--------|----------|-----------|--------|
| S1 | CSV injection in `export_csv()` | CRITICAL | `sanitize_csv_field()` prefixes dangerous chars, wraps in quotes | HECHO |
| S2 | Unbounded entries Vec (DoS) | HIGH | `MAX_ENTRIES = 100_000` cap with FIFO eviction | HECHO |
| S3 | Cost estimation bypass (output ratio) | MEDIUM | Documented: pre-request uses conservative estimate | NOTA |
| S4 | Float NaN/Infinity budget bypass | MEDIUM | `validate_cost()`: rejects NaN/Inf/negative → 0.0 | HECHO |
| S5 | TOCTOU race in concurrent pre_request | MEDIUM | Documented: not thread-safe, wrap in Mutex for concurrent use | NOTA |
| S6 | Persistence tampering (negative costs) | MEDIUM | Schema version + `validate_cost()` on restore | HECHO |
| S7 | MCP info disclosure of spending patterns | MEDIUM | All tools `read_only_hint: true`, aggregated data only | HECHO |
| S8 | Pricing input validation (negative) | MEDIUM | `estimated_cost_saved()` clamps `input_cost_per_million` to `>= 0.0` | HECHO |
| S9 | Website unsupported claims ("40-60%") | MEDIUM | Uses "reduces input tokens significantly" (no percentage) | HECHO |
| S10 | Division by zero in compression_ratio | LOW | Guarded: `if total_candidate_tokens == 0 { 1.0 }` | HECHO |

## New Types

### `CostDashboardSnapshot` (persistence)
- `schema_version: u32` — for forward compatibility
- `entries: Vec<RequestCostEntry>` — all session requests
- `session_start: String` — RFC3339 timestamp
- `budget_config: Option<CostAwareConfig>` — budget snapshot

### `AllocationResult` (new savings fields)
- `total_candidate_tokens: usize` — sum before packing
- `tokens_saved: usize` — candidate minus used
- `compression_ratio: f32` — used / candidate (1.0 = no compression)
- `estimated_cost_saved(f64) -> f64` — USD saved estimate

### Security helpers (private, cost_integration.rs)
- `const MAX_COST: f64 = 1_000_000.0`
- `const MAX_ENTRIES: usize = 100_000`
- `validate_cost(f64) -> f64` — NaN/Inf/neg → 0.0, clamped to MAX_COST
- `sanitize_csv_field(&str) -> String` — CSV formula injection prevention

## Builder API

```rust
use ai_assistant::{AiAssistant, AiConfig, CostAwareConfig};

let config = CostAwareConfig {
    enabled: true,
    daily_budget: Some(5.00),
    monthly_budget: Some(100.00),
    per_request_limit: Some(0.50),
    alert_threshold_pct: 80,
};

let assistant = AiAssistant::new(AiConfig::default())
    .with_cost_config(config);
// CostDashboard now auto-records every LLM call in poll_response()
```

## MCP Tools

All tools are read-only (`read_only_hint: true`) and expose aggregated data only.

| Tool | Returns |
|------|---------|
| `cost_report` | Full `format_report()` output (total, by model, by type, projections) |
| `cost_budget_status` | Remaining budget, projected monthly, status (OK/Warning/Blocked) |
| `cost_savings_summary` | Allocator tokens_saved, estimated_usd_saved, compression_ratio |

## Test Count

- Before: 7,469 (V74)
- After: 7,492 (+23)

Breakdown of new tests:
- `context_budget`: 4 (savings, compression_ratio, cost saved, negative pricing)
- `cost_integration`: 16 (7 security + 4 projection + 2 snapshot + 3 MCP)
- `assistant`: 3 (builder, disabled config, report after init)

**Note**: Lib-only test count with `full,autonomous,scheduler,butler,browser,audio-io,gpu-sharing,gui-pro,eval-suite,hitl,webrtc,devtools,home-automation,workflows,advanced-memory,voice-agent` measures **7,365**. The 7,492 figure follows project convention which includes integration and doc tests.

## Files Modified

| File | LOC delta |
|------|-----------|
| `src/context_budget.rs` | +50 (3 fields, method, tests) |
| `src/cost_integration.rs` | +280 (projection, snapshot, security, MCP, tests) |
| `src/assistant.rs` | +60 (builder, poll_response wiring, tests) |
| `src/lib.rs` | +1 (re-export) |
| `CHANGELOG.md` | +30 (V75 entry) |
| `docs/INNOVATIONS.md` | +20 (innovation #18) |
| `ai_assistant-website/index.html` | +55 (Token Efficiency section) |
| `ai_assistant-website/framework_comparison.html` | +45 (cost rows) |
| **Total** | ~540 LOC |

## Verification

```bash
cargo check --features "full,autonomous,scheduler,butler,browser"
# Finished dev profile [unoptimized + debuginfo] target(s)

cargo clippy --features "full,autonomous,scheduler,butler,browser" -- -W clippy::all
# 0 errors, 0 new warnings

cargo test --features "full,autonomous,scheduler,butler,browser,audio-io,gpu-sharing,gui-pro,eval-suite,hitl,webrtc,devtools,home-automation,workflows,advanced-memory,voice-agent" --lib
# 7,365 passed; 0 failed
```

## Innovations Catalog

Registered as innovation #18 in `docs/INNOVATIONS.md`:
**"Integrated Cost Intelligence with Security-Hardened Dashboard"** — Unique combination of auto-wired cost tracking, budget enforcement, cost projection, context savings estimation, and hardened persistence in a single integrated LLM pipeline.
