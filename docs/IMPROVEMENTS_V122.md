# V122 — Phase B.5: parallel read-only tool execution in `autonomous_loop`

**Date**: 2026-05-05
**Version**: 0.2.69
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § B.5
**Tasks**: #328 (B.5 — parallel tool execution)

## Why

Current frontier-style agents (Claude Code, Cursor, Aider) routinely
emit several tool calls in a single response — the natural shape for
"read these five files and tell me what's going on." The autonomous
runner has been executing them strictly serially, which made every
multi-read iteration cost N × handler-latency even when there was no
ordering constraint between the calls.

V122 closes that gap with the smallest possible change: when *every*
tool in a batch is in a conservative read-only allow-list, the runner
executes them concurrently. This unlocks the dominant case (multi-
file reads, multi-source searches) without introducing dependency
analysis, async runtimes, or feature-flag creep.

## Why an allow-list, not dependency analysis

The "right" answer eventually is read/write annotations on every
tool definition with a write-after-read scheduler. That's a much
bigger change — it touches `unified_tools::ToolDefinition`, every
tool registration site, and the planner. V122 deliberately ships
the conservative slice first: an allow-list of tools known by name
to be side-effect-free. Anything outside the list assumes potential
mutation and falls back to sequential. Same eventual ceiling; much
smaller blast radius today.

## What changed

### `AutonomousAgentConfig::parallel_read_only_tools: bool`

Default `false`. Existing builders are unaffected — same ordering,
same locking, same cost accounting. The builder gains a matching
opt-in:

```rust
pub fn parallel_read_only_tools(mut self, on: bool) -> Self;
```

### `is_read_only_tool_name(&str) -> bool`

Public free function exposing the allow-list so external callers
can align their own classification:

| Read | List/search | Fetch | Inspect |
|---|---|---|---|
| `read_file`, `read`, `cat`, `head`, `tail` | `list_files`, `list_dir`, `ls`, `glob`, `find`, `search`, `grep`, `web_search`, `vector_search`, `rag_search`, `knowledge_search`, `tool_search`, `lookup` | `get_url`, `fetch`, `http_get`, `curl_get` | `stat`, `exists` |

Anything not listed is assumed to potentially mutate state.

### `run_iteration` — parallel branch

After `parse_tool_calls`, the agent first scans for `ask_user`
(short-circuit), then evaluates eligibility:

```rust
let parallel_eligible = self.config.parallel_read_only_tools
    && parsed.len() >= 2
    && parsed.iter().all(|tc| is_read_only_tool_name(&tc.name));
```

When eligible:

1. **Sandbox validation** — every call validates sequentially
   (fail-fast on denial). The sandbox is single-threaded and the
   validation also writes audit state, so we keep it serial.
2. **Build all `ToolCall`s** — convert `ParsedToolCall::arguments`
   (a `HashMap<String, String>`) into the registry's
   `HashMap<String, JsonValue>` shape.
3. **Dispatch via `std::thread::scope`** — each handle calls
   `&self.tool_registry.execute(call)`. Lifetimes are scoped to
   the closure; no `Arc` clones, no static lifetimes needed.
4. **Collect results in original order** — `zip(parsed, results)`
   over the original ordering.
5. **Process results sequentially** — push tool messages onto the
   conversation, accumulate cost via `CostConfig::cost_for`,
   update `tools_called_log`, set the `self-correction` flags.

Side-effects fire in parsed order regardless of how the workers
interleaved, so the LLM-visible message stream is identical to the
sequential path.

### Why `std::thread::scope`

- **`tokio`** would feature-creep `async-runtime` into the
  autonomous runner; we want the runner to stay sync-callable from
  any context.
- **`rayon`** would require gating on the `distributed` feature
  for a general-purpose use case; the autonomous runner shouldn't
  pull it in.
- **`std::thread::scope`** is in std since 1.63, requires no
  Cargo changes, and gives us structured-concurrency lifetime
  safety for `&self.tool_registry` borrows. `ToolHandler` is
  `Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync>`
  (see `unified_tools::ToolHandler`), so the registry is naturally
  shareable across threads.

## Compatibility

- `parallel_read_only_tools` defaults to `false`. Builders that
  never call the new setter behave exactly as before.
- `AutonomousAgentConfig` adds one field (`#[non_exhaustive]`);
  the builder constructs the struct internally, so external
  callers using the builder are unaffected.
- The sequential path is preserved verbatim under `else { … }`,
  not refactored away — the parallel branch is purely additive.

## Tests

4 new tests in `autonomous_loop::tests`:

| Test | Asserts |
|---|---|
| `test_is_read_only_tool_name_classification` | The allow-list is pinned: positive cases (`read_file`, `glob`, `web_search`, `vector_search`, `rag_search`) and negative cases (`write_file`, `delete_file`, `execute_command`, `ask_user`, empty string, `calculate`). |
| `test_parallel_read_only_executes_all_calls` | Two `read_file` calls, each handler sleeps 60 ms. Asserts both ran *and* the whole iteration finished under 200 ms. A strictly sequential schedule would have ≥ 120 ms of pure handler time plus agent overhead. |
| `test_parallel_falls_back_to_sequential_on_unknown_tool` | A mixed batch (`read_file` + `calculate`). `calculate` isn't in the allow-list, so parallel is *not* eligible; both calls still run, sequentially. |
| `test_parallel_disabled_keeps_sequential_path` | Two read-only calls but the flag isn't set; the run completes via the sequential branch. Guards against accidental opt-in. |

All 6,660 lib tests pass under
`cargo test --features "autonomous,self-correction,multi-agent" --lib`.

## What's next

- **V123 (B.6)**: adversary + egress inspectors and the
  `--no-egress` policy flag for closed-network operation.
- **Optional follow-up**: lift the all-or-nothing policy into
  fine-grained scheduling. Add `is_potentially_mutating_tool_name`
  + per-tool `read_paths` / `write_paths` annotations on
  `ToolDefinition`, then schedule "read group → barrier → write
  → read group" partially-parallel batches. The conservative
  V122 policy is the right starting point; the bigger refactor
  earns its keep only after we've measured how often mixed
  batches show up in practice.
