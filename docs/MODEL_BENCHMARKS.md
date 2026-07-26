# Model Benchmarks — Execution-Verified Live-Model Log

A running **lab notebook** for the `ai_test_harness` live-model benchmarks. Unlike
the [CHANGELOG](../CHANGELOG.md) — which records *code changes* — and
[BENCHMARKS.md](BENCHMARKS.md) — which covers *performance/footprint* vs other
frameworks — this file records **model-quality results over time**: which models
pass which tasks, how the curves move as tasks / models / harness evolve, and what
we learned.

> **How to keep this file:** append a **new dated entry at the top of the Log**
> each time you run a meaningful sweep. Don't edit old entries (they record what
> was true then); if a number was wrong, correct it in a new entry. Use the
> template at the bottom.

## What's measured

Three harness categories, all **execution-verified** (the code the model produces
is actually run against `assert` checkers — a PASS means the artifact *works*, not
that it "looks right"). The backend is configurable via env
(`AI_BENCH_PROVIDER` / `AI_BENCH_MODEL` / `AI_BENCH_URL` — see
`src/bin/ai_test_harness/bench_util.rs`), so the same tasks target Ollama,
llama.cpp, LM Studio, vLLM, … or a cloud provider.

| Category | What it measures | Tasks |
|---|---|---|
| `code_gen_bench` | pass@1 on standalone functions: spec → code → run against checker. | 11 |
| `agentic_code` | a live model drives the built-in `AutonomousAgent` over `write_file`/`read_file`/`run_python` tools to build & fix code in a temp workspace. | 5 single-step |
| `agentic_multi` | multi-step iterative coding (build → extend → fix) on **one persistent workspace** (the agent's conversation carries across steps). | 2 × 3 steps |

## How to run

```powershell
# one category, one model
$env:AI_BENCH_MODEL = "qwen2.5-coder:7b-instruct"
.\target\debug\ai_test_harness.exe --category=agentic_multi

# a different backend (same tasks)
$env:AI_BENCH_PROVIDER = "llamacpp"; $env:AI_BENCH_URL = "http://127.0.0.1:8080"

# per-step debug for the agentic categories (dumps each step's model reply + artifact)
$env:AGENTIC_DEBUG = "1"
```

A **sweep** is just a loop over `AI_BENCH_MODEL`. Build the harness with
`cargo build --bin ai_test_harness --features "full,browser"` first.

## How to read / caveats

- **Execution-verified**: PASS = the generated code ran and passed its checker.
- **Temperature 0**, but Ollama is not perfectly deterministic — expect ±1
  run-to-run noise on borderline tasks. Re-run 3× before trusting a single number.
- **Confounds to avoid when authoring tasks:** a solution whose *code* contains a
  `"` character is hard for a model to emit through the JSON tool-call protocol
  (it must escape `\"` and local models often break the JSON) — that measures
  escaping, not coding. Prefer quote-free / single-quote solutions. Phrase agentic
  tasks as "create/write the file" (reliably triggers the tool) rather than
  "implement this stub" (models tend to answer in prose).

---

## Log (newest first)

### 2026-07-26 — first full sweep (harness @ V238 / 0.2.190)

**Setup:** Ollama, 6 local models, temperature 0. Tasks as of V238
(code_gen 11, agentic_code 5, agentic_multi 2).

| Model | code_gen (11) | agentic single (5) | agentic multi (2) |
|---|---|---|---|
| llama3.2:1b | 9/11 | 1/5 | 0/2 |
| qwen2.5:1.5b-instruct | 10/11 | 2/5 | 0/2 |
| gemma2:2b | 10/11 | 3/5 | 0/2 |
| llama3.2:3b | 11/11 | 2/5 | 1/2 |
| qwen2.5-coder:7b-instruct | 11/11 | 5/5 | 2/2 |
| llama3.1:8b | 11/11 | 5/5 | 2/2 |

**Findings:**
- **Single-function code-gen saturates at 3B** — everyone ≥3B is 11/11; it does
  not discriminate capable models.
- **The agentic loop discriminates; multi-step most of all.** Iterative
  build→extend→fix separates cleanly: nothing ≤2B sustains it (0/2), llama3.2:3b
  manages half (1/2), only 7–8B do both. Notably qwen2.5:1.5b does some single-step
  (2/5) but 0/2 multi-step.
- **Usable-as-a-coding-agent threshold (local, these tasks): ~7B coder / 8B.**
- Two **harness bugs** fixed this session (not model faults): the model
  hallucinating the rest of the transcript after its tool-call array (→ keep only
  the first balanced JSON array, `first_json_array`), and the JSON-quote confound
  above (→ replaced a CSV-split task with run-length-encode).

**Commits:** V234–V238 (0.2.186 → 0.2.190).

**Next:** richer tools (`run_command`, `list_dir`); longer multi-step chains; a
**Claude ceiling** (same tasks via the Anthropic provider); harder tasks
(SWE-bench-style repo bugfixes).

<!--
### YYYY-MM-DD — <what changed> (harness @ VNNN / 0.2.x)

**Setup:** <backend, models, temperature, task counts>

| Model | code_gen (N) | agentic single (N) | agentic multi (N) |
|---|---|---|---|
| … | … | … | … |

**Findings:** …

**Commits:** …

**Next:** …
-->
