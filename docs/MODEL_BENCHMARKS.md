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
| `agentic_code` | a live model drives the built-in `AutonomousAgent` over workspace tools (`write_file`, `read_file`, `run_python`, `list_dir`, `run_command`) to build & fix code in a temp workspace. | 5 single-step |
| `agentic_multi` | multi-step iterative coding (build → extend → fix) on **one persistent workspace** (the agent's conversation carries across steps). | 6 (3–4 steps each) |

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
- **Check the model is fully on GPU before trusting a sweep.** `ollama ps` shows a
  `PROCESSOR` column: anything other than ~100% GPU means layers were offloaded to
  CPU because VRAM was busy (desktop apps count!). That inflates times ~10× *and*
  flips results via request timeouts — silently. `nvidia-smi` shows who is holding
  the VRAM. **Freeing VRAM is not enough**: the split is decided when the model
  loads, so run `ollama stop <model>` to force a reload once the GPU is free
  (measured: 166 s → 21.9 s on the same task).
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

### 2026-07-26 (4th) — **clean full sweep**, 6 models × 3 categories (harness @ V240 / 0.2.192)

**Setup:** Ollama, temperature 0, **all models verified 100% GPU** (`ollama ps`)
after freeing VRAM — see the gotcha below. Tasks: code_gen 11, agentic_code 5,
agentic_multi 6. This entry supersedes the invalidated run below.

| Model | code_gen (11) | agentic single (5) | agentic multi (6) |
|---|---|---|---|
| llama3.2:1b | 9/11 | 1/5 | 0/6 |
| qwen2.5:1.5b-instruct | 10/11 | 2/5 | 1/6 |
| gemma2:2b | 10/11 | 1/5 | 1/6 |
| llama3.2:3b | 11/11 | 2/5 | 1/6 |
| qwen2.5-coder:7b-instruct | 11/11 | 5/5 | 5/6 |
| llama3.1:8b | 11/11 | 4/5 | **6/6** |

**Findings:**
- **The three categories form a clean difficulty ladder.** `code_gen` saturates
  from 3B up (11/11 for everyone ≥3B — useless for ranking capable models);
  `agentic_code` splits small vs. large; `agentic_multi` splits hardest.
- **Multi-step is where the tier boundary lives, and it is sharp.** Everything
  ≤3B lands at 0–1 of 6 — i.e. essentially *cannot* sustain iterative development —
  while 7–8B lands at 5–6 of 6. There is no middle tier in this model set.
- **llama3.1:8b is the only 6/6**, edging qwen2.5-coder:7b (5/6) on multi-step even
  though the coder model wins single-step (5/5 vs 4/5). With 5–6 tasks a one-task
  gap is within noise; the honest statement is *both 7–8B models are usable, the
  ≤3B ones are not*.
- **Now with 6 multi-step tasks the numbers are far less noisy** than the 2–3 task
  sets of earlier entries, which is what made the earlier "llama3.2:3b = 1/2"
  reading unstable.

**Gotcha discovered (worth its own line):** freeing VRAM is *not enough*. Ollama
decides CPU/GPU layer split **at load time**, so a model loaded while VRAM was
scarce stays partly on CPU even after the memory frees up. `ollama stop <model>`
to force a reload. Measured on the same task: **166 s → 21.9 s (7.6×)** purely from
reloading into a free GPU.

**Commits:** measurement only (harness unchanged since V240 / 0.2.192).

**Next:** more multi-step tasks to resolve the 7B-vs-8B gap; SWE-bench-style repo
bugfixes; Claude ceiling (still deferred — no API credits).

### 2026-07-26 (3rd) — 6 multi-step tasks + extraction fix; **run invalidated by VRAM starvation** (harness @ V240 / 0.2.192)

**Setup:** Ollama, qwen2.5-coder:7b-instruct. `agentic_multi` grew from 3 to 6
tasks (added: todo list class, matrix utils, and *word counter*, whose 3rd step must
**modify** earlier behaviour — make counting case-insensitive — rather than append).

**No sweep numbers published from this run.** Mid-run the box was VRAM-starved:
`nvidia-smi` showed **13.6 of 16 GB taken by desktop apps** (browser, editors, chat
apps, Steam), so `ollama ps` reported the 7B running **55%/45% CPU/GPU**. Effects:
per-task times went from ~14 s to 170–600 s, and one step died on
`Failed to send request to Ollama` (timeout). Passes observed are still valid
(execution-verified), but the one failure is confounded, so the set is not
comparable to earlier entries.

> **Lesson for this notebook: check `ollama ps` shows 100% GPU before trusting a
> sweep.** A partially CPU-offloaded model changes both timings and pass/fail
> (timeouts), silently.

**Findings (harness, not models):**
- **Extraction was brittle.** Instrumenting the debug dump with *reply length* +
  *balanced-array status* (rather than eyeballing truncated text) showed
  `first_json_array` locking onto the **first** `[` in a reply: when a model emitted
  a malformed array and *then* a well-formed one, the good call was discarded and
  the step silently became a no-op. Fixed by trying every `[` and requiring the
  candidate to actually **parse** (serde_json) and contain a `name` field. Verified:
  a step that yielded `tools=[]` now yields `tools=[write_file,write_file,run_python]`.
- **Deliberately not repairing malformed JSON**: a model that can't emit a valid
  tool call has genuinely failed the protocol; patching it would inflate scores.
  Small models *do* drop closing braces/brackets on longer payloads — that is a real,
  measurable limitation.

**Commits:** V240 (0.2.192).

**Next:** re-run the full sweep (3 categories × models) on a **clean GPU state**;
Claude ceiling still deferred (no API credits).

### 2026-07-26 (later) — richer tools + a 4-step chain (harness @ V239 / 0.2.191)

**Setup:** Ollama, temperature 0. Agents gained two tools (`list_dir`,
`run_command` — whitelisted python/python3/pytest, no shell). `agentic_multi` grew
a third task: **stats module**, a *4-step* build (mean → +median → +mode → +stdev),
longer than the existing 3-step chains.

| Model | agentic multi (3) |
|---|---|
| gemma2:2b | 0/3 |
| llama3.2:3b | 0/3 |
| qwen2.5-coder:7b-instruct | 3/3 |
| llama3.1:8b | 3/3 |

**Findings:**
- **The cut at ~7B is binary on this set**: small models solve *none*, 7–8B solve
  *all three* — including the 4-step chain, so length alone doesn't break the 7–8B
  models here.
- **Correction to the previous entry's reading**: llama3.2:3b went 1/2 → 0/3. With
  samples this small, its earlier 1/2 was borderline rather than a real
  "half-capable" tier. Robust statement: **3B does not reliably sustain multi-step**.
  More tasks per category are needed before per-model numbers deserve trust.

**Commits:** V239 (0.2.191).

**Next:** Claude ceiling still pending — **no `ANTHROPIC_API_KEY` in the
environment**, so the cloud comparison could not run. Also: more multi-step tasks
(to firm up the numbers), and SWE-bench-style repo bugfixes.

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
