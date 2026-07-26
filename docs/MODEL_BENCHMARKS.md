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
| `agentic_multi` | multi-step iterative coding (build → extend → fix) on **one persistent workspace** (the agent's conversation carries across steps). | 10 (3–5 steps each) |
| `agentic_rust` | same agentic loop, but in **Rust**: a throwaway cargo crate per task, verified with `cargo test` so the type/borrow checkers gate every answer. | 12 single-step |
| `agentic_rust_multi` | multi-step Rust on one persistent crate (incl. refactoring a concrete type into a generic one). | 6 × 3 steps |

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

### 2026-07-27 (4th) — second scaffolding lever, same answer: sampling doesn't rescue either (harness @ V244 / 0.2.196)

**Setup:** `agentic_rust` (12 tasks), llama3.2:3b, 100% GPU. New knob
`AI_BENCH_SAMPLES=n` runs *n* **independent** attempts (fresh crate, fresh agent, no
memory of the failure) and passes if any succeeds — best-of-N — with
`AI_BENCH_TEMP` raised so the samples actually differ. This is a *different* lever
from the previous entry's verify→retry: "more shots at the target" instead of
"here is your error, try again".

| Strategy (llama3.2:3b, 12 Rust tasks) | Result | Compute |
|---|---|---|
| single attempt, temp 0 (baseline) | 1/12 | ×1 |
| verify→retry ×3 (previous entry) | 2/12 | ×3 |
| **best-of-3 @ temp 0.7** | **2/12** | ×3 |

**Finding — two independent scaffolding strategies produce the identical marginal
gain.** Both move the 3B from 1/12 to 2/12 for 3× the compute. That the two agree so
exactly is the useful part: it rules out "we picked the wrong kind of scaffolding" as
the explanation. **The 3B's failures are capability failures** — it cannot express
these programs in Rust at all, so neither showing it the compiler error nor giving it
more attempts helps. Scaffolding multiplies a model's chances of *finding* an answer
it could produce; it does not create competence that isn't there.

**Consequence for the BACKLOG's strategy.** The bet that "agentic scaffolding
compensates for a weaker local model, at the cost of time" now has two failed tests
behind it. The workable version of the differentiator looks different: **pick a model
that is already capable** (on this box the 30B MoE — see the entry below) and use
scaffolding to raise *its* reliability, rather than hoping scaffolding lifts a small
model into the capable band.

Still untested, and the only honest remaining route to the original claim: levers that
add **information** rather than attempts — RAG over the codebase, multi-agent review
with a critic, Chain-of-Verification.

**Commits:** V244 (0.2.196).

### 2026-07-27 (3rd) — Rust multi-step doubled (3 → 6 tasks): the MoE separates cleanly (harness @ V243 / 0.2.195)

**Setup:** `agentic_rust_multi` grown from 3 to 6 tasks (added: builder pattern with
validation, matrix ops accumulating over 3 steps, and trait-then-generic-over-it).
All three checkers validated against reference implementations first. Ollama,
temperature 0, GPU state checked.

| Model | Rust multi (3 tasks, previous entry) | **Rust multi (6 tasks)** |
|---|---|---|
| qwen2.5-coder:7b | 2/3 | **3/6** |
| qwen3-coder:30b (MoE) | 3/3 | **6/6** |

**Findings:**
- **Doubling the task count confirmed what 3 tasks only hinted at.** The MoE is
  perfect (6/6) while the 7B drops to half (3/6) — a far more decisive gap than
  3/3 vs 2/3, which was within noise. This is the third time in this notebook that
  a larger task set changed the reading; treat small-N rankings as provisional.
- **For iterative Rust work on this box, `qwen3-coder:30b` is the model to use**,
  despite not fitting in 16 GB — the MoE's ~3B active parameters make the CPU spill
  affordable in a way a dense model of that size never would be.

**Commits:** V243 (0.2.195).

**Next:** the untested scaffolding levers (RAG over the codebase, multi-agent review,
self-consistency) after the verify→retry negative result below; Claude ceiling
still deferred (no API credits).

### 2026-07-27 (2nd) — **negative result**: verify→retry scaffolding does not rescue weaker models (harness @ V242 / 0.2.194)

**Setup:** `agentic_rust` (12 tasks), Ollama, temperature 0, 100% GPU. New knob
`AI_BENCH_SCAFFOLD=n` gives the agent *n* verify→feedback→retry rounds: after each
attempt the crate is compiled and tested, and on failure the model receives the real
cargo output (never the checker source) and is asked to fix `src/lib.rs`.

| Model | scaffold=1 (single shot) | scaffold=3 |
|---|---|---|
| llama3.2:3b | 1/12 | 2/12 |
| llama3.1:8b | 10/12 | 10/12 |

**Finding — the BACKLOG's central hypothesis is NOT supported here.** The bet was
that *"el andamiaje agéntico compensa un modelo local más flojo, a costa de tiempo"*.
On this task set it does neither: it does not rescue the incapable model (+1 task out
of 12, at 3× the time) and does not close the last gap for the model that is already
close (no change at all).

**Most likely why:** the agent *already* self-corrects inside its 6 loop iterations —
it can run `cargo test` and read the compiler's errors unaided — so an extra outer
round carries no new information. What remains are **capability** limits, not slips
that feedback can fix. Retry helps when a model *knows* the answer and slipped; it
does not teach a 3B model to write Rust it never could.

**Scope of the claim (important):** this tests ONE form of scaffolding
(verify→retry) on ONE task set. The other levers the backlog names — RAG over the
codebase, multi-agent roles, Chain-of-Verification, quality gates, self-consistency —
are untested and may behave differently. The honest conclusion is narrow: *execution
feedback alone is not the lever that closes the gap to big models.*

**Commits:** V242 (0.2.194).

**Next:** test a *different* scaffolding lever (multi-agent review, or self-consistency
over N samples) before accepting or rejecting the broader hypothesis.

### 2026-07-27 — Rust enters the benchmark + model shootout (harness @ V241 / 0.2.193)

**Setup:** Ollama, temperature 0, every model checked with `ollama ps`. New
categories `agentic_rust` (12 single-step tasks) and `agentic_rust_multi` (3 tasks ×
3 steps), verified by **`cargo test`** — the compiler is part of the verifier, so the
model must satisfy the type and borrow checkers before an assertion ever runs. All
12 checkers were validated against reference implementations first, and `cargo test`
was confirmed to actually run them (a silent "running 0 tests" would have turned
every task into a false PASS).

Also finished the enlarged Python multi-step set (10 tasks).

| Model | Py multi (10) | Rust single (12) | Rust multi (3) | GPU split |
|---|---|---|---|---|
| llama3.2:3b | 1/6* | 1/12 | — | 100% GPU |
| llama3.1:8b | 6/10 | 10/12 | — | 100% GPU |
| qwen2.5-coder:7b | **8/10** | 12/12 | 2/3 | 100% GPU |
| qwen2.5-coder:14b | — | 12/12 | 2/3 | 20%/80% CPU/GPU |
| qwen3-coder:30b (MoE) | — | 12/12 | **3/3** | 32%/68% CPU/GPU |

\* from the previous 6-task set.

**Findings:**
- **Rust discriminates far harder at the low end than Python.** llama3.2:3b scores
  **1/12** on single-step Rust versus 2/5 on the equivalent Python set. The compiler
  rejects what a Python interpreter would happily run.
- **…but ≥7B still saturates single-step Rust** (12/12 for all three coder models),
  so — exactly as with Python — **multi-step is what separates the top**. Only the
  30B MoE solved all three multi-step Rust tasks, including the one that forces
  rewriting earlier code (turn a concrete `Stack` into `Stack<T>`).
- **Bigger task sets flip conclusions.** On 6 Python multi-step tasks llama3.1:8b
  (6/6) looked better than qwen2.5-coder:7b (5/6); at 10 tasks the coder wins
  decisively, **8/10 vs 6/10**. Treat any ranking drawn from <10 tasks as noise.
- **MoE beats a dense model of half the size, on the same VRAM.** `qwen2.5-coder:14b`
  and `qwen3-coder:30b` both occupy ~18 GB loaded (note: the 9 GB on-disk figure is
  misleading), so both spill onto CPU on a 16 GB card. Despite spilling *more*
  (32% vs 20% CPU), the MoE was both **faster** (470 s vs 586 s on the same 3 tasks)
  and **better** (3/3 vs 2/3) — only ~3B parameters are active per token.
- **Practical recommendation for this box (RTX 4080 SUPER, 16 GB):**
  `qwen2.5-coder:7b` for fast iteration (fits entirely in VRAM, 56 s for the same 3
  tasks, 12/12 single-step); `qwen3-coder:30b` when quality matters most.
  `qwen2.5-coder:14b` is dominated — same footprint as the MoE, slower and weaker.

**Commits:** V239–V241 (0.2.191 → 0.2.193).

**Next:** more Rust multi-step tasks (3 is too few to rank the top models); the
backlog's untested hypothesis that agentic scaffolding rescues a weaker model;
Claude ceiling (still no API credits).

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
