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

Seven harness categories, all **execution-verified** (the code the model produces
is actually run against `assert` checkers — a PASS means the artifact *works*, not
that it "looks right"). The backend is configurable via env
(`AI_BENCH_PROVIDER` / `AI_BENCH_MODEL` / `AI_BENCH_URL` — see
`src/bin/ai_test_harness/bench_util.rs`), so the same tasks target Ollama,
llama.cpp, LM Studio, vLLM, … or a cloud provider.

For the concepts these measurements rest on — quantization and its naming, third-party
quantizers, the KV cache as the real VRAM constraint, context compression and hardware
number formats — see [LOCAL_MODELS.md](LOCAL_MODELS.md).

Sampling is pinned with `AI_BENCH_TEMP` (default **0.5**) and `AI_BENCH_SEED`
(default **42**, or `none` to randomise). Do **not** set the temperature to 0 to
chase determinism: the seed is what buys reproducibility, and near-greedy sampling
crashes the llama.cpp runner outright — see the 2026-07-31 (2nd) entry.

**All seven are repeated and scored as a pass rate** (`AI_BENCH_REPEATS`, default 3) since
V272 — a single live-model run is one sample, not a measurement.

| Category | What it measures | Tasks |
|---|---|---|
| `code_gen_bench` | pass@1 on standalone functions: spec → code → run against checker. | 11 |
| `agentic_code` | a live model drives the built-in `AutonomousAgent` over workspace tools (`write_file`, `read_file`, `run_python`, `list_dir`, `run_command`) to build & fix code in a temp workspace. | 5 single-step |
| `agentic_multi` | multi-step iterative coding (build → extend → fix) on **one persistent workspace** (the agent's conversation carries across steps). | 10 (3–5 steps each) |
| `agentic_rust` | same agentic loop, but in **Rust**: a throwaway cargo crate per task, verified with `cargo test` so the type/borrow checkers gate every answer. | 12 single-step |
| `agentic_rust_multi` | multi-step Rust on one persistent crate. Six **additive** tasks (each step adds to the last) plus four where a late step **invalidates** an earlier one — a rename whose callers must be re-pointed, an enum variant that breaks an existing match, an infallible API that becomes fallible. | 10 (3–5 steps each) |
| `agentic_edit` | the model gets an EXISTING crate (four modules plus a passing test suite) and a change request that states the symptom, never the file. Two gates, scored apart: the seeded tests must still pass, and the requested change must be there. | 2 |
| `agentic_test_gen` | **inverted**: the model gets a *correct* implementation and must write the test suite. Scored by mutation — its tests must accept the reference **and** kill every planted bug. | 12 |

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

These categories are **excluded from `--all`** (which is the regression gate and must
not go red just because the configured model is weak). Run them deliberately:

```powershell
.\target\debug\ai_test_harness.exe --benchmarks   # every model-measuring category
.\target\debug\ai_test_harness.exe --list         # marks which categories are benchmarks
```

Scaffolding knobs, all off by default so plain runs stay comparable:
`AI_BENCH_SCAFFOLD=n` (verify→retry rounds), `AI_BENCH_SAMPLES=n` + `AI_BENCH_TEMP`
(best-of-N), `AI_BENCH_KNOWLEDGE=1` (idiom injection), `AI_BENCH_CRITIC=1`
(multi-agent reviewer).

Two knobs that are *not* scaffolding and belong in every entry's header, because a
result means nothing without them: `AI_BENCH_REPEATS=n` (runs per task, default **3**,
scored as a pass rate — the categories that use it are noted in the table above) and
`AI_BENCH_NUM_CTX=n` (context window; shrinking it is what decides whether a big model
fits entirely in VRAM — see [LOCAL_MODELS.md](LOCAL_MODELS.md)). Both appear in the
`backend=` label the harness prints, so they end up in the log automatically.

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

## Safety and threat model

**This harness compiles and executes code written by a language model**, plus
allowlisted commands, on the machine that runs it. Be clear-eyed about what that
does and does not protect:

- **The command allowlists stop accidents, not attacks.** `run_command` is limited
  to `python` / `python3` / `pytest` (Python tasks) and `cargo build|test|check`
  (Rust tasks), and commands are executed directly — no shell — so there is no `;`,
  `&&` or pipe injection. But `python -c "..."` is arbitrary code execution by
  definition, and `cargo build` runs a `build.rs` **that the model itself can
  write**. Arguments after the subcommand are not validated either.
- **`safe_join` is containment, not a boundary.** It rejects traversal (`..`) and
  anything absolute — including the Windows drive-letter escape fixed in V250,
  where `Path::join` silently replaced the workspace for a path like
  `C:/Windows/x.txt`. That keeps *well-behaved* code inside its workspace; it does
  nothing against code that simply opens whatever path it likes.
- **Therefore the assumed threat model is: a non-adversarial local model, run by
  the author, on the author's own machine.** Under that assumption the design is
  fine, and the value is reproducibility rather than isolation.
- **If you ever point this at an untrusted, remote or prompt-injected model, run
  the tasks in a container** (the crate has a `containers` feature). Nothing in the
  current path would stop hostile generated code from touching the rest of the
  filesystem.

Housekeeping: each task removes its temp workspace when it finishes, but a run
killed mid-flight leaves the directory behind (six were found during this review).
The shared cargo target dir stays small (~9 MB — the task crates have no
dependencies), so disk growth is not a concern in normal use.

---

## Log (newest first)

### 2026-08-04 (5th) — RETRACTION: "never solves it" was one unlucky seed, and repeats never varied the seed

**Every "never" in the four entries below is unsafe, and two specific claims are wrong.**

The trigger was a reader's instinct — *the 14B failing all three runs sounds like a bug* —
so `ledger: an infallible API becomes fallible` was re-run at other seeds. qwen2.5-coder:14b,
`num_ctx` 4096, three repeats per seed:

| seed | result |
|---|---|
| 42 (the published sweep) | **0/3** |
| 7 | 1/3 |
| 1234 | **3/3** |

Pooled, p ≈ **0.44**. Not a capability boundary — an ordinary flaky task, drawn three times
from the one seed where it happened to lose.

**Two independent mistakes met to produce that verdict:**

1. **0 of 3 was read as p = 0.** At p ≈ 0.44 a run of three failures comes up about 17 % of
   the time, and after 0/3 the 97.5 % upper bound on p is ~0.6 (rule of three). Three
   samples cannot separate "never" from "sometimes"; the distribution line said `never 2`
   and it was believed.
2. **The repeats never sampled the seed.** Interleaving decorrelates KV-cache state — real
   variance, and worth catching — but the seed stayed frozen at 42 for every repeat. It
   turned out to dominate everything the interleaving was catching: *within* seed 1234 the
   task is 3/3, *within* seed 42 it is 0/3.

**What this retracts, precisely:**

- *"`state machine` and `ledger` are a capability boundary for the 14B, not a flaky band"* —
  **wrong for `ledger`** (p ≈ 0.44), and unverified for `state machine`, which was measured
  the same way.
- *"blind retry buys nothing on those tasks"* — **wrong**. It was computed from p̂ = 0, and
  at p ≈ 0.44 retrying recovers the task about 82 % of the time at k = 3.
- *"the 7B beats the 14B on `ledger`"* — the direction may hold, but both sides were single
  seeds, so the effect size means nothing yet.

**What survives:** the saturation results (11/11 and 12/12 with sd 0.00 are not going to be
overturned by a seed change), the aggregate ordering 7B < 14B < 30B, and every finding about
the harness itself. What needs re-measuring is anything resting on a *specific task* being
at 0 or 1.

**The fix (V275):** each repeat now strides the base seed, so three repeats are three
seeds. The sweep stays exactly reproducible — the sequence is deterministic given the base —
while the repeats finally vary the dimension that mattered most. Labels read `seed=42x3`
rather than `seed=42`, because "seed=42" had become a half-truth. The stride is the
golden-ratio constant rather than `+1`: seeds 42, 43 and 44 all failed `ledger` while 1234
passed it 3/3, and although three samples cannot show that adjacent seeds are correlated,
repeats landing in one neighbourhood is the exact failure this change exists to remove.

**And a limit worth stating, since it bit once already:** re-run at the strided seeds,
`ledger` came out **0/3 again**. Pooling every observation to date (12 runs) puts p nearer
**0.33**, with a wide per-seed spread. Varying the seed makes the samples independent; it
does not make three of them enough to characterise a single task. Read a per-task 0/3 as
"probably low", never as "never" — the aggregate over ten tasks is what three repeats are
actually sized for.

**The lesson, in one line:** *reproducible* and *independent* are different properties, and
pinning a seed buys the first by destroying the second.

### 2026-08-04 (4th) — re-baseline at 3 repeats: half the suite is saturated, and rank is not a total order

V272 moved four categories onto rate scoring but their published numbers were still
single-run. Re-measured properly (qwen2.5-coder:14b, `num_ctx` 4096, `AI_BENCH_REPEATS=3`),
plus a 7B floor on the widened multi-step set:

| category | 14B @ 3 repeats | shape |
|---|---|---|
| `code_gen_bench` | **11.00/11** | always 11, sd 0.00 — **saturated** |
| `agentic_rust` | **12.00/12** | always 12, sd 0.00 — **saturated** |
| `agentic_multi` | 9.50/10 | always 9, sometimes 1 (5 of 30 runs lost to the backend) |
| `agentic_code` | 4.00/5 | always 4, **never 1** |
| `agentic_rust_multi` | 7.67/10 | always 7, sometimes 1, never 2 |

**Two of six categories no longer say anything about a capable model.** Single-function
generation and single-step Rust are both perfect with zero variance, so against anything
≥14B they cost wall clock and return no information. Run them to place small models; skip
them when comparing serious ones. The signal lives in the multi-step categories, in the one
`agentic_code` task the 14B never solves, and in `agentic_test_gen`.

#### The floor, and a result that breaks the ordering

`agentic_rust_multi` (widened, 10 tasks) across three models, same tasks and verifier:

| model | score | shape | runs lost | blind-retry k=3 |
|---|---|---|---|---|
| `qwen2.5-coder:7b-instruct` | 6.67/10 (mean 0.67, sd 0.39) | always 5, **sometimes 3**, never 2 | 5/30 | **7.71** |
| `qwen2.5-coder:14b` | 7.67/10 (mean 0.77, sd 0.40) | always 7, sometimes 1, never 2 | 0/30 | 7.96 |
| `qwen3-coder:30b` | 9.50/10 (mean 0.95, sd 0.15) | always 9, sometimes 1, **never 0** | 3/30 | 9.88 |

The aggregate ordering is clean, and **the per-task ordering is not**: the 7B solves
`ledger: an infallible API becomes fallible` **3/3**, a task the 14B fails **0/3**. The 14B's
failure there is not reasoning — it emits `Ok(()`, an unclosed delimiter, and never repairs
it across four steps of running `cargo test`. A bigger model losing a task to an emission
quirk is exactly what a single aggregate number hides, and it is why "just pick the bigger
model" is advice, not a law. Check the per-task lines before believing a ranking.

Note also what the projections are for. The 7B has **three tasks in the middle band**, so
blind retry would take it from 6.67 to ~7.71 — a whole task recovered. The 14B gains 0.29
and the 30B 0.38. **Retrying pays for a model at its limit and is close to free money
wasted on a capable one**, which is the sharp form of the settled result that scaffolding
does not create capability.

#### The backend died mid-sweep, and the harness said "ALL 0 TESTS PASSED"

Ollama degraded through the sweep (5 of 30 `agentic_multi` runs ended in `BACKEND CRASH`,
across 4 tasks) and then the daemon **died outright**. The next category found it
unreachable, printed `SKIP`, and the run exited 0 with `ALL 0 TESTS PASSED [1 skipped]`.

Every piece is individually right — skipping when there is no backend is what lets the
battery run on a machine with no GPU — but **the combination lies**: a sweep of N
categories whose daemon dies in the second produces N-1 silent skips and a green summary.
Queued as a fix: if the backend WAS reachable at the start, a later "unreachable" skip is
not a skip, it is an invalid sweep, and it must exit non-zero with a banner.

### 2026-08-04 (3rd) — the multi-step set discriminates again: four tasks that INVALIDATE earlier work

The entry below closed with "the set is spent": the 14B scored 6.00/6, sd 0.00, on the six
multi-step Rust tasks, so the category could no longer rank anything. The diagnosis was
that all six are **additive** — every step adds to what the last one built, so a model
that can write each piece in isolation passes without ever revisiting a decision.

Four tasks were added where a late step **invalidates** an earlier one:

| task | what the late step breaks |
|---|---|
| counter: rename a method and re-point its callers | `add` becomes `bump` and starts returning a value; every call site must change |
| state machine: a new variant breaks the old match | a third enum variant makes an existing match non-exhaustive and changes the cycle |
| ledger: an infallible API becomes fallible | `push` starts returning `Result`; `apply` must stop at the first failure |
| scores: concrete function becomes generic over a late trait | a concrete `best(&[Player])` must become `best<T: Scored>` |

Their oracles were **audited before any model saw them** (12 mutants, reference must pass
and every mutant must fail — the `checker_adequacy` category). The audit paid for itself
immediately: `count_running` counting *everything that is not Idle* passed the first
version of its checker, because the test slice happened to contain no `Paused`. The
separating case was missing — the same shape as every weak oracle found in V256 and V264.

**Result — qwen2.5-coder:14b, num_ctx 4096, 3 repeats over the widened 10-task set: 7.67/10,
mean 0.77, sd 0.40 — always 7, sometimes 1, never 2.** The category ranks again. The two it
never solves (0/3 each, consistent across every run) are both invalidation tasks, and the
debug dumps say why:

- **state machine** — by step 4 it had **dropped the `#[derive(...)]` the first step asked
  for**, so the crate does not even compile (`==` on an enum with no `PartialEq`). Losing
  an instruction from step 1 by step 4 is exactly the capability the category exists to
  measure.
- **ledger** — the logic is **right** (negatives rejected, `apply` stops at the first one,
  `sum` goes through `entries`), and it fails on `Ok(()` — an unclosed delimiter, written
  twice and never repaired despite four steps of being told to run `cargo test`.

Neither is a broken task: both were reproduced against a reference implementation that
passes. Note also what the projection says — with both tasks at p = 0, blind retry projects
only 7.89 at k=2 and 7.96 at k=3, i.e. it recovers a third of one task and nothing else.
This is a capability boundary, not a flaky band.

**And a caveat on the entry below.** In this sweep one of the SIX ORIGINAL tasks —
`errors: Option then custom error type` — came out 2/3, where the previous sweep had it at
3/3 with the whole set at sd 0.00. Three repeats is enough to stop single runs lying; it is
not enough to make a sweep exactly reproducible. Read "6.00/6, sd 0.00" as "at the ceiling
within measurement noise", not as a constant — the same reason a one-task gap between two
models could not be trusted in the first place.

**One confound was removed before publishing this number.** The new prompts said "a public
function `next(s: State) -> State`" where the older tasks say "a public **free** function",
and the model put everything inside an `impl` block. The wording was aligned and the
sweep re-run, so the figure above is measured against the wording now in the tree.

#### And the 30B, on the same widened set: 9.50/10 — it solves BOTH tasks the 14B never does

| model | score | distribution | runs scored | wall clock |
|---|---|---|---|---|
| `qwen2.5-coder:14b` (num_ctx 4096) | 7.67/10 (mean 0.77, sd 0.40) | always 7, sometimes 1, **never 2** | 30/30 | 938 s |
| `qwen3-coder:30b` (num_ctx 2048) | **9.50/10** (mean 0.95, sd 0.15) | always 9, sometimes 1, **never 0** | 27/30 | 2 758 s |

`state machine` and `ledger` — the two the 14B fails 0/3 — the 30B solves **3/3 each**. That
is a capability difference held across three runs on both sides, not a one-task gap between
two single runs, and it is exactly what the additive six could not show: on those, the two
models were indistinguishable at the ceiling.

So the earlier verdict stands but sharpens. **The 14B is the fast pick** — 2.9× quicker
here, fully resident, no runs lost — and it is enough for work that only ever *adds*. **The
30B is the one to reach for when an edit invalidates earlier code**, which is what real
refactoring is. Its own weak spot in this sweep was the late-generic task at 1/2, and it
again lost runs to the runner aborting (3 of 30, across 3 tasks).

Note the projections: 9.75/9.88 for the 30B (one flaky task, so retrying recovers a little)
against 7.89/7.96 for the 14B (two tasks at p = 0, which no amount of retrying moves).

### 2026-08-04 (2nd) — 14B vs 30B, settled with repeats: both at ceiling, and the set is spent

The previous entry left a one-task gap (14B 5/6, 30B 6/6) and said explicitly that one
task in six, from single runs, is the ±1 noise band — not a result. `agentic_rust_multi`
now scores a pass rate over interleaved repeats (V271), so the question is answerable.
Same binary, same session, `AI_BENCH_REPEATS=3`, `ollama stop` between models:

| Model | `num_ctx` | Placement | Score | Runs scored | Wall clock (3 passes) |
|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 4096 | 5 % / 95 % CPU/GPU, 13 GB | **6.00/6** (mean 1.00, sd 0.00) | 18/18 | **636 s** |
| `qwen3-coder:30b` (MoE) | 2048 | 33 % / 67 % CPU/GPU, 19 GB | **6.00/6** (mean 1.00, sd 0.00) | 14/18 | 2 113 s |

**The gap was noise.** The 14B solved every task in all three passes, including
`errors: Option then custom error type`, the one it "failed" last night. The 30B was not
better; it was luckier once, and it is **3.3× slower** on the same six tasks.

**The 30B lost 4 of its 18 runs to backend crashes** (2 on `stack: concrete then generic`,
1 each on `errors` and `builder`), which is why its score rests on 14 runs. This is the
first time the crash accounting mattered on a real sweep: had those four counted as
failures, the 30B would have "scored" 4.33/6 and the entry would have recorded a
capability difference that does not exist. The `LOST` line exists because `1/1` and `3/3`
both print `score=1.00`.

**Practical verdict for multi-step Rust on a 16 GB card:** `qwen2.5-coder:14b` at
`num_ctx` 4096. Fully-enough resident, three times faster, zero runs lost. The 30B keeps
its edge in `agentic_test_gen` (11.00/12 vs 9.67), which is a harder category — that is
where the two models still separate.

**And the set is spent.** A category where the mid-size model is right *every single time*
(sd 0.00, no task in the middle band) cannot rank models any more, and its blind-retry
projection equals its score, so there is nothing to recover either. Harder multi-step
tasks are queued: more steps, and refactors that invalidate an earlier decision rather
than extend it.

**Two smaller notes.** The 14B loaded at 5 % CPU tonight, not the 100 % GPU recorded
earlier at the same `num_ctx` — the split depends on what else holds VRAM *at load time*,
so treat "100 % GPU at 4096" as "fits when the desktop is quiet", not as a constant. And
the 30B ran at 33 % CPU vs 27 % before, for the same reason.

### 2026-08-04 — the 14B is not "dominated": that verdict rested on a premise `num_ctx` removed

An earlier entry concluded `qwen2.5-coder:14b` was **dominated** by the 30B MoE, on the
argument that both loaded at ~18 GB and spilled to CPU, so the MoE was strictly better
at the same footprint. **That premise is gone.** At `num_ctx` 4096 the 14B loads at
13 GB, **100 % on GPU** — the KV cache, not the weights, was what pushed it over
(V269 / LOCAL_MODELS §3). So the comparison had to be re-run with both models measured
the same way, on the same six tasks, rather than inherited from a 3-task set.

| model | `agentic_rust_multi` | wall clock | GPU |
|---|---|---|---|
| qwen2.5-coder:14b, `num_ctx` 4096 | 5 / 6 | **235 s** | **100 %** |
| qwen3-coder:30b, `num_ctx` 2048 | **6 / 6** | 554 s | 73 % (27 % on CPU) |

The 30B solves exactly the task the 14B fails (`errors: Option then custom error type`,
where the 14B lost state across edits — the multi-step failure mode). It is also
**2.4× slower**.

**What this does and does not establish.** One task in six is precisely the ±1 noise
band measured for this kind of pass/fail scoring, and these are single runs with no
repeats. So the defensible statement is *"the 30B is at least as good"*, not *"the 30B
is better"* — a rank ordering on a one-task gap from one run each is exactly the
over-reading this notebook exists to prevent.

What is solid: **"dominated" is wrong now.** The 14B is the fast, fully-resident option;
the 30B is the quality pick when the extra 2.4× is affordable. Settling the capability
question properly needs repeats on both — queued rather than guessed.

### 2026-08-03 (2nd) — the Python oracles were never audited, and four of eleven were incompetent (harness @ V264 / 0.2.216)

`checker_adequacy` (V256) mutation-tested the **Rust** checkers and found two of twelve
unable to reject a plausible wrong answer. The **Python** checkers are older and had
never been audited at all. `python_adequacy` applies the same test, deliberately using
the *same checker text and the same runner* the benchmark uses — re-implementing either
would audit a copy rather than the thing.

**5 of 33 checks failed, across four tasks.** Each accepted an implementation a model
would plausibly write:

| task | what slipped through |
|---|---|
| `has_close_elements` | **both** mutants: no input sat exactly *on* the threshold, so `<=` passed a strictly-less-than spec; and every close pair was adjacent, so comparing only neighbours passed |
| `reverse_words` | every string had single spaces, so `s.split(' ')` — which differs from `s.split()` only on runs of whitespace — went unnoticed |
| `is_prime` | 0 and 1 were covered but nothing negative, so guarding only `n in (0, 1)` reported −7 as prime |
| `longest_common_subsequence` | LCS length coincidentally equalled the shared-character count in every case, so an implementation **ignoring order entirely** passed. `'ab'` vs `'ba'` separates them: LCS 1, shared characters 2 |

The pattern is the same every time: **a missing separating case**. The inputs never
distinguished right from almost-right.

All four are fixed and the audit passes 33/33. It runs inside `--all` (now 692 tests),
so they cannot rot again.

**Two consequences.**

1. **`code_gen_bench` scores from before V264 are not comparable.** A model could score
   a task with a wrong answer, so those numbers contained noise of unknown size.
2. **Python came out worse than Rust (4/11 vs 2/12), and that is not an accident.** Rust
   tasks are gated by a compiler and `cargo test`, which reject whole classes of
   wrongness for free; a Python assert list catches only what someone thought to write
   down. Expect the weakest oracles wherever the language does least for you.

#### Re-measured with the fixed oracles — and the old conclusion survives

| model | `code_gen_bench` pass@1, post-V264 |
|---|---|
| qwen2.5-coder:14b | **11/11** |
| qwen2.5-coder:7b-instruct | **11/11** |
| llama3.2:3b | **11/11** |

So the weakness was **latent**: the oracles could have been satisfied by a wrong answer,
but these models were not writing wrong answers — they wrote correct implementations and
would have passed either way. The earlier finding that **`code_gen_bench` saturates from
~3B upward** stands, and is now resting on checkers that have been shown to reject the
obvious wrong answers rather than merely assumed to.

Worth stating plainly, because it is the good outcome and the less memorable one: **a
repaired instrument that changes none of the numbers has still earned its keep.** It
converts "we believe this" into "we checked this", and it means the *next* model — the
one that does write `s.split(' ')` — will be caught.

#### Operational note: Ollama does not survive a suspend

A scheduled sleep leaves the Ollama server down, and it does not come back on wake. The
harness then reports `SKIP … backend not reachable` and the category passes as **0/0** —
green, and measuring nothing. An unattended overnight sweep can therefore "succeed"
having run zero tasks. Check `curl -s localhost:11434/api/tags` (or that the summary line
shows a task count at all) before trusting an overnight result.

### 2026-08-03 — reading the shape instead of the total, and the bar a repair loop must clear (harness @ V263 / 0.2.215)

Three lines were added to the `agentic_test_gen` summary. All are computed from
data the repeats already produce, so they cost nothing and apply retroactively to
any future sweep.

**Distribution.** A total hides the shape: 6/12 can mean six tasks solved reliably
and six never, or twelve solved half the time — different models to work with. On
qwen2.5-coder:7b:

```
mean rate 0.44 (sd 0.44) — always 4, sometimes 3, never 5
```

The standard deviation equalling the mean is the signature of a **bimodal** model,
and the always/sometimes/never split says it outright.

**The blind-retry projection — the important one.**

```
blind-retry projection: 6.00 at k=2, 6.37 at k=3 (of 12)
```

Attempts are independent, so a task solved with probability `p` succeeds at least
once in `k` tries with probability `1-(1-p)^k`. That is what **simply buying more
lottery tickets** would score. It is therefore the bar any feedback-driven repair
has to clear to have earned its complexity: **a repair loop that merely matches
this number has proved the compiler output added nothing.**

It also bounds the ceiling. A task at `p = 0` stays at 0 for every `k`, so retrying
recovers the *inconsistent band* and nothing else. That is the sharp, quantified
form of the earlier settled result that scaffolding does not create capability —
and it retro-explains it: the identical "+1" that four separate levers produced on
the 3B was almost certainly one flaky task getting a second chance, not a new
capability.

Measured here: **5.33 → ~6.4**, with five tasks no amount of retrying can move.

**Failure modes.** Which way the suites are wrong is more actionable than how many:

```
4 rejected valid code, 1 too weak to catch the bug, 1 produced no tests, 2 backend crash
```

Counted across every task with at least one failing run, not only the tasks that
failed outright — a task solved 2 times in 3 still failed once, and how it failed
is the same evidence.

#### Three models, and what the distribution says that a total does not

| model | score | always / sometimes / never | mean (sd) | blind-retry k=3 | GPU |
|---|---|---|---|---|---|
| qwen2.5-coder:7b-instruct | 5.33–7.33 / 12 | 4 / 3 / 5 | 0.44 (0.44) | ~6.4 | 100 % |
| qwen2.5-coder:14b | 9.67 / 12 | 10 / 1 / 1 | — | — | 100 % (`num_ctx` 4096) |
| **qwen3-coder:30b** (MoE) | **11.00 / 12** | **11 / 0 / 1** | **0.92 (0.28)** | 11.00 | 73 % (27 % on CPU) |

> The 30B row was re-measured at **3 repeats** to match the others (it was 2, which the
> first version of this entry flagged as its weakest point). The score moved
> 11.50 → 11.00 — inside the ±1 band — because `enum + match evaluator` went from 1/2 to
> **0/3**: at two samples it looked borderline, at three it is a consistent failure.
>
> **Its blind-retry projection is 11.00 at k=2 *and* k=3 — identical to the score.**
> That is the sharpest thing this metric can say: with no task in the middle band, every
> task is either always solved or never solved, so **retrying buys literally nothing on
> this model**. Do not build a retry loop for it; there is nothing there to recover.

**The category discriminates cleanly across all three**, and the gaps are far outside
the ±1 noise band.

**Capability and consistency arrive together — now confirmed three times.** The 7B is
smeared across 0/3, 1/3, 2/3 and 3/3; the 14B is nearly decisive; the 30B has
**no task at `p = 0` at all**. Its blind-retry projection is 11.88, which is the
clearest possible statement of where retry helps: there is almost nothing left for
it to recover, because there is no band of incompetence — only one task on the edge.

The 30B also solves **both tasks that defeated the other two** (`explicit lifetimes`
and `dedup preserving first-appearance order`, 2/2 each), and it did so while a
quarter of it ran on CPU, with **zero runs lost**.

One caveat remains on this row: **the first sweep's timings were contaminated** —
this session was compiling during part of it. No run was excluded and nothing timed
out, so the scores stood; the `SLOW` tags did not. The 3-repeat re-run above was done
with nothing else building, so its timings are usable (and show the cost of offload:
53–112 s per task, against ~40 s for the fully-resident 14B).

This strengthens the earlier finding from the Rust categories: **on a 16 GB card the
30B MoE beats the dense 14B even while partly on CPU.** A mixture-of-experts activates
only a few experts per token, so offload costs it far less than it would a dense model.

#### A measurement hygiene note, learned the hard way again

While a `qwen3-coder:30b` sweep was running, this session kept compiling and
running the test suite. With **25 % of that model on CPU**, compilation steals the
very cores its inference needs, inflating latencies and risking the client's 120 s
ceiling — which would surface as excluded runs and a meaningless score. This is
the CPU-offload trap from the 2026-07 entries, entered from the other side: not a
badly loaded model, but a well-loaded model with a busy machine. **Do not build
while measuring.**

### 2026-08-01 — test generation: the corpus completed, and repeats that actually sample the noise (harness @ V259 / 0.2.211)

**What this category asks.** Every other category hands the model a spec and judges the
implementation with *our* tests. This one inverts it: the model receives a **correct**
implementation and must write the test suite, judged exactly the way we judge our own
oracles — its suite must **accept the reference** *and* **kill every planted mutant**.
Both halves matter. A suite that only checks the obvious case passes on broken code; a
suite that invents requirements rejects correct code. Both are worthless, and a naive
"did it write tests?" check would call both a success.

This is the capability that decides whether an agent's self-verification means
anything: an agent that writes weak tests runs them, sees green, and reports success on
broken work.

**Corpus completed** to all 12 `ADEQUACY` tasks (was `.take(8)`, a leftover cap).
`borrow checker: dedup in place` was renamed to **`dedup preserving first-appearance
order`** — it exercises no borrow-checker skill; what it tests is knowing that
`Vec::dedup` only removes *consecutive* duplicates and that first-appearance order must
survive. Entries above this date use the old name.

#### The methodology fix, including the one I got wrong first

A single run is not a measurement — the previous entry quantified ~±1 task of noise.
So each task now runs `AI_BENCH_REPEATS` times (default 3) and scores `passes/repeats`,
with inconsistent tasks listed under a `FLAKY` line.

**The first attempt was wrong in a way that looked like success.** Running the three
repeats back to back reported **zero flaky tasks** — every task scored exactly 1.00 or
0.00, which reads as "the noise is gone". It wasn't: consecutive repeats of one task hit
the backend with near-identical KV-cache state, so they are *correlated samples*. They
agree with each other and hide precisely the variance being measured. Two separate
invocations still disagreed (6.00 vs 5.67).

Interleaving the repeats — pass 1 of every task, then pass 2 — puts eleven other tasks
through the server between one task's samples. Same 36 runs, same cost, and the noise
becomes visible:

| repeat layout | flaky tasks surfaced |
|---|---|
| back to back | **0** |
| interleaved | **4** and **6** (two measurements) |

**Why not just make it deterministic?** Partly because we can't from the client side
(the seed is spent; what remains is llama.cpp's floating-point non-associativity under
varying cache reuse and batch splits). But mostly because we shouldn't: some of the
flapping is the model sitting *at its limit* on that task. A deterministic backend would
freeze the coin on one face — the task would look solid while the model actually solves
it half the time. The rate is the more truthful number. Backend levers exist if true
determinism is ever needed (`OLLAMA_NUM_PARALLEL=1`, or llama.cpp's server with
`cache_prompt: false` and `-np 1`, which the harness already supports via
`AI_BENCH_PROVIDER=llamacpp`), at the cost of re-evaluating every prompt in full.

#### A harness defect found while measuring, which invalidated the first numbers

Both models failed `explicit lifetimes` with "never produced a test file", which is a
suspicious way for a 14B to fail a trivial task. It was ours, not theirs.

The model's reply was a complete, well-formed tool call **except** that it wrote
Rust's escape syntax inside a JSON string: `\u{0}` where JSON demands exactly four hex
digits after `\u`. `serde_json` rejects the whole array, `parse_tool_calls` returns
empty, and the agent loop treats an empty parse as "no tool calls — this is the final
answer". So the call was dropped with **no error anywhere**: nothing was ever
recognised as a call, so nothing could fail. From the outside it looks exactly like a
model that did not try.

Fixed in the library (V260): the response is repaired before parsing —
`\u{XXXX}` transliterated to `\uXXXX`, stray control characters dropped, and
backslashes introducing no valid escape dropped. Escaped backslashes (`C:\\tmp`) and
valid `\uXXXX` survive untouched.

Two things worth recording about the diagnosis, both mistakes:

* The debug print truncated the reply at 500 characters while reporting its *full*
  length, so a complete reply looked like a backend truncating mid-generation. It now
  prints in full.
* The first repair attempt dropped the control character but kept its backslash,
  which orphaned it against the next quote (`\"` → `\\"`) and closed the string
  early — one parse error traded for another. And a debug line measuring
  `ai_assistant::parse_tool_calls` was measuring the **wrong parser**: that name is
  re-exported from `unified_tools`, not from the agent loop.

**The transliteration is deliberately not a favour to the model.** `\u{0}` means code
point zero, so the repaired call writes a NUL where the model plainly meant `""` — and
its own test then fails on correct code. That is a real model error and it is now
scored as one. Deleting the escape instead would have quietly fixed the model's bug
and flattered the result.

**Numbers measured before V260 are not comparable**: any task where a model happened to
emit `\u{…}` was recorded as "never produced a file", i.e. a harness defect counted as
model incompetence. The pre-fix 7B measurements (7.00/12 and 6.00/12) and the pre-fix
14B measurement (10.00/12) are superseded by the re-runs below.

#### A second scoring defect, found the same way

One 7B task had no result line at all. It had hit the **llama.cpp runner crash** — at
temperature 0.5, which settles a question left open in the previous entry: 0.5 avoids
the crash on the input that first exposed it, but is **not immunity**.

Worse, the harness labelled such a run "excluded from the score" while still counting
it as a failed attempt in the pass rate. A crashing runner therefore penalised
whichever model happened to be loaded — the exact confusion the label exists to
prevent. Crashed runs now leave the denominator (`passes/attempts`, not
`passes/repeats`) and are reported separately as runs lost.

#### Results

All at temperature 0.5, seed 42, `num_ctx` 4096, 3 interleaved repeats, both models
fully on GPU.

| model | score | inconsistent tasks | runs lost to crashes |
|---|---|---|---|
| qwen2.5-coder:7b-instruct | **6.67 / 12** (7.33 on the preceding run) | 2–5 | 2 |
| qwen2.5-coder:14b | **9.67 / 12** | 1 | 0 |

The ~3-task gap is far outside the ±1 noise band, so this category **discriminates**.

**Capability and consistency arrive together.** The 14B is decisive: ten tasks at 3/3,
two at 0/3, one at 2/3. The 7B is smeared across 0/3, 1/3, 2/3 and 3/3 — which is what
"at the limit of its competence" looks like when you measure it as a rate instead of
flipping a coin once. That is the strongest argument yet for not chasing a deterministic
backend: determinism would have frozen each of those coins on one face and reported the
7B as reliable at tasks it solves a third of the time.

**Failure modes, in order of frequency.** The dominant one is **inventing requirements**
— asserting behaviour the implementation was never given, producing a suite that fails
on correct code. Second is a suite too weak to kill its mutant (`assert!(true)` in
spirit). Both models fail `dedup preserving first-appearance order` this second way:
neither writes a case with *non-consecutive* duplicates, so both approve a `Vec::dedup`
implementation that is wrong.

Neither failure is visible to a "did it produce tests?" check, which is the whole point
of scoring by mutation.

### 2026-07-31 (2nd) — a "model failure" that was the backend crashing: temperature 0 aborts the llama.cpp runner

**One of the two test-generation suites qwen2.5-coder:7b "failed" was not a model
result at all.** `test-gen: count occurrences` failed 3/3 with the generation never
returning. Once the crash is avoided the same model passes it **in 10.8 s**. The
score was measuring our infrastructure.

**The chain of refutations** (each hypothesis was tested and killed, in order):

| Hypothesis | Test | Verdict |
|---|---|---|
| Backend is dead | `/api/chat`, short message | answers in **0.4 s** — alive |
| Model loops forever | direct probe of the same prompt shape | **3 s**, 20 tokens, `done_reason: stop` |
| Model is just slow (>120 s) | re-issue with a **600 s** ceiling | never returned, and the **GPU sat at 7 %** — nothing was generating |
| `num_ctx` change forces a reload | recomputed both iterations | 8192 in both — no reload |
| Aborted requests leak runner slots | `ollama stop`, fresh runner | crashes identically |

**Root cause, from the Ollama server log** (`%LOCALAPPDATA%\Ollama\server.log`):

```
Assertion failed: found, file llama-sampling.cpp, line 660
post predict ... wsarecv: forced interruption of an existing connection
```

The llama.cpp runner **aborts mid-request**. The socket dies with it, so the client
reports a *send* failure (`Failed to send request to Ollama`) rather than a timeout —
which is why this reads like a dead backend from the harness side.

**The trigger is near-greedy sampling, and it is input-dependent.** Same body, same
model, only `temperature` varied:

| temperature | result |
|---|---|
| 0.0 / 0.1 / 0.2 / 0.3 | **runner crash** |
| 0.5 / 0.7 | answers in 4–7 s |

Sampler knobs do **not** avoid it: `top_k=1`, `top_p=1.0`, `repeat_penalty=1.0`, and
all three together were each measured — every one still crashed. Only temperature
matters.

**Why this is awkward: our benchmarks run at temperature 0 *for* determinism**, which
is exactly the crashing path. Raising the temperature alone buys stability at the cost
of the property we wanted — measured, the same 8-suite category at 0.7 returned a
*different* failing set than the run before it (`trait with two impls` passed,
`count occurrences` and `explicit lifetimes` failed).

**The fix that keeps both: a fixed `seed`.** At `temperature 0.5, seed 42` the output
was **byte-identical across three runs** (1085 tokens, same hash). Blocker:
`ollama_chat_options()` in `src/providers.rs` sends only `temperature` and `num_ctx`,
never `seed` — the library cannot currently express reproducible sampling. Recorded as
an API gap, not yet implemented.

**Harness change.** This class of event is now reported as `BACKEND CRASH, not a model
failure (excluded from the score)`, pointing at the server-log assertion. The previous
wording claimed the ceiling meant "either a dead backend or a model that never
terminated" — both refuted above, so it was wrong.

**Lesson for the notebook:** when a local-model run fails with a request that never
came back, read the backend's own log **before** recording it as a model result. A
crash and an incompetent model are indistinguishable from the client side.

**The fix, and what it did and did not buy.** `AiConfig::seed` now exists and is sent
on every Ollama request path; the benchmark defaults changed from `temperature 0.0,
no seed` to **`temperature 0.5, seed 42`** (`AI_BENCH_TEMP` / `AI_BENCH_SEED`, the
latter accepts `none` to randomise). Report headers now carry both, because a logged
result without its sampling settings cannot be reproduced. Verified with a new
`AGENTIC_TRACE=1` mode that fingerprints every prompt and reply:

* **Per task, it is now exactly reproducible.** Two runs of the same task produced
  byte-identical prompts *and* replies at every turn.
* **Across a multi-task run, it is not.** Two full runs of the 8-suite category still
  diverged — starting at the very first generation, where an *identical* prompt
  returned a different reply. Normalising the runner first (`ollama stop`) did not
  close it either: **2 of 8 verdicts flipped** (`count occurrences` and `trait with two
  impls` swapped), though the total held at 4/8 both times.

The seed removes the *client-side* randomness, which is all it can do. What remains is
llama.cpp's own numerical non-determinism: KV-cache prefix reuse and batch splitting
differ with server state, the reduction order changes, and occasionally a logit
tie flips. **So a single-run category score carries roughly ±1 task of noise**, and
that applies retroactively to every single-run number in this notebook. Treat a
one-task difference between two models as *no* difference.

### 2026-07-31 — trustworthy oracles, unfinished-work detection, and repair that actually works (harness @ V256–V257 / 0.2.209)

**The oracles were audited first, and two were not competent.** Every Rust task is
scored by appending an assert suite and running `cargo test`, which makes that suite
an oracle — yet it had only ever been checked in one direction (a correct
implementation passes). The new `checker_adequacy` category adds the direction that
matters: each task also gets hand-written **mutants that must FAIL**.

Two of the twelve were caught accepting knowingly-wrong code:

| Oracle | Accepted | Why |
|---|---|---|
| `trait with two impls` | a circle computing `PI * r` | the only radius tested was **1.0**, where the bug equals the correct answer |
| `explicit lifetimes` | `longest` that always returns its first argument | **both** cases had the answer in first position |

Both fixed (radius 2; a case with the longer string second, comparing values not
lengths). Adequacy then extended to the six multi-step oracles, which proved sound
first time. **Total: 50/50.** llama3.1:8b still scores 10/12 under the stricter
oracles, so its published number was not inflated.

**Unfinished ≠ wrong.** `todo!()` type-checks, so a crate full of placeholders BUILDS
and an agent that trusts the compiler declares victory — precisely what happened under
pre-chewing. The harness now detects placeholders, re-queues the agent with a specific
instruction, and reports the two outcomes distinctly instead of blaming the compiler
for code that compiled perfectly and was merely half written. It turned an invisible
failure into a measured one: **with pre-chewing, llama3.2:3b leaves placeholders in 8
of 12 tasks.**

**`AI_BENCH_AUTOFIX=1` — compiler-guided repair.** Applies rustc's
`suggested_replacement` spans speculatively and keeps them only if the crate compiles
**and the checker passes**. Verified on the two failures dissected in V255:

- `dedup in place` — **FAIL → PASS**, applying `return` + `;`, both `MaybeIncorrect`,
  which `cargo fix` refuses to apply at all.
- `generic largest<T>` — **correctly rejected**. And this is the important one:
  rustc marks that `+ std::cmp::Ord` suggestion **MachineApplicable**, so `cargo fix`
  *would* have applied it, producing code that compiles and silently breaks the task's
  f64 case. **Verifying by compilation is not merely insufficient, it is unsafe.**
  Only running the tests caught it — which is exactly why the oracle audit had to
  come first.

**Net effect, stated plainly:** on llama3.2:3b across all 12 tasks the repair changes
nothing (1/12 either way) — its failures are not the kind rustc can suggest a usable
fix for. The 8B's full-set net effect could **not** be measured cleanly: it needs
~11 GB resident and only 9.9 GB can be freed without killing system processes, so
every full run carried a CPU-offload warning and was discarded. The mechanism claims
above hold regardless, since they do not depend on generation speed.

The knob stays **off by default**: with it on, a score measures "this model plus a
repair tool", not the model. Accepted repairs are logged so a repaired pass is never
mistaken for the model's own work.

**Commits:** V256–V257 (0.2.208 → 0.2.209).

### 2026-07-29 — what the failures ACTUALLY are, and what "pre-chewing" should mean (harness @ V255 / 0.2.207)

Instead of aggregate scores, this entry dissects the two tasks `llama3.1:8b` fails in
`agentic_rust` (it scores 10/12). Both had been filed under "capability wall". Only
one of them is.

**Case 1 — `borrow checker: dedup in place`.** The model writes:

```rust
v.retain(|x| { if !seen.contains(x) { seen.insert(*x); true } false });
```

The **algorithm is correct** — `retain` + `HashSet` is the idiomatic answer, and is
what the reference implementation uses. There is **no borrow-checker error at all**
(the task name is misleading and should change). It fails on `error[E0308]`: an `if`
without `else` used as an expression. rustc even prints the fix
(`help: … return true;`). The agent had six iterations, could run `cargo test`, and
never applied it.

**Case 2 — `generic largest<T> with trait bounds`.** The model writes
`list.iter().max().unwrap()` against a `T: PartialOrd + Copy` signature. That is a
genuine Rust knowledge gap: `.max()` requires `Ord`, and the task deliberately tests
`f64`, which is **not** `Ord` — with only `PartialOrd` you must fold by hand.

**Why this matters for automated repair.** rustc's own suggestion for case 2 is
"add `+ Ord` to the bound". Applying it was tested: the crate then fails to compile
the float case (`the trait bound f64: Ord is not satisfied`). So **compiling is not
a sufficient acceptance criterion** — a compiler suggestion can look like a fix and
break the requirement. Case 1's suggestion, applied verbatim, compiles *and passes
the checker* (tested).

Note `cargo fix` helps with neither: both suggestions carry
`applicability = MaybeIncorrect`, and `cargo fix` only applies `MachineApplicable`
ones (verified via `--message-format=json`).

**Measured dead end: skeleton pre-chewing.** `AI_BENCH_PRECHEW=1` seeds the exact
signature with `todo!()` bodies so the model supplies only the algorithm. Result:
**10/12 → 9/12** — worse. It fixed neither failure and broke a task that previously
passed (`implement the Iterator trait`). The cause was confirmed by dumping the final
artifact: **two `todo!()` left untouched**. `todo!()` compiles, so `cargo build` told
the agent everything was fine and it stopped. Handing a model a hole to fill also
takes it off the pattern it is best at — writing a complete idiomatic file.

**The mechanism worth building instead** (queued, not implemented): parse
`cargo build --message-format=json`, apply the `suggested_replacement` spans
**speculatively — including `MaybeIncorrect` ones** — and keep the result only if it
compiles **and the tests pass**. It asks the model nothing; it is deterministic
search over rustc's own suggestions with execution as the judge. Expected coverage,
stated honestly: syntactic slips yes (case 1), conceptual gaps no (case 2) — and
case 2 *should* keep failing, because that is a real capability difference the
benchmark exists to measure.

**Commits:** V255 (0.2.207).

### 2026-07-29 — fourth lever (multi-agent critic): the hypothesis is settled (harness @ V250 / 0.2.202)

**Setup:** `agentic_rust` (12 tasks), Ollama, temperature 0, **every run reporting
zero CPU-offload warnings** (the V249 guard), and the 8B baseline reproducing its
known 10/12 — two independent signs the conditions are clean.

`AI_BENCH_CRITIC=1` adds the BACKLOG's multi-agent lever: on failure a SECOND agent
in a reviewer role reads the specification and the produced code and reports
concrete defects, and that critique — not the compiler output — drives the revision
round. Same trigger and round count as `AI_BENCH_SCAFFOLD`, so only the
*information* differs.

| Lever | kind | llama3.2:3b | llama3.1:8b |
|---|---|---|---|
| baseline (single shot) | — | 1/12 | 10/12 |
| `AI_BENCH_SCAFFOLD` (verify→retry) | more attempts | 2/12 | 10/12 |
| `AI_BENCH_SAMPLES` (best-of-N @temp 0.7) | more attempts | 2/12 | — |
| `AI_BENCH_KNOWLEDGE` (idiom injection) | more information | 2/12 | — |
| `AI_BENCH_CRITIC` (multi-agent reviewer) | different reasoner | 2/12 | 10/12 |

**Finding — the BACKLOG's central bet is refuted on this task set.** Four
qualitatively different scaffolding strategies — retry with compiler feedback,
independent resampling, injected reference knowledge, and a separate reviewing
agent — produce the *identical* +1 of 12 on the weak model and *no change at all*
on the stronger one. Scaffolding raises the odds of a model finding an answer it
was already capable of producing; it does not manufacture capability that is
absent, and it does not repair a model that is simply past its depth.

That the four agree so exactly is what makes this conclusive rather than
suggestive: "we picked the wrong scaffolding" is no longer available as an
explanation.

**What this means for the project's differentiator.** The viable version is not
*"scaffolding lets a small local model match a big one"* — that is now measured and
false. It is *"pick a model that is already in the capable band for the task (here
the 30B MoE, or 7–8B for simpler work) and use the scaffolding to make it
reliable and autonomous."* The value of the agentic machinery is **autonomy and
verification**, not capability amplification.

**Caveat, stated so it is not overclaimed:** one benchmark, one language, twelve
small tasks, two models. It says nothing about scaffolding on long-horizon,
multi-file work where planning and memory dominate — which is precisely where the
`agentic_multi` numbers suggest the real differences live.

**Commits:** V249–V250 (0.2.201 → 0.2.202).

### 2026-07-27 (5th) — third lever (information), same +1: the plateau is capability (harness @ V246 / 0.2.198)

**Setup:** `agentic_rust` (12 tasks), llama3.2:3b, 100% GPU. `AI_BENCH_KNOWLEDGE=1`
attaches a `RustIdiomProvider` through the library's own
`AutonomousAgent::with_knowledge_provider` hook: before every iteration it injects
worked examples of the Rust patterns the task needs (generics with bounds, trait +
`dyn` dispatch, explicit lifetimes, implementing `Iterator`, `Option`/`Result`,
`HashMap` entry API, in-place `retain`, enum + match, builder, struct with methods).
The snippets are **generic patterns, never solutions** — handing over the answer
would measure nothing.

| Lever (llama3.2:3b, 12 Rust tasks) | Kind | Result |
|---|---|---|
| baseline, single shot @temp0 | — | 1/12 |
| `AI_BENCH_SCAFFOLD=3` (verify→retry) | more attempts | 2/12 |
| `AI_BENCH_SAMPLES=3` @temp0.7 (best-of-N) | more attempts | 2/12 |
| **`AI_BENCH_KNOWLEDGE=1` (idiom injection)** | **more information** | **2/12** |

**Finding: three qualitatively different levers, the identical +1.** Adding
information does no better than adding attempts. Every strategy plateaus at 2/12,
which is the strongest evidence yet that the limit is the model's **capability** on
these tasks, not what it is told or how many tries it gets.

**A trap worth recording (it invalidated the first run of this experiment).** The
`KnowledgeProvider` contract is `enrich(&self, query: &str)`, and the agent passes
**the last user/tool message** as `query`. Inside an agentic loop that message is
almost always *tool output* — `"[Tool: write_file] wrote 143 bytes to src/lib.rs"` —
not the task. Keyed on that, the provider matched **0 snippets** on the real task
prompt and the experiment silently measured nothing. It fails *quietly*: no error,
just no useful context, and you walk away believing "RAG doesn't help" when it never
ran. Fixed by having the provider carry the task text and match on task + query;
verified by instrumenting the match count before trusting the numbers.
**Consequence for wiring real RAG over the codebase: index by the task, not by the
last message.**

**Commits:** V246 (0.2.198).

**Next:** the levers left are ones that change *who* is reasoning, not what they are
told — multi-agent review with a critic, Chain-of-Verification. Also worth testing
whether knowledge injection helps a model that is already near the boundary (the 8B
at 10/12) rather than one that is far from it.

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
