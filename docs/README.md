# `docs/` — what is here, and what to read first

198 Markdown files live in this directory. 154 of them are `IMPROVEMENTS_V*.md`,
a **closed historical series**. This index exists so that nobody has to guess
which of the rest is current.

## Start here

| You want to know | Read |
|---|---|
| **Where the project is right now** | [`../CHANGELOG.md`](../CHANGELOG.md) — newest entry first |
| **How we work** — workflow, quality gates, the checklist before calling something done | [`modus-operandi.md`](modus-operandi.md) |
| How to use the library at all | [`GETTING_STARTED.md`](GETTING_STARTED.md), then [`GUIDE.md`](GUIDE.md) |
| What the public API looks like | [`API_REFERENCE.md`](API_REFERENCE.md) |
| The ideas behind the design | [`CONCEPTS.md`](CONCEPTS.md), [`INNOVATIONS.md`](INNOVATIONS.md) |

## The historical series — read as history, not as state

`IMPROVEMENTS_V1.md` … `IMPROVEMENTS_V167.md` record how the codebase got here,
one change at a time. **The series ended in March 2026 at version 0.2.119.**
Everything since — more than 120 versions — is in `CHANGELOG.md`.

This matters because of a specific failure that has happened twice: someone
looking for the current state opens the highest-numbered file, finds a coherent
and well-written document, and takes it for today. A document does not have to be
wrong to mislead; it only has to be findable and undated. `IMPROVEMENTS_V167.md`
now carries a banner saying so.

## Subsystem guides

| Subsystem | Guide |
|---|---|
| Academic research: search, review, BibTeX, RAG bridge | [`GUIDE_RESEARCH.md`](GUIDE_RESEARCH.md) · [`RESEARCH_SUBSYSTEM.md`](RESEARCH_SUBSYSTEM.md) (inventory, incl. what is *not* done) |
| Anti-hallucination: faithfulness, CoVe, quality gates | [`GUIDE_ANTI_HALLUCINATION.md`](GUIDE_ANTI_HALLUCINATION.md) |
| Graph RAG | [`GRAPH_RAG_GUIDE.md`](GRAPH_RAG_GUIDE.md) |
| Prompt breeding / fragments | [`PROMPT_BREEDER_GUIDE.md`](PROMPT_BREEDER_GUIDE.md) · [`PROMPT_FRAGMENTS.md`](PROMPT_FRAGMENTS.md) |
| Multi-agent design | [`AGENT_SYSTEM_DESIGN.md`](AGENT_SYSTEM_DESIGN.md) |
| FFI / embedding in other languages | [`FFI.md`](FFI.md) |
| The shipped binaries | [`BINARIES.md`](BINARIES.md) |
| What people build with it | [`USE_CASES.md`](USE_CASES.md) |

## Models and measurement

| | |
|---|---|
| [`MODEL_BENCHMARKS.md`](MODEL_BENCHMARKS.md) | Results per model, **with date and backend**. Read the dates: a result from a different backend or quantisation is a different measurement. |
| [`LOCAL_MODELS.md`](LOCAL_MODELS.md) | Quantisation, KV cache, what fits on the card |
| [`LOCAL_MODELS_CONTEXT_AND_QA.md`](LOCAL_MODELS_CONTEXT_AND_QA.md) | Context sizing and the conversation-quality harness |
| [`RUNTIMES_COMPARISON.md`](RUNTIMES_COMPARISON.md) · [`RUNTIMES_INSTALL.md`](RUNTIMES_INSTALL.md) | Ollama, llama.cpp, LM Studio and friends |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Criterion micro-benchmarks of the library itself |
| [`TESTING.md`](TESTING.md) | Test layout, the harness, what `--all` covers |

## Process, security and legal

| | |
|---|---|
| [`RELEASE_PROCESS.md`](RELEASE_PROCESS.md) · [`PROTECTED_BUILD.md`](PROTECTED_BUILD.md) | Cutting a release |
| [`FEATURE_LIFECYCLE.md`](FEATURE_LIFECYCLE.md) | Adding, deprecating and removing feature flags |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) · [`PGVECTOR_SETUP.md`](PGVECTOR_SETUP.md) | Running it somewhere |
| `SECURITY_AUDIT*.md`, `SECURITY_HARDENING_V15*.md` | Point-in-time audits — each is dated in its own title |
| [`DPIA_TEMPLATE.md`](DPIA_TEMPLATE.md) | GDPR data-protection impact assessment template |
| [`REFERENCES.md`](REFERENCES.md) · [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) | Prior art and credit |

## Automated checks that keep these files honest

Documentation drifts silently, so some of it is enforced:

- `scripts/check_documented_cli.py` — fails if any file here shows an `ai_cli`
  command line the binary would not accept. Written after the research guide was
  found to document five flags that never existed: a wrong flag does not error, it
  is folded into the positional argument, so the command "works" and answers a
  different question.
- `scripts/check_openapi_routes.py` — fails when the embedded server serves a route
  that `/openapi.json` does not declare, or declares one it does not serve. The spec
  is what a third party generates their client from, and the failure is silent in one
  direction: a hidden endpoint never errors, the generated client simply lacks it.
- `scripts/check_binaries_documented.py` — [`BINARIES.md`](BINARIES.md) and
  `Cargo.toml` must agree on which binaries exist, the stated total must be right, and
  no row may link to a page that is not there. The page calls itself the authoritative
  inventory and had been listing 26 of 41.
- `scripts/check_doc_links.py` — counts broken rustdoc links, both classes: an
  unresolved `[link]` and a public page linking to a private item. A ratchet, now at
  zero. Its own first version counted only the first class, so five warnings went
  unseen for the three days the work took.
- `scripts/check_doc_imports.py` — three checks over these files. Every
  `use ai_assistant::…;` inside a ```rust fence must name a module that exists and an
  item *reachable at that path*; and every CamelCase name on a `**Key types**:` line
  must exist. Nothing compiled those fences: 48 of 1037 imported names existed at no
  path at all, 38 more lived somewhere other than where the document said (two of them
  in the README's front-page example), and 22 `**Key types**` entries named types that
  were never written. **Call signatures are not checked** — the script says so in its
  own docstring, because a gate called "doc imports" sitting at zero invites the reader
  to assume more than it verifies. See N91.
- `scripts/check_feature_dep_drift.py` — `dep:X` vs feature `X` in `Cargo.toml`.
- `scripts/check_deprecation_policy.py` — every `#[deprecated]` carries `since` and `note`.
- `scripts/check_bench_budget.py` — benchmark budgets.
- `scripts/check_rustsec_ignores.py` — the suppressed-advisory lists in `deny.toml` and
  `ci.yml` must match, and each entry must have a reason written above it.
- `scripts/check_duplicate_types.py` — refuses a **new** enum name declared twice with
  different variants. 215 duplicated type names measured on 2026-10-01, of which 35 are
  divergent enums; those are frozen in `scripts/duplicate_types_baseline.toml` under
  `[untriaged]`, which says in the file that nobody judged them. It compares **names**,
  not meaning: `StepStatus` and `PlanStepStatus` were the same concept and it could
  never have paired them.
- `scripts/check_readme_numbers.py` — the front page's own figures: source files, lines,
  feature flags and the number of gates in this list. It was written after four of those
  five numbers were found to **understate** the work (6,095 tests against 8,990; 423K
  lines against 558K) while the fifth claimed the whole tree was complete with nothing
  half-finished in it — which was not stale but false. It also refuses that claim, and
  "production-ready", by name. **The test count is not checked**, and the script's own
  docstring says why: getting it right needs the suite to run, and counting `#[test]`
  attributes instead would verify an approximation while appearing to verify the number.

- `scripts/check_deny_graph.py` — `deny.toml` must walk the **whole** feature graph. It
  did not: `all-features = false` meant `cargo deny` checked 299 of 955 crates, so the
  licence, ban and source rules never saw the 656 behind optional features — 69 % of the
  tree, in a crate with 95 feature flags. What makes it gate-worthy is that both
  configurations print the same line, `advisories ok, bans ok, licenses ok, sources ok`:
  the narrow one is not a weaker green, it is an identical one. The symptom that gave it
  away was six suppressed advisories reporting `advisory-not-detected` — which looked
  like six stale entries and was really five live advisories on crates nobody was
  looking at.

- `scripts/check_recheck_dates.py` — every suppressed advisory carries a `Re-check by:`
  date, and that date must be in the future. The policy demanding one was already written
  and already followed; what was missing was anything that enforced it. Measured
  2026-10-13: **six of the eight dates had passed**, with one re-check recorded between
  them. The monthly review workflow was not broken — it ran every month from June to
  October 2026 and opened a tracking issue each time, and **all five issues are still
  open**. A forcing function that writes to the Issues tab of a one-developer repository
  produces no force. Re-checking the six by hand then found that **five of the eight
  written reasons were false or named the wrong cause**, two of them with a condition that
  could never come true — so this gate is also the reason those got corrected. Note it can
  go red with no code change, the day a deadline passes; that is the mechanism, and an
  entry within three weeks of its deadline warns instead of failing.

- `scripts/check_config_constructible.py` — a `#[non_exhaustive]` struct that a public
  function takes as an argument must be configurable from another crate **in one
  expression**. Measured 2026-10-26: **174 of 316** such structs were not. The attribute
  rejects the struct expression, `..Default::default()` included, and that part is
  frequently misremembered as stronger than it is — it does *not* stop you assigning to a
  `pub` field of a value you own, so the usual case was awkward (a `mut` binding, one
  statement per field, no construction in argument position) rather than impossible. The
  class that *is* impossible has its own name here, `NO_SEED`: nothing returns a value and
  there is no `Default`, so no setter can help. Unit tests cannot see any of it, because
  `#[non_exhaustive]` does not apply inside the declaring crate — they build these with a
  literal and pass. The gate also checks **coverage**, every public field needs a setter,
  which is the half that stops the next one: V247 already did this sweep once and closed
  over 58 of ~290 types.

All fifteen run in CI **here**. Three more live in `scripts/` and no job in this repository
runs them:

- `scripts/check_release_ready.py` — a pre-release step that needs a human to read it.

And one tool that is deliberately **not** a `check_*` script, so it is not counted among the
gates and never runs in CI:

- `scripts/sweep_deadlines.py` — *where else does this tree promise to revisit something by a
  date?* It found the defect V402 fixed: `docs/runbooks/rustsec-handling.md`, the written policy
  for the suppression rules V401 had just rewritten, still taught the superseded form with a
  worked example two months expired. It stays a sweep because of its measured noise: the first
  run gave 69 hits, 16 "overdue", and **none of the 16 were real** — the rest were *records*
  (CHANGELOG entries, status-table rows, dated audits), and a gate that fails on 16 things that
  do not matter teaches people to pass it with `--ignore`. Its second output is the `Last
  reviewed:` stamps, where nine of ten are over 90 days old; note that a stale stamp is **not**
  fixed by writing today's date, because the stamp means *somebody read this document that day*.

The other two guard [the website](https://ai-assistant.runawaybrains.com), and are run by
**that** repository's own CI (`.github/workflows/consistency.yml` there), which checks out both
repositories — the facts are measured in this tree and asserted over there, so nothing can
check one against the other from a single checkout. Both import from the checkers above rather
than counting or resolving anything themselves: one measurement, one resolver, several
documents.

- `scripts/check_website_numbers.py` — the hero badges and the `<meta>` descriptions. The
  website carried the identical stale set the README did (369 source files, 61 feature flags,
  `~520K LoC`), because the digits sit bare inside `<span class="num">` and in the text search
  engines show, which a plain word sweep walks past. **Prose is deliberately out of scope**:
  the site states current and historical figures in the same sentences with nothing marking
  which is which, two heuristics were tried and measured, and neither separates them — so the
  script says so where it would otherwise check, instead of approximating.
- `scripts/check_website_examples.py` — the same question as `check_doc_imports.py`, asked of
  the `<pre><code>` blocks. They were a **fourth** uncompiled population of example code, in
  the state the Markdown fences were in before V340: six imports named types the crate does not
  have, on `quickstart.html` and `developer_guide.html` among them — the first pages a stranger
  reads.

Run either locally with, e.g.,
`python scripts/check_website_examples.py ../ai_assistant-website`.

And that paragraph is itself checked, by `scripts/check_checkers_documented.py`: the
list above, the scripts on disk and the steps in the workflows must agree, including
the count. It said "all three" until V307 while five were running, and "all six" until
V340 while eight were — the same drift the rest of this section exists to prevent, one
level up. If you add a claim here that can be checked mechanically, prefer adding the
check to trusting the
prose — and then remember that this paragraph is itself prose.
