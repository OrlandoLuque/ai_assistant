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
- `scripts/check_feature_dep_drift.py` — `dep:X` vs feature `X` in `Cargo.toml`.
- `scripts/check_deprecation_policy.py` — every `#[deprecated]` carries `since` and `note`.
- `scripts/check_bench_budget.py` — benchmark budgets.

All six run in CI. `scripts/check_release_ready.py` is the seventh checker in
`scripts/` and is **not** wired to CI — it is a pre-release manual step.

This list said "all three" until V307, while five were already running. If you add a
claim here that can be checked mechanically, prefer adding the check to trusting the
prose — and then remember that this paragraph is itself prose.
