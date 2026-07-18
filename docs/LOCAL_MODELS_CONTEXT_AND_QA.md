# Local models: context handling, onboarding & conversation-quality testing

_Investigation and engineering record for V192–V199 (July 2026). Everything
here was reproduced and validated live against local models via `ai_cli`._

This document captures why the assistant used to "lose the context and make up
facts it supposedly had at hand", what was fixed so the shipped binaries work
out of the box, the conversation-quality test harness that now guards it, and
the measured results across local models (including 1-bit / ternary models run
through the PrismML `llama.cpp` fork).

---

## 1. Root cause of "lost context / invented prices" — the `num_ctx` bug

**Symptom (reproduced from a real project):** with a knowledge document
injected via `--knowledge`, the assistant answered *"I have no information…"*
or invented a wrong price — even though the correct figure was in the injected
text.

**Root cause:** every Ollama chat request sent only
`options: { temperature }` and **never set `num_ctx`**. Ollama therefore fell
back to its tiny default context window (~2048 tokens) and **silently
truncated** any longer prompt — dropping the injected knowledge and earlier
conversation turns. The only trace was a `debug` log line, so the loss was
invisible to the user.

**Smoking-gun A/B** (same 145 KB knowledge, same `llama3.1:8b`):

| Request | Answer |
|---|---|
| **without `num_ctx`** (old behaviour) | "no tengo información sobre un pliego…" ❌ |
| **with `num_ctx = 32768`** | **490** ✅ |

**Fix (V192):** `providers::ollama_num_ctx` sizes `num_ctx` per request —
auto-fits the prompt (+ a response reserve), quantized to power-of-two steps
(stable across requests, so Ollama does not reload the model), capped at both
a **VRAM-safe ceiling** and the model's real window. Wired into every Ollama
request path (streaming, non-streaming, cancellable, vision, async).
`AiConfig.ollama_num_ctx` / `ai_cli query --num-ctx` let you raise it; a
`log::warn` now fires when the prompt still exceeds the window.

> ⚠️ **Operational caveat:** an over-large `num_ctx` does **not** fail cleanly
> — it **OOM-crashes the Ollama server** (observed: 65536 on an 8B took the
> server down and it had to be restarted). That is why the automatic sizing is
> deliberately conservative (ceiling 16384) and larger windows are opt-in.
> Changing `num_ctx` between requests also makes Ollama reload the model
> (KV-cache reallocation, tens of seconds) — hence the power-of-two
> quantization.

---

## 2. "Just works" onboarding

Driving the CLI as a freshly-downloaded user would surfaced three more bugs,
all fixed so the shipped binaries auto-detect a local model in ~2 s:

- **Model discovery timed out (V193).** Discovery probed the five local
  providers **sequentially**, each with retries; a closed `localhost` port can
  take seconds to refuse a connection, so the dead ports serialized past the
  poll cap and even Ollama's models were lost. Now: single-attempt short-
  timeout probes, run **concurrently**.
- **`scan` didn't detect a running Ollama and took 27 s (V194).** The default
  URLs used `localhost`; `ureq` tries IPv6 `::1` first and stalls on services
  bound only to `127.0.0.1`, past the detector timeout. Switched all
  local-provider defaults to **`127.0.0.1`** (dead ports now RST instantly),
  and `Butler::scan` runs its 14 detectors **concurrently**. Result:
  `scan` detects Ollama with all models in ~2.3 s (was 27 s + "no providers").
- **The shipped binaries lacked `butler` (V194).** `scan` / `providers` — the
  auto-detection — require the `butler` feature, which the `full`/release
  build omitted. `butler` is dependency-free (`autonomous = []`) and is now
  part of `full`, so the distributed binaries auto-detect out of the box. The
  same fixes apply to the GUI binaries (shared config / providers / butler).

---

## 3. Large knowledge without a huge window — lite retrieval (V196)

Injecting a whole large `--knowledge` document forces a big `num_ctx` (costly
VRAM; impossible on small-window models). `knowledge_retrieval::select_relevant`
chunks the document and injects only the passages relevant to the query
(deterministic term-overlap ranker, bilingual stop-words, no embeddings / no
store). `ai_cli query` applies it automatically above ~12 KB (opt out with
`--full-knowledge`).

**Validated:** a 109 KB knowledge doc with the price buried, queried on the
small `llama3.2:3b` with **no** `num_ctx` override → retrieval pulled the
68-char price passage and the model answered **490 EUR** in 2.5 s. The QA
harness (below) also retrieves per turn, so grounding scenarios exercise the
full retrieval + context stack.

---

## 4. Functional runtime profiles (V197, V199)

A **runtime profile** bundles the tunables that make a use case work out of the
box (temperature, `num_ctx`, history depth, whether to retrieve large
knowledge) plus the models it is tuned for. `ai_cli profiles` lists them;
`ai_cli query --profile <name>` (and `ai_gui[-pro] --profile <name>`) apply
one; explicit flags still override.

| Profile | temp | num_ctx | history | retrieval | tuned for |
|---|---|---|---|---|---|
| **mobile** | 0.3 | 4096 | 8 | on | 1–3B: llama3.2:1b/3b, qwen2.5:1.5b/3b, gemma2:2b, phi3.5 |
| **local-balanced** _(default)_ | 0.7 | auto | 20 | on | 7–9B: llama3.1:8b, qwen2.5:7b, mistral:7b, gemma2:9b |
| **local-quality** | 0.7 | auto | 24 | on | 14B+: qwen2.5:14b/32b, gemma2:27b, llama3.1:70b |
| **coding** | 0.2 | auto | 20 | on | qwen2.5-coder, deepseek-coder, codellama |
| **precise** | 0.2 | auto | 20 | on | llama3.1:8b, qwen2.5:7b |
| **creative** | 1.0 | auto | 20 | off | llama3.1:8b, mistral:7b |

The **`mobile`** profile is the on-device tune: low temperature for factual
reliability, a conservative 4096 window, and retrieval **on** so a large
knowledge document still fits the small window.

---

## 5. The conversation-quality harness — and what each test means

`ai_cli qa [--provider/--model/--url/--num-ctx/--profile]` runs a set of
multi-turn *scenarios* against a real model. Each scenario is a sequence of
turns driven through **one** `AiAssistant`, so multi-turn context is genuinely
exercised — turn N can be asked about a fact stated in turn 1. Scoring is
**deterministic**: case-insensitive "must contain" / "must not contain"
substring checks (no LLM judge). A turn passes if all its checks hold; a
scenario passes if all its turns pass.

> **Test conditions (important).** Unless a flag says otherwise, the harness is
> a **bare baseline**: default **Conversation** mode (full history), with **no
> extra subsystems** enabled — no memory manager, no persistent RAG store, no
> anti-hallucination, no quality gates. So the results reflect the model plus
> the core context handling, not machinery layered on top (they are honest /
> pessimistic). Two flags change the conditions: `--fresh-context` runs in
> FreshContext mode (§8), and retrieval defaults to **semantic** embeddings
> with a lexical fallback (`--lexical` forces lexical) (§8).

The built-in scenarios (and what each one measures):

- **`context_recall`** — states a name and a favourite colour, then a
  distraction turn (a joke), then asks for both. Tests basic **multi-turn
  memory** across an unrelated turn.
- **`grounded_price`** — injects a price sheet and asks for one specific
  price. Tests **grounding**: reading a fact out of the provided knowledge
  rather than guessing.
- **`multi_fact_tracking`** — states **four** facts at once (name, city, pet,
  job), then **two** distraction turns, then asks about several of them
  (city + pet, then job). Tests holding **multiple facts simultaneously** and
  retrieving different ones after distractions — much harder than a single
  fact.
- **`fact_update`** — states a favourite colour, then **corrects it**
  mid-conversation, then (after a distraction) asks for the current value.
  Tests using the **most recent** value — i.e. that later turns override
  earlier ones, not just "find the first mention".
- **`multi_grounded`** — a large catalogue (several plan prices buried in
  filler) and **three** price questions across turns. Tests **grounding +
  retrieval + multi-turn** together: each turn must retrieve the right passage
  from a big document and answer a different figure.

The last three are the "larger / more complex" conversations: they separate
models that merely echo the last thing said from models that actually track
and update state.

---

## 6. Measured results across local models

Scores use **greedy decoding (temperature 0)** so they are reproducible —
stochastic sampling made the extreme-quant models vary ±1–2 scenarios per run
(a single sampled run once showed a spurious rescue that did not reproduce).
Run with `ai_cli qa --model <m>` (Ollama) or
`ai_cli qa --provider llamacpp --url <server> --model <m>` (PrismML); add
`--fresh-context` for the FreshContext column.

| Model | Quant | context | grounded | multi_fact | fact_update | multi_grnd | Conv. | Fresh |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **llama3.1:8b** | Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** | **5/5** |
| **llama3.2:3b** _(mobile)_ | Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** | **5/5** |
| **Bonsai-4B** (PrismML) | 1-bit `Q1_0` | ✅ | ✅ | ❌ | ❌→✅ | ✅ | 3/5 | **4/5** |
| **Bonsai-8B** (PrismML) | 1-bit `Q1_0` | ✅ | ✅ | ❌ | ❌→✅ | ✅ | 3/5 | **4/5** |
| **Ternary-Bonsai-4B** (PrismML) | ternary `Q2_0` (~1.58-bit) | ✅ | ✅ | ❌ | ❌→✅ | ✅ | 3/5 | **4/5** |

_(The `fact_update` cell shows Conversation → FreshContext. The mobile 1–3B Q4
models — llama3.2:1b, qwen2.5:1.5b, gemma2:2b — pass the two basic scenarios;
the full 5-scenario suite was run on the models above.)_

### Findings

- **Q4 small models are excellent for complex conversations** — even the mobile
  **3B** (`llama3.2:3b`) scores 5/5 in **both** modes.
- **Extreme-quant (1-bit / 1.58-bit) models hit two walls.** In Conversation
  mode all three (Bonsai-4B/8B, Ternary-4B) score **3/5**, failing
  `multi_fact_tracking` **and** `fact_update`. Quality is **not monotonic with
  size** — the 8B 1-bit is no better than the 4B.
- **FreshContext reliably rescues `fact_update`** (3/5 → 4/5 for all three).
  Its recency-kept recent turns make a mid-conversation correction stick, where
  the raw full history confuses the degraded model into the stale value.
- The **`multi_fact_tracking` wall** — holding four facts at once and pulling
  different ones after distractions — **persists in both modes**. That is the
  ability sub-2-bit quantization damages most.

### Are there "1-bit-class" models that pass all five?

**No.** The best of the extreme-quant class reaches **4/5** (all three, in
FreshContext); none passes `multi_fact_tracking`. Ternary (`Q2_0`, ~1.58-bit)
is a higher-quality extreme quant than pure 1-bit `Q1_0` but hits the same wall.

**Practical recommendation:** for a real conversational assistant that must
track and update state, prefer a **Q4 model of 3B+** (the `mobile` profile on
`llama3.2:3b` is the sweet spot), stepping up to 7–9B (`local-balanced`) when
the hardware allows. Use extreme-quant models only for the simplest
recall/grounding on the most memory-constrained devices. (Structured **memory
extraction / re-injection** — the memory manager — is the likely path to lift
the `multi_fact_tracking` wall even for weak models; see §7.)

---

## 7. FreshContext & knowledge retrieval (lexical vs semantic)

### FreshContext (V200)

FreshContext mode sends only the **latest** turn to the model (to maximize
tokens for knowledge). The catch: a bare `send_message` in FreshContext used to
throw the earlier conversation away entirely (plain `send_message` never called
the RAG retrieval path, and conversation auto-store is off by default), so
multi-turn recall failed — defeating the whole point.

Fixed so FreshContext **retrieves what the current turn needs** from the full
in-memory conversation:

- the **most recent** turns are kept verbatim (recency — so a mid-conversation
  correction/update is **never** dropped by relevance ranking), PLUS
- **older** turns relevant to the question are retrieved and prepended, in
  chronological order.

With this, FreshContext passes all 5 scenarios on llama3.1:8b **and** the mobile
llama3.2:3b — including `fact_update` (a naive relevance-only retrieval could
have surfaced the stale value). Run it with `ai_cli qa --fresh-context`.

### Retrieval rankers and the store types

- **Default persistent store** (`rag.db`): SQLite **FTS5 — lexical** (keyword),
  not semantic. `knowledge_graph.db` is a graph store.
- **Semantic / vector** (embeddings): **opt-in** tier — LanceDB
  (`--features vector-lancedb`), Qdrant, etc. Not in `full` (heavy deps).
- **Ad-hoc knowledge retrieval** (`knowledge_retrieval`, used for `--knowledge`
  and FreshContext history) has two rankers:
  - **lexical** — term-overlap, zero-dependency, instant, offline, deterministic;
  - **semantic** — cosine similarity of **Ollama embeddings** (`/api/embed`,
    e.g. `nomic-embed-text`), which handles paraphrase / synonyms.

**Semantic is the default** in the CLI (`nomic-embed-text`), with an automatic
**lexical fallback** when the embedding model is not reachable. Why default:
semantic is strictly better *when embeddings are available*; lexical is kept as
the always-available, offline, zero-dependency fallback — not merely a toggle.
Force lexical with `--lexical`; pick another embedder with
`--embedding-model <name>`; set it in code via `AiConfig.embedding_model`.

**Why it matters (measured):** a 30 KB document with `"la persona encargada de
las finanzas es Carlos Vega"` buried in filler, queried with the paraphrase
`"¿quién gestiona el dinero de la empresa?"` (no shared content words):

| Ranker | Result |
|---|---|
| lexical (`--lexical`) | "no information found" ❌ |
| semantic (default) | **"Carlos Vega."** ✅ (and faster — fewer, better-targeted passages) |

---

## 8. Running the extreme-quant models (PrismML)

The Bonsai / Ternary-Bonsai weights use a custom `Q1_0` / ternary kernel that
mainline `llama.cpp` does not implement, so they need the PrismML fork:

```sh
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF   # CPU build; add -DGGML_CUDA=ON with CUDA
cmake --build build --config Release -j
# serve a downloaded GGUF (OpenAI-compatible on :8080)
./build/bin/Release/llama-server -m Bonsai-4B-Q1_0.gguf --host 127.0.0.1 --port 8080 -c 4096
```

Then point the assistant at the `llamacpp` provider (its default URL is already
`http://127.0.0.1:8080`):

```sh
ai_cli query --provider llamacpp --model Bonsai-4B-Q1_0 "…"
ai_cli qa    --provider llamacpp --model Bonsai-4B-Q1_0
```

No app changes were needed — the existing `llamacpp` provider plus the
`127.0.0.1` default (V194) cover it.

---

## Quick reference

```sh
ai_cli scan                       # auto-detect providers (butler)
ai_cli models                     # list available local models
ai_cli profiles                   # list runtime profiles
ai_cli query --profile mobile --model llama3.2:3b --knowledge big.md "…"
ai_cli qa --model llama3.1:8b     # run the conversation-quality scenarios
ai_cli qa --fresh-context …       # run them in FreshContext mode
ai_cli query --num-ctx 32768 …    # raise the Ollama window (you have the VRAM)
ai_cli query --full-knowledge …   # inject the whole knowledge doc (skip retrieval)
ai_cli query --lexical …          # lexical retrieval (default is semantic embeddings)
ai_cli query --embedding-model nomic-embed-text …   # pick the semantic embedder
```
