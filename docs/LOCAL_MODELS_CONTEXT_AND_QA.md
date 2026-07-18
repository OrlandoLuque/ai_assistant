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

Run: `ai_cli qa --model <m>` (Ollama) or
`ai_cli qa --provider llamacpp --url <server> --model <m>` (PrismML).

| Model | Quant | context_recall | grounded | multi_fact | fact_update | multi_grounded | Total |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **llama3.1:8b** | Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **llama3.2:3b** _(mobile)_ | Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **llama3.2:1b** _(mobile)_ | Q4 | ✅ | ✅ | — | — | — | passes basics |
| **qwen2.5:1.5b-instruct** _(mobile)_ | Q4 | ✅ | ✅ | — | — | — | passes basics |
| **gemma2:2b** _(mobile)_ | Q4 | ✅ | ✅ | — | — | — | passes basics |
| **Bonsai-4B** (PrismML) | 1-bit `Q1_0` | ✅ | ✅ | ❌ | ✅ | ✅ | **4/5** |
| **Bonsai-8B** (PrismML) | 1-bit `Q1_0` | ✅ | ✅ | ❌ | ❌ | ✅ | **3/5** |
| **Ternary-Bonsai-4B** (PrismML) | ternary `Q2_0` (~1.58-bit) | ✅ | ✅ | ❌ | ✅ | ✅ | **4/5** |

_(The mobile 1–3B Q4 rows were exercised on the two basic scenarios; the
complex three are shown for the models we ran the full suite on.)_

### Findings

- **Q4 small models are excellent for complex conversations.** Even the mobile
  **3B** (`llama3.2:3b`) scores **5/5** — it tracks four facts through
  distractions, honours a mid-conversation update, and grounds several prices
  from a large document.
- **Pure 1-bit (`Q1_0`) models degrade on complexity, and not monotonically
  with size.** `Bonsai-4B` loses one fact of a four-fact set
  (`multi_fact_tracking`); `Bonsai-8B` additionally fails `fact_update` — i.e.
  the **larger** 1-bit model is **worse** on the update test. Extreme 1-bit
  quantization damages exactly the multi-fact / state-update reasoning the
  complex tests target, while leaving single-fact recall and grounding intact.

### Are there "1-bit-class" models that pass the complex tests?

**Partly.** Ternary Bonsai (three states, `Q2_0` container, ~1.58-bit) is a
distinct, higher-quality extreme quant than pure 1-bit `Q1_0`, and it does
measurably better: **`Ternary-Bonsai-4B` scores 4/5** — it recovers the
`fact_update` test that the 1-bit `Bonsai-8B` failed, and passes grounding,
recall and multi-price retrieval. But it still fails the single hardest one,
`multi_fact_tracking` (holding four facts at once and pulling different ones
after distractions).

So: **no extreme-quant (1-bit or 1.58-bit) model passed all five** in these
tests; the best of the class (`Ternary-Bonsai-4B`) reaches **4/5**, versus a
straightforward **5/5** for any Q4 model of 3B+. The common wall for the whole
extreme-quant class is **simultaneous multi-fact tracking** — the ability that
sub-2-bit quantization damages most.

**Practical recommendation:** use **1-bit `Q1_0`** only for the simplest
recall/grounding on the most memory-constrained devices; for a real
conversational assistant that must track and update state, prefer a **Q4 model
of 3B+** (the `mobile` profile on `llama3.2:3b` is the sweet spot), stepping up
to 7–9B (`local-balanced`) when the hardware allows.

---

## 7. Running the extreme-quant models (PrismML)

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
ai_cli query --num-ctx 32768 …    # raise the Ollama window (you have the VRAM)
ai_cli query --full-knowledge …   # inject the whole knowledge doc (skip retrieval)
```
