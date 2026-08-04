# Running Local Models Well — Quantization, KV Cache, Hardware

A **reference** on the things that decide whether a local model fits and performs on a
given card. Its companion is [MODEL_BENCHMARKS.md](MODEL_BENCHMARKS.md), which is a
*lab notebook* of measured results; this file explains the concepts those measurements
sit on.

> **Status of the claims here.** Anything marked **measured** was measured by this
> project's own harness and is reproducible. Anything else is from vendor docs,
> papers or community sources, cited inline — treat it as a hypothesis until the
> harness confirms it. That distinction is the point: the whole reason this project has
> an execution-verified benchmark is that model-card claims are not evidence.

The reference hardware throughout is the development machine: **RTX 4080 SUPER, 16 GB,
compute capability 8.9 (Ada Lovelace)**.

---

## 1. Quantization

Storing weights at lower precision than the 16 bits they were trained in. A 14B model
is ~28 GB at FP16 and ~8 GB at 4 bits — this is what makes local inference possible at
all on consumer cards.

### Reading the names

`Q4_K_M` decomposes as:

| part | meaning |
|---|---|
| `Q4` | 4 bits per weight |
| `_K` | *k-quant*: mixed precision **within** the tensor, not a uniform cast |
| `_M` | medium variant (`_S` small, `_L`/`_XL` large) — how many tensors get promoted to higher precision |

`IQ*` names (e.g. `IQ4_XS`) are **importance-matrix** quants, which perform noticeably
better at very low bit counts.

### The practical rules

* **`Q4_K_M` is the default sweet spot.** `Q5_K_M` when the VRAM is there.
* **Below Q4, degradation accelerates — and it hits reasoning and code harder than
  prose.** Which is exactly the capability this project cares about, so cheap quants
  are a false economy here.
* **At equal VRAM, a big model heavily quantized usually beats a small model lightly
  quantized.** **Measured** in this project: `qwen3-coder:30b` (MoE) beat the dense
  `qwen2.5-coder:14b` at a comparable footprint.

### imatrix

An *importance matrix* is a calibration pass over real text that measures which weights
actually matter, so the quantizer knows where to spend its bits. It costs a one-time
computation at quantization time and is otherwise free — community consensus is to
prefer imatrix ("weighted") quants over static ones at the same size
([mradermacher discussion](https://huggingface.co/mradermacher/Phi-4-reasoning-plus-i1-GGUF/discussions/1)).

---

## 2. Third-party quantizers

Model authors rarely publish good GGUF quants themselves. Three sources matter:

### bartowski and mradermacher

Both mass-produce imatrix GGUFs. On the difference: mradermacher's imatrix dataset is
roughly **double** bartowski's — it contains bartowski's as its first half — which is
the usual argument for preferring it. Measurements are not one-sided though: bartowski's
tend to lead on non-wiki test sets while mradermacher's *static* quants lead on
wiki.test, which at least argues against either being tuned to a benchmark
([discussion](https://huggingface.co/mradermacher/model_requests/discussions/1436)).

**For this project the difference is small enough that it must be measured, not
assumed** — see the open questions below.

### Unsloth Dynamic (`UD-*`)

`UD` = **Unsloth Dynamic**. Rather than one scheme for the whole model, each model gets
a **custom per-layer scheme**: the layers quantized in Gemma differ from those in Llama.
`UD-Q4_K_XL` promotes important matrices to Q5_K where it judges it safe, while
`Q4_K_M` uses Q6_K in those places and is therefore *larger*
([discussion](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/6)).

The claim is better accuracy at smaller size — Unsloth report `UD-Q4_K_XL` beating other
Q4 quants while being ~8 GB smaller on a large model
([Unsloth docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)). Plausible
and well-motivated, **and exactly the kind of vendor claim this project exists to
verify independently**.

### abliterated / uncensored

Not the same thing:

* **uncensored** — fine-tuned so the model stops refusing.
* **abliterated** — a **direct weight edit** that removes the refusal direction from the
  activation space. No retraining; the behaviour is ablated out.

Because abliteration edits weights along a direction found by analysis, it can damage
capability that has nothing to do with refusals. **Working hypothesis: for code
generation these are net negative.** Not yet measured; it is cheap to settle with
`agentic_rust` and `agentic_test_gen`.

---

## 3. The KV cache — the real constraint on 16 GB

**Yes, this is general to every transformer**, which is worth stating plainly because
it is the single most useful thing on this page.

Any autoregressive transformer caches the **key** and **value** projections of every
token it has already seen, at every layer, so it does not recompute them each step.
That cache is not optional — it is what makes generation linear instead of quadratic —
and it grows with context length **and** with the number of concurrent sequences.

Roughly:

```
KV bytes ≈ 2 × layers × kv_heads × head_dim × context × bytes_per_element × sequences
```

### What varies between models

The *existence* of the cache is universal. Its *size* is an architectural choice, and
the spread is enormous:

| attention scheme | effect on the cache | examples |
|---|---|---|
| **MHA** (multi-head) | one K/V per attention head — the expensive baseline | older models |
| **GQA** (grouped-query) | heads share K/V in groups — several times smaller | Llama 2/3, Mistral, Qwen2.5 |
| **MQA** (multi-query) | a single shared K/V | some smaller models |
| **MLA** (latent attention) | K/V compressed into a low-rank latent — dramatically smaller | DeepSeek V2/V3 |
| **sliding window** | local layers only attend to the last N tokens, so long context stops costing linearly | Mistral, Gemma 3 (5 local : 1 global) |

So "how much context can I afford" is not answerable from parameter count alone.

### What is *not* about the model at all

**Measured, and it cost this project a discarded experiment:** `qwen2.5-coder:14b` is
9 GB on disk but asked for ~18 GB and loaded 25% onto CPU. The cause was not the model
— it was Ollama reserving cache for **four parallel sequences** by default. One
sequence was in use.

The levers, in order of impact on this machine:

1. **`OLLAMA_NUM_PARALLEL=1`** — 4× less KV cache, free, and it also removes a source
   of run-to-run non-determinism (cache reuse stops depending on which slot served the
   request). Not yet applied here; it needs an Ollama restart.
2. **Quantize the KV cache** — llama.cpp's `--cache-type-k q8_0 --cache-type-v q8_0`,
   roughly halving it. Quality cost is small but real, so measure it.
3. **Lower the context** — `AI_BENCH_NUM_CTX`. At 4096 the 14B loads 100% on GPU (13 GB).

**A CPU-offloaded model is not merely slower — it is slow enough to hit request
timeouts, which the benchmark then records as the model failing the task.** Three
experiments were invalidated this way before `warn_if_cpu_offloaded()` existed.

### What actually fits on 16 GB — measured

| model | `num_ctx` | loaded | split |
|---|---|---|---|
| qwen2.5-coder:7b-instruct | 8192 | 8.2 GB | **100 % GPU** |
| qwen2.5-coder:14b | 8192 | 18 GB | 25 % CPU / 75 % GPU |
| qwen2.5-coder:14b | **4096** | 13 GB | **100 % GPU** (5 % CPU when the desktop is busy) |
| qwen3-coder:30b | 4096 | 20 GB | 33 % CPU / 67 % GPU |
| qwen3-coder:30b | 2048 | 19 GB | 27 % CPU / 73 % GPU (33 % when the desktop is busy) |

Note the 14B: **the same model, same quantization, goes from a third on CPU to
entirely on GPU purely by halving the context.** That is the KV cache, not the
weights, and it is why `AI_BENCH_NUM_CTX` exists.

The parenthesised numbers are the same models measured again on 2026-08-04 with a few
hundred MB more of the card in use by the desktop. **The split is not a property of the
model, it is decided at load time against whatever VRAM is free then** — so read this
table as "fits when the card is quiet", and check `ollama ps` for the run you are about
to trust rather than assuming last week's placement.

The 30B does not fit at any usable context — 19 GB against a 16 GB card even at
2048. It is still worth running (a MoE activates only a few experts per token, so
it tolerates offload better than a dense model would), but **the split must be
recorded next to any score it produces**, and it cannot be compared head to head
with a fully-resident model without that caveat.

### And do not build while measuring

The corollary, learned by walking into it: with part of a model on CPU, a
compilation on the same machine steals the cores its inference needs. It inflates
latency and can push generations into the client's 120 s ceiling, which surfaces
as excluded runs and a meaningless score. Same trap as a badly loaded model,
entered from the other side — a well-loaded model on a busy machine.

---

## 4. Context compression

Three different families, often conflated:

* **Cache quantization** — store K/V at lower precision. What llama.cpp offers today.
* **Cache eviction** — keep only part of the cache: StreamingLLM (attention sinks plus
  a recent window), H2O (the tokens that actually receive attention).
* **Application-level** — summarisation, retrieval, budget allocation. This is where
  this library's `ContextBudgetAllocator` and FreshContext mode already operate.

### TurboQuant (Google DeepMind, ICLR 2026)

Worth its own note because it is the current state of the art and it is **not** a weight
quantizer — it compresses the **KV cache**. Training-free and data-oblivious, built on
Quantized Johnson–Lindenstrauss and PolarQuant, reported at ~6× cache compression and
~8× faster attention, down to 3-bit cache without meaningful accuracy loss
([Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)).

**Why it is in vLLM but not in llama.cpp — and it is not a rejection.** The upstream
[discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) shows no formal
refusal; the work is **stalled at the fork stage**, with independent Metal, CUDA, Vulkan
and CPU implementations living outside the tree (also
[AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant)). The results are real —
4–5× cache compression, one report of output identical to the f16 baseline on a 35B model
at temperature 0, 262K context on multi-GPU.

Two barriers keep it out: every backend needs its own implementation, and — the decisive
one — **decode gets slower**, reported anywhere from 37% to 8× depending on hardware,
pending fused attention kernels.

That trade-off is exactly why the two engines diverge, and it is worth internalising
because it recurs:

* **vLLM** serves many concurrent long-context requests, where the KV cache is *the*
  binding constraint (PagedAttention exists for this reason). Paying decode speed to fit
  more sequences is a good trade.
* **llama.cpp** serves one latency-sensitive local user. An 8× decode slowdown to save
  memory the user may not be short of is a bad trade.

**Practical consequence for us: on the Ollama / llama.cpp path TurboQuant is not
available today**, and if it arrives it should be measured for throughput, not only for
the memory it frees.

And when it does arrive it is a **strategy, not a default** — the *last* lever on this
list, not the first. Trading decode speed for cache memory only pays when it is the
difference between a model fitting and not fitting; if the model already fits, it is a
pure loss. The cheaper levers come first because they cost nothing in decode:
`OLLAMA_NUM_PARALLEL=1` (4×) and then `q8_0` cache quantization (~2×). Reach for a
technique like this to unlock a 30B at long context that otherwise will not load at all.

---

## 5. Hardware number formats

### NVFP4 — and what it actually means for an Ada card

NVFP4 is NVIDIA's 4-bit floating-point format, introduced with **Blackwell**, where the
tensor cores execute it natively.

**It is nevertheless usable on Ada.** NVFP4 merged into llama.cpp over late March–April
2026, and vLLM supports NVFP4 checkpoints via llm-compressor
([llm-compressor docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)).
There is even an Ada-targeted runtime with custom decode kernels and an FP8 KV cache
([NVFP4-on-4090-vLLM](https://github.com/BenChaliah/NVFP4-on-4090-vLLM)).

The honest summary for a 4080 SUPER: **"FP4 is mostly a memory story — small bandwidth
speedup and no native tensor-core win"**
([InsiderLLM](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)). Older
cards get the memory savings, not the acceleration.

So the reported claim — *"Blackwell formats get much more optimisation attention, so
they may pay off even on Ada"* — is **half right**, and the halves matter:

* **Memory and possibly quality-per-bit: plausibly yes.** A format receiving heavy
  engineering effort can beat older 4-bit schemes at equal size on quality alone.
* **Speed on Ada: no.** The tensor cores cannot execute FP4; the win is bandwidth, not
  compute.

**This is measurable with the existing harness** and is the right way to settle it:
same model, same tasks, NVFP4 vs `Q4_K_M`, comparing both score and tokens/s.

### What supporting it would take here

Less than expected, and an earlier note in this project claiming "our stack could not
touch it" was **wrong**:

1. **The engine already supports it.** llama.cpp has NVFP4; no work needed there.
2. **This library already speaks to it.** `AiProvider::LlamaCpp` and `AiProvider::VLLM`
   exist, and the benchmark can be pointed at either with `AI_BENCH_PROVIDER=llamacpp`
   plus `AI_BENCH_URL`.
3. **What is actually missing** is operational, not architectural: obtaining NVFP4
   weights for the model of interest, and running a llama.cpp server build recent
   enough to include the format. For the vLLM path on Windows, add WSL2.

---

## 6. Open questions — to measure, not to assume

Each of these is a vendor or community claim that this project is equipped to verify:

1. **bartowski vs mradermacher vs Unsloth `UD-*`** at matched size, on `agentic_rust`
   and `agentic_test_gen`. Does the dynamic scheme's claimed edge survive an
   execution-verified benchmark?
2. **Where does quality actually fall off below Q4** for *code*, as opposed to the
   perplexity numbers usually quoted?
3. **Do abliterated variants lose coding capability?** Hypothesis: yes.
4. **KV cache quantization** (`q8_0`) — how much VRAM freed, how much score lost?
5. **`OLLAMA_NUM_PARALLEL=1`** — confirm the 4× cache saving, and whether it also
   removes the residual run-to-run verdict flipping.
6. **NVFP4 vs `Q4_K_M`** on this Ada card: score and throughput.

---

## 7. Terms that did not resolve

Recorded so the next person does not re-run the search:

* **"BitSeek v4"** — no such thing found. Most likely **DeepSeek V4**, misheard. The
  context fits: DeepSeek's **MLA** is the headline architectural answer to KV cache
  size, so it is the name that would come up in a conversation about context
  compression.
* **"Turbo Quan"** — this one is real: **TurboQuant**, see §4.
* **"Gemma 3 8B"** — Gemma 3 ships at 1B / 4B / 12B / 27B, plus Gemma 3n (E2B/E4B) for
  edge. There is no 8B; likely Gemma 2 9B or a community fine-tune.
* **"NVFP4 for Maxwell… Blackwell (N111, N200)"** — Maxwell is 2014-era (compute 5.x)
  and cannot be meant. "N111 / N200" is almost certainly **B100 / B200**.
* **"Boris"** as a quantizer — no match. Possibly a garbled reference; bartowski and
  mradermacher are the two names that consistently come up.
