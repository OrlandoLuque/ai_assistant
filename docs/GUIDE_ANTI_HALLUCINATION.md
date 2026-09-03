# Anti-Hallucination Guide

Every LLM lies. Not maliciously -- statistically. When a model lacks grounding
it fills the gap with plausible-sounding fiction, and your users pay the price.
McKinsey estimates that hallucination-related trust failures cost enterprises
**$1.2 trillion in unrealized AI value** annually. Medical chatbots invent drug
interactions. Legal copilots cite cases that never existed. Customer-support
bots fabricate refund policies.

This guide covers the **anti-hallucination pipeline** in `ai_assistant`: an
integrated, 11-step verification system that goes far beyond anything offered
by LangChain or LlamaIndex today. You will learn **what** each layer does,
**why** it matters, and **how** to wire it into your own application with
copy-paste commands.

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [How It Compares](#2-how-it-compares)
3. [The 11-Step Pipeline](#3-the-11-step-pipeline)
4. [Seven Strategies for Ungrounded Claims](#4-seven-strategies-for-ungrounded-claims)
5. [Configuration Reference](#5-configuration-reference)
6. [Faithfulness Scoring](#6-faithfulness-scoring)
7. [Chain-of-Verification (CoVe)](#7-chain-of-verification-cove)
8. [Quality Gates](#8-quality-gates)
9. [CLI Quick Start](#9-cli-quick-start)
10. [Server / API Integration](#10-server--api-integration)
11. [MCP Tools](#11-mcp-tools)
12. [TOML Configuration](#12-toml-configuration)
13. [Security Considerations](#13-security-considerations)
14. [Presets: One Line to Production](#14-presets-one-line-to-production)
15. [Cross-References](#15-cross-references)

---

## 1. Why This Matters

Hallucinations are not edge cases. They are the **default mode** of every
autoregressive language model. The model does not "know" facts; it predicts the
next token. When the training data is thin, the context is ambiguous, or the
temperature is high, the prediction drifts away from truth.

The consequences are domain-specific but universally expensive:

| Domain | Hallucination example | Cost |
|--------|----------------------|------|
| **Healthcare** | Invented drug interaction | Patient harm, liability |
| **Legal** | Fabricated case citation | Sanctions, malpractice |
| **Finance** | Fictional regulation reference | Compliance violation |
| **Customer support** | Non-existent refund policy | Revenue loss, churn |
| **Engineering** | Wrong API parameter | Production outage |

Traditional mitigations -- "just use RAG" or "add a system prompt saying be
accurate" -- reduce but do not solve the problem. RAG helps when the answer is
in the retrieved context, but the model can still hallucinate *around* the
context, inject details that are not there, or confidently answer when no
relevant context was retrieved at all.

What you need is a **multi-layered verification pipeline** that catches
hallucinations at every stage: before generation (temperature control), during
generation (abstention), and after generation (faithfulness scoring, claim
verification, quality gates).

That is exactly what `ai_assistant` provides.

---

## 2. How It Compares

| Capability | ai_assistant | LangChain | LlamaIndex |
|------------|:------------:|:---------:|:----------:|
| Integrated pipeline (single crate) | 11 steps | No | No |
| Ungrounded claim strategies | 7 | 0 | 0 |
| Calibrated abstention | Yes | No | No |
| Auto-temperature for factual queries | Yes | No | No |
| Chain-of-Verification (CoVe) | Built-in | Manual | No |
| Self-consistency (multi-sample) | Built-in | Manual | No |
| Faithfulness NLI scoring | 2 methods | Partial | Partial |
| Quality gates (CI/CD-ready) | 5 metrics, 3 actions | No | No |
| MCP tool integration | 3 tools | N/A | N/A |
| HTTP API with risk headers | Yes | No | No |
| LLM call budget cap | Yes | No | No |
| Prompt injection protection in claims | Yes | No | No |
| Zero-cost faithfulness option | WordOverlap | No | No |

**LangChain** offers fragmented hallucination detection spread across multiple
packages with no unified pipeline, no strategy system, and no abstention.
Building equivalent functionality requires assembling dozens of third-party
integrations and writing significant custom glue code.

**LlamaIndex** has limited faithfulness evaluation through its evaluation
module, but lacks a hallucination pipeline, claim strategies, CoVe, quality
gates, and abstention.

**ai_assistant** provides all of the above in a single Rust crate, with type
safety, zero-cost abstractions where possible, and strict LLM call budgeting
to prevent runaway costs.

---

## 3. The 11-Step Pipeline

Every response passes through up to 11 verification stages. Steps can be
individually enabled or disabled. The pipeline is designed so that cheap checks
run first and expensive LLM calls are deferred as late as possible.

```
Query
  |
  v
[1] Auto-temperature -----> Factual query? Set temp=0.3
  |
  v
[2] LLM generation -------> Produce raw response
  |
  v
[3] Abstention check -----> Confidence < 0.3? Refuse to answer
  |
  v
[4] Claim decomposition --> Extract atomic claims from response
  |
  v
[5] Faithfulness scoring -> NLI check against source context
  |
  v
[6] Grounded generation --> Anchor each sentence to source chunks
  |
  v
[7] Fact-check (search) --> Only for NliVerdict::Neutral claims
  |
  v
[8] Chain-of-Verification -> Generate questions, answer independently, compare
  |
  v
[9] Self-consistency -----> Multi-sample, divergence metrics
  |
  v
[10] Apply strategy ------> Omit / Mark / Warn / Footnote / Verify / Ask
  |
  v
[11] Quality gates -------> Fail / Warn / Log based on thresholds
  |
  v
Verified Response
```

### Step 1: Auto-Temperature

**What.** When enabled, the pipeline detects whether the query is factual (e.g.
"What is the boiling point of water?") versus creative (e.g. "Write a poem
about water"). Factual queries automatically use a lower temperature
(default `0.3`) to reduce sampling randomness.

**Why.** High temperature is the single largest source of hallucination in
factual queries. Reducing it from the default 0.7-1.0 to 0.3 dramatically
cuts fabrication rates without any additional LLM calls.

### Step 2: LLM Generation

The query is sent to the configured provider (Ollama, OpenAI, Anthropic, etc.)
and the raw response is captured.

### Step 3: Abstention Check

**What.** The model's confidence is evaluated. If it falls below the threshold
(default `0.3`), the pipeline **refuses to answer** rather than risk a
hallucinated response.

**Why.** The most dangerous hallucination is the one delivered with high
apparent confidence. Abstention is the only fully reliable defense: when the
model does not know, it should say so. This is calibrated -- the threshold is
tunable per domain.

### Step 4: Claim Decomposition

**What.** The response is broken into **atomic claims** -- individual factual
statements that can be independently verified.

**How.** Two methods are available:
- `SentenceSplit` -- zero-cost sentence boundary detection
- `LlmDecomposition` -- uses one LLM call for precise claim extraction

**Example.** The sentence "Paris, the capital of France, has a population of
2.1 million" becomes two claims: "Paris is the capital of France" and "Paris
has a population of 2.1 million."

### Step 5: Faithfulness Scoring

**What.** Each claim is scored against the source context using Natural
Language Inference (NLI). The result is one of three verdicts:

- **Entailed** -- the claim is supported by the source context
- **Contradicted** -- the claim conflicts with the source context
- **Neutral** -- the claim is neither supported nor contradicted

**How.** Two methods:
- `WordOverlap` -- Jaccard similarity, zero LLM calls, good for fast filtering
- `LlmNli` -- one LLM call per batch, higher accuracy

### Step 6: Grounded Generation Check

**What.** Every sentence in the response is mapped back to the source chunks
that support it. Sentences that cannot be anchored to any source are flagged as
ungrounded.

**Why.** This is the core "show your work" mechanism. It ensures that the model
is not inventing information that goes beyond what the retrieved context
provides.

### Step 7: Fact-Check with Search

**What.** Claims that received an `NliVerdict::Neutral` (neither supported nor
contradicted by context) are sent to a search backend for external
verification.

**Why.** Neutral claims are the gray area -- they might be true but are not
grounded in the provided context. External search resolves the ambiguity
without discarding potentially valid information.

### Step 8: Chain-of-Verification (CoVe)

**What.** For claims that remain unverified, the pipeline generates
**verification questions**, answers them **independently** (without seeing the
original response), and compares the independent answers against the original
claims.

**Example.** Original claim: "The Great Wall of China is 21,196 km long."
Verification question: "How long is the Great Wall of China?" Independent
answer: "The Great Wall is approximately 21,196 km." Result: **Supported**.

**Why.** CoVe catches the class of hallucinations where the model is
self-consistent within a single response but factually wrong. By generating
answers independently, it breaks the self-reinforcement loop.

### Step 9: Self-Consistency

**What.** The same query is sent to the model multiple times (with different
random seeds). The responses are compared using divergence metrics. High
divergence indicates low reliability.

**Why.** If the model gives a different answer every time you ask, neither
answer is trustworthy. Self-consistency quantifies this instability.

### Step 10: Apply Strategy

The configured `UngroundedClaimStrategy` is applied to any claims that remain
ungrounded after all verification steps. See [Section 4](#4-seven-strategies-for-ungrounded-claims)
for details on all seven strategies.

### Step 11: Quality Gates

Five metrics are checked against configurable thresholds. Each gate can
**Fail** (block the response), **Warn** (add a warning), or **Log** (record
silently). See [Section 8](#8-quality-gates) for details.

---

## 4. Seven Strategies for Ungrounded Claims

When a claim cannot be verified, `ai_assistant` offers seven strategies for
handling it. The strategy is configured via the `UngroundedClaimStrategy` enum:

| Strategy | What it does | Best for | Trade-off |
|----------|-------------|----------|-----------|
| **Omit** | Remove unverified claims entirely | Medical, legal, compliance | May lose valid info |
| **Mark** | Tag with `[unverified]` marker | General use **(DEFAULT)** | User sees noise |
| **Warn** | Emit warning, keep claim intact | Development, debugging | No user-facing protection |
| **Footnote** | Add explanatory footnotes | Academic, research | Longer output |
| **VerifyThenMark** | Verify first, then mark if still ungrounded | Balanced accuracy + visibility | Extra LLM calls |
| **VerifyThenOmit** | Verify first, then remove if still ungrounded | High-stakes with verification budget | Extra LLM calls, may lose info |
| **Ask** | Ask user to confirm/reject each claim | Interactive, human-in-the-loop | Requires user interaction |

### Choosing a Strategy

```
High stakes, no user interaction?     --> Omit or VerifyThenOmit
High stakes, user available?          --> Ask or VerifyThenMark
General purpose?                      --> Mark (default)
Academic / research?                  --> Footnote
Development / testing?                --> Warn
Need verification but want visibility? --> VerifyThenMark
```

### Example: Mark Strategy Output

```
The Earth orbits the Sun at an average distance of 149.6 million km.
[unverified] The exact orbital period is 365.256363004 days.
The Moon is the Earth's only natural satellite.
```

### Example: Footnote Strategy Output

```
The Earth orbits the Sun at an average distance of 149.6 million km.
The exact orbital period is 365.256363004 days.[1]
The Moon is the Earth's only natural satellite.

---
[1] This claim could not be verified against the provided sources.
    Confidence: 0.45. Recommend independent verification.
```

---

## 5. Configuration Reference

The anti-hallucination pipeline is configured via `AntiHallucinationConfig`:

```rust
pub struct AntiHallucinationConfig {
    /// Master switch for the pipeline (default: false)
    pub enabled: bool,

    /// Strategy for ungrounded claims (default: Mark)
    pub ungrounded_strategy: UngroundedClaimStrategy,

    /// Enable abstention -- refuse to answer on low confidence (default: false)
    pub abstention_enabled: bool,

    /// Confidence threshold below which the model refuses to answer (default: 0.3)
    pub abstention_threshold: f64,

    /// Enable confidence scoring on responses (default: true)
    pub confidence_scoring_enabled: bool,

    /// Auto-adjust temperature for factual queries (default: false)
    pub auto_temperature_enabled: bool,

    /// Temperature to use for factual queries (default: 0.3)
    pub factual_query_temperature: f32,

    /// Format string for the [unverified] marker (default: "[unverified] {}")
    pub mark_format: String,

    /// Minimum confidence score to include a claim in output (default: 0.3)
    pub min_confidence_for_output: f64,

    /// Maximum additional LLM calls allowed for verification (default: 5)
    pub max_extra_llm_calls: usize,

    /// Custom message when the model abstains (optional)
    pub abstention_message: Option<String>,
}
```

### Key Design Decisions

- **`enabled` defaults to `false`** -- the pipeline is opt-in. You must
  explicitly enable it. This is intentional: verification adds latency and LLM
  calls, and not every use case needs it.

- **`max_extra_llm_calls` defaults to `5`** -- this is a hard budget cap that
  prevents runaway costs. CoVe, faithfulness scoring, and self-consistency all
  consume LLM calls. The budget ensures that verification never exceeds a
  predictable cost envelope.

- **`mark_format` is customizable** -- you can change `[unverified] {}` to
  any format your downstream UI expects, such as
  `<span class="unverified">{}</span>` for HTML rendering.

---

## 6. Faithfulness Scoring

Faithfulness scoring determines whether a response is supported by the provided
context. This is the core mechanism for detecting RAG hallucinations -- cases
where the model has context but ignores or embellishes it.

### NLI Verdicts

Every claim receives one of three verdicts:

| Verdict | Meaning | Action |
|---------|---------|--------|
| `Entailed` | Claim is supported by context | Pass through |
| `Contradicted` | Claim conflicts with context | Flag or remove |
| `Neutral` | Neither supported nor contradicted | Escalate to fact-check |

### NLI Methods

| Method | Cost | Accuracy | Use when |
|--------|------|----------|----------|
| `WordOverlap` | Zero LLM calls (Jaccard similarity) | Good for filtering | Latency-sensitive, high volume |
| `LlmNli` | 1 LLM call per batch | High | Accuracy-critical, low volume |

### Decomposition Methods

| Method | Cost | Description |
|--------|------|-------------|
| `SentenceSplit` | Zero LLM calls | Split on sentence boundaries |
| `LlmDecomposition` | 1 LLM call | LLM extracts atomic claims precisely |

### FaithfulnessConfig

```rust
pub struct FaithfulnessConfig {
    /// Minimum faithfulness score to pass (default: 0.7)
    pub min_faithfulness_score: f64,
    // ...
}
```

A faithfulness score of `0.7` means that at least 70% of the claims in the
response must be entailed by the source context. Claims that are `Contradicted`
count against the score; `Neutral` claims are escalated to fact-checking.

### Zero-Cost Path

For high-throughput applications (chat, search, content generation), use
`WordOverlap` + `SentenceSplit`. This combination performs faithfulness
scoring with **zero additional LLM calls** -- the only cost is CPU time for
Jaccard similarity computation. This is unique to `ai_assistant`; neither
LangChain nor LlamaIndex offers a zero-LLM-call faithfulness path.

---

## 7. Chain-of-Verification (CoVe)

CoVe is a research-backed technique (Wei et al., 2023) for catching
hallucinations that survive simpler checks. It works by:

1. **Extracting claims** from the response
2. **Generating verification questions** for each claim
3. **Answering those questions independently** (the model does not see the
   original response)
4. **Comparing** the independent answers to the original claims

### Configuration

```rust
pub struct CoVeConfig {
    /// Maximum claims to verify per response (default: 10)
    pub max_claims_to_verify: usize,
    // ...
}
```

### Verification Sources

| Source | Description |
|--------|-------------|
| `RagOnly` | Verify against RAG context only |
| `WebSearchOnly` | Verify against web search results |
| `RagThenWeb` | Try RAG first, fall back to web |
| `Both` | Use both RAG and web search |

### Correction Modes

| Mode | Description |
|------|-------------|
| `Replace` | Replace incorrect claims with corrected versions |
| `Annotate` | Add annotations to claims **(default)** |
| `Footnote` | Add footnotes with verification details |

### Claim Verification Statuses

| Status | Meaning |
|--------|---------|
| `Supported` | Independent verification confirms the claim |
| `Contradicted` | Independent verification refutes the claim |
| `Unverifiable` | Cannot be verified with available sources |
| `PartiallySupported` | Some aspects verified, others not |

### Example Flow

```
Original response: "Python was created by Guido van Rossum in 1989."

Step 1 - Extract claim:
  "Python was created by Guido van Rossum in 1989"

Step 2 - Generate verification question:
  "When was Python created and by whom?"

Step 3 - Answer independently:
  "Python was conceived in the late 1980s by Guido van Rossum.
   Implementation began in December 1989."

Step 4 - Compare:
  Verdict: Supported
  Note: The year 1989 refers to when implementation started;
  the language was first released in 1991.
```

### LLM-backed verification (V132+)

By default `verify_claim()` falls back to word-overlap (Jaccard)
between the claim and the verification context — fast and
deterministic but blind to paraphrase. Attach an LLM callback
via `with_llm_verifier` to get the full CoVe behaviour: the
engine sends a *Supported / Contradicted / Unsupported* prompt
to the model and uses its verdict.

```rust
let cove_engine = ChainOfVerification::new(cove_config)
    .with_llm_verifier(|prompt: &str| -> Option<String> {
        // Call your LLM here. Return the response, or None on timeout.
        my_llm_client.complete(prompt).ok()
    });
```

`ai_cli verify --cove --knowledge <file>` wires this for you and
sets `verification_source = VerificationSource::Both` so that
file-sourced contexts are not filtered out by the source-type
gate.

The two demo knowledge files
(`examples/knowledge_earth.txt`, `examples/knowledge_rust.txt`)
let you try this end-to-end:

```bash
ai_cli verify --provider ollama --model "mistral:7b-instruct" \
  --knowledge examples/knowledge_earth.txt \
  --cove --quality-gates \
  "Tell me about Earth"
```

---

## 8. Quality Gates

Quality gates are threshold-based checks that run at the end of the pipeline.
They are designed to be used in both runtime and CI/CD contexts.

### Five Metrics

| Metric | What it measures | Typical threshold |
|--------|-----------------|-------------------|
| `Faithfulness` | % of claims entailed by context | 0.7 |
| `Confidence` | Model's self-assessed confidence | 0.5 |
| `GroundingRatio` | % of sentences anchored to sources | 0.6 |
| `ConsistencyScore` | Agreement across multi-sample runs | 0.8 |
| `CitationCoverage` | % of claims with source citations | 0.5 |

### Three Actions

| Action | Behavior |
|--------|----------|
| `Fail` | Block the response entirely. Return an error. |
| `Warn` | Add a warning header/footer but deliver the response. |
| `Log` | Record the metric silently. No user-visible effect. |

### Production Defaults

```rust
let gates = QualityGateRunner::production_defaults();
```

This configures sensible thresholds for production use:
- Faithfulness >= 0.7 --> Fail
- Confidence >= 0.5 --> Warn
- GroundingRatio >= 0.6 --> Warn

### CI/CD Usage

Quality gates can be integrated into your CI/CD pipeline to gate deployments.
If your LLM-powered feature fails the faithfulness gate during integration
tests, the build fails -- just like a unit test failure.

`--exit-code-on-fail` is what makes that true, and it is opt-in: without it the
gate result is printed and the process still exits 0, which is fine
interactively and useless in CI. It was documented here before it existed (V292),
so a pipeline copied from this page would have gone green on every failure.

```bash
# Run quality gates as a CI check
cargo run --bin ai_cli --features full -- verify \
    "Test query for the deployment" \
    --quality-gates \
    --min-confidence 0.5 \
    --exit-code-on-fail
```

---

## 9. CLI Quick Start

All anti-hallucination features are accessible from the command line via
`ai_cli`. These commands are copy-paste ready.

### Basic Verification

```bash
# Verify a factual claim with the default Mark strategy
cargo run --bin ai_cli --features full -- verify \
    "Is water wet?" \
    --strategy mark
```

### Full Pipeline with Faithfulness + Quality Gates

```bash
# High-stakes verification: faithfulness scoring + quality gates
cargo run --bin ai_cli --features full -- verify \
    "What are the side effects of ibuprofen?" \
    --strategy verify-mark \
    --faithfulness \
    --quality-gates \
    --min-confidence 0.5
```

### Chain-of-Verification

```bash
# Verify a commonly hallucinated claim with CoVe + footnotes
cargo run --bin ai_cli --features full -- verify \
    "The Great Wall of China is visible from space" \
    --cove \
    --strategy footnote
```

### Strict Mode (Medical / Legal)

```bash
# Remove all unverified claims, require high faithfulness
cargo run --bin ai_cli --features full -- verify \
    "List the contraindications of metformin" \
    --strategy verify-omit \
    --faithfulness \
    --quality-gates \
    --min-confidence 0.7
```

---

## 10. Server / API Integration

The embedded HTTP server exposes verification as REST endpoints.

### Starting the Server with Verification

```bash
cargo run --bin ai_cli --features full -- serve \
    --enable-verification \
    --verification-strategy mark
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/verify/faithfulness` | Evaluate faithfulness of a response against context |
| `POST` | `/api/v1/verify/quality-check` | Run quality gates on text |
| `GET` | `/api/v1/verify/config` | Retrieve current verification configuration |

### Response Headers

Every response from the verification endpoints includes diagnostic headers:

| Header | Example | Description |
|--------|---------|-------------|
| `X-Quality-Score` | `0.85` | Overall quality score (0.0 - 1.0) |
| `X-Faithfulness` | `0.92` | Faithfulness score |
| `X-Hallucination-Risk` | `low` | Risk level: `low`, `medium`, `high` |

### Example: POST /api/v1/verify/faithfulness

```bash
curl -X POST http://localhost:8080/api/v1/verify/faithfulness \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Paris is the capital of France with 2.1M people.",
    "context": "Paris is the capital and largest city of France.",
    "method": "word_overlap"
  }'
```

Response:

```json
{
  "faithfulness_score": 0.75,
  "claims": [
    {
      "text": "Paris is the capital of France",
      "verdict": "Entailed",
      "confidence": 0.95
    },
    {
      "text": "Paris has 2.1M people",
      "verdict": "Neutral",
      "confidence": 0.40
    }
  ]
}
```

---

## 11. MCP Tools

Three MCP tools are exposed for integration with MCP-compatible clients:

| Tool | Description |
|------|-------------|
| `check_faithfulness` | Evaluate the faithfulness of a response against provided context |
| `verify_claims` | Run the CoVe pipeline on a set of claims |
| `run_quality_gates` | Run quality gates on arbitrary text |

These tools are available when the MCP server is started with verification
features enabled. They follow the standard MCP tool calling protocol and can
be used by any MCP-compatible client (Claude Desktop, VS Code extensions,
custom agents, etc.).

---

## 12. TOML Configuration

For applications that use file-based configuration, the full pipeline can be
configured via TOML:

```toml
[anti_hallucination]
enabled = true
ungrounded_strategy = "mark"
abstention_enabled = true
abstention_threshold = 0.3
auto_temperature_enabled = true
min_confidence_for_output = 0.3
max_extra_llm_calls = 5

[[quality_gates]]
name = "faithfulness"
metric = "Faithfulness"
threshold = 0.7
action = "Fail"

[[quality_gates]]
name = "confidence"
metric = "Confidence"
threshold = 0.5
action = "Warn"

[[quality_gates]]
name = "grounding"
metric = "GroundingRatio"
threshold = 0.6
action = "Warn"
```

### Configuration Precedence

1. **Programmatic** (`AntiHallucinationConfig` struct) -- highest priority
2. **TOML file** -- loaded at startup
3. **CLI flags** (`--strategy`, `--min-confidence`, etc.)
4. **Defaults** -- built-in defaults (see [Section 5](#5-configuration-reference))

---

## 13. Security Considerations

The anti-hallucination pipeline is itself a potential attack surface. These
safeguards are built in:

### LLM Call Budget

- `max_extra_llm_calls = 5` (default) -- hard cap on additional LLM calls for
  verification. Prevents adversarial inputs from triggering unbounded LLM
  usage.
- CoVe: `max_claims_to_verify = 10` -- caps the number of claims that enter
  the expensive verification path.
- Self-consistency: max samples = 5 (hard cap 10) -- limits multi-sampling.

### Prompt Injection Protection

Claim decomposition uses **delimiter-based isolation** to prevent prompt
injection via claim text. When claims are sent to the LLM for NLI scoring or
CoVe verification, they are wrapped in structured delimiters that the model
is instructed to treat as data, not instructions.

### Denial-of-Service Mitigation

All verification steps have bounded computation:
- Sentence splitting caps at document length
- Graph traversal is depth-limited
- Search-based fact-checking respects rate limits
- Total pipeline latency is bounded by `max_extra_llm_calls`

---

## 14. Presets: One Line to Production

For common use cases, `ai_assistant` provides one-liner presets that configure
the entire pipeline with battle-tested defaults:

### `AntiHallucinationConfig::production()`

Balanced configuration for production workloads:
- Strategy: `Mark`
- Abstention: enabled at 0.3
- Auto-temperature: enabled
- Confidence scoring: enabled
- Max extra LLM calls: 5

```rust
let config = AntiHallucinationConfig::production();
```

### `AntiHallucinationConfig::strict()`

Maximum safety for high-stakes domains (medical, legal, financial):
- Strategy: `VerifyThenOmit`
- Abstention: enabled at higher threshold
- All verification layers enabled
- Stricter quality gates

```rust
let config = AntiHallucinationConfig::strict();
```

### `AntiHallucinationConfig::permissive()`

Minimal verification for low-stakes or development use:
- Strategy: `Warn`
- Abstention: disabled
- Confidence scoring only
- No quality gates

```rust
let config = AntiHallucinationConfig::permissive();
```

### Quality Gate Preset

```rust
let gates = QualityGateRunner::production_defaults();
// Faithfulness >= 0.7 -> Fail
// Confidence  >= 0.5 -> Warn
// Grounding   >= 0.6 -> Warn
```

---

## 15. Dataset Benchmarks (V90)

V90 adds a harness for running the pipeline against standard community
benchmarks so you can produce numbers that are directly comparable with
published model cards and prior art.

### Registered benchmarks

| Name | Sample type | License | Opt-in |
|------|-------------|---------|--------|
| `truthfulqa`   | QA (correct vs. incorrect references)       | Apache-2.0      | no  |
| `halueval_qa`  | right vs. hallucinated answer pairs          | MIT             | no  |
| `factscore`    | atomic-claim decomposition (bios)            | MIT             | no  |
| `ragas_wikiqa` | contextual QA (question + context + answer)  | Apache-2.0      | no  |
| `fever`        | claim vs. evidence (Supports/Refutes/NEI)    | CC-BY-SA 3.0    | yes |

Opt-in datasets require `--accept-license` on download. Nothing is vendored;
all data is fetched on explicit user action and cached under
`$CARGO_TARGET_DIR/eval_benchmarks/<loader>/`.

### CLI

```bash
ai_cli benchmark list
ai_cli benchmark info truthfulqa
ai_cli benchmark download fever --accept-license

ai_cli benchmark run truthfulqa \
    --provider ollama --model mistral:7b-instruct --limit 50

ai_cli benchmark calibrate halueval_qa \
    --provider ollama --model llama3.2 \
    --limit 200 --objective f1 --json
```

`run` reports total, correct, accuracy, mean score, per-category breakdown,
and the wall-clock duration. `calibrate` adds a post-hoc threshold sweep so
you can see where accuracy / F1 peaks for a given model without re-running.

### HTTP

```
GET /benchmarks             → { "total": N, "benchmarks": [...] }
GET /benchmarks/<name>      → metadata or 404
GET /api/v1/benchmarks...   → same, versioned prefix
```

Both are read-only and return JSON; they only surface the loader registry,
not download or run state, so they are safe to expose to any caller.

### MCP

Two tools are registered via
`ai_assistant::mcp_protocol::register_benchmark_tools(&mut server)`:

- `list_benchmarks` — enumerate all loaders (annotated read-only + idempotent).
- `get_benchmark(name)` — metadata lookup; returns `found: false` when absent.

### Programmatic use

```rust
use ai_assistant::eval_benchmarks::{
    get_loader, run, sweep, default_grid, Objective, RunOptions, report,
};

let loader = get_loader("truthfulqa").unwrap();
let path = loader.download(cache.dir_for(loader.name())?)?;
let samples = loader.load(&path, Some(100))?;

let r = run(loader.name(), &samples, &RunOptions::default(), |prompt| {
    my_llm_call(prompt) // Result<String, String>
});

println!("{}", report::to_text(&r));
let best = sweep(&r, &default_grid(), Objective::F1).best;
println!("Best F1 threshold: {:.2}", best.threshold);
```

## 16. Cross-References

- **Concepts**: [`docs/CONCEPTS.md`](CONCEPTS.md), sections 269-273 cover the
  theoretical foundations of hallucination, NLI, and claim verification.
- **Use Cases**: [`docs/USE_CASES.md`](USE_CASES.md), scenario #10 demonstrates
  anti-hallucination in an end-to-end deployment.
- **Changelog**: [`CHANGELOG.md`](../CHANGELOG.md) tracks when each pipeline
  step was introduced.
- **Implementation History**:
  [`docs/IMPROVEMENTS_V81.md`](IMPROVEMENTS_V81.md) through
  [`docs/IMPROVEMENTS_V90.md`](IMPROVEMENTS_V90.md) document the iterative
  development of the anti-hallucination pipeline, with V90 covering the
  dataset-benchmark harness described above.

---

## Summary

The `ai_assistant` anti-hallucination pipeline is not a single check or a
bolted-on afterthought. It is an **11-step, defense-in-depth system** that
addresses every class of hallucination:

- **Pre-generation**: Auto-temperature reduces randomness for factual queries
- **During generation**: Abstention refuses to answer when confidence is low
- **Post-generation**: Faithfulness scoring, grounding checks, fact-checking,
  CoVe, and self-consistency catch fabrications that slip through
- **Policy enforcement**: Seven strategies let you choose the right trade-off
  for your domain
- **Quality gates**: Measurable thresholds that can fail builds, warn users,
  or log silently

All of this ships in a single Rust crate with zero-cost options for
latency-sensitive workloads, strict LLM call budgets for cost control, and
production-ready presets for one-line setup.

If you are building an LLM application where accuracy matters -- and it always
does -- this is the pipeline to use.
