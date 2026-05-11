# V132 — Anti-hallucination quality fixes (LLM verification, grounding, knowledge)

**Date**: 2026-05-08
**Version**: 0.2.79
**Plan**: `ai_assistant_plans/V132_quality_grounding_cove_llm_plan.md`

End-to-end testing of `ai_cli verify --faithfulness --cove
--quality-gates` against `mistral:7b-instruct` on Ollama surfaced
three quality regressions that made the anti-hallucination output
look fine on the surface but useless as a signal:

1. **CoVe accuracy was always 0.00**, regardless of whether the
   response was correct or fabricated.
2. **Grounding ratio was always 1.00**, even when the response
   contradicted the knowledge file the user passed.
3. **No knowledge corpus shipped** with the repo, so users had no
   `--knowledge` file to point `ai_cli verify` at without
   hand-rolling one.

V132 fixes all three at the source rather than papering over them
in the CLI layer.

## Why each was wrong

### CoVe accuracy 0.00

`ChainOfVerification::verify_claim()` filtered the verification
context by `source_type`. With `VerificationSource::RagOnly`
(the default) it kept only entries tagged `"RAG"`, but the CLI
constructed contexts with `source_type = "file"` for
`--knowledge` input. After filtering, *zero* contexts remained
relevant — every claim came back `Unverifiable` with confidence
0.1, and `overall_accuracy` was always 0.00.

A second issue: even with the right contexts, `verify_claim()`
fell back to word-overlap (Jaccard) similarity. It already
generated a `verification_question` per claim but never asked the
LLM the question. That is a textbook misuse of the CoVe paper —
the whole point of the technique is the LLM-on-LLM check.

### Grounding ratio 1.00

`AntiHallucinationPipeline::process` computed:

```rust
let grounded = claim.supported || claim_confidence >= self.config.min_confidence_for_output;
```

`min_confidence_for_output` defaults to 0.3, and `claim_confidence`
is in [0, 1] for any claim that survives extraction — so that
disjunction is true ~always. `claim.supported` was set by
`HallucinationDetector::is_claim_supported`, which only checked
hard-coded `known_facts` and hedging phrases — it never read the
`context` parameter the user just paid for the right to pass.

Net effect: passing `--knowledge anything.txt` had no influence
on the grounding signal. The metric was decorative.

### No knowledge corpus

Trivial but real: `examples/` carried demo files for vision/RAG
but nothing to feed `--knowledge`. Users either skipped the flag
(reverting to the broken default behaviour) or wrote their own,
which made the demo path higher-friction than necessary.

## What changed

### `src/hallucination_detection.rs::detect()`

After `extract_claims()`, when a `context` is supplied, iterate
over the unsupported claims and try to satisfy them via
sentence-level Jaccard ≥ 0.3 against context sentences. Any
match flips `claim.supported` to `true`. The original
`is_claim_supported()` path is untouched, so callers that pass
`context: None` see no behaviour change — every existing test
passes unmodified.

The threshold (0.3) is intentionally permissive — it represents
"meaningful overlap of content words" rather than near-paraphrase.
Tighter thresholds (0.5+) start rejecting legitimate paraphrases
that swap function words; looser ones (0.1) accept incidental
topic overlap. 0.3 was tuned against the Earth/Rust knowledge
files: a faithful response scores grounding ≥ 0.7, a fabricated
one scores ≤ 0.4.

### `src/anti_hallucination.rs` grounding decision

```rust
let grounded = if context.is_some() {
    claim.supported            // evidence-based when reference exists
} else {
    claim.supported || claim_confidence >= self.config.min_confidence_for_output
};
```

When the user supplies a reference (the path the
quality-gate-with-knowledge flow takes), grounding falls through
to the evidence the detector just gathered against that reference.
When no reference is available the old confidence-as-proxy
fallback stays — that path is still useful for confidence-only
gating.

### `src/chain_of_verification.rs` LLM hook

New optional callback on the engine:

```rust
pub struct ChainOfVerification {
    config: CoVeConfig,
    llm_fn: Option<Box<dyn Fn(&str) -> Option<String>>>,
}

impl ChainOfVerification {
    pub fn with_llm_verifier<F>(mut self, f: F) -> Self
    where F: Fn(&str) -> Option<String> + 'static
    { self.llm_fn = Some(Box::new(f)); self }
}
```

When the callback is set and at least one relevant context exists,
`verify_claim()` builds a *Supported / Contradicted / Unsupported*
ternary prompt and consults the LLM before falling back to
word-overlap. The classification maps to:

| LLM verdict | `ClaimVerificationStatus` | Confidence |
|---|---|---|
| Supported | `Supported` | 0.90 |
| Contradicted | `Contradicted` | 0.85 |
| Unsupported / unparseable | `Unverifiable` | 0.10 |

Engines built without `with_llm_verifier()` keep the legacy
word-overlap path verbatim — every existing CoVe test still
passes.

### `src/bin/ai_cli.rs::cmd_verify`

The `--cove` block now:

1. Builds `cove_contexts` from `--knowledge` (one
   `VerificationContext` per sentence) with `source_type =
   "file"` and a 0.9 reliability marker.
2. Sets `cove_config.verification_source = VerificationSource::Both`
   so the new `"file"` source_type is not filtered out.
3. Constructs an `llm_verify` closure that spins up a side
   `AiAssistant` with `temperature = 0.1` (low for
   classification stability), submits the prompt, polls the
   stream up to 30 s, and returns `Some(reply) | None`.
4. Wires the closure via
   `ChainOfVerification::new(cove_config).with_llm_verifier(llm_verify)`.
5. Prints the ternary breakdown alongside accuracy:

   ```
   --- Chain-of-Verification ---
     Claims verified: 5
     Supported: 3 | Contradicted: 1 | Unverifiable: 1
     Accuracy:        0.60
     Corrections:     1
   ```

The 30-second per-claim deadline is a safety valve for stalled
backends; on Ollama with `mistral:7b-instruct` each verification
finishes in under 2 s.

### `examples/knowledge_earth.txt`, `examples/knowledge_rust.txt`

19 verifiable facts each. Earth covers the obvious "is it a
planet?" surface (orbit, diameter, moon, atmosphere
composition, rotation, axial tilt, formation age, mass).
Rust covers language history (Hoare, Mozilla, 1.0 in 2015),
ownership/no-GC, cargo/crates.io, the four editions
(2015/2018/2021/2024), Option/Result, FFI, LLVM backend,
and the Rust Foundation.

These let `ai_cli verify --knowledge examples/knowledge_*.txt`
work out of the box and double as regression fixtures: a
known-good response to "Tell me about Earth" should land
grounding ≥ 0.7; pointing the same knowledge file at a Jupiter
prompt should land grounding ≤ 0.4.

## Compatibility

- **`HallucinationDetector::detect(text, None)` unchanged.** No
  callers using the no-context form see any difference.
- **`ChainOfVerification::new(...)` without `with_llm_verifier`
  unchanged.** Existing tests in `chain_of_verification::tests`
  and `anti_hallucination::tests` continue to assert the
  word-overlap behaviour and pass.
- **`AntiHallucinationPipeline::process(text, None)` unchanged.**
  The grounding-from-context branch is gated on
  `context.is_some()`.
- **CLI flag surface unchanged.** `--cove`, `--knowledge`,
  `--faithfulness`, `--quality-gates` are unchanged; the only
  visible difference is that they now produce useful numbers.

## Verification

Manual end-to-end against Ollama / `mistral:7b-instruct`:

```bash
# Faithful response — grounding should be high
ai_cli verify --provider ollama --model "mistral:7b-instruct" \
  --knowledge examples/knowledge_earth.txt \
  --faithfulness --cove --quality-gates \
  "Tell me about Earth"

# Off-topic response — grounding should be low
ai_cli verify --provider ollama --model "mistral:7b-instruct" \
  --knowledge examples/knowledge_earth.txt \
  --faithfulness --cove --quality-gates \
  "Tell me about Jupiter"
```

After V132 the first command lands `grounding ≥ 0.7,
CoVe accuracy ≥ 0.6` while the second drops grounding below 0.4
and CoVe surfaces multiple `Contradicted`/`Unverifiable`
claims — the metric finally tracks reality.

## What V132 deliberately does *not* do

- **No new feature flag.** All three fixes ride existing surfaces
  (`HallucinationDetector`, `ChainOfVerification`,
  `AntiHallucinationPipeline`). Adding a flag for "use LLM in
  CoVe" would just mean two configurations to maintain — the
  callback is opt-in by being `Option<Box<dyn Fn>>`.
- **No NLI model bundled.** The LLM the user is already running
  is the verifier. Bundling a separate NLI head would mean a
  second model in memory for marginal accuracy gain on a path
  that is already CPU-bound.
- **No automatic retraining of `min_confidence_for_output`.**
  The grounding fix removes that knob from the
  reference-supplied path; in the no-reference path it keeps its
  existing semantics. Recalibrating the threshold belongs in a
  future cycle that has labelled data.
