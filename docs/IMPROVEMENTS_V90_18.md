# V90.18 — Vision dispatcher + local provider transports + CLI `--image`

**Version**: 0.2.37 → 0.2.38
**Feature flag**: `vision` (no new flag — extends existing surface)
**Date**: 2026-04-26

## Scope

Closes the end-to-end vision path. Earlier rounds delivered the data
shapes and routing knowledge:

- **V90.16** — added `gemma3` (128K, vision), `phi-3.5-vision` (128K, vision)
  and supporting profiles to `ModelRouter`; `Butler::model_runtime_hint`
  helper.
- **V90.17** — `cloud_providers::generate_*_with_images` for OpenAI,
  Anthropic and Gemini, plus `VisionMessage::to_anthropic_format` /
  `to_gemini_format` and longest-pattern-match in `routing::get_profile`.

V90.18 wires those pieces into a single dispatcher and a CLI verb so
operators (and agentic callers) can actually issue a one-shot vision
request without hand-routing to provider-specific functions.

## Public surface

### `vision::generate_vision_response`

```rust
pub fn generate_vision_response(
    config: &AiConfig,
    messages: &[VisionMessage],
    system_prompt: &str,
) -> anyhow::Result<String>
```

Routes the request based on `config.provider`:

| Provider                     | Backend                                        |
|------------------------------|------------------------------------------------|
| OpenAI / Anthropic / Gemini  | `cloud_providers::generate_cloud_response_with_images` |
| Groq / Together / Fireworks  | same (uses OpenAI-compatible adapter)          |
| DeepSeek / Mistral           | same                                           |
| Perplexity / OpenRouter      | same                                           |
| Ollama                       | `providers::generate_ollama_response_with_images` |
| LM Studio                    | `providers::generate_openai_compat_response_with_images` |
| LocalAI / llama.cpp / vLLM   | same                                           |
| text-gen-webui               | same                                           |
| `OpenAICompatible {base_url}`| same (uses the supplied URL)                   |
| Azure OpenAI                 | `bail!` — needs deployment-specific routing    |
| Bedrock                      | `bail!` — feature `aws-bedrock` planned        |
| _other_                      | `bail!("Vision is not supported for ...")`     |

Whether the loaded *model* understands images is independent — the
router profiles in V90.16/V90.17 mark which model patterns claim vision
support.

### Local-provider transports

```rust
pub fn generate_ollama_response_with_images(
    config: &AiConfig,
    messages: &[VisionMessage],
    system_prompt: &str,
) -> Result<String>;

pub fn generate_openai_compat_response_with_images(
    config: &AiConfig,
    messages: &[VisionMessage],
    system_prompt: &str,
) -> Result<String>;
```

Both use 180 s timeout, `temperature` from config and `retry_with_config`
on transport failure. The OpenAI-compatible function resolves the base
URL from the relevant `AiConfig` field (`lm_studio_url`,
`text_gen_webui_url`, `local_ai_url`, `llamacpp_url`, `vllm_url`, or
`OpenAICompatible { base_url }`).

### CLI

```text
ai_cli query --image <path|URL> [--image <...>]* "<prompt>"
ai_cli verify --image <path|URL> [--image <...>]* [--faithfulness] [--cove]
              [--quality-gates] [--knowledge ...] "<prompt>"
```

`query` short-circuits the streaming assistant when at least one
`--image` is supplied: it loads the images, calls
`generate_vision_response`, and prints the result (or JSON with
`--json`).

`verify` does the same but then feeds the response into the existing
anti-hallucination pipeline (faithfulness / CoVe / quality gates), so a
visual answer can be quality-gated identically to a text one.

`load_images` (private CLI helper) accepts paths or `http(s)://` URLs;
local files are validated to exist and be ≤ 20 MB; URLs are passed
through.

Without the `vision` feature, `--image` errors out with a rebuild hint.

## Why

A library user (or agent) with a `VisionMessage` in hand previously had
to know to call `generate_openai_cloud_with_images` for OpenAI,
`generate_anthropic_cloud_with_images` for Anthropic, and that for
Ollama / LM Studio there was no helper at all — they had to build the
HTTP request by hand.

`generate_vision_response` makes "send these images to whatever provider
is configured" a one-liner, which is the natural shape for the agentic
loop and for CLI scripting. The `verify --image` path closes the second
gap: visual answers were the only LLM output that couldn't be routed
through faithfulness / CoVe / quality gates.

## Tests

No new unit tests; the dispatcher is a thin match and the per-provider
transports are integration-tested via real network calls (existing
`#[ignore]` tests). `cargo build --features vision` clean.

## What's still pending for vision

- **Azure OpenAI** vision needs deployment-aware routing (currently `bail!`).
- **Bedrock** vision needs a dedicated `aws-bedrock` feature.
- **Multi-turn** vision (image references across messages) works at the
  type level (`VisionMessage` is a vec) but no CLI surface yet.
- **Agentic loop** (`AgenticLoop`, multi-agent) does not yet *consume*
  images during execution — agents can issue one-shot vision queries via
  the dispatcher, but the loop itself sees only text.
