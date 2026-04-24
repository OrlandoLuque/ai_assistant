# V103.1 — vLLM deep tuning: prefix caching, LoRA, metrics, structured output

**Version:** 0.2.36 → 0.2.37
**Date:** 2026-04-24
**Scope:** Round out V103's vLLM integration with the tuning flags and
observability that production workloads actually need. No new provider,
no new binaries — everything lands as launch-config fields, runtime
helpers, and butler hints.

V103 made vLLM a first-class provider and taught the butler to
recommend it. V103.1 asks the follow-up: *once the user picks vLLM,
what do we do to make it fast, keep it healthy, and hot-swap
adapters?* Nine improvements, wired through the same `VLlmLaunchConfig`
+ butler surface users already know.

---

## 1. Prefix caching for agentic loops

Agentic / multi-agent / autonomous workloads reuse the same system
prompt across every turn. vLLM can hash-match that prefix and reuse
its KV cache instead of recomputing attention over it, giving a
5-30% latency win at zero quality cost.

- `VLlmLaunchConfig.enable_prefix_caching: bool` → emits
  `--enable-prefix-caching` in `vllm_launch_command` /
  `vllm_docker_command`.
- `Butler::recommend_runtime` now appends the flag to the reason
  text and the install hint whenever the workload is agentic. Eval
  batches (distinct prompts) are deliberately excluded.

## 2. Health polling after launch

vLLM can take 30-120s to load weights. `vllm_wait_until_ready(base_url,
timeout, interval)` polls `GET /health` until the server answers, then
runs the full `probe_vllm` to return a `VLlmCapability`. Meant for
post-launch waits (`Command::spawn` then poll) — error on deadline
miss.

## 3. VRAM-aware quantization picker

`RuntimeInfo.gpu_vram_mb: Option<u64>` is now parsed from
`nvidia-smi --query-gpu=memory.total`. New helper
`pick_quantization_for_vram(params_b, vram_mb)`:

- fp16 needs ≈ 2 GiB/B × 1.2 overhead → if VRAM fits, returns `None`
  (use full precision).
- AWQ 4-bit needs ≈ 0.55 GiB/B × 1.3 overhead → otherwise returns
  `Some("awq")`.

## 4. LoRA registry + hot-swap (`src/vllm_lora.rs`)

vLLM launched with `--enable-lora` exposes
`POST /v1/load_lora_adapter {lora_name, lora_path}` and
`POST /v1/unload_lora_adapter {lora_name}`. The module is a thin
JSON-over-HTTP client: `load_lora_adapter` / `unload_lora_adapter`
plus the two request structs.

After load, the adapter appears in `/v1/models` with `lora_name` as
the `id` — use that value as the `model` field in subsequent
chat/completions requests.

## 5. Structured output (`src/vllm_guided.rs`)

`VLlmGuidedOptions { guided_json, guided_regex, guided_choice }` +
`apply_guided(&mut Value, &opts)` helper that injects the guided
fields into an OpenAI-style request body. Only one mode should be
active per request; the helper writes whichever fields the caller
set.

## 6. Prometheus `/metrics` scrape (`src/vllm_metrics.rs`)

vLLM exposes a full Prometheus text-format `/metrics` endpoint.
`scrape_vllm_metrics(base_url)` returns a `VLlmMetrics` with the six
values that actually matter:

| Field | Source metric |
|---|---|
| `running_requests` | `vllm:num_requests_running` |
| `waiting_requests` | `vllm:num_requests_waiting` |
| `gpu_cache_usage`  | `vllm:gpu_cache_usage_perc` (max across models) |
| `prompt_tokens_total` | `vllm:prompt_tokens_total` |
| `generation_tokens_total` | `vllm:generation_tokens_total` |

`VLlmMetrics::saturated()` returns `true` when `waiting ≥ 4` or
`cache_usage ≥ 0.9` — a coarse "scale up TP or upsize GPU" signal.
Zero-dependency text parser — no prometheus client crate.

## 7. Speculative decoding

`VLlmLaunchConfig.speculative_model: Option<String>` +
`num_speculative_tokens: Option<u32>` → emits `--speculative-model` and
`--num-speculative-tokens` flags. Typical pairing: Llama-3.1-70B base
with Llama-3.2-1B as the draft model for 2-3x latency reduction on
easy tokens.

## 8. KV-cache fp8

`VLlmLaunchConfig.kv_cache_dtype: Option<String>` accepts `auto`,
`fp8`, `fp8_e5m2`, `fp8_e4m3`. Halves KV-cache memory with negligible
quality loss on most models, letting the same GPU hold ≈2x the
concurrent sequences.

## 9. Chat template override

`VLlmLaunchConfig.chat_template: Option<String>` emits
`--chat-template "..."`. Escape hatch for models shipped with a
broken or missing `tokenizer_config.json` chat template.

---

## Surface summary

New public items:

```text
ai_assistant::{
    // capability
    vllm_wait_until_ready,

    // LoRA
    load_lora_adapter, unload_lora_adapter,
    LoadLoraRequest, UnloadLoraRequest,

    // metrics
    VLlmMetrics, parse_vllm_metrics, scrape_vllm_metrics,

    // structured output
    VLlmGuidedOptions, apply_guided,

    // butler
    pick_quantization_for_vram,
}
```

Extended items:

- `RuntimeInfo` — `+ gpu_vram_mb: Option<u64>`
- `VLlmLaunchConfig` — `+ enable_prefix_caching`, `+ kv_cache_dtype`,
  `+ speculative_model`, `+ num_speculative_tokens`, `+ chat_template`

## Tests

+13 tests total. No existing tests changed behavior.

| Module | New |
|---|---|
| `butler` | +5 (quantization picker, prefix-caching suggestion) |
| `vllm_launch` | +5 (one per new flag) |
| `vllm_capability` | +1 (wait_until_ready) |
| `vllm_lora` | +4 |
| `vllm_metrics` | +9 |
| `vllm_guided` | +7 |

---

## What V103.1 deliberately does NOT do

- **No LoRA registry persisted to disk.** That's a higher-level
  concern — this module just wraps the two endpoints.
- **No auto-launch of vLLM processes.** `ai_setup install vllm`
  produces the launch command; the user runs it.
- **No structured-output validation.** `apply_guided` injects the
  fields; the server enforces them.
- **No metrics aggregation / dashboards.** Scrape is point-in-time;
  callers who want time-series should wire it into their own
  observability layer.
