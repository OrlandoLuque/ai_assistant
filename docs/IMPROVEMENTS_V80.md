# V80 — Azure OpenAI as first-class provider

> **Version**: 0.2.11 → 0.2.12
> **Status**: complete — 12 new tests green
> **Scope**: new provider variant, 3 dispatch paths, FFI bindings, config file support

## Context

V79 added C FFI bindings. The next gap: Azure OpenAI was not supported
despite being one of the most widely adopted enterprise LLM endpoints.

An `OpenAIConfig::azure()` existed in `src/openai_adapter.rs` but was
dead code — never wired into dispatch, and it used the wrong auth header
(`Bearer` instead of `api-key`).

Azure OpenAI differs from standard OpenAI in two critical ways:

1. **Auth header**: `api-key: {key}` (NOT `Authorization: Bearer {key}`)
2. **URL pattern**: `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21`

Request/response JSON is identical to OpenAI — only the transport differs.

## Workstreams

### WS-1. `AiProvider::AzureOpenAI` variant (`src/config.rs`)

New data-bearing variant:

```rust
AzureOpenAI { endpoint: String, deployment: String },
```

Added arms to all 6 exhaustive matches:
- `display_name` → `"Azure OpenAI"`
- `icon` → `"☁️"`
- `is_openai_compatible` → `true`
- `is_cloud` → `true`
- `get_provider_url` → `endpoint.clone()`
- `get_api_key` → `AZURE_OPENAI_API_KEY` env var fallback

### WS-2. Dedicated cloud functions (`src/cloud_providers.rs`)

The existing `generate_openai_response()` in `providers.rs` is a no-auth
path. Azure needs `api-key: {key}`, so dedicated functions were added:

- `azure_openai_url()` — URL construction helper
- `generate_azure_openai_cloud()` — blocking, `api-key` header
- `generate_azure_openai_streaming()` — SSE streaming, same auth
- `fetch_azure_openai_models()` — static model list (8 models)

Also added `AzureOpenAI` arms to `resolve_api_key`, `generate_cloud_response`,
and `fetch_cloud_models`.

### WS-3. Dispatch routing (`src/providers.rs`)

Three dispatch functions updated:
- `generate_response()` → `generate_azure_openai_cloud()`
- `generate_response_streaming()` → `generate_azure_openai_streaming()`
- `generate_response_streaming_cancellable()` → streaming with pre-check

Also added `AzureOpenAI` arm to `fetch_model_context_size()`.

### WS-4. Config file support (`src/config_file.rs`)

- `to_ai_config()`: `"azure" | "azure_openai"` reads endpoint from
  `custom_url` or `AZURE_OPENAI_ENDPOINT` env, deployment from model
  field or `AZURE_OPENAI_DEPLOYMENT` env.
- `from_ai_config()`: stores as `("azure_openai", Some(endpoint))`.

### WS-5. FFI bindings (`src/ffi.rs`)

- `AiProviderKind::AzureOpenAI` added (18th variant)
- Two new `RefCell<Option<String>>` fields in `Inner`: `azure_endpoint`,
  `azure_deployment`
- `build_provider()` arm requires both setters
- Two new `extern "C"` setters:
  - `ai_assistant_set_azure_endpoint(h, endpoint)`
  - `ai_assistant_set_azure_deployment(h, deployment)`

Entry points: 20 → 22.

### WS-6. Tests (12 new)

| Module | Test | What it covers |
|--------|------|----------------|
| `config.rs` | `test_azure_openai_display_name` | Display name |
| `config.rs` | `test_azure_openai_is_cloud` | Cloud classification |
| `config.rs` | `test_azure_openai_is_openai_compatible` | OpenAI compat flag |
| `config.rs` | `test_azure_openai_get_api_key_env_fallback` | Env var resolution |
| `cloud_providers.rs` | `test_azure_openai_url_construction` | URL builder |
| `cloud_providers.rs` | `test_azure_openai_unreachable_returns_err` | Error on bad endpoint |
| `cloud_providers.rs` | `test_fetch_azure_openai_models` | Static model list |
| `ffi.rs` | `test_set_azure_endpoint_happy_path` | Endpoint setter |
| `ffi.rs` | `test_set_azure_deployment_happy_path` | Deployment setter |
| `ffi.rs` | `test_azure_provider_requires_both_setters` | Missing setter error |
| `tests/ffi_integration.rs` | `azure_provider_kind_exists` | Cross-crate FFI |

## Files changed

| File | Delta | What |
|------|-------|------|
| `src/config.rs` | +25 | Variant + 6 match arms + 4 tests |
| `src/cloud_providers.rs` | +130 | 4 functions + dispatch arms + 3 tests |
| `src/providers.rs` | +16 | 4 dispatch arms |
| `src/config_file.rs` | +12 | to/from config parsing |
| `src/ffi.rs` | +60 | Enum + Inner + setters + 3 tests |
| `tests/ffi_integration.rs` | +12 | Cross-crate test |
| `CHANGELOG.md` | +14 | V80 entry |
| `docs/FFI.md` | +12 | Azure setters + data-bearing update |
| `docs/USE_CASES.md` | +6 | Azure note in case #4 |
| `docs/IMPROVEMENTS_V80.md` | new | This file |
| `Cargo.toml` | +1/-1 | Version bump |

## Security review

| Risk | Mitigation |
|------|------------|
| API key leakage in logs | `AiConfig::Debug` already redacts `api_key`; Azure key follows same path |
| URL injection via endpoint/deployment | `ureq::post(&url)` handles URL escaping; deployment names are alphanumeric |
| PII in cloud requests | Azure is `is_cloud() = true` → existing PII masking applies |
| FFI null/UTF-8 for new setters | Same `guard` + `cstr_to_str` pattern as all existing setters |
| Hardcoded api-version | `2024-10-21` is latest GA; future iteration can make configurable |
