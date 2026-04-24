# V102 — llama.cpp capability probe + GGUF auto-downloader + curated picker widget

**Version:** 0.2.33 → 0.2.34
**Date:** 2026-04-24
**Scope:** Three V101 follow-ups delivered together.

V101 added `AiProvider::LlamaCpp` and a curated-model catalog. V102
closes the three optional gaps listed there:

1. **F1** — `LlamaCppCapability` probe (detects PrismML fork vs upstream)
2. **F2** — generic GGUF auto-downloader (HuggingFace → local cache)
3. **F3** — egui widget that renders the curated catalog

---

## F1 — `LlamaCppCapability` probe

New module `src/llamacpp_capability.rs` (always compiled, no feature
flag). Hits the `llama-server` `/props` endpoint and returns a
structured capability record:

```rust
pub struct LlamaCppCapability {
    pub build_info: Option<String>,
    pub is_prismml_fork: bool,      // heuristic: build name or Q1_0/ternary kernel
    pub supports_q1_0: bool,        // 1.125-bpw quantization
    pub supports_ternary: bool,     // {-1, 0, 1} quantization
    pub default_ctx: Option<u32>,
}

pub fn probe_llamacpp(base_url: &str) -> Result<LlamaCppCapability, String>;
pub fn parse_props(body: &serde_json::Value) -> LlamaCppCapability;
pub fn can_run_quantization(&self, quant: &str) -> bool;  // method
```

The split between `probe_llamacpp` (network) and `parse_props` (pure)
keeps the tests offline. Retries via `RetryConfig::fast` so a single
transient TCP hiccup doesn't fail the probe.

**Use case:** before loading a Bonsai model, call
`probe_llamacpp(url).supports_q1_0` — if false, tell the user they
need the PrismML fork (`github.com/PrismML-Eng/llama.cpp`).

**Tests:** 7. Covers upstream-minimal props, PrismML build-string
detection, ternary-quant-only inference, nested `n_ctx`, unknown-tag
handling, full PrismML fork capabilities, and the `system_info`
fallback.

---

## F2 — GGUF auto-downloader

New module `src/gguf_downloader.rs` (feature flag `auto-download`,
included in `full`). Generic — usable by **any local provider that
loads GGUF**, not just llama.cpp.

### Core API

```rust
pub struct DownloadRequest {
    pub url: String,
    pub dest: PathBuf,
    pub sha256: Option<String>,
    pub bearer_token: Option<String>,  // HF token for gated repos
    pub resume: bool,                  // Range-based resume
    pub timeout: Duration,
}

pub type ProgressFn = dyn FnMut(u64, Option<u64>) + Send;

pub fn download(
    req: &DownloadRequest,
    progress: Option<Box<ProgressFn>>,
) -> Result<DownloadedFile, String>;

pub fn huggingface_resolve_url(repo: &str, file: &str, rev: Option<&str>) -> String;
pub fn default_cache_dir() -> PathBuf;   // platform-aware
```

**Transfer behavior:**
- Writes to `<dest>.part` and atomically `rename()`s to `<dest>` on
  success, so a mid-download interruption never leaves a truncated
  model file that looks complete.
- If `<dest>` already exists and `sha256` matches (or is unset), the
  download is **skipped entirely** — idempotent.
- Resume via `Range: bytes=<existing>-` header when `.part` already
  exists and `resume: true`.
- SHA256 verification on the completed `.part` before rename. Mismatch
  = error, `.part` preserved for inspection.

### Local-provider compatibility

| Provider | How to use the downloaded GGUF |
|----------|--------------------------------|
| `llama.cpp` / `llama-server` | `--model <dest>` |
| LM Studio | Drop file into its models directory |
| Kobold.cpp | `--model <dest>` |
| LocalAI | Reference in its YAML config |
| text-gen-webui | Select from the llama.cpp loader |
| Ollama | See Ollama helpers below |

### Ollama helpers

Ollama uses a content-addressed blob store at `~/.ollama/models/blobs/`,
so a naive `ollama create` would **duplicate** the 8+ GB GGUF on disk.
V102 exposes three registration paths:

```rust
// 1. Write a minimal Modelfile for manual `ollama create`.
pub fn write_ollama_modelfile(modelfile_path: &Path, gguf_path: &Path) -> Result<(), String>;

// 2. Automated: POST to Ollama's /api/create. Duplicates bytes.
pub fn register_with_ollama(
    ollama_url: &str,
    model_name: &str,
    gguf_path: &Path,
) -> Result<(), String>;

// 3. Zero-copy via hard-link. Pre-seeds the blob so Ollama reuses it.
pub fn register_with_ollama_hardlink(
    ollama_url: &str,
    model_name: &str,
    gguf_path: &Path,
    ollama_models_dir: Option<&Path>,
) -> Result<(), String>;

pub fn default_ollama_models_dir() -> PathBuf;
```

**How the hard-link path works:**

1. Hash the GGUF (SHA256).
2. Create `{ollama_models}/blobs/sha256-<hex>` as a hard link to the
   existing file (`std::fs::hard_link`) — same inode, zero data copy.
3. Call `/api/create`. Ollama hashes the source, finds the matching
   blob already present in its store, and skips the copy. Only the
   (tiny) manifest + config blob are newly written.

**Constraint:** hard links require the cache and Ollama's blob
directory to live on the same volume. If not, `hard_link` returns
`ErrorKind::CrossesDevices` and the function returns an actionable
error suggesting `register_with_ollama` (copy path) as fallback.

### Dependencies

Feature `auto-download = ["dep:sha2"]`. `ureq` is already a top-level
dep. No new runtime deps outside `sha2` (which is already used by
`security` and `distributed-network`).

**Tests:** 11. Covers URL construction (HF default/custom revision,
slash stripping), `.part` suffix, SHA256 of known vector (`"abc"`
→ `ba7816bf…`), hex encoding, default cache dir, default Ollama dir,
Modelfile emission, download-request builder, and the hard-link
primitive (same-inode semantics verified).

---

## F3 — Curated-model picker widget

Added to `src/widgets.rs` (feature `egui-widgets`).

```rust
pub struct CuratedModelPickerResponse {
    pub selected: bool,
    pub model_id: Option<String>,
}

pub fn curated_model_picker(
    ui: &mut Ui,
    provider: &AiProvider,
) -> CuratedModelPickerResponse;
```

Each entry renders as a bordered frame with:

- Display name (heading)
- Pills: parameters, quantization, approx_size
- Description (one line)
- **`Requires:` banner** in amber when `CuratedModel::requirements`
  is set — for example "Needs PrismML-Eng/llama.cpp fork (adds Q1_0
  kernel)" on every Bonsai entry.
- Clickable hyperlink to `source_url`
- Monospaced model ID + **Use this model** button

The widget returns `{ selected: true, model_id: Some("...") }` the
frame the user clicks — caller wires it into whatever model-selection
state it owns.

**Tests:** 3. Response default, non-empty picker for LlamaCpp, and a
structural check that Bonsai Q1_0 entries carry a PrismML-fork
`requirements` string.

---

## Public API additions

```rust
// Always compiled
pub use llamacpp_capability::{parse_props, probe_llamacpp, LlamaCppCapability};

// Feature-gated: auto-download
pub use gguf_downloader::{
    default_cache_dir, default_ollama_models_dir, download, huggingface_resolve_url,
    register_with_ollama, register_with_ollama_hardlink, write_ollama_modelfile,
    DownloadRequest, DownloadedFile, ProgressFn,
};
```

## Tests

| Module | Tests | Coverage |
|--------|------:|----------|
| `llamacpp_capability` | 7 | parse, heuristics, ctx, quant gating |
| `gguf_downloader` | 11 | URL, hash, part path, Ollama helpers, hard-link |
| `widgets::v102_picker_tests` | 3 | picker response, catalog presence, PrismML flag |

**Net new tests: 21.** All passing.

## Not in V102 (possible follow-ups)

- A `cargo run --example v102_bonsai_quickstart` that chains
  `download → probe → register_with_ollama_hardlink` end-to-end.
- Streaming manifest parser for the `/api/create` response so the
  progress callback also reflects Ollama's side of the operation.
- A higher-level `ensure_model_ready(&CuratedModel)` convenience
  function that combines the downloader + probe + provider-specific
  registration path.
