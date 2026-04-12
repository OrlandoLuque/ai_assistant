# ai_assistant — C FFI (V79)

V79 introduces **22 `extern "C"` entry points** that wrap
[`AiAssistant`] so the library can be driven from any language with a
C FFI bridge: C, C++, C#, Unity, Unreal, Bevy, Python (via `ctypes`),
Lua, JavaScript (via N-API), ....

> **Primary driver**: NPCs in video games (Proposal 5). The blocking
> `ai_assistant_send_message` path is a drop-in replacement for
> hand-written HTTP clients in game engines.
>
> **Status**: Tier 1 — handle lifecycle, configuration, blocking send,
> callback streaming. RAG document indexing, memory, cancellation,
> multi-agent, and MCP tool-calling are deferred to V79.1+.

## Table of contents

1. [Quick start](#quick-start)
2. [Design principles](#design-principles)
3. [Threading contract](#threading-contract)
4. [Memory model](#memory-model)
5. [Error reporting](#error-reporting)
6. [Panic boundary](#panic-boundary)
7. [Data-bearing provider variants](#data-bearing-provider-variants)
8. [Function reference](#function-reference)
9. [Building and linking](#building-and-linking)
10. [Security mitigations](#security-mitigations)
11. [Deferred to V79.1+](#deferred-to-v791)

## Quick start

```c
#include "ai_assistant.h"
#include <stdio.h>

int main(void) {
    AiAssistantHandle *h = ai_assistant_new_with_prompt(
        "You are a friendly villager in a fantasy RPG.");
    if (h == NULL) {
        fprintf(stderr, "new: %s\n", ai_assistant_last_error());
        return 1;
    }

    ai_assistant_set_provider(h, AI_PROVIDER_KIND_OLLAMA);
    ai_assistant_set_model(h, "llama3.2:3b");

    char *reply = NULL;
    int rc = ai_assistant_send_message(h, "Greet the traveler.", &reply);
    if (rc == 0 && reply != NULL) {
        printf("Villager: %s\n", reply);
        ai_assistant_free_string(reply);
    } else {
        fprintf(stderr, "send: %s\n", ai_assistant_last_error());
    }
    ai_assistant_free(h);
    return 0;
}
```

Full working examples:

- **C**: [`examples/ffi_c/main.c`](../examples/ffi_c/main.c)
  ([README](../examples/ffi_c/README.md))
- **Python** (ctypes): [`examples/ffi_python/main.py`](../examples/ffi_python/main.py)
  ([README](../examples/ffi_python/README.md))
- **Node.js** (koffi): [`examples/ffi_node/index.js`](../examples/ffi_node/index.js)
  ([README](../examples/ffi_node/README.md))
- **Java** (JNA): [`examples/ffi_java/AiAssistantDemo.java`](../examples/ffi_java/AiAssistantDemo.java)
  ([README](../examples/ffi_java/README.md))

## Design principles

- **Zero new runtime dependencies.** The `ffi` Cargo feature is a
  zero-dep `ffi = []`. All C types come from `libc`, which is already
  a direct dep of `ai_assistant`.
- **Opaque handle.** Callers manipulate a `AiAssistantHandle*` pointer
  and never see the Rust struct layout.
- **Single-thread contract (SQLite-style).** No Mutex, no spin-lock,
  no synchronization overhead. Each handle is pinned to the thread
  that first uses it; cross-thread use is a debug-time panic.
- **Panic boundary on every entry.** Rust panics never unwind across
  the FFI boundary — they are caught, stashed as a last-error, and
  converted to the `AI_ERR_PANIC` return code.
- **Strict input validation.** Every `const char*` argument is
  null-checked and UTF-8 validated. `float` temperature is
  strict-rejected on NaN, ±Inf, and out-of-range.
- **Lazy provider construction.** The provider enum is built from the
  stored `AiProviderKind` + companion URL/region only at
  `send_message` time, so setter order is irrelevant.

## Threading contract

**Each handle is single-threaded.** The C caller must use each
`AiAssistantHandle*` from one OS thread only. This matches the
contract of SQLite (non-serialized mode), libcurl handles, and
libsodium state objects.

In **debug builds**, the first FFI call on a handle pins it to the
current thread via an `AtomicU64::compare_exchange(0, id, AcqRel,
Acquire)`. Subsequent calls from any other thread panic, which is
caught by the panic boundary and reported as `AI_ERR_PANIC` with a
last-error message containing the offending thread IDs.

In **release builds**, the thread-pin is compiled out for zero
overhead. The caller is trusted to honor the contract.

If you need to move a handle between threads, the correct pattern is:
1. Stop all use on thread A.
2. Establish a happens-before relationship (e.g. via a channel send
   + receive, or a barrier).
3. Begin use on thread B. **Never** interleave.

In practice, most FFI consumers pin each handle to a dedicated
background worker thread (NPC AI loop, chat worker, etc.) and
communicate via typed message queues.

## Memory model

### Input strings

`const char *` arguments are **borrowed** for the duration of the
call. They must be:

- Non-null (unless explicitly documented otherwise)
- NUL-terminated
- Valid UTF-8

On violation the function returns `AI_ERR_NULL_PTR` or
`AI_ERR_INVALID_UTF8` and stashes a human-readable detail in
`ai_assistant_last_error()`.

### Output strings

`char **out` parameters are **allocated by Rust** via
`CString::into_raw`. The C caller **MUST** free them via
`ai_assistant_free_string`:

```c
char *reply = NULL;
int rc = ai_assistant_send_message(h, "hi", &reply);
if (rc == 0 && reply != NULL) {
    /* use reply ... */
    ai_assistant_free_string(reply);  // <- REQUIRED
}
```

Calling `free(3)` directly on the pointer is **undefined behavior**
— Rust and the C runtime may use different allocators.

### Handles

`AiAssistantHandle *` is allocated by `ai_assistant_new*` via
`Box::into_raw`. The caller must free each handle exactly once via
`ai_assistant_free`. `ai_assistant_free` is **null-safe** — passing
`NULL` is a no-op.

### Double-free

Neither string nor handle double-free is detectable at runtime
without sanitizers. The recommended convention is:

```c
ai_assistant_free(h);
h = NULL;  // prevent accidental re-use

ai_assistant_free_string(reply);
reply = NULL;
```

## Error reporting

Every non-pointer-returning function returns an `int`. The codes are:

| Constant                    | Value | Meaning |
|-----------------------------|-------|---------|
| `AI_OK`                     | `0`   | Success |
| `AI_ERR_NULL_PTR`           | `-1`  | Required pointer was NULL |
| `AI_ERR_INVALID_UTF8`       | `-2`  | String argument was not valid UTF-8 |
| `AI_ERR_PANIC`              | `-3`  | Rust panic caught at FFI boundary |
| `AI_ERR_POISONED`           | `-4`  | Sync primitive poisoned (should not happen) |
| `AI_ERR_INTERNAL`           | `-5`  | Invalid parameter (see `last_error`) |
| `AI_ERR_UNKNOWN_PROVIDER`   | `-6`  | Provider config incomplete (e.g. OpenAICompatible missing URL) |
| `AI_ERR_SEND_FAILED`        | `-7`  | Upstream provider call failed |
| `AI_ERR_NO_RESPONSE`        | `-8`  | No response produced (reserved for V79.1) |

On non-zero return, call `ai_assistant_last_error()` to get a
thread-local human-readable detail. The returned pointer is valid
until the next FFI call on the current thread.

```c
int rc = ai_assistant_set_temperature(h, 99.0f);
if (rc != 0) {
    fprintf(stderr, "rc=%d err=%s\n", rc, ai_assistant_last_error());
    // rc=-5 err=temperature must be in [0.0, 2.0]
}
```

## Panic boundary

Every entry point wraps its body in `std::panic::catch_unwind`. A
caught panic:

1. Is downcast to `&str`/`String`/fallback
2. Stashed in the thread-local `LAST_ERROR` slot
3. Returned as `AI_ERR_PANIC` (or `NULL` for pointer-returning functions)

### ⚠ `panic = "abort"` caveat

The default `release` Cargo profile in this crate has
`panic = "abort"`, which **makes `catch_unwind` a no-op** — any
panic aborts the whole process. For FFI consumers this is almost
never what you want.

**Recommendation**: always build with `--profile release-fast`,
which keeps `panic = "unwind"`:

```bash
cargo build --features ffi --profile release-fast
```

`build.rs` detects the dangerous combo (`release` + `panic = "abort"`
+ `ffi` feature) and emits a `cargo:warning` so you don't get
surprised silently.

## Data-bearing provider variants

`AiProvider` has three variants that carry data:

```rust
OpenAICompatible { base_url: String }
Bedrock { region: String }
AzureOpenAI { endpoint: String, deployment: String }
```

A flat C enum cannot model this. The FFI exposes `AiProviderKind`
with 18 unit values (matching every Rust variant positionally) and
adds **companion setters** for the carried data:

```c
// OpenAICompatible — either order works.
ai_assistant_set_openai_compatible_url(h, "http://localhost:1234/v1");
ai_assistant_set_provider(h, AI_PROVIDER_KIND_OPEN_AI_COMPATIBLE);

// Bedrock
ai_assistant_set_bedrock_region(h, "us-east-1");
ai_assistant_set_provider(h, AI_PROVIDER_KIND_BEDROCK);

// Azure OpenAI (V80) — requires both endpoint and deployment.
ai_assistant_set_azure_endpoint(h, "https://my-resource.openai.azure.com");
ai_assistant_set_azure_deployment(h, "gpt-4o");
ai_assistant_set_provider(h, AI_PROVIDER_KIND_AZURE_OPEN_AI);
```

If you forget a companion setter, `ai_assistant_send_message`
returns `AI_ERR_UNKNOWN_PROVIDER` with a clear last-error:

```
OpenAICompatible requires prior ai_assistant_set_openai_compatible_url
AzureOpenAI requires prior set_azure_endpoint
```

The Rust side uses an **exhaustive match** on `AiProviderKind` to
build the `AiProvider` value, so if a new variant is ever added to
the Rust enum, `src/ffi.rs` will fail to compile until the FFI is
updated. This keeps the ABI and the Rust API in lockstep
automatically.

## Function reference

### Lifecycle

```c
AiAssistantHandle *ai_assistant_new(void);
AiAssistantHandle *ai_assistant_new_with_prompt(const char *prompt);
void ai_assistant_free(AiAssistantHandle *handle);
```

Both constructors are **infallible** at the Rust level (they return
`Self`, not `Result`). They only return NULL if (a) the prompt
string fails null/UTF-8 validation, or (b) Rust panics during
construction. `ai_assistant_free` is null-safe.

### Configuration setters

```c
int ai_assistant_set_system_prompt(AiAssistantHandle *h, const char *prompt);
int ai_assistant_set_provider(AiAssistantHandle *h, AiProviderKind kind);
int ai_assistant_set_openai_compatible_url(AiAssistantHandle *h, const char *url);
int ai_assistant_set_bedrock_region(AiAssistantHandle *h, const char *region);
int ai_assistant_set_azure_endpoint(AiAssistantHandle *h, const char *endpoint);
int ai_assistant_set_azure_deployment(AiAssistantHandle *h, const char *deployment);
int ai_assistant_set_model(AiAssistantHandle *h, const char *model);
int ai_assistant_set_api_key(AiAssistantHandle *h, const char *key);
int ai_assistant_set_ollama_url(AiAssistantHandle *h, const char *url);
int ai_assistant_set_temperature(AiAssistantHandle *h, float temperature);
int ai_assistant_set_max_history(AiAssistantHandle *h, size_t max_history);
```

`set_temperature` strictly rejects NaN, ±Inf, and any finite value
outside `[0.0, 2.0]` with `AI_ERR_INTERNAL`. Clamping is
intentionally not performed — it hides caller bugs.

### Messaging

```c
int ai_assistant_send_message(
    AiAssistantHandle *h,
    const char *prompt,
    char **out);

int ai_assistant_send_message_stream(
    AiAssistantHandle *h,
    const char *prompt,
    void (*callback)(const char *chunk, bool is_final, void *user_data),
    void *user_data);
```

`send_message` is blocking: it returns only after the full response
has been generated. `*out` is set to a Rust-allocated string that
the caller must free with `ai_assistant_free_string`.

`send_message_stream` dispatches `Chunk` events as they arrive via
the callback, then fires one final callback with `is_final = true`
for the terminal `Complete` / `Cancelled` / `Error` event. The
callback runs on the calling thread. The `callback` argument is
non-nullable (Rust function pointer type) — passing NULL is
undefined behavior.

**Re-entrancy warning**: calling any other `ai_assistant_*`
function on the same handle from within the callback is undefined
behavior. Copy the data out and process it after the stream call
returns.

When built with `--features "ffi,rag"`, `send_message` auto-builds
RAG context from the indexed corpus via `generate_sync_with_rag`.
Otherwise it uses the plain `generate_sync(msg, "")` — the core
blocking primitive, not RAG-aware.

### Session control

```c
int ai_assistant_clear_conversation(AiAssistantHandle *h);
int ai_assistant_new_session(AiAssistantHandle *h);
```

### Errors & memory

```c
void ai_assistant_free_string(char *s);       // null-safe
const char *ai_assistant_last_error(void);    // borrowed, thread-local
const char *ai_assistant_version(void);       // static
int ai_assistant_abi_version(void);           // currently 1
```

`ai_assistant_version()` returns the Cargo package version (e.g.
`"0.2.11"`). `ai_assistant_abi_version()` returns a monotonic ABI
counter that bumps on any breaking change to the FFI signatures or
error codes. V79 ships with ABI version **1**.

## Building and linking

The library is built as **three** outputs simultaneously:

```toml
[lib]
crate-type = ["rlib", "cdylib", "staticlib"]
```

| Platform       | cdylib                  | staticlib             | Import lib            |
|----------------|-------------------------|-----------------------|-----------------------|
| Linux          | `libai_assistant.so`    | `libai_assistant.a`   | —                     |
| macOS          | `libai_assistant.dylib` | `libai_assistant.a`   | —                     |
| Windows MSVC   | `ai_assistant.dll`      | `ai_assistant.lib`    | `ai_assistant.dll.lib`|
| Windows GNU    | `ai_assistant.dll`      | `libai_assistant.a`   | `libai_assistant.dll.a` |

All `#[no_mangle] pub extern "C"` symbols are exported from the
`cdylib` automatically — no `.def` file or `__declspec(dllexport)`
is required. Verify with:

```bash
# Linux / macOS
nm -D target/release-fast/libai_assistant.so | grep ai_assistant_

# Windows MSVC
dumpbin /exports target\release-fast\ai_assistant.dll | findstr ai_assistant_
```

The C header is auto-generated by `build.rs` on every
`--features ffi` build and committed at `include/ai_assistant.h`.
The source of truth is always `src/ffi.rs`; the `.h` is regenerated
by `cargo build --features ffi`.

## Security mitigations

V79 ships with 21 explicit mitigations:

| # | Risk | Mitigation |
|---|------|------------|
| S-1 | Null handle deref | `handle_to_inner` checks on every entry |
| S-2 | Null C string arg | `cstr_to_str` checks on every string input |
| S-3 | Invalid UTF-8 input | `CStr::to_str()` → `AI_ERR_INVALID_UTF8` |
| S-4 | Use-after-free (handle) | Documented single-owner contract |
| S-5 | Double-free (handle) | `ai_assistant_free` null-safe; convention to NULL ptr |
| S-6 | Double-free (string) | `ai_assistant_free_string` null-safe |
| S-7 | Panic crossing FFI | `catch_unwind` on every entry |
| S-8 | `panic = "abort"` trap | `build.rs` warning + docs recommend `release-fast` |
| S-9 | Sync primitive poisoning | `AI_ERR_POISONED` (reserved) |
| S-10 | Data race on last-error | `thread_local!` — per-thread, no sharing |
| S-11 | Unaligned handle ptr | `Box::into_raw` guarantees alignment |
| S-12 | ABI drift on provider additions | Exhaustive match in `build_provider` |
| S-13 | Output string buffer issues | `CString::into_raw` NUL-terminates |
| S-14 | Callback re-entrancy deadlock | Documented; not auto-detected |
| S-15 | Callback panic UB | Not caught (user's responsibility) |
| S-16 | Unchecked `f32` NaN/Inf | Strict reject on `set_temperature` |
| S-17 | Wrong-thread handle use | Debug-only `AtomicU64` pin |
| S-18 | Aliasing mutable borrow | `UnsafeCell` + single-thread contract |
| S-19 | Header drift from source | Committed `include/ai_assistant.h`, regenerated by `build.rs` |
| S-20 | `#[non_exhaustive]` match caveat | Documented: ffi.rs is a module, not sibling crate |
| S-21 | `rag` feature dispatch | `#[cfg]` branch selects `generate_sync` vs `generate_sync_with_rag` |

## Deferred to V79.1+

Tier 2 and beyond:

- RAG document indexing from C (`ai_assistant_rag_index_*`)
- Memory system enable/disable from C
- Fallback provider chain configuration
- Cancellation tokens (`ai_assistant_cancel`)
- MCP tool-calling with C callback registration
- PyO3 Python bindings (native, beyond the ctypes example)
- Unity / Unreal / Bevy sample projects
- Async FFI (tokio-style callbacks)
- C# P/Invoke sample project
- Go (cgo) example
- CI header-drift check (diff generated vs committed)
- Valgrind / ASAN integration in test suite
- C++ RAII wrapper headers
