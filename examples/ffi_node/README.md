# ai_assistant — Node.js FFI Example (koffi)

Minimal NPC-style driver that loads `libai_assistant` via
[koffi](https://koffi.dev/) — a pure-JS FFI bridge with no native
compilation step (no `node-gyp`, no `prebuild`).

## Prerequisites

1. **Node.js 16+** (LTS recommended).
2. A built `libai_assistant` shared library:
   ```bash
   cargo build --features ffi --profile release-fast
   ```
3. A running **Ollama** with a pulled model (default: `llama3.2:3b`) at
   `http://localhost:11434`.

## Install & run

```bash
cd examples/ffi_node
npm install
```

### Linux

```bash
LD_LIBRARY_PATH=../../target/release-fast node index.js
```

### macOS

```bash
DYLD_LIBRARY_PATH=../../target/release-fast node index.js
```

### Windows

```cmd
set PATH=..\..\target\release-fast;%PATH%
node index.js
```

## How it works

1. `koffi.load()` opens the shared library.
2. `lib.func()` declares each FFI function with C-style signatures.
3. Standard lifecycle: `new_with_prompt` → configure → `send_message` →
   print → `free_string` → `free`.
4. Streaming uses `koffi.register()` to wrap a JS function as a C
   callback pointer.

## Why koffi?

| Library | Pros | Cons |
|---------|------|------|
| **koffi** | Pure JS, no build step, fast, actively maintained | Newer project |
| ffi-napi | Mature, widely used | Requires `node-gyp` + C compiler |
| node-ffi | Original; well-known | Unmaintained, Node 12 era |

`koffi` was chosen because it works out of the box on all platforms
without a C toolchain on the JS side.

## Memory safety

- Output strings from `ai_assistant_send_message` are freed via
  `ai_assistant_free_string` — never via JS garbage collection.
- The handle is freed in a `try/finally` block.
- The streaming callback must **not** call back into `ai_assistant_*` on
  the same handle (re-entrancy is UB).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Error: Cannot load library` | Set `LD_LIBRARY_PATH` / `PATH` or copy the lib to a system dir. |
| `send_message` returns non-zero | Check that Ollama is running and the model is pulled (`ollama list`). |
| Streaming callback never fires | Ensure `koffi >= 2.9`. Earlier versions had callback registration bugs. |
