# ai_assistant — Python FFI Example (ctypes)

Minimal NPC-style driver that loads `libai_assistant` via Python's
built-in `ctypes` module. No Rust tooling is needed at runtime — just
the shared library.

## Prerequisites

1. **Python 3.8+** (ships with `ctypes`).
2. A built `libai_assistant` shared library:
   ```bash
   cargo build --features ffi --profile release-fast
   ```
3. A running **Ollama** with a pulled model (default: `llama3.2:3b`) at
   `http://localhost:11434`.

## Run

### Linux

```bash
LD_LIBRARY_PATH=target/release-fast \
    python3 examples/ffi_python/main.py
```

### macOS

```bash
DYLD_LIBRARY_PATH=target/release-fast \
    python3 examples/ffi_python/main.py
```

### Windows

```cmd
set PATH=target\release-fast;%PATH%
python examples\ffi_python\main.py
```

## How it works

1. `ctypes.CDLL` loads the shared library.
2. `setup_prototypes()` declares `restype` and `argtypes` for all 20 FFI
   functions — this ensures correct argument marshaling and return-value
   handling.
3. `ai_assistant_new_with_prompt()` → configure → `send_message()` → print
   → `free_string()` → `free()` — the standard lifecycle.
4. A second call demonstrates **streaming** via `CFUNCTYPE` callback.

## Memory safety

- Output strings from `ai_assistant_send_message` are freed via
  `ai_assistant_free_string` — never via Python's garbage collector.
- The handle is freed in a `try/finally` block to avoid leaks.
- The streaming callback must **not** call back into `ai_assistant_*` on
  the same handle (re-entrancy is UB).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OSError: libai_assistant.so: cannot open shared object file` | Set `LD_LIBRARY_PATH` or copy the `.so` to `/usr/local/lib`. |
| `OSError: [WinError 126]` | Ensure `ai_assistant.dll` is on `%PATH%`. |
| `send_message` returns non-zero | Check that Ollama is running and the model is pulled. |
| Garbage in reply | Ensure you call `ai_assistant_free_string`, not Python `del`. |
