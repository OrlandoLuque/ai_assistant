# ai_assistant — FFI C example

A minimal C program that drives an `AiAssistant` handle through the
V79 FFI. Models the **NPC-in-a-game** scenario from `docs/USE_CASES.md`
case #9.

## Prerequisites

- Rust toolchain (stable, 1.70+)
- A C compiler (`gcc`, `clang`, or MSVC `cl`)
- A running [Ollama](https://ollama.com) instance with a model pulled
  (the example uses `llama3.2:3b`)

## Build the library

```bash
# Recommended profile for FFI: keeps panic=unwind so catch_unwind works.
cargo build --features ffi --profile release-fast
```

This produces both a dynamic and a static library, plus regenerates
`include/ai_assistant.h` via `cbindgen`:

| Platform       | cdylib                  | staticlib             | Import lib            |
|----------------|-------------------------|-----------------------|-----------------------|
| Linux          | `libai_assistant.so`    | `libai_assistant.a`   | —                     |
| macOS          | `libai_assistant.dylib` | `libai_assistant.a`   | —                     |
| Windows MSVC   | `ai_assistant.dll`      | `ai_assistant.lib`    | `ai_assistant.dll.lib`|
| Windows GNU    | `ai_assistant.dll`      | `libai_assistant.a`   | `libai_assistant.dll.a` |

Artifacts are placed in `target/release-fast/`.

## Build the C example

### Linux

```bash
gcc -Wall -Wextra -std=c11 -I include \
    examples/ffi_c/main.c \
    -L target/release-fast -lai_assistant \
    -o /tmp/ffi_demo
LD_LIBRARY_PATH=target/release-fast /tmp/ffi_demo
```

### macOS

```bash
clang -Wall -Wextra -std=c11 -I include \
    examples/ffi_c/main.c \
    -L target/release-fast -lai_assistant \
    -o /tmp/ffi_demo
DYLD_LIBRARY_PATH=target/release-fast /tmp/ffi_demo
```

### Windows MSVC

```cmd
cl /W4 /I include examples\ffi_c\main.c ^
   /link /LIBPATH:target\release-fast ai_assistant.dll.lib
set PATH=target\release-fast;%PATH%
main.exe
```

### Windows GNU (MinGW / MSYS2)

```bash
gcc -Wall -Wextra -std=c11 -I include \
    examples/ffi_c/main.c \
    -L target/release-fast -lai_assistant \
    -o ffi_demo.exe
PATH="target/release-fast:$PATH" ./ffi_demo.exe
```

## Expected output

```
ai_assistant FFI demo — version 0.2.11, ABI 1
Villager: Ah, welcome, weary traveler! Come, rest your feet by the well. ...
```

## Troubleshooting

### `error while loading shared libraries: libai_assistant.so`
Set `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to the
`target/release-fast` directory, or install the library to a system
path.

### `ai_assistant.h: No such file or directory`
Run `cargo build --features ffi` first — `build.rs` invokes `cbindgen`
to regenerate the header on every FFI-enabled build. The header lives
at `include/ai_assistant.h` relative to the crate root.

### `ai_assistant_send_message: send failed: ...`
Ollama is not running or the model is not pulled. Start Ollama and
`ollama pull llama3.2:3b`.

### Panics abort the process instead of returning `AI_ERR_PANIC`
The default `release` Cargo profile uses `panic = "abort"`, which
turns `catch_unwind` into a no-op. Use `--profile release-fast` as
shown above.

### Verifying exported symbols

Linux:
```bash
nm -D target/release-fast/libai_assistant.so | grep ai_assistant_ | head
```

Windows MSVC:
```cmd
dumpbin /exports target\release-fast\ai_assistant.dll | findstr ai_assistant_
```

## See also

- `docs/FFI.md` — full API reference
- `include/ai_assistant.h` — auto-generated C header
- `src/ffi.rs` — Rust implementation
