# ai_assistant — Java FFI Example (JNA)

Minimal NPC-style driver that loads `libai_assistant` via
[JNA](https://github.com/java-native-access/jna) — Java Native Access,
the standard zero-JNI library for calling native code from Java.

## Prerequisites

1. **JDK 11+** (LTS recommended).
2. A built `libai_assistant` shared library:
   ```bash
   cargo build --features ffi --profile release-fast
   ```
3. A running **Ollama** with a pulled model (default: `llama3.2:3b`) at
   `http://localhost:11434`.
4. **JNA jar** (download once):
   ```bash
   curl -LO https://repo1.maven.org/maven2/net/java/dev/jna/jna/5.14.0/jna-5.14.0.jar
   ```

Or, if using Maven/Gradle, add the dependency instead:

```xml
<dependency>
    <groupId>net.java.dev.jna</groupId>
    <artifactId>jna</artifactId>
    <version>5.14.0</version>
</dependency>
```

## Compile & run

### Linux

```bash
javac -cp jna-5.14.0.jar examples/ffi_java/AiAssistantDemo.java

LD_LIBRARY_PATH=target/release-fast \
    java -cp "jna-5.14.0.jar:examples/ffi_java" AiAssistantDemo
```

### macOS

```bash
javac -cp jna-5.14.0.jar examples/ffi_java/AiAssistantDemo.java

DYLD_LIBRARY_PATH=target/release-fast \
    java -cp "jna-5.14.0.jar:examples/ffi_java" AiAssistantDemo
```

### Windows

```cmd
javac -cp jna-5.14.0.jar examples\ffi_java\AiAssistantDemo.java

set PATH=target\release-fast;%PATH%
java -cp "jna-5.14.0.jar;examples\ffi_java" AiAssistantDemo
```

## How it works

1. The `AiAssistantLib` interface extends `com.sun.jna.Library` and
   declares every FFI function with Java types. JNA handles marshaling
   automatically:
   - `String` → `const char *` (NUL-terminated UTF-8)
   - `Pointer` → opaque `void *` handle
   - `PointerByReference` → `char **out` (output string parameter)
   - `StreamCallback extends Callback` → C function pointer
2. `Native.load("ai_assistant", ...)` opens the shared library using the
   platform search path.
3. Standard lifecycle: `new_with_prompt` → configure → `send_message` →
   print → `free_string` → `free`.
4. Streaming uses a lambda implementing `StreamCallback`.

## Memory safety

- Output strings from `ai_assistant_send_message` are freed via
  `ai_assistant_free_string(Pointer)` — **not** by the Java GC.
- The handle is freed in a `try/finally` block.
- The streaming callback must **not** call back into `ai_assistant_*` on
  the same handle (re-entrancy is UB).
- Keep a strong reference to the `StreamCallback` lambda during the
  streaming call — if the GC collects it while the native side is still
  invoking it, the JVM will crash.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `UnsatisfiedLinkError: Unable to load library 'ai_assistant'` | Set `LD_LIBRARY_PATH` / `PATH`, or pass `-Djna.library.path=target/release-fast`. |
| `send_message` returns non-zero | Check that Ollama is running and the model is pulled (`ollama list`). |
| Streaming callback crashes | Ensure the callback lambda is kept in a local variable — anonymous lambdas may be GC'd. |

## Alternative: JNI

For production use, a JNI wrapper (hand-written or via
[jni-rs](https://github.com/jni-rs/jni-rs)) avoids JNA's reflection
overhead. JNA is chosen here for simplicity — no C/Rust glue code
required.
