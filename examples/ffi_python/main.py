#!/usr/bin/env python3
"""
ai_assistant — V79 FFI Python example (ctypes).

Minimal "NPC in a game" driver: creates an assistant with a villager
system prompt, sets the Ollama provider + model, sends one blocking
message, prints the reply, and cleans up.

Usage:
    # 1. Build the shared library
    cargo build --features ffi --profile release-fast

    # 2. Run (Linux)
    LD_LIBRARY_PATH=target/release-fast python3 examples/ffi_python/main.py

    # 2. Run (macOS)
    DYLD_LIBRARY_PATH=target/release-fast python3 examples/ffi_python/main.py

    # 2. Run (Windows)
    set PATH=target\\release-fast;%PATH%
    python examples\\ffi_python\\main.py

Requires a running Ollama with at least one pulled model (default:
"llama3.2:3b") at http://localhost:11434.
"""

import ctypes
import ctypes.util
import os
import sys
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
)


def load_library():
    """Load libai_assistant from the build output directory."""
    if sys.platform == "win32":
        name = "ai_assistant.dll"
    elif sys.platform == "darwin":
        name = "libai_assistant.dylib"
    else:
        name = "libai_assistant.so"

    # Try the standard release-fast output dir first, then fall back to
    # the system library search path.
    candidates = [
        os.path.join("target", "release-fast", name),
        os.path.join("target", "release", name),
        os.path.join("target", "debug", name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return CDLL(path)

    # Fall back: let the OS find it via LD_LIBRARY_PATH / PATH / etc.
    return CDLL(name)


# ── Provider enum values (must match AiProviderKind in ai_assistant.h) ──
AI_PROVIDER_KIND_OLLAMA = 0
AI_PROVIDER_KIND_OPEN_AI = 6
AI_PROVIDER_KIND_ANTHROPIC = 7

# ── Callback type for streaming ──
StreamCallback = CFUNCTYPE(None, c_char_p, c_bool, c_void_p)


def setup_prototypes(lib):
    """Declare ctypes prototypes for every FFI function."""
    # Lifecycle
    lib.ai_assistant_new.restype = c_void_p
    lib.ai_assistant_new.argtypes = []

    lib.ai_assistant_new_with_prompt.restype = c_void_p
    lib.ai_assistant_new_with_prompt.argtypes = [c_char_p]

    lib.ai_assistant_free.restype = None
    lib.ai_assistant_free.argtypes = [c_void_p]

    # Configuration setters
    lib.ai_assistant_set_system_prompt.restype = c_int
    lib.ai_assistant_set_system_prompt.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_provider.restype = c_int
    lib.ai_assistant_set_provider.argtypes = [c_void_p, c_int]

    lib.ai_assistant_set_model.restype = c_int
    lib.ai_assistant_set_model.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_api_key.restype = c_int
    lib.ai_assistant_set_api_key.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_ollama_url.restype = c_int
    lib.ai_assistant_set_ollama_url.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_openai_compatible_url.restype = c_int
    lib.ai_assistant_set_openai_compatible_url.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_bedrock_region.restype = c_int
    lib.ai_assistant_set_bedrock_region.argtypes = [c_void_p, c_char_p]

    lib.ai_assistant_set_temperature.restype = c_int
    lib.ai_assistant_set_temperature.argtypes = [c_void_p, c_float]

    lib.ai_assistant_set_max_history.restype = c_int
    lib.ai_assistant_set_max_history.argtypes = [c_void_p, c_size_t]

    # Messaging
    lib.ai_assistant_send_message.restype = c_int
    lib.ai_assistant_send_message.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]

    lib.ai_assistant_send_message_stream.restype = c_int
    lib.ai_assistant_send_message_stream.argtypes = [
        c_void_p,
        c_char_p,
        StreamCallback,
        c_void_p,
    ]

    # Session control
    lib.ai_assistant_clear_conversation.restype = c_int
    lib.ai_assistant_clear_conversation.argtypes = [c_void_p]

    lib.ai_assistant_new_session.restype = c_int
    lib.ai_assistant_new_session.argtypes = [c_void_p]

    # Memory / diagnostics
    lib.ai_assistant_free_string.restype = None
    lib.ai_assistant_free_string.argtypes = [c_char_p]

    lib.ai_assistant_last_error.restype = c_char_p
    lib.ai_assistant_last_error.argtypes = []

    lib.ai_assistant_version.restype = c_char_p
    lib.ai_assistant_version.argtypes = []

    lib.ai_assistant_abi_version.restype = c_int
    lib.ai_assistant_abi_version.argtypes = []


def last_error(lib):
    """Return the thread-local last-error message, or '(no details)'."""
    err = lib.ai_assistant_last_error()
    if err:
        return err.decode("utf-8", errors="replace")
    return "(no details)"


def main():
    lib = load_library()
    setup_prototypes(lib)

    version = lib.ai_assistant_version().decode("utf-8")
    abi = lib.ai_assistant_abi_version()
    print(f"ai_assistant FFI demo (Python) — version {version}, ABI {abi}")

    # ── Create handle ──
    prompt = b"You are a friendly villager in a fantasy RPG. Keep all replies under 40 words and in-character."
    handle = lib.ai_assistant_new_with_prompt(prompt)
    if not handle:
        print(f"ai_assistant_new_with_prompt: {last_error(lib)}", file=sys.stderr)
        sys.exit(1)

    try:
        # ── Configure ──
        rc = lib.ai_assistant_set_provider(handle, AI_PROVIDER_KIND_OLLAMA)
        if rc != 0:
            print(f"set_provider: {last_error(lib)}", file=sys.stderr)
            sys.exit(1)

        rc = lib.ai_assistant_set_model(handle, b"llama3.2:3b")
        if rc != 0:
            print(f"set_model: {last_error(lib)}", file=sys.stderr)
            sys.exit(1)

        rc = lib.ai_assistant_set_temperature(handle, 0.8)
        if rc != 0:
            print(f"set_temperature: {last_error(lib)}", file=sys.stderr)
            sys.exit(1)

        # ── Blocking send ──
        reply_ptr = c_char_p()
        rc = lib.ai_assistant_send_message(
            handle,
            b"A weary traveler approaches. Greet them warmly.",
            ctypes.byref(reply_ptr),
        )
        if rc == 0 and reply_ptr.value:
            print(f"Villager: {reply_ptr.value.decode('utf-8')}")
            lib.ai_assistant_free_string(reply_ptr)
        else:
            print(f"send_message: {last_error(lib)}", file=sys.stderr)
            sys.exit(1)

        # ── Streaming example ──
        print("\n--- Streaming reply ---")

        @StreamCallback
        def on_chunk(chunk, is_final, _user_data):
            text = chunk.decode("utf-8", errors="replace") if chunk else ""
            if is_final:
                print(f"\n[stream complete]")
            else:
                print(text, end="", flush=True)

        rc = lib.ai_assistant_send_message_stream(
            handle,
            b"Tell me about the local tavern.",
            on_chunk,
            None,
        )
        if rc != 0:
            print(f"\nsend_message_stream: {last_error(lib)}", file=sys.stderr)

    finally:
        lib.ai_assistant_free(handle)


if __name__ == "__main__":
    main()
