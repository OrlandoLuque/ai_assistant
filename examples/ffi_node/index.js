#!/usr/bin/env node
/**
 * ai_assistant — V79 FFI Node.js example (ffi-napi / koffi).
 *
 * Minimal "NPC in a game" driver: creates an assistant with a villager
 * system prompt, sets the Ollama provider + model, sends one blocking
 * message, prints the reply, and cleans up.
 *
 * Usage:
 *   # 1. Build the shared library
 *   cargo build --features ffi --profile release-fast
 *
 *   # 2. Install the Node FFI bridge
 *   cd examples/ffi_node && npm install
 *
 *   # 3. Run (Linux)
 *   LD_LIBRARY_PATH=../../target/release-fast node index.js
 *
 *   # 3. Run (macOS)
 *   DYLD_LIBRARY_PATH=../../target/release-fast node index.js
 *
 *   # 3. Run (Windows — from repo root)
 *   set PATH=target\release-fast;%PATH%
 *   node examples\ffi_node\index.js
 *
 * Requires a running Ollama with at least one pulled model (default:
 * "llama3.2:3b") at http://localhost:11434.
 */

"use strict";

const koffi = require("koffi");
const path = require("path");
const os = require("os");

// ── Load library ──────────────────────────────────────────────────────

function libraryPath() {
  const base = path.resolve(__dirname, "..", "..", "target", "release-fast");
  switch (os.platform()) {
    case "win32":
      return path.join(base, "ai_assistant.dll");
    case "darwin":
      return path.join(base, "libai_assistant.dylib");
    default:
      return path.join(base, "libai_assistant.so");
  }
}

const lib = koffi.load(libraryPath());

// ── Provider enum (must match AiProviderKind in ai_assistant.h) ──────

const AI_PROVIDER_KIND_OLLAMA = 0;
const AI_PROVIDER_KIND_OPEN_AI = 6;
const AI_PROVIDER_KIND_ANTHROPIC = 7;

// ── Declare FFI prototypes ───────────────────────────────────────────

// Opaque handle — we use 'void *' from the JS side.

const ai_assistant_new = lib.func("void* ai_assistant_new()");

const ai_assistant_new_with_prompt = lib.func(
  "void* ai_assistant_new_with_prompt(const char *prompt)"
);

const ai_assistant_free = lib.func(
  "void ai_assistant_free(void *handle)"
);

const ai_assistant_set_provider = lib.func(
  "int ai_assistant_set_provider(void *handle, int kind)"
);

const ai_assistant_set_model = lib.func(
  "int ai_assistant_set_model(void *handle, const char *model)"
);

const ai_assistant_set_api_key = lib.func(
  "int ai_assistant_set_api_key(void *handle, const char *key)"
);

const ai_assistant_set_temperature = lib.func(
  "int ai_assistant_set_temperature(void *handle, float temperature)"
);

const ai_assistant_set_max_history = lib.func(
  "int ai_assistant_set_max_history(void *handle, uintptr_t max_history)"
);

const ai_assistant_set_ollama_url = lib.func(
  "int ai_assistant_set_ollama_url(void *handle, const char *url)"
);

const ai_assistant_set_system_prompt = lib.func(
  "int ai_assistant_set_system_prompt(void *handle, const char *prompt)"
);

const ai_assistant_set_openai_compatible_url = lib.func(
  "int ai_assistant_set_openai_compatible_url(void *handle, const char *url)"
);

const ai_assistant_set_bedrock_region = lib.func(
  "int ai_assistant_set_bedrock_region(void *handle, const char *region)"
);

const ai_assistant_send_message = lib.func(
  "int ai_assistant_send_message(void *handle, const char *prompt, char **out)"
);

// Streaming callback: void (*)(const char *chunk, bool is_final, void *user_data)
const StreamCallback = koffi.proto(
  "void StreamCallback(const char *chunk, bool is_final, void *user_data)"
);

const ai_assistant_send_message_stream = lib.func(
  "int ai_assistant_send_message_stream(void *handle, const char *prompt, StreamCallback *callback, void *user_data)"
);

const ai_assistant_clear_conversation = lib.func(
  "int ai_assistant_clear_conversation(void *handle)"
);

const ai_assistant_new_session = lib.func(
  "int ai_assistant_new_session(void *handle)"
);

const ai_assistant_free_string = lib.func(
  "void ai_assistant_free_string(char *s)"
);

const ai_assistant_last_error = lib.func(
  "const char* ai_assistant_last_error()"
);

const ai_assistant_version = lib.func(
  "const char* ai_assistant_version()"
);

const ai_assistant_abi_version = lib.func(
  "int ai_assistant_abi_version()"
);

// ── Helpers ──────────────────────────────────────────────────────────

function lastError() {
  const err = ai_assistant_last_error();
  return err || "(no details)";
}

function check(rc, context) {
  if (rc !== 0) {
    throw new Error(`${context}: rc=${rc} — ${lastError()}`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────

function main() {
  const version = ai_assistant_version();
  const abi = ai_assistant_abi_version();
  console.log(`ai_assistant FFI demo (Node.js) — version ${version}, ABI ${abi}`);

  const handle = ai_assistant_new_with_prompt(
    "You are a friendly villager in a fantasy RPG. " +
    "Keep all replies under 40 words and in-character."
  );
  if (!handle) {
    console.error(`ai_assistant_new_with_prompt: ${lastError()}`);
    process.exit(1);
  }

  try {
    check(ai_assistant_set_provider(handle, AI_PROVIDER_KIND_OLLAMA), "set_provider");
    check(ai_assistant_set_model(handle, "llama3.2:3b"), "set_model");
    check(ai_assistant_set_temperature(handle, 0.8), "set_temperature");

    // ── Blocking send ──
    const outBuf = [null];
    const rc = ai_assistant_send_message(
      handle,
      "A weary traveler approaches. Greet them warmly.",
      outBuf
    );
    if (rc === 0 && outBuf[0]) {
      console.log(`Villager: ${outBuf[0]}`);
      ai_assistant_free_string(outBuf[0]);
    } else {
      console.error(`send_message: ${lastError()}`);
      process.exit(1);
    }

    // ── Streaming example ──
    console.log("\n--- Streaming reply ---");
    const streamCb = koffi.register((chunk, isFinal, _userData) => {
      if (isFinal) {
        console.log("\n[stream complete]");
      } else if (chunk) {
        process.stdout.write(chunk);
      }
    }, koffi.pointer(StreamCallback));

    const rc2 = ai_assistant_send_message_stream(
      handle,
      "Tell me about the local tavern.",
      streamCb,
      null
    );
    if (rc2 !== 0) {
      console.error(`\nsend_message_stream: ${lastError()}`);
    }
  } finally {
    ai_assistant_free(handle);
  }
}

main();
