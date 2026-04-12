/*
 * ai_assistant — V79 FFI Java example (JNA).
 *
 * Minimal "NPC in a game" driver: creates an assistant with a villager
 * system prompt, sets the Ollama provider + model, sends one blocking
 * message, prints the reply, and cleans up.
 *
 * Usage:
 *   # 1. Build the shared library
 *   cargo build --features ffi --profile release-fast
 *
 *   # 2. Download JNA (one-time)
 *   curl -LO https://repo1.maven.org/maven2/net/java/dev/jna/jna/5.14.0/jna-5.14.0.jar
 *
 *   # 3. Compile
 *   javac -cp jna-5.14.0.jar examples/ffi_java/AiAssistantDemo.java
 *
 *   # 4. Run (Linux)
 *   LD_LIBRARY_PATH=target/release-fast \
 *       java -cp jna-5.14.0.jar:examples/ffi_java AiAssistantDemo
 *
 *   # 4. Run (Windows)
 *   set PATH=target\release-fast;%PATH%
 *   java -cp "jna-5.14.0.jar;examples\ffi_java" AiAssistantDemo
 *
 * Requires a running Ollama with a pulled model (default: "llama3.2:3b")
 * at http://localhost:11434.
 */

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

/**
 * JNA interface mapping for the ai_assistant C FFI.
 */
interface AiAssistantLib extends Library {

    // ── Provider enum values ────────────────────────────────────────
    int AI_PROVIDER_KIND_OLLAMA     = 0;
    int AI_PROVIDER_KIND_OPEN_AI    = 6;
    int AI_PROVIDER_KIND_ANTHROPIC  = 7;

    // ── Lifecycle ───────────────────────────────────────────────────
    Pointer ai_assistant_new();
    Pointer ai_assistant_new_with_prompt(String prompt);
    void    ai_assistant_free(Pointer handle);

    // ── Configuration setters ───────────────────────────────────────
    int ai_assistant_set_system_prompt(Pointer handle, String prompt);
    int ai_assistant_set_provider(Pointer handle, int kind);
    int ai_assistant_set_model(Pointer handle, String model);
    int ai_assistant_set_api_key(Pointer handle, String key);
    int ai_assistant_set_ollama_url(Pointer handle, String url);
    int ai_assistant_set_openai_compatible_url(Pointer handle, String url);
    int ai_assistant_set_bedrock_region(Pointer handle, String region);
    int ai_assistant_set_temperature(Pointer handle, float temperature);
    int ai_assistant_set_max_history(Pointer handle, long maxHistory);

    // ── Messaging ───────────────────────────────────────────────────
    int ai_assistant_send_message(Pointer handle, String prompt,
                                  PointerByReference out);

    // Streaming callback type
    interface StreamCallback extends Callback {
        void invoke(String chunk, boolean isFinal, Pointer userData);
    }

    int ai_assistant_send_message_stream(Pointer handle, String prompt,
                                          StreamCallback callback,
                                          Pointer userData);

    // ── Session control ─────────────────────────────────────────────
    int ai_assistant_clear_conversation(Pointer handle);
    int ai_assistant_new_session(Pointer handle);

    // ── Memory / diagnostics ────────────────────────────────────────
    void   ai_assistant_free_string(Pointer s);
    String ai_assistant_last_error();
    String ai_assistant_version();
    int    ai_assistant_abi_version();
}

/**
 * Demo entry point — NPC villager conversation.
 */
public class AiAssistantDemo {

    private static AiAssistantLib loadLib() {
        // JNA will look for the library using the standard platform search:
        //   Linux:   LD_LIBRARY_PATH or /usr/local/lib etc.
        //   macOS:   DYLD_LIBRARY_PATH or /usr/local/lib etc.
        //   Windows: PATH
        return Native.load("ai_assistant", AiAssistantLib.class);
    }

    private static String lastError(AiAssistantLib lib) {
        String err = lib.ai_assistant_last_error();
        return err != null ? err : "(no details)";
    }

    private static void check(AiAssistantLib lib, int rc, String context) {
        if (rc != 0) {
            throw new RuntimeException(
                context + ": rc=" + rc + " — " + lastError(lib));
        }
    }

    public static void main(String[] args) {
        AiAssistantLib lib = loadLib();

        String version = lib.ai_assistant_version();
        int abi = lib.ai_assistant_abi_version();
        System.out.printf("ai_assistant FFI demo (Java) — version %s, ABI %d%n",
                          version, abi);

        Pointer handle = lib.ai_assistant_new_with_prompt(
            "You are a friendly villager in a fantasy RPG. " +
            "Keep all replies under 40 words and in-character.");
        if (handle == null) {
            System.err.println("ai_assistant_new_with_prompt: " + lastError(lib));
            System.exit(1);
        }

        try {
            check(lib, lib.ai_assistant_set_provider(
                handle, AiAssistantLib.AI_PROVIDER_KIND_OLLAMA), "set_provider");
            check(lib, lib.ai_assistant_set_model(
                handle, "llama3.2:3b"), "set_model");
            check(lib, lib.ai_assistant_set_temperature(
                handle, 0.8f), "set_temperature");

            // ── Blocking send ──
            PointerByReference outRef = new PointerByReference();
            int rc = lib.ai_assistant_send_message(
                handle,
                "A weary traveler approaches. Greet them warmly.",
                outRef);
            if (rc == 0 && outRef.getValue() != null) {
                String reply = outRef.getValue().getString(0, "UTF-8");
                System.out.println("Villager: " + reply);
                lib.ai_assistant_free_string(outRef.getValue());
            } else {
                System.err.println("send_message: " + lastError(lib));
                System.exit(1);
            }

            // ── Streaming example ──
            System.out.println("\n--- Streaming reply ---");
            AiAssistantLib.StreamCallback callback = (chunk, isFinal, userData) -> {
                if (isFinal) {
                    System.out.println("\n[stream complete]");
                } else if (chunk != null) {
                    System.out.print(chunk);
                    System.out.flush();
                }
            };

            int rc2 = lib.ai_assistant_send_message_stream(
                handle,
                "Tell me about the local tavern.",
                callback,
                null);
            if (rc2 != 0) {
                System.err.println("\nsend_message_stream: " + lastError(lib));
            }

        } finally {
            lib.ai_assistant_free(handle);
        }
    }
}
