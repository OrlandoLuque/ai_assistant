/*
 * ai_assistant — V79 FFI C example.
 *
 * Minimal "NPC in a game" driver: creates an assistant with a villager
 * system prompt, sets the Ollama provider + model, sends one blocking
 * message, prints the reply, and cleans up.
 *
 * Build (Linux/macOS):
 *   cargo build --features ffi --profile release-fast
 *   gcc -Wall -Wextra -I include examples/ffi_c/main.c \
 *       -L target/release-fast -lai_assistant -o /tmp/ffi_demo
 *   LD_LIBRARY_PATH=target/release-fast /tmp/ffi_demo
 *
 * Build (Windows MSVC):
 *   cargo build --features ffi --profile release-fast
 *   cl /I include examples\ffi_c\main.c /link ^
 *       /LIBPATH:target\release-fast ai_assistant.dll.lib
 *   set PATH=target\release-fast;%PATH%
 *   main.exe
 *
 * Requires a running Ollama with at least one pulled model (default:
 * "llama3.2:3b") at http://localhost:11434.
 */

#include "ai_assistant.h"
#include <stdio.h>
#include <stdlib.h>

static void print_err(const char *context) {
    const char *err = ai_assistant_last_error();
    if (err != NULL) {
        fprintf(stderr, "%s: %s\n", context, err);
    } else {
        fprintf(stderr, "%s: (no details)\n", context);
    }
}

int main(void) {
    printf("ai_assistant FFI demo — version %s, ABI %d\n",
           ai_assistant_version(),
           ai_assistant_abi_version());

    AiAssistantHandle *h = ai_assistant_new_with_prompt(
        "You are a friendly villager in a fantasy RPG. "
        "Keep all replies under 40 words and in-character.");
    if (h == NULL) {
        print_err("ai_assistant_new_with_prompt");
        return 1;
    }

    int rc = ai_assistant_set_provider(h, AI_PROVIDER_KIND_OLLAMA);
    if (rc != 0) {
        print_err("ai_assistant_set_provider");
        ai_assistant_free(h);
        return 1;
    }

    rc = ai_assistant_set_model(h, "llama3.2:3b");
    if (rc != 0) {
        print_err("ai_assistant_set_model");
        ai_assistant_free(h);
        return 1;
    }

    rc = ai_assistant_set_temperature(h, 0.8f);
    if (rc != 0) {
        print_err("ai_assistant_set_temperature");
        ai_assistant_free(h);
        return 1;
    }

    char *reply = NULL;
    rc = ai_assistant_send_message(
        h,
        "A weary traveler approaches. Greet them warmly.",
        &reply);
    if (rc == 0 && reply != NULL) {
        printf("Villager: %s\n", reply);
        ai_assistant_free_string(reply);
        reply = NULL;
    } else {
        print_err("ai_assistant_send_message");
        ai_assistant_free(h);
        return 1;
    }

    ai_assistant_free(h);
    h = NULL;
    return 0;
}
