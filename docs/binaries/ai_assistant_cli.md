# `ai_assistant_cli` — Interactive chat REPL

| Field | Value |
|---|---|
| Group | CLI |
| Binary path | `src/bin/ai_assistant_cli.rs` |
| `required-features` | — |
| Recommended build features | `full,butler` |

## Purpose

Terminal-based chat REPL. Auto-detects local providers (Ollama / LM Studio), loads a default model, and opens an interactive session with slash-commands. Fastest way to "just talk to the thing" from a console.

## Build

```bash
cargo build --release --bin ai_assistant_cli --features "full,butler"
```

## Usage

```bash
ai_assistant_cli
ai_assistant_cli --provider ollama --model llama3.2:1b
```

### REPL slash-commands

| Command | Effect |
|---|---|
| `/help` | Show command list |
| `/models` | List available models for the active provider |
| `/model <name>` | Switch model mid-session |
| `/history` | Print the current transcript |
| `/clear` | Clear the conversation buffer |
| `/save <file>` | Persist the session to JSON |
| `/load <file>` | Restore a saved session |
| `/exit` | Quit |

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/GETTING_STARTED.md`](../GETTING_STARTED.md)
