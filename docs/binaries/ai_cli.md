# `ai_cli` — Power-user CLI

| Field | Value |
|---|---|
| Group | CLI |
| Binary path | `src/bin/ai_cli.rs` |
| `required-features` | — (works with defaults; richer behavior with `full`) |
| Recommended build features | `full,diagnostic-logging` |

## Purpose

Non-interactive command-line driver for scripting, CI/CD, and one-shot queries. It exposes the full surface of the library (providers, RAG, tools, cost, anti-hallucination, research, quality gates) as discrete subcommands.

## Build

```bash
cargo build --release --bin ai_cli --features "full,diagnostic-logging"
```

## Subcommands

`scan`, `providers`, `models`, `config`, `butler`, `query`, `bench`, `test`, `cost` (V77), `verify` (V88), `research` (V88, needs `research`), `quality` (V88), `help`.

## Usage

```bash
# Provider detection + one-shot query
ai_cli scan
ai_cli query -p "Hello" -P ollama
ai_cli models -P ollama
ai_cli config

# Verbose diagnostics (requires feature diagnostic-logging)
ai_cli -v   scan        # info
ai_cli -vv  scan        # debug
ai_cli -vvv scan        # trace
ai_cli --log-file diag.log scan

# V77 cost subcommand
ai_cli cost report  --snapshot cost.json
ai_cli cost budget  --snapshot cost.json
ai_cli cost export  --snapshot cost.json --output cost.csv --force

# V88 verify (anti-hallucination pipeline)
ai_cli verify "Is water wet?" --strategy mark --faithfulness --cove --quality-gates

# V88 research (academic literature, gated on research)
ai_cli research "transformer attention" --providers arxiv,scholar --bibtex --review
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/GUIDE_ANTI_HALLUCINATION.md`](../GUIDE_ANTI_HALLUCINATION.md)
- [`docs/GUIDE_RESEARCH.md`](../GUIDE_RESEARCH.md)
- [`CHANGELOG.md`](../../CHANGELOG.md)
