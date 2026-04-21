# `kpkg_tool` — Knowledge package manager

| Field | Value |
|---|---|
| Group | Knowledge |
| Binary path | `src/bin/kpkg_tool.rs` |
| `required-features` | `rag` |

## Purpose

Create, read, inspect, and extract encrypted knowledge packages (`.kpkg`) used by the RAG pipeline. Uses AES-256-GCM and supports optional Ed25519 signatures for tamper evidence.

## Build

```bash
cargo build --release --bin kpkg_tool --features rag
```

## Subcommands

| Command | Effect |
|---|---|
| `create` | Build a `.kpkg` from a directory |
| `list` | List entries in a package |
| `inspect` | Show metadata + signature info |
| `extract` | Decrypt a package into a directory |

## Usage

```bash
kpkg_tool create --input ./docs --output knowledge.kpkg --name "My Docs"
kpkg_tool list knowledge.kpkg
kpkg_tool inspect knowledge.kpkg
kpkg_tool extract --input knowledge.kpkg --output ./extracted
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/CONCEPTS.md`](../CONCEPTS.md) — `.kpkg` format
