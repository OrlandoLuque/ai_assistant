# `ai_test_harness` — Multi-category integration harness

| Field | Value |
|---|---|
| Group | Testing |
| Binary path | `src/bin/ai_test_harness.rs` |
| `required-features` | `full`, `browser` |

## Purpose

Runs feature-gated integration test categories without having to remember every `cargo test` incantation. Used by CI and by local developers to smoke-test a build against the real provider + browser path.

## Build

```bash
cargo build --release --bin ai_test_harness --features "full,browser"
```

## Categories (selection)

| Flag | Tests covered |
|---|---|
| `--category anti-hallucination` | 3 V88 tests |
| `--category quality-gates` | 4 V88 tests |
| `--category faithfulness` | 2 V88 tests |
| `--category verification` | 2 V88 tests |
| `--category research` | 4 V88 tests (requires `research` feature) |

## Usage

```bash
ai_test_harness --list
ai_test_harness --category anti-hallucination
ai_test_harness --all
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/IMPROVEMENTS_V88.md`](../IMPROVEMENTS_V88.md)
