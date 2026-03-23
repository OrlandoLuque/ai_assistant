# Acknowledgments

## Author

**Orlando José Luque Moraira (Lander)** — sole author and developer.

## Development Assistance

This project was developed with the assistance of Claude (Anthropic) for code generation,
architecture design, documentation, and iterative refinement across 50+ development sessions.

## Research Foundations

The RAG pipeline, context management, and retrieval strategies are informed by research from:

- **Microsoft Research** — LLMLingua context compression series
- **Meta AI** — Self-RAG reflection mechanisms
- **Naver Labs** — Provence relevance filtering
- Various authors of the papers listed in [REFERENCES.md](REFERENCES.md)

## Open Source Dependencies

This project builds on the Rust ecosystem. Key dependencies include:

- [serde](https://serde.rs/) — serialization framework
- [tokio](https://tokio.rs/) — async runtime
- [axum](https://github.com/tokio-rs/axum) — HTTP server framework
- [rusqlite](https://github.com/rusqlite/rusqlite) — SQLite bindings
- [ureq](https://github.com/algesten/ureq) — HTTP client
- [egui](https://github.com/emilk/egui) — immediate-mode GUI
- [rayon](https://github.com/rayon-rs/rayon) — data parallelism
- [uuid](https://github.com/uuid-rs/uuid) — UUID generation
- [chrono](https://github.com/chronotope/chrono) — date/time
- [regex](https://github.com/rust-lang/regex) — regular expressions

See `Cargo.toml` for the complete list.
