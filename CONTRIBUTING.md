# Contributing to ai_assistant

Thank you for your interest in contributing to **ai_assistant**.

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.
See [LICENSE](LICENSE) for details.

## Contributor Assignment Agreement (CLA)

Before any contribution can be accepted, you **must** sign the project's
[Contributor Assignment Agreement (CLA)](CLA.md).

### Why a CLA?

The CLA assigns copyright of your contributions to the project owner.
This allows the project to be offered under commercial licenses in addition
to the noncommercial license, which is essential for the project's
sustainability.

### How to sign

When you open a Pull Request, add a comment stating:

> I have read the CLA and I agree to its terms.

A GitHub Action will verify your signature. Your PR cannot be merged
until the CLA is signed.

## Code Style

This project follows strict quality standards:

- **`cargo fmt`** — all code must be formatted with `rustfmt`
- **`cargo clippy`** — zero warnings (run with `-- -D warnings`)
- **Zero `.unwrap()`** in production code — use proper error handling
  (`?`, `map_err`, `anyhow`, etc.)
- **Zero compiler warnings** — clean compilation across all feature
  flag combinations
- **Tests required** — every new feature or bug fix must include tests

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`
2. **Implement** your changes following the code style above
3. **Add tests** for your changes
4. **Run locally**:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
5. **Open a Pull Request** against `master`
6. **Sign the CLA** by commenting on your PR
7. Wait for review — the project owner will review and provide feedback

## What to Contribute

- Bug fixes with regression tests
- Documentation improvements
- New tests for existing functionality
- Performance improvements with benchmarks

For larger features, please open an issue first to discuss the approach
before investing time in implementation.

## Questions?

Open an issue on the repository or contact the project owner.
