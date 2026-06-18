# IMPROVEMENTS_V159 — HTTPS/TLS for ai_proxy

**Version:** 0.2.109 → 0.2.110
**Scope:** `src/bin/ai_proxy.rs` + `examples/ai_proxy.toml`
**Feature:** `server-axum-tls` (existing; no new feature)

## Why

The binary catalogue page for `ai_proxy` honestly listed "no built-in
TLS termination" as a limitation. But the crate already ships the
`server-axum-tls` feature (axum-server + rustls) used by
`server_axum::run_tls`. `ai_proxy` simply didn't wire it. This closes
that gap so a hardened gateway can terminate HTTPS itself instead of
always needing a reverse proxy in front.

## What changed

### Config + CLI

- New `[tls]` TOML section with `cert_path` / `key_path` (both
  optional; `deny_unknown_fields` like the rest of the schema).
- New `--tls-cert <PATH>` / `--tls-key <PATH>` CLI flags that override
  the file (same CLI-wins precedence as every other flag).
- Resolution threads through `Effective` as `tls_cert` / `tls_key`.

### Serving

- The serve path now branches: if **both** cert and key are resolved,
  serve HTTPS via `axum_server::bind_rustls(addr, RustlsConfig::
  from_pem_file(cert, key))`; otherwise the existing plain
  `axum::serve`.
- Graceful shutdown on the TLS path uses an `axum_server::Handle`
  (`graceful_shutdown(Some(10s))`) since `axum::serve`'s
  `.with_graceful_shutdown` doesn't apply to `axum_server`.
- **CryptoProvider**: rustls 0.23 panics at runtime if it can't pick a
  provider when more than one is compiled in. axum-server's
  `tls-rustls` can pull `aws-lc-rs` alongside the crate's `ring`, so the
  TLS path installs `rustls::crypto::ring::default_provider()` explicitly
  before use (idempotent — Err just means already installed). This was
  caught by the end-to-end smoke test, not the compiler.

### Feature gating

- All TLS code is behind `#[cfg(feature = "server-axum-tls")]`. If TLS
  is configured but the binary lacks the feature, startup fails with a
  clear "rebuild with --features ... server-axum-tls" message rather
  than silently falling back to HTTP.

### Observability

- Startup banner and `--dry-run` report `http` vs `https`.

## Tests

- `test_parse_args_tls_flags` — `--tls-cert`/`--tls-key` parse.
- `test_tls_from_config_file` — `[tls]` section parses + resolves.
- `test_tls_cli_overrides_config` — CLI wins over file.
- `test_no_tls_by_default` — absent ⇒ plain HTTP.
- Existing `test_example_config_uncommented_parses` regression covers
  the new `[tls]` block in the example.

## End-to-end verification

A self-signed run (`openssl req -x509 ... -subj //CN=localhost`) served
`/metrics` over an HTTPS handshake:

```
New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384
subject=CN=localhost
Proxy ready (HTTPS). Forwarding requests on https://127.0.0.1:19449 ...
```

## Build

```bash
cargo build --release --bin ai_proxy \
    --features "server-axum,security,server-axum-tls"

ai_proxy --config ai_proxy.toml \
    --tls-cert /etc/ssl/ai_proxy/fullchain.pem \
    --tls-key  /etc/ssl/ai_proxy/privkey.pem
```

## Follow-ups

- The CI feature matrix already builds `server-axum,server-axum-tls`;
  no matrix change needed.
- Streaming output guardrails over SSE (the other documented gap)
  land in V160.
