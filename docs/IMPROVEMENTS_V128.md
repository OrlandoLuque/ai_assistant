# V128 — Phase C.7: backup/restore CLI

**Date**: 2026-05-06
**Version**: 0.2.75
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.7
**Tasks**: #337 (V128 C.7 — backup/restore CLI)

## Why

The library already had `setup::backup` (gz-tar of config dirs, no
crypto) and `encrypted_knowledge` (kpkg packages for the knowledge
plugin format). Neither covers the operator use case the plan
calls for: **make a sealed, verifiable, optionally-encrypted
snapshot of arbitrary source paths so a future restore can prove
the bytes match what was archived**.

C.7's deliverables are explicit: AES-256-GCM, SHA-256, signature.
The format the plan suggested was tar.zst, but `tar` and `zstd`
aren't dependencies and `zip` already is — V128 uses ZIP to keep
the dep graph stable while still meeting every functional
requirement (multi-file archive + per-file integrity + outer
encryption envelope).

## What changed

### `src/secure_backup.rs` (new module, behind feature `backup`)

The module name is `secure_backup` rather than `backup` because
`setup::backup` already exists and is re-exported as `backup` from
the crate root. `secure_backup` keeps the name unambiguous.

Three public entry points:

```rust
pub fn create_backup(config: &BackupConfig)        -> Result<BackupReport, BackupError>;
pub fn verify_backup(path, key, verify_key)        -> Result<BackupReport, BackupError>;
pub fn restore_backup(path, out_dir, key, vk_opt)  -> Result<BackupReport, BackupError>;
```

#### Archive format

```text
plain (no encryption):
  <output>.zip                    — ZIP (manifest.json + files/<rel>/...)
  <output>.zip.sha256             — hex SHA-256 of the .zip
  <output>.zip.sig (optional)     — Ed25519 signature over the .zip

encrypted:
  <output>.zip.enc                — [1B version | 16B salt | 12B nonce | ciphertext+tag]
  <output>.zip.enc.sha256
  <output>.zip.enc.sig (optional)
```

The ZIP contains:

- `manifest.json` — `BackupManifest { version, created_at, source_label, entries[], total_bytes }`.
  Each `BackupEntry` has `relative_path`, `size`, and lowercase-hex
  `sha256` of the plain-text bytes.
- `files/<rel>` for each source file.

#### Encryption

- AES-256-GCM. Key is 32 bytes, supplied either directly or
  derived from a passphrase via HKDF-SHA256 with the per-archive
  16-byte salt that lives in the envelope header.
- Two ways to specify encryption material via the new
  `EncryptionMaterial` enum:
  ```rust
  pub enum EncryptionMaterial {
      Passphrase(String),  // lib derives + embeds salt
      Key([u8; 32]),       // caller manages key, salt slot is random
  }
  ```
  This avoids the salt-mismatch bug a naive
  derive-then-pass-key shape introduces (key derived from a
  throwaway salt won't match the salt cargo stores in the
  envelope). With `Passphrase`, the lib does the derivation
  *after* generating the salt, so they always agree.

#### Signing

Ed25519 over the **post-encryption** bytes (or post-zip when
plain). This lets a verifier authenticate the archive
without ever decrypting, useful for offline triage or
public-key distribution.

#### Path-safety

`restore_backup` rejects entries whose relative path is empty,
absolute (`/foo`, `\foo`), starts with a Windows drive prefix
(`C:\foo`), or contains a `..` segment — standard zip-slip
hardening.

#### Tests (7 new, all passing)

| Test | Covers |
|---|---|
| `round_trip_plain` | create → verify → restore without crypto |
| `round_trip_encrypted` | passphrase-based encryption + wrong-passphrase rejection |
| `round_trip_signed` | Ed25519 sign + verify + tamper-detection |
| `rejects_zip_slip` | `..`, absolute, drive-prefix paths all rejected |
| `detects_per_file_corruption` | flipping a bit inside the zip → per-file SHA-256 catches it |
| `empty_sources_fails` | empty source list returns Format error |
| `key_derivation_deterministic` | HKDF is deterministic on (passphrase, salt) |

### `src/bin/ai_backup.rs` (new binary)

```text
ai_backup create  --source <p> [--source <p>]... --output <base> [--passphrase-env VAR] [--sign-key <path>] [--label <text>]
ai_backup verify  --input <archive> [--passphrase-env VAR] [--verify-key <path>]
ai_backup restore --input <archive> --output <dir> [--passphrase-env VAR] [--verify-key <path>]
```

- Passphrase is read from a named environment variable, never
  argv (would leak to shell history).
- Sign / verify keys are 32-byte raw Ed25519 secret / public key
  files (matches `SigningKey::to_bytes` / `VerifyingKey::to_bytes`).
- Multiple `--source` flags allowed; each may be a file or
  directory (directories walk recursively).

### `Cargo.toml`

- New feature `backup = ["dep:zip", "dep:aes-gcm", "dep:sha2",
  "dep:ed25519-dalek", "dep:hkdf"]`.
- Added to the `full` feature set.
- New `[[bin]] ai_backup` with `required-features = ["backup"]`.
- **Implicit-feature shim**: adding `dep:` references to
  `aes-gcm`, `zip`, and `pdf-extract` from inside other features
  disabled cargo's implicit-feature creation for the same names.
  Several pre-existing `#[cfg(feature = "aes-gcm")]` /
  `#[cfg(feature = "pdf-extract")]` attributes in
  `src/content_encryption.rs`, `src/document_parsing/parser.rs`,
  etc. depended on those implicits. Restored them as explicit
  pass-through stubs:
  ```toml
  aes-gcm = ["dep:aes-gcm"]
  zip = ["dep:zip"]
  pdf-extract = ["dep:pdf-extract"]
  ```
  Pure mechanical compatibility — no behaviour change.
- `documents` and `rag` features rewritten to use `dep:` form
  for clarity (`dep:zip`, `dep:pdf-extract`, `dep:aes-gcm`).

### `src/lib.rs`

`#[cfg(feature = "backup")] pub mod secure_backup;` between
`audio_priority_protocol` and `batch`.

## Smoke-tested end-to-end

```bash
# Create encrypted, labelled snapshot of two source files.
AI_BACKUP_PASS=hunter2 ai_backup create \
  --source /tmp/src --output /tmp/snap \
  --passphrase-env AI_BACKUP_PASS --label smoke-test
# → /tmp/snap.zip.enc + /tmp/snap.zip.enc.sha256

# Verify (decrypts, walks per-file SHA-256s, prints OK).
AI_BACKUP_PASS=hunter2 ai_backup verify \
  --input /tmp/snap.zip.enc --passphrase-env AI_BACKUP_PASS

# Restore to a fresh dir; bytes match originals.
AI_BACKUP_PASS=hunter2 ai_backup restore \
  --input /tmp/snap.zip.enc --output /tmp/restore \
  --passphrase-env AI_BACKUP_PASS
```

All three subcommands work. `cargo test --lib` reports 6203
passing tests (+7 new from V128, baseline 6196).

## Compatibility

- Pure addition. The new `backup` feature is in the `full` set,
  so any caller already using `default-features = full` picks up
  the binary automatically. Callers not using `full` keep their
  current dep graph.
- The `aes-gcm` / `zip` / `pdf-extract` explicit stubs preserve
  every existing `#[cfg(feature = "X")]` attribute. No source
  files in the codebase changed except `lib.rs` (one new
  `pub mod` line).

## What's next

- V129 / C.8 — GDPR right-to-erasure (`purge_user(user_id)`
  consistent across RAG, memory, audit) + DPIA template.
