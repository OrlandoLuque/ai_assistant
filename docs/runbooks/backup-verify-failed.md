# Runbook: `ai_backup verify` failed

**Severity**: P1 if this is your last good backup; P2 otherwise.
**Owner**: Platform / SRE.
**Last reviewed**: 2026-05-06 (V130).

The `ai_backup verify` subcommand (V128) walks the SHA-256 sidecar,
the per-file `manifest.json` digests, and (optionally) the Ed25519
signature. A non-zero exit means **at least one of those layers
disagrees**. This runbook walks you from the failure to a confident
"this backup is / is not usable".

## 1. Symptoms

* `ai_backup verify` exits non-zero.
* Possible specific errors (from `secure_backup::BackupError`):
  - `IntegrityCheckFailed` — sidecar SHA-256 differs from archive.
  - `Crypto(...)` — `aead::Error` decrypting (wrong passphrase or
    bit flip in ciphertext).
  - `Format(...)` — `manifest.json` missing, malformed, or
    references a relative path that violates zip-slip.
  - `Signature(...)` — Ed25519 verification failed.

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | Wrong passphrase / wrong env var | high |
| 2 | Wrong `--verify-key` (rotated since signing) | medium |
| 3 | Bit rot on the storage medium | medium |
| 4 | Archive truncated (interrupted upload/download) | medium |
| 5 | Tampering — caught by either SHA-256 or signature | low |
| 6 | Bug in a release; report upstream | very low |

## 3. Diagnose

```bash
# Confirm the archive is the size you expect (e.g. matches what s3
# ls says, or what the producer logged).
ls -l <archive>.zip.enc
shasum -a 256 <archive>.zip.enc                    # Linux/macOS
Get-FileHash -Algorithm SHA256 <archive>.zip.enc   # Windows PS

# Compare against the sidecar (sidecar SHA must match the on-disk).
cat <archive>.zip.enc.sha256

# If they disagree → the archive was modified or corrupted in
# storage/transport, regardless of crypto.

# If sidecar matches, drop crypto and inspect manifest:
ai_backup verify --input <archive>.zip.enc \
                 --passphrase-env AI_BACKUP_PASS 2>&1 | head -20

# If the error is signature-only, you might have a stale verify key:
ls -la <verify_key.bin>      # was this rotated?
```

## 4. Mitigate

**A. Passphrase wrong:**
- Re-enter via the env var:
  `AI_BACKUP_PASS=<passphrase> ai_backup verify --input ... --passphrase-env AI_BACKUP_PASS`.
- If the passphrase was rotated, look up the historic one in your
  password manager / vault. AES-GCM has no fallback; the wrong
  passphrase yields *exactly* the same `aead::Error` as bit-rot of
  the tag.

**B. Signature key rotated:**
- Find the public key that was current at the archive's
  `manifest.created_at`. Pass it via `--verify-key`.

**C. Truncation / bit-rot:**
- Re-fetch from the source. If the source is gone, you cannot
  recover this archive — proceed to the *previous* archive in your
  retention window and flag this archive as unusable.

**D. Tamper:**
- If sidecar SHA-256 matches the on-disk hash *and* signature
  fails, someone with sidecar-write access modified the archive.
  Treat as a security incident; preserve the original bytes for
  forensics.

**E. Suspected upstream bug:**
- File a bug with `ai_backup verify --input ... --verbose` output,
  the crate version, and a hex-dump of the first 64 bytes of the
  envelope (`xxd -l 64 <archive>.zip.enc`).

## 5. Resolve

* **Verify on a schedule.** A backup that has never been verified
  is not a backup. Add a daily job (`ai_jobs add ...`) that runs
  `ai_backup verify` against the most recent N archives.
* **Test restore quarterly.** Verification proves bytes are intact;
  restore proves the snapshot is *useful*. They are different
  guarantees.
* **Separate sidecar storage.** If sidecar (`.sha256`) lives in the
  same bucket as the archive, an attacker who can write the archive
  can also rewrite the sidecar. Park sidecars in an immutable
  store (object-lock S3, append-only ledger) so the SHA mismatch
  becomes detectable.
* **Sign every archive** — `--sign-key` adds an Ed25519 signature
  layer that the bucket-write-attacker cannot forge without the
  signing key.

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Archive id / path | |
| Expected size | from producer log |
| Observed size | from §3 |
| Failure mode | sidecar / crypto / format / signature |
| Cause | from §2 |
| Recovery | restored from / accepted loss |
| Action items | owner + due date |
