# Release Process

**Last reviewed**: 2026-05-06 (V131).
**Owner**: Maintainer (Orlando).

## Cadence

* **Patch releases** (`0.2.x` → `0.2.x+1`) — every V cycle (one V
  number per cycle). Backwards-compatible additions only.
* **Minor / major releases** — only when a documented breaking
  change is shipped, following `docs/FEATURE_LIFECYCLE.md`.

The default policy in this repo is patch-level bumps (see
`memory/feedback_versioning.md`); skip minor/major unless the
change actually warrants it.

## Pre-flight

Before you tag, run:

```bash
python3 scripts/check_release_ready.py
```

This verifies that Cargo.toml's `version` matches the tag, that
CHANGELOG.md has an `[Unreleased]` entry for that version, and
that the working tree is clean. It exits non-zero with a specific
message if any of those fails.

If you skipped the V cycle's CHANGELOG update, do that first.

## Cutting the release

1. Confirm `Cargo.toml` is at the version you want to ship and the
   CHANGELOG entry is the V cycle's most recent.
2. Make sure CI is green on `master`.
3. Tag and push:

   ```bash
   tag="v$(grep -m1 '^version =' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
   git tag -a "$tag" -m "Release $tag"
   git push origin "$tag"
   ```

4. Wait for `.github/workflows/release.yml` to finish. It builds
   headless binaries on linux-x64 / macos-x64 / macos-arm64 /
   windows-x64, computes SHA-256 sidecars, signs each archive with
   cosign keyless, and attaches everything to the GitHub release.

5. Wait for `.github/workflows/supply-chain.yml` (also tag-
   triggered) to attach the CycloneDX SBOM (JSON + XML).

6. Verify the release page lists every artifact:
   - `ai_assistant-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` (+ `.sha256`, `.sig`, `.cert`)
   - `ai_assistant-vX.Y.Z-x86_64-apple-darwin.tar.gz` (+ sidecars)
   - `ai_assistant-vX.Y.Z-aarch64-apple-darwin.tar.gz` (+ sidecars)
   - `ai_assistant-vX.Y.Z-x86_64-pc-windows-msvc.zip` (+ sidecars)
   - `ai_assistant.cdx.json`, `ai_assistant.cdx.xml`

   Per `memory/feedback_release_always_include_binaries.md`, a
   release without the binary archives + SHA-256 sidecars is **not
   shipped** — re-run the workflow.

## Verifying a release

Anyone consuming a release should do:

```bash
# 1. Verify the SHA-256.
shasum -a 256 -c ai_assistant-vX.Y.Z-<target>.<ext>.sha256

# 2. Verify the cosign signature (optional but strongly recommended).
cosign verify-blob \
  --certificate ai_assistant-vX.Y.Z-<target>.<ext>.cert \
  --signature   ai_assistant-vX.Y.Z-<target>.<ext>.sig  \
  --certificate-identity-regexp "^https://github\\.com/<owner>/<repo>/\\.github/workflows/release\\.yml@refs/tags/v" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ai_assistant-vX.Y.Z-<target>.<ext>
```

The cosign keyless signing identity binds the artifact to a tag
push on this repository's `release.yml`. A verifier who pins the
identity above is protected against artifacts produced from an
arbitrary fork or branch.

## Rollback

If you need to retract a release:

1. **Delete the GitHub release** (releases page → ⋯ → Delete) but
   *not* the tag — the tag is immutable history. Replace the
   release notes with a "Retracted: <reason>" body and re-publish
   pointing at the same tag.
2. **Bump and re-cut.** Releases on this repo are patch-level; the
   right rollback is to ship `vX.Y.Z+1` containing the fix, not to
   regenerate `vX.Y.Z`.
3. Document the retraction in `CHANGELOG.md` under the new
   version's entry so consumers reading the changelog top-down see
   it.

## Manual / dry-run release

`release.yml` accepts `workflow_dispatch` with a `dry_run` boolean.
Use this to test a release pipeline change without producing a
public release. The dry-run skips the final publish step but still
exercises the build-zip-sign chain on every platform.

Cosign signing is also skipped on dispatch runs because the OIDC
token is only reliably available on `push`-triggered workflows;
this is by design (see workflow comments).
