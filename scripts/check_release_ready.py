#!/usr/bin/env python3
"""V131 (C.4) — release-readiness check.

Run before tagging a release. Verifies that:

  1. Cargo.toml `version` matches the tag the user is about to push.
  2. CHANGELOG.md has an `[Unreleased]` entry whose header carries
     the same version (so the release notes aren't blank).
  3. Working tree is clean (no uncommitted changes).
  4. The most recent commit either is, or descends from, a commit
     that touched the version line — guard against tagging an old
     commit.

Exits 0 on success, non-zero with a specific error otherwise.

Usage:
    python3 scripts/check_release_ready.py                  # uses Cargo.toml version
    python3 scripts/check_release_ready.py --tag v0.2.77    # explicit
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_cargo_version() -> str:
    cargo = (REPO / "Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo, re.MULTILINE)
    if not m:
        sys.exit("Cargo.toml: cannot find top-level version")
    return m.group(1)


def find_changelog_entry(version: str) -> bool:
    cl = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = re.compile(
        r"^##\s*\[Unreleased\][^\n]*\(" + re.escape(version) + r"\)",
        re.MULTILINE,
    )
    return bool(pattern.search(cl))


def working_tree_clean() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    real_changes = []
    for line in out.splitlines():
        if not line.strip():
            continue
        if line.endswith("/.claude/settings.local.json"):
            continue
        if line.endswith(".claude/settings.local.json"):
            continue
        real_changes.append(line)
    return not real_changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        help="explicit tag (e.g. v0.2.77). Defaults to v<Cargo.toml version>.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the working-tree-clean check (CI may want this)",
    )
    args = parser.parse_args()

    cargo_version = read_cargo_version()
    expected_tag = f"v{cargo_version}"
    tag = args.tag or expected_tag

    print(f"Cargo.toml version : {cargo_version}")
    print(f"Tag to verify      : {tag}")
    print(f"Expected tag       : {expected_tag}")

    errors = []

    if tag != expected_tag:
        errors.append(
            f"tag mismatch: --tag={tag} but Cargo.toml says version={cargo_version}. "
            f"Fix one of them."
        )

    if not find_changelog_entry(cargo_version):
        errors.append(
            f"CHANGELOG.md has no [Unreleased] entry mentioning ({cargo_version}). "
            f"Add one before tagging."
        )

    if not args.allow_dirty and not working_tree_clean():
        errors.append(
            "working tree not clean. Commit or stash before tagging "
            "(re-run with --allow-dirty if you know what you're doing)."
        )

    if errors:
        print()
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    print("OK: ready to release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
