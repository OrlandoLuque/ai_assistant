#!/usr/bin/env python3
"""
V154 (#1B) — feature/dep drift gate.

Catches the class of bug that broke AES-256-GCM and PDF parsing in V152:
a feature whose enable-list references the optional DEPENDENCY `dep:X`
instead of the like-named FEATURE `X`, while source code is gated on
`cfg(feature = "X")`.

Why it matters:
    rag = ["rusqlite", "dep:aes-gcm"]   # enables the DEP, not the feature
    aes-gcm = ["dep:aes-gcm"]           # the feature `aes-gcm`
    // src: #[cfg(feature = "aes-gcm")] ...

Building with `rag` (but not `aes-gcm`) turns on the aes-gcm crate but
leaves every `cfg(feature = "aes-gcm")` gate OFF — so the code path is
silently disabled. CI never sees it because the lib tests for that path
sit behind the same cfg and don't compile either. The fix is to
reference the FEATURE:
    rag = ["rusqlite", "aes-gcm"]

The rule enforced here:
    For any feature F such that
      (a) F is a declared feature in [features], AND
      (b) source code uses `cfg(feature = "F")`, AND
      (c) some OTHER feature G (G != F) lists `dep:F` in its enable-list,
    then G is flagged: it should list `F`, not `dep:F`.

The feature F's own definition listing `dep:F` (the canonical
feature->dep mapping, e.g. `aes-gcm = ["dep:aes-gcm"]`) is allowed and
expected — only OTHER features referencing the dep directly are bugs.

Usage:
    python3 scripts/check_feature_dep_drift.py [--manifest Cargo.toml] [--root src]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CFG_FEATURE_RE = re.compile(r'cfg\s*\(\s*feature\s*=\s*"([^"]+)"')
# A feature entry inside [features]:  name = [ ... ]  (value may be multi-line)
FEATURE_ENTRY_RE = re.compile(
    r'^\s*([A-Za-z0-9_\-]+)\s*=\s*\[(.*?)\]',
    re.DOTALL | re.MULTILINE,
)
STR_ITEM_RE = re.compile(r'"([^"]+)"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    p.add_argument("--root", type=Path, default=Path("src"))
    return p.parse_args()


def extract_features_section(manifest_text: str) -> str:
    """Return the raw text of the [features] table only."""
    lines = manifest_text.splitlines()
    out: list[str] = []
    in_features = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_features = stripped == "[features]"
            continue
        if in_features:
            out.append(line)
    return "\n".join(out)


def parse_features(section: str) -> dict[str, list[str]]:
    """Map feature name -> list of enable-list items (strings)."""
    features: dict[str, list[str]] = {}
    for m in FEATURE_ENTRY_RE.finditer(section):
        name = m.group(1)
        body = m.group(2)
        # Strip line comments inside the array body before pulling strings.
        body_no_comments = "\n".join(
            seg.split("#", 1)[0] for seg in body.splitlines()
        )
        items = STR_ITEM_RE.findall(body_no_comments)
        features[name] = items
    return features


def collect_cfg_features(root: Path) -> set[str]:
    found: set[str] = set()
    for rs_path in root.rglob("*.rs"):
        try:
            text = rs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = rs_path.read_text(encoding="utf-8", errors="replace")
        for m in CFG_FEATURE_RE.finditer(text):
            found.add(m.group(1))
    return found


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        sys.stderr.write(f"error: --manifest {args.manifest} not found\n")
        return 2
    if not args.root.is_dir():
        sys.stderr.write(f"error: --root {args.root} is not a directory\n")
        return 2

    manifest_text = args.manifest.read_text(encoding="utf-8")
    section = extract_features_section(manifest_text)
    features = parse_features(section)
    cfg_features = collect_cfg_features(args.root)

    declared = set(features)
    failures: list[str] = []

    # For each feature G, look at every `dep:F` it references.
    for g_name, items in features.items():
        for item in items:
            if not item.startswith("dep:"):
                continue
            dep = item[len("dep:"):]
            # Only a problem when `dep` is ALSO a declared feature that
            # source code gates on, and the referencing feature is not
            # the canonical `dep == feature` mapping.
            if dep in declared and dep in cfg_features and g_name != dep:
                failures.append(
                    f'feature "{g_name}" lists "dep:{dep}" but "{dep}" is a '
                    f'feature with cfg(feature="{dep}") gates in src/. '
                    f'Building "{g_name}" enables the dependency but leaves '
                    f'those cfg gates OFF. Reference the feature instead: '
                    f'replace "dep:{dep}" with "{dep}".'
                )

    if failures:
        print(
            f"FAIL - {len(failures)} feature/dep drift issue(s) "
            "(the V152 class of silent feature-graph break):\n"
        )
        for line in sorted(set(failures)):
            print(f"  {line}\n")
        print("See docs/IMPROVEMENTS_V152.md (sections 1-2) for the bug class.")
        return 1

    n_gated = len(cfg_features & declared)
    print(
        f"OK - no feature/dep drift. {n_gated} cfg-gated feature(s) checked "
        f"against {len(features)} feature definition(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
