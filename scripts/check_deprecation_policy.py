#!/usr/bin/env python3
"""
V127 (Phase C.6) — deprecation-policy gate.

Scans `src/**/*.rs` for `#[deprecated(...)]` attributes and fails
(exit 1) if any of them lack a `since = "..."` field or a
`note = "..."` field. Both are required by the lifecycle policy
documented in `docs/FEATURE_LIFECYCLE.md`:

  - `since = ` is the version that announced the deprecation, used
    by callers to tell when the migration window started.
  - `note = ` is the migration path. Without it a downstream user
    sees "deprecated" and has nowhere to go.

Multi-line attribute syntax is supported — the scanner buffers
until it sees the matching closing parenthesis.

Usage:
    python3 scripts/check_deprecation_policy.py [--root src]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Detect the start of a #[deprecated ...] attribute. We deliberately
# allow internal whitespace and ignore #[deprecated] (no parens) —
# bare-form is the rustc-default "no since/no note" form, which the
# policy forbids regardless.
DEPRECATED_START_RE = re.compile(r"#\s*\[\s*deprecated\b")
SINCE_RE = re.compile(r'since\s*=\s*"[^"]+"')
NOTE_RE = re.compile(r'note\s*=\s*"[^"]+"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("src"),
                   help="directory to scan (default: src)")
    return p.parse_args()


def collect_attributes(text: str, path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, full_attr_text) for each #[deprecated...] occurrence."""
    out: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if DEPRECATED_START_RE.search(line):
            # Walk forward until the bracket-paren depth returns to zero.
            buf: list[str] = []
            depth_paren = 0
            depth_bracket = 0
            start_line = i + 1  # 1-based
            saw_open = False
            j = i
            while j < len(lines):
                seg = lines[j]
                buf.append(seg)
                for ch in seg:
                    if ch == "[":
                        depth_bracket += 1
                        saw_open = True
                    elif ch == "]":
                        depth_bracket -= 1
                    elif ch == "(":
                        depth_paren += 1
                    elif ch == ")":
                        depth_paren -= 1
                if saw_open and depth_bracket == 0 and depth_paren == 0:
                    break
                j += 1
            attr_text = "\n".join(buf)
            out.append((start_line, attr_text))
            i = j + 1
        else:
            i += 1
    return out


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        sys.stderr.write(f"error: --root {args.root} is not a directory\n")
        return 2

    failures: list[str] = []
    total = 0

    for rs_path in sorted(args.root.rglob("*.rs")):
        try:
            text = rs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = rs_path.read_text(encoding="utf-8", errors="replace")

        for line_no, attr in collect_attributes(text, rs_path):
            total += 1
            missing: list[str] = []
            # Empty deprecation form `#[deprecated]` — no parens at all.
            # Has neither `since` nor `note`.
            if not SINCE_RE.search(attr):
                missing.append("since = \"...\"")
            if not NOTE_RE.search(attr):
                missing.append("note = \"...\"")
            if missing:
                rel = rs_path.as_posix()
                failures.append(
                    f"{rel}:{line_no}: missing {', '.join(missing)} "
                    f"in #[deprecated] attribute"
                )

    if failures:
        print(f"FAIL - {len(failures)} of {total} #[deprecated] "
              "attribute(s) violate the lifecycle policy:\n")
        for line in failures:
            print(f"  {line}")
        print()
        print("See docs/FEATURE_LIFECYCLE.md. Required form:")
        print('    #[deprecated(since = "0.2.x", '
              'note = "Use Foo instead - see docs/FEATURE_LIFECYCLE.md.")]')
        return 1

    print(f"OK - {total} #[deprecated] attribute(s) all carry "
          "since + note fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
