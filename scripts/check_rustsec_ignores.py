#!/usr/bin/env python3
"""Keep every place that talks about suppressed advisories agreeing.

Every ``--ignore RUSTSEC-XXXX-NNNN`` is a promise that the advisory does not
apply to our usage. The promise is made in THREE files that nothing kept in step:

* ``deny.toml`` — the ``ignore`` array read by ``cargo deny``.
* ``.github/workflows/ci.yml`` — the ``--ignore`` flags passed to ``cargo audit``.
* ``.github/workflows/supply-chain.yml`` — a second ``cargo audit`` job, which
  exists so a scheduled supply-chain run does not depend on the main CI workflow.

**This script used to compare only the first two**, while an inline shell block in
``supply-chain.yml`` compared all three. Two implementations of one rule, and the
weaker one was the one a human runs locally: on 2026-09-27 I deleted an ignore
from ``ci.yml`` and ``deny.toml``, ran this script, got OK, pushed, and CI went
red on the third file. So the shell block is gone and this is the only
implementation — because that is the shape of defect this repository keeps
finding, and having it inside the *checker* was the joke.

They agree today. Nothing made them, and they have drifted before: an advisory
suppressed in one tool and enforced in the other means the gate that catches it
depends on which tool runs, which is the same as not knowing.

The third place is the monthly review workflow, which derives its list from
``deny.toml``. Its extraction used to be::

    sed -n '/\\[advisories\\]/,/^\\[/p' deny.toml | grep -oE 'RUSTSEC-[0-9]+-[0-9]+'

— a range scan that reads the **comments** as data. ``RUSTSEC-2026-0222`` was
deleted from the list on purpose, and the comment recording that deletion put it
straight back into the review issue. A comment documenting an *absence* was read
as a *presence*, so the operator was sent to re-check something that is not
suppressed, and told by implication that it is.

What this script checks:

1. All the lists contain exactly the same advisory IDs.
2. Every entry in ``deny.toml`` carries a comment above it saying why.
3. No ID is counted from a comment — only from the array itself.

Exit status is 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ADVISORY = re.compile(r"RUSTSEC-\d{4}-\d{4}")
QUOTED_ADVISORY = re.compile(r'"(RUSTSEC-\d{4}-\d{4})"')


def deny_entries(path: Path) -> tuple[list[str], list[str]]:
    """Advisory IDs in ``deny.toml``'s ignore array, and any lacking a reason.

    Only quoted entries inside ``ignore = [ ... ]`` count. A line whose first
    non-space character is ``#`` is a comment: it may mention an advisory, and
    mentioning one is not suppressing it.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    try:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("ignore = ["))
    except StopIteration:
        sys.exit(f"{path}: no `ignore = [` array found")

    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip() == "]"),
        None,
    )
    if end is None:
        sys.exit(f"{path}: the `ignore` array is never closed")

    found: list[str] = []
    undocumented: list[str] = []
    comment_run: list[str] = []

    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_run.append(stripped.lstrip("#").strip())
            continue
        match = QUOTED_ADVISORY.search(stripped)
        if not match:
            if not stripped:
                comment_run.clear()
            continue
        advisory = match.group(1)
        found.append(advisory)
        # A reason has to say something. A comment that is only the ID again
        # is the same as no comment: it repeats what the entry already says.
        reason = " ".join(comment_run).strip()
        without_ids = ADVISORY.sub("", reason).strip(" .,:;-")
        if len(without_ids) < 15:
            undocumented.append(advisory)
        comment_run.clear()

    return found, undocumented


def ci_entries(path: Path) -> list[str]:
    """Advisory IDs actually passed to ``cargo audit`` as ``--ignore`` flags.

    Comments in this file list the same IDs with their reasons, so matching on
    the flag rather than the ID is the whole point.
    """
    text = path.read_text(encoding="utf-8")
    return re.findall(r"--ignore\s+(RUSTSEC-\d{4}-\d{4})", text)


DEFAULT_AUDIT_FILES = (
    ".github/workflows/ci.yml",
    ".github/workflows/supply-chain.yml",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny", default="deny.toml")
    parser.add_argument(
        "--audit",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "a workflow passing --ignore flags to cargo audit; repeatable. "
            f"Defaults to all of: {', '.join(DEFAULT_AUDIT_FILES)}"
        ),
    )
    # Kept so an existing invocation does not break, but it no longer means
    # "the one file to check" -- adding a file must not silently drop the others,
    # which is how a three-way check became a two-way one.
    parser.add_argument("--ci", default=None, help="alias for --audit (deprecated)")
    args = parser.parse_args()

    audit_names = list(args.audit)
    if args.ci:
        audit_names.append(args.ci)
    if not audit_names:
        audit_names = list(DEFAULT_AUDIT_FILES)

    deny_path = Path(args.deny)
    audit_paths = [Path(name) for name in dict.fromkeys(audit_names)]
    for path in (deny_path, *audit_paths):
        if not path.is_file():
            sys.exit(f"not found: {path}")

    deny_list, undocumented = deny_entries(deny_path)
    audit_lists = {path: ci_entries(path) for path in audit_paths}

    problems: list[str] = []

    duplicates = {a for a in deny_list if deny_list.count(a) > 1}
    if duplicates:
        problems.append(f"{deny_path}: listed twice: {', '.join(sorted(duplicates))}")

    # Compared file by file against deny.toml rather than against a union: a union
    # is satisfied when an ID is present SOMEWHERE, which is exactly the state
    # that makes which-gate-catches-it depend on which workflow ran.
    for path, audit_list in audit_lists.items():
        only_deny = sorted(set(deny_list) - set(audit_list))
        only_audit = sorted(set(audit_list) - set(deny_list))
        if only_deny:
            problems.append(
                f"suppressed in {deny_path} but NOT in {path}: {', '.join(only_deny)}\n"
                f"    Add `--ignore <id>` in {path}, or drop it from {deny_path}."
            )
        if only_audit:
            problems.append(
                f"suppressed in {path} but NOT in {deny_path}: {', '.join(only_audit)}\n"
                f"    Add it to the ignore array in {deny_path}, or drop the flag in {path}."
            )

    if undocumented:
        problems.append(
            f"{deny_path}: no reason given above: {', '.join(undocumented)}\n"
            "    Every suppression is a claim that the advisory does not reach our\n"
            "    usage. Say why, and when to look again."
        )

    if problems:
        print("RUSTSEC ignore lists disagree:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nSee docs/runbooks/rustsec-handling.md. The monthly review derives its\n"
            "list from deny.toml, so an entry missing there is never re-checked."
        )
        return 1

    # Names every file it actually compared. The previous message named two files
    # while a second checker compared three, so "OK - identical in deny.toml and
    # ci.yml" was true and still let a real drift through.
    where = ", ".join(str(path) for path in (deny_path, *audit_paths))
    print(f"OK - {len(deny_list)} suppressed advisories, identical in {where}, each with a reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
