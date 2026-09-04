#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that docs/BINARIES.md really is the inventory it claims to be.

BINARIES.md opens with "the authoritative inventory of every executable binary shipped
by the crate". For a long time it listed 26 of 41. A page that makes that claim and is
60 % complete is worse than no page at all: asked "does that binary exist?", it answers
a confident no.

This gate compares three things that must agree:

  1. every `[[bin]]` in Cargo.toml has a row in the summary table,
  2. every row in the table names a binary that Cargo.toml actually declares,
  3. the "Total binaries: N" header matches the real count,

and, so that fixing (1) cannot be done by pointing at pages that do not exist:

  4. every `binaries/<name>.md` link in the table resolves to a real file.

Exit code 0 when they agree, 1 otherwise. Run from the repository root.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARGO = os.path.join(ROOT, "Cargo.toml")
DOC = os.path.join(ROOT, "docs", "BINARIES.md")


def declared_binaries(text):
    """Names from every `[[bin]]` section of Cargo.toml."""
    names = []
    for block in re.split(r"^\[\[bin\]\]\s*$", text, flags=re.M)[1:]:
        # Stop at the next section header so a `name =` further down the file
        # cannot be mistaken for this binary's.
        block = re.split(r"^\[", block, maxsplit=1, flags=re.M)[0]
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', block, flags=re.M)
        if m:
            names.append(m.group(1))
    return names


def table_rows(text):
    """(name, link_target_or_None) for each numbered row of the summary table."""
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cell = line.split("|")[2]
        linked = re.search(r"\[`([^`]+)`\]\(([^)]+)\)", cell)
        if linked:
            rows.append((linked.group(1), linked.group(2)))
            continue
        bare = re.search(r"`([^`]+)`", cell)
        if bare:
            rows.append((bare.group(1), None))
    return rows


def main():
    with open(CARGO, encoding="utf-8") as fh:
        cargo = fh.read()
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()

    declared = declared_binaries(cargo)
    rows = table_rows(doc)
    listed = [name for name, _ in rows]

    problems = []

    missing = [b for b in declared if b not in listed]
    if missing:
        problems.append(
            "DECLARED IN Cargo.toml BUT NOT IN THE TABLE\n"
            "    (the page claims to list every binary; these it does not)\n"
            + "".join("    %s\n" % b for b in missing)
        )

    phantom = [b for b in listed if b not in declared]
    if phantom:
        problems.append(
            "IN THE TABLE BUT NOT DECLARED IN Cargo.toml\n"
            "    (documenting a binary nobody can build)\n"
            + "".join("    %s\n" % b for b in phantom)
        )

    duplicated = sorted({b for b in listed if listed.count(b) > 1})
    if duplicated:
        problems.append(
            "LISTED MORE THAN ONCE\n" + "".join("    %s\n" % b for b in duplicated)
        )

    dangling = []
    for name, target in rows:
        if target is None:
            continue
        path = os.path.join(os.path.dirname(DOC), target.replace("/", os.sep))
        if not os.path.exists(path):
            dangling.append("%s -> %s" % (name, target))
    if dangling:
        problems.append(
            "LINKED TO A PAGE THAT DOES NOT EXIST\n"
            "    (omit the link rather than point at a missing file)\n"
            + "".join("    %s\n" % d for d in dangling)
        )

    stated = re.search(r"\*\*Total binaries:\s*(\d+)\*\*", doc)
    if not stated:
        problems.append("NO 'Total binaries: N' HEADER FOUND\n")
    elif int(stated.group(1)) != len(declared):
        problems.append(
            "THE STATED TOTAL IS WRONG\n"
            "    header says %s, Cargo.toml declares %d\n"
            % (stated.group(1), len(declared))
        )

    if problems:
        sys.stderr.write("\n".join(problems))
        sys.stderr.write(
            "\ndocs/BINARIES.md calls itself the authoritative inventory of every\n"
            "binary. Either it is, or the claim comes out.\n"
        )
        return 1

    print(
        "OK - %d binaries, all declared, all listed, every link resolves."
        % len(declared)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
