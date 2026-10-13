#!/usr/bin/env python3
"""Where else does this tree promise to revisit something by a date?

# This is a SWEEP, not a gate, and the name says so on purpose

It is deliberately NOT called `check_*.py`: it does not run in CI, it has no
pass/fail, and `scripts/check_checkers_documented.py` must not count it among
the gates. Running it produces a list for a human to triage.

The reason is its measured noise ratio, 2026-10-13 (V402): **69 hits, of which
22 are the `deny.toml` entries already gated by `check_recheck_dates.py` and
most of the remaining 47 are RECORDS rather than promises** — CHANGELOG entries
that mention a date, table rows with a "done on" column, audit documents named
after the day they were written. A record is not a broken promise, and a gate
that fails on 45 "overdue" items where two matter trains people to pass it with
`--ignore`. So this stays a sweep until somebody finds a rule that separates a
promise from a record, which is a harder problem than it looks.

What it DID find, on its first run, is the thing that justifies keeping it:
`docs/runbooks/rustsec-handling.md` — the written policy for the suppression
rules V401 had just rewritten — still taught the superseded form, with a worked
example whose date had expired two months earlier. Nothing else in the repo
would have pointed there.

# How to read the output

Two piles are worth your attention and the rest usually is not:

* **A policy or runbook** whose text contradicts the thing it documents. Fix
  the document; that is a real defect, not a stale date.
* **A `Last reviewed:` stamp** on an operational document. Note what the stamp
  means: *somebody read this on that date*. Do NOT bump it without reading the
  document — a fresh stamp is what stops the next person re-reading it, which
  makes an unread document worse than an obviously old one. Eight of the nine
  stamps here said 2026-05-06 or 2026-05-26 and were deliberately left alone.

Run: python scripts/sweep_deadlines.py
     python scripts/sweep_deadlines.py --stamps   (only `Last reviewed:`)
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

INTENT = (
    r"(re-?check|revisit|re-?measure|re-?evaluate|expires?|deadline"
    r"|delete this|remove this|drop this|TODO|FIXME)"
)
DATE = r"(20\d\d)-(\d\d)-(\d\d)"
PAT = re.compile(rf"(?i){INTENT}[^\n]{{0,70}}?{DATE}|{DATE}[^\n]{{0,70}}?{INTENT}")
# `\**` and not `\*?`: the real format is `**Last reviewed**: 2026-05-06 (V130)`
# with TWO asterisks each side. Written with one, this found 1 of 9 stamps and
# reported it as the whole picture -- a sweep that under-reports by 8 is worse
# than no sweep, because its output looks like an answer.
STAMP = re.compile(r"\**Last reviewed\**:?\s*\**\s*(20\d\d)-(\d\d)-(\d\d)")

SKIP_DIRS = {".git", "target", "node_modules", ".venv", "__pycache__"}
EXTS = {".rs", ".py", ".toml", ".md", ".yml", ".yaml"}
# Already enforced mechanically; counted separately so the total is honest.
GATED = {"deny.toml"}
# Records by construction: a date in these is history, not a promise. Separated
# because a sweep that reports 16 overdue items of which none are real teaches
# the reader to skip its output -- which is the exact failure this whole line of
# work is about.
RECORDS = {"CHANGELOG.md"}
# Patterns of path that are dated documents or that DESCRIBE the pattern rather
# than being an instance of it. `check_recheck_dates.py` matched its own
# docstring's worked example; `rustsec-handling.md` matched the expired date it
# quotes while explaining why leaving it there was the mistake.
RECORD_PATTERNS = (
    re.compile(r"^docs/IMPROVEMENTS_V\d+\.md$"),
    re.compile(r"^docs/SECURITY_AUDIT_V\d+\.md$"),
    re.compile(r"^docs/AGENT_SYSTEM_DESIGN\.md$"),  # status table with a date column
    re.compile(r"^scripts/check_recheck_dates\.py$"),
    re.compile(r"^scripts/sweep_deadlines\.py$"),
    re.compile(r"^docs/runbooks/rustsec-handling\.md$"),
)


# The tooling that DESCRIBES these patterns, which must never be read as an
# instance of them. This bit me twice in one sitting: the promises pass matched
# `check_recheck_dates.py`'s worked example, and once that was excluded the
# stamps pass matched THIS file's own docstring quoting `2026-05-06`. Both are
# the same mistake V142 made in the monthly workflow -- reading a comment that
# documents a thing as the thing itself.
TOOLING = {"scripts/sweep_deadlines.py", "scripts/check_recheck_dates.py"}


def is_record(rel: str) -> bool:
    return rel in RECORDS or any(p.match(rel) for p in RECORD_PATTERNS)


def comment_like(line: str, ext: str) -> bool:
    s = line.lstrip()
    if ext == ".rs":
        return s.startswith(("//", "/*", "*"))
    if ext in {".py", ".toml", ".yml", ".yaml"}:
        return s.startswith("#")
    return True  # markdown is prose throughout


def files():
    for f in REPO.rglob("*"):
        if f.is_file() and f.suffix in EXTS and not any(
            p in SKIP_DIRS for p in f.parts
        ):
            yield f


def main() -> int:
    today = dt.date.today()
    stamps_only = "--stamps" in sys.argv
    promises, stamps, gated, records = [], [], 0, 0

    for f in files():
        rel = f.relative_to(REPO).as_posix()
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if rel in TOOLING:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            sm = STAMP.search(line)
            if sm:
                try:
                    stamps.append((rel, i, dt.date(*(int(x) for x in sm.groups()))))
                except ValueError:
                    pass
                continue
            if stamps_only or len(line) > 400 or not comment_like(line, f.suffix):
                continue
            m = PAT.search(line)
            if not m:
                continue
            nums = [x for x in m.groups() if x and x.isdigit()]
            if len(nums) < 3:
                continue
            try:
                when = dt.date(int(nums[0]), int(nums[1]), int(nums[2]))
            except ValueError:
                continue
            if rel in GATED:
                gated += 1
            elif is_record(rel):
                records += 1
            else:
                promises.append((rel, i, when, line.strip()[:110]))

    if not stamps_only:
        past = [p for p in promises if p[2] < today]
        print(f"plazos con fecha, fuera de lo ya cubierto : {len(promises)}")
        print(f"  con la fecha PASADA                    : {len(past)}")
        print(f"  (en deny.toml, ya con puerta           : {gated})")
        print(f"  (registros con fecha, no promesas      : {records})\n")
        for rel, i, when, line in sorted(past, key=lambda r: r[2]):
            print(f"{when}  {rel}:{i}")
            print(f"            {line}")
        print()

    print(f"sellos `Last reviewed`: {len(stamps)}")
    for rel, i, when in sorted(stamps, key=lambda r: r[2]):
        age = (today - when).days
        mark = "  <-- mas de 90 dias" if age > 90 else ""
        print(f"  {when}  ({age:4d} d)  {rel}:{i}{mark}")
    print(
        "\nUn sello viejo NO se arregla poniendo la fecha de hoy: significa que\n"
        "alguien leyo el documento ese dia. Leelo, y entonces la fecha."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
