#!/usr/bin/env python3
"""The website's figures must match the tree, the same way the README's do.

WHY A SECOND FRONT-END INSTEAD OF A SECOND CHECKER
--------------------------------------------------
`check_readme_numbers.py` (V373) measures the tree and checks `README.md`
against it. The website states the SAME facts and nothing checked it: on
2026-10-08 its hero badges carried the identical stale set the README did --
369 source files, 61 feature flags, ~520K lines.

So this imports `measured()` from that file rather than counting anything
itself. One measurement, two documents. The alternative -- a second script with
its own counting -- is the shape that let the RUSTSEC ignore list disagree with
itself across three files, and the shape that let the README's own badge
disagree with its own prose.

WHAT IT CHECKS, AND WHAT IT DOES NOT
------------------------------------
The two places a visitor reads a number as a statement about the project
today, and the two a plain text sweep walks straight past:

* bare digits inside ``<span class="num">N</span><span class="label">…</span>``
* ``<meta name="description">`` and ``og:description`` -- what Google and
  LinkedIn show, and where ``~520K LoC`` survived the first sweep

**Prose is deliberately out of scope**, and the body says why at the point it
would otherwise be checked: the site states current and historical figures in
the same sentences with nothing distinguishing them, and two heuristics were
tried and measured before concluding that none can.

The TEST COUNT is not verified against the suite (that needs a build), but
every place stating it must agree with every other place -- which is the
failure that actually happens.

USAGE
-----
    python scripts/check_website_numbers.py ../ai_assistant-website
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_readme_numbers import LOC_TOLERANCE, measured  # noqa: E402

# label on the badge -> key in measured()
BADGE_KEYS = {
    "source files": "source_files",
    "modulos fuente": "source_files",
    "módulos fuente": "source_files",
    "feature flags": "features",
    "lines of rust": "loc",
    "lineas de codigo": "loc",
    "líneas de código": "loc",
}

BADGE_RE = re.compile(
    r'<span class="num">([^<]+)</span>\s*<span class="label">([^<]+)</span>'
)
META_RE = re.compile(r'<meta[^>]+(?:name|property)="[^"]*description"[^>]+content="([^"]*)"')

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def as_number(raw: str) -> int | None:
    """`558K` -> 558000, `8,991` -> 8991, `~520K LoC` -> 520000."""
    m = re.search(r"(\d[\d,.]*)\s*([Kk])?", raw)
    if not m:
        return None
    n = int(m.group(1).replace(",", "").replace(".", ""))
    return n * 1000 if m.group(2) else n


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/check_website_numbers.py <website-dir>")
        return 1
    site = pathlib.Path(sys.argv[1])
    pages = sorted(p for p in site.glob("*.html") if "backup" not in p.name)
    if len(pages) < 5:
        print(f"ERROR: only {len(pages)} .html pages under {site} -- wrong directory?")
        return 1

    real = measured()
    problems: list[str] = []
    checked = 0
    test_counts: dict[str, set[str]] = {}

    for page in pages:
        # A comment is shown to nobody. One of the matches this removes is a
        # note I wrote myself in og-image-source.html recording what the OLD
        # numbers were -- a gate that fails on its own explanation is a gate
        # people switch off.
        text = COMMENT_RE.sub(" ", page.read_text(encoding="utf-8", errors="replace"))

        # 1. Hero badges: digits in one element, meaning in the next.
        for m in BADGE_RE.finditer(text):
            label = re.sub(r"\s+", " ", m.group(2)).strip().lower()
            key = BADGE_KEYS.get(label)
            if label in ("tests", "tests unitarios"):
                test_counts.setdefault(page.name, set()).add(
                    str(as_number(m.group(1)))
                )
                continue
            if not key:
                continue
            got, want = as_number(m.group(1)), real[key]
            checked += 1
            if got is None:
                continue
            if key == "loc":
                if abs(got - want) / want > LOC_TOLERANCE:
                    problems.append(
                        f"{page.name}: badge says {m.group(1)} lines, the tree has {want:,}"
                    )
            elif got != want:
                problems.append(
                    f"{page.name}: badge says {got:,} {label}, the tree has {want:,}"
                )

        # 2. The text Google and LinkedIn show.
        for m in META_RE.finditer(text):
            for num, unit in re.findall(
                r"(\d[\d,.]*\s*[Kk]?)\s*(LoC|LOC|lines of rust|feature flags|source files)",
                m.group(1),
                re.I,
            ):
                key = BADGE_KEYS.get(unit.lower(), "loc" if "loc" in unit.lower() else None)
                if not key:
                    continue
                got, want = as_number(num), real[key]
                checked += 1
                if got is None:
                    continue
                if key == "loc":
                    if abs(got - want) / want > LOC_TOLERANCE:
                        problems.append(
                            f"{page.name}: <meta description> says {num} {unit}, "
                            f"the tree has {want:,} -- this is the text search engines show"
                        )
                elif got != want:
                    problems.append(
                        f"{page.name}: <meta description> says {got:,} {unit}, "
                        f"the tree has {want:,} -- this is the text search engines show"
                    )

        # 3. PROSE IS NOT CHECKED, and this is the gate's main limitation.
        #
        # The site states current figures and historical ones in the same
        # prose with nothing distinguishing them: "5350 tests, 285 source
        # files" is a correct record of v12 and reads identically to a claim
        # about today. Two heuristics were tried and measured -- a 260-char
        # window round the figure (2 false positives left) and the enclosing
        # line (4) -- and neither separates them, because the MARKUP does not
        # separate them either.
        #
        # So prose is out of scope rather than approximated. The two above ARE
        # checked, and they are where a visitor reads a number as a statement
        # about the project today -- and where it drifted.
        # Making prose checkable means marking the historical blocks in the
        # HTML (about 100 lines across two pages); that is written up in N142
        # rather than guessed at here.

        # The test count: its VALUE is not verified here (that needs the suite
        # to run), but every place stating it must agree with every other --
        # which is the failure that actually happens. Badges above, and the
        # <meta> description here. Prose is out of scope for the reason given.
        for m in META_RE.finditer(text):
            for num in re.findall(r"\b(\d[\d,.]{3,})\s*tests?\b", m.group(1), re.I):
                test_counts.setdefault(page.name, set()).add(str(as_number(num)))

    # Agreement across the WHOLE site, not per page: two pages quoting different
    # totals is the same defect as one page doing it.
    all_counts = set().union(*test_counts.values()) if test_counts else set()
    all_counts.discard("None")
    if len(all_counts) > 1:
        where = ", ".join(f"{p} ({'/'.join(sorted(v))})" for p, v in sorted(test_counts.items()))
        problems.append(
            "the site states more than one test count: "
            + ", ".join(f"{int(c):,}" for c in sorted(all_counts, key=int))
            + f". Not verified against the suite here, but they must agree. Seen in: {where}"
        )

    print(f"\npaginas revisadas : {len(pages)}")
    print(f"cifras comprobadas: {checked}")
    print(f"discrepancias     : {len(problems)}")

    if problems:
        print("\nFALLO: la web dice cifras que el arbol no tiene:\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nMedido del arbol: "
            f"{real['source_files']:,} ficheros, {real['loc']:,} lineas, "
            f"{real['features']} features."
        )
        return 1
    print(
        f"\nOK - la web coincide con el arbol: {real['source_files']:,} ficheros, "
        f"~{real['loc'] // 1000}K lineas, {real['features']} features."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
