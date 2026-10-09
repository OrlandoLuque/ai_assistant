#!/usr/bin/env python3
"""The list of automated checks in docs/README.md must be true.

That section exists because documentation drifts silently. It drifted: it said
"all three run in CI" until V307 while five did, and "all six" until V340 while
eight did. A page whose subject is enforcement, unenforced.

Three sources must agree:

* the ``scripts/check_*.py`` files on disk,
* which of them a workflow under ``.github/workflows/`` actually runs,
* the bullet list and the stated total in ``docs/README.md``.

A checker nobody runs is worse than no checker: it looks like coverage. A
checker that runs but is not listed is invisible to whoever reads the docs to
find out what is guarded. Both are failures here.
"""
from __future__ import annotations

import os
import re
import sys

DOC = "docs/README.md"
SCRIPTS = "scripts"
WORKFLOWS = ".github/workflows"

# Checkers deliberately not wired to CI, with the reason. The doc must describe
# them as manual, and the count of CI-wired ones must exclude them.
MANUAL = {
    "check_release_ready.py": "pre-release manual step, needs a human to read the output",
    # Both are run by the WEBSITE repository's CI (.github/workflows/
    # consistency.yml there), which checks out both repositories. They are
    # "manual" from here because no job in THIS repository runs them -- and
    # saying "manual" when something else runs them automatically would be
    # the narrower-than-the-question kind of true, so it is spelled out.
    "check_website_examples.py": (
        "run by ai_assistant-website's own CI, which checks out both repos; "
        "no job here runs it"
    ),
    "check_website_numbers.py": (
        "run by ai_assistant-website's own CI, which checks out both repos; "
        "no job here runs it"
    ),
}

WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    # Stopped at twelve, which is exactly how many there were. The thirteenth
    # (V385) made the fallback `str(want)` compare "thirteen" against "13" and
    # report a mismatch between a number and itself: «dice "All thirteen run
    # in CI" y son 13 (13)». Latent since the table was written: wrong answer,
    # unreadable message, and only on the commit that fixes the real thing.
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def main() -> int:
    on_disk = {
        f for f in os.listdir(SCRIPTS) if f.startswith("check_") and f.endswith(".py")
    }
    if len(on_disk) < 3:
        sys.exit(f"ERROR: solo {len(on_disk)} checkers en {SCRIPTS}/ -- el listado no funciona")

    workflow_text = ""
    for base, _dirs, files in os.walk(WORKFLOWS):
        for f in files:
            if f.endswith((".yml", ".yaml")):
                workflow_text += open(
                    os.path.join(base, f), encoding="utf-8", errors="replace"
                ).read()
    if not workflow_text:
        sys.exit(f"ERROR: no se ha leido ningun workflow de {WORKFLOWS}/")

    in_ci = {s for s in on_disk if f"{SCRIPTS}/{s}" in workflow_text}
    doc = open(DOC, encoding="utf-8", errors="replace").read()
    listed = {s for s in on_disk if f"{SCRIPTS}/{s}" in doc}

    problems: list[str] = []

    for s in sorted(on_disk - in_ci - set(MANUAL)):
        problems.append(
            f"{s} existe y ningun workflow lo ejecuta. Conectalo a CI, o "
            f"anadelo a MANUAL en este script con el motivo escrito al lado."
        )
    for s in sorted(set(MANUAL) & in_ci):
        problems.append(
            f"{s} esta en MANUAL (\"{MANUAL[s]}\") pero un workflow lo ejecuta. "
            "Quitalo de MANUAL."
        )
    for s in sorted(in_ci - listed):
        problems.append(f"{s} corre en CI y no aparece en {DOC}.")
    for s in sorted(set(MANUAL) - listed):
        problems.append(f"{s} no aparece en {DOC} (deberia, como paso manual).")
    for s in sorted(listed - on_disk):
        problems.append(f"{DOC} nombra {s}, que no existe en {SCRIPTS}/.")

    # The stated total. The sentence is "All <word> run in CI." and then names
    # the manual one as the "<word>th checker".
    want = len(in_ci)
    m = re.search(r"\bAll (\w+) run in CI\b", doc)
    if not m:
        problems.append(
            f'{DOC} ya no contiene la frase "All <numero> run in CI" -- '
            "si se reescribe, hay que actualizar este script."
        )
    elif m.group(1).lower() != WORDS.get(want, str(want)):
        problems.append(
            f'{DOC} dice "All {m.group(1)} run in CI" y son '
            f"{WORDS.get(want, want)} ({want})."
        )

    print(f"checkers en {SCRIPTS}/ : {len(on_disk)}")
    print(f"  ejecutados por CI   : {len(in_ci)}")
    print(f"  manuales declarados : {len(MANUAL)}")
    print(f"  listados en {DOC} : {len(listed)}")

    if problems:
        print(f"\nFALLO: {len(problems)} discrepancia(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
