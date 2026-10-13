#!/usr/bin/env python3
"""Every suppressed advisory carries a re-check deadline. CI enforces it.

# What was wrong, measured 2026-10-13 (V401, N149)

`deny.toml` has a written policy: every `[advisories].ignore` entry must come
with (a) why the advisory does not apply and (b) a re-check trigger, a DATE or
an upstream event. The policy was followed — every entry has both — and then
nothing enforced the dates:

    entries ....................  8
    date trigger already past ..  6
    re-check actually recorded .  1   (lru, V385)

`.github/workflows/rustsec-review-monthly.yml` exists precisely to force this.
It is not broken: it ran on the 1st of every month from June to October 2026
and opened a tracking issue each time. **All five issues are still open.** The
forcing function was built, works, and had no effect for five months, because
it writes to the Issues tab and what actually stops work is a red CI run.

Re-checking those six by hand found that FIVE of the eight written reasons were
false or named the wrong cause — including two whose re-check condition could
never come true. The worst was quick-xml: the reason blamed a Linux GUI chain
parsing "trusted, build-time wayland protocol XML", when the crate is in the
graph because of our own `tabular-polars` feature, through `object_store` with
`cloud` and `http` enabled — i.e. XML off the network. A suppression whose
written reason is false looks reviewed, so nobody re-reads it.

# What this gate checks

For each ignored advisory: there is a `Re-check by:` line, and if it names a
DATE, that date is in the future. A re-check does not add a line and leave the
old date in place — it MOVES the date forward, so the field is always a promise
about the future rather than a record of the past.

Entries whose trigger is only an upstream event are reported by name and NOT
failed: there is nothing mechanical to compare. Saying so is the point. A gate
that quietly passes on what it cannot check reads as coverage it does not have.

# This gate can go red with no code change

That is deliberate, and it is the same property a certificate expiry has: the
deadline is the mechanism. To avoid ambushing anyone, an entry within
`WARN_DAYS` of its deadline prints a warning and still passes; only a date that
has actually passed fails. The remedy is never to push the date out on its own
— it is to re-measure, correct the reason if the measurement contradicts it,
and then set a new date.

Run: python scripts/check_recheck_dates.py
Self-test: python scripts/check_recheck_dates.py --self-test
"""
from __future__ import annotations

import datetime as dt
import re
import sys

CONFIG = "deny.toml"
WARN_DAYS = 21

# The quoted entries inside `ignore = [ ... ]`, and ONLY those. Reading the
# comments as entries is a mistake this repository has already made: V142's
# range scan picked up RUSTSEC-2026-0222 out of a comment that recorded its
# DELETION, so an absence was reported as a presence every month.
IGNORE_BLOCK = re.compile(r"^\s*ignore\s*=\s*\[(.*?)^\s*\]", re.M | re.S)
ENTRY = re.compile(r'"(RUSTSEC-\d{4}-\d{4})"')
# `Re-check by: 2026-10-01 OR when ...` / `Re-check by: when ...`
RECHECK_DATE = re.compile(r"Re-check by:\s*(\d{4})-(\d{2})-(\d{2})")
RECHECK_ANY = re.compile(r"Re-check by:", re.I)


def parse(text: str) -> list[tuple[str, str]]:
    """Pair each ignored advisory id with the comment block above it.

    The block is every line from the previous entry (or the start of the list)
    up to and including the line the id sits on, which is how the file is
    already written: reason first, trigger last, then the quoted id.
    """
    m = IGNORE_BLOCK.search(text)
    if not m:
        return []
    body = m.group(1)
    out: list[tuple[str, str]] = []
    start = 0
    for em in ENTRY.finditer(body):
        out.append((em.group(1), body[start:em.end()]))
        start = em.end()
    return out


def check(text: str, today: dt.date) -> tuple[list[str], list[str]]:
    """Return (problems, notes)."""
    problems: list[str] = []
    notes: list[str] = []

    entries = parse(text)
    if not entries:
        return (
            [
                f"{CONFIG}: no encuentro la lista `ignore = [...]` de "
                "[advisories], o esta vacia. Si de verdad no hay supresiones, "
                "esta puerta sobra; si la hay, el parser dejo de encontrarla y "
                "estaria pasando sin comprobar nada."
            ],
            notes,
        )

    for advisory, block in entries:
        dates = RECHECK_DATE.findall(block)
        if not dates:
            if RECHECK_ANY.search(block):
                notes.append(
                    f"{advisory}: el disparador es solo un evento upstream, "
                    "sin fecha. Esta puerta NO puede comprobarlo."
                )
            else:
                problems.append(
                    f"{advisory} no lleva linea `Re-check by:`. La politica "
                    f"escrita en {CONFIG} exige (a) por que no aplica y (b) un "
                    "disparador de re-chequeo. Sin el, la entrada se queda para "
                    "siempre y parece revisada."
                )
            continue

        # If an entry carries several dates, the LAST one is the live deadline:
        # a renewal appends its new date below the history of what was measured.
        y, mth, d = (int(x) for x in dates[-1])
        try:
            deadline = dt.date(y, mth, d)
        except ValueError:
            problems.append(
                f"{advisory}: la fecha `Re-check by: {y:04d}-{mth:02d}-{d:02d}` "
                "no existe en el calendario."
            )
            continue

        if deadline < today:
            overdue = (today - deadline).days
            problems.append(
                f"{advisory}: el re-chequeo venció el {deadline} "
                f"(hace {overdue} dia(s)). NO lo arregles moviendo la fecha: "
                "vuelve a medir con `cargo tree --all-features -e features -i "
                "<crate>` quien mete el crate DE VERDAD, corrige la razon si la "
                "medicion la contradice, y entonces pon fecha nueva. "
                "Runbook: docs/runbooks/rustsec-handling.md."
            )
        elif (deadline - today).days <= WARN_DAYS:
            notes.append(
                f"{advisory}: vence el {deadline}, dentro de "
                f"{(deadline - today).days} dia(s)."
            )

    return problems, notes


# ---------------------------------------------------------------- self-test

_HOY = dt.date(2026, 10, 13)


def _block(*lines: str) -> str:
    return "[advisories]\nignore = [\n" + "\n".join(lines) + "\n]\n"


SELF_TEST = [
    # (name, text, expect_failure)
    ("el fichero real", None, False),
    (
        "fecha vencida (el bug de V401)",
        _block("    # Re-check by: 2026-10-01 OR when upstream bumps it.",
               '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "fecha futura",
        _block("    # Re-check by: 2027-04-01 OR when upstream bumps it.",
               '    "RUSTSEC-2026-0194",'),
        False,
    ),
    (
        "vence hoy mismo: todavia no ha vencido",
        _block("    # Re-check by: 2026-10-13", '    "RUSTSEC-2026-0194",'),
        False,
    ),
    (
        "vencio ayer",
        _block("    # Re-check by: 2026-10-12", '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "solo evento upstream: pasa, pero se dice",
        _block("    # Re-check by: when the egui stack moves past 0.27.",
               '    "RUSTSEC-2026-0257",'),
        False,
    ),
    (
        "sin disparador ninguno",
        _block("    # Transitiva, no nos afecta.", '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "renovada: la fecha nueva va debajo y es la que cuenta",
        _block("    # Re-check by: 2026-10-01 OR when upstream bumps it.",
               "    # V401 re-check (2026-10-13): medido, sigue afectada.",
               "    # Re-check by: 2027-01-15",
               '    "RUSTSEC-2026-0194",'),
        False,
    ),
    (
        "renovada al reves: la fecha viva es la ULTIMA, y vencio",
        _block("    # Re-check by: 2027-01-15",
               "    # (corregido) Re-check by: 2026-09-01",
               '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "dos entradas, una buena y una vencida",
        _block("    # Re-check by: 2027-04-01", '    "RUSTSEC-2026-0194",',
               "    # Re-check by: 2026-01-01", '    "RUSTSEC-2026-0195",'),
        True,
    ),
    (
        "cada entrada se juzga con SU bloque, no con el del vecino",
        _block("    # Re-check by: 2026-01-01", '    "RUSTSEC-2026-0194",',
               "    # Re-check by: 2027-04-01", '    "RUSTSEC-2026-0195",'),
        True,
    ),
    (
        "lista vacia: la puerta no puede pasar en silencio",
        "[advisories]\nignore = [\n]\n",
        True,
    ),
    (
        "sin lista: el parser no encuentra nada y debe gritar",
        "[advisories]\ndb-path = \"~/.cargo/advisory-db\"\n",
        True,
    ),
    (
        "una fecha en un comentario SUELTO no es un disparador",
        _block("    # Visto el 2026-10-13, parece inofensiva.",
               '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "fecha imposible",
        _block("    # Re-check by: 2026-02-30", '    "RUSTSEC-2026-0194",'),
        True,
    ),
    (
        "un id en un comentario que registra su BORRADO no es una entrada",
        _block("    # RUSTSEC-2026-0222 estaba aqui y se borro en V276.",
               "    # Re-check by: 2027-04-01", '    "RUSTSEC-2026-0194",'),
        False,
    ),
    # This tree is CRLF, and `open(..., encoding=...)` normalises line endings
    # on read -- so the real file arrives here as LF and these cases would
    # never exercise CRLF unless one is written with it on purpose. Worth the
    # four lines: `scripts/mutate.py`'s `--dry-run` once reported matches that
    # the real run could not apply, because its pre-flight and its rewrite read
    # the same file with different newline handling.
    (
        "fecha vencida con finales de linea CRLF",
        _block("    # Re-check by: 2026-10-01", '    "RUSTSEC-2026-0194",')
        .replace("\n", "\r\n"),
        True,
    ),
    (
        "fecha futura con finales de linea CRLF",
        _block("    # Re-check by: 2027-04-01", '    "RUSTSEC-2026-0194",')
        .replace("\n", "\r\n"),
        False,
    ),
]


def self_test() -> int:
    real = open(CONFIG, encoding="utf-8").read()
    bad = 0
    for name, text, expect_failure in SELF_TEST:
        problems, _ = check(real if text is None else text, _HOY)
        failed = bool(problems)
        if failed != expect_failure:
            bad += 1
        mark = "ok " if failed == expect_failure else "MAL"
        want = "debe fallar" if expect_failure else "debe pasar"
        print(f"  [{mark}] {name} ({want}) -> {len(problems)} problema(s)")
        if failed != expect_failure:
            for p in problems:
                print(f"         {p}")
    print()
    if bad:
        print(f"FALLO: {bad} de {len(SELF_TEST)} casos del self-test")
        return 1
    print(f"OK: {len(SELF_TEST)}/{len(SELF_TEST)} casos del self-test")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    try:
        text = open(CONFIG, encoding="utf-8").read()
    except OSError as e:
        print(f"ERROR: no se puede leer {CONFIG}: {e}")
        return 1

    problems, notes = check(text, dt.date.today())
    for n in notes:
        print(f"  aviso: {n}")
    if problems:
        print(f"\nFALLO: {len(problems)} supresion(es) sin re-chequeo al dia:")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nCada `ignore` es la afirmacion de que una advisory no nos "
            "afecta. Esa afirmacion caduca: el upstream publica el arreglo, o "
            "un cambio aqui alcanza el codigo que antes no se tocaba."
        )
        return 1

    total = len(parse(text))
    print(f"OK: las {total} supresiones de {CONFIG} tienen re-chequeo vigente")
    return 0


if __name__ == "__main__":
    sys.exit(main())
