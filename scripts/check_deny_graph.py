#!/usr/bin/env python3
"""`deny.toml` must check the WHOLE feature graph, not the default one.

Measured 2026-10-10 (V385, N146) with `cargo tree -e no-dev --target all`:

    default feature graph ....  299 crates
    every feature ............  955 crates
    never checked ............  656 crates   (69 % of the tree)

`[graph] all-features = false` was the setting, so `cargo deny check licenses
bans sources advisories` only ever walked those 299. With 95 feature flags,
nearly every dependency sits behind one of them, and deny.toml's own header
says what it is for: "a single GPL transitive contaminates a future commercial
dual-license". A copyleft crate behind any optional feature was invisible.

This gate exists because of HOW that failed. `cargo deny` printed

    advisories ok, bans ok, licenses ok, sources ok

in both configurations. The narrow one is not a weaker green, it is the SAME
green. Flip the setting back and 656 crates stop being checked with no signal
anywhere — not a warning, not an exit code, not a diff in the summary line.
Nothing but this script would notice.

Also checks `no-default-features`: setting it to `true` alongside
`all-features = true` is harmless (all-features wins), but on its own it would
shrink the graph further, so it is pinned too and the pairing is spelled out
rather than left to be rediscovered.

Run: python scripts/check_deny_graph.py
Self-test: python scripts/check_deny_graph.py --self-test
"""
from __future__ import annotations

import re
import sys

CONFIG = "deny.toml"

# The setting, and why a plain `in` test is not enough: the key appears in
# comments in this very file (including the explanation above the setting), so
# the check has to anchor to a real assignment at the start of a line.
ALL_FEATURES = re.compile(r"^\s*all-features\s*=\s*(true|false)\s*$", re.M)
NO_DEFAULT = re.compile(r"^\s*no-default-features\s*=\s*(true|false)\s*$", re.M)
# The reason has to travel with the setting. A bare `all-features = true` is
# one cleanup away from being "simplified" by someone who has no way to know
# what it buys.
REASON = re.compile(r"\bN146\b")


def check(text: str) -> list[str]:
    problems: list[str] = []

    found = ALL_FEATURES.findall(text)
    if not found:
        problems.append(
            f"{CONFIG} no declara `all-features` en [graph]. Sin esa clave "
            "cargo-deny recorre solo el grafo por defecto (299 de 955 crates "
            "medidos en V385) y aun asi imprime «licenses ok»."
        )
    elif len(found) > 1:
        problems.append(
            f"{CONFIG} declara `all-features` {len(found)} veces. TOML se "
            "queda con la ultima, asi que leer la primera engana."
        )
    elif found[0] != "true":
        problems.append(
            f"{CONFIG} tiene `all-features = {found[0]}`. Eso deja 656 de 955 "
            "crates sin comprobar licencias, bans ni sources — el 69 % del "
            "arbol — y cargo-deny sigue diciendo «ok» igual que con true. "
            "Si hay un motivo para estrecharlo, tiene que estar escrito y "
            "este script tiene que cambiar con el."
        )

    nd = NO_DEFAULT.findall(text)
    if nd and nd[-1] != "false":
        problems.append(
            f"{CONFIG} tiene `no-default-features = {nd[-1]}`. Junto a "
            "`all-features = true` no hace dano, pero si alguien quita "
            "all-features queda el grafo MAS estrecho todavia."
        )

    if not REASON.search(text):
        problems.append(
            f"{CONFIG} ya no menciona N146. El ajuste `all-features` sin el "
            "motivo al lado se lee como una linea de mas; la medicion que lo "
            "justifica tiene que viajar con el."
        )

    return problems


SELF_TEST = [
    # (name, text, expect_failure)
    ("el fichero real", None, False),
    (
        "all-features = false (el bug de V385)",
        "[graph]\n# N146\nall-features = false\nno-default-features = false\n",
        True,
    ),
    (
        "sin la clave",
        "[graph]\n# N146\nno-default-features = false\n",
        True,
    ),
    (
        "correcto y con el motivo",
        "[graph]\n# ver N146\nall-features = true\nno-default-features = false\n",
        False,
    ),
    (
        "correcto pero sin el motivo escrito",
        "[graph]\nall-features = true\nno-default-features = false\n",
        True,
    ),
    (
        "no-default-features = true por su cuenta",
        "[graph]\n# N146\nall-features = true\nno-default-features = true\n",
        True,
    ),
    (
        "declarado dos veces, la ultima gana",
        "[graph]\n# N146\nall-features = true\nall-features = false\n",
        True,
    ),
    # The check must not be satisfied by the word appearing in prose. This is
    # the failure mode the anchored regex exists for: the comment block above
    # the setting in deny.toml contains the string "all-features = false"
    # while describing what went wrong.
    (
        "solo en un comentario que habla del ajuste",
        "[graph]\n# N146: antes decia all-features = false\n",
        True,
    ),
    (
        "indentado dentro de la tabla",
        "[graph]\n  # N146\n  all-features = true\n",
        False,
    ),
]


def self_test() -> int:
    real = open(CONFIG, encoding="utf-8").read()
    bad = 0
    for name, text, expect_failure in SELF_TEST:
        got = check(real if text is None else text)
        failed = bool(got)
        mark = "ok " if failed == expect_failure else "MAL"
        if failed != expect_failure:
            bad += 1
        want = "debe fallar" if expect_failure else "debe pasar"
        print(f"  [{mark}] {name} ({want}) -> {len(got)} problema(s)")
        if failed != expect_failure and got:
            for p in got:
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

    problems = check(text)
    if problems:
        print(f"FALLO: {len(problems)} problema(s) en {CONFIG}:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"OK: {CONFIG} comprueba el grafo completo de features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
