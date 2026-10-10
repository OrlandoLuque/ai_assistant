#!/usr/bin/env python3
"""Check that the numbers on the README's front page are still true.

WHY THIS GATE EXISTS
--------------------
Measured on 2026-10-08, the front page said:

* `tests-6095` when the suite had **8,990**
* `~423K LOC` when `src/` held **558K**
* `61 feature flags` when `Cargo.toml` declared **98**
* `369 source files` when there were **574**
* and, worst of the five, **"Fully implemented - zero stubs or TODOs"** - which was not
  merely stale but false: capability built and never connected is this project's recurring
  defect class, and two documented sweeps went hunting for exactly that.

Four of those understated the work and one overstated it. Both directions are the same
defect: the README is the first thing anyone reads, and nothing checked it.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
Everything here is cheap and exact: count files, read `Cargo.toml`, grep the workflows.

**The test count is NOT checked, and saying so is the point.** Getting it right means
running the suite (~36 s plus a build), and a gate that is slow gets skipped. Counting
`#[test]` attributes instead would be a different number - it misses parameterised cases
and includes ignored ones - and a gate that checks an approximation while claiming to check
the real thing is worse than no gate.

That matters because of the precedent: `docs/README.md` once said "all three run in CI"
while five did, and the fix was a checker that compares the *stated total* against the gates
that actually run. A gate whose "OK" is narrower than the question it appears to answer is
how that survives. So this one names, in its success message, exactly which numbers it
verified - and the test badge is updated by hand, with the suite's own output as the source.

USAGE
-----
    python scripts/check_readme_numbers.py             # exit 1 on any drift
    python scripts/check_readme_numbers.py --show      # print what it measured, exit 0
    python scripts/check_readme_numbers.py --self-test # prove it fails on bad input
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# LOC is allowed to drift a little: every commit moves it, and failing the build over
# three hundred lines would train everyone to edit the number without reading it. One
# per cent of half a million lines is ~5.6k, which is a real change and not noise.
LOC_TOLERANCE = 0.01


def measured() -> dict[str, int]:
    """The facts, counted from the tree."""
    rs = list((REPO / "src").rglob("*.rs"))
    loc = sum(
        # Bytes, not `str`, and errors ignored: a single non-UTF-8 byte in one file must
        # not take down a gate that is only counting newlines.
        f.read_bytes().count(b"\n")
        for f in rs
    )
    with (REPO / "Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)

    # DISTINCT checkers, not invocations. The first version counted matches and
    # reported 13 against the 11 that `check_checkers_documented.py` knows about:
    # `check_rustsec_ignores.py` and `check_duplicate_types.py` are each invoked
    # twice (once for `--self-test`, once for the ratchet, and across two
    # workflows). Two gates disagreeing about one number is the exact thing this
    # file exists to prevent, so it has to agree with the gate that owns the list.
    gate_scripts: set[str] = set()
    for workflow in (REPO / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        gate_scripts.update(re.findall(r"python3?\s+(scripts/check_\w+\.py)", text))

    return {
        "source_files": len(rs),
        "loc": loc,
        "features": len(manifest.get("features", {})),
        # No binary count: the README does not state one, and measuring a number
        # nobody checks is the loose end this project keeps finding elsewhere.
        # `check_binaries_documented.py` already holds docs/BINARIES.md to account.
        "ci_gates": len(gate_scripts),
        # Same measurement, second place it is stated. Not redundant: the badge
        # and the prose are two independent copies of one fact, and they drifted
        # apart the first time the number moved.
        "ci_gates_prose": len(gate_scripts),
    }


def stated(text: str) -> dict[str, int | None]:
    """The claims, parsed from the README. `None` means the claim is absent."""

    def one(pattern: str, scale: int = 1) -> int | None:
        m = re.search(pattern, text)
        if not m:
            return None
        return int(m.group(1).replace(",", "").replace(".", "")) * scale

    return {
        "source_files": one(r"\*\*(\d[\d,]*) source files\*\*"),
        # `558K` in the badge and in prose. Stored in thousands.
        "loc": one(r"LOC-(\d[\d,]*)K-", scale=1000),
        "features": one(r"\*\*(\d[\d,]*) feature flags\*\*"),
        "ci_gates": one(r"CI%20gates-(\d+)-"),
        # The SAME number in prose. The first version of this gate read only the
        # badge, and the badge went to 12 while the sentence below it still said
        # eleven -- a gate reporting OK about one of two places that state the
        # same fact. Same shape as the RUSTSEC ignore list living in three files
        # with two checkers: the narrower "OK" was true and smaller than the
        # question. Both are parsed, and they must agree with each other.
        "ci_gates_prose": one(r"\*\*(\d+) automated gates\*\*"),
    }


# Claims of completeness, refused by name. Each is a regex over the whole README.
#
# No escape hatch for quoting one in order to disown it. The first version had a
# `(?<!not ")` lookbehind for exactly that and it failed on the README's capital N
# -- so the prose was reworded instead. A blunt check that cannot be fooled beats a
# clever one that can, and the cost is a sentence that says the same thing without
# the words.
FORBIDDEN = [
    (
        r"zero stubs",
        "capability built and never connected is this project's recurring defect class; "
        "two documented sweeps went looking for it and found it",
    ),
    (
        r"[Ff]ully implemented",
        "unfalsifiable as written, and the honest version is docs/CAPABILITIES.md, "
        "which marks each capability hecho / parcial / no",
    ),
    (
        r"[Pp]roduction[- ]ready",
        "it has never run in production and has no external users",
    ),
]


def problems_for(text: str, real: dict[str, int]) -> list[str]:
    """Every way `text` fails to describe `real`. Empty list means the README is true."""
    claim = stated(text)
    problems: list[str] = []

    for key in ("source_files", "features", "ci_gates", "ci_gates_prose"):
        want, got = real[key], claim[key]
        if got is None:
            problems.append(
                f"the README no longer states {key}. It is checked here BECAUSE it drifts; "
                f"removing the claim removes the check. Current value: {want:,}"
            )
        elif got != want:
            problems.append(f"{key}: README says {got:,}, the tree has {want:,}")

    want_loc, got_loc = real["loc"], claim["loc"]
    if got_loc is None:
        problems.append(f"the README no longer states a LOC figure. Current: {want_loc:,}")
    else:
        drift = abs(got_loc - want_loc) / want_loc
        if drift > LOC_TOLERANCE:
            problems.append(
                f"loc: README says ~{got_loc:,}, the tree has {want_loc:,} "
                f"({drift:.1%} off, tolerance {LOC_TOLERANCE:.0%})"
            )

    # The test count is NOT verified against the suite (see this file's header).
    # But it is stated in FOUR places -- badge, status block, verification list,
    # comparison table -- and the failure that actually happens is updating one
    # and forgetting the rest. That needs no suite run: the four must agree with
    # EACH OTHER. Same shape as the gate count above, which drifted between its
    # badge and its sentence the first time it moved.
    counts = {m.group(1).replace(",", "") for m in re.finditer(r"tests-(\d+)-", text)}
    counts |= {m.group(1).replace(",", "") for m in re.finditer(r"\*\*(\d[\d,]*) tests\*\*", text)}
    counts |= {m.group(1).replace(",", "") for m in re.finditer(r"\| Tests \| (\d[\d,]*) \|", text)}
    if len(counts) > 1:
        problems.append(
            "the README states more than one test count: "
            + ", ".join(f"{int(c):,}" for c in sorted(counts, key=int))
            + ". The value is not checked here (it needs the suite to run), but the "
            "four places that state it must agree with each other."
        )

    # A number is only reproducible if the command printed beside it is the one
    # that produces it. V387 found the README offering `cargo test --features
    # full --lib` next to "8,991 tests" -- and that command yields 7,235,
    # because the figure comes from the much wider FEATURES_STD set CI runs.
    # A reader who typed it would conclude the claim was inflated by 1,756,
    # when the real number under the right command is HIGHER. The count itself
    # cannot be checked here, but WHICH COMMAND is named can be.
    problems.extend(_command_problems(text))

    # A claim of completeness is the one that was not merely stale but FALSE, and the
    # reason this file exists at all.
    for forbidden, why in FORBIDDEN:
        if re.search(forbidden, text):
            problems.append(f'the README claims "{forbidden}" again -- {why}')

    return problems


# `cargo test ... --features <X>`. `--features` may be followed by a quoted
# shell variable, a quoted list, or a bare word, so all three are accepted.
_CARGO_TEST_FEATURES = re.compile(
    r"cargo test[^`\n]*?--features\s+(\"[^\"]+\"|'[^']+'|\S+)"
)
# `FEATURES_STD: "full,autonomous,..."` in the workflow's `env:` block.
_CI_ENV_FEATURES = re.compile(r"^\s*(FEATURES_\w+)\s*:\s*\"([^\"]+)\"", re.M)
CI_WORKFLOW = ".github/workflows/ci.yml"


def _resolve(raw: str, env: dict[str, str]) -> frozenset[str] | None:
    """A `--features` argument as a SET of feature names, or None if unresolvable.

    Compared as sets rather than strings for two reasons. The README spells the
    list out — a reader cannot type `$FEATURES_STD` — while the workflow uses
    the variable, so a textual comparison could never match even when the two
    are the same set. And order carries no meaning to cargo, so two spellings
    of one set must not read as two sets.
    """
    raw = raw.strip("\"'")
    if "${{" in raw:
        # A matrix expansion. Its value is only known per-job, so there is
        # nothing to compare against and pretending otherwise would be a
        # guess dressed as a check.
        return None
    for name, value in env.items():
        raw = raw.replace(f"${{{name}}}", value).replace(f"${name}", value)
    if "$" in raw:
        return None
    return frozenset(f.strip() for f in raw.split(",") if f.strip())


def _feature_sets(text: str, env: dict[str, str]) -> set[frozenset[str]]:
    out = set()
    for m in _CARGO_TEST_FEATURES.finditer(text):
        resolved = _resolve(m.group(1), env)
        if resolved:
            out.add(resolved)
    return out


def _command_problems(text: str, ci_text: str | None = None) -> list[str]:
    if ci_text is None:
        try:
            ci_text = open(CI_WORKFLOW, encoding="utf-8").read()
        except OSError:
            return [f"no se puede leer {CI_WORKFLOW} para comparar los comandos"]

    env = {m.group(1): m.group(2) for m in _CI_ENV_FEATURES.finditer(ci_text)}
    readme_sets = _feature_sets(text, env)
    if not readme_sets:
        # Not a failure. The README is allowed not to name a command; it is
        # naming the WRONG one that misleads.
        return []

    ci_sets = _feature_sets(ci_text, env)
    if not ci_sets:
        return [
            f"{CI_WORKFLOW} no contiene ningun `cargo test --features` resoluble "
            "-- si el workflow se reescribe, esta comprobacion hay que actualizarla"
        ]

    out: list[str] = []
    for named in sorted(readme_sets - ci_sets, key=lambda s: sorted(s)):
        shown = ",".join(sorted(named))
        if len(shown) > 90:
            shown = shown[:87] + "..."
        out.append(
            f"el README manda correr los tests con {len(named)} features "
            f"({shown}) y ningun trabajo de {CI_WORKFLOW} usa ese conjunto. "
            "Los conjuntos que CI prueba tienen "
            + ", ".join(str(len(s)) for s in sorted(ci_sets, key=len))
            + " features. Una cifra junto a un comando que no la produce es peor "
            "que una cifra sin comando: invita a reproducirla y a concluir que "
            "esta mal, y una bateria que solo corre quien lee el README no la "
            "cubre ninguna puerta."
        )
    return out


# Every case is a README that is WRONG in one specific way, plus one that is right.
# A gate nobody has seen fail is a gate nobody knows the polarity of: this file's
# first version passed while the README still carried a false completeness claim,
# because the pattern meant to refuse it had a lookbehind that never matched.
SELF_TEST_REAL = {
    "source_files": 574,
    "loc": 558_000,
    "features": 98,
    "ci_gates": 12,
    "ci_gates_prose": 12,
}
TRUE_README = (
    "![](tests-8990-green) ![](LOC-558K-blue) ![](CI%20gates-12-green)\n"
    "**574 source files**, 558K lines. **98 feature flags**.\n"
    "There are **12 automated gates** in CI, and **8,990 tests**.\n"
)
SELF_TEST_CASES = [
    ("a README that matches the tree", TRUE_README, 0),
    ("a stale file count", TRUE_README.replace("**574 source", "**369 source"), 1),
    ("a stale feature count", TRUE_README.replace("**98 feature", "**61 feature"), 1),
    # Both copies of the gate count moved, so both are wrong: 2 problems.
    ("a stale gate count in badge and prose",
     TRUE_README.replace("gates-12-", "gates-11-").replace("**12 automated", "**11 automated"), 2),
    # The one that actually happened: the badge was updated, the sentence was not.
    ("the badge updated but not the prose", TRUE_README.replace("**12 automated", "**11 automated"), 1),
    # The test count: the VALUE is not checked, but the places stating it must agree.
    ("test count updated in the badge only",
     TRUE_README.replace("tests-8990", "tests-8991"), 1),
    ("test count consistent but not verified against the suite",
     TRUE_README.replace("tests-8990", "tests-1").replace("**8,990 tests**", "**1 tests**"), 0),
    ("LOC off by 24%", TRUE_README.replace("LOC-558K", "LOC-423K"), 1),
    # Half a per cent: a real commit moves the number this much, and failing the
    # build over it would train everyone to edit the figure without reading it.
    ("LOC off by 0.5% (inside tolerance)", TRUE_README.replace("558K", "555K"), 0),
    ("the file count deleted rather than fixed", TRUE_README.replace("**574 source files**", "lots of code"), 1),
    ("a completeness claim", TRUE_README + "\nFully implemented - zero stubs or TODOs.\n", 2),
    ("a production-ready claim", TRUE_README + "\nProduction ready.\n", 1),
    # The exact shape that defeated the first version of the gate.
    ('"zero stubs" quoted in order to be disowned', TRUE_README + '\nNot "zero stubs".\n', 1),
    ("every number wrong at once", "nothing here\n", 5),
]

# The command check gets its own cases because it needs a ci.yml to compare
# against, which `problems_for` reads from disk. TRUE_README names no command
# at all, so the cases above are unaffected by it either way.
FAKE_CI = (
    '  FEATURES_STD: "full,autonomous,scheduler"\n'
    '  FEATURES_MIN: "tools,security"\n'
    'run: cargo test --features "$FEATURES_STD" --lib\n'
    'run: cargo test --no-default-features --features "$FEATURES_MIN"\n'
    'run: cargo test --features "${{ matrix.features }}" --lib\n'
)
COMMAND_SELF_TEST = [
    # The bug this check was written for, in the exact shape it was found.
    ("the command names a feature set CI never tests",
     "**8,991 tests** (`cargo test --features full --lib`)", 1),
    # The second instance, found by the check on its first real run: the README's
    # own "Run tests" block named a hand-written six-feature subset, so anyone
    # following the README ran a narrower suite than CI and got a third number.
    ("a hand-written subset in a Run tests block",
     "# Run tests\ncargo test --features \"full,autonomous\" --lib", 1),
    ("the command names the set CI runs",
     '**9,012 tests** (`cargo test --features "$FEATURES_STD" --lib`)', 0),
    ("quoting differences do not matter",
     "**9,012 tests** (`cargo test --features $FEATURES_STD --lib`)", 0),
    # The reason sets are compared and not strings: a reader cannot type
    # `$FEATURES_STD`, so the README spells the list out — and cargo does not
    # care about the order, so neither may this check.
    ("the list spelled out, in another order, is the same set",
     "`cargo test --features scheduler,full,autonomous --lib`", 0),
    ("the minimum set is also one CI runs",
     '`cargo test --no-default-features --features "$FEATURES_MIN"`', 0),
    # Naming no command is allowed: it is naming the WRONG one that misleads.
    ("no command named at all", "**9,012 tests**, measured", 0),
    ("two commands, one of them wrong",
     '`cargo test --features "$FEATURES_STD" --lib` and `cargo test --features gui --lib`', 1),
    # A matrix expansion has no value until the job runs. Skipped, not guessed:
    # resolving it to the literal string would make every README command look
    # wrong against it, which is a check that fails for the wrong reason.
    ("a matrix expansion is skipped, not guessed",
     '`cargo test --features "${{ matrix.features }}" --lib`', 0),
]


def self_test() -> int:
    failures = 0
    for name, text, want in SELF_TEST_CASES:
        got = len(problems_for(text, SELF_TEST_REAL))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got} problem(s), expected {want}")
    for name, text, want in COMMAND_SELF_TEST:
        got = len(_command_problems(text, FAKE_CI))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got} problem(s), expected {want}")
    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures} of {len(SELF_TEST_CASES) + len(COMMAND_SELF_TEST)} cases")
        return 1
    print(f"SELF-TEST OK: {len(SELF_TEST_CASES) + len(COMMAND_SELF_TEST)} cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show", action="store_true", help="print and exit 0")
    parser.add_argument(
        "--self-test", action="store_true", help="check the gate against crafted READMEs"
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not README.is_file():
        print(f"{README} does not exist")
        return 1

    text = README.read_text(encoding="utf-8")
    real = measured()
    claim = stated(text)

    if args.show:
        print("measured from the tree:")
        for k, v in real.items():
            print(f"  {k:<14} {v:>8,}")
        print("\nstated on the README:")
        for k, v in claim.items():
            print(f"  {k:<14} {'(absent)' if v is None else f'{v:>8,}'}")
        return 0

    problems = problems_for(text, real)

    if problems:
        print(f"FAIL - {len(problems)} README claim(s) no longer true:\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nThe README is the first thing anyone reads about this project. A number "
            "there\nthat nobody checks drifts in both directions: four of the five this "
            "gate was\nborn from UNDERSTATED the work, and one overstated it."
        )
        return 1

    print(
        f"OK - README states {claim['source_files']:,} source files, "
        f"~{real['loc'] // 1000}K lines, {claim['features']} features and "
        f"{claim['ci_gates']} CI gates (badge and prose agree), and all five match the tree.\n"
        "Also checked: every `cargo test --features` the README shows names a set that a job "
        f"in {CI_WORKFLOW} really runs, compared as sets so order and `$VAR` spelling do not "
        "matter.\n"
        "NOT checked here (by design, see this file's header): the test count itself, which "
        "needs the suite to run. Its four statements must agree with each other, and the "
        "command beside it must be CI's -- both of which ARE checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
