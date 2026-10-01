#!/usr/bin/env python3
"""Mutation testing for this repository: break an invariant on purpose and check
that a test notices.

A green test suite says the tests pass. It does not say the tests *discriminate*.
The difference is what this script measures: take the smallest code change that
would break a stated invariant, apply it, and see whether anything turns red. If
nothing does, the invariant is unguarded -- and every future refactor of that line
is unprotected.

WHY THIS IS A SCRIPT AND NOT A HABIT
------------------------------------
It was a habit for three weeks, done by hand, and hand-rolling it cost real
mistakes on 2026-09-27 alone:

  1. A pattern that appeared TWICE was asserted to appear once. Applying the
     mutation to both sites at the same time reported "killed" -- because ONE of
     the two branches had a test. The other branch had none, and a column of
     decimals with a missing marker still returned a wrong AVG. Mutating per site
     is what found it. Hence `occurrences` below, and hence a verdict PER SITE.

  2. The classifier said "did not compile" for mutations that compiled fine and
     failed tests, because it tested `'error: ' in output` -- which matches
     cargo's own `error: test failed, to rerun pass --lib`. A mutation that does
     not compile proves NOTHING (the test never ran), so confusing the two
     verdicts is the difference between evidence and noise. Hence NOT_COMPILED as
     a first-class outcome, matched on `error[E` and `could not compile`.

  3. Escaped source snippets inside shell heredocs mangled backslashes more than
     once. Hence spec files on disk, read as TOML, never inline shell strings.

WHAT A SURVIVING MUTATION MEANS -- three answers, and picking the wrong one is
how you end up writing tests to protect code that should be deleted:

  - no test covers the invariant   -> ADD THE TEST
  - the mutated code is redundant  -> DELETE THE CODE
  - the mutation is equivalent     -> nothing; record why in the spec's `note`

USAGE
-----
    python scripts/mutate.py --spec scripts/mutations/tabular.toml
    python scripts/mutate.py --all
    python scripts/mutate.py --spec ... --only M2 --keep-going

This is NOT a per-commit gate: each mutation is a full `cargo test` run, so a
spec of ten mutations is ten test runs. Run it when you add a module or change an
invariant, and on the modules whose spec files say they matter.

SPEC FORMAT (TOML)
------------------
    # scripts/mutations/<module>.toml
    features = "tabular"          # cargo --features for every mutation here
    filter   = "tabular"          # cargo test filter (keeps runs short)

    [[mutation]]
    id      = "M1"
    file    = "src/tabular/sqlite_engine.rs"
    before  = "MixedColumns::NullifyForArithmetic => numeric,"
    after   = "MixedColumns::NullifyForArithmetic => Inferred::Text,"
    invariant = "the default policy types a mixed column numerically"
    occurrences = 1               # optional; must match exactly, else refuse
    note    = ""                  # required if `expect = \"survives\"`
    expect  = "dies"              # "dies" (default) or "survives"

`before` must be an exact substring of the file. It is matched literally, not as
a regex, because a regex that matches slightly more than intended mutates code
you did not read -- and `cargo fmt` reflowing a line has already made one
hand-written mutation "pass" without ever being applied.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO / "scripts" / "mutations"

# Verdicts. NOT_COMPILED is separate from DIED on purpose: a mutation that fails
# to build tells you nothing about the test suite, and counting it as a kill
# inflates the score with exactly the mutations that measured nothing.
DIED = "DIED"
SURVIVED = "SURVIVED"
NOT_COMPILED = "NOT_COMPILED"
NOT_APPLIED = "NOT_APPLIED"


# ---------------------------------------------------------------------------
# Restoring the tree, even when interrupted
# ---------------------------------------------------------------------------
# A mutation left behind is worse than a mutation never run: the next thing to
# read the file -- a human, a build, a commit -- sees sabotaged source with no
# sign of why. So every file this script touches is remembered before the first
# write, restored in a `finally`, and restored AGAIN from an atexit hook and the
# usual signals, with the content hash checked afterwards.
_ORIGINALS: dict[Path, bytes] = {}


def _remember(path: Path) -> bytes:
    raw = path.read_bytes()
    _ORIGINALS.setdefault(path, raw)
    return raw


def _restore_all() -> None:
    for path, raw in list(_ORIGINALS.items()):
        try:
            if path.read_bytes() != raw:
                path.write_bytes(raw)
        except OSError as exc:  # pragma: no cover - filesystem trouble
            print(f"!! could not restore {path}: {exc}", file=sys.stderr)


def _install_guards() -> None:
    atexit.register(_restore_all)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)

            def handler(signum, frame, _previous=previous):
                _restore_all()
                if callable(_previous):
                    _previous(signum, frame)
                else:
                    raise SystemExit(130)

            signal.signal(sig, handler)
        except (ValueError, OSError):  # pragma: no cover - not main thread
            pass


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------
@dataclass
class Mutation:
    id: str
    file: Path
    before: str
    after: str
    invariant: str
    features: str
    filter: str
    occurrences: int | None
    note: str
    expect: str


def load_spec(spec_path: Path) -> list[Mutation]:
    with spec_path.open("rb") as handle:
        data = tomllib.load(handle)

    features = data.get("features")
    if not features:
        raise SystemExit(f"{spec_path}: `features` is required at the top level")
    default_filter = data.get("filter", "")

    out: list[Mutation] = []
    for raw in data.get("mutation", []):
        for required in ("id", "file", "before", "after", "invariant"):
            if required not in raw:
                raise SystemExit(f"{spec_path}: a [[mutation]] is missing `{required}`")
        expect = raw.get("expect", "dies")
        if expect not in ("dies", "survives"):
            raise SystemExit(f"{spec_path}: {raw['id']}: `expect` must be dies|survives")
        note = raw.get("note", "")
        # An expected survivor is a claim that the mutation is behaviourally
        # equivalent or the code is deliberately unguarded. A claim with no
        # argument is not reviewable, and an unargued exemption reads as reviewed
        # to whoever finds it next.
        if expect == "survives" and not note.strip():
            raise SystemExit(
                f"{spec_path}: {raw['id']}: expect=\"survives\" needs a `note` saying "
                f"WHY a surviving mutation is acceptable here"
            )
        if raw["before"] == raw["after"]:
            raise SystemExit(f"{spec_path}: {raw['id']}: `before` and `after` are identical")
        out.append(
            Mutation(
                id=raw["id"],
                file=REPO / raw["file"],
                before=raw["before"],
                after=raw["after"],
                invariant=raw["invariant"],
                features=raw.get("features", features),
                filter=raw.get("filter", default_filter),
                occurrences=raw.get("occurrences"),
                note=note,
                expect=expect,
            )
        )
    if not out:
        raise SystemExit(f"{spec_path}: no [[mutation]] entries")
    return out


# ---------------------------------------------------------------------------
# Running one mutation at one site
# ---------------------------------------------------------------------------
def run_tests(mutation: Mutation) -> tuple[bool, list[str], str]:
    """Returns (compiled, failing test names, raw output)."""
    cmd = ["cargo", "test", "--features", mutation.features, "--lib"]
    if mutation.filter:
        cmd.append(mutation.filter)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = proc.stdout + proc.stderr
    # Match the specific shapes. `'error: ' in out` also matches cargo's own
    # "error: test failed, to rerun pass `--lib`", which is what a KILLED
    # mutation prints -- so that test reports every kill as a build failure.
    compiled = "error[E" not in out and "could not compile" not in out
    failing = [
        line.split()[1]
        for line in out.splitlines()
        if line.startswith("test ") and line.rstrip().endswith("FAILED")
    ]
    return compiled, failing, out


def sites_of(text: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return out
        out.append(found)
        start = found + 1


def toml_escape(value: str) -> str:
    r"""Escape a needle for a one-line TOML basic string.

    A literal newline is illegal inside `"…"`, so the self-test's multi-line
    case died in the TOML parser rather than exercising the matcher — the test
    failed for a reason that had nothing to do with what it was testing.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def read_for_matching(raw: bytes) -> str:
    """Decode a source file into the form every `before` is matched against.

    **One function, two callers, on purpose.** `--dry-run` and the real run used
    to read the same file differently: the pre-flight via `read_text()`, which
    applies Python's universal-newline translation, and the real run via
    `read_bytes().decode()`, which does not. On a CRLF working tree -- which is
    every Windows checkout with `core.autocrlf` -- a multi-line `before` written
    with `\\n` therefore **matched in the pre-flight and failed in the real run**.

    That made every multi-line mutation in this repository unmeasurable while
    `--dry-run` reported all of them fine, which is worse than a plain bug: the
    check that exists to catch a drifted anchor was the thing saying it had not
    drifted. Found on 2026-10-01 when `cargo fmt` rewrote a freshly-written
    LF file as CRLF and two of eight mutations turned into NOT_APPLIED between
    one run and the next.

    Normalising here and writing the mutated copy with `newline="\\n"` is safe:
    the original bytes are restored verbatim afterwards, so the working tree
    keeps whatever convention it had.
    """
    return raw.decode("utf-8").replace("\r\n", "\n")


def apply_one(mutation: Mutation, verbose: bool) -> list[tuple[str, str, str]]:
    """Mutate one site at a time. Returns [(site label, verdict, detail)]."""
    if not mutation.file.exists():
        return [(mutation.id, NOT_APPLIED, f"no such file: {mutation.file}")]

    original = _remember(mutation.file)
    try:
        text = read_for_matching(original)
    except UnicodeDecodeError:
        return [(mutation.id, NOT_APPLIED, "file is not UTF-8")]

    sites = sites_of(text, mutation.before)
    if not sites:
        return [
            (
                mutation.id,
                NOT_APPLIED,
                "`before` not found -- the code moved or `cargo fmt` reflowed it",
            )
        ]
    if mutation.occurrences is not None and len(sites) != mutation.occurrences:
        return [
            (
                mutation.id,
                NOT_APPLIED,
                f"spec says {mutation.occurrences} occurrence(s), file has {len(sites)}",
            )
        ]

    results: list[tuple[str, str, str]] = []
    for index, offset in enumerate(sites, start=1):
        # One site per run. Mutating every site at once asks "does at least one
        # of these have a test?", which passes as soon as the easiest one is
        # covered -- and the uncovered site is always on the branch nobody
        # thought about.
        label = mutation.id if len(sites) == 1 else f"{mutation.id}[{index}/{len(sites)}]"
        line_no = text.count("\n", 0, offset) + 1
        mutated = text[:offset] + mutation.after + text[offset + len(mutation.before):]
        mutation.file.write_text(mutated, encoding="utf-8", newline="\n")
        try:
            compiled, failing, out = run_tests(mutation)
        finally:
            mutation.file.write_bytes(original)
            if mutation.file.read_bytes() != original:  # pragma: no cover
                raise SystemExit(f"FATAL: could not restore {mutation.file}")

        where = f"{show(mutation.file)}:{line_no}"
        if not compiled:
            first = next((l for l in out.splitlines() if l.startswith("error")), "?")
            results.append((label, NOT_COMPILED, f"{where} -- {first}"))
            if verbose:
                print(out[-4000:], file=sys.stderr)
        elif failing:
            short = ", ".join(name.split("::")[-1] for name in failing[:3])
            more = f" (+{len(failing) - 3})" if len(failing) > 3 else ""
            results.append((label, DIED, f"{where} -- {short}{more}"))
        else:
            results.append((label, SURVIVED, f"{where} -- no test distinguishes it"))
    return results


# ---------------------------------------------------------------------------
# The cheap half, and the one worth gating on
# ---------------------------------------------------------------------------
def show(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    `relative_to` raises for a path outside the repository, and the self-test's
    fixture lives in a temp directory on purpose -- so a message that tries to
    prettify it would crash the very check it is reporting.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def dry_run(specs: list[Path]) -> int:
    """Check every spec still describes the code, without running anything.

    This is the part that rots. A mutation spec is a `before` string copied out of
    a source file, and the moment that line is edited -- renamed variable,
    `cargo fmt` reflowing an argument list, a refactor moving the match arm -- the
    spec stops matching and therefore stops testing anything. Silently: a full run
    would report NOT_APPLIED, but nobody runs the full suite of mutations on every
    commit, because each one is a `cargo test`.

    So the expensive half stays manual and THIS half is the gate: seconds to run,
    no compilation, and it fails loudly when a spec has drifted away from the code
    it claims to attack.
    """
    problems = 0
    checked = 0
    for spec_path in specs:
        try:
            mutations = load_spec(spec_path)
        except SystemExit as exc:
            print(f"!! {exc}")
            problems += 1
            continue
        for mutation in mutations:
            checked += 1
            name = f"{spec_path.name} {mutation.id}"
            if not mutation.file.exists():
                print(f"!! {name}: no such file: {show(mutation.file)}")
                problems += 1
                continue
            # The SAME reader the real run uses. Anything else and this
            # pre-flight can pass on a file the real run cannot mutate.
            text = read_for_matching(mutation.file.read_bytes())
            found = len(sites_of(text, mutation.before))
            if found == 0:
                print(
                    f"!! {name}: `before` no longer appears in "
                    f"{show(mutation.file)} -- this mutation tests NOTHING"
                )
                problems += 1
            elif mutation.occurrences is not None and found != mutation.occurrences:
                # Not cosmetic: a pattern that gained a site gained an untested
                # branch, and a pattern that lost one may have lost the code.
                print(
                    f"!! {name}: spec says {mutation.occurrences} occurrence(s), "
                    f"{show(mutation.file)} has {found}"
                )
                problems += 1
            else:
                sites = "" if found == 1 else f" ({found} sites)"
                print(f"   ok {name}: {mutation.invariant}{sites}")

    print()
    if problems:
        print(f"{problems} of {checked} mutation spec(s) no longer match the code.")
        print("Re-read the code and update the spec -- do NOT delete the entry to go green,")
        print("because the invariant is still there and would end up unguarded.")
        return 1
    print(f"{checked} mutation spec(s) all still match the code.")
    return 0


# ---------------------------------------------------------------------------
# Proving the gate discriminates
# ---------------------------------------------------------------------------
SELF_TEST_SPEC = '''\
features = "tabular"
filter = "tabular"

[[mutation]]
id = "SELF"
file = "{file}"
invariant = "the self-test's own fixture"
before = "{before}"
after = "{after}"
occurrences = {occurrences}
'''


def self_test() -> int:
    """Check that `--dry-run` FAILS on each way a spec can stop testing anything.

    Written because the first attempt at this check passed for the wrong reason:
    the script that was supposed to corrupt a spec hit the word `occurrences` in
    the file's *header comment* instead of the mutation entry, so the gate was
    never asked the question and reported OK. A gate that has only ever been run
    against good input is not a gate -- it is a green light nobody has tested.
    """
    import tempfile

    # A throwaway fixture, and NOT this file. The first version used
    # `scripts/mutate.py` as the fixture, which defeated itself: every needle the
    # test searched for was written into this file as a string literal, so
    # "a string that is definitely absent" was present, and the needle that
    # should appear once appeared twice. Three of four cases failed for that one
    # reason. A self-test that reads the file it lives in is measuring itself.
    fixture_body = "alpha\nbeta\nalpha\ngamma\n"
    present_once = "gamma"
    present_twice = "alpha"
    absent = "delta"

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture.txt"
        fixture.write_text(fixture_body, encoding="utf-8")
        gone = Path(tmp) / "never_written.txt"

        cases = [
            ("a spec that still matches", fixture, present_once, 1, 0),
            ("two sites and the spec says so", fixture, present_twice, 2, 0),
            ("`before` no longer in the file", fixture, absent, 1, 1),
            ("the occurrence count is too low", fixture, present_twice, 1, 1),
            ("the occurrence count is too high", fixture, present_once, 2, 1),
            ("the file itself is gone", gone, present_once, 1, 1),
        ]

        for label, target, before, occurrences, expected_rc in cases:
            path = Path(tmp) / "case.toml"
            path.write_text(
                SELF_TEST_SPEC.format(
                    file=target.as_posix(),
                    # Escaped for TOML: a literal newline is illegal inside a
                    # one-line basic string, so the multi-line case would die in
                    # the parser instead of testing the matcher.
                    before=toml_escape(before),
                    after="omega",
                    occurrences=occurrences,
                ),
                encoding="utf-8",
            )
            got = dry_run([path])
            ok = got == expected_rc
            failures += 0 if ok else 1
            print(f"  {'ok ' if ok else 'FAIL'} exit {got} (wanted {expected_rc}) -- {label}")

    # The matcher itself, on bytes, because the bug it guards against lives in
    # `apply_one` and NOT in `--dry-run`.
    #
    # Worth spelling out, because the first version of this test checked the
    # wrong half: `--dry-run` read via `read_text()`, whose universal-newline
    # translation already turned CRLF into `\n`, so a multi-line needle matched
    # there whether the bug was present or not. The broken path was `apply_one`,
    # which decoded raw bytes and kept `\r\n` — so on a CRLF working tree the
    # pre-flight reported eight of eight mutations fine while two of them came
    # back NOT_APPLIED from the real run. A test routed through the healthy path
    # cannot fail, and a test that cannot fail reads as coverage.
    crlf_cases = [
        ("a multi-line needle in a CRLF file", b"one\r\ntwo\r\nthree\r\n", "one\ntwo", 1),
        ("the same needle in an LF file", b"one\ntwo\nthree\n", "one\ntwo", 1),
        ("a single-line needle is unaffected", b"one\r\ntwo\r\n", "two", 1),
        ("a needle that is genuinely absent stays absent", b"one\r\ntwo\r\n", "one\nthree", 0),
    ]
    for label, raw, needle, want in crlf_cases:
        got = len(sites_of(read_for_matching(raw), needle))
        ok = got == want
        failures += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} {got} site(s) (wanted {want}) -- {label}")

    print()
    if failures:
        print(f"{failures} self-test case(s) failed: --dry-run does not catch what it claims to.")
        return 1
    print("--dry-run fails on all three kinds of drift and passes on a good spec,")
    print("and the matcher finds a multi-line needle whatever the line endings are.")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", action="append", default=[], help="a TOML spec file")
    parser.add_argument("--all", action="store_true", help=f"every spec in {SPEC_DIR}")
    parser.add_argument("--only", action="append", default=[], help="run only these ids")
    parser.add_argument("--keep-going", action="store_true", help="always exit 0")
    parser.add_argument("--verbose", action="store_true", help="dump cargo output on build failure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every spec without touching a file or running cargo (CI gate)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove --dry-run actually FAILS on each way a spec can drift",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    specs = [Path(s) for s in args.spec]
    if args.all:
        specs += sorted(SPEC_DIR.glob("*.toml"))
    if not specs:
        parser.error(f"pass --spec FILE or --all (specs live in {SPEC_DIR})")

    if args.dry_run:
        return dry_run(specs)

    _install_guards()

    rows: list[tuple[str, str, str, str, str]] = []
    for spec_path in specs:
        mutations = load_spec(spec_path)
        for mutation in mutations:
            if args.only and mutation.id not in args.only:
                continue
            print(f"-- {spec_path.name} {mutation.id}: {mutation.invariant}", flush=True)
            for label, verdict, detail in apply_one(mutation, args.verbose):
                rows.append((spec_path.name, label, verdict, mutation.expect, detail))
                print(f"   {verdict:<13} {detail}", flush=True)

    _restore_all()

    print()
    print(f"{'spec':<20} {'id':<14} {'verdict':<13} {'expected':<9} detail")
    print("-" * 100)
    bad = 0
    for spec, label, verdict, expect, detail in rows:
        wanted = DIED if expect == "dies" else SURVIVED
        flag = " " if verdict == wanted else "!"
        if verdict != wanted:
            bad += 1
        print(f"{flag}{spec:<19} {label:<14} {verdict:<13} {wanted:<9} {detail}")

    print()
    print(f"{len(rows) - bad} of {len(rows)} mutations behaved as the spec says.")
    if bad:
        print()
        print("A SURVIVED where DIED was expected means one of three things, and")
        print("guessing wrong is how you protect code that should be deleted:")
        print("  1. no test covers the invariant   -> add the test")
        print("  2. the mutated code is redundant  -> delete the code")
        print("  3. the mutation is equivalent     -> set expect=\"survives\" WITH a note")
        print("A NOT_COMPILED proves nothing: the tests never ran. Fix the mutation.")
        print("A NOT_APPLIED usually means the code moved. Re-read it before editing the spec.")
    return 0 if (args.keep_going or not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
