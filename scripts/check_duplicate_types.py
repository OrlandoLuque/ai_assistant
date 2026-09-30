#!/usr/bin/env python3
"""Find type names declared more than once in the crate.

WHY THIS GATE EXISTS
--------------------
The same name in two modules is either an accident that will cost a bug, or a
decision nobody wrote down. Measured cases in this repository:

* **three** implementations of Reciprocal Rank Fusion, and the V316 fix landed in
  one of them (N99);
* **three** implementations of LLM-as-a-judge, none of them wired to anything;
* **two** enums called `StepStatus` — `task_planning` has
  `Pending/InProgress/Done/Blocked/Skipped`, `agent_graph` has
  `Running/Completed/Failed/Skipped`. The graph one **cannot express "blocked"**,
  so an agent step that is waiting on a resource has no state but `Failed`;
* **four** persistence modules (`conversation_snapshot`, `persistence`, `session`,
  `unified_persistence`).

In every case the duplicate was found by accident, months later, while looking for
something else. This makes it mechanical.

WHAT IT DOES *NOT* DO
---------------------
It compares **names**, not meaning. `RrfFusion` and `reciprocal_rank_fuse` are the
same idea and share no characters, so this gate would never pair them. Finding
*semantically* duplicated code is a judgement call and belongs to a reviewer or an
agent, not to a regex. Saying so here matters: a gate that looks like it finds
duplication, and only finds repeated spelling, is worse than no gate — somebody
will trust it.

THE CLASSES IT REPORTS
----------------------
Only the first one fails the build. The rest exist so that it does not drown: a
gate that reports 215 findings when 35 matter gets ignored, and the real ones go
with it.

1. **DIVERGENT** — same name, and *different variants*. The dangerous one: two
   things that look interchangeable and are not.
2. **CLONED** — same name, same variants, two files. Not confusable, but a fix to
   one will never reach the other. That is how the V316 RRF fix landed in one of
   three implementations.
3. **REPEATED** — same name, same kind, nothing to compare (structs, traits,
   aliases). Reported for information only.
4. **CFG_ALTERNATIVES** — one file, sibling inline modules. The ordinary
   `#[cfg]`-gated pattern; exactly one ever compiles.
5. **SEPARATE_BINARIES** — one declaration per `src/bin/*.rs`. Separate crate
   roots share no scope, so these cannot be confused at all.

USAGE
-----
    python scripts/check_duplicate_types.py            # ratchet against baseline
    python scripts/check_duplicate_types.py --list     # everything, no exit code
    python scripts/check_duplicate_types.py --self-test

The baseline lives in `scripts/duplicate_types_baseline.toml` and has **two**
sections, because "I decided this is fine" and "nobody has looked at this yet"
are different claims: `[allowed]` requires a reason per name, `[untriaged]` is the
frozen pre-existing list and says plainly that it was never judged. An unargued
exemption in `[allowed]` is refused — it would read as reviewed to whoever finds
it next, which is how a suppression whose written reason was false survived over
1607 lines of code that did not compile.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "duplicate_types_baseline.toml"

# `pub` is optional: a private duplicate is still two things with one name, and the
# StepStatus pair would have been missed by requiring `pub`.
DECL = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|type)\s+([A-Z]\w*)",
    re.MULTILINE,
)
# The leading identifier of a variant, once the body has been split at depth-0
# commas. Anchored at the start of the *piece*, not of a line — see below.
VARIANT_HEAD = re.compile(r"^\s*([A-Z]\w*)\s*(?:[,({=]|$)")


def enum_variants(text: str, start: int) -> list[str]:
    """Variant names of the enum whose declaration starts at `start`.

    Two things here are load-bearing, and both were found by the self-test:

    1. **Brace-counted body, not regex-delimited.** An enum with a struct variant
       contains nested braces; stopping at the first `}` truncates it and reports a
       false difference between two enums that are actually the same.
    2. **Split at depth-0 commas, not at line starts.** The first version required
       `^\\s{4,}` and so returned NOTHING for a one-line enum
       (`pub enum Alpha { One, Two }`). That is worse than a plain bug: two
       one-line enums would both report an empty variant set, compare as *equal*,
       and be classified REPEATED instead of DIVERGENT — a false negative that
       hides exactly the dangerous class this gate exists to find.
    """
    open_at = text.find("{", start)
    if open_at < 0:
        return []
    depth, i = 0, open_at
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # Comments go first: their commas are depth-0 commas and would split the list.
    body = strip_comments(text[open_at + 1 : i])

    # Split at commas that are at nesting depth 0 of the body.
    pieces, buf, d = [], [], 0
    for c in body:
        if c in "{([":
            d += 1
        elif c in "})]":
            d -= 1
        if c == "," and d == 0:
            pieces.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    pieces.append("".join(buf))

    out = []
    for piece in pieces:
        # Drop attributes and doc comments, which sit above the variant name.
        lines = [
            ln
            for ln in piece.splitlines()
            if not ln.strip().startswith(("#[", "//", "/*", "*"))
        ]
        m = VARIANT_HEAD.match("\n".join(lines).strip())
        if m:
            out.append(m.group(1))
    return out


def strip_comments(body: str) -> str:
    """Blank out `//`-comments, leaving the newlines so line numbers survive.

    Not cosmetic: the comma in a doc comment is a **depth-0 comma**, so splitting
    the body before removing comments cuts the variant list in the middle of a
    sentence. `CaptureMode`'s `/// Auto-record via VAD, queue on silence.` did
    exactly that and the enum reported **zero** variants -- four became none.

    The dangerous version of this bug is not the one that empties the list: a
    comma in the doc comment of the *third* variant drops only some of them, and
    a partial list is indistinguishable from a real difference. The gate would
    have reported a divergence between two identical enums, and whoever chased it
    would have found nothing wrong with the code.

    A `//` inside a string literal is left alone -- `#[doc = "see http://x"]` sits
    in enum bodies and truncating there would delete a real variant.
    """
    out, i, n = [], 0, len(body)
    in_string = False
    while i < n:
        c = body[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:  # an escaped quote does not close it
                out.append(body[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and body[i + 1] == "/":
            while i < n and body[i] != "\n":
                i += 1
            continue  # the newline itself is appended by the next iteration
        out.append(c)
        i += 1
    return "".join(out)


MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*\{", re.MULTILINE)


def mod_ranges(text: str) -> list[tuple[int, int, str]]:
    """`(start, end, name)` for every inline `mod NAME { … }` in the file.

    Needed because of a false positive that would have discredited this gate:
    `wasm_hooks.rs` declares **three** `UseAgentHook`, one each in `mod wasm_impl`,
    `mod wasm_stub` and `mod native`. That is the ordinary `#[cfg]`-gated
    alternative-implementation pattern — exactly one of them ever compiles — and
    reporting it as a duplicate is crying wolf. A gate that cries wolf is ignored,
    and the real findings go with it.
    """
    out = []
    for m in MOD_DECL.finditer(text):
        open_at = text.find("{", m.start())
        depth, i = 0, open_at
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((open_at, i, m.group(1)))
    return out


def scan() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(REPO.joinpath("src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        mods = mod_ranges(text)
        for m in DECL.finditer(text):
            kind, name = m.group(1), m.group(2)
            pos = m.start()
            # Innermost enclosing inline module, if any.
            enclosing = [(e - s, n) for s, e, n in mods if s < pos < e]
            module = min(enclosing)[1] if enclosing else ""
            # Test modules declare throwaway types on purpose; a duplicate there
            # is not the defect this gate is about.
            if module == "tests":
                continue
            entry = {
                "file": str(path.relative_to(REPO)).replace("\\", "/"),
                "line": text.count("\n", 0, pos) + 1,
                "kind": kind,
                "module": module,
            }
            if kind == "enum":
                entry["variants"] = enum_variants(text, pos)
            found[name].append(entry)
    return found


def load_baseline() -> tuple[dict[str, str], set[str]]:
    """`[allowed]` name -> reason, and the frozen-but-unreviewed `[untriaged]` set.

    Two sections, because they mean different things and collapsing them would let
    the file lie. `[allowed]` is a decision: someone looked at both enums and
    concluded the difference is deliberate, and the reason says why — an entry
    without one is rejected, since a reason-less exemption is indistinguishable
    from an oversight and nobody ever re-reads it (the `ffi` suppression that
    claimed "nothing to type-check" over 1607 lines with two compile errors is
    the precedent).

    `[untriaged]` is the opposite: pre-existing duplicates nobody has judged yet.
    They are frozen so the count cannot grow, and the gate prints them as debt on
    every run. Writing 37 invented reasons to fill `[allowed]` would have made the
    file look reviewed when it was not, which is worse than an honest debt list.
    """
    if not BASELINE.is_file():
        return {}, set()
    import tomllib

    with BASELINE.open("rb") as handle:
        data = tomllib.load(handle)
    allowed: dict[str, str] = {}
    for name, reason in (data.get("allowed") or {}).items():
        if not str(reason).strip():
            sys.exit(f"{BASELINE}: `{name}` has no reason. Every exemption needs one.")
        allowed[name] = str(reason)
    untriaged = {str(n) for n in (data.get("untriaged") or {}).get("names", [])}
    both = sorted(set(allowed) & untriaged)
    if both:
        sys.exit(
            f"{BASELINE}: {', '.join(both)} is in both [allowed] and [untriaged]. "
            "A name is either judged or it is not."
        )
    return allowed, untriaged


def baseline_label() -> str:
    """The baseline's path for a message, without assuming it is inside the repo.

    `relative_to` RAISES when it is not, and the one place that path is not inside
    the repo is the test that proves this gate FAILS on a new duplicate -- so the
    convenience call crashed exactly the run whose exit code was the thing being
    measured, making a real rc=1 indistinguishable from a traceback's rc=1.
    """
    try:
        return str(BASELINE.relative_to(REPO))
    except ValueError:
        return str(BASELINE)


def classify(name: str, sites: list[dict]) -> str:
    # One file, different inline modules: the cfg-gated alternative-implementation
    # pattern. Legitimate, and reported apart so it does not drown the rest.
    files = {s["file"] for s in sites}
    modules = {s.get("module", "") for s in sites}
    if len(files) == 1 and len(modules) == len(sites) and "" not in modules:
        return "CFG_ALTERNATIVES"

    # Every `src/bin/*.rs` is its own crate root. Two enums in two different
    # binaries share no scope whatsoever -- nothing can import one where the other
    # is expected, so they are not confusable even with identical names and
    # different variants. `Tab` in three GUI binaries is the case: each one lists
    # its own tabs and always will. A structural fact, so it is classified rather
    # than exempted: an entry in the baseline would imply someone weighed it.
    if len(files) == len(sites) and all(f.startswith("src/bin/") for f in files):
        return "SEPARATE_BINARIES"

    kinds = {s["kind"] for s in sites}
    if kinds == {"enum"}:
        shapes = {tuple(sorted(s.get("variants") or [])) for s in sites}
        if len(shapes) > 1:
            return "DIVERGENT"
        # Same name, same variants, different files: a copy-paste. Not dangerous
        # to confuse, but a change to one will not reach the other.
        return "CLONED"
    return "REPEATED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="print everything, exit 0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    allowed, untriaged = load_baseline()
    dupes = {n: s for n, s in scan().items() if len(s) > 1}

    by_class: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for name, sites in dupes.items():
        by_class[classify(name, sites)][name] = sites
    divergent = by_class["DIVERGENT"]
    cloned = by_class["CLONED"]
    repeated = by_class["REPEATED"]
    cfg_alt = by_class["CFG_ALTERNATIVES"]
    sep_bins = by_class["SEPARATE_BINARIES"]

    def show(title: str, group: dict[str, list[dict]]) -> None:
        if not group:
            return
        print(f"\n=== {title} ({len(group)})")
        for name in sorted(group):
            mark = "  " if name in allowed else ".." if name in untriaged else "!!"
            print(f"{mark} {name}")
            for s in group[name]:
                extra = ""
                if s["kind"] == "enum":
                    v = s.get("variants") or []
                    extra = f"  [{', '.join(v[:6])}{'…' if len(v) > 6 else ''}]"
                print(f"     {s['file']}:{s['line']} ({s['kind']}){extra}")
            if name in allowed:
                print(f"     allowed: {allowed[name]}")

    known = set(allowed) | untriaged
    fresh = sorted(set(divergent) - known)

    # A passing gate is quiet. Dumping all 35 divergent names on a green run trains
    # the reader to scroll past the section, which is where a new one would appear.
    if args.list or fresh:
        show("DIVERGENT — same name, DIFFERENT variants (the dangerous class)", divergent)
    if args.list:
        show("CLONED — same name, same variants, different files", cloned)
        show("REPEATED — same name, same kind, no variants to compare", repeated)
        show("CFG_ALTERNATIVES — one file, sibling modules (legitimate)", cfg_alt)
        show("SEPARATE_BINARIES — one per binary crate, unconfusable", sep_bins)
        print(
            f"\ntotal duplicated names: {len(dupes)}"
            f"  (divergent {len(divergent)}, cloned {len(cloned)},"
            f" repeated {len(repeated)}, cfg-alternatives {len(cfg_alt)},"
            f" separate-binaries {len(sep_bins)})"
        )
        return 0

    print(f"\nduplicated type names: {len(dupes)}  (divergent enums: {len(divergent)})")
    if fresh:
        print("\nDIVERGENT and in neither section of the baseline:")
        for n in fresh:
            print(f"  - {n}")
        print(
            "\nTwo enums with one name and different variants are two things that look\n"
            "interchangeable and are not. Unify them, or — if the difference really is\n"
            f"deliberate — add the name to [allowed] in {baseline_label()}\n"
            "with the reason. [untriaged] is the frozen pre-existing list and is not\n"
            "somewhere to put a new one."
        )
        return 1

    # A name in the baseline that no longer duplicates is debt that got paid; say so
    # instead of leaving the file to accumulate entries nobody can retire.
    stale = sorted(known - set(dupes))
    debt = sorted(untriaged & set(divergent))
    print(f"OK — no new divergent duplicate. Frozen debt: {len(debt)} untriaged.")
    if stale:
        print(
            f"\n{len(stale)} baseline entries no longer duplicate — remove them from\n"
            f"{baseline_label()} so the ratchet keeps tightening: "
            + ", ".join(stale)
        )
    return 0


def self_test() -> int:
    """Prove the scanner sees what it claims, on text with a known answer."""
    import tempfile

    sample = """
pub enum Alpha { One, Two }
pub struct Beta { x: u8 }
enum Gamma {
    WithFields { a: u8, b: u8 },
    Plain,
}
pub enum Delta {
    /// Auto-record via VAD, queue on silence.
    Vad,
    /// Record while held, queue on release.
    PushToTalk,
    Continuous,
}
pub enum Epsilon {
    #[doc = "see http://example.test, second clause"]
    Kept,
    Other,
}
"""
    cases = [
        ("finds a plain enum's variants", "Alpha", ["One", "Two"]),
        # The brace-counted body is what makes this one work: a regex stopping at
        # the first `}` would cut Gamma after WithFields.
        ("does not truncate at a nested brace", "Gamma", ["WithFields", "Plain"]),
        # The real CaptureMode. Before strip_comments this returned [] -- four
        # variants read as none, which is the false-difference generator.
        (
            "a comma inside a doc comment does not eat the variants",
            "Delta",
            ["Vad", "PushToTalk", "Continuous"],
        ),
        # And the other direction: the fix must not treat `//` in a string as a
        # comment, or it would swallow the variant that follows.
        ("a `//` inside a string literal is not a comment", "Epsilon", ["Kept", "Other"]),
    ]
    failures = 0
    for label, name, want in cases:
        m = re.search(rf"enum\s+{name}", sample)
        got = enum_variants(sample, m.start()) if m else []
        ok = got == want
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got}")

    # And the classifier: same name, different variants must be DIVERGENT.
    div = classify(
        "X",
        [
            {"kind": "enum", "variants": ["A", "B"], "file": "a", "line": 1, "module": ""},
            {"kind": "enum", "variants": ["A", "C"], "file": "b", "line": 1, "module": ""},
        ],
    )
    same = classify(
        "Y",
        [
            {"kind": "enum", "variants": ["A", "B"], "file": "a", "line": 1, "module": ""},
            {"kind": "enum", "variants": ["B", "A"], "file": "b", "line": 1, "module": ""},
        ],
    )
    # The sibling-module case, which is the false positive that nearly discredited
    # the gate: three `UseAgentHook` in one file, one per cfg-gated module.
    siblings = classify(
        "Z",
        [
            {"kind": "struct", "file": "a.rs", "line": 1, "module": "wasm_impl"},
            {"kind": "struct", "file": "a.rs", "line": 2, "module": "native"},
        ],
    )
    for label, got, want in [
        ("different variants -> DIVERGENT", div, "DIVERGENT"),
        # Same name and same variants in two files is a copy-paste: not confusable,
        # but a change to one will not reach the other. Its own class since the
        # taxonomy grew.
        ("same variants, two files -> CLONED", same, "CLONED"),
        ("one file, sibling modules -> CFG_ALTERNATIVES", siblings, "CFG_ALTERNATIVES"),
    ]:
        ok = got == want
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got}")

    # The baseline loader's two refusals. Both are the whole point of splitting the
    # file in two sections, so an untested "it rejects that" would be a claim, not a
    # guarantee -- and this repository's recurring defect is exactly the claim that
    # nothing checks.
    global BASELINE  # noqa: PLW0603 -- swapping the target file is the fixture
    real_baseline = BASELINE
    with tempfile.TemporaryDirectory() as tmp:
        for label, body, want_exit in [
            (
                "rejects an [allowed] entry with an empty reason",
                '[allowed]\nFoo = ""\n',
                True,
            ),
            (
                "rejects a name in both sections",
                '[allowed]\nFoo = "deliberate"\n\n[untriaged]\nnames = ["Foo"]\n',
                True,
            ),
            (
                "accepts a reason plus a disjoint untriaged list",
                '[allowed]\nFoo = "deliberate"\n\n[untriaged]\nnames = ["Bar"]\n',
                False,
            ),
        ]:
            path = Path(tmp) / "b.toml"
            path.write_text(body, encoding="utf-8")
            BASELINE = path
            try:
                load_baseline()
                exited = False
            except SystemExit:
                exited = True
            ok = exited == want_exit
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} {label}: exited={exited}")
    BASELINE = real_baseline

    print()
    if failures:
        print(f"{failures} self-test case(s) failed: the scanner does not do what it says.")
        return 1
    print("the scanner reads variants correctly and classifies divergence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
