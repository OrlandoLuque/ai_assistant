#!/usr/bin/env python3
"""Ratchet on rustdoc's doc-link warnings: the count may fall, never rise.

Counts BOTH classes rustdoc reports:

* `unresolved link to X` -- the target does not exist from here.
* `public documentation for X links to private item Y` -- the target exists but
  the reader cannot reach it, so the rendered page shows a link to nothing.

The second was missed for three days while the first was being drained, which
is why the parser names them one by one instead of matching "warning".

# Why a ratchet and not a clean zero

Measured 2026-09-20: 57 links in public documentation point at symbols rustdoc
cannot resolve, so on the rendered page they are dead. Demanding zero today
would mean either fixing all 57 in one sitting or leaving the door open, and
the door is the part that matters -- one of those 57 was introduced and shipped
on the same day the others were found (V332 renamed `default_thread_count` to
`thread_policy` and left the link pointing at the old name). A count that can
only fall stops the growth now and lets the backlog drain in batches.

Lower BASELINE with every batch you fix. When it reaches zero, delete this
script and put `-D rustdoc::broken_intra_doc_links` in RUSTDOCFLAGS instead.

# Why the feature set is part of the contract

The count depends on which features are compiled, and **wider finds more**, not
fewer: `full,local-inference` reported 46 while the CI set reported 57, because
more code compiled means more documentation checked. A gate run against a narrow
set would report a reassuring number that is not true of what CI ships. So the
feature set is passed in and recorded here rather than left to whoever runs it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ZERO, reached 2026-09-22. The drain ran 57 -> 49 -> 45 -> 32 -> 0, and the
# last step also uncovered five warnings of a SECOND class the parser had never
# read, so the true starting point was 62 and not 57.
#
# Now that it is zero the ratchet has done its job and the check is absolute:
# any broken link at all fails. This stays a script rather than becoming
# `-D rustdoc::broken_intra_doc_links` because that flag does not cover the
# private-item lint, and because the per-link listing below is what makes a
# failure fixable instead of merely loud.
BASELINE = 0

# Must match ci.yml's FEATURES_STD. Passed explicitly so a narrower default
# cannot quietly make the gate pass by checking less.
#
# And on 2026-11-24 it did not match: this literal carried 21 features while
# FEATURES_STD carried 24, missing `ffi`, `tabular` and `test-utils`. The
# comment above stated a requirement the code did not meet, which is the exact
# thing this gate exists to catch, one layer in. CI was unaffected -- it passes
# the argument -- but every hand-run reported a number taken over less code
# than CI ships, and the module docstring says why that matters: wider finds
# MORE, not fewer.
#
# A copy cannot be trusted to stay in step, so `self_test` now re-reads
# ci.yml and fails if the two diverge. That is the same shape as the README
# gate cross-checking `ci_gates` against `ci_gates_prose`: one definition,
# two places, and a check that notices.
DEFAULT_FEATURES = (
    "full,autonomous,scheduler,butler,browser,distributed-agents,containers,"
    "audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,"
    "distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite,"
    "chaos-testing,local-inference,ffi,tabular,test-utils"
)

CI_YML = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def features_std_from_ci() -> str | None:
    """`FEATURES_STD` as ci.yml defines it, or None if it cannot be read."""
    try:
        text = CI_YML.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'FEATURES_STD:\s*"([^"]+)"', text)
    return match.group(1) if match else None


# Measured 2026-11-24, and worth knowing BEFORE distrusting a fast run: cargo
# replays rustdoc's cached diagnostics. A fully cached `cargo doc` that finishes
# in 0.02s re-prints every warning the cold run printed, verbatim -- checked on a
# throwaway crate with two broken links, cold run and cached run byte-identical.
# So a green that came back in two seconds is NOT a green that checked nothing.
# The thing to distrust is a non-zero exit, which is what `untrustworthy` below
# is for.
def cargo_doc(features: str) -> subprocess.CompletedProcess[str]:
    """One `cargo doc` run, and nothing else.

    Kept apart from the parsing AND from the deciding so that `--self-test` can
    exercise the other two without a toolchain. The parsing half has had a
    self-test since the day a run of location-less warnings collapsed into one;
    the half that *runs cargo* had none, which is exactly where the defect
    below lived for two months.
    """
    return subprocess.run(
        ["cargo", "doc", "--no-deps", "--features", features],
        capture_output=True,
        text=True,
        errors="replace",
    )


def untrustworthy(returncode: int, stderr: str) -> str | None:
    """Why this run's count cannot be believed, or None when it can.

    This function exists because of an asymmetry that was half written down.
    The old code said, correctly, that "rustdoc emits these as warnings, so a
    zero exit says nothing" -- and then returned the parsed warnings without
    ever looking at the exit code. The unwritten half is that a **non-zero**
    exit says everything about the COUNT: cargo stopped early, so the
    documentation it produced is a subset of what CI documents, and a subset can
    only under-report.

    Measured 2026-11-24, with a feature name that does not exist: cargo exits
    101 having compiled nothing, its stderr carries no `unresolved link to`
    line, so the parse returned an empty list and the gate printed
    `broken intra-doc links: 0 (baseline 0)` and exited **0**. A gate that
    reports green when its instrument never ran is worse than no gate, because
    it is the one you stop checking.

    The rule is the exit code **alone**, not "non-zero and nothing parsed": a
    partial compile can emit real warnings before it stops, and believing those
    would mean reporting a count measured over a smaller body of code than the
    one this gate promises in its own header.
    """
    if returncode == 0:
        return None
    errors = [
        line.strip()
        for line in stderr.splitlines()
        if line.lstrip().startswith("error")
    ]
    detail = errors[0] if errors else "(cargo printed no `error` line)"
    return f"cargo doc exited {returncode}: {detail}"


def parse_warnings(stderr: str) -> list[tuple[str, int, str]]:
    """Read rustdoc's diagnostics. Split out from the cargo call so `--self-test`
    can feed it the shapes that once fooled it, below."""
    found: list[tuple[str, int, str]] = []
    pending: str | None = None

    def flush(location: str = "<no-location>", line_no: int = 0) -> None:
        """Record the warning being parsed, located or not, and clear it.

        Every exit from a warning block goes through here. A `pending` that is
        overwritten instead of flushed is a warning that silently stopped
        existing, which is how a gate loosens without anybody editing it.
        """
        nonlocal pending
        if pending is not None:
            found.append((location, line_no, pending))
            pending = None

    for line in stderr.splitlines():
        # TWO classes, not one. The gate counted only `unresolved link` for its
        # first three days and reported 57 -> 1 while five of these went by
        # unseen -- a checker that reads less than it claims is the defect it
        # exists to catch, so it is spelled out here rather than left to a
        # regex somebody has to notice.
        m = re.search(r"unresolved link to `(.*)`", line)
        if m:
            # MEASURED 2026-09-25, and this cost a wrong diagnosis: rustdoc
            # emits some of these with NO `-->` line at all -- among them every
            # link in a module whose `pub mod x;` declaration carries an outer
            # `///` comment, which makes rustdoc resolve the module's own `//!`
            # docs in the parent scope.
            #
            # The previous version kept one `pending` and only cleared it on
            # seeing a location. So a run of location-less warnings collapsed
            # into ONE, and that survivor inherited the `-->` of an unrelated
            # later warning. It reported 1 of 8 real breakages and blamed a file
            # that had nothing to do with any of them.
            flush()
            pending = m.group(1)
            continue
        m = re.search(r"public documentation for `(.*?)` links to private item `(.*?)`", line)
        if m:
            flush()
            found.append(("<private-link>", 0, f"{m.group(1)} -> {m.group(2)}"))
            continue
        if pending is not None:
            loc = re.search(r"-->\s+(\S+):(\d+):", line)
            if loc:
                flush(loc.group(1), int(loc.group(2)))
                continue
            # Any other diagnostic ends this warning's block, so its location
            # never arrived. `-->` always comes immediately after the `warning:`
            # line when it comes at all, so anything else means there is none.
            if line.startswith("warning") or line.startswith("error"):
                flush()

    # And the last block, which no following line closes.
    flush()
    return found


# Captured verbatim from rustdoc 1.98.1 on 2026-09-25. Two warnings with no
# `-->` line, then one with. The parser this replaced reported ONE finding from
# this input and gave it the located warning's file, because it kept a single
# `pending` and only cleared it on seeing a location.
SELF_TEST_STDERR = """warning: unresolved link to `recall_at_k`
  |
  = note: the link appears in this line:

          | [`recall_at_k`] | of the relevant documents that exist |
             ^^^^^^^^^^^^^
  = note: no item named `recall_at_k` in scope
  = note: `#[warn(rustdoc::broken_intra_doc_links)]` on by default

warning: unresolved link to `ndcg_at_k`
  |
  = note: the link appears in this line:

          | [`ndcg_at_k`] | the whole ordering |
             ^^^^^^^^^^^
  = note: no item named `ndcg_at_k` in scope
  = help: to escape `[` and `]` characters, add '\\' before them

warning: unresolved link to `thread_policy`
  --> src/model_recommender.rs:7:41
   |
7  | /// See [`thread_policy`] for the rest.
   |          ^^^^^^^^^^^^^^^ no item named `thread_policy` in scope
"""


def self_test() -> int:
    """Prove the parser sees every warning, located or not."""
    found = parse_warnings(SELF_TEST_STDERR)
    symbols = sorted(sym for _f, _l, sym in found)
    expected = sorted(["recall_at_k", "ndcg_at_k", "thread_policy"])
    if symbols != expected:
        print(f"SELF-TEST FAIL: expected {expected}, got {symbols}")
        return 1

    located = {sym: (f, l) for f, l, sym in found}
    if located["thread_policy"] != ("src/model_recommender.rs", 7):
        print(f"SELF-TEST FAIL: wrong location: {located['thread_policy']}")
        return 1
    # The two unlocated ones must NOT have stolen the third's location. This is
    # the specific defect: it made eight real breakages read as one, in a file
    # that had nothing to do with them.
    for sym in ("recall_at_k", "ndcg_at_k"):
        if located[sym][0] != "<no-location>":
            print(f"SELF-TEST FAIL: {sym} borrowed a location: {located[sym]}")
            return 1

    # ---- the half that RUNS cargo, which had no test at all until 2026-11-24
    # and is therefore where the defect lived. Five cases, and the fifth is the
    # one that decides the rule.
    cases: list[tuple[int, str, bool]] = [
        # a zero exit is trustworthy whether or not warnings came with it:
        # rustdoc reports these AS warnings, so cargo succeeds either way
        (0, SELF_TEST_STDERR, False),
        (0, "", False),
        # a non-zero exit is not, and the common shape is a feature typo
        (101, "error: the package 'x' does not contain this feature: y", True),
        # ... even when cargo printed no `error` line we can quote
        (101, "", True),
        # THE case: a partial compile can emit real warnings before it stops.
        # Those warnings are true but the COUNT is taken over less code than
        # the gate promises, so they must not buy the run a pass. Writing the
        # rule as "non-zero AND nothing parsed" would have let this through.
        (101, SELF_TEST_STDERR, True),
    ]
    for code, err, must_refuse in cases:
        verdict = untrustworthy(code, err)
        if (verdict is not None) != must_refuse:
            print(f"SELF-TEST FAIL: untrustworthy({code}, ...) -> {verdict!r}")
            return 1

    # The reason has to name the exit code, or a red run is not fixable from
    # the log alone.
    reason = untrustworthy(101, "")
    if reason is None or "101" not in reason:
        print(f"SELF-TEST FAIL: the reason does not name the exit code: {reason!r}")
        return 1

    # ---- the copied feature set, which had drifted by three features ----
    # Not a formality either: on 2026-11-24 DEFAULT_FEATURES carried 21 while
    # ci.yml carried 24, sitting under a comment that said it must match. A
    # copy nothing compares is a copy that drifts, and a narrower one makes
    # this gate report a smaller, untrue number -- wider finds MORE.
    #
    # Unreadable ci.yml is a FAILURE here, not a skip: a check that quietly
    # does not run is the defect this whole file is about.
    ci_features = features_std_from_ci()
    if ci_features is None:
        print(f"SELF-TEST FAIL: cannot read FEATURES_STD from {CI_YML}")
        return 1
    if ci_features != DEFAULT_FEATURES:
        here = set(DEFAULT_FEATURES.split(","))
        there = set(ci_features.split(","))
        print("SELF-TEST FAIL: DEFAULT_FEATURES has drifted from ci.yml's FEATURES_STD.")
        print(f"  in ci.yml and not here: {sorted(there - here)}")
        print(f"  here and not in ci.yml: {sorted(here - there)}")
        return 1

    print("self-test ok: 3 of 3 warnings seen, none given a borrowed location")
    print("self-test ok: 5 of 5 cargo-exit cases, including a failure WITH warnings")
    print(
        "self-test ok: DEFAULT_FEATURES matches ci.yml's FEATURES_STD "
        f"({len(DEFAULT_FEATURES.split(','))} features)"
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    features = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURES
    proc = cargo_doc(features)
    reason = untrustworthy(proc.returncode, proc.stderr)
    if reason is not None:
        sys.stderr.write(f"error: {reason}\n")
        sys.stderr.write(
            "error: the link count is NOT reported for this run. A cargo that\n"
            "error: stopped early documents less code than CI does, so any\n"
            "error: count taken from it can only under-report. Fix the cargo\n"
            "error: failure and run again.\n"
        )
        for line in proc.stderr.splitlines()[-15:]:
            sys.stderr.write(f"  | {line}\n")
        return 2

    found = parse_warnings(proc.stderr)
    n = len(found)

    print(f"broken intra-doc links: {n} (baseline {BASELINE})")

    if n > BASELINE:
        print()
        print(f"FAIL: {n - BASELINE} new broken doc link(s).")
        print("A link to a symbol that does not exist renders as dead text.")
        print("Fix it, or if the brackets were never meant to be a link, escape")
        print(r"them as \[like this\].")
        print()
        for f, line, sym in sorted(found):
            print(f"  {f}:{line}  ->  {sym}")
        return 1

    if n < BASELINE:
        print()
        print(f"{BASELINE - n} fixed since the baseline was set. Lower BASELINE")
        print(f"in {__file__} to {n} so the ground you gained is held.")
        # Deliberately not a failure: a green build should not punish progress.
        # But it is loud, because a baseline nobody lowers stops being a ratchet.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
