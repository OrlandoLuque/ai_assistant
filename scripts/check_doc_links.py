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
DEFAULT_FEATURES = (
    "full,autonomous,scheduler,butler,browser,distributed-agents,containers,"
    "audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,"
    "distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite,"
    "chaos-testing,local-inference"
)


def broken_links(features: str) -> list[tuple[str, int, str]]:
    """Every unresolved intra-doc link, as (file, line, symbol)."""
    proc = subprocess.run(
        ["cargo", "doc", "--no-deps", "--features", features],
        capture_output=True,
        text=True,
        errors="replace",
    )
    # rustdoc emits these as warnings, so a zero exit says nothing. Parse.
    found: list[tuple[str, int, str]] = []
    pending: str | None = None
    for line in proc.stderr.splitlines():
        # TWO classes, not one. The gate counted only `unresolved link` for its
        # first three days and reported 57 -> 1 while five of these went by
        # unseen -- a checker that reads less than it claims is the defect it
        # exists to catch, so it is spelled out here rather than left to a
        # regex somebody has to notice.
        m = re.search(r"unresolved link to `(.*)`", line)
        if m:
            pending = m.group(1)
            continue
        m = re.search(r"public documentation for `(.*?)` links to private item `(.*?)`", line)
        if m:
            found.append(("<private-link>", 0, f"{m.group(1)} -> {m.group(2)}"))
            continue
        if pending is not None:
            loc = re.search(r"-->\s+(\S+):(\d+):", line)
            if loc:
                found.append((loc.group(1), int(loc.group(2)), pending))
                pending = None
    if pending is not None:
        # A warning whose location line never arrived. Count it rather than
        # drop it: an uncounted warning is how a gate silently loosens.
        found.append(("<unknown>", 0, pending))
    return found


def main() -> int:
    features = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURES
    found = broken_links(features)
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
