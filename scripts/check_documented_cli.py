#!/usr/bin/env python3
"""
V292 — documented-CLI gate.

Catches the class of bug found in V291: documentation that shows an `ai_cli`
command line the binary does not accept.

Why it matters, and why it is not cosmetic. `ai_cli`'s flag loops end in a catch-all
that treats any unrecognised word as part of the positional argument:

    other => query_parts.push(other.to_string()),

So `ai_cli research "diffusion models" --bibtex --output out.bib` does not fail. It
searches for the phrase *"diffusion models --output out.bib"*, prints results, and
exits 0. A reader who copy-pastes it gets a plausible wrong answer with no error to
tell them otherwise. The research guide carried five such flags — `--review`,
`--depth`, `--format`, `--year-range`, `--output` — plus `--faithfulness` and
`--quality-gates`, which are real flags of a *different* command, and a
sub-subcommand (`research import-bib`) that never existed. Directly above one of
them the guide said "Copy, paste, run."

What is checked, for every `ai_cli <sub> …` line in the scanned files:

  1. `<sub>` is a real top-level subcommand.
  2. Every `--flag` on that line is accepted by that subcommand — resolved by
     walking the subcommand's `cmd_*` function and the `cmd_*` helpers it calls,
     so `research review --mode` resolves through `cmd_research_review`.

Usage:
    python scripts/check_documented_cli.py [paths...]

With no arguments it scans `docs/*.md` and `../ai_assistant-website/*.html` when
that directory exists. Exits 1 if anything is flagged.

Resolution is deliberately narrow. Widening the rule to "any flag that exists
anywhere in the CLI" is exactly what let `research --output` through on the first
pass: `--output` is real, just not on that command. When a correct flag is
reported, fix the resolver -- do not add it to ALLOWED.
"""

import collections
import glob
import html
import io
import os
import re
import sys

CLI_SRC = os.path.join("src", "bin", "ai_cli.rs")

# (subcommand, flag) pairs the script cannot resolve but which are correct.
# Each needs a reason. An entry without one is a bug being hidden.
#
# Currently empty, and that is the point: an earlier draft needed two entries
# (`cost --snapshot`, `butler --intent`) purely because the call walk stopped at
# `cmd_*` functions and did not read nested dispatch. Both disappeared once the
# resolver was made correct. Prefer fixing the resolver to growing this set — an
# allowlist entry silences a whole (subcommand, flag) pair forever, including the
# day it stops being true.
ALLOWED = set()


def function_bodies(src):
    """Map fn name -> source text, from `fn name(` at column 0 to the next one.

    Bodies are *concatenated* when a name is defined more than once. `cmd_research_ask`
    exists twice — the real one under `cfg(rag)` and a stub that refuses under
    `cfg(not(rag))` — and keeping only the last definition meant reading the stub,
    which parses no flags, so every flag of the real command looked invented.
    """
    bodies = collections.defaultdict(str)
    starts = [(m.group(1), m.start()) for m in re.finditer(r"^fn ([a-z_0-9]+)\(", src, re.M)]
    for i, (name, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(src)
        bodies[name] += src[pos:end]
    return dict(bodies)


def flags_of(bodies, fn, depth=3, seen=None):
    """Flags accepted by `fn`, following calls to any local fn `depth` levels down.

    Following *every* local callee, not just `cmd_*`, matters: `cmd_benchmark_run`
    parses nothing itself and hands `args` to `prepare_benchmark_run`, so a
    cmd-only walk reported four correct flags as invented.
    """
    seen = seen if seen is not None else set()
    if fn in seen or fn not in bodies or depth < 0:
        return set()
    seen.add(fn)
    body = bodies[fn]
    out = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', body))
    for callee in set(re.findall(r"\b([a-z_][a-z_0-9]*)\s*\(", body)):
        if callee in bodies:
            out |= flags_of(bodies, callee, depth - 1, seen)
    return out


def default_targets():
    # `docs/IMPROVEMENTS_V*.md` are a historical record, not documentation: they say what
    # was believed when they were written, and rewriting them to match today's CLI would
    # falsify the record. They are excluded here rather than silently passing, and they
    # can still be scanned on purpose by naming them on the command line.
    targets = [
        p
        for p in sorted(glob.glob(os.path.join("docs", "*.md")))
        if not os.path.basename(p).startswith("IMPROVEMENTS_V")
    ]
    website = os.path.join("..", "ai_assistant-website")
    if os.path.isdir(website):
        targets += sorted(glob.glob(os.path.join(website, "*.html")))
    return targets


def main(argv):
    if not os.path.exists(CLI_SRC):
        print("error: run this from the repository root ({} not found)".format(CLI_SRC))
        return 2

    src = io.open(CLI_SRC, encoding="utf-8").read()
    bodies = function_bodies(src)
    dispatch = dict(re.findall(r'"([a-z][a-z-]*)" => (cmd_[a-z_0-9]+)\(&command_args', src))
    if not dispatch:
        print("error: could not parse the ai_cli dispatch table — has main() changed shape?")
        return 2
    per_cmd = {sub: flags_of(bodies, fn) for sub, fn in dispatch.items()}
    global_flags = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', bodies.get("main", "")))

    # Sub-subcommands: `benchmark run`, `research ask`, `recipes show`… Their flags
    # live in the nested cmd_* function, and attributing them to the parent is how
    # this check would otherwise flag `research ask --top-k`, which is correct.
    nested = {}
    for sub, fn in dispatch.items():
        body = bodies.get(fn, "")
        pairs = re.findall(r'"([a-z][a-z0-9-]*)" => (cmd_[a-z_0-9]+)\(', body)
        # `research` dispatches its subcommands with an `if … == Some("ask")` before
        # the flag loop rather than a match arm, so that shape has to be read too.
        pairs += re.findall(r'Some\("([a-z][a-z0-9-]*)"\)[\s\S]{0,200}?\b(cmd_[a-z_0-9]+)\(', body)
        for word, callee in pairs:
            nested[(sub, word)] = flags_of(bodies, callee) | per_cmd[sub]

    # `ai_cli <sub>` or `cargo run --bin ai_cli --features "…" -- <sub>`.
    invocation = re.compile(r"ai_cli(?:[^\n`]*?\s--)?\s+([a-z][a-z-]*)([^\n`]*)")

    targets = argv[1:] or default_targets()
    issues = collections.defaultdict(set)
    scanned = 0

    for path in targets:
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        scanned += 1
        if path.endswith(".html"):
            text = html.unescape(text)
        # Normalise first: the working tree is CRLF, so a `\`-continuation is
        # "\\\r\n" and joining on "\\\n" alone silently leaves every continued
        # command split — which hides exactly the multi-line blocks where the
        # invented flags live.
        text = text.replace("\r\n", "\n")
        # Shell line continuations: join so a flag on its own line still belongs
        # to the command that started above it.
        text = text.replace("\\\n", " ")
        for m in invocation.finditer(text):
            sub, rest = m.group(1), m.group(2)
            if sub not in per_cmd:
                continue  # not a subcommand at all — some other prose match
            second = re.match(r'\s+"?([a-z][a-z0-9-]*)"?', rest)
            key = (sub, second.group(1)) if second else None
            allowed = (nested.get(key) or per_cmd[sub]) | global_flags
            for flag in re.findall(r"(--[a-z][a-z0-9-]*)", rest):
                if flag not in allowed and (sub, flag) not in ALLOWED:
                    issues[path].add((sub, flag))

    total = sum(len(v) for v in issues.values())
    if total:
        print("Documented flags the subcommand does not accept:\n")
        for path in sorted(issues):
            print("  {}".format(path))
            for sub, flag in sorted(issues[path]):
                print("      ai_cli {:<14} {}".format(sub, flag))
            print()
        print("{} problem(s) across {} file(s).".format(total, len(issues)))
        print("A wrong flag does not error: it is swallowed into the positional")
        print("argument, so the command 'works' and answers the wrong question.")
        return 1

    print("OK - every documented ai_cli flag is accepted by its subcommand ({} files).".format(scanned))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
