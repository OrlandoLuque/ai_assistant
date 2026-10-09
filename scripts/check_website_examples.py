#!/usr/bin/env python3
"""The website's Rust examples must name items the crate actually has.

WHY THIS EXISTS
---------------
V340 found that ```rust fences in Markdown were a third population of example
code that nothing compiled or checked: 86 of 1037 imported names were wrong, 48
at no path at all, two of them on the README's front page.
`scripts/check_doc_imports.py` closed that.

**The website is a fourth population, and it was in the same state.** The same
examples live again as `<pre><code>` in `ai_assistant-website`, where no gate
reaches them -- and it is the copy a stranger reads first, before the repository.

RELATIONSHIP WITH check_doc_imports.py
--------------------------------------
This file does NOT reimplement the resolver. It imports `Crate`, `split_use`
and `crate_surface` from `check_doc_imports` and only changes where the text
comes from: HTML instead of Markdown.

That is deliberate and it is the lesson of the RUSTSEC ignore list, which lived
in three files with two different checkers: the narrower one answered a smaller
question than it appeared to. One resolver, two front-ends, no chance of the
two drifting into disagreeing about what the crate contains.

WHY IT IS NOT IN CI (yet)
-------------------------
The website is a separate repository, so a CI job here cannot see it without
cloning. That is a real decision with a trade-off (it couples this CI to the
other repo staying public), tracked as N142 together with the same question for
the front-page numbers. Until then this is a manual step, declared as such in
`check_checkers_documented.py`.

USAGE
-----
    python scripts/check_website_examples.py ../ai_assistant-website
    python scripts/check_website_examples.py ../ai_assistant-website --list
"""

from __future__ import annotations

import html
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_doc_imports import crate_surface, split_use  # noqa: E402


def code_blocks(page: pathlib.Path) -> list[tuple[int, str]]:
    """`<pre>` contents with the line each one starts on.

    Tags are stripped and entities decoded because the markup carries syntax
    highlighting: `ai_assistant::<span class="k">rag</span>` is one path to a
    reader and three tokens to a regex.
    """
    text = page.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", text, re.S):
        line = text.count("\n", 0, m.start()) + 1
        body = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        out.append((line, body))
    return out


# `use ai_assistant::…;` and `use ai_assistant_core::…;`. The crate on crates.io
# is the second name for a subset of the same tree, so a path that is wrong is
# wrong under either.
USE_RE = re.compile(r"\buse\s+(ai_assistant(?:_core)?::[^;]+);", re.S)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().split("\n\n")[-1])
        return 1
    site = pathlib.Path(sys.argv[1])
    if not site.is_dir():
        print(f"{site} is not a directory")
        return 1

    pages = sorted(p for p in site.glob("*.html") if "backup" not in p.name)
    if len(pages) < 5:
        # Pointing this at the wrong directory must fail loudly, not report a
        # clean sweep of nothing.
        print(f"ERROR: only {len(pages)} .html pages under {site} -- wrong directory?")
        return 1

    crate, mods = crate_surface()

    failures: list[str] = []
    checked = 0
    for page in pages:
        for start, body in code_blocks(page):
            for m in USE_RE.finditer(body):
                stmt = m.group(1)
                # `ai_assistant_core::` resolves against the same tree.
                normalised = stmt.replace("ai_assistant_core::", "ai_assistant::", 1)
                prefix, leaves = split_use(normalised)
                line = start + body.count("\n", 0, m.start())
                root = prefix.split("::")[0] if prefix else ""
                if root and root not in mods:
                    failures.append(
                        f"{page.name}:~{line}  ai_assistant::{prefix}  "
                        f"-- there is no `pub mod {root};` in src/lib.rs"
                    )
                    continue
                reachable = crate.reachable(prefix)
                for leaf in leaves:
                    checked += 1
                    if leaf not in reachable:
                        # Same hint the Markdown gate gives: naming where the
                        # item actually lives turns "wrong" into a fix.
                        where = sorted(
                            mod
                            for mod, names in crate.declared.items()
                            if leaf in names and mod != prefix
                        )
                        hint = (
                            f"it lives in {', '.join(where[:3])}"
                            if where
                            else "it is declared nowhere in src/"
                        )
                        failures.append(
                            f"{page.name}:~{line}  ai_assistant::"
                            f"{prefix + '::' if prefix else ''}{leaf}  -- {hint}"
                        )

    print(f"\npaginas revisadas    : {len(pages)}")
    print(f"nombres importados   : {checked}")
    print(f"que no existen       : {len(failures)}")

    if failures:
        print("\nFALLO: el sitio documenta rutas que la crate no tiene:\n")
        for f in failures:
            print(f"  {f}")
        print(
            "\nLine numbers are approximate (~): the block's start plus the offset\n"
            "inside it, which markup stripping shifts by a line or two."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
