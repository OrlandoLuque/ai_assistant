#!/usr/bin/env python3
"""Ratchet gate: the docs must name types and paths the crate actually has.

Three populations of example code exist in this repository and until V340 only
two of them were checked by anything:

* ``examples/*.rs``     -- compiled by ``cargo clippy --all-targets`` (V247).
* ``///`` doctests      -- compiled by ``cargo test --doc`` (V317).
* ```rust fences in .md -- **nobody**.

The third is the one a reader meets first: ``README.md`` and ``docs/GUIDE.md``
open with code blocks, and ``GUIDE.md`` describes itself as covering "every
feature in the crate ... with code examples". A ``use`` line there is the least
ambiguous claim documentation can make -- it names an item *and* the module it
lives in -- and it was free to be wrong in both halves at once.

Why static and not a compile gate
---------------------------------
Compiling the extracted imports would be stronger, and it was tried first. It
cannot work here: ``full`` turns on 25 of the 95 features, so a compile run
reports every type behind the other 70 as missing, and ``--all-features`` does
not build at all (``vector-lancedb`` needs ``protoc``). A gate that calls
``a2a_protocol`` non-existent because nobody enabled ``a2a`` would be worse than
no gate -- it would teach you to ignore it.

The three checks below are feature-independent by construction, because whether
a name is *declared* in the tree does not depend on which features are on. What
does depend on features is whether it *compiles*, and that is a different (and
much softer) documentation question: saying which feature a type needs.

Checks
------
1. **Module path exists.** The first segment of ``ai_assistant::X::…`` must be a
   ``pub mod X;`` in ``src/lib.rs``. This is what catches a documented path that
   can never resolve under any feature set -- ``ai_assistant::voice`` when the
   module is ``voice_agent``, or ``ai_assistant::context`` when ``context`` is
   ``mod``, not ``pub mod``.
2. **Leaf name is reachable at that path.** Not merely "exists somewhere": the
   name must be declared in the named module or re-exported by it. The first
   version of this gate only checked existence-anywhere and therefore passed
   ``evaluation::LlmJudge`` (which lives in ``llm_judge``) and
   ``advanced_memory::SearchQuery`` (which is one module deeper) -- three real
   errors that ``rustc`` found afterwards and the gate had blessed.

   Re-export chains must be followed to do this, including globs:
   ``advanced_memory/mod.rs`` is thirteen ``pub use x::*;`` lines, so
   ``advanced_memory::EpisodicStore`` is correct even though the struct is
   declared in ``advanced_memory/episodic.rs``. An earlier attempt ignored globs
   and called that an error.
3. **``**Key types**:`` lines name real types.** The same claim in sentence
   form, and the one the guide repeats per feature section. Twenty-two of them
   named types that do not exist -- `MemoryBus`, `TurnDetector`, `JudgeScore`,
   `BreakpointType` -- and two of those sentences described a *mechanism* that
   does not exist either ("agents publish memories to the bus and subscribe to
   memory types they care about": there is no bus, it is a pool you read).

   Scoped to those lines on purpose. A backticked CamelCase word anywhere else
   is as likely to be LangChain's ``ChatOpenAI``, a Windows scheduled task, or
   ``OnceCell`` as it is to be ours -- 24 such mentions are legitimate. On a
   ``**Key types**:`` line the false-positive rate was zero out of twenty-two,
   because that line makes a claim about *this* crate and nothing else.

Both regexes allow leading whitespace. An earlier version of this file anchored
declarations at column 0 and silently reported every type declared inside an
inner ``mod`` block as missing -- including ``VoiceAgent`` and ``WorkflowGraph``,
which exist. See the note in docs/modus-operandi.md about instruments that
under-report; this is the fifth one caught this month, and the counts printed
below exist so that a broken regex shows up as an absurd total instead of a
clean bill of health.
"""
from __future__ import annotations

import os
import re
import sys

# Ratchet. Lower it when you fix imports; never raise it. Every remaining
# failure must be listed in KNOWN below with a reason.
BASELINE = 0

SRC = "src"

# Docs that record the past rather than describe the present: the project
# treats docs/IMPROVEMENTS_V*.md and the dated audits as history, not state
# (see CLAUDE.md). A name in a February design note is not a live promise.
HISTORICAL = re.compile(
    r"(IMPROVEMENTS_V|SECURITY_AUDIT|_AUDIT_|AUDIT_V|CHANGELOG|PENDING|_V\d+\.md$|ARCHIVE)",
    re.I,
)

# Documents that describe what is *proposed*, not what exists. Each one says so
# in its own header; the reason is recorded here so the exemption is reviewable.
DESIGN_DOCS = {
    # "Documento de diseno para convertir ai_assistant de un framework de
    # patrones en un agente de ejecucion real" -- dated 2026-02-15. Names in it
    # are a plan; some were built under other names, some never were.
    "docs/AGENT_SYSTEM_DESIGN.md",
    # Wiring plan: lists the widgets a future GUI would need.
    "docs/GUI_FULL_WIRING_PLAN.md",
    # Feature-lifecycle process doc; its identifiers are hypothetical examples
    # ("OldRagBackend") illustrating the deprecation policy.
    "docs/FEATURE_LIFECYCLE.md",
}

# Imports that fail a check on purpose. A suppression is a claim that the
# failure is not a defect, and a claim with no argument is not reviewable, so
# each entry carries its reason. Format: exact `use` statement -> reason.
KNOWN: dict[str, str] = {}


def live_docs() -> list[str]:
    out: list[str] = []
    for base, _dirs, files in os.walk("docs"):
        if ".git" in base:
            continue
        out += [
            os.path.join(base, f).replace("\\", "/") for f in files if f.endswith(".md")
        ]
    out += [f for f in os.listdir(".") if f.endswith(".md")]
    return sorted(
        f for f in out if not HISTORICAL.search(f) and f not in DESIGN_DOCS
    )


def rust_fences(text: str):
    """Yield (first_line_number, chunk) for each ```rust fence."""
    inside, buf, start = False, [], 0
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not inside and (stripped == "```rust" or stripped.startswith("```rust,")):
                inside, buf, start = True, [], i + 1
            elif inside:
                yield start, "\n".join(buf)
                inside = False
            continue
        if inside:
            buf.append(line)
    if inside:
        yield start, "\n".join(buf)


USE = re.compile(r"^[ \t]*use\s+(ai_assistant::[^;]+);", re.M)

# `pub struct|enum|trait|union|fn|const|static|type|mod` at any indentation.
DECL = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+(?:unsafe\s+|async\s+|extern\s+\"[^\"]*\"\s+)*"
    r"(?:struct|enum|trait|union|fn|const|static|type|mod)\s+([A-Za-z_]\w*)",
    re.M,
)
REEXPORT = re.compile(r"pub use ([^;]+);")

# Two humps or more: `FolderWatcher`, not `Html` or `Rst`. A single-hump word in
# backticks is far more often prose ("`Markdown`") than a type name, and the
# point of check 3 is a zero-false-positive rule.
KEY_TYPE = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+$")


def module_of(path: str) -> str:
    """src/advanced_memory/episodic.rs -> advanced_memory::episodic"""
    m = path.replace("\\", "/")[len(SRC) + 1 : -len(".rs")]
    if m.endswith("/mod"):
        m = m[: -len("/mod")]
    return "" if m == "lib" else m.replace("/", "::")


class Crate:
    """Which names each module offers, following `pub use` chains."""

    def __init__(self) -> None:
        self.declared: dict[str, set[str]] = {}
        self.named: dict[str, set[str]] = {}   # module -> re-exported names
        self.globs: dict[str, set[str]] = {}   # module -> modules glob'd in
        self.files = 0
        self._cache: dict[str, set[str]] = {}

    def load(self) -> None:
        for base, _dirs, fs in os.walk(SRC):
            for f in fs:
                if not f.endswith(".rs"):
                    continue
                self.files += 1
                p = os.path.join(base, f)
                mod = module_of(p)
                text = open(p, encoding="utf-8", errors="replace").read()
                self.declared.setdefault(mod, set()).update(DECL.findall(text))
                named = self.named.setdefault(mod, set())
                globs = self.globs.setdefault(mod, set())
                for body in REEXPORT.findall(text):
                    body = " ".join(
                        "\n".join(
                            re.sub(r"//.*$", "", ln) for ln in body.split("\n")
                        ).split()
                    )
                    target = body.split("{")[0].strip().rstrip(":").rstrip(":")
                    target = re.sub(r"^(crate|self|super)::", "", target).strip("::")
                    if body.rstrip().endswith("*") or body.rstrip().endswith("::*"):
                        # `pub use x::*;` -- every public name of x lands here.
                        src_mod = target.rstrip("*").strip(":").strip()
                        globs.add(self._absolute(mod, src_mod))
                        continue
                    if "{" in body:
                        inner = body.split("{", 1)[1].rsplit("}", 1)[0]
                        pieces = [x.strip() for x in inner.split(",")]
                    else:
                        pieces = [body.split("::")[-1]]
                    for piece in pieces:
                        piece = piece.split(" as ")[-1].strip()
                        piece = piece.split("::")[-1].strip()
                        if piece and re.match(r"^[A-Za-z_]\w*$", piece):
                            named.add(piece)

    @staticmethod
    def _absolute(here: str, target: str) -> str:
        """A glob target is relative to the module doing the re-export."""
        if not target:
            return here
        if target in ("*",):
            return here
        # `pub use episodic::*;` inside advanced_memory/mod.rs means
        # advanced_memory::episodic.
        return f"{here}::{target}" if here else target

    def reachable(self, mod: str, seen: set[str] | None = None) -> set[str]:
        if mod in self._cache:
            return self._cache[mod]
        seen = seen or set()
        if mod in seen:
            return set()
        seen.add(mod)
        out = set(self.declared.get(mod, ())) | set(self.named.get(mod, ()))
        for g in self.globs.get(mod, ()):
            out |= self.reachable(g, seen)
            # A glob into a directory module also reaches what that module's own
            # mod.rs globs in, which the recursion above already covers.
        if len(seen) == 1:
            self._cache[mod] = out
        return out

    def known_modules(self) -> set[str]:
        return set(self.declared) | set(self.named) | set(self.globs)


def crate_surface() -> tuple[Crate, set[str]]:
    crate = Crate()
    crate.load()
    lib = open(os.path.join(SRC, "lib.rs"), encoding="utf-8", errors="replace").read()
    mods = set(re.findall(r"^\s*pub mod ([a-z_][a-z0-9_]*);", lib, re.M))
    root = crate.reachable("")
    # An absurd count means the regexes stopped matching, not that the crate
    # shrank. Fail loudly rather than hand back a green tick.
    if crate.files < 400:
        sys.exit(f"ERROR: solo {crate.files} ficheros .rs bajo {SRC}/ -- el barrido no funciona")
    if len(root) < 1000:
        sys.exit(f"ERROR: solo {len(root)} nombres en la raiz -- el resolutor no funciona")
    if len(mods) < 200:
        sys.exit(f"ERROR: solo {len(mods)} `pub mod` en lib.rs -- el regex de modulos no funciona")
    print(f"ficheros .rs leidos  : {crate.files}")
    print(f"modulos con contenido: {len(crate.known_modules())}")
    print(f"nombres en la raiz   : {len(root)}")
    print(f"`pub mod` en lib.rs  : {len(mods)}")
    return crate, mods


def split_use(path: str) -> tuple[str, list[str]]:
    """`ai_assistant::a::{B, C as D}` -> ("a", ["B", "C"])."""
    body = "\n".join(re.sub(r"//.*$", "", ln) for ln in path.split("\n"))
    body = " ".join(body.split())
    body = body[len("ai_assistant::"):]
    if "{" in body:
        prefix = body.split("{", 1)[0].strip().rstrip(":")
        inner = body.split("{", 1)[1].rsplit("}", 1)[0]
        raw = [x.strip() for x in inner.split(",")]
    else:
        parts = [p.strip() for p in body.split("::")]
        prefix, raw = "::".join(parts[:-1]), [parts[-1]]
    leaves = []
    for r in raw:
        r = r.split(" as ")[0].strip()
        if r and r != "self" and re.match(r"^[A-Za-z_]\w*$", r):
            leaves.append(r)
    return prefix, leaves


def main() -> int:
    crate, mods = crate_surface()
    docs = live_docs()
    if len(docs) < 60:
        sys.exit(f"ERROR: solo {len(docs)} documentos vivos -- el filtro los descarta todos")

    failures: list[str] = []
    emitted: dict[str, str] = {}
    n_imports = n_leaves = 0
    for p in docs:
        text = open(p, encoding="utf-8", errors="replace").read()
        for start, chunk in rust_fences(text):
            for m in USE.finditer(chunk):
                n_imports += 1
                stmt = "use " + " ".join(
                    "\n".join(
                        re.sub(r"//.*$", "", ln) for ln in m.group(1).split("\n")
                    ).split()
                ) + ";"
                line = start + chunk[: m.start()].count("\n")
                emitted.setdefault(stmt, f"{p}:{line}")
                if stmt in KNOWN:
                    continue
                prefix, leaves = split_use(m.group(1))
                n_leaves += len(leaves)
                head = prefix.split("::")[0] if prefix else ""
                if head and head not in mods:
                    failures.append(
                        f"{p}:{line}  ai_assistant::{head}  -- no hay `pub mod {head}` en lib.rs"
                    )
                    continue
                here = crate.reachable(prefix)
                for leaf in leaves:
                    if leaf in here:
                        continue
                    where = sorted(
                        m
                        for m, ns in crate.declared.items()
                        if leaf in ns and m != prefix
                    )
                    hint = (
                        f"esta en {', '.join(where[:3])}"
                        if where
                        else "no se declara en ningun sitio de src/"
                    )
                    failures.append(
                        f"{p}:{line}  ai_assistant::"
                        f"{prefix + '::' if prefix else ''}{leaf}  -- {hint}"
                    )

    if "--emit" in sys.argv:
        # Cross-check escape hatch. This gate is static on purpose (see the
        # module docstring), but the compiler is still the authority, so make it
        # one command to hand it the same imports:
        #
        #   python scripts/check_doc_imports.py --emit examples/_probe.rs
        #   cargo check --example _probe --features "full,a2a,devtools,..."
        #
        # Every failure the compiler then reports should be a type behind a
        # feature you did not enable. If it is anything else, this gate has a
        # hole -- that is how the `evaluation::LlmJudge` class was found.
        dest = sys.argv[sys.argv.index("--emit") + 1]
        body = [
            "// GENERADO por scripts/check_doc_imports.py --emit -- no commitear.",
            "//",
            "// Un modulo por import: dos documentos pueden nombrar legitimamente el",
            "// mismo tipo por rutas distintas, y en un solo ambito eso es E0252.",
            "#![allow(unused_imports, non_snake_case)]",
            "",
        ]
        for i, (stmt, origin) in enumerate(sorted(emitted.items())):
            body.append(f"mod d{i} {{ // {origin}")
            body.append(f"    {stmt}")
            body.append("}")
        body += ["", "fn main() {}", ""]
        open(dest, "w", encoding="utf-8", newline="\n").write("\n".join(body))
        print(f"\nescritos {len(emitted)} imports en {dest}")

    # --- check 3: the `**Key types**:` lines ------------------------------
    everything = set()
    for names in crate.declared.values():
        everything |= names
    for names in crate.named.values():
        everything |= names
    # Enum variants are named on these lines too, and they are declared inside
    # the enum body rather than by a `pub` item, so take any identifier the
    # source mentions. The question here is "does this word exist in the
    # crate", not "is it importable" -- check 2 already covers importability.
    for base, _dirs, fs in os.walk(SRC):
        for f in fs:
            if f.endswith(".rs"):
                everything |= set(
                    re.findall(
                        r"[A-Za-z_]\w*",
                        open(os.path.join(base, f), encoding="utf-8", errors="replace").read(),
                    )
                )

    n_keytypes = n_keynames = 0
    for p in docs:
        for i, line in enumerate(
            open(p, encoding="utf-8", errors="replace").read().split("\n"), 1
        ):
            if "**Key types**" not in line:
                continue
            n_keytypes += 1
            for span in re.findall(r"`([^`\n]{2,80})`", line):
                head = re.split(r"[:<(\[ ]", span.strip())[0]
                if not KEY_TYPE.match(head):
                    continue
                n_keynames += 1
                if head not in everything:
                    failures.append(
                        f"{p}:{i}  **Key types** nombra `{head}`, que no existe en src/"
                    )

    print(f"documentos vivos    : {len(docs)}")
    print(f"`use` encontrados   : {n_imports}")
    print(f"nombres importados  : {n_leaves}")
    print(f"lineas **Key types**: {n_keytypes} ({n_keynames} nombres)")
    if KNOWN:
        print(f"suprimidos con motivo: {len(KNOWN)}")
    print(f"\nimports que fallan  : {len(failures)} (baseline {BASELINE})")
    for f in failures:
        print(f"  {f}")

    if len(failures) > BASELINE:
        print(
            f"\nFALLO: {len(failures)} imports roto(s), el maximo permitido es {BASELINE}.\n"
            "Arregla la ruta o el nombre. Si el documento describe algo que aun no\n"
            "existe, dilo en su cabecera y anadelo a DESIGN_DOCS, o pon el `use`\n"
            "exacto en KNOWN con el motivo escrito al lado."
        )
        return 1
    if len(failures) < BASELINE:
        print(
            f"\nBaja el BASELINE de {BASELINE} a {len(failures)} en "
            "scripts/check_doc_imports.py: la puerta es un ratchet."
        )
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
