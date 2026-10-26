#!/usr/bin/env python3
"""Every `#[non_exhaustive]` input struct must be configurable in one expression.

WHAT `#[non_exhaustive]` ACTUALLY FORBIDS — measured, not remembered
-------------------------------------------------------------------
Two tiny crates, 2026-10-27, because the first version of this gate was written
on a belief and the belief was too strong:

    // crate B, consuming crate A
    let _y = a::PubFields { a: 1, ..Default::default() };   // error[E0639]
    let mut z = a::PubFields::default();  z.a = 1;          // COMPILES AND RUNS

So the attribute forbids the struct **expression**, including with functional
update syntax. It does **not** forbid assigning to a `pub` field of a value you
already own. All 239 input structs here have only `pub` fields, so a consumer was
never locked out of the ones that have a `Default` — what they were locked out of
is doing it in a single expression:

* no `..Default::default()`, ever;
* no construction in argument position — `f(Config::default().with_x(1))` had to
  become three statements and a `mut` binding;
* and nothing at all if no function hands you a first value.

That last one is the hard lockout, and it is a different class: `MockHttpServer`
(N159) had no `Default` and no constructor, so a consumer could not get one at
any price. `VectorDbConfig` (N160) had `Default`, so `dimensions` was reachable
with two statements — awkward, not impossible. Keeping those two apart is the
whole point of the `NO_SEED` class below.

WHY A GATE AND NOT A TEST
-------------------------
`#[non_exhaustive]` does not apply inside the declaring crate, so a unit test
builds these with a literal and passes. The instruments that can see any of this
are doctests (they compile as a separate crate) and a real consumer crate. A gate
is cheaper than either and runs on every push.

And the bulk fix is not the valuable half. N6 (V247) already did this once and
closed over 58 of ~290 types; **one** new config shipped unconfigurable puts us
back. What the gate adds is coverage: every public field needs a setter, so a
config that grows a field without growing its setter fails here instead of being
found by a consumer months later.

WHAT IT CHECKS
--------------
For every `#[non_exhaustive] pub struct` under `src/` (excluding `src/bin/`,
whose crates have no external consumers):

1. is it an **input** — does some `pub fn` take it as an argument?
2. can a consumer get a **first value**: `Default`, a `*Builder`, or any public
   associated function that returns the type?
3. can they then **set every public field**, in one expression?

THE CLASSES
-----------
* **NO_SEED** (fails) — nothing returns a value of this type and there is no
  `Default`. The hard lockout: setters would not help.
* **MISSING** (fails) — a first value is reachable and nothing can be changed in
  one expression.
* **PARTIAL** (fails) — has setters, but not for every public field. The
  divergence case: the struct grew and the setter list did not.
* **BUILDER** (ok) — has a dedicated `*Builder`. The author's decision of
  2026-10-26: the 17 that already have one keep it. A builder earns its extra
  type when there are required fields or validation; these configs are
  all-optional with a `Default`, so setters are the cheaper half of the contract.
* **SETTERS** (ok) — full coverage.
* **NEW** (ok) — a `new` that takes arguments. Whether those arguments reach
  every field is not something a regex can answer; what it *can* say is that a
  consumer is not locked out.
* **OUTPUT** (informational) — nothing public takes it as an argument, so a
  consumer only ever reads one. 77 of them. Not a defect, and listed rather than
  hidden so that a wrong classification can be argued with.

USAGE
-----
    python scripts/check_config_constructible.py             # the gate
    python scripts/check_config_constructible.py --list      # everything
    python scripts/check_config_constructible.py --self-test

The baseline lives in `scripts/config_constructible_baseline.toml`, in two
sections because "I looked at this and it is fine" and "something has to happen
and here is what" are different claims: `[allowed]` and `[pending]`, each needing
a reason per name. The pending count is a ratchet: it may go down, never up, and a
name that has been fixed must be removed — a baseline that keeps names which no
longer fail is how a gate starts lying.
"""

import pathlib
import re
import sys

NL = chr(10)
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BASELINE = ROOT / "scripts" / "config_constructible_baseline.toml"

# `impl` is NOT at column 0 when the type lives inside a module, and a
# `^impl`-anchored regex silently reported 7 types as having no `Default` at all
# while `impl Default for VadConfig` sat four spaces in. Same trap as the
# column-0 regexes in the doc-example gate.
IMPL = re.compile(r"^[ \t]*impl(?:<[^>]*>)?\s+"
                  r"(?:(?P<trait>[\w:]+)(?:<[^>]*>)?\s+for\s+)?"
                  r"(?P<name>\w+)(?:<[^>]*>)?\s*(?:where[^{]*)?\{", re.M)
STRUCT = re.compile(r"(?P<attrs>(?:#\[[^\]]*\]\s*)*)pub struct (?P<name>\w+)"
                    r"(?P<generics><[^>]*>)?\s*(?P<open>[{;(])")
PUBFN = re.compile(r"pub (?:const )?(?:async )?(?:unsafe )?fn \w+(?:<[^>]*>)?\s*\(")
SETTERS_MACRO = re.compile(r"with_setters!\(\s*(?P<name>\w+)\s*\{")
FIELD = re.compile(r"\bpub (?P<name>\w+)\s*:")


def balanced(text, i, open_c="{", close_c="}"):
    """Content from `i` (just past the opening delimiter) to its match.

    `>` after `-` or `=` closes nothing: `Fn(..) -> f64` is not a generic.
    """
    depth = 1
    j = i
    prev = ""
    n = len(text)
    while j < n and depth:
        c = text[j]
        if c == open_c:
            depth += 1
        elif c == close_c and not (c == ">" and prev in "-="):
            depth -= 1
        prev = c
        j += 1
    return text[i:j - 1]


SETTER_FN = re.compile(r"pub fn (?:with_|set_)\w+\s*\(")
# Una funcion asociada publica que DEVUELVE el tipo: eso es de donde sale el
# primer valor. No vale mirar solo `new`: `HardwareInfo::detect()` produce uno y
# no se llama `new`, y sin esto la puerta habria dado por buena una lista
# completa de setters sobre un tipo del que no se puede obtener un valor.
PRODUCER = re.compile(r"pub (?:const )?(?:async )?fn (\w+)\s*\(([^)]*)\)\s*->\s*([^{;]+)")


def hand_written_setters(impl_body):
    """Fields a hand-written setter assigns, read from the setter's body.

    Not from its *name*: `AiConfig::with_model` sets `selected_model`, and a
    name-only rule would have called that field uncovered and demanded a second
    setter for it. What makes a field settable is an assignment, so that is what
    gets looked at.
    """
    covered = set()
    for m in SETTER_FN.finditer(impl_body):
        after = impl_body[m.end():]
        brace = after.find("{")
        if brace < 0:
            continue
        body = balanced(after, brace + 1)
        covered.update(re.findall(r"self\.(\w+)\s*(?:=[^=]|\.push\b|\.insert\b)",
                                  body))
        # La otra forma, que no asigna nada: `Self { campo, ..Default::default() }`
        # --- un constructor que se llama `with_*`. `DiskSpillConfig::with_threshold`
        # y `RrfOptions::with_k` son asi, y una regla que solo mire asignaciones
        # los habria declarado sin camino.
        for lit in re.finditer(r"\bSelf\s*\{", body):
            inner = balanced(body, lit.end())
            for part in inner.split(","):
                part = part.strip()
                if not part or part.startswith(".."):
                    continue
                name = part.split(":", 1)[0].strip()
                if re.fullmatch(r"\w+", name):
                    covered.add(name)
    return covered


def producers_of(impl_body, name):
    """Public associated functions that hand back a value of this type.

    An associated function (no `self` receiver) whose return type mentions
    `Self` or the type's own name. `HardwareInfo::detect()` is one; a rule that
    only knew about `new` would have accepted a complete list of setters on a
    type nobody can get a value of.
    """
    out = []
    for m in PRODUCER.finditer(impl_body):
        fname, args, ret = m.group(1), m.group(2), m.group(3)
        if re.match(r"\s*&?\s*(mut\s+)?self\b", args):
            continue
        if re.search(r"\bSelf\b|\b" + name + r"\b", ret):
            out.append(fname)
    return out


def scan(src=None):
    """Returns one row per `#[non_exhaustive] pub struct`, worst class first."""
    src = pathlib.Path(src) if src else SRC
    impls, traits, decls, params, macro_setters = {}, {}, {}, [], {}

    files = [p for p in sorted(src.rglob("*.rs"))
             if "bin" not in p.relative_to(src).parts[:1]]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(src.parent)).replace(chr(92), "/")

        for m in IMPL.finditer(text):
            body = balanced(text, m.end())
            if m.group("trait"):
                traits.setdefault(m.group("name"), set()).add(
                    m.group("trait").split("::")[-1])
            else:
                impls.setdefault(m.group("name"), []).append(body)

        for m in STRUCT.finditer(text):
            body = balanced(text, m.end()) if m.group("open") == "{" else ""
            decls.setdefault(m.group("name"),
                             (rel, m.group("attrs"), body, m.group("generics")))

        for m in SETTERS_MACRO.finditer(text):
            body = balanced(text, m.end())
            got = set(re.findall(r"with_\w+\s*->\s*(\w+)\s*:", body))
            macro_setters.setdefault(m.group("name"), set()).update(got)

        for m in PUBFN.finditer(text):
            params.append(balanced(text, m.end(), "(", ")"))

    joined = NL.join(params)
    rows = []
    for name, (rel, attrs, body, generics) in sorted(decls.items()):
        if "non_exhaustive" not in attrs:
            continue
        own = NL.join(impls.get(name, []))
        # Sin lineas de comentario: un `/// ... pub algo: ...` en la documentacion
        # de un campo contaria como campo.
        no_comments = NL.join(ln for ln in body.split(NL)
                              if not ln.lstrip().startswith("//"))
        fields = set(FIELD.findall(no_comments))
        hand = hand_written_setters(own)
        from_macro = macro_setters.get(name, set())
        setters = hand | from_macro
        news = re.findall(r"pub (?:const )?fn (new\w*)\s*\(([^)]*)\)", own)
        new_args = any(a.strip() not in ("", "&self", "self", "&mut self")
                       for _, a in news)
        producers = producers_of(own, name)
        has_default = (bool(re.search(r"derive\([^)]*\bDefault\b", attrs))
                       or "Default" in traits.get(name, set()))
        builder = (name + "Builder") in decls
        is_input = bool(re.search(
            r"(?:[:<(&,]|\bimpl Into<)\s*&?(?:mut )?(?:'\w+ )?" + name + r"\b",
            joined))

        # De donde sale el PRIMER valor. Sin esto no hay setter que sirva.
        reachable = has_default or builder or bool(producers)

        if not is_input:
            cls = "OUTPUT"
        elif not reachable:
            cls = "NO_SEED"
        elif builder:
            cls = "BUILDER"
        elif fields and setters >= fields:
            cls = "SETTERS"
        elif setters:
            cls = "PARTIAL"
        elif new_args:
            cls = "NEW"
        else:
            cls = "MISSING"

        rows.append({
            "name": name, "file": rel, "class": cls,
            "fields": sorted(fields), "setters": sorted(setters),
            "missing": sorted(fields - setters),
            "default": has_default, "generics": generics or "",
        })
    return rows


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def load_baseline(path=None):
    """`[allowed]` and `[pending]`, both `name = "reason"`, plus `ratchet = N`.

    A reason is required in **both** sections, which is the point: a name with
    no argument next to it reads as reviewed to whoever finds it next.
    """
    path = pathlib.Path(path) if path else BASELINE
    allowed, pending, ratchet = {}, {}, None
    if not path.exists():
        return allowed, pending, ratchet
    section = None
    for raw in path.read_text(encoding="utf-8").split(NL):
        line = "" if raw.lstrip().startswith("#") else raw.strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"')
        if key == "ratchet":
            ratchet = int(value)
        elif section == "allowed":
            allowed[key] = value
        elif section == "pending":
            pending[key] = value
    return allowed, pending, ratchet


def main():
    argv = sys.argv[1:]
    rows = scan()
    allowed, pending_baseline, ratchet = load_baseline()
    failing = [r for r in rows if r["class"] in ("MISSING", "PARTIAL", "NO_SEED")]

    if "--list" in argv:
        for cls in ("NO_SEED", "MISSING", "PARTIAL", "NEW", "SETTERS", "BUILDER",
                "OUTPUT"):
            sel = [r for r in rows if r["class"] == cls]
            print("== %s: %d ==" % (cls, len(sel)))
            for r in sel:
                extra = ""
                if r["missing"] and cls == "PARTIAL":
                    extra = "  sin setter: " + ", ".join(r["missing"])
                print("   %-40s %-44s%s" % (r["name"], r["file"], extra))
            print()
        return 0

    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print("`#[non_exhaustive] pub struct` en src/ (sin src/bin): %d" % len(rows))
    for cls in ("SETTERS", "BUILDER", "NEW", "OUTPUT", "PARTIAL", "MISSING",
                "NO_SEED"):
        print("   %-15s %d" % (cls, counts.get(cls, 0)))
    print()

    names_failing = {r["name"] for r in failing}
    stale = [n for n in list(pending_baseline) + list(allowed)
             if n not in names_failing]
    new = [r for r in failing
           if r["name"] not in pending_baseline and r["name"] not in allowed]
    unargued = [n for n, why in list(allowed.items()) + list(pending_baseline.items())
                if not why.strip()]

    bad = False
    if new:
        bad = True
        print("FALLO - %d tipo(s) de ENTRADA sin camino de construccion, y no"
              " estan en el baseline:" % len(new))
        for r in new:
            if r["class"] == "NO_SEED":
                why = "NADA devuelve un valor de este tipo (ni Default, ni productor)"
            elif r["class"] == "MISSING":
                why = "se obtiene el valor por defecto y no se puede cambiar nada"
            else:
                why = "setters incompletos: falta " + ", ".join(r["missing"])
            print("   %-38s %-40s %s" % (r["name"], r["file"], why))
        print()
        print("   Arreglo: `with_setters!` junto al struct, un setter por campo")
        print("   publico. Ver scripts/check_config_constructible.py y la")
        print("   seccion de `with_setters!` en docs/GUIDE.md.")
        print()
    if stale:
        bad = True
        print("FALLO - %d nombre(s) del baseline ya NO fallan: quitalos."
              % len(stale))
        print("   " + ", ".join(sorted(stale)))
        print("   Un baseline que guarda nombres arreglados deja de medir nada.")
        print()
    if unargued:
        bad = True
        print("FALLO - entrada del baseline sin motivo escrito: %s"
              % ", ".join(sorted(unargued)))
        print()

    pending = [r["name"] for r in failing if r["name"] in pending_baseline]
    if ratchet is not None and len(pending) > ratchet:
        bad = True
        print("FALLO - el ratchet dice %d pendientes y hay %d."
              % (ratchet, len(pending)))
        print()

    if bad:
        return 1
    print("OK - %d de entrada con camino, %d salidas, %d pendientes en el"
          " baseline (ratchet %s)."
          % (counts.get("SETTERS", 0) + counts.get("BUILDER", 0)
             + counts.get("NEW", 0),
             counts.get("OUTPUT", 0), len(pending), ratchet))
    return 0


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
SELF_TEST = {
    "ok_setters.rs": '''
        #[non_exhaustive]
        #[derive(Default)]
        pub struct Good { pub a: usize, pub b: String }
        with_setters!(Good {
            with_a -> a: usize,
            with_b -> b: String,
        });
        pub fn takes(_c: Good) {}
    ''',
    "partial.rs": '''
        #[non_exhaustive]
        #[derive(Default)]
        pub struct Half { pub a: usize, pub grew: bool }
        with_setters!(Half {
            with_a -> a: usize,
        });
        pub fn takes_half(_c: &Half) {}
    ''',
    "missing.rs": '''
        #[non_exhaustive]
        #[derive(Default)]
        pub struct Locked { pub a: usize }
        pub fn takes_locked(_c: Locked) {}
    ''',
    "indented.rs": '''
        pub mod inner {
            #[non_exhaustive]
            #[derive(Default)]
            pub struct Nested { pub a: usize }
            impl Nested {
                pub fn with_a(mut self, v: usize) -> Self { self.a = v; self }
            }
            pub fn takes_nested(_c: Nested) {}
        }
    ''',
    "output_only.rs": '''
        #[non_exhaustive]
        pub struct Report { pub total: usize }
        pub fn produce() -> Report { Report { total: 0 } }
    ''',
    "no_seed.rs": '''
        #[non_exhaustive]
        pub struct Orphan { pub a: usize }
        impl Orphan {
            pub fn with_a(mut self, v: usize) -> Self { self.a = v; self }
        }
        pub fn takes_orphan(_c: Orphan) {}
    ''',
    "seeded.rs": '''
        #[non_exhaustive]
        pub struct Detected { pub cores: usize }
        impl Detected {
            pub fn detect() -> Self { Detected { cores: 1 } }
            pub fn with_cores(mut self, v: usize) -> Self { self.cores = v; self }
        }
        pub fn takes_detected(_c: Detected) {}
    ''',
    "arrow_type.rs": '''
        #[non_exhaustive]
        #[derive(Default)]
        pub struct Cb { pub f: Option<Box<dyn Fn(&str) -> f64>> }
        with_setters!(Cb {
            with_f -> f: Option<Box<dyn Fn(&str) -> f64>>,
        });
        pub fn takes_cb(_c: Cb) {}
    ''',
}


def self_test():
    import tempfile
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "src"
        src.mkdir()
        for name, body in SELF_TEST.items():
            (src / name).write_text(body, encoding="utf-8")
        got = {r["name"]: r["class"] for r in scan(src)}

    expect = {
        "Good": "SETTERS",       # cobertura completa
        "Half": "PARTIAL",       # el campo `grew` no tiene setter
        "Locked": "MISSING",     # Default y nada mas
        "Nested": "SETTERS",     # `impl` indentado dentro de un modulo
        "Report": "OUTPUT",      # nadie lo toma como argumento
        "Cb": "SETTERS",         # un tipo con `->` dentro
        # Setters completos y NADA que devuelva un valor: inservible.
        "Orphan": "NO_SEED",
        # Lo mismo pero con un productor que no se llama `new`.
        "Detected": "SETTERS",
    }
    for name, want in expect.items():
        if got.get(name) != want:
            failures.append("%s: esperaba %s, dio %s" % (name, want, got.get(name)))

    if failures:
        print("SELF-TEST FALLA:")
        for f in failures:
            print("   " + f)
        return 1
    print("OK - self-test: %d casos, incluido el `impl` indentado y un tipo"
          " con `->`." % len(expect))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
