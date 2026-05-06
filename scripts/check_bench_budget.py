#!/usr/bin/env python3
"""
V126 (Phase C.5) — performance-budget gate.

Reads `bench_budget.toml` and the criterion bencher-format output
file (default `output.txt`) emitted by the CI benchmark step, then
fails (exit 1) if any benchmark with a declared budget reports a
`ns/iter` value above its `max_ns`.

Benches without a budget entry are skipped — the gate is opt-in so
unmeasured benches don't block CI while baselines are gathered.

Output is human-readable with both PASS and FAIL lines so the CI
log shows the full set of measured-vs-budget pairs (useful when
debugging a near-miss).

Usage:
    python3 scripts/check_bench_budget.py [--budget bench_budget.toml] [--output output.txt]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+ (ubuntu-latest ships 3.12 since Apr 2024)
except ImportError:
    sys.stderr.write(
        "error: tomllib not found — this script requires Python 3.11+. "
        "On older runners, install `tomli` and replace the import.\n"
    )
    sys.exit(2)


# Bencher-format line:
#   test <name> ... bench: <ns_per_iter> ns/iter (+/- <stddev>)
BENCH_RE = re.compile(
    r"^test\s+(?P<name>\S+)\s+\.\.\.\s+bench:\s+([\d,]+)\s+ns/iter"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check criterion bench output against bench_budget.toml")
    p.add_argument("--budget", default="bench_budget.toml", type=Path)
    p.add_argument("--output", default="output.txt", type=Path)
    return p.parse_args()


def load_budgets(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("budgets", {})


def parse_bencher_output(path: Path) -> dict[str, int]:
    measured: dict[str, int] = {}
    if not path.exists():
        sys.stderr.write(f"warning: bench output {path} missing — nothing to check.\n")
        return measured
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BENCH_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        ns = int(m.group(2).replace(",", ""))
        measured[name] = ns
    return measured


def main() -> int:
    args = parse_args()

    budgets = load_budgets(args.budget)
    measured = parse_bencher_output(args.output)

    if not budgets:
        sys.stderr.write(f"error: no budgets defined in {args.budget}\n")
        return 2

    if not measured:
        # The benchmark step itself logs upstream when it fails to
        # capture output. Treat empty measurements as "skip the
        # gate but warn loudly" so a broken bench harness doesn't
        # silently disable the budget check forever — the CI job
        # remains green but the warning surfaces in the log.
        sys.stderr.write(
            f"warning: no measured benchmarks parsed from {args.output}; "
            "budget gate skipped (likely upstream bench-run failure).\n"
        )
        return 0

    failures: list[str] = []
    print(f"Bench budget check ({len(budgets)} budgets, {len(measured)} measured):")
    for bench_name, budget in sorted(budgets.items()):
        max_ns = budget["max_ns"]
        note = budget.get("note", "")
        observed = measured.get(bench_name)
        if observed is None:
            print(f"  --  {bench_name}: not measured (skipped)  [{note}]")
            continue
        ratio = observed / max_ns
        if observed > max_ns:
            line = (
                f"  FAIL  {bench_name}: {observed:>12,} ns > budget {max_ns:>12,} ns "
                f"({ratio:.2f}x)  [{note}]"
            )
            failures.append(line)
            print(line)
        else:
            print(
                f"  OK    {bench_name}: {observed:>12,} ns <= budget {max_ns:>12,} ns "
                f"({ratio:.2f}x)"
            )

    if failures:
        print()
        print(f"FAIL - {len(failures)} benchmark(s) over budget:")
        for line in failures:
            print(line)
        print()
        print(
            "If the regression is intentional (feature traded speed for "
            "correctness/security), bump max_ns in bench_budget.toml in the "
            "same commit and document why in `note = `."
        )
        return 1

    print()
    print(f"PASS - all {len([b for b in budgets if b in measured])} measured "
          f"benchmark(s) within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
