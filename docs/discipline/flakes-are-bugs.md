# Test Flakes Are Bugs

When a test is flaky, the right move is rarely `#[ignore]`. Far
more often, the test is catching a real race or boundary
condition that the production code happens to lose less often
than the test does — and silencing the test silences the
warning, not the bug.

This is a project-wide discipline, not a one-off rule. Three
landed examples make the case concrete.

## V135 — context cache race

A `ContextCache::load_or_init(key, loader)` had two callers
racing to populate the same key. Whichever lost the race
returned the *other* caller's partial state (the cache slot was
written before the loader finished), so a downstream comparison
flipped depending on schedule. The flake rate was low because
the race required the loaders to interleave on a sub-millisecond
boundary, but the production failure mode was real: a partially
populated cache entry was sometimes served as if it were
complete. Fix: serialize concurrent loaders for the same key
behind a `OnceCell`-style barrier so the loser of the race
*waits* for the winner's result.

The test got more reliable. The production code got correct.
They were the same fix.

## V135 — `NodeId` collision under churn

The DHT used a 64-bit `NodeId` derived from a per-node random
seed. Under high join/leave churn (the test that surfaced this
spun up and tore down 10 000 nodes in a tight loop) the seed
PRNG was occasionally re-seeded from `SystemTime::now()` with a
resolution coarse enough to produce identical seeds, which then
produced identical `NodeId`s. Two nodes with the same ID broke
the routing table's "node is unique per id" invariant. The
flake rate was tied to scheduler luck on the test runner.

Fix: switch the seed source to `getrandom` (kernel CSPRNG) and
add a debug-only `NodeId` collision check that panics in dev /
returns an error in production. Test now deterministic;
production now resists pathological churn.

## V136 — `ApiKey` boundary-second expiry

`ApiKey::is_valid(now)` had two call sites that disagreed on
whether to use `<` or `<=`. At any single point in time the
disagreement was invisible — the two checks happened in
different code paths and rarely fired in the same second. But at
the *exact* expiry instant, one path said "expired" and the
other said "still valid", and an authenticated request could
slip past one boundary into the other. The test that surfaced
this happened to run in CI right at second-boundary clock ticks
often enough to flake once a week.

Fix: a single `ApiKey::expires_at_or_before(now)` predicate
used by every caller, with the boundary semantics documented in
one place. Tests stable; the auth boundary is now defined by
*one* line of code.

## The pattern

In all three cases, the surface symptom was "this test is flaky,
add a retry or ignore it." The actual issue was a real bug whose
production failure mode was less obvious than the test failure.
The retry would have hidden both the test failure *and* the
production bug — and the production bug would have surfaced
later as a customer report, with much less context to diagnose.

**Default to: assume the test is right and the code is wrong.**
Only fall back to retry/ignore after exhausting the alternative.

## What to do when you hit a flake

1. Run the test in a loop locally with output captured. Get a
   reproduction — flake rate, conditions, payload that triggers
   it.
2. Read the test and the system under test together. Look for
   shared mutable state, time-based predicates, ordering
   assumptions, network or filesystem dependencies.
3. Write a deterministic reproduction. If you can't write one,
   you don't yet understand the bug.
4. Fix the production code so the deterministic reproduction
   passes.
5. Keep the original (non-deterministic) test as a regression
   guard. The fix should make it pass deterministically too.

`#[ignore]` is appropriate when the flake is genuinely
environmental — a network test that can't reach the internet on
an air-gapped runner, for example. It is *not* appropriate as a
substitute for understanding why the test fails sometimes.
