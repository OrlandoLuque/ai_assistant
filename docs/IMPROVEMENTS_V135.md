# V135 — context-cache test flake (0.2.82)

**Status**: shipped 2026-05-11
**Scope**: test-only fix
**Runtime impact**: none

## Why

V134 turned the CI green with two exceptions: Supply Chain
(license fixes worked) and Benchmarks (budget bump worked) — but
one job, `Feature Matrix (precise-tokens)`, failed with a single
test panic out of 6231 tests:

```
thread 'context::tests::test_cached_returns_cached_value_on_second_call'
panicked at src/context.rs:315:13:
fetcher should not be called on cache hit
```

The test wasn't broken on every run — it was a race condition.
The kind that masks itself locally, where `cargo test` happens to
schedule the cache-touching tests in an order that doesn't trip
the bug. CI is more aggressive about scheduling and surfaced it.

## What was wrong

`CONTEXT_SIZE_CACHE` is a process-global `LazyLock<Mutex<HashMap<...>>>`
shared across the entire test binary:

```rust
static CONTEXT_SIZE_CACHE: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
```

Five tests in `src/context.rs` exercise it:

| Test                                                  | Acts on cache                                |
| ----------------------------------------------------- | -------------------------------------------- |
| `test_cached_uses_static_table_when_fetcher_returns_none` | calls `clear_context_size_cache()`           |
| `test_cached_uses_fetcher_when_available`             | calls `clear_context_size_cache()`           |
| `test_cached_returns_cached_value_on_second_call`     | inserts a unique key, expects hit on lookup  |
| `test_clear_context_size_cache`                       | calls `clear_context_size_cache()` + asserts |
| `test_cached_case_insensitive_key`                    | calls `clear_context_size_cache()`           |

Cargo's default test runner executes them on a thread pool. With
N≥4 cores, four of these can be in flight simultaneously. The
losing interleaving is:

1. **T1** runs `test_cached_returns_cached_value_on_second_call`:
   inserts key `"cache-second-call-test-xyzzy-42" → 99_999`.
2. **T2** runs (say) `test_clear_context_size_cache`: calls
   `clear_context_size_cache()`. The HashMap is now empty.
3. **T1** does its second lookup. Cache miss. Fetcher invoked.
   Fetcher is `|_| panic!("fetcher should not be called on cache hit")`.
   Test fails.

The author of the test foresaw exactly this — there's an in-line
comment explaining why they don't call `clear_context_size_cache()`
themselves. But that defence is incomplete: any *other* test
calling `clear` between the two lookups is just as fatal.

## What changed

Added a test-only mutex that serialises every cache-touching test:

```rust
#[cfg(test)]
mod tests {
    static CACHE_TEST_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

    #[test]
    fn test_cached_returns_cached_value_on_second_call() {
        let _guard = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // … existing test body
    }

    // same `_guard` added to the four sibling tests
}
```

Why `unwrap_or_else(|p| p.into_inner())` instead of `unwrap()`:
mutex poisoning. If a *different* cache test panics for an
unrelated reason while holding this lock, every subsequent
cache test would refuse to acquire it and report a confusing
poison error instead of running. `into_inner()` ignores the poison
and proceeds; we're using the mutex purely for ordering, not for
protecting invariants on shared state, so poison recovery is
benign.

## What I considered and rejected

- **Refactor each test to use unique keys, drop the `clear` calls.**
  Doesn't work for `test_clear_context_size_cache`, which is
  *literally* testing the clear function and asserts
  `context_size_cache_len() == 0`. That assertion is also racy
  (parallel tests can insert before the assertion fires) so
  serialising via the lock is the right shape regardless.
- **Use the `serial_test` crate.** Adds a dependency and a
  proc-macro for a five-line problem.
- **Make the cache thread-local.** Changes production code for a
  test-only concern. The cache is shared global state by design.

## Verification

Locally, after the change:

```
$ for i in 1..5; do cargo test --features precise-tokens \
    --lib context::tests::test_cached; done
test result: ok. 4 passed; 0 failed; … finished in 0.00s   (×5)
$ cargo test --features precise-tokens --lib context::tests
test result: ok. 24 passed; 0 failed; …
$ cargo clippy --features precise-tokens --lib -- -D warnings
Finished `dev` profile … 46.13s   (no warnings)
```

CI verification on the next push.

## Out of scope

The four `api_key_rotation` / `distributed::tests::test_replica_tracking`
flakes documented in V133's working notes follow the same pattern
(global state + parallel test runner) but live in different
modules with different invariants. They aren't part of this fix.
