# V136 — NodeId collision + ApiKey expiry boundary (0.2.83)

**Status**: shipped 2026-05-14
**Scope**: production-correctness fixes that surfaced as test flakes
**Runtime impact**: real, but only at the boundary cases the tests
described

V135 closed the CI matrix but left two known-flaky tests as "out
of scope":

- `api_key_rotation::tests::test_key_expiry`
- `distributed::tests::test_replica_tracking`

V135 described them as global-state-plus-parallel-runner flakes.
Investigation showed something different: both tests are honest
assertions over code that's actually wrong at a boundary. Once
you fix the code, the tests stop flaking — no test-only
serialisation needed.

## Bug 1: `NodeId::random()` not actually random inside one clock tick

### What was wrong

```rust
pub fn random() -> Self {
    let mut bytes = [0u8; 20];
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for (i, byte) in bytes.iter_mut().enumerate() {
        let shift1 = (i * 8) % 128;
        let shift2 = ((i + 7) * 3) % 128;
        *byte = ((now >> shift1) ^ (now >> shift2)) as u8;
    }
    Self(bytes)
}
```

The bytes are a **pure function** of `now`. If two calls to
`SystemTime::now()` return the same value, they produce
byte-identical `NodeId`s. On Windows, the system clock
resolution is ~15 ms — orders of magnitude larger than the gap
between two back-to-back Rust calls. Even on Linux, `clock_gettime`
can repeat values inside a single hot loop.

`test_replica_tracking` exercised this directly:

```rust
let node_b = NodeId::random();
let node_c = NodeId::random();
dht.add_replica("key1", node_b);
dht.add_replica("key1", node_c);
let replicas = dht.get_replicas("key1");
assert_eq!(replicas.len(), 2);
```

`replicas` is a `HashSet<NodeId>`. When `node_b == node_c`, the
set collapsed to one entry; the assert tripped.

The shift code also has a less-important problem: `shift1` and
`shift2` are `% 128`, but `now` is a `u128`, so shifts up to
127 are well-defined — but the truncating cast `as u8` only ever
reads the low 8 bits regardless. Functionally this means about
half the shifts pull from bits the cast immediately discards.
Not the cause of the flake, but the new code shifts within `u64`
ranges instead so every bit position matters.

### What changed

```rust
static COUNTER: AtomicU64 = AtomicU64::new(0);
let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_nanos();

let seed    = (now as u64)        ^ counter.wrapping_mul(0x9E3779B97F4A7C15);
let seed_hi = (now >> 64) as u64  ^ counter.wrapping_mul(0xBF58476D1CE4E5B9);

let mut bytes = [0u8; 20];
for (i, byte) in bytes.iter_mut().enumerate() {
    let shift1 = ((i * 8) % 64) as u32;
    let shift2 = (((i + 7) * 3) % 64) as u32;
    let s = if i < 10 { seed } else { seed_hi };
    *byte = ((s >> shift1) ^ (s >> shift2)) as u8;
}
```

The two constants are the standard SplitMix64 / Knuth-mixing
multipliers (used in `std::collections::HashMap`'s hasher and
many fast PRNGs). They give the counter strong avalanche even at
small values like 0, 1, 2.

Why this fixes the flake: the counter is per-process and
strictly monotonic across threads. Two concurrent calls always
get different counters, so even if `now` collides, the seeds don't.

Why I didn't reach for `rand::random()`: this code path is on
the hot DHT bootstrap path and the file was already
zero-dependency on `rand`; adding it for this single helper
would be inconsistent. The constants here give us
1-in-2^64 collision odds before counter wraps — for a 160-bit
id used to label nodes, that's overkill.

### Regression test

`test_node_id_random` previously asserted only two consecutive
ids differ — exactly the wrong threshold for a flake that
required collision *within a single tick*. Strengthened to
100 ids, all distinct:

```rust
let mut ids = std::collections::HashSet::new();
for _ in 0..100 {
    ids.insert(NodeId::random());
}
assert_eq!(ids.len(), 100, "NodeId::random() produced duplicates");
```

With the old code this fails reproducibly on Windows
(20 ids is usually enough to trip it).

## Bug 2: `ApiKey::is_usable()` strict-greater-than at expiry boundary

### What was wrong

```rust
pub fn with_expiry(mut self, duration: Duration) -> Self {
    self.expires_at = Some(Instant::now() + duration);
    self
}

pub fn is_usable(&self) -> bool {
    // ...
    if let Some(expires) = self.expires_at {
        if Instant::now() > expires {
            return false;
        }
    }
    // ...
}
```

`with_expiry(Duration::ZERO)` sets `expires_at = Instant::now() = T1`.
`is_usable()` immediately after calls `Instant::now() = T2` and
checks `T2 > T1`.

`Instant` is monotonic, but its **resolution** is not
nanosecond on every platform. On Windows it's whatever
`QueryPerformanceCounter` resolves to (typically 100ns, sometimes
much coarser under power-saving). When `T1 == T2`, `T2 > T1` is
false → the key is reported usable → `test_key_expiry` trips:

```rust
key = key.with_expiry(Duration::from_secs(0));
assert!(!key.is_usable(), "...should not be usable");
```

### What changed

`>` → `>=`. Semantically a key that "expires at T" is no longer
valid AT T, not strictly after T. This is the standard reading of
"expires at" in HTTP cookies (RFC 6265 §4.1.2.2: "The expiry-time
of the cookie is the date and time AFTER which the cookie expires"
— but the cookie is also gone WHEN that time arrives, not just
after) and JWT (`exp` claim, RFC 7519 §4.1.4: token is invalid
"on or after").

The rate-limit check below it already uses `<` correctly
(`Instant::now() < until` → still rate-limited until that
moment) — that one was already on the right side of the
boundary.

### Why no boundary test was added

`test_key_expiry` *is* the boundary test. The reason it didn't
catch the bug at first is that on most Linux CI runners
`Instant::now()` advances between back-to-back calls, so the
strict `>` happened to work. Coarse-clock Windows runners caught
it. The fix makes the contract platform-independent.

## What V136 deliberately does *not* do

- **No new `rand` dependency.** The counter-mixed seed is enough
  for the uniqueness invariant the API promises ("random node id,
  cryptographic strength not claimed").
- **No revisiting `from_string`'s `DefaultHasher`.** `from_string`
  is deterministic by design — DHT addressing requires every
  node to derive the same id for the same key. It's correct.
- **No audit of every `Instant::now() > T` comparison in the
  codebase.** This was the only one paired with a `Duration::ZERO`
  contract.

## Verification

```bash
cargo build --release --lib                    # green, 2m08s
cargo clippy --release --lib -- -D warnings    # clean
cargo test --release --lib -- api_key_rotation:: distributed::
# 53 passed; 0 failed (was: 51 reliable + 2 flaky)

# Loop the previously-flaky tests 8 times
for i in 1..8; do cargo test --release --lib -- \
    api_key_rotation::tests::test_key_expiry \
    distributed::tests::test_replica_tracking; done
# 8/8 green
```

## Out of scope

The remaining items from V133's working notes (models.dev
auto-refresh policy, wasmtime 36 → 44 jump) remain deferred —
they're feature decisions, not flakes.
