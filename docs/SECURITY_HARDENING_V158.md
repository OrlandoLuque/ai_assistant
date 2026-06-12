# Security Hardening — V158 (2026-06-11)

Closes the last registered storage follow-up from the V155 audit and the
V157 hardening: a **per-peer storage byte quota** with attribution, so one
authenticated peer cannot monopolize a node's storage budget and starve
others. V157's absolute caps (16 MiB/value, 100k keys) already prevented
unbounded OOM; this adds *fairness* and a tighter total bound.

## What changed

The mesh key-value storage map is now wrapped in a `MeshStore` struct that
tracks, per peer, how many value bytes that peer is responsible for. The
attribution is unambiguous: **owner = the peer that sent the write** that
created or last updated the key (`Put`, `Replicate`, or anti-entropy
`SyncData`). This node's own local writes (`local_store`, replicated
`store`) are attributed to a `LOCAL_OWNER` sentinel and are **exempt** from
the per-peer quota — we trust ourselves.

### `MeshStore` (new)

```
struct MeshStore {
    entries: HashMap<String, StoredValue>,   // the data
    bytes_per_peer: HashMap<NodeId, u64>,    // running per-peer byte totals
}
```

- **Reads** go through `Deref<Target = HashMap<…>>`, so every existing
  read site (`get`, `iter`, `len`, `contains_key`, `values`, …) is
  unchanged.
- **There is deliberately no `DerefMut`.** Every mutation goes through
  `put` / `remove` / `retain_unexpired`, which maintain `bytes_per_peer`
  atomically with the map under the same lock — the counters can never
  desync. (The compiler enforces this: a raw `.insert()` on the store
  fails to compile.)
- `StoredValue` gained an `owner: NodeId` field so removals, overwrites,
  and TTL expiry decrement the *correct* peer (including an owner change
  on overwrite).

### Quota enforcement

`storage_admits(store, key, value, owner)` now also rejects a write that
would push `owner` over `MAX_BYTES_PER_PEER` (64 MiB), crediting back any
existing entry under the same key that `owner` already owns (so a same-size
overwrite is always allowed). The check is **O(1)** — a single map lookup,
no scan — so it adds no per-write cost under adversarial load. It runs on
all three peer-write paths (`Put`, `Replicate`, `SyncData`), so a peer
cannot bypass it by choosing a different message type.

With the default 50-connection cap, the per-peer quota also bounds total
peer-attributed storage at ~3.2 GiB (64 MiB × 50), turning V157's loose
absolute bound into a real one.

## Tests

- `test_storage_admits_per_peer_quota` — a peer is rejected at its cap;
  another peer and `LOCAL_OWNER` writes are unaffected; a same-size
  self-overwrite stays within quota.
- `test_meshstore_accounting_on_overwrite_and_remove` — byte counters
  track overwrite (including owner change) and removal correctly.
- The existing `test_two_nodes_connect` live handshake still passes, so
  the real Put/Replicate path works through the accounting wrapper.
- distributed_network 97 tests, full network lib 6756, harness 585/585,
  clippy 0.

## Cleanup

Fixed 2 pre-existing `must_use` warnings in `server_axum` admin-handler
tests (only visible under the network feature set the standard clippy job
doesn't cover), consistent with the V155 network-warning sweep.

## Status: storage follow-ups complete

V157 (per-value + key-count caps) + V158 (per-peer byte quota) together
cover the storage-exhaustion findings from the V155 audit. No storage
follow-ups remain open.
