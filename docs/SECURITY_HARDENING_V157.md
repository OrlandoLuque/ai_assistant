# Security Hardening — V157 (2026-06-11)

Implements the four hardening follow-ups registered by the V155 audit
(`docs/SECURITY_AUDIT_V155.md`). All four landed with code + tests; the
NodeId↔cert binding was evaluated and turned out cheap and high-value,
not the design-cycle change it was tentatively flagged as.

## 1. `can_run_command` — shell-aware command validation

`src/agent_policy.rs`. The previous check took only the FIRST word as the
"base command" and matched the deny-list by substring. A command chained
after an allowed base slipped through: `cargo build; curl evil` → base
`cargo`, allowed.

The new check:
1. **Rejects command/process substitution** (`$(...)`, backticks,
   `<(...)`, `>(...)`) outside single quotes — they smuggle arbitrary
   commands we cannot statically validate. Fail closed.
2. **Splits into segments** on shell control operators (`;`, `|`, `&`,
   newline) while respecting single/double quotes, so
   `git commit -m "a; b"` stays one segment.
3. For **every** segment: strips leading `VAR=value` env assignments,
   takes the first token's **basename** (so `/bin/rm` and `./rm` match the
   deny entry `rm`), and requires it to pass allow + deny. Any denied or
   non-allowed segment fails the whole command.

Covered by `test_can_run_command_blocks_chaining_bypass` (chaining via
`;` `&&` `||` `|` `&` newline, substitution, env-prefix, path-qualified,
plus the legitimate quoted-`;` and multi-segment-all-allowed cases).

**Note — defense in depth:** this is now a tighter boundary, but the
`Inspector` layer (V123) still runs content checks before the sandbox.
A full POSIX shell parser is out of scope; the segment splitter is
conservative (a quoted operator it can't reason about over-blocks rather
than under-blocks).

## 2. Mesh storage exhaustion guards

`src/distributed_network.rs`. Any authenticated peer could `Put` (or
`Replicate`) unbounded data — one giant value or a flood of keys — and
OOM a node. Added O(1) admission control (no per-write scan, so no DoS
amplification under load):
- `MAX_STORED_VALUE_BYTES = 16 MiB` — reject any single oversized value.
- `MAX_STORED_KEYS = 100_000` — reject **new** keys past the cap; updates
  to existing keys are always allowed.

`storage_admits()` gates both the `Put` and `Replicate` handlers (a peer
can't bypass via Replicate). Rejection is surfaced as `success: false` in
the ack (no silent drop). The `Replicate` success flag was kept correct:
an already-current replica (no update needed) still reports success;
only a cap-rejected *needed* write reports failure.

**Follow-up (registered):** a tight total-byte quota with per-peer
attribution needs the storage map to track which peer wrote each key —
deferred. The current per-value + key-count caps bound both OOM vectors
individually.

## 3. Per-target-node cap on the hinted-handoff queue

`src/distributed_network.rs`. The queue had only a global cap (1000). One
dead/flaky peer triggering many failed replications could fill the whole
queue and starve handoffs for other peers. Added `max_per_node` (default
`max_size / 10`, so ~10% per peer) checked in `enqueue`, plus a
`with_max_per_node` builder. Covered by
`test_handoff_per_node_cap_prevents_starvation`.

## 4. NodeId ↔ TLS certificate binding

`src/distributed_network.rs` + `src/node_security.rs`. Identity exchange
took the peer's NodeId from the self-reported `Ping`/`Pong` message — an
authenticated peer (valid cert) could claim **any** NodeId. Now both the
client and server identity-exchange paths call `verify_claimed_node_id`,
which derives the NodeId from the leaf certificate the peer presented
during the (already-verified) mTLS handshake and rejects a mismatch.

This is **free correctness by construction**: a node's own NodeId is
`node_id_from_cert(own_cert)`, so a legitimate peer's claimed id always
equals its cert id; only an impersonator mismatches. **Fail-closed** — if
the cert can't be read (impossible under enforced mTLS) the connection is
rejected. Validated by the real two-node handshake test
`test_two_nodes_connect`, which passes with the binding enforced
(confirming the runtime cert extraction works, not just compiles).

`node_id_from_cert` was promoted from private to `pub(crate)` so the
network layer can reuse the exact derivation.

## Verification

- `agent_policy` 30 tests, `agent_profiles` 18, `distributed_network` 95,
  `node_security` 27 — all pass.
- Full lib suite: network set 6754/0, autonomous set 6873/0.
- `ai_test_harness --all`: 585/585.
- clippy: 0 warnings across the touched feature sets.

## Remaining registered follow-ups

- Per-peer storage byte-quota with attribution (needs storage refactor).
- The `can_run_command` splitter is conservative, not a full POSIX parser
  — acceptable for a fail-closed security boundary.
