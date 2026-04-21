# `ai_gpu_share` — GPU-sharing network client

| Field | Value |
|---|---|
| Group | GPU Sharing |
| Binary path | `src/bin/ai_gpu_share.rs` |
| `required-features` | `full`, `gpu-sharing` |
| Added in | V73 |

## Purpose

Participate in the `ai_assistant` GPU-sharing network: share your GPU to earn credits, spend credits to run inference on peers. SETI-style, with an escrow-based credit system, dynamic pricing, collusion detection, and proof-of-compute challenges.

## Build

```bash
cargo build --release --bin ai_gpu_share --features "full,gpu-sharing"
```

## Subcommands

`start` · `stop` · `status` · `models` · `credits` · `peers` · `backup-keys`.

## Usage

```bash
# Start sharing (provider mode)
ai_gpu_share start --port 9400 --advertise

# Spend credits on a peer query
ai_gpu_share start --port 9400
# then from a client: ai_cli query -P gpu_share -p "hello"

# Account ops
ai_gpu_share credits
ai_gpu_share peers
ai_gpu_share backup-keys --output ./keys.bak
```

## Security highlights

- **Escrow-based credits** — locked before compute, released on verified delivery
- **Triple-signed receipts** — provider + requester + DHT-selected auditor
- **Commit-reveal auditor selection** — unpredictable by either party
- **GPU challenge** — matrix-mult benchmark for Sybil defence
- **Progressive earning fee** — 5–20%, disincentivises hoarding
- **Private networks** — whitelist mode for enterprise / friends

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/IMPROVEMENTS_V73.md`](../IMPROVEMENTS_V73.md)
