# `ai_cluster_node` — Distributed cluster node

| Field | Value |
|---|---|
| Group | Server |
| Binary path | `src/bin/ai_cluster_node.rs` |
| `required-features` | `full`, `server-cluster` |

## Purpose

Spawns a distributed node that joins a QUIC mesh for cluster-wide RAG, agent federation, and CRDT-based shared memory. Multiple nodes gossip state via Kademlia DHT and stay eventually consistent.

## Build

```bash
cargo build --release --bin ai_cluster_node --features "full,server-cluster"
```

## Usage

```bash
# First node
ai_cluster_node --node-id node1 --port 8091 --quic-port 9001

# Joining node (bootstraps from a known peer)
ai_cluster_node --node-id node2 --port 8092 --quic-port 9002 \
    --bootstrap-peers 192.168.1.10:9001
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)
