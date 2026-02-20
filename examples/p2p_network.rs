//! P2P network example with peer discovery and knowledge sharing.
//!
//! Run with: cargo run --example p2p_network --features "p2p,distributed"
//!
//! Demonstrates creating two P2PManager nodes, simulating peer discovery
//! via message passing, broadcasting knowledge, and querying peers.
//! No real network is used -- everything runs in-process via handle_message.

use std::time::{SystemTime, UNIX_EPOCH};

use ai_assistant::{KnowledgeShare, P2PConfig, P2PManager, PeerDataTrust, PeerInfo, PeerMessage};

fn main() {
    println!("=== P2P Network Demo ===\n");

    // -------------------------------------------------------------------------
    // Part 1: Create two P2P nodes with in-process configuration
    // -------------------------------------------------------------------------

    let config_a = P2PConfig {
        enabled: true,
        peer_data_trust: PeerDataTrust::VolatileOnly,
        stun_servers: vec![],    // No real STUN (demo mode)
        bootstrap_nodes: vec![], // No real bootstrap
        enable_upnp: false,
        enable_nat_pmp: false,
        max_peers: 10,
        min_reputation: 0.3,
        ..Default::default()
    };

    let config_b = P2PConfig {
        enabled: true,
        peer_data_trust: PeerDataTrust::MarkAndStore,
        stun_servers: vec![],
        bootstrap_nodes: vec![],
        enable_upnp: false,
        enable_nat_pmp: false,
        max_peers: 10,
        min_reputation: 0.3,
        ..Default::default()
    };

    let mut node_a = P2PManager::new(config_a);
    let mut node_b = P2PManager::new(config_b);

    let id_a = node_a.local_peer_id().to_string();
    let id_b = node_b.local_peer_id().to_string();

    println!("Node A: {}", id_a);
    println!("Node B: {}", id_b);

    // Register each node's peer in the other's reputation system so messages
    // are accepted (new peers start at 0.5 reputation, above min 0.3).
    node_a.register_peer(&id_b);
    node_b.register_peer(&id_a);

    // -------------------------------------------------------------------------
    // Part 2: Ping / Pong — basic peer liveness check
    // -------------------------------------------------------------------------

    println!("\n--- Ping / Pong ---\n");

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Node A sends a Ping to Node B
    let ping = PeerMessage::Ping {
        timestamp: now_secs,
    };
    let response = node_b.handle_message(&id_a, ping);

    match response {
        Some(PeerMessage::Pong { timestamp, peer_id }) => {
            println!(
                "Node B responded with Pong (ts={}, peer_id={})",
                timestamp, peer_id
            );
        }
        other => println!("Unexpected response: {:?}", other),
    }

    // -------------------------------------------------------------------------
    // Part 3: Peer discovery — exchange peer lists
    // -------------------------------------------------------------------------

    println!("\n--- Peer Discovery ---\n");

    // Simulate Node B knowing about a third peer "peer_charlie"
    let charlie_info = vec![PeerInfo {
        id: "peer_charlie".to_string(),
        address: "10.0.0.3:5000".parse().unwrap(),
        reputation: 0.8,
        last_seen: 0,
    }];

    // Node A receives a Peers message from Node B
    node_a.handle_message(
        &id_b,
        PeerMessage::Peers {
            peers: charlie_info,
        },
    );

    println!("Node A now has {} connections", node_a.peer_count());
    let stats_a = node_a.stats();
    println!(
        "  peers: {}, volatile entries: {}",
        stats_a.peer_count, stats_a.volatile_entries
    );

    // Node A asks Node B for its peer list
    let get_peers_resp = node_b.handle_message(&id_a, PeerMessage::GetPeers);
    match get_peers_resp {
        Some(PeerMessage::Peers { peers }) => {
            println!("Node B shared {} peer(s)", peers.len());
            for p in &peers {
                println!("  - {} @ {}", p.id, p.address);
            }
        }
        other => println!("Unexpected: {:?}", other),
    }

    // -------------------------------------------------------------------------
    // Part 4: Knowledge sharing — broadcast and receive
    // -------------------------------------------------------------------------

    println!("\n--- Knowledge Sharing ---\n");

    let knowledge = KnowledgeShare {
        id: "fact-001".to_string(),
        entity: "Rust".to_string(),
        attribute: "type".to_string(),
        value: "systems programming language".to_string(),
        source: format!("node:{}", id_a),
        timestamp: now_secs,
        signature: None,
    };

    // Node A shares knowledge with Node B
    let ack = node_b.handle_message(
        &id_a,
        PeerMessage::ShareKnowledge {
            data: knowledge.clone(),
        },
    );

    match ack {
        Some(PeerMessage::AckKnowledge { id, accepted }) => {
            println!("Node B acknowledged '{}': accepted={}", id, accepted);
        }
        other => println!("Unexpected: {:?}", other),
    }

    // Verify Node B stored the data in volatile memory
    let stored = node_b.get_volatile_data("Rust");
    println!("Node B volatile data for 'Rust': {} entries", stored.len());
    for entry in &stored {
        println!(
            "  {}.{} = '{}' (source: {})",
            entry.entity, entry.attribute, entry.value, entry.source
        );
    }

    // -------------------------------------------------------------------------
    // Part 5: Knowledge query
    // -------------------------------------------------------------------------

    println!("\n--- Knowledge Query ---\n");

    // Node A queries Node B for knowledge about "Rust"
    let query_resp = node_b.handle_message(
        &id_a,
        PeerMessage::QueryKnowledge {
            query: "Rust".to_string(),
        },
    );

    match query_resp {
        Some(PeerMessage::QueryResponse { query, results }) => {
            println!("Query '{}' returned {} result(s):", query, results.len());
            for r in &results {
                println!("  {}.{} = '{}'", r.entity, r.attribute, r.value);
            }
        }
        other => println!("Unexpected: {:?}", other),
    }

    // -------------------------------------------------------------------------
    // Part 6: Broadcast via outgoing message buffer
    // -------------------------------------------------------------------------

    println!("\n--- Broadcast via Buffer ---\n");

    // Node A broadcasts knowledge (buffers messages for all connections)
    let broadcast_data = KnowledgeShare {
        id: "fact-002".to_string(),
        entity: "P2P".to_string(),
        attribute: "protocol".to_string(),
        value: "custom UDP + TCP bootstrap".to_string(),
        source: format!("node:{}", id_a),
        timestamp: now_secs,
        signature: None,
    };
    node_a.broadcast_knowledge(broadcast_data);

    let outgoing = node_a.drain_outgoing();
    println!("Node A buffered {} outgoing message(s)", outgoing.len());
    for (peer, msg) in &outgoing {
        let msg_type = match msg {
            PeerMessage::ShareKnowledge { .. } => "ShareKnowledge",
            _ => "other",
        };
        println!("  -> {} : {}", peer, msg_type);
    }

    // -------------------------------------------------------------------------
    // Part 7: Statistics
    // -------------------------------------------------------------------------

    println!("\n--- Final Statistics ---\n");

    let stats_a = node_a.stats();
    let stats_b = node_b.stats();

    println!(
        "Node A: peers={}, volatile={}, known={}",
        stats_a.peer_count, stats_a.volatile_entries, stats_a.total_peers_known
    );
    println!(
        "Node B: peers={}, volatile={}, known={}",
        stats_b.peer_count, stats_b.volatile_entries, stats_b.total_peers_known
    );

    println!("\nDone.");
}
