//! Example: distributed_network_demo -- Demonstrates QUIC-based distributed networking.
//!
//! Run with: cargo run --example distributed_network_demo --features "distributed-network"
//!
//! This example showcases the distributed networking stack: NetworkConfig,
//! NetworkNode, consistent hashing, replication, failure detection, and
//! key-value storage. Creates a single-node cluster in-process to
//! demonstrate the API without requiring external peers.

use std::net::SocketAddr;
use std::time::Duration;

use ai_assistant::{
    NetworkConfig, NetworkNode, ReplicationConfig, WriteMode,
    NetworkDiscoveryConfig,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Distributed Network Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Network configuration
    // ------------------------------------------------------------------
    println!("--- 1. NetworkConfig ---\n");

    let identity_dir = std::env::temp_dir().join("dist_net_demo_identity");
    let _ = std::fs::create_dir_all(&identity_dir);

    let config = NetworkConfig {
        listen_addr: "127.0.0.1:0".parse::<SocketAddr>().expect("valid address"),
        bootstrap_peers: vec![],
        identity_dir: identity_dir.clone(),
        heartbeat_interval: Duration::from_secs(5),
        replication: ReplicationConfig {
            min_copies: 1,
            max_copies: 3,
            write_mode: WriteMode::Asynchronous,
            read_quorum: 1,
            write_quorum: 1,
            vnodes_per_node: 64,
        },
        discovery: NetworkDiscoveryConfig {
            enable_broadcast: false, // Disabled for demo (no LAN)
            broadcast_port: 9876,
            broadcast_interval: Duration::from_secs(30),
            enable_peer_exchange: true,
        },
        join_token: None,
        max_connections: 10,
        message_timeout: Duration::from_secs(5),
        phi_threshold: 8.0,
    };

    println!("  Listen address:     {}", config.listen_addr);
    println!("  Heartbeat interval: {:?}", config.heartbeat_interval);
    println!("  Replication copies: {}-{}", config.replication.min_copies, config.replication.max_copies);
    println!("  Write mode:         {:?}", config.replication.write_mode);
    println!("  VNodes per node:    {}", config.replication.vnodes_per_node);
    println!("  Max connections:    {}", config.max_connections);
    println!("  Phi threshold:      {}", config.phi_threshold);

    // ------------------------------------------------------------------
    // 2. Create a network node
    // ------------------------------------------------------------------
    println!("\n--- 2. Create NetworkNode ---\n");

    let node = match NetworkNode::new(config) {
        Ok(n) => {
            println!("  Node ID:       {}", n.node_id());
            println!("  Local address: {}", n.local_addr());
            n
        }
        Err(e) => {
            println!("  Error creating node: {}", e);
            println!("  (This can happen if QUIC/TLS initialization fails)");
            let _ = std::fs::remove_dir_all(&identity_dir);
            return;
        }
    };

    // ------------------------------------------------------------------
    // 3. Key-value storage
    // ------------------------------------------------------------------
    println!("\n--- 3. Key-Value Storage ---\n");

    // Store values
    let entries = [
        ("config:model", "llama-3.1-70b"),
        ("config:temperature", "0.7"),
        ("session:user123", "{\"turns\": 5, \"topic\": \"rust\"}"),
    ];

    for (key, value) in &entries {
        match node.store(key, value.as_bytes().to_vec()) {
            Ok(()) => println!("  Stored: {} = {}", key, value),
            Err(e) => println!("  Store error for {}: {}", key, e),
        }
    }

    println!("  Local key count: {}", node.local_key_count());

    // Retrieve values
    println!();
    for (key, _) in &entries {
        match node.get(key) {
            Ok(Some(data)) => {
                let value = String::from_utf8_lossy(&data);
                println!("  Get: {} = {}", key, value);
            }
            Ok(None) => println!("  Get: {} = <not found>", key),
            Err(e) => println!("  Get error for {}: {}", key, e),
        }
    }

    // ------------------------------------------------------------------
    // 4. Consistent hash ring
    // ------------------------------------------------------------------
    println!("\n--- 4. Consistent Hash Ring ---\n");

    let ring = node.ring_info();
    println!("  Total vnodes:       {}", ring.total_vnodes);
    println!("  Total nodes:        {}", ring.total_nodes);
    println!("  Replication factor: {}", ring.replication_factor);

    // Check which nodes own specific keys
    for key in &["user:alice", "user:bob", "model:gpt4"] {
        let owners = node.nodes_for_key(key);
        println!("  Key '{}' -> {} owner(s)", key, owners.len());
    }

    // ------------------------------------------------------------------
    // 5. Network statistics
    // ------------------------------------------------------------------
    println!("\n--- 5. Network Statistics ---\n");

    let stats = node.stats();
    println!("  Node ID:          {}", stats.node_id);
    println!("  Uptime:           {:?}", stats.uptime);
    println!("  Peers connected:  {}", stats.peers_connected);
    println!("  Peers dead:       {}", stats.peers_dead);
    println!("  Messages sent:    {}", stats.messages_sent);
    println!("  Messages received: {}", stats.messages_received);

    // ------------------------------------------------------------------
    // 6. Event polling
    // ------------------------------------------------------------------
    println!("\n--- 6. Event Polling ---\n");

    let events = node.poll_events();
    if events.is_empty() {
        println!("  No pending events (single-node cluster)");
    } else {
        for event in &events {
            println!("  Event: {:?}", event);
        }
    }

    // ------------------------------------------------------------------
    // 7. Join token generation
    // ------------------------------------------------------------------
    println!("\n--- 7. Admission Control ---\n");

    let token = node.generate_join_token(24, Some(5));
    println!("  Generated join token:");
    println!("    Token:      {}", token.token);
    println!("    Valid until: {:?}", token.expires_at);
    println!("    Max uses:    {:?}", token.max_uses);

    // ------------------------------------------------------------------
    // 8. Delete and cleanup
    // ------------------------------------------------------------------
    println!("\n--- 8. Delete & Shutdown ---\n");

    match node.delete("config:temperature") {
        Ok(true) => println!("  Deleted 'config:temperature'"),
        Ok(false) => println!("  'config:temperature' not found"),
        Err(e) => println!("  Delete error: {}", e),
    }

    println!("  Local key count after delete: {}", node.local_key_count());

    // Graceful shutdown
    node.shutdown();
    println!("  Node shut down gracefully");

    // Cleanup temp directory
    let _ = std::fs::remove_dir_all(&identity_dir);

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  distributed-network demo complete.");
    println!("  Capabilities: QUIC transport, consistent hashing,");
    println!("    KV storage, replication, failure detection,");
    println!("    join tokens, event polling.");
    println!("==========================================================");
}
