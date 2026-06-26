use super::*;

// ─── P2P Test Categories (feature = "p2p") ──────────────────────────────────

#[cfg(feature = "p2p")]
pub(crate) fn tests_p2p_nat() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ P2P NAT Traversal")));
    let mut results = Vec::new();

    results.push(run_test("NatType::can_direct_connect", || {
        assert_test!(ai_assistant::NatType::None.can_direct_connect());
        assert_test!(ai_assistant::NatType::FullCone.can_direct_connect());
        assert_test!(ai_assistant::NatType::RestrictedCone.can_direct_connect());
        assert_test!(!ai_assistant::NatType::Symmetric.can_direct_connect());
        assert_test!(!ai_assistant::NatType::Unknown.can_direct_connect());
        Ok(())
    }));

    results.push(run_test("NatType::needs_relay", || {
        assert_test!(ai_assistant::NatType::Symmetric.needs_relay());
        assert_test!(ai_assistant::NatType::Unknown.needs_relay());
        assert_test!(!ai_assistant::NatType::None.needs_relay());
        assert_test!(!ai_assistant::NatType::FullCone.needs_relay());
        assert_test!(!ai_assistant::NatType::RestrictedCone.needs_relay());
        Ok(())
    }));

    results.push(run_test("NatTraversal creation", || {
        let config = ai_assistant::P2PConfig::default();
        let nat = ai_assistant::NatTraversal::new(config);
        assert_test!(
            nat.get_connectable_address().is_none(),
            "Fresh NatTraversal should have no connectable address"
        );
        Ok(())
    }));

    results.push(run_test("P2PConfig defaults", || {
        let config = ai_assistant::P2PConfig::default();
        assert_test!(!config.enabled, "P2P should be disabled by default");
        assert_test!(
            config.stun_servers.len() == 2,
            "Should have 2 default STUN servers"
        );
        assert_test!(config.enable_upnp, "UPnP should be enabled by default");
        assert_test!(
            config.enable_nat_pmp,
            "NAT-PMP should be enabled by default"
        );
        assert_eq_test!(config.max_peers, 50);
        Ok(())
    }));

    results.push(run_test("UPnP disabled returns error", || {
        let mut nat = ai_assistant::NatTraversal::new(ai_assistant::P2PConfig {
            enable_upnp: false,
            ..Default::default()
        });
        let result = nat.try_upnp_mapping(12345, 12345);
        assert_test!(result.is_err(), "Should fail when UPnP disabled");
        assert_eq_test!(result.as_ref().unwrap_err().as_str(), "UPnP disabled");
        Ok(())
    }));

    CategoryResult {
        name: "p2p_nat".to_string(),
        results,
    }
}

#[cfg(feature = "p2p")]
pub(crate) fn tests_p2p_reputation() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ P2P Reputation System")));
    let mut results = Vec::new();

    results.push(run_test("PeerReputation lifecycle", || {
        let mut rep = ai_assistant::PeerReputation::new("test_peer");
        assert_test!(
            (rep.score - 0.5).abs() < 0.01,
            "Initial score should be 0.5"
        );
        rep.record_success();
        rep.record_success();
        rep.record_failure();
        assert_test!(rep.score > 0.5, "Score should increase with net successes");
        Ok(())
    }));

    results.push(run_test("PeerReputation ban/unban cycle", || {
        let mut rep = ai_assistant::PeerReputation::new("ban_test");
        rep.ban("spam");
        assert_test!(rep.banned, "Should be banned");
        assert_test!(!rep.is_trusted(0.1), "Banned peer should not be trusted");

        rep.unban();
        assert_test!(!rep.banned, "Should be unbanned");
        assert_test!(
            (rep.score - 0.1).abs() < f32::EPSILON,
            "Score should be 0.1 after unban"
        );
        assert_test!(
            rep.is_trusted(0.05),
            "Should be trusted at low threshold after unban"
        );
        Ok(())
    }));

    results.push(run_test("PeerReputation accuracy", || {
        let mut rep = ai_assistant::PeerReputation::new("acc_test");
        rep.record_correct_contribution();
        rep.record_correct_contribution();
        rep.record_incorrect_contribution();
        let acc = rep.accuracy();
        assert_test!(
            (acc - 0.666).abs() < 0.01,
            format!("Accuracy should be ~0.666, got {}", acc)
        );
        Ok(())
    }));

    results.push(run_test("ReputationSystem is_trusted", || {
        let mut system = ai_assistant::ReputationSystem::new(0.3);

        // Unknown peer is not trusted
        assert_test!(
            !system.is_trusted("unknown"),
            "Unknown peer should not be trusted"
        );

        // Peer with successes should be trusted
        let rep = system.get_or_create("good_peer");
        for _ in 0..5 {
            rep.record_success();
        }
        assert_test!(
            system.is_trusted("good_peer"),
            "Good peer should be trusted"
        );
        Ok(())
    }));

    results.push(run_test("ReputationSystem get_top_peers", || {
        let mut system = ai_assistant::ReputationSystem::new(0.3);

        // Create peers with different scores
        {
            let r = system.get_or_create("high");
            for _ in 0..10 {
                r.record_success();
                r.record_correct_contribution();
            }
        }
        {
            let r = system.get_or_create("low");
            r.record_success();
        }
        {
            let r = system.get_or_create("banned");
            r.ban("test");
        }

        let top = system.get_top_peers(2);
        assert_eq_test!(top.len(), 2);
        assert_test!(
            top[0].score >= top[1].score,
            "Should be sorted by score descending"
        );
        assert_test!(
            !top.iter().any(|p| p.banned),
            "Banned peers should be excluded"
        );
        Ok(())
    }));

    CategoryResult {
        name: "p2p_reputation".to_string(),
        results,
    }
}

#[cfg(feature = "p2p")]
pub(crate) fn tests_p2p_manager() -> CategoryResult {
    println!("\n{}", bold(&cyan("▶ P2P Manager")));
    let mut results = Vec::new();

    results.push(run_test("P2PManager creation", || {
        let manager = ai_assistant::P2PManager::new(ai_assistant::P2PConfig {
            enabled: true,
            ..Default::default()
        });
        assert_test!(!manager.local_peer_id().is_empty(), "Should have a peer ID");
        assert_eq_test!(manager.peer_count(), 0);
        Ok(())
    }));

    results.push(run_test("P2PManager start disabled", || {
        let mut manager = ai_assistant::P2PManager::new(ai_assistant::P2PConfig {
            enabled: false,
            ..Default::default()
        });
        let result = manager.start();
        assert_test!(result.is_err(), "Should fail when disabled");
        assert_eq_test!(result.as_ref().unwrap_err().as_str(), "P2P is disabled");
        Ok(())
    }));

    results.push(run_test("P2PManager stop clears state", || {
        let mut manager = ai_assistant::P2PManager::new(ai_assistant::P2PConfig {
            enabled: true,
            ..Default::default()
        });
        manager.stop();
        let stats = manager.stats();
        assert_test!(!stats.running, "Should not be running after stop");
        assert_eq_test!(stats.peer_count, 0);
        Ok(())
    }));

    results.push(run_test("P2PManager stats", || {
        let manager = ai_assistant::P2PManager::new(ai_assistant::P2PConfig {
            enabled: true,
            ..Default::default()
        });
        let stats = manager.stats();
        assert_test!(stats.enabled, "Should be enabled");
        assert_test!(!stats.running, "Should not be running initially");
        assert_eq_test!(stats.peer_count, 0);
        assert_eq_test!(stats.volatile_entries, 0);
        assert_eq_test!(stats.banned_peers, 0);
        Ok(())
    }));

    results.push(run_test("P2PManager handle Ping → Pong", || {
        // Use min_reputation: 0.0 and register_peer so the sender is trusted
        let mut manager = ai_assistant::P2PManager::new(ai_assistant::P2PConfig {
            enabled: true,
            min_reputation: 0.0,
            ..Default::default()
        });
        manager.register_peer("sender");

        let response = manager.handle_message(
            "sender",
            ai_assistant::PeerMessage::Ping { timestamp: 12345 },
        );
        assert_test!(response.is_some(), "Ping should produce a Pong response");
        if let Some(ai_assistant::PeerMessage::Pong { peer_id, timestamp }) = response {
            assert_eq_test!(timestamp, 12345);
            assert_test!(!peer_id.is_empty(), "Pong should include our peer ID");
        } else {
            return Err("Expected Pong response".to_string());
        }
        Ok(())
    }));

    results.push(run_test("P2PManager local_peer_id", || {
        let m1 = ai_assistant::P2PManager::new(ai_assistant::P2PConfig::default());
        let m2 = ai_assistant::P2PManager::new(ai_assistant::P2PConfig::default());
        assert_test!(
            !m1.local_peer_id().is_empty(),
            "Peer ID should not be empty"
        );
        assert_test!(
            m1.local_peer_id().starts_with("peer_"),
            "Peer ID should start with 'peer_'"
        );
        // Different instances should have different IDs (timestamp-based)
        // Note: on very fast machines they could be identical if created in same nanosecond
        // so we don't assert inequality — just verify format
        assert_test!(!m2.local_peer_id().is_empty());
        Ok(())
    }));

    CategoryResult {
        name: "p2p_manager".to_string(),
        results,
    }
}
