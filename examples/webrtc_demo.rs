//! WebRTC voice transport demo.
//!
//! Run with: cargo run --example webrtc_demo --features "webrtc,voice-agent"
//!
//! Demonstrates WebRTC signaling: SDP offer/answer,
//! ICE candidates, and transport configuration.
//! Note: actual voice transport requires a network peer.

use ai_assistant::{
    WebRtcConfig, WebRtcTransport, WebRtcAudioCodec, WebRtcIceCandidate,
    IceCandidateType, TurnServer, SdpAnswer,
};

fn main() {
    println!("=== WebRTC Voice Transport Demo ===\n");

    // 1. Configure WebRTC with STUN/TURN servers
    let config = WebRtcConfig {
        stun_servers: vec![
            "stun:stun.l.google.com:19302".to_string(),
            "stun:stun1.l.google.com:19302".to_string(),
        ],
        turn_servers: vec![
            TurnServer {
                url: "turn:turn.example.com:3478".to_string(),
                username: "user".to_string(),
                credential: "pass".to_string(),
            },
        ],
        audio_codec: WebRtcAudioCodec::Opus,
        sample_rate: 48000,
        enable_dtls: true,
        ice_timeout_ms: 5000,
    };

    println!("WebRTC Config:");
    println!("  STUN servers: {:?}", config.stun_servers);
    println!("  TURN servers: {}", config.turn_servers.len());
    println!("  Audio codec: {:?}", config.audio_codec);
    println!("  Sample rate: {}Hz", config.sample_rate);
    println!("  DTLS enabled: {}", config.enable_dtls);
    println!("  ICE timeout: {}ms", config.ice_timeout_ms);

    // 2. Create a transport
    let mut transport = WebRtcTransport::new(config);
    println!("\nTransport created (connected: {})", transport.is_connected());

    // 3. Generate an SDP offer
    let offer = transport.create_offer();
    println!("\nSDP Offer:");
    println!("  Session ID: {}", offer.session_id);
    println!("  SDP (first 100 chars): {}", &offer.sdp[..offer.sdp.len().min(100)]);

    // 4. Simulate receiving a remote SDP answer
    let answer = SdpAnswer {
        sdp: "v=0\r\no=- remote 2 IN IP4 192.168.1.50\r\ns=-\r\n".to_string(),
        session_id: offer.session_id.clone(),
    };
    transport.set_remote_answer(answer);
    println!("\nRemote SDP answer set.");

    // 5. Add ICE candidates
    let candidate = WebRtcIceCandidate {
        candidate: "candidate:1 1 udp 2122260223 192.168.1.100 54321 typ host".to_string(),
        sdp_mid: Some("audio".to_string()),
        sdp_m_line_index: Some(0),
        priority: 2122260223,
        candidate_type: IceCandidateType::Host,
    };
    transport.add_ice_candidate(candidate);
    println!("ICE candidates: {}", transport.ice_candidates().len());

    // 6. Connect
    match transport.connect() {
        Ok(()) => {
            println!("\nConnected!");
            println!("  Stats: {:?}", transport.stats());
        }
        Err(e) => println!("\nConnection failed: {}", e),
    }

    // 7. Disconnect
    transport.disconnect();
    println!("Disconnected (connected: {})", transport.is_connected());

    println!("\n=== Done (no peer connection needed for demo) ===");
}
