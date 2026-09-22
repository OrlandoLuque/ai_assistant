// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ICE (Interactive Connectivity Establishment) types shared by every transport
//! that has to get through a NAT.
//!
//! Three subsystems need the same vocabulary and, until V347, each had written
//! its own copy of it:
//!
//! * `p2p` — peer-to-peer knowledge sharing,
//! * `distributed_rag` — distributed retrieval,
//! * `voice_agent` — WebRTC voice transport.
//!
//! (Plain code spans, not intra-doc links: all three modules are behind
//! features that the documentation build does not turn on, and a link across a
//! feature boundary renders as dead text. Same reason as V338.)
//!
//! The three enums were identical: the same four variants, the same meaning,
//! all `#[non_exhaustive]`. They differed only in which traits they derived,
//! which is not a difference in the type — it is three people arriving at the
//! same standard separately. The cost of keeping them apart was a collision at
//! the crate root, where `p2p` and `voice_agent` both re-exported theirs under
//! the bare name: each feature set compiled alone and no combination of them
//! did.
//!
//! This module is **not** feature-gated on purpose. A shared type that only
//! exists when one of its three consumers is enabled would not be shared.
//!
//! # What is deliberately NOT unified here
//!
//! `IceCandidate` and `IceState` are duplicated too, and they are *not* the
//! same type twice:
//!
//! * `p2p::IceCandidate` carries a `SocketAddr` and a `foundation`;
//!   `distributed_rag::IceCandidate` carries `address: String` plus a separate
//!   `port` and a `protocol`. Different fields, different representation.
//! * `p2p::IceState` has five variants; `distributed_rag::IceState` has seven
//!   (it adds `New` and `Completed`). Merging them would add states to a
//!   running state machine, which is a behaviour change and not a rename.
//!
//! Both are design questions with consequences, not accidental duplication, so
//! they are left alone and said out loud rather than folded in quietly.

use serde::{Deserialize, Serialize};

/// The kind of address an ICE candidate carries, as defined by RFC 8445.
///
/// Derives are the union of what the three former copies had, so no consumer
/// lost a capability in the merge: `p2p`'s lacked `Copy`, `PartialEq` and `Eq`,
/// and `voice_agent`'s lacked `Copy` and `Eq`.
///
/// The variant names are unchanged, which matters more than it looks: serde
/// serialises a fieldless enum as its variant name, so the on-the-wire and
/// on-disk representation is byte-identical to what each module produced
/// before. A peer running an older build still understands a newer one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IceCandidateType {
    /// A direct address of the host itself.
    Host,
    /// A public address learned from a STUN server (`srflx`).
    ServerReflexive,
    /// An address discovered from a peer during connectivity checks (`prflx`).
    PeerReflexive,
    /// An address on a TURN relay, used when nothing direct works.
    Relay,
}

impl std::fmt::Display for IceCandidateType {
    /// Writes the abbreviation SDP itself uses, not the Rust variant name.
    ///
    /// The `_` arm is unreachable inside this crate — `#[non_exhaustive]` only
    /// hides variants from *other* crates — and is kept so that adding a
    /// variant later cannot turn this into a compile error at every call site.
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => write!(f, "host"),
            Self::ServerReflexive => write!(f, "srflx"),
            Self::PeerReflexive => write!(f, "prflx"),
            Self::Relay => write!(f, "relay"),
            _ => write!(f, "unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_writes_the_sdp_abbreviations() {
        assert_eq!(IceCandidateType::Host.to_string(), "host");
        assert_eq!(IceCandidateType::ServerReflexive.to_string(), "srflx");
        assert_eq!(IceCandidateType::PeerReflexive.to_string(), "prflx");
        assert_eq!(IceCandidateType::Relay.to_string(), "relay");
    }

    /// The point of this one is the *format*, not the round trip.
    ///
    /// Merging three enums into one is only safe if the bytes do not move:
    /// `p2p` puts this type on the wire between peers and `distributed_rag`
    /// persists it, so a changed representation would be a silent
    /// incompatibility with every already-deployed build. Serde writes a
    /// fieldless variant as its bare name, and the names did not change —
    /// this test is what says so out loud.
    #[test]
    fn serde_representation_is_the_bare_variant_name() {
        assert_eq!(
            serde_json::to_string(&IceCandidateType::ServerReflexive)
                .expect("a fieldless enum always serialises"),
            "\"ServerReflexive\""
        );
        for candidate in [
            IceCandidateType::Host,
            IceCandidateType::ServerReflexive,
            IceCandidateType::PeerReflexive,
            IceCandidateType::Relay,
        ] {
            let json = serde_json::to_string(&candidate).expect("serialises");
            let back: IceCandidateType = serde_json::from_str(&json).expect("round trips");
            assert_eq!(back, candidate);
        }
    }

    /// `p2p`'s copy was not `Copy` and not `PartialEq`; this is the merge
    /// giving it both rather than taking anything away.
    #[test]
    fn the_merged_type_is_copy_and_comparable() {
        let a = IceCandidateType::Relay;
        let b = a; // moves nothing: Copy
        assert_eq!(a, b);
        assert_ne!(a, IceCandidateType::Host);
    }
}
