// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Proof that a computation actually happened.
//!
//! [`ComputeProof`] binds an LLM output to a specific model fingerprint, nonce,
//! and timing data so that a downstream verifier can check that the result was
//! not forged or replayed.

use serde::{Deserialize, Serialize};

/// Proof-of-computation receipt attached to an LLM inference result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeProof {
    /// SHA-256 hash of the output text.
    pub result_hash: Vec<u8>,
    /// Random nonce — prevents replay of the same proof.
    pub nonce: u64,
    /// Model name + quantisation (e.g. `"llama3-8b-q4_K_M"`).
    pub model_fingerprint: String,
    /// Number of tokens produced in the output.
    pub tokens_generated: u64,
    /// Wall-clock milliseconds spent generating the output.
    pub compute_time_ms: u64,
}

impl ComputeProof {
    /// Generate a proof for the given `output` text.
    pub fn generate(output: &str, model: &str, nonce: u64, time_ms: u64) -> Self {
        let hash = Self::sha256_bytes(output.as_bytes());
        // Rough token estimate: ~4 chars per token.
        let tokens = if output.is_empty() {
            0
        } else {
            (output.len() as u64 + 3) / 4
        };
        Self {
            result_hash: hash.to_vec(),
            nonce,
            model_fingerprint: model.to_string(),
            tokens_generated: tokens,
            compute_time_ms: time_ms,
        }
    }

    /// Verify that `result_hash` matches the SHA-256 of `output`.
    pub fn verify_hash(&self, output: &str) -> bool {
        let expected = Self::sha256_bytes(output.as_bytes());
        self.result_hash == expected.as_slice()
    }

    // ------------------------------------------------------------------
    // Pure-Rust SHA-256 (mirrors binary_integrity.rs implementation)
    // ------------------------------------------------------------------

    /// Delegates to [`crate::sha256::sha256`] (N141).
    ///
    /// This used to be the one hand-written copy with a test against published
    /// digests. The test moved with the implementation, so what vouched for
    /// this function now vouches for all five former call sites.
    fn sha256_bytes(data: &[u8]) -> [u8; 32] {
        crate::sha256::sha256(data)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_verify() {
        let proof = ComputeProof::generate("hello world", "llama3-8b-q4", 42, 150);
        assert!(proof.verify_hash("hello world"));
        assert!(!proof.verify_hash("different text"));
        assert_eq!(proof.nonce, 42);
        assert_eq!(proof.compute_time_ms, 150);
        assert_eq!(proof.model_fingerprint, "llama3-8b-q4");
    }

    #[test]
    fn test_empty_output() {
        let proof = ComputeProof::generate("", "model-a", 0, 0);
        assert!(proof.verify_hash(""));
        assert!(!proof.verify_hash("x"));
        assert_eq!(proof.tokens_generated, 0);
    }

    #[test]
    fn test_different_nonces_same_hash() {
        let p1 = ComputeProof::generate("text", "m", 1, 10);
        let p2 = ComputeProof::generate("text", "m", 2, 10);
        // Same output → same hash, but different nonces.
        assert_eq!(p1.result_hash, p2.result_hash);
        assert_ne!(p1.nonce, p2.nonce);
    }
}
