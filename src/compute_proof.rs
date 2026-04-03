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

    fn sha256_bytes(data: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];

        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        // Pre-processing: pad message
        let bit_len = (data.len() as u64) * 8;
        let mut padded = data.to_vec();
        padded.push(0x80);
        while (padded.len() % 64) != 56 {
            padded.push(0x00);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        // Process 512-bit blocks
        for chunk in padded.chunks(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut result = [0u8; 32];
        for (i, val) in h.iter().enumerate() {
            result[4 * i..4 * i + 4].copy_from_slice(&val.to_be_bytes());
        }
        result
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
