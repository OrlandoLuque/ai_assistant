//! Request signing for verification
//!
//! Sign requests to verify authenticity.

use std::time::{SystemTime, UNIX_EPOCH};

/// Pure-Rust SHA-256 and HMAC-SHA256 implementation (FIPS 180-4 / RFC 2104).
pub(crate) mod sha256 {
    /// SHA-256 round constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
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

    /// Initial hash values (first 32 bits of the fractional parts of the square roots of the first 8 primes)
    const H_INIT: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    #[inline]
    fn ch(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (!x & z)
    }
    #[inline]
    fn maj(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (x & z) ^ (y & z)
    }
    #[inline]
    fn big_sigma0(x: u32) -> u32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }
    #[inline]
    fn big_sigma1(x: u32) -> u32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }
    #[inline]
    fn small_sigma0(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }
    #[inline]
    fn small_sigma1(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }

    /// Compute SHA-256 digest of `data`.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        let mut h = H_INIT;

        // Pre-processing: pad message
        let bit_len = (data.len() as u64) * 8;
        let mut padded = data.to_vec();
        padded.push(0x80);
        while (padded.len() % 64) != 56 {
            padded.push(0x00);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        // Process each 512-bit (64-byte) block
        for block in padded.chunks_exact(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    block[i * 4],
                    block[i * 4 + 1],
                    block[i * 4 + 2],
                    block[i * 4 + 3],
                ]);
            }
            for i in 16..64 {
                w[i] = small_sigma1(w[i - 2])
                    .wrapping_add(w[i - 7])
                    .wrapping_add(small_sigma0(w[i - 15]))
                    .wrapping_add(w[i - 16]);
            }

            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

            for i in 0..64 {
                let t1 = hh
                    .wrapping_add(big_sigma1(e))
                    .wrapping_add(ch(e, f, g))
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
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
            result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
        }
        result
    }

    /// Compute HMAC-SHA256 per RFC 2104.
    pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
        const BLOCK_SIZE: usize = 64;

        // Step 1: if key > block size, hash it; if shorter, zero-pad
        let mut padded_key = [0u8; BLOCK_SIZE];
        if key.len() > BLOCK_SIZE {
            let hashed = sha256(key);
            padded_key[..32].copy_from_slice(&hashed);
        } else {
            padded_key[..key.len()].copy_from_slice(key);
        }

        // Step 2: XOR key with ipad (0x36) and opad (0x5c)
        let mut i_key_pad = [0u8; BLOCK_SIZE];
        let mut o_key_pad = [0u8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            i_key_pad[i] = padded_key[i] ^ 0x36;
            o_key_pad[i] = padded_key[i] ^ 0x5c;
        }

        // Step 3: inner hash = SHA-256(i_key_pad || message)
        let mut inner_data = Vec::with_capacity(BLOCK_SIZE + message.len());
        inner_data.extend_from_slice(&i_key_pad);
        inner_data.extend_from_slice(message);
        let inner_hash = sha256(&inner_data);

        // Step 4: outer hash = SHA-256(o_key_pad || inner_hash)
        let mut outer_data = Vec::with_capacity(BLOCK_SIZE + 32);
        outer_data.extend_from_slice(&o_key_pad);
        outer_data.extend_from_slice(&inner_hash);
        sha256(&outer_data)
    }

    /// Hex-encode a byte slice.
    pub fn hex_encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

/// Signature algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    HmacSha256,
    HmacSha512,
}

/// Signed request
#[derive(Debug, Clone)]
pub struct SignedRequest {
    pub payload: String,
    pub signature: String,
    pub timestamp: u64,
    pub nonce: String,
    pub algorithm: SignatureAlgorithm,
}

/// Request signer
pub struct RequestSigner {
    secret: Vec<u8>,
    algorithm: SignatureAlgorithm,
}

impl RequestSigner {
    pub fn new(secret: &[u8], algorithm: SignatureAlgorithm) -> Self {
        Self {
            secret: secret.to_vec(),
            algorithm,
        }
    }

    pub fn sign(&self, payload: &str) -> SignedRequest {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let nonce = uuid::Uuid::new_v4().to_string();

        let message = format!("{}:{}:{}", timestamp, nonce, payload);
        let signature = self.compute_signature(&message);

        SignedRequest {
            payload: payload.to_string(),
            signature,
            timestamp,
            nonce,
            algorithm: self.algorithm,
        }
    }

    pub fn verify(&self, request: &SignedRequest, max_age_secs: u64) -> Result<(), SignatureError> {
        // Check timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now.saturating_sub(request.timestamp) > max_age_secs {
            return Err(SignatureError::Expired);
        }

        // Verify signature
        let message = format!(
            "{}:{}:{}",
            request.timestamp, request.nonce, request.payload
        );
        let expected = self.compute_signature(&message);

        if !constant_time_compare(&expected, &request.signature) {
            return Err(SignatureError::Invalid);
        }

        Ok(())
    }

    fn compute_signature(&self, message: &str) -> String {
        let mac = sha256::hmac_sha256(&self.secret, message.as_bytes());
        sha256::hex_encode(&mac)
    }
}

fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.bytes().zip(b.bytes()) {
        result |= x ^ y;
    }
    result == 0
}

/// Signature errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureError {
    Invalid,
    Expired,
    MissingFields,
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid => write!(f, "Invalid signature"),
            Self::Expired => write!(f, "Signature expired"),
            Self::MissingFields => write!(f, "Missing required fields"),
        }
    }
}

impl std::error::Error for SignatureError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify() {
        let signer = RequestSigner::new(b"secret", SignatureAlgorithm::HmacSha256);

        let signed = signer.sign("test payload");
        // New HMAC-SHA256 signatures are 64 hex chars
        assert_eq!(signed.signature.len(), 64);
        assert!(signer.verify(&signed, 60).is_ok());
    }

    #[test]
    fn test_invalid_signature() {
        let signer = RequestSigner::new(b"secret", SignatureAlgorithm::HmacSha256);

        let mut signed = signer.sign("test payload");
        signed.signature = "invalid".to_string();

        assert_eq!(signer.verify(&signed, 60), Err(SignatureError::Invalid));
    }

    #[test]
    fn test_sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb924...
        let digest = sha256::sha256(b"");
        let hex = sha256::hex_encode(&digest);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let digest = sha256::sha256(b"abc");
        let hex = sha256::hex_encode(&digest);
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_hmac_sha256_known_vector() {
        // RFC 4231 Test Case 2: HMAC-SHA256 with key="Jefe" and data="what do ya want for nothing?"
        let mac = sha256::hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex = sha256::hex_encode(&mac);
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_different_secrets_different_signatures() {
        let signer1 = RequestSigner::new(b"secret1", SignatureAlgorithm::HmacSha256);
        let signer2 = RequestSigner::new(b"secret2", SignatureAlgorithm::HmacSha256);

        let signed1 = signer1.sign("same payload");
        let signed2 = signer2.sign("same payload");

        // Verify each signer only accepts its own signatures
        assert!(signer1.verify(&signed1, 60).is_ok());
        assert!(signer2.verify(&signed1, 60).is_err());
        assert!(signer2.verify(&signed2, 60).is_ok());
        assert!(signer1.verify(&signed2, 60).is_err());
    }
}
