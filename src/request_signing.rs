//! Request signing for verification
//!
//! Sign requests to verify authenticity.

use std::time::{SystemTime, UNIX_EPOCH};

/// Pure-Rust SHA-256 and HMAC-SHA256 implementation (FIPS 180-4 / RFC 2104).
pub(crate) mod sha256 {
    // The round constants and the six bit-mixing helpers that used to live
    // here went with the compression function (N141): `sha256` below delegates
    // to `crate::sha256`, so K, H_INIT, ch, maj and the four sigmas had no
    // caller left. `hmac_sha256` still needs a SHA-256 and now gets the one
    // the published-vector test vouches for.

    /// Compute SHA-256 digest of `data`.
    /// Delegates to [`crate::sha256::sha256`] (N141).
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        crate::sha256::sha256(data)
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
#[non_exhaustive]
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

/// Request verifier with nonce replay protection.
///
/// Wraps a [`RequestSigner`] and tracks recently seen nonces so that the same
/// signed request cannot be replayed within the `max_age_secs` window.
pub struct RequestVerifier {
    signer: RequestSigner,
    /// Nonces seen recently, with the timestamp they were first seen.
    seen_nonces: std::sync::Mutex<std::collections::HashMap<String, u64>>,
    max_age_secs: u64,
}

impl RequestVerifier {
    /// Create a new verifier with the given secret, algorithm, and max request age.
    pub fn new(secret: &[u8], algorithm: SignatureAlgorithm, max_age_secs: u64) -> Self {
        Self {
            signer: RequestSigner::new(secret, algorithm),
            seen_nonces: std::sync::Mutex::new(std::collections::HashMap::new()),
            max_age_secs,
        }
    }

    /// Verify a signed request, rejecting replays (duplicate nonces).
    pub fn verify(&self, request: &SignedRequest) -> Result<(), SignatureError> {
        // First check signature validity and timestamp
        self.signer.verify(request, self.max_age_secs)?;

        // Then check nonce replay
        let mut nonces = self.seen_nonces.lock().unwrap_or_else(|e| e.into_inner());

        // Evict expired nonces to prevent unbounded growth
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        nonces.retain(|_, ts| now.saturating_sub(*ts) <= self.max_age_secs);

        if nonces.contains_key(&request.nonce) {
            return Err(SignatureError::ReplayedNonce);
        }
        nonces.insert(request.nonce.clone(), request.timestamp);

        Ok(())
    }
}

/// Signature errors
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignatureError {
    Invalid,
    Expired,
    MissingFields,
    /// The nonce has already been used (replay attack).
    ReplayedNonce,
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid => write!(f, "Invalid signature"),
            Self::Expired => write!(f, "Signature expired"),
            Self::MissingFields => write!(f, "Missing required fields"),
            Self::ReplayedNonce => {
                write!(f, "Nonce has already been used (possible replay attack)")
            }
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

    #[test]
    fn test_expired_signature() {
        let signer = RequestSigner::new(b"secret", SignatureAlgorithm::HmacSha256);
        let mut signed = signer.sign("payload");
        // Set timestamp to far in the past
        signed.timestamp = 1000;
        // Re-sign with correct timestamp for the signature to be valid content-wise,
        // but max_age_secs=0 means any age > 0 expires
        assert_eq!(signer.verify(&signed, 0), Err(SignatureError::Expired));
    }

    #[test]
    fn test_signature_error_display() {
        assert_eq!(SignatureError::Invalid.to_string(), "Invalid signature");
        assert_eq!(SignatureError::Expired.to_string(), "Signature expired");
        assert_eq!(
            SignatureError::MissingFields.to_string(),
            "Missing required fields"
        );
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(sha256::hex_encode(&[0x00, 0xff, 0xab]), "00ffab");
        assert_eq!(sha256::hex_encode(&[]), "");
        assert_eq!(sha256::hex_encode(&[0x01]), "01");
    }

    #[test]
    fn test_sha256_longer_input() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let digest = sha256::sha256(input);
        let hex = sha256::hex_encode(&digest);
        assert_eq!(
            hex,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_sign_different_payloads() {
        let signer = RequestSigner::new(b"key", SignatureAlgorithm::HmacSha256);
        let s1 = signer.sign("payload_a");
        let s2 = signer.sign("payload_b");
        // Different payloads produce different signatures (different nonces too, but even content differs)
        assert_ne!(s1.signature, s2.signature);
        assert_eq!(s1.algorithm, SignatureAlgorithm::HmacSha256);
    }
}
