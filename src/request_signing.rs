//! Request signing for verification
//!
//! Sign requests to verify authenticity.

use std::time::{SystemTime, UNIX_EPOCH};

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
        let message = format!("{}:{}:{}", request.timestamp, request.nonce, request.payload);
        let expected = self.compute_signature(&message);

        if !constant_time_compare(&expected, &request.signature) {
            return Err(SignatureError::Invalid);
        }

        Ok(())
    }

    fn compute_signature(&self, message: &str) -> String {
        // Simple hash-based signature (in production, use proper HMAC)
        let mut hash = 0u64;
        for (i, byte) in message.bytes().enumerate() {
            // Use wrapping_pow to avoid overflow
            hash = hash.wrapping_add((byte as u64).wrapping_mul(31u64.wrapping_pow((i % 16) as u32)));
        }
        for (i, byte) in self.secret.iter().enumerate() {
            hash ^= (*byte as u64).wrapping_mul(37u64.wrapping_pow((i % 16) as u32));
        }
        format!("{:016x}", hash)
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
        assert!(signer.verify(&signed, 60).is_ok());
    }

    #[test]
    fn test_invalid_signature() {
        let signer = RequestSigner::new(b"secret", SignatureAlgorithm::HmacSha256);

        let mut signed = signer.sign("test payload");
        signed.signature = "invalid".to_string();

        assert!(signer.verify(&signed, 60).is_err());
    }
}
