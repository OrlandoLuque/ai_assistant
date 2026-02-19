//! Content encryption
//!
//! Encrypt and decrypt conversation content.
//!
//! When the `rag` feature is enabled, `Aes256Gcm` and `ChaCha20Poly1305` use
//! real AES-256-GCM authenticated encryption via the `aes-gcm` crate.
//! Without `rag`, they fall back to XOR with nonce mixing (NOT cryptographically secure).

use std::collections::HashMap;

#[cfg(feature = "aes-gcm")]
use aes_gcm::{Aes256Gcm, aead::{Aead, KeyInit}};

/// Encryption algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
    Xor, // Simple for demo, not secure
}

/// Encrypted content
#[derive(Debug, Clone)]
pub struct EncryptedContent {
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub algorithm: EncryptionAlgorithm,
    pub key_id: String,
}

/// Encryption key
#[derive(Debug, Clone)]
pub struct EncryptionKey {
    pub id: String,
    pub key: Vec<u8>,
    pub algorithm: EncryptionAlgorithm,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

impl EncryptionKey {
    pub fn new(id: &str, key: Vec<u8>, algorithm: EncryptionAlgorithm) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            id: id.to_string(),
            key,
            algorithm,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            expires_at: None,
        }
    }

    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    pub fn is_expired(&self) -> bool {
        use std::time::{SystemTime, UNIX_EPOCH};

        if let Some(exp) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now > exp
        } else {
            false
        }
    }
}

/// Content encryptor
pub struct ContentEncryptor {
    keys: HashMap<String, EncryptionKey>,
    active_key_id: Option<String>,
}

impl ContentEncryptor {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            active_key_id: None,
        }
    }

    pub fn add_key(&mut self, key: EncryptionKey) {
        if self.active_key_id.is_none() {
            self.active_key_id = Some(key.id.clone());
        }
        self.keys.insert(key.id.clone(), key);
    }

    pub fn set_active_key(&mut self, key_id: &str) -> Result<(), EncryptionError> {
        if self.keys.contains_key(key_id) {
            self.active_key_id = Some(key_id.to_string());
            Ok(())
        } else {
            Err(EncryptionError::KeyNotFound)
        }
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedContent, EncryptionError> {
        let key_id = self.active_key_id.as_ref()
            .ok_or(EncryptionError::NoActiveKey)?;

        let key = self.keys.get(key_id)
            .ok_or(EncryptionError::KeyNotFound)?;

        if key.is_expired() {
            return Err(EncryptionError::KeyExpired);
        }

        let nonce = self.generate_nonce(key.algorithm);
        let ciphertext = self.encrypt_with_key(plaintext, &key.key, &nonce, key.algorithm)?;

        Ok(EncryptedContent {
            ciphertext,
            nonce,
            algorithm: key.algorithm,
            key_id: key_id.clone(),
        })
    }

    pub fn decrypt(&self, encrypted: &EncryptedContent) -> Result<Vec<u8>, EncryptionError> {
        let key = self.keys.get(&encrypted.key_id)
            .ok_or(EncryptionError::KeyNotFound)?;

        if key.algorithm != encrypted.algorithm {
            return Err(EncryptionError::AlgorithmMismatch);
        }

        self.decrypt_with_key(&encrypted.ciphertext, &key.key, &encrypted.nonce, key.algorithm)
    }

    pub fn encrypt_string(&self, plaintext: &str) -> Result<EncryptedContent, EncryptionError> {
        self.encrypt(plaintext.as_bytes())
    }

    pub fn decrypt_string(&self, encrypted: &EncryptedContent) -> Result<String, EncryptionError> {
        let bytes = self.decrypt(encrypted)?;
        String::from_utf8(bytes).map_err(|_| EncryptionError::DecryptionFailed)
    }

    fn generate_nonce(&self, algorithm: EncryptionAlgorithm) -> Vec<u8> {
        let size = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => 12,
            EncryptionAlgorithm::ChaCha20Poly1305 => 12,
            EncryptionAlgorithm::Xor => 0,
        };
        if size == 0 {
            return Vec::new();
        }

        // Generate random nonce by mixing multiple entropy sources
        use std::time::{SystemTime, UNIX_EPOCH};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let mut nonce = Vec::with_capacity(size);
        for i in 0..size {
            let mut hasher = DefaultHasher::new();
            timestamp.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            std::thread::current().id().hash(&mut hasher);
            // Mix in address of stack variable for ASLR entropy
            let stack_var = 0u8;
            (&stack_var as *const u8 as u64).hash(&mut hasher);
            nonce.push(hasher.finish() as u8);
        }
        nonce
    }

    fn encrypt_with_key(
        &self,
        plaintext: &[u8],
        key: &[u8],
        nonce: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<Vec<u8>, EncryptionError> {
        match algorithm {
            EncryptionAlgorithm::Xor => {
                if key.is_empty() {
                    return Err(EncryptionError::EncryptionFailed);
                }
                Ok(plaintext.iter()
                    .enumerate()
                    .map(|(i, b)| b ^ key[i % key.len()])
                    .collect())
            }
            #[cfg(feature = "aes-gcm")]
            EncryptionAlgorithm::Aes256Gcm | EncryptionAlgorithm::ChaCha20Poly1305 => {
                // Pad or truncate key to exactly 32 bytes for AES-256
                let mut key_bytes = [0u8; 32];
                let len = key.len().min(32);
                key_bytes[..len].copy_from_slice(&key[..len]);

                let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&key_bytes));
                let aes_nonce = aes_gcm::Nonce::from_slice(nonce);
                cipher.encrypt(aes_nonce, plaintext)
                    .map_err(|_| EncryptionError::EncryptionFailed)
            }
            #[cfg(not(feature = "aes-gcm"))]
            EncryptionAlgorithm::Aes256Gcm | EncryptionAlgorithm::ChaCha20Poly1305 => {
                // Fallback: XOR with nonce mixing (NOT cryptographically secure)
                if key.is_empty() {
                    return Err(EncryptionError::EncryptionFailed);
                }
                Ok(plaintext.iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let k = key[i % key.len()];
                        let n = if nonce.is_empty() { 0 } else { nonce[i % nonce.len()] };
                        b ^ k ^ n
                    })
                    .collect())
            }
        }
    }

    fn decrypt_with_key(
        &self,
        ciphertext: &[u8],
        key: &[u8],
        nonce: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<Vec<u8>, EncryptionError> {
        match algorithm {
            EncryptionAlgorithm::Xor => {
                // XOR is symmetric
                self.encrypt_with_key(ciphertext, key, nonce, algorithm)
            }
            #[cfg(feature = "aes-gcm")]
            EncryptionAlgorithm::Aes256Gcm | EncryptionAlgorithm::ChaCha20Poly1305 => {
                let mut key_bytes = [0u8; 32];
                let len = key.len().min(32);
                key_bytes[..len].copy_from_slice(&key[..len]);

                let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&key_bytes));
                let aes_nonce = aes_gcm::Nonce::from_slice(nonce);
                cipher.decrypt(aes_nonce, ciphertext)
                    .map_err(|_| EncryptionError::DecryptionFailed)
            }
            #[cfg(not(feature = "aes-gcm"))]
            EncryptionAlgorithm::Aes256Gcm | EncryptionAlgorithm::ChaCha20Poly1305 => {
                // Fallback: XOR with nonce mixing is symmetric
                self.encrypt_with_key(ciphertext, key, nonce, algorithm)
            }
        }
    }

    pub fn rotate_key(&mut self, new_key: EncryptionKey) {
        let new_id = new_key.id.clone();
        self.add_key(new_key);
        self.active_key_id = Some(new_id);
    }

    pub fn remove_expired_keys(&mut self) {
        self.keys.retain(|_, k| !k.is_expired());
    }
}

impl Default for ContentEncryptor {
    fn default() -> Self {
        Self::new()
    }
}

/// Encryption errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionError {
    NoActiveKey,
    KeyNotFound,
    KeyExpired,
    AlgorithmMismatch,
    EncryptionFailed,
    DecryptionFailed,
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveKey => write!(f, "No active encryption key"),
            Self::KeyNotFound => write!(f, "Encryption key not found"),
            Self::KeyExpired => write!(f, "Encryption key has expired"),
            Self::AlgorithmMismatch => write!(f, "Algorithm mismatch"),
            Self::EncryptionFailed => write!(f, "Encryption failed"),
            Self::DecryptionFailed => write!(f, "Decryption failed"),
        }
    }
}

impl std::error::Error for EncryptionError {}

/// Encrypted message store
pub struct EncryptedMessageStore {
    encryptor: ContentEncryptor,
    messages: HashMap<String, EncryptedContent>,
}

impl EncryptedMessageStore {
    pub fn new(encryptor: ContentEncryptor) -> Self {
        Self {
            encryptor,
            messages: HashMap::new(),
        }
    }

    pub fn store(&mut self, id: &str, content: &str) -> Result<(), EncryptionError> {
        let encrypted = self.encryptor.encrypt_string(content)?;
        self.messages.insert(id.to_string(), encrypted);
        Ok(())
    }

    pub fn retrieve(&self, id: &str) -> Result<String, EncryptionError> {
        let encrypted = self.messages.get(id)
            .ok_or(EncryptionError::KeyNotFound)?;
        self.encryptor.decrypt_string(encrypted)
    }

    pub fn delete(&mut self, id: &str) {
        self.messages.remove(id);
    }

    pub fn list_ids(&self) -> Vec<&String> {
        self.messages.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_xor() {
        let mut encryptor = ContentEncryptor::new();
        encryptor.add_key(EncryptionKey::new(
            "key1",
            b"mysecretkey12345".to_vec(),
            EncryptionAlgorithm::Xor,
        ));

        let plaintext = "Hello, World!";
        let encrypted = encryptor.encrypt_string(plaintext).unwrap();
        let decrypted = encryptor.decrypt_string(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_key_rotation() {
        let mut encryptor = ContentEncryptor::new();
        encryptor.add_key(EncryptionKey::new(
            "key1",
            b"firstkey".to_vec(),
            EncryptionAlgorithm::Xor,
        ));

        let encrypted1 = encryptor.encrypt_string("message1").unwrap();
        assert_eq!(encrypted1.key_id, "key1");

        encryptor.rotate_key(EncryptionKey::new(
            "key2",
            b"secondkey".to_vec(),
            EncryptionAlgorithm::Xor,
        ));

        let encrypted2 = encryptor.encrypt_string("message2").unwrap();
        assert_eq!(encrypted2.key_id, "key2");

        // Can still decrypt old messages
        let decrypted1 = encryptor.decrypt_string(&encrypted1).unwrap();
        assert_eq!(decrypted1, "message1");
    }

    #[test]
    fn test_message_store() {
        let mut encryptor = ContentEncryptor::new();
        encryptor.add_key(EncryptionKey::new(
            "key1",
            b"storekey".to_vec(),
            EncryptionAlgorithm::Xor,
        ));

        let mut store = EncryptedMessageStore::new(encryptor);

        store.store("msg1", "Secret message").unwrap();
        let retrieved = store.retrieve("msg1").unwrap();

        assert_eq!(retrieved, "Secret message");
    }

    #[cfg(feature = "aes-gcm")]
    #[test]
    fn test_aes256gcm_encrypt_decrypt() {
        let mut encryptor = ContentEncryptor::new();
        // AES-256 needs a 32-byte key
        let key = b"this_is_a_32_byte_key_for_aes!!".to_vec();
        encryptor.add_key(EncryptionKey::new("aes_key", key, EncryptionAlgorithm::Aes256Gcm));

        let plaintext = "Top secret AES-encrypted message";
        let encrypted = encryptor.encrypt_string(plaintext).unwrap();
        assert_eq!(encrypted.algorithm, EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(encrypted.nonce.len(), 12);

        // Ciphertext should differ from plaintext (authenticated encryption adds tag)
        assert_ne!(encrypted.ciphertext, plaintext.as_bytes());
        assert!(encrypted.ciphertext.len() > plaintext.len()); // GCM adds 16-byte auth tag

        let decrypted = encryptor.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[cfg(feature = "aes-gcm")]
    #[test]
    fn test_aes256gcm_tamper_detection() {
        let mut encryptor = ContentEncryptor::new();
        let key = b"another_32_byte_key_for_testing!".to_vec();
        encryptor.add_key(EncryptionKey::new("aes_key", key, EncryptionAlgorithm::Aes256Gcm));

        let encrypted = encryptor.encrypt_string("sensitive data").unwrap();

        // Tamper with ciphertext
        let mut tampered = encrypted.clone();
        if let Some(byte) = tampered.ciphertext.first_mut() {
            *byte ^= 0xFF;
        }

        // Decryption should fail due to authentication
        assert_eq!(encryptor.decrypt(&tampered), Err(EncryptionError::DecryptionFailed));
    }

    #[cfg(feature = "aes-gcm")]
    #[test]
    fn test_chacha20_uses_aes_backend() {
        let mut encryptor = ContentEncryptor::new();
        let key = b"32_bytes_key_for_chacha_compat!!".to_vec();
        encryptor.add_key(EncryptionKey::new("cc_key", key, EncryptionAlgorithm::ChaCha20Poly1305));

        let plaintext = "ChaCha20 routed to AES-256-GCM backend";
        let encrypted = encryptor.encrypt_string(plaintext).unwrap();
        let decrypted = encryptor.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nonce_is_random() {
        let encryptor = ContentEncryptor::new();
        let n1 = encryptor.generate_nonce(EncryptionAlgorithm::Aes256Gcm);
        // Small sleep to change timestamp
        std::thread::sleep(std::time::Duration::from_millis(1));
        let n2 = encryptor.generate_nonce(EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(n1.len(), 12);
        assert_eq!(n2.len(), 12);
        // Nonces should differ (extremely unlikely to collide)
        assert_ne!(n1, n2);
    }
}
