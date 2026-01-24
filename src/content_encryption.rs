//! Content encryption
//!
//! Encrypt and decrypt conversation content.

use std::collections::HashMap;

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
                .unwrap()
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
                .unwrap()
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
        let ciphertext = self.encrypt_with_key(plaintext, &key.key, &nonce, key.algorithm);

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

        (0..size).map(|i| ((i * 17 + 42) % 256) as u8).collect()
    }

    fn encrypt_with_key(
        &self,
        plaintext: &[u8],
        key: &[u8],
        nonce: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Vec<u8> {
        match algorithm {
            EncryptionAlgorithm::Xor => {
                // Simple XOR for demo (NOT SECURE)
                plaintext.iter()
                    .enumerate()
                    .map(|(i, b)| b ^ key[i % key.len()])
                    .collect()
            }
            _ => {
                // In production, use proper crypto library
                // For now, use XOR with nonce mixing
                plaintext.iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let k = key[i % key.len()];
                        let n = if nonce.is_empty() { 0 } else { nonce[i % nonce.len()] };
                        b ^ k ^ n
                    })
                    .collect()
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
        // XOR encryption is symmetric
        Ok(self.encrypt_with_key(ciphertext, key, nonce, algorithm))
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
    fn test_encrypt_decrypt() {
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
}
