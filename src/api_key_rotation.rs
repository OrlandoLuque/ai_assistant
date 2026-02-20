//! API key rotation
//!
//! Automatic rotation and management of API keys.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// API key status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyStatus {
    Active,
    RateLimited,
    Expired,
    Revoked,
    Rotating,
}

/// API key entry
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub id: String,
    pub key: String,
    pub provider: String,
    pub status: KeyStatus,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub last_used: Option<Instant>,
    pub use_count: u64,
    pub error_count: u64,
    pub rate_limit_until: Option<Instant>,
}

impl ApiKey {
    pub fn new(id: &str, key: &str, provider: &str) -> Self {
        Self {
            id: id.to_string(),
            key: key.to_string(),
            provider: provider.to_string(),
            status: KeyStatus::Active,
            created_at: Instant::now(),
            expires_at: None,
            last_used: None,
            use_count: 0,
            error_count: 0,
            rate_limit_until: None,
        }
    }

    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(Instant::now() + duration);
        self
    }

    pub fn is_usable(&self) -> bool {
        if self.status != KeyStatus::Active {
            return false;
        }

        if let Some(expires) = self.expires_at {
            if Instant::now() > expires {
                return false;
            }
        }

        if let Some(until) = self.rate_limit_until {
            if Instant::now() < until {
                return false;
            }
        }

        true
    }

    pub fn mark_used(&mut self) {
        self.last_used = Some(Instant::now());
        self.use_count += 1;
    }

    pub fn mark_error(&mut self) {
        self.error_count += 1;
    }

    pub fn mark_rate_limited(&mut self, duration: Duration) {
        self.status = KeyStatus::RateLimited;
        self.rate_limit_until = Some(Instant::now() + duration);
    }
}

/// Key rotation configuration
#[derive(Debug, Clone)]
pub struct RotationConfig {
    pub auto_rotate: bool,
    pub rotation_interval: Option<Duration>,
    pub max_errors_before_rotation: u64,
    pub rate_limit_recovery_time: Duration,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            auto_rotate: true,
            rotation_interval: Some(Duration::from_secs(86400 * 30)), // 30 days
            max_errors_before_rotation: 10,
            rate_limit_recovery_time: Duration::from_secs(60),
        }
    }
}

/// API key manager with rotation
pub struct ApiKeyManager {
    keys: HashMap<String, Vec<ApiKey>>, // provider -> keys
    current_index: HashMap<String, usize>,
    config: RotationConfig,
}

impl ApiKeyManager {
    pub fn new(config: RotationConfig) -> Self {
        Self {
            keys: HashMap::new(),
            current_index: HashMap::new(),
            config,
        }
    }

    pub fn add_key(&mut self, key: ApiKey) {
        let provider = key.provider.clone();
        self.keys.entry(provider.clone()).or_default().push(key);
        self.current_index.entry(provider).or_insert(0);
    }

    pub fn get_key(&mut self, provider: &str) -> Option<&ApiKey> {
        self.cleanup_rate_limits(provider);

        let keys = self.keys.get(provider)?;
        let index = self.current_index.get(provider).copied().unwrap_or(0);

        // Find next usable key starting from current index
        for i in 0..keys.len() {
            let idx = (index + i) % keys.len();
            if keys[idx].is_usable() {
                self.current_index.insert(provider.to_string(), idx);
                return Some(&keys[idx]);
            }
        }

        None
    }

    pub fn mark_used(&mut self, provider: &str, key_id: &str) {
        if let Some(keys) = self.keys.get_mut(provider) {
            if let Some(key) = keys.iter_mut().find(|k| k.id == key_id) {
                key.mark_used();
            }
        }
    }

    pub fn mark_error(&mut self, provider: &str, key_id: &str) {
        if let Some(keys) = self.keys.get_mut(provider) {
            if let Some(key) = keys.iter_mut().find(|k| k.id == key_id) {
                key.mark_error();

                if self.config.auto_rotate
                    && key.error_count >= self.config.max_errors_before_rotation
                {
                    self.rotate(provider);
                }
            }
        }
    }

    pub fn mark_rate_limited(&mut self, provider: &str, key_id: &str) {
        if let Some(keys) = self.keys.get_mut(provider) {
            if let Some(key) = keys.iter_mut().find(|k| k.id == key_id) {
                key.mark_rate_limited(self.config.rate_limit_recovery_time);
            }
        }

        // Auto-rotate to next key
        if self.config.auto_rotate {
            self.rotate(provider);
        }
    }

    pub fn rotate(&mut self, provider: &str) {
        if let Some(keys) = self.keys.get(provider) {
            if let Some(index) = self.current_index.get_mut(provider) {
                *index = (*index + 1) % keys.len();
            }
        }
    }

    pub fn revoke_key(&mut self, provider: &str, key_id: &str) {
        if let Some(keys) = self.keys.get_mut(provider) {
            if let Some(key) = keys.iter_mut().find(|k| k.id == key_id) {
                key.status = KeyStatus::Revoked;
            }
        }
    }

    pub fn get_stats(&self, provider: &str) -> Option<KeyStats> {
        let keys = self.keys.get(provider)?;

        Some(KeyStats {
            total: keys.len(),
            active: keys
                .iter()
                .filter(|k| k.status == KeyStatus::Active)
                .count(),
            rate_limited: keys
                .iter()
                .filter(|k| k.status == KeyStatus::RateLimited)
                .count(),
            total_uses: keys.iter().map(|k| k.use_count).sum(),
            total_errors: keys.iter().map(|k| k.error_count).sum(),
        })
    }

    fn cleanup_rate_limits(&mut self, provider: &str) {
        if let Some(keys) = self.keys.get_mut(provider) {
            let now = Instant::now();
            for key in keys.iter_mut() {
                if key.status == KeyStatus::RateLimited {
                    if let Some(until) = key.rate_limit_until {
                        if now > until {
                            key.status = KeyStatus::Active;
                            key.rate_limit_until = None;
                        }
                    }
                }
            }
        }
    }
}

impl Default for ApiKeyManager {
    fn default() -> Self {
        Self::new(RotationConfig::default())
    }
}

/// Key statistics
#[derive(Debug, Clone)]
pub struct KeyStats {
    pub total: usize,
    pub active: usize,
    pub rate_limited: usize,
    pub total_uses: u64,
    pub total_errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_rotation() {
        let mut manager = ApiKeyManager::default();

        manager.add_key(ApiKey::new("key1", "secret1", "openai"));
        manager.add_key(ApiKey::new("key2", "secret2", "openai"));

        let key = manager.get_key("openai").unwrap();
        assert_eq!(key.id, "key1");

        manager.mark_rate_limited("openai", "key1");

        let key = manager.get_key("openai").unwrap();
        assert_eq!(key.id, "key2");
    }

    #[test]
    fn test_key_expiry() {
        // Create a key that expires immediately (zero duration)
        let mut key = ApiKey::new("k1", "secret", "openai");
        // Set expires_at to a time already in the past by using a very short duration
        // and then waiting, or more reliably, set it directly.
        // with_expiry sets expires_at to now + duration, so Duration::ZERO means it expires at now.
        key = key.with_expiry(Duration::from_secs(0));
        // Instant::now() >= expires_at, so is_usable should return false
        assert!(
            !key.is_usable(),
            "Key with zero-duration expiry should not be usable"
        );

        // Also verify a key with a long expiry IS usable
        let long_key =
            ApiKey::new("k2", "secret2", "openai").with_expiry(Duration::from_secs(3600));
        assert!(
            long_key.is_usable(),
            "Key with future expiry should be usable"
        );
    }

    #[test]
    fn test_status_transitions() {
        let mut key = ApiKey::new("k1", "secret", "openai");
        assert_eq!(key.status, KeyStatus::Active);

        // Active -> RateLimited
        key.mark_rate_limited(Duration::from_secs(60));
        assert_eq!(key.status, KeyStatus::RateLimited);

        // RateLimited -> Revoked (via direct status set, as revoke_key does)
        key.status = KeyStatus::Revoked;
        assert_eq!(key.status, KeyStatus::Revoked);
        assert!(!key.is_usable(), "Revoked key should not be usable");
    }

    #[test]
    fn test_use_and_error_counting() {
        let mut key = ApiKey::new("k1", "secret", "openai");
        assert_eq!(key.use_count, 0);
        assert_eq!(key.error_count, 0);

        key.mark_used();
        key.mark_used();
        key.mark_used();
        assert_eq!(key.use_count, 3);
        assert!(key.last_used.is_some());

        key.mark_error();
        key.mark_error();
        assert_eq!(key.error_count, 2);
    }

    #[test]
    fn test_get_stats() {
        let mut manager = ApiKeyManager::default();

        manager.add_key(ApiKey::new("k1", "s1", "openai"));
        manager.add_key(ApiKey::new("k2", "s2", "openai"));
        manager.add_key(ApiKey::new("k3", "s3", "openai"));

        // Record some usage and errors
        manager.mark_used("openai", "k1");
        manager.mark_used("openai", "k1");
        manager.mark_used("openai", "k2");
        manager.mark_error("openai", "k3");

        // Revoke one key
        manager.revoke_key("openai", "k3");

        let stats = manager.get_stats("openai").unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.active, 2); // k1 and k2 active, k3 revoked
        assert_eq!(stats.total_uses, 3); // k1 used 2x, k2 used 1x
        assert_eq!(stats.total_errors, 1); // k3 had 1 error

        // No stats for unknown provider
        assert!(manager.get_stats("anthropic").is_none());
    }
}
