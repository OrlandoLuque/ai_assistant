//! WebAssembly support utilities
//!
//! This module provides utilities for running in WebAssembly environments.
//! It includes polyfills and workarounds for features not available in WASM.
//!
//! # Feature Flags
//!
//! Enable the `wasm` feature in Cargo.toml to enable WASM-specific code paths:
//!
//! ```toml
//! [dependencies]
//! ai_assistant = { version = "0.1", features = ["wasm"] }
//! ```
//!
//! # Platform Detection
//!
//! Use the `is_wasm()` function to check if running in WASM:
//!
//! ```rust
//! use ai_assistant::wasm::is_wasm;
//!
//! if is_wasm() {
//!     // WASM-specific code
//! } else {
//!     // Native code
//! }
//! ```

/// Check if we're running in a WASM environment
#[inline]
pub fn is_wasm() -> bool {
    cfg!(target_arch = "wasm32")
}

/// Platform capabilities for WASM vs native
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlatformCapabilities {
    /// Whether file system access is available
    pub filesystem: bool,
    /// Whether network requests are available
    pub network: bool,
    /// Whether multithreading is available
    pub threads: bool,
    /// Whether system time is available
    pub system_time: bool,
    /// Whether SQLite is available
    pub sqlite: bool,
}

impl PlatformCapabilities {
    /// Get capabilities for the current platform
    pub fn current() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                filesystem: false,
                network: true,     // Via fetch API
                threads: false,    // No threads without SharedArrayBuffer
                system_time: true, // Via js_sys
                sqlite: false,     // No native SQLite
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                filesystem: true,
                network: true,
                threads: true,
                system_time: true,
                sqlite: true,
            }
        }
    }

    /// Check if all required capabilities are available
    pub fn has_all(&self, required: &[Capability]) -> bool {
        required.iter().all(|cap| self.has(*cap))
    }

    /// Check if a specific capability is available
    pub fn has(&self, cap: Capability) -> bool {
        match cap {
            Capability::Filesystem => self.filesystem,
            Capability::Network => self.network,
            Capability::Threads => self.threads,
            Capability::SystemTime => self.system_time,
            Capability::Sqlite => self.sqlite,
        }
    }
}

impl Default for PlatformCapabilities {
    fn default() -> Self {
        Self::current()
    }
}

/// Individual platform capabilities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Capability {
    /// File system access
    Filesystem,
    /// Network requests
    Network,
    /// Multithreading
    Threads,
    /// System time access
    SystemTime,
    /// SQLite database
    Sqlite,
}

/// Time utilities that work on both native and WASM
pub mod time {
    use std::time::Duration;

    /// Get current timestamp in milliseconds (works on WASM)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn now_millis() -> u64 {
        js_sys::Date::now() as u64
    }

    /// Get current timestamp in milliseconds (WASM without wasm feature)
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    pub fn now_millis() -> u64 {
        0 // Fallback: enable the `wasm` feature for real js_sys::Date time support
    }

    /// Get current timestamp in milliseconds (native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Sleep for a duration (no-op on WASM — blocking sleep is not possible)
    #[cfg(target_arch = "wasm32")]
    pub fn sleep(_duration: Duration) {
        // No-op: WASM cannot do blocking sleep. Use async setTimeout via web_sys instead.
    }

    /// Sleep for a duration (native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sleep(duration: Duration) {
        std::thread::sleep(duration);
    }

    /// Instant replacement for WASM
    #[derive(Debug, Clone, Copy)]
    pub struct Instant {
        millis: u64,
    }

    impl Instant {
        /// Create a new instant
        pub fn now() -> Self {
            Self {
                millis: now_millis(),
            }
        }

        /// Duration since this instant
        pub fn elapsed(&self) -> Duration {
            let now = now_millis();
            Duration::from_millis(now.saturating_sub(self.millis))
        }

        /// Duration from another instant
        pub fn duration_since(&self, earlier: Self) -> Duration {
            Duration::from_millis(self.millis.saturating_sub(earlier.millis))
        }
    }
}

/// HTTP client abstraction for WASM/native
pub mod http {
    /// HTTP method
    #[derive(Debug, Clone, Copy)]
    pub enum Method {
        Get,
        Post,
        Put,
        Delete,
    }

    /// HTTP response
    #[derive(Debug, Clone)]
    pub struct Response {
        pub status: u16,
        pub body: String,
    }

    /// Simple HTTP request (native implementation)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn request(method: Method, url: &str, body: Option<&str>) -> Result<Response, String> {
        let agent = ureq::agent();

        let request = match method {
            Method::Get => agent.get(url),
            Method::Post => agent.post(url),
            Method::Put => agent.put(url),
            Method::Delete => agent.delete(url),
        };

        let response = if let Some(body) = body {
            request
                .set("Content-Type", "application/json")
                .send_string(body)
        } else {
            request.call()
        };

        match response {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.into_string().unwrap_or_default();
                Ok(Response { status, body })
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Simple HTTP request (WASM — sync HTTP is not possible, use async fetch instead)
    #[cfg(target_arch = "wasm32")]
    pub fn request(_method: Method, _url: &str, _body: Option<&str>) -> Result<Response, String> {
        // WASM cannot perform synchronous HTTP. Use the async web_sys fetch API instead.
        Err("Synchronous HTTP is not available in WASM. Use web_sys::window().fetch() for async requests.".to_string())
    }
}

/// Storage abstraction for WASM (uses localStorage) vs native (file system)
pub mod storage {
    /// Simple key-value storage
    pub trait Storage {
        fn get(&self, key: &str) -> Option<String>;
        fn set(&mut self, key: &str, value: &str) -> Result<(), String>;
        fn remove(&mut self, key: &str) -> Result<(), String>;
        fn keys(&self) -> Vec<String>;
    }

    /// In-memory storage (works everywhere)
    #[derive(Debug, Default)]
    pub struct MemoryStorage {
        data: std::collections::HashMap<String, String>,
    }

    impl Storage for MemoryStorage {
        fn get(&self, key: &str) -> Option<String> {
            self.data.get(key).cloned()
        }

        fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
            self.data.insert(key.to_string(), value.to_string());
            Ok(())
        }

        fn remove(&mut self, key: &str) -> Result<(), String> {
            self.data.remove(key);
            Ok(())
        }

        fn keys(&self) -> Vec<String> {
            self.data.keys().cloned().collect()
        }
    }

    /// Create appropriate storage for the platform
    pub fn create_storage() -> Box<dyn Storage> {
        Box::new(MemoryStorage::default())
    }
}

/// Console logging that works on both platforms
pub mod console {
    /// Log a message (debug level) — real implementation via `web_sys::console`
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn log(message: &str) {
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(message));
    }

    /// Log a message (debug level) — no-op without `wasm` feature
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    pub fn log(_message: &str) {
        // Enable the `wasm` feature for real console.log support
    }

    /// Log a message (debug level) — native
    #[cfg(not(target_arch = "wasm32"))]
    pub fn log(message: &str) {
        println!("{}", message);
    }

    /// Log an error message — real implementation via `web_sys::console`
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn error(message: &str) {
        web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(message));
    }

    /// Log an error message — no-op without `wasm` feature
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    pub fn error(_message: &str) {
        // Enable the `wasm` feature for real console.error support
    }

    /// Log an error message — native
    #[cfg(not(target_arch = "wasm32"))]
    pub fn error(message: &str) {
        log::error!("{}", message);
    }

    /// Log a warning message — real implementation via `web_sys::console`
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn warn(message: &str) {
        web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(message));
    }

    /// Log a warning message — no-op without `wasm` feature
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    pub fn warn(_message: &str) {
        // Enable the `wasm` feature for real console.warn support
    }

    /// Log a warning message — native
    #[cfg(not(target_arch = "wasm32"))]
    pub fn warn(message: &str) {
        log::warn!("{}", message);
    }
}

/// Random number generation for WASM
pub mod random {
    /// Generate a random u64 using `getrandom` (cryptographically secure on WASM)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn random_u64() -> u64 {
        let mut buf = [0u8; 8];
        getrandom::getrandom(&mut buf).expect("getrandom failed");
        u64::from_le_bytes(buf)
    }

    /// Generate a random u64 — fallback without `wasm` feature (weak, time-based)
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
    pub fn random_u64() -> u64 {
        // Enable the `wasm` feature for cryptographically secure randomness via getrandom
        super::time::now_millis()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    /// Generate a random u64 — native (time-seeded LCG)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn random_u64() -> u64 {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        seed.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    /// Generate a random UUID v4
    pub fn random_uuid() -> String {
        let r1 = random_u64();
        let r2 = random_u64();

        format!(
            "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
            (r1 >> 32) as u32,
            ((r1 >> 16) & 0xFFFF) as u16,
            (r1 & 0x0FFF) as u16,
            (0x8000 | ((r2 >> 48) & 0x3FFF)) as u16,
            (r2 & 0xFFFFFFFFFFFF)
        )
    }
}

/// Feature detection helpers
pub mod features {
    use super::{Capability, PlatformCapabilities};

    /// Check if RAG functionality is available
    pub fn rag_available() -> bool {
        let caps = PlatformCapabilities::current();
        caps.has(Capability::Sqlite) && caps.has(Capability::Filesystem)
    }

    /// Check if conversation persistence is available
    pub fn persistence_available() -> bool {
        PlatformCapabilities::current().has(Capability::Filesystem)
    }

    /// Check if async HTTP is required (for WASM)
    pub fn requires_async_http() -> bool {
        cfg!(target_arch = "wasm32")
    }

    /// Check if threading is available
    pub fn threading_available() -> bool {
        PlatformCapabilities::current().has(Capability::Threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_capabilities() {
        let caps = PlatformCapabilities::current();

        // On native, all capabilities should be available
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(caps.filesystem);
            assert!(caps.network);
            assert!(caps.threads);
            assert!(caps.system_time);
            assert!(caps.sqlite);
        }
    }

    #[test]
    fn test_has_capability() {
        let caps = PlatformCapabilities::current();

        #[cfg(not(target_arch = "wasm32"))]
        assert!(caps.has(Capability::Filesystem));
    }

    #[test]
    fn test_random_uuid() {
        let uuid1 = random::random_uuid();
        let uuid2 = random::random_uuid();

        // UUIDs should be different
        assert_ne!(uuid1, uuid2);

        // UUID format check (basic)
        assert_eq!(uuid1.len(), 36);
        assert!(uuid1.contains('-'));
    }

    #[test]
    fn test_memory_storage() {
        use storage::Storage;

        let mut storage = storage::MemoryStorage::default();

        storage.set("key1", "value1").unwrap();
        assert_eq!(storage.get("key1"), Some("value1".to_string()));

        storage.remove("key1").unwrap();
        assert_eq!(storage.get("key1"), None);
    }

    #[test]
    fn test_time_instant() {
        let start = time::Instant::now();
        // Small delay
        for _ in 0..1000 {
            let _ = 1 + 1;
        }
        let _elapsed = start.elapsed();
    }

    #[test]
    fn test_console_functions_native() {
        // On native, these should not panic
        console::log("test log message");
        console::error("test error message");
        console::warn("test warn message");
    }

    #[test]
    fn test_feature_detection_native() {
        // On native, all features should be available
        assert!(features::rag_available());
        assert!(features::persistence_available());
        assert!(!features::requires_async_http());
        assert!(features::threading_available());
    }

    #[test]
    fn test_has_all_capabilities() {
        let caps = PlatformCapabilities::current();
        assert!(caps.has_all(&[Capability::Filesystem, Capability::Network]));
        assert!(caps.has_all(&[])); // Empty list always true
    }

    #[test]
    fn test_now_millis_reasonable() {
        let ts = time::now_millis();
        // Should be a recent Unix timestamp in milliseconds (after 2020-01-01)
        assert!(ts > 1_577_836_800_000);
    }
}
