//! Context usage tracking and token estimation
//!
//! Provides token estimation, model context window sizes, and a global
//! cache for context size lookups that avoids repeated API calls.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Global cache for model context window sizes.
///
/// Keyed by lowercase model name. Populated by `get_model_context_size_cached`
/// and shared across all `AiAssistant` instances.
static CONTEXT_SIZE_CACHE: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Context usage information
#[derive(Debug, Clone)]
pub struct ContextUsage {
    /// Tokens used by system prompt
    pub system_tokens: usize,
    /// Tokens used by knowledge/context
    pub knowledge_tokens: usize,
    /// Tokens used by conversation history
    pub conversation_tokens: usize,
    /// Total tokens in use
    pub total_tokens: usize,
    /// Maximum tokens for the model
    pub max_tokens: usize,
    /// Usage percentage (0-100)
    pub usage_percent: f32,
    /// True if usage is above 70%
    pub is_warning: bool,
    /// True if usage is above 90%
    pub is_critical: bool,
}

impl ContextUsage {
    /// Create a new context usage calculation
    pub fn calculate(
        system_tokens: usize,
        knowledge_tokens: usize,
        conversation_tokens: usize,
        max_tokens: usize,
    ) -> Self {
        let total_tokens = system_tokens + knowledge_tokens + conversation_tokens;

        // Reserve ~20% for response generation
        let effective_max = (max_tokens as f32 * 0.8) as usize;
        let usage_percent = (total_tokens as f32 / effective_max as f32) * 100.0;

        Self {
            system_tokens,
            knowledge_tokens,
            conversation_tokens,
            total_tokens,
            max_tokens,
            usage_percent: usage_percent.min(100.0),
            is_warning: usage_percent > 70.0,
            is_critical: usage_percent > 90.0,
        }
    }

    /// Get remaining tokens available
    pub fn remaining_tokens(&self) -> usize {
        let effective_max = (self.max_tokens as f32 * 0.8) as usize;
        effective_max.saturating_sub(self.total_tokens)
    }
}

/// Estimate token count from text
///
/// Uses an approximation of ~3.5 characters per token, which works well
/// for mixed content (English ~4 chars/token, Spanish ~3 chars/token,
/// code ~2.5 chars/token).
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / 3.5).ceil() as usize
}

/// Get the context window size for common models (in tokens)
pub fn get_model_context_size(model_name: &str) -> usize {
    let name = model_name.to_lowercase();

    // Llama 3.x models
    if name.contains("llama3.2") || name.contains("llama-3.2") {
        return 128_000;
    }
    if name.contains("llama3.1") || name.contains("llama-3.1") || name.contains("llama3:") {
        return 128_000;
    }
    if name.contains("llama2") || name.contains("llama-2") {
        return 4_096;
    }

    // Qwen models
    if name.contains("qwen2.5") || name.contains("qwen2") {
        if name.contains("32b") || name.contains("72b") {
            return 128_000;
        }
        return 32_768;
    }
    if name.contains("qwen") {
        return 32_768;
    }

    // Mistral models
    if name.contains("mixtral") {
        return 32_768;
    }
    if name.contains("mistral") {
        return 32_768;
    }

    // Phi models
    if name.contains("phi3") || name.contains("phi-3") {
        if name.contains("mini") {
            return 4_096;
        }
        return 128_000;
    }
    if name.contains("phi") {
        return 2_048;
    }

    // Gemma models
    if name.contains("gemma2") || name.contains("gemma-2") {
        return 8_192;
    }
    if name.contains("gemma") {
        return 8_192;
    }

    // DeepSeek models
    if name.contains("deepseek") {
        return 32_768;
    }

    // CodeLlama
    if name.contains("codellama") {
        return 16_384;
    }

    // Default conservative estimate
    8_192
}

/// Get the context window size for a model, using a global cache.
///
/// Lookup order:
/// 1. **Global cache** — instant, no network
/// 2. **Provider API** — via the caller-supplied `fetcher` closure
/// 3. **Static table** — `get_model_context_size()` pattern matching
///
/// Results from steps 2 and 3 are stored in the cache for future lookups.
///
/// The `fetcher` closure receives the model name and should query the
/// provider API (e.g. `fetch_model_context_size`). Accepting a closure
/// avoids a circular dependency between `context` and `providers`.
///
/// # Example
///
/// ```rust,no_run
/// use ai_assistant::context::get_model_context_size_cached;
///
/// let size = get_model_context_size_cached("llama3.2:7b", |_name| {
///     // In production, call fetch_model_context_size(config, name) here
///     None
/// });
/// assert_eq!(size, 128_000);
/// ```
pub fn get_model_context_size_cached<F>(model_name: &str, fetcher: F) -> usize
where
    F: FnOnce(&str) -> Option<usize>,
{
    let key = model_name.to_lowercase();

    // 1. Check global cache
    if let Ok(cache) = CONTEXT_SIZE_CACHE.lock() {
        if let Some(&size) = cache.get(&key) {
            return size;
        }
    }

    // 2. Try provider API via fetcher
    let size = fetcher(model_name).unwrap_or_else(|| {
        // 3. Fall back to static table
        get_model_context_size(model_name)
    });

    // Store in cache
    if let Ok(mut cache) = CONTEXT_SIZE_CACHE.lock() {
        cache.insert(key, size);
    }

    size
}

/// Clear the global context size cache.
///
/// Useful in tests to ensure a clean state, or when switching providers
/// where the same model name might report a different context window.
pub fn clear_context_size_cache() {
    if let Ok(mut cache) = CONTEXT_SIZE_CACHE.lock() {
        cache.clear();
    }
}

/// Return the number of entries currently in the context size cache.
///
/// Primarily intended for testing and diagnostics.
pub fn context_size_cache_len() -> usize {
    CONTEXT_SIZE_CACHE.lock().map(|c| c.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        // Roughly 3.5 chars per token
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars / 3.5 = ~1.4, ceil = 2
        assert_eq!(estimate_tokens(""), 0);

        // Longer text
        let text = "This is a longer piece of text for testing.";
        let estimated = estimate_tokens(text);
        assert!(estimated > 0);
        assert!(estimated < text.len()); // Should be less than char count
    }

    #[test]
    fn test_model_context_size() {
        assert_eq!(get_model_context_size("llama3.2:7b"), 128_000);
        assert_eq!(get_model_context_size("llama2:7b"), 4_096);
        assert_eq!(get_model_context_size("qwen2.5:14b"), 32_768);
        assert_eq!(get_model_context_size("mistral:7b"), 32_768);
        assert_eq!(get_model_context_size("unknown-model"), 8_192); // default
    }

    #[test]
    fn test_context_usage() {
        let usage = ContextUsage::calculate(200, 1000, 500, 8192);

        assert_eq!(usage.total_tokens, 1700);
        assert!(usage.usage_percent < 30.0); // 1700 / (8192 * 0.8) = ~26%
        assert!(!usage.is_warning);
        assert!(!usage.is_critical);
    }

    #[test]
    fn test_context_usage_warning() {
        // Test warning level (70-90%)
        let usage = ContextUsage::calculate(200, 4000, 1500, 8192);
        assert!(usage.is_warning); // 5700 / 6553 = ~87%
        assert!(!usage.is_critical); // Not above 90%

        // Test critical level (>90%)
        let usage_critical = ContextUsage::calculate(200, 5000, 1500, 8192);
        assert!(usage_critical.is_warning);
        assert!(usage_critical.is_critical); // 6700 / 6553 = ~102%
    }

    #[test]
    fn test_cached_uses_static_table_when_fetcher_returns_none() {
        clear_context_size_cache();

        let size = get_model_context_size_cached("llama3.2:7b", |_| None);
        assert_eq!(size, 128_000); // falls back to static table
        assert!(context_size_cache_len() >= 1); // cached (other parallel tests may add entries)
    }

    #[test]
    fn test_cached_uses_fetcher_when_available() {
        clear_context_size_cache();

        let size = get_model_context_size_cached("custom-model-xyz", |_| Some(65_536));
        assert_eq!(size, 65_536); // from fetcher, not static table
    }

    #[test]
    fn test_cached_returns_cached_value_on_second_call() {
        // Use a unique key unlikely to collide with other parallel tests.
        // Do NOT clear the global cache here — another test could re-clear
        // it between our two calls, evicting the entry we just inserted.
        let unique_key = "cache-second-call-test-xyzzy-42";

        // First call — fetcher is invoked, value is cached
        let size1 = get_model_context_size_cached(unique_key, |_| Some(99_999));
        assert_eq!(size1, 99_999);

        // Second call — fetcher should NOT be invoked (cache hit)
        let size2 = get_model_context_size_cached(unique_key, |_| {
            panic!("fetcher should not be called on cache hit");
        });
        assert_eq!(size2, 99_999);
    }

    #[test]
    fn test_clear_context_size_cache() {
        clear_context_size_cache();

        get_model_context_size_cached("clear-test", |_| Some(42_000));
        assert!(context_size_cache_len() > 0);

        clear_context_size_cache();
        assert_eq!(context_size_cache_len(), 0);

        // After clear, fetcher is called again
        let size = get_model_context_size_cached("clear-test", |_| Some(77_000));
        assert_eq!(size, 77_000); // new value, not the old 42_000
    }

    #[test]
    fn test_cached_case_insensitive_key() {
        clear_context_size_cache();

        get_model_context_size_cached("MyModel:7B", |_| Some(50_000));

        // Same model, different case — should hit cache
        let size = get_model_context_size_cached("mymodel:7b", |_| {
            panic!("should use cache regardless of case");
        });
        assert_eq!(size, 50_000);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }
}
