//! Model warmup for reducing cold start latency
//!
//! This module provides model warmup capabilities to keep models loaded
//! and ready to respond, reducing the latency of the first request.
//!
//! # Features
//!
//! - **Automatic warmup**: Keep frequently used models warm
//! - **Scheduled warmup**: Warm models at specific times
//! - **Health-aware warming**: Only warm healthy providers
//! - **Resource-conscious**: Configurable warmup intervals

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for model warmup
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Interval between warmup pings
    pub warmup_interval: Duration,
    /// Maximum models to keep warm simultaneously
    pub max_warm_models: usize,
    /// Warmup request timeout
    pub timeout: Duration,
    /// Simple prompt used for warming
    pub warmup_prompt: String,
    /// Automatically warm most-used models
    pub auto_warm_popular: bool,
    /// Number of top models to auto-warm
    pub auto_warm_count: usize,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            warmup_interval: Duration::from_secs(300), // 5 minutes
            max_warm_models: 3,
            timeout: Duration::from_secs(30),
            warmup_prompt: "Hi".to_string(),
            auto_warm_popular: true,
            auto_warm_count: 2,
        }
    }
}

/// Status of a warmed model
#[derive(Debug, Clone)]
pub struct WarmupStatus {
    /// Model identifier
    pub model: String,
    /// Provider URL/name
    pub provider: String,
    /// Last warmup time
    pub last_warmup: Instant,
    /// Last warmup latency
    pub warmup_latency: Duration,
    /// Whether model is currently warm
    pub is_warm: bool,
    /// Total warmup count
    pub warmup_count: usize,
    /// Error count
    pub error_count: usize,
    /// Last error message
    pub last_error: Option<String>,
}

/// Model usage statistics for auto-warming
#[derive(Debug, Clone, Default)]
pub struct ModelUsageStats {
    /// Request count per model
    pub request_counts: HashMap<String, usize>,
    /// Last request time per model
    pub last_request: HashMap<String, Instant>,
    /// Average latency per model
    pub avg_latency: HashMap<String, Duration>,
}

impl ModelUsageStats {
    /// Record a model usage
    pub fn record(&mut self, model: &str, latency: Duration) {
        *self.request_counts.entry(model.to_string()).or_insert(0) += 1;
        self.last_request.insert(model.to_string(), Instant::now());

        let avg = self
            .avg_latency
            .entry(model.to_string())
            .or_insert(Duration::ZERO);
        let count = self.request_counts[model];
        *avg = (*avg * (count - 1) as u32 + latency) / count as u32;
    }

    /// Get top N models by usage
    pub fn top_models(&self, n: usize) -> Vec<String> {
        let mut models: Vec<_> = self.request_counts.iter().collect();
        models.sort_by(|a, b| b.1.cmp(a.1));
        models.into_iter().take(n).map(|(k, _)| k.clone()).collect()
    }
}

/// Warmup manager for keeping models ready
pub struct WarmupManager {
    config: WarmupConfig,
    /// Status of warmed models
    status: Arc<RwLock<HashMap<String, WarmupStatus>>>,
    /// Models to keep warm
    warm_list: Arc<RwLock<Vec<(String, String)>>>, // (model, provider)
    /// Usage statistics
    usage_stats: Arc<Mutex<ModelUsageStats>>,
    /// Statistics
    stats: Arc<Mutex<WarmupStats>>,
}

/// Overall warmup statistics
#[derive(Debug, Clone, Default)]
pub struct WarmupStats {
    /// Total warmup attempts
    pub total_warmups: usize,
    /// Successful warmups
    pub successful_warmups: usize,
    /// Failed warmups
    pub failed_warmups: usize,
    /// Average warmup latency
    pub avg_warmup_latency: Duration,
    /// Currently warm models count
    pub warm_model_count: usize,
}

impl WarmupManager {
    /// Create a new warmup manager
    pub fn new(config: WarmupConfig) -> Self {
        Self {
            config,
            status: Arc::new(RwLock::new(HashMap::new())),
            warm_list: Arc::new(RwLock::new(Vec::new())),
            usage_stats: Arc::new(Mutex::new(ModelUsageStats::default())),
            stats: Arc::new(Mutex::new(WarmupStats::default())),
        }
    }

    /// Add a model to the warm list
    pub fn add_warm_model(&self, model: impl Into<String>, provider: impl Into<String>) {
        let model = model.into();
        let provider = provider.into();

        let mut list = self.warm_list.write().unwrap_or_else(|e| e.into_inner());

        // Check if already in list
        if !list.iter().any(|(m, p)| m == &model && p == &provider) {
            list.push((model, provider));

            // Enforce limit
            while list.len() > self.config.max_warm_models {
                list.remove(0);
            }
        }
    }

    /// Remove a model from the warm list
    pub fn remove_warm_model(&self, model: &str, provider: &str) {
        let mut list = self.warm_list.write().unwrap_or_else(|e| e.into_inner());
        list.retain(|(m, p)| m != model || p != provider);

        // Also remove status
        let key = format!("{}@{}", model, provider);
        if let Ok(mut status) = self.status.write() {
            status.remove(&key);
        }
    }

    /// Record model usage for auto-warming
    pub fn record_usage(&self, model: &str, latency: Duration) {
        if let Ok(mut stats) = self.usage_stats.lock() {
            stats.record(model, latency);
        }
    }

    /// Get models that need warming
    pub fn models_needing_warmup(&self) -> Vec<(String, String)> {
        let now = Instant::now();
        let status = self.status.read().unwrap_or_else(|e| e.into_inner());
        let list = self.warm_list.read().unwrap_or_else(|e| e.into_inner());

        let mut needing_warmup = Vec::new();

        for (model, provider) in list.iter() {
            let key = format!("{}@{}", model, provider);

            let needs_warmup = match status.get(&key) {
                Some(s) => {
                    !s.is_warm || now.duration_since(s.last_warmup) > self.config.warmup_interval
                }
                None => true,
            };

            if needs_warmup {
                needing_warmup.push((model.clone(), provider.clone()));
            }
        }

        needing_warmup
    }

    /// Perform warmup on a model
    pub fn warmup<F>(&self, model: &str, provider: &str, generate: F) -> Result<Duration, String>
    where
        F: Fn(&str, &str, &str) -> Result<String, String>,
    {
        let key = format!("{}@{}", model, provider);
        let start = Instant::now();

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_warmups += 1;
        }

        // Perform warmup request
        let result = generate(&self.config.warmup_prompt, model, provider);
        let latency = start.elapsed();

        let mut status = self.status.write().unwrap_or_else(|e| e.into_inner());
        let entry = status.entry(key.clone()).or_insert_with(|| WarmupStatus {
            model: model.to_string(),
            provider: provider.to_string(),
            last_warmup: Instant::now(),
            warmup_latency: Duration::ZERO,
            is_warm: false,
            warmup_count: 0,
            error_count: 0,
            last_error: None,
        });

        match result {
            Ok(_) => {
                entry.last_warmup = Instant::now();
                entry.warmup_latency = latency;
                entry.is_warm = true;
                entry.warmup_count += 1;
                entry.last_error = None;

                // Update global stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.successful_warmups += 1;
                    stats.warm_model_count = status.values().filter(|s| s.is_warm).count();

                    // Update average latency
                    let total = stats.successful_warmups;
                    stats.avg_warmup_latency =
                        (stats.avg_warmup_latency * (total - 1) as u32 + latency) / total as u32;
                }

                Ok(latency)
            }
            Err(e) => {
                entry.is_warm = false;
                entry.error_count += 1;
                entry.last_error = Some(e.clone());

                if let Ok(mut stats) = self.stats.lock() {
                    stats.failed_warmups += 1;
                    stats.warm_model_count = status.values().filter(|s| s.is_warm).count();
                }

                Err(e)
            }
        }
    }

    /// Warm all models that need it
    pub fn warmup_all<F>(&self, generate: F) -> Vec<(String, Result<Duration, String>)>
    where
        F: Fn(&str, &str, &str) -> Result<String, String>,
    {
        let needing = self.models_needing_warmup();
        let mut results = Vec::new();

        for (model, provider) in needing {
            let result = self.warmup(&model, &provider, &generate);
            results.push((format!("{}@{}", model, provider), result));
        }

        results
    }

    /// Auto-warm popular models based on usage
    pub fn auto_warm_popular(&self) {
        if !self.config.auto_warm_popular {
            return;
        }

        let top_models = {
            let stats = self.usage_stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.top_models(self.config.auto_warm_count)
        };

        // Note: We'd need provider info to actually warm them
        // This just tracks which models are popular
        for model in top_models {
            // Add to warm list with default provider
            // In practice, you'd track model -> provider mapping
            self.add_warm_model(&model, "default");
        }
    }

    /// Get warmup status for a model
    pub fn get_status(&self, model: &str, provider: &str) -> Option<WarmupStatus> {
        let key = format!("{}@{}", model, provider);
        self.status.read().ok()?.get(&key).cloned()
    }

    /// Check if a model is warm
    pub fn is_warm(&self, model: &str, provider: &str) -> bool {
        let key = format!("{}@{}", model, provider);
        self.status
            .read()
            .map(|s| s.get(&key).map(|st| st.is_warm).unwrap_or(false))
            .unwrap_or(false)
    }

    /// Get all warm models
    pub fn warm_models(&self) -> Vec<WarmupStatus> {
        self.status
            .read()
            .map(|s| s.values().filter(|st| st.is_warm).cloned().collect())
            .unwrap_or_default()
    }

    /// Get statistics
    pub fn stats(&self) -> WarmupStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Mark a model as cold (e.g., after long inactivity)
    pub fn mark_cold(&self, model: &str, provider: &str) {
        let key = format!("{}@{}", model, provider);
        if let Ok(mut status) = self.status.write() {
            if let Some(entry) = status.get_mut(&key) {
                entry.is_warm = false;
            }
        }
    }

    /// Clear all warmup data
    pub fn clear(&self) {
        if let Ok(mut status) = self.status.write() {
            status.clear();
        }
        if let Ok(mut list) = self.warm_list.write() {
            list.clear();
        }
    }
}

impl Default for WarmupManager {
    fn default() -> Self {
        Self::new(WarmupConfig::default())
    }
}

/// Scheduled warmup task
pub struct ScheduledWarmup {
    /// Models to warm at this time
    pub models: Vec<(String, String)>,
    /// When to warm (time of day)
    pub time: WarmupTime,
    /// Whether this schedule is active
    pub active: bool,
}

/// Time specification for scheduled warmup
#[derive(Debug, Clone)]
pub enum WarmupTime {
    /// Warm at specific hour (0-23)
    AtHour(u8),
    /// Warm every N minutes
    EveryMinutes(u32),
    /// Warm at startup
    OnStartup,
}

/// Builder for warmup configuration
pub struct WarmupConfigBuilder {
    config: WarmupConfig,
}

impl WarmupConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: WarmupConfig::default(),
        }
    }

    /// Set warmup interval
    pub fn interval(mut self, duration: Duration) -> Self {
        self.config.warmup_interval = duration;
        self
    }

    /// Set max warm models
    pub fn max_models(mut self, count: usize) -> Self {
        self.config.max_warm_models = count;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.config.timeout = duration;
        self
    }

    /// Set warmup prompt
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.warmup_prompt = prompt.into();
        self
    }

    /// Configure auto-warming
    pub fn auto_warm(mut self, enabled: bool, count: usize) -> Self {
        self.config.auto_warm_popular = enabled;
        self.config.auto_warm_count = count;
        self
    }

    /// Build the configuration
    pub fn build(self) -> WarmupConfig {
        self.config
    }
}

impl Default for WarmupConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for warmup-aware providers
pub trait WarmupAware {
    /// Check if provider supports warmup
    fn supports_warmup(&self) -> bool;

    /// Perform a warmup request
    fn warmup(&self, model: &str) -> Result<Duration, String>;

    /// Get recommended warmup interval
    fn recommended_interval(&self) -> Duration;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_manager_creation() {
        let manager = WarmupManager::default();
        assert_eq!(manager.warm_models().len(), 0);
    }

    #[test]
    fn test_add_warm_model() {
        let manager = WarmupManager::default();
        manager.add_warm_model("gpt-4", "openai");
        manager.add_warm_model("llama2", "ollama");

        let needing = manager.models_needing_warmup();
        assert_eq!(needing.len(), 2);
    }

    #[test]
    fn test_warmup_success() {
        let manager = WarmupManager::default();
        manager.add_warm_model("test-model", "test-provider");

        let result = manager.warmup("test-model", "test-provider", |_, _, _| {
            Ok("Response".to_string())
        });

        assert!(result.is_ok());
        assert!(manager.is_warm("test-model", "test-provider"));
    }

    #[test]
    fn test_warmup_failure() {
        let manager = WarmupManager::default();
        manager.add_warm_model("test-model", "test-provider");

        let result = manager.warmup("test-model", "test-provider", |_, _, _| {
            Err("Connection failed".to_string())
        });

        assert!(result.is_err());
        assert!(!manager.is_warm("test-model", "test-provider"));
    }

    #[test]
    fn test_usage_stats() {
        let mut stats = ModelUsageStats::default();

        stats.record("model1", Duration::from_millis(100));
        stats.record("model1", Duration::from_millis(200));
        stats.record("model2", Duration::from_millis(50));

        let top = stats.top_models(1);
        assert_eq!(top[0], "model1");
    }

    #[test]
    fn test_config_builder() {
        let config = WarmupConfigBuilder::new()
            .interval(Duration::from_secs(600))
            .max_models(5)
            .timeout(Duration::from_secs(60))
            .prompt("Hello")
            .auto_warm(true, 3)
            .build();

        assert_eq!(config.warmup_interval, Duration::from_secs(600));
        assert_eq!(config.max_warm_models, 5);
        assert_eq!(config.warmup_prompt, "Hello");
    }

    #[test]
    fn test_max_warm_models_limit() {
        let config = WarmupConfig {
            max_warm_models: 2,
            ..Default::default()
        };
        let manager = WarmupManager::new(config);

        manager.add_warm_model("model1", "provider");
        manager.add_warm_model("model2", "provider");
        manager.add_warm_model("model3", "provider");

        let needing = manager.models_needing_warmup();
        assert_eq!(needing.len(), 2);
    }
}
