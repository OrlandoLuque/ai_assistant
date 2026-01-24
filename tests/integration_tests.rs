//! Integration tests for AI Assistant new modules
//!
//! These tests verify that the new modules work correctly.

// === Error Module Tests ===

mod error_tests {
    use ai_assistant::error::*;

    #[test]
    fn test_error_chain() {
        let config_err = ConfigError::MissingValue {
            field: "api_key".to_string(),
            description: "Required for authentication".to_string(),
        };
        let ai_err: AiError = config_err.into();

        assert_eq!(ai_err.code(), "CONFIG");
        assert!(ai_err.suggestion().is_some());
        assert!(!ai_err.is_recoverable());
    }

    #[test]
    fn test_provider_error_recoverable() {
        let err = AiError::rate_limited(100, 60);
        assert!(err.is_recoverable());

        let err = AiError::model_not_found("ollama", "nonexistent");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_suggestions() {
        let err = AiError::context_exceeded(4096, 5000);
        assert!(err.suggestion().unwrap().contains("RAG"));

        let err = AiError::provider_unavailable("Ollama", "http://localhost:11434");
        assert!(err.suggestion().unwrap().contains("running"));
    }

    #[test]
    fn test_rag_errors() {
        let err = AiError::append_only_violation("delete", "guide.md");
        assert!(err.to_string().contains("append-only"));
        assert!(err.suggestion().unwrap().contains("set_append_only_mode"));
    }
}

// === Progress Module Tests ===

mod progress_tests {
    use ai_assistant::progress::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::Duration;

    #[test]
    fn test_progress_percentage() {
        let p = Progress::new("test", 25, 100);
        assert_eq!(p.percentage(), 25);

        let p = Progress::new("test", 0, 0);
        assert_eq!(p.percentage(), 0);

        let p = Progress::complete("test");
        assert_eq!(p.percentage(), 100);
    }

    #[test]
    fn test_progress_reporter() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let callback: ProgressCallback = Box::new(move |_p| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let mut reporter = ProgressReporter::new(Some(callback));
        reporter.set_min_interval(Duration::from_millis(0));
        reporter.start("test_op", 5);

        for i in 1..=5 {
            reporter.update(i, format!("Step {}", i));
        }

        reporter.complete("Done!");

        assert!(count.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_progress_time_estimates() {
        let mut p = Progress::new("test", 50, 100);
        p.elapsed_ms = 5000; // 5 seconds for 50 items
        p.estimate_remaining();

        assert!(p.remaining_ms.is_some());
        let remaining = p.remaining_ms.unwrap();
        assert!(remaining > 4000 && remaining < 6000);
    }

    #[test]
    fn test_progress_aggregator() {
        let agg = ProgressAggregator::new("batch", 5, None);

        agg.record_success();
        agg.record_success();
        agg.record_success();
        agg.record_failure();
        agg.record_failure();

        let progress = agg.get_progress();
        assert!(progress.is_complete);
        assert!(progress.is_error);
        assert_eq!(progress.current, 5);
        assert!(progress.message.contains("3"));
        assert!(progress.message.contains("2 failed"));
    }

    #[test]
    fn test_remaining_time_format() {
        let mut p = Progress::new("test", 50, 100);
        p.remaining_ms = Some(65000); // 65 seconds
        assert_eq!(p.remaining_human(), Some("1m 5s".to_string()));

        p.remaining_ms = Some(3665000); // 1h 1m 5s
        assert_eq!(p.remaining_human(), Some("1h 1m".to_string()));
    }
}

// === Config File Module Tests ===

mod config_file_tests {
    use ai_assistant::config_file::*;
    use ai_assistant::AiProvider;

    #[test]
    fn test_parse_toml_basic() {
        let toml = r#"
[provider]
type = "ollama"
model = "llama2"

[generation]
temperature = 0.8
max_history = 30
"#;
        let config = ConfigFile::parse(toml, ConfigFormat::Toml).unwrap();
        assert_eq!(config.provider.provider_type, "ollama");
        assert_eq!(config.provider.model, "llama2");
        assert_eq!(config.generation.temperature, 0.8);
        assert_eq!(config.generation.max_history, 30);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "provider": {
                "type": "lmstudio",
                "model": "mistral-7b"
            },
            "generation": {
                "temperature": 0.5
            },
            "rag": {
                "knowledge_enabled": true,
                "knowledge_tokens": 3000
            }
        }"#;

        let config = ConfigFile::parse(json, ConfigFormat::Json).unwrap();
        assert_eq!(config.provider.provider_type, "lmstudio");
        assert_eq!(config.provider.model, "mistral-7b");
        assert_eq!(config.generation.temperature, 0.5);
        assert!(config.rag.knowledge_enabled);
        assert_eq!(config.rag.knowledge_tokens, 3000);
    }

    #[test]
    fn test_to_ai_config() {
        let mut config = ConfigFile::default();
        config.provider.provider_type = "ollama".to_string();
        config.provider.model = "phi3".to_string();
        config.generation.temperature = 0.9;
        config.generation.max_history = 50;

        let ai_config = config.to_ai_config();
        assert!(matches!(ai_config.provider, AiProvider::Ollama));
        assert_eq!(ai_config.selected_model, "phi3");
        assert_eq!(ai_config.temperature, 0.9);
        assert_eq!(ai_config.max_history_messages, 50);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(ConfigFormat::from_content("{\"key\": \"value\"}"), ConfigFormat::Json);
        assert_eq!(ConfigFormat::from_content("[section]\nkey = \"value\""), ConfigFormat::Toml);
        assert_eq!(ConfigFormat::from_content("key = \"value\""), ConfigFormat::Toml);
    }

    #[test]
    fn test_validation() {
        let mut config = ConfigFile::default();
        config.generation.temperature = 0.7;
        assert!(config.validate().is_ok());

        config.generation.temperature = 3.0; // Invalid
        assert!(config.validate().is_err());
    }
}

// === Memory Management Tests ===

mod memory_management_tests {
    use ai_assistant::memory_management::*;

    #[test]
    fn test_bounded_cache_lru_eviction() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(3, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Access "a" to make it recently used
        cache.get(&"a".to_string());

        // Insert "d", should evict "b" (least recently used after "a" was touched)
        cache.insert("d".to_string(), 4);

        assert!(cache.contains(&"a".to_string()));
        assert!(!cache.contains(&"b".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_bounded_cache_fifo_eviction() {
        let mut cache: BoundedCache<i32, &str> = BoundedCache::new(3, EvictionPolicy::Fifo);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.insert(4, "four");

        assert!(!cache.contains(&1)); // First in, first out
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_bounded_cache_stats() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(10, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        cache.get(&"a".to_string()); // Hit
        cache.get(&"c".to_string()); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.entries, 2);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bounded_vec() {
        let mut vec: BoundedVec<i32> = BoundedVec::new(5);

        for i in 1..=10 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.eviction_count(), 5);
        assert_eq!(vec.get(0), Some(&6)); // Oldest kept is 6
        assert_eq!(vec.get(4), Some(&10)); // Newest is 10
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::with_limit(1000);

        tracker.register("cache", Some(500));
        tracker.register("embeddings", Some(500));

        tracker.update("cache", 200);
        tracker.update("embeddings", 300);

        assert_eq!(tracker.total_usage(), 500);
        assert_eq!(tracker.pressure(), MemoryPressure::Normal);

        // Push into warning zone
        tracker.update("cache", 450);
        tracker.update("embeddings", 450);

        assert_eq!(tracker.total_usage(), 900);
        assert_eq!(tracker.pressure(), MemoryPressure::Warning);

        // Push into critical zone
        tracker.update("cache", 600);

        assert_eq!(tracker.pressure(), MemoryPressure::Critical);
    }

    #[test]
    fn test_memory_report() {
        let mut tracker = MemoryTracker::with_limit(1000);

        tracker.register("cache", None);
        tracker.register("data", None);

        tracker.update("cache", 200);
        tracker.update("data", 300);

        let report = tracker.report();

        assert_eq!(report.total_bytes, 500);
        assert!(report.limit_bytes == Some(1000));
        assert_eq!(report.components.len(), 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
