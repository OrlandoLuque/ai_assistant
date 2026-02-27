//! Memory management utilities with limits and eviction policies
//!
//! This module provides bounded caches and memory tracking to prevent
//! unbounded growth of in-memory data structures.
//!
//! # Features
//!
//! - **BoundedCache**: Fixed-size cache with LRU eviction
//! - **BoundedVec**: Vector with maximum size limit
//! - **MemoryTracker**: Track memory usage across components
//! - **EvictionPolicy**: Configurable eviction strategies
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::memory_management::{BoundedCache, EvictionPolicy};
//!
//! let mut cache: BoundedCache<String, Vec<f32>> = BoundedCache::new(100, EvictionPolicy::Lru);
//! cache.insert("key1".to_string(), vec![1.0, 2.0, 3.0]);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::{Duration, Instant};

/// Eviction policy for bounded collections
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict items accessed longest ago
    Lru,
    /// Least Frequently Used - evict items with lowest access count
    Lfu,
    /// First In First Out - evict oldest items
    Fifo,
    /// Time To Live - evict items older than TTL
    Ttl,
    /// Random eviction
    Random,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::Lru
    }
}

/// Statistics for cache operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Number of insertions
    pub insertions: u64,
    /// Current number of entries
    pub entries: usize,
    /// Total estimated memory usage in bytes
    pub memory_bytes: usize,
}

impl CacheStats {
    /// Calculate hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Entry metadata for cache management
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    size_bytes: usize,
}

impl<V> CacheEntry<V> {
    fn new(value: V, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            inserted_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// A bounded cache with configurable eviction policy
pub struct BoundedCache<K, V>
where
    K: Hash + Eq + Clone,
{
    entries: HashMap<K, CacheEntry<V>>,
    order: VecDeque<K>,
    max_entries: usize,
    max_memory_bytes: Option<usize>,
    current_memory: usize,
    policy: EvictionPolicy,
    ttl: Option<Duration>,
    stats: CacheStats,
}

impl<K, V> BoundedCache<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create a new bounded cache with maximum entries
    pub fn new(max_entries: usize, policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            max_memory_bytes: None,
            current_memory: 0,
            policy,
            ttl: None,
            stats: CacheStats::default(),
        }
    }

    /// Create with both entry and memory limits
    pub fn with_memory_limit(
        max_entries: usize,
        max_memory_bytes: usize,
        policy: EvictionPolicy,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            max_memory_bytes: Some(max_memory_bytes),
            current_memory: 0,
            policy,
            ttl: None,
            stats: CacheStats::default(),
        }
    }

    /// Set TTL for entries (used with Ttl policy)
    pub fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = Some(ttl);
        self.policy = EvictionPolicy::Ttl;
    }

    /// Insert a value with estimated size
    pub fn insert_with_size(&mut self, key: K, value: V, size_bytes: usize) {
        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_memory = self.current_memory.saturating_sub(old_entry.size_bytes);
            self.order.retain(|k| k != &key);
        }

        // Evict entries if necessary
        self.evict_if_needed(size_bytes);

        // Insert new entry
        self.entries
            .insert(key.clone(), CacheEntry::new(value, size_bytes));
        self.order.push_back(key);
        self.current_memory += size_bytes;
        self.stats.insertions += 1;
        self.stats.entries = self.entries.len();
        self.stats.memory_bytes = self.current_memory;
    }

    /// Insert a value (uses std::mem::size_of for size estimate)
    pub fn insert(&mut self, key: K, value: V) {
        let size = std::mem::size_of::<V>() + std::mem::size_of::<K>();
        self.insert_with_size(key, value, size);
    }

    /// Get a value (updates access time/count for LRU/LFU)
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Check TTL if applicable
            if let (EvictionPolicy::Ttl, Some(ttl)) = (self.policy, self.ttl) {
                if entry.inserted_at.elapsed() > ttl {
                    // Entry expired
                    self.stats.misses += 1;
                    return None;
                }
            }

            entry.touch();

            // Update order for LRU
            if self.policy == EvictionPolicy::Lru {
                self.order.retain(|k| k != key);
                self.order.push_back(key.clone());
            }

            self.stats.hits += 1;
            Some(&entry.value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Get a value without updating access metadata (peek)
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|e| &e.value)
    }

    /// Remove a value
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.order.retain(|k| k != key);
            self.current_memory = self.current_memory.saturating_sub(entry.size_bytes);
            self.stats.entries = self.entries.len();
            self.stats.memory_bytes = self.current_memory;
            Some(entry.value)
        } else {
            None
        }
    }

    /// Check if key exists
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Get current number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_memory = 0;
        self.stats.entries = 0;
        self.stats.memory_bytes = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Evict expired entries (for TTL policy)
    pub fn evict_expired(&mut self) {
        if let (EvictionPolicy::Ttl, Some(ttl)) = (self.policy, self.ttl) {
            let expired: Vec<K> = self
                .entries
                .iter()
                .filter(|(_, e)| e.inserted_at.elapsed() > ttl)
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired {
                self.remove(&key);
                self.stats.evictions += 1;
            }
        }
    }

    /// Get keys in order (for debugging/inspection)
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.order.iter()
    }

    fn evict_if_needed(&mut self, incoming_size: usize) {
        // Check entry limit
        while self.entries.len() >= self.max_entries {
            self.evict_one();
        }

        // Check memory limit
        if let Some(max_mem) = self.max_memory_bytes {
            while self.current_memory + incoming_size > max_mem && !self.entries.is_empty() {
                self.evict_one();
            }
        }
    }

    fn evict_one(&mut self) {
        let key_to_evict = match self.policy {
            EvictionPolicy::Lru | EvictionPolicy::Fifo | EvictionPolicy::Ttl => {
                // Evict from front (oldest in order)
                self.order.pop_front()
            }
            EvictionPolicy::Lfu => {
                // Find entry with lowest access count
                self.entries
                    .iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::Random => {
                // Simple random using current time as seed
                let idx = (Instant::now().elapsed().as_nanos() as usize) % self.order.len().max(1);
                self.order.get(idx).cloned()
            }
        };

        if let Some(key) = key_to_evict {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_memory = self.current_memory.saturating_sub(entry.size_bytes);
                self.order.retain(|k| k != &key);
                self.stats.evictions += 1;
                self.stats.entries = self.entries.len();
                self.stats.memory_bytes = self.current_memory;
            }
        }
    }
}

/// A bounded vector that maintains a maximum size
pub struct BoundedVec<T> {
    items: VecDeque<T>,
    max_size: usize,
    eviction_count: usize,
}

impl<T> BoundedVec<T> {
    /// Create a new bounded vector
    pub fn new(max_size: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(max_size.min(1000)),
            max_size,
            eviction_count: 0,
        }
    }

    /// Push an item (evicts oldest if at capacity)
    pub fn push(&mut self, item: T) {
        while self.items.len() >= self.max_size {
            self.items.pop_front();
            self.eviction_count += 1;
        }
        self.items.push_back(item);
    }

    /// Push to front
    pub fn push_front(&mut self, item: T) {
        while self.items.len() >= self.max_size {
            self.items.pop_back();
            self.eviction_count += 1;
        }
        self.items.push_front(item);
    }

    /// Get an item by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    /// Get mutable reference
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get eviction count
    pub fn eviction_count(&self) -> usize {
        self.eviction_count
    }

    /// Iterate over items
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Truncate to a smaller size
    pub fn truncate(&mut self, len: usize) {
        while self.items.len() > len {
            self.items.pop_back();
        }
    }

    /// Get last n items
    pub fn last_n(&self, n: usize) -> impl Iterator<Item = &T> {
        let skip = self.items.len().saturating_sub(n);
        self.items.iter().skip(skip)
    }
}

impl<T: Clone> BoundedVec<T> {
    /// Convert to a Vec
    pub fn to_vec(&self) -> Vec<T> {
        self.items.iter().cloned().collect()
    }
}

/// Memory usage tracker for multiple components
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    components: HashMap<String, ComponentMemory>,
    total_limit: Option<usize>,
    warning_threshold: f64,
}

/// Memory info for a single component
#[derive(Debug, Clone, Default)]
pub struct ComponentMemory {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub limit_bytes: Option<usize>,
    pub last_updated: Option<Instant>,
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPressure {
    /// Normal memory usage
    Normal,
    /// Warning threshold exceeded
    Warning,
    /// Critical threshold exceeded
    Critical,
}

impl MemoryTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a total memory limit
    pub fn with_limit(limit_bytes: usize) -> Self {
        Self {
            components: HashMap::new(),
            total_limit: Some(limit_bytes),
            warning_threshold: 0.8,
        }
    }

    /// Set warning threshold (0.0 - 1.0)
    pub fn set_warning_threshold(&mut self, threshold: f64) {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Register a component with optional limit
    pub fn register(&mut self, name: impl Into<String>, limit_bytes: Option<usize>) {
        self.components.insert(
            name.into(),
            ComponentMemory {
                current_bytes: 0,
                peak_bytes: 0,
                limit_bytes,
                last_updated: None,
            },
        );
    }

    /// Update memory usage for a component
    pub fn update(&mut self, name: &str, bytes: usize) {
        if let Some(comp) = self.components.get_mut(name) {
            comp.current_bytes = bytes;
            comp.peak_bytes = comp.peak_bytes.max(bytes);
            comp.last_updated = Some(Instant::now());
        }
    }

    /// Add memory to a component
    pub fn add(&mut self, name: &str, bytes: usize) {
        if let Some(comp) = self.components.get_mut(name) {
            comp.current_bytes = comp.current_bytes.saturating_add(bytes);
            comp.peak_bytes = comp.peak_bytes.max(comp.current_bytes);
            comp.last_updated = Some(Instant::now());
        }
    }

    /// Remove memory from a component
    pub fn remove(&mut self, name: &str, bytes: usize) {
        if let Some(comp) = self.components.get_mut(name) {
            comp.current_bytes = comp.current_bytes.saturating_sub(bytes);
            comp.last_updated = Some(Instant::now());
        }
    }

    /// Get total memory usage
    pub fn total_usage(&self) -> usize {
        self.components.values().map(|c| c.current_bytes).sum()
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.components.values().map(|c| c.peak_bytes).sum()
    }

    /// Get memory pressure level
    pub fn pressure(&self) -> MemoryPressure {
        if let Some(limit) = self.total_limit {
            let usage = self.total_usage();
            let ratio = usage as f64 / limit as f64;

            if ratio >= 1.0 {
                MemoryPressure::Critical
            } else if ratio >= self.warning_threshold {
                MemoryPressure::Warning
            } else {
                MemoryPressure::Normal
            }
        } else {
            MemoryPressure::Normal
        }
    }

    /// Check if a component is over its limit
    pub fn is_over_limit(&self, name: &str) -> bool {
        self.components
            .get(name)
            .and_then(|c| c.limit_bytes.map(|l| c.current_bytes > l))
            .unwrap_or(false)
    }

    /// Get component memory info
    pub fn get(&self, name: &str) -> Option<&ComponentMemory> {
        self.components.get(name)
    }

    /// Get all component names and their usage
    pub fn all_usage(&self) -> Vec<(String, usize)> {
        self.components
            .iter()
            .map(|(k, v)| (k.clone(), v.current_bytes))
            .collect()
    }

    /// Generate a memory report
    pub fn report(&self) -> MemoryReport {
        let total = self.total_usage();
        let peak = self.peak_usage();
        let pressure = self.pressure();

        MemoryReport {
            total_bytes: total,
            peak_bytes: peak,
            limit_bytes: self.total_limit,
            pressure,
            components: self
                .components
                .iter()
                .map(|(k, v)| (k.clone(), v.current_bytes, v.peak_bytes))
                .collect(),
        }
    }
}

/// Memory usage report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub total_bytes: usize,
    pub peak_bytes: usize,
    pub limit_bytes: Option<usize>,
    pub pressure: MemoryPressure,
    pub components: Vec<(String, usize, usize)>, // (name, current, peak)
}

impl MemoryReport {
    /// Format as human-readable string
    pub fn to_string_human(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "Memory Usage: {} / {} (peak: {})\n",
            format_bytes(self.total_bytes),
            self.limit_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "unlimited".to_string()),
            format_bytes(self.peak_bytes)
        ));

        out.push_str(&format!("Pressure: {:?}\n", self.pressure));
        out.push_str("Components:\n");

        for (name, current, peak) in &self.components {
            out.push_str(&format!(
                "  {}: {} (peak: {})\n",
                name,
                format_bytes(*current),
                format_bytes(*peak)
            ));
        }

        out
    }
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Estimate memory size of common types
pub trait MemoryEstimate {
    fn estimate_memory(&self) -> usize;
}

impl MemoryEstimate for String {
    fn estimate_memory(&self) -> usize {
        std::mem::size_of::<String>() + self.capacity()
    }
}

impl<T> MemoryEstimate for Vec<T> {
    fn estimate_memory(&self) -> usize {
        std::mem::size_of::<Vec<T>>() + self.capacity() * std::mem::size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_cache_lru() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(3, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Access "a" to make it recently used
        cache.get(&"a".to_string());

        // Insert "d", should evict "b" (least recently used)
        cache.insert("d".to_string(), 4);

        assert!(cache.contains(&"a".to_string()));
        assert!(!cache.contains(&"b".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_bounded_cache_fifo() {
        let mut cache: BoundedCache<i32, i32> = BoundedCache::new(3, EvictionPolicy::Fifo);

        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);
        cache.insert(4, 40);

        assert!(!cache.contains(&1)); // First in, first out
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_bounded_cache_memory_limit() {
        let mut cache: BoundedCache<String, Vec<u8>> =
            BoundedCache::with_memory_limit(100, 256, EvictionPolicy::Lru);

        cache.insert_with_size("a".to_string(), vec![0; 100], 100);
        cache.insert_with_size("b".to_string(), vec![0; 100], 100);

        // Should evict "a" to make room
        cache.insert_with_size("c".to_string(), vec![0; 100], 100);

        assert!(!cache.contains(&"a".to_string()));
        assert!(cache.memory_usage() <= 256);
    }

    #[test]
    fn test_bounded_vec() {
        let mut vec: BoundedVec<i32> = BoundedVec::new(3);

        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&2));
        assert_eq!(vec.get(2), Some(&4));
        assert_eq!(vec.eviction_count(), 1);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::with_limit(1000);

        tracker.register("cache", Some(500));
        tracker.register("embeddings", Some(500));

        tracker.update("cache", 300);
        tracker.update("embeddings", 200);

        assert_eq!(tracker.total_usage(), 500);
        assert_eq!(tracker.pressure(), MemoryPressure::Normal);

        tracker.update("cache", 500);
        tracker.update("embeddings", 400);

        assert_eq!(tracker.pressure(), MemoryPressure::Warning);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(10, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.get(&"a".to_string()); // hit
        cache.get(&"b".to_string()); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_cache_remove_clear_peek() {
        let mut cache: BoundedCache<String, i32> = BoundedCache::new(10, EvictionPolicy::Lru);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Peek should return value without modifying access metadata
        assert_eq!(cache.peek(&"b".to_string()), Some(&2));

        // Remove a single entry
        let removed = cache.remove(&"a".to_string());
        assert_eq!(removed, Some(1));
        assert!(!cache.contains(&"a".to_string()));
        assert_eq!(cache.len(), 2);

        // Remove non-existent key returns None
        assert_eq!(cache.remove(&"z".to_string()), None);

        // Clear all entries
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_bounded_vec_push_front_truncate_last_n() {
        let mut vec: BoundedVec<i32> = BoundedVec::new(5);

        // push_front keeps items at front, evicts from back when full
        vec.push_front(10);
        vec.push_front(20);
        vec.push_front(30);
        assert_eq!(vec.get(0), Some(&30));
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(2), Some(&10));

        // Push more items using push (back)
        vec.push(40);
        vec.push(50);
        assert_eq!(vec.len(), 5);

        // last_n returns the last n items
        let last_two: Vec<&i32> = vec.last_n(2).collect();
        assert_eq!(last_two, vec![&40, &50]);

        // last_n with n > len returns all items
        let all: Vec<&i32> = vec.last_n(100).collect();
        assert_eq!(all.len(), 5);

        // truncate reduces size from the back
        vec.truncate(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.to_vec(), vec![30, 20, 10]);

        // clear empties everything
        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn test_memory_tracker_add_remove_over_limit_critical() {
        let mut tracker = MemoryTracker::with_limit(1000);
        tracker.register("cache", Some(500));
        tracker.register("buffers", None);

        // add increments memory
        tracker.add("cache", 200);
        tracker.add("cache", 150);
        assert_eq!(tracker.get("cache").unwrap().current_bytes, 350);
        assert!(!tracker.is_over_limit("cache"));

        // Going over component limit
        tracker.add("cache", 200); // 550 > 500
        assert!(tracker.is_over_limit("cache"));

        // remove decrements memory (saturating)
        tracker.remove("cache", 100);
        assert_eq!(tracker.get("cache").unwrap().current_bytes, 450);
        assert!(!tracker.is_over_limit("cache"));

        // Peak should record the maximum
        assert_eq!(tracker.get("cache").unwrap().peak_bytes, 550);

        // Component without limit is never over limit
        tracker.add("buffers", 9999);
        assert!(!tracker.is_over_limit("buffers"));

        // Non-existent component is never over limit
        assert!(!tracker.is_over_limit("nonexistent"));

        // Critical pressure when over total limit
        tracker.update("cache", 600);
        tracker.update("buffers", 500);
        assert_eq!(tracker.pressure(), MemoryPressure::Critical);

        // Verify report
        let report = tracker.report();
        assert_eq!(report.total_bytes, 1100);
        assert_eq!(report.pressure, MemoryPressure::Critical);
        assert_eq!(report.limit_bytes, Some(1000));
    }

    #[test]
    fn test_format_bytes_and_memory_estimate() {
        // format_bytes for various magnitudes
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");

        // MemoryEstimate for String
        let s = String::from("hello");
        let est = s.estimate_memory();
        assert!(est >= std::mem::size_of::<String>() + 5);

        // MemoryEstimate for Vec
        let v: Vec<u64> = vec![1, 2, 3, 4];
        let est = v.estimate_memory();
        assert!(est >= std::mem::size_of::<Vec<u64>>() + 4 * std::mem::size_of::<u64>());

        // CacheStats::hit_rate with zero lookups returns 0.0
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }
}
