//! Memory pinning
//!
//! Pin important messages and facts to prevent eviction.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Pin type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PinType {
    /// User explicitly pinned
    User,
    /// System pinned (e.g., important context)
    System,
    /// Pinned due to high importance
    Importance,
    /// Pinned due to reference by other messages
    Reference,
    /// Temporary pin (will expire)
    Temporary,
}

/// Pinned item
#[derive(Debug, Clone)]
pub struct PinnedItem {
    pub id: String,
    pub pin_type: PinType,
    pub reason: Option<String>,
    pub pinned_at: Instant,
    pub expires_at: Option<Instant>,
    pub priority: u8,
}

impl PinnedItem {
    pub fn new(id: &str, pin_type: PinType) -> Self {
        Self {
            id: id.to_string(),
            pin_type,
            reason: None,
            pinned_at: Instant::now(),
            expires_at: None,
            priority: 50,
        }
    }

    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.to_string());
        self
    }

    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(Instant::now() + duration);
        self.pin_type = PinType::Temporary;
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            Instant::now() > expires
        } else {
            false
        }
    }
}

/// Memory pin manager
pub struct PinManager {
    pins: HashMap<String, PinnedItem>,
    references: HashMap<String, HashSet<String>>, // id -> referenced by
    max_pins: usize,
    auto_cleanup: bool,
}

impl PinManager {
    pub fn new() -> Self {
        Self {
            pins: HashMap::new(),
            references: HashMap::new(),
            max_pins: 100,
            auto_cleanup: true,
        }
    }

    pub fn with_max_pins(mut self, max: usize) -> Self {
        self.max_pins = max;
        self
    }

    /// Pin an item
    pub fn pin(&mut self, item: PinnedItem) -> bool {
        if self.auto_cleanup {
            self.cleanup_expired();
        }

        // Check if we have room
        if self.pins.len() >= self.max_pins && !self.pins.contains_key(&item.id) {
            // Try to remove lowest priority non-user pin
            self.evict_lowest_priority();
        }

        if self.pins.len() >= self.max_pins {
            return false;
        }

        self.pins.insert(item.id.clone(), item);
        true
    }

    /// Pin by ID with type
    pub fn pin_id(&mut self, id: &str, pin_type: PinType) -> bool {
        self.pin(PinnedItem::new(id, pin_type))
    }

    /// Unpin an item
    pub fn unpin(&mut self, id: &str) -> bool {
        self.pins.remove(id).is_some()
    }

    /// Check if an item is pinned
    pub fn is_pinned(&self, id: &str) -> bool {
        if let Some(pin) = self.pins.get(id) {
            !pin.is_expired()
        } else {
            false
        }
    }

    /// Get pin info
    pub fn get_pin(&self, id: &str) -> Option<&PinnedItem> {
        self.pins.get(id).filter(|p| !p.is_expired())
    }

    /// Add a reference (item A references item B)
    pub fn add_reference(&mut self, from_id: &str, to_id: &str) {
        self.references
            .entry(to_id.to_string())
            .or_default()
            .insert(from_id.to_string());

        // Auto-pin referenced items
        if !self.is_pinned(to_id) {
            self.pin(
                PinnedItem::new(to_id, PinType::Reference)
                    .with_reason(&format!("Referenced by {}", from_id)),
            );
        }
    }

    /// Remove a reference
    pub fn remove_reference(&mut self, from_id: &str, to_id: &str) {
        if let Some(refs) = self.references.get_mut(to_id) {
            refs.remove(from_id);

            // If no more references and pin is Reference type, unpin
            if refs.is_empty() {
                if let Some(pin) = self.pins.get(to_id) {
                    if pin.pin_type == PinType::Reference {
                        self.pins.remove(to_id);
                    }
                }
                self.references.remove(to_id);
            }
        }
    }

    /// Get all pinned IDs
    pub fn pinned_ids(&self) -> Vec<&str> {
        self.pins
            .iter()
            .filter(|(_, p)| !p.is_expired())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get pins by type
    pub fn pins_by_type(&self, pin_type: PinType) -> Vec<&PinnedItem> {
        self.pins
            .values()
            .filter(|p| p.pin_type == pin_type && !p.is_expired())
            .collect()
    }

    /// Cleanup expired pins
    pub fn cleanup_expired(&mut self) {
        let expired: Vec<_> = self
            .pins
            .iter()
            .filter(|(_, p)| p.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            self.pins.remove(&id);
        }
    }

    fn evict_lowest_priority(&mut self) {
        // Find lowest priority non-user pin
        let to_remove = self
            .pins
            .iter()
            .filter(|(_, p)| p.pin_type != PinType::User)
            .min_by_key(|(_, p)| p.priority)
            .map(|(id, _)| id.clone());

        if let Some(id) = to_remove {
            self.pins.remove(&id);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PinStats {
        self.cleanup_expired_stats()
    }

    fn cleanup_expired_stats(&self) -> PinStats {
        let active_pins: Vec<_> = self.pins.values().filter(|p| !p.is_expired()).collect();

        let mut by_type: HashMap<PinType, usize> = HashMap::new();
        for pin in &active_pins {
            *by_type.entry(pin.pin_type).or_insert(0) += 1;
        }

        PinStats {
            total_pins: active_pins.len(),
            user_pins: by_type.get(&PinType::User).copied().unwrap_or(0),
            system_pins: by_type.get(&PinType::System).copied().unwrap_or(0),
            reference_pins: by_type.get(&PinType::Reference).copied().unwrap_or(0),
            temporary_pins: by_type.get(&PinType::Temporary).copied().unwrap_or(0),
            total_references: self.references.values().map(|r| r.len()).sum(),
        }
    }
}

impl Default for PinManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pin statistics
#[derive(Debug, Clone)]
pub struct PinStats {
    pub total_pins: usize,
    pub user_pins: usize,
    pub system_pins: usize,
    pub reference_pins: usize,
    pub temporary_pins: usize,
    pub total_references: usize,
}

/// Auto-pinner that automatically pins important items
pub struct AutoPinner {
    keywords: Vec<String>,
    importance_threshold: f64,
}

impl AutoPinner {
    pub fn new() -> Self {
        Self {
            keywords: vec![
                "important".to_string(),
                "remember".to_string(),
                "don't forget".to_string(),
                "key point".to_string(),
                "critical".to_string(),
            ],
            importance_threshold: 0.8,
        }
    }

    pub fn add_keyword(&mut self, keyword: &str) {
        self.keywords.push(keyword.to_lowercase());
    }

    pub fn set_importance_threshold(&mut self, threshold: f64) {
        self.importance_threshold = threshold;
    }

    /// Check if content should be auto-pinned
    pub fn should_pin(&self, content: &str, importance: f64) -> Option<PinType> {
        // Check importance threshold
        if importance >= self.importance_threshold {
            return Some(PinType::Importance);
        }

        // Check keywords
        let lower = content.to_lowercase();
        for keyword in &self.keywords {
            if lower.contains(keyword) {
                return Some(PinType::System);
            }
        }

        None
    }

    /// Suggest pin reason
    pub fn suggest_reason(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();

        for keyword in &self.keywords {
            if lower.contains(keyword) {
                return Some(format!("Contains keyword: {}", keyword));
            }
        }

        None
    }
}

impl Default for AutoPinner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pinning() {
        let mut manager = PinManager::new();

        assert!(manager.pin_id("msg1", PinType::User));
        assert!(manager.is_pinned("msg1"));
        assert!(!manager.is_pinned("msg2"));
    }

    #[test]
    fn test_unpin() {
        let mut manager = PinManager::new();

        manager.pin_id("msg1", PinType::User);
        assert!(manager.is_pinned("msg1"));

        manager.unpin("msg1");
        assert!(!manager.is_pinned("msg1"));
    }

    #[test]
    fn test_temporary_pin() {
        let mut manager = PinManager::new();

        let pin =
            PinnedItem::new("msg1", PinType::Temporary).with_expiry(Duration::from_millis(10));

        manager.pin(pin);
        assert!(manager.is_pinned("msg1"));

        std::thread::sleep(Duration::from_millis(20));
        assert!(!manager.is_pinned("msg1"));
    }

    #[test]
    fn test_references() {
        let mut manager = PinManager::new();

        manager.add_reference("msg2", "msg1");
        assert!(manager.is_pinned("msg1")); // Auto-pinned due to reference

        manager.remove_reference("msg2", "msg1");
        assert!(!manager.is_pinned("msg1")); // Unpinned when no more references
    }

    #[test]
    fn test_auto_pinner() {
        let pinner = AutoPinner::new();

        assert!(pinner.should_pin("This is important!", 0.5).is_some());
        assert!(pinner.should_pin("High importance", 0.9).is_some());
        assert!(pinner.should_pin("Normal message", 0.3).is_none());
    }

    #[test]
    fn test_pin_with_priority() {
        let mut manager = PinManager::new();
        let pin = PinnedItem::new("msg1", PinType::System)
            .with_reason("Critical context")
            .with_priority(99);
        manager.pin(pin);
        let p = manager.get_pin("msg1").unwrap();
        assert_eq!(p.priority, 99);
        assert_eq!(p.reason.as_deref(), Some("Critical context"));
        assert_eq!(p.pin_type, PinType::System);
    }

    #[test]
    fn test_pins_by_type() {
        let mut manager = PinManager::new();
        manager.pin_id("a", PinType::User);
        manager.pin_id("b", PinType::System);
        manager.pin_id("c", PinType::User);
        let user_pins = manager.pins_by_type(PinType::User);
        assert_eq!(user_pins.len(), 2);
        let system_pins = manager.pins_by_type(PinType::System);
        assert_eq!(system_pins.len(), 1);
    }

    #[test]
    fn test_max_pins() {
        let mut manager = PinManager::new().with_max_pins(2);
        assert!(manager.pin_id("a", PinType::User));
        assert!(manager.pin_id("b", PinType::System));
        // Third pin should evict lowest-priority non-user pin (System at priority 50)
        assert!(manager.pin_id("c", PinType::User));
        assert_eq!(manager.pinned_ids().len(), 2);
        // User pin "a" should still be there, System pin "b" should have been evicted
        assert!(manager.is_pinned("a"));
        assert!(manager.is_pinned("c"));
    }

    #[test]
    fn test_stats() {
        let mut manager = PinManager::new();
        manager.pin_id("a", PinType::User);
        manager.pin_id("b", PinType::System);
        manager.add_reference("c", "d"); // Creates a Reference pin for "d"
        let stats = manager.stats();
        assert_eq!(stats.total_pins, 3);
        assert_eq!(stats.user_pins, 1);
        assert_eq!(stats.system_pins, 1);
        assert_eq!(stats.reference_pins, 1);
        assert_eq!(stats.total_references, 1);
    }

    #[test]
    fn test_auto_pinner_custom_keyword() {
        let mut pinner = AutoPinner::new();
        pinner.add_keyword("urgent");
        assert!(pinner.should_pin("This is urgent!", 0.0).is_some());
        let reason = pinner.suggest_reason("This is urgent!");
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("urgent"));
    }
}
