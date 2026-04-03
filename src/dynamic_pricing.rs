// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Dynamic pricing engine for compute services.
//!
//! Adjusts the per-request price based on current load, idle time, and the
//! network median price so that providers remain competitive while covering
//! their costs.

use serde::{Deserialize, Serialize};

/// A dynamic pricer that blends load, idle-time, and market factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPricer {
    /// The base (default) price per request.
    pub base_price: f64,
    /// Absolute minimum price (floor).
    pub min_price: f64,
    /// Absolute maximum price (ceiling).
    pub max_price: f64,
    /// Current load factor in `[0.0, 1.0]`.
    current_load: f32,
    /// When the node became idle (`None` if currently busy).
    #[serde(skip)]
    idle_since: Option<std::time::Instant>,
    /// Number of requests in the last hour (rolling estimate).
    requests_last_hour: u32,
    /// The median price observed across the network.
    network_median_price: f64,
}

impl DynamicPricer {
    /// Create a new pricer with the given base price and floor/ceiling.
    pub fn new(base_price: f64, min: f64, max: f64) -> Self {
        Self {
            base_price,
            min_price: min,
            max_price: max,
            current_load: 0.0,
            idle_since: None,
            requests_last_hour: 0,
            network_median_price: base_price,
        }
    }

    /// Compute the current price considering load, idle, and market factors.
    ///
    /// Formula (conceptual):
    /// ```text
    /// load_factor   = 1.0 + current_load          (1.0 … 2.0)
    /// idle_discount = 1.0 - min(idle_minutes/60, 0.3)  (0.7 … 1.0)
    /// market_factor = 0.5 + 0.5 * (network_median / base)  (clamped 0.5 … 1.5)
    /// price = base * load_factor * idle_discount * market_factor
    /// clamped to [min_price, max_price]
    /// ```
    pub fn current_price(&self) -> f64 {
        // Load: higher load → higher price.
        let load_factor = 1.0 + self.current_load as f64;

        // Idle: the longer idle, the cheaper (up to 30 % discount).
        let idle_minutes = self
            .idle_since
            .map(|t| t.elapsed().as_secs_f64() / 60.0)
            .unwrap_or(0.0);
        let idle_discount = 1.0 - (idle_minutes / 60.0).min(0.3);

        // Market: pull price towards network median.
        let market_factor = if self.base_price > 0.0 {
            (0.5 + 0.5 * (self.network_median_price / self.base_price)).clamp(0.5, 1.5)
        } else {
            1.0
        };

        let price = self.base_price * load_factor * idle_discount * market_factor;
        price.clamp(self.min_price, self.max_price)
    }

    /// Record an incoming request (resets idle timer).
    pub fn record_request(&mut self) {
        self.requests_last_hour = self.requests_last_hour.saturating_add(1);
        self.idle_since = None;
    }

    /// Update the current load fraction `[0.0, 1.0]`.
    pub fn update_load(&mut self, load: f32) {
        self.current_load = load.clamp(0.0, 1.0);
        if load <= 0.0 && self.idle_since.is_none() {
            self.idle_since = Some(std::time::Instant::now());
        } else if load > 0.0 {
            self.idle_since = None;
        }
    }

    /// Inform the pricer of the current network median price.
    pub fn update_market_price(&mut self, median: f64) {
        self.network_median_price = median;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idle_discount() {
        let mut pricer = DynamicPricer::new(10.0, 1.0, 100.0);
        // Simulate being idle for a while.
        pricer.idle_since = Some(std::time::Instant::now() - std::time::Duration::from_secs(1800));
        pricer.update_load(0.0);
        let price = pricer.current_price();
        // Should be cheaper than base due to idle discount.
        assert!(price < 10.0, "price = {}", price);
        assert!(price >= 1.0);
    }

    #[test]
    fn test_surge_pricing() {
        let mut pricer = DynamicPricer::new(10.0, 1.0, 100.0);
        pricer.update_load(0.9);
        let price = pricer.current_price();
        // High load → price above base.
        assert!(price > 10.0, "price = {}", price);
        assert!(price <= 100.0);
    }

    #[test]
    fn test_market_factor() {
        let mut pricer = DynamicPricer::new(10.0, 1.0, 100.0);
        // Market price is double the base → should push price up.
        pricer.update_market_price(20.0);
        let high = pricer.current_price();
        // Market price is half the base → should pull price down.
        pricer.update_market_price(5.0);
        let low = pricer.current_price();
        assert!(high > low, "high={} low={}", high, low);
    }

    #[test]
    fn test_clamp_min_max() {
        let mut pricer = DynamicPricer::new(10.0, 5.0, 15.0);
        // Very high load should not exceed max.
        pricer.update_load(1.0);
        pricer.update_market_price(100.0);
        assert!(pricer.current_price() <= 15.0);

        // Very low market + idle should not go below min.
        pricer.update_load(0.0);
        pricer.idle_since = Some(std::time::Instant::now() - std::time::Duration::from_secs(7200));
        pricer.update_market_price(0.1);
        assert!(pricer.current_price() >= 5.0);
    }
}
