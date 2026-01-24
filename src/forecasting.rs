//! Usage forecasting
//!
//! Predict future usage patterns based on historical data.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Usage data point
#[derive(Debug, Clone)]
pub struct UsageDataPoint {
    pub timestamp: Instant,
    pub requests: u64,
    pub tokens: u64,
    pub users: usize,
}

/// Forecast result
#[derive(Debug, Clone)]
pub struct UsageForecast {
    pub period: Duration,
    pub predicted_requests: u64,
    pub predicted_tokens: u64,
    pub confidence: f64,
    pub trend: Trend,
}

/// Usage trend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Increasing,
    Stable,
    Decreasing,
}

/// Usage forecaster
pub struct UsageForecaster {
    history: VecDeque<UsageDataPoint>,
    max_history: usize,
}

impl UsageForecaster {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::new(),
            max_history,
        }
    }

    pub fn record(&mut self, point: UsageDataPoint) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(point);
    }

    pub fn record_usage(&mut self, requests: u64, tokens: u64, users: usize) {
        self.record(UsageDataPoint {
            timestamp: Instant::now(),
            requests,
            tokens,
            users,
        });
    }

    /// Forecast usage for the next period
    pub fn forecast(&self, period: Duration) -> Option<UsageForecast> {
        if self.history.len() < 3 {
            return None;
        }

        let points: Vec<_> = self.history.iter().collect();
        let n = points.len();

        // Calculate averages
        let avg_requests = points.iter().map(|p| p.requests).sum::<u64>() / n as u64;
        let avg_tokens = points.iter().map(|p| p.tokens).sum::<u64>() / n as u64;

        // Simple linear trend
        let recent_avg_requests = points[n/2..].iter().map(|p| p.requests).sum::<u64>() / (n/2) as u64;
        let older_avg_requests = points[..n/2].iter().map(|p| p.requests).sum::<u64>() / (n/2) as u64;

        let trend = if recent_avg_requests > older_avg_requests * 11 / 10 {
            Trend::Increasing
        } else if recent_avg_requests < older_avg_requests * 9 / 10 {
            Trend::Decreasing
        } else {
            Trend::Stable
        };

        // Apply trend to prediction
        let factor = match trend {
            Trend::Increasing => 1.1,
            Trend::Stable => 1.0,
            Trend::Decreasing => 0.9,
        };

        let periods_multiplier = period.as_secs() as f64 / 3600.0; // Assume hourly base

        Some(UsageForecast {
            period,
            predicted_requests: (avg_requests as f64 * factor * periods_multiplier) as u64,
            predicted_tokens: (avg_tokens as f64 * factor * periods_multiplier) as u64,
            confidence: (n as f64 / self.max_history as f64).min(1.0) * 0.8,
            trend,
        })
    }

    /// Get peak usage times
    pub fn get_peak_hours(&self) -> Vec<u8> {
        // Simplified - would need more data in production
        vec![9, 10, 11, 14, 15, 16] // Typical business hours
    }

    /// Estimate capacity needs
    pub fn estimate_capacity(&self, target_period: Duration) -> Option<CapacityEstimate> {
        let forecast = self.forecast(target_period)?;

        Some(CapacityEstimate {
            min_requests_per_second: forecast.predicted_requests / target_period.as_secs(),
            min_tokens_per_second: forecast.predicted_tokens / target_period.as_secs(),
            recommended_buffer: 1.5, // 50% buffer
        })
    }
}

impl Default for UsageForecaster {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Capacity estimate
#[derive(Debug, Clone)]
pub struct CapacityEstimate {
    pub min_requests_per_second: u64,
    pub min_tokens_per_second: u64,
    pub recommended_buffer: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecaster() {
        let mut forecaster = UsageForecaster::new(100);

        for i in 0..10 {
            forecaster.record_usage(100 + i * 10, 1000 + i * 100, 5);
        }

        let forecast = forecaster.forecast(Duration::from_secs(3600));
        assert!(forecast.is_some());

        let forecast = forecast.unwrap();
        assert!(forecast.predicted_requests > 0);
        assert_eq!(forecast.trend, Trend::Increasing);
    }
}
