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
    /// Optional hour-of-day (0-23) for peak-hour analysis
    pub hour: Option<u8>,
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
            hour: None,
        });
    }

    /// Record a usage data point with a known hour-of-day (0-23)
    pub fn record_usage_with_hour(&mut self, requests: u64, tokens: u64, users: usize, hour: u8) {
        self.record(UsageDataPoint {
            timestamp: Instant::now(),
            requests,
            tokens,
            users,
            hour: Some(hour),
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
        let recent_avg_requests =
            points[n / 2..].iter().map(|p| p.requests).sum::<u64>() / (n / 2) as u64;
        let older_avg_requests =
            points[..n / 2].iter().map(|p| p.requests).sum::<u64>() / (n / 2) as u64;

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

    /// Get peak usage hours by analyzing data points that have hour annotations.
    ///
    /// Groups data points by their `hour` field, computes total requests per hour,
    /// and returns hours whose total requests exceed the overall hourly average.
    /// Returns an empty vec if no data points have hour annotations.
    pub fn get_peak_hours(&self) -> Vec<u8> {
        let mut hour_totals: std::collections::HashMap<u8, u64> = std::collections::HashMap::new();

        for point in &self.history {
            if let Some(h) = point.hour {
                *hour_totals.entry(h).or_insert(0) += point.requests;
            }
        }

        if hour_totals.is_empty() {
            return Vec::new();
        }

        let total: u64 = hour_totals.values().sum();
        let mean = total as f64 / hour_totals.len() as f64;

        let mut peak: Vec<u8> = hour_totals
            .iter()
            .filter(|(_, &reqs)| reqs as f64 > mean)
            .map(|(&h, _)| h)
            .collect();
        peak.sort();
        peak
    }

    /// Compute a sliding-window moving average over the `requests` values of data points.
    ///
    /// Returns one averaged value for each valid window position. If `window` is 0 or
    /// exceeds the number of data points, returns an empty vec.
    pub fn moving_average(&self, window: usize) -> Vec<f64> {
        if window == 0 || window > self.history.len() {
            return Vec::new();
        }
        let values: Vec<f64> = self.history.iter().map(|p| p.requests as f64).collect();
        let mut result = Vec::with_capacity(values.len() - window + 1);
        let mut sum: f64 = values[..window].iter().sum();
        result.push(sum / window as f64);
        for i in window..values.len() {
            sum += values[i] - values[i - window];
            result.push(sum / window as f64);
        }
        result
    }

    /// Detect anomalies: data points whose `requests` value deviates from the mean
    /// by more than `threshold` standard deviations.
    ///
    /// Returns `(index, value)` pairs for each anomalous point.
    pub fn detect_anomalies(&self, threshold: f64) -> Vec<(usize, f64)> {
        if self.history.is_empty() {
            return Vec::new();
        }
        let values: Vec<f64> = self.history.iter().map(|p| p.requests as f64).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Vec::new();
        }

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| ((v - mean) / std_dev).abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect()
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

    #[test]
    fn test_peak_hours_from_data() {
        let mut forecaster = UsageForecaster::new(100);

        // Hours 9 and 14 get high traffic; hours 2 and 3 get low traffic
        for _ in 0..5 {
            forecaster.record_usage_with_hour(500, 1000, 10, 9);
            forecaster.record_usage_with_hour(600, 1200, 12, 14);
            forecaster.record_usage_with_hour(10, 100, 1, 2);
            forecaster.record_usage_with_hour(15, 150, 1, 3);
        }

        let peaks = forecaster.get_peak_hours();
        assert!(peaks.contains(&9), "Hour 9 should be a peak hour");
        assert!(peaks.contains(&14), "Hour 14 should be a peak hour");
        assert!(!peaks.contains(&2), "Hour 2 should not be a peak hour");
        assert!(!peaks.contains(&3), "Hour 3 should not be a peak hour");
    }

    #[test]
    fn test_moving_average() {
        let mut forecaster = UsageForecaster::new(100);

        // Add data points with known request values: 10, 20, 30, 40, 50
        for v in &[10u64, 20, 30, 40, 50] {
            forecaster.record_usage(*v, 0, 0);
        }

        let ma = forecaster.moving_average(3);
        // Windows: [10,20,30]=20.0, [20,30,40]=30.0, [30,40,50]=40.0
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 20.0).abs() < f64::EPSILON);
        assert!((ma[1] - 30.0).abs() < f64::EPSILON);
        assert!((ma[2] - 40.0).abs() < f64::EPSILON);

        // Edge: window larger than data returns empty
        assert!(forecaster.moving_average(10).is_empty());
        // Edge: window of 0 returns empty
        assert!(forecaster.moving_average(0).is_empty());
    }

    #[test]
    fn test_anomaly_detection() {
        let mut forecaster = UsageForecaster::new(100);

        // 9 normal values around 100, then 1 outlier at 1000
        for _ in 0..9 {
            forecaster.record_usage(100, 0, 0);
        }
        forecaster.record_usage(1000, 0, 0);

        let anomalies = forecaster.detect_anomalies(2.0);
        // The outlier at index 9 should be detected
        assert!(!anomalies.is_empty(), "Should detect at least one anomaly");
        assert!(
            anomalies.iter().any(|(idx, _)| *idx == 9),
            "Index 9 (the 1000-value outlier) should be flagged"
        );
        // Normal values should not be flagged
        for (idx, _) in &anomalies {
            assert_eq!(*idx, 9, "Only the outlier should be flagged");
        }
    }
}
