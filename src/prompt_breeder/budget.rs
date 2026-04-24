//! Budget meter — accumulates spend from each LLM call and checks it
//! against the configured `BudgetLimit`. The first limit to trip aborts the
//! run via a `BudgetKind` event on the ledger.

use std::time::{Duration, Instant};

use super::config::BudgetLimit;
use super::ledger::BudgetKind;
use super::llm::{CostEstimator, TokenUsage};

/// Running totals for a single run.
#[derive(Debug)]
pub struct BudgetMeter {
    limit: BudgetLimit,
    calls: u64,
    tokens: u64,
    cost_usd: f64,
    started_at: Instant,
    cost_estimator: CostEstimator,
    fingerprint: String,
}

impl BudgetMeter {
    pub fn new(limit: BudgetLimit, fingerprint: String, estimator: CostEstimator) -> Self {
        Self {
            limit,
            calls: 0,
            tokens: 0,
            cost_usd: 0.0,
            started_at: Instant::now(),
            cost_estimator: estimator,
            fingerprint,
        }
    }

    pub fn record_call(&mut self, usage: TokenUsage) {
        self.calls += 1;
        self.tokens += usage.total();
        self.cost_usd += self.cost_estimator.estimate(&self.fingerprint, usage);
    }

    pub fn calls(&self) -> u64 {
        self.calls
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    pub fn cost_usd(&self) -> f64 {
        self.cost_usd
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Check every configured limit. Returns the first one that trips, if any.
    pub fn check(&self) -> Option<BudgetBreach> {
        check_limit(&self.limit, self)
    }
}

/// Which limit tripped and with what value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetBreach {
    pub kind: BudgetKind,
    pub value: f64,
}

fn check_limit(limit: &BudgetLimit, meter: &BudgetMeter) -> Option<BudgetBreach> {
    match limit {
        BudgetLimit::None => None,
        BudgetLimit::MaxLlmCalls(max) => {
            if meter.calls >= *max {
                Some(BudgetBreach {
                    kind: BudgetKind::LlmCalls,
                    value: meter.calls as f64,
                })
            } else {
                None
            }
        }
        BudgetLimit::MaxTokens(max) => {
            if meter.tokens >= *max {
                Some(BudgetBreach {
                    kind: BudgetKind::Tokens,
                    value: meter.tokens as f64,
                })
            } else {
                None
            }
        }
        BudgetLimit::MaxWallTime(max) => {
            if meter.elapsed() >= *max {
                Some(BudgetBreach {
                    kind: BudgetKind::WallTime,
                    value: meter.elapsed().as_secs_f64(),
                })
            } else {
                None
            }
        }
        BudgetLimit::MaxCostUsd(max) => {
            if meter.cost_usd >= *max {
                Some(BudgetBreach {
                    kind: BudgetKind::CostUsd,
                    value: meter.cost_usd,
                })
            } else {
                None
            }
        }
        BudgetLimit::Composite(list) => {
            for l in list {
                if let Some(b) = check_limit(l, meter) {
                    return Some(b);
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn mk(limit: BudgetLimit) -> BudgetMeter {
        BudgetMeter::new(limit, "test/mock".into(), CostEstimator::default())
    }

    fn usage(in_t: u64, out_t: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: in_t,
            output_tokens: out_t,
        }
    }

    #[test]
    fn none_never_trips() {
        let mut m = mk(BudgetLimit::None);
        for _ in 0..100 {
            m.record_call(usage(1000, 1000));
        }
        assert!(m.check().is_none());
    }

    #[test]
    fn max_calls_trips_exact() {
        let mut m = mk(BudgetLimit::MaxLlmCalls(3));
        m.record_call(usage(1, 1));
        m.record_call(usage(1, 1));
        assert!(m.check().is_none());
        m.record_call(usage(1, 1));
        let b = m.check().expect("breach");
        assert_eq!(b.kind, BudgetKind::LlmCalls);
    }

    #[test]
    fn max_tokens_trips() {
        let mut m = mk(BudgetLimit::MaxTokens(100));
        m.record_call(usage(60, 60));
        let b = m.check().expect("breach");
        assert_eq!(b.kind, BudgetKind::Tokens);
    }

    #[test]
    fn max_wall_time_trips() {
        let mut m = mk(BudgetLimit::MaxWallTime(Duration::from_millis(5)));
        sleep(Duration::from_millis(10));
        let b = m.check().expect("breach");
        assert_eq!(b.kind, BudgetKind::WallTime);
    }

    #[test]
    fn composite_first_trip_wins() {
        let mut m = mk(BudgetLimit::Composite(vec![
            BudgetLimit::MaxLlmCalls(5),
            BudgetLimit::MaxTokens(10),
        ]));
        // Reach tokens first.
        m.record_call(usage(10, 10));
        let b = m.check().expect("breach");
        assert_eq!(b.kind, BudgetKind::Tokens);
    }
}
