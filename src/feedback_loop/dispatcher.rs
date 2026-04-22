//! `FeedbackDispatcher` — routes `TrajectoryRecord` to registered sinks,
//! honoring freeze, privacy tier, and minimum-source thresholds.
//!
//! The dispatcher is synchronous: `submit()` runs the full pipeline inline
//! (freeze check → privacy filter → minimum-sources check → ledger append →
//! fan out to sinks → ledger outcome events). Callers that need async queue
//! semantics should use `FeedbackQueue` in front of it; the queue owns the
//! producer/consumer decoupling so the dispatcher can stay simple.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::ledger::{DispatchEventKind, DispatchLedger, DispatchLedgerError, RetractionLedger};
use super::sinks::FeedbackSink;
use super::trajectory::{PrivacyTier, TrajectoryId, TrajectoryRecord};

/// Outcome of a single submit call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SubmitResult {
    /// Record was dispatched to `sinks` (names of sinks that accepted it).
    Dispatched {
        trajectory: TrajectoryId,
        sinks: Vec<String>,
        failed: Vec<String>,
    },
    /// Record was dropped — reason is one of: frozen, privacy_drop,
    /// insufficient_sources, no_sinks.
    Dropped {
        trajectory: TrajectoryId,
        reason: String,
    },
}

/// Minimum number of non-`None` reward fields required. Below this the
/// dispatcher drops the record (defense against reward hacking).
pub const DEFAULT_MINIMUM_SOURCES: usize = 2;

/// Runtime-tunable dispatcher config.
#[derive(Debug, Clone)]
pub struct FeedbackDispatcherConfig {
    pub minimum_sources: usize,
    /// When true, `PrivacyTier::Confidential` records are still recorded in
    /// the ledger (as Dropped) but never forwarded. Default: true.
    pub drop_confidential: bool,
}

impl Default for FeedbackDispatcherConfig {
    fn default() -> Self {
        Self {
            minimum_sources: DEFAULT_MINIMUM_SOURCES,
            drop_confidential: true,
        }
    }
}

/// Dispatcher. Clone-cheap via `Arc`.
#[derive(Clone)]
pub struct FeedbackDispatcher {
    config: FeedbackDispatcherConfig,
    ledger: DispatchLedger,
    retractions: RetractionLedger,
    sinks: Arc<Vec<Arc<dyn FeedbackSink>>>,
    frozen: Arc<std::sync::atomic::AtomicBool>,
    dropped_total: Arc<AtomicU64>,
    delivered_total: Arc<AtomicU64>,
}

impl FeedbackDispatcher {
    pub fn new(
        config: FeedbackDispatcherConfig,
        ledger: DispatchLedger,
        retractions: RetractionLedger,
        sinks: Vec<Arc<dyn FeedbackSink>>,
    ) -> Self {
        Self {
            config,
            ledger,
            retractions,
            sinks: Arc::new(sinks),
            frozen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dropped_total: Arc::new(AtomicU64::new(0)),
            delivered_total: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Runtime freeze switch. Mirrors `LearningFreezeConfig::freeze_feedback_loop`.
    pub fn set_frozen(&self, frozen: bool) -> Result<(), DispatchLedgerError> {
        let prev = self.frozen.swap(frozen, Ordering::SeqCst);
        if prev != frozen {
            self.ledger
                .append(DispatchEventKind::FreezeChanged { frozen })?;
        }
        Ok(())
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::SeqCst)
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_total.load(Ordering::Relaxed)
    }

    pub fn delivered_count(&self) -> u64 {
        self.delivered_total.load(Ordering::Relaxed)
    }

    pub fn ledger(&self) -> &DispatchLedger {
        &self.ledger
    }

    pub fn retraction_ledger(&self) -> &RetractionLedger {
        &self.retractions
    }

    /// Submit a trajectory. Returns the outcome synchronously.
    pub fn submit(&self, record: TrajectoryRecord) -> Result<SubmitResult, DispatchLedgerError> {
        // 1) ledger: always record receipt (even if we end up dropping — audit trail).
        self.ledger.append(DispatchEventKind::TrajectoryReceived {
            trajectory: record.id.clone(),
            principal: record.principal.clone(),
            privacy: record.privacy_tier,
            outcome: record.outcome,
        })?;

        // 2) freeze check.
        if self.is_frozen() {
            return self.drop(&record, "frozen");
        }

        // 3) privacy drop for confidential tier.
        if self.config.drop_confidential && record.privacy_tier == PrivacyTier::Confidential {
            return self.drop(&record, "privacy_drop");
        }

        // 4) minimum sources check (defense vs reward hacking).
        if record.reward.source_count() < self.config.minimum_sources {
            return self.drop(&record, "insufficient_sources");
        }

        // 5) no sinks registered — technically we accepted the record but
        //    there's nothing to dispatch to. Keep this as a distinct reason
        //    so observability can spot misconfig vs real drops.
        if self.sinks.is_empty() {
            return self.drop(&record, "no_sinks");
        }

        // 6) fan out.
        let mut accepted = Vec::new();
        let mut failed = Vec::new();
        for sink in self.sinks.iter() {
            match sink.deliver(&record) {
                Ok(()) => {
                    self.ledger.append(DispatchEventKind::SinkDispatched {
                        trajectory: record.id.clone(),
                        sink: sink.name().to_string(),
                    })?;
                    accepted.push(sink.name().to_string());
                }
                Err(e) => {
                    self.ledger.append(DispatchEventKind::SinkFailed {
                        trajectory: record.id.clone(),
                        sink: sink.name().to_string(),
                        reason: e.reason.clone(),
                    })?;
                    failed.push(sink.name().to_string());
                }
            }
        }

        self.delivered_total.fetch_add(1, Ordering::Relaxed);
        Ok(SubmitResult::Dispatched {
            trajectory: record.id,
            sinks: accepted,
            failed,
        })
    }

    /// Issue a retraction for a previously-dispatched trajectory. Fans out
    /// `sink.retract()` to every sink and records each propagation.
    pub fn retract(
        &self,
        id: &TrajectoryId,
        reason: impl Into<String>,
    ) -> Result<(), DispatchLedgerError> {
        let reason = reason.into();
        self.retractions.record_request(id, reason.clone())?;
        for sink in self.sinks.iter() {
            match sink.retract(id) {
                Ok(()) => {
                    self.retractions.record_propagated(id, sink.name())?;
                }
                Err(e) => {
                    self.ledger.append(DispatchEventKind::SinkFailed {
                        trajectory: id.clone(),
                        sink: sink.name().to_string(),
                        reason: format!("retract: {}", e.reason),
                    })?;
                }
            }
        }
        Ok(())
    }

    fn drop(
        &self,
        record: &TrajectoryRecord,
        reason: &str,
    ) -> Result<SubmitResult, DispatchLedgerError> {
        self.ledger.append(DispatchEventKind::TrajectoryDropped {
            trajectory: record.id.clone(),
            reason: reason.to_string(),
        })?;
        self.dropped_total.fetch_add(1, Ordering::Relaxed);
        Ok(SubmitResult::Dropped {
            trajectory: record.id.clone(),
            reason: reason.to_string(),
        })
    }
}

impl std::fmt::Debug for FeedbackDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedbackDispatcher")
            .field("frozen", &self.is_frozen())
            .field("sinks", &self.sinks.len())
            .field("dropped", &self.dropped_count())
            .field("delivered", &self.delivered_count())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::ledger::NoopDispatchSigner;
    use super::super::sinks::{CollectorSink, FailingSink};
    use super::super::trajectory::{Outcome, RewardComponents};
    use super::*;

    fn signer() -> Arc<super::super::ledger::NoopDispatchSigner> {
        Arc::new(NoopDispatchSigner::new("node-t"))
    }

    fn dispatcher_with_sinks(sinks: Vec<Arc<dyn FeedbackSink>>) -> FeedbackDispatcher {
        let s: Arc<dyn super::super::ledger::DispatchSigner> = signer();
        FeedbackDispatcher::new(
            FeedbackDispatcherConfig::default(),
            DispatchLedger::new(s.clone()),
            RetractionLedger::new(s),
            sinks,
        )
    }

    fn good_record() -> TrajectoryRecord {
        let mut r = TrajectoryRecord::new("alice");
        r.reward.success = Some(1.0);
        r.reward.faithfulness = Some(0.8);
        r.outcome = Outcome::Success;
        r.privacy_tier = PrivacyTier::Internal;
        r
    }

    #[test]
    fn dispatch_delivers_to_all_sinks() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let ds = Arc::new(CollectorSink::new("dataset"));
        let d = dispatcher_with_sinks(vec![mem.clone(), ds.clone()]);
        let res = d.submit(good_record()).unwrap();
        match res {
            SubmitResult::Dispatched { sinks, failed, .. } => {
                assert_eq!(sinks.len(), 2);
                assert!(failed.is_empty());
            }
            other => panic!("expected Dispatched, got {other:?}"),
        }
        assert_eq!(mem.delivered_count(), 1);
        assert_eq!(ds.delivered_count(), 1);
        assert_eq!(d.delivered_count(), 1);
    }

    #[test]
    fn dispatch_records_partial_failure_without_aborting() {
        let good = Arc::new(CollectorSink::new("good"));
        let bad = Arc::new(FailingSink::new("bad", "timeout"));
        let d = dispatcher_with_sinks(vec![good.clone(), bad]);
        let res = d.submit(good_record()).unwrap();
        match res {
            SubmitResult::Dispatched { sinks, failed, .. } => {
                assert_eq!(sinks, vec!["good".to_string()]);
                assert_eq!(failed, vec!["bad".to_string()]);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(good.delivered_count(), 1);
    }

    #[test]
    fn frozen_dispatcher_drops_records() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem.clone()]);
        d.set_frozen(true).unwrap();
        let res = d.submit(good_record()).unwrap();
        assert!(matches!(res, SubmitResult::Dropped { .. }));
        assert_eq!(mem.delivered_count(), 0);
        assert_eq!(d.dropped_count(), 1);
    }

    #[test]
    fn confidential_records_are_dropped_by_default() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem.clone()]);
        let mut r = good_record();
        r.privacy_tier = PrivacyTier::Confidential;
        let res = d.submit(r).unwrap();
        match res {
            SubmitResult::Dropped { reason, .. } => assert_eq!(reason, "privacy_drop"),
            other => panic!("{other:?}"),
        }
        assert_eq!(mem.delivered_count(), 0);
    }

    #[test]
    fn insufficient_sources_drops_record() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem.clone()]);
        let mut r = TrajectoryRecord::new("bob");
        r.reward = RewardComponents {
            success: Some(1.0),
            ..Default::default()
        };
        let res = d.submit(r).unwrap();
        match res {
            SubmitResult::Dropped { reason, .. } => {
                assert_eq!(reason, "insufficient_sources")
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(mem.delivered_count(), 0);
    }

    #[test]
    fn no_sinks_configured_drops_record() {
        let d = dispatcher_with_sinks(vec![]);
        let res = d.submit(good_record()).unwrap();
        match res {
            SubmitResult::Dropped { reason, .. } => assert_eq!(reason, "no_sinks"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn retract_propagates_to_all_sinks_and_records_ledger() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let ds = Arc::new(CollectorSink::new("dataset"));
        let d = dispatcher_with_sinks(vec![mem.clone(), ds.clone()]);
        let r = good_record();
        let id = r.id.clone();
        d.submit(r).unwrap();
        d.retract(&id, "gdpr").unwrap();
        assert_eq!(mem.retracted().len(), 1);
        assert_eq!(ds.retracted().len(), 1);
        assert_eq!(d.retraction_ledger().len(), 3); // 1 request + 2 propagations.
    }

    #[test]
    fn unfreeze_restores_delivery() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem.clone()]);
        d.set_frozen(true).unwrap();
        let _ = d.submit(good_record()).unwrap();
        d.set_frozen(false).unwrap();
        let res = d.submit(good_record()).unwrap();
        assert!(matches!(res, SubmitResult::Dispatched { .. }));
    }

    #[test]
    fn ledger_records_every_submit() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem]);
        let before = d.ledger().len();
        d.submit(good_record()).unwrap();
        let after = d.ledger().len();
        // 1 received + 1 sink_dispatched = 2 new events at minimum.
        assert!(after >= before + 2);
        d.ledger().verify_chain().unwrap();
    }

    #[test]
    fn idempotent_delivery_does_not_duplicate_in_sink() {
        let mem = Arc::new(CollectorSink::new("memory"));
        let d = dispatcher_with_sinks(vec![mem.clone()]);
        let r = good_record();
        d.submit(r.clone()).unwrap();
        d.submit(r).unwrap();
        assert_eq!(mem.delivered_count(), 1);
    }
}
