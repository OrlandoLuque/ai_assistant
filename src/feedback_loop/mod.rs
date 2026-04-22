//! Feedback Loop — routes trajectory outcomes to learning subsystems.
//!
//! A `FeedbackDispatcher` receives `TrajectoryRecord`s from the Butler at
//! end-of-run, evaluates privacy + freeze + minimum-sources gates, and fans
//! the survivor out to registered `FeedbackSink`s (memory, routing bandit,
//! fragment bandit, skill_forge, dataset writer). Every outcome (received,
//! dispatched, dropped, retracted) lands in a hash-chained `DispatchLedger`
//! so auditors can reconstruct the history.
//!
//! # Freeze
//!
//! When `LearningFreezeConfig::freeze_feedback_loop` is true, the dispatcher
//! still records trajectories in the ledger (as `Dropped { reason: "frozen" }`)
//! but never forwards to sinks. Use `FeedbackDispatcher::set_frozen(true)`.
//!
//! # Opt-in feature
//!
//! Behind `feedback-loop`. Not in `full`. Deps: `distillation`,
//! `advanced-memory`, `blake3`, `ed25519-dalek`.

pub mod dataset;
pub mod dispatcher;
pub mod ledger;
pub mod queue;
pub mod sinks;
pub mod trajectory;

pub use dataset::DatasetWriter;
pub use dispatcher::{
    FeedbackDispatcher, FeedbackDispatcherConfig, SubmitResult, DEFAULT_MINIMUM_SOURCES,
};
pub use ledger::{
    DispatchEvent, DispatchEventKind, DispatchLedger, DispatchLedgerError, DispatchSigner,
    NoopDispatchSigner, RetractionLedger,
};
pub use queue::{FeedbackQueue, OverflowAction, QueueError};
pub use sinks::{CollectorSink, FailingSink, FeedbackSink, SinkError};
pub use trajectory::{Outcome, PrivacyTier, RewardComponents, TrajectoryId, TrajectoryRecord};
