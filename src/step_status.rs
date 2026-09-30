//! The status of one step of work — one enum, previously three.
//!
//! # Why this module exists
//!
//! Three enums described the same concept, each missing something the other two
//! had, and no two of them agreed on a spelling:
//!
//! | where | states |
//! |---|---|
//! | `agent_graph::StepStatus` | `Running`, `Completed`, `Failed`, `Skipped` |
//! | `task_planning::StepStatus` | `Pending`, `InProgress`, `Done`, `Blocked`, `Skipped` |
//! | `agent::PlanStepStatus` | `Pending`, `InProgress`, `Completed`, `Failed(String)`, `Skipped` |
//!
//! Read down the columns rather than across, because that is where the defects
//! are:
//!
//! * **A plan step could not fail.** `task_planning` had no `Failed` at all, so a
//!   step whose work blew up had nowhere to go but `Skipped` or a lie.
//! * **A graph step could not wait.** `agent_graph` had no `Blocked`, so a step
//!   waiting on an unavailable resource had no state but `Failed` — which is the
//!   difference between "retry later" and "give up", thrown away.
//! * **Two of the three lost the reason.** Only `agent::PlanStepStatus` carried
//!   the error text; the other two recorded *that* it failed and not *why*.
//!
//! The third one is the interesting one to have found: a name-comparing gate
//! (`scripts/check_duplicate_types.py`) pairs the two called `StepStatus` and is
//! structurally blind to `PlanStepStatus`. Its own header says so — it compares
//! spelling, not meaning — and this is the first case that proved it.
//!
//! # Open question, deliberately not decided here
//!
//! `Blocked` carries no reason, because none of the three did and inventing one
//! now would be designing for a task that is not designed yet (the
//! interruption-cause taxonomy). When that work lands it will very likely want
//! "blocked *on what*", and turning `Blocked` into `Blocked { reason }` is a
//! breaking change. Cheap here — this crate is not published — but it should be a
//! decision, not a side effect.

use serde::{Deserialize, Serialize};

/// Execution status of a single step of work.
///
/// Used by plans ([`crate::task_planning`]), execution traces
/// ([`crate::agent_graph`]) and the agent's own plan steps.
///
/// The variants form a progression that is not enforced anywhere: a step may go
/// from any state to any other. Nothing here validates a transition, so callers
/// that need `Pending -> InProgress -> Completed` to hold must check it
/// themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepStatus {
    /// Not started.
    Pending,
    /// Currently executing.
    ///
    /// `serde(alias = "Running")` keeps data written by the old
    /// `agent_graph::StepStatus` readable: the two names meant the same thing.
    #[serde(alias = "Running")]
    InProgress,
    /// Cannot proceed, but has not failed — a dependency is unmet or a resource
    /// is unavailable. Distinct from [`StepStatus::Failed`] on purpose: this one
    /// is worth retrying and that one is not.
    Blocked,
    /// Finished successfully.
    ///
    /// `serde(alias = "Done")` keeps plans persisted by the old
    /// `task_planning::StepStatus` readable.
    #[serde(alias = "Done")]
    Completed,
    /// Finished unsuccessfully, with the reason.
    ///
    /// The reason is not optional. Two of the three enums this replaces recorded
    /// only *that* a step failed, which is exactly the information a caller
    /// deciding whether to retry does not have.
    Failed { error: String },
    /// Deliberately not run.
    Skipped,
}

impl StepStatus {
    /// A short, stable label. Not `Display`-derived, so callers can rely on it
    /// for keys and comparisons even if `Display` grows detail later.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::InProgress => "InProgress",
            Self::Blocked => "Blocked",
            Self::Completed => "Completed",
            Self::Failed { .. } => "Failed",
            Self::Skipped => "Skipped",
            // `#[non_exhaustive]` is for consumers, not for this impl: a new
            // variant added here must be handled, and the compiler must say so.
        }
    }

    /// `true` once the step will not run again on its own.
    ///
    /// [`StepStatus::Blocked`] is **not** terminal — that is the whole reason it
    /// exists as a separate state. A blocked step is waiting, and a caller that
    /// treats it as finished is the bug this distinction prevents.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed { .. } | Self::Skipped)
    }

    /// The error text, if this status carries one.
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Failed { error } => Some(error.as_str()),
            _ => None,
        }
    }

    /// Build a [`StepStatus::Failed`] from anything printable.
    pub fn failed(error: impl std::fmt::Display) -> Self {
        Self::Failed {
            error: error.to_string(),
        }
    }
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { error } => write!(f, "Failed: {}", error),
            other => write!(f, "{}", other.as_str()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_states_are_the_three_that_end() {
        assert!(StepStatus::Completed.is_terminal());
        assert!(StepStatus::failed("boom").is_terminal());
        assert!(StepStatus::Skipped.is_terminal());
        assert!(!StepStatus::Pending.is_terminal());
        assert!(!StepStatus::InProgress.is_terminal());
    }

    #[test]
    fn blocked_is_not_terminal() {
        // Its own test, not a line in the one above: "blocked counts as finished"
        // is the specific mistake that made `Failed` the only state a waiting
        // graph step could occupy, and a gate should fail on it by name.
        assert!(!StepStatus::Blocked.is_terminal());
    }

    #[test]
    fn failure_keeps_the_reason() {
        let s = StepStatus::failed("disk full");
        assert_eq!(s.error(), Some("disk full"));
        assert_eq!(s.to_string(), "Failed: disk full");
        // And the other variants carry none, rather than an empty string.
        assert_eq!(StepStatus::Completed.error(), None);
    }

    #[test]
    fn as_str_ignores_the_payload() {
        assert_eq!(StepStatus::failed("a").as_str(), "Failed");
        assert_eq!(StepStatus::failed("b").as_str(), "Failed");
    }

    #[test]
    fn old_spellings_still_deserialize() {
        // The point of the two serde aliases: a plan persisted by the old
        // `task_planning::StepStatus` said "Done", and a trace written by the old
        // `agent_graph::StepStatus` said "Running". Dropping those would have made
        // this refactor silently unable to read its own predecessor's files.
        let done: StepStatus =
            serde_json::from_str("\"Done\"").expect("old \"Done\" must still load");
        assert_eq!(done, StepStatus::Completed);
        let running: StepStatus =
            serde_json::from_str("\"Running\"").expect("old \"Running\" must still load");
        assert_eq!(running, StepStatus::InProgress);
    }

    #[test]
    fn current_spellings_round_trip() {
        // An alias is read-only: serialising must emit the NEW name, or the
        // aliases would quietly become the format instead of a migration path.
        for s in [
            StepStatus::Pending,
            StepStatus::InProgress,
            StepStatus::Blocked,
            StepStatus::Completed,
            StepStatus::Skipped,
            StepStatus::failed("x"),
        ] {
            let json = serde_json::to_string(&s).expect("serialisable");
            let back: StepStatus = serde_json::from_str(&json).expect("round-trips");
            assert_eq!(back, s, "round trip changed {:?} via {}", s, json);
        }
        assert_eq!(
            serde_json::to_string(&StepStatus::Completed).expect("serialisable"),
            "\"Completed\"",
            "must emit the new name, not the alias"
        );
    }
}
