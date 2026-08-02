//! Where work goes when the engine could not make it correct.
//!
//! The [`ledger`](super::ledger) records what *happened*; this records what is
//! *outstanding*. They answer different questions: "what did the engine do last
//! night" versus "what is still waiting for a human".
//!
//! # Why a terminal state instead of returning the output anyway
//!
//! A self-correction run that exhausts its budget has produced an artifact that
//! is known-not-verified. Handing it back like any other result invites the
//! caller to use it, and the single most damaging thing an agent can do is
//! present unfinished work as finished — it is worse than failing loudly,
//! because it removes the signal that anything is wrong.
//!
//! So the rule here is: **quarantine, never merge.** The artifact is preserved
//! exactly as produced, the evidence of every attempt is stored beside it, and
//! it stays in the queue until a human resolves it. Nothing reads a quarantined
//! artifact back into a pipeline.
//!
//! # Layout
//!
//! ```text
//! <dir>/pending/<id>.json      the evidence (a LedgerEntry)
//! <dir>/pending/<id>.artifact  the work itself, byte for byte
//! <dir>/resolved/…             the same pair, moved on resolve
//! ```
//!
//! Two files rather than one embedded blob, so the artifact stays readable with
//! an ordinary editor: a reviewer should never need our tooling to look at the
//! code they are judging.
//!
//! `ai_corrections` is the CLI auditor over this directory, with
//! `ai_corrections_gui` as its companion.

use std::fs;
use std::path::{Path, PathBuf};

use super::ledger::{LedgerEntry, LedgerError};
use super::SelfCorrectionResult;

/// One item awaiting review.
#[derive(Debug, Clone)]
pub struct QuarantineRecord {
    /// Identifier, and the file stem of both stored files.
    pub id: String,
    /// The evidence: attempts, stop reason, totals.
    pub evidence: LedgerEntry,
    /// Path to the artifact as produced.
    pub artifact_path: PathBuf,
}

impl QuarantineRecord {
    /// Read the artifact back. Kept separate from listing so an auditor can
    /// show a queue of hundreds without loading every file.
    pub fn read_artifact(&self) -> Result<String, LedgerError> {
        fs::read_to_string(&self.artifact_path).map_err(LedgerError::Io)
    }

    /// One-line summary for a listing.
    pub fn summary(&self) -> String {
        format!(
            "{}  {}  {} attempt(s)  stopped: {}",
            self.id,
            self.evidence.task_name,
            self.evidence.attempts.len(),
            self.evidence.stop_reason
        )
    }
}

/// A directory of work awaiting human review.
pub struct Quarantine {
    root: PathBuf,
}

impl Quarantine {
    /// Open (creating if needed) a quarantine directory.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, LedgerError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("pending")).map_err(LedgerError::Io)?;
        fs::create_dir_all(root.join("resolved")).map_err(LedgerError::Io)?;
        Ok(Self { root })
    }

    /// The directory being managed.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn pending_dir(&self) -> PathBuf {
        self.root.join("pending")
    }

    fn resolved_dir(&self) -> PathBuf {
        self.root.join("resolved")
    }

    /// Store an unverified artifact together with the evidence.
    ///
    /// Returns the id under which it was filed. Storing a run that *succeeded*
    /// is refused: quarantine is for outstanding work, and letting verified
    /// results in would turn the review queue into a log nobody reads.
    pub fn store<O>(
        &self,
        result: &SelfCorrectionResult<O>,
        artifact: &str,
    ) -> Result<String, LedgerError> {
        if result.succeeded {
            return Err(LedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "refusing to quarantine a successful run — the review queue is \
                 for work that still needs a human",
            )));
        }
        let evidence = LedgerEntry::from_result(result);
        let id = self.next_id(&evidence);
        let dir = self.pending_dir();
        let json = serde_json::to_string_pretty(&evidence).map_err(LedgerError::Serde)?;
        fs::write(dir.join(format!("{id}.json")), json).map_err(LedgerError::Io)?;
        fs::write(dir.join(format!("{id}.artifact")), artifact).map_err(LedgerError::Io)?;
        Ok(id)
    }

    /// Everything still awaiting review, oldest first.
    pub fn pending(&self) -> Result<Vec<QuarantineRecord>, LedgerError> {
        self.list_in(&self.pending_dir())
    }

    /// Everything already dealt with.
    pub fn resolved(&self) -> Result<Vec<QuarantineRecord>, LedgerError> {
        self.list_in(&self.resolved_dir())
    }

    /// Move an item out of the pending queue.
    ///
    /// Deliberately a move rather than a delete: the evidence of what the agent
    /// could not do is exactly the material worth keeping.
    pub fn resolve(&self, id: &str) -> Result<(), LedgerError> {
        let from = self.pending_dir();
        let to = self.resolved_dir();
        let json = from.join(format!("{id}.json"));
        if !json.exists() {
            return Err(LedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no pending item with id '{id}'"),
            )));
        }
        fs::rename(&json, to.join(format!("{id}.json"))).map_err(LedgerError::Io)?;
        let artifact = from.join(format!("{id}.artifact"));
        if artifact.exists() {
            fs::rename(&artifact, to.join(format!("{id}.artifact"))).map_err(LedgerError::Io)?;
        }
        Ok(())
    }

    fn list_in(&self, dir: &Path) -> Result<Vec<QuarantineRecord>, LedgerError> {
        let mut out = Vec::new();
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(LedgerError::Io(e)),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // A malformed entry must not hide the rest of the queue: a reviewer
            // needs to see the other items even if one file got truncated.
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(evidence) = serde_json::from_str::<LedgerEntry>(&text) else {
                continue;
            };
            out.push(QuarantineRecord {
                id: id.to_string(),
                evidence,
                artifact_path: dir.join(format!("{id}.artifact")),
            });
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// `<timestamp>-<task>-<n>`: sorts chronologically, says what it was, and
    /// the counter keeps two runs of the same task in the same second apart.
    fn next_id(&self, evidence: &LedgerEntry) -> String {
        let task: String = evidence
            .task_name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let base = format!("{}-{}", evidence.timestamp, task);
        let dir = self.pending_dir();
        for n in 0..10_000 {
            let candidate = format!("{base}-{n}");
            if !dir.join(format!("{candidate}.json")).exists() {
                return candidate;
            }
        }
        format!("{base}-overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_correction::{AttemptRecord, StopReason};

    fn result(succeeded: bool) -> SelfCorrectionResult<String> {
        SelfCorrectionResult {
            final_output: Some("fn main() {}".to_string()),
            attempts: vec![AttemptRecord {
                attempt_num: 1,
                issues: vec!["compile failed:\nE0308".into()],
                quality_score: 0.5,
                tokens_used: 10,
                cost_usd: 0.0,
                elapsed_ms: 5,
                feedback_given: None,
                succeeded: false,
            }],
            succeeded,
            stop_reason: StopReason::MaxAttempts,
            total_tokens: 10,
            total_cost_usd: 0.0,
            total_elapsed_ms: 5,
            task_name: "code_compile".to_string(),
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ai_quarantine_test_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn stores_artifact_and_evidence_side_by_side() {
        let dir = tmpdir("store");
        let q = Quarantine::open(&dir).unwrap();
        let id = q.store(&result(false), "fn broken() { todo!() }").unwrap();

        let pending = q.pending().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, id);
        // The evidence survives, which is the point: a reviewer must see WHY.
        assert_eq!(pending[0].evidence.attempts.len(), 1);
        assert!(pending[0].evidence.attempts[0].issues[0].contains("E0308"));
        // And the artifact is stored verbatim, readable without our tooling.
        assert_eq!(
            pending[0].read_artifact().unwrap(),
            "fn broken() { todo!() }"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_quarantine_work_that_passed() {
        // Letting successes in would turn the review queue into a log, and a
        // queue nobody can drain is a queue nobody reads.
        let dir = tmpdir("refuse");
        let q = Quarantine::open(&dir).unwrap();
        assert!(q.store(&result(true), "fine").is_err());
        assert!(q.pending().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolving_moves_rather_than_deletes() {
        let dir = tmpdir("resolve");
        let q = Quarantine::open(&dir).unwrap();
        let id = q.store(&result(false), "artifact text").unwrap();

        q.resolve(&id).unwrap();
        assert!(q.pending().unwrap().is_empty(), "queue must drain");

        let resolved = q.resolved().unwrap();
        assert_eq!(resolved.len(), 1, "evidence must survive resolution");
        assert_eq!(resolved[0].read_artifact().unwrap(), "artifact text");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolving_an_unknown_id_is_an_error_not_a_silent_success() {
        let dir = tmpdir("unknown");
        let q = Quarantine::open(&dir).unwrap();
        assert!(q.resolve("nope").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_runs_of_the_same_task_do_not_collide() {
        let dir = tmpdir("collide");
        let q = Quarantine::open(&dir).unwrap();
        let a = q.store(&result(false), "first").unwrap();
        let b = q.store(&result(false), "second").unwrap();
        assert_ne!(
            a, b,
            "same task in the same second must still get two slots"
        );
        assert_eq!(q.pending().unwrap().len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_entry_does_not_hide_the_rest_of_the_queue() {
        let dir = tmpdir("corrupt");
        let q = Quarantine::open(&dir).unwrap();
        q.store(&result(false), "good").unwrap();
        fs::write(dir.join("pending").join("zzz-broken.json"), "{not json").unwrap();

        let pending = q.pending().unwrap();
        assert_eq!(pending.len(), 1, "the readable item must still be listed");
        let _ = fs::remove_dir_all(&dir);
    }
}
