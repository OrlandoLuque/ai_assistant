//! JSONL audit ledger for self-correction runs.
//!
//! Each call to `append` writes exactly one JSON object per line with the
//! full run summary. The ledger is append-only; `read_all` scans the file
//! and returns every entry. The auditor binary `ai_corrections` (V98 GUI
//! companion: `ai_corrections_gui`) reads this format.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::{AttemptRecord, SelfCorrectionResult, StopReason};

/// Errors produced by the ledger.
#[derive(Debug)]
pub enum LedgerError {
    /// Underlying IO error.
    Io(std::io::Error),
    /// Serialization failure.
    Serde(serde_json::Error),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "ledger IO error: {}", e),
            Self::Serde(e) => write!(f, "ledger serialization error: {}", e),
        }
    }
}

impl std::error::Error for LedgerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serde(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for LedgerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for LedgerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

/// A single persisted ledger entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    /// ISO-8601 UTC timestamp when the run finished.
    pub timestamp: String,
    /// Task label (from `CorrectableTask::name`).
    pub task_name: String,
    /// Whether the engine stopped with a successful outcome.
    pub succeeded: bool,
    /// Serialized stop reason (variant name; `FatalIssue` includes message).
    pub stop_reason: String,
    /// Per-attempt history.
    pub attempts: Vec<AttemptRecord>,
    /// Aggregate tokens across all attempts.
    pub total_tokens: usize,
    /// Aggregate cost across all attempts.
    pub total_cost_usd: f64,
    /// Aggregate wall-clock time (ms).
    pub total_elapsed_ms: u64,
}

impl LedgerEntry {
    /// Build an entry from a live result.
    pub fn from_result<O>(result: &SelfCorrectionResult<O>) -> Self {
        Self {
            timestamp: now_utc_iso(),
            task_name: result.task_name.clone(),
            succeeded: result.succeeded,
            stop_reason: stop_reason_to_string(&result.stop_reason),
            attempts: result.attempts.clone(),
            total_tokens: result.total_tokens,
            total_cost_usd: result.total_cost_usd,
            total_elapsed_ms: result.total_elapsed_ms,
        }
    }
}

fn stop_reason_to_string(r: &StopReason) -> String {
    match r {
        StopReason::AllPassed => "AllPassed".to_string(),
        StopReason::MaxAttempts => "MaxAttempts".to_string(),
        StopReason::TokenBudgetExhausted => "TokenBudgetExhausted".to_string(),
        StopReason::CostBudgetExhausted => "CostBudgetExhausted".to_string(),
        StopReason::TimeBudgetExhausted => "TimeBudgetExhausted".to_string(),
        StopReason::NoImprovement => "NoImprovement".to_string(),
        StopReason::QualityRegression => "QualityRegression".to_string(),
        StopReason::RegenerationFailed => "RegenerationFailed".to_string(),
        StopReason::CalibratedAbstention => "CalibratedAbstention".to_string(),
        StopReason::FatalIssue(msg) => format!("FatalIssue: {}", msg),
    }
}

fn now_utc_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Simple seconds-since-epoch; downstream tooling can convert. Avoids
    // pulling in chrono just for a timestamp.
    format!("{secs}")
}

/// Append-only JSONL ledger.
pub struct CorrectionLedger {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl CorrectionLedger {
    /// Open (create if missing) a ledger at `path`. The parent directory must
    /// exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LedgerError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        // Touch the file so readers never error on missing path.
        OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            write_lock: Mutex::new(()),
        })
    }

    /// Path on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one result to the ledger. Serializes a `LedgerEntry` and
    /// writes one line.
    pub fn append<O>(&self, result: &SelfCorrectionResult<O>) -> Result<(), LedgerError> {
        let entry = LedgerEntry::from_result(result);
        let line = serde_json::to_string(&entry)?;
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        writeln!(f, "{}", line)?;
        Ok(())
    }

    /// Read every entry. Malformed lines are skipped silently; the return
    /// value is `(entries, skipped_count)`.
    pub fn read_all(&self) -> Result<(Vec<LedgerEntry>, usize), LedgerError> {
        let f = OpenOptions::new().read(true).open(&self.path)?;
        let reader = BufReader::new(f);
        let mut out = Vec::new();
        let mut skipped = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<LedgerEntry>(&line) {
                Ok(e) => out.push(e),
                Err(_) => skipped += 1,
            }
        }
        Ok((out, skipped))
    }

    /// Count entries without fully deserializing them.
    pub fn count(&self) -> Result<usize, LedgerError> {
        let f = OpenOptions::new().read(true).open(&self.path)?;
        let reader = BufReader::new(f);
        let mut n = 0;
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                n += 1;
            }
        }
        Ok(n)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn dummy_result() -> SelfCorrectionResult<String> {
        SelfCorrectionResult {
            final_output: Some("ok".into()),
            attempts: vec![AttemptRecord {
                attempt_num: 1,
                issues: Vec::new(),
                quality_score: 1.0,
                tokens_used: 100,
                cost_usd: 0.01,
                elapsed_ms: 50,
                feedback_given: None,
                succeeded: true,
            }],
            succeeded: true,
            stop_reason: StopReason::AllPassed,
            total_tokens: 100,
            total_cost_usd: 0.01,
            total_elapsed_ms: 50,
            task_name: "test".into(),
        }
    }

    #[test]
    fn test_ledger_append_and_read() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp); // we just want the path; open will re-create

        let ledger = CorrectionLedger::open(&path).unwrap();
        ledger.append(&dummy_result()).unwrap();
        ledger.append(&dummy_result()).unwrap();

        let (entries, skipped) = ledger.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(skipped, 0);
        assert_eq!(entries[0].task_name, "test");
        assert!(entries[0].succeeded);
    }

    #[test]
    fn test_ledger_count() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);
        let ledger = CorrectionLedger::open(&path).unwrap();
        assert_eq!(ledger.count().unwrap(), 0);
        ledger.append(&dummy_result()).unwrap();
        ledger.append(&dummy_result()).unwrap();
        ledger.append(&dummy_result()).unwrap();
        assert_eq!(ledger.count().unwrap(), 3);
    }

    #[test]
    fn test_ledger_skips_malformed_lines() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);
        let ledger = CorrectionLedger::open(&path).unwrap();
        ledger.append(&dummy_result()).unwrap();
        // Manually append garbage.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f, "this is not json").unwrap();
        drop(f);
        ledger.append(&dummy_result()).unwrap();

        let (entries, skipped) = ledger.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(skipped, 1);
    }

    #[test]
    fn test_stop_reason_serialization() {
        assert_eq!(stop_reason_to_string(&StopReason::AllPassed), "AllPassed");
        assert_eq!(
            stop_reason_to_string(&StopReason::FatalIssue("PII".into())),
            "FatalIssue: PII"
        );
    }
}
