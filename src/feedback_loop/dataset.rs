//! `DatasetWriter` — JSONL sink for offline training pipelines (GEPA/MIPRO).
//!
//! Writes one `TrajectoryRecord` per line, newline-terminated. Rotation is
//! the caller's problem — this is the simplest thing that produces a file.
//! The writer skips `PrivacyTier::Confidential` records (matches the
//! dispatcher's confidential-drop policy).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use super::sinks::{FeedbackSink, SinkError};
use super::trajectory::{PrivacyTier, TrajectoryId, TrajectoryRecord};

pub struct DatasetWriter {
    name: String,
    path: PathBuf,
    inner: Mutex<DatasetInner>,
}

struct DatasetInner {
    writer: BufWriter<File>,
    written: u64,
    skipped_confidential: u64,
    retracted_ids: Vec<TrajectoryId>,
}

impl DatasetWriter {
    pub fn open(name: impl Into<String>, path: impl Into<PathBuf>) -> Result<Self, String> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
            }
        }
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok(Self {
            name: name.into(),
            path,
            inner: Mutex::new(DatasetInner {
                writer: BufWriter::new(f),
                written: 0,
                skipped_confidential: 0,
                retracted_ids: Vec::new(),
            }),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn written_count(&self) -> u64 {
        self.inner.lock().map(|g| g.written).unwrap_or(0)
    }

    pub fn skipped_confidential_count(&self) -> u64 {
        self.inner
            .lock()
            .map(|g| g.skipped_confidential)
            .unwrap_or(0)
    }
}

impl FeedbackSink for DatasetWriter {
    fn name(&self) -> &str {
        &self.name
    }

    fn deliver(&self, record: &TrajectoryRecord) -> Result<(), SinkError> {
        let mut g = self.inner.lock().map_err(|_| SinkError {
            sink: self.name.clone(),
            reason: "poisoned".into(),
        })?;
        if record.privacy_tier == PrivacyTier::Confidential {
            g.skipped_confidential += 1;
            return Ok(());
        }
        let line = serde_json::to_string(record).map_err(|e| SinkError {
            sink: self.name.clone(),
            reason: format!("serialize: {e}"),
        })?;
        g.writer
            .write_all(line.as_bytes())
            .and_then(|()| g.writer.write_all(b"\n"))
            .map_err(|e| SinkError {
                sink: self.name.clone(),
                reason: format!("write: {e}"),
            })?;
        g.writer.flush().map_err(|e| SinkError {
            sink: self.name.clone(),
            reason: format!("flush: {e}"),
        })?;
        g.written += 1;
        Ok(())
    }

    fn retract(&self, id: &TrajectoryId) -> Result<(), SinkError> {
        // Append-only JSONL — we record a tombstone entry and let the
        // downstream consumer ignore retracted ids.
        let mut g = self.inner.lock().map_err(|_| SinkError {
            sink: self.name.clone(),
            reason: "poisoned".into(),
        })?;
        let tombstone = format!(r#"{{"retraction":true,"id":"{}"}}{}"#, id.as_str(), '\n');
        g.writer
            .write_all(tombstone.as_bytes())
            .map_err(|e| SinkError {
                sink: self.name.clone(),
                reason: format!("retract write: {e}"),
            })?;
        g.writer.flush().ok();
        g.retracted_ids.push(id.clone());
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::trajectory::RewardComponents;
    use super::*;
    use std::fs;

    fn tmp_path(suffix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("ai_assistant_feedback_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(format!("dataset_{}_{}.jsonl", std::process::id(), suffix))
    }

    fn sample(principal: &str) -> TrajectoryRecord {
        let mut r = TrajectoryRecord::new(principal);
        r.reward = RewardComponents {
            success: Some(1.0),
            faithfulness: Some(0.5),
            ..Default::default()
        };
        r
    }

    #[test]
    fn writes_one_line_per_record() {
        let p = tmp_path("basic");
        let _ = fs::remove_file(&p);
        let w = DatasetWriter::open("ds", &p).unwrap();
        w.deliver(&sample("alice")).unwrap();
        w.deliver(&sample("bob")).unwrap();
        drop(w);
        let text = fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("principal").is_some());
        }
    }

    #[test]
    fn skips_confidential_records() {
        let p = tmp_path("conf");
        let _ = fs::remove_file(&p);
        let w = DatasetWriter::open("ds", &p).unwrap();
        let mut r = sample("alice");
        r.privacy_tier = PrivacyTier::Confidential;
        w.deliver(&r).unwrap();
        assert_eq!(w.written_count(), 0);
        assert_eq!(w.skipped_confidential_count(), 1);
    }

    #[test]
    fn retraction_writes_tombstone() {
        let p = tmp_path("retract");
        let _ = fs::remove_file(&p);
        let w = DatasetWriter::open("ds", &p).unwrap();
        let r = sample("alice");
        let id = r.id.clone();
        w.deliver(&r).unwrap();
        w.retract(&id).unwrap();
        drop(w);
        let text = fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"retraction\":true"));
    }
}
