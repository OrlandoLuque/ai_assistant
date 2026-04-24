//! Atomic, self-describing checkpoints for a breeder run. Each checkpoint
//! captures `(config_hash, generation, population, lineage, ledger_tip_hash)`
//! so a later run can resume deterministically or so an auditor can replay
//! state at a specific point.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::config::PromptBreederConfig;
use super::population::{LineageDag, Population};

const MAGIC: &[u8] = b"AIBR-CKPT\x01";

/// Serialisable snapshot of the breeder at a generation boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub run_id: String,
    pub generation: u32,
    pub config_hash_hex: String,
    pub ledger_tip_hash_hex: String,
    pub population: Population,
    pub lineage: LineageDag,
}

impl Checkpoint {
    pub fn new(
        run_id: impl Into<String>,
        generation: u32,
        config: &PromptBreederConfig,
        ledger_tip_hash_hex: impl Into<String>,
        population: Population,
        lineage: LineageDag,
    ) -> Self {
        let config_hash_hex = hex_encode(&config.canonical_hash());
        Self {
            run_id: run_id.into(),
            generation,
            config_hash_hex,
            ledger_tip_hash_hex: ledger_tip_hash_hex.into(),
            population,
            lineage,
        }
    }

    /// Validate that a checkpoint on disk matches the current config. A
    /// config hash mismatch means the user edited the run shape between
    /// runs, and resume would be unsafe.
    pub fn matches_config(&self, config: &PromptBreederConfig) -> bool {
        self.config_hash_hex == hex_encode(&config.canonical_hash())
    }
}

/// Write a checkpoint atomically (tmp file + rename). Returns bytes written.
pub fn write(path: &Path, ckpt: &Checkpoint) -> std::io::Result<usize> {
    let payload = serde_json::to_vec(ckpt)
        .map_err(|e| std::io::Error::other(format!("checkpoint serialize: {e}")))?;
    let mut buf = Vec::with_capacity(MAGIC.len() + payload.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&payload);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = tmp_path(path);
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, path)?;
    Ok(buf.len())
}

/// Read and parse a checkpoint file.
pub fn read(path: &Path) -> Result<Checkpoint, CheckpointError> {
    let bytes = std::fs::read(path).map_err(|e| CheckpointError::Io(e.to_string()))?;
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    serde_json::from_slice(&bytes[MAGIC.len()..]).map_err(|e| CheckpointError::Parse(e.to_string()))
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.to_path_buf();
    let file_name = s
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "checkpoint".to_string());
    s.set_file_name(format!("{file_name}.tmp"));
    s
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[derive(Debug, Clone)]
pub enum CheckpointError {
    BadMagic,
    Io(String),
    Parse(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("checkpoint: bad magic / incompatible file"),
            Self::Io(s) => write!(f, "checkpoint io: {s}"),
            Self::Parse(s) => write!(f, "checkpoint parse: {s}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_breeder::config::PromptBreederConfig;
    use crate::prompt_breeder::population::{LineageDag, Population};

    fn cfg() -> PromptBreederConfig {
        let mut c = PromptBreederConfig::new("ollama", "mistral:7b");
        c.task_description = "sort".into();
        c
    }

    #[test]
    fn round_trip_checkpoint() {
        let tmp = std::env::temp_dir().join("ai_breeder_ckpt_roundtrip.json");
        let ckpt = Checkpoint::new(
            "run-1",
            5,
            &cfg(),
            "abc123",
            Population::new(),
            LineageDag::new(),
        );
        let n = write(&tmp, &ckpt).unwrap();
        assert!(n > MAGIC.len());
        let loaded = read(&tmp).unwrap();
        assert_eq!(loaded.run_id, "run-1");
        assert_eq!(loaded.generation, 5);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn matches_config_detects_shape_change() {
        let c1 = cfg();
        let mut c2 = cfg();
        c2.population_size = 40;
        let ckpt = Checkpoint::new("r", 0, &c1, "tip", Population::new(), LineageDag::new());
        assert!(ckpt.matches_config(&c1));
        assert!(!ckpt.matches_config(&c2));
    }

    #[test]
    fn bad_magic_rejected() {
        let tmp = std::env::temp_dir().join("ai_breeder_ckpt_bad.json");
        std::fs::write(&tmp, b"not a checkpoint").unwrap();
        assert!(matches!(read(&tmp), Err(CheckpointError::BadMagic)));
        let _ = std::fs::remove_file(&tmp);
    }
}
