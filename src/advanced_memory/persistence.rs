//! Auto-persistence: configuration for automatic memory persistence with
//! compressed snapshots and checksum verification.

/// Configuration for automatic memory persistence.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutoPersistenceConfig {
    /// Base directory for persistence files.
    pub base_dir: std::path::PathBuf,
    /// Save interval in seconds (0 = disabled).
    pub save_interval_secs: u64,
    /// Maximum number of snapshot files to keep (older ones are rotated out).
    pub max_snapshots: usize,
    /// Whether to save on Drop.
    pub save_on_drop: bool,
}

impl Default for AutoPersistenceConfig {
    fn default() -> Self {
        Self {
            base_dir: std::path::PathBuf::from("."),
            save_interval_secs: 300,
            max_snapshots: 5,
            save_on_drop: true,
        }
    }
}

impl AutoPersistenceConfig {
    /// Create a new config with the given base directory.
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            ..Default::default()
        }
    }

    /// Generate a timestamped snapshot filename for the given store name.
    pub fn snapshot_path(&self, store_name: &str) -> std::path::PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.base_dir
            .join(format!("{}_{}.json", store_name, timestamp))
    }

    /// List existing snapshots for a store, sorted oldest first.
    pub fn list_snapshots(&self, store_name: &str) -> Vec<std::path::PathBuf> {
        let prefix = format!("{}_", store_name);
        let mut snapshots: Vec<std::path::PathBuf> = std::fs::read_dir(&self.base_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|e| e == "json").unwrap_or(false)
                    && p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.starts_with(&prefix))
                        .unwrap_or(false)
            })
            .collect();
        snapshots.sort();
        snapshots
    }

    /// Rotate snapshots: remove oldest files if more than max_snapshots exist.
    pub fn rotate_snapshots(&self, store_name: &str) {
        let mut snapshots = self.list_snapshots(store_name);
        while snapshots.len() > self.max_snapshots {
            if let Some(oldest) = snapshots.first() {
                let _ = std::fs::remove_file(oldest);
            }
            snapshots.remove(0);
        }
    }

    // ============================================================
    // Compressed Snapshots (7.1)
    // ============================================================

    /// Save data as a gzip-compressed JSON snapshot.
    pub fn save_compressed(&self, store_name: &str, data: &[u8]) -> Result<(), String> {
        use std::io::Write;
        let path = self.snapshot_path(store_name).with_extension("json.gz");
        let tmp = path.with_extension("tmp.gz");
        let file = std::fs::File::create(&tmp).map_err(|e| format!("Create error: {}", e))?;
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        encoder.write_all(data).map_err(|e| format!("Compress error: {}", e))?;
        encoder.finish().map_err(|e| format!("Flush error: {}", e))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("Rename error: {}", e))?;
        Ok(())
    }

    /// Load and decompress a gzip JSON snapshot.
    pub fn load_compressed(path: &std::path::Path) -> Result<Vec<u8>, String> {
        use std::io::Read;
        let file = std::fs::File::open(path).map_err(|e| format!("Open error: {}", e))?;
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data).map_err(|e| format!("Decompress error: {}", e))?;
        Ok(data)
    }

    // ============================================================
    // Checksum Verification (7.2)
    // ============================================================

    /// Compute a simple checksum (FNV-1a hash) of data.
    pub fn compute_checksum(data: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Save data with a checksum sidecar file (.checksum).
    pub fn save_with_checksum(path: &std::path::Path, data: &[u8]) -> Result<(), String> {
        let checksum = Self::compute_checksum(data);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, data).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("Rename error: {}", e))?;
        let checksum_path = path.with_extension("checksum");
        std::fs::write(&checksum_path, checksum.to_string())
            .map_err(|e| format!("Checksum write error: {}", e))?;
        Ok(())
    }

    /// Load data and verify its checksum.
    pub fn load_with_checksum(path: &std::path::Path) -> Result<Vec<u8>, String> {
        let data = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        let checksum_path = path.with_extension("checksum");
        if checksum_path.exists() {
            let stored = std::fs::read_to_string(&checksum_path)
                .map_err(|e| format!("Checksum read error: {}", e))?;
            let stored_checksum: u64 = stored.trim().parse()
                .map_err(|e| format!("Checksum parse error: {}", e))?;
            let computed = Self::compute_checksum(&data);
            if stored_checksum != computed {
                return Err(format!("Checksum mismatch: stored={}, computed={}", stored_checksum, computed));
            }
        }
        Ok(data)
    }
}
