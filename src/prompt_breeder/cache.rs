//! Evaluation cache — memoizes `(prompt_hash, input_hash, provider_fp)` →
//! `FitnessScore`. `SelfConsistency{k}` extends the key with `sample_idx` so
//! cached samples retain the variance they were designed to measure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::config::{EvalCacheMode, ProviderFingerprint};
use super::fitness::FitnessScore;

/// Cache lookup key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub prompt_hash_hex: String,
    pub input_hash_hex: String,
    pub fingerprint: String,
    pub sample_idx: u32,
}

impl CacheKey {
    pub fn build(prompt: &str, input: &str, fp: &ProviderFingerprint, sample_idx: u32) -> Self {
        Self {
            prompt_hash_hex: blake3_hex(prompt),
            input_hash_hex: blake3_hex(input),
            fingerprint: fp.as_str().to_string(),
            sample_idx,
        }
    }
}

fn blake3_hex(s: &str) -> String {
    #[cfg(feature = "prompt-breeder")]
    {
        blake3::hash(s.as_bytes()).to_hex().to_string()
    }
    #[cfg(not(feature = "prompt-breeder"))]
    {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);
        format!("defhash:{:016x}", h.finish())
    }
}

/// Outcome of a cache lookup.
#[derive(Debug)]
pub enum CacheHit {
    Miss,
    Hit(FitnessScore),
}

/// In-memory or disk-backed cache.
#[derive(Clone)]
pub struct EvalCache {
    mode: EvalCacheMode,
    inner: Arc<RwLock<HashMap<CacheKey, FitnessScore>>>,
    path: Option<PathBuf>,
}

impl EvalCache {
    pub fn new(mode: EvalCacheMode) -> Self {
        let path = match &mode {
            EvalCacheMode::Persistent { path } => Some(path.clone()),
            _ => None,
        };
        let mut inner = HashMap::new();
        if let Some(p) = &path {
            if let Ok(bytes) = std::fs::read(p) {
                if let Ok(map) = deserialize_cache(&bytes) {
                    inner = map;
                }
            }
        }
        Self {
            mode,
            inner: Arc::new(RwLock::new(inner)),
            path,
        }
    }

    pub fn disabled(&self) -> bool {
        matches!(self.mode, EvalCacheMode::Disabled)
    }

    pub fn get(&self, key: &CacheKey) -> CacheHit {
        if self.disabled() {
            return CacheHit::Miss;
        }
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(_) => return CacheHit::Miss,
        };
        match guard.get(key) {
            Some(s) => CacheHit::Hit(s.clone()),
            None => CacheHit::Miss,
        }
    }

    pub fn put(&self, key: CacheKey, score: FitnessScore) {
        if self.disabled() {
            return;
        }
        if let Ok(mut g) = self.inner.write() {
            g.insert(key, score);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush to disk when in `Persistent` mode. Returns the number of
    /// bytes written. No-op otherwise.
    pub fn flush(&self) -> std::io::Result<usize> {
        let Some(path) = &self.path else {
            return Ok(0);
        };
        let guard = self
            .inner
            .read()
            .map_err(|_| std::io::Error::other("cache lock poisoned"))?;
        let bytes = serialize_cache(&guard)?;
        atomic_write(path, &bytes)?;
        Ok(bytes.len())
    }

    /// Clear the in-memory state (does not delete the on-disk file).
    pub fn clear(&self) {
        if let Ok(mut g) = self.inner.write() {
            g.clear();
        }
    }
}

// Magic bytes + version so we can reject incompatible files cleanly.
const MAGIC: &[u8] = b"AIBR\x01";

fn serialize_cache(map: &HashMap<CacheKey, FitnessScore>) -> std::io::Result<Vec<u8>> {
    let payload =
        serde_json::to_vec(map).map_err(|e| std::io::Error::other(format!("serialize: {e}")))?;
    let mut out = Vec::with_capacity(MAGIC.len() + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&payload);
    Ok(out)
}

fn deserialize_cache(bytes: &[u8]) -> Result<HashMap<CacheKey, FitnessScore>, CacheFormatError> {
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err(CacheFormatError::BadMagic);
    }
    let tail = &bytes[MAGIC.len()..];
    serde_json::from_slice(tail).map_err(|e| CacheFormatError::Parse(e.to_string()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[derive(Debug, Clone)]
pub enum CacheFormatError {
    BadMagic,
    Parse(String),
}

impl std::fmt::Display for CacheFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("bad magic / incompatible cache file"),
            Self::Parse(s) => write!(f, "cache parse error: {s}"),
        }
    }
}

impl std::error::Error for CacheFormatError {}

#[cfg(test)]
mod tests {
    use super::super::config::Metric;
    use super::*;

    fn fp() -> ProviderFingerprint {
        ProviderFingerprint::new("test", "mock")
    }

    fn score() -> FitnessScore {
        let mut s = FitnessScore::new(fp());
        s.set(Metric::Accuracy, 0.5);
        s.aggregate = 0.5;
        s
    }

    #[test]
    fn disabled_mode_never_hits() {
        let c = EvalCache::new(EvalCacheMode::Disabled);
        c.put(CacheKey::build("p", "i", &fp(), 0), score());
        assert!(matches!(
            c.get(&CacheKey::build("p", "i", &fp(), 0)),
            CacheHit::Miss
        ));
    }

    #[test]
    fn enabled_round_trip() {
        let c = EvalCache::new(EvalCacheMode::Enabled);
        let k = CacheKey::build("prompt", "input", &fp(), 0);
        assert!(matches!(c.get(&k), CacheHit::Miss));
        c.put(k.clone(), score());
        match c.get(&k) {
            CacheHit::Hit(s) => assert_eq!(s.aggregate, 0.5),
            _ => panic!("expected hit"),
        }
    }

    #[test]
    fn sample_idx_disambiguates_self_consistency() {
        let c = EvalCache::new(EvalCacheMode::Enabled);
        let k0 = CacheKey::build("p", "i", &fp(), 0);
        let k1 = CacheKey::build("p", "i", &fp(), 1);
        c.put(k0.clone(), score());
        assert!(matches!(c.get(&k0), CacheHit::Hit(_)));
        assert!(matches!(c.get(&k1), CacheHit::Miss));
    }
}
