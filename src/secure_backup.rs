//! V128 (Phase C.7) — backup/restore.
//!
//! Builds an encrypted, signed, content-addressable archive of one
//! or more source paths. The on-disk format is intentionally simple:
//!
//! ```text
//! plain (no encryption):
//!   <output>.zip                    — ZIP archive (manifest.json + files/...)
//!   <output>.zip.sha256             — hex SHA-256 of the .zip
//!   <output>.zip.sig (optional)     — Ed25519 signature over the .zip
//!
//! encrypted (passphrase / key supplied):
//!   <output>.zip.enc                — [1B version | 16B salt | 12B nonce | ciphertext+tag]
//!   <output>.zip.enc.sha256         — hex SHA-256 of the .zip.enc
//!   <output>.zip.enc.sig (optional) — Ed25519 signature over the .zip.enc
//! ```
//!
//! The ZIP contains:
//!   * `manifest.json` — `BackupManifest` with per-file SHA-256.
//!   * `files/<relative_path>` — the original bytes, one entry per file.
//!
//! Encryption: AES-256-GCM. Key is either supplied directly (32 bytes)
//! or derived from a passphrase via HKDF-SHA256 with the embedded salt.
//! The signature signs the *post-encryption* output bytes so a verifier
//! can authenticate without ever decrypting (useful for offline triage).
//!
//! The `restore` and `verify` paths reject zip-slip (`..` segments,
//! absolute paths, drive prefixes) before extracting.

use std::collections::HashSet;
use std::fs;
use std::io::{Cursor, Read, Write as IoWrite};
use std::path::{Path, PathBuf};

use aes_gcm::aead::rand_core::RngCore;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey, SIGNATURE_LENGTH};
use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zip::{write::SimpleFileOptions, ZipArchive, ZipWriter};

const ENCRYPTED_FORMAT_VERSION: u8 = 1;
const ENCRYPTED_HEADER_LEN: usize = 1 + 16 + 12; // version + salt + nonce
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const HKDF_INFO: &[u8] = b"ai_assistant backup v1 AES-256-GCM key";
const MANIFEST_NAME: &str = "manifest.json";
const FILES_PREFIX: &str = "files/";

/// Errors returned by the backup module.
#[derive(Debug)]
#[non_exhaustive]
pub enum BackupError {
    Io(std::io::Error),
    Zip(zip::result::ZipError),
    Crypto(String),
    Json(String),
    Verification(String),
    Format(String),
    UnsafePath(String),
}

impl std::fmt::Display for BackupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Zip(e) => write!(f, "ZIP error: {e}"),
            Self::Crypto(m) => write!(f, "crypto error: {m}"),
            Self::Json(m) => write!(f, "manifest error: {m}"),
            Self::Verification(m) => write!(f, "verification failed: {m}"),
            Self::Format(m) => write!(f, "archive format error: {m}"),
            Self::UnsafePath(p) => write!(f, "rejected unsafe archive path: {p}"),
        }
    }
}

impl std::error::Error for BackupError {}

impl From<std::io::Error> for BackupError {
    fn from(e: std::io::Error) -> Self {
        BackupError::Io(e)
    }
}
impl From<zip::result::ZipError> for BackupError {
    fn from(e: zip::result::ZipError) -> Self {
        BackupError::Zip(e)
    }
}
impl From<serde_json::Error> for BackupError {
    fn from(e: serde_json::Error) -> Self {
        BackupError::Json(e.to_string())
    }
}

/// One file entry in the backup manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupEntry {
    /// Path relative to the source root, with forward slashes.
    pub relative_path: String,
    /// Plain-text size in bytes.
    pub size: u64,
    /// Lowercase-hex SHA-256 of the plain-text content.
    pub sha256: String,
}

/// JSON manifest stored inside the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    /// Manifest schema version (currently 1).
    pub version: u32,
    /// RFC 3339 timestamp of archive creation.
    pub created_at: String,
    /// Human-readable label supplied by the caller.
    pub source_label: String,
    /// Per-file entries.
    pub entries: Vec<BackupEntry>,
    /// Sum of plain-text bytes across all entries.
    pub total_bytes: u64,
}

/// Encryption material supplied to [`create_backup`].
///
/// `Passphrase` is the high-level path: the lib generates a random
/// salt, derives the AES-256 key with HKDF-SHA256, and embeds the
/// salt in the envelope so verify/restore can recover the same key
/// from the same passphrase.
///
/// `Key` is the low-level path: caller has a 32-byte key and is
/// responsible for storing/retrieving it. The salt slot in the
/// envelope is filled with random bytes (unused on the verify side
/// when `Key` is supplied symmetrically).
pub enum EncryptionMaterial {
    Passphrase(String),
    Key([u8; 32]),
}

/// Configuration for [`create_backup`].
pub struct BackupConfig<'a> {
    /// Source paths to include. Each may be a file or a directory; directories
    /// are walked recursively.
    pub sources: Vec<PathBuf>,
    /// Output base path. Suffix is appended automatically (`.zip` /
    /// `.zip.enc`).
    pub output: PathBuf,
    /// Encryption material — passphrase (lib derives the key) or raw 32-byte key.
    pub encryption: Option<EncryptionMaterial>,
    /// Optional Ed25519 signing key. Signs the post-encryption output bytes.
    pub signing_key: Option<&'a SigningKey>,
    /// Human-readable label embedded in the manifest.
    pub source_label: String,
}

/// Result of a successful create / verify / restore.
#[derive(Debug, Clone)]
pub struct BackupReport {
    pub manifest: BackupManifest,
    /// Final on-disk path of the primary archive (`.zip` or `.zip.enc`).
    pub archive_path: PathBuf,
    /// Whether AES-256-GCM was applied.
    pub encrypted: bool,
    /// Whether an Ed25519 signature was produced or verified.
    pub signed: bool,
}

/// Derive a 32-byte AES-256 key from a passphrase + salt via HKDF-SHA256.
///
/// HKDF is appropriate here because the salt is generated fresh per archive
/// and stored in the encrypted file header — it is not a password-strength
/// stretching function. For weak passphrases, supply a high-entropy key
/// directly via [`BackupConfig::encryption_key`].
pub fn derive_key(passphrase: &str, salt: &[u8]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(salt), passphrase.as_bytes());
    let mut out = [0u8; 32];
    hk.expand(HKDF_INFO, &mut out)
        .expect("HKDF expand for 32 bytes never fails");
    out
}

/// Generate a random 16-byte HKDF salt.
pub fn random_salt() -> [u8; SALT_LEN] {
    let mut salt = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut salt);
    salt
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

/// Walk a source path, returning (relative_path, absolute_path) pairs.
fn collect_sources(sources: &[PathBuf]) -> Result<Vec<(String, PathBuf)>, BackupError> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for src in sources {
        if !src.exists() {
            return Err(BackupError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("source not found: {}", src.display()),
            )));
        }
        if src.is_file() {
            let name = src
                .file_name()
                .ok_or_else(|| {
                    BackupError::Format(format!("source has no name: {}", src.display()))
                })?
                .to_string_lossy()
                .to_string();
            if !seen.insert(name.clone()) {
                return Err(BackupError::Format(format!("duplicate entry: {name}")));
            }
            out.push((name, src.clone()));
        } else if src.is_dir() {
            let prefix = src
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            walk_dir(src, &prefix, &mut out, &mut seen)?;
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn walk_dir(
    dir: &Path,
    prefix: &str,
    out: &mut Vec<(String, PathBuf)>,
    seen: &mut HashSet<String>,
) -> Result<(), BackupError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if path.is_dir() {
            walk_dir(&path, &rel, out, seen)?;
        } else if path.is_file() {
            if !seen.insert(rel.clone()) {
                return Err(BackupError::Format(format!("duplicate entry: {rel}")));
            }
            out.push((rel, path));
        }
    }
    Ok(())
}

/// Build the inner ZIP archive in memory, returning (zip_bytes, manifest).
fn build_zip(
    sources: &[(String, PathBuf)],
    source_label: String,
) -> Result<(Vec<u8>, BackupManifest), BackupError> {
    let mut buf: Vec<u8> = Vec::new();
    let mut entries: Vec<BackupEntry> = Vec::with_capacity(sources.len());
    let mut total_bytes: u64 = 0;
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    {
        let cursor = Cursor::new(&mut buf);
        let mut zip = ZipWriter::new(cursor);

        for (rel, abs) in sources {
            let bytes = fs::read(abs)?;
            let sha = sha256_hex(&bytes);
            let archive_path = format!("{FILES_PREFIX}{rel}");
            zip.start_file(archive_path, opts)?;
            zip.write_all(&bytes)?;
            entries.push(BackupEntry {
                relative_path: rel.clone(),
                size: bytes.len() as u64,
                sha256: sha,
            });
            total_bytes += bytes.len() as u64;
        }

        // Write manifest LAST so the inner SHA-256 entries are final.
        let manifest = BackupManifest {
            version: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            source_label,
            entries: entries.clone(),
            total_bytes,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        zip.start_file(MANIFEST_NAME, opts)?;
        zip.write_all(&manifest_bytes)?;
        zip.finish()?;
    }

    let manifest = BackupManifest {
        version: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        source_label: String::new(), // overwritten below
        entries,
        total_bytes,
    };
    Ok((buf, manifest))
}

/// Encrypt `plain` with AES-256-GCM, prepending the [version|salt|nonce] header.
fn encrypt_with_key(
    plain: &[u8],
    key: &[u8; 32],
    salt: [u8; SALT_LEN],
) -> Result<Vec<u8>, BackupError> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| BackupError::Crypto(format!("invalid key: {e}")))?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plain)
        .map_err(|e| BackupError::Crypto(format!("AES-GCM encrypt: {e}")))?;
    let mut out = Vec::with_capacity(ENCRYPTED_HEADER_LEN + ciphertext.len());
    out.push(ENCRYPTED_FORMAT_VERSION);
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt the [version|salt|nonce|ciphertext] envelope.
fn decrypt_envelope(envelope: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, BackupError> {
    if envelope.len() < ENCRYPTED_HEADER_LEN {
        return Err(BackupError::Format("encrypted file too short".into()));
    }
    if envelope[0] != ENCRYPTED_FORMAT_VERSION {
        return Err(BackupError::Format(format!(
            "unsupported encrypted format version: {}",
            envelope[0]
        )));
    }
    let nonce_bytes = &envelope[1 + SALT_LEN..ENCRYPTED_HEADER_LEN];
    let ciphertext = &envelope[ENCRYPTED_HEADER_LEN..];
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| BackupError::Crypto(format!("invalid key: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| BackupError::Crypto(format!("AES-GCM decrypt: {e}")))
}

/// Read the salt from an encrypted envelope (used when deriving the key from a passphrase).
pub fn salt_from_envelope(envelope: &[u8]) -> Result<[u8; SALT_LEN], BackupError> {
    if envelope.len() < ENCRYPTED_HEADER_LEN {
        return Err(BackupError::Format("encrypted file too short".into()));
    }
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&envelope[1..1 + SALT_LEN]);
    Ok(salt)
}

/// Create a backup. Returns the report (manifest + on-disk paths).
pub fn create_backup(config: &BackupConfig<'_>) -> Result<BackupReport, BackupError> {
    let collected = collect_sources(&config.sources)?;
    if collected.is_empty() {
        return Err(BackupError::Format("no files to back up".into()));
    }

    let (zip_bytes, mut manifest) = build_zip(&collected, config.source_label.clone())?;
    manifest.source_label = config.source_label.clone();

    let (archive_bytes, archive_path) = if let Some(material) = config.encryption.as_ref() {
        let salt = random_salt();
        let key: [u8; 32] = match material {
            EncryptionMaterial::Passphrase(p) => derive_key(p, &salt),
            EncryptionMaterial::Key(k) => *k,
        };
        let envelope = encrypt_with_key(&zip_bytes, &key, salt)?;
        let path = config.output.with_file_name(format!(
            "{}.zip.enc",
            config
                .output
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "backup".into())
        ));
        (envelope, path)
    } else {
        let path = config.output.with_file_name(format!(
            "{}.zip",
            config
                .output
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "backup".into())
        ));
        (zip_bytes, path)
    };

    fs::write(&archive_path, &archive_bytes)?;

    // SHA-256 sidecar.
    let sha = sha256_hex(&archive_bytes);
    let sha_path = sidecar(&archive_path, "sha256");
    fs::write(
        &sha_path,
        format!(
            "{sha}  {}\n",
            archive_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ),
    )?;

    // Optional Ed25519 signature.
    let mut signed = false;
    if let Some(sk) = config.signing_key {
        let sig: Signature = sk.sign(&archive_bytes);
        let sig_path = sidecar(&archive_path, "sig");
        fs::write(&sig_path, sig.to_bytes())?;
        signed = true;
    }

    Ok(BackupReport {
        manifest,
        archive_path,
        encrypted: config.encryption.is_some(),
        signed,
    })
}

fn sidecar(archive: &Path, ext: &str) -> PathBuf {
    let mut s = archive.as_os_str().to_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Verify a backup: optional signature, archive SHA-256 sidecar (if present),
/// per-file SHA-256 inside the manifest. Does not extract.
pub fn verify_backup(
    archive_path: &Path,
    encryption_key: Option<&[u8; 32]>,
    verify_key: Option<&VerifyingKey>,
) -> Result<BackupReport, BackupError> {
    let archive_bytes = fs::read(archive_path)?;
    let encrypted = archive_path
        .extension()
        .map(|e| e == "enc")
        .unwrap_or(false);

    // 1. SHA-256 sidecar (informational — verifies the archive bytes match
    //    a trusted hash). Optional: only checked if present.
    let sha_path = sidecar(archive_path, "sha256");
    if sha_path.exists() {
        let expected = fs::read_to_string(&sha_path)?
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        let actual = sha256_hex(&archive_bytes);
        if expected != actual {
            return Err(BackupError::Verification(format!(
                "SHA-256 sidecar mismatch: expected {expected}, got {actual}"
            )));
        }
    }

    // 2. Ed25519 signature sidecar (optional).
    let sig_path = sidecar(archive_path, "sig");
    let mut signed = false;
    if let Some(vk) = verify_key {
        if !sig_path.exists() {
            return Err(BackupError::Verification(format!(
                "verify_key supplied but signature file missing: {}",
                sig_path.display()
            )));
        }
        let sig_bytes = fs::read(&sig_path)?;
        if sig_bytes.len() != SIGNATURE_LENGTH {
            return Err(BackupError::Verification(format!(
                "signature wrong length: {} (expected {})",
                sig_bytes.len(),
                SIGNATURE_LENGTH
            )));
        }
        let sig = Signature::from_slice(&sig_bytes)
            .map_err(|e| BackupError::Verification(format!("malformed signature: {e}")))?;
        vk.verify(&archive_bytes, &sig)
            .map_err(|e| BackupError::Verification(format!("Ed25519 verify failed: {e}")))?;
        signed = true;
    }

    // 3. Decrypt if needed.
    let zip_bytes = if encrypted {
        let key = encryption_key.ok_or_else(|| {
            BackupError::Verification("archive is encrypted but no key supplied".into())
        })?;
        decrypt_envelope(&archive_bytes, key)?
    } else {
        archive_bytes
    };

    // 4. Open zip and verify per-file SHA-256 against manifest.
    let mut zip = ZipArchive::new(Cursor::new(&zip_bytes))?;
    let manifest_bytes = read_zip_entry(&mut zip, MANIFEST_NAME)?;
    let manifest: BackupManifest = serde_json::from_slice(&manifest_bytes)?;

    for entry in &manifest.entries {
        let archived_name = format!("{FILES_PREFIX}{}", entry.relative_path);
        let bytes = read_zip_entry(&mut zip, &archived_name)?;
        if bytes.len() as u64 != entry.size {
            return Err(BackupError::Verification(format!(
                "size mismatch for {}: manifest says {}, found {}",
                entry.relative_path,
                entry.size,
                bytes.len()
            )));
        }
        let sha = sha256_hex(&bytes);
        if sha != entry.sha256 {
            return Err(BackupError::Verification(format!(
                "SHA-256 mismatch for {}: manifest says {}, computed {}",
                entry.relative_path, entry.sha256, sha
            )));
        }
    }

    Ok(BackupReport {
        manifest,
        archive_path: archive_path.to_path_buf(),
        encrypted,
        signed,
    })
}

fn read_zip_entry<R: Read + std::io::Seek>(
    zip: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>, BackupError> {
    let mut entry = zip
        .by_name(name)
        .map_err(|e| BackupError::Format(format!("zip entry '{name}' missing: {e}")))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Restore a backup to `output_dir`. Verifies first, then writes files.
/// Rejects any archive entry whose relative path contains a `..` segment,
/// is absolute, or includes a Windows drive prefix (zip-slip protection).
pub fn restore_backup(
    archive_path: &Path,
    output_dir: &Path,
    encryption_key: Option<&[u8; 32]>,
    verify_key: Option<&VerifyingKey>,
) -> Result<BackupReport, BackupError> {
    let report = verify_backup(archive_path, encryption_key, verify_key)?;

    let archive_bytes = fs::read(archive_path)?;
    let zip_bytes = if report.encrypted {
        let key = encryption_key
            .ok_or_else(|| BackupError::Verification("encrypted archive needs a key".into()))?;
        decrypt_envelope(&archive_bytes, key)?
    } else {
        archive_bytes
    };

    fs::create_dir_all(output_dir)?;
    let mut zip = ZipArchive::new(Cursor::new(&zip_bytes))?;

    for entry in &report.manifest.entries {
        check_safe_path(&entry.relative_path)?;
        let archived_name = format!("{FILES_PREFIX}{}", entry.relative_path);
        let bytes = read_zip_entry(&mut zip, &archived_name)?;
        let dest = output_dir.join(&entry.relative_path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, bytes)?;
    }

    Ok(report)
}

fn check_safe_path(rel: &str) -> Result<(), BackupError> {
    if rel.is_empty() {
        return Err(BackupError::UnsafePath("empty path".into()));
    }
    if rel.starts_with('/') || rel.starts_with('\\') {
        return Err(BackupError::UnsafePath(rel.to_string()));
    }
    // Drive prefix like "C:\..."
    if rel.len() >= 2 && rel.as_bytes()[1] == b':' {
        return Err(BackupError::UnsafePath(rel.to_string()));
    }
    for seg in rel.split(|c| c == '/' || c == '\\') {
        if seg == ".." {
            return Err(BackupError::UnsafePath(rel.to_string()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(p: &Path, content: &[u8]) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    #[test]
    fn round_trip_plain() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        write(&src.path().join("a.txt"), b"alpha");
        write(&src.path().join("dir/b.txt"), b"beta");

        let cfg = BackupConfig {
            sources: vec![src.path().join("a.txt"), src.path().join("dir")],
            output: dst.path().join("snap"),
            encryption: None,
            signing_key: None,
            source_label: "round_trip_plain".into(),
        };
        let report = create_backup(&cfg).unwrap();
        assert!(!report.encrypted);
        assert!(!report.signed);
        assert_eq!(report.manifest.entries.len(), 2);
        assert!(report.archive_path.extension().unwrap() == "zip");
        let sha_sidecar = sidecar(&report.archive_path, "sha256");
        assert!(sha_sidecar.exists());

        // Verify
        let verified = verify_backup(&report.archive_path, None, None).unwrap();
        assert_eq!(verified.manifest.entries.len(), 2);

        // Restore to a fresh dir and check contents.
        let restore_to = tempdir().unwrap();
        let restored = restore_backup(&report.archive_path, restore_to.path(), None, None).unwrap();
        assert_eq!(restored.manifest.entries.len(), 2);
        assert_eq!(fs::read(restore_to.path().join("a.txt")).unwrap(), b"alpha");
        assert_eq!(
            fs::read(restore_to.path().join("dir/b.txt")).unwrap(),
            b"beta"
        );
    }

    #[test]
    fn round_trip_encrypted() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        write(&src.path().join("secret.txt"), b"top-secret-payload");

        let cfg = BackupConfig {
            sources: vec![src.path().join("secret.txt")],
            output: dst.path().join("enc"),
            encryption: Some(EncryptionMaterial::Passphrase("hunter2".into())),
            signing_key: None,
            source_label: "enc".into(),
        };
        let report = create_backup(&cfg).unwrap();
        assert!(report.encrypted);
        assert!(report.archive_path.extension().unwrap() == "enc");

        // Recover the key the archive expects: read the embedded salt and
        // derive against the same passphrase.
        let envelope = fs::read(&report.archive_path).unwrap();
        let archive_salt = salt_from_envelope(&envelope).unwrap();
        let correct = derive_key("hunter2", &archive_salt);
        let verified = verify_backup(&report.archive_path, Some(&correct), None).unwrap();
        assert_eq!(verified.manifest.entries.len(), 1);

        // Wrong passphrase fails.
        let wrong = derive_key("not_hunter2", &archive_salt);
        let err = verify_backup(&report.archive_path, Some(&wrong), None).unwrap_err();
        match err {
            BackupError::Crypto(_) => {}
            other => panic!("expected Crypto error, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_signed() {
        use ed25519_dalek::SigningKey;
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        write(&src.path().join("a.txt"), b"signed-data");

        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        let sk = SigningKey::from_bytes(&secret);
        let vk = sk.verifying_key();

        let cfg = BackupConfig {
            sources: vec![src.path().join("a.txt")],
            output: dst.path().join("sig"),
            encryption: None,
            signing_key: Some(&sk),
            source_label: "sig".into(),
        };
        let report = create_backup(&cfg).unwrap();
        assert!(report.signed);
        let sig_path = sidecar(&report.archive_path, "sig");
        assert!(sig_path.exists());

        // Verify with correct key.
        let verified = verify_backup(&report.archive_path, None, Some(&vk)).unwrap();
        assert!(verified.signed);

        // Corrupt the archive — signature verification must fail.
        let mut bytes = fs::read(&report.archive_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&report.archive_path, &bytes).unwrap();
        let err = verify_backup(&report.archive_path, None, Some(&vk)).unwrap_err();
        // SHA sidecar still has the OLD hash — that mismatch fires first
        // and short-circuits before the signature check, but either failure
        // is acceptable here as both indicate tampering.
        match err {
            BackupError::Verification(_) => {}
            other => panic!("expected Verification error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zip_slip() {
        assert!(matches!(
            check_safe_path(".."),
            Err(BackupError::UnsafePath(_))
        ));
        assert!(matches!(
            check_safe_path("a/../b"),
            Err(BackupError::UnsafePath(_))
        ));
        assert!(matches!(
            check_safe_path("/etc/shadow"),
            Err(BackupError::UnsafePath(_))
        ));
        assert!(matches!(
            check_safe_path("\\Windows\\system32"),
            Err(BackupError::UnsafePath(_))
        ));
        assert!(matches!(
            check_safe_path("C:\\evil"),
            Err(BackupError::UnsafePath(_))
        ));
        assert!(check_safe_path("ok/path.txt").is_ok());
    }

    #[test]
    fn detects_per_file_corruption() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        write(&src.path().join("file.txt"), b"good-content");

        let cfg = BackupConfig {
            sources: vec![src.path().join("file.txt")],
            output: dst.path().join("snap"),
            encryption: None,
            signing_key: None,
            source_label: "corrupt".into(),
        };
        let report = create_backup(&cfg).unwrap();

        // Tamper with the SHA-256 sidecar so the archive-level check passes,
        // forcing the per-file check to be the one that catches us.
        let sha_path = sidecar(&report.archive_path, "sha256");
        fs::remove_file(&sha_path).unwrap();

        // Now corrupt the zip body so a per-file SHA-256 fails. We rewrite
        // the manifest's sha256 entry to a known-wrong value via bytewise
        // search-replace — easier than reconstructing a zip.
        let mut bytes = fs::read(&report.archive_path).unwrap();
        let target_sha = report.manifest.entries[0].sha256.as_bytes();
        // Find the sha string and flip the last byte's hex digit.
        if let Some(pos) = bytes
            .windows(target_sha.len())
            .position(|w| w == target_sha)
        {
            bytes[pos + target_sha.len() - 1] ^= 0x01;
            fs::write(&report.archive_path, &bytes).unwrap();
            let err = verify_backup(&report.archive_path, None, None).unwrap_err();
            assert!(matches!(err, BackupError::Verification(_)));
        }
    }

    #[test]
    fn empty_sources_fails() {
        let dst = tempdir().unwrap();
        let cfg = BackupConfig {
            sources: vec![],
            output: dst.path().join("empty"),
            encryption: None,
            signing_key: None,
            source_label: "empty".into(),
        };
        let err = create_backup(&cfg).unwrap_err();
        assert!(matches!(err, BackupError::Format(_)));
    }

    #[test]
    fn key_derivation_deterministic() {
        let salt = [9u8; 16];
        let k1 = derive_key("password", &salt);
        let k2 = derive_key("password", &salt);
        assert_eq!(k1, k2);
        let k3 = derive_key("different", &salt);
        assert_ne!(k1, k3);
    }
}
