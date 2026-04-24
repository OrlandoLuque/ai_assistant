//! V102: generic GGUF weight auto-downloader.
//!
//! Downloads GGUF model files from Hugging Face (or any HTTPS source)
//! with resume, progress callback, SHA256 verification, and optional
//! bearer token for gated models. Designed to feed any local provider
//! that loads GGUF:
//!
//! - `llama.cpp` / `llama-server` (primary target)
//! - LM Studio (drop file into its models directory)
//! - Kobold.cpp, LocalAI, text-gen-webui (same format)
//! - Ollama via the `write_ollama_modelfile()` helper
//!
//! Feature-gated under `auto-download` so callers who only need the
//! existing provider dispatch keep a minimal dependency surface.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Progress callback: `(bytes_downloaded_so_far, total_bytes_if_known)`.
pub type ProgressFn = dyn FnMut(u64, Option<u64>) + Send;

/// Download request.
pub struct DownloadRequest {
    /// Direct HTTPS URL to the GGUF file. Hugging Face pattern:
    /// `https://huggingface.co/<repo>/resolve/main/<file>.gguf`.
    pub url: String,
    /// Where to write the final file (parent dir is created if missing).
    pub dest: PathBuf,
    /// Optional SHA256 hex (64 chars). When set, the download fails if
    /// the computed hash does not match after transfer.
    pub sha256: Option<String>,
    /// Optional bearer token (HF: from `HF_TOKEN` env var or user input)
    /// required for gated repos.
    pub bearer_token: Option<String>,
    /// If true, attempt `Range`-based resume when the destination
    /// already exists and is smaller than the remote file.
    pub resume: bool,
    /// HTTP timeout for the initial request and each chunk. Defaults to
    /// 60 s — GGUFs are large and upstream can be slow.
    pub timeout: Duration,
}

impl DownloadRequest {
    pub fn new(url: impl Into<String>, dest: impl Into<PathBuf>) -> Self {
        Self {
            url: url.into(),
            dest: dest.into(),
            sha256: None,
            bearer_token: None,
            resume: true,
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_sha256(mut self, sha: impl Into<String>) -> Self {
        self.sha256 = Some(sha.into());
        self
    }

    pub fn with_bearer_token(mut self, tok: impl Into<String>) -> Self {
        self.bearer_token = Some(tok.into());
        self
    }

    pub fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

/// Result of a successful download.
#[derive(Debug, Clone)]
pub struct DownloadedFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: Option<String>,
    pub resumed: bool,
}

/// Download a GGUF. Writes to `<dest>.part` and atomically renames to
/// `<dest>` on success.
///
/// Returns `Err(String)` with a human-readable message. On failure the
/// `.part` file is left in place so the next call can resume.
pub fn download(
    req: &DownloadRequest,
    mut progress: Option<Box<ProgressFn>>,
) -> Result<DownloadedFile, String> {
    if let Some(parent) = req.dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {}", parent.display(), e))?;
    }

    // If the final file already exists and hash matches (or no hash
    // given), skip the download entirely.
    if req.dest.exists() {
        let size = std::fs::metadata(&req.dest).map(|m| m.len()).unwrap_or(0);
        if size > 0 {
            if let Some(expected) = &req.sha256 {
                let got = hash_file(&req.dest)?;
                if &got == expected {
                    return Ok(DownloadedFile {
                        path: req.dest.clone(),
                        bytes: size,
                        sha256: Some(got),
                        resumed: false,
                    });
                }
            } else {
                return Ok(DownloadedFile {
                    path: req.dest.clone(),
                    bytes: size,
                    sha256: None,
                    resumed: false,
                });
            }
        }
    }

    let part = part_path(&req.dest);
    let existing_bytes = if req.resume && part.exists() {
        std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    // Build request.
    let mut agent = ureq::get(&req.url).timeout(req.timeout);
    if let Some(tok) = &req.bearer_token {
        agent = agent.set("Authorization", &format!("Bearer {}", tok));
    }
    if existing_bytes > 0 {
        agent = agent.set("Range", &format!("bytes={}-", existing_bytes));
    }

    let resp = agent
        .call()
        .map_err(|e| format!("GET {}: {}", req.url, e))?;
    let status = resp.status();
    if !(status == 200 || status == 206) {
        return Err(format!("unexpected HTTP status {} for {}", status, req.url));
    }

    // Total = existing + Content-Length (206 reports remaining length).
    let content_length: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let total_expected: Option<u64> = content_length.map(|cl| cl + existing_bytes);

    // Open .part in append-or-create mode.
    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(&part)
        .map_err(|e| format!("open {}: {}", part.display(), e))?;
    out.seek(SeekFrom::Start(existing_bytes))
        .map_err(|e| format!("seek {}: {}", part.display(), e))?;

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut written: u64 = existing_bytes;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("read from {}: {}", req.url, e))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("write {}: {}", part.display(), e))?;
        written += n as u64;
        if let Some(cb) = progress.as_mut() {
            cb(written, total_expected);
        }
    }
    out.flush()
        .map_err(|e| format!("flush {}: {}", part.display(), e))?;
    drop(out);

    // Verify hash if requested.
    let computed_hash = if req.sha256.is_some() {
        Some(hash_file(&part)?)
    } else {
        None
    };
    if let (Some(expected), Some(got)) = (&req.sha256, &computed_hash) {
        if expected != got {
            return Err(format!(
                "sha256 mismatch for {}: expected {}, got {}",
                req.url, expected, got
            ));
        }
    }

    // Atomic rename.
    std::fs::rename(&part, &req.dest)
        .map_err(|e| format!("rename {} -> {}: {}", part.display(), req.dest.display(), e))?;

    Ok(DownloadedFile {
        path: req.dest.clone(),
        bytes: written,
        sha256: computed_hash,
        resumed: existing_bytes > 0,
    })
}

/// Default cache directory for downloaded GGUFs.
///
/// - Windows: `%LOCALAPPDATA%\ai_assistant\models\`
/// - Linux/macOS: `$XDG_CACHE_HOME/ai_assistant/models/` or
///   `~/.cache/ai_assistant/models/`
pub fn default_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(p) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(p).join("ai_assistant").join("models");
        }
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("ai_assistant").join("models");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("ai_assistant")
            .join("models");
    }
    PathBuf::from("./ai_assistant_models")
}

/// Build the Hugging Face direct-download URL from a repo + file.
///
/// Example: `huggingface_resolve_url("prism-ml/Bonsai-8B-gguf",
/// "Bonsai-8B-Q1_0.gguf", None)` →
/// `https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf`
pub fn huggingface_resolve_url(repo: &str, filename: &str, revision: Option<&str>) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo.trim_matches('/'),
        revision.unwrap_or("main"),
        filename.trim_start_matches('/')
    )
}

/// Write an Ollama `Modelfile` pointing at a local GGUF, so users can
/// register the downloaded weights with Ollama via
/// `ollama create <name> -f <modelfile>`.
///
/// The `Modelfile` is the minimum viable entry: a single `FROM` line.
/// Callers who need templates, stop sequences, or parameters should
/// extend the returned file.
pub fn write_ollama_modelfile(modelfile_path: &Path, gguf_path: &Path) -> Result<(), String> {
    if let Some(parent) = modelfile_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {}", parent.display(), e))?;
    }
    let contents = format!("FROM {}\n", gguf_path.display());
    std::fs::write(modelfile_path, contents)
        .map_err(|e| format!("write {}: {}", modelfile_path.display(), e))?;
    Ok(())
}

/// Default Ollama models directory.
///
/// - `$OLLAMA_MODELS` if set
/// - Windows: `%USERPROFILE%\.ollama\models`
/// - Linux/macOS: `$HOME/.ollama/models`
pub fn default_ollama_models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OLLAMA_MODELS") {
        return PathBuf::from(p);
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(up) = std::env::var("USERPROFILE") {
            return PathBuf::from(up).join(".ollama").join("models");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".ollama").join("models");
    }
    PathBuf::from(".ollama/models")
}

/// Zero-copy registration with Ollama via hard-link.
///
/// Instead of letting Ollama copy the GGUF into its blob store
/// (duplicating 8+ GB on disk), this pre-creates the blob entry as a
/// **hard link** to the existing file. When Ollama processes `FROM
/// <path>` it hashes the file, finds the matching blob already in its
/// store, and skips the copy.
///
/// Result: the GGUF bytes exist once on disk; both our cache path and
/// Ollama's blob path reference the same on-disk data.
///
/// # Constraints
///
/// - **Same volume**: hard links require the source and the Ollama
///   blob directory to live on the same filesystem. If they don't,
///   this function returns an error; fall back to
///   [`register_with_ollama`].
/// - **Manifest still written by Ollama**: we never touch the
///   manifest; Ollama writes it through `/api/create` as usual.
/// - **Blob already present**: if a blob with the matching SHA256 is
///   already in Ollama's store, this function leaves it untouched.
///
/// `ollama_models_dir` overrides the auto-detected location (see
/// [`default_ollama_models_dir`]).
pub fn register_with_ollama_hardlink(
    ollama_url: &str,
    model_name: &str,
    gguf_path: &Path,
    ollama_models_dir: Option<&Path>,
) -> Result<(), String> {
    let hash = hash_file(gguf_path)?;
    let ollama_dir = ollama_models_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_ollama_models_dir);
    let blob_dir = ollama_dir.join("blobs");
    std::fs::create_dir_all(&blob_dir)
        .map_err(|e| format!("create_dir_all {}: {}", blob_dir.display(), e))?;
    let blob_path = blob_dir.join(format!("sha256-{}", hash));

    if !blob_path.exists() {
        std::fs::hard_link(gguf_path, &blob_path).map_err(|e| {
            format!(
                "hard_link {} -> {}: {} (hard links require the source and \
                 Ollama's blob dir to be on the same volume; fall back to \
                 register_with_ollama)",
                gguf_path.display(),
                blob_path.display(),
                e
            )
        })?;
    }

    // Ollama still writes the manifest + small config blob via
    // /api/create. When it hashes `gguf_path` it finds our pre-seeded
    // blob and skips the copy.
    register_with_ollama(ollama_url, model_name, gguf_path)
}

/// Register a downloaded GGUF with a running Ollama daemon.
///
/// Hits `POST {ollama_url}/api/create` with a synthetic `Modelfile`
/// body — equivalent to `ollama create <name> -f Modelfile` but
/// without requiring the `ollama` CLI on the caller's machine.
///
/// `ollama_url` defaults to `http://localhost:11434` (Ollama's standard
/// port). Returns `Ok(())` once the daemon has finished streaming
/// progress events. Any error from the daemon (missing file, malformed
/// Modelfile, IO failure) surfaces as `Err(String)`.
pub fn register_with_ollama(
    ollama_url: &str,
    model_name: &str,
    gguf_path: &Path,
) -> Result<(), String> {
    let url = format!("{}/api/create", ollama_url.trim_end_matches('/'));
    let modelfile = format!("FROM {}\n", gguf_path.display());
    let body = serde_json::json!({
        "name": model_name,
        "modelfile": modelfile,
        "stream": false,
    });
    let resp = ureq::post(&url)
        .timeout(Duration::from_secs(600))
        .send_json(body)
        .map_err(|e| format!("POST {}: {}", url, e))?;
    let status = resp.status();
    if !(200..300).contains(&status) {
        let msg = resp.into_string().unwrap_or_default();
        return Err(format!("ollama /api/create returned {}: {}", status, msg));
    }
    Ok(())
}

// --- helpers ---------------------------------------------------------

fn part_path(dest: &Path) -> PathBuf {
    let mut p = dest.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

fn hash_file(p: &Path) -> Result<String, String> {
    let mut f = File::open(p).map_err(|e| format!("open {}: {}", p.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {}", p.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(nybble(b >> 4));
        out.push(nybble(b & 0x0f));
    }
    out
}

fn nybble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + n - 10) as char,
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_url_default_main_revision() {
        let u = huggingface_resolve_url("prism-ml/Bonsai-8B-gguf", "Bonsai-8B-Q1_0.gguf", None);
        assert_eq!(
            u,
            "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf"
        );
    }

    #[test]
    fn hf_url_custom_revision() {
        let u = huggingface_resolve_url("foo/bar", "model.gguf", Some("v1.2"));
        assert_eq!(u, "https://huggingface.co/foo/bar/resolve/v1.2/model.gguf");
    }

    #[test]
    fn hf_url_strips_slashes() {
        let u = huggingface_resolve_url("/foo/bar/", "/model.gguf", None);
        assert_eq!(u, "https://huggingface.co/foo/bar/resolve/main/model.gguf");
    }

    #[test]
    fn part_path_adds_part_suffix() {
        let p = part_path(Path::new("/tmp/x.gguf"));
        assert_eq!(p.to_string_lossy(), "/tmp/x.gguf.part");
    }

    #[test]
    fn hex_lower_encodes_bytes() {
        assert_eq!(hex_lower(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(hex_lower(&[0x00, 0x01, 0x0f, 0xff]), "00010fff");
    }

    #[test]
    fn default_cache_dir_is_under_ai_assistant() {
        let d = default_cache_dir();
        let s = d.to_string_lossy();
        assert!(s.contains("ai_assistant"), "{}", s);
        assert!(s.ends_with("models") || s.contains("models"), "{}", s);
    }

    #[test]
    fn ollama_modelfile_single_from_line() {
        let dir =
            std::env::temp_dir().join(format!("ai_assistant_v102_mfile_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("Bonsai-8B-Q1_0.gguf");
        std::fs::write(&gguf, b"fake-gguf").unwrap();
        let mf = dir.join("Modelfile");
        write_ollama_modelfile(&mf, &gguf).unwrap();
        let body = std::fs::read_to_string(&mf).unwrap();
        assert!(body.starts_with("FROM "));
        assert!(body.contains("Bonsai-8B-Q1_0.gguf"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hash_file_matches_known_vector() {
        let dir =
            std::env::temp_dir().join(format!("ai_assistant_v102_hash_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc.bin");
        std::fs::write(&p, b"abc").unwrap();
        let h = hash_file(&p).unwrap();
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn default_ollama_models_dir_contains_dot_ollama() {
        let d = default_ollama_models_dir();
        let s = d.to_string_lossy();
        assert!(s.contains(".ollama"), "{}", s);
        assert!(s.ends_with("models") || s.contains("models"), "{}", s);
    }

    /// Simulates the hard-link step without calling Ollama: verifies
    /// that a blob named `sha256-<hex>` is created and that it points
    /// at the same on-disk bytes as the source (both are writable via
    /// either path — the defining property of a hard link).
    #[test]
    fn hardlink_creates_blob_at_expected_path() {
        let dir = std::env::temp_dir().join(format!("ai_assistant_v102_hl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("source.gguf");
        std::fs::write(&src, b"same-bytes").unwrap();
        let hash = hash_file(&src).unwrap();

        let ollama_dir = dir.join("ollama_models");
        let blob_dir = ollama_dir.join("blobs");
        std::fs::create_dir_all(&blob_dir).unwrap();
        let blob = blob_dir.join(format!("sha256-{}", hash));

        std::fs::hard_link(&src, &blob).unwrap();
        assert!(blob.exists());
        let via_blob = std::fs::read(&blob).unwrap();
        assert_eq!(&via_blob, b"same-bytes");
        // Mutate through blob; source should reflect it (same inode).
        std::fs::write(&blob, b"other-bytes").unwrap();
        let via_src = std::fs::read(&src).unwrap();
        assert_eq!(&via_src, b"other-bytes");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_request_builder_sets_fields() {
        let req = DownloadRequest::new("http://x/y.gguf", "/tmp/y.gguf")
            .with_sha256("deadbeef")
            .with_bearer_token("hf_xxx")
            .with_resume(false)
            .with_timeout(Duration::from_secs(10));
        assert_eq!(req.url, "http://x/y.gguf");
        assert_eq!(req.sha256.as_deref(), Some("deadbeef"));
        assert_eq!(req.bearer_token.as_deref(), Some("hf_xxx"));
        assert!(!req.resume);
        assert_eq!(req.timeout, Duration::from_secs(10));
    }
}
