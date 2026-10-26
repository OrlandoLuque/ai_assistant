//! HTTP download helper for benchmark datasets.
//!
//! Thin wrapper around the already-present `ureq` crate (no new deps).
//! Responsibilities:
//!  * Try each candidate URL in order, stop at the first 2xx.
//!  * Cap the downloaded size (defensive against bombs / mistaken URLs).
//!  * Write atomically: download to `<path>.part`, rename on success.
//!  * Skip download if the cached file is already the expected size.

use crate::http_client::{HttpClient, HttpRequest, UreqClient};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::types::BenchmarkError;

/// Maximum bytes we will pull for a single benchmark file. 200 MB is well
/// above any dataset we actually support (TruthfulQA ~1 MB, HaluEval ~35 MB,
/// FEVER claim-only ~50 MB).
pub const MAX_DOWNLOAD_BYTES: u64 = 200 * 1024 * 1024;

/// Options for a download. Kept small on purpose; callers that need more
/// control can compose `ureq` directly.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Plazo TOTAL de la descarga, como tope.
    ///
    /// Valia 60 s, y con `timeout_global` eso es de punta a punta --- sigue
    /// contando mientras se escribe el cuerpo. O sea que cualquier corpus que
    /// tardara mas de un minuto en transferirse moria por plazo, y los de aqui
    /// llegan a 50 MB (FEVER) y a los cientos (MLDR). Lo que guarda la descarga
    /// ahora es `stall_timeout`; esto es solo el techo.
    pub timeout: Duration,
    /// Plazo de ATASCO: segundos de silencio antes de rendirse.
    ///
    /// Es el plazo correcto para una descarga, porque los bytes fluyen desde el
    /// primero: un minuto sin recibir nada **si** significa que se murio algo,
    /// y no significa "el fichero es grande".
    pub stall_timeout: Duration,
    /// Optional expected size in bytes for cache-hit short-circuit + sanity check.
    pub expected_size: Option<u64>,
    /// Override of the global download cap.
    pub max_bytes: u64,
    /// User-Agent header.
    pub user_agent: String,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            timeout: crate::http_client::DEFAULT_LONG_LIVED_TIMEOUT,
            stall_timeout: crate::http_client::DEFAULT_STALL_TIMEOUT,
            expected_size: None,
            max_bytes: MAX_DOWNLOAD_BYTES,
            user_agent: concat!("ai_assistant-eval_benchmarks/", env!("CARGO_PKG_VERSION"))
                .to_string(),
        }
    }
}

/// Try each URL in `urls` until one returns 2xx, streaming the body into
/// `dest`. If `dest` already satisfies the expected-size check, returns
/// immediately without touching the network.
///
/// Returns the path (same as `dest`) on success.
pub fn download_file(
    urls: &[&str],
    dest: &Path,
    opts: &DownloadOptions,
) -> Result<PathBuf, BenchmarkError> {
    if urls.is_empty() {
        return Err(BenchmarkError::Network("no download URLs provided".into()));
    }

    // Cache hit?
    if let Ok(meta) = std::fs::metadata(dest) {
        if meta.is_file() {
            match opts.expected_size {
                Some(n) if n > 0 && meta.len() == n => return Ok(dest.to_path_buf()),
                None if meta.len() > 0 => return Ok(dest.to_path_buf()),
                _ => {}
            }
        }
    }

    // Ensure parent dir exists.
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| BenchmarkError::Io(format!("mkdir {}: {e}", parent.display())))?;
    }

    let mut last_err: Option<BenchmarkError> = None;
    for url in urls {
        match download_one(url, dest, opts) {
            Ok(path) => return Ok(path),
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| BenchmarkError::Network("all URLs failed".into())))
}

fn download_one(url: &str, dest: &Path, opts: &DownloadOptions) -> Result<PathBuf, BenchmarkError> {
    // Detras del puerto (N77): el `user_agent` que era configuracion del
    // agente pasa a ser la cabecera que siempre fue, y el brazo de
    // `Error::StatusCode` DESAPARECE --- el no-2xx ya no es un error, asi que la
    // comprobacion de abajo es el unico sitio donde se decide. Antes habia dos,
    // y el de arriba se comia el codigo sin mirar el cuerpo.
    // `send_streaming` y NO `send`: esto descarga un corpus, y traerlo entero a
    // un `Vec<u8>` antes de escribirlo seria el error del "GGUF en RAM" que el
    // propio puerto documenta. El estado se comprueba ANTES de abrir el `.part`,
    // que es para lo que `HttpStream` lo lleva: sin el, un 404 con pagina de
    // error se escribiria y se renombraria como si fuera el corpus.
    let request = HttpRequest::get(url)
        .long_lived(opts.timeout, opts.stall_timeout)
        .header("User-Agent", opts.user_agent.clone());
    let resp = UreqClient
        .send_streaming(&request)
        .map_err(|e| BenchmarkError::Network(format!("{url}: {e:#}")))?;

    let status = resp.status;
    if !(200..300).contains(&status) {
        return Err(BenchmarkError::Http {
            status,
            url: url.to_string(),
        });
    }

    // Write to `<dest>.part` first, then rename.
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let mut file = std::fs::File::create(&part)
        .map_err(|e| BenchmarkError::Io(format!("create {}: {e}", part.display())))?;

    let mut reader = resp.body;
    let mut written: u64 = 0;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| BenchmarkError::Network(format!("read {url}: {e}")))?;
        if n == 0 {
            break;
        }
        written += n as u64;
        if written > opts.max_bytes {
            let _ = std::fs::remove_file(&part);
            return Err(BenchmarkError::SizeCheck(format!(
                "download from {url} exceeded max {} bytes",
                opts.max_bytes
            )));
        }
        file.write_all(&buf[..n])
            .map_err(|e| BenchmarkError::Io(format!("write {}: {e}", part.display())))?;
    }
    file.flush()
        .map_err(|e| BenchmarkError::Io(format!("flush {}: {e}", part.display())))?;
    drop(file);

    if let Some(expected) = opts.expected_size {
        if expected > 0 && written != expected {
            let _ = std::fs::remove_file(&part);
            return Err(BenchmarkError::SizeCheck(format!(
                "{url} returned {written} bytes, expected {expected}"
            )));
        }
    }

    std::fs::rename(&part, dest).map_err(|e| {
        BenchmarkError::Io(format!(
            "rename {} -> {}: {e}",
            part.display(),
            dest.display()
        ))
    })?;
    Ok(dest.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_url_list_is_error() {
        let dest = std::env::temp_dir().join("eb_nope.bin");
        let err = download_file(&[], &dest, &DownloadOptions::default()).unwrap_err();
        match err {
            BenchmarkError::Network(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn cache_hit_short_circuits() {
        let tmp = std::env::temp_dir().join(format!("eb_hit_{}.txt", std::process::id()));
        std::fs::write(&tmp, b"cached").unwrap();
        let got = download_file(
            &["http://invalid.invalid/never-called"],
            &tmp,
            &DownloadOptions {
                expected_size: Some(6),
                ..DownloadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(got, tmp);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn default_options_have_ua_with_version() {
        let opts = DownloadOptions::default();
        assert!(opts.user_agent.starts_with("ai_assistant-eval_benchmarks/"));
        assert!(opts.max_bytes > 0);
    }
    /// Lo mismo que en el descargador de GGUF, y por el mismo motivo: el plazo
    /// de `DownloadOptions` valia 60 s y era TOTAL, asi que cualquier corpus
    /// que tardara mas de un minuto en transferirse moria --- y aqui hay
    /// corpus de 50 MB (FEVER) y de cientos (MLDR). Lo pidio la mutacion M26.
    #[test]
    fn test_a_slow_corpus_download_is_not_a_timeout() {
        let server = crate::http_client::StallingServer::dribbling(12, Duration::from_millis(120));
        let dir = std::env::temp_dir().join(format!("corpus_slow_{}", std::process::id()));
        let dest = dir.join("corpus.jsonl");
        let _ = std::fs::remove_dir_all(&dir);

        let url = server.url();
        let opts = DownloadOptions {
            timeout: Duration::from_secs(10),
            stall_timeout: Duration::from_millis(400),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let path = download_file(&[&url], &dest, &opts)
            .expect("una descarga lenta no es un plazo agotado");
        let elapsed = started.elapsed();

        assert_eq!(std::fs::read(&path).expect("escrito").len(), 12);
        assert!(
            elapsed >= Duration::from_millis(800),
            "acabo en {:?}, asi que el servidor no goteo",
            elapsed
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
