// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Backup and restore for configuration and data directories.
//!
//! Uses the `zip` crate (already a dependency under the `documents` feature,
//! but available as an optional dep) for creating portable archives.
//! Falls back to a simple gzip-based concatenation using `flate2` (always available).

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Information about a completed backup.
#[derive(Debug, Clone)]
pub struct BackupInfo {
    /// Path to the created archive.
    pub path: PathBuf,
    /// Size of the archive in bytes.
    pub size_bytes: u64,
    /// Number of files included.
    pub files_count: usize,
}

/// File entry header for our simple archive format.
/// Format per entry: [path_len: u32][path: bytes][data_len: u64][data: bytes]
const ARCHIVE_MAGIC: &[u8; 8] = b"AIBACKUP";
const ARCHIVE_VERSION: u8 = 1;

/// Create a backup archive of the configuration directory.
///
/// The archive includes all `.toml`, `.json`, and `.db` files.
/// If `include_models` is true, also includes `.gguf` and `.bin` model files
/// (which can be very large).
pub fn create_backup(
    config_dir: &Path,
    output: &Path,
    include_models: bool,
) -> Result<BackupInfo, String> {
    if !config_dir.exists() {
        return Err(format!(
            "Config directory does not exist: {}",
            config_dir.display()
        ));
    }

    let mut files: Vec<(PathBuf, Vec<u8>)> = Vec::new();

    collect_files(config_dir, config_dir, include_models, &mut files)?;

    if files.is_empty() {
        return Err("No files found to backup".to_string());
    }

    let files_count = files.len();

    // Build the archive in memory, then gzip it
    let mut archive = Vec::new();
    archive.extend_from_slice(ARCHIVE_MAGIC);
    archive.push(ARCHIVE_VERSION);

    // Number of entries
    let count_bytes = (files_count as u32).to_le_bytes();
    archive.extend_from_slice(&count_bytes);

    for (rel_path, data) in &files {
        let path_str = rel_path.to_string_lossy();
        let path_bytes = path_str.as_bytes();

        // Path length (u32) + path + data length (u64) + data
        archive.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        archive.extend_from_slice(path_bytes);
        archive.extend_from_slice(&(data.len() as u64).to_le_bytes());
        archive.extend_from_slice(data);
    }

    // Gzip compress
    let file = std::fs::File::create(output)
        .map_err(|e| format!("Cannot create {}: {}", output.display(), e))?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(&archive)
        .map_err(|e| format!("Compression failed: {}", e))?;
    encoder
        .finish()
        .map_err(|e| format!("Failed to finalize archive: {}", e))?;

    let size_bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    Ok(BackupInfo {
        path: output.to_path_buf(),
        size_bytes,
        files_count,
    })
}

/// Restore a backup archive to the target directory.
///
/// Existing files will be overwritten. Directories are created as needed.
pub fn restore_backup(archive: &Path, target_dir: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive)
        .map_err(|e| format!("Cannot open {}: {}", archive.display(), e))?;
    let mut decoder = GzDecoder::new(file);
    let mut data = Vec::new();
    decoder
        .read_to_end(&mut data)
        .map_err(|e| format!("Decompression failed: {}", e))?;

    if data.len() < 13 {
        // magic(8) + version(1) + count(4)
        return Err("Invalid backup archive: too small".to_string());
    }

    // Verify magic
    if &data[0..8] != ARCHIVE_MAGIC {
        return Err("Invalid backup archive: bad magic header".to_string());
    }

    let version = data[8];
    if version != ARCHIVE_VERSION {
        return Err(format!("Unsupported archive version: {}", version));
    }

    let count = u32::from_le_bytes([data[9], data[10], data[11], data[12]]) as usize;
    let mut pos = 13;

    for _ in 0..count {
        if pos + 4 > data.len() {
            return Err("Truncated archive: path length".to_string());
        }
        let path_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + path_len > data.len() {
            return Err("Truncated archive: path data".to_string());
        }
        let path_str = String::from_utf8_lossy(&data[pos..pos + path_len]).to_string();
        pos += path_len;

        if pos + 8 > data.len() {
            return Err("Truncated archive: data length".to_string());
        }
        let data_len = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;

        if pos + data_len > data.len() {
            return Err("Truncated archive: file data".to_string());
        }
        let file_data = &data[pos..pos + data_len];
        pos += data_len;

        // Security: prevent path traversal
        let rel = PathBuf::from(&path_str);
        if rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(format!("Path traversal detected in archive: {}", path_str));
        }

        let out_path = target_dir.join(&rel);
        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out_path, file_data)
            .map_err(|e| format!("Failed to write {}: {}", out_path.display(), e))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Recursively collect files from a directory.
fn collect_files(
    base: &Path,
    dir: &Path,
    include_models: bool,
    files: &mut Vec<(PathBuf, Vec<u8>)>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Cannot read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Directory entry error: {}", e))?;
        let path = entry.path();

        if path.is_dir() {
            collect_files(base, &path, include_models, files)?;
        } else if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let should_include = match ext.as_str() {
                "toml" | "json" | "db" | "sqlite" | "log" => true,
                "gguf" | "bin" => include_models,
                _ => false,
            };

            if should_include {
                let rel_path = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
                let data = std::fs::read(&path)
                    .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
                files.push((rel_path, data));
            }
        }
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_backup_and_restore_roundtrip() {
        let src_dir = std::env::temp_dir().join(format!("ai_backup_src_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&src_dir);

        // Create test files
        let config = src_dir.join("config.toml");
        std::fs::write(&config, "[provider]\ntype = \"ollama\"\n").unwrap();

        let subdir = src_dir.join("sub");
        let _ = std::fs::create_dir_all(&subdir);
        let db = subdir.join("data.db");
        std::fs::write(&db, b"fake-db-content").unwrap();

        // A .txt file should be excluded
        let txt = src_dir.join("notes.txt");
        std::fs::write(&txt, "should not be included").unwrap();

        // Create backup
        let archive = std::env::temp_dir().join(format!("ai_backup_{}.gz", uuid::Uuid::new_v4()));
        let info = create_backup(&src_dir, &archive, false).unwrap();
        assert_eq!(
            info.files_count, 2,
            "Should include .toml and .db but not .txt"
        );
        assert!(info.size_bytes > 0);

        // Restore to a different directory
        let dst_dir = std::env::temp_dir().join(format!("ai_backup_dst_{}", uuid::Uuid::new_v4()));
        restore_backup(&archive, &dst_dir).unwrap();

        // Verify contents
        let restored_config = std::fs::read_to_string(dst_dir.join("config.toml")).unwrap();
        assert!(restored_config.contains("ollama"));

        let restored_db = std::fs::read(dst_dir.join("sub").join("data.db")).unwrap();
        assert_eq!(restored_db, b"fake-db-content");

        // .txt should NOT exist in restore
        assert!(!dst_dir.join("notes.txt").exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(&src_dir);
        let _ = std::fs::remove_dir_all(&dst_dir);
        let _ = std::fs::remove_file(&archive);
    }

    #[test]
    fn test_backup_empty_dir() {
        let src_dir =
            std::env::temp_dir().join(format!("ai_backup_empty_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&src_dir);

        let archive =
            std::env::temp_dir().join(format!("ai_backup_empty_{}.gz", uuid::Uuid::new_v4()));
        let result = create_backup(&src_dir, &archive, false);
        assert!(result.is_err(), "Empty directory should produce an error");

        let _ = std::fs::remove_dir_all(&src_dir);
    }
}
