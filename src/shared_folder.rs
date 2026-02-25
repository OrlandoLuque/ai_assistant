// shared_folder.rs — Shared folder abstraction for host/container file sharing.
//
// Provides a SharedFolder that wraps a local directory which can be
// bind-mounted into Docker containers. Supports cloud sync via the
// CloudStorage trait from cloud_connectors.
//
// Feature-gated behind `containers` (gate applied in lib.rs, not here).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[cfg(feature = "cloud-connectors")]
use crate::cloud_connectors::{CloudStorage, ListOptions};

/// A shared folder between host and Docker containers.
///
/// Wraps a local directory that can be bind-mounted into containers.
/// Supports synchronization to/from cloud storage (S3, Google Drive).
///
/// # Usage
/// ```rust,no_run
/// use ai_assistant::SharedFolder;
///
/// // Create a temporary shared folder
/// let folder = SharedFolder::temp().unwrap();
/// folder.put_file("report.md", b"# Report\nHello").unwrap();
///
/// // The folder can be mounted into a container via bind_mount_spec()
/// let mount = folder.bind_mount_spec();
/// // mount = "/tmp/ai_shared_abc123:/workspace"
/// ```
pub struct SharedFolder {
    /// Host filesystem path
    host_path: PathBuf,
    /// Container mount point (default: /workspace)
    container_path: String,
    /// Whether the folder is temporary (auto-deleted on drop)
    temporary: bool,
    /// Maximum folder size in bytes (advisory, checked on demand)
    max_size_bytes: Option<u64>,
}

impl SharedFolder {
    /// Create a shared folder at the given host path.
    /// Creates the directory if it doesn't exist.
    pub fn new(host_path: impl Into<PathBuf>) -> Result<Self> {
        let path = host_path.into();
        if !path.exists() {
            std::fs::create_dir_all(&path)
                .with_context(|| format!("Failed to create shared folder at {:?}", path))?;
        }
        Ok(Self {
            host_path: path,
            container_path: "/workspace".into(),
            temporary: false,
            max_size_bytes: None,
        })
    }

    /// Create a temporary shared folder that is deleted on drop.
    pub fn temp() -> Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "ai_shared_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create temp shared folder at {:?}", dir))?;
        Ok(Self {
            host_path: dir,
            container_path: "/workspace".into(),
            temporary: true,
            max_size_bytes: Some(100 * 1024 * 1024), // 100 MB default for temp
        })
    }

    /// Set the container mount point (default: "/workspace").
    pub fn with_container_path(mut self, path: impl Into<String>) -> Self {
        self.container_path = path.into();
        self
    }

    /// Set maximum folder size in bytes.
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = Some(bytes);
        self
    }

    /// Get the host filesystem path.
    pub fn host_path(&self) -> &Path {
        &self.host_path
    }

    /// Get the container mount point.
    pub fn container_path(&self) -> &str {
        &self.container_path
    }

    /// Get the Docker bind mount spec: "host_path:container_path".
    pub fn bind_mount_spec(&self) -> String {
        format!("{}:{}", self.host_path.display(), self.container_path)
    }

    /// List all files in the shared folder (recursively).
    /// Returns paths relative to the shared folder root.
    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        Self::list_files_recursive(&self.host_path, &self.host_path, &mut files)?;
        Ok(files)
    }

    fn list_files_recursive(base: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("Failed to read directory {:?}", dir))?
            {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    Self::list_files_recursive(base, &path, files)?;
                } else if let Ok(rel) = path.strip_prefix(base) {
                    files.push(rel.to_path_buf());
                }
            }
        }
        Ok(())
    }

    /// Read a file from the shared folder.
    pub fn get_file(&self, relative_path: &str) -> Result<Vec<u8>> {
        let full_path = self.host_path.join(relative_path);
        std::fs::read(&full_path)
            .with_context(|| format!("Failed to read file {:?}", full_path))
    }

    /// Write a file to the shared folder.
    /// Creates parent directories if needed.
    pub fn put_file(&self, relative_path: &str, data: &[u8]) -> Result<()> {
        // Check size limit
        if let Some(max) = self.max_size_bytes {
            let current = self.size_bytes().unwrap_or(0);
            if current + data.len() as u64 > max {
                anyhow::bail!(
                    "Shared folder size limit exceeded: {} + {} > {}",
                    current,
                    data.len(),
                    max
                );
            }
        }

        let full_path = self.host_path.join(relative_path);
        if let Some(parent) = full_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create directory {:?}", parent))?;
            }
        }
        std::fs::write(&full_path, data)
            .with_context(|| format!("Failed to write file {:?}", full_path))
    }

    /// Delete a file from the shared folder.
    pub fn delete_file(&self, relative_path: &str) -> Result<()> {
        let full_path = self.host_path.join(relative_path);
        if full_path.exists() {
            std::fs::remove_file(&full_path)
                .with_context(|| format!("Failed to delete {:?}", full_path))?;
        }
        Ok(())
    }

    /// Check if a file exists in the shared folder.
    pub fn file_exists(&self, relative_path: &str) -> bool {
        self.host_path.join(relative_path).exists()
    }

    /// Get total size of all files in the shared folder (bytes).
    pub fn size_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        Self::dir_size(&self.host_path, &mut total)?;
        Ok(total)
    }

    fn dir_size(dir: &Path, total: &mut u64) -> Result<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    Self::dir_size(&path, total)?;
                } else {
                    *total += entry.metadata()?.len();
                }
            }
        }
        Ok(())
    }

    /// Sync all files from the shared folder to a cloud storage backend.
    /// Returns the number of files uploaded.
    #[cfg(feature = "cloud-connectors")]
    pub fn sync_to_cloud(&self, storage: &dyn CloudStorage, prefix: &str) -> Result<usize> {
        let files = self.list_files()?;
        let mut count = 0;
        for file in &files {
            let data = self.get_file(&file.to_string_lossy())?;
            let key = format!(
                "{}/{}",
                prefix.trim_end_matches('/'),
                file.display()
            );
            let content_type = Self::guess_content_type(&file.to_string_lossy());
            storage.put(&key, &data, content_type.as_deref())?;
            count += 1;
        }
        Ok(count)
    }

    /// Sync files from cloud storage into the shared folder.
    /// Returns the number of files downloaded.
    #[cfg(feature = "cloud-connectors")]
    pub fn sync_from_cloud(&self, storage: &dyn CloudStorage, prefix: &str) -> Result<usize> {
        let list = storage.list(&ListOptions {
            prefix: Some(prefix.to_string()),
            max_results: Some(1000),
            page_token: None,
        })?;
        let mut count = 0;
        for obj in &list.objects {
            let relative = obj.key.strip_prefix(prefix).unwrap_or(&obj.key);
            let relative = relative.trim_start_matches('/');
            if relative.is_empty() {
                continue;
            }
            let data = storage.get(&obj.key)?;
            self.put_file(relative, &data)?;
            count += 1;
        }
        Ok(count)
    }

    /// Clear all files in the shared folder.
    pub fn clear(&self) -> Result<()> {
        if self.host_path.exists() {
            for entry in std::fs::read_dir(&self.host_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)?;
                } else {
                    std::fs::remove_file(&path)?;
                }
            }
        }
        Ok(())
    }

    /// Guess MIME content type from file extension.
    #[cfg(feature = "cloud-connectors")]
    fn guess_content_type(filename: &str) -> Option<String> {
        let ext = filename.rsplit('.').next()?.to_lowercase();
        match ext.as_str() {
            "pdf" => Some("application/pdf".into()),
            "docx" => Some(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
            ),
            "xlsx" => Some(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".into(),
            ),
            "pptx" => Some(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation".into(),
            ),
            "html" | "htm" => Some("text/html".into()),
            "txt" | "md" | "csv" => Some("text/plain".into()),
            "json" => Some("application/json".into()),
            "png" => Some("image/png".into()),
            "jpg" | "jpeg" => Some("image/jpeg".into()),
            "svg" => Some("image/svg+xml".into()),
            "mp3" => Some("audio/mpeg".into()),
            "wav" => Some("audio/wav".into()),
            "ogg" => Some("audio/ogg".into()),
            _ => None,
        }
    }
}

impl Drop for SharedFolder {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_dir_all(&self.host_path);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cloud-connectors")]
    use crate::cloud_connectors::{CloudObject, ListResult};
    #[cfg(feature = "cloud-connectors")]
    use std::collections::HashMap;
    #[cfg(feature = "cloud-connectors")]
    use std::sync::Mutex;

    /// Helper: create a unique temp directory for a test.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ai_shared_test_{}_{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Helper: clean up a test directory.
    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // ========================================================================
    // Mock cloud storage for sync tests
    // ========================================================================

    #[cfg(feature = "cloud-connectors")]
    struct MockCloudStorage {
        files: Mutex<HashMap<String, Vec<u8>>>,
    }

    #[cfg(feature = "cloud-connectors")]
    impl MockCloudStorage {
        fn new() -> Self {
            Self {
                files: Mutex::new(HashMap::new()),
            }
        }
    }

    #[cfg(feature = "cloud-connectors")]
    impl CloudStorage for MockCloudStorage {
        fn list(&self, options: &ListOptions) -> Result<ListResult> {
            let files = self.files.lock().unwrap();
            let mut objects: Vec<CloudObject> = files
                .iter()
                .filter(|(k, _)| {
                    if let Some(ref prefix) = options.prefix {
                        k.starts_with(prefix)
                    } else {
                        true
                    }
                })
                .map(|(k, v)| CloudObject {
                    key: k.clone(),
                    size: v.len() as u64,
                    last_modified: 0,
                    content_type: None,
                    metadata: HashMap::new(),
                })
                .collect();
            objects.sort_by(|a, b| a.key.cmp(&b.key));
            if let Some(max) = options.max_results {
                objects.truncate(max);
            }
            Ok(ListResult {
                objects,
                next_page_token: None,
            })
        }

        fn get(&self, key: &str) -> Result<Vec<u8>> {
            let files = self.files.lock().unwrap();
            files
                .get(key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Key not found: {}", key))
        }

        fn put(&self, key: &str, data: &[u8], _content_type: Option<&str>) -> Result<()> {
            let mut files = self.files.lock().unwrap();
            files.insert(key.to_string(), data.to_vec());
            Ok(())
        }

        fn delete(&self, key: &str) -> Result<()> {
            let mut files = self.files.lock().unwrap();
            files.remove(key);
            Ok(())
        }

        fn exists(&self, key: &str) -> Result<bool> {
            let files = self.files.lock().unwrap();
            Ok(files.contains_key(key))
        }

        fn provider_name(&self) -> &str {
            "Mock"
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[test]
    fn test_new_creates_directory() {
        let dir = test_dir("new_creates");
        assert!(!dir.exists());

        let folder = SharedFolder::new(&dir).unwrap();
        assert!(dir.exists());
        assert!(dir.is_dir());
        assert_eq!(folder.host_path(), dir);

        cleanup(&dir);
    }

    #[test]
    fn test_temp_creates_directory() {
        let folder = SharedFolder::temp().unwrap();
        let path = folder.host_path().to_path_buf();
        assert!(path.exists());
        assert!(path.is_dir());
        assert!(
            path.to_string_lossy().contains("ai_shared_"),
            "Temp path should contain 'ai_shared_' prefix"
        );
    }

    #[test]
    fn test_put_and_get_file() {
        let dir = test_dir("put_get");
        let folder = SharedFolder::new(&dir).unwrap();

        let data = b"Hello, World!";
        folder.put_file("test.txt", data).unwrap();

        let read_back = folder.get_file("test.txt").unwrap();
        assert_eq!(read_back, data);

        cleanup(&dir);
    }

    #[test]
    fn test_put_file_creates_subdirs() {
        let dir = test_dir("put_subdirs");
        let folder = SharedFolder::new(&dir).unwrap();

        let data = b"nested content";
        folder.put_file("a/b/c/deep.txt", data).unwrap();

        assert!(dir.join("a").join("b").join("c").join("deep.txt").exists());
        let read_back = folder.get_file("a/b/c/deep.txt").unwrap();
        assert_eq!(read_back, data);

        cleanup(&dir);
    }

    #[test]
    fn test_list_files_empty() {
        let dir = test_dir("list_empty");
        let folder = SharedFolder::new(&dir).unwrap();

        let files = folder.list_files().unwrap();
        assert!(files.is_empty());

        cleanup(&dir);
    }

    #[test]
    fn test_list_files_with_content() {
        let dir = test_dir("list_content");
        let folder = SharedFolder::new(&dir).unwrap();

        folder.put_file("a.txt", b"aaa").unwrap();
        folder.put_file("b.txt", b"bbb").unwrap();

        let mut files = folder.list_files().unwrap();
        files.sort();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], PathBuf::from("a.txt"));
        assert_eq!(files[1], PathBuf::from("b.txt"));

        cleanup(&dir);
    }

    #[test]
    fn test_list_files_recursive() {
        let dir = test_dir("list_recursive");
        let folder = SharedFolder::new(&dir).unwrap();

        folder.put_file("root.txt", b"r").unwrap();
        folder.put_file("sub/child.txt", b"c").unwrap();
        folder.put_file("sub/deep/leaf.txt", b"l").unwrap();

        let mut files = folder.list_files().unwrap();
        files.sort();
        assert_eq!(files.len(), 3);

        let file_strs: Vec<String> = files.iter().map(|f| f.to_string_lossy().replace('\\', "/")).collect();
        assert!(file_strs.contains(&"root.txt".to_string()));
        assert!(file_strs.contains(&"sub/child.txt".to_string()));
        assert!(file_strs.contains(&"sub/deep/leaf.txt".to_string()));

        cleanup(&dir);
    }

    #[test]
    fn test_delete_file() {
        let dir = test_dir("delete");
        let folder = SharedFolder::new(&dir).unwrap();

        folder.put_file("to_delete.txt", b"bye").unwrap();
        assert!(folder.file_exists("to_delete.txt"));

        folder.delete_file("to_delete.txt").unwrap();
        assert!(!folder.file_exists("to_delete.txt"));

        // Deleting non-existent file should not error
        folder.delete_file("to_delete.txt").unwrap();

        cleanup(&dir);
    }

    #[test]
    fn test_file_exists() {
        let dir = test_dir("exists");
        let folder = SharedFolder::new(&dir).unwrap();

        assert!(!folder.file_exists("nope.txt"));

        folder.put_file("yes.txt", b"here").unwrap();
        assert!(folder.file_exists("yes.txt"));
        assert!(!folder.file_exists("nope.txt"));

        cleanup(&dir);
    }

    #[test]
    fn test_size_bytes() {
        let dir = test_dir("size");
        let folder = SharedFolder::new(&dir).unwrap();

        assert_eq!(folder.size_bytes().unwrap(), 0);

        folder.put_file("a.txt", b"12345").unwrap(); // 5 bytes
        folder.put_file("b.txt", b"1234567890").unwrap(); // 10 bytes

        assert_eq!(folder.size_bytes().unwrap(), 15);

        cleanup(&dir);
    }

    #[test]
    fn test_size_limit_enforcement() {
        let dir = test_dir("size_limit");
        let folder = SharedFolder::new(&dir).unwrap().with_max_size(20);

        folder.put_file("small.txt", b"0123456789").unwrap(); // 10 bytes, ok

        // This should fail: 10 existing + 15 new = 25 > 20
        let result = folder.put_file("big.txt", b"012345678901234");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("size limit exceeded"),
            "Expected size limit error, got: {}",
            err_msg
        );

        cleanup(&dir);
    }

    #[test]
    fn test_bind_mount_spec() {
        let dir = test_dir("bind_mount");
        let folder = SharedFolder::new(&dir).unwrap();

        let spec = folder.bind_mount_spec();
        assert!(spec.contains(":/workspace"), "spec={}", spec);
        assert!(
            spec.starts_with(&dir.display().to_string()),
            "spec={} should start with {}",
            spec,
            dir.display()
        );

        cleanup(&dir);
    }

    #[test]
    fn test_with_container_path() {
        let dir = test_dir("container_path");
        let folder = SharedFolder::new(&dir)
            .unwrap()
            .with_container_path("/app/data");

        assert_eq!(folder.container_path(), "/app/data");

        let spec = folder.bind_mount_spec();
        assert!(spec.ends_with(":/app/data"), "spec={}", spec);

        cleanup(&dir);
    }

    #[test]
    fn test_clear() {
        let dir = test_dir("clear");
        let folder = SharedFolder::new(&dir).unwrap();

        folder.put_file("a.txt", b"aaa").unwrap();
        folder.put_file("sub/b.txt", b"bbb").unwrap();
        assert_eq!(folder.list_files().unwrap().len(), 2);

        folder.clear().unwrap();

        let files = folder.list_files().unwrap();
        assert!(files.is_empty(), "Expected empty after clear, got {:?}", files);
        // The root directory itself should still exist
        assert!(dir.exists());

        cleanup(&dir);
    }

    #[test]
    #[cfg(feature = "cloud-connectors")]
    fn test_guess_content_type() {
        // Known types
        assert_eq!(
            SharedFolder::guess_content_type("doc.pdf"),
            Some("application/pdf".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("page.html"),
            Some("text/html".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("page.htm"),
            Some("text/html".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("data.json"),
            Some("application/json".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("notes.txt"),
            Some("text/plain".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("readme.md"),
            Some("text/plain".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("data.csv"),
            Some("text/plain".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("photo.png"),
            Some("image/png".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("photo.jpg"),
            Some("image/jpeg".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("photo.jpeg"),
            Some("image/jpeg".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("icon.svg"),
            Some("image/svg+xml".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("song.mp3"),
            Some("audio/mpeg".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("sound.wav"),
            Some("audio/wav".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("clip.ogg"),
            Some("audio/ogg".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("report.docx"),
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("sheet.xlsx"),
            Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".into())
        );
        assert_eq!(
            SharedFolder::guess_content_type("slides.pptx"),
            Some("application/vnd.openxmlformats-officedocument.presentationml.presentation".into())
        );

        // Unknown type
        assert_eq!(SharedFolder::guess_content_type("binary.xyz"), None);

        // Path with directories
        assert_eq!(
            SharedFolder::guess_content_type("dir/sub/file.json"),
            Some("application/json".into())
        );
    }

    #[test]
    fn test_temp_folder_cleanup_on_drop() {
        let path;
        {
            let folder = SharedFolder::temp().unwrap();
            path = folder.host_path().to_path_buf();
            assert!(path.exists(), "Temp folder should exist while alive");

            // Write a file so we verify the whole tree is removed
            folder.put_file("evidence.txt", b"I exist").unwrap();
            assert!(path.join("evidence.txt").exists());
        }
        // After drop, directory should be gone
        assert!(
            !path.exists(),
            "Temp folder {:?} should be cleaned up after drop",
            path
        );
    }

    #[test]
    #[cfg(feature = "cloud-connectors")]
    fn test_sync_to_cloud_mock() {
        let dir = test_dir("sync_to_cloud");
        let folder = SharedFolder::new(&dir).unwrap();

        folder.put_file("hello.txt", b"Hello").unwrap();
        folder.put_file("sub/data.json", b"{\"a\":1}").unwrap();

        let mock = MockCloudStorage::new();
        let count = folder.sync_to_cloud(&mock, "backup/test").unwrap();
        assert_eq!(count, 2);

        // Verify the mock received both files
        let files = mock.files.lock().unwrap();
        assert_eq!(files.len(), 2);

        // Check key format: prefix/relative_path
        assert!(
            files.contains_key("backup/test/hello.txt"),
            "Expected 'backup/test/hello.txt' in {:?}",
            files.keys().collect::<Vec<_>>()
        );

        // On Windows the path separator may be backslash in display(); check both
        let has_sub_data = files.contains_key("backup/test/sub/data.json")
            || files.contains_key("backup/test/sub\\data.json");
        assert!(
            has_sub_data,
            "Expected 'backup/test/sub/data.json' in {:?}",
            files.keys().collect::<Vec<_>>()
        );

        // Verify content
        assert_eq!(files.get("backup/test/hello.txt").unwrap(), b"Hello");

        cleanup(&dir);
    }

    #[test]
    #[cfg(feature = "cloud-connectors")]
    fn test_sync_from_cloud_mock() {
        let dir = test_dir("sync_from_cloud");
        let folder = SharedFolder::new(&dir).unwrap();

        // Seed the mock with files
        let mock = MockCloudStorage::new();
        {
            let mut files = mock.files.lock().unwrap();
            files.insert("backup/report.md".to_string(), b"# Report".to_vec());
            files.insert("backup/img/logo.png".to_string(), b"PNG_DATA".to_vec());
        }

        let count = folder.sync_from_cloud(&mock, "backup/").unwrap();
        assert_eq!(count, 2);

        // Verify local files were created
        assert_eq!(folder.get_file("report.md").unwrap(), b"# Report");
        assert_eq!(folder.get_file("img/logo.png").unwrap(), b"PNG_DATA");

        cleanup(&dir);
    }

    #[test]
    #[cfg(feature = "cloud-connectors")]
    fn test_sync_to_cloud_empty_folder() {
        let dir = test_dir("sync_empty");
        let folder = SharedFolder::new(&dir).unwrap();

        let mock = MockCloudStorage::new();
        let count = folder.sync_to_cloud(&mock, "prefix").unwrap();
        assert_eq!(count, 0);

        let files = mock.files.lock().unwrap();
        assert!(files.is_empty());

        cleanup(&dir);
    }

    #[test]
    fn test_new_existing_directory() {
        let dir = test_dir("existing");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pre.txt"), b"preexisting").unwrap();

        let folder = SharedFolder::new(&dir).unwrap();
        // Should not wipe existing content
        assert_eq!(folder.get_file("pre.txt").unwrap(), b"preexisting");

        cleanup(&dir);
    }

    #[test]
    fn test_default_container_path() {
        let dir = test_dir("default_cp");
        let folder = SharedFolder::new(&dir).unwrap();
        assert_eq!(folder.container_path(), "/workspace");
        cleanup(&dir);
    }
}
