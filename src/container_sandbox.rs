// container_sandbox.rs — Container-based code sandbox with Docker isolation.
//
// Provides ContainerSandbox (wraps ContainerExecutor for Docker-based execution)
// and ExecutionBackend (transparent fallback between container and process isolation).
//
// Requires the `containers` feature flag.
//
// The ContainerSandbox offers the same interface as CodeSandbox but executes
// code inside Docker containers for real process/filesystem/network isolation.

use std::collections::HashMap;
use std::time::Duration;

use crate::code_sandbox::{CodeSandbox, ExecutionResult, Language, SandboxConfig};
use crate::container_executor::{
    ContainerConfig, ContainerError, ContainerExecutor, CreateOptions,
};
use crate::shared_folder::SharedFolder;

// ============================================================================
// Helper: Language -> string key (Language lacks Hash+Eq)
// ============================================================================

fn language_key(lang: &Language) -> String {
    format!("{:?}", lang).to_lowercase()
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for container-based sandboxing.
#[derive(Debug, Clone)]
pub struct ContainerSandboxConfig {
    /// Base sandbox config (timeout, output limits, etc.)
    pub sandbox_config: SandboxConfig,
    /// Docker images per language (keyed by language debug name, lowercase)
    pub images: HashMap<String, String>,
    /// Reuse containers across executions (faster but less isolated)
    pub reuse_containers: bool,
    /// Shared folder for input/output files
    pub shared_folder_path: Option<std::path::PathBuf>,
}

impl ContainerSandboxConfig {
    /// Get the Docker image for a language.
    pub fn image_for(&self, lang: &Language) -> Option<&str> {
        self.images.get(&language_key(lang)).map(|s| s.as_str())
    }

    /// Set the Docker image for a language.
    pub fn set_image(&mut self, lang: &Language, image: String) {
        self.images.insert(language_key(lang), image);
    }
}

impl Default for ContainerSandboxConfig {
    fn default() -> Self {
        let mut images = HashMap::new();
        images.insert("python".into(), "python:3.12-slim".into());
        images.insert("javascript".into(), "node:20-slim".into());
        images.insert("bash".into(), "ubuntu:24.04".into());
        Self {
            sandbox_config: SandboxConfig::default(),
            images,
            reuse_containers: true,
            shared_folder_path: None,
        }
    }
}

// ============================================================================
// ContainerSandbox
// ============================================================================

/// Container-based code sandbox.
///
/// Executes code inside Docker containers for real process/filesystem/network
/// isolation. Falls back to process-based execution if Docker is unavailable.
pub struct ContainerSandbox {
    executor: ContainerExecutor,
    config: ContainerSandboxConfig,
    /// Reusable containers per language (keyed by language debug name)
    warm_containers: HashMap<String, String>,
    /// Shared folder for file I/O
    shared_folder: Option<SharedFolder>,
}

impl ContainerSandbox {
    /// Create a new ContainerSandbox.
    pub fn new(config: ContainerSandboxConfig) -> Result<Self, ContainerError> {
        let executor = ContainerExecutor::new(ContainerConfig::default())?;
        let shared_folder = config
            .shared_folder_path
            .as_ref()
            .and_then(|p| SharedFolder::new(p).ok());
        Ok(Self {
            executor,
            config,
            warm_containers: HashMap::new(),
            shared_folder,
        })
    }

    /// Execute code in a Docker container.
    /// Returns the same ExecutionResult as CodeSandbox for compatibility.
    pub fn execute(&mut self, language: &Language, code: &str) -> ExecutionResult {
        match self.execute_inner(language, code) {
            Ok(result) => result,
            Err(e) => ExecutionResult {
                stdout: String::new(),
                stderr: format!("Container execution failed: {}", e),
                exit_code: -1,
                duration: Duration::from_secs(0),
                timed_out: false,
                truncated: false,
            },
        }
    }

    fn execute_inner(
        &mut self,
        language: &Language,
        code: &str,
    ) -> Result<ExecutionResult, ContainerError> {
        let key = language_key(language);
        let image = self
            .config
            .images
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "ubuntu:24.04".into());

        let container_id = if self.config.reuse_containers {
            if let Some(id) = self.warm_containers.get(&key) {
                id.clone()
            } else {
                let id = self.create_sandbox_container(language, &image)?;
                self.warm_containers.insert(key.clone(), id.clone());
                id
            }
        } else {
            self.create_sandbox_container(language, &image)?
        };

        // Write code to a temp file inside the container via exec
        let ext = language.extension();
        let filename = format!("/tmp/sandbox_code.{}", ext);

        // Use printf to write the code file inside the container.
        // Escape the code for shell safety.
        let escaped_code = code.replace('\\', "\\\\").replace('\'', "'\\''");
        let write_cmd = format!("printf '%s' '{}' > {}", escaped_code, filename);

        let _ = self.executor.exec(
            &container_id,
            &["sh", "-c", &write_cmd],
            Duration::from_secs(5),
        )?;

        // Execute the code
        let interpreter = language.interpreter();
        let result = self.executor.exec(
            &container_id,
            &[interpreter, &filename],
            self.config.sandbox_config.timeout,
        )?;

        let mut stdout = result.stdout;
        let mut truncated = false;
        if stdout.len() > self.config.sandbox_config.max_output_bytes {
            stdout.truncate(self.config.sandbox_config.max_output_bytes);
            truncated = true;
        }

        let mut stderr = result.stderr;
        if stderr.len() > self.config.sandbox_config.max_output_bytes {
            stderr.truncate(self.config.sandbox_config.max_output_bytes);
        }

        // Cleanup if not reusing
        if !self.config.reuse_containers {
            let _ = self.executor.stop(&container_id, 5);
            let _ = self.executor.remove(&container_id, true);
        }

        Ok(ExecutionResult {
            stdout,
            stderr,
            exit_code: result.exit_code as i32,
            duration: result.duration,
            timed_out: result.timed_out,
            truncated,
        })
    }

    fn create_sandbox_container(
        &mut self,
        language: &Language,
        image: &str,
    ) -> Result<String, ContainerError> {
        let lang_abbrev = match language {
            Language::Python => "py",
            Language::JavaScript => "js",
            Language::Bash => "sh",
        };
        let name = format!(
            "ai_sandbox_{}_{}",
            lang_abbrev,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let mut opts = CreateOptions::default();

        // Add shared folder mount if configured
        if let Some(ref folder) = self.shared_folder {
            opts.bind_mounts.push((
                folder.host_path().to_string_lossy().into_owned(),
                "/workspace".into(),
            ));
        }

        // Keep container running with a sleep command
        opts.cmd = Some(vec!["sleep".into(), "3600".into()]);

        let id = self.executor.create(image, &name, opts)?;
        self.executor.start(&id)?;
        Ok(id)
    }

    /// Get the shared folder if configured.
    pub fn shared_folder(&self) -> Option<&SharedFolder> {
        self.shared_folder.as_ref()
    }

    /// Get the configuration.
    pub fn config(&self) -> &ContainerSandboxConfig {
        &self.config
    }

    /// Clean up all warm containers.
    pub fn cleanup(&mut self) {
        for (_, id) in self.warm_containers.drain() {
            let _ = self.executor.stop(&id, 5);
            let _ = self.executor.remove(&id, true);
        }
    }
}

impl Drop for ContainerSandbox {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ============================================================================
// ExecutionBackend — transparent fallback
// ============================================================================

/// Execution backend that automatically selects between container and process
/// isolation based on Docker availability.
pub enum ExecutionBackend {
    /// Docker container isolation (real isolation)
    Container(ContainerSandbox),
    /// Process-level isolation (fallback when Docker unavailable)
    Process(CodeSandbox),
}

impl ExecutionBackend {
    /// Auto-detect: use Container if Docker is available, else Process.
    pub fn auto() -> Self {
        Self::auto_with_config(ContainerSandboxConfig::default(), SandboxConfig::default())
    }

    /// Auto-detect with custom configs.
    pub fn auto_with_config(
        container_config: ContainerSandboxConfig,
        process_config: SandboxConfig,
    ) -> Self {
        if ContainerExecutor::is_docker_available() {
            match ContainerSandbox::new(container_config) {
                Ok(sandbox) => ExecutionBackend::Container(sandbox),
                Err(_) => ExecutionBackend::Process(CodeSandbox::with_config(process_config)),
            }
        } else {
            ExecutionBackend::Process(CodeSandbox::with_config(process_config))
        }
    }

    /// Force container backend.
    pub fn container(config: ContainerSandboxConfig) -> Result<Self, ContainerError> {
        Ok(ExecutionBackend::Container(ContainerSandbox::new(config)?))
    }

    /// Force process backend.
    pub fn process(config: SandboxConfig) -> Self {
        ExecutionBackend::Process(CodeSandbox::with_config(config))
    }

    /// Execute code using the selected backend.
    pub fn execute(&mut self, language: &Language, code: &str) -> ExecutionResult {
        match self {
            ExecutionBackend::Container(sandbox) => sandbox.execute(language, code),
            ExecutionBackend::Process(sandbox) => sandbox.execute(language, code),
        }
    }

    /// Check if the container backend is active.
    pub fn is_container(&self) -> bool {
        matches!(self, ExecutionBackend::Container(_))
    }

    /// Check if the process backend is active.
    pub fn is_process(&self) -> bool {
        matches!(self, ExecutionBackend::Process(_))
    }

    /// Get backend name as a string.
    pub fn backend_name(&self) -> &str {
        match self {
            ExecutionBackend::Container(_) => "container",
            ExecutionBackend::Process(_) => "process",
        }
    }
}

// ============================================================================
// Container name format helper (exposed for testing)
// ============================================================================

/// Build a sandbox container name from a language and timestamp.
/// Format: `ai_sandbox_{abbrev}_{timestamp_ms}`
pub fn sandbox_container_name(language: &Language) -> String {
    let lang_abbrev = match language {
        Language::Python => "py",
        Language::JavaScript => "js",
        Language::Bash => "sh",
    };
    format!(
        "ai_sandbox_{}_{}",
        lang_abbrev,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContainerSandboxConfig tests --

    #[test]
    fn test_container_sandbox_config_default() {
        let config = ContainerSandboxConfig::default();
        assert_eq!(config.sandbox_config.timeout, Duration::from_secs(30));
        assert_eq!(config.sandbox_config.max_output_bytes, 1_048_576);
        assert!(config.sandbox_config.capture_stderr);
        assert!(config.sandbox_config.detect_dangerous);
        assert!(config.reuse_containers);
        assert!(config.shared_folder_path.is_none());
        assert_eq!(config.images.len(), 3);
    }

    #[test]
    fn test_container_sandbox_config_images() {
        let config = ContainerSandboxConfig::default();
        assert_eq!(
            config.image_for(&Language::Python),
            Some("python:3.12-slim")
        );
        assert_eq!(
            config.image_for(&Language::JavaScript),
            Some("node:20-slim")
        );
        assert_eq!(config.image_for(&Language::Bash), Some("ubuntu:24.04"));
    }

    #[test]
    fn test_container_sandbox_config_reuse() {
        let mut config = ContainerSandboxConfig::default();
        assert!(config.reuse_containers);
        config.reuse_containers = false;
        assert!(!config.reuse_containers);
    }

    #[test]
    fn test_container_sandbox_config_clone() {
        let mut config = ContainerSandboxConfig::default();
        config.reuse_containers = false;
        config.shared_folder_path = Some(std::path::PathBuf::from("/tmp/test"));

        let cloned = config.clone();
        assert!(!cloned.reuse_containers);
        assert_eq!(
            cloned.shared_folder_path,
            Some(std::path::PathBuf::from("/tmp/test"))
        );
        assert_eq!(cloned.images.len(), config.images.len());
    }

    #[test]
    fn test_container_sandbox_config_shared_folder() {
        let mut config = ContainerSandboxConfig::default();
        assert!(config.shared_folder_path.is_none());

        let path = std::path::PathBuf::from("/home/user/shared");
        config.shared_folder_path = Some(path.clone());
        assert_eq!(config.shared_folder_path.as_ref().unwrap(), &path);
    }

    #[test]
    fn test_container_sandbox_config_set_image() {
        let mut config = ContainerSandboxConfig::default();
        config.set_image(&Language::Python, "python:3.13-slim".into());
        assert_eq!(
            config.image_for(&Language::Python),
            Some("python:3.13-slim")
        );
        // Other images unchanged
        assert_eq!(
            config.image_for(&Language::JavaScript),
            Some("node:20-slim")
        );
    }

    // -- Language image mapping --

    #[test]
    fn test_language_image_mapping() {
        let config = ContainerSandboxConfig::default();
        // Python -> python:3.12-slim
        assert!(config
            .image_for(&Language::Python)
            .unwrap()
            .starts_with("python:"));
        // JavaScript -> node:20-slim
        assert!(config
            .image_for(&Language::JavaScript)
            .unwrap()
            .starts_with("node:"));
        // Bash -> ubuntu:24.04
        assert!(config
            .image_for(&Language::Bash)
            .unwrap()
            .starts_with("ubuntu:"));
    }

    // -- ExecutionBackend tests (no Docker required) --

    #[test]
    fn test_execution_backend_process_fallback() {
        let backend = ExecutionBackend::process(SandboxConfig::default());
        assert!(backend.is_process());
        assert!(!backend.is_container());
    }

    #[test]
    fn test_execution_backend_name_process() {
        let backend = ExecutionBackend::process(SandboxConfig::default());
        assert_eq!(backend.backend_name(), "process");
    }

    #[test]
    fn test_execution_backend_name_container_fallback() {
        // auto() on a system without Docker should return process backend
        let backend = ExecutionBackend::auto();
        // We cannot guarantee Docker is available, so just check the name
        // is one of the valid values.
        let name = backend.backend_name();
        assert!(name == "container" || name == "process");
    }

    #[test]
    fn test_execution_backend_is_container() {
        // Process backend should not be container
        let backend = ExecutionBackend::process(SandboxConfig::default());
        assert!(!backend.is_container());
    }

    #[test]
    fn test_execution_backend_is_process() {
        let backend = ExecutionBackend::process(SandboxConfig::default());
        assert!(backend.is_process());
    }

    #[test]
    fn test_execution_backend_auto() {
        // auto() should return either container or process depending on Docker
        let backend = ExecutionBackend::auto();
        // Must be one or the other
        assert!(backend.is_container() || backend.is_process());
        // Name must match
        if backend.is_container() {
            assert_eq!(backend.backend_name(), "container");
        } else {
            assert_eq!(backend.backend_name(), "process");
        }
    }

    // -- ExecutionResult tests --

    #[test]
    fn test_execution_result_error_format() {
        // Verify the error ExecutionResult format used when container execution fails
        let result = ExecutionResult {
            stdout: String::new(),
            stderr: format!("Container execution failed: {}", "Docker not running"),
            exit_code: -1,
            duration: Duration::from_secs(0),
            timed_out: false,
            truncated: false,
        };
        assert_eq!(result.exit_code, -1);
        assert!(result.stderr.contains("Container execution failed"));
        assert!(result.stdout.is_empty());
        assert!(!result.timed_out);
        assert!(!result.truncated);
    }

    #[test]
    fn test_execution_result_defaults() {
        let result = ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            duration: Duration::ZERO,
            timed_out: false,
            truncated: false,
        };
        assert!(result.success());
        assert!(result.stdout.is_empty());
        assert!(result.stderr.is_empty());
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.duration, Duration::ZERO);
        assert!(!result.timed_out);
        assert!(!result.truncated);
    }

    // -- Container name format --

    #[test]
    fn test_sandbox_container_name_format() {
        let name_py = sandbox_container_name(&Language::Python);
        assert!(name_py.starts_with("ai_sandbox_py_"));

        let name_js = sandbox_container_name(&Language::JavaScript);
        assert!(name_js.starts_with("ai_sandbox_js_"));

        let name_sh = sandbox_container_name(&Language::Bash);
        assert!(name_sh.starts_with("ai_sandbox_sh_"));

        // Verify the suffix is a number (timestamp)
        let suffix = name_py.strip_prefix("ai_sandbox_py_").unwrap();
        assert!(suffix.parse::<u128>().is_ok());
    }

    // -- Process backend execution --

    #[test]
    fn test_execution_backend_process_execute() {
        let mut backend = ExecutionBackend::process(SandboxConfig::default());
        assert!(backend.is_process());
        // Execute safe code — the actual execution depends on the interpreter
        // being available, but the backend should not panic.
        let result = backend.execute(&Language::Bash, "echo hello");
        // On systems with bash, exit_code should be 0
        // On systems without bash, there will be an error in stderr
        // Either way, it should not panic.
        assert!(result.exit_code == 0 || !result.stderr.is_empty());
    }

    // -- language_key helper --

    #[test]
    fn test_language_key_values() {
        assert_eq!(language_key(&Language::Python), "python");
        assert_eq!(language_key(&Language::JavaScript), "javascript");
        assert_eq!(language_key(&Language::Bash), "bash");
    }
}
