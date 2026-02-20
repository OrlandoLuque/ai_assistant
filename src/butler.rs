//! Butler — auto-detection and configuration
//!
//! Detects the local environment and generates optimal configurations.

use crate::agent_policy::{AgentPolicyBuilder, InternetMode};
use crate::agent_profiles::AgentProfile;
use crate::config::{AiConfig, AiProvider};
use crate::mode_manager::OperationMode;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Detection types
// =============================================================================

/// Result of a single detector run.
pub struct DetectionResult {
    /// Whether the resource was detected.
    pub detected: bool,
    /// Key-value details about what was found.
    pub details: HashMap<String, String>,
    /// Optional suggested configuration string.
    pub suggested_config: Option<String>,
}

impl DetectionResult {
    /// Convenience constructor for a positive detection with details.
    pub fn found(details: HashMap<String, String>) -> Self {
        Self {
            detected: true,
            details,
            suggested_config: None,
        }
    }

    /// Convenience constructor for a negative detection.
    pub fn not_found() -> Self {
        Self {
            detected: false,
            details: HashMap::new(),
            suggested_config: None,
        }
    }
}

/// Trait implemented by each environment detector.
pub trait ResourceDetector: Send + Sync {
    /// Human-readable name of this detector (e.g. "ollama").
    fn name(&self) -> &str;
    /// Run the detection. May make network calls or spawn subprocesses with short timeouts.
    fn detect(&self) -> DetectionResult;
}

// =============================================================================
// Detected environment types
// =============================================================================

/// An LLM provider that was detected in the environment.
pub struct DetectedProvider {
    /// Human-readable name (e.g. "Ollama").
    pub name: String,
    /// Corresponding AiProvider variant.
    pub provider_type: AiProvider,
    /// Base URL for the provider.
    pub url: String,
    /// Models that are expected to be available (may be empty).
    pub available_models: Vec<String>,
}

/// The type of software project detected in the working directory.
pub enum ProjectType {
    Rust,
    Node,
    Python,
    Go,
    Java,
    DotNet,
    Ruby,
    Unknown,
}

/// Version control information.
pub struct VcsInfo {
    /// VCS type, currently always "git".
    pub vcs_type: String,
    /// Current branch name.
    pub branch: String,
    /// Whether the repository has any remotes configured.
    pub has_remotes: bool,
}

/// Runtime environment information.
pub struct RuntimeInfo {
    /// Operating system name.
    pub os: String,
    /// CPU architecture.
    pub arch: String,
    /// Number of logical CPUs.
    pub cpus: usize,
    /// Whether a GPU was detected.
    pub has_gpu: bool,
    /// Whether Docker tooling was detected.
    pub has_docker: bool,
    /// Whether a browser binary was detected.
    pub has_browser: bool,
}

/// Full environment report produced by a Butler scan.
pub struct EnvironmentReport {
    /// LLM providers that were detected.
    pub llm_providers: Vec<DetectedProvider>,
    /// The type of project in the working directory, if any.
    pub project_type: Option<ProjectType>,
    /// VCS information, if a repository was detected.
    pub vcs: Option<VcsInfo>,
    /// Runtime information (OS, CPU, GPU, etc.).
    pub runtime: RuntimeInfo,
    /// List of detected capability names.
    pub capabilities: Vec<String>,
    /// Suggested agent profile name.
    pub suggested_agent_profile: String,
    /// Suggested operation mode.
    pub suggested_mode: OperationMode,
}

// =============================================================================
// Built-in detectors
// =============================================================================

/// Detects Ollama by performing a real HTTP check against its API.
pub struct OllamaDetector {
    pub base_url: String,
}

impl OllamaDetector {
    pub fn new() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self { base_url }
    }
}

impl ResourceDetector for OllamaDetector {
    fn name(&self) -> &str {
        "ollama"
    }

    fn detect(&self) -> DetectionResult {
        let url = format!("{}/api/tags", self.base_url);
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                let mut models = Vec::new();
                if let Ok(body) = resp.into_string() {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(model_list) = val.get("models").and_then(|v| v.as_array()) {
                            for m in model_list {
                                if let Some(name) = m.get("name").and_then(|v| v.as_str()) {
                                    models.push(name.to_string());
                                }
                            }
                        }
                    }
                }
                let mut details = HashMap::new();
                details.insert("url".to_string(), self.base_url.clone());
                details.insert("model_count".to_string(), models.len().to_string());
                if !models.is_empty() {
                    details.insert("models".to_string(), models.join(", "));
                }
                DetectionResult {
                    detected: true,
                    details,
                    suggested_config: None,
                }
            }
            Err(_) => DetectionResult {
                detected: false,
                details: {
                    let mut d = HashMap::new();
                    d.insert("url".to_string(), self.base_url.clone());
                    d.insert(
                        "error".to_string(),
                        "Connection refused or timeout".to_string(),
                    );
                    d
                },
                suggested_config: None,
            },
        }
    }
}

/// Detects LM Studio by performing a real HTTP check against its API.
pub struct LmStudioDetector {
    pub base_url: String,
}

impl LmStudioDetector {
    pub fn new() -> Self {
        let base_url =
            std::env::var("LM_STUDIO_URL").unwrap_or_else(|_| "http://localhost:1234".to_string());
        Self { base_url }
    }
}

impl ResourceDetector for LmStudioDetector {
    fn name(&self) -> &str {
        "lm_studio"
    }

    fn detect(&self) -> DetectionResult {
        let url = format!("{}/v1/models", self.base_url);
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                let mut models = Vec::new();
                if let Ok(body) = resp.into_string() {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(data) = val.get("data").and_then(|v| v.as_array()) {
                            for m in data {
                                if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                                    models.push(id.to_string());
                                }
                            }
                        }
                    }
                }
                let mut details = HashMap::new();
                details.insert("url".to_string(), self.base_url.clone());
                details.insert("model_count".to_string(), models.len().to_string());
                if !models.is_empty() {
                    details.insert("models".to_string(), models.join(", "));
                }
                DetectionResult {
                    detected: true,
                    details,
                    suggested_config: None,
                }
            }
            Err(_) => DetectionResult {
                detected: false,
                details: {
                    let mut d = HashMap::new();
                    d.insert("url".to_string(), self.base_url.clone());
                    d.insert(
                        "error".to_string(),
                        "Connection refused or timeout".to_string(),
                    );
                    d
                },
                suggested_config: None,
            },
        }
    }
}

/// Detects cloud API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
pub struct CloudApiDetector;

impl ResourceDetector for CloudApiDetector {
    fn name(&self) -> &str {
        "cloud_api"
    }

    fn detect(&self) -> DetectionResult {
        let has_openai = std::env::var("OPENAI_API_KEY").is_ok();
        let has_anthropic = std::env::var("ANTHROPIC_API_KEY").is_ok();

        if !has_openai && !has_anthropic {
            return DetectionResult::not_found();
        }

        let mut details = HashMap::new();
        if has_openai {
            details.insert("openai".to_string(), "true".to_string());
        }
        if has_anthropic {
            details.insert("anthropic".to_string(), "true".to_string());
        }
        DetectionResult::found(details)
    }
}

/// Detects project type by looking for common manifest files.
pub struct ProjectTypeDetector {
    pub root: PathBuf,
}

impl ResourceDetector for ProjectTypeDetector {
    fn name(&self) -> &str {
        "project_type"
    }

    fn detect(&self) -> DetectionResult {
        let checks: &[(&str, &str)] = &[
            ("Cargo.toml", "rust"),
            ("package.json", "node"),
            ("requirements.txt", "python"),
            ("pyproject.toml", "python"),
            ("setup.py", "python"),
            ("go.mod", "go"),
            ("pom.xml", "java"),
            ("build.gradle", "java"),
            ("*.csproj", "dotnet"),
            ("*.sln", "dotnet"),
            ("Gemfile", "ruby"),
        ];

        for &(file, lang) in checks {
            if file.starts_with('*') {
                // Glob-style check: look for any file with that extension
                let ext = file.trim_start_matches('*');
                if has_file_with_extension(&self.root, ext) {
                    let mut details = HashMap::new();
                    details.insert("type".to_string(), lang.to_string());
                    details.insert("marker".to_string(), file.to_string());
                    return DetectionResult::found(details);
                }
            } else if self.root.join(file).exists() {
                let mut details = HashMap::new();
                details.insert("type".to_string(), lang.to_string());
                details.insert("marker".to_string(), file.to_string());
                return DetectionResult::found(details);
            }
        }

        // Unknown project type
        let mut details = HashMap::new();
        details.insert("type".to_string(), "unknown".to_string());
        DetectionResult {
            detected: true,
            details,
            suggested_config: None,
        }
    }
}

/// Detects Git by looking for a `.git` directory and reading `HEAD`.
pub struct GitDetector {
    pub root: PathBuf,
}

impl ResourceDetector for GitDetector {
    fn name(&self) -> &str {
        "git"
    }

    fn detect(&self) -> DetectionResult {
        let git_dir = self.root.join(".git");
        if !git_dir.is_dir() {
            return DetectionResult::not_found();
        }

        let mut details = HashMap::new();
        details.insert("vcs".to_string(), "git".to_string());

        // Read current branch from HEAD
        let head_path = git_dir.join("HEAD");
        if let Ok(content) = std::fs::read_to_string(&head_path) {
            let trimmed = content.trim();
            if let Some(branch) = trimmed.strip_prefix("ref: refs/heads/") {
                details.insert("branch".to_string(), branch.to_string());
            } else {
                details.insert("branch".to_string(), "detached".to_string());
            }
        }

        // Check for remotes directory
        let remotes_dir = git_dir.join("refs").join("remotes");
        let has_remotes = remotes_dir.is_dir()
            && std::fs::read_dir(&remotes_dir)
                .map(|mut d| d.next().is_some())
                .unwrap_or(false);
        details.insert("has_remotes".to_string(), has_remotes.to_string());

        DetectionResult::found(details)
    }
}

/// Detects Docker by checking if the docker daemon is running.
pub struct DockerDetector;

impl ResourceDetector for DockerDetector {
    fn name(&self) -> &str {
        "docker"
    }

    fn detect(&self) -> DetectionResult {
        let mut details = HashMap::new();

        // Check if docker command exists and daemon is running
        match std::process::Command::new("docker")
            .args(["info", "--format", "{{.ServerVersion}}"])
            .output()
        {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                details.insert("docker_version".to_string(), version);
                details.insert("daemon_running".to_string(), "true".to_string());

                // Check for Dockerfile
                if std::path::Path::new("Dockerfile").exists()
                    || std::path::Path::new("docker-compose.yml").exists()
                    || std::path::Path::new("docker-compose.yaml").exists()
                {
                    details.insert("has_dockerfile".to_string(), "true".to_string());
                }

                DetectionResult {
                    detected: true,
                    details,
                    suggested_config: None,
                }
            }
            Ok(output) => {
                // docker exists but daemon not running
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                if stderr.contains("Cannot connect") || stderr.contains("not running") {
                    details.insert("docker_installed".to_string(), "true".to_string());
                    details.insert("daemon_running".to_string(), "false".to_string());
                }
                DetectionResult {
                    detected: false,
                    details,
                    suggested_config: None,
                }
            }
            Err(_) => {
                // docker not installed
                DetectionResult {
                    detected: false,
                    details,
                    suggested_config: None,
                }
            }
        }
    }
}

/// Detects GPU availability via subprocess checks and env vars.
pub struct GpuDetector;

impl ResourceDetector for GpuDetector {
    fn name(&self) -> &str {
        "gpu"
    }

    fn detect(&self) -> DetectionResult {
        let mut details = HashMap::new();

        // Try nvidia-smi
        #[cfg(not(target_os = "macos"))]
        {
            match std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
                .output()
            {
                Ok(output) if output.status.success() => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let gpus: Vec<&str> = stdout.trim().lines().collect();
                    details.insert("gpu_type".to_string(), "nvidia".to_string());
                    details.insert("gpu_count".to_string(), gpus.len().to_string());
                    if let Some(first) = gpus.first() {
                        details.insert("gpu_info".to_string(), first.to_string());
                    }
                    return DetectionResult {
                        detected: true,
                        details,
                        suggested_config: None,
                    };
                }
                _ => {}
            }
        }

        // Check CUDA env vars as fallback
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || std::env::var("CUDA_HOME").is_ok() {
            details.insert("gpu_type".to_string(), "nvidia (env vars only)".to_string());
            return DetectionResult {
                detected: true,
                details,
                suggested_config: None,
            };
        }

        // macOS: check for Apple Silicon
        #[cfg(target_os = "macos")]
        {
            match std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
            {
                Ok(output) if output.status.success() => {
                    let cpu = String::from_utf8_lossy(&output.stdout).to_string();
                    if cpu.contains("Apple") {
                        details.insert("gpu_type".to_string(), "apple_silicon".to_string());
                        details.insert("gpu_info".to_string(), cpu.trim().to_string());
                        return DetectionResult {
                            detected: true,
                            details,
                            suggested_config: None,
                        };
                    }
                }
                _ => {}
            }
        }

        DetectionResult {
            detected: false,
            details,
            suggested_config: None,
        }
    }
}

/// Detects browser availability by checking env vars and common installation paths.
pub struct BrowserDetector;

impl ResourceDetector for BrowserDetector {
    fn name(&self) -> &str {
        "browser"
    }

    fn detect(&self) -> DetectionResult {
        let mut details = HashMap::new();

        // Check env vars first
        for var in ["CHROME_BIN", "CHROMIUM_BIN"] {
            if let Ok(path) = std::env::var(var) {
                let p = std::path::PathBuf::from(&path);
                if p.exists() {
                    details.insert("browser_path".to_string(), path);
                    details.insert("source".to_string(), format!("${}", var));
                    return DetectionResult {
                        detected: true,
                        details,
                        suggested_config: None,
                    };
                }
            }
        }

        // Check common installation paths
        let candidates: Vec<&str> = {
            #[cfg(target_os = "windows")]
            {
                vec![
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                ]
            }
            #[cfg(target_os = "macos")]
            {
                vec![
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    "/Applications/Chromium.app/Contents/MacOS/Chromium",
                ]
            }
            #[cfg(target_os = "linux")]
            {
                vec![
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium-browser",
                    "/usr/bin/chromium",
                ]
            }
            #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
            {
                vec![]
            }
        };

        for candidate in candidates {
            if std::path::Path::new(candidate).exists() {
                details.insert("browser_path".to_string(), candidate.to_string());
                details.insert("source".to_string(), "system path".to_string());
                return DetectionResult {
                    detected: true,
                    details,
                    suggested_config: None,
                };
            }
        }

        DetectionResult {
            detected: false,
            details,
            suggested_config: None,
        }
    }
}

/// Detects internet connectivity by performing real HTTP checks.
pub struct NetworkDetector;

impl ResourceDetector for NetworkDetector {
    fn name(&self) -> &str {
        "network"
    }

    fn detect(&self) -> DetectionResult {
        let mut details = HashMap::new();

        // Check proxy env vars
        for var in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "NO_PROXY",
            "no_proxy",
        ] {
            if let Ok(val) = std::env::var(var) {
                details.insert(format!("proxy_{}", var.to_lowercase()), val);
            }
        }

        // Real connectivity test — try multiple endpoints
        let test_urls = [
            "https://httpbin.org/get",
            "https://api.github.com",
            "https://www.google.com",
        ];

        let mut connected = false;
        for url in &test_urls {
            match ureq::get(url)
                .timeout(std::time::Duration::from_secs(3))
                .call()
            {
                Ok(resp) => {
                    details.insert("internet".to_string(), "available".to_string());
                    details.insert("test_url".to_string(), url.to_string());
                    details.insert("status".to_string(), resp.status().to_string());
                    connected = true;
                    break;
                }
                Err(_) => continue,
            }
        }

        if !connected {
            details.insert("internet".to_string(), "unavailable".to_string());
        }

        DetectionResult {
            detected: connected,
            details,
            suggested_config: None,
        }
    }
}

// =============================================================================
// Butler
// =============================================================================

/// The Butler: auto-detects the local environment and suggests optimal configuration.
pub struct Butler {
    detectors: Vec<Box<dyn ResourceDetector>>,
    cache: HashMap<String, DetectionResult>,
}

impl Butler {
    /// Create a new Butler with all built-in detectors pointing to the current directory.
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::with_root(cwd)
    }

    /// Create a new Butler with all built-in detectors pointing to the given root directory.
    pub fn with_root(root: PathBuf) -> Self {
        let detectors: Vec<Box<dyn ResourceDetector>> = vec![
            Box::new(OllamaDetector::new()),
            Box::new(LmStudioDetector::new()),
            Box::new(CloudApiDetector),
            Box::new(ProjectTypeDetector { root: root.clone() }),
            Box::new(GitDetector { root: root.clone() }),
            Box::new(DockerDetector),
            Box::new(GpuDetector),
            Box::new(BrowserDetector),
            Box::new(NetworkDetector),
        ];
        Self {
            detectors,
            cache: HashMap::new(),
        }
    }

    /// Add a custom detector.
    pub fn add_detector(&mut self, d: Box<dyn ResourceDetector>) {
        self.detectors.push(d);
    }

    /// Run all detectors, cache results, and produce an `EnvironmentReport`.
    pub fn scan(&mut self) -> EnvironmentReport {
        // Run every detector and cache by name
        self.cache.clear();
        for detector in &self.detectors {
            let result = detector.detect();
            self.cache.insert(detector.name().to_string(), result);
        }

        // Build report from cached results
        let mut llm_providers = Vec::new();
        let mut capabilities = Vec::new();

        // Ollama
        if let Some(r) = self.cache.get("ollama") {
            if r.detected {
                let url = r
                    .details
                    .get("url")
                    .cloned()
                    .unwrap_or_else(|| "http://localhost:11434".to_string());
                llm_providers.push(DetectedProvider {
                    name: "Ollama".to_string(),
                    provider_type: AiProvider::Ollama,
                    url,
                    available_models: r
                        .details
                        .get("models")
                        .map(|m| m.split(", ").map(|s| s.to_string()).collect())
                        .unwrap_or_default(),
                });
                capabilities.push("ollama".to_string());
            }
        }

        // LM Studio
        if let Some(r) = self.cache.get("lm_studio") {
            if r.detected {
                let url = r
                    .details
                    .get("url")
                    .cloned()
                    .unwrap_or_else(|| "http://localhost:1234".to_string());
                llm_providers.push(DetectedProvider {
                    name: "LM Studio".to_string(),
                    provider_type: AiProvider::LMStudio,
                    url,
                    available_models: r
                        .details
                        .get("models")
                        .map(|m| m.split(", ").map(|s| s.to_string()).collect())
                        .unwrap_or_default(),
                });
                capabilities.push("lm_studio".to_string());
            }
        }

        // Cloud APIs
        if let Some(r) = self.cache.get("cloud_api") {
            if r.detected {
                if r.details
                    .get("anthropic")
                    .map(|v| v == "true")
                    .unwrap_or(false)
                {
                    llm_providers.push(DetectedProvider {
                        name: "Anthropic".to_string(),
                        provider_type: AiProvider::Anthropic,
                        url: "https://api.anthropic.com".to_string(),
                        available_models: Vec::new(),
                    });
                    capabilities.push("anthropic_api".to_string());
                }
                if r.details
                    .get("openai")
                    .map(|v| v == "true")
                    .unwrap_or(false)
                {
                    llm_providers.push(DetectedProvider {
                        name: "OpenAI".to_string(),
                        provider_type: AiProvider::OpenAI,
                        url: "https://api.openai.com".to_string(),
                        available_models: Vec::new(),
                    });
                    capabilities.push("openai_api".to_string());
                }
            }
        }

        // Project type
        let project_type = if let Some(r) = self.cache.get("project_type") {
            if r.detected {
                let pt = r
                    .details
                    .get("type")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                Some(parse_project_type(pt))
            } else {
                None
            }
        } else {
            None
        };

        if project_type.is_some() {
            capabilities.push("project".to_string());
        }

        // VCS
        let vcs = if let Some(r) = self.cache.get("git") {
            if r.detected {
                let branch = r.details.get("branch").cloned().unwrap_or_default();
                let has_remotes = r
                    .details
                    .get("has_remotes")
                    .map(|v| v == "true")
                    .unwrap_or(false);
                capabilities.push("git".to_string());
                Some(VcsInfo {
                    vcs_type: "git".to_string(),
                    branch,
                    has_remotes,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Docker
        let has_docker = self
            .cache
            .get("docker")
            .map(|r| r.detected)
            .unwrap_or(false);
        if has_docker {
            capabilities.push("docker".to_string());
        }

        // GPU
        let has_gpu = self.cache.get("gpu").map(|r| r.detected).unwrap_or(false);
        if has_gpu {
            capabilities.push("gpu".to_string());
        }

        // Browser
        let has_browser = self
            .cache
            .get("browser")
            .map(|r| r.detected)
            .unwrap_or(false);
        if has_browser {
            capabilities.push("browser".to_string());
        }

        // Network
        if let Some(r) = self.cache.get("network") {
            if r.detected {
                capabilities.push("network".to_string());
            }
        }

        let runtime = RuntimeInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            has_gpu,
            has_docker,
            has_browser,
        };

        // Suggest profile and mode
        let (suggested_agent_profile, suggested_mode) = suggest_profile_and_mode(&project_type);

        EnvironmentReport {
            llm_providers,
            project_type,
            vcs,
            runtime,
            capabilities,
            suggested_agent_profile,
            suggested_mode,
        }
    }

    /// Retrieve the cached result for a detector by name (available after `scan`).
    pub fn cached_result(&self, detector_name: &str) -> Option<&DetectionResult> {
        self.cache.get(detector_name)
    }

    /// Suggest an `AiConfig` based on the environment report.
    pub fn suggest_config(&self, report: &EnvironmentReport) -> AiConfig {
        let mut config = AiConfig::default();

        // Priority: Anthropic > OpenAI > Ollama > LMStudio > default
        for provider in &report.llm_providers {
            match &provider.provider_type {
                AiProvider::Anthropic => {
                    config.provider = AiProvider::Anthropic;
                    return config;
                }
                _ => {}
            }
        }
        for provider in &report.llm_providers {
            match &provider.provider_type {
                AiProvider::OpenAI => {
                    config.provider = AiProvider::OpenAI;
                    return config;
                }
                _ => {}
            }
        }
        for provider in &report.llm_providers {
            match &provider.provider_type {
                AiProvider::Ollama => {
                    config.provider = AiProvider::Ollama;
                    config.ollama_url = provider.url.clone();
                    return config;
                }
                _ => {}
            }
        }
        for provider in &report.llm_providers {
            match &provider.provider_type {
                AiProvider::LMStudio => {
                    config.provider = AiProvider::LMStudio;
                    config.lm_studio_url = provider.url.clone();
                    return config;
                }
                _ => {}
            }
        }

        config
    }

    /// Suggest an `AgentProfile` based on the environment report.
    pub fn suggest_agent_profile(&self, report: &EnvironmentReport) -> AgentProfile {
        let has_project = report.project_type.is_some()
            && !matches!(report.project_type.as_ref(), Some(ProjectType::Unknown));

        if has_project {
            // Coding assistant profile
            let mut builder = AgentPolicyBuilder::new()
                .allow_path(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
                .max_iterations(100)
                .max_cost(2.0);

            if report.capabilities.contains(&"network".to_string()) {
                builder = builder.internet(InternetMode::SearchOnly);
            }

            AgentProfile {
                name: "coding-assistant".to_string(),
                description: "Auto-detected coding assistant for local project".to_string(),
                policy: builder.build(),
                model: None,
                system_prompt: Some(
                    "You are a coding assistant. Help the user write, debug, and refactor code."
                        .to_string(),
                ),
                tools: vec![
                    "read_file".to_string(),
                    "write_file".to_string(),
                    "run_command".to_string(),
                ],
                mcp_servers: Vec::new(),
                mode: OperationMode::Programming,
                tags: vec!["coding".to_string(), "auto-detected".to_string()],
            }
        } else {
            // Research agent profile
            let mut builder = AgentPolicyBuilder::new().max_iterations(80).max_cost(3.0);

            if report.capabilities.contains(&"network".to_string()) {
                builder = builder.internet(InternetMode::FullAccess);
            }

            AgentProfile {
                name: "research-agent".to_string(),
                description: "Auto-detected research agent (no project found)".to_string(),
                policy: builder.build(),
                model: None,
                system_prompt: Some(
                    "You are a research agent. Search the web, read documents, and synthesize information."
                        .to_string(),
                ),
                tools: vec![
                    "web_search".to_string(),
                    "web_fetch".to_string(),
                    "read_file".to_string(),
                ],
                mcp_servers: Vec::new(),
                mode: OperationMode::Assistant,
                tags: vec!["research".to_string(), "auto-detected".to_string()],
            }
        }
    }
}

impl Default for Butler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn parse_project_type(s: &str) -> ProjectType {
    match s {
        "rust" => ProjectType::Rust,
        "node" => ProjectType::Node,
        "python" => ProjectType::Python,
        "go" => ProjectType::Go,
        "java" => ProjectType::Java,
        "dotnet" => ProjectType::DotNet,
        "ruby" => ProjectType::Ruby,
        _ => ProjectType::Unknown,
    }
}

fn suggest_profile_and_mode(project_type: &Option<ProjectType>) -> (String, OperationMode) {
    match project_type {
        Some(pt) => match pt {
            ProjectType::Unknown => ("research-agent".to_string(), OperationMode::Assistant),
            _ => ("coding-assistant".to_string(), OperationMode::Programming),
        },
        None => ("research-agent".to_string(), OperationMode::Assistant),
    }
}

/// Check if any file in a directory has the given extension (non-recursive, shallow).
fn has_file_with_extension(dir: &Path, ext: &str) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(ext) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_ollama_detector() {
        let detector = OllamaDetector::new();
        assert_eq!(detector.name(), "ollama");
        let result = detector.detect();
        // With real HTTP check, detection depends on whether Ollama is running
        assert!(result.details.contains_key("url"));
    }

    #[test]
    fn test_cloud_api_no_keys() {
        // Remove the env vars if they happen to be set, then test.
        // We cannot truly guarantee the env, so we test the detector logic
        // by constructing a CloudApiDetector and checking when both vars are absent.
        let detector = CloudApiDetector;
        let result = detector.detect();
        // This test may pass as detected=true if the CI has these keys.
        // At minimum, verify the structure is correct.
        if !result.detected {
            assert!(result.details.is_empty());
        } else {
            assert!(
                result.details.contains_key("openai") || result.details.contains_key("anthropic")
            );
        }
    }

    #[test]
    fn test_project_type_detector_rust() {
        let tmp = std::env::temp_dir().join("butler_test_rust_project");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

        let detector = ProjectTypeDetector { root: tmp.clone() };
        let result = detector.detect();
        assert!(result.detected);
        assert_eq!(result.details.get("type").map(|s| s.as_str()), Some("rust"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_git_detector_no_git() {
        let tmp = std::env::temp_dir().join("butler_test_no_git");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let detector = GitDetector { root: tmp.clone() };
        let result = detector.detect();
        assert!(!result.detected);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_project_type_unknown() {
        let tmp = std::env::temp_dir().join("butler_test_unknown_project");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let detector = ProjectTypeDetector { root: tmp.clone() };
        let result = detector.detect();
        assert!(result.detected);
        assert_eq!(
            result.details.get("type").map(|s| s.as_str()),
            Some("unknown")
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_butler_new() {
        let butler = Butler::new();
        // Should have 9 built-in detectors
        assert_eq!(butler.detectors.len(), 9);
        assert!(butler.cache.is_empty());
    }

    #[test]
    fn test_butler_scan() {
        let tmp = std::env::temp_dir().join("butler_test_scan");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let mut butler = Butler::with_root(tmp.clone());
        let report = butler.scan();

        // With real checks, providers may or may not be detected
        // Runtime info should be populated
        assert!(!report.runtime.os.is_empty());
        assert!(!report.runtime.arch.is_empty());
        assert!(report.runtime.cpus >= 1);
        // Suggested values should be set
        assert!(!report.suggested_agent_profile.is_empty());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_suggest_config_default() {
        let butler = Butler::new();
        // Empty report — no providers detected
        let report = EnvironmentReport {
            llm_providers: Vec::new(),
            project_type: None,
            vcs: None,
            runtime: RuntimeInfo {
                os: "test".to_string(),
                arch: "x86_64".to_string(),
                cpus: 4,
                has_gpu: false,
                has_docker: false,
                has_browser: false,
            },
            capabilities: Vec::new(),
            suggested_agent_profile: "research-agent".to_string(),
            suggested_mode: OperationMode::Assistant,
        };

        let config = butler.suggest_config(&report);
        // Default should be Ollama
        assert_eq!(config.provider, AiProvider::Ollama);
    }

    #[test]
    fn test_suggest_config_with_provider() {
        let butler = Butler::new();
        let report = EnvironmentReport {
            llm_providers: vec![DetectedProvider {
                name: "Anthropic".to_string(),
                provider_type: AiProvider::Anthropic,
                url: "https://api.anthropic.com".to_string(),
                available_models: Vec::new(),
            }],
            project_type: None,
            vcs: None,
            runtime: RuntimeInfo {
                os: "test".to_string(),
                arch: "x86_64".to_string(),
                cpus: 4,
                has_gpu: false,
                has_docker: false,
                has_browser: false,
            },
            capabilities: Vec::new(),
            suggested_agent_profile: "research-agent".to_string(),
            suggested_mode: OperationMode::Assistant,
        };

        let config = butler.suggest_config(&report);
        assert_eq!(config.provider, AiProvider::Anthropic);
    }

    #[test]
    fn test_suggest_agent_profile_coding() {
        let butler = Butler::new();
        let report = EnvironmentReport {
            llm_providers: Vec::new(),
            project_type: Some(ProjectType::Rust),
            vcs: None,
            runtime: RuntimeInfo {
                os: "test".to_string(),
                arch: "x86_64".to_string(),
                cpus: 4,
                has_gpu: false,
                has_docker: false,
                has_browser: false,
            },
            capabilities: Vec::new(),
            suggested_agent_profile: "coding-assistant".to_string(),
            suggested_mode: OperationMode::Programming,
        };

        let profile = butler.suggest_agent_profile(&report);
        assert_eq!(profile.name, "coding-assistant");
        assert_eq!(profile.mode, OperationMode::Programming);
    }

    #[test]
    fn test_suggest_agent_profile_research() {
        let butler = Butler::new();
        let report = EnvironmentReport {
            llm_providers: Vec::new(),
            project_type: None,
            vcs: None,
            runtime: RuntimeInfo {
                os: "test".to_string(),
                arch: "x86_64".to_string(),
                cpus: 4,
                has_gpu: false,
                has_docker: false,
                has_browser: false,
            },
            capabilities: Vec::new(),
            suggested_agent_profile: "research-agent".to_string(),
            suggested_mode: OperationMode::Assistant,
        };

        let profile = butler.suggest_agent_profile(&report);
        assert_eq!(profile.name, "research-agent");
        assert_eq!(profile.mode, OperationMode::Assistant);
    }

    #[test]
    fn test_cached_result() {
        let tmp = std::env::temp_dir().join("butler_test_cache");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let mut butler = Butler::with_root(tmp.clone());
        // Before scan, cache is empty
        assert!(butler.cached_result("ollama").is_none());

        butler.scan();

        // After scan, cache should contain results for all detectors
        assert!(butler.cached_result("ollama").is_some());
        assert!(butler.cached_result("lm_studio").is_some());
        assert!(butler.cached_result("cloud_api").is_some());
        assert!(butler.cached_result("project_type").is_some());
        assert!(butler.cached_result("git").is_some());
        assert!(butler.cached_result("docker").is_some());
        assert!(butler.cached_result("gpu").is_some());
        assert!(butler.cached_result("browser").is_some());
        assert!(butler.cached_result("network").is_some());
        // Non-existent detector should return None
        assert!(butler.cached_result("nonexistent").is_none());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_ollama_detector_real_check() {
        // This tests that the detector doesn't crash and returns a valid result
        // (will return detected=false if Ollama isn't running)
        let detector = OllamaDetector::new();
        let result = detector.detect();
        // Should have URL in details regardless
        assert!(result.details.contains_key("url"));
        if !result.detected {
            assert!(result.details.contains_key("error"));
        }
    }

    #[test]
    fn test_lm_studio_detector_real_check() {
        let detector = LmStudioDetector::new();
        let result = detector.detect();
        assert!(result.details.contains_key("url"));
        if !result.detected {
            assert!(result.details.contains_key("error"));
        }
    }

    #[test]
    fn test_gpu_detector_doesnt_crash() {
        let detector = GpuDetector;
        let result = detector.detect();
        // Should complete without panicking
        let _ = result.detected;
    }

    #[test]
    fn test_browser_detector_checks_paths() {
        let detector = BrowserDetector;
        let result = detector.detect();
        // Should complete and either find or not find a browser
        if result.detected {
            assert!(result.details.contains_key("browser_path"));
        }
    }

    #[test]
    fn test_docker_detector_real_check() {
        let detector = DockerDetector;
        let result = detector.detect();
        // If detected, should have version info
        if result.detected {
            assert!(result.details.contains_key("docker_version"));
        }
    }

    #[test]
    fn test_network_detector_real_check() {
        // This test might be slow if no internet; that's OK
        let detector = NetworkDetector;
        let result = detector.detect();
        if result.detected {
            assert!(result
                .details
                .get("internet")
                .map(|v| v == "available")
                .unwrap_or(false));
        } else {
            assert!(result
                .details
                .get("internet")
                .map(|v| v == "unavailable")
                .unwrap_or(false));
        }
    }
}
