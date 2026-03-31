// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! ai_setup_gui — Desktop GUI for AI Assistant setup and management.
//!
//! Run: `cargo run --bin ai_setup_gui --features gui`
//!
//! Provides a graphical interface for:
//! - Environment prerequisite scanning and installation guidance
//! - Configuration editing with validation and import/export
//! - Server node lifecycle management (start/stop/status)
//! - Docker container orchestration
//! - Ollama model management (list/pull/delete)
//! - Backup and restore of configuration data

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use eframe::egui::{self, Color32, RichText, Ui};

use ai_assistant::config_file::{default_config_path, ConfigFile};
use ai_assistant::setup::{backup, config_ops, docker_ops, node_manager, prereq};

// =============================================================================
// Types
// =============================================================================

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Setup,
    Config,
    Nodes,
    Docker,
    Models,
    Backup,
}

impl Tab {
    fn label(&self) -> &'static str {
        match self {
            Self::Setup => "Setup",
            Self::Config => "Config",
            Self::Nodes => "Nodes",
            Self::Docker => "Docker",
            Self::Models => "Models",
            Self::Backup => "Backup",
        }
    }

    fn icon(&self) -> &'static str {
        match self {
            Self::Setup => "\u{2699}",   // gear
            Self::Config => "\u{1f4dd}", // memo
            Self::Nodes => "\u{1f5a5}", // desktop computer
            Self::Docker => "\u{1f4e6}", // package
            Self::Models => "\u{1f9e0}", // brain
            Self::Backup => "\u{1f4be}", // floppy disk
        }
    }

    fn all() -> &'static [Tab] {
        &[
            Tab::Setup,
            Tab::Config,
            Tab::Nodes,
            Tab::Docker,
            Tab::Models,
            Tab::Backup,
        ]
    }
}

/// Information about an Ollama model (from `ollama list`).
#[derive(Debug, Clone)]
struct OllamaModelEntry {
    name: String,
    size: String,
    modified: String,
}

// =============================================================================
// Background task messages
// =============================================================================

enum BgMessage {
    PrereqsDone(Vec<prereq::PrereqStatus>),
    NodeStatusDone(Result<node_manager::NodeInfo, String>),
    DockerStatusDone(Result<Vec<docker_ops::ContainerStatus>, String>),
    ModelListDone(Result<Vec<OllamaModelEntry>, String>),
    ModelPullDone(Result<String, String>),
    ModelDeleteDone(Result<String, String>),
    BackupDone(Result<backup::BackupInfo, String>),
    RestoreDone(Result<(), String>),
    ConfigValidated(Vec<String>),
    DockerBuildDone(Result<String, String>),
    DockerComposeUpDone(Result<String, String>),
    DockerComposeDownDone(Result<String, String>),
    NodeStartDone(Result<String, String>),
    NodeStopDone(Result<(), String>),
}

// =============================================================================
// Application state
// =============================================================================

struct SetupGuiApp {
    tab: Tab,

    // Background channel
    bg_tx: mpsc::Sender<BgMessage>,
    bg_rx: mpsc::Receiver<BgMessage>,

    // Setup tab
    prereqs: Vec<prereq::PrereqStatus>,
    scan_done: bool,
    scanning: bool,
    install_info: Option<(String, prereq::InstallInstructions)>,

    // Config tab
    config_text: String,
    config_path: String,
    config_modified: bool,
    validation_errors: Vec<String>,
    config_loaded: bool,
    show_api_keys: bool,
    // Editable config fields (section-based)
    cfg_provider_type: String,
    cfg_provider_model: String,
    cfg_provider_api_key: String,
    cfg_url_ollama: String,
    cfg_url_lm_studio: String,
    cfg_temperature: f32,
    cfg_max_history: u32,
    cfg_rag_enabled: bool,
    cfg_rag_knowledge_tokens: u32,
    cfg_rag_conversation_tokens: u32,
    cfg_log_level: String,
    cfg_cache_enabled: bool,
    cfg_cache_max_entries: u32,

    // Nodes tab
    node_info: Option<node_manager::NodeInfo>,
    node_log: String,
    node_refreshing: bool,
    last_node_refresh: Instant,

    // Docker tab
    containers: Vec<docker_ops::ContainerStatus>,
    docker_log: String,
    docker_available: Option<bool>,
    docker_profiles: Vec<(String, bool)>,
    docker_build_features: String,
    docker_refreshing: bool,

    // Models tab
    models: Vec<OllamaModelEntry>,
    model_pull_name: String,
    model_pulling: bool,
    model_deleting: Option<String>,
    models_loaded: bool,

    // Backup tab
    backup_info: Option<backup::BackupInfo>,
    backup_include_models: bool,
    backup_output_path: String,
    restore_archive_path: String,

    // General
    status_message: String,
    status_is_error: bool,
    status_time: Instant,
}

impl SetupGuiApp {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let config_path = default_config_path();

        Self {
            tab: Tab::Setup,
            bg_tx: tx,
            bg_rx: rx,

            // Setup
            prereqs: Vec::new(),
            scan_done: false,
            scanning: false,
            install_info: None,

            // Config
            config_text: String::new(),
            config_path: config_path.display().to_string(),
            config_modified: false,
            validation_errors: Vec::new(),
            config_loaded: false,
            show_api_keys: false,
            cfg_provider_type: "ollama".to_string(),
            cfg_provider_model: "llama3".to_string(),
            cfg_provider_api_key: String::new(),
            cfg_url_ollama: "http://localhost:11434".to_string(),
            cfg_url_lm_studio: "http://localhost:1234".to_string(),
            cfg_temperature: 0.7,
            cfg_max_history: 20,
            cfg_rag_enabled: true,
            cfg_rag_knowledge_tokens: 2000,
            cfg_rag_conversation_tokens: 1500,
            cfg_log_level: "info".to_string(),
            cfg_cache_enabled: true,
            cfg_cache_max_entries: 1000,

            // Nodes
            node_info: None,
            node_log: String::new(),
            node_refreshing: false,
            last_node_refresh: Instant::now(),

            // Docker
            containers: Vec::new(),
            docker_log: String::new(),
            docker_available: None,
            docker_profiles: vec![
                ("redis".to_string(), false),
                ("pgvector".to_string(), false),
            ],
            docker_build_features: "full".to_string(),
            docker_refreshing: false,

            // Models
            models: Vec::new(),
            model_pull_name: String::new(),
            model_pulling: false,
            model_deleting: None,
            models_loaded: false,

            // Backup
            backup_info: None,
            backup_include_models: false,
            backup_output_path: String::new(),
            restore_archive_path: String::new(),

            // General
            status_message: String::new(),
            status_is_error: false,
            status_time: Instant::now(),
        }
    }

    // =========================================================================
    // Background task polling
    // =========================================================================

    fn poll_background(&mut self) {
        while let Ok(msg) = self.bg_rx.try_recv() {
            match msg {
                BgMessage::PrereqsDone(results) => {
                    self.prereqs = results;
                    self.scan_done = true;
                    self.scanning = false;
                    self.set_status("Environment scan complete", false);
                }
                BgMessage::NodeStatusDone(result) => {
                    self.node_refreshing = false;
                    match result {
                        Ok(info) => {
                            self.node_info = Some(info);
                        }
                        Err(e) => {
                            self.node_info = Some(node_manager::NodeInfo {
                                running: false,
                                pid: 0,
                                port: 3000,
                                uptime_secs: 0,
                                health: "unreachable".to_string(),
                            });
                            self.node_log
                                .push_str(&format!("[status] Error: {}\n", e));
                        }
                    }
                }
                BgMessage::DockerStatusDone(result) => {
                    self.docker_refreshing = false;
                    match result {
                        Ok(containers) => {
                            self.containers = containers;
                        }
                        Err(e) => {
                            self.docker_log
                                .push_str(&format!("[status] Error: {}\n", e));
                        }
                    }
                }
                BgMessage::ModelListDone(result) => {
                    match result {
                        Ok(list) => {
                            self.models = list;
                            self.models_loaded = true;
                        }
                        Err(e) => {
                            self.set_status(&format!("Model list error: {}", e), true);
                        }
                    }
                }
                BgMessage::ModelPullDone(result) => {
                    self.model_pulling = false;
                    match result {
                        Ok(msg) => {
                            self.set_status(&msg, false);
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.set_status(&format!("Pull failed: {}", e), true);
                        }
                    }
                }
                BgMessage::ModelDeleteDone(result) => {
                    self.model_deleting = None;
                    match result {
                        Ok(msg) => {
                            self.set_status(&msg, false);
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.set_status(&format!("Delete failed: {}", e), true);
                        }
                    }
                }
                BgMessage::BackupDone(result) => {
                    match result {
                        Ok(info) => {
                            self.set_status(
                                &format!(
                                    "Backup created: {} ({} files, {} bytes)",
                                    info.path.display(),
                                    info.files_count,
                                    info.size_bytes
                                ),
                                false,
                            );
                            self.backup_info = Some(info);
                        }
                        Err(e) => {
                            self.set_status(&format!("Backup failed: {}", e), true);
                        }
                    }
                }
                BgMessage::RestoreDone(result) => match result {
                    Ok(()) => {
                        self.set_status("Restore complete", false);
                    }
                    Err(e) => {
                        self.set_status(&format!("Restore failed: {}", e), true);
                    }
                },
                BgMessage::ConfigValidated(errors) => {
                    self.validation_errors = errors;
                    if self.validation_errors.is_empty() {
                        self.set_status("Configuration is valid", false);
                    } else {
                        self.set_status(
                            &format!(
                                "{} validation error(s)",
                                self.validation_errors.len()
                            ),
                            true,
                        );
                    }
                }
                BgMessage::DockerBuildDone(result) => match result {
                    Ok(msg) => {
                        self.docker_log.push_str(&format!("[build] {}\n", msg));
                        self.set_status("Docker build succeeded", false);
                    }
                    Err(e) => {
                        self.docker_log
                            .push_str(&format!("[build] Error: {}\n", e));
                        self.set_status("Docker build failed", true);
                    }
                },
                BgMessage::DockerComposeUpDone(result) => match result {
                    Ok(msg) => {
                        self.docker_log
                            .push_str(&format!("[compose up] {}\n", msg));
                        self.set_status("Docker Compose up succeeded", false);
                        self.refresh_docker();
                    }
                    Err(e) => {
                        self.docker_log
                            .push_str(&format!("[compose up] Error: {}\n", e));
                        self.set_status("Docker Compose up failed", true);
                    }
                },
                BgMessage::DockerComposeDownDone(result) => match result {
                    Ok(_) => {
                        self.docker_log
                            .push_str("[compose down] Services stopped\n");
                        self.set_status("Docker Compose down succeeded", false);
                        self.refresh_docker();
                    }
                    Err(e) => {
                        self.docker_log
                            .push_str(&format!("[compose down] Error: {}\n", e));
                        self.set_status("Docker Compose down failed", true);
                    }
                },
                BgMessage::NodeStartDone(result) => match result {
                    Ok(msg) => {
                        self.node_log.push_str(&format!("[start] {}\n", msg));
                        self.set_status("Node started", false);
                        self.refresh_node_status();
                    }
                    Err(e) => {
                        self.node_log
                            .push_str(&format!("[start] Error: {}\n", e));
                        self.set_status("Node start failed", true);
                    }
                },
                BgMessage::NodeStopDone(result) => match result {
                    Ok(()) => {
                        self.node_log.push_str("[stop] Server stopped\n");
                        self.set_status("Node stopped", false);
                        self.refresh_node_status();
                    }
                    Err(e) => {
                        self.node_log
                            .push_str(&format!("[stop] Error: {}\n", e));
                        self.set_status("Node stop failed", true);
                    }
                },
            }
        }
    }

    fn set_status(&mut self, msg: &str, is_error: bool) {
        self.status_message = msg.to_string();
        self.status_is_error = is_error;
        self.status_time = Instant::now();
    }

    // =========================================================================
    // Background actions
    // =========================================================================

    fn scan_prerequisites(&mut self) {
        if self.scanning {
            return;
        }
        self.scanning = true;
        self.set_status("Scanning environment...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let results = prereq::check_prerequisites();
            let _ = tx.send(BgMessage::PrereqsDone(results));
        });
    }

    fn refresh_node_status(&mut self) {
        if self.node_refreshing {
            return;
        }
        self.node_refreshing = true;
        self.last_node_refresh = Instant::now();
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = node_manager::node_status();
            let _ = tx.send(BgMessage::NodeStatusDone(result));
        });
    }

    fn refresh_docker(&mut self) {
        if self.docker_refreshing {
            return;
        }
        self.docker_refreshing = true;
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let available = docker_ops::docker_available();
            if available {
                let result = docker_ops::docker_status();
                let _ = tx.send(BgMessage::DockerStatusDone(result));
            } else {
                let _ = tx.send(BgMessage::DockerStatusDone(Err(
                    "Docker is not available".to_string(),
                )));
            }
        });
    }

    fn refresh_models(&mut self) {
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = list_ollama_models();
            let _ = tx.send(BgMessage::ModelListDone(result));
        });
    }

    fn pull_model(&mut self, name: String) {
        if self.model_pulling || name.is_empty() {
            return;
        }
        self.model_pulling = true;
        self.set_status(&format!("Pulling model {}...", name), false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = run_ollama_pull(&name);
            let _ = tx.send(BgMessage::ModelPullDone(result));
        });
    }

    fn delete_model(&mut self, name: String) {
        if self.model_deleting.is_some() {
            return;
        }
        self.model_deleting = Some(name.clone());
        self.set_status(&format!("Deleting model {}...", name), false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = run_ollama_delete(&name);
            let _ = tx.send(BgMessage::ModelDeleteDone(result));
        });
    }

    fn start_node_bg(&mut self) {
        let config_path = PathBuf::from(&self.config_path);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = node_manager::start_node(&config_path, false);
            let _ = tx.send(BgMessage::NodeStartDone(result));
        });
    }

    fn stop_node_bg(&mut self) {
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = node_manager::stop_node();
            let _ = tx.send(BgMessage::NodeStopDone(result));
        });
    }

    fn create_backup_bg(&mut self) {
        let config_dir = PathBuf::from(&self.config_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        let output = if self.backup_output_path.is_empty() {
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            PathBuf::from(format!("ai_assistant_backup_{}.gz", ts))
        } else {
            PathBuf::from(&self.backup_output_path)
        };

        let include_models = self.backup_include_models;
        self.set_status("Creating backup...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = backup::create_backup(&config_dir, &output, include_models);
            let _ = tx.send(BgMessage::BackupDone(result));
        });
    }

    fn restore_backup_bg(&mut self) {
        let archive = PathBuf::from(&self.restore_archive_path);
        let target = PathBuf::from(&self.config_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();

        self.set_status("Restoring backup...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = backup::restore_backup(&archive, &target);
            let _ = tx.send(BgMessage::RestoreDone(result));
        });
    }

    fn docker_build_bg(&mut self) {
        let features = self.docker_build_features.clone();
        self.set_status("Building Docker image...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = docker_ops::docker_build(&features);
            let _ = tx.send(BgMessage::DockerBuildDone(result));
        });
    }

    fn docker_compose_up_bg(&mut self) {
        let profiles: Vec<String> = self
            .docker_profiles
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(name, _)| name.clone())
            .collect();
        self.set_status("Starting Docker Compose...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let profile_refs: Vec<&str> = profiles.iter().map(|s| s.as_str()).collect();
            let result = docker_ops::docker_compose_up(&profile_refs);
            let _ = tx.send(BgMessage::DockerComposeUpDone(result));
        });
    }

    fn docker_compose_down_bg(&mut self) {
        self.set_status("Stopping Docker Compose...", false);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let result = docker_ops::docker_compose_down();
            let _ = tx.send(BgMessage::DockerComposeDownDone(result));
        });
    }

    fn validate_config_bg(&mut self) {
        let path = PathBuf::from(&self.config_path);
        let tx = self.bg_tx.clone();
        std::thread::spawn(move || {
            let errors = match ConfigFile::load(&path) {
                Ok(config) => match config.validate_detailed() {
                    Ok(()) => Vec::new(),
                    Err(errs) => errs.iter().map(|e| format!("{}", e)).collect(),
                },
                Err(e) => vec![format!("Failed to load config: {}", e)],
            };
            let _ = tx.send(BgMessage::ConfigValidated(errors));
        });
    }

    // =========================================================================
    // Config loading helpers
    // =========================================================================

    fn load_config_fields(&mut self) {
        let path = PathBuf::from(&self.config_path);
        if !path.exists() {
            return;
        }
        // Load raw text
        if let Ok(content) = std::fs::read_to_string(&path) {
            self.config_text = content;
        }
        // Load individual fields
        let get = |key: &str| -> String {
            config_ops::get_config_value(&path, key).unwrap_or_default()
        };

        let val = get("provider.type");
        if !val.is_empty() {
            self.cfg_provider_type = val;
        }
        let val = get("provider.model");
        if !val.is_empty() {
            self.cfg_provider_model = val;
        }
        let val = get("provider.api_key");
        if !val.is_empty() {
            self.cfg_provider_api_key = val;
        }
        let val = get("urls.ollama");
        if !val.is_empty() {
            self.cfg_url_ollama = val;
        }
        let val = get("urls.lm_studio");
        if !val.is_empty() {
            self.cfg_url_lm_studio = val;
        }
        if let Ok(v) = get("generation.temperature").parse::<f32>() {
            self.cfg_temperature = v;
        }
        if let Ok(v) = get("generation.max_history").parse::<u32>() {
            self.cfg_max_history = v;
        }
        let val = get("rag.enabled");
        if !val.is_empty() {
            self.cfg_rag_enabled = val == "true";
        }
        if let Ok(v) = get("rag.knowledge_tokens").parse::<u32>() {
            self.cfg_rag_knowledge_tokens = v;
        }
        if let Ok(v) = get("rag.conversation_tokens").parse::<u32>() {
            self.cfg_rag_conversation_tokens = v;
        }
        let val = get("logging.level");
        if !val.is_empty() {
            self.cfg_log_level = val;
        }
        let val = get("cache.enabled");
        if !val.is_empty() {
            self.cfg_cache_enabled = val == "true";
        }
        if let Ok(v) = get("cache.max_entries").parse::<u32>() {
            self.cfg_cache_max_entries = v;
        }

        self.config_loaded = true;
        self.config_modified = false;
    }

    fn save_config_fields(&mut self) {
        let path = PathBuf::from(&self.config_path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let content = format!(
            r#"# AI Assistant configuration — managed by ai_setup_gui
# See: https://ai-assistant.runawaybrains.com/docs/config

[provider]
type = "{}"
model = "{}"
{}

[urls]
ollama = "{}"
lm_studio = "{}"

[generation]
temperature = {}
max_history = {}

[rag]
enabled = {}
knowledge_tokens = {}
conversation_tokens = {}

[logging]
level = "{}"

[cache]
enabled = {}
max_entries = {}
"#,
            self.cfg_provider_type,
            self.cfg_provider_model,
            if self.cfg_provider_api_key.is_empty() {
                String::new()
            } else {
                format!("api_key = \"{}\"", self.cfg_provider_api_key)
            },
            self.cfg_url_ollama,
            self.cfg_url_lm_studio,
            self.cfg_temperature,
            self.cfg_max_history,
            self.cfg_rag_enabled,
            self.cfg_rag_knowledge_tokens,
            self.cfg_rag_conversation_tokens,
            self.cfg_log_level,
            self.cfg_cache_enabled,
            self.cfg_cache_max_entries,
        );

        match std::fs::write(&path, &content) {
            Ok(()) => {
                self.config_text = content;
                self.config_modified = false;
                self.set_status(&format!("Config saved to {}", path.display()), false);
            }
            Err(e) => {
                self.set_status(&format!("Failed to save config: {}", e), true);
            }
        }
    }

    // =========================================================================
    // Rendering — left panel (tab selector)
    // =========================================================================

    fn render_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("tab_panel")
            .resizable(false)
            .exact_width(130.0)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    ui.heading(
                        RichText::new("ai_setup")
                            .strong()
                            .color(Color32::from_rgb(100, 200, 255)),
                    );
                    ui.label(
                        RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .small()
                            .color(Color32::GRAY),
                    );
                    ui.add_space(12.0);
                });

                ui.separator();
                ui.add_space(4.0);

                for tab in Tab::all() {
                    let selected = self.tab == *tab;
                    let text = format!("{} {}", tab.icon(), tab.label());
                    let label = if selected {
                        RichText::new(text)
                            .strong()
                            .color(Color32::from_rgb(100, 200, 255))
                    } else {
                        RichText::new(text).color(Color32::LIGHT_GRAY)
                    };

                    let response = ui.selectable_label(selected, label);
                    if response.clicked() {
                        self.tab = *tab;
                        // Lazy load data when switching to a tab
                        match tab {
                            Tab::Setup if !self.scan_done && !self.scanning => {
                                self.scan_prerequisites();
                            }
                            Tab::Config if !self.config_loaded => {
                                self.load_config_fields();
                            }
                            Tab::Nodes => {
                                self.refresh_node_status();
                            }
                            Tab::Docker => {
                                self.docker_available = Some(docker_ops::docker_available());
                                self.refresh_docker();
                            }
                            Tab::Models if !self.models_loaded => {
                                self.refresh_models();
                            }
                            _ => {}
                        }
                    }
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Platform info
                ui.label(
                    RichText::new(format!(
                        "{}/{}",
                        std::env::consts::OS,
                        std::env::consts::ARCH
                    ))
                    .small()
                    .color(Color32::GRAY),
                );
            });
    }

    // =========================================================================
    // Rendering — status bar
    // =========================================================================

    fn render_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(24.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Fade out status after 10 seconds
                    let age = self.status_time.elapsed().as_secs_f32();
                    let alpha = if age < 8.0 {
                        1.0
                    } else if age < 10.0 {
                        1.0 - (age - 8.0) / 2.0
                    } else {
                        0.0
                    };

                    if alpha > 0.01 && !self.status_message.is_empty() {
                        let color = if self.status_is_error {
                            Color32::from_rgba_unmultiplied(255, 80, 80, (alpha * 255.0) as u8)
                        } else {
                            Color32::from_rgba_unmultiplied(
                                150,
                                220,
                                150,
                                (alpha * 255.0) as u8,
                            )
                        };
                        ui.label(RichText::new(&self.status_message).small().color(color));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("Config: {}", self.config_path))
                                .small()
                                .color(Color32::GRAY),
                        );
                    });
                });
            });
    }

    // =========================================================================
    // Rendering — Setup tab
    // =========================================================================

    fn render_setup_tab(&mut self, ui: &mut Ui) {
        ui.heading("Environment Setup");
        ui.add_space(4.0);
        ui.label("Scan your system for prerequisites and get installation guidance.");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.scanning, egui::Button::new("Scan Again"))
                .clicked()
            {
                self.scan_prerequisites();
            }
            if self.scanning {
                ui.spinner();
                ui.label("Scanning...");
            }
            if self.scan_done && !self.scanning {
                let missing: Vec<_> = self
                    .prereqs
                    .iter()
                    .filter(|p| !p.installed)
                    .map(|p| p.name.as_str())
                    .collect();
                if missing.is_empty() {
                    if ui.button("Apply Recommended Config").clicked() {
                        self.generate_recommended_config();
                        self.set_status("Recommended config applied", false);
                    }
                }
            }
        });

        ui.add_space(12.0);

        if !self.prereqs.is_empty() {
            egui::Grid::new("prereq_grid")
                .num_columns(4)
                .spacing([12.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    // Header
                    ui.label(RichText::new("Status").strong());
                    ui.label(RichText::new("Component").strong());
                    ui.label(RichText::new("Details").strong());
                    ui.label(RichText::new("Action").strong());
                    ui.end_row();

                    let prereqs_clone = self.prereqs.clone();
                    for p in &prereqs_clone {
                        // Status icon
                        if p.installed {
                            ui.label(
                                RichText::new("\u{2713}")
                                    .color(Color32::from_rgb(80, 220, 80))
                                    .strong(),
                            );
                        } else {
                            ui.label(
                                RichText::new("\u{2717}")
                                    .color(Color32::from_rgb(255, 80, 80))
                                    .strong(),
                            );
                        }

                        // Name + version
                        let name_text = if let Some(ref ver) = p.version {
                            format!("{} ({})", p.name, ver)
                        } else {
                            p.name.clone()
                        };
                        ui.label(&name_text);

                        // Details
                        ui.label(
                            RichText::new(&p.details)
                                .small()
                                .color(Color32::LIGHT_GRAY),
                        );

                        // Install button for missing items
                        if !p.installed {
                            let target = p.name.to_lowercase();
                            if target == "ollama" || target == "docker" {
                                if ui.small_button("Install Info").clicked() {
                                    if let Ok(instructions) =
                                        prereq::install_command(&target)
                                    {
                                        self.install_info =
                                            Some((p.name.clone(), instructions));
                                    }
                                }
                            } else {
                                ui.label(""); // no action for API keys / GPU
                            }
                        } else {
                            ui.label(""); // spacer
                        }
                        ui.end_row();
                    }
                });
        } else if !self.scanning {
            ui.label(
                RichText::new("Click \"Scan Again\" to check your environment.")
                    .color(Color32::GRAY),
            );
        }

        // Install instructions popup
        if let Some((ref name, ref instructions)) = self.install_info.clone() {
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.heading(format!("Install: {}", name));

            if !instructions.command.is_empty() {
                ui.label(RichText::new("Command:").strong());
                ui.code(&instructions.command);
            }
            if !instructions.manual_steps.is_empty() {
                ui.add_space(4.0);
                ui.label(RichText::new("Manual steps:").strong());
                ui.label(&instructions.manual_steps);
            }
            if !instructions.url.is_empty() {
                ui.add_space(4.0);
                ui.hyperlink_to("More info", &instructions.url);
            }
            ui.add_space(4.0);
            if ui.button("Dismiss").clicked() {
                self.install_info = None;
            }
        }
    }

    fn generate_recommended_config(&mut self) {
        let ollama_ok = self
            .prereqs
            .iter()
            .any(|s| s.name == "Ollama" && s.installed);
        let openai_ok = self
            .prereqs
            .iter()
            .any(|s| s.name == "OpenAI API Key" && s.installed);

        if ollama_ok {
            self.cfg_provider_type = "ollama".to_string();
            self.cfg_provider_model = "llama3".to_string();
        } else if openai_ok {
            self.cfg_provider_type = "openai".to_string();
            self.cfg_provider_model = "gpt-4o-mini".to_string();
        }

        self.config_modified = true;
        self.save_config_fields();
    }

    // =========================================================================
    // Rendering — Config tab
    // =========================================================================

    fn render_config_tab(&mut self, ui: &mut Ui) {
        ui.heading("Configuration");
        ui.add_space(4.0);

        // Config path
        ui.horizontal(|ui| {
            ui.label("Path:");
            ui.text_edit_singleline(&mut self.config_path);
            if ui.button("Reload").clicked() {
                self.config_loaded = false;
                self.load_config_fields();
            }
        });

        ui.add_space(8.0);

        // Button bar
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() {
                self.save_config_fields();
            }
            if ui.button("Validate").clicked() {
                self.validate_config_bg();
            }
            if ui.button("Export TOML").clicked() {
                let path = PathBuf::from(&self.config_path);
                let output = path.with_extension("export.toml");
                match config_ops::export_config(&path, "toml", &output) {
                    Ok(()) => self.set_status(
                        &format!("Exported to {}", output.display()),
                        false,
                    ),
                    Err(e) => self.set_status(&format!("Export failed: {}", e), true),
                }
            }
            if ui.button("Export JSON").clicked() {
                let path = PathBuf::from(&self.config_path);
                let output = path.with_extension("export.json");
                match config_ops::export_config(&path, "json", &output) {
                    Ok(()) => self.set_status(
                        &format!("Exported to {}", output.display()),
                        false,
                    ),
                    Err(e) => self.set_status(&format!("Export failed: {}", e), true),
                }
            }

            if self.config_modified {
                ui.label(
                    RichText::new("(unsaved changes)")
                        .small()
                        .color(Color32::YELLOW),
                );
            }
        });

        ui.add_space(8.0);

        // Validation results
        if !self.validation_errors.is_empty() {
            ui.group(|ui| {
                ui.label(
                    RichText::new("Validation Errors")
                        .strong()
                        .color(Color32::from_rgb(255, 80, 80)),
                );
                for err in &self.validation_errors.clone() {
                    ui.label(
                        RichText::new(format!("\u{2717} {}", err))
                            .color(Color32::from_rgb(255, 120, 120)),
                    );
                }
            });
            ui.add_space(4.0);
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Provider section
            let mut modified = false;
            egui::CollapsingHeader::new(
                RichText::new("Provider").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("cfg_provider")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Type:");
                        let before = self.cfg_provider_type.clone();
                        egui::ComboBox::from_id_source("provider_type")
                            .selected_text(&self.cfg_provider_type)
                            .show_ui(ui, |ui| {
                                for t in &[
                                    "ollama",
                                    "lm_studio",
                                    "openai",
                                    "anthropic",
                                    "gemini",
                                ] {
                                    ui.selectable_value(
                                        &mut self.cfg_provider_type,
                                        t.to_string(),
                                        *t,
                                    );
                                }
                            });
                        if self.cfg_provider_type != before {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("Model:");
                        if ui.text_edit_singleline(&mut self.cfg_provider_model).changed() {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("API Key:");
                        ui.horizontal(|ui| {
                            if self.show_api_keys {
                                if ui
                                    .text_edit_singleline(&mut self.cfg_provider_api_key)
                                    .changed()
                                {
                                    modified = true;
                                }
                            } else {
                                let mut masked = if self.cfg_provider_api_key.is_empty() {
                                    String::new()
                                } else {
                                    "\u{2022}".repeat(
                                        self.cfg_provider_api_key.len().min(20),
                                    )
                                };
                                ui.add(
                                    egui::TextEdit::singleline(&mut masked).interactive(false),
                                );
                            }
                            if ui
                                .small_button(if self.show_api_keys {
                                    "Hide"
                                } else {
                                    "Show"
                                })
                                .clicked()
                            {
                                self.show_api_keys = !self.show_api_keys;
                            }
                        });
                        ui.end_row();
                    });
            });

            // Server URLs section
            egui::CollapsingHeader::new(
                RichText::new("Server URLs").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("cfg_urls")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Ollama:");
                        if ui.text_edit_singleline(&mut self.cfg_url_ollama).changed() {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("LM Studio:");
                        if ui
                            .text_edit_singleline(&mut self.cfg_url_lm_studio)
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();
                    });
            });

            // Generation section
            egui::CollapsingHeader::new(
                RichText::new("Generation").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("cfg_generation")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Temperature:");
                        if ui
                            .add(egui::Slider::new(&mut self.cfg_temperature, 0.0..=2.0))
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("Max History:");
                        if ui
                            .add(egui::DragValue::new(&mut self.cfg_max_history).clamp_range(1..=100))
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();
                    });
            });

            // RAG section
            egui::CollapsingHeader::new(
                RichText::new("RAG").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("cfg_rag")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Enabled:");
                        if ui.checkbox(&mut self.cfg_rag_enabled, "").changed() {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("Knowledge Tokens:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.cfg_rag_knowledge_tokens)
                                    .clamp_range(0..=16000),
                            )
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("Conversation Tokens:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.cfg_rag_conversation_tokens)
                                    .clamp_range(0..=16000),
                            )
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();
                    });
            });

            // Logging section
            egui::CollapsingHeader::new(
                RichText::new("Logging").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("cfg_logging")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Level:");
                        let before = self.cfg_log_level.clone();
                        egui::ComboBox::from_id_source("log_level")
                            .selected_text(&self.cfg_log_level)
                            .show_ui(ui, |ui| {
                                for level in &["trace", "debug", "info", "warn", "error"] {
                                    ui.selectable_value(
                                        &mut self.cfg_log_level,
                                        level.to_string(),
                                        *level,
                                    );
                                }
                            });
                        if self.cfg_log_level != before {
                            modified = true;
                        }
                        ui.end_row();
                    });
            });

            // Cache section
            egui::CollapsingHeader::new(
                RichText::new("Cache").strong().color(Color32::from_rgb(100, 200, 255)),
            )
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("cfg_cache")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Enabled:");
                        if ui.checkbox(&mut self.cfg_cache_enabled, "").changed() {
                            modified = true;
                        }
                        ui.end_row();

                        ui.label("Max Entries:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.cfg_cache_max_entries)
                                    .clamp_range(0..=100000),
                            )
                            .changed()
                        {
                            modified = true;
                        }
                        ui.end_row();
                    });
            });

            if modified {
                self.config_modified = true;
            }
        });
    }

    // =========================================================================
    // Rendering — Nodes tab
    // =========================================================================

    fn render_nodes_tab(&mut self, ui: &mut Ui) {
        ui.heading("Node Management");
        ui.add_space(4.0);

        // Auto-refresh every 5 seconds
        if self.last_node_refresh.elapsed().as_secs() >= 5 && !self.node_refreshing {
            self.refresh_node_status();
        }

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.node_refreshing, egui::Button::new("Refresh"))
                .clicked()
            {
                self.refresh_node_status();
            }
            if self.node_refreshing {
                ui.spinner();
            }
        });

        ui.add_space(8.0);

        if let Some(ref info) = self.node_info.clone() {
            ui.group(|ui| {
                egui::Grid::new("node_info_grid")
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Status:").strong());
                        if info.running {
                            ui.label(
                                RichText::new("Running")
                                    .color(Color32::from_rgb(80, 220, 80))
                                    .strong(),
                            );
                        } else {
                            ui.label(
                                RichText::new("Stopped")
                                    .color(Color32::from_rgb(255, 80, 80))
                                    .strong(),
                            );
                        }
                        ui.end_row();

                        ui.label(RichText::new("PID:").strong());
                        ui.label(if info.pid > 0 {
                            info.pid.to_string()
                        } else {
                            "N/A".to_string()
                        });
                        ui.end_row();

                        ui.label(RichText::new("Port:").strong());
                        ui.label(info.port.to_string());
                        ui.end_row();

                        ui.label(RichText::new("Health:").strong());
                        let health_color = match info.health.as_str() {
                            "ok" => Color32::from_rgb(80, 220, 80),
                            "unreachable" => Color32::from_rgb(255, 80, 80),
                            _ => Color32::YELLOW,
                        };
                        ui.label(
                            RichText::new(&info.health).color(health_color),
                        );
                        ui.end_row();

                        if info.uptime_secs > 0 {
                            ui.label(RichText::new("Uptime:").strong());
                            ui.label(format_duration(info.uptime_secs));
                            ui.end_row();
                        }
                    });
            });

            ui.add_space(8.0);

            ui.horizontal(|ui| {
                let is_running = info.running;
                if ui
                    .add_enabled(!is_running, egui::Button::new("Start"))
                    .clicked()
                {
                    self.start_node_bg();
                }
                if ui
                    .add_enabled(is_running, egui::Button::new("Stop"))
                    .clicked()
                {
                    self.stop_node_bg();
                }
                if ui
                    .add_enabled(is_running, egui::Button::new("Restart"))
                    .clicked()
                {
                    self.stop_node_bg();
                    // Start will be triggered by the stop completion handler
                }
            });
        } else {
            ui.label(
                RichText::new("Checking node status...")
                    .color(Color32::GRAY),
            );
        }

        // Node log
        if !self.node_log.is_empty() {
            ui.add_space(12.0);
            ui.label(RichText::new("Log:").strong());
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.node_log.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
        }
    }

    // =========================================================================
    // Rendering — Docker tab
    // =========================================================================

    fn render_docker_tab(&mut self, ui: &mut Ui) {
        ui.heading("Docker Management");
        ui.add_space(4.0);

        // Docker availability indicator
        match self.docker_available {
            Some(true) => {
                ui.label(
                    RichText::new("\u{2713} Docker is available")
                        .color(Color32::from_rgb(80, 220, 80)),
                );
            }
            Some(false) => {
                ui.label(
                    RichText::new("\u{2717} Docker is not available")
                        .color(Color32::from_rgb(255, 80, 80)),
                );
                ui.label("Install Docker to use these features.");
                return;
            }
            None => {
                self.docker_available = Some(docker_ops::docker_available());
                return;
            }
        }

        ui.add_space(8.0);

        // Container table
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.docker_refreshing, egui::Button::new("Refresh"))
                .clicked()
            {
                self.refresh_docker();
            }
            if self.docker_refreshing {
                ui.spinner();
            }
        });

        ui.add_space(8.0);

        if self.containers.is_empty() {
            ui.label(
                RichText::new("No running containers.")
                    .color(Color32::GRAY),
            );
        } else {
            egui::Grid::new("docker_containers")
                .num_columns(4)
                .spacing([12.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("Name").strong());
                    ui.label(RichText::new("Status").strong());
                    ui.label(RichText::new("Health").strong());
                    ui.label(RichText::new("Ports").strong());
                    ui.end_row();

                    for c in &self.containers.clone() {
                        ui.label(&c.name);
                        ui.label(&c.status);
                        let health_color = match c.health.as_str() {
                            "healthy" => Color32::from_rgb(80, 220, 80),
                            "unhealthy" => Color32::from_rgb(255, 80, 80),
                            _ => Color32::YELLOW,
                        };
                        ui.label(
                            RichText::new(&c.health).color(health_color),
                        );
                        ui.label(
                            RichText::new(&c.ports)
                                .small()
                                .color(Color32::LIGHT_GRAY),
                        );
                        ui.end_row();
                    }
                });
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        // Docker actions
        ui.label(RichText::new("Actions").strong());
        ui.add_space(4.0);

        // Build
        ui.horizontal(|ui| {
            ui.label("Features:");
            ui.text_edit_singleline(&mut self.docker_build_features);
            if ui.button("Build Image").clicked() {
                self.docker_build_bg();
            }
        });

        ui.add_space(4.0);

        // Profiles
        ui.horizontal(|ui| {
            ui.label("Profiles:");
            let mut profiles = self.docker_profiles.clone();
            for (name, enabled) in &mut profiles {
                ui.checkbox(enabled, name.as_str());
            }
            self.docker_profiles = profiles;
        });

        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button("Compose Up").clicked() {
                self.docker_compose_up_bg();
            }
            if ui.button("Compose Down").clicked() {
                self.docker_compose_down_bg();
            }
        });

        // Docker log
        if !self.docker_log.is_empty() {
            ui.add_space(12.0);
            ui.label(RichText::new("Log:").strong());
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.docker_log.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
        }
    }

    // =========================================================================
    // Rendering — Models tab
    // =========================================================================

    fn render_models_tab(&mut self, ui: &mut Ui) {
        ui.heading("Model Management");
        ui.add_space(4.0);

        // Check if Ollama is available
        let ollama_detected = self
            .prereqs
            .iter()
            .any(|p| p.name == "Ollama" && p.installed);

        if !self.scan_done {
            ui.label("Run an environment scan first (Setup tab) to detect Ollama.");
            if ui.button("Scan Now").clicked() {
                self.scan_prerequisites();
            }
            return;
        }

        if !ollama_detected {
            ui.label(
                RichText::new("Ollama is not detected. Install Ollama to manage local models.")
                    .color(Color32::YELLOW),
            );
            return;
        }

        // Pull model
        ui.horizontal(|ui| {
            ui.label("Pull model:");
            ui.text_edit_singleline(&mut self.model_pull_name);
            if ui
                .add_enabled(
                    !self.model_pulling && !self.model_pull_name.is_empty(),
                    egui::Button::new("Pull"),
                )
                .clicked()
            {
                let name = self.model_pull_name.clone();
                self.pull_model(name);
            }
            if self.model_pulling {
                ui.spinner();
                ui.label("Pulling...");
            }
        });

        ui.add_space(4.0);
        if ui.button("Refresh Model List").clicked() {
            self.refresh_models();
        }

        ui.add_space(8.0);

        if self.models.is_empty() {
            ui.label(
                RichText::new("No models installed. Pull a model to get started.")
                    .color(Color32::GRAY),
            );
        } else {
            egui::Grid::new("model_list")
                .num_columns(4)
                .spacing([12.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("Model").strong());
                    ui.label(RichText::new("Size").strong());
                    ui.label(RichText::new("Modified").strong());
                    ui.label(RichText::new("Action").strong());
                    ui.end_row();

                    let models_clone = self.models.clone();
                    for m in &models_clone {
                        ui.label(&m.name);
                        ui.label(
                            RichText::new(&m.size)
                                .small()
                                .color(Color32::LIGHT_GRAY),
                        );
                        ui.label(
                            RichText::new(&m.modified)
                                .small()
                                .color(Color32::GRAY),
                        );

                        let is_deleting = self
                            .model_deleting
                            .as_ref()
                            .map(|d| d == &m.name)
                            .unwrap_or(false);

                        if is_deleting {
                            ui.spinner();
                        } else if ui.small_button("Delete").clicked() {
                            let name = m.name.clone();
                            self.delete_model(name);
                        }
                        ui.end_row();
                    }
                });
        }
    }

    // =========================================================================
    // Rendering — Backup tab
    // =========================================================================

    fn render_backup_tab(&mut self, ui: &mut Ui) {
        ui.heading("Backup & Restore");
        ui.add_space(4.0);

        // Create backup section
        ui.label(RichText::new("Create Backup").strong());
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Output path:");
            ui.text_edit_singleline(&mut self.backup_output_path);
            ui.label(
                RichText::new("(leave empty for auto-named)")
                    .small()
                    .color(Color32::GRAY),
            );
        });

        ui.checkbox(&mut self.backup_include_models, "Include model files (may be very large)");

        ui.add_space(4.0);
        if ui.button("Create Backup").clicked() {
            self.create_backup_bg();
        }

        // Last backup info
        if let Some(ref info) = self.backup_info {
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label(RichText::new("Last Backup").strong());
                ui.label(format!("Path: {}", info.path.display()));
                ui.label(format!(
                    "Size: {} bytes ({} files)",
                    info.size_bytes, info.files_count
                ));
            });
        }

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);

        // Restore section
        ui.label(RichText::new("Restore from Backup").strong());
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Archive path:");
            ui.text_edit_singleline(&mut self.restore_archive_path);
        });

        ui.add_space(4.0);
        if ui
            .add_enabled(
                !self.restore_archive_path.is_empty(),
                egui::Button::new("Restore"),
            )
            .clicked()
        {
            self.restore_backup_bg();
        }
    }
}

// =============================================================================
// eframe::App implementation
// =============================================================================

impl eframe::App for SetupGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background tasks
        self.poll_background();

        // Auto-refresh while tasks are active
        if self.scanning
            || self.node_refreshing
            || self.docker_refreshing
            || self.model_pulling
            || self.model_deleting.is_some()
        {
            ctx.request_repaint();
        }

        // Auto-refresh nodes tab every 5 seconds
        if self.tab == Tab::Nodes && self.last_node_refresh.elapsed().as_secs() >= 5 {
            ctx.request_repaint();
        }

        // Auto-dismiss status messages
        if !self.status_message.is_empty() && self.status_time.elapsed().as_secs() < 12 {
            ctx.request_repaint();
        }

        // Auto-scan on first frame
        if !self.scan_done && !self.scanning {
            self.scan_prerequisites();
        }

        // Render panels
        self.render_left_panel(ctx);
        self.render_status_bar(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                Tab::Setup => self.render_setup_tab(ui),
                Tab::Config => self.render_config_tab(ui),
                Tab::Nodes => self.render_nodes_tab(ui),
                Tab::Docker => self.render_docker_tab(ui),
                Tab::Models => self.render_models_tab(ui),
                Tab::Backup => self.render_backup_tab(ui),
            }
        });
    }
}

// =============================================================================
// Ollama model helpers
// =============================================================================

/// Parse `ollama list` output into model entries.
fn list_ollama_models() -> Result<Vec<OllamaModelEntry>, String> {
    let output = std::process::Command::new("ollama")
        .arg("list")
        .output()
        .map_err(|e| format!("Failed to run ollama list: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "ollama list failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut models = Vec::new();

    for (i, line) in stdout.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: NAME  ID  SIZE  MODIFIED
        // Columns are whitespace-separated but names and dates can have spaces
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let name = parts[0].to_string();
            // Size is typically "X.Y GB" or "X.Y MB" — find it
            let mut size = String::new();
            let mut modified = String::new();

            // Find the size value (number followed by GB/MB/KB)
            for (j, part) in parts.iter().enumerate().skip(1) {
                if part.ends_with("GB")
                    || part.ends_with("MB")
                    || part.ends_with("KB")
                    || part.ends_with("B")
                {
                    // Previous part might be the numeric portion
                    if j > 1 {
                        size = format!("{} {}", parts[j - 1], part);
                    } else {
                        size = part.to_string();
                    }
                    // Everything after is modified date
                    if j + 1 < parts.len() {
                        modified = parts[j + 1..].join(" ");
                    }
                    break;
                }
            }

            if size.is_empty() {
                // Fallback: just take what we can
                size = parts.get(2).unwrap_or(&"").to_string();
                modified = parts.get(3..).map(|s| s.join(" ")).unwrap_or_default();
            }

            models.push(OllamaModelEntry {
                name,
                size,
                modified,
            });
        }
    }

    Ok(models)
}

/// Run `ollama pull <model>`.
fn run_ollama_pull(name: &str) -> Result<String, String> {
    let output = std::process::Command::new("ollama")
        .args(["pull", name])
        .output()
        .map_err(|e| format!("Failed to run ollama pull: {}", e))?;

    if output.status.success() {
        Ok(format!("Successfully pulled {}", name))
    } else {
        Err(format!(
            "ollama pull {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Run `ollama rm <model>`.
fn run_ollama_delete(name: &str) -> Result<String, String> {
    let output = std::process::Command::new("ollama")
        .args(["rm", name])
        .output()
        .map_err(|e| format!("Failed to run ollama rm: {}", e))?;

    if output.status.success() {
        Ok(format!("Deleted {}", name))
    } else {
        Err(format!(
            "ollama rm {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

// =============================================================================
// Utility helpers
// =============================================================================

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

// =============================================================================
// App icon — 64x64 RGBA with a gear silhouette on gradient background
// =============================================================================

fn generate_app_icon() -> egui::IconData {
    let size = 64usize;
    let mut rgba = vec![0u8; size * size * 4];

    let center = size as f32 / 2.0;
    let radius = center - 2.0;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = (y * size + x) * 4;

            if dist <= radius {
                // Gradient background: teal to dark blue
                let t = dy / (size as f32);
                let r = (20.0 + t * 40.0).clamp(0.0, 255.0) as u8;
                let g = (80.0 + t * 60.0).clamp(0.0, 255.0) as u8;
                let b = (140.0 + t * 60.0).clamp(0.0, 255.0) as u8;

                // Gear shape: outer ring with teeth + inner circle
                let gear_outer = 22.0;
                let gear_inner = 14.0;
                let tooth_height = 6.0;
                let num_teeth = 8;

                let angle = dy.atan2(dx);
                let tooth_angle = std::f32::consts::PI * 2.0 / num_teeth as f32;
                let tooth_phase = ((angle / tooth_angle) + 0.5).fract();
                let in_tooth = tooth_phase < 0.5;
                let effective_outer = if in_tooth {
                    gear_outer + tooth_height
                } else {
                    gear_outer
                };

                let is_gear_ring = dist >= gear_inner && dist <= effective_outer;
                let is_center_hole = dist <= 8.0;
                let is_center_dot = dist <= 4.0;

                if (is_gear_ring && !is_center_hole) || is_center_dot {
                    // White gear
                    rgba[idx] = 240;
                    rgba[idx + 1] = 245;
                    rgba[idx + 2] = 250;
                    rgba[idx + 3] = 255;
                } else {
                    rgba[idx] = r;
                    rgba[idx + 1] = g;
                    rgba[idx + 2] = b;
                    // Soft edge on circle border
                    let edge = ((radius - dist) * 4.0).clamp(0.0, 255.0) as u8;
                    rgba[idx + 3] = edge;
                }
            }
        }
    }

    egui::IconData {
        rgba,
        width: size as u32,
        height: size as u32,
    }
}

// =============================================================================
// Entry point
// =============================================================================

fn main() -> Result<(), eframe::Error> {
    let icon = generate_app_icon();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([950.0, 650.0])
            .with_min_inner_size([700.0, 450.0])
            .with_title("ai_setup_gui")
            .with_icon(std::sync::Arc::new(icon)),
        ..Default::default()
    };

    eframe::run_native(
        "ai_setup_gui",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::new(SetupGuiApp::new())
        }),
    )
}
