// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! AI Assistant GUI — Desktop application for chatting with local LLMs.
//!
//! Run: `cargo run --bin ai_gui --features gui`
//!
//! Features:
//! - Auto-detects local LLM providers (Ollama, LM Studio)
//! - Chat with AI models interactively
//! - Load knowledge packages (.kpkg) or text files for RAG
//! - Monitor metrics, sentiment, topics, and knowledge graph
//! - Butler advisor recommendations

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use eframe::egui::{self, Align2, Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use petgraph::graph::{Graph as PetGraph, NodeIndex};

use ai_assistant::{
    AiAssistant, AiConfig, AiResponse,
    // Analysis
    AdvisorReport, ConversationSentimentAnalysis, SentimentAnalyzer,
    SessionSummarizer, SessionSummary, SummaryConfig, Topic, TopicDetector,
    // Knowledge graph
    KGStats, KnowledgeGraph, KnowledgeGraphConfig,
    PatternEntityExtractor,
    // Security
    AuditEvent, AuditEventType,
    // RAG
    AppKeyProvider, KpkgReader,
    // Butler (advisor types re-exported, Butler itself via module)
    RecommendationPriority,
    // Widgets
    widgets::{self, ChatColors, RagStatus},
    // Context
    ContextUsage,
    // Models
    ModelInfo,
};
use ai_assistant::butler::{Butler, EnvironmentReport, FeatureFlagAnalysis};

// =============================================================================
// Types
// =============================================================================

#[derive(PartialEq, Clone)]
enum AppPhase {
    Scanning,
    Ready,
    NoProviders(String),
}

#[derive(PartialEq, Clone, Copy)]
enum SidebarTab {
    Sessions,
    Knowledge,
    Butler,
}

#[derive(PartialEq, Clone, Copy)]
enum MonitorTab {
    Overview,
    Metrics,
    Analysis,
    Graph,
    Audit,
}

struct ScanResult {
    report: EnvironmentReport,
    config: AiConfig,
    advisor: AdvisorReport,
}

#[derive(Clone)]
struct KnowledgeSourceInfo {
    name: String,
    #[allow(dead_code)]
    file_path: String,
    #[allow(dead_code)]
    is_kpkg: bool,
    doc_count: usize,
    status: KnowledgeStatus,
}

#[derive(Clone, PartialEq)]
enum KnowledgeStatus {
    Pending,
    #[allow(dead_code)]
    Indexing,
    Indexed,
    Error(String),
}

/// Metadata captured when an assistant response completes.
/// Indexed by assistant-message position in conversation.
struct ResponseDetail {
    /// Model used for this response
    model: String,
    /// Response time in ms
    response_time_ms: u64,
    /// Time to first token in ms
    ttft_ms: Option<u64>,
    /// Input tokens
    input_tokens: usize,
    /// Output tokens
    output_tokens: usize,
    /// Knowledge context used (the text sent to the LLM)
    knowledge_context: Option<String>,
    /// Knowledge sources that contributed
    knowledge_sources: Vec<(String, usize, Option<f32>)>, // (source, chunks, relevance)
    /// Total knowledge tokens
    knowledge_tokens: usize,
    /// Thinking/reasoning content (from <think> tags)
    thinking: Option<String>,
    /// Whether context was near limit
    context_near_limit: bool,
}

struct GuiSettings {
    ollama_url: String,
    lm_studio_url: String,
    temperature: f32,
    max_history: usize,
    graph_enabled: bool,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            lm_studio_url: "http://localhost:1234".to_string(),
            temperature: 0.7,
            max_history: 20,
            graph_enabled: false,
        }
    }
}

struct GraphVisualizationData {
    graph: PetGraph<GraphNode, GraphEdge>,
    positions: HashMap<NodeIndex, Pos2>,
    needs_layout: bool,
    stats: Option<KGStats>,
}

struct GraphNode {
    name: String,
    entity_type: String,
    mention_count: usize,
}

struct GraphEdge {
    relation_type: String,
    #[allow(dead_code)]
    weight: f32,
}

// =============================================================================
// Main Application
// =============================================================================

struct AiGuiApp {
    // Core
    assistant: AiAssistant,
    input_text: String,
    selected_model: String,
    colors: ChatColors,
    scroll_to_bottom: bool,
    last_error: Option<String>,
    // Per-response metadata (indexed by assistant message position)
    response_details: Vec<ResponseDetail>,
    expanded_details: std::collections::HashSet<usize>, // which message indices are expanded
    last_knowledge_ctx: Option<String>, // captured before sending

    // Startup
    phase: AppPhase,
    scan_rx: Option<mpsc::Receiver<ScanResult>>,
    scan_result_data: Option<ScanResult>,

    // RAG
    rag_enabled: bool,
    knowledge_sources: Vec<KnowledgeSourceInfo>,
    indexing_display: Option<(String, usize, usize)>,

    // Knowledge graph
    knowledge_graph: Option<KnowledgeGraph>,
    graph_viz: Option<GraphVisualizationData>,
    pattern_extractor: PatternEntityExtractor,
    pending_graph_docs: Vec<(String, String)>,

    // Butler / Advisor
    #[allow(dead_code)]
    feature_analysis: Option<FeatureFlagAnalysis>,

    // Monitoring
    sentiment_cache: Option<ConversationSentimentAnalysis>,
    topics_cache: Vec<Topic>,
    summary_cache: Option<SessionSummary>,
    analysis_msg_count: usize,
    audit_events: Vec<AuditEvent>,

    // UI state
    sidebar_tab: SidebarTab,
    monitor_tab: MonitorTab,
    show_settings: bool,
    show_monitor: bool,
    settings: GuiSettings,
    toasts: Vec<(String, bool, Instant)>,
    last_dir: Option<PathBuf>,

    // Persistence
    data_dir: PathBuf,
}

impl AiGuiApp {
    fn new() -> Self {
        let data_dir = Self::get_data_dir();
        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            eprintln!("Warning: could not create data dir: {}", e);
        }

        let mut assistant = AiAssistant::new();

        // Try loading saved sessions
        let sessions_path = data_dir.join("sessions.json");
        if sessions_path.exists() {
            let _ = assistant.load_sessions_from_file(&sessions_path);
        }

        // Init RAG
        let rag_path = data_dir.join("rag.db");
        let _ = assistant.init_rag(&rag_path);

        // Start Butler scan in background
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut butler = Butler::new();
            let report = butler.scan();
            let config = butler.suggest_config(&report);
            let advisor = butler.advise(&report);
            let _ = tx.send(ScanResult {
                report,
                config,
                advisor,
            });
        });

        Self {
            assistant,
            input_text: String::new(),
            selected_model: String::new(),
            colors: ChatColors::default(),
            scroll_to_bottom: false,
            last_error: None,
            response_details: Vec::new(),
            expanded_details: std::collections::HashSet::new(),
            last_knowledge_ctx: None,

            phase: AppPhase::Scanning,
            scan_rx: Some(rx),
            scan_result_data: None,

            rag_enabled: false,
            knowledge_sources: Vec::new(),
            indexing_display: None,

            knowledge_graph: None,
            graph_viz: None,
            pattern_extractor: PatternEntityExtractor::new(),
            pending_graph_docs: Vec::new(),

            // advisor is in scan_result_data
            feature_analysis: None,

            sentiment_cache: None,
            topics_cache: Vec::new(),
            summary_cache: None,
            analysis_msg_count: 0,
            audit_events: Vec::new(),

            sidebar_tab: SidebarTab::Sessions,
            monitor_tab: MonitorTab::Overview,
            show_settings: false,
            show_monitor: false,
            settings: GuiSettings::default(),
            toasts: Vec::new(),
            last_dir: None,

            data_dir,
        }
    }

    fn get_data_dir() -> PathBuf {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            PathBuf::from(local).join("ai_assistant")
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".ai_assistant")
        } else {
            PathBuf::from(".ai_assistant")
        }
    }

    fn add_toast(&mut self, msg: &str, is_error: bool) {
        self.toasts.push((msg.to_string(), is_error, Instant::now()));
    }

    fn add_audit(&mut self, event_type: AuditEventType) {
        let event = AuditEvent::new(event_type);
        self.audit_events.push(event);
    }

    // =========================================================================
    // Polling (called every frame)
    // =========================================================================

    fn poll_all(&mut self) {
        // 1. Butler scan completion
        if let Some(ref rx) = self.scan_rx {
            if let Ok(result) = rx.try_recv() {
                self.apply_scan_result(result);
            }
        }

        // 2. Model fetching
        if self.assistant.is_fetching_models {
            self.assistant.poll_models();
        }

        // 3. Response polling
        if self.assistant.is_generating {
            while let Some(response) = self.assistant.poll_response() {
                match response {
                    AiResponse::Chunk(_) => {}
                    AiResponse::Complete(_text) => {
                        self.scroll_to_bottom = true;
                        self.capture_response_detail();
                        self.add_audit(AuditEventType::ResponseReceived);
                    }
                    AiResponse::Error(e) => {
                        self.last_error = Some(e);
                    }
                    AiResponse::Cancelled(_) => {
                        self.add_audit(AuditEventType::ResponseCancelled);
                    }
                    AiResponse::ModelsLoaded(_) => {}
                }
            }
        }

        // 4. Indexing polling
        if self.assistant.is_indexing {
            while let Some(progress) = self.assistant.poll_indexing() {
                match progress {
                    ai_assistant::IndexingProgress::Starting {
                        source,
                        total_documents,
                        current,
                    } => {
                        self.indexing_display = Some((source, current, total_documents));
                    }
                    ai_assistant::IndexingProgress::Completed(result) => {
                        for ks in &mut self.knowledge_sources {
                            if ks.name == result.source {
                                ks.status = KnowledgeStatus::Indexed;
                            }
                        }
                        self.add_audit(AuditEventType::DocumentIndexed);
                    }
                    ai_assistant::IndexingProgress::AllComplete { .. } => {
                        self.indexing_display = None;
                        // Index into knowledge graph if enabled
                        if self.settings.graph_enabled {
                            self.index_pending_graph_docs();
                        }
                    }
                    ai_assistant::IndexingProgress::Error { source, error } => {
                        for ks in &mut self.knowledge_sources {
                            if ks.name == source {
                                ks.status = KnowledgeStatus::Error(error.clone());
                            }
                        }
                        self.add_toast(&format!("Indexing error: {}", error), true);
                    }
                }
            }
        }
    }

    fn apply_scan_result(&mut self, result: ScanResult) {
        if result.report.llm_providers.is_empty() {
            self.phase = AppPhase::NoProviders(
                "No local LLM providers detected.\n\n\
                 To get started:\n\
                 1. Install Ollama from https://ollama.ai\n\
                 2. Run: ollama pull llama3\n\
                 3. Click 'Retry Scan' below"
                    .to_string(),
            );
        } else {
            self.assistant.load_config(result.config.clone());

            // Build unified model list from ALL detected providers
            let mut unified_models = Vec::new();
            for provider in &result.report.llm_providers {
                for model_name in &provider.available_models {
                    unified_models.push(ModelInfo::new(
                        model_name.clone(),
                        provider.provider_type.clone(),
                    ));
                }
            }

            if !unified_models.is_empty() {
                // Auto-select first model and set its provider/URL
                self.selected_model = unified_models[0].name.clone();
                self.assistant.config.selected_model = unified_models[0].name.clone();
                self.assistant.config.provider = unified_models[0].provider.clone();
                Self::apply_provider_url_from_report(
                    &mut self.assistant,
                    &result.report,
                    &unified_models[0].provider,
                );
                self.assistant.available_models = unified_models;
            } else {
                // Providers detected but no models listed — try HTTP fetch
                self.assistant.fetch_models();
            }

            self.phase = AppPhase::Ready;

            // Auto-create a default session if none exists
            if self.assistant.current_session.is_none() {
                self.assistant.new_session();
            }
        }

        self.settings.ollama_url = result.config.ollama_url.clone();
        self.settings.lm_studio_url = result.config.lm_studio_url.clone();
        self.scan_rx = None;
        self.scan_result_data = Some(result);
    }

    /// Apply the correct URL for a provider based on scan report data.
    fn apply_provider_url_from_report(
        assistant: &mut AiAssistant,
        report: &EnvironmentReport,
        provider: &ai_assistant::AiProvider,
    ) {
        for p in &report.llm_providers {
            if p.provider_type == *provider {
                match p.provider_type {
                    ai_assistant::AiProvider::Ollama => assistant.config.ollama_url = p.url.clone(),
                    ai_assistant::AiProvider::LMStudio => assistant.config.lm_studio_url = p.url.clone(),
                    _ => assistant.config.custom_url = p.url.clone(),
                }
                break;
            }
        }
    }

    fn retry_scan(&mut self) {
        self.phase = AppPhase::Scanning;
        let (tx, rx) = mpsc::channel();
        let ollama_url = self.settings.ollama_url.clone();
        let lm_studio_url = self.settings.lm_studio_url.clone();
        std::thread::spawn(move || {
            // Set env vars for detection
            std::env::set_var("OLLAMA_HOST", &ollama_url);
            std::env::set_var("LM_STUDIO_URL", &lm_studio_url);
            let mut butler = Butler::new();
            let report = butler.scan();
            let config = butler.suggest_config(&report);
            let advisor = butler.advise(&report);
            let _ = tx.send(ScanResult {
                report,
                config,
                advisor,
            });
        });
        self.scan_rx = Some(rx);
    }

    // =========================================================================
    // Chat
    // =========================================================================

    fn send_message(&mut self, text: String) {
        if text.is_empty() || self.assistant.is_generating {
            return;
        }
        self.last_error = None;

        // Auto-create session if none exists
        if self.assistant.current_session.is_none() {
            self.assistant.new_session();
        }

        if self.rag_enabled && !self.knowledge_sources.is_empty() {
            let (knowledge_ctx, _conversation_ctx) = self.assistant.build_rag_context(&text);
            self.last_knowledge_ctx = Some(knowledge_ctx.clone());
            self.assistant
                .send_message(text, &knowledge_ctx);
        } else {
            self.last_knowledge_ctx = None;
            self.assistant.send_message_simple(text);
        }

        self.add_audit(AuditEventType::MessageSent);
        self.scroll_to_bottom = true;
    }

    fn capture_response_detail(&mut self) {
        let metrics_list = self.assistant.metrics.get_message_metrics();
        let last_metrics = metrics_list.last();

        let mut knowledge_sources = Vec::new();
        let mut knowledge_tokens = 0;
        if let Some(ref usage) = self.assistant.last_knowledge_usage {
            knowledge_tokens = usage.total_tokens;
            for src in &usage.sources {
                knowledge_sources.push((
                    src.source.clone(),
                    src.chunks_used,
                    src.relevance_score,
                ));
            }
        }

        let thinking = self
            .assistant
            .last_thinking_result
            .as_ref()
            .and_then(|t| t.thinking.clone());

        let detail = ResponseDetail {
            model: last_metrics
                .map(|m| m.model.clone())
                .unwrap_or_else(|| self.assistant.config.selected_model.clone()),
            response_time_ms: last_metrics.map(|m| m.total_response_time_ms).unwrap_or(0),
            ttft_ms: last_metrics.and_then(|m| m.time_to_first_token_ms),
            input_tokens: last_metrics.map(|m| m.input_tokens).unwrap_or(0),
            output_tokens: last_metrics.map(|m| m.output_tokens).unwrap_or(0),
            knowledge_context: self.last_knowledge_ctx.take(),
            knowledge_sources,
            knowledge_tokens,
            thinking,
            context_near_limit: last_metrics.map(|m| m.context_near_limit).unwrap_or(false),
        };
        self.response_details.push(detail);
    }

    // =========================================================================
    // File loading
    // =========================================================================

    fn load_knowledge_files(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Knowledge Files", &["kpkg", "md", "txt"])
            .add_filter("All Files", &["*"]);

        if let Some(ref dir) = self.last_dir {
            dialog = dialog.set_directory(dir);
        }

        if let Some(files) = dialog.pick_files() {
            if let Some(parent) = files.first().and_then(|f| f.parent()) {
                self.last_dir = Some(parent.to_path_buf());
            }

            for path in files {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                match ext.as_str() {
                    "kpkg" => self.load_kpkg_file(&path),
                    "md" | "txt" => self.load_text_file(&path),
                    _ => self.add_toast(&format!("Unsupported: {}", ext), true),
                }
            }

            self.assistant.start_background_indexing();
        }
    }

    fn load_kpkg_file(&mut self, path: &Path) {
        match std::fs::read(path) {
            Ok(data) => {
                let reader = KpkgReader::<AppKeyProvider>::with_app_key();
                match reader.read(&data) {
                    Ok(docs) => {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("package")
                            .to_string();
                        let count = docs.len();
                        for doc in &docs {
                            let source = format!("kpkg:{}:{}", name, doc.path);
                            self.assistant
                                .register_knowledge_document(&source, &doc.content);
                            self.pending_graph_docs
                                .push((source, doc.content.clone()));
                        }
                        self.knowledge_sources.push(KnowledgeSourceInfo {
                            name: name.clone(),
                            file_path: path.display().to_string(),
                            is_kpkg: true,
                            doc_count: count,
                            status: KnowledgeStatus::Pending,
                        });
                        self.add_toast(
                            &format!("Loaded {} ({} docs)", name, count),
                            false,
                        );
                    }
                    Err(e) => self.add_toast(&format!("Decrypt error: {}", e), true),
                }
            }
            Err(e) => self.add_toast(&format!("Read error: {}", e), true),
        }
    }

    fn load_text_file(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("file")
                    .to_string();
                self.assistant
                    .register_knowledge_document(&name, &content);
                self.pending_graph_docs
                    .push((name.clone(), content));
                self.knowledge_sources.push(KnowledgeSourceInfo {
                    name: name.clone(),
                    file_path: path.display().to_string(),
                    is_kpkg: false,
                    doc_count: 1,
                    status: KnowledgeStatus::Pending,
                });
                self.add_toast(&format!("Loaded {}", name), false);
            }
            Err(e) => self.add_toast(&format!("Read error: {}", e), true),
        }
    }

    // =========================================================================
    // Knowledge graph
    // =========================================================================

    fn init_knowledge_graph(&mut self) {
        if self.knowledge_graph.is_some() {
            return;
        }
        let kg_path = self.data_dir.join("knowledge_graph.db");
        match KnowledgeGraph::open(&kg_path, KnowledgeGraphConfig::default()) {
            Ok(kg) => {
                self.knowledge_graph = Some(kg);
            }
            Err(e) => {
                self.add_toast(&format!("Graph init error: {}", e), true);
            }
        }
    }

    fn index_pending_graph_docs(&mut self) {
        if !self.settings.graph_enabled {
            return;
        }
        if self.knowledge_graph.is_none() {
            self.init_knowledge_graph();
        }
        let docs = std::mem::take(&mut self.pending_graph_docs);
        if let Some(ref mut kg) = self.knowledge_graph {
            for (source, content) in &docs {
                let _ = kg.index_document(source, content, &self.pattern_extractor);
            }
            self.rebuild_graph_viz();
        }
    }

    fn rebuild_graph_viz(&mut self) {
        let Some(ref kg) = self.knowledge_graph else {
            return;
        };
        let json = match kg.export_json() {
            Ok(j) => j,
            Err(_) => return,
        };
        let stats = kg.stats().ok();

        let mut graph = PetGraph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        // Parse entities
        if let Some(entities) = json.get("entities").and_then(|e| e.as_array()) {
            for e in entities {
                let name = e
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("?")
                    .to_string();
                let etype = e
                    .get("entity_type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("Concept")
                    .to_string();
                let mentions = e
                    .get("mention_count")
                    .and_then(|m| m.as_u64())
                    .unwrap_or(1) as usize;
                let idx = graph.add_node(GraphNode {
                    name: name.clone(),
                    entity_type: etype,
                    mention_count: mentions,
                });
                node_map.insert(name, idx);
            }
        }

        // Parse relations
        if let Some(relations) = json.get("relations").and_then(|r| r.as_array()) {
            for r in relations {
                let from = r
                    .get("from_entity")
                    .and_then(|f| f.as_str())
                    .unwrap_or("");
                let to = r
                    .get("to_entity")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                let rtype = r
                    .get("relation_type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("related")
                    .to_string();
                let weight = r
                    .get("weight")
                    .and_then(|w| w.as_f64())
                    .unwrap_or(1.0) as f32;
                if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(from), node_map.get(to)) {
                    graph.add_edge(
                        from_idx,
                        to_idx,
                        GraphEdge {
                            relation_type: rtype,
                            weight,
                        },
                    );
                }
            }
        }

        self.graph_viz = Some(GraphVisualizationData {
            graph,
            positions: HashMap::new(),
            needs_layout: true,
            stats,
        });
    }

    // =========================================================================
    // Analysis (lazy)
    // =========================================================================

    fn update_analysis_if_needed(&mut self) {
        let msg_count = self.assistant.conversation.len();
        if msg_count == self.analysis_msg_count || msg_count < 2 {
            return;
        }
        self.analysis_msg_count = msg_count;

        let analyzer = SentimentAnalyzer::new();
        self.sentiment_cache = Some(analyzer.analyze_conversation(&self.assistant.conversation));

        let detector = TopicDetector::new();
        self.topics_cache = detector.detect_topics(&self.assistant.conversation);

        let summarizer = SessionSummarizer::new(SummaryConfig::default());
        self.summary_cache = Some(summarizer.summarize(&self.assistant.conversation));
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    fn save_sessions(&self) {
        let path = self.data_dir.join("sessions.json");
        let _ = self.assistant.save_sessions_to_file(&path);
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    fn render_scanning_screen(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() / 3.0);
                ui.spinner();
                ui.add_space(16.0);
                ui.heading("Scanning for LLM providers...");
                ui.label("Checking Ollama, LM Studio, and cloud APIs");
            });
        });
    }

    fn render_no_providers_screen(&mut self, ctx: &egui::Context, msg: String) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() / 4.0);
                ui.heading("No LLM Providers Found");
                ui.add_space(16.0);
                ui.label(&msg);
                ui.add_space(24.0);

                let field_width = (ui.available_width() * 0.6).min(400.0);
                ui.allocate_ui(egui::vec2(field_width, 0.0), |ui| {
                    egui::Grid::new("provider_urls_grid")
                        .num_columns(2)
                        .spacing([8.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Ollama URL:");
                            ui.add_sized(
                                [ui.available_width(), 20.0],
                                egui::TextEdit::singleline(&mut self.settings.ollama_url),
                            );
                            ui.end_row();
                            ui.label("LM Studio URL:");
                            ui.add_sized(
                                [ui.available_width(), 20.0],
                                egui::TextEdit::singleline(&mut self.settings.lm_studio_url),
                            );
                            ui.end_row();
                        });
                });
                ui.add_space(16.0);

                if ui.button("Retry Scan").clicked() {
                    self.retry_scan();
                }
            });
        });
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Model selector
                if let Some(model) = widgets::model_selector(
                    ui,
                    &mut self.selected_model,
                    &self.assistant.available_models,
                ) {
                    self.assistant.config.selected_model = model.name.clone();
                    self.assistant.config.provider = model.provider.clone();
                    self.selected_model = model.name;
                    // Auto-adjust provider URL
                    if let Some(ref scan) = self.scan_result_data {
                        Self::apply_provider_url_from_report(
                            &mut self.assistant,
                            &scan.report,
                            &model.provider,
                        );
                    }
                }

                ui.separator();

                // Connection status
                widgets::connection_status(
                    ui,
                    self.assistant.is_fetching_models,
                    self.assistant.available_models.len(),
                );

                ui.separator();

                // Context usage
                let usage = ContextUsage::calculate(
                    0,
                    0,
                    self.assistant.conversation.len() * 100,
                    self.assistant.config.max_history_messages * 500,
                );
                widgets::context_usage_bar(ui, &usage, 150.0);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Settings button
                    if ui.button("Settings").clicked() {
                        self.show_settings = !self.show_settings;
                    }

                    // Monitor toggle
                    let monitor_label = if self.show_monitor {
                        "Hide Monitor"
                    } else {
                        "Monitor"
                    };
                    if ui.button(monitor_label).clicked() {
                        self.show_monitor = !self.show_monitor;
                    }
                });
            });
        });
    }

    fn render_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar")
            .default_width(220.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                // Tab buttons
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::Sessions, "Sessions");
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::Knowledge, "Knowledge");
                    ui.selectable_value(&mut self.sidebar_tab, SidebarTab::Butler, "Butler");
                });
                ui.separator();

                match self.sidebar_tab {
                    SidebarTab::Sessions => self.render_sessions_tab(ui),
                    SidebarTab::Knowledge => self.render_knowledge_tab(ui),
                    SidebarTab::Butler => self.render_butler_tab(ui),
                }
            });
    }

    fn render_sessions_tab(&mut self, ui: &mut Ui) {
        if ui.button("+ New Chat").clicked() {
            self.assistant.new_session();
            self.add_audit(AuditEventType::SessionCreated);
        }
        ui.add_space(8.0);

        let current_id = self
            .assistant
            .current_session
            .as_ref()
            .map(|s| s.id.as_str());
        let response = widgets::session_list(
            ui,
            &self.assistant.session_store.sessions,
            current_id,
            ui.available_height(),
        );

        if let Some(id) = response.session_to_load {
            self.assistant.load_session(&id);
            self.analysis_msg_count = 0; // invalidate caches
            self.add_audit(AuditEventType::SessionLoaded);
        }
        if let Some(id) = response.session_to_delete {
            self.assistant.session_store.sessions.retain(|s| s.id != id);
            self.add_audit(AuditEventType::SessionDeleted);
        }
    }

    fn render_knowledge_tab(&mut self, ui: &mut Ui) {
        if ui.button("Load Files").clicked() {
            self.load_knowledge_files();
        }
        ui.add_space(8.0);

        // Knowledge sources list
        for ks in &self.knowledge_sources {
            let icon = match &ks.status {
                KnowledgeStatus::Pending => "...",
                KnowledgeStatus::Indexing => ">>>",
                KnowledgeStatus::Indexed => "[OK]",
                KnowledgeStatus::Error(_) => "[!!]",
            };
            ui.horizontal(|ui| {
                ui.label(icon);
                ui.label(&ks.name);
                ui.label(format!("({})", ks.doc_count));
            });
        }

        if !self.knowledge_sources.is_empty() {
            ui.add_space(8.0);

            // Indexing progress
            let pending = self
                .knowledge_sources
                .iter()
                .filter(|k| k.status == KnowledgeStatus::Pending)
                .count();
            widgets::pending_documents_indicator(ui, pending, self.assistant.is_indexing);

            if let Some((ref doc, current, total)) = self.indexing_display {
                widgets::indexing_progress(ui, doc, current, total);
            }
        }

        ui.add_space(8.0);
        ui.separator();

        // RAG toggle
        ui.checkbox(&mut self.rag_enabled, "Use knowledge context");

        if self.rag_enabled {
            let status = RagStatus {
                rag_available: self.assistant.rag_db.is_some(),
                knowledge_enabled: true,
                conversation_enabled: false,
                knowledge_chunks: self
                    .knowledge_sources
                    .iter()
                    .filter(|k| k.status == KnowledgeStatus::Indexed)
                    .map(|k| k.doc_count)
                    .sum(),
                knowledge_tokens: 0,
                archived_messages: 0,
                archived_tokens: 0,
            };
            widgets::rag_status_compact(ui, &status);
        }

        ui.add_space(8.0);
        ui.separator();

        // Knowledge graph toggle
        let prev = self.settings.graph_enabled;
        ui.checkbox(&mut self.settings.graph_enabled, "Knowledge Graph");
        if self.settings.graph_enabled && !prev {
            self.init_knowledge_graph();
        }
    }

    fn render_butler_tab(&mut self, ui: &mut Ui) {
        // Runtime info
        if let Some(ref result) = self.scan_result_data {
            let rt = &result.report.runtime;
            ui.heading("Environment");
            ui.label(format!("OS: {} ({})", rt.os, rt.arch));
            ui.label(format!("CPUs: {}", rt.cpus));
            ui.label(format!(
                "GPU: {}",
                if rt.has_gpu { "Detected" } else { "None" }
            ));
            ui.label(format!(
                "Docker: {}",
                if rt.has_docker { "Available" } else { "No" }
            ));

            ui.add_space(8.0);
            ui.separator();

            // Detected providers
            ui.heading("Providers");
            if result.report.llm_providers.is_empty() {
                ui.label("None detected");
            } else {
                for p in &result.report.llm_providers {
                    ui.label(format!(
                        "{} ({} models)",
                        p.name,
                        p.available_models.len()
                    ));
                    ui.label(format!("  URL: {}", p.url));
                }
            }

            ui.add_space(8.0);
            ui.separator();
        }

        // Re-scan button
        if ui.button("Re-scan Environment").clicked() {
            self.retry_scan();
        }

        ui.add_space(8.0);
        ui.separator();

        // Advisor recommendations
        if let Some(ref report) = self.scan_result_data.as_ref().map(|s| &s.advisor) {
            ui.heading("Recommendations");
            if report.recommendations.is_empty() {
                ui.label("All good! No recommendations.");
            } else {
                egui::ScrollArea::vertical()
                    .max_height(ui.available_height() - 30.0)
                    .show(ui, |ui| {
                        for rec in &report.recommendations {
                            let color = match rec.priority {
                                RecommendationPriority::Critical => Color32::from_rgb(220, 50, 50),
                                RecommendationPriority::High => Color32::from_rgb(220, 140, 30),
                                RecommendationPriority::Medium => Color32::from_rgb(200, 200, 50),
                                RecommendationPriority::Low => Color32::GRAY,
                            };
                            ui.horizontal(|ui| {
                                ui.colored_label(color, format!("[{:?}]", rec.priority));
                                ui.label(&rec.title);
                            });
                            ui.label(&rec.description);
                            if let Some(ref flag) = rec.feature_flag {
                                ui.small(format!("Feature: {}", flag));
                            }
                            ui.add_space(4.0);
                        }
                    });
            }
        }
    }

    fn render_chat_area(&mut self, ui: &mut Ui) {
        let available = ui.available_size();
        let chat_height = if self.show_monitor {
            available.y * 0.65
        } else {
            available.y - 60.0
        };

        // Chat messages area
        egui::ScrollArea::vertical()
            .max_height(chat_height)
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                let max_width = ui.available_width() - 20.0;

                if self.assistant.conversation.is_empty() && !self.assistant.is_generating {
                    // Welcome screen
                    let suggestions = &[
                        "Tell me about Rust ownership",
                        "Summarize this document",
                        "Help me write a function",
                        "Explain async/await",
                    ];
                    if let Some(suggestion) = widgets::welcome_screen(
                        ui,
                        "AI Assistant",
                        "Select a model and start chatting. Load knowledge files for RAG.",
                        suggestions,
                    ) {
                        self.input_text = suggestion;
                    }
                } else {
                    // Render messages with details buttons
                    let mut detail_idx = 0usize;
                    for (_msg_idx, msg) in self.assistant.conversation.iter().enumerate() {
                        widgets::chat_message(ui, msg, &self.colors, max_width);

                        // Add details button for assistant messages
                        if msg.role == "assistant" {
                            let has_detail = detail_idx < self.response_details.len();
                            if has_detail {
                                let is_expanded = self.expanded_details.contains(&detail_idx);
                                let btn_label = if is_expanded {
                                    "Hide details"
                                } else {
                                    "Details"
                                };
                                if ui.small_button(btn_label).clicked() {
                                    if is_expanded {
                                        self.expanded_details.remove(&detail_idx);
                                    } else {
                                        self.expanded_details.insert(detail_idx);
                                    }
                                }

                                if is_expanded {
                                    self.render_response_detail(ui, detail_idx);
                                }
                            }
                            detail_idx += 1;
                        }

                        ui.add_space(4.0);
                    }

                    // Streaming response
                    if self.assistant.is_generating {
                        if self.assistant.current_response.is_empty() {
                            widgets::thinking_indicator(ui, &self.colors);
                        } else {
                            widgets::streaming_response(
                                ui,
                                &self.assistant.current_response,
                                &self.colors,
                                max_width,
                            );
                        }
                    }

                    // Error
                    if let Some(ref err) = self.last_error {
                        widgets::error_message(ui, err, &self.colors);
                    }
                }
            });

        ui.add_space(4.0);

        // Input area
        if self.assistant.is_generating {
            ui.horizontal(|ui| {
                let _response = widgets::chat_input_multiline(
                    ui,
                    &mut self.input_text,
                    true,
                    "Generating...",
                    50.0,
                );
                if ui.button("Stop").clicked() {
                    self.add_toast("Generation will stop after current chunk", false);
                }
            });
        } else {
            let response = widgets::chat_input_multiline(
                ui,
                &mut self.input_text,
                false,
                "Type your message... (Ctrl+Enter to send)",
                50.0,
            );
            if let Some(text) = response {
                self.send_message(text);
            }
        }
    }

    fn render_response_detail(&self, ui: &mut Ui, idx: usize) {
        let Some(detail) = self.response_details.get(idx) else {
            return;
        };

        egui::Frame::none()
            .fill(Color32::from_gray(30))
            .rounding(6.0)
            .inner_margin(egui::Margin::same(8.0))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());

                // Performance metrics
                ui.horizontal(|ui| {
                    ui.colored_label(Color32::from_rgb(120, 180, 255), "Model:");
                    ui.label(&detail.model);
                    ui.separator();
                    ui.colored_label(Color32::from_rgb(120, 180, 255), "Time:");
                    ui.label(format!("{}ms", detail.response_time_ms));
                    if let Some(ttft) = detail.ttft_ms {
                        ui.label(format!("(TTFT: {}ms)", ttft));
                    }
                    ui.separator();
                    ui.colored_label(Color32::from_rgb(120, 180, 255), "Tokens:");
                    ui.label(format!("in:{} out:{}", detail.input_tokens, detail.output_tokens));
                });

                // Context limit warning
                if detail.context_near_limit {
                    ui.colored_label(
                        Color32::from_rgb(255, 180, 50),
                        "Warning: context was near limit",
                    );
                }

                // Knowledge sources
                if !detail.knowledge_sources.is_empty() {
                    ui.add_space(4.0);
                    ui.colored_label(
                        Color32::from_rgb(120, 220, 120),
                        format!("Knowledge ({} tokens):", detail.knowledge_tokens),
                    );
                    for (source, chunks, relevance) in &detail.knowledge_sources {
                        let rel_str = relevance
                            .map(|r| format!(" (relevance: {:.2})", r))
                            .unwrap_or_default();
                        ui.label(format!("  {} — {} chunks{}", source, chunks, rel_str));
                    }
                }

                // Knowledge context (collapsible)
                if let Some(ref ctx_text) = detail.knowledge_context {
                    if !ctx_text.is_empty() {
                        ui.add_space(4.0);
                        ui.collapsing("Knowledge context sent", |ui| {
                            egui::ScrollArea::vertical()
                                .max_height(150.0)
                                .show(ui, |ui| {
                                    ui.monospace(ctx_text);
                                });
                        });
                    }
                }

                // Thinking/reasoning
                if let Some(ref thinking) = detail.thinking {
                    ui.add_space(4.0);
                    ui.collapsing("Reasoning (think tags)", |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(150.0)
                            .show(ui, |ui| {
                                ui.monospace(thinking);
                            });
                    });
                }
            });
    }

    fn render_monitor_panel(&mut self, ctx: &egui::Context) {
        if !self.show_monitor {
            return;
        }

        egui::TopBottomPanel::bottom("monitor")
            .min_height(150.0)
            .default_height(200.0)
            .resizable(true)
            .show(ctx, |ui| {
                // Tab bar
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.monitor_tab, MonitorTab::Overview, "Overview");
                    ui.selectable_value(&mut self.monitor_tab, MonitorTab::Metrics, "Metrics");
                    ui.selectable_value(&mut self.monitor_tab, MonitorTab::Analysis, "Analysis");
                    ui.selectable_value(&mut self.monitor_tab, MonitorTab::Graph, "Graph");
                    ui.selectable_value(&mut self.monitor_tab, MonitorTab::Audit, "Audit");
                });
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.monitor_tab {
                        MonitorTab::Overview => self.render_monitor_overview(ui),
                        MonitorTab::Metrics => self.render_monitor_metrics(ui),
                        MonitorTab::Analysis => self.render_monitor_analysis(ui),
                        MonitorTab::Graph => self.render_monitor_graph(ui),
                        MonitorTab::Audit => self.render_monitor_audit(ui),
                    }
                });
            });
    }

    fn render_monitor_overview(&mut self, ui: &mut Ui) {
        let metrics = self.assistant.metrics.get_session_metrics();
        widgets::session_metrics_compact(ui, &metrics);

        ui.add_space(8.0);

        if let Some(ref sentiment) = self.sentiment_cache {
            widgets::sentiment_badge(ui, &sentiment.overall.sentiment);
        }

        if !self.topics_cache.is_empty() {
            ui.horizontal(|ui| {
                ui.label("Topics:");
                for topic in self.topics_cache.iter().take(5) {
                    ui.label(format!("{} ({:.0}%)", topic.name, topic.relevance * 100.0));
                }
            });
        }
    }

    fn render_monitor_metrics(&mut self, ui: &mut Ui) {
        let session_metrics = self.assistant.metrics.get_session_metrics();
        widgets::session_metrics_panel(ui, &session_metrics);

        ui.add_space(8.0);

        let rag_metrics = self.assistant.metrics.get_rag_quality_metrics();
        widgets::rag_quality_metrics_panel(ui, &rag_metrics);

        ui.add_space(8.0);

        let msg_metrics = self.assistant.metrics.get_message_metrics();
        widgets::advanced_metrics_panel(ui, msg_metrics);
    }

    fn render_monitor_analysis(&mut self, ui: &mut Ui) {
        self.update_analysis_if_needed();

        if self.assistant.conversation.len() < 2 {
            ui.label("Send at least 2 messages to see analysis.");
            return;
        }

        if let Some(ref analysis) = self.sentiment_cache {
            widgets::sentiment_analysis_panel(ui, &analysis.overall);
        }

        ui.add_space(8.0);

        if !self.topics_cache.is_empty() {
            widgets::topics_panel(ui, &self.topics_cache);
        }

        ui.add_space(8.0);

        if let Some(ref summary) = self.summary_cache {
            widgets::session_summary_panel(ui, summary);
        }
    }

    fn render_monitor_graph(&mut self, ui: &mut Ui) {
        if !self.settings.graph_enabled {
            ui.label("Enable 'Knowledge Graph' in the Knowledge tab to visualize.");
            return;
        }

        let Some(ref mut viz) = self.graph_viz else {
            ui.label("Load and index documents to build the graph.");
            return;
        };

        // Stats
        if let Some(ref stats) = viz.stats {
            ui.horizontal(|ui| {
                ui.label(format!("Entities: {}", stats.total_entities));
                ui.label(format!("Relations: {}", stats.total_relations));
                ui.label(format!("Chunks: {}", stats.total_chunks));
            });
            ui.separator();
        }

        if viz.graph.node_count() == 0 {
            ui.label("Graph is empty. Index more documents.");
            return;
        }

        // Render graph
        let size = ui.available_size().min(Vec2::new(800.0, 400.0));
        let (response, painter) = ui.allocate_painter(size, Sense::hover());
        let rect = response.rect;

        // Layout if needed
        if viz.needs_layout {
            layout_graph(viz, rect);
            viz.needs_layout = false;
        }

        // Draw edges
        for edge_idx in viz.graph.edge_indices() {
            if let Some((from, to)) = viz.graph.edge_endpoints(edge_idx) {
                if let (Some(&from_pos), Some(&to_pos)) =
                    (viz.positions.get(&from), viz.positions.get(&to))
                {
                    painter.line_segment(
                        [from_pos, to_pos],
                        Stroke::new(1.0, Color32::from_gray(100)),
                    );
                    let mid = Pos2::new(
                        (from_pos.x + to_pos.x) / 2.0,
                        (from_pos.y + to_pos.y) / 2.0,
                    );
                    let edge = &viz.graph[edge_idx];
                    painter.text(
                        mid,
                        Align2::CENTER_CENTER,
                        &edge.relation_type,
                        egui::FontId::proportional(9.0),
                        Color32::from_gray(140),
                    );
                }
            }
        }

        // Draw nodes
        for node_idx in viz.graph.node_indices() {
            if let Some(&pos) = viz.positions.get(&node_idx) {
                let node = &viz.graph[node_idx];
                let color = entity_type_color(&node.entity_type);
                let radius = 10.0 + (node.mention_count as f32).min(8.0);

                painter.circle_filled(pos, radius, color);
                painter.circle_stroke(pos, radius, Stroke::new(1.0, Color32::WHITE));
                painter.text(
                    Pos2::new(pos.x, pos.y + radius + 6.0),
                    Align2::CENTER_TOP,
                    &node.name,
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }
        }
    }

    fn render_monitor_audit(&self, ui: &mut Ui) {
        if self.audit_events.is_empty() {
            ui.label("No audit events yet.");
        } else {
            widgets::audit_log_panel(ui, &self.audit_events, 50);
        }
    }

    fn render_settings(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }

        let mut open = true;
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Provider URLs");
                ui.horizontal(|ui| {
                    ui.label("Ollama:");
                    ui.text_edit_singleline(&mut self.settings.ollama_url);
                });
                ui.horizontal(|ui| {
                    ui.label("LM Studio:");
                    ui.text_edit_singleline(&mut self.settings.lm_studio_url);
                });

                ui.add_space(12.0);
                ui.heading("Generation");
                ui.add(
                    egui::Slider::new(&mut self.settings.temperature, 0.0..=2.0)
                        .text("Temperature"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.max_history, 5..=50)
                        .text("History depth"),
                );

                ui.add_space(12.0);

                if ui.button("Apply").clicked() {
                    self.assistant.config.ollama_url = self.settings.ollama_url.clone();
                    self.assistant.config.lm_studio_url = self.settings.lm_studio_url.clone();
                    self.assistant.config.temperature = self.settings.temperature;
                    self.assistant.config.max_history_messages = self.settings.max_history;
                    self.assistant.fetch_models();
                    self.add_audit(AuditEventType::ConfigChanged);
                    self.add_toast("Settings applied", false);
                }
            });
        if !open {
            self.show_settings = false;
        }
    }

    fn render_toasts(&mut self, ctx: &egui::Context) {
        // Remove expired toasts (5 seconds)
        self.toasts
            .retain(|(_, _, created)| created.elapsed().as_secs() < 5);

        if self.toasts.is_empty() {
            return;
        }

        egui::Area::new(egui::Id::new("toasts"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
            .show(ctx, |ui| {
                for (msg, is_error, _) in &self.toasts {
                    let color = if *is_error {
                        Color32::from_rgb(200, 60, 60)
                    } else {
                        Color32::from_rgb(60, 140, 60)
                    };
                    egui::Frame::none()
                        .fill(color)
                        .rounding(6.0)
                        .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                        .show(ui, |ui| {
                            ui.colored_label(Color32::WHITE, msg.as_str());
                        });
                    ui.add_space(4.0);
                }
            });
    }
}

// =============================================================================
// eframe::App implementation
// =============================================================================

impl eframe::App for AiGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background tasks
        self.poll_all();

        // Request repaint while tasks are active
        if self.assistant.is_generating
            || self.assistant.is_fetching_models
            || self.assistant.is_indexing
            || self.scan_rx.is_some()
        {
            ctx.request_repaint();
        }

        // Also request repaint if toasts are visible (for auto-dismiss)
        if !self.toasts.is_empty() {
            ctx.request_repaint();
        }

        // Phase-based rendering
        match self.phase.clone() {
            AppPhase::Scanning => {
                self.render_scanning_screen(ctx);
            }
            AppPhase::NoProviders(msg) => {
                self.render_no_providers_screen(ctx, msg);
            }
            AppPhase::Ready => {
                self.render_top_bar(ctx);
                self.render_sidebar(ctx);
                self.render_monitor_panel(ctx);
                self.render_settings(ctx);

                egui::CentralPanel::default().show(ctx, |ui| {
                    self.render_chat_area(ui);
                });
            }
        }

        self.render_toasts(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.assistant.save_current_session();
        self.save_sessions();
    }
}

// =============================================================================
// Graph layout (Fruchterman-Reingold)
// =============================================================================

fn layout_graph(viz: &mut GraphVisualizationData, rect: Rect) {
    let n = viz.graph.node_count();
    if n == 0 {
        return;
    }

    let area = rect.width() * rect.height();
    let k = (area / n as f32).sqrt() * 0.6;
    let center = rect.center();

    // Initialize positions in a circle
    let node_indices: Vec<NodeIndex> = viz.graph.node_indices().collect();
    for (i, &idx) in node_indices.iter().enumerate() {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
        let r = (rect.width().min(rect.height()) / 3.0).min(200.0);
        viz.positions.insert(
            idx,
            Pos2::new(center.x + r * angle.cos(), center.y + r * angle.sin()),
        );
    }

    // Fruchterman-Reingold iterations
    let iterations = 50;
    let mut temperature = rect.width().min(rect.height()) / 4.0;
    let cooling = temperature / iterations as f32;

    for _ in 0..iterations {
        let mut displacements: HashMap<NodeIndex, Vec2> = HashMap::new();
        for &idx in &node_indices {
            displacements.insert(idx, Vec2::ZERO);
        }

        // Repulsive forces (between all node pairs)
        for i in 0..node_indices.len() {
            for j in (i + 1)..node_indices.len() {
                let a = node_indices[i];
                let b = node_indices[j];
                let pa = viz.positions[&a];
                let pb = viz.positions[&b];
                let delta = pa - pb;
                let dist = delta.length().max(1.0);
                let force = k * k / dist;
                let disp = delta / dist * force;

                *displacements.get_mut(&a).unwrap() += disp;
                *displacements.get_mut(&b).unwrap() -= disp;
            }
        }

        // Attractive forces (along edges)
        for edge_idx in viz.graph.edge_indices() {
            if let Some((from, to)) = viz.graph.edge_endpoints(edge_idx) {
                let pa = viz.positions[&from];
                let pb = viz.positions[&to];
                let delta = pa - pb;
                let dist = delta.length().max(1.0);
                let force = dist * dist / k;
                let disp = delta / dist * force;

                *displacements.get_mut(&from).unwrap() -= disp;
                *displacements.get_mut(&to).unwrap() += disp;
            }
        }

        // Apply displacements with temperature limit
        for &idx in &node_indices {
            let disp = displacements[&idx];
            let len = disp.length().max(1.0);
            let clamped = disp / len * len.min(temperature);
            let pos = viz.positions.get_mut(&idx).unwrap();
            *pos += clamped;
            // Clamp to rect with margin
            pos.x = pos.x.clamp(rect.left() + 20.0, rect.right() - 20.0);
            pos.y = pos.y.clamp(rect.top() + 20.0, rect.bottom() - 20.0);
        }

        temperature -= cooling;
    }
}

fn entity_type_color(etype: &str) -> Color32 {
    match etype {
        "Person" => Color32::from_rgb(100, 149, 237),    // Cornflower blue
        "Organization" => Color32::from_rgb(60, 179, 113), // Medium sea green
        "Concept" => Color32::from_rgb(147, 112, 219),   // Medium purple
        "Location" => Color32::from_rgb(255, 165, 0),    // Orange
        "Product" => Color32::from_rgb(0, 206, 209),     // Dark turquoise
        "Event" => Color32::from_rgb(255, 215, 0),       // Gold
        _ => Color32::from_rgb(169, 169, 169),           // Dark gray
    }
}

// =============================================================================
// Entry point
// =============================================================================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("AI Assistant"),
        ..Default::default()
    };

    eframe::run_native(
        "AI Assistant",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::new(AiGuiApp::new())
        }),
    )
}
