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
    // Scheduler
    CronExpression,
    // Sandbox
    CodeSandbox, SandboxLanguage,
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
enum SidebarCategory {
    Chat, Agents, Knowledge, Generate, Optimize, System,
}

impl SidebarCategory {
    fn label(&self) -> &'static str {
        match self { Self::Chat => "CHAT", Self::Agents => "AGENTS", Self::Knowledge => "KNOWLEDGE", Self::Generate => "GENERATE", Self::Optimize => "OPTIMIZE", Self::System => "SYSTEM" }
    }
    fn all() -> &'static [Self] { &[Self::Chat, Self::Agents, Self::Knowledge, Self::Generate, Self::Optimize, Self::System] }
}

#[derive(PartialEq, Clone, Copy)]
enum SidebarPanel {
    Chat, Sessions,
    AgentPool, ToolsMcp, Automation,
    RagSources, Memory, CloudStorage,
    Media, Audio,
    PromptLab, Evaluation,
    Security, Analytics, DevTools, Cluster, Butler, Settings,
}

impl SidebarPanel {
    fn label(&self) -> &'static str {
        match self {
            Self::Chat => "Chat", Self::Sessions => "Sessions",
            Self::AgentPool => "Agent Pool", Self::ToolsMcp => "Tools / MCP", Self::Automation => "Automation",
            Self::RagSources => "RAG Sources", Self::Memory => "Memory", Self::CloudStorage => "Cloud Storage",
            Self::Media => "Media", Self::Audio => "Audio",
            Self::PromptLab => "Prompt Lab", Self::Evaluation => "Evaluation",
            Self::Security => "Security", Self::Analytics => "Analytics", Self::DevTools => "DevTools",
            Self::Cluster => "Cluster", Self::Butler => "Butler", Self::Settings => "Settings",
        }
    }
    #[allow(dead_code)]
    fn category(&self) -> SidebarCategory {
        match self {
            Self::Chat | Self::Sessions => SidebarCategory::Chat,
            Self::AgentPool | Self::ToolsMcp | Self::Automation => SidebarCategory::Agents,
            Self::RagSources | Self::Memory | Self::CloudStorage => SidebarCategory::Knowledge,
            Self::Media | Self::Audio => SidebarCategory::Generate,
            Self::PromptLab | Self::Evaluation => SidebarCategory::Optimize,
            _ => SidebarCategory::System,
        }
    }
    fn panels_for(cat: SidebarCategory) -> &'static [Self] {
        match cat {
            SidebarCategory::Chat => &[Self::Chat, Self::Sessions],
            SidebarCategory::Agents => &[Self::AgentPool, Self::ToolsMcp, Self::Automation],
            SidebarCategory::Knowledge => &[Self::RagSources, Self::Memory, Self::CloudStorage],
            SidebarCategory::Generate => &[Self::Media, Self::Audio],
            SidebarCategory::Optimize => &[Self::PromptLab, Self::Evaluation],
            SidebarCategory::System => &[Self::Security, Self::Analytics, Self::DevTools, Self::Cluster, Self::Butler, Self::Settings],
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum AutomationTab { Scheduler, Browser, Sandbox, Workflows }

#[derive(PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum MemoryTab { Episodic, Procedural, Entity }

#[derive(PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum SecurityTab { Overview, Guardrails, Pii, Rbac, AuditLog }

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

// =============================================================================
// Model Wizard Types
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum ModelCategory {
    Chat,
    Creative,
    Code,
    Math,
    Vision,
    Multilingual,
    Embedding,
    SmallFast,
    LargePowerful,
}

impl ModelCategory {
    fn label(&self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Creative => "Creative",
            Self::Code => "Code",
            Self::Math => "Math",
            Self::Vision => "Vision",
            Self::Multilingual => "Multilingual",
            Self::Embedding => "Embeddings",
            Self::SmallFast => "Small",
            Self::LargePowerful => "Large",
        }
    }

    fn icon(&self) -> &'static str {
        match self {
            Self::Chat => "\u{1f4ac}",       // speech bubble
            Self::Creative => "\u{270d}",     // writing hand
            Self::Code => "\u{1f4bb}",        // laptop
            Self::Math => "\u{1f9ee}",        // abacus
            Self::Vision => "\u{1f441}",      // eye
            Self::Multilingual => "\u{1f310}", // globe
            Self::Embedding => "\u{1f50d}",   // magnifying glass
            Self::SmallFast => "\u{26a1}",    // lightning
            Self::LargePowerful => "\u{1f9e0}", // brain
        }
    }

    fn all() -> &'static [ModelCategory] {
        &[
            Self::Chat, Self::Creative, Self::Code, Self::Math,
            Self::Vision, Self::Multilingual, Self::Embedding,
            Self::SmallFast, Self::LargePowerful,
        ]
    }
}

#[derive(Clone)]
struct CatalogModel {
    name: &'static str,
    size_estimate: &'static str,
    description: &'static str,
    category: ModelCategory,
}

fn model_catalog() -> Vec<CatalogModel> {
    vec![
        // ── Chat (General) ──────────────────────────────────────────────
        CatalogModel { name: "llama3.2",   size_estimate: "~2.0 GB",  description: "Meta Llama 3.2, great all-rounder",       category: ModelCategory::Chat },
        CatalogModel { name: "mistral",    size_estimate: "~4.1 GB",  description: "Mistral 7B, strong reasoning",            category: ModelCategory::Chat },
        CatalogModel { name: "gemma2",     size_estimate: "~5.0 GB",  description: "Google Gemma 2, efficient and capable",    category: ModelCategory::Chat },
        CatalogModel { name: "phi3",       size_estimate: "~2.3 GB",  description: "Microsoft Phi-3, compact but powerful",    category: ModelCategory::Chat },
        CatalogModel { name: "qwen2.5",    size_estimate: "~4.7 GB",  description: "Alibaba Qwen 2.5, well-rounded 7B",       category: ModelCategory::Chat },
        CatalogModel { name: "neural-chat",size_estimate: "~4.1 GB",  description: "Intel fine-tuned Mistral for conversation",category: ModelCategory::Chat },

        // ── Creative / Narration ────────────────────────────────────────
        CatalogModel { name: "nous-hermes2",     size_estimate: "~4.1 GB",  description: "Excellent storytelling and creative writing",   category: ModelCategory::Creative },
        CatalogModel { name: "dolphin-mixtral",   size_estimate: "~26 GB",   description: "Uncensored MoE, rich narrative generation",     category: ModelCategory::Creative },
        CatalogModel { name: "openhermes",        size_estimate: "~4.1 GB",  description: "Fine-tuned for creative and instructional text",category: ModelCategory::Creative },
        CatalogModel { name: "samantha-mistral",  size_estimate: "~4.1 GB",  description: "Companion-style, empathetic conversation",      category: ModelCategory::Creative },
        CatalogModel { name: "yarn-mistral:7b",   size_estimate: "~4.1 GB",  description: "Extended context (128k) for long narratives",   category: ModelCategory::Creative },
        CatalogModel { name: "stablelm2",         size_estimate: "~1.0 GB",  description: "StabilityAI creative text model",               category: ModelCategory::Creative },

        // ── Code ────────────────────────────────────────────────────────
        CatalogModel { name: "codellama",       size_estimate: "~3.8 GB",  description: "Meta's code-specialized Llama",              category: ModelCategory::Code },
        CatalogModel { name: "deepseek-coder",  size_estimate: "~3.8 GB",  description: "DeepSeek code generation model",             category: ModelCategory::Code },
        CatalogModel { name: "qwen2.5-coder",   size_estimate: "~4.7 GB",  description: "Alibaba Qwen code model, multilingual",      category: ModelCategory::Code },
        CatalogModel { name: "starcoder2",       size_estimate: "~3.8 GB",  description: "BigCode StarCoder 2, multi-language code",   category: ModelCategory::Code },
        CatalogModel { name: "codegemma",        size_estimate: "~5.0 GB",  description: "Google code model, fill-in-the-middle",     category: ModelCategory::Code },
        CatalogModel { name: "codellama:34b",    size_estimate: "~19 GB",   description: "Large CodeLlama for complex code tasks",     category: ModelCategory::Code },

        // ── Math / Reasoning ────────────────────────────────────────────
        CatalogModel { name: "mathstral",       size_estimate: "~4.1 GB",  description: "Mistral fine-tuned for math & science",     category: ModelCategory::Math },
        CatalogModel { name: "wizard-math",     size_estimate: "~4.1 GB",  description: "WizardMath, step-by-step problem solving",  category: ModelCategory::Math },
        CatalogModel { name: "deepseek-math",   size_estimate: "~4.1 GB",  description: "DeepSeek math specialist, competition-level", category: ModelCategory::Math },
        CatalogModel { name: "nous-hermes2-mixtral", size_estimate: "~26 GB", description: "MoE with strong logical reasoning",     category: ModelCategory::Math },

        // ── Vision ──────────────────────────────────────────────────────
        CatalogModel { name: "llava",          size_estimate: "~4.7 GB",  description: "Vision + language, image understanding",     category: ModelCategory::Vision },
        CatalogModel { name: "llava-llama3",   size_estimate: "~5.5 GB",  description: "LLaVA on Llama 3 backbone",                 category: ModelCategory::Vision },
        CatalogModel { name: "bakllava",       size_estimate: "~4.7 GB",  description: "BakLLaVA, Mistral-based vision model",      category: ModelCategory::Vision },
        CatalogModel { name: "llava:13b",      size_estimate: "~8.0 GB",  description: "Larger LLaVA for detailed image analysis",  category: ModelCategory::Vision },
        CatalogModel { name: "moondream",      size_estimate: "~1.0 GB",  description: "Tiny vision model, fast image Q&A",         category: ModelCategory::Vision },

        // ── Multilingual ────────────────────────────────────────────────
        CatalogModel { name: "aya",            size_estimate: "~4.8 GB",  description: "Cohere Aya, 100+ languages",                category: ModelCategory::Multilingual },
        CatalogModel { name: "qwen2.5:7b",    size_estimate: "~4.7 GB",  description: "Strong Chinese + English + more",            category: ModelCategory::Multilingual },
        CatalogModel { name: "mistral-nemo",   size_estimate: "~7.1 GB",  description: "Mistral 12B, multilingual + long context",   category: ModelCategory::Multilingual },
        CatalogModel { name: "gemma2:2b",      size_estimate: "~1.6 GB",  description: "Google, good multilingual at small size",    category: ModelCategory::Multilingual },

        // ── Embeddings (for RAG) ────────────────────────────────────────
        CatalogModel { name: "nomic-embed-text",  size_estimate: "~0.3 GB",  description: "Best open-source embedding, 8192 tokens",  category: ModelCategory::Embedding },
        CatalogModel { name: "mxbai-embed-large",  size_estimate: "~0.7 GB",  description: "MixedBread AI, high quality embeddings",   category: ModelCategory::Embedding },
        CatalogModel { name: "all-minilm",         size_estimate: "~0.05 GB", description: "Ultra-light sentence embeddings",          category: ModelCategory::Embedding },
        CatalogModel { name: "snowflake-arctic-embed", size_estimate: "~0.7 GB", description: "Snowflake Arctic, retrieval-optimized", category: ModelCategory::Embedding },

        // ── Small & Fast (< 4 GB VRAM) ─────────────────────────────────
        CatalogModel { name: "llama3.2:1b",   size_estimate: "~0.7 GB",  description: "Tiny Llama, very fast inference",            category: ModelCategory::SmallFast },
        CatalogModel { name: "phi3:mini",      size_estimate: "~1.4 GB",  description: "Microsoft's smallest Phi-3",                category: ModelCategory::SmallFast },
        CatalogModel { name: "tinyllama",      size_estimate: "~0.6 GB",  description: "Smallest practical chat LLM",               category: ModelCategory::SmallFast },
        CatalogModel { name: "qwen2.5:0.5b",  size_estimate: "~0.4 GB",  description: "Ultra-small Qwen model",                    category: ModelCategory::SmallFast },
        CatalogModel { name: "gemma2:2b",      size_estimate: "~1.6 GB",  description: "Google's 2B model, fast and capable",       category: ModelCategory::SmallFast },
        CatalogModel { name: "stablelm2:1.6b", size_estimate: "~1.0 GB",  description: "StabilityAI 1.6B, CPU-friendly",           category: ModelCategory::SmallFast },

        // ── Large & Powerful (16+ GB VRAM) ──────────────────────────────
        CatalogModel { name: "llama3.1:70b",   size_estimate: "~39 GB",   description: "Full-size Llama 3.1, near-GPT-4 quality",   category: ModelCategory::LargePowerful },
        CatalogModel { name: "mixtral",         size_estimate: "~26 GB",   description: "Mixture of Experts, 8x7B routing",          category: ModelCategory::LargePowerful },
        CatalogModel { name: "command-r",       size_estimate: "~20 GB",   description: "Cohere's RAG-optimized 35B model",          category: ModelCategory::LargePowerful },
        CatalogModel { name: "qwen2.5:72b",    size_estimate: "~41 GB",   description: "Alibaba 72B, top-tier open model",          category: ModelCategory::LargePowerful },
        CatalogModel { name: "deepseek-coder:33b", size_estimate: "~19 GB", description: "Large DeepSeek for complex code",         category: ModelCategory::LargePowerful },
        CatalogModel { name: "llama3.1:8b",    size_estimate: "~4.7 GB",  description: "Llama 3.1 8B, 128k context window",         category: ModelCategory::LargePowerful },
    ]
}

/// Strip ANSI escape sequences and Unicode block/box-drawing characters from ollama output.
///
/// Used for both pull progress and error messages, since ollama embeds terminal
/// control codes that render as squares in egui.
fn strip_ansi_and_blocks(raw: &str) -> String {
    // Step 1: Strip ANSI escape sequences (CSI sequences like \x1b[...X, OSC, etc.)
    let mut no_ansi = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // ESC — consume the entire escape sequence
            if let Some(&next) = chars.peek() {
                if next == '[' {
                    // CSI sequence: ESC [ ... (params) final_byte
                    chars.next(); // consume '['
                    // Consume parameter bytes (0x30-0x3F) and intermediate bytes (0x20-0x2F)
                    // then the final byte (0x40-0x7E)
                    loop {
                        match chars.peek() {
                            Some(&ch) if ('\x20'..='\x3f').contains(&ch) => { chars.next(); }
                            Some(&ch) if ('\x40'..='\x7e').contains(&ch) => { chars.next(); break; }
                            _ => break, // malformed, stop consuming
                        }
                    }
                } else if next == ']' {
                    // OSC sequence: ESC ] ... ST (BEL or ESC \)
                    chars.next();
                    loop {
                        match chars.next() {
                            Some('\x07') => break, // BEL terminates
                            Some('\x1b') => { chars.next(); break; } // ESC \ terminates
                            None => break,
                            _ => {}
                        }
                    }
                } else {
                    // Other two-char escape (ESC X)
                    chars.next();
                }
            }
            // else: bare ESC at end, just skip
        } else {
            no_ansi.push(c);
        }
    }

    // Step 2: Remove Unicode block elements and box drawing chars used for progress bar
    let cleaned: String = no_ansi
        .chars()
        .filter(|c| {
            !matches!(*c,
                '\u{2580}'..='\u{259F}' | // block elements (▕█░▏ etc.)
                '\u{2500}'..='\u{257F}'   // box drawing
            )
        })
        .collect();

    // Collapse multiple spaces into single space
    let mut result = String::with_capacity(cleaned.len());
    let mut last_was_space = false;
    for c in cleaned.chars() {
        if c == ' ' {
            if !last_was_space {
                result.push(' ');
            }
            last_was_space = true;
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    // Replace remaining multi-space separators with pipe for readability
    let result = result.trim().to_string();
    result
}

#[allow(dead_code)]
enum PullStatus {
    InProgress { model: String, last_line: String },
    Completed(String),
    Failed { model: String, error: String },
}

enum DeleteStatus {
    Completed(String),
    Failed { model: String, error: String },
}

#[derive(Clone, Copy, PartialEq)]
enum WizardTab {
    Recommended,
    AllModels,
    Installed,
}

#[derive(Clone, Copy, PartialEq)]
enum ModelSort {
    Name,
    Size,
    Category,
}

impl ModelSort {
    fn label(&self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Size => "Size",
            Self::Category => "Category",
        }
    }
}

struct ModelWizardState {
    selected_provider_idx: usize,
    category_filter: Option<ModelCategory>,
    pull_rx: Option<mpsc::Receiver<PullStatus>>,
    pulling_model: Option<String>,
    pull_progress_line: String,
    // Tabs, search, sort, filter
    tab: WizardTab,
    search_text: String,
    sort_by: ModelSort,
    show_installed_only: bool,
    custom_pull_name: String,
    // Delete
    delete_rx: Option<mpsc::Receiver<DeleteStatus>>,
    deleting_model: Option<String>,
}

impl Default for ModelWizardState {
    fn default() -> Self {
        Self {
            selected_provider_idx: 0,
            category_filter: None,
            pull_rx: None,
            pulling_model: None,
            pull_progress_line: String::new(),
            tab: WizardTab::Recommended,
            search_text: String::new(),
            sort_by: ModelSort::Name,
            show_installed_only: false,
            custom_pull_name: String::new(),
            delete_rx: None,
            deleting_model: None,
        }
    }
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
    history_depth: usize,
    timeout_secs: u32,
    /// If true: Enter sends, Ctrl+Enter inserts newline (default).
    /// If false: Ctrl+Enter sends, Enter inserts newline.
    enter_sends: bool,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            lm_studio_url: "http://localhost:1234".to_string(),
            temperature: 0.7,
            max_history: 20,
            graph_enabled: false,
            history_depth: 10,
            timeout_secs: 60,
            enter_sends: true,
        }
    }
}

#[allow(dead_code)]
struct PanelStates {
    // Navigation
    expanded_categories: std::collections::HashSet<u8>, // which sidebar categories are expanded (use category index)

    // Agent Pool
    agent_list: Vec<(String, String, String)>, // (id, name, role)
    new_agent_name: String,
    new_agent_role: String,
    agent_selected: Option<usize>,

    // Tools/MCP
    tool_search: String,
    tool_selected: Option<usize>,

    // Automation
    auto_tab: AutomationTab,
    cron_expression: String,
    cron_task_name: String,
    browser_url: String,
    browser_output: String,
    sandbox_code: String,
    sandbox_language: String,
    sandbox_output: String,

    // Memory
    memory_tab: MemoryTab,
    memory_search: String,

    // Cloud Storage
    cloud_provider: String, // "s3" or "gdrive"
    cloud_path: String,
    cloud_bucket: String,

    // Media
    media_prompt: String,
    media_model: String,
    media_results: Vec<String>, // paths to generated images

    // Audio
    audio_input_path: String,
    audio_output_text: String,
    tts_input: String,
    tts_voice: String,

    // Prompt Lab
    sig_name: String,
    sig_inputs: Vec<(String, String)>, // (name, description)
    sig_outputs: Vec<(String, String)>,
    optimization_running: bool,
    optimization_results: Vec<(String, f64)>, // (config, score)

    // Evaluation
    eval_suite_name: String,
    eval_samples: Vec<(String, String)>, // (input, expected)
    eval_results: Vec<(String, f64, String)>, // (sample, score, output)

    // Security
    security_tab: SecurityTab,
    pii_enabled: bool,
    guardrails_enabled: bool,
    rate_limit_rpm: u32,
    rbac_roles: Vec<(String, Vec<String>)>, // (role, permissions)

    // Analytics
    analytics_time_range: String, // "1h", "24h", "7d"

    // DevTools
    devtools_recording: bool,
    devtools_events: Vec<String>,
    devtools_breakpoints: Vec<String>,

    // Cluster
    cluster_nodes: Vec<(String, String, String)>, // (id, address, status)
    cluster_bootstrap: String,

    // Settings sections
    settings_section: u8, // 0=Provider, 1=Generation, 2=Security, 3=RAG, etc.
    system_prompt: String,
    max_tokens: u32,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

impl Default for PanelStates {
    fn default() -> Self {
        let mut expanded = std::collections::HashSet::new();
        expanded.insert(0); // Chat category expanded by default
        Self {
            expanded_categories: expanded,
            agent_list: Vec::new(), new_agent_name: String::new(), new_agent_role: "Researcher".to_string(), agent_selected: None,
            tool_search: String::new(), tool_selected: None,
            auto_tab: AutomationTab::Scheduler, cron_expression: "0 * * * *".to_string(), cron_task_name: String::new(),
            browser_url: String::new(), browser_output: String::new(),
            sandbox_code: String::new(), sandbox_language: "python".to_string(), sandbox_output: String::new(),
            memory_tab: MemoryTab::Episodic, memory_search: String::new(),
            cloud_provider: "s3".to_string(), cloud_path: "/".to_string(), cloud_bucket: String::new(),
            media_prompt: String::new(), media_model: "dall-e-3".to_string(), media_results: Vec::new(),
            audio_input_path: String::new(), audio_output_text: String::new(), tts_input: String::new(), tts_voice: "alloy".to_string(),
            sig_name: String::new(), sig_inputs: Vec::new(), sig_outputs: Vec::new(), optimization_running: false, optimization_results: Vec::new(),
            eval_suite_name: String::new(), eval_samples: Vec::new(), eval_results: Vec::new(),
            security_tab: SecurityTab::Overview, pii_enabled: true, guardrails_enabled: true, rate_limit_rpm: 60,
            rbac_roles: Vec::new(),
            analytics_time_range: "1h".to_string(),
            devtools_recording: false, devtools_events: Vec::new(), devtools_breakpoints: Vec::new(),
            cluster_nodes: Vec::new(), cluster_bootstrap: String::new(),
            settings_section: 0, system_prompt: String::new(), max_tokens: 4096, top_p: 0.9, frequency_penalty: 0.0, presence_penalty: 0.0,
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
    active_panel: SidebarPanel,
    panels: PanelStates,
    monitor_tab: MonitorTab,
    show_settings: bool,
    show_monitor: bool,
    settings: GuiSettings,
    toasts: Vec<(String, bool, Instant)>,
    last_dir: Option<PathBuf>,

    // Model wizard
    show_model_wizard: bool,
    wizard_state: ModelWizardState,

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

            active_panel: SidebarPanel::Chat,
            panels: PanelStates::default(),
            monitor_tab: MonitorTab::Overview,
            show_settings: false,
            show_monitor: false,
            settings: GuiSettings::default(),
            toasts: Vec::new(),
            last_dir: None,

            show_model_wizard: false,
            wizard_state: ModelWizardState::default(),

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

        // 5. Ollama pull progress
        if let Some(ref rx) = self.wizard_state.pull_rx {
            let mut done = false;
            while let Ok(status) = rx.try_recv() {
                match status {
                    PullStatus::InProgress { last_line, .. } => {
                        self.wizard_state.pull_progress_line = last_line;
                    }
                    PullStatus::Completed(model) => {
                        self.wizard_state.pulling_model = None;
                        self.wizard_state.pull_progress_line.clear();
                        self.add_toast(&format!("Model '{}' installed!", model), false);
                        self.assistant.fetch_models();
                        done = true;
                        break;
                    }
                    PullStatus::Failed { model, error } => {
                        self.wizard_state.pulling_model = None;
                        self.wizard_state.pull_progress_line.clear();
                        self.add_toast(&format!("Failed to pull '{}': {}", model, error), true);
                        done = true;
                        break;
                    }
                }
            }
            if done {
                self.wizard_state.pull_rx = None;
            }
        }

        // 6. Ollama delete completion
        if let Some(ref rx) = self.wizard_state.delete_rx {
            if let Ok(status) = rx.try_recv() {
                match status {
                    DeleteStatus::Completed(ref model) => {
                        self.wizard_state.deleting_model = None;
                        self.add_toast(&format!("Model '{}' deleted", model), false);
                        // Remove from local list immediately so UI updates instantly
                        self.assistant.available_models.retain(|m| m.name != *model);
                        // Also refresh from server to get authoritative list
                        self.assistant.fetch_models();
                    }
                    DeleteStatus::Failed { model, error } => {
                        self.wizard_state.deleting_model = None;
                        let clean_error = strip_ansi_and_blocks(&error);
                        self.add_toast(&format!("Failed to delete '{}': {}", model, clean_error), true);
                    }
                }
                self.wizard_state.delete_rx = None;
            }
        }
    }

    fn apply_scan_result(&mut self, result: ScanResult) {
        if result.report.llm_providers.is_empty() {
            self.phase = AppPhase::NoProviders(String::new());
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

    fn reindex_existing_sources_into_graph(&mut self) {
        if self.knowledge_graph.is_none() {
            self.init_knowledge_graph();
        }
        let sources: Vec<KnowledgeSourceInfo> = self.knowledge_sources.clone();
        for ks in &sources {
            if ks.status != KnowledgeStatus::Indexed {
                continue;
            }
            if ks.is_kpkg {
                if let Ok(data) = std::fs::read(&ks.file_path) {
                    let reader = KpkgReader::<AppKeyProvider>::with_app_key();
                    if let Ok(docs) = reader.read(&data) {
                        for doc in &docs {
                            let source = format!("kpkg:{}:{}", ks.name, doc.path);
                            self.pending_graph_docs.push((source, doc.content.clone()));
                        }
                    }
                }
            } else if let Ok(content) = std::fs::read_to_string(&ks.file_path) {
                self.pending_graph_docs.push((ks.name.clone(), content));
            }
        }
        if !self.pending_graph_docs.is_empty() {
            self.index_pending_graph_docs();
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
                ui.label("Checking Ollama, LM Studio, and cloud API keys");
            });
        });
    }

    fn render_no_providers_screen(&mut self, ctx: &egui::Context, _msg: String) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.label(egui::RichText::new("No LLM Providers Found").size(28.0).strong());
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Install a local LLM provider to get started")
                            .size(14.0)
                            .color(Color32::GRAY),
                    );
                    ui.add_space(24.0);

                    // Provider recommendation cards
                    let card_width = (ui.available_width() * 0.42).min(320.0);
                    ui.horizontal(|ui| {
                        ui.add_space((ui.available_width() - card_width * 2.0 - 16.0).max(0.0) / 2.0);

                        // Ollama card
                        Self::render_provider_card(
                            ui,
                            card_width,
                            "\u{1f999}", // llama emoji
                            "Ollama",
                            "Lightweight CLI tool for running LLMs locally. \
                             The fastest way to get started.",
                            "https://ollama.com",
                            &[
                                "Download and install from the link above",
                                "Open a terminal and run:",
                                "   ollama pull llama3.2",
                                "Click 'Retry Scan' below",
                            ],
                        );
                        ui.add_space(16.0);
                        // LM Studio card
                        Self::render_provider_card(
                            ui,
                            card_width,
                            "\u{1f3ac}", // clapper emoji
                            "LM Studio",
                            "Desktop application with a graphical model \
                             manager and built-in server.",
                            "https://lmstudio.ai",
                            &[
                                "Download and install from the link above",
                                "Launch LM Studio",
                                "Download a model from the Discover tab",
                                "Start the local server (Developer tab)",
                                "Click 'Retry Scan' below",
                            ],
                        );
                    });

                    ui.add_space(20.0);

                    // Cloud API note
                    let note_width = (ui.available_width() * 0.7).min(660.0);
                    ui.allocate_ui(egui::vec2(note_width, 0.0), |ui| {
                        egui::Frame::none()
                            .fill(Color32::from_rgb(40, 40, 30))
                            .rounding(6.0)
                            .inner_margin(12.0)
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(egui::RichText::new("\u{1f4a1}").size(14.0));
                                    ui.label(
                                        egui::RichText::new(
                                            "Cloud APIs: Set OPENAI_API_KEY or ANTHROPIC_API_KEY \
                                             as environment variables, then retry the scan."
                                        )
                                        .size(12.0)
                                        .color(Color32::from_rgb(200, 200, 160)),
                                    );
                                });
                            });
                    });

                    ui.add_space(24.0);
                    ui.separator();
                    ui.add_space(12.0);

                    ui.label(
                        egui::RichText::new("Already installed? Enter custom URLs:")
                            .size(12.0)
                            .color(Color32::GRAY),
                    );
                    ui.add_space(8.0);

                    // URL input fields
                    let field_width = (ui.available_width() * 0.5).min(400.0);
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
                    ui.add_space(20.0);
                });
            });
        });
    }

    fn render_provider_card(
        ui: &mut Ui,
        width: f32,
        icon: &str,
        name: &str,
        description: &str,
        url: &str,
        steps: &[&str],
    ) {
        ui.allocate_ui(egui::vec2(width, 0.0), |ui| {
            egui::Frame::none()
                .fill(Color32::from_rgb(35, 40, 50))
                .rounding(10.0)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.set_min_width(width - 32.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(icon).size(22.0));
                        ui.label(egui::RichText::new(name).size(18.0).strong());
                    });
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(description).size(12.0).color(Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);
                    ui.hyperlink_to(
                        egui::RichText::new(format!("\u{1f517} {}", url)).size(12.0),
                        url,
                    );
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("How to install:").size(12.0).strong());
                    ui.add_space(4.0);
                    let mut step_num = 1;
                    for step in steps.iter() {
                        if step.starts_with("   ") {
                            // Indented code-like text (not numbered)
                            ui.label(
                                egui::RichText::new(*step)
                                    .size(12.0)
                                    .color(Color32::from_rgb(130, 200, 130))
                                    .monospace(),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new(format!("{}. {}", step_num, step))
                                    .size(12.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            step_num += 1;
                        }
                    }
                });
        });
    }

    // =========================================================================
    // Model Wizard
    // =========================================================================

    fn start_ollama_pull(&mut self, model_name: &str) {
        if self.wizard_state.pulling_model.is_some() {
            self.add_toast("Already pulling a model, please wait", true);
            return;
        }

        let name = model_name.to_string();
        let (tx, rx) = mpsc::channel::<PullStatus>();
        self.wizard_state.pull_rx = Some(rx);
        self.wizard_state.pulling_model = Some(name.clone());
        self.wizard_state.pull_progress_line.clear();

        std::thread::spawn(move || {
            let result = std::process::Command::new("ollama")
                .args(["pull", &name])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn();

            match result {
                Ok(mut child) => {
                    // Read stderr (where ollama writes progress)
                    if let Some(stderr) = child.stderr.take() {
                        use std::io::Read as _;
                        let mut reader = std::io::BufReader::new(stderr);
                        let mut buf = [0u8; 512];
                        loop {
                            match reader.read(&mut buf) {
                                Ok(0) => break,
                                Ok(n) => {
                                    let text = String::from_utf8_lossy(&buf[..n]);
                                    // Extract the last meaningful line
                                    let line = text
                                        .split('\r')
                                        .filter(|s| !s.trim().is_empty())
                                        .last()
                                        .unwrap_or("")
                                        .trim()
                                        .to_string();
                                    if !line.is_empty() {
                                        // Clean up progress bar characters for GUI display.
                                        // Ollama outputs: "pulling abc123:  45% ▕████░░░▏ 1.6 GB/26 GB  118 MB/s  3m30s"
                                        // We extract: "pulling abc123: 45% - 1.6 GB/26 GB - 118 MB/s - 3m30s"
                                        let clean = strip_ansi_and_blocks(&line);
                                        let _ = tx.send(PullStatus::InProgress {
                                            model: name.clone(),
                                            last_line: clean,
                                        });
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }
                    match child.wait() {
                        Ok(status) if status.success() => {
                            let _ = tx.send(PullStatus::Completed(name));
                        }
                        Ok(status) => {
                            let _ = tx.send(PullStatus::Failed {
                                model: name,
                                error: format!("Process exited with code: {}", status),
                            });
                        }
                        Err(e) => {
                            let _ = tx.send(PullStatus::Failed {
                                model: name,
                                error: format!("Wait error: {}", e),
                            });
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(PullStatus::Failed {
                        model: name,
                        error: format!(
                            "Could not start ollama. Is it installed and on your PATH? ({})",
                            e
                        ),
                    });
                }
            }
        });
    }

    fn start_ollama_delete(&mut self, model_name: &str) {
        if self.wizard_state.pulling_model.is_some() {
            self.add_toast("Can't delete while pulling a model", true);
            return;
        }
        if self.wizard_state.deleting_model.is_some() {
            self.add_toast("Already deleting a model, please wait", true);
            return;
        }

        let name = model_name.to_string();
        let (tx, rx) = mpsc::channel::<DeleteStatus>();
        self.wizard_state.delete_rx = Some(rx);
        self.wizard_state.deleting_model = Some(name.clone());

        std::thread::spawn(move || {
            let result = std::process::Command::new("ollama")
                .args(["rm", &name])
                .output();

            match result {
                Ok(output) if output.status.success() => {
                    let _ = tx.send(DeleteStatus::Completed(name));
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let raw_error = if stderr.trim().is_empty() {
                        stdout.trim().to_string()
                    } else {
                        stderr.trim().to_string()
                    };
                    let _ = tx.send(DeleteStatus::Failed {
                        model: name,
                        error: raw_error,
                    });
                }
                Err(e) => {
                    let _ = tx.send(DeleteStatus::Failed {
                        model: name,
                        error: format!("Could not run ollama: {}", e),
                    });
                }
            }
        });
    }

    fn render_model_wizard(&mut self, ctx: &egui::Context) {
        if !self.show_model_wizard {
            return;
        }

        let mut open = true;
        let mut pull_request: Option<String> = None;
        let mut delete_request: Option<String> = None;

        egui::Window::new("Model Library")
            .open(&mut open)
            .resizable(true)
            .default_width(580.0)
            .default_height(520.0)
            .show(ctx, |ui| {
                // --- Provider selector + Refresh ---
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Provider:").size(12.0));
                    let providers: Vec<(String, ai_assistant::AiProvider)> =
                        if let Some(ref scan) = self.scan_result_data {
                            scan.report.llm_providers.iter()
                                .map(|p| (p.name.clone(), p.provider_type.clone()))
                                .collect()
                        } else {
                            Vec::new()
                        };
                    if providers.is_empty() {
                        ui.label(egui::RichText::new("No providers detected").color(Color32::GRAY).size(12.0));
                    } else {
                        let selected_name = providers.get(self.wizard_state.selected_provider_idx)
                            .map(|(n, _)| n.as_str()).unwrap_or("Select...");
                        egui::ComboBox::from_id_source("wizard_provider")
                            .selected_text(selected_name).width(160.0)
                            .show_ui(ui, |ui| {
                                for (i, (name, _)) in providers.iter().enumerate() {
                                    ui.selectable_value(&mut self.wizard_state.selected_provider_idx, i, name);
                                }
                            });
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Refresh").clicked() { self.assistant.fetch_models(); }
                        if self.assistant.is_fetching_models { ui.spinner(); }
                    });
                });

                ui.add_space(4.0);

                // --- Tab bar ---
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.wizard_state.tab, WizardTab::Recommended, "Recommended");
                    ui.selectable_value(&mut self.wizard_state.tab, WizardTab::AllModels, "All Models");
                    let installed_count = self.assistant.available_models.len();
                    let installed_label = format!("Installed ({})", installed_count);
                    ui.selectable_value(&mut self.wizard_state.tab, WizardTab::Installed, installed_label);
                });
                ui.separator();

                // --- Shared state ---
                let catalog = model_catalog();
                let installed_names: Vec<String> = self.assistant.available_models.iter().map(|m| m.name.clone()).collect();
                let providers: Vec<(String, ai_assistant::AiProvider)> =
                    if let Some(ref scan) = self.scan_result_data {
                        scan.report.llm_providers.iter().map(|p| (p.name.clone(), p.provider_type.clone())).collect()
                    } else { Vec::new() };
                let selected_provider = providers.get(self.wizard_state.selected_provider_idx).map(|(_, p)| p.clone());
                let is_ollama = matches!(selected_provider, Some(ai_assistant::AiProvider::Ollama));
                let is_cloud = selected_provider.as_ref().map(|p| p.is_cloud()).unwrap_or(false);
                let is_busy = self.wizard_state.pulling_model.is_some() || self.wizard_state.deleting_model.is_some();
                let pulling_name = self.wizard_state.pulling_model.clone();
                let deleting_name = self.wizard_state.deleting_model.clone();
                let pull_line = self.wizard_state.pull_progress_line.clone();

                let is_model_installed = |name: &str| -> bool {
                    installed_names.iter().any(|n| n == name || n.starts_with(&format!("{}:", name)) || name.starts_with(&format!("{}:", n)))
                };

                match self.wizard_state.tab {
                    // =============================================================
                    // TAB 1: Recommended
                    // =============================================================
                    WizardTab::Recommended => {
                        // Category filter
                        ui.horizontal(|ui| {
                            let all_sel = self.wizard_state.category_filter.is_none();
                            if ui.selectable_label(all_sel, egui::RichText::new("All").size(12.0)).clicked() {
                                self.wizard_state.category_filter = None;
                            }
                            for cat in ModelCategory::all() {
                                let label = format!("{} {}", cat.icon(), cat.label());
                                if ui.selectable_label(self.wizard_state.category_filter == Some(*cat), egui::RichText::new(label).size(12.0)).clicked() {
                                    self.wizard_state.category_filter = Some(*cat);
                                }
                            }
                        });
                        ui.separator();

                        // VRAM guide
                        ui.add_space(4.0);
                        egui::CollapsingHeader::new(
                            egui::RichText::new("\u{1f4be} VRAM Guide").size(11.0).color(Color32::LIGHT_GRAY),
                        ).default_open(false).show(ui, |ui| {
                            let info = [
                                ("4 GB",  "Small models, embeddings (tinyllama, phi3:mini)"),
                                ("6 GB",  "Most 7B models (llama3.2, mistral, codellama)"),
                                ("8 GB",  "7B comfortably + some 13B quantized"),
                                ("12 GB", "13B models, multiple small models"),
                                ("16 GB", "13B comfortably, some 33B quantized"),
                                ("24 GB", "33-35B models (command-r)"),
                                ("48 GB+","70B+ models (llama3.1:70b, mixtral)"),
                                ("CPU",   "Any model (slower). 16+ GB RAM recommended."),
                            ];
                            egui::Grid::new("vram_guide").num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
                                for (v, d) in &info {
                                    ui.label(egui::RichText::new(*v).size(11.0).strong().color(Color32::from_rgb(130, 180, 255)));
                                    ui.label(egui::RichText::new(*d).size(10.0).color(Color32::LIGHT_GRAY));
                                    ui.end_row();
                                }
                            });
                        });
                        ui.add_space(4.0);

                        // Model cards
                        egui::ScrollArea::vertical().id_source("rec_scroll").show(ui, |ui| {
                            let filter = self.wizard_state.category_filter;
                            let mut cur_cat: Option<ModelCategory> = None;
                            for model in &catalog {
                                if let Some(f) = filter { if model.category != f { continue; } }
                                // Category heading
                                if cur_cat != Some(model.category) {
                                    cur_cat = Some(model.category);
                                    if filter.is_none() {
                                        ui.add_space(10.0);
                                        ui.label(egui::RichText::new(format!("{} {}", model.category.icon(), model.category.label())).size(14.0).strong());
                                        ui.add_space(4.0);
                                    }
                                }
                                let installed = is_model_installed(model.name);
                                let this_pulling = pulling_name.as_deref() == Some(model.name);
                                let this_deleting = deleting_name.as_deref() == Some(model.name);
                                let (pull, del) = Self::render_model_card(
                                    ui, model.name, model.size_estimate, model.description,
                                    installed, this_pulling, this_deleting, is_ollama, is_cloud, is_busy, &pull_line,
                                );
                                if pull { pull_request = Some(model.name.to_string()); }
                                if del { delete_request = Some(model.name.to_string()); }
                                ui.add_space(3.0);
                            }
                            ui.add_space(8.0);
                        });
                    }

                    // =============================================================
                    // TAB 2: All Models
                    // =============================================================
                    WizardTab::AllModels => {
                        // Search + Sort + Filter
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("\u{1f50d}").size(12.0));
                            ui.add_sized([160.0, 20.0], egui::TextEdit::singleline(&mut self.wizard_state.search_text).hint_text("Search..."));
                            ui.separator();
                            ui.label(egui::RichText::new("Sort:").size(11.0));
                            egui::ComboBox::from_id_source("sort_combo")
                                .selected_text(self.wizard_state.sort_by.label())
                                .width(80.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.wizard_state.sort_by, ModelSort::Name, "Name");
                                    ui.selectable_value(&mut self.wizard_state.sort_by, ModelSort::Size, "Size");
                                    ui.selectable_value(&mut self.wizard_state.sort_by, ModelSort::Category, "Category");
                                });
                            ui.checkbox(&mut self.wizard_state.show_installed_only, egui::RichText::new("Installed only").size(11.0));
                        });
                        ui.separator();

                        // Build merged list: catalog + installed (deduplicated)
                        struct MergedModel {
                            name: String,
                            size: String,
                            description: String,
                            category: Option<ModelCategory>,
                            installed: bool,
                        }
                        let mut merged: Vec<MergedModel> = Vec::new();
                        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                        // Add catalog models
                        for m in &catalog {
                            let inst = is_model_installed(m.name);
                            merged.push(MergedModel {
                                name: m.name.to_string(), size: m.size_estimate.to_string(),
                                description: m.description.to_string(), category: Some(m.category), installed: inst,
                            });
                            seen.insert(m.name.to_string());
                        }
                        // Add installed models not in catalog
                        for m in &self.assistant.available_models {
                            if !seen.contains(&m.name) {
                                let size_str = m.size.clone().unwrap_or_default();
                                merged.push(MergedModel {
                                    name: m.name.clone(), size: size_str,
                                    description: String::new(), category: None, installed: true,
                                });
                                seen.insert(m.name.clone());
                            }
                        }
                        // Filter
                        let search = self.wizard_state.search_text.to_lowercase();
                        if !search.is_empty() {
                            merged.retain(|m| m.name.to_lowercase().contains(&search) || m.description.to_lowercase().contains(&search));
                        }
                        if self.wizard_state.show_installed_only {
                            merged.retain(|m| m.installed);
                        }
                        // Sort
                        match self.wizard_state.sort_by {
                            ModelSort::Name => merged.sort_by(|a, b| a.name.cmp(&b.name)),
                            ModelSort::Size => merged.sort_by(|a, b| a.size.cmp(&b.size)),
                            ModelSort::Category => merged.sort_by(|a, b| {
                                let ca = a.category.map(|c| c.label()).unwrap_or("zzz");
                                let cb = b.category.map(|c| c.label()).unwrap_or("zzz");
                                ca.cmp(cb)
                            }),
                        }

                        ui.label(egui::RichText::new(format!("{} models", merged.len())).size(10.0).color(Color32::GRAY));

                        egui::ScrollArea::vertical().id_source("all_scroll").show(ui, |ui| {
                            for m in &merged {
                                let this_pulling = pulling_name.as_deref() == Some(m.name.as_str());
                                let this_deleting = deleting_name.as_deref() == Some(m.name.as_str());
                                let cat_label = m.category.map(|c| format!("{} ", c.label())).unwrap_or_default();
                                let desc = if m.description.is_empty() { cat_label } else { format!("{}{}", cat_label, m.description) };
                                let (pull, del) = Self::render_model_card(
                                    ui, &m.name, &m.size, &desc,
                                    m.installed, this_pulling, this_deleting, is_ollama, is_cloud, is_busy, &pull_line,
                                );
                                if pull { pull_request = Some(m.name.clone()); }
                                if del { delete_request = Some(m.name.clone()); }
                                ui.add_space(3.0);
                            }

                            // Custom pull section
                            if is_ollama {
                                ui.add_space(8.0);
                                ui.separator();
                                ui.add_space(4.0);
                                ui.label(egui::RichText::new("Pull any model by name:").size(12.0).color(Color32::LIGHT_GRAY));
                                ui.horizontal(|ui| {
                                    ui.add_sized([200.0, 20.0],
                                        egui::TextEdit::singleline(&mut self.wizard_state.custom_pull_name)
                                            .hint_text("e.g. llama3.2:1b"),
                                    );
                                    ui.add_enabled_ui(!is_busy && !self.wizard_state.custom_pull_name.trim().is_empty(), |ui| {
                                        if ui.button("Pull").clicked() {
                                            pull_request = Some(self.wizard_state.custom_pull_name.trim().to_string());
                                            self.wizard_state.custom_pull_name.clear();
                                        }
                                    });
                                });
                            }
                            ui.add_space(8.0);
                        });
                    }

                    // =============================================================
                    // TAB 3: Installed
                    // =============================================================
                    WizardTab::Installed => {
                        // Search + Sort
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("\u{1f50d}").size(12.0));
                            ui.add_sized([200.0, 20.0], egui::TextEdit::singleline(&mut self.wizard_state.search_text).hint_text("Search installed..."));
                            ui.separator();
                            ui.label(egui::RichText::new("Sort:").size(11.0));
                            egui::ComboBox::from_id_source("inst_sort_combo")
                                .selected_text(self.wizard_state.sort_by.label())
                                .width(80.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.wizard_state.sort_by, ModelSort::Name, "Name");
                                    ui.selectable_value(&mut self.wizard_state.sort_by, ModelSort::Size, "Size");
                                });
                        });
                        ui.separator();

                        let mut installed: Vec<&ModelInfo> = self.assistant.available_models.iter().collect();
                        // Filter by search
                        let search = self.wizard_state.search_text.to_lowercase();
                        if !search.is_empty() {
                            installed.retain(|m| m.name.to_lowercase().contains(&search));
                        }
                        // Sort
                        match self.wizard_state.sort_by {
                            ModelSort::Name => installed.sort_by(|a, b| a.name.cmp(&b.name)),
                            ModelSort::Size => installed.sort_by(|a, b| {
                                a.size.as_deref().unwrap_or("").cmp(&b.size.as_deref().unwrap_or(""))
                            }),
                            _ => installed.sort_by(|a, b| a.name.cmp(&b.name)),
                        }

                        if installed.is_empty() {
                            ui.add_space(40.0);
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new("No models installed").size(14.0).color(Color32::GRAY));
                                ui.add_space(8.0);
                                ui.label(egui::RichText::new("Check the Recommended tab to get started").size(12.0).color(Color32::DARK_GRAY));
                            });
                        } else {
                            ui.label(egui::RichText::new(format!("{} models installed", installed.len())).size(10.0).color(Color32::GRAY));
                            egui::ScrollArea::vertical().id_source("inst_scroll").show(ui, |ui| {
                                for m in &installed {
                                    let size_str = m.size.clone().unwrap_or_default();
                                    let desc = m.display_name();
                                    let this_deleting = deleting_name.as_deref() == Some(m.name.as_str());
                                    let this_pulling = pulling_name.as_deref() == Some(m.name.as_str());
                                    let (_, del) = Self::render_model_card(
                                        ui, &m.name, &size_str, &desc,
                                        true, this_pulling, this_deleting, is_ollama, is_cloud, is_busy, &pull_line,
                                    );
                                    if del { delete_request = Some(m.name.clone()); }
                                    ui.add_space(3.0);
                                }
                                ui.add_space(8.0);
                            });
                        }
                    }
                }
            });

        if !open {
            self.show_model_wizard = false;
        }

        if let Some(model_name) = pull_request {
            self.start_ollama_pull(&model_name);
        }
        if let Some(model_name) = delete_request {
            // Resolve the actual installed name (catalog may use "llama3.2" but
            // ollama needs the full tag like "llama3.2:1b")
            let real_name = self.assistant.available_models.iter()
                .find(|m| m.name == model_name
                    || m.name.starts_with(&format!("{}:", model_name))
                    || model_name.starts_with(&format!("{}:", m.name)))
                .map(|m| m.name.clone())
                .unwrap_or(model_name);
            self.start_ollama_delete(&real_name);
        }
    }

    /// Render a single model card. Returns (pull_clicked, delete_clicked).
    fn render_model_card(
        ui: &mut Ui,
        name: &str,
        size: &str,
        description: &str,
        installed: bool,
        is_pulling: bool,
        is_deleting: bool,
        is_ollama: bool,
        is_cloud: bool,
        is_busy: bool,
        pull_line: &str,
    ) -> (bool, bool) {
        let mut pull_clicked = false;
        let mut delete_clicked = false;

        egui::Frame::none()
            .fill(if installed { Color32::from_rgb(25, 40, 30) } else { Color32::from_rgb(30, 32, 38) })
            .rounding(6.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Left: info
                    ui.vertical(|ui| {
                        ui.set_min_width(ui.available_width() - 140.0);
                        ui.horizontal(|ui| {
                            if installed {
                                ui.label(egui::RichText::new("\u{2705}").size(13.0));
                            }
                            ui.label(egui::RichText::new(name).size(13.0).strong());
                            if !size.is_empty() {
                                ui.label(egui::RichText::new(size).size(11.0).color(Color32::GRAY));
                            }
                            if installed {
                                ui.label(egui::RichText::new("(installed)").size(11.0).color(Color32::from_rgb(100, 200, 100)));
                            }
                        });
                        if !description.is_empty() && description != name {
                            ui.label(egui::RichText::new(description).size(11.0).color(Color32::LIGHT_GRAY));
                        }
                        if is_pulling && !pull_line.is_empty() {
                            ui.label(egui::RichText::new(pull_line).size(10.0).color(Color32::YELLOW).monospace());
                        }
                    });

                    // Right: actions
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if is_pulling {
                            ui.spinner();
                            ui.label(egui::RichText::new("Pulling...").size(11.0).color(Color32::YELLOW));
                        } else if is_deleting {
                            ui.spinner();
                            ui.label(egui::RichText::new("Deleting...").size(11.0).color(Color32::from_rgb(255, 120, 100)));
                        } else if installed && is_ollama {
                            // Delete button for installed Ollama models
                            ui.add_enabled_ui(!is_busy, |ui| {
                                let btn = ui.button(egui::RichText::new("Delete").size(11.0).color(Color32::from_rgb(255, 100, 80)));
                                if btn.clicked() {
                                    delete_clicked = true;
                                }
                            });
                        } else if !installed && is_cloud {
                            ui.label(egui::RichText::new("Available").size(11.0).color(Color32::LIGHT_GREEN));
                        } else if !installed && is_ollama {
                            ui.add_enabled_ui(!is_busy, |ui| {
                                if ui.button("Pull").clicked() {
                                    pull_clicked = true;
                                }
                            });
                        } else if !installed {
                            ui.hyperlink_to(egui::RichText::new("Open LM Studio").size(11.0), "https://lmstudio.ai");
                        }
                    });
                });
            });

        (pull_clicked, delete_clicked)
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

                    // Model Library
                    let wizard_label = if self.show_model_wizard {
                        "Close Models"
                    } else {
                        "Model Library"
                    };
                    if ui.button(wizard_label).clicked() {
                        self.show_model_wizard = !self.show_model_wizard;
                    }
                });
            });
        });
    }

    fn render_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar")
            .default_width(200.0)
            .min_width(160.0)
            .show(ctx, |ui| {
                ui.heading("ai_assistant Pro");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (cat_idx, cat) in SidebarCategory::all().iter().enumerate() {
                        let is_expanded = self.panels.expanded_categories.contains(&(cat_idx as u8));
                        ui.horizontal(|ui| {
                            let arrow = if is_expanded { "\u{25BC}" } else { "\u{25B6}" }; // down / right triangle
                            if ui.selectable_label(false, format!("{} {}", arrow, cat.label())).clicked() {
                                if is_expanded {
                                    self.panels.expanded_categories.remove(&(cat_idx as u8));
                                } else {
                                    self.panels.expanded_categories.insert(cat_idx as u8);
                                }
                            }
                        });

                        if is_expanded {
                            for panel in SidebarPanel::panels_for(*cat) {
                                let selected = self.active_panel == *panel;
                                ui.horizontal(|ui| {
                                    ui.add_space(16.0);
                                    if ui.selectable_label(selected, panel.label()).clicked() {
                                        self.active_panel = *panel;
                                    }
                                });
                            }
                        }
                        ui.add_space(2.0);
                    }
                });
            });
    }

    fn render_sessions_panel(&mut self, ui: &mut Ui) {
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

    fn render_knowledge_panel(&mut self, ui: &mut Ui) {
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
            let indexed_chunks: usize = self
                .knowledge_sources
                .iter()
                .filter(|k| k.status == KnowledgeStatus::Indexed)
                .map(|k| k.doc_count)
                .sum();
            let status = RagStatus {
                rag_available: self.assistant.rag_db.is_some(),
                knowledge_enabled: true,
                conversation_enabled: false,
                knowledge_chunks: indexed_chunks,
                knowledge_tokens: 0,
                archived_messages: 0,
                archived_tokens: 0,
            };
            widgets::rag_status_compact(ui, &status);

            // Context budget info
            if !self.selected_model.is_empty() {
                let available = self.assistant.calculate_available_knowledge_tokens("test");
                let conv_tokens: usize = self.assistant.conversation.iter()
                    .map(|m| m.content.len() / 4)
                    .sum();
                ui.add_space(4.0);
                ui.colored_label(
                    Color32::from_gray(140),
                    format!("Context: ~{} tokens available for knowledge (~{} used by conversation)",
                        available, conv_tokens),
                );
                if let Some(ref usage) = self.assistant.last_knowledge_usage {
                    ui.colored_label(
                        Color32::from_gray(140),
                        format!("Last query used {} tokens from {} sources",
                            usage.total_tokens, usage.sources.len()),
                    );
                }
            }
        }

        ui.add_space(8.0);
        ui.separator();

        // Knowledge graph toggle
        let prev = self.settings.graph_enabled;
        ui.checkbox(&mut self.settings.graph_enabled, "Knowledge Graph");
        if self.settings.graph_enabled && !prev {
            self.init_knowledge_graph();
            if !self.pending_graph_docs.is_empty() {
                self.index_pending_graph_docs();
            } else if !self.knowledge_sources.is_empty() {
                self.reindex_existing_sources_into_graph();
            }
        }

        // Reset graph button (only when graph is enabled)
        if self.settings.graph_enabled {
            if ui.button("Rebuild Graph").clicked() {
                if let Some(ref kg) = self.knowledge_graph {
                    let _ = kg.clear();
                }
                self.graph_viz = None;
                if !self.knowledge_sources.is_empty() {
                    self.reindex_existing_sources_into_graph();
                    self.add_toast("Knowledge graph rebuilt", false);
                } else {
                    self.add_toast("Graph cleared (no sources to re-index)", false);
                }
            }
        }
    }

    fn render_butler_panel(&mut self, ui: &mut Ui) {
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

    fn render_agent_pool_panel(&mut self, ui: &mut Ui) {
        ui.heading("Agent Pool");
        ui.add_space(4.0);

        // Create new agent
        egui::Frame::none()
            .fill(Color32::from_gray(30))
            .rounding(8.0)
            .inner_margin(egui::Margin::same(10.0))
            .show(ui, |ui| {
                ui.label("Create Agent");
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut self.panels.new_agent_name);
                });
                ui.horizontal(|ui| {
                    ui.label("Role:");
                    egui::ComboBox::from_id_source("agent_role_combo")
                        .selected_text(&self.panels.new_agent_role)
                        .show_ui(ui, |ui| {
                            for role in &["Coordinator", "Researcher", "Analyst", "Writer", "Reviewer", "Executor", "Validator"] {
                                ui.selectable_value(&mut self.panels.new_agent_role, role.to_string(), *role);
                            }
                        });
                });
                if ui.button("Add Agent").clicked() && !self.panels.new_agent_name.is_empty() {
                    let id = format!("agent_{}", self.panels.agent_list.len() + 1);
                    self.panels.agent_list.push((
                        id,
                        self.panels.new_agent_name.clone(),
                        self.panels.new_agent_role.clone(),
                    ));
                    self.panels.new_agent_name.clear();
                    self.add_toast("Agent created", false);
                }
            });

        ui.add_space(8.0);
        ui.separator();
        ui.label(format!("{} agents registered", self.panels.agent_list.len()));
        ui.add_space(4.0);

        // Agent list
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut delete_idx = None;
            for (i, (id, name, role)) in self.panels.agent_list.iter().enumerate() {
                let selected = self.panels.agent_selected == Some(i);
                egui::Frame::none()
                    .fill(if selected { Color32::from_gray(45) } else { Color32::from_gray(30) })
                    .rounding(6.0)
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(selected, format!("{} ({})", name, role)).clicked() {
                                self.panels.agent_selected = Some(i);
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("Delete").clicked() {
                                    delete_idx = Some(i);
                                }
                                ui.colored_label(Color32::from_gray(120), id);
                            });
                        });
                    });
                ui.add_space(2.0);
            }
            if let Some(idx) = delete_idx {
                self.panels.agent_list.remove(idx);
                self.panels.agent_selected = None;
            }
        });
    }

    fn render_tools_panel(&mut self, ui: &mut Ui) {
        ui.heading("Tools / MCP");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.panels.tool_search);
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Built-in tool categories
            let tool_groups: &[(&str, &[&str])] = &[
                ("Browser Automation", &["navigate", "click", "type_text", "screenshot", "get_text", "execute_script"]),
                ("Code Sandbox", &["execute_python", "execute_javascript", "execute_bash"]),
                ("File Operations", &["read_file", "write_file", "list_directory"]),
                ("RAG / Knowledge", &["search_knowledge", "add_document", "list_sources"]),
                ("Web / HTTP", &["http_get", "http_post", "fetch_url"]),
                ("System", &["get_time", "get_env", "run_command"]),
            ];

            let search_lower = self.panels.tool_search.to_lowercase();
            for (group_name, tools) in tool_groups {
                let matching: Vec<&&str> = tools.iter()
                    .filter(|t| search_lower.is_empty() || t.to_lowercase().contains(&search_lower) || group_name.to_lowercase().contains(&search_lower))
                    .collect();
                if matching.is_empty() { continue; }

                ui.collapsing(*group_name, |ui| {
                    for tool_name in matching {
                        egui::Frame::none()
                            .fill(Color32::from_gray(30))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(6.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.colored_label(Color32::from_rgb(120, 200, 255), *tool_name);
                                    ui.colored_label(Color32::from_gray(100), format!("({} tool)", group_name.split_whitespace().next().unwrap_or("")));
                                });
                            });
                        ui.add_space(1.0);
                    }
                });
            }

            ui.add_space(12.0);
            ui.separator();
            ui.label("MCP Servers");
            ui.add_space(4.0);
            ui.colored_label(Color32::from_gray(140), "Configure MCP server connections in Settings > Provider.");
            ui.label("Supported transports: StdIO, SSE, WebSocket");
            ui.label("Supported auth: OAuth 2.0 with PKCE");
        });
    }

    fn render_automation_panel(&mut self, ui: &mut Ui) {
        ui.heading("Automation");
        ui.add_space(4.0);

        // Sub-tabs
        ui.horizontal(|ui| {
            for (tab, label) in &[(AutomationTab::Scheduler, "Scheduler"), (AutomationTab::Browser, "Browser"), (AutomationTab::Sandbox, "Code Sandbox"), (AutomationTab::Workflows, "Workflows")] {
                if ui.selectable_label(self.panels.auto_tab == *tab, *label).clicked() {
                    self.panels.auto_tab = *tab;
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            match self.panels.auto_tab {
                AutomationTab::Scheduler => {
                    ui.label("Cron Scheduler");
                    ui.add_space(4.0);
                    egui::Frame::none()
                        .fill(Color32::from_gray(30))
                        .rounding(8.0)
                        .inner_margin(egui::Margin::same(10.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Task name:");
                                ui.text_edit_singleline(&mut self.panels.cron_task_name);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Cron expr:");
                                ui.text_edit_singleline(&mut self.panels.cron_expression);
                            });
                            // Validate cron expression
                            match CronExpression::parse(&self.panels.cron_expression) {
                                Ok(_) => {
                                    ui.colored_label(Color32::from_rgb(100, 200, 100), "Valid cron expression");
                                }
                                Err(e) => {
                                    ui.colored_label(Color32::from_rgb(255, 100, 100), format!("Invalid: {}", e));
                                }
                            }
                            if ui.button("Add Job").clicked() && !self.panels.cron_task_name.is_empty() {
                                self.add_toast(&format!("Scheduled: {}", self.panels.cron_task_name), false);
                                self.panels.cron_task_name.clear();
                            }
                        });
                    ui.add_space(8.0);
                    ui.label("Common patterns:");
                    ui.colored_label(Color32::from_gray(140), "  0 * * * *    — every hour");
                    ui.colored_label(Color32::from_gray(140), "  */5 * * * *  — every 5 minutes");
                    ui.colored_label(Color32::from_gray(140), "  0 9 * * 1-5  — weekdays at 9am");
                    ui.colored_label(Color32::from_gray(140), "  0 0 * * *    — daily at midnight");
                }
                AutomationTab::Browser => {
                    ui.label("Browser Automation (Chrome DevTools Protocol)");
                    ui.add_space(4.0);
                    egui::Frame::none()
                        .fill(Color32::from_gray(30))
                        .rounding(8.0)
                        .inner_margin(egui::Margin::same(10.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("URL:");
                                ui.text_edit_singleline(&mut self.panels.browser_url);
                                if ui.button("Navigate").clicked() && !self.panels.browser_url.is_empty() {
                                    self.panels.browser_output = format!("Navigating to {}...\n(Browser session must be started from agent context)", self.panels.browser_url);
                                }
                            });
                            ui.horizontal(|ui| {
                                if ui.button("Screenshot").clicked() {
                                    self.panels.browser_output = "Screenshot capture requires active browser session.".to_string();
                                }
                                if ui.button("Get Page Text").clicked() {
                                    self.panels.browser_output = "Page text extraction requires active browser session.".to_string();
                                }
                            });
                        });
                    ui.add_space(4.0);
                    if !self.panels.browser_output.is_empty() {
                        egui::Frame::none()
                            .fill(Color32::from_gray(25))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.monospace(&self.panels.browser_output);
                            });
                    }
                }
                AutomationTab::Sandbox => {
                    ui.label("Code Sandbox — Safe Execution Environment");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Language:");
                        egui::ComboBox::from_id_source("sandbox_lang")
                            .selected_text(&self.panels.sandbox_language)
                            .show_ui(ui, |ui| {
                                for lang in &["python", "javascript", "bash"] {
                                    ui.selectable_value(&mut self.panels.sandbox_language, lang.to_string(), *lang);
                                }
                            });
                    });
                    ui.add_space(4.0);
                    ui.label("Code:");
                    ui.add(egui::TextEdit::multiline(&mut self.panels.sandbox_code)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(8)
                        .desired_width(f32::INFINITY));

                    ui.horizontal(|ui| {
                        if ui.button("Execute").clicked() && !self.panels.sandbox_code.is_empty() {
                            let sandbox = CodeSandbox::new();
                            let lang = match self.panels.sandbox_language.as_str() {
                                "javascript" => SandboxLanguage::JavaScript,
                                "bash" => SandboxLanguage::Bash,
                                _ => SandboxLanguage::Python,
                            };
                            let result = sandbox.execute(&lang, &self.panels.sandbox_code);
                            let mut out = String::new();
                            if !result.stdout.is_empty() { out.push_str(&format!("STDOUT:\n{}\n", result.stdout)); }
                            if !result.stderr.is_empty() { out.push_str(&format!("STDERR:\n{}\n", result.stderr)); }
                            out.push_str(&format!("Exit: {} | Duration: {:?}", result.exit_code, result.duration));
                            if result.timed_out { out.push_str(" | TIMED OUT"); }
                            self.panels.sandbox_output = out;
                        }
                        if ui.button("Clear").clicked() {
                            self.panels.sandbox_output.clear();
                            self.panels.sandbox_code.clear();
                        }
                    });

                    if !self.panels.sandbox_output.is_empty() {
                        ui.add_space(4.0);
                        ui.label("Output:");
                        egui::Frame::none()
                            .fill(Color32::from_gray(20))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.monospace(&self.panels.sandbox_output);
                            });
                    }
                }
                AutomationTab::Workflows => {
                    ui.label("Event-Driven Workflows");
                    ui.add_space(4.0);
                    ui.label("Define workflows with nodes and event-driven transitions.");
                    ui.label("Supports checkpointing, breakpoints, and durable execution.");
                    ui.add_space(8.0);
                    ui.colored_label(Color32::from_gray(140), "Workflow editor: design workflows visually by chaining nodes.");
                    ui.colored_label(Color32::from_gray(140), "Each node processes an event and produces outputs for downstream nodes.");
                    ui.add_space(8.0);
                    ui.label("Workflow templates:");
                    for (name, desc) in &[
                        ("Document Pipeline", "Ingest -> Parse -> Chunk -> Embed -> Store"),
                        ("Agent Loop", "Plan -> Execute -> Evaluate -> Refine"),
                        ("RAG Pipeline", "Query -> Retrieve -> Rank -> Generate -> Validate"),
                    ] {
                        egui::Frame::none()
                            .fill(Color32::from_gray(30))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(6.0))
                            .show(ui, |ui| {
                                ui.strong(*name);
                                ui.colored_label(Color32::from_gray(160), *desc);
                            });
                        ui.add_space(2.0);
                    }
                }
            }
        });
    }

    fn render_memory_panel(&mut self, ui: &mut Ui) {
        ui.heading("Memory");
        ui.add_space(4.0);

        // Sub-tabs
        ui.horizontal(|ui| {
            for (tab, label) in &[(MemoryTab::Episodic, "Episodic"), (MemoryTab::Procedural, "Procedural"), (MemoryTab::Entity, "Entity")] {
                if ui.selectable_label(self.panels.memory_tab == *tab, *label).clicked() {
                    self.panels.memory_tab = *tab;
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.panels.memory_search);
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            match self.panels.memory_tab {
                MemoryTab::Episodic => {
                    ui.label("Episodic Memory — Experiences and interactions");
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_gray(140), "Episodic memory stores conversation episodes with timestamps.");
                    ui.colored_label(Color32::from_gray(140), "Each episode captures context, emotions, and outcomes.");
                    ui.add_space(8.0);
                    let episode_count = self.assistant.conversation.len() / 2;
                    ui.label(format!("Current session episodes: {}", episode_count));
                    ui.add_space(4.0);
                    // Show recent episodes from conversation
                    for (i, msg) in self.assistant.conversation.iter().enumerate().rev().take(10) {
                        let prefix = if msg.role == "user" { "Q" } else { "A" };
                        let snippet: String = msg.content.chars().take(80).collect();
                        egui::Frame::none()
                            .fill(Color32::from_gray(30))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(6.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.colored_label(Color32::from_rgb(120, 180, 255), format!("#{} [{}]", i, prefix));
                                    ui.label(format!("{}...", snippet));
                                });
                            });
                        ui.add_space(1.0);
                    }
                }
                MemoryTab::Procedural => {
                    ui.label("Procedural Memory — Learned skills and procedures");
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_gray(140), "Procedural memory stores learned patterns, frequently used tools,");
                    ui.colored_label(Color32::from_gray(140), "and successful strategies for future reuse.");
                    ui.add_space(8.0);
                    ui.label("No procedures recorded yet in this session.");
                    ui.label("Procedures are learned from repeated successful task completions.");
                }
                MemoryTab::Entity => {
                    ui.label("Entity Memory — Known entities and relationships");
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_gray(140), "Entity memory tracks people, places, concepts, and their relationships.");
                    ui.add_space(8.0);
                    // Show entities from knowledge graph if available
                    if let Some(ref viz) = self.graph_viz {
                        let count = viz.graph.node_count();
                        ui.label(format!("{} entities in knowledge graph", count));
                        ui.add_space(4.0);
                        let search = self.panels.memory_search.to_lowercase();
                        for idx in viz.graph.node_indices().take(20) {
                            let node = &viz.graph[idx];
                            if !search.is_empty() && !node.name.to_lowercase().contains(&search) {
                                continue;
                            }
                            egui::Frame::none()
                                .fill(Color32::from_gray(30))
                                .rounding(4.0)
                                .inner_margin(egui::Margin::same(6.0))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(Color32::from_rgb(200, 160, 255), &node.entity_type);
                                        ui.strong(&node.name);
                                        ui.colored_label(Color32::from_gray(120), format!("({} mentions)", node.mention_count));
                                    });
                                });
                            ui.add_space(1.0);
                        }
                    } else {
                        ui.label("No entities yet. Load knowledge sources and enable the knowledge graph.");
                    }
                }
            }
        });
    }

    fn render_cloud_panel(&mut self, ui: &mut Ui) {
        ui.heading("Cloud Storage");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Provider:");
            egui::ComboBox::from_id_source("cloud_provider")
                .selected_text(&self.panels.cloud_provider)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.panels.cloud_provider, "s3".to_string(), "Amazon S3");
                    ui.selectable_value(&mut self.panels.cloud_provider, "gdrive".to_string(), "Google Drive");
                    ui.selectable_value(&mut self.panels.cloud_provider, "azure".to_string(), "Azure Blob");
                    ui.selectable_value(&mut self.panels.cloud_provider, "gcs".to_string(), "Google Cloud Storage");
                });
        });

        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            match self.panels.cloud_provider.as_str() {
                "s3" => {
                    ui.label("Amazon S3 Configuration");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Bucket:");
                        ui.text_edit_singleline(&mut self.panels.cloud_bucket);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Path:");
                        ui.text_edit_singleline(&mut self.panels.cloud_path);
                    });
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_gray(140), "Configure AWS credentials via environment variables:");
                    ui.colored_label(Color32::from_gray(120), "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION");
                    ui.add_space(8.0);
                    if self.panels.cloud_bucket.is_empty() {
                        ui.label("Enter a bucket name to browse files.");
                    } else {
                        if ui.button("List Objects").clicked() {
                            self.add_toast("S3 listing requires configured credentials", false);
                        }
                    }
                }
                "gdrive" => {
                    ui.label("Google Drive Configuration");
                    ui.add_space(4.0);
                    ui.colored_label(Color32::from_gray(140), "Configure OAuth token for Google Drive access.");
                    ui.colored_label(Color32::from_gray(140), "Supports file listing, upload, and download.");
                    ui.add_space(8.0);
                    if ui.button("Authenticate").clicked() {
                        self.add_toast("Google Drive OAuth flow not yet wired", false);
                    }
                }
                _ => {
                    ui.label(format!("{} storage — configure credentials in Settings > Provider", self.panels.cloud_provider.to_uppercase()));
                }
            }
        });
    }

    fn render_media_panel(&mut self, ui: &mut Ui) {
        ui.heading("Media Generation");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_id_source("media_model")
                .selected_text(&self.panels.media_model)
                .show_ui(ui, |ui| {
                    for m in &["dall-e-3", "dall-e-2", "stable-diffusion", "flux", "runway", "sora"] {
                        ui.selectable_value(&mut self.panels.media_model, m.to_string(), *m);
                    }
                });
        });

        ui.add_space(4.0);
        ui.label("Prompt:");
        ui.add(egui::TextEdit::multiline(&mut self.panels.media_prompt)
            .desired_rows(3)
            .desired_width(f32::INFINITY));

        ui.horizontal(|ui| {
            if ui.button("Generate Image").clicked() && !self.panels.media_prompt.is_empty() {
                self.add_toast(&format!("Image generation via {} requires API key configuration", self.panels.media_model), false);
            }
            if ui.button("Clear Gallery").clicked() {
                self.panels.media_results.clear();
            }
        });

        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            if self.panels.media_results.is_empty() {
                ui.colored_label(Color32::from_gray(140), "No generated images yet.");
                ui.add_space(8.0);
                ui.label("Supported providers:");
                for (name, desc) in &[
                    ("DALL-E 3", "OpenAI — highest quality, prompt rewriting"),
                    ("DALL-E 2", "OpenAI — fast, lower cost"),
                    ("Stable Diffusion", "Local or API — open source, customizable"),
                    ("Flux", "Replicate — fast, high quality"),
                    ("Runway", "Video generation from text or image"),
                    ("Sora", "OpenAI — text-to-video"),
                ] {
                    egui::Frame::none()
                        .fill(Color32::from_gray(30))
                        .rounding(4.0)
                        .inner_margin(egui::Margin::same(6.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.strong(*name);
                                ui.colored_label(Color32::from_gray(140), *desc);
                            });
                        });
                    ui.add_space(1.0);
                }
            } else {
                ui.label(format!("{} generated images:", self.panels.media_results.len()));
                for path in &self.panels.media_results {
                    ui.label(path);
                }
            }
        });
    }

    fn render_audio_panel(&mut self, ui: &mut Ui) {
        ui.heading("Audio — STT / TTS");
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Speech-to-Text section
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Speech-to-Text (STT)");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Audio file:");
                        ui.text_edit_singleline(&mut self.panels.audio_input_path);
                        if ui.button("Browse...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Audio", &["wav", "mp3", "ogg", "flac", "m4a"])
                                .pick_file()
                            {
                                self.panels.audio_input_path = path.display().to_string();
                            }
                        }
                    });
                    if ui.button("Transcribe").clicked() {
                        if self.panels.audio_input_path.is_empty() {
                            self.add_toast("Select an audio file first", true);
                        } else {
                            self.add_toast("STT requires a configured speech provider (OpenAI, Google, or local Whisper)", false);
                        }
                    }
                    if !self.panels.audio_output_text.is_empty() {
                        ui.add_space(4.0);
                        ui.label("Transcription:");
                        egui::Frame::none()
                            .fill(Color32::from_gray(20))
                            .rounding(4.0)
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.label(&self.panels.audio_output_text);
                            });
                    }
                });

            ui.add_space(12.0);

            // Text-to-Speech section
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Text-to-Speech (TTS)");
                    ui.add_space(4.0);
                    ui.label("Text:");
                    ui.add(egui::TextEdit::multiline(&mut self.panels.tts_input)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY));
                    ui.horizontal(|ui| {
                        ui.label("Voice:");
                        egui::ComboBox::from_id_source("tts_voice")
                            .selected_text(&self.panels.tts_voice)
                            .show_ui(ui, |ui| {
                                for v in &["alloy", "echo", "fable", "onyx", "nova", "shimmer"] {
                                    ui.selectable_value(&mut self.panels.tts_voice, v.to_string(), *v);
                                }
                            });
                    });
                    if ui.button("Synthesize").clicked() {
                        if self.panels.tts_input.is_empty() {
                            self.add_toast("Enter text to synthesize", true);
                        } else {
                            self.add_toast("TTS requires a configured speech provider", false);
                        }
                    }
                });

            ui.add_space(12.0);
            ui.label("Supported providers:");
            ui.colored_label(Color32::from_gray(140), "  OpenAI (Whisper STT + TTS)");
            ui.colored_label(Color32::from_gray(140), "  Google Cloud Speech");
            ui.colored_label(Color32::from_gray(140), "  Piper TTS (local, offline)");
            ui.colored_label(Color32::from_gray(140), "  Coqui TTS (local, offline)");
            ui.colored_label(Color32::from_gray(140), "  Whisper.cpp (local STT, offline)");
        });
    }

    fn render_prompt_lab_panel(&mut self, ui: &mut Ui) {
        ui.heading("Prompt Lab");
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Signature editor
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Prompt Signature");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.panels.sig_name);
                    });

                    ui.add_space(4.0);
                    ui.label("Input fields:");
                    let mut remove_input = None;
                    for (i, (name, desc)) in self.panels.sig_inputs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(name);
                            ui.text_edit_singleline(desc);
                            if ui.small_button("X").clicked() { remove_input = Some(i); }
                        });
                    }
                    if let Some(idx) = remove_input { self.panels.sig_inputs.remove(idx); }
                    if ui.small_button("+ Add input").clicked() {
                        self.panels.sig_inputs.push(("input".to_string(), "description".to_string()));
                    }

                    ui.add_space(4.0);
                    ui.label("Output fields:");
                    let mut remove_output = None;
                    for (i, (name, desc)) in self.panels.sig_outputs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(name);
                            ui.text_edit_singleline(desc);
                            if ui.small_button("X").clicked() { remove_output = Some(i); }
                        });
                    }
                    if let Some(idx) = remove_output { self.panels.sig_outputs.remove(idx); }
                    if ui.small_button("+ Add output").clicked() {
                        self.panels.sig_outputs.push(("output".to_string(), "description".to_string()));
                    }
                });

            ui.add_space(12.0);

            // Optimization
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Optimization");
                    ui.add_space(4.0);
                    ui.label("Optimizers: BootstrapFewShot, Grid Search, Bayesian, GEPA, MIPROv2, SIMBA");
                    ui.label("Metrics: ExactMatch, F1Score, ContainsAnswer, Custom");
                    ui.add_space(4.0);
                    if self.panels.optimization_running {
                        ui.colored_label(Color32::from_rgb(255, 200, 50), "Optimization in progress...");
                    } else if ui.button("Run Optimization").clicked() {
                        if self.panels.sig_name.is_empty() {
                            self.add_toast("Name the signature first", true);
                        } else {
                            self.add_toast("Optimization requires training data and a configured LLM", false);
                        }
                    }

                    if !self.panels.optimization_results.is_empty() {
                        ui.add_space(4.0);
                        ui.label("Results:");
                        for (config, score) in &self.panels.optimization_results {
                            ui.horizontal(|ui| {
                                ui.label(config);
                                ui.colored_label(Color32::from_rgb(100, 200, 100), format!("{:.3}", score));
                            });
                        }
                    }
                });
        });
    }

    fn render_eval_panel(&mut self, ui: &mut Ui) {
        ui.heading("Evaluation");
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Eval suite
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Evaluation Suite");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Suite name:");
                        ui.text_edit_singleline(&mut self.panels.eval_suite_name);
                    });

                    ui.add_space(4.0);
                    ui.label(format!("{} test samples", self.panels.eval_samples.len()));
                    let mut remove_sample = None;
                    for (i, (input, expected)) in self.panels.eval_samples.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(input).hint_text("Input"));
                            ui.add(egui::TextEdit::singleline(expected).hint_text("Expected"));
                            if ui.small_button("X").clicked() { remove_sample = Some(i); }
                        });
                    }
                    if let Some(idx) = remove_sample { self.panels.eval_samples.remove(idx); }
                    if ui.small_button("+ Add sample").clicked() {
                        self.panels.eval_samples.push((String::new(), String::new()));
                    }

                    ui.add_space(4.0);
                    if ui.button("Run Evaluation").clicked() {
                        if self.panels.eval_samples.is_empty() {
                            self.add_toast("Add test samples first", true);
                        } else {
                            self.add_toast("Evaluation requires a configured LLM provider", false);
                        }
                    }
                });

            ui.add_space(12.0);

            // Results
            if !self.panels.eval_results.is_empty() {
                egui::Frame::none()
                    .fill(Color32::from_gray(30))
                    .rounding(8.0)
                    .inner_margin(egui::Margin::same(10.0))
                    .show(ui, |ui| {
                        ui.strong("Results");
                        for (sample, score, output) in &self.panels.eval_results {
                            ui.horizontal(|ui| {
                                let color = if *score > 0.8 { Color32::from_rgb(100, 200, 100) }
                                    else if *score > 0.5 { Color32::from_rgb(200, 200, 100) }
                                    else { Color32::from_rgb(255, 100, 100) };
                                ui.colored_label(color, format!("{:.1}%", score * 100.0));
                                ui.label(sample);
                                ui.colored_label(Color32::from_gray(120), output);
                            });
                        }
                    });
            }

            ui.add_space(12.0);

            // Available tools
            ui.label("Evaluation capabilities:");
            for cap in &[
                "A/B Testing — compare model outputs with statistical significance",
                "Benchmarks — code, math, reasoning, multilingual benchmarks",
                "Hallucination Detection — claim extraction and verification",
                "Red Teaming — automated adversarial testing",
                "Self-Consistency — multi-sample voting for reliability",
                "Output Validation — format and schema compliance",
                "Confidence Scoring — calibrated uncertainty estimates",
            ] {
                ui.colored_label(Color32::from_gray(140), format!("  {}", cap));
            }
        });
    }

    fn render_security_panel(&mut self, ui: &mut Ui) {
        ui.heading("Security");
        ui.add_space(4.0);

        // Sub-tabs
        ui.horizontal(|ui| {
            let tabs = [
                (SecurityTab::Overview, "Overview"),
                (SecurityTab::Pii, "PII Detection"),
                (SecurityTab::Guardrails, "Guardrails"),
                (SecurityTab::Rbac, "RBAC"),
                (SecurityTab::AuditLog, "Audit Log"),
            ];
            for (tab, label) in &tabs {
                if ui.selectable_label(self.panels.security_tab == *tab, *label).clicked() {
                    self.panels.security_tab = *tab;
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            match self.panels.security_tab {
                SecurityTab::Overview => {
                    egui::Frame::none()
                        .fill(Color32::from_gray(30))
                        .rounding(8.0)
                        .inner_margin(egui::Margin::same(12.0))
                        .show(ui, |ui| {
                            ui.heading("Security Status");
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                let pii_color = if self.panels.pii_enabled { Color32::from_rgb(100, 220, 100) } else { Color32::from_rgb(180, 180, 180) };
                                ui.colored_label(pii_color, if self.panels.pii_enabled { "\u{2705} PII Detection: ON" } else { "\u{274c} PII Detection: OFF" });
                            });
                            ui.horizontal(|ui| {
                                let guard_color = if self.panels.guardrails_enabled { Color32::from_rgb(100, 220, 100) } else { Color32::from_rgb(180, 180, 180) };
                                ui.colored_label(guard_color, if self.panels.guardrails_enabled { "\u{2705} Guardrails: ON" } else { "\u{274c} Guardrails: OFF" });
                            });
                            ui.label(format!("Rate limit: {} RPM", self.panels.rate_limit_rpm));
                            ui.label(format!("Audit events: {}", self.audit_events.len()));
                            ui.label(format!("RBAC roles: {}", self.panels.rbac_roles.len()));
                        });
                }
                SecurityTab::Pii => {
                    ui.checkbox(&mut self.panels.pii_enabled, "Enable PII detection");
                    ui.add_space(8.0);
                    if self.panels.pii_enabled {
                        ui.label("Detected PII types: Email, Phone, SSN, Credit Card, IP Address");
                        ui.label("Redaction strategy: Replace with [REDACTED]");
                        ui.add_space(4.0);
                        ui.label("PII detection runs automatically on all messages when enabled.");
                    } else {
                        ui.colored_label(Color32::from_rgb(200, 200, 100), "PII detection is disabled. Enable it to scan messages for personally identifiable information.");
                    }
                }
                SecurityTab::Guardrails => {
                    ui.checkbox(&mut self.panels.guardrails_enabled, "Enable content guardrails");
                    ui.add_space(8.0);
                    if self.panels.guardrails_enabled {
                        ui.label("Active guardrails:");
                        ui.label("  - Toxicity filter");
                        ui.label("  - Bias detection");
                        ui.label("  - Jailbreak/attack prevention");
                        ui.label("  - Constitutional AI principles");
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Rate limit (RPM):");
                        ui.add(egui::DragValue::new(&mut self.panels.rate_limit_rpm).clamp_range(1..=1000));
                    });
                }
                SecurityTab::Rbac => {
                    ui.label("Role-Based Access Control");
                    ui.add_space(4.0);
                    if self.panels.rbac_roles.is_empty() {
                        ui.label("No custom roles defined. Default roles: Admin, User, Guest.");
                        if ui.button("Add default roles").clicked() {
                            self.panels.rbac_roles.push(("Admin".to_string(), vec!["read".to_string(), "write".to_string(), "execute".to_string(), "delete".to_string()]));
                            self.panels.rbac_roles.push(("User".to_string(), vec!["read".to_string(), "write".to_string()]));
                            self.panels.rbac_roles.push(("Guest".to_string(), vec!["read".to_string()]));
                        }
                    } else {
                        for (role, perms) in &self.panels.rbac_roles {
                            egui::Frame::none()
                                .fill(Color32::from_gray(35))
                                .rounding(4.0)
                                .inner_margin(egui::Margin::same(8.0))
                                .show(ui, |ui| {
                                    ui.strong(role);
                                    ui.label(format!("Permissions: {}", perms.join(", ")));
                                });
                            ui.add_space(2.0);
                        }
                    }
                }
                SecurityTab::AuditLog => {
                    ui.label(format!("{} audit events", self.audit_events.len()));
                    ui.add_space(4.0);
                    if self.audit_events.is_empty() {
                        ui.label("No audit events recorded yet.");
                    } else {
                        for event in self.audit_events.iter().rev().take(50) {
                            ui.horizontal(|ui| {
                                ui.monospace(format!("{:?}", event.event_type));
                                ui.colored_label(Color32::from_gray(140), event.timestamp.format("%H:%M:%S").to_string());
                            });
                        }
                    }
                }
            }
        });
    }

    fn render_analytics_panel(&mut self, ui: &mut Ui) {
        ui.heading("Analytics");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Time range:");
            for range in &["1h", "24h", "7d", "30d"] {
                if ui.selectable_label(self.panels.analytics_time_range == *range, *range).clicked() {
                    self.panels.analytics_time_range = range.to_string();
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Session metrics
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Session Metrics");
                    ui.add_space(4.0);
                    let msg_count = self.assistant.conversation.len();
                    let user_msgs = self.assistant.conversation.iter().filter(|m| m.role == "user").count();
                    let ai_msgs = self.assistant.conversation.iter().filter(|m| m.role == "assistant").count();
                    ui.horizontal(|ui| {
                        ui.label(format!("Messages: {}", msg_count));
                        ui.separator();
                        ui.label(format!("User: {}", user_msgs));
                        ui.separator();
                        ui.label(format!("AI: {}", ai_msgs));
                    });

                    // Response time stats from details
                    if !self.response_details.is_empty() {
                        let times: Vec<u64> = self.response_details.iter().map(|d| d.response_time_ms).collect();
                        let avg = times.iter().sum::<u64>() / times.len() as u64;
                        let min = times.iter().min().unwrap_or(&0);
                        let max = times.iter().max().unwrap_or(&0);
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(format!("Avg response: {}ms", avg));
                            ui.separator();
                            ui.label(format!("Min: {}ms", min));
                            ui.separator();
                            ui.label(format!("Max: {}ms", max));
                        });

                        // Token usage
                        let total_in: usize = self.response_details.iter().map(|d| d.input_tokens).sum();
                        let total_out: usize = self.response_details.iter().map(|d| d.output_tokens).sum();
                        ui.horizontal(|ui| {
                            ui.label(format!("Total tokens — in: {}, out: {}", total_in, total_out));
                        });
                    }
                });

            ui.add_space(12.0);

            // Knowledge stats
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Knowledge Base");
                    ui.add_space(4.0);
                    ui.label(format!("Sources loaded: {}", self.knowledge_sources.len()));
                    let indexed = self.knowledge_sources.iter().filter(|k| k.status == KnowledgeStatus::Indexed).count();
                    ui.label(format!("Indexed: {}", indexed));
                    if let Some(ref viz) = self.graph_viz {
                        ui.label(format!("Graph entities: {}", viz.graph.node_count()));
                        ui.label(format!("Graph relations: {}", viz.graph.edge_count()));
                    }
                });

            ui.add_space(12.0);

            // Monitoring stats
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Monitoring");
                    ui.add_space(4.0);
                    ui.label(format!("Audit events: {}", self.audit_events.len()));
                    if let Some(ref sentiment) = self.sentiment_cache {
                        ui.label(format!("Sentiment: {:?}", sentiment.overall));
                    }
                    ui.label(format!("Topics detected: {}", self.topics_cache.len()));
                    ui.label(format!("Models available: {}", self.assistant.available_models.len()));
                });
        });
    }

    fn render_devtools_panel(&mut self, ui: &mut Ui) {
        ui.heading("DevTools — Agent Debugger");
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Recording controls
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let recording_label = if self.panels.devtools_recording { "Stop Recording" } else { "Start Recording" };
                        let recording_color = if self.panels.devtools_recording { Color32::from_rgb(255, 80, 80) } else { Color32::from_rgb(100, 200, 100) };
                        if ui.colored_label(recording_color, recording_label).clicked() {
                            self.panels.devtools_recording = !self.panels.devtools_recording;
                            if self.panels.devtools_recording {
                                self.add_toast("Recording agent events...", false);
                            } else {
                                self.add_toast("Recording stopped", false);
                            }
                        }
                        ui.separator();
                        ui.label(format!("{} events", self.panels.devtools_events.len()));
                        if ui.small_button("Clear").clicked() {
                            self.panels.devtools_events.clear();
                        }
                    });
                });

            ui.add_space(8.0);

            // Breakpoints
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Breakpoints");
                    ui.add_space(4.0);
                    if self.panels.devtools_breakpoints.is_empty() {
                        ui.colored_label(Color32::from_gray(140), "No breakpoints set.");
                    }
                    let mut remove_bp = None;
                    for (i, bp) in self.panels.devtools_breakpoints.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.colored_label(Color32::from_rgb(255, 100, 100), "\u{25cf}"); // red dot
                            ui.label(bp);
                            if ui.small_button("X").clicked() { remove_bp = Some(i); }
                        });
                    }
                    if let Some(idx) = remove_bp { self.panels.devtools_breakpoints.remove(idx); }
                    ui.add_space(4.0);
                    ui.label("Breakpoint types: tool_call, agent_state_change, error, confidence_low");
                });

            ui.add_space(8.0);

            // Event log
            if !self.panels.devtools_events.is_empty() {
                egui::Frame::none()
                    .fill(Color32::from_gray(25))
                    .rounding(8.0)
                    .inner_margin(egui::Margin::same(10.0))
                    .show(ui, |ui| {
                        ui.strong("Event Log");
                        for event in self.panels.devtools_events.iter().rev().take(30) {
                            ui.monospace(event);
                        }
                    });
            }

            ui.add_space(8.0);
            ui.label("Capabilities:");
            ui.colored_label(Color32::from_gray(140), "  ExecutionRecorder — record agent actions for replay");
            ui.colored_label(Color32::from_gray(140), "  PerformanceProfiler — step-by-step timing");
            ui.colored_label(Color32::from_gray(140), "  StateInspector — inspect agent state at any point");
            ui.colored_label(Color32::from_gray(140), "  StateDiff — compare state between steps");
        });
    }

    fn render_cluster_panel(&mut self, ui: &mut Ui) {
        ui.heading("Cluster — Distributed Nodes");
        ui.add_space(4.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Bootstrap
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong("Cluster Configuration");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Bootstrap address:");
                        ui.text_edit_singleline(&mut self.panels.cluster_bootstrap);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Join Cluster").clicked() {
                            if self.panels.cluster_bootstrap.is_empty() {
                                self.add_toast("Enter a bootstrap address", true);
                            } else {
                                self.add_toast(&format!("Joining cluster at {}", self.panels.cluster_bootstrap), false);
                            }
                        }
                        if ui.button("Start as Seed").clicked() {
                            self.add_toast("Starting as seed node on this machine", false);
                        }
                    });
                });

            ui.add_space(8.0);

            // Node list
            egui::Frame::none()
                .fill(Color32::from_gray(30))
                .rounding(8.0)
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.strong(format!("Nodes ({} known)", self.panels.cluster_nodes.len()));
                    ui.add_space(4.0);
                    if self.panels.cluster_nodes.is_empty() {
                        ui.colored_label(Color32::from_gray(140), "No nodes connected. Join or start a cluster.");
                    }
                    for (id, addr, status) in &self.panels.cluster_nodes {
                        ui.horizontal(|ui| {
                            let color = match status.as_str() {
                                "online" => Color32::from_rgb(100, 200, 100),
                                "offline" => Color32::from_rgb(200, 100, 100),
                                _ => Color32::from_rgb(200, 200, 100),
                            };
                            ui.colored_label(color, "\u{25cf}");
                            ui.label(format!("{} — {}", id, addr));
                            ui.colored_label(Color32::from_gray(120), status);
                        });
                    }
                });

            ui.add_space(8.0);

            // Distributed features
            ui.label("Distributed capabilities:");
            for cap in &[
                "DHT (Kademlia) — distributed hash table for key-value storage",
                "CRDTs — conflict-free replicated data types (GCounter, PNCounter, ORSet, LWWMap)",
                "MapReduce — distributed computation over data chunks",
                "QUIC Transport — multiplexed, encrypted node-to-node communication",
                "Consistent Hashing — data sharding across nodes",
                "Failure Detection — phi-accrual heartbeat monitoring",
                "Merkle Sync — anti-entropy data synchronization",
            ] {
                ui.colored_label(Color32::from_gray(140), format!("  {}", cap));
            }
        });
    }

    fn render_settings_panel(&mut self, ui: &mut Ui) {
        ui.heading("Settings");
        ui.add_space(4.0);

        // Section selector
        ui.horizontal(|ui| {
            let sections = ["Provider", "Generation", "Security", "RAG", "Agents", "Memory", "UI"];
            for (i, label) in sections.iter().enumerate() {
                if ui.selectable_label(self.panels.settings_section == i as u8, *label).clicked() {
                    self.panels.settings_section = i as u8;
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            match self.panels.settings_section {
                0 => {
                    // Provider settings
                    ui.label("Provider URLs");
                    ui.horizontal(|ui| {
                        ui.label("Ollama:");
                        ui.text_edit_singleline(&mut self.settings.ollama_url);
                    });
                    ui.horizontal(|ui| {
                        ui.label("LM Studio:");
                        ui.text_edit_singleline(&mut self.settings.lm_studio_url);
                    });
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Default model:");
                        ui.label(&self.selected_model);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Request timeout (s):");
                        let mut timeout = self.settings.timeout_secs;
                        ui.add(egui::DragValue::new(&mut timeout).clamp_range(5..=300));
                        self.settings.timeout_secs = timeout;
                    });
                }
                1 => {
                    // Generation settings
                    ui.label("System prompt:");
                    ui.add(egui::TextEdit::multiline(&mut self.panels.system_prompt)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Temperature:");
                        ui.add(egui::Slider::new(&mut self.settings.temperature, 0.0..=2.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Top P:");
                        ui.add(egui::Slider::new(&mut self.panels.top_p, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max tokens:");
                        ui.add(egui::DragValue::new(&mut self.panels.max_tokens).clamp_range(64..=32768));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Frequency penalty:");
                        ui.add(egui::Slider::new(&mut self.panels.frequency_penalty, -2.0..=2.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Presence penalty:");
                        ui.add(egui::Slider::new(&mut self.panels.presence_penalty, -2.0..=2.0));
                    });
                }
                2 => {
                    // Security settings
                    ui.checkbox(&mut self.panels.pii_enabled, "Enable PII detection");
                    ui.checkbox(&mut self.panels.guardrails_enabled, "Enable guardrails");
                    ui.horizontal(|ui| {
                        ui.label("Rate limit (RPM):");
                        ui.add(egui::DragValue::new(&mut self.panels.rate_limit_rpm).clamp_range(1..=1000));
                    });
                }
                3 => {
                    // RAG settings
                    ui.checkbox(&mut self.rag_enabled, "Enable RAG");
                    ui.horizontal(|ui| {
                        ui.label("History depth:");
                        ui.add(egui::DragValue::new(&mut self.settings.history_depth).clamp_range(1..=100));
                    });
                    ui.checkbox(&mut self.settings.graph_enabled, "Enable knowledge graph");
                }
                4 => {
                    // Agent settings
                    ui.label("Default orchestration strategy, agent pool limits, and approval policies are configured in the Agent Pool panel.");
                }
                5 => {
                    // Memory settings
                    ui.label("Memory stores (episodic, procedural, entity) are browsed in the Memory panel. Configure retention and consolidation here.");
                    ui.add_space(4.0);
                    ui.label("(Memory settings integration pending — use defaults)");
                }
                6 => {
                    // UI settings
                    ui.checkbox(&mut self.settings.enter_sends, "Enter sends message (Shift+Enter for newline)");
                    ui.checkbox(&mut self.show_monitor, "Show monitor panel");
                }
                _ => {}
            }
        });
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
                    self.settings.enter_sends,
                );
                if ui.button("Stop").clicked() {
                    self.add_toast("Generation will stop after current chunk", false);
                }
            });
        } else {
            let send_hint = if self.settings.enter_sends {
                "Type your message... (Enter to send)"
            } else {
                "Type your message... (Ctrl+Enter to send)"
            };
            let response = widgets::chat_input_multiline(
                ui,
                &mut self.input_text,
                false,
                send_hint,
                50.0,
                self.settings.enter_sends,
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
            ui.label("Enable 'Knowledge Graph' in the RAG Sources panel to visualize.");
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
                ui.heading("Input");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.settings.enter_sends, "Enter sends message");
                    ui.label(
                        egui::RichText::new(if self.settings.enter_sends {
                            "(Ctrl+Enter for new line)"
                        } else {
                            "(Enter for new line, Ctrl+Enter sends)"
                        })
                        .size(11.0)
                        .color(Color32::GRAY),
                    );
                });

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
            || self.wizard_state.pulling_model.is_some()
            || self.wizard_state.deleting_model.is_some()
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
                self.render_model_wizard(ctx);

                egui::CentralPanel::default().show(ctx, |ui| {
                    match self.active_panel {
                        SidebarPanel::Chat => { self.render_chat_area(ui); }
                        SidebarPanel::Sessions => { self.render_sessions_panel(ui); }
                        SidebarPanel::AgentPool => { self.render_agent_pool_panel(ui); }
                        SidebarPanel::ToolsMcp => { self.render_tools_panel(ui); }
                        SidebarPanel::Automation => { self.render_automation_panel(ui); }
                        SidebarPanel::RagSources => { self.render_knowledge_panel(ui); }
                        SidebarPanel::Memory => { self.render_memory_panel(ui); }
                        SidebarPanel::CloudStorage => { self.render_cloud_panel(ui); }
                        SidebarPanel::Media => { self.render_media_panel(ui); }
                        SidebarPanel::Audio => { self.render_audio_panel(ui); }
                        SidebarPanel::PromptLab => { self.render_prompt_lab_panel(ui); }
                        SidebarPanel::Evaluation => { self.render_eval_panel(ui); }
                        SidebarPanel::Security => { self.render_security_panel(ui); }
                        SidebarPanel::Analytics => { self.render_analytics_panel(ui); }
                        SidebarPanel::DevTools => { self.render_devtools_panel(ui); }
                        SidebarPanel::Cluster => { self.render_cluster_panel(ui); }
                        SidebarPanel::Butler => { self.render_butler_panel(ui); }
                        SidebarPanel::Settings => { self.render_settings_panel(ui); }
                    }
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
