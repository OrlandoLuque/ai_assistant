//! `ai_logs_gui` — Distributed log viewer with egui GUI.
//!
//! Build: cargo build --release --bin ai_logs_gui --features gui-logs
//! Run:   cargo run --bin ai_logs_gui --features gui-logs

use ai_assistant::distributed_log::{
    parse_log_level, DistributedLogEntry, LogLevel, LogReader, LogTailer, TraceSummary,
};
use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_logs — Distributed Log Viewer")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ai_logs_gui",
        options,
        Box::new(|_cc| Box::new(LogViewerApp::default())),
    )
}

// ============================================================================
// Application state
// ============================================================================

struct LogViewerApp {
    /// Source path (file or directory).
    source_path: String,
    /// Available traces.
    traces: Vec<TraceSummary>,
    /// Currently selected trace ID.
    selected_trace: Option<String>,
    /// Log entries for the selected trace.
    entries: Vec<DistributedLogEntry>,
    /// All entries (unfiltered).
    all_entries: Vec<DistributedLogEntry>,
    /// Filter settings.
    filter: LogFilter,
    /// Live mode (auto-refresh).
    live_mode: bool,
    /// Tailer for live mode.
    tailer: Option<LogTailer>,
    /// Auto-scroll to bottom.
    auto_scroll: bool,
    /// Status message.
    status: String,
    /// Last refresh time.
    last_refresh: std::time::Instant,
    /// Expanded entry index (for detail view).
    expanded_entry: Option<usize>,
}

struct LogFilter {
    min_level: LogLevel,
    node_filter: String,
    operation_filter: String,
    search_text: String,
}

impl Default for LogViewerApp {
    fn default() -> Self {
        Self {
            source_path: "./logs".to_string(),
            traces: Vec::new(),
            selected_trace: None,
            entries: Vec::new(),
            all_entries: Vec::new(),
            filter: LogFilter {
                min_level: LogLevel::Trace,
                node_filter: String::new(),
                operation_filter: String::new(),
                search_text: String::new(),
            },
            live_mode: false,
            tailer: None,
            auto_scroll: true,
            status: "Ready — enter a path and click Load".to_string(),
            last_refresh: std::time::Instant::now(),
            expanded_entry: None,
        }
    }
}

impl eframe::App for LogViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Live mode: poll every second
        if self.live_mode && self.last_refresh.elapsed().as_secs() >= 1 {
            self.poll_new_entries();
            self.last_refresh = std::time::Instant::now();
            ctx.request_repaint_after(std::time::Duration::from_secs(1));
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Source:");
                ui.text_edit_singleline(&mut self.source_path);
                if ui.button("Load").clicked() {
                    self.load_source();
                }
                ui.separator();
                let live_label = if self.live_mode {
                    "⏸ Pause"
                } else {
                    "▶ Live"
                };
                if ui.button(live_label).clicked() {
                    self.live_mode = !self.live_mode;
                    if self.live_mode {
                        self.start_tailing();
                        ctx.request_repaint_after(std::time::Duration::from_secs(1));
                    }
                }
                ui.separator();
                ui.label(&self.status);
            });
        });

        // Left panel: trace list
        egui::SidePanel::left("traces_panel")
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Traces");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let traces_snapshot: Vec<(String, usize, usize)> = self
                        .traces
                        .iter()
                        .map(|t| (t.trace_id.clone(), t.entry_count, t.nodes.len()))
                        .collect();

                    for (trace_id, entry_count, node_count) in &traces_snapshot {
                        let selected = self.selected_trace.as_deref() == Some(trace_id.as_str());
                        let label = format!(
                            "{}\n  {} entries | {} node(s)",
                            truncate_str(trace_id, 32),
                            entry_count,
                            node_count,
                        );

                        if ui.selectable_label(selected, &label).clicked() {
                            self.selected_trace = Some(trace_id.clone());
                            self.apply_trace_filter();
                        }
                    }
                });
            });

        // Bottom panel: filters
        egui::TopBottomPanel::bottom("filter_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Level:");
                egui::ComboBox::from_id_source("level_filter")
                    .selected_text(format!("{}", self.filter.min_level))
                    .show_ui(ui, |ui| {
                        for level in &[
                            LogLevel::Trace,
                            LogLevel::Debug,
                            LogLevel::Info,
                            LogLevel::Warn,
                            LogLevel::Error,
                        ] {
                            if ui
                                .selectable_value(
                                    &mut self.filter.min_level,
                                    *level,
                                    format!("{}", level),
                                )
                                .clicked()
                            {
                                self.apply_trace_filter();
                            }
                        }
                    });

                ui.separator();
                ui.label("Node:");
                if ui
                    .text_edit_singleline(&mut self.filter.node_filter)
                    .changed()
                {
                    self.apply_trace_filter();
                }

                ui.separator();
                ui.label("Search:");
                if ui
                    .text_edit_singleline(&mut self.filter.search_text)
                    .changed()
                {
                    self.apply_trace_filter();
                }

                ui.separator();
                ui.checkbox(&mut self.auto_scroll, "Auto-scroll");
            });
        });

        // Central panel: log entries
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.entries.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a trace from the left panel");
                });
                return;
            }

            ui.heading(format!(
                "Trace: {} ({} entries)",
                self.selected_trace.as_deref().unwrap_or(""),
                self.entries.len()
            ));
            ui.separator();

            let scroll = egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(self.auto_scroll);

            scroll.show(ui, |ui| {
                for (i, entry) in self.entries.iter().enumerate() {
                    let color = level_color(entry.level);
                    let text = format!(
                        "[{}] {} | {} | {} | {}",
                        format_ts(entry.timestamp_ms),
                        entry.level,
                        entry.node_id,
                        entry.operation,
                        entry.message,
                    );

                    let response = ui.colored_label(color, &text);

                    // Click to expand
                    if response.clicked() {
                        self.expanded_entry = if self.expanded_entry == Some(i) {
                            None
                        } else {
                            Some(i)
                        };
                    }

                    // Show details if expanded
                    if self.expanded_entry == Some(i) {
                        ui.indent("detail", |ui| {
                            ui.label(format!("  Trace:  {}", entry.trace_id));
                            ui.label(format!("  Span:   {}", entry.span_id));
                            if let Some(ref parent) = entry.parent_span_id {
                                ui.label(format!("  Parent: {}", parent));
                            }
                            if let Some(dur) = entry.duration_ms {
                                ui.label(format!("  Duration: {}ms", dur));
                            }
                            if !entry.attributes.is_empty() {
                                ui.label("  Attributes:");
                                for (k, v) in &entry.attributes {
                                    ui.label(format!("    {}: {}", k, v));
                                }
                            }
                        });
                    }
                }
            });
        });
    }
}

// ============================================================================
// App methods
// ============================================================================

impl LogViewerApp {
    fn load_source(&mut self) {
        let path = PathBuf::from(&self.source_path);
        match LogReader::list_traces(&path) {
            Ok(traces) => {
                self.status = format!("Loaded {} trace(s) from {}", traces.len(), self.source_path);
                self.traces = traces;
                self.selected_trace = None;
                self.entries.clear();
                self.all_entries.clear();

                // Also load all entries for filtering
                if let Ok(entries) = if path.is_dir() {
                    LogReader::read_dir(&path)
                } else {
                    LogReader::read_file(&path)
                } {
                    self.all_entries = entries;
                }
            }
            Err(e) => {
                self.status = format!("Error: {}", e);
                self.traces.clear();
            }
        }
    }

    fn apply_trace_filter(&mut self) {
        let Some(ref trace_id) = self.selected_trace else {
            self.entries.clear();
            return;
        };

        self.entries = self
            .all_entries
            .iter()
            .filter(|e| {
                e.trace_id == *trace_id
                    && e.level >= self.filter.min_level
                    && (self.filter.node_filter.is_empty()
                        || e.node_id.contains(&self.filter.node_filter))
                    && (self.filter.search_text.is_empty()
                        || e.message
                            .to_lowercase()
                            .contains(&self.filter.search_text.to_lowercase())
                        || e.operation
                            .to_lowercase()
                            .contains(&self.filter.search_text.to_lowercase()))
            })
            .cloned()
            .collect();
    }

    fn start_tailing(&mut self) {
        let path = PathBuf::from(&self.source_path);
        if path.is_file() {
            self.tailer = LogTailer::new(&path).ok();
        } else if path.is_dir() {
            // Tail first .jsonl file found
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    if entry.path().extension().and_then(|e| e.to_str()) == Some("jsonl") {
                        self.tailer = LogTailer::new(&entry.path()).ok();
                        break;
                    }
                }
            }
        }
    }

    fn poll_new_entries(&mut self) {
        if let Some(ref mut tailer) = self.tailer {
            if let Ok(new_entries) = tailer.next_entries() {
                if !new_entries.is_empty() {
                    self.status = format!("{} new entries", new_entries.len());
                    self.all_entries.extend(new_entries.clone());

                    // Update traces
                    let path = PathBuf::from(&self.source_path);
                    if let Ok(traces) = LogReader::list_traces(&path) {
                        self.traces = traces;
                    }

                    // Re-apply filter
                    self.apply_trace_filter();
                }
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn level_color(level: LogLevel) -> egui::Color32 {
    match level {
        LogLevel::Error => egui::Color32::from_rgb(239, 68, 68), // red
        LogLevel::Warn => egui::Color32::from_rgb(234, 179, 8),  // yellow
        LogLevel::Info => egui::Color32::from_rgb(226, 232, 240), // white-ish
        LogLevel::Debug => egui::Color32::from_rgb(148, 163, 184), // gray
        LogLevel::Trace => egui::Color32::from_rgb(100, 116, 139), // dark gray
        _ => egui::Color32::from_rgb(226, 232, 240),             // default
    }
}

fn format_ts(ms: u64) -> String {
    let secs = ms / 1000;
    let millis = ms % 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, s, millis)
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
