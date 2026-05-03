//! `ai_acp_audit_gui` — visual auditor for ACP SLO log files.
//!
//! Build: cargo build --release --bin ai_acp_audit_gui --features gui-acp
//! Run:   cargo run --bin ai_acp_audit_gui --features gui-acp
//!
//! Read-only audit of JSONL log files written by `ai_acp serve`.
//! Shows: list of log files, per-record table, SLO summary panel,
//! per-session aggregates. Per memory rule `feedback_auditable_subsystems`.

#![cfg(feature = "gui-acp")]

use ai_assistant::acp::SloRecord;
use eframe::egui;
use std::path::{Path, PathBuf};

const SLO_HANDSHAKE_MS: u64 = 200;
const SLO_FIRST_CHUNK_MS: u64 = 1000;
const SLO_MIN_CHUNKS_PER_SEC: f64 = 30.0;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_acp_audit — ACP SLO Auditor")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_acp_audit_gui",
        options,
        Box::new(|_cc| Box::new(AuditApp::default())),
    )
}

struct AuditApp {
    dir_input: String,
    files: Vec<PathBuf>,
    selected: Option<PathBuf>,
    records: Vec<SloRecord>,
    status: String,
}

impl Default for AuditApp {
    fn default() -> Self {
        let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        p.push(".ai_assistant");
        p.push("acp_logs");
        Self {
            dir_input: p.display().to_string(),
            files: Vec::new(),
            selected: None,
            records: Vec::new(),
            status: "Press 'Reload' to discover SLO logs.".into(),
        }
    }
}

impl eframe::App for AuditApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Log dir:");
                ui.text_edit_singleline(&mut self.dir_input);
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.label(&self.status);
            });
        });
        egui::SidePanel::left("files")
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("Log files");
                for f in &self.files.clone() {
                    let label = f
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let selected = self.selected.as_ref() == Some(f);
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected = Some(f.clone());
                        self.records = read_log(f).unwrap_or_default();
                    }
                }
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.records.is_empty() {
                ui.label("Select a log file from the left panel.");
                return;
            }
            self.summary_panel(ui);
            ui.separator();
            self.records_table(ui);
        });
    }
}

impl AuditApp {
    fn reload(&mut self) {
        match discover(&self.dir_input) {
            Ok(files) => {
                let n = files.len();
                self.files = files;
                self.status = format!("Discovered {} log file(s)", n);
            }
            Err(e) => {
                self.files.clear();
                self.status = format!("Discovery error: {}", e);
            }
        }
    }

    fn summary_panel(&self, ui: &mut egui::Ui) {
        let mut handshakes = 0usize;
        let mut h_breach = 0usize;
        let mut prompts = 0usize;
        let mut p_breach = 0usize;
        let mut first_chunks = 0usize;
        let mut fc_breach = 0usize;
        for r in &self.records {
            match r.kind.as_str() {
                "handshake" => {
                    handshakes += 1;
                    if r.elapsed_ms > SLO_HANDSHAKE_MS {
                        h_breach += 1;
                    }
                }
                "prompt" => {
                    prompts += 1;
                    if r.chunks > 0 && r.chunks_per_sec < SLO_MIN_CHUNKS_PER_SEC {
                        p_breach += 1;
                    }
                }
                "first_chunk" => {
                    first_chunks += 1;
                    if r.elapsed_ms > SLO_FIRST_CHUNK_MS {
                        fc_breach += 1;
                    }
                }
                _ => {}
            }
        }
        ui.heading("SLO summary");
        egui::Grid::new("summary").striped(true).show(ui, |ui| {
            ui.label("Handshakes");
            ui.label(format!(
                "{} (>{}ms breach: {})",
                handshakes, SLO_HANDSHAKE_MS, h_breach
            ));
            ui.end_row();
            ui.label("Prompts");
            ui.label(format!(
                "{} (<{:.0} chunks/s breach: {})",
                prompts, SLO_MIN_CHUNKS_PER_SEC, p_breach
            ));
            ui.end_row();
            ui.label("First-chunk records");
            ui.label(format!(
                "{} (>{}ms breach: {})",
                first_chunks, SLO_FIRST_CHUNK_MS, fc_breach
            ));
            ui.end_row();
            let any_breach = h_breach > 0 || p_breach > 0 || fc_breach > 0;
            ui.label("Status");
            if any_breach {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), "BREACH");
            } else {
                ui.colored_label(egui::Color32::from_rgb(80, 200, 120), "OK");
            }
            ui.end_row();
        });
    }

    fn records_table(&self, ui: &mut egui::Ui) {
        ui.heading("Records");
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("records").striped(true).show(ui, |ui| {
                ui.strong("kind");
                ui.strong("session");
                ui.strong("elapsed_ms");
                ui.strong("chunks");
                ui.strong("chunks/s");
                ui.end_row();
                for r in &self.records {
                    let breach = match r.kind.as_str() {
                        "handshake" => r.elapsed_ms > SLO_HANDSHAKE_MS,
                        "first_chunk" => r.elapsed_ms > SLO_FIRST_CHUNK_MS,
                        "prompt" => r.chunks > 0 && r.chunks_per_sec < SLO_MIN_CHUNKS_PER_SEC,
                        _ => false,
                    };
                    let color = if breach {
                        egui::Color32::from_rgb(220, 80, 80)
                    } else {
                        ui.style().visuals.text_color()
                    };
                    ui.colored_label(color, &r.kind);
                    ui.label(r.session_id.as_deref().unwrap_or("-"));
                    ui.label(r.elapsed_ms.to_string());
                    ui.label(r.chunks.to_string());
                    ui.label(format!("{:.1}", r.chunks_per_sec));
                    ui.end_row();
                }
            });
        });
    }
}

fn discover(dir: &str) -> std::io::Result<Vec<PathBuf>> {
    let p = Path::new(dir);
    if !p.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(p)? {
        let e = entry?;
        let pth = e.path();
        if pth.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            files.push(pth);
        }
    }
    files.sort();
    Ok(files)
}

fn read_log(path: &Path) -> std::io::Result<Vec<SloRecord>> {
    use std::io::BufRead;
    let f = std::fs::File::open(path)?;
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<SloRecord>(&line) {
            out.push(rec);
        }
    }
    Ok(out)
}
