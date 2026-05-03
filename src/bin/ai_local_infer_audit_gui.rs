//! `ai_local_infer_audit_gui` — visual auditor for local-inference SLO logs.
//!
//! Build: cargo build --release --bin ai_local_infer_audit_gui --features gui-local-inference
//! Run:   cargo run --bin ai_local_infer_audit_gui --features gui-local-inference
//!
//! Read-only audit of JSONL log files written by `ai_local_infer`.
//! Shows: list of log files, per-record table, SLO summary panel.
//! Per memory rule `feedback_auditable_subsystems`.

#![cfg(feature = "gui-local-inference")]

use ai_assistant::local_inference::SloRecord;
use eframe::egui;
use std::path::{Path, PathBuf};

const SLO_LOAD_MS: u64 = 30_000;
const SLO_FIRST_CHUNK_MS: u64 = 1_000;
const SLO_MIN_TPS: f64 = 5.0;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_local_infer_audit — Local Inference SLO Auditor")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_local_infer_audit_gui",
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
        p.push("local_infer_logs");
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
        let mut load_breach = 0usize;
        let mut first_breach = 0usize;
        let mut tps_breach = 0usize;
        for r in &self.records {
            if r.load_ms > SLO_LOAD_MS {
                load_breach += 1;
            }
            if r.first_chunk_ms > SLO_FIRST_CHUNK_MS {
                first_breach += 1;
            }
            if r.generated_tokens > 0 && r.tokens_per_sec < SLO_MIN_TPS {
                tps_breach += 1;
            }
        }
        ui.heading("SLO summary");
        egui::Grid::new("summary").striped(true).show(ui, |ui| {
            ui.label("Records");
            ui.label(self.records.len().to_string());
            ui.end_row();
            ui.label("load_ms breaches");
            ui.label(format!("{} (>{}ms)", load_breach, SLO_LOAD_MS));
            ui.end_row();
            ui.label("first_chunk_ms breaches");
            ui.label(format!("{} (>{}ms)", first_breach, SLO_FIRST_CHUNK_MS));
            ui.end_row();
            ui.label("tokens/sec breaches");
            ui.label(format!("{} (<{:.1})", tps_breach, SLO_MIN_TPS));
            ui.end_row();
            let any_breach = load_breach + first_breach + tps_breach > 0;
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
                ui.strong("backend");
                ui.strong("load_ms");
                ui.strong("first_chunk_ms");
                ui.strong("total_ms");
                ui.strong("gen_tok");
                ui.strong("tok/s");
                ui.strong("gpu used/req");
                ui.end_row();
                for r in &self.records {
                    let load_breach = r.load_ms > SLO_LOAD_MS;
                    let first_breach = r.first_chunk_ms > SLO_FIRST_CHUNK_MS;
                    let tps_breach = r.generated_tokens > 0 && r.tokens_per_sec < SLO_MIN_TPS;
                    let normal = ui.style().visuals.text_color();
                    let red = egui::Color32::from_rgb(220, 80, 80);

                    ui.label(&r.backend);
                    ui.colored_label(
                        if load_breach { red } else { normal },
                        r.load_ms.to_string(),
                    );
                    ui.colored_label(
                        if first_breach { red } else { normal },
                        r.first_chunk_ms.to_string(),
                    );
                    ui.label(r.total_ms.to_string());
                    ui.label(r.generated_tokens.to_string());
                    ui.colored_label(
                        if tps_breach { red } else { normal },
                        format!("{:.1}", r.tokens_per_sec),
                    );
                    ui.label(format!(
                        "{}/{}",
                        r.n_gpu_layers_used, r.n_gpu_layers_requested
                    ));
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
