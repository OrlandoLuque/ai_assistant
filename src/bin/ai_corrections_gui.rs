//! `ai_corrections_gui` — visual auditor for the self-correction review queue.
//!
//! Build: cargo build --release --bin ai_corrections_gui --features gui-corrections
//! Run:   cargo run --bin ai_corrections_gui --features gui-corrections
//!
//! The companion to `ai_corrections`. Shows the work the engine could **not**
//! make correct: why each attempt failed, and the artifact exactly as produced.
//! Per memory rule `feedback_auditable_subsystems`, and promised by the
//! `self_correction` module docs since V98.
//!
//! Read-mostly: the only mutation is resolving an item, which moves it to
//! `resolved/` rather than deleting it — the record of what an agent could not
//! do is the material worth keeping.

#![cfg(feature = "gui-corrections")]

use std::path::PathBuf;

use ai_assistant::self_correction::{Quarantine, QuarantineRecord};
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_corrections — self-correction review queue")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_corrections_gui",
        options,
        Box::new(|_cc| Box::new(ReviewApp::default())),
    )
}

/// Which side of the queue is on screen.
#[derive(PartialEq, Eq, Clone, Copy)]
enum View {
    Pending,
    Resolved,
}

struct ReviewApp {
    dir_input: String,
    view: View,
    items: Vec<QuarantineRecord>,
    selected: Option<usize>,
    /// Artifact text for the selected item, loaded on selection rather than
    /// with the list: a queue of hundreds should not read every file to draw.
    artifact: String,
    status: String,
}

impl Default for ReviewApp {
    fn default() -> Self {
        let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        p.push(".ai_assistant");
        p.push("corrections");
        Self {
            dir_input: p.display().to_string(),
            view: View::Pending,
            items: Vec::new(),
            selected: None,
            artifact: String::new(),
            status: "Press 'Reload' to read the queue.".into(),
        }
    }
}

impl ReviewApp {
    fn reload(&mut self) {
        self.selected = None;
        self.artifact.clear();
        let q = match Quarantine::open(&self.dir_input) {
            Ok(q) => q,
            Err(e) => {
                self.status = format!("cannot open {}: {e}", self.dir_input);
                self.items.clear();
                return;
            }
        };
        let read = match self.view {
            View::Pending => q.pending(),
            View::Resolved => q.resolved(),
        };
        match read {
            Ok(items) => {
                self.status = match (self.view, items.len()) {
                    (View::Pending, 0) => "Nothing awaiting review.".to_string(),
                    (View::Pending, n) => format!("{n} item(s) awaiting review."),
                    (View::Resolved, n) => format!("{n} resolved item(s)."),
                };
                self.items = items;
            }
            Err(e) => {
                self.status = format!("{e}");
                self.items.clear();
            }
        }
    }

    fn select(&mut self, idx: usize) {
        self.selected = Some(idx);
        self.artifact = match self.items.get(idx) {
            Some(rec) => rec
                .read_artifact()
                .unwrap_or_else(|e| format!("<could not read artifact: {e}>")),
            None => String::new(),
        };
    }

    fn resolve_selected(&mut self) {
        let Some(idx) = self.selected else { return };
        let Some(rec) = self.items.get(idx) else {
            return;
        };
        let id = rec.id.clone();
        match Quarantine::open(&self.dir_input).and_then(|q| q.resolve(&id)) {
            Ok(()) => {
                self.status = format!("Resolved '{id}' — moved, not deleted.");
                self.reload();
            }
            Err(e) => self.status = format!("could not resolve '{id}': {e}"),
        }
    }
}

impl eframe::App for ReviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Directory:");
                ui.add(egui::TextEdit::singleline(&mut self.dir_input).desired_width(420.0));
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.separator();
                let mut view = self.view;
                if ui
                    .selectable_value(&mut view, View::Pending, "Awaiting review")
                    .clicked()
                    || ui
                        .selectable_value(&mut view, View::Resolved, "Resolved")
                        .clicked()
                {
                    self.view = view;
                    self.reload();
                }
            });
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        egui::SidePanel::left("queue")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("Queue");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut clicked = None;
                    for (i, rec) in self.items.iter().enumerate() {
                        let label = format!(
                            "{}\n  {} · {} attempt(s) · {}",
                            rec.id,
                            rec.evidence.task_name,
                            rec.evidence.attempts.len(),
                            rec.evidence.stop_reason
                        );
                        if ui
                            .selectable_label(self.selected == Some(i), label)
                            .clicked()
                        {
                            clicked = Some(i);
                        }
                    }
                    if let Some(i) = clicked {
                        self.select(i);
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(idx) = self.selected else {
                ui.centered_and_justified(|ui| {
                    ui.label("Select an item to see why it was not accepted.");
                });
                return;
            };
            // Cloned rather than borrowed: the panel both reads the record and
            // may resolve it, and `resolve_selected` needs `&mut self`.
            let Some(rec) = self.items.get(idx).cloned() else {
                return;
            };

            let mut resolve_clicked = false;
            ui.horizontal(|ui| {
                ui.heading(&rec.evidence.task_name);
                if self.view == View::Pending && ui.button("Mark resolved").clicked() {
                    resolve_clicked = true;
                }
            });
            if resolve_clicked {
                self.resolve_selected();
                return;
            }
            ui.label(format!(
                "stopped: {} · {} tokens · {:.4} USD · {} ms",
                rec.evidence.stop_reason,
                rec.evidence.total_tokens,
                rec.evidence.total_cost_usd,
                rec.evidence.total_elapsed_ms
            ));
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.strong("Why it was not accepted");
                for a in &rec.evidence.attempts {
                    ui.label(format!(
                        "attempt {} — quality {:.2}",
                        a.attempt_num, a.quality_score
                    ));
                    for issue in &a.issues {
                        ui.add(egui::Label::new(egui::RichText::new(issue).monospace()).wrap(true));
                    }
                    ui.add_space(6.0);
                }

                ui.separator();
                ui.strong("The artifact, as produced");
                ui.add(
                    egui::TextEdit::multiline(&mut self.artifact.as_str())
                        .code_editor()
                        .desired_width(f32::INFINITY),
                );
            });
        });
    }
}
