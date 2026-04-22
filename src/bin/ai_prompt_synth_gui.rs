//! `ai_prompt_synth_gui` — Fragment Synthesis auditor GUI (egui/eframe).
//!
//! Desktop companion to `ai_prompt_synth`. Loads a `FragmentLedger` JSONL
//! file, replays the event stream into a per-cluster + per-arm summary, and
//! renders it alongside ledger chain integrity.
//!
//! Read-only by design (see `feedback_auditable_subsystems`).
//!
//! Build: `cargo build --release --bin ai_prompt_synth_gui \
//!   --features "prompt-synthesis gui-pro"`.

use ai_assistant::{FragmentEvent, FragmentEventKind};
use eframe::egui;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_prompt_synth — Fragment Synthesis Auditor")
            .with_inner_size([1180.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_prompt_synth_gui",
        options,
        Box::new(|_cc| Box::new(SynthApp::default())),
    )
}

// =============================================================================
// App state
// =============================================================================

struct SynthApp {
    ledger_path: String,
    events: Vec<FragmentEvent>,
    clusters: BTreeMap<String, ClusterView>,
    ledger_report: Option<LedgerReport>,
    selected_cluster: Option<String>,
    status: String,
    last_load: Instant,
    auto_refresh: bool,
    totals: Totals,
}

#[derive(Default)]
struct Totals {
    events: usize,
    creations: u64,
    selections: u64,
    rewards: u64,
    retirements: u64,
    resizes: u64,
    freeze_changes: u64,
}

struct ClusterView {
    cluster: String,
    arms: BTreeMap<String, ArmView>,
    selections: u64,
    rewards: u64,
}

struct ArmView {
    arm: String,
    provider: String,
    origin: String,
    selections: u64,
    samples: u64,
    reward_sum: f64,
    retired: bool,
    reward_history: Vec<f32>,
}

impl ArmView {
    fn mean_reward(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.reward_sum / self.samples as f64
        }
    }
}

struct LedgerReport {
    total: usize,
    chain_ok: bool,
    first_bad_seq: Option<u64>,
    reason: String,
}

impl Default for SynthApp {
    fn default() -> Self {
        Self {
            ledger_path: "./prompt_synth/ledger.jsonl".to_string(),
            events: Vec::new(),
            clusters: BTreeMap::new(),
            ledger_report: None,
            selected_cluster: None,
            status: "Ready. Point to a ledger JSONL and click Reload.".to_string(),
            last_load: Instant::now() - Duration::from_secs(60),
            auto_refresh: false,
            totals: Totals::default(),
        }
    }
}

// =============================================================================
// Loading + replay
// =============================================================================

impl SynthApp {
    fn reload(&mut self) {
        self.events.clear();
        self.clusters.clear();
        self.ledger_report = None;
        self.totals = Totals::default();

        let path = PathBuf::from(&self.ledger_path);
        if !path.exists() {
            self.status = format!("Ledger not found: {}", path.display());
            return;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                self.status = format!("read: {e}");
                return;
            }
        };
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<FragmentEvent>(line) {
                Ok(ev) => self.events.push(ev),
                Err(e) => {
                    self.status = format!("line {}: {e}", n + 1);
                    return;
                }
            }
        }
        self.verify_chain();
        self.replay();
        self.status = format!(
            "Loaded {} events from {} — {} cluster(s)",
            self.events.len(),
            path.display(),
            self.clusters.len()
        );
        self.last_load = Instant::now();
    }

    fn verify_chain(&mut self) {
        let mut report = LedgerReport {
            total: self.events.len(),
            chain_ok: true,
            first_bad_seq: None,
            reason: String::new(),
        };
        for (i, ev) in self.events.iter().enumerate() {
            if ev.seq != i as u64 {
                report.chain_ok = false;
                report.first_bad_seq = Some(ev.seq);
                report.reason = format!("seq gap at index {i}: got {}", ev.seq);
                break;
            }
            if !ev.verify_self_hash() {
                report.chain_ok = false;
                report.first_bad_seq = Some(ev.seq);
                report.reason = format!("self-hash mismatch at seq {}", ev.seq);
                break;
            }
            if i > 0 && ev.prev_hash_hex != self.events[i - 1].self_hash_hex {
                report.chain_ok = false;
                report.first_bad_seq = Some(ev.seq);
                report.reason = format!("chain break before seq {}", ev.seq);
                break;
            }
        }
        self.ledger_report = Some(report);
    }

    fn replay(&mut self) {
        self.totals.events = self.events.len();
        let events = std::mem::take(&mut self.events);
        for ev in &events {
            match &ev.kind {
                FragmentEventKind::ArmCreated {
                    cluster,
                    arm,
                    provider,
                    origin,
                } => {
                    self.totals.creations += 1;
                    let c = ensure_cluster(&mut self.clusters, cluster.to_string());
                    c.arms
                        .entry(arm.as_str().to_string())
                        .or_insert_with(|| ArmView {
                            arm: arm.as_str().to_string(),
                            provider: provider.as_str().to_string(),
                            origin: origin.to_string(),
                            selections: 0,
                            samples: 0,
                            reward_sum: 0.0,
                            retired: false,
                            reward_history: Vec::new(),
                        });
                }
                FragmentEventKind::ArmSelected { cluster, arm, .. } => {
                    self.totals.selections += 1;
                    let c = ensure_cluster(&mut self.clusters, cluster.to_string());
                    c.selections += 1;
                    if let Some(a) = c.arms.get_mut(arm.as_str()) {
                        a.selections += 1;
                    }
                }
                FragmentEventKind::RewardRecorded {
                    cluster,
                    arm,
                    reward,
                    ..
                } => {
                    self.totals.rewards += 1;
                    let c = ensure_cluster(&mut self.clusters, cluster.to_string());
                    c.rewards += 1;
                    if let Some(a) = c.arms.get_mut(arm.as_str()) {
                        a.samples += 1;
                        a.reward_sum += *reward as f64;
                        a.reward_history.push(*reward);
                    }
                }
                FragmentEventKind::ArmRetired { cluster, arm, .. } => {
                    self.totals.retirements += 1;
                    let c = ensure_cluster(&mut self.clusters, cluster.to_string());
                    if let Some(a) = c.arms.get_mut(arm.as_str()) {
                        a.retired = true;
                    }
                }
                FragmentEventKind::ClusterResized { .. } => {
                    self.totals.resizes += 1;
                }
                FragmentEventKind::FreezeChanged { .. } => {
                    self.totals.freeze_changes += 1;
                }
                _ => {}
            }
        }
        self.events = events;
    }
}

fn ensure_cluster(clusters: &mut BTreeMap<String, ClusterView>, key: String) -> &mut ClusterView {
    clusters.entry(key.clone()).or_insert_with(|| ClusterView {
        cluster: key,
        arms: BTreeMap::new(),
        selections: 0,
        rewards: 0,
    })
}

// =============================================================================
// UI
// =============================================================================

impl eframe::App for SynthApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_refresh && self.last_load.elapsed() > Duration::from_secs(5) {
            self.reload();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Ledger:");
                ui.add(egui::TextEdit::singleline(&mut self.ledger_path).desired_width(420.0));
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto (5s)");
                ui.separator();
                ui.label(format!(
                    "Events: {} | Arms created: {} | Selections: {} | Rewards: {} | Retired: {}",
                    self.totals.events,
                    self.totals.creations,
                    self.totals.selections,
                    self.totals.rewards,
                    self.totals.retirements,
                ));
            });
        });

        egui::SidePanel::left("clusters")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Clusters");
                if let Some(rep) = &self.ledger_report {
                    let (color, txt) = if rep.chain_ok {
                        (
                            egui::Color32::GREEN,
                            format!("Chain OK ({} events)", rep.total),
                        )
                    } else {
                        (egui::Color32::RED, format!("Chain BROKEN: {}", rep.reason))
                    };
                    ui.colored_label(color, txt);
                }
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let keys: Vec<String> = self.clusters.keys().cloned().collect();
                    for key in keys {
                        let c = match self.clusters.get(&key) {
                            Some(c) => c,
                            None => continue,
                        };
                        let active = c.arms.iter().filter(|(_, a)| !a.retired).count();
                        let label = format!(
                            "cluster {} — {} arm(s), {} sel",
                            c.cluster, active, c.selections
                        );
                        let selected = self.selected_cluster.as_deref() == Some(key.as_str());
                        if ui.selectable_label(selected, label).clicked() {
                            self.selected_cluster = Some(key);
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| match self.selected_cluster.clone() {
            Some(key) => match self.clusters.get(&key) {
                Some(c) => render_cluster(ui, c),
                None => {
                    ui.label("Cluster not found.");
                }
            },
            None => render_overview(ui, &self.totals, self.ledger_report.as_ref()),
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(&self.status);
        });

        if self.auto_refresh {
            ctx.request_repaint_after(Duration::from_secs(1));
        }
    }
}

fn render_overview(ui: &mut egui::Ui, totals: &Totals, report: Option<&LedgerReport>) {
    ui.heading("Fragment Synthesis Auditor");
    ui.label("Select a cluster from the left panel to inspect its arms.");
    ui.separator();
    egui::Grid::new("overview_grid")
        .num_columns(2)
        .show(ui, |ui| {
            kv(ui, "Total events", &totals.events.to_string());
            kv(ui, "Arms created", &totals.creations.to_string());
            kv(ui, "Selections", &totals.selections.to_string());
            kv(ui, "Rewards recorded", &totals.rewards.to_string());
            kv(ui, "Arms retired", &totals.retirements.to_string());
            kv(ui, "Cluster resizes", &totals.resizes.to_string());
            kv(ui, "Freeze changes", &totals.freeze_changes.to_string());
        });
    ui.separator();
    if let Some(rep) = report {
        ui.heading("Ledger chain");
        if rep.chain_ok {
            ui.colored_label(egui::Color32::GREEN, format!("OK — {} events", rep.total));
        } else {
            ui.colored_label(egui::Color32::RED, format!("BROKEN: {}", rep.reason));
        }
    }
}

fn render_cluster(ui: &mut egui::Ui, c: &ClusterView) {
    ui.heading(format!("Cluster {}", c.cluster));
    ui.label(format!(
        "Arms: {} | Selections: {} | Rewards: {}",
        c.arms.len(),
        c.selections,
        c.rewards
    ));
    ui.separator();

    egui::ScrollArea::both().show(ui, |ui| {
        egui::Grid::new(format!("arms_{}", c.cluster))
            .num_columns(7)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("arm");
                ui.strong("provider");
                ui.strong("origin");
                ui.strong("selected");
                ui.strong("samples");
                ui.strong("mean");
                ui.strong("status");
                ui.end_row();
                for a in c.arms.values() {
                    ui.label(&a.arm);
                    ui.label(&a.provider);
                    ui.label(&a.origin);
                    ui.label(a.selections.to_string());
                    ui.label(a.samples.to_string());
                    ui.label(format!("{:.3}", a.mean_reward()));
                    if a.retired {
                        ui.colored_label(egui::Color32::YELLOW, "retired");
                    } else {
                        ui.colored_label(egui::Color32::GREEN, "active");
                    }
                    ui.end_row();
                }
            });

        ui.separator();
        ui.heading("Reward history (most recent 64 per arm)");
        for a in c.arms.values() {
            if a.reward_history.is_empty() {
                continue;
            }
            let tail = if a.reward_history.len() > 64 {
                &a.reward_history[a.reward_history.len() - 64..]
            } else {
                &a.reward_history[..]
            };
            ui.collapsing(format!("{} ({} samples)", a.arm, a.samples), |ui| {
                draw_sparkline(ui, tail);
            });
        }
    });
}

fn draw_sparkline(ui: &mut egui::Ui, values: &[f32]) {
    let width = ui.available_width().min(640.0);
    let height: f32 = 48.0;
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 20, 28));
    if values.len() < 2 {
        return;
    }
    let n = values.len();
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 220, 160));
    for i in 1..n {
        let x0 = rect.left() + (i as f32 - 1.0) / (n as f32 - 1.0) * rect.width();
        let x1 = rect.left() + i as f32 / (n as f32 - 1.0) * rect.width();
        let y0 = rect.bottom() - values[i - 1].clamp(0.0, 1.0) * rect.height();
        let y1 = rect.bottom() - values[i].clamp(0.0, 1.0) * rect.height();
        painter.line_segment([egui::pos2(x0, y0), egui::pos2(x1, y1)], stroke);
    }
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.strong(k);
    ui.label(v);
    ui.end_row();
}
