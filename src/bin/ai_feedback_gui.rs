//! `ai_feedback_gui` — Feedback-Loop auditor GUI (egui/eframe).
//!
//! Desktop companion to `ai_feedback`. Loads a `DispatchLedger` JSONL file,
//! replays events into per-sink + drop-reason aggregates, and shows ledger
//! chain integrity at a glance.
//!
//! Read-only by design (see `feedback_auditable_subsystems`).
//!
//! Build: `cargo build --release --bin ai_feedback_gui \
//!   --features "feedback-loop gui-pro"`.

use ai_assistant::{DispatchEvent, DispatchEventKind};
use eframe::egui;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_feedback — Feedback Loop Auditor")
            .with_inner_size([1180.0, 780.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_feedback_gui",
        options,
        Box::new(|_cc| Box::new(FeedbackApp::default())),
    )
}

// =============================================================================
// App state
// =============================================================================

struct FeedbackApp {
    ledger_path: String,
    retract_path: String,
    events: Vec<DispatchEvent>,
    retract_events: Vec<DispatchEvent>,
    totals: Totals,
    ledger_report: Option<LedgerReport>,
    retract_report: Option<LedgerReport>,
    sinks: BTreeMap<String, SinkStats>,
    drops: BTreeMap<String, u64>,
    status: String,
    last_load: Instant,
    auto_refresh: bool,
    show_tab: Tab,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Sinks,
    Drops,
    Retractions,
    Events,
}

#[derive(Default)]
struct Totals {
    events: usize,
    received: u64,
    dispatched: u64,
    failed: u64,
    dropped: u64,
    freeze_changes: u64,
}

#[derive(Default, Clone)]
struct SinkStats {
    dispatched: u64,
    failed: u64,
}

struct LedgerReport {
    total: usize,
    chain_ok: bool,
    reason: String,
}

impl Default for FeedbackApp {
    fn default() -> Self {
        Self {
            ledger_path: "./feedback/dispatch.jsonl".to_string(),
            retract_path: "./feedback/retractions.jsonl".to_string(),
            events: Vec::new(),
            retract_events: Vec::new(),
            totals: Totals::default(),
            ledger_report: None,
            retract_report: None,
            sinks: BTreeMap::new(),
            drops: BTreeMap::new(),
            status: "Ready. Point to a dispatch ledger and click Reload.".to_string(),
            last_load: Instant::now() - Duration::from_secs(60),
            auto_refresh: false,
            show_tab: Tab::Overview,
        }
    }
}

// =============================================================================
// Loading + replay
// =============================================================================

impl FeedbackApp {
    fn reload(&mut self) {
        self.events.clear();
        self.retract_events.clear();
        self.totals = Totals::default();
        self.ledger_report = None;
        self.retract_report = None;
        self.sinks.clear();
        self.drops.clear();

        match load_events(&self.ledger_path) {
            Ok(events) => {
                self.events = events;
                self.ledger_report = Some(verify_chain(&self.events));
                self.replay();
            }
            Err(e) => {
                self.status = format!("dispatch ledger: {e}");
                return;
            }
        }

        if !self.retract_path.trim().is_empty() && PathBuf::from(&self.retract_path).exists() {
            match load_events(&self.retract_path) {
                Ok(events) => {
                    self.retract_events = events;
                    self.retract_report = Some(verify_chain(&self.retract_events));
                }
                Err(e) => {
                    self.status = format!("retractions: {e}");
                    return;
                }
            }
        }

        self.status = format!(
            "Loaded {} dispatch events, {} retraction events",
            self.events.len(),
            self.retract_events.len()
        );
        self.last_load = Instant::now();
    }

    fn replay(&mut self) {
        self.totals.events = self.events.len();
        for ev in &self.events {
            match &ev.kind {
                DispatchEventKind::TrajectoryReceived { .. } => self.totals.received += 1,
                DispatchEventKind::SinkDispatched { sink, .. } => {
                    self.totals.dispatched += 1;
                    self.sinks.entry(sink.clone()).or_default().dispatched += 1;
                }
                DispatchEventKind::SinkFailed { sink, .. } => {
                    self.totals.failed += 1;
                    self.sinks.entry(sink.clone()).or_default().failed += 1;
                }
                DispatchEventKind::TrajectoryDropped { reason, .. } => {
                    self.totals.dropped += 1;
                    *self.drops.entry(reason.clone()).or_insert(0) += 1;
                }
                DispatchEventKind::FreezeChanged { .. } => self.totals.freeze_changes += 1,
                _ => {}
            }
        }
    }
}

fn load_events(path: &str) -> Result<Vec<DispatchEvent>, String> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(format!("not found: {}", path.display()));
    }
    let text = fs::read_to_string(&path).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: DispatchEvent =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        out.push(ev);
    }
    Ok(out)
}

fn verify_chain(events: &[DispatchEvent]) -> LedgerReport {
    let mut rep = LedgerReport {
        total: events.len(),
        chain_ok: true,
        reason: String::new(),
    };
    for (i, ev) in events.iter().enumerate() {
        if ev.seq != i as u64 {
            rep.chain_ok = false;
            rep.reason = format!("seq gap at {i}: got {}", ev.seq);
            break;
        }
        if !ev.verify_self_hash() {
            rep.chain_ok = false;
            rep.reason = format!("self-hash mismatch at seq {}", ev.seq);
            break;
        }
        if i > 0 && ev.prev_hash_hex != events[i - 1].self_hash_hex {
            rep.chain_ok = false;
            rep.reason = format!("chain break before seq {}", ev.seq);
            break;
        }
    }
    rep
}

// =============================================================================
// UI
// =============================================================================

impl eframe::App for FeedbackApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_refresh && self.last_load.elapsed() > Duration::from_secs(5) {
            self.reload();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Dispatch:");
                ui.add(egui::TextEdit::singleline(&mut self.ledger_path).desired_width(360.0));
                ui.label("Retractions:");
                ui.add(egui::TextEdit::singleline(&mut self.retract_path).desired_width(260.0));
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto (5s)");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.show_tab, Tab::Overview, "Overview");
                ui.selectable_value(&mut self.show_tab, Tab::Sinks, "Sinks");
                ui.selectable_value(&mut self.show_tab, Tab::Drops, "Drops");
                ui.selectable_value(&mut self.show_tab, Tab::Retractions, "Retractions");
                ui.selectable_value(&mut self.show_tab, Tab::Events, "Events");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.show_tab {
            Tab::Overview => render_overview(ui, &self.totals, self.ledger_report.as_ref()),
            Tab::Sinks => render_sinks(ui, &self.sinks),
            Tab::Drops => render_drops(ui, &self.drops),
            Tab::Retractions => {
                render_retractions(ui, &self.retract_events, self.retract_report.as_ref())
            }
            Tab::Events => render_events(ui, &self.events),
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
    ui.heading("Feedback Loop Auditor");
    ui.separator();
    egui::Grid::new("overview").num_columns(2).show(ui, |ui| {
        kv(ui, "Total events", &totals.events.to_string());
        kv(ui, "Received", &totals.received.to_string());
        kv(ui, "Dispatched", &totals.dispatched.to_string());
        kv(ui, "Failed", &totals.failed.to_string());
        kv(ui, "Dropped", &totals.dropped.to_string());
        kv(ui, "Freeze changes", &totals.freeze_changes.to_string());
    });
    ui.separator();
    if let Some(rep) = report {
        ui.heading("Dispatch ledger");
        if rep.chain_ok {
            ui.colored_label(egui::Color32::GREEN, format!("OK — {} events", rep.total));
        } else {
            ui.colored_label(egui::Color32::RED, format!("BROKEN: {}", rep.reason));
        }
    }
}

fn render_sinks(ui: &mut egui::Ui, sinks: &BTreeMap<String, SinkStats>) {
    ui.heading("Per-sink stats");
    ui.separator();
    if sinks.is_empty() {
        ui.label("No sink events yet.");
        return;
    }
    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::Grid::new("sinks")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("sink");
                ui.strong("dispatched");
                ui.strong("failed");
                ui.end_row();
                for (name, s) in sinks {
                    ui.label(name);
                    ui.label(s.dispatched.to_string());
                    if s.failed > 0 {
                        ui.colored_label(egui::Color32::YELLOW, s.failed.to_string());
                    } else {
                        ui.label(s.failed.to_string());
                    }
                    ui.end_row();
                }
            });
    });
}

fn render_drops(ui: &mut egui::Ui, drops: &BTreeMap<String, u64>) {
    ui.heading("Drops by reason");
    ui.separator();
    if drops.is_empty() {
        ui.label("No drops recorded.");
        return;
    }
    egui::Grid::new("drops")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.strong("reason");
            ui.strong("count");
            ui.end_row();
            for (reason, n) in drops {
                ui.label(reason);
                ui.label(n.to_string());
                ui.end_row();
            }
        });
}

fn render_retractions(ui: &mut egui::Ui, events: &[DispatchEvent], report: Option<&LedgerReport>) {
    ui.heading("Retractions");
    if let Some(rep) = report {
        if rep.chain_ok {
            ui.colored_label(
                egui::Color32::GREEN,
                format!("Chain OK — {} events", rep.total),
            );
        } else {
            ui.colored_label(egui::Color32::RED, format!("Chain BROKEN: {}", rep.reason));
        }
    }
    ui.separator();
    if events.is_empty() {
        ui.label("No retractions recorded.");
        return;
    }
    egui::ScrollArea::vertical().show(ui, |ui| {
        for ev in events {
            match &ev.kind {
                DispatchEventKind::RetractionRequested { trajectory, reason } => {
                    ui.label(format!(
                        "[{:>4}] REQUEST {} {} reason={}",
                        ev.seq,
                        ev.timestamp.to_rfc3339(),
                        trajectory.as_str(),
                        reason
                    ));
                }
                DispatchEventKind::RetractionPropagated { trajectory, sink } => {
                    ui.label(format!(
                        "[{:>4}] PROPAGATE {} {} sink={}",
                        ev.seq,
                        ev.timestamp.to_rfc3339(),
                        trajectory.as_str(),
                        sink
                    ));
                }
                _ => {}
            }
        }
    });
}

fn render_events(ui: &mut egui::Ui, events: &[DispatchEvent]) {
    ui.heading("Raw dispatch events");
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for ev in events.iter().rev().take(500) {
            ui.label(format!(
                "[{:>4}] {} signer={} {:?}",
                ev.seq,
                ev.timestamp.to_rfc3339(),
                ev.signer,
                ev.kind
            ));
        }
    });
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.strong(k);
    ui.label(v);
    ui.end_row();
}
