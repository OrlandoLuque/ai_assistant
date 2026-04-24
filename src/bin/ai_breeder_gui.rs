//! `ai_breeder_gui` — PromptBreeder V97 auditor GUI (egui/eframe).
//!
//! Desktop companion to `ai_breeder`. Loads a checkpoint file and an optional
//! ledger JSONL file, displays run summary, population table, lineage DAG
//! sizing, ledger chain integrity, and a tail of raw events.
//!
//! Read-only by design (see `feedback_auditable_subsystems`).
//!
//! Build: `cargo build --release --bin ai_breeder_gui \
//!   --features "prompt-breeder gui-pro"`.

use ai_assistant::prompt_breeder::checkpoint as ckpt;
use ai_assistant::prompt_breeder::{BreederEvent, Checkpoint, LedgerEntry, MutationOperator};
use eframe::egui;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_breeder — PromptBreeder V97 Auditor")
            .with_inner_size([1240.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_breeder_gui",
        options,
        Box::new(|_cc| Box::new(BreederApp::default())),
    )
}

// =============================================================================
// App state
// =============================================================================

struct BreederApp {
    ckpt_path: String,
    ledger_path: String,
    checkpoint: Option<Checkpoint>,
    ledger: Vec<LedgerEntry>,
    ledger_report: Option<LedgerReport>,
    op_counts: BTreeMap<String, u64>,
    seeds: u64,
    evaluated: usize,
    fit_min: f64,
    fit_mean: f64,
    fit_max: f64,
    best_id: Option<String>,
    best_fitness: f64,
    fitness_per_gen: Vec<(u32, f64)>,
    diversity_per_gen: Vec<(u32, f64)>,
    status: String,
    last_load: Instant,
    auto_refresh: bool,
    show_tab: Tab,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Population,
    Lineage,
    Ledger,
    Events,
    Fitness,
}

struct LedgerReport {
    total: usize,
    chain_ok: bool,
    reason: String,
}

impl Default for BreederApp {
    fn default() -> Self {
        Self {
            ckpt_path: "./breeder/latest.ckpt".to_string(),
            ledger_path: "./breeder/ledger.jsonl".to_string(),
            checkpoint: None,
            ledger: Vec::new(),
            ledger_report: None,
            op_counts: BTreeMap::new(),
            seeds: 0,
            evaluated: 0,
            fit_min: 0.0,
            fit_mean: 0.0,
            fit_max: 0.0,
            best_id: None,
            best_fitness: 0.0,
            fitness_per_gen: Vec::new(),
            diversity_per_gen: Vec::new(),
            status: "Ready. Point to a checkpoint and click Reload.".to_string(),
            last_load: Instant::now() - Duration::from_secs(60),
            auto_refresh: false,
            show_tab: Tab::Overview,
        }
    }
}

// =============================================================================
// Loading + replay
// =============================================================================

impl BreederApp {
    fn reload(&mut self) {
        self.checkpoint = None;
        self.ledger.clear();
        self.ledger_report = None;
        self.op_counts.clear();
        self.seeds = 0;
        self.evaluated = 0;
        self.fit_min = 0.0;
        self.fit_mean = 0.0;
        self.fit_max = 0.0;
        self.best_id = None;
        self.best_fitness = 0.0;
        self.fitness_per_gen.clear();
        self.diversity_per_gen.clear();

        let ckpt_path = PathBuf::from(&self.ckpt_path);
        if !ckpt_path.exists() {
            self.status = format!("checkpoint not found: {}", ckpt_path.display());
            return;
        }
        match ckpt::read(&ckpt_path) {
            Ok(c) => {
                self.summarise_checkpoint(&c);
                self.checkpoint = Some(c);
            }
            Err(e) => {
                self.status = format!("checkpoint: {e}");
                return;
            }
        }

        if !self.ledger_path.trim().is_empty() {
            let lpath = PathBuf::from(&self.ledger_path);
            if lpath.exists() {
                match load_ledger(&lpath) {
                    Ok(entries) => {
                        self.ledger_report = Some(verify_chain(&entries));
                        self.extract_timeseries(&entries);
                        self.ledger = entries;
                    }
                    Err(e) => {
                        self.status = format!("ledger: {e}");
                        return;
                    }
                }
            }
        }

        let ck_len = self
            .checkpoint
            .as_ref()
            .map(|c| c.population.len())
            .unwrap_or(0);
        self.status = format!(
            "Loaded checkpoint ({} units) + {} ledger entries",
            ck_len,
            self.ledger.len()
        );
        self.last_load = Instant::now();
    }

    fn summarise_checkpoint(&mut self, c: &Checkpoint) {
        let mut fmin = f64::INFINITY;
        let mut fmax = f64::NEG_INFINITY;
        let mut fsum = 0.0;
        let mut count = 0usize;
        for u in c.population.iter() {
            match &u.operator_born {
                Some(op) => {
                    *self
                        .op_counts
                        .entry(operator_label(op).to_string())
                        .or_insert(0) += 1
                }
                None => self.seeds += 1,
            }
            if let Some(f) = &u.fitness {
                count += 1;
                fsum += f.aggregate;
                if f.aggregate < fmin {
                    fmin = f.aggregate;
                }
                if f.aggregate > fmax {
                    fmax = f.aggregate;
                }
            }
        }
        self.evaluated = count;
        if count > 0 {
            self.fit_min = fmin;
            self.fit_max = fmax;
            self.fit_mean = fsum / count as f64;
        }
        if let Some(best) = c.population.best() {
            self.best_id = Some(best.id.clone());
            self.best_fitness = best.fitness_value();
        }
    }

    fn extract_timeseries(&mut self, entries: &[LedgerEntry]) {
        let mut per_gen_best: BTreeMap<u32, f64> = BTreeMap::new();
        let mut cur_gen: u32 = 0;
        for e in entries {
            match &e.event {
                BreederEvent::GenerationStarted { generation } => cur_gen = *generation,
                BreederEvent::FitnessEvaluated { score, .. } => {
                    let entry = per_gen_best.entry(cur_gen).or_insert(f64::NEG_INFINITY);
                    if score.aggregate > *entry {
                        *entry = score.aggregate;
                    }
                }
                BreederEvent::DiversityComputed { generation, score } => {
                    self.diversity_per_gen.push((*generation, *score));
                }
                _ => {}
            }
        }
        self.fitness_per_gen = per_gen_best
            .into_iter()
            .filter(|(_, v)| v.is_finite())
            .collect();
    }
}

fn load_ledger(path: &std::path::Path) -> Result<Vec<LedgerEntry>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ev: LedgerEntry =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        out.push(ev);
    }
    Ok(out)
}

fn verify_chain(entries: &[LedgerEntry]) -> LedgerReport {
    let mut rep = LedgerReport {
        total: entries.len(),
        chain_ok: true,
        reason: String::new(),
    };
    for (i, ev) in entries.iter().enumerate() {
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
        if i == 0 {
            if !ev.prev_hash_hex.is_empty() {
                rep.chain_ok = false;
                rep.reason = format!("first entry must have empty prev_hash (seq {})", ev.seq);
                break;
            }
        } else if ev.prev_hash_hex != entries[i - 1].self_hash_hex {
            rep.chain_ok = false;
            rep.reason = format!("chain break before seq {}", ev.seq);
            break;
        }
    }
    rep
}

fn operator_label(op: &MutationOperator) -> &'static str {
    match op {
        MutationOperator::ZeroOrder => "ZeroOrder",
        MutationOperator::FirstOrder => "FirstOrder",
        MutationOperator::Eda => "Eda",
        MutationOperator::EdaRankAndIndex => "EdaRankAndIndex",
        MutationOperator::LineageBased => "LineageBased",
        MutationOperator::HyperMutationZeroOrder => "HyperMutationZeroOrder",
        MutationOperator::HyperMutationFirstOrder => "HyperMutationFirstOrder",
        MutationOperator::Lamarckian => "Lamarckian",
        MutationOperator::PromptCrossover => "PromptCrossover",
        _ => "Unknown",
    }
}

fn event_label(ev: &BreederEvent) -> String {
    match ev {
        BreederEvent::RunStarted { run_id, .. } => format!("RunStarted run_id={run_id}"),
        BreederEvent::SeedBootstrapped { n, source } => {
            format!("SeedBootstrapped n={n} source={source}")
        }
        BreederEvent::SeedInserted { unit_id, .. } => format!("SeedInserted unit_id={unit_id}"),
        BreederEvent::GenerationStarted { generation } => {
            format!("GenerationStarted gen={generation}")
        }
        BreederEvent::MutationApplied {
            parent_id,
            child_id,
            operator,
        } => format!(
            "MutationApplied op={} parent={parent_id} child={child_id}",
            operator_label(operator)
        ),
        BreederEvent::MutationRejected {
            parent_id,
            operator,
            reason,
        } => format!(
            "MutationRejected op={} parent={parent_id} reason={reason:?}",
            operator_label(operator)
        ),
        BreederEvent::FitnessEvaluated {
            unit_id,
            score,
            cached,
        } => format!(
            "FitnessEvaluated unit={unit_id} agg={:.4} cached={cached}",
            score.aggregate
        ),
        BreederEvent::SelectionPerformed {
            strategy,
            survivors,
        } => format!(
            "SelectionPerformed strategy={strategy:?} survivors={}",
            survivors.len()
        ),
        BreederEvent::DiversityComputed { generation, score } => {
            format!("DiversityComputed gen={generation} score={score:.4}")
        }
        BreederEvent::EvalAugmented {
            n_added,
            augmenter_kind,
        } => format!("EvalAugmented n_added={n_added} kind={augmenter_kind}"),
        BreederEvent::LineageNarrated { unit_id, .. } => {
            format!("LineageNarrated unit_id={unit_id}")
        }
        BreederEvent::SmoothingSampled { unit_id, samples } => {
            format!("SmoothingSampled unit_id={unit_id} samples={samples}")
        }
        BreederEvent::BudgetExhausted { kind, value } => {
            format!("BudgetExhausted kind={kind:?} value={value:.4}")
        }
        BreederEvent::CheckpointWritten { path, .. } => {
            format!("CheckpointWritten path={path}")
        }
        BreederEvent::FreezeChanged { frozen } => format!("FreezeChanged frozen={frozen}"),
        BreederEvent::SafetyFilterApplied { filter_kind } => {
            format!("SafetyFilterApplied kind={filter_kind}")
        }
        BreederEvent::RunCompleted {
            run_id,
            best_id,
            generations,
        } => format!("RunCompleted run_id={run_id} best={best_id} gens={generations}"),
        BreederEvent::RunAborted { run_id, reason } => {
            format!("RunAborted run_id={run_id} reason={reason:?}")
        }
        _ => "UnknownEvent".to_string(),
    }
}

// =============================================================================
// UI
// =============================================================================

impl eframe::App for BreederApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_refresh && self.last_load.elapsed() > Duration::from_secs(5) {
            self.reload();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Checkpoint:");
                ui.add(egui::TextEdit::singleline(&mut self.ckpt_path).desired_width(340.0));
                ui.label("Ledger:");
                ui.add(egui::TextEdit::singleline(&mut self.ledger_path).desired_width(280.0));
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto (5s)");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.show_tab, Tab::Overview, "Overview");
                ui.selectable_value(&mut self.show_tab, Tab::Population, "Population");
                ui.selectable_value(&mut self.show_tab, Tab::Lineage, "Lineage");
                ui.selectable_value(&mut self.show_tab, Tab::Ledger, "Ledger");
                ui.selectable_value(&mut self.show_tab, Tab::Events, "Events");
                ui.selectable_value(&mut self.show_tab, Tab::Fitness, "Fitness");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.show_tab {
            Tab::Overview => self.render_overview(ui),
            Tab::Population => self.render_population(ui),
            Tab::Lineage => self.render_lineage(ui),
            Tab::Ledger => self.render_ledger(ui),
            Tab::Events => self.render_events(ui),
            Tab::Fitness => self.render_fitness(ui),
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(&self.status);
        });

        if self.auto_refresh {
            ctx.request_repaint_after(Duration::from_secs(1));
        }
    }
}

impl BreederApp {
    fn render_overview(&self, ui: &mut egui::Ui) {
        ui.heading("PromptBreeder V97 Auditor");
        ui.separator();
        let Some(c) = &self.checkpoint else {
            ui.label("(no checkpoint loaded)");
            return;
        };
        egui::Grid::new("overview").num_columns(2).show(ui, |ui| {
            kv(ui, "Run ID", &c.run_id);
            kv(ui, "Generation", &c.generation.to_string());
            kv(ui, "Population size", &c.population.len().to_string());
            kv(ui, "Config hash", &c.config_hash_hex);
            kv(ui, "Ledger tip hash", &c.ledger_tip_hash_hex);
            kv(
                ui,
                "Mean fitness",
                &format!("{:.4}", c.population.mean_fitness()),
            );
            kv(
                ui,
                "Best",
                &format!(
                    "{} (fitness {:.4})",
                    self.best_id.as_deref().unwrap_or("-"),
                    self.best_fitness
                ),
            );
            kv(
                ui,
                "Evaluated",
                &format!("{}/{}", self.evaluated, c.population.len()),
            );
            if self.evaluated > 0 {
                kv(
                    ui,
                    "Fitness (min / mean / max)",
                    &format!(
                        "{:.4} / {:.4} / {:.4}",
                        self.fit_min, self.fit_mean, self.fit_max
                    ),
                );
            }
            kv(ui, "Seeds in pop", &self.seeds.to_string());
        });
        ui.separator();
        if let Some(rep) = &self.ledger_report {
            ui.heading("Ledger");
            if rep.chain_ok {
                ui.colored_label(
                    egui::Color32::GREEN,
                    format!("Chain OK — {} entries", rep.total),
                );
            } else {
                ui.colored_label(egui::Color32::RED, format!("BROKEN: {}", rep.reason));
            }
        }
    }

    fn render_population(&self, ui: &mut egui::Ui) {
        ui.heading("Population");
        ui.separator();
        let Some(c) = &self.checkpoint else {
            ui.label("(no checkpoint loaded)");
            return;
        };
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("pop")
                .num_columns(6)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("id");
                    ui.strong("gen");
                    ui.strong("op");
                    ui.strong("fitness");
                    ui.strong("parents");
                    ui.strong("task_prompt");
                    ui.end_row();
                    for u in c.population.iter() {
                        ui.label(&u.id);
                        ui.label(u.generation_born.to_string());
                        ui.label(u.operator_born.as_ref().map(operator_label).unwrap_or("-"));
                        ui.label(
                            u.fitness
                                .as_ref()
                                .map(|f| format!("{:.4}", f.aggregate))
                                .unwrap_or_else(|| "-".to_string()),
                        );
                        ui.label(u.parents.len().to_string());
                        let preview: String = u.task_prompt.chars().take(80).collect();
                        ui.label(preview);
                        ui.end_row();
                    }
                });
        });
    }

    fn render_lineage(&self, ui: &mut egui::Ui) {
        ui.heading("Lineage DAG");
        ui.separator();
        let Some(c) = &self.checkpoint else {
            ui.label("(no checkpoint loaded)");
            return;
        };
        let edges: usize = c.lineage.parents.values().map(|v| v.len()).sum();
        ui.label(format!("Nodes: {}", c.lineage.parents.len()));
        ui.label(format!("Edges (parent→child): {}", edges));
        ui.separator();
        if self.op_counts.is_empty() {
            ui.label("(no mutation-born units)");
        } else {
            ui.label(format!("Seeds: {}", self.seeds));
            egui::Grid::new("ops")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("operator");
                    ui.strong("count");
                    ui.end_row();
                    for (k, v) in &self.op_counts {
                        ui.label(k);
                        ui.label(v.to_string());
                        ui.end_row();
                    }
                });
        }
    }

    fn render_ledger(&self, ui: &mut egui::Ui) {
        ui.heading("Ledger chain");
        ui.separator();
        if let Some(rep) = &self.ledger_report {
            if rep.chain_ok {
                ui.colored_label(
                    egui::Color32::GREEN,
                    format!("Chain OK — {} entries", rep.total),
                );
            } else {
                ui.colored_label(egui::Color32::RED, format!("BROKEN: {}", rep.reason));
            }
        } else {
            ui.label("(no ledger loaded)");
        }
    }

    fn render_events(&self, ui: &mut egui::Ui) {
        ui.heading("Recent events");
        ui.separator();
        if self.ledger.is_empty() {
            ui.label("(no ledger loaded)");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for ev in self.ledger.iter().rev().take(500) {
                ui.label(format!(
                    "[{:>6}] {} signer={} {}",
                    ev.seq,
                    ev.timestamp.to_rfc3339(),
                    ev.signer,
                    event_label(&ev.event)
                ));
            }
        });
    }

    fn render_fitness(&self, ui: &mut egui::Ui) {
        ui.heading("Fitness + diversity timeline");
        ui.separator();
        if self.fitness_per_gen.is_empty() && self.diversity_per_gen.is_empty() {
            ui.label("(no generation events in ledger)");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.strong("Best fitness per generation");
            egui::Grid::new("fit")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("gen");
                    ui.strong("best");
                    ui.end_row();
                    for (g, v) in &self.fitness_per_gen {
                        ui.label(g.to_string());
                        ui.label(format!("{v:.4}"));
                        ui.end_row();
                    }
                });
            ui.separator();
            ui.strong("Diversity per generation");
            egui::Grid::new("div")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("gen");
                    ui.strong("score");
                    ui.end_row();
                    for (g, v) in &self.diversity_per_gen {
                        ui.label(g.to_string());
                        ui.label(format!("{v:.4}"));
                        ui.end_row();
                    }
                });
        });
    }
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.strong(k);
    ui.label(v);
    ui.end_row();
}
