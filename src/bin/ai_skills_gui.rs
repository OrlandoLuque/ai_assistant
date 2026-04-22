//! `ai_skills_gui` — Skill Forge auditor GUI (egui/eframe).
//!
//! Desktop companion to `ai_skills`. Browses a directory of `.skill.json`
//! files, shows metadata, verifies content/blake3 hashes, and renders the
//! ledger chain for any skill store.
//!
//! Build: `cargo build --release --bin ai_skills_gui --features "skill-forge gui-pro"`.

use ai_assistant::{LedgerEvent, SkillDefinition, SkillMode, SkillStatus};
use eframe::egui;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_skills — Skill Forge Auditor")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_skills_gui",
        options,
        Box::new(|_cc| Box::new(SkillsApp::default())),
    )
}

// =============================================================================
// App state
// =============================================================================

struct SkillsApp {
    /// Directory holding `*.skill.json` files.
    skills_dir: String,
    /// Path to an optional ledger JSONL file.
    ledger_path: String,
    /// Loaded skills (latest per file).
    skills: Vec<LoadedSkill>,
    /// Selected index in `skills`.
    selected: Option<usize>,
    /// Loaded ledger events.
    ledger: Vec<LedgerEvent>,
    /// Verification report for the ledger chain.
    ledger_report: Option<LedgerReport>,
    /// Status line at the bottom.
    status: String,
    /// Timestamp of last load (for refresh throttling).
    last_load: Instant,
    /// Auto-refresh every N seconds.
    auto_refresh: bool,
}

struct LoadedSkill {
    path: PathBuf,
    def: SkillDefinition,
    content_ok: bool,
    content_expected: String,
    wasm_ok: Option<bool>,
    wasm_expected: Option<String>,
}

struct LedgerReport {
    total: usize,
    chain_ok: bool,
    first_bad_seq: Option<u64>,
    reason: String,
}

impl Default for SkillsApp {
    fn default() -> Self {
        Self {
            skills_dir: "./skills".to_string(),
            ledger_path: "./skills/ledger.jsonl".to_string(),
            skills: Vec::new(),
            selected: None,
            ledger: Vec::new(),
            ledger_report: None,
            status: "Ready. Choose a skills directory and click Refresh.".to_string(),
            last_load: Instant::now() - Duration::from_secs(60),
            auto_refresh: false,
        }
    }
}

// =============================================================================
// Loading + verification
// =============================================================================

impl SkillsApp {
    fn reload(&mut self) {
        self.skills.clear();
        self.selected = None;
        let dir = PathBuf::from(&self.skills_dir);
        if !dir.is_dir() {
            self.status = format!("Not a directory: {}", dir.display());
            return;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                self.status = format!("read_dir: {e}");
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let is_skill = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".skill.json"))
                .unwrap_or(false);
            if !is_skill {
                continue;
            }
            match load_skill(&path) {
                Ok(def) => {
                    let expected = def.compute_content_hash();
                    let content_ok = expected == def.content_hash_hex;
                    let (wasm_ok, wasm_expected) = match &def.mode {
                        SkillMode::Wasm(a) => {
                            let h = blake3::hash(&a.bytes).to_hex().to_string();
                            (Some(h == a.blake3_hex), Some(h))
                        }
                        _ => (None, None),
                    };
                    self.skills.push(LoadedSkill {
                        path,
                        def,
                        content_ok,
                        content_expected: expected,
                        wasm_ok,
                        wasm_expected,
                    });
                }
                Err(e) => {
                    self.status = format!("skip {}: {e}", path.display());
                }
            }
        }
        self.skills
            .sort_by(|a, b| a.def.id.as_str().cmp(b.def.id.as_str()));
        self.status = format!("Loaded {} skills from {}", self.skills.len(), dir.display());
        self.last_load = Instant::now();
    }

    fn reload_ledger(&mut self) {
        self.ledger.clear();
        self.ledger_report = None;
        let path = PathBuf::from(&self.ledger_path);
        if !path.exists() {
            self.status = format!("Ledger not found: {}", path.display());
            return;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                self.status = format!("read ledger: {e}");
                return;
            }
        };
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<LedgerEvent>(line) {
                Ok(ev) => self.ledger.push(ev),
                Err(e) => {
                    self.status = format!("ledger line {}: {e}", n + 1);
                    return;
                }
            }
        }
        let mut report = LedgerReport {
            total: self.ledger.len(),
            chain_ok: true,
            first_bad_seq: None,
            reason: String::new(),
        };
        for (i, ev) in self.ledger.iter().enumerate() {
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
            if i > 0 && ev.prev_hash_hex != self.ledger[i - 1].self_hash_hex {
                report.chain_ok = false;
                report.first_bad_seq = Some(ev.seq);
                report.reason = format!("chain break before seq {}", ev.seq);
                break;
            }
        }
        self.status = format!(
            "Ledger: {} events, chain {}",
            report.total,
            if report.chain_ok { "OK" } else { "BROKEN" }
        );
        self.ledger_report = Some(report);
    }
}

// =============================================================================
// UI
// =============================================================================

impl eframe::App for SkillsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_refresh && self.last_load.elapsed() > Duration::from_secs(5) {
            self.reload();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Skills dir:");
                ui.add(egui::TextEdit::singleline(&mut self.skills_dir).desired_width(320.0));
                if ui.button("Refresh").clicked() {
                    self.reload();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto (5s)");
                ui.separator();
                ui.label("Ledger:");
                ui.add(egui::TextEdit::singleline(&mut self.ledger_path).desired_width(320.0));
                if ui.button("Verify ledger").clicked() {
                    self.reload_ledger();
                }
            });
        });

        egui::SidePanel::left("skills_list")
            .min_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Skills");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, s) in self.skills.iter().enumerate() {
                        let label = format!(
                            "{} {} {}",
                            if s.content_ok { "OK" } else { "!!" },
                            s.def.id.as_str(),
                            s.def.version,
                        );
                        let response = ui.selectable_label(self.selected == Some(i), label);
                        if response.clicked() {
                            self.selected = Some(i);
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(i) = self.selected {
                if let Some(s) = self.skills.get(i) {
                    render_detail(ui, s);
                }
            } else {
                ui.heading("Skill Forge Auditor");
                ui.label("Select a skill from the left panel to inspect.");
                ui.separator();
                if let Some(rep) = &self.ledger_report {
                    ui.heading("Ledger");
                    ui.label(format!("Events: {}", rep.total));
                    ui.label(format!(
                        "Chain: {}",
                        if rep.chain_ok { "OK" } else { "BROKEN" }
                    ));
                    if !rep.chain_ok {
                        ui.colored_label(egui::Color32::RED, &rep.reason);
                    }
                }
            }
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        if self.auto_refresh {
            ctx.request_repaint_after(Duration::from_secs(1));
        }
    }
}

fn render_detail(ui: &mut egui::Ui, s: &LoadedSkill) {
    ui.heading(s.def.id.as_str());
    ui.label(format!("File: {}", s.path.display()));
    ui.separator();
    egui::Grid::new("skill_grid").num_columns(2).show(ui, |ui| {
        kv(ui, "Version", &s.def.version.to_string());
        kv(ui, "Name", &s.def.name);
        kv(ui, "Status", &status_label(s.def.status));
        kv(ui, "Tenant", &s.def.tenant);
        kv(ui, "Shared xtenant", &s.def.shared_cross_tenant.to_string());
        kv(ui, "Capabilities", &s.def.capabilities.len().to_string());
        kv(ui, "Description", &s.def.description);
        kv(ui, "Content hash", &s.def.content_hash_hex);
    });
    ui.separator();
    if s.content_ok {
        ui.colored_label(egui::Color32::GREEN, "Content hash: OK");
    } else {
        ui.colored_label(egui::Color32::RED, "Content hash: MISMATCH");
        ui.label(format!("  expected: {}", s.content_expected));
        ui.label(format!("  stored:   {}", s.def.content_hash_hex));
    }
    if let Some(ok) = s.wasm_ok {
        if ok {
            ui.colored_label(egui::Color32::GREEN, "WASM blake3: OK");
        } else {
            ui.colored_label(egui::Color32::RED, "WASM blake3: MISMATCH");
            if let Some(h) = &s.wasm_expected {
                ui.label(format!("  expected: {h}"));
            }
        }
    }
    ui.separator();
    ui.collapsing("Mode details", |ui| match &s.def.mode {
        SkillMode::Declarative(steps) => {
            ui.label(format!("Declarative — {} step(s)", steps.len()));
            for (idx, st) in steps.iter().enumerate() {
                ui.label(format!("  [{idx}] {:?}", st.kind));
            }
        }
        SkillMode::Wasm(a) => {
            ui.label(format!("Wasm — {} bytes", a.bytes.len()));
            ui.label(format!("blake3: {}", a.blake3_hex));
            ui.label(format!("signed_by: {}", a.signed_by));
            ui.label(format!("toolchain: {}", a.compile_fingerprint));
            if let Some(src) = &a.source_path {
                ui.label(format!("source: {src}"));
            }
        }
        _ => {
            ui.label("(unknown mode — future variant)");
        }
    });
    ui.collapsing("Raw capabilities", |ui| {
        for c in s.def.capabilities.iter() {
            ui.label(format!("{c:?}"));
        }
    });
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.strong(k);
    ui.label(v);
    ui.end_row();
}

fn status_label(s: SkillStatus) -> String {
    format!("{s}")
}

// =============================================================================
// Helpers
// =============================================================================

fn load_skill(file: &PathBuf) -> Result<SkillDefinition, String> {
    let bytes = fs::read(file).map_err(|e| format!("read: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse: {e}"))
}
