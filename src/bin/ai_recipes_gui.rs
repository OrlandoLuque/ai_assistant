//! `ai_recipes_gui` — visual auditor for recipe definitions.
//!
//! Build: cargo build --release --bin ai_recipes_gui --features gui-recipes
//! Run:   cargo run --bin ai_recipes_gui --features gui-recipes
//!
//! Read-only audit of recipes discovered under a chosen directory.
//! Shows: list, metadata, steps, sub-recipe call graph, validation
//! results. Per memory rule `feedback_auditable_subsystems`.

use ai_assistant::{
    discover_recipes, validate_recipe, Recipe, RecipeConfig, RecipeRegistry, StepKind,
};
use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_recipes — Recipes Auditor")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_recipes_gui",
        options,
        Box::new(|_cc| Box::new(RecipesAuditApp::default())),
    )
}

struct RecipesAuditApp {
    dir_input: String,
    registry: RecipeRegistry,
    selected: Option<String>,
    status: String,
}

impl Default for RecipesAuditApp {
    fn default() -> Self {
        let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        p.push(".ai_assistant");
        p.push("recipes");
        Self {
            dir_input: p.display().to_string(),
            registry: RecipeRegistry::new(),
            selected: None,
            status: "Press 'Reload' to discover recipes.".into(),
        }
    }
}

impl eframe::App for RecipesAuditApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Recipes dir:");
                ui.text_edit_singleline(&mut self.dir_input);
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                ui.label(&self.status);
            });
        });

        egui::SidePanel::left("recipes-list")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading(format!("Recipes ({})", self.registry.len()));
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let names = self.registry.names();
                    for name in names {
                        let selected = self.selected.as_deref() == Some(name.as_str());
                        if ui.selectable_label(selected, &name).clicked() {
                            self.selected = Some(name);
                        }
                    }
                });
                if !self.registry.load_errors.is_empty() {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        format!("{} load error(s)", self.registry.load_errors.len()),
                    );
                    egui::ScrollArea::vertical()
                        .id_source("errs")
                        .max_height(120.0)
                        .show(ui, |ui| {
                            for (p, e) in &self.registry.load_errors {
                                ui.label(format!("{}: {}", p.display(), e));
                            }
                        });
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let cfg = RecipeConfig::default();
            if let Some(name) = &self.selected.clone() {
                if let Some(r) = self.registry.get(name) {
                    self.show_recipe_detail(ui, r, &cfg);
                } else {
                    ui.label("Selected recipe disappeared from registry.");
                }
            } else {
                ui.label("Select a recipe from the list to inspect it.");
                ui.separator();
                self.show_summary(ui, &cfg);
            }
        });
    }
}

impl RecipesAuditApp {
    fn reload(&mut self) {
        let dir = PathBuf::from(&self.dir_input);
        if !dir.is_dir() {
            self.status = format!("Not a directory: {}", dir.display());
            self.registry = RecipeRegistry::new();
            return;
        }
        let cfg = RecipeConfig::default();
        self.registry = discover_recipes(&[dir.clone()], &cfg);
        self.status = format!(
            "Loaded {} recipe(s), {} error(s)",
            self.registry.len(),
            self.registry.load_errors.len()
        );
        if let Some(sel) = &self.selected {
            if self.registry.get(sel).is_none() {
                self.selected = None;
            }
        }
    }

    fn show_summary(&self, ui: &mut egui::Ui, cfg: &RecipeConfig) {
        let total = self.registry.len();
        let mut valid = 0usize;
        let mut shell_count = 0usize;
        let mut step_total = 0usize;
        for (_, r) in self.registry.iter() {
            step_total += r.steps.len();
            for s in &r.steps {
                if matches!(s.kind, StepKind::Shell { .. }) {
                    shell_count += 1;
                }
            }
            if validate_recipe(r, cfg).is_ok() {
                valid += 1;
            }
        }
        ui.heading("Audit summary");
        ui.label(format!("Total recipes: {}", total));
        ui.label(format!("Valid: {}", valid));
        ui.label(format!("Invalid: {}", total - valid));
        ui.label(format!("Total steps: {}", step_total));
        ui.colored_label(
            if shell_count > 0 {
                egui::Color32::YELLOW
            } else {
                egui::Color32::WHITE
            },
            format!("Shell steps: {} (security-sensitive)", shell_count),
        );

        ui.separator();
        ui.heading("Sub-recipe call graph");
        let mut any_edges = false;
        for (name, r) in self.registry.iter() {
            for s in &r.steps {
                if let StepKind::Recipe { recipe, .. } = &s.kind {
                    any_edges = true;
                    let exists = self.registry.get(recipe).is_some();
                    let lbl = format!("{} → {}", name, recipe);
                    if exists {
                        ui.label(lbl);
                    } else {
                        ui.colored_label(egui::Color32::RED, format!("{} (MISSING)", lbl));
                    }
                }
            }
        }
        if !any_edges {
            ui.label("(no sub-recipe calls detected)");
        }
    }

    fn show_recipe_detail(&self, ui: &mut egui::Ui, r: &Recipe, cfg: &RecipeConfig) {
        ui.heading(&r.name);
        ui.label(format!(
            "API: {}    Version: {}",
            r.api_version,
            r.version.as_deref().unwrap_or("-")
        ));
        if let Some(d) = &r.description {
            ui.label(d);
        }
        if let Some(a) = &r.author {
            ui.label(format!("Author: {}", a));
        }
        if !r.tags.is_empty() {
            ui.label(format!("Tags: {}", r.tags.join(", ")));
        }
        if let Some(m) = &r.model {
            ui.label(format!("Model: {}", m));
        }
        ui.label(format!("Source: {}", r.source_path.display()));

        ui.separator();
        match validate_recipe(r, cfg) {
            Ok(()) => ui.colored_label(egui::Color32::LIGHT_GREEN, "✓ Schema valid"),
            Err(e) => ui.colored_label(egui::Color32::RED, format!("✗ {}", e)),
        };

        ui.separator();
        if !r.variables.is_empty() {
            ui.heading(format!("Variables ({})", r.variables.len()));
            egui::Grid::new("vars-grid")
                .num_columns(4)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Name");
                    ui.strong("Required");
                    ui.strong("Default");
                    ui.strong("Description");
                    ui.end_row();
                    for v in &r.variables {
                        ui.label(&v.name);
                        ui.label(if v.required { "yes" } else { "no" });
                        ui.label(v.default.as_deref().unwrap_or("-"));
                        ui.label(v.description.as_deref().unwrap_or(""));
                        ui.end_row();
                    }
                });
            ui.separator();
        }

        ui.heading(format!("Steps ({})", r.steps.len()));
        egui::ScrollArea::vertical().show(ui, |ui| {
            for s in &r.steps {
                let (kind, detail) = match &s.kind {
                    StepKind::Prompt { prompt } => ("prompt", short(prompt)),
                    StepKind::Tool { tool, args } => {
                        ("tool", format!("{} ({} args)", tool, args.len()))
                    }
                    StepKind::Recipe { recipe, args } => {
                        ("recipe", format!("{} ({} args)", recipe, args.len()))
                    }
                    StepKind::Shell { command } => ("shell", short(command)),
                };
                let color = if kind == "shell" {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::WHITE
                };
                ui.horizontal(|ui| {
                    ui.colored_label(color, format!("[{}]", kind));
                    ui.strong(&s.id);
                    ui.label(detail);
                });
            }
        });

        if let Some(o) = &r.output {
            ui.separator();
            ui.label(format!("Output: {}", short(o)));
        }
    }
}

fn short(s: &str) -> String {
    let one = s.replace('\n', " ");
    if one.len() > 90 {
        format!("{}…", ai_assistant::text_util::truncate_str(&one, 90))
    } else {
        one
    }
}
