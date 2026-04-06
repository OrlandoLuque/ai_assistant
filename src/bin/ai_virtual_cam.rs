//! ai_virtual_cam — Real-time webcam effects GUI application.
//!
//! Captures from a physical webcam (or test pattern), processes through a
//! composable video effects chain, and displays a live preview. Virtual camera
//! output planned for Phase 2 (v4l2loopback on Linux, OBS VirtualCam on Windows).
//!
//! Run: `cargo run --bin ai_virtual_cam --features video-io`

use ai_assistant::video_filter::*;
use eframe::egui;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

// ============================================================================
// App State
// ============================================================================

struct VirtualCamApp {
    // Source
    source_mode: SourceMode,
    test_pattern: TestPatternGenerator,

    // Video state
    frame_width: u32,
    frame_height: u32,
    preview_texture: Option<egui::TextureHandle>,
    fps_counter: FpsCounter,

    // Effects (each toggle + parameters)
    mirror_enabled: bool,
    mirror_horizontal: bool,
    pixelate_enabled: bool,
    pixelate_size: u32,
    color_temp_enabled: bool,
    color_temperature: f32,
    vignette_enabled: bool,
    vignette_strength: f32,
    brightness_contrast_enabled: bool,
    brightness: f32,
    contrast: f32,
    grayscale_enabled: bool,
    sepia_enabled: bool,
    invert_enabled: bool,
    posterize_enabled: bool,
    posterize_levels: u8,
    night_vision_enabled: bool,
    vhs_retro_enabled: bool,
    glitch_enabled: bool,
    glitch_intensity: f32,
    cartoon_enabled: bool,
    cartoon_levels: u8,
    cartoon_threshold: u8,
    edge_detect_enabled: bool,
    edge_threshold: u8,
    blur_enabled: bool,
    blur_radius: u32,
    chroma_key_enabled: bool,
    chroma_hue: f32,
    chroma_tolerance: f32,
    matrix_rain_enabled: bool,
    matrix_cell_size: u32,
    matrix_overlay: bool,
    matrix_opacity: f32,
    watermark_enabled: bool,
    watermark_text: String,
    ascii_art_enabled: bool,
    ascii_cell_size: u32,

    // Processing
    is_running: Arc<AtomicBool>,
    latest_frame: Arc<Mutex<Option<VideoFrame>>>,
    status_message: String,
}

#[derive(Clone, Copy, PartialEq)]
enum SourceMode {
    TestColorBars,
    TestGradient,
    // Webcam(usize), // Phase 2: nokhwa capture
}

struct FpsCounter {
    frame_times: std::collections::VecDeque<std::time::Instant>,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_times: std::collections::VecDeque::new(),
        }
    }
    fn tick(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push_back(now);
        while self
            .frame_times
            .front()
            .map(|t| now.duration_since(*t).as_secs_f32() > 1.0)
            .unwrap_or(false)
        {
            self.frame_times.pop_front();
        }
    }
    fn fps(&self) -> f32 {
        self.frame_times.len() as f32
    }
}

impl VirtualCamApp {
    fn new() -> Self {
        Self {
            source_mode: SourceMode::TestGradient,
            test_pattern: TestPatternGenerator::new(),
            frame_width: 640,
            frame_height: 480,
            preview_texture: None,
            fps_counter: FpsCounter::new(),
            mirror_enabled: false,
            mirror_horizontal: true,
            pixelate_enabled: false,
            pixelate_size: 8,
            color_temp_enabled: false,
            color_temperature: 0.0,
            vignette_enabled: false,
            vignette_strength: 0.5,
            brightness_contrast_enabled: false,
            brightness: 0.0,
            contrast: 1.0,
            grayscale_enabled: false,
            sepia_enabled: false,
            invert_enabled: false,
            posterize_enabled: false,
            posterize_levels: 4,
            night_vision_enabled: false,
            vhs_retro_enabled: false,
            glitch_enabled: false,
            glitch_intensity: 0.3,
            cartoon_enabled: false,
            cartoon_levels: 6,
            cartoon_threshold: 40,
            edge_detect_enabled: false,
            edge_threshold: 30,
            blur_enabled: false,
            blur_radius: 2,
            chroma_key_enabled: false,
            chroma_hue: 120.0,
            chroma_tolerance: 40.0,
            matrix_rain_enabled: false,
            matrix_cell_size: 8,
            matrix_overlay: true,
            matrix_opacity: 0.5,
            watermark_enabled: false,
            watermark_text: "ai_virtual_cam".into(),
            ascii_art_enabled: false,
            ascii_cell_size: 8,
            is_running: Arc::new(AtomicBool::new(false)),
            latest_frame: Arc::new(Mutex::new(None)),
            status_message: "Ready. Click Start.".into(),
        }
    }

    fn build_effect_chain(&self) -> VideoEffectChain {
        let mut chain = VideoEffectChain::new();
        // Input processing
        if self.mirror_enabled {
            chain.add_effect(if self.mirror_horizontal {
                Box::new(MirrorEffect::horizontal())
            } else {
                Box::new(MirrorEffect::vertical())
            });
        }
        // Color grading
        if self.brightness_contrast_enabled {
            chain.add_effect(Box::new(BrightnessContrastEffect::new(
                self.brightness,
                self.contrast,
            )));
        }
        if self.color_temp_enabled {
            chain.add_effect(Box::new(ColorTemperatureEffect::new(
                self.color_temperature,
            )));
        }
        if self.vignette_enabled {
            chain.add_effect(Box::new(VignetteEffect::new(self.vignette_strength)));
        }
        if self.grayscale_enabled {
            chain.add_effect(Box::new(GrayscaleEffect::new()));
        }
        if self.sepia_enabled {
            chain.add_effect(Box::new(SepiaEffect::new()));
        }
        if self.invert_enabled {
            chain.add_effect(Box::new(InvertEffect::new()));
        }
        if self.posterize_enabled {
            chain.add_effect(Box::new(PosterizeEffect::new(self.posterize_levels)));
        }
        // Background
        if self.chroma_key_enabled {
            let mut ck = ChromaKeyEffect::green_screen();
            ck.target_hue = self.chroma_hue;
            ck.hue_tolerance = self.chroma_tolerance;
            chain.add_effect(Box::new(ck));
        }
        // Creative
        if self.blur_enabled {
            chain.add_effect(Box::new(BlurEffect::new(self.blur_radius)));
        }
        if self.pixelate_enabled {
            chain.add_effect(Box::new(PixelateEffect::new(self.pixelate_size)));
        }
        if self.night_vision_enabled {
            chain.add_effect(Box::new(NightVisionEffect::new()));
        }
        if self.vhs_retro_enabled {
            chain.add_effect(Box::new(VhsRetroEffect::new()));
        }
        if self.glitch_enabled {
            chain.add_effect(Box::new(GlitchEffect::new(self.glitch_intensity)));
        }
        if self.cartoon_enabled {
            chain.add_effect(Box::new(CartoonEffect::new(
                self.cartoon_levels,
                self.cartoon_threshold,
            )));
        }
        if self.edge_detect_enabled {
            chain.add_effect(Box::new(EdgeDetectionEffect::new(self.edge_threshold)));
        }
        if self.ascii_art_enabled {
            chain.add_effect(Box::new(AsciiArtEffect::new(self.ascii_cell_size)));
        }
        // Overlay
        if self.matrix_rain_enabled {
            let mut mr = MatrixRainEffect::new(self.matrix_cell_size);
            mr.overlay_mode = self.matrix_overlay;
            mr.overlay_opacity = self.matrix_opacity;
            chain.add_effect(Box::new(mr));
        }
        if self.watermark_enabled {
            chain.add_effect(Box::new(WatermarkEffect::new(&self.watermark_text)));
        }
        chain
    }

    fn start(&mut self) {
        if self.is_running.load(Ordering::Relaxed) {
            return;
        }
        self.is_running.store(true, Ordering::Relaxed);
        self.status_message = format!(
            "Running: {} @ {}x{}",
            match self.source_mode {
                SourceMode::TestColorBars => "Test: Color Bars",
                SourceMode::TestGradient => "Test: Gradient",
            },
            self.frame_width,
            self.frame_height
        );
    }

    fn stop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        self.status_message = "Stopped.".into();
    }
}

// ============================================================================
// eframe App
// ============================================================================

impl eframe::App for VirtualCamApp {
    fn update(&mut self, ctx: &egui::Context, _eframe: &mut eframe::Frame) {
        // Generate + process frame each GUI tick when running
        if self.is_running.load(Ordering::Relaxed) {
            let mut frame = match self.source_mode {
                SourceMode::TestColorBars => self
                    .test_pattern
                    .color_bars(self.frame_width, self.frame_height),
                SourceMode::TestGradient => self
                    .test_pattern
                    .gradient(self.frame_width, self.frame_height),
            };
            let mut chain = self.build_effect_chain();
            chain.process_frame(&mut frame);
            self.fps_counter.tick();

            // Update preview texture
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize],
                &frame.data,
            );
            match &mut self.preview_texture {
                Some(tex) => tex.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.preview_texture =
                        Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR));
                }
            }
        }

        // ── Left panel: controls ──
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("ai_virtual_cam");
                ui.small(&self.status_message);
                if self.is_running.load(Ordering::Relaxed) {
                    ui.small(format!(
                        "FPS: {:.0}  |  Chain: {} effects",
                        self.fps_counter.fps(),
                        self.build_effect_chain().len()
                    ));
                }
                ui.separator();

                // Source
                ui.heading("Source");
                ui.horizontal(|ui| {
                    ui.radio_value(
                        &mut self.source_mode,
                        SourceMode::TestColorBars,
                        "Color Bars",
                    );
                    ui.radio_value(&mut self.source_mode, SourceMode::TestGradient, "Gradient");
                });
                ui.horizontal(|ui| {
                    ui.label("Resolution:");
                    egui::ComboBox::from_id_source("res")
                        .selected_text(format!("{}x{}", self.frame_width, self.frame_height))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut (self.frame_width, self.frame_height),
                                (320, 240),
                                "320x240",
                            );
                            ui.selectable_value(
                                &mut (self.frame_width, self.frame_height),
                                (640, 480),
                                "640x480",
                            );
                            ui.selectable_value(
                                &mut (self.frame_width, self.frame_height),
                                (1280, 720),
                                "1280x720",
                            );
                        });
                });
                ui.separator();

                // Start / Stop
                let running = self.is_running.load(Ordering::Relaxed);
                if running {
                    if ui
                        .button(
                            egui::RichText::new("Stop")
                                .color(egui::Color32::RED)
                                .size(16.0),
                        )
                        .clicked()
                    {
                        self.stop();
                    }
                } else if ui
                    .button(
                        egui::RichText::new("Start")
                            .color(egui::Color32::GREEN)
                            .size(16.0),
                    )
                    .clicked()
                {
                    self.start();
                }
                ui.separator();

                // Effects
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Color Grading");
                    ui.checkbox(
                        &mut self.brightness_contrast_enabled,
                        "Brightness / Contrast",
                    );
                    if self.brightness_contrast_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.brightness, -100.0..=100.0)
                                .text("Brightness"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.contrast, 0.5..=3.0)
                                .text("Contrast")
                                .fixed_decimals(2),
                        );
                    }
                    ui.checkbox(&mut self.color_temp_enabled, "Color Temperature");
                    if self.color_temp_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.color_temperature, -100.0..=100.0)
                                .text("Temp"),
                        );
                    }
                    ui.checkbox(&mut self.vignette_enabled, "Vignette");
                    if self.vignette_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.vignette_strength, 0.0..=1.0)
                                .text("Strength"),
                        );
                    }
                    ui.checkbox(&mut self.grayscale_enabled, "Grayscale");
                    ui.checkbox(&mut self.sepia_enabled, "Sepia");
                    ui.checkbox(&mut self.invert_enabled, "Invert");
                    ui.checkbox(&mut self.posterize_enabled, "Posterize");
                    if self.posterize_enabled {
                        let mut l = self.posterize_levels as i32;
                        ui.add(egui::Slider::new(&mut l, 2..=16).text("Levels"));
                        self.posterize_levels = l as u8;
                    }
                    ui.separator();

                    ui.heading("Background");
                    ui.checkbox(&mut self.chroma_key_enabled, "Chroma Key (Green Screen)");
                    if self.chroma_key_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.chroma_hue, 0.0..=360.0).text("Target Hue"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.chroma_tolerance, 5.0..=90.0)
                                .text("Tolerance"),
                        );
                    }
                    ui.separator();

                    ui.heading("Creative");
                    ui.checkbox(&mut self.mirror_enabled, "Mirror");
                    if self.mirror_enabled {
                        ui.checkbox(&mut self.mirror_horizontal, "Horizontal (else Vertical)");
                    }
                    ui.checkbox(&mut self.blur_enabled, "Blur");
                    if self.blur_enabled {
                        let mut r = self.blur_radius as i32;
                        ui.add(egui::Slider::new(&mut r, 1..=10).text("Radius"));
                        self.blur_radius = r as u32;
                    }
                    ui.checkbox(&mut self.pixelate_enabled, "Pixelate");
                    if self.pixelate_enabled {
                        let mut s = self.pixelate_size as i32;
                        ui.add(egui::Slider::new(&mut s, 2..=32).text("Block size"));
                        self.pixelate_size = s as u32;
                    }
                    ui.checkbox(&mut self.night_vision_enabled, "Night Vision");
                    ui.checkbox(&mut self.vhs_retro_enabled, "VHS Retro");
                    ui.checkbox(&mut self.glitch_enabled, "Glitch");
                    if self.glitch_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.glitch_intensity, 0.0..=1.0)
                                .text("Intensity"),
                        );
                    }
                    ui.checkbox(&mut self.cartoon_enabled, "Cartoon");
                    if self.cartoon_enabled {
                        let mut l = self.cartoon_levels as i32;
                        ui.add(egui::Slider::new(&mut l, 2..=16).text("Levels"));
                        self.cartoon_levels = l as u8;
                        let mut t = self.cartoon_threshold as i32;
                        ui.add(egui::Slider::new(&mut t, 10..=200).text("Edge Threshold"));
                        self.cartoon_threshold = t as u8;
                    }
                    ui.checkbox(&mut self.edge_detect_enabled, "Edge Detection");
                    if self.edge_detect_enabled {
                        let mut t = self.edge_threshold as i32;
                        ui.add(egui::Slider::new(&mut t, 5..=200).text("Threshold"));
                        self.edge_threshold = t as u8;
                    }
                    ui.checkbox(&mut self.ascii_art_enabled, "ASCII Art");
                    if self.ascii_art_enabled {
                        let mut s = self.ascii_cell_size as i32;
                        ui.add(egui::Slider::new(&mut s, 4..=16).text("Cell size"));
                        self.ascii_cell_size = s as u32;
                    }
                    ui.separator();

                    ui.heading("Overlay");
                    ui.checkbox(&mut self.matrix_rain_enabled, "Matrix Rain");
                    if self.matrix_rain_enabled {
                        let mut s = self.matrix_cell_size as i32;
                        ui.add(egui::Slider::new(&mut s, 4..=16).text("Cell size"));
                        self.matrix_cell_size = s as u32;
                        ui.checkbox(&mut self.matrix_overlay, "Overlay mode (blend with video)");
                        if self.matrix_overlay {
                            ui.add(
                                egui::Slider::new(&mut self.matrix_opacity, 0.1..=1.0)
                                    .text("Opacity"),
                            );
                        }
                    }
                    ui.checkbox(&mut self.watermark_enabled, "Watermark");
                    if self.watermark_enabled {
                        ui.text_edit_singleline(&mut self.watermark_text);
                    }
                });
            });

        // ── Central panel: preview ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.preview_texture {
                let available = ui.available_size();
                let aspect = self.frame_width as f32 / self.frame_height.max(1) as f32;
                let (w, h) = if available.x / available.y > aspect {
                    (available.y * aspect, available.y)
                } else {
                    (available.x, available.x / aspect)
                };
                ui.centered_and_justified(|ui| {
                    ui.image(egui::load::SizedTexture::new(tex.id(), egui::vec2(w, h)));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("No preview — click Start")
                            .size(20.0)
                            .color(egui::Color32::GRAY),
                    );
                });
            }
        });

        // Request repaint for animation
        if self.is_running.load(Ordering::Relaxed) {
            ctx.request_repaint();
        }
    }
}

// ============================================================================
// Entry point
// ============================================================================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ai_virtual_cam")
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ai_virtual_cam",
        options,
        Box::new(|_cc| Box::new(VirtualCamApp::new())),
    )
}
