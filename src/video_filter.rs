//! Video effect pipeline — composable real-time video processing.
//!
//! Analogous to [`crate::audio_filter`] but for video frames. Each effect
//! implements the [`VideoEffect`] trait and can be chained via
//! [`VideoEffectChain`]. Effects operate on RGBA pixel buffers.

use std::time::Instant;

// ============================================================================
// Core types
// ============================================================================

/// Color space of a video frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Rgba,
}

/// A single video frame with RGBA pixel data.
#[derive(Clone)]
pub struct VideoFrame {
    /// Raw RGBA pixel data (4 bytes per pixel, row-major).
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,
    pub frame_number: u64,
    pub timestamp: Instant,
}

impl VideoFrame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0u8; (width * height * 4) as usize],
            width,
            height,
            color_space: ColorSpace::Rgba,
            frame_number: 0,
            timestamp: Instant::now(),
        }
    }

    pub fn from_rgba(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            color_space: ColorSpace::Rgba,
            frame_number: 0,
            timestamp: Instant::now(),
        }
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Returns (r, g, b, a) at pixel (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> (u8, u8, u8, u8) {
        let idx = ((y * self.width + x) * 4) as usize;
        (
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        )
    }

    /// Sets (r, g, b, a) at pixel (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
        let idx = ((y * self.width + x) * 4) as usize;
        self.data[idx] = r;
        self.data[idx + 1] = g;
        self.data[idx + 2] = b;
        self.data[idx + 3] = a;
    }
}

/// Category of a video effect (for GUI grouping).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoEffectCategory {
    InputProcessing,
    Background,
    Overlay,
    ColorGrading,
    Creative,
    FaceFilter,
    Output,
}

/// A video effect that processes frames in place.
pub trait VideoEffect: Send {
    fn name(&self) -> &str;
    fn process(&mut self, frame: &mut VideoFrame);
    fn is_enabled(&self) -> bool;
    fn set_enabled(&mut self, enabled: bool);
    fn category(&self) -> VideoEffectCategory;
    fn estimated_latency_us(&self) -> u64 {
        100
    }
}

/// Composable chain of video effects applied in order.
pub struct VideoEffectChain {
    effects: Vec<Box<dyn VideoEffect>>,
}

impl VideoEffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    pub fn add_effect(&mut self, effect: Box<dyn VideoEffect>) {
        self.effects.push(effect);
    }

    pub fn process_frame(&mut self, frame: &mut VideoFrame) {
        for effect in &mut self.effects {
            if effect.is_enabled() {
                effect.process(frame);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    pub fn total_latency_us(&self) -> u64 {
        self.effects
            .iter()
            .filter(|e| e.is_enabled())
            .map(|e| e.estimated_latency_us())
            .sum()
    }

    pub fn effect_names(&self) -> Vec<&str> {
        self.effects.iter().map(|e| e.name()).collect()
    }
}

impl Default for VideoEffectChain {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Effect: Mirror
// ============================================================================

pub struct MirrorEffect {
    enabled: bool,
    pub horizontal: bool,
}

impl MirrorEffect {
    pub fn horizontal() -> Self {
        Self {
            enabled: true,
            horizontal: true,
        }
    }
    pub fn vertical() -> Self {
        Self {
            enabled: true,
            horizontal: false,
        }
    }
}

impl VideoEffect for MirrorEffect {
    fn name(&self) -> &str {
        "Mirror"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::InputProcessing
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let w = frame.width as usize;
        let h = frame.height as usize;
        if self.horizontal {
            for y in 0..h {
                for x in 0..w / 2 {
                    let left = (y * w + x) * 4;
                    let right = (y * w + (w - 1 - x)) * 4;
                    for c in 0..4 {
                        frame.data.swap(left + c, right + c);
                    }
                }
            }
        } else {
            let row_bytes = w * 4;
            let mut row_buf = vec![0u8; row_bytes];
            for y in 0..h / 2 {
                let top = y * row_bytes;
                let bot = (h - 1 - y) * row_bytes;
                row_buf.copy_from_slice(&frame.data[top..top + row_bytes]);
                frame.data.copy_within(bot..bot + row_bytes, top);
                frame.data[bot..bot + row_bytes].copy_from_slice(&row_buf);
            }
        }
    }
}

// ============================================================================
// Effect: Pixelate
// ============================================================================

pub struct PixelateEffect {
    enabled: bool,
    /// Block size in pixels. Higher = more pixelated.
    pub block_size: u32,
}

impl PixelateEffect {
    pub fn new(block_size: u32) -> Self {
        Self {
            enabled: true,
            block_size: block_size.max(2),
        }
    }
}

impl VideoEffect for PixelateEffect {
    fn name(&self) -> &str {
        "Pixelate"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        200
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let bs = self.block_size as usize;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let mut by = 0;
        while by < h {
            let bh = bs.min(h - by);
            let mut bx = 0;
            while bx < w {
                let bw = bs.min(w - bx);
                // Average the block
                let mut sr = 0u32;
                let mut sg = 0u32;
                let mut sb = 0u32;
                let count = (bw * bh) as u32;
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + bx + dx) * 4;
                        sr += frame.data[idx] as u32;
                        sg += frame.data[idx + 1] as u32;
                        sb += frame.data[idx + 2] as u32;
                    }
                }
                let ar = (sr / count) as u8;
                let ag = (sg / count) as u8;
                let ab = (sb / count) as u8;
                // Fill the block with average
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + bx + dx) * 4;
                        frame.data[idx] = ar;
                        frame.data[idx + 1] = ag;
                        frame.data[idx + 2] = ab;
                    }
                }
                bx += bs;
            }
            by += bs;
        }
    }
}

// ============================================================================
// Effect: Color Temperature
// ============================================================================

pub struct ColorTemperatureEffect {
    enabled: bool,
    /// −100 (cold/blue) to +100 (warm/orange).
    pub temperature: f32,
}

impl ColorTemperatureEffect {
    pub fn new(temperature: f32) -> Self {
        Self {
            enabled: true,
            temperature: temperature.clamp(-100.0, 100.0),
        }
    }
}

impl VideoEffect for ColorTemperatureEffect {
    fn name(&self) -> &str {
        "Color Temperature"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let t = self.temperature / 100.0;
        let r_shift = (t * 30.0) as i16;
        let b_shift = (-t * 30.0) as i16;
        for px in frame.data.chunks_exact_mut(4) {
            px[0] = (px[0] as i16 + r_shift).clamp(0, 255) as u8;
            px[2] = (px[2] as i16 + b_shift).clamp(0, 255) as u8;
        }
    }
}

// ============================================================================
// Effect: Vignette
// ============================================================================

pub struct VignetteEffect {
    enabled: bool,
    /// Strength 0.0 (none) to 1.0 (strong darkening at edges).
    pub strength: f32,
}

impl VignetteEffect {
    pub fn new(strength: f32) -> Self {
        Self {
            enabled: true,
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

impl VideoEffect for VignetteEffect {
    fn name(&self) -> &str {
        "Vignette"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let cx = frame.width as f32 / 2.0;
        let cy = frame.height as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();
        let w = frame.width as usize;
        for y in 0..frame.height as usize {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / max_r;
                let factor = 1.0 - (dist * dist * self.strength);
                let idx = (y * w + x) * 4;
                frame.data[idx] = (frame.data[idx] as f32 * factor) as u8;
                frame.data[idx + 1] = (frame.data[idx + 1] as f32 * factor) as u8;
                frame.data[idx + 2] = (frame.data[idx + 2] as f32 * factor) as u8;
            }
        }
    }
}

// ============================================================================
// Effect: Night Vision
// ============================================================================

pub struct NightVisionEffect {
    enabled: bool,
    frame_counter: u64,
}

impl NightVisionEffect {
    pub fn new() -> Self {
        Self {
            enabled: true,
            frame_counter: 0,
        }
    }
}

impl VideoEffect for NightVisionEffect {
    fn name(&self) -> &str {
        "Night Vision"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let w = frame.width as usize;
        for y in 0..frame.height as usize {
            // Scanline darken on even lines
            let scanline_dim = if y % 2 == 0 { 0.85f32 } else { 1.0 };
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let r = frame.data[idx] as f32;
                let g = frame.data[idx + 1] as f32;
                let b = frame.data[idx + 2] as f32;
                // Convert to green-tinted monochrome
                let lum = (0.299 * r + 0.587 * g + 0.114 * b) * scanline_dim;
                // Simple pseudo-noise based on position + frame
                let noise =
                    ((x.wrapping_mul(73) ^ y.wrapping_mul(157) ^ self.frame_counter as usize) % 20)
                        as f32
                        - 10.0;
                let green = (lum + noise).clamp(0.0, 255.0) as u8;
                frame.data[idx] = green / 4;
                frame.data[idx + 1] = green;
                frame.data[idx + 2] = green / 6;
            }
        }
    }
}

// ============================================================================
// Effect: VHS Retro
// ============================================================================

pub struct VhsRetroEffect {
    enabled: bool,
    frame_counter: u64,
}

impl VhsRetroEffect {
    pub fn new() -> Self {
        Self {
            enabled: true,
            frame_counter: 0,
        }
    }
}

impl VideoEffect for VhsRetroEffect {
    fn name(&self) -> &str {
        "VHS Retro"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        300
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let w = frame.width as usize;
        let h = frame.height as usize;
        // 1. Chromatic aberration: shift R channel right, B channel left
        let shift = 3usize;
        for y in 0..h {
            let row_start = y * w * 4;
            // Shift R right
            for x in (shift..w).rev() {
                let dst = row_start + x * 4;
                let src = row_start + (x - shift) * 4;
                frame.data[dst] = frame.data[src]; // R only
            }
            // Shift B left
            for x in 0..w.saturating_sub(shift) {
                let dst = row_start + x * 4 + 2;
                let src = row_start + (x + shift) * 4 + 2;
                frame.data[dst] = frame.data[src]; // B only
            }
        }
        // 2. Scanlines
        for y in (0..h).step_by(3) {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                frame.data[idx] = (frame.data[idx] as f32 * 0.7) as u8;
                frame.data[idx + 1] = (frame.data[idx + 1] as f32 * 0.7) as u8;
                frame.data[idx + 2] = (frame.data[idx + 2] as f32 * 0.7) as u8;
            }
        }
        // 3. Random horizontal jitter on a few scanlines
        let jitter_line = (self.frame_counter as usize * 97 + 13) % h;
        let jitter_amount = ((self.frame_counter as usize * 31) % 7) as i32 - 3;
        if jitter_amount != 0 {
            let row = jitter_line * w * 4;
            if jitter_amount > 0 {
                let shift_px = jitter_amount as usize;
                frame
                    .data
                    .copy_within(row..row + (w - shift_px) * 4, row + shift_px * 4);
            }
        }
    }
}

// ============================================================================
// Effect: Glitch (RGB shift + scanline displacement)
// ============================================================================

pub struct GlitchEffect {
    enabled: bool,
    /// Intensity 0.0 (subtle) to 1.0 (heavy).
    pub intensity: f32,
    frame_counter: u64,
}

impl GlitchEffect {
    pub fn new(intensity: f32) -> Self {
        Self {
            enabled: true,
            intensity: intensity.clamp(0.0, 1.0),
            frame_counter: 0,
        }
    }
}

impl VideoEffect for GlitchEffect {
    fn name(&self) -> &str {
        "Glitch"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let w = frame.width as usize;
        let h = frame.height as usize;
        let max_shift = (self.intensity * 20.0) as usize + 1;
        // Displace N random scanlines
        let n_lines = (self.intensity * 15.0) as usize + 1;
        for i in 0..n_lines {
            let y = (self.frame_counter as usize * 41 + i * 137) % h;
            let shift = ((self.frame_counter as usize * 73 + i * 211) % (max_shift * 2)) as i32
                - max_shift as i32;
            if shift > 0 {
                let s = shift as usize;
                let row = y * w * 4;
                frame
                    .data
                    .copy_within(row..row + (w.saturating_sub(s)) * 4, row + s * 4);
            }
        }
        // RGB channel shift (subtle)
        let rgb_shift = (self.intensity * 5.0) as usize + 1;
        for y in 0..h {
            for x in rgb_shift..w {
                let idx = (y * w + x) * 4;
                let src = (y * w + x - rgb_shift) * 4;
                frame.data[idx] = frame.data[src]; // R from left
            }
        }
    }
}

// ============================================================================
// Effect: Cartoon (edge detection + posterize)
// ============================================================================

pub struct CartoonEffect {
    enabled: bool,
    /// Number of color levels per channel (2-16).
    pub levels: u8,
    /// Edge threshold (0-255).
    pub edge_threshold: u8,
}

impl CartoonEffect {
    pub fn new(levels: u8, edge_threshold: u8) -> Self {
        Self {
            enabled: true,
            levels: levels.clamp(2, 16),
            edge_threshold,
        }
    }
}

impl VideoEffect for CartoonEffect {
    fn name(&self) -> &str {
        "Cartoon"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        500
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let step = 256.0 / self.levels as f32;
        // Posterize + Sobel edge detection (approximated)
        let orig = frame.data.clone();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let idx = (y * w + x) * 4;
                // Luminance at 3x3 neighbors for edge detection
                let lum = |px: usize, py: usize| -> i32 {
                    let i = (py * w + px) * 4;
                    (orig[i] as i32 + orig[i + 1] as i32 + orig[i + 2] as i32) / 3
                };
                // Simplified Sobel
                let gx = -lum(x - 1, y - 1) + lum(x + 1, y - 1) - 2 * lum(x - 1, y)
                    + 2 * lum(x + 1, y)
                    - lum(x - 1, y + 1)
                    + lum(x + 1, y + 1);
                let gy = -lum(x - 1, y - 1) - 2 * lum(x, y - 1) - lum(x + 1, y - 1)
                    + lum(x - 1, y + 1)
                    + 2 * lum(x, y + 1)
                    + lum(x + 1, y + 1);
                let edge = ((gx.abs() + gy.abs()) / 2).min(255) as u8;
                if edge > self.edge_threshold {
                    // Edge pixel: black
                    frame.data[idx] = 0;
                    frame.data[idx + 1] = 0;
                    frame.data[idx + 2] = 0;
                } else {
                    // Posterize
                    frame.data[idx] =
                        ((frame.data[idx] as f32 / step).floor() * step).min(255.0) as u8;
                    frame.data[idx + 1] =
                        ((frame.data[idx + 1] as f32 / step).floor() * step).min(255.0) as u8;
                    frame.data[idx + 2] =
                        ((frame.data[idx + 2] as f32 / step).floor() * step).min(255.0) as u8;
                }
            }
        }
    }
}

// ============================================================================
// Effect: Chroma Key (green screen removal)
// ============================================================================

pub struct ChromaKeyEffect {
    enabled: bool,
    /// Target hue (0-360). Green ≈ 120.
    pub target_hue: f32,
    /// Hue tolerance in degrees.
    pub hue_tolerance: f32,
    /// Saturation minimum to be considered "keyed" (0.0-1.0).
    pub min_saturation: f32,
    /// Replacement color (r, g, b) or transparent.
    pub replacement: (u8, u8, u8, u8),
}

impl ChromaKeyEffect {
    /// Green screen preset.
    pub fn green_screen() -> Self {
        Self {
            enabled: true,
            target_hue: 120.0,
            hue_tolerance: 40.0,
            min_saturation: 0.25,
            replacement: (0, 0, 0, 0), // transparent
        }
    }
}

impl VideoEffect for ChromaKeyEffect {
    fn name(&self) -> &str {
        "Chroma Key"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Background
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        for px in frame.data.chunks_exact_mut(4) {
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;
            let sat = if max > 0.0 { delta / max } else { 0.0 };
            if sat < self.min_saturation {
                continue;
            }
            let hue = if delta < 1e-6 {
                0.0
            } else if (max - r).abs() < 1e-6 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (max - g).abs() < 1e-6 {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let hue = if hue < 0.0 { hue + 360.0 } else { hue };
            let diff = (hue - self.target_hue)
                .abs()
                .min(360.0 - (hue - self.target_hue).abs());
            if diff <= self.hue_tolerance {
                px[0] = self.replacement.0;
                px[1] = self.replacement.1;
                px[2] = self.replacement.2;
                px[3] = self.replacement.3;
            }
        }
    }
}

// ============================================================================
// Effect: Matrix Rain
// ============================================================================

/// Procedural Matrix rain animation. Columns of falling green characters.
pub struct MatrixRainEffect {
    enabled: bool,
    /// Width/height of each character cell in pixels.
    pub cell_size: u32,
    /// Fall speed multiplier.
    pub speed: f32,
    /// Columns state: (current_y_position, speed, trail_length).
    columns: Vec<(f32, f32, usize)>,
    /// Character grid: stores brightness 0-255 per cell.
    grid: Vec<u8>,
    grid_cols: usize,
    grid_rows: usize,
    frame_counter: u64,
    /// Whether to render as overlay (blended) or full replacement.
    pub overlay_mode: bool,
    /// Overlay opacity when in overlay mode (0.0 - 1.0).
    pub overlay_opacity: f32,
}

impl MatrixRainEffect {
    pub fn new(cell_size: u32) -> Self {
        Self {
            enabled: true,
            cell_size: cell_size.max(4),
            speed: 1.0,
            columns: Vec::new(),
            grid: Vec::new(),
            grid_cols: 0,
            grid_rows: 0,
            frame_counter: 0,
            overlay_mode: false,
            overlay_opacity: 0.5,
        }
    }

    fn init_grid(&mut self, width: u32, height: u32) {
        let cols = (width / self.cell_size) as usize;
        let rows = (height / self.cell_size) as usize;
        if cols == self.grid_cols && rows == self.grid_rows {
            return;
        }
        self.grid_cols = cols;
        self.grid_rows = rows;
        self.grid = vec![0u8; cols * rows];
        self.columns.clear();
        for i in 0..cols {
            let speed = 0.3 + (i as f32 * 0.73 % 1.0) * 0.7;
            let trail = 5 + (i * 37) % 20;
            let start = -((i * 13 % 40) as f32);
            self.columns.push((start, speed * self.speed, trail));
        }
    }

    fn advance(&mut self) {
        // Fade all cells
        for cell in &mut self.grid {
            *cell = cell.saturating_sub(12);
        }
        // Advance each column
        for (col_idx, (y_pos, speed, trail)) in self.columns.iter_mut().enumerate() {
            *y_pos += *speed;
            let iy = *y_pos as isize;
            // Light up the head
            if iy >= 0 && (iy as usize) < self.grid_rows {
                self.grid[iy as usize * self.grid_cols + col_idx] = 255;
            }
            // When head goes off-screen, reset with random offset
            if iy > (self.grid_rows + *trail) as isize {
                *y_pos = -((*trail as f32)
                    + (self.frame_counter as f32 * 0.1 + col_idx as f32)
                        .sin()
                        .abs()
                        * 20.0);
            }
        }
    }
}

impl VideoEffect for MatrixRainEffect {
    fn name(&self) -> &str {
        "Matrix Rain"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Overlay
    }
    fn estimated_latency_us(&self) -> u64 {
        400
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.init_grid(frame.width, frame.height);
        self.advance();
        let cs = self.cell_size as usize;
        let w = frame.width as usize;
        for row in 0..self.grid_rows {
            for col in 0..self.grid_cols {
                let brightness = self.grid[row * self.grid_cols + col];
                if brightness == 0 && self.overlay_mode {
                    continue;
                }
                // Green with variable brightness; head is white-ish
                let g = brightness;
                let r = if brightness > 240 {
                    200
                } else {
                    brightness / 4
                };
                let b = brightness / 8;
                // Fill cell rectangle
                for dy in 0..cs.min(frame.height as usize - row * cs) {
                    for dx in 0..cs.min(w - col * cs) {
                        let px = col * cs + dx;
                        let py = row * cs + dy;
                        let idx = (py * w + px) * 4;
                        if idx + 3 >= frame.data.len() {
                            continue;
                        }
                        if self.overlay_mode {
                            let alpha = self.overlay_opacity * (brightness as f32 / 255.0);
                            frame.data[idx] =
                                (frame.data[idx] as f32 * (1.0 - alpha) + r as f32 * alpha) as u8;
                            frame.data[idx + 1] = (frame.data[idx + 1] as f32 * (1.0 - alpha)
                                + g as f32 * alpha)
                                as u8;
                            frame.data[idx + 2] = (frame.data[idx + 2] as f32 * (1.0 - alpha)
                                + b as f32 * alpha)
                                as u8;
                        } else {
                            frame.data[idx] = r;
                            frame.data[idx + 1] = g;
                            frame.data[idx + 2] = b;
                            frame.data[idx + 3] = 255;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Effect: Watermark (text overlay)
// ============================================================================

pub struct WatermarkEffect {
    enabled: bool,
    pub text: String,
    /// Position as fraction of frame (0.0 = left/top, 1.0 = right/bottom).
    pub x_frac: f32,
    pub y_frac: f32,
    pub color: (u8, u8, u8),
    pub opacity: f32,
}

impl WatermarkEffect {
    pub fn new(text: &str) -> Self {
        Self {
            enabled: true,
            text: text.to_string(),
            x_frac: 0.02,
            y_frac: 0.95,
            color: (255, 255, 255),
            opacity: 0.6,
        }
    }
}

impl VideoEffect for WatermarkEffect {
    fn name(&self) -> &str {
        "Watermark"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Output
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        // Simple 5x7 pixel font renderer for ASCII watermark
        let start_x = (frame.width as f32 * self.x_frac) as usize;
        let start_y = (frame.height as f32 * self.y_frac) as usize;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let char_w = 6;
        let char_h = 8;
        for (ci, ch) in self.text.chars().enumerate() {
            let cx = start_x + ci * char_w;
            if cx + char_w >= w || start_y + char_h >= h {
                break;
            }
            // Use a very simple bitmap: for any printable char, draw a filled rectangle
            // (proper font rendering would require ab_glyph; this is a functional placeholder)
            let pattern = simple_char_pattern(ch);
            for row in 0..7 {
                let bits = pattern[row];
                for col in 0..5 {
                    if (bits >> (4 - col)) & 1 != 0 {
                        let px = cx + col;
                        let py = start_y + row;
                        let idx = (py * w + px) * 4;
                        if idx + 3 < frame.data.len() {
                            let a = self.opacity;
                            frame.data[idx] = (frame.data[idx] as f32 * (1.0 - a)
                                + self.color.0 as f32 * a)
                                as u8;
                            frame.data[idx + 1] = (frame.data[idx + 1] as f32 * (1.0 - a)
                                + self.color.1 as f32 * a)
                                as u8;
                            frame.data[idx + 2] = (frame.data[idx + 2] as f32 * (1.0 - a)
                                + self.color.2 as f32 * a)
                                as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Minimal 5x7 bitmap font for common ASCII chars. Each character is 7 rows of 5-bit patterns.
fn simple_char_pattern(ch: char) -> [u8; 7] {
    match ch {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        '0'..='9' => {
            let d = ch as u8 - b'0';
            match d {
                0 => [
                    0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
                ],
                1 => [
                    0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
                ],
                2 => [
                    0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
                ],
                3 => [
                    0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
                ],
                _ => [
                    0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
                ],
            }
        }
        _ => [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ], // box for unknown
    }
}

// ============================================================================
// Effect: ASCII Art
// ============================================================================

pub struct AsciiArtEffect {
    enabled: bool,
    pub cell_size: u32,
    pub colored: bool,
}

impl AsciiArtEffect {
    pub fn new(cell_size: u32) -> Self {
        Self {
            enabled: true,
            cell_size: cell_size.max(4),
            colored: true,
        }
    }
}

impl VideoEffect for AsciiArtEffect {
    fn name(&self) -> &str {
        "ASCII Art"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        400
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let cs = self.cell_size as usize;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let chars = b" .:-=+*#%@";
        let orig = frame.data.clone();
        // Black background
        frame.data.fill(0);
        for px in frame.data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let mut by = 0;
        while by < h {
            let bh = cs.min(h - by);
            let mut bx = 0;
            while bx < w {
                let bw = cs.min(w - bx);
                // Average luminance + color of block
                let mut sr = 0u32;
                let mut sg = 0u32;
                let mut sb = 0u32;
                let mut slum = 0u32;
                let count = (bw * bh) as u32;
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + bx + dx) * 4;
                        let r = orig[idx] as u32;
                        let g = orig[idx + 1] as u32;
                        let b = orig[idx + 2] as u32;
                        sr += r;
                        sg += g;
                        sb += b;
                        slum += (r * 77 + g * 150 + b * 29) >> 8;
                    }
                }
                let avg_lum = (slum / count) as usize;
                let char_idx = (avg_lum * (chars.len() - 1)) / 255;
                let _ch = chars[char_idx.min(chars.len() - 1)];
                // Fill block with luminance-mapped color (simulate ASCII density)
                let density = char_idx as f32 / (chars.len() - 1) as f32;
                let (cr, cg, cb) = if self.colored {
                    (
                        ((sr / count) as f32 * density) as u8,
                        ((sg / count) as f32 * density) as u8,
                        ((sb / count) as f32 * density) as u8,
                    )
                } else {
                    let v = (avg_lum as f32 * density) as u8;
                    (v, v, v)
                };
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + bx + dx) * 4;
                        frame.data[idx] = cr;
                        frame.data[idx + 1] = cg;
                        frame.data[idx + 2] = cb;
                    }
                }
                bx += cs;
            }
            by += cs;
        }
    }
}

// ============================================================================
// Effect: Brightness / Contrast
// ============================================================================

pub struct BrightnessContrastEffect {
    enabled: bool,
    /// -100 to +100.
    pub brightness: f32,
    /// 0.5 (low contrast) to 2.0 (high contrast).
    pub contrast: f32,
}

impl BrightnessContrastEffect {
    pub fn new(brightness: f32, contrast: f32) -> Self {
        Self {
            enabled: true,
            brightness: brightness.clamp(-100.0, 100.0),
            contrast: contrast.clamp(0.5, 3.0),
        }
    }
}

impl VideoEffect for BrightnessContrastEffect {
    fn name(&self) -> &str {
        "Brightness/Contrast"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let b = self.brightness;
        let c = self.contrast;
        for px in frame.data.chunks_exact_mut(4) {
            for i in 0..3 {
                let v = px[i] as f32;
                let v = ((v - 128.0) * c + 128.0 + b).clamp(0.0, 255.0);
                px[i] = v as u8;
            }
        }
    }
}

// ============================================================================
// Effect: Grayscale
// ============================================================================

pub struct GrayscaleEffect {
    enabled: bool,
}

impl GrayscaleEffect {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl VideoEffect for GrayscaleEffect {
    fn name(&self) -> &str {
        "Grayscale"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        for px in frame.data.chunks_exact_mut(4) {
            let lum = (px[0] as f32 * 0.299 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114) as u8;
            px[0] = lum;
            px[1] = lum;
            px[2] = lum;
        }
    }
}

// ============================================================================
// Effect: Sepia
// ============================================================================

pub struct SepiaEffect {
    enabled: bool,
}

impl SepiaEffect {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl VideoEffect for SepiaEffect {
    fn name(&self) -> &str {
        "Sepia"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        for px in frame.data.chunks_exact_mut(4) {
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;
            px[0] = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0) as u8;
            px[1] = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0) as u8;
            px[2] = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0) as u8;
        }
    }
}

// ============================================================================
// Effect: Invert Colors
// ============================================================================

pub struct InvertEffect {
    enabled: bool,
}

impl InvertEffect {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl VideoEffect for InvertEffect {
    fn name(&self) -> &str {
        "Invert"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        for px in frame.data.chunks_exact_mut(4) {
            px[0] = 255 - px[0];
            px[1] = 255 - px[1];
            px[2] = 255 - px[2];
        }
    }
}

// ============================================================================
// Effect: Posterize
// ============================================================================

pub struct PosterizeEffect {
    enabled: bool,
    pub levels: u8,
}

impl PosterizeEffect {
    pub fn new(levels: u8) -> Self {
        Self {
            enabled: true,
            levels: levels.clamp(2, 32),
        }
    }
}

impl VideoEffect for PosterizeEffect {
    fn name(&self) -> &str {
        "Posterize"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::ColorGrading
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let step = 256.0 / self.levels as f32;
        for px in frame.data.chunks_exact_mut(4) {
            for i in 0..3 {
                px[i] = ((px[i] as f32 / step).floor() * step).min(255.0) as u8;
            }
        }
    }
}

// ============================================================================
// Effect: Blur (box blur, simple)
// ============================================================================

pub struct BlurEffect {
    enabled: bool,
    /// Kernel radius (1-10). Higher = more blur.
    pub radius: u32,
}

impl BlurEffect {
    pub fn new(radius: u32) -> Self {
        Self {
            enabled: true,
            radius: radius.clamp(1, 10),
        }
    }
}

impl VideoEffect for BlurEffect {
    fn name(&self) -> &str {
        "Blur"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn estimated_latency_us(&self) -> u64 {
        self.radius as u64 * 200
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let w = frame.width as i32;
        let h = frame.height as i32;
        let r = self.radius as i32;
        let orig = frame.data.clone();
        for y in 0..h {
            for x in 0..w {
                let mut sr = 0u32;
                let mut sg = 0u32;
                let mut sb = 0u32;
                let mut count = 0u32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < w && ny >= 0 && ny < h {
                            let idx = ((ny * w + nx) * 4) as usize;
                            sr += orig[idx] as u32;
                            sg += orig[idx + 1] as u32;
                            sb += orig[idx + 2] as u32;
                            count += 1;
                        }
                    }
                }
                let idx = ((y * w + x) * 4) as usize;
                frame.data[idx] = (sr / count) as u8;
                frame.data[idx + 1] = (sg / count) as u8;
                frame.data[idx + 2] = (sb / count) as u8;
            }
        }
    }
}

// ============================================================================
// Effect: Edge Detection
// ============================================================================

pub struct EdgeDetectionEffect {
    enabled: bool,
    pub threshold: u8,
    pub color_edges: bool,
}

impl EdgeDetectionEffect {
    pub fn new(threshold: u8) -> Self {
        Self {
            enabled: true,
            threshold,
            color_edges: false,
        }
    }
}

impl VideoEffect for EdgeDetectionEffect {
    fn name(&self) -> &str {
        "Edge Detection"
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
    fn category(&self) -> VideoEffectCategory {
        VideoEffectCategory::Creative
    }
    fn process(&mut self, frame: &mut VideoFrame) {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let orig = frame.data.clone();
        frame.data.fill(0);
        for px in frame.data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let lum = |px: usize, py: usize| -> i32 {
                    let i = (py * w + px) * 4;
                    (orig[i] as i32 + orig[i + 1] as i32 + orig[i + 2] as i32) / 3
                };
                let gx = (lum(x + 1, y) - lum(x - 1, y)).abs();
                let gy = (lum(x, y + 1) - lum(x, y - 1)).abs();
                let edge = ((gx + gy) / 2).min(255) as u8;
                if edge > self.threshold {
                    let idx = (y * w + x) * 4;
                    if self.color_edges {
                        frame.data[idx] = orig[idx];
                        frame.data[idx + 1] = orig[idx + 1];
                        frame.data[idx + 2] = orig[idx + 2];
                    } else {
                        frame.data[idx] = edge;
                        frame.data[idx + 1] = edge;
                        frame.data[idx + 2] = edge;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Test pattern generator (for testing + fallback when no webcam)
// ============================================================================

/// Generates synthetic test frames for testing and development.
pub struct TestPatternGenerator {
    frame_counter: u64,
}

impl TestPatternGenerator {
    pub fn new() -> Self {
        Self { frame_counter: 0 }
    }

    /// Color bars test pattern (SMPTE-like).
    pub fn color_bars(&mut self, width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height);
        frame.frame_number = self.frame_counter;
        self.frame_counter += 1;
        let colors: [(u8, u8, u8); 8] = [
            (255, 255, 255),
            (255, 255, 0),
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 255),
            (255, 0, 0),
            (0, 0, 255),
            (0, 0, 0),
        ];
        let bar_w = width / 8;
        for y in 0..height {
            for x in 0..width {
                let bar = (x / bar_w.max(1)) as usize % 8;
                let (r, g, b) = colors[bar];
                frame.set_pixel(x, y, r, g, b, 255);
            }
        }
        frame
    }

    /// Gradient test pattern with moving element.
    pub fn gradient(&mut self, width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height);
        frame.frame_number = self.frame_counter;
        self.frame_counter += 1;
        for y in 0..height {
            for x in 0..width {
                let r = (x * 255 / width.max(1)) as u8;
                let g = (y * 255 / height.max(1)) as u8;
                let b = 128;
                frame.set_pixel(x, y, r, g, b, 255);
            }
        }
        // Moving circle
        let cx = (width as f32 / 2.0
            + (self.frame_counter as f32 * 0.05).sin() * width as f32 / 4.0)
            as i32;
        let cy = (height as f32 / 2.0
            + (self.frame_counter as f32 * 0.03).cos() * height as f32 / 4.0)
            as i32;
        let radius = 20i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        frame.set_pixel(px as u32, py as u32, 255, 0, 0, 255);
                    }
                }
            }
        }
        frame
    }

    /// Solid color frame.
    pub fn solid(width: u32, height: u32, r: u8, g: u8, b: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height);
        for px in frame.data.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = 255;
        }
        frame
    }

    /// Checkerboard pattern.
    pub fn checkerboard(width: u32, height: u32, cell_size: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height);
        let cs = cell_size.max(1);
        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / cs) + (y / cs)) % 2 == 0;
                let v = if is_white { 255 } else { 0 };
                frame.set_pixel(x, y, v, v, v, 255);
            }
        }
        frame
    }
}

impl Default for TestPatternGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame() -> VideoFrame {
        TestPatternGenerator::new().color_bars(64, 48)
    }

    fn gradient_frame() -> VideoFrame {
        TestPatternGenerator::new().gradient(64, 48)
    }

    #[test]
    fn video_frame_new_has_correct_size() {
        let f = VideoFrame::new(100, 50);
        assert_eq!(f.data.len(), 100 * 50 * 4);
        assert_eq!(f.pixel_count(), 5000);
    }

    #[test]
    fn video_frame_get_set_pixel() {
        let mut f = VideoFrame::new(10, 10);
        f.set_pixel(5, 3, 100, 200, 50, 255);
        assert_eq!(f.get_pixel(5, 3), (100, 200, 50, 255));
    }

    #[test]
    fn effect_chain_processes_in_order() {
        let mut chain = VideoEffectChain::new();
        chain.add_effect(Box::new(GrayscaleEffect::new()));
        chain.add_effect(Box::new(InvertEffect::new()));
        assert_eq!(chain.len(), 2);
        let mut f = test_frame();
        chain.process_frame(&mut f);
        // After grayscale + invert, white bar should become black
        let (r, g, b, _) = f.get_pixel(0, 0);
        assert!(r == g && g == b, "should be grayscale: r={r} g={g} b={b}");
    }

    #[test]
    fn effect_chain_skips_disabled() {
        let mut chain = VideoEffectChain::new();
        let mut inv = InvertEffect::new();
        inv.set_enabled(false);
        chain.add_effect(Box::new(inv));
        let mut f = test_frame();
        let orig = f.data.clone();
        chain.process_frame(&mut f);
        assert_eq!(f.data, orig);
    }

    #[test]
    fn mirror_horizontal_swaps_pixels() {
        let mut f = VideoFrame::new(10, 1);
        f.set_pixel(0, 0, 255, 0, 0, 255);
        f.set_pixel(9, 0, 0, 0, 255, 255);
        let mut m = MirrorEffect::horizontal();
        m.process(&mut f);
        assert_eq!(f.get_pixel(0, 0), (0, 0, 255, 255));
        assert_eq!(f.get_pixel(9, 0), (255, 0, 0, 255));
    }

    #[test]
    fn pixelate_produces_uniform_blocks() {
        let mut f = gradient_frame();
        let mut px = PixelateEffect::new(8);
        px.process(&mut f);
        // Pixels within the same 8x8 block should be identical
        let p0 = f.get_pixel(0, 0);
        let p1 = f.get_pixel(1, 1);
        let p2 = f.get_pixel(7, 7);
        assert_eq!(p0, p1);
        assert_eq!(p1, p2);
    }

    #[test]
    fn color_temperature_warm_increases_red() {
        let mut f = TestPatternGenerator::solid(4, 4, 128, 128, 128);
        let (r_before, _, b_before, _) = f.get_pixel(0, 0);
        let mut ct = ColorTemperatureEffect::new(50.0);
        ct.process(&mut f);
        let (r_after, _, b_after, _) = f.get_pixel(0, 0);
        assert!(r_after > r_before, "warm should increase red");
        assert!(b_after < b_before, "warm should decrease blue");
    }

    #[test]
    fn vignette_darkens_corners() {
        let mut f = TestPatternGenerator::solid(64, 64, 200, 200, 200);
        let mut v = VignetteEffect::new(1.0);
        v.process(&mut f);
        let corner = f.get_pixel(0, 0);
        let center = f.get_pixel(32, 32);
        assert!(
            corner.0 < center.0,
            "corner should be darker: corner={:?} center={:?}",
            corner,
            center
        );
    }

    #[test]
    fn night_vision_is_green_tinted() {
        let mut f = test_frame();
        let mut nv = NightVisionEffect::new();
        nv.process(&mut f);
        let (r, g, b, _) = f.get_pixel(32, 24);
        assert!(
            g >= r && g >= b,
            "should be green-dominant: r={r} g={g} b={b}"
        );
    }

    #[test]
    fn glitch_modifies_frame() {
        let mut f = test_frame();
        let orig = f.data.clone();
        let mut gl = GlitchEffect::new(0.5);
        gl.process(&mut f);
        assert_ne!(f.data, orig);
    }

    #[test]
    fn chroma_key_removes_green() {
        let mut f = TestPatternGenerator::solid(10, 10, 0, 255, 0);
        let mut ck = ChromaKeyEffect::green_screen();
        ck.process(&mut f);
        let (r, g, b, a) = f.get_pixel(5, 5);
        assert_eq!((r, g, b, a), (0, 0, 0, 0), "pure green should be keyed out");
    }

    #[test]
    fn chroma_key_preserves_non_green() {
        let mut f = TestPatternGenerator::solid(10, 10, 255, 0, 0);
        let mut ck = ChromaKeyEffect::green_screen();
        ck.process(&mut f);
        let (r, _, _, _) = f.get_pixel(5, 5);
        assert_eq!(r, 255, "red should be preserved");
    }

    #[test]
    fn matrix_rain_produces_green_output() {
        let mut mr = MatrixRainEffect::new(8);
        mr.overlay_mode = false;
        // Process several frames to let columns advance
        let mut f = VideoFrame::new(64, 48);
        for _ in 0..30 {
            f = VideoFrame::new(64, 48);
            mr.process(&mut f);
        }
        // Should have some green pixels
        let mut has_green = false;
        for y in 0..48 {
            for x in 0..64 {
                let (_, g, _, _) = f.get_pixel(x, y);
                if g > 50 {
                    has_green = true;
                    break;
                }
            }
        }
        assert!(has_green, "matrix rain should produce green output");
    }

    #[test]
    fn watermark_modifies_bottom_region() {
        let mut f = TestPatternGenerator::solid(200, 200, 0, 0, 0);
        let mut wm = WatermarkEffect::new("ABCDE");
        wm.process(&mut f);
        // Scan watermark region for any non-black pixel
        let y_start = (200.0 * 0.95) as u32;
        let mut found = false;
        for y in y_start..y_start + 7 {
            for x in 2..40 {
                let (r, g, b, _) = f.get_pixel(x, y);
                if r > 0 || g > 0 || b > 0 {
                    found = true;
                    break;
                }
            }
        }
        assert!(found, "watermark should draw text");
    }

    #[test]
    fn ascii_art_reduces_detail() {
        let mut f = gradient_frame();
        let mut aa = AsciiArtEffect::new(8);
        aa.process(&mut f);
        // Pixels within the same block should be identical (like pixelate)
        let p0 = f.get_pixel(0, 0);
        let p1 = f.get_pixel(3, 3);
        assert_eq!(p0, p1);
    }

    #[test]
    fn brightness_contrast_adjusts() {
        let mut f = TestPatternGenerator::solid(4, 4, 128, 128, 128);
        let mut bc = BrightnessContrastEffect::new(50.0, 1.0);
        bc.process(&mut f);
        let (r, _, _, _) = f.get_pixel(0, 0);
        assert!(r > 128, "brightness +50 should increase pixel value");
    }

    #[test]
    fn grayscale_equalizes_channels() {
        let mut f = TestPatternGenerator::solid(4, 4, 200, 100, 50);
        let mut gs = GrayscaleEffect::new();
        gs.process(&mut f);
        let (r, g, b, _) = f.get_pixel(0, 0);
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn sepia_tints_warm() {
        let mut f = TestPatternGenerator::solid(4, 4, 128, 128, 128);
        let mut s = SepiaEffect::new();
        s.process(&mut f);
        let (r, g, b, _) = f.get_pixel(0, 0);
        assert!(r > g && g > b, "sepia should be warm: r={r} g={g} b={b}");
    }

    #[test]
    fn invert_reverses_values() {
        let mut f = TestPatternGenerator::solid(4, 4, 200, 100, 50);
        let mut inv = InvertEffect::new();
        inv.process(&mut f);
        assert_eq!(f.get_pixel(0, 0), (55, 155, 205, 255));
    }

    #[test]
    fn edge_detection_produces_output() {
        let mut f = TestPatternGenerator::checkerboard(64, 64, 8);
        let mut ed = EdgeDetectionEffect::new(30);
        ed.process(&mut f);
        // Should have some bright edge pixels
        let mut has_edge = false;
        for y in 0..64 {
            for x in 0..64 {
                let (r, _, _, _) = f.get_pixel(x, y);
                if r > 100 {
                    has_edge = true;
                    break;
                }
            }
        }
        assert!(has_edge, "edge detection should find edges in checkerboard");
    }

    #[test]
    fn test_pattern_color_bars() {
        let mut gen = TestPatternGenerator::new();
        let f = gen.color_bars(80, 10);
        let (r, g, b, _) = f.get_pixel(0, 0); // first bar = white
        assert!(r > 200 && g > 200 && b > 200);
    }

    #[test]
    fn chain_total_latency() {
        let mut chain = VideoEffectChain::new();
        chain.add_effect(Box::new(GrayscaleEffect::new()));
        chain.add_effect(Box::new(InvertEffect::new()));
        chain.add_effect(Box::new(BlurEffect::new(3)));
        assert!(chain.total_latency_us() > 0);
    }

    #[test]
    fn all_effects_in_chain_no_panic() {
        let mut chain = VideoEffectChain::new();
        chain.add_effect(Box::new(MirrorEffect::horizontal()));
        chain.add_effect(Box::new(PixelateEffect::new(4)));
        chain.add_effect(Box::new(ColorTemperatureEffect::new(30.0)));
        chain.add_effect(Box::new(VignetteEffect::new(0.5)));
        chain.add_effect(Box::new(NightVisionEffect::new()));
        chain.add_effect(Box::new(VhsRetroEffect::new()));
        chain.add_effect(Box::new(GlitchEffect::new(0.3)));
        chain.add_effect(Box::new(CartoonEffect::new(6, 40)));
        chain.add_effect(Box::new(ChromaKeyEffect::green_screen()));
        chain.add_effect(Box::new(MatrixRainEffect::new(6)));
        chain.add_effect(Box::new(WatermarkEffect::new("TEST")));
        chain.add_effect(Box::new(AsciiArtEffect::new(6)));
        chain.add_effect(Box::new(BrightnessContrastEffect::new(10.0, 1.2)));
        chain.add_effect(Box::new(GrayscaleEffect::new()));
        chain.add_effect(Box::new(SepiaEffect::new()));
        chain.add_effect(Box::new(InvertEffect::new()));
        chain.add_effect(Box::new(PosterizeEffect::new(4)));
        chain.add_effect(Box::new(BlurEffect::new(2)));
        chain.add_effect(Box::new(EdgeDetectionEffect::new(30)));
        assert_eq!(chain.len(), 19);
        let mut f = test_frame();
        chain.process_frame(&mut f);
        // Should not panic, and frame should still be valid size
        assert_eq!(f.data.len(), 64 * 48 * 4);
    }
}
