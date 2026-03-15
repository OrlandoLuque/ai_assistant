//! Pure-Rust template-matching OCR engine.

// ============================================================================
// OCR-lite Engine (Template Matching)
// ============================================================================

/// OCR engine configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OcrConfig {
    /// Minimum confidence threshold for character recognition (0.0 - 1.0).
    pub min_confidence: f32,
    /// Expected character height in pixels.
    pub char_height: usize,
    /// Binarization threshold (0-255). If None, uses Otsu's method.
    pub binarize_threshold: Option<u8>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            char_height: 7,
            binarize_threshold: None,
        }
    }
}

/// A glyph template for template matching.
#[derive(Debug, Clone)]
pub struct GlyphTemplate {
    pub character: char,
    pub width: usize,
    pub height: usize,
    pub bitmap: Vec<u8>, // row-major, 0=white, 1=black
}

/// A recognized text line.
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub y_position: usize,
}

/// OCR recognition result.
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub lines: Vec<OcrLine>,
    pub full_text: String,
    pub average_confidence: f32,
}

/// Pure-Rust template-matching OCR engine.
///
/// Uses 5x7 LED-style bitmaps for character recognition via cross-correlation.
pub struct OcrEngine {
    pub templates: Vec<GlyphTemplate>,
    config: OcrConfig,
}

impl OcrEngine {
    /// Create a new OCR engine with default templates (A-Z, a-z, 0-9, common punctuation).
    pub fn with_default_templates(config: OcrConfig) -> Self {
        let mut engine = Self {
            templates: Vec::new(),
            config,
        };
        engine.load_default_templates();
        engine
    }

    /// Create a new OCR engine with custom templates.
    pub fn new(templates: Vec<GlyphTemplate>, config: OcrConfig) -> Self {
        Self { templates, config }
    }

    /// Add a custom glyph template.
    pub fn add_template(&mut self, template: GlyphTemplate) {
        self.templates.push(template);
    }

    /// Recognize text from a grayscale bitmap image.
    /// `image` is row-major grayscale pixels (0-255), `width` x `height`.
    pub fn recognize_bitmap(&self, image: &[u8], width: usize, height: usize) -> OcrResult {
        if image.len() != width * height || width == 0 || height == 0 {
            return OcrResult {
                lines: Vec::new(),
                full_text: String::new(),
                average_confidence: 0.0,
            };
        }

        // Step 1: Binarize
        let binary = self.binarize(image, width, height);

        // Step 2: Detect text lines (horizontal projection)
        let line_ranges = self.detect_text_lines(&binary, width, height);

        // Step 3: For each line, segment characters and match
        let mut lines = Vec::new();
        for (y_start, y_end) in &line_ranges {
            let line_height = y_end - y_start;
            let line_slice: Vec<u8> =
                binary[y_start * width..(y_end * width).min(binary.len())].to_vec();

            let char_ranges = self.segment_characters(&line_slice, width, line_height);
            let mut text = String::new();
            let mut total_conf = 0.0f32;
            let mut char_count = 0;

            let mut last_x_end = 0;
            for (x_start, x_end) in &char_ranges {
                // Detect spaces: if gap > char_width * 0.8
                if *x_start > last_x_end + 3 && !text.is_empty() {
                    text.push(' ');
                }

                // Extract character region
                let char_width = x_end - x_start;
                let mut char_bitmap = vec![0u8; char_width * line_height];
                for row in 0..line_height {
                    for col in 0..char_width {
                        if x_start + col < width {
                            char_bitmap[row * char_width + col] =
                                line_slice[row * width + x_start + col];
                        }
                    }
                }

                let (ch, conf) = self.match_template(&char_bitmap, char_width, line_height);
                if conf >= self.config.min_confidence {
                    text.push(ch);
                    total_conf += conf;
                    char_count += 1;
                }
                last_x_end = *x_end;
            }

            let avg_conf = if char_count > 0 {
                total_conf / char_count as f32
            } else {
                0.0
            };
            if !text.is_empty() {
                lines.push(OcrLine {
                    text: text.clone(),
                    confidence: avg_conf,
                    y_position: *y_start,
                });
            }
        }

        let full_text = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let avg = if lines.is_empty() {
            0.0
        } else {
            lines.iter().map(|l| l.confidence).sum::<f32>() / lines.len() as f32
        };
        OcrResult {
            lines,
            full_text,
            average_confidence: avg,
        }
    }

    /// Binarize a grayscale image using Otsu's method or a fixed threshold.
    pub fn binarize(&self, image: &[u8], _width: usize, _height: usize) -> Vec<u8> {
        let threshold = self
            .config
            .binarize_threshold
            .unwrap_or_else(|| self.otsu_threshold(image));
        image
            .iter()
            .map(|&p| if p < threshold { 1 } else { 0 })
            .collect()
    }

    /// Compute Otsu's optimal binarization threshold.
    pub(crate) fn otsu_threshold(&self, image: &[u8]) -> u8 {
        let mut histogram = [0u32; 256];
        for &pixel in image {
            histogram[pixel as usize] += 1;
        }

        let total = image.len() as f64;
        let mut sum_total = 0.0f64;
        for (i, &count) in histogram.iter().enumerate() {
            sum_total += i as f64 * count as f64;
        }

        let mut sum_bg = 0.0f64;
        let mut weight_bg = 0.0f64;
        let mut max_variance = 0.0f64;
        let mut best_threshold = 0u8;

        for (t, &count) in histogram.iter().enumerate() {
            weight_bg += count as f64;
            if weight_bg == 0.0 {
                continue;
            }

            let weight_fg = total - weight_bg;
            if weight_fg == 0.0 {
                break;
            }

            sum_bg += t as f64 * count as f64;
            let mean_bg = sum_bg / weight_bg;
            let mean_fg = (sum_total - sum_bg) / weight_fg;

            let variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
            if variance > max_variance {
                max_variance = variance;
                best_threshold = t as u8;
            }
        }

        best_threshold
    }

    /// Detect text lines using horizontal projection profile.
    /// Returns (y_start, y_end) pairs for each detected line.
    pub fn detect_text_lines(
        &self,
        binary: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        // Count black pixels per row
        let mut row_sums: Vec<usize> = Vec::with_capacity(height);
        for y in 0..height {
            let sum: usize = binary[y * width..(y + 1) * width]
                .iter()
                .map(|&p| p as usize)
                .sum();
            row_sums.push(sum);
        }

        // Find runs of non-zero rows (text lines)
        let mut lines = Vec::new();
        let mut in_line = false;
        let mut start = 0;
        let threshold = 1; // at least 1 black pixel

        for (y, &sum) in row_sums.iter().enumerate() {
            if sum >= threshold && !in_line {
                in_line = true;
                start = y;
            } else if sum < threshold && in_line {
                in_line = false;
                if y - start >= 3 {
                    // minimum line height
                    lines.push((start, y));
                }
            }
        }
        if in_line && height - start >= 3 {
            lines.push((start, height));
        }

        lines
    }

    /// Segment characters in a binary line image using vertical projection.
    /// Returns (x_start, x_end) pairs for each character.
    pub fn segment_characters(
        &self,
        line_binary: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        // Count black pixels per column
        let mut col_sums: Vec<usize> = Vec::with_capacity(width);
        for x in 0..width {
            let mut sum = 0;
            for y in 0..height {
                if x < width && y * width + x < line_binary.len() {
                    sum += line_binary[y * width + x] as usize;
                }
            }
            col_sums.push(sum);
        }

        // Find runs of non-zero columns (characters)
        let mut chars = Vec::new();
        let mut in_char = false;
        let mut start = 0;

        for (x, &sum) in col_sums.iter().enumerate() {
            if sum > 0 && !in_char {
                in_char = true;
                start = x;
            } else if sum == 0 && in_char {
                in_char = false;
                chars.push((start, x));
            }
        }
        if in_char {
            chars.push((start, width));
        }

        chars
    }

    /// Match a character bitmap against templates using normalized cross-correlation.
    /// Returns (best_char, confidence).
    pub fn match_template(
        &self,
        char_bitmap: &[u8],
        char_width: usize,
        char_height: usize,
    ) -> (char, f32) {
        let mut best_char = '?';
        let mut best_score = -1.0f32;

        for template in &self.templates {
            let score = self.cross_correlate(
                char_bitmap,
                char_width,
                char_height,
                &template.bitmap,
                template.width,
                template.height,
            );
            if score > best_score {
                best_score = score;
                best_char = template.character;
            }
        }

        (best_char, best_score.max(0.0))
    }

    /// Normalized cross-correlation between two bitmaps.
    /// Resizes the smaller to match the larger for comparison.
    fn cross_correlate(
        &self,
        img: &[u8],
        img_w: usize,
        img_h: usize,
        tmpl: &[u8],
        tmpl_w: usize,
        tmpl_h: usize,
    ) -> f32 {
        // Resize both to template size using nearest-neighbor
        let w = tmpl_w;
        let h = tmpl_h;

        let resized_img = Self::resize_nearest(img, img_w, img_h, w, h);

        // NCC = sum(a*b) / sqrt(sum(a^2) * sum(b^2))
        let mut sum_ab = 0.0f64;
        let mut sum_aa = 0.0f64;
        let mut sum_bb = 0.0f64;

        for i in 0..(w * h) {
            let a = *resized_img.get(i).unwrap_or(&0) as f64;
            let b = *tmpl.get(i).unwrap_or(&0) as f64;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
        }

        let denom = (sum_aa * sum_bb).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (sum_ab / denom) as f32
    }

    /// Nearest-neighbor resize.
    fn resize_nearest(
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        let mut dst = vec![0u8; dst_w * dst_h];
        if src_w == 0 || src_h == 0 {
            return dst;
        }
        for y in 0..dst_h {
            for x in 0..dst_w {
                let sx = (x * src_w) / dst_w.max(1);
                let sy = (y * src_h) / dst_h.max(1);
                let si = sy * src_w + sx;
                dst[y * dst_w + x] = *src.get(si).unwrap_or(&0);
            }
        }
        dst
    }

    /// Load default 5x7 LED-style glyph templates for common characters.
    fn load_default_templates(&mut self) {
        // Define templates as 5-wide x 7-tall bitmaps
        // Each string is 7 rows of 5 chars, where '#' = 1 (black) and '.' = 0 (white)
        let templates = vec![
            (
                'A',
                vec![
                    ".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
                ],
            ),
            (
                'B',
                vec![
                    "####.", "#...#", "#...#", "####.", "#...#", "#...#", "####.",
                ],
            ),
            (
                'C',
                vec![
                    ".###.", "#...#", "#....", "#....", "#....", "#...#", ".###.",
                ],
            ),
            (
                'D',
                vec![
                    "####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####.",
                ],
            ),
            (
                'E',
                vec![
                    "#####", "#....", "#....", "###..", "#....", "#....", "#####",
                ],
            ),
            (
                'F',
                vec![
                    "#####", "#....", "#....", "###..", "#....", "#....", "#....",
                ],
            ),
            (
                'H',
                vec![
                    "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
                ],
            ),
            ('I', vec!["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"]),
            (
                'L',
                vec![
                    "#....", "#....", "#....", "#....", "#....", "#....", "#####",
                ],
            ),
            (
                'O',
                vec![
                    ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
                ],
            ),
            (
                'T',
                vec![
                    "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..",
                ],
            ),
            (
                '0',
                vec![
                    ".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###.",
                ],
            ),
            (
                '1',
                vec![
                    "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
                ],
            ),
            (
                ' ',
                vec![
                    ".....", ".....", ".....", ".....", ".....", ".....", ".....",
                ],
            ),
        ];

        for (ch, rows) in templates {
            let width = rows[0].len();
            let height = rows.len();
            let mut bitmap = Vec::with_capacity(width * height);
            for row in &rows {
                for c in row.chars() {
                    bitmap.push(if c == '#' { 1 } else { 0 });
                }
            }
            self.templates.push(GlyphTemplate {
                character: ch,
                width,
                height,
                bitmap,
            });
        }
    }
}
