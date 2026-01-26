//! Model quantization helpers
//!
//! This module provides utilities for working with quantized models,
//! detecting quantization formats, and optimizing model loading.
//!
//! # Features
//!
//! - **Format detection**: Identify GGUF, GGML, GPTQ, AWQ formats
//! - **Quality estimation**: Estimate quality loss from quantization
//! - **Memory calculation**: Calculate VRAM/RAM requirements
//! - **Recommendations**: Suggest optimal quantization for hardware
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::quantization::{QuantizationDetector, QuantFormat, HardwareProfile};
//!
//! let detector = QuantizationDetector::new();
//!
//! // Detect format from model name
//! let format = detector.detect_format("llama-7b-q4_k_m.gguf");
//! assert_eq!(format, Some(QuantFormat::GGUF_Q4_K_M));
//!
//! // Get memory requirements
//! let memory = detector.estimate_memory("7B", &format.unwrap());
//! println!("Requires ~{:.1} GB", memory.total_gb);
//!
//! // Get recommendation for hardware
//! let hw = HardwareProfile { vram_gb: 8.0, ram_gb: 16.0, ..Default::default() };
//! let recommended = detector.recommend_quantization("13B", &hw);
//! ```

use std::collections::HashMap;

/// Quantization format types
///
/// Note: Variant names follow industry-standard conventions (llama.cpp, GGML)
/// rather than Rust CamelCase to match how users refer to these formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain floating point (BF16)
    BF16,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,

    // GGUF/GGML formats
    /// GGUF Q2_K (2-bit with K-quants)
    GGUF_Q2_K,
    /// GGUF Q3_K_S (3-bit K-quants small)
    GGUF_Q3_K_S,
    /// GGUF Q3_K_M (3-bit K-quants medium)
    GGUF_Q3_K_M,
    /// GGUF Q3_K_L (3-bit K-quants large)
    GGUF_Q3_K_L,
    /// GGUF Q4_0 (4-bit legacy)
    GGUF_Q4_0,
    /// GGUF Q4_1 (4-bit legacy with scales)
    GGUF_Q4_1,
    /// GGUF Q4_K_S (4-bit K-quants small)
    GGUF_Q4_K_S,
    /// GGUF Q4_K_M (4-bit K-quants medium) - Most popular
    GGUF_Q4_K_M,
    /// GGUF Q5_0 (5-bit legacy)
    GGUF_Q5_0,
    /// GGUF Q5_1 (5-bit legacy with scales)
    GGUF_Q5_1,
    /// GGUF Q5_K_S (5-bit K-quants small)
    GGUF_Q5_K_S,
    /// GGUF Q5_K_M (5-bit K-quants medium)
    GGUF_Q5_K_M,
    /// GGUF Q6_K (6-bit K-quants)
    GGUF_Q6_K,
    /// GGUF Q8_0 (8-bit)
    GGUF_Q8_0,

    // GPTQ formats
    /// GPTQ 4-bit with group size 128
    GPTQ_4bit_128g,
    /// GPTQ 4-bit with group size 32
    GPTQ_4bit_32g,
    /// GPTQ 8-bit
    GPTQ_8bit,

    // AWQ formats
    /// AWQ 4-bit
    AWQ_4bit,

    // EXL2 formats
    /// EXL2 variable bitrate
    EXL2,

    /// Unknown format
    Unknown,
}

impl QuantFormat {
    /// Get bits per weight for this format
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::FP32 => 32.0,
            Self::FP16 | Self::BF16 => 16.0,
            Self::INT8 | Self::GGUF_Q8_0 | Self::GPTQ_8bit => 8.0,
            Self::GGUF_Q6_K => 6.0,
            Self::GGUF_Q5_0 | Self::GGUF_Q5_1 | Self::GGUF_Q5_K_S | Self::GGUF_Q5_K_M => 5.0,
            Self::GGUF_Q4_0 | Self::GGUF_Q4_1 | Self::GGUF_Q4_K_S | Self::GGUF_Q4_K_M
            | Self::GPTQ_4bit_128g | Self::GPTQ_4bit_32g | Self::AWQ_4bit | Self::INT4 => 4.0,
            Self::GGUF_Q3_K_S | Self::GGUF_Q3_K_M | Self::GGUF_Q3_K_L => 3.0,
            Self::GGUF_Q2_K => 2.0,
            Self::EXL2 => 4.5, // Variable, average
            Self::Unknown => 16.0, // Assume FP16 as safe default
        }
    }

    /// Get estimated quality retention (0.0 - 1.0)
    pub fn quality_retention(&self) -> f32 {
        match self {
            Self::FP32 => 1.0,
            Self::FP16 | Self::BF16 => 0.995,
            Self::INT8 | Self::GGUF_Q8_0 | Self::GPTQ_8bit => 0.99,
            Self::GGUF_Q6_K => 0.98,
            Self::GGUF_Q5_K_M => 0.97,
            Self::GGUF_Q5_K_S | Self::GGUF_Q5_0 | Self::GGUF_Q5_1 => 0.96,
            Self::GGUF_Q4_K_M | Self::GPTQ_4bit_32g => 0.95,
            Self::GGUF_Q4_K_S | Self::GPTQ_4bit_128g | Self::AWQ_4bit => 0.94,
            Self::GGUF_Q4_0 | Self::GGUF_Q4_1 | Self::INT4 => 0.92,
            Self::EXL2 => 0.95,
            Self::GGUF_Q3_K_L => 0.90,
            Self::GGUF_Q3_K_M => 0.88,
            Self::GGUF_Q3_K_S => 0.85,
            Self::GGUF_Q2_K => 0.75,
            Self::Unknown => 0.95,
        }
    }

    /// Get format name as string
    pub fn name(&self) -> &'static str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::INT8 => "INT8",
            Self::INT4 => "INT4",
            Self::GGUF_Q2_K => "Q2_K",
            Self::GGUF_Q3_K_S => "Q3_K_S",
            Self::GGUF_Q3_K_M => "Q3_K_M",
            Self::GGUF_Q3_K_L => "Q3_K_L",
            Self::GGUF_Q4_0 => "Q4_0",
            Self::GGUF_Q4_1 => "Q4_1",
            Self::GGUF_Q4_K_S => "Q4_K_S",
            Self::GGUF_Q4_K_M => "Q4_K_M",
            Self::GGUF_Q5_0 => "Q5_0",
            Self::GGUF_Q5_1 => "Q5_1",
            Self::GGUF_Q5_K_S => "Q5_K_S",
            Self::GGUF_Q5_K_M => "Q5_K_M",
            Self::GGUF_Q6_K => "Q6_K",
            Self::GGUF_Q8_0 => "Q8_0",
            Self::GPTQ_4bit_128g => "GPTQ-4bit-128g",
            Self::GPTQ_4bit_32g => "GPTQ-4bit-32g",
            Self::GPTQ_8bit => "GPTQ-8bit",
            Self::AWQ_4bit => "AWQ-4bit",
            Self::EXL2 => "EXL2",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if format is GGUF compatible
    pub fn is_gguf(&self) -> bool {
        matches!(self,
            Self::GGUF_Q2_K | Self::GGUF_Q3_K_S | Self::GGUF_Q3_K_M | Self::GGUF_Q3_K_L |
            Self::GGUF_Q4_0 | Self::GGUF_Q4_1 | Self::GGUF_Q4_K_S | Self::GGUF_Q4_K_M |
            Self::GGUF_Q5_0 | Self::GGUF_Q5_1 | Self::GGUF_Q5_K_S | Self::GGUF_Q5_K_M |
            Self::GGUF_Q6_K | Self::GGUF_Q8_0
        )
    }

    /// Check if format requires GPU
    pub fn requires_gpu(&self) -> bool {
        matches!(self, Self::GPTQ_4bit_128g | Self::GPTQ_4bit_32g | Self::GPTQ_8bit |
                       Self::AWQ_4bit | Self::EXL2)
    }
}

/// Memory requirements for a model
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Model weights in GB
    pub weights_gb: f64,
    /// KV cache estimate in GB (for context)
    pub kv_cache_gb: f64,
    /// Overhead (CUDA, framework) in GB
    pub overhead_gb: f64,
    /// Total memory needed
    pub total_gb: f64,
    /// Can run on CPU only
    pub cpu_compatible: bool,
    /// Recommended context length for this memory
    pub recommended_context: usize,
}

/// Hardware profile for recommendations
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Available VRAM in GB
    pub vram_gb: f64,
    /// Available RAM in GB
    pub ram_gb: f64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Has CUDA support
    pub has_cuda: bool,
    /// Has Metal support (Apple Silicon)
    pub has_metal: bool,
    /// Has ROCm support (AMD)
    pub has_rocm: bool,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            vram_gb: 0.0,
            ram_gb: 16.0,
            cpu_cores: 4,
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
        }
    }
}

impl HardwareProfile {
    /// Create profile for NVIDIA GPU
    pub fn nvidia(vram_gb: f64, ram_gb: f64) -> Self {
        Self {
            vram_gb,
            ram_gb,
            has_cuda: true,
            ..Default::default()
        }
    }

    /// Create profile for Apple Silicon
    pub fn apple_silicon(unified_memory_gb: f64) -> Self {
        Self {
            vram_gb: unified_memory_gb * 0.7, // ~70% usable for GPU
            ram_gb: unified_memory_gb,
            has_metal: true,
            cpu_cores: 8,
            ..Default::default()
        }
    }

    /// Create CPU-only profile
    pub fn cpu_only(ram_gb: f64, cores: usize) -> Self {
        Self {
            ram_gb,
            cpu_cores: cores,
            ..Default::default()
        }
    }
}

/// Quantization recommendation
#[derive(Debug, Clone)]
pub struct QuantRecommendation {
    /// Recommended format
    pub format: QuantFormat,
    /// Confidence in recommendation (0-1)
    pub confidence: f64,
    /// Reasoning
    pub reason: String,
    /// Expected memory usage
    pub memory: MemoryRequirements,
    /// Alternative formats to consider
    pub alternatives: Vec<QuantFormat>,
    /// Warnings or notes
    pub warnings: Vec<String>,
}

/// Model size categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// ~1B parameters
    Tiny,
    /// ~3B parameters
    Small,
    /// ~7B parameters
    Medium,
    /// ~13B parameters
    Large,
    /// ~30B parameters
    XLarge,
    /// ~70B parameters
    Huge,
    /// Custom size
    Custom(u64),
}

impl ModelSize {
    /// Parse size from string like "7B", "13B", "70B"
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.to_uppercase();
        let s = s.trim();

        // Extract number
        let num_str: String = s.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        let multiplier = if s.ends_with('B') { 1_000_000_000u64 }
                        else if s.ends_with('M') { 1_000_000u64 }
                        else { 1u64 };

        let num: f64 = num_str.parse().ok()?;
        let params = (num * multiplier as f64) as u64;

        Some(Self::from_params(params))
    }

    /// Create from parameter count
    pub fn from_params(params: u64) -> Self {
        match params {
            0..=2_000_000_000 => Self::Tiny,
            2_000_000_001..=5_000_000_000 => Self::Small,
            5_000_000_001..=10_000_000_000 => Self::Medium,
            10_000_000_001..=20_000_000_000 => Self::Large,
            20_000_000_001..=50_000_000_000 => Self::XLarge,
            _ => Self::Huge,
        }
    }

    /// Get approximate parameter count
    pub fn params(&self) -> u64 {
        match self {
            Self::Tiny => 1_000_000_000,
            Self::Small => 3_000_000_000,
            Self::Medium => 7_000_000_000,
            Self::Large => 13_000_000_000,
            Self::XLarge => 30_000_000_000,
            Self::Huge => 70_000_000_000,
            Self::Custom(p) => *p,
        }
    }
}

/// Quantization detector and helper
pub struct QuantizationDetector {
    format_patterns: Vec<(regex::Regex, QuantFormat)>,
}

impl QuantizationDetector {
    /// Create new detector
    pub fn new() -> Self {
        let patterns = vec![
            // GGUF K-quants
            (r"(?i)q2[_-]?k", QuantFormat::GGUF_Q2_K),
            (r"(?i)q3[_-]?k[_-]?s", QuantFormat::GGUF_Q3_K_S),
            (r"(?i)q3[_-]?k[_-]?m", QuantFormat::GGUF_Q3_K_M),
            (r"(?i)q3[_-]?k[_-]?l", QuantFormat::GGUF_Q3_K_L),
            (r"(?i)q4[_-]?k[_-]?s", QuantFormat::GGUF_Q4_K_S),
            (r"(?i)q4[_-]?k[_-]?m", QuantFormat::GGUF_Q4_K_M),
            (r"(?i)q5[_-]?k[_-]?s", QuantFormat::GGUF_Q5_K_S),
            (r"(?i)q5[_-]?k[_-]?m", QuantFormat::GGUF_Q5_K_M),
            (r"(?i)q6[_-]?k", QuantFormat::GGUF_Q6_K),
            // GGUF legacy
            (r"(?i)q4[_-]?0", QuantFormat::GGUF_Q4_0),
            (r"(?i)q4[_-]?1", QuantFormat::GGUF_Q4_1),
            (r"(?i)q5[_-]?0", QuantFormat::GGUF_Q5_0),
            (r"(?i)q5[_-]?1", QuantFormat::GGUF_Q5_1),
            (r"(?i)q8[_-]?0", QuantFormat::GGUF_Q8_0),
            // GPTQ
            (r"(?i)gptq.*4bit.*32g", QuantFormat::GPTQ_4bit_32g),
            (r"(?i)gptq.*4bit", QuantFormat::GPTQ_4bit_128g),
            (r"(?i)gptq.*8bit", QuantFormat::GPTQ_8bit),
            // AWQ
            (r"(?i)awq", QuantFormat::AWQ_4bit),
            // EXL2
            (r"(?i)exl2", QuantFormat::EXL2),
            // Generic
            (r"(?i)fp32", QuantFormat::FP32),
            (r"(?i)fp16|f16", QuantFormat::FP16),
            (r"(?i)bf16", QuantFormat::BF16),
            (r"(?i)int8|8bit", QuantFormat::INT8),
            (r"(?i)int4|4bit", QuantFormat::INT4),
        ];

        let format_patterns = patterns.into_iter()
            .filter_map(|(pattern, format)| {
                regex::Regex::new(pattern).ok().map(|r| (r, format))
            })
            .collect();

        Self { format_patterns }
    }

    /// Detect quantization format from model name/path
    pub fn detect_format(&self, model_name: &str) -> Option<QuantFormat> {
        for (pattern, format) in &self.format_patterns {
            if pattern.is_match(model_name) {
                return Some(*format);
            }
        }

        // Check file extension
        if model_name.ends_with(".gguf") {
            return Some(QuantFormat::GGUF_Q4_K_M); // Default GGUF
        }

        None
    }

    /// Estimate memory requirements for a model
    pub fn estimate_memory(&self, model_size: &str, format: &QuantFormat) -> MemoryRequirements {
        let size = ModelSize::from_str(model_size).unwrap_or(ModelSize::Medium);
        self.estimate_memory_for_size(size, format)
    }

    /// Estimate memory for a model size
    pub fn estimate_memory_for_size(&self, size: ModelSize, format: &QuantFormat) -> MemoryRequirements {
        let params = size.params() as f64;
        let bits = format.bits_per_weight() as f64;

        // Model weights: params * bits / 8 bytes
        let weights_gb = (params * bits / 8.0) / 1_000_000_000.0;

        // KV cache estimate (depends on context, layers, heads)
        // Rough estimate: ~2GB per 4K context for 7B model
        let kv_base = match size {
            ModelSize::Tiny => 0.5,
            ModelSize::Small => 1.0,
            ModelSize::Medium => 2.0,
            ModelSize::Large => 3.0,
            ModelSize::XLarge => 5.0,
            ModelSize::Huge => 8.0,
            ModelSize::Custom(p) => (p as f64 / 7_000_000_000.0) * 2.0,
        };

        // Overhead (CUDA context, framework, etc.)
        let overhead_gb = if format.requires_gpu() { 1.5 } else { 0.5 };

        let total_gb = weights_gb + kv_base + overhead_gb;

        // Recommended context based on remaining memory
        let recommended_context = match total_gb {
            t if t < 4.0 => 8192,
            t if t < 8.0 => 4096,
            t if t < 16.0 => 2048,
            _ => 1024,
        };

        MemoryRequirements {
            weights_gb,
            kv_cache_gb: kv_base,
            overhead_gb,
            total_gb,
            cpu_compatible: !format.requires_gpu(),
            recommended_context,
        }
    }

    /// Recommend quantization format for given model size and hardware
    pub fn recommend_quantization(&self, model_size: &str, hardware: &HardwareProfile) -> QuantRecommendation {
        let size = ModelSize::from_str(model_size).unwrap_or(ModelSize::Medium);
        self.recommend_for_size(size, hardware)
    }

    /// Recommend quantization for model size
    pub fn recommend_for_size(&self, size: ModelSize, hardware: &HardwareProfile) -> QuantRecommendation {
        let available_memory = if hardware.has_cuda || hardware.has_metal || hardware.has_rocm {
            hardware.vram_gb
        } else {
            hardware.ram_gb * 0.8 // Leave some for OS
        };

        // Try formats from highest quality to lowest
        let formats_to_try = if hardware.has_cuda {
            vec![
                QuantFormat::FP16,
                QuantFormat::GPTQ_4bit_32g,
                QuantFormat::AWQ_4bit,
                QuantFormat::GPTQ_4bit_128g,
                QuantFormat::GGUF_Q8_0,
                QuantFormat::GGUF_Q6_K,
                QuantFormat::GGUF_Q5_K_M,
                QuantFormat::GGUF_Q4_K_M,
                QuantFormat::GGUF_Q4_K_S,
                QuantFormat::GGUF_Q3_K_M,
                QuantFormat::GGUF_Q2_K,
            ]
        } else {
            // CPU/Metal - prefer GGUF
            vec![
                QuantFormat::GGUF_Q8_0,
                QuantFormat::GGUF_Q6_K,
                QuantFormat::GGUF_Q5_K_M,
                QuantFormat::GGUF_Q5_K_S,
                QuantFormat::GGUF_Q4_K_M,
                QuantFormat::GGUF_Q4_K_S,
                QuantFormat::GGUF_Q3_K_M,
                QuantFormat::GGUF_Q3_K_S,
                QuantFormat::GGUF_Q2_K,
            ]
        };

        let mut best_format = QuantFormat::GGUF_Q2_K;
        let mut best_memory = self.estimate_memory_for_size(size, &best_format);
        let mut alternatives = Vec::new();
        let mut warnings = Vec::new();

        for format in formats_to_try {
            let memory = self.estimate_memory_for_size(size, &format);

            if memory.total_gb <= available_memory {
                if format.quality_retention() > best_format.quality_retention() {
                    alternatives.push(best_format);
                    best_format = format;
                    best_memory = memory;
                } else {
                    alternatives.push(format);
                }
            }
        }

        // Generate warnings
        if best_memory.total_gb > available_memory * 0.9 {
            warnings.push("Memory usage is close to limit, may experience slowdowns".to_string());
        }

        if best_format == QuantFormat::GGUF_Q2_K {
            warnings.push("Using aggressive quantization, expect quality degradation".to_string());
        }

        if !hardware.has_cuda && !hardware.has_metal && size.params() > 13_000_000_000 {
            warnings.push("Large model on CPU will be slow, consider smaller model".to_string());
        }

        let confidence = if best_memory.total_gb <= available_memory * 0.7 { 0.95 }
                        else if best_memory.total_gb <= available_memory * 0.85 { 0.8 }
                        else { 0.6 };

        let reason = format!(
            "{} provides best quality ({:.0}%) that fits in {:.1}GB",
            best_format.name(),
            best_format.quality_retention() * 100.0,
            available_memory
        );

        QuantRecommendation {
            format: best_format,
            confidence,
            reason,
            memory: best_memory,
            alternatives: alternatives.into_iter().take(3).collect(),
            warnings,
        }
    }

    /// Get all available formats sorted by quality
    pub fn available_formats(&self) -> Vec<QuantFormat> {
        let mut formats = vec![
            QuantFormat::FP32,
            QuantFormat::FP16,
            QuantFormat::BF16,
            QuantFormat::GGUF_Q8_0,
            QuantFormat::GPTQ_8bit,
            QuantFormat::GGUF_Q6_K,
            QuantFormat::GGUF_Q5_K_M,
            QuantFormat::GGUF_Q5_K_S,
            QuantFormat::GGUF_Q5_0,
            QuantFormat::GGUF_Q4_K_M,
            QuantFormat::GPTQ_4bit_32g,
            QuantFormat::AWQ_4bit,
            QuantFormat::GGUF_Q4_K_S,
            QuantFormat::GPTQ_4bit_128g,
            QuantFormat::GGUF_Q4_0,
            QuantFormat::GGUF_Q3_K_L,
            QuantFormat::GGUF_Q3_K_M,
            QuantFormat::GGUF_Q3_K_S,
            QuantFormat::GGUF_Q2_K,
        ];
        formats.sort_by(|a, b| b.quality_retention().partial_cmp(&a.quality_retention()).unwrap());
        formats
    }

    /// Compare two formats
    pub fn compare_formats(&self, a: &QuantFormat, b: &QuantFormat, size: ModelSize) -> FormatComparison {
        let mem_a = self.estimate_memory_for_size(size, a);
        let mem_b = self.estimate_memory_for_size(size, b);

        FormatComparison {
            format_a: *a,
            format_b: *b,
            quality_diff: a.quality_retention() - b.quality_retention(),
            memory_diff_gb: mem_a.total_gb - mem_b.total_gb,
            bits_diff: a.bits_per_weight() - b.bits_per_weight(),
            recommendation: if a.quality_retention() > b.quality_retention() && mem_a.total_gb <= mem_b.total_gb * 1.1 {
                format!("{} is better (higher quality, similar size)", a.name())
            } else if b.quality_retention() > a.quality_retention() && mem_b.total_gb <= mem_a.total_gb * 1.1 {
                format!("{} is better (higher quality, similar size)", b.name())
            } else if mem_a.total_gb < mem_b.total_gb * 0.8 {
                format!("{} is smaller but lower quality", a.name())
            } else {
                "Trade-off between quality and size".to_string()
            },
        }
    }
}

impl Default for QuantizationDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison between two formats
#[derive(Debug, Clone)]
pub struct FormatComparison {
    pub format_a: QuantFormat,
    pub format_b: QuantFormat,
    /// Quality difference (positive = A better)
    pub quality_diff: f32,
    /// Memory difference in GB (positive = A larger)
    pub memory_diff_gb: f64,
    /// Bits per weight difference
    pub bits_diff: f32,
    /// Recommendation text
    pub recommendation: String,
}

/// GGUF file metadata parser
pub struct GgufMetadata {
    /// Model architecture
    pub architecture: Option<String>,
    /// Parameter count
    pub parameter_count: Option<u64>,
    /// Context length
    pub context_length: Option<usize>,
    /// Embedding length
    pub embedding_length: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Quantization type
    pub quantization: Option<QuantFormat>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl GgufMetadata {
    /// Parse metadata from GGUF file header (simplified)
    pub fn from_model_name(name: &str) -> Self {
        let detector = QuantizationDetector::new();
        let quantization = detector.detect_format(name);

        // Try to extract size from name
        let size_pattern = regex::Regex::new(r"(\d+(?:\.\d+)?)[Bb]").ok();
        let parameter_count = size_pattern
            .and_then(|p| p.captures(name))
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse::<f64>().ok())
            .map(|n| (n * 1_000_000_000.0) as u64);

        Self {
            architecture: Self::detect_architecture(name),
            parameter_count,
            context_length: None,
            embedding_length: None,
            num_layers: None,
            quantization,
            metadata: HashMap::new(),
        }
    }

    fn detect_architecture(name: &str) -> Option<String> {
        let lower = name.to_lowercase();

        if lower.contains("llama") { Some("llama".to_string()) }
        else if lower.contains("mistral") { Some("mistral".to_string()) }
        else if lower.contains("mixtral") { Some("mixtral".to_string()) }
        else if lower.contains("phi") { Some("phi".to_string()) }
        else if lower.contains("qwen") { Some("qwen".to_string()) }
        else if lower.contains("gemma") { Some("gemma".to_string()) }
        else if lower.contains("falcon") { Some("falcon".to_string()) }
        else if lower.contains("mpt") { Some("mpt".to_string()) }
        else if lower.contains("starcoder") { Some("starcoder".to_string()) }
        else if lower.contains("codellama") { Some("codellama".to_string()) }
        else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        let detector = QuantizationDetector::new();

        assert_eq!(detector.detect_format("llama-7b-q4_k_m.gguf"), Some(QuantFormat::GGUF_Q4_K_M));
        assert_eq!(detector.detect_format("mistral-Q5_K_S"), Some(QuantFormat::GGUF_Q5_K_S));
        assert_eq!(detector.detect_format("model-gptq-4bit"), Some(QuantFormat::GPTQ_4bit_128g));
        assert_eq!(detector.detect_format("model-awq"), Some(QuantFormat::AWQ_4bit));
    }

    #[test]
    fn test_model_size_parsing() {
        assert_eq!(ModelSize::from_str("7B").map(|s| s.params()), Some(7_000_000_000));
        assert_eq!(ModelSize::from_str("13B").map(|s| s.params()), Some(13_000_000_000));
        assert_eq!(ModelSize::from_str("70B").map(|s| s.params()), Some(70_000_000_000));
    }

    #[test]
    fn test_memory_estimation() {
        let detector = QuantizationDetector::new();

        let mem_q4 = detector.estimate_memory("7B", &QuantFormat::GGUF_Q4_K_M);
        let mem_q8 = detector.estimate_memory("7B", &QuantFormat::GGUF_Q8_0);

        // Q4 should use less memory than Q8
        assert!(mem_q4.weights_gb < mem_q8.weights_gb);
    }

    #[test]
    fn test_quality_retention() {
        assert!(QuantFormat::FP16.quality_retention() > QuantFormat::GGUF_Q4_K_M.quality_retention());
        assert!(QuantFormat::GGUF_Q4_K_M.quality_retention() > QuantFormat::GGUF_Q2_K.quality_retention());
    }

    #[test]
    fn test_recommendation() {
        let detector = QuantizationDetector::new();

        // 8GB GPU should recommend Q4 or Q5 for 7B
        let hw = HardwareProfile::nvidia(8.0, 16.0);
        let rec = detector.recommend_quantization("7B", &hw);

        assert!(rec.memory.total_gb <= 8.0);
        assert!(rec.format.quality_retention() >= 0.9);
    }

    #[test]
    fn test_gguf_metadata() {
        let meta = GgufMetadata::from_model_name("llama-2-7b-chat-q4_k_m.gguf");

        assert_eq!(meta.architecture, Some("llama".to_string()));
        assert_eq!(meta.parameter_count, Some(7_000_000_000));
        assert_eq!(meta.quantization, Some(QuantFormat::GGUF_Q4_K_M));
    }
}
