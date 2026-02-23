//! Image & Video Generation providers and pipelines.
//!
//! Implements v5 roadmap Phase 4 (items 4.1, 4.2, 4.3):
//! - 4.1: Image generation providers (DALL-E, Stable Diffusion, Flux, Local)
//! - 4.2: Image editing (inpaint, outpaint, style transfer, upscale, etc.)
//! - 4.3: Video generation providers (Runway, Sora, Replicate)
//!
//! Feature-gated behind the `media-generation` feature flag. The outer `#[cfg]`
//! guard ensures this entire module compiles away when the feature is not enabled.

#[cfg(feature = "media-generation")]
mod inner {
    use std::collections::HashMap;
    use std::fmt;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::error::{AiError, MediaGenerationError};

    // ========================================================================
    // 4.1 — Image Generation Providers
    // ========================================================================

    // --- Enums --------------------------------------------------------------

    /// Style preset for image generation.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ImageStyle {
        Natural,
        Vivid,
        Artistic,
        Photorealistic,
        Anime,
        Digital,
    }

    impl fmt::Display for ImageStyle {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ImageStyle::Natural => write!(f, "natural"),
                ImageStyle::Vivid => write!(f, "vivid"),
                ImageStyle::Artistic => write!(f, "artistic"),
                ImageStyle::Photorealistic => write!(f, "photorealistic"),
                ImageStyle::Anime => write!(f, "anime"),
                ImageStyle::Digital => write!(f, "digital"),
            }
        }
    }

    /// Quality level for generated images.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ImageQuality {
        Draft,
        Standard,
        HD,
        UltraHD,
    }

    impl fmt::Display for ImageQuality {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ImageQuality::Draft => write!(f, "draft"),
                ImageQuality::Standard => write!(f, "standard"),
                ImageQuality::HD => write!(f, "hd"),
                ImageQuality::UltraHD => write!(f, "ultra_hd"),
            }
        }
    }

    /// Output image format.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ImageFormat {
        Png,
        Jpeg,
        WebP,
        Svg,
    }

    impl fmt::Display for ImageFormat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ImageFormat::Png => write!(f, "png"),
                ImageFormat::Jpeg => write!(f, "jpeg"),
                ImageFormat::WebP => write!(f, "webp"),
                ImageFormat::Svg => write!(f, "svg"),
            }
        }
    }

    // --- Config & result structs --------------------------------------------

    /// Configuration for an image generation request.
    #[derive(Debug, Clone)]
    pub struct ImageGenConfig {
        /// Output width in pixels.
        pub width: u32,
        /// Output height in pixels.
        pub height: u32,
        /// Optional style preset.
        pub style: Option<ImageStyle>,
        /// Quality level.
        pub quality: ImageQuality,
        /// Number of images to generate.
        pub num_images: u32,
        /// Negative prompt (things to avoid).
        pub negative_prompt: Option<String>,
        /// Deterministic seed.
        pub seed: Option<u64>,
        /// Classifier-free guidance scale.
        pub guidance_scale: Option<f32>,
    }

    impl Default for ImageGenConfig {
        fn default() -> Self {
            Self {
                width: 1024,
                height: 1024,
                style: None,
                quality: ImageQuality::Standard,
                num_images: 1,
                negative_prompt: None,
                seed: None,
                guidance_scale: None,
            }
        }
    }

    impl ImageGenConfig {
        /// Validate the configuration, returning an error for invalid params.
        pub fn validate(&self) -> Result<(), AiError> {
            if self.width == 0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "width".to_string(),
                    reason: "width must be greater than 0".to_string(),
                }
                .into());
            }
            if self.height == 0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "height".to_string(),
                    reason: "height must be greater than 0".to_string(),
                }
                .into());
            }
            if self.width > 8192 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "width".to_string(),
                    reason: format!("width {} exceeds maximum 8192", self.width),
                }
                .into());
            }
            if self.height > 8192 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "height".to_string(),
                    reason: format!("height {} exceeds maximum 8192", self.height),
                }
                .into());
            }
            if self.num_images == 0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "num_images".to_string(),
                    reason: "num_images must be at least 1".to_string(),
                }
                .into());
            }
            if self.num_images > 10 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "num_images".to_string(),
                    reason: format!("num_images {} exceeds maximum 10", self.num_images),
                }
                .into());
            }
            if let Some(gs) = self.guidance_scale {
                if gs < 0.0 || gs > 50.0 {
                    return Err(MediaGenerationError::InvalidParams {
                        param: "guidance_scale".to_string(),
                        reason: format!("guidance_scale {} out of range [0.0, 50.0]", gs),
                    }
                    .into());
                }
            }
            Ok(())
        }
    }

    /// A successfully generated image.
    #[derive(Debug, Clone)]
    pub struct GeneratedImage {
        /// Raw image bytes.
        pub bytes: Vec<u8>,
        /// Output format.
        pub format: ImageFormat,
        /// Actual output width.
        pub width: u32,
        /// Actual output height.
        pub height: u32,
        /// Provider may revise the prompt for safety or quality.
        pub revised_prompt: Option<String>,
        /// Seed used for generation (for reproducibility).
        pub seed: Option<u64>,
        /// Time taken to generate, in milliseconds.
        pub generation_time_ms: u64,
    }

    // --- Traits -------------------------------------------------------------

    /// Trait for image generation providers.
    pub trait ImageGenerationProvider: Send + Sync {
        /// Generate one or more images from a text prompt.
        fn generate_image(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError>;

        /// Human-readable provider name.
        fn provider_name(&self) -> &str;

        /// List of image formats this provider supports.
        fn supported_formats(&self) -> Vec<ImageFormat>;
    }

    // --- Concrete providers -------------------------------------------------

    /// DALL-E image generation provider (OpenAI).
    #[derive(Debug, Clone)]
    pub struct DallEProvider {
        pub api_key: String,
        pub model: String,
        pub endpoint: String,
    }

    impl DallEProvider {
        /// Create a new DALL-E provider with the given API key.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: "dall-e-3".to_string(),
                endpoint: "https://api.openai.com/v1/images/generations".to_string(),
            }
        }

        /// Set the model (builder pattern).
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }

        /// Set the endpoint (builder pattern).
        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }
    }

    impl ImageGenerationProvider for DallEProvider {
        fn generate_image(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }

            // In production this would call the OpenAI API via HTTP.
            // The actual HTTP call is stubbed here; tests use mock responses.
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "dall-e"
        }

        fn supported_formats(&self) -> Vec<ImageFormat> {
            vec![ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP]
        }
    }

    /// Stable Diffusion image generation provider (Stability AI REST API).
    #[derive(Debug, Clone)]
    pub struct StableDiffusionProvider {
        pub api_key: String,
        pub model: String,
        pub endpoint: String,
    }

    impl StableDiffusionProvider {
        /// Create a new Stable Diffusion provider with the given API key.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: "stable-diffusion-xl-1024-v1-0".to_string(),
                endpoint: "https://api.stability.ai/v1/generation".to_string(),
            }
        }

        /// Set the model (builder pattern).
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }

        /// Set the endpoint (builder pattern).
        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }
    }

    impl ImageGenerationProvider for StableDiffusionProvider {
        fn generate_image(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }

            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "stable-diffusion"
        }

        fn supported_formats(&self) -> Vec<ImageFormat> {
            vec![ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP]
        }
    }

    /// Flux image generation provider (Black Forest Labs).
    #[derive(Debug, Clone)]
    pub struct FluxProvider {
        pub api_key: String,
        pub model: String,
        pub endpoint: String,
    }

    impl FluxProvider {
        /// Create a new Flux provider with the given API key.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: "flux-1-schnell".to_string(),
                endpoint: "https://api.bfl.ml/v1/image".to_string(),
            }
        }

        /// Set the model (builder pattern).
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }

        /// Set the endpoint (builder pattern).
        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }
    }

    impl ImageGenerationProvider for FluxProvider {
        fn generate_image(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }

            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "flux"
        }

        fn supported_formats(&self) -> Vec<ImageFormat> {
            vec![ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP]
        }
    }

    /// Local diffusion provider (ComfyUI / SD.Next local HTTP).
    #[derive(Debug, Clone)]
    pub struct LocalDiffusionProvider {
        pub base_url: String,
    }

    impl LocalDiffusionProvider {
        /// Create a new local diffusion provider pointing at the given base URL.
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
            }
        }
    }

    impl ImageGenerationProvider for LocalDiffusionProvider {
        fn generate_image(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }

            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: format!("local server at {} not reachable", self.base_url),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "local-diffusion"
        }

        fn supported_formats(&self) -> Vec<ImageFormat> {
            vec![
                ImageFormat::Png,
                ImageFormat::Jpeg,
                ImageFormat::WebP,
                ImageFormat::Svg,
            ]
        }
    }

    // --- Image Provider Router ----------------------------------------------

    /// Routes image generation requests to the correct provider by name.
    pub struct ImageProviderRouter {
        providers: HashMap<String, Box<dyn ImageGenerationProvider>>,
    }

    impl fmt::Debug for ImageProviderRouter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("ImageProviderRouter")
                .field("providers", &self.providers.keys().collect::<Vec<_>>())
                .finish()
        }
    }

    impl ImageProviderRouter {
        /// Create an empty router.
        pub fn new() -> Self {
            Self {
                providers: HashMap::new(),
            }
        }

        /// Register a provider under its `provider_name()`.
        pub fn register(&mut self, provider: Box<dyn ImageGenerationProvider>) {
            let name = provider.provider_name().to_string();
            self.providers.insert(name, provider);
        }

        /// Resolve a provider by name.
        pub fn resolve(&self, name: &str) -> Option<&dyn ImageGenerationProvider> {
            self.providers.get(name).map(|b| b.as_ref())
        }

        /// List all registered provider names.
        pub fn provider_names(&self) -> Vec<String> {
            self.providers.keys().cloned().collect()
        }

        /// Generate an image via the named provider.
        pub fn generate(
            &self,
            provider_name: &str,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<GeneratedImage, AiError> {
            let provider = self.resolve(provider_name).ok_or_else(|| {
                AiError::MediaGeneration(MediaGenerationError::ProviderUnavailable {
                    provider: provider_name.to_string(),
                    reason: format!("no provider registered with name '{}'", provider_name),
                })
            })?;
            provider.generate_image(prompt, config)
        }
    }

    // ========================================================================
    // 4.2 — Image Editing
    // ========================================================================

    /// Operation type for image editing.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ImageEditOperation {
        Inpaint,
        Outpaint,
        StyleTransfer,
        Upscale,
        RemoveBackground,
        Variation,
    }

    impl fmt::Display for ImageEditOperation {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ImageEditOperation::Inpaint => write!(f, "inpaint"),
                ImageEditOperation::Outpaint => write!(f, "outpaint"),
                ImageEditOperation::StyleTransfer => write!(f, "style_transfer"),
                ImageEditOperation::Upscale => write!(f, "upscale"),
                ImageEditOperation::RemoveBackground => write!(f, "remove_background"),
                ImageEditOperation::Variation => write!(f, "variation"),
            }
        }
    }

    /// Configuration for an image editing request.
    #[derive(Debug, Clone)]
    pub struct ImageEditConfig {
        /// The editing operation to apply.
        pub operation: ImageEditOperation,
        /// Text prompt describing the desired edit.
        pub prompt: String,
        /// Optional mask image bytes (for inpaint/outpaint).
        pub mask: Option<Vec<u8>>,
        /// Denoising strength (0.0 = no change, 1.0 = full regeneration).
        pub strength: f32,
        /// Classifier-free guidance scale.
        pub guidance_scale: f32,
    }

    impl Default for ImageEditConfig {
        fn default() -> Self {
            Self {
                operation: ImageEditOperation::Variation,
                prompt: String::new(),
                mask: None,
                strength: 0.75,
                guidance_scale: 7.5,
            }
        }
    }

    impl ImageEditConfig {
        /// Validate the configuration.
        pub fn validate(&self) -> Result<(), AiError> {
            if self.prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "edit prompt cannot be empty".to_string(),
                }
                .into());
            }
            if self.strength < 0.0 || self.strength > 1.0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "strength".to_string(),
                    reason: format!("strength {} out of range [0.0, 1.0]", self.strength),
                }
                .into());
            }
            if self.guidance_scale < 0.0 || self.guidance_scale > 50.0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "guidance_scale".to_string(),
                    reason: format!(
                        "guidance_scale {} out of range [0.0, 50.0]",
                        self.guidance_scale
                    ),
                }
                .into());
            }
            // Inpaint and Outpaint require a mask
            if matches!(
                self.operation,
                ImageEditOperation::Inpaint | ImageEditOperation::Outpaint
            ) && self.mask.is_none()
            {
                return Err(MediaGenerationError::InvalidParams {
                    param: "mask".to_string(),
                    reason: format!("{} requires a mask image", self.operation),
                }
                .into());
            }
            Ok(())
        }
    }

    /// Trait for image editing providers.
    pub trait ImageEditProvider: Send + Sync {
        /// Edit an existing image according to the config.
        fn edit_image(
            &self,
            image: &[u8],
            config: &ImageEditConfig,
        ) -> Result<GeneratedImage, AiError>;
    }

    /// DALL-E image editing provider (OpenAI edits endpoint).
    #[derive(Debug, Clone)]
    pub struct DallEEditProvider {
        pub api_key: String,
        pub endpoint: String,
    }

    impl DallEEditProvider {
        /// Create a new DALL-E edit provider.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                endpoint: "https://api.openai.com/v1/images/edits".to_string(),
            }
        }

        /// Set the endpoint (builder pattern).
        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }
    }

    impl ImageEditProvider for DallEEditProvider {
        fn edit_image(
            &self,
            image: &[u8],
            config: &ImageEditConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if image.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "image".to_string(),
                    reason: "source image cannot be empty".to_string(),
                }
                .into());
            }

            Err(MediaGenerationError::ProviderUnavailable {
                provider: "dall-e-edit".to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }
    }

    /// Stability AI image editing provider.
    #[derive(Debug, Clone)]
    pub struct StabilityEditProvider {
        pub api_key: String,
        pub endpoint: String,
    }

    impl StabilityEditProvider {
        /// Create a new Stability AI edit provider.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                endpoint: "https://api.stability.ai/v1/generation".to_string(),
            }
        }

        /// Set the endpoint (builder pattern).
        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }
    }

    impl ImageEditProvider for StabilityEditProvider {
        fn edit_image(
            &self,
            image: &[u8],
            config: &ImageEditConfig,
        ) -> Result<GeneratedImage, AiError> {
            config.validate()?;
            if image.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "image".to_string(),
                    reason: "source image cannot be empty".to_string(),
                }
                .into());
            }

            Err(MediaGenerationError::ProviderUnavailable {
                provider: "stability-edit".to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }
    }

    // ========================================================================
    // 4.3 — Video Generation Providers
    // ========================================================================

    // --- Enums --------------------------------------------------------------

    /// Video output resolution.
    #[derive(Debug, Clone, PartialEq)]
    pub enum VideoResolution {
        SD480,
        HD720,
        FHD1080,
        UHD4K,
    }

    impl fmt::Display for VideoResolution {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                VideoResolution::SD480 => write!(f, "480p"),
                VideoResolution::HD720 => write!(f, "720p"),
                VideoResolution::FHD1080 => write!(f, "1080p"),
                VideoResolution::UHD4K => write!(f, "4K"),
            }
        }
    }

    /// Aspect ratio for video output.
    #[derive(Debug, Clone, PartialEq)]
    pub enum AspectRatio {
        /// 1:1
        Square,
        /// 4:3
        Landscape,
        /// 3:4
        Portrait,
        /// 16:9
        Widescreen,
        /// Custom width:height ratio.
        Custom(u32, u32),
    }

    impl fmt::Display for AspectRatio {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                AspectRatio::Square => write!(f, "1:1"),
                AspectRatio::Landscape => write!(f, "4:3"),
                AspectRatio::Portrait => write!(f, "3:4"),
                AspectRatio::Widescreen => write!(f, "16:9"),
                AspectRatio::Custom(w, h) => write!(f, "{}:{}", w, h),
            }
        }
    }

    /// Video container / codec format.
    #[derive(Debug, Clone, PartialEq)]
    pub enum VideoFormat {
        Mp4,
        WebM,
        Gif,
    }

    impl fmt::Display for VideoFormat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                VideoFormat::Mp4 => write!(f, "mp4"),
                VideoFormat::WebM => write!(f, "webm"),
                VideoFormat::Gif => write!(f, "gif"),
            }
        }
    }

    /// Status of an async video generation job.
    #[derive(Debug, Clone, PartialEq)]
    pub enum VideoJobStatus {
        Queued,
        Processing { progress_pct: f32 },
        Complete,
        Failed { reason: String },
        Cancelled,
    }

    impl fmt::Display for VideoJobStatus {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                VideoJobStatus::Queued => write!(f, "queued"),
                VideoJobStatus::Processing { progress_pct } => {
                    write!(f, "processing ({:.1}%)", progress_pct)
                }
                VideoJobStatus::Complete => write!(f, "complete"),
                VideoJobStatus::Failed { reason } => write!(f, "failed: {}", reason),
                VideoJobStatus::Cancelled => write!(f, "cancelled"),
            }
        }
    }

    // --- Config & result structs --------------------------------------------

    /// Configuration for a video generation request.
    #[derive(Debug, Clone)]
    pub struct VideoGenConfig {
        /// Desired duration in seconds.
        pub duration_seconds: f32,
        /// Frames per second.
        pub fps: u32,
        /// Output resolution.
        pub resolution: VideoResolution,
        /// Aspect ratio.
        pub aspect_ratio: AspectRatio,
        /// Optional style descriptor.
        pub style: Option<String>,
        /// Deterministic seed.
        pub seed: Option<u64>,
        /// Optional image to use as first frame / reference.
        pub image_prompt: Option<Vec<u8>>,
    }

    impl Default for VideoGenConfig {
        fn default() -> Self {
            Self {
                duration_seconds: 4.0,
                fps: 24,
                resolution: VideoResolution::FHD1080,
                aspect_ratio: AspectRatio::Widescreen,
                style: None,
                seed: None,
                image_prompt: None,
            }
        }
    }

    impl VideoGenConfig {
        /// Validate the configuration.
        pub fn validate(&self) -> Result<(), AiError> {
            if self.duration_seconds <= 0.0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "duration_seconds".to_string(),
                    reason: "duration must be positive".to_string(),
                }
                .into());
            }
            if self.duration_seconds > 120.0 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "duration_seconds".to_string(),
                    reason: format!(
                        "duration {}s exceeds maximum 120s",
                        self.duration_seconds
                    ),
                }
                .into());
            }
            if self.fps == 0 || self.fps > 120 {
                return Err(MediaGenerationError::InvalidParams {
                    param: "fps".to_string(),
                    reason: format!("fps {} out of range [1, 120]", self.fps),
                }
                .into());
            }
            if let AspectRatio::Custom(w, h) = &self.aspect_ratio {
                if *w == 0 || *h == 0 {
                    return Err(MediaGenerationError::InvalidParams {
                        param: "aspect_ratio".to_string(),
                        reason: "custom aspect ratio components must be > 0".to_string(),
                    }
                    .into());
                }
            }
            Ok(())
        }
    }

    /// Handle for a submitted video generation job.
    #[derive(Debug, Clone)]
    pub struct VideoJob {
        /// Unique job identifier from the provider.
        pub job_id: String,
        /// Name of the provider handling this job.
        pub provider: String,
        /// Current status.
        pub status: VideoJobStatus,
        /// Unix timestamp (seconds) when the job was created.
        pub created_at: u64,
        /// Unix timestamp (seconds) of the last status update.
        pub updated_at: u64,
    }

    impl VideoJob {
        /// Create a new job in Queued state.
        pub fn new(job_id: impl Into<String>, provider: impl Into<String>) -> Self {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Self {
                job_id: job_id.into(),
                provider: provider.into(),
                status: VideoJobStatus::Queued,
                created_at: now,
                updated_at: now,
            }
        }

        /// Check whether the job is in a terminal state.
        pub fn is_terminal(&self) -> bool {
            matches!(
                self.status,
                VideoJobStatus::Complete
                    | VideoJobStatus::Failed { .. }
                    | VideoJobStatus::Cancelled
            )
        }
    }

    /// A successfully generated video.
    #[derive(Debug, Clone)]
    pub struct GeneratedVideo {
        /// Raw video bytes.
        pub bytes: Vec<u8>,
        /// Container format.
        pub format: VideoFormat,
        /// Actual duration in seconds.
        pub duration_seconds: f32,
        /// Output resolution.
        pub resolution: VideoResolution,
        /// Time taken to generate, in milliseconds.
        pub generation_time_ms: u64,
    }

    // --- Traits -------------------------------------------------------------

    /// Trait for video generation providers. Video generation is typically
    /// asynchronous: submit a job, poll status, then download when complete.
    pub trait VideoGenerationProvider: Send + Sync {
        /// Submit a new video generation job.
        fn submit_job(
            &self,
            prompt: &str,
            config: &VideoGenConfig,
        ) -> Result<VideoJob, AiError>;

        /// Check the current status of a previously submitted job.
        fn check_status(&self, job: &VideoJob) -> Result<VideoJobStatus, AiError>;

        /// Download the completed video. Should only be called when status is Complete.
        fn download_result(&self, job: &VideoJob) -> Result<GeneratedVideo, AiError>;

        /// Human-readable provider name.
        fn provider_name(&self) -> &str;
    }

    // --- Concrete providers -------------------------------------------------

    /// Runway Gen-3 video generation provider.
    #[derive(Debug, Clone)]
    pub struct RunwayProvider {
        pub api_key: String,
        pub model: String,
    }

    impl RunwayProvider {
        /// Create a new Runway provider.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: "gen-3".to_string(),
            }
        }

        /// Set the model (builder pattern).
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }
    }

    impl VideoGenerationProvider for RunwayProvider {
        fn submit_job(
            &self,
            prompt: &str,
            config: &VideoGenConfig,
        ) -> Result<VideoJob, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn check_status(&self, _job: &VideoJob) -> Result<VideoJobStatus, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn download_result(&self, _job: &VideoJob) -> Result<GeneratedVideo, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "runway"
        }
    }

    /// OpenAI Sora video generation provider.
    #[derive(Debug, Clone)]
    pub struct SoraProvider {
        pub api_key: String,
        pub model: String,
    }

    impl SoraProvider {
        /// Create a new Sora provider.
        pub fn new(api_key: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model: "sora".to_string(),
            }
        }

        /// Set the model (builder pattern).
        pub fn with_model(mut self, model: impl Into<String>) -> Self {
            self.model = model.into();
            self
        }
    }

    impl VideoGenerationProvider for SoraProvider {
        fn submit_job(
            &self,
            prompt: &str,
            config: &VideoGenConfig,
        ) -> Result<VideoJob, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn check_status(&self, _job: &VideoJob) -> Result<VideoJobStatus, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn download_result(&self, _job: &VideoJob) -> Result<GeneratedVideo, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "sora"
        }
    }

    /// Replicate video generation provider.
    #[derive(Debug, Clone)]
    pub struct ReplicateVideoProvider {
        pub api_key: String,
        pub model_id: String,
    }

    impl ReplicateVideoProvider {
        /// Create a new Replicate video provider.
        pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
            Self {
                api_key: api_key.into(),
                model_id: model_id.into(),
            }
        }
    }

    impl VideoGenerationProvider for ReplicateVideoProvider {
        fn submit_job(
            &self,
            prompt: &str,
            config: &VideoGenConfig,
        ) -> Result<VideoJob, AiError> {
            config.validate()?;
            if prompt.is_empty() {
                return Err(MediaGenerationError::InvalidParams {
                    param: "prompt".to_string(),
                    reason: "prompt cannot be empty".to_string(),
                }
                .into());
            }
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn check_status(&self, _job: &VideoJob) -> Result<VideoJobStatus, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn download_result(&self, _job: &VideoJob) -> Result<GeneratedVideo, AiError> {
            Err(MediaGenerationError::ProviderUnavailable {
                provider: self.provider_name().to_string(),
                reason: "HTTP client not available in this build".to_string(),
            }
            .into())
        }

        fn provider_name(&self) -> &str {
            "replicate"
        }
    }

    // --- Video Provider Router ----------------------------------------------

    /// Routes video generation requests to the correct provider by name.
    pub struct VideoProviderRouter {
        providers: HashMap<String, Box<dyn VideoGenerationProvider>>,
    }

    impl fmt::Debug for VideoProviderRouter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("VideoProviderRouter")
                .field("providers", &self.providers.keys().collect::<Vec<_>>())
                .finish()
        }
    }

    impl VideoProviderRouter {
        /// Create an empty router.
        pub fn new() -> Self {
            Self {
                providers: HashMap::new(),
            }
        }

        /// Register a provider under its `provider_name()`.
        pub fn register(&mut self, provider: Box<dyn VideoGenerationProvider>) {
            let name = provider.provider_name().to_string();
            self.providers.insert(name, provider);
        }

        /// Resolve a provider by name.
        pub fn resolve(&self, name: &str) -> Option<&dyn VideoGenerationProvider> {
            self.providers.get(name).map(|b| b.as_ref())
        }

        /// List all registered provider names.
        pub fn provider_names(&self) -> Vec<String> {
            self.providers.keys().cloned().collect()
        }

        /// Submit a video job via the named provider.
        pub fn submit(
            &self,
            provider_name: &str,
            prompt: &str,
            config: &VideoGenConfig,
        ) -> Result<VideoJob, AiError> {
            let provider = self.resolve(provider_name).ok_or_else(|| {
                AiError::MediaGeneration(MediaGenerationError::ProviderUnavailable {
                    provider: provider_name.to_string(),
                    reason: format!("no provider registered with name '{}'", provider_name),
                })
            })?;
            provider.submit_job(prompt, config)
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        // -- Mock helpers for testing ----------------------------------------

        /// A mock image generation provider that returns synthetic images.
        struct MockImageProvider {
            name: String,
            formats: Vec<ImageFormat>,
        }

        impl MockImageProvider {
            fn new(name: &str) -> Self {
                Self {
                    name: name.to_string(),
                    formats: vec![ImageFormat::Png, ImageFormat::Jpeg],
                }
            }
        }

        impl ImageGenerationProvider for MockImageProvider {
            fn generate_image(
                &self,
                prompt: &str,
                config: &ImageGenConfig,
            ) -> Result<GeneratedImage, AiError> {
                config.validate()?;
                if prompt.is_empty() {
                    return Err(MediaGenerationError::InvalidParams {
                        param: "prompt".to_string(),
                        reason: "prompt cannot be empty".to_string(),
                    }
                    .into());
                }
                Ok(GeneratedImage {
                    bytes: vec![0x89, 0x50, 0x4E, 0x47], // PNG magic bytes
                    format: ImageFormat::Png,
                    width: config.width,
                    height: config.height,
                    revised_prompt: Some(format!("mock revision of: {}", prompt)),
                    seed: config.seed,
                    generation_time_ms: 150,
                })
            }

            fn provider_name(&self) -> &str {
                &self.name
            }

            fn supported_formats(&self) -> Vec<ImageFormat> {
                self.formats.clone()
            }
        }

        /// A mock image edit provider that returns synthetic edited images.
        struct MockEditProvider;

        impl ImageEditProvider for MockEditProvider {
            fn edit_image(
                &self,
                image: &[u8],
                config: &ImageEditConfig,
            ) -> Result<GeneratedImage, AiError> {
                config.validate()?;
                if image.is_empty() {
                    return Err(MediaGenerationError::InvalidParams {
                        param: "image".to_string(),
                        reason: "source image cannot be empty".to_string(),
                    }
                    .into());
                }
                Ok(GeneratedImage {
                    bytes: vec![0xFF, 0xD8, 0xFF, 0xE0], // JPEG magic bytes
                    format: ImageFormat::Jpeg,
                    width: 512,
                    height: 512,
                    revised_prompt: None,
                    seed: None,
                    generation_time_ms: 200,
                })
            }
        }

        /// A mock video generation provider with controllable job state.
        struct MockVideoProvider {
            name: String,
            /// Controls what status `check_status` returns.
            simulated_status: VideoJobStatus,
        }

        impl MockVideoProvider {
            fn new(name: &str, status: VideoJobStatus) -> Self {
                Self {
                    name: name.to_string(),
                    simulated_status: status,
                }
            }
        }

        impl VideoGenerationProvider for MockVideoProvider {
            fn submit_job(
                &self,
                prompt: &str,
                config: &VideoGenConfig,
            ) -> Result<VideoJob, AiError> {
                config.validate()?;
                if prompt.is_empty() {
                    return Err(MediaGenerationError::InvalidParams {
                        param: "prompt".to_string(),
                        reason: "prompt cannot be empty".to_string(),
                    }
                    .into());
                }
                Ok(VideoJob::new(format!("mock-job-{}", prompt.len()), &self.name))
            }

            fn check_status(&self, _job: &VideoJob) -> Result<VideoJobStatus, AiError> {
                Ok(self.simulated_status.clone())
            }

            fn download_result(&self, job: &VideoJob) -> Result<GeneratedVideo, AiError> {
                if !matches!(self.simulated_status, VideoJobStatus::Complete) {
                    return Err(MediaGenerationError::GenerationFailed {
                        provider: self.name.clone(),
                        reason: format!(
                            "job {} is not complete (status: {})",
                            job.job_id, self.simulated_status
                        ),
                    }
                    .into());
                }
                Ok(GeneratedVideo {
                    bytes: vec![0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70], // MP4 ftyp
                    format: VideoFormat::Mp4,
                    duration_seconds: 4.0,
                    resolution: VideoResolution::FHD1080,
                    generation_time_ms: 30000,
                })
            }

            fn provider_name(&self) -> &str {
                &self.name
            }
        }

        // ====================================================================
        // 4.1 Image Generation Tests
        // ====================================================================

        #[test]
        fn test_image_gen_config_defaults() {
            let config = ImageGenConfig::default();
            assert_eq!(config.width, 1024);
            assert_eq!(config.height, 1024);
            assert_eq!(config.quality, ImageQuality::Standard);
            assert_eq!(config.num_images, 1);
            assert!(config.style.is_none());
            assert!(config.negative_prompt.is_none());
            assert!(config.seed.is_none());
            assert!(config.guidance_scale.is_none());
        }

        #[test]
        fn test_image_gen_config_validate_ok() {
            let config = ImageGenConfig::default();
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_image_gen_config_validate_zero_width() {
            let config = ImageGenConfig {
                width: 0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("width"));
        }

        #[test]
        fn test_image_gen_config_validate_zero_height() {
            let config = ImageGenConfig {
                height: 0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("height"));
        }

        #[test]
        fn test_image_gen_config_validate_oversized() {
            let config = ImageGenConfig {
                width: 9999,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("width"));
            assert!(err.to_string().contains("exceeds"));
        }

        #[test]
        fn test_image_gen_config_validate_zero_num_images() {
            let config = ImageGenConfig {
                num_images: 0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("num_images"));
        }

        #[test]
        fn test_image_gen_config_validate_too_many_images() {
            let config = ImageGenConfig {
                num_images: 11,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("num_images"));
        }

        #[test]
        fn test_image_gen_config_validate_bad_guidance() {
            let config = ImageGenConfig {
                guidance_scale: Some(51.0),
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("guidance_scale"));
        }

        #[test]
        fn test_image_style_display() {
            assert_eq!(ImageStyle::Natural.to_string(), "natural");
            assert_eq!(ImageStyle::Vivid.to_string(), "vivid");
            assert_eq!(ImageStyle::Artistic.to_string(), "artistic");
            assert_eq!(ImageStyle::Photorealistic.to_string(), "photorealistic");
            assert_eq!(ImageStyle::Anime.to_string(), "anime");
            assert_eq!(ImageStyle::Digital.to_string(), "digital");
        }

        #[test]
        fn test_image_quality_display() {
            assert_eq!(ImageQuality::Draft.to_string(), "draft");
            assert_eq!(ImageQuality::Standard.to_string(), "standard");
            assert_eq!(ImageQuality::HD.to_string(), "hd");
            assert_eq!(ImageQuality::UltraHD.to_string(), "ultra_hd");
        }

        #[test]
        fn test_image_format_display() {
            assert_eq!(ImageFormat::Png.to_string(), "png");
            assert_eq!(ImageFormat::Jpeg.to_string(), "jpeg");
            assert_eq!(ImageFormat::WebP.to_string(), "webp");
            assert_eq!(ImageFormat::Svg.to_string(), "svg");
        }

        #[test]
        fn test_mock_image_provider_generate() {
            let provider = MockImageProvider::new("test-img");
            let config = ImageGenConfig {
                width: 512,
                height: 512,
                seed: Some(42),
                ..Default::default()
            };
            let result = provider.generate_image("a cat", &config).unwrap();
            assert_eq!(result.width, 512);
            assert_eq!(result.height, 512);
            assert_eq!(result.format, ImageFormat::Png);
            assert_eq!(result.seed, Some(42));
            assert!(result.revised_prompt.is_some());
            assert!(result.generation_time_ms > 0);
            assert!(!result.bytes.is_empty());
        }

        #[test]
        fn test_mock_image_provider_empty_prompt() {
            let provider = MockImageProvider::new("test-img");
            let config = ImageGenConfig::default();
            let err = provider.generate_image("", &config).unwrap_err();
            assert!(err.to_string().contains("prompt"));
        }

        #[test]
        fn test_image_provider_name_and_formats() {
            let provider = MockImageProvider::new("my-provider");
            assert_eq!(provider.provider_name(), "my-provider");
            let formats = provider.supported_formats();
            assert!(formats.contains(&ImageFormat::Png));
            assert!(formats.contains(&ImageFormat::Jpeg));
        }

        // -- Concrete provider construction ----------------------------------

        #[test]
        fn test_dalle_provider_defaults() {
            let p = DallEProvider::new("sk-test");
            assert_eq!(p.api_key, "sk-test");
            assert_eq!(p.model, "dall-e-3");
            assert!(p.endpoint.contains("openai.com"));
            assert_eq!(p.provider_name(), "dall-e");
            let fmts = p.supported_formats();
            assert!(fmts.contains(&ImageFormat::Png));
        }

        #[test]
        fn test_dalle_provider_builder() {
            let p = DallEProvider::new("key")
                .with_model("dall-e-2")
                .with_endpoint("http://custom/endpoint");
            assert_eq!(p.model, "dall-e-2");
            assert_eq!(p.endpoint, "http://custom/endpoint");
        }

        #[test]
        fn test_sd_provider_defaults() {
            let p = StableDiffusionProvider::new("sk-stability");
            assert_eq!(p.api_key, "sk-stability");
            assert!(p.model.contains("stable-diffusion"));
            assert!(p.endpoint.contains("stability.ai"));
            assert_eq!(p.provider_name(), "stable-diffusion");
        }

        #[test]
        fn test_sd_provider_builder() {
            let p = StableDiffusionProvider::new("key")
                .with_model("sd-turbo")
                .with_endpoint("http://custom");
            assert_eq!(p.model, "sd-turbo");
            assert_eq!(p.endpoint, "http://custom");
        }

        #[test]
        fn test_flux_provider_defaults() {
            let p = FluxProvider::new("sk-flux");
            assert_eq!(p.api_key, "sk-flux");
            assert!(p.model.contains("flux"));
            assert!(p.endpoint.contains("bfl.ml"));
            assert_eq!(p.provider_name(), "flux");
        }

        #[test]
        fn test_flux_provider_builder() {
            let p = FluxProvider::new("key")
                .with_model("flux-pro")
                .with_endpoint("http://custom");
            assert_eq!(p.model, "flux-pro");
            assert_eq!(p.endpoint, "http://custom");
        }

        #[test]
        fn test_local_diffusion_provider() {
            let p = LocalDiffusionProvider::new("http://localhost:8188");
            assert_eq!(p.base_url, "http://localhost:8188");
            assert_eq!(p.provider_name(), "local-diffusion");
            let fmts = p.supported_formats();
            assert!(fmts.contains(&ImageFormat::Svg));
        }

        // -- Image Provider Router -------------------------------------------

        #[test]
        fn test_image_router_register_resolve() {
            let mut router = ImageProviderRouter::new();
            router.register(Box::new(MockImageProvider::new("alpha")));
            router.register(Box::new(MockImageProvider::new("beta")));

            assert!(router.resolve("alpha").is_some());
            assert!(router.resolve("beta").is_some());
            assert!(router.resolve("gamma").is_none());

            let names = router.provider_names();
            assert_eq!(names.len(), 2);
            assert!(names.contains(&"alpha".to_string()));
            assert!(names.contains(&"beta".to_string()));
        }

        #[test]
        fn test_image_router_generate_success() {
            let mut router = ImageProviderRouter::new();
            router.register(Box::new(MockImageProvider::new("mock")));

            let config = ImageGenConfig::default();
            let result = router.generate("mock", "a sunset", &config).unwrap();
            assert_eq!(result.width, 1024);
            assert_eq!(result.format, ImageFormat::Png);
        }

        #[test]
        fn test_image_router_generate_unknown_provider() {
            let router = ImageProviderRouter::new();
            let config = ImageGenConfig::default();
            let err = router.generate("nonexistent", "prompt", &config).unwrap_err();
            assert!(err.to_string().contains("nonexistent"));
        }

        #[test]
        fn test_image_router_debug() {
            let router = ImageProviderRouter::new();
            let debug = format!("{:?}", router);
            assert!(debug.contains("ImageProviderRouter"));
        }

        // ====================================================================
        // 4.2 Image Editing Tests
        // ====================================================================

        #[test]
        fn test_image_edit_config_defaults() {
            let config = ImageEditConfig::default();
            assert_eq!(config.operation, ImageEditOperation::Variation);
            assert!(config.prompt.is_empty());
            assert!(config.mask.is_none());
            assert!((config.strength - 0.75).abs() < f32::EPSILON);
            assert!((config.guidance_scale - 7.5).abs() < f32::EPSILON);
        }

        #[test]
        fn test_image_edit_config_validate_empty_prompt() {
            let config = ImageEditConfig::default();
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("prompt"));
        }

        #[test]
        fn test_image_edit_config_validate_bad_strength() {
            let config = ImageEditConfig {
                prompt: "edit me".to_string(),
                strength: 1.5,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("strength"));
        }

        #[test]
        fn test_image_edit_config_validate_bad_guidance() {
            let config = ImageEditConfig {
                prompt: "edit me".to_string(),
                guidance_scale: -1.0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("guidance_scale"));
        }

        #[test]
        fn test_image_edit_config_inpaint_requires_mask() {
            let config = ImageEditConfig {
                operation: ImageEditOperation::Inpaint,
                prompt: "fill this area".to_string(),
                mask: None,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("mask"));
            assert!(err.to_string().contains("inpaint"));
        }

        #[test]
        fn test_image_edit_config_outpaint_requires_mask() {
            let config = ImageEditConfig {
                operation: ImageEditOperation::Outpaint,
                prompt: "extend the scene".to_string(),
                mask: None,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("mask"));
        }

        #[test]
        fn test_image_edit_config_inpaint_with_mask_ok() {
            let config = ImageEditConfig {
                operation: ImageEditOperation::Inpaint,
                prompt: "fill with flowers".to_string(),
                mask: Some(vec![0xFF; 100]),
                ..Default::default()
            };
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_image_edit_operation_display() {
            assert_eq!(ImageEditOperation::Inpaint.to_string(), "inpaint");
            assert_eq!(ImageEditOperation::Outpaint.to_string(), "outpaint");
            assert_eq!(
                ImageEditOperation::StyleTransfer.to_string(),
                "style_transfer"
            );
            assert_eq!(ImageEditOperation::Upscale.to_string(), "upscale");
            assert_eq!(
                ImageEditOperation::RemoveBackground.to_string(),
                "remove_background"
            );
            assert_eq!(ImageEditOperation::Variation.to_string(), "variation");
        }

        #[test]
        fn test_mock_edit_provider_success() {
            let provider = MockEditProvider;
            let config = ImageEditConfig {
                operation: ImageEditOperation::Variation,
                prompt: "make it blue".to_string(),
                ..Default::default()
            };
            let image = vec![0x89, 0x50, 0x4E, 0x47]; // some image bytes
            let result = provider.edit_image(&image, &config).unwrap();
            assert_eq!(result.format, ImageFormat::Jpeg);
            assert_eq!(result.width, 512);
            assert_eq!(result.height, 512);
        }

        #[test]
        fn test_mock_edit_provider_empty_image() {
            let provider = MockEditProvider;
            let config = ImageEditConfig {
                prompt: "edit".to_string(),
                ..Default::default()
            };
            let err = provider.edit_image(&[], &config).unwrap_err();
            assert!(err.to_string().contains("image"));
        }

        #[test]
        fn test_dalle_edit_provider_construction() {
            let p = DallEEditProvider::new("sk-edit");
            assert_eq!(p.api_key, "sk-edit");
            assert!(p.endpoint.contains("edits"));
            let p2 = p.with_endpoint("http://custom/edits");
            assert_eq!(p2.endpoint, "http://custom/edits");
        }

        #[test]
        fn test_stability_edit_provider_construction() {
            let p = StabilityEditProvider::new("sk-stab");
            assert_eq!(p.api_key, "sk-stab");
            assert!(p.endpoint.contains("stability.ai"));
            let p2 = p.with_endpoint("http://custom/edit");
            assert_eq!(p2.endpoint, "http://custom/edit");
        }

        // ====================================================================
        // 4.3 Video Generation Tests
        // ====================================================================

        #[test]
        fn test_video_gen_config_defaults() {
            let config = VideoGenConfig::default();
            assert!((config.duration_seconds - 4.0).abs() < f32::EPSILON);
            assert_eq!(config.fps, 24);
            assert_eq!(config.resolution, VideoResolution::FHD1080);
            assert_eq!(config.aspect_ratio, AspectRatio::Widescreen);
            assert!(config.style.is_none());
            assert!(config.seed.is_none());
            assert!(config.image_prompt.is_none());
        }

        #[test]
        fn test_video_gen_config_validate_ok() {
            let config = VideoGenConfig::default();
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_video_gen_config_validate_zero_duration() {
            let config = VideoGenConfig {
                duration_seconds: 0.0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("duration"));
        }

        #[test]
        fn test_video_gen_config_validate_negative_duration() {
            let config = VideoGenConfig {
                duration_seconds: -5.0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("duration"));
        }

        #[test]
        fn test_video_gen_config_validate_too_long() {
            let config = VideoGenConfig {
                duration_seconds: 200.0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("duration"));
            assert!(err.to_string().contains("exceeds"));
        }

        #[test]
        fn test_video_gen_config_validate_zero_fps() {
            let config = VideoGenConfig {
                fps: 0,
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("fps"));
        }

        #[test]
        fn test_video_gen_config_validate_custom_aspect_zero() {
            let config = VideoGenConfig {
                aspect_ratio: AspectRatio::Custom(0, 9),
                ..Default::default()
            };
            let err = config.validate().unwrap_err();
            assert!(err.to_string().contains("aspect_ratio"));
        }

        #[test]
        fn test_video_resolution_display() {
            assert_eq!(VideoResolution::SD480.to_string(), "480p");
            assert_eq!(VideoResolution::HD720.to_string(), "720p");
            assert_eq!(VideoResolution::FHD1080.to_string(), "1080p");
            assert_eq!(VideoResolution::UHD4K.to_string(), "4K");
        }

        #[test]
        fn test_aspect_ratio_display() {
            assert_eq!(AspectRatio::Square.to_string(), "1:1");
            assert_eq!(AspectRatio::Landscape.to_string(), "4:3");
            assert_eq!(AspectRatio::Portrait.to_string(), "3:4");
            assert_eq!(AspectRatio::Widescreen.to_string(), "16:9");
            assert_eq!(AspectRatio::Custom(21, 9).to_string(), "21:9");
        }

        #[test]
        fn test_video_format_display() {
            assert_eq!(VideoFormat::Mp4.to_string(), "mp4");
            assert_eq!(VideoFormat::WebM.to_string(), "webm");
            assert_eq!(VideoFormat::Gif.to_string(), "gif");
        }

        #[test]
        fn test_video_job_status_display() {
            assert_eq!(VideoJobStatus::Queued.to_string(), "queued");
            assert_eq!(
                VideoJobStatus::Processing { progress_pct: 45.5 }.to_string(),
                "processing (45.5%)"
            );
            assert_eq!(VideoJobStatus::Complete.to_string(), "complete");
            assert_eq!(
                VideoJobStatus::Failed {
                    reason: "oom".to_string()
                }
                .to_string(),
                "failed: oom"
            );
            assert_eq!(VideoJobStatus::Cancelled.to_string(), "cancelled");
        }

        #[test]
        fn test_video_job_creation() {
            let job = VideoJob::new("job-123", "test-provider");
            assert_eq!(job.job_id, "job-123");
            assert_eq!(job.provider, "test-provider");
            assert_eq!(job.status, VideoJobStatus::Queued);
            assert!(job.created_at > 0);
            assert!(job.updated_at > 0);
        }

        #[test]
        fn test_video_job_is_terminal() {
            let mut job = VideoJob::new("j", "p");
            assert!(!job.is_terminal());

            job.status = VideoJobStatus::Processing { progress_pct: 50.0 };
            assert!(!job.is_terminal());

            job.status = VideoJobStatus::Complete;
            assert!(job.is_terminal());

            job.status = VideoJobStatus::Failed {
                reason: "error".to_string(),
            };
            assert!(job.is_terminal());

            job.status = VideoJobStatus::Cancelled;
            assert!(job.is_terminal());
        }

        // -- Video job lifecycle: queued -> processing -> complete ------------

        #[test]
        fn test_video_job_lifecycle_complete() {
            let provider = MockVideoProvider::new("mock-vid", VideoJobStatus::Complete);
            let config = VideoGenConfig::default();

            let job = provider.submit_job("a flying car", &config).unwrap();
            assert_eq!(job.status, VideoJobStatus::Queued);
            assert_eq!(job.provider, "mock-vid");

            let status = provider.check_status(&job).unwrap();
            assert_eq!(status, VideoJobStatus::Complete);

            let video = provider.download_result(&job).unwrap();
            assert_eq!(video.format, VideoFormat::Mp4);
            assert_eq!(video.resolution, VideoResolution::FHD1080);
            assert!((video.duration_seconds - 4.0).abs() < f32::EPSILON);
            assert!(!video.bytes.is_empty());
        }

        // -- Video job lifecycle: queued -> failed ----------------------------

        #[test]
        fn test_video_job_lifecycle_failed() {
            let provider = MockVideoProvider::new(
                "mock-vid-fail",
                VideoJobStatus::Failed {
                    reason: "GPU OOM".to_string(),
                },
            );
            let config = VideoGenConfig::default();

            let job = provider.submit_job("complex scene", &config).unwrap();
            assert_eq!(job.status, VideoJobStatus::Queued);

            let status = provider.check_status(&job).unwrap();
            match status {
                VideoJobStatus::Failed { reason } => assert_eq!(reason, "GPU OOM"),
                other => panic!("expected Failed, got {:?}", other),
            }

            // Downloading from a non-complete job should fail
            let err = provider.download_result(&job).unwrap_err();
            assert!(err.to_string().contains("not complete"));
        }

        #[test]
        fn test_video_provider_empty_prompt() {
            let provider = MockVideoProvider::new("mock", VideoJobStatus::Queued);
            let config = VideoGenConfig::default();
            let err = provider.submit_job("", &config).unwrap_err();
            assert!(err.to_string().contains("prompt"));
        }

        // -- Concrete video provider construction ----------------------------

        #[test]
        fn test_runway_provider_defaults() {
            let p = RunwayProvider::new("rw-key");
            assert_eq!(p.api_key, "rw-key");
            assert_eq!(p.model, "gen-3");
            assert_eq!(p.provider_name(), "runway");
        }

        #[test]
        fn test_runway_provider_builder() {
            let p = RunwayProvider::new("key").with_model("gen-4");
            assert_eq!(p.model, "gen-4");
        }

        #[test]
        fn test_sora_provider_defaults() {
            let p = SoraProvider::new("sora-key");
            assert_eq!(p.api_key, "sora-key");
            assert_eq!(p.model, "sora");
            assert_eq!(p.provider_name(), "sora");
        }

        #[test]
        fn test_sora_provider_builder() {
            let p = SoraProvider::new("key").with_model("sora-turbo");
            assert_eq!(p.model, "sora-turbo");
        }

        #[test]
        fn test_replicate_provider_defaults() {
            let p = ReplicateVideoProvider::new("rep-key", "stability/svd");
            assert_eq!(p.api_key, "rep-key");
            assert_eq!(p.model_id, "stability/svd");
            assert_eq!(p.provider_name(), "replicate");
        }

        // -- Video Provider Router -------------------------------------------

        #[test]
        fn test_video_router_register_resolve() {
            let mut router = VideoProviderRouter::new();
            router.register(Box::new(MockVideoProvider::new(
                "vid-a",
                VideoJobStatus::Queued,
            )));
            router.register(Box::new(MockVideoProvider::new(
                "vid-b",
                VideoJobStatus::Complete,
            )));

            assert!(router.resolve("vid-a").is_some());
            assert!(router.resolve("vid-b").is_some());
            assert!(router.resolve("vid-c").is_none());

            let names = router.provider_names();
            assert_eq!(names.len(), 2);
        }

        #[test]
        fn test_video_router_submit_success() {
            let mut router = VideoProviderRouter::new();
            router.register(Box::new(MockVideoProvider::new(
                "mock-router",
                VideoJobStatus::Queued,
            )));
            let config = VideoGenConfig::default();
            let job = router.submit("mock-router", "a dancing robot", &config).unwrap();
            assert_eq!(job.provider, "mock-router");
            assert_eq!(job.status, VideoJobStatus::Queued);
        }

        #[test]
        fn test_video_router_submit_unknown_provider() {
            let router = VideoProviderRouter::new();
            let config = VideoGenConfig::default();
            let err = router.submit("nonexistent", "prompt", &config).unwrap_err();
            assert!(err.to_string().contains("nonexistent"));
        }

        #[test]
        fn test_video_router_debug() {
            let router = VideoProviderRouter::new();
            let debug = format!("{:?}", router);
            assert!(debug.contains("VideoProviderRouter"));
        }
    }
}

// Re-export everything from the inner module when the feature is enabled.
#[cfg(feature = "media-generation")]
pub use inner::*;
