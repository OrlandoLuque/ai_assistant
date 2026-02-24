//! Media generation demo.
//!
//! Run with: cargo run --example media_gen_demo --features media-generation
//!
//! Demonstrates image and video generation configuration,
//! provider setup, and router usage.
//! Note: actual generation requires API keys for external services.

use ai_assistant::{
    ImageGenConfig, DallEProvider, ImageQuality, ImageProviderRouter,
    VideoGenConfig, RunwayProvider, VideoResolution, AspectRatio, VideoProviderRouter,
};

fn main() {
    println!("=== Media Generation Demo ===\n");

    // 1. Image generation config
    let image_config = ImageGenConfig {
        width: 1024,
        height: 1024,
        style: None,
        quality: ImageQuality::HD,
        num_images: 1,
        negative_prompt: Some("blurry, low quality".to_string()),
        seed: Some(42),
        guidance_scale: Some(7.5),
    };

    println!("Image Generation Config:");
    println!("  Size: {}x{}", image_config.width, image_config.height);
    println!("  Quality: {:?}", image_config.quality);
    println!("  Seed: {:?}", image_config.seed);
    println!("  Guidance scale: {:?}", image_config.guidance_scale);

    // 2. DALL-E provider
    let dalle = DallEProvider::new("sk-demo-key-not-real");
    println!("\nDall-E Provider:");
    println!("  Model: {}", dalle.model);
    println!("  Endpoint: {}", dalle.endpoint);

    // 3. Image provider router
    let router = ImageProviderRouter::new();
    println!("\nImage Router: {:?}", router);

    // 4. Video generation config
    let video_config = VideoGenConfig {
        duration_seconds: 4.0,
        fps: 24,
        resolution: VideoResolution::HD720,
        aspect_ratio: AspectRatio::Widescreen,
        style: Some("cinematic".to_string()),
        seed: None,
        image_prompt: None,
    };

    println!("\nVideo Generation Config:");
    println!("  Duration: {}s @ {} FPS", video_config.duration_seconds, video_config.fps);
    println!("  Resolution: {:?}", video_config.resolution);
    println!("  Aspect ratio: {:?}", video_config.aspect_ratio);
    println!("  Style: {:?}", video_config.style);

    // 5. Runway provider
    let runway = RunwayProvider::new("rk-demo-key-not-real");
    println!("\nRunway Provider:");
    println!("  Model: {}", runway.model);

    // 6. Video provider router
    let video_router = VideoProviderRouter::new();
    println!("\nVideo Router: {:?}", video_router);

    println!("\n=== Done (no API keys needed for demo) ===");
}
