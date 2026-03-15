//! Example: vision_demo -- Demonstrates vision/multimodal capabilities of ai_assistant.
//!
//! Run with: cargo run --example vision_demo --features vision
//!
//! This example showcases image input creation (from URL, bytes, base64),
//! vision messages, capabilities checking, image preprocessing,
//! batch image handling, and API format conversion.

use ai_assistant::{
    ImageBatch, ImageData, ImageDetail, ImageInput, ImagePreprocessor, VisionCapabilities,
    VisionMessage,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Vision / Multimodal Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. Creating Image Inputs
    // ------------------------------------------------------------------
    println!("--- 1. Creating Image Inputs ---\n");

    // From URL
    let url_image = ImageInput::from_url("https://example.com/photo.png");
    println!("  URL image:");
    println!("    media_type : {}", url_image.media_type);
    println!("    detail     : {:?}", url_image.detail);
    match &url_image.data {
        ImageData::Url(u) => println!("    source     : URL({})", u),
        ImageData::Base64(_) => println!("    source     : Base64"),
        _ => println!("    source     : other"),
    }

    // From raw bytes (simulated JPEG header)
    let fake_jpeg: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
    let bytes_image = ImageInput::from_bytes(&fake_jpeg, "image/jpeg");
    println!("\n  Bytes image:");
    println!("    media_type : {}", bytes_image.media_type);
    println!("    data_url   : {}...", &bytes_image.to_data_url()[..40]);

    // From base64 string
    let b64_image = ImageInput::from_base64("SGVsbG8gV29ybGQ=", "image/png");
    println!("\n  Base64 image:");
    println!("    media_type : {}", b64_image.media_type);
    println!("    detail     : {:?}", b64_image.detail);

    // With detail level override
    let high_detail = ImageInput::from_url("https://example.com/diagram.png")
        .with_detail(ImageDetail::High);
    println!("\n  High-detail image:");
    println!("    detail     : {:?}", high_detail.detail);
    println!("    est tokens : {}", high_detail.estimate_tokens());

    let low_detail = ImageInput::from_url("https://example.com/thumb.jpg")
        .with_detail(ImageDetail::Low);
    println!("  Low-detail image:");
    println!("    detail     : {:?}", low_detail.detail);
    println!("    est tokens : {}", low_detail.estimate_tokens());
    println!();

    // ------------------------------------------------------------------
    // 2. Vision Messages
    // ------------------------------------------------------------------
    println!("--- 2. Vision Messages ---\n");

    let user_msg = VisionMessage::user(
        "What objects do you see in this image?",
        vec![
            ImageInput::from_url("https://example.com/scene.jpg"),
        ],
    );
    println!("  User message:");
    println!("    role   : {}", user_msg.role);
    println!("    text   : \"{}\"", user_msg.text);
    println!("    images : {}", user_msg.images.len());

    // Multi-image message using builder pattern
    let multi_msg = VisionMessage::user("Compare these two images:", vec![])
        .with_image(ImageInput::from_url("https://example.com/before.jpg"))
        .with_image(ImageInput::from_url("https://example.com/after.jpg"));
    println!("\n  Multi-image message:");
    println!("    text   : \"{}\"", multi_msg.text);
    println!("    images : {}", multi_msg.images.len());

    // System and assistant messages (no images)
    let sys_msg = VisionMessage::system("You are a visual analysis assistant.");
    let asst_msg = VisionMessage::assistant("I can see a cat sitting on a windowsill.");
    println!("\n  System message  : role={}, images={}", sys_msg.role, sys_msg.images.len());
    println!("  Assistant message: role={}, images={}", asst_msg.role, asst_msg.images.len());

    // Convert to API formats
    let openai_fmt = user_msg.to_openai_format();
    println!("\n  OpenAI format (role): {}", openai_fmt["role"]);
    println!("  OpenAI content is array: {}", openai_fmt["content"].is_array());

    let ollama_fmt = user_msg.to_ollama_format();
    println!("  Ollama format (role): {}", ollama_fmt["role"]);
    println!();

    // ------------------------------------------------------------------
    // 3. Vision Capabilities Checker
    // ------------------------------------------------------------------
    println!("--- 3. Vision Capabilities ---\n");

    let caps = VisionCapabilities::new();

    let models = [
        "gpt-4o",
        "gpt-4-vision",
        "claude-3-opus",
        "llava",
        "llama3",
        "gpt-3.5-turbo",
        "mistral",
        "moondream",
        "qwen-vl-plus",
    ];

    println!("  Model                Vision?  Format          Max Images");
    println!("  -------------------  -------  --------------  ----------");
    for model in &models {
        let has_vision = caps.supports_vision(model);
        let format = caps.recommended_format(model);
        let max = caps.max_images(model);
        println!(
            "  {:<21} {:<8} {:<15} {}",
            model,
            if has_vision { "yes" } else { "no" },
            format,
            max,
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 4. Image Preprocessing
    // ------------------------------------------------------------------
    println!("--- 4. Image Preprocessing ---\n");

    let preprocessor = ImagePreprocessor::default();
    println!("  Default preprocessor:");
    println!("    max_width  : {}px", preprocessor.max_width);
    println!("    max_height : {}px", preprocessor.max_height);
    println!("    format     : {}", preprocessor.target_format);
    println!("    quality    : {}", preprocessor.quality);

    let low = ImagePreprocessor::low_detail();
    println!("\n  Low-detail preprocessor:");
    println!("    max_width  : {}px", low.max_width);
    println!("    max_height : {}px", low.max_height);

    let high = ImagePreprocessor::high_detail();
    println!("\n  High-detail preprocessor:");
    println!("    max_width  : {}px", high.max_width);
    println!("    format     : {}", high.target_format);

    // Dimension calculations
    let test_sizes: Vec<(u32, u32)> = vec![
        (800, 600),
        (1920, 1080),
        (4000, 3000),
        (512, 512),
        (6000, 2000),
    ];
    println!("\n  Resize calculations (default preprocessor):");
    println!("    Input         Needs resize?  Output");
    println!("    ------------  -------------  ----------");
    for (w, h) in &test_sizes {
        let needs = preprocessor.needs_resize(*w, *h);
        let (nw, nh) = preprocessor.calculate_dimensions(*w, *h);
        println!(
            "    {:>5}x{:<5}  {:<13}  {}x{}",
            w, h,
            if needs { "yes" } else { "no" },
            nw, nh,
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 5. Image Batch Processing
    // ------------------------------------------------------------------
    println!("--- 5. Image Batch Processing ---\n");

    let mut batch = ImageBatch::new(4);

    println!("  Batch capacity: 4");
    println!("  Remaining    : {}", batch.remaining());

    let added1 = batch.add_url("https://example.com/img1.jpg");
    let added2 = batch.add_url("https://example.com/img2.png");
    let added3 = batch.add(ImageInput::from_bytes(&[0xAA, 0xBB], "image/bmp"));
    let added4 = batch.add_url("https://example.com/img4.gif");
    let added5 = batch.add_url("https://example.com/img5.webp");

    println!("  Added img1   : {}", added1);
    println!("  Added img2   : {}", added2);
    println!("  Added img3   : {}", added3);
    println!("  Added img4   : {}", added4);
    println!("  Added img5   : {} (should be false -- batch is full)", added5);

    println!("  Is full      : {}", batch.is_full());
    println!("  Remaining    : {}", batch.remaining());
    println!("  Image count  : {}", batch.images().len());
    println!("  Est. tokens  : {}", batch.estimate_tokens());

    // Take images out of the batch
    let images = batch.take();
    println!("  Taken images : {}", images.len());

    // ------------------------------------------------------------------
    // 6. Token Estimation Summary
    // ------------------------------------------------------------------
    println!("\n--- 6. Token Estimation ---\n");

    let detail_levels = [ImageDetail::Low, ImageDetail::High, ImageDetail::Auto];
    for detail in &detail_levels {
        let img = ImageInput::from_url("https://example.com/sample.jpg")
            .with_detail(*detail);
        println!("  {:?} detail -> ~{} tokens", detail, img.estimate_tokens());
    }

    // OpenAI format for a high-detail image
    let sample = ImageInput::from_url("https://example.com/chart.png")
        .with_detail(ImageDetail::High);
    let fmt = sample.to_openai_format();
    println!("\n  OpenAI format for High-detail image:");
    println!("    type: {}", fmt["type"]);
    println!("    detail: {}", fmt["image_url"]["detail"]);

    println!("\n==========================================================");
    println!("  Vision demo completed successfully.");
    println!("==========================================================");
}
