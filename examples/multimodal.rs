//! Example: Multi-Modal RAG Pipeline
//!
//! Demonstrates ingesting text and image chunks, querying the pipeline,
//! and inspecting scored results with modality breakdowns.
//!
//! Run with: cargo run --example multimodal --features full

use ai_assistant::{
    MultiModalConfig, MultiModalDocument, MultiModalPipeline,
    ImageCaptionExtractor,
};

fn main() {
    println!("=== Multi-Modal RAG Pipeline Demo ===\n");

    // 1. Build a pipeline with custom config
    let mut config = MultiModalConfig::default();
    config.text_weight = 0.7;
    config.image_weight = 0.3;
    config.top_k = 5;
    config.min_score = 0.05;
    let mut pipeline = MultiModalPipeline::new(config);

    // 2. Add text chunks
    pipeline.add_text(
        "Rust is a systems programming language focused on safety and performance",
        Some("rust-docs"),
    );
    pipeline.add_text(
        "Python is widely used for data science and machine learning applications",
        Some("python-docs"),
    );
    pipeline.add_text(
        "The sunset over the Pacific Ocean creates beautiful orange and purple colors",
        Some("travel-blog"),
    );

    // 3. Add image chunks
    pipeline.add_image(
        "rust_logo.png",
        "The Rust programming language logo with a gear and crab mascot",
        Some("rust-docs"),
    );
    pipeline.add_image(
        "sunset_beach.jpg",
        "A beautiful sunset over the ocean with waves crashing on the beach",
        Some("travel-blog"),
    );
    pipeline.add_image(
        "python_chart.png",
        "A data science visualization chart created with Python matplotlib",
        Some("python-docs"),
    );

    // 4. Add a document parsed from HTML
    let html = r#"<p>WebAssembly allows running Rust code in the browser</p>
    <img src="wasm_diagram.png" alt="WebAssembly compilation pipeline diagram">"#;
    let doc = MultiModalDocument::from_html(html);
    pipeline.add_document(doc);

    println!("Pipeline loaded with {} chunks\n", pipeline.chunk_count());

    // 5. Query: "Rust programming"
    let result = pipeline.query("Rust programming language");
    println!("--- Query: 'Rust programming language' ---");
    println!("{}", result.summary());
    for (i, chunk) in result.chunks.iter().enumerate() {
        let label = match chunk.modality {
            ai_assistant::ModalityType::Image => {
                format!("[IMG] {}", chunk.image_ref.as_deref().unwrap_or("?"))
            }
            _ => format!("[TXT] {:.60}", chunk.content),
        };
        println!("  {}. (score: {:.4}) {}", i + 1, chunk.score, label);
    }

    // 6. Query: "sunset ocean"
    let result2 = pipeline.query("sunset ocean beach");
    println!("\n--- Query: 'sunset ocean beach' ---");
    println!("{}", result2.summary());
    println!("Has images: {}", result2.has_images());
    for chunk in result2.top_text(2) {
        println!("  Text: {:.70}", chunk.content);
    }
    for chunk in result2.top_images(2) {
        println!("  Image: {} (caption: {})",
            chunk.image_ref.as_deref().unwrap_or("?"),
            chunk.caption.as_deref().unwrap_or("none"),
        );
    }

    // 7. ImageCaptionExtractor utility
    println!("\n--- Caption Extraction ---");
    let filenames = ["sunset_over_ocean.jpg", "rust-logo-512x512.png", "IMG_2024.heic"];
    for name in &filenames {
        let caption = ImageCaptionExtractor::from_filename(name);
        println!("  {} -> {:?}", name, caption);
    }

    println!("\nDone!");
}
