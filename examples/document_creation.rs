//! Document creation demo: container-based document pipeline.
//!
//! Run with: cargo run --example document_creation --features containers
//!
//! Demonstrates:
//! - Creating a DocumentPipeline with default config
//! - Creating a document from Markdown content (PDF)
//! - Creating a document from HTML content (DOCX)
//! - Listing supported output formats and conversions
//! - Using SharedFolder for document output

use ai_assistant::{
    ContainerExecutor, DocumentOutputFormat, DocumentPipeline, DocumentPipelineConfig,
    DocumentRequest, SharedFolder, SourceFormat,
};
use std::sync::{Arc, RwLock};

fn main() -> anyhow::Result<()> {
    println!("=== ai_assistant document pipeline demo ===\n");

    // ── 1. DocumentPipelineConfig defaults ───────────────────────────────

    println!("--- DocumentPipelineConfig ---");
    let config = DocumentPipelineConfig::default();
    println!("  Docker image: {}", config.image);
    println!("  Timeout: {}s", config.timeout.as_secs());
    println!("  Reuse container: {}", config.reuse_container);

    // ── 2. Supported conversions ─────────────────────────────────────────

    println!("\n--- Supported conversions ---");
    let conversions = DocumentPipeline::supported_conversions();
    println!("  Total supported format pairs: {}", conversions.len());
    for (src, out) in &conversions {
        println!("    {:?} -> {:?} (.{})", src, out, out.extension());
    }

    // ── 3. Output formats and MIME types ─────────────────────────────────

    println!("\n--- Output formats ---");
    let formats = [
        DocumentOutputFormat::Pdf,
        DocumentOutputFormat::Docx,
        DocumentOutputFormat::Pptx,
        DocumentOutputFormat::Html,
        DocumentOutputFormat::Latex,
        DocumentOutputFormat::Epub,
        DocumentOutputFormat::Odt,
    ];
    for fmt in &formats {
        println!(
            "  {:?}: extension=.{}, mime={}",
            fmt,
            fmt.extension(),
            fmt.mime_type()
        );
    }

    // ── 4. Build document requests (Markdown -> PDF, HTML -> DOCX) ──────

    println!("\n--- Document requests ---");

    let md_content = r#"# Quarterly Report

## Summary

This report covers the key achievements of Q1 2026.

- Launched container sandbox module
- Added document pipeline support
- Integrated SharedFolder with cloud sync

## Conclusion

The project is on track for the v1.0 release.
"#;

    let pdf_request = DocumentRequest::new(md_content, DocumentOutputFormat::Pdf)
        .with_name("quarterly_report")
        .with_metadata("title", "Quarterly Report Q1 2026")
        .with_metadata("author", "AI Assistant")
        .with_metadata("date", "2026-02-21");

    println!("  Request 1: Markdown -> PDF");
    println!("    Name: {}", pdf_request.output_name);
    println!("    Source: {:?}", pdf_request.source_format);
    println!("    Output: {:?}", pdf_request.output_format);
    println!("    Metadata keys: {:?}", pdf_request.metadata.keys().collect::<Vec<_>>());

    let html_content = r#"<html>
<body>
  <h1>Meeting Notes</h1>
  <p>Date: 2026-02-21</p>
  <ul>
    <li>Reviewed container sandbox architecture</li>
    <li>Approved document pipeline design</li>
    <li>Next steps: integration tests</li>
  </ul>
</body>
</html>"#;

    let docx_request = DocumentRequest::new(html_content, DocumentOutputFormat::Docx)
        .with_name("meeting_notes")
        .with_source_format(SourceFormat::Html)
        .with_metadata("title", "Meeting Notes");

    println!("\n  Request 2: HTML -> DOCX");
    println!("    Name: {}", docx_request.output_name);
    println!("    Source: {:?}", docx_request.source_format);
    println!("    Output: {:?}", docx_request.output_format);

    // ── 5. SharedFolder for document output ──────────────────────────────

    println!("\n--- SharedFolder for output ---");
    let folder = SharedFolder::temp()?;
    println!("  Temp folder: {}", folder.host_path().display());
    println!("  Bind mount spec: {}", folder.bind_mount_spec());

    // ── 6. Create pipeline and generate documents (requires Docker) ─────

    println!("\n--- Document generation ---");
    let docker_available = ContainerExecutor::is_docker_available();
    println!("  Docker available: {}", docker_available);

    if docker_available {
        let executor = Arc::new(RwLock::new(
            ContainerExecutor::new(ai_assistant::ContainerConfig::default())?,
        ));
        let mut pipeline = DocumentPipeline::new(
            DocumentPipelineConfig::default(),
            executor,
            folder,
        );

        // Generate PDF from Markdown.
        match pipeline.create(&pdf_request) {
            Ok(result) => {
                println!("  [ok] Created {}", result.filename);
                println!("       Path: {}", result.output_path.display());
                println!("       Size: {} bytes", result.size_bytes);
                println!("       Duration: {:?}", result.duration);
            }
            Err(e) => println!("  [err] PDF generation failed: {}", e),
        }

        // Generate DOCX from HTML.
        match pipeline.create(&docx_request) {
            Ok(result) => {
                println!("  [ok] Created {}", result.filename);
                println!("       Path: {}", result.output_path.display());
                println!("       Size: {} bytes", result.size_bytes);
                println!("       Duration: {:?}", result.duration);
            }
            Err(e) => println!("  [err] DOCX generation failed: {}", e),
        }

        pipeline.cleanup();
    } else {
        println!("  [skip] Docker not available. Showing API usage without execution:");
        println!("    let executor = Arc::new(RwLock::new(ContainerExecutor::new(config)?));");
        println!("    let mut pipeline = DocumentPipeline::new(config, executor, folder);");
        println!("    let result = pipeline.create(&pdf_request)?;");
        println!("    println!(\"Created {{}} ({{}} bytes)\", result.filename, result.size_bytes);");
    }

    println!("\n=== Done ===");
    Ok(())
}
