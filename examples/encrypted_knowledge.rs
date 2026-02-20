//! Encrypted knowledge package (.kpkg) example.
//!
//! Run with: cargo run --example encrypted_knowledge --features rag
//!
//! Demonstrates creating an encrypted knowledge package with the KpkgBuilder,
//! adding documents with priorities and metadata, then decrypting it with
//! KpkgReader and inspecting the contents. Also shows custom passphrase
//! encryption and manifest-only reading.

use ai_assistant::{AppKeyProvider, CustomKeyProvider, KpkgBuilder, KpkgError, KpkgReader};

fn main() {
    println!("=== Encrypted Knowledge Packages (.kpkg) Demo ===\n");

    // -------------------------------------------------------------------------
    // Part 1: Create a package with the default application key
    // -------------------------------------------------------------------------

    println!("--- Part 1: Build and read with application key ---\n");

    let encrypted = KpkgBuilder::<AppKeyProvider>::with_app_key()
        .name("Spacecraft Database")
        .description("Technical specifications for fictional spacecraft")
        .version("1.0.0")
        .default_priority(5)
        // Add a system prompt and persona for RAG usage
        .system_prompt("You are a spacecraft technical expert. Answer questions using the provided specifications.")
        .persona("Precise, technical, always cite the source document.")
        // Add few-shot examples
        .add_example(
            "What shields does the Sabre have?",
            "The Sabre is equipped with dual S3 shield generators providing 2x coverage.",
        )
        .add_example_with_category(
            "Compare the Aurora and Gladius",
            "The Aurora is a civilian starter ship while the Gladius is a dedicated light fighter.",
            "comparison",
        )
        // Configure RAG settings
        .chunk_size(256)
        .top_k(5)
        .min_relevance(0.4)
        // Metadata
        .author("Demo Author")
        .language("en")
        .license("CC-BY-4.0")
        .add_tag("spacecraft")
        .add_tag("specifications")
        // Add documents with priorities (higher = more important)
        .add_document(
            "ships/sabre.md",
            "# Aegis Sabre\n\nThe Sabre is a stealth fighter manufactured by Aegis Dynamics.\n\n## Specifications\n- Role: Stealth Fighter\n- Shields: 2x S3\n- Weapons: 4x S3 hardpoints\n- Max speed: 1220 m/s",
            Some(10),
        )
        .add_document(
            "ships/gladius.md",
            "# Aegis Gladius\n\nThe Gladius is a light fighter with excellent maneuverability.\n\n## Specifications\n- Role: Light Fighter\n- Shields: 1x S3\n- Weapons: 3x S3 hardpoints\n- Max speed: 1245 m/s",
            Some(8),
        )
        .add_document(
            "ships/aurora.md",
            "# RSI Aurora\n\nThe Aurora is an introductory spacecraft suitable for new pilots.\n\n## Specifications\n- Role: Starter / Multi-role\n- Shields: 1x S2\n- Weapons: 2x S2 hardpoints\n- Max speed: 1150 m/s",
            None, // uses default priority (5)
        )
        .build()
        .expect("Failed to build encrypted package");

    println!("Encrypted package size: {} bytes", encrypted.len());

    // Read it back
    let reader = KpkgReader::<AppKeyProvider>::with_app_key();
    let documents = reader.read(&encrypted).expect("Failed to decrypt package");

    println!("Documents extracted: {}", documents.len());
    for doc in &documents {
        println!(
            "  {} ({} bytes, priority {})",
            doc.path,
            doc.content.len(),
            doc.priority
        );
    }

    // Read just the manifest (faster, skips document extraction)
    let manifest = reader
        .read_manifest_only(&encrypted)
        .expect("Failed to read manifest");

    println!("\nManifest:");
    println!("  Name:        {}", manifest.name);
    println!("  Description: {}", manifest.description);
    println!("  Version:     {}", manifest.version);
    if let Some(ref prompt) = manifest.system_prompt {
        println!("  System prompt: {}...", &prompt[..prompt.len().min(60)]);
    }
    if let Some(ref persona) = manifest.persona {
        println!("  Persona: {}", persona);
    }
    println!("  Examples: {} defined", manifest.examples.len());
    if let Some(ref rag) = manifest.rag_config {
        println!(
            "  RAG config: chunk_size={:?}, top_k={:?}",
            rag.chunk_size, rag.top_k
        );
    }
    if let Some(ref meta) = manifest.metadata {
        println!("  Author:   {:?}", meta.author);
        println!("  Language: {:?}", meta.language);
        println!("  Tags:     {:?}", meta.tags);
    }

    // -------------------------------------------------------------------------
    // Part 2: Custom passphrase encryption
    // -------------------------------------------------------------------------

    println!("\n--- Part 2: Custom passphrase encryption ---\n");

    let passphrase = "my-secret-passphrase-2024";

    let custom_encrypted = KpkgBuilder::with_key_provider(CustomKeyProvider::new(passphrase))
        .name("Secret Notes")
        .add_document(
            "notes.md",
            "# Top Secret\n\nThis content is passphrase-protected.",
            None,
        )
        .build()
        .expect("Failed to build custom package");

    println!("Custom-encrypted package: {} bytes", custom_encrypted.len());

    // Decrypt with the correct passphrase
    let custom_reader = KpkgReader::with_key_provider(CustomKeyProvider::new(passphrase));
    let custom_docs = custom_reader
        .read(&custom_encrypted)
        .expect("Failed to decrypt with correct passphrase");
    println!(
        "Decrypted with correct passphrase: {} doc(s)",
        custom_docs.len()
    );
    println!(
        "  Content preview: {}...",
        &custom_docs[0].content[..custom_docs[0].content.len().min(40)]
    );

    // Try with wrong passphrase -- should fail
    let wrong_reader = KpkgReader::with_key_provider(CustomKeyProvider::new("wrong-passphrase"));
    match wrong_reader.read(&custom_encrypted) {
        Err(KpkgError::DecryptionFailed) => {
            println!("Wrong passphrase correctly rejected: DecryptionFailed");
        }
        Err(e) => println!("Unexpected error: {}", e),
        Ok(_) => println!("ERROR: should not decrypt with wrong passphrase!"),
    }

    // -------------------------------------------------------------------------
    // Part 3: Read with manifest (documents + manifest in one call)
    // -------------------------------------------------------------------------

    println!("\n--- Part 3: Read with manifest ---\n");

    let (docs, full_manifest) = reader
        .read_with_manifest(&encrypted)
        .expect("Failed to read with manifest");

    println!(
        "Package '{}' v{}",
        full_manifest.name, full_manifest.version
    );
    println!("Documents: {}", docs.len());
    for ex in &full_manifest.examples {
        println!(
            "  Example: Q=\"{}\" -> A=\"{}...\"",
            ex.input,
            &ex.output[..ex.output.len().min(50)]
        );
    }

    // -------------------------------------------------------------------------
    // Part 4: Error handling
    // -------------------------------------------------------------------------

    println!("\n--- Part 4: Error handling ---\n");

    // Empty package
    match KpkgBuilder::<AppKeyProvider>::with_app_key().build() {
        Err(KpkgError::EmptyPackage) => println!("Empty package error: correct"),
        other => println!("Unexpected: {:?}", other),
    }

    // Data too short
    match reader.read(&[0u8; 10]) {
        Err(KpkgError::DataTooShort) => println!("Data too short error: correct"),
        other => println!("Unexpected: {:?}", other),
    }

    println!("\nDone.");
}
