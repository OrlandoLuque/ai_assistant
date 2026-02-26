//! Example: integrity_check_demo -- Demonstrates binary integrity verification.
//!
//! Run with: cargo run --example integrity_check_demo
//!
//! This example showcases SHA256 hashing, integrity configuration,
//! and the startup verification flow for production deployments.

use ai_assistant::{
    hash_bytes, hash_file, startup_integrity_check, IntegrityChecker, IntegrityConfig,
    IntegrityResult,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- Binary Integrity Check Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. SHA256 Hashing
    // ------------------------------------------------------------------
    println!("--- 1. SHA256 Hashing ---\n");

    let data = b"Hello, world!";
    let hash = hash_bytes(data);
    println!("  hash_bytes(\"Hello, world!\") = {}", hash);
    println!("  Length: {} chars (SHA256 hex)", hash.len());

    // Hash empty data
    let empty_hash = hash_bytes(b"");
    println!("  hash_bytes(\"\")              = {}", empty_hash);

    // Hash the current executable
    let exe_path = std::env::current_exe().expect("current exe path");
    println!("\n  Current executable: {}", exe_path.display());
    match hash_file(&exe_path) {
        Ok(exe_hash) => {
            println!("  SHA256: {}", exe_hash);
            println!("  (This hash changes with every build)");
        }
        Err(e) => println!("  Could not hash exe: {}", e),
    }

    // ------------------------------------------------------------------
    // 2. IntegrityConfig Builder
    // ------------------------------------------------------------------
    println!("\n--- 2. Integrity Configuration ---\n");

    // Build with a known hash
    let config = IntegrityConfig::with_hash(
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    .abort_on_failure(false);

    println!("  Config:");
    println!("    expected_hash:    {:?}", config.expected_hash);
    println!("    abort_on_failure: {}", config.abort_on_failure);
    println!("    binary_path:      {:?}", config.binary_path);
    println!("    public_key_pem:   {:?}", config.public_key_pem);

    // With custom path
    let config2 = IntegrityConfig::with_hash("abc123")
        .with_path("/usr/local/bin/myapp")
        .abort_on_failure(true);
    println!("\n  Config with custom path:");
    println!("    binary_path: {:?}", config2.binary_path);

    // ------------------------------------------------------------------
    // 3. IntegrityResult Variants
    // ------------------------------------------------------------------
    println!("\n--- 3. Integrity Result Variants ---\n");

    let results: Vec<(&str, IntegrityResult)> = vec![
        ("Valid", IntegrityResult::Valid),
        ("Skipped", IntegrityResult::Skipped),
        ("NoHashConfigured", IntegrityResult::NoHashConfigured),
        ("HashMismatch", IntegrityResult::HashMismatch {
            expected: "abc123...".to_string(),
            actual: "def456...".to_string(),
        }),
        ("ReadError", IntegrityResult::ReadError("file not found".to_string())),
        ("SignatureInvalid", IntegrityResult::SignatureInvalid),
    ];

    for (name, result) in &results {
        println!("  {:<20} is_valid={:<5}  was_verified={:<5}  display=\"{}\"",
            name,
            result.is_valid(),
            result.was_verified(),
            result,
        );
    }

    // ------------------------------------------------------------------
    // 4. IntegrityChecker
    // ------------------------------------------------------------------
    println!("\n--- 4. IntegrityChecker ---\n");

    // With defaults (no hash configured → NoHashConfigured)
    let checker = IntegrityChecker::with_defaults();
    let result = checker.verify();
    println!("  Checker with defaults: {}", result);
    println!("    is_valid: {}", result.is_valid());

    // verify_self() — skips in debug builds
    match IntegrityChecker::verify_self() {
        Ok(()) => println!("  verify_self(): Ok (skipped in debug)"),
        Err(e) => println!("  verify_self(): Err({})", e),
    }

    // ------------------------------------------------------------------
    // 5. Startup Integrity Check Flow
    // ------------------------------------------------------------------
    println!("\n--- 5. Startup Integrity Check (Production Flow) ---\n");

    // Simulate what a production binary would do:
    // startup_integrity_check() reads BINARY_INTEGRITY_HASH from env
    let expected_hash = std::env::var("BINARY_INTEGRITY_HASH")
        .unwrap_or_else(|_| "not_set".to_string());

    println!("  BINARY_INTEGRITY_HASH env: {:?}", expected_hash);

    // In production, this would verify the binary hash and abort on mismatch
    startup_integrity_check();
    println!("  startup_integrity_check() completed (no-op in debug builds)");

    // For more control, use IntegrityChecker directly:
    let checker = IntegrityChecker::new(
        IntegrityConfig::with_hash(&expected_hash).abort_on_failure(false),
    );
    let startup_result = checker.verify();
    println!("  Manual verify result: {}", startup_result);
    println!("  is_valid: {}", startup_result.is_valid());

    println!("\n  Production usage:");
    println!("    1. Build release binary");
    println!("    2. Compute: sha256sum target/release/myapp");
    println!("    3. Set env: BINARY_INTEGRITY_HASH=<hash>");
    println!("    4. App calls startup_integrity_check() on launch");
    println!("    5. If hash mismatches → binary was tampered");

    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  Integrity check demo complete.");
    println!("  Capabilities: SHA256 hashing, config builder,");
    println!("    startup verification, tamper detection.");
    println!("==========================================================");
}
