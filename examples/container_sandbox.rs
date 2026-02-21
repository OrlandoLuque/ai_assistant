//! Container sandbox demo: Docker-based isolated code execution.
//!
//! Run with: cargo run --example container_sandbox --features containers
//!
//! Demonstrates:
//! - Creating a ContainerExecutor with default config
//! - Creating a ContainerSandbox with default images (Python, Node, Bash)
//! - Executing Python code in a container
//! - Executing JavaScript code in a container
//! - Using ExecutionBackend for automatic Docker/process fallback
//! - Creating and using a SharedFolder

use ai_assistant::{
    ContainerConfig, ContainerExecutor, ContainerSandbox, ContainerSandboxConfig,
    ExecutionBackend, SandboxLanguage as Language, SharedFolder,
};

fn main() -> anyhow::Result<()> {
    println!("=== ai_assistant container sandbox demo ===\n");

    // ── 1. ContainerExecutor with default config ─────────────────────────

    println!("--- ContainerExecutor ---");
    let docker_available = ContainerExecutor::is_docker_available();
    println!(
        "Docker available: {} (checked via daemon ping)",
        docker_available
    );

    let config = ContainerConfig::default();
    println!(
        "Default config: timeout={}s, memory={}MB, network={}, auto_pull={}",
        config.default_timeout.as_secs(),
        config.default_memory_limit / (1024 * 1024),
        config.default_network_mode,
        config.auto_pull,
    );

    // ── 2. ContainerSandbox with default images ──────────────────────────

    println!("\n--- ContainerSandboxConfig (default images) ---");
    let sandbox_config = ContainerSandboxConfig::default();
    println!(
        "  Python  -> {:?}",
        sandbox_config.image_for(&Language::Python)
    );
    println!(
        "  Node.js -> {:?}",
        sandbox_config.image_for(&Language::JavaScript)
    );
    println!(
        "  Bash    -> {:?}",
        sandbox_config.image_for(&Language::Bash)
    );
    println!(
        "  Reuse containers: {}",
        sandbox_config.reuse_containers
    );

    // ── 3. Execute Python code in a container ────────────────────────────

    println!("\n--- Execute Python code ---");
    if docker_available {
        match ContainerSandbox::new(ContainerSandboxConfig::default()) {
            Ok(mut sandbox) => {
                let python_code = r#"
import sys
print(f"Hello from Python {sys.version_info.major}.{sys.version_info.minor}!")
print(f"2 + 2 = {2 + 2}")
"#;
                let result = sandbox.execute(&Language::Python, python_code);
                println!("  exit_code: {}", result.exit_code);
                println!("  stdout: {}", result.stdout.trim());
                if !result.stderr.is_empty() {
                    println!("  stderr: {}", result.stderr.trim());
                }
                println!("  duration: {:?}", result.duration);

                // ── 4. Execute JavaScript code ───────────────────────────

                println!("\n--- Execute JavaScript code ---");
                let js_code = r#"
const msg = "Hello from Node.js " + process.version + "!";
console.log(msg);
console.log("Array sum:", [1,2,3,4,5].reduce((a,b) => a+b, 0));
"#;
                let result = sandbox.execute(&Language::JavaScript, js_code);
                println!("  exit_code: {}", result.exit_code);
                println!("  stdout: {}", result.stdout.trim());
                if !result.stderr.is_empty() {
                    println!("  stderr: {}", result.stderr.trim());
                }
                println!("  duration: {:?}", result.duration);

                // Cleanup is automatic (Drop), but we can do it explicitly.
                sandbox.cleanup();
            }
            Err(e) => println!("  [skip] Failed to create ContainerSandbox: {}", e),
        }
    } else {
        println!("  [skip] Docker not available. Install Docker to run container demos.");
        println!("  Showing API usage without execution:");
        println!("    let mut sandbox = ContainerSandbox::new(config)?;");
        println!("    let result = sandbox.execute(&Language::Python, \"print('hi')\");");
    }

    // ── 5. ExecutionBackend: automatic Docker/process fallback ───────────

    println!("\n--- ExecutionBackend (auto-detect) ---");
    let mut backend = ExecutionBackend::auto();
    println!(
        "Selected backend: {} (is_container={}, is_process={})",
        backend.backend_name(),
        backend.is_container(),
        backend.is_process(),
    );

    // Run a simple Bash command through whichever backend was selected.
    let result = backend.execute(&Language::Bash, "echo 'Hello from ExecutionBackend!'");
    println!("  exit_code: {}", result.exit_code);
    println!("  stdout: {}", result.stdout.trim());
    if result.exit_code != 0 && !result.stderr.is_empty() {
        println!("  stderr: {}", result.stderr.trim());
    }

    // ── 6. SharedFolder for host/container file sharing ──────────────────

    println!("\n--- SharedFolder ---");
    let folder = SharedFolder::temp()?;
    println!("  Host path: {}", folder.host_path().display());
    println!("  Container mount: {}", folder.container_path());
    println!("  Bind mount spec: {}", folder.bind_mount_spec());

    // Write a file into the shared folder.
    folder.put_file("greeting.txt", b"Hello from the host!")?;
    println!("  Wrote 'greeting.txt' ({} bytes)", folder.size_bytes()?);

    // List files.
    let files = folder.list_files()?;
    println!("  Files in shared folder: {:?}", files);

    // Read it back.
    let content = folder.get_file("greeting.txt")?;
    println!(
        "  Read back: {:?}",
        String::from_utf8_lossy(&content)
    );

    // Clean up.
    folder.clear()?;
    println!("  Cleared shared folder (files: {})", folder.list_files()?.len());

    println!("\n=== Done ===");
    Ok(())
}
