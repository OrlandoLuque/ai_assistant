//! Butler demo — environment detection and configuration.
//!
//! The butler module auto-detects local LLM providers, project type,
//! VCS information, and runtime capabilities, then suggests an optimal
//! configuration.
//!
//! Run with: cargo run --example butler_demo --features "butler"

// The `butler` feature transitively enables `autonomous`, which makes
// both the `butler` module and many autonomous-related types available.

use ai_assistant::butler::{
    Butler, DetectionResult, EnvironmentReport, ProjectType, RuntimeInfo,
};

fn main() {
    println!("=== Butler Environment Detection Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Manual DetectionResult usage
    // -----------------------------------------------------------------------
    let positive = DetectionResult::found({
        let mut d = std::collections::HashMap::new();
        d.insert("url".to_string(), "http://localhost:11434".to_string());
        d.insert("model_count".to_string(), "3".to_string());
        d
    });
    println!("Positive detection: detected={}", positive.detected);
    for (k, v) in &positive.details {
        println!("  {}: {}", k, v);
    }

    let negative = DetectionResult::not_found();
    println!("Negative detection: detected={}\n", negative.detected);

    // -----------------------------------------------------------------------
    // 2. Create a Butler instance and perform a scan
    // -----------------------------------------------------------------------
    println!("--- Butler scan (detects local environment) ---");
    let mut butler = Butler::new();

    // The scan contacts real endpoints (Ollama, LM Studio, etc.) with short
    // timeouts. If nothing is running locally, the detectors simply report
    // "not found" — this is safe to run offline.
    let report: EnvironmentReport = butler.scan();

    // -----------------------------------------------------------------------
    // 3. Inspect the environment report
    // -----------------------------------------------------------------------
    println!("Detected LLM providers: {}", report.llm_providers.len());
    for p in &report.llm_providers {
        println!("  - {} at {} ({} models)", p.name, p.url, p.available_models.len());
    }

    match &report.project_type {
        Some(pt) => {
            let name = match pt {
                ProjectType::Rust => "Rust",
                ProjectType::Node => "Node.js",
                ProjectType::Python => "Python",
                ProjectType::Go => "Go",
                ProjectType::Java => "Java",
                ProjectType::DotNet => ".NET",
                ProjectType::Ruby => "Ruby",
                ProjectType::Unknown => "Unknown",
            };
            println!("Project type: {}", name);
        }
        None => println!("Project type: none detected"),
    }

    if let Some(vcs) = &report.vcs {
        println!("VCS: {} (branch: {}, has remotes: {})", vcs.vcs_type, vcs.branch, vcs.has_remotes);
    } else {
        println!("VCS: not detected");
    }

    let rt: &RuntimeInfo = &report.runtime;
    println!(
        "Runtime: os={}, arch={}, cpus={}, gpu={}, docker={}, browser={}",
        rt.os, rt.arch, rt.cpus, rt.has_gpu, rt.has_docker, rt.has_browser,
    );

    println!("Capabilities: {:?}", report.capabilities);
    println!("Suggested agent profile: {}", report.suggested_agent_profile);
    println!("Suggested operation mode: {:?}", report.suggested_mode);

    // -----------------------------------------------------------------------
    // 4. Use Butler to suggest a configuration
    // -----------------------------------------------------------------------
    let config = butler.suggest_config(&report);
    println!("\n--- Suggested AiConfig ---");
    println!("  Provider: {:?}", config.provider);
    println!("  Selected model: {:?}", config.selected_model);
    println!("  Ollama URL: {:?}", config.ollama_url);

    // -----------------------------------------------------------------------
    // 5. Suggest an agent profile
    // -----------------------------------------------------------------------
    let profile = butler.suggest_agent_profile(&report);
    println!("\n--- Suggested AgentProfile ---");
    println!("  Name: {}", profile.name);
    println!("  System prompt: {:?}", profile.system_prompt.as_deref().map(|s| &s[..s.len().min(60)]));
    println!("  Tools: {:?}", profile.tools);

    // -----------------------------------------------------------------------
    // 6. Speech config (if Whisper/Piper/Coqui were detected)
    // -----------------------------------------------------------------------
    let (stt, tts) = butler.suggest_speech_config();
    println!("\n--- Speech config ---");
    println!("  STT (Whisper): {:?}", stt);
    println!("  TTS (Piper/Coqui): {:?}", tts);

    println!("\nButler demo complete.");
}
