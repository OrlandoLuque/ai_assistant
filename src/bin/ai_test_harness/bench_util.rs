// Shared config for the live-model benchmark categories (real_e2e, code_gen_bench).
//
// Provider / model / endpoint come from the environment so the SAME task set can
// be pointed at Ollama, llama.cpp, or any local OpenAI-compatible server — which
// is the whole point of a benchmark: compare backends and models fairly.
//
//   AI_BENCH_PROVIDER  (default "ollama")
//        ollama | llamacpp | lmstudio | localai | vllm | textgenwebui | koboldcpp
//   AI_BENCH_MODEL     (default "llama3.2:3b")
//   AI_BENCH_URL       (optional) overrides the provider's default endpoint
//
// e.g. run against llama.cpp's server:
//   $env:AI_BENCH_PROVIDER="llamacpp"; $env:AI_BENCH_MODEL="qwen2.5-coder-7b"
//   $env:AI_BENCH_URL="http://127.0.0.1:8080"

use std::time::Duration;

use ai_assistant::{AiAssistant, AiProvider};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

pub(crate) fn bench_provider() -> AiProvider {
    match env_or("AI_BENCH_PROVIDER", "ollama")
        .to_lowercase()
        .as_str()
    {
        "llamacpp" | "llama.cpp" | "llama_cpp" => AiProvider::LlamaCpp,
        "lmstudio" | "lm-studio" => AiProvider::LMStudio,
        "localai" => AiProvider::LocalAI,
        "vllm" => AiProvider::VLLM,
        "textgenwebui" | "text-gen-webui" | "oobabooga" => AiProvider::TextGenWebUI,
        "koboldcpp" | "kobold" => AiProvider::KoboldCpp,
        _ => AiProvider::Ollama,
    }
}

pub(crate) fn bench_model() -> String {
    env_or("AI_BENCH_MODEL", "llama3.2:3b")
}

/// The endpoint URL the assistant will hit (explicit override, else the
/// provider default). Used both to configure the assistant and to probe
/// reachability.
fn bench_endpoint(provider: &AiProvider) -> String {
    if let Ok(u) = std::env::var("AI_BENCH_URL") {
        if !u.trim().is_empty() {
            return u;
        }
    }
    match provider {
        AiProvider::LlamaCpp => "http://127.0.0.1:8080".to_string(),
        AiProvider::LMStudio => "http://127.0.0.1:1234".to_string(),
        AiProvider::VLLM | AiProvider::LocalAI => "http://127.0.0.1:8000".to_string(),
        AiProvider::TextGenWebUI => "http://127.0.0.1:5000".to_string(),
        AiProvider::KoboldCpp => "http://127.0.0.1:5001".to_string(),
        _ => "http://127.0.0.1:11434".to_string(),
    }
}

/// Human-readable "provider/model" for report headers.
pub(crate) fn bench_label() -> String {
    format!("{}/{}", bench_provider().display_name(), bench_model())
}

/// A fresh assistant configured for the selected backend (temperature 0 for
/// reproducibility).
pub(crate) fn bench_assistant() -> AiAssistant {
    let provider = bench_provider();
    let endpoint = bench_endpoint(&provider);
    let mut a = AiAssistant::new();
    a.config.selected_model = bench_model();
    a.config.temperature = 0.0;
    // Point the right URL field at the endpoint. Only an explicit AI_BENCH_URL
    // (or a non-Ollama provider) changes anything from the defaults.
    match &provider {
        AiProvider::Ollama => a.config.ollama_url = endpoint,
        // Every other supported local backend is OpenAI-compatible and, in this
        // codebase, routed via the llama.cpp URL field.
        _ => a.config.llamacpp_url = endpoint,
    }
    a.config.provider = provider;
    a
}

/// Whether the configured backend's endpoint accepts a TCP connection (so the
/// category can skip gracefully when the server isn't running, e.g. in CI).
pub(crate) fn backend_reachable() -> bool {
    use std::net::ToSocketAddrs;
    let ep = bench_endpoint(&bench_provider());
    let hostport = ep
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let hostport = hostport.split('/').next().unwrap_or(hostport);
    let addr = if hostport.contains(':') {
        hostport.to_string()
    } else {
        format!("{hostport}:80")
    };
    addr.to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next())
        .and_then(|a| std::net::TcpStream::connect_timeout(&a, Duration::from_secs(2)).ok())
        .is_some()
}
