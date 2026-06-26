use super::*;

/// Supported provider types
#[derive(Clone, Debug, PartialEq)]
pub enum ProviderType {
    Ollama,
    OpenAI,
    Anthropic,
    OpenAICompatible, // For any OpenAI-compatible endpoint
}

impl ProviderType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "openai" => Some(Self::OpenAI),
            "anthropic" => Some(Self::Anthropic),
            "openai-compatible" | "openaicompatible" => Some(Self::OpenAICompatible),
            _ => None,
        }
    }

    fn default_url(&self) -> &str {
        match self {
            Self::Ollama => "http://localhost:11434",
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com/v1",
            Self::OpenAICompatible => "http://localhost:8080/v1",
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Ollama => "ollama",
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::OpenAICompatible => "openai-compatible",
        }
    }
}

/// Replay configuration
pub struct ReplayConfig {
    pub session_file: String,
    pub provider: Option<String>, // Override provider type
    pub url: Option<String>,      // Override provider URL
    pub model: Option<String>,    // Override model name
    pub api_key: Option<String>,  // API key for OpenAI/Anthropic
    pub compare: bool,
    pub session_index: Option<usize>,
}

/// Load and parse a RAG debug session file
pub fn load_session_file(path: &str) -> Result<Vec<ai_assistant::RagDebugSession>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    // Try parsing as AllSessionsExport first
    if let Ok(export) = serde_json::from_str::<ai_assistant::AllSessionsExport>(&content) {
        return Ok(export.sessions);
    }

    // Try parsing as single session
    if let Ok(session) = serde_json::from_str::<ai_assistant::RagDebugSession>(&content) {
        return Ok(vec![session]);
    }

    // Try parsing as array of sessions
    if let Ok(sessions) = serde_json::from_str::<Vec<ai_assistant::RagDebugSession>>(&content) {
        return Ok(sessions);
    }

    Err("Could not parse file as RagDebugSession, AllSessionsExport, or session array".to_string())
}

/// Check if Ollama is available and list models
fn check_ollama(url: &str) -> Result<Vec<String>, String> {
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(5))
        .build();

    let response = client
        .get(&format!("{}/api/tags", url))
        .call()
        .map_err(|e| format!("Failed to connect to Ollama: {}", e))?;

    let json: serde_json::Value = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let models: Vec<String> = json["models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

/// Check if OpenAI-compatible endpoint is available and list models
fn check_openai_compatible(url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(10))
        .build();

    let mut request = client.get(&format!("{}/models", url.trim_end_matches('/')));
    if let Some(key) = api_key {
        request = request.set("Authorization", &format!("Bearer {}", key));
    }

    let response = request
        .call()
        .map_err(|e| format!("Failed to connect: {}", e))?;

    let json: serde_json::Value = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let models: Vec<String> = json["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

/// Generate a response using Ollama
fn generate_with_ollama(
    url: &str,
    model: &str,
    system_prompt: &str,
    context: &str,
    query: &str,
) -> Result<(String, u64), String> {
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();

    let full_prompt = if context.is_empty() {
        query.to_string()
    } else {
        format!("Context:\n{}\n\nQuestion: {}", context, query)
    };

    let request_body = serde_json::json!({
        "model": model,
        "prompt": full_prompt,
        "system": system_prompt,
        "stream": false,
        "options": {
            "temperature": 0.7,
            "num_predict": 2048
        }
    });

    let start = std::time::Instant::now();

    let response = client
        .post(&format!("{}/api/generate", url))
        .send_json(&request_body)
        .map_err(|e| format!("Failed to generate: {}", e))?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let json: serde_json::Value = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let response_text = json["response"].as_str().unwrap_or("").to_string();

    Ok((response_text, duration_ms))
}

/// Generate a response using OpenAI-compatible API
fn generate_with_openai_compatible(
    url: &str,
    model: &str,
    api_key: Option<&str>,
    system_prompt: &str,
    context: &str,
    query: &str,
) -> Result<(String, u64), String> {
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();

    let user_content = if context.is_empty() {
        query.to_string()
    } else {
        format!("Context:\n{}\n\nQuestion: {}", context, query)
    };

    let request_body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    });

    let start = std::time::Instant::now();

    let mut request = client.post(&format!("{}/chat/completions", url.trim_end_matches('/')));
    if let Some(key) = api_key {
        request = request.set("Authorization", &format!("Bearer {}", key));
    }
    request = request.set("Content-Type", "application/json");

    let response = request
        .send_json(&request_body)
        .map_err(|e| format!("Failed to generate: {}", e))?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let json: serde_json::Value = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let response_text = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok((response_text, duration_ms))
}

/// Generate a response using Anthropic API
fn generate_with_anthropic(
    url: &str,
    model: &str,
    api_key: &str,
    system_prompt: &str,
    context: &str,
    query: &str,
) -> Result<(String, u64), String> {
    let client = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();

    let user_content = if context.is_empty() {
        query.to_string()
    } else {
        format!("Context:\n{}\n\nQuestion: {}", context, query)
    };

    let request_body = serde_json::json!({
        "model": model,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content}
        ]
    });

    let start = std::time::Instant::now();

    let response = client
        .post(&format!("{}/messages", url.trim_end_matches('/')))
        .set("x-api-key", api_key)
        .set("anthropic-version", "2023-06-01")
        .set("Content-Type", "application/json")
        .send_json(&request_body)
        .map_err(|e| format!("Failed to generate: {}", e))?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let json: serde_json::Value = response
        .into_json()
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let response_text = json["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok((response_text, duration_ms))
}

/// Run the replay
pub fn run_replay(config: ReplayConfig) -> Result<(), String> {
    println!(
        "{}",
        bold(&cyan(
            "═══════════════════════════════════════════════════════"
        ))
    );
    println!("{}", bold(&cyan("              RAG SESSION REPLAY")));
    println!(
        "{}",
        bold(&cyan(
            "═══════════════════════════════════════════════════════"
        ))
    );
    println!();

    // Load sessions
    println!("Loading session file: {}", config.session_file);
    let sessions = load_session_file(&config.session_file)?;
    println!("Found {} session(s)", sessions.len());
    println!();

    // Select session first to get provider info from it
    let session_idx = config.session_index.unwrap_or(0);
    if session_idx >= sessions.len() {
        return Err(format!(
            "Session index {} out of range (0-{})",
            session_idx,
            sessions.len() - 1
        ));
    }
    let session = &sessions[session_idx];

    // Determine provider: CLI override > session data > default (ollama)
    let provider_type = if let Some(ref p) = config.provider {
        ProviderType::from_str(p).ok_or_else(|| {
            format!(
                "Unknown provider '{}'. Valid: ollama, openai, anthropic, openai-compatible",
                p
            )
        })?
    } else if let Some(ref p) = session.provider_type {
        ProviderType::from_str(p).unwrap_or(ProviderType::Ollama)
    } else {
        ProviderType::Ollama
    };

    // Determine URL: CLI override > session data > default for provider
    let provider_url = config
        .url
        .clone()
        .or_else(|| session.provider_url.clone())
        .unwrap_or_else(|| provider_type.default_url().to_string());

    // Get API key from CLI or environment
    let api_key = config.api_key.clone().or_else(|| match provider_type {
        ProviderType::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
        ProviderType::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
        _ => None,
    });

    println!(
        "Provider: {} ({})",
        bold(provider_type.as_str()),
        provider_url
    );

    // Check provider and list models
    let available_models = match provider_type {
        ProviderType::Ollama => {
            println!("Checking Ollama...");
            check_ollama(&provider_url)?
        }
        ProviderType::OpenAI | ProviderType::OpenAICompatible => {
            println!("Checking OpenAI-compatible endpoint...");
            check_openai_compatible(&provider_url, api_key.as_deref()).unwrap_or_default()
        }
        ProviderType::Anthropic => {
            // Anthropic doesn't have a model list API, use known models
            vec![
                "claude-3-5-sonnet-20241022".into(),
                "claude-3-5-haiku-20241022".into(),
                "claude-3-opus-20240229".into(),
                "claude-3-sonnet-20240229".into(),
                "claude-3-haiku-20240307".into(),
            ]
        }
    };

    if !available_models.is_empty() {
        println!(
            "Available models: {}",
            available_models
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
        if available_models.len() > 5 {
            println!("  ... and {} more", available_models.len() - 5);
        }
    }
    println!();

    // Determine model: CLI override > session data > auto-select
    let model = if let Some(ref m) = config.model {
        m.clone()
    } else if let Some(ref m) = session.model_name {
        // Use session model if available
        m.clone()
    } else if !available_models.is_empty() {
        // Auto-select based on provider
        match provider_type {
            ProviderType::Ollama => {
                let preferred = ["llama3", "qwen", "mistral", "deepseek"];
                available_models
                    .iter()
                    .find(|m| preferred.iter().any(|p| m.contains(p)))
                    .unwrap_or(&available_models[0])
                    .clone()
            }
            ProviderType::OpenAI => "gpt-4o-mini".to_string(),
            ProviderType::Anthropic => "claude-3-5-haiku-20241022".to_string(),
            ProviderType::OpenAICompatible => available_models[0].clone(),
        }
    } else {
        return Err("No model specified and none available".to_string());
    };
    println!("Using model: {}", green(&model));
    println!();

    // Display session info
    println!(
        "{}",
        bold("─── Original Session ───────────────────────────────────")
    );
    println!("Session ID: {}", session.session_id);
    println!("Query: {}", cyan(&session.query));
    if let Some(ref tier) = session.rag_tier {
        println!("RAG Tier: {}", tier);
    }
    if !session.features_enabled.is_empty() {
        println!("Features: {}", session.features_enabled.join(", "));
    }
    // Show original provider info if available
    if let (Some(ref ptype), Some(ref purl), Some(ref pmodel)) = (
        &session.provider_type,
        &session.provider_url,
        &session.model_name,
    ) {
        println!("Original Provider: {} @ {} ({})", ptype, purl, pmodel);
    }
    println!(
        "Stats: {} chunks retrieved, {} used",
        session.stats.chunks_retrieved, session.stats.chunks_used
    );
    if let Some(ref original_response) = session.final_response {
        println!();
        println!("Original Response:");
        println!(
            "{}",
            yellow(&original_response.chars().take(500).collect::<String>())
        );
        if original_response.len() > 500 {
            println!("... ({} chars total)", original_response.len());
        }
    }
    println!();

    // Get context
    let context = session.final_context.clone().unwrap_or_default();
    if context.is_empty() {
        println!("{}", yellow("Warning: No context found in session"));
    } else {
        println!("Context size: {} chars", context.len());
    }

    // Generate new response
    println!();
    println!(
        "{}",
        bold("─── Generating New Response ────────────────────────────")
    );
    println!("Provider: {} | Model: {}", provider_type.as_str(), model);

    let system_prompt = "You are a helpful assistant. Answer based on the provided context.";

    let (new_response, duration_ms) = match provider_type {
        ProviderType::Ollama => generate_with_ollama(
            &provider_url,
            &model,
            system_prompt,
            &context,
            &session.query,
        )?,
        ProviderType::OpenAI | ProviderType::OpenAICompatible => generate_with_openai_compatible(
            &provider_url,
            &model,
            api_key.as_deref(),
            system_prompt,
            &context,
            &session.query,
        )?,
        ProviderType::Anthropic => {
            let key = api_key
                .as_ref()
                .ok_or("Anthropic API key required (--api-key or ANTHROPIC_API_KEY env var)")?;
            generate_with_anthropic(
                &provider_url,
                &model,
                key,
                system_prompt,
                &context,
                &session.query,
            )?
        }
    };

    println!("Generation time: {}ms", duration_ms);
    println!();
    println!("New Response:");
    println!("{}", green(&new_response));

    // Compare if requested
    if config.compare {
        if let Some(ref original) = session.final_response {
            println!();
            println!(
                "{}",
                bold("─── Comparison ─────────────────────────────────────────")
            );
            println!("Original length: {} chars", original.len());
            println!("New length: {} chars", new_response.len());

            // Simple similarity check - bind to variables to fix lifetime
            let original_lower = original.to_lowercase();
            let new_lower = new_response.to_lowercase();
            let original_words: std::collections::HashSet<&str> =
                original_lower.split_whitespace().collect();
            let new_words: std::collections::HashSet<&str> = new_lower.split_whitespace().collect();
            let common = original_words.intersection(&new_words).count();
            let total = original_words.union(&new_words).count();
            let similarity = if total > 0 {
                common as f64 / total as f64 * 100.0
            } else {
                0.0
            };

            println!("Word overlap: {:.1}%", similarity);
        }
    }

    println!();
    println!(
        "{}",
        bold(&cyan(
            "═══════════════════════════════════════════════════════"
        ))
    );

    Ok(())
}
