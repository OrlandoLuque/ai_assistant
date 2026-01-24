//! Provider-specific API implementations

use std::sync::mpsc::Sender;
use std::io::BufRead;
use anyhow::{Context, Result};

use crate::config::{AiConfig, AiProvider};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::{format_size, ModelInfo};
use crate::session::UserPreferences;
use crate::conversation_control::CancellationToken;

/// Fetch models from Ollama API
pub fn fetch_ollama_models(base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);

    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .context("Failed to connect to Ollama")?;

    let body: serde_json::Value = response.into_json()?;

    let mut models = Vec::new();
    if let Some(model_list) = body.get("models").and_then(|m| m.as_array()) {
        for model in model_list {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                models.push(ModelInfo {
                    name: name.to_string(),
                    provider: AiProvider::Ollama,
                    size: model.get("size").and_then(|s| s.as_u64()).map(format_size),
                    modified_at: model.get("modified_at").and_then(|m| m.as_str()).map(|s| s.to_string()),
                });
            }
        }
    }

    Ok(models)
}

/// Fetch models from OpenAI-compatible API (LM Studio, LocalAI, etc.)
pub fn fetch_openai_compatible_models(base_url: &str, provider: AiProvider) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url);

    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .context("Failed to connect to OpenAI-compatible API")?;

    let body: serde_json::Value = response.into_json()?;

    let mut models = Vec::new();
    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        for model in data {
            if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                models.push(ModelInfo {
                    name: id.to_string(),
                    provider: provider.clone(),
                    size: None,
                    modified_at: None,
                });
            }
        }
    }

    Ok(models)
}

/// Fetch models from Kobold.cpp API
pub fn fetch_kobold_models(base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);

    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .context("Failed to connect to Kobold.cpp")?;

    let body: serde_json::Value = response.into_json()?;

    let mut models = Vec::new();

    // Kobold returns the current loaded model
    if let Some(result) = body.get("result").and_then(|r| r.as_str()) {
        if !result.is_empty() && result != "Read Only" {
            models.push(ModelInfo {
                name: result.to_string(),
                provider: AiProvider::KoboldCpp,
                size: None,
                modified_at: None,
            });
        }
    }

    // If no model from result, check if Kobold is running
    if models.is_empty() {
        let version_url = format!("{}/api/v1/info/version", base_url);
        if ureq::get(&version_url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .is_ok()
        {
            models.push(ModelInfo {
                name: "Kobold Model".to_string(),
                provider: AiProvider::KoboldCpp,
                size: None,
                modified_at: None,
            });
        }
    }

    Ok(models)
}

/// Build the full system prompt with preferences and knowledge
pub fn build_system_prompt(
    base_prompt: &str,
    preferences: &UserPreferences,
    knowledge: &str,
) -> String {
    build_system_prompt_with_notes(base_prompt, preferences, knowledge, "", "")
}

/// Build the full system prompt with preferences, knowledge, and user notes
pub fn build_system_prompt_with_notes(
    base_prompt: &str,
    preferences: &UserPreferences,
    knowledge: &str,
    session_notes: &str,
    knowledge_notes: &str,
) -> String {
    let mut prompt = base_prompt.to_string();

    // Add user notes (global + session) - high priority context
    let has_global_notes = !preferences.global_notes.is_empty();
    let has_session_notes = !session_notes.is_empty();
    let has_knowledge_notes = !knowledge_notes.is_empty();

    if has_global_notes || has_session_notes {
        prompt.push_str("\n\n--- USER NOTES (IMPORTANT) ---\n");

        if has_global_notes {
            prompt.push_str("Global preferences:\n");
            prompt.push_str(&preferences.global_notes);
            prompt.push_str("\n\n");
        }

        if has_session_notes {
            prompt.push_str("Session-specific notes:\n");
            prompt.push_str(session_notes);
            prompt.push_str("\n");
        }

        prompt.push_str("--- END USER NOTES ---\n");
    }

    // Add knowledge notes if any
    if has_knowledge_notes {
        prompt.push_str("\n--- KNOWLEDGE NOTES ---\n");
        prompt.push_str(knowledge_notes);
        prompt.push_str("\n--- END KNOWLEDGE NOTES ---\n");
    }

    // Add knowledge context
    if !knowledge.is_empty() {
        prompt.push_str("\n\n--- KNOWLEDGE BASE ---\n");
        prompt.push_str(knowledge);
        prompt.push_str("\n--- END KNOWLEDGE ---\n");
    }

    // Add user preferences
    if !preferences.ships_owned.is_empty() {
        prompt.push_str(&format!("\nUser's ships: {}", preferences.ships_owned.join(", ")));
    }

    if let Some(ref target) = preferences.target_ship {
        prompt.push_str(&format!("\nUser wants: {}", target));
    }

    if !preferences.interests.is_empty() {
        prompt.push_str(&format!("\nInterests: {}", preferences.interests.join(", ")));
    }

    match preferences.response_style {
        crate::session::ResponseStyle::Concise => prompt.push_str("\n\nBe brief and concise."),
        crate::session::ResponseStyle::Detailed => prompt.push_str("\n\nProvide detailed explanations."),
        crate::session::ResponseStyle::Technical => prompt.push_str("\n\nInclude technical details."),
        crate::session::ResponseStyle::Normal => {}
    }

    prompt
}

/// Build messages array for API request
fn build_messages_array(
    system_prompt: &str,
    conversation: &[ChatMessage],
    max_history: usize,
) -> Vec<serde_json::Value> {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    // System message
    messages.push(serde_json::json!({
        "role": "system",
        "content": system_prompt
    }));

    // Add conversation history (limited)
    let history_start = conversation.len().saturating_sub(max_history);
    for msg in &conversation[history_start..] {
        messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    messages
}

/// Generate streaming response using Ollama API
pub fn generate_ollama_streaming(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
) -> Result<()> {
    let url = format!("{}/api/chat", config.ollama_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "options": {
            "temperature": config.temperature
        }
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send_json(&request_body)
        .context("Failed to send request to Ollama")?;

    let reader = std::io::BufReader::new(response.into_reader());
    let mut full_response = String::new();

    for line in reader.lines() {
        let line = line.context("Failed to read streaming line")?;
        if line.is_empty() {
            continue;
        }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&line) {
            // Ollama format: {"message": {"content": "..."}, "done": false}
            if let Some(content) = chunk.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                if !content.is_empty() {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }

            if chunk.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                break;
            }
        }
    }

    let _ = tx.send(AiResponse::Complete(full_response));
    Ok(())
}

/// Generate streaming response using OpenAI-compatible API
pub fn generate_openai_streaming(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
) -> Result<()> {
    let base_url = match &config.provider {
        AiProvider::LMStudio => &config.lm_studio_url,
        AiProvider::TextGenWebUI => &config.text_gen_webui_url,
        AiProvider::LocalAI => &config.local_ai_url,
        AiProvider::OpenAICompatible { base_url } => base_url,
        _ => return Err(anyhow::anyhow!("Invalid provider for OpenAI-compatible API")),
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "temperature": config.temperature
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send_json(&request_body)
        .context("Failed to send request to LLM API")?;

    let reader = std::io::BufReader::new(response.into_reader());
    let mut full_response = String::new();

    for line in reader.lines() {
        let line = line.context("Failed to read streaming line")?;

        // SSE format: lines start with "data: "
        if !line.starts_with("data: ") {
            continue;
        }

        let json_str = &line[6..]; // Skip "data: " prefix

        if json_str == "[DONE]" {
            break;
        }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(json_str) {
            // OpenAI format: {"choices": [{"delta": {"content": "..."}}]}
            if let Some(content) = chunk.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                if !content.is_empty() {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }
        }
    }

    let _ = tx.send(AiResponse::Complete(full_response));
    Ok(())
}

/// Generate non-streaming response using Ollama API
pub fn generate_ollama_response(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let url = format!("{}/api/chat", config.ollama_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": false,
        "options": {
            "temperature": config.temperature
        }
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&request_body)
        .context("Failed to send request to Ollama")?;

    let body: serde_json::Value = response.into_json()?;

    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(content)
}

/// Generate non-streaming response using OpenAI-compatible API
pub fn generate_openai_response(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let base_url = match &config.provider {
        AiProvider::LMStudio => &config.lm_studio_url,
        AiProvider::TextGenWebUI => &config.text_gen_webui_url,
        AiProvider::LocalAI => &config.local_ai_url,
        AiProvider::OpenAICompatible { base_url } => base_url,
        _ => return Err(anyhow::anyhow!("Invalid provider for OpenAI-compatible API")),
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "temperature": config.temperature,
        "stream": false
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&request_body)
        .context("Failed to send request to API")?;

    let body: serde_json::Value = response.into_json()?;

    let content = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(content)
}

/// Generate response using Kobold.cpp API (non-streaming only)
pub fn generate_kobold_response(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let url = format!("{}/api/v1/generate", config.kobold_url);

    // Build prompt from conversation (Kobold uses a single prompt string)
    let mut full_prompt = format!("### System:\n{}\n\n", system_prompt);

    let history_start = conversation.len().saturating_sub(config.max_history_messages);
    for msg in &conversation[history_start..] {
        let role_name = if msg.role == "user" { "User" } else { "Assistant" };
        full_prompt.push_str(&format!("### {}:\n{}\n\n", role_name, msg.content));
    }

    full_prompt.push_str("### Assistant:\n");

    let request_body = serde_json::json!({
        "prompt": full_prompt,
        "max_length": 2048,
        "temperature": config.temperature,
        "top_p": 0.9,
        "rep_pen": 1.1
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&request_body)
        .context("Failed to send request to Kobold.cpp")?;

    let body: serde_json::Value = response.into_json()?;

    // Kobold returns results in an array
    let content = body
        .get("results")
        .and_then(|r| r.get(0))
        .and_then(|r| r.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(content)
}

/// Generate response with streaming - routes to appropriate provider
pub fn generate_response_streaming(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
) -> Result<()> {
    match &config.provider {
        AiProvider::Ollama => {
            generate_ollama_streaming(config, conversation, system_prompt, tx)
        }
        AiProvider::KoboldCpp => {
            // Kobold doesn't support streaming well, fall back to non-streaming
            let response = generate_kobold_response(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::OpenAICompatible { .. } => {
            generate_openai_streaming(config, conversation, system_prompt, tx)
        }
    }
}

/// Generate response without streaming - routes to appropriate provider
pub fn generate_response(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    match &config.provider {
        AiProvider::Ollama => {
            generate_ollama_response(config, conversation, system_prompt)
        }
        AiProvider::KoboldCpp => {
            generate_kobold_response(config, conversation, system_prompt)
        }
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::OpenAICompatible { .. } => {
            generate_openai_response(config, conversation, system_prompt)
        }
    }
}

// ============================================================================
// Streaming with Cancellation Support
// ============================================================================

/// Generate streaming response using Ollama API with cancellation support
pub fn generate_ollama_streaming_cancellable(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
    cancel_token: &CancellationToken,
) -> Result<()> {
    let url = format!("{}/api/chat", config.ollama_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "options": {
            "temperature": config.temperature
        }
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send_json(&request_body)
        .context("Failed to send request to Ollama")?;

    let reader = std::io::BufReader::new(response.into_reader());
    let mut full_response = String::new();

    for line in reader.lines() {
        // Check for cancellation before processing each chunk
        if cancel_token.is_cancelled() {
            let _ = tx.send(AiResponse::Cancelled(full_response.clone()));
            return Ok(());
        }

        let line = line.context("Failed to read streaming line")?;
        if line.is_empty() {
            continue;
        }

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(content) = chunk.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                if !content.is_empty() {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }

            if chunk.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                break;
            }
        }
    }

    let _ = tx.send(AiResponse::Complete(full_response));
    Ok(())
}

/// Generate streaming response using OpenAI-compatible API with cancellation support
pub fn generate_openai_streaming_cancellable(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
    cancel_token: &CancellationToken,
) -> Result<()> {
    let base_url = match &config.provider {
        AiProvider::LMStudio => &config.lm_studio_url,
        AiProvider::TextGenWebUI => &config.text_gen_webui_url,
        AiProvider::LocalAI => &config.local_ai_url,
        AiProvider::OpenAICompatible { base_url } => base_url,
        _ => return Err(anyhow::anyhow!("Invalid provider for OpenAI-compatible API")),
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "temperature": config.temperature
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send_json(&request_body)
        .context("Failed to send request to OpenAI-compatible API")?;

    let reader = std::io::BufReader::new(response.into_reader());
    let mut full_response = String::new();

    for line in reader.lines() {
        // Check for cancellation before processing each chunk
        if cancel_token.is_cancelled() {
            let _ = tx.send(AiResponse::Cancelled(full_response.clone()));
            return Ok(());
        }

        let line = line.context("Failed to read streaming line")?;

        // Skip empty lines and SSE prefix
        if line.is_empty() || line == "data: [DONE]" {
            continue;
        }

        let json_str = if line.starts_with("data: ") {
            &line[6..]
        } else {
            &line
        };

        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(content) = chunk.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                if !content.is_empty() {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }

            // Check for finish_reason
            if let Some(finish) = chunk.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("finish_reason"))
            {
                if !finish.is_null() {
                    break;
                }
            }
        }
    }

    let _ = tx.send(AiResponse::Complete(full_response));
    Ok(())
}

/// Generate response with streaming and cancellation support - routes to appropriate provider
pub fn generate_response_streaming_cancellable(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
    cancel_token: &CancellationToken,
) -> Result<()> {
    // Check for cancellation before starting
    if cancel_token.is_cancelled() {
        let _ = tx.send(AiResponse::Cancelled(String::new()));
        return Ok(());
    }

    match &config.provider {
        AiProvider::Ollama => {
            generate_ollama_streaming_cancellable(config, conversation, system_prompt, tx, cancel_token)
        }
        AiProvider::KoboldCpp => {
            // Kobold doesn't support streaming well, fall back to non-streaming
            // Check cancellation before blocking call
            if cancel_token.is_cancelled() {
                let _ = tx.send(AiResponse::Cancelled(String::new()));
                return Ok(());
            }
            let response = generate_kobold_response(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::OpenAICompatible { .. } => {
            generate_openai_streaming_cancellable(config, conversation, system_prompt, tx, cancel_token)
        }
    }
}
