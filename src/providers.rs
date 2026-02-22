//! Provider-specific API implementations

use anyhow::{Context, Result};
use std::io::BufRead;
use std::sync::mpsc::Sender;

use crate::config::{AiConfig, AiProvider};
use crate::conversation_control::CancellationToken;
use crate::messages::{AiResponse, ChatMessage};
use crate::models::{format_size, ModelInfo};
use crate::retry::{retry_with_config, RetryConfig};
use crate::session::UserPreferences;

/// Fetch models from Ollama API.
///
/// Uses `RetryConfig::fast()` (2 retries with short backoff) to handle
/// transient connection errors when listing available models.
pub fn fetch_ollama_models(base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    retry_with_config(RetryConfig::fast(), || {
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()?;
        let body: serde_json::Value = response.into_json()?;

        let mut models = Vec::new();
        if let Some(model_list) = body.get("models").and_then(|m| m.as_array()) {
            for model in model_list {
                if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                    models.push(ModelInfo {
                        name: name.to_string(),
                        provider: AiProvider::Ollama,
                        size: model.get("size").and_then(|s| s.as_u64()).map(format_size),
                        modified_at: model
                            .get("modified_at")
                            .and_then(|m| m.as_str())
                            .map(|s| s.to_string()),
                        capabilities: None,
                    });
                }
            }
        }
        Ok(models)
    })
    .context("Failed to fetch Ollama models")
}

/// Fetch models from OpenAI-compatible API (LM Studio, LocalAI, etc.)
///
/// Uses `RetryConfig::fast()` (2 retries) for transient connection errors.
pub fn fetch_openai_compatible_models(
    base_url: &str,
    provider: AiProvider,
) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url);
    retry_with_config(RetryConfig::fast(), || {
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()?;
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
                        capabilities: None,
                    });
                }
            }
        }
        Ok(models)
    })
    .context("Failed to fetch OpenAI-compatible models")
}

/// Fetch models from Kobold.cpp API.
///
/// Uses `RetryConfig::fast()` for the primary model endpoint. The fallback
/// version-check endpoint uses best-effort (no retry) since it's already a fallback.
pub fn fetch_kobold_models(base_url: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/v1/model", base_url);
    let response = retry_with_config(RetryConfig::fast(), || {
        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()?;
        Ok(resp)
    })
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
                capabilities: None,
            });
        }
    }

    // If no model from result, check if Kobold is running (best-effort, no retry)
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
                capabilities: None,
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
        prompt.push_str(&format!(
            "\nUser's ships: {}",
            preferences.ships_owned.join(", ")
        ));
    }

    if let Some(ref target) = preferences.target_ship {
        prompt.push_str(&format!("\nUser wants: {}", target));
    }

    if !preferences.interests.is_empty() {
        prompt.push_str(&format!(
            "\nInterests: {}",
            preferences.interests.join(", ")
        ));
    }

    match preferences.response_style {
        crate::session::ResponseStyle::Concise => prompt.push_str("\n\nBe brief and concise."),
        crate::session::ResponseStyle::Detailed => {
            prompt.push_str("\n\nProvide detailed explanations.")
        }
        crate::session::ResponseStyle::Technical => {
            prompt.push_str("\n\nInclude technical details.")
        }
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

    // Retry only the connection; stream reading is not retried
    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(300))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
            if let Some(content) = chunk
                .get("message")
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
        AiProvider::LMStudio => config.lm_studio_url.clone(),
        AiProvider::TextGenWebUI => config.text_gen_webui_url.clone(),
        AiProvider::LocalAI => config.local_ai_url.clone(),
        AiProvider::OpenAICompatible { base_url } => base_url.clone(),
        AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => config.get_base_url(),
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid provider for OpenAI-compatible API"
            ))
        }
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "temperature": config.temperature
    });

    // Retry only the connection; stream reading is not retried
    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(300))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
            if let Some(content) = chunk
                .get("choices")
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

    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
        AiProvider::LMStudio => config.lm_studio_url.clone(),
        AiProvider::TextGenWebUI => config.text_gen_webui_url.clone(),
        AiProvider::LocalAI => config.local_ai_url.clone(),
        AiProvider::OpenAICompatible { base_url } => base_url.clone(),
        AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => config.get_base_url(),
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid provider for OpenAI-compatible API"
            ))
        }
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "temperature": config.temperature,
        "stream": false
    });

    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&request_body)?;
        Ok(resp)
    })
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

    let history_start = conversation
        .len()
        .saturating_sub(config.max_history_messages);
    for msg in &conversation[history_start..] {
        let role_name = if msg.role == "user" {
            "User"
        } else {
            "Assistant"
        };
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

    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
        AiProvider::Ollama => generate_ollama_streaming(config, conversation, system_prompt, tx),
        AiProvider::KoboldCpp => {
            // Kobold doesn't support streaming well, fall back to non-streaming
            let response = generate_kobold_response(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::OpenAICompatible { .. }
        | AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => {
            generate_openai_streaming(config, conversation, system_prompt, tx)
        }
        AiProvider::Gemini => {
            // Gemini uses its own API format, not OpenAI-compatible
            let response = crate::cloud_providers::generate_gemini_cloud(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
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
        AiProvider::Ollama => generate_ollama_response(config, conversation, system_prompt),
        AiProvider::KoboldCpp => generate_kobold_response(config, conversation, system_prompt),
        AiProvider::LMStudio
        | AiProvider::TextGenWebUI
        | AiProvider::LocalAI
        | AiProvider::OpenAICompatible { .. }
        | AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => {
            generate_openai_response(config, conversation, system_prompt)
        }
        AiProvider::Gemini => {
            crate::cloud_providers::generate_gemini_cloud(config, conversation, system_prompt)
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

    // Retry only the connection; stream reading is not retried
    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(300))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
            if let Some(content) = chunk
                .get("message")
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
        AiProvider::LMStudio => config.lm_studio_url.clone(),
        AiProvider::TextGenWebUI => config.text_gen_webui_url.clone(),
        AiProvider::LocalAI => config.local_ai_url.clone(),
        AiProvider::OpenAICompatible { base_url } => base_url.clone(),
        AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => config.get_base_url(),
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid provider for OpenAI-compatible API"
            ))
        }
    };

    let url = format!("{}/v1/chat/completions", base_url);

    let messages = build_messages_array(system_prompt, conversation, config.max_history_messages);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": messages,
        "stream": true,
        "temperature": config.temperature
    });

    // Retry only the connection; stream reading is not retried
    let response = retry_with_config(config.retry_config.clone(), || {
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(300))
            .send_json(&request_body)?;
        Ok(resp)
    })
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
            if let Some(content) = chunk
                .get("choices")
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
            if let Some(finish) = chunk
                .get("choices")
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
        AiProvider::Ollama => generate_ollama_streaming_cancellable(
            config,
            conversation,
            system_prompt,
            tx,
            cancel_token,
        ),
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
        | AiProvider::OpenAICompatible { .. }
        | AiProvider::OpenAI
        | AiProvider::Anthropic
        | AiProvider::Bedrock { .. }
        | AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => generate_openai_streaming_cancellable(
            config,
            conversation,
            system_prompt,
            tx,
            cancel_token,
        ),
        AiProvider::Gemini => {
            if cancel_token.is_cancelled() {
                let _ = tx.send(AiResponse::Cancelled(String::new()));
                return Ok(());
            }
            let response = crate::cloud_providers::generate_gemini_cloud(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
    }
}

// ============================================================================
// Context Size Detection
// ============================================================================

/// Fetch the context window size for a model from the provider
/// Returns the context size in tokens, or None if it couldn't be determined
pub fn fetch_model_context_size(config: &AiConfig, model_name: &str) -> Option<usize> {
    match &config.provider {
        AiProvider::Ollama => fetch_ollama_model_context(&config.ollama_url, model_name),
        AiProvider::KoboldCpp => fetch_kobold_context_size(&config.kobold_url),
        AiProvider::LMStudio => fetch_openai_model_context(&config.lm_studio_url, model_name),
        AiProvider::TextGenWebUI => {
            fetch_openai_model_context(&config.text_gen_webui_url, model_name)
        }
        AiProvider::LocalAI => fetch_openai_model_context(&config.local_ai_url, model_name),
        AiProvider::OpenAICompatible { base_url } => {
            fetch_openai_model_context(base_url, model_name)
        }
        AiProvider::OpenAI => {
            // OpenAI doesn't expose context size via API; use well-known sizes
            match model_name {
                m if m.starts_with("gpt-4o") => Some(128_000),
                m if m.starts_with("gpt-4-turbo") => Some(128_000),
                m if m.starts_with("gpt-4") => Some(8_192),
                m if m.starts_with("gpt-3.5") => Some(16_385),
                m if m.starts_with("o1") || m.starts_with("o3") => Some(200_000),
                _ => Some(128_000),
            }
        }
        AiProvider::Anthropic => {
            // Anthropic Claude models have known context sizes
            match model_name {
                m if m.contains("opus-4") || m.contains("sonnet-4") || m.contains("haiku-4") => {
                    Some(200_000)
                }
                m if m.contains("3-5") || m.contains("3.5") => Some(200_000),
                m if m.contains("3") => Some(200_000),
                _ => Some(200_000),
            }
        }
        AiProvider::Gemini => {
            // Gemini models support up to 1M token context
            Some(1_048_576)
        }
        AiProvider::Bedrock { .. } => {
            // Bedrock typically runs Claude models with 200K context
            Some(200_000)
        }
        AiProvider::Groq => {
            match model_name {
                m if m.contains("llama-3.3-70b") => Some(128_000),
                m if m.contains("llama-3.1") => Some(131_072),
                m if m.contains("mixtral") => Some(32_768),
                m if m.contains("gemma") => Some(8_192),
                _ => Some(128_000),
            }
        }
        AiProvider::Together | AiProvider::Fireworks => Some(128_000),
        AiProvider::DeepSeek => {
            match model_name {
                m if m.contains("deepseek-chat") || m.contains("deepseek-v3") => Some(64_000),
                m if m.contains("deepseek-coder") => Some(128_000),
                m if m.contains("deepseek-reasoner") => Some(64_000),
                _ => Some(64_000),
            }
        }
        AiProvider::Mistral => {
            match model_name {
                m if m.contains("large") => Some(128_000),
                m if m.contains("medium") || m.contains("small") => Some(32_000),
                m if m.contains("codestral") => Some(256_000),
                _ => Some(128_000),
            }
        }
        AiProvider::Perplexity => Some(128_000),
        AiProvider::OpenRouter => Some(128_000),
    }
}

/// Fetch context size from Ollama using /api/show
fn fetch_ollama_model_context(base_url: &str, model_name: &str) -> Option<usize> {
    let url = format!("{}/api/show", base_url);

    let body = serde_json::json!({
        "name": model_name
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send_json(&body)
        .ok()?;

    let json: serde_json::Value = response.into_json().ok()?;

    // Try to get num_ctx from model_info first (more reliable)
    if let Some(ctx) = json
        .get("model_info")
        .and_then(|info| info.get("llama.context_length"))
        .and_then(|v| v.as_u64())
    {
        return Some(ctx as usize);
    }

    // Fallback: parse parameters string for num_ctx
    if let Some(params) = json.get("parameters").and_then(|p| p.as_str()) {
        for line in params.lines() {
            let line = line.trim();
            if line.starts_with("num_ctx") {
                // Format: "num_ctx                          4096"
                if let Some(value) = line.split_whitespace().last() {
                    if let Ok(ctx) = value.parse::<usize>() {
                        return Some(ctx);
                    }
                }
            }
        }
    }

    // Try modelfile template parameters
    if let Some(template) = json.get("template").and_then(|t| t.as_str()) {
        // Some models have context in template
        if template.contains("context_length") {
            // Parse if present
        }
    }

    None
}

/// Fetch context size from Kobold.cpp
fn fetch_kobold_context_size(base_url: &str) -> Option<usize> {
    // Try the max_context_length endpoint
    let url = format!("{}/api/v1/config/max_context_length", base_url);

    if let Ok(response) = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
    {
        if let Ok(json) = response.into_json::<serde_json::Value>() {
            if let Some(result) = json.get("value").and_then(|v| v.as_u64()) {
                return Some(result as usize);
            }
        }
    }

    // Try alternative endpoint
    let url_alt = format!("{}/api/extra/true_max_context_length", base_url);
    if let Ok(response) = ureq::get(&url_alt)
        .timeout(std::time::Duration::from_secs(5))
        .call()
    {
        if let Ok(json) = response.into_json::<serde_json::Value>() {
            if let Some(result) = json.get("value").and_then(|v| v.as_u64()) {
                return Some(result as usize);
            }
        }
    }

    None
}

/// Fetch context size from OpenAI-compatible API
fn fetch_openai_model_context(base_url: &str, model_name: &str) -> Option<usize> {
    // Try to get specific model info
    let url = format!("{}/v1/models/{}", base_url, model_name);

    if let Ok(response) = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
    {
        if let Ok(json) = response.into_json::<serde_json::Value>() {
            // LM Studio and some others include context_length
            if let Some(ctx) = json.get("context_length").and_then(|v| v.as_u64()) {
                return Some(ctx as usize);
            }
            // Some use max_context_length
            if let Some(ctx) = json.get("max_context_length").and_then(|v| v.as_u64()) {
                return Some(ctx as usize);
            }
            // LocalAI might use context_size
            if let Some(ctx) = json.get("context_size").and_then(|v| v.as_u64()) {
                return Some(ctx as usize);
            }
        }
    }

    // Try listing all models and finding ours
    let models_url = format!("{}/v1/models", base_url);
    if let Ok(response) = ureq::get(&models_url)
        .timeout(std::time::Duration::from_secs(5))
        .call()
    {
        if let Ok(json) = response.into_json::<serde_json::Value>() {
            if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                for model in data {
                    if model.get("id").and_then(|i| i.as_str()) == Some(model_name) {
                        if let Some(ctx) = model.get("context_length").and_then(|v| v.as_u64()) {
                            return Some(ctx as usize);
                        }
                    }
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// ProviderRegistry — global registry mapping "provider/model" to AiConfig
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// A registry that maps `"provider/model"` identifiers to pre-configured
/// [`AiConfig`] instances, with support for short aliases
/// (e.g. `"gpt-4o"` resolving to `"openai/gpt-4o"`).
///
/// # Example
/// ```ignore
/// let reg = ProviderRegistry::with_defaults();
/// let cfg = reg.resolve("gpt-4o").expect("alias should resolve");
/// assert_eq!(cfg.selected_model, "gpt-4o");
/// ```
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    /// Primary registry: "provider/model" -> AiConfig
    entries: HashMap<String, AiConfig>,
    /// Alias table: short name -> canonical "provider/model" key
    aliases: HashMap<String, String>,
}

impl ProviderRegistry {
    /// Create an empty registry with no entries and no aliases.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Register a provider/model combination.
    ///
    /// `provider_model` should follow the `"provider/model"` convention
    /// (e.g. `"openai/gpt-4o"`). If an entry with the same key already
    /// exists it is silently replaced.
    pub fn register(&mut self, provider_model: &str, config: AiConfig) {
        self.entries.insert(provider_model.to_string(), config);
    }

    /// Resolve a `provider_model` string to its [`AiConfig`].
    ///
    /// The lookup first checks the primary registry; if not found it checks
    /// the alias table and follows one level of indirection.
    pub fn resolve(&self, provider_model: &str) -> Option<AiConfig> {
        if let Some(cfg) = self.entries.get(provider_model) {
            return Some(cfg.clone());
        }
        // Try alias -> canonical key
        if let Some(canonical) = self.aliases.get(provider_model) {
            return self.entries.get(canonical).cloned();
        }
        None
    }

    /// Return a sorted list of all registered canonical model identifiers.
    pub fn list_models(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Return a sorted list of all registered aliases.
    pub fn list_aliases(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.aliases.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Register a short alias that maps to a canonical `"provider/model"` key.
    ///
    /// For example: `registry.add_alias("gpt-4o", "openai/gpt-4o")`.
    pub fn add_alias(&mut self, alias: &str, target: &str) {
        self.aliases.insert(alias.to_string(), target.to_string());
    }

    /// Remove a canonical entry (and any aliases pointing to it).
    pub fn remove(&mut self, provider_model: &str) {
        self.entries.remove(provider_model);
        self.aliases.retain(|_alias, target| target != provider_model);
    }

    /// Return the number of canonical entries (excluding aliases).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the registry contains no canonical entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Create a registry pre-populated with common provider/model
    /// combinations and sensible default aliases.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();

        // -- OpenAI models --------------------------------------------------
        let openai_base = || {
            let mut cfg = AiConfig::default();
            cfg.provider = AiProvider::OpenAI;
            cfg
        };

        let mut gpt4o = openai_base();
        gpt4o.selected_model = "gpt-4o".to_string();
        reg.register("openai/gpt-4o", gpt4o);

        let mut gpt4o_mini = openai_base();
        gpt4o_mini.selected_model = "gpt-4o-mini".to_string();
        reg.register("openai/gpt-4o-mini", gpt4o_mini);

        let mut gpt35 = openai_base();
        gpt35.selected_model = "gpt-3.5-turbo".to_string();
        reg.register("openai/gpt-3.5-turbo", gpt35);

        // -- Anthropic models ------------------------------------------------
        let anthropic_base = || {
            let mut cfg = AiConfig::default();
            cfg.provider = AiProvider::Anthropic;
            cfg
        };

        let mut claude_sonnet = anthropic_base();
        claude_sonnet.selected_model = "claude-sonnet".to_string();
        reg.register("anthropic/claude-sonnet", claude_sonnet);

        let mut claude_haiku = anthropic_base();
        claude_haiku.selected_model = "claude-haiku".to_string();
        reg.register("anthropic/claude-haiku", claude_haiku);

        // -- Ollama models ---------------------------------------------------
        let ollama_base = || AiConfig::default(); // default provider is already Ollama

        let mut llama3 = ollama_base();
        llama3.selected_model = "llama3".to_string();
        reg.register("ollama/llama3", llama3);

        let mut mistral = ollama_base();
        mistral.selected_model = "mistral".to_string();
        reg.register("ollama/mistral", mistral);

        // -- LM Studio -------------------------------------------------------
        let mut lmstudio = AiConfig::default();
        lmstudio.provider = AiProvider::LMStudio;
        lmstudio.selected_model = "default".to_string();
        reg.register("lmstudio/default", lmstudio);

        // -- Common aliases --------------------------------------------------
        reg.add_alias("gpt-4o", "openai/gpt-4o");
        reg.add_alias("gpt-4o-mini", "openai/gpt-4o-mini");
        reg.add_alias("gpt-3.5-turbo", "openai/gpt-3.5-turbo");
        reg.add_alias("claude", "anthropic/claude-sonnet");
        reg.add_alias("claude-sonnet", "anthropic/claude-sonnet");
        reg.add_alias("claude-haiku", "anthropic/claude-haiku");
        reg.add_alias("llama3", "ollama/llama3");
        reg.add_alias("mistral", "ollama/mistral");
        reg.add_alias("lmstudio", "lmstudio/default");

        reg
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProviderRegistry tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod provider_registry_tests {
    use super::*;

    #[test]
    fn test_new_registry_is_empty() {
        let reg = ProviderRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.list_models().is_empty());
    }

    #[test]
    fn test_register_and_resolve() {
        let mut reg = ProviderRegistry::new();
        let mut cfg = AiConfig::default();
        cfg.provider = AiProvider::OpenAI;
        cfg.selected_model = "gpt-4o".to_string();
        reg.register("openai/gpt-4o", cfg.clone());

        let resolved = reg.resolve("openai/gpt-4o");
        assert!(resolved.is_some());
        let resolved = resolved.expect("just checked");
        assert_eq!(resolved.selected_model, "gpt-4o");
        assert_eq!(resolved.provider, AiProvider::OpenAI);
    }

    #[test]
    fn test_resolve_unknown_returns_none() {
        let reg = ProviderRegistry::new();
        assert!(reg.resolve("nonexistent/model").is_none());
    }

    #[test]
    fn test_list_models_sorted() {
        let mut reg = ProviderRegistry::new();
        let cfg = AiConfig::default();
        reg.register("ollama/mistral", cfg.clone());
        reg.register("anthropic/claude-sonnet", cfg.clone());
        reg.register("openai/gpt-4o", cfg);

        let models = reg.list_models();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0], "anthropic/claude-sonnet");
        assert_eq!(models[1], "ollama/mistral");
        assert_eq!(models[2], "openai/gpt-4o");
    }

    #[test]
    fn test_alias_resolves_to_canonical() {
        let mut reg = ProviderRegistry::new();
        let mut cfg = AiConfig::default();
        cfg.provider = AiProvider::OpenAI;
        cfg.selected_model = "gpt-4o".to_string();
        reg.register("openai/gpt-4o", cfg);
        reg.add_alias("gpt-4o", "openai/gpt-4o");

        let resolved = reg.resolve("gpt-4o");
        assert!(resolved.is_some());
        assert_eq!(resolved.expect("just checked").selected_model, "gpt-4o");
    }

    #[test]
    fn test_alias_unknown_target_returns_none() {
        let mut reg = ProviderRegistry::new();
        reg.add_alias("ghost", "nonexistent/model");
        assert!(reg.resolve("ghost").is_none());
    }

    #[test]
    fn test_with_defaults_has_openai_models() {
        let reg = ProviderRegistry::with_defaults();
        assert!(reg.resolve("openai/gpt-4o").is_some());
        assert!(reg.resolve("openai/gpt-4o-mini").is_some());
        assert!(reg.resolve("openai/gpt-3.5-turbo").is_some());
    }

    #[test]
    fn test_with_defaults_has_anthropic_models() {
        let reg = ProviderRegistry::with_defaults();
        assert!(reg.resolve("anthropic/claude-sonnet").is_some());
        assert!(reg.resolve("anthropic/claude-haiku").is_some());
    }

    #[test]
    fn test_with_defaults_has_ollama_models() {
        let reg = ProviderRegistry::with_defaults();
        assert!(reg.resolve("ollama/llama3").is_some());
        assert!(reg.resolve("ollama/mistral").is_some());
    }

    #[test]
    fn test_with_defaults_has_lmstudio() {
        let reg = ProviderRegistry::with_defaults();
        let cfg = reg.resolve("lmstudio/default");
        assert!(cfg.is_some());
        assert_eq!(cfg.expect("just checked").provider, AiProvider::LMStudio);
    }

    #[test]
    fn test_with_defaults_aliases_work() {
        let reg = ProviderRegistry::with_defaults();
        // Short aliases should resolve identically to their canonical keys
        let via_alias = reg.resolve("gpt-4o");
        let via_canonical = reg.resolve("openai/gpt-4o");
        assert!(via_alias.is_some());
        assert!(via_canonical.is_some());
        assert_eq!(
            via_alias.expect("just checked").selected_model,
            via_canonical.expect("just checked").selected_model,
        );

        // claude alias
        let claude = reg.resolve("claude");
        assert!(claude.is_some());
        assert_eq!(claude.expect("just checked").selected_model, "claude-sonnet");
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut reg = ProviderRegistry::new();
        let mut cfg1 = AiConfig::default();
        cfg1.selected_model = "model-v1".to_string();
        reg.register("test/model", cfg1);

        let mut cfg2 = AiConfig::default();
        cfg2.selected_model = "model-v2".to_string();
        reg.register("test/model", cfg2);

        let resolved = reg.resolve("test/model").expect("entry exists");
        assert_eq!(resolved.selected_model, "model-v2");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_remove_entry_and_aliases() {
        let mut reg = ProviderRegistry::new();
        let cfg = AiConfig::default();
        reg.register("openai/gpt-4o", cfg);
        reg.add_alias("gpt-4o", "openai/gpt-4o");

        assert!(reg.resolve("gpt-4o").is_some());
        reg.remove("openai/gpt-4o");
        assert!(reg.resolve("openai/gpt-4o").is_none());
        // Alias should also stop resolving because its target is gone
        assert!(reg.resolve("gpt-4o").is_none());
        // And the dangling alias itself should have been cleaned up
        assert!(reg.list_aliases().is_empty());
    }

    #[test]
    fn test_default_trait_gives_empty_registry() {
        let reg = ProviderRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_with_defaults_model_count() {
        let reg = ProviderRegistry::with_defaults();
        // 3 OpenAI + 2 Anthropic + 2 Ollama + 1 LMStudio = 8
        assert_eq!(reg.len(), 8);
    }

    #[test]
    fn test_with_defaults_correct_providers() {
        let reg = ProviderRegistry::with_defaults();
        assert_eq!(
            reg.resolve("openai/gpt-4o").expect("exists").provider,
            AiProvider::OpenAI,
        );
        assert_eq!(
            reg.resolve("anthropic/claude-haiku").expect("exists").provider,
            AiProvider::Anthropic,
        );
        assert_eq!(
            reg.resolve("ollama/llama3").expect("exists").provider,
            AiProvider::Ollama,
        );
    }
}

#[cfg(test)]
mod context_size_tests {
    use super::*;

    #[test]
    fn test_fetch_model_context_size_fallback() {
        // With no server running, should return None (not panic)
        let config = AiConfig::default();
        let result = fetch_model_context_size(&config, "nonexistent");
        assert!(result.is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{ResponseStyle, UserPreferences};

    #[test]
    fn test_build_system_prompt_empty_knowledge() {
        let prefs = UserPreferences::default();
        let result = build_system_prompt("You are a helpful assistant.", &prefs, "");
        assert!(result.starts_with("You are a helpful assistant."));
        assert!(!result.contains("KNOWLEDGE BASE"));
    }

    #[test]
    fn test_build_system_prompt_with_knowledge() {
        let prefs = UserPreferences::default();
        let knowledge = "The Orion constellation contains Betelgeuse.";
        let result = build_system_prompt("Base prompt.", &prefs, knowledge);
        assert!(
            result.contains("KNOWLEDGE BASE"),
            "prompt should contain KNOWLEDGE BASE section"
        );
        assert!(
            result.contains(knowledge),
            "prompt should contain the knowledge text"
        );
    }

    #[test]
    fn test_build_system_prompt_with_notes() {
        let mut prefs = UserPreferences::default();
        prefs.global_notes = "Always answer in English.".to_string();

        let result = build_system_prompt_with_notes(
            "Base prompt.",
            &prefs,
            "Some knowledge.",
            "Focus on astronomy.",
            "Star catalog v2.",
        );

        assert!(
            result.contains("USER NOTES"),
            "prompt should contain USER NOTES section"
        );
        assert!(
            result.contains("Always answer in English."),
            "prompt should contain global notes text"
        );
        assert!(
            result.contains("Session-specific notes:"),
            "prompt should contain session notes header"
        );
        assert!(
            result.contains("Focus on astronomy."),
            "prompt should contain session notes text"
        );
        assert!(
            result.contains("KNOWLEDGE NOTES"),
            "prompt should contain KNOWLEDGE NOTES section"
        );
        assert!(
            result.contains("Star catalog v2."),
            "prompt should contain knowledge notes text"
        );
        assert!(
            result.contains("KNOWLEDGE BASE"),
            "prompt should contain KNOWLEDGE BASE section"
        );
        assert!(
            result.contains("Some knowledge."),
            "prompt should contain knowledge text"
        );
    }

    #[test]
    fn test_build_system_prompt_response_style() {
        let styles_and_expected: Vec<(ResponseStyle, Option<&str>)> = vec![
            (ResponseStyle::Concise, Some("Be brief and concise.")),
            (
                ResponseStyle::Detailed,
                Some("Provide detailed explanations."),
            ),
            (ResponseStyle::Technical, Some("Include technical details.")),
            (ResponseStyle::Normal, None),
        ];

        for (style, expected_fragment) in styles_and_expected {
            let prefs = UserPreferences {
                response_style: style.clone(),
                ..UserPreferences::default()
            };
            let result = build_system_prompt("Base.", &prefs, "");

            if let Some(fragment) = expected_fragment {
                assert!(
                    result.contains(fragment),
                    "Style {:?} should produce '{}' in prompt, got: {}",
                    style,
                    fragment,
                    result
                );
            } else {
                // Normal style should not add any of the other style instructions
                assert!(
                    !result.contains("Be brief and concise."),
                    "Normal should not contain Concise instruction"
                );
                assert!(
                    !result.contains("Provide detailed explanations."),
                    "Normal should not contain Detailed instruction"
                );
                assert!(
                    !result.contains("Include technical details."),
                    "Normal should not contain Technical instruction"
                );
            }
        }
    }

    #[test]
    fn test_build_system_prompt_user_preferences() {
        let prefs = UserPreferences {
            ships_owned: vec!["Cutlass Black".to_string(), "Freelancer".to_string()],
            target_ship: Some("Constellation Andromeda".to_string()),
            interests: vec!["mining".to_string(), "trading".to_string()],
            ..UserPreferences::default()
        };

        let result = build_system_prompt("Base.", &prefs, "");

        assert!(
            result.contains("Cutlass Black"),
            "prompt should contain first ship owned"
        );
        assert!(
            result.contains("Freelancer"),
            "prompt should contain second ship owned"
        );
        assert!(
            result.contains("Constellation Andromeda"),
            "prompt should contain target ship"
        );
        assert!(
            result.contains("mining"),
            "prompt should contain first interest"
        );
        assert!(
            result.contains("trading"),
            "prompt should contain second interest"
        );
    }
}
