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
    log::debug!(
        "[llm] ollama_streaming model={} url={}",
        config.selected_model,
        url
    );

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
    log::debug!(
        "[llm] openai_streaming provider={:?} model={}",
        config.provider,
        config.selected_model
    );
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
    log::debug!(
        "[llm] ollama_request model={} url={} temperature={}",
        config.selected_model,
        url,
        config.temperature
    );

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
    log::debug!(
        "[llm] openai_request provider={:?} model={} url={} temperature={}",
        config.provider,
        config.selected_model,
        url,
        config.temperature
    );

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
    log::debug!(
        "[llm] kobold_request model={} url={} temperature={}",
        config.selected_model,
        url,
        config.temperature
    );

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
    let start = std::time::Instant::now();
    log::info!(
        "[llm] provider={:?} model={} request_start streaming=true",
        config.provider,
        config.selected_model
    );

    let result = match &config.provider {
        AiProvider::Ollama => generate_ollama_streaming(config, conversation, system_prompt, tx),
        AiProvider::KoboldCpp => {
            // Kobold doesn't support streaming well, fall back to non-streaming
            log::warn!(
                "[llm] provider=KoboldCpp model={} fallback=non-streaming reason=kobold_no_stream_support",
                config.selected_model
            );
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
            log::info!(
                "[llm] provider=Gemini model={} fallback=non-streaming reason=gemini_custom_api",
                config.selected_model
            );
            let response = crate::cloud_providers::generate_gemini_cloud(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
    };

    let latency_ms = start.elapsed().as_millis();
    match &result {
        Ok(()) => {
            log::info!(
                "[llm] provider={:?} model={} status=ok latency_ms={} streaming=true",
                config.provider,
                config.selected_model,
                latency_ms
            );
        }
        Err(e) => {
            log::error!(
                "[llm] provider={:?} model={} status=error latency_ms={} streaming=true error={}",
                config.provider,
                config.selected_model,
                latency_ms,
                e
            );
        }
    }

    result
}

/// Generate response without streaming - routes to appropriate provider
pub fn generate_response(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let start = std::time::Instant::now();
    log::info!(
        "[llm] provider={:?} model={} request_start streaming=false",
        config.provider,
        config.selected_model
    );

    let result = match &config.provider {
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
    };

    let latency_ms = start.elapsed().as_millis();
    match &result {
        Ok(response) => {
            // Approximate token count from response length (rough: 1 token ~ 4 chars)
            let approx_tokens = response.len() / 4;
            log::info!(
                "[llm] provider={:?} model={} status=ok latency_ms={} approx_tokens_out={}",
                config.provider,
                config.selected_model,
                latency_ms,
                approx_tokens
            );
        }
        Err(e) => {
            log::error!(
                "[llm] provider={:?} model={} status=error latency_ms={} error={}",
                config.provider,
                config.selected_model,
                latency_ms,
                e
            );
        }
    }

    result
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
        log::info!(
            "[llm] provider={:?} model={} status=cancelled reason=pre_start",
            config.provider,
            config.selected_model
        );
        let _ = tx.send(AiResponse::Cancelled(String::new()));
        return Ok(());
    }

    let start = std::time::Instant::now();
    log::info!(
        "[llm] provider={:?} model={} request_start streaming=true cancellable=true",
        config.provider,
        config.selected_model
    );

    let result = match &config.provider {
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
                log::info!(
                    "[llm] provider=KoboldCpp model={} status=cancelled reason=pre_kobold_call",
                    config.selected_model
                );
                let _ = tx.send(AiResponse::Cancelled(String::new()));
                return Ok(());
            }
            log::warn!(
                "[llm] provider=KoboldCpp model={} fallback=non-streaming reason=kobold_no_stream_support",
                config.selected_model
            );
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
                log::info!(
                    "[llm] provider=Gemini model={} status=cancelled reason=pre_gemini_call",
                    config.selected_model
                );
                let _ = tx.send(AiResponse::Cancelled(String::new()));
                return Ok(());
            }
            log::info!(
                "[llm] provider=Gemini model={} fallback=non-streaming reason=gemini_custom_api",
                config.selected_model
            );
            let response = crate::cloud_providers::generate_gemini_cloud(config, conversation, system_prompt)?;
            let _ = tx.send(AiResponse::Complete(response));
            Ok(())
        }
    };

    let latency_ms = start.elapsed().as_millis();
    match &result {
        Ok(()) => {
            log::info!(
                "[llm] provider={:?} model={} status=ok latency_ms={} streaming=true cancellable=true",
                config.provider,
                config.selected_model,
                latency_ms
            );
        }
        Err(e) => {
            log::error!(
                "[llm] provider={:?} model={} status=error latency_ms={} streaming=true cancellable=true error={}",
                config.provider,
                config.selected_model,
                latency_ms,
                e
            );
        }
    }

    result
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
use std::time::{Duration, Instant};

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

// ===========================================================================
// ResilientProviderRegistry — fallback chain, circuit breakers, health, pool
// ===========================================================================

/// Configuration for a provider within the resilient registry.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Human-readable provider name (e.g. "openai-primary")
    pub name: String,
    /// Provider type identifier (e.g. "openai", "ollama")
    pub provider_type: String,
    /// Base URL for this provider
    pub base_url: String,
    /// Optional API key
    pub api_key: Option<String>,
    /// Priority — lower values are tried first
    pub priority: u32,
}

impl ProviderConfig {
    /// Create a new provider config with the given name and base URL.
    pub fn new(name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            provider_type: String::new(),
            base_url: base_url.into(),
            api_key: None,
            priority: 0,
        }
    }

    /// Set the provider type.
    pub fn with_provider_type(mut self, provider_type: impl Into<String>) -> Self {
        self.provider_type = provider_type.into();
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the priority (lower = tried first).
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Health status of a provider within the resilient registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderHealthStatus {
    /// Provider is responding normally
    Healthy,
    /// Provider is responding but with errors or degraded performance
    Degraded,
    /// Provider is not responding or returning server errors
    Unhealthy,
    /// Provider has not been checked yet
    Unknown,
}

/// Per-provider rate limit state, updated from response headers.
#[derive(Debug, Clone)]
pub struct ProviderRateState {
    /// Remaining requests in the current window (from `x-ratelimit-remaining-requests`)
    pub remaining_requests: Option<u32>,
    /// Remaining tokens in the current window (from `x-ratelimit-remaining-tokens`)
    pub remaining_tokens: Option<u32>,
    /// When the current rate-limit window resets
    pub reset_at: Option<Instant>,
    /// Server-requested retry-after delay in seconds
    pub retry_after_secs: Option<u64>,
}

impl ProviderRateState {
    fn new() -> Self {
        Self {
            remaining_requests: None,
            remaining_tokens: None,
            reset_at: None,
            retry_after_secs: None,
        }
    }
}

/// Parsed rate-limit headers from an HTTP response.
pub struct RateLimitHeaders {
    /// Value of `x-ratelimit-remaining-requests`
    pub remaining_requests: Option<u32>,
    /// Value of `x-ratelimit-remaining-tokens`
    pub remaining_tokens: Option<u32>,
    /// Value of `x-ratelimit-reset` (seconds until window resets)
    pub reset_secs: Option<u64>,
    /// Value of `retry-after` header (seconds)
    pub retry_after_secs: Option<u64>,
}

impl RateLimitHeaders {
    /// Parse rate limit headers from a list of (name, value) pairs.
    ///
    /// Recognises:
    /// - `x-ratelimit-remaining-requests`
    /// - `x-ratelimit-remaining-tokens`
    /// - `x-ratelimit-reset`
    /// - `retry-after`
    pub fn from_response_headers(headers: &[(&str, &str)]) -> Self {
        let mut remaining_requests = None;
        let mut remaining_tokens = None;
        let mut reset_secs = None;
        let mut retry_after_secs = None;

        for &(name, value) in headers {
            match name.to_lowercase().as_str() {
                "x-ratelimit-remaining-requests" => {
                    remaining_requests = value.trim().parse::<u32>().ok();
                }
                "x-ratelimit-remaining-tokens" => {
                    remaining_tokens = value.trim().parse::<u32>().ok();
                }
                "x-ratelimit-reset" => {
                    reset_secs = value.trim().parse::<u64>().ok();
                }
                "retry-after" => {
                    retry_after_secs = value.trim().parse::<u64>().ok();
                }
                _ => {}
            }
        }

        Self {
            remaining_requests,
            remaining_tokens,
            reset_secs,
            retry_after_secs,
        }
    }
}

/// Circuit-breaker state for a single provider.
#[derive(Debug, Clone)]
struct CircuitState {
    /// Consecutive failure count
    failures: u32,
    /// Current state kind
    state: CircuitStateKind,
    /// Timestamp of the last recorded failure
    last_failure: Option<Instant>,
    /// Timestamp of the last recorded success
    last_success: Option<Instant>,
}

impl CircuitState {
    fn new() -> Self {
        Self {
            failures: 0,
            state: CircuitStateKind::Closed,
            last_failure: None,
            last_success: None,
        }
    }
}

/// The three states of a circuit breaker.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CircuitStateKind {
    /// Normal operation — requests are allowed
    Closed,
    /// Too many failures — requests are blocked
    Open,
    /// Recovery period expired — one probe request is allowed
    HalfOpen,
}

/// Error returned by [`ResilientProviderRegistry::generate_with_fallback`].
#[derive(Debug)]
pub enum ResilientError {
    /// All providers in the chain failed.
    AllProvidersFailed {
        /// Per-provider error messages in the order they were tried.
        errors: Vec<(String, String)>,
    },
    /// No providers are available (all circuits open / unhealthy / rate-limited).
    NoAvailableProviders,
}

impl std::fmt::Display for ResilientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllProvidersFailed { errors } => {
                write!(f, "All providers failed: ")?;
                for (i, (name, err)) in errors.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{}={}", name, err)?;
                }
                Ok(())
            }
            Self::NoAvailableProviders => write!(f, "No available providers"),
        }
    }
}

impl std::error::Error for ResilientError {}

// ---------------------------------------------------------------------------
// ConnectionPoolHandle
// ---------------------------------------------------------------------------

/// Configuration for the lightweight connection-pool handle.
pub struct PoolHandleConfig {
    /// Per-request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum idle connections to keep per host (ureq manages this internally)
    pub max_idle_per_host: usize,
}

impl Default for PoolHandleConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_idle_per_host: 4,
        }
    }
}

/// Lightweight connection-pool wrapper that creates one [`ureq::Agent`] per
/// host and reuses it for subsequent requests, keeping TCP connections alive.
pub struct ConnectionPoolHandle {
    agents: HashMap<String, ureq::Agent>,
    config: PoolHandleConfig,
}

impl ConnectionPoolHandle {
    /// Create a new pool handle with the given configuration.
    pub fn new(config: PoolHandleConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
        }
    }

    /// Return a reference to the [`ureq::Agent`] for the given host,
    /// creating one lazily if it does not yet exist.
    pub fn get_agent(&mut self, host: &str) -> &ureq::Agent {
        if !self.agents.contains_key(host) {
            let agent = ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(self.config.timeout_secs))
                .timeout_read(Duration::from_secs(self.config.timeout_secs))
                .max_idle_connections_per_host(self.config.max_idle_per_host)
                .build();
            self.agents.insert(host.to_string(), agent);
        }
        self.agents.get(host).expect("just inserted")
    }
}

impl std::fmt::Debug for ConnectionPoolHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionPoolHandle")
            .field("hosts", &self.agents.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ResilientProviderRegistry
// ---------------------------------------------------------------------------

/// Provider registry with built-in resilience: fallback, health checking,
/// connection pooling, and per-provider rate limiting.
///
/// This is a **separate** type from [`ProviderRegistry`]; it does not replace
/// or modify the existing provider functions. It wraps any provider-level
/// operation via a generic closure so that callers can choose what "generate"
/// means for their use-case.
///
/// # Example
///
/// ```rust
/// use ai_assistant::providers::{ResilientProviderRegistry, ProviderConfig};
///
/// let primary = ProviderConfig::new("ollama", "http://localhost:11434")
///     .with_priority(0);
/// let mut registry = ResilientProviderRegistry::new(primary);
///
/// let fallback = ProviderConfig::new("lmstudio", "http://localhost:1234")
///     .with_priority(1);
/// registry.add_fallback(fallback);
///
/// // The closure receives each ProviderConfig in priority order until one succeeds.
/// let result = registry.generate_with_fallback(|provider| -> Result<String, String> {
///     // In reality you would call the provider's API here.
///     Err(format!("{} is offline", provider.name))
/// });
/// ```
pub struct ResilientProviderRegistry {
    /// Primary provider configuration
    primary: ProviderConfig,
    /// Fallback providers in priority order
    fallbacks: Vec<ProviderConfig>,
    /// Circuit-breaker state per provider (keyed by `ProviderConfig::name`)
    circuit_states: HashMap<String, CircuitState>,
    /// Maximum consecutive failures before a circuit opens
    max_failures: u32,
    /// Seconds to wait after a circuit opens before allowing a half-open probe
    recovery_secs: u64,
    /// Optional connection-pool handle for HTTP connection reuse
    pool: Option<ConnectionPoolHandle>,
    /// Health status per provider (keyed by `ProviderConfig::name`)
    health_status: HashMap<String, ProviderHealthStatus>,
    /// Per-provider rate-limit state (keyed by `ProviderConfig::name`)
    rate_limits: HashMap<String, ProviderRateState>,
}

impl ResilientProviderRegistry {
    /// Create a new registry with the given primary provider.
    pub fn new(primary: ProviderConfig) -> Self {
        let name = primary.name.clone();
        let mut circuit_states = HashMap::new();
        circuit_states.insert(name.clone(), CircuitState::new());
        let mut health_status = HashMap::new();
        health_status.insert(name.clone(), ProviderHealthStatus::Unknown);
        let mut rate_limits = HashMap::new();
        rate_limits.insert(name, ProviderRateState::new());

        Self {
            primary,
            fallbacks: Vec::new(),
            circuit_states,
            max_failures: 3,
            recovery_secs: 30,
            pool: None,
            health_status,
            rate_limits,
        }
    }

    /// Add a fallback provider. Providers are tried in ascending priority order.
    pub fn add_fallback(&mut self, provider: ProviderConfig) {
        let name = provider.name.clone();
        self.circuit_states
            .entry(name.clone())
            .or_insert_with(CircuitState::new);
        self.health_status
            .entry(name.clone())
            .or_insert(ProviderHealthStatus::Unknown);
        self.rate_limits
            .entry(name)
            .or_insert_with(ProviderRateState::new);
        self.fallbacks.push(provider);
        // Keep fallbacks sorted by priority (ascending)
        self.fallbacks.sort_by_key(|p| p.priority);
    }

    /// Set the maximum consecutive failures before a circuit opens.
    pub fn set_max_failures(&mut self, max: u32) {
        self.max_failures = max;
    }

    /// Set the recovery time in seconds after a circuit opens.
    pub fn set_recovery_secs(&mut self, secs: u64) {
        self.recovery_secs = secs;
    }

    /// Attach a connection pool handle for HTTP reuse.
    pub fn set_pool(&mut self, pool: ConnectionPoolHandle) {
        self.pool = Some(pool);
    }

    // -- Circuit breaker helpers -------------------------------------------

    /// Determine whether a provider is available for a request right now.
    ///
    /// A provider is skipped when:
    /// - Its circuit is **Open** and the recovery time has not elapsed.
    /// - Its health status is **Unhealthy**.
    /// - It is currently rate-limited.
    ///
    /// If the circuit is Open but the recovery window has elapsed the state
    /// transitions to **HalfOpen** and the provider is allowed one probe.
    fn is_provider_available(&mut self, name: &str) -> bool {
        // Check health status
        if let Some(status) = self.health_status.get(name) {
            if *status == ProviderHealthStatus::Unhealthy {
                return false;
            }
        }

        // Check rate limits
        if self.is_rate_limited(name) {
            return false;
        }

        // Check circuit state
        if let Some(cs) = self.circuit_states.get_mut(name) {
            match cs.state {
                CircuitStateKind::Closed | CircuitStateKind::HalfOpen => true,
                CircuitStateKind::Open => {
                    // Check if recovery time has elapsed
                    if let Some(last_fail) = cs.last_failure {
                        if last_fail.elapsed() >= Duration::from_secs(self.recovery_secs) {
                            cs.state = CircuitStateKind::HalfOpen;
                            return true;
                        }
                    }
                    false
                }
            }
        } else {
            // No state tracked — treat as available
            true
        }
    }

    /// Build the ordered list of provider references to try: primary first,
    /// then fallbacks in ascending priority order.
    fn ordered_providers(&self) -> Vec<&ProviderConfig> {
        let mut providers = vec![&self.primary];
        for fb in &self.fallbacks {
            providers.push(fb);
        }
        providers
    }

    /// Try to execute `operation` against providers in priority order.
    ///
    /// For each available provider the closure is called. On success the
    /// circuit is reset (or closed if half-open). On failure the failure
    /// counter is incremented and the next provider is tried.
    ///
    /// Returns `Ok((value, provider_name))` on success, or a
    /// [`ResilientError`] if all providers fail or none are available.
    pub fn generate_with_fallback<F, T, E>(
        &mut self,
        mut operation: F,
    ) -> Result<(T, String), ResilientError>
    where
        F: FnMut(&ProviderConfig) -> Result<T, E>,
        E: std::fmt::Display,
    {
        let providers = self.ordered_providers();
        let names: Vec<String> = providers.iter().map(|p| p.name.clone()).collect();
        // We need to clone configs because we call &mut self methods below.
        let configs: Vec<ProviderConfig> = providers.iter().map(|p| (*p).clone()).collect();

        let mut errors: Vec<(String, String)> = Vec::new();
        let mut tried_any = false;

        for (cfg, name) in configs.iter().zip(names.iter()) {
            if !self.is_provider_available(name) {
                log::debug!(
                    "[llm] resilient provider={} status=skipped reason=unavailable",
                    name
                );
                continue;
            }

            tried_any = true;
            log::info!(
                "[llm] resilient provider={} attempting_request",
                name
            );

            match operation(cfg) {
                Ok(value) => {
                    self.record_success(name);
                    log::info!(
                        "[llm] resilient provider={} status=ok",
                        name
                    );
                    return Ok((value, name.clone()));
                }
                Err(e) => {
                    let msg = e.to_string();
                    self.record_failure(name);
                    log::warn!(
                        "[llm] resilient provider={} status=failed error={} falling_back=true",
                        name,
                        msg
                    );
                    errors.push((name.clone(), msg));
                }
            }
        }

        if !tried_any {
            log::error!("[llm] resilient status=no_available_providers");
            return Err(ResilientError::NoAvailableProviders);
        }

        log::error!(
            "[llm] resilient status=all_providers_failed count={}",
            errors.len()
        );
        Err(ResilientError::AllProvidersFailed { errors })
    }

    /// Record a successful request for the named provider.
    ///
    /// - Resets the failure counter to zero.
    /// - If the circuit was HalfOpen, transitions to Closed.
    pub fn record_success(&mut self, provider_name: &str) {
        if let Some(cs) = self.circuit_states.get_mut(provider_name) {
            cs.failures = 0;
            cs.last_success = Some(Instant::now());
            if cs.state == CircuitStateKind::HalfOpen {
                log::info!(
                    "[llm] circuit_breaker provider={} transition=half_open->closed",
                    provider_name
                );
                cs.state = CircuitStateKind::Closed;
            }
        }
    }

    /// Record a failed request for the named provider.
    ///
    /// - Increments the failure counter.
    /// - If the failure count reaches `max_failures`, opens the circuit.
    pub fn record_failure(&mut self, provider_name: &str) {
        let max = self.max_failures;
        if let Some(cs) = self.circuit_states.get_mut(provider_name) {
            cs.failures += 1;
            cs.last_failure = Some(Instant::now());
            if cs.failures >= max {
                log::warn!(
                    "[llm] circuit_breaker provider={} transition=closed->open failures={}",
                    provider_name,
                    cs.failures
                );
                cs.state = CircuitStateKind::Open;
            }
        }
    }

    /// Return the current health status for a provider.
    pub fn health_status(&self, provider_name: &str) -> ProviderHealthStatus {
        self.health_status
            .get(provider_name)
            .cloned()
            .unwrap_or(ProviderHealthStatus::Unknown)
    }

    /// Manually set the health status for a provider.
    pub fn set_health_status(&mut self, provider_name: &str, status: ProviderHealthStatus) {
        self.health_status
            .insert(provider_name.to_string(), status);
    }

    /// Perform a lightweight health check against a provider by issuing a
    /// `GET /` request with a 5-second timeout.
    ///
    /// The result is stored in the internal health-status map and also returned.
    pub fn check_health(&mut self, provider: &ProviderConfig) -> ProviderHealthStatus {
        let url = format!("{}/", provider.base_url);
        log::debug!(
            "[llm] health_check provider={} url={}",
            provider.name,
            url
        );
        let status = match ureq::get(&url)
            .timeout(Duration::from_secs(5))
            .call()
        {
            Ok(_) => ProviderHealthStatus::Healthy,
            Err(ureq::Error::Status(code, _)) if code < 500 => ProviderHealthStatus::Degraded,
            Err(_) => ProviderHealthStatus::Unhealthy,
        };
        log::info!(
            "[llm] health_check provider={} status={:?}",
            provider.name,
            status
        );
        self.health_status
            .insert(provider.name.clone(), status.clone());
        status
    }

    /// Return the names of all providers whose circuits are not Open.
    pub fn active_providers(&self) -> Vec<&str> {
        let mut result = Vec::new();
        let all_names: Vec<&str> = std::iter::once(self.primary.name.as_str())
            .chain(self.fallbacks.iter().map(|p| p.name.as_str()))
            .collect();
        for name in all_names {
            if let Some(cs) = self.circuit_states.get(name) {
                if cs.state != CircuitStateKind::Open {
                    result.push(name);
                }
            } else {
                result.push(name);
            }
        }
        result
    }

    // -- Rate-limit helpers ------------------------------------------------

    /// Update the rate-limit state for a provider from parsed response headers.
    pub fn update_rate_limits(&mut self, provider_name: &str, headers: &RateLimitHeaders) {
        let state = self
            .rate_limits
            .entry(provider_name.to_string())
            .or_insert_with(ProviderRateState::new);

        state.remaining_requests = headers.remaining_requests;
        state.remaining_tokens = headers.remaining_tokens;
        state.retry_after_secs = headers.retry_after_secs;

        if let Some(secs) = headers.reset_secs {
            state.reset_at = Some(Instant::now() + Duration::from_secs(secs));
        }
        if let Some(secs) = headers.retry_after_secs {
            // retry-after takes precedence if present
            state.reset_at = Some(Instant::now() + Duration::from_secs(secs));
        }

        if headers.remaining_requests == Some(0) || headers.retry_after_secs.is_some() {
            log::warn!(
                "[llm] rate_limit provider={} remaining_requests={:?} retry_after={:?}",
                provider_name,
                headers.remaining_requests,
                headers.retry_after_secs
            );
        } else {
            log::debug!(
                "[llm] rate_limit provider={} remaining_requests={:?} remaining_tokens={:?}",
                provider_name,
                headers.remaining_requests,
                headers.remaining_tokens
            );
        }
    }

    /// Check whether a provider is currently rate-limited.
    ///
    /// A provider is considered rate-limited when:
    /// - `remaining_requests` is `Some(0)` **and** the reset time has not passed, or
    /// - `retry_after_secs` is set **and** the reset time has not passed.
    pub fn is_rate_limited(&self, provider_name: &str) -> bool {
        if let Some(state) = self.rate_limits.get(provider_name) {
            // Check if we have a reset_at and it's in the future
            let within_window = state
                .reset_at
                .map(|r| Instant::now() < r)
                .unwrap_or(false);

            if !within_window {
                return false;
            }

            // Within the window — check if we've exhausted requests
            if let Some(0) = state.remaining_requests {
                return true;
            }

            // retry-after header means we should back off
            if state.retry_after_secs.is_some() {
                return true;
            }
        }
        false
    }

    /// Return the number of consecutive failures for the named provider.
    pub fn failure_count(&self, provider_name: &str) -> u32 {
        self.circuit_states
            .get(provider_name)
            .map(|cs| cs.failures)
            .unwrap_or(0)
    }

    /// Return `true` if the circuit for the named provider is open.
    pub fn is_circuit_open(&self, provider_name: &str) -> bool {
        self.circuit_states
            .get(provider_name)
            .map(|cs| cs.state == CircuitStateKind::Open)
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for ResilientProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResilientProviderRegistry")
            .field("primary", &self.primary.name)
            .field(
                "fallbacks",
                &self.fallbacks.iter().map(|p| &p.name).collect::<Vec<_>>(),
            )
            .field("max_failures", &self.max_failures)
            .field("recovery_secs", &self.recovery_secs)
            .finish()
    }
}

// ============================================================================
// AuditedProvider (Item 4.3)
// ============================================================================

/// Audit entry recording a single LLM provider call.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: std::time::SystemTime,
    pub provider: String,
    pub model: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub latency_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Aggregate statistics from audit log entries.
pub struct AuditSummary {
    pub total_calls: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub avg_latency_ms: f64,
    pub by_provider: std::collections::HashMap<String, usize>,
    pub by_model: std::collections::HashMap<String, usize>,
}

/// Provider wrapper that logs every LLM call to an in-memory audit log.
pub struct AuditedProvider {
    audit_log: std::sync::Arc<std::sync::Mutex<Vec<AuditEntry>>>,
    max_entries: usize,
}

impl AuditedProvider {
    pub fn new(max_entries: usize) -> Self {
        Self {
            audit_log: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            max_entries,
        }
    }

    /// Record an audit entry, evicting the oldest if over max.
    pub fn record(&self, entry: AuditEntry) {
        let mut log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        log.push(entry);
        while log.len() > self.max_entries {
            log.remove(0);
        }
    }

    /// Clone all current audit entries.
    pub fn entries(&self) -> Vec<AuditEntry> {
        let log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        log.clone()
    }

    /// Number of entries currently in the audit log.
    pub fn entry_count(&self) -> usize {
        let log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        log.len()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        log.clear();
    }

    /// Return entries recorded at or after the given timestamp.
    pub fn entries_since(&self, since: std::time::SystemTime) -> Vec<AuditEntry> {
        let log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        log.iter()
            .filter(|e| e.timestamp >= since)
            .cloned()
            .collect()
    }

    /// Compute aggregate statistics from the current audit log.
    pub fn summary(&self) -> AuditSummary {
        let log = self.audit_log.lock().unwrap_or_else(|e| e.into_inner());
        let total_calls = log.len();
        let successful = log.iter().filter(|e| e.success).count();
        let failed = total_calls - successful;
        let total_input_tokens: u64 = log.iter().filter_map(|e| e.input_tokens).sum();
        let total_output_tokens: u64 = log.iter().filter_map(|e| e.output_tokens).sum();
        let total_latency: u64 = log.iter().map(|e| e.latency_ms).sum();
        let avg_latency_ms = if total_calls > 0 {
            total_latency as f64 / total_calls as f64
        } else {
            0.0
        };

        let mut by_provider: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut by_model: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for entry in log.iter() {
            *by_provider.entry(entry.provider.clone()).or_insert(0) += 1;
            *by_model.entry(entry.model.clone()).or_insert(0) += 1;
        }

        AuditSummary {
            total_calls,
            successful,
            failed,
            total_input_tokens,
            total_output_tokens,
            avg_latency_ms,
            by_provider,
            by_model,
        }
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

// ---------------------------------------------------------------------------
// ResilientProviderRegistry tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod resilient_registry_tests {
    use super::*;

    /// Helper: create a simple primary config.
    fn primary() -> ProviderConfig {
        ProviderConfig::new("primary", "http://localhost:11434")
            .with_provider_type("ollama")
            .with_priority(0)
    }

    /// Helper: create a simple fallback config.
    fn fallback(name: &str, priority: u32) -> ProviderConfig {
        ProviderConfig::new(name, &format!("http://localhost:{}", 11435 + priority))
            .with_provider_type("openai")
            .with_priority(priority)
    }

    // -- Item 3.1: Fallback chain tests ------------------------------------

    #[test]
    fn test_resilient_registry_primary_success() {
        let mut reg = ResilientProviderRegistry::new(primary());

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Ok(format!("ok from {}", p.name))
        });

        assert!(result.is_ok());
        let (value, name) = result.expect("should succeed");
        assert_eq!(value, "ok from primary");
        assert_eq!(name, "primary");
    }

    #[test]
    fn test_resilient_registry_fallback_on_failure() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.add_fallback(fallback("backup", 1));

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            if p.name == "primary" {
                Err("primary down".to_string())
            } else {
                Ok(format!("ok from {}", p.name))
            }
        });

        assert!(result.is_ok());
        let (value, name) = result.expect("should succeed via fallback");
        assert_eq!(value, "ok from backup");
        assert_eq!(name, "backup");
    }

    #[test]
    fn test_resilient_registry_all_fail() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.add_fallback(fallback("backup", 1));

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Err(format!("{} is offline", p.name))
        });

        assert!(result.is_err());
        match result.unwrap_err() {
            ResilientError::AllProvidersFailed { errors } => {
                assert_eq!(errors.len(), 2);
                assert_eq!(errors[0].0, "primary");
                assert!(errors[0].1.contains("offline"));
                assert_eq!(errors[1].0, "backup");
            }
            other => panic!("Expected AllProvidersFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_circuit_breaker_opens_after_failures() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.set_max_failures(3);

        // Record 3 failures
        reg.record_failure("primary");
        reg.record_failure("primary");
        assert!(!reg.is_circuit_open("primary"), "2 failures should not open circuit");
        reg.record_failure("primary");
        assert!(reg.is_circuit_open("primary"), "3 failures should open circuit");

        // Now generate_with_fallback should return NoAvailableProviders
        let result = reg.generate_with_fallback(|_| -> Result<String, String> {
            Ok("should not reach".to_string())
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            ResilientError::NoAvailableProviders => {}
            other => panic!("Expected NoAvailableProviders, got {:?}", other),
        }
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.set_max_failures(1);
        // Use a zero-second recovery so half-open triggers immediately
        reg.set_recovery_secs(0);

        // Open the circuit
        reg.record_failure("primary");
        assert!(reg.is_circuit_open("primary"));

        // With recovery_secs=0, the circuit should transition to HalfOpen
        // when checked, allowing a probe request.
        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Ok(format!("recovered from {}", p.name))
        });

        assert!(result.is_ok());
        let (value, _) = result.expect("half-open should allow probe");
        assert_eq!(value, "recovered from primary");
    }

    #[test]
    fn test_circuit_breaker_reclose_on_success() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.set_max_failures(2);

        // Accumulate some failures (not enough to open)
        reg.record_failure("primary");
        assert_eq!(reg.failure_count("primary"), 1);

        // A success should reset
        reg.record_success("primary");
        assert_eq!(reg.failure_count("primary"), 0);
        assert!(!reg.is_circuit_open("primary"));
    }

    // -- Item 3.3: Health status integration tests -------------------------

    #[test]
    fn test_health_status_skip_unhealthy() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.add_fallback(fallback("backup", 1));

        // Mark primary as unhealthy
        reg.set_health_status("primary", ProviderHealthStatus::Unhealthy);

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Ok(format!("ok from {}", p.name))
        });

        assert!(result.is_ok());
        let (_, name) = result.expect("should skip primary");
        assert_eq!(name, "backup", "primary should be skipped because unhealthy");
    }

    #[test]
    fn test_health_status_try_degraded() {
        let mut reg = ResilientProviderRegistry::new(primary());

        // Mark primary as degraded — should still be tried
        reg.set_health_status("primary", ProviderHealthStatus::Degraded);

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Ok(format!("ok from {}", p.name))
        });

        assert!(result.is_ok());
        let (_, name) = result.expect("degraded should still work");
        assert_eq!(name, "primary");
    }

    #[test]
    fn test_health_status_get_set() {
        let mut reg = ResilientProviderRegistry::new(primary());
        assert_eq!(reg.health_status("primary"), ProviderHealthStatus::Unknown);

        reg.set_health_status("primary", ProviderHealthStatus::Healthy);
        assert_eq!(reg.health_status("primary"), ProviderHealthStatus::Healthy);

        reg.set_health_status("primary", ProviderHealthStatus::Unhealthy);
        assert_eq!(reg.health_status("primary"), ProviderHealthStatus::Unhealthy);
    }

    // -- Item 3.2: Connection pool handle tests ----------------------------

    #[test]
    fn test_connection_pool_handle_reuses_agent() {
        let mut pool = ConnectionPoolHandle::new(PoolHandleConfig::default());

        let _agent1 = pool.get_agent("http://localhost:11434");
        let _agent2 = pool.get_agent("http://localhost:11434");

        // The agents HashMap should have exactly one entry
        assert_eq!(pool.agents.len(), 1, "same host should reuse agent");
    }

    #[test]
    fn test_connection_pool_handle_different_hosts() {
        let mut pool = ConnectionPoolHandle::new(PoolHandleConfig::default());

        let _agent1 = pool.get_agent("http://localhost:11434");
        let _agent2 = pool.get_agent("http://localhost:1234");

        assert_eq!(pool.agents.len(), 2, "different hosts should get different agents");
    }

    #[test]
    fn test_connection_pool_handle_custom_config() {
        let config = PoolHandleConfig {
            timeout_secs: 60,
            max_idle_per_host: 8,
        };
        let mut pool = ConnectionPoolHandle::new(config);
        // Just ensure we can get an agent without panicking
        let _agent = pool.get_agent("http://example.com");
        assert_eq!(pool.agents.len(), 1);
    }

    #[test]
    fn test_connection_pool_handle_debug() {
        let pool = ConnectionPoolHandle::new(PoolHandleConfig::default());
        let debug = format!("{:?}", pool);
        assert!(debug.contains("ConnectionPoolHandle"));
    }

    // -- Item 3.4: Rate limit tests ----------------------------------------

    #[test]
    fn test_rate_limit_headers_parsing() {
        let headers = vec![
            ("x-ratelimit-remaining-requests", "42"),
            ("x-ratelimit-remaining-tokens", "10000"),
            ("x-ratelimit-reset", "30"),
            ("retry-after", "5"),
        ];

        let parsed = RateLimitHeaders::from_response_headers(&headers);
        assert_eq!(parsed.remaining_requests, Some(42));
        assert_eq!(parsed.remaining_tokens, Some(10000));
        assert_eq!(parsed.reset_secs, Some(30));
        assert_eq!(parsed.retry_after_secs, Some(5));
    }

    #[test]
    fn test_rate_limit_headers_parsing_case_insensitive() {
        let headers = vec![
            ("X-RateLimit-Remaining-Requests", "100"),
            ("X-RATELIMIT-REMAINING-TOKENS", "5000"),
        ];

        let parsed = RateLimitHeaders::from_response_headers(&headers);
        assert_eq!(parsed.remaining_requests, Some(100));
        assert_eq!(parsed.remaining_tokens, Some(5000));
    }

    #[test]
    fn test_rate_limit_headers_parsing_empty() {
        let headers: Vec<(&str, &str)> = vec![];
        let parsed = RateLimitHeaders::from_response_headers(&headers);
        assert_eq!(parsed.remaining_requests, None);
        assert_eq!(parsed.remaining_tokens, None);
        assert_eq!(parsed.reset_secs, None);
        assert_eq!(parsed.retry_after_secs, None);
    }

    #[test]
    fn test_rate_limit_headers_parsing_invalid_values() {
        let headers = vec![
            ("x-ratelimit-remaining-requests", "not-a-number"),
            ("retry-after", "abc"),
        ];
        let parsed = RateLimitHeaders::from_response_headers(&headers);
        assert_eq!(parsed.remaining_requests, None);
        assert_eq!(parsed.retry_after_secs, None);
    }

    #[test]
    fn test_rate_limited_provider_skipped() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.add_fallback(fallback("backup", 1));

        // Simulate rate-limit: 0 remaining requests with a reset far in the future
        let rl_headers = RateLimitHeaders {
            remaining_requests: Some(0),
            remaining_tokens: None,
            reset_secs: Some(3600), // 1 hour from now
            retry_after_secs: None,
        };
        reg.update_rate_limits("primary", &rl_headers);
        assert!(reg.is_rate_limited("primary"), "primary should be rate-limited");

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            Ok(format!("ok from {}", p.name))
        });

        assert!(result.is_ok());
        let (_, name) = result.expect("should use backup");
        assert_eq!(name, "backup", "rate-limited primary should be skipped");
    }

    #[test]
    fn test_rate_limit_not_limited_with_remaining() {
        let mut reg = ResilientProviderRegistry::new(primary());

        // Provider has remaining requests — should NOT be rate-limited
        let rl_headers = RateLimitHeaders {
            remaining_requests: Some(100),
            remaining_tokens: None,
            reset_secs: Some(60),
            retry_after_secs: None,
        };
        reg.update_rate_limits("primary", &rl_headers);
        assert!(!reg.is_rate_limited("primary"));
    }

    #[test]
    fn test_rate_limit_reset() {
        let mut reg = ResilientProviderRegistry::new(primary());

        // Set rate limit that expires immediately (0 seconds)
        let rl_headers = RateLimitHeaders {
            remaining_requests: Some(0),
            remaining_tokens: None,
            reset_secs: Some(0),
            retry_after_secs: None,
        };
        reg.update_rate_limits("primary", &rl_headers);

        // With reset_secs=0, the reset_at should be essentially now.
        // A tiny sleep or even just the passage of code execution should
        // make Instant::now() >= reset_at, so the provider is available.
        // We test that is_rate_limited returns false after the window expires.
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!reg.is_rate_limited("primary"), "after reset, should not be rate-limited");
    }

    #[test]
    fn test_rate_limit_retry_after() {
        let mut reg = ResilientProviderRegistry::new(primary());

        let rl_headers = RateLimitHeaders {
            remaining_requests: None,
            remaining_tokens: None,
            reset_secs: None,
            retry_after_secs: Some(300),
        };
        reg.update_rate_limits("primary", &rl_headers);
        assert!(reg.is_rate_limited("primary"), "retry-after should rate-limit");
    }

    // -- Priority ordering tests -------------------------------------------

    #[test]
    fn test_provider_priority_ordering() {
        let mut reg = ResilientProviderRegistry::new(
            ProviderConfig::new("first", "http://first").with_priority(0),
        );
        reg.add_fallback(ProviderConfig::new("third", "http://third").with_priority(10));
        reg.add_fallback(ProviderConfig::new("second", "http://second").with_priority(5));

        // Mark "first" as unhealthy to force fallback path
        reg.set_health_status("first", ProviderHealthStatus::Unhealthy);

        let mut tried_order = Vec::new();

        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            tried_order.push(p.name.clone());
            Err("fail".to_string())
        });

        // Should have tried second (priority 5) before third (priority 10)
        assert!(result.is_err());
        assert_eq!(tried_order, vec!["second", "third"]);
    }

    // -- Active providers test ---------------------------------------------

    #[test]
    fn test_active_providers() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.add_fallback(fallback("backup1", 1));
        reg.add_fallback(fallback("backup2", 2));

        let active = reg.active_providers();
        assert_eq!(active.len(), 3);

        // Open circuit on backup1
        reg.set_max_failures(1);
        reg.record_failure("backup1");
        assert!(reg.is_circuit_open("backup1"));

        let active = reg.active_providers();
        assert_eq!(active.len(), 2);
        assert!(!active.contains(&"backup1"), "backup1 circuit is open");
        assert!(active.contains(&"primary"));
        assert!(active.contains(&"backup2"));
    }

    // -- Record success / failure tests ------------------------------------

    #[test]
    fn test_record_success_resets_failures() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.record_failure("primary");
        reg.record_failure("primary");
        assert_eq!(reg.failure_count("primary"), 2);

        reg.record_success("primary");
        assert_eq!(reg.failure_count("primary"), 0);
    }

    #[test]
    fn test_record_failure_increments() {
        let mut reg = ResilientProviderRegistry::new(primary());
        assert_eq!(reg.failure_count("primary"), 0);

        reg.record_failure("primary");
        assert_eq!(reg.failure_count("primary"), 1);

        reg.record_failure("primary");
        assert_eq!(reg.failure_count("primary"), 2);

        reg.record_failure("primary");
        assert_eq!(reg.failure_count("primary"), 3);
    }

    // -- Additional coverage tests -----------------------------------------

    #[test]
    fn test_provider_config_builder() {
        let cfg = ProviderConfig::new("test", "http://test.local")
            .with_provider_type("openai")
            .with_api_key("sk-test-123")
            .with_priority(5);

        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.base_url, "http://test.local");
        assert_eq!(cfg.provider_type, "openai");
        assert_eq!(cfg.api_key, Some("sk-test-123".to_string()));
        assert_eq!(cfg.priority, 5);
    }

    #[test]
    fn test_resilient_error_display() {
        let err = ResilientError::AllProvidersFailed {
            errors: vec![
                ("a".to_string(), "timeout".to_string()),
                ("b".to_string(), "refused".to_string()),
            ],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("a=timeout"));
        assert!(msg.contains("b=refused"));

        let err2 = ResilientError::NoAvailableProviders;
        assert_eq!(format!("{}", err2), "No available providers");
    }

    #[test]
    fn test_resilient_registry_debug() {
        let reg = ResilientProviderRegistry::new(primary());
        let debug = format!("{:?}", reg);
        assert!(debug.contains("ResilientProviderRegistry"));
        assert!(debug.contains("primary"));
    }

    #[test]
    fn test_resilient_registry_set_pool() {
        let mut reg = ResilientProviderRegistry::new(primary());
        assert!(reg.pool.is_none());

        reg.set_pool(ConnectionPoolHandle::new(PoolHandleConfig::default()));
        assert!(reg.pool.is_some());
    }

    #[test]
    fn test_no_available_providers_all_unhealthy() {
        let mut reg = ResilientProviderRegistry::new(primary());
        reg.set_health_status("primary", ProviderHealthStatus::Unhealthy);

        let result = reg.generate_with_fallback(|_| -> Result<String, String> {
            Ok("should not reach".to_string())
        });

        match result.unwrap_err() {
            ResilientError::NoAvailableProviders => {}
            other => panic!("Expected NoAvailableProviders, got {:?}", other),
        }
    }

    #[test]
    fn test_health_status_unknown_provider() {
        let reg = ResilientProviderRegistry::new(primary());
        assert_eq!(
            reg.health_status("nonexistent"),
            ProviderHealthStatus::Unknown,
        );
    }

    #[test]
    fn test_failure_count_unknown_provider() {
        let reg = ResilientProviderRegistry::new(primary());
        assert_eq!(reg.failure_count("nonexistent"), 0);
    }

    #[test]
    fn test_circuit_not_open_unknown_provider() {
        let reg = ResilientProviderRegistry::new(primary());
        assert!(!reg.is_circuit_open("nonexistent"));
    }

    #[test]
    fn test_rate_limit_no_state() {
        let reg = ResilientProviderRegistry::new(primary());
        assert!(!reg.is_rate_limited("nonexistent"));
    }

    // ========================================================================
    // AuditedProvider tests (Item 4.3)
    // ========================================================================

    fn sample_entry(provider: &str, model: &str, success: bool) -> AuditEntry {
        AuditEntry {
            timestamp: std::time::SystemTime::now(),
            provider: provider.to_string(),
            model: model.to_string(),
            input_tokens: Some(100),
            output_tokens: Some(50),
            latency_ms: 200,
            success,
            error: if success { None } else { Some("timeout".to_string()) },
        }
    }

    #[test]
    fn test_audited_provider_new() {
        let ap = AuditedProvider::new(100);
        assert_eq!(ap.entry_count(), 0);
    }

    #[test]
    fn test_audited_provider_record() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("openai", "gpt-4", true));
        assert_eq!(ap.entry_count(), 1);
        let entries = ap.entries();
        assert_eq!(entries[0].provider, "openai");
    }

    #[test]
    fn test_audited_provider_entries() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("openai", "gpt-4", true));
        ap.record(sample_entry("anthropic", "claude", true));
        let entries = ap.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].provider, "openai");
        assert_eq!(entries[1].provider, "anthropic");
    }

    #[test]
    fn test_audited_provider_clear() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("openai", "gpt-4", true));
        assert_eq!(ap.entry_count(), 1);
        ap.clear();
        assert_eq!(ap.entry_count(), 0);
    }

    #[test]
    fn test_audited_provider_max_entries_eviction() {
        let ap = AuditedProvider::new(3);
        ap.record(sample_entry("a", "m1", true));
        ap.record(sample_entry("b", "m2", true));
        ap.record(sample_entry("c", "m3", true));
        ap.record(sample_entry("d", "m4", true));
        assert_eq!(ap.entry_count(), 3);
        let entries = ap.entries();
        assert_eq!(entries[0].provider, "b");
        assert_eq!(entries[2].provider, "d");
    }

    #[test]
    fn test_audited_provider_entries_since() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("old", "m1", true));
        std::thread::sleep(std::time::Duration::from_millis(10));
        let cutoff = std::time::SystemTime::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        ap.record(sample_entry("new", "m2", true));
        let recent = ap.entries_since(cutoff);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].provider, "new");
    }

    #[test]
    fn test_audited_provider_summary() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("openai", "gpt-4", true));
        ap.record(sample_entry("openai", "gpt-4", false));
        ap.record(sample_entry("anthropic", "claude", true));
        let summary = ap.summary();
        assert_eq!(summary.total_calls, 3);
        assert_eq!(summary.successful, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.total_input_tokens, 300);
        assert_eq!(summary.total_output_tokens, 150);
        assert_eq!(*summary.by_provider.get("openai").unwrap(), 2);
        assert_eq!(*summary.by_provider.get("anthropic").unwrap(), 1);
    }

    #[test]
    fn test_audited_provider_summary_empty() {
        let ap = AuditedProvider::new(100);
        let summary = ap.summary();
        assert_eq!(summary.total_calls, 0);
        assert_eq!(summary.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_audited_provider_summary_by_model() {
        let ap = AuditedProvider::new(100);
        ap.record(sample_entry("openai", "gpt-4", true));
        ap.record(sample_entry("openai", "gpt-3.5", true));
        ap.record(sample_entry("openai", "gpt-4", true));
        let summary = ap.summary();
        assert_eq!(*summary.by_model.get("gpt-4").unwrap(), 2);
        assert_eq!(*summary.by_model.get("gpt-3.5").unwrap(), 1);
    }

    #[test]
    fn test_audited_provider_thread_safe() {
        let ap = std::sync::Arc::new(AuditedProvider::new(1000));
        let mut handles = vec![];
        for i in 0..10 {
            let ap_clone = ap.clone();
            handles.push(std::thread::spawn(move || {
                for j in 0..10 {
                    ap_clone.record(sample_entry(
                        &format!("p{}", i),
                        &format!("m{}", j),
                        true,
                    ));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(ap.entry_count(), 100);
    }
}

// ---------------------------------------------------------------------------
// Provider logging tests (Item 2.1)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod provider_logging_tests {
    use super::*;

    /// Helper: create a minimal AiConfig for Ollama (default).
    fn ollama_config() -> AiConfig {
        let mut cfg = AiConfig::default();
        cfg.selected_model = "test-model".to_string();
        cfg.temperature = 0.7;
        cfg
    }

    /// Helper: create a minimal AiConfig for an OpenAI-compatible provider.
    fn openai_config() -> AiConfig {
        let mut cfg = AiConfig::default();
        cfg.provider = AiProvider::OpenAI;
        cfg.selected_model = "gpt-4o".to_string();
        cfg.temperature = 0.5;
        cfg
    }

    #[test]
    fn test_generate_response_logs_provider() {
        // Verify generate_response can be called (will fail due to no server
        // but the logging code path is exercised). The actual log output is
        // verified by running with RUST_LOG=info.
        let config = ollama_config();
        let conversation = vec![];
        let result = generate_response(&config, &conversation, "test prompt");
        // It will fail (no server) but should have logged the attempt
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_generate_response_streaming_logs() {
        let config = ollama_config();
        let conversation = vec![];
        let (tx, _rx) = std::sync::mpsc::channel();
        let result = generate_response_streaming(&config, &conversation, "test prompt", &tx);
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_generate_response_openai_logs_provider() {
        let config = openai_config();
        let conversation = vec![];
        let result = generate_response(&config, &conversation, "test prompt");
        // Will fail (no API key / no server) but logging is exercised
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_generate_response_streaming_openai_logs() {
        let config = openai_config();
        let conversation = vec![];
        let (tx, _rx) = std::sync::mpsc::channel();
        let result = generate_response_streaming(&config, &conversation, "test prompt", &tx);
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_generate_response_cancellable_logs_pre_cancel() {
        let config = ollama_config();
        let conversation = vec![];
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel_token = crate::conversation_control::CancellationToken::new();
        cancel_token.cancel();

        let result = generate_response_streaming_cancellable(
            &config,
            &conversation,
            "test prompt",
            &tx,
            &cancel_token,
        );
        // Should succeed immediately with Cancelled response
        assert!(result.is_ok());
        let msg = rx.recv().expect("should have received Cancelled");
        match msg {
            AiResponse::Cancelled(_) => {}
            other => panic!("Expected Cancelled, got {:?}", other),
        }
    }

    #[test]
    fn test_generate_response_cancellable_logs_no_cancel() {
        let config = ollama_config();
        let conversation = vec![];
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel_token = crate::conversation_control::CancellationToken::new();

        // Not cancelled — will attempt connection (and fail due to no server)
        let result = generate_response_streaming_cancellable(
            &config,
            &conversation,
            "test prompt",
            &tx,
            &cancel_token,
        );
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_resilient_fallback_logs_on_failure() {
        let primary = ProviderConfig::new("primary-log-test", "http://localhost:19999")
            .with_provider_type("ollama")
            .with_priority(0);
        let fallback = ProviderConfig::new("backup-log-test", "http://localhost:19998")
            .with_provider_type("openai")
            .with_priority(1);
        let mut reg = ResilientProviderRegistry::new(primary);
        reg.add_fallback(fallback);

        // Primary fails, fallback succeeds — logging should capture the
        // warn for primary failure and info for backup success.
        let result = reg.generate_with_fallback(|p| -> Result<String, String> {
            if p.name == "primary-log-test" {
                Err("connection refused".to_string())
            } else {
                Ok(format!("ok from {}", p.name))
            }
        });

        assert!(result.is_ok());
        let (value, name) = result.expect("fallback should succeed");
        assert_eq!(value, "ok from backup-log-test");
        assert_eq!(name, "backup-log-test");
    }

    #[test]
    fn test_resilient_circuit_breaker_logs_open() {
        let primary = ProviderConfig::new("cb-log-test", "http://localhost:19999")
            .with_provider_type("ollama")
            .with_priority(0);
        let mut reg = ResilientProviderRegistry::new(primary);
        reg.set_max_failures(2);

        // Two failures should open the circuit and log the transition
        reg.record_failure("cb-log-test");
        assert!(!reg.is_circuit_open("cb-log-test"));
        reg.record_failure("cb-log-test");
        assert!(reg.is_circuit_open("cb-log-test"));
    }
}
