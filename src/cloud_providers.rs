// cloud_providers.rs — Native cloud provider support for OpenAI and Anthropic APIs.
//
// Provides direct API integration with cloud LLM providers, requiring API keys.
// Keys are resolved from AiConfig.api_key first, then environment variables
// (OPENAI_API_KEY, ANTHROPIC_API_KEY).
//
// OpenAI uses standard /v1/chat/completions endpoint.
// Anthropic uses /v1/messages with its own format (x-api-key header, different
// message structure, system prompt as top-level parameter).

use anyhow::{Context, Result};
use crate::config::{AiConfig, AiProvider};
use crate::messages::ChatMessage;

// ============================================================================
// API Key Resolution
// ============================================================================

/// Resolve the API key for the current provider.
///
/// Checks `config.api_key` first, then the appropriate environment variable:
/// - `OPENAI_API_KEY` for OpenAI
/// - `ANTHROPIC_API_KEY` for Anthropic
///
/// Returns an error if no key is found for a cloud provider.
pub fn resolve_api_key(config: &AiConfig) -> Result<String> {
    config
        .get_api_key()
        .ok_or_else(|| {
            let env_var = match &config.provider {
                AiProvider::OpenAI => "OPENAI_API_KEY",
                AiProvider::Anthropic => "ANTHROPIC_API_KEY",
                _ => "API_KEY",
            };
            anyhow::anyhow!(
                "No API key found for {}. Set config.api_key or {} environment variable.",
                config.provider.display_name(),
                env_var
            )
        })
}

// ============================================================================
// OpenAI Cloud
// ============================================================================

/// Generate a non-streaming response from OpenAI's API.
///
/// Uses the standard /v1/chat/completions endpoint with Bearer token auth.
pub fn generate_openai_cloud(
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let api_key = resolve_api_key(config)?;
    let base_url = config.get_base_url();
    let url = format!("{}/v1/chat/completions", base_url);

    let mut api_messages = Vec::new();
    if !system_prompt.is_empty() {
        api_messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt,
        }));
    }
    for msg in messages {
        api_messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content,
        }));
    }

    let body = serde_json::json!({
        "model": config.selected_model,
        "messages": api_messages,
        "temperature": config.temperature,
    });

    let response = ureq::post(&url)
        .set("Authorization", &format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&body)
        .context("OpenAI API request failed")?;

    let json: serde_json::Value = response.into_json()
        .context("Failed to parse OpenAI response")?;

    json.get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Unexpected OpenAI response format"))
}

/// Fetch available models from OpenAI's API.
pub fn fetch_openai_cloud_models(config: &AiConfig) -> Result<Vec<String>> {
    let api_key = resolve_api_key(config)?;
    let base_url = config.get_base_url();
    let url = format!("{}/v1/models", base_url);

    let response = ureq::get(&url)
        .set("Authorization", &format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .context("Failed to fetch OpenAI models")?;

    let json: serde_json::Value = response.into_json()
        .context("Failed to parse models response")?;

    let models = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

// ============================================================================
// Anthropic Cloud
// ============================================================================

/// Generate a non-streaming response from Anthropic's Messages API.
///
/// Uses `/v1/messages` with `x-api-key` header and Anthropic-specific format.
/// The system prompt is a top-level parameter, not a message role.
pub fn generate_anthropic_cloud(
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let api_key = resolve_api_key(config)?;
    let base_url = config.get_base_url();
    let url = format!("{}/v1/messages", base_url);

    // Anthropic uses alternating user/assistant messages
    let api_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| {
            let role = match msg.role.as_str() {
                "system" => "user", // Anthropic doesn't support system role in messages
                other => other,
            };
            serde_json::json!({
                "role": role,
                "content": msg.content,
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "model": config.selected_model,
        "messages": api_messages,
        "max_tokens": 4096,
        "temperature": config.temperature,
    });

    // System prompt is a top-level field in Anthropic's API
    if !system_prompt.is_empty() {
        body["system"] = serde_json::json!(system_prompt);
    }

    let response = ureq::post(&url)
        .set("x-api-key", &api_key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&body)
        .context("Anthropic API request failed")?;

    let json: serde_json::Value = response.into_json()
        .context("Failed to parse Anthropic response")?;

    // Anthropic response format: {"content": [{"type": "text", "text": "..."}]}
    json.get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| {
            arr.iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .next()
        })
        .ok_or_else(|| anyhow::anyhow!("Unexpected Anthropic response format"))
}

/// Fetch available models from Anthropic (returns known model list).
///
/// Anthropic doesn't have a public model listing endpoint, so we return
/// the well-known model IDs.
pub fn fetch_anthropic_cloud_models() -> Vec<String> {
    vec![
        "claude-opus-4-6".to_string(),
        "claude-sonnet-4-5-20250929".to_string(),
        "claude-haiku-4-5-20251001".to_string(),
        "claude-3-5-sonnet-20241022".to_string(),
        "claude-3-5-haiku-20241022".to_string(),
        "claude-3-opus-20240229".to_string(),
        "claude-3-sonnet-20240229".to_string(),
        "claude-3-haiku-20240307".to_string(),
    ]
}

// ============================================================================
// Unified Cloud Interface
// ============================================================================

/// Generate a response from any cloud provider.
///
/// Dispatches to the correct cloud API based on the provider type.
/// Returns an error if the provider is not a cloud provider.
pub fn generate_cloud_response(
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    match config.provider {
        AiProvider::OpenAI => generate_openai_cloud(config, messages, system_prompt),
        AiProvider::Anthropic => generate_anthropic_cloud(config, messages, system_prompt),
        _ => anyhow::bail!(
            "{} is not a cloud provider. Use the standard provider functions.",
            config.provider.display_name()
        ),
    }
}

/// Fetch models from any cloud provider.
pub fn fetch_cloud_models(config: &AiConfig) -> Result<Vec<String>> {
    match config.provider {
        AiProvider::OpenAI => fetch_openai_cloud_models(config),
        AiProvider::Anthropic => Ok(fetch_anthropic_cloud_models()),
        _ => anyhow::bail!(
            "{} is not a cloud provider.",
            config.provider.display_name()
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_api_key_from_config() {
        let config = AiConfig {
            provider: AiProvider::OpenAI,
            api_key: "sk-test123".to_string(),
            ..Default::default()
        };
        let key = resolve_api_key(&config).unwrap();
        assert_eq!(key, "sk-test123");
    }

    #[test]
    fn test_resolve_api_key_missing() {
        let config = AiConfig {
            provider: AiProvider::OpenAI,
            api_key: String::new(),
            ..Default::default()
        };
        // This may or may not fail depending on env vars
        let result = resolve_api_key(&config);
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("OPENAI_API_KEY"));
        }
    }

    #[test]
    fn test_resolve_api_key_local_provider() {
        let config = AiConfig {
            provider: AiProvider::Ollama,
            api_key: String::new(),
            ..Default::default()
        };
        // Local providers return None from get_api_key(), so resolve fails
        let result = resolve_api_key(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_anthropic_models() {
        let models = fetch_anthropic_cloud_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("claude")));
    }

    #[test]
    fn test_generate_cloud_response_wrong_provider() {
        let config = AiConfig::default(); // Ollama
        let result = generate_cloud_response(&config, &[], "");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a cloud provider"));
    }

    #[test]
    fn test_fetch_cloud_models_wrong_provider() {
        let config = AiConfig::default(); // Ollama
        let result = fetch_cloud_models(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cloud_provider_base_urls() {
        let mut config = AiConfig::default();

        config.provider = AiProvider::OpenAI;
        assert_eq!(config.get_base_url(), "https://api.openai.com");

        config.provider = AiProvider::Anthropic;
        assert_eq!(config.get_base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn test_is_cloud() {
        assert!(AiProvider::OpenAI.is_cloud());
        assert!(AiProvider::Anthropic.is_cloud());
        assert!(!AiProvider::Ollama.is_cloud());
        assert!(!AiProvider::LMStudio.is_cloud());
    }

    #[test]
    fn test_cloud_display_names() {
        assert_eq!(AiProvider::OpenAI.display_name(), "OpenAI");
        assert_eq!(AiProvider::Anthropic.display_name(), "Anthropic");
    }
}
