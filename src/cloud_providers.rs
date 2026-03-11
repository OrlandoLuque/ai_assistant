//! Cloud-based LLM provider integrations (OpenAI, Anthropic, and others).
//!
//! Provides [`CloudProviderConfig`] and [`CloudProviderType`] for direct API
//! integration with cloud LLM services. API keys are resolved from
//! `AiConfig.api_key` first, then from environment variables.
//!
//! Gated behind the `cloud-providers` feature flag.

use crate::config::{AiConfig, AiProvider};
use crate::messages::ChatMessage;
use anyhow::{Context, Result};

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
    config.get_api_key().ok_or_else(|| {
        let env_var = match &config.provider {
            AiProvider::OpenAI => "OPENAI_API_KEY",
            AiProvider::Anthropic => "ANTHROPIC_API_KEY",
            AiProvider::Gemini => "GOOGLE_API_KEY",
            AiProvider::Bedrock { .. } => "AWS_ACCESS_KEY_ID",
            AiProvider::Groq => "GROQ_API_KEY",
            AiProvider::Together => "TOGETHER_API_KEY",
            AiProvider::Fireworks => "FIREWORKS_API_KEY",
            AiProvider::DeepSeek => "DEEPSEEK_API_KEY",
            AiProvider::Mistral => "MISTRAL_API_KEY",
            AiProvider::Perplexity => "PERPLEXITY_API_KEY",
            AiProvider::OpenRouter => "OPENROUTER_API_KEY",
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

    let json: serde_json::Value = response
        .into_json()
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

    let json: serde_json::Value = response
        .into_json()
        .context("Failed to parse models response")?;

    let models = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.get("id")
                        .and_then(|id| id.as_str())
                        .map(|s| s.to_string())
                })
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

    let json: serde_json::Value = response
        .into_json()
        .context("Failed to parse Anthropic response")?;

    // Anthropic response format: {"content": [{"type": "text", "text": "..."}]}
    json.get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| {
            arr.iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
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
// Google Gemini Cloud
// ============================================================================

/// Generate a non-streaming response from Google Gemini's API.
///
/// Uses the generateContent endpoint with API key authentication.
pub fn generate_gemini_cloud(
    config: &AiConfig,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Result<String> {
    let api_key = resolve_api_key(config)?;
    let model = if config.selected_model.is_empty() {
        "gemini-2.0-flash"
    } else {
        &config.selected_model
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model
    );

    // Build Gemini contents array
    let mut contents = Vec::new();

    // System instruction (Gemini supports it as a separate field)
    if !system_prompt.is_empty() {
        // Gemini uses systemInstruction field
    }

    for msg in messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            "system" => "user", // Gemini doesn't have system role in contents
            other => other,
        };
        contents.push(serde_json::json!({
            "role": role,
            "parts": [{ "text": msg.content }]
        }));
    }

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": 4096
        }
    });

    if !system_prompt.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": [{ "text": system_prompt }]
        });
    }

    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .set("x-goog-api-key", &api_key)
        .timeout(std::time::Duration::from_secs(120))
        .send_json(&body)
        .context("Gemini API request failed")?;

    let json: serde_json::Value = response
        .into_json()
        .context("Failed to parse Gemini response")?;

    // Gemini response: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
    json.get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(|parts| parts.as_array())
        .and_then(|parts| parts.first())
        .and_then(|part| part.get("text"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Unexpected Gemini response format"))
}

/// Fetch available models from Gemini (returns known model list).
pub fn fetch_gemini_cloud_models() -> Vec<String> {
    vec![
        "gemini-2.0-flash".to_string(),
        "gemini-2.0-flash-lite".to_string(),
        "gemini-1.5-pro".to_string(),
        "gemini-1.5-flash".to_string(),
        "gemini-1.5-flash-8b".to_string(),
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
        AiProvider::Gemini => generate_gemini_cloud(config, messages, system_prompt),
        AiProvider::Groq
        | AiProvider::Together
        | AiProvider::Fireworks
        | AiProvider::DeepSeek
        | AiProvider::Mistral
        | AiProvider::Perplexity
        | AiProvider::OpenRouter => generate_openai_cloud(config, messages, system_prompt),
        AiProvider::Bedrock { .. } => {
            anyhow::bail!("AWS Bedrock requires the `aws-bedrock` feature flag.")
        }
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
        AiProvider::Gemini => Ok(fetch_gemini_cloud_models()),
        AiProvider::Bedrock { .. } => Ok(vec![
            "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
            "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            "amazon.titan-text-express-v1".to_string(),
            "meta.llama3-1-70b-instruct-v1:0".to_string(),
        ]),
        AiProvider::Groq => Ok(vec![
            "llama-3.3-70b-versatile".to_string(),
            "llama-3.1-8b-instant".to_string(),
            "mixtral-8x7b-32768".to_string(),
            "gemma2-9b-it".to_string(),
        ]),
        AiProvider::Together => Ok(vec![
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string(),
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            "Qwen/Qwen2.5-72B-Instruct-Turbo".to_string(),
        ]),
        AiProvider::Fireworks => Ok(vec![
            "accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
            "accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
            "accounts/fireworks/models/mixtral-8x7b-instruct".to_string(),
        ]),
        AiProvider::DeepSeek => Ok(vec![
            "deepseek-chat".to_string(),
            "deepseek-coder".to_string(),
            "deepseek-reasoner".to_string(),
        ]),
        AiProvider::Mistral => Ok(vec![
            "mistral-large-latest".to_string(),
            "mistral-medium-latest".to_string(),
            "mistral-small-latest".to_string(),
            "codestral-latest".to_string(),
            "open-mistral-nemo".to_string(),
        ]),
        AiProvider::Perplexity => Ok(vec![
            "sonar-pro".to_string(),
            "sonar".to_string(),
            "sonar-reasoning-pro".to_string(),
            "sonar-reasoning".to_string(),
        ]),
        AiProvider::OpenRouter => Ok(vec![
            "openai/gpt-4o".to_string(),
            "anthropic/claude-3.5-sonnet".to_string(),
            "meta-llama/llama-3.1-70b-instruct".to_string(),
            "google/gemini-2.0-flash-001".to_string(),
        ]),
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not a cloud provider"));
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

    #[test]
    fn test_gemini_is_cloud() {
        assert!(AiProvider::Gemini.is_cloud());
    }

    #[test]
    fn test_gemini_display_name() {
        assert_eq!(AiProvider::Gemini.display_name(), "Google Gemini");
    }

    #[test]
    fn test_gemini_base_url() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::Gemini;
        assert_eq!(
            config.get_base_url(),
            "https://generativelanguage.googleapis.com"
        );
    }

    #[test]
    fn test_fetch_gemini_models() {
        let models = fetch_gemini_cloud_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("gemini")));
        assert!(models.iter().any(|m| m.contains("flash")));
    }

    #[test]
    fn test_fetch_cloud_models_gemini() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::Gemini;
        let models = fetch_cloud_models(&config).unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("gemini")));
    }

    #[test]
    fn test_bedrock_base_url() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::Bedrock {
            region: "us-east-1".to_string(),
        };
        assert_eq!(
            config.get_base_url(),
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn test_bedrock_is_cloud() {
        assert!(AiProvider::Bedrock {
            region: "us-west-2".to_string()
        }
        .is_cloud());
    }

    #[test]
    fn test_bedrock_display_name() {
        assert_eq!(
            AiProvider::Bedrock {
                region: "eu-west-1".to_string()
            }
            .display_name(),
            "AWS Bedrock"
        );
    }

    #[test]
    fn test_fetch_cloud_models_bedrock() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::Bedrock {
            region: "us-east-1".to_string(),
        };
        let models = fetch_cloud_models(&config).unwrap();
        assert!(!models.is_empty());
        assert!(models
            .iter()
            .any(|m| m.contains("claude") || m.contains("anthropic")));
    }

    #[test]
    fn test_generate_cloud_response_bedrock_needs_feature() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::Bedrock {
            region: "us-east-1".to_string(),
        };
        config.api_key = "test-key".to_string();
        let result = generate_cloud_response(&config, &[], "");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("aws-bedrock"));
    }

    #[test]
    fn test_resolve_api_key_gemini() {
        let config = AiConfig {
            provider: AiProvider::Gemini,
            api_key: "gemini-test-key".to_string(),
            ..Default::default()
        };
        let key = resolve_api_key(&config).unwrap();
        assert_eq!(key, "gemini-test-key");
    }

    #[test]
    fn test_resolve_api_key_bedrock() {
        let config = AiConfig {
            provider: AiProvider::Bedrock {
                region: "us-east-1".to_string(),
            },
            api_key: "aws-key".to_string(),
            ..Default::default()
        };
        let key = resolve_api_key(&config).unwrap();
        assert_eq!(key, "aws-key");
    }
}
