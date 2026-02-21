//! Google Gemini provider implementing ProviderPlugin trait.

use std::collections::HashMap;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::{AiConfig, AiProvider};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;
use crate::tools::{ProviderCapabilities, ProviderPlugin, ToolCall, ToolDefinition};

/// Google Gemini API provider.
///
/// Supports all Gemini models (2.0 Flash, 1.5 Pro, etc.) through the
/// `generativelanguage.googleapis.com` REST API with API key authentication.
pub struct GeminiProvider {
    api_key: String,
    base_url: String,
    /// Request timeout (default 120s).
    pub timeout: Duration,
    available: AtomicBool,
}

impl std::fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("available", &self.available.load(Ordering::SeqCst))
            .finish()
    }
}

impl GeminiProvider {
    /// Create a new GeminiProvider with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            timeout: Duration::from_secs(120),
            available: AtomicBool::new(false),
        }
    }

    /// Create a GeminiProvider from environment variables.
    ///
    /// Checks `GOOGLE_API_KEY` first, then `GEMINI_API_KEY`.
    /// Returns an error if neither is set.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .context(
                "Neither GOOGLE_API_KEY nor GEMINI_API_KEY environment variable is set",
            )?;
        Ok(Self::new(&key))
    }

    /// Set the request timeout (builder pattern).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if the Gemini API is reachable.
    pub fn check_health(&self) -> bool {
        let url = format!(
            "{}/v1beta/models?key={}&pageSize=1",
            self.base_url, self.api_key
        );
        match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available.store(is_ok, Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available.store(false, Ordering::SeqCst);
                false
            }
        }
    }

    // ---- internal helpers ----

    /// Normalise the model name so it always has the `models/` prefix that the
    /// Gemini REST API expects.
    fn full_model_name(model: &str) -> String {
        if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{}", model)
        }
    }

    /// Build the Gemini `contents` array from chat messages.
    ///
    /// Role mapping:
    /// - "user"      -> "user"
    /// - "assistant"  -> "model"
    /// - "system"     -> skipped (handled via `systemInstruction`)
    fn build_contents(messages: &[ChatMessage]) -> Vec<Value> {
        messages
            .iter()
            .filter(|m| !m.is_system())
            .map(|m| {
                let role = if m.is_user() { "user" } else { "model" };
                serde_json::json!({
                    "role": role,
                    "parts": [{"text": &m.content}]
                })
            })
            .collect()
    }

    /// Build the full request body for `generateContent`.
    fn build_request_body(
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: Option<&[ToolDefinition]>,
    ) -> Value {
        let contents = Self::build_contents(messages);

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": 4096
            }
        });

        // System instruction
        if !system_prompt.is_empty() {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": system_prompt}]
            });
        }

        // Tool declarations
        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                let declarations: Vec<Value> = tool_defs
                    .iter()
                    .map(|td| {
                        let params = td.to_openai_function()["parameters"].clone();
                        serde_json::json!({
                            "name": td.name,
                            "description": td.description,
                            "parameters": params
                        })
                    })
                    .collect();
                body["tools"] = serde_json::json!([{
                    "functionDeclarations": declarations
                }]);
            }
        }

        body
    }

    /// Parse a generateContent response JSON, returning the text and any tool
    /// calls found in the first candidate.
    fn parse_generate_response(body: &Value) -> (String, Vec<ToolCall>) {
        let mut text = String::new();
        let mut tool_calls = Vec::new();

        let parts = body
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(|p| p.as_array());

        if let Some(parts) = parts {
            for part in parts {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args: HashMap<String, Value> = fc
                        .get("args")
                        .and_then(|a| serde_json::from_value(a.clone()).ok())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall {
                        name,
                        arguments: args,
                        id: String::new(),
                    });
                }
            }
        }

        (text, tool_calls)
    }

    /// Parse a single SSE data line, returning the text chunk (if any).
    fn parse_sse_chunk(data: &str) -> Option<String> {
        let json: Value = serde_json::from_str(data).ok()?;
        let parts = json
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(|p| p.as_array())?;

        let mut chunk = String::new();
        for part in parts {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                chunk.push_str(t);
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }

    /// Parse a model list response, returning `ModelInfo` entries.
    fn parse_model_list(body: &Value) -> Vec<ModelInfo> {
        body.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let raw_name = m.get("name")?.as_str()?;
                        // Strip the "models/" prefix for storage
                        let name = raw_name
                            .strip_prefix("models/")
                            .unwrap_or(raw_name)
                            .to_string();
                        let display = m
                            .get("displayName")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string();
                        let mut info = ModelInfo::new(name, AiProvider::Gemini);
                        if !display.is_empty() {
                            info = info.with_size(display); // repurpose size field for display name
                        }
                        Some(info)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Parse an embedContent response, returning the embedding vector.
    fn parse_embedding_response(body: &Value) -> Option<Vec<f32>> {
        body.get("embedding")
            .and_then(|e| e.get("values"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
    }

    /// Return a hardcoded fallback model list when the API is unreachable.
    fn fallback_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("gemini-2.0-flash", AiProvider::Gemini),
            ModelInfo::new("gemini-2.0-flash-lite", AiProvider::Gemini),
            ModelInfo::new("gemini-1.5-pro", AiProvider::Gemini),
            ModelInfo::new("gemini-1.5-flash", AiProvider::Gemini),
            ModelInfo::new("gemini-1.5-flash-8b", AiProvider::Gemini),
        ]
    }
}

impl ProviderPlugin for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            vision: true,
            embeddings: true,
            json_mode: true,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        if self.available.load(Ordering::SeqCst) {
            return true;
        }
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!(
            "{}/v1beta/models?key={}",
            self.base_url, self.api_key
        );
        match ureq::get(&url).timeout(self.timeout).call() {
            Ok(resp) => {
                let body: Value = resp
                    .into_json()
                    .context("Failed to parse Gemini models response")?;
                let models = Self::parse_model_list(&body);
                if models.is_empty() {
                    Ok(Self::fallback_models())
                } else {
                    Ok(models)
                }
            }
            Err(_) => Ok(Self::fallback_models()),
        }
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let model = Self::full_model_name(&config.selected_model);
        let url = format!(
            "{}/v1beta/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );
        let body = Self::build_request_body(config, messages, system_prompt, None);

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&body)
            .context("Failed to send request to Gemini API")?;

        let resp_body: Value = resp
            .into_json()
            .context("Failed to parse Gemini response")?;

        let (text, _) = Self::parse_generate_response(&resp_body);
        Ok(text)
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let model = Self::full_model_name(&config.selected_model);
        let url = format!(
            "{}/v1beta/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        );
        let body = Self::build_request_body(config, messages, system_prompt, None);

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&body)
            .context("Failed to send streaming request to Gemini API")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            // SSE lines have the format: "data: {json}"
            let data = if let Some(stripped) = line.strip_prefix("data: ") {
                stripped
            } else {
                continue;
            };

            if let Some(chunk_text) = Self::parse_sse_chunk(data) {
                full_response.push_str(&chunk_text);
                let _ = tx.send(AiResponse::Chunk(chunk_text));
            }
        }

        let _ = tx.send(AiResponse::Complete(full_response));
        Ok(())
    }

    fn generate_with_tools(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<(String, Vec<ToolCall>)> {
        let model = Self::full_model_name(&config.selected_model);
        let url = format!(
            "{}/v1beta/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );
        let body = Self::build_request_body(config, messages, system_prompt, Some(tools));

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&body)
            .context("Failed to send tool request to Gemini API")?;

        let resp_body: Value = resp
            .into_json()
            .context("Failed to parse Gemini tool response")?;

        Ok(Self::parse_generate_response(&resp_body))
    }

    fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let url = format!(
                "{}/v1beta/models/text-embedding-004:embedContent?key={}",
                self.base_url, self.api_key
            );
            let payload = serde_json::json!({
                "content": {
                    "parts": [{"text": text}]
                }
            });

            let resp = ureq::post(&url)
                .timeout(self.timeout)
                .send_json(&payload)
                .context("Failed to send embedding request to Gemini API")?;

            let body: Value = resp
                .into_json()
                .context("Failed to parse Gemini embedding response")?;

            match Self::parse_embedding_response(&body) {
                Some(vec) => embeddings.push(vec),
                None => anyhow::bail!("Failed to extract embedding from Gemini response"),
            }
        }

        Ok(embeddings)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_provider_new() {
        let provider = GeminiProvider::new("test-key-123");
        assert_eq!(provider.name(), "gemini");
        assert_eq!(provider.api_key, "test-key-123");
        assert_eq!(
            provider.base_url,
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(provider.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_gemini_provider_capabilities() {
        let provider = GeminiProvider::new("key");
        let caps = provider.capabilities();
        assert!(caps.streaming);
        assert!(caps.tool_calling);
        assert!(caps.vision);
        assert!(caps.embeddings);
        assert!(caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_gemini_provider_from_env_missing() {
        // Ensure neither env var is set
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        let result = GeminiProvider::from_env();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("GOOGLE_API_KEY"));
    }

    #[test]
    fn test_gemini_provider_with_timeout() {
        let provider = GeminiProvider::new("key").with_timeout(Duration::from_secs(30));
        assert_eq!(provider.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_gemini_model_list_parsing() {
        let json: Value = serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "description": "Fast model"
                },
                {
                    "name": "models/gemini-1.5-pro",
                    "displayName": "Gemini 1.5 Pro",
                    "description": "Advanced model"
                }
            ]
        });

        let models = GeminiProvider::parse_model_list(&json);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "gemini-2.0-flash");
        assert_eq!(models[0].provider, AiProvider::Gemini);
        // displayName stored in size field
        assert_eq!(models[0].size.as_deref(), Some("Gemini 2.0 Flash"));
        assert_eq!(models[1].name, "gemini-1.5-pro");
    }

    #[test]
    fn test_gemini_response_parsing() {
        let json: Value = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Hello, "},
                        {"text": "world!"}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });

        let (text, tool_calls) = GeminiProvider::parse_generate_response(&json);
        assert_eq!(text, "Hello, world!");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_gemini_streaming_response_parsing() {
        // Simulate an SSE data payload
        let data = r#"{"candidates":[{"content":{"parts":[{"text":"streaming chunk"}],"role":"model"}}]}"#;
        let result = GeminiProvider::parse_sse_chunk(data);
        assert_eq!(result, Some("streaming chunk".to_string()));

        // Empty text yields None
        let empty = r#"{"candidates":[{"content":{"parts":[{"text":""}],"role":"model"}}]}"#;
        assert!(GeminiProvider::parse_sse_chunk(empty).is_none());

        // Invalid JSON yields None
        assert!(GeminiProvider::parse_sse_chunk("not json").is_none());
    }

    #[test]
    fn test_gemini_embedding_response_parsing() {
        let json: Value = serde_json::json!({
            "embedding": {
                "values": [0.1, 0.2, 0.3, -0.5, 1.0]
            }
        });

        let embedding = GeminiProvider::parse_embedding_response(&json).unwrap();
        assert_eq!(embedding.len(), 5);
        assert!((embedding[0] - 0.1).abs() < 1e-6);
        assert!((embedding[3] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_gemini_role_mapping() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("Bye"),
        ];

        let contents = GeminiProvider::build_contents(&messages);
        // System messages are filtered out
        assert_eq!(contents.len(), 3);

        // First non-system message: user
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello");

        // "assistant" is mapped to "model"
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["text"], "Hi there");

        // Another user message
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"][0]["text"], "Bye");
    }

    #[test]
    fn test_gemini_tool_call_parsing() {
        let json: Value = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me check the weather."},
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {
                                    "location": "London",
                                    "units": "celsius"
                                }
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });

        let (text, tool_calls) = GeminiProvider::parse_generate_response(&json);
        assert_eq!(text, "Let me check the weather.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(
            tool_calls[0].arguments.get("location").and_then(|v| v.as_str()),
            Some("London")
        );
        assert_eq!(
            tool_calls[0].arguments.get("units").and_then(|v| v.as_str()),
            Some("celsius")
        );
    }

    #[test]
    fn test_gemini_full_model_name() {
        assert_eq!(
            GeminiProvider::full_model_name("gemini-2.0-flash"),
            "models/gemini-2.0-flash"
        );
        assert_eq!(
            GeminiProvider::full_model_name("models/gemini-2.0-flash"),
            "models/gemini-2.0-flash"
        );
    }

    #[test]
    fn test_gemini_build_request_body() {
        let config = AiConfig {
            selected_model: "gemini-2.0-flash".to_string(),
            temperature: 0.5,
            ..AiConfig::default()
        };
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
        ];

        let body = GeminiProvider::build_request_body(&config, &messages, "Be helpful", None);

        // System instruction present
        assert!(body.get("systemInstruction").is_some());
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be helpful");

        // Contents: 2 messages
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");

        // Generation config
        assert_eq!(body["generationConfig"]["temperature"], 0.5);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 4096);

        // No tools
        assert!(body.get("tools").is_none());
    }
}
