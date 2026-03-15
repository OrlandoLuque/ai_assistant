//! Anthropic API adapter
//!
//! Adapter for Anthropic's Claude API.

use std::time::Duration;

/// Anthropic model
#[derive(Debug, Clone)]
pub struct AnthropicModel {
    pub id: String,
    pub name: String,
    pub context_length: usize,
    pub supports_vision: bool,
}

impl AnthropicModel {
    /// Claude 3 Opus
    pub fn claude3_opus() -> Self {
        Self {
            id: "claude-3-opus-20240229".to_string(),
            name: "Claude 3 Opus".to_string(),
            context_length: 200000,
            supports_vision: true,
        }
    }

    /// Claude 3 Sonnet
    pub fn claude3_sonnet() -> Self {
        Self {
            id: "claude-3-sonnet-20240229".to_string(),
            name: "Claude 3 Sonnet".to_string(),
            context_length: 200000,
            supports_vision: true,
        }
    }

    /// Claude 3 Haiku
    pub fn claude3_haiku() -> Self {
        Self {
            id: "claude-3-haiku-20240307".to_string(),
            name: "Claude 3 Haiku".to_string(),
            context_length: 200000,
            supports_vision: true,
        }
    }

    /// Claude 2.1
    pub fn claude2() -> Self {
        Self {
            id: "claude-2.1".to_string(),
            name: "Claude 2.1".to_string(),
            context_length: 200000,
            supports_vision: false,
        }
    }
}

/// Anthropic message
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl AnthropicMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::Value::String(content.to_string()),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: serde_json::Value::String(content.to_string()),
        }
    }

    pub fn user_with_image(text: &str, image_base64: &str, media_type: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::json!([
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": text
                }
            ]),
        }
    }
}

/// Anthropic request
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    pub stream: bool,
}

impl AnthropicRequest {
    pub fn new(model: &str, messages: Vec<AnthropicMessage>) -> Self {
        Self {
            model: model.to_string(),
            messages,
            max_tokens: 4096,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
        }
    }

    pub fn with_system(mut self, system: &str) -> Self {
        self.system = Some(system.to_string());
        self
    }

    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

/// Anthropic response
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl AnthropicResponse {
    /// Get the text content from the response
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Anthropic error
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Anthropic client configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
    pub api_version: String,
}

impl AnthropicConfig {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            timeout: Duration::from_secs(120),
            api_version: "2023-06-01".to_string(),
        }
    }
}

/// Anthropic client
pub struct AnthropicClient {
    config: AnthropicConfig,
}

impl AnthropicClient {
    pub fn new(config: AnthropicConfig) -> Self {
        Self { config }
    }

    /// Create a message
    pub fn message(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse, AnthropicAdapterError> {
        let url = format!("{}/v1/messages", self.config.base_url);

        let req = ureq::post(&url)
            .timeout(self.config.timeout)
            .set("Content-Type", "application/json")
            .set("x-api-key", &self.config.api_key)
            .set("anthropic-version", &self.config.api_version);

        let body = serde_json::to_string(&request)
            .map_err(|e| AnthropicAdapterError::Serialization(e.to_string()))?;

        let response = req.send_string(&body);

        match response {
            Ok(resp) => {
                let text = resp
                    .into_string()
                    .map_err(|e| AnthropicAdapterError::Network(e.to_string()))?;

                serde_json::from_str(&text)
                    .map_err(|e| AnthropicAdapterError::Deserialization(format!("{}: {}", e, text)))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                if let Ok(error) = serde_json::from_str::<AnthropicError>(&text) {
                    Err(AnthropicAdapterError::Api {
                        code,
                        message: error.error.message,
                        error_type: error.error.error_type,
                    })
                } else {
                    Err(AnthropicAdapterError::Api {
                        code,
                        message: text,
                        error_type: "unknown".to_string(),
                    })
                }
            }
            Err(e) => Err(AnthropicAdapterError::Network(e.to_string())),
        }
    }

    /// Create a streaming message
    pub fn message_stream<F>(
        &self,
        request: AnthropicRequest,
        mut on_chunk: F,
    ) -> Result<String, AnthropicAdapterError>
    where
        F: FnMut(&str),
    {
        let mut streaming_request = request;
        streaming_request.stream = true;

        let url = format!("{}/v1/messages", self.config.base_url);

        let req = ureq::post(&url)
            .timeout(self.config.timeout)
            .set("Content-Type", "application/json")
            .set("x-api-key", &self.config.api_key)
            .set("anthropic-version", &self.config.api_version);

        let body = serde_json::to_string(&streaming_request)
            .map_err(|e| AnthropicAdapterError::Serialization(e.to_string()))?;

        let response = req
            .send_string(&body)
            .map_err(|e| AnthropicAdapterError::Network(e.to_string()))?;

        let reader = response.into_reader();
        let buf_reader = std::io::BufReader::new(reader);

        let mut full_response = String::new();

        use std::io::BufRead;
        for line in buf_reader.lines() {
            let line = line.map_err(|e| AnthropicAdapterError::Network(e.to_string()))?;

            if line.starts_with("data: ") {
                let data = &line[6..];

                if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                    if event.event_type == "content_block_delta" {
                        if let Some(delta) = event.delta {
                            if let Some(text) = delta.text {
                                on_chunk(&text);
                                full_response.push_str(&text);
                            }
                        }
                    }
                }
            }
        }

        Ok(full_response)
    }
}

#[derive(Debug, serde::Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<DeltaContent>,
}

#[derive(Debug, serde::Deserialize)]
struct DeltaContent {
    text: Option<String>,
}

/// Anthropic adapter error
#[derive(Debug)]
#[non_exhaustive]
pub enum AnthropicAdapterError {
    Network(String),
    Serialization(String),
    Deserialization(String),
    Api {
        code: u16,
        message: String,
        error_type: String,
    },
    RateLimit {
        retry_after: Option<Duration>,
    },
}

impl std::fmt::Display for AnthropicAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "Network error: {}", e),
            Self::Serialization(e) => write!(f, "Serialization error: {}", e),
            Self::Deserialization(e) => write!(f, "Deserialization error: {}", e),
            Self::Api { code, message, .. } => write!(f, "API error {}: {}", code, message),
            Self::RateLimit { retry_after } => {
                if let Some(d) = retry_after {
                    write!(f, "Rate limited, retry after {:?}", d)
                } else {
                    write!(f, "Rate limited")
                }
            }
        }
    }
}

impl std::error::Error for AnthropicAdapterError {}

/// Simple chat helper
pub fn simple_chat(
    api_key: &str,
    model: &str,
    system: &str,
    user_message: &str,
) -> Result<String, AnthropicAdapterError> {
    let client = AnthropicClient::new(AnthropicConfig::new(api_key));

    let request = AnthropicRequest::new(model, vec![AnthropicMessage::user(user_message)])
        .with_system(system);

    let response = client.message(request)?;
    Ok(response.text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = AnthropicMessage::user("Hello");
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn test_request_builder() {
        let messages = vec![AnthropicMessage::user("Hello")];

        let request = AnthropicRequest::new("claude-3-sonnet-20240229", messages)
            .with_system("You are helpful")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        assert_eq!(request.system, Some("You are helpful".to_string()));
        assert_eq!(request.max_tokens, 1000);
    }

    #[test]
    fn test_models() {
        let opus = AnthropicModel::claude3_opus();
        assert!(opus.supports_vision);
        assert!(opus.context_length > 100000);
    }

    #[test]
    fn test_model_presets() {
        let opus = AnthropicModel::claude3_opus();
        assert_eq!(opus.id, "claude-3-opus-20240229");
        assert_eq!(opus.name, "Claude 3 Opus");
        assert_eq!(opus.context_length, 200000);
        assert!(opus.supports_vision);

        let sonnet = AnthropicModel::claude3_sonnet();
        assert_eq!(sonnet.id, "claude-3-sonnet-20240229");
        assert_eq!(sonnet.name, "Claude 3 Sonnet");
        assert_eq!(sonnet.context_length, 200000);
        assert!(sonnet.supports_vision);

        let haiku = AnthropicModel::claude3_haiku();
        assert_eq!(haiku.id, "claude-3-haiku-20240307");
        assert_eq!(haiku.name, "Claude 3 Haiku");
        assert_eq!(haiku.context_length, 200000);
        assert!(haiku.supports_vision);

        let claude2 = AnthropicModel::claude2();
        assert_eq!(claude2.id, "claude-2.1");
        assert_eq!(claude2.name, "Claude 2.1");
        assert_eq!(claude2.context_length, 200000);
        assert!(!claude2.supports_vision);
    }

    #[test]
    fn test_message_types() {
        let usr = AnthropicMessage::user("Hello");
        assert_eq!(usr.role, "user");
        assert_eq!(usr.content, serde_json::Value::String("Hello".to_string()));

        let asst = AnthropicMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
        assert_eq!(asst.content, serde_json::Value::String("Hi there".to_string()));
    }

    #[test]
    fn test_user_with_image() {
        let msg = AnthropicMessage::user_with_image(
            "Describe this image",
            "aW1hZ2VkYXRh",
            "image/png",
        );
        assert_eq!(msg.role, "user");

        let content = msg.content.as_array().expect("content should be an array");
        assert_eq!(content.len(), 2);

        // First element is the image block
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["type"], "base64");
        assert_eq!(content[0]["source"]["media_type"], "image/png");
        assert_eq!(content[0]["source"]["data"], "aW1hZ2VkYXRh");

        // Second element is the text block
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "Describe this image");
    }

    #[test]
    fn test_request_options() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let req = AnthropicRequest::new("claude-3-opus-20240229", messages)
            .with_system("You are a helpful assistant")
            .with_temperature(0.5)
            .with_max_tokens(2048)
            .streaming();

        assert_eq!(req.model, "claude-3-opus-20240229");
        assert_eq!(req.system, Some("You are a helpful assistant".to_string()));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.max_tokens, 2048);
        assert!(req.stream);

        // Verify defaults for unset optional fields
        assert!(req.top_p.is_none());
        assert!(req.top_k.is_none());
        assert!(req.stop_sequences.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = AnthropicConfig::new("sk-ant-test-key");
        assert_eq!(config.api_key, "sk-ant-test-key");
        assert_eq!(config.base_url, "https://api.anthropic.com");
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.api_version, "2023-06-01");
    }

    #[test]
    fn test_request_serialization() {
        let messages = vec![
            AnthropicMessage::user("What is Rust?"),
        ];
        let req = AnthropicRequest::new("claude-3-haiku-20240307", messages)
            .with_system("Be concise")
            .with_max_tokens(512);

        let json_str = serde_json::to_string(&req).expect("should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("should parse as JSON");

        assert_eq!(parsed["model"], "claude-3-haiku-20240307");
        assert_eq!(parsed["max_tokens"], 512);
        assert_eq!(parsed["system"], "Be concise");
        assert_eq!(parsed["stream"], false);

        // messages array
        let msgs = parsed["messages"].as_array().expect("messages should be array");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "What is Rust?");

        // Optional fields should be absent (skip_serializing_if)
        assert!(parsed.get("temperature").is_none());
        assert!(parsed.get("top_p").is_none());
        assert!(parsed.get("top_k").is_none());
        assert!(parsed.get("stop_sequences").is_none());
    }

    #[test]
    fn test_error_display_formatting() {
        let network_err = AnthropicAdapterError::Network("connection refused".to_string());
        assert_eq!(
            format!("{}", network_err),
            "Network error: connection refused"
        );

        let serial_err = AnthropicAdapterError::Serialization("invalid json".to_string());
        assert_eq!(
            format!("{}", serial_err),
            "Serialization error: invalid json"
        );

        let deser_err = AnthropicAdapterError::Deserialization("unexpected token".to_string());
        assert_eq!(
            format!("{}", deser_err),
            "Deserialization error: unexpected token"
        );

        let api_err = AnthropicAdapterError::Api {
            code: 429,
            message: "Too many requests".to_string(),
            error_type: "rate_limit_error".to_string(),
        };
        assert_eq!(
            format!("{}", api_err),
            "API error 429: Too many requests"
        );

        let rate_err_with = AnthropicAdapterError::RateLimit {
            retry_after: Some(Duration::from_secs(30)),
        };
        let display = format!("{}", rate_err_with);
        assert!(display.contains("Rate limited"));
        assert!(display.contains("retry after"));

        let rate_err_without = AnthropicAdapterError::RateLimit { retry_after: None };
        assert_eq!(format!("{}", rate_err_without), "Rate limited");
    }
}
