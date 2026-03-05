//! OpenAI API adapter
//!
//! Adapter for OpenAI-compatible APIs (OpenAI, Azure OpenAI, local proxies).

use std::time::Duration;

/// OpenAI model
#[derive(Debug, Clone)]
pub struct OpenAIModel {
    pub id: String,
    pub name: String,
    pub context_length: usize,
    pub supports_vision: bool,
    pub supports_functions: bool,
}

impl OpenAIModel {
    /// GPT-4 Turbo
    pub fn gpt4_turbo() -> Self {
        Self {
            id: "gpt-4-turbo-preview".to_string(),
            name: "GPT-4 Turbo".to_string(),
            context_length: 128000,
            supports_vision: true,
            supports_functions: true,
        }
    }

    /// GPT-4
    pub fn gpt4() -> Self {
        Self {
            id: "gpt-4".to_string(),
            name: "GPT-4".to_string(),
            context_length: 8192,
            supports_vision: false,
            supports_functions: true,
        }
    }

    /// GPT-3.5 Turbo
    pub fn gpt35_turbo() -> Self {
        Self {
            id: "gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            context_length: 16385,
            supports_vision: false,
            supports_functions: true,
        }
    }
}

/// OpenAI request message
#[derive(Debug, Clone, serde::Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<serde_json::Value>,
}

impl OpenAIMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: serde_json::Value::String(content.to_string()),
            name: None,
            function_call: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::Value::String(content.to_string()),
            name: None,
            function_call: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: serde_json::Value::String(content.to_string()),
            name: None,
            function_call: None,
        }
    }

    pub fn user_with_image(text: &str, image_url: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::json!([
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]),
            name: None,
            function_call: None,
        }
    }
}

/// OpenAI request
#[derive(Debug, Clone, serde::Serialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<serde_json::Value>,
    pub stream: bool,
}

impl OpenAIRequest {
    pub fn new(model: &str, messages: Vec<OpenAIMessage>) -> Self {
        Self {
            model: model.to_string(),
            messages,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            functions: None,
            function_call: None,
            stream: false,
        }
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

/// OpenAI response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIChoice {
    pub index: usize,
    pub message: OpenAIResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub function_call: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI error response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorDetail,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenAIErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// OpenAI client configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub organization: Option<String>,
    pub timeout: Duration,
    pub max_retries: usize,
}

impl OpenAIConfig {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
        }
    }

    pub fn azure(endpoint: &str, api_key: &str, deployment: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: format!("{}/openai/deployments/{}", endpoint, deployment),
            organization: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
        }
    }

    pub fn local(base_url: &str) -> Self {
        Self {
            api_key: String::new(),
            base_url: base_url.to_string(),
            organization: None,
            timeout: Duration::from_secs(120),
            max_retries: 1,
        }
    }
}

/// OpenAI client
pub struct OpenAIClient {
    config: OpenAIConfig,
}

impl OpenAIClient {
    pub fn new(config: OpenAIConfig) -> Self {
        Self { config }
    }

    /// Create a chat completion
    pub fn chat(&self, request: OpenAIRequest) -> Result<OpenAIResponse, OpenAIAdapterError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut req = ureq::post(&url)
            .timeout(self.config.timeout)
            .set("Content-Type", "application/json");

        if !self.config.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.config.api_key));
        }

        if let Some(org) = &self.config.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let body = serde_json::to_string(&request)
            .map_err(|e| OpenAIAdapterError::Serialization(e.to_string()))?;

        let response = req.send_string(&body);

        match response {
            Ok(resp) => {
                let text = resp
                    .into_string()
                    .map_err(|e| OpenAIAdapterError::Network(e.to_string()))?;

                serde_json::from_str(&text)
                    .map_err(|e| OpenAIAdapterError::Deserialization(format!("{}: {}", e, text)))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                if let Ok(error) = serde_json::from_str::<OpenAIError>(&text) {
                    Err(OpenAIAdapterError::Api {
                        code,
                        message: error.error.message,
                        error_type: error.error.error_type,
                    })
                } else {
                    Err(OpenAIAdapterError::Api {
                        code,
                        message: text,
                        error_type: "unknown".to_string(),
                    })
                }
            }
            Err(e) => Err(OpenAIAdapterError::Network(e.to_string())),
        }
    }

    /// Create a streaming chat completion
    pub fn chat_stream<F>(
        &self,
        request: OpenAIRequest,
        mut on_chunk: F,
    ) -> Result<String, OpenAIAdapterError>
    where
        F: FnMut(&str),
    {
        let mut streaming_request = request;
        streaming_request.stream = true;

        let url = format!("{}/chat/completions", self.config.base_url);

        let mut req = ureq::post(&url)
            .timeout(self.config.timeout)
            .set("Content-Type", "application/json");

        if !self.config.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.config.api_key));
        }

        let body = serde_json::to_string(&streaming_request)
            .map_err(|e| OpenAIAdapterError::Serialization(e.to_string()))?;

        let response = req
            .send_string(&body)
            .map_err(|e| OpenAIAdapterError::Network(e.to_string()))?;

        let reader = response.into_reader();
        let buf_reader = std::io::BufReader::new(reader);

        let mut full_response = String::new();

        use std::io::BufRead;
        for line in buf_reader.lines() {
            let line = line.map_err(|e| OpenAIAdapterError::Network(e.to_string()))?;

            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    break;
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            on_chunk(content);
                            full_response.push_str(content);
                        }
                    }
                }
            }
        }

        Ok(full_response)
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>, OpenAIAdapterError> {
        let url = format!("{}/models", self.config.base_url);

        let mut req = ureq::get(&url).timeout(self.config.timeout);

        if !self.config.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.config.api_key));
        }

        let response = req
            .call()
            .map_err(|e| OpenAIAdapterError::Network(e.to_string()))?;

        let text = response
            .into_string()
            .map_err(|e| OpenAIAdapterError::Network(e.to_string()))?;

        let models: ModelsResponse = serde_json::from_str(&text)
            .map_err(|e| OpenAIAdapterError::Deserialization(e.to_string()))?;

        Ok(models.data.into_iter().map(|m| m.id).collect())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamChunk {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamChoice {
    pub index: Option<usize>,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelInfo {
    id: String,
}

/// OpenAI adapter error
#[derive(Debug)]
pub enum OpenAIAdapterError {
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

impl std::fmt::Display for OpenAIAdapterError {
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

impl std::error::Error for OpenAIAdapterError {}

/// Simple chat helper
pub fn simple_chat(
    api_key: &str,
    model: &str,
    messages: Vec<(&str, &str)>,
) -> Result<String, OpenAIAdapterError> {
    let client = OpenAIClient::new(OpenAIConfig::new(api_key));

    let openai_messages: Vec<OpenAIMessage> = messages
        .into_iter()
        .map(|(role, content)| OpenAIMessage {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
            name: None,
            function_call: None,
        })
        .collect();

    let request = OpenAIRequest::new(model, openai_messages);
    let response = client.chat(request)?;

    Ok(response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = OpenAIMessage::user("Hello");
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn test_request_builder() {
        let messages = vec![
            OpenAIMessage::system("You are helpful"),
            OpenAIMessage::user("Hello"),
        ];

        let request = OpenAIRequest::new("gpt-3.5-turbo", messages)
            .with_temperature(0.7)
            .with_max_tokens(100);

        assert_eq!(request.model, "gpt-3.5-turbo");
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_config() {
        let config = OpenAIConfig::new("sk-test");
        assert_eq!(config.base_url, "https://api.openai.com/v1");

        let local = OpenAIConfig::local("http://localhost:8000");
        assert!(local.api_key.is_empty());
    }

    #[test]
    fn test_model_presets() {
        let turbo = OpenAIModel::gpt4_turbo();
        assert_eq!(turbo.id, "gpt-4-turbo-preview");
        assert_eq!(turbo.name, "GPT-4 Turbo");
        assert_eq!(turbo.context_length, 128000);
        assert!(turbo.supports_vision);
        assert!(turbo.supports_functions);

        let gpt4 = OpenAIModel::gpt4();
        assert_eq!(gpt4.id, "gpt-4");
        assert_eq!(gpt4.name, "GPT-4");
        assert_eq!(gpt4.context_length, 8192);
        assert!(!gpt4.supports_vision);
        assert!(gpt4.supports_functions);

        let gpt35 = OpenAIModel::gpt35_turbo();
        assert_eq!(gpt35.id, "gpt-3.5-turbo");
        assert_eq!(gpt35.name, "GPT-3.5 Turbo");
        assert_eq!(gpt35.context_length, 16385);
        assert!(!gpt35.supports_vision);
        assert!(gpt35.supports_functions);
    }

    #[test]
    fn test_message_types() {
        let sys = OpenAIMessage::system("Be helpful");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, serde_json::Value::String("Be helpful".to_string()));
        assert!(sys.name.is_none());
        assert!(sys.function_call.is_none());

        let usr = OpenAIMessage::user("Hello");
        assert_eq!(usr.role, "user");
        assert_eq!(usr.content, serde_json::Value::String("Hello".to_string()));

        let asst = OpenAIMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
        assert_eq!(asst.content, serde_json::Value::String("Hi there".to_string()));
    }

    #[test]
    fn test_user_with_image_message() {
        let msg = OpenAIMessage::user_with_image("Describe this", "https://example.com/img.png");
        assert_eq!(msg.role, "user");

        let content = msg.content.as_array().expect("content should be an array");
        assert_eq!(content.len(), 2);

        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Describe this");

        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_request_options() {
        let messages = vec![OpenAIMessage::user("Hi")];
        let req = OpenAIRequest::new("gpt-4", messages)
            .with_temperature(0.5)
            .with_max_tokens(200)
            .streaming();

        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.max_tokens, Some(200));
        assert!(req.stream);

        // Verify defaults for unset fields
        assert!(req.top_p.is_none());
        assert!(req.frequency_penalty.is_none());
        assert!(req.presence_penalty.is_none());
        assert!(req.stop.is_none());
        assert!(req.functions.is_none());
        assert!(req.function_call.is_none());
    }

    #[test]
    fn test_azure_config() {
        let config = OpenAIConfig::azure(
            "https://myresource.openai.azure.com",
            "az-key-123",
            "my-deployment",
        );
        assert_eq!(
            config.base_url,
            "https://myresource.openai.azure.com/openai/deployments/my-deployment"
        );
        assert_eq!(config.api_key, "az-key-123");
        assert!(config.organization.is_none());
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17
            }
        }"#;

        let resp: OpenAIResponse = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(resp.id, "chatcmpl-abc123");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.created, 1700000000);
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello! How can I help?")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));

        let usage = resp.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn test_error_display_formatting() {
        let network_err = OpenAIAdapterError::Network("timeout".to_string());
        assert_eq!(format!("{}", network_err), "Network error: timeout");

        let serial_err = OpenAIAdapterError::Serialization("bad payload".to_string());
        assert_eq!(
            format!("{}", serial_err),
            "Serialization error: bad payload"
        );

        let deser_err = OpenAIAdapterError::Deserialization("missing field".to_string());
        assert_eq!(
            format!("{}", deser_err),
            "Deserialization error: missing field"
        );

        let api_err = OpenAIAdapterError::Api {
            code: 401,
            message: "Invalid API key".to_string(),
            error_type: "authentication_error".to_string(),
        };
        assert_eq!(
            format!("{}", api_err),
            "API error 401: Invalid API key"
        );

        let rate_err_with = OpenAIAdapterError::RateLimit {
            retry_after: Some(Duration::from_secs(60)),
        };
        let display = format!("{}", rate_err_with);
        assert!(display.contains("Rate limited"));
        assert!(display.contains("retry after"));

        let rate_err_without = OpenAIAdapterError::RateLimit { retry_after: None };
        assert_eq!(format!("{}", rate_err_without), "Rate limited");
    }
}
