//! Hugging Face connector
//!
//! Connect to Hugging Face Inference API and local models.

use std::time::Duration;

/// Hugging Face model info
#[derive(Debug, Clone)]
pub struct HfModel {
    pub id: String,
    pub task: HfTask,
    pub pipeline_tag: Option<String>,
    pub downloads: usize,
    pub likes: usize,
}

/// Hugging Face task types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfTask {
    TextGeneration,
    Text2TextGeneration,
    Conversational,
    Summarization,
    Translation,
    QuestionAnswering,
    FeatureExtraction,
    FillMask,
    TokenClassification,
    TextClassification,
    ZeroShotClassification,
    SentenceSimilarity,
    ImageClassification,
    ObjectDetection,
    ImageSegmentation,
    ImageToText,
    TextToImage,
    AudioClassification,
    AutomaticSpeechRecognition,
}

impl HfTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextGeneration => "text-generation",
            Self::Text2TextGeneration => "text2text-generation",
            Self::Conversational => "conversational",
            Self::Summarization => "summarization",
            Self::Translation => "translation",
            Self::QuestionAnswering => "question-answering",
            Self::FeatureExtraction => "feature-extraction",
            Self::FillMask => "fill-mask",
            Self::TokenClassification => "token-classification",
            Self::TextClassification => "text-classification",
            Self::ZeroShotClassification => "zero-shot-classification",
            Self::SentenceSimilarity => "sentence-similarity",
            Self::ImageClassification => "image-classification",
            Self::ObjectDetection => "object-detection",
            Self::ImageSegmentation => "image-segmentation",
            Self::ImageToText => "image-to-text",
            Self::TextToImage => "text-to-image",
            Self::AudioClassification => "audio-classification",
            Self::AutomaticSpeechRecognition => "automatic-speech-recognition",
        }
    }
}

/// Hugging Face generation parameters
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_full_text: Option<bool>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(256),
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(50),
            repetition_penalty: Some(1.1),
            do_sample: Some(true),
            return_full_text: Some(false),
        }
    }
}

/// Hugging Face request
#[derive(Debug, Clone, serde::Serialize)]
pub struct HfRequest {
    pub inputs: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerationParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HfOptions>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HfOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wait_for_model: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_cache: Option<bool>,
}

impl HfRequest {
    pub fn text_generation(text: &str) -> Self {
        Self {
            inputs: serde_json::Value::String(text.to_string()),
            parameters: Some(GenerationParams::default()),
            options: Some(HfOptions {
                wait_for_model: Some(true),
                use_cache: Some(true),
            }),
        }
    }

    pub fn conversational(messages: Vec<(&str, &str)>) -> Self {
        let past_user_inputs: Vec<_> = messages.iter()
            .filter(|(role, _)| *role == "user")
            .map(|(_, content)| content.to_string())
            .collect();

        let generated_responses: Vec<_> = messages.iter()
            .filter(|(role, _)| *role == "assistant")
            .map(|(_, content)| content.to_string())
            .collect();

        let text = messages.last().map(|(_, c)| c.to_string()).unwrap_or_default();

        Self {
            inputs: serde_json::json!({
                "text": text,
                "past_user_inputs": past_user_inputs,
                "generated_responses": generated_responses
            }),
            parameters: None,
            options: Some(HfOptions {
                wait_for_model: Some(true),
                use_cache: Some(false),
            }),
        }
    }

    pub fn question_answering(question: &str, context: &str) -> Self {
        Self {
            inputs: serde_json::json!({
                "question": question,
                "context": context
            }),
            parameters: None,
            options: None,
        }
    }

    pub fn summarization(text: &str) -> Self {
        Self {
            inputs: serde_json::Value::String(text.to_string()),
            parameters: Some(GenerationParams {
                max_new_tokens: Some(150),
                ..Default::default()
            }),
            options: None,
        }
    }

    pub fn embeddings(texts: Vec<&str>) -> Self {
        Self {
            inputs: serde_json::json!(texts),
            parameters: None,
            options: Some(HfOptions {
                wait_for_model: Some(true),
                use_cache: Some(true),
            }),
        }
    }

    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.parameters = Some(params);
        self
    }
}

/// Hugging Face response
#[derive(Debug, Clone)]
pub enum HfResponse {
    TextGeneration(Vec<TextGenerationResult>),
    Conversational(ConversationalResult),
    QuestionAnswering(QAResult),
    Embeddings(Vec<Vec<f32>>),
    Classification(Vec<ClassificationResult>),
    Raw(serde_json::Value),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextGenerationResult {
    pub generated_text: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConversationalResult {
    pub generated_text: String,
    pub conversation: ConversationHistory,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ConversationHistory {
    pub past_user_inputs: Vec<String>,
    pub generated_responses: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QAResult {
    pub answer: String,
    pub score: f64,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f64,
}

/// Hugging Face client configuration
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub api_token: Option<String>,
    pub base_url: String,
    pub timeout: Duration,
}

impl HfConfig {
    pub fn new() -> Self {
        Self {
            api_token: None,
            base_url: "https://api-inference.huggingface.co/models".to_string(),
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_token(mut self, token: &str) -> Self {
        self.api_token = Some(token.to_string());
        self
    }

    pub fn local(base_url: &str) -> Self {
        Self {
            api_token: None,
            base_url: base_url.to_string(),
            timeout: Duration::from_secs(300),
        }
    }
}

impl Default for HfConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Hugging Face client
pub struct HfClient {
    config: HfConfig,
}

impl HfClient {
    pub fn new(config: HfConfig) -> Self {
        Self { config }
    }

    /// Make an inference request
    pub fn infer(&self, model: &str, request: HfRequest) -> Result<HfResponse, HfError> {
        let url = format!("{}/{}", self.config.base_url, model);

        let mut req = ureq::post(&url)
            .timeout(self.config.timeout)
            .set("Content-Type", "application/json");

        if let Some(token) = &self.config.api_token {
            req = req.set("Authorization", &format!("Bearer {}", token));
        }

        let body = serde_json::to_string(&request)
            .map_err(|e| HfError::Serialization(e.to_string()))?;

        let response = req.send_string(&body);

        match response {
            Ok(resp) => {
                let text = resp.into_string()
                    .map_err(|e| HfError::Network(e.to_string()))?;

                self.parse_response(&text)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                Err(HfError::Api { code, message: text })
            }
            Err(e) => Err(HfError::Network(e.to_string())),
        }
    }

    fn parse_response(&self, text: &str) -> Result<HfResponse, HfError> {
        // Try different response formats
        if let Ok(results) = serde_json::from_str::<Vec<TextGenerationResult>>(text) {
            return Ok(HfResponse::TextGeneration(results));
        }

        if let Ok(result) = serde_json::from_str::<ConversationalResult>(text) {
            return Ok(HfResponse::Conversational(result));
        }

        if let Ok(result) = serde_json::from_str::<QAResult>(text) {
            return Ok(HfResponse::QuestionAnswering(result));
        }

        if let Ok(embeddings) = serde_json::from_str::<Vec<Vec<f32>>>(text) {
            return Ok(HfResponse::Embeddings(embeddings));
        }

        if let Ok(results) = serde_json::from_str::<Vec<Vec<ClassificationResult>>>(text) {
            return Ok(HfResponse::Classification(results.into_iter().flatten().collect()));
        }

        // Fall back to raw JSON
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| HfError::Deserialization(e.to_string()))?;

        Ok(HfResponse::Raw(value))
    }

    /// Generate text
    pub fn generate(&self, model: &str, prompt: &str) -> Result<String, HfError> {
        let request = HfRequest::text_generation(prompt);
        let response = self.infer(model, request)?;

        match response {
            HfResponse::TextGeneration(results) => {
                Ok(results.first().map(|r| r.generated_text.clone()).unwrap_or_default())
            }
            _ => Err(HfError::UnexpectedResponse),
        }
    }

    /// Get embeddings
    pub fn embed(&self, model: &str, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, HfError> {
        let request = HfRequest::embeddings(texts);
        let response = self.infer(model, request)?;

        match response {
            HfResponse::Embeddings(embeds) => Ok(embeds),
            _ => Err(HfError::UnexpectedResponse),
        }
    }

    /// Answer a question
    pub fn answer(&self, model: &str, question: &str, context: &str) -> Result<QAResult, HfError> {
        let request = HfRequest::question_answering(question, context);
        let response = self.infer(model, request)?;

        match response {
            HfResponse::QuestionAnswering(result) => Ok(result),
            _ => Err(HfError::UnexpectedResponse),
        }
    }

    /// Summarize text
    pub fn summarize(&self, model: &str, text: &str) -> Result<String, HfError> {
        let request = HfRequest::summarization(text);
        let response = self.infer(model, request)?;

        match response {
            HfResponse::TextGeneration(results) => {
                Ok(results.first().map(|r| r.generated_text.clone()).unwrap_or_default())
            }
            HfResponse::Raw(v) => {
                // Handle summary_text format
                if let Some(arr) = v.as_array() {
                    if let Some(obj) = arr.first().and_then(|v| v.as_object()) {
                        if let Some(text) = obj.get("summary_text").and_then(|v| v.as_str()) {
                            return Ok(text.to_string());
                        }
                    }
                }
                Err(HfError::UnexpectedResponse)
            }
            _ => Err(HfError::UnexpectedResponse),
        }
    }
}

impl Default for HfClient {
    fn default() -> Self {
        Self::new(HfConfig::default())
    }
}

/// Hugging Face error
#[derive(Debug)]
pub enum HfError {
    Network(String),
    Serialization(String),
    Deserialization(String),
    Api { code: u16, message: String },
    ModelLoading,
    UnexpectedResponse,
}

impl std::fmt::Display for HfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "Network error: {}", e),
            Self::Serialization(e) => write!(f, "Serialization error: {}", e),
            Self::Deserialization(e) => write!(f, "Deserialization error: {}", e),
            Self::Api { code, message } => write!(f, "API error {}: {}", code, message),
            Self::ModelLoading => write!(f, "Model is loading, try again"),
            Self::UnexpectedResponse => write!(f, "Unexpected response format"),
        }
    }
}

impl std::error::Error for HfError {}

/// Popular models
pub mod popular_models {
    pub const LLAMA2_7B: &str = "meta-llama/Llama-2-7b-chat-hf";
    pub const MISTRAL_7B: &str = "mistralai/Mistral-7B-Instruct-v0.1";
    pub const FALCON_7B: &str = "tiiuae/falcon-7b-instruct";
    pub const FLAN_T5_BASE: &str = "google/flan-t5-base";
    pub const BERT_BASE: &str = "bert-base-uncased";
    pub const ROBERTA_BASE: &str = "roberta-base";
    pub const SENTENCE_TRANSFORMERS: &str = "sentence-transformers/all-MiniLM-L6-v2";
    pub const WHISPER_SMALL: &str = "openai/whisper-small";
    pub const STABLE_DIFFUSION: &str = "stabilityai/stable-diffusion-2-1";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = HfRequest::text_generation("Hello, world!");
        assert!(req.parameters.is_some());
    }

    #[test]
    fn test_qa_request() {
        let req = HfRequest::question_answering(
            "What is Python?",
            "Python is a programming language."
        );
        assert!(req.inputs.is_object());
    }

    #[test]
    fn test_config() {
        let config = HfConfig::new().with_token("hf_test");
        assert_eq!(config.api_token, Some("hf_test".to_string()));
    }
}
