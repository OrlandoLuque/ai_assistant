//! Fine-tuning and LoRA adapter support
//!
//! Provides comprehensive fine-tuning capabilities including:
//! - Training dataset management (multiple formats)
//! - OpenAI-compatible fine-tuning API
//! - LoRA adapter management
//! - Hyperparameter configuration
//! - Training job monitoring

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Training data format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingFormat {
    /// OpenAI chat format (messages array)
    OpenAIChat,
    /// OpenAI completion format (prompt/completion pairs)
    OpenAICompletion,
    /// Alpaca instruction format
    Alpaca,
    /// ShareGPT conversation format
    ShareGPT,
    /// Custom JSONL format
    CustomJsonl,
}

impl Default for TrainingFormat {
    fn default() -> Self {
        Self::OpenAIChat
    }
}

/// Chat training example (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTrainingExample {
    pub messages: Vec<ChatTrainingMessage>,
}

/// Chat training message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTrainingMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<serde_json::Value>,
}

impl ChatTrainingMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
            name: None,
            function_call: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
            name: None,
            function_call: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            name: None,
            function_call: None,
        }
    }
}

/// Completion training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTrainingExample {
    pub prompt: String,
    pub completion: String,
}

/// Alpaca instruction format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlpacaTrainingExample {
    pub instruction: String,
    #[serde(default)]
    pub input: String,
    pub output: String,
}

/// ShareGPT conversation format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareGPTExample {
    pub conversations: Vec<ShareGPTTurn>,
}

/// ShareGPT turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareGPTTurn {
    pub from: String, // "human", "gpt", "system"
    pub value: String,
}

/// Training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    pub name: String,
    pub format: TrainingFormat,
    pub examples: Vec<serde_json::Value>,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
}

impl TrainingDataset {
    /// Create a new empty dataset
    pub fn new(name: &str, format: TrainingFormat) -> Self {
        Self {
            name: name.to_string(),
            format,
            examples: Vec::new(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: HashMap::new(),
        }
    }

    /// Add a chat example
    pub fn add_chat_example(&mut self, example: ChatTrainingExample) -> Result<(), DatasetError> {
        if self.format != TrainingFormat::OpenAIChat {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::OpenAIChat,
                got: self.format,
            });
        }
        self.examples.push(serde_json::to_value(example)?);
        Ok(())
    }

    /// Add a completion example
    pub fn add_completion_example(&mut self, example: CompletionTrainingExample) -> Result<(), DatasetError> {
        if self.format != TrainingFormat::OpenAICompletion {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::OpenAICompletion,
                got: self.format,
            });
        }
        self.examples.push(serde_json::to_value(example)?);
        Ok(())
    }

    /// Add an Alpaca example
    pub fn add_alpaca_example(&mut self, example: AlpacaTrainingExample) -> Result<(), DatasetError> {
        if self.format != TrainingFormat::Alpaca {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::Alpaca,
                got: self.format,
            });
        }
        self.examples.push(serde_json::to_value(example)?);
        Ok(())
    }

    /// Add a ShareGPT example
    pub fn add_sharegpt_example(&mut self, example: ShareGPTExample) -> Result<(), DatasetError> {
        if self.format != TrainingFormat::ShareGPT {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::ShareGPT,
                got: self.format,
            });
        }
        self.examples.push(serde_json::to_value(example)?);
        Ok(())
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Export to JSONL format
    pub fn to_jsonl(&self) -> String {
        self.examples
            .iter()
            .filter_map(|ex| serde_json::to_string(ex).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Load from JSONL string
    pub fn from_jsonl(name: &str, format: TrainingFormat, jsonl: &str) -> Result<Self, DatasetError> {
        let mut dataset = Self::new(name, format);

        for (line_num, line) in jsonl.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| DatasetError::ParseError {
                    line: line_num + 1,
                    error: e.to_string(),
                })?;

            dataset.examples.push(value);
        }

        Ok(dataset)
    }

    /// Validate dataset structure
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::default();

        for (idx, example) in self.examples.iter().enumerate() {
            match self.format {
                TrainingFormat::OpenAIChat => {
                    if let Err(e) = self.validate_chat_example(example) {
                        result.errors.push(ValidationError {
                            index: idx,
                            message: e,
                        });
                    }
                }
                TrainingFormat::OpenAICompletion => {
                    if let Err(e) = self.validate_completion_example(example) {
                        result.errors.push(ValidationError {
                            index: idx,
                            message: e,
                        });
                    }
                }
                TrainingFormat::Alpaca => {
                    if let Err(e) = self.validate_alpaca_example(example) {
                        result.errors.push(ValidationError {
                            index: idx,
                            message: e,
                        });
                    }
                }
                TrainingFormat::ShareGPT => {
                    if let Err(e) = self.validate_sharegpt_example(example) {
                        result.errors.push(ValidationError {
                            index: idx,
                            message: e,
                        });
                    }
                }
                TrainingFormat::CustomJsonl => {
                    // Custom format - no validation
                }
            }
        }

        result.valid_count = self.examples.len() - result.errors.len();
        result.total_count = self.examples.len();
        result
    }

    fn validate_chat_example(&self, example: &serde_json::Value) -> Result<(), String> {
        let messages = example.get("messages")
            .ok_or("Missing 'messages' field")?
            .as_array()
            .ok_or("'messages' must be an array")?;

        if messages.is_empty() {
            return Err("Messages array is empty".to_string());
        }

        let mut has_assistant = false;
        for msg in messages {
            let role = msg.get("role")
                .and_then(|r| r.as_str())
                .ok_or("Message missing 'role'")?;

            if msg.get("content").is_none() {
                return Err(format!("Message with role '{}' missing 'content'", role));
            }

            if role == "assistant" {
                has_assistant = true;
            }
        }

        if !has_assistant {
            return Err("No assistant message found".to_string());
        }

        Ok(())
    }

    fn validate_completion_example(&self, example: &serde_json::Value) -> Result<(), String> {
        if example.get("prompt").is_none() {
            return Err("Missing 'prompt' field".to_string());
        }
        if example.get("completion").is_none() {
            return Err("Missing 'completion' field".to_string());
        }
        Ok(())
    }

    fn validate_alpaca_example(&self, example: &serde_json::Value) -> Result<(), String> {
        if example.get("instruction").is_none() {
            return Err("Missing 'instruction' field".to_string());
        }
        if example.get("output").is_none() {
            return Err("Missing 'output' field".to_string());
        }
        Ok(())
    }

    fn validate_sharegpt_example(&self, example: &serde_json::Value) -> Result<(), String> {
        let conversations = example.get("conversations")
            .ok_or("Missing 'conversations' field")?
            .as_array()
            .ok_or("'conversations' must be an array")?;

        if conversations.is_empty() {
            return Err("Conversations array is empty".to_string());
        }

        for turn in conversations {
            if turn.get("from").is_none() {
                return Err("Turn missing 'from' field".to_string());
            }
            if turn.get("value").is_none() {
                return Err("Turn missing 'value' field".to_string());
            }
        }

        Ok(())
    }

    /// Split dataset into train/validation
    pub fn split(&self, validation_ratio: f32) -> (TrainingDataset, TrainingDataset) {
        let split_idx = ((1.0 - validation_ratio) * self.examples.len() as f32) as usize;

        let mut train = Self::new(&format!("{}_train", self.name), self.format);
        let mut val = Self::new(&format!("{}_val", self.name), self.format);

        train.examples = self.examples[..split_idx].to_vec();
        val.examples = self.examples[split_idx..].to_vec();

        (train, val)
    }

    /// Shuffle dataset (simple deterministic shuffle)
    pub fn shuffle(&mut self, seed: u64) {
        let n = self.examples.len();
        if n <= 1 {
            return;
        }

        // Simple LCG shuffle
        let mut state = seed;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            self.examples.swap(i, j);
        }
    }
}

/// Dataset validation result
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub valid_count: usize,
    pub total_count: usize,
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub index: usize,
    pub message: String,
}

/// Dataset error types
#[derive(Debug)]
pub enum DatasetError {
    FormatMismatch {
        expected: TrainingFormat,
        got: TrainingFormat,
    },
    ParseError {
        line: usize,
        error: String,
    },
    SerializationError(serde_json::Error),
    IoError(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatMismatch { expected, got } => {
                write!(f, "Format mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::ParseError { line, error } => {
                write!(f, "Parse error at line {}: {}", line, error)
            }
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<serde_json::Error> for DatasetError {
    fn from(e: serde_json::Error) -> Self {
        Self::SerializationError(e)
    }
}

/// Fine-tuning hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Number of training epochs
    pub n_epochs: Option<u32>,
    /// Batch size (number of examples per batch)
    pub batch_size: Option<u32>,
    /// Learning rate multiplier
    pub learning_rate_multiplier: Option<f64>,
    /// Warmup ratio
    pub warmup_ratio: Option<f64>,
    /// Weight decay
    pub weight_decay: Option<f64>,
    /// Maximum sequence length
    pub max_seq_length: Option<u32>,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: Option<u32>,
    /// LoRA rank (for LoRA fine-tuning)
    pub lora_rank: Option<u32>,
    /// LoRA alpha
    pub lora_alpha: Option<f32>,
    /// LoRA dropout
    pub lora_dropout: Option<f32>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            n_epochs: Some(3),
            batch_size: None,
            learning_rate_multiplier: None,
            warmup_ratio: Some(0.1),
            weight_decay: Some(0.01),
            max_seq_length: None,
            gradient_accumulation_steps: Some(1),
            lora_rank: None,
            lora_alpha: None,
            lora_dropout: None,
        }
    }
}

impl Hyperparameters {
    /// Configure for LoRA fine-tuning
    pub fn with_lora(mut self, rank: u32, alpha: f32, dropout: f32) -> Self {
        self.lora_rank = Some(rank);
        self.lora_alpha = Some(alpha);
        self.lora_dropout = Some(dropout);
        self
    }
}

/// Fine-tuning job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FineTuneStatus {
    Validating,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// Fine-tuning job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneJob {
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub status: FineTuneStatus,
    pub fine_tuned_model: Option<String>,
    pub training_file: String,
    pub validation_file: Option<String>,
    pub hyperparameters: Hyperparameters,
    pub result_files: Vec<String>,
    pub trained_tokens: Option<u64>,
    pub error: Option<FineTuneError>,
    pub events: Vec<FineTuneEvent>,
}

/// Fine-tune error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneError {
    pub code: String,
    pub message: String,
}

/// Fine-tune event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneEvent {
    pub created_at: u64,
    pub level: String,
    pub message: String,
}

/// OpenAI Fine-tuning API client
pub struct OpenAIFineTuneClient {
    api_key: String,
    base_url: String,
    organization: Option<String>,
}

impl OpenAIFineTuneClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        }
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    pub fn with_organization(mut self, org: &str) -> Self {
        self.organization = Some(org.to_string());
        self
    }

    /// Upload a training file
    pub fn upload_file(&self, dataset: &TrainingDataset) -> Result<FileUploadResponse, FineTuneApiError> {
        let jsonl = dataset.to_jsonl();

        let mut request = ureq::post(&format!("{}/files", self.base_url))
            .set("Authorization", &format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            request = request.set("OpenAI-Organization", org);
        }

        // For actual multipart upload, we'd need more complex handling
        // This is a simplified version
        let response = request
            .set("Content-Type", "application/json")
            .send_json(ureq::json!({
                "purpose": "fine-tune",
                "file": jsonl,
            }))
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: serde_json::Value = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(FileUploadResponse {
            id: result["id"].as_str().unwrap_or_default().to_string(),
            bytes: result["bytes"].as_u64().unwrap_or(0),
            created_at: result["created_at"].as_u64().unwrap_or(0),
            filename: result["filename"].as_str().unwrap_or_default().to_string(),
            purpose: result["purpose"].as_str().unwrap_or_default().to_string(),
        })
    }

    /// Create a fine-tuning job
    pub fn create_job(&self, request: CreateFineTuneRequest) -> Result<FineTuneJob, FineTuneApiError> {
        let mut req = ureq::post(&format!("{}/fine_tuning/jobs", self.base_url))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json");

        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let body = serde_json::json!({
            "model": request.model,
            "training_file": request.training_file,
            "validation_file": request.validation_file,
            "hyperparameters": {
                "n_epochs": request.hyperparameters.n_epochs,
                "batch_size": request.hyperparameters.batch_size,
                "learning_rate_multiplier": request.hyperparameters.learning_rate_multiplier,
            },
            "suffix": request.suffix,
        });

        let response = req.send_json(body)
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: FineTuneJob = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(result)
    }

    /// Get fine-tuning job status
    pub fn get_job(&self, job_id: &str) -> Result<FineTuneJob, FineTuneApiError> {
        let mut req = ureq::get(&format!("{}/fine_tuning/jobs/{}", self.base_url, job_id))
            .set("Authorization", &format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let response = req.call()
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: FineTuneJob = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(result)
    }

    /// List fine-tuning jobs
    pub fn list_jobs(&self, limit: Option<u32>) -> Result<Vec<FineTuneJob>, FineTuneApiError> {
        let url = match limit {
            Some(l) => format!("{}/fine_tuning/jobs?limit={}", self.base_url, l),
            None => format!("{}/fine_tuning/jobs", self.base_url),
        };

        let mut req = ureq::get(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let response = req.call()
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: serde_json::Value = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        let jobs: Vec<FineTuneJob> = serde_json::from_value(result["data"].clone())
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(jobs)
    }

    /// Cancel a fine-tuning job
    pub fn cancel_job(&self, job_id: &str) -> Result<FineTuneJob, FineTuneApiError> {
        let mut req = ureq::post(&format!("{}/fine_tuning/jobs/{}/cancel", self.base_url, job_id))
            .set("Authorization", &format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let response = req.call()
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: FineTuneJob = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(result)
    }

    /// List events for a fine-tuning job
    pub fn list_events(&self, job_id: &str) -> Result<Vec<FineTuneEvent>, FineTuneApiError> {
        let mut req = ureq::get(&format!("{}/fine_tuning/jobs/{}/events", self.base_url, job_id))
            .set("Authorization", &format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }

        let response = req.call()
            .map_err(|e| FineTuneApiError::NetworkError(e.to_string()))?;

        let result: serde_json::Value = response.into_json()
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        let events: Vec<FineTuneEvent> = serde_json::from_value(result["data"].clone())
            .map_err(|e| FineTuneApiError::ParseError(e.to_string()))?;

        Ok(events)
    }
}

/// File upload response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUploadResponse {
    pub id: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

/// Create fine-tune request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFineTuneRequest {
    pub model: String,
    pub training_file: String,
    pub validation_file: Option<String>,
    pub hyperparameters: Hyperparameters,
    pub suffix: Option<String>,
}

impl CreateFineTuneRequest {
    pub fn new(model: &str, training_file: &str) -> Self {
        Self {
            model: model.to_string(),
            training_file: training_file.to_string(),
            validation_file: None,
            hyperparameters: Hyperparameters::default(),
            suffix: None,
        }
    }

    pub fn with_validation(mut self, file_id: &str) -> Self {
        self.validation_file = Some(file_id.to_string());
        self
    }

    pub fn with_hyperparameters(mut self, params: Hyperparameters) -> Self {
        self.hyperparameters = params;
        self
    }

    pub fn with_suffix(mut self, suffix: &str) -> Self {
        self.suffix = Some(suffix.to_string());
        self
    }
}

/// Fine-tune API error
#[derive(Debug)]
pub enum FineTuneApiError {
    NetworkError(String),
    ParseError(String),
    ApiError { code: String, message: String },
    Unauthorized,
    RateLimited,
    NotFound,
}

impl std::fmt::Display for FineTuneApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NetworkError(e) => write!(f, "Network error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::ApiError { code, message } => write!(f, "API error {}: {}", code, message),
            Self::Unauthorized => write!(f, "Unauthorized"),
            Self::RateLimited => write!(f, "Rate limited"),
            Self::NotFound => write!(f, "Not found"),
        }
    }
}

impl std::error::Error for FineTuneApiError {}

/// Dataset format converter
pub struct DatasetConverter;

impl DatasetConverter {
    /// Convert Alpaca format to OpenAI Chat format
    pub fn alpaca_to_chat(dataset: &TrainingDataset) -> Result<TrainingDataset, DatasetError> {
        if dataset.format != TrainingFormat::Alpaca {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::Alpaca,
                got: dataset.format,
            });
        }

        let mut chat_dataset = TrainingDataset::new(&dataset.name, TrainingFormat::OpenAIChat);

        for example in &dataset.examples {
            let alpaca: AlpacaTrainingExample = serde_json::from_value(example.clone())?;

            let user_content = if alpaca.input.is_empty() {
                alpaca.instruction
            } else {
                format!("{}\n\nInput: {}", alpaca.instruction, alpaca.input)
            };

            let chat_example = ChatTrainingExample {
                messages: vec![
                    ChatTrainingMessage::user(&user_content),
                    ChatTrainingMessage::assistant(&alpaca.output),
                ],
            };

            chat_dataset.examples.push(serde_json::to_value(chat_example)?);
        }

        Ok(chat_dataset)
    }

    /// Convert ShareGPT to OpenAI Chat format
    pub fn sharegpt_to_chat(dataset: &TrainingDataset) -> Result<TrainingDataset, DatasetError> {
        if dataset.format != TrainingFormat::ShareGPT {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::ShareGPT,
                got: dataset.format,
            });
        }

        let mut chat_dataset = TrainingDataset::new(&dataset.name, TrainingFormat::OpenAIChat);

        for example in &dataset.examples {
            let sharegpt: ShareGPTExample = serde_json::from_value(example.clone())?;

            let messages: Vec<ChatTrainingMessage> = sharegpt.conversations
                .iter()
                .map(|turn| {
                    let role = match turn.from.as_str() {
                        "human" => "user",
                        "gpt" | "assistant" => "assistant",
                        "system" => "system",
                        _ => "user",
                    };
                    ChatTrainingMessage {
                        role: role.to_string(),
                        content: turn.value.clone(),
                        name: None,
                        function_call: None,
                    }
                })
                .collect();

            let chat_example = ChatTrainingExample { messages };
            chat_dataset.examples.push(serde_json::to_value(chat_example)?);
        }

        Ok(chat_dataset)
    }

    /// Convert Completion to Chat format
    pub fn completion_to_chat(dataset: &TrainingDataset, system_prompt: Option<&str>) -> Result<TrainingDataset, DatasetError> {
        if dataset.format != TrainingFormat::OpenAICompletion {
            return Err(DatasetError::FormatMismatch {
                expected: TrainingFormat::OpenAICompletion,
                got: dataset.format,
            });
        }

        let mut chat_dataset = TrainingDataset::new(&dataset.name, TrainingFormat::OpenAIChat);

        for example in &dataset.examples {
            let completion: CompletionTrainingExample = serde_json::from_value(example.clone())?;

            let mut messages = Vec::new();

            if let Some(sys) = system_prompt {
                messages.push(ChatTrainingMessage::system(sys));
            }

            messages.push(ChatTrainingMessage::user(&completion.prompt));
            messages.push(ChatTrainingMessage::assistant(&completion.completion));

            let chat_example = ChatTrainingExample { messages };
            chat_dataset.examples.push(serde_json::to_value(chat_example)?);
        }

        Ok(chat_dataset)
    }
}

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices
    pub rank: u32,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Whether to use bias
    pub bias: LoraBias,
    /// Task type
    pub task_type: LoraTaskType,
}

/// LoRA bias configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoraBias {
    None,
    All,
    LoraOnly,
}

impl Default for LoraBias {
    fn default() -> Self {
        Self::None
    }
}

/// LoRA task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoraTaskType {
    CausalLM,
    Seq2Seq,
    SequenceClassification,
    TokenClassification,
    QuestionAnswering,
}

impl Default for LoraTaskType {
    fn default() -> Self {
        Self::CausalLM
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
            ],
            bias: LoraBias::None,
            task_type: LoraTaskType::CausalLM,
        }
    }
}

impl LoraConfig {
    /// Create config for Llama-style models
    pub fn for_llama() -> Self {
        Self {
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            ..Default::default()
        }
    }

    /// Create config for GPT-style models
    pub fn for_gpt() -> Self {
        Self {
            target_modules: vec![
                "c_attn".to_string(),
                "c_proj".to_string(),
                "c_fc".to_string(),
            ],
            ..Default::default()
        }
    }

    /// Create config for Mistral-style models
    pub fn for_mistral() -> Self {
        Self::for_llama() // Same architecture
    }

    /// Calculate trainable parameters
    pub fn trainable_params(&self, hidden_size: u64, num_layers: u32) -> u64 {
        let params_per_module = 2 * hidden_size * self.rank as u64;
        params_per_module * self.target_modules.len() as u64 * num_layers as u64
    }
}

/// LoRA adapter metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub name: String,
    pub base_model: String,
    pub config: LoraConfig,
    pub created_at: u64,
    pub path: String,
    pub metadata: HashMap<String, String>,
}

impl LoraAdapter {
    /// Create new adapter metadata
    pub fn new(name: &str, base_model: &str, config: LoraConfig, path: &str) -> Self {
        Self {
            name: name.to_string(),
            base_model: base_model.to_string(),
            config,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            path: path.to_string(),
            metadata: HashMap::new(),
        }
    }
}

/// LoRA adapter manager
pub struct LoraManager {
    adapters: HashMap<String, LoraAdapter>,
    active_adapter: Option<String>,
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
        }
    }

    /// Register an adapter
    pub fn register(&mut self, adapter: LoraAdapter) {
        self.adapters.insert(adapter.name.clone(), adapter);
    }

    /// Get an adapter by name
    pub fn get(&self, name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(name)
    }

    /// List all adapters
    pub fn list(&self) -> Vec<&LoraAdapter> {
        self.adapters.values().collect()
    }

    /// List adapters for a specific base model
    pub fn list_for_model(&self, base_model: &str) -> Vec<&LoraAdapter> {
        self.adapters
            .values()
            .filter(|a| a.base_model == base_model)
            .collect()
    }

    /// Set active adapter
    pub fn set_active(&mut self, name: &str) -> bool {
        if self.adapters.contains_key(name) {
            self.active_adapter = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// Get active adapter
    pub fn active(&self) -> Option<&LoraAdapter> {
        self.active_adapter.as_ref().and_then(|n| self.adapters.get(n))
    }

    /// Clear active adapter
    pub fn clear_active(&mut self) {
        self.active_adapter = None;
    }

    /// Remove an adapter
    pub fn remove(&mut self, name: &str) -> Option<LoraAdapter> {
        if self.active_adapter.as_deref() == Some(name) {
            self.active_adapter = None;
        }
        self.adapters.remove(name)
    }
}

impl Default for LoraManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetrics {
    pub step: u64,
    pub epoch: f32,
    pub loss: f64,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
    pub eval_loss: Option<f64>,
    pub tokens_per_second: Option<f64>,
}

/// Training callback for monitoring progress
pub trait TrainingCallback: Send + Sync {
    fn on_step(&mut self, metrics: &TrainingMetrics);
    fn on_epoch_end(&mut self, epoch: u32, metrics: &TrainingMetrics);
    fn on_eval(&mut self, metrics: &TrainingMetrics);
    fn should_stop(&self) -> bool { false }
}

/// Simple logging callback
pub struct LoggingCallback {
    log_every_n_steps: u64,
}

impl LoggingCallback {
    pub fn new(log_every_n_steps: u64) -> Self {
        Self { log_every_n_steps }
    }
}

impl TrainingCallback for LoggingCallback {
    fn on_step(&mut self, metrics: &TrainingMetrics) {
        if metrics.step % self.log_every_n_steps == 0 {
            println!(
                "Step {}: loss={:.4}, lr={:.2e}",
                metrics.step, metrics.loss, metrics.learning_rate
            );
        }
    }

    fn on_epoch_end(&mut self, epoch: u32, metrics: &TrainingMetrics) {
        println!(
            "Epoch {} complete: loss={:.4}",
            epoch, metrics.loss
        );
    }

    fn on_eval(&mut self, metrics: &TrainingMetrics) {
        if let Some(eval_loss) = metrics.eval_loss {
            println!("Eval loss: {:.4}", eval_loss);
        }
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    patience: u32,
    min_delta: f64,
    best_loss: f64,
    wait_count: u32,
    should_stop: bool,
}

impl EarlyStoppingCallback {
    pub fn new(patience: u32, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait_count: 0,
            should_stop: false,
        }
    }
}

impl TrainingCallback for EarlyStoppingCallback {
    fn on_step(&mut self, _metrics: &TrainingMetrics) {}
    fn on_epoch_end(&mut self, _epoch: u32, _metrics: &TrainingMetrics) {}

    fn on_eval(&mut self, metrics: &TrainingMetrics) {
        if let Some(eval_loss) = metrics.eval_loss {
            if eval_loss < self.best_loss - self.min_delta {
                self.best_loss = eval_loss;
                self.wait_count = 0;
            } else {
                self.wait_count += 1;
                if self.wait_count >= self.patience {
                    self.should_stop = true;
                }
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.should_stop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_dataset_chat() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::OpenAIChat);

        let example = ChatTrainingExample {
            messages: vec![
                ChatTrainingMessage::user("Hello"),
                ChatTrainingMessage::assistant("Hi there!"),
            ],
        };

        dataset.add_chat_example(example).unwrap();
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_training_dataset_alpaca() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::Alpaca);

        let example = AlpacaTrainingExample {
            instruction: "Translate to Spanish".to_string(),
            input: "Hello".to_string(),
            output: "Hola".to_string(),
        };

        dataset.add_alpaca_example(example).unwrap();
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_dataset_format_mismatch() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::OpenAIChat);

        let example = AlpacaTrainingExample {
            instruction: "Test".to_string(),
            input: String::new(),
            output: "Output".to_string(),
        };

        let result = dataset.add_alpaca_example(example);
        assert!(matches!(result, Err(DatasetError::FormatMismatch { .. })));
    }

    #[test]
    fn test_dataset_validation_chat() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::OpenAIChat);

        // Valid example
        dataset.examples.push(serde_json::json!({
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        }));

        // Invalid example (no assistant)
        dataset.examples.push(serde_json::json!({
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }));

        let result = dataset.validate();
        assert_eq!(result.valid_count, 1);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_dataset_split() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::OpenAIChat);

        for i in 0..10 {
            dataset.examples.push(serde_json::json!({"id": i}));
        }

        let (train, val) = dataset.split(0.2);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_dataset_jsonl() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::OpenAIChat);

        dataset.examples.push(serde_json::json!({"messages": [{"role": "user", "content": "a"}]}));
        dataset.examples.push(serde_json::json!({"messages": [{"role": "user", "content": "b"}]}));

        let jsonl = dataset.to_jsonl();
        assert!(jsonl.contains('\n'));

        let loaded = TrainingDataset::from_jsonl("loaded", TrainingFormat::OpenAIChat, &jsonl).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_alpaca_to_chat_conversion() {
        let mut dataset = TrainingDataset::new("alpaca", TrainingFormat::Alpaca);

        dataset.examples.push(serde_json::json!({
            "instruction": "Translate",
            "input": "Hello",
            "output": "Hola"
        }));

        let chat = DatasetConverter::alpaca_to_chat(&dataset).unwrap();
        assert_eq!(chat.format, TrainingFormat::OpenAIChat);
        assert_eq!(chat.len(), 1);
    }

    #[test]
    fn test_completion_to_chat_conversion() {
        let mut dataset = TrainingDataset::new("completion", TrainingFormat::OpenAICompletion);

        dataset.examples.push(serde_json::json!({
            "prompt": "Q: What is 2+2?",
            "completion": "A: 4"
        }));

        let chat = DatasetConverter::completion_to_chat(&dataset, Some("You are a math tutor")).unwrap();
        assert_eq!(chat.format, TrainingFormat::OpenAIChat);

        let example = &chat.examples[0];
        let messages = example["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3); // system + user + assistant
    }

    #[test]
    fn test_hyperparameters() {
        let params = Hyperparameters::default()
            .with_lora(16, 32.0, 0.05);

        assert_eq!(params.lora_rank, Some(16));
        assert_eq!(params.lora_alpha, Some(32.0));
        assert_eq!(params.lora_dropout, Some(0.05));
    }

    #[test]
    fn test_lora_config() {
        let config = LoraConfig::for_llama();
        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));

        let params = config.trainable_params(4096, 32);
        assert!(params > 0);
    }

    #[test]
    fn test_lora_manager() {
        let mut manager = LoraManager::new();

        let adapter = LoraAdapter::new(
            "my-adapter",
            "llama-7b",
            LoraConfig::for_llama(),
            "/path/to/adapter",
        );

        manager.register(adapter);

        assert!(manager.get("my-adapter").is_some());
        assert!(manager.set_active("my-adapter"));
        assert!(manager.active().is_some());

        let llama_adapters = manager.list_for_model("llama-7b");
        assert_eq!(llama_adapters.len(), 1);
    }

    #[test]
    fn test_create_finetune_request() {
        let request = CreateFineTuneRequest::new("gpt-4o-mini-2024-07-18", "file-abc123")
            .with_validation("file-def456")
            .with_suffix("custom-model")
            .with_hyperparameters(Hyperparameters {
                n_epochs: Some(5),
                ..Default::default()
            });

        assert_eq!(request.model, "gpt-4o-mini-2024-07-18");
        assert_eq!(request.validation_file, Some("file-def456".to_string()));
        assert_eq!(request.suffix, Some("custom-model".to_string()));
        assert_eq!(request.hyperparameters.n_epochs, Some(5));
    }

    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics {
            step: 100,
            epoch: 1.5,
            loss: 0.5,
            learning_rate: 1e-4,
            grad_norm: Some(1.2),
            eval_loss: Some(0.6),
            tokens_per_second: Some(1000.0),
        };

        assert_eq!(metrics.step, 100);
        assert!(metrics.eval_loss.is_some());
    }

    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStoppingCallback::new(3, 0.01);

        // Improving
        callback.on_eval(&TrainingMetrics { eval_loss: Some(1.0), ..Default::default() });
        assert!(!callback.should_stop());

        callback.on_eval(&TrainingMetrics { eval_loss: Some(0.9), ..Default::default() });
        assert!(!callback.should_stop());

        // Not improving
        callback.on_eval(&TrainingMetrics { eval_loss: Some(0.9), ..Default::default() });
        callback.on_eval(&TrainingMetrics { eval_loss: Some(0.91), ..Default::default() });
        callback.on_eval(&TrainingMetrics { eval_loss: Some(0.92), ..Default::default() });

        assert!(callback.should_stop());
    }

    #[test]
    fn test_dataset_shuffle() {
        let mut dataset = TrainingDataset::new("test", TrainingFormat::CustomJsonl);
        for i in 0..10 {
            dataset.examples.push(serde_json::json!({"id": i}));
        }

        let original: Vec<_> = dataset.examples.iter().map(|e| e["id"].as_i64().unwrap()).collect();
        dataset.shuffle(42);
        let shuffled: Vec<_> = dataset.examples.iter().map(|e| e["id"].as_i64().unwrap()).collect();

        // Should be different order
        assert_ne!(original, shuffled);
        // But same elements
        assert_eq!(dataset.len(), 10);
    }
}
