//! Real provider plugin implementations
//!
//! This module provides concrete implementations of the ProviderPlugin trait
//! for popular local LLM providers like Ollama, LM Studio, text-generation-webui,
//! and any OpenAI-compatible API.

use anyhow::{Context, Result};
use serde_json::Value;
use std::sync::mpsc::Sender;
use std::time::Duration;

use crate::config::{AiConfig, AiProvider};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;
use crate::tools::{ProviderCapabilities, ProviderPlugin, ToolCall, ToolDefinition};

// ============================================================================
// Ollama Provider
// ============================================================================

/// Ollama provider plugin
pub struct OllamaProvider {
    /// Base URL (e.g., http://localhost:11434)
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Whether the provider is currently available
    available: std::sync::atomic::AtomicBool,
}

impl OllamaProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if Ollama is running
    pub fn check_health(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available
                    .store(is_ok, std::sync::atomic::Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    fn build_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<Value> {
        let mut result = Vec::new();

        // Add system prompt
        if !system_prompt.is_empty() {
            result.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        // Add conversation messages
        for msg in messages {
            result.push(serde_json::json!({
                "role": if msg.is_user() { "user" } else { "assistant" },
                "content": msg.content
            }));
        }

        result
    }
}

impl ProviderPlugin for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: false, // Ollama has limited tool support
            vision: true,        // Some models support vision
            embeddings: true,
            json_mode: true,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = ureq::get(&url)
            .timeout(self.timeout)
            .call()
            .context("Failed to connect to Ollama")?;

        let body: Value = resp.into_json()?;
        let models = body
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let name = m.get("name")?.as_str()?.to_string();
                        let size = m
                            .get("size")
                            .and_then(|s| s.as_u64())
                            .map(|s| format!("{:.1} GB", s as f64 / 1_073_741_824.0));

                        let mut model = ModelInfo::new(name, AiProvider::Ollama);
                        if let Some(s) = size {
                            model = model.with_size(s);
                        }
                        Some(model)
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "stream": false,
            "options": {
                "temperature": config.temperature,
            }
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send request to Ollama")?;

        let body: Value = resp.into_json()?;
        let content = body
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        Ok(content.to_string())
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let url = format!("{}/api/chat", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "stream": true,
            "options": {
                "temperature": config.temperature,
            }
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send streaming request to Ollama")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Some(content) = json
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }

                // Check if done
                if json.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                    break;
                }
            }
        }

        let _ = tx.send(AiResponse::Complete(full_response));
        Ok(())
    }

    fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);
        let mut embeddings = Vec::new();

        for text in texts {
            let payload = serde_json::json!({
                "model": "nomic-embed-text",
                "input": text
            });

            let resp = ureq::post(&url).timeout(self.timeout).send_json(&payload)?;

            let body: Value = resp.into_json()?;
            if let Some(embedding) = body
                .get("embeddings")
                .and_then(|e| e.as_array())
                .and_then(|arr| arr.first())
                .and_then(|e| e.as_array())
            {
                let vec: Vec<f32> = embedding
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                embeddings.push(vec);
            }
        }

        Ok(embeddings)
    }
}

// ============================================================================
// LM Studio Provider
// ============================================================================

/// LM Studio provider plugin (OpenAI-compatible API)
pub struct LmStudioProvider {
    /// Base URL (e.g., http://localhost:1234)
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Whether the provider is currently available
    available: std::sync::atomic::AtomicBool,
}

impl LmStudioProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if LM Studio is running
    pub fn check_health(&self) -> bool {
        let url = format!("{}/v1/models", self.base_url);
        match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available
                    .store(is_ok, std::sync::atomic::Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    fn build_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<Value> {
        let mut result = Vec::new();

        if !system_prompt.is_empty() {
            result.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        for msg in messages {
            result.push(serde_json::json!({
                "role": if msg.is_user() { "user" } else { "assistant" },
                "content": msg.content
            }));
        }

        result
    }
}

impl ProviderPlugin for LmStudioProvider {
    fn name(&self) -> &str {
        "lm_studio"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            vision: false,
            embeddings: false,
            json_mode: true,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = ureq::get(&url)
            .timeout(self.timeout)
            .call()
            .context("Failed to connect to LM Studio")?;

        let body: Value = resp.into_json()?;
        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?.to_string();
                        Some(ModelInfo::new(id, AiProvider::LMStudio))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": false
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send request to LM Studio")?;

        let body: Value = resp.into_json()?;
        let content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        Ok(content.to_string())
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": true
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send streaming request to LM Studio")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;

            // SSE format: data: {...}
            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            if let Ok(json) = serde_json::from_str::<Value>(data) {
                if let Some(content) = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }
        }

        let _ = tx.send(AiResponse::Complete(full_response));
        Ok(())
    }
}

// ============================================================================
// Text Generation WebUI Provider (oobabooga)
// ============================================================================

/// Text Generation WebUI (oobabooga) provider plugin
pub struct TextGenWebUIProvider {
    /// Base URL (e.g., http://localhost:5000)
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Whether the provider is currently available
    available: std::sync::atomic::AtomicBool,
}

impl TextGenWebUIProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if text-generation-webui is running
    pub fn check_health(&self) -> bool {
        let url = format!("{}/v1/models", self.base_url);
        match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available
                    .store(is_ok, std::sync::atomic::Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    fn build_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<Value> {
        let mut result = Vec::new();

        if !system_prompt.is_empty() {
            result.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        for msg in messages {
            result.push(serde_json::json!({
                "role": if msg.is_user() { "user" } else { "assistant" },
                "content": msg.content
            }));
        }

        result
    }
}

impl ProviderPlugin for TextGenWebUIProvider {
    fn name(&self) -> &str {
        "text_gen_webui"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            vision: false,
            embeddings: false,
            json_mode: false,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = ureq::get(&url)
            .timeout(self.timeout)
            .call()
            .context("Failed to connect to text-generation-webui")?;

        let body: Value = resp.into_json()?;
        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?.to_string();
                        Some(ModelInfo::new(id, AiProvider::TextGenWebUI))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": false
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send request to text-generation-webui")?;

        let body: Value = resp.into_json()?;
        let content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        Ok(content.to_string())
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": true
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send streaming request to text-generation-webui")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            if let Ok(json) = serde_json::from_str::<Value>(data) {
                if let Some(content) = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
            }
        }

        let _ = tx.send(AiResponse::Complete(full_response));
        Ok(())
    }
}

// ============================================================================
// Kobold.cpp Provider
// ============================================================================

/// Kobold.cpp provider plugin
pub struct KoboldCppProvider {
    /// Base URL (e.g., http://localhost:5001)
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Whether the provider is currently available
    available: std::sync::atomic::AtomicBool,
}

impl KoboldCppProvider {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if Kobold.cpp is running
    pub fn check_health(&self) -> bool {
        let url = format!("{}/api/v1/model", self.base_url);
        match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available
                    .store(is_ok, std::sync::atomic::Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    fn format_prompt(&self, messages: &[ChatMessage], system_prompt: &str) -> String {
        let mut prompt = String::new();

        if !system_prompt.is_empty() {
            prompt.push_str(&format!("System: {}\n\n", system_prompt));
        }

        for msg in messages {
            let role = if msg.is_user() { "User" } else { "Assistant" };
            prompt.push_str(&format!("{}: {}\n\n", role, msg.content));
        }

        prompt.push_str("Assistant:");
        prompt
    }
}

impl ProviderPlugin for KoboldCppProvider {
    fn name(&self) -> &str {
        "kobold_cpp"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            vision: false,
            embeddings: false,
            json_mode: false,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/v1/model", self.base_url);
        let resp = ureq::get(&url)
            .timeout(self.timeout)
            .call()
            .context("Failed to connect to Kobold.cpp")?;

        let body: Value = resp.into_json()?;
        let model_name = body
            .get("result")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");

        Ok(vec![ModelInfo::new(model_name, AiProvider::KoboldCpp)])
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/api/v1/generate", self.base_url);
        let prompt = self.format_prompt(messages, system_prompt);

        let payload = serde_json::json!({
            "prompt": prompt,
            "max_length": 2048,
            "temperature": config.temperature,
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send request to Kobold.cpp")?;

        let body: Value = resp.into_json()?;
        let content = body
            .get("results")
            .and_then(|r| r.as_array())
            .and_then(|arr| arr.first())
            .and_then(|r| r.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        Ok(content.trim().to_string())
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let url = format!("{}/api/extra/generate/stream", self.base_url);
        let prompt = self.format_prompt(messages, system_prompt);

        let payload = serde_json::json!({
            "prompt": prompt,
            "max_length": 2048,
            "temperature": config.temperature,
        });

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send streaming request to Kobold.cpp")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if let Ok(json) = serde_json::from_str::<Value>(data) {
                if let Some(token) = json.get("token").and_then(|t| t.as_str()) {
                    full_response.push_str(token);
                    let _ = tx.send(AiResponse::Chunk(token.to_string()));
                }
            }
        }

        let _ = tx.send(AiResponse::Complete(full_response));
        Ok(())
    }
}

// ============================================================================
// Generic OpenAI-Compatible Provider
// ============================================================================

/// Generic OpenAI-compatible provider plugin
pub struct OpenAICompatibleProvider {
    /// Provider name
    provider_name: String,
    /// Base URL
    pub base_url: String,
    /// API key (optional for local providers)
    pub api_key: Option<String>,
    /// Request timeout
    pub timeout: Duration,
    /// Whether the provider is currently available
    available: std::sync::atomic::AtomicBool,
}

impl OpenAICompatibleProvider {
    pub fn new(name: &str, base_url: &str) -> Self {
        Self {
            provider_name: name.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if the provider is running
    pub fn check_health(&self) -> bool {
        let url = format!("{}/v1/models", self.base_url);
        let mut req = ureq::get(&url).timeout(Duration::from_secs(5));

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        match req.call() {
            Ok(resp) => {
                let is_ok = resp.status() == 200;
                self.available
                    .store(is_ok, std::sync::atomic::Ordering::SeqCst);
                is_ok
            }
            Err(_) => {
                self.available
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                false
            }
        }
    }

    fn build_messages(&self, messages: &[ChatMessage], system_prompt: &str) -> Vec<Value> {
        let mut result = Vec::new();

        if !system_prompt.is_empty() {
            result.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        for msg in messages {
            result.push(serde_json::json!({
                "role": if msg.is_user() { "user" } else { "assistant" },
                "content": msg.content
            }));
        }

        result
    }
}

impl ProviderPlugin for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            vision: false,
            embeddings: true,
            json_mode: true,
            system_prompt: true,
        }
    }

    fn is_available(&self) -> bool {
        self.check_health()
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let mut req = ureq::get(&url).timeout(self.timeout);

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req
            .call()
            .context("Failed to connect to OpenAI-compatible API")?;

        let body: Value = resp.into_json()?;
        let base_url = self.base_url.clone();
        let models = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?.to_string();
                        Some(ModelInfo::new(
                            id,
                            AiProvider::OpenAICompatible {
                                base_url: base_url.clone(),
                            },
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn generate(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": false
        });

        let mut req = ureq::post(&url).timeout(self.timeout);

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req.send_json(&payload).context("Failed to send request")?;

        let body: Value = resp.into_json()?;
        let content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        Ok(content.to_string())
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": true
        });

        let mut req = ureq::post(&url).timeout(self.timeout);

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req
            .send_json(&payload)
            .context("Failed to send streaming request")?;

        let reader = std::io::BufReader::new(resp.into_reader());
        let mut full_response = String::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            if let Ok(json) = serde_json::from_str::<Value>(data) {
                if let Some(content) = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    full_response.push_str(content);
                    let _ = tx.send(AiResponse::Chunk(content.to_string()));
                }
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
        let url = format!("{}/v1/chat/completions", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        let tool_schemas: Vec<Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": t.to_openai_function()
                })
            })
            .collect();

        let mut payload = serde_json::json!({
            "model": config.selected_model,
            "messages": msgs,
            "temperature": config.temperature,
            "stream": false
        });

        if !tool_schemas.is_empty() {
            payload["tools"] = Value::Array(tool_schemas);
        }

        let mut req = ureq::post(&url).timeout(self.timeout);

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req
            .send_json(&payload)
            .context("Failed to send request with tools")?;

        let body: Value = resp.into_json()?;

        // Extract content
        let content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Extract tool calls
        let tool_calls = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .map(|m| crate::tools::ToolRegistry::parse_tool_calls(m))
            .unwrap_or_default();

        Ok((content, tool_calls))
    }

    fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let mut req = ureq::post(&url).timeout(self.timeout);

        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let payload = serde_json::json!({
            "model": "text-embedding-ada-002",
            "input": texts
        });

        let resp = req.send_json(&payload)?;
        let body: Value = resp.into_json()?;

        let embeddings = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        item.get("embedding").and_then(|e| e.as_array()).map(|emb| {
                            emb.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(embeddings)
    }
}

// ============================================================================
// Auto-Discovery
// ============================================================================

/// Provider discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Check Ollama at default port
    pub check_ollama: bool,
    /// Ollama URL
    pub ollama_url: String,
    /// Check LM Studio at default port
    pub check_lm_studio: bool,
    /// LM Studio URL
    pub lm_studio_url: String,
    /// Check text-generation-webui at default port
    pub check_text_gen_webui: bool,
    /// text-generation-webui URL
    pub text_gen_webui_url: String,
    /// Check Kobold.cpp at default port
    pub check_kobold_cpp: bool,
    /// Kobold.cpp URL
    pub kobold_cpp_url: String,
    /// Timeout for health checks
    pub timeout: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            check_ollama: true,
            ollama_url: "http://localhost:11434".to_string(),
            check_lm_studio: true,
            lm_studio_url: "http://localhost:1234".to_string(),
            check_text_gen_webui: true,
            text_gen_webui_url: "http://localhost:5000".to_string(),
            check_kobold_cpp: true,
            kobold_cpp_url: "http://localhost:5001".to_string(),
            timeout: Duration::from_secs(5),
        }
    }
}

/// Discover available LLM providers
pub fn discover_providers(config: &DiscoveryConfig) -> Vec<Box<dyn ProviderPlugin>> {
    let mut providers: Vec<Box<dyn ProviderPlugin>> = Vec::new();

    // Check Ollama
    if config.check_ollama {
        let provider = OllamaProvider::new(&config.ollama_url).with_timeout(config.timeout);
        if provider.check_health() {
            providers.push(Box::new(provider));
        }
    }

    // Check LM Studio
    if config.check_lm_studio {
        let provider = LmStudioProvider::new(&config.lm_studio_url).with_timeout(config.timeout);
        if provider.check_health() {
            providers.push(Box::new(provider));
        }
    }

    // Check text-generation-webui
    if config.check_text_gen_webui {
        let provider =
            TextGenWebUIProvider::new(&config.text_gen_webui_url).with_timeout(config.timeout);
        if provider.check_health() {
            providers.push(Box::new(provider));
        }
    }

    // Check Kobold.cpp
    if config.check_kobold_cpp {
        let provider = KoboldCppProvider::new(&config.kobold_cpp_url).with_timeout(config.timeout);
        if provider.check_health() {
            providers.push(Box::new(provider));
        }
    }

    providers
}

/// Create a registry with all discovered providers
pub fn create_registry_with_discovery(config: &DiscoveryConfig) -> crate::tools::ProviderRegistry {
    let mut registry = crate::tools::ProviderRegistry::new();

    for provider in discover_providers(config) {
        registry.register(provider);
    }

    registry
}

// ============================================================================
// Cloud Provider Presets (OpenAI-compatible endpoints)
// ============================================================================

impl OpenAICompatibleProvider {
    /// Create a Groq provider (fast inference, OpenAI-compatible).
    /// API key from `GROQ_API_KEY` env var or provided directly.
    pub fn groq() -> Self {
        let api_key = std::env::var("GROQ_API_KEY").ok();
        let mut provider = Self::new("groq", "https://api.groq.com/openai");
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }

    /// Create a Together AI provider (hosted open-source models).
    /// API key from `TOGETHER_API_KEY` env var.
    pub fn together() -> Self {
        let api_key = std::env::var("TOGETHER_API_KEY").ok();
        let mut provider = Self::new("together", "https://api.together.xyz");
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }

    /// Create a Fireworks AI provider (optimized inference).
    /// API key from `FIREWORKS_API_KEY` env var.
    pub fn fireworks() -> Self {
        let api_key = std::env::var("FIREWORKS_API_KEY").ok();
        let mut provider = Self::new("fireworks", "https://api.fireworks.ai/inference");
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }

    /// Create a DeepSeek provider (strong coding models).
    /// API key from `DEEPSEEK_API_KEY` env var.
    pub fn deepseek() -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok();
        let mut provider = Self::new("deepseek", "https://api.deepseek.com");
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }

    /// Create a vLLM provider (local high-performance inference server).
    /// No auth required by default.
    pub fn vllm(url: &str) -> Self {
        Self::new("vllm", url)
    }

    /// Create a Mistral AI provider.
    /// API key from `MISTRAL_API_KEY` env var.
    pub fn mistral() -> Self {
        let api_key = std::env::var("MISTRAL_API_KEY").ok();
        let mut provider = Self::new("mistral", "https://api.mistral.ai");
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }

    /// Create a provider from environment variable name and base URL.
    pub fn from_env(name: &str, base_url: &str, env_var: &str) -> Self {
        let api_key = std::env::var(env_var).ok();
        let mut provider = Self::new(name, base_url);
        if let Some(key) = api_key {
            provider.api_key = Some(key);
        }
        provider
    }
}

/// HuggingFace Inference API provider.
///
/// Uses the HuggingFace Inference API for text generation and embeddings.
/// API key from `HF_TOKEN` env var.
pub struct HuggingFaceInferenceProvider {
    api_key: Option<String>,
    model: String,
    timeout: Duration,
    available: std::sync::atomic::AtomicBool,
}

impl HuggingFaceInferenceProvider {
    pub fn new(model: &str) -> Self {
        Self {
            api_key: std::env::var("HF_TOKEN").ok(),
            model: model.to_string(),
            timeout: Duration::from_secs(120),
            available: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    fn base_url(&self) -> String {
        format!("https://api-inference.huggingface.co/models/{}", self.model)
    }
}

impl ProviderPlugin for HuggingFaceInferenceProvider {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: false,
            tool_calling: false,
            vision: false,
            embeddings: true,
            json_mode: false,
            system_prompt: false,
        }
    }

    fn is_available(&self) -> bool {
        self.available.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&limit=20";
        let mut req = ureq::get(url).timeout(Duration::from_secs(10));
        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req.call()?;
        let body: Value = resp.into_json()?;

        let models = body
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_str()?;
                        let downloads = m.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0);
                        Some(ModelInfo {
                            name: id.to_string(),
                            provider: AiProvider::OpenAICompatible {
                                base_url: "https://api-inference.huggingface.co".to_string(),
                            },
                            size: Some(format!("{} downloads", downloads)),
                            modified_at: None,
                            capabilities: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn generate(
        &self,
        _config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
    ) -> Result<String> {
        let url = self.base_url();

        // Build prompt from messages
        let mut prompt = String::new();
        if !system_prompt.is_empty() {
            prompt.push_str(system_prompt);
            prompt.push_str("\n\n");
        }
        for msg in messages {
            let role = if msg.is_user() { "User" } else { "Assistant" };
            prompt.push_str(&format!("{}: {}\n", role, msg.content));
        }
        prompt.push_str("Assistant: ");

        let payload = serde_json::json!({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "return_full_text": false
            }
        });

        let mut req = ureq::post(&url).timeout(self.timeout);
        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req.send_json(&payload)?;
        let body: Value = resp.into_json()?;

        // HF returns array of generated texts
        let text = body
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("generated_text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();

        self.available
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(text)
    }

    fn generate_streaming(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tx: &std::sync::mpsc::Sender<AiResponse>,
    ) -> Result<()> {
        // HF Inference API doesn't support streaming for most models
        let response = self.generate(config, messages, system_prompt)?;
        let _ = tx.send(AiResponse::Complete(response));
        Ok(())
    }

    fn generate_with_tools(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        _tools: &[ToolDefinition],
    ) -> Result<(String, Vec<ToolCall>)> {
        let response = self.generate(config, messages, system_prompt)?;
        Ok((response, Vec::new()))
    }

    fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        );

        let payload = serde_json::json!({
            "inputs": texts
        });

        let mut req = ureq::post(&url).timeout(self.timeout);
        if let Some(ref key) = self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }

        let resp = req.send_json(&payload)?;
        let body: Value = resp.into_json()?;

        let embeddings = body
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|emb| {
                        emb.as_array().map(|e| {
                            e.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

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
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434");
        assert_eq!(provider.name(), "ollama");
        assert!(provider.capabilities().streaming);
    }

    #[test]
    fn test_lm_studio_provider_creation() {
        let provider = LmStudioProvider::new("http://localhost:1234");
        assert_eq!(provider.name(), "lm_studio");
        assert!(provider.capabilities().streaming);
    }

    #[test]
    fn test_openai_compatible_provider() {
        let provider = OpenAICompatibleProvider::new("custom", "http://localhost:8080")
            .with_api_key("test-key");
        assert_eq!(provider.name(), "custom");
        assert!(provider.capabilities().tool_calling);
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert!(config.check_ollama);
        assert!(config.check_lm_studio);
        assert_eq!(config.ollama_url, "http://localhost:11434");
    }

    #[test]
    fn test_groq_preset() {
        let provider = OpenAICompatibleProvider::groq();
        assert_eq!(provider.name(), "groq");
        assert_eq!(provider.base_url, "https://api.groq.com/openai");
    }

    #[test]
    fn test_together_preset() {
        let provider = OpenAICompatibleProvider::together();
        assert_eq!(provider.name(), "together");
        assert_eq!(provider.base_url, "https://api.together.xyz");
    }

    #[test]
    fn test_fireworks_preset() {
        let provider = OpenAICompatibleProvider::fireworks();
        assert_eq!(provider.name(), "fireworks");
        assert!(provider.base_url.contains("fireworks.ai"));
    }

    #[test]
    fn test_deepseek_preset() {
        let provider = OpenAICompatibleProvider::deepseek();
        assert_eq!(provider.name(), "deepseek");
        assert!(provider.base_url.contains("deepseek.com"));
    }

    #[test]
    fn test_vllm_preset() {
        let provider = OpenAICompatibleProvider::vllm("http://gpu-server:8000");
        assert_eq!(provider.name(), "vllm");
        assert_eq!(provider.base_url, "http://gpu-server:8000");
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn test_mistral_preset() {
        let provider = OpenAICompatibleProvider::mistral();
        assert_eq!(provider.name(), "mistral");
        assert!(provider.base_url.contains("mistral.ai"));
    }

    #[test]
    fn test_from_env_preset() {
        let provider = OpenAICompatibleProvider::from_env(
            "custom_cloud",
            "https://my-api.example.com",
            "MY_API_KEY_THAT_DOES_NOT_EXIST",
        );
        assert_eq!(provider.name(), "custom_cloud");
        assert_eq!(provider.base_url, "https://my-api.example.com");
        // Env var doesn't exist, so no key
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn test_huggingface_provider_creation() {
        let provider = HuggingFaceInferenceProvider::new("gpt2")
            .with_api_key("test-key")
            .with_model("meta-llama/Llama-2-7b");
        assert_eq!(provider.name(), "huggingface");
        assert!(provider.capabilities().embeddings);
        assert!(!provider.capabilities().streaming);
        assert!(!provider.capabilities().tool_calling);
    }
}
