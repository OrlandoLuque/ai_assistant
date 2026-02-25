//! Real provider plugin implementations
//!
//! This module provides concrete implementations of the ProviderPlugin trait
//! for popular local LLM providers like Ollama, LM Studio, text-generation-webui,
//! and any OpenAI-compatible API.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
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
            tool_calling: true, // Ollama supports tool calling via /api/chat
            vision: true,       // Some models support vision
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

    fn generate_with_tools(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<(String, Vec<ToolCall>)> {
        let url = format!("{}/api/chat", self.base_url);
        let msgs = self.build_messages(messages, system_prompt);

        // Build tool schemas in OpenAI-compatible format
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
            "stream": false,
            "options": {
                "temperature": config.temperature,
            }
        });

        if !tool_schemas.is_empty() {
            payload["tools"] = Value::Array(tool_schemas);
        }

        let resp = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&payload)
            .context("Failed to send tool request to Ollama")?;

        let body: Value = resp.into_json()?;

        // Extract content from message (Ollama puts it directly on message, not choices[0])
        let content = body
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Parse tool calls from Ollama response format
        // Ollama returns tool_calls directly on message (no choices wrapper)
        let tool_calls = if let Some(tc_array) = body
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
        {
            tc_array
                .iter()
                .enumerate()
                .filter_map(|(i, tc)| {
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    // Ollama returns arguments as an object directly, not a JSON string
                    let arguments: HashMap<String, Value> =
                        if let Some(args_obj) = function.get("arguments") {
                            if let Some(s) = args_obj.as_str() {
                                serde_json::from_str(s).unwrap_or_default()
                            } else {
                                serde_json::from_value(args_obj.clone()).unwrap_or_default()
                            }
                        } else {
                            HashMap::new()
                        };
                    Some(ToolCall {
                        name,
                        arguments,
                        id: format!("ollama_call_{}", i),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok((content, tool_calls))
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
// Prompt-Based Tool Fallback
// ============================================================================

/// Provides tool calling via prompt engineering for providers without native support.
///
/// This helper injects tool definitions into the system prompt and parses
/// JSON-formatted tool calls from the LLM's text output. This allows any
/// text-completion provider to participate in tool-calling workflows.
pub struct PromptToolFallback;

impl PromptToolFallback {
    /// Builds an augmented system prompt with tool definitions injected.
    ///
    /// If `tools` is empty the original prompt is returned unchanged.
    pub fn build_tool_prompt(system_prompt: &str, tools: &[ToolDefinition]) -> String {
        if tools.is_empty() {
            return system_prompt.to_string();
        }

        let mut prompt = system_prompt.to_string();
        prompt.push_str("\n\nAvailable tools:\n");

        for tool in tools {
            let schema = tool.to_openai_function();
            if let Ok(pretty) = serde_json::to_string_pretty(&schema) {
                prompt.push_str(&pretty);
                prompt.push('\n');
            }
        }

        prompt.push_str(
            "\nTo call a tool, respond with a JSON block in this exact format:\n\
             ```json\n\
             {\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}}\n\
             ```\n\
             You may include multiple tool_call blocks if needed. \
             If you do not need to call a tool, respond normally.",
        );

        prompt
    }

    /// Parses tool calls from LLM text output.
    ///
    /// Looks for JSON objects containing a `"tool_call"` key. Accepts JSON
    /// both inline and inside fenced code blocks (````json ... ````).
    pub fn parse_tool_calls_from_text(response: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let mut search = response;

        // Iterate through potential JSON objects in the response
        while let Some(start) = search.find('{') {
            if let Some(json_str) = Self::extract_json_object(&search[start..]) {
                if let Ok(parsed) = serde_json::from_str::<Value>(&json_str) {
                    if let Some(tc) = parsed.get("tool_call") {
                        if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                            let arguments: HashMap<String, Value> = tc
                                .get("arguments")
                                .and_then(|a| serde_json::from_value(a.clone()).ok())
                                .unwrap_or_default();
                            calls.push(ToolCall {
                                name: name.to_string(),
                                arguments,
                                id: format!("fallback_call_{}", calls.len()),
                            });
                        }
                    }
                }
                // Advance past this JSON object
                let consumed = start + json_str.len();
                if consumed < search.len() {
                    search = &search[consumed..];
                } else {
                    break;
                }
            } else {
                // No valid JSON object found starting here, advance past this brace
                if start + 1 < search.len() {
                    search = &search[start + 1..];
                } else {
                    break;
                }
            }
        }

        calls
    }

    /// Extract a balanced JSON object starting from the first `{`.
    fn extract_json_object(s: &str) -> Option<String> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || bytes[0] != b'{' {
            return None;
        }

        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, &b) in bytes.iter().enumerate() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if b == b'\\' && in_string {
                escape_next = true;
                continue;
            }
            if b == b'"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    return Some(s[..=i].to_string());
                }
            }
        }

        None
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
            tool_calling: true, // Via prompt-based fallback
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

    fn generate_with_tools(
        &self,
        config: &AiConfig,
        messages: &[ChatMessage],
        system_prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<(String, Vec<ToolCall>)> {
        let augmented_prompt = PromptToolFallback::build_tool_prompt(system_prompt, tools);
        let response = self.generate(config, messages, &augmented_prompt)?;
        let tool_calls = PromptToolFallback::parse_tool_calls_from_text(&response);
        Ok((response, tool_calls))
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

    /// Create a Perplexity API provider.
    ///
    /// Uses PERPLEXITY_API_KEY from environment.
    pub fn perplexity() -> Self {
        Self::from_env("Perplexity", "https://api.perplexity.ai", "PERPLEXITY_API_KEY")
    }

    /// Create an OpenRouter API provider.
    ///
    /// Uses OPENROUTER_API_KEY from environment.
    pub fn openrouter() -> Self {
        Self::from_env("OpenRouter", "https://openrouter.ai/api", "OPENROUTER_API_KEY")
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

    // ====================================================================
    // PromptToolFallback tests
    // ====================================================================

    #[test]
    fn test_prompt_tool_fallback_build_prompt() {
        let tool = ToolDefinition::new("get_weather", "Get the current weather");
        let result = PromptToolFallback::build_tool_prompt("You are a helpful assistant.", &[tool]);
        assert!(result.starts_with("You are a helpful assistant."));
        assert!(result.contains("Available tools:"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("tool_call"));
    }

    #[test]
    fn test_prompt_tool_fallback_build_prompt_no_tools() {
        let result =
            PromptToolFallback::build_tool_prompt("You are a helpful assistant.", &[]);
        assert_eq!(result, "You are a helpful assistant.");
    }

    #[test]
    fn test_prompt_tool_fallback_parse_tool_call() {
        let response = r#"I'll check the weather for you.
{"tool_call": {"name": "get_weather", "arguments": {"city": "London"}}}
"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments.get("city").and_then(|v| v.as_str()),
            Some("London")
        );
        assert_eq!(calls[0].id, "fallback_call_0");
    }

    #[test]
    fn test_prompt_tool_fallback_parse_tool_call_in_code_block() {
        let response = r#"Let me look that up.
```json
{"tool_call": {"name": "search", "arguments": {"query": "rust programming"}}}
```
"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(
            calls[0].arguments.get("query").and_then(|v| v.as_str()),
            Some("rust programming")
        );
    }

    #[test]
    fn test_prompt_tool_fallback_parse_no_tool_calls() {
        let response = "The weather in London is currently sunny with a high of 22C.";
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_prompt_tool_fallback_parse_multiple_tool_calls() {
        let response = r#"I need to check two things.
{"tool_call": {"name": "get_weather", "arguments": {"city": "London"}}}
And also:
{"tool_call": {"name": "get_time", "arguments": {"timezone": "UTC"}}}
"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[0].id, "fallback_call_0");
        assert_eq!(calls[1].id, "fallback_call_1");
    }

    // ====================================================================
    // Ollama tool calling tests
    // ====================================================================

    #[test]
    fn test_ollama_tool_schema_building() {
        let tool = ToolDefinition::new("get_weather", "Get the current weather");
        let schema = tool.to_openai_function();
        assert_eq!(schema.get("name").unwrap().as_str().unwrap(), "get_weather");
        assert_eq!(
            schema.get("description").unwrap().as_str().unwrap(),
            "Get the current weather"
        );
        assert!(schema.get("parameters").is_some());
        assert_eq!(
            schema
                .get("parameters")
                .unwrap()
                .get("type")
                .unwrap()
                .as_str()
                .unwrap(),
            "object"
        );
    }

    #[test]
    fn test_ollama_tool_call_response_parsing() {
        // Simulate Ollama's response format (tool_calls directly on message, not choices)
        let response_json: Value = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris", "units": "celsius"}
                        }
                    }
                ]
            },
            "done": true
        });

        // Parse the same way the implementation does
        let tool_calls: Vec<ToolCall> = if let Some(tc_array) = response_json
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
        {
            tc_array
                .iter()
                .enumerate()
                .filter_map(|(i, tc)| {
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments: HashMap<String, Value> =
                        if let Some(args_obj) = function.get("arguments") {
                            if let Some(s) = args_obj.as_str() {
                                serde_json::from_str(s).unwrap_or_default()
                            } else {
                                serde_json::from_value(args_obj.clone()).unwrap_or_default()
                            }
                        } else {
                            HashMap::new()
                        };
                    Some(ToolCall {
                        name,
                        arguments,
                        id: format!("ollama_call_{}", i),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(
            tool_calls[0].arguments.get("city").and_then(|v| v.as_str()),
            Some("Paris")
        );
        assert_eq!(
            tool_calls[0]
                .arguments
                .get("units")
                .and_then(|v| v.as_str()),
            Some("celsius")
        );
        assert_eq!(tool_calls[0].id, "ollama_call_0");
    }

    // ====================================================================
    // Capability flag tests
    // ====================================================================

    #[test]
    fn test_kobold_capabilities_updated() {
        let provider = KoboldCppProvider::new("http://localhost:5001");
        assert!(
            provider.capabilities().tool_calling,
            "KoboldCpp should report tool_calling: true (via prompt fallback)"
        );
    }

    #[test]
    fn test_ollama_capabilities_updated() {
        let provider = OllamaProvider::new("http://localhost:11434");
        assert!(
            provider.capabilities().tool_calling,
            "Ollama should report tool_calling: true"
        );
    }

    // ====================================================================
    // NEW: URL trailing-slash trimming
    // ====================================================================

    #[test]
    fn test_ollama_url_trailing_slash_trimmed() {
        let provider = OllamaProvider::new("http://localhost:11434/");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_lm_studio_url_trailing_slash_trimmed() {
        let provider = LmStudioProvider::new("http://localhost:1234/");
        assert_eq!(provider.base_url, "http://localhost:1234");
    }

    #[test]
    fn test_text_gen_webui_url_trailing_slash_trimmed() {
        let provider = TextGenWebUIProvider::new("http://localhost:5000/");
        assert_eq!(provider.base_url, "http://localhost:5000");
    }

    #[test]
    fn test_kobold_url_trailing_slash_trimmed() {
        let provider = KoboldCppProvider::new("http://localhost:5001/");
        assert_eq!(provider.base_url, "http://localhost:5001");
    }

    #[test]
    fn test_openai_compatible_url_trailing_slash_trimmed() {
        let provider = OpenAICompatibleProvider::new("test", "http://example.com/v1/");
        assert_eq!(provider.base_url, "http://example.com/v1");
    }

    // ====================================================================
    // NEW: with_timeout builder chains
    // ====================================================================

    #[test]
    fn test_ollama_with_timeout() {
        let provider =
            OllamaProvider::new("http://localhost:11434").with_timeout(Duration::from_secs(30));
        assert_eq!(provider.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_lm_studio_with_timeout() {
        let provider =
            LmStudioProvider::new("http://localhost:1234").with_timeout(Duration::from_secs(60));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_text_gen_webui_with_timeout() {
        let provider =
            TextGenWebUIProvider::new("http://localhost:5000").with_timeout(Duration::from_secs(45));
        assert_eq!(provider.timeout, Duration::from_secs(45));
    }

    #[test]
    fn test_kobold_with_timeout() {
        let provider =
            KoboldCppProvider::new("http://localhost:5001").with_timeout(Duration::from_secs(90));
        assert_eq!(provider.timeout, Duration::from_secs(90));
    }

    #[test]
    fn test_openai_compatible_with_timeout() {
        let provider = OpenAICompatibleProvider::new("test", "http://example.com")
            .with_timeout(Duration::from_secs(15));
        assert_eq!(provider.timeout, Duration::from_secs(15));
    }

    // ====================================================================
    // NEW: Default timeout values (should be 120s)
    // ====================================================================

    #[test]
    fn test_default_timeout_all_providers() {
        let expected = Duration::from_secs(120);
        assert_eq!(OllamaProvider::new("http://x").timeout, expected);
        assert_eq!(LmStudioProvider::new("http://x").timeout, expected);
        assert_eq!(TextGenWebUIProvider::new("http://x").timeout, expected);
        assert_eq!(KoboldCppProvider::new("http://x").timeout, expected);
        assert_eq!(
            OpenAICompatibleProvider::new("t", "http://x").timeout,
            expected
        );
    }

    // ====================================================================
    // NEW: TextGenWebUIProvider creation + ProviderPlugin impl
    // ====================================================================

    #[test]
    fn test_text_gen_webui_provider_creation() {
        let provider = TextGenWebUIProvider::new("http://localhost:5000");
        assert_eq!(provider.name(), "text_gen_webui");
        assert_eq!(provider.base_url, "http://localhost:5000");
    }

    #[test]
    fn test_text_gen_webui_capabilities() {
        let caps = TextGenWebUIProvider::new("http://localhost:5000").capabilities();
        assert!(caps.streaming);
        assert!(!caps.tool_calling);
        assert!(!caps.vision);
        assert!(!caps.embeddings);
        assert!(!caps.json_mode);
        assert!(caps.system_prompt);
    }

    // ====================================================================
    // NEW: KoboldCppProvider creation
    // ====================================================================

    #[test]
    fn test_kobold_provider_creation() {
        let provider = KoboldCppProvider::new("http://localhost:5001");
        assert_eq!(provider.name(), "kobold_cpp");
        assert_eq!(provider.base_url, "http://localhost:5001");
    }

    // ====================================================================
    // NEW: Full capability flag coverage per provider
    // ====================================================================

    #[test]
    fn test_ollama_all_capabilities() {
        let caps = OllamaProvider::new("http://localhost:11434").capabilities();
        assert!(caps.streaming);
        assert!(caps.tool_calling);
        assert!(caps.vision);
        assert!(caps.embeddings);
        assert!(caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_lm_studio_all_capabilities() {
        let caps = LmStudioProvider::new("http://localhost:1234").capabilities();
        assert!(caps.streaming);
        assert!(!caps.tool_calling);
        assert!(!caps.vision);
        assert!(!caps.embeddings);
        assert!(caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_kobold_all_capabilities() {
        let caps = KoboldCppProvider::new("http://localhost:5001").capabilities();
        assert!(caps.streaming);
        assert!(caps.tool_calling);
        assert!(!caps.vision);
        assert!(!caps.embeddings);
        assert!(!caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_openai_compatible_all_capabilities() {
        let caps = OpenAICompatibleProvider::new("test", "http://x").capabilities();
        assert!(caps.streaming);
        assert!(caps.tool_calling);
        assert!(!caps.vision);
        assert!(caps.embeddings);
        assert!(caps.json_mode);
        assert!(caps.system_prompt);
    }

    #[test]
    fn test_huggingface_all_capabilities() {
        let caps = HuggingFaceInferenceProvider::new("gpt2").capabilities();
        assert!(!caps.streaming);
        assert!(!caps.tool_calling);
        assert!(!caps.vision);
        assert!(caps.embeddings);
        assert!(!caps.json_mode);
        assert!(!caps.system_prompt);
    }

    // ====================================================================
    // NEW: HuggingFace base_url construction
    // ====================================================================

    #[test]
    fn test_huggingface_base_url() {
        let provider = HuggingFaceInferenceProvider::new("gpt2");
        assert_eq!(
            provider.base_url(),
            "https://api-inference.huggingface.co/models/gpt2"
        );
    }

    #[test]
    fn test_huggingface_base_url_with_model_change() {
        let provider = HuggingFaceInferenceProvider::new("gpt2")
            .with_model("meta-llama/Llama-2-7b");
        assert_eq!(
            provider.base_url(),
            "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b"
        );
    }

    // ====================================================================
    // NEW: DiscoveryConfig full field coverage
    // ====================================================================

    #[test]
    fn test_discovery_config_all_defaults() {
        let config = DiscoveryConfig::default();
        assert!(config.check_ollama);
        assert!(config.check_lm_studio);
        assert!(config.check_text_gen_webui);
        assert!(config.check_kobold_cpp);
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.lm_studio_url, "http://localhost:1234");
        assert_eq!(config.text_gen_webui_url, "http://localhost:5000");
        assert_eq!(config.kobold_cpp_url, "http://localhost:5001");
        assert_eq!(config.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_discovery_config_clone() {
        let config = DiscoveryConfig::default();
        let cloned = config.clone();
        assert_eq!(config.ollama_url, cloned.ollama_url);
        assert_eq!(config.check_ollama, cloned.check_ollama);
        assert_eq!(config.timeout, cloned.timeout);
    }

    #[test]
    fn test_discovery_config_debug() {
        let config = DiscoveryConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("DiscoveryConfig"));
        assert!(debug_str.contains("11434"));
    }

    // ====================================================================
    // NEW: build_messages helper (Ollama, LmStudio, TextGenWebUI, OpenAI)
    // ====================================================================

    #[test]
    fn test_ollama_build_messages_with_system_prompt() {
        let provider = OllamaProvider::new("http://localhost:11434");
        let msgs = vec![ChatMessage::user("Hello")];
        let result = provider.build_messages(&msgs, "You are helpful.");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[0]["content"], "You are helpful.");
        assert_eq!(result[1]["role"], "user");
        assert_eq!(result[1]["content"], "Hello");
    }

    #[test]
    fn test_ollama_build_messages_empty_system_prompt() {
        let provider = OllamaProvider::new("http://localhost:11434");
        let msgs = vec![ChatMessage::user("Hi")];
        let result = provider.build_messages(&msgs, "");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
    }

    #[test]
    fn test_ollama_build_messages_mixed_roles() {
        let provider = OllamaProvider::new("http://localhost:11434");
        let msgs = vec![
            ChatMessage::user("Q1"),
            ChatMessage::assistant("A1"),
            ChatMessage::user("Q2"),
        ];
        let result = provider.build_messages(&msgs, "Sys");
        assert_eq!(result.len(), 4); // system + 3 messages
        assert_eq!(result[1]["role"], "user");
        assert_eq!(result[2]["role"], "assistant");
        assert_eq!(result[3]["role"], "user");
    }

    #[test]
    fn test_ollama_build_messages_no_messages() {
        let provider = OllamaProvider::new("http://localhost:11434");
        let result = provider.build_messages(&[], "System");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "system");
    }

    #[test]
    fn test_lm_studio_build_messages_with_system_prompt() {
        let provider = LmStudioProvider::new("http://localhost:1234");
        let msgs = vec![ChatMessage::user("Test")];
        let result = provider.build_messages(&msgs, "Be concise.");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[0]["content"], "Be concise.");
    }

    #[test]
    fn test_openai_compatible_build_messages() {
        let provider = OpenAICompatibleProvider::new("test", "http://x");
        let msgs = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];
        let result = provider.build_messages(&msgs, "System prompt");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["role"], "system");
        assert_eq!(result[1]["role"], "user");
        assert_eq!(result[2]["role"], "assistant");
    }

    // ====================================================================
    // NEW: KoboldCpp format_prompt
    // ====================================================================

    #[test]
    fn test_kobold_format_prompt_with_system() {
        let provider = KoboldCppProvider::new("http://localhost:5001");
        let msgs = vec![ChatMessage::user("Hello there")];
        let result = provider.format_prompt(&msgs, "Be nice.");
        assert!(result.starts_with("System: Be nice.\n\n"));
        assert!(result.contains("User: Hello there\n\n"));
        assert!(result.ends_with("Assistant:"));
    }

    #[test]
    fn test_kobold_format_prompt_no_system() {
        let provider = KoboldCppProvider::new("http://localhost:5001");
        let msgs = vec![ChatMessage::user("Hi")];
        let result = provider.format_prompt(&msgs, "");
        assert!(!result.contains("System:"));
        assert!(result.starts_with("User: Hi\n\n"));
        assert!(result.ends_with("Assistant:"));
    }

    #[test]
    fn test_kobold_format_prompt_multi_turn() {
        let provider = KoboldCppProvider::new("http://localhost:5001");
        let msgs = vec![
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
            ChatMessage::user("And 3+3?"),
        ];
        let result = provider.format_prompt(&msgs, "");
        assert!(result.contains("User: What is 2+2?"));
        assert!(result.contains("Assistant: 4"));
        assert!(result.contains("User: And 3+3?"));
        assert!(result.ends_with("Assistant:"));
    }

    // ====================================================================
    // NEW: PromptToolFallback edge cases
    // ====================================================================

    #[test]
    fn test_prompt_tool_fallback_parse_malformed_json() {
        let response = r#"Here is some text with broken json: {"tool_call": {"name": "broken""#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_prompt_tool_fallback_parse_json_without_tool_call_key() {
        let response = r#"{"other_key": {"name": "not_a_tool"}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_prompt_tool_fallback_parse_tool_call_empty_arguments() {
        let response = r#"{"tool_call": {"name": "ping", "arguments": {}}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(calls[0].arguments.is_empty());
    }

    #[test]
    fn test_prompt_tool_fallback_parse_tool_call_no_arguments_key() {
        let response = r#"{"tool_call": {"name": "noop"}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "noop");
        assert!(calls[0].arguments.is_empty());
    }

    #[test]
    fn test_prompt_tool_fallback_parse_numeric_argument() {
        let response =
            r#"{"tool_call": {"name": "set_temp", "arguments": {"value": 42, "unit": "C"}}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments.get("value").and_then(|v| v.as_i64()), Some(42));
        assert_eq!(
            calls[0].arguments.get("unit").and_then(|v| v.as_str()),
            Some("C")
        );
    }

    #[test]
    fn test_prompt_tool_fallback_parse_nested_json_in_response() {
        // Response that has non-tool-call JSON before the actual tool call
        let response = r#"The config is {"model": "gpt-4"}. Now let me call the tool:
{"tool_call": {"name": "lookup", "arguments": {"id": "abc"}}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn test_prompt_tool_fallback_parse_escaped_quotes_in_args() {
        let response =
            r#"{"tool_call": {"name": "echo", "arguments": {"text": "hello \"world\""}}}"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "echo");
        assert!(calls[0].arguments.contains_key("text"));
    }

    #[test]
    fn test_prompt_tool_fallback_incremental_ids() {
        let response = r#"
{"tool_call": {"name": "a", "arguments": {}}}
{"tool_call": {"name": "b", "arguments": {}}}
{"tool_call": {"name": "c", "arguments": {}}}
"#;
        let calls = PromptToolFallback::parse_tool_calls_from_text(response);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].id, "fallback_call_0");
        assert_eq!(calls[1].id, "fallback_call_1");
        assert_eq!(calls[2].id, "fallback_call_2");
    }

    // ====================================================================
    // NEW: extract_json_object edge cases
    // ====================================================================

    #[test]
    fn test_extract_json_object_simple() {
        let result = PromptToolFallback::extract_json_object(r#"{"key": "value"}"#);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_object_nested() {
        let input = r#"{"outer": {"inner": 1}}"#;
        let result = PromptToolFallback::extract_json_object(input);
        assert_eq!(result.unwrap(), input);
    }

    #[test]
    fn test_extract_json_object_empty_string() {
        let result = PromptToolFallback::extract_json_object("");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_object_not_starting_with_brace() {
        let result = PromptToolFallback::extract_json_object("abc{\"key\":1}");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_object_unbalanced() {
        let result = PromptToolFallback::extract_json_object("{\"key\": \"val\"");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_object_with_trailing_text() {
        let result =
            PromptToolFallback::extract_json_object(r#"{"a": 1} and some more text"#);
        assert_eq!(result.unwrap(), r#"{"a": 1}"#);
    }

    #[test]
    fn test_extract_json_object_brace_inside_string() {
        let input = r#"{"msg": "has { and } inside"}"#;
        let result = PromptToolFallback::extract_json_object(input);
        assert_eq!(result.unwrap(), input);
    }

    // ====================================================================
    // NEW: Perplexity and OpenRouter presets
    // ====================================================================

    #[test]
    fn test_perplexity_preset() {
        let provider = OpenAICompatibleProvider::perplexity();
        assert_eq!(provider.name(), "Perplexity");
        assert!(provider.base_url.contains("perplexity.ai"));
    }

    #[test]
    fn test_openrouter_preset() {
        let provider = OpenAICompatibleProvider::openrouter();
        assert_eq!(provider.name(), "OpenRouter");
        assert!(provider.base_url.contains("openrouter.ai"));
    }

    // ====================================================================
    // NEW: ProviderCapabilities default
    // ====================================================================

    #[test]
    fn test_provider_capabilities_default_all_false() {
        let caps = ProviderCapabilities::default();
        assert!(!caps.streaming);
        assert!(!caps.tool_calling);
        assert!(!caps.vision);
        assert!(!caps.embeddings);
        assert!(!caps.json_mode);
        assert!(!caps.system_prompt);
    }

    // ====================================================================
    // NEW: OpenAICompatible api_key handling
    // ====================================================================

    #[test]
    fn test_openai_compatible_no_api_key_by_default() {
        let provider = OpenAICompatibleProvider::new("test", "http://x");
        assert!(provider.api_key.is_none());
    }

    #[test]
    fn test_openai_compatible_with_api_key_chain() {
        let provider = OpenAICompatibleProvider::new("test", "http://x")
            .with_api_key("sk-abc123")
            .with_timeout(Duration::from_secs(10));
        assert_eq!(provider.api_key.as_deref(), Some("sk-abc123"));
        assert_eq!(provider.timeout, Duration::from_secs(10));
    }

    // ====================================================================
    // NEW: HuggingFace with_api_key and is_available
    // ====================================================================

    #[test]
    fn test_huggingface_with_api_key() {
        let provider = HuggingFaceInferenceProvider::new("gpt2").with_api_key("hf_token123");
        assert_eq!(provider.api_key.as_deref(), Some("hf_token123"));
    }

    #[test]
    fn test_huggingface_is_available_default_false() {
        let provider = HuggingFaceInferenceProvider::new("gpt2");
        // Not available by default since the AtomicBool is initialized to false
        assert!(!provider.is_available());
    }

    // ====================================================================
    // NEW: build_tool_prompt with multiple tools
    // ====================================================================

    #[test]
    fn test_prompt_tool_fallback_build_prompt_multiple_tools() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get the weather"),
            ToolDefinition::new("get_time", "Get the time"),
            ToolDefinition::new("search", "Search the web"),
        ];
        let result = PromptToolFallback::build_tool_prompt("You are helpful.", &tools);
        assert!(result.contains("get_weather"));
        assert!(result.contains("get_time"));
        assert!(result.contains("search"));
        assert!(result.contains("Available tools:"));
        assert!(result.contains("tool_call"));
    }

    // ====================================================================
    // NEW: Ollama tool call response parsing — edge cases
    // ====================================================================

    #[test]
    fn test_ollama_tool_call_response_no_tool_calls() {
        let response_json: Value = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "The weather is sunny."
            },
            "done": true
        });

        let tool_calls: Vec<ToolCall> = if let Some(tc_array) = response_json
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
        {
            tc_array
                .iter()
                .enumerate()
                .filter_map(|(i, tc)| {
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments: HashMap<String, Value> =
                        if let Some(args_obj) = function.get("arguments") {
                            serde_json::from_value(args_obj.clone()).unwrap_or_default()
                        } else {
                            HashMap::new()
                        };
                    Some(ToolCall {
                        name,
                        arguments,
                        id: format!("ollama_call_{}", i),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_ollama_tool_call_response_multiple_calls() {
        let response_json: Value = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"}
                        }
                    },
                    {
                        "function": {
                            "name": "get_time",
                            "arguments": {"tz": "UTC"}
                        }
                    }
                ]
            },
            "done": true
        });

        let tool_calls: Vec<ToolCall> = response_json
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
            .map(|tc_array| {
                tc_array
                    .iter()
                    .enumerate()
                    .filter_map(|(i, tc)| {
                        let function = tc.get("function")?;
                        let name = function.get("name")?.as_str()?.to_string();
                        let arguments: HashMap<String, Value> =
                            if let Some(args_obj) = function.get("arguments") {
                                serde_json::from_value(args_obj.clone()).unwrap_or_default()
                            } else {
                                HashMap::new()
                            };
                        Some(ToolCall {
                            name,
                            arguments,
                            id: format!("ollama_call_{}", i),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "ollama_call_0");
        assert_eq!(tool_calls[1].name, "get_time");
        assert_eq!(tool_calls[1].id, "ollama_call_1");
    }

    #[test]
    fn test_ollama_tool_call_arguments_as_json_string() {
        // Some providers return arguments as a JSON string instead of an object
        let response_json: Value = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": "{\"expr\": \"2+2\"}"
                        }
                    }
                ]
            },
            "done": true
        });

        let tc_array = response_json
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
            .unwrap();

        let tc = &tc_array[0];
        let function = tc.get("function").unwrap();
        let name = function.get("name").unwrap().as_str().unwrap().to_string();
        let arguments: HashMap<String, Value> =
            if let Some(args_obj) = function.get("arguments") {
                if let Some(s) = args_obj.as_str() {
                    serde_json::from_str(s).unwrap_or_default()
                } else {
                    serde_json::from_value(args_obj.clone()).unwrap_or_default()
                }
            } else {
                HashMap::new()
            };

        assert_eq!(name, "calc");
        assert_eq!(
            arguments.get("expr").and_then(|v| v.as_str()),
            Some("2+2")
        );
    }

    // ====================================================================
    // NEW: discover_providers with all checks disabled
    // ====================================================================

    #[test]
    fn test_discover_providers_all_disabled() {
        let config = DiscoveryConfig {
            check_ollama: false,
            check_lm_studio: false,
            check_text_gen_webui: false,
            check_kobold_cpp: false,
            ..DiscoveryConfig::default()
        };
        let providers = discover_providers(&config);
        assert!(providers.is_empty());
    }

    // ====================================================================
    // NEW: Provider name consistency
    // ====================================================================

    #[test]
    fn test_all_provider_names_non_empty() {
        assert!(!OllamaProvider::new("http://x").name().is_empty());
        assert!(!LmStudioProvider::new("http://x").name().is_empty());
        assert!(!TextGenWebUIProvider::new("http://x").name().is_empty());
        assert!(!KoboldCppProvider::new("http://x").name().is_empty());
        assert!(!OpenAICompatibleProvider::new("test", "http://x").name().is_empty());
        assert!(!HuggingFaceInferenceProvider::new("gpt2").name().is_empty());
    }

    #[test]
    fn test_provider_names_are_unique() {
        let names = vec![
            OllamaProvider::new("http://x").name().to_string(),
            LmStudioProvider::new("http://x").name().to_string(),
            TextGenWebUIProvider::new("http://x").name().to_string(),
            KoboldCppProvider::new("http://x").name().to_string(),
            HuggingFaceInferenceProvider::new("gpt2").name().to_string(),
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "All provider names must be unique");
    }

    // ====================================================================
    // NEW: OpenAICompatible custom name propagation
    // ====================================================================

    #[test]
    fn test_openai_compatible_custom_name() {
        let provider = OpenAICompatibleProvider::new("my_custom_provider", "http://x");
        assert_eq!(provider.name(), "my_custom_provider");
    }

    // ====================================================================
    // NEW: ProviderCapabilities Clone and Debug
    // ====================================================================

    #[test]
    fn test_provider_capabilities_clone() {
        let caps = ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            vision: true,
            embeddings: false,
            json_mode: true,
            system_prompt: false,
        };
        let cloned = caps.clone();
        assert_eq!(caps.streaming, cloned.streaming);
        assert_eq!(caps.tool_calling, cloned.tool_calling);
        assert_eq!(caps.vision, cloned.vision);
        assert_eq!(caps.embeddings, cloned.embeddings);
        assert_eq!(caps.json_mode, cloned.json_mode);
        assert_eq!(caps.system_prompt, cloned.system_prompt);
    }

    #[test]
    fn test_provider_capabilities_debug() {
        let caps = ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            vision: false,
            embeddings: false,
            json_mode: true,
            system_prompt: true,
        };
        let debug_str = format!("{:?}", caps);
        assert!(debug_str.contains("ProviderCapabilities"));
        assert!(debug_str.contains("streaming: true"));
        assert!(debug_str.contains("vision: false"));
    }
}
