//! Model profiles for predefined generation settings
//!
//! This module provides pre-configured profiles for different use cases like
//! creative writing, precise technical responses, or balanced general use.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A model profile with predefined generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Profile name
    pub name: String,
    /// Profile description
    pub description: String,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Top-p sampling (0.0 - 1.0)
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: Option<u32>,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: f32,
    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: f32,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// System prompt modifier (appended to system prompt)
    pub system_prompt_modifier: Option<String>,
    /// Custom parameters for specific providers
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self::balanced()
    }
}

impl ModelProfile {
    /// Create a balanced profile for general use
    pub fn balanced() -> Self {
        Self {
            name: "balanced".to_string(),
            description: "Balanced settings for general use".to_string(),
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: None,
            custom_params: HashMap::new(),
        }
    }

    /// Create a creative profile for imaginative responses
    pub fn creative() -> Self {
        Self {
            name: "creative".to_string(),
            description: "Higher temperature for creative and varied responses".to_string(),
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(80),
            repetition_penalty: 1.05,
            frequency_penalty: 0.3,
            presence_penalty: 0.3,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Be creative and imaginative in your responses. Feel free to explore unconventional ideas.".to_string()
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a precise profile for accurate, focused responses
    pub fn precise() -> Self {
        Self {
            name: "precise".to_string(),
            description: "Lower temperature for precise, deterministic responses".to_string(),
            temperature: 0.3,
            top_p: 0.8,
            top_k: Some(20),
            repetition_penalty: 1.15,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Be precise and accurate. Stick to facts and avoid speculation.".to_string(),
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a coding profile optimized for code generation
    pub fn coding() -> Self {
        Self {
            name: "coding".to_string(),
            description: "Optimized settings for code generation".to_string(),
            temperature: 0.2,
            top_p: 0.85,
            top_k: Some(30),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: Some(4096),
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Write clean, well-documented code. Follow best practices and include error handling.".to_string()
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a conversational profile for natural dialogue
    pub fn conversational() -> Self {
        Self {
            name: "conversational".to_string(),
            description: "Natural, flowing conversation style".to_string(),
            temperature: 0.8,
            top_p: 0.92,
            top_k: Some(50),
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Respond in a natural, conversational tone. Be friendly and engaging.".to_string(),
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a concise profile for brief responses
    pub fn concise() -> Self {
        Self {
            name: "concise".to_string(),
            description: "Short, to-the-point responses".to_string(),
            temperature: 0.5,
            top_p: 0.85,
            top_k: Some(30),
            repetition_penalty: 1.2,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: Some(256),
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Be concise and direct. Get to the point quickly without unnecessary elaboration."
                    .to_string(),
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a detailed profile for thorough explanations
    pub fn detailed() -> Self {
        Self {
            name: "detailed".to_string(),
            description: "Thorough, comprehensive responses".to_string(),
            temperature: 0.6,
            top_p: 0.9,
            top_k: Some(40),
            repetition_penalty: 1.05,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Provide detailed, comprehensive explanations. Include examples and cover edge cases.".to_string()
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a translation profile
    pub fn translation() -> Self {
        Self {
            name: "translation".to_string(),
            description: "Optimized for translation tasks".to_string(),
            temperature: 0.3,
            top_p: 0.9,
            top_k: Some(30),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Translate accurately while preserving meaning and tone. Maintain cultural nuances.".to_string()
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a roleplay profile for character interactions
    pub fn roleplay() -> Self {
        Self {
            name: "roleplay".to_string(),
            description: "Character roleplay and storytelling".to_string(),
            temperature: 1.1,
            top_p: 0.95,
            top_k: Some(100),
            repetition_penalty: 1.05,
            frequency_penalty: 0.4,
            presence_penalty: 0.4,
            max_tokens: None,
            stop_sequences: vec![],
            system_prompt_modifier: Some(
                "Stay in character. Be expressive and immersive. Use descriptive language."
                    .to_string(),
            ),
            custom_params: HashMap::new(),
        }
    }

    /// Create a custom profile with a builder
    pub fn custom(name: &str) -> ProfileBuilder {
        ProfileBuilder::new(name)
    }

    /// Set a custom parameter
    pub fn with_custom_param(mut self, key: &str, value: serde_json::Value) -> Self {
        self.custom_params.insert(key.to_string(), value);
        self
    }
}

/// Builder for creating custom profiles
pub struct ProfileBuilder {
    profile: ModelProfile,
}

impl ProfileBuilder {
    /// Create a new profile builder
    pub fn new(name: &str) -> Self {
        Self {
            profile: ModelProfile {
                name: name.to_string(),
                description: String::new(),
                ..ModelProfile::balanced()
            },
        }
    }

    /// Set the description
    pub fn description(mut self, desc: &str) -> Self {
        self.profile.description = desc.to_string();
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.profile.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set top-p
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.profile.top_p = top_p.clamp(0.0, 1.0);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.profile.top_k = Some(top_k);
        self
    }

    /// Set repetition penalty
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.profile.repetition_penalty = penalty.max(0.0);
        self
    }

    /// Set frequency penalty
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.profile.frequency_penalty = penalty.clamp(-2.0, 2.0);
        self
    }

    /// Set presence penalty
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.profile.presence_penalty = penalty.clamp(-2.0, 2.0);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.profile.max_tokens = Some(max);
        self
    }

    /// Add a stop sequence
    pub fn stop_sequence(mut self, seq: &str) -> Self {
        self.profile.stop_sequences.push(seq.to_string());
        self
    }

    /// Set system prompt modifier
    pub fn system_prompt_modifier(mut self, modifier: &str) -> Self {
        self.profile.system_prompt_modifier = Some(modifier.to_string());
        self
    }

    /// Add a custom parameter
    pub fn custom_param(mut self, key: &str, value: serde_json::Value) -> Self {
        self.profile.custom_params.insert(key.to_string(), value);
        self
    }

    /// Build the profile
    pub fn build(self) -> ModelProfile {
        self.profile
    }
}

/// Manager for storing and retrieving profiles
#[derive(Debug, Default)]
pub struct ProfileManager {
    profiles: HashMap<String, ModelProfile>,
    current_profile: Option<String>,
}

impl ProfileManager {
    /// Create a new profile manager with built-in profiles
    pub fn new() -> Self {
        let mut manager = Self {
            profiles: HashMap::new(),
            current_profile: None,
        };

        // Add all built-in profiles
        manager.add_profile(ModelProfile::balanced());
        manager.add_profile(ModelProfile::creative());
        manager.add_profile(ModelProfile::precise());
        manager.add_profile(ModelProfile::coding());
        manager.add_profile(ModelProfile::conversational());
        manager.add_profile(ModelProfile::concise());
        manager.add_profile(ModelProfile::detailed());
        manager.add_profile(ModelProfile::translation());
        manager.add_profile(ModelProfile::roleplay());

        manager.current_profile = Some("balanced".to_string());
        manager
    }

    /// Add a profile
    pub fn add_profile(&mut self, profile: ModelProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Remove a profile
    pub fn remove_profile(&mut self, name: &str) -> Option<ModelProfile> {
        self.profiles.remove(name)
    }

    /// Get a profile by name
    pub fn get_profile(&self, name: &str) -> Option<&ModelProfile> {
        self.profiles.get(name)
    }

    /// Get the current profile
    pub fn current(&self) -> Option<&ModelProfile> {
        self.current_profile
            .as_ref()
            .and_then(|name| self.profiles.get(name))
    }

    /// Set the current profile
    pub fn set_current(&mut self, name: &str) -> bool {
        if self.profiles.contains_key(name) {
            self.current_profile = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// List all available profiles
    pub fn list_profiles(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Get profile descriptions
    pub fn profile_descriptions(&self) -> Vec<(&str, &str)> {
        self.profiles
            .iter()
            .map(|(name, profile)| (name.as_str(), profile.description.as_str()))
            .collect()
    }

    /// Export profiles to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.profiles)
    }

    /// Import profiles from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize, serde_json::Error> {
        let imported: HashMap<String, ModelProfile> = serde_json::from_str(json)?;
        let count = imported.len();
        self.profiles.extend(imported);
        Ok(count)
    }
}

/// Apply a profile to generation parameters
#[derive(Debug, Clone, Default)]
pub struct ProfileApplicator;

impl ProfileApplicator {
    /// Generate OpenAI-compatible parameters from a profile
    pub fn to_openai_params(profile: &ModelProfile) -> serde_json::Value {
        let mut params = serde_json::json!({
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "frequency_penalty": profile.frequency_penalty,
            "presence_penalty": profile.presence_penalty,
        });

        if let Some(max_tokens) = profile.max_tokens {
            params["max_tokens"] = serde_json::json!(max_tokens);
        }

        if !profile.stop_sequences.is_empty() {
            params["stop"] = serde_json::json!(profile.stop_sequences);
        }

        // Add custom params
        if let serde_json::Value::Object(ref mut map) = params {
            for (key, value) in &profile.custom_params {
                map.insert(key.clone(), value.clone());
            }
        }

        params
    }

    /// Generate Ollama-compatible parameters from a profile
    pub fn to_ollama_params(profile: &ModelProfile) -> serde_json::Value {
        let mut options = serde_json::json!({
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "repeat_penalty": profile.repetition_penalty,
        });

        if let Some(top_k) = profile.top_k {
            options["top_k"] = serde_json::json!(top_k);
        }

        if let Some(max_tokens) = profile.max_tokens {
            options["num_predict"] = serde_json::json!(max_tokens);
        }

        if !profile.stop_sequences.is_empty() {
            options["stop"] = serde_json::json!(profile.stop_sequences);
        }

        // Add custom params
        if let serde_json::Value::Object(ref mut map) = options {
            for (key, value) in &profile.custom_params {
                map.insert(key.clone(), value.clone());
            }
        }

        options
    }

    /// Generate Kobold-compatible parameters from a profile
    pub fn to_kobold_params(profile: &ModelProfile) -> serde_json::Value {
        let mut params = serde_json::json!({
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "rep_pen": profile.repetition_penalty,
        });

        if let Some(top_k) = profile.top_k {
            params["top_k"] = serde_json::json!(top_k);
        }

        if let Some(max_tokens) = profile.max_tokens {
            params["max_length"] = serde_json::json!(max_tokens);
        }

        if !profile.stop_sequences.is_empty() {
            params["stop_sequence"] = serde_json::json!(profile.stop_sequences);
        }

        params
    }

    /// Modify system prompt with profile modifier
    pub fn apply_system_modifier(profile: &ModelProfile, system_prompt: &str) -> String {
        match &profile.system_prompt_modifier {
            Some(modifier) if !modifier.is_empty() => {
                format!("{}\n\n{}", system_prompt, modifier)
            }
            _ => system_prompt.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_profiles() {
        let balanced = ModelProfile::balanced();
        assert_eq!(balanced.name, "balanced");
        assert!(balanced.temperature >= 0.0 && balanced.temperature <= 2.0);

        let creative = ModelProfile::creative();
        assert!(creative.temperature > balanced.temperature);

        let precise = ModelProfile::precise();
        assert!(precise.temperature < balanced.temperature);
    }

    #[test]
    fn test_profile_builder() {
        let profile = ModelProfile::custom("my_profile")
            .description("My custom profile")
            .temperature(0.5)
            .top_p(0.85)
            .max_tokens(1024)
            .stop_sequence("END")
            .build();

        assert_eq!(profile.name, "my_profile");
        assert_eq!(profile.temperature, 0.5);
        assert_eq!(profile.max_tokens, Some(1024));
        assert_eq!(profile.stop_sequences, vec!["END"]);
    }

    #[test]
    fn test_profile_manager() {
        let mut manager = ProfileManager::new();

        assert!(manager.get_profile("balanced").is_some());
        assert!(manager.get_profile("creative").is_some());
        assert!(manager.get_profile("precise").is_some());

        let custom = ModelProfile::custom("test")
            .description("Test profile")
            .temperature(0.42)
            .build();
        manager.add_profile(custom);

        assert!(manager.get_profile("test").is_some());
        assert!(manager.set_current("test"));

        let current = manager.current().unwrap();
        assert_eq!(current.temperature, 0.42);
    }

    #[test]
    fn test_openai_params() {
        let profile = ModelProfile::precise();
        let params = ProfileApplicator::to_openai_params(&profile);

        // Use approximate comparison for floats
        let temp = params["temperature"].as_f64().unwrap();
        let top_p = params["top_p"].as_f64().unwrap();
        assert!((temp - 0.3).abs() < 0.01);
        assert!((top_p - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_ollama_params() {
        let profile = ModelProfile::creative();
        let params = ProfileApplicator::to_ollama_params(&profile);

        // Use approximate comparison for floats
        let temp = params["temperature"].as_f64().unwrap();
        let penalty = params["repeat_penalty"].as_f64().unwrap();
        assert!((temp - 1.0).abs() < 0.01);
        assert!((penalty - 1.05).abs() < 0.01);
    }

    #[test]
    fn test_system_modifier() {
        let profile = ModelProfile::coding();
        let base_prompt = "You are an assistant.";
        let modified = ProfileApplicator::apply_system_modifier(&profile, base_prompt);

        assert!(modified.contains(base_prompt));
        assert!(modified.contains("clean, well-documented code"));
    }

    #[test]
    fn test_profile_export_import() {
        let mut manager = ProfileManager::new();
        let custom = ModelProfile::custom("exported").temperature(0.99).build();
        manager.add_profile(custom);

        let json = manager.export_json().unwrap();
        assert!(json.contains("exported"));

        let mut new_manager = ProfileManager::new();
        let count = new_manager.import_json(&json).unwrap();
        assert!(count > 0);
        assert!(new_manager.get_profile("exported").is_some());
    }
}
