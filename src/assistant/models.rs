use super::*;

impl AiAssistant {
    // === Model Discovery ===

    /// Start fetching available models from all providers asynchronously
    pub fn fetch_models(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.rx_models = Some(rx);
        self.is_fetching_models = true;

        let ollama_url = self.config.ollama_url.clone();
        let lm_studio_url = self.config.lm_studio_url.clone();
        let text_gen_webui_url = self.config.text_gen_webui_url.clone();
        let kobold_url = self.config.kobold_url.clone();
        let local_ai_url = self.config.local_ai_url.clone();

        thread::spawn(move || {
            let mut all_models = Vec::new();

            // Try Ollama
            if let Ok(models) = fetch_ollama_models(&ollama_url) {
                all_models.extend(models);
            }

            // Try LM Studio
            if let Ok(models) = fetch_openai_compatible_models(&lm_studio_url, AiProvider::LMStudio)
            {
                all_models.extend(models);
            }

            // Try text-generation-webui
            if let Ok(models) =
                fetch_openai_compatible_models(&text_gen_webui_url, AiProvider::TextGenWebUI)
            {
                all_models.extend(models);
            }

            // Try Kobold.cpp
            if let Ok(models) = fetch_kobold_models(&kobold_url) {
                all_models.extend(models);
            }

            // Try LocalAI
            if let Ok(models) = fetch_openai_compatible_models(&local_ai_url, AiProvider::LocalAI) {
                all_models.extend(models);
            }

            let _ = tx.send(AiResponse::ModelsLoaded(all_models));
        });
    }

    /// Poll for model fetch results. Returns true if models were loaded.
    pub fn poll_models(&mut self) -> bool {
        if let Some(ref rx) = self.rx_models {
            match rx.try_recv() {
                Ok(AiResponse::ModelsLoaded(models)) => {
                    self.available_models = models;
                    self.rx_models = None;
                    self.is_fetching_models = false;

                    // Auto-select first model if none selected
                    if self.config.selected_model.is_empty() && !self.available_models.is_empty() {
                        self.config.selected_model = self.available_models[0].name.clone();
                        self.config.provider = self.available_models[0].provider.clone();
                    }
                    return true;
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_models = None;
                    self.is_fetching_models = false;
                }
            }
        }
        false
    }

    // === Provider Fallback ===

    /// Configure fallback providers for automatic failover.
    ///
    /// When the primary provider fails, the assistant tries each fallback in order.
    /// Each entry is a `(AiProvider, model_name)` pair.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_assistant::{AiAssistant, config::AiProvider};
    ///
    /// let mut ai = AiAssistant::new();
    /// ai.configure_fallback(vec![
    ///     (AiProvider::LMStudio, "local-model".into()),
    ///     (AiProvider::Ollama, "llama3.2:latest".into()),
    /// ]);
    /// ai.enable_fallback();
    /// ```
    pub fn configure_fallback(&mut self, providers: Vec<(AiProvider, String)>) {
        self.fallback_providers = providers;
    }

    /// Enable automatic provider fallback.
    pub fn enable_fallback(&mut self) {
        self.fallback_enabled = true;
    }

    /// Disable automatic provider fallback.
    pub fn disable_fallback(&mut self) {
        self.fallback_enabled = false;
    }

    /// Returns `true` if fallback is enabled and at least one provider is configured.
    pub fn fallback_active(&self) -> bool {
        self.fallback_enabled && !self.fallback_providers.is_empty()
    }

    /// Get the name of the provider that served the last response.
    ///
    /// Updated asynchronously by background generation threads.
    /// Returns `None` before the first response completes.
    pub fn last_provider_used(&self) -> Option<String> {
        self.fallback_last_provider
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    // === API Key Management ===

    /// Initialize the API key manager with custom rotation config.
    ///
    /// Required before adding keys. If not called, `add_api_key` will
    /// initialize a manager with default settings.
    pub fn set_api_key_config(&mut self, config: RotationConfig) {
        self.api_key_manager = Some(ApiKeyManager::new(config));
    }

    /// Add an API key for a provider.
    ///
    /// Creates the key manager with default config if not yet initialized.
    pub fn add_api_key(&mut self, provider: &str, key_id: &str, key_value: &str) {
        if self.api_key_manager.is_none() {
            self.api_key_manager = Some(ApiKeyManager::default());
        }
        let api_key = ApiKey::new(key_id, key_value, provider);
        self.api_key_manager
            .as_mut()
            .expect("api_key_manager must be initialized")
            .add_key(api_key);
    }

    /// Get the current API key for a provider (round-robin, skips rate-limited keys).
    ///
    /// Returns `None` if no usable key is available.
    pub fn get_current_api_key(&mut self, provider: &str) -> Option<String> {
        self.api_key_manager
            .as_mut()
            .and_then(|m| m.get_key(provider))
            .map(|k| k.key.clone())
    }

    /// Mark the current key for a provider as rate-limited, triggering rotation
    /// to the next available key.
    pub fn mark_key_rate_limited(&mut self, provider: &str, key_id: &str) {
        if let Some(ref mut manager) = self.api_key_manager {
            manager.mark_rate_limited(provider, key_id);
        }
    }
}
