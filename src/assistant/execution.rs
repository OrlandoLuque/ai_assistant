use super::*;

impl AiAssistant {
    // === Container Execution ===

    /// Create a new container executor with default configuration.
    #[cfg(feature = "containers")]
    pub fn create_container_executor(
        &self,
    ) -> AiResult<crate::container_executor::ContainerExecutor> {
        crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| AiError::Other(e.to_string()))
    }

    /// Create a container executor with custom configuration.
    #[cfg(feature = "containers")]
    pub fn create_container_executor_with_config(
        &self,
        config: crate::container_executor::ContainerConfig,
    ) -> AiResult<crate::container_executor::ContainerExecutor> {
        crate::container_executor::ContainerExecutor::new(config)
            .map_err(|e| AiError::Other(e.to_string()))
    }

    /// Execute code in an isolated Docker container.
    ///
    /// Automatically selects the appropriate Docker image based on language.
    /// Falls back to process-based execution if Docker is unavailable.
    #[cfg(feature = "containers")]
    pub fn run_code_isolated(
        &self,
        code: &str,
        language: &crate::code_sandbox::Language,
    ) -> AiResult<crate::code_sandbox::ExecutionResult> {
        let mut sandbox = crate::container_sandbox::ContainerSandbox::new(
            crate::container_sandbox::ContainerSandboxConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Ok(sandbox.execute(language, code))
    }

    /// Create a shared folder for container file exchange.
    #[cfg(feature = "containers")]
    pub fn create_shared_folder(&self) -> AiResult<crate::shared_folder::SharedFolder> {
        crate::shared_folder::SharedFolder::temp().map_err(AiError::from)
    }

    // === Document Creation ===

    /// Create a document pipeline with default settings.
    ///
    /// Internally creates a `ContainerExecutor` and a temporary `SharedFolder`.
    #[cfg(feature = "containers")]
    pub fn create_document_pipeline(&self) -> AiResult<crate::document_pipeline::DocumentPipeline> {
        let executor = crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let shared_folder = crate::shared_folder::SharedFolder::temp()?;
        Ok(crate::document_pipeline::DocumentPipeline::new(
            crate::document_pipeline::DocumentPipelineConfig::default(),
            std::sync::Arc::new(std::sync::RwLock::new(executor)),
            shared_folder,
        ))
    }

    /// Create a document by converting content to the specified format.
    ///
    /// Uses container-based pandoc/LibreOffice for conversion.
    #[cfg(feature = "containers")]
    pub fn create_document(
        &self,
        content: &str,
        source_format: crate::document_pipeline::SourceFormat,
        output_format: crate::document_pipeline::OutputFormat,
    ) -> AiResult<crate::document_pipeline::DocumentResult> {
        let executor = crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let shared_folder = crate::shared_folder::SharedFolder::temp()?;
        let mut pipeline = crate::document_pipeline::DocumentPipeline::new(
            crate::document_pipeline::DocumentPipelineConfig::default(),
            std::sync::Arc::new(std::sync::RwLock::new(executor)),
            shared_folder,
        );
        let request = crate::document_pipeline::DocumentRequest {
            content: content.to_string(),
            source_format,
            output_format,
            output_name: "document".into(),
            stylesheet: None,
            extra_args: Vec::new(),
            metadata: std::collections::HashMap::new(),
        };
        pipeline
            .create(&request)
            .map_err(|e| AiError::Other(e.to_string()))
    }

    // === Speech (STT / TTS) ===

    /// Transcribe audio to text using the specified speech provider.
    ///
    /// # Arguments
    /// * `provider_name` - Provider name ("openai", "google", "whisper", "local")
    /// * `audio` - Raw audio bytes
    /// * `format` - Audio encoding format
    /// * `language` - Optional language hint (ISO 639-1)
    #[cfg(feature = "audio")]
    pub fn transcribe(
        &self,
        provider_name: &str,
        audio: &[u8],
        format: crate::speech::AudioFormat,
        language: Option<&str>,
    ) -> AiResult<crate::speech::TranscriptionResult> {
        let provider = crate::speech::create_speech_provider(provider_name)?;
        provider
            .transcribe(audio, format, language)
            .map_err(AiError::from)
    }

    /// Synthesize text to audio using the specified speech provider.
    ///
    /// # Arguments
    /// * `provider_name` - Provider name ("openai", "google", "piper", "coqui", "local")
    /// * `text` - Text to synthesize
    /// * `options` - Synthesis options (voice, format, speed)
    #[cfg(feature = "audio")]
    pub fn synthesize(
        &self,
        provider_name: &str,
        text: &str,
        options: &crate::speech::SynthesisOptions,
    ) -> AiResult<crate::speech::SynthesisResult> {
        let provider = crate::speech::create_speech_provider(provider_name)?;
        provider.synthesize(text, options).map_err(AiError::from)
    }

    /// Get the recommended speech configuration from butler (if available).
    ///
    /// Returns (stt_provider, tts_provider) suggestions based on detected environment.
    #[cfg(all(feature = "audio", feature = "butler"))]
    pub fn suggest_speech_providers(&mut self) -> (Option<String>, Option<String>) {
        let mut butler = crate::butler::Butler::new();
        butler.scan();
        butler.suggest_speech_config()
    }

    // === Voice Cloning (V67) ===

    /// Enroll a voice for cloning using the specified provider.
    ///
    /// # Arguments
    /// * `provider_name` - Clone provider: "elevenlabs" or "xtts"
    /// * `audio` - Raw PCM16 audio bytes (min 3 seconds)
    /// * `name` - Name for the cloned voice
    #[cfg(feature = "audio")]
    pub fn enroll_voice_clone(
        &self,
        provider_name: &str,
        audio: &[u8],
        name: &str,
    ) -> AiResult<String> {
        use crate::speech::VoiceCloneProvider;
        let (quality, warnings) = crate::speech::assess_enrollment_quality(audio, 16000);
        if quality < 0.3 {
            return Err(AiError::Other(format!(
                "Audio quality too low ({:.0}%): {}",
                quality * 100.0,
                warnings.join("; ")
            )));
        }
        match provider_name {
            "elevenlabs" => {
                let provider = crate::speech::ElevenLabsCloneProvider::from_env()?;
                provider
                    .enroll(audio, crate::speech::AudioFormat::Pcm, name, 16000)
                    .map_err(AiError::from)
            }
            "xtts" => {
                let provider = crate::speech::XttsCloneProvider::local();
                provider
                    .enroll(audio, crate::speech::AudioFormat::Pcm, name, 16000)
                    .map_err(AiError::from)
            }
            _ => {
                return Err(AiError::Other(format!(
                    "Unknown clone provider '{}'. Available: elevenlabs, xtts",
                    provider_name
                )))
            }
        }
    }

    /// Synthesize speech using a cloned voice.
    #[cfg(feature = "audio")]
    pub fn synthesize_cloned(
        &self,
        provider_name: &str,
        text: &str,
        voice_id: &str,
    ) -> AiResult<crate::speech::SynthesisResult> {
        use crate::speech::VoiceCloneProvider;
        let options = crate::speech::SynthesisOptions::default();
        match provider_name {
            "elevenlabs" => {
                let provider = crate::speech::ElevenLabsCloneProvider::from_env()?;
                provider
                    .synthesize_cloned(text, &voice_id.to_string(), &options)
                    .map_err(AiError::from)
            }
            "xtts" => {
                let provider = crate::speech::XttsCloneProvider::local();
                provider
                    .synthesize_cloned(text, &voice_id.to_string(), &options)
                    .map_err(AiError::from)
            }
            _ => {
                return Err(AiError::Other(format!(
                    "Unknown clone provider '{}'",
                    provider_name
                )))
            }
        }
    }
}
