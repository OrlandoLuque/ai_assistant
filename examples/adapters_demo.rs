//! Example: adapters_demo -- Demonstrates the external API adapter system.
//!
//! Run with: cargo run --example adapters_demo --features adapters
//!
//! This example showcases how ai_assistant provides unified adapters
//! for OpenAI, Anthropic, HuggingFace, and local LLM providers,
//! including model presets, request building, and provider discovery.

use ai_assistant::{
    // OpenAI adapter
    OpenAIClient, OpenAIConfig, OpenAIMessage, OpenAIModel, OpenAIRequest,
    // Anthropic adapter
    AnthropicClient, AnthropicConfig, AnthropicMessage, AnthropicModel, AnthropicRequest,
    // HuggingFace connector
    HfClient, HfConfig, HfRequest, HfTask,
    // Provider plugins (local LLM discovery)
    DiscoveryConfig, OllamaProvider,
};

fn main() {
    println!("==========================================================");
    println!("  ai_assistant -- External API Adapters Demo");
    println!("==========================================================\n");

    // ------------------------------------------------------------------
    // 1. OpenAI Adapter: Model Presets
    // ------------------------------------------------------------------
    println!("--- 1. OpenAI Model Presets ---\n");

    let gpt4_turbo = OpenAIModel::gpt4_turbo();
    println!("  Model:    {} ({})", gpt4_turbo.name, gpt4_turbo.id);
    println!("  Context:  {} tokens", gpt4_turbo.context_length);
    println!("  Vision:   {}", gpt4_turbo.supports_vision);
    println!("  Functions: {}", gpt4_turbo.supports_functions);

    let gpt4 = OpenAIModel::gpt4();
    println!("\n  Model:    {} ({})", gpt4.name, gpt4.id);
    println!("  Context:  {} tokens", gpt4.context_length);
    println!("  Vision:   {}", gpt4.supports_vision);

    let gpt35 = OpenAIModel::gpt35_turbo();
    println!("\n  Model:    {} ({})", gpt35.name, gpt35.id);
    println!("  Context:  {} tokens", gpt35.context_length);

    // ------------------------------------------------------------------
    // 2. OpenAI Request Building
    // ------------------------------------------------------------------
    println!("\n--- 2. OpenAI Request Building ---\n");

    let messages = vec![
        OpenAIMessage::system("You are a helpful coding assistant."),
        OpenAIMessage::user("Explain Rust's ownership model in one paragraph."),
    ];
    let request = OpenAIRequest::new("gpt-4-turbo-preview", messages)
        .with_temperature(0.7)
        .with_max_tokens(256);

    println!("  Model:       {}", request.model);
    println!("  Temperature: {:?}", request.temperature);
    println!("  Max tokens:  {:?}", request.max_tokens);
    println!("  Streaming:   {}", request.stream);
    println!("  Messages:    {} message(s)", request.messages.len());

    // Demonstrate streaming request variant
    let stream_msgs = vec![OpenAIMessage::user("Hello!")];
    let stream_req = OpenAIRequest::new("gpt-3.5-turbo", stream_msgs).streaming();
    println!("\n  Streaming request: stream={}", stream_req.stream);

    // Demonstrate multi-modal message
    let _vision_msg = OpenAIMessage::user_with_image(
        "Describe this image",
        "https://example.com/photo.png",
    );
    println!("  Vision message role: user (with image URL)");

    // ------------------------------------------------------------------
    // 3. OpenAI Configuration Variants
    // ------------------------------------------------------------------
    println!("\n--- 3. OpenAI Configuration Variants ---\n");

    let cloud_config = OpenAIConfig::new("sk-demo-key-not-real");
    println!("  Cloud config:");
    println!("    Base URL:   {}", cloud_config.base_url);
    println!("    Timeout:    {:?}", cloud_config.timeout);
    println!("    Retries:    {}", cloud_config.max_retries);

    let azure_config = OpenAIConfig::azure(
        "https://myresource.openai.azure.com",
        "az-key-demo",
        "gpt4-deployment",
    );
    println!("\n  Azure config:");
    println!("    Base URL:   {}", azure_config.base_url);

    let local_config = OpenAIConfig::local("http://localhost:8080/v1");
    println!("\n  Local config:");
    println!("    Base URL:   {}", local_config.base_url);
    println!("    API key:    (empty = {})", local_config.api_key.is_empty());

    // Create client (does not make network calls)
    let _client = OpenAIClient::new(cloud_config);
    println!("\n  OpenAIClient created successfully (no network call yet)");

    // ------------------------------------------------------------------
    // 4. Anthropic Adapter: Model Presets
    // ------------------------------------------------------------------
    println!("\n--- 4. Anthropic Model Presets ---\n");

    let models = vec![
        AnthropicModel::claude3_opus(),
        AnthropicModel::claude3_sonnet(),
        AnthropicModel::claude3_haiku(),
        AnthropicModel::claude2(),
    ];

    for m in &models {
        println!(
            "  {:<20} id={:<35} ctx={:<7} vision={}",
            m.name, m.id, m.context_length, m.supports_vision
        );
    }

    // ------------------------------------------------------------------
    // 5. Anthropic Request Building
    // ------------------------------------------------------------------
    println!("\n--- 5. Anthropic Request Building ---\n");

    let anth_messages = vec![
        AnthropicMessage::user("What is the capital of France?"),
        AnthropicMessage::assistant("The capital of France is Paris."),
        AnthropicMessage::user("And of Spain?"),
    ];
    let anth_request = AnthropicRequest::new("claude-3-sonnet-20240229", anth_messages)
        .with_system("You are a geography expert. Be concise.")
        .with_temperature(0.3)
        .with_max_tokens(512);

    println!("  Model:      {}", anth_request.model);
    println!("  System:     {:?}", anth_request.system);
    println!("  Max tokens: {}", anth_request.max_tokens);
    println!("  Messages:   {} message(s)", anth_request.messages.len());

    let anth_config = AnthropicConfig::new("sk-ant-demo-not-real");
    println!("\n  Anthropic config:");
    println!("    Base URL:    {}", anth_config.base_url);
    println!("    API version: {}", anth_config.api_version);
    println!("    Timeout:     {:?}", anth_config.timeout);

    let _anth_client = AnthropicClient::new(anth_config);
    println!("  AnthropicClient created successfully");

    // ------------------------------------------------------------------
    // 6. HuggingFace Connector
    // ------------------------------------------------------------------
    println!("\n--- 6. HuggingFace Connector ---\n");

    // Task types
    let tasks = [
        HfTask::TextGeneration,
        HfTask::Summarization,
        HfTask::QuestionAnswering,
        HfTask::FeatureExtraction,
        HfTask::TextClassification,
        HfTask::ImageToText,
    ];
    println!("  Supported task types:");
    for task in &tasks {
        println!("    - {:?} => \"{}\"", task, task.as_str());
    }

    // Request builders
    let text_req = HfRequest::text_generation("Once upon a time in Rust-land");
    println!("\n  Text generation request created (has params: {})",
        text_req.parameters.is_some());

    let qa_req = HfRequest::question_answering(
        "What is Rust?",
        "Rust is a systems programming language focused on safety and performance.",
    );
    println!("  QA request created (inputs is object: {})", qa_req.inputs.is_object());

    let summ_req = HfRequest::summarization(
        "Rust is a multi-paradigm programming language designed for performance and safety.",
    );
    println!("  Summarization request created (has params: {})",
        summ_req.parameters.is_some());

    // Config variants
    let hf_config = HfConfig::new().with_token("hf_demo_token");
    println!("\n  HF config with token: has_token={}", hf_config.api_token.is_some());
    println!("    Base URL: {}", hf_config.base_url);

    let hf_local = HfConfig::local("http://localhost:8080");
    println!("  HF local config: base_url={}", hf_local.base_url);

    let _hf_client = HfClient::new(hf_config);
    println!("  HfClient created successfully");

    // ------------------------------------------------------------------
    // 7. Provider Discovery Configuration
    // ------------------------------------------------------------------
    println!("\n--- 7. Local Provider Discovery ---\n");

    let discovery = DiscoveryConfig::default();
    println!("  Discovery configuration:");
    println!("    Check Ollama:     {} ({})", discovery.check_ollama, discovery.ollama_url);
    println!("    Check LM Studio:  {} ({})", discovery.check_lm_studio, discovery.lm_studio_url);
    println!("    Check TGW:        {} ({})", discovery.check_text_gen_webui, discovery.text_gen_webui_url);
    println!("    Check KoboldCpp:  {} ({})", discovery.check_kobold_cpp, discovery.kobold_cpp_url);
    println!("    Timeout:          {:?}", discovery.timeout);

    // Create an Ollama provider directly (does not connect yet)
    let ollama = OllamaProvider::new("http://localhost:11434");
    println!("\n  OllamaProvider created for: {}", ollama.base_url);

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    println!("\n==========================================================");
    println!("  Adapters demo complete.");
    println!("  Supported providers: OpenAI, Anthropic, HuggingFace,");
    println!("    Ollama, LM Studio, KoboldCpp, text-generation-webui,");
    println!("    and any OpenAI-compatible API.");
    println!("==========================================================");
}
