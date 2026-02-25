//! AWS Bedrock provider demo — credentials, request building, and model listing.
//!
//! Shows how to configure AWS credentials, build a Bedrock request, and
//! list available models. Does NOT make real API calls (no network required).
//!
//! Run with: cargo run --example bedrock_demo --features "aws-bedrock"

use ai_assistant::{
    AwsCredentials, BedrockMessage, BedrockRequest, SigV4Params,
    fetch_bedrock_models,
};

fn main() {
    println!("=== AWS Bedrock Provider Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Construct AWS credentials manually
    // -----------------------------------------------------------------------
    println!("--- AWS Credentials ---");
    let creds = AwsCredentials::new(
        "AKIAIOSFODNN7EXAMPLE".to_string(),
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
    );
    println!("Access key: {}", creds.access_key_id);
    println!("Session token: {:?}", creds.session_token);

    // Add an optional session token (for temporary credentials / STS)
    let creds_with_token = creds.clone().with_session_token("FwoGZX...example_token".to_string());
    println!("With session token: {:?}\n", creds_with_token.session_token.as_deref());

    // -----------------------------------------------------------------------
    // 2. Build a BedrockRequest
    // -----------------------------------------------------------------------
    println!("--- BedrockRequest ---");
    let request = BedrockRequest {
        model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        region: "us-east-1".to_string(),
        messages: vec![
            BedrockMessage {
                role: "user".to_string(),
                content: "What is the capital of France?".to_string(),
            },
        ],
        system_prompt: Some("You are a helpful geography assistant.".to_string()),
        max_tokens: 1024,
        temperature: 0.7,
    };
    println!("Model: {}", request.model_id);
    println!("Region: {}", request.region);
    println!("Messages: {}", request.messages.len());
    println!(
        "  [0] role={}, content=\"{}\"",
        request.messages[0].role, request.messages[0].content
    );
    println!("System prompt: {:?}", request.system_prompt.as_deref());
    println!("Max tokens: {}", request.max_tokens);
    println!("Temperature: {}", request.temperature);

    // -----------------------------------------------------------------------
    // 3. Build SigV4Params (what would be signed before sending)
    // -----------------------------------------------------------------------
    println!("\n--- SigV4Params ---");
    let body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    });
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();

    let params = SigV4Params {
        method: "POST".to_string(),
        url: format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            request.region, request.model_id,
        ),
        region: request.region.clone(),
        service: "bedrock".to_string(),
        headers: vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("accept".to_string(), "application/json".to_string()),
        ],
        body: body_bytes,
    };
    println!("Method: {}", params.method);
    println!("URL: {}", params.url);
    println!("Region: {}", params.region);
    println!("Service: {}", params.service);
    println!("Headers: {:?}", params.headers);
    println!("Body length: {} bytes", params.body.len());

    // -----------------------------------------------------------------------
    // 4. List available Bedrock models
    // -----------------------------------------------------------------------
    println!("\n--- Available Bedrock Models ---");
    let models = fetch_bedrock_models();
    for (i, model) in models.iter().enumerate() {
        println!("  {}. {}", i + 1, model);
    }
    println!("Total: {} models", models.len());

    // -----------------------------------------------------------------------
    // 5. Multi-turn conversation example
    // -----------------------------------------------------------------------
    println!("\n--- Multi-turn conversation request ---");
    let conversation = BedrockRequest {
        model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        region: "eu-west-1".to_string(),
        messages: vec![
            BedrockMessage {
                role: "user".to_string(),
                content: "What is Rust?".to_string(),
            },
            BedrockMessage {
                role: "assistant".to_string(),
                content: "Rust is a systems programming language focused on safety and performance.".to_string(),
            },
            BedrockMessage {
                role: "user".to_string(),
                content: "What are its key features?".to_string(),
            },
        ],
        system_prompt: None,
        max_tokens: 2048,
        temperature: 0.5,
    };
    println!("Conversation turns: {}", conversation.messages.len());
    for msg in &conversation.messages {
        println!("  [{}] {}", msg.role, &msg.content[..msg.content.len().min(60)]);
    }

    // -----------------------------------------------------------------------
    // 6. Environment-based credentials (demonstration only)
    // -----------------------------------------------------------------------
    println!("\n--- Environment credentials ---");
    match AwsCredentials::from_env() {
        Ok(env_creds) => {
            println!("Found AWS credentials in environment:");
            println!("  Access key: {}...", &env_creds.access_key_id[..8.min(env_creds.access_key_id.len())]);
            println!("  Session token: {}", env_creds.session_token.is_some());
        }
        Err(e) => {
            println!("No AWS credentials in environment (expected in demo): {}", e);
        }
    }

    println!("\nBedrock demo complete.");
}
