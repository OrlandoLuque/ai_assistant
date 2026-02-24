//! DSPy-style prompt signature demo.
//!
//! Run with: cargo run --example prompt_signature_demo --features prompt-signatures
//!
//! Demonstrates prompt signatures, field types, compiled prompts,
//! and the bootstrap few-shot optimizer.

use ai_assistant::{
    FieldType, Signature, SignatureField, CompiledPrompt, BootstrapFewShot, ExactMatch,
};

fn main() {
    println!("=== Prompt Signature Demo ===\n");

    // 1. Define a signature for sentiment classification
    let sig = Signature::new("classify_sentiment", "Classify the sentiment of the given text.")
        .with_instructions("Respond with exactly one of: positive, negative, neutral.")
        .add_input(
            SignatureField::new("text", "The text to analyze"),
        )
        .add_output(
            SignatureField::new("sentiment", "The detected sentiment")
                .with_type(FieldType::Text),
        )
        .add_output(
            SignatureField::new("confidence", "Confidence score 0-1")
                .with_type(FieldType::Number),
        );

    println!("Signature: {}", sig.name);
    println!("Description: {}", sig.description);
    println!("Inputs: {:?}", sig.inputs.iter().map(|f| &f.name).collect::<Vec<_>>());
    println!("Outputs: {:?}", sig.outputs.iter().map(|f| &f.name).collect::<Vec<_>>());

    // 2. Compile the signature into a prompt template
    let compiled: CompiledPrompt = sig.compile();

    println!("\nSystem prompt (first 200 chars):");
    println!("{}", &compiled.system_prompt[..compiled.system_prompt.len().min(200)]);

    println!("\nUser template:");
    println!("{}", compiled.user_template);

    // 3. Render with actual input values
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("text".to_string(), "I love this product! It works great.".to_string());

    let rendered = compiled.render(&inputs);
    println!("\nRendered prompt:");
    println!("{}", &rendered[..rendered.len().min(300)]);

    // 4. Create a BootstrapFewShot optimizer
    let _optimizer = BootstrapFewShot::new(10, Box::new(ExactMatch));
    println!("\nBootstrapFewShot optimizer created (max_examples=10)");
    println!("Metric: ExactMatch");

    println!("\n=== Done ===");
}
