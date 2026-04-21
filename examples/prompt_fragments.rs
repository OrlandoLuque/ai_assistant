//! Demonstrates `prompt_fragments`: composable conditional system prompts.
//!
//! Run with: `cargo run --example prompt_fragments --features prompt-fragments`

#[cfg(feature = "prompt-fragments")]
fn main() {
    use ai_assistant::prompt_fragments::catalog;
    use ai_assistant::{Platform, PromptBuilder, PromptContext, PromptFragment, PromptPreset};

    println!("=== Scenario 1: Agentic loop, bare ===");
    let ctx1 = PromptContext::default()
        .with_platform(Platform::detect())
        .with_tools(vec!["bash".into(), "retrieve".into()]);
    let prompt1 = PromptBuilder::new()
        .with_preset(PromptPreset::AgenticLoop)
        .add_fragment(catalog::unix_shell_note())
        .add_fragment(catalog::windows_shell_note())
        .build(&ctx1);
    println!("{}", prompt1);
    println!();

    println!("=== Scenario 2: Code developer with git detected ===");
    let ctx2 = PromptContext::default()
        .with_platform(Platform::detect())
        .with_tools(vec!["git".into(), "cargo".into(), "bash".into()])
        .with_locale("en");
    let prompt2 = PromptBuilder::new()
        .with_preset(PromptPreset::CodeDeveloper)
        .add_fragment(catalog::unix_shell_note())
        .add_fragment(catalog::windows_shell_note())
        .build(&ctx2);
    println!("{}", prompt2);
    println!();

    println!("=== Scenario 3: RAG assistant, EU region (GDPR fires) ===");
    let ctx3 = PromptContext::default()
        .with_platform(Platform::detect())
        .with_tools(vec!["retrieve".into()])
        .with_region("EU");
    let prompt3 = PromptBuilder::new()
        .with_preset(PromptPreset::RagAssistant)
        .add_fragment(catalog::gdpr_eu_notice())
        .build(&ctx3);
    println!("{}", prompt3);
    println!();

    println!("=== Scenario 4: Custom fragment via context signal ===");
    let ctx4 = PromptContext::default().with_custom("experimental", "on");
    let custom = PromptFragment::new(
        "experimental_notice",
        "Experimental features are enabled; surface a warning before applying destructive actions.",
        ai_assistant::FragmentCategory::Context,
        25,
        |ctx| {
            ctx.custom
                .get("experimental")
                .map(|v| v == "on")
                .unwrap_or(false)
        },
    );
    let prompt4 = PromptBuilder::new()
        .with_preset(PromptPreset::Minimal)
        .add_fragment(custom)
        .build(&ctx4);
    println!("{}", prompt4);
}

#[cfg(not(feature = "prompt-fragments"))]
fn main() {
    eprintln!("This example requires --features prompt-fragments");
}
