//! LM Adapters — Provider-Aware Compilation for different LLM providers.

use super::types::CompiledPrompt;

// ============================================================================
// 1.4 LM Adapters — Provider-Aware Compilation
// ============================================================================

/// A formatted prompt ready for a specific provider.
#[derive(Debug, Clone)]
pub struct FormattedPrompt {
    /// Optional system-level message
    pub system_message: Option<String>,
    /// Conversation messages (role + content pairs)
    pub messages: Vec<FormattedMessage>,
    /// Raw single-string prompt (for completion-style APIs)
    pub raw_prompt: Option<String>,
}

/// A single message in a formatted prompt.
#[derive(Debug, Clone)]
pub struct FormattedMessage {
    /// The role of the message sender (e.g., "system", "user", "assistant")
    pub role: String,
    /// The content of the message
    pub content: String,
}

/// Trait for translating compiled prompts to provider-specific formats.
pub trait LmAdapter: Send + Sync {
    /// Format a compiled prompt for the given provider.
    fn format_for_provider(
        &self,
        compiled: &CompiledPrompt,
        provider_name: &str,
    ) -> FormattedPrompt;
}

/// Formats prompts as chat messages (system + user/assistant demo turns + user query).
pub struct ChatAdapter;

impl LmAdapter for ChatAdapter {
    fn format_for_provider(
        &self,
        compiled: &CompiledPrompt,
        _provider_name: &str,
    ) -> FormattedPrompt {
        let mut messages = Vec::new();

        // System message
        messages.push(FormattedMessage {
            role: "system".to_string(),
            content: compiled.system_prompt.clone(),
        });

        // Demo examples as user/assistant turns
        for example in &compiled.examples {
            let user_content: String = example
                .inputs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(FormattedMessage {
                role: "user".to_string(),
                content: user_content,
            });

            let assistant_content: String = example
                .outputs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(FormattedMessage {
                role: "assistant".to_string(),
                content: assistant_content,
            });
        }

        // User query template
        messages.push(FormattedMessage {
            role: "user".to_string(),
            content: compiled.user_template.clone(),
        });

        FormattedPrompt {
            system_message: Some(compiled.system_prompt.clone()),
            messages,
            raw_prompt: None,
        }
    }
}

/// Formats prompts as a single completion string with delimiters.
pub struct CompletionAdapter;

impl LmAdapter for CompletionAdapter {
    fn format_for_provider(
        &self,
        compiled: &CompiledPrompt,
        _provider_name: &str,
    ) -> FormattedPrompt {
        let mut parts = Vec::new();

        parts.push(compiled.system_prompt.clone());

        for example in &compiled.examples {
            let input_str: String = example
                .inputs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("\n---Input:---\n{}", input_str));

            let output_str: String = example
                .outputs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("---Output:---\n{}", output_str));
        }

        parts.push(format!("\n---Input:---\n{}", compiled.user_template));
        parts.push("---Output:---\n".to_string());

        let raw = parts.join("\n");

        FormattedPrompt {
            system_message: None,
            messages: Vec::new(),
            raw_prompt: Some(raw),
        }
    }
}

/// Formats prompts using a function/tool calling structure for structured output.
pub struct FunctionCallingAdapter;

impl LmAdapter for FunctionCallingAdapter {
    fn format_for_provider(
        &self,
        compiled: &CompiledPrompt,
        _provider_name: &str,
    ) -> FormattedPrompt {
        let mut messages = Vec::new();

        // System message with tool-use framing
        let system_content = format!(
            "{}\n\nYou must respond by calling the `respond` function with the output fields as parameters.",
            compiled.system_prompt
        );
        messages.push(FormattedMessage {
            role: "system".to_string(),
            content: system_content.clone(),
        });

        // Examples as function call demonstrations
        for example in &compiled.examples {
            let user_content: String = example
                .inputs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(FormattedMessage {
                role: "user".to_string(),
                content: user_content,
            });

            // Format as a function call response
            let params: Vec<String> = example
                .outputs
                .iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
                .collect();
            let fn_call = format!("respond({{{}}})", params.join(","));
            messages.push(FormattedMessage {
                role: "assistant".to_string(),
                content: fn_call,
            });
        }

        // User query
        messages.push(FormattedMessage {
            role: "user".to_string(),
            content: compiled.user_template.clone(),
        });

        FormattedPrompt {
            system_message: Some(system_content),
            messages,
            raw_prompt: None,
        }
    }
}

/// Routes provider names to their appropriate adapters.
pub struct AdapterRouter {
    /// Registered (provider_pattern, adapter) pairs
    routes: Vec<(String, Box<dyn LmAdapter>)>,
}

impl AdapterRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
        }
    }

    /// Register an adapter for a provider name pattern.
    ///
    /// The pattern is matched case-insensitively as a substring of the provider name.
    pub fn register(&mut self, provider_pattern: &str, adapter: Box<dyn LmAdapter>) {
        self.routes.push((provider_pattern.to_lowercase(), adapter));
    }

    /// Find the first adapter matching the given provider name.
    pub fn route(&self, provider_name: &str) -> Option<&dyn LmAdapter> {
        let lower = provider_name.to_lowercase();
        for (pattern, adapter) in &self.routes {
            if lower.contains(pattern) {
                return Some(adapter.as_ref());
            }
        }
        None
    }
}
