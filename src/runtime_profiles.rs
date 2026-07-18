//! Functional runtime profiles (V197).
//!
//! A *runtime profile* bundles the settings that make a use case work out of
//! the box — generation temperature, Ollama context window, history depth,
//! whether to retrieve large knowledge — plus the models it is tuned for. It
//! is the "pick one and it just works" layer on top of the raw [`AiConfig`];
//! individual options can still be overridden afterwards.
//!
//! This is distinct from [`crate::profiles::ModelProfile`] (pure sampling
//! parameters) and [`crate::agent_profiles`] (autonomous-agent policy): a
//! runtime profile is the end-to-end chat setup a CLI/GUI user selects.

use crate::config::AiConfig;

/// A named, ready-to-use runtime configuration.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeProfile {
    /// Short identifier (e.g. "mobile").
    pub name: &'static str,
    /// One-line description of the intended use.
    pub description: &'static str,
    /// Generation temperature.
    pub temperature: f32,
    /// Ollama context window override (`None` = auto-size, capped VRAM-safe).
    pub num_ctx: Option<usize>,
    /// How many history messages to keep in context.
    pub max_history: usize,
    /// Whether large injected knowledge should be retrieval-filtered (keeps the
    /// prompt small — essential for the small-window `mobile` profile).
    pub use_knowledge_retrieval: bool,
    /// Model name fragments this profile is tuned for (for recommendation /
    /// docs; not enforced — the user still picks the model).
    pub recommended_models: &'static [&'static str],
}

impl RuntimeProfile {
    /// Apply this profile's tunables to `config` (leaves provider/model/URLs
    /// untouched — those are the user's choice).
    pub fn apply(&self, config: &mut AiConfig) {
        config.temperature = self.temperature;
        config.ollama_num_ctx = self.num_ctx;
        config.max_history_messages = self.max_history;
    }
}

/// All built-in runtime profiles.
pub const BUILTIN_PROFILES: &[RuntimeProfile] = &[
    RuntimeProfile {
        name: "mobile",
        description: "On-device / small models: fast, low-memory, factual. \
                      Conservative window + knowledge retrieval so big context \
                      still fits.",
        temperature: 0.3,
        num_ctx: Some(4096),
        max_history: 8,
        use_knowledge_retrieval: true,
        recommended_models: &[
            "llama3.2:1b",
            "llama3.2:3b",
            "qwen2.5:1.5b",
            "qwen2.5:3b",
            "gemma2:2b",
            "phi3.5",
        ],
    },
    RuntimeProfile {
        name: "local-balanced",
        description: "Everyday local chat on a 7-9B model: good quality with \
                      auto-sized context. The sensible default.",
        temperature: 0.7,
        num_ctx: None,
        max_history: 20,
        use_knowledge_retrieval: true,
        recommended_models: &["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "gemma2:9b"],
    },
    RuntimeProfile {
        name: "local-quality",
        description: "Best local quality on a 14B+ model (needs the VRAM).",
        temperature: 0.7,
        num_ctx: None,
        max_history: 24,
        use_knowledge_retrieval: true,
        recommended_models: &["qwen2.5:14b", "qwen2.5:32b", "gemma2:27b", "llama3.1:70b"],
    },
    RuntimeProfile {
        name: "coding",
        description: "Code generation/editing: low temperature, coder models.",
        temperature: 0.2,
        num_ctx: None,
        max_history: 20,
        use_knowledge_retrieval: true,
        recommended_models: &["qwen2.5-coder", "deepseek-coder", "codellama"],
    },
    RuntimeProfile {
        name: "precise",
        description: "Deterministic, factual answers (low temperature).",
        temperature: 0.2,
        num_ctx: None,
        max_history: 20,
        use_knowledge_retrieval: true,
        recommended_models: &["llama3.1:8b", "qwen2.5:7b"],
    },
    RuntimeProfile {
        name: "creative",
        description: "Imaginative, varied output (high temperature); no \
                      knowledge filtering.",
        temperature: 1.0,
        num_ctx: None,
        max_history: 20,
        use_knowledge_retrieval: false,
        recommended_models: &["llama3.1:8b", "mistral:7b"],
    },
];

/// The profile applied when none is chosen.
pub const DEFAULT_PROFILE: &str = "local-balanced";

/// Look up a built-in profile by name (case-insensitive).
pub fn find(name: &str) -> Option<&'static RuntimeProfile> {
    BUILTIN_PROFILES
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive() {
        assert!(find("mobile").is_some());
        assert!(find("MOBILE").is_some());
        assert!(find("nope").is_none());
        assert!(find(DEFAULT_PROFILE).is_some());
    }

    #[test]
    fn mobile_is_tuned_small_and_safe() {
        let m = find("mobile").unwrap();
        // Low temp for factual reliability, small conservative window, and
        // retrieval on so large knowledge still fits the small context.
        assert!(m.temperature <= 0.4);
        assert_eq!(m.num_ctx, Some(4096));
        assert!(m.use_knowledge_retrieval);
        assert!(m.recommended_models.iter().any(|x| x.contains("1b")));
    }

    #[test]
    fn apply_sets_config_tunables() {
        let mut cfg = AiConfig::default();
        find("mobile").unwrap().apply(&mut cfg);
        assert_eq!(cfg.temperature, 0.3);
        assert_eq!(cfg.ollama_num_ctx, Some(4096));
        assert_eq!(cfg.max_history_messages, 8);
    }
}
