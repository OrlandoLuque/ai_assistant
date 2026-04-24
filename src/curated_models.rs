//! Curated catalog of recommended models per provider.
//!
//! Exposes `suggested_models_for(provider)` → list of hand-picked
//! `CuratedModel` entries. The catalog is a static list, not a live fetch
//! against each provider's `/models` endpoint — the goal is to give users
//! a short, opinionated starting point ("these are known good") rather
//! than an exhaustive registry.
//!
//! Covers the local providers the project targets (Ollama, LM Studio,
//! llama.cpp and the PrismML fork) plus a few cloud families for quick
//! onboarding. Extend by appending to `CURATED_MODELS`.
//!
//! Each entry carries enough metadata for a UI picker: display name,
//! short description, approximate weight size, Hugging Face URL (if any),
//! and a `requirements` note that surfaces non-obvious constraints such
//! as "requires PrismML fork for `Q1_0` quantization".

use crate::config::AiProvider;

/// Curated model entry.
#[derive(Debug, Clone)]
pub struct CuratedModel {
    /// Provider this model is intended for.
    pub provider: AiProvider,
    /// Short identifier (what the user picks, e.g. `Bonsai-8B-Q1_0.gguf`).
    pub id: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// One-line description — what this model is good at.
    pub description: &'static str,
    /// Approximate parameter count (e.g. "8B", "4B", "1.7B").
    pub parameters: &'static str,
    /// Approximate on-disk size (e.g. "1.16 GB", "4.2 GB").
    pub approx_size: &'static str,
    /// Quantization scheme (e.g. "Q1_0", "Q4_K_M", "fp16").
    pub quantization: &'static str,
    /// Hugging Face repo or download URL, if applicable.
    pub source_url: Option<&'static str>,
    /// Special requirements (e.g. "requires PrismML fork of llama.cpp").
    pub requirements: Option<&'static str>,
}

/// Return the curated model list for a given provider.
///
/// For providers without curated entries (cloud APIs where the user picks
/// any model from the provider's dashboard), returns an empty slice.
pub fn suggested_models_for(provider: &AiProvider) -> Vec<CuratedModel> {
    CURATED_MODELS
        .iter()
        .filter(|m| provider_matches(&m.provider, provider))
        .cloned()
        .collect()
}

/// Return every curated entry.
pub fn all_curated_models() -> &'static [CuratedModel] {
    CURATED_MODELS
}

fn provider_matches(catalog: &AiProvider, query: &AiProvider) -> bool {
    use AiProvider::*;
    matches!(
        (catalog, query),
        (Ollama, Ollama)
            | (LMStudio, LMStudio)
            | (LlamaCpp, LlamaCpp)
            | (VLLM, VLLM)
            | (LocalAI, LocalAI)
            | (TextGenWebUI, TextGenWebUI)
            | (KoboldCpp, KoboldCpp)
            | (OpenAI, OpenAI)
            | (Anthropic, Anthropic)
            | (Gemini, Gemini)
    )
}

const CURATED_MODELS: &[CuratedModel] = &[
    // ------------------------------------------------------------------
    // PrismML Bonsai — 1-bit Qwen3 derivatives (llama.cpp / PrismML fork)
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Bonsai-8B-Q1_0.gguf",
        display_name: "PrismML Bonsai 8B (1-bit)",
        description:
            "Qwen3-8B quantized to 1.125 bits/weight (custom Q1_0). Ultra-compressed for edge inference.",
        parameters: "8B",
        approx_size: "1.16 GB",
        quantization: "Q1_0 (1.125 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Bonsai-8B-gguf"),
        requirements: Some(
            "Requires PrismML fork: github.com/PrismML-Eng/llama.cpp (upstream llama.cpp does not ship Q1_0).",
        ),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Bonsai-4B-Q1_0.gguf",
        display_name: "PrismML Bonsai 4B (1-bit)",
        description: "Qwen3-4B 1-bit variant — smaller, faster, lower memory.",
        parameters: "4B",
        approx_size: "~600 MB",
        quantization: "Q1_0 (1.125 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Bonsai-4B-gguf"),
        requirements: Some("Requires PrismML fork of llama.cpp for Q1_0 support."),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Bonsai-1.7B-Q1_0.gguf",
        display_name: "PrismML Bonsai 1.7B (1-bit)",
        description: "Smallest Bonsai 1-bit — fits comfortably on consumer GPUs / integrated graphics.",
        parameters: "1.7B",
        approx_size: "~250 MB",
        quantization: "Q1_0 (1.125 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Bonsai-1.7B-gguf"),
        requirements: Some("Requires PrismML fork of llama.cpp for Q1_0 support."),
    },
    // ------------------------------------------------------------------
    // PrismML Ternary Bonsai — {-1, 0, 1} weights
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "TernaryBonsai-8B.gguf",
        display_name: "PrismML Ternary Bonsai 8B",
        description: "Qwen3-8B with ternary weights — roughly 5x faster than fp16.",
        parameters: "8B",
        approx_size: "~1.8 GB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/collections/prism-ml/ternary-bonsai"),
        requirements: Some("Requires PrismML fork of llama.cpp for ternary kernels."),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "TernaryBonsai-4B.gguf",
        display_name: "PrismML Ternary Bonsai 4B",
        description: "Ternary Qwen3-4B — mid-size compression/quality trade-off.",
        parameters: "4B",
        approx_size: "~900 MB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/collections/prism-ml/ternary-bonsai"),
        requirements: Some("Requires PrismML fork of llama.cpp for ternary kernels."),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "TernaryBonsai-1.7B.gguf",
        display_name: "PrismML Ternary Bonsai 1.7B",
        description: "Smallest ternary variant — edge / robotics workloads.",
        parameters: "1.7B",
        approx_size: "~400 MB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/collections/prism-ml/ternary-bonsai"),
        requirements: Some("Requires PrismML fork of llama.cpp for ternary kernels."),
    },
    // ------------------------------------------------------------------
    // Generic llama.cpp — community-standard GGUF builds (work on upstream)
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        display_name: "Qwen2.5 7B Instruct (Q4_K_M)",
        description: "Strong all-round instruct model, 4-bit quantized.",
        parameters: "7B",
        approx_size: "~4.7 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"),
        requirements: None,
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        display_name: "Llama 3.1 8B Instruct (Q4_K_M)",
        description: "Meta's tool-calling-capable 8B. Good general baseline.",
        parameters: "8B",
        approx_size: "~4.9 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
        requirements: None,
    },
    // ------------------------------------------------------------------
    // Ollama — by name (matches `ollama pull <name>`)
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "qwen2.5:7b-instruct",
        display_name: "Qwen2.5 7B Instruct",
        description: "Strong general-purpose model, good at tool use.",
        parameters: "7B",
        approx_size: "~4.7 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://ollama.com/library/qwen2.5"),
        requirements: None,
    },
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "llama3.1:8b-instruct",
        display_name: "Llama 3.1 8B Instruct",
        description: "Meta Llama 3.1 — solid general baseline with native tool calling.",
        parameters: "8B",
        approx_size: "~4.9 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://ollama.com/library/llama3.1"),
        requirements: None,
    },
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "mistral:7b-instruct",
        display_name: "Mistral 7B Instruct",
        description: "Compact, fast instruct model.",
        parameters: "7B",
        approx_size: "~4.1 GB",
        quantization: "Q4_0",
        source_url: Some("https://ollama.com/library/mistral"),
        requirements: None,
    },
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "deepseek-coder:6.7b",
        display_name: "DeepSeek Coder 6.7B",
        description: "Specialized for code generation and completion.",
        parameters: "6.7B",
        approx_size: "~3.8 GB",
        quantization: "Q4_0",
        source_url: Some("https://ollama.com/library/deepseek-coder"),
        requirements: None,
    },
    // ------------------------------------------------------------------
    // vLLM — HuggingFace repo IDs (GPU-backed, OpenAI-compatible)
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "Qwen/Qwen2.5-7B-Instruct",
        display_name: "Qwen2.5 7B Instruct (vLLM)",
        description:
            "Strong 7B instruct model — single consumer GPU (≥12 GB VRAM). Good default for multi-agent workloads.",
        parameters: "7B",
        approx_size: "~15 GB (fp16)",
        quantization: "fp16 / bf16",
        source_url: Some("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
        requirements: Some("Needs ≥12 GB VRAM at fp16. Use AWQ quantization for ≥8 GB cards."),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "meta-llama/Llama-3.1-8B-Instruct",
        display_name: "Llama 3.1 8B Instruct (vLLM, gated)",
        description: "Meta's tool-calling 8B. Gated — requires HF license acceptance + HF_TOKEN.",
        parameters: "8B",
        approx_size: "~16 GB (fp16)",
        quantization: "fp16",
        source_url: Some("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"),
        requirements: Some("Gated HF repo — accept license at huggingface.co then export HF_TOKEN."),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "Qwen/Qwen2.5-32B-Instruct-AWQ",
        display_name: "Qwen2.5 32B Instruct (AWQ 4-bit, vLLM)",
        description: "32B quality at 4-bit AWQ — fits on a single 24 GB GPU (RTX 3090/4090).",
        parameters: "32B",
        approx_size: "~19 GB (AWQ 4-bit)",
        quantization: "AWQ 4-bit",
        source_url: Some("https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ"),
        requirements: Some("Launch with --quantization awq. Needs ≥24 GB VRAM for KV cache headroom."),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "meta-llama/Llama-3.1-70B-Instruct",
        display_name: "Llama 3.1 70B Instruct (tensor-parallel, vLLM)",
        description:
            "Flagship 70B. Requires multi-GPU tensor parallelism (4x 24 GB or 2x 80 GB). Gated.",
        parameters: "70B",
        approx_size: "~140 GB (fp16)",
        quantization: "fp16 (use AWQ for less VRAM)",
        source_url: Some("https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct"),
        requirements: Some(
            "Multi-GPU only. Launch with --tensor-parallel-size=N where N divides attention heads (64). Gated — needs HF_TOKEN.",
        ),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        display_name: "DeepSeek R1 Distill Qwen 7B (vLLM)",
        description: "Reasoning-tuned distill of DeepSeek R1. Strong chain-of-thought for agentic loops.",
        parameters: "7B",
        approx_size: "~15 GB (fp16)",
        quantization: "fp16",
        source_url: Some(
            "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ),
        requirements: Some("Needs ≥12 GB VRAM at fp16. Ideal for multi-agent reasoning workflows."),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "Qwen/Qwen2.5-Coder-7B-Instruct",
        display_name: "Qwen2.5-Coder 7B Instruct (vLLM)",
        description: "Code-specialized instruct model — excels at multi-file edits and tool-use coding.",
        parameters: "7B",
        approx_size: "~15 GB (fp16)",
        quantization: "fp16",
        source_url: Some("https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"),
        requirements: Some("Needs ≥12 GB VRAM at fp16. Best choice for agentic coding workflows."),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
        display_name: "Llama 3 8B Instruct (FP8, vLLM)",
        description: "FP8-quantized Llama 3 — ~2x throughput vs fp16 on Hopper/Ada GPUs.",
        parameters: "8B",
        approx_size: "~8 GB (FP8)",
        quantization: "FP8",
        source_url: Some("https://huggingface.co/neuralmagic/Meta-Llama-3-8B-Instruct-FP8"),
        requirements: Some(
            "Launch with --quantization fp8. Best on H100/L40S/RTX 4090; falls back on older GPUs.",
        ),
    },
    CuratedModel {
        provider: AiProvider::VLLM,
        id: "BAAI/bge-m3",
        display_name: "BGE-M3 Embeddings (vLLM)",
        description:
            "Multilingual embeddings (100+ languages). Dense + sparse + ColBERT vectors from one model.",
        parameters: "567M",
        approx_size: "~2.3 GB (fp16)",
        quantization: "fp16",
        source_url: Some("https://huggingface.co/BAAI/bge-m3"),
        requirements: Some(
            "Launch with `vllm serve BAAI/bge-m3 --task embed`. Outputs 1024-dim dense vectors by default.",
        ),
    },
    // ------------------------------------------------------------------
    // Cloud anchor entries (optional — helps GUI pickers offer a default)
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::Anthropic,
        id: "claude-opus-4-7",
        display_name: "Claude Opus 4.7",
        description: "Flagship Anthropic model.",
        parameters: "—",
        approx_size: "—",
        quantization: "n/a",
        source_url: Some("https://docs.anthropic.com/en/docs/about-claude/models"),
        requirements: Some("Cloud — requires ANTHROPIC_API_KEY."),
    },
    CuratedModel {
        provider: AiProvider::OpenAI,
        id: "gpt-4o",
        display_name: "GPT-4o",
        description: "OpenAI flagship multimodal model.",
        parameters: "—",
        approx_size: "—",
        quantization: "n/a",
        source_url: Some("https://platform.openai.com/docs/models"),
        requirements: Some("Cloud — requires OPENAI_API_KEY."),
    },
    CuratedModel {
        provider: AiProvider::Gemini,
        id: "gemini-2.0-flash",
        display_name: "Gemini 2.0 Flash",
        description: "Google's fast multimodal model.",
        parameters: "—",
        approx_size: "—",
        quantization: "n/a",
        source_url: Some("https://ai.google.dev/gemini-api/docs/models/gemini"),
        requirements: Some("Cloud — requires GOOGLE_API_KEY."),
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llamacpp_has_bonsai_entries() {
        let models = suggested_models_for(&AiProvider::LlamaCpp);
        assert!(
            models.iter().any(|m| m.id.starts_with("Bonsai-8B")),
            "expected Bonsai-8B in llama.cpp catalog"
        );
        assert!(
            models.iter().any(|m| m.id.contains("Ternary")),
            "expected a Ternary Bonsai in llama.cpp catalog"
        );
    }

    #[test]
    fn bonsai_entries_flag_prismml_fork_requirement() {
        for m in all_curated_models() {
            if m.id.contains("Bonsai") || m.id.contains("Ternary") {
                let req = m
                    .requirements
                    .expect("Bonsai/Ternary must declare PrismML requirement");
                assert!(
                    req.to_lowercase().contains("prismml") || req.to_lowercase().contains("prism"),
                    "requirement should mention PrismML fork: {}",
                    m.id
                );
            }
        }
    }

    #[test]
    fn ollama_has_entries() {
        let models = suggested_models_for(&AiProvider::Ollama);
        assert!(models.len() >= 3, "expected at least 3 Ollama entries");
    }

    #[test]
    fn cloud_entries_flag_api_key() {
        for m in all_curated_models() {
            if m.provider.is_cloud() {
                let req = m.requirements.expect("cloud models must flag API key");
                assert!(
                    req.to_lowercase().contains("api_key") || req.to_lowercase().contains("key")
                );
            }
        }
    }

    #[test]
    fn no_empty_ids_or_display_names() {
        for m in all_curated_models() {
            assert!(!m.id.is_empty());
            assert!(!m.display_name.is_empty());
            assert!(!m.description.is_empty());
        }
    }

    #[test]
    fn all_curated_models_is_nonempty() {
        assert!(!all_curated_models().is_empty());
    }

    #[test]
    fn vllm_catalog_has_entries() {
        let models = suggested_models_for(&AiProvider::VLLM);
        assert!(
            models.len() >= 6,
            "expected at least 6 vLLM curated models, got {}",
            models.len()
        );
    }

    #[test]
    fn vllm_entries_use_huggingface_repo_ids() {
        // HF repo IDs are of the form `org/name` — no local filename,
        // no Ollama `<name>:<tag>` syntax.
        for m in suggested_models_for(&AiProvider::VLLM) {
            assert!(
                m.id.contains('/') && !m.id.ends_with(".gguf") && !m.id.contains(':'),
                "vLLM id must look like a HF repo (org/name): {}",
                m.id
            );
        }
    }

    #[test]
    fn vllm_has_a_coder_entry() {
        let models = suggested_models_for(&AiProvider::VLLM);
        assert!(
            models.iter().any(|m| m.id.to_lowercase().contains("coder")),
            "vLLM catalog should include a coding-specialist model"
        );
    }

    #[test]
    fn vllm_has_an_embedding_entry() {
        let models = suggested_models_for(&AiProvider::VLLM);
        assert!(
            models
                .iter()
                .any(|m| m.display_name.to_lowercase().contains("embed")
                    || m.id.to_lowercase().contains("bge")),
            "vLLM catalog should include an embedding model"
        );
    }

    #[test]
    fn vllm_gated_entries_flag_hf_token() {
        for m in suggested_models_for(&AiProvider::VLLM) {
            if m.id.starts_with("meta-llama/") {
                let req = m.requirements.expect("gated vLLM entry needs requirements");
                assert!(
                    req.to_lowercase().contains("hf_token") || req.to_lowercase().contains("gated"),
                    "gated repo must flag HF_TOKEN requirement: {}",
                    m.id
                );
            }
        }
    }
}
