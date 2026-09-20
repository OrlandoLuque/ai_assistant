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
//! and a `requirements` note that surfaces non-obvious constraints such as
//! needing a particular build of the engine.
//!
//! It also records the **licence** and what that licence lets a redistributor
//! do ([`Redistribution`]). Putting a model file on a drive, or into an
//! installer, is redistributing it, and the families differ sharply: Qwen,
//! Mistral and PrismML's Bonsai are Apache-2.0; Gemma and Llama allow it but
//! charge specific, checkable duties for the privilege. Those values are what
//! the licence texts say, with the date they were read, not a legal opinion --
//! and anything not established reads as "do not ship".

use crate::config::AiProvider;

/// What a model's licence lets a redistributor do.
///
/// # Why this is a field and not a document
///
/// Putting a model file on a drive, or into an installer, is **redistributing
/// it**, and the licences differ on whether that is allowed and on what the
/// distributor owes in return. Until this existed the catalogue recorded no
/// licence at all, so any code asking "may we ship this one?" had nothing to
/// read and the answer lived in somebody's head.
///
/// # What this is not
///
/// These values record what the model card and licence text **say**, with the
/// source and the date of reading in the entry's notes. They are research, not
/// a legal opinion, and deciding what to actually ship is not a decision this
/// file makes. The default is therefore the conservative one: anything not
/// known to be redistributable is treated as not redistributable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Redistribution {
    /// A permissive licence with the usual notice-retention duty: Apache-2.0,
    /// MIT. Shipping the file is allowed; keep the notices.
    Permissive,
    /// Allowed, but the distributor owes specific, checkable things --
    /// forwarding the terms, shipping a NOTICE file, displaying a phrase.
    /// See [`CuratedModel::redistribution_note`].
    WithConditions,
    /// Not established. Treat as "do not ship" until somebody reads the
    /// licence and changes this, which is the safe direction to be wrong in.
    Unknown,
    /// There is nothing to redistribute: the model runs behind somebody's API
    /// and no weights exist to put on a drive. Distinct from `Unknown`, which
    /// means "we did not find out".
    NotDistributed,
}

impl Redistribution {
    /// May the kit put this file on a drive without further human input?
    ///
    /// `WithConditions` is **false** on purpose. The conditions are real
    /// obligations, and a program cannot confirm that a NOTICE file was
    /// written or that "Built with Llama" appears on a page.
    pub fn shippable_unattended(&self) -> bool {
        matches!(self, Redistribution::Permissive)
    }
}

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
    /// The licence the weights are published under, as written on the model
    /// card. The string is the identifier, not a summary.
    pub license: &'static str,
    /// What that licence lets a redistributor do. See [`Redistribution`].
    pub redistribution: Redistribution,
    /// What the distributor owes, in enough detail to act on, plus where it
    /// was read and when. `None` for permissive licences, where retaining the
    /// notices is the whole of it.
    pub redistribution_note: Option<&'static str>,
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
    // PrismML Bonsai — 1-bit Qwen3 derivatives.
    //
    // These ran on the PrismML fork and these entries said so: "upstream
    // llama.cpp does not ship Q1_0". That stopped being true. `GGML_TYPE_Q1_0`
    // is in `ggml-org/llama.cpp` across the CUDA, SYCL and Vulkan backends, and
    // PrismML's own documentation says 1-bit is merged upstream and only
    // ternary (`Q2_0`) still needs the fork.
    //
    // A stale requirement is not a harmless one. This field is shown to the
    // person deciding, so the sentence was telling people they needed to go
    // and fetch a particular fork of llama.cpp to run a model their ordinary
    // build already handles — which, for anyone who is not going to argue with
    // it, means not running it at all.
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Bonsai-27B-Q1_0.gguf",
        display_name: "PrismML Bonsai 27B (1-bit)",
        description:
            "Qwen3-27B at 1 bit/weight. 26.9 B parameters in 3.53 GiB -- the size/capability trade this whole family exists for.",
        parameters: "27B",
        approx_size: "3.80 GB",
        quantization: "Q1_0 (1.125 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Bonsai-27B-gguf"),
        requirements: Some(
            "Needs a recent llama.cpp: `Q1_0` is in upstream now, but a build from before it landed cannot load these.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Tag read 2026-09-20 on prism-ml/Bonsai-27B-gguf itself, not inherited from the sibling entries -- licences are per repository, and the Qwen family is not uniform: Qwen3.8-Flash-Next ships under qwen-community-1.0, which is NOT Apache and carries a separate-licence clause for AI-assistant businesses. Here the chain is clean: Qwen3 is Apache-2.0.",
        ),
    },
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
            "Needs a recent llama.cpp: `Q1_0` is in upstream now, but a build from before it landed cannot load these.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Read 2026-09-16 on the model card. Derived from Qwen3, itself Apache-2.0, so the chain is clean.",
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
        requirements: Some("Needs a recent llama.cpp: `Q1_0` is in upstream now, but a build from before it landed cannot load these."),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Read 2026-09-16 on the model card. Derived from Qwen3, itself Apache-2.0, so the chain is clean."),
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
        requirements: Some("Needs a recent llama.cpp: `Q1_0` is in upstream now, but a build from before it landed cannot load these."),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Read 2026-09-16 on the model card. Derived from Qwen3, itself Apache-2.0, so the chain is clean."),
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
        approx_size: "2.18 GB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf"),
        requirements: Some(
            "Needs the PrismML fork as it stood around mid-2026. Measured 2026-09-18: these load on a July checkout and NOT on the current one, because PrismML broke compatibility with their own earlier ternary format. Also slower than the 1-bit models despite being smaller: 1.56 tokens/second for the 1.7B against 5.13 for the 4B at Q1_0.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Licence tag read 2026-09-18 on the model card. A previous reading said \"not established, the API answers 401\" -- that was wrong, and wrong in the expensive direction: the repository name had been taken from this catalogue's own id field (TernaryBonsai-8B) rather than from Hugging Face (Ternary-Bonsai-8B-gguf). Asking the wrong address and reporting the answer as a property of the thing is how a shippable model gets marked do-not-ship.",
        ),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "TernaryBonsai-4B.gguf",
        display_name: "PrismML Ternary Bonsai 4B",
        description: "Ternary Qwen3-4B — mid-size compression/quality trade-off.",
        parameters: "4B",
        approx_size: "1.07 GB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf"),
        requirements: Some(
            "Needs the PrismML fork as it stood around mid-2026. Measured 2026-09-18: these load on a July checkout and NOT on the current one, because PrismML broke compatibility with their own earlier ternary format. Also slower than the 1-bit models despite being smaller: 1.56 tokens/second for the 1.7B against 5.13 for the 4B at Q1_0.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Licence tag read 2026-09-18 on the model card. A previous reading said \"not established, the API answers 401\" -- that was wrong, and wrong in the expensive direction: the repository name had been taken from this catalogue's own id field (TernaryBonsai-8B) rather than from Hugging Face (Ternary-Bonsai-8B-gguf). Asking the wrong address and reporting the answer as a property of the thing is how a shippable model gets marked do-not-ship.",
        ),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "TernaryBonsai-1.7B.gguf",
        display_name: "PrismML Ternary Bonsai 1.7B",
        description: "Smallest ternary variant — edge / robotics workloads.",
        parameters: "1.7B",
        approx_size: "0.46 GB",
        quantization: "Ternary (~1.6 bpw)",
        source_url: Some("https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf"),
        requirements: Some(
            "Needs the PrismML fork as it stood around mid-2026. Measured 2026-09-18: these load on a July checkout and NOT on the current one, because PrismML broke compatibility with their own earlier ternary format. Also slower than the 1-bit models despite being smaller: 1.56 tokens/second for the 1.7B against 5.13 for the 4B at Q1_0.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Licence tag read 2026-09-18 on the model card. A previous reading said \"not established, the API answers 401\" -- that was wrong, and wrong in the expensive direction: the repository name had been taken from this catalogue's own id field (TernaryBonsai-8B) rather than from Hugging Face (Ternary-Bonsai-8B-gguf). Asking the wrong address and reporting the answer as a property of the thing is how a shippable model gets marked do-not-ship.",
        ),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        display_name: "PrismML Ternary Bonsai 2 27B",
        description:
            "Qwen3.8-27B with ternary weights — reasoning, coding and vision in one file, which is              the argument against carrying a different model per task.",
        parameters: "27B",
        approx_size: "5.95 GB",
        quantization: "PTQ1_0 (1.75 bpw ternary, group 128)",
        source_url: Some("https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf"),
        requirements: Some(
            "Needs the CURRENT PrismML fork: its type is GGML_TYPE_PTQ1_0 = 143, marked Prism-private, so it will never arrive upstream. SPEED IS A STATEMENT ABOUT KERNEL MATURITY, NOT ABOUT THE FORMAT. Measured 2026-09-20 against Bonsai-27B `Q1_0` -- same 26.90 B parameters, same engine, same machine (Intel Iris Xe laptop, fork at PR #198): Vulkan 3.88 pp / 0.23 tg against 5.97 / 1.94; CPU-only 2.43 / 0.05 against 3.70 / 0.81. But the fork has NO x86 SIMD kernel for this type -- the AVX2/AVX-VNNI pull requests (#181, #206) are open, not merged -- which is why the gap widens to 16x on CPU. Third parties report ~96.7 tok/s on an RTX 4090 and ~142 on a 5090, both Ada-or-newer, so on a discrete card the picture is reportedly the opposite. What IS intrinsic: this file is 57% larger than the `Q1_0` of the same model. On an integrated-GPU or CPU-only box, prefer Bonsai-27B-Q1_0 today.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some(
            "Licence tag read 2026-09-18 on the model card, the day after release.",
        ),
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
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Licence tag read 2026-09-16 via the Hugging Face API."),
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
        license: "llama3.1",
        redistribution: Redistribution::WithConditions,
        redistribution_note: Some("Llama 3.1 Community License, read 2026-09-16: ship a copy of the agreement (1.b.i), display \"Built with Llama\" prominently (1.b.i), include the notice \"Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\" (1.b.iii), pass on the Acceptable Use Policy (1.b.iv). Over 700M MAU needs a separate licence from Meta (2)."),
    },
    // ------------------------------------------------------------------
    // Vision (multimodal) — llama.cpp GGUF + mmproj projector files.
    // Order matters in pickers: Qwen2.5-VL is the current open-weight
    // SOTA (OCR / docs / grounding) and should be the default; Gemma 3
    // 4B is the edge tier — slower-but-decent VLM that fits on iGPU /
    // CPU-only laptops where Qwen would not.
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        display_name: "Qwen2.5-VL 7B Instruct (Q4_K_M)",
        description:
            "SOTA open-weight vision-language model. Strong OCR, document, chart, and visual grounding. Apache-2.0. Pair with the matching mmproj.",
        parameters: "7B",
        approx_size: "~4.7 GB + ~1.4 GB mmproj",
        quantization: "Q4_K_M",
        source_url: Some("https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF"),
        requirements: Some(
            "Launch with --mmproj <projector.gguf> from the same repo; pass images as base64.",
        ),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Licence tag read 2026-09-16 via the Hugging Face API."),
    },
    CuratedModel {
        provider: AiProvider::LlamaCpp,
        id: "gemma-3-4b-it-Q4_K_M.gguf",
        display_name: "Gemma 3 4B Instruct (Q4_K_M, vision)",
        description:
            "Edge-tier multimodal — runs on iGPU / CPU-only laptops where Qwen2.5-VL is too heavy. Lower OCR/doc accuracy than Qwen.",
        parameters: "4B",
        approx_size: "~2.6 GB + ~0.8 GB mmproj",
        quantization: "Q4_K_M",
        source_url: Some("https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF"),
        requirements: Some(
            "Launch with --mmproj <projector.gguf>. Use only when 7B+ models do not fit; otherwise prefer Qwen2.5-VL.",
        ),
        license: "gemma",
        redistribution: Redistribution::WithConditions,
        redistribution_note: Some("Gemma Terms of Use section 3.1, read 2026-09-16: forward the section-3.2 use restrictions as an ENFORCEABLE provision, give every recipient a copy of the agreement, and ship a NOTICE text file saying Gemma is subject to the Terms. Commercial distribution is allowed. Download is gated on accepting."),
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
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Licence tag read 2026-09-16 via the Hugging Face API."),
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
        license: "llama3.1",
        redistribution: Redistribution::WithConditions,
        redistribution_note: Some("Llama 3.1 Community License, read 2026-09-16: ship a copy of the agreement (1.b.i), display \"Built with Llama\" prominently (1.b.i), include the notice \"Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\" (1.b.iii), pass on the Acceptable Use Policy (1.b.iv). Over 700M MAU needs a separate licence from Meta (2)."),
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
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Licence tag read 2026-09-16 via the Hugging Face API."),
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
        license: "deepseek-model",
        redistribution: Redistribution::WithConditions,
        redistribution_note: Some("DeepSeek Model License, read 2026-09-16: the paragraph-5 use restrictions MUST be included as an enforceable provision in any agreement (4.a), give recipients a copy of the licence (4.b), retain all notices (4.d). Attachment A forbids military use, harm to minors, discrimination and fully-automated decisions on legal rights, among others."),
    },
    // ------------------------------------------------------------------
    // Ollama — vision (multimodal) models. Same tiering as the LlamaCpp
    // entries above: Qwen2.5-VL is the recommended default, Gemma 3 is
    // the edge fallback. Ollama bundles the projector automatically so
    // there is no separate `--mmproj` step.
    // ------------------------------------------------------------------
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "qwen2.5vl:7b",
        display_name: "Qwen2.5-VL 7B (vision)",
        description:
            "Recommended local vision model. SOTA on OCR / docs / charts / grounding among open weights.",
        parameters: "7B",
        approx_size: "~6 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://ollama.com/library/qwen2.5vl"),
        requirements: Some("Pull with `ollama pull qwen2.5vl:7b`. Pass images via the multimodal API."),
        license: "apache-2.0",
        redistribution: Redistribution::Permissive,
        redistribution_note: Some("Licence tag read 2026-09-16 via the Hugging Face API."),
    },
    CuratedModel {
        provider: AiProvider::Ollama,
        id: "gemma3:4b",
        display_name: "Gemma 3 4B (vision, edge tier)",
        description:
            "Smaller multimodal — fits on integrated GPU / CPU-only laptops. Use when Qwen2.5-VL is too heavy.",
        parameters: "4B",
        approx_size: "~3.3 GB",
        quantization: "Q4_K_M",
        source_url: Some("https://ollama.com/library/gemma3"),
        requirements: Some("Pull with `ollama pull gemma3:4b`. Lower OCR accuracy than qwen2.5vl."),
        license: "gemma",
        redistribution: Redistribution::WithConditions,
        redistribution_note: Some("Gemma Terms of Use section 3.1, read 2026-09-16: forward the section-3.2 use restrictions as an ENFORCEABLE provision, give every recipient a copy of the agreement, and ship a NOTICE text file saying Gemma is subject to the Terms. Commercial distribution is allowed. Download is gated on accepting."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
        license: "proprietary (API only)",
        redistribution: Redistribution::NotDistributed,
        redistribution_note: Some("An API model: there are no weights to put on a drive."),
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
    fn bonsai_entries_say_what_they_actually_need() {
        // This test used to assert that EVERY Bonsai entry demanded the
        // PrismML fork, which is how the claim survived after it stopped
        // being true: the test pinned the sentence rather than checking the
        // fact. `GGML_TYPE_Q1_0` is in `ggml-org/llama.cpp` across the CUDA,
        // SYCL and Vulkan backends, and PrismML's own documentation says 1-bit
        // is merged upstream and only ternary still needs the fork.
        //
        // So the rule under test is the real one, and it separates the two
        // cases. A test that cannot tell them apart cannot catch this drifting
        // again in either direction.
        for m in all_curated_models() {
            let is_ternary = m.id.contains("Ternary");
            let is_bonsai = m.id.contains("Bonsai");
            if !is_bonsai && !is_ternary {
                continue;
            }

            let req = m
                .requirements
                .expect("every Bonsai entry says something about what runs it")
                .to_lowercase();

            if is_ternary {
                assert!(
                    req.contains("prism"),
                    "ternary weights need PrismML's fork; stock builds cannot load them: {}",
                    m.id
                );
            } else {
                assert!(
                    !req.contains("prism"),
                    "Q1_0 is upstream: sending the user after a fork keeps the model                      out of reach just as effectively as not shipping it: {}",
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
    fn every_entry_says_what_its_licence_is() {
        // Putting a model file on a drive is REDISTRIBUTING it. Until this
        // field existed the catalogue recorded no licence at all, so code
        // asking "may we ship this one?" had nothing to read.
        for m in all_curated_models() {
            assert!(!m.license.trim().is_empty(), "{} declares no licence", m.id);
        }
    }

    #[test]
    fn anything_not_plainly_permissive_says_why() {
        // `WithConditions` without the conditions written down is the same as
        // not knowing them, and `Unknown` without a reason cannot be resolved
        // by whoever picks it up next.
        for m in all_curated_models() {
            match m.redistribution {
                Redistribution::Permissive => {}
                _ => {
                    let note = m.redistribution_note.unwrap_or("");
                    assert!(
                        note.len() > 30,
                        "{} is {:?} and says nothing useful about it: {note:?}",
                        m.id,
                        m.redistribution
                    );
                }
            }
        }
    }

    #[test]
    fn only_permissive_ships_without_a_human() {
        // The conditions are real obligations -- forwarding terms as an
        // enforceable provision, shipping a NOTICE file, displaying a phrase
        // on a page. No program can confirm any of that was done, so no
        // program gets to decide it was.
        assert!(Redistribution::Permissive.shippable_unattended());
        assert!(!Redistribution::WithConditions.shippable_unattended());
        assert!(!Redistribution::Unknown.shippable_unattended());
        assert!(!Redistribution::NotDistributed.shippable_unattended());
    }

    #[test]
    fn an_api_model_is_not_merely_unknown() {
        // Two different facts that a single "no" would blur: nothing to ship,
        // versus we did not find out. Only the second is homework.
        for m in all_curated_models() {
            let is_cloud = matches!(
                m.provider,
                AiProvider::OpenAI | AiProvider::Anthropic | AiProvider::Gemini
            );
            if is_cloud {
                assert_eq!(
                    m.redistribution,
                    Redistribution::NotDistributed,
                    "{} runs behind an API; there are no weights to redistribute",
                    m.id
                );
            }
        }
    }

    #[test]
    fn the_notes_say_when_they_were_read() {
        // A licence claim with no date is a claim about an unknown moment.
        // Licences change: Llama and Gemma have both been revised.
        for m in all_curated_models() {
            if let Some(note) = m.redistribution_note {
                if m.redistribution == Redistribution::NotDistributed {
                    continue; // "no weights exist" does not go stale.
                }
                assert!(
                    note.contains("20"),
                    "{} gives no date for when this was read: {note}",
                    m.id
                );
            }
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
