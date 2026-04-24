//! V103: vLLM launch command generators.
//!
//! Helps the user bring up a vLLM server with the right flags without
//! memorizing them. Given a `VLlmLaunchConfig`, produces:
//!
//! - A native `vllm serve ...` shell command (`launch_command()`).
//! - A Docker `docker run ...` equivalent for the official image
//!   (`docker_command()`).
//!
//! Never executes anything. The CLI/butler layer can copy these strings to
//! stdout so the user sees exactly what will run.

use serde::{Deserialize, Serialize};

/// Default Docker image for vLLM (OpenAI-compatible entrypoint).
pub const DEFAULT_VLLM_DOCKER_IMAGE: &str = "vllm/vllm-openai:latest";

/// Default port — vLLM's OpenAI-compatible server listens on 8000.
pub const DEFAULT_VLLM_PORT: u16 = 8000;

/// Configuration for a vLLM launch.
///
/// Fields are `Option<_>` whenever vLLM has a sensible built-in default so
/// the generated command doesn't force every knob. Set only what you need
/// to deviate from defaults.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VLlmLaunchConfig {
    /// HuggingFace repo ID (e.g. `Qwen/Qwen2.5-7B-Instruct`).
    pub repo: String,
    /// Port to listen on. Defaults to 8000.
    pub port: Option<u16>,
    /// Host to bind (default `0.0.0.0`). Set to `127.0.0.1` to restrict.
    pub host: Option<String>,
    /// Tensor parallelism: how many GPUs to shard the model across
    /// (`--tensor-parallel-size`). Only set for multi-GPU setups.
    pub tensor_parallel_size: Option<u8>,
    /// Cap the context window (`--max-model-len`). Useful to bound KV-cache
    /// memory on smaller GPUs.
    pub max_model_len: Option<u32>,
    /// Quantization scheme: `awq`, `gptq`, `fp8`, `bitsandbytes`. Leave
    /// `None` for fp16/bf16.
    pub quantization: Option<String>,
    /// Activation dtype: `auto`, `bfloat16`, `float16`, `float32`.
    pub dtype: Option<String>,
    /// Fraction of GPU memory vLLM may use (0.0–1.0). Default vLLM value
    /// is 0.9; lower this when sharing a GPU with other workloads.
    pub gpu_memory_utilization: Option<f32>,
    /// Whether to enable LoRA hot-swap (`--enable-lora`).
    pub enable_lora: bool,
    /// Pass `--trust-remote-code` — required by some custom HF architectures.
    pub trust_remote_code: bool,
    /// API key expected by the server (`--api-key`). If set, clients must
    /// send `Authorization: Bearer <key>`.
    pub served_api_key: Option<String>,
    /// Whether the HF repo is gated/private and requires `HF_TOKEN`. When
    /// true, the generated command is prefixed with a reminder to export
    /// the token.
    pub hf_token_required: bool,
    /// Enable prefix caching (`--enable-prefix-caching`). Large free
    /// throughput win on workloads that share a common prefix (agentic
    /// loops, multi-agent coordination, RAG with a fixed system prompt).
    pub enable_prefix_caching: bool,
    /// KV-cache dtype (`--kv-cache-dtype`). Accepts `auto`, `fp8`,
    /// `fp8_e5m2`, `fp8_e4m3`. `fp8` roughly doubles KV capacity on
    /// Ada/Hopper GPUs at a small quality cost.
    pub kv_cache_dtype: Option<String>,
    /// Draft model for speculative decoding (`--speculative-model`). A
    /// smaller model that proposes tokens accepted/rejected by the main
    /// model — free latency win for code assist and chat workloads when
    /// draft and target share a tokenizer.
    pub speculative_model: Option<String>,
    /// How many tokens the draft model proposes per step
    /// (`--num-speculative-tokens`). Defaults to vLLM's own default when
    /// `None`.
    pub num_speculative_tokens: Option<u32>,
    /// Override the chat template (`--chat-template`). Some custom
    /// finetunes require explicit templates vLLM can't infer from the
    /// tokenizer.
    pub chat_template: Option<String>,
}

impl VLlmLaunchConfig {
    /// Minimal config: just a repo, everything else default.
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            port: None,
            host: None,
            tensor_parallel_size: None,
            max_model_len: None,
            quantization: None,
            dtype: None,
            gpu_memory_utilization: None,
            enable_lora: false,
            trust_remote_code: false,
            served_api_key: None,
            hf_token_required: false,
            enable_prefix_caching: false,
            kv_cache_dtype: None,
            speculative_model: None,
            num_speculative_tokens: None,
            chat_template: None,
        }
    }

    /// Resolved port (defaults to 8000).
    pub fn effective_port(&self) -> u16 {
        self.port.unwrap_or(DEFAULT_VLLM_PORT)
    }
}

/// Produce a native `vllm serve ...` command string.
///
/// Intended to be pasted into a terminal. Does not shell-escape the repo
/// or api-key fields beyond wrapping them in double quotes — don't feed
/// untrusted input into this.
pub fn vllm_launch_command(cfg: &VLlmLaunchConfig) -> String {
    let mut parts: Vec<String> = Vec::new();
    if cfg.hf_token_required {
        parts.push("HF_TOKEN=$HF_TOKEN".to_string());
    }
    parts.push("vllm".to_string());
    parts.push("serve".to_string());
    parts.push(format!("\"{}\"", cfg.repo));
    parts.push(format!("--port {}", cfg.effective_port()));
    if let Some(host) = &cfg.host {
        parts.push(format!("--host {}", host));
    }
    if let Some(n) = cfg.tensor_parallel_size {
        parts.push(format!("--tensor-parallel-size {}", n));
    }
    if let Some(m) = cfg.max_model_len {
        parts.push(format!("--max-model-len {}", m));
    }
    if let Some(q) = &cfg.quantization {
        parts.push(format!("--quantization {}", q));
    }
    if let Some(d) = &cfg.dtype {
        parts.push(format!("--dtype {}", d));
    }
    if let Some(u) = cfg.gpu_memory_utilization {
        parts.push(format!("--gpu-memory-utilization {:.2}", u));
    }
    if cfg.enable_lora {
        parts.push("--enable-lora".to_string());
    }
    if cfg.enable_prefix_caching {
        parts.push("--enable-prefix-caching".to_string());
    }
    if let Some(kv) = &cfg.kv_cache_dtype {
        parts.push(format!("--kv-cache-dtype {}", kv));
    }
    if let Some(sm) = &cfg.speculative_model {
        parts.push(format!("--speculative-model \"{}\"", sm));
    }
    if let Some(n) = cfg.num_speculative_tokens {
        parts.push(format!("--num-speculative-tokens {}", n));
    }
    if let Some(tpl) = &cfg.chat_template {
        parts.push(format!("--chat-template \"{}\"", tpl));
    }
    if cfg.trust_remote_code {
        parts.push("--trust-remote-code".to_string());
    }
    if let Some(k) = &cfg.served_api_key {
        parts.push(format!("--api-key \"{}\"", k));
    }
    parts.join(" ")
}

/// Produce a `docker run ...` command for the official vLLM image.
///
/// Uses `--gpus all`, mounts `~/.cache/huggingface` into the container so
/// model weights persist between runs, and publishes the vLLM port. Pass
/// `image` = `None` to use the default `vllm/vllm-openai:latest`.
pub fn vllm_docker_command(cfg: &VLlmLaunchConfig, image: Option<&str>) -> String {
    let image = image.unwrap_or(DEFAULT_VLLM_DOCKER_IMAGE);
    let port = cfg.effective_port();

    let mut parts: Vec<String> = Vec::new();
    parts.push("docker run".to_string());
    parts.push("--rm".to_string());
    parts.push("--gpus all".to_string());
    parts.push(format!("-p {}:{}", port, port));
    parts.push("-v \"$HOME/.cache/huggingface\":/root/.cache/huggingface".to_string());
    if cfg.hf_token_required {
        parts.push("-e HF_TOKEN=$HF_TOKEN".to_string());
    }
    parts.push("--ipc=host".to_string()); // required by vLLM for NCCL shared memory
    parts.push(image.to_string());
    parts.push(format!("--model \"{}\"", cfg.repo));
    parts.push(format!("--port {}", port));
    if let Some(n) = cfg.tensor_parallel_size {
        parts.push(format!("--tensor-parallel-size {}", n));
    }
    if let Some(m) = cfg.max_model_len {
        parts.push(format!("--max-model-len {}", m));
    }
    if let Some(q) = &cfg.quantization {
        parts.push(format!("--quantization {}", q));
    }
    if let Some(d) = &cfg.dtype {
        parts.push(format!("--dtype {}", d));
    }
    if let Some(u) = cfg.gpu_memory_utilization {
        parts.push(format!("--gpu-memory-utilization {:.2}", u));
    }
    if cfg.enable_lora {
        parts.push("--enable-lora".to_string());
    }
    if cfg.enable_prefix_caching {
        parts.push("--enable-prefix-caching".to_string());
    }
    if let Some(kv) = &cfg.kv_cache_dtype {
        parts.push(format!("--kv-cache-dtype {}", kv));
    }
    if let Some(sm) = &cfg.speculative_model {
        parts.push(format!("--speculative-model \"{}\"", sm));
    }
    if let Some(n) = cfg.num_speculative_tokens {
        parts.push(format!("--num-speculative-tokens {}", n));
    }
    if let Some(tpl) = &cfg.chat_template {
        parts.push(format!("--chat-template \"{}\"", tpl));
    }
    if cfg.trust_remote_code {
        parts.push("--trust-remote-code".to_string());
    }
    if let Some(k) = &cfg.served_api_key {
        parts.push(format!("--api-key \"{}\"", k));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_launch_command() {
        let cfg = VLlmLaunchConfig::new("Qwen/Qwen2.5-7B-Instruct");
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.starts_with("vllm serve"));
        assert!(cmd.contains("Qwen/Qwen2.5-7B-Instruct"));
        assert!(cmd.contains("--port 8000"));
        assert!(!cmd.contains("--tensor-parallel-size"));
        assert!(!cmd.contains("--enable-lora"));
    }

    #[test]
    fn multi_gpu_command_has_tensor_parallel() {
        let mut cfg = VLlmLaunchConfig::new("meta-llama/Llama-3.1-70B-Instruct");
        cfg.tensor_parallel_size = Some(4);
        cfg.max_model_len = Some(8192);
        cfg.quantization = Some("awq".into());
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--tensor-parallel-size 4"));
        assert!(cmd.contains("--max-model-len 8192"));
        assert!(cmd.contains("--quantization awq"));
    }

    #[test]
    fn gated_repo_prefixes_hf_token_env() {
        let mut cfg = VLlmLaunchConfig::new("meta-llama/Llama-3.1-8B-Instruct");
        cfg.hf_token_required = true;
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.starts_with("HF_TOKEN=$HF_TOKEN vllm serve"));
    }

    #[test]
    fn lora_flag_included_when_enabled() {
        let mut cfg = VLlmLaunchConfig::new("Qwen/Qwen2.5-7B-Instruct");
        cfg.enable_lora = true;
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--enable-lora"));
    }

    #[test]
    fn gpu_memory_utilization_uses_two_decimals() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.gpu_memory_utilization = Some(0.85);
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--gpu-memory-utilization 0.85"), "cmd={}", cmd);
    }

    #[test]
    fn served_api_key_is_quoted() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.served_api_key = Some("secret-123".into());
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--api-key \"secret-123\""));
    }

    #[test]
    fn docker_command_includes_gpu_and_volume() {
        let cfg = VLlmLaunchConfig::new("Qwen/Qwen2.5-7B-Instruct");
        let cmd = vllm_docker_command(&cfg, None);
        assert!(cmd.contains("--gpus all"));
        assert!(cmd.contains("-v \"$HOME/.cache/huggingface\":/root/.cache/huggingface"));
        assert!(cmd.contains("vllm/vllm-openai:latest"));
        assert!(cmd.contains("-p 8000:8000"));
        assert!(cmd.contains("--ipc=host"));
    }

    #[test]
    fn docker_command_custom_image() {
        let cfg = VLlmLaunchConfig::new("x/y");
        let cmd = vllm_docker_command(&cfg, Some("my-registry/vllm:0.6.3"));
        assert!(cmd.contains("my-registry/vllm:0.6.3"));
        assert!(!cmd.contains("vllm/vllm-openai:latest"));
    }

    #[test]
    fn docker_command_forwards_hf_token_when_gated() {
        let mut cfg = VLlmLaunchConfig::new("meta-llama/Llama-3.1-8B-Instruct");
        cfg.hf_token_required = true;
        let cmd = vllm_docker_command(&cfg, None);
        assert!(cmd.contains("-e HF_TOKEN=$HF_TOKEN"));
    }

    #[test]
    fn port_defaults_to_8000() {
        let cfg = VLlmLaunchConfig::new("x/y");
        assert_eq!(cfg.effective_port(), DEFAULT_VLLM_PORT);
    }

    #[test]
    fn custom_port_respected() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.port = Some(9999);
        assert_eq!(cfg.effective_port(), 9999);
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--port 9999"));
        let dcmd = vllm_docker_command(&cfg, None);
        assert!(dcmd.contains("-p 9999:9999"));
    }

    #[test]
    fn prefix_caching_flag_emitted() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.enable_prefix_caching = true;
        assert!(vllm_launch_command(&cfg).contains("--enable-prefix-caching"));
        assert!(vllm_docker_command(&cfg, None).contains("--enable-prefix-caching"));
    }

    #[test]
    fn kv_cache_dtype_flag_emitted() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.kv_cache_dtype = Some("fp8".into());
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--kv-cache-dtype fp8"));
    }

    #[test]
    fn speculative_decoding_flags_emitted() {
        let mut cfg = VLlmLaunchConfig::new("Qwen/Qwen2.5-14B-Instruct");
        cfg.speculative_model = Some("Qwen/Qwen2.5-0.5B-Instruct".into());
        cfg.num_speculative_tokens = Some(5);
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--speculative-model \"Qwen/Qwen2.5-0.5B-Instruct\""));
        assert!(cmd.contains("--num-speculative-tokens 5"));
    }

    #[test]
    fn chat_template_flag_quoted() {
        let mut cfg = VLlmLaunchConfig::new("x/y");
        cfg.chat_template = Some("/opt/templates/custom.jinja".into());
        let cmd = vllm_launch_command(&cfg);
        assert!(cmd.contains("--chat-template \"/opt/templates/custom.jinja\""));
    }

    #[test]
    fn new_flags_absent_by_default() {
        let cfg = VLlmLaunchConfig::new("x/y");
        let cmd = vllm_launch_command(&cfg);
        assert!(!cmd.contains("--enable-prefix-caching"));
        assert!(!cmd.contains("--kv-cache-dtype"));
        assert!(!cmd.contains("--speculative-model"));
        assert!(!cmd.contains("--num-speculative-tokens"));
        assert!(!cmd.contains("--chat-template"));
    }
}
