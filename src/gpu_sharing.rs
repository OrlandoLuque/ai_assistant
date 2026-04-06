// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! GPU Sharing Network — distributed LLM inference across a peer-to-peer GPU mesh.
//!
//! This module orchestrates the GPU sharing network where nodes can provide their
//! local GPU capacity for LLM inference, request inference from remote peers,
//! or act as gateways routing traffic between consumers and providers.
//!
//! # Architecture
//!
//! - **Provider**: Exposes local GPU models for remote inference requests
//! - **Gateway**: Routes requests to the best provider based on goal (fastest, cheapest)
//! - **Both**: Acts as both provider and gateway simultaneously
//!
//! # Security
//!
//! - Commit-reveal protocol for fair receipt ID generation
//! - Triple-signed transaction receipts (provider + requester + auditor)
//! - GPU benchmark challenges for Sybil defense
//! - PII tokenization before sending prompts to remote nodes
//!
//! # Credit System
//!
//! Credits are transferred between nodes for inference work. Transactions have
//! a maturity period before credits become available, and an auditor randomly
//! verifies a percentage of transactions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::collusion_detection::CollusionDetector;
use crate::credit_system::CreditManager;
use crate::dynamic_pricing::DynamicPricer;

// ============================================================================
// Configuration
// ============================================================================

/// Top-level configuration for the GPU sharing network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSharingConfig {
    /// Whether GPU sharing is enabled.
    pub enabled: bool,
    /// Operating mode: Provider, Gateway, or Both.
    pub mode: SharingMode,
    /// Maximum concurrent inference jobs to provide to the network.
    pub max_concurrent_provide: usize,
    /// Maximum concurrent inference requests to send to the network.
    pub max_concurrent_request: usize,
    /// Stake amount in credits required to join the network.
    pub stake_amount: f64,
    /// Seconds before earned credits mature and become spendable.
    pub credit_maturity_secs: u64,
    /// Hours between GPU benchmark challenge rounds.
    pub challenge_interval_hours: u64,
    /// Privacy level for outgoing prompts: "none", "tokenize", "aggressive", "paranoid".
    pub privacy_level: String,
    /// Percentage of transactions (0–100) that auditors verify.
    pub auditor_verify_percent: u32,
    /// Transaction fee percentage taken by the network pool.
    pub transaction_fee_percent: f64,
    /// Request routing configuration.
    pub routing: RoutingConfig,
    /// Pricing configuration for provided inference.
    pub pricing: PricingConfig,
    /// Network access configuration.
    pub network: GpuNetworkConfig,
}

impl Default for GpuSharingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SharingMode::Both,
            max_concurrent_provide: 3,
            max_concurrent_request: 5,
            stake_amount: 50.0,
            credit_maturity_secs: 3600,
            challenge_interval_hours: 24,
            privacy_level: "tokenize".to_string(),
            auditor_verify_percent: 20,
            transaction_fee_percent: 5.0,
            routing: RoutingConfig::default(),
            pricing: PricingConfig::default(),
            network: GpuNetworkConfig::default(),
        }
    }
}

/// Operating mode for a GPU sharing node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharingMode {
    /// Only provides local GPU for remote inference.
    Provider,
    /// Only routes requests to remote providers.
    Gateway,
    /// Both provides and routes (default).
    Both,
}

impl std::fmt::Display for SharingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SharingMode::Provider => write!(f, "Provider"),
            SharingMode::Gateway => write!(f, "Gateway"),
            SharingMode::Both => write!(f, "Both"),
        }
    }
}

/// Request routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Routing strategy for incoming requests.
    pub strategy: RoutingStrategy,
    /// Timeout in seconds for queued requests before falling back.
    pub queue_timeout_secs: u64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::Auto,
            queue_timeout_secs: 30,
        }
    }
}

/// Strategy for routing inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Only use local GPU, never send to network.
    LocalOnly,
    /// Prefer local GPU, fall back to network when busy.
    LocalPriority,
    /// Only send to network, never use local GPU.
    NetworkOnly,
    /// Prefer network, fall back to local when no peers available.
    NetworkPriority,
    /// Route to the cheapest provider.
    Cheapest,
    /// Route to the fastest provider.
    Fastest,
    /// Automatically decide based on load and credits.
    Auto,
}

/// Pricing configuration for inference provided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// Pricing mode: "fixed" or "dynamic".
    pub mode: String,
    /// Base price per 1K tokens.
    pub base_price: f64,
    /// Minimum price per 1K tokens (floor for dynamic pricing).
    pub min_price: f64,
    /// Maximum price per 1K tokens (ceiling for dynamic pricing).
    pub max_price: f64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            mode: "dynamic".to_string(),
            base_price: 1.0,
            min_price: 0.1,
            max_price: 10.0,
        }
    }
}

/// Network access configuration.
///
/// Named `GpuNetworkConfig` to avoid conflict with `distributed_network::NetworkConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuNetworkConfig {
    /// Network mode: "public" or "private".
    pub mode: String,
    /// Whitelisted node IDs (only relevant in private mode).
    pub whitelist: Vec<String>,
    /// Whether the network requires an invite to join.
    pub invite_only: bool,
}

impl Default for GpuNetworkConfig {
    fn default() -> Self {
        Self {
            mode: "public".to_string(),
            whitelist: Vec::new(),
            invite_only: false,
        }
    }
}

// ============================================================================
// Capability Advertisement
// ============================================================================

/// Description of a node's GPU hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapability {
    /// GPU vendor (e.g. "NVIDIA", "AMD", "Apple").
    pub vendor: String,
    /// GPU model name (e.g. "RTX 4090", "RX 7900 XTX").
    pub model: String,
    /// Video RAM in megabytes.
    pub vram_mb: u32,
    /// CUDA compute capability (e.g. "8.9") or None for non-NVIDIA.
    pub compute_capability: Option<String>,
    /// Driver version string.
    pub driver_version: Option<String>,
}

/// A model offered by a provider node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOffer {
    /// Model name (e.g. "llama3.1:70b").
    pub model_name: String,
    /// Quantization format (e.g. "Q4_K_M", "Q8_0", "FP16").
    pub quantization: String,
    /// Maximum context length supported.
    pub max_context: usize,
    /// Estimated generation speed in tokens per second.
    pub estimated_tokens_per_second: f32,
    /// Whether the model is currently loaded in VRAM.
    pub loaded: bool,
}

/// Capability advertisement broadcast by a node to the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilityAd {
    /// Unique node identifier.
    pub node_id: String,
    /// GPU hardware description.
    pub gpu: GpuCapability,
    /// Models available on this node.
    pub models: Vec<ModelOffer>,
    /// Price per 1K tokens in credits.
    pub price_per_1k_tokens: f64,
    /// Overall availability score (0.0 = offline, 1.0 = fully available).
    pub availability: f32,
    /// Global reputation score (0.0 = untrusted, 1.0 = perfect).
    pub reputation: f32,
    /// Per-model reputation scores.
    pub reputation_by_model: HashMap<String, f32>,
    /// Current GPU load (0.0 = idle, 1.0 = fully loaded).
    pub current_load: f32,
    /// Maximum concurrent inference slots.
    pub max_concurrent: usize,
    /// Unix timestamp of last update.
    pub updated_at: u64,
}

// ============================================================================
// Transaction Receipt (triple-signed, with commit-reveal)
// ============================================================================

/// Result of an auditor's verification of a transaction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditResult {
    /// Audit is pending (not yet verified).
    Pending,
    /// Auditor verified the transaction as correct.
    Verified,
    /// Auditor found the transaction to be fraudulent.
    Failed,
    /// Transaction was not selected for audit.
    Skipped,
}

/// Triple-signed receipt for a completed inference transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Unique receipt identifier (derived from commit-reveal nonces).
    pub id: String,
    /// Original request ID.
    pub request_id: String,
    /// Node ID of the provider that performed inference.
    pub provider_node: String,
    /// Node ID of the requester that submitted the prompt.
    pub requester_node: String,
    /// Node ID of the auditor that (optionally) verified the transaction.
    pub auditor_node: String,
    /// Model used for inference.
    pub model: String,
    /// Number of input tokens processed.
    pub tokens_in: u64,
    /// Number of output tokens generated.
    pub tokens_out: u64,
    /// Credits transferred from requester to provider.
    pub credits_transferred: f64,
    /// Fee amount taken by the network pool.
    pub fee_amount: f64,
    /// Unix timestamp of the transaction.
    pub timestamp: u64,
    /// Provider's cryptographic signature.
    pub provider_signature: Vec<u8>,
    /// Requester's cryptographic signature.
    pub requester_signature: Vec<u8>,
    /// Auditor's cryptographic signature.
    pub auditor_signature: Vec<u8>,
    /// Result of the audit (if any).
    pub audit_result: AuditResult,
    /// Unix timestamp when credits mature and become spendable.
    pub maturity_time: u64,
}

/// State machine for the commit-reveal protocol.
///
/// Ensures fair receipt ID generation: the requester commits to a nonce hash
/// before the provider reveals their nonce, preventing either party from
/// manipulating the receipt ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRevealState {
    /// Hash of the requester's nonce (commitment phase).
    pub requester_commitment: Vec<u8>,
    /// Provider's nonce (revealed after commitment).
    pub provider_nonce: Option<u64>,
    /// Requester's nonce (revealed after provider nonce).
    pub requester_nonce: Option<u64>,
    /// Computed receipt ID (available after both nonces revealed).
    pub receipt_id: Option<String>,
}

impl CommitRevealState {
    /// Create a new commitment from a requester nonce.
    /// The commitment is a simple hash of the nonce bytes.
    pub fn new_commitment(nonce: u64) -> Self {
        let bytes = nonce.to_le_bytes();
        // Simple commitment: FNV-1a hash of the nonce bytes
        let mut hash = 0xcbf29ce484222325u64;
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Self {
            requester_commitment: hash.to_le_bytes().to_vec(),
            provider_nonce: None,
            requester_nonce: None,
            receipt_id: None,
        }
    }

    /// Set the provider's nonce (after seeing the requester's commitment).
    pub fn set_provider_nonce(&mut self, nonce: u64) {
        self.provider_nonce = Some(nonce);
    }

    /// Reveal the requester's nonce and compute the receipt ID.
    /// Returns the receipt ID if both nonces are now available, or None if
    /// the provider nonce hasn't been set yet.
    pub fn reveal_requester_nonce(&mut self, nonce: u64) -> Option<String> {
        self.requester_nonce = Some(nonce);
        if let Some(provider_nonce) = self.provider_nonce {
            let id = Self::compute_receipt_id(nonce, provider_nonce, "");
            self.receipt_id = Some(id.clone());
            Some(id)
        } else {
            None
        }
    }

    /// Compute a deterministic receipt ID from both nonces and the request ID.
    pub fn compute_receipt_id(nonce_r: u64, nonce_p: u64, request_id: &str) -> String {
        // Combine nonces and request_id into a deterministic hash
        let mut hash = 0xcbf29ce484222325u64;
        for &b in &nonce_r.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for &b in &nonce_p.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for &b in request_id.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("rcpt-{:016x}", hash)
    }

    /// Verify that a revealed nonce matches the original commitment.
    pub fn verify_commitment(&self, nonce: u64) -> bool {
        let bytes = nonce.to_le_bytes();
        let mut hash = 0xcbf29ce484222325u64;
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        self.requester_commitment == hash.to_le_bytes().to_vec()
    }
}

// ============================================================================
// Inference Request/Response
// ============================================================================

/// Goal for an inference request — determines provider selection strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RequestGoal {
    /// Select the provider with lowest latency.
    Fastest,
    /// Select the provider with lowest price.
    Cheapest,
    /// Balance between speed and cost.
    Balanced,
    /// Let the system decide based on context.
    Auto,
}

/// An inference request to be routed to a provider.
///
/// Use `InferenceRequest::new()` to create — it automatically applies PII masking.
/// The PII token map is NOT included in the request (stays local).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Model to use for inference.
    pub model: String,
    /// Prompt text (PII-masked automatically by constructor).
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature for generation.
    pub temperature: f32,
    /// Maximum credits to spend on this request.
    pub budget_credits: f64,
    /// Goal for provider selection.
    pub goal: RequestGoal,
    /// Commitment hash (hash of requester nonce) for commit-reveal.
    pub commitment: Vec<u8>,
}

impl InferenceRequest {
    /// Create a new request with automatic PII masking.
    ///
    /// Returns (request, pii_map). The pii_map stays LOCAL — use it to unmask
    /// the response when it comes back. NEVER send pii_map over the network.
    pub fn new(
        model: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        budget_credits: f64,
        goal: RequestGoal,
        nonce: u64,
    ) -> (Self, crate::pii_tokenizer::PiiTokenMap) {
        let mut tokenizer = crate::pii_tokenizer::PiiTokenizer::with_default();
        let (masked_prompt, pii_map) = tokenizer.mask(prompt);

        let commitment = {
            // Simple hash of nonce for commit-reveal
            let bytes = nonce.to_le_bytes();
            let mut hash = vec![0u8; 32];
            for (i, &b) in bytes.iter().enumerate() {
                hash[i % 32] ^= b;
            }
            hash
        };

        (
            Self {
                request_id: uuid::Uuid::new_v4().to_string(),
                model: model.to_string(),
                prompt: masked_prompt,
                max_tokens,
                temperature,
                budget_credits,
                goal,
                commitment,
            },
            pii_map,
        )
    }
}

/// Response from a provider after completing inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Original request ID.
    pub request_id: String,
    /// Generated response text.
    pub response: String,
    /// Number of input tokens processed.
    pub tokens_in: u64,
    /// Number of output tokens generated.
    pub tokens_out: u64,
    /// End-to-end latency in milliseconds.
    pub latency_ms: u64,
    /// Provider's nonce for commit-reveal protocol.
    pub provider_nonce: u64,
    /// Serialized compute proof (if verification is enabled).
    pub proof: Option<Vec<u8>>,
}

// ============================================================================
// Provider Selection
// ============================================================================

/// Selects providers from capability advertisements based on request goals.
pub struct ProviderSelector;

impl ProviderSelector {
    /// Select the top-K providers for a given model and goal.
    ///
    /// Filters advertisements to those offering the requested model, then
    /// scores and ranks them according to the goal.
    pub fn select_providers<'a>(
        ads: &'a [NodeCapabilityAd],
        model: &str,
        goal: &RequestGoal,
        top_k: usize,
    ) -> Vec<&'a NodeCapabilityAd> {
        let mut candidates: Vec<&NodeCapabilityAd> = ads
            .iter()
            .filter(|ad| {
                ad.models.iter().any(|m| m.model_name == model)
                    && ad.reputation >= 0.3
                    && ad.availability > 0.0
            })
            .collect();

        candidates.sort_by(|a, b| {
            let score_a = Self::score_provider(a, goal);
            let score_b = Self::score_provider(b, goal);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(top_k);
        candidates
    }

    /// Score a provider advertisement for a given goal.
    ///
    /// Higher scores are better. The scoring weights depend on the goal:
    /// - Fastest: heavily weights speed (tokens/s) and low load
    /// - Cheapest: heavily weights low price
    /// - Balanced: equal weight to speed, price, and reputation
    /// - Auto: same as Balanced
    pub fn score_provider(ad: &NodeCapabilityAd, goal: &RequestGoal) -> f64 {
        // Find the best model offer for speed estimation
        let best_tps = ad
            .models
            .iter()
            .map(|m| m.estimated_tokens_per_second)
            .fold(0.0f32, f32::max);

        let speed_score = (best_tps / 100.0).min(1.0) as f64;
        let price_score = 1.0 / (1.0 + ad.price_per_1k_tokens);
        let rep_score = ad.reputation as f64;
        let load_score = 1.0 - ad.current_load as f64;
        let avail_score = ad.availability as f64;

        match goal {
            RequestGoal::Fastest => {
                speed_score * 0.5 + load_score * 0.25 + rep_score * 0.15 + avail_score * 0.10
            }
            RequestGoal::Cheapest => {
                price_score * 0.6 + rep_score * 0.2 + avail_score * 0.1 + speed_score * 0.1
            }
            RequestGoal::Balanced | RequestGoal::Auto => {
                speed_score * 0.25
                    + price_score * 0.25
                    + rep_score * 0.25
                    + load_score * 0.15
                    + avail_score * 0.10
            }
        }
    }
}

// ============================================================================
// GPU Challenge (Sybil defense)
// ============================================================================

/// A GPU benchmark challenge used for Sybil defense.
///
/// Nodes must periodically prove they actually have the GPU they claim
/// by completing a timed computation challenge calibrated to their
/// claimed VRAM and compute capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkChallenge {
    /// Matrix dimension for the benchmark computation.
    pub matrix_size: usize,
    /// Minimum GFLOPS expected for the claimed hardware.
    pub expected_min_gflops: f32,
    /// Time limit in milliseconds to complete the challenge.
    pub time_limit_ms: u64,
    /// Random nonce to prevent precomputation.
    pub nonce: u64,
}

/// Result submitted by a node after completing a GPU challenge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuChallengeResult {
    /// Hash of the computation result (proves work was done).
    pub result_hash: Vec<u8>,
    /// GFLOPS achieved during the benchmark.
    pub gflops_achieved: f32,
    /// Time taken in milliseconds.
    pub time_ms: u64,
}

impl GpuBenchmarkChallenge {
    /// Generate a challenge calibrated to the claimed VRAM.
    ///
    /// Larger VRAM claims require larger matrix sizes and higher GFLOPS.
    pub fn generate(claimed_vram_mb: u32) -> Self {
        // Scale matrix size with VRAM: ~512 per 4GB
        let matrix_size = ((claimed_vram_mb as f64 / 4096.0) * 512.0)
            .max(128.0)
            .min(4096.0) as usize;

        // Expected GFLOPS scales with VRAM (rough heuristic)
        let expected_min_gflops = (claimed_vram_mb as f32 / 1024.0) * 2.0;

        // Time limit: generous but bounded
        let time_limit_ms = 10_000; // 10 seconds

        // Nonce from system time
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        Self {
            matrix_size,
            expected_min_gflops,
            time_limit_ms,
            nonce,
        }
    }

    /// Verify a challenge result.
    ///
    /// Checks that:
    /// 1. The result was computed within the time limit
    /// 2. The achieved GFLOPS meets the minimum threshold
    /// 3. The result hash is non-empty (basic sanity check)
    pub fn verify(&self, result: &GpuChallengeResult) -> bool {
        // Must complete within time limit
        if result.time_ms > self.time_limit_ms {
            return false;
        }
        // Must achieve minimum GFLOPS
        if result.gflops_achieved < self.expected_min_gflops {
            return false;
        }
        // Must have a non-empty result hash
        if result.result_hash.is_empty() {
            return false;
        }
        true
    }
}

// ============================================================================
// Routing Decision
// ============================================================================

/// Result of a routing decision for an inference request.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    /// Process locally on this node's GPU.
    Local,
    /// Queue locally and wait for a slot.
    LocalWait,
    /// Send to the network for remote processing.
    Network,
    /// Cannot route — error description.
    Error(String),
}

/// Decide how to route a request based on strategy, queue state, and credits.
///
/// # Arguments
/// * `strategy` — the configured routing strategy
/// * `local_queue` — number of requests currently queued locally
/// * `credits` — available credit balance
/// * `prompt_tokens` — estimated number of tokens in the prompt
pub fn route_request(
    strategy: &RoutingStrategy,
    local_queue: usize,
    credits: f64,
    prompt_tokens: usize,
) -> RouteDecision {
    match strategy {
        RoutingStrategy::LocalOnly => {
            if local_queue < 10 {
                RouteDecision::Local
            } else {
                RouteDecision::LocalWait
            }
        }
        RoutingStrategy::LocalPriority => {
            if local_queue < 5 {
                RouteDecision::Local
            } else if credits > 0.0 {
                RouteDecision::Network
            } else {
                RouteDecision::LocalWait
            }
        }
        RoutingStrategy::NetworkOnly => {
            if credits > 0.0 {
                RouteDecision::Network
            } else {
                RouteDecision::Error("No credits for network inference".to_string())
            }
        }
        RoutingStrategy::NetworkPriority => {
            if credits > 0.0 {
                RouteDecision::Network
            } else {
                RouteDecision::Local
            }
        }
        RoutingStrategy::Cheapest => {
            // If local is free and available, use it; otherwise network
            if local_queue < 3 {
                RouteDecision::Local
            } else if credits > 0.0 {
                RouteDecision::Network
            } else {
                RouteDecision::LocalWait
            }
        }
        RoutingStrategy::Fastest => {
            // If local queue is short, local is likely faster; otherwise network
            if local_queue == 0 {
                RouteDecision::Local
            } else if credits > 0.0 {
                RouteDecision::Network
            } else {
                RouteDecision::LocalWait
            }
        }
        RoutingStrategy::Auto => {
            // Heuristic: consider queue depth, credits, and prompt size
            let local_busy = local_queue > 3;
            let has_credits = credits > 0.0;
            let large_prompt = prompt_tokens > 2000;

            if !local_busy {
                RouteDecision::Local
            } else if has_credits && (local_busy || large_prompt) {
                RouteDecision::Network
            } else if local_busy && !has_credits {
                RouteDecision::LocalWait
            } else {
                RouteDecision::Local
            }
        }
    }
}

// ============================================================================
// Integrated GPU Sharing Node
// ============================================================================

/// Type alias for compute proofs used in inference verification.
pub type InferenceProof = crate::compute_proof::ComputeProof;

/// Complete GPU sharing node state combining all subsystems.
///
/// Composes `CreditManager`, `DynamicPricer`, `CollusionDetector`, and
/// `ComputeProof` into a single coherent node that can participate in the
/// GPU sharing network.
pub struct GpuSharingNode {
    /// Node configuration.
    pub config: GpuSharingConfig,
    /// Credit balance and escrow manager for this node.
    pub credits: CreditManager,
    /// Dynamic pricer tracking supply/demand for pricing adjustments.
    pub pricer: DynamicPricer,
    /// Collusion detector monitoring transaction patterns.
    pub collusion: CollusionDetector,
    /// Advertised capabilities (set when the node joins the network).
    pub capabilities: Option<NodeCapabilityAd>,
    /// Currently active inference requests.
    pub active_requests: Vec<InferenceRequest>,
    /// Transaction receipts pending maturity.
    pub receipts_pending: Vec<TransactionReceipt>,
}

impl GpuSharingNode {
    /// Create a new GPU sharing node with the given config and node ID.
    pub fn new(config: GpuSharingConfig, node_id: &str) -> Self {
        let pricer = DynamicPricer::new(
            config.pricing.base_price,
            config.pricing.min_price,
            config.pricing.max_price,
        );
        Self {
            config,
            credits: CreditManager::new(node_id),
            pricer,
            collusion: CollusionDetector::new(10000),
            capabilities: None,
            active_requests: Vec::new(),
            receipts_pending: Vec::new(),
        }
    }

    /// Get the current dynamic price per 1K tokens.
    pub fn current_price(&self) -> f64 {
        self.pricer.current_price()
    }

    /// Check whether this node can afford a transaction of the given cost.
    pub fn can_afford(&self, cost: f64) -> bool {
        self.credits.effective_balance() >= cost
    }

    /// Record a transaction for collusion monitoring.
    pub fn record_transaction(&mut self, from: &str, to: &str, _amount: f64, _receipt_id: &str) {
        self.collusion.record_transaction(from, to);
        // Credits handled via escrow in CreditManager
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = GpuSharingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.mode, SharingMode::Both);
        assert_eq!(config.max_concurrent_provide, 3);
        assert_eq!(config.max_concurrent_request, 5);
        assert!((config.stake_amount - 50.0).abs() < f64::EPSILON);
        assert_eq!(config.credit_maturity_secs, 3600);
        assert_eq!(config.challenge_interval_hours, 24);
        assert_eq!(config.privacy_level, "tokenize");
        assert_eq!(config.auditor_verify_percent, 20);
        assert!((config.transaction_fee_percent - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.routing.strategy, RoutingStrategy::Auto);
        assert_eq!(config.pricing.mode, "dynamic");
        assert_eq!(config.network.mode, "public");
    }

    #[test]
    fn test_commit_reveal_roundtrip() {
        let requester_nonce = 12345u64;
        let provider_nonce = 67890u64;

        // Step 1: Requester creates commitment
        let mut state = CommitRevealState::new_commitment(requester_nonce);
        assert!(state.provider_nonce.is_none());
        assert!(state.requester_nonce.is_none());
        assert!(state.receipt_id.is_none());

        // Step 2: Provider sets their nonce
        state.set_provider_nonce(provider_nonce);
        assert_eq!(state.provider_nonce, Some(provider_nonce));

        // Step 3: Requester reveals their nonce
        let receipt_id = state.reveal_requester_nonce(requester_nonce);
        assert!(receipt_id.is_some());
        assert!(receipt_id.as_ref().unwrap().starts_with("rcpt-"));

        // Verify commitment matches
        assert!(state.verify_commitment(requester_nonce));
        // Wrong nonce should not verify
        assert!(!state.verify_commitment(99999));

        // Determinism: same inputs produce same receipt ID
        let id2 = CommitRevealState::compute_receipt_id(requester_nonce, provider_nonce, "");
        assert_eq!(receipt_id.unwrap(), id2);
    }

    #[test]
    fn test_commit_reveal_without_provider_nonce() {
        let mut state = CommitRevealState::new_commitment(42);
        // Reveal without provider nonce should return None
        let result = state.reveal_requester_nonce(42);
        assert!(result.is_none());
    }

    fn make_test_ad(
        node_id: &str,
        tps: f32,
        price: f64,
        reputation: f32,
        load: f32,
    ) -> NodeCapabilityAd {
        NodeCapabilityAd {
            node_id: node_id.to_string(),
            gpu: GpuCapability {
                vendor: "NVIDIA".to_string(),
                model: "RTX 4090".to_string(),
                vram_mb: 24576,
                compute_capability: Some("8.9".to_string()),
                driver_version: Some("555.42".to_string()),
            },
            models: vec![ModelOffer {
                model_name: "llama3.1:70b".to_string(),
                quantization: "Q4_K_M".to_string(),
                max_context: 8192,
                estimated_tokens_per_second: tps,
                loaded: true,
            }],
            price_per_1k_tokens: price,
            availability: 1.0,
            reputation,
            reputation_by_model: HashMap::new(),
            current_load: load,
            max_concurrent: 3,
            updated_at: 1700000000,
        }
    }

    #[test]
    fn test_provider_selector_fastest() {
        let ads = vec![
            make_test_ad("slow", 10.0, 0.5, 0.9, 0.1),
            make_test_ad("fast", 80.0, 2.0, 0.9, 0.1),
            make_test_ad("medium", 40.0, 1.0, 0.9, 0.1),
        ];
        let selected =
            ProviderSelector::select_providers(&ads, "llama3.1:70b", &RequestGoal::Fastest, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].node_id, "fast");
    }

    #[test]
    fn test_provider_selector_cheapest() {
        let ads = vec![
            make_test_ad("expensive", 50.0, 5.0, 0.9, 0.1),
            make_test_ad("cheap", 30.0, 0.1, 0.9, 0.1),
            make_test_ad("mid", 40.0, 1.0, 0.9, 0.1),
        ];
        let selected =
            ProviderSelector::select_providers(&ads, "llama3.1:70b", &RequestGoal::Cheapest, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].node_id, "cheap");
    }

    #[test]
    fn test_provider_selector_balanced() {
        let ads = vec![
            make_test_ad("a", 50.0, 1.0, 0.9, 0.2),
            make_test_ad("b", 30.0, 0.5, 0.8, 0.5),
            make_test_ad("c", 70.0, 3.0, 0.7, 0.1),
        ];
        let selected =
            ProviderSelector::select_providers(&ads, "llama3.1:70b", &RequestGoal::Balanced, 3);
        assert_eq!(selected.len(), 3);
        // All three should be returned — order depends on balanced scoring
    }

    #[test]
    fn test_provider_selector_filters_bad_reputation() {
        let ads = vec![
            make_test_ad("good", 50.0, 1.0, 0.9, 0.1),
            make_test_ad("bad", 80.0, 0.1, 0.1, 0.0), // reputation below 0.3 threshold
        ];
        let selected =
            ProviderSelector::select_providers(&ads, "llama3.1:70b", &RequestGoal::Fastest, 5);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].node_id, "good");
    }

    #[test]
    fn test_provider_selector_filters_wrong_model() {
        let ads = vec![make_test_ad("node1", 50.0, 1.0, 0.9, 0.1)];
        let selected =
            ProviderSelector::select_providers(&ads, "mistral:7b", &RequestGoal::Auto, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_gpu_challenge_generate_and_verify_pass() {
        let challenge = GpuBenchmarkChallenge::generate(24576); // 24GB VRAM
        assert!(challenge.matrix_size > 0);
        assert!(challenge.expected_min_gflops > 0.0);
        assert!(challenge.time_limit_ms > 0);

        let result = GpuChallengeResult {
            result_hash: vec![1, 2, 3, 4],
            gflops_achieved: challenge.expected_min_gflops + 10.0,
            time_ms: challenge.time_limit_ms - 1000,
        };
        assert!(challenge.verify(&result));
    }

    #[test]
    fn test_gpu_challenge_verify_fail_too_slow() {
        let challenge = GpuBenchmarkChallenge::generate(8192);
        let result = GpuChallengeResult {
            result_hash: vec![1, 2, 3],
            gflops_achieved: challenge.expected_min_gflops + 5.0,
            time_ms: challenge.time_limit_ms + 1, // Over time limit
        };
        assert!(!challenge.verify(&result));
    }

    #[test]
    fn test_gpu_challenge_verify_fail_low_gflops() {
        let challenge = GpuBenchmarkChallenge::generate(16384);
        let result = GpuChallengeResult {
            result_hash: vec![1, 2, 3],
            gflops_achieved: challenge.expected_min_gflops * 0.5, // Below threshold
            time_ms: 5000,
        };
        assert!(!challenge.verify(&result));
    }

    #[test]
    fn test_gpu_challenge_verify_fail_empty_hash() {
        let challenge = GpuBenchmarkChallenge::generate(8192);
        let result = GpuChallengeResult {
            result_hash: vec![], // Empty
            gflops_achieved: challenge.expected_min_gflops + 5.0,
            time_ms: 5000,
        };
        assert!(!challenge.verify(&result));
    }

    #[test]
    fn test_route_request_local_only() {
        let decision = route_request(&RoutingStrategy::LocalOnly, 3, 100.0, 500);
        assert_eq!(decision, RouteDecision::Local);

        let decision = route_request(&RoutingStrategy::LocalOnly, 15, 100.0, 500);
        assert_eq!(decision, RouteDecision::LocalWait);
    }

    #[test]
    fn test_route_request_local_priority_with_queue() {
        // Low queue: local
        let decision = route_request(&RoutingStrategy::LocalPriority, 2, 100.0, 500);
        assert_eq!(decision, RouteDecision::Local);

        // High queue + credits: network
        let decision = route_request(&RoutingStrategy::LocalPriority, 8, 100.0, 500);
        assert_eq!(decision, RouteDecision::Network);

        // High queue + no credits: local wait
        let decision = route_request(&RoutingStrategy::LocalPriority, 8, 0.0, 500);
        assert_eq!(decision, RouteDecision::LocalWait);
    }

    #[test]
    fn test_route_request_network_only() {
        let decision = route_request(&RoutingStrategy::NetworkOnly, 0, 50.0, 500);
        assert_eq!(decision, RouteDecision::Network);

        let decision = route_request(&RoutingStrategy::NetworkOnly, 0, 0.0, 500);
        matches!(decision, RouteDecision::Error(_));
    }

    #[test]
    fn test_route_request_auto_decisions() {
        // Low queue: local
        let decision = route_request(&RoutingStrategy::Auto, 1, 100.0, 500);
        assert_eq!(decision, RouteDecision::Local);

        // High queue + credits: network
        let decision = route_request(&RoutingStrategy::Auto, 5, 100.0, 500);
        assert_eq!(decision, RouteDecision::Network);

        // High queue + no credits: local wait
        let decision = route_request(&RoutingStrategy::Auto, 5, 0.0, 500);
        assert_eq!(decision, RouteDecision::LocalWait);
    }

    #[test]
    fn test_transaction_receipt_serialization() {
        let receipt = TransactionReceipt {
            id: "rcpt-0000000000000001".to_string(),
            request_id: "req-001".to_string(),
            provider_node: "node-a".to_string(),
            requester_node: "node-b".to_string(),
            auditor_node: "node-c".to_string(),
            model: "llama3.1:70b".to_string(),
            tokens_in: 100,
            tokens_out: 200,
            credits_transferred: 1.5,
            fee_amount: 0.075,
            timestamp: 1700000000,
            provider_signature: vec![1, 2, 3],
            requester_signature: vec![4, 5, 6],
            auditor_signature: vec![7, 8, 9],
            audit_result: AuditResult::Verified,
            maturity_time: 1700003600,
        };
        let json = serde_json::to_string(&receipt).expect("serialize receipt");
        let deserialized: TransactionReceipt =
            serde_json::from_str(&json).expect("deserialize receipt");
        assert_eq!(deserialized.id, receipt.id);
        assert_eq!(deserialized.tokens_in, 100);
        assert_eq!(deserialized.tokens_out, 200);
        assert_eq!(deserialized.audit_result, AuditResult::Verified);
    }

    #[test]
    fn test_request_goal_auto_selection() {
        // Auto goal should behave like Balanced in scoring
        let ad = make_test_ad("test", 50.0, 1.0, 0.9, 0.1);
        let auto_score = ProviderSelector::score_provider(&ad, &RequestGoal::Auto);
        let balanced_score = ProviderSelector::score_provider(&ad, &RequestGoal::Balanced);
        assert!((auto_score - balanced_score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_node_capability_ad_serialization() {
        let ad = make_test_ad("test-node", 45.0, 1.5, 0.85, 0.3);
        let json = serde_json::to_string(&ad).expect("serialize ad");
        let deserialized: NodeCapabilityAd = serde_json::from_str(&json).expect("deserialize ad");
        assert_eq!(deserialized.node_id, "test-node");
        assert_eq!(deserialized.gpu.vram_mb, 24576);
        assert_eq!(deserialized.models.len(), 1);
        assert_eq!(deserialized.models[0].model_name, "llama3.1:70b");
    }

    #[test]
    fn test_audit_result_variants() {
        let variants = vec![
            AuditResult::Pending,
            AuditResult::Verified,
            AuditResult::Failed,
            AuditResult::Skipped,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).expect("serialize audit result");
            let deserialized: AuditResult =
                serde_json::from_str(&json).expect("deserialize audit result");
            assert_eq!(&deserialized, variant);
        }
    }

    #[test]
    fn test_sharing_mode_display() {
        assert_eq!(SharingMode::Provider.to_string(), "Provider");
        assert_eq!(SharingMode::Gateway.to_string(), "Gateway");
        assert_eq!(SharingMode::Both.to_string(), "Both");
    }

    #[test]
    fn test_inference_request_serialization() {
        let req = InferenceRequest {
            request_id: "req-test-001".to_string(),
            model: "mistral:7b".to_string(),
            prompt: "Hello, world!".to_string(),
            max_tokens: 256,
            temperature: 0.7,
            budget_credits: 5.0,
            goal: RequestGoal::Fastest,
            commitment: vec![10, 20, 30],
        };
        let json = serde_json::to_string(&req).expect("serialize request");
        let deserialized: InferenceRequest =
            serde_json::from_str(&json).expect("deserialize request");
        assert_eq!(deserialized.request_id, "req-test-001");
        assert_eq!(deserialized.model, "mistral:7b");
        assert_eq!(deserialized.goal, RequestGoal::Fastest);
    }

    #[test]
    fn test_gpu_challenge_scaling() {
        // Small VRAM should produce smaller matrix
        let small = GpuBenchmarkChallenge::generate(4096);
        let large = GpuBenchmarkChallenge::generate(24576);
        assert!(small.matrix_size <= large.matrix_size);
        assert!(small.expected_min_gflops <= large.expected_min_gflops);
    }

    // ── GpuSharingNode tests ───────────────────────────────────────

    #[test]
    fn test_gpu_sharing_node_creation() {
        let config = GpuSharingConfig::default();
        let node = GpuSharingNode::new(config, "test-node-1");

        // Default dynamic pricing: base_price = 1.0
        assert!((node.current_price() - 1.0).abs() < f64::EPSILON);
        // Fresh node starts with 0 balance
        assert!(!node.can_afford(1.0));
        // No capabilities advertised yet
        assert!(node.capabilities.is_none());
        assert!(node.active_requests.is_empty());
        assert!(node.receipts_pending.is_empty());
    }

    #[test]
    fn test_gpu_sharing_node_record_transaction() {
        let config = GpuSharingConfig::default();
        let mut node = GpuSharingNode::new(config, "test-node-2");

        // Recording a transaction should not panic
        node.record_transaction("node-a", "node-b", 5.0, "rcpt-test");
        // Collusion detector should have tracked the transaction pair
        // (internal state — we just verify it doesn't panic)
    }
}
