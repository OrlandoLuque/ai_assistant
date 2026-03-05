//! BPE token counter for precise token counting
//!
//! Provides a pure-Rust byte-level BPE tokenizer alongside the existing
//! approximate character-based estimation from [`crate::context::estimate_tokens`].
//!
//! # Overview
//!
//! - [`ApproximateCounter`] — wraps the existing ~3.5 chars/token heuristic
//! - [`BpeTokenCounter`] — byte-level BPE with the top ~200 merge rules
//! - [`ProviderTokenCounter`] — dispatches to the right counter based on model name
//! - [`TokenBudget`] / [`TokenAllocation`] — helpers for allocating tokens across prompt parts

/// Trait for counting tokens in text.
pub trait TokenCounter: Send + Sync {
    /// Count the number of tokens in `text`.
    fn count(&self, text: &str) -> usize;

    /// Count the total tokens across a slice of `(role, content)` messages.
    ///
    /// Implementations should include per-message overhead (formatting tokens).
    fn count_messages(&self, messages: &[(&str, &str)]) -> usize;
}

// =============================================================================
// ApproximateCounter
// =============================================================================

/// A fast, approximate token counter that delegates to
/// [`crate::context::estimate_tokens`] (~3.5 chars/token).
#[derive(Debug)]
pub struct ApproximateCounter;

impl ApproximateCounter {
    pub fn new() -> Self {
        Self
    }
}

impl TokenCounter for ApproximateCounter {
    fn count(&self, text: &str) -> usize {
        crate::context::estimate_tokens(text)
    }

    fn count_messages(&self, messages: &[(&str, &str)]) -> usize {
        let mut total = 0;
        for (role, content) in messages {
            // ~4 tokens overhead per message for role + formatting
            total += 4;
            total += self.count(role);
            total += self.count(content);
        }
        total
    }
}

// =============================================================================
// BPE merge table
// =============================================================================

/// A single BPE merge rule: (left bytes, right bytes).
///
/// The index in the array is the merge priority (lower = higher priority).
type MergeRule = (&'static [u8], &'static [u8]);

/// Top ~200 most common byte-pair merges for English text, inspired by
/// GPT-style cl100k_base tokenizers. These are ordered by frequency
/// (index 0 = highest priority merge).
const BPE_MERGES: &[MergeRule] = &[
    // Rank 0-9: very common English bigrams and space-prefixed letters
    (b" ", b"t"),   // " t"
    (b"t", b"h"),   // "th"
    (b" ", b"a"),   // " a"
    (b"h", b"e"),   // "he"
    (b"i", b"n"),   // "in"
    (b"e", b"r"),   // "er"
    (b" ", b"s"),   // " s"
    (b"r", b"e"),   // "re"
    (b"o", b"n"),   // "on"
    (b" ", b"c"),   // " c"
    // Rank 10-19
    (b"a", b"t"),   // "at"
    (b"e", b"n"),   // "en"
    (b"n", b"d"),   // "nd"
    (b"t", b"i"),   // "ti"
    (b"e", b"s"),   // "es"
    (b"o", b"r"),   // "or"
    (b"t", b"e"),   // "te"
    (b" ", b"o"),   // " o"
    (b"e", b"d"),   // "ed"
    (b"i", b"s"),   // "is"
    // Rank 20-29
    (b"i", b"t"),   // "it"
    (b"a", b"n"),   // "an"
    (b"a", b"r"),   // "ar"
    (b"a", b"l"),   // "al"
    (b" ", b"b"),   // " b"
    (b"o", b"u"),   // "ou"
    (b" ", b"w"),   // " w"
    (b"l", b"e"),   // "le"
    (b" ", b"d"),   // " d"
    (b" ", b"f"),   // " f"
    // Rank 30-39
    (b"i", b"o"),   // "io"
    (b"o", b"t"),   // "ot"
    (b" ", b"m"),   // " m"
    (b"a", b"s"),   // "as"
    (b"e", b"l"),   // "el"
    (b"c", b"t"),   // "ct"
    (b"n", b"t"),   // "nt"
    (b"l", b"l"),   // "ll"
    (b" ", b"p"),   // " p"
    (b"s", b"t"),   // "st"
    // Rank 40-49
    (b" ", b"h"),   // " h"
    (b"e", b"c"),   // "ec"
    (b"i", b"c"),   // "ic"
    (b"i", b"g"),   // "ig"
    (b" ", b"i"),   // " i"
    (b" ", b"n"),   // " n"
    (b"o", b"m"),   // "om"
    (b"a", b"d"),   // "ad"
    (b"u", b"r"),   // "ur"
    (b"i", b"v"),   // "iv"
    // Rank 50-59
    (b"e", b"m"),   // "em"
    (b"a", b"c"),   // "ac"
    (b"o", b"l"),   // "ol"
    (b" ", b"e"),   // " e"
    (b" ", b"r"),   // " r"
    (b"u", b"s"),   // "us"
    (b"a", b"g"),   // "ag"
    (b" ", b"l"),   // " l"
    (b"i", b"l"),   // "il"
    (b"e", b"a"),   // "ea"
    // Rank 60-69
    (b" ", b"g"),   // " g"
    (b"v", b"e"),   // "ve"
    (b"u", b"t"),   // "ut"
    (b"i", b"d"),   // "id"
    (b"u", b"n"),   // "un"
    (b"e", b"t"),   // "et"
    (b"o", b"w"),   // "ow"
    (b"r", b"o"),   // "ro"
    (b"l", b"y"),   // "ly"
    (b"o", b"f"),   // "of"
    // Rank 70-79
    (b"r", b"a"),   // "ra"
    (b"r", b"i"),   // "ri"
    (b"n", b"e"),   // "ne"
    (b"c", b"o"),   // "co"
    (b"c", b"e"),   // "ce"
    (b"i", b"r"),   // "ir"
    (b" ", b"u"),   // " u"
    (b"u", b"l"),   // "ul"
    (b"a", b"m"),   // "am"
    (b"a", b"i"),   // "ai"
    // Rank 80-89
    (b"p", b"e"),   // "pe"
    (b"s", b"e"),   // "se"
    (b"p", b"r"),   // "pr"
    (b"u", b"e"),   // "ue"
    (b"o", b"s"),   // "os"
    (b"s", b"s"),   // "ss"
    (b"i", b"m"),   // "im"
    (b"a", b"b"),   // "ab"
    (b"l", b"a"),   // "la"
    (b"p", b"o"),   // "po"
    // Rank 90-99
    (b"i", b"e"),   // "ie"
    (b"d", b"e"),   // "de"
    (b"o", b"d"),   // "od"
    (b"u", b"d"),   // "ud"
    (b"t", b"r"),   // "tr"
    (b"m", b"e"),   // "me"
    (b"i", b"a"),   // "ia"
    (b"u", b"m"),   // "um"
    (b"c", b"h"),   // "ch"
    (b"a", b"p"),   // "ap"
    // Rank 100-109
    (b"f", b"o"),   // "fo"
    (b"l", b"o"),   // "lo"
    (b"g", b"e"),   // "ge"
    (b"n", b"o"),   // "no"
    (b"s", b"h"),   // "sh"
    (b"r", b"s"),   // "rs"
    (b"p", b"l"),   // "pl"
    (b"w", b"a"),   // "wa"
    (b"e", b"e"),   // "ee"
    (b"o", b"o"),   // "oo"
    // Rank 110-119
    (b"w", b"h"),   // "wh"
    (b"g", b"h"),   // "gh"
    (b"m", b"a"),   // "ma"
    (b"i", b"f"),   // "if"
    (b"c", b"a"),   // "ca"
    (b"d", b"i"),   // "di"
    (b"f", b"i"),   // "fi"
    (b"b", b"e"),   // "be"
    (b"g", b"o"),   // "go"
    (b"t", b"o"),   // "to"
    // Rank 120-129
    (b"d", b"o"),   // "do"
    (b"n", b"g"),   // "ng"
    (b"k", b"e"),   // "ke"
    (b"w", b"i"),   // "wi"
    (b"s", b"i"),   // "si"
    (b"b", b"l"),   // "bl"
    (b"m", b"o"),   // "mo"
    (b"n", b"a"),   // "na"
    (b"b", b"o"),   // "bo"
    (b"w", b"e"),   // "we"
    // Rank 130-139
    (b"d", b"a"),   // "da"
    (b"l", b"i"),   // "li"
    (b"r", b"u"),   // "ru"
    (b"v", b"i"),   // "vi"
    (b"h", b"a"),   // "ha"
    (b"h", b"i"),   // "hi"
    (b"c", b"l"),   // "cl"
    (b"g", b"r"),   // "gr"
    (b"f", b"r"),   // "fr"
    (b"p", b"a"),   // "pa"
    // Rank 140-149
    (b"c", b"r"),   // "cr"
    (b"s", b"o"),   // "so"
    (b"s", b"u"),   // "su"
    (b"b", b"u"),   // "bu"
    (b"m", b"i"),   // "mi"
    (b"n", b"i"),   // "ni"
    (b"n", b"s"),   // "ns"
    (b"p", b"i"),   // "pi"
    (b"d", b"u"),   // "du"
    (b"k", b"i"),   // "ki"
    // Rank 150-159
    (b"t", b"u"),   // "tu"
    (b"s", b"p"),   // "sp"
    (b"s", b"c"),   // "sc"
    (b"f", b"e"),   // "fe"
    (b"g", b"a"),   // "ga"
    (b"g", b"i"),   // "gi"
    (b"j", b"u"),   // "ju"
    (b"w", b"o"),   // "wo"
    (b"f", b"l"),   // "fl"
    (b"a", b"v"),   // "av"
    // Rank 160-169
    (b"r", b"y"),   // "ry"
    (b"a", b"w"),   // "aw"
    (b"o", b"p"),   // "op"
    (b"e", b"x"),   // "ex"
    (b"l", b"u"),   // "lu"
    (b"b", b"r"),   // "br"
    (b"d", b"r"),   // "dr"
    (b"c", b"k"),   // "ck"
    (b"e", b"p"),   // "ep"
    (b"h", b"o"),   // "ho"
    // Rank 170-179
    (b"k", b"n"),   // "kn"
    (b"m", b"u"),   // "mu"
    (b"o", b"v"),   // "ov"
    (b"a", b"k"),   // "ak"
    (b"v", b"a"),   // "va"
    (b"u", b"p"),   // "up"
    (b"f", b"u"),   // "fu"
    (b"n", b"u"),   // "nu"
    (b"g", b"u"),   // "gu"
    (b"y", b"o"),   // "yo"
    // Rank 180-189
    (b"s", b"w"),   // "sw"
    (b"t", b"w"),   // "tw"
    (b"i", b"p"),   // "ip"
    (b"o", b"c"),   // "oc"
    (b"e", b"w"),   // "ew"
    (b"a", b"y"),   // "ay"
    (b"o", b"g"),   // "og"
    (b"u", b"b"),   // "ub"
    (b"i", b"x"),   // "ix"
    (b"o", b"b"),   // "ob"
    // Rank 190-199
    (b"u", b"g"),   // "ug"
    (b"e", b"g"),   // "eg"
    (b"y", b"s"),   // "ys"
    (b"e", b"v"),   // "ev"
    (b"a", b"x"),   // "ax"
    (b"u", b"c"),   // "uc"
    (b"o", b"k"),   // "ok"
    (b"i", b"b"),   // "ib"
    (b"o", b"x"),   // "ox"
    (b"u", b"i"),   // "ui"
];

// =============================================================================
// BpeTokenCounter
// =============================================================================

/// A pure-Rust byte-level BPE tokenizer.
///
/// Uses the top ~200 most common BPE merge rules to approximate the
/// token counts produced by GPT-4-class tokenizers (cl100k_base).
#[derive(Debug)]
pub struct BpeTokenCounter {
    /// Precomputed merge table: for each merge rule, the (left, right) byte
    /// sequences and the merged result.
    merges: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

impl BpeTokenCounter {
    /// Create a new `BpeTokenCounter` with the built-in merge table.
    pub fn new() -> Self {
        let merges = BPE_MERGES
            .iter()
            .map(|(left, right)| {
                let mut merged = left.to_vec();
                merged.extend_from_slice(right);
                (left.to_vec(), right.to_vec(), merged)
            })
            .collect();

        Self { merges }
    }

    /// Encode `text` into BPE tokens, returned as byte sequences.
    pub fn encode(&self, text: &str) -> Vec<Vec<u8>> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with each byte as its own token
        let mut tokens: Vec<Vec<u8>> = text.bytes().map(|b| vec![b]).collect();

        // Iteratively apply merge rules in priority order
        for (left, right, merged) in &self.merges {
            if tokens.len() <= 1 {
                break;
            }
            tokens = Self::apply_merge(&tokens, left, right, merged);
        }

        tokens
    }

    /// Apply a single merge rule across the token list.
    ///
    /// Scans for adjacent `(left, right)` pairs and replaces them with
    /// `merged`. Returns the new token list.
    fn apply_merge(
        tokens: &[Vec<u8>],
        left: &[u8],
        right: &[u8],
        merged: &[u8],
    ) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                result.push(merged.to_vec());
                i += 2;
            } else {
                result.push(tokens[i].clone());
                i += 1;
            }
        }

        result
    }
}

impl TokenCounter for BpeTokenCounter {
    fn count(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    fn count_messages(&self, messages: &[(&str, &str)]) -> usize {
        let mut total = 0;
        for (role, content) in messages {
            // ~4 tokens overhead per message (role markers + formatting)
            total += 4;
            total += self.count(role);
            total += self.count(content);
        }
        total
    }
}

// =============================================================================
// ProviderTokenCounter
// =============================================================================

/// Dispatches to the correct [`TokenCounter`] based on model name.
///
/// - OpenAI (`gpt-*`) and Anthropic (`claude-*`) models use [`BpeTokenCounter`]
/// - Local / unknown models fall back to [`ApproximateCounter`]
#[derive(Debug)]
pub struct ProviderTokenCounter {
    bpe: BpeTokenCounter,
    approximate: ApproximateCounter,
}

impl ProviderTokenCounter {
    pub fn new() -> Self {
        Self {
            bpe: BpeTokenCounter::new(),
            approximate: ApproximateCounter::new(),
        }
    }

    /// Return the appropriate counter for `model`.
    ///
    /// Uses BPE for cloud/known models (more accurate), approximate for local/unknown.
    pub fn for_model(&self, model: &str) -> &dyn TokenCounter {
        let name = model.to_lowercase();

        if name.starts_with("gpt-")
            || name.starts_with("gpt4")
            || name.starts_with("o1")
            || name.starts_with("o3")
            || name.starts_with("o4")
            || name.starts_with("claude-")
            || name.starts_with("claude3")
            || name.starts_with("claude4")
            || name.contains("gemini")
            || name.contains("mistral")
            || name.contains("deepseek")
            || name.contains("command-r")
        {
            &self.bpe
        } else {
            // Ollama, LM Studio, local models, unknown
            &self.approximate
        }
    }
}

// =============================================================================
// TokenBudget / TokenAllocation
// =============================================================================

/// Describes how to divide a total token budget across prompt sections.
#[derive(Debug)]
pub struct TokenBudget {
    /// Total tokens available.
    pub total: usize,
}

/// The result of splitting a [`TokenBudget`] into sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAllocation {
    /// Tokens allocated for the system prompt.
    pub system_tokens: usize,
    /// Tokens allocated for knowledge / RAG context.
    pub knowledge_tokens: usize,
    /// Tokens allocated for conversation history.
    pub history_tokens: usize,
    /// Tokens reserved for the model's response.
    pub response_reserve: usize,
}

impl TokenBudget {
    /// Create a new budget with the given total token count.
    pub fn new(total: usize) -> Self {
        Self { total }
    }

    /// Allocate the budget according to the given percentages (0.0 – 1.0).
    ///
    /// The remainder after `system_pct + knowledge_pct + history_pct` is
    /// reserved for the response.
    ///
    /// # Panics
    ///
    /// Panics if the sum of percentages exceeds 1.0.
    pub fn allocate(
        &self,
        system_pct: f64,
        knowledge_pct: f64,
        history_pct: f64,
    ) -> TokenAllocation {
        let sum = system_pct + knowledge_pct + history_pct;
        assert!(
            sum <= 1.0 + f64::EPSILON,
            "percentages must not exceed 1.0 (got {sum})"
        );

        let system_tokens = (self.total as f64 * system_pct) as usize;
        let knowledge_tokens = (self.total as f64 * knowledge_pct) as usize;
        let history_tokens = (self.total as f64 * history_pct) as usize;
        let response_reserve = self.total.saturating_sub(system_tokens + knowledge_tokens + history_tokens);

        TokenAllocation {
            system_tokens,
            knowledge_tokens,
            history_tokens,
            response_reserve,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ApproximateCounter matches existing estimate_tokens
    #[test]
    fn test_approximate_counter() {
        let counter = ApproximateCounter::new();
        let text = "This is a sample sentence for testing.";
        assert_eq!(counter.count(text), crate::context::estimate_tokens(text));
        assert_eq!(counter.count(""), crate::context::estimate_tokens(""));
        assert_eq!(counter.count("hello"), crate::context::estimate_tokens("hello"));
    }

    // 2. BPE on empty string
    #[test]
    fn test_bpe_empty_string() {
        let bpe = BpeTokenCounter::new();
        assert_eq!(bpe.count(""), 0);
        assert!(bpe.encode("").is_empty());
    }

    // 3. BPE on a single word
    #[test]
    fn test_bpe_single_word() {
        let bpe = BpeTokenCounter::new();
        let count = bpe.count("hello");
        // "hello" is 5 bytes; after merges like "he", "ll" it should be 1-3 tokens
        assert!(count >= 1, "got {count}");
        assert!(count <= 5, "got {count}");
    }

    // 4. BPE on a sentence produces fewer tokens than characters
    #[test]
    fn test_bpe_sentence() {
        let bpe = BpeTokenCounter::new();
        let text = "The quick brown fox jumps over the lazy dog.";
        let count = bpe.count(text);
        assert!(count > 0);
        assert!(
            count < text.len(),
            "BPE count ({count}) should be less than char count ({})",
            text.len()
        );
    }

    // 5. BPE handles Unicode correctly
    #[test]
    fn test_bpe_unicode() {
        let bpe = BpeTokenCounter::new();
        // Non-ASCII: each char is multi-byte in UTF-8
        let text = "caf\u{00e9} na\u{00ef}ve";
        let count = bpe.count(text);
        assert!(count > 0);
        // Should still produce a reasonable count — at minimum 1 token per UTF-8 byte
        // that does not get merged
        assert!(count <= text.len());
    }

    // 6. BPE on longer text
    #[test]
    fn test_bpe_long_text() {
        let bpe = BpeTokenCounter::new();
        let text = "Artificial intelligence is the simulation of human intelligence \
                    processes by computer systems. These processes include learning, \
                    reasoning, and self-correction. Particular applications of AI \
                    include expert systems, natural language processing, speech \
                    recognition, and machine vision.";
        let count = bpe.count(text);
        assert!(count > 0);
        assert!(count < text.len());
        // With merges, we expect roughly 40-60% compression on English prose
        assert!(
            count < text.len() * 80 / 100,
            "expected meaningful compression, got {count} tokens for {} bytes",
            text.len()
        );
    }

    // 7. BPE on whitespace
    #[test]
    fn test_bpe_whitespace() {
        let bpe = BpeTokenCounter::new();
        assert!(bpe.count(" ") > 0);
        assert!(bpe.count("\t") > 0);
        assert!(bpe.count("\n") > 0);
        assert!(bpe.count("   \n\t  ") > 0);
    }

    // 8. BPE on numbers
    #[test]
    fn test_bpe_numbers() {
        let bpe = BpeTokenCounter::new();
        let count = bpe.count("1234567890");
        assert!(count > 0);
        // Digits are individual bytes; few merge rules apply, so count ~ byte count
        assert!(count <= 10);
    }

    // 9. BPE on special characters / punctuation
    #[test]
    fn test_bpe_special_chars() {
        let bpe = BpeTokenCounter::new();
        let text = "Hello, world! @#$%^&*()";
        let count = bpe.count(text);
        assert!(count > 0);
        assert!(count <= text.len());
    }

    // 10. ProviderTokenCounter routes OpenAI to BPE
    #[test]
    fn test_provider_counter_openai() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "Test sentence for token counting.";

        let counter = provider.for_model("gpt-4o");
        assert_eq!(counter.count(text), bpe.count(text));
    }

    // 11. ProviderTokenCounter routes Claude to BPE
    #[test]
    fn test_provider_counter_claude() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "Another test sentence.";

        let counter = provider.for_model("claude-3-opus");
        assert_eq!(counter.count(text), bpe.count(text));
    }

    // 12. ProviderTokenCounter routes Ollama/local to Approximate
    #[test]
    fn test_provider_counter_ollama() {
        let provider = ProviderTokenCounter::new();
        let approx = ApproximateCounter::new();
        let text = "Local model test.";

        let counter = provider.for_model("llama3:8b");
        assert_eq!(counter.count(text), approx.count(text));
    }

    // 13. ProviderTokenCounter routes unknown model to Approximate
    #[test]
    fn test_provider_counter_unknown() {
        let provider = ProviderTokenCounter::new();
        let approx = ApproximateCounter::new();
        let text = "Unknown model fallback.";

        let counter = provider.for_model("some-random-model");
        assert_eq!(counter.count(text), approx.count(text));
    }

    // 14. TokenBudget allocation percentages sum correctly
    #[test]
    fn test_token_budget_allocation() {
        let budget = TokenBudget::new(10_000);
        let alloc = budget.allocate(0.10, 0.40, 0.30);

        assert_eq!(alloc.system_tokens, 1_000);
        assert_eq!(alloc.knowledge_tokens, 4_000);
        assert_eq!(alloc.history_tokens, 3_000);
        assert_eq!(alloc.response_reserve, 2_000);

        // All parts must sum to total
        let sum = alloc.system_tokens
            + alloc.knowledge_tokens
            + alloc.history_tokens
            + alloc.response_reserve;
        assert_eq!(sum, 10_000);
    }

    // 15. Message counting includes per-message overhead
    #[test]
    fn test_message_counting() {
        let bpe = BpeTokenCounter::new();
        let messages: &[(&str, &str)] = &[
            ("system", "You are a helpful assistant."),
            ("user", "Hello!"),
        ];

        let total = bpe.count_messages(messages);

        // Total must be strictly greater than content alone (due to overhead)
        let content_only: usize = messages
            .iter()
            .map(|(r, c)| bpe.count(r) + bpe.count(c))
            .sum();
        assert!(
            total > content_only,
            "message counting ({total}) should exceed content-only ({content_only}) due to per-message overhead"
        );

        // Overhead = 4 tokens per message * 2 messages = 8
        assert_eq!(total, content_only + 8);
    }

    // 16. ProviderTokenCounter routes Gemini to BPE
    #[test]
    fn test_provider_counter_gemini() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "Gemini model routing test.";
        assert_eq!(provider.for_model("gemini-1.5-pro").count(text), bpe.count(text));
    }

    // 17. ProviderTokenCounter routes DeepSeek to BPE
    #[test]
    fn test_provider_counter_deepseek() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "DeepSeek routing test.";
        assert_eq!(provider.for_model("deepseek-v2").count(text), bpe.count(text));
    }

    // 18. ProviderTokenCounter routes Mistral to BPE
    #[test]
    fn test_provider_counter_mistral() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "Mistral routing test.";
        assert_eq!(provider.for_model("mistral-large").count(text), bpe.count(text));
    }

    // 19. ProviderTokenCounter routes o4-* to BPE
    #[test]
    fn test_provider_counter_o4() {
        let provider = ProviderTokenCounter::new();
        let bpe = BpeTokenCounter::new();
        let text = "o4-mini model test.";
        assert_eq!(provider.for_model("o4-mini").count(text), bpe.count(text));
    }
}
