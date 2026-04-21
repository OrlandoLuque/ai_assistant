//! In-crate user-stall detection for agentic loops.
//!
//! A **stall** is a state where the agent is spending budget (tool calls,
//! tokens, time) without making progress that the user values. Two cheap
//! signals cover most cases and do not require an LLM call:
//!
//! 1. **Repeated tool calls.** If the same `(tool_name, args)` is invoked
//!    three or more times in the recent window, the agent is looping.
//! 2. **User frustration.** If the latest user message classifies as
//!    [`EmotionCategory::Frustrated`] with a keyword heuristic, the agent
//!    should stop before spending further budget.
//!
//! Either signal fires [`StallDecision::Stalled`]. The agent's main loop
//! checks the heuristic at each iteration and, when stalled, sets
//! [`LoopStatus::UserStalled`](crate::agentic_loop::LoopStatus::UserStalled),
//! emits the `user_stall_events_total` telemetry counter, and opens an
//! OpenTelemetry span named [`SPAN_NAME`] with a `signal` attribute.
//!
//! **Privacy.** No user message text is persisted — only a FNV-1a hash of the
//! tool call signature and the detected emotion category. This matches the
//! guarantees made by the `pii_tokenizer` module.
//!
//! Feature-gated behind `stall-detection`. Implies `autonomous` (agentic
//! loop), `audio` (emotion classification), and `analytics` (telemetry).
//!
//! # Robustness extensions (v0.2.27)
//!
//! Beyond the two original signals, [`KeywordStallDetector`] can optionally:
//!
//! * **Detect overheating** — too many tool calls within a sliding time
//!   window, regardless of repetition. See [`RateThresholds`]. Enabled via
//!   [`KeywordStallDetector::with_rate_thresholds`].
//! * **Classify frustration in Spanish, French, or German** — in addition to
//!   the existing English path through `KeywordEmotionDetector`. See
//!   [`StallLanguage`] and [`KeywordStallDetector::with_language`].

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::emotion_detection::{EmotionCategory, EmotionDetector, KeywordEmotionDetector};

/// OpenTelemetry span name emitted when a stall is detected.
pub const SPAN_NAME: &str = "agent.user_stall_detected";

/// Maximum number of recent tool-call hashes kept by
/// [`KeywordStallDetector`]. Fixed at 8 to match the task spec (Eje 2/6).
pub const RING_BUFFER_SIZE: usize = 8;

/// Minimum repeat count that triggers a `RepeatedToolCall` stall.
pub const REPEAT_THRESHOLD: usize = 3;

/// The kind of signal that caused a stall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StallSignal {
    /// The last user message classified as [`EmotionCategory::Frustrated`].
    Frustrated,
    /// The same tool call hash appeared `>= REPEAT_THRESHOLD` times in the
    /// last `RING_BUFFER_SIZE` invocations.
    RepeatedToolCall,
    /// Tool-call rate exceeded the configured [`RateThresholds`] — lots of
    /// different tool calls fired inside the sliding window. Distinct from
    /// `RepeatedToolCall`, which looks for identical invocations only.
    Overheating,
}

impl StallSignal {
    /// String form used for the OpenTelemetry span attribute.
    pub fn as_str(&self) -> &'static str {
        match self {
            StallSignal::Frustrated => "Frustrated",
            StallSignal::RepeatedToolCall => "RepeatedToolCall",
            StallSignal::Overheating => "Overheating",
        }
    }
}

impl std::fmt::Display for StallSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The verdict produced by a [`StallHeuristic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StallDecision {
    /// No stall; the agent should continue.
    Continue,
    /// A stall was detected; the agent should stop.
    Stalled(StallSignal),
}

/// Languages the in-crate frustration lexicon supports.
///
/// English falls through to the existing [`KeywordEmotionDetector`] (which is
/// broader than a simple word list). Spanish, French, and German use the
/// compact [`StallKeywordLexicon`] static word lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StallLanguage {
    English,
    Spanish,
    French,
    German,
}

impl StallLanguage {
    /// String form for telemetry / debug.
    pub fn as_str(&self) -> &'static str {
        match self {
            StallLanguage::English => "en",
            StallLanguage::Spanish => "es",
            StallLanguage::French => "fr",
            StallLanguage::German => "de",
        }
    }
}

/// Static keyword lists for per-language frustration detection.
///
/// Keywords are matched case-insensitively against the user message with
/// simple substring containment. The lexicon is **intentionally small** —
/// this is a cheap first-pass signal; the LLM-assisted wrapper (feature
/// `stall-detection-llm`) is the path for nuanced cases.
pub struct StallKeywordLexicon;

impl StallKeywordLexicon {
    /// Return the frustration keyword slice for the given language. The
    /// returned slice contains lowercase tokens / phrases.
    pub fn frustration_keywords(lang: StallLanguage) -> &'static [&'static str] {
        match lang {
            StallLanguage::English => &[
                "frustrated",
                "frustrating",
                "annoying",
                "annoyed",
                "stuck",
                "doesn't work",
                "does not work",
                "not working",
                "useless",
                "give up",
            ],
            StallLanguage::Spanish => &[
                "frustrado",
                "frustrada",
                "molesto",
                "molesta",
                "enfadado",
                "enfadada",
                "harto",
                "harta",
                "no funciona",
                "estoy atascado",
                "estoy atascada",
                "me rindo",
                "inutil",
                "inútil",
            ],
            StallLanguage::French => &[
                "frustré",
                "frustrée",
                "énervé",
                "énervée",
                "agacé",
                "agacée",
                "bloqué",
                "bloquée",
                "ne marche pas",
                "ne fonctionne pas",
                "j'en ai marre",
                "j'abandonne",
                "inutile",
            ],
            StallLanguage::German => &[
                "frustriert",
                "genervt",
                "verärgert",
                "feststecken",
                "funktioniert nicht",
                "keine ahnung",
                "gebe auf",
                "unbrauchbar",
                "nutzlos",
            ],
        }
    }

    /// Returns true iff `text` contains any frustration keyword for `lang`.
    /// Case-insensitive; simple substring match. No tokenization — this is
    /// deliberately cheap.
    pub fn contains_frustration(text: &str, lang: StallLanguage) -> bool {
        let lower = text.to_lowercase();
        Self::frustration_keywords(lang)
            .iter()
            .any(|kw| lower.contains(kw))
    }
}

/// Default window used by [`RateThresholds::default`].
pub const DEFAULT_RATE_WINDOW: Duration = Duration::from_secs(60);

/// Default tool-call budget per minute used by [`RateThresholds::default`].
/// Set high enough that normal agentic work does not trip; low enough that a
/// runaway loop does.
pub const DEFAULT_RATE_MAX_CALLS: usize = 30;

/// Rate thresholds for the optional overheating signal.
///
/// The detector keeps a sliding window of tool-call timestamps covering
/// [`Self::window`]. If the count exceeds [`Self::max_calls`] within the
/// window, [`StallSignal::Overheating`] fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateThresholds {
    /// Sliding window length.
    pub window: Duration,
    /// Maximum tool-call count within the window before firing.
    pub max_calls: usize,
}

impl RateThresholds {
    /// Construct a threshold explicitly.
    pub fn new(window: Duration, max_calls: usize) -> Self {
        Self { window, max_calls }
    }
}

impl Default for RateThresholds {
    fn default() -> Self {
        Self {
            window: DEFAULT_RATE_WINDOW,
            max_calls: DEFAULT_RATE_MAX_CALLS,
        }
    }
}

/// Heuristic that inspects the recent agent trace and decides whether the
/// agent is stalled.
///
/// Implementations must be side-effect free except for internal state
/// updates via the `observe_*` methods. They should not perform I/O.
pub trait StallHeuristic: Send + Sync {
    /// Record a tool call invocation. `args_hash` should be a stable hash of
    /// the serialized arguments (see [`hash_tool_call`]).
    fn observe_tool_call(&mut self, tool_name: &str, args_hash: u64);

    /// Record the latest user message. Implementations must **not** persist
    /// the raw text — only derived signals (emotion category, length, etc.).
    fn observe_user_message(&mut self, text: &str);

    /// Return the current stall verdict without mutating state.
    fn check(&self) -> StallDecision;

    /// Clear any accumulated state. Called when a new conversation starts or
    /// when the caller wants a clean slate.
    fn reset(&mut self);
}

/// FNV-1a hash for tool-call signatures.
///
/// Stable, dependency-free, and matches the hashing style used by
/// [`crate::telemetry`] for sampling decisions.
pub fn hash_tool_call(tool_name: &str, args_bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in tool_name.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Separator byte so `foo`+`bar` does not collide with `foob`+`ar`.
    hash ^= 0xFF;
    hash = hash.wrapping_mul(FNV_PRIME);
    for &byte in args_bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Default keyword-based stall detector.
///
/// Combines a fixed-size ring buffer of tool-call hashes with a frustration
/// classification of the most recent user message. Stores only derived
/// signals — no raw text.
///
/// By default, frustration is classified for **English** via
/// [`KeywordEmotionDetector`]. Other languages (Spanish / French / German)
/// use the compact [`StallKeywordLexicon`] and are selected via
/// [`Self::with_language`].
///
/// Optionally tracks **tool-call timestamps** for overheating detection —
/// disabled by default; enable with [`Self::with_rate_thresholds`].
pub struct KeywordStallDetector {
    recent_hashes: VecDeque<u64>,
    recent_timestamps: VecDeque<Instant>,
    last_emotion: Option<EmotionCategory>,
    last_was_frustrated: bool,
    emotion_detector: KeywordEmotionDetector,
    language: StallLanguage,
    rate_thresholds: Option<RateThresholds>,
}

impl KeywordStallDetector {
    pub fn new() -> Self {
        Self {
            recent_hashes: VecDeque::with_capacity(RING_BUFFER_SIZE),
            recent_timestamps: VecDeque::new(),
            last_emotion: None,
            last_was_frustrated: false,
            emotion_detector: KeywordEmotionDetector::new(),
            language: StallLanguage::English,
            rate_thresholds: None,
        }
    }

    /// Select the frustration lexicon language. Defaults to English, which
    /// delegates to [`KeywordEmotionDetector`] for a richer signal.
    pub fn with_language(mut self, language: StallLanguage) -> Self {
        self.language = language;
        self
    }

    /// Enable overheating detection with the given thresholds. When omitted,
    /// [`StallSignal::Overheating`] never fires.
    pub fn with_rate_thresholds(mut self, thresholds: RateThresholds) -> Self {
        self.rate_thresholds = Some(thresholds);
        self
    }

    /// Current language setting.
    pub fn language(&self) -> StallLanguage {
        self.language
    }

    /// Current rate thresholds, if overheating detection is enabled.
    pub fn rate_thresholds(&self) -> Option<RateThresholds> {
        self.rate_thresholds
    }

    /// The most recent emotion category classified from a user message,
    /// if any has been observed yet. Only populated for English.
    pub fn last_emotion(&self) -> Option<EmotionCategory> {
        self.last_emotion
    }

    /// Number of hashes currently in the ring buffer (0..=RING_BUFFER_SIZE).
    pub fn recent_hash_count(&self) -> usize {
        self.recent_hashes.len()
    }

    /// Number of tool-call timestamps currently tracked for overheating
    /// detection. Zero unless `with_rate_thresholds` was called.
    pub fn recent_timestamp_count(&self) -> usize {
        self.recent_timestamps.len()
    }

    fn most_repeated(&self) -> Option<(u64, usize)> {
        let mut best: Option<(u64, usize)> = None;
        for &h in &self.recent_hashes {
            let count = self.recent_hashes.iter().filter(|&&x| x == h).count();
            match best {
                Some((_, best_count)) if count <= best_count => {}
                _ => best = Some((h, count)),
            }
        }
        best
    }

    /// Count tool-call timestamps within the configured window. Returns 0 if
    /// `rate_thresholds` is None. Uses `now` so tests can inject a clock.
    fn calls_in_window_at(&self, now: Instant) -> usize {
        let Some(thresholds) = self.rate_thresholds else {
            return 0;
        };
        let cutoff = now.checked_sub(thresholds.window);
        self.recent_timestamps
            .iter()
            .filter(|ts| match cutoff {
                Some(c) => **ts >= c,
                None => true,
            })
            .count()
    }
}

impl Default for KeywordStallDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl StallHeuristic for KeywordStallDetector {
    fn observe_tool_call(&mut self, tool_name: &str, args_hash: u64) {
        // Mix tool_name into args_hash so different tools don't collide on
        // identical-looking args.
        let mut combined = hash_tool_call(tool_name, &[]);
        combined ^= args_hash;

        if self.recent_hashes.len() >= RING_BUFFER_SIZE {
            self.recent_hashes.pop_front();
        }
        self.recent_hashes.push_back(combined);

        if let Some(thresholds) = self.rate_thresholds {
            let now = Instant::now();
            self.recent_timestamps.push_back(now);
            // Evict timestamps outside the window. Keep at most max_calls+1
            // entries so the window stays bounded even if callers spam.
            let cutoff = now.checked_sub(thresholds.window);
            while let Some(front) = self.recent_timestamps.front() {
                let expired = match cutoff {
                    Some(c) => *front < c,
                    None => false,
                };
                if expired || self.recent_timestamps.len() > thresholds.max_calls.saturating_add(1)
                {
                    self.recent_timestamps.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    fn observe_user_message(&mut self, text: &str) {
        // Classify via keyword heuristic. We never keep the raw text.
        match self.language {
            StallLanguage::English => {
                let emotion = self
                    .emotion_detector
                    .detect_from_text(text)
                    .ok()
                    .map(|state| state.category);
                self.last_emotion = emotion;
                self.last_was_frustrated = emotion == Some(EmotionCategory::Frustrated);
            }
            other => {
                self.last_emotion = None;
                self.last_was_frustrated = StallKeywordLexicon::contains_frustration(text, other);
            }
        }
    }

    fn check(&self) -> StallDecision {
        // Precedence: RepeatedToolCall (definite loop) > Overheating
        // (real-time rate) > Frustrated (lagging user signal).
        if let Some((_, count)) = self.most_repeated() {
            if count >= REPEAT_THRESHOLD {
                return StallDecision::Stalled(StallSignal::RepeatedToolCall);
            }
        }
        if let Some(thresholds) = self.rate_thresholds {
            let count = self.calls_in_window_at(Instant::now());
            if count > thresholds.max_calls {
                return StallDecision::Stalled(StallSignal::Overheating);
            }
        }
        if self.last_was_frustrated {
            return StallDecision::Stalled(StallSignal::Frustrated);
        }
        StallDecision::Continue
    }

    fn reset(&mut self) {
        self.recent_hashes.clear();
        self.recent_timestamps.clear();
        self.last_emotion = None;
        self.last_was_frustrated = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_detector_reports_continue() {
        let det = KeywordStallDetector::new();
        assert_eq!(det.check(), StallDecision::Continue);
        assert_eq!(det.recent_hash_count(), 0);
        assert_eq!(det.last_emotion(), None);
    }

    #[test]
    fn three_identical_tool_calls_stall() {
        let mut det = KeywordStallDetector::new();
        let h = hash_tool_call("search", b"{\"q\":\"rust\"}");
        for _ in 0..3 {
            det.observe_tool_call("search", h);
        }
        assert_eq!(
            det.check(),
            StallDecision::Stalled(StallSignal::RepeatedToolCall)
        );
    }

    #[test]
    fn two_identical_tool_calls_do_not_stall() {
        let mut det = KeywordStallDetector::new();
        let h = hash_tool_call("search", b"{\"q\":\"rust\"}");
        det.observe_tool_call("search", h);
        det.observe_tool_call("search", h);
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn different_tool_calls_do_not_stall() {
        let mut det = KeywordStallDetector::new();
        for i in 0..8 {
            let args = format!("{{\"q\":\"q{}\"}}", i);
            let h = hash_tool_call("search", args.as_bytes());
            det.observe_tool_call("search", h);
        }
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn ring_buffer_evicts_old_hashes() {
        let mut det = KeywordStallDetector::new();
        let repeated = hash_tool_call("a", b"x");
        // Three repeats first — would trigger...
        for _ in 0..3 {
            det.observe_tool_call("a", repeated);
        }
        assert!(matches!(det.check(), StallDecision::Stalled(_)));

        // ...then flood with a different hash so the repeats roll out.
        for i in 0..RING_BUFFER_SIZE {
            let h = hash_tool_call("b", &i.to_le_bytes());
            det.observe_tool_call("b", h);
        }
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn frustrated_message_triggers_stall() {
        let mut det = KeywordStallDetector::new();
        det.observe_user_message("This doesn't work, I'm stuck and it's annoying");
        assert_eq!(det.check(), StallDecision::Stalled(StallSignal::Frustrated));
    }

    #[test]
    fn neutral_message_does_not_stall() {
        let mut det = KeywordStallDetector::new();
        det.observe_user_message("Please list the files in the directory");
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn repeat_signal_wins_over_frustration() {
        // If both fire we want to see RepeatedToolCall first because it has
        // a stronger invariant (budget is actively being burned).
        let mut det = KeywordStallDetector::new();
        let h = hash_tool_call("x", b"y");
        for _ in 0..3 {
            det.observe_tool_call("x", h);
        }
        det.observe_user_message("I'm frustrated");
        assert_eq!(
            det.check(),
            StallDecision::Stalled(StallSignal::RepeatedToolCall)
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut det = KeywordStallDetector::new();
        let h = hash_tool_call("x", b"y");
        for _ in 0..3 {
            det.observe_tool_call("x", h);
        }
        det.observe_user_message("I'm frustrated and stuck");
        det.reset();
        assert_eq!(det.check(), StallDecision::Continue);
        assert_eq!(det.recent_hash_count(), 0);
        assert_eq!(det.last_emotion(), None);
    }

    #[test]
    fn hash_tool_call_is_deterministic() {
        let a = hash_tool_call("search", b"{\"q\":\"rust\"}");
        let b = hash_tool_call("search", b"{\"q\":\"rust\"}");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_tool_call_discriminates_name_and_args() {
        let same_name = hash_tool_call("search", b"{\"q\":\"rust\"}");
        let other_name = hash_tool_call("find", b"{\"q\":\"rust\"}");
        let other_args = hash_tool_call("search", b"{\"q\":\"go\"}");
        assert_ne!(same_name, other_name);
        assert_ne!(same_name, other_args);
    }

    #[test]
    fn hash_tool_call_avoids_boundary_collision() {
        // "foo" + "bar" must not hash the same as "foob" + "ar".
        let a = hash_tool_call("foo", b"bar");
        let b = hash_tool_call("foob", b"ar");
        assert_ne!(a, b);
    }

    #[test]
    fn stall_signal_display_matches_as_str() {
        assert_eq!(StallSignal::Frustrated.to_string(), "Frustrated");
        assert_eq!(
            StallSignal::RepeatedToolCall.to_string(),
            "RepeatedToolCall"
        );
    }

    #[test]
    fn span_name_is_stable() {
        assert_eq!(SPAN_NAME, "agent.user_stall_detected");
    }

    // --- V95 extensions: overheating, multi-language, rate thresholds ------

    #[test]
    fn overheating_signal_as_str() {
        assert_eq!(StallSignal::Overheating.as_str(), "Overheating");
        assert_eq!(StallSignal::Overheating.to_string(), "Overheating");
    }

    #[test]
    fn default_rate_thresholds_are_sane() {
        let rt = RateThresholds::default();
        assert_eq!(rt.window, DEFAULT_RATE_WINDOW);
        assert_eq!(rt.max_calls, DEFAULT_RATE_MAX_CALLS);
        assert!(rt.max_calls >= 10); // sanity
    }

    #[test]
    fn rate_thresholds_new_stores_inputs() {
        let rt = RateThresholds::new(Duration::from_secs(5), 3);
        assert_eq!(rt.window, Duration::from_secs(5));
        assert_eq!(rt.max_calls, 3);
    }

    #[test]
    fn overheating_fires_when_rate_exceeds_threshold() {
        // Narrow threshold: 2 calls per 60s. Distinct args so RepeatedToolCall
        // does not fire first.
        let rt = RateThresholds::new(Duration::from_secs(60), 2);
        let mut det = KeywordStallDetector::new().with_rate_thresholds(rt);
        for i in 0u32..3 {
            let h = hash_tool_call("tool", &i.to_le_bytes());
            det.observe_tool_call("tool", h);
        }
        assert_eq!(
            det.check(),
            StallDecision::Stalled(StallSignal::Overheating)
        );
    }

    #[test]
    fn overheating_does_not_fire_without_rate_thresholds() {
        let mut det = KeywordStallDetector::new();
        for i in 0u32..100 {
            let h = hash_tool_call("tool", &i.to_le_bytes());
            det.observe_tool_call("tool", h);
        }
        // No rate thresholds configured → no Overheating signal.
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn repeat_beats_overheating() {
        // Identical calls would trigger BOTH RepeatedToolCall and Overheating;
        // precedence says RepeatedToolCall wins.
        let rt = RateThresholds::new(Duration::from_secs(60), 2);
        let mut det = KeywordStallDetector::new().with_rate_thresholds(rt);
        let h = hash_tool_call("x", b"y");
        for _ in 0..3 {
            det.observe_tool_call("x", h);
        }
        assert_eq!(
            det.check(),
            StallDecision::Stalled(StallSignal::RepeatedToolCall)
        );
    }

    #[test]
    fn overheating_beats_frustration() {
        // Precedence: Overheating (real-time) > Frustrated (lagging).
        let rt = RateThresholds::new(Duration::from_secs(60), 1);
        let mut det = KeywordStallDetector::new().with_rate_thresholds(rt);
        for i in 0u32..3 {
            let h = hash_tool_call("tool", &i.to_le_bytes());
            det.observe_tool_call("tool", h);
        }
        det.observe_user_message("I'm frustrated");
        assert_eq!(
            det.check(),
            StallDecision::Stalled(StallSignal::Overheating)
        );
    }

    #[test]
    fn spanish_frustration_keyword_triggers_stall() {
        let mut det = KeywordStallDetector::new().with_language(StallLanguage::Spanish);
        det.observe_user_message("Esto no funciona, estoy harto");
        assert_eq!(det.check(), StallDecision::Stalled(StallSignal::Frustrated));
    }

    #[test]
    fn french_frustration_keyword_triggers_stall() {
        let mut det = KeywordStallDetector::new().with_language(StallLanguage::French);
        det.observe_user_message("Ça ne marche pas, j'en ai marre");
        assert_eq!(det.check(), StallDecision::Stalled(StallSignal::Frustrated));
    }

    #[test]
    fn german_frustration_keyword_triggers_stall() {
        let mut det = KeywordStallDetector::new().with_language(StallLanguage::German);
        det.observe_user_message("Das funktioniert nicht, ich bin frustriert");
        assert_eq!(det.check(), StallDecision::Stalled(StallSignal::Frustrated));
    }

    #[test]
    fn non_english_neutral_message_does_not_stall() {
        let mut det = KeywordStallDetector::new().with_language(StallLanguage::Spanish);
        det.observe_user_message("Por favor lista los ficheros del directorio");
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn spanish_detector_does_not_populate_emotion() {
        // last_emotion() is only populated for English (via
        // KeywordEmotionDetector). Other languages match the lexicon but
        // skip emotion classification.
        let mut det = KeywordStallDetector::new().with_language(StallLanguage::Spanish);
        det.observe_user_message("Estoy harto");
        assert_eq!(det.last_emotion(), None);
    }

    #[test]
    fn lexicon_contains_frustration_is_case_insensitive() {
        assert!(StallKeywordLexicon::contains_frustration(
            "I AM FRUSTRATED WITH THIS",
            StallLanguage::English
        ));
        assert!(StallKeywordLexicon::contains_frustration(
            "ESTOY HARTO",
            StallLanguage::Spanish
        ));
    }

    #[test]
    fn stall_language_as_str_matches() {
        assert_eq!(StallLanguage::English.as_str(), "en");
        assert_eq!(StallLanguage::Spanish.as_str(), "es");
        assert_eq!(StallLanguage::French.as_str(), "fr");
        assert_eq!(StallLanguage::German.as_str(), "de");
    }

    #[test]
    fn reset_clears_rate_timestamps() {
        let rt = RateThresholds::new(Duration::from_secs(60), 2);
        let mut det = KeywordStallDetector::new().with_rate_thresholds(rt);
        for i in 0u32..3 {
            let h = hash_tool_call("t", &i.to_le_bytes());
            det.observe_tool_call("t", h);
        }
        assert!(det.recent_timestamp_count() > 0);
        det.reset();
        assert_eq!(det.recent_timestamp_count(), 0);
        assert_eq!(det.check(), StallDecision::Continue);
    }

    #[test]
    fn with_language_and_rate_thresholds_chain() {
        let det = KeywordStallDetector::new()
            .with_language(StallLanguage::French)
            .with_rate_thresholds(RateThresholds::new(Duration::from_secs(10), 5));
        assert_eq!(det.language(), StallLanguage::French);
        assert_eq!(det.rate_thresholds().map(|r| r.max_calls), Some(5));
    }
}
