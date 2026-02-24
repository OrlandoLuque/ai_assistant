//! Reasoning trace capture, storage, and analysis.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// 4.2 Reasoning Trace Capture
// ============================================================================

/// A single step in a reasoning trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// The thought or reasoning content of this step.
    pub thought: String,
    /// Optional conclusion drawn from this step.
    pub conclusion: Option<String>,
    /// Evidence supporting this step's reasoning.
    pub evidence: Vec<String>,
    /// Confidence level for this step (0.0..1.0).
    pub confidence: f64,
    /// Estimated token count for this step.
    pub token_count: usize,
}

/// A complete reasoning trace for a signature execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// The individual reasoning steps.
    pub steps: Vec<ReasoningStep>,
    /// Total token count across all steps.
    pub total_tokens: usize,
    /// The signature ID that produced this trace.
    pub signature_id: String,
    /// Hash of the input that produced this trace.
    pub input_hash: String,
}

impl ReasoningTrace {
    /// Create a new empty reasoning trace.
    pub fn new(signature_id: String, input_hash: String) -> Self {
        Self {
            steps: Vec::new(),
            total_tokens: 0,
            signature_id,
            input_hash,
        }
    }

    /// Add a reasoning step to this trace.
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.total_tokens += step.token_count;
        self.steps.push(step);
    }

    /// Return the number of reasoning steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Compute the average confidence across all steps.
    pub fn avg_confidence(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.iter().map(|s| s.confidence).sum();
        sum / self.steps.len() as f64
    }

    /// Return the total token count.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Return the conclusion of the last step, if any.
    pub fn final_conclusion(&self) -> Option<&str> {
        self.steps.last().and_then(|s| s.conclusion.as_deref())
    }
}

/// Extracts reasoning traces from text using configurable markers.
pub struct TraceExtractor {
    /// Markers that indicate the start of a reasoning thought.
    pub(crate) thought_markers: Vec<String>,
    /// Markers that indicate a conclusion within a step.
    pub(crate) conclusion_markers: Vec<String>,
}

impl TraceExtractor {
    /// Create a new trace extractor with no markers.
    pub fn new() -> Self {
        Self {
            thought_markers: Vec::new(),
            conclusion_markers: Vec::new(),
        }
    }

    /// Create a trace extractor with common default markers.
    pub fn with_defaults() -> Self {
        Self {
            thought_markers: vec![
                "<thinking>".to_string(),
                "Let me think".to_string(),
                "Step 1:".to_string(),
                "Step 2:".to_string(),
                "Step 3:".to_string(),
                "Step 4:".to_string(),
                "Step 5:".to_string(),
                "First,".to_string(),
                "Next,".to_string(),
                "Then,".to_string(),
            ],
            conclusion_markers: vec![
                "Therefore".to_string(),
                "In conclusion".to_string(),
                "Answer:".to_string(),
                "Thus,".to_string(),
                "So,".to_string(),
                "Finally,".to_string(),
                "</thinking>".to_string(),
            ],
        }
    }

    /// Add a thought marker.
    pub fn add_thought_marker(&mut self, marker: String) {
        self.thought_markers.push(marker);
    }

    /// Add a conclusion marker.
    pub fn add_conclusion_marker(&mut self, marker: String) {
        self.conclusion_markers.push(marker);
    }

    /// Extract a reasoning trace from the given text.
    ///
    /// Splits text by thought markers into segments, then looks for conclusion
    /// markers within each segment to identify conclusions.
    pub fn extract(&self, text: &str) -> ReasoningTrace {
        let mut trace = ReasoningTrace::new(String::new(), String::new());

        if text.trim().is_empty() {
            return trace;
        }

        // If no thought markers configured, treat entire text as one step
        if self.thought_markers.is_empty() {
            let conclusion = self.find_conclusion(text);
            let token_count = text.split_whitespace().count();
            trace.add_step(ReasoningStep {
                thought: text.to_string(),
                conclusion,
                evidence: Vec::new(),
                confidence: 0.5,
                token_count,
            });
            return trace;
        }

        // Find all marker positions and sort by position
        let mut split_positions: Vec<usize> = Vec::new();
        for marker in &self.thought_markers {
            let mut search_start = 0;
            while let Some(pos) = text[search_start..].find(marker.as_str()) {
                split_positions.push(search_start + pos);
                search_start += pos + marker.len();
            }
        }
        split_positions.sort();
        split_positions.dedup();

        if split_positions.is_empty() {
            // No markers found, treat entire text as one step
            let conclusion = self.find_conclusion(text);
            let token_count = text.split_whitespace().count();
            trace.add_step(ReasoningStep {
                thought: text.to_string(),
                conclusion,
                evidence: Vec::new(),
                confidence: 0.5,
                token_count,
            });
            return trace;
        }

        // Extract segments between markers
        // Include text before first marker if non-empty
        if split_positions[0] > 0 {
            let pre_text = text[..split_positions[0]].trim();
            if !pre_text.is_empty() {
                let conclusion = self.find_conclusion(pre_text);
                let token_count = pre_text.split_whitespace().count();
                trace.add_step(ReasoningStep {
                    thought: pre_text.to_string(),
                    conclusion,
                    evidence: Vec::new(),
                    confidence: 0.5,
                    token_count,
                });
            }
        }

        for (idx, &pos) in split_positions.iter().enumerate() {
            let end = if idx + 1 < split_positions.len() {
                split_positions[idx + 1]
            } else {
                text.len()
            };
            let segment = text[pos..end].trim();
            if segment.is_empty() {
                continue;
            }
            let conclusion = self.find_conclusion(segment);
            let token_count = segment.split_whitespace().count();
            // Higher confidence for segments with conclusions
            let confidence = if conclusion.is_some() { 0.8 } else { 0.5 };
            trace.add_step(ReasoningStep {
                thought: segment.to_string(),
                conclusion,
                evidence: Vec::new(),
                confidence,
                token_count,
            });
        }

        trace
    }

    /// Look for a conclusion within a text segment.
    fn find_conclusion(&self, text: &str) -> Option<String> {
        for marker in &self.conclusion_markers {
            if let Some(pos) = text.find(marker.as_str()) {
                let after_marker = text[pos + marker.len()..].trim();
                if !after_marker.is_empty() {
                    return Some(after_marker.to_string());
                } else {
                    return Some(marker.clone());
                }
            }
        }
        None
    }
}

/// Stores reasoning traces indexed by signature ID.
pub struct TraceStore {
    traces: HashMap<String, Vec<ReasoningTrace>>,
    max_per_signature: usize,
}

impl TraceStore {
    /// Create a new trace store with the given per-signature capacity.
    pub fn new(max_per_signature: usize) -> Self {
        Self {
            traces: HashMap::new(),
            max_per_signature,
        }
    }

    /// Store a reasoning trace.
    pub fn store(&mut self, trace: ReasoningTrace) {
        let entry = self.traces.entry(trace.signature_id.clone()).or_default();
        if entry.len() >= self.max_per_signature {
            entry.remove(0); // Remove oldest
        }
        entry.push(trace);
    }

    /// Get all traces for a given signature ID.
    pub fn get(&self, signature_id: &str) -> Option<&[ReasoningTrace]> {
        self.traces.get(signature_id).map(|v| v.as_slice())
    }

    /// Get traces with average confidence at or above the threshold.
    pub fn get_best(&self, signature_id: &str, min_confidence: f64) -> Vec<&ReasoningTrace> {
        match self.traces.get(signature_id) {
            Some(traces) => traces
                .iter()
                .filter(|t| t.avg_confidence() >= min_confidence)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Return the total number of traces across all signatures.
    pub fn trace_count(&self) -> usize {
        self.traces.values().map(|v| v.len()).sum()
    }

    /// Return the number of distinct signatures stored.
    pub fn signature_count(&self) -> usize {
        self.traces.len()
    }

    /// Clear all stored traces.
    pub fn clear(&mut self) {
        self.traces.clear();
    }
}

/// Analyzes patterns across multiple reasoning traces.
pub struct TraceAnalyzer;

impl TraceAnalyzer {
    /// Create a new trace analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Find common words/phrases that appear in more than 50% of the given traces.
    pub fn common_patterns(&self, traces: &[ReasoningTrace]) -> Vec<String> {
        if traces.is_empty() {
            return Vec::new();
        }

        let threshold = traces.len() as f64 * 0.5;

        // Count word occurrences across traces (count each word once per trace)
        let mut word_trace_count: HashMap<String, usize> = HashMap::new();

        for trace in traces {
            let mut seen_in_trace: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for step in &trace.steps {
                for word in step.thought.split_whitespace() {
                    let normalized = word.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string();
                    if normalized.len() >= 3 {
                        seen_in_trace.insert(normalized);
                    }
                }
            }
            for word in seen_in_trace {
                *word_trace_count.entry(word).or_insert(0) += 1;
            }
        }

        let mut common: Vec<String> = word_trace_count
            .into_iter()
            .filter(|(_, count)| *count as f64 > threshold)
            .map(|(word, _)| word)
            .collect();

        common.sort();
        common
    }

    /// Compute the average number of steps across traces.
    pub fn avg_steps(&self, traces: &[ReasoningTrace]) -> f64 {
        if traces.is_empty() {
            return 0.0;
        }
        let total: usize = traces.iter().map(|t| t.step_count()).sum();
        total as f64 / traces.len() as f64
    }

    /// Compute the average confidence across all traces.
    pub fn avg_confidence(&self, traces: &[ReasoningTrace]) -> f64 {
        if traces.is_empty() {
            return 0.0;
        }
        let total: f64 = traces.iter().map(|t| t.avg_confidence()).sum();
        total / traces.len() as f64
    }

    /// Compute the fraction of traces with average confidence at or above the threshold.
    pub fn success_rate(&self, traces: &[ReasoningTrace], min_confidence: f64) -> f64 {
        if traces.is_empty() {
            return 0.0;
        }
        let successes = traces
            .iter()
            .filter(|t| t.avg_confidence() >= min_confidence)
            .count();
        successes as f64 / traces.len() as f64
    }
}
