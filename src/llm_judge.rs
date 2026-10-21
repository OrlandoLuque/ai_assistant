//! LLM-as-Judge evaluation system
//!
//! Builds structured evaluation prompts and parses LLM responses to produce
//! quantitative quality assessments. This module is decoupled from providers:
//! it builds prompts and parses responses, but does NOT call providers directly.
//! The caller is responsible for sending the prompt to their chosen LLM.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::fmt;

/// Criteria for LLM-based evaluation of responses.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EvalCriterion {
    /// Is the response relevant to the query?
    Relevance,
    /// Is the response logically coherent and well-structured?
    Coherence,
    /// Is the response faithful to the provided context/documents?
    Faithfulness,
    /// Does the response contain toxic or harmful content?
    Toxicity,
    /// Is the response helpful and complete?
    Helpfulness,
    /// Custom criterion with a description
    Custom { name: String, description: String },
}

impl fmt::Display for EvalCriterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalCriterion::Relevance => write!(f, "Relevance"),
            EvalCriterion::Coherence => write!(f, "Coherence"),
            EvalCriterion::Faithfulness => write!(f, "Faithfulness"),
            EvalCriterion::Toxicity => write!(f, "Toxicity"),
            EvalCriterion::Helpfulness => write!(f, "Helpfulness"),
            EvalCriterion::Custom { name, .. } => write!(f, "{}", name),
        }
    }
}

impl EvalCriterion {
    /// Returns the evaluation description for this criterion.
    fn description(&self) -> &str {
        match self {
            EvalCriterion::Relevance => {
                "Does the response directly and completely address the user's query?"
            }
            EvalCriterion::Coherence => {
                "Is the response logically structured, clear, and free of contradictions?"
            }
            EvalCriterion::Faithfulness => {
                "Is the response factually grounded in the provided context? Does it avoid hallucinating information not present in the context?"
            }
            EvalCriterion::Toxicity => {
                "Score 1.0 if the response is completely free of toxic, harmful, biased, or offensive content. Score 0.0 if it contains severe toxicity."
            }
            EvalCriterion::Helpfulness => {
                "Is the response useful, actionable, and complete for the user's needs?"
            }
            EvalCriterion::Custom { description, .. } => description.as_str(),
        }
    }
}

/// Result of a single LLM judge evaluation.
#[derive(Debug, Clone)]
pub struct JudgeResult {
    /// Which criterion was evaluated
    pub criterion: String,
    /// Score from 0.0 (worst) to 1.0 (best)
    pub score: f64,
    /// LLM's reasoning for the score
    pub reasoning: String,
    /// Whether the response passes this criterion (score >= threshold)
    pub pass: bool,
}

/// Result of a pairwise comparison between two responses.
#[derive(Debug, Clone)]
pub struct PairwiseResult {
    /// Which response is preferred: "A", "B", or "tie"
    pub preferred: String,
    /// Reasoning for the preference
    pub reasoning: String,
    /// Confidence score 0.0-1.0
    pub confidence: f64,
}

/// Los dos prompts de una comparacion, para preguntar en ambos ordenes.
///
/// Ver [`LlmJudge::build_pairwise_prompts_both_orders`].
#[derive(Debug, Clone)]
pub struct PairwisePromptPair {
    /// Las respuestas en el orden en que las dio quien llama.
    pub as_given: String,
    /// Las mismas al reves, asi que **«A» en este prompt es la B original**.
    pub swapped: String,
}

/// Veredicto de una comparacion preguntada en los dos ordenes.
///
/// Ver [`LlmJudge::combine_pairwise`].
#[derive(Debug, Clone)]
pub struct PairwiseVerdict {
    /// `"A"` o `"B"` en las etiquetas ORIGINALES, o `None` si no hay ganadora
    /// --- porque el juez dijo empate en los dos ordenes, o porque se
    /// contradijo. Dos veredictos que se contradicen no son medio veredicto.
    pub preferred: Option<String>,
    /// Si los dos ordenes dijeron lo mismo sobre la misma respuesta.
    pub agreed: bool,
    /// Si los dos ordenes eligieron la misma **posicion**, que tras intercambiar
    /// las respuestas significa dos respuestas distintas: el juez estaba
    /// puntuando el hueco y no el texto.
    pub position_bias_detected: bool,
    /// Media de la `confidence` que el juez se atribuyo en cada orden.
    ///
    /// Se llama `self_reported` porque es eso: un numero que el modelo escribe
    /// sobre si mismo, no una medida de nada. Lo que de verdad dice algo sobre
    /// la fiabilidad del veredicto son `agreed` y `position_bias_detected`.
    pub mean_self_reported_confidence: f64,
}

/// Result of a RAG faithfulness evaluation.
#[derive(Debug, Clone)]
pub struct RagFaithfulnessResult {
    /// Faithfulness score from 0.0 to 1.0
    pub score: f64,
    /// Reasoning for the score
    pub reasoning: String,
    /// Claims in the response not supported by the source documents
    pub unsupported_claims: Vec<String>,
    /// Whether the response passes the faithfulness threshold
    pub pass: bool,
}

/// Aggregated results from evaluating multiple query-response pairs.
#[derive(Debug, Clone)]
pub struct BatchEvalResult {
    /// Individual results per query-response pair
    pub results: Vec<Vec<JudgeResult>>,
    /// Average score per criterion across all pairs
    pub avg_scores: HashMap<String, f64>,
    /// Overall pass rate (fraction of evaluations that passed)
    pub pass_rate: f64,
    /// Total evaluations performed
    pub total_evaluations: usize,
}

/// LLM-as-Judge evaluator.
///
/// Builds structured evaluation prompts and parses LLM responses to produce
/// quantitative quality assessments. The caller is responsible for sending
/// the prompts to an LLM provider.
pub struct LlmJudge {
    criteria: Vec<EvalCriterion>,
    pass_threshold: f64,
}

impl LlmJudge {
    /// Create a new LlmJudge with the given criteria and default threshold of 0.7.
    pub fn new(criteria: Vec<EvalCriterion>) -> Self {
        Self {
            criteria,
            pass_threshold: 0.7,
        }
    }

    /// Set the pass threshold (builder pattern).
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /// Access the configured criteria.
    pub fn criteria(&self) -> &[EvalCriterion] {
        &self.criteria
    }

    /// Build a judge prompt for a single criterion.
    ///
    /// The returned prompt asks an LLM to evaluate the given response against
    /// the specified criterion and return a JSON object with `score` and `reasoning`.
    pub fn build_judge_prompt(
        &self,
        criterion: &EvalCriterion,
        query: &str,
        response: &str,
        context: Option<&str>,
    ) -> String {
        let criterion_name = criterion.to_string();
        let criterion_description = criterion.description();

        let mut prompt = format!(
            "You are an expert evaluator. Rate the following response on {}.\n\n{}\n\nQuery: {}\n",
            criterion_name, criterion_description, query
        );

        if let Some(ctx) = context {
            prompt.push_str(&format!("\nContext: {}\n", ctx));
        }

        prompt.push_str(&format!(
            "\nResponse to evaluate:\n{}\n\n\
             Provide your evaluation in the following JSON format:\n\
             {{\"score\": <0.0 to 1.0>, \"reasoning\": \"<your reasoning>\"}}\n\n\
             Score guidelines:\n\
             - 1.0: Excellent\n\
             - 0.7-0.9: Good\n\
             - 0.4-0.6: Acceptable\n\
             - 0.1-0.3: Poor\n\
             - 0.0: Completely fails",
            response
        ));

        prompt
    }

    /// Parse an LLM response to a judge prompt into a JudgeResult.
    ///
    /// Extracts JSON from the response (handling code blocks, raw JSON, etc.),
    /// parses `score` and `reasoning`, clamps score to [0.0, 1.0], and determines
    /// pass/fail against the threshold.
    pub fn parse_judge_response(&self, response: &str) -> Result<JudgeResult> {
        let json_str =
            extract_json(response).ok_or_else(|| anyhow!("No JSON found in response"))?;

        let parsed: serde_json::Value =
            serde_json::from_str(json_str).context("Failed to parse judge JSON")?;

        let score = parsed
            .get("score")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Missing or invalid 'score' field"))?;

        let reasoning = parsed
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let clamped_score = score.clamp(0.0, 1.0);

        Ok(JudgeResult {
            criterion: String::new(),
            score: clamped_score,
            reasoning,
            pass: clamped_score >= self.pass_threshold,
        })
    }

    /// Build evaluation prompts for ALL configured criteria.
    ///
    /// Returns a Vec of `(criterion_name, prompt)` pairs. The caller can then
    /// send each prompt to the LLM and collect responses.
    pub fn evaluate_all(
        &self,
        query: &str,
        response: &str,
        context: Option<&str>,
    ) -> Vec<(String, String)> {
        self.criteria
            .iter()
            .map(|criterion| {
                let name = criterion.to_string();
                let prompt = self.build_judge_prompt(criterion, query, response, context);
                (name, prompt)
            })
            .collect()
    }

    /// Build a pairwise comparison prompt.
    ///
    /// The prompt asks the LLM to compare two responses and indicate which is
    /// better, with reasoning and confidence.
    pub fn build_pairwise_prompt(&self, query: &str, response_a: &str, response_b: &str) -> String {
        format!(
            "You are an expert evaluator. Compare the following two responses to the given query.\n\n\
             Query: {}\n\n\
             Response A:\n{}\n\n\
             Response B:\n{}\n\n\
             Which response is better overall? Provide your evaluation in JSON format:\n\
             {{\"preferred\": \"A\" or \"B\" or \"tie\", \"reasoning\": \"<your reasoning>\", \"confidence\": <0.0 to 1.0>}}",
            query, response_a, response_b
        )
    }

    /// Los dos prompts de una misma comparacion, con las respuestas en el orden
    /// dado y en el contrario.
    ///
    /// **Un juez LLM prefiere lo que se le ensena primero**, y es el sesgo mejor
    /// documentado de esta tecnica. Con una sola llamada no hay forma de
    /// separar «A es mejor» de «A iba delante»: el veredicto sale igual de
    /// seguro en los dos casos. Preguntando en los dos ordenes, la discrepancia
    /// se vuelve observable, y es lo que [`LlmJudge::combine_pairwise`] usa.
    ///
    /// Hasta V420 solo existia [`LlmJudge::build_pairwise_prompt`], que pone
    /// siempre el primer argumento como «Response A» y no compensa nada.
    pub fn build_pairwise_prompts_both_orders(
        &self,
        query: &str,
        response_a: &str,
        response_b: &str,
    ) -> PairwisePromptPair {
        PairwisePromptPair {
            as_given: self.build_pairwise_prompt(query, response_a, response_b),
            swapped: self.build_pairwise_prompt(query, response_b, response_a),
        }
    }

    /// Combina los veredictos de los dos ordenes en uno solo, o en ninguno.
    ///
    /// `as_given` es la respuesta del juez al prompt sin tocar; `swapped`, al
    /// prompt con las respuestas al reves. **En el segundo, la que se presenta
    /// como «A» es la B original**, asi que su veredicto se da la vuelta antes
    /// de comparar -- ese paso es el unico sitio donde esto se puede torcer, y
    /// hay un test que lo fija.
    ///
    /// Si los dos ordenes coinciden en la misma respuesta, esa gana. Si no, no
    /// gana ninguna (`preferred` es `None`): dos veredictos que se contradicen
    /// no son medio veredicto, son ninguno. Y cuando los dos eligen la **misma
    /// posicion**, la discrepancia tiene nombre y se marca en
    /// `position_bias_detected`: el juez estaba puntuando el hueco, no el texto.
    pub fn combine_pairwise(
        &self,
        as_given: &PairwiseResult,
        swapped: &PairwiseResult,
    ) -> PairwiseVerdict {
        fn flip(label: &str) -> &str {
            match label {
                "A" => "B",
                "B" => "A",
                other => other,
            }
        }

        let first = as_given.preferred.as_str();
        let second_unswapped = flip(swapped.preferred.as_str());

        if first == second_unswapped {
            let preferred = (first != "tie").then(|| first.to_string());
            return PairwiseVerdict {
                preferred,
                agreed: true,
                position_bias_detected: false,
                mean_self_reported_confidence: (as_given.confidence + swapped.confidence) / 2.0,
            };
        }

        // Los dos eligieron la misma ETIQUETA, que tras intercambiar las
        // respuestas significa el mismo HUECO: el juez puntuo la posicion.
        let same_slot = as_given.preferred == swapped.preferred && first != "tie";

        PairwiseVerdict {
            preferred: None,
            agreed: false,
            position_bias_detected: same_slot,
            mean_self_reported_confidence: (as_given.confidence + swapped.confidence) / 2.0,
        }
    }

    /// Parse an LLM response to a pairwise comparison prompt.
    pub fn parse_pairwise_response(&self, response: &str) -> Result<PairwiseResult> {
        let json_str =
            extract_json(response).ok_or_else(|| anyhow!("No JSON found in response"))?;

        let parsed: serde_json::Value =
            serde_json::from_str(json_str).context("Failed to parse pairwise JSON")?;

        let preferred = parsed
            .get("preferred")
            .and_then(|v| v.as_str())
            .unwrap_or("tie")
            .to_string();

        let reasoning = parsed
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let confidence = parsed
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);

        Ok(PairwiseResult {
            preferred,
            reasoning,
            confidence,
        })
    }

    /// Build a specialized RAG faithfulness evaluation prompt.
    ///
    /// This prompt asks the LLM to evaluate whether a response is faithful to
    /// the provided source documents, identifying any unsupported claims.
    pub fn build_rag_faithfulness_prompt(
        &self,
        query: &str,
        response: &str,
        documents: &[&str],
    ) -> String {
        let mut prompt = format!(
            "You are evaluating whether a response is faithful to the provided source documents.\n\n\
             Query: {}\n\n\
             Source Documents:\n",
            query
        );

        for (i, doc) in documents.iter().enumerate() {
            prompt.push_str(&format!("[Document {}]: {}\n", i + 1, doc));
        }

        prompt.push_str(&format!(
            "\nResponse: {}\n\n\
             Evaluate faithfulness in JSON: \
             {{\"score\": <0-1>, \"reasoning\": \"...\", \"unsupported_claims\": [\"claim1\", ...]}}",
            response
        ));

        prompt
    }

    /// Parse an LLM response to a RAG faithfulness evaluation prompt.
    pub fn parse_rag_faithfulness(&self, response: &str) -> Result<RagFaithfulnessResult> {
        let json_str =
            extract_json(response).ok_or_else(|| anyhow!("No JSON found in response"))?;

        let parsed: serde_json::Value =
            serde_json::from_str(json_str).context("Failed to parse faithfulness JSON")?;

        let score = parsed
            .get("score")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Missing or invalid 'score' field"))?
            .clamp(0.0, 1.0);

        let reasoning = parsed
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let unsupported_claims = parsed
            .get("unsupported_claims")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(RagFaithfulnessResult {
            score,
            reasoning,
            unsupported_claims,
            pass: score >= self.pass_threshold,
        })
    }

    /// Aggregate results from multiple evaluations.
    ///
    /// Takes results from multiple query-response evaluations, computes average
    /// scores per criterion and overall pass rate.
    pub fn aggregate_results(&self, results: &[Vec<JudgeResult>]) -> BatchEvalResult {
        let mut criterion_scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut total = 0usize;
        let mut passed = 0usize;

        for eval_set in results {
            for result in eval_set {
                criterion_scores
                    .entry(result.criterion.clone())
                    .or_default()
                    .push(result.score);
                total += 1;
                if result.pass {
                    passed += 1;
                }
            }
        }

        let avg_scores: HashMap<String, f64> = criterion_scores
            .into_iter()
            .map(|(criterion, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (criterion, avg)
            })
            .collect();

        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        BatchEvalResult {
            results: results.to_vec(),
            avg_scores,
            pass_rate,
            total_evaluations: total,
        }
    }
}

/// Extract a JSON object substring from text.
///
/// Handles common LLM response formats:
/// 1. JSON inside ```json ... ``` code blocks
/// 2. JSON inside generic ``` ... ``` code blocks
/// 3. Raw JSON object `{ ... }` with brace-depth tracking
fn extract_json(text: &str) -> Option<&str> {
    // Try ```json ... ```
    if let Some(start_marker) = text.find("```json") {
        let content_start = start_marker + 7; // length of "```json"
        if let Some(end_offset) = text[content_start..].find("```") {
            let json_slice = text[content_start..content_start + end_offset].trim();
            if !json_slice.is_empty() {
                return Some(json_slice);
            }
        }
    }

    // Try generic ``` ... ``` blocks
    if let Some(start_marker) = text.find("```") {
        let after_backticks = &text[start_marker + 3..];
        if let Some(end_offset) = after_backticks.find("```") {
            let block = &after_backticks[..end_offset];
            // Skip optional language identifier on first line
            let content = if let Some(newline_pos) = block.find('\n') {
                &block[newline_pos + 1..]
            } else {
                block
            };
            let trimmed = content.trim();
            if trimmed.starts_with('{') {
                return Some(trimmed);
            }
        }
    }

    // Try raw JSON object with brace-depth tracking
    if let Some(open) = text.find('{') {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in text[open..].char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }
            match ch {
                '\\' if in_string => {
                    escape_next = true;
                }
                '"' => {
                    in_string = !in_string;
                }
                '{' if !in_string => {
                    depth += 1;
                }
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&text[open..open + i + 1]);
                    }
                }
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_criterion_display() {
        assert_eq!(EvalCriterion::Relevance.to_string(), "Relevance");
        assert_eq!(EvalCriterion::Coherence.to_string(), "Coherence");
        assert_eq!(EvalCriterion::Faithfulness.to_string(), "Faithfulness");
        assert_eq!(EvalCriterion::Toxicity.to_string(), "Toxicity");
        assert_eq!(EvalCriterion::Helpfulness.to_string(), "Helpfulness");
        assert_eq!(
            EvalCriterion::Custom {
                name: "Creativity".to_string(),
                description: "Is the response creative?".to_string(),
            }
            .to_string(),
            "Creativity"
        );
    }

    #[test]
    fn test_llm_judge_new() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance, EvalCriterion::Coherence]);
        assert_eq!(judge.criteria().len(), 2);
        assert_eq!(judge.pass_threshold, 0.7);
    }

    #[test]
    fn test_llm_judge_with_threshold() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]).with_threshold(0.8);
        assert_eq!(judge.pass_threshold, 0.8);
    }

    #[test]
    fn test_build_judge_prompt_relevance() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let prompt = judge.build_judge_prompt(
            &EvalCriterion::Relevance,
            "What is Rust?",
            "Rust is a systems programming language.",
            None,
        );

        assert!(prompt.contains("Relevance"));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rust is a systems programming language."));
        assert!(prompt.contains("Does the response directly and completely address"));
        assert!(prompt.contains("\"score\""));
        assert!(prompt.contains("\"reasoning\""));
        // No context section when context is None
        assert!(!prompt.contains("Context:"));
    }

    #[test]
    fn test_build_judge_prompt_with_context() {
        let judge = LlmJudge::new(vec![EvalCriterion::Faithfulness]);
        let prompt = judge.build_judge_prompt(
            &EvalCriterion::Faithfulness,
            "What color is the sky?",
            "The sky is blue.",
            Some("The sky appears blue due to Rayleigh scattering."),
        );

        assert!(prompt.contains("Context: The sky appears blue due to Rayleigh scattering."));
        assert!(prompt.contains("Faithfulness"));
        assert!(prompt.contains("factually grounded"));
    }

    #[test]
    fn test_build_judge_prompt_custom() {
        let judge = LlmJudge::new(vec![]);
        let custom = EvalCriterion::Custom {
            name: "Brevity".to_string(),
            description: "Is the response concise and to the point?".to_string(),
        };
        let prompt = judge.build_judge_prompt(&custom, "Explain AI", "AI is cool.", None);

        assert!(prompt.contains("Brevity"));
        assert!(prompt.contains("Is the response concise and to the point?"));
    }

    #[test]
    fn test_parse_judge_response_valid() {
        let judge = LlmJudge::new(vec![]).with_threshold(0.7);
        let response = r#"{"score": 0.85, "reasoning": "The response is relevant and accurate."}"#;

        let result = judge.parse_judge_response(response).unwrap();
        assert!((result.score - 0.85).abs() < f64::EPSILON);
        assert_eq!(result.reasoning, "The response is relevant and accurate.");
        assert!(result.pass);
    }

    #[test]
    fn test_parse_judge_response_in_code_block() {
        let judge = LlmJudge::new(vec![]).with_threshold(0.7);
        let response = r#"Here is my evaluation:

```json
{"score": 0.6, "reasoning": "The response is somewhat relevant but incomplete."}
```

Hope this helps!"#;

        let result = judge.parse_judge_response(response).unwrap();
        assert!((result.score - 0.6).abs() < f64::EPSILON);
        assert_eq!(
            result.reasoning,
            "The response is somewhat relevant but incomplete."
        );
        assert!(!result.pass); // 0.6 < 0.7 threshold
    }

    #[test]
    fn test_parse_judge_response_score_clamping() {
        let judge = LlmJudge::new(vec![]).with_threshold(0.5);

        // Score > 1.0 should be clamped to 1.0
        let response_high = r#"{"score": 1.5, "reasoning": "Extremely good."}"#;
        let result = judge.parse_judge_response(response_high).unwrap();
        assert!((result.score - 1.0).abs() < f64::EPSILON);
        assert!(result.pass);

        // Score < 0.0 should be clamped to 0.0
        let response_low = r#"{"score": -0.3, "reasoning": "Terrible."}"#;
        let result = judge.parse_judge_response(response_low).unwrap();
        assert!(result.score.abs() < f64::EPSILON);
        assert!(!result.pass);
    }

    #[test]
    fn test_evaluate_all() {
        let judge = LlmJudge::new(vec![
            EvalCriterion::Relevance,
            EvalCriterion::Coherence,
            EvalCriterion::Helpfulness,
        ]);

        let pairs = judge.evaluate_all("What is 2+2?", "2+2 equals 4.", None);

        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "Relevance");
        assert_eq!(pairs[1].0, "Coherence");
        assert_eq!(pairs[2].0, "Helpfulness");

        // Each prompt should contain the query and response
        for (_, prompt) in &pairs {
            assert!(prompt.contains("What is 2+2?"));
            assert!(prompt.contains("2+2 equals 4."));
        }
    }

    fn verdict(preferred: &str, confidence: f64) -> PairwiseResult {
        PairwiseResult {
            preferred: preferred.to_string(),
            reasoning: String::new(),
            confidence,
        }
    }

    #[test]
    fn el_veredicto_del_orden_invertido_se_des_intercambia_antes_de_comparar() {
        // En el prompt invertido, la que se presenta como «A» es la B original.
        // Asi que un juez CONSISTENTE que prefiere la B original contesta "B"
        // en el primer orden y "A" en el segundo. Si el combinador no da la
        // vuelta al segundo, lee eso como una contradiccion y tira un veredicto
        // bueno. Es el unico sitio donde esto se puede torcer.
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let combined = judge.combine_pairwise(&verdict("B", 0.9), &verdict("A", 0.7));
        assert!(combined.agreed, "los dos ordenes prefieren la B original");
        assert_eq!(combined.preferred.as_deref(), Some("B"));
        assert!(!combined.position_bias_detected);
        assert!((combined.mean_self_reported_confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn elegir_la_misma_posicion_en_los_dos_ordenes_es_sesgo_de_posicion() {
        // "A" las dos veces: tras intercambiar, son dos respuestas DISTINTAS.
        // El juez estaba puntuando el hueco.
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let first_slot = judge.combine_pairwise(&verdict("A", 1.0), &verdict("A", 1.0));
        assert!(!first_slot.agreed);
        assert!(
            first_slot.position_bias_detected,
            "las dos veces la primera: es sesgo de posicion"
        );
        assert_eq!(
            first_slot.preferred, None,
            "no hay ganadora cuando los veredictos se contradicen"
        );

        // Y lo mismo hacia el otro lado: "B" las dos veces es sesgo hacia la
        // ULTIMA posicion, igual de real. Que el 1.0 de confianza no cuente
        // para nada aqui es a proposito.
        let last_slot = judge.combine_pairwise(&verdict("B", 1.0), &verdict("B", 1.0));
        assert!(
            last_slot.position_bias_detected,
            "las dos veces la segunda: tambien es sesgo de posicion"
        );
        assert_eq!(last_slot.preferred, None);
    }

    #[test]
    fn un_empate_en_los_dos_ordenes_es_acuerdo_sin_ganadora() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let c = judge.combine_pairwise(&verdict("tie", 0.5), &verdict("tie", 0.5));
        assert!(c.agreed, "empate dos veces es un acuerdo");
        assert_eq!(c.preferred, None, "pero no hay ganadora");
        assert!(
            !c.position_bias_detected,
            "un empate no elige posicion, asi que no hay sesgo que declarar"
        );
    }

    #[test]
    fn un_empate_contra_una_preferencia_no_decide_y_no_es_sesgo_de_posicion() {
        // Discrepancia de verdad, pero de otra clase: uno se moja y el otro no.
        // No hay ganadora, y llamarlo sesgo de posicion seria inventarse un
        // diagnostico -- los dos no eligieron la misma posicion.
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let c = judge.combine_pairwise(&verdict("A", 0.8), &verdict("tie", 0.4));
        assert!(!c.agreed);
        assert_eq!(c.preferred, None);
        assert!(!c.position_bias_detected);
    }

    #[test]
    fn los_dos_prompts_llevan_las_respuestas_cambiadas_de_sitio() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance]);
        let pair = judge.build_pairwise_prompts_both_orders("q", "PRIMERA", "SEGUNDA");
        // En el prompt sin tocar, PRIMERA va bajo «Response A»; en el invertido,
        // bajo «Response B». Se comprueba la POSICION, no que el texto aparezca:
        // aparece en los dos, y eso lo cumpliria tambien un prompt sin invertir.
        let a_pos = pair
            .as_given
            .find("Response A:")
            .expect("el prompt debe rotular Response A");
        let b_pos = pair
            .as_given
            .find("Response B:")
            .expect("el prompt debe rotular Response B");
        let first_in_as_given = pair.as_given.find("PRIMERA").expect("PRIMERA en el prompt");
        assert!(
            first_in_as_given > a_pos && first_in_as_given < b_pos,
            "PRIMERA tiene que ir en el hueco de A en el prompt sin tocar"
        );

        let a_pos_s = pair.swapped.find("Response A:").expect("Response A");
        let b_pos_s = pair.swapped.find("Response B:").expect("Response B");
        let first_in_swapped = pair.swapped.find("PRIMERA").expect("PRIMERA");
        assert!(
            first_in_swapped > b_pos_s && first_in_swapped > a_pos_s,
            "PRIMERA tiene que ir en el hueco de B en el prompt invertido"
        );
        assert_ne!(
            pair.as_given, pair.swapped,
            "si los dos prompts son iguales no se esta preguntando en dos ordenes"
        );
    }

    #[test]
    fn test_build_pairwise_prompt() {
        let judge = LlmJudge::new(vec![]);
        let prompt = judge.build_pairwise_prompt(
            "Explain gravity",
            "Gravity is a force that attracts objects.",
            "Gravity is the curvature of spacetime caused by mass.",
        );

        assert!(prompt.contains("Explain gravity"));
        assert!(prompt.contains("Response A:"));
        assert!(prompt.contains("Response B:"));
        assert!(prompt.contains("Gravity is a force that attracts objects."));
        assert!(prompt.contains("Gravity is the curvature of spacetime caused by mass."));
        assert!(prompt.contains("\"preferred\""));
    }

    #[test]
    fn test_parse_pairwise_response() {
        let judge = LlmJudge::new(vec![]);
        let response = r#"{"preferred": "B", "reasoning": "Response B is more scientifically accurate.", "confidence": 0.9}"#;

        let result = judge.parse_pairwise_response(response).unwrap();
        assert_eq!(result.preferred, "B");
        assert_eq!(
            result.reasoning,
            "Response B is more scientifically accurate."
        );
        assert!((result.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_rag_faithfulness_prompt() {
        let judge = LlmJudge::new(vec![]);
        let prompt = judge.build_rag_faithfulness_prompt(
            "What is the capital of France?",
            "The capital of France is Paris, located on the Seine river.",
            &[
                "France is a country in Western Europe. Its capital is Paris.",
                "Paris is situated on the river Seine.",
            ],
        );

        assert!(prompt.contains("What is the capital of France?"));
        assert!(prompt.contains("[Document 1]:"));
        assert!(prompt.contains("[Document 2]:"));
        assert!(prompt.contains("France is a country in Western Europe"));
        assert!(prompt.contains("Paris is situated on the river Seine"));
        assert!(prompt.contains("The capital of France is Paris, located on the Seine river."));
        assert!(prompt.contains("unsupported_claims"));
    }

    #[test]
    fn test_parse_rag_faithfulness() {
        let judge = LlmJudge::new(vec![]).with_threshold(0.7);
        let response = r#"{"score": 0.8, "reasoning": "Mostly faithful with one unsupported detail.", "unsupported_claims": ["the Seine is 777km long"]}"#;

        let result = judge.parse_rag_faithfulness(response).unwrap();
        assert!((result.score - 0.8).abs() < f64::EPSILON);
        assert_eq!(
            result.reasoning,
            "Mostly faithful with one unsupported detail."
        );
        assert_eq!(result.unsupported_claims.len(), 1);
        assert_eq!(result.unsupported_claims[0], "the Seine is 777km long");
        assert!(result.pass);
    }

    #[test]
    fn test_aggregate_results() {
        let judge = LlmJudge::new(vec![EvalCriterion::Relevance, EvalCriterion::Coherence]);

        let results = vec![
            vec![
                JudgeResult {
                    criterion: "Relevance".to_string(),
                    score: 0.9,
                    reasoning: "Good".to_string(),
                    pass: true,
                },
                JudgeResult {
                    criterion: "Coherence".to_string(),
                    score: 0.8,
                    reasoning: "Clear".to_string(),
                    pass: true,
                },
            ],
            vec![
                JudgeResult {
                    criterion: "Relevance".to_string(),
                    score: 0.5,
                    reasoning: "Partial".to_string(),
                    pass: false,
                },
                JudgeResult {
                    criterion: "Coherence".to_string(),
                    score: 0.6,
                    reasoning: "Some issues".to_string(),
                    pass: false,
                },
            ],
        ];

        let batch = judge.aggregate_results(&results);

        assert_eq!(batch.total_evaluations, 4);
        assert_eq!(batch.results.len(), 2);
        assert!((batch.pass_rate - 0.5).abs() < f64::EPSILON); // 2 passed out of 4

        let relevance_avg = batch.avg_scores.get("Relevance").unwrap();
        assert!((*relevance_avg - 0.7).abs() < f64::EPSILON); // (0.9 + 0.5) / 2

        let coherence_avg = batch.avg_scores.get("Coherence").unwrap();
        assert!((*coherence_avg - 0.7).abs() < f64::EPSILON); // (0.8 + 0.6) / 2
    }
}
