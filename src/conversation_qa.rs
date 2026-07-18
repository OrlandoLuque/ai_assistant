//! Conversation QA harness (V195).
//!
//! Runs multi-turn conversation *scenarios* against a real provider/model and
//! scores each turn with simple, deterministic keyword checks. This formalizes
//! the manual context-following / grounding tests that surfaced the Ollama
//! `num_ctx` truncation bug (V192) into a reusable harness the CLI can run
//! (`ai_cli qa`) as a pre-release quality gate over recommended models.
//!
//! Scoring is intentionally heuristic (case-insensitive substring presence /
//! absence) so it is deterministic and needs no LLM judge. Each scenario runs
//! through a single [`AiAssistant`] so multi-turn context is genuinely
//! exercised (turn N can be asked about a fact stated in turn 1).

use std::time::{Duration, Instant};

use crate::config::AiConfig;
use crate::messages::AiResponse;
use crate::AiAssistant;

/// Default per-turn generation timeout.
pub const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(120);

/// One user turn plus what its answer must / must not contain.
#[derive(Debug, Clone)]
pub struct QaTurn {
    /// The user prompt for this turn.
    pub prompt: String,
    /// Case-insensitive substrings that MUST appear in the answer.
    pub must_contain: Vec<String>,
    /// Case-insensitive substrings that must NOT appear in the answer.
    pub must_not_contain: Vec<String>,
}

impl QaTurn {
    /// A turn with no expectations yet (e.g. a setup/distraction turn).
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
        }
    }

    /// Require `kw` to appear (case-insensitive) in the answer.
    pub fn expect(mut self, kw: impl Into<String>) -> Self {
        self.must_contain.push(kw.into());
        self
    }

    /// Require `kw` to be absent from the answer.
    pub fn forbid(mut self, kw: impl Into<String>) -> Self {
        self.must_not_contain.push(kw.into());
        self
    }

    /// Score a produced answer against this turn's expectations.
    pub fn evaluate(&self, answer: &str) -> QaTurnResult {
        let lower = answer.to_lowercase();
        let missing: Vec<String> = self
            .must_contain
            .iter()
            .filter(|k| !lower.contains(&k.to_lowercase()))
            .cloned()
            .collect();
        let forbidden_hit: Vec<String> = self
            .must_not_contain
            .iter()
            .filter(|k| lower.contains(&k.to_lowercase()))
            .cloned()
            .collect();
        QaTurnResult {
            prompt: self.prompt.clone(),
            answer: answer.to_string(),
            passed: missing.is_empty() && forbidden_hit.is_empty(),
            missing,
            forbidden_hit,
            latency_ms: 0,
            error: None,
        }
    }
}

/// Outcome of running one [`QaTurn`].
#[derive(Debug, Clone)]
pub struct QaTurnResult {
    /// The prompt that was sent.
    pub prompt: String,
    /// The model's answer (empty on error).
    pub answer: String,
    /// Whether every expectation held.
    pub passed: bool,
    /// Expected substrings that were absent.
    pub missing: Vec<String>,
    /// Forbidden substrings that appeared.
    pub forbidden_hit: Vec<String>,
    /// Wall-clock latency of the turn.
    pub latency_ms: u64,
    /// Generation error (timeout / provider failure), if any.
    pub error: Option<String>,
}

/// A multi-turn scenario with optional injected knowledge.
#[derive(Debug, Clone)]
pub struct QaScenario {
    /// Short identifier.
    pub name: String,
    /// What the scenario checks.
    pub description: String,
    /// Knowledge injected as context on every turn (grounding scenarios).
    pub knowledge: Option<String>,
    /// The ordered turns.
    pub turns: Vec<QaTurn>,
}

/// Outcome of running a [`QaScenario`].
#[derive(Debug, Clone)]
pub struct QaScenarioResult {
    /// Scenario name.
    pub name: String,
    /// Whether every turn passed.
    pub passed: bool,
    /// Per-turn results.
    pub turns: Vec<QaTurnResult>,
}

impl QaScenario {
    /// Run this scenario against `config`'s provider/model with the default
    /// per-turn timeout.
    pub fn run(&self, config: &AiConfig) -> QaScenarioResult {
        self.run_with_timeout(config, DEFAULT_TURN_TIMEOUT, false)
    }

    /// Run with an explicit per-turn timeout and context mode. When
    /// `fresh_context` is set, the assistant uses FreshContext mode (only the
    /// latest turn is sent to the model, with the relevant earlier turns
    /// retrieved and injected) — used to verify FreshContext still recalls.
    pub fn run_with_timeout(
        &self,
        config: &AiConfig,
        per_turn: Duration,
        fresh_context: bool,
    ) -> QaScenarioResult {
        let mut assistant = AiAssistant::new();
        assistant.config = config.clone();
        if fresh_context {
            assistant.set_context_mode(crate::ContextMode::FreshContext);
        }

        let mut turns = Vec::with_capacity(self.turns.len());
        for turn in &self.turns {
            // Mirror the CLI: when the knowledge is large, retrieve only the
            // passages relevant to *this* turn's question, so the harness
            // exercises the full retrieval + context stack (not just a huge
            // window).
            let knowledge = match &self.knowledge {
                Some(k) if k.len() > 12_000 => {
                    crate::knowledge_retrieval::select_relevant(k, &turn.prompt, 8_000)
                }
                Some(k) => k.clone(),
                None => String::new(),
            };
            let start = Instant::now();
            assistant.send_message(turn.prompt.clone(), &knowledge);

            let mut answer: Option<String> = None;
            let mut error: Option<String> = None;
            loop {
                match assistant.poll_response() {
                    // `poll_response` also commits the assistant turn to the
                    // conversation, so multi-turn context carries forward.
                    Some(AiResponse::Complete(text)) => {
                        answer = Some(text);
                        break;
                    }
                    Some(AiResponse::Cancelled(text)) => {
                        answer = Some(text);
                        break;
                    }
                    Some(AiResponse::Error(e)) => {
                        error = Some(e);
                        break;
                    }
                    // Chunk (streaming), None (nothing yet), or any other
                    // variant: keep polling until Complete/Error/timeout.
                    _ => {}
                }
                if start.elapsed() > per_turn {
                    error = Some(format!("generation timed out after {per_turn:?}"));
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }

            let latency_ms = start.elapsed().as_millis() as u64;
            let mut result = match &answer {
                Some(a) => turn.evaluate(a),
                None => QaTurnResult {
                    prompt: turn.prompt.clone(),
                    answer: String::new(),
                    passed: false,
                    missing: turn.must_contain.clone(),
                    forbidden_hit: Vec::new(),
                    latency_ms,
                    error,
                },
            };
            result.latency_ms = latency_ms;
            turns.push(result);
        }

        QaScenarioResult {
            name: self.name.clone(),
            passed: turns.iter().all(|t| t.passed),
            turns,
        }
    }
}

/// A small price sheet used by the grounding scenario. The Startup line is the
/// exact fact the model must recover from a large-ish context.
const PRICE_KNOWLEDGE: &str = "\
Commercial license price sheet (EUR per year, VAT excluded):
- Individual / indie developer: 49 EUR per year.
- Startup (fewer than 10 employees): 490 EUR per year.
- Business (10 to 250 employees): 2400 EUR per year.
- Enterprise (more than 250 employees): 12000 EUR per year.
A one-time perpetual TDM license for a single dataset costs 3500 EUR.";

/// Built-in scenarios covering the failure modes found by hand: multi-turn
/// context retention and knowledge grounding.
pub fn builtin_scenarios() -> Vec<QaScenario> {
    vec![
        QaScenario {
            name: "context_recall".to_string(),
            description: "Recalls a fact stated in turn 1 after an unrelated turn".to_string(),
            knowledge: None,
            turns: vec![
                QaTurn::new(
                    "Me llamo Orlando y mi color favorito es el verde. Recuérdalo, \
                     te preguntaré luego.",
                ),
                QaTurn::new("Cuéntame un chiste muy corto sobre programadores."),
                QaTurn::new("¿Cómo me llamo y cuál es mi color favorito? Responde en una frase.")
                    .expect("Orlando")
                    .expect("verde"),
            ],
        },
        QaScenario {
            name: "grounded_price".to_string(),
            description: "Answers a specific price present in the injected knowledge".to_string(),
            knowledge: Some(PRICE_KNOWLEDGE.to_string()),
            turns: vec![QaTurn::new(
                "Según el pliego, ¿cuánto cuesta al año la licencia para una Startup? \
                 Responde solo con el importe exacto.",
            )
            .expect("490")],
        },
        // --- Larger / more complex conversations ---
        QaScenario {
            name: "multi_fact_tracking".to_string(),
            description: "Tracks several facts over turns and recalls them after distractions"
                .to_string(),
            knowledge: None,
            turns: vec![
                QaTurn::new(
                    "Te doy varios datos sobre mí, guárdalos: me llamo Ana, vivo en Sevilla, \
                     tengo un perro llamado Toby y trabajo de arquitecta.",
                ),
                QaTurn::new("Cuéntame un dato curioso muy corto sobre el mar."),
                QaTurn::new("Cuéntame otro dato curioso corto, ahora sobre el espacio."),
                QaTurn::new("¿En qué ciudad vivo y cómo se llama mi perro? Una frase.")
                    .expect("Sevilla")
                    .expect("Toby"),
                QaTurn::new("¿De qué trabajo? Una palabra.").expect("arquitecta"),
            ],
        },
        QaScenario {
            name: "fact_update".to_string(),
            description: "Uses the most recent value after a fact is updated mid-conversation"
                .to_string(),
            knowledge: None,
            turns: vec![
                QaTurn::new("Mi color favorito es el azul."),
                QaTurn::new("Espera, cámbialo: a partir de ahora mi color favorito es el rojo."),
                QaTurn::new("Hablemos de otra cosa: dime un número entre 1 y 10."),
                QaTurn::new("¿Cuál es mi color favorito ahora? Una sola palabra.").expect("rojo"),
            ],
        },
        QaScenario {
            name: "multi_grounded".to_string(),
            description:
                "Answers several prices across turns from a large doc (exercises retrieval)"
                    .to_string(),
            knowledge: Some(big_support_plans_knowledge()),
            turns: vec![
                QaTurn::new(
                    "Según el catálogo, ¿cuánto cuesta al mes el plan Premium? Solo el importe.",
                )
                .expect("99"),
                QaTurn::new("¿Y el plan Estándar? Solo el importe.").expect("49"),
                QaTurn::new("¿Y el plan Empresa? Solo el importe.").expect("290"),
            ],
        },
    ]
}

/// A large support-plans catalogue with the four plan prices buried in filler,
/// so the `multi_grounded` scenario exercises per-turn retrieval.
fn big_support_plans_knowledge() -> String {
    let mut doc = String::new();
    for i in 0..150 {
        doc.push_str(&format!(
            "Sección {i}: notas de integración, SLA, límites de API y detalles de \
             despliegue del producto para el perfil de build {i}.\n\n"
        ));
    }
    doc.push_str(
        "Catálogo de planes de soporte (EUR al mes): Básico 19, Estándar 49, \
         Premium 99, Empresa 290.\n\n",
    );
    for i in 150..300 {
        doc.push_str(&format!(
            "Sección {i}: más notas de arquitectura, colas, caché y observabilidad \
             del perfil {i}.\n\n"
        ));
    }
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_keyword_presence_and_absence() {
        let turn = QaTurn::new("q")
            .expect("Orlando")
            .expect("verde")
            .forbid("azul");

        // All expectations satisfied (case-insensitive).
        let ok = turn.evaluate("Te llamas ORLANDO y tu color es Verde.");
        assert!(ok.passed);
        assert!(ok.missing.is_empty() && ok.forbidden_hit.is_empty());

        // Missing an expected keyword fails.
        let miss = turn.evaluate("Te llamas Orlando.");
        assert!(!miss.passed);
        assert_eq!(miss.missing, vec!["verde".to_string()]);

        // A forbidden keyword fails.
        let forb = turn.evaluate("Orlando, tu color es verde, o quizá azul.");
        assert!(!forb.passed);
        assert_eq!(forb.forbidden_hit, vec!["azul".to_string()]);
    }

    #[test]
    fn builtins_are_well_formed() {
        let scenarios = builtin_scenarios();
        assert!(scenarios.iter().any(|s| s.name == "context_recall"));
        let grounded = scenarios
            .iter()
            .find(|s| s.name == "grounded_price")
            .expect("grounded_price scenario present");
        // The answer it checks for is actually in the injected knowledge.
        assert!(grounded.knowledge.as_deref().unwrap().contains("490"));
        assert!(grounded.turns[0].must_contain.contains(&"490".to_string()));
    }
}
