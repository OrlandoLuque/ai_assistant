//! LLM-as-Judge automated grading: rubrics, metrics, and calibration.

use serde::{Deserialize, Serialize};

use super::types::EvalMetric;

// ============================================================================
// 4.3 LLM-as-Judge Automated Grading
// ============================================================================

/// A single criterion in a judge rubric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCriterion {
    /// Name of the criterion (e.g., "relevance", "fluency").
    pub name: String,
    /// Description of what this criterion measures.
    pub description: String,
    /// Weight of this criterion in the overall score.
    pub weight: f64,
    /// Score scale as (min, max).
    pub scale: (f64, f64),
}

/// A rubric defining multiple criteria for judging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeRubric {
    /// The criteria that make up this rubric.
    pub criteria: Vec<JudgeCriterion>,
    /// Overall instruction for the judge.
    pub overall_instruction: String,
}

impl JudgeRubric {
    /// Create a new rubric with the given overall instruction.
    pub fn new(overall_instruction: String) -> Self {
        Self {
            criteria: Vec::new(),
            overall_instruction,
        }
    }

    /// Add a criterion to this rubric.
    pub fn add_criterion(&mut self, criterion: JudgeCriterion) {
        self.criteria.push(criterion);
    }

    /// Return the number of criteria.
    pub fn criterion_count(&self) -> usize {
        self.criteria.len()
    }

    /// Return the sum of all criterion weights.
    pub fn total_weight(&self) -> f64 {
        self.criteria.iter().map(|c| c.weight).sum()
    }

    /// Validate the rubric: checks that weights are positive and criteria are non-empty.
    pub fn validate(&self) -> Result<(), String> {
        if self.criteria.is_empty() {
            return Err("Rubric must have at least one criterion".to_string());
        }
        for criterion in &self.criteria {
            if criterion.weight <= 0.0 {
                return Err(format!(
                    "Criterion '{}' has non-positive weight: {}",
                    criterion.name, criterion.weight
                ));
            }
        }
        Ok(())
    }
}

/// The result of an LLM-as-judge evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptJudgeResult {
    /// Media PONDERADA de las notas por criterio, normalizada a `0.0..=1.0`.
    ///
    /// Hasta V416 este campo se documentaba como «The weighted overall score» y
    /// **no estaba ponderado**: era el numero que devolvia el `scorer` para toda
    /// la rubrica, sin tocar. El campo `weight` de cada criterio solo se usaba
    /// para sumarlo y comprobar que no era cero.
    pub overall_score: f64,
    /// Nota de cada criterio, **puntuada por separado**.
    ///
    /// Antes se fabricaban: todos los criterios recibian el mismo valor global
    /// reescalado a su rango. Quien leyera esta lista creia que se habia
    /// evaluado cada criterio; no se habia evaluado ninguno.
    pub per_criterion: Vec<CriterionScore>,
    /// Explicacion del veredicto global, si quien puntua la da.
    pub reasoning: String,
    /// Cuanto COINCIDEN los criterios entre si, de `0.0` (maximo desacuerdo) a
    /// `1.0` (todos la misma nota). `None` con menos de dos criterios, porque
    /// con uno solo no hay dispersion que medir.
    ///
    /// **No es la confianza en que el veredicto sea correcto**, y el nombre lo
    /// dice a proposito. Hasta V416 aqui habia un `confidence: 0.8` constante
    /// --un numero inventado con pinta de medido--, y hasta habia un test que
    /// lo fijaba. Un juez cuyos criterios se contradicen es un dato util; un
    /// 0,8 fijo no es dato de nada.
    pub criterion_agreement: Option<f64>,
}

/// Score for a single criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    /// Name of the criterion that was scored.
    pub criterion_name: String,
    /// La nota, en la escala declarada por el propio criterio (`scale`).
    pub score: f64,
    /// Explicacion de ESTA nota, si quien puntua la da.
    ///
    /// `None` cuando el `scorer` solo devuelve un numero, que es el caso
    /// habitual. Es `Option` y no una frase generica a proposito: antes aqui se
    /// escribia «Score for 'X' based on overall evaluation», que **parece** una
    /// explicacion y no explica nada.
    pub reasoning: Option<String>,
}

/// Configuration for an LLM-as-judge evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JudgeConfig {
    /// The rubric to use for evaluation.
    pub rubric: JudgeRubric,
    /// Few-shot examples of judge evaluations.
    pub few_shot_examples: Vec<JudgeExample>,
    /// Temperature for judge inference.
    pub temperature: f64,
}

/// A few-shot example demonstrating judge behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeExample {
    /// The input that was evaluated.
    pub input: String,
    /// The output that was evaluated.
    pub output: String,
    /// The expected score.
    pub expected_score: f64,
    /// The expected reasoning.
    pub reasoning: String,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            rubric: JudgeRubric::new(String::new()),
            few_shot_examples: Vec::new(),
            temperature: 0.0,
        }
    }
}

/// An evaluation metric that uses an LLM-as-judge scoring function.
pub struct JudgeMetric {
    pub(crate) config: JudgeConfig,
    pub(crate) scorer: CriterionScorer,
}

/// Funcion que puntua **un** criterio: `(input, output, criterio) -> nota`.
///
/// Recibe el criterio entero, no la rubrica, porque ese es el cambio que hace
/// que los pesos sirvan para algo: hasta V416 el `scorer` devolvia **un** numero
/// para toda la rubrica, asi que no habia notas por criterio que ponderar y el
/// campo `weight` era decorativo.
///
/// La nota se espera dentro de `criterio.scale`; si se sale, se recorta.
pub type CriterionScorer = Box<dyn Fn(&str, &str, &JudgeCriterion) -> f64 + Send + Sync>;

impl JudgeMetric {
    /// Construye el juez, **validando la rubrica**.
    ///
    /// Devuelve `Err` si [`JudgeRubric::validate`] falla. Ese validador existia
    /// desde el principio y no lo llamaba nadie (otro caso de «construido y sin
    /// conectar»), asi que una rubrica sin criterios o con pesos negativos se
    /// aceptaba y producia numeros sin sentido en silencio.
    pub fn new(config: JudgeConfig, scorer: CriterionScorer) -> Result<Self, String> {
        config.rubric.validate()?;
        Ok(Self { config, scorer })
    }

    /// Evalua un par entrada/salida puntuando **cada criterio por separado** y
    /// combinandolos con sus pesos.
    ///
    /// Las notas se normalizan a `0..=1` con la `scale` de cada criterio antes
    /// de promediar, porque si no, un criterio puntuado de 0 a 10 aplastaria a
    /// uno de 0 a 1 por el mero hecho de usar otra regla.
    pub fn evaluate(&self, input: &str, output: &str) -> PromptJudgeResult {
        let criteria = &self.config.rubric.criteria;
        let total_weight = self.config.rubric.total_weight();

        let mut per_criterion = Vec::with_capacity(criteria.len());
        let mut normalised = Vec::with_capacity(criteria.len());
        let mut weighted_sum = 0.0;

        for criterion in criteria {
            let (min_s, max_s) = criterion.scale;
            let (lo, hi) = if min_s <= max_s {
                (min_s, max_s)
            } else {
                (max_s, min_s)
            };

            let score = (self.scorer)(input, output, criterion).clamp(lo, hi);
            let span = hi - lo;
            let unit = if span.abs() > f64::EPSILON {
                (score - lo) / span
            } else {
                // Escala degenerada (min == max): no hay rango que normalizar.
                0.0
            };

            weighted_sum += unit * criterion.weight;
            normalised.push(unit);
            per_criterion.push(CriterionScore {
                criterion_name: criterion.name.clone(),
                score,
                // Un `scorer` numerico no da explicacion, y decirlo con `None`
                // es mas honesto que rellenarlo con una frase.
                reasoning: None,
            });
        }

        // `validate()` garantiza pesos positivos y al menos un criterio, pero
        // `evaluate` no puede asumir que se llamo: el campo `config` es publico.
        let overall_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let criterion_agreement = if normalised.len() >= 2 {
            let max = normalised.iter().copied().fold(f64::MIN, f64::max);
            let min = normalised.iter().copied().fold(f64::MAX, f64::min);
            Some((1.0 - (max - min)).clamp(0.0, 1.0))
        } else {
            None
        };

        PromptJudgeResult {
            overall_score,
            per_criterion,
            reasoning: format!(
                "Evaluated output against rubric: {}",
                self.config.rubric.overall_instruction
            ),
            criterion_agreement,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &JudgeConfig {
        &self.config
    }
}

impl EvalMetric for JudgeMetric {
    fn name(&self) -> &str {
        "judge_metric"
    }

    fn score(&self, predicted: &str, expected: &str) -> f64 {
        // Delega en `evaluate` a proposito: antes llamaba al `scorer` por su
        // cuenta, asi que `score()` y `evaluate().overall_score` podian dar
        // numeros distintos para la misma entrada. Una sola definicion.
        self.evaluate(predicted, expected).overall_score
    }
}

/// A set of calibration examples for measuring judge accuracy.
pub struct CalibrationSet {
    examples: Vec<CalibrationExample>,
}

/// A single calibration example with a known human score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExample {
    /// The input text.
    pub input: String,
    /// The output text to judge.
    pub output: String,
    /// The human-assigned ground truth score.
    pub human_score: f64,
}

impl CalibrationSet {
    /// Create a new empty calibration set.
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Add a calibration example.
    pub fn add(&mut self, example: CalibrationExample) {
        self.examples.push(example);
    }

    /// Error medio absoluto entre lo que puntua el juez y lo que puntuo una
    /// persona. **Es la medida que dice si el juez sirve para algo**: sin ella,
    /// las notas son numeros de procedencia desconocida.
    ///
    /// Devuelve `None` con el conjunto vacio. Antes devolvia `0.0`, que es
    /// indistinguible de «acuerdo perfecto» -- el peor valor posible para
    /// confundirse, porque un conjunto vacio se leia como un juez impecable.
    pub fn calibration_error(&self, judge: &JudgeMetric) -> Option<f64> {
        if self.examples.is_empty() {
            return None;
        }

        let total_error: f64 = self
            .examples
            .iter()
            .map(|ex| {
                // Por `evaluate` y no llamando al `scorer` a pelo: lo segundo se
                // saltaba los pesos y la normalizacion por escala, asi que el
                // error se medía contra un numero que NO es el que el juez
                // publica. Dos definiciones de «la nota del juez» es una de mas.
                let judge_score = judge.evaluate(&ex.input, &ex.output).overall_score;
                (judge_score - ex.human_score).abs()
            })
            .sum();

        Some(total_error / self.examples.len() as f64)
    }

    /// Return a slice of all calibration examples.
    pub fn examples(&self) -> &[CalibrationExample] {
        &self.examples
    }

    /// Return the number of calibration examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return whether the calibration set is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}
