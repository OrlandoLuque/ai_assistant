//! A/B testing framework for prompt and model experimentation.
//!
//! Supports multi-variant experiments with deterministic user assignment,
//! metric tracking, statistical significance testing, and early stopping.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in the A/B testing framework.
#[derive(Debug, Clone, PartialEq)]
pub enum AbTestError {
    /// The requested experiment was not found.
    ExperimentNotFound,
    /// The experiment has already been stopped or completed.
    AlreadyStopped,
    /// The variant configuration is invalid (e.g. fewer than 2 variants).
    InvalidVariants,
    /// Not enough data has been collected to produce a result.
    InsufficientData,
}

impl std::fmt::Display for AbTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExperimentNotFound => write!(f, "experiment not found"),
            Self::AlreadyStopped => write!(f, "experiment already stopped"),
            Self::InvalidVariants => write!(f, "invalid variant configuration"),
            Self::InsufficientData => write!(f, "insufficient data for analysis"),
        }
    }
}

impl std::error::Error for AbTestError {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    Running,
    Stopped,
    Completed,
}

/// A single variant within an experiment.
#[derive(Debug, Clone)]
pub struct ExperimentVariant {
    pub name: String,
    pub description: String,
    /// Traffic weight in 0.0..=1.0. Weights across all variants should sum to
    /// approximately 1.0, but are normalized internally.
    pub traffic_weight: f64,
}

impl ExperimentVariant {
    pub fn new(name: &str, description: &str, traffic_weight: f64) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            traffic_weight: traffic_weight.clamp(0.0, 1.0),
        }
    }
}

/// An experiment definition.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub variants: Vec<ExperimentVariant>,
    pub status: ExperimentStatus,
    pub created_at: u64,
    pub stopped_at: Option<u64>,
}

/// The result of assigning a user to a variant.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantAssignment {
    pub experiment_id: String,
    pub user_id: String,
    pub variant_index: usize,
    pub variant_name: String,
}

/// A single metric data point recorded against a variant.
#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub variant_index: usize,
    pub value: f64,
    pub timestamp: u64,
}

/// Aggregate statistics for one variant.
#[derive(Debug, Clone)]
pub struct VariantStats {
    pub variant_name: String,
    pub sample_size: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub conversion_count: usize,
    pub conversion_rate: f64,
}

/// Overall result of an experiment analysis.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub variants: Vec<VariantStats>,
    pub is_significant: bool,
    pub p_value: f64,
    pub confidence_level: f64,
    pub recommended_variant: Option<String>,
}

// ---------------------------------------------------------------------------
// FNV-1a hash (deterministic, same pattern as telemetry.rs)
// ---------------------------------------------------------------------------

fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// VariantAssigner
// ---------------------------------------------------------------------------

/// Deterministic variant assignment using FNV-1a hashing.
pub struct VariantAssigner;

impl VariantAssigner {
    /// Assign a user to a variant index.
    ///
    /// The assignment is deterministic: the same `(user_id, experiment_id)`
    /// pair always returns the same variant index.  The `weights` slice
    /// contains the traffic weight for each variant and is normalized
    /// internally.
    pub fn assign(user_id: &str, experiment_id: &str, weights: &[f64]) -> usize {
        if weights.is_empty() {
            return 0;
        }
        if weights.len() == 1 {
            return 0;
        }

        let key = format!("{}:{}", user_id, experiment_id);
        let hash = fnv1a_hash(key.as_bytes());
        // Map hash to [0.0, 1.0)
        let normalized = (hash % 10000) as f64 / 10000.0;

        // Normalize weights
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }

        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w / total;
            if normalized < cumulative {
                return i;
            }
        }

        // Fallback to last variant (floating point edge case)
        weights.len() - 1
    }
}

// ---------------------------------------------------------------------------
// SignificanceCalculator
// ---------------------------------------------------------------------------

/// Statistical significance calculators for experiment analysis.
pub struct SignificanceCalculator;

impl SignificanceCalculator {
    /// Welch's two-sample t-test.
    ///
    /// Returns `(t_statistic, p_value)`.  The p-value is two-tailed.
    pub fn welch_t_test(group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
        if group_a.len() < 2 || group_b.len() < 2 {
            return (0.0, 1.0);
        }

        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;

        let mean_a = group_a.iter().sum::<f64>() / n_a;
        let mean_b = group_b.iter().sum::<f64>() / n_b;

        let var_a = group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
        let var_b = group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

        let se_sq = var_a / n_a + var_b / n_b;
        if se_sq == 0.0 {
            return (0.0, 1.0);
        }
        let se = se_sq.sqrt();
        let t = (mean_a - mean_b).abs() / se;

        // Welch-Satterthwaite degrees of freedom
        let num = se_sq * se_sq;
        let denom =
            (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
        if denom == 0.0 {
            return (t, 1.0);
        }
        let df = num / denom;

        // Two-tailed p-value from Student's t-distribution
        let p = 2.0 * (1.0 - student_t_cdf(t, df));
        (t, p.clamp(0.0, 1.0))
    }

    /// Chi-square test for homogeneity of proportions.
    ///
    /// Each element of `conversions` is `(successes, total_trials)` for one
    /// variant.  Returns `(chi2_statistic, p_value)`.
    pub fn chi_square_test(conversions: &[(usize, usize)]) -> (f64, f64) {
        if conversions.len() < 2 {
            return (0.0, 1.0);
        }

        let total_success: usize = conversions.iter().map(|(s, _)| s).sum();
        let total_n: usize = conversions.iter().map(|(_, n)| n).sum();
        if total_n == 0 {
            return (0.0, 1.0);
        }

        let overall_rate = total_success as f64 / total_n as f64;

        let mut chi2 = 0.0;
        for &(success, n) in conversions {
            if n == 0 {
                continue;
            }
            let failure = n - success;
            let expected_success = n as f64 * overall_rate;
            let expected_failure = n as f64 * (1.0 - overall_rate);

            if expected_success > 0.0 {
                chi2 += (success as f64 - expected_success).powi(2) / expected_success;
            }
            if expected_failure > 0.0 {
                chi2 += (failure as f64 - expected_failure).powi(2) / expected_failure;
            }
        }

        let df = (conversions.len() - 1) as f64;
        let p = 1.0 - chi_square_cdf(chi2, df);
        (chi2, p.clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------------------
// Statistical helper functions
// ---------------------------------------------------------------------------

/// Log-gamma function via Lanczos approximation (g=7, n=9 coefficients).
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        let ln_pi = std::f64::consts::PI.ln();
        return ln_pi - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}

/// Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let ln_prefix =
        a * x.ln() + b * (1.0 - x).ln() - ln_gamma(a) - ln_gamma(b) + ln_gamma(a + b) - a.ln();

    let max_iter = 200;
    let epsilon = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step
        let numerator_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + numerator_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        result *= d * c;

        // Odd step
        let numerator_odd =
            -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + numerator_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    (ln_prefix + result.ln()).exp()
}

/// CDF of Student's t-distribution with `df` degrees of freedom, evaluated at `t`.
fn student_t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    let ibeta = regularized_incomplete_beta(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

/// CDF of the chi-square distribution with `k` degrees of freedom.
///
/// Uses the regularized lower incomplete gamma function:
///   P(k/2, x/2)
fn chi_square_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    regularized_lower_gamma(k / 2.0, x / 2.0)
}

/// Regularized lower incomplete gamma function P(a, x) = gamma(a,x) / Gamma(a).
///
/// Uses the series expansion for x < a+1, and the continued fraction otherwise.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_cf(a, x)
    }
}

/// Lower incomplete gamma via series expansion.
fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let epsilon = 1e-14;

    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..=max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * epsilon {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Upper incomplete gamma via continued fraction (Legendre).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let epsilon = 1e-14;
    let tiny = 1e-30;

    let mut b_n = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b_n;
    let mut h = d;

    for n in 1..=max_iter {
        let n_f = n as f64;
        let a_n = -n_f * (n_f - a);
        b_n += 2.0;
        d = a_n * d + b_n;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b_n + a_n / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// ExperimentManager
// ---------------------------------------------------------------------------

/// Manages A/B test experiments, user assignment, metric recording, and
/// statistical analysis.
pub struct ExperimentManager {
    experiments: HashMap<String, Experiment>,
    /// Metric values per experiment.  Key = experiment_id.
    metrics: HashMap<String, Vec<MetricRecord>>,
    /// Conversion counts per experiment.  Key = experiment_id, inner key = variant_index.
    conversions: HashMap<String, HashMap<usize, usize>>,
    /// Total observations per variant (for conversion rate denominator).
    observations: HashMap<String, HashMap<usize, usize>>,
    next_id: u64,
}

impl ExperimentManager {
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            metrics: HashMap::new(),
            conversions: HashMap::new(),
            observations: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new experiment.  Returns the experiment ID on success.
    ///
    /// At least 2 variants are required.
    pub fn create_experiment(
        &mut self,
        name: &str,
        variants: Vec<ExperimentVariant>,
    ) -> Result<String, AbTestError> {
        if variants.len() < 2 {
            return Err(AbTestError::InvalidVariants);
        }

        let id = format!("exp_{}", self.next_id);
        self.next_id += 1;

        let experiment = Experiment {
            id: id.clone(),
            name: name.to_string(),
            variants,
            status: ExperimentStatus::Running,
            created_at: now_secs(),
            stopped_at: None,
        };

        self.experiments.insert(id.clone(), experiment);
        self.metrics.insert(id.clone(), Vec::new());
        self.conversions.insert(id.clone(), HashMap::new());
        self.observations.insert(id.clone(), HashMap::new());

        Ok(id)
    }

    /// Deterministically assign a user to a variant of the given experiment.
    pub fn assign_user(
        &self,
        experiment_id: &str,
        user_id: &str,
    ) -> Result<VariantAssignment, AbTestError> {
        let experiment = self
            .experiments
            .get(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        let weights: Vec<f64> = experiment.variants.iter().map(|v| v.traffic_weight).collect();
        let variant_index = VariantAssigner::assign(user_id, experiment_id, &weights);
        let variant_name = experiment.variants[variant_index].name.clone();

        Ok(VariantAssignment {
            experiment_id: experiment_id.to_string(),
            user_id: user_id.to_string(),
            variant_index,
            variant_name,
        })
    }

    /// Record a metric value for a variant.
    pub fn record_metric(&mut self, experiment_id: &str, variant_index: usize, value: f64) {
        let record = MetricRecord {
            variant_index,
            value,
            timestamp: now_secs(),
        };
        self.metrics
            .entry(experiment_id.to_string())
            .or_insert_with(Vec::new)
            .push(record);
    }

    /// Record a conversion event for a variant.
    pub fn record_conversion(&mut self, experiment_id: &str, variant_index: usize) {
        *self
            .conversions
            .entry(experiment_id.to_string())
            .or_insert_with(HashMap::new)
            .entry(variant_index)
            .or_insert(0) += 1;

        *self
            .observations
            .entry(experiment_id.to_string())
            .or_insert_with(HashMap::new)
            .entry(variant_index)
            .or_insert(0) += 1;
    }

    /// Record a non-conversion observation (user was exposed but did not convert).
    pub fn record_exposure(&mut self, experiment_id: &str, variant_index: usize) {
        *self
            .observations
            .entry(experiment_id.to_string())
            .or_insert_with(HashMap::new)
            .entry(variant_index)
            .or_insert(0) += 1;
    }

    /// Compute experiment results with the given confidence level (e.g. 0.95).
    pub fn get_results(
        &self,
        experiment_id: &str,
        confidence: f64,
    ) -> Result<ExperimentResult, AbTestError> {
        let experiment = self
            .experiments
            .get(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        let metrics = self
            .metrics
            .get(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        let conversions = self.conversions.get(experiment_id);
        let observations = self.observations.get(experiment_id);

        // Group metric values by variant index
        let num_variants = experiment.variants.len();
        let mut variant_values: Vec<Vec<f64>> = vec![Vec::new(); num_variants];
        for record in metrics {
            if record.variant_index < num_variants {
                variant_values[record.variant_index].push(record.value);
            }
        }

        // Check that we have at least some data
        let total_samples: usize = variant_values.iter().map(|v| v.len()).sum();
        if total_samples == 0 {
            return Err(AbTestError::InsufficientData);
        }

        // Build per-variant stats
        let mut variant_stats = Vec::with_capacity(num_variants);
        for (i, variant) in experiment.variants.iter().enumerate() {
            let vals = &variant_values[i];
            let sample_size = vals.len();
            let (mean, std_dev, min, max) = if sample_size > 0 {
                compute_stats(vals)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            let conversion_count = conversions
                .and_then(|c| c.get(&i))
                .copied()
                .unwrap_or(0);
            let observation_count = observations
                .and_then(|o| o.get(&i))
                .copied()
                .unwrap_or(0);
            let conversion_rate = if observation_count > 0 {
                conversion_count as f64 / observation_count as f64
            } else {
                0.0
            };

            variant_stats.push(VariantStats {
                variant_name: variant.name.clone(),
                sample_size,
                mean,
                std_dev,
                min,
                max,
                conversion_count,
                conversion_rate,
            });
        }

        // Statistical significance: compare first two variants (control vs treatment)
        let (is_significant, p_value) = if variant_values[0].len() >= 2
            && variant_values[1].len() >= 2
        {
            let (_, p) =
                SignificanceCalculator::welch_t_test(&variant_values[0], &variant_values[1]);
            let alpha = 1.0 - confidence;
            (p < alpha, p)
        } else {
            (false, 1.0)
        };

        // Determine recommended variant (the one with the highest mean, if significant)
        let recommended_variant = if is_significant {
            variant_stats
                .iter()
                .filter(|s| s.sample_size > 0)
                .max_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal))
                .map(|s| s.variant_name.clone())
        } else {
            None
        };

        Ok(ExperimentResult {
            experiment_id: experiment_id.to_string(),
            variants: variant_stats,
            is_significant,
            p_value,
            confidence_level: confidence,
            recommended_variant,
        })
    }

    /// Stop a running experiment.
    pub fn stop_experiment(&mut self, experiment_id: &str) -> Result<(), AbTestError> {
        let experiment = self
            .experiments
            .get_mut(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        if experiment.status != ExperimentStatus::Running {
            return Err(AbTestError::AlreadyStopped);
        }

        experiment.status = ExperimentStatus::Stopped;
        experiment.stopped_at = Some(now_secs());
        Ok(())
    }

    /// List all experiments.
    pub fn list_experiments(&self) -> Vec<&Experiment> {
        self.experiments.values().collect()
    }

    /// Check whether an experiment should be stopped early because one
    /// variant is already a clear winner at the given p-value threshold.
    ///
    /// Returns `Ok(Some(winner_name))` if significance was reached, `Ok(None)`
    /// otherwise.
    pub fn check_early_stopping(
        &self,
        experiment_id: &str,
        p_threshold: f64,
    ) -> Result<Option<String>, AbTestError> {
        let experiment = self
            .experiments
            .get(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        let metrics = self
            .metrics
            .get(experiment_id)
            .ok_or(AbTestError::ExperimentNotFound)?;

        let num_variants = experiment.variants.len();
        let mut variant_values: Vec<Vec<f64>> = vec![Vec::new(); num_variants];
        for record in metrics {
            if record.variant_index < num_variants {
                variant_values[record.variant_index].push(record.value);
            }
        }

        // Need at least 2 samples per variant
        if variant_values[0].len() < 2 || variant_values[1].len() < 2 {
            return Ok(None);
        }

        let (_, p) =
            SignificanceCalculator::welch_t_test(&variant_values[0], &variant_values[1]);

        if p < p_threshold {
            // Identify the winner by highest mean
            let mean_0 = variant_values[0].iter().sum::<f64>() / variant_values[0].len() as f64;
            let mean_1 = variant_values[1].iter().sum::<f64>() / variant_values[1].len() as f64;
            let winner_idx = if mean_0 >= mean_1 { 0 } else { 1 };
            Ok(Some(experiment.variants[winner_idx].name.clone()))
        } else {
            Ok(None)
        }
    }
}

impl Default for ExperimentManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute mean, standard deviation, min, and max of a non-empty slice.
fn compute_stats(values: &[f64]) -> (f64, f64, f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = if values.len() > 1 {
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std_dev, min, max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- VariantAssigner tests -----------------------------------------------

    #[test]
    fn test_deterministic_assignment() {
        let w = vec![0.5, 0.5];
        let a1 = VariantAssigner::assign("user_42", "exp_1", &w);
        let a2 = VariantAssigner::assign("user_42", "exp_1", &w);
        let a3 = VariantAssigner::assign("user_42", "exp_1", &w);
        assert_eq!(a1, a2);
        assert_eq!(a2, a3);
    }

    #[test]
    fn test_different_users_distributed() {
        let w = vec![0.5, 0.5];
        let mut counts = [0usize; 2];
        for i in 0..1000 {
            let user = format!("user_{}", i);
            let idx = VariantAssigner::assign(&user, "exp_dist", &w);
            counts[idx] += 1;
        }
        // With 50/50 split, each bucket should be roughly 500 +/- 100
        assert!(
            counts[0] > 400 && counts[0] < 600,
            "variant 0 got {} assignments",
            counts[0]
        );
        assert!(
            counts[1] > 400 && counts[1] < 600,
            "variant 1 got {} assignments",
            counts[1]
        );
    }

    #[test]
    fn test_three_variant_split() {
        let w = vec![0.33, 0.33, 0.34];
        let mut counts = [0usize; 3];
        for i in 0..3000 {
            let user = format!("u_{}", i);
            let idx = VariantAssigner::assign(&user, "exp_3way", &w);
            counts[idx] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c > 700 && c < 1300,
                "variant {} got {} assignments (expected ~1000)",
                i,
                c
            );
        }
    }

    #[test]
    fn test_weighted_split() {
        let w = vec![0.8, 0.2];
        let mut counts = [0usize; 2];
        for i in 0..1000 {
            let user = format!("w_{}", i);
            let idx = VariantAssigner::assign(&user, "exp_weighted", &w);
            counts[idx] += 1;
        }
        // ~80% should be in variant 0
        assert!(
            counts[0] > 700,
            "variant 0 got {} (expected >700)",
            counts[0]
        );
        assert!(
            counts[1] < 300,
            "variant 1 got {} (expected <300)",
            counts[1]
        );
    }

    // -- ExperimentManager tests ---------------------------------------------

    #[test]
    fn test_create_experiment() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("control", "baseline", 0.5),
            ExperimentVariant::new("treatment", "new prompt", 0.5),
        ];
        let id = mgr.create_experiment("prompt test", variants).unwrap();
        assert!(id.starts_with("exp_"));

        let exp = mgr.experiments.get(&id).unwrap();
        assert_eq!(exp.name, "prompt test");
        assert_eq!(exp.status, ExperimentStatus::Running);
        assert_eq!(exp.variants.len(), 2);
    }

    #[test]
    fn test_record_metric() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("test", variants).unwrap();
        mgr.record_metric(&id, 0, 1.0);
        mgr.record_metric(&id, 0, 2.0);
        mgr.record_metric(&id, 1, 3.0);

        let records = mgr.metrics.get(&id).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].value, 1.0);
        assert_eq!(records[2].variant_index, 1);
    }

    #[test]
    fn test_variant_stats() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("stats", variants).unwrap();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            mgr.record_metric(&id, 0, v);
        }
        // Also add some data for B so we can get results
        for v in [10.0, 20.0] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.get_results(&id, 0.95).unwrap();
        let a_stats = &result.variants[0];
        assert_eq!(a_stats.sample_size, 5);
        assert!((a_stats.mean - 3.0).abs() < 1e-9, "mean should be 3.0");
        // std_dev of [1,2,3,4,5] with sample variance = sqrt(10/4) = sqrt(2.5)
        let expected_sd = (2.5_f64).sqrt();
        assert!(
            (a_stats.std_dev - expected_sd).abs() < 1e-9,
            "std_dev should be {}, got {}",
            expected_sd,
            a_stats.std_dev
        );
        assert!((a_stats.min - 1.0).abs() < 1e-9);
        assert!((a_stats.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_results_significant() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("sig", variants).unwrap();

        // Group A: high values
        for v in [10.0, 11.0, 12.0, 10.0, 11.0] {
            mgr.record_metric(&id, 0, v);
        }
        // Group B: low values
        for v in [1.0, 2.0, 3.0, 1.0, 2.0] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.get_results(&id, 0.95).unwrap();
        assert!(result.is_significant, "should be significant");
        assert!(result.p_value < 0.05, "p_value {} should be < 0.05", result.p_value);
    }

    #[test]
    fn test_get_results_not_significant() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("nosig", variants).unwrap();

        // Very similar groups
        for v in [5.0, 5.1, 4.9, 5.0, 5.1] {
            mgr.record_metric(&id, 0, v);
        }
        for v in [5.0, 4.9, 5.1, 5.0, 4.9] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.get_results(&id, 0.95).unwrap();
        assert!(!result.is_significant, "should not be significant");
    }

    #[test]
    fn test_conversion_tracking() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("conv", variants).unwrap();

        // 10 exposures for variant 0, 3 conversions
        for _ in 0..7 {
            mgr.record_exposure(&id, 0);
        }
        for _ in 0..3 {
            mgr.record_conversion(&id, 0);
        }

        // Need metric data too so get_results doesn't fail with InsufficientData
        mgr.record_metric(&id, 0, 1.0);
        mgr.record_metric(&id, 0, 0.0);
        mgr.record_metric(&id, 1, 0.5);
        mgr.record_metric(&id, 1, 0.5);

        let result = mgr.get_results(&id, 0.95).unwrap();
        let a_stats = &result.variants[0];
        assert_eq!(a_stats.conversion_count, 3);
        // Total observations for variant 0 = 7 exposures + 3 conversions = 10
        assert!(
            (a_stats.conversion_rate - 0.3).abs() < 1e-9,
            "conversion_rate should be 0.3, got {}",
            a_stats.conversion_rate
        );
    }

    #[test]
    fn test_chi_square_basic() {
        // Group A: 80/100 conversions, Group B: 60/100
        let conversions = vec![(80, 100), (60, 100)];
        let (chi2, p) = SignificanceCalculator::chi_square_test(&conversions);
        assert!(chi2 > 0.0, "chi2 should be positive");
        // This difference should be significant
        assert!(p < 0.05, "p={} should be < 0.05", p);
    }

    #[test]
    fn test_stop_experiment() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("stop", variants).unwrap();

        mgr.stop_experiment(&id).unwrap();
        let exp = mgr.experiments.get(&id).unwrap();
        assert_eq!(exp.status, ExperimentStatus::Stopped);
        assert!(exp.stopped_at.is_some());

        // Second stop should fail
        let err = mgr.stop_experiment(&id).unwrap_err();
        assert_eq!(err, AbTestError::AlreadyStopped);
    }

    #[test]
    fn test_list_experiments() {
        let mut mgr = ExperimentManager::new();
        let v = || {
            vec![
                ExperimentVariant::new("A", "", 0.5),
                ExperimentVariant::new("B", "", 0.5),
            ]
        };
        mgr.create_experiment("exp1", v()).unwrap();
        mgr.create_experiment("exp2", v()).unwrap();
        mgr.create_experiment("exp3", v()).unwrap();

        let list = mgr.list_experiments();
        assert_eq!(list.len(), 3);

        let names: Vec<&str> = list.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"exp1"));
        assert!(names.contains(&"exp2"));
        assert!(names.contains(&"exp3"));
    }

    #[test]
    fn test_early_stopping_triggered() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("winner", "", 0.5),
            ExperimentVariant::new("loser", "", 0.5),
        ];
        let id = mgr.create_experiment("early", variants).unwrap();

        // Clear winner
        for v in [100.0, 101.0, 99.0, 100.0, 102.0] {
            mgr.record_metric(&id, 0, v);
        }
        for v in [1.0, 2.0, 1.0, 2.0, 1.0] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.check_early_stopping(&id, 0.05).unwrap();
        assert_eq!(result, Some("winner".to_string()));
    }

    #[test]
    fn test_early_stopping_not_triggered() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("no_early", variants).unwrap();

        // Very similar
        for v in [5.0, 5.1, 4.9, 5.0, 5.1] {
            mgr.record_metric(&id, 0, v);
        }
        for v in [5.0, 4.9, 5.1, 5.0, 4.9] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.check_early_stopping(&id, 0.05).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_invalid_experiment() {
        let mgr = ExperimentManager::new();
        let err = mgr.get_results("nonexistent", 0.95).unwrap_err();
        assert_eq!(err, AbTestError::ExperimentNotFound);
    }

    #[test]
    fn test_empty_metrics() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("A", "", 0.5),
            ExperimentVariant::new("B", "", 0.5),
        ];
        let id = mgr.create_experiment("empty", variants).unwrap();

        let err = mgr.get_results(&id, 0.95).unwrap_err();
        assert_eq!(err, AbTestError::InsufficientData);
    }

    #[test]
    fn test_experiment_lifecycle() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("control", "old prompt", 0.5),
            ExperimentVariant::new("treatment", "new prompt", 0.5),
        ];

        // Create
        let id = mgr.create_experiment("lifecycle", variants).unwrap();
        assert_eq!(
            mgr.experiments.get(&id).unwrap().status,
            ExperimentStatus::Running
        );

        // Assign
        let assignment = mgr.assign_user(&id, "user_123").unwrap();
        assert_eq!(assignment.experiment_id, id);
        assert!(assignment.variant_index < 2);

        // Record metrics
        for v in [8.0, 9.0, 7.0, 8.5, 8.0] {
            mgr.record_metric(&id, 0, v);
        }
        for v in [3.0, 4.0, 2.0, 3.5, 3.0] {
            mgr.record_metric(&id, 1, v);
        }

        // Get results
        let result = mgr.get_results(&id, 0.95).unwrap();
        assert!(result.is_significant);

        // Stop
        mgr.stop_experiment(&id).unwrap();
        assert_eq!(
            mgr.experiments.get(&id).unwrap().status,
            ExperimentStatus::Stopped
        );
    }

    #[test]
    fn test_single_variant_error() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![ExperimentVariant::new("only_one", "", 1.0)];
        let err = mgr.create_experiment("bad", variants).unwrap_err();
        assert_eq!(err, AbTestError::InvalidVariants);
    }

    #[test]
    fn test_recommended_variant() {
        let mut mgr = ExperimentManager::new();
        let variants = vec![
            ExperimentVariant::new("worse", "", 0.5),
            ExperimentVariant::new("better", "", 0.5),
        ];
        let id = mgr.create_experiment("rec", variants).unwrap();

        // Variant 1 ("better") has clearly higher values
        for v in [1.0, 2.0, 1.0, 2.0, 1.0] {
            mgr.record_metric(&id, 0, v);
        }
        for v in [100.0, 101.0, 99.0, 100.0, 102.0] {
            mgr.record_metric(&id, 1, v);
        }

        let result = mgr.get_results(&id, 0.95).unwrap();
        assert!(result.is_significant);
        assert_eq!(result.recommended_variant, Some("better".to_string()));
    }
}
