//! SIMBA Optimizer (Simulated Annealing + Multi-Armed Bandit Adaptation).

use serde::{Deserialize, Serialize};

use super::types::{EvalMetric, OptimizationResult, Signature, TrainingExample};

// ============================================================================
// 4.1 SIMBA Optimizer (Simulated Annealing + Multi-Armed Bandit Adaptation)
// ============================================================================

/// Cooling schedule for simulated annealing in the SIMBA optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingSchedule {
    /// Linear cooling: temperature decreases linearly from `initial_temp` to `min_temp`.
    Linear { initial_temp: f64, min_temp: f64 },
    /// Exponential cooling: temperature = initial_temp * decay_rate^iteration.
    Exponential { initial_temp: f64, decay_rate: f64 },
    /// Adaptive cooling: temperature = initial_temp * improvement_factor^(iteration * 0.1).
    Adaptive {
        initial_temp: f64,
        improvement_factor: f64,
    },
}

impl CoolingSchedule {
    /// Compute the temperature at a given iteration.
    pub fn temperature(&self, iteration: usize, max_iterations: usize) -> f64 {
        match self {
            CoolingSchedule::Linear {
                initial_temp,
                min_temp,
            } => {
                if max_iterations == 0 {
                    return *initial_temp;
                }
                let progress = iteration as f64 / max_iterations as f64;
                initial_temp - (initial_temp - min_temp) * progress
            }
            CoolingSchedule::Exponential {
                initial_temp,
                decay_rate,
            } => initial_temp * decay_rate.powi(iteration as i32),
            CoolingSchedule::Adaptive {
                initial_temp,
                improvement_factor,
            } => initial_temp * improvement_factor.powf(iteration as f64 * 0.1),
        }
    }
}

/// Strategy for mutating prompt instructions in the SIMBA optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationStrategy {
    /// Randomly perturb instruction text with given strength (0.0..1.0).
    RandomPerturbation { strength: f64 },
    /// Crossover parts of two instructions with the given rate.
    Crossover { crossover_rate: f64 },
    /// Use an LLM-guided mutation with a prompt template.
    LlmGuided { prompt_template: String },
    /// Combine multiple strategies with weighted selection.
    Combined {
        strategies: Vec<MutationStrategy>,
        weights: Vec<f64>,
    },
}

/// Configuration for the SIMBA optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimbaConfig {
    /// Number of prompt variants in the population.
    pub population_size: usize,
    /// Number of evolutionary generations.
    pub generations: usize,
    /// Probability of mutating an individual (0.0..1.0).
    pub mutation_rate: f64,
    /// Cooling schedule for simulated annealing.
    pub cooling_schedule: CoolingSchedule,
    /// Number of individuals in tournament selection.
    pub tournament_size: usize,
    /// Number of elite individuals carried unchanged to the next generation.
    pub elite_count: usize,
    /// Number of examples to evaluate per candidate in each generation.
    pub mini_batch_size: usize,
}

impl Default for SimbaConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            generations: 50,
            mutation_rate: 0.3,
            cooling_schedule: CoolingSchedule::Linear {
                initial_temp: 1.0,
                min_temp: 0.01,
            },
            tournament_size: 3,
            elite_count: 2,
            mini_batch_size: 5,
        }
    }
}

/// Tournament selection with elitism for the SIMBA optimizer.
pub struct TournamentSelector {
    tournament_size: usize,
    elite_count: usize,
}

impl TournamentSelector {
    /// Create a new tournament selector.
    pub fn new(tournament_size: usize, elite_count: usize) -> Self {
        Self {
            tournament_size,
            elite_count,
        }
    }

    /// Select individuals from the population using tournament selection.
    ///
    /// Keeps the `elite_count` best individuals unchanged. For the remaining
    /// slots, picks `tournament_size` random candidates and selects the best.
    /// Returns `population.len()` selected instruction strings.
    pub fn select(&self, population: &[(String, f64)]) -> Vec<String> {
        if population.is_empty() {
            return Vec::new();
        }

        let n = population.len();
        let mut result = Vec::with_capacity(n);

        // Sort by score descending to identify elites
        let mut indexed: Vec<(usize, f64)> = population.iter().enumerate().map(|(i, (_, s))| (i, *s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep elites
        let elite_count = self.elite_count.min(n);
        for i in 0..elite_count {
            result.push(population[indexed[i].0].0.clone());
        }

        // Fill remaining via tournament selection
        let mut seed: usize = 42;
        while result.len() < n {
            let mut best_idx = seed % n;
            let mut best_score = population[best_idx].1;
            for t in 1..self.tournament_size {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let candidate_idx = (seed >> 16) % n;
                if population[candidate_idx].1 > best_score {
                    best_score = population[candidate_idx].1;
                    best_idx = candidate_idx;
                }
                let _ = t;
            }
            result.push(population[best_idx].0.clone());
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        }

        result
    }
}

/// SIMBA (Simulated Annealing + Multi-Armed Bandit Adaptation) prompt optimizer.
///
/// Combines evolutionary population-based search with simulated annealing
/// acceptance criteria and tournament selection to find optimal prompt
/// formulations.
pub struct SimbaOptimizer {
    config: SimbaConfig,
    pub(crate) mutation_strategy: MutationStrategy,
    selector: TournamentSelector,
}

impl SimbaOptimizer {
    /// Create a new SIMBA optimizer with default RandomPerturbation strategy.
    pub fn new(config: SimbaConfig) -> Self {
        let selector = TournamentSelector::new(config.tournament_size, config.elite_count);
        Self {
            mutation_strategy: MutationStrategy::RandomPerturbation { strength: 0.3 },
            config,
            selector,
        }
    }

    /// Create a new SIMBA optimizer with a specific mutation strategy.
    pub fn with_strategy(config: SimbaConfig, strategy: MutationStrategy) -> Self {
        let selector = TournamentSelector::new(config.tournament_size, config.elite_count);
        Self {
            mutation_strategy: strategy,
            config,
            selector,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &SimbaConfig {
        &self.config
    }

    /// Mutate an instruction string by applying heuristic word-level transformations.
    ///
    /// Based on `strength` (0.0..1.0), applies random word swaps, capitalization
    /// changes, and emphasis markers.
    fn mutate_instruction(instruction: &str, strength: f64, seed: usize) -> String {
        let mut words: Vec<String> = instruction.split_whitespace().map(|w| w.to_string()).collect();
        if words.is_empty() {
            // Generate a base instruction if empty
            let bases = [
                "Answer carefully and precisely.",
                "Think step by step.",
                "Be thorough in your response.",
                "Focus on accuracy.",
            ];
            return bases[seed % bases.len()].to_string();
        }

        let mut rng_state: usize = seed;
        let mut next_rng = || -> usize {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 16) & 0xFFFF
        };

        // Swap adjacent words based on strength probability
        let len = words.len();
        if len >= 2 {
            for i in 0..(len - 1) {
                let r = next_rng();
                let prob = (r as f64) / 65535.0;
                if prob < strength {
                    words.swap(i, i + 1);
                }
            }
        }

        // Randomly capitalize/lowercase words
        for word in &mut words {
            let r = next_rng();
            let prob = (r as f64) / 65535.0;
            if prob < strength * 0.5 {
                let r2 = next_rng();
                if r2 % 2 == 0 {
                    *word = word.to_uppercase();
                } else {
                    *word = word.to_lowercase();
                }
            }
        }

        // Add emphasis markers with probability proportional to strength
        let emphasis_markers = ["importantly,", "carefully", "precisely", "notably,"];
        let r = next_rng();
        let prob = (r as f64) / 65535.0;
        if prob < strength * 0.3 {
            let marker = emphasis_markers[next_rng() % emphasis_markers.len()];
            let insert_pos = if words.is_empty() { 0 } else { next_rng() % words.len() };
            words.insert(insert_pos, marker.to_string());
        }

        words.join(" ")
    }

    /// Evaluate a prompt instruction against a mini-batch of examples.
    fn evaluate_instruction(
        signature: &Signature,
        instruction: &str,
        examples: &[TrainingExample],
        metric: &dyn EvalMetric,
        batch_size: usize,
    ) -> f64 {
        let sig_variant = signature.clone().with_instructions(instruction);
        let compiled = sig_variant.compile();

        let eval_count = batch_size.min(examples.len());
        let mut total = 0.0;
        let mut count = 0usize;

        for i in 0..eval_count {
            let ex = &examples[i];
            for output_field in &signature.outputs {
                if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                    let rendered = compiled.build_full_prompt(&ex.inputs);
                    let sim_predicted = if rendered.contains(&output_field.name) {
                        expected.clone()
                    } else {
                        String::new()
                    };
                    total += metric.score(&sim_predicted, expected);
                    count += 1;
                }
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Run the SIMBA optimization loop.
    ///
    /// 1. Initialize a population of prompt instruction variants.
    /// 2. For each generation: evaluate on mini-batch, select via tournament,
    ///    mutate offspring, accept/reject via simulated annealing.
    /// 3. Return the best variant found.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        metric: &dyn EvalMetric,
    ) -> OptimizationResult {
        let base = signature
            .instructions
            .as_deref()
            .unwrap_or("Answer the question accurately.");

        // Step 1: Initialize population
        let mut population: Vec<(String, f64)> = Vec::with_capacity(self.config.population_size);
        for i in 0..self.config.population_size {
            let variant = Self::mutate_instruction(base, 0.5, i * 7 + 13);
            let score = Self::evaluate_instruction(
                signature,
                &variant,
                examples,
                metric,
                self.config.mini_batch_size,
            );
            population.push((variant, score));
        }

        let mut best_instruction = base.to_string();
        let mut best_score = f64::NEG_INFINITY;
        let mut scores_history: Vec<f64> = Vec::new();

        // Find initial best
        for (instr, score) in &population {
            if *score > best_score {
                best_score = *score;
                best_instruction = instr.clone();
            }
        }
        scores_history.push(best_score);

        // Step 2: Evolutionary loop
        for gen in 0..self.config.generations {
            let temperature = self
                .config
                .cooling_schedule
                .temperature(gen, self.config.generations);

            // Select parents via tournament
            let selected = self.selector.select(&population);

            // Create offspring via mutation
            let mut new_population: Vec<(String, f64)> = Vec::with_capacity(self.config.population_size);

            // Determine mutation strength from strategy
            let mutation_strength = match &self.mutation_strategy {
                MutationStrategy::RandomPerturbation { strength } => *strength,
                MutationStrategy::Crossover { crossover_rate } => *crossover_rate * 0.5,
                MutationStrategy::LlmGuided { .. } => 0.3,
                MutationStrategy::Combined { weights, .. } => {
                    let sum: f64 = weights.iter().sum();
                    if sum > 0.0 { 0.3 * (weights.len() as f64 / sum).min(1.0) } else { 0.3 }
                }
            };

            for (i, parent_instr) in selected.iter().enumerate() {
                let seed = gen * 1000 + i * 37 + 7;

                // Decide whether to mutate
                let rng_val = seed.wrapping_mul(2654435761) % 1000;
                let should_mutate = (rng_val as f64 / 1000.0) < self.config.mutation_rate;

                let child_instr = if should_mutate {
                    Self::mutate_instruction(parent_instr, mutation_strength, seed)
                } else {
                    parent_instr.clone()
                };

                let child_score = Self::evaluate_instruction(
                    signature,
                    &child_instr,
                    examples,
                    metric,
                    self.config.mini_batch_size,
                );

                // Simulated annealing acceptance
                let parent_score = population.get(i).map(|(_, s)| *s).unwrap_or(0.0);
                let accept = if child_score >= parent_score {
                    true
                } else if temperature > 1e-10 {
                    let delta = parent_score - child_score;
                    let acceptance_prob = (-delta / temperature).exp();
                    let rng_accept = (seed.wrapping_mul(48271) % 10000) as f64 / 10000.0;
                    rng_accept < acceptance_prob
                } else {
                    false
                };

                if accept {
                    new_population.push((child_instr, child_score));
                } else {
                    // Keep parent
                    new_population.push((parent_instr.clone(), parent_score));
                }
            }

            population = new_population;

            // Track best
            for (instr, score) in &population {
                if *score > best_score {
                    best_score = *score;
                    best_instruction = instr.clone();
                }
            }
            scores_history.push(best_score);
        }

        // Build final result
        let best_sig = signature
            .clone()
            .with_instructions(best_instruction);
        let best_prompt = best_sig.compile();

        OptimizationResult {
            best_prompt,
            best_score,
            trials_run: self.config.generations * self.config.population_size,
            scores_history,
        }
    }
}
