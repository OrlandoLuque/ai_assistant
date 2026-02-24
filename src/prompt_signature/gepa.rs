//! GEPA — Genetic Pareto Optimizer (NSGA-II inspired multi-objective optimization).

use crate::error::AiError;

use super::types::{
    CompiledPrompt, EvalMetric, EvaluationBudget, PromptExample, Signature, TrainingExample,
};

// ============================================================================
// 1.1 GEPA — Genetic Pareto Optimizer (NSGA-II inspired)
// ============================================================================

/// Configuration for the GEPA multi-objective genetic optimizer.
#[derive(Debug, Clone)]
pub struct GEPAConfig {
    /// Population size per generation
    pub population_size: usize,
    /// Number of generations to evolve
    pub generations: usize,
    /// Probability of mutating an individual (0.0..1.0)
    pub mutation_rate: f64,
    /// Probability of crossing over two parents (0.0..1.0)
    pub crossover_rate: f64,
    /// Number of elite individuals carried unchanged to the next generation
    pub elitism_count: usize,
    /// Tournament selection pool size
    pub tournament_size: usize,
}

impl Default for GEPAConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            generations: 10,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            elitism_count: 2,
            tournament_size: 3,
        }
    }
}

/// A single solution on the Pareto front.
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// The compiled prompt for this solution
    pub compiled: CompiledPrompt,
    /// Scores on each objective (higher is better)
    pub scores: Vec<f64>,
    /// Non-domination rank (0 = first front)
    pub rank: usize,
    /// Crowding distance for diversity preservation
    pub crowding_distance: f64,
}

/// A collection of Pareto-optimal solutions.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    /// All solutions with assigned ranks and crowding distances
    pub solutions: Vec<ParetoSolution>,
}

impl ParetoFront {
    /// Returns true if solution `b` dominates solution `a`.
    ///
    /// Domination: all scores of `b` >= `a`, and at least one strictly >.
    pub fn is_dominated(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() || a.is_empty() {
            return false;
        }
        let mut at_least_one_strictly_better = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if *bi < *ai {
                return false;
            }
            if *bi > *ai {
                at_least_one_strictly_better = true;
            }
        }
        at_least_one_strictly_better
    }

    /// Assign non-domination ranks (NSGA-II fast non-dominated sort) and
    /// compute crowding distances for each front.
    pub fn compute(solutions: &mut [ParetoSolution]) {
        let n = solutions.len();
        if n == 0 {
            return;
        }

        // domination_count[i] = number of solutions that dominate i
        let mut domination_count: Vec<usize> = vec![0; n];
        // dominated_set[i] = indices that i dominates
        let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if Self::is_dominated(&solutions[j].scores, &solutions[i].scores) {
                    // i dominates j
                    dominated_set[i].push(j);
                } else if Self::is_dominated(&solutions[i].scores, &solutions[j].scores) {
                    domination_count[i] += 1;
                }
            }
        }

        // Assign ranks front by front
        let mut current_front: Vec<usize> = Vec::new();
        for i in 0..n {
            if domination_count[i] == 0 {
                solutions[i].rank = 0;
                current_front.push(i);
            }
        }

        let mut rank = 0;
        while !current_front.is_empty() {
            let mut next_front: Vec<usize> = Vec::new();
            for &i in &current_front {
                for &j in &dominated_set[i] {
                    domination_count[j] = domination_count[j].saturating_sub(1);
                    if domination_count[j] == 0 {
                        solutions[j].rank = rank + 1;
                        next_front.push(j);
                    }
                }
            }
            rank += 1;
            current_front = next_front;
        }

        // Compute crowding distance per front
        let max_rank = solutions.iter().map(|s| s.rank).max().unwrap_or(0);
        let num_objectives = solutions.first().map(|s| s.scores.len()).unwrap_or(0);

        for r in 0..=max_rank {
            let indices: Vec<usize> = (0..n).filter(|&i| solutions[i].rank == r).collect();
            if indices.len() <= 2 {
                for &i in &indices {
                    solutions[i].crowding_distance = f64::INFINITY;
                }
                continue;
            }
            for &i in &indices {
                solutions[i].crowding_distance = 0.0;
            }
            for m in 0..num_objectives {
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_by(|&a, &b| {
                    solutions[a].scores[m]
                        .partial_cmp(&solutions[b].scores[m])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                // Boundary points get infinite distance
                let first = sorted_indices[0];
                let last = sorted_indices[sorted_indices.len() - 1];
                solutions[first].crowding_distance = f64::INFINITY;
                solutions[last].crowding_distance = f64::INFINITY;

                let f_max = solutions[last].scores[m];
                let f_min = solutions[first].scores[m];
                let range = f_max - f_min;
                if range < 1e-12 {
                    continue;
                }
                for k in 1..(sorted_indices.len() - 1) {
                    let prev = sorted_indices[k - 1];
                    let next = sorted_indices[k + 1];
                    let idx = sorted_indices[k];
                    if solutions[idx].crowding_distance.is_finite() {
                        solutions[idx].crowding_distance +=
                            (solutions[next].scores[m] - solutions[prev].scores[m]) / range;
                    }
                }
            }
        }
    }

    /// Get references to all solutions at the given rank.
    pub fn get_front(&self, rank: usize) -> Vec<&ParetoSolution> {
        self.solutions.iter().filter(|s| s.rank == rank).collect()
    }
}

/// GEPA multi-objective genetic optimizer.
///
/// Uses NSGA-II style non-dominated sorting, crowding distance, and
/// tournament selection to evolve a population of compiled prompts
/// across multiple objectives simultaneously.
pub struct GEPAOptimizer {
    config: GEPAConfig,
}

impl GEPAOptimizer {
    /// Create a new GEPA optimizer with the given configuration.
    pub fn new(config: GEPAConfig) -> Self {
        Self { config }
    }

    /// Initialize a population of compiled prompts from a signature and examples.
    pub(crate) fn initialize_population(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
    ) -> Vec<CompiledPrompt> {
        let mut population = Vec::with_capacity(self.config.population_size);

        let instruction_pool = [
            "",
            "Be concise and precise.",
            "Think step by step before answering.",
            "Provide a detailed and thorough response.",
            "Focus on accuracy above all else.",
            "Use clear and simple language.",
            "Consider multiple perspectives.",
            "Cite specific evidence when possible.",
            "Start with the most important information.",
            "Be exhaustive in your coverage.",
            "Prioritize clarity over completeness.",
        ];

        for i in 0..self.config.population_size {
            // Vary instruction
            let instr = instruction_pool[i % instruction_pool.len()];
            let sig_variant = if instr.is_empty() {
                signature.clone()
            } else {
                signature.clone().with_instructions(instr)
            };

            // Vary number of demos
            let max_demos = examples.len().min(5);
            let num_demos = if max_demos > 0 { (i % max_demos) + 1 } else { 0 };
            let demos: Vec<PromptExample> = examples
                .iter()
                .take(num_demos)
                .map(|ex| PromptExample {
                    inputs: ex.inputs.clone(),
                    outputs: ex.expected_outputs.clone(),
                })
                .collect();

            population.push(sig_variant.compile_with_examples(&demos));
        }

        population
    }

    /// Evaluate all solutions against all metrics.
    fn evaluate_population(
        signature: &Signature,
        solutions: &mut [ParetoSolution],
        examples: &[TrainingExample],
        metrics: &[&dyn EvalMetric],
    ) {
        let num_metrics = metrics.len();
        for sol in solutions.iter_mut() {
            sol.scores = vec![0.0; num_metrics];
            if examples.is_empty() {
                continue;
            }
            for (m_idx, metric) in metrics.iter().enumerate() {
                let mut total = 0.0;
                let mut count = 0usize;
                for ex in examples {
                    for output_field in &signature.outputs {
                        if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                            let rendered = sol.compiled.build_full_prompt(&ex.inputs);
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
                sol.scores[m_idx] = if count > 0 { total / count as f64 } else { 0.0 };
            }
        }
    }

    /// Tournament selection: pick the best individual from a random subset.
    pub(crate) fn select_parent<'a>(
        &self,
        solutions: &'a [ParetoSolution],
        seed: usize,
    ) -> &'a ParetoSolution {
        let n = solutions.len();
        let mut best_idx = seed % n;
        for t in 1..self.config.tournament_size {
            let candidate_idx = (seed.wrapping_mul(31).wrapping_add(t * 17)) % n;
            let candidate = &solutions[candidate_idx];
            let best = &solutions[best_idx];
            // Prefer lower rank; on tie, prefer higher crowding distance
            if candidate.rank < best.rank
                || (candidate.rank == best.rank
                    && candidate.crowding_distance > best.crowding_distance)
            {
                best_idx = candidate_idx;
            }
        }
        &solutions[best_idx]
    }

    /// Crossover: combine demos from two parents.
    pub(crate) fn crossover(parent_a: &CompiledPrompt, parent_b: &CompiledPrompt, seed: usize) -> CompiledPrompt {
        let mut child_examples = Vec::new();
        // Interleave examples from both parents
        let max_len = parent_a.examples.len().max(parent_b.examples.len());
        for i in 0..max_len {
            if (seed.wrapping_add(i)) % 2 == 0 {
                if i < parent_a.examples.len() {
                    child_examples.push(parent_a.examples[i].clone());
                } else if i < parent_b.examples.len() {
                    child_examples.push(parent_b.examples[i].clone());
                }
            } else if i < parent_b.examples.len() {
                child_examples.push(parent_b.examples[i].clone());
            } else if i < parent_a.examples.len() {
                child_examples.push(parent_a.examples[i].clone());
            }
        }

        // Choose system prompt from parent with longer one (more instructions)
        let system_prompt = if parent_a.system_prompt.len() >= parent_b.system_prompt.len() {
            parent_a.system_prompt.clone()
        } else {
            parent_b.system_prompt.clone()
        };

        CompiledPrompt {
            system_prompt,
            user_template: parent_a.user_template.clone(),
            examples: child_examples,
        }
    }

    /// Mutate a compiled prompt: swap/remove/add demos, perturb instruction text.
    pub(crate) fn mutate(compiled: &CompiledPrompt, seed: usize) -> CompiledPrompt {
        let mut result = compiled.clone();

        let mutation_type = seed % 4;
        match mutation_type {
            0 => {
                // Swap two examples if possible
                if result.examples.len() >= 2 {
                    let i = seed % result.examples.len();
                    let j = (seed / 3 + 1) % result.examples.len();
                    if i != j {
                        result.examples.swap(i, j);
                    }
                }
            }
            1 => {
                // Remove last example if any
                if !result.examples.is_empty() {
                    let idx = seed % result.examples.len();
                    result.examples.remove(idx);
                }
            }
            2 => {
                // Duplicate an example (add demo)
                if !result.examples.is_empty() {
                    let idx = seed % result.examples.len();
                    let dup = result.examples[idx].clone();
                    result.examples.push(dup);
                }
            }
            _ => {
                // Perturb instruction text
                let suffixes = [
                    " Be precise.",
                    " Think carefully.",
                    " Focus on accuracy.",
                    " Be thorough.",
                ];
                let suffix = suffixes[seed % suffixes.len()];
                result.system_prompt.push_str(suffix);
            }
        }

        result
    }

    /// Run the full multi-objective optimization.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        metrics: &[&dyn EvalMetric],
        budget: &mut EvaluationBudget,
    ) -> Result<ParetoFront, AiError> {
        if metrics.is_empty() {
            return Err(AiError::other("GEPA requires at least one metric"));
        }

        // Initialize population
        let initial_prompts = self.initialize_population(signature, examples);
        let mut solutions: Vec<ParetoSolution> = initial_prompts
            .into_iter()
            .map(|compiled| ParetoSolution {
                compiled,
                scores: Vec::new(),
                rank: 0,
                crowding_distance: 0.0,
            })
            .collect();

        // Evaluate initial population
        if !budget.try_use() {
            return Err(AiError::other("GEPA budget exhausted before first evaluation"));
        }
        Self::evaluate_population(signature, &mut solutions, examples, metrics);
        ParetoFront::compute(&mut solutions);

        // Evolve for remaining generations
        for gen in 0..self.config.generations {
            if !budget.try_use() {
                break;
            }

            let mut next_gen: Vec<ParetoSolution> = Vec::with_capacity(self.config.population_size);

            // Elitism: carry over the best individuals
            let mut elite_indices: Vec<usize> = (0..solutions.len()).collect();
            elite_indices.sort_by(|&a, &b| {
                solutions[a]
                    .rank
                    .cmp(&solutions[b].rank)
                    .then(
                        solutions[b]
                            .crowding_distance
                            .partial_cmp(&solutions[a].crowding_distance)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            });
            for &idx in elite_indices.iter().take(self.config.elitism_count.min(solutions.len())) {
                next_gen.push(solutions[idx].clone());
            }

            // Fill rest via selection + crossover + mutation
            let mut child_seed = gen * 1000;
            while next_gen.len() < self.config.population_size {
                child_seed += 1;
                let parent_a = self.select_parent(&solutions, child_seed);
                let parent_b = self.select_parent(&solutions, child_seed.wrapping_mul(7));

                let child_compiled =
                    if (child_seed as f64 / 1000.0).fract() < self.config.crossover_rate {
                        Self::crossover(&parent_a.compiled, &parent_b.compiled, child_seed)
                    } else {
                        parent_a.compiled.clone()
                    };

                let child_compiled =
                    if (child_seed as f64 / 997.0).fract() < self.config.mutation_rate {
                        Self::mutate(&child_compiled, child_seed)
                    } else {
                        child_compiled
                    };

                next_gen.push(ParetoSolution {
                    compiled: child_compiled,
                    scores: Vec::new(),
                    rank: 0,
                    crowding_distance: 0.0,
                });
            }

            // Evaluate and sort new generation
            Self::evaluate_population(signature, &mut next_gen, examples, metrics);
            ParetoFront::compute(&mut next_gen);

            solutions = next_gen;
        }

        Ok(ParetoFront { solutions })
    }
}
