//! `Unit` (the atom of evolution) + `Population` + `LineageDag`.
//!
//! A `Unit` carries both a `task_prompt` and a `mutation_prompt`; the
//! latter evolves alongside the former under self-referential operators.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::{MutationOperator, ProviderFingerprint};
use super::fitness::FitnessScore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unit {
    pub id: String,
    pub task_prompt: String,
    pub mutation_prompt: String,
    pub generation_born: u32,
    pub parents: Vec<String>,
    pub operator_born: Option<MutationOperator>,
    pub fitness: Option<FitnessScore>,
    pub evaluations: u32,
    pub fingerprint: ProviderFingerprint,
}

impl Unit {
    pub fn seed(
        id: impl Into<String>,
        task_prompt: impl Into<String>,
        mutation_prompt: impl Into<String>,
        fingerprint: ProviderFingerprint,
    ) -> Self {
        Self {
            id: id.into(),
            task_prompt: task_prompt.into(),
            mutation_prompt: mutation_prompt.into(),
            generation_born: 0,
            parents: Vec::new(),
            operator_born: None,
            fitness: None,
            evaluations: 0,
            fingerprint,
        }
    }

    pub fn child(
        id: impl Into<String>,
        task_prompt: impl Into<String>,
        mutation_prompt: impl Into<String>,
        parents: Vec<String>,
        operator: MutationOperator,
        generation: u32,
        fingerprint: ProviderFingerprint,
    ) -> Self {
        Self {
            id: id.into(),
            task_prompt: task_prompt.into(),
            mutation_prompt: mutation_prompt.into(),
            generation_born: generation,
            parents,
            operator_born: Some(operator),
            fitness: None,
            evaluations: 0,
            fingerprint,
        }
    }

    pub fn fitness_value(&self) -> f64 {
        self.fitness.as_ref().map(|f| f.aggregate).unwrap_or(0.0)
    }

    pub fn is_evaluated(&self) -> bool {
        self.fitness.is_some()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Population {
    units: Vec<Unit>,
}

impl Population {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_units(units: Vec<Unit>) -> Self {
        Self { units }
    }

    pub fn push(&mut self, u: Unit) {
        self.units.push(u);
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Unit> {
        self.units.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Unit> {
        self.units.iter_mut()
    }

    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    pub fn units_mut(&mut self) -> &mut [Unit] {
        &mut self.units
    }

    pub fn into_units(self) -> Vec<Unit> {
        self.units
    }

    pub fn get(&self, id: &str) -> Option<&Unit> {
        self.units.iter().find(|u| u.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Unit> {
        self.units.iter_mut().find(|u| u.id == id)
    }

    pub fn clear(&mut self) {
        self.units.clear();
    }

    pub fn retain<F: FnMut(&Unit) -> bool>(&mut self, f: F) {
        self.units.retain(f);
    }

    /// Highest-fitness evaluated unit.
    pub fn best(&self) -> Option<&Unit> {
        self.units
            .iter()
            .filter(|u| u.is_evaluated())
            .max_by(|a, b| {
                a.fitness_value()
                    .partial_cmp(&b.fitness_value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Mean aggregate fitness across evaluated units, or 0 if none.
    pub fn mean_fitness(&self) -> f64 {
        let vals: Vec<f64> = self
            .iter()
            .filter(|u| u.is_evaluated())
            .map(|u| u.fitness_value())
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    }
}

// =============================================================================
// Lineage DAG
// =============================================================================

/// Directed acyclic graph of parent/child relationships. Populated as the
/// breeder creates each new unit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageDag {
    /// `child_id → parent_ids`.
    pub parents: HashMap<String, Vec<String>>,
    /// `parent_id → child_ids`.
    pub children: HashMap<String, Vec<String>>,
}

impl LineageDag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_seed(&mut self, id: &str) {
        self.parents.entry(id.to_string()).or_default();
    }

    pub fn insert_child(&mut self, child_id: &str, parent_ids: &[String]) {
        self.parents
            .insert(child_id.to_string(), parent_ids.to_vec());
        for p in parent_ids {
            self.children
                .entry(p.clone())
                .or_default()
                .push(child_id.to_string());
        }
    }

    /// Walk ancestors breadth-first. Stops at seeds (no parents).
    pub fn ancestors(&self, id: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut out = Vec::new();
        queue.push_back(id.to_string());
        while let Some(cur) = queue.pop_front() {
            if let Some(parents) = self.parents.get(&cur) {
                for p in parents {
                    if seen.insert(p.clone()) {
                        out.push(p.clone());
                        queue.push_back(p.clone());
                    }
                }
            }
        }
        out
    }

    /// Walk descendants breadth-first.
    pub fn descendants(&self, id: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut out = Vec::new();
        queue.push_back(id.to_string());
        while let Some(cur) = queue.pop_front() {
            if let Some(kids) = self.children.get(&cur) {
                for k in kids {
                    if seen.insert(k.clone()) {
                        out.push(k.clone());
                        queue.push_back(k.clone());
                    }
                }
            }
        }
        out
    }

    pub fn lineage_depth(&self, id: &str) -> u32 {
        let mut max_depth = 0u32;
        let parents = self.parents.get(id).cloned().unwrap_or_default();
        for p in parents {
            let d = 1 + self.lineage_depth(&p);
            if d > max_depth {
                max_depth = d;
            }
        }
        max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp() -> ProviderFingerprint {
        ProviderFingerprint::new("test", "mock")
    }

    #[test]
    fn seed_unit_has_no_parents() {
        let u = Unit::seed("u1", "task", "mut", fp());
        assert!(u.parents.is_empty());
        assert!(!u.is_evaluated());
        assert_eq!(u.fitness_value(), 0.0);
    }

    #[test]
    fn population_best_requires_evaluated_unit() {
        let mut pop = Population::new();
        let mut u = Unit::seed("u1", "task", "mut", fp());
        let mut score = FitnessScore::new(fp());
        score.aggregate = 0.7;
        u.fitness = Some(score);
        pop.push(u);
        pop.push(Unit::seed("u2", "task", "mut", fp()));
        assert_eq!(pop.best().unwrap().id, "u1");
    }

    #[test]
    fn lineage_walks_ancestors() {
        let mut dag = LineageDag::new();
        dag.insert_seed("a");
        dag.insert_child("b", &["a".into()]);
        dag.insert_child("c", &["b".into()]);
        let anc = dag.ancestors("c");
        assert!(anc.contains(&"a".to_string()));
        assert!(anc.contains(&"b".to_string()));
    }

    #[test]
    fn lineage_depth_root_is_zero() {
        let mut dag = LineageDag::new();
        dag.insert_seed("a");
        dag.insert_child("b", &["a".into()]);
        assert_eq!(dag.lineage_depth("a"), 0);
        assert_eq!(dag.lineage_depth("b"), 1);
    }
}
