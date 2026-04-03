// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Collusion detection for peer-to-peer compute networks.
//!
//! Analyses transaction history to identify pairs or clusters of nodes that
//! exhibit suspicious patterns (high reciprocity, low partner diversity,
//! or isolated trading clusters).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// CollusionDetector
// ---------------------------------------------------------------------------

/// Analyses transaction history to surface suspicious node pairs or clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionDetector {
    /// Recorded transactions: `(from, to, timestamp)`.
    transactions: Vec<(String, String, u64)>,
    /// Maximum number of transactions to keep in the rolling window.
    max_history: usize,
}

/// Result of a collusion analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionReport {
    /// Pairs with a suspicion score above the threshold.
    pub suspicious_pairs: Vec<(String, String, f64)>,
    /// Isolated clusters of nodes that only trade among themselves.
    pub isolated_clusters: Vec<Vec<String>>,
}

impl CollusionDetector {
    /// Create a new detector that retains at most `max_history` transactions.
    pub fn new(max_history: usize) -> Self {
        Self {
            transactions: Vec::new(),
            max_history,
        }
    }

    /// Record a transaction from `from` to `to`.
    pub fn record_transaction(&mut self, from: &str, to: &str) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.transactions
            .push((from.to_string(), to.to_string(), ts));
        if self.transactions.len() > self.max_history {
            self.transactions.remove(0);
        }
    }

    /// Run the full analysis and return a [`CollusionReport`].
    pub fn analyze(&self) -> CollusionReport {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut node_partners: HashMap<String, HashSet<String>> = HashMap::new();

        for (from, to, _) in &self.transactions {
            let key = Self::ordered_pair(from, to);
            *pair_counts.entry(key).or_insert(0) += 1;
            node_partners
                .entry(from.clone())
                .or_default()
                .insert(to.clone());
            node_partners
                .entry(to.clone())
                .or_default()
                .insert(from.clone());
        }

        let total = self.transactions.len().max(1) as f64;

        // Suspicious pairs: high reciprocity + low partner diversity.
        let mut suspicious_pairs = Vec::new();
        for ((a, b), count) in &pair_counts {
            let freq = *count as f64 / total;
            let diversity_a = node_partners.get(a).map(|s| s.len()).unwrap_or(1) as f64;
            let diversity_b = node_partners.get(b).map(|s| s.len()).unwrap_or(1) as f64;
            let avg_diversity = (diversity_a + diversity_b) / 2.0;

            // Reciprocity component.
            let reciprocity = self.reciprocity_ratio(a, b);

            // Score: high frequency + high reciprocity + low diversity → suspicious.
            let score = freq * 0.3 + reciprocity * 0.4 + (1.0 / avg_diversity) * 0.3;
            if score > 0.3 {
                suspicious_pairs.push((a.clone(), b.clone(), score));
            }
        }
        suspicious_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Isolated clusters: connected components where all members trade only
        // with each other.
        let isolated_clusters = self.find_isolated_clusters(&node_partners);

        CollusionReport {
            suspicious_pairs,
            isolated_clusters,
        }
    }

    /// Quick check: is the pair `(node_a, node_b)` suspicious?
    pub fn is_suspicious(&self, node_a: &str, node_b: &str) -> bool {
        let report = self.analyze();
        let key = Self::ordered_pair(node_a, node_b);
        report
            .suspicious_pairs
            .iter()
            .any(|(a, b, _)| (a.as_str(), b.as_str()) == (key.0.as_str(), key.1.as_str()))
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Fraction of transactions between (a, b) that go in both directions.
    fn reciprocity_ratio(&self, a: &str, b: &str) -> f64 {
        let ab = self
            .transactions
            .iter()
            .filter(|(f, t, _)| f == a && t == b)
            .count();
        let ba = self
            .transactions
            .iter()
            .filter(|(f, t, _)| f == b && t == a)
            .count();
        let total = ab + ba;
        if total == 0 {
            return 0.0;
        }
        let min = ab.min(ba) as f64;
        let max = ab.max(ba) as f64;
        if max == 0.0 {
            0.0
        } else {
            min / max
        }
    }

    /// Find connected components where every member's partners are within
    /// the same component (isolated clusters).
    fn find_isolated_clusters(
        &self,
        partners: &HashMap<String, HashSet<String>>,
    ) -> Vec<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut clusters: Vec<Vec<String>> = Vec::new();

        for node in partners.keys() {
            if visited.contains(node) {
                continue;
            }
            // BFS to find connected component.
            let mut component: Vec<String> = Vec::new();
            let mut queue: Vec<String> = vec![node.clone()];
            while let Some(n) = queue.pop() {
                if !visited.insert(n.clone()) {
                    continue;
                }
                component.push(n.clone());
                if let Some(neighbours) = partners.get(&n) {
                    for nb in neighbours {
                        if !visited.contains(nb) {
                            queue.push(nb.clone());
                        }
                    }
                }
            }

            // Check if the component is isolated: every member's partners
            // are a subset of the component.
            let comp_set: HashSet<&str> = component.iter().map(|s| s.as_str()).collect();
            let is_isolated = component.iter().all(|n| {
                partners
                    .get(n)
                    .map(|ps| ps.iter().all(|p| comp_set.contains(p.as_str())))
                    .unwrap_or(true)
            });
            // Only flag clusters of 2+ nodes that are truly isolated.
            if is_isolated && component.len() >= 2 {
                component.sort();
                clusters.push(component);
            }
        }

        clusters
    }

    /// Canonical ordering so (a,b) == (b,a).
    fn ordered_pair(a: &str, b: &str) -> (String, String) {
        if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_pattern_ok() {
        let mut det = CollusionDetector::new(1000);
        // Many different pairs → no single pair should be suspicious.
        for i in 0..20 {
            let from = format!("node-{}", i);
            let to = format!("node-{}", (i + 1) % 20);
            det.record_transaction(&from, &to);
        }
        let report = det.analyze();
        // With 20 different pairs and even distribution, suspicion should be low.
        assert!(
            report.suspicious_pairs.is_empty() || report.suspicious_pairs[0].2 < 0.5,
            "pairs: {:?}",
            report.suspicious_pairs
        );
    }

    #[test]
    fn test_suspicious_pair() {
        let mut det = CollusionDetector::new(1000);
        // Two nodes trading exclusively with each other — very suspicious.
        for _ in 0..50 {
            det.record_transaction("alice", "bob");
            det.record_transaction("bob", "alice");
        }
        assert!(det.is_suspicious("alice", "bob"));
    }

    #[test]
    fn test_isolated_cluster() {
        let mut det = CollusionDetector::new(1000);
        // Cluster {x, y, z} only trades among itself.
        for _ in 0..10 {
            det.record_transaction("x", "y");
            det.record_transaction("y", "z");
            det.record_transaction("z", "x");
        }
        let report = det.analyze();
        assert!(
            !report.isolated_clusters.is_empty(),
            "clusters: {:?}",
            report.isolated_clusters
        );
        let cluster = &report.isolated_clusters[0];
        assert!(cluster.contains(&"x".to_string()));
        assert!(cluster.contains(&"y".to_string()));
        assert!(cluster.contains(&"z".to_string()));
    }

    #[test]
    fn test_diverse_transactions_not_suspicious() {
        let mut det = CollusionDetector::new(1000);
        // Alice trades with many different partners.
        for i in 0..30 {
            det.record_transaction("alice", &format!("partner-{}", i));
        }
        // Alice should not be suspicious with any single partner.
        assert!(!det.is_suspicious("alice", "partner-0"));
    }
}
