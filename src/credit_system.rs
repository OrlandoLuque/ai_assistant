// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Generic credit economy for a distributed compute network.
//!
//! Provides [`CreditManager`] for per-node balance tracking (earn / spend /
//! escrow), [`EpochManager`] for periodic checkpointing, and [`NetworkPool`]
//! for the shared fee pool.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CreditManager
// ---------------------------------------------------------------------------

/// Per-node credit balance with escrow and pending-maturity support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditManager {
    balance: f64,
    pending: HashMap<String, PendingCredit>,
    escrow: HashMap<String, f64>,
    stake_locked: f64,
    node_id: String,
}

/// A credit amount that is maturing and not yet spendable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingCredit {
    pub amount: f64,
    pub created_at: u64,
    pub maturity_secs: u64,
    pub receipt_id: String,
}

impl CreditManager {
    /// Create a new manager for `node_id` with zero balance.
    pub fn new(node_id: &str) -> Self {
        Self {
            balance: 0.0,
            pending: HashMap::new(),
            escrow: HashMap::new(),
            stake_locked: 0.0,
            node_id: node_id.to_string(),
        }
    }

    /// Current confirmed balance (does **not** include pending or escrow).
    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// Effective balance: confirmed minus stake, plus any matured pending.
    /// Does **not** mutate state — call [`mature_pending`] first to actualise.
    pub fn effective_balance(&self) -> f64 {
        self.balance - self.stake_locked
    }

    /// Record an earning.  If `maturity_secs > 0` the amount goes to pending;
    /// otherwise it is immediately available.
    pub fn earn(&mut self, amount: f64, receipt_id: &str, maturity_secs: u64) {
        if maturity_secs == 0 {
            self.balance += amount;
        } else {
            self.pending.insert(
                receipt_id.to_string(),
                PendingCredit {
                    amount,
                    created_at: Self::now_secs(),
                    maturity_secs,
                    receipt_id: receipt_id.to_string(),
                },
            );
        }
    }

    /// Spend `amount` from the confirmed balance.
    pub fn spend(&mut self, amount: f64) -> Result<(), String> {
        if self.effective_balance() < amount {
            return Err(format!(
                "insufficient balance: effective {:.4} < {:.4}",
                self.effective_balance(),
                amount
            ));
        }
        self.balance -= amount;
        Ok(())
    }

    /// Lock `amount` in escrow for `request_id`.
    pub fn escrow_lock(&mut self, amount: f64, request_id: &str) -> Result<(), String> {
        if self.effective_balance() < amount {
            return Err("insufficient balance for escrow".to_string());
        }
        self.balance -= amount;
        self.escrow.insert(request_id.to_string(), amount);
        Ok(())
    }

    /// Release a previously locked escrow.
    ///
    /// If `to_provider` is `true` the funds are considered paid out (removed).
    /// Otherwise they are returned to the caller's balance.
    pub fn escrow_release(&mut self, request_id: &str, to_provider: bool) {
        if let Some(amount) = self.escrow.remove(request_id) {
            if !to_provider {
                self.balance += amount;
            }
        }
    }

    /// Move any matured pending credits to the confirmed balance.
    pub fn mature_pending(&mut self) {
        let now = Self::now_secs();
        let matured: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, p)| now.saturating_sub(p.created_at) >= p.maturity_secs)
            .map(|(k, _)| k.clone())
            .collect();

        for key in matured {
            if let Some(p) = self.pending.remove(&key) {
                self.balance += p.amount;
            }
        }
    }

    /// Progressive fee rate based on balance tier.
    ///
    /// | Balance        | Fee   |
    /// |----------------|-------|
    /// | < 100          | 5 %   |
    /// | 100 .. 1 000   | 3 %   |
    /// | 1 000 .. 10 000| 1 %   |
    /// | >= 10 000      | 0.5 % |
    pub fn earning_fee_rate(&self) -> f64 {
        if self.balance < 100.0 {
            0.05
        } else if self.balance < 1_000.0 {
            0.03
        } else if self.balance < 10_000.0 {
            0.01
        } else {
            0.005
        }
    }

    /// Return the node id.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    // Monotonic seconds since UNIX epoch (or 0 if unavailable).
    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// EpochManager
// ---------------------------------------------------------------------------

/// Periodic checkpoint of the global balance state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochCheckpoint {
    pub epoch_id: u64,
    pub balances: HashMap<String, f64>,
    pub merkle_root: Vec<u8>,
    pub witness_signatures: Vec<(String, Vec<u8>)>,
    pub created_at: u64,
}

/// Manages epoch boundaries and checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochManager {
    pub current_epoch: u64,
    pub epoch_duration_secs: u64,
    pub checkpoints: Vec<EpochCheckpoint>,
    last_epoch_time: u64,
}

impl EpochManager {
    /// Create a new epoch manager with the given epoch duration.
    pub fn new(duration_secs: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            current_epoch: 0,
            epoch_duration_secs: duration_secs,
            checkpoints: Vec::new(),
            last_epoch_time: now,
        }
    }

    /// Returns `true` if enough time has elapsed to close the current epoch.
    pub fn should_close_epoch(&self, now: u64) -> bool {
        now.saturating_sub(self.last_epoch_time) >= self.epoch_duration_secs
    }

    /// Close the current epoch and produce a checkpoint.
    pub fn close_epoch(&mut self, balances: HashMap<String, f64>) -> EpochCheckpoint {
        self.current_epoch += 1;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_epoch_time = now;

        // Simple merkle root: hash of concatenated sorted balance entries.
        let mut sorted: Vec<_> = balances.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let data: String = sorted
            .iter()
            .map(|(k, v)| format!("{}:{:.8}", k, v))
            .collect::<Vec<_>>()
            .join("|");
        let merkle_root = simple_hash(data.as_bytes());

        let cp = EpochCheckpoint {
            epoch_id: self.current_epoch,
            balances,
            merkle_root: merkle_root.to_vec(),
            witness_signatures: Vec::new(),
            created_at: now,
        };
        self.checkpoints.push(cp.clone());
        cp
    }
}

// ---------------------------------------------------------------------------
// NetworkPool
// ---------------------------------------------------------------------------

/// Shared fee pool for the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPool {
    pub balance: f64,
    pub daily_inflow: f64,
    pub daily_outflow: f64,
}

impl NetworkPool {
    pub fn new() -> Self {
        Self {
            balance: 0.0,
            daily_inflow: 0.0,
            daily_outflow: 0.0,
        }
    }

    /// Add a fee to the pool.
    pub fn add_fee(&mut self, amount: f64) {
        self.balance += amount;
        self.daily_inflow += amount;
    }

    /// Disburse a grant from the pool.
    pub fn disburse_grant(&mut self, amount: f64) -> Result<(), String> {
        if self.balance < amount {
            return Err("pool balance insufficient".to_string());
        }
        self.balance -= amount;
        self.daily_outflow += amount;
        Ok(())
    }

    /// Ratio of inflow to outflow.  Values > 1.0 indicate a healthy pool.
    pub fn health_ratio(&self) -> f64 {
        if self.daily_outflow <= 0.0 {
            if self.daily_inflow > 0.0 {
                return f64::INFINITY;
            }
            return 1.0;
        }
        self.daily_inflow / self.daily_outflow
    }
}

impl Default for NetworkPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal hash helper (FNV-1a → 8 bytes)
// ---------------------------------------------------------------------------

fn simple_hash(data: &[u8]) -> [u8; 8] {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h.to_be_bytes()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_earn_and_spend() {
        let mut cm = CreditManager::new("node-a");
        cm.earn(100.0, "r1", 0);
        assert!((cm.balance() - 100.0).abs() < f64::EPSILON);
        assert!(cm.spend(40.0).is_ok());
        assert!((cm.balance() - 60.0).abs() < f64::EPSILON);
        assert!(cm.spend(200.0).is_err());
    }

    #[test]
    fn test_escrow_lock_release() {
        let mut cm = CreditManager::new("node-b");
        cm.earn(50.0, "r1", 0);
        assert!(cm.escrow_lock(30.0, "req1").is_ok());
        assert!((cm.balance() - 20.0).abs() < f64::EPSILON);

        // Release back to caller.
        cm.escrow_release("req1", false);
        assert!((cm.balance() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_escrow_to_provider() {
        let mut cm = CreditManager::new("node-c");
        cm.earn(50.0, "r1", 0);
        assert!(cm.escrow_lock(30.0, "req1").is_ok());
        cm.escrow_release("req1", true);
        // Funds paid out — not returned.
        assert!((cm.balance() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_maturity() {
        let mut cm = CreditManager::new("node-d");
        // Earn with 0-second maturity → immediate.
        cm.earn(10.0, "r1", 0);
        assert!((cm.balance() - 10.0).abs() < f64::EPSILON);

        // Earn with very short maturity.  We'll call mature_pending immediately.
        cm.earn(20.0, "r2", 0); // 0 maturity = instant
        cm.mature_pending();
        assert!((cm.balance() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progressive_fee() {
        let mut cm = CreditManager::new("n");
        assert!((cm.earning_fee_rate() - 0.05).abs() < f64::EPSILON);
        cm.earn(500.0, "r1", 0);
        assert!((cm.earning_fee_rate() - 0.03).abs() < f64::EPSILON);
        cm.earn(5000.0, "r2", 0);
        assert!((cm.earning_fee_rate() - 0.01).abs() < f64::EPSILON);
        cm.earn(5000.0, "r3", 0);
        assert!((cm.earning_fee_rate() - 0.005).abs() < f64::EPSILON);
    }

    #[test]
    fn test_epoch_close() {
        let mut em = EpochManager::new(60);
        let mut balances = HashMap::new();
        balances.insert("a".to_string(), 100.0);
        balances.insert("b".to_string(), 200.0);

        let cp = em.close_epoch(balances);
        assert_eq!(cp.epoch_id, 1);
        assert!(!cp.merkle_root.is_empty());
        assert_eq!(em.checkpoints.len(), 1);
    }

    #[test]
    fn test_epoch_should_close() {
        let em = EpochManager::new(60);
        let past = em.last_epoch_time;
        assert!(!em.should_close_epoch(past + 30));
        assert!(em.should_close_epoch(past + 60));
        assert!(em.should_close_epoch(past + 120));
    }

    #[test]
    fn test_pool_health() {
        let mut pool = NetworkPool::new();
        pool.add_fee(100.0);
        assert!(pool.health_ratio().is_infinite());
        assert!(pool.disburse_grant(40.0).is_ok());
        assert!((pool.health_ratio() - 2.5).abs() < f64::EPSILON);
        assert!(pool.disburse_grant(200.0).is_err());
    }
}
