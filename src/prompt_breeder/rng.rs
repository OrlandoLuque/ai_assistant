//! Deterministic xorshift\* PRNG. Same pattern as
//! `prompt_synthesis::exploration` — we avoid pulling `rand` for the
//! `prompt-breeder` feature, so a run can be reproduced from nothing more
//! than its seed.

use serde::{Deserialize, Serialize};

/// 64-bit xorshift\* as described in Marsaglia's original paper. Small state,
/// good enough for GA-style tie-breaking. Never use for cryptography.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreederRng {
    state: u64,
}

impl BreederRng {
    /// Construct from a seed. Zero is remapped to a fixed non-zero so the
    /// generator never stalls on the all-zero trap.
    pub fn from_seed(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    /// Draw a fresh 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, n)`. Returns 0 when `n == 0`.
    pub fn gen_range_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }

    /// Uniform in `[0.0, 1.0)`.
    pub fn gen_unit(&mut self) -> f64 {
        // Take the upper 53 bits so the result is in [0, 1) without bias.
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    /// Fisher–Yates shuffle in place.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.gen_range_usize(i + 1);
            v.swap(i, j);
        }
    }

    /// Pick a random element, or `None` if the slice is empty.
    pub fn choose<'a, T>(&mut self, v: &'a [T]) -> Option<&'a T> {
        if v.is_empty() {
            None
        } else {
            let i = self.gen_range_usize(v.len());
            Some(&v[i])
        }
    }

    /// Weighted choice. Returns the index of the chosen element.
    pub fn weighted_choice(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
        if total <= 0.0 {
            return if weights.is_empty() {
                None
            } else {
                Some(self.gen_range_usize(weights.len()))
            };
        }
        let mut acc = 0.0;
        let target = self.gen_unit() * total;
        for (i, w) in weights.iter().enumerate() {
            if w.is_finite() && *w > 0.0 {
                acc += w;
                if target <= acc {
                    return Some(i);
                }
            }
        }
        Some(weights.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = BreederRng::from_seed(42);
        let mut b = BreederRng::from_seed(42);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn zero_seed_does_not_stall() {
        let mut r = BreederRng::from_seed(0);
        let a = r.next_u64();
        let b = r.next_u64();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn gen_range_is_bounded() {
        let mut r = BreederRng::from_seed(1);
        for _ in 0..1000 {
            let v = r.gen_range_usize(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn gen_unit_in_range() {
        let mut r = BreederRng::from_seed(7);
        for _ in 0..1000 {
            let v = r.gen_unit();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn weighted_choice_respects_weights() {
        let mut r = BreederRng::from_seed(123);
        let mut count_a = 0;
        let mut count_b = 0;
        for _ in 0..2000 {
            match r.weighted_choice(&[0.9, 0.1]).unwrap() {
                0 => count_a += 1,
                1 => count_b += 1,
                _ => unreachable!(),
            }
        }
        assert!(count_a > count_b * 3);
    }
}
