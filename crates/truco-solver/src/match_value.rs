use serde::{Deserialize, Serialize};
use truco_engine::MATCH_TARGET;

const TABLE_DIM: usize = MATCH_TARGET as usize + 1;

/// Stores the equilibrium match-win probability for each (score, dealer) state.
/// `values[s0][s1][d]` = probability that player 0 wins the match given the hand
/// at score (s0, s1) is dealt by player `d` (0 or 1).
///
/// The dealer dimension is required because the engine's dealer strictly
/// ALTERNATES each hand, and the pé (dealer) advantage is large. A hand that
/// transitions to a non-terminal score continues to a new hand dealt by the
/// OTHER player, so continuation values must be looked up at `1 - dealer`.
///
/// Terminal states (score >= 12) are pre-filled and dealer-independent:
/// - (12, _) → 1.0 (player 0 already won)
/// - (_, 12) → 0.0 (player 1 already won)
///
/// Non-terminal states are filled incrementally as subgames are solved.
/// Unset states default to 0.5 (no information).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchValueTable {
    values: [[[f64; 2]; TABLE_DIM]; TABLE_DIM],
    /// True once `values[s0][s1][d]` was written by the solve pipeline (or is terminal).
    #[serde(default = "default_solved_grid")]
    solved: [[[bool; 2]; TABLE_DIM]; TABLE_DIM],
}

fn default_solved_grid() -> [[[bool; 2]; TABLE_DIM]; TABLE_DIM] {
    [[[false; 2]; TABLE_DIM]; TABLE_DIM]
}

impl MatchValueTable {
    /// Create a new table with terminal states filled in and all others at 0.5.
    pub fn new() -> Self {
        let n = TABLE_DIM;
        let mut values = [[[0.5f64; 2]; TABLE_DIM]; TABLE_DIM];
        let mut solved = [[[false; 2]; TABLE_DIM]; TABLE_DIM];

        // Terminal states (dealer-independent)
        for s in 0..n {
            for d in 0..2 {
                values[MATCH_TARGET as usize][s][d] = 1.0; // player 0 won
                values[s][MATCH_TARGET as usize][d] = 0.0; // player 1 won
                solved[MATCH_TARGET as usize][s][d] = true;
                solved[s][MATCH_TARGET as usize][d] = true;
            }
        }
        // Both at 12 shouldn't happen, but define it
        for d in 0..2 {
            values[MATCH_TARGET as usize][MATCH_TARGET as usize][d] = 0.5;
            solved[MATCH_TARGET as usize][MATCH_TARGET as usize][d] = true;
        }

        Self { values, solved }
    }

    /// Rebuild from legacy bincode that only stored a 2D, dealer-averaged `values`
    /// (no `solved` bitset and no dealer dimension). Each legacy cell is copied
    /// into both dealer slots; interior cells are treated as unsolved; terminal
    /// layout must match [`Self::new`].
    pub fn from_values_legacy(values_2d: [[f64; TABLE_DIM]; TABLE_DIM]) -> Self {
        let mut values = [[[0.5f64; 2]; TABLE_DIM]; TABLE_DIM];
        let mut solved = [[[false; 2]; TABLE_DIM]; TABLE_DIM];
        for i in 0..TABLE_DIM {
            for j in 0..TABLE_DIM {
                for d in 0..2 {
                    values[i][j][d] = values_2d[i][j];
                    if i == MATCH_TARGET as usize || j == MATCH_TARGET as usize {
                        solved[i][j][d] = true;
                    }
                }
            }
        }
        Self { values, solved }
    }

    /// Get player 0's match-win probability from score (s0, s1) when the hand is
    /// dealt by player `dealer`. Terminal scores are dealer-independent.
    pub fn get(&self, s0: u8, s1: u8, dealer: u8) -> f64 {
        if s0 >= MATCH_TARGET {
            return 1.0;
        }
        if s1 >= MATCH_TARGET {
            return 0.0;
        }
        self.values[s0 as usize][s1 as usize][dealer as usize]
    }

    /// Set player 0's match-win probability at score (s0, s1) for `dealer` and
    /// mark solved (in-play only).
    pub fn set(&mut self, s0: u8, s1: u8, dealer: u8, value: f64) {
        self.values[s0 as usize][s1 as usize][dealer as usize] = value;
        if (s0 as usize) < MATCH_TARGET as usize && (s1 as usize) < MATCH_TARGET as usize {
            self.solved[s0 as usize][s1 as usize][dealer as usize] = true;
        }
    }

    /// Check if a (score, dealer) state has been solved by the pipeline (or is
    /// terminal). Terminal scores are always considered solved.
    pub fn is_solved(&self, s0: u8, s1: u8, dealer: u8) -> bool {
        if s0 >= MATCH_TARGET || s1 >= MATCH_TARGET {
            return true;
        }
        self.solved[s0 as usize][s1 as usize][dealer as usize]
    }
}

impl Default for MatchValueTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine the order in which score states should be solved.
/// Returns (s0, s1) pairs from highest total score to lowest.
pub fn solve_order() -> Vec<(u8, u8)> {
    let max = MATCH_TARGET - 1; // 11
    let mut states = Vec::new();

    // Solve from highest combined score downward
    for total in (0..=(2 * max)).rev() {
        for s0 in 0..=max {
            let s1 = total.saturating_sub(s0);
            if s1 <= max && s0 + s1 == total {
                states.push((s0, s1));
            }
        }
    }

    states
}

/// Subset of [`solve_order`] where `s0 + s1 >= min_s0 + min_s1` (inclusive).
pub fn solve_order_from(min_s0: u8, min_s1: u8) -> Vec<(u8, u8)> {
    let threshold = min_s0 as u32 + min_s1 as u32;
    solve_order_between((MATCH_TARGET as u32 - 1) * 2, threshold)
}

/// Subset of [`solve_order`] where `max_total >= s0 + s1 >= min_total` (inclusive).
pub fn solve_order_between(max_total: u32, min_total: u32) -> Vec<(u8, u8)> {
    if max_total < min_total {
        return Vec::new();
    }
    solve_order()
        .into_iter()
        .filter(|&(s0, s1)| {
            let total = s0 as u32 + s1 as u32;
            total <= max_total && total >= min_total
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_values() {
        let mv = MatchValueTable::new();
        // Terminal scores are dealer-independent.
        for d in 0..2 {
            assert_eq!(mv.get(12, 0, d), 1.0);
            assert_eq!(mv.get(12, 11, d), 1.0);
            assert_eq!(mv.get(0, 12, d), 0.0);
            assert_eq!(mv.get(11, 12, d), 0.0);
        }
    }

    #[test]
    fn test_default_non_terminal() {
        let mv = MatchValueTable::new();
        for d in 0..2 {
            assert_eq!(mv.get(0, 0, d), 0.5);
            assert_eq!(mv.get(5, 7, d), 0.5);
            assert_eq!(mv.get(11, 11, d), 0.5);
            assert!(!mv.is_solved(0, 0, d));
            assert!(!mv.is_solved(11, 11, d));
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.48);
        assert!((mv.get(11, 11, 0) - 0.48).abs() < 1e-10);
        assert!(mv.is_solved(11, 11, 0));
    }

    #[test]
    fn test_per_dealer_values_are_independent() {
        let mut mv = MatchValueTable::new();
        mv.set(7, 5, 0, 0.7);
        mv.set(7, 5, 1, 0.3);
        assert!((mv.get(7, 5, 0) - 0.7).abs() < 1e-10);
        assert!((mv.get(7, 5, 1) - 0.3).abs() < 1e-10);
        assert!(mv.is_solved(7, 5, 0));
        assert!(mv.is_solved(7, 5, 1));
    }

    #[test]
    fn test_terminal_get_is_dealer_independent() {
        let mut mv = MatchValueTable::new();
        // Even after attempting to set a terminal cell, get() short-circuits to
        // the terminal value regardless of dealer.
        mv.set(12, 3, 0, 0.123);
        mv.set(12, 3, 1, 0.456);
        assert_eq!(mv.get(12, 3, 0), 1.0);
        assert_eq!(mv.get(12, 3, 1), 1.0);
        assert_eq!(mv.get(4, 12, 0), 0.0);
        assert_eq!(mv.get(4, 12, 1), 0.0);
    }

    #[test]
    fn test_solve_order_starts_with_highest() {
        let order = solve_order();
        assert_eq!(order[0], (11, 11));
    }

    #[test]
    fn test_solve_order_covers_all_states() {
        let order = solve_order();
        let max = MATCH_TARGET - 1;
        let expected = ((max + 1) * (max + 1)) as usize; // 12 * 12 = 144
        assert_eq!(order.len(), expected);

        // Every (s0, s1) with s0,s1 in 0..=11 should appear
        for s0 in 0..=max {
            for s1 in 0..=max {
                assert!(order.contains(&(s0, s1)), "missing ({}, {})", s0, s1);
            }
        }
    }

    #[test]
    fn test_solve_order_descending_total() {
        let order = solve_order();
        for i in 1..order.len() {
            let total_prev = order[i - 1].0 + order[i - 1].1;
            let total_curr = order[i].0 + order[i].1;
            assert!(
                total_curr <= total_prev,
                "order not descending at index {}: ({},{}) after ({},{})",
                i,
                order[i].0,
                order[i].1,
                order[i - 1].0,
                order[i - 1].1,
            );
        }
    }

    #[test]
    fn test_solve_order_from_10x10() {
        let o = solve_order_from(10, 10);
        assert!(o.contains(&(11, 11)));
        assert!(o.contains(&(10, 10)));
        // Sum threshold 20: (9,11) and (11,9) qualify; (9,10) does not.
        assert!(o.contains(&(9, 11)));
        assert!(o.contains(&(11, 9)));
        assert!(!o.contains(&(9, 10)));
        assert!(!o.contains(&(10, 9)));
    }

    #[test]
    fn test_solve_order_between_22_and_20() {
        let o = solve_order_between(22, 20);
        assert_eq!(o[0], (11, 11));
        assert!(o.contains(&(11, 10)));
        assert!(o.contains(&(10, 11)));
        assert!(o.contains(&(10, 10)));
        assert!(o.contains(&(9, 11)));
        assert!(o.contains(&(11, 9)));
        assert!(!o.contains(&(9, 10)));
        assert!(!o.contains(&(8, 11)));
    }
}
