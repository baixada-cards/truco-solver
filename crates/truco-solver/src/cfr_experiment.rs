//! The mutable experiment file for autoresearch.
//!
//! This file contains the core CFR iteration logic that the LLM modifies.
//! Everything else (tree building, exploitability, info sets) is immutable
//! infrastructure in other files.
//!
//! Current algorithm: DCFR+ combining CFR+ clamping with DCFR discounting (α=1.5, γ=2).

use truco_engine::{Player, Score};

use crate::game_tree::{GameTree, NodeId, NodeView, PrebuiltTrees};
use crate::match_value::MatchValueTable;
use crate::strategy::StrategyTable;

/// Per-solve state. Add any auxiliary data structures here.
pub struct ExperimentState {
    pub alpha: f64,
    pub gamma: f64,
}

/// Called once before iterations begin.
pub fn init(_num_info_sets: usize) -> ExperimentState {
    ExperimentState {
        alpha: 1.5, // DCFR positive regret discount exponent
        gamma: 2.0, // quadratic weighting for average strategy
    }
}

/// Run one CFR iteration: traverse all trees for the given traversing player.
pub fn iterate(
    state: &mut ExperimentState,
    iteration: u64,
    traversing: Player,
    prebuilt: &PrebuiltTrees,
    table: &mut StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
) {
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if tree.is_empty() {
                continue;
            }
            traverse_tree(
                tree,
                0,
                traversing,
                dealer,
                [1.0, 1.0],
                entry.weight,
                iteration,
                state.gamma,
                score,
                match_values,
                &prebuilt.info_sets,
                table,
            );
        }
    }
}

/// Called after each iteration. Apply DCFR discounting to positive regrets only.
/// Negative regrets are already clamped to 0 inline (CFR+ style).
pub fn post_iterate(state: &mut ExperimentState, iteration: u64, table: &mut StrategyTable) {
    let t = iteration as f64;

    // DCFR+ discounting:
    // - Positive regrets: discount by t^α / (t^α + 1)  (DCFR α factor)
    // - Negative regrets: already clamped to 0 by CFR+ inline, so nothing to do
    // This combines the benefits of CFR+ (no negative regret accumulation)
    // with DCFR (downweighting early iterations).
    let t_alpha = t.powf(state.alpha);
    let pos_discount = t_alpha / (t_alpha + 1.0);

    for (_, data) in table.data.iter_mut() {
        for r in data.cumulative_regret.iter_mut() {
            // r is already >= 0 due to CFR+ clamping in traverse_tree.
            if *r > 0.0 {
                *r *= pos_discount;
            }
        }
        // cumulative_strategy is NOT discounted; weighting handled by t^gamma in iterate().
    }
}

// ─── Tree traversal (DCFR+ with quadratic strategy weighting) ────────────────────────────────────────
//
// Key invariant: traverse_tree returns the COUNTERFACTUAL value for the
// traversing player, weighted by:
//   - chance_weight (deal probability)
//   - reach_probs of the NON-traversing player (opponent reach)
//
// The traversing player's own reach is NOT included in the return value.
// This is the standard CFR formulation for counterfactual regret.
//
// CFR+: cumulative regrets are clamped to >= 0 after each update.
// Average strategy uses quadratic weighting: weight = t^gamma * traversing_reach.

#[allow(clippy::too_many_arguments)]
fn traverse_tree(
    tree: &GameTree,
    node_id: NodeId,
    traversing: Player,
    dealer: Player,
    reach_probs: [f64; 2],
    chance_weight: f64,
    iteration: u64,
    gamma: f64,
    score: &Score,
    match_values: &MatchValueTable,
    infos: &[(
        crate::info_set::InfoSetKey,
        crate::info_set::InfoSet,
        crate::game_tree::InfoActions,
    )],
    table: &mut StrategyTable,
) -> f64 {
    use truco_engine::MATCH_TARGET;

    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            let hand_value = payoff_p0.abs();
            let winner: Player = if payoff_p0 > 0.0 { 0 } else { 1 };

            let new_score = match winner {
                0 => Score {
                    zero: (score.zero + hand_value as u8).min(MATCH_TARGET),
                    one: score.one,
                },
                _ => Score {
                    zero: score.zero,
                    one: (score.one + hand_value as u8).min(MATCH_TARGET),
                },
            };

            let p0_match_value = if new_score.zero >= MATCH_TARGET {
                1.0
            } else if new_score.one >= MATCH_TARGET {
                -1.0
            } else {
                // Continuation hand is dealt by the other player (dealer alternates).
                2.0 * match_values.get(new_score.zero, new_score.one, 1 - dealer) - 1.0
            };

            // Return value from p0's perspective, weighted by:
            // chance_weight * opponent_reach
            let opponent_reach = reach_probs[1 - traversing as usize];
            let sign = if traversing == 0 { 1.0 } else { -1.0 };
            chance_weight * opponent_reach * sign * p0_match_value
        }

        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            let num_actions = edges.len();
            // The experiment keeps its historical key-based table access; the
            // node stores a dense index into `infos`, whose entry carries the key.
            let info_set_key = &infos[table_idx as usize].0;

            // Get current strategy via regret matching
            let strategy = {
                let data = table
                    .data
                    .get(info_set_key)
                    .expect("info_set_key pre-inserted during tree build");
                data.current_strategy()
            };

            if player == traversing {
                // Traversing player node: compute counterfactual action values
                let mut action_values = vec![0.0f64; num_actions];
                let mut node_value = 0.0f64;

                for (i, e) in edges.iter().enumerate() {
                    // Pass down reach probs with traversing player's strategy applied
                    let mut new_reach = reach_probs;
                    new_reach[player as usize] *= strategy[i];

                    action_values[i] = traverse_tree(
                        tree,
                        e.child,
                        traversing,
                        dealer,
                        new_reach,
                        chance_weight,
                        iteration,
                        gamma,
                        score,
                        match_values,
                        infos,
                        table,
                    );
                    node_value += strategy[i] * action_values[i];
                }

                // Update cumulative regrets with CFR+ clamping (clamp to >= 0)
                let data = table
                    .data
                    .get_mut(info_set_key)
                    .expect("info_set_key pre-inserted during tree build");
                for i in 0..num_actions {
                    let instant_regret = action_values[i] - node_value;
                    data.cumulative_regret[i] =
                        (data.cumulative_regret[i] + instant_regret).max(0.0);
                }

                // Update average strategy with quadratic weighting: weight = t^gamma * traversing_reach
                // This aggressively down-weights early poor strategies
                let traversing_reach = reach_probs[player as usize];
                let t = iteration as f64;
                let weight = t.powf(gamma) * traversing_reach * chance_weight;
                for i in 0..num_actions {
                    data.cumulative_strategy[i] += weight * strategy[i];
                }

                node_value
            } else {
                // Opponent node: just traverse with updated reach probs
                let mut node_value = 0.0f64;
                for (i, e) in edges.iter().enumerate() {
                    if strategy[i] > 1e-12 {
                        let mut new_reach = reach_probs;
                        new_reach[player as usize] *= strategy[i];

                        node_value += traverse_tree(
                            tree,
                            e.child,
                            traversing,
                            dealer,
                            new_reach,
                            chance_weight,
                            iteration,
                            gamma,
                            score,
                            match_values,
                            infos,
                            table,
                        );
                    }
                }
                node_value
            }
        }
    }
}
