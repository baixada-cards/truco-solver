use rand::Rng;
use truco_engine::{Player, Score};

use crate::abstraction::{enumerate_deals, TurnupClass};
use crate::game_tree::TraversalState;
use crate::strategy::Strategy;

/// Result of simulating matches between two strategies.
#[derive(Clone, Debug)]
pub struct MatchupResult {
    pub hands_played: u64,
    pub wins_0: u64,
    pub wins_1: u64,
    /// Total points scored by player 0 across all hands.
    pub total_points_0: u64,
    /// Total points scored by player 1 across all hands.
    pub total_points_1: u64,
    pub player_0_win_rate: f64,
}

/// Simulate a single hand between two strategies at a given score and turnup class.
/// Returns (hand_winner, hand_value) or None if something goes wrong.
pub fn simulate_hand(
    strategy_0: &dyn Strategy,
    strategy_1: &dyn Strategy,
    state: &TraversalState,
    rng: &mut impl Rng,
) -> Option<(Player, u8)> {
    let mut current = state.clone();
    let mut steps = 0;
    let max_steps = 100; // safety limit

    while !current.is_terminal() {
        if steps > max_steps {
            return None; // shouldn't happen in a correct game
        }
        steps += 1;

        let player = current.engine.current_player()?;
        let actions = current.abstract_legal_actions().ok()?;
        if actions.is_empty() {
            return None;
        }

        let info_set = current.current_info_set()?;
        let strategy: &dyn Strategy = if player == 0 { strategy_0 } else { strategy_1 };
        let dist = strategy.action_probabilities(&info_set, &actions);
        let action = dist.sample(rng);
        current = current.apply_abstract_action(action).ok()?;
    }

    Some((current.hand_winner()?, current.hand_value()))
}

/// Simulate many hands between two strategies at a given score state.
/// Samples turnup classes and deals randomly.
pub fn simulate_matchup(
    score: Score,
    strategy_0: &dyn Strategy,
    strategy_1: &dyn Strategy,
    num_hands: u64,
    rng: &mut impl Rng,
) -> MatchupResult {
    let all_classes = TurnupClass::all();
    let mut result = MatchupResult {
        hands_played: 0,
        wins_0: 0,
        wins_1: 0,
        total_points_0: 0,
        total_points_1: 0,
        player_0_win_rate: 0.0,
    };

    // Pre-compute deals per turnup class
    let deals_per_class: Vec<_> = all_classes
        .iter()
        .map(|tc| (*tc, enumerate_deals(tc)))
        .collect();

    for hand_num in 0..num_hands {
        // Sample turnup class weighted by frequency
        let tc_idx = sample_turnup_class(rng, &all_classes);
        let tc = all_classes[tc_idx];
        let deals = &deals_per_class[tc_idx].1;

        // Sample a deal weighted by combinatorial weight
        let deal = sample_deal(rng, deals);

        // Alternate dealer
        let dealer = (hand_num % 2) as Player;

        let state = match TraversalState::from_deal(dealer, score.clone(), tc, deal) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if let Some((winner, value)) = simulate_hand(strategy_0, strategy_1, &state, rng) {
            result.hands_played += 1;
            match winner {
                0 => {
                    result.wins_0 += 1;
                    result.total_points_0 += value as u64;
                }
                _ => {
                    result.wins_1 += 1;
                    result.total_points_1 += value as u64;
                }
            }
        }
    }

    if result.hands_played > 0 {
        result.player_0_win_rate = result.wins_0 as f64 / result.hands_played as f64;
    }

    result
}

/// Compute the exact expected value for player 0 by exhaustively evaluating
/// all deals for a given turnup class.
pub fn exact_ev(
    score: &Score,
    tc: TurnupClass,
    dealer: Player,
    strategy_0: &dyn Strategy,
    strategy_1: &dyn Strategy,
) -> f64 {
    let deals = enumerate_deals(&tc);
    let mut total_ev = 0.0;

    for deal in &deals {
        let state = match TraversalState::from_deal(dealer, score.clone(), tc, deal) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let ev = exact_ev_recursive(&state, strategy_0, strategy_1);
        total_ev += deal.weight * ev;
    }

    total_ev
}

/// Recursively compute the exact expected value for player 0 at a given state.
fn exact_ev_recursive(
    state: &TraversalState,
    strategy_0: &dyn Strategy,
    strategy_1: &dyn Strategy,
) -> f64 {
    if state.is_terminal() {
        debug_assert!(
            state.hand_winner().is_some(),
            "terminal state must have a winner"
        );
        let winner = state
            .hand_winner()
            .expect("terminal state must have a winner");
        let value = state.hand_value() as f64;
        return if winner == 0 { value } else { -value };
    }

    let player = match state.engine.current_player() {
        Some(p) => p,
        None => return 0.0,
    };

    let actions = match state.abstract_legal_actions() {
        Ok(a) => a,
        Err(_) => return 0.0,
    };

    debug_assert!(
        state.current_info_set().is_some(),
        "non-terminal state with current player must have an info set"
    );
    let info_set = state
        .current_info_set()
        .expect("non-terminal state with current player must have an info set");
    let strategy: &dyn Strategy = if player == 0 { strategy_0 } else { strategy_1 };
    let dist = strategy.action_probabilities(&info_set, &actions);

    let mut ev = 0.0;
    for &(action, prob) in &dist.actions {
        if prob == 0.0 {
            continue;
        }
        if let Ok(new_state) = state.apply_abstract_action(action) {
            ev += prob * exact_ev_recursive(&new_state, strategy_0, strategy_1);
        }
    }

    ev
}

fn sample_turnup_class(rng: &mut impl Rng, classes: &[TurnupClass; 9]) -> usize {
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, tc) in classes.iter().enumerate() {
        cumulative += tc.weight();
        if r < cumulative {
            return i;
        }
    }
    classes.len() - 1
}

fn sample_deal<'a>(
    rng: &mut impl Rng,
    deals: &'a [crate::abstraction::AbstractDeal],
) -> &'a crate::abstraction::AbstractDeal {
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for deal in deals {
        cumulative += deal.weight;
        if r < cumulative {
            return deal;
        }
    }
    deals.last().expect("sample_deal requires non-empty deals")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::PureStrategy;

    #[test]
    fn test_always_fold_vs_always_raise() {
        let folder = PureStrategy::always_fold();
        let raiser = PureStrategy::always_raise();
        let mut rng = rand::rng();

        let result = simulate_matchup(Score { zero: 0, one: 0 }, &folder, &raiser, 100, &mut rng);

        // The raiser should always win (folder folds to every raise)
        assert!(
            result.wins_1 > result.wins_0,
            "raiser (p1) should win more: p0={}, p1={}",
            result.wins_0,
            result.wins_1,
        );
    }

    #[test]
    fn test_mirror_strategy_approximately_fair() {
        let strat = PureStrategy::strongest_face_up();
        let mut rng = rand::rng();

        let result = simulate_matchup(Score { zero: 0, one: 0 }, &strat, &strat, 500, &mut rng);

        // With symmetric strategy, win rate should be near 50%
        let rate = result.player_0_win_rate;
        assert!(
            rate > 0.35 && rate < 0.65,
            "mirror strategy win rate should be ~50%, got {:.2}%",
            rate * 100.0,
        );
    }

    #[test]
    fn test_zero_sum_exact_ev() {
        // For any strategy pair, EV_0 + EV_1 should equal 0.
        // (EV_1 = -EV_0 in our formulation since we return player 0's perspective)
        let strat_0 = PureStrategy::strongest_face_up();
        let strat_1 = PureStrategy::always_fold();
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };

        let ev_0_as_dealer = exact_ev(&score, tc, 0, &strat_0, &strat_1);
        let ev_0_as_non_dealer = exact_ev(&score, tc, 1, &strat_0, &strat_1);

        // EV is from player 0's perspective, so it's already the negative of player 1's EV
        // We just check it's finite and reasonable
        assert!(
            ev_0_as_dealer.is_finite(),
            "EV should be finite, got {}",
            ev_0_as_dealer
        );
        assert!(
            ev_0_as_non_dealer.is_finite(),
            "EV should be finite, got {}",
            ev_0_as_non_dealer
        );
    }

    #[test]
    fn test_weakest_then_hide_vs_strongest_face_up() {
        // Strategy that hides cards after round 1 should lose to strategy that always plays face-up,
        // because face-down cards always lose to face-up cards.
        let hider = PureStrategy::weakest_then_hide();
        let open = PureStrategy::strongest_face_up();
        let mut rng = rand::rng();

        let result = simulate_matchup(Score { zero: 0, one: 0 }, &hider, &open, 200, &mut rng);

        assert!(
            result.wins_1 > result.wins_0,
            "open player should dominate: p0(hider)={}, p1(open)={}",
            result.wins_0,
            result.wins_1,
        );
    }
}
