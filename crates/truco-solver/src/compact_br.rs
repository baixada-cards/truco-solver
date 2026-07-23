//! Space-for-time exact best response over dynamic `TraversalState` DFS.
//!
//! The production CFR oracle keeps a full packed tree arena and node-indexed
//! scratch arrays. This prototype recomputes histories once per decision depth
//! instead: much more CPU, but memory is proportional to compact policy rows,
//! resolved BR choices, and one depth's aggregates rather than every node.

use ahash::AHashMap;
use truco_engine::{Player, Score};

use crate::abstraction::{AbstractDeal, TurnupClass};
use crate::cfr::terminal_p0_value;
use crate::game_tree::{
    MissingPolicyFallback, PolicyLookup, PolicyValueSource, TraversalState, TreeRules,
};
use crate::info_set::{AbstractAction, InfoSetKey};
use crate::match_value::MatchValueTable;
use crate::strategy::ActionProbs;

#[derive(Clone, Debug)]
pub struct CompactBestResponseResult {
    pub total: f64,
    pub chosen_info_sets: usize,
    pub max_depth: usize,
    pub dfs_visits: u64,
}

#[derive(Clone, Debug)]
pub struct CompactProfileResult {
    /// Player-0 value, using the same sum-over-included-dealers convention as
    /// the exact BR core. With a dealer filter this is the one dealer's value.
    pub p0_value: f64,
    /// Probability mass by signed hand payoff, indexed as `payoff + 12`.
    /// When both dealers are included this sums to two, matching the value.
    pub outcome_probabilities: [f64; 25],
    pub missing_decisions: u64,
    pub dfs_visits: u64,
    pub projection_stats: ProjectionStats,
}

/// How probability mass that the source policy saved on an action absent from
/// the current tree is projected. Absent actions arise when a checkpoint was
/// solved before a proof-scoped prune (or projected across bands), so the
/// choice decides which strategy the evaluation is actually about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DominatedProjection {
    /// Renormalize the surviving probabilities proportionally. This spreads
    /// the pruned action's mass over ALL survivors, producing a genuinely
    /// different (usually worse) strategy than the saved one.
    Renormalize,
    /// Move mass from a pruned `PlayFaceDown(c)` onto the same card's
    /// `PlayFaceUp(c)` and from a certificate-pruned `AcceptRaise` onto
    /// `Fold` — the action each dominance proof names as the (weak)
    /// dominator — then renormalize any remaining unmappable mass. The
    /// result weakly dominates the saved strategy, so a fresh certificate
    /// of the projected policy is an honest bound for what would ship.
    Remap,
}

/// Per-row projection diagnostics, aggregated per traversal visit.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowProjection {
    pub missing: bool,
    /// Saved mass moved onto its proof-named dominator at this row.
    pub remapped_mass: f64,
    /// Saved mass on absent actions that had no dominator to receive it and
    /// was renormalized away (assumes normalized source rows).
    pub dropped_mass: f64,
}

/// Aggregate projection diagnostics over every profile-pass decision visit.
/// Visit-weighted, not reach-weighted: a nonzero count is a flag that the
/// policy and tree disagree, not an equity estimate.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProjectionStats {
    pub remapped_visits: u64,
    pub remapped_mass: f64,
    pub dropped_visits: u64,
    pub dropped_mass: f64,
}

impl ProjectionStats {
    fn record(&mut self, row: &RowProjection) {
        if row.remapped_mass > 0.0 {
            self.remapped_visits += 1;
            self.remapped_mass += row.remapped_mass;
        }
        // Compact policy rows are stored as normalized f32, so a matched row
        // can come up short by a few ulps; only count drops above that noise.
        if row.dropped_mass > 1e-6 {
            self.dropped_visits += 1;
            self.dropped_mass += row.dropped_mass;
        }
    }
}

#[derive(Clone, Debug)]
struct DepthAggregate {
    actions: Vec<AbstractAction>,
    q: Vec<f64>,
    weight: f64,
}

/// Project a policy row onto the target legal actions. Complete rows preserve
/// their saved probabilities, project mass saved on tree-pruned actions per
/// the selected [`DominatedProjection`], and renormalize. Any incomplete row
/// uses the explicitly selected fallback for the whole row; there is no
/// silent mixing of known and invented probabilities.
pub fn projected_policy_row(
    policy: &dyn PolicyLookup,
    key: InfoSetKey,
    actions: &[AbstractAction],
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
) -> (ActionProbs, RowProjection) {
    let mapped: Vec<Option<f64>> = actions
        .iter()
        .map(|&action| policy.action_probability(key, action, values))
        .collect();
    let incomplete = mapped.iter().any(Option::is_none);
    if !incomplete {
        let mut probabilities: ActionProbs = mapped
            .into_iter()
            .map(|value| value.unwrap_or(0.0).max(0.0))
            .collect();
        let mut remapped_mass = 0.0;
        if projection == DominatedProjection::Remap {
            for (i, &action) in actions.iter().enumerate() {
                // The only prunes that remove a saved action while its row
                // survives name exactly these dominators: a hidden play's
                // own face-up card, and Fold for a certificate-pruned accept.
                let pruned_source = match action {
                    AbstractAction::PlayFaceUp(card) => AbstractAction::PlayFaceDown(card),
                    AbstractAction::Fold => AbstractAction::AcceptRaise,
                    _ => continue,
                };
                if actions.contains(&pruned_source) {
                    continue;
                }
                if let Some(mass) = policy.action_probability(key, pruned_source, values) {
                    if mass > 0.0 {
                        probabilities[i] += mass;
                        remapped_mass += mass;
                    }
                }
            }
        }
        // Source rows are stored normalized, so any shortfall left after
        // remapping is mass saved on absent actions with no dominator here.
        let dropped_mass = (1.0 - probabilities.iter().sum::<f64>()).max(0.0);
        normalize_or_uniform(&mut probabilities);
        return (
            probabilities,
            RowProjection {
                missing: false,
                remapped_mass,
                dropped_mass,
            },
        );
    }

    let mut probabilities: ActionProbs = match fallback {
        MissingPolicyFallback::All => vec![1.0; actions.len()].into_iter().collect(),
        MissingPolicyFallback::First => {
            let mut row: ActionProbs = vec![0.0; actions.len()].into_iter().collect();
            if !row.is_empty() {
                row[0] = 1.0;
            }
            row
        }
        MissingPolicyFallback::AllExceptRaise => actions
            .iter()
            .map(|action| (!matches!(action, AbstractAction::Raise(_))) as u8 as f64)
            .collect(),
    };
    normalize_or_uniform(&mut probabilities);
    (
        probabilities,
        RowProjection {
            missing: true,
            remapped_mass: 0.0,
            dropped_mass: 0.0,
        },
    )
}

fn normalize_or_uniform(probabilities: &mut ActionProbs) {
    let total: f64 = probabilities.iter().sum();
    if total > 0.0 {
        for probability in probabilities {
            *probability /= total;
        }
    } else if !probabilities.is_empty() {
        let uniform = 1.0 / probabilities.len() as f64;
        probabilities.fill(uniform);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compact_profile_value(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    rules: TreeRules,
    match_values: &MatchValueTable,
) -> Result<CompactProfileResult, truco_engine::EngineError> {
    let mut probabilities = [0.0; 25];
    let mut missing_decisions = 0u64;
    let mut projection_stats = ProjectionStats::default();
    let mut dfs_visits = 0u64;
    for deal in deals {
        for dealer in 0..=1u8 {
            if dealer_filter.is_some_and(|wanted| wanted != dealer) {
                continue;
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            let outcomes = profile_outcomes(
                &state,
                policy,
                values,
                fallback,
                projection,
                &mut missing_decisions,
                &mut projection_stats,
                &mut dfs_visits,
            )?;
            for (target, outcome) in probabilities.iter_mut().zip(outcomes) {
                *target += deal.weight * outcome;
            }
        }
    }

    let mut p0_value = 0.0;
    for (i, &probability) in probabilities.iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let payoff = i as i8 - 12;
        // Dealer alternation affects continuation value. With both dealers in
        // one result, recompute value directly instead of deriving it from the
        // merged kernel; allocator callers use a dealer filter.
        if dealer_filter.is_none() {
            p0_value = profile_value_direct(
                score,
                tc,
                deals,
                dealer_filter,
                policy,
                values,
                fallback,
                projection,
                rules,
                match_values,
                &mut dfs_visits,
            )?;
            break;
        }
        p0_value += probability
            * terminal_p0_value(payoff as f64, dealer_filter.unwrap(), score, match_values);
    }

    Ok(CompactProfileResult {
        p0_value,
        outcome_probabilities: probabilities,
        missing_decisions,
        dfs_visits,
        projection_stats,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compact_best_response_value(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    rules: TreeRules,
    match_values: &MatchValueTable,
    br_player: Player,
) -> Result<CompactBestResponseResult, truco_engine::EngineError> {
    let mut max_depth = 0usize;
    let mut dfs_visits = 0u64;
    for deal in deals {
        for dealer in 0..=1u8 {
            if dealer_filter.is_some_and(|wanted| wanted != dealer) {
                continue;
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            discover_br_depth(&state, br_player, &mut max_depth, &mut dfs_visits)?;
        }
    }

    let mut chosen: AHashMap<InfoSetKey, AbstractAction> = AHashMap::new();
    for target_depth in (0..=max_depth).rev() {
        let mut aggregates: AHashMap<InfoSetKey, DepthAggregate> = AHashMap::new();
        for deal in deals {
            for dealer in 0..=1u8 {
                if dealer_filter.is_some_and(|wanted| wanted != dealer) {
                    continue;
                }
                let state =
                    TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
                collect_depth(
                    &state,
                    target_depth,
                    deal.weight,
                    br_player,
                    dealer,
                    score,
                    match_values,
                    policy,
                    values,
                    fallback,
                    projection,
                    &chosen,
                    &mut aggregates,
                    &mut dfs_visits,
                )?;
            }
        }
        for (key, aggregate) in aggregates {
            let best = aggregate
                .q
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            debug_assert!(aggregate.weight >= 0.0);
            chosen.insert(key, aggregate.actions[best]);
        }
    }

    let mut total = 0.0;
    for deal in deals {
        for dealer in 0..=1u8 {
            if dealer_filter.is_some_and(|wanted| wanted != dealer) {
                continue;
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            total += deal.weight
                * br_evaluate(
                    &state,
                    br_player,
                    dealer,
                    score,
                    match_values,
                    policy,
                    values,
                    fallback,
                    projection,
                    &chosen,
                    &mut dfs_visits,
                )?;
        }
    }

    Ok(CompactBestResponseResult {
        total,
        chosen_info_sets: chosen.len(),
        max_depth,
        dfs_visits,
    })
}

fn discover_br_depth(
    state: &TraversalState,
    br_player: Player,
    max_depth: &mut usize,
    visits: &mut u64,
) -> Result<(), truco_engine::EngineError> {
    *visits += 1;
    if state.is_terminal() {
        return Ok(());
    }
    let info = state.current_info_set().expect("non-terminal info set");
    if info.player == br_player {
        *max_depth = (*max_depth).max(info.history.actions().len());
    }
    for action in state.abstract_legal_actions()? {
        discover_br_depth(
            &state.apply_abstract_action(action)?,
            br_player,
            max_depth,
            visits,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_depth(
    state: &TraversalState,
    target_depth: usize,
    weight: f64,
    br_player: Player,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    chosen: &AHashMap<InfoSetKey, AbstractAction>,
    aggregates: &mut AHashMap<InfoSetKey, DepthAggregate>,
    visits: &mut u64,
) -> Result<(), truco_engine::EngineError> {
    *visits += 1;
    if state.is_terminal() || weight == 0.0 {
        return Ok(());
    }
    let info = state.current_info_set().expect("non-terminal info set");
    let depth = info.history.actions().len();
    let actions = state.abstract_legal_actions()?;
    if info.player == br_player && depth == target_depth {
        let key = info.key();
        let mut values_by_action = Vec::with_capacity(actions.len());
        for &action in &actions {
            values_by_action.push(br_evaluate(
                &state.apply_abstract_action(action)?,
                br_player,
                dealer,
                score,
                match_values,
                policy,
                values,
                fallback,
                projection,
                chosen,
                visits,
            )?);
        }
        let aggregate = aggregates.entry(key).or_insert_with(|| DepthAggregate {
            actions: actions.clone(),
            q: vec![0.0; actions.len()],
            weight: 0.0,
        });
        assert_eq!(aggregate.actions, actions, "one info set changed actions");
        for (q, value) in aggregate.q.iter_mut().zip(values_by_action) {
            *q += weight * value;
        }
        aggregate.weight += weight;
        return Ok(());
    }
    if depth >= target_depth {
        return Ok(());
    }

    if info.player == br_player {
        for action in actions {
            collect_depth(
                &state.apply_abstract_action(action)?,
                target_depth,
                weight,
                br_player,
                dealer,
                score,
                match_values,
                policy,
                values,
                fallback,
                projection,
                chosen,
                aggregates,
                visits,
            )?;
        }
    } else {
        let (probabilities, _) =
            projected_policy_row(policy, info.key(), &actions, values, fallback, projection);
        for (&action, probability) in actions.iter().zip(probabilities) {
            if probability > 0.0 {
                collect_depth(
                    &state.apply_abstract_action(action)?,
                    target_depth,
                    weight * probability,
                    br_player,
                    dealer,
                    score,
                    match_values,
                    policy,
                    values,
                    fallback,
                    projection,
                    chosen,
                    aggregates,
                    visits,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn br_evaluate(
    state: &TraversalState,
    br_player: Player,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    chosen: &AHashMap<InfoSetKey, AbstractAction>,
    visits: &mut u64,
) -> Result<f64, truco_engine::EngineError> {
    *visits += 1;
    if state.is_terminal() {
        let winner = state.hand_winner().expect("terminal winner");
        let payoff = if winner == 0 {
            state.hand_value() as f64
        } else {
            -(state.hand_value() as f64)
        };
        let p0 = terminal_p0_value(payoff, dealer, score, match_values);
        return Ok(if br_player == 0 { p0 } else { -p0 });
    }
    let info = state.current_info_set().expect("non-terminal info set");
    let actions = state.abstract_legal_actions()?;
    if info.player == br_player {
        let selected = chosen
            .get(&info.key())
            .copied()
            .expect("deeper BR information set must already be resolved");
        br_evaluate(
            &state.apply_abstract_action(selected)?,
            br_player,
            dealer,
            score,
            match_values,
            policy,
            values,
            fallback,
            projection,
            chosen,
            visits,
        )
    } else {
        let (probabilities, _) =
            projected_policy_row(policy, info.key(), &actions, values, fallback, projection);
        let mut value = 0.0;
        for (&action, probability) in actions.iter().zip(probabilities) {
            if probability > 0.0 {
                value += probability
                    * br_evaluate(
                        &state.apply_abstract_action(action)?,
                        br_player,
                        dealer,
                        score,
                        match_values,
                        policy,
                        values,
                        fallback,
                        projection,
                        chosen,
                        visits,
                    )?;
            }
        }
        Ok(value)
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_outcomes(
    state: &TraversalState,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    missing_decisions: &mut u64,
    projection_stats: &mut ProjectionStats,
    visits: &mut u64,
) -> Result<[f64; 25], truco_engine::EngineError> {
    *visits += 1;
    if state.is_terminal() {
        let winner = state.hand_winner().expect("terminal winner");
        let payoff = if winner == 0 {
            state.hand_value() as i8
        } else {
            -(state.hand_value() as i8)
        };
        let mut outcomes = [0.0; 25];
        outcomes[(payoff + 12) as usize] = 1.0;
        return Ok(outcomes);
    }
    let info = state.current_info_set().expect("non-terminal info set");
    let actions = state.abstract_legal_actions()?;
    let (probabilities, row) =
        projected_policy_row(policy, info.key(), &actions, values, fallback, projection);
    *missing_decisions += row.missing as u64;
    projection_stats.record(&row);
    let mut outcomes = [0.0; 25];
    for (&action, probability) in actions.iter().zip(probabilities) {
        if probability == 0.0 {
            continue;
        }
        let child = profile_outcomes(
            &state.apply_abstract_action(action)?,
            policy,
            values,
            fallback,
            projection,
            missing_decisions,
            projection_stats,
            visits,
        )?;
        for (total, value) in outcomes.iter_mut().zip(child) {
            *total += probability * value;
        }
    }
    Ok(outcomes)
}

#[allow(clippy::too_many_arguments)]
fn profile_value_direct(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    rules: TreeRules,
    match_values: &MatchValueTable,
    visits: &mut u64,
) -> Result<f64, truco_engine::EngineError> {
    let mut total = 0.0;
    for deal in deals {
        for dealer in 0..=1u8 {
            if dealer_filter.is_some_and(|wanted| wanted != dealer) {
                continue;
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            total += deal.weight
                * profile_value_node(
                    &state,
                    dealer,
                    score,
                    policy,
                    values,
                    fallback,
                    projection,
                    match_values,
                    visits,
                )?;
        }
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
fn profile_value_node(
    state: &TraversalState,
    dealer: Player,
    score: &Score,
    policy: &dyn PolicyLookup,
    values: PolicyValueSource,
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
    match_values: &MatchValueTable,
    visits: &mut u64,
) -> Result<f64, truco_engine::EngineError> {
    *visits += 1;
    if state.is_terminal() {
        let winner = state.hand_winner().expect("terminal winner");
        let payoff = if winner == 0 {
            state.hand_value() as f64
        } else {
            -(state.hand_value() as f64)
        };
        return Ok(terminal_p0_value(payoff, dealer, score, match_values));
    }
    let info = state.current_info_set().expect("non-terminal info set");
    let actions = state.abstract_legal_actions()?;
    let (probabilities, _) =
        projected_policy_row(policy, info.key(), &actions, values, fallback, projection);
    let mut total = 0.0;
    for (&action, probability) in actions.iter().zip(probabilities) {
        if probability > 0.0 {
            total += probability
                * profile_value_node(
                    &state.apply_abstract_action(action)?,
                    dealer,
                    score,
                    policy,
                    values,
                    fallback,
                    projection,
                    match_values,
                    visits,
                )?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::cfr::{best_response_value, compute_game_value_per_dealer, subsample_deals};
    use crate::game_tree::build_all_trees_with_dealer;
    use crate::strategy::StrategyTable;

    fn assert_matches_prebuilt(deal_limit: usize) {
        let score = Score { zero: 10, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let mut deals = enumerate_deals(&tc);
        subsample_deals(&mut deals, deal_limit);
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let mut table = StrategyTable::new();
        for (_, info, actions) in &prebuilt.info_sets {
            let data = table.get_or_insert(info, actions);
            data.cumulative_strategy.fill(1.0);
        }
        let mv = MatchValueTable::new();
        let expected_br0 = best_response_value(&prebuilt, &table, &score, &mv, 0);
        let expected_br1 = best_response_value(&prebuilt, &table, &score, &mv, 1);
        let expected_profile = compute_game_value_per_dealer(&prebuilt, &table, &score, &mv).0;

        // Remap projection must be a no-op on a policy solved for this tree:
        // no saved action is absent, so nothing can be remapped or dropped.
        let compact0 = compact_best_response_value(
            &score,
            tc,
            &deals,
            Some(0),
            &table,
            PolicyValueSource::Average,
            MissingPolicyFallback::All,
            DominatedProjection::Remap,
            TreeRules::Current,
            &mv,
            0,
        )
        .unwrap();
        let compact1 = compact_best_response_value(
            &score,
            tc,
            &deals,
            Some(0),
            &table,
            PolicyValueSource::Average,
            MissingPolicyFallback::All,
            DominatedProjection::Remap,
            TreeRules::Current,
            &mv,
            1,
        )
        .unwrap();
        let profile = compact_profile_value(
            &score,
            tc,
            &deals,
            Some(0),
            &table,
            PolicyValueSource::Average,
            MissingPolicyFallback::All,
            DominatedProjection::Remap,
            TreeRules::Current,
            &mv,
        )
        .unwrap();

        assert!((compact0.total - expected_br0).abs() < 1e-10);
        assert!((compact1.total - expected_br1).abs() < 1e-10);
        assert!((profile.p0_value - expected_profile).abs() < 1e-10);
        assert_eq!(profile.projection_stats.remapped_visits, 0);
        assert_eq!(profile.projection_stats.dropped_visits, 0);
    }

    #[test]
    fn compact_dfs_matches_prebuilt_exact_br_and_profile_value() {
        assert_matches_prebuilt(12);
    }

    #[test]
    fn compact_dfs_matches_prebuilt_oracle_at_300_deals() {
        assert_matches_prebuilt(300);
    }

    mod projection {
        use super::*;
        use crate::abstraction::{AbstractCard, TurnupClass};
        use crate::info_set::{ActionHistory, InfoSet};
        use smallvec::smallvec;

        fn table_with_row(
            actions: &[AbstractAction],
            cumulative: &[f64],
        ) -> (StrategyTable, InfoSetKey) {
            let info = InfoSet {
                player: 1,
                is_dealer: false,
                turnup_class: TurnupClass {
                    blocked_plain_level: 0,
                },
                starting_hand: smallvec![
                    AbstractCard::Plain(0),
                    AbstractCard::Plain(1),
                    AbstractCard::Plain(2)
                ],
                history: ActionHistory::new(),
            };
            let mut table = StrategyTable::new();
            let data = table.get_or_insert(&info, actions);
            data.cumulative_strategy.copy_from_slice(cumulative);
            (table, info.key())
        }

        #[test]
        fn remap_moves_pruned_hide_mass_onto_same_card_face_up() {
            let card = AbstractCard::Plain(4);
            let saved = [
                AbstractAction::PlayFaceUp(card),
                AbstractAction::PlayFaceDown(card),
                AbstractAction::Raise(3),
            ];
            let (table, key) = table_with_row(&saved, &[0.2, 0.5, 0.3]);
            let target = [AbstractAction::PlayFaceUp(card), AbstractAction::Raise(3)];

            let (remapped, row) = projected_policy_row(
                &table,
                key,
                &target,
                PolicyValueSource::Average,
                MissingPolicyFallback::All,
                DominatedProjection::Remap,
            );
            assert!((remapped[0] - 0.7).abs() < 1e-12);
            assert!((remapped[1] - 0.3).abs() < 1e-12);
            assert!(!row.missing);
            assert!((row.remapped_mass - 0.5).abs() < 1e-12);
            assert!(row.dropped_mass < 1e-12);

            let (renormalized, row) = projected_policy_row(
                &table,
                key,
                &target,
                PolicyValueSource::Average,
                MissingPolicyFallback::All,
                DominatedProjection::Renormalize,
            );
            assert!((renormalized[0] - 0.4).abs() < 1e-12);
            assert!((renormalized[1] - 0.6).abs() < 1e-12);
            assert!((row.dropped_mass - 0.5).abs() < 1e-12);
        }

        #[test]
        fn remap_moves_certificate_pruned_accept_mass_onto_fold() {
            let saved = [
                AbstractAction::AcceptRaise,
                AbstractAction::Fold,
                AbstractAction::Raise(6),
            ];
            let (table, key) = table_with_row(&saved, &[0.6, 0.2, 0.2]);
            let target = [AbstractAction::Fold, AbstractAction::Raise(6)];

            let (remapped, row) = projected_policy_row(
                &table,
                key,
                &target,
                PolicyValueSource::Average,
                MissingPolicyFallback::All,
                DominatedProjection::Remap,
            );
            assert!((remapped[0] - 0.8).abs() < 1e-12);
            assert!((remapped[1] - 0.2).abs() < 1e-12);
            assert!((row.remapped_mass - 0.6).abs() < 1e-12);
        }

        #[test]
        fn remap_leaves_unpruned_sources_and_incomplete_rows_alone() {
            let card = AbstractCard::Plain(4);
            let saved = [
                AbstractAction::PlayFaceUp(card),
                AbstractAction::PlayFaceDown(card),
            ];
            let (table, key) = table_with_row(&saved, &[0.25, 0.75]);

            // Hide still legal in the target: mass must stay where it was saved.
            let (unchanged, row) = projected_policy_row(
                &table,
                key,
                &saved,
                PolicyValueSource::Average,
                MissingPolicyFallback::All,
                DominatedProjection::Remap,
            );
            assert!((unchanged[0] - 0.25).abs() < 1e-12);
            assert!((unchanged[1] - 0.75).abs() < 1e-12);
            assert_eq!(row.remapped_mass, 0.0);

            // A target action absent from the saved row still takes the
            // whole-row fallback, never a remap mixture.
            let target = [AbstractAction::PlayFaceUp(card), AbstractAction::Raise(3)];
            let (fallback_row, row) = projected_policy_row(
                &table,
                key,
                &target,
                PolicyValueSource::Average,
                MissingPolicyFallback::AllExceptRaise,
                DominatedProjection::Remap,
            );
            assert!(row.missing);
            assert!((fallback_row[0] - 1.0).abs() < 1e-12);
            assert_eq!(fallback_row[1], 0.0);
        }
    }
}
