//! Cheap, explicitly sampled whole-match error allocation.
//!
//! A fixed policy is evaluated on small deterministic deal panels at one
//! representative score per tree-shape band. Profile hand-outcome kernels are
//! reused across the 12x12 score DAG. Compact one-hand best responses then
//! estimate where unilateral error is encountered under profile reach. This is
//! a prioritization instrument, not a whole-match exploitability certificate:
//! it uses profile (not adversarial) reach and representative-state gains.

use std::collections::HashMap;

use truco_engine::{Player, Score, MATCH_TARGET};

use crate::abstraction::{AbstractDeal, TurnupClass};
use crate::compact_br::{compact_best_response_value, compact_profile_value, DominatedProjection};
use crate::game_tree::{MissingPolicyFallback, PolicyLookup, PolicyValueSource, TreeRules};
use crate::match_value::MatchValueTable;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationBand {
    MaoBoth,
    MaoP0,
    MaoP1,
    Raise3,
    Raise6,
    Raise9,
    Raise12,
}

pub const ALLOCATION_BANDS: [AllocationBand; 7] = [
    AllocationBand::MaoBoth,
    AllocationBand::MaoP0,
    AllocationBand::MaoP1,
    AllocationBand::Raise3,
    AllocationBand::Raise6,
    AllocationBand::Raise9,
    AllocationBand::Raise12,
];

impl AllocationBand {
    pub fn label(self) -> &'static str {
        match self {
            Self::MaoBoth => "mao-11x11",
            Self::MaoP0 => "mao-p0",
            Self::MaoP1 => "mao-p1",
            Self::Raise3 => "ladder-{1,3}",
            Self::Raise6 => "ladder-{1,3,6}",
            Self::Raise9 => "ladder-{1,3,6,9}",
            Self::Raise12 => "ladder-full",
        }
    }

    pub fn representative(self) -> Score {
        match self {
            Self::MaoBoth => Score { zero: 11, one: 11 },
            Self::MaoP0 => Score { zero: 11, one: 10 },
            Self::MaoP1 => Score { zero: 10, one: 11 },
            Self::Raise3 => Score { zero: 9, one: 9 },
            Self::Raise6 => Score { zero: 8, one: 8 },
            Self::Raise9 => Score { zero: 5, one: 5 },
            Self::Raise12 => Score { zero: 0, one: 0 },
        }
    }
}

pub fn allocation_band(score: &Score) -> AllocationBand {
    if score.zero == 11 && score.one == 11 {
        AllocationBand::MaoBoth
    } else if score.zero == 11 {
        AllocationBand::MaoP0
    } else if score.one == 11 {
        AllocationBand::MaoP1
    } else {
        match score.zero.min(score.one) {
            9..=10 => AllocationBand::Raise3,
            6..=8 => AllocationBand::Raise6,
            3..=5 => AllocationBand::Raise9,
            _ => AllocationBand::Raise12,
        }
    }
}

pub struct DealPanel<'a> {
    pub tc: TurnupClass,
    pub deals: &'a [AbstractDeal],
}

#[derive(Clone, Debug, Default)]
pub struct BandAllocationEstimate {
    pub visits: f64,
    pub gain0: f64,
    pub gain1: f64,
    pub contribution0: f64,
    pub contribution1: f64,
}

#[derive(Clone, Debug)]
pub struct AllocationEstimate {
    pub initial_profile_p0: f64,
    pub expected_hands: f64,
    /// Reach-weighted sum of representative one-hand deviations. These are
    /// allocator priority masses and may exceed the bounded match utility when
    /// several hands are visited; they are not exploitability or equity pp.
    pub priority_gain0_mass: f64,
    pub priority_gain1_mass: f64,
    pub priority_error_mass: f64,
    pub missing_profile_decisions: u64,
    pub dfs_visits: u64,
    pub by_band: HashMap<AllocationBand, BandAllocationEstimate>,
}

type KernelKey = (AllocationBand, u8, Player);

#[allow(clippy::too_many_arguments)]
pub fn estimate_whole_match_allocation(
    policy: &dyn PolicyLookup,
    panels: &[DealPanel<'_>],
    fallback: MissingPolicyFallback,
    projection: DominatedProjection,
) -> Result<AllocationEstimate, truco_engine::EngineError> {
    assert!(!panels.is_empty(), "allocator needs at least one TC panel");
    let total_tc_weight: f64 = panels.iter().map(|panel| panel.tc.weight()).sum();
    assert!(total_tc_weight > 0.0);

    let mut kernels: HashMap<KernelKey, [f64; 25]> = HashMap::new();
    let mut missing_profile_decisions = 0u64;
    let mut dfs_visits = 0u64;
    let placeholder_mv = MatchValueTable::new();
    for band in ALLOCATION_BANDS {
        let score = band.representative();
        for panel in panels {
            for dealer in 0..=1u8 {
                let profile = compact_profile_value(
                    &score,
                    panel.tc,
                    panel.deals,
                    Some(dealer),
                    policy,
                    PolicyValueSource::Average,
                    fallback,
                    projection,
                    TreeRules::Current,
                    &placeholder_mv,
                )?;
                missing_profile_decisions += profile.missing_decisions;
                dfs_visits += profile.dfs_visits;
                kernels.insert(
                    (band, panel.tc.blocked_plain_level, dealer),
                    profile.outcome_probabilities,
                );
            }
        }
    }

    // Backward induction evaluates the sampled fixed policy at every score.
    let mut match_values = MatchValueTable::new();
    for total in (0..=22u8).rev() {
        for s0 in 0..=11u8 {
            let Some(s1) = total.checked_sub(s0) else {
                continue;
            };
            if s1 > 11 {
                continue;
            }
            let score = Score { zero: s0, one: s1 };
            let band = allocation_band(&score);
            for dealer in 0..=1u8 {
                let mut probability = 0.0;
                for panel in panels {
                    let tc_weight = panel.tc.weight() / total_tc_weight;
                    let outcomes = kernels[&(band, panel.tc.blocked_plain_level, dealer)];
                    probability += tc_weight
                        * outcomes
                            .iter()
                            .enumerate()
                            .map(|(i, &mass)| {
                                if mass == 0.0 {
                                    0.0
                                } else {
                                    mass * continuation_probability(
                                        i as i8 - 12,
                                        dealer,
                                        &score,
                                        &match_values,
                                    )
                                }
                            })
                            .sum::<f64>();
                }
                match_values.set(s0, s1, dealer, probability);
            }
        }
    }

    // Forward profile reach. This is expected visit count, not probability of
    // ever visiting: a band can be encountered on several hands in one match.
    let mut reach = [[[0.0f64; 2]; 12]; 12];
    reach[0][0] = [0.5, 0.5];
    let mut band_dealer_visits: HashMap<(AllocationBand, Player), f64> = HashMap::new();
    for total in 0..=22u8 {
        for s0 in 0..=11u8 {
            let Some(s1) = total.checked_sub(s0) else {
                continue;
            };
            if s1 > 11 {
                continue;
            }
            let score = Score { zero: s0, one: s1 };
            let band = allocation_band(&score);
            for dealer in 0..=1u8 {
                let state_reach = reach[s0 as usize][s1 as usize][dealer as usize];
                if state_reach == 0.0 {
                    continue;
                }
                *band_dealer_visits.entry((band, dealer)).or_default() += state_reach;
                for panel in panels {
                    let tc_weight = panel.tc.weight() / total_tc_weight;
                    let outcomes = kernels[&(band, panel.tc.blocked_plain_level, dealer)];
                    for (i, &outcome_mass) in outcomes.iter().enumerate() {
                        let mass = state_reach * tc_weight * outcome_mass;
                        if mass == 0.0 {
                            continue;
                        }
                        if let Some(next) = next_nonterminal_score(&score, i as i8 - 12) {
                            reach[next.zero as usize][next.one as usize][(1 - dealer) as usize] +=
                                mass;
                        }
                    }
                }
            }
        }
    }

    let mut by_band = HashMap::new();
    let mut priority_gain0_mass = 0.0;
    let mut priority_gain1_mass = 0.0;
    for band in ALLOCATION_BANDS {
        let score = band.representative();
        let mut estimate = BandAllocationEstimate::default();
        for dealer in 0..=1u8 {
            let visits = band_dealer_visits
                .get(&(band, dealer))
                .copied()
                .unwrap_or(0.0);
            estimate.visits += visits;
            let mut gain0 = 0.0;
            let mut gain1 = 0.0;
            for panel in panels {
                let tc_weight = panel.tc.weight() / total_tc_weight;
                let outcomes = kernels[&(band, panel.tc.blocked_plain_level, dealer)];
                let profile_p0 = outcomes
                    .iter()
                    .enumerate()
                    .map(|(i, &mass)| {
                        if mass == 0.0 {
                            0.0
                        } else {
                            mass * (2.0
                                * continuation_probability(
                                    i as i8 - 12,
                                    dealer,
                                    &score,
                                    &match_values,
                                )
                                - 1.0)
                        }
                    })
                    .sum::<f64>();
                let br0 = compact_best_response_value(
                    &score,
                    panel.tc,
                    panel.deals,
                    Some(dealer),
                    policy,
                    PolicyValueSource::Average,
                    fallback,
                    projection,
                    TreeRules::Current,
                    &match_values,
                    0,
                )?;
                let br1 = compact_best_response_value(
                    &score,
                    panel.tc,
                    panel.deals,
                    Some(dealer),
                    policy,
                    PolicyValueSource::Average,
                    fallback,
                    projection,
                    TreeRules::Current,
                    &match_values,
                    1,
                )?;
                dfs_visits += br0.dfs_visits + br1.dfs_visits;
                gain0 += tc_weight * (br0.total - profile_p0).max(0.0);
                gain1 += tc_weight * (br1.total + profile_p0).max(0.0);
            }
            estimate.gain0 += visits * gain0;
            estimate.gain1 += visits * gain1;
        }
        estimate.contribution0 = estimate.gain0;
        estimate.contribution1 = estimate.gain1;
        priority_gain0_mass += estimate.gain0;
        priority_gain1_mass += estimate.gain1;
        by_band.insert(band, estimate);
    }

    let expected_hands: f64 = band_dealer_visits.values().sum();
    Ok(AllocationEstimate {
        initial_profile_p0: (match_values.get(0, 0, 0) + match_values.get(0, 0, 1)) / 2.0,
        expected_hands,
        priority_gain0_mass,
        priority_gain1_mass,
        priority_error_mass: (priority_gain0_mass + priority_gain1_mass) / 2.0,
        missing_profile_decisions,
        dfs_visits,
        by_band,
    })
}

fn continuation_probability(
    payoff: i8,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
) -> f64 {
    match next_nonterminal_score(score, payoff) {
        Some(next) => match_values.get(next.zero, next.one, 1 - dealer),
        None if payoff > 0 => 1.0,
        None => 0.0,
    }
}

fn next_nonterminal_score(score: &Score, payoff: i8) -> Option<Score> {
    debug_assert_ne!(payoff, 0);
    let mut next = score.clone();
    if payoff > 0 {
        next.zero = next.zero.saturating_add(payoff as u8).min(MATCH_TARGET);
    } else {
        next.one = next
            .one
            .saturating_add(payoff.unsigned_abs())
            .min(MATCH_TARGET);
    }
    (next.zero < MATCH_TARGET && next.one < MATCH_TARGET).then_some(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::cfr::subsample_deals;
    use crate::strategy::StrategyTable;

    #[test]
    fn sampled_allocator_is_finite_and_covers_every_band() {
        let tc = TurnupClass {
            blocked_plain_level: 4,
        };
        let mut deals = enumerate_deals(&tc);
        subsample_deals(&mut deals, 2);
        let panels = [DealPanel { tc, deals: &deals }];
        let estimate = estimate_whole_match_allocation(
            &StrategyTable::new(),
            &panels,
            MissingPolicyFallback::AllExceptRaise,
            DominatedProjection::Remap,
        )
        .unwrap();
        assert!((0.0..=1.0).contains(&estimate.initial_profile_p0));
        assert!(estimate.expected_hands > 0.0);
        assert!(estimate.priority_error_mass.is_finite());
        assert!(estimate.priority_gain0_mass >= 0.0);
        assert!(estimate.priority_gain1_mass >= 0.0);
        assert_eq!(estimate.by_band.len(), ALLOCATION_BANDS.len());
    }
}
