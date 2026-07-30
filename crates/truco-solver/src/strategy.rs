use std::collections::HashMap;

use smallvec::SmallVec;

use crate::info_set::{AbstractAction, InfoSet, InfoSetKey};

/// Action-probability vector, inline-allocated: no solver node has more than
/// ~7 abstract actions (3 face-up + 3 face-down plays + a raise), so strategy
/// lookups in the CFR hot loop never touch the heap.
pub type ActionProbs = SmallVec<[f64; 8]>;

/// Action probability distribution at an information set.
#[derive(Clone, Debug)]
pub struct ActionDistribution {
    pub actions: Vec<(AbstractAction, f64)>,
}

impl ActionDistribution {
    /// Sample an action according to the distribution.
    pub fn sample(&self, rng: &mut impl rand::Rng) -> AbstractAction {
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        for &(action, prob) in &self.actions {
            cumulative += prob;
            if r < cumulative {
                return action;
            }
        }
        // Fallback to last action (handles floating-point edge case)
        self.actions
            .last()
            .expect("ActionDistribution must not be empty")
            .0
    }

    /// Get the probability of a specific action.
    pub fn probability_of(&self, action: AbstractAction) -> f64 {
        self.actions
            .iter()
            .find(|(a, _)| *a == action)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }
}

/// A strategy assigns action probabilities at each information set.
pub trait Strategy {
    fn action_probabilities(
        &self,
        info_set: &InfoSet,
        legal_actions: &[AbstractAction],
    ) -> ActionDistribution;
}

/// Per-information-set accumulated data during CFR.
#[derive(Clone, Debug)]
pub struct InfoSetData {
    /// Cumulative regret for each action (CFR+: always >= 0).
    pub cumulative_regret: Vec<f64>,
    /// Cumulative strategy (CFR+: linear-weighted sum).
    pub cumulative_strategy: Vec<f64>,
    /// This iteration's instantaneous regret, accumulated across deals during
    /// the owner's traversal and folded into `cumulative_regret` at iteration
    /// end. EMPTY except in buffered sweep modes (SyncCFR+/PCFR+), which call
    /// [`Self::ensure_prediction_buffers`] before iterating — at ~11M info
    /// sets these vectors are ~1 GB, so plain CFR+/DCFR must not pay for them.
    pub pending_regret: Vec<f64>,
    /// Previous iteration's instantaneous regret — the prediction used by
    /// [`Self::predictive_strategy`] (PCFR+ mode only; empty otherwise).
    pub last_regret: Vec<f64>,
    /// The legal actions at this info set (in fixed order).
    pub actions: Vec<AbstractAction>,
}

impl InfoSetData {
    pub fn new(actions: Vec<AbstractAction>) -> Self {
        let n = actions.len();
        Self {
            cumulative_regret: vec![0.0; n],
            cumulative_strategy: vec![0.0; n],
            pending_regret: Vec::new(),
            last_regret: Vec::new(),
            actions,
        }
    }

    /// Size the buffered-sweep vectors (idempotent). Called once per solve by
    /// the buffered sweep modes; see the field docs.
    pub fn ensure_prediction_buffers(&mut self) {
        let n = self.actions.len();
        if self.pending_regret.len() != n {
            self.pending_regret = vec![0.0; n];
        }
        if self.last_regret.len() != n {
            self.last_regret = vec![0.0; n];
        }
    }

    /// Predictive (optimistic) regret matching: play proportional to
    /// `max(0, R + r_last)`, using the previous iteration's instantaneous
    /// regret as the prediction of the next one (PCFR+, Farina et al.).
    /// Optimism damps the regret-matching oscillation on cyclic structures
    /// like the mão-de-onze accept screening.
    pub fn predictive_strategy(&self) -> ActionProbs {
        if self.last_regret.len() != self.cumulative_regret.len() {
            // Buffers not initialized (non-buffered mode): no prediction.
            return self.current_strategy();
        }
        let mut pred: ActionProbs = self
            .cumulative_regret
            .iter()
            .zip(self.last_regret.iter())
            .map(|(r, l)| (r + l).max(0.0))
            .collect();
        let positive_sum: f64 = pred.iter().sum();
        if positive_sum > 0.0 {
            for p in pred.iter_mut() {
                *p /= positive_sum;
            }
            pred
        } else {
            uniform_probs(self.actions.len())
        }
    }

    /// Compute current strategy from regrets via regret matching.
    /// Returns action probabilities in the same order as `self.actions`.
    pub fn current_strategy(&self) -> ActionProbs {
        let positive_sum: f64 = self.cumulative_regret.iter().map(|r| r.max(0.0)).sum();

        if positive_sum > 0.0 {
            self.cumulative_regret
                .iter()
                .map(|r| r.max(0.0) / positive_sum)
                .collect()
        } else {
            // Uniform strategy when all regrets are non-positive
            uniform_probs(self.actions.len())
        }
    }

    /// Get the average (converged) strategy from cumulative strategy sums.
    pub fn average_strategy(&self) -> ActionProbs {
        let total: f64 = self.cumulative_strategy.iter().sum();
        if total > 0.0 {
            self.cumulative_strategy.iter().map(|s| s / total).collect()
        } else {
            uniform_probs(self.actions.len())
        }
    }
}

/// Uniform distribution over `n` actions.
pub fn uniform_probs(n: usize) -> ActionProbs {
    let mut v = ActionProbs::new();
    v.resize(n, 1.0 / n as f64);
    v
}

impl InfoSetData {
    /// Get the average strategy as an ActionDistribution.
    pub fn average_strategy_distribution(&self) -> ActionDistribution {
        let probs = self.average_strategy();
        ActionDistribution {
            actions: self.actions.iter().copied().zip(probs).collect(),
        }
    }
}

/// The strategy table: maps info set keys to their accumulated data.
#[derive(Clone, Debug)]
pub struct StrategyTable {
    pub data: HashMap<InfoSetKey, InfoSetData>,
    pub info_sets: HashMap<InfoSetKey, InfoSet>,
}

impl StrategyTable {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            info_sets: HashMap::new(),
        }
    }

    /// Get or create the entry for an info set.
    pub fn get_or_insert(
        &mut self,
        info_set: &InfoSet,
        actions: &[AbstractAction],
    ) -> &mut InfoSetData {
        let key = info_set.key();
        match self.info_sets.entry(key) {
            std::collections::hash_map::Entry::Occupied(existing) => {
                debug_assert_eq!(
                    existing.get(),
                    info_set,
                    "info-set metadata mismatch for {:?}",
                    key
                );
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(info_set.clone());
            }
        }
        self.data
            .entry(key)
            .or_insert_with(|| InfoSetData::new(actions.to_vec()))
    }

    /// Look up an existing entry.
    pub fn get(&self, info_set: &InfoSet) -> Option<&InfoSetData> {
        self.data.get(&info_set.key())
    }

    pub fn get_info_set(&self, key: InfoSetKey) -> Option<&InfoSet> {
        self.info_sets.get(&key)
    }

    pub fn insert_serialized(
        &mut self,
        key: InfoSetKey,
        info_set: InfoSet,
        data: InfoSetData,
    ) -> Option<InfoSetData> {
        self.info_sets.insert(key, info_set);
        self.data.insert(key, data)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// The Nash equilibrium strategy from the solver.
pub struct EquilibriumStrategy {
    pub table: StrategyTable,
}

impl Strategy for EquilibriumStrategy {
    fn action_probabilities(
        &self,
        info_set: &InfoSet,
        legal_actions: &[AbstractAction],
    ) -> ActionDistribution {
        match self.table.get(info_set) {
            Some(data) => data.average_strategy_distribution(),
            None => {
                // Unknown info set: uniform
                let n = legal_actions.len() as f64;
                ActionDistribution {
                    actions: legal_actions.iter().map(|&a| (a, 1.0 / n)).collect(),
                }
            }
        }
    }
}

/// A deterministic pure strategy defined by a closure.
pub struct PureStrategy {
    policy: Box<dyn Fn(&InfoSet, &[AbstractAction]) -> AbstractAction + Send + Sync>,
}

impl PureStrategy {
    pub fn new(
        policy: impl Fn(&InfoSet, &[AbstractAction]) -> AbstractAction + Send + Sync + 'static,
    ) -> Self {
        Self {
            policy: Box::new(policy),
        }
    }

    /// Strategy that always picks the first legal action.
    pub fn first_action() -> Self {
        Self::new(|_, actions| actions[0])
    }

    /// Strategy that always folds when possible, otherwise plays the weakest card face-up.
    pub fn always_fold() -> Self {
        Self::new(|_, actions| {
            // Prefer Fold > FoldEleven > weakest PlayFaceUp
            for &a in actions {
                if matches!(a, AbstractAction::Fold) {
                    return a;
                }
            }
            for &a in actions {
                if matches!(a, AbstractAction::FoldEleven) {
                    return a;
                }
            }
            // Play weakest card face-up
            actions
                .iter()
                .filter(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
                .min_by_key(|a| match a {
                    AbstractAction::PlayFaceUp(c) => *c,
                    _ => unreachable!(),
                })
                .copied()
                .unwrap_or(actions[0])
        })
    }

    /// Strategy that always plays the strongest card face-up, never raises,
    /// accepts raises, and accepts eleven.
    pub fn strongest_face_up() -> Self {
        Self::new(|_, actions| {
            // Accept raises and eleven
            for &a in actions {
                if matches!(
                    a,
                    AbstractAction::AcceptRaise | AbstractAction::AcceptEleven
                ) {
                    return a;
                }
            }
            // Play strongest card face-up
            actions
                .iter()
                .filter(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
                .max_by_key(|a| match a {
                    AbstractAction::PlayFaceUp(c) => *c,
                    _ => unreachable!(),
                })
                .copied()
                .unwrap_or(actions[0])
        })
    }

    /// Strategy that plays the weakest card face-up in round 1,
    /// then plays face-down in subsequent rounds.
    pub fn weakest_then_hide() -> Self {
        Self::new(|_info_set, actions| {
            // Accept raises and eleven decisions
            for &a in actions {
                if matches!(
                    a,
                    AbstractAction::AcceptRaise | AbstractAction::AcceptEleven
                ) {
                    return a;
                }
            }

            // Check if face-down is available (only after round 1)
            let has_face_down = actions
                .iter()
                .any(|a| matches!(a, AbstractAction::PlayFaceDown(_)));

            if has_face_down {
                // Play weakest card face-down
                actions
                    .iter()
                    .filter(|a| matches!(a, AbstractAction::PlayFaceDown(_)))
                    .min_by_key(|a| match a {
                        AbstractAction::PlayFaceDown(c) => *c,
                        _ => unreachable!(),
                    })
                    .copied()
                    .unwrap_or(actions[0])
            } else {
                // Round 1: play weakest face-up
                actions
                    .iter()
                    .filter(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
                    .min_by_key(|a| match a {
                        AbstractAction::PlayFaceUp(c) => *c,
                        _ => unreachable!(),
                    })
                    .copied()
                    .unwrap_or(actions[0])
            }
        })
    }

    /// Strategy that always raises when possible, never folds.
    pub fn always_raise() -> Self {
        Self::new(|_, actions| {
            // Accept eleven
            for &a in actions {
                if matches!(a, AbstractAction::AcceptEleven) {
                    return a;
                }
            }
            // Raise if available
            for &a in actions {
                if matches!(a, AbstractAction::Raise(_)) {
                    return a;
                }
            }
            // Accept raise (re-raise not available)
            for &a in actions {
                if matches!(a, AbstractAction::AcceptRaise) {
                    return a;
                }
            }
            // Play strongest face-up
            actions
                .iter()
                .filter(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
                .max_by_key(|a| match a {
                    AbstractAction::PlayFaceUp(c) => *c,
                    _ => unreachable!(),
                })
                .copied()
                .unwrap_or(actions[0])
        })
    }
}

impl Strategy for PureStrategy {
    fn action_probabilities(
        &self,
        info_set: &InfoSet,
        legal_actions: &[AbstractAction],
    ) -> ActionDistribution {
        let chosen = (self.policy)(info_set, legal_actions);
        ActionDistribution {
            actions: legal_actions
                .iter()
                .map(|&a| (a, if a == chosen { 1.0 } else { 0.0 }))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbstractCard, TurnupClass};
    use smallvec::smallvec;

    #[test]
    fn test_regret_matching_uniform_when_no_regret() {
        let data = InfoSetData::new(vec![
            AbstractAction::PlayFaceUp(AbstractCard::Plain(0)),
            AbstractAction::PlayFaceUp(AbstractCard::Plain(5)),
            AbstractAction::Raise(3),
        ]);
        let strategy = data.current_strategy();
        assert_eq!(strategy.len(), 3);
        for &p in &strategy {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_regret_matching_with_positive_regrets() {
        let mut data = InfoSetData::new(vec![
            AbstractAction::PlayFaceUp(AbstractCard::Plain(0)),
            AbstractAction::PlayFaceUp(AbstractCard::Plain(5)),
        ]);
        data.cumulative_regret = vec![3.0, 1.0];
        let strategy = data.current_strategy();
        assert!((strategy[0] - 0.75).abs() < 1e-10);
        assert!((strategy[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_regret_matching_ignores_negative() {
        let mut data = InfoSetData::new(vec![
            AbstractAction::PlayFaceUp(AbstractCard::Plain(0)),
            AbstractAction::PlayFaceUp(AbstractCard::Plain(5)),
            AbstractAction::Raise(3),
        ]);
        data.cumulative_regret = vec![-5.0, 2.0, 8.0];
        let strategy = data.current_strategy();
        assert!((strategy[0] - 0.0).abs() < 1e-10);
        assert!((strategy[1] - 0.2).abs() < 1e-10);
        assert!((strategy[2] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_pure_strategy_always_fold() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let hand = smallvec![
            AbstractCard::Plain(1),
            AbstractCard::Plain(3),
            AbstractCard::Plain(7)
        ];
        let info_set = InfoSet::new(0, false, tc, hand);

        let strat = PureStrategy::always_fold();
        let actions = vec![AbstractAction::AcceptRaise, AbstractAction::Fold];
        let dist = strat.action_probabilities(&info_set, &actions);
        assert_eq!(dist.probability_of(AbstractAction::Fold), 1.0);
        assert_eq!(dist.probability_of(AbstractAction::AcceptRaise), 0.0);
    }
}
