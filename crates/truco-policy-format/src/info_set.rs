use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::abstraction::{AbstractCard, AbstractHand, TurnupClass};
use truco_engine::Player;

/// An action in the solver's abstract action space.
/// Replaces concrete card IDs with abstract card types.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum AbstractAction {
    PlayFaceUp(AbstractCard),
    PlayFaceDown(AbstractCard),
    /// Opponent played a card face-down (we don't know which card).
    OpponentPlayedHidden,
    Raise(u8),
    AcceptRaise,
    Fold,
    AcceptEleven,
    FoldEleven,
}

impl AbstractAction {
    /// Compact u8 codec for the packed tree arena (see `game_tree::GameTree`).
    /// 0..=12 face-up by card type; 13..=25 face-down; 26 hidden; 27..=30 the
    /// raise ladder (3/6/9/12); 31..=34 accept/fold/accept-eleven/fold-eleven.
    pub fn to_u8(self) -> u8 {
        match self {
            AbstractAction::PlayFaceUp(c) => c.type_index() as u8,
            AbstractAction::PlayFaceDown(c) => 13 + c.type_index() as u8,
            AbstractAction::OpponentPlayedHidden => 26,
            AbstractAction::Raise(to) => {
                debug_assert!(to % 3 == 0 && (3..=12).contains(&to));
                27 + (to / 3 - 1)
            }
            AbstractAction::AcceptRaise => 31,
            AbstractAction::Fold => 32,
            AbstractAction::AcceptEleven => 33,
            AbstractAction::FoldEleven => 34,
        }
    }

    pub fn try_from_u8(v: u8) -> Option<Self> {
        use crate::abstraction::AbstractCard;
        Some(match v {
            0..=12 => AbstractAction::PlayFaceUp(AbstractCard::from_type_index(v as usize)),
            13..=25 => {
                AbstractAction::PlayFaceDown(AbstractCard::from_type_index((v - 13) as usize))
            }
            26 => AbstractAction::OpponentPlayedHidden,
            27..=30 => AbstractAction::Raise((v - 27 + 1) * 3),
            31 => AbstractAction::AcceptRaise,
            32 => AbstractAction::Fold,
            33 => AbstractAction::AcceptEleven,
            34 => AbstractAction::FoldEleven,
            _ => return None,
        })
    }

    pub fn from_u8(v: u8) -> Self {
        Self::try_from_u8(v).unwrap_or_else(|| panic!("invalid packed action byte {v}"))
    }
}

/// Compact action history: sequence of actions visible to the info set's player.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ActionHistory {
    actions: SmallVec<[AbstractAction; 16]>,
}

impl ActionHistory {
    pub fn new() -> Self {
        Self {
            actions: SmallVec::new(),
        }
    }

    pub fn push(&mut self, action: AbstractAction) {
        self.actions.push(action);
    }

    pub fn actions(&self) -> &[AbstractAction] {
        &self.actions
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

impl Default for ActionHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Everything a player knows when it's their turn to act.
/// This is the key used to index into the strategy table.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct InfoSet {
    /// Which player's perspective this info set represents.
    pub player: Player,
    /// Whether this player is the dealer (pé) of the current hand. Position is
    /// public information and strategically real (the pé plays last each trick,
    /// a measured +5.6pp edge), so it must be part of the info set identity.
    /// Without it the SAME key appears in both dealer trees of a solve — most
    /// visibly the mão-de-onze accept/fold node at the empty history, but also
    /// card-play nodes where the visible action string coincides while own/
    /// opponent attribution differs — forcing CFR to learn one position-averaged
    /// policy for two different states (the 11x10 exploitability wall).
    pub is_dealer: bool,
    /// The turnup class (public information).
    pub turnup_class: TurnupClass,
    /// The player's own starting hand (sorted, abstract cards).
    /// We keep the full starting hand rather than just remaining cards
    /// because the starting hand is part of the information set identity.
    pub starting_hand: AbstractHand,
    /// The action history visible to this player.
    pub history: ActionHistory,
}

/// A 64-bit key for fast info set lookup, derived by hashing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InfoSetKey(pub u64);

impl InfoSet {
    pub fn new(
        player: Player,
        is_dealer: bool,
        turnup_class: TurnupClass,
        starting_hand: AbstractHand,
    ) -> Self {
        Self {
            player,
            is_dealer,
            turnup_class,
            starting_hand,
            history: ActionHistory::new(),
        }
    }

    /// Compute a hash key for fast lookup.
    ///
    /// Uses a FIXED-seed hasher so the key is deterministic across processes —
    /// `ahash::AHasher::default()` seeds from per-process RNG, which would make a
    /// checkpoint/strategy written by one solve unmatchable (so silently
    /// un-resumable / un-warm-startable) by another. Keys are 64-bit; collisions
    /// are astronomically unlikely at these info-set counts.
    pub fn key(&self) -> InfoSetKey {
        let state = ahash::RandomState::with_seeds(
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        );
        InfoSetKey(state.hash_one(self))
    }

    /// Record an action from this player's perspective.
    /// For own actions: record the actual action (including card identity for face-down).
    /// For opponent's face-down plays: record OpponentPlayedHidden.
    pub fn record_own_action(&mut self, action: AbstractAction) {
        self.history.push(action);
    }

    /// Record an opponent's action from this player's perspective.
    pub fn record_opponent_action(&mut self, action: AbstractAction) {
        match action {
            AbstractAction::PlayFaceDown(_) => {
                self.history.push(AbstractAction::OpponentPlayedHidden);
            }
            other => {
                self.history.push(other);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn test_action_u8_roundtrip() {
        let mut all: Vec<AbstractAction> = vec![
            AbstractAction::OpponentPlayedHidden,
            AbstractAction::AcceptRaise,
            AbstractAction::Fold,
            AbstractAction::AcceptEleven,
            AbstractAction::FoldEleven,
        ];
        for i in 0..AbstractCard::NUM_TYPES {
            all.push(AbstractAction::PlayFaceUp(AbstractCard::from_type_index(i)));
            all.push(AbstractAction::PlayFaceDown(AbstractCard::from_type_index(
                i,
            )));
        }
        for to in [3u8, 6, 9, 12] {
            all.push(AbstractAction::Raise(to));
        }
        let mut seen = std::collections::HashSet::new();
        for a in all {
            let b = a.to_u8();
            assert!(seen.insert(b), "codec collision at {b}");
            assert_eq!(AbstractAction::from_u8(b), a);
        }
        assert_eq!(AbstractAction::try_from_u8(35), None);
        assert_eq!(AbstractAction::try_from_u8(0xff), None);
    }

    #[test]
    fn test_info_set_equality() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };

        let is1 = InfoSet::new(0, false, tc, hand.clone());
        let is2 = InfoSet::new(0, false, tc, hand.clone());
        assert_eq!(is1, is2);
        assert_eq!(is1.key(), is2.key());
    }

    #[test]
    fn test_position_affects_info_set() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };

        // Same player, same hand, same (empty) history — e.g. the mão-de-onze
        // accept/fold node — must be distinct per position or CFR is forced to
        // one position-averaged policy across both dealer trees.
        let as_dealer = InfoSet::new(0, true, tc, hand.clone());
        let as_non_dealer = InfoSet::new(0, false, tc, hand);
        assert_ne!(as_dealer, as_non_dealer);
        assert_ne!(as_dealer.key(), as_non_dealer.key());
    }

    #[test]
    fn test_different_players_different_info_sets() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };

        let is1 = InfoSet::new(0, false, tc, hand.clone());
        let is2 = InfoSet::new(1, false, tc, hand);
        assert_ne!(is1, is2);
    }

    #[test]
    fn test_history_affects_info_set() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };

        let mut is1 = InfoSet::new(0, false, tc, hand.clone());
        let is2 = InfoSet::new(0, false, tc, hand);

        is1.record_own_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(2)));
        assert_ne!(is1, is2);
    }

    #[test]
    fn test_opponent_hidden_play_hides_card() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };

        // From our perspective, opponent playing any card face-down looks the same
        let mut is1 = InfoSet::new(0, false, tc, hand.clone());
        is1.record_opponent_action(AbstractAction::PlayFaceDown(AbstractCard::Plain(0)));

        let mut is2 = InfoSet::new(0, false, tc, hand);
        is2.record_opponent_action(AbstractAction::PlayFaceDown(AbstractCard::Manilha(3)));

        assert_eq!(is1, is2);
    }

    #[test]
    fn golden_info_set_key_v1() {
        let hand: AbstractHand = smallvec![
            AbstractCard::Plain(2),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(1),
        ];
        let mut info_set = InfoSet::new(
            1,
            false,
            TurnupClass {
                blocked_plain_level: 3,
            },
            hand,
        );
        info_set.record_own_action(AbstractAction::Raise(3));
        info_set.record_opponent_action(AbstractAction::PlayFaceDown(AbstractCard::Plain(4)));

        assert_eq!(info_set.key(), InfoSetKey(0xae26_698a_016e_544c));
    }
}
