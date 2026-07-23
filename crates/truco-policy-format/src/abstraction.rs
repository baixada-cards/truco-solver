use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use truco_engine::{Card, Rank, Suit, Turnup};

/// Abstract card in the solver's card space.
/// Plain cards are suit-independent non-manilha cards ordered by strength.
/// Manilha cards are distinguished by suit (which determines their relative strength).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub enum AbstractCard {
    /// Non-manilha card. Strength 0 (weakest) to 8 (strongest).
    Plain(u8),
    /// Manilha card. 0=Diamonds (weakest), 1=Spades, 2=Hearts, 3=Clubs (strongest).
    Manilha(u8),
}

impl AbstractCard {
    pub const NUM_TYPES: usize = 13;

    /// Convert to a flat index 0..12 for array indexing.
    pub fn type_index(self) -> usize {
        match self {
            AbstractCard::Plain(s) => s as usize,
            AbstractCard::Manilha(s) => 9 + s as usize,
        }
    }

    /// Convert from a flat index 0..12.
    pub fn from_type_index(i: usize) -> Self {
        if i < 9 {
            AbstractCard::Plain(i as u8)
        } else {
            AbstractCard::Manilha((i - 9) as u8)
        }
    }
}

/// Identifies a turnup class: which plain strength level has only 3 copies
/// instead of 4 (because one card of that rank is the face-up turnup).
///
/// There are 9 distinct turnup classes (blocked_plain_level 0..=8).
/// Turnup ranks Two and Three both map to class 8.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TurnupClass {
    pub blocked_plain_level: u8,
}

impl TurnupClass {
    /// Number of available cards for a given abstract card type.
    pub fn available_count(&self, card: AbstractCard) -> u8 {
        match card {
            AbstractCard::Plain(level) => {
                if level == self.blocked_plain_level {
                    3
                } else {
                    4
                }
            }
            AbstractCard::Manilha(_) => 1,
        }
    }

    /// Total cards in the deck for this turnup class (always 39).
    pub fn deck_size(&self) -> u8 {
        39
    }

    /// Build the availability array: counts[type_index] = available count.
    pub fn availability(&self) -> [u8; AbstractCard::NUM_TYPES] {
        let mut counts = [0u8; AbstractCard::NUM_TYPES];
        for (i, count) in counts.iter_mut().enumerate() {
            *count = self.available_count(AbstractCard::from_type_index(i));
        }
        counts
    }

    /// All 9 distinct turnup classes.
    pub fn all() -> [TurnupClass; 9] {
        let mut classes = [TurnupClass {
            blocked_plain_level: 0,
        }; 9];
        for (i, class) in classes.iter_mut().enumerate() {
            *class = TurnupClass {
                blocked_plain_level: i as u8,
            };
        }
        classes
    }

    /// Probability weight for this turnup class (out of 10 equally likely turnup ranks).
    /// Classes 0..=7 have weight 1/10 each. Class 8 has weight 2/10 (Two and Three
    /// both produce blocked_plain_level=8).
    pub fn weight(&self) -> f64 {
        if self.blocked_plain_level == 8 {
            2.0 / 10.0
        } else {
            1.0 / 10.0
        }
    }
}

/// Map a concrete card to its abstract representation given a turnup.
pub fn abstract_card(card: &Card, turnup: &Turnup) -> AbstractCard {
    let manilha_rank = turnup.rank.next_for_manilha();
    if card.rank == manilha_rank {
        AbstractCard::Manilha(card.suit.manilha_strength() as u8)
    } else {
        let full_index = card.rank.index();
        let manilha_index = manilha_rank.index();
        let plain = if full_index < manilha_index {
            full_index
        } else {
            full_index - 1
        };
        AbstractCard::Plain(plain as u8)
    }
}

/// Derive the turnup class from a concrete turnup card.
pub fn turnup_class(turnup: &Turnup) -> TurnupClass {
    let turnup_index = turnup.rank.index();
    let manilha_index = turnup.rank.next_for_manilha().index();
    let blocked = if turnup_index < manilha_index {
        turnup_index
    } else {
        turnup_index - 1
    };
    TurnupClass {
        blocked_plain_level: blocked as u8,
    }
}

/// A sorted hand of 3 abstract cards.
pub type AbstractHand = SmallVec<[AbstractCard; 3]>;

/// An abstract deal: both players' hands plus the turnup class.
#[derive(Clone, Debug)]
pub struct AbstractDeal {
    pub hands: [AbstractHand; 2],
    /// Combinatorial weight: number of concrete deals that map to this abstract deal,
    /// divided by total number of concrete deals. Sums to 1 over all abstract deals.
    pub weight: f64,
}

/// Enumerate all abstract deals for a given turnup class.
/// Returns (abstract_hand_0, abstract_hand_1, weight) triples.
/// Weights sum to 1.0.
pub fn enumerate_deals(tc: &TurnupClass) -> Vec<AbstractDeal> {
    let avail = tc.availability();
    let hands_0 = enumerate_hands(&avail);
    let mut deals = Vec::new();
    let mut total_weight = 0.0;

    for (hand_0, ways_0) in &hands_0 {
        let mut remaining = avail;
        for &card in hand_0.iter() {
            remaining[card.type_index()] -= 1;
        }
        let hands_1 = enumerate_hands(&remaining);
        for (hand_1, ways_1) in &hands_1 {
            let w = (*ways_0 as f64) * (*ways_1 as f64);
            total_weight += w;
            deals.push(AbstractDeal {
                hands: [hand_0.clone(), hand_1.clone()],
                weight: w,
            });
        }
    }

    // Normalize weights to sum to 1.
    if total_weight > 0.0 {
        for deal in &mut deals {
            deal.weight /= total_weight;
        }
    }

    deals
}

/// Enumerate all multisets of size 3 from the given availability counts.
/// Returns (sorted hand, number of concrete ways to draw this hand).
fn enumerate_hands(avail: &[u8; AbstractCard::NUM_TYPES]) -> Vec<(AbstractHand, u64)> {
    let mut results = Vec::new();
    let mut hand = SmallVec::new();
    enumerate_hands_rec(avail, 0, 3, &mut hand, 1, &mut results);
    results
}

fn enumerate_hands_rec(
    avail: &[u8; AbstractCard::NUM_TYPES],
    start_type: usize,
    remaining: usize,
    hand: &mut AbstractHand,
    ways: u64,
    results: &mut Vec<(AbstractHand, u64)>,
) {
    if remaining == 0 {
        results.push((hand.clone(), ways));
        return;
    }
    for t in start_type..AbstractCard::NUM_TYPES {
        let max_take = (avail[t] as usize).min(remaining);
        if max_take == 0 {
            continue;
        }
        for take in 1..=max_take {
            let card = AbstractCard::from_type_index(t);
            for _ in 0..take {
                hand.push(card);
            }
            let new_ways = ways * choose(avail[t] as u64, take as u64);
            enumerate_hands_rec(avail, t + 1, remaining - take, hand, new_ways, results);
            for _ in 0..take {
                hand.pop();
            }
        }
    }
}

/// Binomial coefficient C(n, k).
fn choose(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// All 10 Rank values, for iterating.
pub const ALL_RANKS: [Rank; 10] = [
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Queen,
    Rank::Jack,
    Rank::King,
    Rank::Ace,
    Rank::Two,
    Rank::Three,
];

pub const ALL_SUITS: [Suit; 4] = [Suit::Diamonds, Suit::Spades, Suit::Hearts, Suit::Clubs];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstract_card_ordering() {
        // Plain cards are weaker than manilhas
        assert!(AbstractCard::Plain(8) < AbstractCard::Manilha(0));
        // Plains ordered by strength
        assert!(AbstractCard::Plain(0) < AbstractCard::Plain(8));
        // Manilhas ordered by suit strength
        assert!(AbstractCard::Manilha(0) < AbstractCard::Manilha(3));
    }

    #[test]
    fn test_type_index_roundtrip() {
        for i in 0..13 {
            assert_eq!(AbstractCard::from_type_index(i).type_index(), i);
        }
    }

    #[test]
    fn test_turnup_class_from_turnup() {
        // Turnup = 7 of Hearts → manilha = Queen (index 4)
        // 7 has index 3, which is < 4, so blocked = 3
        let tc = turnup_class(&Turnup {
            rank: Rank::Seven,
            suit: Suit::Hearts,
        });
        assert_eq!(tc.blocked_plain_level, 3);

        // Turnup = Three of Spades → manilha = Four (index 0)
        // Three has index 9, which is > 0, so blocked = 9 - 1 = 8
        let tc = turnup_class(&Turnup {
            rank: Rank::Three,
            suit: Suit::Spades,
        });
        assert_eq!(tc.blocked_plain_level, 8);

        // Turnup = Two of Clubs → manilha = Three (index 9)
        // Two has index 8, which is < 9, so blocked = 8
        let tc = turnup_class(&Turnup {
            rank: Rank::Two,
            suit: Suit::Clubs,
        });
        assert_eq!(tc.blocked_plain_level, 8);
    }

    #[test]
    fn test_two_and_three_turnups_same_class() {
        let tc_two = turnup_class(&Turnup {
            rank: Rank::Two,
            suit: Suit::Hearts,
        });
        let tc_three = turnup_class(&Turnup {
            rank: Rank::Three,
            suit: Suit::Hearts,
        });
        assert_eq!(tc_two, tc_three);
    }

    #[test]
    fn test_abstract_card_mapping() {
        // Turnup = 7 of Hearts → manilha rank is Queen
        let turnup = Turnup {
            rank: Rank::Seven,
            suit: Suit::Hearts,
        };

        // Queen of any suit is a manilha
        let queen_clubs = Card {
            id: "qc".into(),
            rank: Rank::Queen,
            suit: Suit::Clubs,
        };
        assert_eq!(
            abstract_card(&queen_clubs, &turnup),
            AbstractCard::Manilha(3)
        );

        let queen_diamonds = Card {
            id: "qd".into(),
            rank: Rank::Queen,
            suit: Suit::Diamonds,
        };
        assert_eq!(
            abstract_card(&queen_diamonds, &turnup),
            AbstractCard::Manilha(0)
        );

        // Four is the weakest non-manilha (index 0)
        let four = Card {
            id: "4s".into(),
            rank: Rank::Four,
            suit: Suit::Spades,
        };
        assert_eq!(abstract_card(&four, &turnup), AbstractCard::Plain(0));

        // Three is the strongest non-manilha (index 9 in full, manilha is at 4,
        // so plain = 9 - 1 = 8)
        let three = Card {
            id: "3s".into(),
            rank: Rank::Three,
            suit: Suit::Spades,
        };
        assert_eq!(abstract_card(&three, &turnup), AbstractCard::Plain(8));

        // Seven is below manilha (Queen at index 4), so plain = 3
        let seven = Card {
            id: "7d".into(),
            rank: Rank::Seven,
            suit: Suit::Diamonds,
        };
        assert_eq!(abstract_card(&seven, &turnup), AbstractCard::Plain(3));

        // Jack is above manilha (Queen at index 4, Jack at index 5)
        // plain = 5 - 1 = 4
        let jack = Card {
            id: "jh".into(),
            rank: Rank::Jack,
            suit: Suit::Hearts,
        };
        assert_eq!(abstract_card(&jack, &turnup), AbstractCard::Plain(4));
    }

    #[test]
    fn test_availability_sums_to_39() {
        for tc in TurnupClass::all() {
            let avail = tc.availability();
            let total: u8 = avail.iter().sum();
            assert_eq!(total, 39, "turnup class {:?} deck size wrong", tc);
        }
    }

    #[test]
    fn test_turnup_class_weights_sum_to_1() {
        let total: f64 = TurnupClass::all().iter().map(|tc| tc.weight()).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_deal_weights_sum_to_1() {
        // Test with one turnup class
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let deals = enumerate_deals(&tc);
        let total: f64 = deals.iter().map(|d| d.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "deal weights sum to {} (expected 1.0)",
            total
        );
    }

    #[test]
    fn test_all_hands_have_3_cards() {
        let tc = TurnupClass {
            blocked_plain_level: 4,
        };
        let deals = enumerate_deals(&tc);
        for deal in &deals {
            assert_eq!(deal.hands[0].len(), 3);
            assert_eq!(deal.hands[1].len(), 3);
        }
    }

    #[test]
    fn test_no_deal_exceeds_availability() {
        let tc = TurnupClass {
            blocked_plain_level: 2,
        };
        let avail = tc.availability();
        let deals = enumerate_deals(&tc);
        for deal in &deals {
            let mut used = [0u8; AbstractCard::NUM_TYPES];
            for &card in deal.hands[0].iter().chain(deal.hands[1].iter()) {
                used[card.type_index()] += 1;
            }
            for i in 0..AbstractCard::NUM_TYPES {
                assert!(
                    used[i] <= avail[i],
                    "type {} used {} but only {} available",
                    i,
                    used[i],
                    avail[i]
                );
            }
        }
    }

    #[test]
    fn test_nine_distinct_turnup_classes() {
        let mut classes: Vec<TurnupClass> = ALL_RANKS
            .iter()
            .map(|r| {
                turnup_class(&Turnup {
                    rank: *r,
                    suit: Suit::Hearts,
                })
            })
            .collect();
        classes.sort_by_key(|tc| tc.blocked_plain_level);
        classes.dedup();
        assert_eq!(classes.len(), 9);
    }
}
