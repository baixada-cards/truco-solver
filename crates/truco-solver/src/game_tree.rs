use truco_engine::state::Visibility;
use truco_engine::{Action, Card, Engine, Hands, Player, Rank, Score, Suit, Turnup};

use crate::abstraction::{
    AbstractCard, AbstractDeal, AbstractHand, TurnupClass, ALL_RANKS, ALL_SUITS,
};
use crate::info_set::{AbstractAction, ActionHistory, InfoSet, InfoSetKey};
use crate::strategy::ActionProbs;
use crate::strategy::StrategyTable;

/// Solver-time legal-action metadata. A boxed slice keeps the allocation used
/// by the historical `Vec` but drops its unused capacity word: 16 bytes of
/// row metadata instead of 24, without inflating every row to the maximum
/// action count as an inline small-vector would.
pub type InfoActions = Box<[AbstractAction]>;

/// Public read-only average-strategy profile, indexed by an arena's local
/// `table_idx`. Lets the deep-solve certificate feed its per-subgame / trunk
/// composed rows (kept in various backings) to the crate-private best-response
/// core without exposing that core's trait. `average_strategy(idx)` returns the
/// row for info set `idx`; `len()` is the info-set count of the backing arena.
pub trait AverageProfile {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn average_strategy(&self, idx: usize) -> ActionProbs;
}

impl AverageProfile for [ActionProbs] {
    fn len(&self) -> usize {
        <[ActionProbs]>::len(self)
    }
    fn average_strategy(&self, idx: usize) -> ActionProbs {
        self[idx].clone()
    }
}

impl AverageProfile for Vec<ActionProbs> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn average_strategy(&self, idx: usize) -> ActionProbs {
        self[idx].clone()
    }
}

/// Which action-pruning rule set a traversal builds. A saved policy is only
/// meaningful on the tree it was solved for, so evaluators must be able to
/// reconstruct the tree that existed when an artifact was produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreeRules {
    /// All current prunes, including the 2026-07-16 proof-scoped hidden-play
    /// and forced-fold removals.
    #[default]
    Current,
    /// The tree as it existed before the 2026-07-16 proof-scoped prunes:
    /// score-aware raise pruning only. Use to evaluate/certify checkpoints
    /// solved before those prunes on the tree they were actually solved for.
    LegacyPreProofPrunes,
    /// `Current` plus asymmetric (per-acting-player) raise pruning. The
    /// deployed rule prunes raises when `min(score) + on_table >= MATCH_TARGET`
    /// (both players match-decided). This variant instead prunes the ACTING
    /// player's raises when `their own score + on_table >= MATCH_TARGET`,
    /// even if the opponent is not yet match-decided. Since the acting score is
    /// always >= min(score), this is a strict superset of the deployed prune.
    ///
    /// Proof it is value-preserving: if the acting player p already reaches
    /// MATCH_TARGET by securing `on_table` (win the hand, OR the opponent folds
    /// the raise and concedes `on_table`), escalating adds no prize — the match
    /// caps p's winnings — and no fold leverage: an opponent that folds concedes
    /// `on_table` to p and thereby loses the match, so it strictly prefers to
    /// accept (any positive hand-win chance beats a certain match loss) and
    /// never folds. The raise is therefore pure downside for p and weakly
    /// dominated. This is exactly the deployed rule's own argument applied per
    /// acting player rather than to the lower-scored player. Experimental until
    /// its value/exploitability A/B gate passes.
    AsymmetricRaisePrune,
}

/// State tracked during game tree traversal, bridging abstract and concrete worlds.
#[derive(Clone, Debug)]
pub struct TraversalState {
    /// The concrete engine state.
    pub engine: Engine,
    /// Turnup class for this hand.
    pub turnup_class: TurnupClass,
    /// Each player's starting hand in abstract form (sorted).
    pub starting_hands: [AbstractHand; 2],
    /// Each player's action history from their own perspective.
    pub histories: [ActionHistory; 2],
    /// Mapping from concrete card IDs to abstract cards, for the current hand.
    pub card_map: Vec<(String, AbstractCard)>,
    /// Which pruning rule set `abstract_legal_actions` applies.
    pub rules: TreeRules,
}

impl TraversalState {
    /// Create a traversal state from an abstract deal.
    /// Uses canonical concrete cards internally.
    pub fn from_deal(
        dealer: Player,
        score: Score,
        tc: TurnupClass,
        deal: &AbstractDeal,
    ) -> Result<Self, truco_engine::EngineError> {
        Self::from_deal_with_rules(dealer, score, tc, deal, TreeRules::Current)
    }

    /// Create a traversal state using an explicit pruning rule set.
    pub fn from_deal_with_rules(
        dealer: Player,
        score: Score,
        tc: TurnupClass,
        deal: &AbstractDeal,
        rules: TreeRules,
    ) -> Result<Self, truco_engine::EngineError> {
        let (turnup, hands, card_map) = realize_deal(tc, deal)?;
        let engine = Engine::new_hand(dealer, score, turnup, hands)?;
        Ok(Self {
            engine,
            turnup_class: tc,
            starting_hands: deal.hands.clone(),
            histories: [ActionHistory::new(), ActionHistory::new()],
            card_map,
            rules,
        })
    }

    /// Get the abstract actions available to the current player.
    pub fn abstract_legal_actions(&self) -> Result<Vec<AbstractAction>, truco_engine::EngineError> {
        let player = match self.engine.current_player() {
            Some(p) => p,
            None => return Ok(vec![]),
        };
        let concrete_actions = self.engine.strategic_legal_actions(player)?;
        let mut abstract_actions = Vec::new();
        let mut seen = Vec::new();

        for action in &concrete_actions {
            let abs = self.to_abstract_action(action);
            if !seen.contains(&abs) {
                seen.push(abs);
                abstract_actions.push(abs);
            }
        }

        // Score-aware raise pruning. Once the stake currently on the table already
        // makes the hand match-deciding for BOTH players — i.e.
        // `min(score) + on_table_stake >= MATCH_TARGET`, so even the lower-scored
        // player reaches the target by winning at that stake — escalating further
        // is a dominated, value-equivalent action: it adds no prize (you can't win
        // more than the match) and carries no fold leverage (folding the higher
        // raise also concedes the match, so a rational opponent never folds it).
        // Removing these raises shrinks the tree without changing the game value
        // or the equilibrium. Example: at 10x10 truco (stake 3) already decides the
        // match, so seis/nove/doze are pruned and the ladder collapses to {1, 3}.
        let st = self.engine.state();
        let on_table = match &st.pending_raise {
            Some(pending) => pending.to,
            None => st.hand_value,
        };
        let min_score = st.score.zero.min(st.score.one);
        // The deployed rule gates on min(score); the experimental asymmetric
        // variant gates on the acting player's own score (>= min), pruning the
        // higher-scored player's dominated raises in lopsided states. See the
        // value-preservation proof on `TreeRules::AsymmetricRaisePrune`.
        let raise_prune_score = if self.rules == TreeRules::AsymmetricRaisePrune {
            if player == 0 {
                st.score.zero
            } else {
                st.score.one
            }
        } else {
            min_score
        };
        if raise_prune_score + on_table >= truco_engine::MATCH_TARGET {
            abstract_actions.retain(|a| !matches!(a, AbstractAction::Raise(_)));
        }

        // The proof-scoped prunes below postdate 2026-07-16. Legacy-tree
        // traversals stop here so pre-prune artifacts can be evaluated on the
        // tree they were solved for.
        if self.rules == TreeRules::LegacyPreProofPrunes {
            return Ok(abstract_actions);
        }

        // Proof-scoped hidden-card dominance. A face-up play of the SAME card
        // directly weakly dominates face-down when responding in rounds 2 or
        // 3: playing the card resolves the round, so there is no later response
        // whose strategy can depend on whether the card was revealed. If both
        // versions lose round 2, its leader either won round 1 or round 1 tied,
        // so that loss also ends the hand and concealment has no future value.
        //
        // A round-3 LEADER is different: revealing the card can change the
        // responder's raise decision, so range/signalling effects make a broad
        // prune unsafe. Only when the responder cannot raise do we first remove
        // its directly-dominated hide, leaving one forced face-up response;
        // leader-hide is then weakly dominated too. This covers mão de onze,
        // a maxed ladder, a responder that made the last raise, and the solver's
        // existing score-aware match-deciding-stake prune.
        //
        // This deliberately does not impose blanket "never hide during mão de
        // onze": concealing which card remains can matter when leading round 2.
        let round_index = st.completed_rounds.len();
        let responding_in_round = !st.current_round.plays.is_empty();
        let responder = 1 - player;
        let score_disables_raises = st.score.zero == 11 || st.score.one == 11;
        let responder_can_raise_after_play = !score_disables_raises
            && st.hand_value < truco_engine::MATCH_TARGET
            && st.last_raised_by != Some(responder)
            && min_score + st.hand_value < truco_engine::MATCH_TARGET;
        let responding_after_round_one = round_index >= 1 && responding_in_round;
        let final_lead_without_raise_response =
            round_index >= 2 && !responding_in_round && !responder_can_raise_after_play;
        if responding_after_round_one || final_lead_without_raise_response {
            abstract_actions.retain(|a| !matches!(a, AbstractAction::PlayFaceDown(_)));
        }

        // Forced final-round fold certificate. Suppose the player responding
        // to a raise won round 2 but lost round 1, led the final round, and has
        // already played either face-down or the globally weakest face-up card.
        // After accepting, the raiser is the terminal card responder. Its hide
        // is directly dominated above, and every face-up card ties or beats a
        // Plain(0); a final tie awards the hand to the round-1 winner (the
        // raiser). Accept therefore guarantees the same hand loss at the NEW
        // stake, while Fold concedes only the smaller previous stake. Re-raise
        // remains live where legal because bluff/fold equity is strategic.
        let forced_final_loss_after_accept = st.pending_raise.is_some()
            && round_index == 2
            && st.current_round.plays.len() == 1
            && st.current_round.plays[0].player == player
            && st.completed_rounds[0].winner == Some(responder)
            && (st.current_round.plays[0].visibility == Visibility::Down
                || self.lookup_card(&st.current_round.plays[0].card.id) == AbstractCard::Plain(0));
        if forced_final_loss_after_accept {
            abstract_actions.retain(|a| !matches!(a, AbstractAction::AcceptRaise));
        }

        Ok(abstract_actions)
    }

    /// Apply an abstract action, returning the new state.
    pub fn apply_abstract_action(
        &self,
        abs_action: AbstractAction,
    ) -> Result<Self, truco_engine::EngineError> {
        let player = self
            .engine
            .current_player()
            .ok_or(truco_engine::EngineError::HandAlreadyDecided)?;
        let concrete = self
            .to_concrete_action(player, abs_action)
            .ok_or(truco_engine::EngineError::CardNotInHand)?;
        let mut new_state = self.clone();
        new_state.engine.apply_action(player, &concrete)?;

        let opponent = 1 - player;
        new_state.histories[player as usize].push(abs_action);
        new_state.histories[opponent as usize].push(match abs_action {
            AbstractAction::PlayFaceDown(_) => AbstractAction::OpponentPlayedHidden,
            other => other,
        });

        Ok(new_state)
    }

    /// Build the info set for the current player.
    pub fn current_info_set(&self) -> Option<InfoSet> {
        let player = self.engine.current_player()?;
        Some(InfoSet {
            player,
            is_dealer: player == self.engine.state().dealer,
            turnup_class: self.turnup_class,
            starting_hand: self.starting_hands[player as usize].clone(),
            history: self.histories[player as usize].clone(),
        })
    }

    pub fn is_terminal(&self) -> bool {
        self.engine.is_hand_over()
    }

    pub fn hand_winner(&self) -> Option<Player> {
        self.engine.hand_winner()
    }

    pub fn hand_value(&self) -> u8 {
        self.engine.state().hand_value
    }

    pub fn score(&self) -> &Score {
        &self.engine.state().score
    }

    fn to_abstract_action(&self, action: &Action) -> AbstractAction {
        match action {
            Action::PlayFaceUp { card_id } => AbstractAction::PlayFaceUp(self.lookup_card(card_id)),
            Action::PlayFaceDown { card_id } => {
                AbstractAction::PlayFaceDown(self.lookup_card(card_id))
            }
            Action::Raise { to } => AbstractAction::Raise(*to),
            Action::AcceptRaise => AbstractAction::AcceptRaise,
            Action::Fold => AbstractAction::Fold,
            Action::AcceptEleven => AbstractAction::AcceptEleven,
            Action::FoldEleven => AbstractAction::FoldEleven,
            Action::ConcedeHand => {
                unreachable!("concede_hand is intentionally excluded from solver action space")
            }
        }
    }

    fn to_concrete_action(&self, player: Player, abs_action: AbstractAction) -> Option<Action> {
        match abs_action {
            AbstractAction::PlayFaceUp(card) => {
                let card_id = self.find_card_in_hand(player, card)?;
                Some(Action::PlayFaceUp { card_id })
            }
            AbstractAction::PlayFaceDown(card) => {
                let card_id = self.find_card_in_hand(player, card)?;
                Some(Action::PlayFaceDown { card_id })
            }
            AbstractAction::Raise(to) => Some(Action::Raise { to }),
            AbstractAction::AcceptRaise => Some(Action::AcceptRaise),
            AbstractAction::Fold => Some(Action::Fold),
            AbstractAction::AcceptEleven => Some(Action::AcceptEleven),
            AbstractAction::FoldEleven => Some(Action::FoldEleven),
            AbstractAction::OpponentPlayedHidden => {
                unreachable!("OpponentPlayedHidden is not a real action")
            }
        }
    }

    fn lookup_card(&self, card_id: &str) -> AbstractCard {
        self.card_map
            .iter()
            .find(|(id, _)| id == card_id)
            .map(|(_, ac)| *ac)
            .expect("card_id not in card_map")
    }

    fn find_card_in_hand(
        &self,
        player: Player,
        target: AbstractCard,
    ) -> Option<std::sync::Arc<str>> {
        let hand = self.engine.state().hands.player(player);
        for card in hand {
            if self.lookup_card(&card.id) == target {
                return Some(card.id.clone());
            }
        }
        None
    }
}

// ─── Pre-built game tree for fast CFR iteration ─────────────────────────

/// Index into the node arena.
pub type NodeId = u32;

/// 8-byte-aligned shared byte buffer backing tree arenas: either built in
/// memory or memory-mapped from a treepack file. The mmap path is the
/// STREAMING path — the OS pages a larger-than-RAM arena in and out at SSD
/// speed, so deep-state trees no longer need to fit in memory; and the same
/// file is the GCS-cacheable artifact (trees are identical for every score
/// state within a ladder band, so one artifact serves a whole band).
#[derive(Clone)]
pub enum TreeBytes {
    Owned(std::sync::Arc<Vec<u64>>),
    Mapped(std::sync::Arc<memmap2::Mmap>),
}

impl TreeBytes {
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            TreeBytes::Owned(v) => bytes_of_u64s(v),
            TreeBytes::Mapped(m) => &m[..],
        }
    }
}

#[inline]
fn bytes_of_u64s(v: &[u64]) -> &[u8] {
    // SAFETY: u64 -> u8 reinterpretation is always valid; length in bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Typed window into a [`TreeBytes`] buffer. Deref to `[PackedNode]` /
/// `[PackedEdge]` keeps every traversal call-site unchanged.
#[derive(Clone)]
pub struct NodeSlice {
    buf: TreeBytes,
    byte_off: usize,
    count: usize,
}

#[derive(Clone)]
pub struct EdgeSlice {
    buf: TreeBytes,
    byte_off: usize,
    count: usize,
}

impl std::ops::Deref for NodeSlice {
    type Target = [PackedNode];
    #[inline]
    fn deref(&self) -> &[PackedNode] {
        let b = &self.buf.as_bytes()[self.byte_off..];
        debug_assert!(b.as_ptr() as usize % std::mem::align_of::<PackedNode>() == 0);
        debug_assert!(b.len() >= self.count * std::mem::size_of::<PackedNode>());
        // SAFETY: buffer outlives self via Arc; offset is align-of-PackedNode
        // aligned by construction (base is 8-aligned, offsets are multiples of
        // the struct sizes); PackedNode is repr(C) with every bit pattern valid.
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const PackedNode, self.count) }
    }
}

impl std::ops::Deref for EdgeSlice {
    type Target = [PackedEdge];
    #[inline]
    fn deref(&self) -> &[PackedEdge] {
        let b = &self.buf.as_bytes()[self.byte_off..];
        debug_assert!(b.as_ptr() as usize % std::mem::align_of::<PackedEdge>() == 0);
        debug_assert!(b.len() >= self.count * std::mem::size_of::<PackedEdge>());
        // SAFETY: as above; PackedEdge is repr(C), all bit patterns valid.
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const PackedEdge, self.count) }
    }
}

/// A compact, pre-built game tree: a cheap HANDLE (Arc'd typed windows) into
/// the build's shared flat arena. 12-byte packed nodes plus an 8-byte edge
/// array, ~30 B/node effective and cache-linear; node ids and edge offsets are
/// tree-relative, so a handle behaves exactly like the old owned struct.
#[derive(Clone)]
pub struct GameTree {
    pub nodes: NodeSlice,
    pub edges: EdgeSlice,
}

/// 12-byte packed node. `n_actions == 0` marks a terminal (payoff in
/// `payoff`, an i8 — hand values are ±1/3/6/9/12); otherwise a player node
/// whose edges live at `edges[edge_off .. edge_off + n_actions]` and whose
/// info set is `PrebuiltTrees::info_sets[table_idx]` (the dense accumulator
/// index — the key itself is recoverable from that entry).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PackedNode {
    pub table_idx: u32,
    pub edge_off: u32,
    pub n_actions: u8,
    pub player: u8,
    pub payoff: i8,
    _pad: u8,
}

/// An action edge: child id + u8-encoded action (see `AbstractAction::to_u8`).
/// repr(C) with explicit padding so the arena bytes are a stable, mmap-safe
/// on-disk format.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PackedEdge {
    pub child: NodeId,
    pub action_u8: u8,
    _pad: [u8; 3],
}

impl PackedEdge {
    #[inline]
    pub fn new(action_u8: u8, child: NodeId) -> Self {
        Self {
            child,
            action_u8,
            _pad: [0; 3],
        }
    }
}

impl PackedEdge {
    #[inline]
    pub fn action(&self) -> AbstractAction {
        AbstractAction::from_u8(self.action_u8)
    }
}

/// Borrowed view of a node, shaped like the old enum so traversals stay
/// pattern-match-based.
pub enum NodeView<'a> {
    Terminal {
        payoff_p0: f64,
    },
    Player {
        player: Player,
        table_idx: u32,
        edges: &'a [PackedEdge],
    },
}

impl GameTree {
    /// Empty handle (used for the dealer excluded by a dealer filter).
    pub fn empty() -> Self {
        static EMPTY: std::sync::OnceLock<std::sync::Arc<Vec<u64>>> = std::sync::OnceLock::new();
        let buf = TreeBytes::Owned(
            EMPTY
                .get_or_init(|| std::sync::Arc::new(Vec::new()))
                .clone(),
        );
        GameTree {
            nodes: NodeSlice {
                buf: buf.clone(),
                byte_off: 0,
                count: 0,
            },
            edges: EdgeSlice {
                buf,
                byte_off: 0,
                count: 0,
            },
        }
    }

    /// Handle over a span of shared arenas. Offsets/counts are in ELEMENTS
    /// relative to the arena start; arena byte bases must be 8-aligned
    /// (guaranteed for both `Vec<u64>` and mmap backings).
    #[allow(clippy::too_many_arguments)]
    pub fn from_arena(
        nodes_buf: &TreeBytes,
        edges_buf: &TreeBytes,
        nodes_base_bytes: usize,
        edges_base_bytes: usize,
        node_off: usize,
        node_count: usize,
        edge_off: usize,
        edge_count: usize,
    ) -> Self {
        GameTree {
            nodes: NodeSlice {
                buf: nodes_buf.clone(),
                byte_off: nodes_base_bytes + node_off * std::mem::size_of::<PackedNode>(),
                count: node_count,
            },
            edges: EdgeSlice {
                buf: edges_buf.clone(),
                byte_off: edges_base_bytes + edge_off * std::mem::size_of::<PackedEdge>(),
                count: edge_count,
            },
        }
    }

    #[inline]
    pub fn view(&self, id: NodeId) -> NodeView<'_> {
        let n = &self.nodes[id as usize];
        if n.n_actions == 0 {
            NodeView::Terminal {
                payoff_p0: n.payoff as f64,
            }
        } else {
            NodeView::Player {
                player: n.player,
                table_idx: n.table_idx,
                edges: &self.edges[n.edge_off as usize..n.edge_off as usize + n.n_actions as usize],
            }
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Assigns dense indices to info sets in first-encounter order and records
/// their metadata. `entries[idx]` is the info set with `table_idx == idx`.
#[derive(Default)]
pub struct InfoSetRegistry {
    pub index_of: std::collections::HashMap<InfoSetKey, u32>,
    pub entries: Vec<(InfoSetKey, InfoSet, InfoActions)>,
}

impl InfoSetRegistry {
    fn register(&mut self, key: InfoSetKey, info_set: InfoSet, actions: &[AbstractAction]) -> u32 {
        match self.index_of.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let idx = self.entries.len() as u32;
                e.insert(idx);
                self.entries
                    .push((key, info_set, actions.to_vec().into_boxed_slice()));
                idx
            }
        }
    }
}

/// Build-time scratch for one deal's tree (plain Vecs; copied into the shared
/// arena afterwards).
#[derive(Default)]
struct TreeScratch {
    nodes: Vec<PackedNode>,
    edges: Vec<PackedEdge>,
}

/// Growable byte arena (8-aligned via u64 backing) that trees are packed into.
#[derive(Default)]
pub struct ArenaBuilder {
    words: Vec<u64>,
    bytes_used: usize,
}

impl ArenaBuilder {
    fn append<T: Copy>(&mut self, items: &[T]) -> usize {
        let elem_off = self.bytes_used / std::mem::size_of::<T>();
        debug_assert_eq!(self.bytes_used % std::mem::size_of::<T>(), 0);
        let add_bytes = std::mem::size_of_val(items);
        let need_words = (self.bytes_used + add_bytes).div_ceil(8);
        self.words.resize(need_words, 0);
        // SAFETY: destination is within the freshly resized words buffer;
        // T is one of the repr(C), all-bit-patterns-valid packed structs.
        unsafe {
            let dst = (self.words.as_mut_ptr() as *mut u8).add(self.bytes_used);
            std::ptr::copy_nonoverlapping(items.as_ptr() as *const u8, dst, add_bytes);
        }
        self.bytes_used += add_bytes;
        elem_off
    }

    fn finish(self) -> TreeBytes {
        TreeBytes::Owned(std::sync::Arc::new(self.words))
    }
}

/// Build a complete game tree for a single deal. `table_idx` values are
/// assigned by a build-local registry — meaningful only relative to trees
/// built with the same registry (use [`build_all_trees`] for cross-deal
/// consistency).
pub fn build_game_tree(state: &TraversalState) -> Result<GameTree, truco_engine::EngineError> {
    let mut scratch = TreeScratch::default();
    let mut registry = InfoSetRegistry::default();
    build_node(state, &mut scratch, &mut registry)?;
    let mut nodes_arena = ArenaBuilder::default();
    let node_off = nodes_arena.append(&scratch.nodes);
    let mut edges_arena = ArenaBuilder::default();
    let edge_off = edges_arena.append(&scratch.edges);
    let (nc, ec) = (scratch.nodes.len(), scratch.edges.len());
    let nbuf = nodes_arena.finish();
    let ebuf = edges_arena.finish();
    Ok(GameTree {
        nodes: NodeSlice {
            buf: nbuf,
            byte_off: node_off * std::mem::size_of::<PackedNode>(),
            count: nc,
        },
        edges: EdgeSlice {
            buf: ebuf,
            byte_off: edge_off * std::mem::size_of::<PackedEdge>(),
            count: ec,
        },
    })
}

fn build_node(
    state: &TraversalState,
    tree: &mut TreeScratch,
    registry: &mut InfoSetRegistry,
) -> Result<NodeId, truco_engine::EngineError> {
    if state.is_terminal() {
        debug_assert!(
            state.hand_winner().is_some(),
            "terminal state must have a winner"
        );
        let winner = state
            .hand_winner()
            .expect("terminal state must have a winner");
        let value = state.hand_value() as i8;
        let payoff_p0 = if winner == 0 { value } else { -value };
        let id = tree.nodes.len() as NodeId;
        tree.nodes.push(PackedNode {
            table_idx: 0,
            edge_off: 0,
            n_actions: 0,
            player: 0,
            payoff: payoff_p0,
            _pad: 0,
        });
        return Ok(id);
    }

    let actions = state.abstract_legal_actions()?;
    // A non-terminal state must offer actions. A silent zero-payoff terminal
    // here would be interpreted downstream as "hand ended, nobody scored" and
    // look up a continuation at the SAME score — quiet value corruption.
    assert!(
        !actions.is_empty(),
        "non-terminal state with no legal actions"
    );

    debug_assert!(
        state.current_info_set().is_some(),
        "non-terminal state with actions must have an info set"
    );
    debug_assert!(
        state.engine.current_player().is_some(),
        "non-terminal state with actions must have a current player"
    );
    let info_set = state
        .current_info_set()
        .expect("non-terminal state with actions must have an info set");
    let player = state
        .engine
        .current_player()
        .expect("non-terminal state with actions must have a current player");
    let info_set_key = info_set.key();
    let table_idx = registry.register(info_set_key, info_set, &actions);

    // Reserve the node slot and a CONTIGUOUS edge block before recursing
    // (children append their own edges after ours).
    let id = tree.nodes.len() as NodeId;
    let edge_off = tree.edges.len();
    tree.nodes.push(PackedNode {
        table_idx,
        edge_off: edge_off as u32,
        n_actions: actions.len() as u8,
        player,
        payoff: 0,
        _pad: 0,
    });
    for &action in &actions {
        tree.edges.push(PackedEdge::new(action.to_u8(), 0)); // child filled below
    }

    for (i, &action) in actions.iter().enumerate() {
        let child_state = state.apply_abstract_action(action)?;
        let child_id = build_node(&child_state, tree, registry)?;
        tree.edges[edge_off + i].child = child_id;
    }

    Ok(id)
}

/// Result of a size-only tree walk: how big a (score, tc, dealer) subgame is,
/// without paying for any solver-ready allocations (no `PackedNode`/`PackedEdge`
/// arena, no `InfoSetRegistry`, no regret/strategy tables).
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeSizeCount {
    pub total_nodes: u64,
    pub num_info_sets: usize,
}

/// Which closure of a fixed policy to count without materializing a solver tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyTreeMode {
    /// Follow supported actions for both players.
    Profile,
    /// Enumerate every action for this player, but only supported actions for
    /// the fixed opponent. This is the tree an exact unilateral best response
    /// to the policy can reach.
    BestResponse(Player),
    /// Union of both unilateral best-response closures, counted without
    /// double-counting paths that occur in both.
    BestResponseUnion,
}

impl PolicyTreeMode {
    fn initial_mask(self) -> u8 {
        match self {
            PolicyTreeMode::Profile | PolicyTreeMode::BestResponse(_) => 1,
            PolicyTreeMode::BestResponseUnion => 0b11,
        }
    }

    fn child_mask(self, active_mask: u8, player: Player, supported: bool) -> u8 {
        match self {
            PolicyTreeMode::Profile => u8::from(supported),
            PolicyTreeMode::BestResponse(responder) => u8::from(player == responder || supported),
            PolicyTreeMode::BestResponseUnion => {
                let mut next = 0;
                for responder in 0..2u8 {
                    let bit = 1 << responder;
                    if active_mask & bit != 0 && (player == responder || supported) {
                        next |= bit;
                    }
                }
                next
            }
        }
    }
}

/// Which saved-policy values define support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyValueSource {
    /// The linearly weighted average strategy used for solution/export/eval.
    Average,
    /// The final regret-matched strategy. Requires a full checkpoint because a
    /// strategy-only artifact intentionally does not preserve regrets.
    Current,
}

/// Minimal policy lookup needed by the DFS census. Implementations may be a
/// full resumable [`StrategyTable`] or a compact, average-only streaming load
/// that discards solve metadata after hashing each serialized info set.
pub trait PolicyLookup {
    fn action_probability(
        &self,
        key: InfoSetKey,
        action: AbstractAction,
        values: PolicyValueSource,
    ) -> Option<f64>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PolicyLookup for StrategyTable {
    fn action_probability(
        &self,
        key: InfoSetKey,
        action: AbstractAction,
        values: PolicyValueSource,
    ) -> Option<f64> {
        let data = self.data.get(&key)?;
        let i = data.actions.iter().position(|&saved| saved == action)?;
        match values {
            PolicyValueSource::Average => data.average_strategy().get(i).copied(),
            PolicyValueSource::Current => data.current_strategy().get(i).copied(),
        }
    }

    fn len(&self) -> usize {
        StrategyTable::len(self)
    }
}

/// Explicit support used when a projected policy has no entry/action in a
/// deeper band. There is intentionally no silent default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingPolicyFallback {
    /// Treat every legal action as supported.
    All,
    /// Treat only the first legal action as supported.
    First,
    /// Keep every non-raise action, but do not seed a previously unavailable
    /// deeper raise. This is a support skeleton, not a complete equilibrium.
    AllExceptRaise,
}

/// Size and policy-coverage diagnostics for one policy-aware DFS census.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyTreeSizeCount {
    pub total_nodes: u64,
    pub num_info_sets: usize,
    /// Distinct reached info sets whose entry was absent or whose legal action
    /// set contained an action absent from the source policy.
    pub policy_missing_info_sets: usize,
    /// Reached decision-node action edges before/after the selected closure
    /// rule. These count repeated occurrences of an information set across
    /// deals, matching the node/edge arena cost rather than table slots.
    pub legal_actions: u64,
    pub kept_actions: u64,
    pub legal_raises: u64,
    pub kept_raises: u64,
}

/// Count the union of three paths through an already-built full tree:
///
/// 1. the fixed profile's supported actions;
/// 2. player 0's single exact best-response action at their own info sets,
///    with player 1 fixed to the profile; and
/// 3. player 1's corresponding exact best response.
///
/// Unlike [`PolicyTreeMode::BestResponseUnion`], this does not retain every
/// responder action. The best-response choices must come from the exact
/// backward-induction pass over this same `PrebuiltTrees`, so the count is a
/// structural decision gate rather than an approximate safety claim.
pub fn count_chosen_best_response_union(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
) -> PolicyTreeSizeCount {
    assert_eq!(strategies.len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[0].len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[1].len(), prebuilt.info_sets.len());
    assert!(support_threshold.is_finite() && support_threshold >= 0.0);

    let mut seen = vec![false; prebuilt.info_sets.len()];
    let mut count = PolicyTreeSizeCount::default();
    for entry in &prebuilt.entries {
        for tree in [&entry.tree_dealer_0, &entry.tree_dealer_1] {
            if !tree.is_empty() {
                count_chosen_union_node(
                    tree,
                    0,
                    0b111,
                    strategies,
                    chosen_actions,
                    support_threshold,
                    &mut seen,
                    &mut count,
                );
            }
        }
    }
    count.num_info_sets = seen.into_iter().filter(|reached| *reached).count();
    count
}

/// Count the actual first restricted-game arena induced by retaining, at each
/// reached information set, every supported profile action plus that acting
/// player's exact chosen best-response action.
///
/// This is generally larger than [`count_chosen_best_response_union`]: a
/// re-solve may combine retained actions from different profile/BR paths, so
/// its tree contains their cross-product rather than only three fixed paths.
pub fn count_chosen_best_response_closure(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
) -> PolicyTreeSizeCount {
    assert_eq!(strategies.len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[0].len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[1].len(), prebuilt.info_sets.len());
    assert!(support_threshold.is_finite() && support_threshold >= 0.0);

    let mut seen = vec![false; prebuilt.info_sets.len()];
    let mut count = PolicyTreeSizeCount::default();
    for entry in &prebuilt.entries {
        for tree in [&entry.tree_dealer_0, &entry.tree_dealer_1] {
            if !tree.is_empty() {
                count_chosen_closure_node(
                    tree,
                    0,
                    strategies,
                    chosen_actions,
                    support_threshold,
                    &mut seen,
                    &mut count,
                );
            }
        }
    }
    count.num_info_sets = seen.into_iter().filter(|reached| *reached).count();
    count
}

fn count_chosen_closure_node(
    tree: &GameTree,
    node_id: NodeId,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
    seen: &mut [bool],
    count: &mut PolicyTreeSizeCount,
) {
    count.total_nodes += 1;
    let NodeView::Player {
        player,
        table_idx,
        edges,
    } = tree.view(node_id)
    else {
        return;
    };

    let idx = table_idx as usize;
    seen[idx] = true;
    let probabilities = &strategies[idx];
    assert_eq!(probabilities.len(), edges.len());
    let best_profile_action = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("player node must have actions");
    let chosen = chosen_actions[player as usize][idx] as usize;
    assert!(chosen < edges.len(), "responder choice must be legal");

    count.legal_actions += edges.len() as u64;
    count.legal_raises += edges
        .iter()
        .filter(|edge| matches!(edge.action(), AbstractAction::Raise(_)))
        .count() as u64;
    for (i, edge) in edges.iter().enumerate() {
        let supported = probabilities[i] > support_threshold || i == best_profile_action;
        if !supported && i != chosen {
            continue;
        }
        count.kept_actions += 1;
        if matches!(edge.action(), AbstractAction::Raise(_)) {
            count.kept_raises += 1;
        }
        count_chosen_closure_node(
            tree,
            edge.child,
            strategies,
            chosen_actions,
            support_threshold,
            seen,
            count,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn count_chosen_union_node(
    tree: &GameTree,
    node_id: NodeId,
    active_mask: u8,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
    seen: &mut [bool],
    count: &mut PolicyTreeSizeCount,
) {
    debug_assert_ne!(active_mask, 0);
    count.total_nodes += 1;
    let NodeView::Player {
        player,
        table_idx,
        edges,
    } = tree.view(node_id)
    else {
        return;
    };

    let idx = table_idx as usize;
    seen[idx] = true;
    let probabilities = &strategies[idx];
    assert_eq!(probabilities.len(), edges.len());
    let best_profile_action = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("player node must have actions");

    count.legal_actions += edges.len() as u64;
    count.legal_raises += edges
        .iter()
        .filter(|edge| matches!(edge.action(), AbstractAction::Raise(_)))
        .count() as u64;

    for (i, edge) in edges.iter().enumerate() {
        let supported = probabilities[i] > support_threshold || i == best_profile_action;
        let mut child_mask = u8::from(active_mask & 0b001 != 0 && supported);
        for responder in 0..2usize {
            let bit = 1 << (responder + 1);
            if active_mask & bit == 0 {
                continue;
            }
            let keep = if player as usize == responder {
                chosen_actions[responder][idx] as usize == i
            } else {
                supported
            };
            if keep {
                child_mask |= bit;
            }
        }
        if child_mask == 0 {
            continue;
        }
        count.kept_actions += 1;
        if matches!(edge.action(), AbstractAction::Raise(_)) {
            count.kept_raises += 1;
        }
        count_chosen_union_node(
            tree,
            edge.child,
            child_mask,
            strategies,
            chosen_actions,
            support_threshold,
            seen,
            count,
        );
    }
}

/// `(deals_done, deals_total, running_count)`, invoked after each deal.
pub type PolicyTreeSizeProgressCb<'a> = &'a mut dyn FnMut(usize, usize, &PolicyTreeSizeCount);

/// `(deals_done, deals_total, running_count)`, invoked after each deal.
pub type TreeSizeProgressCb<'a> = &'a mut dyn FnMut(usize, usize, &TreeSizeCount);

/// Walk every deal's tree purely to count nodes and distinct info sets,
/// tracking only a `HashSet<u64>` of seen info-set keys (~8-24 bytes/entry)
/// instead of the full per-info-set solver state `build_node` accumulates
/// (~1-1.5KB/entry once regret/strategy arrays are attached downstream — see
/// `RESEARCH_NARRATIVE.md`'s tree-size-survey entry). Recursion depth is
/// bounded by a single hand's length (a handful of rounds), so stack use is
/// negligible regardless of how many distinct info sets exist in total.
pub fn count_tree_size(
    score: &Score,
    tc: TurnupClass,
    dealer: Player,
    deals: &[AbstractDeal],
    progress: Option<TreeSizeProgressCb<'_>>,
) -> Result<TreeSizeCount, truco_engine::EngineError> {
    count_tree_size_with_rules(score, tc, dealer, deals, TreeRules::Current, progress)
}

/// Like [`count_tree_size`] but with an explicit pruning rule set, so tree-size
/// A/Bs (e.g. asymmetric raise pruning) can be measured without a solve.
pub fn count_tree_size_with_rules(
    score: &Score,
    tc: TurnupClass,
    dealer: Player,
    deals: &[AbstractDeal],
    rules: TreeRules,
    mut progress: Option<TreeSizeProgressCb<'_>>,
) -> Result<TreeSizeCount, truco_engine::EngineError> {
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut count = TreeSizeCount::default();
    for (i, deal) in deals.iter().enumerate() {
        let state = TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
        count_node_size(&state, &mut seen, &mut count.total_nodes)?;
        count.num_info_sets = seen.len();
        if let Some(cb) = progress.as_deref_mut() {
            cb(i + 1, deals.len(), &count);
        }
    }
    Ok(count)
}

fn count_node_size(
    state: &TraversalState,
    seen: &mut std::collections::HashSet<u64>,
    total_nodes: &mut u64,
) -> Result<(), truco_engine::EngineError> {
    *total_nodes += 1;
    if state.is_terminal() {
        return Ok(());
    }
    let actions = state.abstract_legal_actions()?;
    assert!(
        !actions.is_empty(),
        "non-terminal state with no legal actions"
    );
    if let Some(info_set) = state.current_info_set() {
        seen.insert(info_set.key().0);
    }
    for &action in &actions {
        let child_state = state.apply_abstract_action(action)?;
        count_node_size(&child_state, seen, total_nodes)?;
    }
    Ok(())
}

/// Count the part of a game tree needed by a fixed policy profile or by one or
/// both unilateral best responses to it. Like [`count_tree_size`], this is a
/// space-for-time DFS: it never builds a node arena or regret table.
///
/// `support_threshold` keeps actions with probability strictly greater than
/// the threshold. The maximum-probability known action is always retained so a
/// threshold cannot accidentally turn a valid policy into an empty one.
pub fn count_policy_tree_size(
    score: &Score,
    tc: TurnupClass,
    dealer: Player,
    deals: &[AbstractDeal],
    policy: &dyn PolicyLookup,
    mode: PolicyTreeMode,
    values: PolicyValueSource,
    support_threshold: f64,
    missing_fallback: MissingPolicyFallback,
    mut progress: Option<PolicyTreeSizeProgressCb<'_>>,
) -> Result<PolicyTreeSizeCount, truco_engine::EngineError> {
    assert!(
        support_threshold.is_finite() && support_threshold >= 0.0,
        "support threshold must be finite and >= 0"
    );
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut count = PolicyTreeSizeCount::default();
    for (i, deal) in deals.iter().enumerate() {
        let state = TraversalState::from_deal(dealer, score.clone(), tc, deal)?;
        count_policy_node_size(
            &state,
            policy,
            mode,
            values,
            support_threshold,
            missing_fallback,
            mode.initial_mask(),
            &mut seen,
            &mut count,
        )?;
        count.num_info_sets = seen.len();
        if let Some(cb) = progress.as_deref_mut() {
            cb(i + 1, deals.len(), &count);
        }
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn count_policy_node_size(
    state: &TraversalState,
    policy: &dyn PolicyLookup,
    mode: PolicyTreeMode,
    values: PolicyValueSource,
    support_threshold: f64,
    missing_fallback: MissingPolicyFallback,
    active_mask: u8,
    seen: &mut std::collections::HashSet<u64>,
    count: &mut PolicyTreeSizeCount,
) -> Result<(), truco_engine::EngineError> {
    debug_assert_ne!(active_mask, 0);
    count.total_nodes += 1;
    if state.is_terminal() {
        return Ok(());
    }

    let actions = state.abstract_legal_actions()?;
    assert!(
        !actions.is_empty(),
        "non-terminal state with no legal actions"
    );
    let info_set = state
        .current_info_set()
        .expect("non-terminal state with actions must have an info set");
    let player = info_set.player;
    let key = info_set.key();
    let (supported, incomplete) = policy_support(
        policy,
        key,
        &actions,
        values,
        support_threshold,
        missing_fallback,
    );

    let is_new_info = seen.insert(key.0);
    if is_new_info && incomplete {
        count.policy_missing_info_sets += 1;
    }
    count.legal_actions += actions.len() as u64;
    count.legal_raises += actions
        .iter()
        .filter(|a| matches!(a, AbstractAction::Raise(_)))
        .count() as u64;

    for (i, &action) in actions.iter().enumerate() {
        let child_mask = mode.child_mask(active_mask, player, supported[i]);
        if child_mask == 0 {
            continue;
        }
        count.kept_actions += 1;
        if matches!(action, AbstractAction::Raise(_)) {
            count.kept_raises += 1;
        }
        let child_state = state.apply_abstract_action(action)?;
        count_policy_node_size(
            &child_state,
            policy,
            mode,
            values,
            support_threshold,
            missing_fallback,
            child_mask,
            seen,
            count,
        )?;
    }
    Ok(())
}

fn policy_support(
    policy: &dyn PolicyLookup,
    key: InfoSetKey,
    legal_actions: &[AbstractAction],
    values: PolicyValueSource,
    support_threshold: f64,
    missing_fallback: MissingPolicyFallback,
) -> (Vec<bool>, bool) {
    let fallback = |i: usize, action: AbstractAction| match missing_fallback {
        MissingPolicyFallback::All => true,
        MissingPolicyFallback::First => i == 0,
        MissingPolicyFallback::AllExceptRaise => !matches!(action, AbstractAction::Raise(_)),
    };

    let mapped: Vec<Option<f64>> = legal_actions
        .iter()
        .copied()
        .map(|action| policy.action_probability(key, action, values))
        .collect();
    if mapped.iter().all(Option::is_none) {
        let mut support: Vec<bool> = legal_actions
            .iter()
            .copied()
            .enumerate()
            .map(|(i, action)| fallback(i, action))
            .collect();
        if !support.iter().any(|&keep| keep) && !support.is_empty() {
            support[0] = true;
        }
        return (support, true);
    }
    let incomplete = mapped.iter().any(Option::is_none);

    let best_known = mapped
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.map(|p| (i, p)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    let mut support: Vec<bool> = mapped
        .iter()
        .enumerate()
        .map(|(i, p)| match p {
            Some(p) => *p > support_threshold || best_known == Some(i),
            None => fallback(i, legal_actions[i]),
        })
        .collect();
    if !support.iter().any(|&keep| keep) && !support.is_empty() {
        support[0] = true;
    }
    (support, incomplete)
}

/// Pre-built game tree for all deals in a turnup class + score combination.
pub struct PrebuiltTrees {
    /// Each entry: (deal_weight, game_tree_for_dealer_0, game_tree_for_dealer_1)
    pub entries: Vec<DealTrees>,
    /// All info sets encountered, with their (InfoSet, actions) for
    /// initialization — in `table_idx` order: `info_sets[i]` is the info set
    /// whose player nodes carry `table_idx == i`.
    pub info_sets: Vec<(InfoSetKey, InfoSet, InfoActions)>,
    /// Arena backing + span table (element offsets/counts per deal per dealer),
    /// retained so treepack export can write the arenas without re-walking the
    /// handles. `spans[i] = [(node_off, node_count, edge_off, edge_count); 2]`.
    pub nodes_buf: TreeBytes,
    pub edges_buf: TreeBytes,
    pub spans: Vec<[(u64, u64, u64, u64); 2]>,
}

impl std::fmt::Debug for PrebuiltTrees {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrebuiltTrees")
            .field("deals", &self.entries.len())
            .field("info_sets", &self.info_sets.len())
            .field("node_bytes", &self.nodes_buf.as_bytes().len())
            .field("edge_bytes", &self.edges_buf.as_bytes().len())
            .finish()
    }
}

pub struct DealTrees {
    pub weight: f64,
    pub tree_dealer_0: GameTree,
    pub tree_dealer_1: GameTree,
}

/// Build all game trees for a given score and turnup class.
pub fn build_all_trees(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
) -> Result<PrebuiltTrees, truco_engine::EngineError> {
    build_all_trees_with_dealer(score, tc, deals, None)
}

/// Materialize the local action-set closure measured by
/// [`count_chosen_best_response_closure`]. The returned tree is solver-ready:
/// table indices are compacted, and each info set's legal-action metadata is
/// filtered into exactly the same order as its retained edges.
///
/// This consumes no game-theoretic authority by itself. A caller must solve the
/// restricted game and audit the resulting profile against the full tree,
/// adding newly selected exact-BR actions until the requested global bound is
/// met.
pub fn restrict_prebuilt_to_chosen_closure(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
) -> PrebuiltTrees {
    let allowed =
        chosen_closure_action_masks(prebuilt, strategies, chosen_actions, support_threshold);
    restrict_prebuilt_to_action_masks(prebuilt, &allowed)
}

/// Full-tree-indexed action masks for the first chosen-BR closure. Keeping the
/// masks separate lets later double-oracle rounds union in newly selected BR
/// actions without dropping anything retained by an earlier round.
pub fn chosen_closure_action_masks(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    chosen_actions: [&[u8]; 2],
    support_threshold: f64,
) -> Vec<Box<[bool]>> {
    assert_eq!(strategies.len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[0].len(), prebuilt.info_sets.len());
    assert_eq!(chosen_actions[1].len(), prebuilt.info_sets.len());
    assert!(support_threshold.is_finite() && support_threshold >= 0.0);
    prebuilt
        .info_sets
        .iter()
        .enumerate()
        .map(|(idx, (_, info, actions))| {
            let probabilities = &strategies[idx];
            assert_eq!(probabilities.len(), actions.len());
            let best = probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("info set must have actions");
            let chosen = chosen_actions[info.player as usize][idx] as usize;
            assert!(chosen < actions.len(), "responder choice must be legal");
            (0..actions.len())
                .map(|i| probabilities[i] > support_threshold || i == best || i == chosen)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect()
}

/// Materialize any monotone full-tree-indexed action masks into a compact,
/// solver-ready arena. At least one action must be retained per info set that
/// becomes reachable.
pub fn restrict_prebuilt_to_action_masks(
    prebuilt: &PrebuiltTrees,
    allowed: &[Box<[bool]>],
) -> PrebuiltTrees {
    assert_eq!(allowed.len(), prebuilt.info_sets.len());
    for (idx, (_, _, actions)) in prebuilt.info_sets.iter().enumerate() {
        assert_eq!(allowed[idx].len(), actions.len());
        assert!(allowed[idx].iter().any(|keep| *keep));
    }

    let mut old_to_new: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut info_sets = Vec::new();
    let mut nodes_arena = ArenaBuilder::default();
    let mut edges_arena = ArenaBuilder::default();
    let mut spans = Vec::with_capacity(prebuilt.entries.len());
    let mut weights = Vec::with_capacity(prebuilt.entries.len());
    let mut scratch = TreeScratch::default();

    for entry in &prebuilt.entries {
        let mut span = [(0u64, 0u64, 0u64, 0u64); 2];
        for (dealer, tree) in [(0usize, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if tree.is_empty() {
                continue;
            }
            scratch.nodes.clear();
            scratch.edges.clear();
            copy_action_mask_node(
                tree,
                0,
                prebuilt,
                allowed,
                &mut old_to_new,
                &mut info_sets,
                &mut scratch,
            );
            let node_off = nodes_arena.append(&scratch.nodes);
            let edge_off = edges_arena.append(&scratch.edges);
            span[dealer] = (
                node_off as u64,
                scratch.nodes.len() as u64,
                edge_off as u64,
                scratch.edges.len() as u64,
            );
        }
        spans.push(span);
        weights.push(entry.weight);
    }

    let nodes_buf = nodes_arena.finish();
    let edges_buf = edges_arena.finish();
    let entries = assemble_entries(&nodes_buf, &edges_buf, 0, 0, &spans, &weights);
    PrebuiltTrees {
        entries,
        info_sets,
        nodes_buf,
        edges_buf,
        spans,
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_action_mask_node(
    source: &GameTree,
    source_id: NodeId,
    prebuilt: &PrebuiltTrees,
    allowed: &[Box<[bool]>],
    old_to_new: &mut std::collections::HashMap<u32, u32>,
    info_sets: &mut Vec<(InfoSetKey, InfoSet, InfoActions)>,
    target: &mut TreeScratch,
) -> NodeId {
    let NodeView::Player {
        player,
        table_idx,
        edges,
    } = source.view(source_id)
    else {
        let NodeView::Terminal { payoff_p0 } = source.view(source_id) else {
            unreachable!()
        };
        let id = target.nodes.len() as NodeId;
        target.nodes.push(PackedNode {
            table_idx: 0,
            edge_off: 0,
            n_actions: 0,
            player: 0,
            payoff: payoff_p0 as i8,
            _pad: 0,
        });
        return id;
    };

    let old_idx = table_idx as usize;
    let old_actions = &prebuilt.info_sets[old_idx].2;
    assert_eq!(old_actions.len(), edges.len());
    let kept: Vec<usize> = (0..edges.len()).filter(|&i| allowed[old_idx][i]).collect();
    assert!(!kept.is_empty());

    let new_table_idx = match old_to_new.entry(table_idx) {
        std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let new_idx = info_sets.len() as u32;
            entry.insert(new_idx);
            let (key, info, _) = &prebuilt.info_sets[old_idx];
            let actions: Vec<_> = kept.iter().map(|&i| old_actions[i]).collect();
            info_sets.push((*key, info.clone(), actions.into_boxed_slice()));
            new_idx
        }
    };

    let id = target.nodes.len() as NodeId;
    let edge_off = target.edges.len();
    target.nodes.push(PackedNode {
        table_idx: new_table_idx,
        edge_off: edge_off as u32,
        n_actions: kept.len() as u8,
        player,
        payoff: 0,
        _pad: 0,
    });
    for &i in &kept {
        target.edges.push(PackedEdge::new(edges[i].action_u8, 0));
    }
    for (new_i, &old_i) in kept.iter().enumerate() {
        let child = copy_action_mask_node(
            source,
            edges[old_i].child,
            prebuilt,
            allowed,
            old_to_new,
            info_sets,
            target,
        );
        target.edges[edge_off + new_i].child = child;
    }
    id
}

/// Like [`build_all_trees`], but optionally restricted to a single dealer
/// arrangement. Since info sets encode position, the dealer-0 and dealer-1
/// games share no info sets and interact only through the (already known)
/// match-value table — they are fully independent games. Solving them in
/// separate processes halves the tree memory per run and is mathematically
/// identical to a joint solve. The excluded dealer's tree is left empty.
pub fn build_all_trees_with_dealer(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
) -> Result<PrebuiltTrees, truco_engine::EngineError> {
    build_all_trees_with_dealer_rules(score, tc, deals, dealer_filter, TreeRules::Current)
}

/// Like [`build_all_trees_with_dealer`] but with an explicit pruning rule set,
/// so a solve can be run on the experimental asymmetric-raise-pruned tree.
pub fn build_all_trees_with_dealer_rules(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
) -> Result<PrebuiltTrees, truco_engine::EngineError> {
    let mut registry = InfoSetRegistry::default();
    let mut nodes_arena = ArenaBuilder::default();
    let mut edges_arena = ArenaBuilder::default();
    let mut spans: Vec<[(u64, u64, u64, u64); 2]> = Vec::with_capacity(deals.len());
    let mut weights: Vec<f64> = Vec::with_capacity(deals.len());
    let mut scratch = TreeScratch::default();

    let include = |dealer: Player| dealer_filter.is_none() || dealer_filter == Some(dealer);

    for deal in deals {
        let mut span = [(0u64, 0u64, 0u64, 0u64); 2];
        for dealer in 0..2u8 {
            if !include(dealer) {
                continue;
            }
            scratch.nodes.clear();
            scratch.edges.clear();
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            build_node(&state, &mut scratch, &mut registry)?;
            let node_off = nodes_arena.append(&scratch.nodes);
            let edge_off = edges_arena.append(&scratch.edges);
            span[dealer as usize] = (
                node_off as u64,
                scratch.nodes.len() as u64,
                edge_off as u64,
                scratch.edges.len() as u64,
            );
        }
        spans.push(span);
        weights.push(deal.weight);
    }

    let nodes_buf = nodes_arena.finish();
    let edges_buf = edges_arena.finish();
    let entries = assemble_entries(&nodes_buf, &edges_buf, 0, 0, &spans, &weights);

    Ok(PrebuiltTrees {
        entries,
        info_sets: registry.entries,
        nodes_buf,
        edges_buf,
        spans,
    })
}

// ─── Deep-solve trunk arena + per-subgame local builds (plan 84 Phase 5) ───

/// One round-2 boundary crossing recorded by [`build_trunk_arena`]: its node in
/// the TRUNK arena (a truncated leaf), the public state that groups it into a
/// subgame, both players' boundary views, and the engine `state` that seeds the
/// on-demand per-subgame subtree build (`build_subgame_local`). The captured
/// `state` replays the exact recursion the full build would run below the
/// boundary, so the local subtree is structurally identical to the full arena's
/// — only its `table_idx` values are subgame-local instead of global.
#[derive(Clone)]
pub struct TrunkCrossing {
    /// Flat tree index (matches `subgame::flat_trees` / `flat_trees_dw` order).
    pub tree_idx: u32,
    /// Boundary node id in the trunk arena (a truncated Player leaf).
    pub node: NodeId,
    pub deal_weight: f64,
    pub dealer: Player,
    /// Dealer byte followed by the masked public projection — equal iff two
    /// crossings share a subgame (identical to `subgame::PublicKey`'s bytes).
    pub public_key: Vec<u8>,
    pub view_p0: InfoSetKey,
    pub view_p1: InfoSetKey,
    /// Replay seed: deal index (into the build's `deals` slice) plus the raw
    /// root→boundary action codes. A full `TraversalState` per crossing is
    /// deliberately NOT stored — at 0×0 that is ~7.8M crossings × ~10 KB of
    /// engine state ≈ 78 GB, the 2026-07-22 OOM. Rebuild on demand with
    /// [`replay_crossing_state`] (~40 bytes/crossing resident instead).
    pub deal_idx: u32,
    pub path: Box<[u8]>,
}

/// Rebuild a crossing's boundary `TraversalState` from its replay seed.
pub fn replay_crossing_state(
    score: &Score,
    tc: TurnupClass,
    deal: &AbstractDeal,
    dealer: Player,
    rules: TreeRules,
    path: &[u8],
) -> Result<TraversalState, truco_engine::EngineError> {
    let mut state = TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
    for &code in path {
        state = state.apply_abstract_action(crate::info_set::AbstractAction::from_u8(code))?;
    }
    Ok(state)
}

/// The trunk-only arena plus the boundary crossings it truncated at.
pub struct TrunkArena {
    /// Trunk arena: every deal tree built down to (and including) the round-2
    /// boundary, where the subtree is replaced by a truncated Player leaf whose
    /// edges point at a shared dummy terminal (never followed — the CFR/BR
    /// boundary hooks intercept the node first). `info_sets` holds only trunk
    /// info sets plus the boundary roots (inert: the trunk never trains them).
    pub prebuilt: PrebuiltTrees,
    /// Crossings in build order (deal-major, dealer 0 then 1, preorder within a
    /// tree). `deep.rs` groups these into subgames by `public_key`.
    pub crossings: Vec<TrunkCrossing>,
}

fn trunk_is_boundary(state: &TraversalState) -> bool {
    let st = state.engine.state();
    st.completed_rounds.len() == 1
        && st.current_round.plays.is_empty()
        && st.pending_raise.is_none()
}

fn trunk_view(state: &TraversalState, p: Player) -> InfoSet {
    InfoSet {
        player: p,
        is_dealer: p == state.engine.state().dealer,
        turnup_class: state.turnup_class,
        starting_hand: state.starting_hands[p as usize].clone(),
        history: state.histories[p as usize].clone(),
    }
}

fn trunk_push_public(public: &mut Vec<u8>, action: AbstractAction) {
    let masked = match action {
        AbstractAction::PlayFaceDown(_) => AbstractAction::OpponentPlayedHidden,
        other => other,
    };
    public.push(masked.to_u8());
}

#[allow(clippy::too_many_arguments)]
fn build_node_trunk(
    state: &TraversalState,
    tree: &mut TreeScratch,
    registry: &mut InfoSetRegistry,
    crossings: &mut Vec<TrunkCrossing>,
    tree_idx: u32,
    deal_weight: f64,
    dealer: Player,
    deal_idx: u32,
    crossed: bool,
    public: &mut Vec<u8>,
    path: &mut Vec<u8>,
    boundary_slots: &mut Vec<usize>,
) -> Result<NodeId, truco_engine::EngineError> {
    if state.is_terminal() {
        let winner = state
            .hand_winner()
            .expect("terminal state must have a winner");
        let value = state.hand_value() as i8;
        let payoff_p0 = if winner == 0 { value } else { -value };
        let id = tree.nodes.len() as NodeId;
        tree.nodes.push(PackedNode {
            table_idx: 0,
            edge_off: 0,
            n_actions: 0,
            player: 0,
            payoff: payoff_p0,
            _pad: 0,
        });
        return Ok(id);
    }

    let actions = state.abstract_legal_actions()?;
    assert!(!actions.is_empty(), "non-terminal state with no actions");
    let info_set = state
        .current_info_set()
        .expect("non-terminal state must have an info set");
    let player = state
        .engine
        .current_player()
        .expect("non-terminal state must have a current player");
    let table_idx = registry.register(info_set.key(), info_set, &actions);

    let id = tree.nodes.len() as NodeId;
    let edge_off = tree.edges.len();
    tree.nodes.push(PackedNode {
        table_idx,
        edge_off: edge_off as u32,
        n_actions: actions.len() as u8,
        player,
        payoff: 0,
        _pad: 0,
    });
    for &action in &actions {
        tree.edges.push(PackedEdge::new(action.to_u8(), 0));
    }

    if !crossed && trunk_is_boundary(state) {
        // Truncate: record the crossing (with the seed state) and leave the
        // edges pointing at the (yet-to-be-appended) shared dummy terminal.
        for i in 0..actions.len() {
            boundary_slots.push(edge_off + i);
        }
        let mut key = Vec::with_capacity(public.len() + 1);
        key.push(dealer);
        key.extend_from_slice(public);
        crossings.push(TrunkCrossing {
            tree_idx,
            node: id,
            deal_weight,
            dealer,
            public_key: key,
            view_p0: trunk_view(state, 0).key(),
            view_p1: trunk_view(state, 1).key(),
            deal_idx,
            path: path.clone().into_boxed_slice(),
        });
        return Ok(id);
    }

    for (i, &action) in actions.iter().enumerate() {
        let child_state = state.apply_abstract_action(action)?;
        trunk_push_public(public, action);
        path.push(action.to_u8());
        let child_id = build_node_trunk(
            &child_state,
            tree,
            registry,
            crossings,
            tree_idx,
            deal_weight,
            dealer,
            deal_idx,
            crossed,
            public,
            path,
            boundary_slots,
        )?;
        tree.edges[edge_off + i].child = child_id;
        path.pop();
        public.pop();
    }
    Ok(id)
}

/// Build the trunk-only arena for a (score, tc, dealer) cell: every deal tree
/// down to the round-2 boundary, where the subtree is truncated to a leaf and
/// the crossing recorded. This is the memory win of the deep path — the trunk
/// is ~1% of the full node count at a deep full-ladder cell. See [`TrunkArena`].
pub fn build_trunk_arena(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
) -> Result<TrunkArena, truco_engine::EngineError> {
    let mut registry = InfoSetRegistry::default();
    let mut nodes_arena = ArenaBuilder::default();
    let mut edges_arena = ArenaBuilder::default();
    let mut spans: Vec<[(u64, u64, u64, u64); 2]> = Vec::with_capacity(deals.len());
    let mut weights: Vec<f64> = Vec::with_capacity(deals.len());
    let mut scratch = TreeScratch::default();
    let mut crossings: Vec<TrunkCrossing> = Vec::new();
    let mut tree_idx: u32 = 0;

    let include = |dealer: Player| dealer_filter.is_none() || dealer_filter == Some(dealer);

    for (deal_idx, deal) in deals.iter().enumerate() {
        let mut span = [(0u64, 0u64, 0u64, 0u64); 2];
        for dealer in 0..2u8 {
            if !include(dealer) {
                continue;
            }
            scratch.nodes.clear();
            scratch.edges.clear();
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            let mut public: Vec<u8> = Vec::new();
            let mut path: Vec<u8> = Vec::new();
            let mut boundary_slots: Vec<usize> = Vec::new();
            build_node_trunk(
                &state,
                &mut scratch,
                &mut registry,
                &mut crossings,
                tree_idx,
                deal.weight,
                dealer,
                deal_idx as u32,
                false,
                &mut public,
                &mut path,
                &mut boundary_slots,
            )?;
            if !boundary_slots.is_empty() {
                // One shared dummy terminal; every truncated boundary edge
                // points here (never followed — hooks intercept the boundary
                // node). Its presence keeps `reach_excluding` well-defined.
                let dummy = scratch.nodes.len() as NodeId;
                scratch.nodes.push(PackedNode {
                    table_idx: 0,
                    edge_off: 0,
                    n_actions: 0,
                    player: 0,
                    payoff: 0,
                    _pad: 0,
                });
                for &slot in &boundary_slots {
                    scratch.edges[slot].child = dummy;
                }
            }
            let node_off = nodes_arena.append(&scratch.nodes);
            let edge_off = edges_arena.append(&scratch.edges);
            span[dealer as usize] = (
                node_off as u64,
                scratch.nodes.len() as u64,
                edge_off as u64,
                scratch.edges.len() as u64,
            );
            tree_idx += 1;
        }
        spans.push(span);
        weights.push(deal.weight);
    }

    let nodes_buf = nodes_arena.finish();
    let edges_buf = edges_arena.finish();
    let entries = assemble_entries(&nodes_buf, &edges_buf, 0, 0, &spans, &weights);
    Ok(TrunkArena {
        prebuilt: PrebuiltTrees {
            entries,
            info_sets: registry.entries,
            nodes_buf,
            edges_buf,
            spans,
        },
        crossings,
    })
}

/// Build the local arena for ONE subgame: one member subtree per `(state,
/// weight)` (all sharing the subgame's `dealer`), all sharing a single local
/// `InfoSetRegistry`. The local registry IS the subgame's compact accumulator
/// index; `flat_trees`/`flat_trees_dw` yield the members in `states` order with
/// `tree_idx == member index`. Returns the local `PrebuiltTrees` and the
/// `InfoSetKey -> local table_idx` map the streaming certificate resolves rows
/// through. Info sets are shared across members (the subgame's private hands
/// ranging over consistent deals), exactly as the full arena would share them.
pub fn build_subgame_local(
    states: &[(TraversalState, f64)],
    dealer: Player,
) -> Result<(PrebuiltTrees, std::collections::HashMap<InfoSetKey, u32>), truco_engine::EngineError>
{
    let mut registry = InfoSetRegistry::default();
    let mut nodes_arena = ArenaBuilder::default();
    let mut edges_arena = ArenaBuilder::default();
    let mut spans: Vec<[(u64, u64, u64, u64); 2]> = Vec::with_capacity(states.len());
    let mut weights: Vec<f64> = Vec::with_capacity(states.len());
    let mut scratch = TreeScratch::default();

    for (state, weight) in states {
        scratch.nodes.clear();
        scratch.edges.clear();
        build_node(state, &mut scratch, &mut registry)?;
        let node_off = nodes_arena.append(&scratch.nodes);
        let edge_off = edges_arena.append(&scratch.edges);
        let mut span = [(0u64, 0u64, 0u64, 0u64); 2];
        span[dealer as usize] = (
            node_off as u64,
            scratch.nodes.len() as u64,
            edge_off as u64,
            scratch.edges.len() as u64,
        );
        spans.push(span);
        weights.push(*weight);
    }

    let nodes_buf = nodes_arena.finish();
    let edges_buf = edges_arena.finish();
    let entries = assemble_entries(&nodes_buf, &edges_buf, 0, 0, &spans, &weights);
    let key_to_idx: std::collections::HashMap<InfoSetKey, u32> = registry
        .entries
        .iter()
        .enumerate()
        .map(|(i, (k, _, _))| (*k, i as u32))
        .collect();
    Ok((
        PrebuiltTrees {
            entries,
            info_sets: registry.entries,
            nodes_buf,
            edges_buf,
            spans,
        },
        key_to_idx,
    ))
}

/// Construct per-deal `GameTree` handles over shared arenas. `nodes_base` /
/// `edges_base` are byte offsets of the arenas within the backing buffers
/// (0 for owned arenas; section offsets for a mapped treepack file).
pub fn assemble_entries(
    nodes_buf: &TreeBytes,
    edges_buf: &TreeBytes,
    nodes_base: usize,
    edges_base: usize,
    spans: &[[(u64, u64, u64, u64); 2]],
    weights: &[f64],
) -> Vec<DealTrees> {
    spans
        .iter()
        .zip(weights.iter())
        .map(|(span, &weight)| {
            let tree = |d: usize| {
                let (no, nc, eo, ec) = span[d];
                if nc == 0 {
                    return GameTree::empty();
                }
                GameTree::from_arena(
                    nodes_buf,
                    edges_buf,
                    nodes_base,
                    edges_base,
                    no as usize,
                    nc as usize,
                    eo as usize,
                    ec as usize,
                )
            };
            DealTrees {
                weight,
                tree_dealer_0: tree(0),
                tree_dealer_1: tree(1),
            }
        })
        .collect()
}

/// Signature of the tree-shape equivalence class for a (score, dealer-filter)
/// solve. Trees are IDENTICAL for every score state within a class (measured:
/// 6x6 and 8x8 match node-for-node): shape depends only on the available
/// action ladder — scores enter payoffs only through the runtime mv lookup.
/// Classes: 11x11 (no eleven decision, no raises); one-player-at-11 mão de
/// onze, oriented by which seat decides; otherwise the reachable raise ladder
/// determined by min(score) under the `min + stake >= 12` pruning rule.
pub fn band_signature(score: &Score, dealer_filter: Option<Player>) -> String {
    let band = if score.zero == 11 && score.one == 11 {
        "mao1111".to_string()
    } else if score.zero == 11 {
        "mao-p0".to_string()
    } else if score.one == 11 {
        "mao-p1".to_string()
    } else {
        let min = score.zero.min(score.one);
        let mut sig = String::from("L1");
        let mut stake = 1u8;
        for rung in [3u8, 6, 9, 12] {
            if min + stake >= truco_engine::MATCH_TARGET {
                break;
            }
            sig.push('-');
            sig.push_str(&rung.to_string());
            stake = rung;
        }
        sig
    };
    match dealer_filter {
        Some(d) => format!("{band}-d{d}"),
        None => format!("{band}-dboth"),
    }
}

// ─── Concrete card realization helpers ──────────────────────────────────

fn realize_deal(
    tc: TurnupClass,
    deal: &AbstractDeal,
) -> Result<(Turnup, Hands, Vec<(String, AbstractCard)>), truco_engine::EngineError> {
    let (turnup_rank, manilha_rank) = canonical_turnup_for_class(tc);
    let turnup = Turnup {
        rank: turnup_rank,
        suit: Suit::Hearts,
    };

    let mut card_map = Vec::new();
    let mut used_suits: std::collections::HashMap<Rank, Vec<Suit>> =
        std::collections::HashMap::new();

    let (hand0, hand1) = {
        let mut realize_hand = |hand: &AbstractHand,
                                player_idx: usize|
         -> Result<Vec<Card>, truco_engine::EngineError> {
            let mut cards = Vec::new();
            for (card_idx, &abs_card) in hand.iter().enumerate() {
                let (rank, suit) =
                    pick_concrete_card(abs_card, &turnup, manilha_rank, &mut used_suits)
                        .ok_or(truco_engine::EngineError::InvalidInitialState)?;
                let id = format!("p{}c{}", player_idx, card_idx);
                card_map.push((id.clone(), abs_card));
                cards.push(Card {
                    id: id.into(),
                    rank,
                    suit,
                });
            }
            Ok(cards)
        };

        let hand0 = realize_hand(&deal.hands[0], 0)?;
        let hand1 = realize_hand(&deal.hands[1], 1)?;
        (hand0, hand1)
    };

    let hands = Hands {
        zero: hand0.into(),
        one: hand1.into(),
    };

    Ok((turnup, hands, card_map))
}

fn canonical_turnup_for_class(tc: TurnupClass) -> (Rank, Rank) {
    for &rank in &ALL_RANKS {
        let manilha = rank.next_for_manilha();
        let manilha_idx = manilha.index();
        let rank_idx = rank.index();
        let blocked = if rank_idx < manilha_idx {
            rank_idx
        } else {
            rank_idx - 1
        };
        if blocked == tc.blocked_plain_level as usize {
            return (rank, manilha);
        }
    }
    unreachable!(
        "no rank produces blocked_plain_level {}",
        tc.blocked_plain_level
    )
}

fn pick_concrete_card(
    abs: AbstractCard,
    turnup: &Turnup,
    manilha_rank: Rank,
    used: &mut std::collections::HashMap<Rank, Vec<Suit>>,
) -> Option<(Rank, Suit)> {
    match abs {
        AbstractCard::Manilha(suit_idx) => {
            let suit = ALL_SUITS[suit_idx as usize];
            used.entry(manilha_rank).or_default().push(suit);
            Some((manilha_rank, suit))
        }
        AbstractCard::Plain(strength) => {
            let rank = plain_index_to_rank(strength, manilha_rank);
            let suit = pick_available_suit(rank, turnup, used)?;
            used.entry(rank).or_default().push(suit);
            Some((rank, suit))
        }
    }
}

fn plain_index_to_rank(strength: u8, manilha_rank: Rank) -> Rank {
    let manilha_idx = manilha_rank.index();
    let full_idx = if (strength as usize) < manilha_idx {
        strength as usize
    } else {
        strength as usize + 1
    };
    ALL_RANKS[full_idx]
}

fn pick_available_suit(
    rank: Rank,
    turnup: &Turnup,
    used: &std::collections::HashMap<Rank, Vec<Suit>>,
) -> Option<Suit> {
    let used_suits = used.get(&rank).map(|v| v.as_slice()).unwrap_or(&[]);
    for &suit in &ALL_SUITS {
        if rank == turnup.rank && suit == turnup.suit {
            continue;
        }
        if used_suits.contains(&suit) {
            continue;
        }
        return Some(suit);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::strategy::{InfoSetData, StrategyTable};

    #[test]
    fn boxed_info_actions_use_less_row_metadata_than_vec() {
        assert!(std::mem::size_of::<InfoActions>() < std::mem::size_of::<Vec<AbstractAction>>());
    }

    #[test]
    fn test_realize_deal_produces_valid_engine() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        for deal in deals.iter().take(5) {
            let result = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, deal);
            assert!(
                result.is_ok(),
                "failed to create engine from deal: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_abstract_legal_actions_at_start() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];
        let state = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, deal).unwrap();

        let actions = state.abstract_legal_actions().unwrap();
        assert!(!actions.is_empty());
        for a in &actions {
            match a {
                AbstractAction::PlayFaceUp(_) | AbstractAction::Raise(_) => {}
                _ => panic!("unexpected action at start: {:?}", a),
            }
        }
    }

    #[test]
    fn test_raise_pruning_by_score() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];

        // 10x10: truco (stake 3) already decides the match, so no seis re-raise.
        let s10 = TraversalState::from_deal(0, Score { zero: 10, one: 10 }, tc, deal).unwrap();
        let a10 = s10.abstract_legal_actions().unwrap();
        assert!(
            a10.iter().any(|a| matches!(a, AbstractAction::Raise(3))),
            "truco should be available at 10x10: {:?}",
            a10
        );
        let after_truco = s10.apply_abstract_action(AbstractAction::Raise(3)).unwrap();
        let opp = after_truco.abstract_legal_actions().unwrap();
        assert!(
            !opp.iter().any(|a| matches!(a, AbstractAction::Raise(_))),
            "no re-raise once stake 3 decides the match at 10x10: {:?}",
            opp
        );

        // 6x6: truco does not yet decide the match (6+3=9), so seis stays live.
        let s6 = TraversalState::from_deal(0, Score { zero: 6, one: 6 }, tc, deal).unwrap();
        let after_truco6 = s6.apply_abstract_action(AbstractAction::Raise(3)).unwrap();
        let opp6 = after_truco6.abstract_legal_actions().unwrap();
        assert!(
            opp6.iter().any(|a| matches!(a, AbstractAction::Raise(6))),
            "seis should still be available after truco at 6x6: {:?}",
            opp6
        );
    }

    #[test]
    fn hidden_play_dominance_keeps_only_round_two_leads() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deal = AbstractDeal {
            hands: [
                [
                    AbstractCard::Plain(0),
                    AbstractCard::Plain(1),
                    AbstractCard::Plain(8),
                ]
                .into_iter()
                .collect(),
                [
                    AbstractCard::Plain(2),
                    AbstractCard::Plain(3),
                    AbstractCard::Plain(7),
                ]
                .into_iter()
                .collect(),
            ],
            weight: 1.0,
        };
        // 11x11 is useful here: the safe rule still leaves the strategically
        // live round-two lead available during mão de onze.
        let state = TraversalState::from_deal(0, Score { zero: 11, one: 11 }, tc, &deal).unwrap();

        // Player 1 leads round 1 with 2; player 0's 8 wins it.
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(2)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(8)))
            .unwrap();

        // The round-one winner leads round 2. Hiding is still available here.
        let actions = state.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, AbstractAction::PlayFaceDown(_))),
            "a round-two leader may still conceal the remaining card"
        );

        // Once player 0 leads the 0, player 1 is the second mover. Every hide
        // is weakly dominated by showing the same card.
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(0)))
            .unwrap();
        let actions = state.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, AbstractAction::PlayFaceDown(_))),
            "the second play of round two must not retain dominated hides"
        );

        // Player 1's 7 wins round 2, producing a 1-1 mão-de-onze hand. With
        // raises disabled, neither mover may hide in the final round.
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(7)))
            .unwrap();
        let actions = state.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, AbstractAction::PlayFaceDown(_))),
            "round-three leader must not retain dominated hides"
        );
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(3)))
            .unwrap();
        let actions = state.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, AbstractAction::PlayFaceDown(_))),
            "round-three responder must not retain dominated hides"
        );

        // In an ordinary score state, however, the round-three leader's reveal
        // can change the responder's raise strategy. Preserve that hide; only
        // the responder's terminal hide is directly dominated.
        let regular = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, &deal).unwrap();
        let regular = regular
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(2)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(8)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(0)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(7)))
            .unwrap();
        let actions = regular.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, AbstractAction::PlayFaceDown(_))),
            "round-three leader hide must survive while the responder can raise"
        );
        let regular = regular
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(3)))
            .unwrap();
        let actions = regular.abstract_legal_actions().unwrap();
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, AbstractAction::PlayFaceDown(_))),
            "round-three responder hide is directly dominated even when raising is available"
        );
    }

    #[test]
    fn final_weakest_card_raise_response_forces_fold_at_nine_all() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deal = AbstractDeal {
            hands: [
                [
                    AbstractCard::Plain(0),
                    AbstractCard::Plain(1),
                    AbstractCard::Plain(8),
                ]
                .into_iter()
                .collect(),
                [
                    AbstractCard::Plain(2),
                    AbstractCard::Plain(3),
                    AbstractCard::Plain(7),
                ]
                .into_iter()
                .collect(),
            ],
            weight: 1.0,
        };
        let state = TraversalState::from_deal(0, Score { zero: 9, one: 9 }, tc, &deal).unwrap();

        // Player 1 wins round 1; player 0 wins round 2 and leads the last 4.
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(7)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(1)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(2)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(8)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(0)))
            .unwrap()
            .apply_abstract_action(AbstractAction::Raise(3))
            .unwrap();

        // Accept loses three points with certainty; fold loses one. A seis
        // bluff is already removed because stake 3 decides a 9x9 match.
        assert_eq!(
            state.abstract_legal_actions().unwrap(),
            vec![AbstractAction::Fold]
        );

        // At a lower score the same showdown certificate removes calling, but
        // not a re-raise: the latter can still win through fold equity.
        let state = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, &deal).unwrap();
        let state = state
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(7)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(1)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(2)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(8)))
            .unwrap()
            .apply_abstract_action(AbstractAction::PlayFaceUp(AbstractCard::Plain(0)))
            .unwrap()
            .apply_abstract_action(AbstractAction::Raise(3))
            .unwrap();
        let actions = state.abstract_legal_actions().unwrap();
        assert!(!actions.contains(&AbstractAction::AcceptRaise));
        assert!(actions.contains(&AbstractAction::Fold));
        assert!(actions.contains(&AbstractAction::Raise(6)));
    }

    #[test]
    fn test_count_tree_size_matches_full_build() {
        // count_tree_size must agree exactly with the real solver-ready build
        // (build_all_trees_with_dealer) on both node count and distinct
        // info-set count, for every score tier this project cares about —
        // this is the correctness check that must pass before trusting the
        // lightweight counter on trees too large to fully build.
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let subset: Vec<_> = deals.into_iter().take(15).collect();

        for score in [
            Score { zero: 0, one: 0 },
            Score { zero: 6, one: 6 },
            Score { zero: 10, one: 10 },
            Score { zero: 10, one: 9 },
            Score { zero: 11, one: 5 },
        ] {
            for dealer in 0..2u8 {
                let prebuilt =
                    build_all_trees_with_dealer(&score, tc, &subset, Some(dealer)).unwrap();
                let expected_nodes: u64 = prebuilt
                    .spans
                    .iter()
                    .map(|span| span[dealer as usize].1)
                    .sum();
                let expected_info_sets = prebuilt.info_sets.len();

                let counted = count_tree_size(&score, tc, dealer, &subset, None).unwrap();

                assert_eq!(
                    counted.total_nodes, expected_nodes,
                    "node count mismatch at {}x{} dealer {}",
                    score.zero, score.one, dealer
                );
                assert_eq!(
                    counted.num_info_sets, expected_info_sets,
                    "info-set count mismatch at {}x{} dealer {}",
                    score.zero, score.one, dealer
                );
            }
        }
    }

    #[test]
    fn asymmetric_raise_prune_shrinks_lopsided_but_not_symmetric() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(20).collect();
        let count = |score: Score, rules: TreeRules| {
            count_tree_size_with_rules(&score, tc, 0, &deals, rules, None).unwrap()
        };

        // Symmetric scores: the acting player's score equals min(score), so the
        // asymmetric gate reduces to the deployed rule — trees are identical.
        for sym in [
            Score { zero: 6, one: 6 },
            Score { zero: 3, one: 3 },
            Score { zero: 0, one: 0 },
        ] {
            let cur = count(sym.clone(), TreeRules::Current);
            let asym = count(sym.clone(), TreeRules::AsymmetricRaisePrune);
            assert_eq!(
                (cur.total_nodes, cur.num_info_sets),
                (asym.total_nodes, asym.num_info_sets),
                "asymmetric prune changed the symmetric score {}x{}",
                sym.zero,
                sym.one
            );
        }

        // Lopsided scores in the same band as their symmetric mate must be
        // strictly smaller under the asymmetric rule: the higher-scored player's
        // now-dominated raises are removed while the symmetric mate keeps both.
        for (lop, mate) in [
            (Score { zero: 9, one: 6 }, Score { zero: 6, one: 6 }),
            (Score { zero: 9, one: 0 }, Score { zero: 0, one: 0 }),
        ] {
            let cur = count(lop.clone(), TreeRules::Current);
            let asym = count(lop.clone(), TreeRules::AsymmetricRaisePrune);
            let mate_cur = count(mate.clone(), TreeRules::Current);
            // Current rule keys on min(score), so the lopsided cell matches its
            // symmetric band-mate; the asymmetric rule breaks that tie downward.
            assert_eq!(cur.num_info_sets, mate_cur.num_info_sets);
            assert!(
                asym.num_info_sets < cur.num_info_sets,
                "asymmetric prune did not shrink lopsided {}x{} ({} vs {})",
                lop.zero,
                lop.one,
                asym.num_info_sets,
                cur.num_info_sets
            );
        }
    }

    #[test]
    fn test_policy_profile_with_full_support_matches_raw_count() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(3).collect();
        let score = Score { zero: 10, one: 10 };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let mut policy = StrategyTable::new();
        for (key, info, actions) in &prebuilt.info_sets {
            let mut data = InfoSetData::new(actions.to_vec());
            data.cumulative_strategy.fill(1.0);
            policy.insert_serialized(*key, info.clone(), data);
        }

        let raw = count_tree_size(&score, tc, 0, &deals, None).unwrap();
        let supported = count_policy_tree_size(
            &score,
            tc,
            0,
            &deals,
            &policy,
            PolicyTreeMode::Profile,
            PolicyValueSource::Average,
            0.0,
            MissingPolicyFallback::All,
            None,
        )
        .unwrap();

        assert_eq!(supported.total_nodes, raw.total_nodes);
        assert_eq!(supported.num_info_sets, raw.num_info_sets);
        assert_eq!(supported.policy_missing_info_sets, 0);
        assert_eq!(supported.kept_actions, supported.legal_actions);
    }

    #[test]
    fn test_best_response_union_contains_profile_and_each_response() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(3).collect();
        let score = Score { zero: 10, one: 10 };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let mut policy = StrategyTable::new();
        for (key, info, actions) in &prebuilt.info_sets {
            let mut data = InfoSetData::new(actions.to_vec());
            data.cumulative_strategy.fill(0.0);
            data.cumulative_strategy[0] = 1.0;
            policy.insert_serialized(*key, info.clone(), data);
        }

        let count = |mode| {
            count_policy_tree_size(
                &score,
                tc,
                0,
                &deals,
                &policy,
                mode,
                PolicyValueSource::Average,
                0.0,
                MissingPolicyFallback::All,
                None,
            )
            .unwrap()
        };
        let raw = count_tree_size(&score, tc, 0, &deals, None).unwrap();
        let profile = count(PolicyTreeMode::Profile);
        let br0 = count(PolicyTreeMode::BestResponse(0));
        let br1 = count(PolicyTreeMode::BestResponse(1));
        let union = count(PolicyTreeMode::BestResponseUnion);

        assert!(profile.total_nodes < raw.total_nodes);
        assert!(br0.total_nodes >= profile.total_nodes);
        assert!(br1.total_nodes >= profile.total_nodes);
        assert!(union.total_nodes >= br0.total_nodes);
        assert!(union.total_nodes >= br1.total_nodes);
        assert!(union.total_nodes <= raw.total_nodes);
        assert_eq!(union.policy_missing_info_sets, 0);
    }

    #[test]
    fn test_chosen_best_response_union_is_bounded_by_all_action_union() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(3).collect();
        let score = Score { zero: 10, one: 10 };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let mut policy = StrategyTable::new();
        let mut strategies = Vec::with_capacity(prebuilt.info_sets.len());
        let mut chosen0 = vec![u8::MAX; prebuilt.info_sets.len()];
        let mut chosen1 = vec![u8::MAX; prebuilt.info_sets.len()];
        for (idx, (key, info, actions)) in prebuilt.info_sets.iter().enumerate() {
            let mut data = InfoSetData::new(actions.to_vec());
            data.cumulative_strategy.fill(0.0);
            data.cumulative_strategy[0] = 1.0;
            policy.insert_serialized(*key, info.clone(), data);

            let mut probabilities = ActionProbs::from_vec(vec![0.0; actions.len()]);
            probabilities[0] = 1.0;
            strategies.push(probabilities);
            if info.player == 0 {
                chosen0[idx] = (actions.len() - 1) as u8;
            } else {
                chosen1[idx] = (actions.len() - 1) as u8;
            }
        }

        let count = |mode| {
            count_policy_tree_size(
                &score,
                tc,
                0,
                &deals,
                &policy,
                mode,
                PolicyValueSource::Average,
                0.0,
                MissingPolicyFallback::All,
                None,
            )
            .unwrap()
        };
        let profile = count(PolicyTreeMode::Profile);
        let all_action_union = count(PolicyTreeMode::BestResponseUnion);
        let chosen_union =
            count_chosen_best_response_union(&prebuilt, &strategies, [&chosen0, &chosen1], 0.0);
        let chosen_closure =
            count_chosen_best_response_closure(&prebuilt, &strategies, [&chosen0, &chosen1], 0.0);
        let restricted =
            restrict_prebuilt_to_chosen_closure(&prebuilt, &strategies, [&chosen0, &chosen1], 0.0);
        let restricted_nodes: usize = restricted
            .entries
            .iter()
            .map(|entry| entry.tree_dealer_0.num_nodes() + entry.tree_dealer_1.num_nodes())
            .sum();

        assert!(chosen_union.total_nodes >= profile.total_nodes);
        assert!(chosen_union.num_info_sets >= profile.num_info_sets);
        assert!(chosen_closure.total_nodes >= chosen_union.total_nodes);
        assert!(chosen_closure.num_info_sets >= chosen_union.num_info_sets);
        assert!(chosen_closure.total_nodes <= all_action_union.total_nodes);
        assert!(chosen_closure.num_info_sets <= all_action_union.num_info_sets);
        assert_eq!(chosen_union.policy_missing_info_sets, 0);
        assert_eq!(chosen_closure.policy_missing_info_sets, 0);
        assert_eq!(restricted_nodes as u64, chosen_closure.total_nodes);
        assert_eq!(restricted.info_sets.len(), chosen_closure.num_info_sets);
        for entry in &restricted.entries {
            for tree in [&entry.tree_dealer_0, &entry.tree_dealer_1] {
                for node in tree.nodes.iter() {
                    if node.n_actions > 0 {
                        assert_eq!(
                            node.n_actions as usize,
                            restricted.info_sets[node.table_idx as usize].2.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_policy_support_reports_projected_missing_raise() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(1).collect();
        let score = Score { zero: 10, one: 10 };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let (key, info, actions) = prebuilt
            .info_sets
            .iter()
            .find(|(_, _, actions)| !actions.is_empty())
            .unwrap();
        let mut data = InfoSetData::new(actions.to_vec());
        data.cumulative_strategy.fill(1.0);
        let mut policy = StrategyTable::new();
        policy.insert_serialized(*key, info.clone(), data);

        let mut projected_actions = actions.to_vec();
        projected_actions.push(AbstractAction::Raise(12));
        let (support, incomplete) = policy_support(
            &policy,
            *key,
            &projected_actions,
            PolicyValueSource::Average,
            0.0,
            MissingPolicyFallback::AllExceptRaise,
        );
        assert!(incomplete);
        assert!(!support[projected_actions.len() - 1]);

        let (high_threshold_support, _) = policy_support(
            &policy,
            *key,
            actions,
            PolicyValueSource::Average,
            2.0,
            MissingPolicyFallback::All,
        );
        assert_eq!(
            high_threshold_support.iter().filter(|&&keep| keep).count(),
            1,
            "the argmax action must survive even above the probability threshold"
        );
    }

    #[test]
    fn test_apply_abstract_action() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];
        let state = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, deal).unwrap();

        let actions = state.abstract_legal_actions().unwrap();
        let play_action = actions
            .iter()
            .find(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
            .unwrap();

        let new_state = state.apply_abstract_action(*play_action).unwrap();
        assert!(!new_state.is_terminal());
        assert_eq!(new_state.engine.current_player(), Some(0));
    }

    #[test]
    fn test_info_set_constructed_correctly() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];
        let state = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, deal).unwrap();

        let info_set = state.current_info_set().unwrap();
        assert_eq!(info_set.player, 1);
        // Dealer is player 0, so the acting player 1 (who leads) is not the pé.
        assert!(!info_set.is_dealer);
        assert_eq!(info_set.starting_hand, deal.hands[1]);
        assert!(info_set.history.is_empty());
    }

    #[test]
    fn test_dealer_trees_share_no_info_sets() {
        // The dealer-0 and dealer-1 games are independent; with position in the
        // info set their key spaces must be fully disjoint. Before this held,
        // the 11x10 accept/fold node (empty history) collided across the trees
        // — one position-averaged accept policy — and some card-play histories
        // aliased with opposite own/opponent attribution.
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let score = Score { zero: 11, one: 10 };

        let subset: Vec<_> = deals.into_iter().take(30).collect();
        let mut keys: [std::collections::HashSet<u64>; 2] = Default::default();
        for dealer in 0..2u8 {
            let prebuilt = build_all_trees_with_dealer(&score, tc, &subset, Some(dealer)).unwrap();
            for (key, _, _) in &prebuilt.info_sets {
                keys[dealer as usize].insert(key.0);
            }
        }
        assert!(!keys[0].is_empty() && !keys[1].is_empty());
        let shared: Vec<_> = keys[0].intersection(&keys[1]).collect();
        assert!(
            shared.is_empty(),
            "{} info-set keys appear in both dealer trees",
            shared.len()
        );
    }

    #[test]
    fn test_build_all_trees_with_dealer_filter() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(10).collect();
        let score = Score { zero: 11, one: 10 };

        let only_d1 = build_all_trees_with_dealer(&score, tc, &deals, Some(1)).unwrap();
        for entry in &only_d1.entries {
            assert!(entry.tree_dealer_0.nodes.is_empty());
            assert!(!entry.tree_dealer_1.nodes.is_empty());
        }

        // The filtered build's info sets are exactly the dealer-1 half of the
        // unfiltered build.
        let both = build_all_trees(&score, tc, &deals).unwrap();
        let d1_keys: std::collections::HashSet<u64> =
            only_d1.info_sets.iter().map(|(k, _, _)| k.0).collect();
        let both_d1_keys: std::collections::HashSet<u64> = both
            .info_sets
            .iter()
            .filter(|(_, is, _)| is.is_dealer == (is.player == 1))
            .map(|(k, _, _)| k.0)
            .collect();
        assert_eq!(d1_keys, both_d1_keys);
    }

    #[test]
    fn test_mao_de_onze_actions() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];
        let state = TraversalState::from_deal(0, Score { zero: 11, one: 0 }, tc, deal).unwrap();

        let actions = state.abstract_legal_actions().unwrap();
        assert!(actions.contains(&AbstractAction::AcceptEleven));
        assert!(actions.contains(&AbstractAction::FoldEleven));
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_deduplication_of_same_strength_cards() {
        use smallvec::smallvec;
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deal = AbstractDeal {
            hands: [
                smallvec![
                    AbstractCard::Plain(0),
                    AbstractCard::Plain(0),
                    AbstractCard::Plain(5)
                ],
                smallvec![
                    AbstractCard::Plain(1),
                    AbstractCard::Plain(2),
                    AbstractCard::Plain(7)
                ],
            ],
            weight: 1.0,
        };
        let state = TraversalState::from_deal(0, Score { zero: 0, one: 0 }, tc, &deal).unwrap();

        let actions = state.abstract_legal_actions().unwrap();
        let play_actions: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, AbstractAction::PlayFaceUp(_)))
            .collect();
        assert_eq!(play_actions.len(), 3);
    }

    #[test]
    fn test_build_game_tree_small() {
        // Build a game tree for a single deal at 11x11
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let deals = enumerate_deals(&tc);
        let deal = &deals[0];
        let state = TraversalState::from_deal(0, Score { zero: 11, one: 11 }, tc, deal).unwrap();

        let tree = build_game_tree(&state).unwrap();
        assert!(!tree.nodes.is_empty());

        // Count terminal and player nodes
        let terminals = tree.nodes.iter().filter(|n| n.n_actions == 0).count();
        let players = tree.nodes.iter().filter(|n| n.n_actions > 0).count();
        assert!(terminals > 0, "should have terminal nodes");
        assert!(players > 0, "should have player nodes");
        // Every player node's edge block is in bounds and children resolve.
        for id in 0..tree.nodes.len() {
            if let NodeView::Player { edges, .. } = tree.view(id as NodeId) {
                for e in edges {
                    assert!((e.child as usize) < tree.nodes.len());
                    let _ = e.action(); // decodes without panicking
                }
            }
        }
    }
}
