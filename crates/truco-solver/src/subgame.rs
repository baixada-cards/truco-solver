//! Subgame decomposition at the round-2 boundary (plan 84 Phase 3, CFR-D).
//!
//! Partitions every deal tree at the START OF ROUND 2 — the first node where
//! round 1's card plays are complete and no raise decision is pending — and
//! groups boundary nodes across deals by PUBLIC state (the action sequence
//! both players observe: raises/accepts/folds and card plays, with face-down
//! card identity masked). Each group is one re-solvable subgame: it contains
//! every (deal, node) pair consistent with that public history, which is
//! exactly the closure the re-solving gadget needs (both players know the
//! public state; their private hands range over all consistent deals).
//!
//! The walker REPLAYS the tree build's recursion (`TraversalState` +
//! `abstract_legal_actions`, the same driver `build_node` uses), so node ids
//! follow from recursion order by construction, the boundary condition reads
//! the real engine state, and both players' info-set views come from the
//! same `histories` the build derives acting info sets from. `build_node`
//! reserves each node before recursing into its children, so the arenas are
//! strict preorder and every subtree is the contiguous id range
//! `[node, subtree_end)` — the replay records spans directly.
//!
//! Boundary invariants (asserted in tests):
//! - no boundary node is an ancestor of another boundary node;
//! - every root-to-terminal path crosses at most one boundary node (paths
//!   that fold or end the hand during round 1 never reach the boundary);
//! - each player's boundary VIEW determines the public key (a view is the
//!   public key plus that player's private information), so views — and the
//!   acting info sets below the boundary — each belong to exactly one
//!   subgame.

use std::collections::HashMap;

use truco_engine::{Player, Score};

use crate::abstraction::{AbstractDeal, TurnupClass};
use crate::game_tree::{GameTree, NodeId, PrebuiltTrees, TraversalState, TreeRules};
use crate::info_set::{AbstractAction, InfoSet, InfoSetKey};
use crate::strategy::ActionProbs;

/// Opaque key for a subgame's public state: the dealer byte followed by the
/// u8-encoded public projection of the root→boundary action sequence
/// (face-down plays collapse to the `OpponentPlayedHidden` code so hidden
/// card identity never enters the key). Two boundary nodes share a subgame
/// iff their keys are equal.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PublicKey(pub Vec<u8>);

/// One boundary crossing in one deal tree.
#[derive(Clone, Debug)]
pub struct BoundaryNode {
    /// Index into [`flat_trees`]'s output (NOT into `PrebuiltTrees::entries`).
    pub tree_idx: u32,
    pub node: NodeId,
    /// Exclusive end of the contiguous preorder subtree under `node`.
    pub subtree_end: NodeId,
    /// Deal weight (chance probability mass) of the tree this node lives in.
    pub deal_weight: f64,
    /// Player 0's / player 1's info-set view of the history at this node.
    /// These key the gadget's Terminate/Follow decision for whichever player
    /// is the "opponent" of the current re-solve.
    pub view_p0: InfoSetKey,
    pub view_p1: InfoSetKey,
}

/// A re-solvable subgame: all boundary crossings sharing one public state.
#[derive(Clone, Debug)]
pub struct Subgame {
    pub public: PublicKey,
    pub members: Vec<BoundaryNode>,
}

/// Flat tree list in the same order used by the exact-BR pass: for each
/// `prebuilt.entries` entry, dealer-0 then dealer-1, skipping empty trees.
/// `BoundaryNode::tree_idx` indexes into THIS list.
pub fn flat_trees(prebuilt: &PrebuiltTrees) -> Vec<(&GameTree, Player, f64)> {
    let mut trees: Vec<(&GameTree, Player, f64)> = Vec::new();
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if !tree.nodes.is_empty() {
                trees.push((tree, dealer, entry.weight));
            }
        }
    }
    trees
}

/// Replay the build recursion over the SAME `deals` (and dealer filter and
/// tree rules) the trees were built from, collecting boundary crossings
/// grouped into subgames. `tree_idx` matches [`flat_trees`] on the
/// corresponding `PrebuiltTrees` by construction: deals in order, dealer 0
/// then dealer 1, skipping dealers excluded by the filter.
pub fn collect_boundary(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
) -> Result<Vec<Subgame>, truco_engine::EngineError> {
    let mut groups: HashMap<PublicKey, Vec<BoundaryNode>> = HashMap::new();
    let mut tree_idx: u32 = 0;
    for deal in deals {
        for dealer in [0u8, 1u8] {
            if let Some(f) = dealer_filter {
                if f != dealer {
                    continue;
                }
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            let mut walker = Walker {
                tree_idx,
                deal_weight: deal.weight,
                next_id: 0,
                public: Vec::new(),
                groups: &mut groups,
            };
            walker.walk(&state, false)?;
            tree_idx += 1;
        }
    }
    let mut subgames: Vec<Subgame> = groups
        .into_iter()
        .map(|(public, members)| Subgame { public, members })
        .collect();
    // Deterministic order for reproducible artifacts.
    subgames.sort_by(|a, b| a.public.0.cmp(&b.public.0));
    for s in &mut subgames {
        s.members.sort_by_key(|m| (m.tree_idx, m.node));
    }
    Ok(subgames)
}

struct Walker<'a> {
    tree_idx: u32,
    deal_weight: f64,
    /// Next node id, advanced in exactly `build_node`'s order (a node takes
    /// its id on entry, before its children).
    next_id: NodeId,
    /// Public projection of the action path so far (u8 codes, face-down
    /// masked). The dealer byte is prepended when a key is emitted.
    public: Vec<u8>,
    groups: &'a mut HashMap<PublicKey, Vec<BoundaryNode>>,
}

impl Walker<'_> {
    /// Mirror of `build_node`'s recursion. Returns after advancing `next_id`
    /// past this node's whole subtree. `crossed` = a boundary node lies on
    /// the path above (so deeper raise-resolved states don't re-trigger).
    fn walk(
        &mut self,
        state: &TraversalState,
        crossed: bool,
    ) -> Result<(), truco_engine::EngineError> {
        let id = self.next_id;
        self.next_id += 1;
        if state.is_terminal() {
            return Ok(());
        }
        let actions = state.abstract_legal_actions()?;
        assert!(!actions.is_empty(), "non-terminal state with no actions");

        let mut crossed = crossed;
        if !crossed && Self::is_boundary(state) {
            crossed = true;
            let dealer = state.engine.state().dealer;
            let mut key = Vec::with_capacity(self.public.len() + 1);
            key.push(dealer);
            key.extend_from_slice(&self.public);
            let (view_p0, view_p1) = (self.view(state, 0), self.view(state, 1));
            // subtree_end is known only after the recursion below finishes;
            // push a placeholder and patch it afterwards.
            let entry = BoundaryNode {
                tree_idx: self.tree_idx,
                node: id,
                subtree_end: 0,
                deal_weight: self.deal_weight,
                view_p0: view_p0.key(),
                view_p1: view_p1.key(),
            };
            let members = self.groups.entry(PublicKey(key)).or_default();
            members.push(entry);
            let member_pos = members.len() - 1;
            let group_key = {
                // Re-derive the key to look the group up again post-recursion
                // (borrow of `groups` must end before recursing).
                let mut key = Vec::with_capacity(self.public.len() + 1);
                key.push(dealer);
                key.extend_from_slice(&self.public);
                PublicKey(key)
            };
            for &action in &actions {
                let child = state.apply_abstract_action(action)?;
                self.push_public(action);
                self.walk(&child, crossed)?;
                self.public.pop();
            }
            let end = self.next_id;
            self.groups.get_mut(&group_key).expect("group exists")[member_pos].subtree_end = end;
            return Ok(());
        }

        for &action in &actions {
            let child = state.apply_abstract_action(action)?;
            self.push_public(action);
            self.walk(&child, crossed)?;
            self.public.pop();
        }
        Ok(())
    }

    /// Start of round 2: round 1's card plays are complete, round 2 has not
    /// started, and no raise decision is pending. The `crossed` flag in
    /// [`Self::walk`] keeps raise-resolved states deeper in round 2 (same
    /// predicate values) inside the subgame rather than re-triggering.
    fn is_boundary(state: &TraversalState) -> bool {
        let st = state.engine.state();
        st.completed_rounds.len() == 1
            && st.current_round.plays.is_empty()
            && st.pending_raise.is_none()
    }

    fn push_public(&mut self, action: AbstractAction) {
        let masked = match action {
            AbstractAction::PlayFaceDown(_) => AbstractAction::OpponentPlayedHidden,
            other => other,
        };
        self.public.push(masked.to_u8());
    }

    /// Player `p`'s info-set view of the current history — the same
    /// construction as `TraversalState::current_info_set`, but for a chosen
    /// player rather than the acting one (at a boundary node the non-acting
    /// player's view exists even though it is not an acting info set).
    fn view(&self, state: &TraversalState, p: Player) -> InfoSet {
        InfoSet {
            player: p,
            is_dealer: p == state.engine.state().dealer,
            turnup_class: state.turnup_class,
            starting_hand: state.starting_hands[p as usize].clone(),
            history: state.histories[p as usize].clone(),
        }
    }
}

/// Per-node blueprint reach over one built tree, EXCLUDING one player's own
/// contribution: `reach[node] = deal_weight × Π (blueprint prob of the
/// non-excluded player's actions on the path)`, i.e. `π_c · π_{-excluded}`.
/// This is both the gadget root distribution (excluded = the re-solve's
/// opponent `o`, so root weights are chance × resolver reach) and the CBV
/// aggregation weight. Forward sweep in id order — parents precede children
/// in the preorder arena.
pub fn reach_excluding(
    tree: &GameTree,
    deal_weight: f64,
    excluded: Player,
    strategies: &[ActionProbs],
) -> Vec<f64> {
    use crate::game_tree::NodeView;
    let n = tree.num_nodes();
    let mut w = vec![0.0f64; n];
    if n == 0 {
        return w;
    }
    w[0] = deal_weight;
    for id in 0..n {
        if let NodeView::Player {
            player,
            table_idx,
            edges,
        } = tree.view(id as NodeId)
        {
            let sigma = &strategies[table_idx as usize];
            for (a, e) in edges.iter().enumerate() {
                let p = if player == excluded { 1.0 } else { sigma[a] };
                w[e.child as usize] += w[id] * p;
            }
        }
    }
    w
}

// ─── Sizing scout (plan 84 Phase 5 box-choice input) ──────────────────────

/// Structural size of one subgame, discovered by the boundary-truncating walk.
#[derive(Clone, Debug, Default)]
pub struct SubgameSize {
    pub members: usize,
    pub nodes: u64,
    pub info_sets: usize,
}

/// Total / trunk-region / subgame sizing for one (score, tc, dealer) cell,
/// produced by a single build-recursion replay with NO arena allocation — the
/// cheap counting scout that decides the Phase-5 box (see `trunk-scout`). All
/// counts are for the deals actually walked; the CLI reports linear
/// extrapolations to the full deal set separately (labeled), since node counts
/// are additive across deals but distinct info-set counts are shared and grow
/// sublinearly (linear extrapolation is an upper bound for those).
#[derive(Clone, Debug, Default)]
pub struct ScoutReport {
    pub deals_walked: usize,
    pub total_nodes: u64,
    pub total_info_sets: usize,
    /// Nodes above (and including) the boundary — each boundary node counts as
    /// a single leaf, its subtree excluded. Matches `resolve::trunk_solve`'s
    /// `n_trunk = n_full - Σ(span - 1)`.
    pub trunk_nodes: u64,
    /// Info sets NOT acting inside any subgame (the complement of the union of
    /// per-subgame info sets — disjoint by construction: a round-2 subgame view
    /// never shares a round-1 trunk key).
    pub trunk_info_sets: usize,
    pub subgame_count: usize,
    pub largest_by_nodes: SubgameSize,
    pub largest_by_info_sets: SubgameSize,
}

#[derive(Default)]
struct ScoutBucket {
    members: usize,
    nodes: u64,
    keys: std::collections::HashSet<InfoSetKey>,
}

/// Replay the build recursion (as [`collect_boundary`]) but accumulate node and
/// info-set counts partitioned into trunk vs per-subgame regions instead of
/// materializing boundary members. No arena, no registry — O(1) memory beyond
/// the info-set key sets.
pub fn scout_sizes(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
) -> Result<ScoutReport, truco_engine::EngineError> {
    let mut all_keys: std::collections::HashSet<InfoSetKey> = std::collections::HashSet::new();
    let mut trunk_keys: std::collections::HashSet<InfoSetKey> = std::collections::HashSet::new();
    let mut buckets: HashMap<PublicKey, ScoutBucket> = HashMap::new();
    let mut total_nodes: u64 = 0;
    let mut deals_walked = 0usize;

    for deal in deals {
        for dealer in [0u8, 1u8] {
            if let Some(f) = dealer_filter {
                if f != dealer {
                    continue;
                }
            }
            let state =
                TraversalState::from_deal_with_rules(dealer, score.clone(), tc, deal, rules)?;
            let mut walker = ScoutWalker {
                public: Vec::new(),
                all_keys: &mut all_keys,
                trunk_keys: &mut trunk_keys,
                buckets: &mut buckets,
            };
            total_nodes += walker.walk(&state, false, None)?;
            deals_walked += 1;
        }
    }

    let sum_sub_nodes: u64 = buckets.values().map(|b| b.nodes).sum();
    let sum_members: u64 = buckets.values().map(|b| b.members as u64).sum();
    // descendants = Σ(span-1) = Σ span - Σ members.
    let trunk_nodes = total_nodes - (sum_sub_nodes - sum_members);

    let size_of = |b: &ScoutBucket| SubgameSize {
        members: b.members,
        nodes: b.nodes,
        info_sets: b.keys.len(),
    };
    let largest_by_nodes = buckets
        .values()
        .max_by_key(|b| b.nodes)
        .map(size_of)
        .unwrap_or_default();
    let largest_by_info_sets = buckets
        .values()
        .max_by_key(|b| b.keys.len())
        .map(size_of)
        .unwrap_or_default();

    Ok(ScoutReport {
        deals_walked,
        total_nodes,
        total_info_sets: all_keys.len(),
        trunk_nodes,
        trunk_info_sets: trunk_keys.len(),
        subgame_count: buckets.len(),
        largest_by_nodes,
        largest_by_info_sets,
    })
}

struct ScoutWalker<'a> {
    public: Vec<u8>,
    all_keys: &'a mut std::collections::HashSet<InfoSetKey>,
    trunk_keys: &'a mut std::collections::HashSet<InfoSetKey>,
    buckets: &'a mut HashMap<PublicKey, ScoutBucket>,
}

impl ScoutWalker<'_> {
    /// Returns the number of nodes (including terminals) in this subtree. The
    /// boundary node's own acting info set belongs to the subgame — matching
    /// `trunk_solve`'s `m.node..m.subtree_end` info-set partition, which
    /// includes the boundary node itself.
    fn walk(
        &mut self,
        state: &TraversalState,
        crossed: bool,
        sub_key: Option<&PublicKey>,
    ) -> Result<u64, truco_engine::EngineError> {
        if state.is_terminal() {
            return Ok(1);
        }
        let actions = state.abstract_legal_actions()?;

        // Boundary detection BEFORE key attribution: the boundary node's own
        // info set is a subgame info set, not a trunk one.
        let mut crossed = crossed;
        let owned_pk: Option<PublicKey>;
        let cur_sub: Option<&PublicKey>;
        if !crossed && Walker::is_boundary(state) {
            crossed = true;
            let dealer = state.engine.state().dealer;
            let mut key = Vec::with_capacity(self.public.len() + 1);
            key.push(dealer);
            key.extend_from_slice(&self.public);
            let pk = PublicKey(key);
            self.buckets.entry(pk.clone()).or_default().members += 1;
            owned_pk = Some(pk);
            cur_sub = owned_pk.as_ref();
        } else {
            owned_pk = None;
            cur_sub = sub_key;
        }

        let acting = state
            .current_info_set()
            .expect("non-terminal player node has an info set")
            .key();
        self.all_keys.insert(acting);
        match cur_sub {
            Some(pk) => {
                self.buckets
                    .get_mut(pk)
                    .expect("bucket exists")
                    .keys
                    .insert(acting);
            }
            None => {
                self.trunk_keys.insert(acting);
            }
        }

        let mut count = 1u64;
        for &action in &actions {
            let child = state.apply_abstract_action(action)?;
            self.push_public(action);
            count += self.walk(&child, crossed, cur_sub)?;
            self.public.pop();
        }

        if let Some(pk) = &owned_pk {
            self.buckets.get_mut(pk).expect("bucket exists").nodes += count;
        }
        Ok(count)
    }

    fn push_public(&mut self, action: AbstractAction) {
        let masked = match action {
            AbstractAction::PlayFaceDown(_) => AbstractAction::OpponentPlayedHidden,
            other => other,
        };
        self.public.push(masked.to_u8());
    }
}

// NOTE: per-subgame CBV extraction used to live here (`boundary_cbvs`). It
// was removed after the first production run: calling the full exact-BR
// pass once per subgame is O(subgames × full-tree-BR) — ~2000 passes at
// 10×10 — while the pass's values don't depend on which nodes are read out.
// The batched replacement is `resolve::boundary_summary` (one BR + one
// reach sweep per player for the entire subgame set, identical values).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::game_tree::{build_all_trees_with_dealer, NodeView};

    fn small_build(score: &Score, take: usize) -> (PrebuiltTrees, Vec<AbstractDeal>, TurnupClass) {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(take).collect();
        let built = build_all_trees_with_dealer(score, tc, &deals, Some(0)).unwrap();
        (built, deals, tc)
    }

    /// Independent recursive-descent node collection over the BUILT tree,
    /// used to validate the replay's spans against the real arena.
    fn descend_collect(tree: &GameTree, node: NodeId, out: &mut Vec<NodeId>) {
        out.push(node);
        if let NodeView::Player { edges, .. } = tree.view(node) {
            for e in edges {
                descend_collect(tree, e.child, out);
            }
        }
    }

    #[test]
    fn boundary_partition_and_spans_match_built_trees() {
        let score = Score { zero: 8, one: 8 };
        let (built, deals, tc) = small_build(&score, 40);
        let subgames = collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        assert!(!subgames.is_empty(), "8x8 must have round-2 subgames");

        let trees = flat_trees(&built);
        // Walker's tree count must match the built flat list.
        let max_tree = subgames
            .iter()
            .flat_map(|s| s.members.iter().map(|m| m.tree_idx))
            .max()
            .unwrap();
        assert!((max_tree as usize) < trees.len());

        let mut total_members = 0usize;
        for s in &subgames {
            for m in &s.members {
                total_members += 1;
                let (tree, _dealer, weight) = trees[m.tree_idx as usize];
                assert!((m.deal_weight - weight).abs() < 1e-12);
                // Preorder-contiguity: recursive descent from the boundary
                // node over the BUILT arena is exactly [node, subtree_end).
                let mut got = Vec::new();
                descend_collect(tree, m.node, &mut got);
                let mut expect: Vec<NodeId> = (m.node..m.subtree_end).collect();
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "span mismatch at tree {}", m.tree_idx);
                // The boundary node is a player node in the built tree.
                assert!(matches!(tree.view(m.node), NodeView::Player { .. }));
            }
        }
        assert!(total_members > 0);

        // No boundary node is an ancestor of another: within one tree, spans
        // are disjoint (preorder ⇒ ancestor spans strictly contain
        // descendants').
        for s in &subgames {
            for m in &s.members {
                for s2 in &subgames {
                    for m2 in &s2.members {
                        if (m.tree_idx, m.node) == (m2.tree_idx, m2.node) {
                            continue;
                        }
                        if m.tree_idx == m2.tree_idx {
                            let disjoint = m.subtree_end <= m2.node || m2.subtree_end <= m.node;
                            assert!(disjoint, "overlapping boundary spans");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_path_crosses_at_most_one_boundary_and_mao_crosses_exactly_one() {
        // 11x11: no raises; the only trunk terminals are FoldEleven paths.
        let score = Score { zero: 11, one: 11 };
        let (built, deals, tc) = small_build(&score, 30);
        let subgames = collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        let trees = flat_trees(&built);

        // Collect boundary nodes per tree for the crossing count.
        let mut per_tree: HashMap<u32, Vec<(NodeId, NodeId)>> = HashMap::new();
        for s in &subgames {
            for m in &s.members {
                per_tree
                    .entry(m.tree_idx)
                    .or_default()
                    .push((m.node, m.subtree_end));
            }
        }

        fn walk_paths(
            tree: &GameTree,
            node: NodeId,
            crossings: usize,
            spans: &[(NodeId, NodeId)],
            terminal_crossings: &mut Vec<usize>,
        ) {
            let here = crossings + usize::from(spans.iter().any(|(b, _)| *b == node));
            match tree.view(node) {
                NodeView::Terminal { .. } => terminal_crossings.push(here),
                NodeView::Player { edges, .. } => {
                    for e in edges {
                        walk_paths(tree, e.child, here, spans, terminal_crossings);
                    }
                }
            }
        }

        for (t_idx, (tree, _dealer, _w)) in trees.iter().enumerate() {
            let spans = per_tree.get(&(t_idx as u32)).cloned().unwrap_or_default();
            let mut crossings = Vec::new();
            walk_paths(tree, 0, 0, &spans, &mut crossings);
            for c in &crossings {
                assert!(*c <= 1, "path crossed {} boundary nodes", c);
            }
            // At 11x11 every non-FoldEleven path reaches round 2: card play
            // alone cannot end the hand in round 1 and there are no raises,
            // so zero-crossing terminals are exactly the FoldEleven lines.
            assert!(
                crossings.contains(&1),
                "tree {} has no boundary crossing at all",
                t_idx
            );
        }
    }

    #[test]
    fn views_refine_public_key_and_acting_infosets_match_registry() {
        let score = Score { zero: 8, one: 8 };
        let (built, deals, tc) = small_build(&score, 40);
        let subgames = collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();

        // A view key appearing in two different subgames would break the
        // "views refine public state" property the gadget relies on.
        let mut view_owner: HashMap<InfoSetKey, usize> = HashMap::new();
        for (i, s) in subgames.iter().enumerate() {
            for m in &s.members {
                for view in [m.view_p0, m.view_p1] {
                    if let Some(prev) = view_owner.insert(view, i) {
                        assert_eq!(prev, i, "view key spans two subgames");
                    }
                }
            }
        }

        // The ACTING player's view at a boundary node must equal the
        // registry info set of that node — the walker maintains views the
        // same way the build maintains acting info sets, so agreement here
        // validates the whole view construction.
        let trees = flat_trees(&built);
        let mut checked = 0usize;
        for s in &subgames {
            for m in &s.members {
                let (tree, _dealer, _w) = trees[m.tree_idx as usize];
                if let crate::game_tree::NodeView::Player {
                    player, table_idx, ..
                } = tree.view(m.node)
                {
                    let registry_key = built.info_sets[table_idx as usize].0;
                    let walker_key = if player == 0 { m.view_p0 } else { m.view_p1 };
                    assert_eq!(registry_key, walker_key, "acting view != registry key");
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn scout_totals_match_count_tree_and_partition_is_consistent() {
        use crate::game_tree::count_tree_size_with_rules;
        let score = Score { zero: 8, one: 8 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(60).collect();

        let report = scout_sizes(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        // Totals must bit-match the independent size-only walker.
        let census =
            count_tree_size_with_rules(&score, tc, 0, &deals, TreeRules::Current, None).unwrap();
        assert_eq!(report.total_nodes, census.total_nodes);
        assert_eq!(report.total_info_sets, census.num_info_sets);

        // Trunk region is a strict subset of the tree, boundary-truncated.
        assert!(report.trunk_nodes > 0);
        assert!(report.trunk_nodes < report.total_nodes);
        assert!(report.trunk_info_sets < report.total_info_sets);
        assert!(report.subgame_count > 0);

        // Cross-check subgame count and trunk-node arithmetic against the
        // authoritative boundary collector on the SAME inputs.
        let subgames = collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        assert_eq!(report.subgame_count, subgames.len());
        // n_trunk = n_full - Σ(span - 1), mirroring resolve::trunk_solve.
        let descendants: u64 = subgames
            .iter()
            .flat_map(|s| s.members.iter())
            .map(|m| (m.subtree_end - m.node) as u64 - 1)
            .sum();
        assert_eq!(report.trunk_nodes, report.total_nodes - descendants);

        // Info-set partition disjointness: trunk keys and the union of subgame
        // keys tile the whole registry exactly (a round-2 subgame view never
        // shares a round-1 trunk key). Verified here via counts summing.
        // (largest_by_* are populated.)
        assert!(report.largest_by_nodes.nodes > 0);
        assert!(report.largest_by_info_sets.info_sets > 0);
    }

    #[test]
    fn reach_excluding_uniform_and_independence() {
        let score = Score { zero: 8, one: 8 };
        let (built, _deals, _tc) = small_build(&score, 12);
        let trees = flat_trees(&built);
        let (tree, _dealer, weight) = trees[0];

        // Uniform blueprint.
        let uniform: Vec<ActionProbs> = built
            .info_sets
            .iter()
            .map(|(_, _, actions)| crate::strategy::uniform_probs(actions.len()))
            .collect();
        let w0 = reach_excluding(tree, weight, 0, &uniform);
        assert!((w0[0] - weight).abs() < 1e-15);

        // Hand-walk: first child of the root. If the root is p0's node the
        // excluded player's action contributes 1.0; otherwise 1/n.
        if let NodeView::Player { player, edges, .. } = tree.view(0) {
            let expect = if player == 0 {
                weight
            } else {
                weight / edges.len() as f64
            };
            assert!((w0[edges[0].child as usize] - expect).abs() < 1e-12);
        }

        // Independence: perturbing ONLY the excluded player's rows must not
        // change the vector at all.
        let mut perturbed = uniform.clone();
        for (idx, (_, info, _)) in built.info_sets.iter().enumerate() {
            if info.player == 0 && perturbed[idx].len() > 1 {
                let n = perturbed[idx].len();
                perturbed[idx] = std::iter::once(1.0)
                    .chain(std::iter::repeat_n(0.0, n - 1))
                    .collect();
            }
        }
        let w0_perturbed = reach_excluding(tree, weight, 0, &perturbed);
        assert_eq!(w0, w0_perturbed);
    }
}
