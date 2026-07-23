//! Deep-cell CFR-D solve (plan 84 Phase 5).
//!
//! A memory-decomposed rewrite of [`crate::resolve::trunk_solve`] for cells too
//! big to hold in one arena (0×0: 3.81 B nodes, ~757 M info sets, 3,135
//! subgames). The full-arena `trunk_solve` shares ONE global `PrebuiltTrees` /
//! `table_idx` / accumulator across trunk, subgames, CBV folds, and
//! certification. This path instead:
//!
//! 1. Builds a TRUNK-ONLY arena ([`crate::game_tree::build_trunk_arena`]) where
//!    the round-2 boundary is a truncated leaf — the trunk is ~1% of the nodes.
//! 2. Builds each subgame's subtree ON DEMAND with a FRESH LOCAL registry
//!    ([`crate::game_tree::build_subgame_local`]); that local registry is the
//!    subgame's compact accumulator index. Per-subgame [`cfr::DenseAccum`]s stay
//!    alive across rounds (their sum ≈ one full accumulator; ~20 GB at 0×0 with
//!    `accum-f32`). The arenas are rebuilt per round by default (`keep_arenas`
//!    off) or kept if memory allows.
//! 3. Certifies by a decomposed, streaming best response: each subgame is
//!    best-responded independently (its conditional boundary value injected into
//!    the trunk BR), so the whole-game exploitability is assembled without ever
//!    materializing the full arena — yet bit-identical to the full-arena BR on
//!    the composed profile.
//!
//! Every intra-round coupling reads the CURRENT regret-matching iterate (the
//! 2026-07-21 plateau lesson); the arithmetic per node is identical to the
//! full-arena path, so `--jobs 1` reproduces `trunk_solve`'s certificates.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use truco_engine::{Player, Score};

use crate::abstraction::{AbstractDeal, TurnupClass};
use crate::cfr::{self, ResolveMember, SubgameRoot};
use crate::game_tree::{self, PrebuiltTrees, TraversalState, TreeRules, TrunkCrossing};
use crate::info_set::InfoSetKey;
use crate::resolve::SubgameStats;
use crate::strategy::ActionProbs;
use crate::subgame;

const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Schedule / resource knobs for [`deep_solve`].
#[derive(Clone, Copy, Debug)]
pub struct DeepConfig {
    pub rounds: usize,
    pub trunk_iters: u64,
    pub subgame_iters: u64,
    pub final_iters: u64,
    pub baseline_iters: u64,
    /// Subgame-parallel worker count (trunk sweep stays single-threaded).
    pub jobs: usize,
    /// Keep each subgame's local arena alive across rounds (else rebuild it from
    /// the captured boundary state every round — the default, lower peak RSS).
    pub keep_arenas: bool,
    /// Run the streaming certificate (`false` = report `NaN` eps, skip BR work).
    pub certify: bool,
    /// Also run the two gadget recoveries + their certificates (`--certify
    /// full`). `false` (`--certify raw`) certifies only the raw profile —
    /// the cheap benchmark mode.
    pub certify_recoveries: bool,
}

/// From-scratch deep solve report — mirrors [`crate::resolve::TrunkReport`].
#[derive(Debug)]
pub struct DeepReport {
    pub raw_eps: f64,
    pub composed_eps_tail: f64,
    pub composed_eps_br: f64,
    pub composed_eps: f64,
    pub game_value_per_dealer: (f64, f64),
    pub game_value: f64,
    pub subgames: usize,
    pub per_subgame: Vec<SubgameStats>,
    pub largest: SubgameStats,
    pub trunk_visits: u64,
    pub subgame_visits: u64,
    pub final_visits: u64,
    pub total_visits: u64,
    pub baseline_visits: u64,
    pub multiplier: f64,
    /// Trunk arena info-set count (the resident registry — the memory headline).
    pub trunk_info_sets: usize,
    /// Sum of per-subgame local info-set counts (≈ one full registry).
    pub subgame_info_sets: usize,
}

/// Replay context for materializing boundary states from compact seeds.
/// The 2026-07-22 OOM lesson: holding a `TraversalState` per crossing is
/// ~78 GB at 0×0; seeds are ~40 bytes/crossing and replay is trivial next
/// to sweep cost.
#[derive(Clone, Copy)]
pub(crate) struct ReplayCtx<'a> {
    pub score: &'a Score,
    pub tc: TurnupClass,
    pub deals: &'a [AbstractDeal],
    pub rules: TreeRules,
}

impl ReplayCtx<'_> {
    /// Materialize `(state, deal_weight)` pairs for one subgame's members.
    fn materialize(
        &self,
        seeds: &[(u32, Box<[u8]>, f64)],
        dealer: Player,
    ) -> Vec<(TraversalState, f64)> {
        seeds
            .iter()
            .map(|(deal_idx, path, w)| {
                let st = game_tree::replay_crossing_state(
                    self.score,
                    self.tc,
                    &self.deals[*deal_idx as usize],
                    dealer,
                    self.rules,
                    path,
                )
                .expect("replay boundary state");
                (st, *w)
            })
            .collect()
    }
}

/// One subgame's persistent solve state. `seeds` regenerate boundary states
/// for on-demand local builds; `dense` is the warm accumulator carried
/// across rounds.
struct SubgameState {
    dealer: Player,
    /// Member metadata, indexed by local flat tree index (build order).
    trunk_tree_idx: Vec<u32>,
    trunk_node: Vec<game_tree::NodeId>,
    deal_weight: Vec<f64>,
    view: Vec<[InfoSetKey; 2]>,
    /// Compact replay seeds `(deal_idx, action path, deal_weight)` — member
    /// order; rebuild the local arena via [`ReplayCtx::materialize`].
    seeds: Vec<(u32, Box<[u8]>, f64)>,
    /// Local info-set count and key→local-index map (stable across rebuilds).
    n_local: usize,
    key_to_local: HashMap<InfoSetKey, u32>,
    /// Local node count of each member subtree (for [`ResolveMember`] spans).
    member_nodes: Vec<usize>,
    /// Warm accumulator across rounds.
    dense: cfr::DenseAccum,
    /// Cached local arena when `keep_arenas`.
    arena: Option<PrebuiltTrees>,
    /// Structural stats (for the report / largest).
    stats: SubgameStats,
}

/// Borrow the cached arena (kept mode) or hold a freshly built one — avoids a
/// `Clone` bound on the (large, `Arc`-backed) `PrebuiltTrees`.
enum ArenaRef<'a> {
    Borrowed(&'a PrebuiltTrees),
    Owned(PrebuiltTrees),
}

impl ArenaRef<'_> {
    fn get(&self) -> &PrebuiltTrees {
        match self {
            ArenaRef::Borrowed(p) => p,
            ArenaRef::Owned(p) => p,
        }
    }
}

impl SubgameState {
    /// Local arena for this round: cached if kept, else replayed + built.
    fn arena(&self, keep: bool, ctx: ReplayCtx<'_>) -> ArenaRef<'_> {
        if keep {
            ArenaRef::Borrowed(self.arena.as_ref().expect("kept arena"))
        } else {
            let states = ctx.materialize(&self.seeds, self.dealer);
            let (pb, _) =
                game_tree::build_subgame_local(&states, self.dealer).expect("subgame build");
            ArenaRef::Owned(pb)
        }
    }
}

/// Per-round contributions a subgame hands back to the single-threaded reducer.
struct RoundContribs {
    /// `(trunk_tree_idx, trunk_node, v0)` boundary values (p0 perspective).
    boundary: Vec<(u32, game_tree::NodeId, f64)>,
    /// Per opponent `o`, `(view, weight, weight * v_o)` CBV fold terms.
    cbv: [Vec<(InfoSetKey, f64, f64)>; 2],
}

/// Group the trunk arena's boundary crossings into subgames (by public key),
/// deterministically ordered exactly like [`subgame::collect_boundary`].
fn group_subgames(crossings: Vec<TrunkCrossing>) -> Vec<Vec<TrunkCrossing>> {
    let mut groups: HashMap<Vec<u8>, Vec<TrunkCrossing>> = HashMap::new();
    for c in crossings {
        groups.entry(c.public_key.clone()).or_default().push(c);
    }
    let mut keyed: Vec<(Vec<u8>, Vec<TrunkCrossing>)> = groups.into_iter().collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, members) in &mut keyed {
        members.sort_by_key(|m| (m.tree_idx, m.node));
    }
    keyed.into_iter().map(|(_, m)| m).collect()
}

/// Build the persistent [`SubgameState`] for one subgame's crossings.
fn init_subgame(
    members: Vec<TrunkCrossing>,
    keep_arenas: bool,
    ctx: ReplayCtx<'_>,
) -> SubgameState {
    let dealer = members[0].dealer;
    let seeds: Vec<(u32, Box<[u8]>, f64)> = members
        .iter()
        .map(|m| (m.deal_idx, m.path.clone(), m.deal_weight))
        .collect();
    let states = ctx.materialize(&seeds, dealer);
    let (arena, key_to_local) =
        game_tree::build_subgame_local(&states, dealer).expect("subgame build");
    let n_local = arena.info_sets.len();
    let dense = cfr::new_resolve_accum(&arena);
    let trees = subgame::flat_trees(&arena);
    let member_nodes: Vec<usize> = (0..members.len()).map(|i| trees[i].0.num_nodes()).collect();
    // Structural stats: node span = local tree nodes; info sets = local count.
    let nodes: usize = member_nodes.iter().sum();
    let stats = SubgameStats {
        members: members.len(),
        nodes,
        info_sets: n_local,
    };
    SubgameState {
        dealer,
        trunk_tree_idx: members.iter().map(|m| m.tree_idx).collect(),
        trunk_node: members.iter().map(|m| m.node).collect(),
        deal_weight: members.iter().map(|m| m.deal_weight).collect(),
        view: members.iter().map(|m| [m.view_p0, m.view_p1]).collect(),
        seeds,
        n_local,
        key_to_local,
        member_nodes,
        dense,
        arena: if keep_arenas { Some(arena) } else { None },
        stats,
    }
}

/// `reach_excluding(excluded=0)` and `reach_excluding(excluded=1)` at every node
/// of each member trunk tree, keyed by trunk tree idx. `reach[tree][excluded]
/// [node]` = Π of the NON-excluded player's `rows` probs along the path.
///
/// `deal_weight_init` chooses the initial reach so the FLOATING-POINT order
/// matches `trunk_solve`: subgame ROOTS want the deal-weight factored out
/// (init 1, chance applied separately by the traversal — `reach_excluding(_,
/// 1.0, _)`), while CBV/BR weights want it folded IN (`reach_excluding(_,
/// deal_weight, _)`). Folding in vs. multiplying `π × dw` afterward differ by
/// one rounding, which is invisible to the continuous gadget solve but can flip
/// a near-tied best-response argmax — so this must mirror `trunk_solve` exactly.
fn trunk_reach(
    trunk: &PrebuiltTrees,
    rows: &[ActionProbs],
    tree_idxs: &std::collections::HashSet<u32>,
    deal_weight_init: bool,
) -> HashMap<u32, [Vec<f64>; 2]> {
    let trees = subgame::flat_trees(trunk);
    let mut out = HashMap::with_capacity(tree_idxs.len());
    for &t in tree_idxs {
        let (tree, _d, w) = trees[t as usize];
        let init = if deal_weight_init { w } else { 1.0 };
        let e0 = subgame::reach_excluding(tree, init, 0, rows);
        let e1 = subgame::reach_excluding(tree, init, 1, rows);
        out.insert(t, [e0, e1]);
    }
    out
}

/// Materialize a dense accumulator's average rows as `Vec<ActionProbs>`.
fn average_rows(dense: &cfr::DenseAccum, n: usize) -> Vec<ActionProbs> {
    (0..n).map(|i| dense.average_row(i)).collect()
}

fn current_rows(dense: &cfr::DenseAccum, n: usize) -> Vec<ActionProbs> {
    (0..n).map(|i| dense.current_row(i)).collect()
}

/// Build the per-flat-tree sorted `(node, v0)` boundary list for the trunk hook.
fn build_boundary_per_tree(
    n_trees: usize,
    flat: &[(u32, game_tree::NodeId, f64)],
) -> Vec<Vec<(game_tree::NodeId, f64)>> {
    let mut per_tree = vec![Vec::new(); n_trees];
    for &(t, node, v) in flat {
        per_tree[t as usize].push((node, v));
    }
    for v in &mut per_tree {
        v.sort_by_key(|&(n, _)| n);
    }
    per_tree
}

/// One subgame's per-round work: refresh roots from the current trunk iterate,
/// run the gadget-free subgame sweeps, then read out boundary values + CBV fold
/// terms under the resulting current iterate. Pure w.r.t. other subgames (its
/// `dense` and local arena are disjoint), so it parallelizes cleanly.
#[allow(clippy::too_many_arguments)]
fn subgame_round(
    sg: &mut SubgameState,
    keep_arenas: bool,
    reach_unit: &HashMap<u32, [Vec<f64>; 2]>,
    reach_dw: &HashMap<u32, [Vec<f64>; 2]>,
    sub_iter_base: u64,
    subgame_iters: u64,
    fold_cbv: bool,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
    ctx: ReplayCtx<'_>,
) -> RoundContribs {
    // Disjoint field borrows: `arena` reads `sg.arena`/`sg.seeds`, the sweep
    // mutates `sg.dense` — accessed as separate fields (not via `&self`).
    let owned_arena;
    let arena: &PrebuiltTrees = if keep_arenas {
        sg.arena.as_ref().expect("kept arena")
    } else {
        let states = ctx.materialize(&sg.seeds, sg.dealer);
        owned_arena = game_tree::build_subgame_local(&states, sg.dealer)
            .expect("subgame build")
            .0;
        &owned_arena
    };
    let n_members = sg.trunk_node.len();
    let subgame_infosets: Vec<u32> = (0..sg.n_local as u32).collect();

    // Roots from the CURRENT trunk iterate: each member enters with its own
    // trunk reach split (π_0, π_1) and deal-weight chance.
    let roots: Vec<SubgameRoot> = (0..n_members)
        .map(|i| {
            let t = sg.trunk_tree_idx[i];
            let node = sg.trunk_node[i] as usize;
            let r = &reach_unit[&t];
            SubgameRoot {
                tree_idx: i as u32,
                node: 0,
                // reach[excluded=1] = π_0 ; reach[excluded=0] = π_1.
                reach: [r[1][node], r[0][node]],
                chance: sg.deal_weight[i],
            }
        })
        .collect();

    for k in 0..subgame_iters {
        cfr::subgame_root_iteration(
            arena,
            &mut sg.dense,
            &roots,
            &subgame_infosets,
            sub_iter_base + k + 1,
            score,
            match_values,
        );
    }

    // Boundary values under the current subgame iterate (p0 perspective).
    let cur = current_rows(&sg.dense, sg.n_local);
    let member_nodes: Vec<(u32, game_tree::NodeId)> =
        (0..n_members).map(|i| (i as u32, 0)).collect();
    let bvals = cfr::profile_boundary_values(arena, &cur, score, match_values, &member_nodes);

    let mut boundary = Vec::with_capacity(n_members);
    for i in 0..n_members {
        boundary.push((sg.trunk_tree_idx[i], sg.trunk_node[i], bvals[i]));
    }

    let mut cbv: [Vec<(InfoSetKey, f64, f64)>; 2] = [Vec::new(), Vec::new()];
    if fold_cbv {
        for o in [0u8, 1u8] {
            for i in 0..n_members {
                let t = sg.trunk_tree_idx[i];
                let node = sg.trunk_node[i] as usize;
                // Deal weight folded INTO the reach init (matches trunk_solve's
                // reach_excluding(_, deal_weight, _) fp order — see trunk_reach).
                let weight = reach_dw[&t][o as usize][node];
                if weight > 0.0 {
                    let v_o = if o == 0 { bvals[i] } else { -bvals[i] };
                    cbv[o as usize].push((sg.view[i][o as usize], weight, weight * v_o));
                }
            }
        }
    }

    RoundContribs { boundary, cbv }
}

/// Composed subgame rows for one recovery (raw / gadget-recovered), local-idx
/// order — one per subgame.
type SubgameRows = Vec<Vec<ActionProbs>>;

/// Certify a composed profile (trunk rows + per-subgame rows) by the decomposed
/// streaming best response. Returns exploitability, bit-identical to the
/// full-arena BR on the same composed profile.
#[allow(clippy::too_many_arguments)]
fn deep_exploitability(
    trunk: &PrebuiltTrees,
    trunk_rows: &[ActionProbs],
    subgames: &[SubgameState],
    keep_arenas: bool,
    sg_rows: &SubgameRows,
    trunk_avg_reach: &HashMap<u32, [Vec<f64>; 2]>,
    n_trees: usize,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
    ctx: ReplayCtx<'_>,
) -> f64 {
    // Per-tree boundary injection for each BR player. Each subgame arena is
    // built once (outer loop) so the streaming certificate never holds more
    // than one subgame arena at a time.
    let mut inject: [Vec<Vec<(game_tree::NodeId, f64)>>; 2] =
        [vec![Vec::new(); n_trees], vec![Vec::new(); n_trees]];
    for (si, sg) in subgames.iter().enumerate() {
        let arena = sg.arena(keep_arenas, ctx);
        let a = arena.get();
        let n_members = sg.trunk_node.len();
        let member_nodes: Vec<(u32, game_tree::NodeId)> =
            (0..n_members).map(|i| (i as u32, 0)).collect();
        for br_player in [0u8, 1u8] {
            // Member root weights = opponent trunk reach into the boundary
            // (reach_excluding(excluded=br) × deal weight, folded into init).
            let root_w: Vec<f64> = (0..n_members)
                .map(|i| {
                    trunk_avg_reach[&sg.trunk_tree_idx[i]][br_player as usize]
                        [sg.trunk_node[i] as usize]
                })
                .collect();
            let vals = cfr::best_response_boundary_values_profile(
                a,
                &sg_rows[si],
                score,
                match_values,
                br_player,
                &member_nodes,
                Some(&root_w),
            );
            for i in 0..n_members {
                inject[br_player as usize][sg.trunk_tree_idx[i] as usize]
                    .push((sg.trunk_node[i], vals[i]));
            }
        }
    }
    let mut br = [0.0f64; 2];
    for br_player in [0u8, 1u8] {
        for v in &mut inject[br_player as usize] {
            v.sort_by_key(|&(n, _)| n);
        }
        br[br_player as usize] = cfr::best_response_value_with_boundary_inject(
            trunk,
            &trunk_rows.to_vec(),
            score,
            match_values,
            br_player,
            &inject[br_player as usize],
        );
    }
    (br[0] + br[1]) / 2.0
}

/// p0-perspective composed subgame boundary values (for the decomposed game
/// value), per flat trunk tree, sorted.
fn composed_boundary_p0(
    subgames: &[SubgameState],
    keep_arenas: bool,
    sg_rows: &SubgameRows,
    n_trees: usize,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
    ctx: ReplayCtx<'_>,
) -> Vec<Vec<(game_tree::NodeId, f64)>> {
    let mut per_tree = vec![Vec::new(); n_trees];
    for (si, sg) in subgames.iter().enumerate() {
        let arena = sg.arena(keep_arenas, ctx);
        let n_members = sg.trunk_node.len();
        let member_nodes: Vec<(u32, game_tree::NodeId)> =
            (0..n_members).map(|i| (i as u32, 0)).collect();
        let bvals = cfr::profile_boundary_values(
            arena.get(),
            &sg_rows[si],
            score,
            match_values,
            &member_nodes,
        );
        for i in 0..n_members {
            per_tree[sg.trunk_tree_idx[i] as usize].push((sg.trunk_node[i], bvals[i]));
        }
    }
    for v in &mut per_tree {
        v.sort_by_key(|&(n, _)| n);
    }
    per_tree
}

// ─── Checkpoint ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct DeepCheckpoint {
    format_version: u32,
    /// Next round to execute (rounds `0..next_round` are complete).
    next_round: usize,
    trunk_iter: u64,
    sub_iter_base: u64,
    // Config echo (asserted on resume).
    score: (u8, u8),
    tc_level: u8,
    dealer: u8,
    deal_count: usize,
    rounds: usize,
    trunk_iters: u64,
    subgame_iters: u64,
    warmup: usize,
    subgame_count: usize,
    // Trunk accumulator (always f64 on disk).
    trunk_regret: Vec<f64>,
    trunk_strategy: Vec<f64>,
    // Per-subgame accumulators in subgame-index order.
    sg_regret: Vec<Vec<f64>>,
    sg_strategy: Vec<Vec<f64>>,
    // Running CBV numerator/denominator per opponent.
    cbv_num: [Vec<(u64, f64)>; 2],
    cbv_den: [Vec<(u64, f64)>; 2],
    // Boundary values feeding the next round's trunk phase.
    boundary_flat: Vec<(u32, u32, f64)>,
}

fn atomic_write(path: &Path, ckpt: &DeepCheckpoint) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = path.with_extension("deeptmp");
    {
        let file = std::fs::File::create(&tmp)?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);
        bincode::serialize_into(&mut w, ckpt).map_err(|e| std::io::Error::other(e.to_string()))?;
        use std::io::Write;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}

fn load_checkpoint(path: &Path) -> std::io::Result<DeepCheckpoint> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::with_capacity(4 * 1024 * 1024, file);
    bincode::deserialize_from(reader).map_err(|e| std::io::Error::other(e.to_string()))
}

fn cbv_map_to_vec(m: &HashMap<InfoSetKey, f64>) -> Vec<(u64, f64)> {
    let mut v: Vec<(u64, f64)> = m.iter().map(|(k, val)| (k.0, *val)).collect();
    v.sort_by_key(|&(k, _)| k);
    v
}

fn cbv_vec_to_map(v: &[(u64, f64)]) -> HashMap<InfoSetKey, f64> {
    v.iter().map(|&(k, val)| (InfoSetKey(k), val)).collect()
}

// ─── Main driver ───────────────────────────────────────────────────────────

/// Optional checkpoint control for [`deep_solve`].
#[derive(Clone)]
pub struct CheckpointCfg {
    pub path: std::path::PathBuf,
    pub every: usize,
    pub resume: bool,
    /// Stop after this many rounds are complete (a chunked run): write a
    /// checkpoint and return early with a `NaN` report — the final chunk
    /// (`None`) runs recovery/certification. `None` runs to `cfg.rounds`.
    pub stop_after: Option<usize>,
}

fn nan_report(subgames: usize) -> DeepReport {
    DeepReport {
        raw_eps: f64::NAN,
        composed_eps_tail: f64::NAN,
        composed_eps_br: f64::NAN,
        composed_eps: f64::NAN,
        game_value_per_dealer: (f64::NAN, f64::NAN),
        game_value: f64::NAN,
        subgames,
        per_subgame: Vec::new(),
        largest: SubgameStats::default(),
        trunk_visits: 0,
        subgame_visits: 0,
        final_visits: 0,
        total_visits: 0,
        baseline_visits: 0,
        multiplier: f64::NAN,
        trunk_info_sets: 0,
        subgame_info_sets: 0,
    }
}

/// From-scratch deep CFR-D solve. `deals` / `dealer` / `rules` must be the
/// inputs the caller intends (the trunk arena is built here). Returns the
/// composed profile keyed by info-set key (subgame rows override the inert
/// trunk boundary roots) plus the report.
#[allow(clippy::too_many_arguments)]
pub fn deep_solve(
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer: Player,
    rules: TreeRules,
    match_values: &crate::match_value::MatchValueTable,
    cfg: DeepConfig,
    checkpoint: Option<&CheckpointCfg>,
) -> (DeepReport, HashMap<u64, ActionProbs>) {
    let t0 = std::time::Instant::now();
    // Optional bounded worker pool (jobs==1 → run on the calling thread).
    let pool = if cfg.jobs > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(cfg.jobs)
                .build()
                .expect("rayon pool"),
        )
    } else {
        None
    };

    let ctx = ReplayCtx {
        score,
        tc,
        deals,
        rules,
    };

    // 1. Trunk arena + boundary crossings.
    let trunk_arena = game_tree::build_trunk_arena(score, tc, deals, Some(dealer), rules)
        .expect("trunk arena build");
    let trunk = trunk_arena.prebuilt;
    let n_trunk_info = trunk.info_sets.len();
    let n_trees = subgame::flat_trees(&trunk).len();

    // 2. Group crossings into subgames, build persistent per-subgame state.
    let groups = group_subgames(trunk_arena.crossings);
    let mut subgames: Vec<SubgameState> = groups
        .into_iter()
        .map(|m| init_subgame(m, cfg.keep_arenas, ctx))
        .collect();
    let subgame_info_sets: usize = subgames.iter().map(|s| s.n_local).sum();

    // Trunk accumulator + fold list (all trunk indices; boundary roots are
    // inert — the hook prevents any accumulation there).
    let mut trunk_dense = cfr::new_resolve_accum(&trunk);
    let trunk_infosets: Vec<u32> = (0..n_trunk_info as u32).collect();

    // Member tree-idx set (for the trunk reach sweeps).
    let mut member_tree_set = std::collections::HashSet::new();
    for sg in &subgames {
        for &t in &sg.trunk_tree_idx {
            member_tree_set.insert(t);
        }
    }

    let warmup = cfg.rounds / 2;
    let mut trunk_iter: u64 = 0;
    let mut sub_iter_base: u64 = 0;
    // Round 1 runs the trunk against PRESENT-but-zero boundary values (the hook
    // must fire and return 0, exactly as trunk_solve seeds `vec![0.0; members]`
    // — an EMPTY list would instead let the trunk descend into the dummy leaf).
    let mut zero_boundary: Vec<(u32, game_tree::NodeId, f64)> = Vec::new();
    for sg in &subgames {
        for i in 0..sg.trunk_node.len() {
            zero_boundary.push((sg.trunk_tree_idx[i], sg.trunk_node[i], 0.0));
        }
    }
    let mut boundary_per_tree = build_boundary_per_tree(n_trees, &zero_boundary);
    let mut cbv_num: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];
    let mut cbv_den: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];
    let mut start_round = 0usize;

    // Resume: restore accumulators, counters, CBV, boundary values.
    if let Some(ck) = checkpoint {
        if ck.resume && ck.path.exists() {
            let saved = load_checkpoint(&ck.path).expect("load deep checkpoint");
            assert_eq!(
                saved.format_version, CHECKPOINT_FORMAT_VERSION,
                "checkpoint format mismatch"
            );
            assert_eq!(
                saved.score,
                (score.zero, score.one),
                "checkpoint score echo"
            );
            assert_eq!(saved.tc_level, tc.blocked_plain_level, "checkpoint tc echo");
            assert_eq!(saved.dealer, dealer, "checkpoint dealer echo");
            assert_eq!(saved.deal_count, deals.len(), "checkpoint deal-count echo");
            assert_eq!(saved.rounds, cfg.rounds, "checkpoint rounds echo");
            assert_eq!(
                saved.trunk_iters, cfg.trunk_iters,
                "checkpoint trunk_iters echo"
            );
            assert_eq!(
                saved.subgame_iters, cfg.subgame_iters,
                "checkpoint subgame_iters echo"
            );
            assert_eq!(saved.warmup, warmup, "checkpoint warmup echo");
            assert_eq!(
                saved.subgame_count,
                subgames.len(),
                "checkpoint subgame-count echo"
            );
            trunk_dense.load_regret_strategy_f64(&saved.trunk_regret, &saved.trunk_strategy);
            for (i, sg) in subgames.iter_mut().enumerate() {
                sg.dense
                    .load_regret_strategy_f64(&saved.sg_regret[i], &saved.sg_strategy[i]);
            }
            for o in 0..2 {
                cbv_num[o] = cbv_vec_to_map(&saved.cbv_num[o]);
                cbv_den[o] = cbv_vec_to_map(&saved.cbv_den[o]);
            }
            let boundary_flat: Vec<(u32, game_tree::NodeId, f64)> = saved
                .boundary_flat
                .iter()
                .map(|&(t, n, v)| (t, n, v))
                .collect();
            boundary_per_tree = build_boundary_per_tree(n_trees, &boundary_flat);
            trunk_iter = saved.trunk_iter;
            sub_iter_base = saved.sub_iter_base;
            start_round = saved.next_round;
        }
    }

    // 3. Alternation.
    eprintln!(
        "DEEP_PHASE init_done_s={:.1} start_round={}",
        t0.elapsed().as_secs_f64(),
        start_round
    );
    let mut t_round = std::time::Instant::now();
    for round in start_round..cfg.rounds {
        if round == warmup {
            trunk_dense.clear_strategy();
            for sg in &mut subgames {
                sg.dense.clear_strategy();
            }
            trunk_iter = 0;
            sub_iter_base = 0;
        }

        // Trunk phase (single-threaded).
        for _ in 0..cfg.trunk_iters {
            trunk_iter += 1;
            cfr::trunk_iteration(
                &trunk,
                &mut trunk_dense,
                &boundary_per_tree,
                &trunk_infosets,
                trunk_iter,
                score,
                match_values,
            );
        }

        // Trunk reach at boundaries from the CURRENT trunk iterate.
        let trunk_cur = current_rows(&trunk_dense, n_trunk_info);
        let reach_unit = trunk_reach(&trunk, &trunk_cur, &member_tree_set, false);
        let reach_dw = trunk_reach(&trunk, &trunk_cur, &member_tree_set, true);

        let fold_cbv = round >= warmup;
        // Subgame phase (parallel over subgames; each dense/arena is disjoint).
        let run = |subgames: &mut [SubgameState]| -> Vec<RoundContribs> {
            let f = |sg: &mut SubgameState| {
                subgame_round(
                    sg,
                    cfg.keep_arenas,
                    &reach_unit,
                    &reach_dw,
                    sub_iter_base,
                    cfg.subgame_iters,
                    fold_cbv,
                    score,
                    match_values,
                    ctx,
                )
            };
            if cfg.jobs > 1 {
                pool.as_ref()
                    .unwrap()
                    .install(|| subgames.par_iter_mut().map(f).collect())
            } else {
                subgames.iter_mut().map(f).collect()
            }
        };
        let contribs = run(&mut subgames);
        sub_iter_base += cfg.subgame_iters;

        // Reduce boundary values (unique per node) and CBV folds (FIXED subgame
        // order — deterministic regardless of completion order).
        let mut boundary_flat: Vec<(u32, game_tree::NodeId, f64)> = Vec::new();
        for c in &contribs {
            boundary_flat.extend_from_slice(&c.boundary);
        }
        boundary_per_tree = build_boundary_per_tree(n_trees, &boundary_flat);
        if fold_cbv {
            for c in &contribs {
                for o in 0..2 {
                    for &(view, w, wv) in &c.cbv[o] {
                        *cbv_num[o].entry(view).or_default() += wv;
                        *cbv_den[o].entry(view).or_default() += w;
                    }
                }
            }
        }

        // Checkpoint at cadence, on the final round, and at a chunk stop.
        let mut stop_here = false;
        if let Some(ck) = checkpoint {
            let is_last = round + 1 == cfg.rounds;
            stop_here = ck.stop_after == Some(round + 1);
            if ck.every > 0 && ((round + 1) % ck.every == 0 || is_last || stop_here) {
                let (tr, ts) = trunk_dense.regret_strategy_f64();
                let mut sg_regret = Vec::with_capacity(subgames.len());
                let mut sg_strategy = Vec::with_capacity(subgames.len());
                for sg in &subgames {
                    let (r, s) = sg.dense.regret_strategy_f64();
                    sg_regret.push(r);
                    sg_strategy.push(s);
                }
                let ckpt = DeepCheckpoint {
                    format_version: CHECKPOINT_FORMAT_VERSION,
                    next_round: round + 1,
                    trunk_iter,
                    sub_iter_base,
                    score: (score.zero, score.one),
                    tc_level: tc.blocked_plain_level,
                    dealer,
                    deal_count: deals.len(),
                    rounds: cfg.rounds,
                    trunk_iters: cfg.trunk_iters,
                    subgame_iters: cfg.subgame_iters,
                    warmup,
                    subgame_count: subgames.len(),
                    trunk_regret: tr,
                    trunk_strategy: ts,
                    sg_regret,
                    sg_strategy,
                    cbv_num: [cbv_map_to_vec(&cbv_num[0]), cbv_map_to_vec(&cbv_num[1])],
                    cbv_den: [cbv_map_to_vec(&cbv_den[0]), cbv_map_to_vec(&cbv_den[1])],
                    boundary_flat: boundary_flat.iter().map(|&(t, n, v)| (t, n, v)).collect(),
                };
                atomic_write(&ck.path, &ckpt).expect("write deep checkpoint");
            }
        }
        eprintln!(
            "DEEP_PHASE round={} wall_s={:.1}",
            round,
            t_round.elapsed().as_secs_f64()
        );
        t_round = std::time::Instant::now();
        if stop_here {
            // Chunked run: state is checkpointed; the final chunk does recovery.
            return (nan_report(subgames.len()), HashMap::new());
        }
    }

    // 4. Final recovery + certification.
    let trunk_avg = average_rows(&trunk_dense, n_trunk_info);
    // Deal weight folded into the reach init (trunk_solve fp order).
    let trunk_avg_reach = trunk_reach(&trunk, &trunk_avg, &member_tree_set, true);

    // Final-phase subgame arenas are built ON DEMAND, one at a time (rebuild
    // mode) — never all 990 at once — so peak RSS stays bounded by the trunk +
    // the persistent accumulators + a single subgame arena.

    // Raw composed subgame rows (subgame averages).
    let raw_rows: SubgameRows = subgames
        .iter()
        .map(|sg| average_rows(&sg.dense, sg.n_local))
        .collect();

    let per_subgame: Vec<SubgameStats> = subgames.iter().map(|s| s.stats).collect();
    let largest = per_subgame
        .iter()
        .copied()
        .max_by_key(|s| s.nodes)
        .unwrap_or_default();

    // CBV-by-view maps (tail loop averages).
    let mut cbv_tail: [HashMap<InfoSetKey, Option<f64>>; 2] = [HashMap::new(), HashMap::new()];
    for o in 0..2 {
        for (view, den) in &cbv_den[o] {
            if *den > 0.0 {
                cbv_tail[o].insert(*view, Some(cbv_num[o][view] / den));
            }
        }
    }

    let compute_eps = |sg_rows: &SubgameRows| -> f64 {
        if !cfg.certify {
            return f64::NAN;
        }
        deep_exploitability(
            &trunk,
            &trunk_avg,
            &subgames,
            cfg.keep_arenas,
            sg_rows,
            &trunk_avg_reach,
            n_trees,
            score,
            match_values,
            ctx,
        )
    };

    let t_cert = std::time::Instant::now();
    let raw_eps = compute_eps(&raw_rows);
    eprintln!(
        "DEEP_PHASE raw_cert_s={:.1} raw_eps={raw_eps:.9}",
        t_cert.elapsed().as_secs_f64()
    );

    let (tail_rows, composed_eps_tail, br_rows, composed_eps_br) = if cfg.certify_recoveries {
        // Recovery A: gadget re-solve from the tail-averaged CBVs.
        let tail_rows = recover(
            &subgames,
            cfg.keep_arenas,
            &raw_rows,
            &cbv_tail,
            &trunk_avg_reach,
            cfg.final_iters,
            score,
            match_values,
            ctx,
        );
        let composed_eps_tail = compute_eps(&tail_rows);

        // Recovery B: gadget re-solve from BR-based CBVs against the raw profile.
        let cbv_br = br_cbvs(
            &subgames,
            cfg.keep_arenas,
            &raw_rows,
            &trunk_avg_reach,
            score,
            match_values,
            ctx,
        );
        let br_rows = recover(
            &subgames,
            cfg.keep_arenas,
            &raw_rows,
            &cbv_br,
            &trunk_avg_reach,
            cfg.final_iters,
            score,
            match_values,
            ctx,
        );
        let composed_eps_br = compute_eps(&br_rows);
        (tail_rows, composed_eps_tail, br_rows, composed_eps_br)
    } else {
        (
            SubgameRows::default(),
            f64::NAN,
            SubgameRows::default(),
            f64::NAN,
        )
    };

    // Best certified profile.
    let (best_rows, composed_eps) = {
        let mut best: (&SubgameRows, f64) = (&raw_rows, raw_eps);
        if cfg.certify && cfg.certify_recoveries {
            if composed_eps_tail < best.1 {
                best = (&tail_rows, composed_eps_tail);
            }
            if composed_eps_br < best.1 {
                best = (&br_rows, composed_eps_br);
            }
        }
        (best.0.clone(), best.1)
    };

    // Decomposed game value of the certified profile.
    let inject_p0 = composed_boundary_p0(
        &subgames,
        cfg.keep_arenas,
        &best_rows,
        n_trees,
        score,
        match_values,
        ctx,
    );
    let (v_d0, v_d1) =
        cfr::game_value_per_dealer_deep(&trunk, &trunk_avg, score, match_values, &inject_p0);

    // Visit accounting — reconstruct trunk_solve's structural counters exactly.
    // Trunk arena nodes = (nodes above + incl boundary) + one dummy terminal per
    // tree with a crossing; subtracting the dummies recovers trunk_solve's
    // `n_trunk`. Each member's LOCAL tree is exactly the full arena's boundary
    // span, so their sum is `n_sub_total`; `n_full = n_trunk + descendants`.
    let trees = subgame::flat_trees(&trunk);
    let trunk_arena_nodes: u64 = trees.iter().map(|(t, _, _)| t.nodes.len() as u64).sum();
    let n_dummies = member_tree_set.len() as u64;
    let n_trunk = trunk_arena_nodes - n_dummies;
    let n_sub_total: u64 = subgames
        .iter()
        .map(|s| s.member_nodes.iter().map(|&n| n as u64).sum::<u64>())
        .sum();
    let total_members: u64 = subgames.iter().map(|s| s.trunk_node.len() as u64).sum();
    let descendants = n_sub_total - total_members;
    let n_full = n_trunk + descendants;
    let rounds = cfg.rounds as u64;
    let trunk_visits = rounds * 2 * n_trunk * cfg.trunk_iters;
    let subgame_visits = rounds * cfg.subgame_iters * 2 * n_sub_total;
    let final_visits = 2 * 2 * cfg.final_iters * n_sub_total;
    let total_visits = trunk_visits + subgame_visits + final_visits;
    let baseline_visits = 2 * n_full * cfg.baseline_iters;
    let multiplier = total_visits as f64 / baseline_visits.max(1) as f64;

    // 5. Compose keyed profile (subgame rows override inert trunk boundary roots).
    let mut composed: HashMap<u64, ActionProbs> =
        HashMap::with_capacity(subgame_info_sets + n_trunk_info);
    for (si, sg) in subgames.iter().enumerate() {
        for (key, &local) in &sg.key_to_local {
            composed.insert(key.0, best_rows[si][local as usize].clone());
        }
    }
    for (idx, (key, _, _)) in trunk.info_sets.iter().enumerate() {
        composed
            .entry(key.0)
            .or_insert_with(|| trunk_avg[idx].clone());
    }

    let report = DeepReport {
        raw_eps,
        composed_eps_tail,
        composed_eps_br,
        composed_eps,
        game_value_per_dealer: (v_d0, v_d1),
        game_value: (v_d0 + v_d1) / 2.0,
        subgames: subgames.len(),
        per_subgame,
        largest,
        trunk_visits,
        subgame_visits,
        final_visits,
        total_visits,
        baseline_visits,
        multiplier,
        trunk_info_sets: n_trunk_info,
        subgame_info_sets,
    };
    (report, composed)
}

/// Gadget re-solve every subgame from `cbv` (both players) starting from
/// `base_rows`; returns the recovered per-subgame rows.
#[allow(clippy::too_many_arguments)]
fn recover(
    subgames: &[SubgameState],
    keep_arenas: bool,
    base_rows: &SubgameRows,
    cbv: &[HashMap<InfoSetKey, Option<f64>>; 2],
    trunk_avg_reach: &HashMap<u32, [Vec<f64>; 2]>,
    final_iters: u64,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
    ctx: ReplayCtx<'_>,
) -> SubgameRows {
    subgames
        .iter()
        .enumerate()
        .map(|(si, sg)| {
            let arena = sg.arena(keep_arenas, ctx);
            resolve_one(
                sg,
                arena.get(),
                &base_rows[si],
                cbv,
                trunk_avg_reach,
                final_iters,
                score,
                match_values,
            )
        })
        .collect()
}

/// Gadget re-solve ONE subgame (both players) into a fresh row vector.
#[allow(clippy::too_many_arguments)]
fn resolve_one(
    sg: &SubgameState,
    arena: &PrebuiltTrees,
    base: &[ActionProbs],
    cbv: &[HashMap<InfoSetKey, Option<f64>>; 2],
    trunk_avg_reach: &HashMap<u32, [Vec<f64>; 2]>,
    final_iters: u64,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
) -> Vec<ActionProbs> {
    let mut rows = base.to_vec();
    let mut dense = cfr::new_resolve_accum(arena);
    let n_members = sg.trunk_node.len();
    for p in [0u8, 1u8] {
        let o = 1 - p;
        let members: Vec<ResolveMember> = (0..n_members)
            .map(|i| {
                let t = sg.trunk_tree_idx[i];
                let node = sg.trunk_node[i] as usize;
                let root_weight = trunk_avg_reach[&t][o as usize][node];
                ResolveMember {
                    tree_idx: i as u32,
                    node: 0,
                    subtree_end: sg.member_nodes[i] as game_tree::NodeId,
                    root_weight,
                    view_o: sg.view[i][o as usize],
                }
            })
            .collect();
        let out = cfr::resolve_subgame(
            arena,
            &members,
            &cbv[o as usize],
            score,
            match_values,
            p,
            final_iters,
            &mut dense,
        );
        for (idx, probs) in out {
            rows[idx as usize] = probs;
        }
    }
    rows
}

/// BR-based CBVs against `raw_rows` (the Phase-3 oracle path, blueprint := the
/// deep run's own raw profile). Aggregated per opponent view.
fn br_cbvs(
    subgames: &[SubgameState],
    keep_arenas: bool,
    raw_rows: &SubgameRows,
    trunk_avg_reach: &HashMap<u32, [Vec<f64>; 2]>,
    score: &Score,
    match_values: &crate::match_value::MatchValueTable,
    ctx: ReplayCtx<'_>,
) -> [HashMap<InfoSetKey, Option<f64>>; 2] {
    let mut num: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];
    let mut den: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];
    // Build each arena once; both opponents' BR read-outs share it.
    for (si, sg) in subgames.iter().enumerate() {
        let arena = sg.arena(keep_arenas, ctx);
        let n_members = sg.trunk_node.len();
        let member_nodes: Vec<(u32, game_tree::NodeId)> =
            (0..n_members).map(|i| (i as u32, 0)).collect();
        for o in [0u8, 1u8] {
            let root_w: Vec<f64> = (0..n_members)
                .map(|i| {
                    trunk_avg_reach[&sg.trunk_tree_idx[i]][o as usize][sg.trunk_node[i] as usize]
                })
                .collect();
            let vals = cfr::best_response_boundary_values_profile(
                arena.get(),
                &raw_rows[si],
                score,
                match_values,
                o,
                &member_nodes,
                Some(&root_w),
            );
            for i in 0..n_members {
                let w = root_w[i];
                if w > 0.0 {
                    let view = sg.view[i][o as usize];
                    *num[o as usize].entry(view).or_default() += w * vals[i];
                    *den[o as usize].entry(view).or_default() += w;
                }
            }
        }
    }
    let mut out: [HashMap<InfoSetKey, Option<f64>>; 2] = [HashMap::new(), HashMap::new()];
    for o in 0..2 {
        for (view, d) in &den[o] {
            if *d > 0.0 {
                out[o].insert(*view, Some(num[o][view] / d));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::game_tree::build_all_trees_with_dealer;
    use crate::match_value::MatchValueTable;

    /// Shared 8×8 dealer-0 setup (same synthetic match values as the
    /// `resolve::trunk_solve` tests, so the two paths solve the same game).
    fn setup(take: usize) -> (Score, TurnupClass, Vec<AbstractDeal>, MatchValueTable) {
        let score = Score { zero: 8, one: 8 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        for (a, b, v) in [(9u8, 8u8, 0.55), (11, 8, 0.72), (8, 9, 0.45), (8, 11, 0.28)] {
            mv.set(a, b, 0, v);
            mv.set(a, b, 1, v);
        }
        let mut deals = enumerate_deals(&tc);
        crate::cfr::subsample_deals(&mut deals, take);
        (score, tc, deals, mv)
    }

    fn cfg(rounds: usize, jobs: usize, keep: bool) -> DeepConfig {
        DeepConfig {
            rounds,
            trunk_iters: 3,
            subgame_iters: 3,
            final_iters: 80,
            baseline_iters: 90,
            jobs,
            keep_arenas: keep,
            certify: true,
            certify_recoveries: true,
        }
    }

    /// Gate A: `--deep --jobs 1` reproduces the full-arena `trunk_solve`
    /// certificates and game value to ~1e-9 (bit-identical arithmetic).
    #[test]
    fn deep_equivalence_with_full_arena_trunk_solve() {
        let (score, tc, deals, mv) = setup(24);
        let subgames =
            subgame::collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        let built = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let tcfg = crate::resolve::TrunkConfig {
            rounds: 6,
            trunk_iters: 3,
            subgame_iters: 3,
            final_iters: 80,
            baseline_iters: 90,
        };
        let (_c, full) = crate::resolve::trunk_solve(&built, &score, tc, &mv, &subgames, tcfg);

        let (deep, _composed) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            None,
        );

        let tol = 1e-9;
        assert!(
            (deep.raw_eps - full.raw_eps).abs() < tol,
            "raw {} vs {}",
            deep.raw_eps,
            full.raw_eps
        );
        assert!(
            (deep.composed_eps_tail - full.composed_eps_tail).abs() < tol,
            "tail {} vs {}",
            deep.composed_eps_tail,
            full.composed_eps_tail
        );
        assert!(
            (deep.composed_eps_br - full.composed_eps_br).abs() < tol,
            "br {} vs {}",
            deep.composed_eps_br,
            full.composed_eps_br
        );
        assert!(
            (deep.composed_eps - full.composed_eps).abs() < tol,
            "composed {} vs {}",
            deep.composed_eps,
            full.composed_eps
        );
        assert!(
            (deep.game_value - full.game_value).abs() < tol,
            "gv {} vs {}",
            deep.game_value,
            full.game_value
        );
        assert_eq!(deep.subgames, full.subgames);
        assert_eq!(deep.total_visits, full.total_visits);
        assert_eq!(deep.baseline_visits, full.baseline_visits);
    }

    /// Gate B: `--jobs 4` matches `--jobs 1` (disjoint per-subgame state, folds
    /// in fixed subgame order → deterministic, bit-identical).
    #[test]
    fn deep_jobs_parallel_matches_serial() {
        let (score, tc, deals, mv) = setup(24);
        let (s1, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            None,
        );
        let (s4, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 4, false),
            None,
        );
        assert!(
            (s1.raw_eps - s4.raw_eps).abs() < 1e-12,
            "{} vs {}",
            s1.raw_eps,
            s4.raw_eps
        );
        assert!((s1.composed_eps - s4.composed_eps).abs() < 1e-12);
        assert!((s1.game_value - s4.game_value).abs() < 1e-12);
    }

    /// The `--keep-arenas` toggle produces identical results (only memory /
    /// speed differ), validating the per-round rebuild against the cached arena.
    #[test]
    fn deep_keep_arenas_matches_rebuild() {
        let (score, tc, deals, mv) = setup(24);
        let (rebuild, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            None,
        );
        let (kept, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, true),
            None,
        );
        assert!((rebuild.raw_eps - kept.raw_eps).abs() < 1e-12);
        assert!((rebuild.composed_eps - kept.composed_eps).abs() < 1e-12);
    }

    /// Gate C: a straight 6-round run equals a chunked run (stop at 2, resume to
    /// 4, resume to 6) — identical certificates from a mid-run checkpoint.
    #[test]
    fn deep_checkpoint_roundtrip() {
        let (score, tc, deals, mv) = setup(24);
        let dir = std::env::temp_dir().join(format!("deep_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("deep.ckpt");

        let (straight, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            None,
        );

        // Chunk 1: run rounds 0..2, checkpoint, stop (NaN report).
        let ck1 = CheckpointCfg {
            path: path.clone(),
            every: 1,
            resume: false,
            stop_after: Some(2),
        };
        let (_p1, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            Some(&ck1),
        );
        // Chunk 2: resume, run rounds 2..4, checkpoint, stop.
        let ck2 = CheckpointCfg {
            path: path.clone(),
            every: 1,
            resume: true,
            stop_after: Some(4),
        };
        let (_p2, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            Some(&ck2),
        );
        // Chunk 3: resume, run rounds 4..6, full recovery + certificate.
        let ck3 = CheckpointCfg {
            path: path.clone(),
            every: 1,
            resume: true,
            stop_after: None,
        };
        let (resumed, _) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            Some(&ck3),
        );

        assert!(
            (resumed.raw_eps - straight.raw_eps).abs() < 1e-12,
            "resumed raw {} vs straight {}",
            resumed.raw_eps,
            straight.raw_eps
        );
        assert!((resumed.composed_eps - straight.composed_eps).abs() < 1e-12);
        assert!((resumed.composed_eps_tail - straight.composed_eps_tail).abs() < 1e-12);
        assert!((resumed.composed_eps_br - straight.composed_eps_br).abs() < 1e-12);
        assert!((resumed.game_value - straight.game_value).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Gate D: the decomposed streaming certificate equals the in-memory
    /// whole-arena exploitability on the SAME composed profile (exact match).
    #[test]
    fn deep_streaming_cert_matches_full_arena() {
        let (score, tc, deals, mv) = setup(24);
        let (deep, composed) = deep_solve(
            &score,
            tc,
            &deals,
            0,
            TreeRules::Current,
            &mv,
            cfg(6, 1, false),
            None,
        );

        // Materialize the composed profile against a full arena, aligned to its
        // info-set order, then run the ordinary whole-arena exploitability.
        let built = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let profile: Vec<ActionProbs> = built
            .info_sets
            .iter()
            .map(|(key, _, actions)| {
                composed
                    .get(&key.0)
                    .cloned()
                    .unwrap_or_else(|| crate::strategy::uniform_probs(actions.len()))
            })
            .collect();
        let in_memory =
            crate::cfr::compute_exploitability_from_action_probs(&built, &profile, &score, &mv);
        // `composed` holds the BEST recovery; compare to that certificate.
        assert!(
            (in_memory - deep.composed_eps).abs() < 1e-9,
            "streaming composed {} vs in-memory {}",
            deep.composed_eps,
            in_memory
        );
    }
}
