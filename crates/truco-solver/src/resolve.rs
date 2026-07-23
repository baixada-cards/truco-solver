//! Safe subgame re-solving (CFR-D / Burch–Johanson–Bowling 2014) —
//! orchestration layer. Plan 84 Phase 3.
//!
//! # What this module does
//!
//! Given a BLUEPRINT profile (average strategies for every info set — in the
//! prototype, an existing exact solve), it re-solves the play inside each
//! round-2 subgame (see [`crate::subgame`]) from scratch, using only the
//! blueprint's boundary summary — per-boundary-view counterfactual
//! best-response values (CBVs) and the resolver-side reach distribution —
//! and composes a full profile whose exploitability is provably no worse
//! than the blueprint's (up to the re-solve's own convergence slack).
//!
//! # The re-solving gadget
//!
//! To re-solve player `p`'s strategy in subgame `S` (opponent `o = 1-p`):
//!
//! - **Root distribution.** Each boundary crossing `h ∈ S` enters with
//!   weight `w(h) = π_c(h) · π_p^blueprint(h)` (chance/deal weight times the
//!   RESOLVER's blueprint reach; `o`'s own reach is excluded —
//!   `subgame::reach_excluding(tree, deal_weight, o, blueprint)` at `h`).
//!   Crossings with `w(h) = 0` are skipped entirely.
//!
//! - **Terminate/Follow layer.** Before play continues at `h`, `o` gets a
//!   synthetic decision at its boundary VIEW (its info-set key at `h`, which
//!   generally is not an acting info set): **Terminate** ends the gadget
//!   game with payoff `CBV_o(view)` — the value `o` could have gotten by
//!   best-responding to the blueprint from that view — or **Follow** into
//!   the real subtree, where both players play strategies re-solved fresh.
//!   In equilibrium of this gadget, `p`'s re-solved strategy concedes `o`
//!   no more at any boundary view than the blueprint did, which is exactly
//!   the safety guarantee: composing (blueprint outside `S`, re-solved `p`
//!   inside `S`) cannot increase `p`'s exploitability beyond the
//!   blueprint's plus the gadget's own convergence ε.
//!
//! - The gadget layer lives in code (`cfr::resolve_subgame`'s side
//!   accumulator keyed by `o`'s view key), NOT in the packed tree arena —
//!   the subtree below `h` is traversed by the existing SyncCFR+ machinery
//!   unchanged.
//!
//! To rebuild the FULL profile inside `S`, the gadget runs twice (once per
//! resolved player); composition takes `p`-owned rows from the `p`-re-solve.
//!
//! # Certification
//!
//! The composed profile is certified by the ordinary full-tree exact BR
//! (`cfr::compute_exploitability_from_action_probs`) — self-certifying, no
//! ground truth needed. The ground-truth harness additionally checks the
//! DEBUG properties (repair of a deliberately corrupted subgame), which is
//! what isolates gadget bugs from trunk-quality effects.
//!
//! # Prototype limitations (deliberate)
//!
//! - The re-solve allocates a full-size accumulator and touches only the
//!   subgame's rows; production would remap to a compact local index (the
//!   structural per-subgame numbers for plan 84 Phase 5 come from
//!   [`SubgameStats`], not from the prototype's RSS).
//! - The blueprint IS the CFV source ("oracle mode"); the trunk-CFR loop
//!   that would produce CFVs without a full solve is future work (Phase 4).
//! - Fixed-iteration re-solves; an internal gadget-exploitability stopping
//!   rule can come later — the composed certificate is the real gate.

use std::collections::HashMap;

use truco_engine::{Player, Score};

use crate::abstraction::{AbstractDeal, TurnupClass};
use crate::cfr::{self, ResolveMember};
use crate::game_tree::{NodeView, PrebuiltTrees, TreeRules};
use crate::match_value::MatchValueTable;
use crate::strategy::ActionProbs;
use crate::subgame::{self, Subgame};

/// Structural size of one subgame (drives the Phase-5 cost model: the
/// production per-subgame footprint is proportional to these, not to the
/// prototype's full-size accumulator).
#[derive(Clone, Copy, Debug, Default)]
pub struct SubgameStats {
    pub members: usize,
    pub nodes: usize,
    pub info_sets: usize,
}

/// Result of re-solving every subgame and composing.
#[derive(Debug)]
pub struct ResolveReport {
    pub blueprint_eps: f64,
    pub composed_eps: f64,
    pub subgames: usize,
    pub per_subgame: Vec<SubgameStats>,
    pub largest: SubgameStats,
}

/// Precomputed boundary summary for a set of subgames: per opponent `o`,
/// the CBV of every boundary view and the per-member gadget root weights
/// (`π_c · π_{1-o}` at the crossing). Computed from ONE exact-BR pass and
/// one reach sweep per player for the WHOLE subgame set — the BR pass
/// depends only on the blueprint, not on which nodes are read out, so
/// per-subgame passes (the first production run's fatal O(subgames ×
/// full-tree-BR) mistake, 2026-07-18) produce identical values at ~2000×
/// the cost.
pub struct BoundarySummary {
    /// `cbv_for_o[o]`: CBV keyed by `o`'s boundary view.
    pub cbv_for_o: [HashMap<crate::info_set::InfoSetKey, Option<f64>>; 2],
    /// `root_w_for_o[o][subgame_idx][member_idx]`: reach excluding `o`.
    pub root_w_for_o: [Vec<Vec<f64>>; 2],
}

/// Build the [`BoundarySummary`] for `subgames` against `blueprint`.
pub fn boundary_summary(
    prebuilt: &PrebuiltTrees,
    blueprint: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    subgames: &[Subgame],
) -> BoundarySummary {
    let trees = subgame::flat_trees(prebuilt);
    let all_nodes: Vec<(u32, crate::game_tree::NodeId)> = subgames
        .iter()
        .flat_map(|s| s.members.iter().map(|m| (m.tree_idx, m.node)))
        .collect();
    // Flattened member indices grouped by tree, so each tree's reach vector
    // (O(nodes) memory) is computed once and dropped before the next tree.
    let mut by_tree: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, (t, _)) in all_nodes.iter().enumerate() {
        by_tree.entry(*t).or_default().push(i);
    }

    let mut cbv_for_o: [HashMap<crate::info_set::InfoSetKey, Option<f64>>; 2] =
        [HashMap::new(), HashMap::new()];
    let mut root_w_for_o: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];

    for o in [0u8, 1u8] {
        // ONE certified BR pass for every boundary node at once.
        let values = cfr::best_response_boundary_values(
            prebuilt,
            blueprint,
            score,
            match_values,
            o,
            &all_nodes,
        );
        // One reach sweep per tree.
        let mut w_flat = vec![0.0f64; all_nodes.len()];
        for (t, idxs) in &by_tree {
            let (tree, _dealer, weight) = trees[*t as usize];
            let reach = subgame::reach_excluding(tree, weight, o, blueprint);
            for &i in idxs {
                w_flat[i] = reach[all_nodes[i].1 as usize];
            }
        }
        // Aggregate CBVs per view and slice weights back per subgame.
        let mut num: HashMap<crate::info_set::InfoSetKey, f64> = HashMap::new();
        let mut den: HashMap<crate::info_set::InfoSetKey, f64> = HashMap::new();
        let mut flat = 0usize;
        let mut per_sg: Vec<Vec<f64>> = Vec::with_capacity(subgames.len());
        for s in subgames {
            let mut ws = Vec::with_capacity(s.members.len());
            for m in &s.members {
                let w = w_flat[flat];
                let v = values[flat];
                flat += 1;
                ws.push(w);
                if w > 0.0 {
                    let view = if o == 0 { m.view_p0 } else { m.view_p1 };
                    *num.entry(view).or_default() += w * v;
                    *den.entry(view).or_default() += w;
                }
            }
            per_sg.push(ws);
        }
        let mut cbv: HashMap<crate::info_set::InfoSetKey, Option<f64>> = HashMap::new();
        for s in subgames {
            for m in &s.members {
                let view = if o == 0 { m.view_p0 } else { m.view_p1 };
                cbv.entry(view).or_insert_with(|| {
                    den.get(&view).filter(|d| **d > 0.0).map(|d| num[&view] / d)
                });
            }
        }
        cbv_for_o[o as usize] = cbv;
        root_w_for_o[o as usize] = per_sg;
    }

    BoundarySummary {
        cbv_for_o,
        root_w_for_o,
    }
}

/// Re-solve every subgame against `blueprint` and return the composed
/// profile plus the report. `deals`/`dealer_filter`/`rules` must be the same
/// inputs `prebuilt` was built from (the boundary walker replays the build).
pub fn resolve_all(
    prebuilt: &PrebuiltTrees,
    blueprint: &[ActionProbs],
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
    match_values: &MatchValueTable,
    resolve_iters: u64,
) -> (Vec<ActionProbs>, ResolveReport) {
    let subgames =
        subgame::collect_boundary(score, tc, deals, dealer_filter, rules).expect("boundary replay");
    let mut composed: Vec<ActionProbs> = blueprint.to_vec();
    // One BR/reach pass per player for the WHOLE subgame set.
    let summary = boundary_summary(prebuilt, blueprint, score, match_values, &subgames);
    // One shared accumulator for the whole run — resolve_subgame clears the
    // rows it touched, so per-subgame allocation/zeroing never happens.
    let mut dense = cfr::new_resolve_accum(prebuilt);
    let mut per_subgame = Vec::with_capacity(subgames.len());
    for (sg_idx, s) in subgames.iter().enumerate() {
        per_subgame.push(resolve_one_into(
            prebuilt,
            &mut composed,
            score,
            match_values,
            s,
            sg_idx,
            &summary,
            resolve_iters,
            &mut dense,
        ));
    }
    let blueprint_eps =
        cfr::compute_exploitability_from_action_probs(prebuilt, blueprint, score, match_values);
    let composed_eps =
        cfr::compute_exploitability_from_action_probs(prebuilt, &composed, score, match_values);
    let largest = per_subgame
        .iter()
        .copied()
        .max_by_key(|s| s.nodes)
        .unwrap_or_default();
    let report = ResolveReport {
        blueprint_eps,
        composed_eps,
        subgames: subgames.len(),
        per_subgame,
        largest,
    };
    (composed, report)
}

/// Re-solve ONE subgame (both players) from a precomputed
/// [`BoundarySummary`] and overwrite the corresponding rows of `composed`.
/// Returns the subgame's structural stats.
///
/// The summary is computed from the CFV-source profile — normally the
/// blueprint itself; the repair experiment computes it from the CLEAN
/// blueprint while `composed` starts corrupted, which is precisely the
/// "reconstruct subgame play from the boundary summary alone" claim.
fn resolve_one_into(
    prebuilt: &PrebuiltTrees,
    composed: &mut [ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    s: &Subgame,
    sg_idx: usize,
    summary: &BoundarySummary,
    resolve_iters: u64,
    dense: &mut crate::cfr::DenseAccum,
) -> SubgameStats {
    let trees = subgame::flat_trees(prebuilt);

    // Structural stats: contiguous spans make node/info-set counting linear.
    let mut stat_infosets: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut stat_nodes = 0usize;
    for m in &s.members {
        stat_nodes += (m.subtree_end - m.node) as usize;
        let (tree, _d, _w) = trees[m.tree_idx as usize];
        for id in m.node..m.subtree_end {
            if let NodeView::Player { table_idx, .. } = tree.view(id) {
                stat_infosets.insert(table_idx);
            }
        }
    }

    for p in [0u8, 1u8] {
        let o = 1 - p;
        let root_w = &summary.root_w_for_o[o as usize][sg_idx];
        let members: Vec<ResolveMember> = s
            .members
            .iter()
            .zip(root_w.iter())
            .map(|(m, &w)| ResolveMember {
                tree_idx: m.tree_idx,
                node: m.node,
                subtree_end: m.subtree_end,
                root_weight: w,
                view_o: if o == 0 { m.view_p0 } else { m.view_p1 },
            })
            .collect();
        let rows = cfr::resolve_subgame(
            prebuilt,
            &members,
            &summary.cbv_for_o[o as usize],
            score,
            match_values,
            p,
            resolve_iters,
            dense,
        );
        for (idx, probs) in rows {
            composed[idx as usize] = probs;
        }
    }

    SubgameStats {
        members: s.members.len(),
        nodes: stat_nodes,
        info_sets: stat_infosets.len(),
    }
}

/// Ground-truth debug harness: corrupt the blueprint inside one subgame
/// (uniform strategies), re-solve that subgame from the UNCORRUPTED
/// blueprint's boundary summary, and verify exploitability returns to ≈ the
/// blueprint's. Demonstrates that the boundary summary alone suffices to
/// reconstruct subgame play — the core CFR-D claim — and isolates gadget
/// correctness from trunk quality.
pub fn repair_experiment(
    prebuilt: &PrebuiltTrees,
    blueprint: &[ActionProbs],
    score: &Score,
    tc: TurnupClass,
    deals: &[AbstractDeal],
    dealer_filter: Option<Player>,
    rules: TreeRules,
    match_values: &MatchValueTable,
    subgame_index: usize,
    resolve_iters: u64,
) -> RepairReport {
    let subgames =
        subgame::collect_boundary(score, tc, deals, dealer_filter, rules).expect("boundary replay");
    let s = &subgames[subgame_index];
    let trees = subgame::flat_trees(prebuilt);

    // Corrupt: uniform play at every info set acting inside the subgame.
    let mut corrupted: Vec<ActionProbs> = blueprint.to_vec();
    for m in &s.members {
        let (tree, _d, _w) = trees[m.tree_idx as usize];
        for id in m.node..m.subtree_end {
            if let NodeView::Player { table_idx, .. } = tree.view(id) {
                let n = corrupted[table_idx as usize].len();
                corrupted[table_idx as usize] = crate::strategy::uniform_probs(n);
            }
        }
    }

    let blueprint_eps =
        cfr::compute_exploitability_from_action_probs(prebuilt, blueprint, score, match_values);
    let corrupted_eps =
        cfr::compute_exploitability_from_action_probs(prebuilt, &corrupted, score, match_values);

    // Repair: CBVs and root weights from the CLEAN blueprint (the boundary
    // summary a trunk solve would have stored); rows overwrite the corrupted
    // profile. Summary computed for just this subgame.
    let mut repaired = corrupted.clone();
    let target = std::slice::from_ref(s);
    let summary = boundary_summary(prebuilt, blueprint, score, match_values, target);
    let mut dense = cfr::new_resolve_accum(prebuilt);
    resolve_one_into(
        prebuilt,
        &mut repaired,
        score,
        match_values,
        s,
        0,
        &summary,
        resolve_iters,
        &mut dense,
    );
    let repaired_eps =
        cfr::compute_exploitability_from_action_probs(prebuilt, &repaired, score, match_values);

    RepairReport {
        blueprint_eps,
        corrupted_eps,
        repaired_eps,
    }
}

#[derive(Debug)]
pub struct RepairReport {
    pub blueprint_eps: f64,
    pub corrupted_eps: f64,
    pub repaired_eps: f64,
}

// ─── Trunk-CFR loop: CFVs without a full solve (plan 84 Phase 4) ───────────

/// Schedule knobs for [`trunk_solve`].
#[derive(Clone, Copy, Debug)]
pub struct TrunkConfig {
    /// Trunk/subgame alternation rounds.
    pub rounds: usize,
    /// SyncCFR+ iterations of the trunk per round (each = both player passes).
    pub trunk_iters: u64,
    /// SyncCFR+ iterations of the gadget-free subgame re-solve per round.
    pub subgame_iters: u64,
    /// Gadget re-solve iterations in the final recovery.
    pub final_iters: u64,
    /// Monolithic iteration count the visit multiplier is measured against.
    pub baseline_iters: u64,
}

/// Result of a from-scratch trunk-CFR solve (no blueprint anywhere).
#[derive(Debug)]
pub struct TrunkReport {
    /// Exploitability of the RAW averaged profile (trunk average + subgame
    /// average), before any gadget recovery. Separates loop quality from
    /// CBV/recovery quality: if `raw_eps` is small and `composed_eps` is not,
    /// the recovery inputs (CBVs) are the loose link, not the alternation.
    pub raw_eps: f64,
    /// Composed eps after gadget recovery from the TAIL-AVERAGED loop CBVs.
    pub composed_eps_tail: f64,
    /// Composed eps after gadget recovery from BR-based CBVs computed against
    /// the raw profile itself ([`boundary_summary`] — the Phase-3 oracle path
    /// with our own from-scratch profile as the blueprint).
    pub composed_eps_br: f64,
    /// `min(raw_eps, composed_eps_tail, composed_eps_br)` — the certificate of
    /// the profile actually returned.
    pub composed_eps: f64,
    /// Composed game value per dealer `(v_p0_deals, v_p1_deals)`, p0 perspective.
    pub game_value_per_dealer: (f64, f64),
    /// `(v_d0 + v_d1) / 2`.
    pub game_value: f64,
    pub subgames: usize,
    pub per_subgame: Vec<SubgameStats>,
    pub largest: SubgameStats,
    /// Structural node-visit counters (full traversal, both player passes,
    /// pruning-independent so runs reproduce). See [`trunk_solve`].
    pub trunk_visits: u64,
    pub subgame_visits: u64,
    pub final_visits: u64,
    pub total_visits: u64,
    pub baseline_visits: u64,
    /// `total_visits / baseline_visits`.
    pub multiplier: f64,
}

/// From-scratch CFR-D: solve the trunk (round-2 boundary treated as cached
/// value terminals) and the subgames (gadget-free, warm across rounds) in
/// alternation, accumulate per-boundary-view CBVs, then recover each subgame
/// behind the Phase-3 gadget from the accumulated CBVs and compose.
///
/// No blueprint is used anywhere. `subgames` must come from
/// [`crate::subgame::collect_boundary`] on the same build inputs `prebuilt`
/// was built from (sorted deterministically, as `collect_boundary` returns).
///
/// # Visit accounting
///
/// The three visit counters are STRUCTURAL — a full traversal of the relevant
/// region, both alternating player passes per iteration, ignoring runtime
/// regret-matching pruning. This makes them deterministic (runs reproduce) and
/// lets the multiplier compare like with like against the monolithic baseline
/// (`2 · full_nodes · baseline_iters`). The per-round boundary-value and reach
/// sweeps that produce the cached terminals and accumulated CBVs (no extra
/// best-response pass — the boundary values are reused) are prototype overhead
/// and are NOT counted; the multiplier measures re-solve (trunk + subgame +
/// final) traversal work only.
pub fn trunk_solve(
    prebuilt: &PrebuiltTrees,
    score: &Score,
    tc: TurnupClass,
    match_values: &MatchValueTable,
    subgames: &[Subgame],
    cfg: TrunkConfig,
) -> (Vec<ActionProbs>, TrunkReport) {
    use crate::info_set::InfoSetKey;
    use std::collections::{HashMap, HashSet};
    let _ = tc; // build inputs already baked into `prebuilt`/`subgames`.

    let trees = subgame::flat_trees(prebuilt);
    let n_info = prebuilt.info_sets.len();

    // Flattened member nodes and opponent views, subgame-major then member
    // order (deterministic, since collect_boundary sorts both).
    let all_member_nodes: Vec<(u32, crate::game_tree::NodeId)> = subgames
        .iter()
        .flat_map(|s| s.members.iter().map(|m| (m.tree_idx, m.node)))
        .collect();
    let all_member_views: Vec<[InfoSetKey; 2]> = subgames
        .iter()
        .flat_map(|s| s.members.iter().map(|m| [m.view_p0, m.view_p1]))
        .collect();

    // Info-set partition: subgame info sets are those acting inside any member
    // span; trunk info sets are the complement (the two are disjoint — a
    // round-2 subgame view never shares a round-1 trunk key).
    let mut subgame_infoset_set: HashSet<u32> = HashSet::new();
    for s in subgames {
        for m in &s.members {
            let (tree, _d, _w) = trees[m.tree_idx as usize];
            for id in m.node..m.subtree_end {
                if let NodeView::Player { table_idx, .. } = tree.view(id) {
                    subgame_infoset_set.insert(table_idx);
                }
            }
        }
    }
    let mut subgame_infosets: Vec<u32> = subgame_infoset_set.iter().copied().collect();
    subgame_infosets.sort_unstable();
    let trunk_infosets: Vec<u32> = (0..n_info as u32)
        .filter(|i| !subgame_infoset_set.contains(i))
        .collect();

    // Structural node counts for the visit multiplier.
    let n_full: u64 = trees.iter().map(|(t, _, _)| t.nodes.len() as u64).sum();
    let mut descendants: u64 = 0; // subtree nodes below (excluding) each boundary
    let mut n_sub_total: u64 = 0; // subtree nodes at/below each boundary
    for s in subgames {
        for m in &s.members {
            let span = (m.subtree_end - m.node) as u64;
            n_sub_total += span;
            descendants += span - 1;
        }
    }
    let n_trunk = n_full - descendants;

    // Combined average profile: subgame rows from the warm subgame accumulator,
    // everything else from the trunk accumulator.
    let build_profile = |trunk: &cfr::DenseAccum, sub: &cfr::DenseAccum| -> Vec<ActionProbs> {
        (0..n_info)
            .map(|idx| {
                if subgame_infoset_set.contains(&(idx as u32)) {
                    sub.average_row(idx)
                } else {
                    trunk.average_row(idx)
                }
            })
            .collect()
    };

    // Current-iterate combined profile: the intra-round couplings (boundary
    // value cache, subgame roots, CBV folds) must see the CURRENT
    // regret-matching strategies — SyncCFR+ backups are defined against the
    // frozen current iterate, and coupling through the lagging average made
    // the alternation chase a stale target (the r90/r150/r300 ≈ 0.018
    // composed-eps plateau; SOLVER_BENCHMARKS 2026-07-21).
    let build_current_profile =
        |trunk: &cfr::DenseAccum, sub: &cfr::DenseAccum| -> Vec<ActionProbs> {
            (0..n_info)
                .map(|idx| {
                    if subgame_infoset_set.contains(&(idx as u32)) {
                        sub.current_row(idx)
                    } else {
                        trunk.current_row(idx)
                    }
                })
                .collect()
        };

    // Group boundary values (aligned to all_member_nodes) into the per-tree,
    // node-sorted lookup the trunk hook binary-searches.
    let build_boundary = |values: &[f64]| -> Vec<Vec<(crate::game_tree::NodeId, f64)>> {
        let mut per_tree: Vec<Vec<(crate::game_tree::NodeId, f64)>> = vec![Vec::new(); trees.len()];
        for (&(t, node), &v) in all_member_nodes.iter().zip(values) {
            per_tree[t as usize].push((node, v));
        }
        for v in &mut per_tree {
            v.sort_by_key(|&(n, _)| n);
        }
        per_tree
    };

    // Gadget-free subgame roots under a profile: each crossing enters with its
    // per-player trunk reach (deal weight excluded) and deal-weight chance.
    // Reach at a boundary node depends only on the profile's trunk rows.
    let build_roots = |profile: &[ActionProbs]| -> Vec<cfr::SubgameRoot> {
        let mut tree_reach: HashMap<u32, (Vec<f64>, Vec<f64>)> = HashMap::new();
        let mut roots: Vec<cfr::SubgameRoot> = Vec::with_capacity(all_member_nodes.len());
        for &(t, node) in &all_member_nodes {
            let (tree, _d, w) = trees[t as usize];
            let e = tree_reach.entry(t).or_insert_with(|| {
                (
                    subgame::reach_excluding(tree, 1.0, 1, profile),
                    subgame::reach_excluding(tree, 1.0, 0, profile),
                )
            });
            roots.push(cfr::SubgameRoot {
                tree_idx: t,
                node,
                reach: [e.0[node as usize], e.1[node as usize]],
                chance: w,
            });
        }
        roots
    };

    let mut trunk_dense = cfr::new_resolve_accum(prebuilt);
    let mut subgame_dense = cfr::new_resolve_accum(prebuilt);

    // Round 1 trunk runs with neutral (0) boundary values.
    let mut boundary_per_tree = build_boundary(&vec![0.0; all_member_nodes.len()]);

    let mut trunk_iter: u64 = 0;
    let mut sub_iter: u64 = 0;
    // Average only over the converged tail: the early rounds solve against a
    // moving trunk range / stale boundary values, and their strategy average
    // blends incompatible ranges (exploitable). Keep the warm regrets, but
    // zero both strategy accumulators once past the warmup so the final
    // average reflects only the (near-)fixed-point rounds.
    let warmup = cfg.rounds / 2;

    // Running reach-weighted counterfactual-value accumulators per opponent
    // view, over the converged tail (same reach convention as
    // boundary_summary: reach EXCLUDING the opponent). Their average is the
    // Terminate value for final recovery — the equilibrium counterfactual
    // value, which is TIGHT even though any single average-profile snapshot's
    // best-response value is loose (the gadget preserves the CBV it is given,
    // so a loose CBV yields a loose composed certificate).
    let mut cbv_num: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];
    let mut cbv_den: [HashMap<InfoSetKey, f64>; 2] = [HashMap::new(), HashMap::new()];

    for round in 0..cfg.rounds {
        if round == warmup {
            trunk_dense.clear_strategy();
            subgame_dense.clear_strategy();
            trunk_iter = 0;
            sub_iter = 0;
        }
        // 1. Trunk phase (uses current cached boundary values).
        for _ in 0..cfg.trunk_iters {
            trunk_iter += 1;
            cfr::trunk_iteration(
                prebuilt,
                &mut trunk_dense,
                &boundary_per_tree,
                &trunk_infosets,
                trunk_iter,
                score,
                match_values,
            );
        }

        // 2. Refresh subgame roots from the CURRENT trunk iterate and run the
        //    gadget-free subgame phase (warm accumulators for fast regret
        //    convergence).
        let profile = build_current_profile(&trunk_dense, &subgame_dense);
        let roots = build_roots(&profile);
        for _ in 0..cfg.subgame_iters {
            sub_iter += 1;
            cfr::subgame_root_iteration(
                prebuilt,
                &mut subgame_dense,
                &roots,
                &subgame_infosets,
                sub_iter,
                score,
                match_values,
            );
        }

        // 3. Boundary values (subgame value under the CURRENT combined
        //    iterate, p0 perspective): cache the pair for the next trunk
        //    phase, and — in the converged tail — fold the per-opponent
        //    counterfactual value into the running CBV averages (average of
        //    current-iterate CFVs across rounds — the CFR-D quantity — not
        //    snapshots of the lagging average profile). Reuses the SAME
        //    boundary values, so no extra best-response pass per round.
        let profile = build_current_profile(&trunk_dense, &subgame_dense);
        let bvals = cfr::profile_boundary_values(
            prebuilt,
            &profile,
            score,
            match_values,
            &all_member_nodes,
        );
        boundary_per_tree = build_boundary(&bvals);

        if round >= warmup {
            for o in [0u8, 1u8] {
                let mut rw_by_tree: HashMap<u32, Vec<f64>> = HashMap::new();
                for (i, &(t, node)) in all_member_nodes.iter().enumerate() {
                    let (tree, _d, w) = trees[t as usize];
                    let rw = rw_by_tree
                        .entry(t)
                        .or_insert_with(|| subgame::reach_excluding(tree, w, o, &profile));
                    let weight = rw[node as usize];
                    if weight > 0.0 {
                        // o's counterfactual value = ±(p0-perspective value).
                        let v_o = if o == 0 { bvals[i] } else { -bvals[i] };
                        let view = all_member_views[i][o as usize];
                        *cbv_num[o as usize].entry(view).or_default() += weight * v_o;
                        *cbv_den[o as usize].entry(view).or_default() += weight;
                    }
                }
            }
        }
    }

    // Final recovery: gadget re-solve per subgame from the accumulated Terminate
    // CBVs and trunk-average root weights, then compose.
    let profile_final = build_profile(&trunk_dense, &subgame_dense);
    let mut cbv_for_o: [HashMap<InfoSetKey, Option<f64>>; 2] = [HashMap::new(), HashMap::new()];
    for o in 0..2 {
        for (view, den) in &cbv_den[o] {
            if *den > 0.0 {
                cbv_for_o[o].insert(*view, Some(cbv_num[o][view] / den));
            }
        }
    }
    let mut root_w_for_o: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
    for o in [0u8, 1u8] {
        let mut rw_by_tree: HashMap<u32, Vec<f64>> = HashMap::new();
        let mut per_sg: Vec<Vec<f64>> = Vec::with_capacity(subgames.len());
        for s in subgames {
            let mut ws = Vec::with_capacity(s.members.len());
            for m in &s.members {
                let (tree, _d, w) = trees[m.tree_idx as usize];
                let rw = rw_by_tree
                    .entry(m.tree_idx)
                    .or_insert_with(|| subgame::reach_excluding(tree, w, o, &profile_final));
                ws.push(rw[m.node as usize]);
            }
            per_sg.push(ws);
        }
        root_w_for_o[o as usize] = per_sg;
    }
    let summary = BoundarySummary {
        cbv_for_o,
        root_w_for_o,
    };

    // Certify the RAW averaged profile first — the diagnostic that separates
    // loop quality from recovery/CBV quality.
    let raw_eps = cfr::compute_exploitability_from_action_probs(
        prebuilt,
        &profile_final,
        score,
        match_values,
    );

    // Recovery A: gadget re-solve from the tail-averaged loop CBVs.
    let mut composed_tail = profile_final.clone();
    let mut dense = cfr::new_resolve_accum(prebuilt);
    let mut per_subgame = Vec::with_capacity(subgames.len());
    for (sg_idx, s) in subgames.iter().enumerate() {
        per_subgame.push(resolve_one_into(
            prebuilt,
            &mut composed_tail,
            score,
            match_values,
            s,
            sg_idx,
            &summary,
            cfg.final_iters,
            &mut dense,
        ));
    }
    let composed_eps_tail = cfr::compute_exploitability_from_action_probs(
        prebuilt,
        &composed_tail,
        score,
        match_values,
    );

    // Recovery B: gadget re-solve from BR-based CBVs against the raw profile
    // (Phase-3 oracle path, blueprint := our own from-scratch profile).
    let summary_br = boundary_summary(prebuilt, &profile_final, score, match_values, subgames);
    let mut composed_br = profile_final.clone();
    for (sg_idx, s) in subgames.iter().enumerate() {
        resolve_one_into(
            prebuilt,
            &mut composed_br,
            score,
            match_values,
            s,
            sg_idx,
            &summary_br,
            cfg.final_iters,
            &mut dense,
        );
    }
    let composed_eps_br =
        cfr::compute_exploitability_from_action_probs(prebuilt, &composed_br, score, match_values);

    // Return the best of the three certified profiles.
    let (composed, composed_eps) = {
        let mut best = (profile_final, raw_eps);
        if composed_eps_tail < best.1 {
            best = (composed_tail, composed_eps_tail);
        }
        if composed_eps_br < best.1 {
            best = (composed_br, composed_eps_br);
        }
        best
    };
    let (v_d0, v_d1) =
        cfr::game_value_per_dealer_from_action_probs(prebuilt, &composed, score, match_values);
    let largest = per_subgame
        .iter()
        .copied()
        .max_by_key(|s| s.nodes)
        .unwrap_or_default();

    let rounds = cfg.rounds as u64;
    let trunk_visits = rounds * 2 * n_trunk * cfg.trunk_iters;
    let subgame_visits = rounds * cfg.subgame_iters * 2 * n_sub_total;
    // Gadget recovery runs once per resolved player (2), each K iterations of
    // both alternating passes (2) over the subgame subtrees.
    let final_visits = 2 * 2 * cfg.final_iters * n_sub_total;
    let total_visits = trunk_visits + subgame_visits + final_visits;
    let baseline_visits = 2 * n_full * cfg.baseline_iters;
    let multiplier = total_visits as f64 / baseline_visits.max(1) as f64;

    let report = TrunkReport {
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
    };
    (composed, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::enumerate_deals;
    use crate::game_tree::build_all_trees_with_dealer;
    use crate::match_value::MatchValueTable;

    /// Small 8x8 blueprint solved with the production algorithm, plus the
    /// synthetic-but-complete successor match values the solve needs.
    fn blueprint_8x8(
        take: usize,
        iters: u64,
    ) -> (
        PrebuiltTrees,
        Vec<AbstractDeal>,
        TurnupClass,
        Score,
        MatchValueTable,
        Vec<ActionProbs>,
    ) {
        let score = Score { zero: 8, one: 8 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        for (a, b, v) in [(9u8, 8u8, 0.55), (11, 8, 0.72), (8, 9, 0.45), (8, 11, 0.28)] {
            mv.set(a, b, 0, v);
            mv.set(a, b, 1, v);
        }
        // Match the solve's internal subset exactly: subsample_deals is
        // STRIDED, not a prefix.
        let mut deals = enumerate_deals(&tc);
        crate::cfr::subsample_deals(&mut deals, take);
        let built = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let (table, _stats) = crate::cfr::solve_with_limit_algo(
            score,
            tc,
            iters,
            &mv,
            Some(take),
            Some(0),
            crate::cfr::CfrAlgorithm::SyncCfrPlus,
            iters,
            1,
            None,
        );
        // Blueprint as dense ActionProbs aligned with built.info_sets.
        let blueprint: Vec<ActionProbs> = built
            .info_sets
            .iter()
            .map(|(key, _, actions)| {
                table
                    .data
                    .get(key)
                    .map(|d| d.average_strategy())
                    .unwrap_or_else(|| crate::strategy::uniform_probs(actions.len()))
            })
            .collect();
        (built, deals, tc, Score { zero: 8, one: 8 }, mv, blueprint)
    }

    #[test]
    fn resolve_all_preserves_blueprint_quality() {
        let (built, deals, tc, score, mv, blueprint) = blueprint_8x8(24, 60);
        let (_composed, report) = resolve_all(
            &built,
            &blueprint,
            &score,
            tc,
            &deals,
            Some(0),
            TreeRules::Current,
            &mv,
            120,
        );
        assert!(report.subgames > 0);
        eprintln!(
            "resolve_all: blueprint={:.6} composed={:.6} subgames={} largest={:?}",
            report.blueprint_eps, report.composed_eps, report.subgames, report.largest
        );
        // Safety: composing re-solved subgames must not degrade the
        // blueprint beyond the re-solve's own convergence slack.
        let slack = 0.01;
        assert!(
            report.composed_eps <= report.blueprint_eps + slack,
            "composed {} vs blueprint {} (+{} slack)",
            report.composed_eps,
            report.blueprint_eps,
            slack
        );
    }

    #[test]
    fn trunk_solve_from_scratch_certifies_and_counts() {
        // Reuse the 8x8 match-value + deal setup; the blueprint is unused here
        // (Phase 4 solves from scratch).
        let (built, deals, tc, score, mv, _blueprint) = blueprint_8x8(24, 60);
        let subgames =
            subgame::collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        assert!(!subgames.is_empty());

        let cfg = TrunkConfig {
            rounds: 5,
            trunk_iters: 3,
            subgame_iters: 3,
            final_iters: 80,
            baseline_iters: 90,
        };
        let (composed, report) = trunk_solve(&built, &score, tc, &mv, &subgames, cfg);
        eprintln!(
            "trunk_solve: composed_eps={:.6} gv={:.6} gv_d0={:.6} gv_d1={:.6} mult={:.4} \
             visits(trunk={} subgame={} final={} total={} baseline={})",
            report.composed_eps,
            report.game_value,
            report.game_value_per_dealer.0,
            report.game_value_per_dealer.1,
            report.multiplier,
            report.trunk_visits,
            report.subgame_visits,
            report.final_visits,
            report.total_visits,
            report.baseline_visits,
        );

        // Certificate is finite, non-negative, and beats a uniform profile —
        // the from-scratch loop must recover real subgame play.
        assert!(report.composed_eps.is_finite() && report.composed_eps >= 0.0);
        // The returned profile is the one the report certifies.
        assert_eq!(composed.len(), built.info_sets.len());
        let recert = cfr::compute_exploitability_from_action_probs(&built, &composed, &score, &mv);
        assert!((recert - report.composed_eps).abs() < 1e-12);
        let uniform: Vec<ActionProbs> = built
            .info_sets
            .iter()
            .map(|(_, _, actions)| crate::strategy::uniform_probs(actions.len()))
            .collect();
        let uniform_eps =
            cfr::compute_exploitability_from_action_probs(&built, &uniform, &score, &mv);
        assert!(
            report.composed_eps < uniform_eps,
            "composed {} did not beat uniform {}",
            report.composed_eps,
            uniform_eps
        );

        // Game value is a valid match equity.
        assert!(report.game_value.abs() <= 1.0);

        // Visit counters are self-consistent.
        assert_eq!(
            report.total_visits,
            report.trunk_visits + report.subgame_visits + report.final_visits
        );
        assert!(report.trunk_visits > 0);
        assert!(report.subgame_visits > 0);
        assert!(report.final_visits > 0);
        assert!(report.baseline_visits > 0);
        assert!(report.multiplier.is_finite() && report.multiplier > 0.0);
        assert!(
            (report.multiplier - report.total_visits as f64 / report.baseline_visits as f64).abs()
                < 1e-9
        );
        assert_eq!(report.subgames, subgames.len());
        assert_eq!(report.per_subgame.len(), subgames.len());
    }

    #[test]
    fn repair_experiment_reconstructs_corrupted_subgame() {
        let (built, deals, tc, score, mv, blueprint) = blueprint_8x8(24, 60);
        // Pick the structurally largest subgame so corruption actually bites.
        let subgames =
            subgame::collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();
        let trees = subgame::flat_trees(&built);
        let biggest = (0..subgames.len())
            .max_by_key(|i| {
                subgames[*i]
                    .members
                    .iter()
                    .map(|m| (m.subtree_end - m.node) as usize)
                    .sum::<usize>()
            })
            .unwrap();
        let _ = trees;
        let report = repair_experiment(
            &built,
            &blueprint,
            &score,
            tc,
            &deals,
            Some(0),
            TreeRules::Current,
            &mv,
            biggest,
            200,
        );
        eprintln!(
            "repair: blueprint={:.6} corrupted={:.6} repaired={:.6}",
            report.blueprint_eps, report.corrupted_eps, report.repaired_eps
        );
        // Corruption must measurably hurt, and repair must recover most of
        // it using ONLY the boundary summary.
        assert!(
            report.corrupted_eps > report.blueprint_eps + 1e-4,
            "corruption did not bite: {} vs {}",
            report.corrupted_eps,
            report.blueprint_eps
        );
        let recovered = (report.corrupted_eps - report.repaired_eps)
            / (report.corrupted_eps - report.blueprint_eps);
        assert!(
            recovered > 0.8,
            "repair recovered only {:.1}% (blueprint {}, corrupted {}, repaired {})",
            recovered * 100.0,
            report.blueprint_eps,
            report.corrupted_eps,
            report.repaired_eps
        );
    }
}
