//! Teacher export: the compact training artifact behind plan 71 steps 2–3
//! and the EV-feedback product feature.
//!
//! One `.teach` file per solved (score, tc, dealer) carries, per info set in
//! `table_idx` order:
//!   - the average-strategy probabilities (f32),
//!   - per-action **one-shot-deviation Q values**: the EV of taking action
//!     `a` at this info set and playing the equilibrium everywhere else,
//!     from the acting player's perspective in ±1 match space, averaged with
//!     counterfactual (opponent-reach × deal) weights. `maxQ − Q(a)` is
//!     exactly the "EV you left on the table" number a trainer UI shows —
//!     and equal-Q support actions are all optimal, as expected at
//!     equilibrium.
//!   - per-action **one-shot-deviation point values** (`pts`): the same
//!     one-shot-deviation traversal as Q, but the terminal value is the raw
//!     HAND-point differential (acting player's points minus opponent's,
//!     e.g. ±1/±3/±6/±9/±12 at the stake in force when the hand ended,
//!     including fold outcomes) instead of the match-equity value looked up
//!     through `MatchValueTable`. `pts` answers "how many hand points does
//!     this action cost", which is denominated the same way a player counts
//!     — unlike `q`, it does NOT depend on score or dealer (a hand's point
//!     value is self-contained), so it is comparable across score states in
//!     a way `q` is not. `pts` is NOT part of the exact-BR certificate: the
//!     certified exploitability measure is defined over match equity, and
//!     "how exploitable is this point-value deviation" is a different
//!     (uncomputed) question — see `pts` docs on `ChartAction`/`export-chart`.
//!   - own/opponent counterfactual reach mass (f32), so training can
//!     importance-sample and the UI can flag rarely-reached spots.
//!
//! Info-set metadata (keys, `InfoSet` features, action codes) is identical
//! for every state in a ladder band, so it is NOT repeated here — it lives
//! in the band treepack, or in the optional band-meta sidecar written by
//! `save_band_meta`. A `.teach` file is bound to its band by the same
//! `sig_hash` the treepack uses.

use std::fs::{self, File};
use std::path::Path;

use crate::bincode_v1;
use crate::cfr::terminal_p0_value;
use crate::game_tree::{GameTree, NodeId, NodeView, PrebuiltTrees};
use crate::info_set::{AbstractAction, InfoSet, InfoSetKey};
use crate::match_value::MatchValueTable;
use crate::storage::StorageError;
use crate::strategy::{ActionProbs, InfoSetData, StrategyTable};
use truco_engine::{Player, Score};

const MAGIC: u64 = 0x5452_5543_4f54_4541; // "TRUCOTEA"
/// v2 adds the `pts_values` section (one-shot-deviation hand-point values)
/// after `q_values`; v1 files are no longer produced or read.
const VERSION: u64 = 2;
const PP_PER_Q_UNIT: f64 = 50.0;
pub const RESIDUE_ASSERT_Q_GAP_PP: f64 = 5.0;
pub const RESIDUE_ASSERT_MAX_INFO_SET_MASS: f64 = 0.03;
pub const PURIFY_MAX_PROB: f64 = 0.01;
pub const PURIFY_MIN_Q_GAP_PP: f64 = 5.0;
/// Header layout (u64 words):
/// [magic, version, sig_hash, n_info_sets, flat_len,
///  score0<<8|score1, tc<<8|dealer, reserved]
const HEADER_WORDS: usize = 8;

/// Per-band teacher tensors in `table_idx` order (SoA, flat).
#[derive(Clone, Debug)]
pub struct TeacherData {
    /// `offsets[i]..offsets[i+1]` indexes `probs`/`q_values`/`pts_values` for
    /// info set `i` (one trailing entry, so `offsets.len() == n_info_sets + 1`).
    pub offsets: Vec<u32>,
    pub probs: Vec<f32>,
    pub q_values: Vec<f32>,
    /// One-shot-deviation hand-point values — see the module doc comment.
    pub pts_values: Vec<f32>,
    pub own_reach: Vec<f32>,
    pub opp_reach: Vec<f32>,
}

impl TeacherData {
    pub fn n_info_sets(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn probs_of(&self, idx: usize) -> &[f32] {
        &self.probs[self.offsets[idx] as usize..self.offsets[idx + 1] as usize]
    }

    pub fn q_of(&self, idx: usize) -> &[f32] {
        &self.q_values[self.offsets[idx] as usize..self.offsets[idx + 1] as usize]
    }

    pub fn pts_of(&self, idx: usize) -> &[f32] {
        &self.pts_values[self.offsets[idx] as usize..self.offsets[idx + 1] as usize]
    }
}

#[derive(Clone, Debug)]
pub struct QGapMassSummary {
    pub q_gap_pp: f64,
    pub max_action_prob: f64,
    pub info_sets: usize,
    pub touched_info_sets: usize,
    pub total_info_set_mass: f64,
    pub max_info_set_mass: f64,
    pub max_touched_action_prob: f64,
    pub max_q_gap_touched_pp: f64,
    pub info_sets_ge_0_1pct: usize,
    pub info_sets_ge_0_5pct: usize,
    pub info_sets_ge_1pct: usize,
    pub info_sets_ge_3pct: usize,
}

impl QGapMassSummary {
    pub fn empty(q_gap_pp: f64, max_action_prob: f64) -> Self {
        Self {
            q_gap_pp,
            max_action_prob,
            info_sets: 0,
            touched_info_sets: 0,
            total_info_set_mass: 0.0,
            max_info_set_mass: 0.0,
            max_touched_action_prob: 0.0,
            max_q_gap_touched_pp: 0.0,
            info_sets_ge_0_1pct: 0,
            info_sets_ge_0_5pct: 0,
            info_sets_ge_1pct: 0,
            info_sets_ge_3pct: 0,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        assert!(
            (self.q_gap_pp - other.q_gap_pp).abs() < 1e-12,
            "cannot merge summaries with different q-gap thresholds"
        );
        assert!(
            (self.max_action_prob - other.max_action_prob).abs() < 1e-12,
            "cannot merge summaries with different action-probability caps"
        );
        self.info_sets += other.info_sets;
        self.touched_info_sets += other.touched_info_sets;
        self.total_info_set_mass += other.total_info_set_mass;
        self.max_info_set_mass = self.max_info_set_mass.max(other.max_info_set_mass);
        self.max_touched_action_prob = self
            .max_touched_action_prob
            .max(other.max_touched_action_prob);
        self.max_q_gap_touched_pp = self.max_q_gap_touched_pp.max(other.max_q_gap_touched_pp);
        self.info_sets_ge_0_1pct += other.info_sets_ge_0_1pct;
        self.info_sets_ge_0_5pct += other.info_sets_ge_0_5pct;
        self.info_sets_ge_1pct += other.info_sets_ge_1pct;
        self.info_sets_ge_3pct += other.info_sets_ge_3pct;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PurificationConfig {
    pub max_prob: f64,
    pub min_q_gap_pp: f64,
}

impl Default for PurificationConfig {
    fn default() -> Self {
        Self {
            max_prob: PURIFY_MAX_PROB,
            min_q_gap_pp: PURIFY_MIN_Q_GAP_PP,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PurificationReport {
    pub max_prob: f64,
    pub min_q_gap_pp: f64,
    pub info_sets: usize,
    pub touched_info_sets: usize,
    pub actions_zeroed: usize,
    pub mass_removed: f64,
    pub max_info_set_mass_removed: f64,
    pub max_q_gap_touched_pp: f64,
}

/// Convert `.teach` probabilities into solver-native action-probability rows.
pub fn action_probs_from_teacher(data: &TeacherData) -> Vec<ActionProbs> {
    (0..data.n_info_sets())
        .map(|idx| data.probs_of(idx).iter().map(|&p| p as f64).collect())
        .collect()
}

/// Summarize, per info set, the raw probability mass assigned to actions whose
/// one-shot-deviation Q gap exceeds `q_gap_pp` match-win percentage points.
pub fn summarize_q_gap_mass(data: &TeacherData, q_gap_pp: f64) -> QGapMassSummary {
    summarize_q_gap_mass_with_prob_cap(data, q_gap_pp, f64::INFINITY)
}

/// Like [`summarize_q_gap_mass`], but only counts individually-small action
/// probabilities below `max_action_prob`. This is the dominated-residue
/// diagnostic used by purification and export assertions.
pub fn summarize_q_gap_mass_with_prob_cap(
    data: &TeacherData,
    q_gap_pp: f64,
    max_action_prob: f64,
) -> QGapMassSummary {
    let mut summary = QGapMassSummary::empty(q_gap_pp, max_action_prob);
    summary.info_sets = data.n_info_sets();

    for idx in 0..data.n_info_sets() {
        let probs = data.probs_of(idx);
        let q_values = data.q_of(idx);
        if probs.is_empty() {
            continue;
        }
        let best_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut mass: f64 = 0.0;
        let mut max_touched_action_prob: f64 = 0.0;
        let mut max_gap: f64 = 0.0;
        for (&p, &q) in probs.iter().zip(q_values.iter()) {
            let p = p as f64;
            let gap_pp = (best_q - q as f64).max(0.0) * PP_PER_Q_UNIT;
            if gap_pp > q_gap_pp && p > 0.0 && p < max_action_prob {
                mass += p;
                max_touched_action_prob = max_touched_action_prob.max(p);
                max_gap = max_gap.max(gap_pp);
            }
        }
        if mass > 0.0 {
            summary.touched_info_sets += 1;
            summary.total_info_set_mass += mass;
            summary.max_info_set_mass = summary.max_info_set_mass.max(mass);
            summary.max_touched_action_prob =
                summary.max_touched_action_prob.max(max_touched_action_prob);
            summary.max_q_gap_touched_pp = summary.max_q_gap_touched_pp.max(max_gap);
            if mass >= 0.001 {
                summary.info_sets_ge_0_1pct += 1;
            }
            if mass >= 0.005 {
                summary.info_sets_ge_0_5pct += 1;
            }
            if mass >= 0.01 {
                summary.info_sets_ge_1pct += 1;
            }
            if mass >= 0.03 {
                summary.info_sets_ge_3pct += 1;
            }
        }
    }

    summary
}

/// Loud non-destructive export guard: measure residue in raw `.teach` tensors
/// and fail only when a future solve carries too much mass across a large Q gap.
pub fn assert_q_gap_residue(
    data: &TeacherData,
    q_gap_pp: f64,
    max_info_set_mass: f64,
) -> QGapMassSummary {
    let summary = summarize_q_gap_mass_with_prob_cap(data, q_gap_pp, PURIFY_MAX_PROB);
    assert!(
        summary.max_info_set_mass <= max_info_set_mass + 1e-12,
        "Q-gap residue assertion failed: max small-action (p < {:.2}%) info-set mass {:.4}% above {:.2}pp Q-gap, limit {:.4}%",
        PURIFY_MAX_PROB * 100.0,
        summary.max_info_set_mass * 100.0,
        q_gap_pp,
        max_info_set_mass * 100.0
    );
    summary
}

/// Purify finite-iteration average-strategy residue for chart export only.
///
/// `.teach` remains raw. The returned profile zeros small-probability actions
/// whose Q gap is large, then renormalizes the info-set row.
pub fn purify_teacher_probs(
    data: &TeacherData,
    config: PurificationConfig,
) -> (Vec<ActionProbs>, PurificationReport) {
    let mut purified = Vec::with_capacity(data.n_info_sets());
    let mut report = PurificationReport {
        max_prob: config.max_prob,
        min_q_gap_pp: config.min_q_gap_pp,
        info_sets: data.n_info_sets(),
        touched_info_sets: 0,
        actions_zeroed: 0,
        mass_removed: 0.0,
        max_info_set_mass_removed: 0.0,
        max_q_gap_touched_pp: 0.0,
    };

    for idx in 0..data.n_info_sets() {
        let probs = data.probs_of(idx);
        let q_values = data.q_of(idx);
        let best_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut row: ActionProbs = probs.iter().map(|&p| p as f64).collect();
        let mut removed = 0.0;
        let mut touched = false;
        for (i, p) in row.iter_mut().enumerate() {
            let gap_pp = (best_q - q_values[i] as f64).max(0.0) * PP_PER_Q_UNIT;
            if *p > 0.0 && *p < config.max_prob && gap_pp > config.min_q_gap_pp {
                removed += *p;
                *p = 0.0;
                touched = true;
                report.actions_zeroed += 1;
                report.max_q_gap_touched_pp = report.max_q_gap_touched_pp.max(gap_pp);
            }
        }
        if touched {
            let remaining: f64 = row.iter().sum();
            assert!(
                remaining > 0.0,
                "purification removed every action at info set {}",
                idx
            );
            for p in row.iter_mut() {
                *p /= remaining;
            }
            report.touched_info_sets += 1;
            report.mass_removed += removed;
            report.max_info_set_mass_removed = report.max_info_set_mass_removed.max(removed);
        }
        purified.push(row);
    }

    (purified, report)
}

struct QAccum<'a> {
    strategies: &'a [ActionProbs],
    offsets: &'a [u32],
    q_num: Vec<f64>,
    pts_num: Vec<f64>,
    own_mass: Vec<f64>,
    opp_mass: Vec<f64>,
}

/// Resolve the per-`table_idx` average strategies for a saved solution.
pub fn strategies_by_table_idx(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
) -> Result<Vec<ActionProbs>, StorageError> {
    let mut strategies = Vec::with_capacity(prebuilt.info_sets.len());
    let mut projected_rows = 0usize;
    let mut uniform_rows = 0usize;
    for (key, _, actions) in &prebuilt.info_sets {
        let probs = match table.data.get(key) {
            Some(data) => {
                let (probs, projected, uniform) = project_average_strategy_by_action(data, actions);
                projected_rows += usize::from(projected);
                uniform_rows += usize::from(uniform);
                probs
            }
            // Info sets never touched by the solve fall back to uniform —
            // same convention as the solver's own average_strategy.
            None => {
                uniform_rows += 1;
                crate::strategy::uniform_probs(actions.len())
            }
        };
        strategies.push(probs);
    }
    if projected_rows > 0 || uniform_rows > 0 {
        eprintln!(
            "strategy/tree action compatibility: projected {} rows by action identity; {} rows used uniform fallback",
            projected_rows,
            uniform_rows,
        );
    }
    Ok(strategies)
}

/// Project a saved policy onto the target tree's exact legal-action vector.
/// Treepacks and checkpoints are long-lived artifacts; a proof-backed action
/// prune may remove actions between their creation dates without changing an
/// info-set key. Probabilities therefore map by action identity, never slot.
/// Removed source actions are discarded and retained mass is renormalized. If
/// the target adds an action the saved policy never knew, fall back to uniform
/// rather than fabricating a strategically meaningful zero for that action.
fn project_average_strategy_by_action(
    source: &InfoSetData,
    target_actions: &[AbstractAction],
) -> (ActionProbs, bool, bool) {
    if source.actions.as_slice() == target_actions {
        return (source.average_strategy(), false, false);
    }

    let source_slots: Option<Vec<_>> = target_actions
        .iter()
        .map(|target| {
            source
                .actions
                .iter()
                .position(|candidate| candidate == target)
        })
        .collect();
    let Some(source_slots) = source_slots else {
        return (
            crate::strategy::uniform_probs(target_actions.len()),
            true,
            true,
        );
    };
    let source_probs = source.average_strategy();
    let mut projected: ActionProbs = source_slots
        .into_iter()
        .map(|slot| source_probs[slot])
        .collect();
    let retained: f64 = projected.iter().sum();
    if retained > 0.0 {
        for p in &mut projected {
            *p /= retained;
        }
        (projected, true, false)
    } else {
        (
            crate::strategy::uniform_probs(target_actions.len()),
            true,
            true,
        )
    }
}

/// Compute teacher tensors with a single σ̄-vs-σ̄ traversal per deal tree.
///
/// `dealer` selects which per-deal tree to walk (solve-tc jobs are
/// per-dealer). Values are computed exactly — no sampling.
pub fn compute_teacher_data(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
) -> (TeacherData, f64) {
    let n = prebuilt.info_sets.len();
    let mut offsets = Vec::with_capacity(n + 1);
    let mut acc = 0u32;
    for (_, _, actions) in &prebuilt.info_sets {
        offsets.push(acc);
        acc += actions.len() as u32;
    }
    offsets.push(acc);

    let mut accum = QAccum {
        strategies,
        offsets: &offsets,
        q_num: vec![0.0; acc as usize],
        pts_num: vec![0.0; acc as usize],
        own_mass: vec![0.0; n],
        opp_mass: vec![0.0; n],
    };

    let mut game_value = 0.0;
    let mut weight_sum = 0.0;
    for entry in &prebuilt.entries {
        let tree = if dealer == 0 {
            &entry.tree_dealer_0
        } else {
            &entry.tree_dealer_1
        };
        if tree.nodes.is_empty() {
            continue;
        }
        let (v, _pts) = q_traverse(
            tree,
            0,
            dealer,
            score,
            match_values,
            &mut accum,
            entry.weight,
            1.0,
            1.0,
        );
        game_value += entry.weight * v;
        weight_sum += entry.weight;
    }
    if weight_sum > 0.0 {
        game_value /= weight_sum;
    }

    // Counterfactual averages; zero-mass info sets keep Q/pts = 0 and are
    // filterable via opp_reach == 0.
    let QAccum {
        q_num,
        pts_num,
        own_mass,
        opp_mass,
        ..
    } = accum;
    let average_over_mass = |num: &[f64]| -> Vec<f32> {
        (0..n)
            .flat_map(|i| {
                let (lo, hi) = (offsets[i] as usize, offsets[i + 1] as usize);
                let mass = opp_mass[i];
                num[lo..hi]
                    .iter()
                    .map(move |&x| if mass > 0.0 { (x / mass) as f32 } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let q_values = average_over_mass(&q_num);
    let pts_values = average_over_mass(&pts_num);

    let probs: Vec<f32> = strategies
        .iter()
        .flat_map(|p| p.iter().map(|&x| x as f32))
        .collect();
    debug_assert_eq!(probs.len(), acc as usize);

    let data = TeacherData {
        offsets,
        probs,
        q_values,
        pts_values,
        own_reach: own_mass.iter().map(|&x| x as f32).collect(),
        opp_reach: opp_mass.iter().map(|&x| x as f32).collect(),
    };
    (data, game_value)
}

/// Returns `(match_value, pts_value)` in p0-perspective space — match equity
/// in ±1 space via [`terminal_p0_value`], and the raw hand-point differential
/// (acting player's points minus opponent's for this hand only) — while
/// accumulating per-action Q and pts numerators (acting player's perspective)
/// and reach masses. Both channels share one traversal since the recursive
/// combination (probability-weighted sum over actions) is identical whether
/// the terminal value is match equity or raw hand points.
#[allow(clippy::too_many_arguments)]
fn q_traverse(
    tree: &GameTree,
    node_id: NodeId,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    accum: &mut QAccum,
    deal_weight: f64,
    reach_p0: f64,
    reach_p1: f64,
) -> (f64, f64) {
    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => (
            terminal_p0_value(payoff_p0, dealer, score, match_values),
            payoff_p0,
        ),
        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            let idx = table_idx as usize;
            let probs = accum.strategies[idx].clone();
            let (own_reach, opp_reach, sign) = if player == 0 {
                (reach_p0, reach_p1, 1.0)
            } else {
                (reach_p1, reach_p0, -1.0)
            };
            let cf_weight = deal_weight * opp_reach;
            accum.own_mass[idx] += deal_weight * own_reach;
            accum.opp_mass[idx] += cf_weight;

            let base = accum.offsets[idx] as usize;
            let mut node_value = 0.0;
            let mut node_pts = 0.0;
            for (i, e) in edges.iter().enumerate() {
                let (r0, r1) = if player == 0 {
                    (reach_p0 * probs[i], reach_p1)
                } else {
                    (reach_p0, reach_p1 * probs[i])
                };
                let (v, pts) = q_traverse(
                    tree,
                    e.child,
                    dealer,
                    score,
                    match_values,
                    accum,
                    deal_weight,
                    r0,
                    r1,
                );
                accum.q_num[base + i] += cf_weight * sign * v;
                accum.pts_num[base + i] += cf_weight * sign * pts;
                node_value += probs[i] * v;
                node_pts += probs[i] * pts;
            }
            (node_value, node_pts)
        }
    }
}

/// Serialize teacher tensors. Atomic (tmp + rename), all sections f32/u32
/// little-endian at 8-aligned offsets after the u64 header.
pub fn save_teach(
    path: &Path,
    data: &TeacherData,
    score: &Score,
    tc: u8,
    dealer: Player,
    sig_hash: u64,
) -> Result<(), StorageError> {
    use std::io::Write;

    let n = data.n_info_sets() as u64;
    let flat = data.probs.len() as u64;
    if data.offsets.len() != n as usize + 1
        || data.q_values.len() != flat as usize
        || data.pts_values.len() != flat as usize
        || data.own_reach.len() != n as usize
        || data.opp_reach.len() != n as usize
    {
        return Err(StorageError::Serialize(format!(
            "inconsistent teacher tensor lengths: n={} flat={} offsets={} q={} pts={} own={} opp={}",
            n,
            flat,
            data.offsets.len(),
            data.q_values.len(),
            data.pts_values.len(),
            data.own_reach.len(),
            data.opp_reach.len(),
        )));
    }
    let header: [u64; HEADER_WORDS] = [
        MAGIC,
        VERSION,
        sig_hash,
        n,
        flat,
        ((score.zero as u64) << 8) | score.one as u64,
        ((tc as u64) << 8) | dealer as u64,
        0,
    ];

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let tmp = path.with_extension("teach.tmp");
    let mut w = std::io::BufWriter::with_capacity(
        8 << 20,
        File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?,
    );
    let io = |e: std::io::Error| StorageError::Io(e.to_string());
    for h in header {
        w.write_all(&h.to_le_bytes()).map_err(io)?;
    }
    for v in &data.offsets {
        w.write_all(&v.to_le_bytes()).map_err(io)?;
    }
    // pad offsets section to 8 bytes
    if data.offsets.len() % 2 == 1 {
        w.write_all(&[0u8; 4]).map_err(io)?;
    }
    for section in [
        &data.probs,
        &data.q_values,
        &data.pts_values,
        &data.own_reach,
        &data.opp_reach,
    ] {
        for v in section.iter() {
            w.write_all(&v.to_le_bytes()).map_err(io)?;
        }
        if section.len() % 2 == 1 {
            w.write_all(&[0u8; 4]).map_err(io)?;
        }
    }
    // `BufWriter::drop` deliberately discards flush errors. A large Study
    // export can otherwise be renamed and reported as successful even when
    // only a prefix reached disk, leaving `load_teach` to fail later with the
    // unhelpful "truncated" error. Flush and verify the exact wire length
    // before the atomic rename.
    w.flush().map_err(io)?;
    w.get_ref().sync_all().map_err(io)?;
    drop(w);
    let padded_f32_bytes = |count: usize| count * 4 + if count % 2 == 1 { 4 } else { 0 };
    let expected_len = HEADER_WORDS * 8
        + padded_f32_bytes(data.offsets.len())
        + padded_f32_bytes(data.probs.len())
        + padded_f32_bytes(data.q_values.len())
        + padded_f32_bytes(data.pts_values.len())
        + padded_f32_bytes(data.own_reach.len())
        + padded_f32_bytes(data.opp_reach.len());
    let actual_len = fs::metadata(&tmp)
        .map_err(|e| StorageError::Io(e.to_string()))?
        .len() as usize;
    if actual_len != expected_len {
        return Err(StorageError::Serialize(format!(
            "teacher wire length mismatch: expected {} bytes, wrote {}",
            expected_len, actual_len
        )));
    }
    fs::rename(&tmp, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Load a `.teach` file, validating magic/version/band binding and bounds.
pub fn load_teach(path: &Path, expect_sig_hash: Option<u64>) -> Result<TeacherData, StorageError> {
    let bytes = fs::read(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let word = |i: usize| -> Result<u64, StorageError> {
        bytes
            .get(i * 8..i * 8 + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(|| StorageError::Deserialize("teach truncated header".into()))
    };
    if word(0)? != MAGIC {
        return Err(StorageError::Deserialize("teach bad magic".into()));
    }
    if word(1)? != VERSION {
        return Err(StorageError::Deserialize("teach version mismatch".into()));
    }
    if let Some(expected) = expect_sig_hash {
        if word(2)? != expected {
            return Err(StorageError::Deserialize(
                "teach band-signature mismatch".into(),
            ));
        }
    }
    let n = word(3)? as usize;
    let flat = word(4)? as usize;

    let mut off = HEADER_WORDS * 8;
    let take_u32s = |off: &mut usize, count: usize| -> Result<Vec<u32>, StorageError> {
        let end = *off + count * 4;
        let s = bytes
            .get(*off..end)
            .ok_or_else(|| StorageError::Deserialize("teach truncated".into()))?;
        let v = s
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        *off = end + if count % 2 == 1 { 4 } else { 0 };
        Ok(v)
    };
    let offsets = take_u32s(&mut off, n + 1)?;
    let take_f32s = |off: &mut usize, count: usize| -> Result<Vec<f32>, StorageError> {
        let end = *off + count * 4;
        let s = bytes
            .get(*off..end)
            .ok_or_else(|| StorageError::Deserialize("teach truncated".into()))?;
        let v = s
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        *off = end + if count % 2 == 1 { 4 } else { 0 };
        Ok(v)
    };
    let probs = take_f32s(&mut off, flat)?;
    let q_values = take_f32s(&mut off, flat)?;
    let pts_values = take_f32s(&mut off, flat)?;
    let own_reach = take_f32s(&mut off, n)?;
    let opp_reach = take_f32s(&mut off, n)?;

    if offsets.last().copied() != Some(flat as u32) {
        return Err(StorageError::Deserialize("teach offsets corrupt".into()));
    }
    Ok(TeacherData {
        offsets,
        probs,
        q_values,
        pts_values,
        own_reach,
        opp_reach,
    })
}

/// Write the once-per-band info-set metadata sidecar (bincode of
/// `(key, InfoSet, actions)` in `table_idx` order) for consumers that do not
/// want to open the treepack.
pub fn save_band_meta(
    path: &Path,
    info_sets: &[(InfoSetKey, InfoSet, crate::game_tree::InfoActions)],
) -> Result<(), StorageError> {
    let raw: Vec<(u64, &InfoSet, &[AbstractAction])> =
        info_sets.iter().map(|(k, i, a)| (k.0, i, &a[..])).collect();
    let bytes = bincode_v1::serialize(&raw).map_err(|e| StorageError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(path, bytes).map_err(|e| StorageError::Io(e.to_string()))
}

const BRGAP_MAGIC: u64 = 0x5452_5543_4252_4741; // "TRUCBRGA"
const BRGAP_VERSION: u64 = 1;
/// Header layout (u64 words):
/// [magic, version, sig_hash, count, score0<<8|score1, tc<<8|dealer]
const BRGAP_HEADER_WORDS: usize = 6;

/// One info set's stored best-response quality record — plan 75
/// (`plans/75-per-infoset-solution-quality-br-gap.md`). `gap = br_value -
/// eq_value` is the adversarial off-equilibrium quality measure: how much a
/// genuine best-responder gains by deviating at this info set, versus the
/// value the shipped (purified) profile realizes there. `weight` is the
/// counterfactual weight (chance x opponent reach) the values were resolved
/// under. Full-tree: unlike the chart JSON's `"br"` field, this is not
/// filtered by `--min-depth`/`--max-depth` — the underlying best-response
/// pass already covers the whole game (see
/// `cfr::best_response_full_from_action_probs`), so there is no compute
/// reason to restrict coverage here.
#[derive(Clone, Copy, Debug)]
pub struct BrGapRecord {
    pub table_idx: u32,
    pub br_value: f32,
    pub eq_value: f32,
    pub gap: f32,
    pub weight: f32,
}

/// Serialize a full-tree per-info-set BR-gap table. Atomic (tmp + rename),
/// same little-endian/8-aligned-header convention as [`save_teach`]. Records
/// are written in the order given; callers should sort by `table_idx` for a
/// deterministic file.
pub fn save_br_gaps(
    path: &Path,
    records: &[BrGapRecord],
    score: &Score,
    tc: u8,
    dealer: Player,
    sig_hash: u64,
) -> Result<(), StorageError> {
    use std::io::Write;

    let header: [u64; BRGAP_HEADER_WORDS] = [
        BRGAP_MAGIC,
        BRGAP_VERSION,
        sig_hash,
        records.len() as u64,
        ((score.zero as u64) << 8) | score.one as u64,
        ((tc as u64) << 8) | dealer as u64,
    ];

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let tmp = path.with_extension("brgaps.tmp");
    let mut w = std::io::BufWriter::with_capacity(
        8 << 20,
        File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?,
    );
    let io = |e: std::io::Error| StorageError::Io(e.to_string());
    for h in header {
        w.write_all(&h.to_le_bytes()).map_err(io)?;
    }
    for r in records {
        w.write_all(&r.table_idx.to_le_bytes()).map_err(io)?;
        w.write_all(&r.br_value.to_le_bytes()).map_err(io)?;
        w.write_all(&r.eq_value.to_le_bytes()).map_err(io)?;
        w.write_all(&r.gap.to_le_bytes()).map_err(io)?;
        w.write_all(&r.weight.to_le_bytes()).map_err(io)?;
    }
    drop(w);
    fs::rename(&tmp, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Load a BR-gap table written by [`save_br_gaps`], validating magic/version
/// and (optionally) the band signature it's bound to.
pub fn load_br_gaps(
    path: &Path,
    expect_sig_hash: Option<u64>,
) -> Result<Vec<BrGapRecord>, StorageError> {
    let bytes = fs::read(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let word = |i: usize| -> Result<u64, StorageError> {
        bytes
            .get(i * 8..i * 8 + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(|| StorageError::Deserialize("brgaps truncated header".into()))
    };
    if word(0)? != BRGAP_MAGIC {
        return Err(StorageError::Deserialize("brgaps bad magic".into()));
    }
    if word(1)? != BRGAP_VERSION {
        return Err(StorageError::Deserialize("brgaps version mismatch".into()));
    }
    if let Some(expected) = expect_sig_hash {
        if word(2)? != expected {
            return Err(StorageError::Deserialize(
                "brgaps band-signature mismatch".into(),
            ));
        }
    }
    let count = word(3)? as usize;

    const RECORD_LEN: usize = 20; // u32 + 4×f32
    let start = BRGAP_HEADER_WORDS * 8;
    let end = start + count * RECORD_LEN;
    let body = bytes
        .get(start..end)
        .ok_or_else(|| StorageError::Deserialize("brgaps truncated body".into()))?;

    let mut records = Vec::with_capacity(count);
    for chunk in body.chunks_exact(RECORD_LEN) {
        let table_idx = u32::from_le_bytes(chunk[0..4].try_into().expect("4 bytes"));
        let br_value = f32::from_le_bytes(chunk[4..8].try_into().expect("4 bytes"));
        let eq_value = f32::from_le_bytes(chunk[8..12].try_into().expect("4 bytes"));
        let gap = f32::from_le_bytes(chunk[12..16].try_into().expect("4 bytes"));
        let weight = f32::from_le_bytes(chunk[16..20].try_into().expect("4 bytes"));
        records.push(BrGapRecord {
            table_idx,
            br_value,
            eq_value,
            gap,
            weight,
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{enumerate_deals, TurnupClass};
    use crate::cfr::{solve_with_limit_algo, CfrAlgorithm};
    use crate::game_tree::build_all_trees_with_dealer;

    #[test]
    fn projects_saved_average_onto_target_actions_by_identity() {
        let a = AbstractAction::Raise(3);
        let b = AbstractAction::Fold;
        let c = AbstractAction::AcceptRaise;
        let mut source = InfoSetData::new(vec![a, b, c]);
        source.cumulative_strategy = vec![20.0, 30.0, 50.0];

        let (projected, changed, uniform) = project_average_strategy_by_action(&source, &[c, a]);
        assert!(changed);
        assert!(!uniform);
        assert_eq!(projected.len(), 2);
        assert!((projected[0] - 5.0 / 7.0).abs() < 1e-12);
        assert!((projected[1] - 2.0 / 7.0).abs() < 1e-12);

        let missing = AbstractAction::Raise(6);
        let (fallback, changed, uniform) = project_average_strategy_by_action(&source, &[missing]);
        assert!(changed);
        assert!(uniform);
        assert_eq!(fallback.as_slice(), &[1.0]);

        let (fallback, changed, uniform) =
            project_average_strategy_by_action(&source, &[c, missing]);
        assert!(changed);
        assert!(uniform);
        assert_eq!(fallback.as_slice(), &[0.5, 0.5]);
    }

    #[test]
    fn teacher_data_matches_solver_game_value_and_round_trips() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let score = Score { zero: 11, one: 10 };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        // Small but real solve (strided deals) so σ̄ is meaningful.
        let (table, stats) = solve_with_limit_algo(
            score.clone(),
            tc,
            60,
            &mv,
            Some(400),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            1,
            None,
        );
        let (v_d0, _) = stats.game_value_per_dealer.expect("game value");

        let deals = {
            let mut all = enumerate_deals(&tc);
            crate::cfr::subsample_deals(&mut all, 400);
            all
        };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, None).unwrap();
        let strategies = strategies_by_table_idx(&prebuilt, &table).unwrap();
        let (data, game_value) = compute_teacher_data(&prebuilt, &strategies, 0, &score, &mv);

        // The σ̄-vs-σ̄ root value must reproduce the solver's own game value.
        assert!(
            (game_value - v_d0).abs() < 1e-9,
            "teacher game value {game_value} != solver {v_d0}"
        );

        // Strategy rows are probability vectors; Q stays in ±1 match space;
        // pts stays within the hand-point ladder (±12, plus float slack).
        for i in 0..data.n_info_sets() {
            let p = data.probs_of(i);
            assert!(!p.is_empty());
            let sum: f32 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "probs must normalize");
            for &q in data.q_of(i) {
                assert!((-1.0001..=1.0001).contains(&q), "Q out of match space");
            }
            for &pts in data.pts_of(i) {
                assert!(
                    (-12.0001..=12.0001).contains(&pts),
                    "pts out of the hand-point ladder"
                );
            }
        }

        // At a converged info set, the played (support) action should not be
        // dominated: check the highest-mass info set as a smoke signal.
        let hot = (0..data.n_info_sets())
            .max_by(|&a, &b| data.opp_reach[a].total_cmp(&data.opp_reach[b]))
            .unwrap();
        let p = data.probs_of(hot);
        let q = data.q_of(hot);
        let played = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        let best = q.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            q[played] >= best - 0.05,
            "most-played action is far from max-Q at the hottest info set"
        );

        // Save/load round-trip.
        let dir = std::env::temp_dir().join("truco_teach_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("t.teach");
        save_teach(&path, &data, &score, 0, 0, 42).unwrap();
        let loaded = load_teach(&path, Some(42)).unwrap();
        assert_eq!(loaded.offsets, data.offsets);
        assert_eq!(loaded.probs, data.probs);
        assert_eq!(loaded.q_values, data.q_values);
        assert_eq!(loaded.pts_values, data.pts_values);
        assert_eq!(loaded.own_reach, data.own_reach);
        assert_eq!(loaded.opp_reach, data.opp_reach);
        assert!(load_teach(&path, Some(43)).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for the eq_value/br_value opponent-model mismatch
    /// found while sanity-checking the 2026-07-11 BR-gap canary run: using
    /// `.teach`'s raw-continuation `q` for eq_value while `br_value` was
    /// computed against a purified-strategy opponent broke the
    /// `br_value >= eq_value` best-response guarantee (empirically, gaps
    /// went meaningfully negative even at high-weight nodes). This test
    /// reproduces `run_export_chart`'s corrected logic directly: eq_value
    /// must come from a `compute_teacher_data` pass run under the SAME
    /// (purified) profile that `best_response_gaps_from_action_probs` was
    /// best-responded against.
    #[test]
    fn br_value_never_below_eq_value_under_matched_purified_profile() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let score = Score { zero: 11, one: 10 };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (table, _stats) = solve_with_limit_algo(
            score.clone(),
            tc,
            60,
            &mv,
            Some(400),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            1,
            None,
        );

        let deals = {
            let mut all = enumerate_deals(&tc);
            crate::cfr::subsample_deals(&mut all, 400);
            all
        };
        // Dealer-restricted, matching real `run_export_chart` usage
        // (`cfr::load_or_build_trees(..., Some(dealer))`): with `None` the
        // registry spans BOTH dealer trees and `best_response_gaps_from_action_probs`
        // (which has no dealer filter of its own) picks up weight from
        // dealer-1-only info sets that a dealer-0-restricted
        // `compute_teacher_data` call never visits — a spurious mismatch
        // unrelated to the eq_value/br_value fix under test.
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let raw_strategies = strategies_by_table_idx(&prebuilt, &table).unwrap();
        let (raw_data, _) = compute_teacher_data(&prebuilt, &raw_strategies, 0, &score, &mv);
        let (purified_strategies, _report) =
            purify_teacher_probs(&raw_data, PurificationConfig::default());

        // The buggy quantity: eq_value from RAW-continuation q, compared
        // against a purified-opponent best response. Kept to prove the old
        // code path really did produce negative gaps (i.e. this test would
        // have caught the bug), not just to prove the new path is clean.
        let (purified_data, _) =
            compute_teacher_data(&prebuilt, &purified_strategies, 0, &score, &mv);

        let mut old_negative_gaps = 0usize;
        let mut new_negative_gaps = 0usize;
        let mut checked_with_weight = 0usize;

        for br_player in [0u8, 1u8] {
            let brs = crate::cfr::best_response_gaps_from_action_probs(
                &prebuilt,
                &purified_strategies,
                &score,
                &mv,
                br_player,
            );
            for r in brs {
                if r.weight <= 0.0 || !r.br_value.is_finite() {
                    continue;
                }
                let idx = r.table_idx as usize;
                let probs = &purified_strategies[idx];

                let old_eq_value: f64 = probs
                    .iter()
                    .zip(raw_data.q_of(idx).iter())
                    .map(|(&p, &qi)| p * qi as f64)
                    .sum();
                let new_eq_value: f64 = probs
                    .iter()
                    .zip(purified_data.q_of(idx).iter())
                    .map(|(&p, &qi)| p * qi as f64)
                    .sum();

                checked_with_weight += 1;
                if r.br_value < old_eq_value - 1e-6 {
                    old_negative_gaps += 1;
                }
                if r.br_value < new_eq_value - 1e-6 {
                    new_negative_gaps += 1;
                    eprintln!(
                        "table_idx={idx} player={br_player} br_value={:.6} eq_value={:.6} weight={:.6}",
                        r.br_value, new_eq_value, r.weight
                    );
                }
            }
        }

        assert!(checked_with_weight > 0, "no weighted info sets to check");
        assert_eq!(
            new_negative_gaps, 0,
            "{new_negative_gaps}/{checked_with_weight} info sets have br_value < eq_value \
             under the matched purified profile — best-response guarantee violated"
        );
        // Sanity: confirm this test setup is actually sensitive to the bug
        // (i.e. the raw-continuation eq_value really did produce negative
        // gaps on this fixture before the fix).
        assert!(
            old_negative_gaps > 0,
            "expected the raw-continuation eq_value to reproduce the known mismatch on this \
             fixture; if this now legitimately passes, the regression check is no longer useful"
        );
    }

    #[test]
    fn save_br_gaps_round_trips() {
        let score = Score { zero: 11, one: 10 };
        let records = vec![
            BrGapRecord {
                table_idx: 0,
                br_value: 0.125,
                eq_value: 0.100,
                gap: 0.025,
                weight: 0.5,
            },
            BrGapRecord {
                table_idx: 7,
                br_value: -0.3,
                eq_value: -0.32,
                gap: 0.02,
                weight: 1.25,
            },
        ];

        let dir = std::env::temp_dir().join("truco_brgaps_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("t.brgaps");
        save_br_gaps(&path, &records, &score, 0, 1, 99).unwrap();
        let loaded = load_br_gaps(&path, Some(99)).unwrap();
        assert_eq!(loaded.len(), records.len());
        for (a, b) in loaded.iter().zip(records.iter()) {
            assert_eq!(a.table_idx, b.table_idx);
            assert_eq!(a.br_value, b.br_value);
            assert_eq!(a.eq_value, b.eq_value);
            assert_eq!(a.gap, b.gap);
            assert_eq!(a.weight, b.weight);
        }
        assert!(load_br_gaps(&path, Some(100)).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    /// `best_response_full_from_action_probs` must be exactly equivalent to
    /// calling `best_response_value_from_action_probs` (aggregate) and
    /// `best_response_gaps_from_action_probs` (per-info-set detail)
    /// separately — it exists only to avoid paying for the underlying
    /// backward-induction pass twice, not to change what it computes.
    #[test]
    fn best_response_full_matches_separate_total_and_gaps_calls() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let score = Score { zero: 11, one: 10 };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (table, _stats) = solve_with_limit_algo(
            score.clone(),
            tc,
            60,
            &mv,
            Some(400),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            1,
            None,
        );
        let deals = {
            let mut all = enumerate_deals(&tc);
            crate::cfr::subsample_deals(&mut all, 400);
            all
        };
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();
        let strategies = strategies_by_table_idx(&prebuilt, &table).unwrap();

        for br_player in [0u8, 1u8] {
            let total = crate::cfr::best_response_value_from_action_probs(
                &prebuilt,
                &strategies,
                &score,
                &mv,
                br_player,
            );
            let gaps = crate::cfr::best_response_gaps_from_action_probs(
                &prebuilt,
                &strategies,
                &score,
                &mv,
                br_player,
            );
            let full = crate::cfr::best_response_full_from_action_probs(
                &prebuilt,
                &strategies,
                &score,
                &mv,
                br_player,
            );

            assert_eq!(
                full.total, total,
                "aggregate mismatch for player {br_player}"
            );
            assert_eq!(
                full.per_info_set.len(),
                gaps.len(),
                "per-info-set count mismatch for player {br_player}"
            );
            let mut full_sorted = full.per_info_set.clone();
            full_sorted.sort_by_key(|r| r.table_idx);
            let mut gaps_sorted = gaps;
            gaps_sorted.sort_by_key(|r| r.table_idx);
            for (a, b) in full_sorted.iter().zip(gaps_sorted.iter()) {
                assert_eq!(a.table_idx, b.table_idx);
                assert_eq!(a.weight, b.weight);
                assert!(
                    a.br_value.is_nan() && b.br_value.is_nan() || a.br_value == b.br_value,
                    "br_value mismatch at table_idx={}: {} vs {}",
                    a.table_idx,
                    a.br_value,
                    b.br_value
                );
            }
        }
    }

    /// Independent oracle: plain probability-weighted expectation of the raw
    /// terminal payoff (p0 perspective) under `strategies`, walked fresh
    /// (not via `q_traverse`) so it can cross-check `pts` arithmetic.
    fn expected_payoff_p0(tree: &GameTree, node_id: NodeId, strategies: &[ActionProbs]) -> f64 {
        match tree.view(node_id) {
            NodeView::Terminal { payoff_p0 } => payoff_p0,
            NodeView::Player {
                table_idx, edges, ..
            } => {
                let probs = &strategies[table_idx as usize];
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, e)| probs[i] * expected_payoff_p0(tree, e.child, strategies))
                    .sum()
            }
        }
    }

    /// Collects every reachable terminal's `|payoff_p0|` (hand-point stake)
    /// under `node_id`, ignoring probabilities — used to confirm a subtree
    /// actually reaches the stake magnitudes it should.
    fn collect_terminal_magnitudes(tree: &GameTree, node_id: NodeId, out: &mut Vec<i64>) {
        match tree.view(node_id) {
            NodeView::Terminal { payoff_p0 } => out.push(payoff_p0.abs().round() as i64),
            NodeView::Player { edges, .. } => {
                for e in edges.iter() {
                    collect_terminal_magnitudes(tree, e.child, out);
                }
            }
        }
    }

    /// Exercises `pts` sign convention, stake scaling, and fold terminals at
    /// a mão-de-onze root (a score with one side at 11 always opens on the
    /// AcceptEleven/FoldEleven decision — see `eleven_decision_children` in
    /// `cfr.rs`). Uses a single deal (so the root info set is touched by
    /// exactly one traversal, with no cross-deal aggregation) and a uniform
    /// strategy (pts is defined for ANY profile, not just an equilibrium, so
    /// no CFR solve is needed to check the arithmetic). Returns the acting
    /// player at the root, so the caller can confirm both signs are covered.
    fn check_pts_root_eleven(score: Score, dealer_filter: Option<Player>) -> Player {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut deals = enumerate_deals(&tc);
        crate::cfr::subsample_deals(&mut deals, 1);
        assert_eq!(deals.len(), 1, "expected exactly one deal for this check");

        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, dealer_filter).unwrap();
        let dealer = dealer_filter.unwrap_or(0);
        let tree = if dealer == 0 {
            &prebuilt.entries[0].tree_dealer_0
        } else {
            &prebuilt.entries[0].tree_dealer_1
        };
        let (player, table_idx, edges) = match tree.view(0) {
            NodeView::Player {
                player,
                table_idx,
                edges,
            } => (player, table_idx, edges),
            NodeView::Terminal { .. } => panic!("expected a decision at the root"),
        };
        let accept_i = edges
            .iter()
            .position(|e| e.action() == AbstractAction::AcceptEleven)
            .expect("root must offer AcceptEleven at a mão-de-onze score");
        let fold_i = edges
            .iter()
            .position(|e| e.action() == AbstractAction::FoldEleven)
            .expect("root must offer FoldEleven at a mão-de-onze score");

        let strategies: Vec<ActionProbs> = prebuilt
            .info_sets
            .iter()
            .map(|(_, _, actions)| crate::strategy::uniform_probs(actions.len()))
            .collect();
        let (data, _) = compute_teacher_data(
            &prebuilt,
            &strategies,
            dealer,
            &score,
            &MatchValueTable::new(),
        );

        let sign = if player == 0 { 1.0 } else { -1.0 };
        let pts = data.pts_of(table_idx as usize);

        // Fold terminal: FoldEleven concedes exactly the base stake (1 point)
        // with no further branching, so the oracle is the terminal itself.
        let fold_payoff = match tree.view(edges[fold_i].child) {
            NodeView::Terminal { payoff_p0 } => payoff_p0,
            NodeView::Player { .. } => panic!("FoldEleven must lead straight to a terminal"),
        };
        assert!(
            (fold_payoff.abs() - 1.0).abs() < 1e-9,
            "FoldEleven should concede exactly the base stake, got {fold_payoff}"
        );
        assert!(
            (pts[fold_i] as f64 - sign * fold_payoff).abs() < 1e-6,
            "pts sign/value mismatch on FoldEleven: pts={} expected={}",
            pts[fold_i],
            sign * fold_payoff
        );

        // Accept branch: `apply_accept_eleven` (truco-engine `engine.rs`)
        // unconditionally sets the hand value to `ELEVEN_AUTO_HAND_VALUE`
        // (3) and mão de onze disables further raises, so EVERY terminal
        // reachable from here is exactly stake 3 — a fixed 1 -> 3 jump from
        // the fold stake, not the base stake again and not the full ladder.
        let mut magnitudes = Vec::new();
        collect_terminal_magnitudes(tree, edges[accept_i].child, &mut magnitudes);
        magnitudes.sort_unstable();
        magnitudes.dedup();
        assert_eq!(
            magnitudes,
            vec![3],
            "AcceptEleven should fix the hand at stake 3 with no re-raise, got {magnitudes:?}"
        );

        let expected_accept = expected_payoff_p0(tree, edges[accept_i].child, &strategies);
        assert!(
            (pts[accept_i] as f64 - sign * expected_accept).abs() < 1e-3,
            "pts sign/value mismatch on AcceptEleven: pts={} expected={}",
            pts[accept_i],
            sign * expected_accept
        );

        player
    }

    #[test]
    fn pts_sign_convention_stake_scaling_and_fold_terminals() {
        // Whichever side sits at 11 gets the eleven decision; trying both
        // sides (and both dealer filters) exercises both sign branches.
        let mut seen_players = std::collections::HashSet::new();
        for score in [Score { zero: 11, one: 0 }, Score { zero: 0, one: 11 }] {
            for dealer in [0u8, 1u8] {
                seen_players.insert(check_pts_root_eleven(score.clone(), Some(dealer)));
            }
        }
        assert!(
            seen_players.contains(&0) && seen_players.contains(&1),
            "expected to exercise both acting-player signs, saw {seen_players:?}"
        );
    }

    /// Same oracle cross-check as [`check_pts_root_eleven`], but at a REGULAR
    /// (non-eleven) root, where the full raise ladder {1,3,6,9,12} is legal
    /// (min(score) == 0), to exercise `pts` across a genuinely re-raised
    /// stake sequence rather than the fixed 1 -> 3 eleven jump.
    fn check_pts_regular_root(score: Score, dealer_filter: Option<Player>) {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut deals = enumerate_deals(&tc);
        crate::cfr::subsample_deals(&mut deals, 1);
        assert_eq!(deals.len(), 1, "expected exactly one deal for this check");

        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, dealer_filter).unwrap();
        let dealer = dealer_filter.unwrap_or(0);
        let tree = if dealer == 0 {
            &prebuilt.entries[0].tree_dealer_0
        } else {
            &prebuilt.entries[0].tree_dealer_1
        };
        let (player, table_idx, edges) = match tree.view(0) {
            NodeView::Player {
                player,
                table_idx,
                edges,
            } => (player, table_idx, edges),
            NodeView::Terminal { .. } => panic!("expected a decision at the root"),
        };
        assert!(
            edges.iter().any(|e| e.action() == AbstractAction::Raise(3)),
            "expected the raise ladder open at the root of a min(score)==0 hand"
        );

        let mut magnitudes = Vec::new();
        collect_terminal_magnitudes(tree, 0, &mut magnitudes);
        magnitudes.sort_unstable();
        magnitudes.dedup();
        assert!(
            magnitudes.len() > 2,
            "expected several stake magnitudes across the full raise ladder, got {magnitudes:?}"
        );
        assert!(
            magnitudes.iter().all(|&m| [1, 3, 6, 9, 12].contains(&m)),
            "unexpected stake magnitude in {magnitudes:?}"
        );

        let strategies: Vec<ActionProbs> = prebuilt
            .info_sets
            .iter()
            .map(|(_, _, actions)| crate::strategy::uniform_probs(actions.len()))
            .collect();
        let (data, _) = compute_teacher_data(
            &prebuilt,
            &strategies,
            dealer,
            &score,
            &MatchValueTable::new(),
        );
        let sign = if player == 0 { 1.0 } else { -1.0 };
        let pts = data.pts_of(table_idx as usize);

        for (i, e) in edges.iter().enumerate() {
            let expected = expected_payoff_p0(tree, e.child, &strategies);
            assert!(
                (pts[i] as f64 - sign * expected).abs() < 1e-3,
                "pts sign/value mismatch on action {:?}: pts={} expected={}",
                e.action(),
                pts[i],
                sign * expected
            );
        }
    }

    #[test]
    fn pts_matches_oracle_at_full_raise_ladder_node() {
        check_pts_regular_root(Score { zero: 1, one: 0 }, Some(0));
        check_pts_regular_root(Score { zero: 0, one: 1 }, Some(1));
    }
}
