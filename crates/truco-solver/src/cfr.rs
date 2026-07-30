use std::io::IsTerminal;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use truco_engine::{Player, Score, MATCH_TARGET};

use crate::abstraction::{enumerate_deals, AbstractHand, TurnupClass};
use crate::game_tree::{
    build_all_trees_with_dealer, GameTree, NodeId, NodeView, PackedEdge, PolicyLookup,
    PolicyValueSource, PrebuiltTrees,
};
use crate::info_set::InfoSet;
use crate::info_set::{AbstractAction, ActionHistory, InfoSetKey};
use crate::match_value::MatchValueTable;
use crate::storage::{save_checkpoint_iter, CheckpointMeta, CheckpointStream, StorageError};
use crate::strategy::{ActionProbs, InfoSetData, StrategyTable};

/// Solve-time accumulator element (plan 84 Phase 1). `f64` by default; the
/// `accum-f32` feature narrows [`DenseAccum`]'s regret/strategy/pending/last
/// to `f32`, halving hot accumulator RAM. All traversal math stays `f64` —
/// narrowing happens only at the accumulator read/write boundary — and all
/// on-disk artifacts stay f64-formatted (widened on write).
#[cfg(feature = "accum-f32")]
pub(crate) type Acc = f32;
#[cfg(not(feature = "accum-f32"))]
pub(crate) type Acc = f64;

/// An externally-frozen accept/fold policy for the mão-de-onze (eleven) decision.
///
/// At an asymmetric mão-de-onze state (e.g. 11x10), the score-11 player (the
/// DECIDER) acts first at the empty history with the 2-action set
/// `{AcceptEleven, FoldEleven}`. Historically that info set had the SAME
/// abstraction key in the dealer-0 and dealer-1 trees (the abstraction did not
/// encode position), so it could not be learned per-dealer by CFR alone and was
/// frozen externally, per dealer, keyed by the decider's starting hand. `InfoSet`
/// now encodes position (`is_dealer`), so plain CFR learns the accept per dealer
/// jointly with the card play — that is the mainline path. The freeze machinery
/// is kept for diagnostics and controlled experiments (`solve-asym`).
///
/// `accept[dealer]` holds the set of decider hands that ACCEPT in that dealer
/// arrangement. Membership ⇒ accept; absence ⇒ fold.
#[derive(Clone, Debug, Default)]
pub struct AcceptPolicy {
    pub accept: [std::collections::HashSet<AbstractHand>; 2],
}

impl AcceptPolicy {
    /// Whether the given decider `hand` accepts under `dealer` (0 or 1).
    pub fn accepts(&self, dealer: Player, hand: &AbstractHand) -> bool {
        self.accept[dealer as usize].contains(hand)
    }
}

/// Detect an "eleven-decision" node: a player node whose action set is exactly
/// `{AcceptEleven, FoldEleven}`. Returns the (accept_child, fold_child) node ids
/// when it is one, else `None`.
///
/// Such a node belongs to the DECIDER (the score-11 player). Its action list is
/// always `[AcceptEleven, FoldEleven]` (engine order), but we locate each child
/// by action rather than position to be robust.
fn eleven_decision_children(edges: &[PackedEdge]) -> Option<(NodeId, NodeId)> {
    if edges.len() != 2 {
        return None;
    }
    let accept = edges
        .iter()
        .find(|e| e.action() == AbstractAction::AcceptEleven)
        .map(|e| e.child)?;
    let fold = edges
        .iter()
        .find(|e| e.action() == AbstractAction::FoldEleven)
        .map(|e| e.child)?;
    Some((accept, fold))
}

/// Statistics collected during a solve.
#[derive(Clone, Debug)]
pub struct SolveStats {
    pub score: (u8, u8),
    pub turnup_class: u8,
    pub iterations: u64,
    pub num_deals: usize,
    pub num_info_sets: usize,
    pub total_nodes: usize,
    pub total_duration_secs: f64,
    pub build_tree_secs: f64,
    pub per_iteration_secs: f64,
    pub estimated_memory_bytes: usize,
    pub exploitability: Option<f64>,
    /// (iteration, exploitability) pairs tracked during the solve.
    pub exploitability_history: Vec<(u64, f64)>,
    /// Expected payoff to player 0 in `[-1, 1]` under average vs average (filled for CFR+ tree solves).
    /// Average of the two per-dealer values (back-compat).
    pub game_value_p0: Option<f64>,
    /// Player 0's value in `[-1, 1]` (avg vs avg) split by who deals the hand:
    /// `(value_when_p0_deals, value_when_p1_deals)`. Filled for tree solves.
    pub game_value_per_dealer: Option<(f64, f64)>,
}

impl std::fmt::Display for SolveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Score: {}x{} | TC: {}",
            self.score.0, self.score.1, self.turnup_class
        )?;
        writeln!(f, "Iterations: {}", self.iterations)?;
        writeln!(
            f,
            "Deals: {} | Info sets: {} | Nodes: {}",
            self.num_deals, self.num_info_sets, self.total_nodes
        )?;
        writeln!(
            f,
            "Time: {:.1}s total ({:.2}s tree build, {:.3}s/iter)",
            self.total_duration_secs, self.build_tree_secs, self.per_iteration_secs
        )?;
        writeln!(
            f,
            "Memory: {:.1} MB",
            self.estimated_memory_bytes as f64 / 1_048_576.0
        )?;
        if let Some(expl) = self.exploitability {
            writeln!(f, "Exploitability: {:.6}", expl)?;
        }
        if !self.exploitability_history.is_empty() {
            writeln!(f, "Convergence:")?;
            for &(iter, expl) in &self.exploitability_history {
                writeln!(f, "  iter {:>5}: expl {:.6}", iter, expl)?;
            }
        }
        if let Some(gv) = self.game_value_p0 {
            writeln!(f, "Game value (P0, avg vs avg): {:.6}", gv)?;
        }
        if let Some((v_d0, v_d1)) = self.game_value_per_dealer {
            writeln!(
                f,
                "Game value per dealer (P0): p0-deals {:+.6} | p1-deals {:+.6}",
                v_d0, v_d1
            )?;
        }
        Ok(())
    }
}

/// Run CFR+ for a given score state and turnup class.
/// Phase 1: Build game trees for all deals (one-time cost).
/// Phase 2: Run CFR+ iterations over the pre-built trees (fast).
pub fn solve(
    score: Score,
    tc: TurnupClass,
    iterations: u64,
    match_values: &MatchValueTable,
) -> (StrategyTable, SolveStats) {
    solve_with_limit(score, tc, iterations, match_values, None, None)
}

/// Like `solve`, but optionally limits the number of deals processed and/or
/// restricts the solve to a single dealer arrangement (the two dealer games are
/// independent — see [`build_all_trees_with_dealer`]).
/// Useful for quick tests (pass `Some(500)` to use only 500 deals instead of ~140k).
pub fn solve_with_limit(
    score: Score,
    tc: TurnupClass,
    iterations: u64,
    match_values: &MatchValueTable,
    max_deals: Option<usize>,
    dealer_filter: Option<Player>,
) -> (StrategyTable, SolveStats) {
    solve_with_limit_algo(
        score,
        tc,
        iterations,
        match_values,
        max_deals,
        dealer_filter,
        CfrAlgorithm::CfrPlus,
        1,
        1,
        None,
    )
}

/// Like [`solve_with_limit`], but with an explicit algorithm and exploitability
/// cadence — the deal-limited counterpart of `solve_until` for experiments that
/// must run locally (e.g. mid-density convergence probes).
#[allow(clippy::too_many_arguments)]
pub fn solve_with_limit_algo(
    score: Score,
    tc: TurnupClass,
    iterations: u64,
    match_values: &MatchValueTable,
    max_deals: Option<usize>,
    dealer_filter: Option<Player>,
    algorithm: CfrAlgorithm,
    expl_every: u64,
    jobs: usize,
    tremble: Option<TrembleSchedule>,
) -> (StrategyTable, SolveStats) {
    let start = Instant::now();

    // Phase 1: enumerate deals and build game trees
    let mut deals = enumerate_deals(&tc);
    if let Some(limit) = max_deals {
        subsample_deals(&mut deals, limit);
    }
    let num_deals = deals.len();
    info!(
        "Solving score ({}, {}) tc={} | {} deals, {} iterations",
        score.zero, score.one, tc.blocked_plain_level, num_deals, iterations
    );

    let build_start = Instant::now();
    let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, dealer_filter)
        .expect("tree building failed: enumerate_deals produces valid deals");
    let build_secs = build_start.elapsed().as_secs_f64();

    let total_nodes: usize = prebuilt
        .entries
        .iter()
        .map(|e| e.tree_dealer_0.nodes.len() + e.tree_dealer_1.nodes.len())
        .sum();

    info!(
        "  Built {} game trees ({} total nodes) in {:.1}s",
        num_deals * 2,
        total_nodes,
        build_secs
    );

    info!("  {} unique info sets", prebuilt.info_sets.len());

    // Compute terminal payoffs using match values
    // At 11x11, all hand results are terminal (winning gives >= 12)
    // For other scores, we need continuation values.

    // Phase 2: CFR+ iterations with within-iteration progress bar
    let iter_start = Instant::now();
    let mut exploitability_history: Vec<(u64, f64)> = Vec::new();
    let num_entries = prebuilt.entries.len() as u64;

    let pb = ProgressBar::new(num_entries);
    pb.set_style(
        ProgressStyle::with_template(
            "  iter {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} deals ({per_sec})",
        )
        .expect("valid progress bar template")
        .progress_chars("█▓▒░  "),
    );

    let mode = SweepMode::of(&algorithm);
    // Hot-loop representation: SoA accumulators built directly from the info-set
    // offsets (no StrategyTable round-trip). `table` is created fresh at the end.
    let mut dense = DenseAccum::zeros(&prebuilt.info_sets);
    let mut table = StrategyTable::new();
    if mode.buffered() {
        dense.ensure_buffers(mode == SweepMode::Predictive);
    }
    let parallel = parallel_setup(jobs, mode, &prebuilt.info_sets);
    let mut par_bufs: Vec<LocalAccum> = parallel
        .as_ref()
        .map(|p| (0..p.jobs).map(|_| LocalAccum::zeros(p.total)).collect())
        .unwrap_or_default();
    for t in 1..=iterations {
        let traversing = (t % 2) as Player;
        let tremble_eps = tremble.map(|ts| ts.eps_at(t, 0, iterations)).unwrap_or(0.0);

        pb.set_message(format!("{:>4}/{}", t, iterations));
        pb.set_position(0);
        pb.reset_elapsed();

        if let Some(plan) = &parallel {
            plan.pool.install(|| {
                parallel_sync_sweep(
                    &prebuilt,
                    &mut dense,
                    plan,
                    &mut par_bufs,
                    traversing,
                    t,
                    &score,
                    match_values,
                    None,
                    tremble_eps,
                    None,
                )
            });
        } else {
            for entry in &prebuilt.entries {
                for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
                    if tree.nodes.is_empty() {
                        continue; // dealer excluded by the build's dealer filter
                    }
                    cfr_traverse_tree(
                        tree,
                        0,
                        traversing,
                        dealer,
                        [1.0, 1.0],
                        entry.weight,
                        t,
                        &score,
                        match_values,
                        None,
                        mode,
                        &prebuilt.info_sets,
                        &mut dense,
                        tremble_eps,
                        None,
                        None,
                    );
                }
                pb.inc(1);
            }

            if mode.buffered() {
                fold_pending_regrets(&mut dense, mode, &prebuilt.info_sets, traversing);
            }
        }

        // DCFR discounting (same schedule as `solve_until`).
        if let CfrAlgorithm::Dcfr { alpha, beta, gamma } = &algorithm {
            let t_f = t as f64;
            let pos_discount = t_f.powf(*alpha) / (t_f.powf(*alpha) + 1.0);
            let neg_discount = t_f.powf(*beta) / (t_f.powf(*beta) + 1.0);
            let strat_discount = (t_f / (t_f + 1.0)).powf(*gamma);
            for r in dense.regret.iter_mut() {
                if *r > 0.0 {
                    *r *= pos_discount as Acc;
                } else {
                    *r *= neg_discount as Acc;
                }
            }
            for st in dense.strategy.iter_mut() {
                *st *= strat_discount as Acc;
            }
        }

        if t % expl_every == 0 || t == iterations {
            let expl = compute_exploitability_dense(&prebuilt, &dense, &score, match_values, None);
            exploitability_history.push((t, expl));
            pb.println(format!(
                "  iter {:>4}/{}: expl = {:.6}",
                t, iterations, expl
            ));
        }
    }

    pb.finish_and_clear();

    let total_secs = start.elapsed().as_secs_f64();
    let iter_secs = iter_start.elapsed().as_secs_f64();
    let final_expl = exploitability_history.last().map(|h| h.1);
    let (v_d0, v_d1) =
        compute_game_value_per_dealer_dense(&prebuilt, &dense, &score, match_values, None);
    let game_value_p0 = Some((v_d0 + v_d1) / 2.0);
    let game_value_per_dealer = Some((v_d0, v_d1));

    dense.into_table(&prebuilt.info_sets, &mut table);
    let mem = estimate_memory_bytes(&table, &prebuilt);

    let stats = SolveStats {
        score: (score.zero, score.one),
        turnup_class: tc.blocked_plain_level,
        iterations,
        num_deals,
        num_info_sets: table.len(),
        total_nodes,
        total_duration_secs: total_secs,
        build_tree_secs: build_secs,
        per_iteration_secs: if iterations > 0 {
            iter_secs / iterations as f64
        } else {
            0.0
        },
        estimated_memory_bytes: mem,
        exploitability: final_expl,
        exploitability_history,
        game_value_p0,
        game_value_per_dealer,
    };

    info!("{}", stats);

    (table, stats)
}

/// Load a band-shared treepack when cached, else build and cache it.
pub fn load_or_build_trees(
    cache_dir: Option<&std::path::Path>,
    score: &Score,
    tc: TurnupClass,
    deals: &[crate::abstraction::AbstractDeal],
    dealer_filter: Option<Player>,
) -> std::sync::Arc<PrebuiltTrees> {
    let tc_level = tc.blocked_plain_level;
    if let Some(dir) = cache_dir {
        let path = dir.join(crate::treepack::treepack_name(
            score,
            tc_level,
            dealer_filter,
        ));
        if path.exists() {
            match crate::treepack::load_treepack(&path, score, tc_level, dealer_filter) {
                Ok(p) => {
                    info!("  loaded treepack {} (mmap)", path.display());
                    return std::sync::Arc::new(p);
                }
                Err(e) => log::warn!("  treepack {} unusable ({}); rebuilding", path.display(), e),
            }
        }
        let prebuilt = build_all_trees_with_dealer(score, tc, deals, dealer_filter)
            .expect("tree building failed: enumerate_deals produces valid deals");
        match crate::treepack::save_treepack(&path, &prebuilt, score, tc_level, dealer_filter) {
            Ok(()) => info!("  saved treepack {}", path.display()),
            Err(e) => log::warn!("  treepack save failed: {}", e),
        }
        return std::sync::Arc::new(prebuilt);
    }
    std::sync::Arc::new(
        build_all_trees_with_dealer(score, tc, deals, dealer_filter)
            .expect("tree building failed: enumerate_deals produces valid deals"),
    )
}

/// Algorithm variant for CFR solving.
#[derive(Clone, Debug)]
pub enum CfrAlgorithm {
    /// Standard CFR+: clamp negative regrets to 0, uniform iteration weighting.
    CfrPlus,
    /// Discounted CFR: weight iteration t by t^α for regrets and t^β for
    /// average strategy. Recommended: α=1.5, β=0, γ=2.
    /// Often converges 2-3× faster than CFR+ in practice.
    Dcfr { alpha: f64, beta: f64, gamma: f64 },
    /// Synchronous CFR+: identical to CFR+ except the strategy is FROZEN for
    /// the whole deal sweep — instantaneous regrets accumulate in a buffer and
    /// fold into cumulative regret (with the RM+ clamp) at iteration end. The
    /// classic CFR+ analysis assumes this; the historical solver recomputed
    /// strategies from live regrets mid-sweep (asynchronous), which converges
    /// on low-pooling configs but floors when info sets are shared across many
    /// deals (the mão-de-onze accept wall).
    SyncCfrPlus,
    /// Predictive CFR+ (PCFR+): synchronous sweep plus optimistic regret
    /// matching — play proportional to `max(0, R + r_last)` where `r_last` is
    /// the previous iteration's instantaneous regret. Optimism damps
    /// regret-matching oscillation on cyclic couplings (accept screening).
    PcfrPlus,
}

impl CfrAlgorithm {
    /// Recommended DCFR parameters from Brown & Sandholm (2019).
    pub fn dcfr_default() -> Self {
        CfrAlgorithm::Dcfr {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        }
    }
}

impl Default for CfrAlgorithm {
    fn default() -> Self {
        CfrAlgorithm::CfrPlus
    }
}

/// Epsilon-tremble schedule for a "trembling-hand" perturbed-game refinement
/// pass (warm-started resume of an existing checkpoint; see `SOLVER_PLAN.md`
/// / `RESEARCH_NARRATIVE.md` 2026-07-11, and `QUESTIONS.md` Q3).
///
/// At every player node, the strategy actually used to propagate reach,
/// compute regret, AND accumulate the average strategy is floored:
/// `σ'(a) = ε/|A| + (1-ε)·σ(a)`, where σ is the ordinary regret-matching (or
/// predictive) strategy for that sweep mode. This runs CFR on a PERTURBED
/// extensive-form game — every info set's behavior strategy is forced into
/// the interior of the simplex, one totally-mixed game per Selten/van Damme's
/// trembling-hand construction — not on the base game. Consequences:
/// - every info set receives real counterfactual visits every iteration (own
///   reach has an `(ε/|A|)^depth` floor), so previously-untrained descendants
///   get regret-matched against real data instead of uniform initialization
///   noise;
/// - the resulting average strategy is an ε-equilibrium of the PERTURBED
///   game, not an exact equilibrium of the base game — as ε→0 this
///   approaches a sequentially rational (quasi-perfect-flavoured) refinement
///   of the base equilibrium, but is not identical to it at any ε>0;
/// - the zero-prob branch-pruning fast path in the traversal is a no-op while
///   trembling is active (every action has probability ≥ ε/|A| > 0), so
///   sweeps revert to full-width tree walks — budget iteration cost close to
///   an EARLY/unconverged sweep's cost, not a late/pruned one's.
///
/// Exact-BR certification (`compute_exploitability*`/`best_response_value*`)
/// is untouched by this struct: it always measures the true best response
/// against whatever ends up in the accumulators, trembled or not — trembling
/// only changes what gets accumulated during solving, never how it is later
/// certified.
#[derive(Clone, Copy, Debug)]
pub struct TrembleSchedule {
    /// Tremble weight at the start of the (possibly resumed) run.
    pub eps_start: f64,
    /// Tremble weight at `max_iters`; the schedule anneals linearly toward
    /// this so the perturbation shrinks over the added iterations. Set equal
    /// to `eps_start` for a constant tremble.
    pub eps_end: f64,
}

impl TrembleSchedule {
    /// Tremble weight for iteration `t`, linearly interpolated over
    /// `[start_iter, max_iters]`. Falls back to `eps_start` when the horizon
    /// is unknown (`max_iters == u64::MAX`) or degenerate, since annealing
    /// needs a known endpoint.
    pub fn eps_at(&self, t: u64, start_iter: u64, max_iters: u64) -> f64 {
        if max_iters == u64::MAX || max_iters <= start_iter {
            return self.eps_start;
        }
        let progress = (t.saturating_sub(start_iter)) as f64 / (max_iters - start_iter) as f64;
        let progress = progress.clamp(0.0, 1.0);
        self.eps_start + (self.eps_end - self.eps_start) * progress
    }
}

/// Reversible regret-based pruning for synchronous CFR+ sweeps.
///
/// CFR+ clamps the strategy-driving cumulative regret at zero, so it cannot
/// distinguish a barely losing zero-regret action from one that has been
/// consistently bad. When enabled, the solver therefore maintains a separate
/// opt-in `f32` shadow sum of the UNCLAMPED instantaneous regrets. After the
/// warm-up, a traverser's action is temporarily skipped only when both its
/// current CFR+ probability is exactly zero and its shadow regret is below
/// `-threshold`. Every `revisit_every_rounds` pair of player sweeps is full
/// width, allowing a formerly bad action and its descendants to recover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegretPruningConfig {
    /// Number of individual alternating-player sweeps before any pruning.
    pub warmup_iters: u64,
    /// Absolute negative shadow-regret magnitude required for pruning.
    pub threshold: f32,
    /// Full-width revisit cadence in two-sweep CFR rounds.
    pub revisit_every_rounds: u64,
}

impl RegretPruningConfig {
    fn after_start_iter(mut self, start_iter: u64) -> Self {
        // The unclamped shadow is intentionally not checkpointed. After an
        // exact resume it starts empty, so give it a fresh warm-up instead of
        // treating the checkpoint's absolute iteration as pruning evidence.
        self.warmup_iters = self.warmup_iters.saturating_add(start_iter);
        self
    }

    fn prunes_on_iteration(&self, iteration: u64) -> bool {
        if iteration <= self.warmup_iters {
            return false;
        }
        let round = iteration.saturating_sub(1) / 2;
        !round.is_multiple_of(self.revisit_every_rounds)
    }
}

/// Floor `strategy` at `σ'(a) = ε/|A| + (1-ε)·σ(a)` (a no-op copy when
/// `eps <= 0.0`). See [`TrembleSchedule`] for the semantics.
#[inline]
fn tremble_strategy(mut strategy: ActionProbs, eps: f64) -> ActionProbs {
    if eps <= 0.0 {
        return strategy;
    }
    let floor = eps / strategy.len() as f64;
    for p in strategy.iter_mut() {
        *p = floor + (1.0 - eps) * *p;
    }
    strategy
}

/// Configuration for a CFR solve run.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    pub max_iters: u64,
    pub target_expl: f64,
    pub algorithm: CfrAlgorithm,
    /// Compute exploitability every N iterations. Higher = faster, less granular.
    pub expl_every: u64,
    /// Optional wall-clock budget in seconds. When set, the solve stops once the
    /// elapsed time (covering tree build + iterations) reaches this budget.
    pub time_budget_secs: Option<f64>,
    /// Optional path to write a full-state checkpoint during/after the solve.
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// Write a checkpoint at least this often (in seconds) while solving.
    pub checkpoint_every_secs: Option<f64>,
    /// Optional warm-start source: a solved table from a structurally related
    /// higher state (e.g. 11x11). Its card-play info sets are copied into this
    /// solve's matching info sets — matched by (player, tc, hand, history) after
    /// stripping a single leading AcceptEleven from this state's history. Lets an
    /// asymmetric state (e.g. 11x10) start from the equilibrium card play and only
    /// learn the accept/fold decision plus the opponent's range adaptation.
    pub warmstart_source: Option<std::sync::Arc<StrategyTable>>,
    /// Disk-backed warm-start source loaded only while regrets are copied.
    /// This avoids retaining a second full strategy table throughout a large
    /// pipeline solve.
    pub warmstart_checkpoint: Option<std::path::PathBuf>,
    /// Iteration to continue average-strategy weighting from when warm-starting
    /// (so the transferred cumulative strategy keeps its effective weight).
    pub warmstart_iter: u64,
    /// The warm-start source has the same score-band tree signature as this
    /// solve. Every matching info-set key/action vector transfers regrets
    /// directly, while the cumulative average and iteration counter reset so
    /// the neighboring score's different terminal utilities do not create
    /// average-strategy inertia.
    pub warmstart_same_band: bool,
    /// Explicitly allow a checkpoint from another score and/or turn-up class
    /// in the same score-band/dealer game. Target info sets are matched by
    /// rewriting only their public turn-up-class field when needed; card/action
    /// identities keep their abstract strength meaning. Both regret and
    /// average-strategy accumulators transfer so a tremble-trained off-path
    /// policy survives. Every target still requires independent exact-BR
    /// certification because terminal utilities and chance weights can differ.
    /// Rows made impossible by the different blocked-card multiplicity are
    /// left fresh. This is experimental and must never happen implicitly.
    pub warmstart_cross_turnup: bool,
    /// Optional externally-frozen mão-de-onze accept/fold policy. When set, every
    /// eleven-decision node (the decider's `{AcceptEleven, FoldEleven}` node) is
    /// forced to the policy's choice in all three traversals (CFR forward pass,
    /// equity pass, and best response) instead of using regret matching / the
    /// stored average strategy. `None` ⇒ unchanged behavior.
    pub accept_policy: Option<std::sync::Arc<AcceptPolicy>>,
    /// When resuming, warm-start the regrets only and reset the cumulative average
    /// strategy + iteration counter to 0. Use when the game changed between solves
    /// (e.g. a new frozen accept set) so the average isn't dragged by stale play.
    pub resume_average_reset: bool,
    /// Restrict the solve to a single dealer arrangement. The two dealer games
    /// share no info sets (position is in the info set) and interact only via
    /// the match-value table, so per-dealer solves are exact and halve the tree
    /// memory. `None` ⇒ solve both dealer games in one process.
    pub dealer_filter: Option<Player>,
    /// Optional deterministic, strided deal subset for cheap experiments.
    /// Production solves leave this unset. A subset must not share a tree-cache
    /// path with a full solve because its prebuilt tree arena is different.
    pub max_deals: Option<usize>,
    /// Worker threads for the deal sweep. >1 requires a buffered sweep mode
    /// (SyncCFR+): the strategy is frozen per iteration, so traversal is
    /// read-only on the accumulators and deals fan out across threads with
    /// atomic accumulation. Cuts a job's RAM-hours by ~the thread count.
    pub jobs: usize,
    /// Directory of treepack artifacts. Trees are identical for every score
    /// state within a ladder band, so the artifact `treepack_name(score, tc,
    /// dealer_filter)` is loaded (memory-mapped — larger-than-RAM arenas
    /// stream from disk) when present, and written after a fresh build.
    pub tree_cache: Option<std::path::PathBuf>,
    /// Optional already-built solver arena, used by restricted-game research.
    /// Its deal order and dealer filter must match this solve's configuration.
    pub prebuilt_override: Option<std::sync::Arc<PrebuiltTrees>>,
    /// Optional ε-tremble schedule (see [`TrembleSchedule`]). `None` (the
    /// default) is the exact historical behavior — no perturbation, no extra
    /// per-node cost. Set to warm-start-refine an existing checkpoint so
    /// every info set accumulates a trained average strategy.
    pub tremble: Option<TrembleSchedule>,
    /// Optional reversible regret pruning. Supported only by SyncCFR+ and
    /// intentionally incompatible with trembling (which gives every action
    /// positive probability and is meant to train all branches).
    pub regret_pruning: Option<RegretPruningConfig>,
    /// Write the final average-strategy artifact directly from the dense
    /// accumulators to this path before any hash-table rebuild. Combined with
    /// `skip_return_table` this removes the end-of-solve peak where dense
    /// accumulators and a full `StrategyTable` coexist.
    pub strategy_output: Option<std::path::PathBuf>,
    /// Return an empty `StrategyTable` instead of rebuilding one from the
    /// dense accumulators. Only valid when the caller consumes artifacts
    /// (checkpoint / `strategy_output` / stats) rather than the table itself.
    pub skip_return_table: bool,
    /// Pruning rule set for the tree this solve builds. Default `Current`.
    /// `AsymmetricRaisePrune` is experimental (value-preservation gate).
    pub tree_rules: crate::game_tree::TreeRules,
}

impl Default for SolveConfig {
    fn default() -> Self {
        SolveConfig {
            max_iters: u64::MAX,
            target_expl: 0.01,
            algorithm: CfrAlgorithm::CfrPlus,
            expl_every: 1,
            time_budget_secs: None,
            checkpoint_path: None,
            checkpoint_every_secs: None,
            warmstart_source: None,
            warmstart_checkpoint: None,
            warmstart_iter: 0,
            warmstart_same_band: false,
            warmstart_cross_turnup: false,
            accept_policy: None,
            resume_average_reset: false,
            dealer_filter: None,
            max_deals: None,
            jobs: 1,
            tree_cache: None,
            prebuilt_override: None,
            tremble: None,
            regret_pruning: None,
            strategy_output: None,
            skip_return_table: false,
            tree_rules: crate::game_tree::TreeRules::Current,
        }
    }
}

/// Run CFR building trees once, iterating until `target_expl` is reached
/// or `max_iters` is exhausted.
///
/// `on_iter` is called whenever exploitability is computed (every `config.expl_every`
/// iterations) with `(iter_num, exploitability, iter_wall_secs)`.
///
/// If `resume` is `Some((table, start_iter))`, the saved cumulative regrets and
/// strategy sums are copied into the freshly built table and iteration counting
/// continues from `start_iter`.
///
/// Returns the final strategy table and stats.
pub fn solve_until<F>(
    score: Score,
    tc: TurnupClass,
    config: &SolveConfig,
    match_values: &MatchValueTable,
    resume: Option<(StrategyTable, u64)>,
    mut on_iter: F,
) -> (StrategyTable, SolveStats)
where
    F: FnMut(u64, f64, f64), // (iter, exploitability, wall_secs_since_last_report)
{
    let start = Instant::now();

    let mut deals = enumerate_deals(&tc);
    if let Some(limit) = config.max_deals {
        assert!(
            config.tree_cache.is_none(),
            "--max-deals cannot be combined with --tree-cache"
        );
        subsample_deals(&mut deals, limit);
    }
    let num_deals = deals.len();

    let algo_name = match &config.algorithm {
        CfrAlgorithm::CfrPlus => "CFR+".to_string(),
        CfrAlgorithm::Dcfr { alpha, beta, gamma } => {
            format!("DCFR(α={}, β={}, γ={})", alpha, beta, gamma)
        }
        CfrAlgorithm::SyncCfrPlus => "SyncCFR+".to_string(),
        CfrAlgorithm::PcfrPlus => "PCFR+".to_string(),
    };
    info!(
        "solve_until: score ({},{}) tc={} | {} deals | max_iters={} target_expl={} | algo={} | expl_every={}",
        score.zero, score.one, tc.blocked_plain_level, num_deals,
        config.max_iters, config.target_expl, algo_name, config.expl_every,
    );

    let build_start = Instant::now();
    let prebuilt = config.prebuilt_override.clone().unwrap_or_else(|| {
        // Experimental non-default tree rules bypass the treepack cache, whose
        // path is keyed only on (score, tc, dealer) and would otherwise collide
        // a pruned tree with the standard one.
        if config.tree_rules != crate::game_tree::TreeRules::Current {
            return std::sync::Arc::new(
                crate::game_tree::build_all_trees_with_dealer_rules(
                    &score,
                    tc,
                    &deals,
                    config.dealer_filter,
                    config.tree_rules,
                )
                .expect("tree building failed"),
            );
        }
        load_or_build_trees(
            config.tree_cache.as_deref(),
            &score,
            tc,
            &deals,
            config.dealer_filter,
        )
    });
    assert_eq!(
        prebuilt.entries.len(),
        deals.len(),
        "prebuilt override must match the solve's deal set"
    );
    let build_secs = build_start.elapsed().as_secs_f64();

    let total_nodes: usize = prebuilt
        .entries
        .iter()
        .map(|e| e.tree_dealer_0.nodes.len() + e.tree_dealer_1.nodes.len())
        .sum();

    info!(
        "  Built {} trees ({} nodes) in {:.1}s",
        num_deals * 2,
        total_nodes,
        build_secs
    );

    // Only the resume/warm-start paths need the StrategyTable to overlay onto;
    // a fresh solve builds the SoA accumulators directly (skipping the 39.5M
    // InfoSetData construction that dominates peak RSS).
    assert!(
        config.warmstart_source.is_none() || config.warmstart_checkpoint.is_none(),
        "use only one warm-start source"
    );
    // In-memory seeds still use the compatibility table path. A checkpoint
    // seed can project straight into DenseAccum below, avoiding a complete
    // empty target StrategyTable at the warm-start peak.
    let has_table_seed = resume.is_some() || config.warmstart_source.is_some();
    let mut table = StrategyTable::new();
    if has_table_seed {
        for (_key, info_set, actions) in &prebuilt.info_sets {
            table.get_or_insert(info_set, actions);
        }
    }
    info!("  {} unique info sets", prebuilt.info_sets.len());

    // Resume: copy saved accumulators into the freshly built table and continue
    // iteration counting from where the checkpoint left off.
    let mut start_iter = 0u64;
    if let Some((rtable, start_t)) = resume {
        // `resume_average_reset` warm-starts the REGRETS only (so the current
        // strategy continues near where it was) but leaves the cumulative
        // strategy at the freshly-built zero and restarts iteration counting.
        // This avoids CFR+'s t^gamma average inertia carrying stale play across a
        // change in the game (e.g. a new frozen accept set in the asym solver),
        // where a high resumed iteration count makes new iterations too lightly
        // weighted to move the average.
        let restored = rtable
            .data
            .iter()
            .filter(|(key, source)| {
                table
                    .data
                    .get(key)
                    .is_some_and(|target| target.actions == source.actions)
            })
            .count();
        // A partial checkpoint (for example a deterministic --max-deals
        // benchmark) is a valid regret seed but not an exact resume: retaining
        // its iteration/average weight would starve the fresh rows. Degrade it
        // automatically to regrets-only, iteration-zero warm-start semantics.
        let reset_avg = config.resume_average_reset || restored != prebuilt.info_sets.len();
        for (key, data) in rtable.data {
            if let Some(target) = table.data.get_mut(&key) {
                if target.actions != data.actions {
                    continue;
                }
                target.cumulative_regret = data.cumulative_regret.clone();
                if !reset_avg {
                    target.cumulative_strategy = data.cumulative_strategy.clone();
                }
            }
        }
        start_iter = if reset_avg { 0 } else { start_t };
        info!(
            "  resumed from iteration {} ({} info sets){}",
            start_iter,
            restored,
            if reset_avg { ", average reset" } else { "" }
        );
    }

    // Warm-start either from an identical same-band tree (direct key/action
    // match; regrets only, average reset) or from a structurally-related mão
    // state (strip one leading AcceptEleven; historical behavior).
    if let Some(src) = &config.warmstart_source {
        assert!(
            !config.warmstart_cross_turnup,
            "profile-transfer warm-start requires a disk checkpoint with metadata"
        );
        let stats = apply_warmstart(
            &prebuilt.info_sets,
            &mut table,
            src,
            config.warmstart_same_band,
            None,
        );
        if config.warmstart_same_band {
            start_iter = 0;
            info!(
                "  same-band warm-start: transferred {} regret rows; average/iteration reset",
                stats.direct
            );
        } else {
            if config.warmstart_iter > start_iter {
                start_iter = config.warmstart_iter;
            }
            info!(
                "  related-state warm-start: transferred {} remapped card-play rows (continue from iter {})",
                stats.remapped, start_iter
            );
        }
    }
    let mut checkpoint_dense: Option<DenseAccum> = None;
    if let Some(path) = &config.warmstart_checkpoint {
        let mut streamed = false;
        if !has_table_seed {
            match CheckpointStream::open(path) {
                Ok(mut source) => {
                    let meta = source.meta().clone();
                    let (_, same_band) = checkpoint_warmstart_relation(
                        &meta,
                        &score,
                        tc,
                        config.dealer_filter,
                        config.warmstart_cross_turnup,
                    );
                    let dense = checkpoint_dense
                        .get_or_insert_with(|| DenseAccum::zeros(&prebuilt.info_sets));
                    let source_turnup = config.warmstart_cross_turnup.then_some(meta.turnup_class);
                    let stats = apply_warmstart_stream(
                        &prebuilt.info_sets,
                        dense,
                        &mut source,
                        tc,
                        same_band,
                        source_turnup,
                    )
                    .unwrap_or_else(|error| {
                        panic!("failed to stream warm-start {}: {error}", path.display())
                    });
                    start_iter = report_checkpoint_warmstart(
                        stats,
                        &meta,
                        &score,
                        tc,
                        config.warmstart_cross_turnup,
                        same_band,
                        prebuilt.info_sets.len(),
                        start_iter,
                    );
                    info!(
                        "  streamed {} checkpoint rows directly into dense accumulators",
                        meta.num_info_sets
                    );
                    streamed = true;
                }
                Err(error) => {
                    info!(
                        "  checkpoint stream unavailable for {} ({}); falling back to compatibility loader",
                        path.display(), error
                    );
                }
            }
        }

        if !streamed {
            match crate::storage::load_checkpoint(path) {
                Ok((src, meta)) => {
                    let (_, same_band) = checkpoint_warmstart_relation(
                        &meta,
                        &score,
                        tc,
                        config.dealer_filter,
                        config.warmstart_cross_turnup,
                    );
                    let source_turnup = config.warmstart_cross_turnup.then_some(meta.turnup_class);
                    let stats = if has_table_seed {
                        apply_warmstart(
                            &prebuilt.info_sets,
                            &mut table,
                            &src,
                            same_band,
                            source_turnup,
                        )
                    } else {
                        let dense = checkpoint_dense
                            .get_or_insert_with(|| DenseAccum::zeros(&prebuilt.info_sets));
                        apply_warmstart_dense(
                            &prebuilt.info_sets,
                            dense,
                            &src,
                            same_band,
                            source_turnup,
                        )
                    };
                    start_iter = report_checkpoint_warmstart(
                        stats,
                        &meta,
                        &score,
                        tc,
                        config.warmstart_cross_turnup,
                        same_band,
                        prebuilt.info_sets.len(),
                        start_iter,
                    );
                }
                Err(error) => {
                    panic!("failed to load warm-start {}: {error}", path.display());
                }
            }
        }
    }

    // Wall-clock deadline (covers tree build + iterations, by design).
    let deadline = config
        .time_budget_secs
        .map(|budget| start + std::time::Duration::from_secs_f64(budget));
    let checkpoint_interval = config
        .checkpoint_every_secs
        .map(std::time::Duration::from_secs_f64);

    // Within-iteration progress bar — hidden if stderr is not a tty
    // (e.g. when the output is being piped) so control codes don't corrupt logs.
    let is_tty = std::io::stderr().is_terminal();
    let num_entries = prebuilt.entries.len() as u64;
    let pb = if is_tty {
        let bar = ProgressBar::new(num_entries);
        bar.set_style(
            ProgressStyle::with_template(
                "  iter {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})",
            )
            .expect("valid progress bar template")
            .progress_chars("█▓▒░  "),
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    let iter_start = Instant::now();
    let mut exploitability_history: Vec<(u64, f64)> = Vec::new();
    let mut t = start_iter;
    let mut last_report_time = Instant::now();
    let mut last_checkpoint_time = Instant::now();
    let sweep_mode = SweepMode::of(&config.algorithm);
    let regret_pruning = config
        .regret_pruning
        .map(|pruning| pruning.after_start_iter(start_iter));
    if let Some(pruning) = regret_pruning {
        assert!(
            matches!(config.algorithm, CfrAlgorithm::SyncCfrPlus),
            "regret pruning currently requires SyncCFR+"
        );
        assert!(
            pruning.threshold > 0.0,
            "regret-pruning threshold must be > 0"
        );
        assert!(
            pruning.revisit_every_rounds > 0,
            "regret-pruning revisit cadence must be > 0"
        );
        assert!(
            config.tremble.is_none(),
            "regret pruning and trembling are intentionally incompatible"
        );
    }
    // Hot-loop representation: accumulators in dense table_idx order (the
    // table is drained here and rebuilt before returning / for checkpoints).
    let mut dense = if let Some(dense) = checkpoint_dense {
        dense
    } else if has_table_seed {
        DenseAccum::from_table(&prebuilt.info_sets, &mut table)
    } else {
        DenseAccum::zeros(&prebuilt.info_sets)
    };
    // Free the (possibly seeded) table's info-set HashMap during the hot solve;
    // `into_table` rebuilds it at the end.
    table = StrategyTable::new();
    if sweep_mode.buffered() {
        dense.ensure_buffers(sweep_mode == SweepMode::Predictive);
    }
    if regret_pruning.is_some() {
        dense.enable_regret_pruning();
    }
    let parallel = parallel_setup(config.jobs, sweep_mode, &prebuilt.info_sets);
    let mut par_bufs: Vec<LocalAccum> = parallel
        .as_ref()
        .map(|p| (0..p.jobs).map(|_| LocalAccum::zeros(p.total)).collect())
        .unwrap_or_default();
    let checkpoint_meta = |t: u64| CheckpointMeta {
        score: (score.zero, score.one),
        turnup_class: tc,
        algo: algo_name.clone(),
        iteration: t,
        num_info_sets: prebuilt.info_sets.len(),
        dealer_filter: config.dealer_filter,
    };

    loop {
        t += 1;
        let traversing = (t % 2) as Player;
        let tremble_eps = config
            .tremble
            .map(|ts| ts.eps_at(t, start_iter, config.max_iters))
            .unwrap_or(0.0);

        pb.set_message(format!("{}", t));
        pb.set_position(0);
        pb.reset_elapsed();

        let accept_policy = config.accept_policy.as_deref();
        if let Some(plan) = &parallel {
            plan.pool.install(|| {
                parallel_sync_sweep(
                    &prebuilt,
                    &mut dense,
                    plan,
                    &mut par_bufs,
                    traversing,
                    t,
                    &score,
                    match_values,
                    accept_policy,
                    tremble_eps,
                    regret_pruning.as_ref(),
                )
            });
        } else {
            for entry in &prebuilt.entries {
                for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
                    if tree.nodes.is_empty() {
                        continue; // dealer excluded by config.dealer_filter
                    }
                    cfr_traverse_tree(
                        tree,
                        0,
                        traversing,
                        dealer,
                        [1.0, 1.0],
                        entry.weight,
                        t,
                        &score,
                        match_values,
                        accept_policy,
                        sweep_mode,
                        &prebuilt.info_sets,
                        &mut dense,
                        tremble_eps,
                        regret_pruning.as_ref(),
                        None,
                    );
                }
                pb.inc(1);
            }

            if sweep_mode.buffered() {
                fold_pending_regrets(&mut dense, sweep_mode, &prebuilt.info_sets, traversing);
            }
        }

        // Apply DCFR discounting after each iteration
        if let CfrAlgorithm::Dcfr { alpha, beta, gamma } = &config.algorithm {
            let t_f = t as f64;
            // Discount factor for positive regrets: t^α / (t^α + 1)
            let pos_discount = t_f.powf(*alpha) / (t_f.powf(*alpha) + 1.0);
            // Discount factor for negative regrets: t^β / (t^β + 1)
            let neg_discount = t_f.powf(*beta) / (t_f.powf(*beta) + 1.0);
            // Discount factor for average strategy: (t/(t+1))^γ
            let strat_discount = (t_f / (t_f + 1.0)).powf(*gamma);

            for r in dense.regret.iter_mut() {
                if *r > 0.0 {
                    *r *= pos_discount as Acc;
                } else {
                    *r *= neg_discount as Acc;
                }
            }
            for st in dense.strategy.iter_mut() {
                *st *= strat_discount as Acc;
            }
        }

        // Compute exploitability periodically
        let should_compute_expl = t % config.expl_every == 0 || t == 1 || t >= config.max_iters;

        if should_compute_expl {
            let expl = compute_exploitability_dense(
                &prebuilt,
                &dense,
                &score,
                match_values,
                accept_policy,
            );
            let secs_since_report = last_report_time.elapsed().as_secs_f64();
            last_report_time = Instant::now();
            exploitability_history.push((t, expl));

            let pruning_suffix = regret_pruning
                .map(|pruning| {
                    format!(
                        " | prune_candidates={}/{}",
                        dense.regret_pruning_candidates(pruning.threshold),
                        dense.regret.len()
                    )
                })
                .unwrap_or_default();
            let msg = format!(
                "  iter {:>4}: expl = {:.6}  ({:.1}s){}",
                t, expl, secs_since_report, pruning_suffix
            );
            if is_tty {
                pb.println(&msg);
            } else {
                eprintln!("{}", msg);
            }

            on_iter(t, expl, secs_since_report);

            if expl <= config.target_expl || t >= config.max_iters {
                break;
            }
        } else if t >= config.max_iters {
            // Hit max iters without an exploitability check — compute one final time
            let expl = compute_exploitability_dense(
                &prebuilt,
                &dense,
                &score,
                match_values,
                accept_policy,
            );
            let secs_since_report = last_report_time.elapsed().as_secs_f64();
            exploitability_history.push((t, expl));
            on_iter(t, expl, secs_since_report);
            break;
        }

        // Periodic full-state checkpoint (written directly from the dense
        // accumulators — no table rebuild).
        if let (Some(path), Some(interval)) = (&config.checkpoint_path, checkpoint_interval) {
            if last_checkpoint_time.elapsed() >= interval {
                match save_checkpoint_iter(
                    path,
                    checkpoint_meta(t),
                    dense.checkpoint_iter(&prebuilt.info_sets),
                ) {
                    Ok(()) => info!("  checkpoint saved at iter {} -> {}", t, path.display()),
                    Err(e) => log::warn!("  checkpoint save failed at iter {}: {}", t, e),
                }
                last_checkpoint_time = Instant::now();
            }
        }

        // Wall-clock budget: stop once the deadline is reached.
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                info!(
                    "  time budget reached after {:.1}s at iter {} — stopping",
                    start.elapsed().as_secs_f64(),
                    t
                );
                break;
            }
        }
    }

    // Final full-state checkpoint on any break (eps / max_iters / time budget).
    if let Some(path) = &config.checkpoint_path {
        match save_checkpoint_iter(
            path,
            checkpoint_meta(t),
            dense.checkpoint_iter(&prebuilt.info_sets),
        ) {
            Ok(()) => info!(
                "  final checkpoint saved at iter {} -> {}",
                t,
                path.display()
            ),
            Err(e) => log::warn!("  final checkpoint save failed at iter {}: {}", t, e),
        }
    }

    pb.finish_and_clear();

    let total_secs = start.elapsed().as_secs_f64();
    let iter_secs_total = iter_start.elapsed().as_secs_f64();
    let final_expl = exploitability_history.last().map(|h| h.1);
    let (v_d0, v_d1) = compute_game_value_per_dealer_dense(
        &prebuilt,
        &dense,
        &score,
        match_values,
        config.accept_policy.as_deref(),
    );
    let game_value_p0 = Some((v_d0 + v_d1) / 2.0);
    let game_value_per_dealer = Some((v_d0, v_d1));

    // Direct dense artifact write: the average strategy leaves the process
    // before (or instead of) the hash-table rebuild, so the dense+table
    // overlap never becomes the memory high-water mark.
    if let Some(path) = &config.strategy_output {
        let strategy_meta = crate::storage::SolvedStateMeta {
            score: (score.zero, score.one),
            turnup_class: tc,
            iterations: t,
            num_info_sets: prebuilt.info_sets.len(),
        };
        match crate::storage::save_strategy_rows(
            path,
            strategy_meta,
            dense.strategy_rows(&prebuilt.info_sets),
        ) {
            Ok(()) => info!("  strategy saved from dense -> {}", path.display()),
            Err(e) => log::warn!("  dense strategy save failed: {}", e),
        }
    }

    let num_info_sets = prebuilt.info_sets.len();
    if config.skip_return_table {
        drop(dense);
    } else {
        dense.into_table(&prebuilt.info_sets, &mut table);
    }
    let mem = estimate_memory_bytes(&table, &prebuilt);

    let stats = SolveStats {
        score: (score.zero, score.one),
        turnup_class: tc.blocked_plain_level,
        iterations: t,
        num_deals,
        num_info_sets,
        total_nodes,
        total_duration_secs: total_secs,
        build_tree_secs: build_secs,
        per_iteration_secs: if t > 0 {
            iter_secs_total / t as f64
        } else {
            0.0
        },
        estimated_memory_bytes: mem,
        exploitability: final_expl,
        exploitability_history,
        game_value_p0,
        game_value_per_dealer,
    };

    info!("{}", stats);
    (table, stats)
}

/// How the traversal reads strategies and writes regrets.
#[derive(Clone, Copy, PartialEq)]
enum SweepMode {
    /// Historical behavior: strategy recomputed from live regrets at every
    /// visit; regret updates applied (and clamped) in place mid-sweep.
    Async,
    /// Strategy from cumulative regret, frozen for the sweep; instantaneous
    /// regrets buffered in `pending_regret` and folded at iteration end.
    Sync,
    /// Like `Sync`, but the strategy adds the last iteration's instantaneous
    /// regret as a prediction (optimistic regret matching / PCFR+).
    Predictive,
}

impl SweepMode {
    fn of(algo: &CfrAlgorithm) -> Self {
        match algo {
            CfrAlgorithm::CfrPlus | CfrAlgorithm::Dcfr { .. } => SweepMode::Async,
            CfrAlgorithm::SyncCfrPlus => SweepMode::Sync,
            CfrAlgorithm::PcfrPlus => SweepMode::Predictive,
        }
    }

    fn buffered(self) -> bool {
        !matches!(self, SweepMode::Async)
    }
}

/// Build the rayon pool + atomic accumulators when a parallel sweep is
/// requested and sound. Parallelism requires the SYNC sweep (frozen strategy;
/// traversal read-only on accumulators): Async reads its own mid-sweep writes
/// and Predictive's last-regret bookkeeping isn't threaded yet — both fall
/// back to the serial path with a warning.
fn parallel_setup(jobs: usize, mode: SweepMode, infos: &InfoMeta) -> Option<ParallelPlan> {
    if jobs <= 1 {
        return None;
    }
    if mode != SweepMode::Sync {
        log::warn!("--jobs {jobs} requires --algo sync; running single-threaded");
        return None;
    }
    // Prefix offsets: slot base per table_idx, one f64 slot per (info set, action).
    let mut off = Vec::with_capacity(infos.len() + 1);
    let mut total = 0u32;
    for (_, _, actions) in infos {
        off.push(total);
        total += actions.len() as u32;
    }
    off.push(total);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("rayon pool");
    Some(ParallelPlan {
        pool,
        jobs,
        off,
        total: total as usize,
    })
}

/// Plan for PARALLEL synchronous sweeps. The accumulator is THREAD-LOCAL, not
/// shared: the previous shared-atomic version false-shared cache lines (8
/// AtomicU64 per 64-byte line, so threads writing *different* info sets still
/// ping-ponged the line) and scaled only ~1.2x to 8 cores. Each worker folds a
/// contiguous chunk of deals into its own flat buffer with plain adds, then the
/// buffers are reduced and applied to the dense table once.
///
/// One buffer suffices per iteration: an info set is either the traversing
/// player's (accumulates regret) or the opponent's (accumulates average-
/// strategy weight) — never both in the same iteration — so the write meaning
/// is recovered at apply time from the info set's player vs `traversing`.
struct ParallelPlan {
    pool: rayon::ThreadPool,
    jobs: usize,
    off: Vec<u32>,
    total: usize,
}

/// Thread-local flat accumulator (`total` f64 slots, indexed `off[idx] + a`).
struct LocalAccum {
    buf: Vec<f64>,
}

impl LocalAccum {
    fn zeros(total: usize) -> Self {
        Self {
            buf: vec![0.0; total],
        }
    }
    fn reset(&mut self) {
        // Reuse the allocation across iterations: memset an already-resident
        // buffer, so no page faults after the first sweep (the previous
        // allocate-per-sweep version spent 95% of CPU in the kernel faulting
        // ~40 GB of fresh buffers every iteration).
        self.buf.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Parallel-sweep traversal: identical control flow to [`cfr_traverse_tree`]
/// in `SweepMode::Sync`, but writes go to a thread-local [`LocalAccum`] (plain
/// adds, no atomics) so deals traverse concurrently against the frozen `dense`.
#[allow(clippy::too_many_arguments)]
fn cfr_traverse_tree_par(
    tree: &GameTree,
    node_id: NodeId,
    traversing: Player,
    dealer: Player,
    reach_probs: [f64; 2],
    chance_weight: f64,
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    infos: &InfoMeta,
    dense: &DenseAccum,
    off: &[u32],
    local: &mut LocalAccum,
    tremble_eps: f64,
    regret_pruning: Option<&RegretPruningConfig>,
) -> f64 {
    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            let p0_value = terminal_p0_value(payoff_p0, dealer, score, match_values);
            if traversing == 0 {
                p0_value
            } else {
                -p0_value
            }
        }
        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            let idx = table_idx as usize;
            let num_actions = edges.len();

            if let Some(policy) = accept_policy {
                if let Some((accept_child, fold_child)) = eleven_decision_children(edges) {
                    let hand = &infos[idx].1.starting_hand;
                    let chosen = if policy.accepts(dealer, hand) {
                        accept_child
                    } else {
                        fold_child
                    };
                    return cfr_traverse_tree_par(
                        tree,
                        chosen,
                        traversing,
                        dealer,
                        reach_probs,
                        chance_weight,
                        iteration,
                        score,
                        match_values,
                        accept_policy,
                        infos,
                        dense,
                        off,
                        local,
                        tremble_eps,
                        regret_pruning,
                    );
                }
            }

            // Frozen strategy: cumulative_regret is read-only during the sweep.
            // Tremble floors it (see TrembleSchedule) before it is used for
            // reach propagation, regret, or average-strategy accumulation.
            let strategy = tremble_strategy(dense.current_strategy(idx), tremble_eps);

            let mut action_values: crate::strategy::ActionProbs =
                smallvec::smallvec![0.0; num_actions];
            let mut node_value = 0.0;
            let is_owner = player == traversing;
            let mut pruned_actions = 0u16;
            for (i, e) in edges.iter().enumerate() {
                // Exact zero-prob pruning — see cfr_traverse_tree. A no-op
                // while trembling (strategy[i] is never exactly 0 then).
                if !is_owner && strategy[i] == 0.0 {
                    continue;
                }
                if is_owner
                    && dense.should_prune_action(idx, i, strategy[i], iteration, regret_pruning)
                {
                    pruned_actions |= 1 << i;
                    continue;
                }
                let mut new_reach = reach_probs;
                new_reach[player as usize] *= strategy[i];
                let value = cfr_traverse_tree_par(
                    tree,
                    e.child,
                    traversing,
                    dealer,
                    new_reach,
                    chance_weight,
                    iteration,
                    score,
                    match_values,
                    accept_policy,
                    infos,
                    dense,
                    off,
                    local,
                    tremble_eps,
                    regret_pruning,
                );
                action_values[i] = value;
                node_value += strategy[i] * value;
            }

            let base = off[idx] as usize;
            if player == traversing {
                let opponent = 1 - player;
                for i in 0..num_actions {
                    if pruned_actions & (1 << i) != 0 {
                        continue;
                    }
                    let regret = action_values[i] - node_value;
                    local.buf[base + i] += reach_probs[opponent as usize] * chance_weight * regret;
                }
            } else {
                let w = (iteration as f64) * reach_probs[player as usize] * chance_weight;
                if w != 0.0 {
                    for i in 0..num_actions {
                        local.buf[base + i] += w * strategy[i];
                    }
                }
            }

            node_value
        }
    }
}

/// One full parallel synchronous sweep: each worker folds a contiguous slice
/// of the deals into its OWN persistent buffer (`bufs`, allocated once and
/// reused every iteration), then the buffers are merged into the dense table
/// (RM+ clamp for regret slots).
#[allow(clippy::too_many_arguments)]
fn parallel_sync_sweep(
    prebuilt: &PrebuiltTrees,
    dense: &mut DenseAccum,
    plan: &ParallelPlan,
    bufs: &mut [LocalAccum],
    traversing: Player,
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    tremble_eps: f64,
    regret_pruning: Option<&RegretPruningConfig>,
) {
    use rayon::prelude::*;
    let dense_ref: &DenseAccum = dense;
    let infos = &prebuilt.info_sets;
    let off = &plan.off;
    let entries = &prebuilt.entries;
    let chunk_len = entries.len().div_ceil(plan.jobs).max(1);

    // par_iter_mut over the FIXED buffer set: worker w owns bufs[w] exclusively
    // and processes chunk w. No per-iteration allocation, no shared writes.
    bufs.par_iter_mut().enumerate().for_each(|(w, local)| {
        local.reset();
        let start = w * chunk_len;
        if start >= entries.len() {
            return;
        }
        let end = (start + chunk_len).min(entries.len());
        for entry in &entries[start..end] {
            for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
                if tree.is_empty() {
                    continue;
                }
                cfr_traverse_tree_par(
                    tree,
                    0,
                    traversing,
                    dealer,
                    [1.0, 1.0],
                    entry.weight,
                    iteration,
                    score,
                    match_values,
                    accept_policy,
                    infos,
                    dense_ref,
                    off,
                    local,
                    tremble_eps,
                    regret_pruning,
                );
            }
        }
    });

    // Apply: an info set that acted as the traversing player this iteration gets
    // a clamped regret update; otherwise an average-strategy add. Worker buffers
    // and the dense table share the same global slot indexing (both built from
    // `off`), so sum per slot.
    for idx in 0..infos.len() {
        let base = off[idx] as usize;
        let end = off[idx + 1] as usize;
        let is_traversing = infos[idx].1.player == traversing;
        for slot in base..end {
            let mut v = 0.0;
            for b in bufs.iter() {
                v += b.buf[slot];
            }
            if v == 0.0 {
                continue;
            }
            if is_traversing {
                if !dense.prune_regret.is_empty() {
                    dense.prune_regret[slot] += v as f32;
                }
                dense.regret[slot] = ((dense.regret[slot] as f64 + v).max(0.0)) as Acc;
            } else {
                dense.strategy[slot] = (dense.strategy[slot] as f64 + v) as Acc;
            }
        }
    }
}

/// Info-set metadata slice type, in `table_idx` order (= `PrebuiltTrees::info_sets`).
type InfoMeta = [(InfoSetKey, InfoSet, crate::game_tree::InfoActions)];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WarmStartTransferStats {
    direct: usize,
    remapped: usize,
}

fn checkpoint_warmstart_relation(
    meta: &CheckpointMeta,
    score: &Score,
    tc: TurnupClass,
    dealer_filter: Option<Player>,
    cross_turnup: bool,
) -> (Score, bool) {
    assert_eq!(
        meta.dealer_filter, dealer_filter,
        "warm-start dealer filter mismatch"
    );
    let source_score = Score {
        zero: meta.score.0,
        one: meta.score.1,
    };
    let same_band = crate::game_tree::band_signature(&source_score, meta.dealer_filter)
        == crate::game_tree::band_signature(score, dealer_filter);
    if cross_turnup {
        assert!(
            same_band,
            "profile-transfer warm-start requires the same tree band"
        );
        assert!(
            source_score != *score || meta.turnup_class != tc,
            "profile-transfer source must differ by score or turn-up class"
        );
    } else {
        assert_eq!(meta.turnup_class, tc, "warm-start turnup class mismatch");
    }
    (source_score, same_band)
}

fn report_checkpoint_warmstart(
    stats: WarmStartTransferStats,
    meta: &CheckpointMeta,
    score: &Score,
    tc: TurnupClass,
    cross_turnup: bool,
    same_band: bool,
    target_rows: usize,
    start_iter: u64,
) -> u64 {
    if cross_turnup {
        println!(
            "  disk profile transfer {}x{}/tc{} -> {}x{}/tc{}: transferred {} of {} rows ({:.2}%); average/iteration preserved at {}",
            meta.score.0,
            meta.score.1,
            meta.turnup_class.blocked_plain_level,
            score.zero,
            score.one,
            tc.blocked_plain_level,
            stats.direct,
            target_rows,
            100.0 * stats.direct as f64 / target_rows.max(1) as f64,
            meta.iteration,
        );
        meta.iteration
    } else if same_band {
        info!(
            "  disk same-band warm-start: transferred {} regret rows; average/iteration reset",
            stats.direct
        );
        0
    } else {
        let iteration = meta.iteration.max(start_iter);
        info!(
            "  disk related-state warm-start: transferred {} remapped rows (continue from iter {})",
            stats.remapped, iteration
        );
        iteration
    }
}

fn warmstart_source_data<'a>(
    key: &InfoSetKey,
    info: &InfoSet,
    source: &'a StrategyTable,
    same_band: bool,
    source_turnup: Option<TurnupClass>,
) -> Option<&'a InfoSetData> {
    if let Some(source_turnup) = source_turnup {
        let mut candidate = info.clone();
        candidate.turnup_class = source_turnup;
        source.data.get(&candidate.key())
    } else if same_band {
        source.data.get(key)
    } else {
        let acts = info.history.actions();
        if acts.first() != Some(&AbstractAction::AcceptEleven) {
            return None;
        }
        let mut stripped = ActionHistory::new();
        for action in &acts[1..] {
            stripped.push(*action);
        }
        let candidate = InfoSet {
            player: info.player,
            is_dealer: info.is_dealer,
            turnup_class: info.turnup_class,
            starting_hand: info.starting_hand.clone(),
            history: stripped,
        };
        source.data.get(&candidate.key())
    }
}

fn apply_warmstart(
    infos: &InfoMeta,
    target: &mut StrategyTable,
    source: &StrategyTable,
    same_band: bool,
    source_turnup: Option<TurnupClass>,
) -> WarmStartTransferStats {
    let mut stats = WarmStartTransferStats::default();
    for (key, info, actions) in infos {
        let source_data = warmstart_source_data(key, info, source, same_band, source_turnup);
        let Some(source_data) = source_data else {
            continue;
        };
        let Some(destination) = target.data.get_mut(key) else {
            continue;
        };
        if same_band {
            // A restricted arena may retain only a subset of the structurally
            // identical source row. Transfer each retained action by identity;
            // never reinterpret regret by position after actions were removed.
            let source_slots: Option<Vec<usize>> = actions
                .iter()
                .map(|action| {
                    source_data
                        .actions
                        .iter()
                        .position(|candidate| candidate == action)
                })
                .collect();
            let Some(source_slots) = source_slots else {
                continue;
            };
            for (destination_slot, &source_slot) in source_slots.iter().enumerate() {
                destination.cumulative_regret[destination_slot] =
                    source_data.cumulative_regret[source_slot];
            }
            if source_turnup.is_some() {
                // The donor's average is the valuable part of this experiment:
                // it contains the off-path behavior learned under trembling.
                // The target game differs only in chance multiplicity, so keep
                // that average as a prior and let subsequent target iterations
                // reweight it. Ordinary same-band score transfer still resets.
                for (destination_slot, source_slot) in source_slots.into_iter().enumerate() {
                    destination.cumulative_strategy[destination_slot] =
                        source_data.cumulative_strategy[source_slot];
                }
            }
            stats.direct += 1;
        } else {
            if source_data.actions.as_slice() != &actions[..] {
                continue;
            }
            destination
                .cumulative_regret
                .clone_from(&source_data.cumulative_regret);
            destination
                .cumulative_strategy
                .clone_from(&source_data.cumulative_strategy);
            stats.remapped += 1;
        }
    }
    stats
}

/// Project a loaded checkpoint directly into table-indexed accumulators.
/// This is semantically identical to `apply_warmstart` followed by
/// `DenseAccum::from_table`, but it never constructs the full target hash table.
fn apply_warmstart_dense(
    infos: &InfoMeta,
    target: &mut DenseAccum,
    source: &StrategyTable,
    same_band: bool,
    source_turnup: Option<TurnupClass>,
) -> WarmStartTransferStats {
    let mut stats = WarmStartTransferStats::default();
    for (idx, (key, info, actions)) in infos.iter().enumerate() {
        let Some(source_data) = warmstart_source_data(key, info, source, same_band, source_turnup)
        else {
            continue;
        };
        let base = target.off[idx] as usize;
        if same_band {
            let source_slots: Option<Vec<usize>> = actions
                .iter()
                .map(|action| {
                    source_data
                        .actions
                        .iter()
                        .position(|candidate| candidate == action)
                })
                .collect();
            let Some(source_slots) = source_slots else {
                continue;
            };
            for (destination_slot, &source_slot) in source_slots.iter().enumerate() {
                target.regret[base + destination_slot] =
                    source_data.cumulative_regret[source_slot] as Acc;
            }
            if source_turnup.is_some() {
                for (destination_slot, source_slot) in source_slots.into_iter().enumerate() {
                    target.strategy[base + destination_slot] =
                        source_data.cumulative_strategy[source_slot] as Acc;
                }
            }
            stats.direct += 1;
        } else {
            if source_data.actions.as_slice() != &actions[..] {
                continue;
            }
            let end = base + actions.len();
            for (dst, src) in target.regret[base..end]
                .iter_mut()
                .zip(&source_data.cumulative_regret)
            {
                *dst = *src as Acc;
            }
            for (dst, src) in target.strategy[base..end]
                .iter_mut()
                .zip(&source_data.cumulative_strategy)
            {
                *dst = *src as Acc;
            }
            stats.remapped += 1;
        }
    }
    stats
}

/// Stream one positioned checkpoint directly into target-indexed dense state.
/// The temporary key->index map is much smaller than a full source
/// `StrategyTable`, and each serialized source row is dropped immediately.
fn apply_warmstart_stream(
    infos: &InfoMeta,
    target: &mut DenseAccum,
    source: &mut CheckpointStream,
    target_turnup: TurnupClass,
    same_band: bool,
    source_turnup: Option<TurnupClass>,
) -> Result<WarmStartTransferStats, StorageError> {
    let index: ahash::AHashMap<InfoSetKey, u32> = infos
        .iter()
        .enumerate()
        .map(|(idx, (key, _, _))| (*key, idx as u32))
        .collect();
    let mut stats = WarmStartTransferStats::default();

    while let Some(mut entry) = source.next_entry()? {
        if source_turnup.is_some() {
            entry.info_set.turnup_class = target_turnup;
        } else if !same_band {
            let source_actions = entry.info_set.history.actions();
            let mut prefixed = ActionHistory::new();
            prefixed.push(AbstractAction::AcceptEleven);
            for action in source_actions {
                prefixed.push(*action);
            }
            entry.info_set.history = prefixed;
        }

        let Some(&idx) = index.get(&entry.info_set.key()) else {
            continue;
        };
        let idx = idx as usize;
        let actions = &infos[idx].2;
        let base = target.off[idx] as usize;
        if same_band {
            let source_slots: Option<Vec<usize>> = actions
                .iter()
                .map(|action| {
                    entry
                        .actions
                        .iter()
                        .position(|candidate| candidate == action)
                })
                .collect();
            let Some(source_slots) = source_slots else {
                continue;
            };
            for (destination_slot, &source_slot) in source_slots.iter().enumerate() {
                target.regret[base + destination_slot] =
                    entry.cumulative_regret[source_slot] as Acc;
            }
            if source_turnup.is_some() {
                for (destination_slot, source_slot) in source_slots.into_iter().enumerate() {
                    target.strategy[base + destination_slot] =
                        entry.cumulative_strategy[source_slot] as Acc;
                }
            }
            stats.direct += 1;
        } else {
            if entry.actions.as_slice() != &actions[..] {
                continue;
            }
            let end = base + actions.len();
            for (dst, src) in target.regret[base..end]
                .iter_mut()
                .zip(&entry.cumulative_regret)
            {
                *dst = *src as Acc;
            }
            for (dst, src) in target.strategy[base..end]
                .iter_mut()
                .zip(&entry.cumulative_strategy)
            {
                *dst = *src as Acc;
            }
            stats.remapped += 1;
        }
    }

    Ok(stats)
}

/// SoA accumulator table for the hot solve loop. Replaces a
/// `Vec<InfoSetData>` (39.5M little heap vectors, pointer-chased on every node
/// visit) with a few contiguous `Vec<f64>` indexed by `off[idx] + action`.
/// Cuts ~5x the per-entry `Vec` header overhead and turns each node's strategy
/// read into a contiguous slice. `StrategyTable`/`InfoSetData` remain the
/// storage/API representation; this is a transient built at solve start and
/// converted back at the end.
pub(crate) struct DenseAccum {
    /// Prefix slot offsets, `len == n_info_sets + 1`.
    off: Vec<u32>,
    /// Cumulative regret (CFR+: clamped >= 0), one slot per (info set, action).
    regret: Vec<Acc>,
    /// Cumulative (iteration-weighted) strategy sum.
    strategy: Vec<Acc>,
    /// This iteration's buffered instantaneous regret (buffered sweep modes;
    /// empty otherwise). Alternating CFR writes regrets for only one player
    /// per sweep, so player 0 and player 1 reuse this same compact address
    /// space instead of reserving one slot for every action in the table.
    /// Stays `f64` even under `accum-f32`: it is transient (per-sweep), and
    /// keeping within-sweep accumulation wide means the cumulative arrays see
    /// exactly one narrowing per iteration — mirroring the parallel path's
    /// f64 `LocalAccum`, and avoiding f32 absorption across a 140k-deal sweep.
    pending: Vec<f64>,
    /// Per-info-set base in the owning player's overlaid `pending` space.
    pending_off: Vec<u32>,
    /// Required pending slots for each player; the allocation is their max.
    pending_slots: [usize; 2],
    /// Previous iteration's instantaneous regret (PCFR+ only; empty otherwise).
    /// `f64` like `pending` (transient, only allocated in predictive mode).
    last: Vec<f64>,
    /// Optional unclamped regret shadow used only by reversible regret pruning.
    /// `f32` is sufficient for the pruning classifier and halves its opt-in
    /// memory cost versus another full regret accumulator.
    prune_regret: Vec<f32>,
}

impl DenseAccum {
    fn offsets(infos: &InfoMeta) -> (Vec<u32>, usize) {
        let mut off = Vec::with_capacity(infos.len() + 1);
        let mut total = 0u32;
        for (_, _, actions) in infos {
            off.push(total);
            total += actions.len() as u32;
        }
        off.push(total);
        (off, total as usize)
    }

    fn pending_layout(infos: &InfoMeta) -> (Vec<u32>, [usize; 2]) {
        let mut off = Vec::with_capacity(infos.len());
        let mut totals = [0u32; 2];
        for (_, info, actions) in infos {
            let player = info.player as usize;
            off.push(totals[player]);
            totals[player] += actions.len() as u32;
        }
        (off, [totals[0] as usize, totals[1] as usize])
    }

    /// Zero accumulators sized directly from the info-set offsets — no
    /// `StrategyTable` intermediary (the fresh-solve fast path; building then
    /// draining 39.5M `InfoSetData` is the memory peak otherwise).
    fn zeros(infos: &InfoMeta) -> Self {
        let (off, total) = Self::offsets(infos);
        let (pending_off, pending_slots) = Self::pending_layout(infos);
        Self {
            off,
            regret: vec![0.0; total],
            strategy: vec![0.0; total],
            pending: Vec::new(),
            pending_off,
            pending_slots,
            last: Vec::new(),
            prune_regret: Vec::new(),
        }
    }

    /// Drain accumulators out of `table` into SoA form (missing entries = 0).
    fn from_table(infos: &InfoMeta, table: &mut StrategyTable) -> Self {
        let (off, total) = Self::offsets(infos);
        let (pending_off, pending_slots) = Self::pending_layout(infos);
        let mut regret = vec![0.0; total];
        let mut strategy = vec![0.0; total];
        for (idx, (key, _, _)) in infos.iter().enumerate() {
            if let Some(data) = table.data.remove(key) {
                let base = off[idx] as usize;
                for (dst, src) in regret[base..base + data.cumulative_regret.len()]
                    .iter_mut()
                    .zip(&data.cumulative_regret)
                {
                    *dst = *src as Acc;
                }
                for (dst, src) in strategy[base..base + data.cumulative_strategy.len()]
                    .iter_mut()
                    .zip(&data.cumulative_strategy)
                {
                    *dst = *src as Acc;
                }
            }
        }
        Self {
            off,
            regret,
            strategy,
            pending: Vec::new(),
            pending_off,
            pending_slots,
            last: Vec::new(),
            prune_regret: Vec::new(),
        }
    }

    /// Non-draining clone (one-shot evaluation of loaded tables).
    fn clone_from_table(infos: &InfoMeta, table: &StrategyTable) -> Self {
        let (off, total) = Self::offsets(infos);
        let (pending_off, pending_slots) = Self::pending_layout(infos);
        let mut regret = vec![0.0; total];
        let mut strategy = vec![0.0; total];
        for (idx, (key, _, _)) in infos.iter().enumerate() {
            if let Some(data) = table.data.get(key) {
                let base = off[idx] as usize;
                for (dst, src) in regret[base..base + data.cumulative_regret.len()]
                    .iter_mut()
                    .zip(&data.cumulative_regret)
                {
                    *dst = *src as Acc;
                }
                for (dst, src) in strategy[base..base + data.cumulative_strategy.len()]
                    .iter_mut()
                    .zip(&data.cumulative_strategy)
                {
                    *dst = *src as Acc;
                }
            }
        }
        Self {
            off,
            regret,
            strategy,
            pending: Vec::new(),
            pending_off,
            pending_slots,
            last: Vec::new(),
            prune_regret: Vec::new(),
        }
    }

    /// Rebuild the `StrategyTable` (average strategy + regrets) for return/save.
    fn into_table(self, infos: &InfoMeta, table: &mut StrategyTable) {
        for (idx, (key, info, actions)) in infos.iter().enumerate() {
            let base = self.off[idx] as usize;
            let end = self.off[idx + 1] as usize;
            let data = InfoSetData {
                cumulative_regret: self.regret[base..end].iter().map(|x| *x as f64).collect(),
                cumulative_strategy: self.strategy[base..end].iter().map(|x| *x as f64).collect(),
                pending_regret: Vec::new(),
                last_regret: Vec::new(),
                actions: actions.to_vec(),
            };
            table.insert_serialized(*key, info.clone(), data);
        }
    }

    #[inline]
    fn range(&self, idx: usize) -> std::ops::Range<usize> {
        self.off[idx] as usize..self.off[idx + 1] as usize
    }

    /// Allocate the buffered-sweep vectors (idempotent). `pending` for any
    /// buffered mode; `last` only for the predictive (PCFR+) mode.
    fn ensure_buffers(&mut self, predictive: bool) {
        let n = self.regret.len();
        let pending_n = self.pending_slots[0].max(self.pending_slots[1]);
        if self.pending.len() != pending_n {
            self.pending = vec![0.0; pending_n];
        }
        if predictive && self.last.len() != n {
            self.last = vec![0.0; n];
        }
    }

    fn enable_regret_pruning(&mut self) {
        if self.prune_regret.len() != self.regret.len() {
            self.prune_regret = vec![0.0; self.regret.len()];
        }
    }

    #[inline]
    fn should_prune_action(
        &self,
        idx: usize,
        action: usize,
        current_probability: f64,
        iteration: u64,
        config: Option<&RegretPruningConfig>,
    ) -> bool {
        let Some(config) = config else {
            return false;
        };
        if current_probability != 0.0 || !config.prunes_on_iteration(iteration) {
            return false;
        }
        let slot = self.off[idx] as usize + action;
        self.prune_regret[slot] <= -config.threshold
    }

    fn regret_pruning_candidates(&self, threshold: f32) -> usize {
        self.regret
            .iter()
            .zip(&self.prune_regret)
            .filter(|(clamped, shadow)| **clamped == 0.0 && **shadow <= -threshold)
            .count()
    }

    /// Regret-matching current strategy over info set `idx`.
    #[inline]
    fn current_strategy(&self, idx: usize) -> crate::strategy::ActionProbs {
        let r = &self.regret[self.range(idx)];
        let positive_sum: f64 = r.iter().map(|x| (*x as f64).max(0.0)).sum();
        if positive_sum > 0.0 {
            r.iter()
                .map(|x| (*x as f64).max(0.0) / positive_sum)
                .collect()
        } else {
            crate::strategy::uniform_probs(r.len())
        }
    }

    /// Public-in-crate accessor for the average strategy of info set `idx`,
    /// for the trunk-CFR orchestration in `resolve.rs` (which composes the
    /// trunk and subgame accumulators' averages into one profile).
    pub(crate) fn average_row(&self, idx: usize) -> crate::strategy::ActionProbs {
        self.average_strategy(idx)
    }

    /// Public-in-crate accessor for the CURRENT regret-matching strategy of
    /// info set `idx`. The trunk-CFR loop's intra-round couplings (cached
    /// boundary values, subgame roots, per-round CBV folds) must evaluate the
    /// current iterate — SyncCFR+ backups are defined against the frozen
    /// current strategy; the average lags by ~half the averaging window and
    /// coupling through it makes the alternation chase a stale target.
    pub(crate) fn current_row(&self, idx: usize) -> crate::strategy::ActionProbs {
        self.current_strategy(idx)
    }

    /// Zero the cumulative-strategy accumulator (keeps regrets). The trunk-CFR
    /// loop uses this to re-average the subgames cleanly against the converged
    /// trunk range after the warm alternation (whose average blends the
    /// shifting early-round ranges).
    pub(crate) fn clear_strategy(&mut self) {
        self.strategy.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Widen the regret/strategy accumulators to `f64` for checkpointing (a
    /// no-op copy in the default build; the on-disk format is always `f64` so
    /// `accum-f32` and default checkpoints stay byte-compatible).
    pub(crate) fn regret_strategy_f64(&self) -> (Vec<f64>, Vec<f64>) {
        (
            self.regret.iter().map(|x| *x as f64).collect(),
            self.strategy.iter().map(|x| *x as f64).collect(),
        )
    }

    /// Load regret/strategy from a checkpoint's `f64` vectors (narrowing under
    /// `accum-f32`). Lengths must match this accumulator's layout.
    pub(crate) fn load_regret_strategy_f64(&mut self, regret: &[f64], strategy: &[f64]) {
        assert_eq!(regret.len(), self.regret.len(), "checkpoint regret length");
        assert_eq!(
            strategy.len(),
            self.strategy.len(),
            "checkpoint strategy length"
        );
        for (d, s) in self.regret.iter_mut().zip(regret) {
            *d = *s as Acc;
        }
        for (d, s) in self.strategy.iter_mut().zip(strategy) {
            *d = *s as Acc;
        }
    }

    /// Average (converged) strategy over info set `idx`.
    #[inline]
    fn average_strategy(&self, idx: usize) -> crate::strategy::ActionProbs {
        let svec = &self.strategy[self.range(idx)];
        let total: f64 = svec.iter().map(|x| *x as f64).sum();
        if total > 0.0 {
            svec.iter().map(|x| *x as f64 / total).collect()
        } else {
            crate::strategy::uniform_probs(svec.len())
        }
    }

    /// Iterator of `(key, info, actions, regret, strategy)` slices for
    /// checkpointing directly from SoA, no `StrategyTable` rebuild.
    /// Row view of the cumulative strategy sums, for direct average-strategy
    /// artifact writes (`save_strategy_rows`) without a table rebuild.
    fn strategy_rows<'a>(
        &'a self,
        infos: &'a InfoMeta,
    ) -> impl Iterator<Item = (u64, &'a InfoSet, &'a [AbstractAction], &'a [Acc])> {
        infos
            .iter()
            .enumerate()
            .map(move |(idx, (key, info, actions))| {
                let base = self.off[idx] as usize;
                let end = self.off[idx + 1] as usize;
                (key.0, info, &actions[..], &self.strategy[base..end])
            })
    }

    fn checkpoint_iter<'a>(
        &'a self,
        infos: &'a InfoMeta,
    ) -> impl Iterator<Item = (u64, &'a InfoSet, &'a [AbstractAction], &'a [Acc], &'a [Acc])> {
        infos
            .iter()
            .enumerate()
            .map(move |(idx, (key, info, actions))| {
                let base = self.off[idx] as usize;
                let end = self.off[idx + 1] as usize;
                (
                    key.0,
                    info,
                    &actions[..],
                    &self.regret[base..end],
                    &self.strategy[base..end],
                )
            })
    }

    /// Optimistic (predictive) strategy over info set `idx` (PCFR+).
    #[inline]
    fn predictive_strategy(&self, idx: usize) -> crate::strategy::ActionProbs {
        if self.last.len() != self.regret.len() {
            return self.current_strategy(idx);
        }
        let rng = self.range(idx);
        let mut pred: crate::strategy::ActionProbs = self.regret[rng.clone()]
            .iter()
            .zip(self.last[rng].iter())
            .map(|(r, l)| (*r as f64 + *l).max(0.0))
            .collect();
        let positive_sum: f64 = pred.iter().sum();
        if positive_sum > 0.0 {
            for p in pred.iter_mut() {
                *p /= positive_sum;
            }
            pred
        } else {
            crate::strategy::uniform_probs(pred.len())
        }
    }
}

trait AverageStrategyProfile {
    fn len(&self) -> usize;
    fn average_strategy(&self, idx: usize) -> ActionProbs;
}

impl AverageStrategyProfile for DenseAccum {
    fn len(&self) -> usize {
        self.off.len() - 1
    }

    fn average_strategy(&self, idx: usize) -> ActionProbs {
        DenseAccum::average_strategy(self, idx)
    }
}

struct FixedAverageStrategy<'a> {
    strategies: &'a [ActionProbs],
}

impl AverageStrategyProfile for FixedAverageStrategy<'_> {
    fn len(&self) -> usize {
        self.strategies.len()
    }

    fn average_strategy(&self, idx: usize) -> ActionProbs {
        self.strategies[idx].clone()
    }
}

/// End-of-iteration fold for buffered sweep modes: apply this iteration's
/// buffered instantaneous regrets to the cumulative regrets (RM+ clamp), and
/// remember them as the next iteration's prediction (PCFR+).
fn fold_pending_regrets(
    dense: &mut DenseAccum,
    mode: SweepMode,
    infos: &InfoMeta,
    traversing: Player,
) {
    let remember = mode == SweepMode::Predictive;
    if remember {
        // Preserve the historical PCFR+ behavior: the non-traverser's last
        // instantaneous regret is zero on this alternating sweep.
        dense.last.fill(0.0);
    }
    for (idx, (_, info, _)) in infos.iter().enumerate() {
        if info.player != traversing {
            continue;
        }
        let global = dense.range(idx);
        let pending_base = dense.pending_off[idx] as usize;
        for (action, slot) in global.enumerate() {
            let pending_slot = pending_base + action;
            let pending = dense.pending[pending_slot];
            if !dense.prune_regret.is_empty() {
                dense.prune_regret[slot] += pending as f32;
            }
            if pending != 0.0 {
                // Widen-add-narrow: one rounding per iteration per slot (a
                // no-op in the default f64 build).
                dense.regret[slot] = ((dense.regret[slot] as f64 + pending).max(0.0)) as Acc;
                dense.pending[pending_slot] = 0.0;
            }
            if remember {
                dense.last[slot] = pending;
            }
        }
    }
}

/// CFR+ traversal over a pre-built game tree. No allocations per node.
///
/// `boundary` is the trunk-CFR loop's hook (plan 84 Phase 4): a slice of
/// `(node_id, v0)` pairs — sorted by `node_id` — naming nodes to treat as
/// TERMINALS whose payoff is the cached subgame value `v0` (player-0
/// perspective). Whole-game production solves pass `None`, in which case the
/// added per-node check is a single `is_none` branch (zero-cost). When present,
/// hitting a listed node returns its value without descending or accumulating,
/// so trunk sweeps stop at the round-2 boundary.
#[allow(clippy::too_many_arguments)]
fn cfr_traverse_tree(
    tree: &GameTree,
    node_id: NodeId,
    traversing: Player,
    dealer: Player,
    reach_probs: [f64; 2],
    chance_weight: f64,
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    mode: SweepMode,
    infos: &InfoMeta,
    dense: &mut DenseAccum,
    tremble_eps: f64,
    regret_pruning: Option<&RegretPruningConfig>,
    boundary: Option<&[(NodeId, f64)]>,
) -> f64 {
    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            // payoff_p0 is the hand_value with sign (+ for p0 win, - for p1 win)
            let hand_value = payoff_p0.abs();
            let winner: Player = if payoff_p0 > 0.0 { 0 } else { 1 };

            // Compute match-level payoff
            let new_score = match winner {
                0 => Score {
                    zero: (score.zero + hand_value as u8).min(MATCH_TARGET),
                    one: score.one,
                },
                _ => Score {
                    zero: score.zero,
                    one: (score.one + hand_value as u8).min(MATCH_TARGET),
                },
            };

            if new_score.zero >= MATCH_TARGET {
                return if traversing == 0 { 1.0 } else { -1.0 };
            }
            if new_score.one >= MATCH_TARGET {
                return if traversing == 1 { 1.0 } else { -1.0 };
            }

            // Non-terminal continuation: the next hand is dealt by the OTHER
            // player (the dealer strictly alternates each hand).
            let p0_win_prob = match_values.get(new_score.zero, new_score.one, 1 - dealer);
            let p0_value = 2.0 * p0_win_prob - 1.0;
            if traversing == 0 {
                p0_value
            } else {
                -p0_value
            }
        }
        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            // Trunk-CFR boundary hook: a listed round-2 boundary node is a
            // terminal whose value is the cached subgame value `v0` (p0
            // perspective), converted to the traverser's perspective. No
            // descent, no accumulation — the subtree is a re-solved subgame.
            if let Some(boundary) = boundary {
                if let Ok(pos) = boundary.binary_search_by(|(n, _)| n.cmp(&node_id)) {
                    let v0 = boundary[pos].1;
                    return if traversing == 0 { v0 } else { -v0 };
                }
            }

            let idx = table_idx as usize;
            let num_actions = edges.len();

            // Frozen mão-de-onze decision: at an eleven-decision node, do NOT use
            // regret matching and do NOT accumulate regret/strategy. Force the
            // policy's choice and recurse only into the chosen child with reach
            // probabilities unchanged (probability 1 on the chosen action), so
            // folded hands contribute zero reach to the card play.
            if let Some(policy) = accept_policy {
                if let Some((accept_child, fold_child)) = eleven_decision_children(edges) {
                    let hand = &infos[idx].1.starting_hand;
                    let chosen = if policy.accepts(dealer, hand) {
                        accept_child
                    } else {
                        fold_child
                    };
                    return cfr_traverse_tree(
                        tree,
                        chosen,
                        traversing,
                        dealer,
                        reach_probs,
                        chance_weight,
                        iteration,
                        score,
                        match_values,
                        accept_policy,
                        mode,
                        infos,
                        dense,
                        tremble_eps,
                        regret_pruning,
                        boundary,
                    );
                }
            }

            // Get current strategy, then floor it per TrembleSchedule (see its
            // doc comment) before it is used for reach, regret, or averaging.
            let strategy = match mode {
                SweepMode::Async | SweepMode::Sync => dense.current_strategy(idx),
                SweepMode::Predictive => dense.predictive_strategy(idx),
            };
            let strategy = tremble_strategy(strategy, tremble_eps);

            let mut action_values: crate::strategy::ActionProbs =
                smallvec::smallvec![0.0; num_actions];
            let mut node_value = 0.0;
            let is_owner = player == traversing;
            let mut pruned_actions = 0u16;

            for (i, e) in edges.iter().enumerate() {
                // Exact pruning: below a NON-traversing branch played with
                // probability 0, every quantity this pass can accumulate is
                // weighted by that 0 (the traverser's counterfactual reach and
                // this node's value contribution alike), and in alternating
                // CFR the skipped subtree's own-player accumulation happens on
                // the other pass, where its own reach carries the same 0. The
                // traverser's own zero-prob actions must still be evaluated —
                // regret needs all action values. RM+ produces exact zeros for
                // most actions after early iterations, so this prunes a large
                // fraction of every sweep.
                if !is_owner && strategy[i] == 0.0 {
                    continue;
                }
                if is_owner
                    && dense.should_prune_action(idx, i, strategy[i], iteration, regret_pruning)
                {
                    pruned_actions |= 1 << i;
                    continue;
                }
                let mut new_reach = reach_probs;
                new_reach[player as usize] *= strategy[i];

                let value = cfr_traverse_tree(
                    tree,
                    e.child,
                    traversing,
                    dealer,
                    new_reach,
                    chance_weight,
                    iteration,
                    score,
                    match_values,
                    accept_policy,
                    mode,
                    infos,
                    dense,
                    tremble_eps,
                    regret_pruning,
                    boundary,
                );

                action_values[i] = value;
                node_value += strategy[i] * value;
            }

            let base = dense.off[idx] as usize;
            if player == traversing {
                let opponent = 1 - player;
                for i in 0..num_actions {
                    if pruned_actions & (1 << i) != 0 {
                        continue;
                    }
                    let regret = action_values[i] - node_value;
                    let delta = reach_probs[opponent as usize] * chance_weight * regret;
                    if mode.buffered() {
                        // Sync/PCFR+: collect this iteration's instantaneous
                        // regret; it is folded into cumulative_regret (with the
                        // RM+ clamp) at iteration end, keeping the strategy
                        // frozen for the whole sweep.
                        let pending_base = dense.pending_off[idx] as usize;
                        dense.pending[pending_base + i] += delta;
                    } else {
                        dense.regret[base + i] =
                            ((dense.regret[base + i] as f64 + delta).max(0.0)) as Acc;
                    }
                }
            } else {
                let w = (iteration as f64) * reach_probs[player as usize] * chance_weight;
                for i in 0..num_actions {
                    dense.strategy[base + i] =
                        (dense.strategy[base + i] as f64 + w * strategy[i]) as Acc;
                }
            }

            node_value
        }
    }
}

// ─── Exploitability (best-response) computation ─────────────────────────

/// Compute the exploitability of the average strategy in the table.
///
/// Exploitability = (BR_value_p0 + BR_value_p1) / 2
/// where BR_value_pi is the EV player i achieves by playing a best response
/// against the average strategy of player (1-i).
///
/// Exploitability ε is the AVERAGE of the two players' unilateral deviation
/// gains. Their gains sum to `2ε`, so either player's individual gain is at
/// most `2ε` (and can exceed ε when the two sides are asymmetric). At Nash
/// equilibrium, exploitability = 0.
pub fn compute_exploitability(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
) -> f64 {
    compute_exploitability_with_accept_policy(prebuilt, table, score, match_values, None)
}

/// Compute exploitability for a fixed average-strategy profile already aligned
/// to `prebuilt.info_sets` / `table_idx` order. Used by teacher-chart
/// certification, where the source strategy is a `.teach` tensor rather than a
/// serialized CFR accumulator table.
pub fn compute_exploitability_from_action_probs(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
) -> f64 {
    let br_value_p0 =
        best_response_value_from_action_probs(prebuilt, strategies, score, match_values, 0);
    let br_value_p1 =
        best_response_value_from_action_probs(prebuilt, strategies, score, match_values, 1);
    (br_value_p0 + br_value_p1) / 2.0
}

/// Like [`compute_exploitability`], but with an optional frozen mão-de-onze
/// accept policy. At an eleven-decision node (the decider's own node) the
/// best-responder still MAXes when it is the best-responding player — this is
/// what lets exploitability MEASURE how suboptimal the frozen accept set is.
/// When the node is the opponent's, the frozen choice is followed.
pub fn compute_exploitability_with_accept_policy(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
) -> f64 {
    // Mainline (no frozen accept): LEGAL per-info-set best response.
    if accept_policy.is_none() {
        let br_value_p0 = best_response_value(prebuilt, table, score, match_values, 0);
        let br_value_p1 = best_response_value(prebuilt, table, score, match_values, 1);
        return (br_value_p0 + br_value_p1) / 2.0;
    }

    // Frozen-accept diagnostics (`solve-asym`) keep the historical clairvoyant
    // measure — an upper bound; see `best_response_value_clairvoyant`.
    let br_value_p0 =
        best_response_value_clairvoyant(prebuilt, table, score, match_values, accept_policy, 0);
    let br_value_p1 =
        best_response_value_clairvoyant(prebuilt, table, score, match_values, accept_policy, 1);
    (br_value_p0 + br_value_p1) / 2.0
}

/// Dense-table core of [`compute_exploitability_with_accept_policy`], used by
/// the solve loops (which hold accumulators in `table_idx` order).
fn compute_exploitability_dense(
    prebuilt: &PrebuiltTrees,
    dense: &DenseAccum,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
) -> f64 {
    if accept_policy.is_none() {
        let br0 = best_response_resolve_for_profile(prebuilt, dense, score, match_values, 0).total;
        let br1 = best_response_resolve_for_profile(prebuilt, dense, score, match_values, 1).total;
        return (br0 + br1) / 2.0;
    }
    let br0 = best_response_value_clairvoyant_dense(
        prebuilt,
        dense,
        score,
        match_values,
        accept_policy,
        0,
    );
    let br1 = best_response_value_clairvoyant_dense(
        prebuilt,
        dense,
        score,
        match_values,
        accept_policy,
        1,
    );
    (br0 + br1) / 2.0
}

/// Compute the EV achievable by `br_player` when playing a LEGAL best response
/// against the average strategy of the opponent: one action per INFO SET,
/// chosen to maximize the counterfactual value aggregated across every deal and
/// node the info set spans. Info sets are resolved deepest-first (a decision's
/// descendants in the same player's view always have strictly longer
/// histories), then the strategy profile is evaluated top-down.
///
/// Returns the EV from `br_player`'s perspective (positive = winning).
pub fn best_response_value(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
) -> f64 {
    let dense = DenseAccum::clone_from_table(&prebuilt.info_sets, table);
    best_response_resolve_for_profile(prebuilt, &dense, score, match_values, br_player).total
}

/// Compute the legal best-response value against a fixed average-strategy
/// profile in `table_idx` order.
pub fn best_response_value_from_action_probs(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
) -> f64 {
    assert_eq!(
        strategies.len(),
        prebuilt.info_sets.len(),
        "fixed profile must align with prebuilt.info_sets"
    );
    let fixed = FixedAverageStrategy { strategies };
    best_response_resolve_for_profile(prebuilt, &fixed, score, match_values, br_player).total
}

/// One `br_player` info set's result from a best-response backward-induction
/// pass (see [`best_response_gaps_from_action_probs`]). `br_value` is the
/// counterfactual-weight-normalized value achieved by deviating to the best
/// response at this info set and playing optimally through every descendant,
/// against the OTHER player's fixed average strategy. `weight` is the total
/// counterfactual weight (chance x the other player's average-strategy
/// reach — does NOT include `br_player`'s own reach, since a best-responder
/// can reach any of their own info sets simply by choosing to) the value was
/// resolved under. `weight == 0.0` means no line consistent with the other
/// player's fixed strategy ever reaches this info set at all; `br_value` is
/// meaningless in that case and reported as `f64::NAN`.
#[derive(Clone, Copy, Debug)]
pub struct InfoSetBestResponse {
    pub table_idx: u32,
    pub br_value: f64,
    pub weight: f64,
}

/// Per-info-set best-response values against a fixed average-strategy
/// profile, restricted to `br_player`'s own info sets (call once per player
/// to cover the whole game). Reuses the exact same backward-induction pass
/// as [`best_response_value_from_action_probs`] — same certified numbers,
/// just captures what that function already computes internally and
/// discards. See [`InfoSetBestResponse`] for field semantics, and
/// `RESEARCH_NARRATIVE.md` 2026-07-11 ("Per-infoset best-response gap") for
/// why this is a stronger off-equilibrium quality test than self-loss: it
/// recurses through every descendant via genuine best-response choices,
/// never trusting the solver's own (possibly still undertrained) `q`.
pub fn best_response_gaps_from_action_probs(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
) -> Vec<InfoSetBestResponse> {
    assert_eq!(
        strategies.len(),
        prebuilt.info_sets.len(),
        "fixed profile must align with prebuilt.info_sets"
    );
    let fixed = FixedAverageStrategy { strategies };
    best_response_resolve_for_profile(prebuilt, &fixed, score, match_values, br_player).per_info_set
}

/// Both [`best_response_value_from_action_probs`]'s aggregate AND
/// [`best_response_gaps_from_action_probs`]'s per-info-set detail, from a
/// SINGLE backward-induction pass. Calling those two functions separately on
/// the same inputs pays for the pass twice even though it always computes
/// both internally (see their doc comments) — use this instead whenever a
/// caller wants both, e.g. `--certify` and `--br-gaps` combined
/// (plan 75, `plans/75-per-infoset-solution-quality-br-gap.md`).
pub fn best_response_full_from_action_probs(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
) -> BestResponseResult {
    assert_eq!(
        strategies.len(),
        prebuilt.info_sets.len(),
        "fixed profile must align with prebuilt.info_sets"
    );
    let fixed = FixedAverageStrategy { strategies };
    best_response_resolve_for_profile(prebuilt, &fixed, score, match_values, br_player)
}

/// Aggregate plus per-info-set detail from one backward-induction
/// best-response pass; see [`InfoSetBestResponse`].
pub struct BestResponseResult {
    pub total: f64,
    pub per_info_set: Vec<InfoSetBestResponse>,
    /// Chosen action INDEX for each `table_idx` owned by the responder;
    /// `u8::MAX` for the fixed opponent's rows. Exposing the decisions already
    /// computed by the exact pass lets restricted-game/double-oracle tooling
    /// add selected deviations without retaining every candidate branch.
    pub chosen_actions: Vec<u8>,
}

/// Dense-table core of [`best_response_value`] (used directly by the solve
/// loops, which hold the accumulators in `table_idx` order).
fn best_response_resolve_for_profile<P: AverageStrategyProfile>(
    prebuilt: &PrebuiltTrees,
    profile: &P,
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
) -> BestResponseResult {
    best_response_resolve_core(
        prebuilt,
        profile,
        score,
        match_values,
        br_player,
        &[],
        None,
        None,
    )
    .0
}

/// Per-node best-response values at requested `(tree_idx, node)` pairs —
/// the value `br_player` achieves best-responding from that node onward
/// against the fixed profile, from `br_player`'s perspective, WITHOUT the
/// node's counterfactual weight (the same conditional per-node value
/// [`br_eval_node`] memoizes). `tree_idx` indexes the flat tree list the BR
/// pass builds internally: for each `prebuilt.entries` entry, dealer-0 then
/// dealer-1, skipping empty trees (the shared contract with
/// `subgame::flat_trees`). Runs the standard certified 3-pass machinery —
/// the returned values cannot drift from what certification computes.
///
/// This is the CFV source for safe subgame re-solving (plan 84 Phase 3):
/// boundary CBVs aggregate these per-node values over each opponent view.
pub fn best_response_boundary_values(
    prebuilt: &PrebuiltTrees,
    strategies: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
    nodes: &[(u32, NodeId)],
) -> Vec<f64> {
    assert_eq!(
        strategies.len(),
        prebuilt.info_sets.len(),
        "fixed profile must align with prebuilt.info_sets"
    );
    let fixed = FixedAverageStrategy { strategies };
    best_response_resolve_core(
        prebuilt,
        &fixed,
        score,
        match_values,
        br_player,
        nodes,
        None,
        None,
    )
    .1
}

/// Deep-solve decomposed BR (plan 84 Phase 5): the best-response value for
/// `br_player` over an arena whose round-2 boundary nodes are value terminals.
/// `inject_per_tree[t_idx]` is that flat tree's sorted `(node_id, v)` boundary
/// values in `br_player` perspective (the conditional values a per-subgame BR
/// pass produced). Used by the streaming certificate: the trunk arena carries
/// ~1% of the nodes and each subgame is best-responded independently, so the
/// whole-game BR is assembled without ever materializing the full arena — yet
/// the value is bit-identical to [`best_response_value_from_action_probs`] on
/// the composed profile (same arithmetic, reorganized across the boundary).
pub fn best_response_value_with_boundary_inject<P>(
    prebuilt: &PrebuiltTrees,
    profile: &P,
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
    inject_per_tree: &[Vec<(NodeId, f64)>],
) -> f64
where
    P: crate::game_tree::AverageProfile,
{
    let adapter = AverageProfileAdapter(profile);
    best_response_resolve_core(
        prebuilt,
        &adapter,
        score,
        match_values,
        br_player,
        &[],
        Some(inject_per_tree),
        None,
    )
    .0
    .total
}

/// Per-boundary-node conditional BR values for `br_player` against a fixed
/// profile, on a (sub)arena, WITH per-member root weights folded into the
/// tree deal weights. Returns the read-out value at each requested node — the
/// deep-solve subgame BR that feeds [`best_response_value_with_boundary_inject`].
pub fn best_response_boundary_values_profile<P>(
    prebuilt: &PrebuiltTrees,
    profile: &P,
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
    nodes: &[(u32, NodeId)],
    root_weights: Option<&[f64]>,
) -> Vec<f64>
where
    P: crate::game_tree::AverageProfile,
{
    let adapter = AverageProfileAdapter(profile);
    best_response_resolve_core(
        prebuilt,
        &adapter,
        score,
        match_values,
        br_player,
        nodes,
        None,
        root_weights,
    )
    .1
}

/// Bridges the public [`crate::game_tree::AverageProfile`] trait (deep-solve
/// composed rows, various backings) to the private `AverageStrategyProfile`.
struct AverageProfileAdapter<'a, P: crate::game_tree::AverageProfile>(&'a P);

impl<P: crate::game_tree::AverageProfile> AverageStrategyProfile for AverageProfileAdapter<'_, P> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn average_strategy(&self, idx: usize) -> ActionProbs {
        self.0.average_strategy(idx)
    }
}

/// Shared core: the 3-pass exact BR, plus an optional read-out of per-node
/// memoized values at `boundary` nodes (computed after pass C, when `chosen`
/// is fully resolved — un-visited subtrees evaluate lazily through the same
/// memo).
fn best_response_resolve_core<P: AverageStrategyProfile>(
    prebuilt: &PrebuiltTrees,
    profile: &P,
    score: &Score,
    match_values: &MatchValueTable,
    br_player: Player,
    boundary: &[(u32, NodeId)],
    inject: Option<&[Vec<(NodeId, f64)>]>,
    root_weights: Option<&[f64]>,
) -> (BestResponseResult, Vec<f64>) {
    use std::collections::HashMap;
    let tree_inject = |t_idx: usize| inject.map(|per_tree| per_tree[t_idx].as_slice());
    let root_weight =
        |t_idx: usize, deal_weight: f64| root_weights.map(|rw| rw[t_idx]).unwrap_or(deal_weight);

    assert_eq!(
        profile.len(),
        prebuilt.info_sets.len(),
        "average-strategy profile must align with prebuilt.info_sets"
    );

    // Flat list of (tree, dealer, deal weight) for every built tree.
    let mut trees: Vec<(&GameTree, Player, f64)> = Vec::new();
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if !tree.nodes.is_empty() {
                trees.push((tree, dealer, entry.weight));
            }
        }
    }

    // Pass A (top-down): counterfactual weight of every node — chance weight ×
    // opponent reach; the BR player's own actions contribute reach 1. Also
    // record each info set's depth (= history length: every action appends to
    // both players' histories), derived from the trees rather than the
    // strategy table so sparsely-populated tables (e.g. MCCFR) still order
    // correctly.
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(trees.len());
    // BR-player info set membership: table_idx -> [(tree_idx, node_id)].
    let mut members: HashMap<u32, Vec<(u32, NodeId)>> = HashMap::new();
    let mut idx_depth: HashMap<u32, u32> = HashMap::new();

    for (t_idx, (tree, _dealer, deal_weight)) in trees.iter().enumerate() {
        let mut w = vec![0.0f64; tree.nodes.len()];
        let mut depth = vec![0u32; tree.nodes.len()];
        w[0] = root_weight(t_idx, *deal_weight);
        // Trees are arenas built preorder, but children always have larger ids,
        // so a forward sweep sees parents before children.
        let inj_tree = tree_inject(t_idx);
        for id in 0..tree.nodes.len() {
            // Deep-solve boundary hook: an injected node is a value terminal —
            // no member collection and no descent (its subtree lives in a
            // subgame arena, not here). br_eval_node returns its value directly.
            if let Some(inj) = inj_tree {
                if inj
                    .binary_search_by(|(n, _)| n.cmp(&(id as NodeId)))
                    .is_ok()
                {
                    continue;
                }
            }
            let NodeView::Player {
                player,
                table_idx,
                edges,
            } = tree.view(id as NodeId)
            else {
                continue;
            };
            let wn = w[id];
            for e in edges {
                depth[e.child as usize] = depth[id] + 1;
            }
            if player == br_player {
                members
                    .entry(table_idx)
                    .or_default()
                    .push((t_idx as u32, id as NodeId));
                idx_depth.insert(table_idx, depth[id]);
                for e in edges {
                    w[e.child as usize] += wn;
                }
            } else {
                let avg = profile.average_strategy(table_idx as usize);
                for (i, e) in edges.iter().enumerate() {
                    w[e.child as usize] += wn * avg[i];
                }
            }
        }
        weights.push(w);
    }

    // Pass B: resolve BR choices deepest-first. Q(I, a) sums, over the info
    // set's nodes, cf-weight × value of the action's subtree; deeper BR
    // choices are already fixed when a subtree is evaluated.
    let mut idxs: Vec<u32> = members.keys().copied().collect();
    idxs.sort_by_key(|i| (std::cmp::Reverse(idx_depth[i]), *i));

    // Chosen action per table_idx; u8::MAX = unresolved.
    let mut chosen: Vec<u8> = vec![u8::MAX; profile.len()];
    let mut memo: Vec<Vec<f64>> = trees
        .iter()
        .map(|(tree, _, _)| vec![f64::NAN; tree.nodes.len()])
        .collect();
    // Per-info-set (q[best], total counterfactual weight), kept alongside
    // `chosen` so the per-info-set BR value (q[best] / weight) can be
    // reported without a second traversal.
    let mut idx_result: HashMap<u32, (f64, f64)> = HashMap::with_capacity(members.len());

    for idx in idxs {
        let nodes = &members[&idx];
        let num_actions = {
            let (t_idx, node_id) = nodes[0];
            match trees[t_idx as usize].0.view(node_id) {
                NodeView::Player { edges, .. } => edges.len(),
                _ => unreachable!("member nodes are player nodes"),
            }
        };
        let mut q = vec![0.0f64; num_actions];
        let mut idx_weight = 0.0;
        for &(t_idx, node_id) in nodes {
            let (tree, dealer, _) = trees[t_idx as usize];
            let NodeView::Player { edges, .. } = tree.view(node_id) else {
                unreachable!()
            };
            let wn = weights[t_idx as usize][node_id as usize];
            idx_weight += wn;
            if wn == 0.0 {
                continue;
            }
            let inj_tree = tree_inject(t_idx as usize);
            for (i, e) in edges.iter().enumerate() {
                let v = br_eval_node(
                    tree,
                    e.child,
                    br_player,
                    dealer,
                    score,
                    match_values,
                    profile,
                    &chosen,
                    &mut memo[t_idx as usize],
                    inj_tree,
                );
                q[i] += wn * v;
            }
        }
        let best = q
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite BR values"))
            .map(|(i, _)| i)
            .unwrap_or(0);
        chosen[idx as usize] = best as u8;
        idx_result.insert(idx, (q[best], idx_weight));
    }

    // Pass C: evaluate the resolved profile from each root.
    let mut total = 0.0;
    for (t_idx, (tree, dealer, deal_weight)) in trees.iter().enumerate() {
        let inj_tree = tree_inject(t_idx);
        total += root_weight(t_idx, *deal_weight)
            * br_eval_node(
                tree,
                0,
                br_player,
                *dealer,
                score,
                match_values,
                profile,
                &chosen,
                &mut memo[t_idx],
                inj_tree,
            );
    }

    let per_info_set = idx_result
        .into_iter()
        .map(|(table_idx, (q_best, weight))| InfoSetBestResponse {
            table_idx,
            br_value: if weight > 0.0 {
                q_best / weight
            } else {
                f64::NAN
            },
            weight,
        })
        .collect();

    // Boundary read-out: per-node conditional BR values through the same
    // memo pass C used (lazy for subtrees pass C never reached).
    let boundary_values: Vec<f64> = boundary
        .iter()
        .map(|&(t_idx, node_id)| {
            let (tree, dealer, _) = trees[t_idx as usize];
            let inj_tree = tree_inject(t_idx as usize);
            br_eval_node(
                tree,
                node_id,
                br_player,
                dealer,
                score,
                match_values,
                profile,
                &chosen,
                &mut memo[t_idx as usize],
                inj_tree,
            )
        })
        .collect();

    (
        BestResponseResult {
            total,
            per_info_set,
            chosen_actions: chosen,
        },
        boundary_values,
    )
}

/// Evaluate a subtree under (BR player: resolved `chosen` actions; opponent:
/// average strategy), memoized per node. Values are from `br_player`'s
/// perspective and do NOT include the node's own counterfactual weight.
///
/// `inject` is the deep-solve decomposed-BR hook (plan 84 Phase 5): a slice of
/// `(node_id, v)` pairs — sorted by `node_id`, in `br_player` perspective —
/// naming nodes (the trunk arena's round-2 boundary nodes) to treat as value
/// TERMINALS returning `v` without descending. Whole-arena BR callers pass
/// `None`, a single `is_none` branch (bit-identical to the historical path).
#[allow(clippy::too_many_arguments)]
fn br_eval_node<P: AverageStrategyProfile>(
    tree: &GameTree,
    node_id: NodeId,
    br_player: Player,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    profile: &P,
    chosen: &[u8],
    memo: &mut Vec<f64>,
    inject: Option<&[(NodeId, f64)]>,
) -> f64 {
    let cached = memo[node_id as usize];
    if !cached.is_nan() {
        return cached;
    }
    if let Some(inj) = inject {
        if let Ok(pos) = inj.binary_search_by(|(n, _)| n.cmp(&node_id)) {
            let v = inj[pos].1;
            memo[node_id as usize] = v;
            return v;
        }
    }
    let value = match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            let p0_value = terminal_p0_value(payoff_p0, dealer, score, match_values);
            if br_player == 0 {
                p0_value
            } else {
                -p0_value
            }
        }
        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            if player == br_player {
                let choice = chosen[table_idx as usize];
                assert_ne!(
                    choice,
                    u8::MAX,
                    "deeper BR info sets are resolved before evaluation"
                );
                let child = edges[choice as usize].child;
                br_eval_node(
                    tree,
                    child,
                    br_player,
                    dealer,
                    score,
                    match_values,
                    profile,
                    chosen,
                    memo,
                    inject,
                )
            } else {
                let avg = profile.average_strategy(table_idx as usize);
                let mut v = 0.0;
                for (i, e) in edges.iter().enumerate() {
                    if avg[i] > 0.0 {
                        v += avg[i]
                            * br_eval_node(
                                tree,
                                e.child,
                                br_player,
                                dealer,
                                score,
                                match_values,
                                profile,
                                chosen,
                                memo,
                                inject,
                            );
                    }
                }
                v
            }
        }
    };
    memo[node_id as usize] = value;
    value
}

// ─── Safe subgame re-solving (plan 84 Phase 3, CFR-D gadget) ───────────────

/// One subgame boundary crossing, as the gadget loop needs it (a projection
/// of `subgame::BoundaryNode` kept dependency-light on this side).
#[derive(Clone, Copy, Debug)]
pub struct ResolveMember {
    /// Index into the flat tree list (`subgame::flat_trees` order).
    pub tree_idx: u32,
    pub node: NodeId,
    pub subtree_end: NodeId,
    /// Gadget root weight: `π_c · π_p^blueprint` at the boundary node
    /// (`subgame::reach_excluding` with the OPPONENT excluded).
    pub root_weight: f64,
    /// The opponent's info-set view key at this crossing (Terminate/Follow
    /// decision key).
    pub view_o: crate::info_set::InfoSetKey,
}

/// Terminate/Follow accumulator for one opponent boundary view. Two synthetic
/// actions: 0 = Terminate (payoff = the view's CBV), 1 = Follow.
#[derive(Clone, Copy, Debug, Default)]
struct TfAccum {
    regret: [f64; 2],
    pending: [f64; 2],
    strat: [f64; 2],
}

impl TfAccum {
    fn current(&self) -> [f64; 2] {
        let pos = [self.regret[0].max(0.0), self.regret[1].max(0.0)];
        let sum = pos[0] + pos[1];
        if sum > 0.0 {
            [pos[0] / sum, pos[1] / sum]
        } else {
            [0.5, 0.5]
        }
    }
}

/// Re-solve player `p`'s strategy inside ONE subgame behind the
/// Burch–Johanson–Bowling opt-out gadget, against fixed boundary CBVs for
/// the opponent `o = 1 - p`. Returns the re-solved AVERAGE strategy rows for
/// every info set acting inside the subgame that belongs to `p`.
///
/// Gadget semantics (see `resolve.rs` module docs for the derivation):
/// - each member enters as CHANCE with `root_weight` (resolver reach folded
///   into chance, so both players' in-subgame reaches start at 1);
/// - before play continues, `o` picks Terminate (payoff `cbv[view_o]`, from
///   `o`'s perspective) or Follow (the real subtree) at an accumulator keyed
///   by `view_o` — regret-matched, buffered per sweep, folded at iteration
///   end exactly like the SyncCFR+ table accumulators;
/// - below the boundary, both players re-solve fresh via the ordinary
///   buffered SyncCFR+ traversal over the existing packed subtrees.
///
/// The accumulator is caller-owned and full-size (`prebuilt.info_sets`) but
/// only subgame rows are ever touched: the traversal never leaves the
/// members' contiguous subtree spans, so this function folds and finally
/// CLEARS exactly the touched rows — the caller allocates once and reuses
/// the buffer across thousands of subgames (a per-call `zeros` at
/// production scale is a multi-GB memset per subgame, and folding the whole
/// table per sweep is O(total info sets) instead of O(subgame)). `dense`
/// must arrive zeroed with buffers ensured; it is returned to that state.
/// Members with `root_weight == 0` or an unreachable view CBV are skipped —
/// they contribute nothing to the gadget objective.
pub(crate) fn resolve_subgame(
    prebuilt: &PrebuiltTrees,
    members: &[ResolveMember],
    cbv_by_view: &std::collections::HashMap<crate::info_set::InfoSetKey, Option<f64>>,
    score: &Score,
    match_values: &MatchValueTable,
    p: Player,
    iters: u64,
    dense: &mut DenseAccum,
) -> Vec<(u32, ActionProbs)> {
    use std::collections::HashMap;

    let o = 1 - p;
    // Flat tree list — the shared `tree_idx` contract with subgame.rs.
    let mut trees: Vec<(&GameTree, Player)> = Vec::new();
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if !tree.nodes.is_empty() {
                trees.push((tree, dealer));
            }
        }
    }

    let live: Vec<&ResolveMember> = members
        .iter()
        .filter(|m| m.root_weight > 0.0 && matches!(cbv_by_view.get(&m.view_o), Some(Some(_))))
        .collect();

    // Touched info sets, collected once from the contiguous member spans —
    // drives both the per-sweep subset fold and the final targeted clear.
    // (Spans can overlap across members only via shared info sets, not
    // nodes; the set dedupes.)
    let mut touched_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for m in &live {
        let (tree, _dealer) = trees[m.tree_idx as usize];
        for id in m.node..m.subtree_end {
            if let NodeView::Player { table_idx, .. } = tree.view(id) {
                touched_set.insert(table_idx);
            }
        }
    }
    let mut touched: Vec<u32> = touched_set.into_iter().collect();
    touched.sort_unstable();

    let mut tf: HashMap<crate::info_set::InfoSetKey, TfAccum> = HashMap::new();
    for m in &live {
        tf.entry(m.view_o).or_default();
    }

    for t in 1..=iters {
        for traversing in [0u8, 1u8] {
            for m in &live {
                let (tree, dealer) = trees[m.tree_idx as usize];
                let cbv = cbv_by_view[&m.view_o].expect("live members have CBVs");
                let sigma_tf = tf[&m.view_o].current();

                if traversing == o {
                    // o's Terminate/Follow decision node. Counterfactual
                    // weight = chance only (p's in-subgame reach is 1, its
                    // blueprint reach is folded into root_weight).
                    let v_t = cbv;
                    let v_f = cfr_traverse_tree(
                        tree,
                        m.node,
                        traversing,
                        dealer,
                        [1.0, 1.0],
                        m.root_weight,
                        t,
                        score,
                        match_values,
                        None,
                        SweepMode::Sync,
                        &prebuilt.info_sets,
                        dense,
                        0.0,
                        None,
                        None,
                    );
                    let v_node = sigma_tf[0] * v_t + sigma_tf[1] * v_f;
                    let acc = tf.get_mut(&m.view_o).expect("tf entry");
                    acc.pending[0] += m.root_weight * (v_t - v_node);
                    acc.pending[1] += m.root_weight * (v_f - v_node);
                } else {
                    // p traverses: o's Terminate/Follow acts as a fixed
                    // opponent decision — Follow scales o's reach; o's view
                    // accumulates average-strategy mass exactly like a
                    // non-traversing table node (iteration-weighted own
                    // reach × chance).
                    let mut reach = [1.0, 1.0];
                    reach[o as usize] = sigma_tf[1];
                    if sigma_tf[1] > 0.0 {
                        cfr_traverse_tree(
                            tree,
                            m.node,
                            traversing,
                            dealer,
                            reach,
                            m.root_weight,
                            t,
                            score,
                            match_values,
                            None,
                            SweepMode::Sync,
                            &prebuilt.info_sets,
                            dense,
                            0.0,
                            None,
                            None,
                        );
                    }
                    let w = (t as f64) * m.root_weight;
                    let acc = tf.get_mut(&m.view_o).expect("tf entry");
                    acc.strat[0] += w * sigma_tf[0];
                    acc.strat[1] += w * sigma_tf[1];
                }
            }
            // Fold the sweep: table accumulators (RM+ clamp) then the
            // gadget's own Terminate/Follow buffers, mirroring SyncCFR+'s
            // freeze-then-fold discipline. Subset fold — O(subgame), not
            // O(table).
            fold_pending_regrets_subset(dense, &prebuilt.info_sets, traversing, &touched);
            if traversing == o {
                for acc in tf.values_mut() {
                    for a in 0..2 {
                        if acc.pending[a] != 0.0 {
                            acc.regret[a] = (acc.regret[a] + acc.pending[a]).max(0.0);
                            acc.pending[a] = 0.0;
                        }
                    }
                }
            }
        }
    }

    // Collect p-owned rows from the touched set, then return the shared
    // accumulator to all-zeros by clearing exactly the touched rows (their
    // regret/strategy ranges and pending slots — pending is already zero
    // after the final fold, cleared again defensively for the skipped-fold
    // case of an all-zero sweep).
    let mut rows: Vec<(u32, ActionProbs)> = Vec::new();
    for &idx in &touched {
        if prebuilt.info_sets[idx as usize].1.player == p {
            rows.push((idx, dense.average_strategy(idx as usize)));
        }
    }
    for &idx in &touched {
        let rng = dense.range(idx as usize);
        dense.regret[rng.clone()].fill(0.0);
        dense.strategy[rng.clone()].fill(0.0);
        let pending_base = dense.pending_off[idx as usize] as usize;
        let n = rng.len();
        dense.pending[pending_base..pending_base + n].fill(0.0);
    }
    rows
}

/// Caller-owned accumulator for [`resolve_subgame`]: allocated and
/// buffer-ensured once, reused (and internally re-cleared) across every
/// subgame re-solve of a run.
pub(crate) fn new_resolve_accum(prebuilt: &PrebuiltTrees) -> DenseAccum {
    let mut dense = DenseAccum::zeros(&prebuilt.info_sets);
    dense.ensure_buffers(false);
    dense
}

/// [`fold_pending_regrets`] restricted to an explicit info-set subset —
/// identical per-slot semantics, O(subset) instead of O(table). Used by the
/// subgame re-solve loop, whose sweeps only ever write inside one subgame.
fn fold_pending_regrets_subset(
    dense: &mut DenseAccum,
    infos: &InfoMeta,
    traversing: Player,
    subset: &[u32],
) {
    for &idx in subset {
        let idx = idx as usize;
        if infos[idx].1.player != traversing {
            continue;
        }
        let global = dense.range(idx);
        let pending_base = dense.pending_off[idx] as usize;
        for (action, slot) in global.enumerate() {
            let pending_slot = pending_base + action;
            let pending = dense.pending[pending_slot];
            if !dense.prune_regret.is_empty() {
                dense.prune_regret[slot] += pending as f32;
            }
            if pending != 0.0 {
                dense.regret[slot] = ((dense.regret[slot] as f64 + pending).max(0.0)) as Acc;
                dense.pending[pending_slot] = 0.0;
            }
        }
    }
}

// ─── Trunk-CFR loop drivers (plan 84 Phase 4) ─────────────────────────────

/// Flat `(tree, dealer, deal_weight)` list matching [`crate::subgame::flat_trees`]
/// order (the shared `tree_idx` contract). Kept here so the Phase-4 drivers
/// don't reach across crates for tree references.
fn flat_trees_dw(prebuilt: &PrebuiltTrees) -> Vec<(&GameTree, Player, f64)> {
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

/// One trunk SyncCFR+ iteration (both alternating player passes) with the
/// round-2 boundary treated as value terminals. `boundary_per_tree[flat_idx]`
/// is the sorted `(node, v0)` list for that flat tree (empty ⇒ no boundary).
/// Only `trunk_infosets` are folded — the traversal never writes below a
/// boundary. `dense` must have pending buffers ensured. Reuses the shared
/// [`cfr_traverse_tree`]; no second CFR implementation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trunk_iteration(
    prebuilt: &PrebuiltTrees,
    dense: &mut DenseAccum,
    boundary_per_tree: &[Vec<(NodeId, f64)>],
    trunk_infosets: &[u32],
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
) {
    let trees = flat_trees_dw(prebuilt);
    for traversing in [0u8, 1u8] {
        for (t_idx, (tree, dealer, weight)) in trees.iter().enumerate() {
            let boundary = boundary_per_tree.get(t_idx).map(|v| v.as_slice());
            cfr_traverse_tree(
                tree,
                0,
                traversing,
                *dealer,
                [1.0, 1.0],
                *weight,
                iteration,
                score,
                match_values,
                None,
                SweepMode::Sync,
                &prebuilt.info_sets,
                dense,
                0.0,
                None,
                boundary,
            );
        }
        fold_pending_regrets_subset(dense, &prebuilt.info_sets, traversing, trunk_infosets);
    }
}

/// One boundary crossing as a gadget-free subgame root: the trunk reach split
/// per player (WITHOUT deal weight) and the deal weight as the chance factor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubgameRoot {
    pub tree_idx: u32,
    pub node: NodeId,
    /// `[π_0^trunk(h), π_1^trunk(h)]` — each player's own trunk reach product.
    pub reach: [f64; 2],
    /// Deal weight `π_c(h)`.
    pub chance: f64,
}

/// One SyncCFR+ iteration of the gadget-FREE subgame re-solve (both alternating
/// passes). Each root enters the ordinary traversal with its trunk reach and
/// deal-weight chance, so the counterfactual weights match the whole-game solve
/// restricted to the subgame. All subgames share one `dense` (their info sets
/// are disjoint) and stay warm across rounds. Only `subgame_infosets` fold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn subgame_root_iteration(
    prebuilt: &PrebuiltTrees,
    dense: &mut DenseAccum,
    roots: &[SubgameRoot],
    subgame_infosets: &[u32],
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
) {
    let trees = flat_trees_dw(prebuilt);
    for traversing in [0u8, 1u8] {
        for r in roots {
            let (tree, dealer, _w) = trees[r.tree_idx as usize];
            cfr_traverse_tree(
                tree,
                r.node,
                traversing,
                dealer,
                r.reach,
                r.chance,
                iteration,
                score,
                match_values,
                None,
                SweepMode::Sync,
                &prebuilt.info_sets,
                dense,
                0.0,
                None,
                None,
            );
        }
        fold_pending_regrets_subset(dense, &prebuilt.info_sets, traversing, subgame_infosets);
    }
}

/// Player-0-perspective expected value of a subtree under a FIXED profile
/// (both players play `profile`'s average strategy at every node), memoized
/// per node. Mirrors [`eq_traverse_tree`] but reads `&[ActionProbs]` and starts
/// at an arbitrary node — used for the trunk's cached boundary value pairs.
fn profile_eval_node(
    tree: &GameTree,
    node_id: NodeId,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    profile: &[ActionProbs],
    memo: &mut [f64],
) -> f64 {
    let cached = memo[node_id as usize];
    if !cached.is_nan() {
        return cached;
    }
    let value = match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            terminal_p0_value(payoff_p0, dealer, score, match_values)
        }
        NodeView::Player {
            table_idx, edges, ..
        } => {
            let avg = &profile[table_idx as usize];
            let mut v = 0.0;
            for (i, e) in edges.iter().enumerate() {
                if avg[i] > 0.0 {
                    v += avg[i]
                        * profile_eval_node(
                            tree,
                            e.child,
                            dealer,
                            score,
                            match_values,
                            profile,
                            memo,
                        );
                }
            }
            v
        }
    };
    memo[node_id as usize] = value;
    value
}

/// Player-0-perspective profile value at each `(tree_idx, node)` under both
/// players' average strategies in `profile`. The trunk-CFR loop caches these as
/// the boundary value pairs `(v0, -v0)`. One memo per touched tree keeps repeat
/// boundary reads within a tree linear.
pub fn profile_boundary_values(
    prebuilt: &PrebuiltTrees,
    profile: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
    nodes: &[(u32, NodeId)],
) -> Vec<f64> {
    assert_eq!(
        profile.len(),
        prebuilt.info_sets.len(),
        "profile must align with prebuilt.info_sets"
    );
    let trees = flat_trees_dw(prebuilt);
    let mut memos: std::collections::HashMap<u32, Vec<f64>> = std::collections::HashMap::new();
    nodes
        .iter()
        .map(|&(t_idx, node)| {
            let (tree, dealer, _w) = trees[t_idx as usize];
            let memo = memos
                .entry(t_idx)
                .or_insert_with(|| vec![f64::NAN; tree.nodes.len()]);
            profile_eval_node(tree, node, dealer, score, match_values, profile, memo)
        })
        .collect()
}

/// Whole-game value per dealer under a fixed average-strategy profile (both
/// players play `profile`), the [`compute_game_value_per_dealer`] analog for a
/// `&[ActionProbs]` profile — used to score the composed trunk-CFR result.
pub fn game_value_per_dealer_from_action_probs(
    prebuilt: &PrebuiltTrees,
    profile: &[ActionProbs],
    score: &Score,
    match_values: &MatchValueTable,
) -> (f64, f64) {
    assert_eq!(
        profile.len(),
        prebuilt.info_sets.len(),
        "profile must align with prebuilt.info_sets"
    );
    let mut values = [0.0f64; 2];
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if tree.nodes.is_empty() {
                continue;
            }
            let mut memo = vec![f64::NAN; tree.nodes.len()];
            values[dealer as usize] += entry.weight
                * profile_eval_node(tree, 0, dealer, score, match_values, profile, &mut memo);
        }
    }
    (values[0], values[1])
}

/// Player-0-perspective value of a trunk tree under `trunk_profile`, with the
/// round-2 boundary nodes returning the injected (already p0-perspective)
/// composed subgame values instead of descending. Memoized per node.
fn profile_eval_node_inject<P: crate::game_tree::AverageProfile>(
    tree: &GameTree,
    node_id: NodeId,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    profile: &P,
    inject: &[(NodeId, f64)],
    memo: &mut [f64],
) -> f64 {
    let cached = memo[node_id as usize];
    if !cached.is_nan() {
        return cached;
    }
    if let Ok(pos) = inject.binary_search_by(|(n, _)| n.cmp(&node_id)) {
        let v = inject[pos].1;
        memo[node_id as usize] = v;
        return v;
    }
    let value = match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            terminal_p0_value(payoff_p0, dealer, score, match_values)
        }
        NodeView::Player {
            table_idx, edges, ..
        } => {
            let avg = profile.average_strategy(table_idx as usize);
            let mut v = 0.0;
            for (i, e) in edges.iter().enumerate() {
                if avg[i] > 0.0 {
                    v += avg[i]
                        * profile_eval_node_inject(
                            tree,
                            e.child,
                            dealer,
                            score,
                            match_values,
                            profile,
                            inject,
                            memo,
                        );
                }
            }
            v
        }
    };
    memo[node_id as usize] = value;
    value
}

/// Deep-solve decomposed game value per dealer: the composed profile's value
/// (both players play the composed profile) evaluated over the trunk arena,
/// with each boundary node returning its p0-perspective composed subgame value
/// from `inject_p0_per_tree[flat_tree_idx]` (sorted `(node, v0)`). Bit-identical
/// to [`game_value_per_dealer_from_action_probs`] on the full composed profile.
pub fn game_value_per_dealer_deep<P: crate::game_tree::AverageProfile>(
    trunk: &PrebuiltTrees,
    trunk_profile: &P,
    score: &Score,
    match_values: &MatchValueTable,
    inject_p0_per_tree: &[Vec<(NodeId, f64)>],
) -> (f64, f64) {
    let trees = flat_trees_dw(trunk);
    let mut values = [0.0f64; 2];
    for (t_idx, (tree, dealer, weight)) in trees.iter().enumerate() {
        let mut memo = vec![f64::NAN; tree.nodes.len()];
        let v0 = profile_eval_node_inject(
            tree,
            0,
            *dealer,
            score,
            match_values,
            trunk_profile,
            &inject_p0_per_tree[t_idx],
            &mut memo,
        );
        values[*dealer as usize] += weight * v0;
    }
    (values[0], values[1])
}

/// Player 0's match-level value (±1) of a terminal hand result, honoring the
/// dealer-exact continuation lookup.
pub(crate) fn terminal_p0_value(
    payoff_p0: f64,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
) -> f64 {
    let hand_value = payoff_p0.abs();
    let winner: Player = if payoff_p0 > 0.0 { 0 } else { 1 };
    let new_score = match winner {
        0 => Score {
            zero: (score.zero + hand_value as u8).min(MATCH_TARGET),
            one: score.one,
        },
        _ => Score {
            zero: score.zero,
            one: (score.one + hand_value as u8).min(MATCH_TARGET),
        },
    };
    if new_score.zero >= MATCH_TARGET {
        1.0
    } else if new_score.one >= MATCH_TARGET {
        -1.0
    } else {
        2.0 * match_values.get(new_score.zero, new_score.one, 1 - dealer) - 1.0
    }
}

/// CLAIRVOYANT best-response EV — the historical measure, kept for diagnostics
/// and for the frozen-accept (`solve-asym`) tooling. Maximizes independently
/// inside every deal's tree, which lets the "best response" condition on the
/// opponent's hidden hand: `Σ_deals max` instead of the legal `max Σ_deals`.
/// STRICT upper bound on real exploitability; the gap is the value of
/// clairvoyance and does NOT vanish at equilibrium (it produced the phantom
/// "walls" at 11×11 ~0.05 and 11×10 ~0.15/game — see SOLVER_BENCHMARKS
/// 2026-07-02).
#[allow(clippy::too_many_arguments)]
pub fn best_response_value_clairvoyant(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    br_player: Player,
) -> f64 {
    let dense = DenseAccum::clone_from_table(&prebuilt.info_sets, table);
    best_response_value_clairvoyant_dense(
        prebuilt,
        &dense,
        score,
        match_values,
        accept_policy,
        br_player,
    )
}

/// Dense-table core of [`best_response_value_clairvoyant`].
#[allow(clippy::too_many_arguments)]
fn best_response_value_clairvoyant_dense(
    prebuilt: &PrebuiltTrees,
    dense: &DenseAccum,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    br_player: Player,
) -> f64 {
    let mut total_value = 0.0;

    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if tree.nodes.is_empty() {
                continue; // dealer excluded by the build's dealer filter
            }
            total_value += entry.weight
                * br_traverse_tree(
                    tree,
                    0,
                    br_player,
                    dealer,
                    1.0, // opponent reach
                    score,
                    match_values,
                    accept_policy,
                    &prebuilt.info_sets,
                    dense,
                );
        }
    }

    total_value
}

/// Traverse a pre-built tree computing the best-response value for `br_player`.
///
/// At `br_player`'s nodes: take the max over action values (best response).
/// At opponent's nodes: weight by the opponent's average strategy.
#[allow(clippy::too_many_arguments)]
fn br_traverse_tree(
    tree: &GameTree,
    node_id: NodeId,
    br_player: Player,
    dealer: Player,
    opponent_reach: f64,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    infos: &InfoMeta,
    dense: &DenseAccum,
) -> f64 {
    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            let p0_value = terminal_p0_value(payoff_p0, dealer, score, match_values);
            // Return value from br_player's perspective, weighted by opponent reach
            let value = if br_player == 0 { p0_value } else { -p0_value };
            opponent_reach * value
        }
        NodeView::Player {
            player,
            table_idx,
            edges,
        } => {
            let idx = table_idx as usize;

            // Frozen mão-de-onze decision when it is the OPPONENT's node: follow
            // the policy's choice rather than the (meaningless) stored average
            // strategy of the frozen node. When it is the best-responder's OWN
            // node we deliberately fall through to the max branch below, so
            // exploitability measures how suboptimal the frozen accept set is.
            if let Some(policy) = accept_policy {
                if player != br_player {
                    if let Some((accept_child, fold_child)) = eleven_decision_children(edges) {
                        let hand = &infos[idx].1.starting_hand;
                        let chosen = if policy.accepts(dealer, hand) {
                            accept_child
                        } else {
                            fold_child
                        };
                        return br_traverse_tree(
                            tree,
                            chosen,
                            br_player,
                            dealer,
                            opponent_reach,
                            score,
                            match_values,
                            accept_policy,
                            infos,
                            dense,
                        );
                    }
                }
            }

            if player == br_player {
                // Best-responding player: take the max over actions
                let mut best_value = f64::NEG_INFINITY;
                for e in edges {
                    let value = br_traverse_tree(
                        tree,
                        e.child,
                        br_player,
                        dealer,
                        opponent_reach,
                        score,
                        match_values,
                        accept_policy,
                        infos,
                        dense,
                    );
                    if value > best_value {
                        best_value = value;
                    }
                }
                best_value
            } else {
                // Opponent: use their average strategy (uniform if never visited
                // — average_strategy of all-zero sums is uniform).
                let avg_strategy = dense.average_strategy(idx);
                let mut node_value = 0.0;

                for (i, e) in edges.iter().enumerate() {
                    let prob = avg_strategy[i];
                    if prob > 0.0 {
                        let value = br_traverse_tree(
                            tree,
                            e.child,
                            br_player,
                            dealer,
                            opponent_reach * prob,
                            score,
                            match_values,
                            accept_policy,
                            infos,
                            dense,
                        );
                        node_value += value;
                    }
                }
                node_value
            }
        }
    }
}

// ─── Game value under average strategy (for match-value extraction) ─────

/// Expected payoff to player 0 in `[-1, 1]` when both players follow the
/// average strategy, weighted over deals and both dealer assignments.
///
/// Divide by 2 after summing dealer-0 and dealer-1 trees (same convention as
/// [`best_response_value`]).
pub fn compute_game_value(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
) -> f64 {
    let (v_d0, v_d1) = compute_game_value_per_dealer(prebuilt, table, score, match_values);
    (v_d0 + v_d1) / 2.0
}

/// Like [`compute_game_value_per_dealer`], but follows an optional frozen
/// mão-de-onze accept policy at eleven-decision nodes (avg vs avg elsewhere).
pub fn compute_game_value_per_dealer_with_accept_policy(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
) -> (f64, f64) {
    let dense = DenseAccum::clone_from_table(&prebuilt.info_sets, table);
    compute_game_value_per_dealer_dense(prebuilt, &dense, score, match_values, accept_policy)
}

/// Dense-table core of [`compute_game_value_per_dealer_with_accept_policy`].
fn compute_game_value_per_dealer_dense(
    prebuilt: &PrebuiltTrees,
    dense: &DenseAccum,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
) -> (f64, f64) {
    let mut values = [0.0f64; 2];
    for entry in &prebuilt.entries {
        for (dealer, tree) in [(0, &entry.tree_dealer_0), (1, &entry.tree_dealer_1)] {
            if tree.nodes.is_empty() {
                continue; // dealer excluded by the build's dealer filter
            }
            values[dealer as usize] += entry.weight
                * eq_traverse_tree(
                    tree,
                    0,
                    dealer,
                    score,
                    match_values,
                    accept_policy,
                    &prebuilt.info_sets,
                    dense,
                );
        }
    }
    (values[0], values[1])
}

/// Like `compute_game_value`, but returns player 0's value (in `[-1, 1]`, avg vs
/// avg) split by who deals the hand: `(value_when_p0_deals, value_when_p1_deals)`.
/// The pé (dealer) advantage is exposed by the gap between these — they average
/// to `compute_game_value`. The dealer plays last, so the dealer is favored:
/// `value_when_p0_deals > 0 > value_when_p1_deals` (by symmetry, `v_d1 = -v_d0`
/// when the score is symmetric).
pub fn compute_game_value_per_dealer(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    match_values: &MatchValueTable,
) -> (f64, f64) {
    compute_game_value_per_dealer_with_accept_policy(prebuilt, table, score, match_values, None)
}

/// Traverse using both players' average strategies at every decision node.
///
/// When `accept_policy` is set, an eleven-decision node follows the frozen
/// choice (recurse only into the chosen child) instead of averaging.
#[allow(clippy::too_many_arguments)]
fn eq_traverse_tree(
    tree: &GameTree,
    node_id: NodeId,
    dealer: Player,
    score: &Score,
    match_values: &MatchValueTable,
    accept_policy: Option<&AcceptPolicy>,
    infos: &InfoMeta,
    dense: &DenseAccum,
) -> f64 {
    match tree.view(node_id) {
        NodeView::Terminal { payoff_p0 } => {
            terminal_p0_value(payoff_p0, dealer, score, match_values)
        }
        NodeView::Player {
            table_idx, edges, ..
        } => {
            let idx = table_idx as usize;
            // Frozen mão-de-onze decision: follow the policy's choice.
            if let Some(policy) = accept_policy {
                if let Some((accept_child, fold_child)) = eleven_decision_children(edges) {
                    let hand = &infos[idx].1.starting_hand;
                    let chosen = if policy.accepts(dealer, hand) {
                        accept_child
                    } else {
                        fold_child
                    };
                    return eq_traverse_tree(
                        tree,
                        chosen,
                        dealer,
                        score,
                        match_values,
                        accept_policy,
                        infos,
                        dense,
                    );
                }
            }

            let avg_strategy = dense.average_strategy(idx);
            let mut node_value = 0.0;
            for (i, e) in edges.iter().enumerate() {
                node_value += avg_strategy[i]
                    * eq_traverse_tree(
                        tree,
                        e.child,
                        dealer,
                        score,
                        match_values,
                        accept_policy,
                        infos,
                        dense,
                    );
            }
            node_value
        }
    }
}

// ─── Accept-equity extraction (for the freeze + policy-iteration solver) ──

/// For each dealer arrangement and each decider starting hand, compute the
/// equity (in `[0, 1]` win probability) of ACCEPTING the mão-de-onze and PLAYING
/// the resulting card-play subtree under the current average strategy.
///
/// The root of each per-dealer tree is the eleven-decision node (the decider's
/// `{AcceptEleven, FoldEleven}` node). We locate its AcceptEleven child and value
/// that subtree with [`eq_traverse_tree`] (avg vs avg, no accept policy below it),
/// giving player 0's ±1 value of *playing* that hand `h`. We accumulate this
/// weighted by the deal weight, keyed by the decider's starting hand, then
/// normalize per hand and convert to a `[0, 1]` win prob via `(v + 1) / 2`.
///
/// Returned maps are indexed by dealer (0 or 1). The equity is "value if you
/// accept" and is therefore independent of the current accept set — only of the
/// card-play strategy below the accept node.
///
/// Note: the value is always player 0's win prob. The decider is whoever has
/// score 11 in the surrounding state; the CLI applies the correct fold threshold
/// per dealer using the same player-0 convention.
pub fn extract_accept_equities(
    prebuilt: &PrebuiltTrees,
    table: &StrategyTable,
    score: &Score,
    mv: &MatchValueTable,
    decider: Player,
) -> [std::collections::HashMap<AbstractHand, f64>; 2] {
    use std::collections::HashMap;

    // Per dealer: per-hand (sum_weighted_value, sum_weight).
    let mut sums: [HashMap<AbstractHand, (f64, f64)>; 2] = [HashMap::new(), HashMap::new()];

    let dense = DenseAccum::clone_from_table(&prebuilt.info_sets, table);
    for entry in &prebuilt.entries {
        for (dealer, tree) in [
            (0usize, &entry.tree_dealer_0),
            (1usize, &entry.tree_dealer_1),
        ] {
            if tree.is_empty() {
                continue; // dealer excluded by the build's dealer filter
            }
            let NodeView::Player {
                table_idx, edges, ..
            } = tree.view(0)
            else {
                continue;
            };
            let Some((accept_child, _fold_child)) = eleven_decision_children(edges) else {
                continue;
            };
            let hand = prebuilt.info_sets[table_idx as usize]
                .1
                .starting_hand
                .clone();

            // The decider's ±1 value of playing this hand when the DECIDER plays a
            // best response to the opponent's current average card play. Using the
            // best-response value (not the average) is essential: a currently-folded
            // hand has zero reach, so its card-play info sets are never trained and
            // its *average*-play value is garbage (uniform) — which would keep it
            // wrongly folded forever. The best-response value is the true "value if
            // you accept and play well", so `accept iff value > F` becomes
            // consistent with what BR_decider would do (removing that exploitability).
            let ev = br_traverse_tree(
                tree,
                accept_child,
                decider,
                dealer as Player,
                1.0,
                score,
                mv,
                None,
                &prebuilt.info_sets,
                &dense,
            );
            // Convert the decider's EV to player 0's ±1 value for the maps below.
            let v = if decider == 0 { ev } else { -ev };

            let slot = sums[dealer].entry(hand).or_insert((0.0, 0.0));
            slot.0 += entry.weight * v;
            slot.1 += entry.weight;
        }
    }

    let mut out: [HashMap<AbstractHand, f64>; 2] = [HashMap::new(), HashMap::new()];
    for dealer in 0..2 {
        for (hand, (sum_v, sum_w)) in &sums[dealer] {
            if *sum_w > 0.0 {
                let avg_v = sum_v / sum_w; // P0 ±1 value of playing this hand
                let win_prob = (avg_v + 1.0) / 2.0; // map to [0, 1]
                out[dealer].insert(hand.clone(), win_prob);
            }
        }
    }
    out
}

// ─── MCCFR External Sampling ────────────────────────────────────────────

/// Public re-export of deal enumeration so callers can pre-load deals.
pub fn enumerate_deals_pub(tc: &TurnupClass) -> Vec<crate::abstraction::AbstractDeal> {
    enumerate_deals(tc)
}

/// Run a fixed number of MCCFR iterations into an existing strategy table,
/// using a pre-loaded deal list. Used by the `compare` mode to interleave
/// iteration and exploitability measurement without re-loading deals.
pub fn run_mccfr_chunk(
    score: Score,
    tc: TurnupClass,
    iterations: u64,
    match_values: &MatchValueTable,
    rng: &mut impl rand::Rng,
    deals: &[crate::abstraction::AbstractDeal],
    table: &mut StrategyTable,
) {
    // Build cumulative weight distribution once per chunk (cheap)
    let mut cumulative_weights = Vec::with_capacity(deals.len());
    let mut running = 0.0f64;
    for deal in deals {
        running += deal.weight;
        cumulative_weights.push(running);
    }

    for t in 1..=iterations {
        let traversing = (t % 2) as Player;

        let r: f64 = rng.random();
        let deal_idx = cumulative_weights
            .partition_point(|&w| w < r)
            .min(deals.len() - 1);
        let deal = &deals[deal_idx];

        let dealer: Player = if rng.random::<bool>() { 0 } else { 1 };

        let state =
            match crate::game_tree::TraversalState::from_deal(dealer, score.clone(), tc, deal) {
                Ok(s) => s,
                Err(_) => continue,
            };

        mccfr_external_sampling(
            &state,
            traversing,
            dealer,
            [1.0, 1.0],
            t,
            &score,
            match_values,
            table,
            rng,
        );
    }
}

#[derive(Debug)]
struct MccfrBatchUpdate {
    regret: ActionProbs,
    strategy: ActionProbs,
}

impl MccfrBatchUpdate {
    fn zeros(actions: usize) -> Self {
        Self {
            regret: smallvec::smallvec![0.0; actions],
            strategy: smallvec::smallvec![0.0; actions],
        }
    }
}

/// Summary of a true frozen-strategy external-sampling mini-batch chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MccfrMiniBatchStats {
    pub batches: u64,
    pub samples: u64,
    pub info_sets_before: usize,
    pub info_sets_after: usize,
}

/// Run external-sampling MCCFR in true mini-batches. Within each batch the
/// strategy is frozen, `batch_size` independent chance/deal samples accumulate
/// into a sparse pending map, and the combined regret/average updates fold only
/// after the batch. The table persists between calls, allowing bounded chunks
/// and periodic evaluation without rebuilding any game arena.
///
/// When `seed` is present, a newly visited row starts from the seed's AVERAGE
/// action probabilities encoded as positive pseudo-regret of total
/// `seed_regret_mass`. The cumulative average still starts at zero. Common
/// actions therefore inherit neighboring-score play while a deeper band's new
/// raises start at zero current probability but remain enumerated whenever
/// their owner traverses, so MCCFR can discover them.
#[allow(clippy::too_many_arguments)]
pub fn run_mccfr_minibatch_chunk(
    score: Score,
    tc: TurnupClass,
    batches: u64,
    batch_size: usize,
    start_batch: u64,
    match_values: &MatchValueTable,
    rng: &mut impl rand::Rng,
    deals: &[crate::abstraction::AbstractDeal],
    table: &mut StrategyTable,
    seed: Option<&dyn PolicyLookup>,
    seed_regret_mass: f64,
    dealer_filter: Option<Player>,
) -> MccfrMiniBatchStats {
    assert!(batch_size > 0, "MCCFR batch size must be positive");
    assert!(!deals.is_empty(), "MCCFR requires at least one deal");
    assert!(
        seed_regret_mass >= 0.0,
        "seed regret mass must be nonnegative"
    );

    let mut cumulative_weights = Vec::with_capacity(deals.len());
    let mut running = 0.0f64;
    for deal in deals {
        running += deal.weight;
        cumulative_weights.push(running);
    }
    let info_sets_before = table.len();

    for batch in 0..batches {
        let batch_index = start_batch + batch + 1;
        let traversing = (batch_index % 2) as Player;
        let mut pending: std::collections::HashMap<InfoSetKey, MccfrBatchUpdate> =
            std::collections::HashMap::new();

        for _ in 0..batch_size {
            let r: f64 = rng.random();
            let deal_idx = cumulative_weights
                .partition_point(|&weight| weight < r)
                .min(deals.len() - 1);
            let deal = &deals[deal_idx];
            let dealer = dealer_filter.unwrap_or_else(|| if rng.random::<bool>() { 0 } else { 1 });
            let Ok(state) =
                crate::game_tree::TraversalState::from_deal(dealer, score.clone(), tc, deal)
            else {
                continue;
            };
            mccfr_external_sampling_batched(
                &state,
                traversing,
                dealer,
                [1.0, 1.0],
                &score,
                match_values,
                table,
                &mut pending,
                seed,
                seed_regret_mass,
                rng,
            );
        }

        for (key, update) in pending {
            let data = table
                .data
                .get_mut(&key)
                .expect("batched MCCFR update row inserted during traversal");
            for i in 0..data.actions.len() {
                if update.regret[i] != 0.0 {
                    data.cumulative_regret[i] =
                        (data.cumulative_regret[i] + update.regret[i]).max(0.0);
                }
                data.cumulative_strategy[i] += update.strategy[i];
            }
        }
    }

    MccfrMiniBatchStats {
        batches,
        samples: batches.saturating_mul(batch_size as u64),
        info_sets_before,
        info_sets_after: table.len(),
    }
}

/// Run MCCFR with external sampling for a given score state and turnup class.
///
/// Unlike full CFR+, each iteration samples a single deal and samples
/// opponent/chance actions. Much cheaper per iteration but noisier.
pub fn solve_mccfr(
    score: Score,
    tc: TurnupClass,
    iterations: u64,
    match_values: &MatchValueTable,
    rng: &mut impl rand::Rng,
) -> (StrategyTable, SolveStats) {
    let start = Instant::now();
    let deals = enumerate_deals(&tc);
    let num_deals = deals.len();

    info!(
        "MCCFR solving score ({}, {}) tc={} | {} deals, {} iterations",
        score.zero, score.one, tc.blocked_plain_level, num_deals, iterations
    );

    // Build cumulative weight distribution for sampling deals
    let mut cumulative_weights = Vec::with_capacity(deals.len());
    let mut running = 0.0;
    for deal in &deals {
        running += deal.weight;
        cumulative_weights.push(running);
    }

    let mut table = StrategyTable::new();

    let iter_start = Instant::now();
    for t in 1..=iterations {
        let traversing = (t % 2) as Player;

        // Sample a deal
        let r: f64 = rng.random();
        let deal_idx = cumulative_weights
            .partition_point(|&w| w < r)
            .min(deals.len() - 1);
        let deal = &deals[deal_idx];

        // Sample dealer (50/50)
        let dealer: Player = if rng.random::<bool>() { 0 } else { 1 };

        // Build traversal state for this deal
        let state =
            match crate::game_tree::TraversalState::from_deal(dealer, score.clone(), tc, deal) {
                Ok(s) => s,
                Err(_) => continue,
            };

        // Run external sampling CFR traversal
        mccfr_external_sampling(
            &state,
            traversing,
            dealer,
            [1.0, 1.0],
            t,
            &score,
            match_values,
            &mut table,
            rng,
        );

        if t <= 3 || t % 10000 == 0 || t == iterations {
            let elapsed = iter_start.elapsed().as_secs_f64();
            info!(
                "  MCCFR iteration {}/{} ({:.0} iters/sec)",
                t,
                iterations,
                t as f64 / elapsed
            );
        }
    }

    let total_secs = start.elapsed().as_secs_f64();
    let iter_secs = iter_start.elapsed().as_secs_f64();

    let stats = SolveStats {
        score: (score.zero, score.one),
        turnup_class: tc.blocked_plain_level,
        iterations,
        num_deals,
        num_info_sets: table.len(),
        total_nodes: 0, // not tracked during MCCFR iteration
        total_duration_secs: total_secs,
        build_tree_secs: 0.0,
        per_iteration_secs: if iterations > 0 {
            iter_secs / iterations as f64
        } else {
            0.0
        },
        estimated_memory_bytes: table.len() * 100, // rough estimate
        exploitability: None, // compute separately via compute_exploitability_from_deals()
        exploitability_history: Vec::new(),
        game_value_p0: None,
        game_value_per_dealer: None,
    };

    info!("{}", stats);

    (table, stats)
}

/// External sampling MCCFR traversal.
///
/// At chance nodes: a single deal is already sampled (no chance node in the tree).
/// At traversing player nodes: enumerate all actions.
/// At opponent nodes: sample one action according to current strategy.
fn mccfr_external_sampling(
    state: &crate::game_tree::TraversalState,
    traversing: Player,
    dealer: Player,
    reach_probs: [f64; 2],
    iteration: u64,
    score: &Score,
    match_values: &MatchValueTable,
    table: &mut StrategyTable,
    rng: &mut impl rand::Rng,
) -> f64 {
    if state.is_terminal() {
        debug_assert!(
            state.hand_winner().is_some(),
            "terminal state must have a winner"
        );
        let winner = state
            .hand_winner()
            .expect("terminal state must have a winner");
        let hand_value = state.hand_value();

        let new_score = match winner {
            0 => Score {
                zero: (score.zero + hand_value).min(MATCH_TARGET),
                one: score.one,
            },
            _ => Score {
                zero: score.zero,
                one: (score.one + hand_value).min(MATCH_TARGET),
            },
        };

        let p0_value = if new_score.zero >= MATCH_TARGET {
            1.0
        } else if new_score.one >= MATCH_TARGET {
            -1.0
        } else {
            // Continuation hand is dealt by the other player.
            2.0 * match_values.get(new_score.zero, new_score.one, 1 - dealer) - 1.0
        };

        return if traversing == 0 { p0_value } else { -p0_value };
    }

    let player = match state.engine.current_player() {
        Some(p) => p,
        None => return 0.0,
    };

    let actions = match state.abstract_legal_actions() {
        Ok(a) if !a.is_empty() => a,
        _ => return 0.0,
    };

    debug_assert!(
        state.current_info_set().is_some(),
        "non-terminal state with actions must have an info set"
    );
    let info_set = state
        .current_info_set()
        .expect("non-terminal state with actions must have an info set");
    let num_actions = actions.len();

    // Ensure info set exists in table
    table.get_or_insert(&info_set, &actions);

    // Get current strategy
    let strategy = {
        let data = table
            .data
            .get(&info_set.key())
            .expect("info_set_key just inserted via get_or_insert");
        data.current_strategy()
    };

    if player == traversing {
        // Traversing player: enumerate all actions, update regrets
        let mut action_values = vec![0.0f64; num_actions];
        let mut node_value = 0.0;

        for (i, &action) in actions.iter().enumerate() {
            let mut new_reach = reach_probs;
            new_reach[player as usize] *= strategy[i];

            let child_state = match state.apply_abstract_action(action) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let value = mccfr_external_sampling(
                &child_state,
                traversing,
                dealer,
                new_reach,
                iteration,
                score,
                match_values,
                table,
                rng,
            );
            action_values[i] = value;
            node_value += strategy[i] * value;
        }

        // Update regrets (CFR+ style: clamp to 0)
        let key = info_set.key();
        let data = table
            .data
            .get_mut(&key)
            .expect("info_set_key pre-inserted during tree build");
        for i in 0..num_actions {
            let regret = action_values[i] - node_value;
            data.cumulative_regret[i] = (data.cumulative_regret[i] + regret).max(0.0);
        }

        node_value
    } else {
        // Opponent: sample one action according to current strategy
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut sampled_idx = num_actions - 1;
        for (i, &prob) in strategy.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                sampled_idx = i;
                break;
            }
        }

        let action = actions[sampled_idx];
        let child_state = match state.apply_abstract_action(action) {
            Ok(s) => s,
            Err(_) => return 0.0,
        };

        // Update cumulative strategy for opponent
        let key = info_set.key();
        let data = table
            .data
            .get_mut(&key)
            .expect("info_set_key pre-inserted during tree build");
        for i in 0..num_actions {
            data.cumulative_strategy[i] += reach_probs[player as usize] * strategy[i];
        }

        let mut new_reach = reach_probs;
        new_reach[player as usize] *= strategy[sampled_idx];

        mccfr_external_sampling(
            &child_state,
            traversing,
            dealer,
            new_reach,
            iteration,
            score,
            match_values,
            table,
            rng,
        )
    }
}

fn ensure_seeded_mccfr_row(
    table: &mut StrategyTable,
    info_set: &InfoSet,
    actions: &[AbstractAction],
    seed: Option<&dyn PolicyLookup>,
    seed_regret_mass: f64,
) {
    let key = info_set.key();
    if table.data.contains_key(&key) {
        return;
    }
    let data = table.get_or_insert(info_set, actions);
    let Some(seed) = seed else {
        return;
    };
    let mut probabilities: ActionProbs = actions
        .iter()
        .map(|&action| {
            seed.action_probability(key, action, PolicyValueSource::Average)
                .unwrap_or(0.0)
                .max(0.0)
        })
        .collect();
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 {
        return;
    }
    for (regret, probability) in data
        .cumulative_regret
        .iter_mut()
        .zip(probabilities.iter_mut())
    {
        *probability /= total;
        *regret = *probability * seed_regret_mass;
    }
}

#[allow(clippy::too_many_arguments)]
fn mccfr_external_sampling_batched(
    state: &crate::game_tree::TraversalState,
    traversing: Player,
    dealer: Player,
    reach_probs: [f64; 2],
    score: &Score,
    match_values: &MatchValueTable,
    table: &mut StrategyTable,
    pending: &mut std::collections::HashMap<InfoSetKey, MccfrBatchUpdate>,
    seed: Option<&dyn PolicyLookup>,
    seed_regret_mass: f64,
    rng: &mut impl rand::Rng,
) -> f64 {
    if state.is_terminal() {
        let winner = state
            .hand_winner()
            .expect("terminal batched MCCFR state must have a winner");
        let hand_value = state.hand_value();
        let new_score = match winner {
            0 => Score {
                zero: (score.zero + hand_value).min(MATCH_TARGET),
                one: score.one,
            },
            _ => Score {
                zero: score.zero,
                one: (score.one + hand_value).min(MATCH_TARGET),
            },
        };
        let p0_value = if new_score.zero >= MATCH_TARGET {
            1.0
        } else if new_score.one >= MATCH_TARGET {
            -1.0
        } else {
            2.0 * match_values.get(new_score.zero, new_score.one, 1 - dealer) - 1.0
        };
        return if traversing == 0 { p0_value } else { -p0_value };
    }

    let Some(player) = state.engine.current_player() else {
        return 0.0;
    };
    let Ok(actions) = state.abstract_legal_actions() else {
        return 0.0;
    };
    if actions.is_empty() {
        return 0.0;
    }
    let info_set = state
        .current_info_set()
        .expect("non-terminal batched MCCFR node must have an info set");
    let key = info_set.key();
    ensure_seeded_mccfr_row(table, &info_set, &actions, seed, seed_regret_mass);
    let strategy = table
        .data
        .get(&key)
        .expect("batched MCCFR row just inserted")
        .current_strategy();

    if player == traversing {
        let mut action_values: ActionProbs = smallvec::smallvec![0.0; actions.len()];
        let mut node_value = 0.0;
        for (i, &action) in actions.iter().enumerate() {
            let Ok(child_state) = state.apply_abstract_action(action) else {
                continue;
            };
            let mut new_reach = reach_probs;
            new_reach[player as usize] *= strategy[i];
            let value = mccfr_external_sampling_batched(
                &child_state,
                traversing,
                dealer,
                new_reach,
                score,
                match_values,
                table,
                pending,
                seed,
                seed_regret_mass,
                rng,
            );
            action_values[i] = value;
            node_value += strategy[i] * value;
        }
        let update = pending
            .entry(key)
            .or_insert_with(|| MccfrBatchUpdate::zeros(actions.len()));
        for i in 0..actions.len() {
            update.regret[i] += action_values[i] - node_value;
        }
        node_value
    } else {
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut sampled_idx = actions.len() - 1;
        for (i, &probability) in strategy.iter().enumerate() {
            cumulative += probability;
            if r < cumulative {
                sampled_idx = i;
                break;
            }
        }
        let update = pending
            .entry(key)
            .or_insert_with(|| MccfrBatchUpdate::zeros(actions.len()));
        let weight = reach_probs[player as usize];
        for i in 0..actions.len() {
            update.strategy[i] += weight * strategy[i];
        }
        let Ok(child_state) = state.apply_abstract_action(actions[sampled_idx]) else {
            return 0.0;
        };
        let mut new_reach = reach_probs;
        new_reach[player as usize] *= strategy[sampled_idx];
        mccfr_external_sampling_batched(
            &child_state,
            traversing,
            dealer,
            new_reach,
            score,
            match_values,
            table,
            pending,
            seed,
            seed_regret_mass,
            rng,
        )
    }
}

/// Compute exploitability for a strategy table by building trees from deals.
/// Use `max_deals` to limit tree building for quick approximate measurement.
pub fn compute_exploitability_from_deals(
    score: &Score,
    tc: TurnupClass,
    table: &StrategyTable,
    match_values: &MatchValueTable,
    max_deals: Option<usize>,
) -> f64 {
    compute_exploitability_from_deals_with_dealer(score, tc, table, match_values, max_deals, None)
}

/// Deal-limited exploitability with an optional single-dealer tree, matching
/// the exact per-dealer decomposition used by production solves.
pub fn compute_exploitability_from_deals_with_dealer(
    score: &Score,
    tc: TurnupClass,
    table: &StrategyTable,
    match_values: &MatchValueTable,
    max_deals: Option<usize>,
    dealer_filter: Option<Player>,
) -> f64 {
    let mut deals = enumerate_deals(&tc);
    if let Some(limit) = max_deals {
        subsample_deals(&mut deals, limit);
    }
    compute_exploitability_on_deals_with_dealer(
        score,
        tc,
        table,
        match_values,
        &deals,
        dealer_filter,
    )
}

/// Exact exploitability on a caller-provided weighted deal set. This lets
/// sampling experiments keep evaluation deals disjoint from their training
/// subset instead of accidentally reporting in-sample convergence.
pub fn compute_exploitability_on_deals_with_dealer(
    score: &Score,
    tc: TurnupClass,
    table: &StrategyTable,
    match_values: &MatchValueTable,
    deals: &[crate::abstraction::AbstractDeal],
    dealer_filter: Option<Player>,
) -> f64 {
    let prebuilt = build_all_trees_with_dealer(score, tc, deals, dealer_filter)
        .expect("tree building failed: enumerate_deals produces valid deals");
    compute_exploitability(&prebuilt, table, score, match_values)
}

/// Reduce `deals` to at most `limit` entries for quick tests, then re-normalize
/// weights. Stride-subsamples rather than truncating: `enumerate_deals` iterates
/// player 0's hand outermost, so a prefix would give player 0 only one or two
/// distinct hands — useless for anything hand-distribution-dependent (e.g. the
/// mão-de-onze accept range). Striding keeps the subset spread over both
/// players' hand spaces.
pub fn subsample_deals(deals: &mut Vec<crate::abstraction::AbstractDeal>, limit: usize) {
    if limit > 0 && limit < deals.len() {
        let n = deals.len();
        *deals = (0..limit).map(|i| deals[i * n / limit].clone()).collect();
    }
    let total_weight: f64 = deals.iter().map(|d| d.weight).sum();
    if total_weight > 0.0 {
        for deal in deals.iter_mut() {
            deal.weight /= total_weight;
        }
    }
}

fn estimate_memory_bytes(table: &StrategyTable, prebuilt: &PrebuiltTrees) -> usize {
    let mut bytes = 0usize;

    // Strategy table
    for (_, data) in &table.data {
        let n = data.actions.len();
        bytes += 8 + 24 + 8 * n + 24 + 8 * n + 24 + 2 * n;
    }
    bytes += table.len() + 56;

    // Game trees: packed 12-byte nodes + 8-byte edges.
    for entry in &prebuilt.entries {
        bytes += entry.tree_dealer_0.nodes.len() * 12 + entry.tree_dealer_0.edges.len() * 8;
        bytes += entry.tree_dealer_1.nodes.len() * 12 + entry.tree_dealer_1.edges.len() * 8;
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbstractCard;
    use crate::storage::save_checkpoint;
    use rand::SeedableRng;
    use smallvec::smallvec;

    fn stream_warmstart_for_test(
        name: &str,
        infos: &InfoMeta,
        source: &StrategyTable,
        target_turnup: TurnupClass,
        same_band: bool,
        source_turnup: Option<TurnupClass>,
    ) -> (WarmStartTransferStats, DenseAccum) {
        let dir = std::env::temp_dir().join(format!(
            "truco_solver_stream_warmstart_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("source.ckpt.bin");
        let meta = CheckpointMeta {
            score: (0, 0),
            turnup_class: source_turnup.unwrap_or(target_turnup),
            algo: "SyncCFR+".into(),
            iteration: 10,
            num_info_sets: source.len(),
            dealer_filter: None,
        };
        save_checkpoint(&path, source, meta).unwrap();
        let mut stream = CheckpointStream::open(&path).unwrap();
        let mut dense = DenseAccum::zeros(infos);
        let stats = apply_warmstart_stream(
            infos,
            &mut dense,
            &mut stream,
            target_turnup,
            same_band,
            source_turnup,
        )
        .unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        (stats, dense)
    }

    #[test]
    fn sync_pending_regrets_overlay_player_local_slots_exactly() {
        let tc = TurnupClass {
            blocked_plain_level: 2,
        };
        let hand = smallvec![
            AbstractCard::Plain(1),
            AbstractCard::Plain(4),
            AbstractCard::Manilha(0),
        ];
        let p0_dealer = InfoSet::new(0, true, tc, hand.clone());
        let p1_dealer = InfoSet::new(1, true, tc, hand.clone());
        let p0_non_dealer = InfoSet::new(0, false, tc, hand);
        let infos = vec![
            (
                p0_dealer.key(),
                p0_dealer,
                vec![AbstractAction::Fold, AbstractAction::AcceptRaise].into_boxed_slice(),
            ),
            (
                p1_dealer.key(),
                p1_dealer,
                vec![
                    AbstractAction::Fold,
                    AbstractAction::AcceptRaise,
                    AbstractAction::Raise(3),
                ]
                .into_boxed_slice(),
            ),
            (
                p0_non_dealer.key(),
                p0_non_dealer,
                vec![
                    AbstractAction::Fold,
                    AbstractAction::AcceptRaise,
                    AbstractAction::Raise(3),
                    AbstractAction::Raise(6),
                ]
                .into_boxed_slice(),
            ),
        ];
        let mut dense = DenseAccum::zeros(&infos);
        dense.ensure_buffers(false);

        assert_eq!(dense.regret.len(), 9);
        assert_eq!(dense.pending_slots, [6, 3]);
        assert_eq!(dense.pending.len(), 6);
        assert_eq!(dense.pending_off, vec![0, 0, 2]);

        dense.pending[0..2].copy_from_slice(&[1.0, 2.0]);
        dense.pending[2..6].copy_from_slice(&[3.0, 4.0, 5.0, 6.0]);
        fold_pending_regrets(&mut dense, SweepMode::Sync, &infos, 0);
        assert_eq!(
            dense.regret,
            vec![1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(dense.pending.iter().all(|value| *value == 0.0));

        dense.pending[0..3].copy_from_slice(&[7.0, 8.0, 9.0]);
        fold_pending_regrets(&mut dense, SweepMode::Sync, &infos, 1);
        assert_eq!(
            dense.regret,
            vec![1.0, 2.0, 7.0, 8.0, 9.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(dense.pending.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_regret_pruning_revisits_both_player_sweeps_as_a_pair() {
        let config = RegretPruningConfig {
            warmup_iters: 4,
            threshold: 0.1,
            revisit_every_rounds: 3,
        };
        for iteration in 1..=4 {
            assert!(!config.prunes_on_iteration(iteration));
        }
        assert!(config.prunes_on_iteration(5));
        assert!(config.prunes_on_iteration(6));
        // Iterations 7 and 8 are the two player passes of the same full-width
        // revisit round. A modulo on raw iterations would revisit only one.
        assert!(!config.prunes_on_iteration(7));
        assert!(!config.prunes_on_iteration(8));
        assert!(config.prunes_on_iteration(9));

        let resumed = config.after_start_iter(100);
        assert!(!resumed.prunes_on_iteration(104));
        assert!(resumed.prunes_on_iteration(105));
    }

    #[test]
    fn test_regret_pruning_requires_zero_current_probability_and_negative_shadow() {
        let info = InfoSet::new(
            0,
            true,
            TurnupClass {
                blocked_plain_level: 2,
            },
            smallvec![
                AbstractCard::Plain(1),
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
            ],
        );
        let infos = vec![(
            info.key(),
            info,
            vec![AbstractAction::Fold, AbstractAction::AcceptRaise].into_boxed_slice(),
        )];
        let mut dense = DenseAccum::zeros(&infos);
        dense.enable_regret_pruning();
        dense.prune_regret[0] = -0.5;
        let config = RegretPruningConfig {
            warmup_iters: 2,
            threshold: 0.1,
            revisit_every_rounds: 10,
        };

        assert!(dense.should_prune_action(0, 0, 0.0, 3, Some(&config)));
        assert!(!dense.should_prune_action(0, 0, 0.1, 3, Some(&config)));
        assert!(!dense.should_prune_action(0, 1, 0.0, 3, Some(&config)));
        assert!(!dense.should_prune_action(0, 0, 0.0, 1, Some(&config)));
        assert!(!dense.should_prune_action(0, 0, 0.0, 1, None));
    }

    #[test]
    fn test_same_band_warmstart_copies_retained_action_regrets_and_resets_average() {
        let info = InfoSet::new(
            0,
            true,
            TurnupClass {
                blocked_plain_level: 2,
            },
            smallvec![
                AbstractCard::Plain(1),
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
            ],
        );
        let actions = vec![
            AbstractAction::Raise(3),
            AbstractAction::PlayFaceUp(AbstractCard::Plain(1)),
        ];
        let retained_actions = vec![actions[1]];
        let infos = vec![(
            info.key(),
            info.clone(),
            retained_actions.iter().copied().collect(),
        )];
        let mut source = StrategyTable::new();
        let source_data = source.get_or_insert(&info, &actions);
        source_data.cumulative_regret = vec![7.0, 2.0];
        source_data.cumulative_strategy = vec![90.0, 10.0];
        let mut target = StrategyTable::new();
        target.get_or_insert(&info, &retained_actions);

        let stats = apply_warmstart(&infos, &mut target, &source, true, None);
        let target_data = target.get(&info).unwrap();
        assert_eq!(stats.direct, 1);
        assert_eq!(stats.remapped, 0);
        assert_eq!(target_data.cumulative_regret, vec![2.0]);
        assert_eq!(target_data.cumulative_strategy, vec![0.0]);

        let mut dense = DenseAccum::zeros(&infos);
        let dense_stats = apply_warmstart_dense(&infos, &mut dense, &source, true, None);
        assert_eq!(dense_stats, stats);
        assert_eq!(dense.regret, vec![2.0]);
        assert_eq!(dense.strategy, vec![0.0]);

        let (stream_stats, streamed) =
            stream_warmstart_for_test("same_band", &infos, &source, info.turnup_class, true, None);
        assert_eq!(stream_stats, stats);
        assert_eq!(streamed.regret, dense.regret);
        assert_eq!(streamed.strategy, dense.strategy);
    }

    #[test]
    fn test_cross_turnup_warmstart_rekeys_and_preserves_trained_average() {
        let source_tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let target_tc = TurnupClass {
            blocked_plain_level: 1,
        };
        let source_info = InfoSet::new(
            0,
            true,
            source_tc,
            smallvec![
                AbstractCard::Plain(2),
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
            ],
        );
        let mut target_info = source_info.clone();
        target_info.turnup_class = target_tc;
        let actions = vec![
            AbstractAction::PlayFaceUp(AbstractCard::Plain(2)),
            AbstractAction::PlayFaceUp(AbstractCard::Manilha(0)),
        ];
        let infos = vec![(
            target_info.key(),
            target_info.clone(),
            actions.iter().copied().collect(),
        )];

        let mut source = StrategyTable::new();
        let source_data = source.get_or_insert(&source_info, &actions);
        source_data.cumulative_regret = vec![3.0, 11.0];
        source_data.cumulative_strategy = vec![2.0, 98.0];
        let mut target = StrategyTable::new();
        target.get_or_insert(&target_info, &actions);

        let stats = apply_warmstart(&infos, &mut target, &source, true, Some(source_tc));
        let target_data = target.get(&target_info).unwrap();
        assert_eq!(stats.direct, 1);
        assert_eq!(stats.remapped, 0);
        assert_eq!(target_data.cumulative_regret, vec![3.0, 11.0]);
        assert_eq!(target_data.cumulative_strategy, vec![2.0, 98.0]);

        let mut dense = DenseAccum::zeros(&infos);
        let dense_stats = apply_warmstart_dense(&infos, &mut dense, &source, true, Some(source_tc));
        assert_eq!(dense_stats, stats);
        assert_eq!(dense.regret, vec![3.0, 11.0]);
        assert_eq!(dense.strategy, vec![2.0, 98.0]);

        let (stream_stats, streamed) = stream_warmstart_for_test(
            "cross_turnup",
            &infos,
            &source,
            target_tc,
            true,
            Some(source_tc),
        );
        assert_eq!(stream_stats, stats);
        assert_eq!(streamed.regret, dense.regret);
        assert_eq!(streamed.strategy, dense.strategy);
    }

    #[test]
    fn test_cross_score_profile_transfer_preserves_trained_average() {
        let tc = TurnupClass {
            blocked_plain_level: 3,
        };
        let info = InfoSet::new(
            1,
            false,
            tc,
            smallvec![
                AbstractCard::Plain(0),
                AbstractCard::Plain(5),
                AbstractCard::Manilha(1),
            ],
        );
        let actions = vec![
            AbstractAction::PlayFaceUp(AbstractCard::Plain(0)),
            AbstractAction::PlayFaceUp(AbstractCard::Manilha(1)),
        ];
        let infos = vec![(info.key(), info.clone(), actions.iter().copied().collect())];
        let mut source = StrategyTable::new();
        let source_data = source.get_or_insert(&info, &actions);
        source_data.cumulative_regret = vec![13.0, 1.0];
        source_data.cumulative_strategy = vec![4.0, 96.0];
        let mut dense = DenseAccum::zeros(&infos);

        // Score is deliberately absent from InfoSet identity. Passing the
        // same TC as an explicit profile donor distinguishes this operation
        // from ordinary same-band regret-only warm-start semantics.
        let stats = apply_warmstart_dense(&infos, &mut dense, &source, true, Some(tc));
        assert_eq!(stats.direct, 1);
        assert_eq!(dense.regret, vec![13.0, 1.0]);
        assert_eq!(dense.strategy, vec![4.0, 96.0]);

        let meta = CheckpointMeta {
            score: (11, 10),
            turnup_class: tc,
            algo: "SyncCFR+".into(),
            iteration: 10,
            num_info_sets: 1,
            dealer_filter: Some(0),
        };
        let target = Score { zero: 11, one: 9 };
        let (source_score, same_band) =
            checkpoint_warmstart_relation(&meta, &target, tc, Some(0), true);
        assert_eq!(source_score, Score { zero: 11, one: 10 });
        assert!(same_band);
    }

    #[test]
    fn test_related_mao_warmstart_keeps_historical_history_remap() {
        let source_info = InfoSet::new(
            1,
            false,
            TurnupClass {
                blocked_plain_level: 2,
            },
            smallvec![
                AbstractCard::Plain(1),
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
            ],
        );
        let mut target_info = source_info.clone();
        target_info.history.push(AbstractAction::AcceptEleven);
        let actions = vec![AbstractAction::PlayFaceUp(AbstractCard::Plain(1))];
        let infos = vec![(
            target_info.key(),
            target_info.clone(),
            actions.iter().copied().collect(),
        )];
        let mut source = StrategyTable::new();
        let source_data = source.get_or_insert(&source_info, &actions);
        source_data.cumulative_regret = vec![3.0];
        source_data.cumulative_strategy = vec![11.0];
        let mut target = StrategyTable::new();
        target.get_or_insert(&target_info, &actions);

        let stats = apply_warmstart(&infos, &mut target, &source, false, None);
        let target_data = target.get(&target_info).unwrap();
        assert_eq!(stats.direct, 0);
        assert_eq!(stats.remapped, 1);
        assert_eq!(target_data.cumulative_regret, vec![3.0]);
        assert_eq!(target_data.cumulative_strategy, vec![11.0]);

        let mut dense = DenseAccum::zeros(&infos);
        let dense_stats = apply_warmstart_dense(&infos, &mut dense, &source, false, None);
        assert_eq!(dense_stats, stats);
        assert_eq!(dense.regret, vec![3.0]);
        assert_eq!(dense.strategy, vec![11.0]);

        let (stream_stats, streamed) = stream_warmstart_for_test(
            "mao_remap",
            &infos,
            &source,
            target_info.turnup_class,
            false,
            None,
        );
        assert_eq!(stream_stats, stats);
        assert_eq!(streamed.regret, dense.regret);
        assert_eq!(streamed.strategy, dense.strategy);
    }

    #[test]
    fn test_accept_policy_membership() {
        let mut policy = AcceptPolicy::default();
        let h_accept: AbstractHand = smallvec![
            AbstractCard::Manilha(3),
            AbstractCard::Plain(7),
            AbstractCard::Plain(2),
        ];
        let h_fold: AbstractHand = smallvec![
            AbstractCard::Plain(0),
            AbstractCard::Plain(1),
            AbstractCard::Plain(2),
        ];
        // Accept only h_accept, only when dealer == 0.
        policy.accept[0].insert(h_accept.clone());

        assert!(policy.accepts(0, &h_accept));
        assert!(!policy.accepts(1, &h_accept)); // dealer 1 set is empty -> fold
        assert!(!policy.accepts(0, &h_fold)); // not a member -> fold
        assert!(!policy.accepts(1, &h_fold));
    }

    #[test]
    fn test_eleven_decision_children_detection() {
        let edge = |a: AbstractAction, child: NodeId| PackedEdge::new(a.to_u8(), child);
        // Exactly {AcceptEleven, FoldEleven}: detected, children resolved by action.
        let accept_id: NodeId = 11;
        let fold_id: NodeId = 22;
        let edges = vec![
            edge(AbstractAction::AcceptEleven, accept_id),
            edge(AbstractAction::FoldEleven, fold_id),
        ];
        assert_eq!(eleven_decision_children(&edges), Some((accept_id, fold_id)));

        // Robust to reversed order.
        let reversed = vec![
            edge(AbstractAction::FoldEleven, fold_id),
            edge(AbstractAction::AcceptEleven, accept_id),
        ];
        assert_eq!(
            eleven_decision_children(&reversed),
            Some((accept_id, fold_id))
        );

        // A card-play / raise node is NOT an eleven-decision node.
        let play = vec![
            edge(AbstractAction::PlayFaceUp(AbstractCard::Plain(0)), 1),
            edge(AbstractAction::Raise(3), 2),
        ];
        assert_eq!(eleven_decision_children(&play), None);

        // Wrong arity / missing AcceptEleven -> None.
        let single = vec![edge(AbstractAction::AcceptEleven, accept_id)];
        assert_eq!(eleven_decision_children(&single), None);
    }

    /// Quick test: solve with limited deals to validate infrastructure
    #[test]
    fn test_solve_11x11_runs() {
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();
        let (table, stats) = solve_with_limit(score, tc, 2, &mv, Some(200), None);
        assert!(!table.is_empty(), "strategy table is empty after solving");
        assert!(stats.num_info_sets > 0);
        assert!(stats.exploitability.is_some());
        eprintln!("{}", stats);
    }

    #[test]
    fn test_exploitability_non_negative() {
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();
        let (_, stats) = solve_with_limit(score, tc, 5, &mv, Some(200), None);
        let expl = stats.exploitability.unwrap();
        assert!(
            expl >= -1e-10,
            "exploitability should be non-negative, got {}",
            expl
        );
    }

    #[test]
    fn test_exploitability_decreases_with_iterations() {
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();

        let (_, stats_10) = solve_with_limit(score.clone(), tc, 10, &mv, Some(200), None);
        let (_, stats_50) = solve_with_limit(score, tc, 50, &mv, Some(200), None);

        let expl_10 = stats_10.exploitability.unwrap();
        let expl_50 = stats_50.exploitability.unwrap();
        eprintln!(
            "Exploitability: 10 iters = {:.6}, 50 iters = {:.6}",
            expl_10, expl_50
        );

        // After more iterations, exploitability should generally be lower
        // (not strictly monotone due to averaging, but should trend down)
        assert!(
            expl_50 < expl_10 + 0.01,
            "50 iterations should not be much worse than 10: {} vs {}",
            expl_50,
            expl_10
        );
    }

    #[test]
    fn test_best_response_gaps_plumbing() {
        // Structural correctness of best_response_gaps_from_action_probs
        // (RESEARCH_NARRATIVE.md 2026-07-11 "Per-infoset best-response gap"):
        // no full-deal solve needed, this just checks the refactored
        // best_response_resolve_for_profile wiring is sound, not convergence
        // quality (that's measured against real checkpoints separately).
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();
        let (table, _) = solve_with_limit(score.clone(), tc, 30, &mv, Some(20), None);

        // Rebuild the exact same trees the solve used (subsample_deals is
        // deterministic), so table_idx lines up with a fresh PrebuiltTrees.
        let mut deals = enumerate_deals(&tc);
        subsample_deals(&mut deals, 20);
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, None)
            .expect("tree building failed for a valid deal set");

        let dense = DenseAccum::clone_from_table(&prebuilt.info_sets, &table);
        let strategies: Vec<crate::strategy::ActionProbs> = (0..prebuilt.info_sets.len())
            .map(|i| dense.average_strategy(i))
            .collect();

        let full0 = best_response_full_from_action_probs(&prebuilt, &strategies, &score, &mv, 0);
        let gaps0 = &full0.per_info_set;
        let gaps1 = best_response_gaps_from_action_probs(&prebuilt, &strategies, &score, &mv, 1);

        assert!(
            !gaps0.is_empty(),
            "player 0 should own at least one info set"
        );
        assert!(
            !gaps1.is_empty(),
            "player 1 should own at least one info set"
        );

        let idx0: std::collections::HashSet<u32> = gaps0.iter().map(|r| r.table_idx).collect();
        let idx1: std::collections::HashSet<u32> = gaps1.iter().map(|r| r.table_idx).collect();
        assert!(
            idx0.is_disjoint(&idx1),
            "an info set cannot belong to both players"
        );
        assert_eq!(
            idx0.len(),
            gaps0.len(),
            "table_idx must be unique within one player's results"
        );
        assert_eq!(
            idx1.len(),
            gaps1.len(),
            "table_idx must be unique within one player's results"
        );
        assert_eq!(full0.chosen_actions.len(), prebuilt.info_sets.len());
        for (idx, (_, info, actions)) in prebuilt.info_sets.iter().enumerate() {
            let chosen = full0.chosen_actions[idx];
            if info.player == 0 {
                assert!((chosen as usize) < actions.len());
            } else {
                assert_eq!(chosen, u8::MAX);
            }
        }

        for r in gaps0.iter().chain(gaps1.iter()) {
            assert!(
                r.weight >= 0.0,
                "counterfactual weight cannot be negative: idx {} weight {}",
                r.table_idx,
                r.weight
            );
            if r.weight > 0.0 {
                assert!(
                    r.br_value.is_finite(),
                    "idx {} has positive weight but non-finite br_value {}",
                    r.table_idx,
                    r.br_value
                );
                assert!(
                    (-1.0 - 1e-9..=1.0 + 1e-9).contains(&r.br_value),
                    "idx {} br_value {} outside the [-1, 1] match-value range",
                    r.table_idx,
                    r.br_value
                );
            } else {
                assert!(
                    r.br_value.is_nan(),
                    "idx {} has zero weight but non-NaN br_value {}",
                    r.table_idx,
                    r.br_value
                );
            }
        }

        // Same backward-induction pass as the trusted aggregate — spot-check
        // the two entrypoints didn't diverge under the refactor.
        let total0 = best_response_value_from_action_probs(&prebuilt, &strategies, &score, &mv, 0);
        let total1 = best_response_value_from_action_probs(&prebuilt, &strategies, &score, &mv, 1);
        assert!(total0.is_finite() && total1.is_finite());
    }

    #[test]
    fn test_solve_11x10_tiny_converges() {
        // Regression test for the mão-de-onze wall. The decider's accept/fold
        // node sits at the empty history; before `InfoSet` encoded position it
        // had the same key in both dealer trees, forcing one position-averaged
        // accept policy — brute CFR walled at ~0.33 exploitability at full
        // 11x10 no matter the algorithm. On this exact tiny config the
        // pre-position code walls at 0.0965 (still ~0.0965 after 300 iters);
        // with the accept learned per dealer, CFR+ passes 0.0102 by iter 51
        // and reaches 0.0003 by iter 300.
        let score = Score { zero: 11, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        // Dealer-exact continuation: folding the mão de onze goes to 11x11 with
        // the dealer flipped, worth 0.556/0.444 (not 0.5) to player 0.
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (table, stats) = solve_with_limit(score, tc, 120, &mv, Some(300), None);
        let expl = stats.exploitability.unwrap();
        eprintln!("tiny 11x10 exploitability after 120 iters: {:.6}", expl);
        assert!(
            expl < 0.02,
            "tiny 11x10 should converge well below the ~0.0965 shared-accept wall, got {}",
            expl
        );

        // The structural fix itself: the decider's accept/fold node (player 0,
        // empty history) exists as TWO info sets — one per position — so the
        // accept policy is per-dealer, not position-averaged.
        let mut accept_hands: [std::collections::HashSet<AbstractHand>; 2] = Default::default();
        for (key, info) in &table.info_sets {
            if info.player == 0 && info.history.is_empty() {
                let data = &table.data[key];
                assert!(data.actions.contains(&AbstractAction::AcceptEleven));
                accept_hands[info.is_dealer as usize].insert(info.starting_hand.clone());
            }
        }
        assert_eq!(
            accept_hands[0], accept_hands[1],
            "every decider hand should have an accept node in both positions"
        );
        assert!(!accept_hands[0].is_empty());
    }

    #[test]
    fn test_dealer_filtered_solve_matches_joint() {
        // The two dealer games share no info sets, so the dealer-0 slice of a
        // joint solve runs the exact same update sequence as a dealer-0-only
        // solve — the per-dealer game value must match to float precision.
        let score = Score { zero: 11, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (_, joint) = solve_with_limit(score.clone(), tc, 10, &mv, Some(150), None);
        let (_, only_d0) = solve_with_limit(score, tc, 10, &mv, Some(150), Some(0));

        let (joint_d0, _) = joint.game_value_per_dealer.unwrap();
        let (filtered_d0, filtered_d1) = only_d0.game_value_per_dealer.unwrap();
        assert!(
            (joint_d0 - filtered_d0).abs() < 1e-12,
            "filtered d0 value {} != joint d0 value {}",
            filtered_d0,
            joint_d0
        );
        // The excluded dealer's tree was never built; its value accumulates 0.
        assert_eq!(filtered_d1, 0.0);
        assert!(only_d0.num_info_sets < joint.num_info_sets);
    }

    #[test]
    fn test_parallel_sync_sweep_matches_serial() {
        // The parallel sweep must be the same algorithm as serial SyncCFR+ —
        // only floating-point addition order may differ (atomic accumulation),
        // so trajectories agree to tight tolerance.
        //
        // Under `accum-f32` the two paths also differ in WHERE they narrow to
        // f32: serial narrows the opponent's strategy sums per node visit,
        // while the parallel path buffers them in f64 `LocalAccum`s and
        // narrows once per iteration. The trajectories are then only
        // f32-rounding-close, not identical; the exact-BR A/B against an f64
        // control (plan 84 Phase 1) is the accuracy gate for that mode.
        let score = Score { zero: 11, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (_, serial) = solve_with_limit_algo(
            score.clone(),
            tc,
            40,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            10,
            1,
            None,
        );
        let (_, par) = solve_with_limit_algo(
            score,
            tc,
            40,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            10,
            4,
            None,
        );
        for ((it_a, a), (it_b, b)) in serial
            .exploitability_history
            .iter()
            .zip(par.exploitability_history.iter())
        {
            assert_eq!(it_a, it_b);
            #[cfg(feature = "accum-f32")]
            let tol = 5e-3;
            #[cfg(not(feature = "accum-f32"))]
            let tol = 1e-6;
            assert!(
                (a - b).abs() < tol,
                "serial vs parallel diverged at iter {}: {} vs {}",
                it_a,
                a,
                b
            );
        }
    }

    #[test]
    fn test_tremble_schedule_eps_at() {
        let ts = TrembleSchedule {
            eps_start: 0.05,
            eps_end: 0.01,
        };
        // At the start of the window: eps_start exactly.
        assert_eq!(ts.eps_at(100, 100, 200), 0.05);
        // At the end of the window: eps_end (within float tolerance).
        assert!((ts.eps_at(200, 100, 200) - 0.01).abs() < 1e-12);
        // Midpoint: linear interpolation.
        assert!((ts.eps_at(150, 100, 200) - 0.03).abs() < 1e-12);
        // Past the horizon: clamped to eps_end, not extrapolated further down.
        assert!((ts.eps_at(250, 100, 200) - 0.01).abs() < 1e-12);
        // Before the window (shouldn't happen in practice, but must not panic
        // or invert): clamped to eps_start.
        assert_eq!(ts.eps_at(50, 100, 200), 0.05);
        // Unknown horizon (max_iters unbounded) falls back to a constant eps_start.
        assert_eq!(ts.eps_at(100_000, 0, u64::MAX), 0.05);
        // Degenerate window (max_iters <= start_iter) also falls back.
        assert_eq!(ts.eps_at(100, 100, 100), 0.05);
    }

    #[test]
    fn test_tremble_strategy_floor() {
        // eps <= 0.0 is an exact no-op (bit-identical), never a floor.
        let s: ActionProbs = smallvec![1.0, 0.0, 0.0];
        assert_eq!(tremble_strategy(s.clone(), 0.0), s);
        assert_eq!(tremble_strategy(s.clone(), -1.0), s);

        // eps > 0.0 floors every action at eps/|A| and renormalizes the mass
        // that would otherwise concentrate on the dominant action.
        let floored = tremble_strategy(s, 0.2);
        let n = 3.0;
        let floor = 0.2 / n;
        assert!((floored[1] - floor).abs() < 1e-12, "{:?}", floored);
        assert!((floored[2] - floor).abs() < 1e-12, "{:?}", floored);
        assert!((floored[0] - (floor + 0.8)).abs() < 1e-12, "{:?}", floored);
        let sum: f64 = floored.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "trembled strategy must still sum to 1: {:?}",
            floored
        );
    }

    #[test]
    fn test_tremble_eps_zero_matches_no_tremble() {
        // A tremble schedule that is constantly 0.0 must reproduce the exact
        // (bit-identical) trajectory of no tremble at all -- the "default off,
        // existing workflows untouched" requirement.
        let score = Score { zero: 11, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        let (_, no_tremble) = solve_with_limit_algo(
            score.clone(),
            tc,
            25,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            25,
            1,
            None,
        );
        let (_, zero_tremble) = solve_with_limit_algo(
            score,
            tc,
            25,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            25,
            1,
            Some(TrembleSchedule {
                eps_start: 0.0,
                eps_end: 0.0,
            }),
        );
        assert_eq!(
            no_tremble.exploitability, zero_tremble.exploitability,
            "eps=0.0 tremble must be bit-identical to no tremble at all"
        );
    }

    #[test]
    fn test_tremble_floors_every_visited_action_reach() {
        // Trembling guarantees sigma'(a) = eps/|A| + (1-eps)*sigma(a) >= eps/|A|
        // > 0 for EVERY action on EVERY visit, so as soon as an info set
        // accumulates any average-strategy mass at all, every one of its
        // actions must also carry strictly positive mass. This is the
        // "own-reach floor" acceptance criterion from QUESTIONS.md Q3: the
        // Study lab's trainedness flag (own-reach < 1e-3) relies on no
        // exported action probability collapsing to exactly zero.
        let score = Score { zero: 11, one: 10 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.5564);
        mv.set(11, 11, 1, 0.4437);

        // Baseline: no tremble. RM+ commonly drives a regret-dominated action
        // to exact 0 once visited -- confirm that actually happens on this
        // config, so the contrast below is not vacuous.
        let (baseline, _) = solve_with_limit_algo(
            score.clone(),
            tc,
            30,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            1,
            None,
        );
        let baseline_has_exact_zero = baseline.data.values().any(|d| {
            let total: f64 = d.cumulative_strategy.iter().sum();
            total > 0.0 && d.cumulative_strategy.contains(&0.0)
        });
        assert!(
            baseline_has_exact_zero,
            "expected at least one visited info set with an exact-zero action \
             WITHOUT trembling on this config, otherwise the contrast below is vacuous"
        );

        // With a constant tremble, no visited info set may have an exact-zero
        // action in its accumulated average strategy.
        let (trembled, _) = solve_with_limit_algo(
            score,
            tc,
            30,
            &mv,
            Some(300),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            1,
            Some(TrembleSchedule {
                eps_start: 0.1,
                eps_end: 0.1,
            }),
        );
        for (key, data) in &trembled.data {
            let total: f64 = data.cumulative_strategy.iter().sum();
            if total > 0.0 {
                for (i, &s) in data.cumulative_strategy.iter().enumerate() {
                    assert!(
                        s > 0.0,
                        "info set {:?} action {} has exact-zero avg-strategy mass \
                         despite trembling (row total={})",
                        key,
                        i,
                        total
                    );
                }
            }
        }
    }

    #[test]
    fn test_mccfr_seed_initializes_current_strategy_without_stale_average() {
        let info = InfoSet::new(
            0,
            true,
            TurnupClass {
                blocked_plain_level: 0,
            },
            smallvec![
                AbstractCard::Plain(1),
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
            ],
        );
        let actions = vec![AbstractAction::Fold, AbstractAction::AcceptRaise];
        let mut source = StrategyTable::new();
        let source_data = source.get_or_insert(&info, &actions);
        source_data.cumulative_strategy = vec![9.0, 1.0];
        let mut target = StrategyTable::new();

        ensure_seeded_mccfr_row(&mut target, &info, &actions, Some(&source), 10.0);
        let data = target.data.get(&info.key()).unwrap();
        assert_eq!(data.cumulative_regret, vec![9.0, 1.0]);
        assert_eq!(data.current_strategy().as_slice(), &[0.9, 0.1]);
        assert_eq!(data.cumulative_strategy, vec![0.0, 0.0]);
    }

    #[test]
    fn test_mccfr_minibatch_chunk_updates_persistent_sparse_table() {
        let score = Score { zero: 8, one: 8 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();
        let mut deals = enumerate_deals(&tc);
        subsample_deals(&mut deals, 100);
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let mut table = StrategyTable::new();

        let first = run_mccfr_minibatch_chunk(
            score.clone(),
            tc,
            10,
            4,
            0,
            &mv,
            &mut rng,
            &deals,
            &mut table,
            None,
            0.0,
            Some(0),
        );
        assert_eq!(first.samples, 40);
        assert_eq!(first.info_sets_before, 0);
        assert!(first.info_sets_after > 0);
        let after_first = table.len();

        let second = run_mccfr_minibatch_chunk(
            score,
            tc,
            10,
            4,
            10,
            &mv,
            &mut rng,
            &deals,
            &mut table,
            None,
            0.0,
            Some(0),
        );
        assert_eq!(second.info_sets_before, after_first);
        assert!(second.info_sets_after >= second.info_sets_before);
        for data in table.data.values() {
            let strategy = data.current_strategy();
            assert!((strategy.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_mccfr_basic() {
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let mv = MatchValueTable::new();
        let mut rng = rand::rng();

        // MCCFR doesn't build trees upfront, so iteration is fast
        let (table, stats) = solve_mccfr(score.clone(), tc, 5000, &mv, &mut rng);
        assert!(!table.is_empty());
        assert!(stats.num_info_sets > 0);
        eprintln!("{}", stats);

        // Compute exploitability on a subset of deals (fast)
        let expl = compute_exploitability_from_deals(&score, tc, &table, &mv, Some(200));
        eprintln!("MCCFR exploitability (200 deals): {:.6}", expl);
        assert!(expl.is_finite());
    }

    #[test]
    #[ignore] // Slow: runs many iterations. Use `cargo test -- --ignored`
    fn test_solve_produces_valid_strategies() {
        let _ = env_logger::try_init();
        let score = Score { zero: 11, one: 11 };
        let tc = TurnupClass {
            blocked_plain_level: 4,
        };
        let mv = MatchValueTable::new();
        let (table, stats) = solve(score, tc, 100, &mv);

        eprintln!("{}", stats);

        for (_, data) in &table.data {
            let avg = data.average_strategy();
            let sum: f64 = avg.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "strategy probabilities sum to {} (expected 1.0)",
                sum
            );
            for &p in &avg {
                assert!(p >= 0.0, "negative probability in average strategy");
            }
        }
    }
}
