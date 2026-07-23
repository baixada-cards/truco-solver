//! Local Phase-4 validation gate (plan 84): from-scratch trunk-CFR at 8x8,
//! tc0, dealer 0, on a strided deal subset, compared against a monolithic
//! SyncCFR+ solve of the SAME subset under the SAME (synthetic-but-complete)
//! match values. `#[ignore]`d — it is a ~9-minute gate at the default 300
//! deals, not a CI unit test (the fast self-consistency check lives in
//! `resolve.rs`). Run with:
//!
//! ```text
//! PROBE_DEALS=300 PROBE_BASE=300 \
//!   cargo test -p truco-solver --test trunk_probe -- --ignored --nocapture
//! ```

use truco_engine::Score;
use truco_solver::abstraction::{enumerate_deals, TurnupClass};
use truco_solver::cfr::{self, CfrAlgorithm};
use truco_solver::game_tree::{build_all_trees_with_dealer, TreeRules};
use truco_solver::match_value::MatchValueTable;
use truco_solver::resolve::{trunk_solve, TrunkConfig};
use truco_solver::strategy::ActionProbs;
use truco_solver::subgame;

/// Complete synthetic 8x8 match-value table: every reachable successor score
/// gets a fixed, deterministic win prob so the monolithic solve and the trunk
/// solve certify the SAME game.
fn complete_mv_8x8() -> MatchValueTable {
    let mut mv = MatchValueTable::new();
    for zero in 0..=20u8 {
        for one in 0..=20u8 {
            if zero >= truco_engine::MATCH_TARGET || one >= truco_engine::MATCH_TARGET {
                continue;
            }
            let v = (0.5 + 0.02 * (zero as f64 - one as f64)).clamp(0.05, 0.95);
            for dealer in 0..2 {
                mv.set(zero, one, dealer, v);
            }
        }
    }
    mv
}

#[test]
#[ignore]
fn trunk_probe_8x8() {
    let score = Score { zero: 8, one: 8 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let take = std::env::var("PROBE_DEALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300usize);
    let base_iters: u64 = std::env::var("PROBE_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let mut deals = enumerate_deals(&tc);
    cfr::subsample_deals(&mut deals, take);
    let mv = complete_mv_8x8();
    let built = build_all_trees_with_dealer(&score, tc, &deals, Some(0)).unwrap();

    // Monolithic baseline on the same subset.
    let (table, stats) = cfr::solve_with_limit_algo(
        score.clone(),
        tc,
        base_iters,
        &mv,
        Some(take),
        Some(0),
        CfrAlgorithm::SyncCfrPlus,
        10,
        1,
        None,
    );
    let mono: Vec<ActionProbs> = built
        .info_sets
        .iter()
        .map(|(key, _, actions)| {
            table
                .data
                .get(key)
                .map(|d| d.average_strategy())
                .unwrap_or_else(|| truco_solver::strategy::uniform_probs(actions.len()))
        })
        .collect();
    let mono_eps = cfr::compute_exploitability_from_action_probs(&built, &mono, &score, &mv);
    let (m0, m1) = cfr::game_value_per_dealer_from_action_probs(&built, &mono, &score, &mv);
    let mono_gv = (m0 + m1) / 2.0;
    eprintln!(
        "MONO deals={take} iters={} eps={:.6} gv={:.6}",
        stats.iterations, mono_eps, mono_gv
    );

    let subgames =
        subgame::collect_boundary(&score, tc, &deals, Some(0), TreeRules::Current).unwrap();

    for (rounds, t, r) in [(30usize, 3u64, 3u64), (90, 1, 1)] {
        let cfg = TrunkConfig {
            rounds,
            trunk_iters: t,
            subgame_iters: r,
            final_iters: 120,
            baseline_iters: 90,
        };
        let (_c, rep) = trunk_solve(&built, &score, tc, &mv, &subgames, cfg);
        eprintln!(
            "TRUNK ({rounds},{t},{r}) eps={:.6} gv={:.6} gv_delta={:.6} mult={:.3} \
             visits(trunk={} subgame={} final={} total={})",
            rep.composed_eps,
            rep.game_value,
            (rep.game_value - mono_gv).abs(),
            rep.multiplier,
            rep.trunk_visits,
            rep.subgame_visits,
            rep.final_visits,
            rep.total_visits,
        );
        // Gate: composed certificate comfortably under the production target.
        assert!(
            rep.composed_eps <= 0.015,
            "schedule ({rounds},{t},{r}) composed eps {} exceeds 0.015",
            rep.composed_eps
        );
    }
}
