// Timing probe for the solver hot paths (ignored by default):
//   cargo test --release -p truco-solver --test perf_probe -- --ignored --nocapture
// Reference numbers (M-series laptop, 3000 strided deals, 11x10 TC0):
//   2026-07-02 pre-optimization:  build 2.87s, 0.211s/iter
//   2026-07-02 post (single-pass build + SmallVec strategies):
//                                 build 1.52s, 0.182s/iter
//   2026-07-03 post (dense table_idx accumulators, no per-visit hashing):
//                                 build 1.48s, 0.037s/iter
//   2026-07-03 post (packed 12B-node tree arena, ~4.5x less tree RAM):
//                                 build 1.57s, 0.041s/iter
//   2026-07-04 post (engine Arc<str> card ids + SmallVec hands/rounds --
//                    Engine::clone near-allocation-free):
//                                 build 0.89s, 0.021s/iter
//   Bench VM (n2-standard-16, full 10x10 TC0 d0, 352M nodes): peak RSS
//   11.25 GB (was ~60 GB); sync jobs=16 -> 90 iters to 0.0098 in 4352s.
//   Shared-atomic --jobs scaled only ~1.2x (false sharing); thread-local
//   rewrite scaling measured on the VM.
use std::time::Instant;
use truco_engine::Score;
use truco_solver::abstraction::{enumerate_deals, TurnupClass};
use truco_solver::cfr::{solve_with_limit_algo, CfrAlgorithm};
use truco_solver::game_tree::build_all_trees;
use truco_solver::match_value::MatchValueTable;

#[test]
#[ignore]
fn perf_parallel_scaling() {
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let score = Score { zero: 11, one: 10 };
    let mut mv = MatchValueTable::new();
    mv.set(11, 11, 0, 0.5564);
    mv.set(11, 11, 1, 0.4437);
    for jobs in [1usize, 4, 8] {
        let t = Instant::now();
        let (_, stats) = solve_with_limit_algo(
            score.clone(),
            tc,
            30,
            &mv,
            Some(3000),
            None,
            CfrAlgorithm::SyncCfrPlus,
            30,
            jobs,
            None,
        );
        eprintln!(
            "sync jobs={jobs}: {:.3}s/iter (total {:.2}s, final expl {:.6})",
            stats.per_iteration_secs,
            t.elapsed().as_secs_f64(),
            stats.exploitability.unwrap()
        );
    }
}

#[test]
#[ignore]
fn perf_baseline() {
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let score = Score { zero: 11, one: 10 };
    let mut deals = enumerate_deals(&tc);
    let n = deals.len();
    deals = (0..3000).map(|i| deals[i * n / 3000].clone()).collect();

    let t0 = Instant::now();
    let prebuilt = build_all_trees(&score, tc, &deals).unwrap();
    eprintln!(
        "build 3000 deals: {:.2}s ({} info sets)",
        t0.elapsed().as_secs_f64(),
        prebuilt.info_sets.len()
    );
    drop(prebuilt);

    let mut mv = MatchValueTable::new();
    mv.set(11, 11, 0, 0.5564);
    mv.set(11, 11, 1, 0.4437);
    let t1 = Instant::now();
    let (_, stats) = solve_with_limit_algo(
        score,
        tc,
        30,
        &mv,
        Some(3000),
        None,
        CfrAlgorithm::CfrPlus,
        30,
        1,
        None,
    );
    eprintln!(
        "solve 30 iters: {:.2}s total, {:.3}s/iter (build {:.2}s)",
        t1.elapsed().as_secs_f64(),
        stats.per_iteration_secs,
        stats.build_tree_secs
    );
}
