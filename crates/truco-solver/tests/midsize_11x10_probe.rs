// Mid-density 11x10 convergence probes (ignored by default — run explicitly:
//   cargo test --release -p truco-solver --test midsize_11x10_probe -- --ignored --nocapture
// ). The 300-deal tiny config converges to ~0 because each decider hand meets
// ~1 opponent hand — no real screening coupling. These probes scale deal
// density to reproduce (and then attack) the full-scale accept wall locally.
use truco_engine::Score;
use truco_solver::abstraction::TurnupClass;
use truco_solver::cfr::{solve_with_limit_algo, CfrAlgorithm};
use truco_solver::match_value::MatchValueTable;

fn mv_11x10() -> MatchValueTable {
    let mut mv = MatchValueTable::new();
    mv.set(11, 11, 0, 0.5564);
    mv.set(11, 11, 1, 0.4437);
    mv
}

fn run(label: &str, deals: usize, iters: u64, algo: CfrAlgorithm) {
    let score = Score { zero: 11, one: 10 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let (_, stats) = solve_with_limit_algo(
        score,
        tc,
        iters,
        &mv_11x10(),
        Some(deals),
        None,
        algo,
        10,
        1,
        None,
    );
    eprintln!("== {} (deals={}, iters={})", label, deals, iters);
    for (it, e) in &stats.exploitability_history {
        eprintln!("  iter {:>4}: {:.6}", it, e);
    }
}

#[test]
#[ignore]
fn probe_cfrplus_density() {
    for deals in [1000, 3000] {
        run("CFR+", deals, 300, CfrAlgorithm::CfrPlus);
    }
}

#[test]
#[ignore]
fn probe_cfrplus_1000_long() {
    // Floor vs slow-convergence discriminator: 1000-deal config walls at ~0.16
    // by iter 90. If a true floor, expl at iter 3000 stays ~0.15 (=> bug —
    // CFR must converge in a finite zero-sum game). If it keeps decaying,
    // it's an algorithmic speed problem.
    run("CFR+ long", 1000, 3000, CfrAlgorithm::CfrPlus);
}

#[test]
#[ignore]
fn probe_sync_1000() {
    // Synchronous sweep (strategy frozen per iteration): tests whether the
    // asynchronous mid-sweep strategy drift is the cause of the 0.153 floor
    // (CFR+ 1000-deal control: 0.1596 @ 300 iters, 0.1533 @ 3000).
    run("SyncCFR+", 1000, 300, CfrAlgorithm::SyncCfrPlus);
}

#[test]
#[ignore]
fn probe_pcfr_1000() {
    run("PCFR+", 1000, 300, CfrAlgorithm::PcfrPlus);
}

#[test]
#[ignore]
fn probe_sync_vs_async_3000() {
    // Decides the parallel-sweep question: sync sweeps (strategy frozen per
    // iteration) are read-only on regrets during traversal, so deals can be
    // fanned across threads — 16 cores sharing one job's RAM cuts RAM-hours
    // ~16x. Worth it iff sync's iterations-to-target stay within ~2x of async.
    run("async CFR+", 3000, 400, CfrAlgorithm::CfrPlus);
    run("sync CFR+", 3000, 400, CfrAlgorithm::SyncCfrPlus);
    run("sync CFR+", 1000, 400, CfrAlgorithm::SyncCfrPlus);
}

#[test]
#[ignore]
fn probe_sync_deep_tail() {
    // How low can eps go before a floor appears? Verifies the ~T^-1.9 sync
    // tail holds toward 1e-5 at mid density (FP/eval floors would show here
    // long before full scale).
    run("sync deep tail", 3000, 2000, CfrAlgorithm::SyncCfrPlus);
}

#[test]
#[ignore]
fn probe_dcfr_3000() {
    run("DCFR", 3000, 300, CfrAlgorithm::dcfr_default());
}

/// Solve at mid density, then localize the residual: per-player best-response
/// gain per dealer game (decider = p0), and the accept-node average strategies
/// that are mixing (the screening boundary).
#[test]
#[ignore]
fn probe_split_3000() {
    use truco_solver::abstraction::enumerate_deals;
    use truco_solver::cfr::best_response_value;
    use truco_solver::game_tree::build_all_trees_with_dealer;
    use truco_solver::info_set::AbstractAction;

    let score = Score { zero: 11, one: 10 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let mv = mv_11x10();
    let deals_n = 3000usize;

    let (table, stats) = solve_with_limit_algo(
        score.clone(),
        tc,
        300,
        &mv,
        Some(deals_n),
        None,
        CfrAlgorithm::CfrPlus,
        50,
        1,
        None,
    );
    eprintln!(
        "final expl (both games summed, /2): {:.6}",
        stats.exploitability.unwrap()
    );

    // Same stride logic as subsample_deals.
    let mut deals = enumerate_deals(&tc);
    let n = deals.len();
    deals = (0..deals_n)
        .map(|i| deals[i * n / deals_n].clone())
        .collect();
    let w: f64 = deals.iter().map(|d| d.weight).sum();
    for d in &mut deals {
        d.weight /= w;
    }

    for dealer in 0..2u8 {
        let prebuilt = build_all_trees_with_dealer(&score, tc, &deals, Some(dealer)).unwrap();
        let br_decider = best_response_value(&prebuilt, &table, &score, &mv, 0);
        let br_opponent = best_response_value(&prebuilt, &table, &score, &mv, 1);
        eprintln!(
            "dealer={}: BR gain decider(p0)={:.6}  opponent(p1)={:.6}",
            dealer, br_decider, br_opponent
        );
    }

    // Accept-node mixing census: how many hands are pure-accept / pure-fold /
    // mixing in the average strategy, per position.
    for is_dealer in [true, false] {
        let (mut acc, mut fold, mut mixed) = (0, 0, 0);
        for (key, info) in &table.info_sets {
            if info.player != 0 || !info.history.is_empty() || info.is_dealer != is_dealer {
                continue;
            }
            let data = &table.data[key];
            let Some(ai) = data
                .actions
                .iter()
                .position(|a| *a == AbstractAction::AcceptEleven)
            else {
                continue;
            };
            let p = data.average_strategy()[ai];
            if p > 0.99 {
                acc += 1;
            } else if p < 0.01 {
                fold += 1;
            } else {
                mixed += 1;
            }
        }
        eprintln!(
            "decider position is_dealer={}: pure-accept={} pure-fold={} mixing={}",
            is_dealer, acc, fold, mixed
        );
    }
}
