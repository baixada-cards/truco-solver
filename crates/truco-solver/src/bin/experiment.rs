//! Experiment runner binary for autoresearch.
//!
//! Builds game trees once, then runs the experimental algorithm from
//! `cfr_experiment.rs` for a fixed time budget, and reports exploitability.
//!
//! Usage:
//!   EXPERIMENT_TIME_BUDGET=600 cargo run --release -p truco-solver --bin experiment

use std::time::Instant;

use truco_engine::{Player, Score};
use truco_solver::abstraction::{enumerate_deals, TurnupClass};
use truco_solver::cfr::compute_exploitability;
use truco_solver::cfr_experiment;
use truco_solver::game_tree::build_all_trees;
use truco_solver::match_value::MatchValueTable;
use truco_solver::strategy::StrategyTable;

fn main() {
    env_logger::init();

    let time_budget_secs: u64 = std::env::var("EXPERIMENT_TIME_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let score = Score { zero: 11, one: 11 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let mv = MatchValueTable::new();

    eprintln!("=== Experiment Runner ===");
    eprintln!(
        "Score: {}x{} | TC: {} | Budget: {}s",
        score.zero, score.one, tc.blocked_plain_level, time_budget_secs
    );

    // Phase 1: Build trees (not counted in time budget)
    eprintln!("Building game trees...");
    let build_start = Instant::now();
    let deals = enumerate_deals(&tc);
    let prebuilt = build_all_trees(&score, tc, &deals)
        .expect("tree building failed: enumerate_deals produces valid deals");
    let build_secs = build_start.elapsed().as_secs_f64();
    eprintln!(
        "Trees built in {:.1}s ({} deals, {} entries)",
        build_secs,
        deals.len(),
        prebuilt.entries.len()
    );

    // Initialize strategy table
    let mut table = StrategyTable::new();
    for (_key, info_set, actions) in &prebuilt.info_sets {
        table.get_or_insert(info_set, actions);
    }
    eprintln!("{} info sets", table.len());

    // Phase 2: Initialize experiment state
    let mut state = cfr_experiment::init(table.len());

    // Phase 3: Run iterations for the time budget
    eprintln!("Starting iterations (budget: {}s)...", time_budget_secs);
    let iter_start = Instant::now();
    let mut iteration = 0u64;

    loop {
        iteration += 1;
        let traversing = (iteration % 2) as Player;

        // Run one iteration
        cfr_experiment::iterate(
            &mut state, iteration, traversing, &prebuilt, &mut table, &score, &mv,
        );

        // Post-iteration processing
        cfr_experiment::post_iterate(&mut state, iteration, &mut table);

        let elapsed = iter_start.elapsed().as_secs();
        if elapsed >= time_budget_secs {
            break;
        }

        // Log progress every 10 iterations
        if iteration.is_multiple_of(10) {
            eprintln!(
                "  iteration {} ({:.0}s / {}s)",
                iteration,
                iter_start.elapsed().as_secs_f64(),
                time_budget_secs,
            );
        }
    }

    let iter_secs = iter_start.elapsed().as_secs_f64();
    eprintln!("Completed {} iterations in {:.1}s", iteration, iter_secs);

    // Phase 4: Compute exploitability (not counted in time budget)
    eprintln!("Computing exploitability...");
    let expl_start = Instant::now();
    let expl = compute_exploitability(&prebuilt, &table, &score, &mv);
    let expl_secs = expl_start.elapsed().as_secs_f64();
    eprintln!("Exploitability computed in {:.1}s", expl_secs);

    // Output results in parseable format
    println!("---");
    println!("exploitability: {:.6}", expl);
    println!("iterations: {}", iteration);
    println!("iter_wall_secs: {:.1}", iter_secs);
    println!("build_secs: {:.1}", build_secs);
    println!("info_sets: {}", table.len());
}
