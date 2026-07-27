use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use truco_engine::Score;
use truco_solver::abstraction::{enumerate_deals, TurnupClass};
use truco_solver::allocation_scout::{
    estimate_whole_match_allocation, DealPanel, ALLOCATION_BANDS,
};
use truco_solver::cfr;
use truco_solver::compact_br::{
    compact_best_response_value, compact_profile_value, projected_policy_row, DominatedProjection,
};
use truco_solver::game_tree::{MissingPolicyFallback, PolicyLookup, PolicyValueSource, TreeRules};
use truco_solver::info_set::{AbstractAction, InfoSet, InfoSetKey};
use truco_solver::match_value::{solve_order_between, MatchValueTable};
use truco_solver::storage::{
    load_checkpoint, load_compact_average_checkpoint,
    load_compact_average_checkpoint_with_player_swap, load_compact_average_policy,
    load_match_values, load_strategy, save_match_values, save_strategy, stream_strategy_rows,
    CompactAveragePolicy, SolvedStateMeta,
};

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "solve-tc" => run_solve_tc(&args),
        "eval-ckpt" => run_eval_ckpt(&args),
        "solve-asym" => run_solve_asym(&args),
        "set-mv" => run_set_mv(&args),
        "checkpoint-to-strategy" => run_checkpoint_to_strategy(&args),
        "export-teacher" => run_export_teacher(&args),
        "export-band-meta" => run_export_band_meta(&args),
        "audit-teach-residue" => run_audit_teach_residue(&args),
        "export-chart" => run_export_chart(&args),
        "export-bot-policy" => run_export_bot_policy(&args),
        "dealer-advantage" => run_dealer_advantage(&args),
        "compare" => run_compare(&args),
        "mccfr-bench" => run_mccfr_bench(&args),
        "treesize" => run_tree_size_survey(),
        "count-tree" => run_count_tree(&args),
        "restricted-bench" => run_restricted_bench(&args),
        "compact-br" => run_compact_br(&args),
        "resolve-subgames" => run_resolve_subgames(&args),
        "trunk-solve" => run_trunk_solve(&args),
        "trunk-scout" => run_trunk_scout(&args),
        "allocation-scout" => run_allocation_scout(&args),
        "compare-policies" => run_compare_policies(&args),
        "policy-stats" => run_policy_stats(&args),
        "benchmark" => run_benchmark(),
        "pipeline" => run_pipeline(&args),
        _ => {
            eprintln!("Usage: solve <mode> [options]");
            eprintln!();
            eprintln!("  solve-tc [--score SxS] [--tc N] [--eps E] [--max-iters N]");
            eprintln!("           [--max-deals N]  # deterministic cheap benchmark subset");
            eprintln!("           [--algo cfr+|dcfr] [--expl-every N] [--log FILE]");
            eprintln!("           [--regret-prune-threshold X --regret-prune-warmup N]");
            eprintln!("           [--regret-prune-revisit N]  # SyncCFR+, opt-in");
            eprintln!(
                "           [--time-budget SECS] [--checkpoint PATH] [--checkpoint-every SECS]"
            );
            eprintln!("           [--resume PATH] [--extra-iters N] [--data-dir DIR]");
            eprintln!("           [--warmstart-from CKPT --warmstart-profile-transfer]");
            eprintln!("           [--tremble-eps E] [--tremble-eps-end E2]");
            eprintln!("      Solve one (score, TC) pair until exploitability <= E or the time");
            eprintln!("      budget elapses; checkpoints full state and writes the strategy file.");
            eprintln!("      Defaults: score=11x11  tc=0  eps=0.01  algo=cfr+  expl-every=10");
            eprintln!(
                "                data-dir=solutions  checkpoint-every=300  checkpoint=solutions/{{s0}}x{{s1}}/tc{{tc}}.ckpt.bin"
            );
            eprintln!();
            eprintln!("  solve-asym --score 11x10 --tc N --match-values PATH");
            eprintln!("           [--warmstart-from CKPT] [--rounds R] [--inner-eps E]");
            eprintln!("           [--inner-time-budget S] [--inner-max-iters N] [--out DIR]");
            eprintln!("      Dealer-exact freeze-accept + equity policy-iteration solve of an");
            eprintln!("      asymmetric mão-de-onze state. Freezes the decider's accept/fold");
            eprintln!("      decision per dealer, alternating an inner CFR card-play solve with");
            eprintln!("      an accept-set update until the accept sets are stable.");
            eprintln!("      Defaults: tc=0  rounds=8  inner-eps=0.005  out=solutions/asym");
            eprintln!();
            eprintln!("  pipeline [--from-score SxS] [--to-score SxS] [--eps E] [--data-dir DIR]");
            eprintln!("           [--warmstart-neighbors]");
            eprintln!("           [--max-iters N] [--expl-every N] [--algo cfr+|dcfr]");
            eprintln!("           [--save-all-strategies]");
            eprintln!("      Top-down solve with match-value checkpoints; parallel TCs per level.");
            eprintln!("      Defaults: --from-score 11x11  --to-score 10x10");
            eprintln!("      Legacy alias: --min-score SxS behaves like --to-score SxS");
            eprintln!();
            eprintln!("  compare [--iters N] [--expl-every N] [--log FILE]");
            eprintln!("      Run CFR+ vs DCFR for N iterations each and compare");
            eprintln!("      exploitability vs wall-time. Default: --iters 100  --expl-every 10");
            eprintln!();
            eprintln!("  mccfr-bench [--score SxS] [--tc N] [--dealer D]");
            eprintln!("      [--samples N] [--batch-size N] [--eval-deals N]");
            eprintln!("      [--seed-checkpoint PATH | --seed-strategy PATH]");
            eprintln!("      [--seed-regret-mass X] [--rng-seed N] [--out PATH]");
            eprintln!("      True frozen-strategy external-sampling mini-batch scout;");
            eprintln!("      evaluation is exact on the fixed strided --eval-deals subset.");
            eprintln!();
            eprintln!("  export-bot-policy (--checkpoint PATH | --strategy PATH) --out FILE.tpb");
            eprintln!(
                "      [--dealer 0|1]  # required with --strategy; overrides checkpoint meta"
            );
            eprintln!("      Export the average strategy as the live solver bot's mmap lookup");
            eprintln!("      artifact (sorted fixed-width records, u8-quantized probs). Prints");
            eprintln!("      the policy-dir manifest entry as JSON on stdout. RAM: rows are");
            eprintln!("      collected before sorting — budget ~4 GB for a 10x10 profile.");
            eprintln!();
            eprintln!("  audit-teach-residue [--teach-dir DIR | --teach PATH ...]");
            eprintln!("      Sweep .teach files for per-info-set mass above 1/5/20pp Q gaps.");
            eprintln!();
            eprintln!("  treesize");
            eprintln!("      Measure tree size at a range of score states (no CFR).");
            eprintln!();
            eprintln!("  count-tree [--score SxS] [--tc N] [--dealer D] [--max-deals N]");
            eprintln!("      [--policy-checkpoint PATH | --policy-strategy PATH | --policy-empty]");
            eprintln!("      [--policy-mode profile|br0|br1|br-union|chosen-br-union]");
            eprintln!("      [--policy-mode chosen-br-closure|chosen-br-both|all]");
            eprintln!("      [--policy-values average|current]");
            eprintln!("      [--support-thresholds 0,1e-8,1e-6,1e-4]");
            eprintln!("      [--missing-policy all|first|all-except-raise]");
            eprintln!("      Count the raw tree, or policy support / unilateral-BR closures,");
            eprintln!("      using space-for-time DFS without solver-ready allocations.");
            eprintln!("      chosen-br-* additionally requires --match-values and builds");
            eprintln!("      the full tree once to obtain certified per-info-set BR choices.");
            eprintln!();
            eprintln!("  restricted-bench --policy-checkpoint PATH --match-values PATH");
            eprintln!("      [--score SxS] [--tc N] [--dealer D] --max-deals N");
            eprintln!("      [--support-threshold 1e-4] [--eps E] [--max-iters N]");
            eprintln!("      [--oracle-rounds N] [--skip-full-control]");
            eprintln!("      Cheap restricted CFR + iterative full-BR accuracy gate.");
            eprintln!();
            eprintln!("  compact-br --policy-checkpoint PATH --match-values PATH");
            eprintln!("      [--project-dominated remap|renormalize]  # pruned-action mass");
            eprintln!("      [--legacy-tree]  # evaluate on the pre-2026-07-16 tree");
            eprintln!();
            eprintln!("  resolve-subgames --score SxS --tc N --dealer D --blueprint CKPT.bin");
            eprintln!("      --match-values MV.bin [--max-deals N] [--iters K (default 120)]");
            eprintln!("      [--repair-subgame biggest|IDX] [--composed-out PATH] [--legacy-tree]");
            eprintln!("      Safe subgame re-solving (CFR-D, plan 84): re-solve every round-2");
            eprintln!("      subgame against the blueprint and certify the composed profile.");
            eprintln!(
                "      --blueprint accepts a full CFR checkpoint or a saved average-strategy"
            );
            eprintln!(
                "      artifact. Prints RESOLVE_REPORT (+ REPAIR_REPORT with --repair-subgame)"
            );
            eprintln!(
                "      and per-phase wall times. Use --max-deals locally (full builds are 6GB+)."
            );
            eprintln!();
            eprintln!("  trunk-solve --score SxS --tc N --dealer D --match-values MV.bin");
            eprintln!("      [--max-deals N] [--rounds N (30)] [--trunk-sweeps T (3)]");
            eprintln!("      [--subgame-iters R (3)] [--final-iters K (120)]");
            eprintln!("      [--baseline-iters 90] [--legacy-tree] [--composed-out PATH]");
            eprintln!("      [--deep [--jobs N (1)] [--keep-arenas] [--certify full|raw|skip]");
            eprintln!(
                "       [--checkpoint PATH --checkpoint-every N] [--resume]]  # Phase-5 deep cell"
            );
            eprintln!(
                "      From-scratch CFR-D (plan 84 Phase 4): alternate trunk sweeps (round-2"
            );
            eprintln!("      boundary as cached value terminals) with gadget-free warm subgame");
            eprintln!("      re-solves, accumulate boundary CBVs, recover behind the gadget, and");
            eprintln!("      certify. No blueprint. Prints TRUNK_* lines incl. the re-solve");
            eprintln!("      multiplier vs a monolithic --baseline-iters solve.");
            eprintln!();
            eprintln!(
                "  trunk-scout --score SxS --tc N --dealer D [--max-deals N] [--legacy-tree]"
            );
            eprintln!("      Cheap sizing scout (no arena): total / trunk-region / subgame node");
            eprintln!("      and info-set counts for the Phase-5 box choice. On a strided");
            eprintln!("      --max-deals subset also prints a labeled linear extrapolation.");
            eprintln!();
            eprintln!("  compare-policies --a PATH --b PATH [--remap-turnup] [--reach-weighted] [--dump-divergent K [--dump-min-tv TV]]");
            eprintln!("      Unweighted row-level similarity of two saved average strategies.");
            eprintln!("      [--score SxS] [--tc N] [--dealer D] [--max-deals N]");
            eprintln!("      [--missing-policy all|first|all-except-raise] [--control]");
            eprintln!("      Space-for-time exact BR over DFS; --control A/Bs the arena oracle.");
            eprintln!();
            eprintln!("  allocation-scout --policy-checkpoint PATH [--policy-checkpoint PATH ...]");
            eprintln!("      [--max-deals N] [--panels N]");
            eprintln!("      [--missing-policy all|first|all-except-raise]");
            eprintln!("      Deterministic sampled score-DAG reach/error allocator. Checkpoints");
            eprintln!("      may cover several TCs; missing TCs are explicitly renormalized away.");
            eprintln!();
            eprintln!("  benchmark");
            eprintln!("      10 iterations at 11x11, TC 0, with extrapolations.");
            std::process::exit(1);
        }
    }
}

// ─── pipeline ────────────────────────────────────────────────────────────────

fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().max(1))
        .unwrap_or(1)
}

fn same_band_neighbor_scores(score: &Score) -> Vec<Score> {
    let target_band = truco_solver::game_tree::band_signature(score, None);
    [
        (score.zero.saturating_add(1), score.one),
        (score.zero, score.one.saturating_add(1)),
    ]
    .into_iter()
    .filter(|&(zero, one)| zero <= 11 && one <= 11 && zero >= one)
    .map(|(zero, one)| Score { zero, one })
    .filter(|candidate| {
        candidate != score
            && truco_solver::game_tree::band_signature(candidate, None) == target_band
    })
    .collect()
}

fn neighboring_warmstart_checkpoint(
    data_dir: &Path,
    score: &Score,
    tc: TurnupClass,
) -> Option<(Score, PathBuf)> {
    same_band_neighbor_scores(score)
        .into_iter()
        .find_map(|candidate| {
            let path = data_dir
                .join(format!("{}x{}", candidate.zero, candidate.one))
                .join(format!("tc{}.ckpt.bin", tc.blocked_plain_level));
            path.exists().then_some((candidate, path))
        })
}

fn run_pipeline(args: &[String]) {
    let from_score = parse_score_flag(args, "--from-score", Score { zero: 11, one: 11 });
    let to_score = if has_flag(args, "--to-score") {
        parse_score_flag(args, "--to-score", Score { zero: 10, one: 10 })
    } else {
        parse_score_flag(args, "--min-score", Score { zero: 10, one: 10 })
    };
    let from_total = from_score.zero as u32 + from_score.one as u32;
    let to_total = to_score.zero as u32 + to_score.one as u32;
    if from_total < to_total {
        eprintln!(
            "error: --from-score {}x{} must be at or above --to-score {}x{}",
            from_score.zero, from_score.one, to_score.zero, to_score.one
        );
        std::process::exit(2);
    }
    let eps = parse_f64_flag(args, "--eps", 0.01);
    let max_iters = parse_u64_flag(args, "--max-iters", u64::MAX);
    let expl_every = parse_u64_flag(args, "--expl-every", 50);
    let data_dir = parse_str_flag(args, "--data-dir", "solutions");
    let jobs = parse_usize_flag(args, "--jobs", default_jobs());
    let algo = parse_algorithm(args);
    let save_all = parse_bool_flag(args, "--save-all-strategies");
    let warmstart_neighbors = parse_bool_flag(args, "--warmstart-neighbors");

    let config = cfr::SolveConfig {
        max_iters,
        target_expl: eps,
        algorithm: algo.clone(),
        expl_every,
        ..Default::default()
    };

    let dir = Path::new(&data_dir);
    fs::create_dir_all(dir).expect("create data-dir");
    let mv_path = dir.join("match_values.bin");

    let mut mv = if mv_path.exists() {
        load_match_values(&mv_path).unwrap_or_else(|e| {
            eprintln!(
                "warn: could not load {}: {}, starting fresh",
                mv_path.display(),
                e
            );
            MatchValueTable::new()
        })
    } else {
        MatchValueTable::new()
    };

    let states = solve_order_between(from_total, to_total);
    println!(
        "\n=== pipeline  window={}x{} -> {}x{}  eps={}  jobs={}  data-dir={}  algo={:?} ===",
        from_score.zero, from_score.one, to_score.zero, to_score.one, eps, jobs, data_dir, algo
    );
    println!("  {} score states in scope", states.len());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("rayon thread pool");

    let mut idx = 0usize;
    while idx < states.len() {
        let (s0, s1) = states[idx];
        let total = s0 + s1;
        let mut group = vec![(s0, s1)];
        idx += 1;
        while idx < states.len() && states[idx].0 + states[idx].1 == total {
            group.push(states[idx]);
            idx += 1;
        }

        // A score is fully solved only once both dealer cells are written.
        group.retain(|&(a, b)| !(mv.is_solved(a, b, 0) && mv.is_solved(a, b, 1)));
        // Player-swap symmetry: (s1, s0) with dealer d is (s0, s1) with dealer
        // 1-d under relabeling, so only the s0 >= s1 triangle is solved and the
        // mirror cells are filled from it below. Halves the asymmetric grid and
        // guarantees the mirrors are exactly consistent.
        group.retain(|&(a, b)| a >= b);
        if group.is_empty() {
            continue;
        }

        println!(
            "\n--- Level total={}  scores={:?}  ({} parallel subgames) ---",
            total,
            group,
            group.len() * 9
        );

        let jobs_vec: Vec<(Score, TurnupClass)> = group
            .iter()
            .flat_map(|&(zs, os)| {
                TurnupClass::all()
                    .into_iter()
                    .map(move |tc| (Score { zero: zs, one: os }, tc))
            })
            .collect();

        let mv_arc = Arc::new(mv.clone());
        let config_arc = Arc::new(config.clone());
        let data_dir_arc = Arc::new(data_dir.clone());
        let algo_label = algo_label_of(&algo);

        let level_start = Instant::now();
        let results: Vec<_> = pool.install(|| {
            jobs_vec
                .into_par_iter()
                .map(|(score, tc)| {
                    // Per-subgame full-state checkpoint so a killed/preempted
                    // pipeline level resumes mid-solve instead of restarting
                    // every subgame of the level from scratch.
                    let ckpt_path = Path::new(data_dir_arc.as_str())
                        .join(format!("{}x{}", score.zero, score.one))
                        .join(format!("tc{}.ckpt.bin", tc.blocked_plain_level));
                    let resume = match load_checkpoint(&ckpt_path) {
                        Ok((rtable, meta))
                            if meta.score == (score.zero, score.one)
                                && meta.turnup_class == tc
                                && meta.algo == algo_label
                                && meta.dealer_filter.is_none() =>
                        {
                            println!(
                                "  resuming {}x{} tc{} from iter {}",
                                score.zero, score.one, tc.blocked_plain_level, meta.iteration
                            );
                            Some((rtable, meta.iteration))
                        }
                        _ => None,
                    };
                    let mut job_config = config_arc.as_ref().clone();
                    job_config.checkpoint_path = Some(ckpt_path);
                    job_config.checkpoint_every_secs = Some(600.0);
                    if resume.is_none() && warmstart_neighbors {
                        if let Some((source_score, source_path)) = neighboring_warmstart_checkpoint(
                            Path::new(data_dir_arc.as_str()),
                            &score,
                            tc,
                        ) {
                            println!(
                                "  warm-starting {}x{} tc{} from neighboring {}x{}",
                                score.zero,
                                score.one,
                                tc.blocked_plain_level,
                                source_score.zero,
                                source_score.one
                            );
                            job_config.warmstart_checkpoint = Some(source_path);
                            job_config.warmstart_same_band = true;
                        }
                    }
                    let (table, stats) = cfr::solve_until(
                        score.clone(),
                        tc,
                        &job_config,
                        mv_arc.as_ref(),
                        resume,
                        |_iter, _expl, _secs| {},
                    );
                    // Per-dealer game values (±1 space): (value_when_p0_deals,
                    // value_when_p1_deals). Stored separately so continuations
                    // land on the correct dealer of the next hand.
                    let (gv_d0, gv_d1) =
                        stats.game_value_per_dealer.unwrap_or((f64::NAN, f64::NAN));
                    (
                        score.zero,
                        score.one,
                        tc.blocked_plain_level,
                        gv_d0,
                        gv_d1,
                        table,
                        stats,
                    )
                })
                .collect()
        });

        // TC-weighted aggregation of the per-dealer game values, kept separate
        // for dealer 0 and dealer 1.
        let mut sum_wgv_d0: HashMap<(u8, u8), f64> = HashMap::new();
        let mut sum_wgv_d1: HashMap<(u8, u8), f64> = HashMap::new();
        for (zs, os, tc_level, gv_d0, gv_d1, _, _) in &results {
            let tc = TurnupClass {
                blocked_plain_level: *tc_level,
            };
            let w = tc.weight();
            *sum_wgv_d0.entry((*zs, *os)).or_insert(0.0) += w * gv_d0;
            *sum_wgv_d1.entry((*zs, *os)).or_insert(0.0) += w * gv_d1;
        }

        let level_secs = level_start.elapsed().as_secs_f64();
        let keys: Vec<(u8, u8)> = sum_wgv_d0.keys().copied().collect();
        for (zs, os) in keys {
            let v_d0 = sum_wgv_d0[&(zs, os)];
            let v_d1 = sum_wgv_d1[&(zs, os)];
            // Convert each per-dealer ±1 value to a P0 match-win probability.
            let match_p0_d0 = v_d0 / 2.0 + 0.5;
            let match_p0_d1 = v_d1 / 2.0 + 0.5;
            mv.set(zs, os, 0, match_p0_d0);
            mv.set(zs, os, 1, match_p0_d1);
            if zs != os {
                // Mirror state by player swap: mv(s1, s0, d) = 1 - mv(s0, s1, 1-d).
                mv.set(os, zs, 0, 1.0 - match_p0_d1);
                mv.set(os, zs, 1, 1.0 - match_p0_d0);
            }
            let exs: Vec<f64> = results
                .iter()
                .filter(|(a, b, _, _, _, _, _)| *a == zs && *b == os)
                .filter_map(|(_, _, _, _, _, _, st)| st.exploitability)
                .collect();
            let worst_expl =
                exs.iter()
                    .cloned()
                    .fold(f64::NAN, |acc, x| if acc.is_nan() { x } else { acc.max(x) });
            println!(
                "  score {}x{}  P0_win[p0-deals]={:.6}  P0_win[p1-deals]={:.6}  max_expl={:.6}  (level {:.1}s)",
                zs, os, match_p0_d0, match_p0_d1, worst_expl, level_secs
            );
        }

        save_match_values(&mv_path, &mv).expect("checkpoint match_values.bin");

        for (zs, os, tc_level, _gv_d0, _gv_d1, table, stats) in results {
            let st = zs as u32 + os as u32;
            if save_all || st == to_total {
                let sub = dir.join(format!("{}x{}", zs, os));
                fs::create_dir_all(&sub).expect("strategy subdir");
                let path = sub.join(format!("tc{}.bin", tc_level));
                let meta = SolvedStateMeta {
                    score: (zs, os),
                    turnup_class: TurnupClass {
                        blocked_plain_level: tc_level,
                    },
                    iterations: stats.iterations,
                    num_info_sets: stats.num_info_sets,
                };
                save_strategy(&path, &table, meta).expect("save strategy");
            }
        }
    }

    println!(
        "\n=== pipeline finished  checkpoint: {} ===",
        mv_path.display()
    );
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn parse_missing_policy(args: &[String]) -> MissingPolicyFallback {
    match parse_str_flag(args, "--missing-policy", "all-except-raise").as_str() {
        "all" => MissingPolicyFallback::All,
        "first" => MissingPolicyFallback::First,
        "all-except-raise" => MissingPolicyFallback::AllExceptRaise,
        other => panic!("unknown --missing-policy {other}; expected all|first|all-except-raise"),
    }
}

fn parse_dominated_projection(args: &[String]) -> DominatedProjection {
    match parse_str_flag(args, "--project-dominated", "remap").as_str() {
        "remap" => DominatedProjection::Remap,
        "renormalize" => DominatedProjection::Renormalize,
        other => panic!("unknown --project-dominated {other}; expected remap|renormalize"),
    }
}

// ─── compact exact best response ───────────────────────────────────────────

fn run_compact_br(args: &[String]) {
    let checkpoint_path = parse_opt_str_flag(args, "--policy-checkpoint")
        .expect("compact-br requires --policy-checkpoint PATH");
    let match_values_path = parse_opt_str_flag(args, "--match-values")
        .expect("compact-br requires --match-values PATH");
    let (policy, meta) = load_compact_average_checkpoint(Path::new(&checkpoint_path))
        .expect("stream compact checkpoint policy");
    let score = parse_score_flag(
        args,
        "--score",
        Score {
            zero: meta.score.0,
            one: meta.score.1,
        },
    );
    let tc_level = parse_u8_flag(args, "--tc", meta.turnup_class.blocked_plain_level);
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let dealer = parse_u8_flag(args, "--dealer", meta.dealer_filter.unwrap_or(0));
    assert!(dealer < 2, "--dealer must be 0 or 1");
    assert_eq!(
        tc, meta.turnup_class,
        "compact checkpoint turnup class does not match --tc"
    );
    if let Some(source_dealer) = meta.dealer_filter {
        assert_eq!(
            dealer, source_dealer,
            "dealer-filtered checkpoint does not match --dealer"
        );
    }
    let max_deals = parse_usize_flag(args, "--max-deals", usize::MAX);
    let fallback = parse_missing_policy(args);
    let projection = parse_dominated_projection(args);
    let rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else {
        TreeRules::Current
    };
    let mv = load_match_values(Path::new(&match_values_path)).expect("load --match-values");
    // A hand from (s0, s1) can end at (s0+k, s1) or (s0, s1+k) for any stake
    // k. Every NON-terminal successor cell must be solved: unsolved cells
    // default to probability 0.5, which maps to a payoff of exactly 0 and
    // silently zeroes the whole evaluation instead of failing.
    for stake in 1..=12u8 {
        for (a, b) in [
            (score.zero + stake, score.one),
            (score.zero, score.one + stake),
        ] {
            if a >= truco_engine::MATCH_TARGET || b >= truco_engine::MATCH_TARGET {
                continue;
            }
            assert!(
                mv.is_solved(a, b, 1 - dealer),
                "--match-values has no solved value for successor score {a}x{b} \
                 (dealer {}); this table cannot certify {}x{}",
                1 - dealer,
                score.zero,
                score.one,
            );
        }
    }
    let mut deals = enumerate_deals(&tc);
    let total_deals = deals.len();
    cfr::subsample_deals(&mut deals, max_deals);

    println!(
        "COMPACT_BR_START score={}x{} tc={} dealer={} deals={}/{} policy={} fallback={:?} projection={:?} rules={:?}",
        score.zero,
        score.one,
        tc_level,
        dealer,
        deals.len(),
        total_deals,
        checkpoint_path,
        fallback,
        projection,
        rules
    );
    let started = Instant::now();
    let profile = compact_profile_value(
        &score,
        tc,
        &deals,
        Some(dealer),
        &policy,
        PolicyValueSource::Average,
        fallback,
        projection,
        rules,
        &mv,
    )
    .expect("compact profile evaluation");
    let after_profile = started.elapsed().as_secs_f64();
    let br0 = compact_best_response_value(
        &score,
        tc,
        &deals,
        Some(dealer),
        &policy,
        PolicyValueSource::Average,
        fallback,
        projection,
        rules,
        &mv,
        0,
    )
    .expect("compact player-0 BR");
    let after_br0 = started.elapsed().as_secs_f64();
    let br1 = compact_best_response_value(
        &score,
        tc,
        &deals,
        Some(dealer),
        &policy,
        PolicyValueSource::Average,
        fallback,
        projection,
        rules,
        &mv,
        1,
    )
    .expect("compact player-1 BR");
    let elapsed = started.elapsed().as_secs_f64();
    let gain0 = br0.total - profile.p0_value;
    let gain1 = br1.total + profile.p0_value;
    let exploitability = (br0.total + br1.total) / 2.0;
    println!(
        "COMPACT_BR_RESULT profile_p0={:.12} br0={:.12} br1={:.12} gain0={:.12} gain1={:.12} epsilon={:.12} profile_s={:.3} br0_s={:.3} br1_s={:.3} total_s={:.3} chosen0={} chosen1={} max_depth0={} max_depth1={} dfs_visits={} missing_profile_decisions={}",
        profile.p0_value,
        br0.total,
        br1.total,
        gain0,
        gain1,
        exploitability,
        after_profile,
        after_br0 - after_profile,
        elapsed - after_br0,
        elapsed,
        br0.chosen_info_sets,
        br1.chosen_info_sets,
        br0.max_depth,
        br1.max_depth,
        profile.dfs_visits + br0.dfs_visits + br1.dfs_visits,
        profile.missing_decisions,
    );
    let stats = profile.projection_stats;
    println!(
        "COMPACT_BR_PROJECTION remapped_visits={} remapped_mass={:.6} dropped_visits={} dropped_mass={:.6}",
        stats.remapped_visits, stats.remapped_mass, stats.dropped_visits, stats.dropped_mass,
    );

    if has_flag(args, "--control") {
        let control_started = Instant::now();
        let (table, control_meta) =
            load_checkpoint(Path::new(&checkpoint_path)).expect("load control checkpoint");
        assert_eq!(control_meta.score, meta.score);
        let prebuilt =
            truco_solver::game_tree::build_all_trees_with_dealer(&score, tc, &deals, Some(dealer))
                .expect("build control arena");
        // The arena oracle addresses strategy rows positionally, so it can
        // only evaluate a checkpoint whose saved action sets match this
        // tree exactly; a mismatched (older-tree) checkpoint would silently
        // misread rows or panic mid-pass. Detect and skip instead.
        let mismatched_rows = prebuilt
            .info_sets
            .iter()
            .filter(|(key, _, actions)| {
                table
                    .data
                    .get(key)
                    .is_some_and(|row| row.actions.as_slice() != &actions[..])
            })
            .count();
        if mismatched_rows > 0 {
            println!(
                "COMPACT_BR_CONTROL_SKIPPED mismatched_action_rows={} (checkpoint was solved on a different tree; the positional arena oracle cannot evaluate it)",
                mismatched_rows
            );
            return;
        }
        let values = cfr::compute_game_value_per_dealer(&prebuilt, &table, &score, &mv);
        let control_profile = if dealer == 0 { values.0 } else { values.1 };
        let control_br0 = cfr::best_response_value(&prebuilt, &table, &score, &mv, 0);
        let control_br1 = cfr::best_response_value(&prebuilt, &table, &score, &mv, 1);
        println!(
            "COMPACT_BR_CONTROL profile_p0={:.12} br0={:.12} br1={:.12} epsilon={:.12} total_s={:.3} diff_profile={:.3e} diff_br0={:.3e} diff_br1={:.3e}",
            control_profile,
            control_br0,
            control_br1,
            (control_br0 + control_br1) / 2.0,
            control_started.elapsed().as_secs_f64(),
            profile.p0_value - control_profile,
            br0.total - control_br0,
            br1.total - control_br1,
        );
    }
}

// ─── safe subgame re-solving (CFR-D) ───────────────────────────────────────

/// Re-solve every round-2 subgame against a blueprint and certify the composed
/// profile (plan 84). Also doubles as a measurement harness: each phase prints
/// its wall time. The blueprint may be either a full CFR checkpoint or a saved
/// average-strategy artifact — the latter is loaded via the checkpoint-parse
/// fallback (average strategies without regrets are all resolve needs).
fn run_resolve_subgames(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 11 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let blueprint_path = parse_opt_str_flag(args, "--blueprint")
        .expect("resolve-subgames requires --blueprint PATH");
    let match_values_path = parse_opt_str_flag(args, "--match-values")
        .expect("resolve-subgames requires --match-values PATH");
    let max_deals = parse_opt_str_flag(args, "--max-deals")
        .map(|v| v.parse::<usize>().expect("--max-deals must be a number"));
    let iters = parse_u64_flag(args, "--iters", 120);
    let repair_arg = parse_opt_str_flag(args, "--repair-subgame");
    let composed_out = parse_opt_str_flag(args, "--composed-out");
    let rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else {
        TreeRules::Current
    };

    // 1. Enumerate deals (strided subsample, exactly like run_solve_tc).
    let mut deals = enumerate_deals(&tc);
    let total_deals = deals.len();
    if let Some(limit) = max_deals {
        cfr::subsample_deals(&mut deals, limit);
    }

    println!(
        "RESOLVE_SUBGAMES_START score={}x{} tc={} dealer={} deals={}/{} iters={} rules={:?} blueprint={} match_values={}",
        score.zero,
        score.one,
        tc_level,
        dealer,
        deals.len(),
        total_deals,
        iters,
        rules,
        blueprint_path,
        match_values_path,
    );

    // 2. Build trees (respecting --legacy-tree).
    let t_build = Instant::now();
    let built = truco_solver::game_tree::build_all_trees_with_dealer_rules(
        &score,
        tc,
        &deals,
        Some(dealer),
        rules,
    )
    .expect("build trees");
    let build_s = t_build.elapsed().as_secs_f64();
    println!(
        "RESOLVE_PHASE build_s={:.3} info_sets={} deals={}",
        build_s,
        built.info_sets.len(),
        deals.len()
    );

    // 3. Load the blueprint. Prefer a full CFR checkpoint; fall back to a saved
    //    average-strategy artifact (average strategies without regrets are all
    //    the resolver needs). Then densify to Vec<ActionProbs> aligned with
    //    built.info_sets, defaulting unmatched rows to uniform.
    let t_bp = Instant::now();
    let bp_path = Path::new(&blueprint_path);
    let (table, meta_kind) = match load_checkpoint(bp_path) {
        Ok((table, meta)) => {
            assert_eq!(
                meta.score,
                (score.zero, score.one),
                "--blueprint checkpoint score {:?} does not match --score {}x{}",
                meta.score,
                score.zero,
                score.one
            );
            assert_eq!(
                meta.turnup_class, tc,
                "--blueprint checkpoint turnup class does not match --tc"
            );
            if let Some(source_dealer) = meta.dealer_filter {
                assert_eq!(
                    source_dealer, dealer,
                    "dealer-filtered --blueprint checkpoint does not match --dealer"
                );
            }
            (table, "checkpoint")
        }
        Err(ckpt_err) => match load_strategy(bp_path) {
            Ok((table, smeta)) => {
                assert_eq!(
                    smeta.score,
                    (score.zero, score.one),
                    "--blueprint strategy score {:?} does not match --score {}x{}",
                    smeta.score,
                    score.zero,
                    score.one
                );
                assert_eq!(
                    smeta.turnup_class, tc,
                    "--blueprint strategy turnup class does not match --tc"
                );
                // SolvedStateMeta carries no dealer_filter; the successor
                // assertion and boundary replay below still pin the dealer.
                (table, "strategy")
            }
            Err(strat_err) => panic!(
                "--blueprint {} parsed as neither a checkpoint ({}) nor a strategy artifact ({})",
                blueprint_path, ckpt_err, strat_err
            ),
        },
    };

    let mut matched = 0usize;
    let mut defaulted = 0usize;
    let blueprint: Vec<truco_solver::strategy::ActionProbs> = built
        .info_sets
        .iter()
        .map(|(key, _info, actions)| match table.data.get(key) {
            Some(d) => {
                matched += 1;
                d.average_strategy()
            }
            None => {
                defaulted += 1;
                truco_solver::strategy::uniform_probs(actions.len())
            }
        })
        .collect();
    // The hash-map table is a multi-GB pool at production scale (39.5M
    // heap-vec entries at 10x10) and everything downstream reads only the
    // densified blueprint — release it before the resolve allocates.
    drop(table);
    let bp_load_s = t_bp.elapsed().as_secs_f64();
    let total_rows = built.info_sets.len();
    let matched_frac = if total_rows > 0 {
        matched as f64 / total_rows as f64
    } else {
        0.0
    };
    println!(
        "RESOLVE_BLUEPRINT source={} load_s={:.3} matched={} defaulted_uniform={} total={} matched_frac={:.4}",
        meta_kind, bp_load_s, matched, defaulted, total_rows, matched_frac
    );
    if matched_frac < 0.90 {
        println!(
            "RESOLVE_WARNING only {:.1}% of info sets matched the blueprint; the rest defaulted to \
             uniform. This blueprint likely does not correspond to this (score, tc, dealer, tree) — \
             results below are not meaningful.",
            matched_frac * 100.0
        );
        eprintln!(
            "WARNING: only {:.1}% of info sets matched the blueprint (defaulted {} to uniform).",
            matched_frac * 100.0,
            defaulted
        );
    }

    // 4. Load match values and run the successor-solvedness assertion (unsolved
    //    successors silently zero the evaluation).
    let mv = load_match_values(Path::new(&match_values_path)).expect("load --match-values");
    for stake in 1..=12u8 {
        for (a, b) in [
            (score.zero + stake, score.one),
            (score.zero, score.one + stake),
        ] {
            if a >= truco_engine::MATCH_TARGET || b >= truco_engine::MATCH_TARGET {
                continue;
            }
            assert!(
                mv.is_solved(a, b, 1 - dealer),
                "--match-values has no solved value for successor score {a}x{b} \
                 (dealer {}); this table cannot certify {}x{}",
                1 - dealer,
                score.zero,
                score.one,
            );
        }
    }

    // 5. Resolve every subgame and compose.
    let t_resolve = Instant::now();
    let (composed, report) = truco_solver::resolve::resolve_all(
        &built,
        &blueprint,
        &score,
        tc,
        &deals,
        Some(dealer),
        rules,
        &mv,
        iters,
    );
    let resolve_s = t_resolve.elapsed().as_secs_f64();
    println!("RESOLVE_PHASE resolve_all_s={:.3}", resolve_s);

    println!(
        "RESOLVE_REPORT blueprint_eps={:.12} composed_eps={:.12} subgames={} largest_members={} largest_nodes={} largest_infosets={}",
        report.blueprint_eps,
        report.composed_eps,
        report.subgames,
        report.largest.members,
        report.largest.nodes,
        report.largest.info_sets,
    );

    // Per-subgame node-count summary: count, total, max, median.
    let mut node_counts: Vec<usize> = report.per_subgame.iter().map(|s| s.nodes).collect();
    let sg_count = node_counts.len();
    let total_nodes: usize = node_counts.iter().sum();
    let max_nodes = node_counts.iter().copied().max().unwrap_or(0);
    node_counts.sort_unstable();
    let median_nodes = if sg_count == 0 {
        0
    } else {
        node_counts[sg_count / 2]
    };
    println!(
        "RESOLVE_SUBGAME_STATS count={} total_nodes={} max_nodes={} median_nodes={}",
        sg_count, total_nodes, max_nodes, median_nodes
    );

    // 6. Optional repair experiment on one subgame.
    if let Some(spec) = repair_arg {
        let index = if spec == "biggest" {
            // Largest total node count. resolve_all pushes per_subgame in the
            // same order collect_boundary enumerates (which repair_experiment
            // re-derives deterministically), so an argmax over per_subgame node
            // counts is the boundary index repair_experiment expects.
            report
                .per_subgame
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| s.nodes)
                .map(|(i, _)| i)
                .expect("at least one subgame to repair")
        } else {
            let idx = spec
                .parse::<usize>()
                .expect("--repair-subgame must be 'biggest' or an integer index");
            assert!(
                idx < report.subgames,
                "--repair-subgame {} out of range (only {} subgames)",
                idx,
                report.subgames
            );
            idx
        };
        let t_repair = Instant::now();
        let repair = truco_solver::resolve::repair_experiment(
            &built,
            &blueprint,
            &score,
            tc,
            &deals,
            Some(dealer),
            rules,
            &mv,
            index,
            iters,
        );
        let repair_s = t_repair.elapsed().as_secs_f64();
        println!(
            "RESOLVE_PHASE repair_s={:.3} subgame_index={}",
            repair_s, index
        );
        println!(
            "REPAIR_REPORT blueprint_eps={:.12} corrupted_eps={:.12} repaired_eps={:.12}",
            repair.blueprint_eps, repair.corrupted_eps, repair.repaired_eps
        );
    }

    // 7. Optional: write the composed profile as a strategy artifact. Rows are
    //    (key, &info, &actions, &probs); save_strategy_rows normalizes, and
    //    ActionProbs are already normalized (they normalize to themselves).
    if let Some(out) = composed_out {
        let meta = SolvedStateMeta {
            score: (score.zero, score.one),
            turnup_class: tc,
            iterations: iters,
            num_info_sets: built.info_sets.len(),
        };
        let rows = built
            .info_sets
            .iter()
            .zip(composed.iter())
            .map(|((key, info, actions), probs)| (key.0, info, &actions[..], &probs[..]));
        truco_solver::storage::save_strategy_rows(Path::new(&out), meta, rows)
            .expect("write --composed-out");
        println!(
            "RESOLVE_COMPOSED_OUT path={} rows={}",
            out,
            built.info_sets.len()
        );
    }

    println!(
        "RESOLVE_SUBGAMES_DONE build_s={:.3} blueprint_load_s={:.3} resolve_all_s={:.3}",
        build_s, bp_load_s, resolve_s
    );
}

fn run_trunk_solve(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 8, one: 8 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let match_values_path = parse_opt_str_flag(args, "--match-values")
        .expect("trunk-solve requires --match-values PATH");
    let max_deals = parse_opt_str_flag(args, "--max-deals")
        .map(|v| v.parse::<usize>().expect("--max-deals must be a number"));
    let rounds = parse_usize_flag(args, "--rounds", 30);
    let trunk_iters = parse_u64_flag(args, "--trunk-sweeps", 3);
    let subgame_iters = parse_u64_flag(args, "--subgame-iters", 3);
    let final_iters = parse_u64_flag(args, "--final-iters", 120);
    let baseline_iters = parse_u64_flag(args, "--baseline-iters", 90);
    let composed_out = parse_opt_str_flag(args, "--composed-out");
    let rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else {
        TreeRules::Current
    };
    // Deep path (plan 84 Phase 5): trunk-only arena + per-subgame local
    // registries + streaming certificate, for cells too big for one arena.
    let deep = has_flag(args, "--deep");
    let jobs = parse_usize_flag(args, "--jobs", 1);
    let keep_arenas = has_flag(args, "--keep-arenas");
    let certify_mode = parse_opt_str_flag(args, "--certify").unwrap_or_else(|| "full".into());
    let certify = certify_mode != "skip";
    let certify_recoveries = certify_mode == "full";
    let checkpoint_path = parse_opt_str_flag(args, "--checkpoint");
    let checkpoint_every = parse_usize_flag(args, "--checkpoint-every", 0);
    let resume = has_flag(args, "--resume");

    // 1. Enumerate deals (strided subsample, exactly like run_solve_tc).
    let mut deals = enumerate_deals(&tc);
    let total_deals = deals.len();
    if let Some(limit) = max_deals {
        cfr::subsample_deals(&mut deals, limit);
    }

    if deep {
        run_deep_solve(
            &score,
            tc,
            tc_level,
            dealer,
            &deals,
            total_deals,
            rules,
            &match_values_path,
            rounds,
            trunk_iters,
            subgame_iters,
            final_iters,
            baseline_iters,
            jobs,
            keep_arenas,
            certify,
            certify_recoveries,
            checkpoint_path,
            checkpoint_every,
            resume,
            composed_out,
        );
        return;
    }

    println!(
        "TRUNK_START score={}x{} tc={} dealer={} deals={}/{} rounds={} trunk_sweeps={} subgame_iters={} final_iters={} baseline_iters={} rules={:?} match_values={}",
        score.zero,
        score.one,
        tc_level,
        dealer,
        deals.len(),
        total_deals,
        rounds,
        trunk_iters,
        subgame_iters,
        final_iters,
        baseline_iters,
        rules,
        match_values_path,
    );

    // 2. Build trees (respecting --legacy-tree).
    let t_build = Instant::now();
    let built = truco_solver::game_tree::build_all_trees_with_dealer_rules(
        &score,
        tc,
        &deals,
        Some(dealer),
        rules,
    )
    .expect("build trees");
    let build_s = t_build.elapsed().as_secs_f64();
    println!(
        "TRUNK_PHASE build_s={:.3} info_sets={} deals={}",
        build_s,
        built.info_sets.len(),
        deals.len()
    );

    // 3. Load match values and run the successor-solvedness assertion.
    let mv = load_match_values(Path::new(&match_values_path)).expect("load --match-values");
    for stake in 1..=12u8 {
        for (a, b) in [
            (score.zero + stake, score.one),
            (score.zero, score.one + stake),
        ] {
            if a >= truco_engine::MATCH_TARGET || b >= truco_engine::MATCH_TARGET {
                continue;
            }
            assert!(
                mv.is_solved(a, b, 1 - dealer),
                "--match-values has no solved value for successor score {a}x{b} \
                 (dealer {}); this table cannot certify {}x{}",
                1 - dealer,
                score.zero,
                score.one,
            );
        }
    }

    // 4. Decompose at the round-2 boundary (same inputs the trees were built
    //    from — the boundary walker replays the build).
    let t_boundary = Instant::now();
    let subgames = truco_solver::subgame::collect_boundary(&score, tc, &deals, Some(dealer), rules)
        .expect("boundary replay");
    let boundary_s = t_boundary.elapsed().as_secs_f64();
    println!(
        "TRUNK_PHASE boundary_s={:.3} subgames={}",
        boundary_s,
        subgames.len()
    );

    // 5. Trunk-CFR loop from scratch.
    let cfg = truco_solver::resolve::TrunkConfig {
        rounds,
        trunk_iters,
        subgame_iters,
        final_iters,
        baseline_iters,
    };
    let t_solve = Instant::now();
    let (composed, report) =
        truco_solver::resolve::trunk_solve(&built, &score, tc, &mv, &subgames, cfg);
    let solve_s = t_solve.elapsed().as_secs_f64();
    println!("TRUNK_PHASE trunk_solve_s={:.3}", solve_s);

    println!(
        "TRUNK_VISITS trunk={} subgame={} final={} total={} baseline={} multiplier={:.4}",
        report.trunk_visits,
        report.subgame_visits,
        report.final_visits,
        report.total_visits,
        report.baseline_visits,
        report.multiplier,
    );
    println!(
        "TRUNK_REPORT composed_eps={:.12} raw_eps={:.12} composed_eps_tail={:.12} composed_eps_br={:.12} game_value={:.9} game_value_d0={:.9} game_value_d1={:.9} subgames={} largest_members={} largest_nodes={} largest_infosets={}",
        report.composed_eps,
        report.raw_eps,
        report.composed_eps_tail,
        report.composed_eps_br,
        report.game_value,
        report.game_value_per_dealer.0,
        report.game_value_per_dealer.1,
        report.subgames,
        report.largest.members,
        report.largest.nodes,
        report.largest.info_sets,
    );

    // 6. Optional: write the composed profile as a strategy artifact.
    if let Some(out) = composed_out {
        let meta = SolvedStateMeta {
            score: (score.zero, score.one),
            turnup_class: tc,
            iterations: (rounds as u64) * (trunk_iters + subgame_iters) + final_iters,
            num_info_sets: built.info_sets.len(),
        };
        let rows = built
            .info_sets
            .iter()
            .zip(composed.iter())
            .map(|((key, info, actions), probs)| (key.0, info, &actions[..], &probs[..]));
        truco_solver::storage::save_strategy_rows(Path::new(&out), meta, rows)
            .expect("write --composed-out");
        println!(
            "TRUNK_COMPOSED_OUT path={} rows={}",
            out,
            built.info_sets.len()
        );
    }

    println!(
        "TRUNK_DONE build_s={:.3} boundary_s={:.3} trunk_solve_s={:.3}",
        build_s, boundary_s, solve_s
    );
}

/// Deep-cell CFR-D solve (plan 84 Phase 5): trunk-only arena, per-subgame local
/// registries with persistent accumulators, rayon over subgames, resumable
/// checkpoints, and a decomposed streaming certificate. Memory-decomposed
/// counterpart to `run_trunk_solve`; bit-identical certificates at `--jobs 1`.
#[allow(clippy::too_many_arguments)]
fn run_deep_solve(
    score: &Score,
    tc: TurnupClass,
    tc_level: u8,
    dealer: u8,
    deals: &[truco_solver::abstraction::AbstractDeal],
    total_deals: usize,
    rules: TreeRules,
    match_values_path: &str,
    rounds: usize,
    trunk_iters: u64,
    subgame_iters: u64,
    final_iters: u64,
    baseline_iters: u64,
    jobs: usize,
    keep_arenas: bool,
    certify: bool,
    certify_recoveries: bool,
    checkpoint_path: Option<String>,
    checkpoint_every: usize,
    resume: bool,
    composed_out: Option<String>,
) {
    println!(
        "DEEP_START score={}x{} tc={} dealer={} deals={}/{} rounds={} trunk_sweeps={} subgame_iters={} final_iters={} baseline_iters={} jobs={} keep_arenas={} certify={} rules={:?} match_values={}",
        score.zero,
        score.one,
        tc_level,
        dealer,
        deals.len(),
        total_deals,
        rounds,
        trunk_iters,
        subgame_iters,
        final_iters,
        baseline_iters,
        jobs,
        keep_arenas,
        certify,
        rules,
        match_values_path,
    );

    let mv = load_match_values(Path::new(match_values_path)).expect("load --match-values");
    for stake in 1..=12u8 {
        for (a, b) in [
            (score.zero + stake, score.one),
            (score.zero, score.one + stake),
        ] {
            if a >= truco_engine::MATCH_TARGET || b >= truco_engine::MATCH_TARGET {
                continue;
            }
            assert!(
                mv.is_solved(a, b, 1 - dealer),
                "--match-values has no solved value for successor score {a}x{b} (dealer {})",
                1 - dealer,
            );
        }
    }

    let cfg = truco_solver::deep::DeepConfig {
        rounds,
        trunk_iters,
        subgame_iters,
        final_iters,
        baseline_iters,
        jobs,
        keep_arenas,
        certify,
        certify_recoveries,
    };
    let checkpoint = checkpoint_path.map(|p| truco_solver::deep::CheckpointCfg {
        path: std::path::PathBuf::from(p),
        every: checkpoint_every,
        resume,
        stop_after: None,
    });

    let t_solve = Instant::now();
    // The composed artifact streams straight to `--composed-out` INSIDE the
    // solve: the old path returned a whole-profile HashMap (~757 M rows at 0×0)
    // and then rebuilt the full arena to enumerate keys — the two things the
    // deep path exists to avoid, and the 2026-07-23 post-certificate OOM.
    let artifact_path = composed_out.as_ref().map(std::path::PathBuf::from);
    let report = truco_solver::deep::deep_solve(
        score,
        tc,
        deals,
        dealer,
        rules,
        &mv,
        cfg,
        checkpoint.as_ref(),
        artifact_path
            .as_deref()
            .map(truco_solver::deep::ArtifactSink::File),
    );
    let solve_s = t_solve.elapsed().as_secs_f64();

    println!(
        "DEEP_MEM trunk_info_sets={} subgame_info_sets={} subgames={}",
        report.trunk_info_sets, report.subgame_info_sets, report.subgames,
    );
    println!(
        "DEEP_VISITS trunk={} subgame={} final={} total={} baseline={} multiplier={:.4}",
        report.trunk_visits,
        report.subgame_visits,
        report.final_visits,
        report.total_visits,
        report.baseline_visits,
        report.multiplier,
    );
    println!(
        "DEEP_REPORT composed_eps={:.12} raw_eps={:.12} composed_eps_tail={:.12} composed_eps_br={:.12} game_value={:.9} game_value_d0={:.9} game_value_d1={:.9} subgames={} largest_members={} largest_nodes={} largest_infosets={} solve_s={:.3}",
        report.composed_eps,
        report.raw_eps,
        report.composed_eps_tail,
        report.composed_eps_br,
        report.game_value,
        report.game_value_per_dealer.0,
        report.game_value_per_dealer.1,
        report.subgames,
        report.largest.members,
        report.largest.nodes,
        report.largest.info_sets,
        solve_s,
    );

    if let Some(out) = composed_out {
        println!("DEEP_COMPOSED_OUT path={out}");
    }

    println!("DEEP_DONE solve_s={:.3}", solve_s);
}

/// Cheap sizing scout for the Phase-5 box choice: total / trunk-region /
/// subgame node and info-set counts for one (score, tc, dealer) cell, via a
/// build-recursion replay with no arena allocation. On a strided `--max-deals`
/// subset it also prints a linear extrapolation to the full deal set (labeled;
/// exact-ish for nodes, an upper bound for the shared info-set counts).
fn run_trunk_scout(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 0, one: 0 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let max_deals = parse_opt_str_flag(args, "--max-deals")
        .map(|v| v.parse::<usize>().expect("--max-deals must be a number"));
    let rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else if has_flag(args, "--asymmetric-raise-prune") {
        TreeRules::AsymmetricRaisePrune
    } else {
        TreeRules::Current
    };

    let mut deals = enumerate_deals(&tc);
    let total_deals = deals.len();
    if let Some(limit) = max_deals {
        cfr::subsample_deals(&mut deals, limit);
    }
    let sampled = deals.len();

    println!(
        "SCOUT_START score={}x{} tc={} dealer={} deals={}/{} rules={:?}",
        score.zero, score.one, tc_level, dealer, sampled, total_deals, rules,
    );

    let t0 = Instant::now();
    let report =
        truco_solver::subgame::scout_sizes(&score, tc, &deals, Some(dealer), rules).expect("scout");
    let elapsed = t0.elapsed().as_secs_f64();

    println!(
        "SCOUT_RESULT deals_walked={} total_nodes={} total_info_sets={} trunk_nodes={} trunk_info_sets={} subgames={} largest_nodes(members={} nodes={} info_sets={}) largest_info_sets(members={} nodes={} info_sets={}) elapsed_s={:.1}",
        report.deals_walked,
        report.total_nodes,
        report.total_info_sets,
        report.trunk_nodes,
        report.trunk_info_sets,
        report.subgame_count,
        report.largest_by_nodes.members,
        report.largest_by_nodes.nodes,
        report.largest_by_nodes.info_sets,
        report.largest_by_info_sets.members,
        report.largest_by_info_sets.nodes,
        report.largest_by_info_sets.info_sets,
        elapsed,
    );

    if sampled < total_deals {
        let factor = total_deals as f64 / sampled as f64;
        println!(
            "SCOUT_EXTRAP factor={:.4} (×{sampled}→{total_deals} deals) total_nodes~={:.3e} trunk_nodes~={:.3e} total_info_sets<~={:.3e} trunk_info_sets<~={:.3e} subgames~={:.3e}",
            factor,
            report.total_nodes as f64 * factor,
            report.trunk_nodes as f64 * factor,
            report.total_info_sets as f64 * factor,
            report.trunk_info_sets as f64 * factor,
            report.subgame_count as f64 * factor,
        );
        println!(
            "NOTE: nodes are additive across deals (extrapolation ~exact); distinct info-set and subgame counts are SHARED across deals and grow SUBLINEARLY toward a saturation ceiling, so their ×factor values are UPPER BOUNDS, not estimates. IMPORTANT: the largest-subgame sizes here are a subset LOWER BOUND and are still growing — a subgame pools every deal consistent with its public history, so more deals add members and enlarge the biggest subgame's subtree. Full-deal largest-subgame sizing needs a full (or much larger) deal walk; sweep --max-deals to observe the trend."
        );
    }
}

#[derive(Default)]
struct LayeredCompactPolicy {
    layers: Vec<CompactAveragePolicy>,
}

impl PolicyLookup for LayeredCompactPolicy {
    fn action_probability(
        &self,
        key: InfoSetKey,
        action: AbstractAction,
        values: PolicyValueSource,
    ) -> Option<f64> {
        self.layers
            .iter()
            .find_map(|layer| layer.action_probability(key, action, values))
    }

    fn len(&self) -> usize {
        self.layers.iter().map(PolicyLookup::len).sum()
    }
}

// ─── sampled whole-match allocator ─────────────────────────────────────────

fn run_allocation_scout(args: &[String]) {
    let paths = parse_multi_str_flag(args, "--policy-checkpoint");
    assert!(
        !paths.is_empty(),
        "allocation-scout requires at least one --policy-checkpoint PATH"
    );
    let max_deals = parse_usize_flag(args, "--max-deals", 24);
    let panel_count = parse_usize_flag(args, "--panels", 3);
    assert!(panel_count > 0, "--panels must be positive");
    assert!(
        max_deals >= panel_count,
        "--max-deals must be at least --panels"
    );
    let fallback = parse_missing_policy(args);
    let projection = parse_dominated_projection(args);

    let mut policy = LayeredCompactPolicy::default();
    let mut turnups = Vec::new();
    let mut seen_turnups = HashSet::new();
    let mut sources = Vec::new();
    for path in &paths {
        let (layer, meta) = load_compact_average_checkpoint_with_player_swap(Path::new(path), true)
            .expect("stream and symmetrize allocator checkpoint");
        let level = meta.turnup_class.blocked_plain_level;
        assert!(
            seen_turnups.insert(level),
            "allocation-scout accepts at most one checkpoint per TC (duplicate tc{level})"
        );
        turnups.push(meta.turnup_class);
        sources.push(format!(
            "tc{}:{}x{}:dealer={:?}:{}",
            level, meta.score.0, meta.score.1, meta.dealer_filter, path
        ));
        policy.layers.push(layer);
    }
    turnups.sort_by_key(|tc| tc.blocked_plain_level);
    let represented_weight: f64 = turnups.iter().map(TurnupClass::weight).sum();
    println!(
        "ALLOCATION_START panels={} max_deals_per_tc={} represented_tcs={:?} represented_tc_weight={:.6} fallback={:?} projection={:?} compact_rows={} sources={}",
        panel_count,
        max_deals,
        turnups
            .iter()
            .map(|tc| tc.blocked_plain_level)
            .collect::<Vec<_>>(),
        represented_weight,
        fallback,
        projection,
        policy.len(),
        sources.join(",")
    );
    println!(
        "ALLOCATION_SCOPE sampled/profile-reach representative-band estimate; not an exact whole-match BR certificate; omitted TC weights are renormalized"
    );

    let mut tc_deals = Vec::new();
    for &tc in &turnups {
        let mut deals = enumerate_deals(&tc);
        cfr::subsample_deals(&mut deals, max_deals);
        tc_deals.push((tc, deals));
    }

    let started = Instant::now();
    let mut estimates = Vec::new();
    for panel_idx in 0..panel_count {
        let mut split_deals = Vec::new();
        for (tc, deals) in &tc_deals {
            let mut split: Vec<_> = deals
                .iter()
                .enumerate()
                .filter(|(i, _)| i % panel_count == panel_idx)
                .map(|(_, deal)| deal.clone())
                .collect();
            assert!(!split.is_empty(), "empty deterministic deal panel");
            cfr::subsample_deals(&mut split, usize::MAX);
            split_deals.push((*tc, split));
        }
        let panels: Vec<_> = split_deals
            .iter()
            .map(|(tc, deals)| DealPanel {
                tc: *tc,
                deals: deals.as_slice(),
            })
            .collect();
        let panel_started = Instant::now();
        let estimate = estimate_whole_match_allocation(&policy, &panels, fallback, projection)
            .expect("sampled allocation panel");
        println!(
            "ALLOCATION_PANEL panel={} initial_profile_p0={:.8} expected_hands={:.6} priority_gain0_mass={:.8} priority_gain1_mass={:.8} priority_error_mass={:.8} missing_profile_decisions={} dfs_visits={} wall_s={:.3}",
            panel_idx,
            estimate.initial_profile_p0,
            estimate.expected_hands,
            estimate.priority_gain0_mass,
            estimate.priority_gain1_mass,
            estimate.priority_error_mass,
            estimate.missing_profile_decisions,
            estimate.dfs_visits,
            panel_started.elapsed().as_secs_f64(),
        );
        estimates.push(estimate);
    }

    let n = estimates.len() as f64;
    let mean = |value: fn(&truco_solver::allocation_scout::AllocationEstimate) -> f64| {
        estimates.iter().map(value).sum::<f64>() / n
    };
    let range = |value: fn(&truco_solver::allocation_scout::AllocationEstimate) -> f64| {
        estimates
            .iter()
            .map(value)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            })
    };
    let (mass_lo, mass_hi) = range(|estimate| estimate.priority_error_mass);
    let mean_mass = mean(|estimate| estimate.priority_error_mass);
    println!(
        "ALLOCATION_RESULT panels={} mean_profile_p0={:.8} mean_expected_hands={:.6} mean_priority_gain0_mass={:.8} mean_priority_gain1_mass={:.8} mean_priority_error_mass={:.8} panel_range_error_mass=[{:.8},{:.8}] total_wall_s={:.3}",
        panel_count,
        mean(|estimate| estimate.initial_profile_p0),
        mean(|estimate| estimate.expected_hands),
        mean(|estimate| estimate.priority_gain0_mass),
        mean(|estimate| estimate.priority_gain1_mass),
        mean_mass,
        mass_lo,
        mass_hi,
        started.elapsed().as_secs_f64(),
    );
    for band in ALLOCATION_BANDS {
        let rows: Vec<_> = estimates
            .iter()
            .map(|estimate| &estimate.by_band[&band])
            .collect();
        let mean_visits = rows.iter().map(|row| row.visits).sum::<f64>() / n;
        let mean_gain0 = rows.iter().map(|row| row.contribution0).sum::<f64>() / n;
        let mean_gain1 = rows.iter().map(|row| row.contribution1).sum::<f64>() / n;
        let panel_eps: Vec<f64> = rows
            .iter()
            .map(|row| (row.contribution0 + row.contribution1) / 2.0)
            .collect();
        let band_lo = panel_eps.iter().copied().fold(f64::INFINITY, f64::min);
        let band_hi = panel_eps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "ALLOCATION_BAND band={} representative={}x{} mean_visits={:.6} mean_contribution_gain0_mass={:.8} mean_contribution_gain1_mass={:.8} mean_contribution_error_mass={:.8} share_pct={:.3} panel_range_error_mass=[{:.8},{:.8}]",
            band.label(),
            band.representative().zero,
            band.representative().one,
            mean_visits,
            mean_gain0,
            mean_gain1,
            (mean_gain0 + mean_gain1) / 2.0,
            if mean_mass > 0.0 {
                100.0 * (mean_gain0 + mean_gain1) / (2.0 * mean_mass)
            } else {
                0.0
            },
            band_lo,
            band_hi,
        );
    }
}

// ─── descriptive policy similarity ──────────────────────────────────────────

/// Compare two saved average strategies row by row on their shared info-set
/// key space. Statistics are UNWEIGHTED over table rows — a descriptive map of
/// where the policies differ, not a reach-weighted behavioral distance and not
/// an exploitability substitute.
fn run_compare_policies(args: &[String]) {
    const MAX_ACTS: usize = 8;
    const DEPTH_BINS: usize = 13;
    const PURE_THRESHOLD: f32 = 0.99;

    let a_path = parse_opt_str_flag(args, "--a").expect("compare-policies requires --a PATH");
    let b_path = parse_opt_str_flag(args, "--b").expect("compare-policies requires --b PATH");
    let remap_turnup = has_flag(args, "--remap-turnup");
    let reach_weighted = has_flag(args, "--reach-weighted");
    let reach_dealer = parse_u8_flag(args, "--dealer", 0);
    // Q6 instrument: dump the top-K most divergent matched rows with their
    // info-set metadata (instead of only aggregate stats). Ranked by
    // reach·TV when --reach-weighted, else by TV. `--dump-min-tv` bounds the
    // candidate pool so the collection stays cheap.
    let dump_k = parse_usize_flag(args, "--dump-divergent", 0);
    let dump_min_tv = parse_f64_flag(args, "--dump-min-tv", 0.05) as f32;
    let reach_rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else {
        TreeRules::Current
    };

    #[derive(Clone, Copy)]
    struct CompactRow {
        len: u8,
        depth: u8,
        actions: [u8; MAX_ACTS],
        probabilities: [f32; MAX_ACTS],
    }
    let compact = |row: &truco_solver::storage::StrategyRow| -> CompactRow {
        assert!(
            row.actions.len() <= MAX_ACTS,
            "row has {} actions; compare maximum is {MAX_ACTS}",
            row.actions.len()
        );
        let mut out = CompactRow {
            len: row.actions.len() as u8,
            depth: row.info_set.history.actions().len().min(DEPTH_BINS - 1) as u8,
            actions: [0; MAX_ACTS],
            probabilities: [0.0; MAX_ACTS],
        };
        for (i, (&action, &probability)) in row
            .actions
            .iter()
            .zip(row.average_strategy.iter())
            .enumerate()
        {
            out.actions[i] = action.to_u8();
            out.probabilities[i] = probability;
        }
        out
    };
    let argmax = |row: &CompactRow| -> Option<u8> {
        (0..row.len as usize)
            .max_by(|&i, &j| row.probabilities[i].total_cmp(&row.probabilities[j]))
            .map(|i| row.actions[i])
    };
    let max_prob = |row: &CompactRow| -> f32 {
        (0..row.len as usize)
            .map(|i| row.probabilities[i])
            .fold(0.0, f32::max)
    };

    let started = Instant::now();
    let mut a_rows: ahash::AHashMap<InfoSetKey, CompactRow> = ahash::AHashMap::new();
    let a_meta = stream_strategy_rows(Path::new(&a_path), |row| {
        a_rows.insert(row.info_set.key(), compact(&row));
    })
    .expect("stream --a strategy");
    let a_tc = a_meta.turnup_class;

    println!(
        "COMPARE_META a={} ({}x{} tc{}) b={} remap_turnup={} note=unweighted-table-rows",
        a_path, a_meta.score.0, a_meta.score.1, a_tc.blocked_plain_level, b_path, remap_turnup
    );

    struct RowPolicy<'a>(&'a ahash::AHashMap<InfoSetKey, CompactRow>);
    impl PolicyLookup for RowPolicy<'_> {
        fn action_probability(
            &self,
            key: InfoSetKey,
            action: AbstractAction,
            values: PolicyValueSource,
        ) -> Option<f64> {
            if values != PolicyValueSource::Average {
                return None;
            }
            let row = self.0.get(&key)?;
            let code = action.to_u8();
            (0..row.len as usize)
                .find(|&i| row.actions[i] == code)
                .map(|i| row.probabilities[i] as f64)
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    // Reach of policy A over its own supported tree: every decision node's
    // visit probability accrues to the acting player's info-set row, so one
    // pass weights both players' rows.
    let mut reach: ahash::AHashMap<InfoSetKey, f64> = ahash::AHashMap::new();
    if reach_weighted {
        let reach_started = Instant::now();
        let a_tc_full = a_meta.turnup_class;
        let deals = enumerate_deals(&a_tc_full);
        let policy = RowPolicy(&a_rows);
        let score = Score {
            zero: a_meta.score.0,
            one: a_meta.score.1,
        };
        for deal in &deals {
            let state = truco_solver::game_tree::TraversalState::from_deal_with_rules(
                reach_dealer,
                score.clone(),
                a_tc_full,
                deal,
                reach_rules,
            )
            .expect("reach traversal state");
            accumulate_reach(&state, deal.weight, &policy, &mut reach);
        }
        let total_reach: f64 = reach.values().sum();
        println!(
            "COMPARE_REACH rows_with_reach={} total_reach={:.6} deals={} dealer={} rules={:?} wall_s={:.3}",
            reach.len(),
            total_reach,
            deals.len(),
            reach_dealer,
            reach_rules,
            reach_started.elapsed().as_secs_f64(),
        );
    }

    let mut rows_b = 0u64;
    let mut matched = 0u64;
    let mut action_set_mismatch = 0u64;
    let mut argmax_agree = 0u64;
    let mut pure_both = 0u64;
    let mut pure_agree = 0u64;
    let mut tvs: Vec<f32> = Vec::new();
    let mut depth_count = [0u64; DEPTH_BINS];
    let mut depth_tv_sum = [0.0f64; DEPTH_BINS];
    let mut depth_agree = [0u64; DEPTH_BINS];
    let mut w_total = 0.0f64;
    let mut w_tv = 0.0f64;
    let mut w_agree = 0.0f64;
    let mut w_tv_le_1e3 = 0.0f64;
    let mut depth_w = [0.0f64; DEPTH_BINS];
    let mut depth_w_tv = [0.0f64; DEPTH_BINS];
    struct Divergent {
        score: f64,
        tv: f32,
        reach: f64,
        info_set: InfoSet,
        a: CompactRow,
        b: CompactRow,
    }
    let mut divergent: Vec<Divergent> = Vec::new();
    let b_meta = stream_strategy_rows(Path::new(&b_path), |row| {
        rows_b += 1;
        let key = if remap_turnup {
            let mut mapped = row.info_set.clone();
            mapped.turnup_class = a_tc;
            mapped.key()
        } else {
            row.info_set.key()
        };
        let Some(a_row) = a_rows.get(&key) else {
            return;
        };
        matched += 1;
        let b_row = compact(&row);

        let mut same_sets = a_row.len == b_row.len;
        if same_sets {
            let mut a_sorted: Vec<u8> = a_row.actions[..a_row.len as usize].to_vec();
            let mut b_sorted: Vec<u8> = b_row.actions[..b_row.len as usize].to_vec();
            a_sorted.sort_unstable();
            b_sorted.sort_unstable();
            same_sets = a_sorted == b_sorted;
        }
        if !same_sets {
            action_set_mismatch += 1;
        }

        // Total variation over the union of action codes (absent action = 0).
        let mut l1 = 0.0f64;
        for i in 0..a_row.len as usize {
            let b_p = (0..b_row.len as usize)
                .find(|&j| b_row.actions[j] == a_row.actions[i])
                .map_or(0.0, |j| b_row.probabilities[j]);
            l1 += (a_row.probabilities[i] as f64 - b_p as f64).abs();
        }
        for j in 0..b_row.len as usize {
            let in_a = (0..a_row.len as usize).any(|i| a_row.actions[i] == b_row.actions[j]);
            if !in_a {
                l1 += b_row.probabilities[j] as f64;
            }
        }
        let tv = (l1 / 2.0) as f32;
        tvs.push(tv);

        let agree = argmax(a_row) == argmax(&b_row);
        argmax_agree += agree as u64;
        if max_prob(a_row) > PURE_THRESHOLD && max_prob(&b_row) > PURE_THRESHOLD {
            pure_both += 1;
            pure_agree += agree as u64;
        }
        let depth = b_row.depth as usize;
        depth_count[depth] += 1;
        depth_tv_sum[depth] += tv as f64;
        depth_agree[depth] += agree as u64;
        if reach_weighted {
            let w = reach.get(&key).copied().unwrap_or(0.0);
            if w > 0.0 {
                w_total += w;
                w_tv += w * tv as f64;
                w_agree += w * agree as u64 as f64;
                if tv <= 1e-3 {
                    w_tv_le_1e3 += w;
                }
                depth_w[depth] += w;
                depth_w_tv[depth] += w * tv as f64;
            }
        }
        if dump_k > 0 && tv >= dump_min_tv {
            let row_reach = reach.get(&key).copied().unwrap_or(0.0);
            let score = if reach_weighted {
                row_reach * tv as f64
            } else {
                tv as f64
            };
            if score > 0.0 {
                divergent.push(Divergent {
                    score,
                    tv,
                    reach: row_reach,
                    info_set: row.info_set.clone(),
                    a: *a_row,
                    b: b_row,
                });
                if divergent.len() >= dump_k.saturating_mul(8).max(4096) {
                    divergent.sort_by(|x, y| y.score.total_cmp(&x.score));
                    divergent.truncate(dump_k);
                }
            }
        }
    })
    .expect("stream --b strategy");

    let rows_a = a_rows.len() as u64;
    println!(
        "COMPARE_SUMMARY rows_a={} rows_b={} matched={} only_a={} only_b={} action_set_mismatch={} b_score={}x{} b_tc={}",
        rows_a,
        rows_b,
        matched,
        rows_a - matched,
        rows_b - matched,
        action_set_mismatch,
        b_meta.score.0,
        b_meta.score.1,
        b_meta.turnup_class.blocked_plain_level,
    );
    if matched == 0 {
        println!("COMPARE_TV no matched rows");
        return;
    }

    tvs.sort_unstable_by(f32::total_cmp);
    let quantile = |q: f64| -> f32 { tvs[((tvs.len() - 1) as f64 * q).round() as usize] };
    let mean_tv: f64 = tvs.iter().map(|&tv| tv as f64).sum::<f64>() / tvs.len() as f64;
    let frac_below =
        |limit: f32| -> f64 { tvs.partition_point(|&tv| tv <= limit) as f64 / tvs.len() as f64 };
    println!(
        "COMPARE_TV mean={:.6} median={:.6} p90={:.6} p99={:.6} max={:.6} frac_le_1e-3={:.4} frac_le_1e-2={:.4} frac_le_5e-2={:.4} frac_le_1e-1={:.4} frac_le_2.5e-1={:.4} frac_le_5e-1={:.4}",
        mean_tv,
        quantile(0.5),
        quantile(0.9),
        quantile(0.99),
        quantile(1.0),
        frac_below(1e-3),
        frac_below(1e-2),
        frac_below(5e-2),
        frac_below(1e-1),
        frac_below(2.5e-1),
        frac_below(5e-1),
    );
    println!(
        "COMPARE_AGREEMENT argmax_agree={:.4} pure_both_frac={:.4} pure_argmax_agree={:.4}",
        argmax_agree as f64 / matched as f64,
        pure_both as f64 / matched as f64,
        if pure_both > 0 {
            pure_agree as f64 / pure_both as f64
        } else {
            0.0
        },
    );
    for depth in 0..DEPTH_BINS {
        if depth_count[depth] == 0 {
            continue;
        }
        println!(
            "COMPARE_DEPTH depth={} count={} mean_tv={:.6} argmax_agree={:.4}",
            depth,
            depth_count[depth],
            depth_tv_sum[depth] / depth_count[depth] as f64,
            depth_agree[depth] as f64 / depth_count[depth] as f64,
        );
    }
    if reach_weighted && w_total > 0.0 {
        println!(
            "COMPARE_WEIGHTED matched_reach={:.6} mean_tv={:.6} argmax_agree={:.4} frac_le_1e-3={:.4}",
            w_total,
            w_tv / w_total,
            w_agree / w_total,
            w_tv_le_1e3 / w_total,
        );
        for depth in 0..DEPTH_BINS {
            if depth_w[depth] <= 0.0 {
                continue;
            }
            println!(
                "COMPARE_WEIGHTED_DEPTH depth={} reach={:.6} mean_tv={:.6}",
                depth,
                depth_w[depth],
                depth_w_tv[depth] / depth_w[depth],
            );
        }
    }
    if dump_k > 0 {
        divergent.sort_by(|x, y| y.score.total_cmp(&x.score));
        divergent.truncate(dump_k);
        let mix = |row: &CompactRow| -> String {
            (0..row.len as usize)
                .map(|i| format!("{}:{:.4}", row.actions[i], row.probabilities[i]))
                .collect::<Vec<_>>()
                .join("|")
        };
        println!(
            "DIVERGENT_META k={} min_tv={} ranked_by={} format=hand/history-as-action-codes",
            divergent.len(),
            dump_min_tv,
            if reach_weighted { "reach*tv" } else { "tv" },
        );
        for (rank, d) in divergent.iter().enumerate() {
            let hand = d
                .info_set
                .starting_hand
                .iter()
                .map(|c| c.type_index().to_string())
                .collect::<Vec<_>>()
                .join(",");
            let history = d
                .info_set
                .history
                .actions()
                .iter()
                .map(|a| a.to_u8().to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "DIVERGENT_ROW rank={} score={:.8} tv={:.4} reach={:.8} player={:?} is_dealer={} depth={} hand={} history=[{}] a={} b={}",
                rank + 1,
                d.score,
                d.tv,
                d.reach,
                d.info_set.player,
                d.info_set.is_dealer,
                d.info_set.history.actions().len(),
                hand,
                history,
                mix(&d.a),
                mix(&d.b),
            );
        }
    }
    println!("COMPARE_DONE wall_s={:.3}", started.elapsed().as_secs_f64());
}

/// Purity/mixing statistics of one saved average strategy, unweighted over
/// table rows and (optionally) weighted by the policy's own on-path reach.
/// "Pure" means the row's maximum action probability exceeds the threshold.
/// Usage: export-bot-policy (--checkpoint PATH | --strategy PATH) --out FILE.tpb [--dealer 0|1]
///
/// Streams a solved profile's average strategy into the mmap-able bot-policy
/// artifact (`bot_policy.rs`) consumed by the live-match solver bot. One
/// invocation per (score, tc, dealer) profile; the printed JSON line is the
/// entry ops collect into the policy directory's manifest.json.
fn run_export_bot_policy(args: &[String]) {
    use smallvec::SmallVec;
    use truco_solver::bot_policy::{write_bot_policy, MAX_ACTIONS};
    use truco_solver::storage::{stream_checkpoint_policy_rows, StrategyRow};

    let out = parse_opt_str_flag(args, "--out").expect("export-bot-policy requires --out FILE");
    let strategy = parse_opt_str_flag(args, "--strategy");
    let checkpoint = parse_opt_str_flag(args, "--checkpoint");
    let dealer_flag = parse_opt_str_flag(args, "--dealer")
        .map(|s| s.parse::<u8>().expect("--dealer takes 0 or 1"));

    let started = Instant::now();
    type PolicyRow = (u64, SmallVec<[u8; 8]>, SmallVec<[f32; 8]>);
    let mut rows: Vec<PolicyRow> = Vec::new();
    let mut push_row = |row: StrategyRow| {
        debug_assert!(row.actions.len() <= MAX_ACTIONS);
        rows.push((
            row.info_set.key().0,
            row.actions.iter().map(|a| a.to_u8()).collect(),
            row.average_strategy.iter().copied().collect(),
        ));
    };

    let (score, tc, dealer, source) = match (&checkpoint, &strategy) {
        (Some(path), None) => {
            let meta = stream_checkpoint_policy_rows(Path::new(path), &mut push_row)
                .expect("stream --checkpoint");
            let dealer = dealer_flag
                .or(meta.dealer_filter)
                .expect("checkpoint has no dealer restriction; pass --dealer 0|1 explicitly");
            (
                meta.score,
                meta.turnup_class,
                dealer,
                format!("checkpoint:{path}"),
            )
        }
        (None, Some(path)) => {
            let meta =
                stream_strategy_rows(Path::new(path), &mut push_row).expect("stream --strategy");
            let dealer = dealer_flag.expect("strategy metadata has no dealer; pass --dealer 0|1");
            (
                meta.score,
                meta.turnup_class,
                dealer,
                format!("strategy:{path}"),
            )
        }
        _ => panic!("export-bot-policy requires exactly one of --checkpoint or --strategy"),
    };
    assert!(matches!(dealer, 0 | 1), "--dealer takes 0 or 1");

    let row_count = rows.len();
    let written =
        write_bot_policy(Path::new(&out), rows.into_iter()).expect("write bot policy artifact");
    assert_eq!(written as usize, row_count);

    let file_name = Path::new(&out)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| out.clone());
    println!(
        "{{\"score\":[{},{}],\"tc\":{},\"dealer\":{},\"file\":\"{}\",\"rows\":{},\"source\":\"{}\"}}",
        score.0, score.1, tc.blocked_plain_level, dealer, file_name, row_count, source
    );
    eprintln!(
        "export-bot-policy: {} rows -> {} in {:.1}s",
        row_count,
        out,
        started.elapsed().as_secs_f64()
    );
}

fn run_policy_stats(args: &[String]) {
    const MAX_ACTS: usize = 8;
    const DEPTH_BINS: usize = 13;

    let path = parse_opt_str_flag(args, "--path").expect("policy-stats requires --path FILE");
    let reach_weighted = has_flag(args, "--reach-weighted");
    let reach_dealer = parse_u8_flag(args, "--dealer", 0);
    let reach_rules = if has_flag(args, "--legacy-tree") {
        TreeRules::LegacyPreProofPrunes
    } else {
        TreeRules::Current
    };

    #[derive(Clone, Copy)]
    struct StatRow {
        len: u8,
        depth: u8,
        max_prob: f32,
        actions: [u8; MAX_ACTS],
        probabilities: [f32; MAX_ACTS],
    }
    let started = Instant::now();
    let mut rows: ahash::AHashMap<InfoSetKey, StatRow> = ahash::AHashMap::new();
    let meta = stream_strategy_rows(Path::new(&path), |row| {
        let mut out = StatRow {
            len: row.actions.len() as u8,
            depth: row.info_set.history.actions().len().min(DEPTH_BINS - 1) as u8,
            max_prob: 0.0,
            actions: [0; MAX_ACTS],
            probabilities: [0.0; MAX_ACTS],
        };
        for (i, (&action, &probability)) in row
            .actions
            .iter()
            .zip(row.average_strategy.iter())
            .enumerate()
        {
            out.actions[i] = action.to_u8();
            out.probabilities[i] = probability;
            out.max_prob = out.max_prob.max(probability);
        }
        rows.insert(row.info_set.key(), out);
    })
    .expect("stream --path strategy");

    let thresholds = [0.9f32, 0.99, 0.999];
    let total = rows.len() as f64;
    let mut unweighted = [0u64; 3];
    let mut mean_max = 0.0f64;
    let mut depth_count = [0u64; DEPTH_BINS];
    let mut depth_pure99 = [0u64; DEPTH_BINS];
    for row in rows.values() {
        mean_max += row.max_prob as f64;
        for (i, &t) in thresholds.iter().enumerate() {
            unweighted[i] += (row.max_prob > t) as u64;
        }
        depth_count[row.depth as usize] += 1;
        depth_pure99[row.depth as usize] += (row.max_prob > 0.99) as u64;
    }
    println!(
        "POLICY_STATS path={} score={}x{} tc{} rows={} mean_max_prob={:.4} pure_gt_0.9={:.4} pure_gt_0.99={:.4} pure_gt_0.999={:.4}",
        path,
        meta.score.0,
        meta.score.1,
        meta.turnup_class.blocked_plain_level,
        rows.len(),
        mean_max / total,
        unweighted[0] as f64 / total,
        unweighted[1] as f64 / total,
        unweighted[2] as f64 / total,
    );
    for depth in 0..DEPTH_BINS {
        if depth_count[depth] == 0 {
            continue;
        }
        println!(
            "POLICY_STATS_DEPTH depth={} rows={} pure_gt_0.99={:.4}",
            depth,
            depth_count[depth],
            depth_pure99[depth] as f64 / depth_count[depth] as f64,
        );
    }

    if reach_weighted {
        struct RowPolicy<'a>(&'a ahash::AHashMap<InfoSetKey, StatRow>);
        impl PolicyLookup for RowPolicy<'_> {
            fn action_probability(
                &self,
                key: InfoSetKey,
                action: AbstractAction,
                values: PolicyValueSource,
            ) -> Option<f64> {
                if values != PolicyValueSource::Average {
                    return None;
                }
                let row = self.0.get(&key)?;
                let code = action.to_u8();
                (0..row.len as usize)
                    .find(|&i| row.actions[i] == code)
                    .map(|i| row.probabilities[i] as f64)
            }
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        let mut reach: ahash::AHashMap<InfoSetKey, f64> = ahash::AHashMap::new();
        let deals = enumerate_deals(&meta.turnup_class);
        let score = Score {
            zero: meta.score.0,
            one: meta.score.1,
        };
        for deal in &deals {
            let state = truco_solver::game_tree::TraversalState::from_deal_with_rules(
                reach_dealer,
                score.clone(),
                meta.turnup_class,
                deal,
                reach_rules,
            )
            .expect("reach traversal state");
            accumulate_reach(&state, deal.weight, &RowPolicy(&rows), &mut reach);
        }
        let mut w_total = 0.0f64;
        let mut w_mean_max = 0.0f64;
        let mut w_pure = [0.0f64; 3];
        let mut w_depth = [0.0f64; DEPTH_BINS];
        let mut w_depth_pure99 = [0.0f64; DEPTH_BINS];
        for (key, &w) in &reach {
            let Some(row) = rows.get(key) else { continue };
            w_total += w;
            w_mean_max += w * row.max_prob as f64;
            for (i, &t) in thresholds.iter().enumerate() {
                if row.max_prob > t {
                    w_pure[i] += w;
                }
            }
            w_depth[row.depth as usize] += w;
            if row.max_prob > 0.99 {
                w_depth_pure99[row.depth as usize] += w;
            }
        }
        println!(
            "POLICY_STATS_WEIGHTED dealer={} rules={:?} total_reach={:.6} mean_max_prob={:.4} pure_gt_0.9={:.4} pure_gt_0.99={:.4} pure_gt_0.999={:.4}",
            reach_dealer,
            reach_rules,
            w_total,
            w_mean_max / w_total,
            w_pure[0] / w_total,
            w_pure[1] / w_total,
            w_pure[2] / w_total,
        );
        for depth in 0..DEPTH_BINS {
            if w_depth[depth] <= 0.0 {
                continue;
            }
            println!(
                "POLICY_STATS_WEIGHTED_DEPTH depth={} reach={:.6} pure_gt_0.99={:.4}",
                depth,
                w_depth[depth],
                w_depth_pure99[depth] / w_depth[depth],
            );
        }
    }
    println!(
        "POLICY_STATS_DONE wall_s={:.3}",
        started.elapsed().as_secs_f64()
    );
}

/// Accumulate policy-A reach over its own supported tree. Every decision
/// node's visit probability is added to the acting player's row weight.
fn accumulate_reach(
    state: &truco_solver::game_tree::TraversalState,
    weight: f64,
    policy: &dyn PolicyLookup,
    reach: &mut ahash::AHashMap<InfoSetKey, f64>,
) {
    if weight <= 0.0 || state.is_terminal() {
        return;
    }
    let info = state.current_info_set().expect("non-terminal info set");
    let actions = state.abstract_legal_actions().expect("legal actions");
    let key = info.key();
    *reach.entry(key).or_insert(0.0) += weight;
    let (probabilities, _) = projected_policy_row(
        policy,
        key,
        &actions,
        PolicyValueSource::Average,
        MissingPolicyFallback::AllExceptRaise,
        DominatedProjection::Remap,
    );
    for (&action, probability) in actions.iter().zip(probabilities) {
        if probability > 0.0 {
            let child = state
                .apply_abstract_action(action)
                .expect("apply reach action");
            accumulate_reach(&child, weight * probability, policy, reach);
        }
    }
}

// ─── solve-tc ────────────────────────────────────────────────────────────────

fn run_solve_tc(args: &[String]) {
    // Parse optional flags
    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 11 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let target = parse_f64_flag(args, "--eps", 0.01);
    let mut max_iters = parse_u64_flag(args, "--max-iters", u64::MAX);
    let max_deals = parse_opt_str_flag(args, "--max-deals").map(|value| {
        value
            .parse::<usize>()
            .expect("--max-deals must be a number")
    });
    if max_deals.is_some() {
        assert!(
            parse_opt_str_flag(args, "--checkpoint").is_some()
                && parse_opt_str_flag(args, "--data-dir").is_some(),
            "--max-deals is benchmark-only and requires explicit --checkpoint and --data-dir paths"
        );
        assert!(
            parse_opt_str_flag(args, "--resume").is_none(),
            "--max-deals cannot use --resume; use --warmstart-from for a regrets-only seed"
        );
    }
    // Convenience for a resumed refinement run whose starting iteration isn't
    // known ahead of time (e.g. warm-started tremble refinement): "run N MORE
    // iterations past whatever the checkpoint resumes at", resolved below once
    // the checkpoint's iteration is known. Takes precedence over --max-iters.
    let extra_iters = parse_opt_str_flag(args, "--extra-iters")
        .map(|v| v.parse::<u64>().expect("--extra-iters must be a number"));
    let expl_every = parse_u64_flag(args, "--expl-every", 10);
    let algo = parse_algorithm(args);
    let log_path = parse_str_flag(args, "--log", "results/solve-tc.log");
    let data_dir = parse_str_flag(args, "--data-dir", "solutions");

    // Optional single-dealer solve. The two dealer games share no info sets
    // (position is in the info set) and interact only via the match-value
    // table, so solving them in separate processes is exact — and each build
    // is roughly half the memory of a joint solve.
    let dealer_filter: Option<u8> = parse_opt_str_flag(args, "--dealer").map(|d| {
        let d: u8 = d.parse().expect("--dealer must be 0 or 1");
        assert!(d < 2, "--dealer must be 0 or 1");
        d
    });
    // Per-dealer artifacts get a ".dN" infix so a d0 and a d1 run don't clobber
    // each other (or a joint run's files).
    let dealer_suffix = dealer_filter
        .map(|d| format!(".d{}", d))
        .unwrap_or_default();

    // Checkpoint / time-budget flags.
    let time_budget = parse_opt_f64_flag(args, "--time-budget");
    let default_ckpt = format!(
        "{}/{}x{}/tc{}{}.ckpt.bin",
        data_dir, score.zero, score.one, tc_level, dealer_suffix
    );
    let checkpoint_path = parse_str_flag(args, "--checkpoint", &default_ckpt);
    let checkpoint_every = parse_f64_flag(args, "--checkpoint-every", 300.0);
    let resume_path = parse_opt_str_flag(args, "--resume");

    // Optional ε-tremble refinement pass (see `cfr::TrembleSchedule`). Off by
    // default (`--tremble-eps` unset or 0.0) so existing workflows are
    // untouched. `--tremble-eps-end` anneals toward a smaller ε by
    // `--max-iters`; defaults to `--tremble-eps` (constant tremble) when unset.
    let tremble_eps_start = parse_f64_flag(args, "--tremble-eps", 0.0);
    let tremble = if tremble_eps_start > 0.0 {
        let eps_end = parse_f64_flag(args, "--tremble-eps-end", tremble_eps_start);
        Some(cfr::TrembleSchedule {
            eps_start: tremble_eps_start,
            eps_end,
        })
    } else {
        None
    };
    let regret_pruning = parse_opt_f64_flag(args, "--regret-prune-threshold").map(|threshold| {
        cfr::RegretPruningConfig {
            warmup_iters: parse_u64_flag(args, "--regret-prune-warmup", 20),
            threshold: threshold as f32,
            revisit_every_rounds: parse_u64_flag(args, "--regret-prune-revisit", 10),
        }
    });

    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };

    // Match values for the score states this subgame can transition into. Only
    // 11x11 can run on the defaults (no continuations). Everything below —
    // including 11x10, whose fold continuation reads the dealer-exact
    // mv(11,11,·) = 0.556/0.444, NOT 0.5 — must load a table seeded with the
    // higher states (see `set-mv`).
    let mv = match parse_opt_str_flag(args, "--match-values") {
        Some(p) if Path::new(&p).exists() => match load_match_values(Path::new(&p)) {
            Ok(t) => {
                println!("loaded match values from {}", p);
                t
            }
            Err(e) => {
                eprintln!(
                    "warn: could not load match values {}: {}; using fresh",
                    p, e
                );
                MatchValueTable::new()
            }
        },
        Some(p) => {
            eprintln!("warn: --match-values {} not found; using fresh", p);
            MatchValueTable::new()
        }
        None => MatchValueTable::new(),
    };

    let algo_label = algo_label_of(&algo);

    // Resume from an existing checkpoint when requested and present, after
    // validating that it matches the requested run.
    let mut resume: Option<(truco_solver::strategy::StrategyTable, u64)> = None;
    if let Some(rpath) = &resume_path {
        let path = Path::new(rpath);
        if path.exists() {
            match load_checkpoint(path) {
                Ok((table, meta)) => {
                    if meta.score != (score.zero, score.one)
                        || meta.turnup_class != tc
                        || meta.algo != algo_label
                        || meta.dealer_filter != dealer_filter
                    {
                        eprintln!(
                            "WARNING: checkpoint {} does not match this run \
                             (ckpt score={:?} tc={} algo={} dealer={:?} vs requested \
                             score=({},{}) tc={} algo={} dealer={:?}); ignoring it.",
                            path.display(),
                            meta.score,
                            meta.turnup_class.blocked_plain_level,
                            meta.algo,
                            meta.dealer_filter,
                            score.zero,
                            score.one,
                            tc_level,
                            algo_label,
                            dealer_filter,
                        );
                    } else {
                        println!(
                            "Resuming from checkpoint {} at iteration {} ({} info sets).",
                            path.display(),
                            meta.iteration,
                            meta.num_info_sets
                        );
                        if let Some(extra) = extra_iters {
                            max_iters = meta.iteration + extra;
                            println!(
                                "--extra-iters {} -> max_iters = {} (resumed at {})",
                                extra, max_iters, meta.iteration
                            );
                        }
                        resume = Some((table, meta.iteration));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "WARNING: failed to load checkpoint {}: {}; starting fresh.",
                        path.display(),
                        e
                    );
                }
            }
        } else {
            eprintln!(
                "WARNING: --resume {} does not exist; starting fresh.",
                path.display()
            );
        }
    }

    // Optional warm-start from a structurally related solved state's checkpoint
    // (e.g. 11x11) — transfers its card-play strategy + regrets into this solve.
    let warmstart_checkpoint = match parse_opt_str_flag(args, "--warmstart-from") {
        Some(p) if Path::new(&p).exists() => {
            println!(
                "warm-start from {} (disk-backed; source released before CFR)",
                p
            );
            Some(PathBuf::from(p))
        }
        Some(p) => {
            eprintln!("warn: --warmstart-from {} not found; ignoring", p);
            None
        }
        None => None,
    };
    let warmstart_cross_turnup = parse_bool_flag(args, "--warmstart-profile-transfer")
        || parse_bool_flag(args, "--warmstart-cross-turnup");
    assert!(
        !warmstart_cross_turnup || warmstart_checkpoint.is_some(),
        "--warmstart-profile-transfer requires --warmstart-from CKPT"
    );

    // The usable average-strategy artifact is written by the solver directly
    // from its dense accumulators (no end-of-solve StrategyTable rebuild, so
    // dense+table never coexist at peak).
    let strategy_path = Path::new(&data_dir)
        .join(format!("{}x{}", score.zero, score.one))
        .join(format!("tc{}{}.bin", tc_level, dealer_suffix));

    let config = cfr::SolveConfig {
        max_iters,
        target_expl: target,
        algorithm: algo.clone(),
        expl_every,
        time_budget_secs: time_budget,
        checkpoint_path: Some(std::path::PathBuf::from(&checkpoint_path)),
        checkpoint_every_secs: Some(checkpoint_every),
        warmstart_source: None,
        warmstart_checkpoint,
        warmstart_iter: 0,
        warmstart_same_band: false,
        warmstart_cross_turnup,
        accept_policy: None,
        resume_average_reset: false,
        dealer_filter,
        max_deals,
        jobs: parse_usize_flag(args, "--jobs", 1),
        tree_cache: parse_opt_str_flag(args, "--tree-cache").map(std::path::PathBuf::from),
        prebuilt_override: None,
        tremble,
        regret_pruning,
        strategy_output: Some(strategy_path.clone()),
        skip_return_table: true,
        tree_rules: if has_flag(args, "--asymmetric-raise-prune") {
            truco_solver::game_tree::TreeRules::AsymmetricRaisePrune
        } else {
            truco_solver::game_tree::TreeRules::Current
        },
    };

    // Ensure results dir exists
    if let Some(parent) = std::path::Path::new(&log_path).parent() {
        fs::create_dir_all(parent).ok();
    }

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("cannot open log file");

    let algo_name = format!("{:?}", algo);
    let budget_label = time_budget
        .map(|b| format!("{}s", b))
        .unwrap_or_else(|| "none".to_string());
    let dealer_label = dealer_filter
        .map(|d| d.to_string())
        .unwrap_or_else(|| "both".to_string());
    let tremble_label = tremble
        .map(|ts| format!("{}->{}", ts.eps_start, ts.eps_end))
        .unwrap_or_else(|| "off".to_string());
    let regret_pruning_label = regret_pruning
        .map(|pruning| {
            format!(
                "threshold={} warmup={} revisit_rounds={}",
                pruning.threshold, pruning.warmup_iters, pruning.revisit_every_rounds
            )
        })
        .unwrap_or_else(|| "off".to_string());
    let header = format!(
        "\n=== solve-tc  score={}x{}  tc={}  dealer={}  deals={}  eps={}  max_iters={}  algo={}  \
         expl_every={}  time_budget={}  checkpoint={}  checkpoint_every={}s  tremble={}  regret_pruning={} ===",
        score.zero,
        score.one,
        tc_level,
        dealer_label,
        max_deals
            .map(|limit| limit.to_string())
            .unwrap_or_else(|| "all".to_string()),
        target,
        max_iters,
        algo_name,
        expl_every,
        budget_label,
        checkpoint_path,
        checkpoint_every,
        tremble_label,
        regret_pruning_label,
    );
    println!("{}", header);
    writeln!(log_file, "{}", header).ok();

    let col_header = format!(
        "{:>6}  {:>14}  {:>10}  {:>10}",
        "iter", "exploitability", "secs", "total_s"
    );
    println!("{}", col_header);
    writeln!(log_file, "{}", col_header).ok();

    let wall_start = Instant::now();

    let (table, stats) = cfr::solve_until(
        score.clone(),
        tc,
        &config,
        &mv,
        resume,
        |iter, expl, secs| {
            let total_s = wall_start.elapsed().as_secs_f64();
            let line = format!(
                "{:>6}  {:>14.6}  {:>10.1}  {:>10.1}",
                iter, expl, secs, total_s
            );
            println!("{}", line);
            std::io::stdout().flush().ok();
            writeln!(log_file, "{}", line).ok();
            log_file.flush().ok();
        },
    );

    // The solver wrote the average-strategy solution directly from dense
    // accumulators (see `strategy_output` above).
    drop(table);
    if strategy_path.exists() {
        let msg = format!("Saved strategy to {}", strategy_path.display());
        println!("{}", msg);
        writeln!(log_file, "{}", msg).ok();
    } else {
        eprintln!(
            "WARNING: dense strategy save did not produce {}",
            strategy_path.display()
        );
    }

    // Sidecar with this (score, TC) per-dealer game values (P0, ±1 space) so the
    // orchestrator can aggregate the per-TC values into per-dealer match values
    // without parsing logs. Joint run: "v_d0 v_d1" on the first line, the
    // dealer-averaged value on the second. Single-dealer run (the sidecar gets
    // the same ".dN" infix as the strategy): just that dealer's value — the
    // other tree was never built, so its value would be meaningless.
    if let Some((v_d0, v_d1)) = stats.game_value_per_dealer {
        let gv_path = strategy_path.with_extension("gv");
        let contents = match dealer_filter {
            Some(0) => format!("{}\n", v_d0),
            Some(_) => format!("{}\n", v_d1),
            None => format!("{} {}\n{}\n", v_d0, v_d1, (v_d0 + v_d1) / 2.0),
        };
        if let Err(e) = fs::write(&gv_path, contents) {
            eprintln!("WARNING: failed to write game value sidecar: {}", e);
        } else {
            match dealer_filter {
                Some(d) => println!(
                    "Wrote game value v_d{}={:.6} to {}",
                    d,
                    if d == 0 { v_d0 } else { v_d1 },
                    gv_path.display()
                ),
                None => println!(
                    "Wrote game values v_d0={:.6} v_d1={:.6} to {}",
                    v_d0,
                    v_d1,
                    gv_path.display()
                ),
            }
        }
    }

    let footer = format!(
        "\nDone in {:.1}s | {} iters | final expl = {:.6} | {} info sets | {:.0} MB",
        stats.total_duration_secs,
        stats.iterations,
        stats.exploitability.unwrap_or(f64::NAN),
        stats.num_info_sets,
        stats.estimated_memory_bytes as f64 / 1_048_576.0,
    );
    println!("{}", footer);
    writeln!(log_file, "{}", footer).ok();
}

// ─── eval-ckpt ───────────────────────────────────────────────────────────────

/// Evaluate a saved checkpoint's average strategy under the EXACT (legal
/// per-info-set) best response: exploitability + per-dealer game values.
/// Usage: eval-ckpt --ckpt PATH [--score SxS] [--tc N] [--dealer D]
///        [--match-values PATH] [--max-deals N]
fn run_eval_ckpt(args: &[String]) {
    let ckpt_path = parse_str_flag(args, "--ckpt", "");
    assert!(!ckpt_path.is_empty(), "--ckpt PATH is required");
    let (table, meta) = load_checkpoint(Path::new(&ckpt_path)).expect("load checkpoint");
    println!(
        "checkpoint: score={}x{} tc={} algo={} iter={} info_sets={} dealer={:?}",
        meta.score.0,
        meta.score.1,
        meta.turnup_class.blocked_plain_level,
        meta.algo,
        meta.iteration,
        meta.num_info_sets,
        meta.dealer_filter,
    );

    let score = parse_score_flag(
        args,
        "--score",
        Score {
            zero: meta.score.0,
            one: meta.score.1,
        },
    );
    let tc = TurnupClass {
        blocked_plain_level: parse_u8_flag(args, "--tc", meta.turnup_class.blocked_plain_level),
    };
    let dealer_filter: Option<u8> = parse_opt_str_flag(args, "--dealer")
        .map(|d| d.parse().expect("--dealer must be 0 or 1"))
        .or(meta.dealer_filter);
    let mv = match parse_opt_str_flag(args, "--match-values") {
        Some(p) => load_match_values(Path::new(&p)).expect("load match values"),
        None => MatchValueTable::new(),
    };

    let mut deals = truco_solver::cfr::enumerate_deals_pub(&tc);
    if let Some(limit) = parse_opt_str_flag(args, "--max-deals") {
        let limit: usize = limit.parse().expect("--max-deals");
        if limit > 0 && limit < deals.len() {
            let n = deals.len();
            deals = (0..limit).map(|i| deals[i * n / limit].clone()).collect();
        }
        let w: f64 = deals.iter().map(|d| d.weight).sum();
        for d in &mut deals {
            d.weight /= w;
        }
    }
    println!(
        "building trees: {} deals, dealer={:?} ...",
        deals.len(),
        dealer_filter
    );
    let prebuilt =
        truco_solver::game_tree::build_all_trees_with_dealer(&score, tc, &deals, dealer_filter)
            .expect("tree build");

    let br0 = cfr::best_response_value(&prebuilt, &table, &score, &mv, 0);
    let br1 = cfr::best_response_value(&prebuilt, &table, &score, &mv, 1);
    let (v_d0, v_d1) = cfr::compute_game_value_per_dealer(&prebuilt, &table, &score, &mv);
    println!("exact BR EV: p0={:+.6}  p1={:+.6}", br0, br1);
    println!(
        "exact exploitability ((br0+br1)/2): {:.6}",
        (br0 + br1) / 2.0
    );
    println!(
        "game value per dealer (p0, ±1): d0={:+.6}  d1={:+.6}",
        v_d0, v_d1
    );
}

// ─── solve-asym ────────────────────────────────────────────────────────────────

/// Dealer-exact "freeze-accept + equity policy-iteration" solver for an
/// asymmetric mão-de-onze state (e.g. 11x10).
///
/// At such a state the score-11 player (the DECIDER) acts first with the
/// 2-action set {AcceptEleven, FoldEleven}. That info set shares an abstraction
/// key across both dealer trees, so it cannot be learned per-dealer by CFR
/// alone. We freeze it externally per dealer (keyed by the decider's hand) and
/// alternate:
///   (1) an inner CFR card-play solve with the accept set frozen, then
///   (2) recompute the accept set from each hand's "value if you accept" equity
///       versus its fixed fold value F[d].
/// Iterate until the accept sets are stable.
fn run_solve_asym(args: &[String]) {
    use std::collections::HashSet;
    use truco_solver::abstraction::{enumerate_deals, AbstractHand};
    use truco_solver::cfr::{extract_accept_equities, AcceptPolicy};
    use truco_solver::game_tree::build_all_trees;

    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 10 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let rounds = parse_u64_flag(args, "--rounds", 8);
    let inner_eps = parse_f64_flag(args, "--inner-eps", 0.005);
    let inner_time_budget = parse_opt_f64_flag(args, "--inner-time-budget");
    let inner_max_iters = parse_u64_flag(args, "--inner-max-iters", u64::MAX);
    let expl_every = parse_u64_flag(args, "--expl-every", 25);
    let algo = parse_algorithm(args);
    let out_dir = parse_str_flag(args, "--out", "solutions/asym");

    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };

    // Identify the decider: the player whose score is 11 (MATCH_TARGET - 1).
    let at_eleven_p0 = score.zero == truco_engine::MATCH_TARGET - 1;
    let at_eleven_p1 = score.one == truco_engine::MATCH_TARGET - 1;
    if !(at_eleven_p0 || at_eleven_p1) {
        eprintln!(
            "error: solve-asym needs an asymmetric mão-de-onze score (one side at {}); got {}x{}",
            truco_engine::MATCH_TARGET - 1,
            score.zero,
            score.one
        );
        std::process::exit(2);
    }
    if at_eleven_p0 && at_eleven_p1 {
        eprintln!(
            "error: solve-asym is for ASYMMETRIC states; {}x{} is symmetric (use solve-tc)",
            score.zero, score.one
        );
        std::process::exit(2);
    }
    // The decider is the score-11 player (p0 if at_eleven_p0, else p1).
    let decider: u8 = if at_eleven_p0 { 0 } else { 1 };

    // Match values: must already contain the post-fold continuation states.
    let mv = match parse_opt_str_flag(args, "--match-values") {
        Some(p) if Path::new(&p).exists() => match load_match_values(Path::new(&p)) {
            Ok(t) => {
                println!("loaded match values from {}", p);
                t
            }
            Err(e) => {
                eprintln!("error: could not load --match-values {}: {}", p, e);
                std::process::exit(2);
            }
        },
        Some(p) => {
            eprintln!("error: --match-values {} not found", p);
            std::process::exit(2);
        }
        None => {
            eprintln!("error: --match-values PATH is required for solve-asym");
            std::process::exit(2);
        }
    };

    // Per-dealer fold value F[d] for the DECIDER, expressed as the decider's own
    // win probability. The decider folding gives the OTHER player +1; the next
    // hand is then dealt by the other player (dealer flips), so we look up at
    // `1 - d`. mv.get returns player 0's win prob, so we flip for a p1 decider.
    let fold_score = if decider == 0 {
        // p0 (decider) folds -> p1 +1
        Score {
            zero: score.zero,
            one: (score.one + 1).min(truco_engine::MATCH_TARGET),
        }
    } else {
        Score {
            zero: (score.zero + 1).min(truco_engine::MATCH_TARGET),
            one: score.one,
        }
    };
    let mut fold_decider: [f64; 2] = [0.0, 0.0];
    for d in 0..2u8 {
        let p0_win = mv.get(fold_score.zero, fold_score.one, 1 - d);
        fold_decider[d as usize] = if decider == 0 { p0_win } else { 1.0 - p0_win };
    }
    println!(
        "decider = p{}  fold-score = {}x{}  F[d0]={:.6}  F[d1]={:.6}  (decider win% on fold)",
        decider, fold_score.zero, fold_score.one, fold_decider[0], fold_decider[1]
    );

    // Build the per-deal trees once for the between-round equity extraction.
    let deals = enumerate_deals(&tc);
    println!(
        "building {} trees for {}x{} tc{} ...",
        deals.len() * 2,
        score.zero,
        score.one,
        tc_level
    );
    let prebuilt = build_all_trees(&score, tc, &deals).expect("build trees");

    // Optional warm-start source (e.g. an 11x11 checkpoint): transferred on the
    // FIRST inner solve so round-0 equities reflect the equilibrium card play.
    let (warmstart_source, warmstart_iter) = match parse_opt_str_flag(args, "--warmstart-from") {
        Some(p) if Path::new(&p).exists() => match load_checkpoint(Path::new(&p)) {
            Ok((t, m)) => {
                println!("warm-start from {} (source iter {})", p, m.iteration);
                (Some(Arc::new(t)), m.iteration)
            }
            Err(e) => {
                eprintln!("warn: --warmstart-from {} load failed: {}; ignoring", p, e);
                (None, 0)
            }
        },
        Some(p) => {
            eprintln!("warn: --warmstart-from {} not found; ignoring", p);
            (None, 0)
        }
        None => (None, 0),
    };

    // Decide whether a hand accepts given its per-dealer equity (decider win%).
    let accepts = |equity_decider: f64, d: usize| equity_decider > fold_decider[d];

    // Initialize the accept set to "accept all" (membership = accept). The first
    // inner solve + equity extraction produces the first data-driven set.
    let mut accept_policy = AcceptPolicy::default();
    {
        // Seed accept-all: every decider hand present in the trees accepts.
        for entry in &prebuilt.entries {
            for (d, tree) in [
                (0usize, &entry.tree_dealer_0),
                (1usize, &entry.tree_dealer_1),
            ] {
                if tree.is_empty() {
                    continue;
                }
                if let truco_solver::game_tree::NodeView::Player {
                    table_idx, edges, ..
                } = tree.view(0)
                {
                    let is_eleven = edges.len() == 2
                        && edges.iter().any(|e| {
                            e.action() == truco_solver::info_set::AbstractAction::AcceptEleven
                        });
                    if is_eleven {
                        let hand = prebuilt.info_sets[table_idx as usize]
                            .1
                            .starting_hand
                            .clone();
                        accept_policy.accept[d].insert(hand);
                    }
                }
            }
        }
    }
    println!(
        "round 0 init: accept-all  |A_d0|={}  |A_d1|={}",
        accept_policy.accept[0].len(),
        accept_policy.accept[1].len()
    );

    fs::create_dir_all(&out_dir).expect("create --out dir");

    let mut final_table = None;
    let mut final_stats = None;
    let mut converged_round = None;
    // Carries the previous round's solved table so the next round refines the
    // card play from it (exact-key resume on the same tree) instead of cold.
    let mut carry: Option<(truco_solver::strategy::StrategyTable, u64)> = None;

    for r in 0..rounds {
        let policy_arc = Arc::new(accept_policy.clone());
        let config = cfr::SolveConfig {
            max_iters: inner_max_iters,
            target_expl: inner_eps,
            algorithm: algo.clone(),
            expl_every,
            time_budget_secs: inner_time_budget,
            checkpoint_path: None,
            checkpoint_every_secs: None,
            // Warm-start only the first inner solve; subsequent rounds keep
            // refining from a freshly built table (cheap relative to the solve).
            warmstart_source: if r == 0 {
                warmstart_source.clone()
            } else {
                None
            },
            warmstart_checkpoint: None,
            warmstart_iter: if r == 0 { warmstart_iter } else { 0 },
            warmstart_same_band: false,
            warmstart_cross_turnup: false,
            accept_policy: Some(policy_arc),
            // Rounds 1+ resume the regrets from the previous round but reset the
            // average + iteration: the accept set changed, so the previous average
            // is stale and CFR+'s t^gamma weighting (at a high resumed iteration)
            // would keep it stuck. A fresh average over the warm current strategy
            // converges to the new filtered-range equilibrium.
            resume_average_reset: r > 0,
            dealer_filter: None,
            max_deals: None,
            jobs: 1,
            tree_cache: None,
            prebuilt_override: None,
            tremble: None,
            regret_pruning: None,
            strategy_output: None,
            skip_return_table: false,
            tree_rules: truco_solver::game_tree::TreeRules::Current,
        };

        println!(
            "\n=== round {}/{}  inner-eps={}  budget={:?}s  |A_d0|={}  |A_d1|={} ===",
            r + 1,
            rounds,
            inner_eps,
            inner_time_budget,
            accept_policy.accept[0].len(),
            accept_policy.accept[1].len()
        );

        // Round 0 warm-starts from 11x11 (via config.warmstart_source); later
        // rounds resume from the previous round's table (same tree, exact keys).
        let resume = carry.take();
        let (table, stats) = cfr::solve_until(
            score.clone(),
            tc,
            &config,
            &mv,
            resume,
            |iter, expl, secs| {
                println!("  iter {:>5}  expl={:.6}  ({:.1}s)", iter, expl, secs);
                std::io::stdout().flush().ok();
            },
        );

        // Recompute accept set from each hand's "value if you accept" equity.
        let equities = extract_accept_equities(&prebuilt, &table, &score, &mv, decider);

        let mut next = AcceptPolicy::default();
        let mut changed = 0usize;
        for (d, equities_d) in equities.iter().enumerate() {
            for (hand, &p0_win) in equities_d {
                // equities are P0 win%; convert to the decider's win%.
                let equity_decider = if decider == 0 { p0_win } else { 1.0 - p0_win };
                if accepts(equity_decider, d) {
                    next.accept[d].insert(hand.clone());
                }
            }
            // Count hands whose accept/fold membership flipped this round.
            let prev: &HashSet<AbstractHand> = &accept_policy.accept[d];
            let cur: &HashSet<AbstractHand> = &next.accept[d];
            let union: HashSet<&AbstractHand> = prev.union(cur).collect();
            for h in union {
                if prev.contains(h) != cur.contains(h) {
                    changed += 1;
                }
            }
        }

        println!(
            "  -> |A'_d0|={}  |A'_d1|={}  changed={}  inner_expl={:.6}",
            next.accept[0].len(),
            next.accept[1].len(),
            changed,
            stats.exploitability.unwrap_or(f64::NAN)
        );
        // Per-dealer value of the decider this round (mv we're computing). Round 0
        // (accept-all) must equal the 11x11 per-dealer value by construction.
        if let Some((v_d0, v_d1)) = stats.game_value_per_dealer {
            let (w0, w1) = if decider == 0 {
                ((v_d0 + 1.0) / 2.0, (v_d1 + 1.0) / 2.0)
            } else {
                ((1.0 - v_d0) / 2.0, (1.0 - v_d1) / 2.0)
            };
            println!(
                "     decider value: p0-deals {:+.4} (win {:.4}) | p1-deals {:+.4} (win {:.4})",
                v_d0, w0, v_d1, w1
            );
        }
        // Localize the residual exploitability: BR_p0 (decider best-responds, incl.
        // deviating the accept) vs BR_p1 (opponent best-responds to the frozen play).
        let br0 = cfr::best_response_value_clairvoyant(
            &prebuilt,
            &table,
            &score,
            &mv,
            Some(&accept_policy),
            0,
        );
        let br1 = cfr::best_response_value_clairvoyant(
            &prebuilt,
            &table,
            &score,
            &mv,
            Some(&accept_policy),
            1,
        );
        println!(
            "     BR_p0(decider)={:+.4}  BR_p1(opp)={:+.4}  -> expl={:.4}",
            br0,
            br1,
            (br0 + br1) / 2.0
        );

        let stable =
            next.accept[0] == accept_policy.accept[0] && next.accept[1] == accept_policy.accept[1];

        accept_policy = next;
        let iters = stats.iterations;
        final_stats = Some(stats);

        if stable {
            converged_round = Some(r + 1);
            println!("  accept sets stable after round {} — stopping", r + 1);
            final_table = Some(table);
            break;
        }
        // Refine from this round's solved card play next round.
        carry = Some((table, iters));
    }

    let table = final_table
        .or_else(|| carry.map(|(t, _)| t))
        .expect("at least one round runs");
    let stats = final_stats.expect("at least one round runs");
    let policy_arc = Arc::new(accept_policy.clone());

    // Per-dealer exploitability and best-response values under the final policy.
    let expl = cfr::compute_exploitability_with_accept_policy(
        &prebuilt,
        &table,
        &score,
        &mv,
        Some(&policy_arc),
    );

    // Per-dealer game value mv(score, d): convert ±1 P0 values to P0 win prob.
    let (v_d0, v_d1) = cfr::compute_game_value_per_dealer_with_accept_policy(
        &prebuilt,
        &table,
        &score,
        &mv,
        Some(&policy_arc),
    );
    let mv_p0_d0 = (v_d0 + 1.0) / 2.0;
    let mv_p0_d1 = (v_d1 + 1.0) / 2.0;

    println!(
        "\n=== solve-asym result  {}x{}  tc{} ===",
        score.zero, score.one, tc_level
    );
    if let Some(r) = converged_round {
        println!("  converged after {} round(s)", r);
    } else {
        println!("  did NOT converge within {} round(s)", rounds);
    }
    println!(
        "  final |A_d0|={}  |A_d1|={}",
        accept_policy.accept[0].len(),
        accept_policy.accept[1].len()
    );
    println!(
        "  exploitability (avg over dealers, vs frozen accept) = {:.6}",
        expl
    );
    println!(
        "  game value P0: p0-deals {:+.6} (win% {:.6}) | p1-deals {:+.6} (win% {:.6})",
        v_d0, mv_p0_d0, v_d1, mv_p0_d1
    );

    // Write accept sets + per-dealer mv to JSON.
    let json = build_asym_json(
        &score,
        tc_level,
        decider,
        &fold_decider,
        &accept_policy,
        mv_p0_d0,
        mv_p0_d1,
        expl,
        converged_round,
    );
    let json_path = Path::new(&out_dir).join(format!(
        "{}x{}_tc{}_accept.json",
        score.zero, score.one, tc_level
    ));
    match fs::write(&json_path, json) {
        Ok(()) => println!("  wrote accept sets + mv -> {}", json_path.display()),
        Err(e) => eprintln!("WARNING: failed to write {}: {}", json_path.display(), e),
    }

    // Write the average-strategy solution.
    let strat_path =
        Path::new(&out_dir).join(format!("{}x{}_tc{}.bin", score.zero, score.one, tc_level));
    let strat_meta = SolvedStateMeta {
        score: (score.zero, score.one),
        turnup_class: tc,
        iterations: stats.iterations,
        num_info_sets: stats.num_info_sets,
    };
    match save_strategy(&strat_path, &table, strat_meta) {
        Ok(()) => println!("  wrote strategy -> {}", strat_path.display()),
        Err(e) => eprintln!("WARNING: failed to save strategy: {}", e),
    }
}

/// Serialize the accept sets + per-dealer match values to a JSON string.
#[allow(clippy::too_many_arguments)]
fn build_asym_json(
    score: &Score,
    tc_level: u8,
    decider: u8,
    fold_decider: &[f64; 2],
    policy: &truco_solver::cfr::AcceptPolicy,
    mv_p0_d0: f64,
    mv_p0_d1: f64,
    exploitability: f64,
    converged_round: Option<u64>,
) -> String {
    use serde_json::json;

    let hands_for = |d: usize| -> Vec<String> {
        let mut v: Vec<String> = policy.accept[d]
            .iter()
            .map(|h| format!("{:?}", h.as_slice()))
            .collect();
        v.sort();
        v
    };

    let doc = json!({
        "score": [score.zero, score.one],
        "turnup_class": tc_level,
        "decider_player": decider,
        "fold_value_decider": { "dealer_0": fold_decider[0], "dealer_1": fold_decider[1] },
        "match_value_p0": { "dealer_0": mv_p0_d0, "dealer_1": mv_p0_d1 },
        "exploitability": exploitability,
        "converged_round": converged_round,
        "accept_counts": { "dealer_0": policy.accept[0].len(), "dealer_1": policy.accept[1].len() },
        "accept_hands": { "dealer_0": hands_for(0), "dealer_1": hands_for(1) },
    });
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
}

// ─── dealer-advantage ────────────────────────────────────────────────────────

/// Compute the pé (dealer) advantage from a solved strategy file, by evaluating
/// the average strategy separately on the dealer-0 and dealer-1 trees.
/// Usage: dealer-advantage <strategy.bin>
fn run_dealer_advantage(args: &[String]) {
    let path = args
        .iter()
        .skip(2)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: solve dealer-advantage <strategy.bin>");
            std::process::exit(2);
        });

    let (table, meta) = load_strategy(Path::new(&path)).expect("load strategy");
    let score = Score {
        zero: meta.score.0,
        one: meta.score.1,
    };
    let tc = meta.turnup_class;
    eprintln!(
        "Rebuilding trees for {}x{} tc{} to evaluate per-dealer value...",
        score.zero, score.one, tc.blocked_plain_level
    );
    let deals = truco_solver::abstraction::enumerate_deals(&tc);
    let prebuilt =
        truco_solver::game_tree::build_all_trees(&score, tc, &deals).expect("build trees");
    let mv = MatchValueTable::new();
    let (v_d0, v_d1) = cfr::compute_game_value_per_dealer(&prebuilt, &table, &score, &mv);

    let p0_win_when_p0_deals = (v_d0 + 1.0) / 2.0;
    let p0_win_when_p1_deals = (v_d1 + 1.0) / 2.0;
    let dealer_win = (v_d0 + 1.0) / 2.0; // dealer == p0 in the dealer-0 tree
    println!(
        "\n=== pé (dealer) advantage  {}x{}  tc{} ===",
        score.zero, score.one, tc.blocked_plain_level
    );
    println!(
        "  p0 value when p0 deals (±1): {:+.6}  -> p0 win% {:.4}",
        v_d0, p0_win_when_p0_deals
    );
    println!(
        "  p0 value when p1 deals (±1): {:+.6}  -> p0 win% {:.4}",
        v_d1, p0_win_when_p1_deals
    );
    println!("  avg game value (±1):         {:+.6}", (v_d0 + v_d1) / 2.0);
    println!(
        "  DEALER win% = {:.4}   (advantage over even = {:+.4})",
        dealer_win,
        dealer_win - 0.5
    );
}

/// Extract the average-strategy `.bin` (the format `export-teacher
/// --strategy` reads) out of a full-state `--resume` checkpoint. A
/// checkpoint carries strictly more (regret accumulators, resume metadata);
/// this just drops what `export-teacher` doesn't need. Written for
/// retrofits (e.g. the 2026-07-11 tremble refinement) that upload
/// `refined.ckpt.bin` without also uploading the plain `.bin` that a normal
/// solve produces alongside it.
///
/// Usage: checkpoint-to-strategy --checkpoint PATH --out PATH
fn run_checkpoint_to_strategy(args: &[String]) {
    let ckpt_path = parse_str_flag(args, "--checkpoint", "solutions/tc0.ckpt.bin");
    let out_path = parse_str_flag(args, "--out", "solutions/tc0.bin");

    let (table, ckpt_meta) = load_checkpoint(Path::new(&ckpt_path)).expect("load --checkpoint");
    println!(
        "checkpoint: {} ({} info sets, {} iters, algo={})",
        ckpt_path, ckpt_meta.num_info_sets, ckpt_meta.iteration, ckpt_meta.algo
    );
    let meta = SolvedStateMeta {
        score: ckpt_meta.score,
        turnup_class: ckpt_meta.turnup_class,
        iterations: ckpt_meta.iteration,
        num_info_sets: ckpt_meta.num_info_sets,
    };
    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    save_strategy(Path::new(&out_path), &table, meta).expect("save --out");
    println!("wrote {}", out_path);
}

// ─── set-mv ──────────────────────────────────────────────────────────────────

/// Build or patch a match_values.bin by setting explicit interior states.
/// Usage: set-mv --mv-out PATH [--mv-in PATH] --set s0:s1:dealer:value [--set ...]
/// Starts from --mv-in if given (else a fresh table with terminals + 0.5
/// interior), applies each --set, and saves to --mv-out. Used by the
/// orchestrator to seed lower-state solves with the higher states' values.
/// `dealer` is 0 or 1 and selects which dealer cell of (s0, s1) to set.
/// Export the compact teacher artifact (.teach) for a solved (score, tc,
/// dealer): average strategy + one-shot-deviation Q values + reach masses,
/// per info set in table_idx order. See teacher_export.rs for semantics.
///
/// Usage: export-teacher --score SxS --tc N --dealer D
///        [--strategy PATH] [--match-values PATH] [--tree-cache DIR]
///        [--out PATH] [--band-meta PATH] [--max-deals N]
fn run_export_teacher(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 11 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let dealer: u8 = parse_str_flag(args, "--dealer", "0")
        .parse()
        .expect("--dealer must be 0 or 1");
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };

    let default_strategy = format!(
        "solutions/{}x{}/tc{}.d{}.bin",
        score.zero, score.one, tc_level, dealer
    );
    let strategy_path = parse_str_flag(args, "--strategy", &default_strategy);
    let mv_path = parse_str_flag(args, "--match-values", "solutions/match_values.bin");
    let out_path = parse_str_flag(
        args,
        "--out",
        &format!(
            "teacher/{}x{}/tc{}.d{}.teach",
            score.zero, score.one, tc_level, dealer
        ),
    );
    let tree_cache = parse_opt_str_flag(args, "--tree-cache").map(std::path::PathBuf::from);
    let band_meta = parse_opt_str_flag(args, "--band-meta");
    let max_deals = parse_opt_str_flag(args, "--max-deals").map(|v| v.parse::<usize>().unwrap());
    let allow_residue = parse_bool_flag(args, "--allow-residue");

    let match_values = load_match_values(Path::new(&mv_path)).expect("load --match-values table");
    let (table, meta) =
        truco_solver::storage::load_strategy(Path::new(&strategy_path)).expect("load --strategy");
    println!(
        "strategy: {} ({} info sets, {} iters)",
        strategy_path, meta.num_info_sets, meta.iterations
    );

    let mut deals = truco_solver::abstraction::enumerate_deals(&tc);
    if let Some(limit) = max_deals {
        cfr::subsample_deals(&mut deals, limit);
    }
    let prebuilt =
        cfr::load_or_build_trees(tree_cache.as_deref(), &score, tc, &deals, Some(dealer));

    let strategies = truco_solver::teacher_export::strategies_by_table_idx(&prebuilt, &table)
        .expect("strategy lookup");
    let t0 = Instant::now();
    let (data, game_value) = truco_solver::teacher_export::compute_teacher_data(
        &prebuilt,
        &strategies,
        dealer,
        &score,
        &match_values,
    );
    println!(
        "teacher pass: {} info sets in {:.1}s | game value (P0, +/-1) = {:+.6}",
        data.n_info_sets(),
        t0.elapsed().as_secs_f64(),
        game_value
    );
    // --allow-residue downgrades this from a hard export-blocking assertion to a
    // loud warning, for known-less-converged solves (e.g. an early scout run)
    // that are expected to fail the regression guard calibrated against the
    // clean tc0/d0 column. It still measures and prints the same numbers.
    let residue = if allow_residue {
        truco_solver::teacher_export::summarize_q_gap_mass_with_prob_cap(
            &data,
            truco_solver::teacher_export::RESIDUE_ASSERT_Q_GAP_PP,
            truco_solver::teacher_export::PURIFY_MAX_PROB,
        )
    } else {
        truco_solver::teacher_export::assert_q_gap_residue(
            &data,
            truco_solver::teacher_export::RESIDUE_ASSERT_Q_GAP_PP,
            truco_solver::teacher_export::RESIDUE_ASSERT_MAX_INFO_SET_MASS,
        )
    };
    if allow_residue
        && residue.max_info_set_mass
            > truco_solver::teacher_export::RESIDUE_ASSERT_MAX_INFO_SET_MASS
    {
        println!(
            "WARNING: --allow-residue set; residue {:.4}% exceeds the normal {:.4}% export guard. \
             This export is not certification-quality; label it provisional.",
            residue.max_info_set_mass * 100.0,
            truco_solver::teacher_export::RESIDUE_ASSERT_MAX_INFO_SET_MASS * 100.0
        );
    }
    println!(
        "residue assertion: max {:.4}% info-set mass on p<{:.2}% actions above {:.1}pp Q-gap (limit {:.4}%; touched {}/{})",
        residue.max_info_set_mass * 100.0,
        residue.max_action_prob * 100.0,
        residue.q_gap_pp,
        truco_solver::teacher_export::RESIDUE_ASSERT_MAX_INFO_SET_MASS * 100.0,
        residue.touched_info_sets,
        residue.info_sets
    );
    // Cross-check against the solve's own .gv sidecar when present.
    let gv_path = Path::new(&strategy_path).with_extension("gv");
    if let Ok(contents) = fs::read_to_string(&gv_path) {
        if let Some(v) = contents
            .split_whitespace()
            .next()
            .and_then(|t| t.parse::<f64>().ok())
        {
            let delta = (v - game_value).abs();
            println!("gv sidecar: {:+.6} (|delta| = {:.2e})", v, delta);
            assert!(
                delta < 1e-6,
                "teacher game value disagrees with the solve's gv sidecar"
            );
        }
    }

    let sig = truco_solver::treepack::band_sig_hash(&score, tc_level, Some(dealer));
    truco_solver::teacher_export::save_teach(
        Path::new(&out_path),
        &data,
        &score,
        tc_level,
        dealer,
        sig,
    )
    .expect("save .teach");
    println!("wrote {}", out_path);

    if let Some(meta_path) = band_meta {
        truco_solver::teacher_export::save_band_meta(Path::new(&meta_path), &prebuilt.info_sets)
            .expect("save band meta");
        println!("wrote {}", meta_path);
    }
}

/// Rewrite a band-meta sidecar from the tree cache alone — for when the
/// strategy .bin behind a .teach is gone but its meta was overwritten by a
/// later run against a different band (the meta is band-scoped, not
/// score-scoped, so `export-teacher --band-meta` calls clobber each other
/// across bands).
///
/// Usage: export-band-meta --score SxS --tc N --dealer D
///        [--tree-cache DIR] --out PATH
fn run_export_band_meta(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 11 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let out_path = parse_str_flag(args, "--out", "teacher/band.meta");
    let tree_cache = parse_opt_str_flag(args, "--tree-cache").map(std::path::PathBuf::from);

    let deals = truco_solver::abstraction::enumerate_deals(&tc);
    let prebuilt =
        cfr::load_or_build_trees(tree_cache.as_deref(), &score, tc, &deals, Some(dealer));
    truco_solver::teacher_export::save_band_meta(Path::new(&out_path), &prebuilt.info_sets)
        .expect("save band meta");
    println!(
        "wrote {} ({} info sets)",
        out_path,
        prebuilt.info_sets.len()
    );
}

/// Sweep one or more raw `.teach` files and print per-info-set mass assigned to
/// actions with large Q gaps. This is the step-0 diagnostic behind plan 74.
///
/// Usage: audit-teach-residue [--teach-dir DIR | --teach PATH ...]
///        [--thresholds 1,5,20]
fn run_audit_teach_residue(args: &[String]) {
    let thresholds: Vec<f64> = parse_str_flag(args, "--thresholds", "1,5,20")
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric threshold"))
        .collect();
    let mut paths = parse_multi_str_flag(args, "--teach")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        let dir = parse_str_flag(args, "--teach-dir", "teacher");
        paths = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read --teach-dir {}: {}", dir, e))
            .map(|entry| entry.expect("read dir entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "teach"))
            .collect();
    }
    paths.sort();
    assert!(!paths.is_empty(), "no .teach files found");

    let mut totals: Vec<truco_solver::teacher_export::QGapMassSummary> = thresholds
        .iter()
        .map(|&threshold| {
            truco_solver::teacher_export::QGapMassSummary::empty(
                threshold,
                truco_solver::teacher_export::PURIFY_MAX_PROB,
            )
        })
        .collect();

    println!(
        "residue action cap: p < {:.2}%",
        truco_solver::teacher_export::PURIFY_MAX_PROB * 100.0
    );
    println!(
        "{:<28} {:>7} {:>11} {:>11} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "file", "gap_pp", "touched", "max_mass", ">=0.1%", ">=0.5%", ">=1%", ">=3%", "max_gap"
    );
    for path in &paths {
        let data = truco_solver::teacher_export::load_teach(path, None)
            .unwrap_or_else(|e| panic!("load {}: {}", path.display(), e));
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<teach>");
        for (i, &threshold) in thresholds.iter().enumerate() {
            let summary = truco_solver::teacher_export::summarize_q_gap_mass_with_prob_cap(
                &data,
                threshold,
                truco_solver::teacher_export::PURIFY_MAX_PROB,
            );
            print_residue_summary(file, &summary);
            totals[i].merge(&summary);
        }
    }
    println!();
    for summary in &totals {
        print_residue_summary("TOTAL", summary);
    }
}

fn print_residue_summary(file: &str, summary: &truco_solver::teacher_export::QGapMassSummary) {
    println!(
        "{:<28} {:>7.1} {:>5}/{:<5} {:>10.4}% {:>10} {:>10} {:>10} {:>10} {:>9.2}",
        file,
        summary.q_gap_pp,
        summary.touched_info_sets,
        summary.info_sets,
        summary.max_info_set_mass * 100.0,
        summary.info_sets_ge_0_1pct,
        summary.info_sets_ge_0_5pct,
        summary.info_sets_ge_1pct,
        summary.info_sets_ge_3pct,
        summary.max_q_gap_touched_pp
    );
}

fn round_prob(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn round_q(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// Emit chart-ready JSON for the Study lab from a .teach file + its
/// band-meta sidecar (no tree rebuild). Groups info sets into decision
/// nodes by (observed history, player, is_dealer) and includes every node
/// up to --max-depth observed actions. Histories and actions use the
/// stable AbstractAction u8 codec (see info_set.rs).
///
/// Each action carries, alongside `p`/`raw_p`/`q`:
///   - `pts`: the one-shot-deviation HAND-point value of taking this action
///     now and following the exported strategy afterward — acting player's
///     points minus opponent's points for THIS hand only (±1/±3/±6/±9/±12,
///     including mão-de-onze fold/accept and truco-fold outcomes at
///     whatever stake is in force when the hand ends). Positive means the
///     acting player nets points on average; sign is from the ACTING
///     player's perspective at this node, not p0's. Unlike `q` (match
///     equity, ±1), `pts` does not depend on score/dealer/match-value
///     lookups — a hand's point value is self-contained — so it is NOT part
///     of the exact-BR certificate (`certificate.pts_certified` is always
///     `false`): the certificate measures exploitability in match-equity
///     space, and a raw-point-maximizing deviation is not the objective
///     either player is actually playing.
///
/// Each row also carries, alongside `hand`/`w`:
///   - `own_reach`: the acting player's OWN counterfactual reach to this row
///     (deal-weight × product of the acting player's own average-strategy
///     action probabilities along the history to this node) — the same
///     σ̄-vs-σ̄ traversal that produces `w` (`opp_reach`), just attributed to
///     the acting player instead of their opponent. Lets a consumer flag
///     rows the acting player's OWN equilibrium play would rarely reach
///     (distinct from `w`, which flags rows the OPPONENT rarely steers
///     into). Additive field — existing `study-chart/v1` consumers ignore
///     unknown keys.
///
/// Usage: export-chart --teach PATH --band-meta PATH --out PATH
///        [--min-depth N] [--max-depth N] [--score SxS] [--tc N] [--dealer D]
///        [--certify --tree-cache DIR --match-values PATH] [--br-gaps]
///        [--br-gap-out PATH]
///
/// `--min-depth` skips shallower nodes so deep continuation files (trick 2
/// onward) can ship separately from the eagerly-fetched trick-1 file.
///
/// `--br-gaps` adds a per-row `"br"` field: a genuine adversarial
/// best-response value against the SHIPPED (purified) profile, computed
/// independently of this node's own (possibly still noisy) `q` — see
/// `RESEARCH_NARRATIVE.md` 2026-07-11 "Per-infoset best-response gap".
/// Implies building the same trees as `--certify` (shared if both are set;
/// when both are set the purified best-response pass runs once, not twice —
/// see `cfr::best_response_full_from_action_probs`).
/// `gap = br_value - eq_value` is `null` where `weight == 0` (no line
/// consistent with the fixed opponent profile reaches that info set).
///
/// `--br-gap-out PATH` (requires `--br-gaps`) additionally writes a
/// full-tree, non-depth-filtered `BrGapRecord` table (see plan 75,
/// `plans/75-per-infoset-solution-quality-br-gap.md`) via
/// `teacher_export::save_br_gaps`. The chart JSON's `"br"` row field stays
/// windowed to `[--min-depth, --max-depth]` like every other row field; this
/// artifact is the whole game, for callers that want per-infoset solution
/// quality independent of which chart window a viewer happens to have open.
fn run_export_chart(args: &[String]) {
    use std::collections::BTreeMap;
    use truco_solver::info_set::{AbstractAction, InfoSet};

    let teach_path = parse_str_flag(args, "--teach", "teacher/out.teach");
    let meta_path = parse_str_flag(args, "--band-meta", "teacher/band.meta");
    let out_path = parse_str_flag(args, "--out", "teacher/chart.json");
    let min_depth = parse_u64_flag(args, "--min-depth", 0) as usize;
    let max_depth = parse_u64_flag(args, "--max-depth", 2) as usize;
    let score = parse_score_flag(args, "--score", Score { zero: 0, one: 0 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let certify = parse_bool_flag(args, "--certify");
    // Per-info-set best-response gap (see RESEARCH_NARRATIVE.md 2026-07-11,
    // "Per-infoset best-response gap"): a rigorous, adversarial measurement
    // of off-equilibrium quality, distinct from self-loss/own-reach. Implies
    // building the same PrebuiltTrees as --certify (cheap to share if both
    // are requested).
    let br_gaps = parse_bool_flag(args, "--br-gaps");
    let br_gap_out = parse_opt_str_flag(args, "--br-gap-out");
    assert!(
        br_gap_out.is_none() || br_gaps,
        "--br-gap-out requires --br-gaps"
    );
    let tree_cache = parse_opt_str_flag(args, "--tree-cache").map(PathBuf::from);
    let mv_path = parse_str_flag(args, "--match-values", "solutions/match_values.bin");

    let expect_sig = truco_solver::treepack::band_sig_hash(&score, tc_level, Some(dealer));
    let data = truco_solver::teacher_export::load_teach(Path::new(&teach_path), Some(expect_sig))
        .expect("load .teach");
    let raw = fs::read(&meta_path).expect("read band meta");
    let info_sets: Vec<(u64, InfoSet, Vec<AbstractAction>)> =
        bincode::deserialize(&raw).expect("decode band meta");
    assert_eq!(
        info_sets.len(),
        data.n_info_sets(),
        "band meta / teach size mismatch"
    );

    let purify_config = truco_solver::teacher_export::PurificationConfig::default();
    let raw_residue = truco_solver::teacher_export::summarize_q_gap_mass_with_prob_cap(
        &data,
        truco_solver::teacher_export::RESIDUE_ASSERT_Q_GAP_PP,
        truco_solver::teacher_export::PURIFY_MAX_PROB,
    );
    let raw_strategies = truco_solver::teacher_export::action_probs_from_teacher(&data);
    let (purified_strategies, purification) =
        truco_solver::teacher_export::purify_teacher_probs(&data, purify_config);

    let (raw_eps, purified_eps, br_map, purified_teach) = if certify || br_gaps {
        let tc = TurnupClass {
            blocked_plain_level: tc_level,
        };
        let match_values =
            load_match_values(Path::new(&mv_path)).expect("load --match-values table");
        let deals = truco_solver::abstraction::enumerate_deals(&tc);
        let prebuilt =
            cfr::load_or_build_trees(tree_cache.as_deref(), &score, tc, &deals, Some(dealer));

        let raw_eps = if certify {
            Some(cfr::compute_exploitability_from_action_probs(
                &prebuilt,
                &raw_strategies,
                &score,
                &match_values,
            ))
        } else {
            None
        };

        // Best response against the SHIPPED (purified) profile — the one a
        // real opponent actually faces. One combined call per player (rather
        // than a separate exploitability call plus a separate gaps call):
        // `best_response_full_from_action_probs` returns both the aggregate
        // AND the per-info-set detail from a single backward-induction pass,
        // so requesting both --certify and --br-gaps doesn't pay for that
        // pass twice. Each call only returns that player's own info sets, so
        // the two per-info-set maps are disjoint and safe to merge.
        let (purified_eps, br_map) = if certify || br_gaps {
            let mut total = 0.0;
            let mut map: std::collections::HashMap<u32, cfr::InfoSetBestResponse> =
                std::collections::HashMap::new();
            for br_player in [0u8, 1u8] {
                let full = cfr::best_response_full_from_action_probs(
                    &prebuilt,
                    &purified_strategies,
                    &score,
                    &match_values,
                    br_player,
                );
                total += full.total;
                if br_gaps {
                    for r in full.per_info_set {
                        map.insert(r.table_idx, r);
                    }
                }
            }
            (Some(total / 2.0), if br_gaps { Some(map) } else { None })
        } else {
            (None, None)
        };

        if let (Some(raw_eps), Some(purified_eps)) = (raw_eps, purified_eps) {
            assert!(
                purified_eps <= raw_eps + 1e-9,
                "purified exploitability {:.12} exceeds raw {:.12}",
                purified_eps,
                raw_eps
            );
        }
        let purified_eps = if certify { purified_eps } else { None };

        // eq_value (below) must compare against the SAME fixed opponent that
        // br_value was computed against, i.e. continuation under
        // `purified_strategies` for both players. `.teach`'s stored `q`
        // (`data.q_of`) was computed via `strategies_by_table_idx`'s RAW,
        // unpurified average strategy (see teacher_export::compute_teacher_data
        // / strategies_by_table_idx) — using it here would silently compare
        // br_value against a different opponent and can make gap go negative
        // even for a genuine best response. Recompute q under the purified
        // profile specifically for this purpose; the row's own shipped `"q"`
        // field stays sourced from `data` (raw), unchanged.
        let purified_teach = if br_gaps {
            let (d, _game_value) = truco_solver::teacher_export::compute_teacher_data(
                &prebuilt,
                &purified_strategies,
                dealer,
                &score,
                &match_values,
            );
            Some(d)
        } else {
            None
        };

        // Full-tree BR-gap artifact (plan 75 step 3): unlike the chart JSON's
        // per-row `"br"` field, NOT filtered by `--min-depth`/`--max-depth` —
        // `br_map`/`purified_teach` already cover every info set the
        // best-response pass reached, so there is no compute reason to
        // narrow this to whatever window the chart happens to be exporting.
        if let Some(out_path) = &br_gap_out {
            let map = br_map.as_ref().expect("br_map set when --br-gap-out given");
            let pdata = purified_teach
                .as_ref()
                .expect("purified_teach set when --br-gap-out given");
            let mut records: Vec<truco_solver::teacher_export::BrGapRecord> = map
                .values()
                .filter(|r| r.weight > 0.0 && r.br_value.is_finite())
                .map(|r| {
                    let idx = r.table_idx as usize;
                    let pq = pdata.q_of(idx);
                    let probs = &purified_strategies[idx];
                    let eq_value: f64 = probs
                        .iter()
                        .zip(pq.iter())
                        .map(|(&p, &qi)| p * qi as f64)
                        .sum();
                    truco_solver::teacher_export::BrGapRecord {
                        table_idx: r.table_idx,
                        br_value: r.br_value as f32,
                        eq_value: eq_value as f32,
                        gap: (r.br_value - eq_value) as f32,
                        weight: r.weight as f32,
                    }
                })
                .collect();
            records.sort_by_key(|r| r.table_idx);
            println!(
                "writing {} br-gap records ({} table_idx entries, {} filtered as unreachable) to {}",
                records.len(),
                map.len(),
                map.len() - records.len(),
                out_path
            );
            truco_solver::teacher_export::save_br_gaps(
                Path::new(out_path),
                &records,
                &score,
                tc_level,
                dealer,
                expect_sig,
            )
            .expect("write --br-gap-out artifact");
        }

        (raw_eps, purified_eps, br_map, purified_teach)
    } else {
        (None, None, None, None)
    };

    // node key: (history u8s, player, is_dealer) -> rows
    let mut nodes: BTreeMap<(Vec<u8>, u8, bool), Vec<serde_json::Value>> = BTreeMap::new();
    for (idx, (_, info, actions)) in info_sets.iter().enumerate() {
        let hist = info.history.actions();
        if hist.len() < min_depth || hist.len() > max_depth {
            continue;
        }
        let hist_u8: Vec<u8> = hist.iter().map(|a| a.to_u8()).collect();
        let mut hand: Vec<u8> = info
            .starting_hand
            .iter()
            .map(|c| c.type_index() as u8)
            .collect();
        hand.sort_unstable_by(|a, b| b.cmp(a)); // strongest first
        let raw_probs = data.probs_of(idx);
        let probs = &purified_strategies[idx];
        let q = data.q_of(idx);
        let pts = data.pts_of(idx);
        let acts: Vec<serde_json::Value> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                serde_json::json!({
                    "c": a.to_u8(),
                    "p": round_prob(probs[i]),
                    "raw_p": round_prob(raw_probs[i] as f64),
                    "q": round_q(q[i] as f64),
                    "pts": round_q(pts[i] as f64),
                })
            })
            .collect();
        let mut row = serde_json::json!({
            "table_idx": idx,
            "hand": hand,
            "w": data.opp_reach[idx],
            "own_reach": data.own_reach[idx],
            "actions": acts,
        });
        if let (Some(map), Some(pdata)) = (&br_map, &purified_teach) {
            // eq_value: the value under continuation of the SAME (purified)
            // profile that br_value was best-responded against — NOT
            // `.teach`'s stored `q`, which reflects RAW-strategy continuation
            // (see `purified_teach` above) and would compare br_value against
            // a different opponent than it was actually computed against.
            // br_value: a genuine adversarial best response to the fixed
            // (purified) opponent, computed independently of this node's own
            // (possibly still noisy) q. gap = br_value - eq_value.
            let pq = pdata.q_of(idx);
            let eq_value: f64 = probs
                .iter()
                .zip(pq.iter())
                .map(|(&p, &qi)| p * qi as f64)
                .sum();
            let entry = map.get(&(idx as u32));
            let (br_value, weight) = entry
                .map(|r| (r.br_value, r.weight))
                .unwrap_or((f64::NAN, 0.0));
            row["br"] = serde_json::json!({
                "eq_value": round_q(eq_value),
                "br_value": if br_value.is_finite() { Some(round_q(br_value)) } else { None },
                "gap": if br_value.is_finite() { Some(round_q(br_value - eq_value)) } else { None },
                "weight": weight,
            });
        }
        nodes
            .entry((hist_u8, info.player, info.is_dealer))
            .or_default()
            .push(row);
    }

    let node_values: Vec<serde_json::Value> = nodes
        .into_iter()
        .map(|((hist, player, is_dealer), mut rows)| {
            rows.sort_by_key(|r| r["hand"].to_string());
            serde_json::json!({
                "history": hist,
                "player": player,
                "is_dealer": is_dealer,
                "rows": rows,
            })
        })
        .collect();

    let doc = serde_json::json!({
        "format": "study-chart/v1",
        "score": [score.zero, score.one],
        "tc": tc_level,
        "dealer": dealer,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "br_gaps": br_gaps,
        "certificate": {
            "format": "study-purification-certificate/v1",
            "certified": certify,
            "raw_eps": raw_eps,
            "purified_eps": purified_eps,
            "mass_removed": purification.mass_removed,
            "max_info_set_mass_removed": purification.max_info_set_mass_removed,
            "max_qgap_touched_pp": purification.max_q_gap_touched_pp,
            "touched_info_sets": purification.touched_info_sets,
            "actions_zeroed": purification.actions_zeroed,
            "purify_max_prob": purification.max_prob,
            "purify_min_qgap_pp": purification.min_q_gap_pp,
            "assert_qgap_pp": raw_residue.q_gap_pp,
            "assert_max_info_set_mass": truco_solver::teacher_export::RESIDUE_ASSERT_MAX_INFO_SET_MASS,
            "raw_max_info_set_mass_above_assert_qgap": raw_residue.max_info_set_mass,
            "raw_touched_info_sets_above_assert_qgap": raw_residue.touched_info_sets,
            // `pts` (hand-point EV) is NOT covered by the exact-BR certificate
            // above: that machinery measures exploitability in match-equity
            // space, and "best response to raw hand points" isn't the
            // objective either player is actually playing (match equity is
            // concave near the match target), so it isn't a meaningful
            // quantity to certify. `pts` is a diagnostic derived from the
            // same certified/purified strategy profile, not an independently
            // certified value.
            "pts_certified": false,
        },
        "nodes": node_values,
    });
    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&out_path, serde_json::to_string(&doc).unwrap()).expect("write chart json");
    println!(
        "wrote {} nodes to {} (purified {} actions; raw_eps={:?} purified_eps={:?})",
        doc["nodes"].as_array().unwrap().len(),
        out_path,
        purification.actions_zeroed,
        raw_eps,
        purified_eps
    );
}

fn run_set_mv(args: &[String]) {
    let mv_out = parse_str_flag(args, "--mv-out", "solutions/match_values.bin");
    let mut mv = match parse_opt_str_flag(args, "--mv-in") {
        Some(p) if Path::new(&p).exists() => {
            load_match_values(Path::new(&p)).expect("load --mv-in")
        }
        Some(p) => {
            eprintln!("warn: --mv-in {} not found; starting fresh", p);
            MatchValueTable::new()
        }
        None => MatchValueTable::new(),
    };

    let mut applied = 0;
    for i in 0..args.len() {
        if args[i] == "--set" {
            let spec = args.get(i + 1).expect("--set needs s0:s1:dealer:value");
            let parts: Vec<&str> = spec.split(':').collect();
            assert_eq!(
                parts.len(),
                4,
                "--set expects s0:s1:dealer:value, got {}",
                spec
            );
            let s0: u8 = parts[0].parse().expect("s0");
            let s1: u8 = parts[1].parse().expect("s1");
            let dealer: u8 = parts[2].parse().expect("dealer");
            assert!(dealer < 2, "dealer must be 0 or 1, got {}", dealer);
            let v: f64 = parts[3].parse().expect("value");
            mv.set(s0, s1, dealer, v);
            println!("set mv({}, {}, dealer={}) = {:.6}", s0, s1, dealer, v);
            applied += 1;
        }
    }

    if let Some(parent) = Path::new(&mv_out).parent() {
        fs::create_dir_all(parent).ok();
    }
    save_match_values(Path::new(&mv_out), &mv).expect("save --mv-out");
    println!("wrote {} ({} states set)", mv_out, applied);
}

// ─── compare ─────────────────────────────────────────────────────────────────

fn run_mccfr_bench(args: &[String]) {
    use truco_solver::game_tree::PolicyLookup;

    let score = parse_score_flag(args, "--score", Score { zero: 0, one: 0 });
    let tc = TurnupClass {
        blocked_plain_level: parse_u8_flag(args, "--tc", 0),
    };
    let dealer: u8 = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let requested_samples = parse_u64_flag(args, "--samples", 100_000);
    let batch_size = parse_usize_flag(args, "--batch-size", 32);
    assert!(batch_size > 0, "--batch-size must be positive");
    let total_batches = requested_samples.div_ceil(batch_size as u64);
    let eval_every_batches =
        parse_u64_flag(args, "--eval-every-batches", (total_batches / 4).max(1));
    assert!(eval_every_batches > 0);
    let eval_deals = parse_usize_flag(args, "--eval-deals", 200);
    assert!(eval_deals > 0, "--eval-deals must be positive");
    let max_train_deals = parse_opt_str_flag(args, "--max-train-deals").map(|value| {
        value
            .parse::<usize>()
            .expect("--max-train-deals must be a number")
    });
    let seed_regret_mass = parse_f64_flag(args, "--seed-regret-mass", 10.0);
    let rng_seed = parse_u64_flag(args, "--rng-seed", 1);

    let mv = match parse_opt_str_flag(args, "--match-values") {
        Some(path) => load_match_values(Path::new(&path)).expect("load --match-values"),
        None => MatchValueTable::new(),
    };
    let seed_checkpoint = parse_opt_str_flag(args, "--seed-checkpoint");
    let seed_strategy = parse_opt_str_flag(args, "--seed-strategy");
    assert!(
        seed_checkpoint.is_none() || seed_strategy.is_none(),
        "use only one MCCFR seed source"
    );
    let (seed_policy, seed_label): (Option<Box<dyn PolicyLookup>>, String) =
        if let Some(path) = seed_checkpoint {
            let (table, meta) = load_checkpoint(Path::new(&path)).expect("load --seed-checkpoint");
            assert_eq!(meta.turnup_class, tc, "seed turnup class mismatch");
            (
                Some(Box::new(table)),
                format!("checkpoint:{}x{}", meta.score.0, meta.score.1),
            )
        } else if let Some(path) = seed_strategy {
            let (policy, meta) =
                load_compact_average_policy(Path::new(&path)).expect("load --seed-strategy");
            assert_eq!(meta.turnup_class, tc, "seed turnup class mismatch");
            (
                Some(Box::new(policy)),
                format!("strategy:{}x{}", meta.score.0, meta.score.1),
            )
        } else {
            (None, "none".to_string())
        };

    let mut deals = cfr::enumerate_deals_pub(&tc);
    if let Some(limit) = max_train_deals {
        cfr::subsample_deals(&mut deals, limit);
    }
    let mut eval_deal_set = cfr::enumerate_deals_pub(&tc);
    if eval_deals > 0 && eval_deals < eval_deal_set.len() {
        let all = eval_deal_set;
        let n = all.len();
        let stride = n / eval_deals;
        let offset = (stride / 2).max(1);
        eval_deal_set = (0..eval_deals)
            .map(|i| all[(i * n / eval_deals + offset) % n].clone())
            .collect();
        let total_weight: f64 = eval_deal_set.iter().map(|deal| deal.weight).sum();
        for deal in &mut eval_deal_set {
            deal.weight /= total_weight;
        }
    }
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut table = truco_solver::strategy::StrategyTable::new();
    let start = Instant::now();
    let mut evaluation_secs = 0.0;
    let mut completed_batches = 0u64;

    println!(
        "mccfr-bench score={}x{} tc={} dealer={} train_deals={} requested_samples={} batch_size={} batches={} eval_deals={} eval_panel=mid-stride seed={} seed_mass={} rng_seed={}",
        score.zero,
        score.one,
        tc.blocked_plain_level,
        dealer,
        deals.len(),
        requested_samples,
        batch_size,
        total_batches,
        eval_deals,
        seed_label,
        seed_regret_mass,
        rng_seed,
    );

    while completed_batches < total_batches {
        let chunk = eval_every_batches.min(total_batches - completed_batches);
        let stats = cfr::run_mccfr_minibatch_chunk(
            score.clone(),
            tc,
            chunk,
            batch_size,
            completed_batches,
            &mv,
            &mut rng,
            &deals,
            &mut table,
            seed_policy.as_deref(),
            seed_regret_mass,
            Some(dealer),
        );
        completed_batches += chunk;
        let samples = completed_batches * batch_size as u64;
        let eval_start = Instant::now();
        let exploitability = cfr::compute_exploitability_on_deals_with_dealer(
            &score,
            tc,
            &table,
            &mv,
            &eval_deal_set,
            Some(dealer),
        );
        let this_eval_secs = eval_start.elapsed().as_secs_f64();
        evaluation_secs += this_eval_secs;
        println!(
            "RESULT mode=mccfr-minibatch batch_size={} batches={} samples={} info_sets={} new_info_sets={} eval_deals={} exploitability={:.9} train_elapsed_s={:.3} eval_elapsed_s={:.3}",
            batch_size,
            completed_batches,
            samples,
            table.len(),
            stats.info_sets_after - stats.info_sets_before,
            eval_deals,
            exploitability,
            start.elapsed().as_secs_f64() - evaluation_secs,
            this_eval_secs,
        );
        std::io::stdout().flush().ok();
    }

    if let Some(path) = parse_opt_str_flag(args, "--out") {
        let meta = SolvedStateMeta {
            score: (score.zero, score.one),
            turnup_class: tc,
            iterations: completed_batches * batch_size as u64,
            num_info_sets: table.len(),
        };
        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent).ok();
        }
        save_strategy(Path::new(&path), &table, meta).expect("save --out");
        println!("wrote {}", path);
    }
}

fn run_compare(args: &[String]) {
    let max_iters = parse_u64_flag(args, "--iters", 100);
    let expl_every = parse_u64_flag(args, "--expl-every", 10);
    let log_path = parse_str_flag(args, "--log", "results/compare.log");

    let score = Score { zero: 11, one: 11 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let mv = MatchValueTable::new();

    fs::create_dir_all("results").ok();
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("cannot open log file");

    let header = format!(
        "\n=== compare  score=11x11  tc=0  iters={}  expl_every={} ===",
        max_iters, expl_every
    );
    println!("{}", header);
    writeln!(log_file, "{}", header).ok();

    let col_header = format!(
        "{:>6}  {:>14}  {:>10}  {:>8}",
        "iter", "exploitability", "wall_secs", "algo"
    );
    println!("{}", col_header);
    writeln!(log_file, "{}", col_header).ok();

    // ── CFR+ ──────────────────────────────────────────────────────────────
    println!("\n--- CFR+ ---");
    writeln!(log_file, "\n--- CFR+ ---").ok();

    let cfr_config = cfr::SolveConfig {
        max_iters,
        target_expl: 0.0, // run all iters
        algorithm: cfr::CfrAlgorithm::CfrPlus,
        expl_every,
        ..Default::default()
    };

    let wall_start = Instant::now();
    let (_table_cfr, _stats_cfr) = cfr::solve_until(
        score.clone(),
        tc,
        &cfr_config,
        &mv,
        None,
        |iter, expl, _s| {
            let wall_s = wall_start.elapsed().as_secs_f64();
            let line = format!(
                "{:>6}  {:>14.6}  {:>10.1}  {:>8}",
                iter, expl, wall_s, "CFR+"
            );
            println!("{}", line);
            std::io::stdout().flush().ok();
            writeln!(log_file, "{}", line).ok();
            log_file.flush().ok();
        },
    );

    let cfr_total = wall_start.elapsed().as_secs_f64();
    let cfr_msg = format!("CFR+ total: {:.1}s", cfr_total);
    println!("{}", cfr_msg);
    writeln!(log_file, "{}", cfr_msg).ok();

    // ── DCFR ──────────────────────────────────────────────────────────────
    println!("\n--- DCFR ---");
    writeln!(log_file, "\n--- DCFR ---").ok();

    let dcfr_config = cfr::SolveConfig {
        max_iters,
        target_expl: 0.0,
        algorithm: cfr::CfrAlgorithm::dcfr_default(),
        expl_every,
        ..Default::default()
    };

    let wall_start_dcfr = Instant::now();
    let (_table_dcfr, _stats_dcfr) = cfr::solve_until(
        score.clone(),
        tc,
        &dcfr_config,
        &mv,
        None,
        |iter, expl, _s| {
            let wall_s = wall_start_dcfr.elapsed().as_secs_f64();
            let line = format!(
                "{:>6}  {:>14.6}  {:>10.1}  {:>8}",
                iter, expl, wall_s, "DCFR"
            );
            println!("{}", line);
            std::io::stdout().flush().ok();
            writeln!(log_file, "{}", line).ok();
            log_file.flush().ok();
        },
    );

    let dcfr_total = wall_start_dcfr.elapsed().as_secs_f64();
    let dcfr_msg = format!("DCFR total: {:.1}s", dcfr_total);
    println!("{}", dcfr_msg);
    writeln!(log_file, "{}", dcfr_msg).ok();

    println!("\nSaved to {}", log_path);
}

// ─── treesize ────────────────────────────────────────────────────────────────

/// Size-only tree walk for one (score, tc, dealer): no PrebuiltTrees arena,
/// no InfoSetRegistry, no regret/strategy allocations -- just a running node
/// counter and a HashSet<u64> of info-set keys (see
/// `game_tree::count_tree_size`). Exists so untested ladder tiers (whose
/// solver-ready memory footprint is unknown and could be enormous) can be
/// sized cheaply before committing to a real solve. Validated exact-match
/// against the real builder in `game_tree::tests::test_count_tree_size_matches_full_build`.
fn run_count_tree(args: &[String]) {
    let score = parse_score_flag(args, "--score", Score { zero: 11, one: 11 });
    let tc_level = parse_u8_flag(args, "--tc", 0);
    let dealer = parse_u8_flag(args, "--dealer", 0);
    let max_deals = parse_opt_str_flag(args, "--max-deals").map(|v| v.parse::<usize>().unwrap());
    let progress_every = parse_opt_str_flag(args, "--progress-every")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(0);
    let policy_checkpoint = parse_opt_str_flag(args, "--policy-checkpoint");
    let policy_strategy = parse_opt_str_flag(args, "--policy-strategy");
    let policy_empty = has_flag(args, "--policy-empty");
    assert!(
        usize::from(policy_checkpoint.is_some())
            + usize::from(policy_strategy.is_some())
            + usize::from(policy_empty)
            <= 1,
        "use only one policy source"
    );

    let tc = TurnupClass {
        blocked_plain_level: tc_level,
    };
    let mut deals = truco_solver::abstraction::enumerate_deals(&tc);
    let total_deals_available = deals.len();
    if let Some(limit) = max_deals {
        cfr::subsample_deals(&mut deals, limit);
    }

    println!(
        "count-tree: score={}x{} tc={} dealer={} deals={}/{} (--max-deals {})",
        score.zero,
        score.one,
        tc_level,
        dealer,
        deals.len(),
        total_deals_available,
        max_deals
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unset (all)".into())
    );

    if policy_checkpoint.is_none() && policy_strategy.is_none() && !policy_empty {
        let t0 = Instant::now();
        let mut cb = |i: usize, n: usize, c: &truco_solver::game_tree::TreeSizeCount| {
            if progress_every > 0 && i.is_multiple_of(progress_every) {
                eprintln!(
                    "  ... {}/{} deals | {} distinct info sets | {} nodes | {:.1}s elapsed",
                    i,
                    n,
                    c.num_info_sets,
                    c.total_nodes,
                    t0.elapsed().as_secs_f64()
                );
            }
        };
        let progress: Option<truco_solver::game_tree::TreeSizeProgressCb<'_>> =
            if progress_every > 0 {
                Some(&mut cb)
            } else {
                None
            };

        let count_rules = if has_flag(args, "--asymmetric-raise-prune") {
            truco_solver::game_tree::TreeRules::AsymmetricRaisePrune
        } else if has_flag(args, "--legacy-tree") {
            truco_solver::game_tree::TreeRules::LegacyPreProofPrunes
        } else {
            truco_solver::game_tree::TreeRules::Current
        };
        let count = truco_solver::game_tree::count_tree_size_with_rules(
            &score,
            tc,
            dealer,
            &deals,
            count_rules,
            progress,
        )
        .unwrap();
        let elapsed = t0.elapsed().as_secs_f64();

        println!(
            "RESULT mode=raw score={}x{} tc={} dealer={} rules={:?} deals_sampled={} total_nodes={} num_info_sets={} elapsed_s={:.1}",
            score.zero, score.one, tc_level, dealer, count_rules, deals.len(), count.total_nodes, count.num_info_sets, elapsed
        );
    } else {
        run_policy_tree_counts(
            args,
            &score,
            tc,
            dealer,
            &deals,
            policy_checkpoint.as_deref(),
            policy_strategy.as_deref(),
            policy_empty,
            progress_every,
        );
    }
    if let Some(limit) = max_deals {
        if limit < total_deals_available {
            println!(
                "NOTE: sampled {}/{} deals ({:.4}%) -- info-set count above is a LOWER BOUND on the full-deal-set count, not an extrapolation.",
                limit, total_deals_available, 100.0 * limit as f64 / total_deals_available as f64
            );
        }
    }
}

type LoadedCensusPolicy<'a> = (
    Box<dyn truco_solver::game_tree::PolicyLookup>,
    (u8, u8),
    TurnupClass,
    Option<u8>,
    bool,
    &'a str,
);

#[allow(clippy::too_many_arguments)]
fn run_policy_tree_counts(
    args: &[String],
    score: &Score,
    tc: TurnupClass,
    dealer: u8,
    deals: &[truco_solver::abstraction::AbstractDeal],
    checkpoint_path: Option<&str>,
    strategy_path: Option<&str>,
    policy_empty: bool,
    progress_every: usize,
) {
    use truco_solver::game_tree::{MissingPolicyFallback, PolicyTreeMode, PolicyValueSource};

    let (policy, source_score, source_tc, source_dealer, has_regrets, source_path):
        LoadedCensusPolicy<'_> = if let Some(path) = checkpoint_path {
        let (table, meta) = load_checkpoint(Path::new(path)).expect("load --policy-checkpoint");
        (
            Box::new(table),
            meta.score,
            meta.turnup_class,
            meta.dealer_filter,
            true,
            path,
        )
    } else if let Some(path) = strategy_path {
        let (table, meta) =
            load_compact_average_policy(Path::new(path)).expect("stream --policy-strategy");
        (
            Box::new(table),
            meta.score,
            meta.turnup_class,
            None,
            false,
            path,
        )
    } else {
        debug_assert!(policy_empty);
        (
            Box::new(truco_solver::strategy::StrategyTable::new()),
            (score.zero, score.one),
            tc,
            None,
            false,
            "<empty policy; fallback only>",
        )
    };
    assert_eq!(source_tc, tc, "policy turnup class does not match --tc");
    if let Some(source_dealer) = source_dealer {
        assert_eq!(
            source_dealer, dealer,
            "dealer-filtered checkpoint does not match --dealer"
        );
    }

    let values_label = parse_str_flag(args, "--policy-values", "average");
    let values = match values_label.as_str() {
        "average" => PolicyValueSource::Average,
        "current" => {
            assert!(
                has_regrets,
                "--policy-values current requires --policy-checkpoint"
            );
            PolicyValueSource::Current
        }
        other => panic!("unknown --policy-values {other}; expected average|current"),
    };
    let missing_label = parse_str_flag(args, "--missing-policy", "all-except-raise");
    let missing = match missing_label.as_str() {
        "all" => MissingPolicyFallback::All,
        "first" => MissingPolicyFallback::First,
        "all-except-raise" => MissingPolicyFallback::AllExceptRaise,
        other => panic!("unknown --missing-policy {other}; expected all|first|all-except-raise"),
    };
    let mode_label = parse_str_flag(args, "--policy-mode", "all");
    let chosen_modes: Vec<(&str, bool)> = match mode_label.as_str() {
        "chosen-br-union" => vec![("chosen-br-union", false)],
        "chosen-br-closure" => vec![("chosen-br-closure", true)],
        "chosen-br-both" => vec![("chosen-br-union", false), ("chosen-br-closure", true)],
        _ => Vec::new(),
    };
    let modes: Vec<(&str, PolicyTreeMode)> = match mode_label.as_str() {
        "profile" => vec![("profile", PolicyTreeMode::Profile)],
        "br0" => vec![("br0", PolicyTreeMode::BestResponse(0))],
        "br1" => vec![("br1", PolicyTreeMode::BestResponse(1))],
        "br-union" => vec![("br-union", PolicyTreeMode::BestResponseUnion)],
        "all" => vec![
            ("profile", PolicyTreeMode::Profile),
            ("br0", PolicyTreeMode::BestResponse(0)),
            ("br1", PolicyTreeMode::BestResponse(1)),
            ("br-union", PolicyTreeMode::BestResponseUnion),
        ],
        "chosen-br-union" | "chosen-br-closure" | "chosen-br-both" => Vec::new(),
        other => panic!(
            "unknown --policy-mode {other}; expected profile|br0|br1|br-union|chosen-br-union|chosen-br-closure|chosen-br-both|all"
        ),
    };
    let thresholds_label = parse_str_flag(args, "--support-thresholds", "0");
    let thresholds: Vec<f64> = thresholds_label
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .expect("--support-thresholds must be comma-separated numbers")
        })
        .collect();
    assert!(
        !thresholds.is_empty(),
        "at least one support threshold is required"
    );

    println!(
        "policy: path={} source_score={}x{} entries={} values={} missing_policy={}{}",
        source_path,
        source_score.0,
        source_score.1,
        policy.len(),
        values_label,
        missing_label,
        if source_score != (score.zero, score.one) {
            " (projected across scores)"
        } else {
            ""
        }
    );

    if !chosen_modes.is_empty() {
        assert_eq!(
            source_score,
            (score.zero, score.one),
            "chosen-BR census requires a policy solved at the target score"
        );
        assert!(
            !policy_empty,
            "chosen-BR census requires a complete saved policy"
        );
        let match_values_path = parse_opt_str_flag(args, "--match-values")
            .expect("chosen-BR policy modes require --match-values");
        let match_values =
            load_match_values(Path::new(&match_values_path)).expect("load --match-values");
        run_chosen_br_counts(
            score,
            tc,
            dealer,
            deals,
            policy.as_ref(),
            values,
            &thresholds,
            &match_values,
            &chosen_modes,
        );
        return;
    }

    // Run sequentially: each census owns a potentially large seen-key HashSet.
    // Holding all mode/threshold sets simultaneously would defeat the counter's
    // space-for-time purpose on full-ladder trees.
    for threshold in thresholds {
        for &(label, mode) in &modes {
            let t0 = Instant::now();
            let mut cb = |i: usize, n: usize, c: &truco_solver::game_tree::PolicyTreeSizeCount| {
                if progress_every > 0 && i.is_multiple_of(progress_every) {
                    eprintln!(
                        "  ... mode={} threshold={} {}/{} deals | {} info sets | {} nodes | {} policy misses | {:.1}s",
                        label,
                        threshold,
                        i,
                        n,
                        c.num_info_sets,
                        c.total_nodes,
                        c.policy_missing_info_sets,
                        t0.elapsed().as_secs_f64()
                    );
                }
            };
            let progress: Option<truco_solver::game_tree::PolicyTreeSizeProgressCb<'_>> =
                if progress_every > 0 {
                    Some(&mut cb)
                } else {
                    None
                };
            let count = truco_solver::game_tree::count_policy_tree_size(
                score,
                tc,
                dealer,
                deals,
                policy.as_ref(),
                mode,
                values,
                threshold,
                missing,
                progress,
            )
            .unwrap();
            println!(
                "RESULT mode={} threshold={} score={}x{} tc={} dealer={} deals_sampled={} total_nodes={} num_info_sets={} policy_missing_info_sets={} kept_actions={} legal_actions={} kept_raises={} legal_raises={} elapsed_s={:.1}",
                label,
                threshold,
                score.zero,
                score.one,
                tc.blocked_plain_level,
                dealer,
                deals.len(),
                count.total_nodes,
                count.num_info_sets,
                count.policy_missing_info_sets,
                count.kept_actions,
                count.legal_actions,
                count.kept_raises,
                count.legal_raises,
                t0.elapsed().as_secs_f64()
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chosen_br_counts(
    score: &Score,
    tc: TurnupClass,
    dealer: u8,
    deals: &[truco_solver::abstraction::AbstractDeal],
    policy: &dyn truco_solver::game_tree::PolicyLookup,
    values: truco_solver::game_tree::PolicyValueSource,
    thresholds: &[f64],
    match_values: &MatchValueTable,
    chosen_modes: &[(&str, bool)],
) {
    use truco_solver::strategy::ActionProbs;

    let build_started = Instant::now();
    eprintln!("chosen-BR census: building full tree for exact BR decisions...");
    let prebuilt =
        truco_solver::game_tree::build_all_trees_with_dealer(score, tc, deals, Some(dealer))
            .expect("build full tree for chosen-BR census");
    eprintln!(
        "chosen-BR census: built {} info sets in {:.1}s; aligning policy...",
        prebuilt.info_sets.len(),
        build_started.elapsed().as_secs_f64()
    );

    let strategies: Vec<ActionProbs> = prebuilt
        .info_sets
        .iter()
        .map(|(key, _, actions)| {
            let mut probabilities = ActionProbs::with_capacity(actions.len());
            for &action in actions.iter() {
                let probability = policy
                    .action_probability(*key, action, values)
                    .unwrap_or_else(|| {
                        panic!(
                            "saved policy is incomplete at key={} action={action:?}",
                            key.0
                        )
                    });
                assert!(
                    probability.is_finite() && probability >= 0.0,
                    "invalid saved probability at key={} action={action:?}",
                    key.0
                );
                probabilities.push(probability);
            }
            let sum: f64 = probabilities.iter().sum();
            assert!(sum > 0.0, "empty saved distribution at key={}", key.0);
            for probability in &mut probabilities {
                *probability /= sum;
            }
            probabilities
        })
        .collect();

    let br_started = Instant::now();
    let br0 =
        cfr::best_response_full_from_action_probs(&prebuilt, &strategies, score, match_values, 0);
    let br0_total = br0.total;
    let chosen0 = br0.chosen_actions;
    drop(br0.per_info_set);
    eprintln!(
        "chosen-BR census: player-0 exact BR={br0_total:.9} in {:.1}s",
        br_started.elapsed().as_secs_f64()
    );

    let br1_started = Instant::now();
    let br1 =
        cfr::best_response_full_from_action_probs(&prebuilt, &strategies, score, match_values, 1);
    let br1_total = br1.total;
    let chosen1 = br1.chosen_actions;
    drop(br1.per_info_set);
    eprintln!(
        "chosen-BR census: player-1 exact BR={br1_total:.9} in {:.1}s",
        br1_started.elapsed().as_secs_f64()
    );

    for &threshold in thresholds {
        for &(label, closure) in chosen_modes {
            let count_started = Instant::now();
            let count = if closure {
                truco_solver::game_tree::count_chosen_best_response_closure(
                    &prebuilt,
                    &strategies,
                    [&chosen0, &chosen1],
                    threshold,
                )
            } else {
                truco_solver::game_tree::count_chosen_best_response_union(
                    &prebuilt,
                    &strategies,
                    [&chosen0, &chosen1],
                    threshold,
                )
            };
            println!(
                "RESULT mode={} threshold={} score={}x{} tc={} dealer={} deals_sampled={} total_nodes={} num_info_sets={} kept_actions={} legal_actions={} kept_raises={} legal_raises={} br0_total={:.9} br1_total={:.9} build_and_br_s={:.1} count_s={:.1}",
                label,
                threshold,
                score.zero,
                score.one,
                tc.blocked_plain_level,
                dealer,
                deals.len(),
                count.total_nodes,
                count.num_info_sets,
                count.kept_actions,
                count.legal_actions,
                count.kept_raises,
                count.legal_raises,
                br0_total,
                br1_total,
                build_started.elapsed().as_secs_f64(),
                count_started.elapsed().as_secs_f64(),
            );
        }
    }
}

fn run_restricted_bench(args: &[String]) {
    use truco_solver::game_tree::PolicyLookup;
    use truco_solver::strategy::ActionProbs;

    let score = parse_score_flag(args, "--score", Score { zero: 8, one: 8 });
    let tc = TurnupClass {
        blocked_plain_level: parse_u8_flag(args, "--tc", 0),
    };
    let dealer = parse_u8_flag(args, "--dealer", 0);
    assert!(dealer < 2, "--dealer must be 0 or 1");
    let max_deals = parse_opt_str_flag(args, "--max-deals")
        .expect("restricted-bench requires --max-deals")
        .parse::<usize>()
        .expect("--max-deals must be a number");
    let threshold = parse_f64_flag(args, "--support-threshold", 1e-4);
    let target = parse_f64_flag(args, "--eps", 0.01);
    let max_iters = parse_u64_flag(args, "--max-iters", 100);
    let expl_every = parse_u64_flag(args, "--expl-every", 10);
    let oracle_rounds = parse_usize_flag(args, "--oracle-rounds", 3);
    assert!(oracle_rounds > 0, "--oracle-rounds must be positive");
    let skip_full_control = has_flag(args, "--skip-full-control");
    let checkpoint_path = parse_opt_str_flag(args, "--policy-checkpoint")
        .expect("restricted-bench requires --policy-checkpoint");
    let match_values_path = parse_opt_str_flag(args, "--match-values")
        .expect("restricted-bench requires --match-values");
    let (policy, meta) =
        load_checkpoint(Path::new(&checkpoint_path)).expect("load --policy-checkpoint");
    let policy = Arc::new(policy);
    assert_eq!(meta.turnup_class, tc, "policy turnup class mismatch");
    assert_eq!(meta.dealer_filter, Some(dealer), "policy dealer mismatch");
    let policy_score = Score {
        zero: meta.score.0,
        one: meta.score.1,
    };
    assert_eq!(
        truco_solver::game_tree::band_signature(&policy_score, meta.dealer_filter),
        truco_solver::game_tree::band_signature(&score, Some(dealer)),
        "policy must come from the target score or a same-band neighbor"
    );
    println!(
        "RESTRICTED_SOURCE policy_score={}x{} target_score={}x{} source_iter={}",
        policy_score.zero, policy_score.one, score.zero, score.one, meta.iteration,
    );
    let match_values =
        load_match_values(Path::new(&match_values_path)).expect("load --match-values");

    let mut deals = truco_solver::abstraction::enumerate_deals(&tc);
    cfr::subsample_deals(&mut deals, max_deals);
    let build_started = Instant::now();
    let full = Arc::new(
        truco_solver::game_tree::build_all_trees_with_dealer(&score, tc, &deals, Some(dealer))
            .expect("build full benchmark tree"),
    );
    let full_build_s = build_started.elapsed().as_secs_f64();
    let source_strategies: Vec<ActionProbs> = full
        .info_sets
        .iter()
        .map(|(key, _, actions)| {
            let mut probabilities: ActionProbs = actions
                .iter()
                .map(|&action| {
                    policy
                        .action_probability(
                            *key,
                            action,
                            truco_solver::game_tree::PolicyValueSource::Average,
                        )
                        .unwrap_or_else(|| {
                            panic!("checkpoint missing key={} action={action:?}", key.0)
                        })
                })
                .collect();
            let sum: f64 = probabilities.iter().sum();
            assert!(sum > 0.0, "empty checkpoint distribution at key={}", key.0);
            for probability in &mut probabilities {
                *probability /= sum;
            }
            probabilities
        })
        .collect();
    let source_br_started = Instant::now();
    let br0 = cfr::best_response_full_from_action_probs(
        &full,
        &source_strategies,
        &score,
        &match_values,
        0,
    );
    let chosen0 = br0.chosen_actions;
    drop(br0.per_info_set);
    let br1 = cfr::best_response_full_from_action_probs(
        &full,
        &source_strategies,
        &score,
        &match_values,
        1,
    );
    let chosen1 = br1.chosen_actions;
    drop(br1.per_info_set);
    let mut allowed = truco_solver::game_tree::chosen_closure_action_masks(
        &full,
        &source_strategies,
        [&chosen0, &chosen1],
        threshold,
    );
    let source_br_s = source_br_started.elapsed().as_secs_f64();
    let full_nodes: usize = full
        .entries
        .iter()
        .map(|entry| entry.tree_dealer_0.num_nodes() + entry.tree_dealer_1.num_nodes())
        .sum();
    let base_config = cfr::SolveConfig {
        max_iters,
        target_expl: target,
        algorithm: cfr::CfrAlgorithm::SyncCfrPlus,
        expl_every,
        dealer_filter: Some(dealer),
        max_deals: Some(max_deals),
        warmstart_source: Some(policy.clone()),
        warmstart_same_band: true,
        ..Default::default()
    };
    let (full_iters, full_eps, full_wall) = if skip_full_control {
        (0, f64::NAN, f64::NAN)
    } else {
        let mut full_config = base_config.clone();
        full_config.prebuilt_override = Some(full.clone());
        let full_started = Instant::now();
        let (_full_table, full_stats) = cfr::solve_until(
            score.clone(),
            tc,
            &full_config,
            &match_values,
            None,
            |_iter, _expl, _secs| {},
        );
        (
            full_stats.iterations,
            full_stats.exploitability.unwrap_or(f64::NAN),
            full_started.elapsed().as_secs_f64(),
        )
    };
    println!(
        "RESTRICTED_CONTROL full_iters={} full_eps={:.9} full_build_s={:.3} full_wall_s={:.3} full_end_to_end_s={:.3} source_br_s={:.3} restricted_initial_setup_s={:.3}",
        full_iters,
        full_eps,
        full_build_s,
        full_wall,
        full_build_s + full_wall,
        source_br_s,
        full_build_s + source_br_s,
    );

    let mut certified = false;
    let mut cumulative_restricted_wall = 0.0;
    let mut cumulative_build_wall = 0.0;
    let mut cumulative_audit_wall = 0.0;
    for round in 0..oracle_rounds {
        let round_build_started = Instant::now();
        let restricted = Arc::new(truco_solver::game_tree::restrict_prebuilt_to_action_masks(
            &full, &allowed,
        ));
        let restricted_nodes: usize = restricted
            .entries
            .iter()
            .map(|entry| entry.tree_dealer_0.num_nodes() + entry.tree_dealer_1.num_nodes())
            .sum();
        let restricted_infos = restricted.info_sets.len();
        let round_build_s = round_build_started.elapsed().as_secs_f64();
        cumulative_build_wall += round_build_s;

        let mut restricted_config = base_config.clone();
        restricted_config.prebuilt_override = Some(restricted.clone());
        let restricted_started = Instant::now();
        let (restricted_table, restricted_stats) = cfr::solve_until(
            score.clone(),
            tc,
            &restricted_config,
            &match_values,
            None,
            |_iter, _expl, _secs| {},
        );
        let restricted_wall = restricted_started.elapsed().as_secs_f64();
        cumulative_restricted_wall += restricted_wall;

        // Define a complete full-tree profile for the exact audit: use the
        // restricted mix wherever that arena has a row, zero its excluded
        // actions, and retain the source policy only outside the arena.
        let mut audited = source_strategies.clone();
        for (old_idx, (key, _, full_actions)) in full.info_sets.iter().enumerate() {
            let Some(data) = restricted_table.data.get(key) else {
                continue;
            };
            let restricted_probabilities = data.average_strategy();
            let mut probabilities = ActionProbs::from_vec(vec![0.0; full_actions.len()]);
            for (restricted_idx, &action) in data.actions.iter().enumerate() {
                let full_idx = full_actions
                    .iter()
                    .position(|&candidate| candidate == action)
                    .expect("restricted action must exist in full action list");
                probabilities[full_idx] = restricted_probabilities[restricted_idx];
            }
            audited[old_idx] = probabilities;
        }
        let audit_started = Instant::now();
        let audit0 =
            cfr::best_response_full_from_action_probs(&full, &audited, &score, &match_values, 0);
        let audit1 =
            cfr::best_response_full_from_action_probs(&full, &audited, &score, &match_values, 1);
        let audited_exploitability = (audit0.total + audit1.total) / 2.0;
        let audit_s = audit_started.elapsed().as_secs_f64();
        cumulative_audit_wall += audit_s;
        let cumulative_round_wall =
            cumulative_build_wall + cumulative_restricted_wall + cumulative_audit_wall;
        let restricted_end_to_end = full_build_s + source_br_s + cumulative_round_wall;
        println!(
            "RESTRICTED_ROUND round={} restricted_nodes={} restricted_infos={} shrink_nodes={:.3} shrink_infos={:.3} build_s={:.3} iterations={} internal_eps={:.9} full_audit_eps={:.9} solve_wall_s={:.3} audit_s={:.3} cumulative_round_wall_s={:.3} restricted_end_to_end_s={:.3} speedup_vs_control={:.3}",
            round,
            restricted_nodes,
            restricted_infos,
            full_nodes as f64 / restricted_nodes as f64,
            full.info_sets.len() as f64 / restricted_infos as f64,
            round_build_s,
            restricted_stats.iterations,
            restricted_stats.exploitability.unwrap_or(f64::NAN),
            audited_exploitability,
            restricted_wall,
            audit_s,
            cumulative_round_wall,
            restricted_end_to_end,
            (full_build_s + full_wall) / restricted_end_to_end,
        );
        if audited_exploitability <= target {
            certified = true;
            break;
        }

        let mut added = 0usize;
        for (idx, (_, info, actions)) in full.info_sets.iter().enumerate() {
            let chosen = if info.player == 0 {
                audit0.chosen_actions[idx]
            } else {
                audit1.chosen_actions[idx]
            } as usize;
            assert!(
                chosen < actions.len(),
                "audit responder choice must be legal"
            );
            if !allowed[idx][chosen] {
                allowed[idx][chosen] = true;
                added += 1;
            }
        }
        println!(
            "RESTRICTED_ORACLE_ADD round={} added_actions={}",
            round, added
        );
        if added == 0 {
            break;
        }
    }
    println!(
        "RESTRICTED_FINAL certified={} target={} rounds_limit={} cumulative_build_wall_s={:.3} cumulative_solve_wall_s={:.3} cumulative_audit_wall_s={:.3} cumulative_round_wall_s={:.3} restricted_end_to_end_s={:.3}",
        certified,
        target,
        oracle_rounds,
        cumulative_build_wall,
        cumulative_restricted_wall,
        cumulative_audit_wall,
        cumulative_build_wall + cumulative_restricted_wall + cumulative_audit_wall,
        full_build_s
            + source_br_s
            + cumulative_build_wall
            + cumulative_restricted_wall
            + cumulative_audit_wall,
    );
}

fn run_tree_size_survey() {
    let mv = MatchValueTable::new();
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };

    let scores: Vec<Score> = vec![
        Score { zero: 11, one: 11 },
        Score { zero: 10, one: 11 },
        Score { zero: 10, one: 10 },
        Score { zero: 9, one: 10 },
        Score { zero: 8, one: 8 },
        Score { zero: 6, one: 6 },
        Score { zero: 4, one: 4 },
        Score { zero: 2, one: 2 },
        Score { zero: 0, one: 0 },
    ];

    println!("=== Tree Size Survey (TC {}) ===", tc.blocked_plain_level);
    println!(
        "{:>8}  {:>13}  {:>10}  {:>9}  {:>9}",
        "score", "total_nodes", "info_sets", "build_s", "mem_MB"
    );

    for score in scores {
        let t = Instant::now();
        let (_table, stats) = cfr::solve_with_limit(score.clone(), tc, 0, &mv, None, None);
        println!(
            "{:>4}x{:<3}  {:>13}  {:>10}  {:>9.1}  {:>9.1}",
            score.zero,
            score.one,
            stats.total_nodes,
            stats.num_info_sets,
            t.elapsed().as_secs_f64(),
            stats.estimated_memory_bytes as f64 / 1_048_576.0
        );
        std::io::stdout().flush().ok();
    }
}

// ─── benchmark ───────────────────────────────────────────────────────────────

fn run_benchmark() {
    let score = Score { zero: 11, one: 11 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let mv = MatchValueTable::new();

    println!("=== Benchmark: 10 iters at 11x11, TC 0 ===");
    let (_table, stats) = cfr::solve(score, tc, 10, &mv);
    println!("{}", stats);
    println!(
        "Extrapolations (per iter = {:.1}s):",
        stats.per_iteration_secs
    );
    println!(
        "  300 iters / 1 TC:  {:.0}s  ({:.1} hr)",
        stats.per_iteration_secs * 300.0,
        stats.per_iteration_secs * 300.0 / 3600.0
    );
}

// ─── flag parsing helpers ─────────────────────────────────────────────────────

fn parse_score_flag(args: &[String], flag: &str, default: Score) -> Score {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            let s = &args[i + 1];
            // Accept "11x11" or "11"
            if let Some((a, b)) = s.split_once('x') {
                if let (Ok(z), Ok(o)) = (a.parse::<u8>(), b.parse::<u8>()) {
                    return Score { zero: z, one: o };
                }
            } else if let Ok(v) = s.parse::<u8>() {
                return Score { zero: v, one: v };
            }
        }
    }
    default
}

fn parse_u8_flag(args: &[String], flag: &str, default: u8) -> u8 {
    parse_u64_flag(args, flag, default as u64) as u8
}

fn parse_u64_flag(args: &[String], flag: &str, default: u64) -> u64 {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            if let Ok(v) = args[i + 1].parse() {
                return v;
            }
        }
    }
    default
}

fn parse_usize_flag(args: &[String], flag: &str, default: usize) -> usize {
    parse_u64_flag(args, flag, default as u64) as usize
}

fn parse_bool_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_f64_flag(args: &[String], flag: &str, default: f64) -> f64 {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            if let Ok(v) = args[i + 1].parse() {
                return v;
            }
        }
    }
    default
}

fn parse_str_flag(args: &[String], flag: &str, default: &str) -> String {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn parse_multi_str_flag(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            values.push(args[i + 1].clone());
        }
    }
    values
}

fn parse_opt_str_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn parse_opt_f64_flag(args: &[String], flag: &str) -> Option<f64> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            if let Ok(v) = args[i + 1].parse() {
                return Some(v);
            }
        }
    }
    None
}

/// Human-readable algorithm label — must match the label `solve_until` writes
/// into checkpoint metadata, since resume validation compares them.
fn algo_label_of(algo: &cfr::CfrAlgorithm) -> String {
    match algo {
        cfr::CfrAlgorithm::CfrPlus => "CFR+".to_string(),
        cfr::CfrAlgorithm::Dcfr { alpha, beta, gamma } => {
            format!("DCFR(α={}, β={}, γ={})", alpha, beta, gamma)
        }
        cfr::CfrAlgorithm::SyncCfrPlus => "SyncCFR+".to_string(),
        cfr::CfrAlgorithm::PcfrPlus => "PCFR+".to_string(),
    }
}

fn parse_algorithm(args: &[String]) -> cfr::CfrAlgorithm {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == "--algo" {
            return match args[i + 1].to_lowercase().as_str() {
                "dcfr" => cfr::CfrAlgorithm::dcfr_default(),
                "cfr+" | "cfr" | "cfrplus" => cfr::CfrAlgorithm::CfrPlus,
                "sync" | "synccfr+" | "sync-cfr+" => cfr::CfrAlgorithm::SyncCfrPlus,
                "pcfr+" | "pcfr" | "pcfrplus" => cfr::CfrAlgorithm::PcfrPlus,
                _ => {
                    eprintln!("Unknown algorithm '{}', using CFR+", args[i + 1]);
                    cfr::CfrAlgorithm::CfrPlus
                }
            };
        }
    }
    cfr::CfrAlgorithm::CfrPlus
}

#[cfg(test)]
mod tests {
    use super::same_band_neighbor_scores;
    use truco_engine::Score;

    fn pairs(score: Score) -> Vec<(u8, u8)> {
        same_band_neighbor_scores(&score)
            .into_iter()
            .map(|candidate| (candidate.zero, candidate.one))
            .collect()
    }

    #[test]
    fn pipeline_neighbor_selection_stays_in_the_exact_tree_band() {
        assert!(pairs(Score { zero: 10, one: 10 }).is_empty());
        assert_eq!(pairs(Score { zero: 10, one: 9 }), vec![(10, 10)]);
        assert_eq!(pairs(Score { zero: 9, one: 9 }), vec![(10, 9)]);
        assert_eq!(pairs(Score { zero: 8, one: 8 }), vec![(9, 8)]);
        assert_eq!(pairs(Score { zero: 11, one: 9 }), vec![(11, 10)]);
    }
}
