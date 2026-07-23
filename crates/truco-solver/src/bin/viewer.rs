//! Solution viewer: inspect equilibrium strategies at specific game states.
//!
//! Usage:
//!   cargo run --release --bin viewer
//!   cargo run --release --bin viewer -- path/to/strategy.bin
//!
//! Solves a small subgame or loads a saved strategy and lets you inspect it.

use std::path::Path;

use smallvec::smallvec;
use truco_engine::Score;
use truco_solver::abstraction::{AbstractCard, TurnupClass};
use truco_solver::cfr;
use truco_solver::info_set::{AbstractAction, InfoSet};
use truco_solver::match_value::MatchValueTable;
use truco_solver::storage::{load_strategy, SolvedStateMeta};
use truco_solver::strategy::StrategyTable;

fn main() {
    env_logger::init();

    eprintln!("=== Truco Solution Viewer ===\n");
    let args: Vec<String> = std::env::args().collect();

    let mut strategy_path: Option<String> = None;
    let mut json_out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--json" {
            json_out = args.get(i + 1).cloned();
            i += 2;
        } else {
            if strategy_path.is_none() {
                strategy_path = Some(args[i].clone());
            }
            i += 1;
        }
    }

    if let Some(out) = json_out {
        let path = strategy_path.expect("--json requires a strategy file path");
        let (table, meta) = load_strategy(Path::new(&path)).expect("load strategy");
        export_charts_json(&table, &meta, &out);
        return;
    }

    let maybe_path = strategy_path.as_deref().map(Path::new);

    if let Some(path) = maybe_path {
        let (table, meta) = load_strategy(path).expect("load strategy");
        eprintln!("Loaded strategy: {}\n", path.display());
        print_loaded_meta(&meta);
        print_strategy_summary(&table);
        print_section("Sample Stored Info Sets");
        print_sample_info_sets(&table, 10);

        if meta.score == (11, 11) {
            let tc = meta.turnup_class;
            print_section("11x11 OPENING PLAY: What to do with various hands at mão de onze");
            print_mao_de_onze_decisions(&table, tc);
            print_section("11x11 CARD PLAY: After both accept, what cards to play first?");
            print_opening_card_play(&table, tc);
            print_section("11x11 HAND STRENGTH ANALYSIS");
            print_hand_quality_analysis(&table, tc);
        }
        return;
    }

    eprintln!("Solving 11x11, turnup class 0, 100 iterations...\n");

    let score = Score { zero: 11, one: 11 };
    let tc = TurnupClass {
        blocked_plain_level: 0,
    };
    let mv = MatchValueTable::new();

    let (table, stats) = cfr::solve(score, tc, 100, &mv);

    eprintln!("\n{}\n", stats);

    // Print some interesting strategy profiles
    print_section("1. OPENING PLAY: What to do with various hands at 11x11 (mão de onze)");
    print_mao_de_onze_decisions(&table, tc);

    print_section("2. CARD PLAY: After both accept, what cards to play first?");
    print_opening_card_play(&table, tc);

    print_section("3. HAND STRENGTH ANALYSIS: Accept/fold rates by hand quality");
    print_hand_quality_analysis(&table, tc);

    print_section("4. STRATEGY SUMMARY STATISTICS");
    print_strategy_summary(&table);

    print_section("5. SAMPLE INFO SETS");
    print_sample_info_sets(&table, 10);
}

fn print_section(title: &str) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("{}", title);
    eprintln!("{}\n", "=".repeat(60));
}

fn print_loaded_meta(meta: &SolvedStateMeta) {
    eprintln!(
        "Score: {}x{} | TC: {} | Iterations: {} | Info sets: {}",
        meta.score.0,
        meta.score.1,
        meta.turnup_class.blocked_plain_level,
        meta.iterations,
        meta.num_info_sets
    );
}

/// Export the chart-relevant strategy slices as compact JSON for the frontend
/// solution viewer. Emits the OPENING decision (empty-history info sets — the
/// leader's first action of the hand) keyed by abstract hand, with the full
/// average-strategy action distribution. Cards are encoded as their 0..12
/// strength index (0..8 = plain levels, 9..12 = manilhas ouros..zap). Actions
/// are encoded "FU:i" (play card i face-up) / "FD:i" (face-down) / debug name.
fn export_charts_json(table: &StrategyTable, meta: &SolvedStateMeta, out: &str) {
    fn enc(a: AbstractAction) -> String {
        match a {
            AbstractAction::PlayFaceUp(c) => format!("FU:{}", c.type_index()),
            AbstractAction::PlayFaceDown(c) => format!("FD:{}", c.type_index()),
            other => format!("{:?}", other),
        }
    }
    let mut hands: Vec<serde_json::Value> = Vec::new();
    for (key, data) in &table.data {
        let Some(info) = table.get_info_set(*key) else {
            continue;
        };
        if !info.history.is_empty() {
            continue;
        }
        let avg = data.average_strategy();
        let mut actions: Vec<serde_json::Value> = Vec::new();
        for (a, p) in data.actions.iter().zip(avg.iter()) {
            if *p < 0.005 {
                continue;
            }
            actions.push(serde_json::json!({
                "a": enc(*a),
                "p": (*p * 1e4).round() / 1e4,
            }));
        }
        let mut cards: Vec<u8> = info
            .starting_hand
            .iter()
            .map(|c| c.type_index() as u8)
            .collect();
        cards.sort_unstable();
        hands.push(serde_json::json!({
            "cards": cards,
            "player": info.player,
            "actions": actions,
        }));
    }
    hands.sort_by(|a, b| a["cards"].to_string().cmp(&b["cards"].to_string()));
    let n = hands.len();
    let doc = serde_json::json!({
        "score": [meta.score.0, meta.score.1],
        "turnup_class": meta.turnup_class.blocked_plain_level,
        "iterations": meta.iterations,
        "num_info_sets": meta.num_info_sets,
        "decisions": {
            "opening": { "hands": hands }
        }
    });
    std::fs::write(
        out,
        serde_json::to_string_pretty(&doc).expect("serialize json"),
    )
    .expect("write json");
    eprintln!("wrote {} ({} opening info sets)", out, n);
}

/// At 11x11, both players face a mão de onze decision. Show accept/fold rates.
fn print_mao_de_onze_decisions(table: &StrategyTable, tc: TurnupClass) {
    // Sample hands at different strength levels
    let sample_hands = vec![
        (
            "Trash (P0, P1, P2)",
            smallvec![
                AbstractCard::Plain(0),
                AbstractCard::Plain(1),
                AbstractCard::Plain(2)
            ],
        ),
        (
            "Low (P0, P2, P4)",
            smallvec![
                AbstractCard::Plain(0),
                AbstractCard::Plain(2),
                AbstractCard::Plain(4)
            ],
        ),
        (
            "Medium (P3, P5, P7)",
            smallvec![
                AbstractCard::Plain(3),
                AbstractCard::Plain(5),
                AbstractCard::Plain(7)
            ],
        ),
        (
            "Strong (P6, P7, P8)",
            smallvec![
                AbstractCard::Plain(6),
                AbstractCard::Plain(7),
                AbstractCard::Plain(8)
            ],
        ),
        (
            "One manilha (P2, P5, M0)",
            smallvec![
                AbstractCard::Plain(2),
                AbstractCard::Plain(5),
                AbstractCard::Manilha(0)
            ],
        ),
        (
            "One manilha strong (P5, P8, M2)",
            smallvec![
                AbstractCard::Plain(5),
                AbstractCard::Plain(8),
                AbstractCard::Manilha(2)
            ],
        ),
        (
            "Two manilhas (P4, M0, M1)",
            smallvec![
                AbstractCard::Plain(4),
                AbstractCard::Manilha(0),
                AbstractCard::Manilha(1)
            ],
        ),
        (
            "Three manilhas (M0, M1, M2)",
            smallvec![
                AbstractCard::Manilha(0),
                AbstractCard::Manilha(1),
                AbstractCard::Manilha(2)
            ],
        ),
        (
            "Zap + trash (P0, P1, M3)",
            smallvec![
                AbstractCard::Plain(0),
                AbstractCard::Plain(1),
                AbstractCard::Manilha(3)
            ],
        ),
    ];

    eprintln!("{:<35} {:>10} {:>10}", "Hand", "Accept%", "Fold%");
    eprintln!("{:-<55}", "");

    for (label, hand) in &sample_hands {
        // Player 1 (non-dealer) decides first at mão de onze
        let info_set = InfoSet::new(1, false, tc, hand.clone());
        if let Some(data) = table.get(&info_set) {
            let avg = data.average_strategy();
            let actions = &data.actions;

            let accept_prob = actions
                .iter()
                .zip(avg.iter())
                .find(|(a, _)| matches!(a, AbstractAction::AcceptEleven))
                .map(|(_, p)| *p)
                .unwrap_or(0.0);
            let fold_prob = 1.0 - accept_prob;

            eprintln!(
                "{:<35} {:>9.1}% {:>9.1}%",
                label,
                accept_prob * 100.0,
                fold_prob * 100.0,
            );
        } else {
            eprintln!("{:<35} {:>10}", label, "(not found)");
        }
    }
}

/// After both players accept mão de onze, what card does the first player lead?
fn print_opening_card_play(table: &StrategyTable, tc: TurnupClass) {
    let sample_hands = vec![
        (
            "Medium (P3, P5, P7)",
            smallvec![
                AbstractCard::Plain(3),
                AbstractCard::Plain(5),
                AbstractCard::Plain(7)
            ],
        ),
        (
            "Strong (P6, P7, P8)",
            smallvec![
                AbstractCard::Plain(6),
                AbstractCard::Plain(7),
                AbstractCard::Plain(8)
            ],
        ),
        (
            "One manilha (P2, P5, M0)",
            smallvec![
                AbstractCard::Plain(2),
                AbstractCard::Plain(5),
                AbstractCard::Manilha(0)
            ],
        ),
        (
            "Zap + trash (P0, P1, M3)",
            smallvec![
                AbstractCard::Plain(0),
                AbstractCard::Plain(1),
                AbstractCard::Manilha(3)
            ],
        ),
    ];

    for (label, hand) in &sample_hands {
        // After both accept, player 1 (non-dealer) plays first
        // History: player saw AcceptEleven (own), then AcceptEleven (from opponent)
        let mut info_set = InfoSet::new(1, false, tc, hand.clone());
        info_set.history.push(AbstractAction::AcceptEleven);
        info_set.history.push(AbstractAction::AcceptEleven);

        eprintln!("  Hand: {}", label);
        if let Some(data) = table.get(&info_set) {
            let avg = data.average_strategy();
            for (action, prob) in data.actions.iter().zip(avg.iter()) {
                if *prob > 0.001 {
                    eprintln!("    {:?}: {:.1}%", action, prob * 100.0);
                }
            }
        } else {
            eprintln!("    (info set not found — may need different history encoding)");
        }
        eprintln!();
    }
}

/// Analyze accept/fold patterns by hand "quality" (sum of card strengths)
fn print_hand_quality_analysis(table: &StrategyTable, _tc: TurnupClass) {
    let mut quality_buckets: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); 30]; // (total_accept, count, n)

    for (key, data) in &table.data {
        let _ = key; // we iterate all info sets
        let actions = &data.actions;

        // Only look at mão de onze decision points
        if !actions.contains(&AbstractAction::AcceptEleven) {
            continue;
        }
        if actions.len() != 2 {
            continue;
        }

        let avg = data.average_strategy();
        let accept_prob = actions
            .iter()
            .zip(avg.iter())
            .find(|(a, _)| matches!(a, AbstractAction::AcceptEleven))
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        // We can't reconstruct the hand from just the key, so we count overall stats
        quality_buckets[0].0 += accept_prob;
        quality_buckets[0].1 += 1.0;
        quality_buckets[0].2 += 1;
    }

    if quality_buckets[0].2 > 0 {
        let avg_accept = quality_buckets[0].0 / quality_buckets[0].1;
        eprintln!(
            "Overall mão de onze accept rate: {:.1}% (across {} info sets)",
            avg_accept * 100.0,
            quality_buckets[0].2,
        );
    }
}

/// Print summary statistics about the strategy
fn print_strategy_summary(table: &StrategyTable) {
    let mut total_info_sets = 0usize;
    let mut pure_strategy_count = 0usize; // info sets where one action has > 95% probability
    let mut avg_actions = 0.0f64;
    let mut max_actions = 0usize;

    for data in table.data.values() {
        total_info_sets += 1;
        avg_actions += data.actions.len() as f64;
        max_actions = max_actions.max(data.actions.len());

        let avg = data.average_strategy();
        if avg.iter().any(|&p| p > 0.95) {
            pure_strategy_count += 1;
        }
    }

    eprintln!("Total info sets: {}", total_info_sets);
    eprintln!(
        "Average actions per info set: {:.2}",
        avg_actions / total_info_sets as f64
    );
    eprintln!("Max actions at any info set: {}", max_actions);
    eprintln!(
        "Near-pure strategies (>95% on one action): {} ({:.1}%)",
        pure_strategy_count,
        pure_strategy_count as f64 / total_info_sets as f64 * 100.0
    );
}

fn print_sample_info_sets(table: &StrategyTable, limit: usize) {
    let mut keys: Vec<_> = table.data.keys().copied().collect();
    keys.sort_by_key(|key| key.0);

    for key in keys.into_iter().take(limit) {
        let Some(info_set) = table.get_info_set(key) else {
            continue;
        };
        let Some(data) = table.data.get(&key) else {
            continue;
        };
        let avg = data.average_strategy();

        eprintln!(
            "key={} player={} hand={:?} history={:?}",
            key.0,
            info_set.player,
            info_set.starting_hand,
            info_set.history.actions()
        );
        for (action, prob) in data.actions.iter().zip(avg.iter()) {
            if *prob > 0.001 {
                eprintln!("  {:?}: {:.1}%", action, prob * 100.0);
            }
        }
        eprintln!();
    }
}
