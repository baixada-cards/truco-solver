# Truco CFR Autoresearch — Agent Instructions

## Context

You are optimizing a Counterfactual Regret Minimization (CFR) algorithm for
solving the Brazilian card game Truco. The game has been abstracted into
information sets and a pre-built game tree. Your job is to find the algorithm
variant that minimizes **exploitability** within a **fixed 10-minute time budget**.

## Setup

1. Read the in-scope files for full context:
   - This file (`program.md`) — your instructions.
   - `crates/truco-solver/src/cfr_experiment.rs` — **the file you modify.**
   - `results.tsv` — experiment history.
2. Verify the project compiles: `cargo check -p truco-solver`
3. Confirm setup, then begin experimenting.

## The experiment file

`crates/truco-solver/src/cfr_experiment.rs` contains three functions:

- **`init()`** — allocate any per-solve state you need.
- **`iterate()`** — one CFR iteration: traverse all trees, update regrets and
  strategy. This is the hot loop.
- **`post_iterate()`** — called after each iteration. Apply discounting,
  pruning, rescaling, or any post-processing.

You may add helper functions, structs, constants. You may change the algorithm
in any way — the only constraint is the interface (the three function signatures).

Game trees are PACKED (2026-07-03): nodes are 12-byte structs with a shared
edge array — match on `tree.view(node_id)` (`NodeView::Terminal { payoff_p0 }`
/ `NodeView::Player { player, table_idx, edges }`), iterate `edges` (each has
`.child` and `.action()`), and recover an info-set key as
`prebuilt.info_sets[table_idx as usize].0` (the experiment's `traverse_tree`
now takes an `infos` slice for exactly this).

The internal `traverse_tree` helper now receives the hand's `dealer: Player`
(0 for `entry.tree_dealer_0`, 1 for `entry.tree_dealer_1`). The solver is
dealer-exact: at non-terminal continuations it looks up the match value at
`match_values.get(new_score.zero, new_score.one, 1 - dealer)` because the dealer
strictly alternates between hands. If you rewrite the traversal, thread the
dealer through and keep the `1 - dealer` continuation lookup.

## What you optimize

**Exploitability** after 10 minutes of iteration time (excluding tree build).
Lower is better. Current best is in `results.tsv`.

Exploitability ε means a perfect opponent can achieve a win rate of
`(1 + ε) / 2` against the strategy. For reference:
- ε = 0.1575 → opponent wins 57.875% (current best recorded autoresearch run)
- ε = 0.05 → opponent wins 52.5%
- ε = 0.01 → opponent wins 50.5% (our target)
- ε = 0.001 → opponent wins 50.05% (very strong)

As of the current `results.tsv`, the best kept autoresearch result is:

- commit `38fd48c`
- exploitability `0.157519`
- 70 iterations in `866.9s`
- description: `DCFR+ combining CFR+ regret clamping with DCFR-style discounting (α=1.5, γ=2) - best of both worlds`

This benchmark is stricter than the longer manual solver comparisons elsewhere
in the repo. Do not compare the raw numbers directly without accounting for the
shorter fixed iteration budget.

## What you CAN do

- Modify `cfr_experiment.rs` — everything is fair game.
- Change the regret update formula, discounting schedule, strategy computation.
- Add auxiliary data structures (per-info-set state, etc.)
- Try entirely novel approaches.
- Use any algorithm from the CFR family or invent new ones.

## What you CANNOT do

- Modify any other Rust file.
- Change the evaluation metric or time budget.
- Add external crate dependencies.

## Running an experiment

```bash
cargo run --release -p truco-solver --bin experiment 2>&1 | tee run.log
```

The binary will:
1. Build game trees (~4 min, excluded from budget)
2. Run your algorithm for exactly 10 minutes
3. Compute and print exploitability

Extract the result:
```bash
grep "^exploitability:" run.log
```

## The experiment loop

LOOP FOREVER:

1. Read `results.tsv` to see what's been tried and what works.
2. Think of a new idea. Consider:
   - What worked before? What failed? Why?
   - Are there academic papers on CFR variants you can draw from?
   - Can you combine two ideas that each helped a little?
   - Is there something fundamentally different to try?
3. Modify `cfr_experiment.rs` with your idea.
4. `git commit -m "experiment: <short description>"`
5. Run: `cargo run --release -p truco-solver --bin experiment > run.log 2>&1`
6. If compilation fails, fix and retry.
7. Extract result: `grep "^exploitability:" run.log`
8. Log to `results.tsv`.
9. If exploitability improved: **keep** (advance branch).
10. If worse or equal: **discard** (`git reset --hard HEAD~1`).

**NEVER STOP.** The human may be asleep. Keep running experiments until
manually interrupted. If stuck, try wilder ideas.

## Known algorithm variants to consider

- **CFR+**: Clamp negative regrets to 0. Simple, solid baseline.
- **DCFR**: Discount old regrets by t^α/(t^α+1). Typically α=1.5, β=0, γ=2.
- **Linear CFR**: Weight iteration t by t (linear growth).
- **PCFR+ (Predictive CFR+)**: Use momentum on regrets.
- **Alternating vs simultaneous updates**: Alternate which player is traversed,
  or update both simultaneously.
- **Regret matching+** vs **regret matching**: Different strategy computation.
- **Warm starting**: Initialize regrets based on card strength heuristics.
- **Adaptive discounting**: Change discount schedule based on convergence rate.

## Game-specific hints

- The game tree has ~100M nodes across ~140k deals.
- There are ~11.2M unique information sets.
- At score 11x11, there are no raises (mão de onze), so the game tree is
  relatively simple: accept/fold → 3 rounds of card play.
- Each iteration traverses all deals. Per-iteration cost is dominated by
  tree traversal, not post-processing.
- Exploitability computation costs ~15s (same as one iteration). It is
  computed by the harness after the time budget expires — not by your code.
- The checked-in `cfr_experiment.rs` is just the current working file, not
  necessarily the best experiment ever recorded. Treat `results.tsv` as the
  source of truth for the best known autoresearch result.
