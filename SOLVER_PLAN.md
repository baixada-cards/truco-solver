# Truco Solver Plan

## Goal

Build a new Rust crate `truco-solver` that computes Nash equilibrium strategies for Truco via Counterfactual Regret Minimization (CFR), solving the game incrementally by score state (starting at 11x11, then 10x11, 11x10, etc.), with first-class support for querying strategy profiles (win%, action distributions, EV).

---

## 1. Card Abstraction

Truco has 40 physical cards (10 ranks × 4 suits). The turnup card defines which rank becomes manilha, and for non-manilha cards suit is irrelevant to strength. We collapse the card space, but the **turnup acts as a blocker** — it is removed from the deck and cannot appear in either player's hand, which affects the available card distribution.

**Abstract card types (13 total per turnup class):**
- 9 non-manilha strength levels (suit-independent). These are the 9 ranks that are *not* the manilha rank, ordered by their `Rank::index()`. We call these `Plain(0)` through `Plain(8)`.
- 4 manilhas distinguished by suit: `Manilha(Diamonds)`, `Manilha(Spades)`, `Manilha(Hearts)`, `Manilha(Clubs)`.

### Turnup blocker effect

The turnup card is a specific physical card (e.g., 7 of Hearts) that is removed from the deck and placed face-up on the table. It determines which rank becomes manilha: **all four cards of the next rank** are manilhas (e.g., if the turnup is a 7, all four Queens become manilhas — regardless of suit).

The turnup card itself is always a non-manilha (a plain card). The blocker effect: since the turnup is removed from the deck, only **3 copies** of its rank remain available (instead of the usual 4). For example, if the turnup is a 7 of Hearts, there are only 3 remaining 7s in the deck (Diamonds, Spades, Clubs). All other non-manilha ranks still have 4 copies. All 4 manilhas are available (they are a different rank from the turnup).

**Key clarification:** The turnup's rank is still a weak card — the 7 itself has its usual strength. What changes is purely the availability: players are slightly less likely to hold that rank. The next rank (Queens in this example) become manilhas, and all four are available.

### Turnup classes

Two turnups are strategically equivalent if they produce the same blocked plain-strength level. Since the turnup's rank is always one of the 10 ranks, and the manilha rank is determined by the turnup rank, there are 10 possible turnup ranks. Under our abstraction, the turnup rank maps to one of 9 plain strength levels. Two turnup ranks (Two and Three) map to the same blocked plain level (8), yielding **9 distinct turnup classes**.

Each turnup class is defined by `blocked_plain_level ∈ {0, 1, ..., 8}`:
- All 4 manilha cards are available (one per suit)
- The blocked plain level has only **3** remaining cards
- All other 8 non-manilha strength levels have **4** cards each

**Deck composition per turnup class:**
- 39 cards total (40 minus the turnup)
- 4 manilhas + 3 blocked + 32 other = 39 ✓

**Turnup class weights:** Classes 0–7 each correspond to 1 of the 10 equally likely turnup ranks → weight 1/10. Class 8 corresponds to 2 turnup ranks (Two and Three) → weight 2/10.

**Cross-class solving:** We solve each turnup class independently. Information sets include the turnup class, so strategies for different turnup classes are completely independent. This means **all 9 turnup classes can be solved in parallel** — a key parallelism opportunity.

**Simplification possibility:** We could approximate by ignoring the blocker effect (pretending 4 copies of every plain level exist). This is a ~2.5% distortion in card availability for one level. Worth benchmarking after the exact solver works.

### Mapping concrete to abstract

```rust
enum AbstractCard {
    Plain(u8),      // 0 (weakest) to 8 (strongest)
    Manilha(u8),    // 0=Diamonds, 1=Spades, 2=Hearts, 3=Clubs
}

struct TurnupClass {
    blocked_plain_level: u8,  // which Plain(i) has only 3 copies
}
```

Mapping: If `card.rank == manilha_rank` → `Manilha(suit_strength)`. Otherwise → `Plain(strength_index)` among the 9 non-manilha ranks.

### Abstract hands and deals

An abstract hand is a sorted multiset of 3 `AbstractCard` values. An abstract deal is a pair of abstract hands drawn without replacement from the 39-card abstract deck. The combinatorial weight of a deal accounts for the multiplicities of each abstract card type (e.g., C(4,2) = 6 ways to pick 2 copies of a 4-available plain type).

We enumerate all ~140k abstract deals per turnup class with their normalized weights (summing to 1.0).

---

## 2. Information Sets

An information set encodes everything a player knows at a decision point:

```rust
struct InfoSet {
    player: Player,
    is_dealer: bool,                 // position: is this player the pé?
    turnup_class: TurnupClass,
    starting_hand: AbstractHand,     // sorted [AbstractCard; 3]
    history: ActionHistory,          // sequence of visible actions
}
```

`is_dealer` (position) is public information and strategically real — the pé
plays last each trick, a measured +5.6pp edge. Without it the same key appears
in both dealer trees of a solve: most visibly the mão-de-onze accept/fold node
(empty history), but also card-play nodes where the visible action string
coincides while own/opponent attribution differs. That merging forced CFR into
one position-averaged policy for two different states and was the structural
cause of the 11×10 exploitability wall (~0.33). With position in the key, the
dealer-0 and dealer-1 games share no info sets — they are fully independent
games coupled only through the match-value table, and can be solved in
separate processes (`solve-tc --dealer N`).

### Action history encoding

```rust
enum AbstractAction {
    PlayFaceUp(AbstractCard),
    PlayFaceDown(AbstractCard),      // own perspective: knows which card
    OpponentPlayedHidden,            // opponent's face-down: identity unknown
    Raise(u8),
    AcceptRaise,
    Fold,
    AcceptEleven,
    FoldEleven,
}
```

When building the info set for player P:
- P's face-down plays: include card identity (`PlayFaceDown(card)`)
- Opponent's face-down plays: `OpponentPlayedHidden`
- All other actions: fully visible

### Info set hashing

Info sets are hashed to `u64` keys via AHash for O(1) lookup in the strategy table.

---

## 3. Game Tree and Engine Integration

### Two-layer architecture

We use **two complementary representations** that coexist:

1. **Engine layer (source of truth):** The existing `Engine` from `truco-engine` handles all game logic — legal actions, action application, round resolution, mão de onze, raise ladder, etc. During tree building, we create `Engine` instances via `TraversalState` and use clone-and-advance to explore all branches. The Engine's extensive test suite gives us confidence in correctness.

2. **Pre-built tree layer (fast iteration):** Before CFR iterations begin, we traverse the Engine once to build a compact `GameTree` stored in a flat arena (`Vec<GameNode>`). Each node is either `Terminal { payoff_p0 }` or `Player { player, info_set_key, actions: Vec<(AbstractAction, NodeId)> }`. CFR iterations then operate purely on these pre-built trees — **zero Engine calls, zero heap allocations per node**.

This separation means:
- We get the Engine's correctness guarantees for the game logic
- We get arena-based speed for the hot CFR loop
- If we ever implement a lighter-weight game state, we can validate it against the Engine

### TraversalState

The `TraversalState` bridges abstract and concrete:
- Wraps an `Engine` instance + turnup class + abstract hands + action histories
- Maintains a `card_map: Vec<(String, AbstractCard)>` for bidirectional lookup
- Handles deduplication: multiple concrete cards may map to the same abstract action
- Realizes abstract deals into concrete cards for the Engine

### Game tree size at different scores

**Critical observation:** At mão de onze (score 11), **raises are not available**. The only extra node is the accept/fold eleven decision at the top. This means the 11x11 game tree is actually **smaller** than trees at lower scores where the full raise ladder (1 → 3 → 6 → 9 → 12) applies. Lower scores like 5x3 have much deeper raise/fold/accept branching.

---

## 4. CFR Algorithm

### Algorithm: CFR+ (primary)

| Algorithm | Key idea | Pros | Cons |
|-----------|----------|------|------|
| **CFR+** | Clamp negative regrets to 0, linear-weighted averaging | Fast convergence (~10x fewer iterations than vanilla), simple | Slightly more complex than vanilla |
| **MCCFR (External Sampling)** | Sample opponent + chance nodes | Low memory, handles large trees | Noisy, needs many more iterations |
| **DCFR (Discounted CFR)** | Discount factors on regrets | Best known convergence rates | More hyperparameters |
| **DCFR+** | DCFR discounting plus CFR+ regret clipping | Natural extension of the current best measured solver; autoresearch found a promising hybrid | Needs first-class main-solver implementation and apples-to-apples benchmarks |
| **PDCFR+** | Predict next-iteration regrets/advantages on top of DCFR+ | Promising if Truco regrets are predictable between iterations | More state and tuning; not implemented yet |
| **LCFR (Linear CFR)** | Linear weighting (DCFR special case) | Near-DCFR, simpler | Less battle-tested |

**Plan:** Start with CFR+ for all score states. The pre-built tree approach makes CFR+ feasible even for larger trees. If memory becomes the bottleneck at lower scores (where raise branching explodes the tree), we fall back to MCCFR external sampling. DCFR is implemented and currently the best measured mainline variant at `11x11`; DCFR+ and PDCFR+ should be benchmarked as tabular variants before attempting any neural/model-free DeepPDCFR-style system.

### CFR+ specifics

- **Regret update:** `R(I, a) = max(0, R(I, a) + r(I, a))` — clamp to 0 eliminates "regret debt"
- **Strategy averaging:** linear-weighted by iteration `t`: `S_avg(I, a) += t * reach_prob * σ_t(I, a)`
- **Alternating updates:** iteration `t` updates regrets for player `t % 2` only

### MCCFR fallback — detailed tradeoff

**What changes:** Instead of enumerating all ~140k deals each iteration, MCCFR samples one deal (or a small batch). Additionally, at opponent nodes, it samples a single action weighted by the current strategy rather than recursing into all actions.

**The tradeoff in concrete terms:**
- **CFR+ full iteration:** ~140k deals × 2 dealer sides × full tree traversal ≈ **11s per iteration**. Each iteration makes guaranteed progress toward equilibrium. Convergence is smooth and deterministic.
- **MCCFR single-sample iteration:** ~0.08ms per iteration (140,000× faster). But each iteration provides a very noisy gradient signal. You need **orders of magnitude more iterations** to converge to the same quality.

**Total compute is similar or worse with MCCFR**, but it has advantages when:
- The full tree doesn't fit in memory (MCCFR doesn't need pre-built trees)
- You want to checkpoint/resume at fine granularity
- The chance node has so many outcomes that enumeration is truly infeasible

**For Truco:** CFR+ with pre-built trees is clearly better at 11x11 (140k deals fits comfortably). MCCFR might be needed at lower scores only if tree + strategy table memory exceeds available RAM.

### Core data structures

```rust
struct InfoSetData {
    cumulative_regret: Vec<f64>,        // per action, clamped ≥ 0
    cumulative_strategy: Vec<f64>,      // iteration-weighted sum
    actions: Vec<AbstractAction>,       // legal actions in fixed order
}

struct StrategyTable {
    data: HashMap<InfoSetKey, InfoSetData>,
}
```

### Convergence

We now have real convergence data for the `11x11`, `TC 0` benchmark:

- CFR+ reached exploitability **0.050954** at 100 iterations in ~3299s when
  exploitability was computed every iteration.
- DCFR (`alpha=1.5`, `beta=0`, `gamma=2`) reached **0.047481** at 120
  iterations in 2396.4s with `expl_every=10`.
- Tail convergence slows sharply after ~50 iterations, so naive `1 / T`
  extrapolations are too optimistic once we are already near equilibrium.

**Measurement caveat (2026-07-02):** the numbers above were computed with the
historical CLAIRVOYANT best response (per-deal max — the "best responder" saw
the opponent's hand) and are strict upper bounds; the "tail slowdown" was the
approach to the clairvoyance gap, not slow convergence. The exact per-info-set
best response is now the default measure; on mid-density 11×10 probes the same
strategies that measured 0.15–0.26 clairvoyant are at 0.009–0.012 exact. See
`SOLVER_BENCHMARKS.md` 2026-07-02 (later) before comparing exploitability
numbers across time.

Open work is no longer "do we have any convergence data?" but:

- extend the measurements to `10x11`, `10x10`, and lower-score states
- determine whether the same DCFR parameters remain best outside `11x11`
- quantify how much `expl_every` and pipeline parallelism change real wall time

---

## 5. Subgame Decomposition and Incremental Solving

### Terminal payoff at match level

When a hand ends at score (S0, S1) with hand_value V:
- Winner gets V points added to their score
- If new score ≥ 12: match ends → payoff = ±1
- Otherwise: payoff = continuation value from `MatchValueTable`

### Solving order

Bottom-up from highest total score:

```
Phase 1: (11,11) — mão de onze for both players, NO raises available
Phase 2: (10,11), (11,10), (10,10) — mão de onze for one player
Phase 3: (9,11), (9,10), (9,9), (10,9), (11,9)
...
Phase 12: (0,0)
```

Total: 144 score states (12×12 grid, 0..=11 for each player).

### Continuation value lookup

```rust
struct MatchValueTable {
    values: [[f64; 13]; 13],   // values[s0][s1] = P(player 0 wins match)
}
```

Terminal states: `values[12][_] = 1.0`, `values[_][12] = 0.0`.

For CFR terminal payoff: `p0_value = 2 * match_win_prob - 1` (maps [0,1] → [-1,1]).

### Mão de onze

When a player has score 11, they choose AcceptEleven or FoldEleven. Folding gives 1 point to the opponent. Accepting sets hand_value to 3. **No raises are available** during mão de onze hands, making these game trees smaller than typical hands.

### Turnup class independence

**Key insight:** Since information sets include the turnup class, and the turnup is public knowledge, **strategies for different turnup classes are completely independent**. There is no strategic interaction between turnup classes — they share only the match-level continuation values.

This means:
- All 9 turnup classes at a given score can be solved **in parallel**
- The match-level value for a score state is the weighted average: `match_value(s0, s1) = Σ_tc weight(tc) × cfr_value(s0, s1, tc)`
- On a 9+ core machine, turnup-class parallelism alone provides ~9× speedup

---

## 6. Strategy Storage and Serialization

### Size estimates

Per turnup class at 11x11:
- ~11.2M info sets (from benchmark)
- Each info set stores: key (8 bytes) + action indices (~4 actions × 2 bytes) + strategy (4 × 4 bytes f32) = ~32 bytes
- **~358 MB per turnup class** serialized
- 9 turnup classes × 358 MB = **~3.2 GB per score state**
- 144 score states × 3.2 GB = **~460 GB total** (if all score states have similar info set counts)

This is large but manageable with compression and selective loading. During solving, we only need one score state's strategy tables in memory at a time.

**Optimization opportunities:**
- Compress strategies (most info sets have only 2-4 actions)
- Store only average strategy, not cumulative regrets (for solved states)
- Use varint encoding for keys
- LZ4/zstd compression on the binary (typically 3-5× reduction)

### File layout

```
solutions/
├── meta.json                    # Index: solved states, iterations, exploitability
├── match_values.bin             # MatchValueTable (compact: 13×13 f64 grid = 1.3KB)
├── strategy_11_11_tc0.bin       # Strategy table per (score, turnup_class)
├── strategy_11_11_tc1.bin
├── ...
└── strategy_0_0_tc8.bin
```

### Loading for incremental solving

When solving score state (S0, S1):
1. Load `match_values.bin` for continuation values (tiny: 1.3KB)
2. Do NOT load strategy tables of other states — only the match-level value matters
3. Only one full strategy table set (9 turnup classes) in memory at a time

---

## 7. Query Interface

### Core queries

```rust
struct SolutionReader {
    meta: SolutionMeta,
    match_values: MatchValueTable,
    // Lazily loaded strategy tables
}

impl SolutionReader {
    fn match_win_probability(&self, score: (u8, u8)) -> f64;
    fn hand_ev(&self, score: (u8, u8)) -> f64;
    fn strategy_at(&self, score, player, hand, history) -> ActionDistribution;
    fn simulate_matchup(&self, score, s0, s1, num_hands, rng) -> MatchupResult;
}
```

### Strategy trait

```rust
trait Strategy {
    fn action_probabilities(&self, info_set: &InfoSet, legal_actions: &[AbstractAction]) -> ActionDistribution;
}
```

Implementations: `EquilibriumStrategy` (from solver), `PureStrategy` (deterministic, for testing), `RandomStrategy` (for property-based testing).

---

## 8. Testing Plan

### 8.1 Unit tests for abstractions ✅ (implemented)

- Card abstraction round-trip: concrete card + turnup → abstract card → verify strength ordering
- Info set construction: equality, hashing, opponent hidden play masking
- Deal enumeration: weights sum to 1, all hands have 3 cards, no deal exceeds availability

### 8.2 Deterministic strategy tests ✅ (implemented)

- **Always fold vs always raise:** raiser wins every hand
- **Weakest-then-hide vs strongest-face-up:** open player dominates (face-down cards lose)
- **Mirror symmetry:** same strategy for both → ~50% win rate

### 8.3 Zero-sum invariant tests ✅ (partially implemented)

- **EV sums to zero:** For any strategy pair, `exact_ev(p0_perspective) = -exact_ev(p1_perspective)`
- **Per-node zero-sum:** Terminal payoffs satisfy `payoff_p0 + payoff_p1 == 0`

### 8.4 Property-based / randomized tests

Use randomized strategies to stress-test universal invariants:

- **Zero-sum:** `EV_0 + EV_1 == 0` for randomly generated strategy pairs
- **Probability conservation:** action probs sum to 1.0 at every info set
- **Legal action validity:** every sampled action is legal
- **Match termination:** no infinite loops, score monotonically increases
- **Score bounds:** scores never exceed 12

Generate `RandomStrategy` with uniformly random weights per info set, then verify invariants hold exactly (within float tolerance).

### 8.5 CFR convergence tests

- Solve 11x11 with increasing iterations, verify exploitability decreases
- At symmetric scores (S, S), match_win_prob ≈ 0.5
- Strategy probabilities always sum to 1.0 and are non-negative

### 8.6 Serialization round-trip tests ✅ (implemented)

- Match value serialize/deserialize round-trip
- Strategy table serialize/deserialize round-trip

### 8.7 Integration tests

- Full 11x11 solve → store → load → query
- Incremental solve: 11x11 → 10x11 with continuation values
- Verify solved equilibrium is unexploitable

---

## 9. Crate Structure

```
crates/truco-solver/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── abstraction.rs       # AbstractCard, TurnupClass, deal enumeration
│   ├── info_set.rs          # InfoSet, ActionHistory, AbstractAction, hashing
│   ├── game_tree.rs         # TraversalState, PrebuiltTrees, GameNode arena
│   ├── cfr.rs               # CFR+ solver, solve(), compute_game_value(), exploitability
│   ├── strategy.rs          # Strategy trait, EquilibriumStrategy, PureStrategy
│   ├── match_value.rs       # MatchValueTable, solve_order(), solve_order window helpers
│   ├── simulate.rs          # simulate_hand(), simulate_matchup(), exact_ev()
│   ├── storage.rs           # Serialization/deserialization
│   └── bin/
│       └── solve.rs         # CLI: solve-tc, compare, pipeline, treesize, benchmark
```

---

## 10. Implementation Status

### Completed ✅

- **Solver-independent runtime policy contract (2026-07-23):** `truco-policy-format` now owns the abstract card/action vocabulary, fixed-seed information-set keys, TPB1 mmap codec, and `truco-policy-bot/v1` manifest/schema. `truco-solver` consumes and compatibility-re-exports those types; the production `truco-policy-bot` depends directly on the small contract crate and retains `truco-solver` only as a development dependency for solve-side traversal parity tests. The contract documents TPB1's historical encoded-little-endian key ordering, has golden key/byte vectors, rejects unsafe manifest paths and malformed headers/action codes, and keeps CFR, checkpoints, experiments, and research tooling out of the runtime dependency graph. Verification: 26 policy-format tests, 14 policy-bot tests, and the solver library suite (93 passed, 1 ignored).
- **Phase 1:** Abstractions, info sets, deal enumeration (with tests)
- **Phase 2:** Game tree traversal via Engine, pre-built tree arena, PureStrategy, simulation
- **Phase 3:** CFR+ solver with pre-built trees, MatchValueTable
- **Phase 3b:** MCCFR (external sampling) solver — alternative algorithm for comparison
- **Phase 3c:** Exploitability computation (best-response calculation)
- **Phase 4:** Basic storage (match values, strategy serialization, full info-set metadata preserved in saved strategies)
- **Phase 5 (partial):** CLI solver binary with `solve-tc`, `benchmark`, `compare`, `treesize`, and `viewer` modes
- **Phase 5b:** `solve pipeline` — top-down incremental solve with score-window control (`--from-score` → `--to-score`), TC-weighted match values from `compute_game_value`, rayon parallelism across subgames per level, `match_values.bin` checkpoints (`MatchValueTable` + solved bitset), optional strategy dumps for the bottom solve level; `truco-ops solve` now includes a TUI pipeline planner for choosing the solve window before launching VM work
- **Phase 5c:** `solve-tc` full-state checkpoint/resume + wall-clock budget (`00e8d3c`) — `--time-budget`, `--checkpoint`/`--checkpoint-every` (atomic full-state writes: regrets + strategy + iteration), and `--resume` (validates score/TC/algo, overlays accumulators onto the rebuilt tree, continues iteration counting). Makes long single-subgame solves interruptible and genuinely resumable.
- **Full 11×11 solve:** all 9 turn-up classes solved to ~0.05 exploitability on 3 parallel VMs (2026-06-28, see `SOLVER_BENCHMARKS.md`); game value ≈ 0 per TC validates zero-sum correctness. Checkpoints retained for later `--resume` to push lower.
- **Determinism fix (`69497c5`):** `InfoSet::key()` is now fixed-seed (was per-process RNG, which silently broke cross-process `--resume`/warm-start); loads rekey from the info set.
- **Dealer-exact match values (`ca82882`):** the match-value table is indexed by `(score, dealer)`; continuations look up `mv(new_score, 1−dealer)` to honour the engine's strict dealer alternation. `solve dealer-advantage` reports the pé advantage (measured **55.6% / 44.4%** at 11×11). Necessary for every state below 11×11; 11×11 itself is unaffected (no continuations).
- **Raise-ladder pruning (`948e7c5`):** solver-side, value-exact — prune raises once `min(score)+stake ≥ MATCH_TARGET`; collapses the 10×10 ladder to {1,3}. Engine untouched.
- **`solve-asym` (`8745253`…`833813e`):** dealer-exact freeze-accept + equity policy-iteration for asymmetric mão-de-onze states. Validated the *card play* is solvable (opponent best-response gain ≈0.025) and the accept-all==11×11 identity, but the **accept-set iteration does not converge** (fictitious-play oscillation) — see the 11×10 saga in `RESEARCH_NARRATIVE.md`. Superseded by position-in-info-set; kept for diagnostics.
- **Position in the info set (2026-07-02):** `InfoSet` gained `is_dealer`, making the dealer-0 and dealer-1 games fully disjoint in key space — the structural fix for the 11×10 wall. On a tiny strided 11×10 config the pre-fix solver walls at expl ≈0.0965 while the fixed solver reaches 0.0003 (see `SOLVER_BENCHMARKS.md` 2026-07-02). Includes: `solve-tc --dealer {0|1}` for exact per-dealer solves at ~half the memory, `.dN`-suffixed artifacts, checkpoint meta carrying `dealer_filter`, and legacy loaders that expand pre-position artifacts into both position variants (so the stranded 11×11 checkpoints stay usable for resume/warm-start). Deal subsampling for tests is now strided (a prefix gave player 0 only one hand).
- **Exact best response (2026-07-02, `b3689e6`):** the historical exploitability measure was CLAIRVOYANT (per-deal max — could condition on the opponent's hand), a non-vanishing upper bound that manufactured the residual "walls" at density. `best_response_value` now computes the legal per-info-set best response; the old measure survives as `best_response_value_clairvoyant`. New `solve eval-ckpt` evaluates any checkpoint under the exact measure. `SyncCfrPlus` and `PcfrPlus` (buffered synchronous sweeps; predictive regret matching) added as first-class variants while falsifying dynamics hypotheses. All exploitability numbers recorded before this date are clairvoyant bounds — see `SOLVER_BENCHMARKS.md`.
- **Teacher export purification/certification (2026-07-09):** `.teach` remains the raw teacher artifact, but `solve export-chart` now purifies chart probabilities non-destructively and certifies them with the same exact legal BR machinery. Chart actions carry purified `p`, raw `raw_p`, and one-shot-deviation `q`; the top-level `certificate` (`study-purification-certificate/v1`) records `raw_eps`, `purified_eps`, `mass_removed`, `max_info_set_mass_removed`, `max_qgap_touched_pp`, `touched_info_sets`, `actions_zeroed`, `purify_max_prob`, `purify_min_qgap_pp`, `assert_qgap_pp`, `assert_max_info_set_mass`, `raw_max_info_set_mass_above_assert_qgap`, and `raw_touched_info_sets_above_assert_qgap`. Current thresholds from the step-0 sweep: export assertion counts only individually-small actions (`p < 1%`) and fails if total info-set mass above Q-gap `>5pp` exceeds **3%**; purification uses `p < 1%` and Q-gap `>5pp`; the Study UI display gate uses `p < 3%` and Q-gap `>1pp` unless raw residue is toggled on. The tc0/d0 11-column certified with `purified_eps <= raw_eps` for every rung and the derived Study chart artifacts are mirrored at `gs://truco-solver-runs/teacher2-20260704/study/`; raw teacher artifacts remain raw (see `SOLVER_BENCHMARKS.md` 2026-07-09).
- **Per-action hand-point EV (`pts`) in Study chart exports (2026-07-10):** `.teach` format bumped to v2, adding a `pts_values` tensor computed by the SAME `q_traverse` σ̄-vs-σ̄ walk as `q` — the terminal value is the engine's raw signed hand-point differential (`payoff_p0`, already ±1/±3/±6/±9/±12 including mão-de-onze and fold outcomes) instead of the match-equity value looked up through `MatchValueTable`, so `pts` needs no score/dealer context and costs one more numerator array, not a new traversal. `solve export-chart` now emits `pts` on every action by default (additive field, `study-chart/v1` format string unchanged — `truco-frontend/src/lib/study-data.ts` parses by field access). `pts` is intentionally NOT covered by the exact-BR certificate (`certificate.pts_certified: false`): certification measures match-equity exploitability, and "best response to raw hand points" isn't the objective either player actually plays. Unit tests (`teacher_export.rs`) cover sign convention (acting-player perspective, both signs), stake scaling (full {1,3,6,9,12} ladder at a regular node, fixed 1→3 jump at a mão-de-onze accept), and fold terminals, each cross-checked against an independently-written oracle traversal, not just the production code path. All 15 shipped Study spots (11×11/11×10 d0+d1, 11×9…11×0 d0, provisional 10×10 d0) were re-exported from their existing checkpoints (no re-solve) on a temporary GCP VM and verified to reproduce the prior `raw_eps`/`purified_eps`/`actions_zeroed` bit-for-bit; see `RESEARCH_NARRATIVE.md` 2026-07-10 for the full story including the 10×10 `--allow-residue` wrinkle.
- **Epsilon-tremble warm-started refinement of off-equilibrium spots (2026-07-11, implements `QUESTIONS.md` Q3 route 1):** `cfr::TrembleSchedule` floors every player node's behavior strategy at `σ'(a) = ε/|A| + (1-ε)·σ(a)` before it is used for reach propagation, regret, and average-strategy accumulation — CFR on a perturbed extensive-form game (Selten/van Damme trembling-hand construction) rather than the base game, so every info set (including previously near-zero-reach branches that regret-matched to near-uniform noise) accumulates real training. Flag-gated and default OFF (`SolveConfig.tremble: Option<TrembleSchedule>`, `solve-tc --tremble-eps E [--tremble-eps-end E2]`, annealed schedule; `--extra-iters N` resolves a resume-relative iteration budget without a discovery pass); wire/checkpoint formats unchanged, so `--resume` against existing checkpoints keeps working unmodified. Exact-BR certification is untouched — it measures the true best response to whatever the accumulators end up holding, trembled or not. Also exports the solver's own `own_reach` tensor (already computed, previously unused) as an additive `study-chart/v1` row field, giving a genuine σ̄-traversal reach number instead of relying on the study lab's client-side path-product approximation. All 5 shipped tc0 spots (11×11 d0/d1, 11×10 d0/d1, 10×10 d0 — the columns actually in `truco-frontend/public/study/manifest.json`) were warm-started (+200 iters at 11×11/11×10, +100 at 10×10 given its ~17× higher per-iteration cost once the tremble floor defeats zero-prob pruning) and re-exported/re-certified: purified eps grew from ~1e-5/2.4e-4 to at most 1.3e-3 absolute (small, expected, reported), self-loss-flagged share collapsed 8-16×, and the concrete garbage probe from `RESEARCH_NARRATIVE.md` open question 0 (a hand hiding manilhas at certain-loss q=-1.0) now plays the correct card. Full numbers, the RAW-vs-purified own-reach nuance, and four ops bugs fixed along the way are in `SOLVER_BENCHMARKS.md` 2026-07-11 and `RESEARCH_NARRATIVE.md` 2026-07-11.
- **Per-infoset BR-gap Study release (2026-07-16):** `export-chart` retains additive `table_idx` keys so the Study Lab can lazily join charted holdings to a separate `BrGapRecord` table without bloating each JSON window. Fresh exports from the final post-tremble checkpoints now cover all five refined solutions: 11×11 d0/d1, 11×10 d0/d1, and 10×10 d0. The release has 34.6M reachable records; its five compressed BR tables total 208.6 MiB (the 10×10 table is 156.3 MiB). The UI uses adversarial BR-gap as the headline quality cue, with self-loss and own-reach as supporting diagnostics. Release preparation validates each chart and binary header against manifest score/TC/dealer bindings. The original 11×10 d1 pilot discrepancy is closed: it came from an older uncertified export, while the released replacement is from the final stage-5 checkpoint and reports raw eps 0.0030554 and purified eps 0.0001092. The 10×10 export retains its known provisional caveat because raw purification residue is 3.936%, above the normal 3% guard, despite completing exact raw/purified global BR evaluation.
- **Experimental cross-turn-up policy transfer (2026-07-16):** `solve-tc --warmstart-from CKPT --warmstart-cross-turnup` explicitly rekeys a same-score/dealer checkpoint by turn-up class and preserves both regrets and the tremble-trained average. The all-deal TC0→TC1 11×11 canary needed one target iteration (36.7s including its exact BR evaluation): the mapped profile certified at raw/purified epsilon 0.0027315/0.0004361, while its reach-weighted local BR-gap mean was 0.00937pp versus 0.729pp for the native, non-refined TC1 profile. This is a strong cheap-expansion result, not an implicit default; repeat across the remaining classes and retain exact certification. Teacher export also now projects long-lived strategy action vectors onto the current treepack by action identity and validates tensor/wire lengths before atomic rename.
- **`accum-f32` reduced-precision accumulators (2026-07-17, plan 84 Phase 1):** cargo feature on `truco-solver` narrowing `DenseAccum`'s persistent `regret`/`strategy` arrays to f32 while keeping the transient per-sweep buffers (`pending`, `last`, parallel `LocalAccum`) f64 — within-sweep accumulation stays wide and each cumulative slot sees exactly one narrowing per iteration (widen-add-narrow at every fold site, a no-op in the default f64 build). All traversal math stays f64. Checkpoint/strategy artifacts stay f64-formatted via a generic `AccumElem` widening boundary in `storage.rs`, so files from both builds are byte-compatible and the default build's bytes are unchanged. Serial-vs-parallel trajectories are only f32-rounding-close under the feature (they narrow strategy sums at different points), so that test gains a feature-conditional tolerance. 97/97 unit tests pass in both modes. The plan-79 Phase 6 accuracy A/B (exact-BR ε vs f64 control + RSS) is the adoption gate — pending, see plan 84.
- **CFR-D safe subgame re-solving, core implementation (2026-07-18, plan 84 Phase 3):** three new pieces. `subgame.rs` decomposes every deal tree at the round-2 boundary by REPLAYING the build recursion (`TraversalState` + `abstract_legal_actions`), so node ids, engine-state boundary tests, and both players' info-set views are correct by construction; boundary crossings group into subgames by public-state key (face-down identity masked), and subtrees are contiguous preorder id ranges. `cfr.rs` gains `best_response_boundary_values` (per-node BR values through the certified 3-pass memo — the CBV source) and `resolve_subgame` (the Burch–Johanson–Bowling Terminate/Follow gadget as a code-level side accumulator over the existing packed subtrees; SyncCFR+ freeze-then-fold discipline for both the table and the gadget buffers). `resolve.rs` orchestrates: root weights `π_c·π_p` via `reach_excluding`, per-view CBV aggregation, two gadget runs per subgame (one per resolved player), composition, and full-tree exact-BR certification. Verified: boundary invariants (unique crossing per path, spans match built arenas, views refine public state, acting views equal registry keys), composed ε ≤ blueprint ε + slack on a solved 8×8 subset, and the repair experiment — corrupting a subgame to uniform and re-solving from the CLEAN boundary summary recovers >80% of the exploitability damage, demonstrating subgame play is reconstructible from boundary values alone. Remaining: CLI surface + production-scale 10×10 validation (GCP), then Phase 4 subgame-parallel fleet shape.
- **Budgeted cross-score/turn-up Study expansion (2026-07-16):** the explicit flag is now named `--warmstart-profile-transfer` (the old cross-turn-up spelling remains accepted) and permits another score in the same tree band. Four TC0 canaries showed why exact selection, not unconditional replacement, is required: both 11x0 dealers transferred at raw eps below 0.00027, 11x9/d1 transferred at 0.00604, but 11x9/d0 remained at 0.01951 after a bounded 10-minute tail. The resumable fleet probes each target for at most 90 seconds, selects transfer only within raw/purified/local-BR limits, and otherwise publishes the native converged strategy's real BR table. Candidate rows upload atomically, retry by skipping complete rows, stop only between profiles at the soft budget deadline, and have an independent eight-hour VM deletion backstop.

### Remaining 🔲

- [x] **11×10 SOLVED (2026-07-03).** All 18 subgames to exact exploitability 0.0058–0.0121; mv(11,10,·) = 0.6218/0.6297 (position-inverted — see `SOLVER_BENCHMARKS.md`); artifacts + seeded `match_values.bin` in `gs://truco-solver-runs/11x10-20260702/`. Dealer-0 games stopped at the 2000-iteration cap slightly above the 0.01 target; resumable if tighter is needed.
> **Direction change (2026-07-04): the full tabular grid is superseded by the
> neural distillation pipeline — `plans/71-neural-distillation-pipeline.md`.**
> Exact solving continues only as the TEACHER: a diverse band sample to very
> low ε (mão column, {1,3} triangle, a {1,3,6} sample), then supervised
> distillation, self-play extension (score/stake as input features so the net
> warm-starts uncovered deep states from its learned band strategy), and a
> deferred-scope exploitability certification of the net. Items below that
> pertain to the full grid are parked, not deleted.

- [x] **Stage B / 10×10 SOLVED (2026-07-09/10).** All 9 TCs converged below ε=2.5e-4 (900-930 iters, 4.90-7.56h wall each) on `n2-custom-2-76800-ext` spot workers; dealer-0 only, `mv(10,10,1) = 1 − mv(10,10,0)`. See `SOLVER_BENCHMARKS.md`'s "2026-07-09 (later)" entry for the fleet results. Remaining Stage-B work is just 9×9 and 10×9 (27 jobs across both, all TCs/dealers) — see the cost entry below.
- [x] **Ladder-by-score cost map — treesize survey completed (2026-07-15/16).** `min(score) + stake ≥ 12` prunes raises, so: min ≥ 9 → ladder **{1,3}**; min 6–8 → {1,3,6}; min 3–5 → {1,3,6,9}; min ≤ 2 → full {1,3,6,9,12}; mão-de-onze states (either player at 11) have no raising at all. The min=8, min=5, and min≤2 bands are now measured EXACTLY (not sampled) via the new `solve count-tree` DFS-counting tool — see `SOLVER_BENCHMARKS.md`'s "2026-07-15/16 — Exact tree-size census" entry for the full table, the job-count derivation, and the refined ≈$505K spot / $1.12M on-demand estimate for everything still remaining.
- [x] **Policy-aware census + actual-policy decision run (2026-07-16; static closure rejected).** `solve count-tree` counts profile support, each unilateral best-response closure, and their union at configurable thresholds without materializing the tree; compact loading avoids reconstructing the solver table. On the actual solved 10x10/TC0/d0 policy and all 140,118 deals, exact/`1e-8` BR union was the raw 39.51M info sets, `1e-6` retained 39.10M, and even `1e-4` retained 31.27M (versus 12.05M profile-only). Therefore do not build the proposed static all-response-actions restricted arena. The next structural candidate is a space-for-time deterministic-BR/double-oracle census that adds only chosen deviations and retains a global exact-BR stopping test. Full numbers and <$0.20 decision-run cost are in `SOLVER_BENCHMARKS.md` 2026-07-16.
- [x] **Chosen-action BR union + actual restricted-action closure census (2026-07-16; builder gate passed at `1e-4`).** Exact per-infoset BR choices feed two counters: three fixed paths and the cross-combined local action closure a restricted solver would allocate. On solved 10x10/TC0/d0, `1e-4` closure retained 74.99M nodes / 12.15M info sets: only 0.76% above fixed paths and 4.70x / 3.25x below raw. Exact support shrank only 1.47x / 1.15x. Proceed to a cheap restricted-solver prototype, but re-solve and rerun global exact BR until gain <=0.01; round zero is not certified. Both production counts peaked at 28.1 GiB and cumulative compute stayed <$0.50.
- [x] **Generic same-band warm-start transfer + production A/B (2026-07-16; 1.61x wall win).** `--warmstart-from` detects identical band signatures and copies exact-key/exact-action regrets while resetting cumulative average + iteration count for changed terminal utilities; the disk-backed source is now dropped before dense CFR allocation in both pipeline and `solve-tc`. The old 11x11→mão remap remains separate. The $0 7x7→8x8 gate was 90→10 iterations / 2.7x wall. The <=$2 production scout (all deals, actual 10x10→10x9 TC0/d0) was 90→40 iterations and 45m13s→28m03s: **1.61x end-to-end**. Use that measured factor, not the local 9x iteration result, for planning.
- [x] **Restricted solver + iterative exact-BR oracle (2026-07-16; safe, strict production composition rejected).** A compact solver-ready action-mask arena preserves action identity; same-band warm starts transfer retained-action regret subsets. Every round maps back to a complete full-tree profile, runs both exact BRs, and monotonically adds their selected actions. The local stress test caught a false restricted epsilon and the usable 7x7→8x8 composition was 19% slower. The <=$2 actual 10x10→10x9 scout certified `0.009965` after three rounds, but took 41m54.5s / 61.2 GiB versus 28m02.9s / 55.7 GiB for full warm: 49% slower because repeated solves plus 537s of audits erased the 2.64x final info-set shrink. Do not deploy or assign a strict-`0.01` multiplier. A deliberately looser `0.011` would accept round zero at exact `0.010560` and provisionally ~1.35x speed, but that no-output single spot is only a future approximation gate.
- [x] **Warm-up + reversible regret-pruning gate (2026-07-16; do not enable for epsilon=0.01).** Opt-in SyncCFR+ now tracks an unclamped `f32` shadow regret, temporarily skips only zero-current-probability traverser actions below a negative threshold, revisits both player sweeps together, and re-warms after resume. A conservative 300-deal warm A/B matched exact exploitability within `8e-6` and saved 4.7% wall; an aggressive cadence worsened epsilon 3.1x. Since the same warm target already crosses epsilon=0.01 before pruning begins, the feature has zero stopping-time benefit and does not justify a paid scout. See `SOLVER_BENCHMARKS.md` 2026-07-16.
- [x] **First lossless representation A/B (2026-07-16; safe but modest).** SyncCFR+ now overlays player-local pending-regret slots, and solve-time legal-action metadata uses a 16-byte boxed slice rather than a 24-byte `Vec`. The accepted path wrote byte-identical checkpoints/strategies/values and identical exact-epsilon trajectories. Its fixed metadata saving is 8 bytes/info set (~316 MiB at 39.51M rows), but the small A/B showed no wall-time win; assign no cost multiplier. An inline SmallVec candidate increased RSS and was rejected. Reduced-precision accumulators remain an opt-in accuracy benchmark, not an assumption.
- [x] **Seeded mini-batch MCCFR gate (2026-07-16; rejected for deep bands).** Implemented persistent sparse mini-batches, lazy neighboring-policy pseudo-regret seeds, and fixed held-out-panel evaluation. Batch 32 was effectively identical to batch 1 locally. In the cost-capped 10x10-policy -> 0x0 full-ladder Spot scout, 1M samples grew 20.74M sparse info sets / 14.3 GiB RSS while held-out-panel epsilon worsened 0.226 -> 0.241 instead of approaching 0.01. Keep the experiment surface, but do not scale this path; full CFR remains preferable where it fits and deterministic-BR/double-oracle structure is the next candidate.
- [x] **Proof-scoped ex-ante pruning (2026-07-16; correct, no production cost credit).** The builder removes second-mover hides in rounds 2/3, final-leader hides only when no raise response exists, and a final raise call that must lose after a hidden/globally-weakest lead. It preserves round-2 leader concealment, final reveal/raise signaling, and every legal bluff re-raise. Exact all-deal counts fall 1.76-2.26x by nodes but only 1.07-1.08x by info sets. The actual 10x10->10x9 warm target certified at 0.009622 in 40 iterations, yet took 30m49s / 61.5 GiB peak versus 28m03s / 55.7 GiB before. Keep the lossless action rules; assign no solve multiplier. The follow-up observable-card multiplicity generalization produced exactly zero new pruning at 300 deals: `Plain(1)`'s required history is unreachable and higher ranks cannot exhaust all lower copies from the available observations. The proof family is closed.
- [x] **Direct-to-dense checkpoint warm load (2026-07-16; lossless shallow cost win).** Disk warm starts no longer build a full empty target `StrategyTable`: same-band, cross-turn-up, and mao-remap source rows project directly into dense accumulators. Unit fixtures match the compatibility path exactly, and the all-deal 10x10->10x9 production trajectory/value were identical. Wall fell 30m49s -> 29m03s and peak RSS 61.47 -> 43.58 GiB; a right-sized 55-GiB worker is about 1.46x cheaper for this exact shallow path. Do not assign a full-grid multiplier until a deep-band scout passes; the remaining peak is the fully deserialized source checkpoint, so the next gate is a row-streamable format.
- [x] **Row-streamed source checkpoint (2026-07-16; lossless, output overlap is next).** Current positioned checkpoints were already serialized as metadata + row count + individual rows, so `CheckpointStream` now validates and projects one source row at a time; legacy files retain the compatibility loader. Same-band, cross-turn-up and mao-remap arrays match exactly. The all-deal 10x10->10x9 run again certified `0.009622` / value `0.198171`, with 30m49s wall and 31.60 GiB VmHWM versus 29m03s / 43.58 GiB direct-dense. A right-sized 40-GiB worker is ~24% cheaper per completed shallow job despite 6.1% extra wall. The remaining peak occurs after solving, when dense accumulators overlap the reconstructed output `StrategyTable`; stream dense rows directly to checkpoint/strategy next. Deep-tier cost credit is still prohibited.
- [x] **Sampled reach/error allocator (2026-07-16; prioritization only).** `allocation-scout` uses deterministic deal panels, complete score-DAG profile evaluation, player-swapped dealer projection, explicit missing-action fallback and both unilateral one-hand deviations. It reports panel ranges and cumulative profile-reach-weighted error mass—not exploitability or equity pp. A 96-deal/three-panel TC0 donor run assigned 32.9% to the full ladder and 25.4% to `{1,3,6,9}` under `all-except-raise`; alternate uniform missing-action support also left a majority in those two bands. This robustly prioritizes a deep-band raise-policy benchmark but cannot set final epsilons.
- [x] **Compact exact-BR production gate (2026-07-17; memory/time pass, portability hazard found).** Dynamic depth-wise DFS retains compact policy rows, one depth's counterfactual aggregates and chosen responder actions instead of a solver arena. It matches the arena oracle on 12 and 300 deals (`1.85e-10`, and 5e-11 on a matched 300-deal control); the all-deal 10x10/TC0/d0 run took 1,414s single-threaded at **5.95 GiB peak** on a 16-GiB Spot worker, so exact certification no longer needs arena-class RAM (~$10K -> ~$8K modeled whole-grid, still parallelizable). Its printed epsilon (0.016309 versus the checkpoint's certified 0.000248) is a projection artifact: pre-dominance checkpoints keep ~36% average mass on pruned hidden plays and cannot be ported onto the proof-pruned tree by any local row operation — certify/export old artifacts only on their own tree or after a warm re-solve. Projection is now explicit (`--project-dominated`) and reported (`COMPACT_BR_PROJECTION`), and `--legacy-tree` reconstructs the pre-prune tree: the all-deal self-certification on it reproduced the solve-era certificate exactly (epsilon 0.000248280, value to 12 digits), closing the equality loop between the two independent BR implementations at production scale. Details in `SOLVER_BENCHMARKS.md` 2026-07-16/17 and plan 79 Phase 8.
- [x] **Cost-optimization program (plan 79) — concluded/paused 2026-07-17.** Ran every lever behind a cheap discriminating gate ($2/feature cap): ε=1e-2 repricing, same-band warm starts, proof-scoped + asymmetric raise pruning, compact exact-BR certification, the phase-9 memory-representation chain, and negatives (MCCFR, static BR-union). Whole-grid exact bracket moved $505K → **~$12–16K**, still >10× the <$1K goal and walled by deep symmetric cells. Full ledger and cold-resume record in [EXACT_SOLVING.md](EXACT_SOLVING.md); the program is paused in favor of the neural approach ([plan 83](plans/83-neural-policy-approach.md)).
- [ ] **Build-path refactor (last of the review-pass three).** Tree building clones the whole `TraversalState` (engine + `String` card ids + linear `card_map` scans) at every node. Design: realize deals into index-based cards (u8 ids), replace `card_map` with a fixed array, and make `apply_abstract_action` advance a mutable state with undo (or a cheap copyable state struct) instead of cloning. Expected another ~2-4× on builds; matters at 10×10 scale where builds are tens of GB. Not started — the dense-table refactor (done) was the bigger win.
- [ ] **Measure the TRUE exploitability of the solved 11×11 strategies** with `eval-ckpt` (their recorded ~0.05 was the clairvoyance gap; the strategies are likely much closer to equilibrium than believed).
- [ ] **Full-game exploitability certification (after an assembled full policy).** Because the strategy is memoryless across hands (per-hand info sets, reshuffled deals, score+dealer as the only persistent state), an exact best response to the assembled match strategy decomposes by score state: backward-induct over the (score, dealer) DAG, running the exact per-info-set BR at each state with terminal payoffs from the *exploiter's own* BR-value table (not the mv table). Two sweeps (one per exploiting seat) yield the EXACT full-game exploitability at 0×0 — measured, not bounded — plus a per-state map of BR gains showing where refinement compute should go. The score DAG is cheap; today's materialized tree build + two-BR implementation projects to roughly **$10K** over the raw lattice (461.8s versus 22,932s for a tight 10x10 solve, ~2.0% of the $505K baseline), so it is not a cheap allocation loop. Plan 79 separates a <=$2 sampled reach/error allocator from a compact DFS/depth-wise BR evaluator that must first match the existing oracle exactly while removing solver-arena RAM. Per-subgame eps targets (currently 0.01 ≈ 0.5pp average match-win equity) remain arbitrary stopping points, not method limits.
- [ ] **(Optional) tighten mv(11,11,·) by resuming the legacy 11×11 checkpoints.** NOT a prerequisite for 11×10: an aliasing census (SOLVER_BENCHMARKS 2026-07-02 addendum) found zero position-aliased info sets at 11×11 — the ~0.05 was the lax target, not an abstraction floor, so the existing solve and its 0.5564/0.4437 values are valid as-is. Resuming (joint mode; legacy meta has `dealer_filter: None`) for a few more hours per TC would shrink the ~±0.01-scale tolerance those constants pass down the lattice.

Solver and autoresearch work is intentionally CLI- and agent-driven. This
repository owns the portable solver and `autoresearch/` harness; private VM
inventory, launch/fetch automation, and live credentials are owned by
`baixada-ops`. The former VPS web control plane and controller remain retired
rather than maintained as a second orchestration path.

- [ ] Extend convergence benchmarks beyond the current 11x11 baseline to 10x11, 10x10, and selected lower-score states
- [ ] Add first-class DCFR+ and PDCFR+ solver variants and benchmark them against DCFR
- [ ] Ship CFR autoresearch with a Claude Code-backed runner contract:
  - label the user-facing workflow as **CFR autoresearch**, not generic autoresearch, so future non-CFR research can live beside it cleanly
  - add launch controls for max iterations, max total LLM budget, max per-iteration LLM budget, Claude model, Claude effort level, and exploration mode
  - treat LLM budgets as LLM spend only, not VM/GCP compute spend
  - count max iterations as attempted proposals, not accepted improvements
  - hardcode model/effort choices initially instead of accepting arbitrary strings; default to worker `sonnet`/`high`
  - use `claude -p --max-budget-usd <amount>` for each iteration's hard budget cap, with the runner enforcing max iterations and overall budget
  - skip advisor in the first shipped workflow; interactive `/advisor` and hidden `--advisor` are too fragile for automation today, and a separate advisor call adds complexity before we know it improves outcomes
  - restrict Claude's edit tools to `crates/truco-solver/src/cfr_experiment.rs`, deny Bash and `.env*` reads during proposal generation, and reject any attempt whose post-run diff touches other files
  - support two exploration modes: backlog-guided exploration over the listed solver ideas, and free exploration within the CFR experiment interface
  - add MLflow tracking for parameters, metrics, artifacts, prompts, diffs, Claude output, and benchmark logs; default to logging directly to a persistent personal-server MLflow deployment reachable only over Tailscale through `CFR_AUTORESEARCH_MLFLOW_TRACKING_URI`, with a local file store retained as a development/offline fallback
  - let the private operations layer open the CFR MLflow dashboard, preferring its Tailscale-only tracking URL and falling back to a loopback `mlflow ui` for local file stores
  - add committed non-secret env examples and runtime `op://...` resolution that injects secrets into child processes without writing resolved values to disk, prompts, logs, or MLflow params
  - keep the runner responsible for full-file replacement, acceptance/rejection, git commits for accepted candidates, result logging, and budget accounting; the coding agent should focus on proposing/editing the CFR experiment
- [ ] Promote the viewer from a basic saved-strategy inspector to a richer query tool for arbitrary info sets
- [x] Full 11x11 solve (all 9 turnup classes) to a lax ~0.05 exploitability (2026-06-28); resume from checkpoints to drive lower
- [ ] Extend pipeline to full grid (0x0) with operational runbook
- [ ] Property-based tests with `RandomStrategy`
- [ ] Strategy compression for storage
- [ ] Finer-grained checkpointing (per subgame) if spot preemption becomes painful
- [ ] Neural net distillation (future phase)

---

## 11. Performance Estimates

### Measured: GCP benchmark (c2-standard-8, 11x11, TC 0)

Run date: 2026-03-24. Machine: c2-standard-8 (8 vCPU, 32 GB RAM), Debian 12.

| Metric | Value |
|--------|-------|
| Abstract deals per turnup class | 140,118 |
| Total game tree nodes (all deals, both dealers) | 99,736,516 |
| Unique info sets | 11,223,052 |
| Tree build time | **178s** (~3 min) |
| Per-iteration time (CFR+) | **~16s** |
| Exploitability computation | **~11s** |
| Memory (trees + strategy table) | **5,829 MB** (~5.7 GB) |
| MCCFR throughput | ~20,000 iters/sec |

### 11×11 convergence snapshot (measured)

| Iterations | Exploitability | Time (incl. tree build) |
|-----------|---------------|------------------------|
| 5 | 0.6493 | 274s |
| 10 | 0.2788 | 411s |
| 20 | 0.1196 | 518s |
| 50 | 0.0619 | 995s |
| 100 | 0.0510 | 3299s |

DCFR on the same subgame reached **0.0475** at 120 iterations in **2396.4s**
with `expl_every=10`. The main planning takeaway is that `11x11` is already
measured well enough to say the tail is slow; lower-score states and the full
raise ladder are now the bigger unknown than the topmost subgame itself.

### Planning projections (still approximate)

For `11x11`, `TC 0`, the current evidence supports these coarse planning
numbers:

| Target ε | CFR+ baseline (every-iter expl) | Notes |
|----------|---------------------------------|-------|
| 0.050 | ~100 iterations / ~55 min | directly measured |
| 0.030 | ~250 iterations / ~2.2 hr | extrapolated |
| 0.010 | ~700-900 iterations / ~6-8 hr | extrapolated, conservative |

These are useful for budgeting only. They should not be treated as full-game
runtime estimates because:

- `11x11` is the smallest tree in the game
- lower scores reintroduce the full raise ladder
- DCFR and `expl_every` improve wall time enough that the eventual production
  operating point will differ from the original CFR+ baseline

### Important caveats

1. **Lower-score tree sizes unknown.** The 11×11 trees are the smallest (no raise branching — mão de onze). At scores with full raise ladder (1→3→6→9→12), trees could be 3-5× larger. Must measure with `treesize` mode.
2. **Exploitability overhead.** Computing exploitability adds ~11s per evaluation. If computed every iteration, that's +68% overhead. Should be computed every 10-50 iterations in production runs.
3. **MCCFR is now a measured fallback, not the frontrunner.** At 11×11 it is materially worse than CFR+/DCFR at equal wall time, so it should be reserved for score states where pre-built trees stop fitting comfortably in memory.
4. **Within-subgame solves are still single-threaded.** The pipeline can parallelize across independent score/turnup jobs, but the hot loop for one subgame still uses one core.
5. **Saved strategy files are now inspectable, but the query UX is still thin.** Serialized strategies preserve full info-set metadata and can be loaded back into the viewer, but we still lack a polished arbitrary-query interface on top of that data.

---

## 12. Open Questions

1. **Lower-score tree sizes:** At scores without mão de onze, the full raise ladder (1→3→6→9→12) creates much deeper branching. How many info sets at e.g. 5x3? This remains the biggest unknown for full-game cost.

2. **Cross-score convergence:** Does DCFR keep its edge over CFR+ at `10x11`, `10x10`, and lower, or is the current result specific to `11x11`?

3. **Turnup class consolidation:** If equilibrium strategies are nearly identical across turnup classes, consolidating would save 9× compute. Worth checking after solving 11x11 for all 9 classes.

4. **Symmetry exploitation:** At symmetric scores (S, S), player 0 and 1's strategies mirror. We can halve stored info sets if storage pressure becomes real.

5. **MCCFR transition point:** At what score does the pre-built tree exceed available RAM enough to justify switching algorithms?

6. **Advanced tabular variants:** Do DCFR+ or PDCFR+ improve convergence enough
   over the current DCFR baseline to matter before we consider neural
   DeepPDCFR-style approximation?

7. **Compression and queryability:** With ~3.2 GB per score state and 144 states, total storage is still enormous. What compression scheme preserves query performance while keeping the saved strategies practical to inspect?

8. **Agent-backed CFR autoresearch:** The current runner asks provider APIs for a
   full replacement `cfr_experiment.rs`; the shipped loop should instead delegate
   each iteration to Claude Code. Local `claude` exposes
   `--max-budget-usd` for print-mode runs, which gives us a native per-iteration
   dollar cap. The ops launcher should collect max iterations, overall LLM budget,
   per-iteration LLM budget, model, effort, and exploration mode before starting a
   VM run. LLM budgets should mean LLM spend only; VM/GCP compute should remain a
   separate operational budget.

9. **Single-file proposal sandbox:** Claude Code permissions can allow broad
   reads while restricting `Edit`/`Write` to
   `crates/truco-solver/src/cfr_experiment.rs`. Bash must be denied for proposal
   generation because file permission rules do not constrain arbitrary shell
   subprocesses, and `.env*` reads should be denied so local secret references are
   invisible to the LLM. The runner should still validate the final diff and
   reject any attempt that touched files outside the experiment surface.

10. **Advisor pass:** Do not include advisor in the first shipped workflow.
   Interactive Claude Code has `/advisor`, and local Claude Code `2.1.128`
   contains a hidden `--advisor <model>` option, but `claude -p --advisor opus
   ...` currently rejects the flag. A separate `opus`/`xhigh` advisor call would
   be scriptable, but it adds budget accounting and orchestration complexity
   before we know it improves solver outcomes. Revisit advisor only if Claude
   exposes a stable automation surface or early non-advisor runs plateau.

11. **Guided vs free CFR exploration:** Backlog-guided mode should nudge Claude to
   try variations of the known solver ideas (for example regret pruning, lazy CFR,
   DCFR+, PDCFR+, sampling/parallelism variants) while still allowing combinations.
   Free exploration should keep the same harness and mutable-file limits but avoid
   anchoring the proposal to the current backlog. We should record the selected
   mode in `results.tsv` so later benchmark interpretation is honest.

12. **Experiment tracking:** MLflow is still the right fit, but the default
   target should now be a persistent personal-server MLflow deployment reachable
   only over Tailscale. CFR autoresearch VMs should log directly to
   a project-level tracking URI that the runner exports as
   `MLFLOW_TRACKING_URI`, so run history, artifacts, and dashboards survive VM
   teardown without copying a temporary tracking database over SSH. Keep a
   file-backed `autoresearch/mlruns` mode as a local/offline fallback. We should
   let the private operations layer open the dashboard. MLflow should log complete
   attempted-run metrics/artifacts, while git remains the accepted-candidate
   lineage and human-review layer.

13. **Environment configuration:** Non-secret CFR autoresearch defaults should
   live in committed `.env.example` files such as `autoresearch/.env.example`.
   `.env` and `.env.*` must stay ignored, except example files. Local env values
   may contain `op://...` references; the ops runner should resolve those via
   1Password in memory and inject only the resolved child-process environment,
   never resolved secret values in files, prompts, logs, or MLflow params.

14. **Operations interface:** Keep the project’s operations CLI- and
   agent-driven. The public repository owns portable solver/research commands;
   `baixada-ops` owns live VM work, deployment, and result retrieval. Agents can
   supervise longer-running research work. Do not recreate a separate web
   control plane without a deliberate product decision.

---

## 13. Cost Estimates

> **Superseded 2026-07-16.** The numbers below were Fermi estimates from
> before any real tree-size data existed, built on a since-abandoned
> single-monolithic-VM model. `SOLVER_BENCHMARKS.md`'s "2026-07-15/16 — Exact
> tree-size census" entry has grounded per-tier numbers instead: exact
> info-set counts for all 5 ladder tiers (measured directly, not sampled —
> tree size turns out to depend only on the ladder tier, not the exact score),
> real per-job wall times from the already-solved 10×10 fleet, and real GCP
> N2-custom pricing. Refined total for everything still remaining (Stage B's
> last 2 states + the {1,3,6}/{1,3,6,9}/{1,3,6,9,12} tiers): **≈$505K spot /
> $1.12M on-demand**, dominated by the {1,3,6,9,12} tier, which also likely
> needs a different machine family (M2/M3) or a disk-backed architecture
> change since its ~1.54 TB/job RAM requirement probably exceeds what an N2
> custom-extended VM can provision. Left in place below for history.
>
> **Update 2026-07-17:** the cost-optimization program (plan 79) then took that
> ~$505K down to **~$12–16K** (ε=0.01, warm starts, asymmetric raise pruning),
> still >10× the goal. See [EXACT_SOLVING.md](EXACT_SOLVING.md) §4 for the full
> bracket history; the exact line is paused in favor of the neural approach.

### Compute: solving the full game on GCP

**Machine:** n2-highmem-64 (64 vCPUs, 512 GB RAM)
- On-demand: ~$4.19/hour
- Spot (preemptible): ~$1.00-1.25/hour (60-70% discount)

**Estimated wall time:** 5-7 days ≈ 120-168 hours

| Scenario | Time | On-demand cost | Spot cost |
|----------|------|---------------|-----------|
| Optimistic (3 days) | 72 hrs | **$302** | **$72-90** |
| Expected (5 days) | 120 hrs | **$503** | **$120-150** |
| Pessimistic (7 days) | 168 hrs | **$703** | **$168-210** |

**Risk with spot:** preemptible VMs can be terminated with 30s notice. Need checkpointing to resume. Since each score+turnup solve is independent, losing one in-progress solve wastes at most ~3 hours of compute. Spot is strongly preferred given the savings.

**Recommendation:** Start with a shorter benchmark run (1-2 days, ~$20-50 spot) to validate projections before committing to a full solve.

### Storage and serving: solution web app

**Uncompressed solution size:** ~460 GB (all 144 score states × 9 turnup classes)

| Component | Cost/month | Notes |
|-----------|-----------|-------|
| VM (e2-small) | ~$12 | 2 vCPU, 2 GB RAM — sufficient for serving |
| SSD Persistent Disk (500 GB) | ~$85 | Uncompressed solutions, fast random access |
| **Total** | **~$100/month** | |

**Alternative: Cloud Storage + Cloud Run (cheaper for low traffic)**

| Component | Cost/month | Notes |
|-----------|-----------|-------|
| Cloud Storage (460 GB, standard) | ~$10 | Pay per GB stored |
| Cloud Run (minimal traffic) | ~$0-5 | Pay per request, free tier covers light use |
| **Total** | **~$10-15/month** | Requires lazy loading strategies from GCS |

**Recommendation for a fun project with few users:** Cloud Storage + Cloud Run at ~$10-15/month. Strategies are loaded on-demand from GCS when a user queries a specific score state. For the most-queried score states (0x0, 11x11), keep them warm in memory.

**Torrents for full download:** Compressed solutions (~100 GB with LZ4, ~60 GB with zstd) are feasible as a torrent. Users could run the solution viewer locally.

---

## 14. Neural Network Distillation (Future Phase)

> **Superseded 2026-07-17 by [plans/83-neural-policy-approach.md](plans/83-neural-policy-approach.md).**
> This section assumed "solve fully, then distill (pure SL, no RL)". We are NOT
> paying for a full tabular solution: the exact program landed at ~$12–16K for
> the whole grid, still >10× the goal (see [EXACT_SOLVING.md](EXACT_SOLVING.md)).
> The live plan is **SL on the partial exact corpus (10×10 + the mão row) as an
> upper bound and warm start, then deep RL for the unsolved deep bands.** The
> architecture/tooling notes below remain useful; the "we have perfect training
> data for the whole game" premise does not. Read plan 83 first.

### Concept (original — see caveat above)

Once we have the full tabular solution, we can train a small neural network to **approximate** the equilibrium strategy. This is pure supervised learning — no RL needed since we have perfect training data.

**Input:** Info set features (player, turnup class, hand composition, action history)
**Output:** Action probability distribution (softmax over legal actions)
**Loss:** Cross-entropy between NN output and tabular equilibrium strategy

### Why this is appealing

- **Smartphone-sized:** A small NN (100KB-1MB) could replace 460 GB of lookup tables
- **Generalization:** The NN may learn underlying strategic patterns rather than memorizing each info set, potentially playing well even in slightly modified game variants
- **Deployment:** Ship a single model file instead of a massive strategy database
- **Paper contribution:** "We solved Truco exactly, then distilled the solution into a neural network that fits on a phone"

### Architecture considerations

- **MLP:** Simple, fast inference. ~3-5 layers, 128-256 hidden units. Feature engineering matters: represent cards as strength vectors, history as sequence of action types.
- **Small transformer:** Could handle variable-length action histories more naturally. 2-3 layers, 64-128 dim.
- **Hybrid:** MLP for card features + small RNN/transformer for history.

### Training data

- ~11.2M info sets per turnup class × 9 classes × 144 score states ≈ billions of training examples
- But many are redundant (same hand at different histories). Can subsample.
- Train/val split by score state to test generalization across scores.

### Quality measurement

- **Strategy distance:** KL divergence between NN strategy and tabular strategy
- **Exploitability of NN:** Compute best-response EV against NN strategy. Should be close to tabular exploitability.
- **Head-to-head:** NN vs tabular in simulated matches. Tabular should win, but by how much?

### Tooling

Leverage Karpathy's `autoresearch` setup for rapid experimentation with architectures and hyperparameters. Export training data as a flat binary (info set features → strategy vector) for fast loading.

---

## 15. Concrete Milestones

### Milestone 1: GCP-Ready Benchmark And Pipeline Baseline

Get everything in shape to run meaningful benchmark comparisons on GCP:

1. ✅ Exploitability computation (best-response)
2. ✅ MCCFR implementation (external sampling)
3. ✅ CLI modes: `benchmark`, `compare`, `solve-tc`, `treesize`, `pipeline`
4. ✅ Initial 11x11 convergence data for CFR+, MCCFR, and DCFR
5. ✅ Demo viewer binary for inspecting a freshly solved in-memory strategy
6. 🔲 Benchmark 10x10 and 10x11 tree sizes (to validate that lower scores are feasible)
7. ✅ VM/ops workflow for detached pipeline runs, pushing values, and pulling results

**Deliverable:** A solver binary that can run on GCP and produce:
- Convergence curves (exploitability vs iterations) for both CFR+ and MCCFR
- Strategy comparisons between the two algorithms
- Solution files for a few score states that we can inspect

### Milestone 2: Solution Inspection

Before committing to a full solve, validate the solutions make strategic sense:

1. 🔲 Solution viewer: query equilibrium action at arbitrary game states from saved strategy files
2. ✅ Store enough info-set metadata in serialized strategies to support those queries cleanly
3. 🔲 Sanity checks using domain knowledge (e.g., "with 3 manilhas, the solver should raise aggressively")
4. 🔲 Compare CFR+, DCFR, DCFR+, and PDCFR+ solutions for similarity where they are solved deeply enough
5. 🔲 Document interesting strategic findings

### Milestone 3: Full Solve

1. 🔲 Choose algorithm based on Milestone 1 results
2. 🔲 Extend checkpointing beyond match values so interrupted long-running subgames are cheaper to resume
3. 🔲 Run full 144-state solve on GCP
4. 🔲 Validate all solutions (exploitability, zero-sum, symmetry)

### Milestone 4: Deployment

1. 🔲 Solution web app (query strategies online)
2. 🔲 Downloadable solution files (torrent/direct)
3. 🔲 Paper writeup

### Milestone 5: Neural Net

1. 🔲 Export training data from tabular solutions
2. 🔲 Train NN approximation using autoresearch
3. 🔲 Measure quality (exploitability, head-to-head vs tabular)
4. 🔲 Ship smartphone-ready model
