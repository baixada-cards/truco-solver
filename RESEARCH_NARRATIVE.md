# Solving Truco: A Research Narrative

*Work in progress. This document captures decisions, findings, and open
questions as the project evolves. Intended as raw material for a paper or
blog post.*

---

## What is Truco?

Truco is a two-player trick-taking card game played with a 40-card Spanish
deck. It is enormously popular in Brazil and several other Latin American
countries. The core loop is simple: players are dealt three cards each, one
card is turned face-up to determine the "manilha" (the top trump rank), and
they play three rounds of card-by-card tricks. The player who wins two of the
three rounds wins the hand.

What makes Truco strategically interesting is its betting structure: at any
point a player may "raise the stakes" (call *truco*, then *seis*, *nove*,
*doze*), escalating the hand's point value from 1 up to 12 match points. The
opponent may fold, accept, or re-raise. A match ends when a player reaches 12
points.

The combination of private information (hidden cards), a stake-raising ladder,
a trump that changes each hand, and the ability to hide cards in later rounds
creates a rich bluffing and deception game that is far harder to analyze than
it might appear.

---

## Why solve it?

Truco is a solved game in principle — it is finite, two-player, zero-sum, and
has perfect recall. But it has never been solved in practice, at least not
publicly. The game tree is large enough that naive enumeration is infeasible,
and the card abstraction required to make it tractable is non-trivial.

The motivation is threefold:

1. **Scientific:** Characterize the Nash equilibrium of Truco. What is the
   optimal bluffing frequency? When should you hide your best card? How much
   does having the highest trump (the *zap*) actually matter?

2. **Product:** Build a strong bot opponent and a strategy explorer for a
   Truco web app. Players can inspect what the solver recommends at any
   decision point.

3. **Methods:** Apply and compare modern CFR variants (CFR+, DCFR) to a game
   of this scale, and experiment with LLM-driven autonomous algorithm
   improvement — a methodology inspired by Karpathy's autoresearch framework.

---

## Game tree structure

The game has 78 unique score states (taking advantage of the symmetry that
score (a, b) from player 0's perspective is equivalent to (b, a) from player
1's perspective), solved in descending order of total points since later states
are needed as continuation values. Each score state has 9 strategically
distinct "turnup classes" (the turnup card determines the manilha rank and
removes one card of its own rank from the deck).

At score 11×11 (the simplest subgame — the *mão de onze* scenario where both
players are at match point and raises are disabled):

| Metric | Value |
|--------|-------|
| Abstract deals per turnup class | ~140,000 |
| Game tree nodes (all deals × 2 dealers) | 99,736,516 |
| Unique information sets | 11,223,052 |
| Tree build time | ~178s on GCP c2-standard-8 |
| Trees + strategy table memory | ~5.8 GB |

Lower score states are larger due to the stake-raising branching (raise
decisions add up to 4 extra actions per player per hand, exponentially
expanding the tree).

---

## Card abstraction

The key insight enabling tractability is that suit is irrelevant in Truco
except for manilha rank ordering. Specifically:

- There are 4 manilhas (one per suit), ordered by suit strength
- There are 9 plain rank levels (non-manilha), unordered by suit
- The turnup card determines which rank becomes manilha, and removes one
  card of its own rank from the deck (3 remaining instead of 4)

So we abstract each card to either `Manilha(suit_rank: 0-3)` or
`Plain(strength: 0-8)`. An information set is identified by:

- The player's 3-card abstract hand
- The turnup class (which rank is the manilha + which plain level is the
  "blocked" one with only 3 copies)
- The action history (bets + visible played cards)

This reduces the game tree from the full combinatorial explosion of 40×39×38
deals to ~140,000 abstract deals per turnup class — a reduction of roughly 40×.

---

## Algorithm choices

We use **Counterfactual Regret Minimization (CFR)**, the standard algorithm
for computing Nash equilibria in imperfect information games. The key design
decisions:

### Pre-built game trees

Rather than re-traversing the game tree each iteration using the full engine,
we build a compact arena representation of all game trees once (220s one-time
cost) and iterate over the in-memory arena. This drops per-iteration cost from
being dominated by engine overhead to pure floating-point arithmetic over
pre-allocated arrays.

### CFR+ vs DCFR

We benchmarked two variants at 11×11, TC 0, for 100+ iterations each:

**CFR+** (Brown et al., 2015): clamps negative regrets to zero. Simple and
robust.

**DCFR** (Brown & Sandholm, 2019): discounts old regrets and cumulative
strategy by `t^α / (t^α + 1)` and `(t/(t+1))^γ` respectively. Gives more
weight to recent iterations.

Results so far at `11x11`, `TC 0`:

| Iter | CFR+ expl | DCFR expl | Notes |
|------|-----------|-----------|-------|
| 10 | 0.279 | 0.186 | DCFR clearly ahead early |
| 20 | 0.120 | 0.076 | advantage still substantial |
| 50 | 0.062 | 0.052 | gap narrows |
| 100 | 0.051 | 0.048 | tail convergence is slow for both |
| 120 | n/a | 0.047 | best current measured DCFR point |

DCFR converges faster early, but the bigger strategic lesson is that both
algorithms slow dramatically in the tail. The project is no longer blocked on
"does CFR work for this game at all?"; it is blocked on how expensive the
lower-score states become once the full raise ladder returns.

### Exploitability as convergence metric

Exploitability ε measures how much a best-responding opponent can gain against
the average strategy. At Nash equilibrium, ε = 0. For a game with values in
[-1, +1], ε = 0.01 means the best-responder achieves a win rate of at most
50.5% — essentially unbeatable in practice for humans.

We compute exploitability via full best-response traversal of the pre-built
trees, which costs ~15s (one extra iteration-equivalent per measurement). We
compute it every 10 iterations during production runs.

### MCCFR: ruled out for this problem

Monte Carlo CFR (external sampling) samples deals rather than enumerating all
~140,000. It is attractive when trees don't fit in memory. However, at equal
wall time (3300s), MCCFR achieved exploitability 0.126 vs CFR+'s 0.051 —
2.5× worse. Since 11×11 fits in memory (12 GB), pre-built trees plus the
full-tree CFR family (CFR+/DCFR) is the right default choice.

---

## Autoresearch: LLM-driven algorithm optimization

Inspired by Karpathy's autoresearch project, we built a framework for
autonomous algorithm improvement:

- A **fixed harness** handles tree building, exploitability evaluation, git
  versioning, and result logging.
- A **single mutable Rust file** (`cfr_experiment.rs`) contains the iteration
  logic: regret updates, discounting, strategy computation.
- An **LLM** (selected from the configured provider, currently Anthropic or
  OpenAI) proposes modifications, reviews the git history of past experiments,
  and iterates.
- In the current direct-provider harness, each experiment runs for exactly 10
  minutes of iteration time (excluding tree build) and is evaluated by final
  exploitability.

The LLM sees: the full instruction document (`program.md`), the current code,
all past results (`results.tsv`), and git diffs of previous experiments.

The next operational step is to make this explicitly **CFR autoresearch** rather
than generic repo autoresearch, and to move the active loop from direct provider
API calls to Claude Code. Claude Code is the intended first shipped backend
because it supports a native per-iteration dollar cap with
`claude -p --max-budget-usd <amount>`. The ops launcher should also capture max
iterations, overall LLM budget, model, effort level, and whether the run is
backlog-guided or free exploration. The first Claude Code runner should skip
advisor because the automation surface is fragile today, restrict edits to
`cfr_experiment.rs`, deny Bash and `.env*` reads during proposal generation, and
let the harness own benchmarks and source promotion.

This is an experiment in applying autonomous research to algorithm engineering
rather than neural network training. The space of CFR variants is vast and
not fully explored in the literature; an AI agent iterating overnight might
find combinations or schedules that human researchers haven't tried.

---

## Results so far

*(Updated as experiments complete.)*

### Benchmark: 11×11, TC 0

| Date | Algorithm | Iters | Exploitability | Wall time |
|------|-----------|-------|---------------|-----------|
| 2026-03-24 | CFR+ (every-iter expl) | 100 | 0.051 | 3300s |
| 2026-03-24 | MCCFR | 8.35M | 0.126 | 3305s |
| 2026-03-25 | DCFR (α=1.5, β=0, γ=2) | 120 | 0.047 | 2396s |

### Autoresearch experiments

The autoresearch harness uses a stricter metric than the long manual benchmark
runs above: a fixed 10-minute iteration budget, excluding tree build and final
exploitability computation. Those numbers are useful for comparing experiment
variants to each other, but they should not be compared directly to the
40-55-minute manual runs above.

| Date | Commit | Description | Iters | Exploitability | Iter wall time |
|------|--------|-------------|-------|---------------|----------------|
| 2026-04-07 | `4b1300b` | baseline | 25 | 0.663562 | 861.7s |
| 2026-04-07 | `38fd48c` | DCFR+ hybrid with regret clamping and discounting | 70 | 0.157519 | 866.9s |

That is a large improvement over the harness baseline. The remaining question
is whether autoresearch can close the gap to the stronger hand-driven DCFR
results, or whether the current search space and 10-minute budget are still too
narrow.

### Full 11×11 subgame — all 9 turn-up classes (2026-06-28)

The first solve of the *entire* 11×11 subgame, not just TC 0. Three VMs in
parallel, CFR+, ~3300s budget per turn-up class, using the new full-state
checkpoint/resume path (`solve solve-tc --time-budget --checkpoint --resume`).

| TC | Final exploitability | Iters | Game value (P0) |
|----|---------------------|-------|-----------------|
| 0  | 0.0533 | 81  | 0.000078 |
| 1  | 0.0525 | 83  | 0.000175 |
| 2  | 0.0514 | 89  | 0.000095 |
| 3  | 0.0486 | 107 | 0.000115 |
| 4  | 0.0503 | 90  | 0.000092 |
| 5  | 0.0512 | 90  | 0.000123 |
| 6  | 0.0504 | 100 | 0.000072 |
| 7  | 0.0520 | 90  | 0.000135 |
| 8  | 0.0511 | 84  | 0.000183 |

Two things worth recording. First, the **game value is ≈ 0 for every turn-up
class** (≤ 0.0002 in the ±1-normalized space), i.e. match_value(11,11) ≈ 0.5.
That is forced by symmetry — 11×11 is a symmetric position — so it is a clean
end-to-end correctness check on the whole pipeline (tree build, traversal,
terminal-value mapping, averaging). Second, the **nine final exploitabilities
land in a tight 0.005-wide band** despite being fully independent solves on
different VMs; the turn-up class barely moves the difficulty of the subgame.

The convergence wall (below) shows up here too: CFR+ flattens around ~0.05 after
~85 iterations. Pushing lower is now a *resume*, not a restart — each TC's
1.7 GB checkpoint (regrets + strategy + iteration) is parked on its stopped VM's
disk.

---

## Descending the lattice: the dealer, and the mão-de-onze wall (2026-06-29/30)

With 11×11 solved, the next states down are 11×10 and 10×10. This is where the
project stopped being "run CFR on a bigger tree" and turned into a sequence of
conceptual corrections. Documenting the whole thing — including the dead ends —
because the dead ends were where the real understanding came from.

### The dependency chain, and a shortcut that wasn't

Score states must be solved top-down: a state's terminal payoffs look up the
*match value* of the states it transitions into. 11×11 is special — every hand
ends the match, so it needs nothing else and is exact. 11×10's only non-terminal
transition is the at-11 player folding the mão de onze, which concedes one point
and goes to 11×11. 10×10 transitions to 11×10 / 10×11. So the order is 11×11 →
11×10 → 10×10, and by player-swap symmetry we only need to *solve* 11×10 (10×11's
value is 1 − mv(11,10)).

We first tried brute CFR on 11×10 (9 TCs, 3 VMs, eps 0.05). It **walled at ~0.33
exploitability** — the same slow, worse-than-1/T flattening we'd feared. DCFR
behaved *identically* (0.374 vs 0.367 at iter 45), so the wall is **structural,
not algorithmic**: no CFR variant was going to fix it. We verified it is *not* a
bug — the engine models 11×10 correctly (accept → hand worth 3, a final hand
where winning = winning the match; fold → concede 1 → 11×11; no raising).

### The dealer bug (pé advantage is real)

While reasoning about the fold value we found a genuine modelling error. The
match-value table was indexed by **score only**, and the solver **averaged over
the two dealer assignments**. But the engine's dealer strictly *alternates*
(`next_dealer = other_player`), and the dealer (pé) plays last each trick — a real
positional edge. We measured it from the existing 11×11 solve: **the dealer wins
55.6% vs 44.4%** at mão de onze (+5.6pp). Averaging that away is wrong. So we made
the solver **dealer-exact**: the match table gained a dealer dimension, and every
continuation now looks up `mv(new_score, 1 − current_dealer)` to honour
alternation. 11×11 stays valid (no continuations); the fix matters for every
lower state. The user's framing: "11×10 is two different scores — is the 11-player
the dealer or not." Exactly.

### Why the wall: screening, and a shared info set

The 11×10 difficulty is the **accept/fold screening game**: the at-11 player's
accept range and the opponent's response chase each other, and CFR converges that
coupling very slowly. Worse, we discovered the accept/fold decision *can't even be
learned per-dealer*: it happens at the empty history (before any card), and the
info set doesn't encode position, so it's the **same key in both dealer trees** —
plain CFR is forced to average one accept policy over both positions, washing out
the pé advantage the user cares about.

### The decomposition (freeze the accept, iterate on equity)

The way around both problems: **freeze** the at-11 player's accept decision
externally, **per dealer**, and let CFR solve only the card play (which converges
like 11×11). Then iterate: solve card play → read each hand's equity → set
`accept iff equity > F[dealer]` → repeat. The fold constant `F` is the
dealer-specific 11×11 value (0.444 or 0.556, *not* 0.5). A clean correctness
check fell out and passed exactly: **accept-all 11×10 is literally the 11×11 game**
(no continuations consulted), so round 0's per-dealer value prints 0.556/0.444 —
the 11×11 numbers, to 4 decimals.

### Dead ends along the way (all instructive)

- **A determinism bug** surfaced first: `InfoSet::key()` used a per-process
  *random* hash seed, so a checkpoint written by one solve hashed to different
  keys in another — `--resume` and cross-state warm-start silently transferred
  *zero* entries. (In-process solves were always fine, so nothing prior was
  corrupted.) Fixed with a fixed-seed hash + rekey-on-load.
- **Warm-starting** the card play from 11×11 did **not** break the wall — the
  transferred play is tuned for the *full* range, but at 11×10 the opponent faces
  the *accept-filtered* range and must re-adapt (the screening is untouched). It
  even tracked slightly worse. Lesson: don't let CFR learn the accept; freeze it.
- **"Average inertia" hypothesis, refuted.** The frozen-accept solve plateaued at
  ~0.26 (better than brute 0.33, not the ~0.05 we wanted). We guessed CFR+'s
  `t^γ` averaging was carrying stale play across accept-set changes, and added a
  "warm regrets, fresh average each round" mode. It made **no difference** (0.270
  vs 0.257) — so the residual was real, not an averaging artifact.
- **The actual bug: a folded-hand equity blind spot.** A folded hand has *zero
  reach*, so its card-play info sets are never trained (they stay uniform). Its
  "value if accepted" was therefore measured as if played *randomly* → it looks
  weak → it stays folded forever — while a best-responder happily *accepts* it and
  plays it well. That mislabelled tail *is* the ~0.26. The fix (under test): value
  each hand by the **decider's best response** (optimal play vs the opponent's
  current strategy), so `accept iff value > F` is consistent with what the
  best-response would do, and can't be exploited.

The best-response-equity fix did **not** crack it either — and the reason is the
most useful thing we learned. Reading the exploitability as a per-player split
(`BR_p0` for the decider, `BR_p1` for the opponent; each is a sum over the two
dealer trees) localizes the entire residual:

- **Opponent's best-response gain ≈ 0.025** — the at-10 player is already playing
  essentially optimally. **The card play is solved.**
- **Decider's best-response gain ≈ 0.51** — the whole ~0.27 exploitability is the
  at-11 player's **accept decision**.
- And the accept set does **not converge**: it grew 403 → 301 → 349 round over
  round. Best-responding the accept against a *fixed* opponent overshoots (accept
  more → but a real opponent would then defend), so it oscillates; the
  average-equity variant instead froze at a non-equilibrium ~0.26.

This is textbook **fictitious-play non-convergence on a screening game**: freezing
one side of a coupled decision and best-responding the other cannot reach the
joint equilibrium. So the elegant decomposition *does* buy us the hard part (the
card play) but *cannot* by itself close the accept coupling.

**The indicated fix — let CFR solve the accept, don't freeze it.** CFR's whole
job is coupled equilibria: its regret-matching + strategy averaging is precisely
what makes screening games converge. The reason we froze the accept in the first
place was that the accept info set is *dealer-shared* (it sits at the empty
history, and the info set doesn't encode position), so plain CFR could only learn
one position-averaged accept policy — which is *itself* likely a big part of why
brute CFR walled at 0.33. So the clean next step is a one-field abstraction change:
**add position (dealer) to the info set** so the accept node is distinct per dealer
tree, then run plain dealer-exact CFR on 11×10 and see whether a per-position,
jointly-learned accept finally beats 0.33. The card play being already solved is
the encouraging part. (A fallback if it still walls: fictitious-play *averaging* of
the accept sets across rounds, which converges where best-response oscillates.)

### Position in the info set: the wall falls (2026-07-02)

A fresh look sharpened the framing before implementing. The dealer-0 and
dealer-1 games at 11×10 are **completely independent games** — a strategy in one
never constrains the other; they touch only through the already-known
match-value table. So this was never "CFR struggles with screening" (poker's
fold-or-play root decision *is* a screening game and CFR eats it for breakfast).
It was the implementation accidentally splicing two different games together via
shared info-set keys. And the splice was wider than the accept node: since
`ActionHistory` records *what* was played but not *who* acted, card-play info
sets also alias across dealer trees — `[PlayFaceUp(X), PlayFaceUp(Y)]` at
player 0's turn means "opponent led X, I answered Y" in the dealer-0 tree but
"I led X, opponent answered Y, and I won the trick" in the dealer-1 tree. Same
key, different states. (That aliasing exists at 11×11 too, and is now a suspect
for part of its ~0.05 floor — see open question 1c.)

`InfoSet` gained an `is_dealer` field, which cleanly kills both problems: the
two dealer games become disjoint in key space. Corollaries that fell out:

- **Per-dealer solves are exact.** With disjoint keys there is zero benefit to
  solving both dealer trees in one process, so `solve-tc --dealer {0|1}` builds
  only one tree per run — half the memory (~50M nodes instead of ~100M), two
  jobs that parallelize across machines *and any zone*, which matters during the
  us-central1-a capacity stockout. A filtered solve reproduces the joint solve's
  per-dealer value to <1e-12 (it literally runs the same update sequence).
- **Legacy artifacts survive.** Old checkpoints/strategies (pre-position layout,
  bincode-positional) load via a fallback that expands each entry into both
  position variants — the right semantics, since a legacy shared entry held the
  position-averaged accumulators. The stranded 11×11 checkpoints remain usable
  for resume/warm-start.
- **A test-harness bug got flushed out.** The first tiny-config validation
  produced *bit-identical* results before and after the fix — suspicious enough
  to chase, and the cause was that `solve_with_limit`'s deal cap took a *prefix*
  of the enumeration, which iterates player 0's hand outermost: the "tiny game"
  gave the decider a single hand, so there was nothing to screen. Deal capping
  is now strided across the enumeration. Lesson: a validation config must be
  checked to actually *contain* the phenomenon being validated.

With the strided tiny config (300 deals, TC 0, dealer-exact fold values), the
A/B is unambiguous: the pre-fix solver walls at exploitability **0.0965** (flat
from iteration ~50 through 300 — the same shape as the full-scale 0.33 wall),
while the fixed solver passes 0.0102 by iteration 51 and reaches **0.0003** by
300, clean ~1/T all the way down. The accept is learned per position, jointly
with the card play, no freezing, no oscillation. The full-scale 11×10 run
(designed in `SOLVER_BENCHMARKS.md`, 2026-07-02) is now an ops task, not a
research question: seed `mv(11,11,·)` via `set-mv`, run `--dealer 0` and
`--dealer 1` jobs per TC, aggregate the `.gv` sidecars into `mv(11,10,·)`.

### The wall was the ruler: clairvoyant best response (2026-07-02, later the same day)

The position fix was necessary but the full-scale run **still crawled**: both
dealer games flattened toward ~0.15/game while the tiny config dove to 0.0003.
Chasing that gap turned into the most instructive sequence of falsifications in
the project so far:

1. **Reproduce at mid density.** With strided deal caps, the "wall" height
   scales with deal density: 300 deals → ~0; 1,000 → 0.16; 3,000 → 0.26; full
   → ~0.30. So the tiny config had been too easy in a specific way — with ~1
   deal per info set there is no pooling, hence (it turned out) no measurement
   gap either.
2. **Falsify the algorithm hypotheses.** DCFR ≡ CFR+ on the mid config (again).
   A floor-vs-slow discriminator: 10× the iterations (300 → 3,000) moved
   exploitability 0.1596 → 0.1533 — a hard floor, slower than any legitimate
   CFR regime (even worst-case 1/√T predicts 0.05). A floor contradicts CFR's
   convergence guarantee on a finite zero-sum game, so either the dynamics
   deviated from theory or the measurement did. Two dynamics suspects were
   implemented and falsified in one afternoon: **synchronous sweeps**
   (strategy frozen per iteration — the classic analysis assumes this; ours
   recomputed from live regrets mid-sweep) and **predictive regret matching**
   (PCFR+, optimism against cyclic screening dynamics). Both hit the *same*
   floor to three decimals. Algorithm-independence meant: look at the ruler.
3. **The ruler was broken.** `br_traverse_tree` maximized independently inside
   every deal's tree — `Σ_deals max` instead of the legal `max_per_info_set
   Σ_deals`. That best response conditions on the opponent's hidden hand: it is
   **clairvoyant**, a strict upper bound whose gap (the value of card-reading)
   does not vanish at equilibrium and grows exactly with info-set pooling — the
   density scaling, the algorithm-independence, the tiny-config immunity, and
   the "residual is all in the decider's accept" split (a clairvoyant decider
   accepts precisely when his hand beats *this* opponent hand) all follow.
4. **Confirmation.** The identical 3,000-deal solve measuring 0.2569 under the
   clairvoyant measure is at **0.0118 exact exploitability** — near-equilibrium
   all along. Under the exact measure every config converges cleanly and
   monotonically (1,000 deals: 0.0093 @ iter 290).

Consequences, in order of importance:

- **Every historical exploitability number in this project is a clairvoyant
  upper bound**, including the 11×11 "wall at ~0.05" — which is now best read
  as "the clairvoyance gap at 11×11 is ≈0.05," with the true exploitability of
  the solved 11×11 strategies unknown but likely far lower (measurable via the
  new `eval-ckpt` mode). The "tail slowdown after ~50 iterations" was the
  approach to the clairvoyance gap, not near-equilibrium fine-tuning.
- **Game values were never affected** — `eq_traverse_tree` (avg vs avg) had no
  such defect, so mv(11,11,·) = 0.5564/0.4437 and the pé-advantage results
  stand as-is.
- **The position fix stands** — it was validated at low density where the two
  measures agree (0.0965 wall vs 0.0003, both effectively exact), and the
  aliasing census is measurement-independent. Two real defects stacked: a
  genuine abstraction bug (shared accept node) under a broken measure that
  would have shown a wall even after the abstraction was fixed.
- The exact best response is now the default (`best_response_value`:
  counterfactual weights top-down, per-info-set argmax resolved deepest-first,
  profile evaluated top-down). The clairvoyant measure survives as
  `best_response_value_clairvoyant` for diagnostics — the gap between the two
  is itself an interesting statistic (how much is card-reading worth against
  this strategy?).

Meta-lesson for the narrative: we spent three "walls" (0.33 brute, 0.26
freeze-accept, 0.15 post-position) attributing to the *game* or the *algorithm*
what belonged to the *measurement*. The tell was present from the start —
exploitability plateaus that were algorithm-independent and density-dependent —
but it only became legible once the tiny config's false all-clear was traced to
missing pooling. When a metric floors, calibrate the metric against a case with
a known answer at the same *structure* (pooling), not just the same rules.

### The cost collapse (2026-07-03): from $115k to pocket change

10×10 — the first raise-enabled state — measured at 352M nodes / ~60 GB /
~17h per subgame, and a Fermi over the remaining lattice landed at **~$115k**
(±3×), dominated by the 51 deep states. Unacceptable, so the day pivoted to
lossless optimization, and four multiplicative levers landed within hours of
each other:

1. **Synchronous sweeps (~10×).** The solver had always recomputed strategies
   from live regrets mid-sweep (asynchronous Gauss-Seidel). The textbook CFR+
   iteration — strategy frozen for the whole sweep, buffered regrets folded at
   the end — was sitting unimplemented until the clairvoyant-BR investigation
   incidentally built the buffering machinery. Under the exact measure it
   converges in ~10× fewer iterations (tail ≈T^-1.9 vs ≈T^-0.6), replicated
   at three scales including real 10×10 (0.0095 @ iteration 90). The async
   drift had been quietly poisoning the averaged strategy all along.
2. **Dense accumulators (4.9× time).** Nodes carry a dense `table_idx`; the
   hot loop indexes a flat vector instead of hashing a 64-bit key per visit.
3. **Packed trees (~4.5× RAM).** 12-byte nodes + one shared edge array replace
   a 48-byte enum with a heap Vec per node. RAM is what prices the machine,
   and cost = RAM × time.
4. **Parallel sync sweeps (÷N RAM-hours).** Sync's frozen strategy makes the
   sweep read-only on accumulators, so deals fan across threads with atomic
   accumulation — one job now uses all of a machine's cores instead of
   holding 60 GB hostage for a single-threaded day.

Compounded, the same full-game program prices at **~$500–2,000** (bench-VM
measurement pending), and per-subgame costs collapsed from ~$12 (11×10-class,
async, single-thread) to the first measured $0.18 (11×9 subgame, one worker,
still async). Meta-lesson, twice in two days: the expensive thing was never
the game — it was unexamined defaults (a clairvoyant ruler, an asynchronous
sweep, a pointer-heavy tree) that each looked like "the obvious
implementation" until the bill made them visible.

### The pivot: exact solver as teacher, not product (2026-07-04)

The optimization campaign ended with honest numbers: ~10× from synchronous
sweeps (real, validated at full scale), memory work that helped less than
hoped (57.7 → 40.6 GB measured), threading a measured dead end, exact
zero-prob pruning (1.75×+), and band-shared mmap treepacks (trees identical
within a ladder band — one artifact per band×TC×dealer, mapped rather than
loaded, which is also the larger-than-RAM streaming path). Even so, the deep
lattice priced at ~$5-12k — and the 8×8 measurement (1.11B nodes, 129M info
sets, ~120 GB) made clear the full-ladder band multiplies from there.

So the project pivots to what SOLVER_PLAN §14 always listed as the endgame,
just earlier: **solve a diverse sample of subgames exactly to very low ε
(~$50-80 at post-optimization prices), distill into a small neural net,
extend to the full game by self-play, and measure the net's exploitability
with the exact-BR machinery.** Key design insight (user's): make score and
current stake input features with a masked full action space — the net then
enters unsolved deep states playing its learned shallow-band strategy (at 0×0
it simply doesn't re-raise past 6 yet), and self-play only has to learn when
the 9/12 rungs improve on a policy that is already competent. Full plan and
budget: `plans/71-neural-distillation-pipeline.md`.

### A separate, clean win: raise-ladder pruning

Orthogonal but important for 10×10 and below: once the stake on the table already
makes the hand match-deciding for *both* players (`min(score) + stake ≥ 12`),
raising further is dominated — no extra prize, and folding the higher raise also
concedes the match. Pruning those raises is *value-exact* and collapses the 10×10
betting ladder from {1,3,6,9,12} to just **{1,3}** (only truco matters), which is
what makes 10×10 tractable at all. The user spotted this ("raising to 6+ does
nothing when 3 already ends the game"); we implemented it solver-side, leaving the
authoritative engine untouched.

### Hand-point EV alongside match-equity EV in the Study charts (2026-07-10)

The Study lab (plan 73) shipped `q` — one-shot-deviation EV in ±1 match-equity
space — so a chart can show "how much match-win probability does this action
cost." But match equity is not how a player at the table reasons about a
single hand: they think in points ("that truco call cost us 3"), and match
equity's concavity near the target means a 3-point swing at 0×0 and the same
3-point swing at 11×9 read very differently in ±1 space even though the raw
stake is identical. The user wanted a second, point-denominated number
alongside `q` so mistake costs are legible both ways.

The fix turned out to be nearly free: `teacher_export::q_traverse` already
threads a single terminal value (`terminal_p0_value`, the match-equity lookup
through `MatchValueTable`) up the same σ̄-vs-σ̄ tree walk used for `q`. The
node directly *underneath* that match-equity conversion — `payoff_p0`, the
engine's raw signed hand-point differential (already exactly ±1/±3/±6/±9/±12
via `state.hand_value()`, mão-de-onze and fold terminals included) — is
already the "hand points" quantity we wanted. So `q_traverse` now returns a
`(match_value, pts_value)` pair and accumulates two numerators instead of
one; `pts` costs one more f32 array in `.teach` (bumped to format v2) and one
more JSON field per action, no new traversal, no new BR machinery.

Certification did NOT carry over for free, and on reflection shouldn't: the
exact-BR certificate measures exploitability in match-equity space, which is
the actual objective either player is optimizing. "Best response to raw hand
points" isn't a coherent equilibrium concept — a player who deviates to
maximize this hand's points while ignoring match equity isn't playing a
different real strategy, they're optimizing the wrong thing — so `pts` ships
uncertified (`certificate.pts_certified: false`) rather than inventing a
parallel BR pass to certify a quantity nobody is actually playing to.

Backfilling all 15 shipped spots (11×11/11×10 d0+d1, 11×9…11×0 d0, the
provisional 10×10 d0) was a re-export from the same checkpoints, not a
re-solve — confirmed by `raw_eps`/`purified_eps`/`actions_zeroed` matching
the previously-shipped charts bit-for-bit at every spot. Ran on a temporary
GCP VM (`n2-custom-2-76800-ext`, 75 GB RAM) rather than locally: the mão-band
spots need only a few GB, but 10×10's 352M-node/39.5M-info-set tree is the
same one flagged elsewhere in this doc as too large for a laptop, and its
export hit the *existing* 3%-mass Q-gap-residue assertion (3.9230% — matching
`raw_max_info_set_mass_above_assert_qgap: 0.0392` already recorded in the
shipped "provisional" chart), so it needed the same `--allow-residue`
downgrade the original provisional export must have used. Sanity checks: at
11×11 (stake fixed at 3, no continuations) `pts` reproduces `3·q` to within
export rounding (max deviation 0.0002 across 64,185 actions); at 10×10,
accept-raise actions (locking in the higher stake) show visibly larger mean
`|pts|` (1.57) than pre-raise play actions (0.78).

### Warm-started ε-tremble refinement of off-equilibrium spots (2026-07-11)

Open question 0 above (garbage mixes at rarely-reached info sets) and Q3 in
`QUESTIONS.md` both trace to the same cause: certified exploitability is
reach-weighted, so it is blind to whatever CFR left at a node it barely
visited. A hand that essentially never leads the way this one did gets
almost zero counterfactual weight every iteration, so its regret sits near
zero forever and regret matching returns something close to uniform —
which is exactly the "hides a winner 21% of the time" bug. The three routes
weighed in Q3 were: (1) perturb the solver so every node gets real visits,
(2) purify off-path nodes toward their (possibly-also-garbage) q-argmax
after the fact, (3) resolve subgames on demand. Route (1) went first because
the existing full-state `--resume` (Phase 5c) makes it a bounded top-up on
already-converged checkpoints, not a re-solve.

**The construction.** `cfr::TrembleSchedule` floors the strategy actually
used for reach propagation, regret, AND average-strategy accumulation:
`σ'(a) = ε/|A| + (1-ε)·σ(a)`. This is CFR run on a PERTURBED extensive-form
game — every info set forced into the interior of the simplex, one
totally-mixed game per Selten/van Damme's trembling-hand construction — not
the base game. Two consequences worth being explicit about: every info set
gets real counterfactual visits every iteration (own-reach floors at
`(ε/|A|)^depth`, so descendants of a rarely-taken branch finally accumulate
regret-matched, non-uniform strategies instead of noise); and the exported
average is an ε-equilibrium of the PERTURBED game, not an exact equilibrium
of the base game — a small, deliberate, and reported departure from the
"exact" solves everywhere else in this doc. Exact-BR certification is
untouched — it always measures the true best response to whatever ends up
in the accumulators, so the same machinery honestly reports how much the
perturbation cost.

The one mechanical surprise: the traversal's zero-prob branch-pruning fast
path (skip a subtree when a non-owner's strategy assigns it exactly 0,
which is most of the tree once RM+ has converged) is a no-op while
trembling is active, since no action is ever exactly 0 anymore. Sweeps
revert to full-width tree walks — iteration cost close to an early/
unconverged sweep, not a late/pruned one. This is why the guardrail
mattered: a short "burn 5-10 minutes and read the real cost" canary on the
actual checkpoints (not a synthetic benchmark) before committing to the
iteration budget, per the task's own instruction to "confirm the
per-iteration costs against SOLVER_BENCHMARKS before launching."

**Results.** All 5 shipped tc0 spots (11×11 d0/d1, 11×10 d0/d1, 10×10 d0)
were warm-started with `--extra-iters` (a small `solve-tc` addition: resolve
`max_iters = checkpoint_iteration + N` once the checkpoint is loaded, so a
resume-relative budget doesn't need a throwaway discovery pass) and
`--tremble-eps 0.05 --tremble-eps-end 0.01`. Measured cost: ~5s/iter at
11×11/11×10 (5.6M info sets, no or a shallow raise ladder) vs ~85s/iter at
10×10 (39.5M info sets, {1,3} ladder, pruning fully defeated) — roughly the
17× tree-size ratio, confirming the "no pruning benefit" mechanism directly
rather than by proxy. +200 iterations sufficed for the 11-column spots
(~1100s wall each, run in parallel on one VM); 10×10 got +100 in ~2.36h
(the tree-size/cost tradeoff meant matching iteration counts wasn't
worth the wall-clock — the self-loss numbers below show diminishing but
still real returns at half the dose). Total cost was a small fraction of
the ~10-25 USD guardrail (two short-lived VMs, on-demand, under 3 hours of
wall time combined). See `SOLVER_BENCHMARKS.md` 2026-07-11 for the full
per-spot table (iteration counts, wall times, before/after certificate
eps, and self-loss/own-reach flagged shares).

**The one nuance worth remembering:** own-reach measured on the RAW
(pre-purification) average strategy collapsed exactly to the predicted
floor (0% flagged at 11×11, versus 51% before) — direct confirmation the
mechanism does what it says. Measured on the PURIFIED `p` (what the study
lab actually ships and reads), own-reach only partially improved (roughly
halved, not collapsed). This is not a shortfall in the fix: purification
correctly re-zeros actions that really are dominated, and a descendant of a
correctly-pruned branch has genuinely near-zero reach in the real
equilibrium — no amount of training changes that fact, nor should it.
What tremble fixes is the STRATEGY you'd find if you (or a human opponent
playing "wrong") ever did end up there, which is exactly what self-loss
measures, and self-loss collapsed 8-16× at the cheap spots. Own-reach and
self-loss are answering different questions ("how often does this matter"
vs "is the answer here any good"), and it turns out only the second one was
ever going to move by construction.

---

## Open questions

0. **Off-path infosets export garbage mixes (found 2026-07-10, study lab).**
   At `11x10 v4 : a 4 4` (mão at 11 led the 4 into a tie), the exported mix
   for mão holding 5♥5♠(+played 4) hides each manilha with p≈0.21 at
   q=−1.0 — a certain match loss — versus q=+0.815 for playing face-up. The
   infoset's reach weight is ~6.4e-06: the equilibrium almost never leads the
   4 from that hand, CFR barely visits the node, and regret matching leaves
   near-arbitrary mass there. Reach-weighted exact-BR certification is blind
   to it by construction, so certificates pass while off-path strategies are
   nonsense. This matters for the study lab, whose whole point is walking
   arbitrary lines. UI now warns when a holding's joint reach is <0.01% of
   deals ("weakly trained — trust win %/pts over frequencies"). Candidate
   solver-side fix for a future re-export: purify off-path infosets by
   q-argmax (or sweep purification with a reach-conditional threshold so
   dominated actions at negligible-reach infosets are zeroed regardless of
   their raw probability).

   **Fixed (2026-07-11), see the dated entry below.** Warm-started ε-tremble
   refinement re-trained exactly this info set: the two manilha HIDE
   actions' probability dropped from p≈0.21 each (q=−1.0, certain loss) to
   raw_p≈0.86% each (purified away to p=0.0 outright), while the two PLAY
   actions rose from p≈0.29 to p=0.50 each (98.3% combined raw mass). The
   reach weight itself barely moved (6.42e-6 → 4.94e-6, as expected —
   trembling retrains the STRATEGY at a node, it does not manufacture
   equilibrium reach that isn't there).


1. **Convergence wall:** Both CFR+ and DCFR slow dramatically after ~50
   iterations at 11×11. Is this a property of the game (near-equilibrium
   fine-tuning is inherently slow) or an algorithmic limitation that better
   discounting could overcome? *(ANSWERED 2026-07-02: neither — it was the
   measurement. The historical best response was clairvoyant (per-deal max),
   a strict upper bound whose gap doesn't vanish at equilibrium; the "wall at
   ~0.05" is the clairvoyance gap of an essentially converged strategy. With
   the exact per-info-set best response there is no wall at any tested density.
   Full trail in "The wall was the ruler" above. Follow-up: run `eval-ckpt` on
   the 11×11 checkpoints to learn their true exploitability.)*

1b. **Asymmetric mão-de-onze states — the right method.** Is the freeze-accept +
   best-response-equity policy iteration the correct, general way to solve every
   "one player at 11" state, and does it generalise down the lattice (each state
   seeded from its solved neighbour)? Does the accept-set iteration always
   converge, or can it cycle (needing damping)? Is the optimal accept ever
   genuinely *mixed* near the boundary rather than a pure threshold?
   *(Answered 2026-07-02: freeze-and-iterate was the wrong frame entirely. With
   position in the info set, plain dealer-exact CFR learns the accept jointly
   with the card play and the tiny-config wall vanishes — no external iteration,
   no damping question. Mixing at the boundary is handled natively by regret
   matching. Remaining: confirm at full scale on a VM.)*

1c. **Does position-in-info-set lower the 11×11 floor?** *(Answered same day,
   negatively — by measurement rather than a solve.)* A census of info sets
   differing only by position found **zero** aliased keys at 11×11 across
   strided deal subsets from 300 to 30,000 (the history encoding disambiguates
   attribution: own face-down plays record the card, the opponent's record
   `OpponentPlayedHidden`, and face-up trick winners are computable). At 11×10
   the aliased keys number **exactly 403 — one per decider hand**: the accept
   nodes and nothing else. So ~0.05 at 11×11 was just the lax target, not an
   abstraction floor; the existing 11×11 solve, its 0.5564/0.4437 per-dealer
   values, and its checkpoints remain valid as-is, and no 11×11 re-solve is
   needed before 11×10. Tightening mv(11,11,·) by resuming the legacy
   checkpoints is an optional precision improvement, not a prerequisite.
   Caveat for lower states: `Raise`/`AcceptRaise` tokens are
   attribution-ambiguous the same way the accept node was, so real card-play
   aliasing may reappear at raise-enabled states (10×10 and below) — position
   in the key inoculates those too.

2. **Lower-score tree sizes:** The 11×11 subgame has no raises, making it the
   smallest in the game. We don't yet know tree sizes at, say, 5×5 where the
   full raise ladder is available. This is the critical unknown for
   estimating total compute.

3. **Turnup class independence:** Each of the 9 turnup classes at a given
   score state is completely independent — different set of deals, different
   information sets, no shared state. We can solve them in parallel with no
   coordination. *(Demonstrated 2026-06-28: all 9 TCs at 11×11 solved on 3 VMs
   in parallel with no coordination; finals landed in a 0.005-wide band, so the
   turn-up class barely affects difficulty.)*

4. **Score state dependencies:** Score states must be solved in descending
   order of total points (terminal states first, working backwards). The
   "widest" layers are totals 10-12, each containing ~6 unique score pairs ×
   9 TCs = 54 independent subgames.

5. **Neural net distillation:** With a full tabular solution, we can train a
   small NN to approximate it. Input: info set features (~30 floats). Output:
   action probabilities. A 100KB MLP could fit on a smartphone. How much
   exploitability does distillation sacrifice?

6. **Multiple equilibria:** Do CFR+ and DCFR converge to the same strategy?
   We expect very similar equilibria given the ordinal card strength structure,
   but haven't verified.

7. **Autoresearch gap:** The best recorded autoresearch run is far better than
   the harness baseline but still much worse than the longer hand-tuned DCFR
   benchmark. Is that mostly the shorter budget, the current mutable-file
   interface, or the proposal search strategy? This is an inference from the
   current results, not yet a proved diagnosis.

8. **Advanced tabular variants before neural CFR:** Recent DeepPDCFR work points
   at DCFR+ and PDCFR+ as the relevant tabular ingredients for us. We should
   first implement and benchmark those variants in the exact tabular solver;
   only if tabular solving becomes impractical should we attempt the full
   neural/model-free DeepPDCFR-style approach.

9. **CFR autoresearch operating model:** Should the default run be
   backlog-guided, free exploration, or a scheduled mix? The first shipped ops
   workflow should expose both modes and record the chosen mode, model, effort,
   max iterations, per-iteration LLM budget, and overall LLM budget with each
   experiment so the results remain interpretable.

10. **Tracking system:** MLflow is likely the right middle ground for this
    personal project: richer than `results.tsv`, reusable for future
    autoresearch loops, and straightforward to keep private behind Tailscale.
    The default target should be a persistent personal-server MLflow deployment
    rather than a temporary localhost store; VM runs should log directly to
    a project-level tracking URI that the runner exports as
    `MLFLOW_TRACKING_URI`, so history and artifacts survive VM teardown without
    SSH database moves. Keep a local file-backed store as a development/offline
    fallback. MLflow should log all attempted candidates and their artifacts,
    while git remains the mechanism for preserving, reviewing, and promoting
    accepted code states. The workflow should expose the dashboard through both
    `make ops` and a direct `make` target, preferring the Tailscale-only MLflow
    URL and falling back to a loopback `mlflow ui` for local file stores.

11. **Environment hygiene:** CFR autoresearch needs explicit non-secret env
    examples and secret indirection. `.env` files should stay ignored and denied
    to Claude proposal runs. If a local env value is an `op://...` 1Password
    reference, the ops runner should resolve it in memory and inject it into the
    child process without storing the secret value in files, prompts, logs, or
    MLflow params.

12. **Operations interface:** Solver operations stay CLI- and agent-driven.
    `make ops` provides the direct controls for local play, VM work, solver runs,
    deployment, and retrieval; agents can supervise the longer-running work. The
    retired private web console is deliberately not a future product direction.

13. **Dominated-action residue: the average lies where the strategy doesn't.**
    Browsing the freshly-solved 11-column in the Study lab surfaced a subtle
    product problem: the DCFR *average* strategy puts a sliver of mass on
    strictly dominated actions (canonical case: fold two manilhas at the mão de
    onze, where `Q(accept) − Q(fold) ≈ 33pp` yet `p(fold) > 0`). It is *not* a
    solver bug — CFR+ regret matching drives the *current* strategy to exactly
    zero there within a few sweeps; the residue is early-iterate weight trapped
    in the average, i.e. convergence noise carrying zero strategy information
    (only meta-information about how converged the solve is). At the old
    moderate-ε exports it reached ~0.6% (visible; the frontend renders ≥0.1%);
    at the clean 1e-5 column it's ~0.02% (below the display floor). A 1e-5
    exploitability bound does not rule it out — the wasted equity is far below
    the bound — so the metric reads "converged" while a user sees "fold two
    manilhas" and concludes the solver is broken. **Resolution direction (plan
    74):** keep `.teach` raw (training data + convergence diagnostic); for
    *future* solves use tail averaging (average restart → dominated mass becomes
    exactly zero by construction, mixed spots untouched); for the *current*
    column use **purify-then-certify** — zero the small large-Q-gap mass,
    renormalize, then re-run the exact best response to *measure* that purified
    ε ≤ raw ε and ship the strategy with that certificate (turning "Data is
    exact" from an ε/δ heuristic into a measured claim; cf. Ganzfried &
    Sandholm, where purified strategies sometimes *beat* the raw average). The
    two mechanisms cross-check: a tail-averaged solve should certify with zero
    purification needed. Open sub-question: does the residue-mass histogram
    (per-info-set mass above 1pp/5pp/20pp Q-gap) reveal any *genuinely* mixed
    near-dominated spots, or is everything above ~1pp Q-gap pure noise? The
    quantification sweep (plan 74 step 0) will answer this and set the
    purify/assert thresholds. **Answer (2026-07-09):** the relevant residue
    diagnostic must be capped to individually-small actions (`p < 1%`); an
    uncapped all-action sweep measures strategic/non-residue rows. Under that
    cap, the current tc0/d0 column has worst per-info-set mass of 2.9948% at
    Q-gap >5pp and zero rows at or above 3%, so the export assertion is
    `p < 1%`, Q-gap >5pp, total mass <=3%. Purifying that mass and re-running
    the exact legal BR certified every 11-column rung at or below raw
    exploitability (e.g. 11x9 raw 9.95e-6 -> purified 4.15e-6); 11x0 needed no
    purification. The Study files now ship purified `p`, raw `raw_p`, unchanged
    `q`, and a certificate block; the UI gates display with `p < 3%` and Q-gap
    >1pp unless "Raw residue" is enabled.

14. **Per-infoset best-response gap — does the tremble refinement reach
    near-Nash quality, or just "better than garbage"? (opened 2026-07-11)**
    Self-loss and own-reach (the two tests used to accept the 2026-07-11
    tremble refinement above) are proxies, not exploitability measures.
    Self-loss compares a node's mix against its own children's `q` — but `q`
    is itself CFR's self-play value, computed by the same possibly-still-
    undertrained descendants; a subtree that is internally self-consistent
    but jointly wrong would show self-loss ≈ 0 while still being genuinely
    exploitable. Own-reach only measures whether a node was visited, not
    whether the resulting strategy is any good. Neither can rule out — or
    confirm — that the newly-trained off-path spots are actually close to
    equilibrium quality rather than merely no-longer-random.

    A real test exists and is nearly free: `best_response_value_for_profile`
    (`cfr.rs:1889-2036`) already computes a genuine backward-induction best
    response at every one of the best-responder's info sets as an
    intermediate of its bottom-up pass (`cfr.rs:1963-2017`) — a value that
    recurses through the *entire* remaining subtree using real best-response
    choices throughout, not the solver's own possibly-noisy `q`. It is
    currently thrown away after producing the one whole-game aggregate.
    Exposing it (normalize by the counterfactual weight already accumulated
    per info set) gives `br_value` per node; `gap = br_value − Σ_a p(a)q(a)`
    (the second term already sits in the shipped chart) is a rigorous,
    adversarial measurement — not a proxy — of how exploitable a specific
    off-path spot is. A small gap at a previously-flagged info set is a real
    claim: no strategy, human or otherwise, can beat this exact opponent
    profile by more than that margin from that point on.

    Plan: expose per-infoset `br_value`, run it against both the pre-tremble
    checkpoints (`gs://truco-solver-runs/teacher2-20260704/`,
    `10x10-full-20260709/`) and the post-tremble ones
    (`gs://truco-solver-runs/tremble-refine-20260711/*/refined.ckpt.bin` —
    both generations are still retained) for the previously self-loss/
    own-reach-flagged info sets across the 5 shipped spots. Use the result to
    decide, per spot, whether the current run needs more trembling (gap
    still large and dropping fast) or is ready for a clean (non-trembled)
    convergence tail (gap already small, or improvement rate has
    plateaued) — checkpointing partway through any further tremble dose so
    the decision is made from an actual trajectory, not a single before/
    after snapshot.

    **Bug found and fixed while implementing this (2026-07-11):** the first
    `--br-gaps` export computed `eq_value` from `.teach`'s stored `q`, which
    is a self-play value under the RAW (unpurified) average strategy
    (`strategies_by_table_idx` pulls `data.average_strategy()` directly).
    `br_value` was best-responded against the PURIFIED profile. Comparing
    the two therefore compared `br_value` against a different implicit
    opponent than it was computed against, breaking the `br_value ≥
    eq_value` best-response guarantee — surfaced as large, real-looking
    negative gaps. Fix (commit `a95d621`): recompute `eq_value` from a
    second `compute_teacher_data` pass run under `purified_strategies` (the
    SAME profile `br_value` was resolved against), used only inside the
    `"br"` sub-object; the row's shipped `"q"` field is untouched. Verified
    with a regression test (`teacher_export.rs`,
    `br_value_never_below_eq_value_under_matched_purified_profile`) that
    reproduces the violation on the old code path and asserts it's gone on
    the new one.

    **False alarm chased afterward, root cause + lesson:** a re-run of the
    5-spot campaign with the fix still showed thousands of negative gaps at
    `11x10-d0`. Four independent reproductions (single-spot sequential,
    4-way-parallel, and an exact rebuild of the fix commit with zero other
    changes) all came back completely clean, and every flagged row in the
    dirty dataset matched the OLD (pre-fix) formula exactly
    (`eq_value == Σ purified_p(a)·raw_q(a)`) — proving the "dirty" download
    was stale, pre-fix data, not a live bug. Root cause: the campaign
    script's upload step used `gsutil ... || true`, silently swallowing
    upload failures; combined with never deleting prior-run output objects
    at the same GCS path (a deliberate safety property of this
    environment — bulk deletes require explicit authorization), a failed
    upload for one spot was indistinguishable from success and left old
    data in place under a passing-looking `ALL_DONE` marker. Fixed by
    checking the upload's own exit code (retry once, then hard-fail into
    `FAILED.txt`) rather than deleting old outputs preemptively — this
    closes the actual hole without depending on cleanup discipline. Lesson
    for future GCS-backed campaign scripts: never mask an upload step with
    `|| true` when the destination path can already hold a prior run's
    output: a masked failure there is a *stale-success*, strictly worse
    than a visible one.

    **Result (5 spots, both generations, weight > 1e-4 nodes only, gaps in
    match-equity pp via ×50):**

    | spot | pre mean | pre p90 | post mean | post p90 | reduction |
    |---|---|---|---|---|---|
    | 11x11-d0 | 3.74pp | 14.39pp | 0.33pp | 1.24pp | 91% |
    | 11x11-d1 | 5.04pp | 18.81pp | 0.52pp | 1.49pp | 90% |
    | 11x10-d0 | 5.93pp | 24.55pp | 0.50pp | 1.55pp | 92% |
    | 11x10-d1 | 9.47pp | 30.24pp | 0.63pp | 1.94pp | 93% |
    | 10x10-d0 | 1.76pp | 5.77pp | 0.49pp | 1.41pp | 72% |

    Aggregated across all 5 spots (reach-weighted): pre-tremble mean gap
    4.53pp, with 13.8% of total reach-weight sitting at info sets exploitable
    by more than 10pp — a real, material amount of bad play. Post-tremble:
    mean gap 0.48pp, and the >10pp-exploitable weight share collapses to
    0.076%. Median (weighted) gap is 0.0pp both before and after — most of
    the game was always fine; the problem (and the fix) is concentrated in
    the tail. The trembling-hand refinement is a large, genuine, *measured*
    improvement in off-equilibrium quality, not just a change in appearance:
    this directly answers the open question the whole tremble-refinement
    effort was undertaken to resolve. It is not a perfect fix — a small tail
    remains post-tremble (99th percentile still 2.7-7pp per spot, and
    individual worst-case info sets as high as 35pp at 10x10-d0), so
    "reasonably converged, materially better, but not uniformly solved to
    machine precision everywhere" is the honest characterization, not
    "superhuman everywhere." Status: measured, done. Whether to spend more
    trembling on the remaining tail vs. move to a clean convergence tail is
    the open step-3 decision this measurement was built to inform.

    **Convergence-rate follow-up (2026-07-11):** rather than guess at the
    step-3 decision from a single before/after snapshot, resumed the
    `11x10-d1` post-tremble checkpoint (iteration 1320, the spot with the
    worst starting tail: p99 5.49pp) through 4 more stages of 300 tremble
    iterations each (constant ε=0.01, matching where the original 0.05→0.01
    anneal ended — held flat so any slowdown is attributable to diminishing
    returns from more *time*, not a shrinking perturbation), measuring the
    full BR-gap distribution at each stage on a spot `c2-standard-8` VM
    (`--jobs 8`, ~3.2-3.9s/iteration — about 13x faster than the original
    single-job ~52s/iteration this baseline was measured at). Two real bugs
    surfaced and were fixed along the way: (1) `--eps` (the convergence-
    stopping target) defaults to 0.01, and since this checkpoint's
    exploitability was already below that, the first attempt's solve-tc
    stopped after the first `expl_every` check (10 iterations) instead of
    running the full `--extra-iters` — fixed with an explicit `--eps 0`;
    (2) none beyond that (one bug, not two).

    | | iter | weighted mean | weighted p90 | weighted p99 | max |
    |---|---|---|---|---|---|
    | baseline | 1320 | 0.6271pp | 1.9350pp | 5.4850pp | 9.7350pp |
    | stage 1 | 1620 | 0.3665pp | 1.0750pp | 3.4600pp | 8.8450pp |
    | stage 2 | 1920 | 0.2233pp | 0.7350pp | 2.6000pp | 5.8250pp |
    | stage 3 | 2220 | 0.1654pp | 0.5350pp | 2.0650pp | 4.8650pp |
    | stage 4 | 2520 | 0.1355pp | 0.4400pp | 1.7750pp | 4.8100pp |

    Stage-over-stage reduction in the weighted mean: 41.6% → 39.1% → 26.0% →
    18.0%. This is a clean, gradually-decaying-but-still-substantial curve —
    the per-stage multiplicative "fraction remaining" (0.584, 0.609, 0.741,
    0.819) is climbing toward 1.0 (diminishing returns, as the mechanism
    argument in item 15 predicts) but has NOT flattened to ~0% by stage 4:
    an 18% cut on the 4th consecutive 300-iteration block is still a real
    improvement, not noise. Total cost for all 4 stages: ~69 minutes of
    solve time (plus export/measurement overhead) on one spot VM, on the
    order of $0.20-0.30 — cheap enough that "give it more time" is a very
    good trade at this point, and applying the same extension to the other
    4 spots (which started with smaller tails than `11x10-d1`'s) should be
    similarly cheap. Verdict: more trembling is still clearly worth it here;
    there's no evidence yet of having hit the point where a clean
    (non-trembled) convergence tail would be a better use of compute than
    continuing to tremble. Open: how many more stages before the curve
    truly flattens, and whether the other 4 spots show the same shape.

    **Eps-annealing validation (2026-07-12): the hypothesis did NOT hold —
    sustained beats annealed.** The convergence-rate curve above fits a
    geometric decay extrapolating to a p99 floor around ~1.45pp at constant
    ε=0.01 — above the 1pp target — consistent with the tremble-floor
    mechanism (`σ'(a) = ε/|A| + (1-ε)σ(a)` permanently bakes in an `ε/|A|`
    floor a true best-responder can always exploit at fixed ε). The natural
    next hypothesis: anneal ε DOWN instead of holding it flat, to shrink the
    floor itself. Tested by resuming the SAME iteration-1320 checkpoint for
    500 more iterations with `--tremble-eps 0.01 --tremble-eps-end 0.001`
    (linear anneal, `TrembleSchedule::eps_at`, `cfr.rs:496`) — reaching
    iteration 1820, i.e. landing inside the constant-ε trajectory's
    stage-1(1620)–stage-2(1920) window, so directly comparable.

    | | iter | weighted mean | weighted p99 |
    |---|---|---|---|
    | constant ε=0.01, stage 1 | 1620 | 0.3665pp | 3.460pp |
    | **annealed ε 0.01→0.001** | **1820** | **0.3522pp** | **2.970pp** |
    | constant ε=0.01, stage 2 | 1920 | 0.2233pp | 2.600pp |

    Linearly interpolating the constant-ε curve to iteration 1820 (67% of
    the way from stage 1 to stage 2) predicts ≈0.271pp mean / ≈2.89pp p99 at
    that iteration count *if annealing were exactly as effective as holding ε
    flat*. The measured annealed result (0.352pp / 2.97pp) is **worse than
    that interpolation on both metrics** — clearly worse on the mean (30%
    higher than predicted), roughly matched-to-slightly-worse on p99. No
    breakthrough past the ~1.45pp floor; if anything, annealing performed
    less well than just continuing at constant ε for the same 500-iteration
    budget.

    This makes sense mechanistically, not just empirically: `eps_at` anneals
    *linearly* over the whole 500-iteration window, so ε is already
    meaningfully below 0.01 for most of the run (e.g. ≈0.0046 by iteration
    1620) — meaning less of the run spends time at full trembling strength,
    i.e. less time actively defeating the zero-prob pruning lockout that
    causes off-path garbage in the first place (item 0 / item 15's
    mechanism). And because CFR+'s averaging weight is *linear in iteration
    number*, the low-ε tail iterations — which behave closer to a
    non-repairing clean continuation — get disproportionately large weight
    in the final average, same effect flagged for the *sustained* schedule
    in item 15 but here working in the wrong direction (less repair, not
    more, dominating the average). Verdict: for further refinement of an
    already-converged checkpoint, **hold ε constant rather than anneal it
    down** — item 15's "early-tremble-then-anneal" idea may still be right
    for a *from-scratch* solve (where early lockout-prevention is the goal
    and the ε is annealed toward zero over a much larger iteration budget,
    so the low-ε tail is a much smaller share of total weight), but it does
    not transfer to this refinement setting at this schedule length.

    (Methodology note: this and the table above both apply the same
    `weight > 1e-4` practical-significance floor used throughout this
    session's BR-gap distribution stats — confirmed by reproducing the
    baseline's exact recorded `n=10786` before trusting the new number
    against it; a first pass without that floor and with an incorrect ×100
    pp-conversion factor instead of the established `PP_PER_Q_UNIT=50` gave
    implausible results that would have been reported as "annealing badly
    regresses quality" had they not been checked against the known-good
    baseline count first.)

    **Constant-ε benchmark extended to the other 4 spots (2026-07-12):**
    since constant ε beat annealing, ran the same 4-stage×300-iteration
    treatment (3 stages×100 for `10x10-d0`, whose ~39.5M-info-set tree is
    ~7× the ~5.6M of the other four — see below) on `11x10-d0`, `11x11-d0`,
    and `11x11-d1`, each on its own spot VM in a different GCP region
    (`c2-standard-8`'s C2-CPU quota turned out to be 8 **per region**, so
    the 4 spots had to be spread across `us-central1`/`us-west1`/`us-east1`/
    `us-east4` rather than run in one). `10x10-d0` was OOM-killed 49s into
    its first stage on `c2-standard-8` (32 GB) — its original single-job
    solve already peaked at 11.25 GB and `--jobs 8`'s parallel tree-build
    pushed it over; switched that spot to `n2-highmem-8` (64 GB, same
    vCPU/job count) and it ran cleanly.

    | spot | baseline p99 | final-stage p99 | extrapolated floor |
    |---|---|---|---|
    | 11x10-d1 (original, 4×300 iters) | 5.485pp | 1.775pp | ~1.4485pp |
    | 11x10-d0 (4×300 iters) | 5.910pp | 2.120pp | ~0.9525pp |
    | 11x11-d0 (4×300 iters) | 2.675pp | 1.070pp | ~0.7614pp |
    | 11x11-d1 (4×300 iters) | 6.490pp | 2.575pp | ~0.9615pp |
    | 10x10-d0 (3×100 iters) | 4.740pp | 3.250pp | ~2.4405pp |

    (Floors extrapolated by the same geometric-decay fit throughout: average
    the ratio of successive stage-over-stage differences, then sum the
    resulting geometric series from the last observed point. `10x10-d0`'s
    fit rests on only 2 ratios instead of 3 — noticeably less robust than
    the others', and see the dosage caveat below. All are rough estimates,
    not certified bounds.)

    The interesting result: **11x10-d1 looks like an outlier, not the
    template — and 10x10-d0 looks like a different, worse outlier.** Three
    of five spots (`11x10-d0`, `11x11-d0`, `11x11-d1`) extrapolate *below*
    1pp (0.76–0.96pp) — including `11x11-d1`, which started with the single
    worst baseline p99 of all five spots (6.49pp) yet is tracking toward a
    lower floor than 11x10-d1. `11x11-d0` is already at 1.07pp after just
    1200 more iterations (~17 min at `--jobs 8`) and likely crosses 1pp
    within another stage or two. `11x10-d1` (~1.45pp) and now `10x10-d0`
    (~2.44pp, the worst floor of all five) extrapolate *above* target.

    `10x10-d0`'s number carries a real caveat the others don't: it only got
    300 total additional iterations (3×100) against the other four's 1200
    (4×300), scaled down going in because its ~39.5M-info-set tree (~7× the
    ~5.6M of the other four) costs proportionally more per iteration. A
    ~2.44pp floor extrapolated from a much smaller "dose" than the other
    four spots' 1200-iteration-based estimates is not apples-to-apples —
    it's plausibly pessimistic (less training time fit before the geometric
    curve stabilizes), but it could also genuinely reflect that a much
    larger tree needs more than proportionally more iterations to repair to
    the same standard. Distinguishing those needs a matched-iteration rerun
    on `10x10-d0`, not yet done.

    Overall verdict on "$0.5–1.5 total, ~30–90 min/spot to get p99<1pp
    everywhere": holds for 3 of 5 spots via constant-ε alone. `11x10-d1` and
    `10x10-d0` both need either more iterations than tested here (open
    whether their curves would eventually cross 1pp given enough), a
    different approach, or acceptance that their floors sit above target —
    and `10x10-d0` specifically needs a fairer (matched-dose) measurement
    before drawing a firm conclusion either way.

    **Infra notes for future GCP campaigns on larger-tree spots** (all hit
    on `10x10-d0` specifically, none on the other four): (1) `c2-standard-8`
    OOM-killed within 49s of `--jobs 8` on a tree whose original single-job
    solve already peaked at 11.25 GB — switched to `n2-highmem-8` (same
    vCPU/job count, double the RAM). (2) `export-teacher`'s hard
    `assert_q_gap_residue` gate (a deliberate, correct guard against
    shipping an under-converged raw strategy) fired on this intentionally
    "mid-refinement" checkpoint — the existing `--allow-residue` flag
    (designed for exactly this "known-less-converged intermediate solve"
    case) downgrades it to a measured warning. (3) `10x10-d0`'s ~6.3 GB
    checkpoints, retained across stages without cleanup, exhausted a 50 GB
    boot disk (`No usable temporary directory found`, not an OOM) —
    resolved by deleting `resume_ckpt` once each stage's strategy is
    extracted from it, plus bumping to a 150 GB disk for headroom. None of
    these affected the other four (~5.6M-info-set) spots, which all ran
    clean on the original `c2-standard-8`/50 GB/no-`--allow-residue`
    configuration — all three are specifically artifacts of `10x10-d0`'s
    much larger tree.

    **Pushing all 5 spots to ~95%-of-floor (2026-07-12, in progress):**
    given the cost estimate above was cheap (~$3-5 total, hours not days),
    resumed each spot from its ORIGINAL tremble-refine checkpoint again
    (the previous campaign didn't preserve intermediate checkpoints, so this
    redoes the already-measured stages too) for enough additional
    constant-ε=0.01 iterations to close ~95% of the remaining gap to each
    spot's extrapolated floor: `11x10-d1` +2700 (9×300 total from baseline),
    `11x10-d0` +3900 (13×300), `11x11-d0` +3300 (11×300), `11x11-d1` +4200
    (14×300), `10x10-d0` +1200 (12×100). This time each stage's checkpoint
    is uploaded to `latest.ckpt.bin`/`latest_stage.txt` so a preempted spot
    VM resumes from its last successful stage instead of restarting —
    verified working (`found prior progress: stage N already done` in
    serial output) after two real spot-preemption events during the run.

    | spot | predicted floor | actual result | beat prediction by | crossed 1pp? |
    |---|---|---|---|---|
    | 11x11-d0 | ~0.7614pp | **0.450pp** | 0.31pp | yes |
    | 11x10-d1 | ~1.4485pp | **0.875pp** | 0.57pp | yes |
    | 11x10-d0 | ~0.9525pp | **0.615pp** | 0.34pp | yes |
    | 11x11-d1 | ~0.9615pp | **0.740pp** | 0.22pp | yes |
    | 10x10-d0 | ~2.4405pp | **1.655pp** | 0.79pp | **no** |

    **Campaign closed 2026-07-13.** All five spots beat their own
    extrapolated floors, `10x10-d0` by the largest absolute margin of any
    spot (0.79pp) — but it's also the only one of the five that didn't
    cross the 1pp target, landing at 1.655pp. That's consistent with, not a
    contradiction of, the "floors are too pessimistic" finding below:
    `10x10-d0` started from a much worse predicted floor (its own 4-point
    fit rested on only 2 ratios from a deliberately-scaled-down initial
    dose, the least reliable of the five), so even a proportionally similar
    overshoot past its floor wasn't enough to close the larger absolute gap
    to 1pp. Whether more iterations would eventually cross it, or whether
    `10x10-d0`'s much larger tree (~39.5M vs ~5.6M info sets) has a
    genuinely harder floor, remains open — this campaign didn't answer that,
    only that "the ~30–90 min/spot estimate crosses 1pp everywhere" does
    NOT hold for `10x10-d0` at the dose tested here.

    The other four not only crossed the 1pp target but beat their own
    extrapolated floors by a comfortable margin — including `11x10-d1`,
    whose floor was the whole reason the eps-annealing experiment above got
    run in the first place (predicted to plateau at ~1.45pp, actually
    reached 0.875pp). The 4-point geometric-decay fit used throughout this
    investigation is systematically **too pessimistic**: it captures the
    SHAPE of the near-term decay correctly enough to rank spots and
    estimate iteration budgets, but the true curve keeps delivering more
    improvement past where the naive extrapolation says it should flatten
    out. Practical implication: don't treat these floor numbers as a hard
    ceiling on what's achievable — they're a reasonable lower bound on how
    many more iterations are worth trying, not an upper bound on the
    resulting quality.

    Sanity-checked the redo against the original measurement: stage 1 of
    `11x11-d0`'s rerun reproduces the prior campaign's stage 1 numbers
    exactly (p99 2.005pp both times, same n), confirming `solve-tc` is
    deterministic and the redo-from-scratch approach (forced by not having
    preserved checkpoints the first time around) didn't introduce drift.
    `10x10-d0` was preempted (spot reclaim) twice during its ~12-hour total
    campaign and cleanly resumed both times from its last completed stage
    via the `latest.ckpt.bin` + `latest_stage.txt` mechanism added for this
    round, without redoing any already-finished stages — a real point in
    favor of building that resilience in from the start for any future
    long-running spot-instance campaign, rather than treating it as
    optional. The one unavoidable redo cost, paid once by every spot (not
    specific to the preemptions), was re-running the stages this campaign
    shares with the earlier one — the prior (rate2) campaign didn't
    preserve intermediate checkpoints, so extending any spot's trajectory
    meant starting over from the original `refined.ckpt.bin` and redoing
    already-measured ground. Total campaign cost across all 5 spots stayed
    in the same ~$3-5 ballpark as the original estimate despite that.

    **Refit floors from the full rate5 trajectories, and cost of a further
    push (2026-07-13).** The rate2/rate5 floor estimates above were fit from
    only 2-4 points each. With all 9-14 stages per spot now on hand, refit
    the geometric decay ratio from each spot's last 5 stage-over-stage
    ratios (more representative of current-regime behavior than the early,
    noisier ratios) and re-extrapolated the floor and the iteration budget
    to close 95% of the remaining gap to it — the same target definition
    used to plan the rate5 campaign itself:

    | spot | decay ratio | new floor | current p99 | +iters for 95%-of-floor | +time | +$ | lands at |
    |---|---|---|---|---|---|---|---|
    | 11x11-d0 | 0.910 | ~0.00pp | 0.450pp | 9,600 | 8.2h | $1.48 | ~0.02pp |
    | 11x10-d1 | 0.832 | ~0.33pp | 0.875pp | 5,100 | 5.0h | $0.90 | ~0.36pp |
    | 11x10-d0 | 0.818 | ~0.37pp | 0.615pp | 4,500 | 4.3h | $0.78 | ~0.38pp |
    | 11x11-d1 | 0.871 | ~0.20pp | 0.740pp | 6,600 | 7.2h | $1.31 | ~0.23pp |
    | 10x10-d0 | 0.872 | ~1.08pp | 1.655pp | 2,200 | 27.3h | $3.36 | ~1.11pp |

    (Time/$ use each spot's own measured per-iteration rate: ~3.0-3.9s/iter
    on `c2-standard-8` spot for the four ~5.6M-info-set spots, ~44.7s/iter on
    `n2-highmem-8` spot for `10x10-d0`'s ~39.5M-info-set tree — `10x10-d0`'s
    iterations are individually ~12x slower, not just more numerous, which
    is why it dominates both the time and cost columns despite needing the
    fewest raw iterations of the five.) Running all five in parallel (one
    VM per spot, as both prior campaigns did) costs about **$7.83 total,
    gated by `10x10-d0` at ~27 hours** — the other four would sit idle
    (finished) for all but the first ~8 hours of that window.

    The `10x10-d0` row is the one worth reading carefully rather than
    trusting at face value. Its refit floor (~1.08pp) is a large downward
    revision from the rate2-era prediction (~2.44pp) — expected, since more
    data close to the asymptote makes for a better fit — but even a full
    95%-of-floor push under this model lands at ~1.11pp, **still above the
    1pp target**. Taken literally, the model says no finite amount of
    constant-ε=0.01 tremble refinement from here reaches 1pp. That should
    NOT be read as "1pp is unreachable": every one of the five floor
    predictions made this investigation (rate2 AND this refit) has proven
    too pessimistic once actually tested, `10x10-d0` itself by the largest
    margin of the five (beat its rate2-era floor by 0.79pp). The honest
    summary is that the extrapolation can't currently promise a specific
    iteration count that crosses 1pp for this spot — the only way to find
    out is to spend the ~$3-4/27h on more iterations and measure again,
    the same way the last four rounds of "the floor says X" were each
    resolved by just running it.

    **rate6 campaign (2026-07-13): push-with-plateau-detection, then verify
    that stopping trembling doesn't undo the repair.** Rather than a fixed
    iteration budget, resumed all five spots from their rate5 endpoints with
    smaller stages (300 iters/~20min for the four `c2-standard-8` spots, 50
    iters/~37min for `10x10-d0` on `n2-highmem-8`) and a self-stop rule: a
    spot powers itself off once its last two stage-over-stage weighted-p99
    drops both fall under 0.02pp. Three spots plateaued cleanly this way —
    `11x11-d0` at 0.310pp, `11x10-d1` at 0.465pp, `11x11-d1` at 0.240pp (the
    last after an unusual single-stage jump from 0.32→0.24pp, confirmed
    genuine rather than a fluke by two subsequent flat stages, which is
    exactly what triggered its stop). `10x10-d0` and `11x10-d0` kept
    improving past this note's cutoff. One real deployment bug on the first
    launch attempt: the VMs were created without `--scopes`, defaulting to
    `devstorage.read_only` — every upload silently 403'd for ~30 minutes
    before being caught (no progress.log, no checkpoints, stage 14 of
    `11x10-d0` solved but never persisted). Fixed by deleting and relaunching
    with `--scopes=cloud-platform`; cost of the mistake was ~30min of wasted
    compute per spot, nothing structural.

    Once a spot plateaus, does resuming it with `--tremble-eps 0` (back to
    plain CFR+) risk undoing what trembling just repaired? The mechanism
    argues no (see the `check_stall`-adjacent discussion this session): the
    zero-prob pruning shortcut freezes a subtree wherever it currently
    stands, not at some earlier state, so once an upstream action's
    probability returns to 0 the subtree re-freezes at its *tremble-improved*
    values. Tested empirically on the two earliest-plateaued spots (5 more
    stages each, `--tremble-eps 0 --tremble-eps-end 0`, resumed from the
    plateaued checkpoint): confirmed. Both continued to improve rather than
    regress — `11x11-d0` 0.310pp → 0.225pp, `11x10-d1` 0.465pp → 0.350pp,
    smoothly across all 5 stages, no reversal at any point.

    One hypothesis this test refuted: that the residue this project has been
    carrying since the trembling work began (`assert_q_gap_residue`, the
    reason `--allow-residue` is needed on these checkpoints) would gradually
    wash out of the average once trembling stopped, since CFR+'s
    linear-in-iteration averaging should dilute old floor-induced mass with
    enough new floor-free iterations. It did not move at all: `11x11-d0`'s
    residue read exactly 2.1372% on every one of the 5 stages, `11x10-d1`'s
    exactly 2.6643% — not "roughly stable," bit-for-bit identical each
    export. The most likely explanation is the same pruning mechanism this
    whole investigation is about: whichever info set is driving the
    reported max-residue figure is itself sitting in a branch that gets
    re-frozen (upstream action back at exactly 0) the moment trembling
    stops, so its raw average strategy — the exact thing the residue check
    reads — stops changing at all, not just slowly. If that's right, this
    residue is structurally sticky for any checkpoint that went through
    tremble-then-detremble refinement: it won't self-resolve from more
    plain iterations, only from another (even brief) trembling pulse that
    revisits that specific frozen branch, or from accepting that these
    checkpoints permanently need `--allow-residue` and rely on chart-export
    purification (which does zero out small-mass/large-gap actions) rather
    than the raw guard. Not yet confirmed by directly inspecting which
    info set is responsible — only inferred from the exactness of the
    stall — but consistent with everything else this investigation has
    found about how the pruning shortcut behaves.

    **Campaign conclusion (2026-07-15): all 5 spots under the 1pp target.**
    Four of the five spots resolved cleanly in one tremble → detremble
    round: `11x11-d0` 0.310pp → **0.225pp**, `11x10-d1` 0.465pp →
    **0.350pp**, `11x11-d1` 0.240pp → **0.215pp**, `11x10-d0` 0.275pp →
    **0.225pp** — monotonic improvement during detrembling, residue
    bit-exact-flat throughout, matching the pattern above precisely.

    `10x10-d0` needed a second full round. Its first tremble → detremble
    pass landed at 1.025pp (down from a 1.035pp trembled plateau) — a
    genuine miss, and one that came with a qualitatively different signature
    the whole way: p99 was non-monotonic during detrembling (1.025 → 1.025
    → 1.025 → 1.020 → **1.025**, ending almost exactly where it started) and
    residue *drifted* (3.78% → 3.91% → 3.97% → 3.87%) rather than freezing
    bit-exact like the other four. Best explanation: this spot's tree is
    ~7x larger (~39.5M info sets vs ~5.6M) and was left markedly less
    converged when trembling stopped (1.035pp vs. 0.24-0.47pp for the rest),
    so more than one candidate info set was plausibly competing for
    "worst," with the tail statistics bouncing between them rather than
    tracking one stable frozen bottleneck.

    Given how cheap another round was (order $2-3, hours not days), just ran
    it: resumed trembling directly from the 1.025pp post-tremble checkpoint
    (constant ε=0.01, same plateau-detector), which plateaued cleanly at
    **0.920pp** after 5 stages — crossing 1pp for the first time in this
    investigation. A second detremble-verification pass confirmed no
    regression, settling into a stable noisy band (0.915-0.925pp over 5
    stages) rather than the clean monotonic drop the other four spots
    showed — the same non-monotonic signature as round 1, but this time
    safely under target throughout. Final: **0.925pp**.

    Two practical takeaways for any future spot that misses the 1pp target
    on its first tremble/detremble pass: (1) a second round is a legitimate,
    cheap thing to try — it isn't re-doing wasted work, the checkpoint
    already reflects round 1's progress, and the mechanism (unfreezing
    pruned branches) applies exactly the same way starting from a
    detrembled state as it does from a fresh one; (2) the non-monotonic
    p99 / drifting-residue signature seen on `10x10-d0` both times is worth
    treating as a real, distinct pattern (not measurement noise) on
    trees this size and this far from convergence — the "residue freezes
    bit-exact" finding above should be read as "true once a single
    bottleneck branch dominates," not as a universal property of
    detrembled checkpoints.

    Infra note: also measured where the ~47 min/50-iter-stage wall-clock on
    `10x10-d0` actually goes, from real per-step logs — 62% solving (29 min),
    37% the `--certify --br-gaps` best-response pass (17.5 min), 1% the
    `.teach` export. Checking less often (bigger iteration batches between
    certify passes) trades monitoring granularity for wall-clock: doubling
    batch size saves ~24%, and the ceiling as batch size → ∞ is ~1.62x, not
    2x, since the certify pass is real cost but still the minority of the
    time — solving itself doesn't get any faster no matter how rarely it's
    checked.

15. **Trembling-hand scheduling for future ground-up solves: early-only vs
    sustained-through-run (opened 2026-07-11).** The 2026-07-11 refinement
    patched trembling onto the *end* of an already-converged run. A cleaner
    question: should future from-scratch solves use trembling from early
    iterations by default, so the off-equilibrium garbage problem never
    occurs in the first place, rather than being repaired after the fact?

    The mechanism argues for a narrower version of that idea than "on for
    most of the run." The pruning lockout that *causes* off-path garbage
    (`cfr.rs:1698`, see item 0 and the entry above) is specifically an
    early-run phenomenon — an action hits exactly 0 once, early, and its
    subtree is frozen at initialization noise for the rest of the run.
    Trembling only needs to survive long enough to prevent *that*, not the
    whole run. Sustaining it through most of a solve pays close to the
    full-width-tree-walk tax for the run's entire duration: the measured 17×
    per-iteration overhead at 10×10 came specifically from trembling
    defeating pruning that had *already accumulated* over ~900 prior
    iterations — at iteration 1 of a fresh solve almost nothing is pruned
    yet, so tremble there costs little extra, but by the tail of a normal
    run (where sync CFR+ spends most of its wall-clock, per the "fast early
    convergence... slows dramatically in the tail" observation above) most
    of the tree is pruned and reverting it to full-width is expensive.
    Sustaining tremble through that whole tail risks multiplying the cost of
    any future full-scale solve by something in the same ballpark as that
    17× figure — a real threat to the ~$10-12k full-game Fermi estimate
    ("The cost collapse" above), not a hypothetical one.

    An early-tremble-then-anneal-to-(near-)zero-fairly-soon schedule should
    capture most of the benefit (early lockout prevention) at a fraction of
    the cost (little pruning to defeat that early anyway). It should also
    *reduce* final-average dilution relative to the 2026-07-11 retrofit,
    not just relative to a sustained schedule: CFR+'s averaging weight is
    linear in iteration number (`cfr.rs:1741-1745`), so the retrofit's
    late-appended trembled iterations got outsized weight in the final sum
    (explaining the ~60× purified-eps growth measured at the 11-column
    spots), while an early schedule's trembled iterations would get
    comparatively tiny weight, swamped by the long clean tail that follows.

    Status: open, unproven — a prediction from the mechanism, not yet a
    measured result. Should be tested cheaply (three schedules — none,
    early-only, sustained — on a small/canary game, comparing final eps,
    flagged-info-set share, and total cost) before being adopted as default
    policy for any future full-scale solve.

    **Should the pruning shortcut just be removed instead (asked
    2026-07-15, after the 5-spot campaign closed)?** No — the 17× figure
    above is exactly the cost of doing that permanently rather than as a
    bounded repair, and it would apply to every iteration of every future
    solve, not just a repair phase, which is a real threat to solve
    affordability at this game's scale. The pruning itself is sound once an
    action's probability is genuinely, permanently zero in equilibrium; the
    actual defect is that the current implementation treats "currently
    zero" as "provably zero forever" with no grace period, so a premature
    zero (insufficient evidence, not a real equilibrium property) freezes
    a subtree that never gets a chance to prove itself wrong. Bounded
    early-only trembling (above) is one fix — schedule *when* pruning is
    allowed to engage. A second, more surgical idea, not implemented or
    checked against the regret-matching code, and not yet compared for
    cost against the trembling schedule: change the pruning *trigger*
    itself to require an action's probability to stay at (or near) zero for
    some minimum number of consecutive iterations before the shortcut
    engages, rather than pruning on the very first zero — this would target
    only newly-forming, still-unproven zeros rather than perturbing
    already-well-established ones, and might avoid re-defeating pruning
    that's already correctly locked in on branches that were never in
    question.

16. **Per-infoset solution quality, stored and shown (plan 75, 2026-07-12):**
    the BR-gap machinery from item 14 has so far been an ad-hoc GCP
    diagnostic — computed only on request, at `--min-depth 0 --max-depth 4`,
    never wired into the production export pipeline or the frontend. The
    user's goal is a real product feature: tell a Study-lab user *this
    specific node's* solution quality, not just the spot-wide certified
    exploitability (which, being reach-weighted, is blind to a rare-but-real
    off-path node — see plan 75's "Why" section). Implemented the solver-side
    plumbing (plan 75 steps 1–3):
    - `cfr::best_response_full_from_action_probs` (`cfr.rs`) returns both the
      aggregate `.total` (what `--certify` needs) and the per-info-set
      `.per_info_set` detail (what `--br-gaps` needs) from a SINGLE
      backward-induction pass — `best_response_resolve_for_profile` already
      computed both internally and one or the other was always being
      discarded (`best_response_value_from_action_probs` kept only `.total`;
      `best_response_gaps_from_action_probs` kept only `.per_info_set`).
      `run_export_chart` previously called both separately for the purified
      profile whenever `--certify` and `--br-gaps` were both requested,
      re-running the same pass twice; it now calls the combined function
      once per player. Verified bit-for-bit equivalent to the two separate
      calls it replaces (`best_response_full_matches_separate_total_and_gaps_calls`,
      `teacher_export.rs`).
    - A new full-tree, compact columnar artifact —
      `teacher_export::{BrGapRecord, save_br_gaps, load_br_gaps}` — one record
      per reachable info set (`table_idx, br_value, eq_value, gap, weight`,
      all `f32` after `table_idx`, 20 bytes/record), same
      magic/version/sig_hash-header + atomic tmp-then-rename convention as
      `.teach`. `export-chart` gained `--br-gap-out PATH` (requires
      `--br-gaps`): unlike the chart JSON's `"br"` row field, which stays
      windowed to whatever `--min-depth`/`--max-depth` the chart export is
      using, this artifact covers every info set the best-response pass
      reached — there was never a compute reason for the depth limit, only
      an inlined-JSON-size one, and this artifact isn't inlined into chart
      rows. Round-trip covered by `save_br_gaps_round_trips`.

    Completed the end-to-end `11x10-d1` pilot on 2026-07-14. The exact BR
    pass emitted 2,746,361 reachable records: 52 MiB raw / 13 MiB gzip. Both
    the shallow and deep chart windows now retain additive `table_idx` keys;
    the Study Lab lazily decodes and validates the compact table against its
    score/TC/dealer header, then uses BR-gap as the single headline node-quality
    cue (solid ≤1pp, caution 1–5pp, weak >5pp). Self-loss and own-reach are
    demoted to a technical disclosure explaining a weak/off-path node, and
    remain the fallback for older spots without the artifact. This validates
    plan 75 steps 4–5 while intentionally keeping `--br-gaps`/`--br-gap-out`
    opt-in until broader payload and UX evaluation warrants standardizing it.

17. **Exact per-tier tree-size census, trading space for time (2026-07-15/16).**
    Asked whether tree/info-set size could be *measured* without paying the
    solver's own RAM cost — PSPACE-vs-EXP intuition: a plain DFS over
    `TraversalState` needs only `O(depth)` stack space, independent of how
    many total info sets exist, since it never materializes the tree. New
    `count_tree_size` (`game_tree.rs`) + `solve count-tree` validated against
    the real tree builder (bit-for-bit match, 5 scores × both dealers) and at
    production scale (reproduced the existing 39,508,752-info-set 10×10
    anchor in 138s / ~2.25 GB RSS locally — no GCP high-RAM VM needed to
    *count* a tree whose *solve* needs ~75 GB). Measured all 4 remaining
    ladder tiers exactly: mão de onze 5,611,123 → {1,3} 39,508,752 (7.04x) →
    {1,3,6} 129,144,643 (3.27x) → {1,3,6,9} 341,656,035 (2.65x) →
    {1,3,6,9,12} 812,865,845 (2.38x) info sets — a steadily *decelerating*
    growth ratio, not the flat 3-7x range the old Fermi estimate assumed.

    **Unexpected structural finding:** within a ladder tier, info-set count
    depends *only* on the tier, not the exact score — 9×9 and 10×9 reproduced
    10×10's count exactly, 6×6 matched 8×8, 3×3 matched 5×5, and 1×1/2×2 both
    matched 0×0. So each tier's number is exact for every state in it, not a
    representative sample with unquantified spread. This makes sense in
    hindsight given the ladder-pruning rule is itself keyed only on
    `min(score)` crossing a threshold, not on the score's exact value, but it
    was not obvious beforehand that this would leave the *entire* tree
    bit-identical rather than just the same order of magnitude.

    Also surfaced that the real 10×10 fleet already runs on 2-vCPU
    custom-extended machines (`n2-custom-2-76800-ext`) because **the solve
    itself is single-threaded** — vCPU count only matters for the few-minute
    build step, so cost scales almost entirely with RAM. Combined with real
    GCP N2-custom pricing (pulled from the Cloud Billing Catalog API) and the
    10×10 fleet's actual wall times (4.90-7.56h, mean 6.37h), this produced a
    refined total of **≈$505K spot / $1.12M on-demand** for everything still
    remaining (Stage B's last 2 states plus the {1,3,6}/{1,3,6,9}/
    {1,3,6,9,12} tiers) — see `SOLVER_BENCHMARKS.md`'s "2026-07-15/16" entry
    for the full per-tier table. This replaces the old, never-validated
    "$303k-$3.35M" full-game range with a ~2.2x-wide band and identifies
    where the cost concentrates: the `{1,3,6,9,12}` tier is ~89% of it, and
    its ~1.54 TB/job RAM requirement likely exceeds what N2 custom-extended
    can provision at all (N2's largest predefined config tops out at 864 GB)
    — probably needs GCP's M2/M3 memory-optimized family or a disk-backed
    architecture change, so that tier's number is an optimistic floor, not a
    firm quote.

    **What's still an assumption, not a measurement:** time/cost scale
    linearly off tree size only under the assumption that every tier
    converges in ~900 iterations like the 10×10 anchor. Deeper raise ladders
    are more strategically complex and plausibly need *more* iterations to
    reach the same ε, not just a bigger per-iteration cost — this round only
    measured tree size, not convergence rate, for the three newly-measured
    tiers. Worth an actual scout run (even a bounded, early-terminated one) on
    an {1,3,6} or {1,3,6,9} state before committing real budget.

    **Does zero-prob branch pruning (item 5 above) change this estimate?
    Asked 2026-07-16 — no, and the reason clarifies what the estimate is
    actually made of.** Pruning is not a lever separate from these numbers;
    it was already running during every real solve the estimate is anchored
    on (mão, 10×10), so its benefit is already baked into both the actual
    timings and the linear extrapolation off them. More fundamentally, it
    can't touch the memory side at all: the regret/strategy table is sized
    off the raw info-set count at tree-*build* time, before CFR has run a
    single iteration — pruning only skips subtree *traversal* in later
    iterations once an action's probability is provably zero, it never
    shrinks the table that was already allocated. That's precisely why
    `count-tree` (a plain enumeration, no CFR) matches what the real builder
    allocates: it's measuring the same thing pruning can't shrink. So the
    `{1,3,6,9,12}` tier's ~1.54 TB/job wall stands regardless of how much of
    that tree turns out to be off-equilibrium once solved. The one place this
    reasoning leaves genuinely open (not resolved either way): whether deeper
    ladders get pruned *more* aggressively than {1,3} per unit of extra tree
    size — if so the linear time extrapolation above is conservative (real
    cost lower); if not, it holds. Only a real scout run distinguishes these,
    and every untested tier's RAM requirement (245 GB+) exceeds this
    project's local machine, so that scout has to run on a real (cheap but
    non-zero cost) GCP VM — not done yet, pending a decision to spend on it.

    **Full sunk + projected total, for the record:** mão (216 jobs, done) —
    exact aggregate cost was never logged as a single total, unlike 10×10;
    reconstructing from the known per-job anchor (1380 iters/78 min/5.61M
    info sets on one `n2-highmem-16`) puts it in the tens of dollars, not
    worth more precision than that. 10×10 (9 jobs, done) has a real, logged
    actual cost of **$20.07** ($18.31 compute + $1.36 disk + $0.40 watcher,
    `SOLVER_BENCHMARKS.md` "2026-07-09 (later)") — notably, this independently
    reproduces the exact $0.314/hr spot rate this session derived fresh from
    the Cloud Billing Catalog API, a good cross-check that the pricing model
    above is right. Both sunk costs together (~$20-100) are rounding error
    against the ≈$505K spot / $1.12M on-demand remaining total, so the
    grand total for a write-up is the same number either way.

18. **The response to the $505K census is support closure, not another small
    representation win (2026-07-16).** A sub-$1K target needs >500x against the
    raw estimate, so the first new instrument is a policy-aware version of the
    space-for-time DFS counter. It distinguishes the seductive but insufficient
    profile-support tree from the unilateral best-response closures that retain
    every possible deviating action for one player while pruning only the fixed
    opponent's unsupported actions. The union of both closures is the relevant
    candidate arena for a restricted-game/double-oracle architecture.

    The implementation is opt-in and changes no solve behavior. It supports
    average/current policy values, probability thresholds, and explicit missing
    behavior when projecting a shallow policy into a deeper band. A compact
    streaming policy loader exists specifically so the measurement does not
    recreate the solver's full metadata/RAM problem. Regression tests prove a
    full-support profile reproduces raw count-tree exactly and enforce
    `profile <= BR0/BR1 <= union <= raw`.

    First $0 local baseline at 0x0/TC0/d0 on 1,000 strided deals used an EMPTY
    seed with `all-except-raise` fallback—not an equilibrium and labelled as
    such. Raw was 47.94M nodes / 16.51M info sets; the no-new-raise profile was
    0.355M / 0.120M, but the honest two-BR union was 2.520M / 0.859M: still a
    19.0x/19.2x reduction, yet ~7x larger than profile support alone. This is
    promising plumbing evidence, not the decision: the existing solved 10x10
    policy must supply the next thresholds before a restricted builder is
    commissioned.

    Also corrected the meaning of the proposed relaxed target. Exploitability
    epsilon is the AVERAGE of the two unilateral deviation gains in [-1,1]
    utility. At epsilon=0.01 that is 0.5pp average match-win equity left to best
    response; the gains sum to 2epsilon, so one asymmetric side can be as high
    as 1.0pp. Repricing the census by the measured 90-vs-900-iteration 10x10
    ratio lands near $50.5K spot—worth doing, nowhere near sufficient alone.

    Finally, generic same-band warm-start transfer now copies exact-key/action
    REGRETS from the neighboring score while resetting its stale average and
    iteration weight. The old mão-de-onze history-remap path remains separate.
    It also passed its first $0 effectiveness gate: on adjacent 7x7 -> 8x8,
    TC0/d0, 300 identical strided deals, and deliberately nonuniform symmetric
    continuation values, warm reached epsilon<=0.01 in 10 iterations / 2.7s
    versus cold 90 / 7.4s. At iteration 100 the exact exploitability was
    0.000172 warm versus 0.006642 cold (38.6x lower). This is not yet a
    production multiplier, but it justifies pipeline predecessor selection and
    a single <=$2 spot confirmation. The opt-in pipeline selector is now wired:
    it searches only already-solved one-point-higher states with an identical
    tree-band signature, never overrides a true resume, and releases the
    disk-loaded source table immediately after copying regrets. The full gated
    program, including
    reversible post-warm-up regret pruning, deep-band mini-batched MCCFR,
    precision A/B, and proof-required ex-ante dominance, lives in plan 79.

19. **Reversible regret pruning works mechanically, but warm starts make it
    irrelevant at the chosen stopping target (2026-07-16).** CFR+'s stored
    regrets are clamped nonnegative, so "strongly negative cumulative regret"
    does not exist in the production representation. The opt-in implementation
    therefore adds a compact `f32` shadow of unclamped instantaneous regret,
    begins only after a warm-up, prunes only traverser actions whose ordinary
    current probability is exactly zero, and periodically walks the complete
    tree for both alternating player passes. The shadow is not checkpointed;
    resumed jobs safely warm it again rather than pruning from no evidence.

    On the 300-deal adjacent-score warm control, a conservative full revisit
    every 2 CFR rounds held the exact exploitability trajectory within `8e-6`
    absolute of unpruned and changed iteration-100 epsilon only 0.000172 ->
    0.000180, while reducing total wall 8.6s -> 8.2s. A 10-round revisit was a
    bad trade: 8.1s but epsilon 0.000527. The decisive observation is earlier:
    neighboring-score regrets already put the target at epsilon=0.001661 by
    iteration 10, the end of the pruning warm-up. For the epsilon=0.01 cost
    program, there are no post-warm-up iterations left to accelerate. This is
    a useful negative result: keep the mechanism for tight/deep refinements,
    but do not spend on a production scout or multiply its small speedup into
    the full-game budget.

20. **The Study release now uses the final refined profiles and external
    derived artifacts (2026-07-16).** The earlier 11×10 dealer-1 BR pilot was
    not evidence of a solver-certificate disagreement: its chart had simply
    been exported before the final stage-5 checkpoint and without `--certify`,
    so it carried neither raw nor purified epsilon. A single reproducible
    re-export from each final post-tremble checkpoint produced matching shallow
    charts, deep charts, and full-tree BR tables for all five refined spots.
    The replacement 11×10 d1 profile has raw epsilon 0.0030554 and purified
    epsilon 0.0001092; its chart and 2.70M-record BR table now describe the same
    strategy.

    The final five tables contain 34.61M reachable records. The 10×10 table is
    the payload outlier at 156.25 MiB gzip versus roughly 13 MiB for each other
    spot, validating the earlier warning not to put these binaries in Git. The
    10×10 profile completed exact raw/purified BR evaluation (0.0029086 and
    0.0014782), but its raw low-probability/high-Q-gap residue is 3.936%, above
    the normal 3% teacher-export guard. It is included because it is one of the
    five completed refined solutions, while retaining that explicit
    provisional caveat.

    The immutable browser release is stored in a separate public-derived GCS
    bucket; no checkpoints, strategies, tree caches, or solver logs cross that
    boundary. A manifest validator checks JSON shape and score/TC/dealer
    bindings for charts and binary headers before publication. The Study route
    can be direct-URL-only and `noindex`; this is discoverability control, not
    authentication, so every object in that bucket is intentionally public.

21. **Actual-policy support is small; static best-response closure is not
    (2026-07-16).** The policy-aware DFS decision run used the already-solved
    10x10/TC0/dealer-0 average policy, all 140,118 deals, and no re-solve. Raw
    is 352.30M nodes / 39.51M info sets. Exact positive profile support reaches
    236.92M / 33.97M, but player 1's response closure and therefore the union
    are exactly raw. At threshold `1e-6`, the profile falls to 210.34M / 30.62M
    while the union remains 341.15M / 39.10M. At the aggressive `1e-4`, the
    profile is only 74.37M / 12.05M, yet the honest union is still 245.06M /
    31.27M: only a 1.44x node and 1.26x info-set reduction.

    That kills the first restricted-arena design cheaply. Profile-only counts
    are useful for allocating training effort, not for claiming adversarial
    safety. The all-actions closure is also deliberately pessimistic for a
    double oracle: it retains every action the responder may evaluate, whereas
    a deterministic best response chooses one action per information set. The
    next worthwhile structural instrument is therefore a space-for-time oracle
    that counts the profile plus the two CHOSEN BR policies, then iterates
    restricted solving and oracle additions until exact global BR finds no
    profitable deviation. The decision run cost roughly $0.15 compute, took
    54m24s, and peaked at 4.42 GiB RSS—well below its $2 feature cap.
    The existing exact BR result now exposes the chosen action index it already
    resolved for every responder infoset; tests enforce that opponent rows
    remain unresolved and every responder choice is legal. The next counter is
    also implemented: after exact BR resolution on a full prebuilt tree, a
    three-bit traversal counts the union of profile, chosen BR0, and chosen BR1
    paths without double-counting. A tiny local test establishes the expected
    bounds against profile and the pessimistic all-action union.

    The production result clears the next gate, but only with the user's
    off-equilibrium allowance. Exact positive support retained 33.98M info sets
    (1.16x shrink); threshold `1e-4` retained 12.06M info sets / 74.42M nodes
    (3.28x / 4.73x shrink). Exact chosen responses added only ~8k rows beyond
    profile at that threshold. This says the current solution already contains
    nearly every selected exploit path; most size is tiny policy support.

    One more distinction matters before a builder: a restricted game allows
    both players to cross-combine retained profile/BR actions, whereas this
    three-bit result counts only three fixed paths. A new local action-set
    closure counter is implemented and is the real arena-size gate. It does NOT
    yet say the reduced game is solved: after re-solving, another BR can choose
    different actions. A credible double oracle must repeat full exact-BR
    audits and grow the arena until global gain <=0.01. The first fixed-path
    union is an optimistic lower bound, not a fleet multiplier.

    The production action-closure rerun is reassuring: at `1e-4` the actual
    cross-combined arena is 12.149M info sets / 74.994M nodes, only 0.76% above
    the fixed paths and 3.25x / 4.70x below raw. Cross-combination does not eat
    the gain, so a restricted builder is justified—but the size factor alone
    is not a budget multiplier.

    The builder now exists with the missing safety loop. On a 300-deal 8x8
    stress test, restricted round zero claimed internal epsilon 0.00515 while
    the full-tree audit found 0.01528; exact BR therefore did precisely the job
    required of it. Monotone additions certified round two at 0.00671, with a
    9.89x-smaller final info table and a fully charged 1.95x wall speedup.

    But using a solved TARGET policy to choose the initial arena is circular.
    The real 7x7→8x8 neighbor test is the sobering result: its less-converged
    source left only a 1.57x info shrink, full and restricted warm solves both
    stopped at iteration 10, and the extra closure/audit made restricted 19%
    slower. Therefore retract the provisional $1.2K-$5.8K multiplication.
    The production warm scout independently measured only 1.61x end-to-end
    (10x10→10x9: 90→40 iterations, 45m13s→28m03s), putting the evidence-based
    relaxed+warm estimate near $31K.

    The <=$2 sparse-source composition scout has now closed that question.
    Starting from the solved 10x10 source, restricted round zero retained
    12.44M info sets and solved internally to 0.00808, but exact full BR found
    0.01056 and added 6.24M actions. Round one still missed narrowly at
    0.010126. Round two certified 0.009965 with 14.95M info sets (2.64x fewer
    than full), yet the fully charged command took 41m54.5s versus 28m02.9s
    full warm and used more peak memory. Three restarted solves and 537s of
    exact audits dominated the arena saving. This is a clean negative result:
    the oracle makes the approximation safe, but not cheap enough in this
    composition. Keep the benchmark/certificate machinery; reject this strict
    restricted architecture and retain the ~$31K evidence-based estimate.

    If accuracy is relaxed once more, there is a narrowly bounded lead:
    target 0.011 would have accepted round zero's exact 0.010560 at roughly
    20m47s charged time, provisionally 1.35x faster. That changes the allowed
    average unilateral equity from at most 0.50pp to 0.55pp and would only move
    the rough budget toward $23K. It wrote no production result and comes from
    one shallow-band spot, so it is a candidate gate rather than fleet credit.

22. **Representation changes need empirical gates even when the layout story
    sounds obvious (2026-07-16).** SyncCFR+'s alternating sweeps now overlay
    player-local pending-regret slots, and solve-time action metadata drops the
    unused `Vec` capacity word by using boxed slices. Both are lossless: the
    829k-info-set warm A/B produced byte-identical checkpoint, strategy, and
    game-value artifacts and exactly the same exploitability trajectory. The
    boxed header saves 8 bytes per info set (~316 MiB at the measured 39.51M
    10x10 rows), but small-run peak RSS moved only 1.403 -> 1.399 GB and wall
    did not improve, so this receives no projected fleet multiplier.

    The rejected alternative is more instructive: an inline
    `SmallVec<[Action;8]>` eliminated millions of tiny allocations but inflated
    every row to maximum inline capacity, increasing peak RSS to 1.483 GB and
    wall to 5.64s. Preserve the exact cleanup, reject the inline layout, and
    test any `f32` accumulator proposal separately against exact BR/value and
    resume round trips.

    The attachment's proposed generic hand-strength dominance pruning fails
    the same proof standard for a different reason: Truco uses weak-hand raises
    as bluffs, so low equity under one policy is not dominance against every
    opponent strategy. Only a universally certified forced-result rule (for
    example, accepting a raise cannot avoid a worse loss than folding in every
    hidden deal and continuation consistent with the information set) is a
    credible candidate. No broad hand-strength rule currently has that
    certificate, so changing the game tree would be unjustified.

23. **Mini-batching is neutral and neighboring-policy MCCFR still explodes
    (2026-07-16).** The deep-band fallback now has the missing experimental
    machinery: persistent sparse state, frozen-strategy mini-batches, lazy
    pseudo-regret seeding from a saved average policy, and a fixed mid-stride
    evaluation panel. Local batch-1 and batch-32 runs were nearly identical in
    speed and exploitability, so batching itself is not a convergence or
    throughput lever. A seemingly encouraging restricted 1M-sample result
    (epsilon 0.034) was in-sample; disjoint evaluation exposed it as optimism.

    The cost-capped production-shaped scout projected the complete solved
    10x10 policy into 0x0's full ladder and trained over all 140,118 deals.
    Held-out-panel epsilon was 0.226 at 250k samples and worsened monotonically
    to 0.241 at 1M, while sparse state grew from 6.96M to 20.74M info sets and
    peaked at 14.3 GiB RSS. The shallow seed initializes shared histories but
    cannot cover new deeper-raise histories, and stale average mass is
    intentionally reset; the untrained expansion overwhelms the benefit.
    Retain the implementation for research, but reject it as the current
    deep-band cost solution. The completed attempt ran 4m19s; including a
    setup-only metadata-path failure, cloud spend was only a few cents.

24. **A refined turn-up class transfers surprisingly well to another blocker
    class (2026-07-16).** TC0 was the only class that received the full
    tremble/detremble repair, while the other retained 11×11 solutions are
    globally excellent but contain the familiar low-reach garbage. The first
    all-deal canary explicitly rekeyed TC0's final dealer-0 checkpoint into TC1,
    preserved its trained average, and ran one TC1 iteration plus exact global
    certification. Against native TC1, the mapped profile reduced the
    reach-weighted mean per-infoset BR gap from 0.729pp to 0.00937pp (77.8×)
    and the reach weight above 5pp from 4.469% to 0.0000036%. This is direct
    evidence that abstract card/action meanings carry most of the policy and
    the blocked-rank probability perturbation is small enough for reuse.

    The transfer is not mathematically free: raw/purified global epsilon moved
    from 0.00000933/0.00000668 to 0.0027315/0.0004361. In match-equity terms
    the purified value is still only 0.0218pp, comparable to the accepted TC0
    Study release, after 36.7 seconds for the one target iteration and exact BR.
    Therefore the practical route to all turn-ups is now: map a refined donor,
    certify every target independently, and add a short plain-CFR tail only
    where its measured certificate requires it. Do not pay for nine fresh
    trembling campaigns, and do not infer success for the remaining seven
    classes from this single TC1 canary.

    The canary also caught a long-lived-artifact mismatch unrelated to the
    transfer hypothesis. The saved TC1 strategy predated a proof-backed action
    pruning change and held roughly twice as many probability slots as its
    current treepack. Teacher export silently trusted positional equality and
    produced inconsistent tensors. Projection is now by action identity with
    renormalization, serialization validates tensor and exact wire lengths,
    and the measured baseline exports cleanly. Artifact schemas should
    eventually bind the legal-action version explicitly; the compatibility
    projection is the safe bridge for existing retained data.

25. **The safe dominance pass found real rules, but raw-tree savings are not
    automatically solver savings (2026-07-16).** The key distinction came from
    the proposed examples themselves. "Always raise with the stronger final
    card" is unsafe because revealing strength changes the range that can raise.
    Blanket "never hide during mao de onze" is also unsafe: a round-2 leader
    still has two possible remaining cards, and concealment can affect which
    response the opponent chooses. But the second mover in round 2 cannot have
    won round 1; if its hide loses, the hand ends, while showing the same card
    can only tie/win. Round-3 responding hides are terminally dominated too.
    A round-3 lead hide becomes removable only when the responder cannot raise
    and has one forced card.

    The user's 9x9/weak-final-card fold example also admits a clean certificate.
    If the raise caller lost round 1 and led the final round hidden or with the
    globally weakest card, accepting must lose against the raiser's final
    face-up card; ties belong to the round-1 winner. Fold loses only the old
    stake. At lower scores a re-raise is deliberately retained because bluff/
    fold equity is real; at 9x9 the existing match-deciding-stake rule removes
    it, leaving only Fold. This is proof from the information set, not a rule
    learned from one equilibrium profile.

    The code and regression tests implement exactly those boundaries. On 300
    deals, nodes fell 2.322M -> 1.253M, info sets 829k -> 629k, and a matched
    100-iteration warm solve took 6.1s instead of 8.6s while value moved only
    `2.3e-6` and both profiles certified near `1.8e-4`. Across all deals the
    node reduction remains large—2.26x at mao, 1.96x at `{1,3}`, 1.84x at
    `{1,3,6}`, 1.80x at `{1,3,6,9}`, and 1.76x in the full
    ladder—but overlapping histories mean information sets shrink only about
    7-8%.

    The paid warm scout prevents over-selling even that result. Actual
    10x10->10x9 TC0/d0 still stopped at 40 iterations with a slightly better
    exact epsilon (0.009622 versus 0.009825), but end-to-end wall worsened
    28m03s -> 30m49s and VmHWM 55.7 -> 61.5 GiB. Most removed histories were
    already free during CFR iterations because exact-zero opponent pruning
    skipped them. Peak RAM instead came from materializing the target table and
    deserializing the source table at once; dropping `src` later lowers steady
    RSS but not the worker size. Keep the lossless rules and assign no cost
    multiplier to dominance itself.

    The immediate <=$2 representation gate then removed the empty target table:
    disk warm starts now allocate dense accumulators directly and copy matching
    source action slots into them. Local same-band, cross-turn-up, and mao-remap
    fixtures match the old path exactly. The all-deal production rerun also
    matched every recorded epsilon checkpoint and the final value exactly, cut
    wall 30m49s -> 29m03s, and cut VmHWM 61.47 -> 43.58 GiB. Right-sizing from
    75 to 55 GiB makes that identical shallow job about 1.46x cheaper. This is a
    useful lossless win, not the answer to the budget: the source checkpoint is
    still fully deserialized, deep tiers are unmeasured, and even a deliberately
    uniform extrapolation would move the $31K bracket only toward ~$23K. A
    row-streamable checkpoint plus a deep-band memory scout is the next loader
    gate.

    This also sharpened the answer about global error budgeting. Backward
    induction through the score DAG is cheap, but today's local exact-BR pass
    materializes every hand tree. Measured 10x10 build plus two BRs is 461.8s,
    about 2.0% of a 900-iteration tight solve; applied to the raw $505K lattice,
    an exact whole-grid certificate is roughly $10K with the current evaluator.
    It is not "gazillions," but it is not a routine sub-$1K allocator either.
    The next program therefore has two lanes: a <=$2 sampled reach/error map to
    prioritize work (explicitly not a certificate), and a compact DFS/depth-wise
    BR evaluator whose tiny/full controls must exactly match the current oracle.
    Until one of those structural gates passes, sunk compute remains only tens
    of dollars but the evidence-backed strict projected full-grid spend remains
    about $31K spot. Do not launch the tabular grid.

26. **Cross-score policy transfer needs per-target selection, not faith
    (2026-07-16).** Extending the successful cross-turn-up mapping to scores in
    the same mao-de-onze tree band produced a deliberately adversarial canary
    set: 11x10 TC0 donor to 11x0 and 11x9, both dealers. The far-score result was
    unexpectedly excellent for both dealers—raw epsilon 0.000195/0.000266 and
    weighted local BR gap 0.0110/0.0122pp. Near-score dealer 1 was also usable at
    raw/purified epsilon 0.00604/0.00319 and local gap 0.0943pp. Near-score
    dealer 0 was not: raw/purified epsilon stayed 0.01951/0.01615 after 22 target
    iterations, even though its local gap improved from 12.29pp to 0.139pp.

    The failure is informative rather than a reason to discard transfer. Score
    is absent from info-set identity, so card/action policy maps perfectly, but
    the preserved average was trained against different terminal match values.
    A few new regrets cannot move thousands of iterations of average inertia.
    The deployment policy is therefore a portfolio: cheaply probe the refined
    donor, certify it exactly, retain it only inside explicit global and local
    limits, and fall back to the already-converged native target otherwise. A
    fallback still gets a full BR-gap table; Study honestly exposes its weak
    off-equilibrium branches instead of pretending every target was refined.
    This directly implements the user's stated tolerance for naturally bad
    off-equilibrium play while protecting ordinary on-path play.

    Operationally, the campaign is sharded but resumable. A worker uploads all
    three candidate artifacts only after chart binding, exact BR, deep export,
    gzip, and checksums succeed. It checks a seven-hour soft deadline only
    between profiles, while Compute Engine deletes the VM at eight hours even
    if guest cleanup fails. The initial four canaries and one setup-only retry
    cost roughly ten cents; all their VMs and boot disks were deleted.

27. **The next cheap gates separate prioritization, certification, and solver
    memory instead of pretending one metric answers all three (2026-07-16).**
    The observable-card extension to the final forced-fold proof closed with a
    useful negative result. Exact 300-deal counts did not move at 0x0 or 9x9.
    `Plain(1)` can become the weakest unseen remainder only by exhausting all
    `Plain(0)` copies, but that observation pattern is incompatible with the
    caller first losing and then winning the first two rounds; stronger ranks
    require more visible lower cards than exist. The implemented globally
    weakest-card case is the whole reachable family, not merely its first
    member.

    The sampled allocator is intentionally a different object from an exact
    whole-match best response. It computes fixed-policy hand-outcome kernels on
    deterministic strided panels, solves the complete score DAG for profile
    continuation values, then weights representative-band one-hand deviation
    gains by profile reach. Because a match visits several hands, the summed
    error mass can exceed one and must not be converted to match-equity pp. On
    the first 96-deal/three-panel TC0 donor, `all-except-raise` projection put
    32.9% of priority in the full ladder and 25.4% in `{1,3,6,9}`. Uniform
    missing-action sensitivity also left a majority in those deepest bands.
    This robustly says where to spend the next tiny benchmark—learn newly legal
    raise behavior in a deep band—but not what the final global epsilon is.

    Compact depth-wise DFS separately addresses certification memory. It
    retains average policy, one depth's counterfactual action aggregates and
    resolved responder choices, recomputing histories instead of allocating a
    solver arena. Tiny and 300-deal controls match the materialized oracle; the
    actual-checkpoint 300-deal differences are at most `1.85e-10`. The local
    price is CPU: 2.791s compact versus 1.509s materialized. (Its all-deal
    production gate later PASSED — 5.95 GiB / 1,414s at 10×10, 8.13 GiB at 0×0;
    see item 28 and the 2026-07-17 benchmark entry — and exposed the
    checkpoint-portability hazard now guarded by --legacy-tree/--project-dominated.)

    Finally, source checkpoint streaming is a real but bounded lossless win.
    The positioned format was already row-delimited; a validating row reader
    now projects directly into dense accumulators and all three transfer modes
    match the compatibility path exactly. Production 10x10->10x9 again reached
    epsilon `0.009622` at iteration 40 and value `0.198171`. During solving,
    RSS stayed near 16 GiB; the actual 31.60-GiB peak occurred afterward when
    dense accumulators overlapped reconstruction of the output strategy table.
    Wall rose 29m03s -> 30m49s, but 40-GiB right-sizing still makes the complete
    shallow job about 24% cheaper than the 55-GiB direct loader. This identifies
    the next representation gate precisely: serialize checkpoint/strategy rows
    from dense storage without rebuilding the hash table. As before, no shallow
    memory multiplier is credited to unmeasured deep tiers.

    That gate closed the next day (phase 9c): the checkpoint writer had been
    materializing an owned copy of every row plus the whole multi-GiB file in
    RAM, and the returned StrategyTable rebuild was pure overlap. Streaming
    both writers from the dense accumulators and skipping the table rebuild
    reproduced the production epsilon trajectory at every checkpoint while
    peak RSS fell 31.60 -> 16.19 GiB — the solve-phase plateau is finally the
    true peak, and the shallow warm job's memory class chain reads 75 -> 55
    -> 40 -> 24 GiB across three lossless representation phases. The lesson
    repeats itself: measured VmHWM, not steady-state RSS, is the billing
    reality, and every one of these peaks was an artifact of representation,
    not of the algorithm.

28. **Tree-changing prunes silently orphan the existing checkpoint library,
    and a "healthy" scout number was the symptom (2026-07-17).** The compact
    exact-BR all-deal production gate passed exactly as designed — 5.95 GiB
    peak and 1,414s single-threaded for whole-game BR over all 140,118
    10x10/TC0/d0 deals, removing arena-class RAM from certification — but it
    printed `epsilon=0.016309` for a checkpoint certified at 0.000248, with
    zero missing-policy decisions and no other warning. The evaluator was
    innocent: on matched trees it agrees with the arena oracle to 5e-11. The
    2026-07-16 proof-scoped prunes had changed the game tree, and the
    2026-07-09 policy was silently renormalized onto it.

    The instructive failure came from trying to fix that with a smarter
    projection. Moving each pruned hidden play's mass onto the same card's
    face-up play (the action the dominance proof names) helped only 0.216 ->
    0.196 on a 300-deal control, and re-solving the source 4x tighter changed
    nothing: old equilibria genuinely keep ~36% average mass on hidden plays.
    Two lessons. First, weak dominance guarantees a good continuation exists
    after substituting the dominating action — it does not make a fixed old
    policy's continuation good, so no local row operation ports a
    pre-dominance checkpoint. Second, equilibria freely mix onto
    payoff-tied actions, so "the branch is dominated" never implies "the
    saved policy barely uses it." Practical policy: prunes are free for
    fresh/warm solves (warm starts heal projection in the measured 40
    iterations), but any certification or export of an older artifact must
    happen on its own tree or after a warm re-solve, and projection is now
    explicit and loudly reported (`--project-dominated`,
    `COMPACT_BR_PROJECTION` mass diagnostics, and a clean arena-control skip
    instead of a mid-pass panic).

29. **Neighboring equilibria are ~94% row-identical, and the differences sit
    exactly where transfer hurts (2026-07-17).** A $0 descriptive pass
    (`solve compare-policies`) joined converged native strategies on their
    shared key space. 11x10 vs 11x9 (tc0/d0): median row TV distance is
    exactly zero, 93.7% of 5.6M rows are within 1e-3, argmax agrees 96%,
    and where both policies are near-pure they agree 99.99%. The same holds
    for 11x10 tc0 vs tc1. The pairs differ in WHERE they disagree:
    cross-score divergence concentrates at the root (depth-0 mean TV 0.140 —
    the mão accept/fold responds to the opponent's score) while cross-TC
    divergence doesn't (depth-0 TV 0.003). That explains the canary paradox:
    a transfer can agree on 96% of rows and still certify at 0.0195, because
    exploitability is carried by a small, shallow, score-sensitive core that
    CFR average inertia repairs slowly. For distillation, the encouraging
    half is the 99.99%: the crisp part of optimal play is stable across
    neighboring states, so human-facing rules can likely be taught once per
    band rather than once per score.

    The reach-weighted follow-up sharpened this considerably. Weighting each
    row by its visit probability under the policy's own play (computed on
    the legacy tree these artifacts were solved for), the same cross-score
    pair moves from mean TV 0.017 to **0.129** and argmax agreement drops
    from 96% to 86%; only half the play mass sits on near-identical rows.
    The divergent score-sensitive core is tiny in row count but carries
    about half of actual play. Both views matter: the unweighted map says
    transfer gets ~94% of the table for free, the weighted map says the
    remaining rows are exactly where the match is played — so short warm
    tails must fix a concentrated, shallow, high-reach set, and
    certification (not row counting) remains the only honest accept gate.

30. **The 225-profile release fleet validated profile transfer at scale:
    85% of spots shipped a certified transfer (2026-07-17).** The full
    Study lattice (11x0-11x11 x 2 dealers x TC0-8, plus 10x10 x TC0-8)
    was built by four Spot workers in 3.0-4.6h wall each for well under
    the $5 budget, surviving one preemption via marker-idempotent
    replacement. Per the 225 self-certification markers: 192 spots (85%)
    accepted the 90-second cross-score/turn-up transfer (worst raw eps
    0.00627 against the 0.01 gate), 5 exported existing refined
    checkpoints, 28 (12%) fell back to the converged native profile. The
    fallback set confirms item 29's mechanism at scale: those native
    profiles pair near-zero global raw eps (median 1e-5) with large
    reach-weighted per-decision BR gaps (median 2.55pp, max 12.6pp),
    while every accepted transfer stayed under 0.098pp — the transferred
    donors are systematically better teachers off the certified path,
    and the spots where transfer fails its global gate are the same
    score-sensitive cores where natives teach worst. Published as
    immutable release `20260717-full-225-v1`. Cleanup lesson: the 8h
    `maxRunDuration` + `instanceTerminationAction=DELETE` backstop
    deleted every VM and disk automatically after guest poweroff — the
    hard budget backstop doubles as zero-touch cleanup.

31. **Asymmetric raise pruning — the user's shower intuition was a real
    structural win (2026-07-17).** The observation: at 9×6, the player at 9
    already reaches 12 by winning at truco (stake 3), so *only* the 6-player
    meaningfully raises to retruco. The deployed prune misses this because it
    gates on `min(score)` — a single global test — so at 9×6 (min 6) it keeps
    both players' retruco branches, identical to 6×6. The fix is to gate on the
    *acting* player's own score: the same dominance argument the deployed rule
    already uses (an opponent that folds a match-deciding raise loses the match,
    so never folds; the raise is pure downside) applies per acting player, not
    just to the lower-scored one.

    Measured, it is both valid and large. Symmetric cells are exactly unchanged;
    lopsided cells collapse (9×0 → 0.115× info sets, a solve that ran 12.6×
    faster on 9× less RAM). Game value is preserved to within convergence noise
    at every tested cell including the most-aggressive 9×0 (Δ 6.1e-6) and the
    mão-de-onze edge 11×6 (identical). A full 121-cell sweep puts the
    whole-grid cost at 0.38–0.53× — the full-ladder tier that holds 89% of the
    cost drops to 0.36–0.51× — moving the evidence-based bracket from ~$31K
    toward ~$12–16K.

    The deeper lesson for the program: the memory-representation work of the
    prior week was lossless and elegant but could not touch the headline number
    because the deep bands stayed structurally huge. The first thing that
    actually dents them came not from engineering the solver but from noticing
    a game-theoretic redundancy in the *tree* — a strategy-independent dominance
    that shrinks exactly the lopsided deep cells where the money is. It does not
    solve the residual: the handful of symmetric-deep cells (0×0, 1×1, 2×2) do
    not shrink and still need the giant worker. But it confirms that the
    remaining gap to <$1K, if it closes at all, closes through structure —
    dominance, abstraction, a cheaper deep-band algorithm — not through more
    representation tuning.

32. **Decision to pivot from exact solving to a neural policy (2026-07-17).**
    After asymmetric raise pruning brought the whole-grid exact bracket to
    ~$12–16K — still >10× the <$1K goal, and walled by a few deep symmetric
    cells that no lossless method shrinks — the owner elected to spend the
    remaining budget on a neural approach rather than keep buying exact
    structure at diminishing returns. The exact line is documented for cold
    resume in `EXACT_SOLVING.md` (consolidated status, ledger of every
    optimization, tooling, and the residual open problems) and paused, not
    abandoned.

    The neural plan (plan 83) is NOT the old "solve fully then distill" (§14):
    we never had, and will not buy, a full tabular solution. The exact corpus
    is 10×10 (9 TC, d0+symmetry, ε≈2.5e-4) plus the entire mão-de-onze row
    11×0…11×11 (both dealers, all TC) — the two cheapest tiers. The plan is
    supervised learning on that corpus first (a clean target and a practical
    upper bound on the easy path; the measured purity/similarity structure says
    a net conditioned on score/tc/dealer should fit it well and share heavily
    across turn-ups), then deep RL warm-started from the SL net to reach the
    unsolved deep bands. The decisive metric stays the exact one: `compact-br`
    exploitability of the net on the tree it plays. The crux to measure early is
    generalization — an SL net trained only on {1,3}+mão has never seen a live
    deep-ladder raise, so how far it degrades one band deeper decides whether RL
    is required or merely helpful.

33. **CFR-D safe subgame re-solving: built, then validated at production
    scale for $3.45 (2026-07-18, plan 84 Phase 3).** After the lossless-prune
    audit (item 31's family closed empty on the symmetric deep cells), the
    owner chose decomposition over abstraction. One long session built the
    whole thing: round-2 boundary decomposition by REPLAYING the tree build's
    recursion (ids/views/boundary from the same code path that built the
    arenas — no reverse engineering), the Burch–Johanson–Bowling
    Terminate/Follow gadget as a code-level side accumulator over the
    existing packed subtrees, batched CBV extraction through the certified
    exact-BR pass, composition, and certification. The 10×10 ground-truth
    run (all 140k deals, the real fleet artifact): a subgame corrupted to
    uniform (ε 0.000248→0.00878) repaired from the boundary summary ALONE to
    ε 0.000248437 — +1.6e-7. The decomposition number that matters ahead:
    the largest of 495 re-solve units is 0.52% of the cell, so deep-cell
    re-solves are commodity-box-sized.

    Two lessons the production runs taught (both caught by backstops, not
    budget): CBV extraction must be one BR pass for ALL subgames, not one
    per subgame (the pass doesn't depend on which nodes you read out — the
    7h-backstop kill that found this cost $2.20, the single biggest line in
    the $3.45 total); and artifacts must be evaluated on the tree generation
    they were solved for (`--legacy-tree`) — a positionally-mangled action
    vector certifies at 0.09 instead of 0.00025, and the certificate itself
    is what caught it. Remaining for the deep-band payoff: the trunk-CFR
    loop producing CFVs without a full solve (Phase 4), then the composed
    deep-cell benchmark (Phase 5).

34. **From-scratch CFR-D: the average-coupling plateau, and decomposed
    parity with monolithic (2026-07-21, plan 84 Phase 4).** The trunk loop
    (no blueprint anywhere: trunk sweeps over boundary-as-terminal cached
    values, warm gadget-free subgame re-solves, tail-averaged CBVs, gadget
    recovery) hit a wall only production could show: composed eps ~0.018
    at r90, r150 AND r300 — iterations were not the constraint. A single
    instrumented run carrying THREE certificates (raw / tail-CBV-recovered
    / BR-CBV-recovered) split the blame cleanly: the loop itself was weak
    (raw 0.0465), recovery was repairing it (0.0176), and BR-based CBVs
    were correctly worse (0.0334 — BR against a weak blueprint is
    over-generous, vindicating the tail-CBV design). Root cause: every
    intra-round coupling read the AVERAGE profile, which lags the current
    regret-matching iterate by ~half the averaging window — SyncCFR+
    backups are defined against the frozen current strategy, so the
    trunk↔subgame alternation chased a stale target and stalled at the lag
    scale. Routing the couplings through the current iterate collapsed the
    toy trunk-vs-monolithic gap to BETTER than parity (0.000239 vs
    0.000354) and the production run to raw eps 0.001122 with game value
    5.5e-5 from the teacher ground truth, ~1.8× monolithic wall net.
    Lesson for the paper: in decomposed CFR, "which profile do the
    couplings see" is a first-class correctness knob, invisible at toy
    scale (a 3× hint, not a wall) and decisive at production scale. Phases
    3+4 together: 8 spot runs, $6.70, every negative result caught by a
    backstop or a certificate rather than a budget.

---

## Technical choices worth documenting

- **Rust** for the solver: zero-cost abstractions, no GC pauses during CFR
  iteration, easy interop with the existing game engine crate.
- **Pre-built arena trees** rather than dynamic traversal: 30× per-iteration
  speedup by eliminating engine overhead and heap allocations.
- **Persist full info-set metadata in saved strategies** so solution artifacts
  remain inspectable after a solve rather than collapsing into anonymous action
  vectors.
- **`ahash`** for info set hashing: significantly faster than standard
  `HashMap` for short fixed-size keys.
- **`smallvec`** for transient action-probability vectors in traversal; fixed
  solve-time legal-action metadata instead uses boxed slices, because an
  inline eight-action row was measured to increase rather than reduce RSS.
- **GCP c2-standard-8** for benchmarking: compute-optimized (higher per-core
  clock) vs the memory-optimized n2-highmem we started with. At 19% memory
  usage, RAM was never the constraint.

---

## What a paper might look like

**Title:** *Solving Truco: Nash Equilibrium Computation for a Brazilian Card
Game with LLM-Assisted Algorithm Research*

**Contributions:**
1. First published solution of Truco using CFR, with full strategy tables
   available for inspection.
2. Systematic comparison of CFR variants (CFR+, DCFR) at this game scale.
3. A methodology for LLM-driven autonomous CFR algorithm improvement,
   applied to a real game-solving problem.
4. A neural network distillation of the full solution into a phone-deployable
   model (future work).

**Venue:** AAAI, IJCAI, NeurIPS Game Theory workshop, or similar.
